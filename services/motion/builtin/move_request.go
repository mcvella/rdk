package builtin

import (
	"bytes"
	"context"
	"fmt"
	"math"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/golang/geo/r3"
	"github.com/pkg/errors"
	goutils "go.viam.com/utils"

	"go.viam.com/rdk/components/base"
	"go.viam.com/rdk/components/base/kinematicbase"
	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/motionplan"
	"go.viam.com/rdk/pointcloud"
	"go.viam.com/rdk/referenceframe"
	"go.viam.com/rdk/resource"
	"go.viam.com/rdk/robot/framesystem"
	"go.viam.com/rdk/services/motion"
	"go.viam.com/rdk/services/motion/builtin/state"
	"go.viam.com/rdk/services/slam"
	"go.viam.com/rdk/services/vision"
	"go.viam.com/rdk/spatialmath"
)

const (
	defaultReplanCostFactor = 1.0
	defaultMaxReplans       = -1 // Values below zero will replan infinitely
	baseStopTimeout         = time.Second * 5
)

var errGoalWithinPlanDeviation = errors.New("no need to move, already within planDeviationMM")

// validatedMotionConfiguration is a copy of the motion.MotionConfiguration type
// which has been validated to conform to the expectations of the builtin
// motion servicl.
type validatedMotionConfiguration struct {
	obstacleDetectors     []motion.ObstacleDetectorName
	positionPollingFreqHz float64
	obstaclePollingFreqHz float64
	planDeviationMM       float64
	linearMPerSec         float64
	angularDegsPerSec     float64
}

type requestType uint8

const (
	requestTypeUnspecified requestType = iota
	requestTypeMoveOnGlobe
	requestTypeMoveOnMap
)

// moveRequest is a structure that contains all the information necessary for to make a move call.
type moveRequest struct {
	requestType requestType
	// geoPoseOrigin is only set if requestType == requestTypeMoveOnGlobe
	geoPoseOrigin spatialmath.GeoPose
	// poseOrigin is only set if requestType == requestTypeMoveOnMap
	poseOrigin        spatialmath.Pose
	logger            logging.Logger
	config            *validatedMotionConfiguration
	planRequest       *motionplan.PlanRequest
	seedPlan          motionplan.Plan
	kinematicBase     kinematicbase.KinematicBase
	obstacleDetectors map[vision.Service][]resource.Name
	replanCostFactor  float64
	fsService         framesystem.Service

	executeBackgroundWorkers *sync.WaitGroup
	responseChan             chan moveResponse
	// replanners for the move request
	// if we ever have to add additional instances we should figure out how to make this more scalable
	position, obstacle *replanner
	// waypointIndex tracks the waypoint we are currently executing on
	waypointIndex *atomic.Int32
}

// plan creates a plan using the currentInputs of the robot and the moveRequest's planRequest.
func (mr *moveRequest) Plan(ctx context.Context) (state.PlanResponse, error) {
	inputs, err := mr.kinematicBase.CurrentInputs(ctx)
	if err != nil {
		return state.PlanResponse{}, err
	}
	// TODO: this is really hacky and we should figure out a better place to store this information
	if len(mr.kinematicBase.Kinematics().DoF()) == 2 {
		inputs = inputs[:2]
	}
	mr.planRequest.StartConfiguration = map[string][]referenceframe.Input{mr.kinematicBase.Kinematics().Name(): inputs}

	// TODO(RSDK-5634): this should pass in mr.seedplan and the appropriate replanCostFactor once this bug is found and fixed.
	plan, err := motionplan.Replan(ctx, mr.planRequest, nil, 0)
	if err != nil {
		return state.PlanResponse{}, err
	}

	waypoints, err := plan.GetFrameSteps(mr.kinematicBase.Kinematics().Name())
	if err != nil {
		return state.PlanResponse{}, err
	}

	switch mr.requestType {
	case requestTypeMoveOnMap:
		planSteps, err := motionplan.PlanToPlanSteps(plan, mr.kinematicBase.Name(), *mr.planRequest, mr.poseOrigin)
		if err != nil {
			return state.PlanResponse{}, err
		}

		return state.PlanResponse{
			Waypoints:        waypoints,
			Motionplan:       plan,
			PosesByComponent: planSteps,
		}, nil
	case requestTypeMoveOnGlobe:
		// safe to use mr.poseOrigin since it is nil for requestTypeMoveOnGlobe
		planSteps, err := motionplan.PlanToPlanSteps(plan, mr.kinematicBase.Name(), *mr.planRequest, mr.poseOrigin)
		if err != nil {
			return state.PlanResponse{}, err
		}
		geoPoses := motionplan.PlanStepsToGeoPoses(planSteps, mr.kinematicBase.Name(), mr.geoPoseOrigin)

		// NOTE: Here we are smuggling GeoPoses into Poses by component
		planSteps, err = toGeoPosePlanSteps(planSteps, geoPoses)
		if err != nil {
			return state.PlanResponse{}, err
		}

		return state.PlanResponse{
			Waypoints:        waypoints,
			Motionplan:       plan,
			PosesByComponent: planSteps,
		}, nil
	case requestTypeUnspecified:
		fallthrough
	default:
		return state.PlanResponse{}, fmt.Errorf("invalid moveRequest.requestType: %d", mr.requestType)
	}
}

// execute attempts to follow a given Plan starting from the index percribed by waypointIndex.
// Note that waypointIndex is an atomic int that is incremented in this function after each waypoint has been successfully reached.
func (mr *moveRequest) execute(ctx context.Context, waypoints state.Waypoints, waypointIndex *atomic.Int32) (state.ExecuteResponse, error) {
	// Iterate through the list of waypoints and issue a command to move to each
	for i := int(waypointIndex.Load()); i < len(waypoints); i++ {
		select {
		case <-ctx.Done():
			mr.logger.CDebugf(ctx, "calling kinematicBase.Stop due to %s\n", ctx.Err())
			if stopErr := mr.stop(); stopErr != nil {
				return state.ExecuteResponse{}, errors.Wrap(ctx.Err(), stopErr.Error())
			}
			return state.ExecuteResponse{}, nil
		default:
			mr.planRequest.Logger.CInfo(ctx, waypoints[i])
			if err := mr.kinematicBase.GoToInputs(ctx, waypoints[i]); err != nil {
				// If there is an error on GoToInputs, stop the component if possible before returning the error
				mr.logger.CDebugf(ctx, "calling kinematicBase.Stop due to %s\n", err)
				if stopErr := mr.stop(); stopErr != nil {
					return state.ExecuteResponse{}, errors.Wrap(err, stopErr.Error())
				}
				return state.ExecuteResponse{}, err
			}
			if i < len(waypoints)-1 {
				waypointIndex.Add(1)
			}
		}
	}
	// the plan has been fully executed so check to see if where we are at is close enough to the goal.
	return mr.deviatedFromPlan(ctx, waypoints, len(waypoints)-1)
}

// deviatedFromPlan takes a list of waypoints and an index of a waypoint on that Plan and returns whether or not it is still
// following the plan as described by the PlanDeviation specified for the moveRequest.
func (mr *moveRequest) deviatedFromPlan(ctx context.Context, waypoints state.Waypoints, waypointIndex int) (state.ExecuteResponse, error) {
	errorState, err := mr.kinematicBase.ErrorState(ctx, waypoints, waypointIndex)
	if err != nil {
		return state.ExecuteResponse{}, err
	}
	if errorState.Point().Norm() > mr.config.planDeviationMM {
		msg := "error state exceeds planDeviationMM; planDeviationMM: %f, errorstate.Point().Norm(): %f, errorstate.Point(): %#v "
		reason := fmt.Sprintf(msg, mr.config.planDeviationMM, errorState.Point().Norm(), errorState.Point())
		return state.ExecuteResponse{Replan: true, ReplanReason: reason}, nil
	}
	return state.ExecuteResponse{}, nil
}

func (mr *moveRequest) obstaclesIntersectPlan(
	ctx context.Context,
	waypoints state.Waypoints,
	waypointIndex int,
) (state.ExecuteResponse, error) {
	var plan motionplan.Plan
	// We only care to check against waypoints we have not reached yet.
	for _, inputs := range waypoints[waypointIndex:] {
		input := make(map[string][]referenceframe.Input)
		input[mr.kinematicBase.Name().Name] = inputs
		plan = append(plan, input)
	}

	for visSrvc, cameraNames := range mr.obstacleDetectors {
		for _, camName := range cameraNames {
			mr.logger.Debugf(
				"proceeding to get detections from vision service: %s with camera: %s",
				visSrvc.Name().ShortName(),
				camName.ShortName(),
			)

			// get detections from vision service
			detections, err := visSrvc.GetObjectPointClouds(ctx, camName.Name, nil)
			if err != nil {
				return state.ExecuteResponse{}, err
			}

			// Note: detections are initially observed from the camera frame but must be transformed to be in
			// world frame. We cannot use the inputs of the base to transform the detections since they are relative

			// get the current position of the base which we will use to transform the detection into world coordinates
			currentPosition, err := mr.kinematicBase.CurrentPosition(ctx)
			if err != nil {
				return state.ExecuteResponse{}, err
			}

			// determine transform of camera to base
			cameraOrigin := referenceframe.NewPoseInFrame(camName.ShortName(), spatialmath.NewZeroPose())
			baseToCamera, err := mr.fsService.TransformPose(ctx, cameraOrigin, mr.kinematicBase.Name().ShortName(), nil)
			if err != nil {
				// here we make the assumption the base is coincident with the camera
				mr.logger.Debugf(
					"we assume the base named: %s is coincident with the camera named: %s due to err: %v",
					mr.kinematicBase.Name().ShortName(), camName.ShortName(), err.Error(),
				)
				baseToCamera = cameraOrigin
			}

			// Any obstacles specified by the worldstate of the moveRequest will also re-detected here.
			// There is no need to append the new detections to the existing worldstate.
			// We can safely build from scratch without excluding any valuable information.
			geoms := []spatialmath.Geometry{}
			for i, detection := range detections {
				// put the detection in the base coordinate frame
				geometry := detection.Geometry.Transform(baseToCamera.Pose())

				// put the detection into its position in the world with the base coordinate frame
				geometry = geometry.Transform(currentPosition.Pose())
				label := camName.Name + "_transientObstacle_" + strconv.Itoa(i)
				if geometry.Label() != "" {
					label += "_" + geometry.Label()
				}
				geometry.SetLabel(label)
				geoms = append(geoms, geometry)
			}
			gif := referenceframe.NewGeometriesInFrame(referenceframe.World, geoms)
			// want to have all geometry's be in the world coordinate frame
			tf, err := mr.planRequest.FrameSystem.Transform(mr.planRequest.StartConfiguration, gif, mr.planRequest.FrameSystem.World().Name())
			if err != nil {
				return state.ExecuteResponse{}, err
			}
			transformedGIF, ok := tf.((*referenceframe.GeometriesInFrame))
			if !ok {
				return state.ExecuteResponse{}, errors.New("cannot cast transformable as *referenceframe.GeometriesInFrame")
			}
			gifs := []*referenceframe.GeometriesInFrame{transformedGIF}
			worldState, err := referenceframe.NewWorldState(gifs, nil)
			if err != nil {
				return state.ExecuteResponse{}, err
			}

			currentInputs, err := mr.kinematicBase.CurrentInputs(ctx)
			if err != nil {
				return state.ExecuteResponse{}, err
			}

			// get the pose difference between where the robot is versus where it ought to be.
			errorState, err := mr.kinematicBase.ErrorState(ctx, waypoints, waypointIndex)
			if err != nil {
				return state.ExecuteResponse{}, err
			}

			if err := motionplan.CheckPlan(
				mr.kinematicBase.Kinematics(), // frame we wish to check for collisions
				plan,                          // remainder of plan we wish to check against
				worldState,                    // detected obstacles by this instance of camera + service
				mr.planRequest.FrameSystem,
				currentPosition.Pose(), // currentPosition of robot accounts for errorState
				currentInputs,
				errorState, // deviation of robot from plan
				lookAheadDistanceMM,
				mr.planRequest.Logger,
			); err != nil {
				mr.planRequest.Logger.CInfo(ctx, err.Error())
				return state.ExecuteResponse{Replan: true, ReplanReason: err.Error()}, nil
			}
		}
	}
	return state.ExecuteResponse{}, nil
}

func kbOptionsFromCfg(motionCfg *validatedMotionConfiguration, validatedExtra validatedExtra) kinematicbase.Options {
	kinematicsOptions := kinematicbase.NewKinematicBaseOptions()

	if motionCfg.linearMPerSec > 0 {
		kinematicsOptions.LinearVelocityMMPerSec = motionCfg.linearMPerSec * 1000
	}

	if motionCfg.angularDegsPerSec > 0 {
		kinematicsOptions.AngularVelocityDegsPerSec = motionCfg.angularDegsPerSec
	}

	if motionCfg.planDeviationMM > 0 {
		kinematicsOptions.PlanDeviationThresholdMM = motionCfg.planDeviationMM
	}

	if validatedExtra.motionProfile != "" {
		kinematicsOptions.PositionOnlyMode = validatedExtra.motionProfile == motionplan.PositionOnlyMotionProfile
	}

	kinematicsOptions.GoalRadiusMM = motionCfg.planDeviationMM
	kinematicsOptions.HeadingThresholdDegrees = 8
	return kinematicsOptions
}

func validateNotNan(f float64, name string) error {
	if math.IsNaN(f) {
		return errors.Errorf("%s may not be NaN", name)
	}
	return nil
}

func validateNotNeg(f float64, name string) error {
	if f < 0 {
		return errors.Errorf("%s may not be negative", name)
	}
	return nil
}

func validateNotNegNorNaN(f float64, name string) error {
	if err := validateNotNan(f, name); err != nil {
		return err
	}
	return validateNotNeg(f, name)
}

func newValidatedMotionCfg(motionCfg *motion.MotionConfiguration, reqType requestType) (*validatedMotionConfiguration, error) {
	empty := &validatedMotionConfiguration{}
	vmc := &validatedMotionConfiguration{
		angularDegsPerSec:     defaultAngularDegsPerSec,
		linearMPerSec:         defaultLinearMPerSec,
		obstaclePollingFreqHz: defaultObstaclePollingHz,
		positionPollingFreqHz: defaultPositionPollingHz,
		obstacleDetectors:     []motion.ObstacleDetectorName{},
	}

	switch reqType {
	case requestTypeMoveOnGlobe:
		vmc.planDeviationMM = defaultGlobePlanDeviationM * 1e3
	case requestTypeMoveOnMap:
		vmc.planDeviationMM = defaultSlamPlanDeviationM * 1e3
	case requestTypeUnspecified:
		fallthrough
	default:
		return empty, fmt.Errorf("invalid moveRequest.requestType: %d", reqType)
	}

	if motionCfg == nil {
		return vmc, nil
	}

	if err := validateNotNegNorNaN(motionCfg.LinearMPerSec, "LinearMPerSec"); err != nil {
		return empty, err
	}

	if err := validateNotNegNorNaN(motionCfg.AngularDegsPerSec, "AngularDegsPerSec"); err != nil {
		return empty, err
	}

	if err := validateNotNegNorNaN(motionCfg.PlanDeviationMM, "PlanDeviationMM"); err != nil {
		return empty, err
	}

	if err := validateNotNegNorNaN(motionCfg.ObstaclePollingFreqHz, "ObstaclePollingFreqHz"); err != nil {
		return empty, err
	}

	if err := validateNotNegNorNaN(motionCfg.PositionPollingFreqHz, "PositionPollingFreqHz"); err != nil {
		return empty, err
	}

	if motionCfg.LinearMPerSec != 0 {
		vmc.linearMPerSec = motionCfg.LinearMPerSec
	}

	if motionCfg.AngularDegsPerSec != 0 {
		vmc.angularDegsPerSec = motionCfg.AngularDegsPerSec
	}

	if motionCfg.PlanDeviationMM != 0 {
		vmc.planDeviationMM = motionCfg.PlanDeviationMM
	}

	if motionCfg.ObstaclePollingFreqHz != 0 {
		vmc.obstaclePollingFreqHz = motionCfg.ObstaclePollingFreqHz
	}

	if motionCfg.PositionPollingFreqHz != 0 {
		vmc.positionPollingFreqHz = motionCfg.PositionPollingFreqHz
	}

	if motionCfg.ObstacleDetectors != nil {
		vmc.obstacleDetectors = motionCfg.ObstacleDetectors
	}

	return vmc, nil
}

func (ms *builtIn) newMoveOnGlobeRequest(
	ctx context.Context,
	req motion.MoveOnGlobeReq,
	seedPlan motionplan.Plan,
	replanCount int,
) (state.PlannerExecutor, error) {
	valExtra, err := newValidatedExtra(req.Extra)
	if err != nil {
		return nil, err
	}

	if valExtra.maxReplans >= 0 {
		if replanCount > valExtra.maxReplans {
			return nil, fmt.Errorf("exceeded maximum number of replans: %d", valExtra.maxReplans)
		}
	}

	motionCfg, err := newValidatedMotionCfg(req.MotionCfg, requestTypeMoveOnGlobe)
	if err != nil {
		return nil, err
	}
	// ensure arguments are well behaved
	obstacles := req.Obstacles
	if obstacles == nil {
		obstacles = []*spatialmath.GeoObstacle{}
	}
	if req.Destination == nil {
		return nil, errors.New("destination cannot be nil")
	}

	if math.IsNaN(req.Destination.Lat()) || math.IsNaN(req.Destination.Lng()) {
		return nil, errors.New("destination may not contain NaN")
	}

	// build kinematic options
	kinematicsOptions := kbOptionsFromCfg(motionCfg, valExtra)

	// build the localizer from the movement sensor
	movementSensor, ok := ms.movementSensors[req.MovementSensorName]
	if !ok {
		return nil, resource.DependencyNotFoundError(req.MovementSensorName)
	}
	origin, _, err := movementSensor.Position(ctx, nil)
	if err != nil {
		return nil, err
	}

	heading, err := movementSensor.CompassHeading(ctx, nil)
	if err != nil {
		return nil, err
	}

	// add an offset between the movement sensor and the base if it is applicable
	baseOrigin := referenceframe.NewPoseInFrame(req.ComponentName.ShortName(), spatialmath.NewZeroPose())
	movementSensorToBase, err := ms.fsService.TransformPose(ctx, baseOrigin, movementSensor.Name().ShortName(), nil)
	if err != nil {
		// here we make the assumption the movement sensor is coincident with the base
		movementSensorToBase = baseOrigin
	}
	// Create a localizer from the movement sensor, and collapse reported orientations to 2d
	localizer := motion.TwoDLocalizer(motion.NewMovementSensorLocalizer(movementSensor, origin, movementSensorToBase.Pose()))

	// create a KinematicBase from the componentName
	baseComponent, ok := ms.components[req.ComponentName]
	if !ok {
		return nil, resource.NewNotFoundError(req.ComponentName)
	}
	b, ok := baseComponent.(base.Base)
	if !ok {
		return nil, fmt.Errorf("cannot move component of type %T because it is not a Base", baseComponent)
	}

	fs, err := ms.fsService.FrameSystem(ctx, nil)
	if err != nil {
		return nil, err
	}

	// Important: GeoPointToPose will create a pose such that incrementing latitude towards north increments +Y, and incrementing
	// longitude towards east increments +X. Heading is not taken into account. This pose must therefore be transformed based on the
	// orientation of the base such that it is a pose relative to the base's current location.
	goalPoseRaw := spatialmath.NewPoseFromPoint(spatialmath.GeoPointToPoint(req.Destination, origin))
	// construct limits
	straightlineDistance := goalPoseRaw.Point().Norm()
	if straightlineDistance > maxTravelDistanceMM {
		return nil, fmt.Errorf("cannot move more than %d kilometers", int(maxTravelDistanceMM*1e-6))
	}
	limits := []referenceframe.Limit{
		{Min: -straightlineDistance * 3, Max: straightlineDistance * 3},
		{Min: -straightlineDistance * 3, Max: straightlineDistance * 3},
		{Min: -2 * math.Pi, Max: 2 * math.Pi},
	} // Note: this is only for diff drive, not used for PTGs
	ms.logger.CDebugf(ctx, "base limits: %v", limits)

	kb, err := kinematicbase.WrapWithKinematics(ctx, b, ms.logger, localizer, limits, kinematicsOptions)
	if err != nil {
		return nil, err
	}

	geomsRaw := spatialmath.GeoObstaclesToGeometries(obstacles, origin)

	mr, err := ms.relativeMoveRequestFromAbsolute(
		ctx,
		motionCfg,
		ms.logger,
		kb,
		goalPoseRaw,
		fs,
		geomsRaw,
		valExtra,
		requestTypeMoveOnGlobe,
	)
	if err != nil {
		return nil, err
	}
	mr.seedPlan = seedPlan
	mr.replanCostFactor = valExtra.replanCostFactor
	mr.geoPoseOrigin = *spatialmath.NewGeoPose(origin, heading)
	return mr, nil
}

// newMoveOnMapRequest instantiates a moveRequest intended to be used in the context of a MoveOnMap call.
func (ms *builtIn) newMoveOnMapRequest(
	ctx context.Context,
	req motion.MoveOnMapReq,
	seedPlan motionplan.Plan,
	replanCount int,
) (state.PlannerExecutor, error) {
	valExtra, err := newValidatedExtra(req.Extra)
	if err != nil {
		return nil, err
	}

	if valExtra.maxReplans >= 0 {
		if replanCount > valExtra.maxReplans {
			return nil, fmt.Errorf("exceeded maximum number of replans: %d", valExtra.maxReplans)
		}
	}

	motionCfg, err := newValidatedMotionCfg(req.MotionCfg, requestTypeMoveOnMap)
	if err != nil {
		return nil, err
	}

	if req.Destination == nil {
		return nil, errors.New("destination cannot be nil")
	}

	// get the SLAM Service from the slamName
	slamSvc, ok := ms.slamServices[req.SlamName]
	if !ok {
		return nil, resource.DependencyNotFoundError(req.SlamName)
	}

	// gets the extents of the SLAM map
	limits, err := slam.Limits(ctx, slamSvc)
	if err != nil {
		return nil, err
	}
	limits = append(limits, referenceframe.Limit{Min: -2 * math.Pi, Max: 2 * math.Pi})

	// create a KinematicBase from the componentName
	component, ok := ms.components[req.ComponentName]
	if !ok {
		return nil, resource.DependencyNotFoundError(req.ComponentName)
	}
	b, ok := component.(base.Base)
	if !ok {
		return nil, fmt.Errorf("cannot move component of type %T because it is not a Base", component)
	}

	// build kinematic options
	kinematicsOptions := kbOptionsFromCfg(motionCfg, valExtra)

	fs, err := ms.fsService.FrameSystem(ctx, nil)
	if err != nil {
		return nil, err
	}

	// Create a localizer from the movement sensor, and collapse reported orientations to 2d
	localizer := motion.TwoDLocalizer(motion.NewSLAMLocalizer(slamSvc))
	kb, err := kinematicbase.WrapWithKinematics(ctx, b, ms.logger, localizer, limits, kinematicsOptions)
	if err != nil {
		return nil, err
	}

	goalPoseAdj := spatialmath.Compose(req.Destination, motion.SLAMOrientationAdjustment)

	// get point cloud data in the form of bytes from pcd
	pointCloudData, err := slam.PointCloudMapFull(ctx, slamSvc)
	if err != nil {
		return nil, err
	}
	// store slam point cloud data  in the form of a recursive octree for collision checking
	octree, err := pointcloud.ReadPCDToBasicOctree(bytes.NewReader(pointCloudData))
	if err != nil {
		return nil, err
	}

	req.Obstacles = append(req.Obstacles, octree)

	mr, err := ms.relativeMoveRequestFromAbsolute(
		ctx,
		motionCfg,
		ms.logger,
		kb,
		goalPoseAdj,
		fs,
		req.Obstacles,
		valExtra,
		requestTypeMoveOnMap,
	)
	if err != nil {
		return nil, err
	}
	return mr, nil
}

func (ms *builtIn) relativeMoveRequestFromAbsolute(
	ctx context.Context,
	motionCfg *validatedMotionConfiguration,
	logger logging.Logger,
	kb kinematicbase.KinematicBase,
	goalPoseInWorld spatialmath.Pose,
	fs referenceframe.FrameSystem,
	worldObstacles []spatialmath.Geometry,
	valExtra validatedExtra,
	reqType requestType,
) (*moveRequest, error) {
	// replace original base frame with one that knows how to move itself and allow planning for
	kinematicFrame := kb.Kinematics()
	if err := fs.ReplaceFrame(kinematicFrame); err != nil {
		// If the base frame is not in the frame system, add it to world. This will result in planning for a frame system containing
		// only world and the base after the FrameSystemSubset.
		err = fs.AddFrame(kinematicFrame, fs.Frame(referenceframe.World))
		if err != nil {
			return nil, err
		}
	}
	// We want to disregard anything in the FS whose eventual parent is not the base, because we don't know where it is.
	baseOnlyFS, err := fs.FrameSystemSubset(kinematicFrame)
	if err != nil {
		return nil, err
	}

	startPoseIF, err := kb.CurrentPosition(ctx)
	if err != nil {
		return nil, err
	}
	startPose := startPoseIF.Pose()
	startPoseInv := spatialmath.PoseInverse(startPose)

	goal := referenceframe.NewPoseInFrame(referenceframe.World, spatialmath.PoseBetween(startPose, goalPoseInWorld))

	// Here we determine if we already are at the goal
	// If our motion profile is position_only then, we only check against our current & desired position
	// Conversely if our motion profile is anything else, then we also need to check again our
	// current & desired orientation
	if valExtra.motionProfile == motionplan.PositionOnlyMotionProfile {
		if spatialmath.PoseAlmostCoincidentEps(goal.Pose(), spatialmath.NewZeroPose(), motionCfg.planDeviationMM) {
			return nil, errGoalWithinPlanDeviation
		}
	} else {
		if spatialmath.OrientationAlmostEqual(goal.Pose().Orientation(), spatialmath.NewZeroPose().Orientation()) &&
			spatialmath.PoseAlmostCoincidentEps(goal.Pose(), spatialmath.NewZeroPose(), motionCfg.planDeviationMM) {
			return nil, errGoalWithinPlanDeviation
		}
	}

	// convert GeoObstacles into GeometriesInFrame with respect to the base's starting point
	geoms := make([]spatialmath.Geometry, 0, len(worldObstacles))
	for _, geom := range worldObstacles {
		geoms = append(geoms, geom.Transform(startPoseInv))
	}

	gif := referenceframe.NewGeometriesInFrame(referenceframe.World, geoms)
	worldState, err := referenceframe.NewWorldState([]*referenceframe.GeometriesInFrame{gif}, nil)
	if err != nil {
		return nil, err
	}

	obstacleDetectors := make(map[vision.Service][]resource.Name)
	for _, obstacleDetectorNamePair := range motionCfg.obstacleDetectors {
		// get vision service
		visionServiceName := obstacleDetectorNamePair.VisionServiceName
		visionSvc, ok := ms.visionServices[visionServiceName]
		if !ok {
			return nil, resource.DependencyNotFoundError(visionServiceName)
		}

		// add camera to vision service map
		camList, ok := obstacleDetectors[visionSvc]
		if !ok {
			obstacleDetectors[visionSvc] = []resource.Name{obstacleDetectorNamePair.CameraName}
		} else {
			camList = append(camList, obstacleDetectorNamePair.CameraName)
			obstacleDetectors[visionSvc] = camList
		}
	}

	currentInputs, _, err := ms.fsService.CurrentInputs(ctx)
	if err != nil {
		return nil, err
	}

	var backgroundWorkers sync.WaitGroup

	var waypointIndex atomic.Int32
	waypointIndex.Store(1)

	// effectively don't poll if the PositionPollingFreqHz is not provided
	positionPollingFreq := time.Duration(math.MaxInt64)
	if motionCfg.positionPollingFreqHz > 0 {
		positionPollingFreq = time.Duration(1000/motionCfg.positionPollingFreqHz) * time.Millisecond
	}

	// effectively don't poll if the ObstaclePollingFreqHz is not provided
	obstaclePollingFreq := time.Duration(math.MaxInt64)
	if motionCfg.obstaclePollingFreqHz > 0 {
		obstaclePollingFreq = time.Duration(1000/motionCfg.obstaclePollingFreqHz) * time.Millisecond
	}

	mr := &moveRequest{
		config: motionCfg,
		logger: ms.logger,
		planRequest: &motionplan.PlanRequest{
			Logger:             logger,
			Goal:               goal,
			Frame:              kinematicFrame,
			FrameSystem:        baseOnlyFS,
			StartConfiguration: currentInputs,
			WorldState:         worldState,
			Options:            valExtra.extra,
		},
		kinematicBase:     kb,
		replanCostFactor:  valExtra.replanCostFactor,
		obstacleDetectors: obstacleDetectors,
		fsService:         ms.fsService,

		executeBackgroundWorkers: &backgroundWorkers,

		responseChan: make(chan moveResponse, 1),

		waypointIndex: &waypointIndex,
		requestType:   reqType,
	}
	if reqType == requestTypeMoveOnMap {
		mr.poseOrigin = startPose
	}

	// TODO: Change deviatedFromPlan to just query positionPollingFreq on the struct & the same for the obstaclesIntersectPlan
	mr.position = newReplanner(positionPollingFreq, mr.deviatedFromPlan)
	mr.obstacle = newReplanner(obstaclePollingFreq, mr.obstaclesIntersectPlan)
	return mr, nil
}

type moveResponse struct {
	err             error
	executeResponse state.ExecuteResponse
}

func (mr moveResponse) String() string {
	return fmt.Sprintf("builtin.moveResponse{executeResponse: %#v, err: %v}", mr.executeResponse, mr.err)
}

func (mr *moveRequest) start(ctx context.Context, waypoints [][]referenceframe.Input) {
	if ctx.Err() != nil {
		return
	}
	mr.executeBackgroundWorkers.Add(1)
	goutils.ManagedGo(func() {
		mr.position.startPolling(ctx, waypoints, mr.waypointIndex)
	}, mr.executeBackgroundWorkers.Done)

	mr.executeBackgroundWorkers.Add(1)
	goutils.ManagedGo(func() {
		mr.obstacle.startPolling(ctx, waypoints, mr.waypointIndex)
	}, mr.executeBackgroundWorkers.Done)

	// spawn function to execute the plan on the robot
	mr.executeBackgroundWorkers.Add(1)
	goutils.ManagedGo(func() {
		executeResp, err := mr.execute(ctx, waypoints, mr.waypointIndex)
		resp := moveResponse{executeResponse: executeResp, err: err}
		mr.responseChan <- resp
	}, mr.executeBackgroundWorkers.Done)
}

func (mr *moveRequest) listen(ctx context.Context) (state.ExecuteResponse, error) {
	select {
	case <-ctx.Done():
		mr.logger.CDebugf(ctx, "context err: %s", ctx.Err())
		return state.ExecuteResponse{}, ctx.Err()

	case resp := <-mr.responseChan:
		mr.logger.CDebugf(ctx, "execution response: %s", resp)
		return resp.executeResponse, resp.err

	case resp := <-mr.position.responseChan:
		mr.logger.CDebugf(ctx, "position response: %s", resp)
		return resp.executeResponse, resp.err

	case resp := <-mr.obstacle.responseChan:
		mr.logger.CDebugf(ctx, "obstacle response: %s", resp)
		return resp.executeResponse, resp.err
	}
}

func (mr *moveRequest) Execute(ctx context.Context, waypoints state.Waypoints) (state.ExecuteResponse, error) {
	defer mr.executeBackgroundWorkers.Wait()
	cancelCtx, cancelFn := context.WithCancel(ctx)
	defer cancelFn()

	mr.start(cancelCtx, waypoints)
	return mr.listen(cancelCtx)
}

func (mr *moveRequest) stop() error {
	stopCtx, cancelFn := context.WithTimeout(context.Background(), baseStopTimeout)
	defer cancelFn()
	if stopErr := mr.kinematicBase.Stop(stopCtx, nil); stopErr != nil {
		mr.logger.Errorf("kinematicBase.Stop returned error %s", stopErr)
		return stopErr
	}
	return nil
}

func toGeoPosePlanSteps(posesByComponent []motionplan.PlanStep, geoPoses []spatialmath.GeoPose) ([]motionplan.PlanStep, error) {
	if len(geoPoses) != len(posesByComponent) {
		msg := "GeoPoses (len: %d) & PosesByComponent (len: %d) must have the same length"
		return nil, fmt.Errorf(msg, len(geoPoses), len(posesByComponent))
	}
	steps := make([]motionplan.PlanStep, 0, len(posesByComponent))
	for i, ps := range posesByComponent {
		if len(ps) == 0 {
			continue
		}

		if l := len(ps); l > 1 {
			return nil, fmt.Errorf("only single component or fewer plan steps supported, received plan step with %d componenents", l)
		}

		var resourceName resource.Name
		for k := range ps {
			resourceName = k
		}
		geoPose := geoPoses[i]
		heading := math.Mod(math.Abs(geoPose.Heading()-360), 360)
		o := &spatialmath.OrientationVectorDegrees{OZ: 1, Theta: heading}
		poseContainingGeoPose := spatialmath.NewPose(r3.Vector{X: geoPose.Location().Lng(), Y: geoPose.Location().Lat()}, o)
		steps = append(steps, map[resource.Name]spatialmath.Pose{resourceName: poseContainingGeoPose})
	}
	return steps, nil
}
