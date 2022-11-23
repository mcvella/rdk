//go:build !arm

package builtin

import (
	"context"
	"image"
	"math"

	"github.com/edaniels/golog"
	"github.com/pkg/errors"
	"go.opencensus.io/trace"

	"go.viam.com/rdk/config"
	inf "go.viam.com/rdk/ml/inference"
	"go.viam.com/rdk/rimage"
	"go.viam.com/rdk/services/vision"
	"go.viam.com/rdk/utils"
	kp "go.viam.com/rdk/vision/keypoints"
	"go.viam.com/rdk/vision/objectdetection"
)

// FeatureMatchDetectorConfig specifies the fields necessary for creating a feature match detector.
type FeatureMatchDetectorConfig struct {
	// this should come from the attributes part of the detector config
	ReferenceImagePath string `json:"reference_image_path"`
	MaxDist            int    `json:"max_match_distance,omitempty"`
}

type OrbKP struct {
	descriptors [][]uint64
	keypoints   kp.KeyPoints
}

// NewFeatureMatchDetector creates an RDK detector given a DetectorConfig. In other words, this
// function returns a function from image-->[]objectdetection.Detection. It does this by making calls to
// a keypoints package and wrapping the result.
func NewFeatureMatchDetector(
	ctx context.Context,
	cfg *vision.VisModelConfig,
	logger golog.Logger,
) (objectdetection.Detector, error) {
	ctx, span := trace.StartSpan(ctx, "service::vision::NewFeatureMatchDetector")
	defer span.End()

	// Read those parameters into a FeatureMatchDetectorConfig
	var t FeatureMatchDetectorConfig
	fmParams, err := config.TransformAttributeMapToStruct(&t, cfg.Parameters)
	if err != nil {
		return nil, errors.New("error getting parameters from config")
	}
	params, ok := fmParams.(*FeatureMatchDetectorConfig)
	if !ok {
		err := utils.NewUnexpectedTypeError(params, fmParams)
		return nil, errors.Wrapf(err, "register feature match detector %s", cfg.Name)
	}

	// load reference image and compute keypoints
	img, err := rimage.NewImageFromFile(params.ReferenceImagePath)
	if err != nil {
		return nil, errors.Wrap(err, "something wrong with loading the reference image")
	}
	refKPs, err := getImageKeypoints(ctx, img)
	if err != nil {
		return nil, errors.Wrap(err, "something wrong computing keypoints")
	}

	// This function to be returned is the detector.
	return func(ctx context.Context, img image.Image) ([]objectdetection.Detection, error) {
		if params.MaxDist == 0 {
			params.MaxDist = 50
		}
		matchingConf := &kp.MatchingConfig{
			DoCrossCheck: true,
			MaxDist:      params.MaxDist,
		}
		imgKPs, err := getImageKeypoints(ctx, rimage.ConvertImage(img))
		if err != nil {
			return nil, errors.Wrap(err, "something wrong getting image keypoints")
		}
		matches := kp.MatchDescriptors(refKPs.descriptors, imgKPs.descriptors, matchingConf, logger)
		bounds := getBoundingBox(matches, imgKPs.keypoints)

		// Only ever return max one detection
		var detections = []objectdetection.Detection{}
		detections[0] = objectdetection.NewDetection(bounds, 1, "match")
		return detections, nil
	}, nil
}

// getBoundingBox returns a rectangle based on min/max x,y of matches in the match image
func getBoundingBox(matches []kp.DescriptorMatch, pts kp.KeyPoints) image.Rectangle {
	min := image.Point{math.MaxInt32, math.MaxInt32}
	max := image.Point{0, 0}

	for _, match := range matches {
		m := pts[match.Idx2]
		if m.X < min.X {
			min.X = m.X
		}
		if m.Y < min.Y {
			min.Y = m.Y
		}

		if m.X > max.X {
			max.X = m.X
		}
		if m.Y > max.Y {
			max.Y = m.Y
		}
	}

	return image.Rectangle{min, max}
}

// getImageKeypoints reads an image from the specified path and
// returns descriptors and keypoints, which are cached for detector matching
func getImageKeypoints(ctx context.Context, img *rimage.Image) (*OrbKP, error) {
	_, span := trace.StartSpan(ctx, "service::vision::getImageKeypoints")
	defer span.End()

	orbConf := &kp.ORBConfig{
		Layers:          4,
		DownscaleFactor: 2,
		FastConf: &kp.FASTConfig{
			NMatchesCircle: 9,
			NMSWinSize:     7,
			Threshold:      20,
			Oriented:       true,
			Radius:         16,
		},
		BRIEFConf: &kp.BRIEFConfig{
			N:              512,
			Sampling:       2,
			UseOrientation: true,
			PatchSize:      48,
		},
	}

	imG := rimage.MakeGray(img)
	samplePoints := kp.GenerateSamplePairs(orbConf.BRIEFConf.Sampling, orbConf.BRIEFConf.N, orbConf.BRIEFConf.PatchSize)
	var kps OrbKP
	orb, kp, err := kp.ComputeORBKeypoints(imG, samplePoints, orbConf)
	if err != nil {
		return nil, err
	}
	kps.descriptors = orb
	kps.keypoints = kp

	return &kps, nil
}

// matchFeatures first converts an input image to a buffer using the imageToBuffer func
// and then performs descriptor matching.
func matchFeatures(ctx context.Context, model *inf.TFLiteStruct, image image.Image) ([]interface{}, error) {
	_, span := trace.StartSpan(ctx, "service::vision::tfliteInfer")
	defer span.End()

	// Converts the image to bytes before sending it off
	switch model.Info.InputTensorType {
	case inf.UInt8:
		imgBuff := ImageToUInt8Buffer(image)
		out, err := model.Infer(imgBuff)
		if err != nil {
			return nil, errors.Wrap(err, "couldn't infer from model")
		}
		return out, nil
	case inf.Float32:
		imgBuff := ImageToFloatBuffer(image)
		out, err := model.Infer(imgBuff)
		if err != nil {
			return nil, errors.Wrap(err, "couldn't infer from model")
		}
		return out, nil
	default:
		return nil, errors.New("invalid input type. try uint8 or float32")
	}
}
