// Package register registers all relevant Sensors
package register

import (
	// for Sensors.
	_ "go.viam.com/rdk/components/sensor/charge"
	_ "go.viam.com/rdk/components/sensor/ds18b20"
	_ "go.viam.com/rdk/components/sensor/fake"
	_ "go.viam.com/rdk/components/sensor/ultrasonic"
)
