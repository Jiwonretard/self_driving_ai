# Model card: DroneNav Tiny

## Intended use

`DroneNav Tiny` classifies a forward-facing camera frame into one of four high-level navigation suggestions: `forward`, `left`, `right`, or `stop`. It is intended as a perception input to a separate flight controller on a small indoor research drone.

## Not intended for

- direct motor control or attitude stabilization
- operation near people, roads, property, or unrestricted outdoor airspace
- estimating obstacle distance, optical flow, depth, altitude, or pose
- replacing a range sensor, IMU, barometer, GPS, visual-inertial odometry, or flight-controller fail-safe

## Architecture

- 96×96 RGB input
- one 3×3 strided convolution
- three depthwise 3×3 + pointwise 1×1 blocks
- global mean pooling and four-way softmax
- full integer post-training quantization with representative images

The graph is deliberately restricted to five TFLite Micro operator types so the firmware can use a small resolver instead of linking every operator.

## Required evaluation

Overall accuracy alone is not an acceptance criterion. Measure a confusion matrix, per-class recall, and especially `stop` recall on session-separated data. Also record on-device latency, tensor-arena usage, and failure behavior with camera disconnects and corrupted frames.

