# Model contract

The firmware intentionally rejects models that do not match this exact interface.

| Item | Value |
|---|---|
| Input tensor | `[1, 96, 96, 3]`, `uint8`, RGB |
| Spatial preprocessing | Center crop to square, nearest-neighbor resize |
| Class 0 | `forward` |
| Class 1 | `left` |
| Class 2 | `right` |
| Class 3 | `stop` |
| Output tensor | `[1, 4]`, `uint8` softmax |
| Allowed ops | Conv2D, DepthwiseConv2D, Mean, FullyConnected, Softmax |
| Fail-safe | Any error or low confidence becomes `stop` |

Changing class order, image dimensions, input type, or preprocessing requires a matching firmware change. Do not replace only the `.tflite` file without rerunning on-device contract and fail-safe tests.

