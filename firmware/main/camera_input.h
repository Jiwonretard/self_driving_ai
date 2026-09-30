#pragma once

#include <cstdint>

#include "esp_err.h"

namespace drone_nav {

constexpr int kImageWidth = 96;
constexpr int kImageHeight = 96;
constexpr int kImageChannels = 3;

esp_err_t InitCamera();

// Captures a center-cropped frame and quantizes RGB pixels into the model input.
esp_err_t CaptureModelInput(uint8_t* destination, float scale, int zero_point);

}  // namespace drone_nav

