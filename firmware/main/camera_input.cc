#include "camera_input.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "esp_camera.h"
#include "esp_log.h"
#include "sdkconfig.h"

namespace drone_nav {
namespace {

constexpr char kTag[] = "camera_input";

uint8_t QuantizePixel(uint8_t value, float scale, int zero_point) {
  const int quantized = static_cast<int>(std::lround(value / scale)) + zero_point;
  return static_cast<uint8_t>(std::clamp(quantized, 0, 255));
}

}  // namespace

esp_err_t InitCamera() {
  camera_config_t config = {};
  config.pin_pwdn = CONFIG_DRONE_CAM_PIN_PWDN;
  config.pin_reset = CONFIG_DRONE_CAM_PIN_RESET;
  config.pin_xclk = CONFIG_DRONE_CAM_PIN_XCLK;
  config.pin_sccb_sda = CONFIG_DRONE_CAM_PIN_SIOD;
  config.pin_sccb_scl = CONFIG_DRONE_CAM_PIN_SIOC;
  config.pin_d7 = CONFIG_DRONE_CAM_PIN_D7;
  config.pin_d6 = CONFIG_DRONE_CAM_PIN_D6;
  config.pin_d5 = CONFIG_DRONE_CAM_PIN_D5;
  config.pin_d4 = CONFIG_DRONE_CAM_PIN_D4;
  config.pin_d3 = CONFIG_DRONE_CAM_PIN_D3;
  config.pin_d2 = CONFIG_DRONE_CAM_PIN_D2;
  config.pin_d1 = CONFIG_DRONE_CAM_PIN_D1;
  config.pin_d0 = CONFIG_DRONE_CAM_PIN_D0;
  config.pin_vsync = CONFIG_DRONE_CAM_PIN_VSYNC;
  config.pin_href = CONFIG_DRONE_CAM_PIN_HREF;
  config.pin_pclk = CONFIG_DRONE_CAM_PIN_PCLK;
  config.xclk_freq_hz = 20000000;
  config.ledc_timer = LEDC_TIMER_0;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.pixel_format = PIXFORMAT_RGB565;
  config.frame_size = FRAMESIZE_QQVGA;  // 160x120, then center-crop to square.
  config.jpeg_quality = 12;
  config.fb_count = 1;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.grab_mode = CAMERA_GRAB_LATEST;

  const esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    ESP_LOGE(kTag, "esp_camera_init failed: %s", esp_err_to_name(err));
  }
  return err;
}

esp_err_t CaptureModelInput(uint8_t* destination, float scale, int zero_point) {
  if (destination == nullptr || scale <= 0.0f) {
    return ESP_ERR_INVALID_ARG;
  }

  camera_fb_t* frame = esp_camera_fb_get();
  if (frame == nullptr) {
    ESP_LOGE(kTag, "camera frame capture failed");
    return ESP_FAIL;
  }
  if (frame->format != PIXFORMAT_RGB565 || frame->width == 0 || frame->height == 0) {
    ESP_LOGE(kTag, "unexpected camera format=%d size=%ux%u", frame->format,
             frame->width, frame->height);
    esp_camera_fb_return(frame);
    return ESP_ERR_INVALID_STATE;
  }
  const size_t expected_bytes = frame->width * frame->height * 2;
  if (frame->len < expected_bytes) {
    ESP_LOGE(kTag, "short RGB565 frame: %u < %zu", frame->len, expected_bytes);
    esp_camera_fb_return(frame);
    return ESP_ERR_INVALID_SIZE;
  }

  const size_t crop = std::min(frame->width, frame->height);
  const size_t x_offset = (frame->width - crop) / 2;
  const size_t y_offset = (frame->height - crop) / 2;

  for (size_t y = 0; y < kImageHeight; ++y) {
    const size_t source_y = y_offset + (y * crop) / kImageHeight;
    for (size_t x = 0; x < kImageWidth; ++x) {
      const size_t source_x = x_offset + (x * crop) / kImageWidth;
      const size_t source_index = (source_y * frame->width + source_x) * 2;
      // esp32-camera returns RGB565 bytes in big-endian pixel order.
      const uint16_t pixel =
          (static_cast<uint16_t>(frame->buf[source_index]) << 8) |
          frame->buf[source_index + 1];
      const uint8_t red = static_cast<uint8_t>(((pixel >> 11) & 0x1f) * 255 / 31);
      const uint8_t green = static_cast<uint8_t>(((pixel >> 5) & 0x3f) * 255 / 63);
      const uint8_t blue = static_cast<uint8_t>((pixel & 0x1f) * 255 / 31);
      const size_t destination_index = (y * kImageWidth + x) * kImageChannels;
      destination[destination_index] = QuantizePixel(red, scale, zero_point);
      destination[destination_index + 1] = QuantizePixel(green, scale, zero_point);
      destination[destination_index + 2] = QuantizePixel(blue, scale, zero_point);
    }
  }

  esp_camera_fb_return(frame);
  return ESP_OK;
}

}  // namespace drone_nav
