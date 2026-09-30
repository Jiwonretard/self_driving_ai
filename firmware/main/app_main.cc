#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "camera_input.h"
#include "decision_filter.h"
#include "driver/uart.h"
#include "esp_check.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sdkconfig.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

extern const uint8_t model_start[] asm(
    "_binary_model_drone_nav_int8_tflite_start");

namespace drone_nav {
namespace {

constexpr char kTag[] = "drone_nav";
constexpr size_t kClassCount = 4;

esp_err_t InitUart() {
  const uart_port_t port = static_cast<uart_port_t>(CONFIG_DRONE_NAV_UART_PORT);
  uart_config_t config = {};
  config.baud_rate = CONFIG_DRONE_NAV_UART_BAUD;
  config.data_bits = UART_DATA_8_BITS;
  config.parity = UART_PARITY_DISABLE;
  config.stop_bits = UART_STOP_BITS_1;
  config.flow_ctrl = UART_HW_FLOWCTRL_DISABLE;
  config.rx_flow_ctrl_thresh = 0;
  config.source_clk = UART_SCLK_DEFAULT;
  ESP_RETURN_ON_ERROR(uart_driver_install(port, 512, 0, 0, nullptr, 0), kTag,
                      "uart_driver_install");
  ESP_RETURN_ON_ERROR(uart_param_config(port, &config), kTag, "uart_param_config");
  return uart_set_pin(port, CONFIG_DRONE_NAV_UART_TX_PIN, CONFIG_DRONE_NAV_UART_RX_PIN,
                      UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);
}

void SendDecision(uint32_t sequence, Command command, float confidence,
                  int64_t inference_ms) {
  char line[96];
  const int length = std::snprintf(line, sizeof(line), "NAV,%" PRIu32 ",%s,%.3f,%" PRId64
                                                       "\n",
                                   sequence, CommandName(command), confidence, inference_ms);
  if (length > 0) {
    uart_write_bytes(static_cast<uart_port_t>(CONFIG_DRONE_NAV_UART_PORT), line,
                     std::min(length, static_cast<int>(sizeof(line) - 1)));
  }
}

float Dequantize(uint8_t value, const TfLiteQuantizationParams& params) {
  return (static_cast<int>(value) - params.zero_point) * params.scale;
}

}  // namespace

void Run() {
  ESP_ERROR_CHECK(InitUart());
  ESP_ERROR_CHECK(InitCamera());

  const tflite::Model* model = tflite::GetModel(model_start);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(kTag, "TFLite schema mismatch: model=%d runtime=%d", model->version(),
             TFLITE_SCHEMA_VERSION);
    return;
  }

  tflite::MicroMutableOpResolver<5> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddMean();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();

  constexpr size_t kArenaBytes = CONFIG_DRONE_NAV_TENSOR_ARENA_KB * 1024;
  auto* arena = static_cast<uint8_t*>(
      heap_caps_malloc(kArenaBytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (arena == nullptr) {
    arena = static_cast<uint8_t*>(heap_caps_malloc(kArenaBytes, MALLOC_CAP_8BIT));
  }
  if (arena == nullptr) {
    ESP_LOGE(kTag, "cannot allocate %zu-byte tensor arena", kArenaBytes);
    return;
  }

  tflite::MicroInterpreter interpreter(model, resolver, arena, kArenaBytes);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    ESP_LOGE(kTag, "AllocateTensors failed; increase DRONE_NAV_TENSOR_ARENA_KB");
    heap_caps_free(arena);
    return;
  }

  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);
  const bool valid_input =
      input != nullptr && input->type == kTfLiteUInt8 && input->dims->size == 4 &&
      input->dims->data[0] == 1 && input->dims->data[1] == kImageHeight &&
      input->dims->data[2] == kImageWidth && input->dims->data[3] == kImageChannels;
  const bool valid_output = output != nullptr && output->type == kTfLiteUInt8 &&
                            output->bytes == kClassCount;
  if (!valid_input || !valid_output) {
    ESP_LOGE(kTag, "model contract mismatch; expected uint8 [1,96,96,3] -> [1,4]");
    heap_caps_free(arena);
    return;
  }

  DecisionFilter filter(CONFIG_DRONE_NAV_CONFIDENCE_PERCENT / 100.0f,
                        CONFIG_DRONE_NAV_CONFIRM_FRAMES);
  uint32_t sequence = 0;
  ESP_LOGI(kTag, "ready; tensor arena used=%zu/%zu bytes",
           interpreter.arena_used_bytes(), kArenaBytes);

  while (true) {
    ++sequence;
    if (CaptureModelInput(input->data.uint8, input->params.scale,
                          input->params.zero_point) != ESP_OK) {
      filter.Reset();
      SendDecision(sequence, Command::kStop, 0.0f, 0);
      vTaskDelay(pdMS_TO_TICKS(CONFIG_DRONE_NAV_FRAME_INTERVAL_MS));
      continue;
    }

    const int64_t started_us = esp_timer_get_time();
    if (interpreter.Invoke() != kTfLiteOk) {
      ESP_LOGE(kTag, "inference failed");
      filter.Reset();
      SendDecision(sequence, Command::kStop, 0.0f, 0);
      continue;
    }
    const int64_t inference_ms = (esp_timer_get_time() - started_us) / 1000;

    size_t best_index = 0;
    for (size_t i = 1; i < kClassCount; ++i) {
      if (output->data.uint8[i] > output->data.uint8[best_index]) {
        best_index = i;
      }
    }
    const float confidence =
        std::clamp(Dequantize(output->data.uint8[best_index], output->params), 0.0f, 1.0f);
    const Command command = filter.Update(static_cast<Command>(best_index), confidence);
    SendDecision(sequence, command, confidence, inference_ms);
    ESP_LOGI(kTag, "seq=%" PRIu32 " command=%s raw=%s confidence=%.3f latency=%" PRId64
                   "ms",
             sequence, CommandName(command), CommandName(static_cast<Command>(best_index)),
             confidence, inference_ms);
    vTaskDelay(pdMS_TO_TICKS(CONFIG_DRONE_NAV_FRAME_INTERVAL_MS));
  }
}

}  // namespace drone_nav

extern "C" void app_main() {
  drone_nav::Run();
}
