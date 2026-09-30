#include "decision_filter.h"

#include <algorithm>

namespace drone_nav {

const char* CommandName(Command command) {
  switch (command) {
    case Command::kForward:
      return "forward";
    case Command::kLeft:
      return "left";
    case Command::kRight:
      return "right";
    case Command::kStop:
      return "stop";
  }
  return "stop";
}

DecisionFilter::DecisionFilter(float confidence_threshold, size_t confirmation_frames)
    : confidence_threshold_(std::clamp(confidence_threshold, 0.0f, 1.0f)),
      confirmation_frames_(std::max<size_t>(confirmation_frames, 1)) {}

Command DecisionFilter::Update(Command candidate, float confidence) {
  if (candidate == Command::kStop || confidence < confidence_threshold_) {
    Reset();
    return Command::kStop;
  }
  if (candidate != pending_) {
    pending_ = candidate;
    pending_count_ = 1;
    return Command::kStop;
  }
  ++pending_count_;
  return pending_count_ >= confirmation_frames_ ? pending_ : Command::kStop;
}

void DecisionFilter::Reset() {
  pending_ = Command::kStop;
  pending_count_ = 0;
}

}  // namespace drone_nav

