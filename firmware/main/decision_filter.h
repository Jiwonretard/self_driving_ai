#pragma once

#include <cstddef>

namespace drone_nav {

enum class Command : size_t { kForward = 0, kLeft = 1, kRight = 2, kStop = 3 };

const char* CommandName(Command command);

class DecisionFilter {
 public:
  DecisionFilter(float confidence_threshold, size_t confirmation_frames);
  Command Update(Command candidate, float confidence);
  void Reset();

 private:
  float confidence_threshold_;
  size_t confirmation_frames_;
  Command pending_ = Command::kStop;
  size_t pending_count_ = 0;
};

}  // namespace drone_nav

