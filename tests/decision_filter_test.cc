#include <cassert>

#include "decision_filter.h"

int main() {
  using drone_nav::Command;
  using drone_nav::DecisionFilter;

  DecisionFilter filter(0.72f, 3);
  assert(filter.Update(Command::kForward, 0.90f) == Command::kStop);
  assert(filter.Update(Command::kForward, 0.90f) == Command::kStop);
  assert(filter.Update(Command::kForward, 0.90f) == Command::kForward);
  assert(filter.Update(Command::kLeft, 0.90f) == Command::kStop);
  assert(filter.Update(Command::kLeft, 0.50f) == Command::kStop);
  assert(filter.Update(Command::kLeft, 0.90f) == Command::kStop);
  assert(filter.Update(Command::kLeft, 0.90f) == Command::kStop);
  assert(filter.Update(Command::kLeft, 0.90f) == Command::kLeft);
  assert(filter.Update(Command::kStop, 1.0f) == Command::kStop);
  return 0;
}

