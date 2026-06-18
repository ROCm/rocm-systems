// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <string_view>

namespace rocjitsu::plugins::race_detector {

/// Abstract profiling interface for optional scope-level timing.
/// WaveRaceState and RaceDetector hold a pointer to this — nullptr
/// means profiling is disabled (zero overhead).
class ProfilerInterface {
public:
  virtual ~ProfilerInterface() = default;
  virtual void beginScope(std::string_view key) = 0;
  virtual void endScope() = 0;
};

/// No-op profiler. All operations are zero-cost.
class NullProfiler : public ProfilerInterface {
public:
  void beginScope(std::string_view) override {}
  void endScope() override {}

  static NullProfiler &instance() {
    static NullProfiler inst;
    return inst;
  }
};

} // namespace rocjitsu::plugins::race_detector
