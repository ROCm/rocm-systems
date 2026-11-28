// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Establish the default host FP environment for a test and restore its caller.
// Production code assumes this environment and never changes it.
#pragma once

#include "util/host_target.h"
#include <cfenv>
#if RJ_HOST_ARCH_X86
#include <xmmintrin.h>
#endif

namespace rocjitsu {

class ScopedDefaultFpEnvironment {
public:
  ScopedDefaultFpEnvironment() : saved_(std::fegetenv(&environment_) == 0) {
    if (saved_) {
      std::fesetenv(FE_DFL_ENV);
    }
  }
  ScopedDefaultFpEnvironment(const ScopedDefaultFpEnvironment &) = delete;
  ScopedDefaultFpEnvironment &operator=(const ScopedDefaultFpEnvironment &) = delete;
  ~ScopedDefaultFpEnvironment() {
    if (saved_) {
      std::fesetenv(&environment_);
    }
  }

private:
  std::fenv_t environment_{};
  bool saved_;
};

} // namespace rocjitsu
