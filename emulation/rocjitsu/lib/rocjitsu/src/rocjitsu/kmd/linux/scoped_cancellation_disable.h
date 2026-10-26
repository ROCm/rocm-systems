// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <pthread.h>

namespace rocjitsu {

/// @brief Keep deferred pthread cancellation out of cleanup and publication.
/// @details Restoring the caller's state leaves a deferred request pending until
/// the next cancellation point. Do not use this to cover blocking waits. Like
/// the surrounding C++ code, this does not support asynchronous cancellation.
class ScopedCancellationDisable {
public:
  ScopedCancellationDisable() {
    const int result = pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &previous_state_);
    assert(result == 0);
    (void)result;
  }
  ~ScopedCancellationDisable() {
    const int result = pthread_setcancelstate(previous_state_, nullptr);
    assert(result == 0);
    (void)result;
  }
  ScopedCancellationDisable(const ScopedCancellationDisable &) = delete;
  ScopedCancellationDisable &operator=(const ScopedCancellationDisable &) = delete;

private:
  int previous_state_ = PTHREAD_CANCEL_ENABLE;
};

} // namespace rocjitsu
