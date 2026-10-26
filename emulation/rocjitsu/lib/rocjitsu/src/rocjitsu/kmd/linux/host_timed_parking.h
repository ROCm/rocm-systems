// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file host_timed_parking.h
/// @brief Bounded libc waits for runtime-owned synchronization words.

#pragma once

#include "rocjitsu/kmd/linux/scoped_cancellation_disable.h"

#include <cstdint>
#include <ctime>

namespace rocjitsu {

/// @brief Parking policy that rechecks ownership after a bounded libc sleep.
/// @details Libc has no named futex wrapper. Use the lock's existing bounded
/// timeout contract without allocating descriptors or resolving symbols on a
/// contended path. Release cannot interrupt the sleep, so contended handoff may
/// take the current backoff interval. Defer cancellation during each sleep:
/// lock acquisition is noexcept and must remove its parked-waiter registration.
class HostTimedParking {
public:
  static void wait(uint32_t *, uint32_t, long timeout_ns) {
    ScopedCancellationDisable cancellation;
    const timespec timeout{0, timeout_ns};
    static_cast<void>(::clock_nanosleep(CLOCK_MONOTONIC, 0, &timeout, nullptr));
  }

  static void wake(uint32_t *) {}
};

} // namespace rocjitsu
