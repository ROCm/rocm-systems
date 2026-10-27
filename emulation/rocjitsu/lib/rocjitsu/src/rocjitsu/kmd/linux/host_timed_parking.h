// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file host_timed_parking.h
/// @brief Wake-capable bounded libc waits for runtime-owned synchronization words.

#pragma once

#include "rocjitsu/kmd/linux/scoped_cancellation_disable.h"

#include <array>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <mutex>

namespace rocjitsu {

/// @brief Parking policy with monotonic libc waits that release can interrupt.
/// @details Share condition-variable buckets instead of enlarging every backing
/// stripe lock. Its per-lock waiter hint keeps uncontended release out of these
/// buckets. Collisions may wake unrelated waiters, which recheck ownership.
/// Defer cancellation during bounded internal parking: lock acquisition is
/// noexcept and must remove its parked-waiter registration before returning.
class HostTimedParking {
public:
  static void wait(std::atomic<uint32_t> *word, uint32_t expected, long timeout_ns) {
    ScopedCancellationDisable cancellation;
    timespec deadline{};
    if (::clock_gettime(CLOCK_MONOTONIC, &deadline) != 0)
      return;
    deadline.tv_sec += timeout_ns / 1'000'000'000;
    deadline.tv_nsec += timeout_ns % 1'000'000'000;
    deadline.tv_sec += deadline.tv_nsec / 1'000'000'000;
    deadline.tv_nsec %= 1'000'000'000;
    bucket_for(word).wait(word, expected, &deadline);
  }

  static void wake(std::atomic<uint32_t> *word) { bucket_for(word).wake(); }

private:
  class Bucket {
  public:
    Bucket() {
      pthread_condattr_t attributes;
      require_success(pthread_condattr_init(&attributes));
      require_success(pthread_condattr_setclock(&attributes, CLOCK_MONOTONIC));
      require_success(pthread_cond_init(&condition_, &attributes));
      require_success(pthread_condattr_destroy(&attributes));
    }

    void wait(std::atomic<uint32_t> *word, uint32_t expected, const timespec *deadline) {
      std::lock_guard guard(mutex_);
      // Pair this recheck with wake's mutex acquisition so release cannot fall
      // between observing the held word and entering the condition wait.
      if (word->load(std::memory_order_relaxed) != expected)
        return;
      const int result = pthread_cond_timedwait(&condition_, mutex_.native_handle(), deadline);
      assert(result == 0 || result == ETIMEDOUT);
      (void)result;
    }

    void wake() {
      std::lock_guard guard(mutex_);
      // A bucket can contain several lock words; notify_one could select a
      // waiter for a different stripe and strand the intended one until timeout.
      require_success(pthread_cond_broadcast(&condition_));
    }

  private:
    static void require_success(int result) {
      if (result != 0)
        std::abort();
    }

    std::mutex mutex_;
    // The fixed buckets and their native condition variables live until exit.
    pthread_cond_t condition_;
  };

  static Bucket &bucket_for(const std::atomic<uint32_t> *word) {
    static std::array<Bucket, 64> buckets;
    return buckets[(reinterpret_cast<uintptr_t>(word) >> 6) % buckets.size()];
  }
};

} // namespace rocjitsu
