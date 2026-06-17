// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file spinlock.h
/// @brief TTAS (test-and-test-and-set) spinlock for low-contention critical sections.

#ifndef UTIL_SPINLOCK_H_
#define UTIL_SPINLOCK_H_

#include <atomic>

// TSAN annotations: let ThreadSanitizer track this spinlock as a mutex.
#if defined(__SANITIZE_THREAD__)
#define SPINLOCK_TSAN 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define SPINLOCK_TSAN 1
#endif
#endif

#ifdef SPINLOCK_TSAN
#include <sanitizer/tsan_interface.h>
#endif

namespace util {

/// @brief TTAS spinlock suitable for sub-microsecond critical sections.
///
/// @details Uses a test-and-test-and-set strategy: spins on a relaxed load
/// (cache-local) until the lock appears free, then attempts an exchange
/// (cache-line transfer). Uses C++20 atomic::wait() for efficient spinning
/// (maps to pause on x86, WFE on AArch64).
///
/// Use for hot paths where the critical section is very short (e.g.,
/// cross-partition queue push/drain). For longer critical sections or
/// blocking waits, prefer std::mutex.
class Spinlock {
public:
  void lock() noexcept {
#ifdef SPINLOCK_TSAN
    __tsan_mutex_pre_lock(this, 0);
#endif
    while (true) {
      // Outer TTAS spin: cache-local relaxed read avoids coherence traffic.
      while (locked_.load(std::memory_order_relaxed))
        locked_.wait(true, std::memory_order_relaxed);
      // Attempt to acquire via exchange (cache-line transfer).
      if (!locked_.exchange(true, std::memory_order_acquire)) {
#ifdef SPINLOCK_TSAN
        __tsan_mutex_post_lock(this, 0, 0);
#endif
        return;
      }
    }
  }

  void unlock() noexcept {
#ifdef SPINLOCK_TSAN
    __tsan_mutex_pre_unlock(this, 0);
#endif
    locked_.store(false, std::memory_order_release);
    locked_.notify_one();
#ifdef SPINLOCK_TSAN
    __tsan_mutex_post_unlock(this, 0);
#endif
  }

  bool try_lock() noexcept {
#ifdef SPINLOCK_TSAN
    __tsan_mutex_pre_lock(this, __tsan_mutex_try_lock);
#endif
    bool acquired = !locked_.exchange(true, std::memory_order_acquire);
#ifdef SPINLOCK_TSAN
    if (acquired)
      __tsan_mutex_post_lock(this, __tsan_mutex_try_lock, 0);
    else
      __tsan_mutex_post_lock(this, __tsan_mutex_try_lock | __tsan_mutex_try_lock_failed, 0);
#endif
    return acquired;
  }

private:
  std::atomic<bool> locked_{false};
};

} // namespace util

#endif // UTIL_SPINLOCK_H_
