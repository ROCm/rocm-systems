// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Lightweight lock for short hot-path critical sections.

#pragma once

#include <atomic>

// Let ThreadSanitizer track this lock as a mutex. Without the annotations TSan
// sees the state it protects as unsynchronised and reports races on it. The
// interface is compiler-provided and the symbols are weak, so annotating keeps
// the core dependency-free.
#if defined(__SANITIZE_THREAD__)
#define HAZARD_CORE_SPIN_LOCK_TSAN 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define HAZARD_CORE_SPIN_LOCK_TSAN 1
#endif
#endif

#ifdef HAZARD_CORE_SPIN_LOCK_TSAN
#include <sanitizer/tsan_interface.h>
#endif

namespace hazard_core {

class SpinLock {
  std::atomic_flag is_locked_ = ATOMIC_FLAG_INIT;

public:
  void lock() noexcept {
#ifdef HAZARD_CORE_SPIN_LOCK_TSAN
    __tsan_mutex_pre_lock(this, 0);
#endif
    while (is_locked_.test_and_set(std::memory_order_acquire))
      is_locked_.wait(true, std::memory_order_relaxed);
#ifdef HAZARD_CORE_SPIN_LOCK_TSAN
    __tsan_mutex_post_lock(this, 0, 0);
#endif
  }

  void unlock() noexcept {
#ifdef HAZARD_CORE_SPIN_LOCK_TSAN
    __tsan_mutex_pre_unlock(this, 0);
#endif
    is_locked_.clear(std::memory_order_release);
    is_locked_.notify_one();
#ifdef HAZARD_CORE_SPIN_LOCK_TSAN
    __tsan_mutex_post_unlock(this, 0);
#endif
  }
};

} // namespace hazard_core
