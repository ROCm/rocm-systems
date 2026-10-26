// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file conditional_wake_spinlock.h
/// @brief Parking lock whose release needs no read-modify-write on Linux.

#pragma once

#if defined(__linux__)

#include "util/host_target.h"

#include <atomic>
#include <cstdint>

// TSAN annotations: let ThreadSanitizer track this lock as a mutex.
#if defined(__SANITIZE_THREAD__)
#define CONDITIONAL_WAKE_SPINLOCK_TSAN 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define CONDITIONAL_WAKE_SPINLOCK_TSAN 1
#endif
#endif

#ifdef CONDITIONAL_WAKE_SPINLOCK_TSAN
#include <sanitizer/tsan_interface.h>
#endif

namespace util {

/// @brief Parking lock for many independent, rarely contended critical sections.
///
/// @details Choose this when uncontended release cost dominates. Sustained
/// contention can have slower handoffs than Spinlock, so this is not a general
/// replacement. The lock must outlive every in-flight unlock() call: release
/// still reads the waiter hint after publishing ownership. A next owner must
/// not destroy the lock until those calls return. Non-Linux hosts use the
/// existing Spinlock implementation.
///
/// An uncontended acquisition is one compare-exchange, and a release
/// is one store plus a load of the parked-waiter count in the same cache
/// line. A contended acquisition spins briefly and then parks on the lock
/// word through the selected host policy with a bounded timeout. Release has no
/// store-load fence, so a waiter that registers while a release is in flight
/// can miss its wakeup; it then rechecks the lock when its park times out.
/// Each failed recheck doubles the timeout up to kMaxParkNs; scheduling can
/// extend any sleep beyond its timeout. A policy can wait on the lock word or
/// simply sleep until its timeout. Keeping the policy explicit avoids the
/// standard library's shared waiter buckets, in which one unrelated sleeper
/// makes every release enter the kernel.
///
/// @tparam Parking Host wait policy with wait(word, expected, timeout_ns) and
/// wake(word). Wait must return after its timeout even if a wake is missed.
/// Keeping host calls in the policy lets interposers select their native API.
template <typename Parking> class ConditionalWakeSpinlock {
public:
  void lock() noexcept {
#ifdef CONDITIONAL_WAKE_SPINLOCK_TSAN
    __tsan_mutex_pre_lock(this, 0);
#endif
    if (!try_acquire())
      lock_contended();
#ifdef CONDITIONAL_WAKE_SPINLOCK_TSAN
    __tsan_mutex_post_lock(this, 0, 0);
#endif
  }

  void unlock() noexcept {
#ifdef CONDITIONAL_WAKE_SPINLOCK_TSAN
    __tsan_mutex_pre_unlock(this, 0);
#endif
    state_.store(kFree, std::memory_order_release);
    if (parked_.load(std::memory_order_relaxed) != 0)
      Parking::wake(futex_word());
#ifdef CONDITIONAL_WAKE_SPINLOCK_TSAN
    __tsan_mutex_post_unlock(this, 0);
#endif
  }

  bool try_lock() noexcept {
#ifdef CONDITIONAL_WAKE_SPINLOCK_TSAN
    __tsan_mutex_pre_lock(this, __tsan_mutex_try_lock);
#endif
    bool acquired = try_acquire();
#ifdef CONDITIONAL_WAKE_SPINLOCK_TSAN
    if (acquired)
      __tsan_mutex_post_lock(this, __tsan_mutex_try_lock, 0);
    else
      __tsan_mutex_post_lock(this, __tsan_mutex_try_lock | __tsan_mutex_try_lock_failed, 0);
#endif
    return acquired;
  }

private:
  static constexpr uint32_t kFree = 0;
  static constexpr uint32_t kLocked = 1;
  static constexpr unsigned kSpinCount = 64;
  static constexpr long kFirstParkNs = 50'000;
  static constexpr long kMaxParkNs = 2'000'000;

  // Native word-wait policies require a lock-free 32-bit storage location.
  static_assert(std::atomic<uint32_t>::is_always_lock_free);
  static_assert(sizeof(std::atomic<uint32_t>) == sizeof(uint32_t) &&
                alignof(std::atomic<uint32_t>) == alignof(uint32_t));

  uint32_t *futex_word() noexcept { return reinterpret_cast<uint32_t *>(&state_); }

  bool try_acquire() noexcept {
    uint32_t expected = kFree;
    return state_.compare_exchange_strong(expected, kLocked, std::memory_order_acquire,
                                          std::memory_order_relaxed);
  }

  // Keep this small wait-loop hint local rather than depending on simulator code.
  static void relax_cpu() noexcept {
#if RJ_HOST_HAS_GNU_X86_PAUSE_BUILTIN
    __builtin_ia32_pause();
#elif RJ_HOST_ARCH_AARCH64
    asm volatile("yield");
#else
    std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
  }

  void lock_contended() noexcept {
    for (unsigned spin = 0; spin < kSpinCount; ++spin) {
      if (state_.load(std::memory_order_relaxed) == kFree && try_acquire())
        return;
      relax_cpu();
    }
    long timeout_ns = kFirstParkNs;
    for (;;) {
      // The seq_cst increment orders registration before the recheck below,
      // so a release that saw no parked waiter either published kFree before
      // this recheck or raced it; the timeout bounds that race.
      parked_.fetch_add(1, std::memory_order_seq_cst);
      if (state_.load(std::memory_order_seq_cst) != kFree) {
        // One bounded wait. Wakeup, timeout, EINTR and EAGAIN all return to
        // the ownership recheck, and none of them resets the backoff.
        Parking::wait(futex_word(), kLocked, timeout_ns);
      }
      parked_.fetch_sub(1, std::memory_order_relaxed);
      if (try_acquire())
        return;
      timeout_ns = timeout_ns > kMaxParkNs / 2 ? kMaxParkNs : timeout_ns * 2;
    }
  }

  std::atomic<uint32_t> state_{kFree};
  std::atomic<uint32_t> parked_{0};
};

} // namespace util

#ifdef CONDITIONAL_WAKE_SPINLOCK_TSAN
#undef CONDITIONAL_WAKE_SPINLOCK_TSAN
#endif

#else // !defined(__linux__)

#include "util/spinlock.h"

namespace util {

/// @brief Without futex timeouts, other hosts keep the existing parking Spinlock.
template <typename Parking> using ConditionalWakeSpinlock = Spinlock;

} // namespace util

#endif // defined(__linux__)
