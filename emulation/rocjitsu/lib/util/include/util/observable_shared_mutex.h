// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file observable_shared_mutex.h
/// @brief A shared mutex that reports how many writers are waiting on it.

#include "util/distributed_shared_mutex.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <shared_mutex>

namespace util {

/// @brief A shared mutex that reports how many writers are blocked on it.
///
/// @details What a mutex guarantees is an absence: while it is held, the work
/// it excludes has not happened. An absence can otherwise only be asserted by
/// waiting and observing that nothing occurred, which proves nothing -- an
/// implementation that never takes the lock satisfies that check whenever the
/// machine is loaded enough to leave the excluded thread unscheduled, so the
/// test passes precisely when the system is least able to justify it.
///
/// Counting blocked writers turns the guarantee into something observable. A
/// waiter can be waited for, so exclusion becomes a fact rather than an
/// inference from elapsed time, and code that never acquires the lock never
/// raises the count and so fails rather than passing by accident.
///
/// The count is maintained only on the exclusive path, which is the contended
/// one worth observing; readers pay nothing for it.
class ObservableSharedMutex {
public:
  /// @brief Exclusive ownership, counted for as long as it waits.
  class ExclusiveGuard {
  public:
    explicit ExclusiveGuard(ObservableSharedMutex &owner) {
      owner.blocked_writers_.fetch_add(1, std::memory_order_release);
      lock_ = std::unique_lock(owner.mutex_);
      owner.blocked_writers_.fetch_sub(1, std::memory_order_release);
    }

    ExclusiveGuard(const ExclusiveGuard &) = delete;
    ExclusiveGuard &operator=(const ExclusiveGuard &) = delete;

    void unlock() { lock_.unlock(); }

  private:
    std::unique_lock<DistributedSharedMutex> lock_;
  };

  /// @brief Take shared ownership. Readers are not counted.
  [[nodiscard]] std::shared_lock<DistributedSharedMutex> lock_shared() {
    return std::shared_lock(mutex_);
  }

  /// @brief Take exclusive ownership, counted while it blocks.
  [[nodiscard]] ExclusiveGuard lock_exclusive() { return ExclusiveGuard(*this); }

  /// @brief How many threads are currently blocked taking this exclusively.
  [[nodiscard]] uint64_t blocked_writers() const {
    return blocked_writers_.load(std::memory_order_acquire);
  }

private:
  DistributedSharedMutex mutex_;
  std::atomic<uint64_t> blocked_writers_{0};
};

} // namespace util
