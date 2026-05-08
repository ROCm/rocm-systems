// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file refcount.h
/// @brief Thread-safe reference-counting mixin for opaque C API handles.

#ifndef ROCJITSU_REFCOUNT_H_
#define ROCJITSU_REFCOUNT_H_

#include <atomic>
#include <cassert>
#include <cstdint>

namespace rocjitsu {

/// @brief Reference-counting mixin for opaque C API handle types.
///
/// @details Provides retain/release semantics with atomic operations for thread
/// safety. Objects start with refcount 0. `retain` increments, `release`
/// decrements. When `destroy` is called, the object is marked for destruction.
/// The backing memory is freed when refcount reaches 0 after a destroy has been
/// requested. Exactly one of destroy() or the last release() returns true,
/// never both (no double-free).
///
/// Implementation: the destroyed flag and refcount are packed into a single
/// atomic uint32_t (bit 31 = destroyed, bits [0:30] = refcount). This allows
/// destroy() to atomically set the flag and check the count in one CAS,
/// eliminating the TOCTOU race between separate destroyed_ and ref_count_
/// atomics.
class RefCounted {
public:
  /// @brief Increment the reference count.
  void retain() noexcept {
    [[maybe_unused]] uint32_t prev = state_.fetch_add(1, std::memory_order_relaxed);
    assert((prev & COUNT_MASK) < COUNT_MASK && "refcount overflow");
    assert(!(prev & DESTROYED_BIT) && "retain() called after destroy()");
  }

  /// @brief Decrement the reference count.
  /// @retval true Destroyed and last reference dropped; caller should free.
  /// @retval false Still has references or not yet destroyed.
  bool release() noexcept {
    uint32_t prev = state_.fetch_sub(1, std::memory_order_acq_rel);
    assert((prev & COUNT_MASK) > 0 && "release() called with zero refcount");
    // If we were the last reference AND destroyed, claim the free.
    return prev == (DESTROYED_BIT | 1);
  }

  /// @brief Mark the object for destruction.
  /// @retval true Refcount is already 0; caller should free immediately.
  /// @retval false Outstanding references remain; last release() frees.
  bool destroy() noexcept {
    uint32_t prev = state_.fetch_or(DESTROYED_BIT, std::memory_order_acq_rel);
    assert(!(prev & DESTROYED_BIT) && "destroy() called twice");
    // If refcount was already 0, we claim the free.
    return (prev & COUNT_MASK) == 0;
  }

  /// @brief Current reference count.
  uint32_t ref_count() const noexcept {
    return state_.load(std::memory_order_relaxed) & COUNT_MASK;
  }

  /// @brief Whether destroy() has been called.
  bool is_destroyed() const noexcept {
    return (state_.load(std::memory_order_relaxed) & DESTROYED_BIT) != 0;
  }

private:
  static constexpr uint32_t DESTROYED_BIT = 1u << 31;
  static constexpr uint32_t COUNT_MASK = ~DESTROYED_BIT;

  /// @brief Packed state: bit 31 = destroyed flag, bits [0:30] = refcount.
  std::atomic<uint32_t> state_{0};
};

} // namespace rocjitsu

#endif // ROCJITSU_REFCOUNT_H_
