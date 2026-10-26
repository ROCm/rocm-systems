// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>

namespace util {

/// @brief Atomic isolated from neighboring objects on 64-byte cache lines.
/// @details Alignment and padding also apply in arrays and make_shared allocations.
/// Retains stricter alignment requirements of the underlying atomic type.
template <typename T> struct alignas(64) alignas(std::atomic<T>) AlignedAtomic : std::atomic<T> {
  using std::atomic<T>::atomic;
  using std::atomic<T>::operator=;
};

} // namespace util
