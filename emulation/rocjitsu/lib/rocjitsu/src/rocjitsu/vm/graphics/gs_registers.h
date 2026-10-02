// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <mutex>

namespace rocjitsu::amdgpu {

/// RDNA3/3.5 NGG registers, retained across draws and submissions of one queue.
/// The first eight registers are 32 bits; the remaining eight are 64 bits.
class GsRegisters {
public:
  uint64_t modify(uint32_t index, uint32_t operand, bool subtract) {
    assert(index < values_.size());
    std::lock_guard lock(mutex_);
    const uint64_t previous = values_[index];
    const uint64_t value = subtract ? previous - operand : previous + operand;
    values_[index] = index < 8 ? uint32_t(value) : value;
    return previous;
  }

  std::array<uint64_t, 2> streamout_stats(uint32_t stream) const {
    assert(stream < 4);
    std::lock_guard lock(mutex_);
    return {values_[8 + 2 * stream], values_[9 + 2 * stream]};
  }

private:
  mutable std::mutex mutex_;
  std::array<uint64_t, 16> values_{};
};

} // namespace rocjitsu::amdgpu
