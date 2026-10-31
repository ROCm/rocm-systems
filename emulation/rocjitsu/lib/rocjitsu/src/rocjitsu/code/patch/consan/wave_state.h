// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

namespace rocjitsu::consan {

/// @brief Memory-backed per-wave ordering and loss state for ConSan probes.
/// @details The runtime owns the storage until dispatch completion. Counter
/// locations belong exclusively to one wave and are supplied by probes.hip,
/// independently of this helper's implementation. Preparation requests full EXEC
/// for every probe call; only lane zero writes the counters. No value survives
/// in an implicit register.
class DeviceWaveState {
public:
  using Counter = volatile unsigned __attribute__((address_space(1))) *;

  __device__ __attribute__((always_inline)) DeviceWaveState(Counter sequence, Counter losses)
      : sequence_(sequence), losses_(losses) {}

  __device__ __attribute__((always_inline)) void initialize(unsigned lane) const {
    if (lane == 0) {
      *sequence_ = 0;
      *losses_ = 0;
    }
  }

  __device__ __attribute__((always_inline)) unsigned next_sequence(unsigned lane) const {
    const unsigned next = __builtin_amdgcn_readfirstlane(*sequence_) + 1;
    if (lane == 0)
      *sequence_ = next;
    return next;
  }

  __device__ __attribute__((always_inline)) void record_loss(unsigned lane) const {
    if (lane == 0)
      *losses_ = *losses_ + 1;
  }

  __device__ __attribute__((always_inline)) bool has_loss() const { return *losses_ != 0; }

private:
  Counter sequence_;
  Counter losses_;
};

} // namespace rocjitsu::consan
