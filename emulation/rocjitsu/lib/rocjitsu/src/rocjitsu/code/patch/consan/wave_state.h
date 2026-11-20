// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

namespace rocjitsu::consan {

/// @brief Memory-backed per-wave ordering and loss state for ConSan probes.
/// @details The runtime owns the storage until dispatch completion. Counter
/// locations belong exclusively to one wave and are supplied by probes.hip,
/// independently of this helper's implementation. Preparation requests full EXEC
/// for every probe call. Participating lanes must agree on the counter locations
/// and include lane zero, the only writer. A caller may narrow EXEC to lane zero
/// for scalar publication. No value survives in an implicit register.
class DeviceWaveState {
public:
  using Counter = volatile unsigned __attribute__((address_space(1))) *;

  __device__ __attribute__((always_inline)) DeviceWaveState(Counter sequence, Counter losses,
                                                            Counter exhausted)
      : sequence_(sequence), losses_(losses), exhausted_(exhausted) {}

  __device__ __attribute__((always_inline)) void initialize(unsigned lane) const {
    if (lane == 0) {
      *sequence_ = 0;
      *losses_ = 0;
      *exhausted_ = 0;
    }
  }

  /// Return the next value, or zero when the counter cannot advance. The stored
  /// value stays saturated and exhaustion remains visible until initialization.
  __device__ __attribute__((always_inline)) unsigned advance(Counter counter, unsigned lane) const {
    const unsigned next = __builtin_amdgcn_readfirstlane(*counter) + 1;
    // Compute the zero bit using scalar arithmetic before selecting the writer.
    // Keeping carry masks out of divergent regions makes every compiled probe
    // path self-contained under the callable ABI's conservative CFG analysis.
    // overflow is one exactly when next wraps to zero. At UINT_MAX, next=0:
    // next-overflow stores UINT_MAX again, while returning zero signals exhaustion.
    const unsigned overflow = (~(next | (0u - next))) >> 31;
    if (lane == 0) {
      *counter = next - overflow;
      *exhausted_ = *exhausted_ | overflow;
    }
    return next;
  }

  __device__ __attribute__((always_inline)) unsigned next_sequence(unsigned lane) const {
    return advance(sequence_, lane);
  }

  __device__ __attribute__((always_inline)) void record_loss(unsigned lane) const {
    advance(losses_, lane);
  }

  __device__ __attribute__((always_inline)) bool has_loss() const { return *losses_ != 0; }

private:
  Counter sequence_;
  Counter losses_;
  Counter exhausted_;
};

} // namespace rocjitsu::consan
