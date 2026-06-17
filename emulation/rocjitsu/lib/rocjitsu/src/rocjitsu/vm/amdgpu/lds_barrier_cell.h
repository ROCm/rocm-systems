// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_LDS_BARRIER_CELL_H_
#define ROCJITSU_VM_AMDGPU_LDS_BARRIER_CELL_H_

#include <cstdint>

namespace rocjitsu {
namespace amdgpu {

namespace lds_barrier_cell {

// Raw LDS barrier cell used by the gfx1250 DS barrier-arrive instructions.
// The AMDGCN ISA raw async-barrier object uses [15:0] pending, [31:16]
// phase, and [47:32] init count. This is intentionally distinct from MLIR's
// ds_barrier_state helper; kernels that poll the raw cell directly test bit
// 16 for phase parity.
constexpr uint64_t kPendingMask = 0xffffull;
constexpr uint64_t kPhaseMask = 0xffffull;
constexpr uint64_t kInitCountMask = 0xffffull;
constexpr uint32_t kPhaseShift = 16;
constexpr uint32_t kInitCountShift = 32;
constexpr uint64_t kReservedMask = 0xffff000000000000ull;

inline uint64_t pending_count(uint64_t state) { return state & kPendingMask; }

inline uint64_t phase(uint64_t state) { return (state >> kPhaseShift) & kPhaseMask; }

inline uint64_t init_count(uint64_t state) { return (state >> kInitCountShift) & kInitCountMask; }

inline uint64_t update_arrive(uint64_t state, uint64_t decrement = 1) {
  uint64_t pending = pending_count(state);
  uint64_t phase_value = phase(state);
  const uint64_t initial_count = init_count(state);

  if (decrement == 0) {
    // Explicitly arriving zero waves is a no-op; it must not flip phase on an
    // already-drained cell.
    return state;
  }

  if (decrement <= pending) {
    pending -= decrement;
  } else {
    decrement -= pending + 1;
    phase_value = (phase_value - 1) & kPhaseMask;

    const uint64_t phase_period = initial_count + 1;
    const uint64_t full_periods = decrement / phase_period;
    phase_value = (phase_value - full_periods) & kPhaseMask;

    const uint64_t remainder = decrement % phase_period;
    pending = initial_count - remainder;
  }

  return (state & kReservedMask) | (initial_count << kInitCountShift) |
         (phase_value << kPhaseShift) | pending;
}

inline uint64_t init_state(uint32_t arrivals_per_phase) {
  const uint64_t count = arrivals_per_phase == 0 ? 0 : arrivals_per_phase - 1;
  return ((count & kInitCountMask) << kInitCountShift) | (count & kPendingMask);
}

inline bool phase_parity(uint64_t state) { return (phase(state) & 1ull) != 0; }

} // namespace lds_barrier_cell

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_LDS_BARRIER_CELL_H_
