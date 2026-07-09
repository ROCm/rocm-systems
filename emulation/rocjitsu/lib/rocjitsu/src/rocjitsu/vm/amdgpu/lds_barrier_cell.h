// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_LDS_BARRIER_CELL_H_
#define ROCJITSU_VM_AMDGPU_LDS_BARRIER_CELL_H_

#include <cstdint>

namespace rocjitsu {
namespace amdgpu {

/// Raw LDS barrier cell used by the gfx1250 DS barrier-arrive instructions.
///
/// The AMDGCN ISA raw async-barrier object uses [15:0] pending, [31:16]
/// phase, and [47:32] init count. This is intentionally distinct from MLIR's
/// ds_barrier_state helper; kernels that poll the raw cell directly test bit
/// 16 for phase parity.
constexpr uint64_t kLdsBarrierCellPendingMask = 0xffffull;
constexpr uint64_t kLdsBarrierCellPhaseMask = 0xffffull;
constexpr uint64_t kLdsBarrierCellInitCountMask = 0xffffull;
constexpr uint32_t kLdsBarrierCellPhaseShift = 16;
constexpr uint32_t kLdsBarrierCellInitCountShift = 32;
constexpr uint64_t kLdsBarrierCellReservedMask = 0xffff000000000000ull;

/// Return the remaining arrivals before the current phase completes.
inline uint64_t lds_barrier_cell_pending_count(uint64_t state) {
  return state & kLdsBarrierCellPendingMask;
}

/// Return the raw 16-bit phase counter.
inline uint64_t lds_barrier_cell_phase(uint64_t state) {
  return (state >> kLdsBarrierCellPhaseShift) & kLdsBarrierCellPhaseMask;
}

/// Return the encoded initial pending count for each phase.
inline uint64_t lds_barrier_cell_init_count(uint64_t state) {
  return (state >> kLdsBarrierCellInitCountShift) & kLdsBarrierCellInitCountMask;
}

/// Apply one or more barrier-arrive operations to a raw barrier cell.
///
/// A zero decrement is a no-op. A decrement larger than the remaining pending
/// count may complete multiple phases.
inline uint64_t lds_barrier_cell_update_arrive(uint64_t state, uint64_t decrement = 1) {
  uint64_t pending = lds_barrier_cell_pending_count(state);
  uint64_t phase_value = lds_barrier_cell_phase(state);
  const uint64_t initial_count = lds_barrier_cell_init_count(state);

  if (decrement == 0) {
    // Explicitly arriving zero waves is a no-op; it must not flip phase on an
    // already-drained cell.
    return state;
  }

  if (decrement <= pending) {
    pending -= decrement;
  } else {
    decrement -= pending + 1;
    phase_value = (phase_value - 1) & kLdsBarrierCellPhaseMask;

    const uint64_t phase_period = initial_count + 1;
    const uint64_t full_periods = decrement / phase_period;
    phase_value = (phase_value - full_periods) & kLdsBarrierCellPhaseMask;

    const uint64_t remainder = decrement % phase_period;
    pending = initial_count - remainder;
  }

  return (state & kLdsBarrierCellReservedMask) | (initial_count << kLdsBarrierCellInitCountShift) |
         (phase_value << kLdsBarrierCellPhaseShift) | pending;
}

/// Initialize a raw barrier cell for the requested arrivals per phase.
///
/// The encoding stores `arrivals_per_phase - 1`, so requests for zero or one
/// arrival both encode as zero pending arrivals in the initial phase.
inline uint64_t lds_barrier_cell_init_state(uint32_t arrivals_per_phase) {
  const uint64_t count = arrivals_per_phase == 0 ? 0 : arrivals_per_phase - 1;
  return ((count & kLdsBarrierCellInitCountMask) << kLdsBarrierCellInitCountShift) |
         (count & kLdsBarrierCellPendingMask);
}

/// Return the low bit of the raw phase counter used by polling kernels.
inline bool lds_barrier_cell_phase_parity(uint64_t state) {
  return (lds_barrier_cell_phase(state) & 1ull) != 0;
}

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_LDS_BARRIER_CELL_H_
