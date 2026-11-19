// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_LDS_BARRIER_CELL_H_
#define ROCJITSU_VM_AMDGPU_LDS_BARRIER_CELL_H_

#include <cstdint>

namespace rocjitsu {
namespace amdgpu {

/// Raw LDS barrier cell used by the gfx1250 DS barrier-arrive instructions.
///
/// RocJITsu models WIDTH=29 from CDNA5 ISA section 11.2.2. Bits [28:0] hold
/// the pending count, [31:29] the phase, and [47:32] the initial count.
/// Polling kernels test bit 29 for phase parity.
constexpr uint64_t kLdsBarrierCellBarStateMask = 0xffffffffull;
constexpr uint64_t kLdsBarrierCellPendingMask = (1ull << 29) - 1;
constexpr uint64_t kLdsBarrierCellPhaseMask = 0x7ull;
constexpr uint64_t kLdsBarrierCellInitCountMask = 0xffffull;
constexpr uint32_t kLdsBarrierCellPhaseShift = 29;
constexpr uint32_t kLdsBarrierCellPendingRollUnderBit = 1u << (kLdsBarrierCellPhaseShift - 1);
constexpr uint32_t kLdsBarrierCellInitCountShift = 32;
constexpr uint64_t kLdsBarrierCellReservedMask = 0xffff000000000000ull;

/// Return the remaining arrivals before the current phase completes.
inline uint64_t lds_barrier_cell_pending_count(uint64_t state) {
  return state & kLdsBarrierCellPendingMask;
}

/// Return the raw three-bit phase counter.
inline uint64_t lds_barrier_cell_phase(uint64_t state) {
  return (state >> kLdsBarrierCellPhaseShift) & kLdsBarrierCellPhaseMask;
}

/// Return the encoded initial pending count for each phase.
inline uint64_t lds_barrier_cell_init_count(uint64_t state) {
  return (state >> kLdsBarrierCellInitCountShift) & kLdsBarrierCellInitCountMask;
}

/// Subtract the update once from the packed 32-bit phase and pending count.
/// CDNA5 ISA section 15.15 tests bit 28 of the result for roll-under at WIDTH=29.
/// When set, keep the phase from the subtraction and reload the initial count.
inline uint64_t lds_barrier_cell_update_arrive(uint64_t state, uint32_t update = 1) {
  const uint32_t next = static_cast<uint32_t>(state) - update;
  uint64_t result = (state & ~kLdsBarrierCellBarStateMask) | next;
  if (next & kLdsBarrierCellPendingRollUnderBit)
    result = (result & ~kLdsBarrierCellPendingMask) | lds_barrier_cell_init_count(state);
  return result;
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
