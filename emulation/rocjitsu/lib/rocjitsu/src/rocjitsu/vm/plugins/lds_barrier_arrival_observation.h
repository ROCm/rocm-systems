// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file lds_barrier_arrival_observation.h
/// @brief One arrival at an LDS barrier cell.

#pragma once

#include <cstdint>

namespace rocjitsu::amdgpu {

/// @brief One arrival at an LDS barrier cell (CDNA5 ISA 11.2.2).
///
/// @details A barrier cell is a 64-bit LDS word holding a pending count, a
/// phase and an initial count (CDNA5 ISA 11.2.2, Table 70). An arrival
/// subtracts its update value from the pending count; when the count rolls
/// under, the phase changes and the count is reloaded.
///
/// The DS arrive instructions arrive once per active lane, each lane at its
/// own address, as LDS atomics do (CDNA5 ISA 11.2.1). The ISA does not order
/// the lanes; the simulator arrives, and reports, in lane order. An update
/// value of zero leaves the cell unchanged and is reported with `old_state ==
/// new_state`. A tensor DMA instruction whose descriptor requests it arrives
/// once, after the transfer has moved its data (CDNA5 ISA 10.11.3). Each
/// arrival is reported after the cell has been updated.
struct LdsBarrierArrivalObservation {
  /// What arrived, which determines the operations the arrival is ordered
  /// after.
  enum class Source : uint8_t {
    /// ds_atomic_barrier_arrive_rtn_b64: an ordinary LDS operation, in order
    /// with the wave's other LDS operations (CDNA5 ISA 11.2.2).
    DsAtomic,
    /// ds_atomic_async_barrier_arrive_b64: in order with the wave's async
    /// loads to LDS (CDNA5 ISA 11.2.2, 10.8).
    AsyncDsAtomic,
    /// The arrival a tensor DMA descriptor requests, sent when the transfer
    /// completes (CDNA5 ISA 10.11.3).
    TensorDma,
  };

  Source source = Source::DsAtomic;
  uint32_t cell_address = 0;    ///< Absolute CU LDS byte address of the 64-bit cell.
  uint64_t old_state = 0;       ///< Cell contents before the arrival.
  uint64_t new_state = 0;       ///< Cell contents after the arrival.
  bool phase_completed = false; ///< The pending count rolled under, changing the phase.

  bool operator==(const LdsBarrierArrivalObservation &) const = default;
};

} // namespace rocjitsu::amdgpu
