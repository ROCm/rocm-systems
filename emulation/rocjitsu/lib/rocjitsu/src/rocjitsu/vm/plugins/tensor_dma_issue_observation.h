// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file tensor_dma_issue_observation.h
/// @brief The LDS bytes of one tensor DMA instruction, reported when it issues.

#pragma once

#include "rocjitsu/vm/amdgpu/lds_byte_range.h"

#include <span>

namespace rocjitsu::amdgpu {

/// @brief A tensor DMA instruction as it issues, before its global requests
/// and LDS writes.
///
/// @details Every tensor DMA instruction that issues is reported once, whether
/// or not the transfer later waits for memory. That includes an instruction
/// that moves no data: a NOP tile (`tile_dim0 = 0`), a gather with no valid
/// indices or a store whose tile lies wholly outside the tensor, each of which
/// still completes on TENSORcnt, and a NULL descriptor (`count = 0`), which the
/// emulator also counts on TENSORcnt. A store has already sampled its LDS
/// source bytes when it is reported; no global request has been made and a
/// load has written no LDS.
/// Counting a NULL descriptor is an emulator assumption: hardware behavior for
/// that encoding is not established.
///
/// The LDS ranges are the bytes the transfer will write (loads) or read
/// (stores), in element order, with adjacent elements merged. A load writes
/// every element of its tile, zero-filling those outside the tensor; a store
/// reads only in-bounds elements. Padding bytes and bytes outside the wave's
/// LDS allocation, which are not transferred, are excluded, so a range can end
/// mid-element: an element that straddles the end of the allocation contributes
/// only its leading bytes. Consumers must treat the ranges as byte ranges, not
/// as whole elements. An instruction that moves no data reports no ranges.
///
/// Empty ranges describe only the transfer's LDS payload, not its global
/// accesses. When the whole tile lies past the wave's LDS allocation, a load
/// still reads its in-bounds tensor elements from global memory and can fault
/// there, and a store still writes zeros to its global destination.
///
/// The ranges borrow execution-owned storage and are valid only for the
/// duration of the callback.
struct TensorDmaIssueObservation {
  bool is_load = true; ///< True for a global-to-LDS load, false for an LDS-to-global store.
  std::span<const LdsByteRange> lds_ranges{}; ///< LDS bytes the transfer will touch.
};

} // namespace rocjitsu::amdgpu
