// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file tensor_dma_issue_observation.h
/// @brief The LDS bytes of one tensor DMA instruction, reported when it issues.

#pragma once

#include <cstdint>
#include <span>

namespace rocjitsu::amdgpu {

/// @brief One contiguous run of LDS bytes touched by a tensor DMA transfer.
struct TensorDmaLdsRange {
  uint32_t address = 0; ///< Absolute CU LDS byte address.
  uint32_t bytes = 0;

  bool operator==(const TensorDmaLdsRange &) const = default;
};

/// @brief A tensor DMA instruction as it issues, before any of its data moves.
///
/// @details Every tensor DMA instruction that issues is reported once, before
/// its transfer starts and whether or not the transfer later waits for memory.
/// That includes an instruction that moves no data, such as a store whose tile
/// lies wholly outside the tensor, which still completes on TENSORcnt, and a
/// NULL descriptor (`count = 0`), which the emulator also counts on TENSORcnt.
/// Counting a NULL descriptor is an emulator assumption: hardware behavior for
/// that encoding is not established. The LDS ranges are the bytes the transfer
/// will write (loads) or read (stores), in element order, with adjacent
/// elements merged. A load writes every element of its tile, zero-filling those
/// outside the tensor; a store reads only in-bounds elements. Padding bytes are
/// excluded. A NULL descriptor reports no ranges; its descriptor is not
/// validated, so its element_size_bytes need not describe a transfer.
///
/// The ranges borrow execution-owned storage and are valid only for the
/// duration of the callback.
struct TensorDmaIssueObservation {
  bool is_load = true; ///< True for a global-to-LDS load, false for an LDS-to-global store.
  uint32_t element_size_bytes = 0;                 ///< Descriptor element size.
  std::span<const TensorDmaLdsRange> lds_ranges{}; ///< LDS bytes the transfer will touch.
};

} // namespace rocjitsu::amdgpu
