// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file tensor_dma_memory_access_observation.h
/// @brief Completed global-memory accesses performed by one tensor DMA instruction.

#pragma once

#include <cstdint>
#include <span>
#include <string_view>

namespace rocjitsu::amdgpu {

/// @brief The in-bounds global requests attempted by one tensor DMA instruction.
///
/// @details The address span contains only in-bounds global element base
/// addresses, in the exact order in which the implementation attempted them.
/// Duplicates are preserved. The callback is emitted only after the instruction
/// and any descriptor-requested atomic-barrier arrival return normally. Memory
/// access outcomes are intentionally not filtered, matching FFM's observation
/// boundary.
///
/// The span borrows execution-owned storage and is valid only for the duration
/// of the callback. A consumer that keeps an observation must copy it.
struct TensorDmaMemoryAccessObservation {
  /// @brief Instruction mnemonic; points at static storage.
  std::string_view mnemonic;
  uint64_t pc = 0;              ///< PC of the issuing wavefront.
  uint32_t compute_unit_id = 0; ///< Component id of the issuing compute unit.
  uint32_t dispatch_id = 0;
  uint32_t queue_id = 0;
  uint32_t workgroup_id = 0;
  uint32_t wavefront_id = 0; ///< Wavefront slot within the compute unit.
  uint32_t process_id = 0;   ///< VMID of the global addresses.
  uint32_t element_size_bytes = 0;
  bool is_load = true; ///< True for global-to-LDS, false for LDS-to-global.
  std::span<const uint64_t> addresses;
};

} // namespace rocjitsu::amdgpu
