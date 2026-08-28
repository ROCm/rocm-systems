// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file semantic/cdna3_scratch.h
/// @brief CDNA3 private-scratch spill/fill sequence emission.

#pragma once

#include "rocjitsu/code/dbt/semantic_scratch.h"

#include <cstdint>
#include <vector>

namespace rocjitsu {

/// @brief Target-specific materializer for semantic scratch preservation.
///
/// @details The architecture-neutral allocator decides which VGPRs must be
/// preserved and assigns private-segment slots. This class owns every CDNA3
/// detail needed to realize that decision: FLAT_SCRATCH opcodes, immediate
/// range constraints, and the conservative wait after each save/fill batch.
class Cdna3ScratchEmitter final {
public:
  /// CDNA3 FLAT_SCRATCH OFFSET is a signed 13-bit immediate. Semantic spill
  /// storage is appended at non-negative offsets, so only the positive half is
  /// available to this emitter.
  static constexpr uint32_t kMaxDwordOffset = 4095;

  /// @brief Allocation limits consumed by SemanticScratchAllocator.
  [[nodiscard]] static constexpr SemanticScratchPolicy allocation_policy() {
    return SemanticScratchPolicy{.max_vgprs = 256, .max_spill_dword_offset = kMaxDwordOffset};
  }

  /// @brief Whether every dword in @p range has an encodable CDNA3 offset.
  [[nodiscard]] static bool can_address(const SemanticSpillRange &range);

  /// @brief Append one FLAT_SCRATCH dword store without a wait.
  static void append_store_dword(std::vector<uint32_t> &words, uint8_t vgpr, uint32_t byte_offset);

  /// @brief Append one FLAT_SCRATCH dword load without a wait.
  static void append_load_dword(std::vector<uint32_t> &words, uint8_t vgpr, uint32_t byte_offset);

  /// @brief Append a conservative wait for all outstanding CDNA3 counters.
  static void append_wait(std::vector<uint32_t> &words);

  /// @brief Save a spill-backed lease and wait before its registers are reused.
  /// @returns false if the lease is not a supported, encodable VGPR range.
  [[nodiscard]] static bool append_save(std::vector<uint32_t> &words,
                                        const SemanticScratchLease &lease);

  /// @brief Restore a spill-backed lease and wait before returning to guest code.
  /// @returns false if the lease is not a supported, encodable VGPR range.
  [[nodiscard]] static bool append_restore(std::vector<uint32_t> &words,
                                           const SemanticScratchLease &lease);
};

} // namespace rocjitsu
