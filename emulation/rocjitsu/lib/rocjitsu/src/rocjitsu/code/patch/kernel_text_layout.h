// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "rocjitsu/code/rj_code.h"

namespace rocjitsu {

class BasicBlock;
class Instruction;
struct KdTranslation;

/// @brief Relocated placement of one source CFG block in one emitted kernel.
///
/// @details A source block can be emitted more than once when multiple kernel
/// entries reach shared code. The first relocation implementation rejects shared
/// blocks, but this struct is still kernel-local so later duplication can reuse
/// the same fixup model.
struct BlockPlacement {
  BasicBlock *block = nullptr; ///< Source CFG block.
  uint64_t source_start = 0;   ///< Original .text-relative block start.
  uint64_t source_end = 0;     ///< Original .text-relative block end.
  uint64_t target_start = 0;   ///< New .text-relative block start.
  uint64_t target_end = 0;     ///< New .text-relative block end.
};

/// @brief Pending direct PC-relative branch fixup in one relocated kernel.
///
/// @details Source offsets are resolved through the kernel-local placement map
/// after all reachable blocks and local cave bytes have final target offsets.
struct BranchFixup {
  const Instruction *inst = nullptr; ///< Decoded source branch instruction.
  uint64_t source_inst_offset = 0;   ///< Original .text offset of the branch.
  uint64_t source_target_offset = 0; ///< Original .text offset of the branch target.
  uint64_t target_inst_offset = 0;   ///< New .text offset of the branch instruction.
};

/// @brief Physical output layout for one translated kernel.
///
/// @details Blocks are emitted in original .text order. This is intentional: it
/// preserves every CFG fallthrough edge as physical adjacency, so DBT only
/// patches explicit PC-relative branch immediates. Expansion bodies are appended
/// after the emitted body as a local cave.
struct KernelTextLayout {
  KdTranslation *translation = nullptr;   ///< Descriptor plan for this kernel.
  uint64_t source_entry = 0;              ///< Original descriptor entry offset.
  uint64_t target_entry = 0;              ///< Final descriptor entry offset.
  uint64_t target_body_entry = 0;         ///< Relocated original entry offset.
  uint64_t body_begin = 0;                ///< First emitted body byte.
  uint64_t body_end = 0;                  ///< One-past-end of emitted body.
  uint64_t cave_begin = 0;                ///< First local cave byte.
  uint64_t cave_end = 0;                  ///< One-past-end of local cave.
  std::vector<BlockPlacement> blocks;     ///< Kernel-local block placements.
  std::vector<BranchFixup> branch_fixups; ///< Explicit branch patches.
};

void append_words(std::vector<uint8_t> &text, std::span<const uint32_t> words);

void append_nop_padding(std::vector<uint8_t> &text, uint64_t byte_count, rj_code_arch_t arch);

[[nodiscard]] uint64_t padding_for_residue(uint64_t current_offset, uint64_t target_residue,
                                           uint64_t alignment);

[[nodiscard]] std::optional<uint64_t> target_for_source_offset(const KernelTextLayout &layout,
                                                               uint64_t source_offset);

[[nodiscard]] std::optional<uint64_t> target_for_source_fallthrough(const KernelTextLayout &layout,
                                                                    std::span<const uint8_t> text,
                                                                    uint64_t source_offset,
                                                                    rj_code_arch_t arch);

} // namespace rocjitsu
