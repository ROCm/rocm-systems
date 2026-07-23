// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "rocjitsu/analysis/indirect_branch_discovery.h"
#include "rocjitsu/code/rj_code.h"

namespace rocjitsu {

/// @brief Reserved dwords for a canonical recovered indirect transfer.
inline constexpr size_t kMaxRecoveredIndirectTransferWords = 6;

/// @brief Reserved dwords for the largest direct long-branch sequence.
inline constexpr size_t kMaxDirectBranchTransferWords = 7;

class BasicBlock;
class Instruction;

/// @brief Relocated placement of one source CFG block in one emitted kernel.
///
/// @details A source block can be emitted more than once when multiple kernel
/// entries reach shared code. Each KernelTextLayout is kernel-local, so the same
/// source range can have a different target range in another kernel's layout.
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
/// after all reachable blocks have final target offsets.
struct BranchFixup {
  const Instruction *inst = nullptr; ///< Decoded source branch instruction.
  uint64_t source_inst_offset = 0;   ///< Original .text offset of the branch.
  uint64_t source_target_offset = 0; ///< Original .text offset of the branch target.
  uint64_t target_inst_offset = 0;   ///< New .text offset of the branch patch window.
  uint64_t target_window_bytes = 0;  ///< Reserved bytes available for final branch encoding.
};

/// @brief Pending recovered indirect branch/call window in one relocated kernel.
///
/// @details Static recovery gives DBT one concrete source target for an indirect
/// `s_setpc_b64` or `s_swappc_b64` consumer. The translated block reserves a fixed
/// worst-case window at the consumer site; after all blocks have final target
/// offsets, the patch layer rewrites that window to either a direct control
/// transfer or a canonical long getpc/add/setpc-or-swappc sequence.
struct RecoveredIndirectFixup {
  uint64_t source_call_offset = 0;   ///< Original .text offset of setpc/swappc.
  uint64_t source_target_offset = 0; ///< Recovered original .text target offset.
  uint64_t target_window_offset = 0; ///< New .text offset of the reserved window.
  uint16_t target_sreg = 0;          ///< Low SGPR pair used for the rebuilt target PC.
  uint16_t return_sreg = 0;          ///< Low SGPR pair receiving return PC for calls.
  bool is_call = false;              ///< True for swappc-like calls, false for setpc jumps.
};

/// @brief Stable failure category returned across the text-layout boundary.
///
/// @details Diagnostic classification must not depend on human-readable wording.
/// The caller still chooses its feature-specific diagnostic kind for ordinary
/// invalid-layout failures, while resource limits map uniformly to the DBT
/// resource-limit diagnostic.
enum class TextLayoutFailureCategory {
  None,
  InvalidLayout,
  ResourceLimit,
};

/// @brief Stable machine-readable reason for a relocation failure.
enum class TextLayoutFailureReason {
  None,
  BranchOutOfRange,
};

/// @brief Result of applying a relocation fixup.
struct TextRelocationResult {
  bool ok = true;
  TextLayoutFailureCategory failure = TextLayoutFailureCategory::None;
  TextLayoutFailureReason reason = TextLayoutFailureReason::None;
  uint64_t source_offset = 0;
  std::string message;
};

/// @brief Descriptor-neutral entry layout requested by DBT or DBI.
struct KernelEntryLayoutPlan {
  /// @brief True when hardware may enter again at source entry plus 256 bytes.
  bool has_kernarg_preload_firmware_skip = false;

  /// @brief Source `.text` offset of the compatible-firmware preload entry.
  uint64_t kernarg_preload_firmware_entry_text_offset = 0;

  /// @brief Target instructions that must execute before the relocated body.
  std::vector<uint32_t> prologue_words;
};

/// @brief Minimal placement facts needed to emit a skipped-kernel no-op stub.
struct SkippedKernelLayoutPlan {
  /// @brief Original entry offset whose 256-byte residue must be preserved.
  uint64_t source_entry = 0;

  /// @brief Emit `s_endpgm` stubs at both legal preload firmware entry addresses.
  bool has_kernarg_preload_firmware_skip = false;
};

/// @brief Result of appending descriptor-visible kernel text.
///
/// @details The patch layer owns final .text placement for relocated kernels.
/// DBT still owns diagnostics and descriptor bookkeeping, so the result reports
/// enough offsets for those callers without exposing launch-stub layout details.
struct KernelTextAppendResult {
  bool ok = true;
  TextLayoutFailureCategory failure = TextLayoutFailureCategory::None;
  uint64_t source_offset = 0;
  uint64_t target_delta = 0;
  uint64_t target_entry = 0;
  uint64_t target_body_entry = 0;
  std::string message;
};

/// @brief Physical output layout for one translated kernel.
///
/// @details Blocks are emitted in original .text order. This is intentional: it
/// preserves every CFG fallthrough edge as physical adjacency, so DBT only
/// patches explicit PC-relative branch immediates and reserved recovered-indirect
/// windows after all translated block starts are known.
struct KernelTextLayout {
  KernelEntryLayoutPlan entry_plan; ///< Entry code requested by the producer.
  uint64_t source_entry = 0;        ///< Original descriptor entry offset.
  uint64_t target_entry = 0;        ///< Final descriptor entry offset.
  uint64_t target_body_entry = 0;   ///< Relocated original entry offset.
  uint64_t body_begin = 0;          ///< First emitted body byte.
  uint64_t body_end = 0;            ///< One-past-end of emitted body.
  /// SGPR pair reserved by the descriptor for out-of-range direct branches.
  ///
  /// Direct branches normally patch in place as one SOPP/SOPK instruction. When
  /// semantic expansions push a branch target outside the signed 16-bit branch
  /// range, DBT rewrites the reserved branch window into a canonical
  /// getpc/add/setpc sequence. The SGPR pair named here is descriptor-grown
  /// scratch and is never a guest live register.
  std::optional<uint16_t> long_branch_sgpr;
  /// Non-executed SOPP branch slots available for SGPR-free long direct branches.
  ///
  /// Full-SGPR kernels cannot build an arbitrary target PC in a scratch scalar
  /// pair. The translator can instead insert skipped branch-island pools while
  /// emitting the body. Each recorded offset names one private `s_branch` slot
  /// that patch_direct_branch_fixups() may dedicate to a single out-of-range
  /// direct branch chain.
  std::vector<uint64_t> branch_island_slots;
  std::vector<BlockPlacement> blocks;     ///< Kernel-local block placements.
  std::vector<BranchFixup> branch_fixups; ///< Explicit branch patches.
  std::vector<RecoveredIndirectFixup> recovered_indirect_fixups;
  /// Source-side builders that must be rewritten because one indirect consumer
  /// has multiple recovered targets and therefore cannot be replaced by one
  /// direct transfer window.
  std::vector<IndirectCallFixup> recovered_builder_fixups;
};

void append_words(std::vector<uint8_t> &text, std::span<const uint32_t> words);

void append_nop_padding(std::vector<uint8_t> &text, uint64_t byte_count, rj_code_arch_t arch);

[[nodiscard]] uint64_t padding_for_residue(uint64_t current_offset, uint64_t target_residue,
                                           uint64_t alignment);

[[nodiscard]] std::optional<uint64_t> target_for_source_offset(const KernelTextLayout &layout,
                                                               uint64_t source_offset);

/// @brief Return true if @p plan's preload launch stubs fit the ABI window.
[[nodiscard]] bool kernarg_preload_launch_window_fits(const KernelEntryLayoutPlan &plan);

/// @brief Choose the fixed patch-window size for a direct branch.
///
/// @details This centralizes patch-layer knowledge of long direct-branch
/// windows and conditional branch-island windows. DBT records the source branch
/// and reserves exactly the number of bytes the patch layer may later rewrite.
[[nodiscard]] uint64_t direct_branch_patch_window_bytes(const Instruction &inst,
                                                        uint64_t source_inst_offset,
                                                        uint64_t source_target_offset,
                                                        bool can_use_long_direct_branches);

/// @brief First body offset at which an SGPR-free branch-island pool should appear.
[[nodiscard]] uint64_t first_direct_branch_island_pool_offset();

/// @brief Next body offset at which an SGPR-free branch-island pool should appear.
[[nodiscard]] uint64_t next_direct_branch_island_pool_offset(uint64_t current_body_size);

/// @brief Append one skipped branch-island pool to @p kernel_text and record its slots.
void append_direct_branch_island_pool(std::vector<uint8_t> &kernel_text, KernelTextLayout &layout,
                                      rj_code_arch_t arch);

/// @brief Add @p delta to body-relative target offsets in @p layout.
///
/// @details Descriptor-visible entry stubs are not source-block placements, so
/// target_entry is deliberately left untouched and set by the caller after the
/// final hardware entry location is known.
void rebase_kernel_text_layout(KernelTextLayout &layout, uint64_t delta);

/// @brief Append a target-ISA `s_endpgm` body for a skipped kernel.
///
/// @details The neutral plan contains only placement facts needed by the patch
/// layer. DBT and DBI retain ownership of descriptor policy and diagnostics.
[[nodiscard]] KernelTextAppendResult append_skipped_kernel_stub(std::vector<uint8_t> &text,
                                                                const SkippedKernelLayoutPlan &plan,
                                                                rj_code_arch_t arch);

/// @brief Append one relocated kernel body plus any descriptor-visible entry stubs.
///
/// @details @p layout must contain body-relative block/fixup offsets. This
/// function chooses final output padding, rebases the layout into .text
/// coordinates, appends @p kernel_text, and emits kernarg-preload or descriptor
/// prologue launch stubs when the descriptor requires them.
[[nodiscard]] KernelTextAppendResult
append_relocated_kernel_text(std::vector<uint8_t> &translated_text, KernelTextLayout &layout,
                             std::span<const uint8_t> kernel_text, rj_code_arch_t arch);

/// @brief Patch all direct PC-relative branches recorded in @p layout.
[[nodiscard]] TextRelocationResult patch_direct_branch_fixups(std::vector<uint8_t> &text,
                                                              const KernelTextLayout &layout,
                                                              rj_code_arch_t arch);

/// @brief Patch all recovered indirect branch/call windows recorded in @p layout.
[[nodiscard]] TextRelocationResult patch_recovered_indirect_fixups(std::vector<uint8_t> &text,
                                                                   const KernelTextLayout &layout,
                                                                   rj_code_arch_t arch);

/// @brief Patch recovered source-side PC builders for multi-target consumers.
[[nodiscard]] TextRelocationResult patch_recovered_builder_fixups(std::vector<uint8_t> &text,
                                                                  const KernelTextLayout &layout,
                                                                  rj_code_arch_t arch);

} // namespace rocjitsu
