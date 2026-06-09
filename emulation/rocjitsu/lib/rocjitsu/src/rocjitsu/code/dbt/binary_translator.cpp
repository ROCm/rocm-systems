// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/dbt/binary_translator.h"

#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/dbt/generated/encoding_cdna4_to_cdna3.h"
#include "rocjitsu/code/dbt/generated/encoding_cdna4_to_rdna3.h"
#include "rocjitsu/code/dbt/generated/encoding_cdna4_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna4_to_cdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna4_to_rdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna4_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_types.h"
#include "rocjitsu/code/dbt/kernel_descriptor_translator.h"
#include "rocjitsu/code/dbt/semantic_translator.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rocjitsu {

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

struct IndirectCallFixup {
  uint64_t source_getpc_offset = 0;
  uint64_t source_add_lo_offset = 0;
  uint64_t source_add_hi_offset = 0;
  uint64_t source_call_offset = 0;
  uint64_t source_target_offset = 0;
  uint16_t source_call_sreg = 0;
  uint64_t target_getpc_offset = 0;
  uint64_t target_add_lo_offset = 0;
  uint64_t target_add_hi_offset = 0;
  int32_t add_lo_literal_adjust = 0;
  bool add_hi_has_literal = true;
};

struct DirectCallReturn {
  uint64_t source_call_offset = 0;
  uint16_t return_sreg = 0;
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
  std::vector<IndirectCallFixup> indirect_call_fixups;
  std::vector<DirectCallReturn> direct_call_returns;
};

namespace {

inline constexpr uint64_t kKernargPreloadSkipBytes = 256;

EncodingTranslateFn select_encoding_translator(rj_code_arch_t guest, rj_code_arch_t host) {
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_RDNA4)
    return cdna4_to_rdna4::translate_encoding_cdna4_to_rdna4;
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_CDNA3)
    return cdna4_to_cdna3::translate_encoding_cdna4_to_cdna3;
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_RDNA3)
    return cdna4_to_rdna3::translate_encoding_cdna4_to_rdna3;
  return nullptr;
}

LegalizationLookupFn select_legalization(rj_code_arch_t guest, rj_code_arch_t host) {
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_RDNA4) {
    return [](uint16_t enc_id, uint16_t opcode) -> const InstructionLegalization * {
      return lookup(kLegalization_cdna4_to_rdna4, enc_id, opcode);
    };
  }
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_CDNA3) {
    return [](uint16_t enc_id, uint16_t opcode) -> const InstructionLegalization * {
      return lookup(kLegalization_cdna4_to_cdna3, enc_id, opcode);
    };
  }
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_RDNA3) {
    return [](uint16_t enc_id, uint16_t opcode) -> const InstructionLegalization * {
      return lookup(kLegalization_cdna4_to_rdna3, enc_id, opcode);
    };
  }
  return nullptr;
}

[[nodiscard]] BasicBlock *block_for_offset(const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                                           uint64_t offset) {
  for (const auto &block : blocks) {
    if (block && block->start_offset() <= offset && offset < block->end_offset())
      return block.get();
  }
  return nullptr;
}

[[nodiscard]] bool compute_sopp_branch_offset(uint64_t branch_pc, uint64_t target,
                                              int16_t &offset_dwords) {
  // SOPP branches encode a signed dword offset from the next instruction. Keep
  // the range check shared so both cave entry and return branches fail closed.
  constexpr int64_t kBranchPcBiasBytes = static_cast<int64_t>(sizeof(uint32_t));
  constexpr uint64_t kMaxSignedTarget = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
  constexpr uint64_t kMaxSignedBranchPc =
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max() - kBranchPcBiasBytes);
  // The PCs are unsigned until this check passes. Compare against the casted
  // signed int64_t limits so the later signed conversion, and branch_pc + 4,
  // cannot overflow.
  if (branch_pc > kMaxSignedBranchPc || target > kMaxSignedTarget)
    return false;

  const int64_t delta_bytes = static_cast<int64_t>(target) - (static_cast<int64_t>(branch_pc) + 4);
  if (delta_bytes % static_cast<int64_t>(sizeof(uint32_t)) != 0)
    return false;

  const int64_t delta_dwords = delta_bytes / static_cast<int64_t>(sizeof(uint32_t));
  if (delta_dwords < std::numeric_limits<int16_t>::min() ||
      delta_dwords > std::numeric_limits<int16_t>::max())
    return false;

  offset_dwords = static_cast<int16_t>(delta_dwords);
  return true;
}

[[nodiscard]] std::vector<uint32_t> raw_words_for_inst(const Instruction &inst) {
  const uint32_t *raw = inst.raw_encoding();
  if (!raw)
    return {};
  return {raw, raw + inst.size() / sizeof(uint32_t)};
}

void append_words(std::vector<uint8_t> &text, std::span<const uint32_t> words) {
  const auto *bytes = reinterpret_cast<const uint8_t *>(words.data());
  text.insert(text.end(), bytes, bytes + words.size() * sizeof(uint32_t));
}

void append_nop_padding(std::vector<uint8_t> &text, uint64_t byte_count, rj_code_arch_t arch) {
  assert(byte_count % sizeof(uint32_t) == 0 && "padding must be word-aligned");
  for (uint64_t off = 0; off < byte_count; off += sizeof(uint32_t)) {
    const uint32_t nop = build_s_nop(0, arch);
    append_words(text, std::span<const uint32_t>(&nop, 1));
  }
}

[[nodiscard]] uint64_t padding_for_residue(uint64_t current_offset, uint64_t target_residue,
                                           uint64_t alignment) {
  const uint64_t current_residue = current_offset % alignment;
  return (target_residue + alignment - current_residue) % alignment;
}

[[nodiscard]] uint64_t preserve_entry_skip_window_offset(const KernelTextLayout &layout,
                                                         const BasicBlock &block, uint64_t cursor) {
  if (block.start_offset() < layout.source_entry)
    return cursor;

  // Kernels with kernarg preloading have two valid hardware entry paths:
  // incompatible firmware executes the descriptor entry and branches around the
  // preload compatibility prologue, while compatible firmware adds 256 bytes to
  // KERNEL_CODE_ENTRY_BYTE_OFFSET and starts at the real body directly. Local
  // body compaction must therefore keep the first 256 bytes after the descriptor
  // entry stable. Otherwise compatible firmware lands in the middle of compacted
  // translated code with the preloaded SGPR ABI assumed but not honored.
  const uint64_t source_delta = block.start_offset() - layout.source_entry;
  if (source_delta > kKernargPreloadSkipBytes)
    return cursor;

  return std::max(cursor, layout.body_begin + source_delta);
}

[[nodiscard]] std::optional<uint64_t> target_for_source_offset(const KernelTextLayout &layout,
                                                               uint64_t source_offset) {
  for (const BlockPlacement &placement : layout.blocks) {
    if (source_offset >= placement.source_start && source_offset < placement.source_end)
      return placement.target_start + (source_offset - placement.source_start);
  }
  return std::nullopt;
}

[[nodiscard]] uint32_t text_word_at(std::span<const uint8_t> text, uint64_t offset) {
  uint32_t word = 0;
  if (offset + sizeof(word) <= text.size())
    std::memcpy(&word, text.data() + offset, sizeof(word));
  return word;
}

[[nodiscard]] bool is_nop_padding_gap(std::span<const uint8_t> text, uint64_t begin, uint64_t end,
                                      rj_code_arch_t arch) {
  if (begin > end || end > text.size() || (begin % sizeof(uint32_t)) != 0 ||
      (end % sizeof(uint32_t)) != 0)
    return false;

  const uint32_t nop = build_s_nop(0, arch);
  for (uint64_t off = begin; off < end; off += sizeof(uint32_t)) {
    if (text_word_at(text, off) != nop)
      return false;
  }
  return true;
}

[[nodiscard]] std::optional<uint64_t> target_for_source_fallthrough(const KernelTextLayout &layout,
                                                                    std::span<const uint8_t> text,
                                                                    uint64_t source_offset,
                                                                    rj_code_arch_t arch) {
  if (auto target = target_for_source_offset(layout, source_offset))
    return target;

  // Local cave expansions return to the source instruction's next PC. When the
  // compiler placed only s_nop padding between two reachable blocks, that next
  // PC is not present in the compact relocated body. Skipping that source-only
  // padding here keeps the body layout compact while preserving the observable
  // fallthrough target: executing zero or more nops reaches the next block.
  for (const BlockPlacement &placement : layout.blocks) {
    if (source_offset >= placement.source_start)
      continue;
    if (is_nop_padding_gap(text, source_offset, placement.source_start, arch))
      return placement.target_start;
    break;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<uint16_t> s_getpc_sdst(const Instruction &inst, uint32_t word) {
  if (inst.size() != sizeof(uint32_t))
    return std::nullopt;
  if ((word >> 23) != kSop1EncodingPrefix)
    return std::nullopt;
  // CDNA SOP1 opcode: s_getpc_b64 = 0x1c.
  if (((word >> 8) & 0xffu) != 0x1cu)
    return std::nullopt;
  return static_cast<uint16_t>((word >> 16) & 0x7fu);
}

[[nodiscard]] bool sop2_literal_to_sreg(const Instruction &inst, uint32_t word,
                                        uint32_t literal_word, uint32_t opcode, uint16_t sdst,
                                        uint16_t ssrc0, uint32_t &literal) {
  if (inst.size() != 2 * sizeof(uint32_t))
    return false;
  if ((word >> 30) != kSop2EncodingPrefix)
    return false;
  if (((word >> 23) & 0x7fu) != opcode)
    return false;
  if (((word >> 16) & 0x7fu) != sdst)
    return false;
  if (((word >> 8) & 0xffu) != 255u)
    return false;
  if ((word & 0xffu) != ssrc0)
    return false;
  literal = literal_word;
  return true;
}

[[nodiscard]] bool sop2_literal_inline_to_sreg(const Instruction &inst, uint32_t word,
                                               uint32_t literal_word, uint32_t opcode,
                                               uint16_t sdst, uint16_t inline_src1,
                                               uint32_t &literal) {
  if (inst.size() != 2 * sizeof(uint32_t))
    return false;
  if ((word >> 30) != kSop2EncodingPrefix)
    return false;
  if (((word >> 23) & 0x7fu) != opcode)
    return false;
  if (((word >> 16) & 0x7fu) != sdst)
    return false;
  if (((word >> 8) & 0xffu) != inline_src1)
    return false;
  if ((word & 0xffu) != 255u)
    return false;
  literal = literal_word;
  return true;
}

[[nodiscard]] bool sop2_sreg_inline_to_sreg(const Instruction &inst, uint32_t word, uint32_t opcode,
                                            uint16_t sdst, uint16_t ssrc0, uint16_t inline_src1) {
  if (inst.size() != sizeof(uint32_t))
    return false;
  if ((word >> 30) != kSop2EncodingPrefix)
    return false;
  if (((word >> 23) & 0x7fu) != opcode)
    return false;
  if (((word >> 16) & 0x7fu) != sdst)
    return false;
  if (((word >> 8) & 0xffu) != inline_src1)
    return false;
  return (word & 0xffu) == ssrc0;
}

[[nodiscard]] std::optional<uint16_t> s_swappc_ssrc0(const Instruction &inst, uint32_t word) {
  if (inst.size() != sizeof(uint32_t))
    return std::nullopt;
  if ((word >> 23) != kSop1EncodingPrefix)
    return std::nullopt;
  // CDNA SOP1 opcode: s_swappc_b64 = 0x1e.
  if (((word >> 8) & 0xffu) != 0x1eu)
    return std::nullopt;
  return static_cast<uint16_t>(word & 0xffu);
}

[[nodiscard]] bool s_swappc_from_sreg(const Instruction &inst, uint32_t word, uint16_t ssrc0) {
  auto actual = s_swappc_ssrc0(inst, word);
  return actual && *actual == ssrc0;
}

[[nodiscard]] std::optional<uint16_t> s_setpc_ssrc0(const Instruction &inst, uint32_t word) {
  if (inst.size() != sizeof(uint32_t))
    return std::nullopt;
  if ((word >> 23) != kSop1EncodingPrefix)
    return std::nullopt;
  // CDNA SOP1 opcode: s_setpc_b64 = 0x1d.
  if (((word >> 8) & 0xffu) != 0x1du)
    return std::nullopt;
  return static_cast<uint16_t>(word & 0xffu);
}

[[nodiscard]] bool s_setpc_from_sreg(const Instruction &inst, uint32_t word, uint16_t ssrc0) {
  auto actual = s_setpc_ssrc0(inst, word);
  return actual && *actual == ssrc0;
}

[[nodiscard]] std::optional<uint16_t> s_call_sdst(const Instruction &inst, uint32_t word) {
  if (inst.size() != sizeof(uint32_t))
    return std::nullopt;
  if (std::string_view(inst.mnemonic()) != "s_call_b64")
    return std::nullopt;
  return static_cast<uint16_t>((word >> 16) & 0x7fu);
}

[[nodiscard]] std::vector<IndirectCallFixup>
recover_static_indirect_call_targets(std::span<const Instruction *const> insts,
                                     std::span<const uint64_t> offsets,
                                     std::span<const uint8_t> text) {
  std::vector<IndirectCallFixup> fixups;
  for (size_t i = 0; i + 3 < insts.size(); ++i) {
    auto sdst = s_getpc_sdst(*insts[i], text_word_at(text, offsets[i]));
    if (!sdst || *sdst >= 127)
      continue;

    uint32_t lo = 0;
    uint32_t hi = 0;
    uint64_t add_lo_offset = offsets[i + 1];
    uint64_t add_hi_offset = offsets[i + 2];
    int32_t add_lo_literal_adjust = 0;
    bool add_hi_has_literal = true;
    bool matched_static_target = false;
    constexpr uint16_t kInlineInt0 = 128;
    // CDNA SOP2 opcodes: s_add_u32 = 0, s_addc_u32 = 4.
    if (sop2_literal_to_sreg(*insts[i + 1], text_word_at(text, offsets[i + 1]),
                             text_word_at(text, offsets[i + 1] + sizeof(uint32_t)), 0, *sdst, *sdst,
                             lo)) {
      if (sop2_literal_to_sreg(*insts[i + 2], text_word_at(text, offsets[i + 2]),
                               text_word_at(text, offsets[i + 2] + sizeof(uint32_t)), 4,
                               static_cast<uint16_t>(*sdst + 1), static_cast<uint16_t>(*sdst + 1),
                               hi)) {
        matched_static_target = true;
      } else if (sop2_sreg_inline_to_sreg(*insts[i + 2], text_word_at(text, offsets[i + 2]), 4,
                                          static_cast<uint16_t>(*sdst + 1),
                                          static_cast<uint16_t>(*sdst + 1), kInlineInt0)) {
        hi = 0;
        add_hi_has_literal = false;
        matched_static_target = true;
      }
    }
    if (!matched_static_target && i + 4 < insts.size()) {
      // Tensile emits static skips as:
      //   s_getpc_b64 s[pc:pc+1]
      //   s_add_i32 temp, literal, 4
      //   s_add_u32 pc, pc, temp
      //   s_addc_u32 pc+1, pc+1, 0
      //   s_setpc_b64 s[pc:pc+1]
      //
      // The literal is one instruction earlier than the low add into the PC
      // register and must be patched to delta - 4.
      uint32_t temp_literal = 0;
      constexpr uint16_t kInlineInt4 = 132;
      const auto temp_sdst =
          static_cast<uint16_t>((text_word_at(text, offsets[i + 1]) >> 16) & 0x7fu);
      if (!sop2_literal_inline_to_sreg(*insts[i + 1], text_word_at(text, offsets[i + 1]),
                                       text_word_at(text, offsets[i + 1] + sizeof(uint32_t)), 2,
                                       temp_sdst, kInlineInt4, temp_literal))
        continue;
      if (!sop2_sreg_inline_to_sreg(*insts[i + 2], text_word_at(text, offsets[i + 2]), 0, *sdst,
                                    *sdst, temp_sdst))
        continue;
      if (!sop2_sreg_inline_to_sreg(*insts[i + 3], text_word_at(text, offsets[i + 3]), 4,
                                    static_cast<uint16_t>(*sdst + 1),
                                    static_cast<uint16_t>(*sdst + 1), kInlineInt0))
        continue;

      lo = temp_literal + 4u;
      hi = 0;
      add_lo_offset = offsets[i + 1];
      add_hi_offset = offsets[i + 3];
      add_lo_literal_adjust = -4;
      add_hi_has_literal = false;
      matched_static_target = true;
    }
    if (!matched_static_target) {
      continue;
    }

    std::vector<size_t> call_indices;
    for (size_t j = i + 3; j < insts.size(); ++j) {
      const uint32_t word = text_word_at(text, offsets[j]);
      if (s_swappc_from_sreg(*insts[j], word, *sdst) || s_setpc_from_sreg(*insts[j], word, *sdst)) {
        call_indices.push_back(j);
      }
    }
    if (call_indices.empty())
      continue;

    const int64_t addend = static_cast<int64_t>((static_cast<uint64_t>(hi) << 32) | lo);
    const int64_t base = static_cast<int64_t>(offsets[i] + insts[i]->size());
    const int64_t target = base + addend;
    if (target < 0 || static_cast<uint64_t>(target) >= text.size())
      continue;

    for (size_t call_index : call_indices) {
      fixups.push_back({.source_getpc_offset = offsets[i],
                        .source_add_lo_offset = add_lo_offset,
                        .source_add_hi_offset = add_hi_offset,
                        .source_call_offset = offsets[call_index],
                        .source_target_offset = static_cast<uint64_t>(target),
                        .source_call_sreg = *sdst,
                        .add_lo_literal_adjust = add_lo_literal_adjust,
                        .add_hi_has_literal = add_hi_has_literal});
    }
  }
  return fixups;
}

[[nodiscard]] std::vector<IndirectCallFixup>
recover_static_indirect_call_targets(const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                                     std::span<const uint8_t> text) {
  std::vector<std::pair<uint64_t, const Instruction *>> source_order;
  for (const auto &block : blocks) {
    if (!block)
      continue;
    uint64_t offset = block->start_offset();
    for (const Instruction &inst : block->instructions()) {
      source_order.emplace_back(offset, &inst);
      offset += inst.size();
    }
  }
  std::ranges::sort(source_order, {}, &std::pair<uint64_t, const Instruction *>::first);

  std::vector<const Instruction *> insts;
  std::vector<uint64_t> offsets;
  insts.reserve(source_order.size());
  offsets.reserve(source_order.size());
  for (const auto &[offset, inst] : source_order) {
    offsets.push_back(offset);
    insts.push_back(inst);
  }

  return recover_static_indirect_call_targets(
      std::span<const Instruction *const>(insts.data(), insts.size()), offsets, text);
}

[[nodiscard]] std::optional<IndirectCallFixup>
recover_carried_indirect_call_target(const KernelTextLayout &layout, const Instruction &inst,
                                     uint32_t word, uint64_t source_call_offset) {
  auto ssrc0 = s_swappc_ssrc0(inst, word);
  if (!ssrc0)
    ssrc0 = s_setpc_ssrc0(inst, word);
  if (!ssrc0)
    return std::nullopt;

  const IndirectCallFixup *latest = nullptr;
  for (const IndirectCallFixup &fixup : layout.indirect_call_fixups) {
    if (fixup.source_call_sreg != *ssrc0 || fixup.source_call_offset >= source_call_offset)
      continue;
    if (latest == nullptr || fixup.source_call_offset > latest->source_call_offset)
      latest = &fixup;
  }
  if (!latest)
    return std::nullopt;

  IndirectCallFixup carried = *latest;
  carried.source_call_offset = source_call_offset;
  return carried;
}

[[nodiscard]] bool recovered_direct_call_return(const KernelTextLayout &layout,
                                                const Instruction &inst, uint32_t word,
                                                uint64_t source_offset) {
  auto ssrc0 = s_setpc_ssrc0(inst, word);
  if (!ssrc0)
    return false;

  return std::ranges::any_of(layout.direct_call_returns, [&](const DirectCallReturn &call) {
    return call.return_sreg == *ssrc0 && call.source_call_offset < source_offset;
  });
}

[[nodiscard]] bool words_changed(std::span<const uint32_t> before,
                                 std::span<const uint32_t> after) {
  if (before.size() != after.size())
    return true;
  return !std::ranges::equal(before, after);
}

void append_diagnostic(std::vector<TranslationDiagnostic> &diagnostics, DiagnosticSeverity severity,
                       DiagnosticKind kind, std::string message,
                       std::optional<uint64_t> guest_offset = std::nullopt,
                       std::string mnemonic = {}, std::vector<std::string> required_work = {}) {
  diagnostics.push_back({.severity = severity,
                         .kind = kind,
                         .guest_offset = guest_offset,
                         .mnemonic = std::move(mnemonic),
                         .message = std::move(message),
                         .required_work = std::move(required_work)});
}

void append_error(std::vector<TranslationDiagnostic> &diagnostics, DiagnosticKind kind,
                  std::string message, std::optional<uint64_t> guest_offset = std::nullopt,
                  std::string mnemonic = {}, std::vector<std::string> required_work = {}) {
  append_diagnostic(diagnostics, DiagnosticSeverity::Error, kind, std::move(message), guest_offset,
                    std::move(mnemonic), std::move(required_work));
}

void append_diagnostics(std::vector<TranslationDiagnostic> &dst,
                        const std::vector<TranslationDiagnostic> &src) {
  dst.insert(dst.end(), src.begin(), src.end());
}

[[nodiscard]] std::vector<uint64_t> kernel_entry_offsets(std::span<const KdTranslation> kernels) {
  std::vector<uint64_t> offsets;
  offsets.reserve(kernels.size());
  for (const KdTranslation &kernel : kernels)
    offsets.push_back(kernel.entry_text_offset);

  std::ranges::sort(offsets);
  offsets.erase(std::ranges::unique(offsets).begin(), offsets.end());
  return offsets;
}

struct KernelTranslationScope {
  KdTranslation *translation = nullptr;
  BasicBlock *entry = nullptr;
  std::vector<BasicBlock *> blocks;
};

[[nodiscard]] std::vector<BasicBlock *>
reachable_kernel_blocks(const std::vector<std::unique_ptr<BasicBlock>> &blocks, BasicBlock &entry,
                        const std::unordered_set<uint64_t> &kernel_entries,
                        std::span<const IndirectCallFixup> recovered_indirect_targets) {
  std::unordered_set<const BasicBlock *> reachable;
  std::vector<BasicBlock *> stack{&entry};

  while (!stack.empty()) {
    BasicBlock *block = stack.back();
    stack.pop_back();
    if (block == nullptr || !reachable.insert(block).second)
      continue;

    for (const IndirectCallFixup &fixup : recovered_indirect_targets) {
      if (fixup.source_getpc_offset < block->start_offset() ||
          fixup.source_getpc_offset >= block->end_offset())
        continue;
      BasicBlock *target = block_for_offset(blocks, fixup.source_target_offset);
      if (target == nullptr)
        continue;
      if (target->start_offset() != entry.start_offset() &&
          kernel_entries.contains(target->start_offset()))
        continue;
      stack.push_back(target);
    }

    for (BasicBlock *succ : block->successors()) {
      if (succ == nullptr)
        continue;
      if (succ->start_offset() != entry.start_offset() &&
          kernel_entries.contains(succ->start_offset()))
        continue;
      stack.push_back(succ);
    }
  }

  std::vector<BasicBlock *> ordered;
  ordered.reserve(reachable.size());
  for (const auto &block : blocks) {
    if (block && reachable.contains(block.get()))
      ordered.push_back(block.get());
  }
  return ordered;
}

[[nodiscard]] std::vector<KernelTranslationScope>
kernel_translation_scopes(const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                          std::span<KdTranslation> kernels,
                          std::span<const IndirectCallFixup> recovered_indirect_targets) {
  std::vector<KernelTranslationScope> scopes;
  const auto entries = kernel_entry_offsets(kernels);
  if (entries.empty())
    return scopes;

  std::unordered_set<uint64_t> entry_set(entries.begin(), entries.end());
  std::vector<KdTranslation *> ordered_kernels;
  ordered_kernels.reserve(kernels.size());
  std::unordered_set<uint64_t> seen_entries;
  for (KdTranslation &kernel : kernels) {
    if (seen_entries.insert(kernel.entry_text_offset).second)
      ordered_kernels.push_back(&kernel);
  }

  std::ranges::sort(ordered_kernels, [](const auto *lhs, const auto *rhs) {
    return lhs->entry_text_offset < rhs->entry_text_offset;
  });

  scopes.reserve(ordered_kernels.size());
  for (KdTranslation *kernel : ordered_kernels) {
    BasicBlock *entry = block_for_offset(blocks, kernel->entry_text_offset);
    if (entry == nullptr)
      continue;

    scopes.push_back(
        {kernel, entry,
         reachable_kernel_blocks(blocks, *entry, entry_set, recovered_indirect_targets)});
  }
  return scopes;
}

} // namespace

BinaryTranslator::~BinaryTranslator() = default;

BinaryTranslator::BinaryTranslator(rj_code_arch_t guest_arch, rj_code_arch_t host_arch,
                                   uint32_t target_mach, BinaryTranslatorOptions options)
    : guest_arch_(guest_arch), host_arch_(host_arch),
      target_mach_(target_mach ? target_mach : elf_mach_for_arch(host_arch)), options_(options),
      encoding_translate_(select_encoding_translator(guest_arch, host_arch)),
      legalization_lookup_(select_legalization(guest_arch, host_arch)),
      semantic_translator_(std::make_unique<SemanticTranslator>(guest_arch, host_arch)) {}

void BinaryTranslator::set_trace_callback(TranslationTraceCallback callback) {
  trace_callback_ = std::move(callback);
}

TranslatedCodeObject BinaryTranslator::translate(const AmdGpuCodeObject &obj) {
  TranslatedCodeObject result;
  result.host_arch = host_arch_;
  diagnostics_ = &result.diagnostics;

  CodeObjectPatcher patcher(obj);
  auto leave_unchanged = [&]() {
    diagnostics_ = nullptr;
    const auto *image = reinterpret_cast<const uint8_t *>(obj.image_data());
    result.elf_bytes.assign(image, image + obj.image_size());
    return result;
  };
  auto text = patcher.text_bytes();
  if (text.empty()) {
    append_error(result.diagnostics, DiagnosticKind::ResourceLimit,
                 "code object does not expose a non-empty .text section for translation");
    return leave_unchanged();
  }

  // Per-kernel text relocation strategy:
  // 1. Decode kernel descriptors and use their entry offsets as translation roots.
  // 2. Decode .text into basic blocks and compute the blocks reachable from each root.
  // 3. Reject shared reachable blocks for this first implementation instead of
  //    guessing how to preserve public kernel references.
  // 4. Emit each kernel's reachable blocks into a compact, source-ordered body.
  // 5. Translate instructions in that relocated body and append oversized
  //    expansions or descriptor ABI prologues into the kernel-local cave.
  // 6. Patch direct PC-relative branches through the kernel-local placement map.
  // 7. Replace the ELF .text payload and redirect descriptors to their new entries.
  auto decoder = Decoder::create(guest_arch_);
  if (!decoder) {
    append_error(result.diagnostics, DiagnosticKind::UnsupportedGuestArch,
                 "unsupported guest_arch: no decoder available");
    return leave_unchanged();
  }

  // Phase 1: descriptor translation gives DBT the source kernel roots and any
  // target descriptor/prologue bytes that must be materialized with the body.
  KernelDescriptorTranslator descriptor_translator(guest_arch_, host_arch_);
  auto descriptor_translations = descriptor_translator.translate_image(
      patcher.image_bytes(), patcher.text_offset(), patcher.text_size(),
      KernelDescriptorTranslationOptions{});
  bool descriptors_supported = true;
  for (const auto &translation : descriptor_translations) {
    append_diagnostics(result.diagnostics, translation.diagnostics);
    descriptors_supported &= translation.supported;
  }
  if (!descriptors_supported) {
    append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                 "kernel descriptor translation requires unsupported resource or ABI "
                 "virtualization; leaving code object unchanged");
    return leave_unchanged();
  }

  if (descriptor_translations.empty()) {
    append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                 "kernel descriptors are required for kernel-level translation");
    return leave_unchanged();
  }

  const auto entry_offsets = kernel_entry_offsets(descriptor_translations);
  // Phase 2: build a CFG over .text and reduce each descriptor root to the
  // source blocks that this kernel owns in the initial relocation strategy.
  auto blocks = BasicBlock::build(obj, *decoder, entry_offsets);
  auto recovered_indirect_targets = recover_static_indirect_call_targets(blocks, text);
  auto scopes =
      kernel_translation_scopes(blocks, descriptor_translations, recovered_indirect_targets);

  if (scopes.size() != entry_offsets.size()) {
    append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                 "kernel descriptor entry offsets are required to map to decoded text blocks");
    return leave_unchanged();
  }

  std::vector<uint8_t> translated_text;
  const bool continue_after_failure = options_.debug_continue_after_failure;

  auto copy_original_instruction = [&](const Instruction &inst, uint64_t offset,
                                       uint64_t target_offset) {
    const uint32_t inst_size = inst.size();
    std::memcpy(translated_text.data() + target_offset, text.data() + offset, inst_size);
    if (!trace_callback_)
      return;

    // Continued-failure mode is diagnostic-only. Emit an explicit copy event so
    // diff reports make it clear which failed source instruction was preserved.
    const auto source_words = raw_words_for_inst(inst);
    trace_callback_({.source_offset = offset,
                     .source_size = inst_size,
                     .source_words = source_words,
                     .legalization = nullptr,
                     .copied_original = true,
                     .semantic_lowering = false,
                     .changed = false,
                     .emitted_in_cave = false,
                     .target_offset = target_offset,
                     .target_words = source_words});
  };

  auto continue_after_instruction_error = [&](const Instruction &inst, uint64_t offset,
                                              uint64_t target_offset) {
    if (!continue_after_failure)
      return false;
    copy_original_instruction(inst, offset, target_offset);
    return true;
  };

  for (const KernelTranslationScope &scope : scopes) {
    if (scope.blocks.empty())
      continue;

    // Phase 4: assign compact target offsets for this kernel before translating
    // instructions. Local cave writes may append bytes, so body placement must be
    // fixed first.
    KernelTextLayout layout;
    layout.translation = scope.translation;
    layout.source_entry = scope.translation->entry_text_offset;

    // The entry block is not necessarily the first reachable source block. The
    // relocated body is compact, so compute the entry delta from emitted block
    // sizes rather than from original .text spacing. This keeps the launch
    // address aligned without preserving unrelated padding or helper gaps.
    uint64_t entry_delta = 0;
    bool found_entry_delta = false;
    for (BasicBlock *block : scope.blocks) {
      if (layout.source_entry >= block->start_offset() &&
          layout.source_entry < block->end_offset()) {
        entry_delta += layout.source_entry - block->start_offset();
        found_entry_delta = true;
        break;
      }
      entry_delta += block->size();
    }
    if (!found_entry_delta) {
      append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                   "kernel descriptor entry offset is not present in the reachable body",
                   layout.source_entry);
      return leave_unchanged();
    }

    const uint64_t body_padding =
        padding_for_residue(translated_text.size() + entry_delta, layout.source_entry % 256, 256);
    append_nop_padding(translated_text, body_padding, host_arch_);

    layout.body_begin = translated_text.size();
    uint64_t cursor = layout.body_begin;
    layout.blocks.reserve(scope.blocks.size());
    for (BasicBlock *block : scope.blocks) {
      cursor = preserve_entry_skip_window_offset(layout, *block, cursor);
      layout.blocks.push_back({.block = block,
                               .source_start = block->start_offset(),
                               .source_end = block->end_offset(),
                               .target_start = cursor,
                               .target_end = cursor + block->size()});
      cursor += block->size();
    }
    layout.body_end = cursor;
    layout.cave_begin = layout.body_end;
    layout.cave_end = layout.body_end;

    // Reserve the compact source-ordered body before instruction translation.
    // Expansion helpers can then append local cave bytes without invalidating
    // any precomputed body placements.
    append_nop_padding(translated_text, layout.body_end - translated_text.size(), host_arch_);
    auto body_entry = target_for_source_offset(layout, layout.source_entry);
    if (!body_entry) {
      append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                   "kernel descriptor entry offset is not present in the relocated body",
                   layout.source_entry);
      return leave_unchanged();
    }
    layout.target_body_entry = *body_entry;

    if (!scope.translation->prologue_words.empty()) {
      // Descriptor prologues are hardware entry points. Align the cave prologue
      // to the original entry residue, then branch into the relocated body.
      const uint64_t prologue_padding =
          padding_for_residue(translated_text.size(), layout.source_entry % 256, 256);
      append_nop_padding(translated_text, prologue_padding, host_arch_);
      layout.target_entry = translated_text.size();
      append_words(translated_text, scope.translation->prologue_words);

      int16_t branch_dwords = 0;
      const uint64_t branch_pc = translated_text.size();
      if (!compute_sopp_branch_offset(branch_pc, layout.target_body_entry, branch_dwords)) {
        append_error(result.diagnostics, DiagnosticKind::ResourceLimit,
                     "kernel descriptor prologue branch range exceeds s_branch simm16; leaving "
                     "code object unchanged",
                     layout.source_entry);
        return leave_unchanged();
      }
      const uint32_t branch = build_s_branch(branch_dwords, host_arch_);
      append_words(translated_text, std::span<const uint32_t>(&branch, 1));
      layout.cave_end = translated_text.size();
    } else {
      layout.target_entry = layout.target_body_entry;
    }

    scope.translation->target_entry_text_offset = layout.target_entry;
    scope.translation->target_body_entry_text_offset = layout.target_body_entry;

    TranslationContext kernel_context(
        scope.translation->target_vgpr_count, scope.translation->target_agpr_count,
        scope.translation->target_accvgpr_base, scope.translation->target_sgpr_count);
    LivenessAnalysisOptions liveness_options;
    // Keep generated DBT temporaries outside the guest kernel's declared VGPR
    // footprint. Some CDNA kernels have descriptor-reserved or otherwise hidden
    // state at the first register after the highest explicit operand, and
    // liveness cannot see those implicit uses.
    liveness_options.min_free_vgpr = static_cast<uint16_t>(
        std::min<uint32_t>(kernel_context.num_vgprs + 1u, std::numeric_limits<uint16_t>::max()));
    if (options_.debug_min_free_vgpr)
      liveness_options.min_free_vgpr =
          std::max(liveness_options.min_free_vgpr, *options_.debug_min_free_vgpr);
    LivenessAnalysis liveness(KernelBlockScope(scope.blocks), liveness_options);

    // Phase 5: translate each relocated body instruction. Oversized semantic
    // expansions branch into this kernel's private cave immediately after the body.
    auto write_debug_stop = [&](const Instruction &inst, uint64_t source_offset,
                                uint64_t relocated_offset) {
      std::vector<uint32_t> stop_words(inst.size() / sizeof(uint32_t), build_s_nop(0, host_arch_));
      stop_words[0] = pack_sopp(1, 0);
      std::memcpy(translated_text.data() + relocated_offset, stop_words.data(), inst.size());
      if (trace_callback_) {
        const auto source_words = raw_words_for_inst(inst);
        trace_callback_({.source_offset = source_offset,
                         .source_size = static_cast<uint32_t>(inst.size()),
                         .source_words = source_words,
                         .legalization = nullptr,
                         .copied_original = false,
                         .semantic_lowering = false,
                         .changed = true,
                         .emitted_in_cave = false,
                         .target_offset = relocated_offset,
                         .target_words = stop_words});
      }
    };

    for (const BlockPlacement &placement : layout.blocks) {
      BasicBlock *block = placement.block;
      for (IndirectCallFixup fixup : recovered_indirect_targets) {
        if (fixup.source_getpc_offset < block->start_offset() ||
            fixup.source_getpc_offset >= block->end_offset())
          continue;
        auto getpc_target = target_for_source_offset(layout, fixup.source_getpc_offset);
        auto add_lo_target = target_for_source_offset(layout, fixup.source_add_lo_offset);
        auto add_hi_target = target_for_source_offset(layout, fixup.source_add_hi_offset);
        if (getpc_target && add_lo_target && add_hi_target) {
          fixup.target_getpc_offset = *getpc_target;
          fixup.target_add_lo_offset = *add_lo_target;
          fixup.target_add_hi_offset = *add_hi_target;
          layout.indirect_call_fixups.push_back(fixup);
        }
      }

      uint64_t offset = block->start_offset();
      uint64_t target_offset = placement.target_start;
      for (auto it = block->instructions().begin(); it != block->instructions().end(); ++it) {
        const auto &inst = *it;
        const uint32_t inst_size = inst.size();

        if (options_.debug_stop_before_offset && offset >= *options_.debug_stop_before_offset) {
          write_debug_stop(inst, offset, target_offset);
          offset += inst_size;
          target_offset += inst_size;
          continue;
        }

        const bool recovered_indirect_call =
            std::ranges::any_of(layout.indirect_call_fixups, [&](const IndirectCallFixup &fixup) {
              return fixup.source_call_offset == offset;
            });
        if (!recovered_indirect_call) {
          auto carried = recover_carried_indirect_call_target(layout, inst,
                                                              text_word_at(text, offset), offset);
          if (carried)
            layout.indirect_call_fixups.push_back(*carried);
        }
        const bool has_recovered_indirect_call =
            std::ranges::any_of(layout.indirect_call_fixups, [&](const IndirectCallFixup &fixup) {
              return fixup.source_call_offset == offset;
            });
        const uint32_t word = text_word_at(text, offset);
        const bool recovered_indirect_return =
            s_setpc_from_sreg(inst, word, 30) ||
            recovered_direct_call_return(layout, inst, word, offset);
        const auto direct_branch_delta = inst.branch_offset_bytes();
        if ((inst.flags() & (INDIRECT_BRANCH | INDIRECT_CALL)) != 0 &&
            !has_recovered_indirect_call && !recovered_indirect_return && !direct_branch_delta) {
          append_error(result.diagnostics, DiagnosticKind::Legalization,
                       "indirect branch or call target recovery is not implemented for relocated "
                       "kernel text",
                       offset, std::string(inst.mnemonic()));
          if (continue_after_instruction_error(inst, offset, target_offset)) {
            offset += inst_size;
            target_offset += inst_size;
            continue;
          }
          return leave_unchanged();
        }

        if (direct_branch_delta) {
          if (auto sdst = s_call_sdst(inst, word))
            layout.direct_call_returns.push_back(
                {.source_call_offset = offset, .return_sreg = *sdst});

          // Record direct branches while emitting the body, but patch only after
          // every block has a final target placement. This keeps fallthrough
          // implicit and limits fixups to explicit PC-relative edges.
          const int64_t source_target =
              static_cast<int64_t>(offset + inst_size) + static_cast<int64_t>(*direct_branch_delta);
          if (source_target < 0) {
            append_error(result.diagnostics, DiagnosticKind::Legalization,
                         "direct branch target is outside the source .text range", offset,
                         std::string(inst.mnemonic()));
            if (continue_after_instruction_error(inst, offset, target_offset)) {
              offset += inst_size;
              target_offset += inst_size;
              continue;
            }
            return leave_unchanged();
          }
          layout.branch_fixups.push_back(
              {.inst = &inst,
               .source_inst_offset = offset,
               .source_target_offset = static_cast<uint64_t>(source_target),
               .target_inst_offset = target_offset});
        }

        const uint32_t *raw = inst.raw_encoding();
        if (!raw) {
          std::memcpy(translated_text.data() + target_offset, text.data() + offset, inst_size);
          if (trace_callback_) {
            trace_callback_({.source_offset = offset,
                             .source_size = inst_size,
                             .source_words = {},
                             .legalization = nullptr,
                             .copied_original = true,
                             .semantic_lowering = false,
                             .changed = false,
                             .emitted_in_cave = false,
                             .target_offset = target_offset,
                             .target_words = {}});
          }
          offset += inst_size;
          target_offset += inst_size;
          continue;
        }

        const InstructionLegalization *leg = nullptr;
        if (legalization_lookup_)
          leg = legalization_lookup_(inst.encoding_id(), inst.opcode());

        const uint16_t dst_opcode = leg ? leg->target_opcode : inst.opcode();

        // Try semantic lowering before raw encoding translation. A matched
        // semantic rule that cannot safely emit code is a translation error:
        // falling through would silently preserve guest semantics on the wrong
        // host ISA.
        {
          auto expansion =
              semantic_translator_->try_lower_expand(inst, offset, liveness, kernel_context);
          if (expansion.status == ExpandStatus::Failed) {
            append_error(result.diagnostics, DiagnosticKind::ExpandFailed,
                         expansion.message.empty()
                             ? "semantic EXPAND rule matched, but could not safely lower"
                             : expansion.message,
                         offset, std::string(inst.mnemonic()), std::move(expansion.required_work));
            if (continue_after_instruction_error(inst, offset, target_offset)) {
              offset += inst_size;
              target_offset += inst_size;
              continue;
            }
            return leave_unchanged();
          }

          if (expansion.status == ExpandStatus::Success) {
            const bool emitted_in_cave = expansion.words.size() * sizeof(uint32_t) > inst_size;
            const uint64_t event_target_offset = emitted_in_cave ? layout.cave_end : target_offset;
            SemanticReplacement repl{target_offset, target_offset + inst_size,
                                     std::move(expansion.words)};
            if (!apply_semantic(repl, translated_text, layout, text, offset + inst_size)) {
              if (continue_after_instruction_error(inst, offset, target_offset)) {
                offset += inst_size;
                target_offset += inst_size;
                continue;
              }
              return leave_unchanged();
            }
            if (trace_callback_) {
              const auto source_words = raw_words_for_inst(inst);
              trace_callback_({.source_offset = offset,
                               .source_size = inst_size,
                               .source_words = source_words,
                               .legalization = leg,
                               .copied_original = false,
                               .semantic_lowering = true,
                               .changed = true,
                               .emitted_in_cave = emitted_in_cave,
                               .target_offset = event_target_offset,
                               .target_words = repl.target_words});
            }
            offset += inst_size;
            target_offset += inst_size;
            continue;
          }
        }

        if (leg && leg->action == Action::Expand) {
          append_error(result.diagnostics, DiagnosticKind::ExpandMissing,
                       "legalization requires EXPAND, but no expansion rule is implemented", offset,
                       std::string(inst.mnemonic()),
                       {"Add a semantic expansion rule for this mnemonic."});
          if (continue_after_instruction_error(inst, offset, target_offset)) {
            offset += inst_size;
            target_offset += inst_size;
            continue;
          }
          return leave_unchanged();
        }

        if (!handle_encoding(inst, offset, target_offset, translated_text, dst_opcode, layout, text,
                             leg)) {
          if (continue_after_instruction_error(inst, offset, target_offset)) {
            offset += inst_size;
            target_offset += inst_size;
            continue;
          }
          return leave_unchanged();
        }
        offset += inst_size;
        target_offset += inst_size;
      }
    }

    if (continue_after_failure && has_error_diagnostic(result.diagnostics))
      continue;

    // Phase 6: now that the local body and cave have final offsets, rewrite only
    // the direct branch immediate bits using the kernel-local source placement.
    for (const BranchFixup &fixup : layout.branch_fixups) {
      auto target_target = target_for_source_offset(layout, fixup.source_target_offset);
      if (!target_target) {
        append_error(result.diagnostics, DiagnosticKind::Legalization,
                     "direct branch target is not present in the kernel-local relocated body",
                     fixup.source_inst_offset,
                     fixup.inst ? std::string(fixup.inst->mnemonic()) : std::string{});
        return leave_unchanged();
      }

      // The source decoder reports branch deltas from the source instruction's
      // next PC. Recompute that same next-PC-relative delta in relocated .text
      // coordinates and patch only the immediate bits of the translated branch.
      const int64_t new_delta = static_cast<int64_t>(*target_target) -
                                static_cast<int64_t>(fixup.target_inst_offset + fixup.inst->size());
      std::vector<uint32_t> words(fixup.inst->size() / sizeof(uint32_t));
      std::memcpy(words.data(), translated_text.data() + fixup.target_inst_offset,
                  fixup.inst->size());
      if (!patch_pcrel_branch_offset(*fixup.inst, words, new_delta, host_arch_)) {
        append_error(result.diagnostics, DiagnosticKind::ResourceLimit,
                     "direct branch relocation exceeds encoded branch range",
                     fixup.source_inst_offset, std::string(fixup.inst->mnemonic()));
        return leave_unchanged();
      }
      std::memcpy(translated_text.data() + fixup.target_inst_offset, words.data(),
                  fixup.inst->size());
    }

    for (const IndirectCallFixup &fixup : layout.indirect_call_fixups) {
      auto target_target = target_for_source_offset(layout, fixup.source_target_offset);
      if (!target_target) {
        append_error(result.diagnostics, DiagnosticKind::Legalization,
                     "recovered indirect call target is not present in the kernel-local relocated "
                     "body",
                     fixup.source_call_offset, "s_swappc_b64");
        return leave_unchanged();
      }

      const int64_t base = static_cast<int64_t>(fixup.target_getpc_offset + sizeof(uint32_t));
      const int64_t delta = static_cast<int64_t>(*target_target) - base;
      const auto delta_bits = static_cast<uint64_t>(delta);
      const uint32_t lo =
          static_cast<uint32_t>(delta_bits + static_cast<uint64_t>(fixup.add_lo_literal_adjust));
      const uint32_t hi = static_cast<uint32_t>(delta_bits >> 32);

      std::memcpy(translated_text.data() + fixup.target_add_lo_offset + sizeof(uint32_t), &lo,
                  sizeof(lo));
      if (fixup.add_hi_has_literal) {
        std::memcpy(translated_text.data() + fixup.target_add_hi_offset + sizeof(uint32_t), &hi,
                    sizeof(hi));
      } else if (hi != 0) {
        append_error(result.diagnostics, DiagnosticKind::ResourceLimit,
                     "recovered indirect branch high PC delta requires a literal addc patch, but "
                     "the source used inline zero",
                     fixup.source_call_offset, "s_setpc_b64");
        return leave_unchanged();
      }
    }

    if (kernel_context.required_vgpr_count > kernel_context.num_vgprs)
      scope.translation->target_vgpr_count = kernel_context.required_vgpr_count;
    if (kernel_context.required_sgpr_count > kernel_context.num_sgprs)
      scope.translation->target_sgpr_count = kernel_context.required_sgpr_count;

    if (scope.translation->target_vgpr_count != kernel_context.num_vgprs ||
        scope.translation->target_sgpr_count != kernel_context.num_sgprs) {
      // Semantic rules may allocate descriptor-backed scratch registers beyond
      // the kernel's original SGPR/VGPR counts. Recompute the descriptor with
      // those larger minimums before patching it into the output image.
      KernelDescriptorTranslationOptions descriptor_options;
      descriptor_options.minimum_vgprs = scope.translation->target_vgpr_count;
      descriptor_options.minimum_sgprs = scope.translation->target_sgpr_count;

      // Descriptor growth is intentionally done after instruction lowering so
      // each kernel is translated once. Only descriptors that enter this code
      // scope need the larger floor; rescanning the whole image would also
      // recompute unrelated kernels and risks mixing diagnostics across scopes.
      bool recomputed_descriptor = false;
      for (KdTranslation &translation : descriptor_translations) {
        if (translation.entry_text_offset != scope.translation->entry_text_offset)
          continue;

        auto updated = descriptor_translator.translate_descriptor(
            patcher.image_bytes(), translation.descriptor_file_offset,
            translation.entry_text_offset, descriptor_options);
        if (!updated) {
          append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                       "kernel descriptor translation could not be recomputed; leaving code object "
                       "unchanged");
          return leave_unchanged();
        }

        append_diagnostics(result.diagnostics, updated->diagnostics);
        if (!updated->supported) {
          append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                       "kernel descriptor translation requires unsupported resource or ABI "
                       "virtualization; leaving code object unchanged");
          return leave_unchanged();
        }

        if (updated->prologue_words != translation.prologue_words) {
          append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                       "kernel descriptor prologue changed after relocated text was emitted; "
                       "leaving code object unchanged");
          return leave_unchanged();
        }

        updated->target_entry_text_offset = layout.target_entry;
        updated->target_body_entry_text_offset = layout.target_body_entry;
        translation = std::move(*updated);
        recomputed_descriptor = true;
      }

      if (!recomputed_descriptor) {
        append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                     "kernel descriptor translation could not be recomputed; leaving code object "
                     "unchanged");
        return leave_unchanged();
      }
    }

    for (KdTranslation &translation : descriptor_translations) {
      if (translation.entry_text_offset != layout.source_entry)
        continue;
      translation.target_entry_text_offset = layout.target_entry;
      translation.target_body_entry_text_offset = layout.target_body_entry;
    }
  }

  if (continue_after_failure && has_error_diagnostic(result.diagnostics))
    return leave_unchanged();

  // Phase 7: write the relocated .text and descriptor entry offsets into the ELF.
  // Reachability-driven emission intentionally drops source padding and other
  // unreachable bytes. Keep the first implementation's ELF mutation one-sided
  // by padding the relocated .text back to at least the original size; local
  // caves have already been placed next to their kernels, so this tail padding
  // does not reintroduce the old global-cave branch-distance problem.
  if (translated_text.size() < text.size())
    append_nop_padding(translated_text, text.size() - translated_text.size(), host_arch_);

  std::unordered_set<uint64_t> applied_descriptors;
  for (const KdTranslation &translation : descriptor_translations) {
    if (applied_descriptors.insert(translation.descriptor_file_offset).second) {
      if (!patcher.apply_kernel_descriptor_translation(translation, host_arch_)) {
        append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                     "kernel descriptor translation could not be applied safely; leaving code "
                     "object unchanged");
        return leave_unchanged();
      }
    }
  }

  if (!patcher.replace_text(translated_text)) {
    append_error(result.diagnostics, DiagnosticKind::ResourceLimit,
                 "relocated .text could not be materialized safely; leaving code object unchanged");
    return leave_unchanged();
  }

  if (target_mach_)
    patcher.update_elf_flags(target_mach_);

  diagnostics_ = nullptr;
  result.elf_bytes = patcher.emit();
  return result;
}

bool BinaryTranslator::apply_semantic(const SemanticReplacement &repl, std::vector<uint8_t> &text,
                                      KernelTextLayout &layout, std::span<const uint8_t> orig_text,
                                      uint64_t source_return_offset) {
  assert(repl.matched() && "apply_semantic called with unmatched replacement");
  assert(repl.start_offset < repl.end_offset && "invalid replacement range");
  assert(repl.end_offset <= text.size() && "replacement exceeds text bounds");

  const uint32_t source_size = repl.end_offset - repl.start_offset;
  const uint32_t target_size = repl.target_words.size() * 4;

  if (target_size <= source_size) {
    std::memcpy(text.data() + repl.start_offset, repl.target_words.data(), target_size);
    if (target_size < source_size)
      std::memset(text.data() + repl.start_offset + target_size, 0, source_size - target_size);
    return true;
  }

  const uint64_t cave_byte_offset = layout.cave_end;
  const uint64_t stub_next =
      target_for_source_fallthrough(layout, orig_text, source_return_offset, guest_arch_)
          .value_or(repl.start_offset + source_size);
  const uint64_t branch_pc = repl.start_offset;

  // s_branch simm16 targets (PC + 4 + simm16*4).
  int16_t fwd_dwords = 0;
  if (!compute_sopp_branch_offset(branch_pc, cave_byte_offset, fwd_dwords)) {
    if (diagnostics_)
      append_error(*diagnostics_, DiagnosticKind::ResourceLimit,
                   "code cave branch range exceeds s_branch simm16; leaving code object unchanged",
                   repl.start_offset);
    return false;
  }

  const uint32_t stub = build_s_branch(fwd_dwords, host_arch_);
  std::memcpy(text.data() + repl.start_offset, &stub, 4);
  for (uint64_t off = repl.start_offset + 4; off < repl.end_offset; off += 4) {
    const uint32_t nop = build_s_nop(0, host_arch_);
    std::memcpy(text.data() + off, &nop, 4);
  }

  auto cave_words = repl.target_words;
  int16_t ret_dwords = 0;
  const uint64_t return_branch_pc = cave_byte_offset + cave_words.size() * sizeof(uint32_t);
  if (!compute_sopp_branch_offset(return_branch_pc, stub_next, ret_dwords)) {
    if (diagnostics_)
      append_error(*diagnostics_, DiagnosticKind::ResourceLimit,
                   "code cave return branch range exceeds s_branch simm16; leaving code object "
                   "unchanged",
                   repl.start_offset);
    return false;
  }
  cave_words.push_back(build_s_branch(ret_dwords, host_arch_));

  append_words(text, cave_words);
  layout.cave_end = text.size();
  return true;
}

bool BinaryTranslator::handle_encoding(const Instruction &inst, uint64_t offset,
                                       uint64_t target_offset, std::vector<uint8_t> &text,
                                       uint16_t dst_opcode, KernelTextLayout &layout,
                                       std::span<const uint8_t> orig_text,
                                       const InstructionLegalization *leg) {
  const uint32_t *raw = inst.raw_encoding();
  assert(raw && "handle_encoding called without raw encoding");
  const bool tracing = static_cast<bool>(trace_callback_);
  const auto source_words = tracing ? raw_words_for_inst(inst) : std::vector<uint32_t>{};

  auto emit_trace = [&](bool copied_original, bool changed, bool emitted_in_cave,
                        uint64_t target_offset, std::span<const uint32_t> target_words) {
    if (!trace_callback_)
      return;
    trace_callback_({.source_offset = offset,
                     .source_size = static_cast<uint32_t>(inst.size()),
                     .source_words = source_words,
                     .legalization = leg,
                     .copied_original = copied_original,
                     .semantic_lowering = false,
                     .changed = changed,
                     .emitted_in_cave = emitted_in_cave,
                     .target_offset = target_offset,
                     .target_words = target_words});
  };

  if (!encoding_translate_) {
    std::memcpy(text.data() + target_offset, raw, inst.size());
    emit_trace(true, false, false, target_offset, source_words);
    return true;
  }

  const uint32_t w0 = raw[0];
  const uint32_t w1 = inst.size() > 4 ? raw[1] : 0;
  const uint32_t w2 = inst.size() > 8 ? raw[2] : 0;

  auto tr = encoding_translate_(inst.encoding_id(), w0, w1, w2, dst_opcode);

  if (tr.word_count == 0) {
    std::memcpy(text.data() + target_offset, raw, inst.size());
    emit_trace(true, false, false, target_offset, source_words);
    return true;
  }

  // Append trailing literal constant when the source instruction is larger
  // than the translated encoding. This handles single-word formats (SOP1,
  // SOP2, VOP1, VOP2, etc.) with a 32-bit literal appended when a source
  // operand is 0xFF. The encoding translator returns the format's native
  // word count; the literal is always one extra word beyond that.
  // Guard: only append if the gap is exactly one word (the literal). Larger
  // gaps would indicate a format mismatch, not a trailing literal.
  const uint32_t translated_bytes = tr.word_count * 4u;
  const uint32_t orig_bytes = inst.size();
  if (orig_bytes - translated_bytes == 4 && tr.word_count < 3) {
    uint32_t lit_word;
    std::memcpy(&lit_word, orig_text.data() + offset + translated_bytes, 4);
    tr.words[tr.word_count++] = lit_word;
  }

  const uint32_t target_size = tr.word_count * 4u;
  const auto target_words = std::span<const uint32_t>(tr.words, tr.word_count);
  const bool emitted_in_cave = target_size > orig_bytes;
  const uint64_t event_target_offset = emitted_in_cave ? layout.cave_end : target_offset;
  const bool changed = tracing && words_changed(source_words, target_words);

  if (target_size <= orig_bytes) {
    std::memcpy(text.data() + target_offset, tr.words, target_size);
  } else {
    SemanticReplacement repl{
        target_offset, target_offset + inst.size(), {tr.words, tr.words + tr.word_count}};
    if (!apply_semantic(repl, text, layout, orig_text, offset + orig_bytes))
      return false;
  }
  emit_trace(false, changed, emitted_in_cave, event_target_offset, target_words);
  return true;
}

} // namespace rocjitsu
