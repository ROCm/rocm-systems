// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/kernel_text_layout.h"

#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <utility>

namespace rocjitsu {

[[nodiscard]] TextRelocationResult relocation_ok() { return {}; }

[[nodiscard]] TextRelocationResult relocation_error(uint64_t source_offset, std::string message) {
  return {.ok = false, .source_offset = source_offset, .message = std::move(message)};
}

void append_words(std::vector<uint8_t> &text, std::span<const uint32_t> words) {
  if (words.empty())
    return;

  const size_t old_size = text.size();
  const size_t extra_bytes = words.size() * sizeof(uint32_t);
  text.resize(old_size + extra_bytes);
  std::memcpy(text.data() + old_size, words.data(), extra_bytes);
}

void append_nop_padding(std::vector<uint8_t> &text, uint64_t byte_count, rj_code_arch_t arch) {
  assert(byte_count % sizeof(uint32_t) == 0 && "padding must be word-aligned");
  if (byte_count == 0)
    return;

  // Large translated code objects can compact reachable text substantially, but
  // CodeObjectPatcher still expects the final section to be padded back to the
  // original size when the replacement is smaller. Resize once here so padding
  // a 100+ MiB tail stays memory-bandwidth bound instead of looping once per
  // instruction word.
  const size_t old_size = text.size();
  const size_t extra = static_cast<size_t>(byte_count);
  text.resize(old_size + extra);

  const uint32_t nop = build_s_nop(0, arch);
  std::memcpy(text.data() + old_size, &nop, sizeof(nop));
  size_t filled = sizeof(nop);
  while (filled < extra) {
    const size_t copy_size = std::min(filled, extra - filled);
    std::memcpy(text.data() + old_size + filled, text.data() + old_size, copy_size);
    filled += copy_size;
  }
}

[[nodiscard]] uint64_t padding_for_residue(uint64_t current_offset, uint64_t target_residue,
                                           uint64_t alignment) {
  const uint64_t current_residue = current_offset % alignment;
  return (target_residue + alignment - current_residue) % alignment;
}

[[nodiscard]] std::optional<uint64_t> target_for_source_offset(const KernelTextLayout &layout,
                                                               uint64_t source_offset) {
  if (layout.blocks.empty())
    return std::nullopt;

  // Blocks are emitted in source order and are non-overlapping in source space
  // for the current scope; binary search preserves the prior semantics of the
  // linear scan while reducing lookup complexity to O(log N).
  const auto it = std::upper_bound(layout.blocks.begin(), layout.blocks.end(), source_offset,
                                   [](uint64_t source, const BlockPlacement &placement) {
                                     return source < placement.source_start;
                                   });
  if (it == layout.blocks.begin())
    return std::nullopt;

  const BlockPlacement &placement = *(it - 1);
  if (source_offset != placement.source_start)
    return std::nullopt;

  return placement.target_start;
}

void rebase_kernel_text_layout(KernelTextLayout &layout, uint64_t delta) {
  // Descriptor entries can be synthetic launch/prologue stubs rather than
  // source-block locations. Callers set target_entry only after those final
  // hardware-visible offsets are known, so rebase only body-relative state here.
  layout.target_body_entry += delta;
  layout.body_begin += delta;
  layout.body_end += delta;

  for (BlockPlacement &placement : layout.blocks) {
    placement.target_start += delta;
    placement.target_end += delta;
  }
  for (BranchFixup &fixup : layout.branch_fixups)
    fixup.target_inst_offset += delta;
  for (RecoveredIndirectFixup &fixup : layout.recovered_indirect_fixups)
    fixup.target_window_offset += delta;
  for (IndirectCallFixup &fixup : layout.recovered_builder_fixups) {
    fixup.target_getpc_offset += delta;
    fixup.target_recovery_begin_offset += delta;
    fixup.target_recovery_end_offset += delta;
  }
}

[[nodiscard]] bool text_contains_range(std::span<const uint8_t> text, uint64_t offset,
                                       uint64_t size) {
  return offset <= text.size() && size <= text.size() - offset;
}

[[nodiscard]] bool append_recovered_indirect_sequence(std::vector<uint32_t> &words,
                                                      const RecoveredIndirectFixup &fixup,
                                                      uint64_t target_offset, rj_code_arch_t arch) {
  if (const auto direct_simm =
          compute_sopp_branch_simm16(fixup.target_window_offset, target_offset)) {
    if (fixup.is_call)
      words.push_back(build_s_call_b64(fixup.return_sreg, *direct_simm, arch));
    else
      words.push_back(build_s_branch(*direct_simm, arch));
    return true;
  }

  constexpr uint64_t kMaxSigned = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
  if (fixup.target_window_offset > kMaxSigned - sizeof(uint32_t) || target_offset > kMaxSigned)
    return false;

  // The long form intentionally rebuilds the final translated target in the same
  // SGPR pair consumed by the original setpc/swappc. The preceding source-side
  // address builder may still execute, but this sequence overwrites the pair
  // immediately before the actual control transfer.
  words.push_back(build_s_getpc_b64(fixup.target_sreg, arch));
  const int64_t base = static_cast<int64_t>(fixup.target_window_offset + sizeof(uint32_t));
  const int64_t delta = static_cast<int64_t>(target_offset) - base;
  if (!append_pc_delta_builder(words, arch, fixup.target_sreg, delta))
    return false;
  if (fixup.is_call)
    words.push_back(build_s_swappc_b64(fixup.return_sreg, fixup.target_sreg, arch));
  else
    words.push_back(build_s_setpc_b64(fixup.target_sreg, arch));
  return words.size() <= kMaxRecoveredIndirectTransferWords;
}

TextRelocationResult patch_direct_branch_fixups(std::vector<uint8_t> &text,
                                                const KernelTextLayout &layout,
                                                rj_code_arch_t arch) {
  for (const BranchFixup &fixup : layout.branch_fixups) {
    auto target_target = target_for_source_offset(layout, fixup.source_target_offset);
    if (!target_target) {
      return relocation_error(
          fixup.source_inst_offset,
          "direct branch target is not present in the kernel-local relocated body");
    }
    if (fixup.inst == nullptr) {
      return relocation_error(fixup.source_inst_offset,
                              "direct branch relocation is missing decoded instruction metadata");
    }
    if (!text_contains_range(text, fixup.target_inst_offset, fixup.inst->size())) {
      return relocation_error(fixup.source_inst_offset,
                              "direct branch relocation points outside translated .text");
    }

    // The source decoder reports branch deltas from the source instruction's
    // next PC. Recompute that same next-PC-relative delta in relocated .text
    // coordinates and patch only the immediate bits of the translated branch.
    const int64_t new_delta = static_cast<int64_t>(*target_target) -
                              static_cast<int64_t>(fixup.target_inst_offset + fixup.inst->size());
    std::vector<uint32_t> words(fixup.inst->size() / sizeof(uint32_t));
    std::memcpy(words.data(), text.data() + fixup.target_inst_offset, fixup.inst->size());
    if (!patch_pcrel_branch_offset(*fixup.inst, words, new_delta, arch)) {
      return relocation_error(fixup.source_inst_offset,
                              "direct branch relocation exceeds encoded branch range");
    }
    std::memcpy(text.data() + fixup.target_inst_offset, words.data(), fixup.inst->size());
  }

  return relocation_ok();
}

TextRelocationResult patch_recovered_indirect_fixups(std::vector<uint8_t> &text,
                                                     const KernelTextLayout &layout,
                                                     rj_code_arch_t arch) {
  std::unordered_map<uint64_t, uint64_t> patched_windows;
  for (const RecoveredIndirectFixup &fixup : layout.recovered_indirect_fixups) {
    auto target_target = target_for_source_offset(layout, fixup.source_target_offset);
    if (!target_target) {
      return relocation_error(
          fixup.source_call_offset,
          "recovered indirect branch target is not present in the kernel-local relocated body");
    }

    auto [window_it, inserted] =
        patched_windows.emplace(fixup.target_window_offset, static_cast<uint64_t>(*target_target));
    if (!inserted) {
      if (window_it->second != static_cast<uint64_t>(*target_target)) {
        return relocation_error(fixup.source_call_offset,
                                "recovered indirect branch has multiple incompatible targets");
      }
      continue;
    }

    constexpr uint64_t kWindowBytes =
        kMaxRecoveredIndirectTransferWords * static_cast<uint64_t>(sizeof(uint32_t));
    if (!text_contains_range(text, fixup.target_window_offset, kWindowBytes)) {
      return relocation_error(fixup.source_call_offset,
                              "recovered indirect branch window points outside translated .text");
    }

    std::vector<uint32_t> words;
    if (!append_recovered_indirect_sequence(words, fixup, *target_target, arch)) {
      return relocation_error(
          fixup.source_call_offset,
          "target ISA cannot encode canonical recovered indirect branch sequence");
    }
    if (words.size() > kMaxRecoveredIndirectTransferWords) {
      return relocation_error(fixup.source_call_offset,
                              "recovered indirect branch sequence exceeds reserved window");
    }

    std::memcpy(text.data() + fixup.target_window_offset, words.data(),
                words.size() * sizeof(uint32_t));
    const uint32_t nop = build_s_nop(0, arch);
    for (uint64_t off = fixup.target_window_offset + words.size() * sizeof(uint32_t);
         off < fixup.target_window_offset + kWindowBytes; off += sizeof(uint32_t)) {
      std::memcpy(text.data() + off, &nop, sizeof(nop));
    }
  }

  return relocation_ok();
}

TextRelocationResult patch_recovered_builder_fixups(std::vector<uint8_t> &text,
                                                    const KernelTextLayout &layout,
                                                    rj_code_arch_t arch) {
  std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> rewritten_regions;
  for (const IndirectCallFixup &fixup : layout.recovered_builder_fixups) {
    auto target_target = target_for_source_offset(layout, fixup.source_target_offset);
    if (!target_target) {
      return relocation_error(
          fixup.source_call_offset,
          "recovered indirect branch target is not present in the kernel-local relocated body");
    }

    if (fixup.target_recovery_begin_offset > fixup.target_recovery_end_offset) {
      return relocation_error(fixup.source_call_offset,
                              "recovered indirect branch builder range is malformed");
    }
    const uint64_t recovery_size =
        fixup.target_recovery_end_offset - fixup.target_recovery_begin_offset;
    if (!text_contains_range(text, fixup.target_recovery_begin_offset, recovery_size)) {
      return relocation_error(fixup.source_call_offset,
                              "recovered indirect branch builder points outside translated .text");
    }

    // One source-side builder may feed multiple consumers. Rewriting the same
    // builder more than once is valid only when every consumer agrees on the
    // final relocated target and byte range.
    const auto rewrite_key =
        std::pair{fixup.target_recovery_end_offset, static_cast<uint64_t>(*target_target)};
    auto [rewrite_it, inserted] =
        rewritten_regions.emplace(fixup.target_recovery_begin_offset, rewrite_key);
    if (!inserted) {
      if (rewrite_it->second != rewrite_key) {
        return relocation_error(
            fixup.source_call_offset,
            "recovered indirect branch builder is reused for incompatible targets");
      }
      continue;
    }

    constexpr uint64_t kMaxSigned = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    if (fixup.target_getpc_offset > kMaxSigned - sizeof(uint32_t) || *target_target > kMaxSigned) {
      return relocation_error(
          fixup.source_call_offset,
          "target ISA cannot encode canonical recovered indirect branch builder");
    }

    const int64_t base = static_cast<int64_t>(fixup.target_getpc_offset + sizeof(uint32_t));
    const int64_t delta = static_cast<int64_t>(*target_target) - base;
    std::vector<uint32_t> replacement_words;
    if (!append_pc_delta_builder(replacement_words, arch, fixup.source_call_sreg, delta)) {
      return relocation_error(
          fixup.source_call_offset,
          "target ISA cannot encode canonical recovered indirect branch builder");
    }

    const uint64_t replacement_size = replacement_words.size() * sizeof(uint32_t);
    if (replacement_size > recovery_size) {
      return relocation_error(fixup.source_call_offset,
                              "recovered indirect branch builder does not fit in its source range");
    }

    std::memcpy(text.data() + fixup.target_recovery_begin_offset, replacement_words.data(),
                replacement_size);
    const uint32_t nop = build_s_nop(0, arch);
    for (uint64_t off = fixup.target_recovery_begin_offset + replacement_size;
         off < fixup.target_recovery_end_offset; off += sizeof(uint32_t)) {
      std::memcpy(text.data() + off, &nop, sizeof(nop));
    }
  }

  return relocation_ok();
}

} // namespace rocjitsu
