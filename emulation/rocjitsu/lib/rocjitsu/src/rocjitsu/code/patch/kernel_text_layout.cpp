// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/kernel_text_layout.h"

#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <sstream>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace rocjitsu {

namespace {

constexpr uint64_t kKernargPreloadSkipBytes = 256;
// SOPP reaches almost 128 KiB in either direction, and a branch-island chain
// keeps each hop within a 64 KiB safety interval. Pools are placed before
// control-flow windows grow. A dense interval of recovered indirect consumers
// or direct branches can grow to their respective maximum transfer widths.
// An 8 KiB grid remains within the 64 KiB interval under either bound.
constexpr uint64_t kDirectBranchIslandSpacingBytes = 8 * 1024;
static_assert(kDirectBranchIslandSpacingBytes *
                  std::max(kMaxRecoveredIndirectTransferWords, kMaxDirectBranchTransferWords) <=
              64 * 1024);
} // namespace

[[nodiscard]] TextRelocationResult relocation_ok() { return {}; }

[[nodiscard]] TextRelocationResult
relocation_error(uint64_t source_offset, std::string message,
                 TextLayoutFailureCategory failure = TextLayoutFailureCategory::InvalidLayout,
                 TextLayoutFailureReason reason = TextLayoutFailureReason::None) {
  return {.ok = false,
          .failure = failure,
          .reason = reason,
          .source_offset = source_offset,
          .required_windows = {},
          .message = std::move(message),
          .compact_builder_fallbacks = {}};
}

[[nodiscard]] KernelTextAppendResult
kernel_text_append_ok(uint64_t target_delta, uint64_t target_entry, uint64_t target_body_entry) {
  return {.ok = true,
          .failure = TextLayoutFailureCategory::None,
          .source_offset = 0,
          .target_delta = target_delta,
          .target_entry = target_entry,
          .target_body_entry = target_body_entry,
          .message = {}};
}

[[nodiscard]] KernelTextAppendResult kernel_text_append_error(
    uint64_t source_offset, std::string message,
    TextLayoutFailureCategory failure = TextLayoutFailureCategory::InvalidLayout) {
  return {.ok = false,
          .failure = failure,
          .source_offset = source_offset,
          .target_delta = 0,
          .target_entry = 0,
          .target_body_entry = 0,
          .message = std::move(message)};
}

[[nodiscard]] std::string direct_branch_range_error(uint64_t branch_offset, uint64_t target_offset,
                                                    int64_t delta_bytes) {
  std::ostringstream os;
  os << "direct branch relocation exceeds encoded branch range";
  os << " (branch .text+0x" << std::hex << branch_offset;
  os << " target .text+0x" << target_offset;
  os << std::dec << " delta_bytes=" << delta_bytes << ")";
  return os.str();
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

[[nodiscard]] uint64_t kernel_entry_stub_bytes(const KernelEntryLayoutPlan &translation) {
  return translation.prologue_words.size() * sizeof(uint32_t) + sizeof(uint32_t);
}

void write_words_at(std::vector<uint8_t> &dst, uint64_t offset, std::span<const uint32_t> words) {
  if (words.empty())
    return;
  std::memcpy(dst.data() + offset, words.data(), words.size() * sizeof(uint32_t));
}

[[nodiscard]] bool write_launch_stub(std::vector<uint8_t> &text,
                                     const KernelEntryLayoutPlan &translation, uint64_t stub_offset,
                                     uint64_t target_offset, rj_code_arch_t arch) {
  uint64_t cursor = stub_offset;
  write_words_at(text, cursor, translation.prologue_words);
  cursor += translation.prologue_words.size() * sizeof(uint32_t);

  const auto branch_dwords = compute_sopp_branch_simm16(cursor, target_offset);
  if (!branch_dwords)
    return false;
  const uint32_t branch = build_s_branch(*branch_dwords, arch);
  write_words_at(text, cursor, std::span<const uint32_t>(&branch, 1));
  return true;
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

bool kernarg_preload_launch_window_fits(const KernelEntryLayoutPlan &translation) {
  return !translation.has_kernarg_preload_firmware_skip ||
         kernel_entry_stub_bytes(translation) <= kKernargPreloadSkipBytes;
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
  for (uint64_t &slot : layout.branch_island_slots)
    slot += delta;
  for (RecoveredIndirectFixup &fixup : layout.recovered_indirect_fixups)
    fixup.target_window_offset += delta;
  for (IndirectCallFixup &fixup : layout.recovered_builder_fixups) {
    fixup.target_getpc_offset += delta;
    fixup.target_recovery_begin_offset += delta;
    fixup.target_recovery_end_offset += delta;
  }
}

KernelTextAppendResult append_skipped_kernel_stub(std::vector<uint8_t> &text,
                                                  const SkippedKernelLayoutPlan &plan,
                                                  rj_code_arch_t arch) {
  const uint64_t source_entry = plan.source_entry;
  const uint64_t padding = padding_for_residue(text.size(), source_entry % 256, 256);
  append_nop_padding(text, padding, arch);
  const uint64_t target_entry = text.size();

  // Keep skipped symbols loadable without placing guest ISA bytes in the
  // target ELF. HIP and ROCR may cache or query every descriptor in a module
  // even when the application dispatches only a subset of kernels. End the
  // wave immediately if one is dispatched: a trap can abort or wedge its HSA
  // queue, while the mandatory load-time diagnostic loudly reports that this
  // no-op body cannot produce valid kernel outputs.
  const uint32_t endpgm = build_s_endpgm(arch);
  append_words(text, std::span<const uint32_t>(&endpgm, 1));
  if (plan.has_kernarg_preload_firmware_skip) {
    append_nop_padding(text, kKernargPreloadSkipBytes - sizeof(uint32_t), arch);
    append_words(text, std::span<const uint32_t>(&endpgm, 1));
  }

  return kernel_text_append_ok(0, target_entry, target_entry);
}

KernelTextAppendResult append_relocated_kernel_text(std::vector<uint8_t> &translated_text,
                                                    KernelTextLayout &layout,
                                                    std::span<const uint8_t> kernel_text,
                                                    rj_code_arch_t arch) {
  auto body_entry = target_for_source_offset(layout, layout.source_entry);
  if (!body_entry) {
    return kernel_text_append_error(
        layout.source_entry, "kernel descriptor entry offset is not present in the relocated body");
  }
  layout.target_body_entry = *body_entry;

  std::optional<uint64_t> preload_body_entry;
  if (layout.entry_plan.has_kernarg_preload_firmware_skip) {
    const uint64_t source_preload_entry =
        layout.entry_plan.kernarg_preload_firmware_entry_text_offset;
    preload_body_entry = target_for_source_offset(layout, source_preload_entry);
    if (!preload_body_entry) {
      return kernel_text_append_error(
          source_preload_entry,
          "kernarg preload firmware entry offset is not present in the relocated body");
    }
    if (!kernarg_preload_launch_window_fits(layout.entry_plan)) {
      return kernel_text_append_error(
          layout.source_entry,
          "kernel descriptor prologue does not fit in the 256-byte kernarg preload compatibility "
          "window",
          TextLayoutFailureCategory::ResourceLimit);
    }
  }

  const bool has_descriptor_prologue = !layout.entry_plan.prologue_words.empty();
  uint64_t target_delta = 0;
  if (layout.entry_plan.has_kernarg_preload_firmware_skip) {
    // Kernarg-preload kernels have two hardware-visible entries separated by
    // exactly 256 bytes. Reserve that launch window before appending the body;
    // the stubs are written after the body offsets have been rebased.
    const uint64_t launch_padding =
        padding_for_residue(translated_text.size(), layout.source_entry % 256, 256);
    append_nop_padding(translated_text, launch_padding, arch);
    layout.target_entry = translated_text.size();
    const uint64_t launch_end =
        layout.target_entry + kKernargPreloadSkipBytes + kernel_entry_stub_bytes(layout.entry_plan);
    append_nop_padding(translated_text, launch_end - translated_text.size(), arch);
    target_delta = translated_text.size();
  } else if (has_descriptor_prologue) {
    // Descriptor ABI prologues are hardware-visible entry stubs. Place the stub
    // before the relocated body so large kernels do not depend on a single
    // SOPP branch reaching backward across the entire emitted body. Keep both
    // the descriptor entry and relocated guest entry on the original entry
    // residue: the former is the hardware launch address, while the latter
    // preserves the body placement invariant used by kernels without prologues.
    const uint64_t launch_padding =
        padding_for_residue(translated_text.size(), layout.source_entry % 256, 256);
    append_nop_padding(translated_text, launch_padding, arch);
    layout.target_entry = translated_text.size();
    const uint64_t launch_end = layout.target_entry + kernel_entry_stub_bytes(layout.entry_plan);
    append_nop_padding(translated_text, launch_end - translated_text.size(), arch);
    const uint64_t body_padding = padding_for_residue(
        translated_text.size() + layout.target_body_entry, layout.source_entry % 256, 256);
    append_nop_padding(translated_text, body_padding, arch);
    target_delta = translated_text.size();
  } else {
    const uint64_t body_padding = padding_for_residue(
        translated_text.size() + layout.target_body_entry, layout.source_entry % 256, 256);
    append_nop_padding(translated_text, body_padding, arch);
    target_delta = translated_text.size();
  }

  rebase_kernel_text_layout(layout, target_delta);
  translated_text.insert(translated_text.end(), kernel_text.begin(), kernel_text.end());

  if (layout.entry_plan.has_kernarg_preload_firmware_skip) {
    assert(preload_body_entry && "preload body entry was checked before rebase");
    if (!write_launch_stub(translated_text, layout.entry_plan, layout.target_entry,
                           layout.target_body_entry, arch)) {
      return kernel_text_append_error(layout.source_entry,
                                      "kernarg preload launch branch cannot encode target body",
                                      TextLayoutFailureCategory::ResourceLimit);
    }
    if (!write_launch_stub(translated_text, layout.entry_plan,
                           layout.target_entry + kKernargPreloadSkipBytes,
                           *preload_body_entry + target_delta, arch)) {
      return kernel_text_append_error(
          layout.entry_plan.kernarg_preload_firmware_entry_text_offset,
          "kernarg preload firmware launch branch cannot encode target body",
          TextLayoutFailureCategory::ResourceLimit);
    }
  } else if (has_descriptor_prologue) {
    if (!write_launch_stub(translated_text, layout.entry_plan, layout.target_entry,
                           layout.target_body_entry, arch)) {
      return kernel_text_append_error(
          layout.source_entry, "kernel descriptor prologue branch range exceeds s_branch simm16",
          TextLayoutFailureCategory::ResourceLimit);
    }
  } else {
    layout.target_entry = layout.target_body_entry;
  }

  return kernel_text_append_ok(target_delta, layout.target_entry, layout.target_body_entry);
}

[[nodiscard]] bool text_contains_range(std::span<const uint8_t> text, uint64_t offset,
                                       uint64_t size) {
  return offset <= text.size() && size <= text.size() - offset;
}

[[nodiscard]] bool append_recovered_indirect_sequence(std::vector<uint32_t> &words,
                                                      const RecoveredIndirectFixup &fixup,
                                                      uint64_t target_offset, rj_code_arch_t arch) {
  if (!fixup.preserve_marked_long_transfer) {
    if (const auto direct_simm =
            compute_sopp_branch_simm16(fixup.target_window_offset, target_offset)) {
      if (fixup.is_call)
        words.push_back(build_s_call_b64(fixup.return_sreg, *direct_simm, arch));
      else
        words.push_back(build_s_branch(*direct_simm, arch));
      return true;
    }
  }

  if (fixup.keep_dynamic_when_long) {
    // The builder immediately ahead was rewritten in place to the relocated target, so the
    // original transfer already lands correctly and needs no window at all. Emitting one would
    // cost five words and leave an unmarked getpc/add/consumer triple for the next pass to wrap.
    // One word in, one word out, so the reserved window is exact and .text cannot change size.
    if (fixup.is_call)
      words.push_back(build_s_swappc_b64(fixup.return_sreg, fixup.target_sreg, arch));
    else
      words.push_back(build_s_setpc_b64(fixup.target_sreg, arch));
    return true;
  }

  // Every long recovered transfer carries the same non-padding marker. A later
  // pass can then preserve the exact window even if CFG compaction has brought
  // its target back within direct SOPP range.
  words.push_back(build_s_nop(kLongDirectBranchMarkerNopImmediate, arch));
  const uint64_t sequence_offset = fixup.target_window_offset + sizeof(uint32_t);

  constexpr uint64_t kMaxSigned = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
  if (sequence_offset > kMaxSigned - sizeof(uint32_t) || target_offset > kMaxSigned)
    return false;

  // The long form intentionally rebuilds the final translated target in the same
  // SGPR pair consumed by the original setpc/swappc. The preceding source-side
  // address builder may still execute, but this sequence overwrites the pair
  // immediately before the actual control transfer.
  words.push_back(build_s_getpc_b64(fixup.target_sreg, arch));
  const int64_t base = static_cast<int64_t>(sequence_offset + sizeof(uint32_t));
  const int64_t delta = static_cast<int64_t>(target_offset) - base;
  if (!append_pc_delta_builder(words, arch, fixup.target_sreg, delta))
    return false;
  if (fixup.is_call)
    words.push_back(build_s_swappc_b64(fixup.return_sreg, fixup.target_sreg, arch));
  else
    words.push_back(build_s_setpc_b64(fixup.target_sreg, arch));
  return true;
}

[[nodiscard]] bool append_long_pc_transfer(std::vector<uint32_t> &words, rj_code_arch_t arch,
                                           uint16_t pc_sreg, uint64_t sequence_offset,
                                           uint64_t target_offset,
                                           std::optional<uint16_t> call_sdst) {
  constexpr uint64_t kMaxSigned = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
  if (sequence_offset > kMaxSigned - sizeof(uint32_t) || target_offset > kMaxSigned)
    return false;

  words.push_back(build_s_getpc_b64(pc_sreg, arch));
  const int64_t base = static_cast<int64_t>(sequence_offset + sizeof(uint32_t));
  const int64_t delta = static_cast<int64_t>(target_offset) - base;
  if (!append_pc_delta_builder(words, arch, pc_sreg, delta))
    return false;

  if (call_sdst)
    words.push_back(build_s_swappc_b64(*call_sdst, pc_sreg, arch));
  else
    words.push_back(build_s_setpc_b64(pc_sreg, arch));
  return true;
}

[[nodiscard]] bool conditional_branch_can_invert(std::string_view mnemonic) {
  // COND_BRANCH alone includes conditionals without a defined adjacent inverse.
  // Mnemonic identity is an ISA-wide semantic contract in rocjitsu, so this
  // target-independent whitelist is safer than assuming every conditional
  // opcode can be inverted by toggling its low bit.
  return mnemonic == "s_cbranch_scc0" || mnemonic == "s_cbranch_scc1" ||
         mnemonic == "s_cbranch_vccz" || mnemonic == "s_cbranch_vccnz" ||
         mnemonic == "s_cbranch_execz" || mnemonic == "s_cbranch_execnz";
}

uint64_t initial_direct_branch_patch_window_bytes(const Instruction &inst) { return inst.size(); }

void rebase_text_offset(uint64_t &offset, std::span<const TextLayoutInsertion> insertions) {
  assert(std::ranges::is_sorted(
      insertions, {}, [](const TextLayoutInsertion &insertion) { return insertion.offset; }));
  const uint64_t original_offset = offset;
  for (const TextLayoutInsertion &insertion : insertions) {
    if (original_offset < insertion.offset)
      break;
    offset += insertion.size;
  }
}

TextOffsetRebaser::TextOffsetRebaser(std::span<const TextLayoutInsertion> insertions) {
  assert(std::ranges::is_sorted(
      insertions, {}, [](const TextLayoutInsertion &insertion) { return insertion.offset; }));
  offsets_.reserve(insertions.size());
  shifts_.reserve(insertions.size());
  uint64_t running = 0;
  for (const TextLayoutInsertion &insertion : insertions) {
    running += insertion.size;
    offsets_.push_back(insertion.offset);
    shifts_.push_back(running);
  }
}

void TextOffsetRebaser::rebase(uint64_t &offset) const {
  // Same result as rebase_text_offset(): add every insertion at or before this offset. Done as a
  // prefix sum plus one binary search rather than a scan, because the scan is called once per
  // block bound and per fixup, which made it quadratic in a large scope -- 11% of translation time
  // on a 745 MB device image.
  const auto it = std::ranges::upper_bound(offsets_, offset);
  if (it == offsets_.begin())
    return;
  offset += shifts_[static_cast<size_t>(std::distance(offsets_.begin(), it)) - 1];
}

std::optional<std::vector<TextLayoutInsertion>>
grow_control_flow_windows(std::vector<uint8_t> &text, KernelTextLayout &layout,
                          std::span<const ControlFlowWindowRequirement> requirements,
                          rj_code_arch_t arch) {
  struct ValidatedGrowth {
    ControlFlowWindowKind kind = ControlFlowWindowKind::DirectBranch;
    size_t fixup_index = 0;
    uint64_t required_window_bytes = 0;
    TextLayoutInsertion insertion;
  };

  std::vector<ValidatedGrowth> growths;
  growths.reserve(requirements.size());
  uint64_t total_insertion_bytes = 0;
  for (const ControlFlowWindowRequirement &requirement : requirements) {
    size_t fixup_index = 0;
    uint64_t target_window_offset = 0;
    uint64_t target_window_bytes = 0;
    if (requirement.kind == ControlFlowWindowKind::DirectBranch) {
      const auto selected =
          std::ranges::find_if(layout.branch_fixups, [&](const BranchFixup &fixup) {
            return fixup.source_inst_offset == requirement.source_inst_offset;
          });
      if (selected == layout.branch_fixups.end())
        return std::nullopt;
      if (!selected->allow_window_growth)
        return std::nullopt;
      fixup_index = static_cast<size_t>(std::distance(layout.branch_fixups.begin(), selected));
      target_window_offset = selected->target_inst_offset;
      target_window_bytes = selected->target_window_bytes;
    } else {
      const auto selected = std::ranges::find_if(
          layout.recovered_indirect_fixups, [&](const RecoveredIndirectFixup &fixup) {
            return fixup.source_call_offset == requirement.source_inst_offset;
          });
      if (selected == layout.recovered_indirect_fixups.end())
        return std::nullopt;
      fixup_index =
          static_cast<size_t>(std::distance(layout.recovered_indirect_fixups.begin(), selected));
      target_window_offset = selected->target_window_offset;
      target_window_bytes = selected->target_window_bytes;
    }

    if (target_window_bytes == 0 || target_window_bytes % sizeof(uint32_t) != 0 ||
        requirement.required_window_bytes <= target_window_bytes ||
        requirement.required_window_bytes % sizeof(uint32_t) != 0 ||
        target_window_offset > text.size() ||
        target_window_bytes > text.size() - target_window_offset)
      return std::nullopt;

    if (std::ranges::any_of(growths, [&](const ValidatedGrowth &growth) {
          return growth.kind == requirement.kind && growth.fixup_index == fixup_index;
        }))
      return std::nullopt;

    const uint64_t insertion_offset = target_window_offset + target_window_bytes;
    const uint64_t insertion_size = requirement.required_window_bytes - target_window_bytes;
    if (insertion_offset > text.size() ||
        insertion_size > std::numeric_limits<size_t>::max() - text.size() ||
        total_insertion_bytes >
            static_cast<uint64_t>(std::numeric_limits<size_t>::max()) - insertion_size)
      return std::nullopt;

    total_insertion_bytes += insertion_size;
    growths.push_back({.kind = requirement.kind,
                       .fixup_index = fixup_index,
                       .required_window_bytes = requirement.required_window_bytes,
                       .insertion = {.offset = insertion_offset, .size = insertion_size}});
  }

  if (total_insertion_bytes >
      static_cast<uint64_t>(std::numeric_limits<size_t>::max() - text.size()))
    return std::nullopt;

  std::ranges::sort(growths, {},
                    [](const ValidatedGrowth &growth) { return growth.insertion.offset; });
  for (size_t i = 1; i < growths.size(); ++i) {
    if (growths[i - 1].insertion.offset >= growths[i].insertion.offset)
      return std::nullopt;
  }

  std::vector<uint8_t> grown_text;
  grown_text.reserve(text.size() + static_cast<size_t>(total_insertion_bytes));
  uint64_t copied_until = 0;
  std::vector<TextLayoutInsertion> insertions;
  insertions.reserve(growths.size());
  for (const ValidatedGrowth &growth : growths) {
    const TextLayoutInsertion insertion = growth.insertion;
    grown_text.insert(grown_text.end(), text.begin() + static_cast<std::ptrdiff_t>(copied_until),
                      text.begin() + static_cast<std::ptrdiff_t>(insertion.offset));
    append_nop_padding(grown_text, insertion.size, arch);
    copied_until = insertion.offset;
    insertions.push_back(insertion);
  }
  grown_text.insert(grown_text.end(), text.begin() + static_cast<std::ptrdiff_t>(copied_until),
                    text.end());

  for (const ValidatedGrowth &growth : growths) {
    if (growth.kind == ControlFlowWindowKind::DirectBranch) {
      layout.branch_fixups[growth.fixup_index].target_window_bytes = growth.required_window_bytes;
    } else {
      layout.recovered_indirect_fixups[growth.fixup_index].target_window_bytes =
          growth.required_window_bytes;
    }
  }

  const TextOffsetRebaser rebaser(insertions);
  for (BlockPlacement &block : layout.blocks) {
    rebaser.rebase(block.target_start);
    rebaser.rebase(block.target_end);
  }
  for (BranchFixup &fixup : layout.branch_fixups)
    rebaser.rebase(fixup.target_inst_offset);
  for (RecoveredIndirectFixup &fixup : layout.recovered_indirect_fixups)
    rebaser.rebase(fixup.target_window_offset);
  for (IndirectCallFixup &fixup : layout.recovered_builder_fixups) {
    rebaser.rebase(fixup.target_getpc_offset);
    rebaser.rebase(fixup.target_recovery_begin_offset);
    rebaser.rebase(fixup.target_recovery_end_offset);
  }
  for (uint64_t &slot : layout.branch_island_slots)
    rebaser.rebase(slot);
  rebaser.rebase(layout.body_end);

  text = std::move(grown_text);
  return insertions;
}

uint64_t first_direct_branch_island_pool_offset() { return kDirectBranchIslandSpacingBytes; }

uint64_t next_direct_branch_island_pool_offset(uint64_t current_body_size) {
  return current_body_size + kDirectBranchIslandSpacingBytes;
}

void append_direct_branch_island_pool(std::vector<uint8_t> &kernel_text, KernelTextLayout &layout,
                                      rj_code_arch_t arch) {
  const uint32_t marker = build_s_nop(kBranchIslandPoolMarkerNopImmediate, arch);
  append_words(kernel_text, std::span<const uint32_t>(&marker, 1));

  const uint64_t skip_offset = kernel_text.size();
  const uint32_t skip_pool =
      build_s_branch(static_cast<int16_t>(kDirectBranchIslandPoolSlots), arch);
  append_words(kernel_text, std::span<const uint32_t>(&skip_pool, 1));

  // Normal fallthrough executes the skip above and lands after the pool. A
  // patched out-of-range branch may instead target one of these private slots,
  // each of which is later rewritten to an unconditional branch to the next
  // island or final target.
  const uint32_t placeholder = build_s_branch(0, arch);
  for (uint16_t i = 0; i < kDirectBranchIslandPoolSlots; ++i) {
    layout.branch_island_slots.push_back(skip_offset + sizeof(uint32_t) +
                                         static_cast<uint64_t>(i) * sizeof(uint32_t));
    append_words(kernel_text, std::span<const uint32_t>(&placeholder, 1));
  }
}

[[nodiscard]] std::optional<uint32_t> build_inverted_conditional_skip(const Instruction &inst,
                                                                      uint32_t translated_word,
                                                                      uint64_t window_offset,
                                                                      uint64_t window_bytes,
                                                                      rj_code_arch_t arch) {
  if (!conditional_branch_can_invert(inst.mnemonic()))
    return std::nullopt;
  const auto skip = compute_sopp_branch_simm16(window_offset, window_offset + window_bytes);
  if (!skip)
    return std::nullopt;

  // The supported conditional SOPP opcodes are encoded in adjacent false/true
  // pairs, but the first opcode in each pair is odd. XORing the low bit would
  // therefore cross pair boundaries (and turns gfx1250 s_cbranch_execnz opcode
  // 38 into invalid opcode 39). Move explicitly to the adjacent inverse while
  // preserving the translated ISA's SOPP opcode numbering.
  const uint32_t op = (translated_word >> 16) & 0x7fu;
  const std::string_view mnemonic = inst.mnemonic();
  const bool invert_to_next =
      mnemonic == "s_cbranch_scc0" || mnemonic == "s_cbranch_vccz" || mnemonic == "s_cbranch_execz";
  return build_sopp_encoding(arch, invert_to_next ? op + 1u : op - 1u,
                             static_cast<uint16_t>(*skip));
}

[[nodiscard]] std::optional<uint16_t> direct_call_return_sgpr(const Instruction &inst,
                                                              uint32_t translated_word) {
  // Generated s_call has call metadata and a PC-relative label. Register-target
  // calls share the call flag but do not expose a direct branch displacement.
  if ((inst.flags() & INDIRECT_CALL) == 0 || !inst.branch_offset_bytes())
    return std::nullopt;
  return static_cast<uint16_t>((translated_word >> 16) & 0x7fu);
}

[[nodiscard]] std::optional<size_t> find_branch_island_slot(uint64_t branch_pc,
                                                            uint64_t target_offset,
                                                            std::span<const uint64_t> island_slots,
                                                            std::span<const uint8_t> island_used) {
  if (target_offset > branch_pc) {
    std::optional<size_t> best;
    for (size_t i = 0; i < island_slots.size(); ++i) {
      if (island_used[i])
        continue;
      const uint64_t slot = island_slots[i];
      if (slot <= branch_pc || slot >= target_offset)
        continue;
      if (!compute_sopp_branch_simm16(branch_pc, slot))
        continue;
      if (!best || slot > island_slots[*best])
        best = i;
    }
    return best;
  }

  if (target_offset < branch_pc) {
    std::optional<size_t> best;
    for (size_t i = 0; i < island_slots.size(); ++i) {
      if (island_used[i])
        continue;
      const uint64_t slot = island_slots[i];
      if (slot >= branch_pc || slot <= target_offset)
        continue;
      if (!compute_sopp_branch_simm16(branch_pc, slot))
        continue;
      if (!best || slot < island_slots[*best])
        best = i;
    }
    return best;
  }

  return std::nullopt;
}

[[nodiscard]] std::optional<std::vector<uint64_t>>
allocate_branch_island_chain(uint64_t branch_pc, uint64_t target_offset,
                             std::span<const uint64_t> island_slots,
                             std::vector<uint8_t> &island_used) {
  std::vector<uint64_t> chain;
  uint64_t current_pc = branch_pc;

  while (!compute_sopp_branch_simm16(current_pc, target_offset)) {
    const auto slot_index =
        find_branch_island_slot(current_pc, target_offset, island_slots, island_used);
    if (!slot_index)
      return std::nullopt;
    island_used[*slot_index] = true;
    current_pc = island_slots[*slot_index];
    chain.push_back(current_pc);
  }

  return chain;
}

[[nodiscard]] bool append_branch_island_direct_sequence(
    std::vector<uint32_t> &words, const Instruction &inst, uint32_t translated_word,
    uint64_t window_offset, uint64_t window_bytes, uint64_t first_target, rj_code_arch_t arch) {
  if ((inst.flags() & BRANCH) != 0) {
    const auto simm = compute_sopp_branch_simm16(window_offset, first_target);
    if (!simm)
      return false;
    words.push_back(build_s_branch(*simm, arch));
    return true;
  }

  if (direct_call_return_sgpr(inst, translated_word))
    return false;

  if (window_bytes < 2 * sizeof(uint32_t))
    return false;
  auto inverted = build_inverted_conditional_skip(inst, translated_word, window_offset,
                                                  2 * sizeof(uint32_t), arch);
  if (!inverted)
    return false;
  const auto simm = compute_sopp_branch_simm16(window_offset + sizeof(uint32_t), first_target);
  if (!simm)
    return false;

  words.push_back(*inverted);
  words.push_back(build_s_branch(*simm, arch));
  return true;
}

[[nodiscard]] bool append_long_direct_branch_sequence(std::vector<uint32_t> &words,
                                                      const Instruction &inst,
                                                      uint32_t translated_word,
                                                      uint64_t window_offset, uint64_t window_bytes,
                                                      uint64_t target_offset, rj_code_arch_t arch,
                                                      uint16_t pc_sreg) {
  const uint32_t marker = build_s_nop(kLongDirectBranchMarkerNopImmediate, arch);
  if ((inst.flags() & BRANCH) != 0) {
    words.push_back(marker);
    return append_long_pc_transfer(words, arch, pc_sreg, window_offset + sizeof(uint32_t),
                                   target_offset, std::nullopt);
  }

  if (auto call_sdst = direct_call_return_sgpr(inst, translated_word)) {
    words.push_back(marker);
    return append_long_pc_transfer(words, arch, pc_sreg, window_offset + sizeof(uint32_t),
                                   target_offset, call_sdst);
  }

  auto inverted =
      build_inverted_conditional_skip(inst, translated_word, window_offset, window_bytes, arch);
  if (!inverted)
    return false;
  words.push_back(*inverted);
  words.push_back(marker);
  return append_long_pc_transfer(words, arch, pc_sreg, window_offset + 2 * sizeof(uint32_t),
                                 target_offset, std::nullopt);
}

TextRelocationResult patch_direct_branch_fixups(std::vector<uint8_t> &text,
                                                const KernelTextLayout &layout,
                                                rj_code_arch_t arch) {
  struct PlannedWindowPatch {
    uint64_t offset = 0;
    std::vector<uint32_t> words;
  };
  struct PlannedIslandPatch {
    uint64_t offset = 0;
    uint32_t word = 0;
  };

  std::vector<uint8_t> branch_island_used(layout.branch_island_slots.size(), 0);
  std::vector<PlannedWindowPatch> window_patches;
  window_patches.reserve(layout.branch_fixups.size());
  std::vector<PlannedIslandPatch> island_patches;
  TextRelocationResult growth_required = relocation_error(
      0, "direct branches require wider patch windows", TextLayoutFailureCategory::ResourceLimit,
      TextLayoutFailureReason::BranchOutOfRange);

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
    if (fixup.target_window_bytes < static_cast<uint64_t>(fixup.inst->size()) ||
        fixup.target_window_bytes % sizeof(uint32_t) != 0) {
      return relocation_error(fixup.source_inst_offset,
                              "direct branch relocation has malformed patch window");
    }
    if (fixup.translated_words.size() * sizeof(uint32_t) !=
        static_cast<size_t>(fixup.inst->size())) {
      return relocation_error(
          fixup.source_inst_offset,
          "direct branch relocation is missing its pristine translated encoding");
    }
    if (!text_contains_range(text, fixup.target_inst_offset, fixup.target_window_bytes)) {
      return relocation_error(fixup.source_inst_offset,
                              "direct branch relocation points outside translated .text");
    }

    // The source decoder reports branch deltas from the source instruction's
    // next PC. Recompute that same next-PC-relative delta in relocated .text
    // coordinates and patch only the immediate bits of the translated branch.
    const int64_t new_delta = static_cast<int64_t>(*target_target) -
                              static_cast<int64_t>(fixup.target_inst_offset + fixup.inst->size());
    std::vector<uint32_t> words = fixup.translated_words;
    if (!patch_pcrel_branch_offset(*fixup.inst, words, new_delta, arch)) {
      if (!layout.long_branch_sgpr) {
        const uint64_t first_branch_pc = (fixup.inst->flags() & BRANCH) != 0
                                             ? fixup.target_inst_offset
                                             : fixup.target_inst_offset + sizeof(uint32_t);
        auto chain = allocate_branch_island_chain(first_branch_pc, *target_target,
                                                  layout.branch_island_slots, branch_island_used);
        if (!chain || chain->empty()) {
          return relocation_error(
              fixup.source_inst_offset,
              direct_branch_range_error(fixup.target_inst_offset, *target_target, new_delta),
              TextLayoutFailureCategory::ResourceLimit, TextLayoutFailureReason::BranchOutOfRange);
        }

        std::vector<uint32_t> island_words;
        if (!append_branch_island_direct_sequence(
                island_words, *fixup.inst, words.front(), fixup.target_inst_offset,
                fixup.target_window_bytes, chain->front(), arch)) {
          const bool can_grow_conditional_source =
              (fixup.inst->flags() & COND_BRANCH) != 0 &&
              conditional_branch_can_invert(fixup.inst->mnemonic()) &&
              fixup.target_window_bytes < 2 * sizeof(uint32_t);
          if (!can_grow_conditional_source) {
            return relocation_error(
                fixup.source_inst_offset,
                "direct branch relocation cannot build SGPR-free island sequence");
          }
          if (!fixup.allow_window_growth) {
            return relocation_error(
                fixup.source_inst_offset,
                "fixed-size direct branch window cannot grow for an island sequence",
                TextLayoutFailureCategory::ResourceLimit,
                TextLayoutFailureReason::BranchOutOfRange);
          }
          if (growth_required.required_windows.empty())
            growth_required.source_offset = fixup.source_inst_offset;
          growth_required.required_windows.push_back(
              {.kind = ControlFlowWindowKind::DirectBranch,
               .source_inst_offset = fixup.source_inst_offset,
               .required_window_bytes = 2 * sizeof(uint32_t)});
          continue;
        }
        words = std::move(island_words);
        for (size_t i = 0; i < chain->size(); ++i) {
          const uint64_t next =
              i + 1 < chain->size() ? (*chain)[i + 1] : static_cast<uint64_t>(*target_target);
          const auto simm = compute_sopp_branch_simm16((*chain)[i], next);
          if (!simm || !text_contains_range(text, (*chain)[i], sizeof(uint32_t))) {
            return relocation_error(fixup.source_inst_offset,
                                    "direct branch island patch points outside translated .text");
          }
          island_patches.push_back({.offset = (*chain)[i], .word = build_s_branch(*simm, arch)});
        }
      } else {
        std::vector<uint32_t> long_words;
        if (!append_long_direct_branch_sequence(long_words, *fixup.inst, words.front(),
                                                fixup.target_inst_offset, fixup.target_window_bytes,
                                                *target_target, arch, *layout.long_branch_sgpr)) {
          return relocation_error(fixup.source_inst_offset,
                                  "direct branch relocation cannot build long branch sequence");
        }
        if (long_words.size() * sizeof(uint32_t) > fixup.target_window_bytes) {
          if (!fixup.allow_window_growth) {
            return relocation_error(fixup.source_inst_offset,
                                    "fixed-size direct branch window cannot grow for a long branch",
                                    TextLayoutFailureCategory::ResourceLimit,
                                    TextLayoutFailureReason::BranchOutOfRange);
          }
          if (growth_required.required_windows.empty())
            growth_required.source_offset = fixup.source_inst_offset;
          growth_required.required_windows.push_back(
              {.kind = ControlFlowWindowKind::DirectBranch,
               .source_inst_offset = fixup.source_inst_offset,
               .required_window_bytes = long_words.size() * sizeof(uint32_t)});
          continue;
        }
        words = std::move(long_words);
      }
    }

    std::vector<uint32_t> window_words(fixup.target_window_bytes / sizeof(uint32_t),
                                       build_s_nop(0, arch));
    std::copy(words.begin(), words.end(), window_words.begin());
    window_patches.push_back(
        {.offset = fixup.target_inst_offset, .words = std::move(window_words)});
  }

  if (!growth_required.required_windows.empty())
    return growth_required;

  for (const PlannedWindowPatch &patch : window_patches) {
    std::memcpy(text.data() + patch.offset, patch.words.data(),
                patch.words.size() * sizeof(uint32_t));
  }
  for (const PlannedIslandPatch &patch : island_patches)
    std::memcpy(text.data() + patch.offset, &patch.word, sizeof(patch.word));

  return relocation_ok();
}

TextRelocationResult patch_recovered_indirect_fixups(std::vector<uint8_t> &text,
                                                     const KernelTextLayout &layout,
                                                     rj_code_arch_t arch) {
  struct PlannedWindowPatch {
    uint64_t offset = 0;
    std::vector<uint32_t> words;
  };

  std::unordered_map<uint64_t, uint64_t> patched_windows;
  std::vector<PlannedWindowPatch> window_patches;
  window_patches.reserve(layout.recovered_indirect_fixups.size());
  TextRelocationResult growth_required = relocation_error(
      0, "recovered indirect branches require wider patch windows",
      TextLayoutFailureCategory::ResourceLimit, TextLayoutFailureReason::BranchOutOfRange);
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

    if (fixup.target_window_bytes == 0 || fixup.target_window_bytes % sizeof(uint32_t) != 0 ||
        !text_contains_range(text, fixup.target_window_offset, fixup.target_window_bytes)) {
      return relocation_error(fixup.source_call_offset,
                              "recovered indirect branch window points outside translated .text");
    }

    std::vector<uint32_t> words;
    if (!append_recovered_indirect_sequence(words, fixup, *target_target, arch)) {
      return relocation_error(
          fixup.source_call_offset,
          "target ISA cannot encode canonical recovered indirect branch sequence",
          TextLayoutFailureCategory::ResourceLimit);
    }
    if (words.size() * sizeof(uint32_t) > fixup.target_window_bytes) {
      // The current sequence builder is bounded by this constant. Retain the
      // check as a fail-closed contract if a future encoding adds words.
      if (words.size() > kMaxRecoveredIndirectTransferWords) {
        return relocation_error(fixup.source_call_offset,
                                "recovered indirect branch sequence exceeds maximum window",
                                TextLayoutFailureCategory::ResourceLimit);
      }
      if (growth_required.required_windows.empty())
        growth_required.source_offset = fixup.source_call_offset;
      growth_required.required_windows.push_back(
          {.kind = ControlFlowWindowKind::RecoveredIndirect,
           .source_inst_offset = fixup.source_call_offset,
           .required_window_bytes = words.size() * sizeof(uint32_t)});
      continue;
    }

    std::vector<uint32_t> window_words(fixup.target_window_bytes / sizeof(uint32_t),
                                       build_s_nop(0, arch));
    std::copy(words.begin(), words.end(), window_words.begin());
    window_patches.push_back(
        {.offset = fixup.target_window_offset, .words = std::move(window_words)});
  }

  if (!growth_required.required_windows.empty())
    return growth_required;

  for (const PlannedWindowPatch &patch : window_patches) {
    std::memcpy(text.data() + patch.offset, patch.words.data(),
                patch.words.size() * sizeof(uint32_t));
  }
  return relocation_ok();
}

TextRelocationResult patch_recovered_builder_fixups(std::vector<uint8_t> &text,
                                                    const KernelTextLayout &layout,
                                                    rj_code_arch_t arch) {
  std::unordered_map<uint64_t, std::tuple<uint64_t, uint64_t, bool, uint16_t>> rewritten_regions;
  std::vector<uint64_t> compact_builder_fallbacks;
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

    // One source-side builder may feed multiple consumers. Only the first one
    // rewrites the range, so reuse is valid only when every consumer asks for
    // the same replacement. Every input the emitted words depend on has to be in
    // this key: the target and byte range, the drain requirement, and the SGPR
    // pair the add is written against. Omitting the pair let a second consumer
    // that names a different builder register silently inherit the first
    // consumer's replacement, which a lane-banked dispatcher can produce because
    // its producer and consumer pairs differ.
    const auto rewrite_key =
        std::tuple{fixup.target_recovery_end_offset, static_cast<uint64_t>(*target_target),
                   fixup.source_requires_xcnt_drain, fixup.source_builder_sreg};
    auto [rewrite_it, inserted] =
        rewritten_regions.emplace(fixup.target_recovery_begin_offset, rewrite_key);
    if (!inserted) {
      if (rewrite_it->second != rewrite_key) {
        return relocation_error(
            fixup.source_call_offset,
            "recovered indirect branch builder is reused for incompatible replacements");
      }
      continue;
    }

    constexpr uint64_t kMaxSigned = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    if (fixup.target_getpc_offset > kMaxSigned - sizeof(uint32_t) || *target_target > kMaxSigned) {
      return relocation_error(
          fixup.source_call_offset,
          "target ISA cannot encode canonical recovered indirect branch builder",
          TextLayoutFailureCategory::ResourceLimit);
    }

    const int64_t base = static_cast<int64_t>(fixup.target_getpc_offset + sizeof(uint32_t));
    const int64_t delta = static_cast<int64_t>(*target_target) - base;
    std::vector<uint32_t> replacement_words;
    if (fixup.source_requires_xcnt_drain) {
      // The drain the source range held ordered the writes to this same SGPR
      // pair, so it has to come before the builder rather than be NOP-filled
      // away with the rest of the range.
      const auto drain = build_s_wait_xcnt(arch);
      if (!drain) {
        return relocation_error(
            fixup.source_call_offset,
            "target ISA cannot encode the XCNT drain the recovered builder requires",
            TextLayoutFailureCategory::ResourceLimit);
      }
      replacement_words.push_back(*drain);
    }
    // Emit the literal64 add form. This builder survives into the output, so a later translation
    // pass has to be able to account for it, and the relocation lattice models only that encoding
    // -- it cannot be widened to the compact one, because the patcher writes an eight-byte delta
    // into the literal slot.
    //
    // Every rewritten builder gets the wide form, not just the ones whose consumer became a
    // window. Which builders still have a consumer on the NEXT pass is not knowable here -- a
    // whole-scope fixup has none to begin with, and a recovered consumer can disappear into a
    // direct transfer -- so predicting who needs to stay visible gets it wrong. Widening
    // unconditionally and falling back when it does not fit is both simpler and measurably
    // better: on RCCL it takes the unaccounted-builder count on re-translation from 32 to 0.
    //
    // The pair named is the one the builder's own getpc wrote, not the one its consumer reads.
    // The replacement covers only the delta half and leaves that getpc in place, so any other
    // pair would be added to whatever it happened to hold. The two coincide for a direct
    // getpc/add/swappc chain and diverge for a lane-banked dispatcher, which restores the address
    // into a different pair between the two.
    const size_t replacement_begin = replacement_words.size();
    if (!append_pc_delta_builder(replacement_words, arch, fixup.source_builder_sreg, delta,
                                 /*prefer_literal64=*/true)) {
      return relocation_error(
          fixup.source_call_offset,
          "target ISA cannot encode canonical recovered indirect branch builder",
          TextLayoutFailureCategory::ResourceLimit);
    }
    // The wide form does not always fit a range sized for the compact one. Falling back keeps the
    // builder correct and keeps objects that translate today translating; what it gives up is
    // visibility to the lattice, which only matters if an unrecovered indirect transfer also
    // survives into this object -- and in that case the next pass refuses rather than accepting
    // something wrong. Widening the range is not available here: the patcher writes in place.
    if ((replacement_words.size() * sizeof(uint32_t)) > recovery_size) {
      replacement_words.resize(replacement_begin);
      if (!append_pc_delta_builder(replacement_words, arch, fixup.source_builder_sreg, delta)) {
        return relocation_error(
            fixup.source_call_offset,
            "target ISA cannot encode canonical recovered indirect branch builder",
            TextLayoutFailureCategory::ResourceLimit);
      }
      compact_builder_fallbacks.push_back(fixup.source_call_offset);
    }

    const uint64_t replacement_size = replacement_words.size() * sizeof(uint32_t);
    if (replacement_size > recovery_size) {
      return relocation_error(fixup.source_call_offset,
                              "recovered indirect branch builder does not fit in its source range",
                              TextLayoutFailureCategory::ResourceLimit);
    }

    std::memcpy(text.data() + fixup.target_recovery_begin_offset, replacement_words.data(),
                replacement_size);
    const uint32_t nop = build_s_nop(0, arch);
    for (uint64_t off = fixup.target_recovery_begin_offset + replacement_size;
         off < fixup.target_recovery_end_offset; off += sizeof(uint32_t)) {
      std::memcpy(text.data() + off, &nop, sizeof(nop));
    }
  }

  TextRelocationResult result = relocation_ok();
  result.compact_builder_fallbacks = std::move(compact_builder_fallbacks);
  return result;
}

} // namespace rocjitsu
