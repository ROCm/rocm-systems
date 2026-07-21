// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/dbt/binary_translator.h"

#include "rocjitsu/analysis/kernel_scope.h"
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
#include "rocjitsu/code/patch/kernel_text_layout.h"
#include "rocjitsu/isa/arch/amdgpu/isa_properties.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rocjitsu {

namespace {

inline constexpr uint64_t kKernargPreloadSkipBytes = 256;

// Translation output is generated from one accepted code object. Bound the
// additional ELF storage, including alignment padding, before the patcher makes
// any potentially large allocation. Other patcher callers can choose a
// different explicit budget.
constexpr size_t kMaxTranslatedFileGrowth = size_t{64} * 1024 * 1024;

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

[[nodiscard]] std::vector<uint32_t> raw_words_for_inst(const Instruction &inst) {
  const uint32_t *raw = inst.raw_encoding();
  if (!raw)
    return {};
  return {raw, raw + inst.size() / sizeof(uint32_t)};
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

[[nodiscard]] std::vector<uint64_t>
kernel_hardware_entry_offsets(std::span<const KdTranslation> kernels) {
  std::vector<uint64_t> offsets;
  offsets.reserve(kernels.size() * 2);
  for (const KdTranslation &kernel : kernels) {
    offsets.push_back(kernel.entry_text_offset);
    if (kernel.has_kernarg_preload)
      offsets.push_back(kernel.kernarg_preload_entry_text_offset);
  }

  std::ranges::sort(offsets);
  offsets.erase(std::ranges::unique(offsets).begin(), offsets.end());
  return offsets;
}

[[nodiscard]] std::vector<uint64_t> kernel_block_leaders(std::span<const KdTranslation> kernels,
                                                         std::span<const uint8_t> text) {
  std::vector<uint64_t> offsets;
  offsets.reserve(kernels.size() * 2);
  for (const KdTranslation &kernel : kernels) {
    offsets.push_back(kernel.entry_text_offset);
    // AMDHSA kernarg preloading is descriptor-controlled. When
    // kernarg_preload_spec_length is non-zero, compatible CP firmware starts at
    // KERNEL_CODE_ENTRY_BYTE_OFFSET + 256. That address is a real hardware entry,
    // not merely padding, so split a block there and seed reachability from it.
    if (kernel.has_kernarg_preload && kernel.kernarg_preload_entry_text_offset < text.size())
      offsets.push_back(kernel.kernarg_preload_entry_text_offset);
  }

  std::ranges::sort(offsets);
  offsets.erase(std::ranges::unique(offsets).begin(), offsets.end());
  return offsets;
}

struct KernelTranslationScope {
  KdTranslation *translation = nullptr;
  BasicBlock *entry = nullptr;
  std::vector<BasicBlock *> blocks;
  std::vector<ScopedCfgEdge> liveness_edges;
  std::unordered_set<uint64_t> call_return_offsets;
};

[[nodiscard]] std::vector<KernelTranslationScope>
kernel_translation_scopes(const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                          const BlockOffsetIndex &block_index, std::span<KdTranslation> kernels,
                          std::span<const uint8_t> text) {
  std::vector<KernelTranslationScope> scopes;
  const auto entries = kernel_entry_offsets(kernels);
  if (entries.empty())
    return scopes;

  const auto hardware_entries = kernel_hardware_entry_offsets(kernels);
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
    KernelScopeRequest request{.entry_offset = kernel->entry_text_offset,
                               .additional_entry_offsets = {}};
    if (kernel->has_kernarg_preload)
      request.additional_entry_offsets.push_back(kernel->kernarg_preload_entry_text_offset);

    auto scope = build_kernel_cfg_scope(blocks, block_index, request, hardware_entries, text);
    if (!scope)
      continue;
    scopes.push_back({.translation = kernel,
                      .entry = scope->entry,
                      .blocks = std::move(scope->blocks),
                      .liveness_edges = std::move(scope->liveness_edges),
                      .call_return_offsets = std::move(scope->call_return_offsets)});
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
  // 2. Decode .text into basic blocks and recover static indirect-branch targets.
  // 3. Compute each kernel's reachable block set. Ordinary CFG successors are
  //    followed directly; call_edges() are followed only to include the callee
  //    body in the current kernel-local scope.
  // 4. Emit each kernel's reachable blocks into a compact, source-ordered body.
  // 5. Translate instructions in that relocated body and append oversized
  //    expansions or descriptor ABI prologues into the kernel-local cave.
  // 6. Patch direct PC-relative branches through the kernel-local placement map.
  // 7. Rewrite recovered indirect-branch address builders in place. Static PC
  //    recovery identifies the original getpc-relative instructions that built
  //    the branch target and records only ranges large enough for the canonical
  //    relocated PC-delta builder. BasicBlock models setpc-style targets as
  //    ordinary CFG successors and validated swappc calls as call_edges(), so
  //    DBT can add return continuations only inside the kernel scope that owns
  //    the call site. After relocation, DBT overwrites the old builder range
  //    with the canonical sequence and pads any leftover words with s_nop.
  // 8. Replace the ELF .text payload and redirect descriptors to their new entries.
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
  auto block_leaders = kernel_block_leaders(descriptor_translations, text);

  // Phase 2: build a CFG over .text, including recovered indirect targets as
  // block leaders, then compute one source-reachable block set per descriptor
  // root. These sets are intentionally kernel-local: if two roots reach the same
  // helper block, Phase 3 emits that helper into both relocated bodies so every
  // branch or call target can be resolved through the current kernel's placement
  // map without borrowing another kernel's return continuation.
  auto blocks = BasicBlock::build(obj, *decoder, guest_arch_, block_leaders);
  const BlockOffsetIndex block_index = build_block_offset_index(blocks);
  auto scopes = kernel_translation_scopes(blocks, block_index, descriptor_translations, text);

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

  auto write_words_at = [](std::vector<uint8_t> &dst, uint64_t offset,
                           std::span<const uint32_t> words) {
    if (words.empty())
      return;
    std::memcpy(dst.data() + offset, words.data(), words.size() * sizeof(uint32_t));
  };

  auto write_launch_stub = [&](KernelTextLayout &layout, uint64_t stub_offset,
                               uint64_t target_offset) {
    uint64_t cursor = stub_offset;
    write_words_at(translated_text, cursor, layout.translation->prologue_words);
    cursor += layout.translation->prologue_words.size() * sizeof(uint32_t);

    const auto branch_dwords = compute_sopp_branch_simm16(cursor, target_offset);
    assert(branch_dwords &&
           "kernarg preload launch stubs are synthesized adjacent to their relocated bodies");
    const uint32_t branch = build_s_branch(*branch_dwords, host_arch_);
    write_words_at(translated_text, cursor, std::span<const uint32_t>(&branch, 1));
  };

  for (const KernelTranslationScope &scope : scopes) {
    if (scope.blocks.empty())
      continue;

    // Phase 3: assign compact target offsets for this kernel before translating
    // instructions. The layout is per-kernel, so source blocks shared by multiple
    // kernels are deliberately duplicated with distinct target offsets. Local
    // cave writes may append bytes, so body placement must be fixed first.
    KernelTextLayout layout;
    layout.translation = scope.translation;
    layout.source_entry = scope.translation->entry_text_offset;
    const bool has_kernarg_preload = scope.translation->has_kernarg_preload;
    const uint64_t source_preload_entry = scope.translation->kernarg_preload_entry_text_offset;
    const uint64_t prologue_bytes = scope.translation->prologue_words.size() * sizeof(uint32_t);
    const uint64_t launch_stub_bytes = prologue_bytes + sizeof(uint32_t);
    if (has_kernarg_preload && launch_stub_bytes > kKernargPreloadSkipBytes) {
      append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                   "kernel descriptor prologue does not fit in the 256-byte kernarg preload "
                   "compatibility window; leaving code object unchanged",
                   layout.source_entry);
      return leave_unchanged();
    }

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

    if (has_kernarg_preload) {
      // Kernarg-preload kernels have two hardware-visible entries:
      //
      //   compat_firmware_entry:
      //     <compatibility preload words>
      //     s_branch common_entry
      //     ... // firmware_entry is 256 bytes after compat_firmware_entry
      //   firmware_entry:
      //     <preloaded-kernarg path words>
      //     s_branch common_entry
      //   common_entry:
      //     ...
      //
      // Compatible CP firmware enters at descriptor entry + 256 when
      // kernarg_preload_spec_length is non-zero. Older/incompatible firmware
      // enters at the descriptor entry. DBT relocates the original source
      // blocks normally, then synthesizes a fresh launch window:
      //
      //   new_compat_firmware_entry:
      //     <descriptor ABI prologue words>
      //     s_branch compat_firmware_entry_translated
      //     ... // new_firmware_entry is exactly 256 bytes later
      //   new_firmware_entry:
      //     <descriptor ABI prologue words>
      //     s_branch firmware_entry_translated
      //
      // The descriptor is redirected to new_compat_firmware_entry. Compatible
      // firmware still adds 256 and lands on new_firmware_entry; older firmware
      // executes the compatibility stub. The original 256-byte source window is
      // no longer preserved as layout padding, because the ABI requirement is
      // the pair of legal entries, not the original physical bytes.
      const uint64_t launch_padding =
          padding_for_residue(translated_text.size(), layout.source_entry % 256, 256);
      append_nop_padding(translated_text, launch_padding, host_arch_);
      layout.target_entry = translated_text.size();
      const uint64_t launch_end =
          layout.target_entry + kKernargPreloadSkipBytes + launch_stub_bytes;
      append_nop_padding(translated_text, launch_end - translated_text.size(), host_arch_);
    } else {
      const uint64_t body_padding =
          padding_for_residue(translated_text.size() + entry_delta, layout.source_entry % 256, 256);
      append_nop_padding(translated_text, body_padding, host_arch_);
    }

    layout.body_begin = translated_text.size();
    uint64_t cursor = layout.body_begin;
    layout.blocks.reserve(scope.blocks.size());
    for (BasicBlock *block : scope.blocks) {
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

    if (has_kernarg_preload) {
      auto preload_body_entry = target_for_source_offset(layout, source_preload_entry);
      if (!preload_body_entry) {
        append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                     "kernarg preload firmware entry offset is not present in the relocated body",
                     source_preload_entry);
        return leave_unchanged();
      }
      write_launch_stub(layout, layout.target_entry, layout.target_body_entry);
      write_launch_stub(layout, layout.target_entry + kKernargPreloadSkipBytes,
                        *preload_body_entry);
    } else if (!scope.translation->prologue_words.empty()) {
      // Descriptor prologues are hardware entry points. Align the cave prologue
      // to the original entry residue, then branch into the relocated body.
      const uint64_t prologue_padding =
          padding_for_residue(translated_text.size(), layout.source_entry % 256, 256);
      append_nop_padding(translated_text, prologue_padding, host_arch_);
      layout.target_entry = translated_text.size();
      append_words(translated_text, scope.translation->prologue_words);

      const uint64_t branch_pc = translated_text.size();
      const auto branch_dwords = compute_sopp_branch_simm16(branch_pc, layout.target_body_entry);
      if (!branch_dwords) {
        append_error(result.diagnostics, DiagnosticKind::ResourceLimit,
                     "kernel descriptor prologue branch range exceeds s_branch simm16; leaving "
                     "code object unchanged",
                     layout.source_entry);
        return leave_unchanged();
      }
      const uint32_t branch = build_s_branch(*branch_dwords, host_arch_);
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
    liveness_options.max_free_vgpr =
        static_cast<uint16_t>(isa_properties(host_arch_).max_addressable_vgprs_per_wf);
    if (options_.debug_min_free_vgpr)
      liveness_options.min_free_vgpr = *options_.debug_min_free_vgpr;
    LivenessAnalysis liveness(KernelBlockScope(scope.blocks), liveness_options,
                              scope.liveness_edges);

    // Phase 4: translate each relocated body instruction. Oversized semantic
    // expansions branch into this kernel's private cave immediately after the body.
    //
    // Return-like s_setpc_b64 instructions are accepted only when they are the
    // terminator of a block reached from a validated call edge in this
    // kernel-local scope.
    const std::unordered_set<uint64_t> &valid_call_return_offsets = scope.call_return_offsets;
    std::unordered_set<uint64_t> recovered_indirect_call_offsets;
    for (const BlockPlacement &placement : layout.blocks) {
      BasicBlock *block = placement.block;
      for (const IndirectCallFixup &source_fixup : block->static_indirect_call_fixups()) {
        IndirectCallFixup fixup = source_fixup;
        auto getpc_target = target_for_source_offset(layout, fixup.source_getpc_offset);
        auto recovery_begin_target =
            target_for_source_offset(layout, fixup.source_recovery_begin_offset);
        auto recovery_end_target =
            target_for_source_offset(layout, fixup.source_recovery_end_offset);
        if (getpc_target && recovery_begin_target && recovery_end_target) {
          fixup.target_getpc_offset = *getpc_target;
          fixup.target_recovery_begin_offset = *recovery_begin_target;
          fixup.target_recovery_end_offset = *recovery_end_target;
          layout.indirect_call_fixups.push_back(fixup);
          recovered_indirect_call_offsets.insert(fixup.source_call_offset);
        }
      }

      uint64_t offset = block->start_offset();
      uint64_t target_offset = placement.target_start;
      for (auto it = block->instructions().begin(); it != block->instructions().end(); ++it) {
        const auto &inst = *it;
        const uint32_t inst_size = inst.size();

        const bool has_recovered_indirect_call = recovered_indirect_call_offsets.contains(offset);
        const bool recovered_indirect_return = valid_call_return_offsets.contains(offset);
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

    // Phase 5: now that the local body and cave have final offsets, rewrite only
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

    std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> rewritten_indirect_recovery_regions;
    for (const IndirectCallFixup &fixup : layout.indirect_call_fixups) {
      auto target_target = target_for_source_offset(layout, fixup.source_target_offset);
      if (!target_target) {
        append_error(result.diagnostics, DiagnosticKind::Legalization,
                     "recovered indirect call target is not present in the kernel-local relocated "
                     "body",
                     fixup.source_call_offset, "indirect branch");
        return leave_unchanged();
      }

      // One recovered getpc/address-recovery region may feed multiple later
      // s_setpc/s_swappc consumers. All consumers are marked as recovered, but
      // the original builder must only be rewritten once. A duplicate begin
      // offset is valid only when it describes the same rewritten byte range
      // and the same relocated target. Anything else means two consumers are
      // trying to use one source builder for different addresses, which cannot
      // be fixed by silently keeping the first rewrite.
      const auto rewrite_key =
          std::pair{fixup.target_recovery_end_offset, static_cast<uint64_t>(*target_target)};
      auto [rewrite_it, inserted] = rewritten_indirect_recovery_regions.emplace(
          fixup.target_recovery_begin_offset, rewrite_key);
      if (!inserted) {
        if (rewrite_it->second != rewrite_key) {
          append_error(result.diagnostics, DiagnosticKind::Legalization,
                       "recovered indirect branch builder is reused for incompatible targets",
                       fixup.source_call_offset, "indirect branch");
          return leave_unchanged();
        }
        continue;
      }

      const int64_t base = static_cast<int64_t>(fixup.target_getpc_offset + sizeof(uint32_t));
      const int64_t delta = static_cast<int64_t>(*target_target) - base;

      std::vector<uint32_t> replacement_words;
      if (!append_pc_delta_builder(replacement_words, host_arch_, fixup.source_call_sreg, delta)) {
        append_error(result.diagnostics, DiagnosticKind::ResourceLimit,
                     "target ISA cannot encode canonical recovered indirect branch builder",
                     fixup.source_call_offset, "indirect branch");
        return leave_unchanged();
      }

      [[maybe_unused]] const uint64_t recovery_size =
          fixup.target_recovery_end_offset - fixup.target_recovery_begin_offset;
      [[maybe_unused]] const uint64_t replacement_size =
          replacement_words.size() * sizeof(uint32_t);
      if (replacement_size > recovery_size) {
        append_error(result.diagnostics, DiagnosticKind::ResourceLimit,
                     "recovered indirect branch builder does not fit in its source range",
                     fixup.source_call_offset, "indirect branch");
        return leave_unchanged();
      }
      assert(replacement_size <= recovery_size &&
             "static PC recovery must only record in-place replaceable builder ranges");

      std::memcpy(translated_text.data() + fixup.target_recovery_begin_offset,
                  replacement_words.data(), replacement_size);
      for (uint64_t off = fixup.target_recovery_begin_offset + replacement_size;
           off < fixup.target_recovery_end_offset; off += sizeof(uint32_t)) {
        const uint32_t nop = build_s_nop(0, host_arch_);
        std::memcpy(translated_text.data() + off, &nop, sizeof(nop));
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
      // scope need the larger register counts; rescanning the whole image would
      // also recompute unrelated kernels and risks mixing diagnostics across
      // scopes.
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

  // Phase 6: write the relocated .text and descriptor entry offsets into the ELF.
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

  if (!patcher.replace_text(translated_text, kMaxTranslatedFileGrowth)) {
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
  const auto fwd_dwords = compute_sopp_branch_simm16(branch_pc, cave_byte_offset);
  if (!fwd_dwords) {
    if (diagnostics_)
      append_error(*diagnostics_, DiagnosticKind::ResourceLimit,
                   "code cave branch range exceeds s_branch simm16; leaving code object unchanged",
                   repl.start_offset);
    return false;
  }

  const uint32_t stub = build_s_branch(*fwd_dwords, host_arch_);
  std::memcpy(text.data() + repl.start_offset, &stub, 4);
  for (uint64_t off = repl.start_offset + 4; off < repl.end_offset; off += 4) {
    const uint32_t nop = build_s_nop(0, host_arch_);
    std::memcpy(text.data() + off, &nop, 4);
  }

  auto cave_words = repl.target_words;
  const uint64_t return_branch_pc = cave_byte_offset + cave_words.size() * sizeof(uint32_t);
  const auto ret_dwords = compute_sopp_branch_simm16(return_branch_pc, stub_next);
  if (!ret_dwords) {
    if (diagnostics_)
      append_error(*diagnostics_, DiagnosticKind::ResourceLimit,
                   "code cave return branch range exceeds s_branch simm16; leaving code object "
                   "unchanged",
                   repl.start_offset);
    return false;
  }
  cave_words.push_back(build_s_branch(*ret_dwords, host_arch_));

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
