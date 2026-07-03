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
#include "rocjitsu/code/patch/kernel_text_layout.h"
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
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rocjitsu {

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

[[nodiscard]] std::vector<uint32_t> raw_words_for_inst(const Instruction &inst) {
  const uint32_t *raw = inst.raw_encoding();
  if (!raw)
    return {};
  return {raw, raw + inst.size() / sizeof(uint32_t)};
}

[[nodiscard]] uint32_t text_word_at(std::span<const uint8_t> text, uint64_t offset) {
  uint32_t word = 0;
  if (offset + sizeof(word) <= text.size())
    std::memcpy(&word, text.data() + offset, sizeof(word));
  return word;
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
};

/// @brief Sorted index from source .text byte offsets to decoded blocks.
///
/// @details DBT relocation repeatedly maps descriptor entries, branch targets,
/// and recovered indirect targets back to the BasicBlock that owns a source
/// offset. Keeping this compact sorted index avoids rebuilding that lookup while
/// preserving BasicBlock ownership in the vector returned by BasicBlock::build().
using BlockOffsetIndex = std::vector<std::pair<uint64_t, BasicBlock *>>;
using BlockPositionIndex = std::unordered_map<const BasicBlock *, size_t>;

[[nodiscard]] BlockOffsetIndex
build_block_offset_index(const std::vector<std::unique_ptr<BasicBlock>> &blocks) {
  BlockOffsetIndex index;
  index.reserve(blocks.size());
  for (const auto &block : blocks) {
    if (block != nullptr)
      index.emplace_back(block->start_offset(), block.get());
  }
  std::ranges::sort(index, {}, &std::pair<uint64_t, BasicBlock *>::first);
  return index;
}

[[nodiscard]] BlockPositionIndex
build_block_position_index(const std::vector<std::unique_ptr<BasicBlock>> &blocks) {
  BlockPositionIndex index;
  index.reserve(blocks.size());
  for (size_t i = 0; i < blocks.size(); ++i) {
    if (blocks[i] != nullptr)
      index.emplace(blocks[i].get(), i);
  }
  return index;
}

[[nodiscard]] BasicBlock *block_for_offset(const BlockOffsetIndex &index, uint64_t offset) {
  auto it = std::ranges::upper_bound(index, offset, std::less<>{},
                                     &std::pair<uint64_t, BasicBlock *>::first);
  if (it == index.begin())
    return nullptr;
  --it;

  BasicBlock *block = it->second;
  if (block == nullptr || offset >= block->end_offset())
    return nullptr;
  return block;
}

[[nodiscard]] std::vector<BasicBlock *>
reachable_kernel_blocks(const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                        const BlockOffsetIndex &block_index,
                        const BlockPositionIndex &block_positions, BasicBlock &entry,
                        const std::unordered_set<uint64_t> &kernel_entries,
                        const std::unordered_set<uint64_t> &own_entries) {
  std::vector<uint8_t> reachable(blocks.size(), 0);
  std::vector<size_t> reached_indices;
  std::vector<size_t> stack;
  auto push_block = [&](BasicBlock *block) {
    auto it = block_positions.find(block);
    if (it != block_positions.end())
      stack.push_back(it->second);
  };
  push_block(&entry);
  for (const uint64_t own_entry : own_entries) {
    if (own_entry == entry.start_offset())
      continue;
    if (BasicBlock *extra_entry = block_for_offset(block_index, own_entry);
        extra_entry != nullptr && extra_entry != &entry) {
      push_block(extra_entry);
    }
  }

  while (!stack.empty()) {
    const size_t block_idx = stack.back();
    stack.pop_back();
    if (block_idx >= blocks.size() || reachable[block_idx])
      continue;
    reachable[block_idx] = 1;
    reached_indices.push_back(block_idx);
    BasicBlock *block = blocks[block_idx].get();
    assert(block != nullptr && "reachable walk stack should contain only decoded blocks");

    for (BasicBlock *succ : block->successors()) {
      assert(succ != nullptr && "BasicBlock successors should never be null");
      if (!own_entries.contains(succ->start_offset()) &&
          kernel_entries.contains(succ->start_offset()))
        continue;
      push_block(succ);
    }
    // Ordinary CFG successors describe control that always follows from the
    // current program counter: fallthroughs, conditional targets, direct branch
    // targets, and recovered non-returning setpc targets. Call edges are tracked
    // separately because a shared callee block can return to different
    // continuations depending on which call site entered it. Reachability for
    // translation still has to include the callee body, but later liveness gets
    // explicit call/return edges rather than treating every possible return as a
    // global CFG successor.
    for (const BasicBlock::CallEdge &call : block->call_edges()) {
      BasicBlock *callee = call.callee;
      assert(callee != nullptr && "BasicBlock call edges should always have a callee");
      if (!own_entries.contains(callee->start_offset()) &&
          kernel_entries.contains(callee->start_offset()))
        continue;
      push_block(callee);
    }
  }

  std::ranges::sort(reached_indices);
  std::vector<BasicBlock *> ordered;
  ordered.reserve(reached_indices.size());
  for (size_t block_idx : reached_indices) {
    if (blocks[block_idx])
      ordered.push_back(blocks[block_idx].get());
  }
  return ordered;
}

[[nodiscard]] std::vector<KernelTranslationScope>
kernel_translation_scopes(const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                          const BlockOffsetIndex &block_index, std::span<KdTranslation> kernels) {
  std::vector<KernelTranslationScope> scopes;
  const auto entries = kernel_entry_offsets(kernels);
  if (entries.empty())
    return scopes;

  const BlockPositionIndex block_positions = build_block_position_index(blocks);
  const auto hardware_entries = kernel_hardware_entry_offsets(kernels);
  std::unordered_set<uint64_t> entry_set(hardware_entries.begin(), hardware_entries.end());
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
    BasicBlock *entry = block_for_offset(block_index, kernel->entry_text_offset);
    if (entry == nullptr)
      continue;
    std::unordered_set<uint64_t> own_entries{kernel->entry_text_offset};
    if (kernel->has_kernarg_preload) {
      if (block_for_offset(block_index, kernel->kernarg_preload_entry_text_offset) == nullptr)
        continue;
      own_entries.insert(kernel->kernarg_preload_entry_text_offset);
    }

    scopes.push_back({kernel, entry,
                      reachable_kernel_blocks(blocks, block_index, block_positions, *entry,
                                              entry_set, own_entries)});
  }
  return scopes;
}

/// @brief Return whether an instruction is an `s_setpc_b64` through one SGPR pair.
///
/// @details Return-like scalar control flow is left as an indirect branch in the
/// translated instruction stream, so DBT must validate that the block terminator
/// reads the call edge's saved return SGPR. This helper intentionally checks the
/// raw SOP1 source field instead of broader instruction semantics: only the exact
/// `s_setpc_b64 s[return:return+1]` form is a scoped call return.
[[nodiscard]] bool s_setpc_from_sreg(const Instruction &inst, uint32_t word, uint16_t ssrc0) {
  if (inst.size() != sizeof(uint32_t) || inst.mnemonic() != "s_setpc_b64")
    return false;
  return static_cast<uint16_t>(word & 0xffu) == ssrc0;
}

/// @brief Find return blocks inside one context-sensitive call target.
///
/// @details Call-like scalar control flow is not represented as a normal CFG
/// edge from the callee back to every possible continuation. The same helper
/// block can be entered by multiple kernels or multiple call sites, and the
/// correct continuation is the one selected by the return SGPR written at that
/// call site. This walk therefore stays inside @p allowed_blocks, follows only
/// ordinary successors within the callee body, and reports terminators that
/// return through @p return_sreg. The caller then pairs each return with the
/// specific continuation from the call edge being analyzed.
[[nodiscard]] std::vector<BasicBlock *>
function_return_blocks(BasicBlock &callee, uint16_t return_sreg, std::span<const uint8_t> text,
                       const std::unordered_set<BasicBlock *> &allowed_blocks) {
  std::vector<BasicBlock *> returns;
  std::vector<BasicBlock *> stack{&callee};
  std::unordered_set<BasicBlock *> visited;

  while (!stack.empty()) {
    BasicBlock *block = stack.back();
    stack.pop_back();
    assert(block != nullptr && "return-block walk stack should contain only decoded blocks");
    if (!allowed_blocks.contains(block) || !visited.insert(block).second)
      continue;

    const Instruction *term = block->terminator();
    assert(term != nullptr && "decoded BasicBlock should contain at least one instruction");
    if (s_setpc_from_sreg(*term, text_word_at(text, term->src_loc()), return_sreg)) {
      returns.push_back(block);
      continue;
    }

    for (BasicBlock *succ : block->successors()) {
      assert(succ != nullptr && "BasicBlock successors should never be null");
      stack.push_back(succ);
    }
  }

  return returns;
}

/// @brief Collect validated return-like terminators for one kernel scope.
///
/// @details Binary translation rejects unresolved indirect branches after CFG
/// construction, but a call-return `s_setpc_b64` is intentionally left as an
/// indirect instruction in the emitted code: its dynamic target is the return PC
/// saved by the matching `s_call_b64` or `s_swappc_b64`. To avoid accepting an
/// arbitrary `s_setpc_b64`, this helper only marks return offsets that are
/// reachable from a `BasicBlock::CallEdge` whose callee and continuation both
/// belong to the current kernel-local scope.
[[nodiscard]] std::unordered_set<uint64_t>
scoped_call_return_offsets(std::span<BasicBlock *const> blocks, std::span<const uint8_t> text) {
  std::unordered_set<BasicBlock *> allowed_blocks;
  allowed_blocks.reserve(blocks.size());
  for (BasicBlock *block : blocks) {
    assert(block != nullptr && "kernel scope should contain only decoded blocks");
    allowed_blocks.insert(block);
  }

  std::unordered_set<uint64_t> returns;
  for (BasicBlock *block : blocks) {
    assert(block != nullptr && "kernel scope should contain only decoded blocks");
    for (const BasicBlock::CallEdge &call : block->call_edges()) {
      assert(call.callee != nullptr && "BasicBlock call edges should always have a callee");
      assert(call.continuation != nullptr &&
             "BasicBlock call edges should always have a continuation");
      if (!allowed_blocks.contains(call.callee) || !allowed_blocks.contains(call.continuation))
        continue;

      for (BasicBlock *return_block :
           function_return_blocks(*call.callee, call.return_sreg, text, allowed_blocks)) {
        const Instruction *term = return_block->terminator();
        assert(term != nullptr && "function_return_blocks returns non-empty decoded blocks");
        returns.insert(term->src_loc());
      }
    }
  }
  return returns;
}

/// @brief Materialize context-sensitive call edges for liveness.
///
/// @details `BasicBlock` deliberately separates call edges from ordinary CFG
/// successors. The translator still needs liveness to see the effects of a
/// call: values live into the callee are used by the callee, and values live
/// after the call continuation must be live at each validated return block.
/// This helper converts each scoped call edge into temporary analysis edges
/// `caller -> callee` and `return -> continuation` without mutating the CFG or
/// creating cross-kernel return edges.
[[nodiscard]] std::vector<ScopedCfgEdge>
scoped_call_liveness_edges(std::span<BasicBlock *const> blocks, std::span<const uint8_t> text) {
  std::unordered_set<BasicBlock *> allowed_blocks;
  allowed_blocks.reserve(blocks.size());
  for (BasicBlock *block : blocks) {
    assert(block != nullptr && "kernel scope should contain only decoded blocks");
    allowed_blocks.insert(block);
  }

  std::vector<ScopedCfgEdge> edges;
  for (BasicBlock *block : blocks) {
    assert(block != nullptr && "kernel scope should contain only decoded blocks");
    for (const BasicBlock::CallEdge &call : block->call_edges()) {
      assert(call.callee != nullptr && "BasicBlock call edges should always have a callee");
      assert(call.continuation != nullptr &&
             "BasicBlock call edges should always have a continuation");
      if (!allowed_blocks.contains(call.callee) || !allowed_blocks.contains(call.continuation))
        continue;

      edges.push_back({.from = block, .to = call.callee});
      for (BasicBlock *return_block :
           function_return_blocks(*call.callee, call.return_sreg, text, allowed_blocks)) {
        edges.push_back({.from = return_block, .to = call.continuation});
      }
    }
  }

  return edges;
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
  // 5. Translate instructions with a forward cursor; oversized replacements grow
  //    the current block and recovered indirect transfers reserve patch windows.
  // 6. Patch direct PC-relative branches through the kernel-local placement map.
  // 7. Patch recovered indirect branches/calls either at their reserved transfer
  //    windows or, for multi-target consumers, by rewriting the recovered
  //    source-side PC builders once per distinct builder range. BasicBlock
  //    models setpc-style targets as ordinary CFG successors and validated
  //    swappc calls as call_edges(), so DBT can add return continuations only
  //    inside the kernel scope that owns the call site.
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
  auto scopes = kernel_translation_scopes(blocks, block_index, descriptor_translations);

  if (scopes.size() != entry_offsets.size()) {
    append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                 "kernel descriptor entry offsets are required to map to decoded text blocks");
    return leave_unchanged();
  }

  std::vector<uint8_t> translated_text;
  translated_text.reserve(text.size());
  const bool continue_after_failure = options_.debug_continue_after_failure;

  struct PendingTrace {
    uint64_t source_offset = 0;
    uint32_t source_size = 0;
    std::vector<uint32_t> source_words;
    const InstructionLegalization *legalization = nullptr;
    bool copied_original = false;
    bool semantic_lowering = false;
    bool changed = false;
    uint64_t target_offset = 0;
    std::vector<uint32_t> target_words;
  };

  auto queue_trace = [&](std::vector<PendingTrace> &pending, const Instruction &inst,
                         uint64_t offset, const InstructionLegalization *leg, bool copied_original,
                         bool semantic_lowering, bool changed, uint64_t target_offset,
                         std::vector<uint32_t> target_words) {
    if (!trace_callback_)
      return;
    pending.push_back({.source_offset = offset,
                       .source_size = static_cast<uint32_t>(inst.size()),
                       .source_words = raw_words_for_inst(inst),
                       .legalization = leg,
                       .copied_original = copied_original,
                       .semantic_lowering = semantic_lowering,
                       .changed = changed,
                       .target_offset = target_offset,
                       .target_words = std::move(target_words)});
  };

  auto flush_traces = [&](std::vector<PendingTrace> &pending, uint64_t target_delta) {
    if (!trace_callback_)
      return;
    for (PendingTrace &trace : pending) {
      trace_callback_({.source_offset = trace.source_offset,
                       .source_size = trace.source_size,
                       .source_words = trace.source_words,
                       .legalization = trace.legalization,
                       .copied_original = trace.copied_original,
                       .semantic_lowering = trace.semantic_lowering,
                       .changed = trace.changed,
                       .emitted_in_cave = false,
                       .target_offset = trace.target_offset + target_delta,
                       .target_words = trace.target_words});
    }
  };

  auto copy_original_instruction = [&](const Instruction &inst, uint64_t offset,
                                       std::vector<uint8_t> &kernel_text,
                                       std::vector<PendingTrace> &pending_traces) {
    const uint32_t inst_size = inst.size();
    const uint64_t target_offset = kernel_text.size();
    const auto *words = reinterpret_cast<const uint32_t *>(text.data() + offset);
    std::vector<uint32_t> copied_words(words, words + inst_size / sizeof(uint32_t));
    append_words(kernel_text, copied_words);
    // Continued-failure mode is diagnostic-only. Emit an explicit copy event so
    // diff reports make it clear which failed source instruction was preserved.
    queue_trace(pending_traces, inst, offset, nullptr, true, false, false, target_offset,
                std::move(copied_words));
  };

  auto continue_after_instruction_error = [&](const Instruction &inst, uint64_t offset,
                                              std::vector<uint8_t> &kernel_text,
                                              std::vector<PendingTrace> &pending_traces) {
    if (!continue_after_failure)
      return false;
    copy_original_instruction(inst, offset, kernel_text, pending_traces);
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

  auto relocation_diagnostic_kind = [](const TextRelocationResult &relocation) {
    if (relocation.message.find("range") != std::string::npos ||
        relocation.message.find("exceeds") != std::string::npos ||
        relocation.message.find("cannot encode") != std::string::npos ||
        relocation.message.find("outside translated") != std::string::npos) {
      return DiagnosticKind::ResourceLimit;
    }
    return DiagnosticKind::Legalization;
  };

  for (const KernelTranslationScope &scope : scopes) {
    if (scope.blocks.empty())
      continue;

    // Phase 3: translate this kernel into a temporary, source-ordered body. The
    // body starts at offset zero while it is being built; after final padding and
    // any launch window are chosen, every recorded target offset is rebased into
    // the output .text. This lets instruction expansions change block sizes
    // without precomputing speculative side-region offsets.
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

    TranslationContext kernel_context(
        scope.translation->target_vgpr_count, scope.translation->target_agpr_count,
        scope.translation->target_accvgpr_base, scope.translation->target_sgpr_count);
    LivenessAnalysisOptions liveness_options;
    if (options_.debug_min_free_vgpr)
      liveness_options.min_free_vgpr = *options_.debug_min_free_vgpr;
    const auto liveness_edges = scoped_call_liveness_edges(KernelBlockScope(scope.blocks), text);
    LivenessAnalysis liveness(KernelBlockScope(scope.blocks), liveness_options, liveness_edges);

    // Phase 4: translate each relocated body instruction at the current cursor.
    // Return-like s_setpc_b64 instructions are accepted only when they are the
    // terminator of a block reached from a validated call edge in this
    // kernel-local scope. Recovered indirect setpc/swappc consumers reserve a
    // fixed maximum-size window when recovery proves one effective target. When
    // one dynamic consumer has multiple recovered targets, no single direct
    // window can preserve semantics; DBT keeps the original indirect consumer
    // and asks the patch layer to rewrite each source-side PC builder once.
    const std::unordered_set<uint64_t> valid_call_return_offsets =
        scoped_call_return_offsets(KernelBlockScope(scope.blocks), text);
    struct RecoveredConsumer {
      std::vector<IndirectCallFixup> fixups;
      bool use_transfer_window = false;
      IndirectCallFixup window_fixup;
    };
    std::unordered_map<uint64_t, RecoveredConsumer> recovered_indirect_by_call;
    for (BasicBlock *block : scope.blocks) {
      for (const IndirectCallFixup &source_fixup : block->static_indirect_call_fixups()) {
        recovered_indirect_by_call[source_fixup.source_call_offset].fixups.push_back(source_fixup);
      }
    }

    std::vector<IndirectCallFixup> pending_builder_fixups;
    for (auto &[source_call_offset, consumer] : recovered_indirect_by_call) {
      if (consumer.fixups.empty())
        continue;

      const IndirectCallFixup &first = consumer.fixups.front();
      bool single_effective_target = true;
      for (const IndirectCallFixup &fixup : consumer.fixups) {
        if (fixup.source_call_sreg != first.source_call_sreg ||
            fixup.source_is_call != first.source_is_call ||
            fixup.source_return_sreg != first.source_return_sreg) {
          append_error(result.diagnostics, DiagnosticKind::Legalization,
                       "recovered indirect branch has inconsistent consumer metadata",
                       source_call_offset, "indirect branch");
          return leave_unchanged();
        }
        if (fixup.source_target_offset != first.source_target_offset)
          single_effective_target = false;
      }

      if (single_effective_target) {
        consumer.use_transfer_window = true;
        consumer.window_fixup = first;
      } else {
        pending_builder_fixups.insert(pending_builder_fixups.end(), consumer.fixups.begin(),
                                      consumer.fixups.end());
      }
    }

    std::vector<uint8_t> kernel_text;
    std::vector<PendingTrace> pending_traces;
    uint64_t source_body_size = 0;
    for (BasicBlock *block : scope.blocks)
      source_body_size += block->size();
    const uint64_t recovered_window_growth =
        recovered_indirect_by_call.size() * kMaxRecoveredIndirectTransferWords * sizeof(uint32_t);
    kernel_text.reserve(static_cast<size_t>(std::min<uint64_t>(
        source_body_size + recovered_window_growth, std::numeric_limits<size_t>::max())));

    std::unordered_set<uint64_t> needed_builder_source_offsets;
    needed_builder_source_offsets.reserve(pending_builder_fixups.size() * 3);
    for (const IndirectCallFixup &fixup : pending_builder_fixups) {
      needed_builder_source_offsets.insert(fixup.source_getpc_offset);
      needed_builder_source_offsets.insert(fixup.source_recovery_begin_offset);
      needed_builder_source_offsets.insert(fixup.source_recovery_end_offset);
    }
    std::unordered_map<uint64_t, uint64_t> target_offset_by_source_offset;
    target_offset_by_source_offset.reserve(needed_builder_source_offsets.size());
    layout.body_begin = 0;
    layout.blocks.reserve(scope.blocks.size());
    for (BasicBlock *block : scope.blocks) {
      BlockPlacement placement{.block = block,
                               .source_start = block->start_offset(),
                               .source_end = block->end_offset(),
                               .target_start = kernel_text.size(),
                               .target_end = kernel_text.size()};

      for (auto it = block->instructions().begin(); it != block->instructions().end(); ++it) {
        const auto &inst = *it;
        const uint64_t offset = inst.src_loc();
        const uint64_t target_offset = kernel_text.size();
        const uint32_t inst_size = inst.size();
        if (needed_builder_source_offsets.contains(offset))
          target_offset_by_source_offset.emplace(offset, target_offset);

        const auto recovered_it = recovered_indirect_by_call.find(offset);
        const bool has_recovered_indirect_call = recovered_it != recovered_indirect_by_call.end();
        const bool recovered_indirect_return = valid_call_return_offsets.contains(offset);
        const auto direct_branch_delta = inst.branch_offset_bytes();
        if ((inst.flags() & (INDIRECT_BRANCH | INDIRECT_CALL)) != 0 &&
            !has_recovered_indirect_call && !recovered_indirect_return && !direct_branch_delta) {
          append_error(result.diagnostics, DiagnosticKind::Legalization,
                       "indirect branch or call target recovery is not implemented for relocated "
                       "kernel text",
                       offset, std::string(inst.mnemonic()));
          if (continue_after_instruction_error(inst, offset, kernel_text, pending_traces)) {
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
            if (continue_after_instruction_error(inst, offset, kernel_text, pending_traces)) {
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

        if (has_recovered_indirect_call && recovered_it->second.use_transfer_window) {
          const IndirectCallFixup &source_fixup = recovered_it->second.window_fixup;
          layout.recovered_indirect_fixups.push_back(
              {.source_call_offset = source_fixup.source_call_offset,
               .source_target_offset = source_fixup.source_target_offset,
               .target_window_offset = target_offset,
               .target_sreg = source_fixup.source_call_sreg,
               .return_sreg = source_fixup.source_return_sreg,
               .is_call = source_fixup.source_is_call});
          append_nop_padding(kernel_text, kMaxRecoveredIndirectTransferWords * sizeof(uint32_t),
                             host_arch_);
          continue;
        }

        const uint32_t *raw = inst.raw_encoding();
        if (!raw) {
          copy_original_instruction(inst, offset, kernel_text, pending_traces);
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
            if (continue_after_instruction_error(inst, offset, kernel_text, pending_traces)) {
              continue;
            }
            return leave_unchanged();
          }

          if (expansion.status == ExpandStatus::Success) {
            std::vector<uint32_t> target_words = std::move(expansion.words);
            append_words(kernel_text, target_words);
            queue_trace(pending_traces, inst, offset, leg, false, true, true, target_offset,
                        std::move(target_words));
            continue;
          }
        }

        if (leg && leg->action == Action::Expand) {
          append_error(result.diagnostics, DiagnosticKind::ExpandMissing,
                       "legalization requires EXPAND, but no expansion rule is implemented", offset,
                       std::string(inst.mnemonic()),
                       {"Add a semantic expansion rule for this mnemonic."});
          if (continue_after_instruction_error(inst, offset, kernel_text, pending_traces)) {
            continue;
          }
          return leave_unchanged();
        }

        bool copied_original = false;
        bool changed = false;
        std::vector<uint32_t> target_words;
        if (!handle_encoding(inst, offset, kernel_text, dst_opcode, text,
                             trace_callback_ != nullptr, copied_original, changed, target_words)) {
          if (continue_after_instruction_error(inst, offset, kernel_text, pending_traces)) {
            continue;
          }
          return leave_unchanged();
        }
        queue_trace(pending_traces, inst, offset, leg, copied_original, false, changed,
                    target_offset, std::move(target_words));
      }
      placement.target_end = kernel_text.size();
      layout.blocks.push_back(placement);
      if (needed_builder_source_offsets.contains(block->end_offset()))
        target_offset_by_source_offset.emplace(block->end_offset(), kernel_text.size());
    }
    layout.body_end = kernel_text.size();

    if (continue_after_failure && has_error_diagnostic(result.diagnostics))
      continue;

    for (IndirectCallFixup fixup : pending_builder_fixups) {
      const auto getpc_it = target_offset_by_source_offset.find(fixup.source_getpc_offset);
      const auto begin_it = target_offset_by_source_offset.find(fixup.source_recovery_begin_offset);
      const auto end_it = target_offset_by_source_offset.find(fixup.source_recovery_end_offset);
      if (getpc_it == target_offset_by_source_offset.end() ||
          begin_it == target_offset_by_source_offset.end() ||
          end_it == target_offset_by_source_offset.end()) {
        append_error(result.diagnostics, DiagnosticKind::Legalization,
                     "recovered indirect branch builder is not fully present in the relocated body",
                     fixup.source_call_offset, "indirect branch");
        return leave_unchanged();
      }

      fixup.target_getpc_offset = getpc_it->second;
      fixup.target_recovery_begin_offset = begin_it->second;
      fixup.target_recovery_end_offset = end_it->second;
      layout.recovered_builder_fixups.push_back(fixup);
    }

    auto body_entry = target_for_source_offset(layout, layout.source_entry);
    if (!body_entry) {
      append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                   "kernel descriptor entry offset is not present in the relocated body",
                   layout.source_entry);
      return leave_unchanged();
    }
    layout.target_body_entry = *body_entry;

    std::optional<uint64_t> preload_body_entry;
    if (has_kernarg_preload) {
      preload_body_entry = target_for_source_offset(layout, source_preload_entry);
      if (!preload_body_entry) {
        append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                     "kernarg preload firmware entry offset is not present in the relocated body",
                     source_preload_entry);
        return leave_unchanged();
      }
    }

    uint64_t target_delta = 0;
    if (has_kernarg_preload) {
      // Kernarg-preload kernels have two hardware-visible entries separated by
      // exactly 256 bytes. Reserve that launch window before appending the body;
      // the stubs are written after the body offsets have been rebased.
      const uint64_t launch_padding =
          padding_for_residue(translated_text.size(), layout.source_entry % 256, 256);
      append_nop_padding(translated_text, launch_padding, host_arch_);
      layout.target_entry = translated_text.size();
      const uint64_t launch_end =
          layout.target_entry + kKernargPreloadSkipBytes + launch_stub_bytes;
      append_nop_padding(translated_text, launch_end - translated_text.size(), host_arch_);
      target_delta = translated_text.size();
    } else {
      const uint64_t body_padding = padding_for_residue(
          translated_text.size() + layout.target_body_entry, layout.source_entry % 256, 256);
      append_nop_padding(translated_text, body_padding, host_arch_);
      target_delta = translated_text.size();
    }

    rebase_kernel_text_layout(layout, target_delta);
    translated_text.insert(translated_text.end(), kernel_text.begin(), kernel_text.end());

    if (has_kernarg_preload) {
      assert(preload_body_entry && "preload body entry was checked before rebase");
      write_launch_stub(layout, layout.target_entry, layout.target_body_entry);
      write_launch_stub(layout, layout.target_entry + kKernargPreloadSkipBytes,
                        *preload_body_entry + target_delta);
    } else if (!scope.translation->prologue_words.empty()) {
      // Descriptor prologues are hardware entry points. Align the prologue to the
      // original entry residue, then branch into the relocated body. The prologue
      // lives in .text but is not part of the source block placement map.
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
    } else {
      layout.target_entry = layout.target_body_entry;
    }

    // Phase 5: now that every emitted source block has a final target offset,
    // patch explicit direct branches, recovered source-side builders, and
    // recovered indirect transfer windows.
    if (auto patched = patch_direct_branch_fixups(translated_text, layout, host_arch_);
        !patched.ok) {
      append_error(result.diagnostics, relocation_diagnostic_kind(patched), patched.message,
                   patched.source_offset);
      return leave_unchanged();
    }

    if (auto patched = patch_recovered_builder_fixups(translated_text, layout, host_arch_);
        !patched.ok) {
      append_error(result.diagnostics, relocation_diagnostic_kind(patched), patched.message,
                   patched.source_offset, "indirect branch");
      return leave_unchanged();
    }

    if (auto patched = patch_recovered_indirect_fixups(translated_text, layout, host_arch_);
        !patched.ok) {
      append_error(result.diagnostics, relocation_diagnostic_kind(patched), patched.message,
                   patched.source_offset, "indirect branch");
      return leave_unchanged();
    }

    flush_traces(pending_traces, target_delta);

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
  // unreachable bytes. Keep ELF mutation one-sided by padding the relocated
  // .text back to at least the original size.
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
  result.elf_bytes = std::move(patcher).emit();
  return result;
}

bool BinaryTranslator::handle_encoding(const Instruction &inst, uint64_t offset,
                                       std::vector<uint8_t> &text, uint16_t dst_opcode,
                                       std::span<const uint8_t> orig_text, bool collect_trace_words,
                                       bool &copied_original, bool &changed,
                                       std::vector<uint32_t> &target_words) {
  const uint32_t *raw = inst.raw_encoding();
  assert(raw && "handle_encoding called without raw encoding");
  copied_original = false;
  changed = false;
  if (collect_trace_words)
    target_words.clear();

  if (!encoding_translate_) {
    copied_original = true;
    const size_t word_count = inst.size() / sizeof(uint32_t);
    if (collect_trace_words)
      target_words.assign(raw, raw + word_count);
    append_words(text, std::span<const uint32_t>(raw, word_count));
    return true;
  }

  const uint32_t w0 = raw[0];
  const uint32_t w1 = inst.size() > 4 ? raw[1] : 0;
  const uint32_t w2 = inst.size() > 8 ? raw[2] : 0;

  auto tr = encoding_translate_(inst.encoding_id(), w0, w1, w2, dst_opcode);

  if (tr.word_count == 0) {
    copied_original = true;
    const size_t word_count = inst.size() / sizeof(uint32_t);
    if (collect_trace_words)
      target_words.assign(raw, raw + word_count);
    append_words(text, std::span<const uint32_t>(raw, word_count));
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

  append_words(text, std::span<const uint32_t>(tr.words, tr.word_count));
  if (collect_trace_words) {
    target_words.assign(tr.words, tr.words + tr.word_count);
    changed = words_changed(raw_words_for_inst(inst), target_words);
  }
  return true;
}

} // namespace rocjitsu
