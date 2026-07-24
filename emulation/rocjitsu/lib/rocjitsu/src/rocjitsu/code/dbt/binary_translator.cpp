// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/dbt/binary_translator.h"

#include "rocjitsu/analysis/def_use_chain.h"
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
#include "rocjitsu/code/dbt/lds_virtualization.h"
#include "rocjitsu/code/dbt/legalization/gfx1250_b0_to_a0.h"
#include "rocjitsu/code/dbt/semantic_translator.h"
#include "rocjitsu/code/dbt/virtual_lds.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/code/patch/kernarg_extension.h"
#include "rocjitsu/code/patch/kernel_text_layout.h"
#include "rocjitsu/code/patch/sidecar_metadata.h"
#include "rocjitsu/code/relocation_function_table.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/isa_properties.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/isa_traits.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rocjitsu {

namespace {

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

/// @brief Recognize clang's gfx1250 body for an unreachable kernel specialization.
///
/// @details rocPRIM emits one trampoline specialization per supported target.
/// Specializations which cannot be selected for the compiled device end in
/// `__builtin_unreachable()`. Current clang lowers those empty gfx1250 bodies to
/// one 64-bit `s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1`
/// instruction, followed by zero alignment padding. WAVE_MODE[25] is
/// REPLAY_MODE; the instruction is ordinary prologue state rather than a CFG
/// terminator, but the source function has no defined fallthrough execution.
///
/// Keep the match deliberately exact. Other reachable fallthrough into opaque
/// text remains a hard error, including a longer block which merely ends with
/// this instruction.
[[nodiscard]] bool is_gfx1250_compiler_unreachable_stub(const BasicBlock &block,
                                                        rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_GFX1250 || !block.falls_through_to_undecodable_text() ||
      block.num_instructions() != 1 || block.size() != 2 * sizeof(uint32_t))
    return false;

  const Instruction *inst = block.terminator();
  if (inst == nullptr || inst->mnemonic() != "s_setreg_imm32_b32" || inst->size() != 8)
    return false;
  const uint32_t *words = inst->raw_encoding();
  return words != nullptr && words[0] == 0xb9800641u && words[1] == 1u;
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

void append_warning(std::vector<TranslationDiagnostic> &diagnostics, DiagnosticKind kind,
                    std::string message, std::optional<uint64_t> guest_offset = std::nullopt,
                    std::string mnemonic = {}, std::vector<std::string> required_work = {}) {
  append_diagnostic(diagnostics, DiagnosticSeverity::Warning, kind, std::move(message),
                    guest_offset, std::move(mnemonic), std::move(required_work));
}

void append_diagnostics(std::vector<TranslationDiagnostic> &dst,
                        const std::vector<TranslationDiagnostic> &src) {
  dst.insert(dst.end(), src.begin(), src.end());
}

/// @brief Return a human-readable kernel label for diagnostics.
///
/// @details Some code objects carry empty kernel symbol names. Falling back to
/// the source .text entry offset keeps skip/failure diagnostics useful for
/// debugging because the user can still identify which code-object entry failed.
[[nodiscard]] std::string kernel_label(const KdTranslation &translation) {
  if (!translation.kernel_name.empty())
    return translation.kernel_name;

  std::ostringstream os;
  os << ".text+0x" << std::hex << translation.entry_text_offset;
  return os.str();
}

[[nodiscard]] uint32_t max_descriptor_sgpr_allocation_for_long_branch(rj_code_arch_t arch) {
  // Long direct branches consume their scratch pair at the final
  // s_setpc_b64/s_swappc_b64 transfer, so DBT may only use a pair that can be
  // made descriptor-backed for the destination kernel.
  return arch_descriptor_sgpr_allocation_limit(arch);
}

/// @brief Find the next even SGPR pair that can be descriptor-backed for a branch thunk.
[[nodiscard]] std::optional<uint16_t> next_long_branch_sgpr_pair(const TranslationContext &context,
                                                                 rj_code_arch_t arch) {
  const uint32_t current = std::max(context.num_sgprs, context.required_sgpr_count);
  const uint32_t base = (current + 1u) & ~1u;
  if (base > 126)
    return std::nullopt;

  const uint32_t max_descriptor_sgprs = max_descriptor_sgpr_allocation_for_long_branch(arch);
  if (max_descriptor_sgprs != 0 && base + 2 > max_descriptor_sgprs)
    return std::nullopt;
  return static_cast<uint16_t>(base);
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
    if (kernel.has_kernarg_preload_firmware_skip)
      offsets.push_back(kernel.kernarg_preload_firmware_entry_text_offset);
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
    if (kernel.has_kernarg_preload_firmware_skip &&
        kernel.kernarg_preload_firmware_entry_text_offset < text.size())
      offsets.push_back(kernel.kernarg_preload_firmware_entry_text_offset);
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

/// @brief Descriptor state mutated by one kernel-scope translation transaction.
struct DescriptorVariantCheckpoint {
  size_t index = 0;
  KdTranslation translation;
};

[[nodiscard]] uint64_t kernel_scope_key(const KdTranslation &kernel) {
  assert(kernel.entry_text_offset <= (std::numeric_limits<uint64_t>::max() >> 1) &&
         "kernel entry offset is too large to pack with variant bit");
  return (kernel.entry_text_offset << 1) | (kernel.needs_lds_overflow_buf ? 1u : 0u);
}

[[nodiscard]] bool same_kernel_scope_variant(const KdTranslation &lhs, const KdTranslation &rhs) {
  return lhs.entry_text_offset == rhs.entry_text_offset &&
         lhs.needs_lds_overflow_buf == rhs.needs_lds_overflow_buf;
}

[[nodiscard]] std::vector<DescriptorVariantCheckpoint>
checkpoint_scope_descriptors(std::span<const KdTranslation> translations,
                             const KdTranslation &scope_translation) {
  std::vector<DescriptorVariantCheckpoint> checkpoint;
  for (size_t i = 0; i < translations.size(); ++i) {
    if (same_kernel_scope_variant(translations[i], scope_translation))
      checkpoint.push_back({.index = i, .translation = translations[i]});
  }
  return checkpoint;
}

[[nodiscard]] size_t kernel_translation_scope_count(std::span<const KdTranslation> kernels) {
  std::unordered_set<uint64_t> keys;
  for (const KdTranslation &kernel : kernels)
    keys.insert(kernel_scope_key(kernel));
  return keys.size();
}

[[nodiscard]] bool scope_uses_virtualizable_lds(const KernelTranslationScope &scope,
                                                rj_code_arch_t guest_arch,
                                                rj_code_arch_t host_arch) {
  if (scope.translation == nullptr)
    return false;
  if (scope.translation->target_lds_size != 0)
    return true;

  for (BasicBlock *block : scope.blocks) {
    if (block == nullptr)
      continue;
    for (const Instruction &inst : block->instructions()) {
      if (source_instruction_uses_virtualizable_lds(inst, guest_arch, host_arch))
        return true;
    }
  }
  return false;
}

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

[[nodiscard]] std::unordered_set<uint64_t>
attach_relocation_table_call_edges(const BlockOffsetIndex &block_index,
                                   std::span<const RelocationFunctionTable> tables,
                                   std::span<const RelocationTableDispatch> dispatches) {
  std::unordered_set<uint64_t> accepted_calls;
  for (const RelocationTableDispatch &dispatch : dispatches) {
    if (dispatch.table_index >= tables.size())
      continue;
    BasicBlock *source = block_for_offset(block_index, dispatch.source_call_offset);
    if (source == nullptr || source->terminator() == nullptr ||
        source->terminator()->src_loc() != dispatch.source_call_offset)
      continue;
    BasicBlock *continuation = block_for_offset(block_index, source->end_offset());
    if (continuation == nullptr || continuation->start_offset() != source->end_offset())
      continue;

    std::vector<BasicBlock *> callees;
    callees.reserve(tables[dispatch.table_index].entries.size());
    bool complete = true;
    for (const RelocationFunctionPointer &entry : tables[dispatch.table_index].entries) {
      BasicBlock *callee = block_for_offset(block_index, entry.target_text_offset);
      if (callee == nullptr || callee->start_offset() != entry.target_text_offset) {
        complete = false;
        break;
      }
      callees.push_back(callee);
    }
    if (!complete || callees.empty())
      continue;

    for (BasicBlock *callee : callees) {
      source->add_call_edge({.kind = BasicBlock::CallEdgeKind::IndirectSwapPc,
                             .callee = callee,
                             .continuation = continuation,
                             .source_call_offset = dispatch.source_call_offset,
                             .return_sreg = dispatch.return_sreg});
    }
    accepted_calls.insert(dispatch.source_call_offset);
  }
  return accepted_calls;
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
  std::unordered_set<uint64_t> seen_scopes;
  for (KdTranslation &kernel : kernels) {
    if (seen_scopes.insert(kernel_scope_key(kernel)).second)
      ordered_kernels.push_back(&kernel);
  }

  std::ranges::sort(ordered_kernels, [](const auto *lhs, const auto *rhs) {
    if (lhs->entry_text_offset != rhs->entry_text_offset)
      return lhs->entry_text_offset < rhs->entry_text_offset;
    return lhs->needs_lds_overflow_buf < rhs->needs_lds_overflow_buf;
  });

  scopes.reserve(ordered_kernels.size());
  for (KdTranslation *kernel : ordered_kernels) {
    BasicBlock *entry = block_for_offset(block_index, kernel->entry_text_offset);
    if (entry == nullptr)
      continue;
    std::unordered_set<uint64_t> own_entries{kernel->entry_text_offset};
    if (kernel->has_kernarg_preload_firmware_skip) {
      if (block_for_offset(block_index, kernel->kernarg_preload_firmware_entry_text_offset) ==
          nullptr)
        continue;
      own_entries.insert(kernel->kernarg_preload_firmware_entry_text_offset);
    }

    scopes.push_back({kernel, entry,
                      reachable_kernel_blocks(blocks, block_index, block_positions, *entry,
                                              entry_set, own_entries)});
  }
  return scopes;
}

/// @brief Return whether an instruction is a scalar set-PC through one SGPR pair.
///
/// @details Return-like scalar control flow is left as an indirect branch in the
/// translated instruction stream, so DBT must validate that the block terminator
/// reads the call edge's saved return SGPR. This helper intentionally checks the
/// raw SOP1 source field instead of broader instruction semantics: only the exact
/// `s_setpc_b64`/`s_set_pc_i64 s[return:return+1]` form is a scoped call return.
[[nodiscard]] bool s_setpc_from_sreg(const Instruction &inst, uint32_t word, uint16_t ssrc0) {
  const std::string_view mnemonic = inst.mnemonic();
  if (inst.size() != sizeof(uint32_t) || (mnemonic != "s_setpc_b64" && mnemonic != "s_set_pc_i64"))
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

/// @brief Commit translated text, descriptor plans, and runtime metadata to one ELF image.
///
/// @details BinaryTranslator owns analysis and per-kernel lowering. This helper
/// owns the separate commit responsibility: applying completed descriptor plans,
/// replacing `.text`, appending sidecar descriptors, resolving their final virtual
/// addresses, and serializing runtime metadata. It mutates only its private patcher
/// copy, so any failure leaves the caller free to return the original code object.
[[nodiscard]] std::optional<std::vector<uint8_t>> materialize_translated_code_object(
    CodeObjectPatcher patcher, std::vector<uint8_t> translated_text, uint64_t original_text_size,
    std::span<const TextOffsetRelocation> text_relocations,
    std::span<const PcRelativeDataRelocation> data_relocations,
    std::span<const KdTranslation> translations, rj_code_arch_t host_arch, uint32_t target_mach,
    std::vector<TranslationDiagnostic> &diagnostics) {
  if (translated_text.size() < original_text_size)
    append_nop_padding(translated_text, original_text_size - translated_text.size(), host_arch);

  std::unordered_set<uint64_t> applied_descriptors;
  for (const KdTranslation &translation : translations) {
    if (translation.sidecar_descriptor)
      continue;
    if (!applied_descriptors.insert(translation.descriptor_file_offset).second)
      continue;
    if (!patcher.apply_kernel_descriptor_translation(translation, host_arch)) {
      append_error(
          diagnostics, DiagnosticKind::KernelDescriptor,
          translation.skipped
              ? "skipped kernel descriptor could not be patched to a target stub safely; leaving "
                "code object unchanged"
              : "kernel descriptor translation could not be applied safely; leaving code object "
                "unchanged");
      return std::nullopt;
    }
  }

  if (!patcher.replace_text(translated_text, text_relocations, data_relocations)) {
    append_error(diagnostics, DiagnosticKind::ResourceLimit,
                 "relocated .text could not be materialized safely; leaving code object unchanged");
    return std::nullopt;
  }

  std::vector<uint64_t> sidecar_descriptor_vaddrs(translations.size(), 0);
  std::vector<KdTranslation> sidecar_descriptors;
  std::vector<size_t> sidecar_indices;
  for (size_t i = 0; i < translations.size(); ++i) {
    const KdTranslation &translation = translations[i];
    if (!translation.sidecar_descriptor || !translation.needs_lds_overflow_buf ||
        translation.skipped)
      continue;
    sidecar_descriptors.push_back(translation);
    sidecar_indices.push_back(i);
  }
  if (!sidecar_descriptors.empty()) {
    auto appended =
        patcher.append_sidecar_descriptor_translations(sidecar_descriptors, host_arch, 64);
    if (!appended || appended->size() != sidecar_descriptors.size()) {
      append_error(diagnostics, DiagnosticKind::ResourceLimit,
                   "virtual LDS sidecar descriptors could not be materialized safely; leaving code "
                   "object unchanged");
      return std::nullopt;
    }
    for (size_t i = 0; i < appended->size(); ++i)
      sidecar_descriptor_vaddrs[sidecar_indices[i]] = (*appended)[i].vaddr;
  }

  const auto patched_image = patcher.image_bytes();
  AmdGpuCodeObject patched_layout(patched_image.data(), patched_image.size());
  if (!patched_layout.is_valid()) {
    append_error(diagnostics, DiagnosticKind::ResourceLimit,
                 "relocated ELF could not be reparsed for runtime metadata; leaving code object "
                 "unchanged");
    return std::nullopt;
  }

  std::vector<SidecarVariantMetadata> sidecar_metadata;
  std::vector<KernargExtensionMetadata> kernarg_extension_metadata;
  std::vector<VirtualLdsKernelMetadata> virtual_lds_metadata;
  for (size_t i = 0; i < translations.size(); ++i) {
    const KdTranslation &translation = translations[i];
    if (!translation.sidecar_descriptor || !translation.needs_lds_overflow_buf ||
        translation.skipped) {
      continue;
    }
    const auto normal_translation =
        std::ranges::find_if(translations, [&](const KdTranslation &candidate) {
          return !candidate.sidecar_descriptor && !candidate.skipped &&
                 candidate.kernel_name == translation.kernel_name;
        });
    if (normal_translation == translations.end()) {
      append_error(diagnostics, DiagnosticKind::KernelDescriptor,
                   "virtual LDS metadata could not find the normal descriptor translation; "
                   "leaving code object unchanged");
      return std::nullopt;
    }
    const uint64_t descriptor_vaddr =
        patched_layout.kernel_descriptor_offset(translation.kernel_name);
    if (descriptor_vaddr == 0) {
      append_error(diagnostics, DiagnosticKind::KernelDescriptor,
                   "virtual LDS metadata could not find the translated kernel descriptor symbol; "
                   "leaving code object unchanged");
      return std::nullopt;
    }

    const uint64_t sidecar_descriptor_vaddr = sidecar_descriptor_vaddrs[i];
    if (sidecar_descriptor_vaddr == 0) {
      append_error(diagnostics, DiagnosticKind::KernelDescriptor,
                   "virtual LDS metadata could not find the appended sidecar descriptor; leaving "
                   "code object unchanged");
      return std::nullopt;
    }

    // Sidecar identity, kernarg extension layout, and virtual-LDS policy are
    // serialized independently. Their stable kernel/variant names are the join
    // key; no generic mechanism embeds another feature's fields.
    sidecar_metadata.push_back(SidecarVariantMetadata{
        .kernel_name = translation.kernel_name,
        .variant_name = std::string(kVirtualLdsSidecarVariantName),
        .normal_descriptor_vaddr = descriptor_vaddr,
        .variant_descriptor_vaddr = sidecar_descriptor_vaddr,
    });

    KernargExtensionMetadata kernarg_extension{
        .kernel_name = translation.kernel_name,
        .variant_name = std::string(kVirtualLdsSidecarVariantName),
        .original_kernarg_size = translation.kernarg_size,
        .payloads = {{
            .size = kVirtualLdsRuntimeStateBytes,
            .alignment = alignof(uint64_t),
            .name = std::string(kVirtualLdsRuntimeStatePayloadName),
        }},
    };
    const KernargExtensionPayloadLayout payload_layout{
        .size = kernarg_extension.payloads.front().size,
        .alignment = kernarg_extension.payloads.front().alignment,
    };
    const auto wrapper_layout = make_kernarg_extension_layout(
        kernarg_extension.original_kernarg_size, std::span{&payload_layout, 1});
    if (!wrapper_layout || wrapper_layout->payload_offsets.empty() ||
        wrapper_layout->payload_offsets.front() !=
            translation.lds_overflow_kernarg_pointer_offset) {
      append_error(diagnostics, DiagnosticKind::KernelDescriptor,
                   "virtual LDS kernarg extension layout disagrees with the translated entry "
                   "prologue; leaving code object unchanged");
      return std::nullopt;
    }
    kernarg_extension_metadata.push_back(std::move(kernarg_extension));

    VirtualLdsKernelMetadata record{};
    record.kernel_name = translation.kernel_name;
    record.sidecar_variant_name = std::string(kVirtualLdsSidecarVariantName);
    record.static_lds_bytes = translation.lds_overflow_size;
    // AQL private_segment_size can include a dynamic call-stack request above
    // the normal descriptor's fixed allocation. Dispatch rewriting needs both
    // fixed sizes to preserve that dynamic portion while switching variants;
    // neither loaded descriptor address is guaranteed to be CPU-readable.
    record.normal_private_segment_size = normal_translation->target_private_size;
    record.virtual_private_segment_size = translation.target_private_size;
    record.virtual_lds_base_sgpr = translation.virtual_lds_lowering.base_sgpr;
    record.flags |= kVirtualLdsFlagRuntimeStateBlock;
    if (translation.workgroup_id_sgpr_x >= 0)
      record.flags |= kVirtualLdsFlagWorkgroupIdX;
    if (translation.workgroup_id_sgpr_y >= 0)
      record.flags |= kVirtualLdsFlagWorkgroupIdY;
    if (translation.workgroup_id_sgpr_z >= 0)
      record.flags |= kVirtualLdsFlagWorkgroupIdZ;
    virtual_lds_metadata.push_back(std::move(record));
  }

  if (!sidecar_metadata.empty()) {
    const auto metadata_bytes = serialize_sidecar_metadata(sidecar_metadata);
    if (metadata_bytes.empty() ||
        !patcher.append_nonalloc_section(kSidecarMetadataSectionName, metadata_bytes, 8)) {
      append_error(diagnostics, DiagnosticKind::ResourceLimit,
                   "sidecar metadata could not be materialized safely; leaving code object "
                   "unchanged");
      return std::nullopt;
    }
  }

  if (!kernarg_extension_metadata.empty()) {
    const auto metadata_bytes = serialize_kernarg_extension_metadata(kernarg_extension_metadata);
    if (metadata_bytes.empty() ||
        !patcher.append_nonalloc_section(kKernargExtensionMetadataSectionName, metadata_bytes, 8)) {
      append_error(diagnostics, DiagnosticKind::ResourceLimit,
                   "kernarg extension metadata could not be materialized safely; leaving code "
                   "object unchanged");
      return std::nullopt;
    }
  }

  if (!virtual_lds_metadata.empty()) {
    const auto metadata_bytes = serialize_virtual_lds_metadata(virtual_lds_metadata);
    if (metadata_bytes.empty() ||
        !patcher.append_nonalloc_section(kVirtualLdsMetadataSectionName, metadata_bytes, 8)) {
      append_error(diagnostics, DiagnosticKind::ResourceLimit,
                   "virtual LDS metadata could not be materialized safely; leaving code object "
                   "unchanged");
      return std::nullopt;
    }
  }

  if (target_mach)
    patcher.update_elf_flags(target_mach);
  return std::move(patcher).emit();
}

} // namespace

BinaryTranslator::~BinaryTranslator() = default;

BinaryTranslator::BinaryTranslator(rj_code_arch_t guest_arch, rj_code_arch_t host_arch,
                                   uint32_t target_mach, BinaryTranslatorOptions options)
    : guest_arch_(guest_arch), host_arch_(host_arch),
      target_mach_(target_mach ? target_mach : elf_mach_for_arch(host_arch)), options_(options),
      encoding_translate_(select_encoding_translator(guest_arch, host_arch)),
      legalization_lookup_(select_legalization(guest_arch, host_arch)),
      semantic_translator_(std::make_unique<SemanticTranslator>(
          guest_arch, host_arch, options.input_revision, options.output_revision)) {}

const InstructionLegalization *
BinaryTranslator::lookup_legalization(const Instruction &inst) const {
  // gfx1250 B0 and A0 have the same structural ISA, so the generated cross-ISA
  // tables cannot express their revision-specific behavior. Affected decoded
  // instructions are classified by the handwritten errata policy; everything
  // else intentionally has no entry and follows the raw same-ISA copy path.
  if (guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250 && host_arch_ == ROCJITSU_CODE_ARCH_GFX1250 &&
      options_.input_revision == ProcessorRevision::Gfx1250B0 &&
      options_.output_revision == ProcessorRevision::Gfx1250A0)
    return gfx1250_b0_to_a0_legalization(inst);

  return legalization_lookup_ ? legalization_lookup_(inst.encoding_id(), inst.opcode()) : nullptr;
}

void BinaryTranslator::set_trace_callback(TranslationTraceCallback callback) {
  trace_callback_ = std::move(callback);
}

TranslatedCodeObject BinaryTranslator::translate(const AmdGpuCodeObject &obj) {
  TranslatedCodeObject result;
  result.host_arch = host_arch_;

  CodeObjectPatcher patcher(obj);
  auto leave_unchanged = [&]() {
    const auto *image = reinterpret_cast<const uint8_t *>(obj.image_data());
    result.elf_bytes.assign(image, image + obj.image_size());
    return result;
  };

  // A same-architecture gfx1250 translation is direction-specific: A0 and B0
  // share an ELF machine ID, so both revisions must be given and must select a
  // supported direction. Enforce this here as well as in the C API so a direct or
  // future internal caller cannot bypass the check and get a silent identity copy
  // that skips every required workaround.
  if (guest_arch_ == ROCJITSU_CODE_ARCH_GFX1250 && host_arch_ == ROCJITSU_CODE_ARCH_GFX1250) {
    if (options_.input_revision == ProcessorRevision::Unspecified ||
        options_.output_revision == ProcessorRevision::Unspecified) {
      append_error(result.diagnostics, DiagnosticKind::Legalization,
                   "gfx1250 same-target translation requires both input and output silicon "
                   "revisions");
      return leave_unchanged();
    }
    // The implemented errata are deliberately one-way. Treating A0 input as B0
    // output would silently preserve A0 workaround sequences while claiming the
    // opposite direction, so fail before modifying the code object.
    if (options_.input_revision == ProcessorRevision::Gfx1250A0 &&
        options_.output_revision == ProcessorRevision::Gfx1250B0) {
      append_error(result.diagnostics, DiagnosticKind::Legalization,
                   "gfx1250 A0-to-B0 translation is not supported");
      return leave_unchanged();
    }
  }

  auto text = patcher.text_bytes();
  if (text.empty()) {
    append_error(result.diagnostics, DiagnosticKind::ResourceLimit,
                 "code object does not expose a non-empty .text section for translation");
    return leave_unchanged();
  }

  // DBT relocates instructions within .text (compaction, expansion, per-kernel
  // block placement) but does not rewrite relocation places that land inside
  // .text. An in-.text relocation would therefore be applied to the wrong
  // translated bytes. Fail closed rather than silently miscompile.
  if (patcher.has_relocations_within_text()) {
    append_error(result.diagnostics, DiagnosticKind::Legalization,
                 "code object has a relocation place inside .text; relocating instructions would "
                 "apply it to the wrong bytes and is not supported");
    return leave_unchanged();
  }

  // The patcher can retarget ordinary zero-addend symbol references and
  // symbol-less RELATIVE64 addends through the final source-to-target offset
  // map. Keep less explicit forms fail-closed until their relocation-specific
  // addend semantics are modeled.
  if (patcher.has_unsupported_relocation_to_text()) {
    append_error(result.diagnostics, DiagnosticKind::Legalization,
                 "code object has an unsupported relocation referencing .text; section symbols, "
                 "implicit addends, and named-symbol addends cannot be remapped safely");
    return leave_unchanged();
  }

  // Per-kernel text relocation strategy:
  // 1. Translate descriptors first so their source entries and ABI state define
  //    the normal and sidecar kernel scopes.
  // 2. Decode .text, recover bounded static indirect targets, and form each
  //    scope from ordinary CFG successors plus validated call edges.
  // 3. Emit each scope into a source-ordered local body. Semantic expansions
  //    grow that body and control transfers reserve explicit patch windows.
  // 4. Place entry stubs and bodies in final .text coordinates, then repair
  //    direct transfers, recovered indirect transfers, and their PC builders.
  // 5. Feed discovered register/private-memory requirements back into each
  //    descriptor and commit .text, descriptors, sidecars, metadata, and flags.
  auto decoder = Decoder::create(guest_arch_);
  if (!decoder) {
    append_error(result.diagnostics, DiagnosticKind::UnsupportedGuestArch,
                 "unsupported guest_arch: no decoder available");
    return leave_unchanged();
  }

  // Phase 1: descriptor translation gives DBT the source kernel roots and any
  // target descriptor/prologue bytes that must be materialized with the body.
  const bool skip_failed_kernels = options_.skip_failed_kernels;
  KernelDescriptorTranslator descriptor_translator(guest_arch_, host_arch_);
  const bool can_emit_sidecar_descriptors = supports_virtual_lds_sidecars(guest_arch_, host_arch_);
  KernelDescriptorTranslationOptions initial_descriptor_options;
  initial_descriptor_options.allow_oversized_lds = can_emit_sidecar_descriptors;
  auto descriptor_translations =
      descriptor_translator.translate_image(patcher.image_bytes(), patcher.text_offset(),
                                            patcher.text_size(), initial_descriptor_options);
  bool descriptors_supported = true;
  for (const auto &translation : descriptor_translations) {
    if (translation.supported || !skip_failed_kernels)
      append_diagnostics(result.diagnostics, translation.diagnostics);
    descriptors_supported &= translation.supported;
  }
  if (!descriptors_supported && !skip_failed_kernels) {
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

  const auto relocation_function_tables = discover_relocation_function_tables(obj);
  auto block_leaders = kernel_block_leaders(descriptor_translations, text);
  for (const RelocationFunctionTable &table : relocation_function_tables) {
    for (const RelocationFunctionPointer &entry : table.entries)
      block_leaders.push_back(entry.target_text_offset);
  }
  std::ranges::sort(block_leaders);
  block_leaders.erase(std::ranges::unique(block_leaders).begin(), block_leaders.end());

  // Phase 2: build a CFG over .text, including recovered indirect targets as
  // block leaders, then compute one source-reachable block set per descriptor
  // root. These sets are intentionally kernel-local: if two roots reach the same
  // helper block, Phase 3 emits that helper into both relocated bodies so every
  // branch or call target can be resolved through the current kernel's placement
  // map without borrowing another kernel's return continuation.
  auto blocks = BasicBlock::build(obj, *decoder, guest_arch_, block_leaders);
  const BlockOffsetIndex block_index = build_block_offset_index(blocks);
  const uint64_t text_vaddr = obj.text_sections().front()->vaddr();
  const auto relocation_table_dispatches =
      discover_relocation_table_dispatches(blocks, relocation_function_tables, text_vaddr);
  const auto relocation_table_calls = attach_relocation_table_call_edges(
      block_index, relocation_function_tables, relocation_table_dispatches);
  auto scopes = kernel_translation_scopes(blocks, block_index, descriptor_translations);

  if (can_emit_sidecar_descriptors) {
    std::vector<KdTranslation> sidecar_variants;
    for (const KernelTranslationScope &scope : scopes) {
      if (scope.translation == nullptr)
        continue;
      const uint32_t host_lds_bytes = arch_lds_bytes(host_arch_);
      const bool static_lds_exceeds_host =
          host_lds_bytes != 0 && scope.translation->target_lds_size > host_lds_bytes;
      // Dynamic LDS is only known at dispatch time, so every LDS-using kernel
      // needs a virtual sidecar. The sidecar descriptor owns the wrapper ABI
      // and may enable a target-only kernarg segment pointer when the source
      // descriptor left room in the 16 initialized User SGPRs.
      if (!static_lds_exceeds_host &&
          !scope_uses_virtualizable_lds(scope, guest_arch_, host_arch_)) {
        continue;
      }

      KernelDescriptorTranslationOptions virtual_descriptor_options;
      virtual_descriptor_options.virtualize_lds = true;
      auto virtual_translation = descriptor_translator.translate_descriptor(
          patcher.image_bytes(), scope.translation->descriptor_file_offset,
          scope.translation->entry_text_offset, virtual_descriptor_options);
      if (!virtual_translation) {
        append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                     "virtual LDS sidecar descriptor could not be computed; leaving code object "
                     "unchanged",
                     scope.translation->entry_text_offset);
        return leave_unchanged();
      }
      virtual_translation->kernel_name = scope.translation->kernel_name;
      virtual_translation->sidecar_descriptor = true;
      sidecar_variants.push_back(std::move(*virtual_translation));
    }
    if (!sidecar_variants.empty()) {
      descriptor_translations.insert(descriptor_translations.end(),
                                     std::make_move_iterator(sidecar_variants.begin()),
                                     std::make_move_iterator(sidecar_variants.end()));
      scopes = kernel_translation_scopes(blocks, block_index, descriptor_translations);
    }
  }

  const size_t expected_scope_count = kernel_translation_scope_count(descriptor_translations);
  if (scopes.size() != expected_scope_count) {
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

  auto relocation_diagnostic_kind = [&](const TextRelocationResult &relocation) {
    if (relocation.failure == TextLayoutFailureCategory::ResourceLimit)
      return DiagnosticKind::ResourceLimit;
    return DiagnosticKind::Legalization;
  };

  auto materialization_diagnostic_kind = [&](const KernelTextAppendResult &materialization) {
    if (materialization.failure == TextLayoutFailureCategory::ResourceLimit)
      return DiagnosticKind::ResourceLimit;
    return DiagnosticKind::KernelDescriptor;
  };

  struct KernelFailure {
    DiagnosticKind kind = DiagnosticKind::Legalization;
    std::string message;
    std::optional<uint64_t> guest_offset;
    std::string mnemonic;
    std::vector<std::string> required_work;
  };

  auto make_kernel_failure = [](DiagnosticKind kind, std::string message,
                                std::optional<uint64_t> guest_offset = std::nullopt,
                                std::string mnemonic = {},
                                std::vector<std::string> required_work = {}) {
    return KernelFailure{kind, std::move(message), guest_offset, std::move(mnemonic),
                         std::move(required_work)};
  };

  auto emit_skipped_kernel = [&](const KernelTranslationScope &scope,
                                 KernelFailure failure) -> bool {
    assert(scope.translation != nullptr && "kernel scope should have descriptor translation");
    if (scope.blocks.empty()) {
      append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                   "cannot skip failed kernel without a decoded source block",
                   scope.translation->entry_text_offset);
      return false;
    }

    const uint64_t source_entry = scope.translation->entry_text_offset;
    const bool skipped_uses_virtual_lds = scope.translation->needs_lds_overflow_buf;
    auto skipped_text = append_skipped_kernel_stub(
        translated_text,
        {.source_entry = scope.translation->entry_text_offset,
         .has_kernarg_preload_firmware_skip = scope.translation->has_kernarg_preload_firmware_skip},
        host_arch_);
    if (!skipped_text.ok) {
      append_error(result.diagnostics, materialization_diagnostic_kind(skipped_text),
                   skipped_text.message, skipped_text.source_offset);
      return false;
    }

    for (KdTranslation &translation : descriptor_translations) {
      if (translation.entry_text_offset != source_entry)
        continue;
      // Normal hardware-LDS and virtual-LDS sidecar descriptors share the same
      // source entry, but a sidecar translation failure must not turn a normal
      // descriptor that can still launch on hardware LDS into a no-op. Skip
      // together any descriptor of the failing variant. Additionally, when the
      // sidecar variant is the one failing, a normal descriptor whose static LDS
      // only fit because a sidecar was promised is itself undispatchable (its
      // advertised LDS would fault on the host), so it must be stubbed too.
      const bool same_variant = translation.needs_lds_overflow_buf == skipped_uses_virtual_lds;
      const bool orphaned_by_sidecar_failure = skipped_uses_virtual_lds &&
                                               !translation.needs_lds_overflow_buf &&
                                               translation.static_lds_requires_sidecar;
      if (!same_variant && !orphaned_by_sidecar_failure)
        continue;
      translation.target_entry_text_offset = skipped_text.target_entry;
      translation.target_body_entry_text_offset = skipped_text.target_body_entry;
      // A skipped descriptor must describe the target stub, not the failed
      // guest kernel. Leaving oversized SGPR/LDS/private requirements in place
      // can make HIP fail during launch even though the entry points at safe
      // target code. Granulated zero encodes the minimum allocation bucket.
      translation.configure_skipped_stub();
    }

    std::string message =
        "*** SKIPPED KERNEL " + kernel_label(*scope.translation) +
        " REPLACED WITH S_ENDPGM; DISPATCHING IT WILL SILENTLY PRODUCE INVALID OUTPUTS *** "
        "Translation error: " +
        std::move(failure.message);
    append_warning(result.diagnostics, DiagnosticKind::KernelSkipped, std::move(message),
                   failure.guest_offset ? failure.guest_offset
                                        : std::optional<uint64_t>(source_entry),
                   std::move(failure.mnemonic), std::move(failure.required_work));
    return true;
  };

  // Per-scope relocation output. Declared before fail_or_skip_kernel so a skip can
  // truncate them back to their pre-scope sizes together with translated_text —
  // otherwise a scope that fails AFTER appending relocations (e.g. during branch
  // fixup or descriptor recompute) would leave stale entries whose source mapping
  // then applies to the replacement stub or a later kernel.
  std::vector<TextOffsetRelocation> text_relocations;
  std::vector<PcRelativeDataRelocation> data_relocations;

  auto fail_or_skip_kernel =
      [&](const KernelTranslationScope &scope, KernelFailure failure, size_t output_begin,
          const std::vector<DescriptorVariantCheckpoint> &descriptor_snapshot,
          size_t text_relocations_begin, size_t data_relocations_begin) -> bool {
    if (!skip_failed_kernels) {
      append_error(result.diagnostics, failure.kind, std::move(failure.message),
                   failure.guest_offset, std::move(failure.mnemonic),
                   std::move(failure.required_work));
      return false;
    }

    const uint64_t source_entry = scope.translation->entry_text_offset;
    translated_text.resize(output_begin);
    // Discard any relocation records this scope committed before failing.
    text_relocations.resize(text_relocations_begin);
    data_relocations.resize(data_relocations_begin);
    for (const DescriptorVariantCheckpoint &saved : descriptor_snapshot) {
      if (saved.index >= descriptor_translations.size()) {
        append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                     "descriptor checkpoint index changed during skip rollback", source_entry);
        return false;
      }
      descriptor_translations[saved.index] = saved.translation;
    }

    KernelTranslationScope restored_scope = scope;
    restored_scope.translation = nullptr;
    for (KdTranslation &translation : descriptor_translations) {
      if (!same_kernel_scope_variant(translation, *scope.translation))
        continue;
      restored_scope.translation = &translation;
      break;
    }
    if (restored_scope.translation == nullptr) {
      append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                   "failed kernel descriptor was lost during skip rollback", source_entry);
      return false;
    }

    return emit_skipped_kernel(restored_scope, std::move(failure));
  };

  auto reserve_long_branch_sgpr_pair = [&](TranslationContext &context) -> std::optional<uint16_t> {
    auto base = next_long_branch_sgpr_pair(context, host_arch_);
    if (!base)
      return std::nullopt;
    context.require_sgprs(static_cast<uint32_t>(*base) + 2);
    return base;
  };

  for (const KernelTranslationScope &scope : scopes) {
    if (scope.blocks.empty())
      continue;
    assert(scope.translation != nullptr && "kernel scope should have descriptor translation");
    if (scope.translation->skipped)
      continue;

    const size_t output_begin = translated_text.size();
    const size_t text_relocations_begin = text_relocations.size();
    const size_t data_relocations_begin = data_relocations.size();
    const auto descriptor_snapshot =
        checkpoint_scope_descriptors(descriptor_translations, *scope.translation);
    bool skip_scope = false;

    if (!scope.translation->supported) {
      auto failure = make_kernel_failure(
          DiagnosticKind::KernelDescriptor,
          "kernel descriptor translation requires unsupported resource or ABI virtualization");
      for (const TranslationDiagnostic &diagnostic : scope.translation->diagnostics) {
        if (diagnostic.severity != DiagnosticSeverity::Error)
          continue;
        failure.kind = diagnostic.kind;
        failure.message = diagnostic.message;
        failure.guest_offset = diagnostic.guest_offset;
        failure.mnemonic = diagnostic.mnemonic;
        failure.required_work = diagnostic.required_work;
        break;
      }
      if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                              text_relocations_begin, data_relocations_begin))
        continue;
      return leave_unchanged();
    }

    const auto opaque_fallthrough =
        std::ranges::find_if(scope.blocks, [&](const BasicBlock *block) {
          return block != nullptr && block->falls_through_to_undecodable_text() &&
                 !is_gfx1250_compiler_unreachable_stub(*block, guest_arch_);
        });
    if (opaque_fallthrough != scope.blocks.end()) {
      auto failure =
          make_kernel_failure(DiagnosticKind::Legalization,
                              "reachable kernel code falls through into undecodable .text bytes",
                              (*opaque_fallthrough)->end_offset());
      if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                              text_relocations_begin, data_relocations_begin))
        continue;
      return leave_unchanged();
    }

    // Phase 3: translate this kernel into a temporary, source-ordered body. The
    // body starts at offset zero while it is being built; after final padding and
    // any launch window are chosen, every recorded target offset is rebased into
    // the output .text. This lets instruction expansions change block sizes
    // without precomputing speculative side-region offsets.
    KernelTextLayout layout;
    layout.source_entry = scope.translation->entry_text_offset;

    TranslationContext kernel_context(
        scope.translation->target_vgpr_count, scope.translation->target_agpr_count,
        scope.translation->target_accvgpr_base, scope.translation->target_sgpr_count,
        scope.translation->target_private_size);
    if (scope.translation->needs_lds_overflow_buf) {
      auto virtual_lds_base =
          reserve_virtual_lds_base_sgpr_pair(kernel_context, KernelBlockScope(scope.blocks),
                                             *scope.translation, guest_arch_, host_arch_);
      if (!virtual_lds_base) {
        auto failure = make_kernel_failure(
            DiagnosticKind::ResourceLimit,
            "virtual LDS lowering cannot reserve a backing-buffer SGPR pair", layout.source_entry);
        if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                                text_relocations_begin, data_relocations_begin))
          continue;
        return leave_unchanged();
      }
      kernel_context.virtualize_lds = true;
      kernel_context.virtual_lds_base_sgpr = virtual_lds_base->base;
      kernel_context.virtual_lds_base_sgpr_spill_per_use = virtual_lds_base->spill_per_use;
      kernel_context.virtual_lds_kernarg_segment_ptr_sgpr =
          scope.translation->kernarg_segment_ptr_sgpr;
      kernel_context.virtual_lds_kernarg_pointer_offset =
          scope.translation->lds_overflow_kernarg_pointer_offset;
      scope.translation->virtual_lds_lowering.base_sgpr = virtual_lds_base->base;
      scope.translation->virtual_lds_lowering.prologue_temp_sgpr = virtual_lds_base->prologue_temp;
      scope.translation->virtual_lds_lowering.base_sgpr_spill_per_use =
          virtual_lds_base->spill_per_use;
      if (virtual_lds_base->spill_per_use) {
        const auto pointer_spill = kernel_context.reserve_persistent_semantic_spill_dwords(2);
        if (!pointer_spill) {
          auto failure = make_kernel_failure(
              DiagnosticKind::ResourceLimit,
              "virtual LDS backing-pointer spill offset overflows the 32-bit private segment",
              layout.source_entry);
          if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                                  text_relocations_begin, data_relocations_begin))
            continue;
          return leave_unchanged();
        }
        kernel_context.virtual_lds_base_pointer_spilled = true;
        kernel_context.virtual_lds_base_pointer_spill_offset = *pointer_spill;
        scope.translation->virtual_lds_lowering.base_pointer_spilled = true;
        scope.translation->virtual_lds_lowering.base_pointer_spill_offset = *pointer_spill;
      }
      if (!append_virtual_lds_entry_prologue(*scope.translation, guest_arch_, host_arch_)) {
        auto failure = make_kernel_failure(
            DiagnosticKind::KernelDescriptor,
            "virtual LDS lowering cannot materialize backing-buffer pointer entry prologue",
            layout.source_entry);
        if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                                text_relocations_begin, data_relocations_begin))
          continue;
        return leave_unchanged();
      }
    }

    layout.entry_plan = {
        .has_kernarg_preload_firmware_skip = scope.translation->has_kernarg_preload_firmware_skip,
        .kernarg_preload_firmware_entry_text_offset =
            scope.translation->kernarg_preload_firmware_entry_text_offset,
        .prologue_words = scope.translation->prologue_words,
    };
    if (!kernarg_preload_launch_window_fits(layout.entry_plan)) {
      auto failure = make_kernel_failure(
          DiagnosticKind::KernelDescriptor,
          "kernel descriptor prologue does not fit in the 256-byte kernarg preload compatibility "
          "window",
          layout.source_entry);
      if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                              text_relocations_begin, data_relocations_begin))
        continue;
      return leave_unchanged();
    }
    const bool can_use_long_direct_branches =
        next_long_branch_sgpr_pair(kernel_context, host_arch_).has_value();
    LivenessAnalysisOptions liveness_options;
    liveness_options.max_free_vgpr =
        static_cast<uint16_t>(isa_properties(host_arch_).max_addressable_vgprs_per_wf);
    liveness_options.arch = guest_arch_;
    liveness_options.entry_block = scope.entry;
    liveness_options.text = text;
    if (options_.debug_min_free_vgpr)
      liveness_options.min_free_vgpr = *options_.debug_min_free_vgpr;
    std::vector<const Instruction *> live_before_instructions;
    if (semantic_translator_ && semantic_translator_->has_rules()) {
      // Semantic expansion rules are the only DBT users of instruction-level
      // live-before data. The block dataflow still covers the full kernel, but
      // filtering the stored snapshots avoids retaining one RegisterSet per
      // decoded instruction in very large ML kernels.
      for (BasicBlock *block : scope.blocks) {
        if (block == nullptr)
          continue;
        for (const Instruction &inst : block->instructions()) {
          if (semantic_translator_->has_expand_rule(inst.encoding_id(), inst.opcode()))
            live_before_instructions.push_back(&inst);
        }
      }
      liveness_options.restrict_live_before_to_instructions = true;
      liveness_options.live_before_instructions = std::span<const Instruction *const>(
          live_before_instructions.data(), live_before_instructions.size());
    }
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
          auto failure =
              make_kernel_failure(DiagnosticKind::Legalization,
                                  "recovered indirect branch has inconsistent consumer metadata",
                                  source_call_offset, "indirect branch");
          if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                                  text_relocations_begin, data_relocations_begin)) {
            skip_scope = true;
            break;
          }
          return leave_unchanged();
        }
        if (fixup.source_target_offset != first.source_target_offset)
          single_effective_target = false;
      }
      if (skip_scope)
        break;

      // An incomplete fact means at least one predecessor left the PC pair
      // unconstrained. That path has no recovered builder or target to relocate,
      // yet its runtime SGPR pair may hold an original .text address; after DBT
      // relocates .text, the retained dynamic transfer would jump to stale or moved
      // bytes, and the unknown target block may be absent from the emitted scope.
      // We cannot prove the unconstrained path is free of a relocatable text
      // address, so fail closed for the whole consumer rather than relocate only
      // the known builders.
      const bool any_incomplete = std::ranges::any_of(
          consumer.fixups, [](const IndirectCallFixup &fixup) { return fixup.source_incomplete; });
      if (any_incomplete) {
        auto failure = make_kernel_failure(
            DiagnosticKind::Legalization,
            "recovered indirect branch has an unconstrained predecessor path that cannot be "
            "relocated",
            source_call_offset, "indirect branch");
        if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                                text_relocations_begin, data_relocations_begin)) {
          skip_scope = true;
          break;
        }
        return leave_unchanged();
      }

      // A complete consumer with one effective target can become a direct window.
      // A complete multi-target consumer keeps the dynamic consumer and rewrites
      // each source-side builder (relocation/liveness).
      if (single_effective_target) {
        consumer.use_transfer_window = true;
        consumer.window_fixup = first;
      } else {
        pending_builder_fixups.insert(pending_builder_fixups.end(), consumer.fixups.begin(),
                                      consumer.fixups.end());
      }
    }
    if (skip_scope)
      continue;

    std::vector<uint8_t> kernel_text;
    std::vector<PendingTrace> pending_traces;
    uint64_t source_body_size = 0;
    for (BasicBlock *block : scope.blocks)
      source_body_size += block->size();
    const uint64_t recovered_window_growth =
        recovered_indirect_by_call.size() * kMaxRecoveredIndirectTransferWords * sizeof(uint32_t);
    kernel_text.reserve(static_cast<size_t>(std::min<uint64_t>(
        source_body_size + recovered_window_growth, std::numeric_limits<size_t>::max())));

    std::unordered_map<uint64_t, uint64_t> target_offset_by_source_offset;
    target_offset_by_source_offset.reserve(
        static_cast<size_t>(source_body_size / sizeof(uint32_t)) + scope.blocks.size());
    layout.body_begin = 0;
    layout.blocks.reserve(scope.blocks.size());
    uint64_t next_branch_island_pool_offset = first_direct_branch_island_pool_offset();
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
        // Ask the semantic translator directly whether this instruction has an
        // expand rule. The previous positional cursor into live_before_instructions
        // silently depended on that vector being built in the exact same block/
        // instruction iteration order as this loop; querying by encoding/opcode
        // removes that hidden coupling.
        const bool has_semantic_expand_rule =
            semantic_translator_ != nullptr &&
            semantic_translator_->has_expand_rule(inst.encoding_id(), inst.opcode());
        // Record every instruction start, not just recovered-PC builder
        // boundaries. The same final map keeps ELF labels attached to the
        // relocated instruction stream after semantic expansions change sizes.
        target_offset_by_source_offset.emplace(offset, target_offset);

        const auto recovered_it = recovered_indirect_by_call.find(offset);
        const bool has_recovered_indirect_call = recovered_it != recovered_indirect_by_call.end();
        const bool has_relocation_table_call = relocation_table_calls.contains(offset);
        const bool recovered_indirect_return = valid_call_return_offsets.contains(offset);
        const auto direct_branch_delta = inst.branch_offset_bytes();
        if ((inst.flags() & (INDIRECT_BRANCH | INDIRECT_CALL)) != 0 &&
            !has_recovered_indirect_call && !has_relocation_table_call &&
            !recovered_indirect_return && !direct_branch_delta) {
          auto failure = make_kernel_failure(
              DiagnosticKind::Legalization,
              "indirect branch or call target recovery is not implemented for relocated kernel "
              "text",
              offset, std::string(inst.mnemonic()));
          if (continue_after_failure && !skip_failed_kernels) {
            append_error(result.diagnostics, failure.kind, failure.message, failure.guest_offset,
                         failure.mnemonic, failure.required_work);
            if (continue_after_instruction_error(inst, offset, kernel_text, pending_traces)) {
              continue;
            }
          }
          if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                                  text_relocations_begin, data_relocations_begin)) {
            skip_scope = true;
            break;
          }
          return leave_unchanged();
        }

        if (direct_branch_delta) {
          // Record direct branches while emitting the body, but patch only after
          // every block has a final target placement. This keeps fallthrough
          // implicit and limits fixups to explicit PC-relative edges. Emit the
          // branch into a fixed-size patch window. Kernels with a legal
          // descriptor-backed SGPR pair reserve the long form up front; kernels
          // already at the SGPR allocation limit keep compact branch slots so
          // DBT does not create artificial range pressure it cannot repair.
          const int64_t source_target =
              static_cast<int64_t>(offset + inst_size) + static_cast<int64_t>(*direct_branch_delta);
          if (source_target < 0) {
            auto failure =
                make_kernel_failure(DiagnosticKind::Legalization,
                                    "direct branch target is outside the source .text range",
                                    offset, std::string(inst.mnemonic()));
            if (continue_after_failure && !skip_failed_kernels) {
              append_error(result.diagnostics, failure.kind, failure.message, failure.guest_offset,
                           failure.mnemonic, failure.required_work);
              if (continue_after_instruction_error(inst, offset, kernel_text, pending_traces)) {
                continue;
              }
            }
            if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                                    text_relocations_begin, data_relocations_begin)) {
              skip_scope = true;
              break;
            }
            return leave_unchanged();
          }
          const uint64_t branch_window_bytes = direct_branch_patch_window_bytes(
              inst, offset, static_cast<uint64_t>(source_target), can_use_long_direct_branches);
          layout.branch_fixups.push_back(
              {.inst = &inst,
               .source_inst_offset = offset,
               .source_target_offset = static_cast<uint64_t>(source_target),
               .target_inst_offset = target_offset,
               .target_window_bytes = branch_window_bytes});

          if (!inst.raw_encoding()) {
            auto failure = make_kernel_failure(DiagnosticKind::Legalization,
                                               "direct branch is missing raw encoding", offset,
                                               std::string(inst.mnemonic()));
            if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                                    text_relocations_begin, data_relocations_begin)) {
              skip_scope = true;
              break;
            }
            return leave_unchanged();
          }

          const InstructionLegalization *branch_leg = lookup_legalization(inst);
          const uint16_t branch_dst_opcode = branch_leg ? branch_leg->target_opcode : inst.opcode();

          bool copied_original = false;
          bool changed = false;
          std::vector<uint32_t> target_words;
          if (!handle_encoding(inst, offset, kernel_text, branch_dst_opcode, text,
                               trace_callback_ != nullptr, copied_original, changed,
                               target_words)) {
            if (continue_after_instruction_error(inst, offset, kernel_text, pending_traces))
              continue;
            return leave_unchanged();
          }
          append_nop_padding(kernel_text, branch_window_bytes - inst.size(), host_arch_);
          queue_trace(pending_traces, inst, offset, branch_leg, copied_original, false, changed,
                      target_offset, std::move(target_words));
          continue;
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

        const InstructionLegalization *leg = lookup_legalization(inst);

        const uint16_t dst_opcode = leg ? leg->target_opcode : inst.opcode();

        // Try semantic lowering before raw encoding translation. A matched
        // semantic rule that cannot safely emit code is a translation error:
        // falling through would silently preserve guest semantics on the wrong
        // host ISA.
        if (has_semantic_expand_rule) {
          auto expansion =
              semantic_translator_->try_lower_expand(inst, offset, text, liveness, kernel_context);
          if (expansion.status == ExpandStatus::Failed) {
            auto failure = make_kernel_failure(
                DiagnosticKind::ExpandFailed,
                expansion.message.empty()
                    ? "semantic EXPAND rule matched, but could not safely lower"
                    : expansion.message,
                offset, std::string(inst.mnemonic()), std::move(expansion.required_work));
            if (continue_after_failure && !skip_failed_kernels) {
              append_error(result.diagnostics, failure.kind, failure.message, failure.guest_offset,
                           failure.mnemonic, failure.required_work);
              if (continue_after_instruction_error(inst, offset, kernel_text, pending_traces)) {
                continue;
              }
            }
            if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                                    text_relocations_begin, data_relocations_begin)) {
              skip_scope = true;
              break;
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

        {
          auto virtual_lds_expansion =
              lower_virtual_lds_instruction(inst, kernel_context, guest_arch_, host_arch_);
          if (virtual_lds_expansion.status == ExpandStatus::Failed) {
            auto failure = make_kernel_failure(DiagnosticKind::ExpandFailed,
                                               virtual_lds_expansion.message.empty()
                                                   ? "virtual LDS lowering failed"
                                                   : virtual_lds_expansion.message,
                                               offset, std::string(inst.mnemonic()),
                                               std::move(virtual_lds_expansion.required_work));
            if (continue_after_failure && !skip_failed_kernels) {
              append_error(result.diagnostics, failure.kind, failure.message, failure.guest_offset,
                           failure.mnemonic, failure.required_work);
              if (continue_after_instruction_error(inst, offset, kernel_text, pending_traces)) {
                continue;
              }
            }
            if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                                    text_relocations_begin, data_relocations_begin)) {
              skip_scope = true;
              break;
            }
            return leave_unchanged();
          }

          if (virtual_lds_expansion.status == ExpandStatus::Success) {
            std::vector<uint32_t> target_words = std::move(virtual_lds_expansion.words);
            append_words(kernel_text, target_words);
            queue_trace(pending_traces, inst, offset, leg, false, true, true, target_offset,
                        std::move(target_words));
            continue;
          }
        }

        if (leg && leg->action == Action::Expand) {
          auto failure = make_kernel_failure(
              DiagnosticKind::ExpandMissing,
              "legalization requires EXPAND, but no expansion rule is implemented", offset,
              std::string(inst.mnemonic()), {"Add a semantic expansion rule for this mnemonic."});
          if (continue_after_failure && !skip_failed_kernels) {
            append_error(result.diagnostics, failure.kind, failure.message, failure.guest_offset,
                         failure.mnemonic, failure.required_work);
            if (continue_after_instruction_error(inst, offset, kernel_text, pending_traces)) {
              continue;
            }
          }
          if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                                  text_relocations_begin, data_relocations_begin)) {
            skip_scope = true;
            break;
          }
          return leave_unchanged();
        }

        // A stepping-only translation has identical encodings for ordinary
        // instructions. Copy the authoritative source text, including literal
        // and modifier suffix words, instead of reconstructing bytes from the
        // decoder's base-format raw encoding. Direct branches and recovered
        // indirect transfers have already taken their relocation paths above;
        // explicit errata expansions have already continued or failed closed.
        if (guest_arch_ == host_arch_ && leg == nullptr) {
          copy_original_instruction(inst, offset, kernel_text, pending_traces);
          continue;
        }

        // Cross-arch translation must have a legalization decision for every
        // opcode. When a lookup function exists (i.e. this is not a same-arch
        // identity pass) but the opcode is absent from the table, or the table
        // marks it Illegal, re-encoding it verbatim with the guest opcode number
        // would silently produce a different — possibly valid but wrong — host
        // instruction. Fail loudly instead of that silent passthrough. A null
        // lookup function means same-arch identity translation, where verbatim
        // copy is correct, so that path is intentionally not gated here.
        if (legalization_lookup_ != nullptr && (leg == nullptr || leg->action == Action::Illegal)) {
          auto failure = make_kernel_failure(
              DiagnosticKind::Legalization,
              leg == nullptr
                  ? "no legalization entry for this opcode on the target ISA; refusing to emit the "
                    "guest encoding verbatim"
                  : "legalization marks this opcode illegal on the target ISA",
              offset, std::string(inst.mnemonic()),
              {"Add a legalization/substitution/expansion entry for this mnemonic in the amdisa "
               "codegen pipeline."});
          if (continue_after_failure && !skip_failed_kernels) {
            append_error(result.diagnostics, failure.kind, failure.message, failure.guest_offset,
                         failure.mnemonic, failure.required_work);
            if (continue_after_instruction_error(inst, offset, kernel_text, pending_traces)) {
              continue;
            }
          }
          if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                                  text_relocations_begin, data_relocations_begin)) {
            skip_scope = true;
            break;
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
      if (skip_scope)
        break;
      placement.target_end = kernel_text.size();
      layout.blocks.push_back(placement);
      target_offset_by_source_offset.emplace(block->end_offset(), kernel_text.size());
      if (!can_use_long_direct_branches && block != scope.blocks.back() &&
          kernel_text.size() >= next_branch_island_pool_offset) {
        append_direct_branch_island_pool(kernel_text, layout, host_arch_);
        next_branch_island_pool_offset = next_direct_branch_island_pool_offset(kernel_text.size());
      }
    }
    if (skip_scope)
      continue;
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
        auto failure = make_kernel_failure(
            DiagnosticKind::Legalization,
            "recovered indirect branch builder is not fully present in the relocated body",
            fixup.source_call_offset, "indirect branch");
        if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                                text_relocations_begin, data_relocations_begin)) {
          skip_scope = true;
          break;
        }
        return leave_unchanged();
      }

      fixup.target_getpc_offset = getpc_it->second;
      fixup.target_recovery_begin_offset = begin_it->second;
      fixup.target_recovery_end_offset = end_it->second;
      layout.recovered_builder_fixups.push_back(fixup);
    }
    if (skip_scope)
      continue;

    auto materialized =
        append_relocated_kernel_text(translated_text, layout, kernel_text, host_arch_);
    if (!materialized.ok) {
      auto failure = make_kernel_failure(materialization_diagnostic_kind(materialized),
                                         materialized.message, materialized.source_offset);
      if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                              text_relocations_begin, data_relocations_begin))
        continue;
      return leave_unchanged();
    }
    const uint64_t target_delta = materialized.target_delta;
    for (const auto &[source_offset, target_offset] : target_offset_by_source_offset) {
      text_relocations.push_back(
          {.source_offset = source_offset, .target_offset = target_offset + target_delta});
    }
    for (const RelocationTableDispatch &dispatch : relocation_table_dispatches) {
      if (!target_offset_by_source_offset.contains(dispatch.source_call_offset))
        continue;
      const auto getpc = target_offset_by_source_offset.find(dispatch.source_getpc_offset);
      const auto add = target_offset_by_source_offset.find(dispatch.source_address_add_offset);
      if (getpc == target_offset_by_source_offset.end() ||
          add == target_offset_by_source_offset.end()) {
        append_error(result.diagnostics, DiagnosticKind::Legalization,
                     "relocation-table GOT address builder is not fully present in the relocated "
                     "body",
                     dispatch.source_call_offset, "s_swap_pc_i64");
        return leave_unchanged();
      }
      data_relocations.push_back(
          {.target_getpc_offset = getpc->second + target_delta,
           .target_literal_offset = add->second + target_delta + sizeof(uint32_t),
           .source_target_vaddr = dispatch.source_table_address_vaddr});
    }

    // Phase 5: now that every emitted source block has a final target offset,
    // patch explicit direct branches, recovered source-side builders, and
    // recovered indirect transfer windows.
    auto patched_direct_branches = patch_direct_branch_fixups(translated_text, layout, host_arch_);
    if (!patched_direct_branches.ok &&
        patched_direct_branches.reason == TextLayoutFailureReason::BranchOutOfRange) {
      if (auto sgpr = reserve_long_branch_sgpr_pair(kernel_context)) {
        layout.long_branch_sgpr = *sgpr;
        patched_direct_branches = patch_direct_branch_fixups(translated_text, layout, host_arch_);
      }
    }
    if (!patched_direct_branches.ok) {
      auto failure = make_kernel_failure(relocation_diagnostic_kind(patched_direct_branches),
                                         patched_direct_branches.message,
                                         patched_direct_branches.source_offset);
      if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                              text_relocations_begin, data_relocations_begin))
        continue;
      return leave_unchanged();
    }

    if (auto patched = patch_recovered_builder_fixups(translated_text, layout, host_arch_);
        !patched.ok) {
      auto failure = make_kernel_failure(relocation_diagnostic_kind(patched), patched.message,
                                         patched.source_offset, "indirect branch");
      if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                              text_relocations_begin, data_relocations_begin))
        continue;
      return leave_unchanged();
    }

    if (auto patched = patch_recovered_indirect_fixups(translated_text, layout, host_arch_);
        !patched.ok) {
      auto failure = make_kernel_failure(relocation_diagnostic_kind(patched), patched.message,
                                         patched.source_offset, "indirect branch");
      if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                              text_relocations_begin, data_relocations_begin))
        continue;
      return leave_unchanged();
    }

    if (kernel_context.required_vgpr_count > kernel_context.num_vgprs)
      scope.translation->target_vgpr_count = kernel_context.required_vgpr_count;
    if (kernel_context.required_sgpr_count > kernel_context.num_sgprs)
      scope.translation->target_sgpr_count = kernel_context.required_sgpr_count;
    if (kernel_context.required_private_segment_fixed_size >
        kernel_context.private_segment_fixed_size)
      scope.translation->target_private_size = kernel_context.required_private_segment_fixed_size;

    if (scope.translation->target_vgpr_count != kernel_context.num_vgprs ||
        scope.translation->target_sgpr_count != kernel_context.num_sgprs ||
        scope.translation->target_private_size != kernel_context.private_segment_fixed_size) {
      // Semantic rules may allocate descriptor-backed scratch registers or
      // per-lane private spill slots beyond the kernel's original resources.
      // Recompute the descriptor with those larger minimums before patching it
      // into the output image.
      KernelDescriptorTranslationOptions descriptor_options;
      descriptor_options.minimum_vgprs = scope.translation->target_vgpr_count;
      descriptor_options.minimum_sgprs = scope.translation->target_sgpr_count;
      descriptor_options.private_segment_fixed_size_addend =
          scope.translation->target_private_size - kernel_context.private_segment_fixed_size;
      descriptor_options.virtualize_lds = scope.translation->needs_lds_overflow_buf;
      descriptor_options.allow_oversized_lds =
          can_emit_sidecar_descriptors && !scope.translation->needs_lds_overflow_buf;

      // Descriptor growth is intentionally done after instruction lowering so
      // each kernel is translated once. Only descriptors that enter this code
      // scope need the larger register counts; rescanning the whole image would
      // also recompute unrelated kernels and risks mixing diagnostics across
      // scopes.
      bool recomputed_descriptor = false;
      for (KdTranslation &translation : descriptor_translations) {
        if (!same_kernel_scope_variant(translation, *scope.translation))
          continue;

        auto updated = descriptor_translator.translate_descriptor(
            patcher.image_bytes(), translation.descriptor_file_offset,
            translation.entry_text_offset, descriptor_options);
        if (!updated) {
          auto failure =
              make_kernel_failure(DiagnosticKind::KernelDescriptor,
                                  "kernel descriptor translation could not be recomputed");
          if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                                  text_relocations_begin, data_relocations_begin)) {
            skip_scope = true;
            break;
          }
          return leave_unchanged();
        }
        updated->kernel_name = translation.kernel_name;
        updated->sidecar_descriptor = translation.sidecar_descriptor;
        updated->virtual_lds_lowering = translation.virtual_lds_lowering;
        // Register feedback also recomputes ordinary, non-virtual descriptors.
        // Only a virtual-LDS variant owns a backing-pointer entry prologue;
        // asking an unrelated ISA pair to materialize one makes otherwise
        // valid SGPR/VGPR growth fail after semantic lowering.
        if (updated->needs_lds_overflow_buf &&
            !append_virtual_lds_entry_prologue(*updated, guest_arch_, host_arch_)) {
          auto failure = make_kernel_failure(
              DiagnosticKind::KernelDescriptor,
              "virtual LDS lowering cannot materialize backing-buffer pointer entry prologue");
          if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                                  text_relocations_begin, data_relocations_begin)) {
            skip_scope = true;
            break;
          }
          return leave_unchanged();
        }

        if (!updated->supported) {
          if (skip_failed_kernels) {
            auto failure = make_kernel_failure(DiagnosticKind::KernelDescriptor,
                                               "kernel descriptor translation requires unsupported "
                                               "resource or ABI virtualization");
            for (const TranslationDiagnostic &diagnostic : updated->diagnostics) {
              if (diagnostic.severity != DiagnosticSeverity::Error)
                continue;
              failure.kind = diagnostic.kind;
              failure.message = diagnostic.message;
              failure.guest_offset = diagnostic.guest_offset;
              failure.mnemonic = diagnostic.mnemonic;
              failure.required_work = diagnostic.required_work;
              break;
            }
            if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                                    text_relocations_begin, data_relocations_begin)) {
              skip_scope = true;
              break;
            }
          }
          append_diagnostics(result.diagnostics, updated->diagnostics);
          append_error(result.diagnostics, DiagnosticKind::KernelDescriptor,
                       "kernel descriptor translation requires unsupported resource or ABI "
                       "virtualization; leaving code object unchanged");
          return leave_unchanged();
        }
        append_diagnostics(result.diagnostics, updated->diagnostics);

        if (updated->prologue_words != translation.prologue_words) {
          auto failure = make_kernel_failure(
              DiagnosticKind::KernelDescriptor,
              "kernel descriptor prologue changed after relocated text was emitted");
          if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                                  text_relocations_begin, data_relocations_begin)) {
            skip_scope = true;
            break;
          }
          return leave_unchanged();
        }

        updated->target_entry_text_offset = layout.target_entry;
        updated->target_body_entry_text_offset = layout.target_body_entry;
        translation = std::move(*updated);
        recomputed_descriptor = true;
      }
      if (skip_scope)
        continue;

      if (!recomputed_descriptor) {
        auto failure = make_kernel_failure(DiagnosticKind::KernelDescriptor,
                                           "kernel descriptor translation could not be recomputed");
        if (fail_or_skip_kernel(scope, std::move(failure), output_begin, descriptor_snapshot,
                                text_relocations_begin, data_relocations_begin))
          continue;
        return leave_unchanged();
      }
    }

    flush_traces(pending_traces, target_delta);

    for (KdTranslation &translation : descriptor_translations) {
      if (!same_kernel_scope_variant(translation, *scope.translation))
        continue;
      translation.target_entry_text_offset = layout.target_entry;
      translation.target_body_entry_text_offset = layout.target_body_entry;
    }
  }

  if (continue_after_failure && has_error_diagnostic(result.diagnostics))
    return leave_unchanged();

  // Phase 6 commits the completed translation plan without mixing ELF mutation
  // and sidecar metadata construction into the per-kernel lowering transaction.
  auto materialized = materialize_translated_code_object(
      std::move(patcher), std::move(translated_text), text.size(), text_relocations,
      data_relocations, descriptor_translations, host_arch_, target_mach_, result.diagnostics);
  if (!materialized)
    return leave_unchanged();
  result.elf_bytes = std::move(*materialized);
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
