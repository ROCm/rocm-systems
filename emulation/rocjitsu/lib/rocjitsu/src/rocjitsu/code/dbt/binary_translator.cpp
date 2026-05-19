// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/dbt/binary_translator.h"

#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/dbt/generated/encoding_cdna4_to_rdna3.h"
#include "rocjitsu/code/dbt/generated/encoding_cdna4_to_rdna4.h"
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
#include <span>
#include <unordered_set>
#include <vector>

namespace rocjitsu {

namespace {

constexpr uint32_t kConservativeLoweringMinimumVgprs = 128;

EncodingTranslateFn select_encoding_translator(rj_code_arch_t guest, rj_code_arch_t host) {
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_RDNA4)
    return cdna4_to_rdna4::translate_encoding_cdna4_to_rdna4;
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
  constexpr uint64_t kMaxSignedTarget =
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
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
  const KdTranslation *translation = nullptr;
  BasicBlock *entry = nullptr;
  std::vector<BasicBlock *> blocks;
};

[[nodiscard]] std::vector<BasicBlock *>
reachable_kernel_blocks(const std::vector<std::unique_ptr<BasicBlock>> &blocks, BasicBlock &entry,
                        const std::unordered_set<uint64_t> &kernel_entries) {
  std::unordered_set<const BasicBlock *> reachable;
  std::vector<BasicBlock *> stack{&entry};

  while (!stack.empty()) {
    BasicBlock *block = stack.back();
    stack.pop_back();
    if (block == nullptr || !reachable.insert(block).second)
      continue;

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
                          std::span<const KdTranslation> kernels) {
  std::vector<KernelTranslationScope> scopes;
  const auto entries = kernel_entry_offsets(kernels);
  if (entries.empty())
    return scopes;

  std::unordered_set<uint64_t> entry_set(entries.begin(), entries.end());
  std::vector<const KdTranslation *> ordered_kernels;
  ordered_kernels.reserve(kernels.size());
  std::unordered_set<uint64_t> seen_entries;
  for (const KdTranslation &kernel : kernels) {
    if (seen_entries.insert(kernel.entry_text_offset).second)
      ordered_kernels.push_back(&kernel);
  }

  std::ranges::sort(ordered_kernels, [](const auto *lhs, const auto *rhs) {
    return lhs->entry_text_offset < rhs->entry_text_offset;
  });

  scopes.reserve(ordered_kernels.size());
  for (const KdTranslation *kernel : ordered_kernels) {
    BasicBlock *entry = block_for_offset(blocks, kernel->entry_text_offset);
    if (entry == nullptr)
      continue;

    scopes.push_back({kernel, entry, reachable_kernel_blocks(blocks, *entry, entry_set)});
  }
  return scopes;
}

} // namespace

BinaryTranslator::~BinaryTranslator() = default;

BinaryTranslator::BinaryTranslator(rj_code_arch_t guest_arch, rj_code_arch_t host_arch,
                                   uint32_t target_mach)
    : guest_arch_(guest_arch), host_arch_(host_arch),
      target_mach_(target_mach ? target_mach : elf_mach_for_arch(host_arch)),
      encoding_translate_(select_encoding_translator(guest_arch, host_arch)),
      legalization_lookup_(select_legalization(guest_arch, host_arch)),
      semantic_translator_(std::make_unique<SemanticTranslator>(guest_arch, host_arch)) {}

TranslatedCodeObject BinaryTranslator::translate(const AmdGpuCodeObject &obj) {
  TranslatedCodeObject result;
  result.host_arch = host_arch_;
  warnings_ = &result.warnings;

  auto leave_unchanged = [&]() {
    const auto *image = reinterpret_cast<const uint8_t *>(obj.image_data());
    result.elf_bytes.assign(image, image + obj.image_size());
    warnings_ = nullptr;
    return result;
  };

  CodeObjectPatcher patcher(obj);
  auto text = patcher.text_bytes();
  if (text.empty()) {
    result.elf_bytes = patcher.emit();
    return result;
  }

  auto decoder = Decoder::create(guest_arch_);
  if (!decoder) {
    result.warnings.push_back("unsupported guest_arch: no decoder available");
    result.elf_bytes = patcher.emit();
    return result;
  }
  KernelDescriptorTranslator descriptor_translator(guest_arch_, host_arch_);
  KernelDescriptorTranslationOptions descriptor_options;
  // Semantic lowerings allocate temporary VGPRs from liveness. Descriptor
  // translation runs before those choices are known, so keep the historical
  // 128-VGPR headroom for now.
  // TODO: Have lowerings report their actual highest temporary VGPR demand and
  // use that instead of this conservative floor.
  descriptor_options.minimum_vgprs = kConservativeLoweringMinimumVgprs;
  const auto descriptor_translations = descriptor_translator.translate_image(
      patcher.image_bytes(), patcher.text_offset(), patcher.text_size(), descriptor_options);
  bool descriptors_supported = true;
  for (const auto &translation : descriptor_translations) {
    result.warnings.insert(result.warnings.end(), translation.warnings.begin(),
                           translation.warnings.end());
    descriptors_supported &= translation.supported;
  }
  if (!descriptors_supported) {
    result.warnings.push_back("kernel descriptor translation requires unsupported resource or ABI "
                              "virtualization; leaving code object unchanged");
    result.elf_bytes = patcher.emit();
    warnings_ = nullptr;
    return result;
  }

  if (descriptor_translations.empty()) {
    result.warnings.push_back("kernel descriptors are required for kernel-level translation");
    warnings_ = nullptr;
    return result;
  }

  const auto entry_offsets = kernel_entry_offsets(descriptor_translations);
  auto blocks = BasicBlock::build(obj, *decoder, entry_offsets);
  auto scopes = kernel_translation_scopes(blocks, descriptor_translations);

  if (scopes.size() != entry_offsets.size()) {
    result.warnings.push_back(
        "kernel descriptor entry offsets are required to map to decoded text blocks");
    warnings_ = nullptr;
    return result;
  }

  std::vector<uint8_t> translated_text(text.begin(), text.end());

  // Code caves live in a separate executable section that is placed immediately
  // after the original .text bytes. Treating that section as a .text-relative
  // continuation keeps existing instruction addresses stable while avoiding any
  // dependency on compiler-emitted NOP padding after s_endpgm.
  patcher.set_cave_start(text.size());

  std::unordered_set<const BasicBlock *> translated_blocks;
  bool warned_shared_blocks = false;
  for (const KernelTranslationScope &scope : scopes) {
    if (scope.blocks.empty())
      continue;

    LivenessAnalysis liveness(KernelBlockScope(scope.blocks));

    for (BasicBlock *block : scope.blocks) {
      if (block == nullptr)
        continue;
      if (!translated_blocks.insert(block).second) {
        if (!warned_shared_blocks) {
          result.warnings.push_back(
              "basic block is reachable from multiple kernel entries; using first kernel "
              "liveness for shared code");
          warned_shared_blocks = true;
        }
        continue;
      }

      uint64_t offset = block->start_offset();
      for (auto it = block->instructions().begin(); it != block->instructions().end(); ++it) {
        const auto &inst = *it;
        const uint32_t inst_size = inst.size();

        const uint32_t *raw = inst.raw_encoding();
        if (!raw) {
          std::memcpy(translated_text.data() + offset, text.data() + offset, inst_size);
          offset += inst_size;
          continue;
        }

        const InstructionLegalization *leg = nullptr;
        if (legalization_lookup_)
          leg = legalization_lookup_(inst.encoding_id(), inst.opcode());

        const uint16_t dst_opcode = leg ? leg->target_opcode : inst.opcode();

        // Try semantic lowering for Expand and Lower actions.
        // For Expand: must lower (NOP-fill if unhandled).
        // For Lower: try lowering first, fall through to encoding if unhandled.
        {
          auto expansion = semantic_translator_->try_lower_expand(inst, offset, liveness);
          if (!expansion.empty()) {
            SemanticReplacement repl{offset, offset + inst_size, std::move(expansion)};
            if (!apply_semantic(repl, translated_text, patcher))
              return leave_unchanged();
            offset += inst_size;
            continue;
          }
        }

        if (leg && leg->action == Action::Expand) {
          result.warnings.push_back("EXPAND not yet implemented for " +
                                    std::string(inst.mnemonic()));
          const uint32_t nop = build_s_nop(0, host_arch_);
          for (uint32_t i = 0; i < inst_size; i += 4)
            std::memcpy(translated_text.data() + offset + i, &nop, 4);
          offset += inst_size;
          continue;
        }

        if (!handle_encoding(inst, offset, translated_text, dst_opcode, patcher, text))
          return leave_unchanged();
        offset += inst_size;
      }
    }
  }

  std::unordered_set<uint64_t> applied_descriptors;
  for (const KdTranslation &translation : descriptor_translations) {
    if (applied_descriptors.insert(translation.descriptor_file_offset).second) {
      if (!patcher.apply_kernel_descriptor_translation(translation, host_arch_)) {
        result.warnings.push_back("kernel descriptor translation could not be applied safely; "
                                  "leaving code object unchanged");
        return leave_unchanged();
      }
    }
  }

  patcher.overwrite_text(translated_text);
  if (!patcher.append_cave_section()) {
    result.warnings.push_back(
        "code cave section could not be materialized safely; leaving code object unchanged");
    return leave_unchanged();
  }

  if (target_mach_)
    patcher.update_elf_flags(target_mach_);

  result.elf_bytes = patcher.emit();
  warnings_ = nullptr;
  return result;
}

bool BinaryTranslator::apply_semantic(const SemanticReplacement &repl, std::vector<uint8_t> &text,
                                      CodeObjectPatcher &patcher) {
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

  const uint64_t cave_byte_offset = patcher.cave_start() + patcher.cave_body_size();
  const uint64_t stub_next = repl.start_offset + source_size;
  const uint64_t branch_pc = repl.start_offset;

  // s_branch simm16 targets (PC + 4 + simm16*4).
  int16_t fwd_dwords = 0;
  if (!compute_sopp_branch_offset(branch_pc, cave_byte_offset, fwd_dwords)) {
    if (warnings_)
      warnings_->push_back("code cave branch range exceeds s_branch simm16; leaving code object "
                           "unchanged");
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
    if (warnings_)
      warnings_->push_back("code cave return branch range exceeds s_branch simm16; leaving code "
                           "object unchanged");
    return false;
  }
  cave_words.push_back(build_s_branch(ret_dwords, host_arch_));

  patcher.append_cave_body(cave_words);
  return true;
}

bool BinaryTranslator::handle_encoding(const Instruction &inst, uint64_t offset,
                                       std::vector<uint8_t> &text, uint16_t dst_opcode,
                                       CodeObjectPatcher &patcher,
                                       std::span<const uint8_t> orig_text) {
  const uint32_t *raw = inst.raw_encoding();
  assert(raw && "handle_encoding called without raw encoding");
  if (!encoding_translate_) {
    std::memcpy(text.data() + offset, raw, inst.size());
    return true;
  }

  const uint32_t w0 = raw[0];
  const uint32_t w1 = inst.size() > 4 ? raw[1] : 0;
  const uint32_t w2 = inst.size() > 8 ? raw[2] : 0;

  auto tr = encoding_translate_(inst.encoding_id(), w0, w1, w2, dst_opcode);

  if (tr.word_count == 0) {
    std::memcpy(text.data() + offset, raw, inst.size());
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
  if (target_size <= orig_bytes) {
    std::memcpy(text.data() + offset, tr.words, target_size);
  } else {
    SemanticReplacement repl{offset, offset + inst.size(), {tr.words, tr.words + tr.word_count}};
    if (!apply_semantic(repl, text, patcher))
      return false;
  }
  return true;
}

} // namespace rocjitsu
