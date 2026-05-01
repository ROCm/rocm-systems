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
#include <sstream>
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

std::string hex_u64(uint64_t value) {
  std::ostringstream out;
  out << "0x" << std::hex << value;
  return out.str();
}

void write_words_with_nop_padding(std::vector<uint8_t> &text, uint64_t offset,
                                  std::span<const uint32_t> words, uint64_t source_size,
                                  rj_code_arch_t arch) {
  const uint64_t target_size = static_cast<uint64_t>(words.size()) * sizeof(uint32_t);
  if (target_size > 0)
    std::memcpy(text.data() + offset, words.data(), target_size);

  const uint32_t nop = build_s_nop(0, arch);
  for (uint64_t off = target_size; off < source_size; off += sizeof(uint32_t))
    std::memcpy(text.data() + offset + off, &nop, sizeof(nop));
}

// The DBT keeps all original instruction offsets stable. Expansions are placed
// in the trailing source NOP padding, so this returns the first byte of that
// padding by scanning backward to the last real source instruction.
uint64_t find_trailing_nop_cave_start(std::span<const uint8_t> text, rj_code_arch_t guest_arch) {
  const uint32_t nop = build_s_nop(0, guest_arch);
  uint64_t code_end = text.size();
  for (size_t word = text.size() / sizeof(uint32_t); word > 0; --word) {
    uint32_t value = 0;
    std::memcpy(&value, text.data() + (word - 1) * sizeof(uint32_t), sizeof(value));
    if (value != nop) {
      code_end = word * sizeof(uint32_t);
      break;
    }
  }
  return code_end;
}

bool signed_byte_delta(uint64_t target, uint64_t base, int64_t &delta) {
  if (target >= base) {
    const uint64_t diff = target - base;
    if (diff > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
      return false;
    delta = static_cast<int64_t>(diff);
    return true;
  }

  const uint64_t diff = base - target;
  if (diff > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    return false;
  delta = -static_cast<int64_t>(diff);
  return true;
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

  // Expansion failures must not emit a partially patched ELF: a branch stub
  // without a valid cave body is worse than a clear diagnostic.
  auto fail_closed = [&]() {
    result.elf_bytes.clear();
    warnings_ = nullptr;
    return result;
  };

  CodeObjectPatcher patcher(obj);
  auto text = patcher.text_bytes();
  if (text.empty()) {
    result.elf_bytes = patcher.emit();
    warnings_ = nullptr;
    return result;
  }
  if (text.size() % sizeof(uint32_t) != 0) {
    result.warnings.push_back("fatal: .text size " + std::to_string(text.size()) +
                              " is not instruction-aligned");
    return fail_closed();
  }

  auto decoder = Decoder::create(guest_arch_);
  if (!decoder) {
    result.warnings.push_back("unsupported guest_arch: no decoder available");
    result.elf_bytes = patcher.emit();
    warnings_ = nullptr;
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

  patcher.set_cave_start(find_trailing_nop_cave_start(text, guest_arch_));

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

        // Try semantic lowering for Expand and Lower actions.
        // For Expand: must lower; unhandled expansion is a fail-closed diagnostic.
        // For Lower: try lowering first, fall through to encoding if unhandled.
        {
          auto expansion = semantic_translator_->try_lower_expand(inst, offset, liveness);
          if (!expansion.empty()) {
            SemanticReplacement repl{offset, offset + inst_size, std::move(expansion)};
            if (!apply_semantic(repl, translated_text, patcher, inst.mnemonic()))
              return fail_closed();
            offset += inst_size;
            continue;
          }
        }

        if (leg && leg->action == Action::Expand) {
          result.warnings.push_back("fatal: unsupported expansion for " +
                                    std::string(inst.mnemonic()) + " at .text+" +
                                    hex_u64(offset) + " (encoding_id=" +
                                    std::to_string(inst.encoding_id()) +
                                    ", opcode=" + std::to_string(inst.opcode()) + ")");
          return fail_closed();
        }

        if (!handle_encoding(inst, offset, translated_text, leg, patcher, text))
          return fail_closed();
        offset += inst_size;
      }
    }
  }

  std::unordered_set<uint64_t> applied_descriptors;
  for (const KdTranslation &translation : descriptor_translations) {
    if (applied_descriptors.insert(translation.descriptor_file_offset).second) {
      if (!patcher.apply_kernel_descriptor_translation(translation, host_arch_)) {
        result.warnings.push_back(
            "fatal: kernel descriptor translation could not be applied safely");
        return fail_closed();
      }
    }
  }

  // Write the preflighted cave body into the trailing NOP padding at the end
  // of .text. Exhaustion is fatal because branch stubs would otherwise jump
  // to bytes that were never populated with the expanded instruction stream.
  const auto &cave = patcher.cave_body();
  if (!cave.empty()) {
    const uint64_t cave_start = patcher.cave_start();
    if (cave_start > text.size() || cave.size() > text.size() - cave_start) {
      result.warnings.push_back("fatal: code cave exhausted while emitting cave body at .text+" +
                                hex_u64(cave_start) + " (needed " + std::to_string(cave.size()) +
                                " bytes, capacity " + std::to_string(patcher.cave_capacity()) +
                                " bytes)");
      return fail_closed();
    }
    std::memcpy(translated_text.data() + cave_start, cave.data(), cave.size());
  }

  patcher.overwrite_text(translated_text);

  if (target_mach_)
    patcher.update_elf_flags(target_mach_);

  result.elf_bytes = patcher.emit();
  warnings_ = nullptr;
  return result;
}

bool BinaryTranslator::apply_semantic(const SemanticReplacement &repl, std::vector<uint8_t> &text,
                                      CodeObjectPatcher &patcher, std::string_view context) {
  auto fail = [this, context](const std::string &message) {
    if (warnings_)
      warnings_->push_back("fatal: expansion failed for " + std::string(context) + ": " + message);
    return false;
  };

  if (!repl.matched())
    return fail("empty replacement");
  if (repl.start_offset >= repl.end_offset || repl.end_offset > text.size())
    return fail("invalid replacement range [.text+" + hex_u64(repl.start_offset) + ", .text+" +
                hex_u64(repl.end_offset) + ")");
  if ((repl.start_offset % sizeof(uint32_t)) != 0 || (repl.end_offset % sizeof(uint32_t)) != 0)
    return fail("replacement range is not instruction-aligned");

  constexpr uint64_t kDwordBytes = sizeof(uint32_t);
  if (repl.target_words.size() > std::numeric_limits<uint64_t>::max() / kDwordBytes)
    return fail("replacement is too large");
  const uint64_t source_size = repl.end_offset - repl.start_offset;
  const uint64_t target_size = static_cast<uint64_t>(repl.target_words.size()) * sizeof(uint32_t);

  if (patcher.text_range_has_relocation(repl.start_offset, repl.end_offset))
    return fail("source range [.text+" + hex_u64(repl.start_offset) + ", .text+" +
                hex_u64(repl.end_offset) + ") has a relocation target");

  if (target_size <= source_size) {
    write_words_with_nop_padding(text, repl.start_offset, repl.target_words, source_size,
                                 host_arch_);
    return true;
  }

  // Larger replacements use an in-place branch stub and a cave body. This keeps
  // existing branch offsets, metadata offsets, and basic-block starts valid.
  if (source_size < kDwordBytes)
    return fail("source range is too small for a branch stub at .text+" +
                hex_u64(repl.start_offset));

  const uint64_t cave_body_size = patcher.cave_body_size();
  if (patcher.cave_start() > std::numeric_limits<uint64_t>::max() - cave_body_size)
    return fail("code cave offset overflow");
  const uint64_t cave_byte_offset = patcher.cave_start() + cave_body_size;
  if (target_size > std::numeric_limits<uint64_t>::max() - kDwordBytes)
    return fail("replacement is too large for code cave");
  const uint64_t cave_payload_bytes = target_size + kDwordBytes;
  if (cave_byte_offset > std::numeric_limits<uint64_t>::max() - cave_payload_bytes)
    return fail("code cave offset overflow");
  const uint64_t cave_end = cave_byte_offset + cave_payload_bytes;
  const bool needed_overflows =
      cave_body_size > std::numeric_limits<uint64_t>::max() - cave_payload_bytes;
  const uint64_t needed_bytes =
      needed_overflows ? std::numeric_limits<uint64_t>::max() : cave_body_size + cave_payload_bytes;
  if (needed_overflows || needed_bytes > patcher.cave_capacity()) {
    return fail("code cave exhausted at .text+" + hex_u64(patcher.cave_start()) + " (needed " +
                std::to_string(needed_bytes) + " bytes, capacity " +
                std::to_string(patcher.cave_capacity()) + " bytes)");
  }
  if (patcher.text_range_has_relocation(cave_byte_offset, cave_end))
    return fail("code cave range [.text+" + hex_u64(cave_byte_offset) + ", .text+" +
                hex_u64(cave_end) + ") has a relocation target");

  const uint64_t stub_next = repl.end_offset;
  const uint64_t branch_pc = repl.start_offset;

  // s_branch simm16 is relative to the next dword: PC + 4 + simm16*4.
  if (branch_pc > std::numeric_limits<uint64_t>::max() - kDwordBytes)
    return fail("forward branch source offset overflow");
  int64_t fwd_bytes = 0;
  if (!signed_byte_delta(cave_byte_offset, branch_pc + kDwordBytes, fwd_bytes))
    return fail("forward branch byte delta overflow");
  if ((fwd_bytes % static_cast<int64_t>(sizeof(uint32_t))) != 0)
    return fail("forward branch target .text+" + hex_u64(cave_byte_offset) +
                " is not dword-aligned");
  const auto fwd_dwords = fwd_bytes / static_cast<int64_t>(sizeof(uint32_t));
  if (fwd_dwords < std::numeric_limits<int16_t>::min() ||
      fwd_dwords > std::numeric_limits<int16_t>::max())
    return fail("forward branch from .text+" + hex_u64(branch_pc) + " to cave .text+" +
                hex_u64(cave_byte_offset) + " exceeds s_branch simm16 range");

  const uint32_t stub = build_s_branch(static_cast<int16_t>(fwd_dwords), host_arch_);
  std::memcpy(text.data() + repl.start_offset, &stub, 4);
  for (uint64_t off = repl.start_offset + 4; off < repl.end_offset; off += 4) {
    const uint32_t nop = build_s_nop(0, host_arch_);
    std::memcpy(text.data() + off, &nop, 4);
  }

  auto cave_words = repl.target_words;
  // The cave body ends with a return branch to the first source instruction
  // after the replaced range.
  int64_t ret_bytes = 0;
  if (!signed_byte_delta(stub_next, cave_end, ret_bytes))
    return fail("return branch byte delta overflow");
  if ((ret_bytes % static_cast<int64_t>(sizeof(uint32_t))) != 0)
    return fail("return branch target .text+" + hex_u64(stub_next) + " is not dword-aligned");
  const auto ret_dwords = ret_bytes / static_cast<int64_t>(sizeof(uint32_t));
  if (ret_dwords < std::numeric_limits<int16_t>::min() ||
      ret_dwords > std::numeric_limits<int16_t>::max())
    return fail("return branch from cave .text+" +
                hex_u64(cave_end - static_cast<uint64_t>(sizeof(uint32_t))) + " to .text+" +
                hex_u64(stub_next) + " exceeds s_branch simm16 range");
  cave_words.push_back(build_s_branch(static_cast<int16_t>(ret_dwords), host_arch_));

  patcher.append_cave_body(cave_words);
  return true;
}

bool BinaryTranslator::handle_encoding(const Instruction &inst, uint64_t offset,
                                       std::vector<uint8_t> &text,
                                       const InstructionLegalization *leg,
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
  const uint16_t dst_opcode = leg ? leg->target_opcode : inst.opcode();

  auto tr = encoding_translate_(inst.encoding_id(), w0, w1, w2, dst_opcode);

  if (tr.word_count == 0) {
    if (warnings_ && leg && leg->action != Action::Identity) {
      warnings_->push_back("encoding translation missing for " + std::string(inst.mnemonic()));
    }
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
  if (translated_bytes <= orig_bytes && orig_bytes - translated_bytes == 4 && tr.word_count < 3) {
    uint32_t lit_word;
    std::memcpy(&lit_word, orig_text.data() + offset + translated_bytes, 4);
    tr.words[tr.word_count++] = lit_word;
  }

  const uint32_t target_size = tr.word_count * 4u;
  if (target_size <= orig_bytes) {
    write_words_with_nop_padding(text, offset, std::span<const uint32_t>(tr.words, tr.word_count),
                                 orig_bytes, host_arch_);
  } else {
    SemanticReplacement repl{offset, offset + inst.size(), {tr.words, tr.words + tr.word_count}};
    if (!apply_semantic(repl, text, patcher, inst.mnemonic()))
      return false;
  }
  return true;
}

} // namespace rocjitsu
