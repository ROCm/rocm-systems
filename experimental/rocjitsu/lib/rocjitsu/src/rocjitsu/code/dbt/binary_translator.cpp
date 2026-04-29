// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/dbt/binary_translator.h"

#include "rocjitsu/analysis/register_liveness.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/dbt/generated/encoding_cdna4_to_rdna3.h"
#include "rocjitsu/code/dbt/generated/encoding_cdna4_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna4_to_rdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna4_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_types.h"
#include "rocjitsu/code/dbt/semantic_translator.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <cassert>
#include <climits>
#include <cstring>

namespace rocjitsu {

namespace {

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
  auto blocks = BasicBlock::build(obj, *decoder);

  std::vector<uint8_t> translated_text(text.size(), 0);

  // Find the end of actual code (after s_endpgm) to place the cave body
  // in the NOP padding. Scan backwards from the end of .text for the first
  // non-NOP instruction to determine where NOP padding starts.
  uint64_t code_end = text.size();
  {
    const auto *data = reinterpret_cast<const uint32_t *>(text.data());
    const size_t words = text.size() / 4;
    // Scan backwards to find the last non-NOP word.
    // s_nop encodes as 0xBF800000 on both CDNA4 and RDNA4.
    for (size_t i = words; i > 0; --i) {
      if (data[i - 1] != 0xBF800000) {
        code_end = i * 4;
        break;
      }
    }
  }
  // Align cave start to 4 bytes (already is, since instructions are 4-byte aligned).
  patcher.set_cave_start(code_end);

  for (const auto &block : blocks) {
    auto liveness = RegisterLiveness::compute(*block);

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
          apply_semantic(repl, translated_text, patcher);
          offset += inst_size;
          continue;
        }
      }

      if (leg && leg->action == Action::Expand) {
        result.warnings.push_back("EXPAND not yet implemented for " + std::string(inst.mnemonic()));
        const uint32_t nop = build_s_nop(0, host_arch_);
        for (uint32_t i = 0; i < inst_size; i += 4)
          std::memcpy(translated_text.data() + offset + i, &nop, 4);
        offset += inst_size;
        continue;
      }

      handle_encoding(inst, offset, translated_text, dst_opcode, patcher, text);
      offset += inst_size;
    }
  }

  // Rewrite workgroup_id SGPR references to TTMP registers.
  // This runs after encoding translation so it reads from translated_text.
  auto wg_info = patcher.workgroup_id_info();
  for (const auto &block : blocks) {
    auto wg_rewrites =
        semantic_translator_->rewrite_workgroup_ids(*block, wg_info, translated_text);
    for (const auto &repl : wg_rewrites)
      apply_semantic(repl, translated_text, patcher);
  }

  // Write cave body into the NOP padding at the end of .text.
  // cave_start was set to the end of actual code (after s_endpgm).
  // The cave body overwrites the NOP padding between code_end and text.size().
  // TODO: When caves exceed available NOP padding, allocate a new .text section
  // instead of asserting. For now, typical kernels have 256+ bytes of NOP padding
  // which is sufficient for the current expansion rules.
  const auto &cave = patcher.cave_body();
  if (!cave.empty()) {
    const uint64_t cave_start = patcher.cave_start();
    if (cave_start + cave.size() > text.size()) {
      result.warnings.push_back("cave body (" + std::to_string(cave.size()) +
                                " bytes) exceeds .text NOP padding (" +
                                std::to_string(text.size() - cave_start) + " bytes available)");
    } else {
      std::memcpy(translated_text.data() + cave_start, cave.data(), cave.size());
    }
  }

  patcher.overwrite_text(translated_text);

  if (target_mach_)
    patcher.update_elf_flags(target_mach_);

  patcher.patch_kernel_descriptors_for_wave64();

  result.elf_bytes = patcher.emit();
  warnings_ = nullptr;
  return result;
}

void BinaryTranslator::apply_semantic(const SemanticReplacement &repl, std::vector<uint8_t> &text,
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
    return;
  }

  const uint64_t cave_byte_offset = patcher.cave_start() + patcher.cave_body_size();
  const uint64_t stub_next = repl.start_offset + source_size;
  const uint64_t branch_pc = repl.start_offset;

  // s_branch simm16 targets (PC + 4 + simm16*4).
  const auto fwd_dwords = static_cast<int64_t>(cave_byte_offset - (branch_pc + 4)) / 4;
  assert(fwd_dwords >= INT16_MIN && fwd_dwords <= INT16_MAX &&
         "branch offset exceeds simm16 range");

  const uint32_t stub = build_s_branch(static_cast<int16_t>(fwd_dwords), host_arch_);
  std::memcpy(text.data() + repl.start_offset, &stub, 4);
  for (uint64_t off = repl.start_offset + 4; off < repl.end_offset; off += 4) {
    const uint32_t nop = build_s_nop(0, host_arch_);
    std::memcpy(text.data() + off, &nop, 4);
  }

  auto cave_words = repl.target_words;
  const auto ret_dwords = (static_cast<int64_t>(stub_next) -
                           static_cast<int64_t>(cave_byte_offset + cave_words.size() * 4 + 4)) /
                          4;
  assert(ret_dwords >= INT16_MIN && ret_dwords <= INT16_MAX &&
         "return branch offset exceeds simm16 range");
  cave_words.push_back(build_s_branch(static_cast<int16_t>(ret_dwords), host_arch_));

  patcher.append_cave_body(cave_words);
}

void BinaryTranslator::handle_encoding(const Instruction &inst, uint64_t offset,
                                       std::vector<uint8_t> &text, uint16_t dst_opcode,
                                       CodeObjectPatcher &patcher,
                                       std::span<const uint8_t> orig_text) {
  const uint32_t *raw = inst.raw_encoding();
  assert(raw && "handle_encoding called without raw encoding");
  if (!encoding_translate_) {
    std::memcpy(text.data() + offset, raw, inst.size());
    return;
  }

  const uint32_t w0 = raw[0];
  const uint32_t w1 = inst.size() > 4 ? raw[1] : 0;
  const uint32_t w2 = inst.size() > 8 ? raw[2] : 0;

  auto tr = encoding_translate_(inst.encoding_id(), w0, w1, w2, dst_opcode);

  if (tr.word_count == 0) {
    std::memcpy(text.data() + offset, raw, inst.size());
    return;
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
    apply_semantic(repl, text, patcher);
  }
}

} // namespace rocjitsu
