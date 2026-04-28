// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/dbt/binary_translator.h"

#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/dbt/generated/encoding_cdna4_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna4_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_types.h"
#include "rocjitsu/code/dbt/sdwa_lowering.h"
#include "rocjitsu/code/dbt/semantic_translator.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <cassert>
#include <climits>
#include <cstring>
#include <optional>

namespace rocjitsu {

namespace {

EncodingTranslateFn select_encoding_translator(rj_code_arch_t guest, rj_code_arch_t host) {
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_RDNA4)
    return translate_encoding_cdna4_to_rdna4;
  return nullptr;
}

LegalizationLookupFn select_legalization(rj_code_arch_t guest, rj_code_arch_t host) {
  if (guest == ROCJITSU_CODE_ARCH_CDNA4 && host == ROCJITSU_CODE_ARCH_RDNA4) {
    return [](uint16_t enc_id, uint16_t opcode) -> const InstructionLegalization * {
      return lookup(kLegalization_cdna4_to_rdna4, enc_id, opcode);
    };
  }
  return nullptr;
}

[[nodiscard]] bool is_cdna_arch(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_CDNA1 || arch == ROCJITSU_CODE_ARCH_CDNA2 ||
         arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4;
}

[[nodiscard]] bool is_rdna_arch(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2 ||
         arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
         arch == ROCJITSU_CODE_ARCH_RDNA4;
}

[[nodiscard]] std::optional<uint8_t>
wavefront_size_from_kernel_descriptors(const AmdGpuCodeObject &obj) {
  using KD = rocr::llvm::amdhsa::kernel_descriptor_t;
  namespace kd = rocr::llvm::amdhsa;

  const auto *image = reinterpret_cast<const uint8_t *>(obj.image_data());
  const size_t image_size = obj.image_size();
  if (image == nullptr || image_size < sizeof(Elf64_Ehdr))
    return std::nullopt;

  const auto *ehdr = reinterpret_cast<const Elf64_Ehdr *>(image);
  if (ehdr->e_shoff + static_cast<uint64_t>(ehdr->e_shnum) * sizeof(Elf64_Shdr) > image_size)
    return std::nullopt;

  const auto *shdr = reinterpret_cast<const Elf64_Shdr *>(image + ehdr->e_shoff);
  std::optional<uint8_t> wf_size;
  for (int i = 0; i < ehdr->e_shnum; ++i) {
    if (shdr[i].sh_type != SHT_SYMTAB && shdr[i].sh_type != SHT_DYNSYM)
      continue;
    if (shdr[i].sh_offset + shdr[i].sh_size > image_size)
      continue;
    if (shdr[i].sh_entsize == 0)
      continue;

    const auto *symtab = reinterpret_cast<const Elf64_Sym *>(image + shdr[i].sh_offset);
    const size_t nsyms = shdr[i].sh_size / sizeof(Elf64_Sym);
    for (size_t j = 0; j < nsyms; ++j) {
      if (symtab[j].st_size != sizeof(KD))
        continue;
      const uint16_t sec_idx = symtab[j].st_shndx;
      if (sec_idx >= ehdr->e_shnum)
        continue;
      if (symtab[j].st_value < shdr[sec_idx].sh_addr)
        continue;

      const uint64_t file_off =
          shdr[sec_idx].sh_offset + (symtab[j].st_value - shdr[sec_idx].sh_addr);
      if (file_off + sizeof(KD) > image_size)
        continue;

      const auto *desc = reinterpret_cast<const KD *>(image + file_off);
      const bool wave32 = AMDHSA_BITS_GET(desc->kernel_code_properties,
                                          kd::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32);
      if (!wave32)
        return 64;
      wf_size = 32;
    }
  }

  return wf_size;
}

[[nodiscard]] uint8_t liveness_wavefront_size(const AmdGpuCodeObject &obj,
                                              rj_code_arch_t guest_arch) {
  if (is_cdna_arch(guest_arch))
    return 64;

  if (is_rdna_arch(guest_arch)) {
    if (auto wf_size = wavefront_size_from_kernel_descriptors(obj))
      return *wf_size;
    return 32;
  }

  return 64;
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
  LivenessAnalysis liveness(blocks, liveness_wavefront_size(obj, guest_arch_));

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

      if (guest_arch_ == ROCJITSU_CODE_ARCH_CDNA4 && host_arch_ == ROCJITSU_CODE_ARCH_RDNA4 &&
          is_cdna4_vop2_sdwa_form(inst)) {
        uint32_t ext_word = 0;
        if (inst_size >= 8)
          std::memcpy(&ext_word, text.data() + offset + 4, 4);
        auto lowering =
            lower_cdna4_vop2_sdwa_to_rdna4(inst, offset, liveness, dst_opcode, ext_word);
        if (!lowering.empty()) {
          SemanticReplacement repl{offset, offset + inst_size, std::move(lowering)};
          apply_semantic(repl, translated_text, patcher);
        } else {
          result.warnings.push_back("unsupported SDWA form: " + std::string(inst.disassemble()));
          const uint32_t nop = build_s_nop(0, host_arch_);
          for (uint32_t i = 0; i < inst_size; i += 4)
            std::memcpy(translated_text.data() + offset + i, &nop, 4);
        }
        offset += inst_size;
        continue;
      }

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
  std::vector<SemanticTranslator::WorkGroupRewriteState> wg_states;
  wg_states.reserve(wg_info.size());
  for (const auto &info : wg_info)
    wg_states.push_back({info, info.sgpr_wg_id_x >= 0});
  for (const auto &block : blocks) {
    auto wg_rewrites =
        semantic_translator_->rewrite_workgroup_ids(*block, wg_states, translated_text);
    for (const auto &repl : wg_rewrites)
      apply_semantic(repl, translated_text, patcher);
  }

  // Write cave body into the NOP padding at the end of .text.
  // cave_start was set to the end of actual code (after s_endpgm).
  // The cave body overwrites the NOP padding between code_end and text.size().
  const auto &cave = patcher.cave_body();
  if (!cave.empty()) {
    const uint64_t cave_start = patcher.cave_start();
    assert(cave_start + cave.size() <= text.size() && "cave body exceeds .text NOP padding");
    std::memcpy(translated_text.data() + cave_start, cave.data(), cave.size());
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
    if (target_size < source_size) {
      // 0x00000000 decodes as v_illegal on RDNA4, so any unused instruction
      // slots in an in-place replacement must be filled with real host NOPs.
      const uint32_t nop = build_s_nop(0, host_arch_);
      for (uint64_t off = repl.start_offset + target_size; off < repl.end_offset; off += 4)
        std::memcpy(text.data() + off, &nop, sizeof(nop));
    }
    return;
  }

  const uint64_t cave_byte_offset = patcher.cave_start() + patcher.cave_offset();
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

  // Append any trailing literal constant words that the encoding translator
  // did not consume. Single-word formats (SOP1, SOP2, VOP1, VOP2, etc.) can
  // have a 32-bit literal appended when a source operand is 0xFF. The
  // encoding translator only translates the opcode word; the literal passes
  // through unchanged. Read from the original text since raw_encoding()
  // only covers the struct portion, not the trailing literal.
  const uint32_t translated_bytes = tr.word_count * 4u;
  const uint32_t orig_bytes = inst.size();
  if (translated_bytes < orig_bytes) {
    const uint32_t literal_words = (orig_bytes - translated_bytes) / 4;
    for (uint32_t i = 0; i < literal_words && tr.word_count < 3; ++i) {
      uint32_t lit_offset = offset + translated_bytes + i * 4;
      uint32_t lit_word;
      std::memcpy(&lit_word, orig_text.data() + lit_offset, 4);
      tr.words[tr.word_count++] = lit_word;
    }
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
