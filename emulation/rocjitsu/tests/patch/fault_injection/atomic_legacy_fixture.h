// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "../../dbi_test_util.h"
#include "barrier_oracle.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/opcodes.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/target_registry.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu::test {

struct LegacyAtomicTargetCase {
  rj_code_target_id_t target;
  uint32_t machine;
  IsaFamily arch;
  uint32_t wave;
  bool cdna;
  const char *name;
};

inline constexpr std::array<LegacyAtomicTargetCase, 4> kLegacyAtomicTargetCases = {{
    {.target = ROCJITSU_CODE_TARGET_GFX950,
     .machine = EF_AMDGPU_MACH_AMDGCN_GFX950,
     .arch = IsaFamily::Cdna4,
     .wave = 64,
     .cdna = true,
     .name = "Gfx950Wave64"},
    {.target = ROCJITSU_CODE_TARGET_GFX942,
     .machine = EF_AMDGPU_MACH_AMDGCN_GFX942,
     .arch = IsaFamily::Cdna3,
     .wave = 64,
     .cdna = true,
     .name = "Gfx942Wave64"},
    {.target = ROCJITSU_CODE_TARGET_GFX1100,
     .machine = EF_AMDGPU_MACH_AMDGCN_GFX1100,
     .arch = IsaFamily::Rdna3,
     .wave = 32,
     .cdna = false,
     .name = "Gfx1100Wave32"},
    {.target = ROCJITSU_CODE_TARGET_GFX1100,
     .machine = EF_AMDGPU_MACH_AMDGCN_GFX1100,
     .arch = IsaFamily::Rdna3,
     .wave = 64,
     .cdna = false,
     .name = "Gfx1100Wave64"},
}};

enum class LegacyAtomicForm { FlatAdd, GlobalAdd, BufferAdd, FlatCas, GlobalCas, BufferCas };

inline constexpr std::array<LegacyAtomicForm, 6> kLegacyAtomicForms = {
    LegacyAtomicForm::FlatAdd, LegacyAtomicForm::GlobalAdd, LegacyAtomicForm::BufferAdd,
    LegacyAtomicForm::FlatCas, LegacyAtomicForm::GlobalCas, LegacyAtomicForm::BufferCas};

inline bool legacy_atomic_buffer(LegacyAtomicForm form) {
  return form == LegacyAtomicForm::BufferAdd || form == LegacyAtomicForm::BufferCas;
}

inline bool legacy_atomic_global(LegacyAtomicForm form) {
  return form == LegacyAtomicForm::GlobalAdd || form == LegacyAtomicForm::GlobalCas;
}

inline bool legacy_atomic_compare(LegacyAtomicForm form) {
  return form == LegacyAtomicForm::FlatCas || form == LegacyAtomicForm::GlobalCas ||
         form == LegacyAtomicForm::BufferCas;
}

inline bool legacy_atomic_signed(const LegacyAtomicTargetCase &, LegacyAtomicForm form) {
  return legacy_atomic_global(form);
}

inline std::string legacy_atomic_mnemonic(const LegacyAtomicTargetCase &target,
                                          LegacyAtomicForm form) {
  const std::string prefix = legacy_atomic_buffer(form)   ? "buffer_atomic_"
                             : legacy_atomic_global(form) ? "global_atomic_"
                                                          : "flat_atomic_";
  if (target.cdna)
    return prefix + (legacy_atomic_compare(form) ? "cmpswap" : "add");
  return prefix + (legacy_atomic_compare(form) ? "cmpswap_b32" : "add_u32");
}

// Literal opcode/operand/cache fixtures, independent of generated builders and
// production displacement helpers. CDNA GLOBAL clears SVE and uses s[4:5];
// FLAT clears its unused SADDR field, matching LLVM-emitted encodings. MUBUF uses s[8:11], s7 and
// v[4:5].
inline std::array<uint32_t, 2> legacy_atomic_words(const LegacyAtomicTargetCase &target,
                                                   LegacyAtomicForm form, int32_t byte_offset = 4,
                                                   bool returning = true) {
  const bool buffer = legacy_atomic_buffer(form);
  const bool global = legacy_atomic_global(form);
  const bool compare = legacy_atomic_compare(form);
  if (!target.cdna) {
    const uint32_t base = buffer   ? (compare ? 0xe0d07000u : 0xe0d47000u)
                          : global ? (compare ? 0xdcd2e000u : 0xdcd6e000u)
                                   : (compare ? 0xdcd0e000u : 0xdcd4e000u);
    const uint32_t mask = global ? 0x1fffu : 0xfffu;
    const uint32_t vdst = returning ? 0x06000000u : 0u;
    return {(base & ~(mask | 0x4000u)) | (returning ? 0x4000u : 0u) |
                (static_cast<uint32_t>(byte_offset) & mask),
            buffer   ? 0x07c20204u
            : global ? vdst | 0x00040204u
                     : vdst | 0x007c0204u};
  }
  const uint32_t word0 = buffer   ? (compare ? 0xe106f000u : 0xe10af000u)
                         : global ? (compare ? 0xdf078000u : 0xdf0b8000u)
                                  : (compare ? 0xdf070000u : 0xdf0b0000u);
  const uint32_t return_mask = buffer ? 0x4000u : 0x10000u;
  const uint32_t mask = global ? 0x1fffu : 0xfffu;
  return {(word0 & ~(mask | return_mask)) | (returning ? return_mask : 0u) |
              (static_cast<uint32_t>(byte_offset) & mask),
          buffer   ? 0x07020204u
          : global ? 0x06040204u
                   : 0x06000204u};
}

inline std::vector<uint8_t> make_legacy_atomic_sequence_elf(const LegacyAtomicTargetCase &target,
                                                            std::vector<uint32_t> words) {
  words.push_back(target.cdna ? 0xbf8c0000u : 0xbf890000u); // s_waitcnt all counters zero.
  words.push_back(build_s_endpgm(target.arch).value());
  std::vector<uint8_t> image = make_amdgpu_kernel_elf(words, 0, 3, target.machine, 3, 0, false,
                                                      false, false, false, target.wave == 32);
  image[EI_DATA] = ELFDATA2LSB;
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  return image;
}

inline std::vector<uint8_t>
make_legacy_atomic_elf(const LegacyAtomicTargetCase &target,
                       LegacyAtomicForm form = LegacyAtomicForm::FlatAdd, int32_t byte_offset = 4,
                       bool returning = true) {
  const std::array<uint32_t, 2> words = legacy_atomic_words(target, form, byte_offset, returning);
  return make_legacy_atomic_sequence_elf(target, {words[0], words[1]});
}

inline std::array<uint32_t, 2> legacy_wide_ds_atomic_words(const LegacyAtomicTargetCase &target) {
  switch (target.arch) {
  case IsaFamily::Cdna3:
    return cdna3::build_ds(cdna3::kDsAddU64Ds, {.offset0 = 4, .addr = 2, .data0 = 4});
  case IsaFamily::Cdna4:
    return cdna4::build_ds(cdna4::kDsAddU64Ds, {.offset0 = 4, .addr = 2, .data0 = 4});
  case IsaFamily::Rdna3:
    return rdna3::build_ds(rdna3::kDsAddU64Ds, {.offset0 = 4, .addr = 2, .data0 = 4});
  default:
    return {};
  }
}

// A valid DS64 atomic remains selectable but unsupported ahead of three supported forms.
inline std::vector<uint8_t> make_mixed_legacy_atomic_elf(const LegacyAtomicTargetCase &target) {
  const std::array<uint32_t, 2> wide = legacy_wide_ds_atomic_words(target);
  const std::array<uint32_t, 2> buffer = legacy_atomic_words(target, LegacyAtomicForm::BufferCas);
  const std::array<uint32_t, 2> global = legacy_atomic_words(target, LegacyAtomicForm::GlobalAdd);
  const std::array<uint32_t, 2> flat = legacy_atomic_words(target, LegacyAtomicForm::FlatCas);
  return make_legacy_atomic_sequence_elf(
      target, {wide[0], wide[1], buffer[0], buffer[1], global[0], global[1], flat[0], flat[1]});
}

// FLAT atomics have no scratch segment: this entire input must fail validation.
inline std::vector<uint8_t>
make_invalid_scratch_legacy_atomic_elf(const LegacyAtomicTargetCase &target) {
  std::array<uint32_t, 2> scratch = legacy_atomic_words(target, LegacyAtomicForm::GlobalAdd);
  const uint32_t shift = target.cdna ? 14 : 16;
  scratch[0] = (scratch[0] & ~(3u << shift)) | (1u << shift);
  const std::array<uint32_t, 2> buffer = legacy_atomic_words(target, LegacyAtomicForm::BufferCas);
  const std::array<uint32_t, 2> global = legacy_atomic_words(target, LegacyAtomicForm::GlobalAdd);
  const std::array<uint32_t, 2> flat = legacy_atomic_words(target, LegacyAtomicForm::FlatCas);
  return make_legacy_atomic_sequence_elf(target, {scratch[0], scratch[1], buffer[0], buffer[1],
                                                  global[0], global[1], flat[0], flat[1]});
}

inline std::vector<uint8_t> make_shared_legacy_atomic_elf(const LegacyAtomicTargetCase &target,
                                                          bool second_owner = true) {
  const uint32_t nop = build_s_nop(0, target.arch).value();
  const uint32_t end = build_s_endpgm(target.arch).value();
  std::vector<uint32_t> words(128, nop);
  for (uint32_t entry : {0u, 256u}) {
    const size_t start = entry / 4;
    if (entry == 256 && !second_owner) {
      words[start] = end;
    } else {
      words[start] = build_s_call_b64(30, (512 - entry - 4) / 4, target.arch).value();
      words[start + 1] = end;
    }
  }
  const std::array<uint32_t, 2> atomic =
      legacy_atomic_words(target, LegacyAtomicForm::GlobalCas, -4);
  words.insert(words.end(), atomic.begin(), atomic.end());
  words.push_back(target.cdna ? 0xbf8c0000u : 0xbf890000u);
  words.push_back(build_s_setpc_b64(30, target.arch).value());
  std::vector<uint8_t> image = make_amdgpu_two_kernel_elf(words, 0, 3, target.machine,
                                                          {target.wave == 32, target.wave == 32});
  image[EI_DATA] = ELFDATA2LSB;
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  const AmdGpuCodeObject source(image.data(), image.size());
  const Section &text = *source.text_sections().front();
  const std::vector<KernelDescriptorInfo> descriptors =
      scan_kernel_descriptors(image, text.sectionOffset(), text.size(), text.sectionHeaderIndex());
  for (const KernelDescriptorInfo &kernel : descriptors) {
    if (kernel.kernel_name != "test_kernel1")
      continue;
    amdhsa::kernel_descriptor_t descriptor = kernel.descriptor;
    descriptor.kernel_code_entry_byte_offset += 256;
    std::memcpy(image.data() + kernel.descriptor_file_offset, &descriptor, sizeof(descriptor));
  }
  return image;
}

inline std::optional<uint64_t> legacy_atomic_guest(const Section &text, uint64_t source_offset,
                                                   rj_code_target_id_t target) {
  if (source_offset > text.size() || text.size() - source_offset < 4 ||
      source_offset > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) - 4)
    return std::nullopt;
  uint32_t word = 0;
  std::memcpy(&word, text.data() + source_offset, sizeof(word));
  std::unique_ptr<Decoder> decoder = Decoder::create(default_isa_target_registry(), target);
  if (!decoder)
    return std::nullopt;
  const DecodeResult branch = decoder->decode_window(std::span(&word, 1), source_offset);
  if (branch.failed() || branch.value()->mnemonic() != "s_branch")
    return std::nullopt;
  const std::optional<int64_t> delta = branch.value()->branch_offset_bytes();
  const int64_t base = static_cast<int64_t>(source_offset + 4);
  if (!delta || (*delta > 0 && base > std::numeric_limits<int64_t>::max() - *delta))
    return std::nullopt;
  const int64_t guest = base + *delta;
  if (guest < 0 || static_cast<uint64_t>(guest) > text.size())
    return std::nullopt;
  return static_cast<uint64_t>(guest);
}

// Infer the documented legacy field from source words, independently of both FI
// and B2. The expected writer shares ELF layout machinery only. Verify complete
// source splices, 8-byte guests, continuations, waits, metadata and operands.
inline bool legacy_atomic_output_matches(const FaultMutationContext &source, uint64_t selected,
                                         uint32_t delta, std::span<const uint8_t> candidate) {
  if (!source.source().is_valid() || source.source().text_sections().size() != 1)
    return false;
  const Section &before_text = *source.source().text_sections().front();
  if (selected > before_text.size() || before_text.size() - selected < 8)
    return false;
  std::array<uint32_t, 2> original = {};
  std::memcpy(original.data(), before_text.data() + selected, sizeof(original));
  const bool cdna = source.source().target_id() == ROCJITSU_CODE_TARGET_GFX950 ||
                    source.source().target_id() == ROCJITSU_CODE_TARGET_GFX942;
  const uint32_t prefix = original[0] & 0xfc000000u;
  const uint32_t segment = (original[0] >> (cdna ? 14 : 16)) & 3u;
  if ((prefix != 0xe0000000u && prefix != 0xdc000000u) ||
      (prefix == 0xdc000000u && segment != 0 && segment != 2))
    return false;
  const bool signed_offset = prefix == 0xdc000000u && segment == 2;
  const uint32_t mask = signed_offset ? 0x1fffu : 0xfffu;
  const uint32_t encoded = original[0] & mask;
  const int64_t old_offset =
      signed_offset && encoded >= 0x1000u ? static_cast<int64_t>(encoded) - 0x2000 : encoded;
  const int64_t final_offset = old_offset + delta;
  if (final_offset < (signed_offset ? -4096 : 0) || final_offset > 4095)
    return false;
  const std::array<uint32_t, 2> replacement = {
      (original[0] & ~mask) | (static_cast<uint32_t>(final_offset) & mask), original[1]};
  std::vector<uint8_t> expected_bytes(sizeof(original));
  std::vector<uint8_t> replacement_bytes(sizeof(replacement));
  std::memcpy(expected_bytes.data(), original.data(), sizeof(original));
  std::memcpy(replacement_bytes.data(), replacement.data(), sizeof(replacement));
  const std::array<ExpectedReplacement, 1> edits = {ExpectedReplacement{
      .offset = selected, .expected = expected_bytes, .replacement = replacement_bytes}};
  const IsaTargetDescriptor *target =
      default_isa_target_registry().find(source.source().target_id());
  if (!target)
    return false;
  const std::optional<std::vector<uint8_t>> expected =
      expected_replacement_image(source.source(), edits, target->family);
  if (!expected || !std::ranges::equal(*expected, candidate))
    return false;
  const FaultMutationContext mutant(candidate);
  if (!mutant.source().is_valid() || mutant.source().text_sections().size() != 1 ||
      mutant.source().target_id() != source.source().target_id() ||
      !descriptors_preserved(source, mutant))
    return false;
  const Section &text = *mutant.source().text_sections().front();
  const std::optional<uint64_t> guest =
      legacy_atomic_guest(text, selected, source.source().target_id());
  if (!guest || *guest > text.size() || text.size() - *guest < 12)
    return false;
  std::unique_ptr<Decoder> decoder =
      Decoder::create(default_isa_target_registry(), source.source().target_id());
  if (!decoder || !branch_reaches(*decoder, text, selected, *guest) ||
      !branch_reaches(*decoder, text, *guest + 8, selected + 8))
    return false;
  std::array<uint32_t, 2> actual = {};
  std::memcpy(actual.data(), text.data() + *guest, sizeof(actual));
  const DecodeResult before = decoder->decode_window(original, selected);
  const DecodeResult after = decoder->decode_window(actual, *guest);
  if (before.failed() || after.failed() || before.value()->size() != 8 ||
      after.value()->size() != 8 || before.value()->mnemonic() != after.value()->mnemonic() ||
      before.value()->opcode() != after.value()->opcode() ||
      before.value()->num_src_operands() != after.value()->num_src_operands() ||
      before.value()->num_dst_operands() != after.value()->num_dst_operands())
    return false;
  for (int index = 0; index < before.value()->num_operands(); ++index) {
    const bool input = index < before.value()->num_src_operands();
    const int operand = input ? index : index - before.value()->num_src_operands();
    const Operand *old_operand =
        input ? before.value()->src_operand(operand) : before.value()->dst_operand(operand);
    const Operand *new_operand =
        input ? after.value()->src_operand(operand) : after.value()->dst_operand(operand);
    if (!old_operand || !new_operand || old_operand->name() != new_operand->name() ||
        old_operand->encoding_value() != new_operand->encoding_value() ||
        old_operand->size_bits() != new_operand->size_bits() ||
        old_operand->decoded_vgpr() != new_operand->decoded_vgpr() ||
        old_operand->register_byte_mask() != new_operand->register_byte_mask())
      return false;
  }
  return actual == replacement;
}

} // namespace rocjitsu::test
