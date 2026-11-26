// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "../../dbi_test_util.h"
#include "barrier_oracle.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_selectors.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace rocjitsu::test {

struct AtomicTargetCase {
  rj_code_target_id_t target;
  uint32_t machine;
  IsaFamily arch;
  uint32_t wave;
  const char *name;
};
inline constexpr std::array<AtomicTargetCase, 6> kAtomicTargetCases = {{
    {.target = ROCJITSU_CODE_TARGET_GFX1250,
     .machine = EF_AMDGPU_MACH_AMDGCN_GFX1250,
     .arch = IsaFamily::Cdna5,
     .wave = 32,
     .name = "Gfx1250Wave32"},
    {.target = ROCJITSU_CODE_TARGET_GFX1251,
     .machine = EF_AMDGPU_MACH_AMDGCN_GFX1251,
     .arch = IsaFamily::Cdna5,
     .wave = 32,
     .name = "Gfx1251Wave32"},
    {.target = ROCJITSU_CODE_TARGET_GFX1201,
     .machine = EF_AMDGPU_MACH_AMDGCN_GFX1201,
     .arch = IsaFamily::Rdna4,
     .wave = 32,
     .name = "Gfx1201Wave32"},
    {.target = ROCJITSU_CODE_TARGET_GFX1201,
     .machine = EF_AMDGPU_MACH_AMDGCN_GFX1201,
     .arch = IsaFamily::Rdna4,
     .wave = 64,
     .name = "Gfx1201Wave64"},
    {.target = ROCJITSU_CODE_TARGET_GFX1200,
     .machine = EF_AMDGPU_MACH_AMDGCN_GFX1200,
     .arch = IsaFamily::Rdna4,
     .wave = 32,
     .name = "Gfx1200Wave32"},
    {.target = ROCJITSU_CODE_TARGET_GFX1200,
     .machine = EF_AMDGPU_MACH_AMDGCN_GFX1200,
     .arch = IsaFamily::Rdna4,
     .wave = 64,
     .name = "Gfx1200Wave64"},
}};

enum class AtomicTestForm { FlatAdd, GlobalAdd, FlatCas, GlobalCas };

inline std::array<uint32_t, 3> atomic_opcode_test_words(const AtomicTargetCase &fixture,
                                                        uint16_t opcode, bool global,
                                                        int32_t offset = 0, bool scale = false) {
  const uint32_t encoded_offset = static_cast<uint32_t>(offset) & 0xffffffu;
  if (fixture.arch == IsaFamily::Cdna5) {
    if (global)
      return cdna5::build_vglobal(opcode, {.saddr = amdgpu::kModernNullSelector,
                                           .vdst = 6,
                                           .scale_offset = static_cast<uint8_t>(scale),
                                           .scope = 2,
                                           .th = 1,
                                           .vsrc = 4,
                                           .vaddr = 2,
                                           .ioffset = encoded_offset});
    return cdna5::build_vflat(opcode, {.saddr = amdgpu::kModernNullSelector,
                                       .vdst = 6,
                                       .scale_offset = static_cast<uint8_t>(scale),
                                       .scope = 2,
                                       .th = 1,
                                       .vsrc = 4,
                                       .vaddr = 2,
                                       .ioffset = encoded_offset});
  }
  if (global)
    return rdna4::build_vglobal(opcode, {.saddr = amdgpu::kModernNullSelector,
                                         .vdst = 6,
                                         .scope = 2,
                                         .th = 1,
                                         .vsrc = 4,
                                         .vaddr = 2,
                                         .ioffset = encoded_offset});
  return rdna4::build_vflat(opcode, {.saddr = amdgpu::kModernNullSelector,
                                     .vdst = 6,
                                     .scope = 2,
                                     .th = 1,
                                     .vsrc = 4,
                                     .vaddr = 2,
                                     .ioffset = encoded_offset});
}

inline std::array<uint32_t, 3> atomic_test_words(const AtomicTargetCase &fixture,
                                                 AtomicTestForm form = AtomicTestForm::FlatAdd,
                                                 int32_t offset = 0, bool scale = false) {
  const bool global = form == AtomicTestForm::GlobalAdd || form == AtomicTestForm::GlobalCas;
  const bool compare_exchange =
      form == AtomicTestForm::FlatCas || form == AtomicTestForm::GlobalCas;
  const uint16_t opcode =
      compare_exchange ? rdna4::kFlatAtomicCmpswapB32Vflat : rdna4::kFlatAtomicAddU32Vflat;
  return atomic_opcode_test_words(fixture, opcode, global, offset, scale);
}

inline std::vector<uint8_t> make_atomic_sequence_elf(const AtomicTargetCase &fixture,
                                                     std::vector<uint32_t> words) {
  words.push_back(fixture.arch == IsaFamily::Cdna5
                      ? cdna5::build_sopp(cdna5::kSWaitLoadcntSopp)[0]
                      : rdna4::build_sopp(rdna4::kSWaitLoadcntSopp)[0]);
  words.push_back(build_s_endpgm(fixture.arch).value());
  std::vector<uint8_t> image = make_amdgpu_kernel_elf(words, 0, 3, fixture.machine, 3, 0, false,
                                                      false, false, false, fixture.wave == 32);
  image[EI_DATA] = ELFDATA2LSB;
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  return image;
}

inline std::vector<uint8_t> make_atomic_elf(const AtomicTargetCase &fixture,
                                            AtomicTestForm form = AtomicTestForm::FlatAdd,
                                            int32_t offset = 0, bool scale = false) {
  const std::array<uint32_t, 3> atomic = atomic_test_words(fixture, form, offset, scale);
  return make_atomic_sequence_elf(fixture, std::vector<uint32_t>(atomic.begin(), atomic.end()));
}

inline std::vector<uint8_t> make_mixed_atomic_elf(const AtomicTargetCase &fixture) {
  const std::array<uint32_t, 3> buffer =
      fixture.arch == IsaFamily::Cdna5
          ? cdna5::build_vbuffer(
                cdna5::kBufferAtomicAddU32Vbuffer,
                {.vdata = 6, .rsrc = 4, .scope = 2, .th = 1, .offen = 1, .vaddr = 2})
          : rdna4::build_vbuffer(
                rdna4::kBufferAtomicAddU32Vbuffer,
                {.vdata = 6, .rsrc = 4, .scope = 2, .th = 1, .offen = 1, .vaddr = 2});
  const std::array<uint32_t, 2> ds = {0xd8800004u, 0x06000402u};
  const std::array<uint32_t, 3> flat = atomic_test_words(fixture);
  return make_atomic_sequence_elf(
      fixture, {buffer[0], buffer[1], buffer[2], ds[0], ds[1], flat[0], flat[1], flat[2]});
}

inline std::vector<uint8_t> make_shared_atomic_elf(const AtomicTargetCase &fixture,
                                                   bool indirect = false,
                                                   bool second_owner = true) {
  const IsaFamily arch = fixture.arch;
  const uint32_t nop = build_s_nop(0, arch).value();
  const uint32_t end = build_s_endpgm(arch).value();
  std::vector<uint32_t> words(128, nop);
  for (uint32_t entry : {0u, 256u}) {
    const size_t start = entry / 4;
    if (entry == 256 && !second_owner) {
      words[start] = end;
    } else if (!indirect) {
      words[start] = build_s_call_b64(30, (512 - entry - 4) / 4, arch).value();
      words[start + 1] = end;
    } else {
      words[start] = build_s_getpc_b64(8, arch).value();
      words[start + 1] = build_s_add_u32(8, 8, 255, arch).value();
      words[start + 2] = 512 - entry - 4;
      words[start + 3] = build_s_addc_u32(9, 9, 128, arch).value();
      words[start + 4] = build_s_swappc_b64(30, 8, arch).value();
      words[start + 5] = end;
    }
  }
  const std::array<uint32_t, 3> atomic = atomic_test_words(fixture);
  words.insert(words.end(), atomic.begin(), atomic.end());
  words.push_back(build_s_setpc_b64(30, arch).value());
  std::vector<uint8_t> image = make_amdgpu_two_kernel_elf(words, 0, 3, fixture.machine,
                                                          {fixture.wave == 32, fixture.wave == 32});
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

inline std::optional<uint64_t> atomic_emitted_guest(const Section &text, uint64_t source_offset,
                                                    rj_code_target_id_t target) {
  if (source_offset > text.size() || text.size() - source_offset < 4 ||
      source_offset > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) - 4)
    return std::nullopt;
  uint32_t branch = 0;
  std::memcpy(&branch, text.data() + source_offset, sizeof(branch));
  std::unique_ptr<Decoder> decoder = Decoder::create(default_isa_target_registry(), target);
  if (!decoder)
    return std::nullopt;
  const DecodeResult decoded = decoder->decode_window(std::span(&branch, 1), source_offset);
  if (decoded.failed() || decoded.value()->mnemonic() != "s_branch")
    return std::nullopt;
  const std::optional<int64_t> delta = decoded.value()->branch_offset_bytes();
  const int64_t base = static_cast<int64_t>(source_offset + 4);
  if (!delta || (*delta > 0 && base > std::numeric_limits<int64_t>::max() - *delta))
    return std::nullopt;
  const int64_t destination = base + *delta;
  if (destination < 0 || static_cast<uint64_t>(destination) > text.size())
    return std::nullopt;
  return static_cast<uint64_t>(destination);
}

// Compute the allowed instruction delta independently of the production field
// helper or FI record, then verify the complete emitted image and guest mapping.
inline bool atomic_output_matches(const FaultMutationContext &source, uint64_t selected_offset,
                                  uint32_t byte_delta, std::span<const uint8_t> candidate) {
  const Section &original_text = *source.source().text_sections().front();
  if (selected_offset > original_text.size() || original_text.size() - selected_offset < 12)
    return false;
  const auto *original = reinterpret_cast<const uint8_t *>(original_text.data());
  std::array<uint32_t, 3> words = {};
  std::memcpy(words.data(), original + selected_offset, sizeof(words));
  const uint32_t encoded = words[2] >> 8;
  const int64_t old_offset =
      encoded >= 0x800000u ? static_cast<int64_t>(encoded) - 0x1000000 : encoded;
  const int64_t new_offset = old_offset + byte_delta;
  if (new_offset < -0x800000 || new_offset > 0x7fffff)
    return false;
  words[2] = (words[2] & 0xffu) | ((static_cast<uint32_t>(new_offset) & 0xffffffu) << 8);
  std::vector<uint8_t> expected_bytes(original + selected_offset, original + selected_offset + 12);
  std::vector<uint8_t> replacement(sizeof(words));
  std::memcpy(replacement.data(), words.data(), sizeof(words));
  const std::array<ExpectedReplacement, 1> edits = {ExpectedReplacement{
      .offset = selected_offset, .expected = expected_bytes, .replacement = replacement}};
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
      atomic_emitted_guest(text, selected_offset, source.source().target_id());
  if (!guest || *guest > text.size() || text.size() - *guest < 16)
    return false;
  std::unique_ptr<Decoder> decoder =
      Decoder::create(default_isa_target_registry(), source.source().target_id());
  if (!decoder || !branch_reaches(*decoder, text, selected_offset, *guest) ||
      !branch_reaches(*decoder, text, *guest + 12, selected_offset + 12))
    return false;
  std::array<uint32_t, 3> original_words = {};
  std::array<uint32_t, 3> guest_words = {};
  std::memcpy(original_words.data(), original + selected_offset, sizeof(original_words));
  std::memcpy(guest_words.data(), text.data() + *guest, sizeof(guest_words));
  const DecodeResult before = decoder->decode_window(original_words, selected_offset);
  const DecodeResult after = decoder->decode_window(guest_words, *guest);
  if (before.failed() || after.failed() || before.value()->size() != 12 ||
      after.value()->size() != 12 || before.value()->mnemonic() != after.value()->mnemonic() ||
      before.value()->opcode() != after.value()->opcode() ||
      before.value()->num_src_operands() != after.value()->num_src_operands() ||
      before.value()->num_dst_operands() != after.value()->num_dst_operands())
    return false;
  for (int index = 0; index < before.value()->num_operands(); ++index) {
    const Operand *old_operand =
        index < before.value()->num_src_operands()
            ? before.value()->src_operand(index)
            : before.value()->dst_operand(index - before.value()->num_src_operands());
    const Operand *new_operand =
        index < after.value()->num_src_operands()
            ? after.value()->src_operand(index)
            : after.value()->dst_operand(index - after.value()->num_src_operands());
    if (!old_operand || !new_operand ||
        old_operand->encoding_value() != new_operand->encoding_value() ||
        old_operand->size_bits() != new_operand->size_bits() ||
        old_operand->decoded_vgpr() != new_operand->decoded_vgpr())
      return false;
  }
  return std::memcmp(text.data() + *guest, replacement.data(), replacement.size()) == 0;
}

} // namespace rocjitsu::test
