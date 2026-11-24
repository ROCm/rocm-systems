// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <string_view>

#include "../../dbi_test_util.h"
#include "barrier_oracle.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/isa_family.h"

#include <array>
#include <cstring>
#include <vector>

namespace rocjitsu::test {

inline std::optional<uint64_t>
lds_emitted_guest(const Section &text, uint64_t source_offset,
                  rj_code_target_id_t target = ROCJITSU_CODE_TARGET_GFX1201) {
  if (source_offset > text.size() || text.size() - source_offset < 4) {
    return std::nullopt;
  }
  const auto *bytes = reinterpret_cast<const uint8_t *>(text.data());
  uint32_t branch = 0;
  std::memcpy(&branch, bytes + source_offset, sizeof(branch));
  std::unique_ptr<Decoder> decoder = Decoder::create(default_isa_target_registry(), target);
  if (!decoder) {
    return std::nullopt;
  }
  const DecodeResult decoded = decoder->decode_window(std::span(&branch, 1), source_offset);
  if (decoded.failed() || decoded.value()->mnemonic() != "s_branch") {
    return std::nullopt;
  }
  const std::optional<int64_t> delta = decoded.value()->branch_offset_bytes();
  if (!delta || source_offset > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) - 4) {
    return std::nullopt;
  }
  const int64_t base = static_cast<int64_t>(source_offset + 4);
  if (*delta > 0 && base > std::numeric_limits<int64_t>::max() - *delta) {
    return std::nullopt;
  }
  const int64_t branch_target = base + *delta;
  if (branch_target < 0 || static_cast<uint64_t>(branch_target) > text.size()) {
    return std::nullopt;
  }
  return static_cast<uint64_t>(branch_target);
}

inline std::vector<uint8_t> make_lds_elf(uint32_t granulated_vgprs = 1, bool wave32 = false,
                                         uint32_t target = EF_AMDGPU_MACH_AMDGCN_GFX1201,
                                         bool dual_address = false) {
  const std::array<uint32_t, 2> first =
      rdna4::build_vds(dual_address ? rdna4::kDsLoad2addrB32Vds : rdna4::kDsStoreB32Vds,
                       {.offset0 = 4, .addr = 2, .data0 = 3});
  const std::array<uint32_t, 2> second =
      rdna4::build_vds(rdna4::kDsStoreB32Vds, {.offset0 = 8, .addr = 1, .data0 = 3});
  const std::vector<uint32_t> words = {first[0], first[1], second[0], second[1], 0xbfb00000u};
  std::vector<uint8_t> image = make_amdgpu_kernel_elf(words, 0, 3, target, granulated_vgprs, 0,
                                                      false, false, false, false, wave32);
  image[EI_DATA] = ELFDATA2LSB;
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  return image;
}

inline std::vector<uint8_t> make_cdna5_lds_elf(std::vector<uint32_t> words,
                                               uint32_t granulated_vgprs = 0) {
  words.push_back(cdna5::build_sopp(cdna5::kSEndpgmSopp)[0]);
  std::vector<uint8_t> image =
      make_amdgpu_kernel_elf(words, 0, 3, EF_AMDGPU_MACH_AMDGCN_GFX1250, granulated_vgprs);
  image[EI_DATA] = ELFDATA2LSB;
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  return image;
}

inline std::vector<uint8_t> make_lds_load_elf(uint32_t target, bool wave32 = false,
                                              uint32_t granulated_vgprs = 1) {
  // Independent llvm-mc witnesses: load v5, v2 offset:4; store v1, v3 offset:8.
  const std::vector<uint32_t> words = {0xd8d80004u, 0x05000002u, 0xd8340008u, 0x00000301u,
                                       0xbfb00000u};
  std::vector<uint8_t> image = make_amdgpu_kernel_elf(words, 0, 3, target, granulated_vgprs, 0,
                                                      false, false, false, false, wave32);
  image[EI_DATA] = ELFDATA2LSB;
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  return image;
}

// Follow the emitted branch and decode the replacement rather than treating
// source offsets as output guest offsets. Also compare the independent complete
// expected image, descriptor state, continuation and unselected text.
inline bool lds_output_matches(const FaultMutationContext &source, uint64_t selected_offset,
                               uint32_t replacement_vgpr, std::span<const uint8_t> candidate,
                               bool load = false) {
  const Section &original_text = *source.source().text_sections().front();
  if (selected_offset > original_text.size() || original_text.size() - selected_offset < 8) {
    return false;
  }
  const auto *original = reinterpret_cast<const uint8_t *>(original_text.data());
  std::vector<uint8_t> expected_bytes(original + selected_offset, original + selected_offset + 8);
  std::vector<uint8_t> replacement = expected_bytes;
  replacement[4] = static_cast<uint8_t>(replacement_vgpr);
  const std::array<ExpectedReplacement, 1> edits = {ExpectedReplacement{
      .offset = selected_offset, .expected = expected_bytes, .replacement = replacement}};
  const IsaFamily arch = source.source().target_id() == ROCJITSU_CODE_TARGET_GFX1250
                             ? IsaFamily::Cdna5
                             : IsaFamily::Rdna4;
  const std::optional<std::vector<uint8_t>> expected =
      expected_replacement_image(source.source(), edits, arch);
  if (!expected || !std::ranges::equal(*expected, candidate)) {
    return false;
  }
  const FaultMutationContext mutant(candidate);
  if (!mutant.source().is_valid() || mutant.source().text_sections().size() != 1 ||
      !descriptors_preserved(source, mutant)) {
    return false;
  }
  const Section &text = *mutant.source().text_sections().front();
  const auto *bytes = reinterpret_cast<const uint8_t *>(text.data());
  const rj_code_target_id_t target = source.source().target_id();
  const std::optional<uint64_t> emitted_guest = lds_emitted_guest(text, selected_offset, target);
  if (!emitted_guest) {
    return false;
  }
  const uint64_t guest = *emitted_guest;
  if (guest > text.size() || text.size() - guest < 12) {
    return false;
  }
  std::unique_ptr<Decoder> decoder = Decoder::create(default_isa_target_registry(), target);
  if (!decoder || !branch_reaches(*decoder, text, selected_offset, guest) ||
      !branch_reaches(*decoder, text, guest + 8, selected_offset + 8)) {
    return false;
  }
  std::array<uint32_t, 2> words = {};
  std::memcpy(words.data(), bytes + guest, sizeof(words));
  const DecodeResult decoded = decoder->decode_window(words, guest);
  if (decoded.failed() ||
      (decoded.value()->mnemonic() != "ds_store_b32" &&
       decoded.value()->mnemonic() != "ds_load_b32") ||
      decoded.value()->size() != 8 || !decoded.value()->src_operand(0) ||
      decoded.value()->src_operand(0)->encoding_value() != static_cast<int>(replacement_vgpr)) {
    return false;
  }
  if (load && (decoded.value()->mnemonic() != "ds_load_b32" || !decoded.value()->dst_operand(0) ||
               decoded.value()->dst_operand(0)->encoding_value() != 5)) {
    return false;
  }
  return std::memcmp(bytes + guest, replacement.data(), replacement.size()) == 0;
}

struct LdsFormCase {
  rj_code_target_id_t target;
  uint32_t machine;
  IsaFamily arch;
  std::string_view form;
  std::array<uint32_t, 2> words;
};

inline std::vector<uint8_t> make_lds_form_elf(const LdsFormCase &fixture, bool banked = false,
                                              bool wave32 = false) {
  std::vector<uint32_t> words;
  if (banked) {
    words.push_back(cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 1})[0]);
  }
  words.insert(words.end(), fixture.words.begin(), fixture.words.end());
  words.push_back(fixture.arch == IsaFamily::Cdna3 || fixture.arch == IsaFamily::Cdna4
                      ? 0xbf810000u
                      : 0xbfb00000u);
  std::vector<uint8_t> image = make_amdgpu_kernel_elf(words, 0, 3, fixture.machine, banked ? 16 : 3,
                                                      3, false, false, false, false, wave32);
  image[EI_DATA] = ELFDATA2LSB;
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  return image;
}
inline bool lds_form_output_matches(const FaultMutationContext &source, const LdsFormCase &fixture,
                                    uint64_t offset, uint32_t reg,
                                    std::span<const uint8_t> output) {
  const Section &original = *source.source().text_sections().front();
  const auto *raw = reinterpret_cast<const uint8_t *>(original.data());
  if (offset > original.size() || original.size() - offset < 8) {
    return false;
  }
  std::vector<uint8_t> old(raw + offset, raw + offset + 8), replacement = old;
  replacement[4] = static_cast<uint8_t>(reg);
  const std::array<ExpectedReplacement, 1> edits = {
      ExpectedReplacement{.offset = offset, .expected = old, .replacement = replacement}};
  const std::optional<std::vector<uint8_t>> expected =
      expected_replacement_image(source.source(), edits, fixture.arch);
  if (!expected || !std::ranges::equal(*expected, output)) {
    return false;
  }
  const FaultMutationContext mutant(output);
  if (!mutant.source().is_valid() || mutant.source().target_id() != fixture.target ||
      mutant.source().text_sections().size() != 1 || !descriptors_preserved(source, mutant)) {
    return false;
  }
  const Section &text = *mutant.source().text_sections().front();
  const std::optional<uint64_t> guest = lds_emitted_guest(text, offset, fixture.target);
  if (!guest || *guest > text.size() || text.size() - *guest < 12) {
    return false;
  }
  std::unique_ptr<Decoder> decoder = Decoder::create(default_isa_target_registry(), fixture.target);
  if (!decoder || !branch_reaches(*decoder, text, offset, *guest) ||
      !branch_reaches(*decoder, text, *guest + 8, offset + 8)) {
    return false;
  }
  const auto *bytes = reinterpret_cast<const uint8_t *>(text.data());
  if (std::memcmp(bytes + *guest, replacement.data(), 8) != 0) {
    return false;
  }
  std::array<uint32_t, 2> actual = {};
  std::memcpy(actual.data(), bytes + *guest, 8);
  const DecodeResult before = decoder->decode_window(fixture.words, offset);
  const DecodeResult after = decoder->decode_window(actual, *guest);
  if (before.failed() || after.failed() || before.value()->mnemonic() != fixture.form ||
      after.value()->mnemonic() != fixture.form || !after.value()->src_operand(0) ||
      after.value()->src_operand(0)->encoding_value() != static_cast<int>(reg) ||
      before.value()->num_src_operands() != after.value()->num_src_operands() ||
      before.value()->num_dst_operands() != after.value()->num_dst_operands()) {
    return false;
  }
  for (int i = 1; i < before.value()->num_src_operands(); ++i) {
    if (before.value()->src_operand(i)->encoding_value() !=
        after.value()->src_operand(i)->encoding_value()) {
      return false;
    }
  }
  for (int i = 0; i < before.value()->num_dst_operands(); ++i) {
    if (before.value()->dst_operand(i)->encoding_value() !=
        after.value()->dst_operand(i)->encoding_value()) {
      return false;
    }
  }
  return true;
}
} // namespace rocjitsu::test
