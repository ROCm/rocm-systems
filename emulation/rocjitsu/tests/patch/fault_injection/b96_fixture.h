// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

#include "lds_fixture.h"
#include "rocjitsu/code/amdgpu_elf.h"

namespace rocjitsu::test {
struct B96Target {
  const char *name;
  rj_code_target_id_t target;
  IsaFamily arch;
  uint32_t machine;
  bool cdna;
};
inline constexpr std::array<B96Target, 5> kB96Targets = {{
    {.name = "Gfx1200",
     .target = ROCJITSU_CODE_TARGET_GFX1200,
     .arch = IsaFamily::Rdna4,
     .machine = EF_AMDGPU_MACH_AMDGCN_GFX1200,
     .cdna = false},
    {.name = "Gfx1201",
     .target = ROCJITSU_CODE_TARGET_GFX1201,
     .arch = IsaFamily::Rdna4,
     .machine = EF_AMDGPU_MACH_AMDGCN_GFX1201,
     .cdna = false},
    {.name = "Gfx942",
     .target = ROCJITSU_CODE_TARGET_GFX942,
     .arch = IsaFamily::Cdna3,
     .machine = EF_AMDGPU_MACH_AMDGCN_GFX942,
     .cdna = true},
    {.name = "Gfx950",
     .target = ROCJITSU_CODE_TARGET_GFX950,
     .arch = IsaFamily::Cdna4,
     .machine = EF_AMDGPU_MACH_AMDGCN_GFX950,
     .cdna = true},
    {.name = "Gfx1250",
     .target = ROCJITSU_CODE_TARGET_GFX1250,
     .arch = IsaFamily::Cdna5,
     .machine = EF_AMDGPU_MACH_AMDGCN_GFX1250,
     .cdna = false},
}};
inline std::vector<uint8_t> make_b96_elf(const B96Target &target, uint32_t granulated = 3,
                                         uint32_t accum_offset = 3, bool gds = false,
                                         bool acc = false, std::span<const uint32_t> prefix = {},
                                         bool wave32 = false) {
  const uint32_t modifier = (gds ? 0x00010000u : 0u) | (acc ? 0x02000000u : 0u);
  const uint32_t load = (target.cdna ? 0xd9fc0004u : 0xdbf80004u) | modifier;
  const uint32_t store = (target.cdna ? 0xd9bc0004u : 0xdb780004u) | modifier;
  std::vector<uint32_t> words(prefix.begin(), prefix.end());
  words.insert(words.end(), {load, 0x04000002u, store, 0x00000402u, load, 0x02000002u,
                             target.cdna ? 0xbf810000u : 0xbfb00000u});
  std::vector<uint8_t> image = make_amdgpu_kernel_elf(
      words, 0, 3, target.machine, granulated, accum_offset, false, false, false, false, wave32);
  image[EI_DATA] = ELFDATA2LSB;
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  return image;
}
inline bool b96_output_matches(const FaultMutationContext &source, const B96Target &target,
                               uint64_t offset, uint32_t vgpr, std::span<const uint8_t> output) {
  const Section &original = *source.source().text_sections().front();
  if (offset > original.size() || original.size() - offset < 8) {
    return false;
  }
  const auto *raw = reinterpret_cast<const uint8_t *>(original.data());
  std::vector<uint8_t> old_bytes(raw + offset, raw + offset + 8);
  std::vector<uint8_t> replacement = old_bytes;
  replacement[4] = static_cast<uint8_t>(vgpr);
  const std::array<ExpectedReplacement, 1> edits = {
      ExpectedReplacement{.offset = offset, .expected = old_bytes, .replacement = replacement}};
  const std::optional<std::vector<uint8_t>> expected =
      expected_replacement_image(source.source(), edits, target.arch);
  if (!expected || !std::ranges::equal(*expected, output)) {
    return false;
  }
  const FaultMutationContext mutant(output);
  if (!mutant.source().is_valid() || mutant.source().target_id() != target.target ||
      mutant.source().text_sections().size() != 1 || !descriptors_preserved(source, mutant)) {
    return false;
  }
  const Section &text = *mutant.source().text_sections().front();
  const std::optional<uint64_t> guest = lds_emitted_guest(text, offset, target.target);
  if (!guest || *guest > text.size() || text.size() - *guest < 12) {
    return false;
  }
  std::unique_ptr<Decoder> decoder = Decoder::create(default_isa_target_registry(), target.target);
  if (!decoder || !branch_reaches(*decoder, text, offset, *guest) ||
      !branch_reaches(*decoder, text, *guest + 8, offset + 8)) {
    return false;
  }
  const auto *bytes = reinterpret_cast<const uint8_t *>(text.data());
  if (std::memcmp(bytes + *guest, replacement.data(), 8) != 0) {
    return false;
  }
  std::array<uint32_t, 2> original_words = {}, guest_words = {};
  std::memcpy(original_words.data(), raw + offset, 8);
  std::memcpy(guest_words.data(), bytes + *guest, 8);
  const DecodeResult before = decoder->decode_window(original_words, offset);
  const DecodeResult after = decoder->decode_window(guest_words, *guest);
  if (before.failed() || after.failed() ||
      before.value()->mnemonic() != after.value()->mnemonic() || after.value()->size() != 8 ||
      !after.value()->src_operand(0) ||
      after.value()->src_operand(0)->encoding_value() != static_cast<int>(vgpr)) {
    return false;
  }
  const bool load = before.value()->mnemonic() == (target.cdna ? "ds_read_b96" : "ds_load_b96");
  const bool store = before.value()->mnemonic() == (target.cdna ? "ds_write_b96" : "ds_store_b96");
  if (!load && !store) {
    return false;
  }
  const Operand *old_tuple = load ? before.value()->dst_operand(0) : before.value()->src_operand(1);
  const Operand *new_tuple = load ? after.value()->dst_operand(0) : after.value()->src_operand(1);
  if (!old_tuple || !new_tuple || !old_tuple->decoded_vgpr() || !new_tuple->decoded_vgpr()) {
    return false;
  }
  return old_tuple->decoded_vgpr()->width == 3 && new_tuple->decoded_vgpr()->width == 3 &&
         old_tuple->decoded_vgpr()->index == new_tuple->decoded_vgpr()->index;
}
} // namespace rocjitsu::test
