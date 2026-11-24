// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

#include "lds_fixture.h"
#include "rocjitsu/code/amdgpu_elf.h"

namespace rocjitsu::test {

struct TwoAddressTarget {
  rj_code_target_id_t target;
  IsaFamily arch;
  uint32_t machine;
  bool cdna;
};
inline constexpr std::array<TwoAddressTarget, 4> kTwoAddressTargets = {{
    {.target = ROCJITSU_CODE_TARGET_GFX942,
     .arch = IsaFamily::Cdna3,
     .machine = EF_AMDGPU_MACH_AMDGCN_GFX942,
     .cdna = true},
    {.target = ROCJITSU_CODE_TARGET_GFX950,
     .arch = IsaFamily::Cdna4,
     .machine = EF_AMDGPU_MACH_AMDGCN_GFX950,
     .cdna = true},
    {.target = ROCJITSU_CODE_TARGET_GFX1200,
     .arch = IsaFamily::Rdna4,
     .machine = EF_AMDGPU_MACH_AMDGCN_GFX1200,
     .cdna = false},
    {.target = ROCJITSU_CODE_TARGET_GFX1201,
     .arch = IsaFamily::Rdna4,
     .machine = EF_AMDGPU_MACH_AMDGCN_GFX1201,
     .cdna = false},
}};

inline std::vector<uint8_t> make_two_address_elf(const TwoAddressTarget &target,
                                                 bool wave32 = false) {
  // Concrete llvm-mc witnesses with two nonzero offset fields and distinct data operands.
  const std::vector<uint32_t> words =
      target.cdna
          ? std::vector<uint32_t>{0xd8ee0301u, 0x04000002u, 0xd89e0502u, 0x00060402u, 0xbf810000u}
          : std::vector<uint32_t>{0xd9dc0301u, 0x04000002u, 0xd8380502u, 0x00040302u, 0xbfb00000u};
  std::vector<uint8_t> image =
      make_amdgpu_kernel_elf(words, 0, 3, target.machine, 3, 3, false, false, false, false, wave32);
  image[EI_DATA] = ELFDATA2LSB;
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  return image;
}

inline bool two_address_output_matches(const FaultMutationContext &source,
                                       const TwoAddressTarget &target, uint64_t offset,
                                       uint32_t vgpr, std::span<const uint8_t> output) {
  const Section &original = *source.source().text_sections().front();
  if (offset > original.size() || original.size() - offset < 8)
    return false;
  const auto *raw = reinterpret_cast<const uint8_t *>(original.data());
  std::vector<uint8_t> old_bytes(raw + offset, raw + offset + 8);
  std::vector<uint8_t> replacement = old_bytes;
  replacement[4] = static_cast<uint8_t>(vgpr);
  const std::array<ExpectedReplacement, 1> edits = {
      ExpectedReplacement{.offset = offset, .expected = old_bytes, .replacement = replacement}};
  const std::optional<std::vector<uint8_t>> expected =
      expected_replacement_image(source.source(), edits, target.arch);
  if (!expected || !std::ranges::equal(*expected, output))
    return false;
  const FaultMutationContext mutant(output);
  if (!mutant.source().is_valid() || mutant.source().target_id() != target.target ||
      mutant.source().text_sections().size() != 1 || !descriptors_preserved(source, mutant))
    return false;
  const Section &text = *mutant.source().text_sections().front();
  const std::optional<uint64_t> guest = lds_emitted_guest(text, offset, target.target);
  if (!guest || *guest > text.size() || text.size() - *guest < 12)
    return false;
  std::unique_ptr<Decoder> decoder = Decoder::create(default_isa_target_registry(), target.target);
  if (!decoder || !branch_reaches(*decoder, text, offset, *guest) ||
      !branch_reaches(*decoder, text, *guest + 8, offset + 8))
    return false;
  const auto *bytes = reinterpret_cast<const uint8_t *>(text.data());
  if (std::memcmp(bytes + *guest, replacement.data(), 8) != 0)
    return false;
  std::array<uint32_t, 2> words = {};
  std::memcpy(words.data(), bytes + *guest, 8);
  const DecodeResult decoded = decoder->decode_window(words, *guest);
  const std::string_view form = target.cdna
                                    ? (offset == 0 ? "ds_read2_b64" : "ds_write2st64_b64")
                                    : (offset == 0 ? "ds_load_2addr_b64" : "ds_store_2addr_b32");
  return decoded.succeeded() && decoded.value()->mnemonic() == form &&
         decoded.value()->src_operand(0) &&
         decoded.value()->src_operand(0)->encoding_value() == static_cast<int>(vgpr);
}
} // namespace rocjitsu::test
