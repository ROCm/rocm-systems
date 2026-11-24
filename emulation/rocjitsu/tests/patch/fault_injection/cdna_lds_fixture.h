// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

#include "lds_fixture.h"
#include "rocjitsu/isa/isa_family.h"

#include <array>
#include <cstring>
#include <limits>

namespace rocjitsu::test {

struct CdnaLdsTarget {
  const char *name;
  rj_code_target_id_t target;
  IsaFamily arch;
  uint32_t elf_machine;
};

inline constexpr std::array<CdnaLdsTarget, 2> kCdnaLdsTargets = {{
    {.name = "Gfx942",
     .target = ROCJITSU_CODE_TARGET_GFX942,
     .arch = IsaFamily::Cdna3,
     .elf_machine = EF_AMDGPU_MACH_AMDGCN_GFX942},
    {.name = "Gfx950",
     .target = ROCJITSU_CODE_TARGET_GFX950,
     .arch = IsaFamily::Cdna4,
     .elf_machine = EF_AMDGPU_MACH_AMDGCN_GFX950},
}};

inline std::vector<uint8_t> make_cdna_lds_elf(const CdnaLdsTarget &row, bool acc = false,
                                              uint32_t granulated = 3, uint32_t accum_offset = 3,
                                              bool gds = false, bool wave32 = false) {
  // llvm-mc witnesses: ds_write_b32 v2,v3 offset:4 and
  // ds_write_b128 v1,v[4:7] offset:16. ACC selects the data operands only.
  const uint32_t acc_bit = acc ? 0x02000000u : 0u;
  const uint32_t gds_bit = gds ? 0x00010000u : 0u;
  const std::vector<uint32_t> words = {0xd81a0004u | acc_bit | gds_bit, 0x00000302u,
                                       0xd9be0010u | acc_bit, 0x00000401u, 0xbf810000u};
  std::vector<uint8_t> image = make_amdgpu_kernel_elf(
      words, 0, 3, row.elf_machine, granulated, accum_offset, false, false, false, false, wave32);
  image[EI_DATA] = ELFDATA2LSB;
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  return image;
}

inline bool cdna_lds_output_matches(const FaultMutationContext &source, const CdnaLdsTarget &row,
                                    uint64_t selected_offset, uint32_t replacement_vgpr,
                                    std::span<const uint8_t> candidate) {
  const Section &original_text = *source.source().text_sections().front();
  if (selected_offset > original_text.size() || original_text.size() - selected_offset < 8)
    return false;
  const auto *original = reinterpret_cast<const uint8_t *>(original_text.data());
  std::vector<uint8_t> expected_bytes(original + selected_offset, original + selected_offset + 8);
  std::vector<uint8_t> replacement = expected_bytes;
  replacement[4] = static_cast<uint8_t>(replacement_vgpr);
  const std::array<ExpectedReplacement, 1> edits = {ExpectedReplacement{
      .offset = selected_offset, .expected = expected_bytes, .replacement = replacement}};
  const std::optional<std::vector<uint8_t>> expected =
      expected_replacement_image(source.source(), edits, row.arch);
  if (!expected || !std::ranges::equal(*expected, candidate))
    return false;
  const FaultMutationContext mutant(candidate);
  if (!mutant.source().is_valid() || mutant.source().target_id() != row.target ||
      mutant.source().text_sections().size() != 1 || !descriptors_preserved(source, mutant))
    return false;
  const Section &text = *mutant.source().text_sections().front();
  if (selected_offset > text.size() || text.size() - selected_offset < 4)
    return false;
  const auto *bytes = reinterpret_cast<const uint8_t *>(text.data());
  uint32_t branch = 0;
  std::memcpy(&branch, bytes + selected_offset, sizeof(branch));
  std::unique_ptr<Decoder> decoder = Decoder::create(default_isa_target_registry(), row.target);
  if (!decoder)
    return false;
  const DecodeResult jump = decoder->decode_window(std::span(&branch, 1), selected_offset);
  if (jump.failed() || jump.value()->mnemonic() != "s_branch")
    return false;
  const std::optional<int64_t> delta = jump.value()->branch_offset_bytes();
  if (!delta || selected_offset > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) - 4)
    return false;
  const int64_t base = static_cast<int64_t>(selected_offset + 4);
  if (*delta > 0 && base > std::numeric_limits<int64_t>::max() - *delta)
    return false;
  const int64_t signed_guest = base + *delta;
  if (signed_guest < 0)
    return false;
  const uint64_t guest = static_cast<uint64_t>(signed_guest);
  if (guest > text.size() || text.size() - guest < 12 ||
      !branch_reaches(*decoder, text, guest + 8, selected_offset + 8))
    return false;
  std::array<uint32_t, 2> words = {};
  std::memcpy(words.data(), bytes + guest, sizeof(words));
  const DecodeResult decoded = decoder->decode_window(words, guest);
  if (decoded.failed() || decoded.value()->size() != 8 ||
      (decoded.value()->mnemonic() != "ds_write_b32" &&
       decoded.value()->mnemonic() != "ds_write_b128") ||
      decoded.value()->num_src_operands() < 2 ||
      decoded.value()->src_operand(0)->encoding_value() != static_cast<int>(replacement_vgpr))
    return false;
  return std::memcmp(bytes + guest, replacement.data(), replacement.size()) == 0;
}

} // namespace rocjitsu::test
