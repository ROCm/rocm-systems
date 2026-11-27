// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "atomic_cdna_exchange_fixture.h"
#include "atomic_fixture.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/opcodes.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace rocjitsu::test {

inline constexpr std::array<AtomicTargetCase, 10> kAtomicDsTargetCases = {{
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
    {.target = ROCJITSU_CODE_TARGET_GFX950,
     .machine = EF_AMDGPU_MACH_AMDGCN_GFX950,
     .arch = IsaFamily::Cdna4,
     .wave = 64,
     .name = "Gfx950Wave64"},
    {.target = ROCJITSU_CODE_TARGET_GFX942,
     .machine = EF_AMDGPU_MACH_AMDGCN_GFX942,
     .arch = IsaFamily::Cdna3,
     .wave = 64,
     .name = "Gfx942Wave64"},
    {.target = ROCJITSU_CODE_TARGET_GFX1100,
     .machine = EF_AMDGPU_MACH_AMDGCN_GFX1100,
     .arch = IsaFamily::Rdna3,
     .wave = 32,
     .name = "Gfx1100Wave32"},
    {.target = ROCJITSU_CODE_TARGET_GFX1100,
     .machine = EF_AMDGPU_MACH_AMDGCN_GFX1100,
     .arch = IsaFamily::Rdna3,
     .wave = 64,
     .name = "Gfx1100Wave64"},
}};

// The native builders place these shared operation numbers in each family's own encoding.
enum class AtomicDsForm : uint16_t {
  Add = 0,
  CompareStore = 16,
  AddReturning = 32,
  ExchangeReturning = 45,
  Exchange2 = 46,
  Exchange2Stride64 = 47,
  CompareStoreReturning = 48,
  WideAdd = 64,
};

inline std::string_view atomic_ds_form(const AtomicTargetCase &fixture, AtomicDsForm form) {
  const bool cdna = fixture.arch == IsaFamily::Cdna3 || fixture.arch == IsaFamily::Cdna4;
  switch (form) {
  case AtomicDsForm::Add:
    return "ds_add_u32";
  case AtomicDsForm::AddReturning:
    return "ds_add_rtn_u32";
  case AtomicDsForm::CompareStore:
    return cdna ? "ds_cmpst_b32" : "ds_cmpstore_b32";
  case AtomicDsForm::CompareStoreReturning:
    return cdna ? "ds_cmpst_rtn_b32" : "ds_cmpstore_rtn_b32";
  case AtomicDsForm::ExchangeReturning:
    return cdna ? "ds_wrxchg_rtn_b32" : "ds_storexchg_rtn_b32";
  case AtomicDsForm::Exchange2:
    return cdna ? "ds_wrxchg2_rtn_b32" : "ds_storexchg_2addr_rtn_b32";
  case AtomicDsForm::Exchange2Stride64:
    return cdna ? "ds_wrxchg2st64_rtn_b32" : "ds_storexchg_2addr_stride64_rtn_b32";
  case AtomicDsForm::WideAdd:
    return "ds_add_u64";
  }
  return {};
}

inline std::array<uint32_t, 2> atomic_ds_words(const AtomicTargetCase &fixture,
                                               AtomicDsForm form = AtomicDsForm::Add,
                                               uint8_t offset0 = 0, uint8_t offset1 = 1,
                                               bool modifiers = false) {
  const uint16_t opcode = static_cast<uint16_t>(form);
  const uint8_t flag = static_cast<uint8_t>(modifiers);
  switch (fixture.arch) {
  case IsaFamily::Cdna3:
    return cdna3::build_ds(opcode, {.offset0 = offset0,
                                    .offset1 = offset1,
                                    .gds = flag,
                                    .acc = flag,
                                    .addr = 2,
                                    .data0 = 4,
                                    .data1 = 6,
                                    .vdst = 8});
  case IsaFamily::Cdna4:
    return cdna4::build_ds(opcode, {.offset0 = offset0,
                                    .offset1 = offset1,
                                    .gds = flag,
                                    .acc = flag,
                                    .addr = 2,
                                    .data0 = 4,
                                    .data1 = 6,
                                    .vdst = 8});
  case IsaFamily::Rdna3:
    return rdna3::build_ds(opcode, {.offset0 = offset0,
                                    .offset1 = offset1,
                                    .gds = flag,
                                    .addr = 2,
                                    .data0 = 4,
                                    .data1 = 6,
                                    .vdst = 8});
  case IsaFamily::Cdna5:
    return cdna5::build_vds(
        opcode,
        {.offset0 = offset0, .offset1 = offset1, .addr = 2, .data0 = 4, .data1 = 6, .vdst = 8});
  default:
    return rdna4::build_vds(
        opcode,
        {.offset0 = offset0, .offset1 = offset1, .addr = 2, .data0 = 4, .data1 = 6, .vdst = 8});
  }
}

inline uint32_t atomic_ds_wait(const AtomicTargetCase &fixture) {
  switch (fixture.arch) {
  case IsaFamily::Cdna3:
    return cdna3::build_sopp(cdna3::kSWaitcntSopp)[0];
  case IsaFamily::Cdna4:
    return cdna4::build_sopp(cdna4::kSWaitcntSopp)[0];
  case IsaFamily::Rdna3:
    return rdna3::build_sopp(rdna3::kSWaitcntSopp)[0];
  case IsaFamily::Cdna5:
    return cdna5::build_sopp(cdna5::kSWaitDscntSopp)[0];
  default:
    return rdna4::build_sopp(rdna4::kSWaitDscntSopp)[0];
  }
}

inline std::vector<uint8_t> make_atomic_ds_sequence_elf(const AtomicTargetCase &fixture,
                                                        std::vector<uint32_t> words) {
  words.push_back(atomic_ds_wait(fixture));
  words.push_back(build_s_endpgm(fixture.arch).value());
  const bool cdna = fixture.arch == IsaFamily::Cdna3 || fixture.arch == IsaFamily::Cdna4;
  std::vector<uint8_t> image =
      make_amdgpu_kernel_elf(words, 0, 3, fixture.machine, cdna ? 15 : 3, cdna ? 7 : 0, false,
                             false, false, false, fixture.wave == 32);
  image[EI_DATA] = ELFDATA2LSB;
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  return image;
}

inline std::vector<uint8_t> make_atomic_ds_elf(const AtomicTargetCase &fixture,
                                               AtomicDsForm form = AtomicDsForm::Add,
                                               uint8_t offset0 = 0, uint8_t offset1 = 1,
                                               bool modifiers = false) {
  const std::array<uint32_t, 2> words = atomic_ds_words(fixture, form, offset0, offset1, modifiers);
  return make_atomic_ds_sequence_elf(fixture, {words[0], words[1]});
}

inline std::vector<uint8_t> make_mixed_native_cdna_exchange_elf(const AtomicTargetCase &fixture,
                                                                bool modifiers = false) {
  const std::array<std::array<uint32_t, 2>, 5> instructions = {
      native_cdna_exchange_words(fixture, NativeCdnaExchangeForm::ExchangeB64, modifiers),
      native_cdna_exchange_words(fixture, NativeCdnaExchangeForm::Exchange2Stride64B32, modifiers),
      native_cdna_exchange_words(fixture, NativeCdnaExchangeForm::ExchangeB32, modifiers),
      native_cdna_exchange_words(fixture, NativeCdnaExchangeForm::Exchange2B32, modifiers),
      atomic_ds_words(fixture, AtomicDsForm::AddReturning, 4, 5, modifiers)};
  std::vector<uint32_t> words;
  for (const std::array<uint32_t, 2> &instruction : instructions)
    words.insert(words.end(), instruction.begin(), instruction.end());
  return make_atomic_ds_sequence_elf(fixture, std::move(words));
}

// Reproduce the original offset0-only mutation independently of the production
// field helper, including the intentional lack of carry into offset1.
inline bool atomic_ds_output_matches(const FaultMutationContext &source, uint64_t selected_offset,
                                     uint32_t delta, std::span<const uint8_t> candidate) {
  const Section &original_text = *source.source().text_sections().front();
  if (selected_offset > original_text.size() || original_text.size() - selected_offset < 8)
    return false;
  const auto *original = reinterpret_cast<const uint8_t *>(original_text.data());
  std::array<uint32_t, 2> original_words = {};
  std::memcpy(original_words.data(), original + selected_offset, sizeof(original_words));
  const uint64_t next = static_cast<uint64_t>(original_words[0] & 0xffu) + delta;
  if (next > 255 || next % 4 != 0)
    return false;
  std::array<uint32_t, 2> expected_words = original_words;
  expected_words[0] = (original_words[0] & ~0xffu) | static_cast<uint32_t>(next);
  std::vector<uint8_t> expected_source(original + selected_offset, original + selected_offset + 8);
  std::vector<uint8_t> replacement(sizeof(expected_words));
  std::memcpy(replacement.data(), expected_words.data(), replacement.size());
  const std::array<ExpectedReplacement, 1> edits = {ExpectedReplacement{
      .offset = selected_offset, .expected = expected_source, .replacement = replacement}};
  const IsaTargetDescriptor *binding =
      default_isa_target_registry().find(source.source().target_id());
  if (!binding)
    return false;
  const std::optional<std::vector<uint8_t>> expected =
      expected_replacement_image(source.source(), edits, binding->family);
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
  if (!guest || *guest > text.size() || text.size() - *guest < 12)
    return false;
  std::unique_ptr<Decoder> decoder =
      Decoder::create(default_isa_target_registry(), source.source().target_id());
  if (!decoder || !branch_reaches(*decoder, text, selected_offset, *guest) ||
      !branch_reaches(*decoder, text, *guest + 8, selected_offset + 8))
    return false;
  std::array<uint32_t, 2> guest_words = {};
  std::memcpy(guest_words.data(), text.data() + *guest, sizeof(guest_words));
  const DecodeResult before = decoder->decode_window(original_words, selected_offset);
  const DecodeResult after = decoder->decode_window(guest_words, *guest);
  return before.succeeded() && after.succeeded() && before.value()->size() == 8 &&
         after.value()->size() == 8 && before.value()->mnemonic() == after.value()->mnemonic() &&
         before.value()->opcode() == after.value()->opcode() && guest_words == expected_words;
}

} // namespace rocjitsu::test
