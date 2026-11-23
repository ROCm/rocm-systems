// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "../../gfx1251_lds_fixture.h"
#include "../../rdna3_lds_fixture.h"
#include "lds_fixture.h"

#include <array>
#include <cstring>
#include <span>
#include <vector>

namespace rocjitsu::test {

struct TargetLdsCase {
  LdsFormCase form;
  std::array<uint32_t, 2> expected;
  uint32_t wave;
};

// llvm-mc witnesses: s_alloc_vgpr 64; v_mov_b32 v40, 0; s_alloc_vgpr 32.
inline constexpr std::array<uint32_t, 3> kLdsDynamicVgprPrefix = {0xbe8053c0u, 0x7e500280u,
                                                                  0xbe8053a0u};

inline void enable_target_lds_dynamic_vgprs(std::vector<uint8_t> &image, size_t owner = 0) {
  const AmdGpuCodeObject source(image.data(), image.size());
  ASSERT_TRUE(source.is_valid());
  ASSERT_EQ(source.text_sections().size(), 1u);
  const Section &text = *source.text_sections().front();
  const std::vector<KernelDescriptorInfo> kernels =
      scan_kernel_descriptors(image, text.sectionOffset(), text.size(), text.sectionHeaderIndex());
  ASSERT_LT(owner, kernels.size());
  amdhsa::kernel_descriptor_t descriptor = kernels[owner].descriptor;
  RJ_AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3,
                     amdhsa::COMPUTE_PGM_RSRC3_GFX125_ENABLE_DYNAMIC_VGPR, 1);
  std::memcpy(image.data() + kernels[owner].descriptor_file_offset, &descriptor,
              sizeof(descriptor));
}

inline std::vector<TargetLdsCase> target_lds_cases(rj_code_target_id_t target) {
  std::vector<TargetLdsCase> cases;
  if (target == ROCJITSU_CODE_TARGET_GFX1100) {
    for (const Rdna3LdsFixture &fixture : kRdna3LdsFixtures) {
      for (uint32_t wave : {32u, 64u}) {
        cases.push_back({.form = {.target = target,
                                  .machine = EF_AMDGPU_MACH_AMDGCN_GFX1100,
                                  .arch = IsaFamily::Rdna3,
                                  .form = fixture.form,
                                  .words = fixture.source},
                         .expected = fixture.expected,
                         .wave = wave});
      }
    }
  } else if (target == ROCJITSU_CODE_TARGET_GFX1251) {
    for (const Gfx1251LdsFixture &fixture : kGfx1251LdsFixtures) {
      cases.push_back({.form = {.target = target,
                                .machine = EF_AMDGPU_MACH_AMDGCN_GFX1251,
                                .arch = IsaFamily::Cdna5,
                                .form = fixture.form,
                                .words = fixture.source},
                       .expected = fixture.expected,
                       .wave = 32});
    }
  }
  return cases;
}

inline uint32_t target_lds_allocation(const TargetLdsCase &fixture, uint32_t granulated = 3) {
  const uint32_t granule = fixture.form.target == ROCJITSU_CODE_TARGET_GFX1251 ? 16u
                           : fixture.wave == 32                                ? 8u
                                                                               : 4u;
  return (granulated + 1) * granule;
}

inline std::vector<uint8_t> make_target_lds_elf(const TargetLdsCase &fixture,
                                                std::span<const uint32_t> prefix = {},
                                                std::span<const uint32_t> suffix = {},
                                                uint32_t granulated = 3) {
  std::vector<uint32_t> words(prefix.begin(), prefix.end());
  words.insert(words.end(), fixture.form.words.begin(), fixture.form.words.end());
  words.insert(words.end(), suffix.begin(), suffix.end());
  words.push_back(build_s_nop(7, fixture.form.arch).value());
  words.push_back(build_s_endpgm(fixture.form.arch).value());
  std::vector<uint8_t> image =
      make_amdgpu_kernel_elf(words, 0, 3, fixture.form.machine, granulated, 3, false, false, false,
                             false, fixture.wave == 32);
  image[EI_DATA] = ELFDATA2LSB;
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  return image;
}

inline std::vector<uint8_t> make_target_shared_lds_elf(const TargetLdsCase &fixture,
                                                       std::array<uint32_t, 2> granulated = {3, 3},
                                                       std::span<const uint32_t> first_prefix = {},
                                                       std::span<const uint32_t> second_prefix = {},
                                                       std::array<uint32_t, 2> waves = {}) {
  const IsaFamily arch = fixture.form.arch;
  std::vector<uint32_t> words(128, build_s_nop(0, arch).value());
  const std::array<std::span<const uint32_t>, 2> prefixes = {first_prefix, second_prefix};
  for (size_t owner = 0; owner < 2; ++owner) {
    size_t cursor = owner * 64;
    for (uint32_t word : prefixes[owner])
      words[cursor++] = word;
    const int16_t displacement = static_cast<int16_t>((512 - cursor * 4 - 4) / 4);
    words[cursor++] = build_s_call_b64(30, displacement, arch).value();
    words[cursor] = build_s_endpgm(arch).value();
  }
  words.insert(words.end(), fixture.form.words.begin(), fixture.form.words.end());
  words.push_back(build_s_setpc_b64(30, arch).value());
  std::vector<uint8_t> image = make_amdgpu_two_kernel_elf(words, 0, 3, fixture.form.machine);
  image[EI_DATA] = ELFDATA2LSB;
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  const AmdGpuCodeObject source(image.data(), image.size());
  const Section &text = *source.text_sections().front();
  const std::vector<KernelDescriptorInfo> descriptors =
      scan_kernel_descriptors(image, text.sectionOffset(), text.size(), text.sectionHeaderIndex());
  for (const KernelDescriptorInfo &kernel : descriptors) {
    const size_t owner = kernel.kernel_name == "test_kernel1" ? 1 : 0;
    amdhsa::kernel_descriptor_t descriptor = kernel.descriptor;
    descriptor.kernel_code_entry_byte_offset += static_cast<int64_t>(owner * 256);
    RJ_AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                       amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT, granulated[owner]);
    const uint32_t wave = waves[owner] == 0 ? fixture.wave : waves[owner];
    RJ_AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                       amdhsa::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32, wave == 32);
    std::memcpy(image.data() + kernel.descriptor_file_offset, &descriptor, sizeof(descriptor));
  }
  return image;
}

inline bool target_lds_output_matches(const FaultMutationContext &source,
                                      const TargetLdsCase &fixture, uint64_t offset, uint32_t reg,
                                      std::span<const uint8_t> output) {
  if (!lds_form_output_matches(source, fixture.form, offset, reg, output))
    return false;
  if (reg != 6)
    return true;
  // The v6 words came from separate assembler inputs, independent of the
  // replacement helper and the whole-image oracle's byte edit.
  const AmdGpuCodeObject mutant(output.data(), output.size());
  const Section &text = *mutant.text_sections().front();
  const std::optional<uint64_t> guest = lds_emitted_guest(text, offset, fixture.form.target);
  if (!guest || *guest > text.size() || text.size() - *guest < sizeof(fixture.expected))
    return false;
  return std::memcmp(text.data() + *guest, fixture.expected.data(), sizeof(fixture.expected)) == 0;
}

} // namespace rocjitsu::test
