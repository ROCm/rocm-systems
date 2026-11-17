// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "ordinary_fixture.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <span>
#include <string_view>
#include <vector>

namespace rocjitsu::test {

inline constexpr std::array<OrdinaryTargetCase, 2> kCdna34OrdinaryCases = {{
    {ROCJITSU_CODE_TARGET_GFX950, EF_AMDGPU_MACH_AMDGCN_GFX950, IsaFamily::Cdna4, 64,
     "Gfx950Wave64"},
    {ROCJITSU_CODE_TARGET_GFX942, EF_AMDGPU_MACH_AMDGCN_GFX942, IsaFamily::Cdna3, 64,
     "Gfx942Wave64"},
}};
inline constexpr uint32_t kCdna34OrdinaryWait = 0xbf8c0f70u;
inline constexpr std::array<uint32_t, 2> kCdna34OrdinaryCache = {0xe0a48000u, 0};

// Original CDNA4 acquire witness: DE508004/027F0000; BF8C0F70;
// E0A48000/0. SC0/SC1 select native load scope; NT is independent.
inline std::array<uint32_t, 2> cdna34_ordinary_load(int32_t offset = 4, bool modifiers = false) {
  // GLOBAL preserves SC0/NT modifiers; scratch-only SVE (bit13) stays clear.
  const uint32_t flags = modifiers ? (1u << 16) | (1u << 17) : 0;
  return {0xde508000u | flags | (static_cast<uint32_t>(offset) & 0x1fffu),
          modifiers ? 0x07040503u : 0x027f0000u};
}

inline std::vector<uint32_t> cdna34_ordinary_words(int32_t offset = 4, bool modifiers = false) {
  const std::array<uint32_t, 2> load = cdna34_ordinary_load(offset, modifiers);
  return {load[0], load[1], kCdna34OrdinaryWait, kCdna34OrdinaryCache[0], kCdna34OrdinaryCache[1]};
}

inline std::vector<uint8_t> make_cdna34_ordinary_elf(const OrdinaryTargetCase &fixture,
                                                     int32_t offset = 4, bool modifiers = false) {
  return make_ordinary_words_elf(fixture, cdna34_ordinary_words(offset, modifiers));
}

inline bool mark_legacy_ordinary_wave32(std::vector<uint8_t> &image, std::string_view owner) {
  const AmdGpuCodeObject source(image.data(), image.size());
  const Section &text = *source.text_sections().front();
  const std::vector<KernelDescriptorInfo> descriptors =
      scan_kernel_descriptors(image, text.sectionOffset(), text.size(), text.sectionHeaderIndex());
  for (const KernelDescriptorInfo &kernel : descriptors) {
    if (kernel.kernel_name != owner)
      continue;
    amdhsa::kernel_descriptor_t descriptor = kernel.descriptor;
    RJ_AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                       amdhsa::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32, 1);
    std::memcpy(image.data() + kernel.descriptor_file_offset, &descriptor, sizeof(descriptor));
    return true;
  }
  return false;
}

inline std::vector<uint8_t> make_legacy_ordinary_shared_elf(const OrdinaryTargetCase &fixture,
                                                            std::span<const uint32_t> body,
                                                            uint32_t cache_relative_offset,
                                                            bool second_owner = true,
                                                            bool second_owner_cache_only = false) {
  const uint32_t end = build_s_endpgm(fixture.arch).value();
  std::vector<uint32_t> words(128, build_s_nop(0, fixture.arch).value());
  for (uint32_t entry : {0u, 256u}) {
    if (entry == 256 && !second_owner) {
      words[entry / 4] = end;
      continue;
    }
    const uint32_t destination =
        512 + (entry == 256 && second_owner_cache_only ? cache_relative_offset : 0);
    words[entry / 4] = build_s_call_b64(30, (destination - entry - 4) / 4, fixture.arch).value();
    words[entry / 4 + 1] = end;
  }
  words.insert(words.end(), body.begin(), body.end());
  words.push_back(build_s_setpc_b64(30, fixture.arch).value());
  std::vector<uint8_t> image = make_amdgpu_two_kernel_elf(words, 0, 3, fixture.machine,
                                                          {fixture.wave == 32, fixture.wave == 32});
  image[EI_DATA] = ELFDATA2LSB;
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  const AmdGpuCodeObject source(image.data(), image.size());
  const Section &text = *source.text_sections().front();
  const std::vector<KernelDescriptorInfo> descriptors =
      scan_kernel_descriptors(image, text.sectionOffset(), text.size(), text.sectionHeaderIndex());
  for (const KernelDescriptorInfo &kernel : descriptors)
    if (kernel.kernel_name == "test_kernel1") {
      amdhsa::kernel_descriptor_t descriptor = kernel.descriptor;
      descriptor.kernel_code_entry_byte_offset += 256;
      std::memcpy(image.data() + kernel.descriptor_file_offset, &descriptor, sizeof(descriptor));
    }
  if (!add_ordinary_function_symbol(image, 0, 8, "test_kernel0") ||
      !add_ordinary_function_symbol(image, 256, second_owner ? 8 : 4, "test_kernel1") ||
      !add_ordinary_function_symbol(image, 512, body.size_bytes() + 4, "legacy_acquire_helper"))
    return {};
  return image;
}

inline std::vector<uint8_t>
make_legacy_ordinary_split_container_elf(const OrdinaryTargetCase &fixture,
                                         std::vector<uint32_t> body, uint64_t split) {
  if (split == 0 || split % sizeof(uint32_t) != 0 || split >= body.size() * sizeof(uint32_t))
    return {};
  // A FUNC symbol does not create a CFG block leader. Preserve a real closed
  // kernel flow while making the second container's entry a branch target.
  body.insert(body.begin(), ordinary_vccz_branch(static_cast<int16_t>(split / 4), fixture.arch));
  split += sizeof(uint32_t);
  body.push_back(build_s_endpgm(fixture.arch).value());
  std::vector<uint8_t> image = make_amdgpu_kernel_elf(body, 0, 3, fixture.machine, 3, 0, false,
                                                      false, false, false, fixture.wave == 32);
  image[EI_DATA] = ELFDATA2LSB;
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  if (!add_ordinary_function_symbol(image, 0, split, "test_kernel") ||
      !add_ordinary_function_symbol(image, split, body.size() * 4 - split, "cache_helper"))
    return {};
  return image;
}

} // namespace rocjitsu::test
