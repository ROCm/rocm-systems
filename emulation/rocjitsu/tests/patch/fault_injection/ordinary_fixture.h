// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "../../dbi_test_util.h"
#include "rocjitsu/code/builders/instruction_builder.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace rocjitsu::test {

struct OrdinaryTargetCase {
  rj_code_target_id_t target;
  uint32_t machine;
  IsaFamily arch;
  uint32_t wave;
  const char *name;
};

inline constexpr std::array<OrdinaryTargetCase, 6> kOrdinaryTargetCases = {{
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

// Literal encodings come from the original ordinary acquire fixtures. Keeping
// cache scope zero also protects the accepted load-scope != cache-scope case.
inline std::array<uint32_t, 3> ordinary_load_words(uint32_t scope = 2, int32_t offset = 0) {
  return {0xee050004u, 7u | (scope << 18),
          ((static_cast<uint32_t>(offset) & 0xffffffu) << 8) | 10u};
}
inline constexpr std::array<uint32_t, 3> kOrdinaryAcquireCache = {0xee0ac000u, 0, 0};
inline constexpr uint32_t kOrdinaryWaitLoadZero = 0xbfc00000u;
inline constexpr uint32_t kOrdinaryDelay = 0xbf870001u;

inline bool add_ordinary_function_symbol(std::vector<uint8_t> &image, uint64_t offset,
                                         uint64_t size, std::string_view name) {
  Elf64_Ehdr header = {};
  std::memcpy(&header, image.data(), sizeof(header));
  for (size_t index = 0; index < header.e_shnum; ++index) {
    const size_t symbol_header_offset = header.e_shoff + index * sizeof(Elf64_Shdr);
    Elf64_Shdr symbols = {};
    std::memcpy(&symbols, image.data() + symbol_header_offset, sizeof(symbols));
    if (symbols.sh_type != SHT_SYMTAB)
      continue;
    const size_t string_header_offset = header.e_shoff + symbols.sh_link * sizeof(Elf64_Shdr);
    Elf64_Shdr strings = {};
    std::memcpy(&strings, image.data() + string_header_offset, sizeof(strings));
    std::vector<uint8_t> names(image.begin() + strings.sh_offset,
                               image.begin() + strings.sh_offset + strings.sh_size);
    const uint32_t name_offset = static_cast<uint32_t>(names.size());
    names.insert(names.end(), name.begin(), name.end());
    names.push_back(0);
    std::vector<uint8_t> table(image.begin() + symbols.sh_offset,
                               image.begin() + symbols.sh_offset + symbols.sh_size);
    const Elf64_Sym symbol = {.st_name = name_offset,
                              .st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeFunc),
                              .st_other = 0,
                              .st_shndx = 1,
                              .st_value = 0x1100 + offset,
                              .st_size = size};
    const size_t symbol_offset = table.size();
    table.resize(table.size() + sizeof(symbol));
    std::memcpy(table.data() + symbol_offset, &symbol, sizeof(symbol));
    strings.sh_offset = image.size();
    strings.sh_size = names.size();
    image.insert(image.end(), names.begin(), names.end());
    image.resize((image.size() + 7) & ~size_t{7});
    symbols.sh_offset = image.size();
    symbols.sh_size = table.size();
    image.insert(image.end(), table.begin(), table.end());
    std::memcpy(image.data() + symbol_header_offset, &symbols, sizeof(symbols));
    std::memcpy(image.data() + string_header_offset, &strings, sizeof(strings));
    return true;
  }
  return false;
}

// Only a test-fixture mutation: retain the actual FUNC symbol and alter its
// literal size. The production container query owns all extent inference.
inline bool zero_ordinary_function_size(std::vector<uint8_t> &image, uint64_t entry) {
  Elf64_Ehdr header = {};
  std::memcpy(&header, image.data(), sizeof(header));
  size_t changed = 0;
  for (size_t index = 0; index < header.e_shnum; ++index) {
    Elf64_Shdr symbols = {};
    std::memcpy(&symbols, image.data() + header.e_shoff + index * sizeof(Elf64_Shdr),
                sizeof(symbols));
    if (symbols.sh_type != SHT_SYMTAB || symbols.sh_entsize != sizeof(Elf64_Sym))
      continue;
    for (uint64_t offset = 0; offset + sizeof(Elf64_Sym) <= symbols.sh_size;
         offset += sizeof(Elf64_Sym)) {
      Elf64_Sym symbol = {};
      std::memcpy(&symbol, image.data() + symbols.sh_offset + offset, sizeof(symbol));
      if (elf_symbol_type(symbol.st_info) != kElfSymbolTypeFunc || symbol.st_shndx != 1 ||
          symbol.st_value != 0x1100 + entry)
        continue;
      symbol.st_size = 0;
      std::memcpy(image.data() + symbols.sh_offset + offset, &symbol, sizeof(symbol));
      ++changed;
    }
  }
  return changed == 1;
}

inline std::vector<uint8_t> make_ordinary_words_elf(const OrdinaryTargetCase &fixture,
                                                    std::vector<uint32_t> words) {
  words.push_back(build_s_endpgm(fixture.arch).value());
  std::vector<uint8_t> image = make_amdgpu_kernel_elf(words, 0, 3, fixture.machine, 3, 0, false,
                                                      false, false, false, fixture.wave == 32);
  image[EI_DATA] = ELFDATA2LSB;
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  if (!add_ordinary_function_symbol(image, 0, words.size() * sizeof(uint32_t), "test_kernel"))
    return {};
  return image;
}

inline std::vector<uint32_t> ordinary_acquire_words(uint32_t scope = 2, int32_t offset = 0,
                                                    bool delay = false, bool second_cache = false) {
  const std::array<uint32_t, 3> load = ordinary_load_words(scope, offset);
  std::vector<uint32_t> words(load.begin(), load.end());
  words.push_back(kOrdinaryWaitLoadZero);
  if (delay)
    words.push_back(kOrdinaryDelay);
  words.insert(words.end(), kOrdinaryAcquireCache.begin(), kOrdinaryAcquireCache.end());
  if (second_cache)
    words.insert(words.end(), kOrdinaryAcquireCache.begin(), kOrdinaryAcquireCache.end());
  return words;
}

inline std::vector<uint8_t> make_ordinary_elf(const OrdinaryTargetCase &fixture, uint32_t scope = 2,
                                              int32_t offset = 0, bool delay = false,
                                              bool second_cache = false) {
  return make_ordinary_words_elf(fixture,
                                 ordinary_acquire_words(scope, offset, delay, second_cache));
}

inline uint32_t ordinary_vccz_branch(int16_t displacement, IsaFamily arch) {
  const bool cdna = arch == IsaFamily::Cdna3 || arch == IsaFamily::Cdna4;
  return (cdna ? 0xbf860000u : 0xbfa30000u) | static_cast<uint16_t>(displacement);
}

inline std::vector<uint8_t> make_ordinary_fallthrough_elf(const OrdinaryTargetCase &fixture) {
  // The entry's skip edge creates another predecessor at the cache leader.
  // The load block itself reaches the cache through one exact fallthrough.
  std::vector<uint32_t> words{ordinary_vccz_branch(5, fixture.arch)};
  const std::array<uint32_t, 3> load = ordinary_load_words();
  words.insert(words.end(), load.begin(), load.end());
  words.push_back(kOrdinaryWaitLoadZero);
  words.push_back(kOrdinaryDelay);
  words.insert(words.end(), kOrdinaryAcquireCache.begin(), kOrdinaryAcquireCache.end());
  return make_ordinary_words_elf(fixture, std::move(words));
}

inline std::vector<uint8_t> make_ordinary_poll_elf(const OrdinaryTargetCase &fixture,
                                                   uint32_t prefix_words = 0,
                                                   bool bypass_to_exit = false) {
  const uint32_t nop = build_s_nop(0, fixture.arch).value();
  std::vector<uint32_t> words;
  if (bypass_to_exit)
    words.push_back(ordinary_vccz_branch(static_cast<int16_t>(prefix_words + 6), fixture.arch));
  words.insert(words.end(), prefix_words, nop);
  // Original make_rdna4_flag_self_loop_acquire_code_object, retaining the
  // compiler bookkeeping both before the branch and on the fallthrough exit.
  words.insert(words.end(),
               {0xee050006u, 0x00080001u, 0x00000000u, kOrdinaryWaitLoadZero, 0x7c9a0280u});
  const int16_t loop_delta = -static_cast<int16_t>(prefix_words + 6);
  words.push_back(0xbfa40000u | static_cast<uint16_t>(loop_delta));
  words.insert(words.end(), {0xbf118004u, 0xbead0080u, 0x980080c1u, 0xbf028016u, 0xee0ac07cu,
                             0x00080000u, 0x00000000u});
  return make_ordinary_words_elf(fixture, std::move(words));
}

inline std::vector<uint8_t> make_shared_ordinary_elf(const OrdinaryTargetCase &fixture,
                                                     bool second_owner = true,
                                                     bool second_owner_cache_only = false) {
  const uint32_t nop = build_s_nop(0, fixture.arch).value();
  const uint32_t end = build_s_endpgm(fixture.arch).value();
  std::vector<uint32_t> words(128, nop);
  for (uint32_t entry : {0u, 256u}) {
    const size_t start = entry / 4;
    if (entry == 256 && !second_owner) {
      words[start] = end;
      continue;
    }
    const uint32_t destination = entry == 256 && second_owner_cache_only ? 528 : 512;
    words[start] = build_s_call_b64(30, (destination - entry - 4) / 4, fixture.arch).value();
    words[start + 1] = end;
  }
  const std::vector<uint32_t> body = ordinary_acquire_words();
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
  for (const KernelDescriptorInfo &kernel : descriptors) {
    if (kernel.kernel_name != "test_kernel1")
      continue;
    amdhsa::kernel_descriptor_t descriptor = kernel.descriptor;
    descriptor.kernel_code_entry_byte_offset += 256;
    std::memcpy(image.data() + kernel.descriptor_file_offset, &descriptor, sizeof(descriptor));
  }
  if (!add_ordinary_function_symbol(image, 0, 8, "test_kernel0") ||
      !add_ordinary_function_symbol(image, 256, second_owner ? 8 : 4, "test_kernel1") ||
      !add_ordinary_function_symbol(image, 512, 32, "shared_acquire_helper"))
    return {};
  return image;
}

} // namespace rocjitsu::test
