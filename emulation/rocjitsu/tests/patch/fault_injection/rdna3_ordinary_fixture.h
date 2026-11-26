// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "legacy_ordinary_fixture.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace rocjitsu::test {

inline constexpr std::array<OrdinaryTargetCase, 2> kRdna3OrdinaryCases = {{
    {ROCJITSU_CODE_TARGET_GFX1100, EF_AMDGPU_MACH_AMDGCN_GFX1100, IsaFamily::Rdna3, 32,
     "Gfx1100Wave32"},
    {ROCJITSU_CODE_TARGET_GFX1100, EF_AMDGPU_MACH_AMDGCN_GFX1100, IsaFamily::Rdna3, 64,
     "Gfx1100Wave64"},
}};
inline constexpr uint32_t kRdna3OrdinaryWait = 0xbf8903f7u;
inline constexpr std::array<uint32_t, 4> kRdna3OrdinaryCache = {0xe0b00000u, 0, 0xe0ac0000u, 0};

// Original analysis_test.cpp:4517 global_load_b32 v2, v[0:1], off
// offset:4 glc; vmcnt(0); buffer_gl1_inv; buffer_gl0_inv.
// GLOBAL keeps scratch-only SVE clear; the modifier variant retains ignored VDATA five.
inline std::array<uint32_t, 2> rdna3_ordinary_load(int32_t offset = 4, bool modifiers = false) {
  const uint32_t flags = modifiers ? (1u << 13) | (1u << 15) : 0;
  return {0xdc524000u | flags | (static_cast<uint32_t>(offset) & 0x1fffu),
          modifiers ? 0x07040503u : 0x027c0000u};
}

inline std::vector<uint32_t> rdna3_ordinary_words(int32_t offset = 4, bool modifiers = false,
                                                  size_t pair_waits = 0) {
  const std::array<uint32_t, 2> load = rdna3_ordinary_load(offset, modifiers);
  std::vector<uint32_t> words{load[0], load[1], kRdna3OrdinaryWait, kRdna3OrdinaryCache[0],
                              kRdna3OrdinaryCache[1]};
  words.insert(words.end(), pair_waits, kRdna3OrdinaryWait);
  words.push_back(kRdna3OrdinaryCache[2]);
  words.push_back(kRdna3OrdinaryCache[3]);
  return words;
}

inline std::vector<uint8_t> make_rdna3_ordinary_elf(const OrdinaryTargetCase &fixture,
                                                    int32_t offset = 4, bool modifiers = false,
                                                    size_t pair_waits = 0) {
  return make_ordinary_words_elf(fixture, rdna3_ordinary_words(offset, modifiers, pair_waits));
}

} // namespace rocjitsu::test
