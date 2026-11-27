// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "atomic_fixture.h"

#include <array>
#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

namespace rocjitsu::test {

inline constexpr std::array<AtomicTargetCase, 10> kNativeAtomicInventoryCases = {{
    kAtomicTargetCases[0],
    kAtomicTargetCases[1],
    kAtomicTargetCases[2],
    kAtomicTargetCases[3],
    kAtomicTargetCases[4],
    kAtomicTargetCases[5],
    {.target = ROCJITSU_CODE_TARGET_GFX942,
     .machine = EF_AMDGPU_MACH_AMDGCN_GFX942,
     .arch = IsaFamily::Cdna3,
     .wave = 64,
     .name = "Gfx942Wave64"},
    {.target = ROCJITSU_CODE_TARGET_GFX950,
     .machine = EF_AMDGPU_MACH_AMDGCN_GFX950,
     .arch = IsaFamily::Cdna4,
     .wave = 64,
     .name = "Gfx950Wave64"},
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

struct NativeAtomicInventorySample {
  std::vector<uint32_t> words;
  std::string_view form;
  amdgpu::MemoryKind memory_kind = amdgpu::MemoryKind::Atomic;
};

inline std::vector<NativeAtomicInventorySample>
native_atomic_inventory_samples(const AtomicTargetCase &target) {
  if (target.arch == IsaFamily::Cdna3 || target.arch == IsaFamily::Cdna4)
    return {{.words = {0xd97a0000u, 0u}, .form = "ds_consume"},
            {.words = {0xd97c0000u, 0u}, .form = "ds_append"},
            {.words = {0xd8680000u, 0u}, .form = "ds_wrap_rtn_b32"},
            {.words = {0xc1080000u, 0u}, .form = "s_buffer_atomic_add"},
            {.words = {0xd8800000u, 0u}, .form = "ds_add_u64"}};
  if (target.arch == IsaFamily::Rdna3)
    return {{.words = {0xd8f40000u, 0u}, .form = "ds_consume"},
            {.words = {0xd8f80000u, 0u}, .form = "ds_append"},
            {.words = {0xd8d00000u, 0u}, .form = "ds_wrap_rtn_b32"},
            {.words = {0xf0300100u, 0u}, .form = "image_atomic_add"},
            {.words = {0xd9000000u, 0u}, .form = "ds_add_u64"},
            {.words = {0xd8fc0000u, 0u},
             .form = "ds_ordered_count",
             .memory_kind = amdgpu::MemoryKind::Other},
            {.words = {0xd9e80000u, 0u},
             .form = "ds_add_gs_reg_rtn",
             .memory_kind = amdgpu::MemoryKind::Other},
            {.words = {0xd9ec0000u, 0u},
             .form = "ds_sub_gs_reg_rtn",
             .memory_kind = amdgpu::MemoryKind::Other}};
  if (target.arch == IsaFamily::Rdna4)
    // Native image atomics need dmask=1 and the complete three-word VIMAGE.
    return {{.words = {0xd8f40000u, 0u}, .form = "ds_consume"},
            {.words = {0xd8f80000u, 0u}, .form = "ds_append"},
            {.words = {0xd0430000u, 0u, 0u}, .form = "image_atomic_add_uint"},
            {.words = {0xd9000000u, 0u}, .form = "ds_add_u64"}};
  return {{.words = {0xd8f40000u, 0u}, .form = "ds_consume"},
          {.words = {0xd8f80000u, 0u}, .form = "ds_append"},
          {.words = {0xd9580000u, 0u}, .form = "ds_atomic_async_barrier_arrive_b64"},
          {.words = {0xd9d40000u, 0u}, .form = "ds_atomic_barrier_arrive_rtn_b64"},
          {.words = {0xd9000000u, 0u}, .form = "ds_add_u64"}};
}

inline bool native_atomic_inventory_has_flat(const AtomicTargetCase &target) {
  return target.arch == IsaFamily::Cdna5 || target.arch == IsaFamily::Rdna4;
}

inline std::vector<uint8_t> make_native_atomic_inventory_elf(const AtomicTargetCase &target) {
  std::vector<uint32_t> words;
  for (const NativeAtomicInventorySample &sample : native_atomic_inventory_samples(target))
    words.insert(words.end(), sample.words.begin(), sample.words.end());
  words.push_back(build_s_nop(0, target.arch).value());
  if (native_atomic_inventory_has_flat(target)) {
    const std::array<uint32_t, 3> flat = atomic_test_words(target);
    words.insert(words.end(), flat.begin(), flat.end());
    return make_atomic_sequence_elf(target, std::move(words));
  }
  words.push_back(build_s_endpgm(target.arch).value());
  std::vector<uint8_t> image = make_amdgpu_kernel_elf(words, 0, 3, target.machine, 15, 7, false,
                                                      false, false, false, target.wave == 32);
  image[EI_DATA] = ELFDATA2LSB;
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  return image;
}

} // namespace rocjitsu::test
