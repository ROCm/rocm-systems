// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/rj_code.h"

#include <array>
#include <string_view>

namespace rocjitsu::consan {

/// @brief Declarative identities for the native LDS lowering implemented here.
/// @details Compiler consumers additionally resolve the selected registry binding;
/// runtime and offline readers do not link a decoder or execution provider.
class TargetProfile {
public:
  uint32_t id;
  std::string_view name;
  rj_code_arch_t arch;
  rj_code_target_id_t public_id;
  uint32_t elf_machine;
  uint32_t default_wave_size;
  bool wave64;

  bool operator==(const TargetProfile &) const = default;
};

inline constexpr std::array kNativeLdsTargets = {
    TargetProfile{.id = 942,
                  .name = "gfx942",
                  .arch = ROCJITSU_CODE_ARCH_CDNA3,
                  .public_id = ROCJITSU_CODE_TARGET_GFX942,
                  .elf_machine = EF_AMDGPU_MACH_AMDGCN_GFX942,
                  .default_wave_size = 64,
                  .wave64 = true},
    TargetProfile{.id = 950,
                  .name = "gfx950",
                  .arch = ROCJITSU_CODE_ARCH_CDNA4,
                  .public_id = ROCJITSU_CODE_TARGET_GFX950,
                  .elf_machine = EF_AMDGPU_MACH_AMDGCN_GFX950,
                  .default_wave_size = 64,
                  .wave64 = true},
    TargetProfile{.id = 1100,
                  .name = "gfx1100",
                  .arch = ROCJITSU_CODE_ARCH_RDNA3,
                  .public_id = ROCJITSU_CODE_TARGET_INVALID,
                  .elf_machine = EF_AMDGPU_MACH_AMDGCN_GFX1100,
                  .default_wave_size = 32,
                  .wave64 = true},
    TargetProfile{.id = 1201,
                  .name = "gfx1201",
                  .arch = ROCJITSU_CODE_ARCH_RDNA4,
                  .public_id = ROCJITSU_CODE_TARGET_GFX1201,
                  .elf_machine = EF_AMDGPU_MACH_AMDGCN_GFX1201,
                  .default_wave_size = 32,
                  .wave64 = true},
    TargetProfile{.id = 1250,
                  .name = "gfx1250",
                  .arch = ROCJITSU_CODE_ARCH_CDNA5,
                  .public_id = ROCJITSU_CODE_TARGET_GFX1250,
                  .elf_machine = EF_AMDGPU_MACH_AMDGCN_GFX1250,
                  .default_wave_size = 32,
                  .wave64 = false},
};

inline constexpr const TargetProfile *target_profile(uint32_t id) {
  for (const TargetProfile &target : kNativeLdsTargets)
    if (target.id == id)
      return &target;
  return nullptr;
}

inline constexpr const TargetProfile *target_profile(std::string_view name) {
  for (const TargetProfile &target : kNativeLdsTargets)
    if (target.name == name)
      return &target;
  return nullptr;
}

inline constexpr bool supports_wave(const TargetProfile &target, uint32_t wave_size) {
  return wave_size == target.default_wave_size || (target.wave64 && wave_size == 64);
}

} // namespace rocjitsu::consan
