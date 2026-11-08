// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/rj_code.h"

#include <cstdint>

namespace rocjitsu::test {

struct DropTargetRow {
  const char *name;
  rj_code_target_id_t target;
  rj_code_arch_t arch;
  uint32_t wave_size;
  bool split;
};
constexpr DropTargetRow kRdna3DropRows[] = {
    {.name = "Gfx1100Wave32",
     .target = ROCJITSU_CODE_TARGET_GFX1100,
     .arch = ROCJITSU_CODE_ARCH_RDNA3,
     .wave_size = 32,
     .split = false},
    {.name = "Gfx1100Wave64",
     .target = ROCJITSU_CODE_TARGET_GFX1100,
     .arch = ROCJITSU_CODE_ARCH_RDNA3,
     .wave_size = 64,
     .split = false},
};
constexpr DropTargetRow kDropRows[] = {
    {.name = "Gfx1200Wave32",
     .target = ROCJITSU_CODE_TARGET_GFX1200,
     .arch = ROCJITSU_CODE_ARCH_RDNA4,
     .wave_size = 32,
     .split = true},
    {.name = "Gfx1200Wave64",
     .target = ROCJITSU_CODE_TARGET_GFX1200,
     .arch = ROCJITSU_CODE_ARCH_RDNA4,
     .wave_size = 64,
     .split = true},
    {.name = "Gfx1201Wave32",
     .target = ROCJITSU_CODE_TARGET_GFX1201,
     .arch = ROCJITSU_CODE_ARCH_RDNA4,
     .wave_size = 32,
     .split = true},
    {.name = "Gfx1201Wave64",
     .target = ROCJITSU_CODE_TARGET_GFX1201,
     .arch = ROCJITSU_CODE_ARCH_RDNA4,
     .wave_size = 64,
     .split = true},
    {.name = "Gfx1250Wave32",
     .target = ROCJITSU_CODE_TARGET_GFX1250,
     .arch = ROCJITSU_CODE_ARCH_CDNA5,
     .wave_size = 32,
     .split = true},
    {.name = "Gfx1251Wave32",
     .target = ROCJITSU_CODE_TARGET_GFX1251,
     .arch = ROCJITSU_CODE_ARCH_CDNA5,
     .wave_size = 32,
     .split = true},
    {.name = "Gfx942Wave64",
     .target = ROCJITSU_CODE_TARGET_GFX942,
     .arch = ROCJITSU_CODE_ARCH_CDNA3,
     .wave_size = 64,
     .split = false},
    {.name = "Gfx950Wave64",
     .target = ROCJITSU_CODE_TARGET_GFX950,
     .arch = ROCJITSU_CODE_ARCH_CDNA4,
     .wave_size = 64,
     .split = false},
    kRdna3DropRows[0],
    kRdna3DropRows[1],
};

} // namespace rocjitsu::test
