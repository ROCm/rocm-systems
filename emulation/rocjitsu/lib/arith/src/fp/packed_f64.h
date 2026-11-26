// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Packed FP64 result policy and ISA-specific masked publication declarations.
#pragma once
#include <stdint.h>
namespace rocjitsu::arith {
// Apply packed selection NaN/output-denormal rules before publishing any overlapping D tuple.
// Staged results and input pointers contain four independent VGPR planes of32 lanes each.
void publish_packed_f64_x86_64_v3(bool selection, uint64_t flags, uint32_t exec_mask, uint32_t mode,
                                  uint32_t *const *d, uint32_t *const *staged,
                                  const uint32_t *const *a, const uint32_t *const *b);
void publish_packed_f64_x86_64_v4(bool selection, uint64_t flags, uint32_t exec_mask, uint32_t mode,
                                  uint32_t *const *d, uint32_t *const *staged,
                                  const uint32_t *const *a, const uint32_t *const *b);
} // namespace rocjitsu::arith
