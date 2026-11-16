// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Two-register copy kernels. Sources may alias either destination register.
#pragma once
#include <stdint.h>
namespace rocjitsu::arith {
// Copy selected lanes after reading both source words; requires the named ISA.
void pair_move_x86_64_v3(int lanes, uint64_t exec_mask, uint32_t *const *d, const uint32_t *low,
                         const uint32_t *high);
void pair_move_x86_64_v4(int lanes, uint64_t exec_mask, uint32_t *const *d, const uint32_t *low,
                         const uint32_t *high);
} // namespace rocjitsu::arith
