// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Packed element permutation backend contracts; inputs are arrays of VGPRs.
#pragma once

#include <stdint.h>

namespace rocjitsu::arith {
// Select sixteen packed elements with four-bit indices. All source words are
// consumed before storing a lane chunk, allowing arbitrary whole-VGPR aliases.
void packed_permute_x86_64_v3(int bits, int lanes, uint64_t exec_mask, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c);
void packed_permute_x86_64_v4(int bits, int lanes, uint64_t exec_mask, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c);
} // namespace rocjitsu::arith
