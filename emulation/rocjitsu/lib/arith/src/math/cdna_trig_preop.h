// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Compact CDNA lookup backends evaluate and publish complete FP64 results.
#pragma once
#include <stdint.h>

namespace rocjitsu::arith {
// Process a whole Wave32/64 using the named ISA and the 51-chunk table.
void cdna_preop_x86_64_v3(int lanes, const uint32_t *a_hi, const uint32_t *b,
                          const uint32_t *chunks, int omod, unsigned rounding, bool clamp,
                          uint64_t exec_mask, uint32_t *const *d);
void cdna_preop_x86_64_v4(int lanes, const uint32_t *a_hi, const uint32_t *b,
                          const uint32_t *chunks, int omod, unsigned rounding, bool clamp,
                          uint64_t exec_mask, uint32_t *const *d);
} // namespace rocjitsu::arith
