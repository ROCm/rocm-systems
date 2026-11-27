// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Internal SIMD swap kernels; public dispatch validates controls before calling them.
#pragma once

#include <stdint.h>

namespace rocjitsu::arith {
#if defined(RJ_ARITH_HAVE_X86_64_V3)
// Exchange active Wave32 words using AVX2; A and D may coincide.
void swap_words_x86_64_v3(uint32_t exec_mask, uint32_t *d, uint32_t *a);

// Exchange selected halves using AVX2, preserving other halves and inactive lanes.
void swap_half_x86_64_v3(int lanes, uint64_t exec_mask, uint32_t mode, uint32_t *d, uint32_t *a);

// Exchange adjacent rows using AVX2, with independent destination masks.
// Defined for each wave size that holds a row pair.
template <int Lanes, int Stride>
void swap_rows_x86_64_v3(uint64_t exec_mask, uint32_t *d, uint32_t *a);
#endif

#if defined(RJ_ARITH_HAVE_X86_64_V4)
// Exchange active Wave32 words using AVX-512; A and D may coincide.
void swap_words_x86_64_v4(uint32_t exec_mask, uint32_t *d, uint32_t *a);

// Exchange selected halves using AVX-512, preserving other halves and inactive lanes.
void swap_half_x86_64_v4(int lanes, uint64_t exec_mask, uint32_t mode, uint32_t *d, uint32_t *a);

// Exchange adjacent rows using AVX-512, with independent destination masks.
// Defined for each wave size that holds a row pair.
template <int Lanes, int Stride>
void swap_rows_x86_64_v4(uint64_t exec_mask, uint32_t *d, uint32_t *a);
#endif

} // namespace rocjitsu::arith
