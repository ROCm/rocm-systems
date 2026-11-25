// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Internal whole-word move kernels; public dispatch validates controls.
#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

#if defined(RJ_ARITH_HAVE_X86_64_V3)
// Copy active Wave32 words with ABS/NEG; requires AVX2 and permits D == A.
void move_words_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a);
#endif

#if defined(RJ_ARITH_HAVE_X86_64_V4)
// Copy active Wave32 words with ABS/NEG; requires AVX-512 and permits D == A.
void move_words_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a);
#endif

} // namespace rocjitsu::arith
