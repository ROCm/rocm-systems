// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Private x86 entry points for the empirical gfx12 sparse half DOT4 model.
#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Inputs were validated by the public dispatcher. dest holds the tied C/D
// registers as 32-lane sections. All inputs may alias the destination.
template <bool Bf16, int WaveSize, bool Packed>
void swmmac_exact16_x86_64_v3(uint32_t mode, bool saturate, uint32_t *const *dest,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *index);

template <bool Bf16, int WaveSize, bool Packed>
void swmmac_exact16_x86_64_v4(uint32_t mode, bool saturate, uint32_t *const *dest,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *index);

} // namespace rocjitsu::arith
