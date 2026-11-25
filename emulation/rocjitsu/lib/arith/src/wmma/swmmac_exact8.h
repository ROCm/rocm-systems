// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Exact gfx12 sparse DOT8 backends. D holds eight 32-lane sections of the
// tied accumulator. Key is the bit offset of the selected metadata bank.
#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

void swmmac_exact8_x86_64_v3(bool bf8_a, bool bf8_b, bool wave64, int key, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *index);

void swmmac_exact8_x86_64_v4(bool bf8_a, bool bf8_b, bool wave64, int key, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *index);

} // namespace rocjitsu::arith
