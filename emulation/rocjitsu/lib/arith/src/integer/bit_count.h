// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Population count, bit reversal, and first/last significant-bit searches.
// Operation selectors and backend declarations shared by dispatch and SIMD code.

#pragma once

#include "common/internal.h"

#include <stdint.h>

namespace rocjitsu::arith {

enum class BitCount { Leading, Trailing, Sign, Population, MaskedLow, MaskedHigh };

// Count or reorder bits. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <BitCount Op, int Lanes>
void bit_count_x86_64_v3(ExecMask<Lanes> exec_mask, uint32_t *d, const uint32_t *a,
                         const uint32_t *b);

// Count or reorder bits. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <BitCount Op, int Lanes>
void bit_count_x86_64_v4(ExecMask<Lanes> exec_mask, uint32_t *d, const uint32_t *a,
                         const uint32_t *b);

} // namespace rocjitsu::arith
