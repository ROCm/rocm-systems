// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Saturating INT16-to-U8 packing and selection of two FP16 source halfwords.
// Operation selectors and backend declarations shared by dispatch and SIMD code.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Pack saturated bytes or selected FP16 halves. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Saturate>
void pack_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                    const uint32_t *a, const uint32_t *b, bool quiet_nan);

// Pack saturated bytes or selected FP16 halves. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Saturate>
void pack_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                    const uint32_t *a, const uint32_t *b, bool quiet_nan);

} // namespace rocjitsu::arith
