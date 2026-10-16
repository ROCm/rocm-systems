// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP16 scaling by powers of two and integer exponent extraction.
// Operation selectors and backend declarations shared by dispatch and SIMD code.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Scale or decompose FP16 exponents. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Ldexp>
void half_exponent_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                             const uint32_t *a, const uint32_t *b);

// Scale FP16 by powers of two with guest input/output flushing; preserve inactive lanes.
// Requires the named CPU features and validated instruction controls.
void half_ldexp_guest_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                                const uint32_t *a, const uint32_t *b);

} // namespace rocjitsu::arith
