// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP32/FP64 scaling by powers of two with guest rounding and denormal policies.
// Operation selectors and backend declarations shared by dispatch and SIMD code.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Default denormal policy keeps the original SIMD kernels; guest variants
// additionally implement explicit input/output flushing.
void ldexp_x86_64_v3(bool fp64, uint32_t exec_mask, uint32_t instruction_flags, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *b);

// Scale FP32/FP64 by powers of two. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void ldexp_guest_x86_64_v3(bool fp64, uint64_t flags, uint32_t exec_mask,
                           uint32_t instruction_flags, uint32_t *const *d, const uint32_t *const *a,
                           const uint32_t *b);

// Scale FP32/FP64 by powers of two. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void ldexp_x86_64_v4(bool fp64, uint32_t exec_mask, uint32_t instruction_flags, uint32_t *const *d,
                     const uint32_t *const *a, const uint32_t *b);

// Scale FP32/FP64 by powers of two. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void ldexp_guest_x86_64_v4(bool fp64, uint64_t flags, uint32_t exec_mask,
                           uint32_t instruction_flags, uint32_t *const *d, const uint32_t *const *a,
                           const uint32_t *b);

} // namespace rocjitsu::arith
