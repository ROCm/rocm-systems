// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP32/FP64 scaling by powers of two with guest rounding and denormal policies.
// Operation selectors and backend declarations shared by dispatch and SIMD code.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Stage exact normal FP64 scales, zeros and non-finite encodings. Return a mask
// of lanes needing scalar rounding/normalization; those retain their modified
// input encoding in result. Do not publish destinations.
uint32_t ldexp_f64_normal_x86_64_v3(uint64_t flags, uint32_t instruction_flags, bool quiet_snan,
                                    uint32_t result[2][32], const uint32_t *const *a,
                                    const uint32_t *b);

// AVX-512 counterpart of the staged exact-normal FP64 kernel above.
uint32_t ldexp_f64_normal_x86_64_v4(uint64_t flags, uint32_t instruction_flags, bool quiet_snan,
                                    uint32_t result[2][32], const uint32_t *const *a,
                                    const uint32_t *b);

// Publish a staged FP64 result, preserving every lane outside exec_mask.
void ldexp_f64_store_x86_64_v3(uint32_t exec_mask, uint32_t *const *d,
                               const uint32_t result[2][32]);

// AVX-512 masked-store counterpart of the staged FP64 publication above.
void ldexp_f64_store_x86_64_v4(uint32_t exec_mask, uint32_t *const *d,
                               const uint32_t result[2][32]);

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
