// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Cross-family declarations connecting public dispatch to x86 SIMD kernels.
// Internal helpers shared by instruction implementations; not part of the public C API.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

#if defined(RJ_ARITH_HAVE_X86_64_V3)

// Loose FP16/BF16 WMMA with FP32 output. All source reads precede destination
// writes so D may overlap any input.
// Requires the named CPU features and validated instruction controls.
template <bool Bf16, int Lanes, bool Rdna3>
void wmma_f32_x86_64_v3(uint32_t instruction_flags, uint32_t *const *d, const uint32_t *const *a,
                        const uint32_t *const *b, const uint32_t *const *c);

// Compute empirical bit-exact FP16/BF16 A*B+C for the selected gfx layout
// and wave size. Requires validated instruction_flags; D may alias inputs.
template <bool Bf16, int Lanes, bool Rdna3>
void wmma_exact_x86_64_v3(uint32_t instruction_flags, uint32_t *const *d, const uint32_t *const *a,
                          const uint32_t *const *b, const uint32_t *const *c);

// Loose FP32 FMA, supporting D/input aliases. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
void fma_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags, uint32_t *d,
                   const uint32_t *a, const uint32_t *b, const uint32_t *c);

// Nearest guest rounding; special/tiny results are repaired with the inherited model.
void fma_exact_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags,
                         uint32_t *d, const uint32_t *a, const uint32_t *b, const uint32_t *c,
                         bool rdna3);

// Loose FP16 fused multiply-add with nearest guest rounding; dispatch uses scalar
// arithmetic for directed modes. Loose FP32 instead ignores guest rounding on SIMD.
// Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void half_fma_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags, uint32_t *d,
                        const uint32_t *a, const uint32_t *b, const uint32_t *c);

// Exact FP16 FMA with nearest guest rounding. Preserves inactive lanes and aliases.
// The caller must disable status reporting or select CLAMP, which suppresses status
// in fp/half_fma_scalar.h. This backend never accumulates FP status.
void half_fma_exact_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags,
                              uint32_t *d, const uint32_t *a, const uint32_t *b, const uint32_t *c,
                              bool rdna3);

#endif

} // namespace rocjitsu::arith
