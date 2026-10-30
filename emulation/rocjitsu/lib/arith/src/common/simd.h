// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Cross-family declarations connecting public dispatch to x86 SIMD kernels.
// Internal helpers shared by instruction implementations; not part of the public C API.

#pragma once

#include "fp/fma.h"

#include <stdint.h>

namespace rocjitsu::arith {
// Loose FP16/BF16 DOT2 with a selected 16-bit result half.
// Preserve the other destination half and inactive lanes; D may alias inputs.
// Requires the named CPU features and validated instruction controls.
void half_dot_x86_64_v4(bool bf16, bool saturate, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                        const uint32_t *a, const uint32_t *b, const uint32_t *c);

// Loose four-product FP8/BF8 dot with FP32 accumulation.
// Preserve inactive lanes; D may alias inputs.
// Requires the named CPU features and validated instruction controls.
void fp8_dot_x86_64_v4(bool bf8_a, bool bf8_b, uint32_t exec_mask, uint32_t modifiers, uint32_t *d,
                       const uint32_t *a, const uint32_t *b, const uint32_t *c);

#if defined(RJ_ARITH_HAVE_X86_64_V3)
// Evaluate permlane64 for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
void permlane64_x86_64_v3(uint64_t exec_mask, uint32_t *d, const uint32_t *a);

// Evaluate move half for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
void move_half_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a);

// Evaluate swap half for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
void swap_half_x86_64_v3(int lanes, uint64_t exec_mask, uint32_t mode, uint32_t *d, uint32_t *a);

// Evaluate ds bpermute for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
void ds_bpermute_x86_64_v3(int lanes, int group, uint64_t exec_mask, uint32_t *d,
                           const uint32_t *addr, const uint32_t *data, uint16_t offset);

// Evaluate ds swizzle for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
void ds_swizzle_x86_64_v3(int lanes, uint64_t exec_mask, uint32_t *d, const uint32_t *a,
                          uint16_t offset);

// Loose FP16/BF16 DOT2 with a selected 16-bit result half.
// Preserve the other destination half and inactive lanes; D may alias inputs.
// Requires the named CPU features and validated instruction controls.
void half_dot_x86_64_v3(bool bf16, bool saturate, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                        const uint32_t *a, const uint32_t *b, const uint32_t *c);

// Loose FP8/BF8 WMMA in gfx12 Wave32 layout. D may alias any input.
// Requires the named CPU features and validated instruction controls.
template <bool Bf8A, bool Bf8B>
void fp8_wmma_x86_64_v3(uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c);

// Loose four-product FP8/BF8 dot with FP32 accumulation.
// Preserve inactive lanes; D may alias inputs.
// Requires the named CPU features and validated instruction controls.
void fp8_dot_x86_64_v3(bool bf8_a, bool bf8_b, uint32_t exec_mask, uint32_t modifiers, uint32_t *d,
                       const uint32_t *a, const uint32_t *b, const uint32_t *c);

// Loose two-product FP16/BF16 dot with FP32 accumulation and half selectors.
// Preserve inactive lanes; D may alias inputs.
// Requires the named CPU features and validated instruction controls.
void dot2_x86_64_v3(bool bf16, uint32_t exec_mask, uint32_t modifiers, uint32_t *d,
                    const uint32_t *a, const uint32_t *b, const uint32_t *c);

// Exact packed INT4/INT8 dot with wrapping or saturating INT32 accumulation.
// Half selection is resolved by the caller; preserve inactive lanes and aliases.
// Requires the named CPU features and validated instruction controls.
void integer_dot_x86_64_v3(int bits, bool unsigned_acc, uint32_t exec_mask, uint32_t modifiers,
                           uint32_t *d, const uint32_t *a, const uint32_t *b, const uint32_t *c);

// Exact integer WMMA in gfx12 Wave32 layout, with staged CLAMP. D may alias inputs.
// Requires the named CPU features and validated instruction controls.
template <int Bits, int K>
void integer_wmma_x86_64_v3(uint32_t instruction_flags, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c);

// Exact integer WMMA in gfx11 replicated layout for either wave size.
// D may alias inputs; each sixteen-lane group supplies its own factors.
// Requires the named CPU features and validated instruction controls.
template <int Bits, int Lanes>
void integer_wmma_replicated_x86_64_v3(uint32_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c);

// Loose FP16/BF16 WMMA with FP32 output in gfx11 replicated or gfx12 layout.
// Supports both wave sizes and D/input aliasing.
// Requires the named CPU features and validated instruction controls.
template <bool Bf16, int Lanes, bool Rdna3>
void wmma_f32_x86_64_v3(uint32_t instruction_flags, uint32_t *const *d, const uint32_t *const *a,
                        const uint32_t *const *b, const uint32_t *const *c);

// Compute empirical bit-exact FP16/BF16 A*B+C for the selected gfx layout
// and wave size. Requires validated instruction flags; D may alias inputs.
template <bool Bf16, int Lanes, bool Rdna3, bool Packed>
void wmma_exact_x86_64_v3(uint32_t instruction_flags, bool saturate, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *c);

// Loose WMMA with packed FP16/BF16 output in gfx11 replicated or gfx12 layout.
// Supports both wave sizes, saturation, and D/input aliasing.
// Requires the named CPU features and validated instruction controls.
template <bool Bf16, int Lanes, bool Rdna3>
void wmma_packed_x86_64_v3(uint32_t instruction_flags, bool saturate, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c);

// Compute A*literal+B when (Operands == rocjitsu::arith::FmaOperands::MultiplyLiteral), otherwise
// A*B+literal. Operands must be MultiplyLiteral or AddLiteral. Preserve inactive lanes. Requires
// the named CPU features and validated instruction controls.
template <rocjitsu::arith::FmaOperands Operands>
void literal_fma_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t literal, uint32_t *d,
                           const uint32_t *a, const uint32_t *b);

// Loose FP32 FMA with source/output modifiers and guest denormal controls.
// Preserves inactive lanes and supports D/input aliasing.
// Requires the named CPU features and validated instruction controls.
void fma_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags, uint32_t *d,
                   const uint32_t *a, const uint32_t *b, const uint32_t *c);

// Compute loose FP32 FMA with DX9 zero-product rules and forced output flushing.
// Preserve inactive lanes.
// Requires the named CPU features and validated instruction controls.
void fma_dx9_zero_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags,
                            uint32_t *d, const uint32_t *a, const uint32_t *b, const uint32_t *c);

// Exact FP16 FMA with nearest guest rounding. Preserves inactive lanes and aliases.
// The caller must disable status reporting or select CLAMP, which suppresses status
// in fp/half_fma_scalar.h. This backend never accumulates FP status.
void half_fma_exact_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags,
                              uint32_t *d, const uint32_t *a, const uint32_t *b, const uint32_t *c,
                              bool rdna3);

// Nearest guest rounding; special/tiny results are repaired with the inherited model.
void fma_exact_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags,
                         uint32_t *d, const uint32_t *a, const uint32_t *b, const uint32_t *c,
                         bool dx9_zero, bool rdna3);

// Loose FP16 FMA with nearest guest rounding; directed modes use exact AVX-512
// or scalar arithmetic. Loose FP32 SIMD ignores guest rounding. Preserve inactive lanes.
// Requires the named CPU features and validated instruction controls.
template <FmaOperands Operands = FmaOperands::Registers>
void half_fma_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags, uint32_t *d,
                        const uint32_t *a, const uint32_t *b, const uint32_t *c,
                        uint16_t literal = 0);

#endif

#if defined(RJ_ARITH_HAVE_X86_64_V4)
// Loose BF16 DOT2 using widened FP32 FMA, with half selectors.
// Preserve inactive lanes; D may alias inputs.
// Requires the named CPU features and validated instruction controls.
void dot2_bf16_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                         const uint32_t *b, const uint32_t *c);

// Compute A*literal+B when (Operands == rocjitsu::arith::FmaOperands::MultiplyLiteral), otherwise
// A*B+literal. Operands must be MultiplyLiteral or AddLiteral. Preserve inactive lanes. Requires
// the named CPU features and validated instruction controls.
template <rocjitsu::arith::FmaOperands Operands>
void literal_fma_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t literal, uint32_t *d,
                           const uint32_t *a, const uint32_t *b);

// Loose FP32 FMA with source/output modifiers and guest denormal controls.
// Preserves inactive lanes and supports D/input aliasing.
// Requires the named CPU features and validated instruction controls.
void fma_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags, uint32_t *d,
                   const uint32_t *a, const uint32_t *b, const uint32_t *c);

// Compute loose FP32 FMA with DX9 zero-product rules and forced output flushing.
// Preserve inactive lanes.
// Requires the named CPU features and validated instruction controls.
void fma_dx9_zero_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags,
                            uint32_t *d, const uint32_t *a, const uint32_t *b, const uint32_t *c);

// All FP16 inputs and numeric policies; nonfinite lanes use scalar repair.
// Status reporting requires gfx12, nearest guest rounding, and no denormal flushing.
// The caller validates this domain before dispatch; D may alias inputs.
void half_fma_exact_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags,
                              uint32_t *d, const uint32_t *a, const uint32_t *b, const uint32_t *c,
                              uint32_t *state, bool rdna3);

// Compute empirical bit-exact FP16/BF16 A*B+C for the selected gfx layout
// and wave size. Requires validated instruction flags; D may alias inputs.
template <bool Bf16, int Lanes, bool Rdna3, bool Packed>
void wmma_exact_x86_64_v4(uint32_t instruction_flags, bool saturate, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *c);

// All FP32 inputs and guest rounding modes; exceptional/tiny lanes use scalar repair.
void fma_exact_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags,
                         uint32_t *d, const uint32_t *a, const uint32_t *b, const uint32_t *c,
                         bool dx9_zero, bool rdna3);

#endif

#if defined(RJ_ARITH_HAVE_X86_64_V4_BF16)

// Accumulate loose BF16 WMMA using native DPBF16 when inputs are eligible.
// May decline without writes for screened input ranges. Native DPBF16 can
// flush subnormal partial sums, even before a normal result; loose semantics
// permit this difference.
template <int Lanes, bool Rdna3, bool Packed>
bool wmma_x86_64_v4_bf16(uint32_t instruction_flags, uint32_t *const *d, const uint32_t *const *a,
                         const uint32_t *const *b, const uint32_t *const *c);

#endif

#if defined(RJ_ARITH_HAVE_AVX512VNNI)

// Exact integer WMMA in gfx12 Wave32 layout, with staged CLAMP. D may alias inputs.
// Requires the named CPU features and validated instruction controls.
template <int Bits, int K>
void integer_wmma_avx512vnni(uint32_t instruction_flags, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c);

#endif

#if defined(RJ_ARITH_HAVE_X86_64_V4)
// Evaluate division FMA with exact guest rounding; decline without writes when unsupported.
bool div_fmas_exact_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t *d, const uint32_t *a,
                              const uint32_t *b, const uint32_t *c, uint32_t *state,
                              uint32_t mode = 0);
#endif

} // namespace rocjitsu::arith
