// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Cross-family declarations connecting public dispatch to x86 SIMD kernels.
// Internal helpers shared by instruction implementations; not part of the public C API.

#pragma once

#include "fp/fma.h"

#include <stdint.h>

namespace rocjitsu::arith {

#if defined(RJ_ARITH_HAVE_X86_64_V3)

// Evaluate fp8 wmma for the caller-selected operation.
// Requires the named CPU features and validated instruction controls.
template <bool Bf8A, bool Bf8B>
void fp8_wmma_x86_64_v3(uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c);

// Evaluate integer wmma for the caller-selected operation.
// Requires the named CPU features and validated instruction controls.
template <int Bits, int K>
void integer_wmma_x86_64_v3(uint32_t modifiers, uint32_t *const *d, const uint32_t *const *a,
                            const uint32_t *const *b, const uint32_t *const *c);

// Evaluate integer wmma replicated for the caller-selected operation.
// Requires the named CPU features and validated instruction controls.
template <int Bits, int Lanes>
void integer_wmma_replicated_x86_64_v3(uint32_t modifiers, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c);

// Evaluate wmma f32 for the caller-selected operation.
// Requires the named CPU features and validated instruction controls.
template <bool Bf16, int Lanes, bool Rdna3>
void wmma_f32_x86_64_v3(uint32_t modifiers, uint32_t *const *d, const uint32_t *const *a,
                        const uint32_t *const *b, const uint32_t *const *c);

// Evaluate wmma packed for the caller-selected operation.
// Requires the named CPU features and validated instruction controls.
template <bool Bf16, int Lanes, bool Rdna3>
void wmma_packed_x86_64_v3(uint32_t modifiers, bool saturate, uint32_t *const *d,
                           const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c);

// Evaluate literal fma for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Multiply>
void literal_fma_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t literal, uint32_t *d,
                           const uint32_t *a, const uint32_t *b);

// Evaluate fma for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
void fma_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t modifiers, uint32_t *d,
                   const uint32_t *a, const uint32_t *b, const uint32_t *c);

// Evaluate fma dx9 zero for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
void fma_dx9_zero_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t modifiers, uint32_t *d,
                            const uint32_t *a, const uint32_t *b, const uint32_t *c);
#endif

#if defined(RJ_ARITH_HAVE_X86_64_V4)

// Evaluate literal fma for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
template <bool Multiply>
void literal_fma_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t literal, uint32_t *d,
                           const uint32_t *a, const uint32_t *b);

// Evaluate fma for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
void fma_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t modifiers, uint32_t *d,
                   const uint32_t *a, const uint32_t *b, const uint32_t *c);

// Evaluate fma dx9 zero for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
void fma_dx9_zero_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t modifiers, uint32_t *d,
                            const uint32_t *a, const uint32_t *b, const uint32_t *c);
#endif

#if defined(RJ_ARITH_HAVE_AVX512BF16)
// Accumulate loose BF16 WMMA using native DPBF16 when inputs are eligible.
// May decline without writes for screened input ranges. Native DPBF16 can
// flush subnormal partial sums, even before a normal result; loose semantics
// permit this difference.
template <int Lanes, bool Rdna3, bool Packed>
bool wmma_avx512bf16(uint32_t modifiers, uint32_t *const *d, const uint32_t *const *a,
                     const uint32_t *const *b, const uint32_t *const *c);

#endif

#if defined(RJ_ARITH_HAVE_AVX512VNNI)
// Evaluate integer wmma avx512vnni for the caller-selected operation.
// Requires the named CPU features and validated instruction controls.
template <int Bits, int K>
void integer_wmma_avx512vnni(uint32_t modifiers, uint32_t *const *d, const uint32_t *const *a,
                             const uint32_t *const *b, const uint32_t *const *c);
#endif

#if defined(RJ_ARITH_HAVE_X86_64_V3)
// Nearest guest rounding; special/tiny results are repaired with the inherited model.

// Exact FP16 FMA with nearest guest rounding. Preserves inactive lanes and aliases.
// The caller must disable status reporting or select CLAMP, which suppresses status
// in fp/half_fma_scalar.h. This backend never accumulates FP status.
void half_fma_exact_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                              const uint32_t *a, const uint32_t *b, const uint32_t *c,
                              bool rdna3 = false);

// Evaluate FP16 fused multiply-add. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.

// Compute empirical bit-exact FP16/BF16 A*B+C for the selected gfx layout
// and wave size. Requires validated mode; D may alias inputs.
template <bool Bf16, int Lanes, bool Rdna3, bool Packed>
void wmma_exact_x86_64_v3(uint32_t mode, bool saturate, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *c);

#endif

#if defined(RJ_ARITH_HAVE_X86_64_V4)
// All FP32 inputs and guest rounding modes; exceptional/tiny lanes use scalar repair.

// All FP16 inputs and numeric policies. Unsupported reporting policies decline
// without changing outputs; nonfinite lanes use the inherited scalar model.
bool half_fma_exact_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                              const uint32_t *a, const uint32_t *b, const uint32_t *c,
                              uint32_t *state, bool rdna3 = false);

// Compute empirical bit-exact FP16/BF16 A*B+C for the selected gfx layout
// and wave size. Requires validated mode; D may alias inputs.
template <bool Bf16, int Lanes, bool Rdna3, bool Packed>
void wmma_exact_x86_64_v4(uint32_t mode, bool saturate, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *c);

#endif

#if defined(RJ_ARITH_HAVE_X86_64_V3)
// Nearest guest rounding; special/tiny results are repaired with the inherited model.
bool fma_exact_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                         const uint32_t *a, const uint32_t *b, const uint32_t *c, bool dx9_zero,
                         bool rdna3);

// Evaluate half fma exact for the caller-selected operation. Preserve destination lanes
// outside exec_mask.
// Requires the named CPU features and validated instruction controls. Return false to request
// fallback.

// Evaluate packed fma exact for the caller-selected operation. Preserve destination lanes
// outside exec_mask.
// Requires the named CPU features and validated instruction controls. Return false to request
// fallback.
bool packed_fma_exact_x86_64_v3(uint64_t flags, uint32_t exec_mask, const uint32_t modes[2],
                                uint32_t *d, const uint32_t *a, const uint32_t *b,
                                const uint32_t *c, uint32_t *state, bool rdna3 = false);

#endif

#if defined(RJ_ARITH_HAVE_X86_64_V4)

// All FP32 inputs and guest rounding modes; exceptional/tiny lanes use scalar repair.
bool fma_exact_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                         const uint32_t *a, const uint32_t *b, const uint32_t *c,
                         bool dx9_zero = false, bool rdna3 = false);

// All FP16 inputs and numeric policies. Unsupported reporting policies decline
// without changing outputs; nonfinite lanes use the inherited scalar model.

// Evaluate packed fma exact for the caller-selected operation. Preserve destination lanes
// outside exec_mask.
// Requires the named CPU features and validated instruction controls. Return false to request
// fallback.
bool packed_fma_exact_x86_64_v4(uint64_t flags, uint32_t exec_mask, const uint32_t modes[2],
                                uint32_t *d, const uint32_t *a, const uint32_t *b,
                                const uint32_t *c, uint32_t *state, bool rdna3 = false);

#endif

#if defined(RJ_ARITH_HAVE_X86_64_V3)

// Evaluate FP16 fused multiply-add. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <FmaOperands Operands = FmaOperands::Registers>
void half_fma_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                        const uint32_t *a, const uint32_t *b, const uint32_t *c,
                        uint16_t literal = 0);

#endif
#if defined(RJ_ARITH_HAVE_X86_64_V4)
// Attempt finite MUL domains without writes on decline; uses default host FP state.
// FP status reporting follows the resolved semantics.
bool mul_exact_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                         const uint32_t *a, const uint32_t *b, uint32_t *state);
#endif

} // namespace rocjitsu::arith
