// SPDX-License-Identifier: MIT

// Cross-family declarations connecting public dispatch to x86 SIMD kernels.
// Internal helpers shared by instruction implementations; not part of the public C API.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {
// Evaluate half dot for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
void half_dot_x86_64_v4(bool bf16, bool saturate, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                        const uint32_t *a, const uint32_t *b, const uint32_t *c);

// Evaluate fp8 dot for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
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

// Evaluate half dot for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
void half_dot_x86_64_v3(bool bf16, bool saturate, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                        const uint32_t *a, const uint32_t *b, const uint32_t *c);

// Evaluate fp8 wmma for the caller-selected operation.
// Requires the named CPU features and validated instruction controls.
void fp8_wmma_x86_64_v3(bool bf8_a, bool bf8_b, uint32_t *const *d, const uint32_t *const *a,
                        const uint32_t *const *b, const uint32_t *const *c);

// Evaluate fp8 dot for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
void fp8_dot_x86_64_v3(bool bf8_a, bool bf8_b, uint32_t exec_mask, uint32_t modifiers, uint32_t *d,
                       const uint32_t *a, const uint32_t *b, const uint32_t *c);

// Evaluate dot2 for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
void dot2_x86_64_v3(bool bf16, uint32_t exec_mask, uint32_t modifiers, uint32_t *d,
                    const uint32_t *a, const uint32_t *b, const uint32_t *c);

// Evaluate integer dot for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
void integer_dot_x86_64_v3(int bits, bool unsigned_acc, uint32_t exec_mask, uint32_t modifiers,
                           uint32_t *d, const uint32_t *a, const uint32_t *b, const uint32_t *c);

// Evaluate integer wmma for the caller-selected operation.
// Requires the named CPU features and validated instruction controls.
void integer_wmma_x86_64_v3(int bits, int k, uint32_t modifiers, uint32_t *const *d,
                            const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *c);

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
// Evaluate dot2 bf16 for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
void dot2_bf16_x86_64_v4(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                         const uint32_t *b, const uint32_t *c);

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
// Returns false without destination writes when fixed DAZ/FTZ would be unsafe.
template <int Lanes, bool Rdna3, bool Packed>
bool wmma_avx512bf16(uint32_t modifiers, uint32_t *const *d, const uint32_t *const *a,
                     const uint32_t *const *b, const uint32_t *const *c);

#endif

#if defined(RJ_ARITH_HAVE_AVX512VNNI)
// Evaluate integer wmma avx512vnni for the caller-selected operation.
// Requires the named CPU features and validated instruction controls.
void integer_wmma_avx512vnni(int bits, int k, uint32_t modifiers, uint32_t *const *d,
                             const uint32_t *const *a, const uint32_t *const *b,
                             const uint32_t *const *c);
#endif

} // namespace rocjitsu::arith
