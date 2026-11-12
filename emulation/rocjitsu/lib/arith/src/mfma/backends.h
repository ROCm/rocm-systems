// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// MFMA backend declarations. Each implementation has an ISA-specific symbol.
#pragma once
#include <stdint.h>

namespace rocjitsu::arith {
// XF32 factors retain ten fraction bits before multiplication.
inline constexpr uint32_t kMfmaXf32InputMask = 0xffffe000u;

template <int M, int Batch>
void mfma_f64_x86_64_v3(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                        const uint32_t *const *b, const uint32_t *const *c);
template <int M, int Batch>
void mfma_f64_x86_64_v4(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                        const uint32_t *const *b, const uint32_t *const *c);

template <int M, int K, int Batch, bool Wmma>
void matrix_i8_x86_64_v3(uint64_t controls, bool clamp_overflow, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c);
template <int M, int K, int Batch, bool Wmma>
void matrix_i8_x86_64_v4(uint64_t controls, bool clamp_overflow, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c);

template <int M, int K, int Batch, bool Bf16>
void mfma_half_x86_64_v3(uint64_t instruction_flags, bool clamp_overflow, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c);
template <int M, int K, int Batch, bool Bf16>
void mfma_half_x86_64_v4(uint64_t instruction_flags, bool clamp_overflow, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c);
template <int M, int K, int Batch, bool Xf32 = false>
void mfma_f32_x86_64_v3(uint64_t controls, bool clamp_overflow, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c);
template <int M, int K, int Batch, bool Xf32 = false>
void mfma_f32_x86_64_v4(uint64_t controls, bool clamp_overflow, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c);
template <int M, int K, bool Bf8A, bool Bf8B, bool Fnuz>
void mfma_fp8_x86_64_v3(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                        const uint32_t *const *b, const uint32_t *const *c);
template <int M, int K, bool Bf8A, bool Bf8B, bool Fnuz>
void mfma_fp8_x86_64_v4(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                        const uint32_t *const *b, const uint32_t *const *c);
template <int M, int K>
void mfma_mixed_x86_64_v3(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                          const uint32_t *const *b, const uint32_t *const *c);
template <int M, int K>
void mfma_mixed_x86_64_v4(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                          const uint32_t *const *b, const uint32_t *const *c);
template <int M, int K>
void mfma_scaled_x86_64_v3(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                           const uint32_t *const *b, const uint32_t *const *c,
                           const uint32_t *scale_a, const uint32_t *scale_b, bool clamp_overflow);
template <int M, int K>
void mfma_scaled_x86_64_v4(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                           const uint32_t *const *b, const uint32_t *const *c,
                           const uint32_t *scale_a, const uint32_t *scale_b, bool clamp_overflow);
} // namespace rocjitsu::arith
