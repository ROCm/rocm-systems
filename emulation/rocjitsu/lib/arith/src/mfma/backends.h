// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// MFMA backend declarations. Each implementation has an ISA-specific symbol.
#pragma once
#include "mfma/f32_model.h"
#include "mfma/gfx942_exact.h"
#include <stdint.h>

namespace rocjitsu::arith {
// XF32 factors retain ten fraction bits before multiplication.
inline constexpr uint32_t kMfmaXf32InputMask = 0xffffe000u;

template <MfmaGfx942Format Format, int M, int K, int Batch>
void mfma_gfx942_exact_x86_64_v3(uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c);
template <MfmaGfx942Format Format, int M, int K, int Batch>
void mfma_gfx942_exact_x86_64_v4(uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b, const uint32_t *const *c);
template <MfmaGfx942Format Format, int M, int K>
void smfmac_gfx942_exact_x86_64_v3(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, const uint32_t *const *index);
template <MfmaGfx942Format Format, int M, int K>
void smfmac_gfx942_exact_x86_64_v4(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b, const uint32_t *const *index);

// gfx1250 Wave32 byte matrix operands have passed public validation and metadata normalization.
using Gfx1250IntegerBackend = void(uint64_t, bool, uint32_t *const *, const uint32_t *const *,
                                   const uint32_t *const *, const uint32_t *const *,
                                   const uint32_t *const *);
template <bool Sparse> Gfx1250IntegerBackend gfx1250_integer_x86_64_v3;
template <bool Sparse> Gfx1250IntegerBackend gfx1250_integer_x86_64_v4;
template <bool Sparse> Gfx1250IntegerBackend gfx1250_integer_avx512vnni;

template <int M, int K, bool Wmma>
void sparse_matrix_i8_x86_64_v3(uint64_t controls, bool clamp, bool clamp_per_pass,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *index);
template <int M, int K, bool Wmma>
void sparse_matrix_i8_x86_64_v4(uint64_t controls, bool clamp, bool clamp_per_pass,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *index);
template <int M, int K>
void sparse_matrix_i8_avx512vnni(uint64_t controls, bool clamp, uint32_t *const *d,
                                 const uint32_t *const *a, const uint32_t *const *b,
                                 const uint32_t *const *index);

template <int M, int K, int Batch, bool Wmma, int ABlockLanes = 0>
void matrix_i8_x86_64_v3(uint64_t controls, bool clamp_overflow, bool clamp_per_pass,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c);
template <int M, int K, int Batch, bool Wmma, int ABlockLanes = 0>
void matrix_i8_x86_64_v4(uint64_t controls, bool clamp_overflow, bool clamp_per_pass,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c);
template <int M, int K, int Batch>
void matrix_i8_avx512vnni(uint64_t controls, bool clamp_overflow, uint32_t *const *d,
                          const uint32_t *const *a, const uint32_t *const *b,
                          const uint32_t *const *c);

template <int M, int K, int Batch, bool Bf16, MfmaF32Model Model = MfmaF32Model::Default>
void mfma_half_x86_64_v3(uint64_t instruction_flags, uint64_t flags, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c);
template <int M, int K, int Batch, bool Bf16, MfmaF32Model Model = MfmaF32Model::Default>
void mfma_half_x86_64_v4(uint64_t instruction_flags, uint64_t flags, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *c);
template <int M, int K, int Batch, int ABlockLanes = 0>
bool mfma_half_x86_64_v4_bf16(uint64_t instruction_flags, uint64_t flags, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c);
template <int M, int K, int Batch, bool Xf32 = false, MfmaF32Model Model = MfmaF32Model::Default>
void mfma_f32_x86_64_v3(uint64_t controls, uint64_t flags, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c);
template <int M, int K, int Batch, bool Xf32 = false, MfmaF32Model Model = MfmaF32Model::Default>
void mfma_f32_x86_64_v4(uint64_t controls, uint64_t flags, uint32_t *const *d,
                        const uint32_t *const *a, const uint32_t *const *b,
                        const uint32_t *const *c);
template <int M, int K, bool Bf8A, bool Bf8B, bool Fnuz>
void mfma_fp8_x86_64_v3(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                        const uint32_t *const *b, const uint32_t *const *c);
template <int M, int K, bool Bf8A, bool Bf8B, bool Fnuz>
void mfma_fp8_x86_64_v4(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                        const uint32_t *const *b, const uint32_t *const *c);
template <int M, int K, bool Bf8A, bool Bf8B, bool Fnuz>
bool mfma_fp8_x86_64_v4_bf16(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                             const uint32_t *const *b, const uint32_t *const *c,
                             bool preserve_compatibility);
template <int M, int K>
void mfma_mixed_x86_64_v3(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                          const uint32_t *const *b, const uint32_t *const *c);
template <int M, int K>
void mfma_mixed_x86_64_v4(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                          const uint32_t *const *b, const uint32_t *const *c);
template <int M, int K>
bool mfma_mixed_x86_64_v4_bf16(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c,
                               bool preserve_compatibility);
template <int M, int K>
bool mfma_mixed_avx512vnni(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                           const uint32_t *const *b, const uint32_t *const *c);
template <int M, int K>
void mfma_scaled_x86_64_v3(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                           const uint32_t *const *b, const uint32_t *const *c,
                           const uint32_t *scale_a, const uint32_t *scale_b, bool clamp_overflow);
template <int M, int K>
void mfma_scaled_x86_64_v4(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                           const uint32_t *const *b, const uint32_t *const *c,
                           const uint32_t *scale_a, const uint32_t *scale_b, bool clamp_overflow);
template <int M, int K>
void mfma_scaled_x86_64_v4_bf16(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c,
                                const uint32_t *scale_a, const uint32_t *scale_b,
                                bool clamp_overflow);
template <int M, int K>
bool mfma_scaled_avx512vnni(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                            const uint32_t *const *b, const uint32_t *const *c,
                            const uint32_t *scale_a, const uint32_t *scale_b, bool clamp_overflow);
template <int M, int K, int Bits, bool BfA, bool BfB, bool Fnuz>
void smfmac_float_x86_64_v3(uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *index, bool clamp_overflow,
                            bool preserve_compatibility);
template <int M, int K, int Bits, bool BfA, bool BfB, bool Fnuz>
void smfmac_float_x86_64_v4(uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                            const uint32_t *const *index, bool clamp_overflow);
} // namespace rocjitsu::arith
