// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// ISA-specific gfx1250 floating WMMA backend declarations.
#pragma once
#include <stdint.h>

namespace rocjitsu::arith {
void wmma_gfx1250_f32_x86_64_v3(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c);
void wmma_gfx1250_f32_x86_64_v4(uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c);

// Operand/result formats, including the mixed FP32-accumulator BF16-result form.
enum class Wmma1250Form { F32F16, F32Bf16, F16, Bf16, Bf16F32 };
template <Wmma1250Form Form>
void wmma_gfx1250_half_x86_64_v3(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                 const uint32_t *const *a, const uint32_t *const *b,
                                 const uint32_t *const *c);
template <Wmma1250Form Form>
void wmma_gfx1250_half_x86_64_v4(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                 const uint32_t *const *a, const uint32_t *const *b,
                                 const uint32_t *const *c);
template <int K, bool Bf8A, bool Bf8B, bool Packed>
void wmma_gfx1250_fp8_x86_64_v3(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                const uint32_t *const *c);
template <int K, bool Bf8A, bool Bf8B, bool Packed>
void wmma_gfx1250_fp8_x86_64_v4(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                const uint32_t *const *a, const uint32_t *const *b,
                                const uint32_t *const *c);
} // namespace rocjitsu::arith
