// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// MFMA backend declarations. Each implementation has an ISA-specific symbol.
#pragma once
#include <stdint.h>

namespace rocjitsu::arith {
template <int M, int K, int Batch, bool Bf16>
void mfma_half_x86_64_v3(uint64_t instruction_flags, uint32_t *const *d, const uint32_t *const *a,
                         const uint32_t *const *b, const uint32_t *const *c);
template <int M, int K, int Batch, bool Bf16>
void mfma_half_x86_64_v4(uint64_t instruction_flags, uint32_t *const *d, const uint32_t *const *a,
                         const uint32_t *const *b, const uint32_t *const *c);
} // namespace rocjitsu::arith
