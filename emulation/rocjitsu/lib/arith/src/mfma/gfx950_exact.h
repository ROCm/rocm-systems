// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Baseline-only entry points to the MI350X-calibrated gfx950 matrix and BF16 DOT2 models.
#pragma once

#include "common/internal.h"

#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

enum class MfmaGfx950Format { F16, Bf16, Fp8Fp8, Fp8Bf8, Bf8Fp8, Bf8Bf8 };

// Produce a complete gfx950 dense MFMA wave with default CBSZ/ABID/BLGP, supporting whole-VGPR
// aliases and per-pass RJ_ARITH_FP16_OVFL saturation.
void mfma_gfx950_exact(MfmaGfx950Format format, int m, int k, int batches, bool fp16_ovfl,
                       uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                       const uint32_t *const *c);

// Produce a complete gfx950 sparse MFMA wave (compact or wide) with CBSZ/ABID metadata selection,
// supporting whole-VGPR aliases and per-pass RJ_ARITH_FP16_OVFL saturation.
void smfmac_gfx950_exact(MfmaGfx950Format format, int m, int k, uint64_t controls, bool fp16_ovfl,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *index);

// Evaluate one lane of gfx950 V_DOT2{,C}_F32_BF16 from signed/selected BF16 halves and modified C.
uint32_t dot2_bf16_gfx950_exact(uint16_t a0, uint16_t b0, uint16_t a1, uint16_t b1,
                                uint32_t c_bits);

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
