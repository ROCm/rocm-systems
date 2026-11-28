// SPDX-License-Identifier: MIT

// Baseline-only entry to the gfx942 model; callers validate the qualified controls.
#pragma once

#include "common/internal.h"

namespace rocjitsu::arith {
enum class MfmaGfx942Format { F16, Bf16, Xf32, Fp8Fp8, Fp8Bf8, Bf8Fp8, Bf8Bf8 };

// Qualified MI300X dense MFMA controls: CBSZ 0..4 with ABID < 2^CBSZ and any BLGP 0..7.
constexpr bool is_gfx942_mfma_controls_qualified(uint64_t controls) {
  const uint64_t cbsz = controls & 7ULL;
  const uint64_t abid = (controls >> 3) & 15ULL;
  return cbsz <= 4ULL && abid < (1ULL << cbsz);
}
} // namespace rocjitsu::arith

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Produce a complete wave over pre-staged/permuted source VGPRs with optional per-step
// no-Inf/NaN clamping, supporting whole-VGPR aliases.
void mfma_gfx942_exact(MfmaGfx942Format format, int m, int k, int batches, bool no_inf_nan,
                       uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                       const uint32_t *const *c);

// Produce a complete sparse MFMA wave with CBSZ/ABID metadata selection and optional per-step
// no-Inf/NaN clamping, supporting whole-VGPR aliases.
void smfmac_gfx942_exact(MfmaGfx942Format format, int m, uint64_t controls, bool no_inf_nan,
                         uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                         const uint32_t *const *index);
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
