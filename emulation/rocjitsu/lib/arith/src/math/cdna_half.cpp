// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// CDNA FP16 math reuses Rocjitsu's existing models, half narrowing and guest-mode policies.
// RDNA validation does not establish empirical bit-exactness on these architectures.

#include "common/sdwa_dispatch.h"
#include "common/validation.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_math.h"

#include <stdint.h>

namespace {

// Forward the inherited model while rejecting strict requests before touching storage.
// Compatibility retains the implementation's numeric and FP-status policy.
template <bool AllowSdwa = false, typename Call, typename Mask>
int inherited_half_math(Call call, uint64_t flags, Mask exec_mask, uint64_t mode,
                        uint32_t *const *d, const uint32_t *const *a, uint32_t *state) {
  const uint64_t semantics = flags & RJ_ARITH_SEMANTICS_MASK;
  if (semantics != RJ_ARITH_SEMANTICS_COMPATIBILITY) {
    if (int error = rocjitsu::arith::validate_semantics(flags)) {
      return error;
    }
  }
  if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS ||
      semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  if constexpr (AllowSdwa) {
    return rocjitsu::arith::with_sdwa_half_finalization(
        flags, exec_mask, mode, d,
        [&](uint64_t core_mode) { return call(flags, exec_mask, core_mode, d, a, state); });
  }
  return call(flags, exec_mask, mode, d, a, state);
}

// Forward MI350X-validated FP16 SFU semantics; reject uncaptured active OMOD in strict mode 2.
template <typename Call, typename Mask>
int gfx950_half_math(Call call, uint64_t flags, Mask exec_mask, uint64_t mode, uint32_t *const *d,
                     const uint32_t *const *a, uint32_t *trapsts) {
  if (mode & RJ_ARITH_ALU_SDWA) {
    return inherited_half_math<true>(call, flags, exec_mask, mode, d, a, trapsts);
  }
  const uint64_t semantics = flags & RJ_ARITH_SEMANTICS_MASK;
  const bool active_omod = (mode & RJ_ARITH_ALU_OMOD_MASK) &&
                           (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) &&
                           !(flags & RJ_ARITH_FP_IEEE_MODE);
  if (active_omod) {
    if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
      if (int error = call(flags, 0, mode, nullptr, nullptr, nullptr)) {
        return error;
      }
      return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
    }
    if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE) {
      flags = (flags & ~RJ_ARITH_SEMANTICS_MASK) | RJ_ARITH_SEMANTICS_LOOSE;
    }
  }
  return call(flags, exec_mask, mode, d, a, trapsts);
}

} // namespace

// Evaluate the inherited FP16 rcp model for active lanes and preserve other output bits.
int rj_arith_v_rcp_f16_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return inherited_half_math<true>(rj_arith_v_rcp_f16_w32, flags, exec_mask, instruction_flags, d,
                                   a, trapsts);
}

// Evaluate the inherited FP16 rcp model for active lanes and preserve other output bits.
int rj_arith_v_rcp_f16_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return inherited_half_math<true>(rj_arith_v_rcp_f16_w64, flags, exec_mask, instruction_flags, d,
                                   a, trapsts);
}

// Evaluate the inherited FP16 rcp model for active lanes and preserve other output bits.
int rj_arith_v_rcp_f16_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  return inherited_half_math(rj_arith_v_rcp_f16_w32_gfx12, flags, exec_mask, instruction_flags, d,
                             a, excp_flag_user);
}

// Evaluate the inherited FP16 rsq model for active lanes and preserve other output bits.
int rj_arith_v_rsq_f16_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return inherited_half_math<true>(rj_arith_v_rsq_f16_w32, flags, exec_mask, instruction_flags, d,
                                   a, trapsts);
}

// Evaluate the inherited FP16 rsq model for active lanes and preserve other output bits.
int rj_arith_v_rsq_f16_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return inherited_half_math<true>(rj_arith_v_rsq_f16_w64, flags, exec_mask, instruction_flags, d,
                                   a, trapsts);
}

// Evaluate the inherited FP16 rsq model for active lanes and preserve other output bits.
int rj_arith_v_rsq_f16_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  return inherited_half_math(rj_arith_v_rsq_f16_w32_gfx12, flags, exec_mask, instruction_flags, d,
                             a, excp_flag_user);
}

// Evaluate the inherited FP16 sqrt model for active lanes and preserve other output bits.
int rj_arith_v_sqrt_f16_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *trapsts) {
  return inherited_half_math<true>(rj_arith_v_sqrt_f16_w32, flags, exec_mask, instruction_flags, d,
                                   a, trapsts);
}

// Evaluate the inherited FP16 sqrt model for active lanes and preserve other output bits.
int rj_arith_v_sqrt_f16_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *trapsts) {
  return inherited_half_math<true>(rj_arith_v_sqrt_f16_w64, flags, exec_mask, instruction_flags, d,
                                   a, trapsts);
}

// Evaluate the inherited FP16 sqrt model for active lanes and preserve other output bits.
int rj_arith_v_sqrt_f16_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *excp_flag_user) {
  return inherited_half_math(rj_arith_v_sqrt_f16_w32_gfx12, flags, exec_mask, instruction_flags, d,
                             a, excp_flag_user);
}

// Evaluate the inherited FP16 exp model for active lanes and preserve other output bits.
int rj_arith_v_exp_f16_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return inherited_half_math<true>(rj_arith_v_exp_f16_w32, flags, exec_mask, instruction_flags, d,
                                   a, trapsts);
}

// Evaluate the inherited FP16 exp model for active lanes and preserve other output bits.
int rj_arith_v_exp_f16_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return inherited_half_math<true>(rj_arith_v_exp_f16_w64, flags, exec_mask, instruction_flags, d,
                                   a, trapsts);
}

// Evaluate the inherited FP16 exp model for active lanes and preserve other output bits.
int rj_arith_v_exp_f16_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  return inherited_half_math(rj_arith_v_exp_f16_w32_gfx12, flags, exec_mask, instruction_flags, d,
                             a, excp_flag_user);
}

// Evaluate the inherited FP16 log model for active lanes and preserve other output bits.
int rj_arith_v_log_f16_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return inherited_half_math<true>(rj_arith_v_log_f16_w32, flags, exec_mask, instruction_flags, d,
                                   a, trapsts);
}

// Evaluate the inherited FP16 log model for active lanes and preserve other output bits.
int rj_arith_v_log_f16_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return inherited_half_math<true>(rj_arith_v_log_f16_w64, flags, exec_mask, instruction_flags, d,
                                   a, trapsts);
}

// Evaluate the inherited FP16 log model for active lanes and preserve other output bits.
int rj_arith_v_log_f16_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  return inherited_half_math(rj_arith_v_log_f16_w32_gfx12, flags, exec_mask, instruction_flags, d,
                             a, excp_flag_user);
}

// Evaluate the inherited FP16 sin model for active lanes and preserve other output bits.
int rj_arith_v_sin_f16_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return inherited_half_math<true>(rj_arith_v_sin_f16_w32, flags, exec_mask, instruction_flags, d,
                                   a, trapsts);
}

// Evaluate the inherited FP16 sin model for active lanes and preserve other output bits.
int rj_arith_v_sin_f16_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return inherited_half_math<true>(rj_arith_v_sin_f16_w64, flags, exec_mask, instruction_flags, d,
                                   a, trapsts);
}

// Evaluate the inherited FP16 sin model for active lanes and preserve other output bits.
int rj_arith_v_sin_f16_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  return inherited_half_math(rj_arith_v_sin_f16_w32_gfx12, flags, exec_mask, instruction_flags, d,
                             a, excp_flag_user);
}

// Evaluate the inherited FP16 cos model for active lanes and preserve other output bits.
int rj_arith_v_cos_f16_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return inherited_half_math<true>(rj_arith_v_cos_f16_w32, flags, exec_mask, instruction_flags, d,
                                   a, trapsts);
}

// Evaluate the inherited FP16 cos model for active lanes and preserve other output bits.
int rj_arith_v_cos_f16_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return inherited_half_math<true>(rj_arith_v_cos_f16_w64, flags, exec_mask, instruction_flags, d,
                                   a, trapsts);
}

// Evaluate the inherited FP16 cos model for active lanes and preserve other output bits.
int rj_arith_v_cos_f16_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  return inherited_half_math(rj_arith_v_cos_f16_w32_gfx12, flags, exec_mask, instruction_flags, d,
                             a, excp_flag_user);
}

// MI350X (gfx950) exhaustively validated FP16 SFU entry points.
int rj_arith_v_rcp_f16_w32_gfx950(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return gfx950_half_math(rj_arith_v_rcp_f16_w32, flags, exec_mask, instruction_flags, d, a,
                          trapsts);
}

int rj_arith_v_rcp_f16_w64_gfx950(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return gfx950_half_math(rj_arith_v_rcp_f16_w64, flags, exec_mask, instruction_flags, d, a,
                          trapsts);
}

int rj_arith_v_rsq_f16_w32_gfx950(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return gfx950_half_math(rj_arith_v_rsq_f16_w32, flags, exec_mask, instruction_flags, d, a,
                          trapsts);
}

int rj_arith_v_rsq_f16_w64_gfx950(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return gfx950_half_math(rj_arith_v_rsq_f16_w64, flags, exec_mask, instruction_flags, d, a,
                          trapsts);
}

int rj_arith_v_sqrt_f16_w32_gfx950(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *trapsts) {
  return gfx950_half_math(rj_arith_v_sqrt_f16_w32, flags, exec_mask, instruction_flags, d, a,
                          trapsts);
}

int rj_arith_v_sqrt_f16_w64_gfx950(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *trapsts) {
  return gfx950_half_math(rj_arith_v_sqrt_f16_w64, flags, exec_mask, instruction_flags, d, a,
                          trapsts);
}

int rj_arith_v_exp_f16_w32_gfx950(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return gfx950_half_math(rj_arith_v_exp_f16_w32, flags, exec_mask, instruction_flags, d, a,
                          trapsts);
}

int rj_arith_v_exp_f16_w64_gfx950(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return gfx950_half_math(rj_arith_v_exp_f16_w64, flags, exec_mask, instruction_flags, d, a,
                          trapsts);
}

int rj_arith_v_log_f16_w32_gfx950(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return gfx950_half_math(rj_arith_v_log_f16_w32, flags, exec_mask, instruction_flags, d, a,
                          trapsts);
}

int rj_arith_v_log_f16_w64_gfx950(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return gfx950_half_math(rj_arith_v_log_f16_w64, flags, exec_mask, instruction_flags, d, a,
                          trapsts);
}

int rj_arith_v_sin_f16_w32_gfx950(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return gfx950_half_math(rj_arith_v_sin_f16_w32, flags, exec_mask, instruction_flags, d, a,
                          trapsts);
}

int rj_arith_v_sin_f16_w64_gfx950(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return gfx950_half_math(rj_arith_v_sin_f16_w64, flags, exec_mask, instruction_flags, d, a,
                          trapsts);
}

int rj_arith_v_cos_f16_w32_gfx950(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return gfx950_half_math(rj_arith_v_cos_f16_w32, flags, exec_mask, instruction_flags, d, a,
                          trapsts);
}

int rj_arith_v_cos_f16_w64_gfx950(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return gfx950_half_math(rj_arith_v_cos_f16_w64, flags, exec_mask, instruction_flags, d, a,
                          trapsts);
}
