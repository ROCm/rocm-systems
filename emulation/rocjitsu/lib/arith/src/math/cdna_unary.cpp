// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP32 math reuses the models already used by CDNA and older Radeon handlers.
// Compatibility retains those models and their status policies. Their validation on
// RDNA hardware does not establish empirical bit-exactness on these architectures.

#include "common/sdwa_dispatch.h"
#include "common/validation.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_math.h"

#include <stdint.h>

namespace {

// Forward the inherited model without advertising new hardware-validated semantics.
// Reject strict requests before any destination or FP-status access.
template <typename Call, typename Mask>
int inherited_math(Call call, uint64_t flags, Mask exec_mask, uint64_t instruction_flags,
                   uint32_t *const *d, const uint32_t *const *a, uint32_t *state) {
  const uint64_t semantics = flags & RJ_ARITH_SEMANTICS_MASK;
  // The selected implementation resolves Compatibility to its own numeric/status policy.
  if (semantics != RJ_ARITH_SEMANTICS_COMPATIBILITY) {
    if (int error = rocjitsu::arith::validate_semantics(flags)) {
      return error;
    }
  }
  if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS ||
      semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  return call(flags, exec_mask, instruction_flags, d, a, state);
}

// Native SDWA arithmetic uses nearest rounding; output follows its separate inherited policy.
// The underlying call owns qualification and status before output scaling and clamp.
template <rocjitsu::arith::SdwaOutput Output, typename Call, typename Mask>
int sdwa_math(Call call, uint64_t flags, Mask exec_mask, uint64_t instruction_flags,
              uint32_t *const *d, const uint32_t *const *a, uint32_t *state) {
  const uint64_t core_flags =
      instruction_flags & RJ_ARITH_ALU_SDWA ? flags & ~RJ_ARITH_FP_ROUND_MASK : flags;
  return rocjitsu::arith::with_sdwa_output<Output>(
      flags, exec_mask, instruction_flags, d, [&](uint64_t core_mode) {
        return inherited_math(call, core_flags, exec_mask, core_mode, d, a, state);
      });
}

// Forward MI350X-calibrated FP32 SFU entry points. SQRT retains Compatibility INVALID causes
// while rejecting explicit mode 3; operations without an active-OMOD capture reject effective
// OMOD in mode 2 and fall back in mode 1.
template <bool ActiveOmodCaptured, bool DenormGatesOmod, bool RejectExplicitStatus, typename Call,
          typename Mask>
int gfx950_math(Call call, uint64_t flags, Mask exec_mask, uint64_t instruction_flags,
                uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  if (instruction_flags & RJ_ARITH_ALU_SDWA) {
    constexpr auto sdwa_out = DenormGatesOmod ? rocjitsu::arith::SdwaOutput::F32Mode
                                              : rocjitsu::arith::SdwaOutput::F32Nearest;
    return sdwa_math<sdwa_out>(call, flags, exec_mask, instruction_flags, d, a, trapsts);
  }
  const uint64_t semantics = flags & RJ_ARITH_SEMANTICS_MASK;
  if constexpr (RejectExplicitStatus) {
    if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL) {
      const uint64_t check_flags =
          (flags & ~RJ_ARITH_SEMANTICS_MASK) | RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS;
      if (int error = call(check_flags, 0, instruction_flags, nullptr, nullptr, nullptr)) {
        return error;
      }
      return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
    }
  }
  if constexpr (!ActiveOmodCaptured) {
    const bool active_omod = (instruction_flags & RJ_ARITH_ALU_OMOD_MASK) &&
                             !(flags & RJ_ARITH_FP_IEEE_MODE) &&
                             (!DenormGatesOmod || (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS));
    if (active_omod) {
      if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
        if (int error = call(flags, 0, instruction_flags, nullptr, nullptr, nullptr)) {
          return error;
        }
        return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
      }
      if (semantics == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE) {
        flags = (flags & ~RJ_ARITH_SEMANTICS_MASK) | RJ_ARITH_SEMANTICS_LOOSE;
      }
    }
  }
  return call(flags, exec_mask, instruction_flags, d, a, trapsts);
}

} // namespace

// Evaluate the inherited rcp model for the selected lanes, preserving inactive outputs.
int rj_arith_v_rcp_f32_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return sdwa_math<rocjitsu::arith::SdwaOutput::F32Nearest>(
      rj_arith_v_rcp_f32_w32, flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Evaluate the inherited rcp model for the selected lanes, preserving inactive outputs.
int rj_arith_v_rcp_f32_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return sdwa_math<rocjitsu::arith::SdwaOutput::F32Nearest>(
      rj_arith_v_rcp_f32_w64, flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Evaluate the inherited rcp model for the selected lanes, preserving inactive outputs.
int rj_arith_v_rcp_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  return inherited_math(rj_arith_v_rcp_f32_w32_gfx12, flags, exec_mask, instruction_flags, d, a,
                        excp_flag_user);
}

// Evaluate the inherited rsq model for the selected lanes, preserving inactive outputs.
int rj_arith_v_rsq_f32_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return sdwa_math<rocjitsu::arith::SdwaOutput::F32Nearest>(
      rj_arith_v_rsq_f32_w32, flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Evaluate the inherited rsq model for the selected lanes, preserving inactive outputs.
int rj_arith_v_rsq_f32_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return sdwa_math<rocjitsu::arith::SdwaOutput::F32Nearest>(
      rj_arith_v_rsq_f32_w64, flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Evaluate the inherited rsq model for the selected lanes, preserving inactive outputs.
int rj_arith_v_rsq_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  return inherited_math(rj_arith_v_rsq_f32_w32_gfx12, flags, exec_mask, instruction_flags, d, a,
                        excp_flag_user);
}

// Evaluate the inherited sqrt model for the selected lanes, preserving inactive outputs.
int rj_arith_v_sqrt_f32_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *trapsts) {
  return sdwa_math<rocjitsu::arith::SdwaOutput::F32Nearest>(
      rj_arith_v_sqrt_f32_w32, flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Evaluate the inherited sqrt model for the selected lanes, preserving inactive outputs.
int rj_arith_v_sqrt_f32_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *trapsts) {
  return sdwa_math<rocjitsu::arith::SdwaOutput::F32Nearest>(
      rj_arith_v_sqrt_f32_w64, flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Evaluate the inherited sqrt model for the selected lanes, preserving inactive outputs.
int rj_arith_v_sqrt_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    uint32_t *excp_flag_user) {
  return inherited_math(rj_arith_v_sqrt_f32_w32_gfx12, flags, exec_mask, instruction_flags, d, a,
                        excp_flag_user);
}

// Evaluate the inherited exp model for the selected lanes, preserving inactive outputs.
int rj_arith_v_exp_f32_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return sdwa_math<rocjitsu::arith::SdwaOutput::F32Nearest>(
      rj_arith_v_exp_f32_w32, flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Evaluate the inherited exp model for the selected lanes, preserving inactive outputs.
int rj_arith_v_exp_f32_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return sdwa_math<rocjitsu::arith::SdwaOutput::F32Nearest>(
      rj_arith_v_exp_f32_w64, flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Evaluate the inherited exp model for the selected lanes, preserving inactive outputs.
int rj_arith_v_exp_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  return inherited_math(rj_arith_v_exp_f32_w32_gfx12, flags, exec_mask, instruction_flags, d, a,
                        excp_flag_user);
}

// Evaluate the inherited log model for the selected lanes, preserving inactive outputs.
int rj_arith_v_log_f32_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return sdwa_math<rocjitsu::arith::SdwaOutput::F32Nearest>(
      rj_arith_v_log_f32_w32, flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Evaluate the inherited log model for the selected lanes, preserving inactive outputs.
int rj_arith_v_log_f32_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return sdwa_math<rocjitsu::arith::SdwaOutput::F32Nearest>(
      rj_arith_v_log_f32_w64, flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Evaluate the inherited log model for the selected lanes, preserving inactive outputs.
int rj_arith_v_log_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  return inherited_math(rj_arith_v_log_f32_w32_gfx12, flags, exec_mask, instruction_flags, d, a,
                        excp_flag_user);
}

// Evaluate the inherited sin model for the selected lanes, preserving inactive outputs.
int rj_arith_v_sin_f32_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return sdwa_math<rocjitsu::arith::SdwaOutput::F32Mode>(rj_arith_v_sin_f32_w32, flags, exec_mask,
                                                         instruction_flags, d, a, trapsts);
}

// Evaluate the inherited sin model for the selected lanes, preserving inactive outputs.
int rj_arith_v_sin_f32_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return sdwa_math<rocjitsu::arith::SdwaOutput::F32Mode>(rj_arith_v_sin_f32_w64, flags, exec_mask,
                                                         instruction_flags, d, a, trapsts);
}

// Evaluate the inherited sin model for the selected lanes, preserving inactive outputs.
int rj_arith_v_sin_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  return inherited_math(rj_arith_v_sin_f32_w32_gfx12, flags, exec_mask, instruction_flags, d, a,
                        excp_flag_user);
}

// Evaluate the inherited cos model for the selected lanes, preserving inactive outputs.
int rj_arith_v_cos_f32_w32_gfx942(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return sdwa_math<rocjitsu::arith::SdwaOutput::F32Nearest>(
      rj_arith_v_cos_f32_w32, flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Evaluate the inherited cos model for the selected lanes, preserving inactive outputs.
int rj_arith_v_cos_f32_w64_gfx942(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return sdwa_math<rocjitsu::arith::SdwaOutput::F32Nearest>(
      rj_arith_v_cos_f32_w64, flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Evaluate the inherited cos model for the selected lanes, preserving inactive outputs.
int rj_arith_v_cos_f32_w32_gfx1250(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *excp_flag_user) {
  return inherited_math(rj_arith_v_cos_f32_w32_gfx12, flags, exec_mask, instruction_flags, d, a,
                        excp_flag_user);
}

// Legacy encoding retains the same model and policy as ordinary CDNA exp.
int rj_arith_v_exp_legacy_f32_w32_gfx942(uint64_t flags, uint32_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, uint32_t *trapsts) {
  return rj_arith_v_exp_f32_w32_gfx942(flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Legacy encoding retains the same model and policy as ordinary CDNA exp.
int rj_arith_v_exp_legacy_f32_w64_gfx942(uint64_t flags, uint64_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, uint32_t *trapsts) {
  return rj_arith_v_exp_f32_w64_gfx942(flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Legacy encoding retains the same model and policy as ordinary CDNA log.
int rj_arith_v_log_legacy_f32_w32_gfx942(uint64_t flags, uint32_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, uint32_t *trapsts) {
  return rj_arith_v_log_f32_w32_gfx942(flags, exec_mask, instruction_flags, d, a, trapsts);
}

// Legacy encoding retains the same model and policy as ordinary CDNA log.
int rj_arith_v_log_legacy_f32_w64_gfx942(uint64_t flags, uint64_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, uint32_t *trapsts) {
  return rj_arith_v_log_f32_w64_gfx942(flags, exec_mask, instruction_flags, d, a, trapsts);
}

// MI350X (gfx950) calibrated FP32 SFU entry points.
int rj_arith_v_rcp_f32_w32_gfx950(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return gfx950_math<true, false, false>(rj_arith_v_rcp_f32_w32, flags, exec_mask,
                                         instruction_flags, d, a, trapsts);
}

int rj_arith_v_rcp_f32_w64_gfx950(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return gfx950_math<true, false, false>(rj_arith_v_rcp_f32_w64, flags, exec_mask,
                                         instruction_flags, d, a, trapsts);
}

int rj_arith_v_rsq_f32_w32_gfx950(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return gfx950_math<false, false, false>(rj_arith_v_rsq_f32_w32, flags, exec_mask,
                                          instruction_flags, d, a, trapsts);
}

int rj_arith_v_rsq_f32_w64_gfx950(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return gfx950_math<false, false, false>(rj_arith_v_rsq_f32_w64, flags, exec_mask,
                                          instruction_flags, d, a, trapsts);
}

int rj_arith_v_sqrt_f32_w32_gfx950(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *trapsts) {
  return gfx950_math<false, false, true>(rj_arith_v_sqrt_f32_w32, flags, exec_mask,
                                         instruction_flags, d, a, trapsts);
}

int rj_arith_v_sqrt_f32_w64_gfx950(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   uint32_t *trapsts) {
  return gfx950_math<false, false, true>(rj_arith_v_sqrt_f32_w64, flags, exec_mask,
                                         instruction_flags, d, a, trapsts);
}

int rj_arith_v_exp_f32_w32_gfx950(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return gfx950_math<true, false, false>(rj_arith_v_exp_f32_w32, flags, exec_mask,
                                         instruction_flags, d, a, trapsts);
}

int rj_arith_v_exp_f32_w64_gfx950(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return gfx950_math<true, false, false>(rj_arith_v_exp_f32_w64, flags, exec_mask,
                                         instruction_flags, d, a, trapsts);
}

int rj_arith_v_log_f32_w32_gfx950(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return gfx950_math<false, false, false>(rj_arith_v_log_f32_w32, flags, exec_mask,
                                          instruction_flags, d, a, trapsts);
}

int rj_arith_v_log_f32_w64_gfx950(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return gfx950_math<false, false, false>(rj_arith_v_log_f32_w64, flags, exec_mask,
                                          instruction_flags, d, a, trapsts);
}

int rj_arith_v_sin_f32_w32_gfx950(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return gfx950_math<false, true, false>(rj_arith_v_sin_f32_w32, flags, exec_mask,
                                         instruction_flags, d, a, trapsts);
}

int rj_arith_v_sin_f32_w64_gfx950(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return gfx950_math<false, true, false>(rj_arith_v_sin_f32_w64, flags, exec_mask,
                                         instruction_flags, d, a, trapsts);
}

int rj_arith_v_cos_f32_w32_gfx950(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return gfx950_math<false, false, false>(rj_arith_v_cos_f32_w32, flags, exec_mask,
                                          instruction_flags, d, a, trapsts);
}

int rj_arith_v_cos_f32_w64_gfx950(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a, uint32_t *trapsts) {
  return gfx950_math<false, false, false>(rj_arith_v_cos_f32_w64, flags, exec_mask,
                                          instruction_flags, d, a, trapsts);
}
