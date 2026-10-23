// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Saturating INT16-to-U8 packing and selection of two FP16 source halfwords.
// Owns public entry points and their validation/dispatch policy.

#include "conversion/pack.h"
#include "common/internal.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <algorithm>
#include <stdint.h>

namespace {

// Pack saturated bytes or selected FP16 halves for the selected lanes. Reject invalid controls
// before changing outputs.
template <bool Saturate>
int run(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, bool quiet_nan = true) {
  if (mode >> 32) {
    return rocjitsu::arith::execute_dpp(
        flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
          return run<Saturate>(flags, exec_mask, uint32_t(mode), d, source, b, quiet_nan);
        });
  }
  const uint32_t known = Saturate
                             ? RJ_ARITH_ALU_HIGH_D | RJ_ARITH_ALU_CLAMP
                             : RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B | RJ_ARITH_ALU_ABS_A |
                                   RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B;
  if (int error = rocjitsu::arith::validate(flags, mode & ~known, true,
                                            Saturate ? 0
                                                     : RJ_ARITH_FP_IEEE_MODE |
                                                           RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                                           RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  const uint32_t *bp = nullptr;
  if constexpr (!Saturate) {
    bp = b[0];
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V4) {
    rocjitsu::arith::pack_x86_64_v4<Saturate>(flags, exec_mask, mode, d[0], a[0], bp, quiet_nan);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
    rocjitsu::arith::pack_x86_64_v3<Saturate>(flags, exec_mask, mode, d[0], a[0], bp, quiet_nan);
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    uint32_t x = a[0][lane];
    if constexpr (Saturate) {
      // Clamp a signed low halfword to the unsigned byte range.
      auto convert = [](uint32_t raw) {
        int32_t value = int32_t(raw) - ((raw & 32768) ? 65536 : 0);
        return uint32_t(std::clamp(value, 0, 255));
      };

      // The packed-byte form updates only the selected destination halfword.
      uint32_t packed = convert(x & 65535) | (convert(x >> 16) << 8);
      unsigned shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
      result[lane] = (d[0][lane] & ~(65535u << shift)) | (packed << shift);
    } else {
      x = (x >> (mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0)) & 65535;
      uint32_t y = (bp[lane] >> (mode & RJ_ARITH_ALU_HIGH_B ? 16 : 0)) & 65535;
      if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
        if ((x & 0x7fff) < 0x400) {
          x &= 0x8000;
        }
        if ((y & 0x7fff) < 0x400) {
          y &= 0x8000;
        }
      }
      if (quiet_nan && (x & 32767) > 0x7c00) {
        x |= 0x200;
      }
      if (quiet_nan && (y & 32767) > 0x7c00) {
        y |= 0x200;
      }
      if (mode & RJ_ARITH_ALU_ABS_A) {
        x &= 32767;
      }
      if (mode & RJ_ARITH_ALU_ABS_B) {
        y &= 32767;
      }
      if (mode & RJ_ARITH_ALU_NEG_A) {
        x ^= 32768;
      }
      if (mode & RJ_ARITH_ALU_NEG_B) {
        y ^= 32768;
      }
      result[lane] = x | (y << 16);
    }
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Pack saturated bytes or selected FP16 halves for Wave64. Reject invalid controls before
// changing outputs.
template <bool Saturate>
int run_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
               const uint32_t *const *a, const uint32_t *const *b, bool quiet_nan = true) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = run<Saturate>(flags, 0, mode, nullptr, nullptr, nullptr, quiet_nan)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  for (int half = 0; half < 2; ++half) {
    uint64_t half_mode = mode;
    if (mode & RJ_ARITH_DPP16) {
      // Each half-wave uses only its own two DPP16 rows.
      uint64_t rows = (mode & RJ_ARITH_DPP_ROW_MASK) >> RJ_ARITH_DPP_ROW_SHIFT;
      half_mode =
          (mode & ~RJ_ARITH_DPP_ROW_MASK) | (((rows >> (half * 2)) & 3) << RJ_ARITH_DPP_ROW_SHIFT);
    }
    uint32_t *dest = d[0] + half * 32;
    const uint32_t *source_a = a[0] + half * 32, *source_b = nullptr;
    if constexpr (!Saturate) {
      source_b = b[0] + half * 32;
    }
    if (int error = run<Saturate>(flags, uint32_t(exec_mask >> (half * 32)), half_mode, &dest,
                                  &source_a, &source_b, quiet_nan)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Saturate packed INT16 values to unsigned bytes with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_sat_pk_u8_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<true>(flags, exec_mask, instruction_flags, d, a, nullptr);
}

// Pack the selected source halfwords for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_pack_b32_f16_w32_gfx12(uint64_t flags, uint32_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return run<false>(flags, exec_mask, instruction_flags, d, a, b);
}

// Saturate packed INT16 values to unsigned bytes with packed elements for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_sat_pk_u8_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run_wave64<true>(flags, exec_mask, instruction_flags, d, a, nullptr);
}

// Pack the selected source halfwords for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_pack_b32_f16_w64_gfx12(uint64_t flags, uint64_t exec_mask,
                                      uint64_t instruction_flags, uint32_t *const *d,
                                      const uint32_t *const *a, const uint32_t *const *b,
                                      [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  return run_wave64<false>(flags, exec_mask, instruction_flags, d, a, b);
}

// Pack the selected source halfwords for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_pack_b32_f16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (int error =
          rocjitsu::arith::validate(flags, 0, false,
                                    RJ_ARITH_FP_IEEE_MODE | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                        RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    return error;
  }
  return run<false>(flags, exec_mask, instruction_flags, d, a, b,
                    bool(flags & RJ_ARITH_FP_IEEE_MODE));
}

// Pack the selected source halfwords for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_pack_b32_f16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, [[maybe_unused]] uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (int error =
          rocjitsu::arith::validate(flags, 0, false,
                                    RJ_ARITH_FP_IEEE_MODE | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                        RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    return error;
  }
  return run_wave64<false>(flags, exec_mask, instruction_flags, d, a, b,
                           bool(flags & RJ_ARITH_FP_IEEE_MODE));
}
