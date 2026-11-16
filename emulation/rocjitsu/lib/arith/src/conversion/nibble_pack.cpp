// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Saturate signed/unsigned bytes into four nibbles in a selected destination half.
#include "conversion/nibble_pack.h"
#include "common/validation.h"
#include "register/dpp.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_conversion.h"
namespace {
// Validate controls before reads; both halves of an aliased source remain available.
template <int Lanes, bool Signed>
int run(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
        const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (mode >> 32) {
    if constexpr (Lanes == 32) {
      return rocjitsu::arith::execute_dpp(
          flags, uint32_t(exec_mask), mode, a, [&](uint32_t active, const uint32_t *const *source) {
            return run<32, Signed>(flags, active, uint32_t(mode), d, source);
          });
    } else {
      // Validate before either half writes; DPP routes within 16-lane rows.
      if (int error = run<32, Signed>(flags, 0, mode, nullptr, nullptr)) {
        return error;
      }
      if (!exec_mask) {
        return RJ_ARITH_SUCCESS;
      }
      for (int half = 0; half < 2; ++half) {
        uint32_t *dest = d[0] + 32 * half;
        const uint32_t *source = a[0] + 32 * half;
        const uint64_t controls = rocjitsu::arith::dpp_half_wave_instruction_flags(mode, half);
        if (int error = run<32, Signed>(flags, uint32_t(exec_mask >> (32 * half)), controls, &dest,
                                        &source)) {
          return error;
        }
      }
      return RJ_ARITH_SUCCESS;
    }
  }
  if (int error =
          rocjitsu::arith::validate(flags, mode & ~(RJ_ARITH_ALU_HIGH_D | RJ_ARITH_ALU_CLAMP))) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    rocjitsu::arith::nibble_pack_x86_64_v4(Signed, Lanes, exec_mask, uint32_t(mode), d[0], a[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    rocjitsu::arith::nibble_pack_x86_64_v3(Signed, Lanes, exec_mask, uint32_t(mode), d[0], a[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
  int shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  for (int lane = 0; lane < Lanes; ++lane) {
    uint32_t raw = a[0][lane], packed = 0;
    for (int element = 0; element < 4; ++element) {
      int value = (raw >> (8 * element)) & 255;
      if constexpr (Signed) {
        value = (value ^ 128) - 128;
        value = value < -8 ? -8 : value > 7 ? 7 : value;
      } else {
        value = value > 15 ? 15 : value;
      }
      packed |= (uint32_t(value) & 15) << (element * 4);
    }
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = (d[0][lane] & ~(0xffffU << shift)) | (packed << shift);
    }
  }
  return RJ_ARITH_SUCCESS;
}
} // namespace

// Pack saturated bytes into the selected destination halfword.
int rj_arith_v_sat_pk4_i4_i8_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                 uint32_t *const *d, const uint32_t *const *a) {
  return run<32, true>(flags, exec_mask, mode, d, a);
}

// Pack saturated bytes into the selected destination halfword.
int rj_arith_v_sat_pk4_u4_u8_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                 uint32_t *const *d, const uint32_t *const *a) {
  return run<32, false>(flags, exec_mask, mode, d, a);
}

// Pack saturated bytes into the selected destination halfword.
int rj_arith_v_sat_pk4_i4_i8_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                 uint32_t *const *d, const uint32_t *const *a) {
  return run<64, true>(flags, exec_mask, mode, d, a);
}

// Pack saturated bytes into the selected destination halfword.
int rj_arith_v_sat_pk4_u4_u8_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                 uint32_t *const *d, const uint32_t *const *a) {
  return run<64, false>(flags, exec_mask, mode, d, a);
}
