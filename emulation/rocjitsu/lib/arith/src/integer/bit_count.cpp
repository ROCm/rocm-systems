// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Population count, bit reversal, and first/last significant-bit searches.
// Owns public entry points and their validation/dispatch policy.

#include "integer/bit_count.h"
#include "common/bits.h"
#include "common/internal.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <stdint.h>

namespace {

// Low-bit masks for widths 0 through 32. A table also avoids Clang's SSE2
// vectorization of variable shifts through float-to-integer conversions,
// which can spuriously raise host FP exceptions for out-of-range counts.
static const uint32_t prefix_masks[] = {
    0x0u,       0x1u,        0x3u,        0x7u,        0xfu,       0x1fu,      0x3fu,
    0x7fu,      0xffu,       0x1ffu,      0x3ffu,      0x7ffu,     0xfffu,     0x1fffu,
    0x3fffu,    0x7fffu,     0xffffu,     0x1ffffu,    0x3ffffu,   0x7ffffu,   0xfffffu,
    0x1fffffu,  0x3fffffu,   0x7fffffu,   0xffffffu,   0x1ffffffu, 0x3ffffffu, 0x7ffffffu,
    0xfffffffu, 0x1fffffffu, 0x3fffffffu, 0x7fffffffu, 0xffffffffu};

// Count or reorder bits.
template <rocjitsu::arith::BitCount Op> uint32_t evaluate(uint32_t a, uint32_t b, int lane) {
  if constexpr (Op == rocjitsu::arith::BitCount::Sign) {
    a ^= 0u - (a >> 31);
  }
  if constexpr (Op == rocjitsu::arith::BitCount::Leading || Op == rocjitsu::arith::BitCount::Sign ||
                Op == rocjitsu::arith::BitCount::Trailing) {
    return rocjitsu::arith::bit_count_zero<Op == rocjitsu::arith::BitCount::Trailing>(a);
  }
  if constexpr (Op == rocjitsu::arith::BitCount::MaskedLow) {
    a &= prefix_masks[lane < 32 ? lane : 32];
  }
  if constexpr (Op == rocjitsu::arith::BitCount::MaskedHigh) {
    a &= prefix_masks[lane < 32 ? 0 : lane - 32];
  }
  return rocjitsu::arith::bit_population(a) + b;
}

// Count or reorder bits for the selected lanes. Reject invalid controls before changing
// outputs.
template <rocjitsu::arith::BitCount Op, int Lanes>
int bit_count(uint64_t flags, rocjitsu::arith::ExecMask<Lanes> exec_mask, uint64_t mode,
              uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b) {
  if (mode >> 32) {
    if constexpr (Lanes == 32) {
      return rocjitsu::arith::execute_dpp(
          flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
            return bit_count<Op, Lanes>(flags, exec_mask, uint32_t(mode), d, source, b);
          });
    } else {
      if (!(mode & (RJ_ARITH_DPP8 | RJ_ARITH_DPP16)) || !rocjitsu::arith::valid_dpp(mode)) {
        return RJ_ARITH_ERROR_INVALID_FLAGS;
      }
      if (int error = bit_count<Op, Lanes>(flags, 0, uint32_t(mode), nullptr, nullptr, nullptr)) {
        return error;
      }
      if (!exec_mask) {
        return RJ_ARITH_SUCCESS;
      }

      // Route both halves before any stores. The counting kernel must still see
      // physical lanes 0..63: routing changes A, not the lane's prefix mask.
      uint32_t permuted[64];
      uint64_t filtered = 0;
      for (int half = 0; half < 2; ++half) {
        uint32_t half_exec = uint32_t(exec_mask >> (half * 32));
        if (mode & RJ_ARITH_DPP8) {
          rocjitsu::arith::dpp8_source(flags, half_exec, mode, permuted + half * 32,
                                       a[0] + half * 32);
        } else {
          uint64_t rows = (mode & RJ_ARITH_DPP_ROW_MASK) >> RJ_ARITH_DPP_ROW_SHIFT;
          uint64_t half_mode = (mode & ~RJ_ARITH_DPP_ROW_MASK) |
                               (((rows >> (half * 2)) & 3) << RJ_ARITH_DPP_ROW_SHIFT);
          half_exec = rocjitsu::arith::dpp16_source(flags, half_exec, half_mode,
                                                    permuted + half * 32, a[0] + half * 32);
        }
        filtered |= uint64_t(half_exec) << (half * 32);
      }
      const uint32_t *source = permuted;
      return bit_count<Op, Lanes>(flags, filtered, uint32_t(mode), d, &source, b);
    }
  }

  if (int error = rocjitsu::arith::validate(flags, mode & ~RJ_ARITH_ALU_CLAMP, true)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  const uint32_t *bp = nullptr;
  if constexpr (Op == rocjitsu::arith::BitCount::Population ||
                Op == rocjitsu::arith::BitCount::MaskedLow ||
                Op == rocjitsu::arith::BitCount::MaskedHigh) {
    bp = b[0];
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V4) {
    rocjitsu::arith::bit_count_x86_64_v4<Op, Lanes>(exec_mask, d[0], a[0], bp);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  // Wave32 MBCNT_HI only copies B; AVX2 showed no substantial gain there.
  if constexpr (Op != rocjitsu::arith::BitCount::MaskedHigh || Lanes != 32) {
    if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
      rocjitsu::arith::bit_count_x86_64_v3<Op, Lanes>(exec_mask, d[0], a[0], bp);
      return RJ_ARITH_SUCCESS;
    }
  }
#endif
  uint32_t result[Lanes];
  for (int lane = 0; lane < Lanes; ++lane) {
    result[lane] = evaluate<Op>(a[0][lane], bp ? bp[lane] : 0, unsigned(lane));
  }
  for (int lane = 0; lane < Lanes; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Count leading zeros for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_clz_i32_u32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return bit_count<rocjitsu::arith::BitCount::Leading, 32>(flags, exec_mask, instruction_flags, d,
                                                           a, nullptr);
}

// Count trailing zeros for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_ctz_i32_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return bit_count<rocjitsu::arith::BitCount::Trailing, 32>(flags, exec_mask, instruction_flags, d,
                                                            a, nullptr);
}

// Count redundant sign bits for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cls_i32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return bit_count<rocjitsu::arith::BitCount::Sign, 32>(flags, exec_mask, instruction_flags, d, a,
                                                        nullptr);
}

// Count set bits and add the accumulator for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_bcnt_u32_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return bit_count<rocjitsu::arith::BitCount::Population, 32>(flags, exec_mask, instruction_flags,
                                                              d, a, b);
}

// Execute MBCNT_LO_U32_B32 for the selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_mbcnt_lo_u32_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return bit_count<rocjitsu::arith::BitCount::MaskedLow, 32>(flags, exec_mask, instruction_flags, d,
                                                             a, b);
}

// Execute MBCNT_HI_U32_B32 for the selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_mbcnt_hi_u32_b32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return bit_count<rocjitsu::arith::BitCount::MaskedHigh, 32>(flags, exec_mask, instruction_flags,
                                                              d, a, b);
}

// Execute MBCNT_LO_U32_B32 for the selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_mbcnt_lo_u32_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return bit_count<rocjitsu::arith::BitCount::MaskedLow, 64>(flags, exec_mask, instruction_flags, d,
                                                             a, b);
}

// Execute MBCNT_HI_U32_B32 for the selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_mbcnt_hi_u32_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return bit_count<rocjitsu::arith::BitCount::MaskedHigh, 64>(flags, exec_mask, instruction_flags,
                                                              d, a, b);
}

// Count leading zeros for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_clz_i32_u32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return bit_count<rocjitsu::arith::BitCount::Leading, 64>(flags, exec_mask, instruction_flags, d,
                                                           a, nullptr);
}

// Count trailing zeros for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_ctz_i32_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return bit_count<rocjitsu::arith::BitCount::Trailing, 64>(flags, exec_mask, instruction_flags, d,
                                                            a, nullptr);
}

// Count redundant sign bits for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_cls_i32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return bit_count<rocjitsu::arith::BitCount::Sign, 64>(flags, exec_mask, instruction_flags, d, a,
                                                        nullptr);
}

// Count set bits and add the accumulator for selected lanes.
// Return errors before changing outputs. Inactive destination lanes are preserved.
int rj_arith_v_bcnt_u32_b32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return bit_count<rocjitsu::arith::BitCount::Population, 64>(flags, exec_mask, instruction_flags,
                                                              d, a, b);
}
