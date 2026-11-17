// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Wide low-precision widening: gfx950 PK32 uses an FP32 exponent and preserves D
// for special scales; gfx1250 PK8/PK16 selects an E8M0 byte, including its NaN.
// Both share finite decoding, output rounding, and alias-safe SIMD backends.

#include "common/float_encoding.h"
#include "common/validation.h"
#include "conversion/scaled_widen_impl.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_conversion.h"
#include <stdint.h>

namespace {
using namespace rocjitsu::arith;

// Decode one narrow encoding, preserving signed zero and the OCP NaN sign.
template <ScaledSource Src> float decode(uint32_t code) {
  if constexpr (Src == ScaledSource::Fp4) {
    constexpr float values[] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
    return as_float(as_bits(values[code & 7]) | ((code & 8) << 28));
  } else if constexpr (Src == ScaledSource::Fp8 || Src == ScaledSource::Bf8) {
    return fp8_to_float<Src == ScaledSource::Bf8>(uint8_t(code));
  }
  constexpr bool Bf6 = Src == ScaledSource::Bf6;
  constexpr int fraction = Bf6 ? 2 : 3, bias = Bf6 ? 3 : 1;
  uint32_t sign = (code & 32) << 26;
  uint32_t exponent = (code & 31) >> fraction;
  uint32_t mantissa = code & ((1U << fraction) - 1);
  uint32_t magnitude = exponent ? ((exponent + 127 - bias) << 23) | (mantissa << (23 - fraction))
                                : as_bits(float(mantissa) * (Bf6 ? 0.0625f : 0.125f));
  return as_float(sign | magnitude);
}

// Expand active lanes, staging results to permit arbitrary register overlap.
// ByteScale selects gfx1250 PK8/PK16 E8M0; otherwise use gfx950 PK32/FP32 scales.
template <int Lanes, ScaledSource Src, ScaledDestination Dst, bool ByteScale = false>
int convert(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
            const uint32_t *const *a, const uint32_t *const *b) {
  flags = resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  constexpr uint64_t allowed_mode = ByteScale ? RJ_ARITH_CVT_BYTE_3 : 0;
  if (int error = validate(flags, mode & ~allowed_mode)) {
    return error;
  }
  if ((flags & RJ_ARITH_SEMANTICS_MASK) >= RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  constexpr int bits = Src == ScaledSource::Fp4                                 ? 4
                       : (Src == ScaledSource::Fp6 || Src == ScaledSource::Bf6) ? 6
                                                                                : 8;
  constexpr int elements = ByteScale ? (bits == 6 ? 16 : 8) : 32;
  int scale_shift = ByteScale ? int((mode >> 16) & 3) * 8 : 23;
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    wide_scaled_widen_x86_64_v4<Src, ByteScale>(Lanes, exec_mask, Dst, flags & RJ_ARITH_FP16_OVFL,
                                                scale_shift, d, a, b[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    wide_scaled_widen_x86_64_v3<Src, ByteScale>(Lanes, exec_mask, Dst, flags & RJ_ARITH_FP16_OVFL,
                                                scale_shift, d, a, b[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
  constexpr bool f32 = Dst == ScaledDestination::F32;
  constexpr int words = f32 ? elements : elements / 2;
  uint32_t result[words][Lanes];
  uint64_t write_mask = exec_mask;
  for (int lane = 0; lane < Lanes; ++lane) {
    if (!((exec_mask >> lane) & 1)) {
      continue;
    }
    uint32_t exponent = (b[0][lane] >> scale_shift) & 255;
    if (!ByteScale && exponent == 255) {
      write_mask &= ~(1ULL << lane);
      continue;
    }
    float scale = as_float(exponent == 255 ? 0x7fc00000U : exponent ? exponent << 23 : 0x00400000U);
    for (int element = 0; element < elements; ++element) {
      int bit = bits * element, word = bit / 32, shift = bit % 32;
      uint32_t code = a[word][lane] >> shift;
      if (shift + bits > 32) {
        code |= a[word + 1][lane] << (32 - shift);
      }
      // E8M0 NaN scales have a canonical result even if the input is also NaN.
      // Avoid relying on host operand ordering to choose between two NaN signs.
      float value_f32 =
          exponent == 255 ? as_float(0x7fc00000U) : decode<Src>(code & ((1U << bits) - 1)) * scale;
      uint32_t value = encode_scaled_widen<Dst>(value_f32, flags & RJ_ARITH_FP16_OVFL);
      if constexpr (f32) {
        result[element][lane] = value;
      } else if (element % 2) {
        result[element / 2][lane] |= value << 16;
      } else {
        result[element / 2][lane] = value;
      }
    }
  }
  for (int word = 0; word < words; ++word) {
    for (int lane = 0; lane < Lanes; ++lane) {
      if ((write_mask >> lane) & 1) {
        d[word][lane] = result[word][lane];
      }
    }
  }
  return RJ_ARITH_SUCCESS;
}
} // namespace

// Scale and widen 32 FP6 values per active lane to F32.
int rj_arith_v_cvt_scalef32_pk32_f32_fp6_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                                    uint64_t instruction_flags, uint32_t *const *d,
                                                    const uint32_t *const *a,
                                                    const uint32_t *const *b) {
  return convert<32, ScaledSource::Fp6, ScaledDestination::F32>(flags, exec_mask, instruction_flags,
                                                                d, a, b);
}

// Scale and widen 32 FP6 values per active lane to F32.
int rj_arith_v_cvt_scalef32_pk32_f32_fp6_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                                    uint64_t instruction_flags, uint32_t *const *d,
                                                    const uint32_t *const *a,
                                                    const uint32_t *const *b) {
  return convert<64, ScaledSource::Fp6, ScaledDestination::F32>(flags, exec_mask, instruction_flags,
                                                                d, a, b);
}

// Scale and widen 32 FP6 values per active lane to F16.
int rj_arith_v_cvt_scalef32_pk32_f16_fp6_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                                    uint64_t instruction_flags, uint32_t *const *d,
                                                    const uint32_t *const *a,
                                                    const uint32_t *const *b) {
  return convert<32, ScaledSource::Fp6, ScaledDestination::F16>(flags, exec_mask, instruction_flags,
                                                                d, a, b);
}

// Scale and widen 32 FP6 values per active lane to F16.
int rj_arith_v_cvt_scalef32_pk32_f16_fp6_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                                    uint64_t instruction_flags, uint32_t *const *d,
                                                    const uint32_t *const *a,
                                                    const uint32_t *const *b) {
  return convert<64, ScaledSource::Fp6, ScaledDestination::F16>(flags, exec_mask, instruction_flags,
                                                                d, a, b);
}

// Scale and widen 32 FP6 values per active lane to BF16.
int rj_arith_v_cvt_scalef32_pk32_bf16_fp6_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                                     uint64_t instruction_flags, uint32_t *const *d,
                                                     const uint32_t *const *a,
                                                     const uint32_t *const *b) {
  return convert<32, ScaledSource::Fp6, ScaledDestination::Bf16>(flags, exec_mask,
                                                                 instruction_flags, d, a, b);
}

// Scale and widen 32 FP6 values per active lane to BF16.
int rj_arith_v_cvt_scalef32_pk32_bf16_fp6_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                                     uint64_t instruction_flags, uint32_t *const *d,
                                                     const uint32_t *const *a,
                                                     const uint32_t *const *b) {
  return convert<64, ScaledSource::Fp6, ScaledDestination::Bf16>(flags, exec_mask,
                                                                 instruction_flags, d, a, b);
}

// Scale and widen 32 BF6 values per active lane to F32.
int rj_arith_v_cvt_scalef32_pk32_f32_bf6_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                                    uint64_t instruction_flags, uint32_t *const *d,
                                                    const uint32_t *const *a,
                                                    const uint32_t *const *b) {
  return convert<32, ScaledSource::Bf6, ScaledDestination::F32>(flags, exec_mask, instruction_flags,
                                                                d, a, b);
}

// Scale and widen 32 BF6 values per active lane to F32.
int rj_arith_v_cvt_scalef32_pk32_f32_bf6_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                                    uint64_t instruction_flags, uint32_t *const *d,
                                                    const uint32_t *const *a,
                                                    const uint32_t *const *b) {
  return convert<64, ScaledSource::Bf6, ScaledDestination::F32>(flags, exec_mask, instruction_flags,
                                                                d, a, b);
}

// Scale and widen 32 BF6 values per active lane to F16.
int rj_arith_v_cvt_scalef32_pk32_f16_bf6_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                                    uint64_t instruction_flags, uint32_t *const *d,
                                                    const uint32_t *const *a,
                                                    const uint32_t *const *b) {
  return convert<32, ScaledSource::Bf6, ScaledDestination::F16>(flags, exec_mask, instruction_flags,
                                                                d, a, b);
}

// Scale and widen 32 BF6 values per active lane to F16.
int rj_arith_v_cvt_scalef32_pk32_f16_bf6_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                                    uint64_t instruction_flags, uint32_t *const *d,
                                                    const uint32_t *const *a,
                                                    const uint32_t *const *b) {
  return convert<64, ScaledSource::Bf6, ScaledDestination::F16>(flags, exec_mask, instruction_flags,
                                                                d, a, b);
}

// Scale and widen 32 BF6 values per active lane to BF16.
int rj_arith_v_cvt_scalef32_pk32_bf16_bf6_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                                     uint64_t instruction_flags, uint32_t *const *d,
                                                     const uint32_t *const *a,
                                                     const uint32_t *const *b) {
  return convert<32, ScaledSource::Bf6, ScaledDestination::Bf16>(flags, exec_mask,
                                                                 instruction_flags, d, a, b);
}

// Scale and widen 32 BF6 values per active lane to BF16.
int rj_arith_v_cvt_scalef32_pk32_bf16_bf6_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                                     uint64_t instruction_flags, uint32_t *const *d,
                                                     const uint32_t *const *a,
                                                     const uint32_t *const *b) {
  return convert<64, ScaledSource::Bf6, ScaledDestination::Bf16>(flags, exec_mask,
                                                                 instruction_flags, d, a, b);
}

// Widen sixteen packed FP6 values using the selected E8M0 scale byte.
int rj_arith_v_cvt_scale_pk16_f32_fp6_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                                  uint64_t instruction_flags, uint32_t *const *d,
                                                  const uint32_t *const *a,
                                                  const uint32_t *const *b) {
  return convert<32, ScaledSource::Fp6, ScaledDestination::F32, true>(flags, exec_mask,
                                                                      instruction_flags, d, a, b);
}

// Widen sixteen packed FP6 values using the selected E8M0 scale byte.
int rj_arith_v_cvt_scale_pk16_f16_fp6_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                                  uint64_t instruction_flags, uint32_t *const *d,
                                                  const uint32_t *const *a,
                                                  const uint32_t *const *b) {
  return convert<32, ScaledSource::Fp6, ScaledDestination::F16, true>(flags, exec_mask,
                                                                      instruction_flags, d, a, b);
}

// Widen sixteen packed FP6 values using the selected E8M0 scale byte.
int rj_arith_v_cvt_scale_pk16_bf16_fp6_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                                   uint64_t instruction_flags, uint32_t *const *d,
                                                   const uint32_t *const *a,
                                                   const uint32_t *const *b) {
  return convert<32, ScaledSource::Fp6, ScaledDestination::Bf16, true>(flags, exec_mask,
                                                                       instruction_flags, d, a, b);
}

// Widen sixteen packed BF6 values using the selected E8M0 scale byte.
int rj_arith_v_cvt_scale_pk16_f32_bf6_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                                  uint64_t instruction_flags, uint32_t *const *d,
                                                  const uint32_t *const *a,
                                                  const uint32_t *const *b) {
  return convert<32, ScaledSource::Bf6, ScaledDestination::F32, true>(flags, exec_mask,
                                                                      instruction_flags, d, a, b);
}

// Widen sixteen packed BF6 values using the selected E8M0 scale byte.
int rj_arith_v_cvt_scale_pk16_f16_bf6_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                                  uint64_t instruction_flags, uint32_t *const *d,
                                                  const uint32_t *const *a,
                                                  const uint32_t *const *b) {
  return convert<32, ScaledSource::Bf6, ScaledDestination::F16, true>(flags, exec_mask,
                                                                      instruction_flags, d, a, b);
}

// Widen sixteen packed BF6 values using the selected E8M0 scale byte.
int rj_arith_v_cvt_scale_pk16_bf16_bf6_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                                   uint64_t instruction_flags, uint32_t *const *d,
                                                   const uint32_t *const *a,
                                                   const uint32_t *const *b) {
  return convert<32, ScaledSource::Bf6, ScaledDestination::Bf16, true>(flags, exec_mask,
                                                                       instruction_flags, d, a, b);
}

// Widen eight packed FP4 values using the selected E8M0 scale byte.
int rj_arith_v_cvt_scale_pk8_f32_fp4_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                                 uint64_t instruction_flags, uint32_t *const *d,
                                                 const uint32_t *const *a,
                                                 const uint32_t *const *b) {
  return convert<32, ScaledSource::Fp4, ScaledDestination::F32, true>(flags, exec_mask,
                                                                      instruction_flags, d, a, b);
}

// Widen eight packed FP4 values using the selected E8M0 scale byte.
int rj_arith_v_cvt_scale_pk8_f16_fp4_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                                 uint64_t instruction_flags, uint32_t *const *d,
                                                 const uint32_t *const *a,
                                                 const uint32_t *const *b) {
  return convert<32, ScaledSource::Fp4, ScaledDestination::F16, true>(flags, exec_mask,
                                                                      instruction_flags, d, a, b);
}

// Widen eight packed FP4 values using the selected E8M0 scale byte.
int rj_arith_v_cvt_scale_pk8_bf16_fp4_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                                  uint64_t instruction_flags, uint32_t *const *d,
                                                  const uint32_t *const *a,
                                                  const uint32_t *const *b) {
  return convert<32, ScaledSource::Fp4, ScaledDestination::Bf16, true>(flags, exec_mask,
                                                                       instruction_flags, d, a, b);
}

// Widen eight packed FP8 values using the selected E8M0 scale byte.
int rj_arith_v_cvt_scale_pk8_f32_fp8_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                                 uint64_t instruction_flags, uint32_t *const *d,
                                                 const uint32_t *const *a,
                                                 const uint32_t *const *b) {
  return convert<32, ScaledSource::Fp8, ScaledDestination::F32, true>(flags, exec_mask,
                                                                      instruction_flags, d, a, b);
}

// Widen eight packed FP8 values using the selected E8M0 scale byte.
int rj_arith_v_cvt_scale_pk8_f16_fp8_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                                 uint64_t instruction_flags, uint32_t *const *d,
                                                 const uint32_t *const *a,
                                                 const uint32_t *const *b) {
  return convert<32, ScaledSource::Fp8, ScaledDestination::F16, true>(flags, exec_mask,
                                                                      instruction_flags, d, a, b);
}

// Widen eight packed FP8 values using the selected E8M0 scale byte.
int rj_arith_v_cvt_scale_pk8_bf16_fp8_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                                  uint64_t instruction_flags, uint32_t *const *d,
                                                  const uint32_t *const *a,
                                                  const uint32_t *const *b) {
  return convert<32, ScaledSource::Fp8, ScaledDestination::Bf16, true>(flags, exec_mask,
                                                                       instruction_flags, d, a, b);
}

// Widen eight packed BF8 values using the selected E8M0 scale byte.
int rj_arith_v_cvt_scale_pk8_f32_bf8_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                                 uint64_t instruction_flags, uint32_t *const *d,
                                                 const uint32_t *const *a,
                                                 const uint32_t *const *b) {
  return convert<32, ScaledSource::Bf8, ScaledDestination::F32, true>(flags, exec_mask,
                                                                      instruction_flags, d, a, b);
}

// Widen eight packed BF8 values using the selected E8M0 scale byte.
int rj_arith_v_cvt_scale_pk8_f16_bf8_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                                 uint64_t instruction_flags, uint32_t *const *d,
                                                 const uint32_t *const *a,
                                                 const uint32_t *const *b) {
  return convert<32, ScaledSource::Bf8, ScaledDestination::F16, true>(flags, exec_mask,
                                                                      instruction_flags, d, a, b);
}

// Widen eight packed BF8 values using the selected E8M0 scale byte.
int rj_arith_v_cvt_scale_pk8_bf16_bf8_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                                  uint64_t instruction_flags, uint32_t *const *d,
                                                  const uint32_t *const *a,
                                                  const uint32_t *const *b) {
  return convert<32, ScaledSource::Bf8, ScaledDestination::Bf16, true>(flags, exec_mask,
                                                                       instruction_flags, d, a, b);
}
