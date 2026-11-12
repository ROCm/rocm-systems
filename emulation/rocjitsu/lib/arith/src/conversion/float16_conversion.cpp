// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Public FP16/BF16 conversion entry points, validation, Wave64 splitting and dispatch.
#include "conversion/float16_conversion.h"
#include "common/validation.h"
#include "conversion/float16_conversion_impl.h"
#include "register/dpp.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_conversion.h"

#include <stdint.h>

namespace {
using rocjitsu::arith::Float16Conversion;
using rocjitsu::arith::Float16Format;

// Convert one wave half after validation, retaining aliased inputs until consumed.
template <Float16Format Format, Float16Conversion Op>
int convert(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
            const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (mode >> 32) {
    return rocjitsu::arith::execute_dpp(
        flags, exec_mask, mode, a, [&](uint32_t exec_mask, const uint32_t *const *source) {
          return convert<Format, Op>(flags, exec_mask, uint32_t(mode), d, source, b, c);
        });
  }
  constexpr uint32_t arithmetic_modifiers =
      RJ_ARITH_ALU_ABS_A | RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_ABS_C | RJ_ARITH_ALU_NEG_A |
      RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_NEG_C | RJ_ARITH_ALU_CLAMP | RJ_ARITH_ALU_OMOD_MASK;
  constexpr uint32_t known =
      arithmetic_modifiers | (Op == Float16Conversion::Widen        ? RJ_ARITH_ALU_HIGH_A
                              : Op == Float16Conversion::Stochastic ? RJ_ARITH_ALU_HIGH_D
                                                                    : 0);
  if (int error = rocjitsu::arith::validate(flags, mode & ~uint64_t(known))) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    rocjitsu::arith::float16_conversion_x86_64_v4(Format, Op, flags, exec_mask, mode, d[0], a[0],
                                                  b ? b[0] : nullptr, c ? c[0] : nullptr);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    rocjitsu::arith::float16_conversion_x86_64_v3(Format, Op, flags, exec_mask, mode, d[0], a[0],
                                                  b ? b[0] : nullptr, c ? c[0] : nullptr);
    return RJ_ARITH_SUCCESS;
  }
#endif
  uint32_t result[32];
  const auto source = [&](uint32_t raw, int operand) {
    if constexpr (Op == Float16Conversion::Pack || Op == Float16Conversion::Stochastic) {
      if (mode & (RJ_ARITH_ALU_ABS_A << operand)) {
        raw &= 0x7fffffffU;
      }
      if (mode & (RJ_ARITH_ALU_NEG_A << operand)) {
        raw ^= 0x80000000U;
      }
    }
    return raw;
  };
  bool saturate = flags & RJ_ARITH_FP16_OVFL;
  for (int lane = 0; lane < 32; ++lane) {
    if constexpr (Op == Float16Conversion::Widen) {
      result[lane] = (a[0][lane] >> (mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0)) << 16;
    } else if constexpr (Op == Float16Conversion::Stochastic) {
      uint32_t value = rocjitsu::arith::narrow_float16<Format>(source(a[0][lane], 0), b[0][lane],
                                                               true, saturate);
      int shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
      result[lane] = (d[0][lane] & ~(0xffffU << shift)) | (value << shift);
    } else if constexpr (Op == Float16Conversion::StochasticPack) {
      result[lane] =
          rocjitsu::arith::narrow_packed_sr<Format>(a[0][lane], uint16_t(c[0][lane]), saturate) |
          (rocjitsu::arith::narrow_packed_sr<Format>(b[0][lane], uint16_t(c[0][lane] >> 16),
                                                     saturate)
           << 16);
    } else {
      result[lane] =
          rocjitsu::arith::narrow_float16<Format>(source(a[0][lane], 0), 0, false, saturate) |
          (rocjitsu::arith::narrow_float16<Format>(source(b[0][lane], 1), 0, false, saturate)
           << 16);
    }
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Validate every control before splitting Wave64; pointers are one VGPR per operand.
template <Float16Format Format, Float16Conversion Op, int Lanes>
int execute(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
            const uint32_t *const *a, const uint32_t *const *b = nullptr,
            const uint32_t *const *c = nullptr) {
  // The inherited gfx1250 packed FP16 model is not hardware-qualified for
  // output-denormal and stochastic boundary behavior. Do not advertise exactness.
  constexpr bool qualified =
      Format != Float16Format::F16 ||
      (Op != Float16Conversion::Pack && Op != Float16Conversion::StochasticPack);
  flags = rocjitsu::arith::resolve_compatibility(
      flags,
      qualified ? RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS : RJ_ARITH_SEMANTICS_LOOSE);
  if constexpr (!qualified) {
    if (rocjitsu::arith::is_exact(flags) &&
        (flags & RJ_ARITH_SEMANTICS_MASK) != RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE) {
      return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
    }
  }
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  if (int error = convert<Format, Op>(flags, 0, mode, nullptr, nullptr, nullptr, nullptr)) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  for (int half = 0; half < Lanes / 32; ++half) {
    uint32_t *dest = d[0] + 32 * half;
    const uint32_t *av = a[0] + 32 * half, *bv = b ? b[0] + 32 * half : nullptr,
                   *cv = c ? c[0] + 32 * half : nullptr;
    if (int error =
            convert<Format, Op>(flags, uint32_t(exec_mask >> (32 * half)),
                                rocjitsu::arith::dpp_half_wave_instruction_flags(mode, half), &dest,
                                &av, b ? &bv : nullptr, c ? &cv : nullptr)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Widen the selected BF16 half to FP32; preserve inactive lanes.
int rj_arith_v_cvt_f32_bf16_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                uint32_t *const *d, const uint32_t *const *a) {
  return execute<Float16Format::Bf16, Float16Conversion::Widen, 32>(flags, exec_mask, mode, d, a);
}

// Widen the selected BF16 half to FP32; preserve inactive lanes.
int rj_arith_v_cvt_f32_bf16_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                uint32_t *const *d, const uint32_t *const *a) {
  return execute<Float16Format::Bf16, Float16Conversion::Widen, 64>(flags, exec_mask, mode, d, a);
}

// Pack two FP32 operands into BF16 using nearest-even rounding; preserve inactive lanes.
int rj_arith_v_cvt_pk_bf16_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b) {
  return execute<Float16Format::Bf16, Float16Conversion::Pack, 32>(flags, exec_mask, mode, d, a, b);
}

// Pack two FP32 operands into BF16 using nearest-even rounding; preserve inactive lanes.
int rj_arith_v_cvt_pk_bf16_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b) {
  return execute<Float16Format::Bf16, Float16Conversion::Pack, 64>(flags, exec_mask, mode, d, a, b);
}

// Narrow FP32 to one BF16 half using a seed; preserve the other half; preserve inactive lanes.
int rj_arith_v_cvt_sr_bf16_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b) {
  return execute<Float16Format::Bf16, Float16Conversion::Stochastic, 32>(flags, exec_mask, mode, d,
                                                                         a, b);
}

// Narrow FP32 to one BF16 half using a seed; preserve the other half; preserve inactive lanes.
int rj_arith_v_cvt_sr_bf16_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                   uint32_t *const *d, const uint32_t *const *a,
                                   const uint32_t *const *b) {
  return execute<Float16Format::Bf16, Float16Conversion::Stochastic, 64>(flags, exec_mask, mode, d,
                                                                         a, b);
}

// Pack two FP32 operands into BF16 using successive stochastic seeds; preserve inactive lanes.
int rj_arith_v_cvt_sr_pk_bf16_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                      uint32_t *const *d, const uint32_t *const *a,
                                      const uint32_t *const *b, const uint32_t *const *c) {
  return execute<Float16Format::Bf16, Float16Conversion::StochasticPack, 32>(flags, exec_mask, mode,
                                                                             d, a, b, c);
}

// Pack two FP32 operands into BF16 using successive stochastic seeds; preserve inactive lanes.
int rj_arith_v_cvt_sr_pk_bf16_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                      uint32_t *const *d, const uint32_t *const *a,
                                      const uint32_t *const *b, const uint32_t *const *c) {
  return execute<Float16Format::Bf16, Float16Conversion::StochasticPack, 64>(flags, exec_mask, mode,
                                                                             d, a, b, c);
}

// Pack two FP32 operands into FP16 using nearest-even rounding; preserve inactive lanes.
int rj_arith_v_cvt_pk_f16_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b) {
  return execute<Float16Format::F16, Float16Conversion::Pack, 32>(flags, exec_mask, mode, d, a, b);
}

// Pack two FP32 operands into FP16 using nearest-even rounding; preserve inactive lanes.
int rj_arith_v_cvt_pk_f16_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b) {
  return execute<Float16Format::F16, Float16Conversion::Pack, 64>(flags, exec_mask, mode, d, a, b);
}

// Narrow FP32 to one FP16 half using a seed; preserve the other half; preserve inactive lanes.
int rj_arith_v_cvt_sr_f16_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b) {
  return execute<Float16Format::F16, Float16Conversion::Stochastic, 32>(flags, exec_mask, mode, d,
                                                                        a, b);
}

// Narrow FP32 to one FP16 half using a seed; preserve the other half; preserve inactive lanes.
int rj_arith_v_cvt_sr_f16_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b) {
  return execute<Float16Format::F16, Float16Conversion::Stochastic, 64>(flags, exec_mask, mode, d,
                                                                        a, b);
}

// Pack two FP32 operands into FP16 using successive stochastic seeds; preserve inactive lanes.
int rj_arith_v_cvt_sr_pk_f16_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, const uint32_t *const *c) {
  return execute<Float16Format::F16, Float16Conversion::StochasticPack, 32>(flags, exec_mask, mode,
                                                                            d, a, b, c);
}

// Pack two FP32 operands into FP16 using successive stochastic seeds; preserve inactive lanes.
int rj_arith_v_cvt_sr_pk_f16_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                     uint32_t *const *d, const uint32_t *const *a,
                                     const uint32_t *const *b, const uint32_t *const *c) {
  return execute<Float16Format::F16, Float16Conversion::StochasticPack, 64>(flags, exec_mask, mode,
                                                                            d, a, b, c);
}
