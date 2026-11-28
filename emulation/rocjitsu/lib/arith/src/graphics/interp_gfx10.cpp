// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx10 LDS-coefficient interpolation. Memory access belongs to the caller;
// numeric preparation, guest rounding and output modifiers belong here.
#include "graphics/interp_gfx10.h"
#include "common/alu.h"
#include "common/float_encoding.h"
#include "common/validation.h"
#include "fp/mixed_fma.h"
#include "graphics/rounding.h"
#include "rj_arith/detail/rj_arith_fp.h"
#include "rj_arith/detail/rj_arith_graphics.h"

namespace {
using namespace rocjitsu::arith;
enum class Stage { Mov, P1, P2, P1LL, P1LV, P2Half };

// Evaluate a staged wave, preserving inactive lanes and arbitrary operand overlap.
// Compatibility retains the imported gfx10 model; strict empirical qualification is absent.
template <Stage Op, int Lanes>
int interpolate(uint64_t flags, ExecMask<Lanes> exec_mask, uint64_t mode, uint32_t *const *d,
                const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  const bool loose = (flags & RJ_ARITH_SEMANTICS_MASK) != RJ_ARITH_SEMANTICS_COMPATIBILITY;
  if (int error = loose_only(flags, false)) {
    return error;
  }
  constexpr bool half = Op == Stage::P1LL || Op == Stage::P1LV || Op == Stage::P2Half;
  constexpr bool half_c = Op == Stage::P1LL || Op == Stage::P1LV;
  constexpr bool half_out = Op == Stage::P2Half;
  constexpr uint64_t fp =
      kFmaGuestFpFlags | RJ_ARITH_FP_DX10_CLAMP_OFF | RJ_ARITH_FP_INPUT_DENORMALS_F16_MASK;
  constexpr uint64_t known = RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_NEG_C | RJ_ARITH_ALU_ABS_B |
                             RJ_ARITH_ALU_ABS_C | RJ_ARITH_ALU_CLAMP | RJ_ARITH_ALU_OMOD_MASK |
                             (half ? RJ_ARITH_ALU_HIGH_A : 0) |
                             (half_out ? RJ_ARITH_ALU_HIGH_D : 0);
  if (int error = validate(flags, mode & ~known, fp)) {
    return error;
  }
  if ((flags & RJ_ARITH_FP_INPUT_DENORMALS_F16_MASK) == RJ_ARITH_FP_INPUT_DENORMALS_F16_MASK) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  // Only P1LV and P2_F16 have an explicit source-2 arithmetic operand.
  // Other stages accept but ignore its otherwise unused modifier fields.
  if constexpr (Op != Stage::P1LV && Op != Stage::P2Half) {
    mode &= ~(RJ_ARITH_ALU_NEG_C | RJ_ARITH_ALU_ABS_C);
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  alignas(64) uint32_t result[Lanes]{};
  const auto half_input = [&](uint32_t bits, bool high) {
    auto value = uint16_t(bits >> (high ? 16 : 0));
    const auto policy = flags & RJ_ARITH_FP_INPUT_DENORMALS_F16_MASK;
    const bool flush = policy ? policy == RJ_ARITH_FP_FLUSH_INPUT_DENORMALS_F16
                              : bool(flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS);
    if (flush && !(value & 0x7c00)) {
      value &= 0x8000;
    }
    return as_bits(f16_to_float(value));
  };
  const unsigned rounding = unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
  // gfx10 flushes half outputs after narrowing; keep that final stage below.
  uint64_t arithmetic_flags = flags;
  if constexpr (half_out) {
    arithmetic_flags &= ~RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS;
  }
  const uint32_t mixed_mode = (uint32_t(mode) & ~(RJ_ARITH_ALU_OMOD_MASK | RJ_ARITH_ALU_HIGH_D)) |
                              (half ? RJ_ARITH_MIX_F16_A : 0) | (half_c ? RJ_ARITH_MIX_F16_C : 0) |
                              (half_c && (mode & RJ_ARITH_ALU_HIGH_A) ? RJ_ARITH_ALU_HIGH_C : 0);
  bool prepared = false;
  ExecMask<Lanes> repair = 0;
  if constexpr (Op != Stage::Mov) {
    // Share native nearest-even FMA; directed guest rounding retains the imported model.
    const bool native = loose && !(flags & RJ_ARITH_FP_ROUND_MASK);
#if defined(RJ_ARITH_HAVE_X86_64_V4)
    if (!prepared && cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
      for (int base = 0; base < Lanes; base += 32) {
        if (native) {
          mixed_fma_loose_x86_64_v4(half_out ? MixedFma::Low : MixedFma::Float, true,
                                    arithmetic_flags, uint32_t(exec_mask >> base),
                                    half_out ? mixed_mode : mixed_mode & ~RJ_ARITH_ALU_CLAMP,
                                    result + base, a[0] + base, b[0] + base, c[0] + base);
        } else {
          mixed_fma_exact_x86_64_v4(half_out ? MixedFma::Low : MixedFma::Float, true,
                                    arithmetic_flags, uint32_t(exec_mask >> base),
                                    half_out ? mixed_mode : mixed_mode & ~RJ_ARITH_ALU_CLAMP,
                                    result + base, a[0] + base, b[0] + base, c[0] + base);
        }
      }
      prepared = true;
    }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
    if (!prepared && cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
      for (int base = 0; base < Lanes; base += 32) {
        if (native) {
          if constexpr (half_out) {
            mixed_fma_half_x86_64_v3(false, arithmetic_flags, uint32_t(exec_mask >> base),
                                     mixed_mode, result + base, a[0] + base, b[0] + base,
                                     c[0] + base);
          } else {
            mixed_fma_float_guest_x86_64_v3(arithmetic_flags, uint32_t(exec_mask >> base),
                                            mixed_mode & ~RJ_ARITH_ALU_CLAMP, result + base,
                                            a[0] + base, b[0] + base, c[0] + base);
          }
        } else if constexpr (half_out) {
          mixed_fma_half_guest_x86_64_v3(false, arithmetic_flags, uint32_t(exec_mask >> base),
                                         mixed_mode, result + base, a[0] + base, b[0] + base,
                                         c[0] + base);
        } else if (rounding) {
          repair |= ExecMask<Lanes>(interp_gfx10_directed_x86_64_v3(
                        arithmetic_flags, uint32_t(exec_mask >> base), mixed_mode, result + base,
                        a[0] + base, b[0] + base, c[0] + base))
                    << base;
        } else {
          mixed_fma_float_exact_x86_64_v3(true, arithmetic_flags, uint32_t(exec_mask >> base),
                                          mixed_mode & ~RJ_ARITH_ALU_CLAMP, result + base,
                                          a[0] + base, b[0] + base, c[0] + base);
        }
      }
      prepared = true;
    }
#endif
    if (!prepared && native) {
      auto call = half_out ? rj_arith_v_fma_mixlo_f16_w32 : rj_arith_v_fma_mix_f32_w32;
      for (int base = 0; base < Lanes; base += 32) {
        uint32_t *output = result + base;
        const uint32_t *x = a[0] + base, *y = b[0] + base, *z = c[0] + base;
        if (int error = call(arithmetic_flags, uint32_t(exec_mask >> base),
                             half_out ? mixed_mode : mixed_mode & ~RJ_ARITH_ALU_CLAMP, &output, &x,
                             &y, &z, nullptr)) {
          return error;
        }
      }
      prepared = true;
    }
  }
  for (int lane = 0; lane < Lanes; ++lane) {
    if (!((exec_mask >> lane) & 1)) {
      continue;
    }
    bool computed = prepared && !((repair >> lane) & 1);
    if constexpr (half_out) {
      // The inherited nonfinite FP64 FMA may select NaN payloads differently
      // across compilers. Share that baseline policy, rather than the mixed-FMA
      // backend's separately qualified NaN ordering, for these lanes only.
      const uint32_t coefficient = (a[0][lane] >> (mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0)) & 0x7fff;
      computed &= loose || (coefficient < 0x7c00 && (b[0][lane] & 0x7fffffff) < 0x7f800000 &&
                            (c[0][lane] & 0x7fffffff) < 0x7f800000);
    }
    const uint32_t x = !computed && half ? half_input(a[0][lane], mode & RJ_ARITH_ALU_HIGH_A)
                       : !computed       ? a[0][lane]
                                         : 0;
    uint32_t value = x;
    if constexpr (Op != Stage::Mov) {
      const uint32_t y = !computed ? alu_input_bits(b[0][lane], mode >> 1) : 0;
      const uint32_t z =
          !computed ? alu_input_bits(half_c ? half_input(c[0][lane], mode & RJ_ARITH_ALU_HIGH_A)
                                            : c[0][lane],
                                     mode >> 2)
                    : 0;
      if (computed) {
        value = result[lane];
      } else if constexpr (half_out) {
        const auto input = [&](uint32_t bits) { return as_float(alu_input_bits(bits, 0, flags)); };
        value = graphics_rounding::fma_f32_to_f16_nearest_environment(
            as_float(x), input(y), input(z), rounding, mode & RJ_ARITH_ALU_CLAMP,
            flags & RJ_ARITH_FP16_OVFL, !(flags & RJ_ARITH_FP_DX10_CLAMP_OFF));
      } else {
        unsigned denorm = (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS ? 0 : 1) |
                          (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS ? 0 : 2);
        value = as_bits(graphics_rounding::fma_f32(as_float(x), as_float(y), as_float(z), rounding,
                                                   denorm, flags & RJ_ARITH_FP_IEEE_MODE));
      }
    }
    if constexpr (half_out) {
      if ((flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) && !(value & 0x7c00)) {
        value &= 0x8000;
      }
      unsigned shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
      value = (d[0][lane] & ~(0xffffU << shift)) | ((value & 0xffff) << shift);
    } else {
      value = scale_omod_f32_bits(value, unsigned((rdna3_output_modifiers(flags, mode) >> 6) & 3),
                                  rounding);
      if (mode & RJ_ARITH_ALU_CLAMP) {
        value = clamp_f32_bits(value, flags & RJ_ARITH_FP_DX10_CLAMP_OFF);
      }
    }
    result[lane] = value;
  }
  for (int lane = 0; lane < Lanes; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}
} // namespace

// Evaluate gfx10 INTERP_MOV_F32 for the selected lanes.
int rj_arith_v_interp_mov_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c) {
  return interpolate<Stage::Mov, 32>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Evaluate gfx10 INTERP_MOV_F32 for the selected lanes.
int rj_arith_v_interp_mov_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c) {
  return interpolate<Stage::Mov, 64>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Evaluate gfx10 INTERP_P1_F32 for the selected lanes.
int rj_arith_v_interp_p1_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c) {
  return interpolate<Stage::P1, 32>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Evaluate gfx10 INTERP_P1_F32 for the selected lanes.
int rj_arith_v_interp_p1_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c) {
  return interpolate<Stage::P1, 64>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Evaluate gfx10 INTERP_P2_F32 for the selected lanes.
int rj_arith_v_interp_p2_f32_w32_gfx10(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c) {
  return interpolate<Stage::P2, 32>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Evaluate gfx10 INTERP_P2_F32 for the selected lanes.
int rj_arith_v_interp_p2_f32_w64_gfx10(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c) {
  return interpolate<Stage::P2, 64>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Evaluate gfx10 INTERP_P1LL_F16 for the selected lanes.
int rj_arith_v_interp_p1ll_f16_w32_gfx10(uint64_t flags, uint32_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         const uint32_t *const *c) {
  return interpolate<Stage::P1LL, 32>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Evaluate gfx10 INTERP_P1LL_F16 for the selected lanes.
int rj_arith_v_interp_p1ll_f16_w64_gfx10(uint64_t flags, uint64_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         const uint32_t *const *c) {
  return interpolate<Stage::P1LL, 64>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Evaluate gfx10 INTERP_P1LV_F16 for the selected lanes.
int rj_arith_v_interp_p1lv_f16_w32_gfx10(uint64_t flags, uint32_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         const uint32_t *const *c) {
  return interpolate<Stage::P1LV, 32>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Evaluate gfx10 INTERP_P1LV_F16 for the selected lanes.
int rj_arith_v_interp_p1lv_f16_w64_gfx10(uint64_t flags, uint64_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         const uint32_t *const *c) {
  return interpolate<Stage::P1LV, 64>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Evaluate gfx10 INTERP_P2_F16 for the selected lanes.
int rj_arith_v_interp_p2_f16_w32_gfx10(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c) {
  return interpolate<Stage::P2Half, 32>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Evaluate gfx10 INTERP_P2_F16 for the selected lanes.
int rj_arith_v_interp_p2_f16_w64_gfx10(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c) {
  return interpolate<Stage::P2Half, 64>(flags, exec_mask, instruction_flags, d, a, b, c);
}
