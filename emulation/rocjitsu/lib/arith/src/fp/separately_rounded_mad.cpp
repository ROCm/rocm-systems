// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Older CDNA/RDNA FP32 MAD: round and flush the product before adding C.
// Reuses the existing scalar/AVX2/AVX-512 FMA arithmetic for both steps.

#include "common/alu.h"
#include "common/validation.h"
#include "fp/fma.h"
#include "rj_arith/detail/rj_arith_fp.h"
#include "rj_arith/detail/rj_arith_register.h"

#include <stdint.h>

namespace {

// Evaluate scalar/default or directed fallback arithmetic. Inputs may alias D;
// both rounded stages finish before masked publication. Requires validated flags.
template <bool Legacy, typename Mask>
[[gnu::noinline]] int fallback(uint64_t flags, Mask exec_mask, uint32_t output_modifiers,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c) {
  using namespace rocjitsu::arith;
  constexpr int lanes = int(sizeof(Mask) * 8);
  uint32_t input[3][lanes], product[lanes], extra[lanes], result[lanes];
  const uint32_t *sources[] = {a[0], b[0], c[0]};
  for (int lane = 0; lane < lanes; ++lane) {
    for (int operand = 0; operand < 3; ++operand) {
      input[operand][lane] =
          alu_input_bits(sources[operand][lane], uint32_t(output_modifiers >> operand),
                         RJ_ARITH_FP_FLUSH_INPUT_DENORMALS);
    }
    if constexpr (Legacy) {
      if (!(input[0][lane] & 0x7fffffffU) || !(input[1][lane] & 0x7fffffffU)) {
        input[0][lane] = 0;
        input[1][lane] = 0x3f800000U;
      }
    }
    // A zero addend with the product's sign makes FMA implement multiplication,
    // including negative zero, without fusing the subsequent addition of C.
    extra[lane] = (input[0][lane] ^ input[1][lane]) & 0x80000000U;
  }
  if ((flags & (RJ_ARITH_CPU_MASK | RJ_ARITH_FP_ROUND_MASK)) == RJ_ARITH_CPU_BASELINE) {
    // This is the original scalar MAD arithmetic, under the contracted default
    // host rounding. Volatile keeps the product separate from the following add.
    for (int lane = 0; lane < lanes; ++lane) {
      volatile float rounded_product = as_float(input[0][lane]) * as_float(input[1][lane]);
      const uint32_t p =
          alu_input_bits(as_bits(rounded_product), 0, RJ_ARITH_FP_FLUSH_INPUT_DENORMALS);
      result[lane] = alu_input_bits(as_bits(as_float(p) + as_float(input[2][lane])), 0,
                                    RJ_ARITH_FP_FLUSH_INPUT_DENORMALS);
    }
  } else {
    const uint64_t numeric_flags =
        (flags & ~(RJ_ARITH_FP_DX10_CLAMP_OFF | RJ_ARITH_SEMANTICS_MASK |
                   RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) |
        (is_exact(flags) ? RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS
                         : RJ_ARITH_SEMANTICS_LOOSE) |
        RJ_ARITH_FP_FLUSH_INPUT_DENORMALS;
    for (int offset = 0; offset < lanes; offset += 32) {
      uint32_t *p = product + offset;
      const uint32_t *x = input[0] + offset, *y = input[1] + offset, *z = extra + offset;
      const int error =
          rj_arith_v_fma_f32_w32_gfx12(numeric_flags, 0xffffffffU, 0, &p, &x, &y, &z, nullptr);
      if (error) {
        return error;
      }
    }
    // Flush only after FMA has packed the guest-rounded result: a tiny exact
    // product can round upward to minimum normal and must then survive.
    for (auto &value : product) {
      value = alu_input_bits(value, 0, RJ_ARITH_FP_FLUSH_INPUT_DENORMALS);
    }
    for (auto &one : extra) {
      one = 0x3f800000U;
    }
    for (int offset = 0; offset < lanes; offset += 32) {
      uint32_t *out = result + offset;
      const uint32_t *p = product + offset, *one = extra + offset, *z = input[2] + offset;
      const int error =
          rj_arith_v_fma_f32_w32_gfx12(numeric_flags, 0xffffffffU, 0, &out, &p, &one, &z, nullptr);
      if (error) {
        return error;
      }
    }
  }
  for (auto &value : result) {
    value = alu_input_bits(value, 0, RJ_ARITH_FP_FLUSH_INPUT_DENORMALS);
  }
  // Both arithmetic steps are already rounded and flushed. Output scaling is
  // separate and follows the original MODE gate and DX10 NaN-clamping policy.
  if (output_modifiers & (RJ_ARITH_ALU_OMOD_MASK | RJ_ARITH_ALU_CLAMP)) {
    for (auto &value : result) {
      value = scale_omod_f32_bits(value, (output_modifiers >> 6) & 3,
                                  unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24));
      if (output_modifiers & RJ_ARITH_ALU_CLAMP) {
        value = clamp_f32_bits(value, flags & RJ_ARITH_FP_DX10_CLAMP_OFF);
      }
    }
  }
  const uint32_t *out = result;
  const uint64_t move_flags = flags & RJ_ARITH_CPU_MASK;
  if constexpr (sizeof(Mask) == 4) {
    return rj_arith_v_mov_b32_w32(move_flags, exec_mask, 0, d, &out);
  } else {
    return rj_arith_v_mov_b32_w64(move_flags, exec_mask, 0, d, &out);
  }
}

// Validate and dispatch MAD, keeping fallback scratch space off the SIMD path.
template <bool Legacy, typename Mask>
int run(uint64_t flags, Mask exec_mask, uint64_t instruction_flags, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  using namespace rocjitsu::arith;
  flags = resolve_compatibility(flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = validate(flags, instruction_flags & ~uint64_t(kFma32InstructionFlags),
                           kFmaGuestFpFlags | RJ_ARITH_FP_DX10_CLAMP_OFF)) {
    return error;
  }
  if (reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  // OMOD eligibility uses guest MODE, before MAD's unconditional internal flushing.
  const uint32_t output_modifiers = uint32_t(rdna3_output_modifiers(flags, instruction_flags));
  [[maybe_unused]] constexpr int lanes = int(sizeof(Mask) * 8);
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    for (int offset = 0; offset < lanes; offset += 32) {
      mad_f32_x86_64_v4(
          Legacy, flags | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS,
          uint32_t(exec_mask >> offset), output_modifiers, d[0] + offset, a[0] + offset,
          b[0] + offset, c[0] + offset);
    }
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3) &&
      (!(flags & RJ_ARITH_FP_ROUND_MASK) || !is_exact(flags))) {
    for (int offset = 0; offset < lanes; offset += 32) {
      mad_f32_x86_64_v3(
          Legacy, flags | RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS,
          uint32_t(exec_mask >> offset), output_modifiers, d[0] + offset, a[0] + offset,
          b[0] + offset, c[0] + offset);
    }
    return RJ_ARITH_SUCCESS;
  }
#endif
  return fallback<Legacy>(flags, exec_mask, output_modifiers, d, a, b, c);
}

} // namespace

// Compute separately rounded A*B+C, preserving inactive lanes; FP status is unsupported.
int rj_arith_v_mad_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c, uint32_t *) {
  return run<false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Wave64 counterpart, with the same two-rounding and overlap contract.
int rj_arith_v_mad_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c, uint32_t *) {
  return run<false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Legacy MAD replaces a product with +0 when either flushed multiplier is zero.
int rj_arith_v_mad_legacy_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c, uint32_t *) {
  return run<true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Wave64 counterpart of the legacy zero-product form.
int rj_arith_v_mad_legacy_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                  uint32_t *const *d, const uint32_t *const *a,
                                  const uint32_t *const *b, const uint32_t *const *c, uint32_t *) {
  return run<true>(flags, exec_mask, instruction_flags, d, a, b, c);
}
