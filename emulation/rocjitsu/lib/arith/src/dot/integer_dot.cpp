// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Packed INT16/INT8/INT4 dot products with wrapping or saturating INT32 accumulation.
// Owns public entry points and their validation/dispatch policy.
// Integer DOT semantics follow rocjitsu codegen/execute/packed.py.

#include "common/simd.h"
#include "common/validation.h"
#include "dot/operand_halves.h"
#include "rj_arith/detail/rj_arith_dot.h"

#include <stdint.h>

namespace {

// Accumulate packed integer dot products for selected lanes with validated controls.
template <int Bits, bool Unsigned>
int dot(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  if (exec_mask == 0) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    rocjitsu::arith::integer_dot_x86_64_v4(Bits, Unsigned, exec_mask, instruction_flags, d[0], a[0],
                                           b[0], c[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (rocjitsu::arith::cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    rocjitsu::arith::integer_dot_x86_64_v3(Bits, Unsigned, exec_mask, instruction_flags, d[0], a[0],
                                           b[0], c[0]);
    return RJ_ARITH_SUCCESS;
  }
#endif
  if (instruction_flags & rocjitsu::arith::kDotHalfSelection) {
    return rocjitsu::arith::with_dot_halves(
        instruction_flags, a, b,
        [&](uint32_t selected, const uint32_t *const *av, const uint32_t *const *bv) {
          return dot<Bits, Unsigned>(flags, exec_mask, selected, d, av, bv, c);
        });
  }
  const int sign_a = instruction_flags & RJ_ARITH_DOT_SIGNED_A ? 1 << (Bits - 1) : 0;
  const int sign_b = instruction_flags & RJ_ARITH_DOT_SIGNED_B ? 1 << (Bits - 1) : 0;
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    int64_t sum = c[0][lane];
    if (!Unsigned && (c[0][lane] & 0x80000000)) {
      sum -= INT64_C(1) << 32;
    }
    for (int shift = 0; shift < 32; shift += Bits) {
      int x = ((a[0][lane] >> shift) & ((1 << Bits) - 1));
      int y = ((b[0][lane] >> shift) & ((1 << Bits) - 1));
      sum += int64_t((x ^ sign_a) - sign_a) * ((y ^ sign_b) - sign_b);
    }
    if (instruction_flags & RJ_ARITH_DOT_CLAMP) {
      sum = rocjitsu::arith::clamp(sum, Unsigned ? INT64_C(0) : int64_t(INT32_MIN),
                                   Unsigned ? int64_t(UINT32_MAX) : int64_t(INT32_MAX));
    }
    result[lane] = uint32_t(sum);
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }
  return RJ_ARITH_SUCCESS;
}

// Validate one instruction before reading operands; both wave widths share the same
// arithmetic. Fixed signed forms supply signed factors without exposing IU controls.
template <int Bits, bool Unsigned, bool FixedSigned, bool Accumulate, bool IgnoreClamp,
          typename Mask>
int run(uint64_t flags, Mask exec_mask, uint64_t instruction_flags, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  flags = rocjitsu::arith::resolve_compatibility(
      flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  const uint32_t known =
      Accumulate
          ? 0
          : rocjitsu::arith::kDotHalfSelection | RJ_ARITH_DOT_CLAMP |
                (Unsigned || FixedSigned ? 0 : RJ_ARITH_DOT_SIGNED_A | RJ_ARITH_DOT_SIGNED_B);
  if (int error = rocjitsu::arith::validate(flags, instruction_flags & ~uint64_t(known))) {
    return error;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  if constexpr (FixedSigned) {
    instruction_flags |= RJ_ARITH_DOT_SIGNED_A | RJ_ARITH_DOT_SIGNED_B;
  }
  if constexpr (IgnoreClamp) {
    instruction_flags &= ~uint64_t(RJ_ARITH_DOT_CLAMP);
  }
  for (int offset = 0; offset < int(sizeof(Mask) * 8); offset += 32) {
    uint32_t *dest = d[0] + offset;
    const uint32_t *source_a = a[0] + offset, *source_b = b[0] + offset;
    const uint32_t *source_c = Accumulate ? dest : c[0] + offset;
    if (int error =
            dot<Bits, Unsigned>(flags, uint32_t(exec_mask >> offset), uint32_t(instruction_flags),
                                &dest, &source_a, &source_b, &source_c)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Accumulate packed products, preserving inactive lanes; validation errors leave outputs unchanged.
int rj_arith_v_dot4_i32_iu8_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c) {
  return run<8, false, false, false, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate packed products, preserving inactive lanes; validation errors leave outputs unchanged.
int rj_arith_v_dot4_i32_iu8_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c) {
  return run<8, false, false, false, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate packed products, preserving inactive lanes; validation errors leave outputs unchanged.
int rj_arith_v_dot4_u32_u8_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c) {
  return run<8, true, false, false, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate packed products, preserving inactive lanes; validation errors leave outputs unchanged.
int rj_arith_v_dot4_u32_u8_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c) {
  return run<8, true, false, false, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate packed products, preserving inactive lanes; validation errors leave outputs unchanged.
int rj_arith_v_dot8_i32_iu4_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c) {
  return run<4, false, false, false, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate packed products, preserving inactive lanes; validation errors leave outputs unchanged.
int rj_arith_v_dot8_i32_iu4_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c) {
  return run<4, false, false, false, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate packed products, preserving inactive lanes; validation errors leave outputs unchanged.
int rj_arith_v_dot8_u32_u4_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c) {
  return run<4, true, false, false, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate packed products, preserving inactive lanes; validation errors leave outputs unchanged.
int rj_arith_v_dot8_u32_u4_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c) {
  return run<4, true, false, false, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate packed products, preserving inactive lanes; validation errors leave outputs unchanged.
int rj_arith_v_dot2_i32_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c) {
  return run<16, false, true, false, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate packed products, preserving inactive lanes; validation errors leave outputs unchanged.
int rj_arith_v_dot2_i32_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c) {
  return run<16, false, true, false, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate packed products, preserving inactive lanes; validation errors leave outputs unchanged.
int rj_arith_v_dot2_u32_u16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c) {
  return run<16, true, false, false, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate packed products, preserving inactive lanes; validation errors leave outputs unchanged.
int rj_arith_v_dot2_u32_u16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b, const uint32_t *const *c) {
  return run<16, true, false, false, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate packed products, preserving inactive lanes; validation errors leave outputs unchanged.
int rj_arith_v_dot4_i32_i8_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c) {
  return run<8, false, true, false, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate packed products, preserving inactive lanes; validation errors leave outputs unchanged.
int rj_arith_v_dot4_i32_i8_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c) {
  return run<8, false, true, false, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate packed products, preserving inactive lanes; validation errors leave outputs unchanged.
int rj_arith_v_dot8_i32_i4_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c) {
  return run<4, false, true, false, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate packed products, preserving inactive lanes; validation errors leave outputs unchanged.
int rj_arith_v_dot8_i32_i4_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                               uint32_t *const *d, const uint32_t *const *a,
                               const uint32_t *const *b, const uint32_t *const *c) {
  return run<4, false, true, false, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate packed products, preserving inactive lanes; validation errors leave outputs unchanged.
int rj_arith_v_dot2c_i32_i16_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b) {
  return run<16, false, true, true, false>(flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Accumulate packed products, preserving inactive lanes; validation errors leave outputs unchanged.
int rj_arith_v_dot2c_i32_i16_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                 uint32_t *const *d, const uint32_t *const *a,
                                 const uint32_t *const *b) {
  return run<16, false, true, true, false>(flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Accumulate packed products, preserving inactive lanes; validation errors leave outputs unchanged.
int rj_arith_v_dot4c_i32_i8_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b) {
  return run<8, false, true, true, false>(flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Accumulate packed products, preserving inactive lanes; validation errors leave outputs unchanged.
int rj_arith_v_dot4c_i32_i8_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b) {
  return run<8, false, true, true, false>(flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Accumulate packed products, preserving inactive lanes; validation errors leave outputs unchanged.
int rj_arith_v_dot8c_i32_i4_w32(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b) {
  return run<4, false, true, true, false>(flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Accumulate packed products, preserving inactive lanes; validation errors leave outputs unchanged.
int rj_arith_v_dot8c_i32_i4_w64(uint64_t flags, uint64_t exec_mask, uint64_t instruction_flags,
                                uint32_t *const *d, const uint32_t *const *a,
                                const uint32_t *const *b) {
  return run<4, false, true, true, false>(flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Accumulate packed products, preserving inactive lanes; validation errors leave outputs unchanged.
int rj_arith_v_dot4_i32_iu8_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c) {
  return run<8, false, false, false, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate packed products, preserving inactive lanes; validation errors leave outputs unchanged.
int rj_arith_v_dot4_i32_iu8_w64_gfx1250(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c) {
  return run<8, false, false, false, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate packed products, preserving inactive lanes; validation errors leave outputs unchanged.
int rj_arith_v_dot4_u32_u8_w32_gfx1250(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c) {
  return run<8, true, false, false, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate packed products, preserving inactive lanes; validation errors leave outputs unchanged.
int rj_arith_v_dot4_u32_u8_w64_gfx1250(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c) {
  return run<8, true, false, false, true>(flags, exec_mask, instruction_flags, d, a, b, c);
}
