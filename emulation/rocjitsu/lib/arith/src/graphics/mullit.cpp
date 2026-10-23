// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Lit-multiply arithmetic and special-value fixup controlled by the third operand.
// Owns public entry points and their validation/dispatch policy.

#include "graphics/mullit.h"
#include "common/alu.h"
#include "common/internal.h"
#include "register/dpp.h"
#include "rj_arith/rj_arith.h"

#include <cmath>
#include <stdint.h>

namespace {

// Compute lit-multiply results for active Wave32 lanes, preserving inactive destinations and
// input/output aliases.
template <bool Rdna3>
[[gnu::noinline]] void scalar(uint32_t exec_mask, uint32_t mode, uint64_t flags, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c) {
  uint32_t result[32];
  for (int lane = 0; lane < 32; ++lane) {
    float x = rocjitsu::arith::alu_input(a[0][lane], mode);
    float y = rocjitsu::arith::alu_input(b[0][lane], mode >> 1);
    float z = rocjitsu::arith::alu_input(c[0][lane], mode >> 2);
    uint32_t ybits = rocjitsu::arith::as_bits(y);
    float value;
    if (ybits == 0xff7fffff || ybits == 0xff800000 || std::isnan(y) || !(z > 0)) {
      value = rocjitsu::arith::as_float(0xff7fffff);
    } else {
      value = x == 0 || y == 0 ? 0.0f : x * y;
    }
    unsigned scale = (mode >> 6) & 3;
    if (scale) {
      // RDNA4 flushes before scaling; RDNA3 can normalize a tiny product.
      if constexpr (!Rdna3) {
        if (!(rocjitsu::arith::as_bits(value) & 0x7f800000)) {
          value = 0;
        }
      }
      value *= scale == 1 ? 2.0f : scale == 2 ? 4.0f : 0.5f;
      uint32_t bits = rocjitsu::arith::as_bits(value);
      if (!(bits & 0x7f800000)) {
        value = Rdna3 ? 0.0f : rocjitsu::arith::as_float(bits & 0x80000000);
      }
    }
    if ((mode & RJ_ARITH_ALU_CLAMP) &&
        !(Rdna3 && (flags & RJ_ARITH_FP_DX10_CLAMP_OFF) && std::isnan(value))) {
      value = !(value > 0) ? 0 : value > 1 ? 1 : value;
    }
    result[lane] = rocjitsu::arith::as_bits(value);
  }
  for (int lane = 0; lane < 32; ++lane) {
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = result[lane];
    }
  }
}

// Dispatch validated Wave32 lit-multiply arithmetic to the selected CPU backend.
template <bool Rdna3>
void compute([[maybe_unused]] uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
             const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V4) {
    rocjitsu::arith::mullit_x86_64_v4<Rdna3>(exec_mask, mode, flags, d[0], a[0], b[0], c[0]);
    return;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if ((flags & RJ_ARITH_CPU_MASK) >= RJ_ARITH_CPU_X86_64_V3) {
    rocjitsu::arith::mullit_x86_64_v3<Rdna3>(exec_mask, mode, flags, d[0], a[0], b[0], c[0]);
    return;
  }
#endif
  scalar<Rdna3>(exec_mask, mode, flags, d, a, b, c);
}

// Apply explicit guest flushing around the same arithmetic kernels. Compute all
// results before any destination write so every supported alias remains safe.
template <bool Rdna3>
[[gnu::noinline]] void compute_flushed(uint64_t flags, uint32_t exec_mask, uint32_t mode,
                                       uint32_t *const *d, const uint32_t *const *a,
                                       const uint32_t *const *b, const uint32_t *const *c) {
  uint32_t inputs[3][32], result[32];
  const uint32_t *sources[] = {a[0], b[0], c[0]};
  if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
    for (int operand = 0; operand < 3; ++operand) {
      for (int lane = 0; lane < 32; ++lane) {
        uint32_t bits = sources[operand][lane];
        inputs[operand][lane] = (bits & 0x7fffffff) < 0x00800000 ? bits & 0x80000000 : bits;
      }
      sources[operand] = inputs[operand];
    }
  }
  uint32_t *dest = result;
  compute<Rdna3>(flags, UINT32_MAX, mode, &dest, &sources[0], &sources[1], &sources[2]);
  for (int lane = 0; lane < 32; ++lane) {
    uint32_t bits = result[lane];
    if ((flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) && (bits & 0x7fffffff) < 0x00800000) {
      bits &= 0x80000000;
    }
    if ((exec_mask >> lane) & 1) {
      d[0][lane] = bits;
    }
  }
}

// Multiply with lit special-value fixup for the selected lanes. Reject invalid controls before
// changing outputs.
template <bool Rdna3>
int run(uint64_t flags, uint32_t exec_mask, uint64_t instruction_flags, uint32_t *const *d,
        const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
        [[maybe_unused]] uint32_t *excp_flag_user) {
  if (rocjitsu::arith::reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (instruction_flags >> 32) {
    return rocjitsu::arith::execute_dpp(flags, exec_mask, instruction_flags, a,
                                        [&](uint32_t exec_mask, const uint32_t *const *source) {
                                          return run<Rdna3>(flags, exec_mask,
                                                            uint32_t(instruction_flags), d, source,
                                                            b, c, nullptr);
                                        });
  }
  if (int error = rocjitsu::arith::validate(flags, instruction_flags & ~0x1ffU, false,
                                            RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                                RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS |
                                                RJ_ARITH_FP_IEEE_MODE | RJ_ARITH_FP_ROUND_MASK |
                                                (Rdna3 ? RJ_ARITH_FP_DX10_CLAMP_OFF : 0))) {
    return error;
  }
  if constexpr (Rdna3) {
    if (!(flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) || (flags & RJ_ARITH_FP_IEEE_MODE)) {
      instruction_flags &= ~uint64_t(RJ_ARITH_ALU_OMOD_HALF);
    }
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  if (flags & (RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
    compute_flushed<Rdna3>(flags, exec_mask, instruction_flags, d, a, b, c);
  } else {
    compute<Rdna3>(flags, exec_mask, instruction_flags, d, a, b, c);
  }
  return RJ_ARITH_SUCCESS;
}

// Multiply with lit special-value fixup for Wave64. Reject invalid controls before changing
// outputs.
template <bool Rdna3>
int run_wave64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
               const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c,
               [[maybe_unused]] uint32_t *excp_flag_user) {
  // Validate before either Wave64 half can modify outputs, even when EXEC is empty.
  if (int error = run<Rdna3>(flags, 0, mode, nullptr, nullptr, nullptr, nullptr, excp_flag_user)) {
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
    const uint32_t *av = a[0] + half * 32, *bv = b[0] + half * 32, *cv = c[0] + half * 32;
    if (int error = run<Rdna3>(flags, uint32_t(exec_mask >> (half * 32)), half_mode, &dest, &av,
                               &bv, &cv, excp_flag_user)) {
      return error;
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Multiply with lit special-value fixup for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_mullit_f32_w32(uint64_t flags, uint32_t exec_mask, uint64_t mode, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<true>(flags, exec_mask, mode, d, a, b, c, trapsts);
}

// Multiply with lit special-value fixup for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_mullit_f32_w64(uint64_t flags, uint64_t exec_mask, uint64_t mode, uint32_t *const *d,
                              const uint32_t *const *a, const uint32_t *const *b,
                              const uint32_t *const *c, uint32_t *trapsts) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run_wave64<true>(flags, exec_mask, mode, d, a, b, c, trapsts);
}

// Multiply with lit special-value fixup for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_mullit_f32_w32_gfx12(uint64_t flags, uint32_t exec_mask, uint64_t mode,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c,
                                    [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run<false>(flags, exec_mask, mode, d, a, b, c, excp_flag_user);
}

// Multiply with lit special-value fixup for selected lanes.
// Return errors before changing outputs. FP status is accumulated only when resolved semantics
// require it.
int rj_arith_v_mullit_f32_w64_gfx12(uint64_t flags, uint64_t exec_mask, uint64_t mode,
                                    uint32_t *const *d, const uint32_t *const *a,
                                    const uint32_t *const *b, const uint32_t *const *c,
                                    [[maybe_unused]] uint32_t *excp_flag_user) {
  flags = rocjitsu::arith::resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = rocjitsu::arith::validate_semantics(flags)) {
    return error;
  }
  return run_wave64<false>(flags, exec_mask, mode, d, a, b, c, excp_flag_user);
}
