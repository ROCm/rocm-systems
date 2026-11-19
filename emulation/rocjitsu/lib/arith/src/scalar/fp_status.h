// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Scalar FP exception classification, inherited from Rocjitsu shared/fp_status.h with the
// hardware overflow/tininess rule fitted to gfx1201 captures.
// Numerical causes belong here; committing guest registers and traps belongs to the VM.

#pragma once

#include "common/float_encoding.h"
#include "common/validation.h"
#include "fp/fma_integer.h"
#include "fp/rounding_status.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Decode one scalar FP16/FP32 encoding without changing host FP controls.
template <bool Half> inline double scalar_fp_value(uint32_t bits) {
  return Half ? double(f16_to_float(uint16_t(bits))) : double(as_float(bits));
}

// Preserve scalar input classification: flushing suppresses denormal reporting.
template <bool Half> inline uint32_t scalar_input_causes(uint32_t bits, uint64_t flags) {
  constexpr uint32_t magnitude = Half ? 0x7fff : 0x7fffffff;
  constexpr uint32_t infinity = Half ? 0x7c00 : 0x7f800000;
  constexpr uint32_t quiet = Half ? 0x200 : 0x400000;
  constexpr uint32_t normal = Half ? 0x400 : 0x800000;
  bits &= magnitude;
  if (bits > infinity && !(bits & quiet)) {
    return RJ_ARITH_EXCEPTION_INVALID;
  }
  return !(flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) && bits && bits < normal
             ? RJ_ARITH_EXCEPTION_INPUT_DENORM
             : 0;
}

// Classify the scalar FMA model, also used for ADD/SUB/MUL lowering, with the shared
// hardware overflow/tininess rule of fp/rounding_status.h.
// Preserve the inherited suppression of input-denormal causes for NaN results.
template <bool Half>
inline uint32_t scalar_fma_causes(uint64_t flags, uint32_t a, uint32_t b, uint32_t c,
                                  uint32_t result) {
  constexpr uint32_t sign = Half ? 0x8000 : 0x80000000;
  constexpr uint32_t magnitude = sign - 1;
  constexpr uint32_t infinity = Half ? 0x7c00 : 0x7f800000;
  constexpr uint32_t normal = Half ? 0x400 : 0x800000;
  uint32_t inputs[] = {a, b, c};
  uint32_t causes = 0;
  for (auto &input : inputs) {
    input &= sign | magnitude;
    if ((flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) && (input & magnitude) < normal) {
      input &= sign;
    }
    causes |= scalar_input_causes<Half>(input, flags);
  }
  a = inputs[0];
  b = inputs[1];
  c = inputs[2];
  const uint32_t x = a & magnitude, y = b & magnitude, z = c & magnitude;
  if ((x == 0 && y == infinity) || (x == infinity && y == 0) ||
      ((x == infinity || y == infinity) && x <= infinity && y <= infinity && z == infinity &&
       ((a ^ b ^ c) & sign))) {
    causes |= RJ_ARITH_EXCEPTION_INVALID;
  }
  if (x < infinity && y < infinity && z < infinity) {
    // Exact round-to-odd FP64 intermediate; FP16 operands widen exactly to FP32.
    auto widen = [](uint32_t bits) { return Half ? as_bits(f16_to_float(uint16_t(bits))) : bits; };
    const uint64_t exact = fma_finite_round_odd(widen(a), widen(b), widen(c));
    if (exact & 0x7fffffffffffffffULL) {
      causes |= rounding_causes<Half>(exact, result, (flags & RJ_ARITH_FP_ROUND_MASK) >> 24);
    }
  }
  if ((result & magnitude) > infinity) {
    causes &= ~RJ_ARITH_EXCEPTION_INPUT_DENORM;
  }
  return causes;
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
