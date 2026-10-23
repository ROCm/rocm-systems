// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Compile scalar FMA repairs once with the library's baseline CPU settings.

#include "fp/scalar_repair.h"
#include "fp/fma_scalar.h"
#include "fp/half_fma_scalar.h"
#include "fp/mixed_fma_scalar.h"

namespace rocjitsu::arith {

// Apply the scalar FP32 instruction model to an exceptional SIMD lane.
[[gnu::noinline]] uint32_t repair_fma_f32(uint32_t a, uint32_t b, uint32_t c, uint64_t flags,
                                          uint32_t mode, bool rdna3, bool dx9_zero) {
  return fma_instruction_value(a, b, c, flags, mode, rdna3, dx9_zero);
}

// Apply the scalar FP16 model after the SIMD caller selects operand halves.
[[gnu::noinline]] uint16_t repair_fma_f16(uint16_t a, uint16_t b, uint16_t c, uint32_t mode,
                                          uint64_t flags, bool rdna3, uint32_t *status) {
  return rdna3 ? half_fma_value<true>(a, b, c, mode, flags, status)
               : half_fma_value<false>(a, b, c, mode, flags, status);
}

// Apply the inherited mixed-FMA model to one exceptional SIMD lane.
[[gnu::noinline]] uint32_t repair_mixed_fma(MixedFma destination, bool rdna3, uint64_t flags,
                                            uint32_t mode, uint32_t a, uint32_t b, uint32_t c,
                                            uint32_t d) {
  uint32_t operands[] = {a, b, c};
  for (int i = 0; i < 3; ++i) {
    operands[i] = mixed_fma_input(operands[i], mode >> i, mixed_fma_flush_input(flags, mode >> i));
  }
  if (destination == MixedFma::Float) {
    uint32_t value = fma_value_f32(operands[0], operands[1], operands[2], flags, rdna3);
    return mode & RJ_ARITH_ALU_CLAMP ? fma_clamp_f32(value, flags) : value;
  }
  int shift = destination == MixedFma::High ? 16 : 0;
  auto half =
      mixed_fma_half_value(operands[0], operands[1], operands[2], mode & RJ_ARITH_ALU_CLAMP, flags);
  return (d & ~(0xffffU << shift)) | (uint32_t(half) << shift);
}

// Resolve exceptional decoded mixed operands without emitting scalar model code in SIMD objects.
[[gnu::noinline]] uint16_t repair_mixed_special(uint32_t a, uint32_t b, uint32_t c) {
  return mixed_fma_special(a, b, c);
}

} // namespace rocjitsu::arith
