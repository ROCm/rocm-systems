// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Compile scalar FMA repairs once with the library's baseline CPU settings.

#include "fp/scalar_repair.h"
#include "fp/fma_scalar.h"
#include "fp/half_fma_scalar.h"

namespace rocjitsu::arith {

// Apply the scalar FP32 instruction model to an exceptional SIMD lane.
[[gnu::noinline]] uint32_t repair_fma_f32(uint32_t a, uint32_t b, uint32_t c, uint64_t flags,
                                          uint32_t instruction_flags, bool rdna3) {
  return fma_instruction_value(a, b, c, flags, instruction_flags, rdna3);
}

// Apply the scalar FP16 model after the SIMD caller selects operand halves.
[[gnu::noinline]] uint16_t repair_fma_f16(uint16_t a, uint16_t b, uint16_t c, uint64_t flags,
                                          uint32_t instruction_flags, bool rdna3) {
  return rdna3 ? half_fma_value<true>(a, b, c, instruction_flags, flags)
               : half_fma_value<false>(a, b, c, instruction_flags, flags);
}

} // namespace rocjitsu::arith
