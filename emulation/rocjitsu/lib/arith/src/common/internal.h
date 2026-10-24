// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Shared flag validation, semantics resolution, FP bit casts, and CPU-query state.
// Internal helpers shared by instruction implementations; not part of the public C API.

#ifndef RJ_ARITH_INTERNAL_H_
#define RJ_ARITH_INTERNAL_H_

#include "rj_arith/rj_arith.h"

#include <bit>
#include <stdint.h>
#include <type_traits>

namespace rocjitsu::arith {

// Floating matrix arithmetic uses its fixed model, ignoring guest rounding,
// denormal, IEEE and DX10 controls (WMMA has no CLAMP modifier).
// FP16_OVFL is accepted separately and has no effect on FP32 matrix results.
inline constexpr uint64_t kMatrixIgnoredFpFlags =
    RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | RJ_ARITH_FP_IEEE_MODE |
    RJ_ARITH_FP_ROUND_MASK | RJ_ARITH_FP_DX10_CLAMP_OFF;

// Supported instruction fields for the FMA and matrix families.
inline constexpr uint32_t kFma32InstructionFlags =
    RJ_ARITH_ALU_NEG_A | RJ_ARITH_ALU_NEG_B | RJ_ARITH_ALU_NEG_C | RJ_ARITH_ALU_ABS_A |
    RJ_ARITH_ALU_ABS_B | RJ_ARITH_ALU_ABS_C | RJ_ARITH_ALU_OMOD_MASK | RJ_ARITH_ALU_CLAMP;
inline constexpr uint32_t kFma16InstructionFlags = kFma32InstructionFlags | RJ_ARITH_ALU_HIGH_A |
                                                   RJ_ARITH_ALU_HIGH_B | RJ_ARITH_ALU_HIGH_C |
                                                   RJ_ARITH_ALU_HIGH_D;
inline constexpr uint32_t kWmmaInstructionFlags = RJ_ARITH_WMMA_NEG_LO_A | RJ_ARITH_WMMA_NEG_LO_B |
                                                  RJ_ARITH_WMMA_NEG_C | RJ_ARITH_WMMA_NEG_HI_A |
                                                  RJ_ARITH_WMMA_NEG_HI_B | RJ_ARITH_WMMA_ABS_C;

// EXEC has one bit per lane in the instruction wave.
template <int Lanes>
  requires(Lanes == 32 || Lanes == 64)
using ExecMask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;

// Reinterpret FP32 bits as float without conversion or aliasing violations.
inline float as_float(uint32_t v) { return std::bit_cast<float>(v); }

// Return the FP32 bit pattern without numeric conversion.
inline uint32_t as_bits(float f) { return std::bit_cast<uint32_t>(f); }

// Resolve the legacy rocjitsu policy selected by this entry point's wrapper.
// Explicit semantics are never reinterpreted.
inline uint64_t resolve_compatibility(uint64_t flags, uint64_t semantics) {
  return (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY
             ? (flags & ~RJ_ARITH_SEMANTICS_MASK) | semantics
             : flags;
}

// These queries operate on resolved semantics; validation rejects reserved values.
inline bool is_exact(uint64_t flags) {
  const uint64_t semantics = flags & RJ_ARITH_SEMANTICS_MASK;
  return semantics >= RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE &&
         semantics <= RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL;
}

// Return whether resolved semantics require guest FP status reporting.
inline bool reports_fp_status(uint64_t flags) {
  return (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL;
}

// Accept ordinary semantics modes 0..3; reject reserved or unresolved compatibility modes.
inline int validate_semantics(uint64_t flags) {
  return (flags & RJ_ARITH_SEMANTICS_MASK) > RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL
             ? RJ_ARITH_ERROR_INVALID_FLAGS
             : RJ_ARITH_SUCCESS;
}

// Validate CPU, semantics, instruction bits, and supported guest FP flags. Return an error
// without accessing operands or outputs.
inline int validate(uint64_t flags, uint32_t instruction_flags, bool supports_exact = false,
                    uint64_t supported_fp_flags = 0) {
  // Only advertised guest FP controls are accepted; compatibility must already be resolved by
  // the entry point.
  constexpr uint64_t known = RJ_ARITH_CPU_MASK | RJ_ARITH_SEMANTICS_MASK | RJ_ARITH_FP16_OVFL;
  if ((flags & ~(known | supported_fp_flags)) ||
      (flags & RJ_ARITH_CPU_MASK) > RJ_ARITH_CPU_X86_64_ZEN4 || instruction_flags ||
      validate_semantics(flags)) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if (!supports_exact &&
      (flags & RJ_ARITH_SEMANTICS_MASK) >= RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  return RJ_ARITH_SUCCESS;
}

// Snapshot of CPUID and XCR0 for deterministic OS-state/feature decoder tests.
struct CpuState {
  uint32_t leaf1_ecx = 0, leaf7_ebx = 0, leaf7_ecx = 0, leaf71_eax = 0, ext1_ecx = 0;
  uint64_t xcr0 = 0;
};

// Return the highest supported CPU level from captured CPUID/XCR0 state; performs no host
// queries.
uint64_t decode_cpu(const CpuState &s);

} // namespace rocjitsu::arith

#endif
