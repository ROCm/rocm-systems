// SPDX-License-Identifier: MIT

// Shared flag validation, semantics resolution, FP bit casts, and CPU-query state.
// Internal helpers shared by instruction implementations; not part of the public C API.

#ifndef RJ_ARITH_INTERNAL_H_
#define RJ_ARITH_INTERNAL_H_

#include "rj_arith/rj_arith.h"

#include <cstring>
#include <stdint.h>
#include <type_traits>

namespace rocjitsu::arith {

// Floating matrix arithmetic ignores guest denormal and IEEE controls.
// FP16 finite overflow remains a separate, effective packed-output setting.
inline constexpr uint64_t matrix_ignored_fp_flags =
    RJ_ARITH_FP_FLUSH_INPUT_DENORMALS | RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS | RJ_ARITH_FP_IEEE_MODE;

// EXEC has one bit per lane in the instruction wave.
template <int Lanes> using ExecMask = std::conditional_t<Lanes == 32, uint32_t, uint64_t>;

// Reinterpret FP32 bits as float without conversion or aliasing violations.
inline float as_float(uint32_t v) {
  float f;
  std::memcpy(&f, &v, sizeof(f));
  return f;
}

// Return the FP32 bit pattern without numeric conversion.
inline uint32_t as_bits(float f) {
  uint32_t v;
  std::memcpy(&v, &f, sizeof(v));
  return v;
}

// Temporary migration policy, resolved at the public instruction boundary before
// validation or composition. The selected mode comes from rocjitsu's pre-unification
// instruction policy (parent of fef6569e8ec). Never reinterpret an explicit mode.
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

// A composed instruction may calculate its own status around a numeric sub-operation.
inline uint64_t without_fp_status(uint64_t flags) {
  return reports_fp_status(flags) ? (flags & ~RJ_ARITH_SEMANTICS_MASK) |
                                        RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS
                                  : flags;
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
