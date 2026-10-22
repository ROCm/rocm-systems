// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Shared flag validation, FP bit casts, and CPU-query state.
// Internal helpers shared by instruction implementations; not part of the public C API.

#ifndef RJ_ARITH_INTERNAL_H_
#define RJ_ARITH_INTERNAL_H_

#include "rj_arith/rj_arith.h"

#include <bit>
#include <stdint.h>
#include <type_traits>

namespace rocjitsu::arith {

// Reinterpret FP32 bits as float without conversion or aliasing violations.
inline float as_float(uint32_t v) { return std::bit_cast<float>(v); }

// Return the FP32 bit pattern without numeric conversion.
inline uint32_t as_bits(float f) { return std::bit_cast<uint32_t>(f); }

// Accept ordinary semantics modes 0..3; reject reserved modes, including mode 7.
inline int validate_semantics(uint64_t flags) {
  return (flags & RJ_ARITH_SEMANTICS_MASK) > RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL
             ? RJ_ARITH_ERROR_INVALID_FLAGS
             : RJ_ARITH_SUCCESS;
}

// Validate CPU, semantics, and guest FP flags without accessing operands or outputs.
inline int validate(uint64_t flags) {
  // Only the advertised CPU levels, semantics modes, and FP16_OVFL are accepted.
  constexpr uint64_t known = RJ_ARITH_CPU_MASK | RJ_ARITH_SEMANTICS_MASK | RJ_ARITH_FP16_OVFL;
  if ((flags & ~known) || (flags & RJ_ARITH_CPU_MASK) > RJ_ARITH_CPU_X86_64_ZEN4 ||
      validate_semantics(flags)) {
    return RJ_ARITH_ERROR_INVALID_FLAGS;
  }
  if ((flags & RJ_ARITH_SEMANTICS_MASK) >= RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
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
