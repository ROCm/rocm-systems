// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_FP_CONTROL_X86_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_FP_CONTROL_X86_H_

#include <cstdint>
#include <xmmintrin.h>

namespace rocjitsu::amdgpu::fp_mode::detail {

struct HostFpControl {
  using State = uint32_t;
  static State save() { return _mm_getcsr(); }
  static void clear_denormals() {
    constexpr uint32_t kDazMask = 1u << 6;
    constexpr uint32_t kFtzMask = 1u << 15;
    _mm_setcsr(_mm_getcsr() & ~(kDazMask | kFtzMask));
  }
  static void restore(State state) { _mm_setcsr(state); }
  static bool native_arithmetic_matches() {
    // MXCSR rounding can differ from the x87 rounding reported by fegetround().
    return (_mm_getcsr() & ((1u << 6) | (1u << 15) | _MM_ROUND_MASK)) == 0;
  }
};

} // namespace rocjitsu::amdgpu::fp_mode::detail

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_FP_CONTROL_X86_H_
