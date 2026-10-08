// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_FP_CONTROL_AARCH64_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_FP_CONTROL_AARCH64_H_

#include <cstdint>

namespace rocjitsu::amdgpu::fp_mode::detail {

inline uint64_t read_fpcr() {
  uint64_t value;
  __asm__ volatile("mrs %0, fpcr" : "=r"(value) : : "memory");
  return value;
}

inline void write_fpcr(uint64_t value) {
  __asm__ volatile("msr fpcr, %0" : : "r"(value) : "memory");
}

struct HostFpControl {
  using State = uint64_t;
  static State save() { return read_fpcr(); }
  static void clear_denormals() {
    constexpr uint64_t kFizMask = uint64_t{1} << 0;
    constexpr uint64_t kFz16Mask = uint64_t{1} << 19;
    constexpr uint64_t kFzMask = uint64_t{1} << 24;
    write_fpcr(read_fpcr() & ~(kFizMask | kFz16Mask | kFzMask));
  }
  static void restore(State state) { write_fpcr(state); }
  static bool native_arithmetic_matches() {
    return (read_fpcr() & ((uint64_t{1} << 0) | (uint64_t{1} << 19) | (uint64_t{1} << 24))) == 0;
  }
};

} // namespace rocjitsu::amdgpu::fp_mode::detail

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_FP_CONTROL_AARCH64_H_
