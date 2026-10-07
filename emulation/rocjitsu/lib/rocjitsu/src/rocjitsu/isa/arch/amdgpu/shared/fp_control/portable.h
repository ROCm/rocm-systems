// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_FP_CONTROL_PORTABLE_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_FP_CONTROL_PORTABLE_H_

namespace rocjitsu::amdgpu::fp_mode::detail {

struct HostFpControl {
  struct State {};
  static State save() { return {}; }
  static void clear_denormals() {}
  static void restore(State) {}
  static bool native_arithmetic_matches() { return false; }
};

} // namespace rocjitsu::amdgpu::fp_mode::detail

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_FP_CONTROL_PORTABLE_H_
