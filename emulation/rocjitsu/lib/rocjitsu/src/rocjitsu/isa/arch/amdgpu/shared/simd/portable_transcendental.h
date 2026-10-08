// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_PORTABLE_TRANSCENDENTAL_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_PORTABLE_TRANSCENDENTAL_H_

#include "rocjitsu/isa/arch/amdgpu/shared/simd/common_transcendental.h"
#include "util/simd.h"

namespace rocjitsu::amdgpu::transcendental {

/// @brief Evaluate a native batch with one saved host floating-point environment.
template <bool Logarithm>
inline util::native<float> log_exp_f16_simd(util::native<float> x, uint32_t denorm_mode,
                                            bool fp16_ovfl, bool quiet_snan) {
  fp_mode::ScopedEnvironment environment(0);
  return util::map_native_convert_scalar<float, float>(x, [=](float lane) {
    return detail::log_exp_f16_nearest<Logarithm>(lane, denorm_mode, fp16_ovfl, quiet_snan);
  });
}

template <HalfOperation Op>
inline util::native<float> map_f16_simd(util::native<float> source, uint32_t denorm_mode,
                                        bool fp16_ovfl, bool quiet_snan) {
  return util::map_native_convert_scalar<float, float>(
      source, [=](float lane) { return map_f16<Op>(lane, denorm_mode, fp16_ovfl, quiet_snan); });
}

} // namespace rocjitsu::amdgpu::transcendental

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_PORTABLE_TRANSCENDENTAL_H_
