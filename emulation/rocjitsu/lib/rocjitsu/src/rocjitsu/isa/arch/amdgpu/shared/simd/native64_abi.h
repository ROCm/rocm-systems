// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_NATIVE64_ABI_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_NATIVE64_ABI_H_

#include "util/simd.h"

namespace rocjitsu::amdgpu::simd_backend {

inline constexpr bool native64_masks = true;
using mixed_fma_abi = util::stdx::simd_abi::native<double>;

} // namespace rocjitsu::amdgpu::simd_backend

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_NATIVE64_ABI_H_
