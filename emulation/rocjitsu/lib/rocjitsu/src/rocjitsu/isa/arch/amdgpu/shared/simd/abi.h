// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_ABI_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_ABI_H_

#include "util/simd.h"

#if UTIL_SIMD_BROKEN_NATIVE_64BIT_MASKS
#include "rocjitsu/isa/arch/amdgpu/shared/simd/compat64_abi.h"
#else
#include "rocjitsu/isa/arch/amdgpu/shared/simd/native64_abi.h"
#endif

namespace rocjitsu::amdgpu::simd_backend {
#if defined(__FMA__)
inline constexpr bool has_fma = true;
#else
inline constexpr bool has_fma = false;
#endif
} // namespace rocjitsu::amdgpu::simd_backend

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_ABI_H_
