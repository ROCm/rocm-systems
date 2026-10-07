// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_MATH64_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_MATH64_H_

#if UTIL_SIMD_BROKEN_NATIVE_64BIT_MASKS
#include "rocjitsu/isa/arch/amdgpu/shared/simd/compat64_math.h"
#else
#include "rocjitsu/isa/arch/amdgpu/shared/simd/native64_math.h"
#endif

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_MATH64_H_
