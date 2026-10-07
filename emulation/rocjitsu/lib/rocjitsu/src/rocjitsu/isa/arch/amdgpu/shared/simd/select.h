// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_SELECT_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_SELECT_H_

#include "rocjitsu/isa/arch/amdgpu/shared/simd/abi.h"
#include "rocjitsu/isa/arch/amdgpu/shared/simd/common.h"

#include "rocjitsu/isa/arch/amdgpu/shared/simd/math64.h"

#include "rocjitsu/isa/arch/amdgpu/shared/simd/execute.h"
#include "rocjitsu/isa/arch/amdgpu/shared/simd/portable_math.h"

#if UTIL_SIMD_BROKEN_NATIVE_64BIT_MASKS
#include "rocjitsu/isa/arch/amdgpu/shared/simd/compat64_dot.h"
#else
#include "rocjitsu/isa/arch/amdgpu/shared/simd/native64_dot.h"
#endif

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_SELECT_H_
