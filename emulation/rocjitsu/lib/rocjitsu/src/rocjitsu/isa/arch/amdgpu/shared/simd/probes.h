// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_PROBES_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_PROBES_H_

#include "util/simd.h"

#if UTIL_SIMD_BROKEN_NATIVE_64BIT_MASKS
#include "rocjitsu/isa/arch/amdgpu/shared/simd/compat64_probes.h"
#else
#include "rocjitsu/isa/arch/amdgpu/shared/simd/native64_probes.h"
#endif

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_PROBES_H_
