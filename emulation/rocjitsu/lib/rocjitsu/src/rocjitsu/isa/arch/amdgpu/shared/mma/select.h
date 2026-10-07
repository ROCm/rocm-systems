// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_SELECT_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_SELECT_H_

#if __has_include(<experimental/simd>)
#include "rocjitsu/isa/arch/amdgpu/shared/mma/portable_dense.h"
#else
#include "rocjitsu/isa/arch/amdgpu/shared/mma/no_stdx_dense.h"
#endif

#if defined(__AVX512F__) && __has_include(<experimental/simd>)
#include "rocjitsu/isa/arch/amdgpu/shared/mma/avx512_dense.h"
#include "rocjitsu/isa/arch/amdgpu/shared/mma/avx512_smfmac.h"
#include "rocjitsu/isa/arch/amdgpu/shared/mma/avx512_swmmac_k128.h"
#include "rocjitsu/isa/arch/amdgpu/shared/mma/avx512_swmmac_k64.h"
#else
#include "rocjitsu/isa/arch/amdgpu/shared/mma/no_avx512_dense.h"
#include "rocjitsu/isa/arch/amdgpu/shared/mma/portable.h"
#endif

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_SELECT_H_
