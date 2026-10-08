// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_FP_CONTROL_SELECT_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_FP_CONTROL_SELECT_H_

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include "rocjitsu/isa/arch/amdgpu/shared/fp_control/x86.h"
#elif defined(__aarch64__)
#include "rocjitsu/isa/arch/amdgpu/shared/fp_control/aarch64.h"
#else
#include "rocjitsu/isa/arch/amdgpu/shared/fp_control/portable.h"
#endif

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_FP_CONTROL_SELECT_H_
