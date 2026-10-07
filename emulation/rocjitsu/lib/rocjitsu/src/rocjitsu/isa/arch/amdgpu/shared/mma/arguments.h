// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_ARGUMENTS_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_ARGUMENTS_H_

#include <cstdint>

namespace rocjitsu::amdgpu::mma_backend {

struct MatrixShape {
  uint32_t M;
  uint32_t N;
  uint32_t K;
};

} // namespace rocjitsu::amdgpu::mma_backend

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_ARGUMENTS_H_
