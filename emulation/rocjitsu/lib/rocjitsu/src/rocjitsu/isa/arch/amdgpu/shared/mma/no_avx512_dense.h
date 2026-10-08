// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_NO_AVX512_DENSE_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_NO_AVX512_DENSE_H_

#include "rocjitsu/isa/arch/amdgpu/shared/mma/common.h"

namespace rocjitsu::amdgpu::mma_backend {

inline bool try_exec_wmma_f32_16x16x32_f16_simd(auto &, uint32_t, uint32_t, uint32_t, uint32_t,
                                                uint32_t, uint32_t) {
  return false;
}

inline bool try_exec_wmma_f32_16x16x32_bf16_simd(auto &, uint32_t, uint32_t, uint32_t, uint32_t,
                                                 uint32_t, uint32_t) {
  return false;
}

template <uint32_t M, uint32_t N, uint32_t K, bool A_FP8, bool B_FP8, bool FNUZ = false>
bool try_exec_wmma_f32_f8_spec_simd(auto &, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                                    uint32_t) {
  return false;
}

inline bool try_exec_wmma_bf16f32_16x16x32_bf16_simd(auto &, uint32_t, uint32_t, uint32_t, uint32_t,
                                                     uint32_t, uint32_t) {
  return false;
}

template <uint32_t M, uint32_t N, uint32_t K>
bool try_exec_wmma_f16_spec_simd(auto &, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) {
  return false;
}

template <uint32_t M, uint32_t N, uint32_t K>
bool try_exec_wmma_bf16_spec_simd(auto &, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) {
  return false;
}

template <uint32_t M, uint32_t N, uint32_t K, bool A_FP8, bool B_FP8, bool FNUZ = false>
bool try_exec_wmma_f16_f8_spec_simd(auto &, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                                    bool) {
  return false;
}

inline bool try_exec_wmma_i32_16x16x64_iu8_simd(auto &, uint32_t, uint32_t, uint32_t, uint32_t,
                                                bool, bool, bool, uint32_t) {
  return false;
}

template <uint32_t M, uint32_t N, uint32_t K, bool A_FP8, bool B_FP8, bool FNUZ = false>
bool try_exec_f32_mfma_f8_spec_simd(auto &, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                                    uint32_t, uint32_t, uint32_t) {
  return false;
}

template <uint32_t M, uint32_t N, uint32_t K, uint32_t BATCH = 1>
bool try_exec_i32_mfma_i8_spec_simd(auto &, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) {
  return false;
}

} // namespace rocjitsu::amdgpu::mma_backend

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_NO_AVX512_DENSE_H_
