// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_NO_STDX_DENSE_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_NO_STDX_DENSE_H_

#include "rocjitsu/isa/arch/amdgpu/shared/mma/common.h"

namespace rocjitsu::amdgpu::mma_backend {

template <typename ExtractA, typename ExtractB>
bool try_exec_f32_mixed_simd(auto &, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                             auto &, auto &, auto &) {
  return false;
}

template <typename ExtractA, typename ExtractB>
bool try_exec_wmma_f32_mixed_simd(auto &, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                                  uint32_t, uint32_t, uint32_t, ExtractA, ExtractB, uint32_t,
                                  uint32_t, uint32_t, auto &) {
  return false;
}

template <typename ExtractA, typename ExtractB, typename ScaleAWord, typename ScaleBWord>
bool try_exec_wmma_f32_scaled_mixed_simd(auto &, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                                         uint32_t, uint32_t, uint32_t, ExtractA, ExtractB, uint32_t,
                                         ScaleAWord, ScaleBWord, uint32_t, uint32_t, uint32_t,
                                         uint32_t, bool, uint32_t, auto &, auto &, auto &) {
  return false;
}

template <typename ExtractA, typename ExtractB>
bool try_exec_swmmac_f32_mixed_simd(auto &, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                                    uint32_t, uint32_t, ExtractA, ExtractB, uint32_t, auto &,
                                    auto &, auto &, auto &) {
  return false;
}

template <typename ExtractA, typename ExtractB, typename ReadAcc, typename PackResult>
bool try_exec_wmma_packed16_simd(auto &, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                                 uint32_t, ExtractA, ExtractB, ReadAcc, PackResult, uint32_t,
                                 uint32_t, auto &) {
  return false;
}

template <typename ExtractA, typename ExtractB, typename ReadAcc, typename PackResult>
bool try_exec_swmmac_packed16_simd(auto &, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                                   uint32_t, ExtractA, ExtractB, PackResult, uint32_t, auto &,
                                   auto &, auto &, auto &) {
  return false;
}

template <typename ExtractA, typename ExtractB, typename ScaleBlock>
bool try_exec_f32_scaled_impl_simd(auto &, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                                   uint32_t, uint32_t, uint32_t, uint32_t, ExtractA, ExtractB,
                                   ScaleBlock, uint32_t, uint32_t, auto &, auto &, auto &) {
  return false;
}

inline bool try_exec_i32_i8_simd(auto &, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                                 uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, auto &, auto &) {
  return false;
}

template <typename ExtractA, typename ExtractB>
bool try_exec_wmma_i32_simd(auto &, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                            uint32_t, ExtractA, ExtractB, bool, uint32_t, uint32_t, auto &) {
  return false;
}

template <typename ExtractA, typename ExtractB>
bool try_exec_swmmac_i32_simd(auto &, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                              uint32_t, ExtractA, ExtractB, bool, uint32_t, uint32_t, auto &,
                              auto &, auto &) {
  return false;
}

inline bool try_exec_f64_simd(auto &, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                              uint32_t, uint32_t, auto &, auto &) {
  return false;
}

template <uint32_t M, uint32_t N, uint32_t K>
bool try_exec_wmma_f32_f32_spec_simd(auto &, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                                     uint32_t) {
  return false;
}

template <uint32_t M, uint32_t N, uint32_t K, uint32_t BATCH>
bool try_exec_f32_mfma_f32_spec_simd(auto &, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                                     uint32_t, uint32_t, uint32_t) {
  return false;
}

template <uint32_t M, uint32_t N, uint32_t K, uint32_t BATCH = 1>
bool try_exec_f32_mfma_f16_spec_simd(auto &, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                                     uint32_t, uint32_t, uint32_t) {
  return false;
}

template <uint32_t M, uint32_t N, uint32_t K, uint32_t BATCH = 1>
bool try_exec_f32_mfma_bf16_spec_simd(auto &, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                                      uint32_t, uint32_t, uint32_t) {
  return false;
}

} // namespace rocjitsu::amdgpu::mma_backend

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_NO_STDX_DENSE_H_
