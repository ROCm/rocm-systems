// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_PORTABLE_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_PORTABLE_H_

#include "rocjitsu/isa/arch/amdgpu/shared/mma/common.h"

namespace rocjitsu {
namespace amdgpu {

/// Keep eligibility checks outside the stack-heavy implementation so
/// forced-scalar and unsupported executions avoid its scratch frame.
template <typename ExtractA, typename ExtractB>
[[gnu::always_inline]] inline bool try_exec_swmmac_16x16x128_8bit(
    auto &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t in_bits, uint32_t dst, uint32_t s0,
    uint32_t s1, uint32_t acc_base, uint32_t index_base, uint32_t index_entries, uint32_t index_key,
    ExtractA, ExtractB, SwmmacK128Result result_format, uint32_t const_acc = ACC_FROM_VGPR,
    uint32_t wave_size = WMMA_WAVE32, bool fp16_ovfl = false) {

  (void)cu;
  (void)M;
  (void)N;
  (void)K;
  (void)in_bits;
  (void)dst;
  (void)s0;
  (void)s1;
  (void)acc_base;
  (void)index_base;
  (void)index_entries;
  (void)index_key;
  (void)result_format;
  (void)const_acc;
  (void)wave_size;
  (void)fp16_ovfl;
  return false;
}

/// Keep eligibility checks outside the stack-heavy implementation so
/// forced-scalar and non-AVX-512 executions avoid its scratch frame.
template <SmfmacLayout Layout, uint32_t M, uint32_t N, uint32_t K, typename ExtractA,
          typename ExtractB>
[[gnu::always_inline]] inline bool smfmac_try_avx512(auto &cu, uint32_t dst, uint32_t s0,
                                                     uint32_t s1, uint32_t idx_base, ExtractA,
                                                     ExtractB) {
  (void)cu;
  (void)dst;
  (void)s0;
  (void)s1;
  (void)idx_base;
  return false;
}

[[gnu::always_inline]] inline bool
try_exec_swmmac_16x16x64_16bit(auto &, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                               uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, SwmmacK64Input,
                               SwmmacK64Accumulator, SwmmacK64Result, uint32_t, uint32_t,
                               bool = false) {
  return false;
}

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_PORTABLE_H_
