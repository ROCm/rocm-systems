// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_RDNA4_MMA_EXEC_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_RDNA4_MMA_EXEC_H_

/// @file WMMA execution stubs for rdna4 (reuses shared MFMA register math).
///
/// RDNA WMMA uses Wave32 matrix tiles with a different register layout than
/// CDNA MFMA. The shared MMA register mapping functions assume
/// Wave64; RDNA-specific register layout is a Phase C.8 deliverable.

#include "rocjitsu/isa/arch/amdgpu/shared/mma_exec.h"

namespace rocjitsu {
namespace rdna4 {

/// RDNA WMMA resolve_acc — uses Unified mode (no separate AccVGPR file).
template <typename F>
inline uint32_t resolve_acc(uint32_t vb, uint32_t dst, int src2_ev, uint32_t &const_acc,
                            F &&get_const) {
  return amdgpu::resolve_acc<amdgpu::AccMode::Unified>(vb, dst, src2_ev, const_acc,
                                                       std::forward<F>(get_const));
}

template <typename ExtractA, typename ExtractB>
void exec_swmmac_f32_mixed(amdgpu::ComputeUnitCore &cu, uint32_t M, uint32_t N, uint32_t K,
                           uint32_t B, uint32_t a_bits, uint32_t b_bits, uint32_t dst, uint32_t s0,
                           uint32_t s1, uint32_t acc_base, uint32_t index_base,
                           uint32_t index_entries, uint32_t index_key, ExtractA ea, ExtractB eb) {
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t val;
  };
  std::vector<Result> results;
  results.reserve(M * N * B);

  const uint32_t compressed_k = K / 2;
  for (uint32_t b = 0; b < B; ++b) {
    for (uint32_t row = 0; row < M; ++row) {
      for (uint32_t col = 0; col < N; ++col) {
        auto out = amdgpu::output_loc_32(M, N, row, col, b);
        float acc = std::bit_cast<float>(cu.read_vgpr(acc_base + out.reg, out.lane));
        for (uint32_t ck = 0; ck < compressed_k; ++ck) {
          auto al = amdgpu::input_loc(M, compressed_k, B, row, ck, b, a_bits);
          const uint32_t local_ck = amdgpu::input_local_element(al);
          const uint64_t index_set =
              amdgpu::read_swmmac_index_set(cu, index_base, al.lane, index_entries, index_key);
          const uint32_t dense_k = amdgpu::swmmac_dense_k(index_set, ck, local_ck);
          auto bl = amdgpu::input_loc(N, K, B, col, dense_k, b, b_bits);
          acc += ea(cu, s0, al) * eb(cu, s1, bl);
        }
        results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(acc)});
      }
    }
  }
  for (const auto &r : results)
    cu.write_vgpr(dst + r.reg, r.lane, r.val);
}

template <typename ExtractA, typename ExtractB>
void exec_swmmac_f32(amdgpu::ComputeUnitCore &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t B,
                     uint32_t in_bits, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t acc_base,
                     uint32_t index_base, uint32_t index_entries, uint32_t index_key, ExtractA ea,
                     ExtractB eb) {
  exec_swmmac_f32_mixed(cu, M, N, K, B, in_bits, in_bits, dst, s0, s1, acc_base, index_base,
                        index_entries, index_key, ea, eb);
}

template <typename ExtractA, typename ExtractB>
void exec_swmmac_i32(amdgpu::ComputeUnitCore &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t B,
                     uint32_t in_bits, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t acc_base,
                     uint32_t index_base, uint32_t index_entries, uint32_t index_key, ExtractA ea,
                     ExtractB eb, bool clamp) {
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t val;
  };
  std::vector<Result> results;
  results.reserve(M * N * B);

  const uint32_t compressed_k = K / 2;
  for (uint32_t b = 0; b < B; ++b) {
    for (uint32_t row = 0; row < M; ++row) {
      for (uint32_t col = 0; col < N; ++col) {
        auto out = amdgpu::output_loc_32(M, N, row, col, b);
        int64_t acc =
            static_cast<int64_t>(static_cast<int32_t>(cu.read_vgpr(acc_base + out.reg, out.lane)));
        for (uint32_t ck = 0; ck < compressed_k; ++ck) {
          auto al = amdgpu::input_loc(M, compressed_k, B, row, ck, b, in_bits);
          const uint32_t local_ck = amdgpu::input_local_element(al);
          const uint64_t index_set =
              amdgpu::read_swmmac_index_set(cu, index_base, al.lane, index_entries, index_key);
          const uint32_t dense_k = amdgpu::swmmac_dense_k(index_set, ck, local_ck);
          auto bl = amdgpu::input_loc(N, K, B, col, dense_k, b, in_bits);
          acc += static_cast<int64_t>(ea(cu, s0, al)) * static_cast<int64_t>(eb(cu, s1, bl));
        }
        results.push_back({out.reg, out.lane, amdgpu::pack_i32_acc(acc, clamp)});
      }
    }
  }
  for (const auto &r : results)
    cu.write_vgpr(dst + r.reg, r.lane, r.val);
}

} // namespace rdna4
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_RDNA4_MMA_EXEC_H_
