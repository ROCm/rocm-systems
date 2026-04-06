// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_CDNA4_MFMA_EXEC_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_CDNA4_MFMA_EXEC_H_

/// @file CDNA4 Matrix Fused Multiply-Add (MFMA) register mapping and execution.
///
/// Uses the same GFX9 register layout formulas as CDNA3. The key conventions:
///
///   - Output D[i][j]: i is the register dimension (matrix column),
///     j is the lane dimension (matrix row). Call output_loc with
///     i=column, j=row to get the physical (vgpr_offset, lane).
///
///   - Input A[row][k]: use input_loc(dim=M, K, B, i=row, k, b, bits).
///     Input B[col][k]: use input_loc(dim=N, K, B, i=col, k, b, bits).

#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "util/data_types.h"

#include <bit>
#include <cstdint>
#include <vector>

namespace rocjitsu {
namespace cdna4 {
namespace mfma {

struct InputLoc {
  uint32_t vgpr_offset;
  uint32_t lane;
  uint32_t sub_element;
};

struct OutputLoc {
  uint32_t reg;
  uint32_t lane;
};

inline uint32_t dst_base(uint32_t vb, int ev) { return vb + static_cast<uint32_t>(ev); }

inline uint32_t src_base(uint32_t vb, int ev) {
  if (ev >= 768)
    return vb + static_cast<uint32_t>(ev - 768);
  return (ev >= 256) ? vb + static_cast<uint32_t>(ev - 256) : vb + static_cast<uint32_t>(ev);
}

constexpr uint32_t ACC_FROM_VGPR = UINT32_MAX;

template <typename F>
inline uint32_t resolve_acc(uint32_t vb, uint32_t dst, int src2_ev, uint32_t &const_acc,
                            F &&get_const) {
  if (src2_ev >= 768 && src2_ev <= 1023) {
    const_acc = ACC_FROM_VGPR;
    return vb + static_cast<uint32_t>(src2_ev - 768);
  }
  if (src2_ev >= 256 && src2_ev <= 511) {
    const_acc = ACC_FROM_VGPR;
    return vb + static_cast<uint32_t>(src2_ev - 256);
  }
  const_acc = get_const();
  return dst;
}

inline InputLoc input_loc(uint32_t dim, uint32_t K, uint32_t B, uint32_t i, uint32_t k, uint32_t b,
                          uint32_t data_bits) {
  uint32_t lanes_per_block = 64 / (dim * B);
  uint32_t elems_per_group = K / lanes_per_block;

  uint32_t local = k % elems_per_group;
  uint32_t lane = b * dim + (k / elems_per_group) * dim * B + i;

  if (data_bits == 64)
    return {local * 2, lane, 0};
  if (data_bits == 32)
    return {local, lane, 0};
  uint32_t per_dword = 32 / data_bits;
  return {local / per_dword, lane, local % per_dword};
}

inline OutputLoc output_loc_32(uint32_t M, uint32_t N, uint32_t i, uint32_t j, uint32_t b) {
  uint32_t multirows = 64 / N;
  uint32_t mn_div_4 = (M * N) / 4;
  uint32_t blocks_per_reg = (64 + mn_div_4 - 1) / mn_div_4;

  uint32_t reg = b * ((M * N) / 64) + (i / (4 * multirows)) * 4 + (i % 4);
  uint32_t lane = (b % blocks_per_reg) * N + ((i / 4) % multirows) * blocks_per_reg * N + j;
  return {reg, lane};
}

inline OutputLoc output_loc_64(uint32_t M, uint32_t N, uint32_t i, uint32_t j, uint32_t b) {
  uint32_t multirows = 64 / N;
  uint32_t mn = M * N;
  uint32_t blocks_per_reg = (mn > 0) ? (64 + mn - 1) / mn : 1;

  uint32_t local = b * (mn / 64) + (i / multirows);
  uint32_t lane = (b % blocks_per_reg) * N + (i % multirows) * blocks_per_reg * N + j;
  return {local * 2, lane};
}

inline float extract_f32(amdgpu::ComputeUnitCore &cu, uint32_t base, const InputLoc &loc) {
  return std::bit_cast<float>(cu.read_vgpr(base + loc.vgpr_offset, loc.lane));
}

inline float extract_f16(amdgpu::ComputeUnitCore &cu, uint32_t base, const InputLoc &loc) {
  uint32_t raw = cu.read_vgpr(base + loc.vgpr_offset, loc.lane);
  return util::f16_to_f32(static_cast<uint16_t>((raw >> (loc.sub_element * 16)) & 0xFFFF));
}

inline float extract_bf16(amdgpu::ComputeUnitCore &cu, uint32_t base, const InputLoc &loc) {
  uint32_t raw = cu.read_vgpr(base + loc.vgpr_offset, loc.lane);
  return util::bf16_to_f32(static_cast<uint16_t>((raw >> (loc.sub_element * 16)) & 0xFFFF));
}

inline int32_t extract_i8(amdgpu::ComputeUnitCore &cu, uint32_t base, const InputLoc &loc) {
  uint32_t raw = cu.read_vgpr(base + loc.vgpr_offset, loc.lane);
  return static_cast<int32_t>(static_cast<int8_t>((raw >> (loc.sub_element * 8)) & 0xFF));
}

inline float extract_fp8(amdgpu::ComputeUnitCore &cu, uint32_t base, const InputLoc &loc) {
  uint32_t raw = cu.read_vgpr(base + loc.vgpr_offset, loc.lane);
  return util::fp8_e4m3_to_f32(static_cast<uint8_t>((raw >> (loc.sub_element * 8)) & 0xFF));
}

inline float extract_bf8(amdgpu::ComputeUnitCore &cu, uint32_t base, const InputLoc &loc) {
  uint32_t raw = cu.read_vgpr(base + loc.vgpr_offset, loc.lane);
  return util::bf8_e5m2_to_f32(static_cast<uint8_t>((raw >> (loc.sub_element * 8)) & 0xFF));
}

inline double extract_f64(amdgpu::ComputeUnitCore &cu, uint32_t base, const InputLoc &loc) {
  uint32_t lo = cu.read_vgpr(base + loc.vgpr_offset, loc.lane);
  uint32_t hi = cu.read_vgpr(base + loc.vgpr_offset + 1, loc.lane);
  return std::bit_cast<double>(static_cast<uint64_t>(hi) << 32 | lo);
}

template <typename ExtractA, typename ExtractB>
void exec_f32(amdgpu::ComputeUnitCore &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t B,
              uint32_t in_bits, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2, ExtractA ea,
              ExtractB eb, uint32_t const_acc = ACC_FROM_VGPR) {
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t val;
  };
  std::vector<Result> results;
  results.reserve(M * N * B);
  for (uint32_t b = 0; b < B; ++b) {
    for (uint32_t row = 0; row < M; ++row) {
      for (uint32_t col = 0; col < N; ++col) {
        auto out = output_loc_32(M, N, col, row, b);
        float acc = (const_acc != ACC_FROM_VGPR)
                        ? std::bit_cast<float>(const_acc)
                        : std::bit_cast<float>(cu.read_vgpr(s2 + out.reg, out.lane));
        for (uint32_t k = 0; k < K; ++k) {
          auto al = input_loc(M, K, B, row, k, b, in_bits);
          auto bl = input_loc(N, K, B, col, k, b, in_bits);
          acc += ea(cu, s0, al) * eb(cu, s1, bl);
        }
        results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(acc)});
      }
    }
  }
  for (const auto &r : results)
    cu.write_vgpr(dst + r.reg, r.lane, r.val);
}

inline void exec_i32_i8(amdgpu::ComputeUnitCore &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t B,
                        uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                        uint32_t const_acc = ACC_FROM_VGPR) {
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t val;
  };
  std::vector<Result> results;
  results.reserve(M * N * B);
  for (uint32_t b = 0; b < B; ++b) {
    for (uint32_t row = 0; row < M; ++row) {
      for (uint32_t col = 0; col < N; ++col) {
        auto out = output_loc_32(M, N, col, row, b);
        int32_t acc = (const_acc != ACC_FROM_VGPR)
                          ? static_cast<int32_t>(const_acc)
                          : static_cast<int32_t>(cu.read_vgpr(s2 + out.reg, out.lane));
        for (uint32_t k = 0; k < K; ++k) {
          auto al = input_loc(M, K, B, row, k, b, 8);
          auto bl = input_loc(N, K, B, col, k, b, 8);
          acc += extract_i8(cu, s0, al) * extract_i8(cu, s1, bl);
        }
        results.push_back({out.reg, out.lane, static_cast<uint32_t>(acc)});
      }
    }
  }
  for (const auto &r : results)
    cu.write_vgpr(dst + r.reg, r.lane, r.val);
}

inline void exec_f64(amdgpu::ComputeUnitCore &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t B,
                     uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                     uint32_t const_acc = ACC_FROM_VGPR) {
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t lo;
    uint32_t hi;
  };
  std::vector<Result> results;
  results.reserve(M * N * B);
  for (uint32_t b = 0; b < B; ++b) {
    for (uint32_t row = 0; row < M; ++row) {
      for (uint32_t col = 0; col < N; ++col) {
        auto out = output_loc_64(M, N, col, row, b);
        double acc;
        if (const_acc != ACC_FROM_VGPR) {
          uint64_t bits64 =
              static_cast<uint64_t>(const_acc) | (static_cast<uint64_t>(const_acc) << 32);
          acc = std::bit_cast<double>(bits64);
        } else {
          uint32_t lo = cu.read_vgpr(s2 + out.reg, out.lane);
          uint32_t hi = cu.read_vgpr(s2 + out.reg + 1, out.lane);
          acc = std::bit_cast<double>(static_cast<uint64_t>(hi) << 32 | lo);
        }
        for (uint32_t k = 0; k < K; ++k) {
          auto al = input_loc(M, K, B, row, k, b, 64);
          auto bl = input_loc(N, K, B, col, k, b, 64);
          acc += extract_f64(cu, s0, al) * extract_f64(cu, s1, bl);
        }
        uint64_t bits = std::bit_cast<uint64_t>(acc);
        results.push_back(
            {out.reg, out.lane, static_cast<uint32_t>(bits), static_cast<uint32_t>(bits >> 32)});
      }
    }
  }
  for (const auto &r : results) {
    cu.write_vgpr(dst + r.reg, r.lane, r.lo);
    cu.write_vgpr(dst + r.reg + 1, r.lane, r.hi);
  }
}

} // namespace mfma
} // namespace cdna4
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_CDNA4_MFMA_EXEC_H_
