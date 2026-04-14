// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MFMA_EXEC_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MFMA_EXEC_H_

/// @file Shared Matrix Fused Multiply-Add (MFMA) register mapping and execution.
///
/// Implements the GFX9 MFMA register layout formulas from the AMD Matrix
/// Instruction Calculator (InstCalcGfx9). Shared across CDNA1-4 (all use the
/// same GFX9 encoding family for MFMA).
///
/// AccMode governs how the accumulator source (src2) is resolved:
///   - Unified:  CDNA3/4 — VGPR and AccVGPR share a single file; encoding
///               range 256-511 = VGPR, 768-1023 = AccVGPR (alias).
///   - Separate: CDNA2 — dedicated AccVGPR file, src2 ranges differ.
///   - VgprOnly: CDNA1 — no AccVGPR; src2 is always a VGPR or constant.
///
/// Key conventions:
///   - Output D[i][j]: i is the register dimension (matrix column),
///     j is the lane dimension (matrix row). Call output_loc with
///     i=column, j=row to get the physical (vgpr_offset, lane).
///   - Input A[row][k]: use input_loc(dim=M, K, B, i=row, k, b, bits).
///     Input B[col][k]: use input_loc(dim=N, K, B, i=col, k, b, bits).

#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "util/data_types.h"
#include "util/meta_programming.h"

#include <bit>
#include <cstdint>
#include <vector>

namespace rocjitsu {
namespace amdgpu {
namespace mfma {

/// Accumulator register mode, determined by CDNA generation.
enum class AccMode {
  Unified,  ///< CDNA3/4: VGPR and AccVGPR are in a unified file.
  Separate, ///< CDNA2: dedicated AccVGPR file (encoding base 512 for dst).
  VgprOnly, ///< CDNA1: no AccVGPR; src2 is always a VGPR or constant.
};

struct InputLoc {
  uint32_t vgpr_offset;
  uint32_t lane;
  uint32_t sub_element;
};

struct OutputLoc {
  uint32_t reg;
  uint32_t lane;
};

/// AccVGPR offset within a unified VGPR block. On CDNA3/4, AccVGPRs occupy
/// the second half of the 512-register block: acc0 = vgpr_base + 256.
constexpr uint32_t ACC_VGPR_OFFSET = 256;

/// Resolve VGPR base for an MFMA destination operand (OPR_VGPR_OR_ACCVGPR).
/// Encoding: 0-255 = ArchVGPR, 512-767 = AccVGPR (acc0-acc255).
inline uint32_t dst_base(uint32_t vb, int ev) {
  if (ev >= 512)
    return vb + ACC_VGPR_OFFSET + static_cast<uint32_t>(ev - 512);
  return vb + static_cast<uint32_t>(ev);
}

/// Resolve VGPR base for an MFMA source operand (OPR_SRC_VGPR_OR_ACCVGPR).
/// Encoding: 256-511 = ArchVGPR (v0-v255), 768-1023 = AccVGPR (acc0-acc255).
inline uint32_t src_base(uint32_t vb, int ev) {
  if (ev >= 768)
    return vb + ACC_VGPR_OFFSET + static_cast<uint32_t>(ev - 768);
  return (ev >= 256) ? vb + static_cast<uint32_t>(ev - 256) : vb + static_cast<uint32_t>(ev);
}

/// Sentinel value indicating the accumulator comes from a register, not a constant.
constexpr uint32_t ACC_FROM_VGPR = UINT32_MAX;

/// Resolve the accumulator source (src2) for MFMA instructions.
///
/// src2 uses OPR_SRC_VGPR_OR_ACCVGPR_OR_CONST which can be a VGPR (ev 256-511),
/// an ACCVGPR (ev 768-1023), or an inline constant (ev 0-255, e.g. literal 0
/// for zero-initializing the accumulator).
///
/// For VGPR/ACCVGPR operands, sets const_acc to ACC_FROM_VGPR and returns the
/// physical VGPR base. For constants, sets const_acc to the constant value and
/// returns dst (unused by exec functions in the constant case).
///
/// @tparam Mode AccMode for the current ISA generation.
/// @param const_acc Output: the constant value, or ACC_FROM_VGPR if src2 is a register.
/// @param get_const Lazy callback returning the 32-bit constant value; only
///        called when src2 is not a VGPR. Typically: [&]{ return src2.read_scalar(wf); }
template <AccMode Mode = AccMode::Unified, typename F>
inline uint32_t resolve_acc(uint32_t vb, uint32_t dst, int src2_ev, uint32_t &const_acc,
                            F &&get_const) {
  if constexpr (Mode == AccMode::Unified || Mode == AccMode::Separate) {
    if (src2_ev >= 768 && src2_ev <= 1023) {
      const_acc = ACC_FROM_VGPR;
      return vb + ACC_VGPR_OFFSET + static_cast<uint32_t>(src2_ev - 768);
    }
    if (src2_ev >= 256 && src2_ev <= 511) {
      const_acc = ACC_FROM_VGPR;
      return vb + static_cast<uint32_t>(src2_ev - 256);
    }
    const_acc = get_const();
    return dst;
  } else if constexpr (Mode == AccMode::VgprOnly) {
    if (src2_ev >= 256 && src2_ev <= 511) {
      const_acc = ACC_FROM_VGPR;
      return vb + static_cast<uint32_t>(src2_ev - 256);
    }
    const_acc = get_const();
    return dst;
  } else {
    static_assert(util::always_false_v<F>, "unhandled AccMode");
  }
}

/// Compute input element location for the GFX9 MFMA register layout.
///
/// @param dim Outer dimension (M for A matrix, N for B matrix)
/// @param K   Reduction dimension
/// @param B   Number of blocks
/// @param i   Outer index (row for A, column for B)
/// @param k   Reduction index
/// @param b   Block index
/// @param data_bits Element size in bits (8, 16, 32, or 64)
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

/// Compute 32-bit output element location for the GFX9 MFMA register layout.
inline OutputLoc output_loc_32(uint32_t M, uint32_t N, uint32_t i, uint32_t j, uint32_t b) {
  uint32_t multirows = 64 / N;
  uint32_t mn_div_4 = (M * N) / 4;
  uint32_t blocks_per_reg = (64 + mn_div_4 - 1) / mn_div_4;

  uint32_t reg = b * ((M * N) / 64) + (i / (4 * multirows)) * 4 + (i % 4);
  uint32_t lane = (b % blocks_per_reg) * N + ((i / 4) % multirows) * blocks_per_reg * N + j;
  return {reg, lane};
}

/// Compute 64-bit output element location for the GFX9 MFMA register layout.
/// Returns reg as a VGPR offset (each f64 element occupies 2 consecutive VGPRs).
inline OutputLoc output_loc_64(uint32_t M, uint32_t N, uint32_t i, uint32_t j, uint32_t b) {
  uint32_t multirows = 64 / N;
  uint32_t mn = M * N;
  uint32_t blocks_per_reg = (mn > 0) ? (64 + mn - 1) / mn : 1;

  uint32_t local = b * (mn / 64) + (i / multirows);
  uint32_t lane = (b % blocks_per_reg) * N + (i % multirows) * blocks_per_reg * N + j;
  return {local * 2, lane};
}

// ---------------------------------------------------------------------------
// Element extraction functions
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Execution kernels
// ---------------------------------------------------------------------------

/// Generic MFMA execute for f32 output: D = C + A x B.
///
/// All inputs are read before any outputs are written to avoid WAR hazards
/// when destination registers overlap source registers.
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

/// MFMA execute for i32 output with i8 input: D = C + A x B.
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

/// MFMA execute for f64 output with f64 input: D = C + A x B.
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
} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MFMA_EXEC_H_
