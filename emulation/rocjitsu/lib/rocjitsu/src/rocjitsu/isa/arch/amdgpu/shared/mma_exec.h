// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_EXEC_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_EXEC_H_

/// @file Shared Matrix Multiply-Accumulate (MMA) register mapping and execution.
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

#include "rocjitsu/isa/arch/amdgpu/shared/accvgpr_layout.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "util/data_types.h"
#include "util/except.h"
#include "util/meta_programming.h"
#include "util/simd.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace rocjitsu {
namespace amdgpu {
// MFMA register mapping, element extraction, and execution functions.

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
  uint32_t bit_offset = 0;
  uint32_t data_bits = 32;
};

struct OutputLoc {
  uint32_t reg;
  uint32_t lane;
};

struct PackedOutputLoc {
  uint32_t reg;
  uint32_t lane;
  uint32_t sub_element;
};

/// Resolve VGPR base for an MFMA destination operand.
/// The acc_cd bit in the MFMA encoding determines whether the destination
/// is in the arch VGPR bank (acc_cd=0, gfx950 unified model) or the
/// AccVGPR bank (acc_cd=1, gfx942 separate bank model).
/// Encoding 0-255 = v[0-255] or acc[0-255] depending on acc_cd.
/// Encoding 512-767 = acc[0-255] via OpSel (always AccVGPR bank).
inline uint32_t dst_base(uint32_t vb, int ev, uint32_t acc_cd = 1) {
  if (ev >= 512)
    return vb + ACC_VGPR_OFFSET + static_cast<uint32_t>(ev - 512);
  if (acc_cd)
    return vb + ACC_VGPR_OFFSET + static_cast<uint32_t>(ev);
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

constexpr uint32_t WMMA_WAVE32 = 32;
constexpr uint32_t WMMA_WAVE64 = 64;

inline uint32_t wmma_c_modifier(uint32_t neg, uint32_t neg_hi) {
  return ((neg >> 2) & 0x1u) | (((neg_hi >> 2) & 0x1u) << 1);
}

inline float apply_wmma_c_modifier(float acc, uint32_t c_modifier) {
  if (c_modifier & 0x2u)
    acc = std::fabs(acc);
  if (c_modifier & 0x1u)
    acc = -acc;
  return acc;
}

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
uint32_t resolve_acc(uint32_t vb, uint32_t dst, int src2_ev, uint32_t &const_acc, F &&get_const) {
  if constexpr (Mode == AccMode::Unified || Mode == AccMode::Separate) {
    if (src2_ev >= 768 && src2_ev <= 1023) {
      const_acc = ACC_FROM_VGPR;
      return vb + ACC_VGPR_OFFSET + static_cast<uint32_t>(src2_ev - 768);
    }
    if (src2_ev >= 256 && src2_ev <= 511) {
      const_acc = ACC_FROM_VGPR;
      // When MFMA writes to AccVGPR bank (acc_cd=1, dst >= vb+256),
      // the accumulator source at encoding 256-511 also refers to
      // AccVGPRs to maintain consistency. This matches gfx942 behavior
      // where v_accvgpr_write initializes the AccVGPR bank.
      if (dst >= vb + ACC_VGPR_OFFSET) {
        util::Logger::vm([&](auto &os) {
          os << std::format("MFMA resolve_acc: acc_cd path, dst={} vb={} src2_ev={} → acc_base={}",
                            dst, vb, src2_ev, vb + ACC_VGPR_OFFSET + (src2_ev - 256));
        });
        return vb + ACC_VGPR_OFFSET + static_cast<uint32_t>(src2_ev - 256);
      }
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

  uint32_t local, lane;
  // Sub-byte formats (fp4, fp6) pack contiguously even at high K; only
  // byte-or-wider elements (fp8+) use the chunked 16-element lane layout.
  if (elems_per_group > 16 && data_bits >= 8) {
    uint32_t chunk = k / 16;
    uint32_t g = chunk % lanes_per_block;
    local = (chunk / lanes_per_block) * 16 + (k % 16);
    lane = b * dim + g * dim * B + i;
  } else {
    local = k % elems_per_group;
    lane = b * dim + (k / elems_per_group) * dim * B + i;
  }

  if (data_bits == 64)
    return {local * 2, lane, 0, 0, data_bits};
  if (data_bits == 32)
    return {local, lane, 0, 0, data_bits};
  uint32_t bit = local * data_bits;
  uint32_t bit_in_word = bit % 32;
  uint32_t sub_element = (32 % data_bits == 0) ? (bit_in_word / data_bits) : 0;
  return {bit / 32, lane, sub_element, bit_in_word, data_bits};
}

inline uint32_t input_local_element(const InputLoc &loc) {
  return (loc.vgpr_offset * 32u + loc.bit_offset) / loc.data_bits;
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

inline void require_wmma_wave32(const amdgpu::ComputeUnitCore &cu) {
  if (cu.wf_size() != WMMA_WAVE32)
    throw util::ConfigError("gfx1250 WMMA requires wave32");
}

inline void require_gfx11_wmma_wave_size(uint32_t wave_size) {
  if (wave_size != WMMA_WAVE32 && wave_size != WMMA_WAVE64)
    throw util::ConfigError("gfx11 WMMA requires wave32 or wave64");
}

inline void require_gfx12_wmma_wave_size(uint32_t wave_size) {
  if (wave_size != WMMA_WAVE32 && wave_size != WMMA_WAVE64)
    throw util::ConfigError("gfx12 WMMA requires wave32 or wave64");
}

// gfx1250 WMMA is wave32-only. 16x16 WMMA v3 operands interleave the K
// dimension across the two lane groups in two register blocks.
inline InputLoc wmma_input_loc(uint32_t dim, uint32_t K, uint32_t i, uint32_t k,
                               uint32_t data_bits) {
  uint32_t lanes_per_group = WMMA_WAVE32 / dim;
  uint32_t elems_per_group = K / lanes_per_group;

  uint32_t local = k % elems_per_group;
  uint32_t lane = (k / elems_per_group) * dim + i;

  if (dim == 16 && K >= 32) {
    uint32_t block_elems = elems_per_group / 2;
    if (data_bits == 4 && K == 128)
      block_elems = 16;
    if (block_elems != 0) {
      const uint32_t lane_group = (k / block_elems) % lanes_per_group;
      const uint32_t reg_group = k / (block_elems * lanes_per_group);
      local = reg_group * block_elems + (k % block_elems);
      lane = lane_group * dim + i;
    }
  }

  if (data_bits == 64)
    return {local * 2, lane, 0, 0, data_bits};
  if (data_bits == 32)
    return {local, lane, 0, 0, data_bits};
  uint32_t bit = local * data_bits;
  uint32_t bit_in_word = bit % 32;
  uint32_t sub_element = (32 % data_bits == 0) ? (bit_in_word / data_bits) : 0;
  return {bit / 32, lane, sub_element, bit_in_word, data_bits};
}

inline InputLoc gfx11_wmma_input_loc(uint32_t dim, uint32_t K, uint32_t i, uint32_t k,
                                     uint32_t data_bits, uint32_t lane_group) {
  (void)K;
  // GFX11 WMMA R3 source operands replicate the full 16x16 tile into both
  // wave32 halfwaves. This matches LLVM's GFX11 builtin docs and was checked
  // against gfx1100 hardware: K is packed contiguously inside each halfwave
  // rather than split between lanes 0..15 and 16..31 as on gfx12/gfx1250.
  uint32_t lane = lane_group * dim + i;
  uint32_t bit = k * data_bits;
  uint32_t bit_in_word = bit % 32;
  uint32_t sub_element = (32 % data_bits == 0) ? (bit_in_word / data_bits) : 0;
  return {bit / 32, lane, sub_element, bit_in_word, data_bits};
}

inline InputLoc wmma_packed_input_loc(uint32_t lane, uint32_t slot, uint32_t data_bits) {
  const uint32_t bit = slot * data_bits;
  const uint32_t bit_in_word = bit % 32;
  const uint32_t sub_element = (32 % data_bits == 0) ? (bit_in_word / data_bits) : 0;
  return {bit / 32, lane, sub_element, bit_in_word, data_bits};
}

inline InputLoc gfx12_wmma_input_loc(uint32_t wave_size, uint32_t dim, uint32_t K, uint32_t i,
                                     uint32_t k, uint32_t data_bits) {
  require_gfx12_wmma_wave_size(wave_size);
  if (wave_size == WMMA_WAVE32)
    return wmma_input_loc(dim, K, i, k, data_bits);
  if (dim == 16 && K == 16) {
    const uint32_t lane = i + 16u * ((k >> 2) & 1u) + 32u * ((k >> 3) & 1u);
    const uint32_t slot = 2u * ((k >> 1) & 1u) + (k & 1u);
    return wmma_packed_input_loc(lane, slot, data_bits);
  }
  throw util::UnimplementedInst("unsupported gfx12 wave64 WMMA input layout");
}

inline InputLoc wmma_f8f6f4_ab_input_loc(uint32_t dim, uint32_t K, uint32_t i, uint32_t k,
                                         uint32_t data_bits) {
  if (dim == 16 && K == 128 && data_bits <= 8) {
    const uint32_t lane = i + 16u * ((k >> 2) & 1u);
    const uint32_t reg = ((k >> 1) & 1u) + 2u * ((k >> 3) & 1u) + 4u * ((k >> 4) & 1u) +
                         8u * ((k >> 5) & 1u) + 16u * ((k >> 6) & 1u);
    return wmma_packed_input_loc(lane, 2u * reg + (k & 1u), data_bits);
  }
  return wmma_input_loc(dim, K, i, k, data_bits);
}

inline InputLoc wmma_f8f6f4_mixed_subbyte_input_loc(uint32_t dim, uint32_t K, uint32_t i,
                                                    uint32_t k, uint32_t data_bits) {
  if (dim == 16 && K == 128 && data_bits < 8) {
    const uint32_t lane = i + 16u * ((k >> 5) & 1u);
    const uint32_t slot = 32u * ((k >> 6) & 1u) + 16u * ((k >> 2) & 1u) + 8u * ((k >> 4) & 1u) +
                          4u * ((k >> 3) & 1u) + 2u * ((k >> 1) & 1u) + (k & 1u);
    return wmma_packed_input_loc(lane, slot, data_bits);
  }
  return wmma_f8f6f4_ab_input_loc(dim, K, i, k, data_bits);
}

inline InputLoc wmma_f8f6f4_input_loc(uint32_t dim, uint32_t K, uint32_t i, uint32_t k,
                                      uint32_t data_bits, bool mixed_subbyte) {
  if (mixed_subbyte)
    return wmma_f8f6f4_mixed_subbyte_input_loc(dim, K, i, k, data_bits);
  return wmma_f8f6f4_ab_input_loc(dim, K, i, k, data_bits);
}

inline InputLoc wmma_f4_32x16x128_a_input_loc(uint32_t row, uint32_t k) {
  const uint32_t row_group = row >> 3;
  const uint32_t lane_row = row - 8u * ((row_group + 1u) >> 1u);
  const uint32_t lane = lane_row + 16u * ((k >> 2) & 1u);
  const uint32_t slot = 64u * (row_group & 1u) + (k & 3u) + 4u * (k >> 3);
  return wmma_packed_input_loc(lane, slot, 4);
}

inline InputLoc wmma_a_input_loc(uint32_t M, uint32_t K, uint32_t row, uint32_t k, uint32_t a_bits,
                                 uint32_t b_bits) {
  if (M == 32 && K == 128 && a_bits == 4 && b_bits == 4)
    return wmma_f4_32x16x128_a_input_loc(row, k);
  return wmma_f8f6f4_input_loc(M, K, row, k, a_bits, a_bits < 8 && b_bits == 8);
}

inline InputLoc wmma_b_input_loc(uint32_t N, uint32_t K, uint32_t col, uint32_t k, uint32_t a_bits,
                                 uint32_t b_bits) {
  return wmma_f8f6f4_input_loc(N, K, col, k, b_bits, b_bits < 8 && a_bits == 8);
}

inline InputLoc gfx12_wmma_a_input_loc(uint32_t wave_size, uint32_t M, uint32_t K, uint32_t row,
                                       uint32_t k, uint32_t a_bits, uint32_t b_bits) {
  if (wave_size == WMMA_WAVE32)
    return wmma_a_input_loc(M, K, row, k, a_bits, b_bits);
  return gfx12_wmma_input_loc(wave_size, M, K, row, k, a_bits);
}

inline InputLoc gfx12_wmma_b_input_loc(uint32_t wave_size, uint32_t N, uint32_t K, uint32_t col,
                                       uint32_t k, uint32_t a_bits, uint32_t b_bits) {
  if (wave_size == WMMA_WAVE32)
    return wmma_b_input_loc(N, K, col, k, a_bits, b_bits);
  return gfx12_wmma_input_loc(wave_size, N, K, col, k, b_bits);
}

inline OutputLoc wmma_output_loc_32(uint32_t M, uint32_t N, uint32_t row, uint32_t col) {
  uint32_t elems_per_lane = (M * N) / WMMA_WAVE32;
  uint32_t lane = (row / elems_per_lane) * N + col;
  uint32_t reg = row % elems_per_lane;
  return {reg, lane};
}

inline OutputLoc gfx12_wmma_output_loc_32(uint32_t wave_size, uint32_t M, uint32_t N, uint32_t row,
                                          uint32_t col) {
  require_gfx12_wmma_wave_size(wave_size);
  if (wave_size == WMMA_WAVE32)
    return wmma_output_loc_32(M, N, row, col);
  if (M == 16 && N == 16) {
    const uint32_t lane = col + 16u * ((row >> 3) & 1u) + 32u * ((row >> 2) & 1u);
    return {row & 3u, lane};
  }
  throw util::UnimplementedInst("unsupported gfx12 wave64 WMMA output layout");
}

inline OutputLoc gfx11_wmma_output_loc_32(uint32_t wave_size, uint32_t M, uint32_t N, uint32_t row,
                                          uint32_t col) {
  (void)M;
  uint32_t rows_per_reg = wave_size / N;
  uint32_t lane = (row % rows_per_reg) * N + col;
  uint32_t reg = row / rows_per_reg;
  return {reg, lane};
}

inline PackedOutputLoc wmma_output_loc_16(uint32_t M, uint32_t N, uint32_t row, uint32_t col) {
  uint32_t elems_per_lane = (M * N) / WMMA_WAVE32;
  uint32_t lane = (row / elems_per_lane) * N + col;
  uint32_t elem = row % elems_per_lane;
  return {elem / 2, lane, elem % 2};
}

inline PackedOutputLoc gfx12_wmma_output_loc_16(uint32_t wave_size, uint32_t M, uint32_t N,
                                                uint32_t row, uint32_t col) {
  require_gfx12_wmma_wave_size(wave_size);
  if (wave_size == WMMA_WAVE32)
    return wmma_output_loc_16(M, N, row, col);
  if (M == 16 && N == 16) {
    const uint32_t lane = col + 16u * ((row >> 3) & 1u) + 32u * ((row >> 2) & 1u);
    const uint32_t elem = row & 3u;
    return {elem / 2u, lane, elem & 1u};
  }
  throw util::UnimplementedInst("unsupported gfx12 wave64 WMMA output layout");
}

constexpr PackedOutputLoc gfx11_wmma_output_loc_16(uint32_t wave_size, uint32_t M, uint32_t N,
                                                   uint32_t row, uint32_t col, uint32_t opsel) {
  (void)M;
  uint32_t rows_per_reg = wave_size / N;
  uint32_t lane = (row % rows_per_reg) * N + col;
  uint32_t reg = row / rows_per_reg;
  return {reg, lane, opsel & 0x1u};
}

inline uint64_t read_swmmac_index_set(amdgpu::ComputeUnitCore &cu, uint32_t index_base,
                                      uint32_t lane, uint32_t index_entries, uint32_t index_key) {
  // gfx1250 FMT_WMMA_INDEX_SET and FMT_WMMA_INDEX_SET2 contain 16 and 32
  // packed 2-bit entries respectively. LLVM names the matching SWMMAC index
  // selector by entry count, not by total bit count.
  switch (index_entries) {
  case 8:
    return (cu.read_vgpr(index_base, lane) >> (8 * (index_key & 0x3u))) & 0xFFu;
  case 16:
    if (index_key & 0x1u)
      return (cu.read_vgpr(index_base, lane) >> 16) & 0xFFFFu;
    return cu.read_vgpr(index_base, lane);
  case 32: {
    if (index_key & 0x1u)
      return cu.read_vgpr(index_base + 1, lane);
    uint64_t lo = cu.read_vgpr(index_base, lane);
    uint64_t hi = cu.read_vgpr(index_base + 1, lane);
    return lo | (hi << 32);
  }
  default:
    throw util::UnimplementedInst("unsupported SWMMAC sparse index width");
  }
}

inline uint32_t swmmac_dense_k(uint64_t index_set, uint32_t compressed_k,
                               uint32_t local_compressed_k) {
  // The gfx1250 XML FMT_WMMA_INDEX_SET* formats describe packed 2-bit indices.
  // Adjacent index entries describe the two active positions in each dense 2:4
  // K group and are constrained by the ISA spec as index0 < index1.
  const uint32_t sparse_index = (index_set >> (2 * local_compressed_k)) & 0x3u;
  return (compressed_k / 2) * 4 + sparse_index;
}

struct SwmmacIndexLoc {
  uint32_t lane;
  uint32_t local_compressed_k;
};

inline SwmmacIndexLoc swmmac_index_loc(uint32_t M, uint32_t K, uint32_t elem_bits, uint32_t row,
                                       uint32_t compressed_k, uint32_t index_entries) {
  // RDNA4 K=32 SWMMAC uses the gfx12 builtin layout: each row's 16 sparse
  // 2-bit entries are split across two lanes. f16/bf16 split by pairs of
  // K-groups; fp8/bf8/iu8 split linearly by the low/high K=16 block.
  if (M == 16 && K == 32 && index_entries == 16) {
    const uint32_t group = compressed_k / 2;
    const uint32_t slot = compressed_k & 1u;
    if (elem_bits <= 8) {
      return {row + 16u * (compressed_k / 8u), compressed_k % 8u};
    }
    return {row + 16u * ((group / 2u) & 1u), 2u * (group & 1u) + 4u * (group / 4u) + slot};
  }
  return {row + (compressed_k / index_entries) * M, compressed_k % index_entries};
}

inline SwmmacIndexLoc swmmac_index_loc(uint32_t wave_size, uint32_t M, uint32_t K,
                                       uint32_t elem_bits, uint32_t row, uint32_t compressed_k,
                                       uint32_t index_entries) {
  require_gfx12_wmma_wave_size(wave_size);
  if (wave_size == WMMA_WAVE32)
    return swmmac_index_loc(M, K, elem_bits, row, compressed_k, index_entries);
  if (M == 16 && K == 32 && index_entries == 16) {
    const uint32_t group = compressed_k / 2u;
    const uint32_t slot = compressed_k & 1u;
    const uint32_t block =
        (elem_bits <= 8) ? (2u * ((group >> 1) & 1u) + (group >> 2)) : (group >> 1);
    return {row + 16u * block, 2u * (group & 1u) + slot};
  }
  throw util::UnimplementedInst("unsupported gfx12 wave64 SWMMAC index layout");
}

inline InputLoc swmmac_a_input_loc(uint32_t M, uint32_t K, uint32_t row, uint32_t compressed_k,
                                   uint32_t elem_bits) {
  if (M == 16 && K == 32) {
    const uint32_t group = compressed_k / 2u;
    const uint32_t slot = compressed_k & 1u;
    if (elem_bits == 8) {
      return wmma_packed_input_loc(row + 16u * (group / 4u), 2u * (group & 3u) + slot, elem_bits);
    }
    if (elem_bits >= 16) {
      const uint32_t side = (group / 2u) & 1u;
      const uint32_t a_gpr = 2u * (group / 4u) + (group & 1u);
      return wmma_packed_input_loc(row + 16u * side, 2u * a_gpr + slot, elem_bits);
    }
  }
  return wmma_input_loc(M, K / 2, row, compressed_k, elem_bits);
}

inline InputLoc swmmac_a_input_loc(uint32_t wave_size, uint32_t M, uint32_t K, uint32_t row,
                                   uint32_t compressed_k, uint32_t elem_bits) {
  require_gfx12_wmma_wave_size(wave_size);
  if (wave_size == WMMA_WAVE32)
    return swmmac_a_input_loc(M, K, row, compressed_k, elem_bits);
  if (M == 16 && K == 32) {
    const uint32_t group = compressed_k / 2u;
    const uint32_t slot = compressed_k & 1u;
    if (elem_bits == 8) {
      const uint32_t block = 2u * ((group >> 1) & 1u) + (group >> 2);
      return wmma_packed_input_loc(row + 16u * block, 2u * (group & 1u) + slot, elem_bits);
    }
    if (elem_bits >= 16)
      return wmma_packed_input_loc(row + 16u * (group >> 1), 2u * (group & 1u) + slot, elem_bits);
  }
  throw util::UnimplementedInst("unsupported gfx12 wave64 SWMMAC A layout");
}

inline InputLoc swmmac_b_input_loc(uint32_t N, uint32_t K, uint32_t col, uint32_t dense_k,
                                   uint32_t elem_bits) {
  if (N == 16 && K == 32) {
    if (elem_bits == 8)
      return wmma_packed_input_loc(col + 16u * (dense_k / 16u), dense_k % 16u, elem_bits);
    if (elem_bits >= 16) {
      const uint32_t lane = col + 16u * ((dense_k / 8u) & 1u);
      const uint32_t slot = 8u * (dense_k / 16u) + 2u * ((dense_k / 2u) & 3u) + (dense_k & 1u);
      return wmma_packed_input_loc(lane, slot, elem_bits);
    }
  }
  return wmma_input_loc(N, K, col, dense_k, elem_bits);
}

inline InputLoc swmmac_b_input_loc(uint32_t wave_size, uint32_t N, uint32_t K, uint32_t col,
                                   uint32_t dense_k, uint32_t elem_bits) {
  require_gfx12_wmma_wave_size(wave_size);
  if (wave_size == WMMA_WAVE32)
    return swmmac_b_input_loc(N, K, col, dense_k, elem_bits);
  if (N == 16 && K == 32) {
    if (elem_bits == 8) {
      const uint32_t lane = col + 32u * ((dense_k >> 3) & 1u) + 16u * (dense_k >> 4);
      const uint32_t slot = 4u * ((dense_k >> 2) & 1u) + (dense_k & 3u);
      return wmma_packed_input_loc(lane, slot, elem_bits);
    }
    if (elem_bits >= 16)
      return wmma_packed_input_loc(col + 16u * (dense_k / 8u), dense_k % 8u, elem_bits);
  }
  throw util::UnimplementedInst("unsupported gfx12 wave64 SWMMAC B layout");
}

// ---------------------------------------------------------------------------
// Lane permutation for cbsz/abid (A broadcast) and blgp (B permutation)
// ---------------------------------------------------------------------------

/// @brief Permute the A-matrix lane based on cbsz and abid fields.
///
/// @details When cbsz > 0, a block of S = 64/(1<<cbsz) lanes is broadcast to
/// all other blocks. abid selects which block is the broadcast source.
/// cbsz=0 means no broadcast (identity).
inline uint32_t permute_a_lane(uint32_t lane, uint32_t cbsz, uint32_t abid) {
  if (cbsz == 0)
    return lane;
  uint32_t S = 64 >> cbsz;
  return (lane % S) + S * abid;
}

/// @brief Permute the B-matrix lane based on the blgp field.
///
/// @details Per AMD ISA Table 29:
///   0: identity (l_b)
///   1: broadcast first 32 lanes  (l_b % 32)
///   2: broadcast second 32 lanes (l_b % 32 + 32)
///   3: rotate 16 lanes left      ((l_b + 16) % 64)
///   4: broadcast first 16 lanes  (l_b % 16)
///   5: broadcast second 16 lanes (l_b % 16 + 16)
///   6: broadcast third 16 lanes  (l_b % 16 + 32)
///   7: broadcast fourth 16 lanes (l_b % 16 + 48)
inline uint32_t permute_b_lane(uint32_t lane, uint32_t blgp) {
  switch (blgp) {
  case 0:
    return lane;
  case 1:
    return lane % 32;
  case 2:
    return lane % 32 + 32;
  case 3:
    return (lane + 16) % 64;
  case 4:
    return lane % 16;
  case 5:
    return lane % 16 + 16;
  case 6:
    return lane % 16 + 32;
  case 7:
    return lane % 16 + 48;
  default:
    return lane;
  }
}

// ---------------------------------------------------------------------------
// Element extraction functions
// ---------------------------------------------------------------------------

inline uint32_t packed_mask(uint32_t bits) { return bits >= 32 ? UINT32_MAX : ((1u << bits) - 1u); }

inline uint32_t read_packed(amdgpu::ComputeUnitCore &cu, uint32_t base, const InputLoc &loc) {
  uint32_t raw = cu.read_vgpr(base + loc.vgpr_offset, loc.lane) >> loc.bit_offset;
  if (loc.bit_offset + loc.data_bits > 32) {
    uint32_t next = cu.read_vgpr(base + loc.vgpr_offset + 1, loc.lane);
    raw |= next << (32 - loc.bit_offset);
  }
  return raw & packed_mask(loc.data_bits);
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

inline int32_t extract_u8(amdgpu::ComputeUnitCore &cu, uint32_t base, const InputLoc &loc) {
  uint32_t raw = cu.read_vgpr(base + loc.vgpr_offset, loc.lane);
  return static_cast<int32_t>((raw >> (loc.sub_element * 8)) & 0xFF);
}

inline int32_t sign_extend_packed(uint32_t value, uint32_t bits) {
  uint32_t sign = 1u << (bits - 1);
  return static_cast<int32_t>((value ^ sign) - sign);
}

inline int32_t extract_i4(amdgpu::ComputeUnitCore &cu, uint32_t base, const InputLoc &loc) {
  return sign_extend_packed(read_packed(cu, base, loc), 4);
}

inline int32_t extract_u4(amdgpu::ComputeUnitCore &cu, uint32_t base, const InputLoc &loc) {
  return static_cast<int32_t>(read_packed(cu, base, loc));
}

inline float extract_fp8_ocp(amdgpu::ComputeUnitCore &cu, uint32_t base, const InputLoc &loc) {
  uint32_t raw = cu.read_vgpr(base + loc.vgpr_offset, loc.lane);
  return util::fp8_e4m3_ocp_to_f32(static_cast<uint8_t>((raw >> (loc.sub_element * 8)) & 0xFF));
}

inline float extract_bf8_ocp(amdgpu::ComputeUnitCore &cu, uint32_t base, const InputLoc &loc) {
  uint32_t raw = cu.read_vgpr(base + loc.vgpr_offset, loc.lane);
  return util::bf8_e5m2_ocp_to_f32(static_cast<uint8_t>((raw >> (loc.sub_element * 8)) & 0xFF));
}

inline float extract_fp8_fnuz(amdgpu::ComputeUnitCore &cu, uint32_t base, const InputLoc &loc) {
  uint32_t raw = cu.read_vgpr(base + loc.vgpr_offset, loc.lane);
  return util::fp8_e4m3_fnuz_to_f32(static_cast<uint8_t>((raw >> (loc.sub_element * 8)) & 0xFF));
}

inline float extract_bf8_fnuz(amdgpu::ComputeUnitCore &cu, uint32_t base, const InputLoc &loc) {
  uint32_t raw = cu.read_vgpr(base + loc.vgpr_offset, loc.lane);
  return util::bf8_e5m2_fnuz_to_f32(static_cast<uint8_t>((raw >> (loc.sub_element * 8)) & 0xFF));
}

inline float extract_fp8(amdgpu::ComputeUnitCore &cu, uint32_t base, const InputLoc &loc) {
  return extract_fp8_ocp(cu, base, loc);
}

inline float extract_bf8(amdgpu::ComputeUnitCore &cu, uint32_t base, const InputLoc &loc) {
  return extract_bf8_ocp(cu, base, loc);
}

inline float extract_fp4(amdgpu::ComputeUnitCore &cu, uint32_t base, const InputLoc &loc) {
  return util::fp4_e2m1_to_f32(static_cast<uint8_t>(read_packed(cu, base, loc)));
}

inline float extract_fp6(amdgpu::ComputeUnitCore &cu, uint32_t base, const InputLoc &loc) {
  return util::fp6_e2m3_to_f32(static_cast<uint8_t>(read_packed(cu, base, loc)));
}

inline float extract_bf6(amdgpu::ComputeUnitCore &cu, uint32_t base, const InputLoc &loc) {
  return util::bf6_e3m2_to_f32(static_cast<uint8_t>(read_packed(cu, base, loc)));
}

inline double extract_f64(amdgpu::ComputeUnitCore &cu, uint32_t base, const InputLoc &loc) {
  uint32_t lo = cu.read_vgpr(base + loc.vgpr_offset, loc.lane);
  uint32_t hi = cu.read_vgpr(base + loc.vgpr_offset + 1, loc.lane);
  return std::bit_cast<double>(static_cast<uint64_t>(hi) << 32 | lo);
}

inline float decode_e8m0_scale(uint8_t raw) { return util::e8m0_to_f32(raw); }

inline float decode_wmma_scale_byte(uint8_t raw, uint32_t fmt) {
  switch (fmt) {
  case 0:
    return decode_e8m0_scale(raw);
  case 2:
    return util::fp8_e4m3_to_f32(raw);
  default:
    throw util::UnimplementedInst("unsupported WMMA scale format");
  }
}

inline uint32_t wmma_scale_lane(uint32_t index, uint32_t scale_select) {
  return index + ((scale_select & 0x1u) ? 16u : 0u);
}

inline uint32_t wmma_f4_32x16x128_a_scale_lane(uint32_t row) {
  const uint32_t row_group = row >> 3;
  if (row_group == 1)
    return row + 8u;
  if (row_group == 2)
    return row - 8u;
  return row;
}

inline uint32_t wmma_a_scale_lane(uint32_t M, uint32_t K, uint32_t row, uint32_t scale_select,
                                  uint32_t a_bits, uint32_t b_bits) {
  if (M == 32 && K == 128 && a_bits == 4 && b_bits == 4)
    return wmma_f4_32x16x128_a_scale_lane(row);
  return wmma_scale_lane(row, scale_select);
}

inline uint32_t wmma_scale_byte(const InputLoc &loc) {
  return (input_local_element(loc) / 16u) & 0x3u;
}

inline uint32_t wmma_b_fp4_scale_byte(uint32_t k) {
  return ((k / 16u) & 0x1u) | (((k / 64u) & 0x1u) << 1u);
}

inline uint32_t wmma_f8f6f4_scale_byte(uint32_t k, uint32_t data_bits, bool mixed_pair,
                                       bool scale16) {
  if (scale16) {
    if (mixed_pair || data_bits == 8)
      return 2u * (k >> 5) + ((k >> 2) & 1u);
    return 4u * (k >> 6) + 2u * ((k >> 2) & 1u) + ((k >> 5) & 1u);
  }
  if (mixed_pair || data_bits == 8)
    return k >> 5;
  return 2u * (k >> 6) + ((k >> 2) & 1u);
}

template <typename Run> bool dispatch_matrix_fmt_pair(uint32_t a_fmt, uint32_t b_fmt, Run run) {
  switch (a_fmt) {
  case 0:
    switch (b_fmt) {
    case 0:
      run(8, 8, extract_fp8, extract_fp8);
      return true;
    case 1:
      run(8, 8, extract_fp8, extract_bf8);
      return true;
    case 2:
      run(8, 6, extract_fp8, extract_fp6);
      return true;
    case 3:
      run(8, 6, extract_fp8, extract_bf6);
      return true;
    case 4:
      run(8, 4, extract_fp8, extract_fp4);
      return true;
    }
    return false;
  case 1:
    switch (b_fmt) {
    case 0:
      run(8, 8, extract_bf8, extract_fp8);
      return true;
    case 1:
      run(8, 8, extract_bf8, extract_bf8);
      return true;
    case 2:
      run(8, 6, extract_bf8, extract_fp6);
      return true;
    case 3:
      run(8, 6, extract_bf8, extract_bf6);
      return true;
    case 4:
      run(8, 4, extract_bf8, extract_fp4);
      return true;
    }
    return false;
  case 2:
    switch (b_fmt) {
    case 0:
      run(6, 8, extract_fp6, extract_fp8);
      return true;
    case 1:
      run(6, 8, extract_fp6, extract_bf8);
      return true;
    case 2:
      run(6, 6, extract_fp6, extract_fp6);
      return true;
    case 3:
      run(6, 6, extract_fp6, extract_bf6);
      return true;
    case 4:
      run(6, 4, extract_fp6, extract_fp4);
      return true;
    }
    return false;
  case 3:
    switch (b_fmt) {
    case 0:
      run(6, 8, extract_bf6, extract_fp8);
      return true;
    case 1:
      run(6, 8, extract_bf6, extract_bf8);
      return true;
    case 2:
      run(6, 6, extract_bf6, extract_fp6);
      return true;
    case 3:
      run(6, 6, extract_bf6, extract_bf6);
      return true;
    case 4:
      run(6, 4, extract_bf6, extract_fp4);
      return true;
    }
    return false;
  case 4:
    switch (b_fmt) {
    case 0:
      run(4, 8, extract_fp4, extract_fp8);
      return true;
    case 1:
      run(4, 8, extract_fp4, extract_bf8);
      return true;
    case 2:
      run(4, 6, extract_fp4, extract_fp6);
      return true;
    case 3:
      run(4, 6, extract_fp4, extract_bf6);
      return true;
    case 4:
      run(4, 4, extract_fp4, extract_fp4);
      return true;
    }
    return false;
  default:
    return false;
  }
}

// ---------------------------------------------------------------------------
// Execution kernels
// ---------------------------------------------------------------------------

/// Shared SIMD core for the MFMA/WMMA/SWMMAC executors. For every (row,col),
/// Cbuf[row*stride+col] += sum_k Abuf[row*K+k] * Bbuf[k*stride+col], run as
/// native-width rows over the col (N) dimension with a scalar tail. A/B/C must
/// already be hoisted into dense buffers (lane permutation, per-element scale,
/// and 2:4 sparsity gather folded in by the caller). Float uses fused FMA
/// (matching the hardware's single-rounding MACs; the scalar reference is
/// non-fused, so f32 agrees to a few ULP and packed f16/bf16 rounds identically);
/// integer MAC is exact, so the SIMD and scalar paths are bit-identical.
/// Templated so the `if constexpr (has_stdx_simd)` callers never instantiate it
/// on a platform without <experimental/simd>.
///
/// Cbuf is the in/out accumulator: each (row,col) is read as the starting value,
/// the K products are added, and the result written back. Callers come in two
/// flavors — most pre-seed Cbuf with the hardware C accumulator (D = C + A*B in
/// place), while the int32 WMMA paths instead leave Cbuf ZERO and add the
/// hardware accumulator afterward in wider precision (for saturation). That is
/// why the staging buffers are zero-initialized as a uniform convention: it is
/// load-bearing for the zero-start callers and harmless (redundant) for the
/// pre-seeded ones.
///
/// Read-bounds: only the used region is touched — Abuf[row*K+k] for row<M,k<K,
/// and Bbuf/Cbuf[..*stride+col] for col<N (the vectorized loop is bounded by
/// `col + W <= N`, the remainder is a scalar tail at col<N). The padding columns
/// [N, stride) — present only so each row starts W-aligned — are never read.
template <typename T>
void wmma_simd_matmul(uint32_t M, uint32_t N, uint32_t K, uint32_t W, uint32_t stride,
                      const T *Abuf, const T *Bbuf, T *Cbuf) {
  for (uint32_t row = 0; row < M; ++row) {
    uint32_t col = 0;
    for (; col + W <= N; col += W) {
      util::native<T> c;
      c.copy_from(&Cbuf[row * stride + col], util::stdx::vector_aligned);
      for (uint32_t k = 0; k < K; ++k) {
        util::native<T> a(Abuf[row * K + k]);
        util::native<T> bv;
        bv.copy_from(&Bbuf[k * stride + col], util::stdx::vector_aligned);
        if constexpr (std::is_floating_point_v<T>)
          c = util::stdx::fma(a, bv, c);
        else
          c += a * bv;
      }
      c.copy_to(&Cbuf[row * stride + col], util::stdx::vector_aligned);
    }
    for (; col < N; ++col) {
      T acc = Cbuf[row * stride + col];
      for (uint32_t k = 0; k < K; ++k) {
        if constexpr (std::is_floating_point_v<T>)
          acc = std::fma(Abuf[row * K + k], Bbuf[k * stride + col], acc);
        else
          acc += Abuf[row * K + k] * Bbuf[k * stride + col];
      }
      Cbuf[row * stride + col] = acc;
    }
  }
}

/// Adjust a GFX9 InputLoc for wave32 physical VGPR addressing.
///
/// The GFX9 MFMA layout uses 64 virtual lanes. On wave64 this maps directly
/// to physical registers. On wave32, each logical VGPR spans two physical
/// VGPRs. Without stride adjustment, read_vgpr(base+V, lane>=32) aliases
/// with read_vgpr(base+V+1, lane%32), corrupting data from the next logical
/// VGPR. The fix strides the logical VGPR offset by 64/wf_size on wave32.
///
/// Returns @p loc unchanged when wf_size >= 64.
inline InputLoc physicalize_loc(const InputLoc &loc, uint32_t wf_size) {
  if (wf_size >= 64)
    return loc;
  const uint32_t stride = 64 / wf_size;
  return {loc.vgpr_offset * stride + loc.lane / wf_size, loc.lane % wf_size, loc.sub_element,
          loc.bit_offset, loc.data_bits};
}

/// Adjust a GFX9 OutputLoc for wave32 physical VGPR addressing.
/// No-op when wf_size >= 64.
inline OutputLoc physicalize_out(const OutputLoc &loc, uint32_t wf_size) {
  if (wf_size >= 64)
    return loc;
  const uint32_t stride = 64 / wf_size;
  return {loc.reg * stride + loc.lane / wf_size, loc.lane % wf_size};
}

/// Adjust a GFX9 PackedOutputLoc for wave32 physical VGPR addressing.
/// No-op when wf_size >= 64.
inline PackedOutputLoc physicalize_packed_out(const PackedOutputLoc &loc, uint32_t wf_size) {
  if (wf_size >= 64)
    return loc;
  const uint32_t stride = 64 / wf_size;
  return {loc.reg * stride + loc.lane / wf_size, loc.lane % wf_size, loc.sub_element};
}

/// Map f32 output position to packed 16-bit output position using GFX9 layout.
/// Two consecutive f32 register positions pack into one 32-bit VGPR as two
/// 16-bit sub-elements: reg/2 holds the VGPR offset, reg%2 the sub-element.
inline PackedOutputLoc output_loc_16(uint32_t M, uint32_t N, uint32_t i, uint32_t j, uint32_t b) {
  auto f32 = output_loc_32(M, N, i, j, b);
  return {f32.reg / 2, f32.lane, f32.reg % 2};
}

/// Generic MFMA execute for f32 output: D = C + A x B.
///
/// All inputs are read before any outputs are written to avoid WAR hazards
/// when destination registers overlap source registers.
///
/// @param cbsz  A-matrix broadcast block size (0 = no broadcast).
/// @param abid  A-matrix broadcast source block ID.
/// @param blgp  B-matrix lane group permutation pattern.
template <typename ExtractA, typename ExtractB>
void exec_f32_mixed(amdgpu::ComputeUnitCore &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t B,
                    uint32_t a_bits, uint32_t b_bits, uint32_t dst, uint32_t s0, uint32_t s1,
                    uint32_t s2, ExtractA ea, ExtractB eb, uint32_t const_acc = ACC_FROM_VGPR,
                    uint32_t cbsz = 0, uint32_t abid = 0, uint32_t blgp = 0) {
  const uint32_t wf = cu.wf_size();
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t val;
  };
  std::vector<Result> results;
  results.reserve(M * N * B);

  // Scalar reference: D[i][j] = C[i][j] + sum_k A[i][k] * B[k][j], accumulated
  // per output in K order (non-fused multiply-add).
  auto run_scalar = [&]() {
    for (uint32_t b = 0; b < B; ++b) {
      for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t col = 0; col < N; ++col) {
          // AMD convention: i=row (register dimension), j=col (lane dimension).
          auto out = physicalize_out(output_loc_32(M, N, row, col, b), wf);
          float acc = (const_acc != ACC_FROM_VGPR)
                          ? std::bit_cast<float>(const_acc)
                          : std::bit_cast<float>(cu.read_vgpr(s2 + out.reg, out.lane));
          for (uint32_t k = 0; k < K; ++k) {
            auto al = input_loc(M, K, B, row, k, b, a_bits);
            auto bl = input_loc(N, K, B, col, k, b, b_bits);
            // Apply cbsz/abid lane permutation to A input.
            if (cbsz != 0)
              al.lane = permute_a_lane(al.lane, cbsz, abid);
            // Apply blgp lane permutation to B input.
            if (blgp != 0)
              bl.lane = permute_b_lane(bl.lane, blgp);
            float a_val = ea(cu, s0, physicalize_loc(al, wf));
            float b_val = eb(cu, s1, physicalize_loc(bl, wf));
            acc += a_val * b_val;
          }
          results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(acc)});
        }
      }
    }
  };

  // SIMD fast path. Works for any MFMA shape and any f32-producing extract
  // (f16/bf16/fp8/bf8/f32). Per block: hoist A into row-major (row,k), B into
  // row-major (k,col), and C into (row,col) dense f32 buffers (lane
  // permutation folded in during the hoist), then run the dense MxNxK matmul
  // as native-width FMA rows over the N (column) dimension. Uses fused FMA,
  // matching the GFX9 MFMA hardware's single-rounding MACs (the scalar path
  // above is non-fused; results agree to a few ULP). N columns that don't
  // fill a full SIMD lane group fall to a scalar (fused) tail.
  if constexpr (util::has_stdx_simd) {
    // Pad the column (N) leading dimension up to a SIMD-width multiple so
    // every matmul row starts W-aligned: the inner loop then uses aligned
    // loads/stores for any N, and the staging buffers live on the stack
    // (no per-call heap allocation). MAX_* bound every real MFMA shape;
    // anything larger (or a forced-scalar run) falls back to the scalar path.
    constexpr uint32_t W = static_cast<uint32_t>(util::native<float>::size());
    constexpr size_t MAX_AB = 2048;      // max M*K over all MFMA shapes
    constexpr size_t MAX_BSTRIDE = 4096; // max K*stride
    constexpr size_t MAX_C = 1024;       // max M*stride
    // Combined stack frame for the three staging buffers below (currently 28
    // KiB); tripwire so an added/larger shape can't silently blow the stack.
    static_assert((MAX_AB + MAX_BSTRIDE + MAX_C) * sizeof(float) <= 48 * 1024,
                  "MFMA SIMD staging buffers exceed the 48 KiB stack budget");
    const uint32_t stride = ((N + W - 1) / W) * W;
    if (util::force_scalar() || static_cast<size_t>(M) * K > MAX_AB ||
        static_cast<size_t>(K) * stride > MAX_BSTRIDE || static_cast<size_t>(M) * stride > MAX_C) {
      run_scalar();
    } else {
      // Zero-initialized as the uniform staging-buffer convention (see the
      // zero-init policy on wmma_simd_matmul). C is pre-seeded just below, so the
      // init is redundant here, but matching the WMMA paths keeps one rule.
      alignas(64) float Abuf[MAX_AB] = {};
      alignas(64) float Bbuf[MAX_BSTRIDE] = {};
      alignas(64) float Cbuf[MAX_C] = {};
      for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t row = 0; row < M; ++row)
          for (uint32_t col = 0; col < N; ++col) {
            auto out = physicalize_out(output_loc_32(M, N, row, col, b), wf);
            Cbuf[row * stride + col] =
                (const_acc != ACC_FROM_VGPR)
                    ? std::bit_cast<float>(const_acc)
                    : std::bit_cast<float>(cu.read_vgpr(s2 + out.reg, out.lane));
          }
        for (uint32_t row = 0; row < M; ++row)
          for (uint32_t k = 0; k < K; ++k) {
            auto al = input_loc(M, K, B, row, k, b, a_bits);
            if (cbsz != 0)
              al.lane = permute_a_lane(al.lane, cbsz, abid);
            Abuf[row * K + k] = ea(cu, s0, physicalize_loc(al, wf));
          }
        for (uint32_t k = 0; k < K; ++k)
          for (uint32_t col = 0; col < N; ++col) {
            auto bl = input_loc(N, K, B, col, k, b, b_bits);
            if (blgp != 0)
              bl.lane = permute_b_lane(bl.lane, blgp);
            Bbuf[k * stride + col] = eb(cu, s1, physicalize_loc(bl, wf));
          }
        wmma_simd_matmul<float>(M, N, K, W, stride, Abuf, Bbuf, Cbuf);
        for (uint32_t row = 0; row < M; ++row)
          for (uint32_t col = 0; col < N; ++col) {
            auto out = physicalize_out(output_loc_32(M, N, row, col, b), wf);
            results.push_back(
                {out.reg, out.lane, std::bit_cast<uint32_t>(Cbuf[row * stride + col])});
          }
      }
    }
  } else {
    run_scalar();
  }

  bool has_nan = false;
  for (const auto &r : results) {
    cu.write_vgpr(dst + r.reg, r.lane, r.val);
    float fval = std::bit_cast<float>(r.val);
    if (std::isnan(fval) || std::isinf(fval))
      has_nan = true;
  }
  if (has_nan) {
    util::Logger::vm([&](auto &os) {
      os << std::format("MFMA_NAN_DETECTED dst=v{} s0=v{} s1=v{} s2=v{} M={} N={} K={}", dst, s0,
                        s1, s2, M, N, K);
      for (const auto &r : results) {
        float fval = std::bit_cast<float>(r.val);
        if (std::isnan(fval) || std::isinf(fval))
          os << std::format("\n[rj log VM]   reg={} lane={} val={:#x}({}) "
                            "a=[{:#x},{:#x}] b=[{:#x},{:#x}]",
                            r.reg, r.lane, r.val, fval, cu.read_vgpr(s0, r.lane),
                            cu.read_vgpr(s0 + 1, r.lane), cu.read_vgpr(s1, r.lane),
                            cu.read_vgpr(s1 + 1, r.lane));
      }
    });
  }
  util::Logger::vm([&](auto &os) {
    static thread_local uint64_t mfma_count = 0;
    if (++mfma_count > 30)
      return;
    os << std::format("MFMA_F32 #{} M={} N={} K={} B={} dst=v{} s0=v{} s1=v{} s2=v{}", mfma_count,
                      M, N, K, B, dst, s0, s1, s2);
    for (uint32_t ln : {0u, 1u, 4u, 8u, 16u, 31u, 32u, 48u, 63u}) {
      os << std::format("\n[rj log VM]   L{}: s0=[{:#x},{:#x},{:#x},{:#x}]"
                        " s1=[{:#x},{:#x},{:#x},{:#x}]"
                        " out=[{:#x},{:#x},{:#x},{:#x}]",
                        ln, cu.read_vgpr(s0, ln), cu.read_vgpr(s0 + 1, ln),
                        cu.read_vgpr(s0 + 2, ln), cu.read_vgpr(s0 + 3, ln), cu.read_vgpr(s1, ln),
                        cu.read_vgpr(s1 + 1, ln), cu.read_vgpr(s1 + 2, ln),
                        cu.read_vgpr(s1 + 3, ln), cu.read_vgpr(dst, ln), cu.read_vgpr(dst + 1, ln),
                        cu.read_vgpr(dst + 2, ln), cu.read_vgpr(dst + 3, ln));
    }
  });
}

template <typename ExtractA, typename ExtractB>
void exec_f32(amdgpu::ComputeUnitCore &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t B,
              uint32_t in_bits, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2, ExtractA ea,
              ExtractB eb, uint32_t const_acc = ACC_FROM_VGPR, uint32_t cbsz = 0, uint32_t abid = 0,
              uint32_t blgp = 0) {
  exec_f32_mixed(cu, M, N, K, B, in_bits, in_bits, dst, s0, s1, s2, ea, eb, const_acc, cbsz, abid,
                 blgp);
}

/// Column (N) leading-dimension pitch rounding A/B/C buffers up to a SIMD-width
/// multiple, so every matmul row starts W-aligned. Bounds every real WMMA shape
/// (M,N <= 16, K <= 128); anything larger falls back to the scalar path.
constexpr size_t WMMA_SIMD_MAX_AB = 2048;      // max M*K
constexpr size_t WMMA_SIMD_MAX_BSTRIDE = 4096; // max K*stride
constexpr size_t WMMA_SIMD_MAX_C = 1024;       // max M*stride
// Combined stack frame for the WMMA/SWMMAC staging buffers (currently 28 KiB for
// the float and int32 paths); tripwire against silent stack-frame growth.
static_assert((WMMA_SIMD_MAX_AB + WMMA_SIMD_MAX_BSTRIDE + WMMA_SIMD_MAX_C) * sizeof(float) <=
                  48 * 1024,
              "WMMA SIMD staging buffers exceed the 48 KiB stack budget");

/// GFX9-layout packed 16-bit output execution: D = C + A x B, result packed f16/bf16.
/// Uses GFX9 input_loc for inputs and output_loc_16 (derived from output_loc_32)
/// for outputs. The accumulator is read/written in packed 16-bit format.
template <typename ExtractA, typename ExtractB, typename ReadAcc, typename PackResult>
void exec_packed16_gfx9(amdgpu::ComputeUnitCore &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t B,
                        uint32_t in_bits, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                        ExtractA ea, ExtractB eb, ReadAcc read_acc, PackResult pack_result,
                        uint32_t const_acc = ACC_FROM_VGPR) {
  const uint32_t wf = cu.wf_size();
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t sub_element;
    uint16_t val;
  };
  std::vector<Result> results;
  results.reserve(M * N * B);
  for (uint32_t b = 0; b < B; ++b) {
    for (uint32_t row = 0; row < M; ++row) {
      for (uint32_t col = 0; col < N; ++col) {
        auto out = physicalize_packed_out(output_loc_16(M, N, row, col, b), wf);
        float acc = (const_acc != ACC_FROM_VGPR)
                        ? std::bit_cast<float>(const_acc)
                        : read_acc(cu, s2 + out.reg, out.lane, out.sub_element);
        for (uint32_t k = 0; k < K; ++k) {
          auto al = physicalize_loc(input_loc(M, K, B, row, k, b, in_bits), wf);
          auto bl = physicalize_loc(input_loc(N, K, B, col, k, b, in_bits), wf);
          acc += ea(cu, s0, al) * eb(cu, s1, bl);
        }
        results.push_back({out.reg, out.lane, out.sub_element, pack_result(acc)});
      }
    }
  }

  // Determine physical VGPR count for the packed output.
  uint32_t max_reg = 0;
  for (const auto &r : results)
    if (r.reg > max_reg)
      max_reg = r.reg;
  uint32_t dst_regs = max_reg + 1;

  std::vector<uint32_t> words(dst_regs * wf, 0);
  std::vector<uint8_t> masks(dst_regs * wf, 0);
  for (const auto &r : results) {
    uint32_t idx = r.reg * wf + r.lane;
    uint32_t shift = r.sub_element * 16;
    words[idx] = (words[idx] & ~(0xFFFFu << shift)) | (static_cast<uint32_t>(r.val) << shift);
    masks[idx] |= 1u << r.sub_element;
  }
  for (uint32_t reg = 0; reg < dst_regs; ++reg) {
    for (uint32_t lane = 0; lane < wf; ++lane) {
      uint32_t idx = reg * wf + lane;
      uint32_t word = words[idx];
      if (masks[idx] != 0x3u) {
        uint32_t old = cu.read_vgpr(dst + reg, lane);
        if ((masks[idx] & 0x1u) == 0)
          word = (word & 0xFFFF0000u) | (old & 0x0000FFFFu);
        if ((masks[idx] & 0x2u) == 0)
          word = (word & 0x0000FFFFu) | (old & 0xFFFF0000u);
      }
      cu.write_vgpr(dst + reg, lane, word);
    }
  }
}

/// Convenience wrappers for GFX9 packed f16/bf16 output.
template <typename ExtractA, typename ExtractB>
void exec_f16_gfx9(amdgpu::ComputeUnitCore &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t B,
                   uint32_t in_bits, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                   ExtractA ea, ExtractB eb, uint32_t const_acc = ACC_FROM_VGPR) {
  exec_packed16_gfx9(
      cu, M, N, K, B, in_bits, dst, s0, s1, s2, ea, eb,
      [](amdgpu::ComputeUnitCore &c, uint32_t base, uint32_t lane, uint32_t sub) -> float {
        uint32_t raw = c.read_vgpr(base, lane);
        return util::f16_to_f32(static_cast<uint16_t>((raw >> (sub * 16)) & 0xFFFF));
      },
      [](float v) -> uint16_t { return util::f32_to_f16(v); }, const_acc);
}

template <typename ExtractA, typename ExtractB>
void exec_bf16_gfx9(amdgpu::ComputeUnitCore &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t B,
                    uint32_t in_bits, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                    ExtractA ea, ExtractB eb, uint32_t const_acc = ACC_FROM_VGPR) {
  exec_packed16_gfx9(
      cu, M, N, K, B, in_bits, dst, s0, s1, s2, ea, eb,
      [](amdgpu::ComputeUnitCore &c, uint32_t base, uint32_t lane, uint32_t sub) -> float {
        uint32_t raw = c.read_vgpr(base, lane);
        return util::bf16_to_f32(static_cast<uint16_t>((raw >> (sub * 16)) & 0xFFFF));
      },
      [](float v) -> uint16_t { return util::f32_to_bf16(v); }, const_acc);
}

inline uint32_t pack_i32_acc(int64_t acc, bool clamp) {
  if (clamp) {
    acc = std::clamp(acc, static_cast<int64_t>(std::numeric_limits<int32_t>::min()),
                     static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
  }
  return static_cast<uint32_t>(acc);
}

/// GFX9-layout i32 output with configurable input bit-width and extractors.
/// When clamp=false, accumulates in uint32_t (defined wrapping, matching
/// hardware). When clamp=true, accumulates in int64_t and saturates to
/// int32_t range via pack_i32_acc.
template <typename ExtractA, typename ExtractB>
void exec_i32_mixed(amdgpu::ComputeUnitCore &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t B,
                    uint32_t in_bits, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                    ExtractA ea, ExtractB eb, uint32_t const_acc = ACC_FROM_VGPR,
                    bool clamp = false, uint32_t cbsz = 0, uint32_t abid = 0, uint32_t blgp = 0) {
  const uint32_t wf = cu.wf_size();
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
        auto out = physicalize_out(output_loc_32(M, N, row, col, b), wf);
        auto products = [&](uint32_t k) {
          auto al = input_loc(M, K, B, row, k, b, in_bits);
          auto bl = input_loc(N, K, B, col, k, b, in_bits);
          if (cbsz != 0)
            al.lane = permute_a_lane(al.lane, cbsz, abid);
          if (blgp != 0)
            bl.lane = permute_b_lane(bl.lane, blgp);
          return std::pair{ea(cu, s0, physicalize_loc(al, wf)),
                           eb(cu, s1, physicalize_loc(bl, wf))};
        };
        uint32_t val;
        if (clamp) {
          int64_t acc = (const_acc != ACC_FROM_VGPR)
                            ? static_cast<int64_t>(static_cast<int32_t>(const_acc))
                            : static_cast<int64_t>(
                                  static_cast<int32_t>(cu.read_vgpr(s2 + out.reg, out.lane)));
          for (uint32_t k = 0; k < K; ++k) {
            auto [a, b_val] = products(k);
            acc += static_cast<int64_t>(a) * static_cast<int64_t>(b_val);
          }
          val = pack_i32_acc(acc, true);
        } else {
          uint32_t acc =
              (const_acc != ACC_FROM_VGPR) ? const_acc : cu.read_vgpr(s2 + out.reg, out.lane);
          for (uint32_t k = 0; k < K; ++k) {
            auto [a, b_val] = products(k);
            acc += static_cast<uint32_t>(a * b_val);
          }
          val = acc;
        }
        results.push_back({out.reg, out.lane, val});
      }
    }
  }
  for (const auto &r : results)
    cu.write_vgpr(dst + r.reg, r.lane, r.val);
}

template <typename ExtractA, typename ExtractB>
void exec_wmma_f32_mixed(amdgpu::ComputeUnitCore &cu, uint32_t M, uint32_t N, uint32_t K,
                         uint32_t a_bits, uint32_t b_bits, uint32_t dst, uint32_t s0, uint32_t s1,
                         uint32_t s2, ExtractA ea, ExtractB eb, uint32_t const_acc = ACC_FROM_VGPR,
                         uint32_t c_modifier = 0, uint32_t wave_size = WMMA_WAVE32) {
  require_gfx12_wmma_wave_size(wave_size);
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t val;
  };
  std::vector<Result> results;
  results.reserve(M * N);

  auto run_scalar = [&]() {
    for (uint32_t row = 0; row < M; ++row) {
      for (uint32_t col = 0; col < N; ++col) {
        auto out = gfx12_wmma_output_loc_32(wave_size, M, N, row, col);
        float acc = (const_acc != ACC_FROM_VGPR)
                        ? std::bit_cast<float>(const_acc)
                        : std::bit_cast<float>(cu.read_vgpr(s2 + out.reg, out.lane));
        acc = apply_wmma_c_modifier(acc, c_modifier);
        for (uint32_t k = 0; k < K; ++k) {
          auto al = gfx12_wmma_a_input_loc(wave_size, M, K, row, k, a_bits, b_bits);
          auto bl = gfx12_wmma_b_input_loc(wave_size, N, K, col, k, a_bits, b_bits);
          acc += ea(cu, s0, al) * eb(cu, s1, bl);
        }
        results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(acc)});
      }
    }
  };

  // SIMD fast path: hoist A/B/C into dense f32 buffers, then run the dense
  // MxNxK matmul as native-width FMA rows over the N (column) dimension.
  if constexpr (util::has_stdx_simd) {
    constexpr uint32_t W = static_cast<uint32_t>(util::native<float>::size());
    const uint32_t stride = ((N + W - 1) / W) * W;
    if (util::force_scalar() || static_cast<size_t>(M) * K > WMMA_SIMD_MAX_AB ||
        static_cast<size_t>(K) * stride > WMMA_SIMD_MAX_BSTRIDE ||
        static_cast<size_t>(M) * stride > WMMA_SIMD_MAX_C) {
      run_scalar();
    } else {
      alignas(64) float Abuf[WMMA_SIMD_MAX_AB] = {};
      alignas(64) float Bbuf[WMMA_SIMD_MAX_BSTRIDE] = {};
      alignas(64) float Cbuf[WMMA_SIMD_MAX_C] = {};
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t col = 0; col < N; ++col) {
          auto out = gfx12_wmma_output_loc_32(wave_size, M, N, row, col);
          Cbuf[row * stride + col] = apply_wmma_c_modifier(
              (const_acc != ACC_FROM_VGPR)
                  ? std::bit_cast<float>(const_acc)
                  : std::bit_cast<float>(cu.read_vgpr(s2 + out.reg, out.lane)),
              c_modifier);
        }
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t k = 0; k < K; ++k)
          Abuf[row * K + k] =
              ea(cu, s0, gfx12_wmma_a_input_loc(wave_size, M, K, row, k, a_bits, b_bits));
      for (uint32_t k = 0; k < K; ++k)
        for (uint32_t col = 0; col < N; ++col)
          Bbuf[k * stride + col] =
              eb(cu, s1, gfx12_wmma_b_input_loc(wave_size, N, K, col, k, a_bits, b_bits));
      wmma_simd_matmul<float>(M, N, K, W, stride, Abuf, Bbuf, Cbuf);
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t col = 0; col < N; ++col) {
          auto out = gfx12_wmma_output_loc_32(wave_size, M, N, row, col);
          results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(Cbuf[row * stride + col])});
        }
    }
  } else {
    run_scalar();
  }

  for (const auto &r : results)
    cu.write_vgpr(dst + r.reg, r.lane, r.val);
}

template <typename ExtractA, typename ExtractB>
void exec_gfx11_wmma_f32_mixed(amdgpu::ComputeUnitCore &cu, uint32_t wave_size, uint32_t M,
                               uint32_t N, uint32_t K, uint32_t a_bits, uint32_t b_bits,
                               uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2, ExtractA ea,
                               ExtractB eb, uint32_t const_acc = ACC_FROM_VGPR,
                               uint32_t c_modifier = 0) {
  require_gfx11_wmma_wave_size(wave_size);
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t val;
  };
  std::vector<Result> results;
  results.reserve(M * N);

  for (uint32_t row = 0; row < M; ++row) {
    for (uint32_t col = 0; col < N; ++col) {
      auto out = gfx11_wmma_output_loc_32(wave_size, M, N, row, col);
      uint32_t lane_group = out.lane / N;
      float acc = (const_acc != ACC_FROM_VGPR)
                      ? std::bit_cast<float>(const_acc)
                      : std::bit_cast<float>(cu.read_vgpr(s2 + out.reg, out.lane));
      acc = apply_wmma_c_modifier(acc, c_modifier);
      for (uint32_t k = 0; k < K; ++k) {
        auto al = gfx11_wmma_input_loc(M, K, row, k, a_bits, lane_group);
        auto bl = gfx11_wmma_input_loc(N, K, col, k, b_bits, lane_group);
        acc += ea(cu, s0, al) * eb(cu, s1, bl);
      }
      results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(acc)});
    }
  }

  for (const auto &r : results)
    cu.write_vgpr(dst + r.reg, r.lane, r.val);
}

template <typename ExtractA, typename ExtractB>
void exec_gfx11_wmma_f32(amdgpu::ComputeUnitCore &cu, uint32_t wave_size, uint32_t M, uint32_t N,
                         uint32_t K, uint32_t in_bits, uint32_t dst, uint32_t s0, uint32_t s1,
                         uint32_t s2, ExtractA ea, ExtractB eb, uint32_t const_acc = ACC_FROM_VGPR,
                         uint32_t c_modifier = 0) {
  exec_gfx11_wmma_f32_mixed(cu, wave_size, M, N, K, in_bits, in_bits, dst, s0, s1, s2, ea, eb,
                            const_acc, c_modifier);
}

template <typename ExtractA, typename ExtractB, typename ReadAcc, typename PackResult>
void exec_gfx11_wmma_packed16(amdgpu::ComputeUnitCore &cu, uint32_t wave_size, uint32_t M,
                              uint32_t N, uint32_t K, uint32_t in_bits, uint32_t dst, uint32_t s0,
                              uint32_t s1, uint32_t s2, uint32_t opsel, ExtractA ea, ExtractB eb,
                              ReadAcc read_acc, PackResult pack_result,
                              uint32_t const_acc = ACC_FROM_VGPR) {
  require_gfx11_wmma_wave_size(wave_size);
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t sub_element;
    uint16_t val;
  };
  std::vector<Result> results;
  results.reserve(M * N);

  for (uint32_t row = 0; row < M; ++row) {
    for (uint32_t col = 0; col < N; ++col) {
      auto out = gfx11_wmma_output_loc_16(wave_size, M, N, row, col, opsel);
      uint32_t lane_group = out.lane / N;
      float acc = (const_acc != ACC_FROM_VGPR)
                      ? std::bit_cast<float>(const_acc)
                      : read_acc(cu, s2 + out.reg, out.lane, out.sub_element);
      for (uint32_t k = 0; k < K; ++k) {
        auto al = gfx11_wmma_input_loc(M, K, row, k, in_bits, lane_group);
        auto bl = gfx11_wmma_input_loc(N, K, col, k, in_bits, lane_group);
        acc += ea(cu, s0, al) * eb(cu, s1, bl);
      }
      results.push_back({out.reg, out.lane, out.sub_element, pack_result(acc)});
    }
  }

  uint32_t max_reg = 0;
  for (const auto &r : results)
    max_reg = std::max(max_reg, r.reg);
  const uint32_t dst_regs = max_reg + 1;

  std::vector<uint32_t> words(dst_regs * wave_size, 0);
  std::vector<uint8_t> masks(dst_regs * wave_size, 0);
  for (const auto &r : results) {
    uint32_t idx = r.reg * wave_size + r.lane;
    uint32_t shift = r.sub_element * 16;
    words[idx] = (words[idx] & ~(0xFFFFu << shift)) | (static_cast<uint32_t>(r.val) << shift);
    masks[idx] |= 1u << r.sub_element;
  }
  for (uint32_t reg = 0; reg < dst_regs; ++reg) {
    for (uint32_t lane = 0; lane < wave_size; ++lane) {
      uint32_t idx = reg * wave_size + lane;
      uint32_t word = words[idx];
      if (masks[idx] != 0x3u) {
        uint32_t old = cu.read_vgpr(dst + reg, lane);
        if ((masks[idx] & 0x1u) == 0)
          word = (word & 0xFFFF0000u) | (old & 0x0000FFFFu);
        if ((masks[idx] & 0x2u) == 0)
          word = (word & 0x0000FFFFu) | (old & 0xFFFF0000u);
      }
      cu.write_vgpr(dst + reg, lane, word);
    }
  }
}

template <typename ExtractA, typename ExtractB>
void exec_gfx11_wmma_f16(amdgpu::ComputeUnitCore &cu, uint32_t wave_size, uint32_t M, uint32_t N,
                         uint32_t K, uint32_t in_bits, uint32_t dst, uint32_t s0, uint32_t s1,
                         uint32_t s2, uint32_t opsel, ExtractA ea, ExtractB eb,
                         uint32_t const_acc = ACC_FROM_VGPR) {
  exec_gfx11_wmma_packed16(
      cu, wave_size, M, N, K, in_bits, dst, s0, s1, s2, opsel, ea, eb,
      [](amdgpu::ComputeUnitCore &cu, uint32_t reg, uint32_t lane, uint32_t sub) {
        uint32_t raw = cu.read_vgpr(reg, lane);
        return util::f16_to_f32(static_cast<uint16_t>((raw >> (sub * 16)) & 0xFFFF));
      },
      [](float val) { return util::f32_to_f16(val); }, const_acc);
}

template <typename ExtractA, typename ExtractB>
void exec_gfx11_wmma_bf16(amdgpu::ComputeUnitCore &cu, uint32_t wave_size, uint32_t M, uint32_t N,
                          uint32_t K, uint32_t in_bits, uint32_t dst, uint32_t s0, uint32_t s1,
                          uint32_t s2, uint32_t opsel, ExtractA ea, ExtractB eb,
                          uint32_t const_acc = ACC_FROM_VGPR) {
  exec_gfx11_wmma_packed16(
      cu, wave_size, M, N, K, in_bits, dst, s0, s1, s2, opsel, ea, eb,
      [](amdgpu::ComputeUnitCore &cu, uint32_t reg, uint32_t lane, uint32_t sub) {
        uint32_t raw = cu.read_vgpr(reg, lane);
        return util::bf16_to_f32(static_cast<uint16_t>((raw >> (sub * 16)) & 0xFFFF));
      },
      [](float val) { return util::f32_to_bf16(val); }, const_acc);
}

template <typename ExtractA, typename ExtractB, typename ScaleAWord, typename ScaleBWord>
void exec_wmma_f32_scaled_mixed(amdgpu::ComputeUnitCore &cu, uint32_t M, uint32_t N, uint32_t K,
                                uint32_t a_bits, uint32_t b_bits, uint32_t dst, uint32_t s0,
                                uint32_t s1, uint32_t s2, ExtractA ea, ExtractB eb,
                                uint32_t const_acc, ScaleAWord scale_a_word,
                                ScaleBWord scale_b_word, uint32_t matrix_a_scale,
                                uint32_t matrix_b_scale, uint32_t matrix_a_scale_fmt,
                                uint32_t matrix_b_scale_fmt, bool scale16 = false,
                                uint32_t c_modifier = 0) {
  require_wmma_wave32(cu);
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t val;
  };
  std::vector<Result> results;
  results.reserve(M * N);
  const bool mixed_subbyte_a = a_bits < 8 && b_bits == 8;
  const bool mixed_subbyte_b = b_bits < 8 && a_bits == 8;
  const bool mixed_pair = mixed_subbyte_a || mixed_subbyte_b;

  auto scale_for = [](uint64_t scale_word, uint32_t scale_byte, uint32_t scale_fmt) -> float {
    return decode_wmma_scale_byte(static_cast<uint8_t>((scale_word >> (scale_byte * 8)) & 0xffu),
                                  scale_fmt);
  };

  auto run_scalar = [&]() {
    for (uint32_t row = 0; row < M; ++row) {
      for (uint32_t col = 0; col < N; ++col) {
        auto out = wmma_output_loc_32(M, N, row, col);
        float acc = (const_acc != ACC_FROM_VGPR)
                        ? std::bit_cast<float>(const_acc)
                        : std::bit_cast<float>(cu.read_vgpr(s2 + out.reg, out.lane));
        acc = apply_wmma_c_modifier(acc, c_modifier);
        const uint64_t a_scale_word =
            scale_a_word(wmma_a_scale_lane(M, K, row, matrix_a_scale, a_bits, b_bits));
        const uint64_t b_scale_word = scale_b_word(wmma_scale_lane(col, matrix_b_scale));
        for (uint32_t k = 0; k < K; ++k) {
          auto al = wmma_a_input_loc(M, K, row, k, a_bits, b_bits);
          auto bl = wmma_b_input_loc(N, K, col, k, a_bits, b_bits);
          const uint32_t a_scale_byte = wmma_f8f6f4_scale_byte(k, a_bits, mixed_pair, scale16);
          const uint32_t b_scale_byte = wmma_f8f6f4_scale_byte(k, b_bits, mixed_pair, scale16);
          acc += ea(cu, s0, al) * eb(cu, s1, bl) *
                 scale_for(a_scale_word, a_scale_byte, matrix_a_scale_fmt) *
                 scale_for(b_scale_word, b_scale_byte, matrix_b_scale_fmt);
        }
        results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(acc)});
      }
    }
  };

  if constexpr (util::has_stdx_simd) {
    constexpr uint32_t W = static_cast<uint32_t>(util::native<float>::size());
    const uint32_t stride = ((N + W - 1) / W) * W;
    if (util::force_scalar() || static_cast<size_t>(M) * K > WMMA_SIMD_MAX_AB ||
        static_cast<size_t>(K) * stride > WMMA_SIMD_MAX_BSTRIDE ||
        static_cast<size_t>(M) * stride > WMMA_SIMD_MAX_C) {
      run_scalar();
    } else {
      alignas(64) float Abuf[WMMA_SIMD_MAX_AB] = {};
      alignas(64) float Bbuf[WMMA_SIMD_MAX_BSTRIDE] = {};
      alignas(64) float Cbuf[WMMA_SIMD_MAX_C] = {};
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t col = 0; col < N; ++col) {
          auto out = wmma_output_loc_32(M, N, row, col);
          Cbuf[row * stride + col] = apply_wmma_c_modifier(
              (const_acc != ACC_FROM_VGPR)
                  ? std::bit_cast<float>(const_acc)
                  : std::bit_cast<float>(cu.read_vgpr(s2 + out.reg, out.lane)),
              c_modifier);
        }
      for (uint32_t row = 0; row < M; ++row) {
        const uint64_t a_scale_word =
            scale_a_word(wmma_a_scale_lane(M, K, row, matrix_a_scale, a_bits, b_bits));
        for (uint32_t k = 0; k < K; ++k) {
          auto al = wmma_a_input_loc(M, K, row, k, a_bits, b_bits);
          const uint32_t a_scale_byte = wmma_f8f6f4_scale_byte(k, a_bits, mixed_pair, scale16);
          Abuf[row * K + k] =
              ea(cu, s0, al) * scale_for(a_scale_word, a_scale_byte, matrix_a_scale_fmt);
        }
      }
      for (uint32_t col = 0; col < N; ++col) {
        const uint64_t b_scale_word = scale_b_word(wmma_scale_lane(col, matrix_b_scale));
        for (uint32_t k = 0; k < K; ++k) {
          auto bl = wmma_b_input_loc(N, K, col, k, a_bits, b_bits);
          const uint32_t b_scale_byte = wmma_f8f6f4_scale_byte(k, b_bits, mixed_pair, scale16);
          Bbuf[k * stride + col] =
              eb(cu, s1, bl) * scale_for(b_scale_word, b_scale_byte, matrix_b_scale_fmt);
        }
      }
      wmma_simd_matmul<float>(M, N, K, W, stride, Abuf, Bbuf, Cbuf);
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t col = 0; col < N; ++col) {
          auto out = wmma_output_loc_32(M, N, row, col);
          results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(Cbuf[row * stride + col])});
        }
    }
  } else {
    run_scalar();
  }

  for (const auto &r : results)
    cu.write_vgpr(dst + r.reg, r.lane, r.val);
}

template <typename ExtractA, typename ExtractB>
void exec_wmma_f32(amdgpu::ComputeUnitCore &cu, uint32_t M, uint32_t N, uint32_t K,
                   uint32_t in_bits, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                   ExtractA ea, ExtractB eb, uint32_t const_acc = ACC_FROM_VGPR,
                   uint32_t c_modifier = 0, uint32_t wave_size = WMMA_WAVE32) {
  exec_wmma_f32_mixed(cu, M, N, K, in_bits, in_bits, dst, s0, s1, s2, ea, eb, const_acc, c_modifier,
                      wave_size);
}

/// Fast path for v_wmma_f32_16x16x32_f16 (gfx1250, wave32) — the WMMA analogue
/// of exec_f32_mfma_16x16x32_f16. Compile-time M/N/K let the compiler fully
/// unroll the 16-row x 32-K matmul into straight-line AVX-512 FMAs; the f16
/// inputs are bulk-converted once with F16C (one vector op per 16 halves)
/// instead of branchy per-element extract_f16; VGPRs are accessed through one
/// vgpr_data base pointer per operand and the result scatters directly (no
/// Result staging vector). Falls back to the generic exec_wmma_f32 without
/// AVX-512 / under force-scalar.
inline void exec_wmma_f32_16x16x32_f16(amdgpu::ComputeUnitCore &cu, uint32_t dst, uint32_t s0,
                                       uint32_t s1, uint32_t s2, uint32_t const_acc = ACC_FROM_VGPR,
                                       uint32_t c_modifier = 0) {
  constexpr uint32_t M = 16, N = 16, K = 32, in_bits = 16;
  if constexpr (!util::has_stdx_simd) {
    exec_wmma_f32(cu, M, N, K, in_bits, dst, s0, s1, s2, amdgpu::extract_f16, amdgpu::extract_f16,
                  const_acc, c_modifier);
    return;
  } else {
    if (util::force_scalar() || util::native<float>::size() != 16) {
      exec_wmma_f32(cu, M, N, K, in_bits, dst, s0, s1, s2, amdgpu::extract_f16, amdgpu::extract_f16,
                    const_acc, c_modifier);
      return;
    }
    require_wmma_wave32(cu);
    const uint32_t wf = cu.wf_size();
    const uint32_t *a_words = reinterpret_cast<const uint32_t *>(cu.vgpr_data(s0));
    const uint32_t *b_words = reinterpret_cast<const uint32_t *>(cu.vgpr_data(s1));
    const uint32_t *c_words = reinterpret_cast<const uint32_t *>(cu.vgpr_data(s2));
    alignas(64) float A_buf[M * K]; // A[row][k]
    alignas(64) float B_buf[K * N]; // B[k][col]
    alignas(64) float C_buf[M * N]; // C[row][col]
    // A and B each occupy 8 VGPRs x wf lanes = 2*8*wf packed f16 (16 f16/lane
    // for K=32). Bulk-convert the region to f32 once, then the hoist is a pure
    // f32 index-shuffle (f16 j of word w sub s -> flat (w*wf+lane)*2+s).
    constexpr uint32_t NUM_IN_REGS = 8;
    const uint32_t n_halves = 2 * NUM_IN_REGS * wf;
    alignas(64) float A_f32[2 * NUM_IN_REGS * 64];
    alignas(64) float B_f32[2 * NUM_IN_REGS * 64];
    util::f16_to_f32_block(reinterpret_cast<const uint16_t *>(a_words), A_f32, n_halves);
    util::f16_to_f32_block(reinterpret_cast<const uint16_t *>(b_words), B_f32, n_halves);
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = wmma_output_loc_32(M, N, row, col);
        C_buf[row * N + col] = apply_wmma_c_modifier(
            (const_acc != ACC_FROM_VGPR) ? std::bit_cast<float>(const_acc)
                                         : std::bit_cast<float>(c_words[out.reg * wf + out.lane]),
            c_modifier);
      }
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t k = 0; k < K; ++k) {
        auto al = wmma_input_loc(M, K, row, k, in_bits);
        A_buf[row * K + k] = A_f32[(al.vgpr_offset * wf + al.lane) * 2 + al.sub_element];
      }
    for (uint32_t k = 0; k < K; ++k)
      for (uint32_t col = 0; col < N; ++col) {
        auto bl = wmma_input_loc(N, K, col, k, in_bits);
        B_buf[k * N + col] = B_f32[(bl.vgpr_offset * wf + bl.lane) * 2 + bl.sub_element];
      }
    // Dense 16x32 * 32x16 -> 16x16 matmul, 16-lane stdx FMA per row.
    for (uint32_t row = 0; row < M; ++row) {
      util::native<float> c_row;
      c_row.copy_from(&C_buf[row * N], util::stdx::vector_aligned);
      for (uint32_t k = 0; k < K; ++k) {
        util::native<float> a_bcast(A_buf[row * K + k]);
        util::native<float> b_row;
        b_row.copy_from(&B_buf[k * N], util::stdx::vector_aligned);
        c_row = util::stdx::fma(a_bcast, b_row, c_row);
      }
      c_row.copy_to(&C_buf[row * N], util::stdx::vector_aligned);
    }
    // Scatter directly back to VGPRs (no Result staging vector).
    uint32_t *d_words = reinterpret_cast<uint32_t *>(cu.vgpr_data(dst));
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = wmma_output_loc_32(M, N, row, col);
        d_words[out.reg * wf + out.lane] = std::bit_cast<uint32_t>(C_buf[row * N + col]);
      }
  }
}

/// Fast path for v_wmma_f32_16x16x32_bf16 / v_wmma_bf16f32_16x16x32_bf16
/// (gfx1250, wave32). Identical to exec_wmma_f32_16x16x32_f16 except the bulk
/// input convert is the bf16 zero-extend (no F16C needed). Falls back to the
/// generic exec_wmma_f32 without AVX-512 / under force-scalar.
inline void exec_wmma_f32_16x16x32_bf16(amdgpu::ComputeUnitCore &cu, uint32_t dst, uint32_t s0,
                                        uint32_t s1, uint32_t s2,
                                        uint32_t const_acc = ACC_FROM_VGPR,
                                        uint32_t c_modifier = 0) {
  constexpr uint32_t M = 16, N = 16, K = 32, in_bits = 16;
  if constexpr (!util::has_stdx_simd) {
    exec_wmma_f32(cu, M, N, K, in_bits, dst, s0, s1, s2, amdgpu::extract_bf16, amdgpu::extract_bf16,
                  const_acc, c_modifier);
    return;
  } else {
    if (util::force_scalar() || util::native<float>::size() != 16) {
      exec_wmma_f32(cu, M, N, K, in_bits, dst, s0, s1, s2, amdgpu::extract_bf16,
                    amdgpu::extract_bf16, const_acc, c_modifier);
      return;
    }
    require_wmma_wave32(cu);
    const uint32_t wf = cu.wf_size();
    const uint32_t *a_words = reinterpret_cast<const uint32_t *>(cu.vgpr_data(s0));
    const uint32_t *b_words = reinterpret_cast<const uint32_t *>(cu.vgpr_data(s1));
    const uint32_t *c_words = reinterpret_cast<const uint32_t *>(cu.vgpr_data(s2));
    alignas(64) float A_buf[M * K]; // A[row][k]
    alignas(64) float B_buf[K * N]; // B[k][col]
    alignas(64) float C_buf[M * N]; // C[row][col]
    // A and B each occupy 8 VGPRs x wf lanes = 2*8*wf packed bf16 (16 bf16/lane
    // for K=32). Bulk-convert the region to f32 once, then the hoist is a pure
    // f32 index-shuffle (bf16 j of word w sub s -> flat (w*wf+lane)*2+s).
    constexpr uint32_t NUM_IN_REGS = 8;
    const uint32_t n_halves = 2 * NUM_IN_REGS * wf;
    alignas(64) float A_f32[2 * NUM_IN_REGS * 64];
    alignas(64) float B_f32[2 * NUM_IN_REGS * 64];
    util::bf16_to_f32_block(reinterpret_cast<const uint16_t *>(a_words), A_f32, n_halves);
    util::bf16_to_f32_block(reinterpret_cast<const uint16_t *>(b_words), B_f32, n_halves);
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = wmma_output_loc_32(M, N, row, col);
        C_buf[row * N + col] = apply_wmma_c_modifier(
            (const_acc != ACC_FROM_VGPR) ? std::bit_cast<float>(const_acc)
                                         : std::bit_cast<float>(c_words[out.reg * wf + out.lane]),
            c_modifier);
      }
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t k = 0; k < K; ++k) {
        auto al = wmma_input_loc(M, K, row, k, in_bits);
        A_buf[row * K + k] = A_f32[(al.vgpr_offset * wf + al.lane) * 2 + al.sub_element];
      }
    for (uint32_t k = 0; k < K; ++k)
      for (uint32_t col = 0; col < N; ++col) {
        auto bl = wmma_input_loc(N, K, col, k, in_bits);
        B_buf[k * N + col] = B_f32[(bl.vgpr_offset * wf + bl.lane) * 2 + bl.sub_element];
      }
    // Dense 16x32 * 32x16 -> 16x16 matmul, 16-lane stdx FMA per row.
    for (uint32_t row = 0; row < M; ++row) {
      util::native<float> c_row;
      c_row.copy_from(&C_buf[row * N], util::stdx::vector_aligned);
      for (uint32_t k = 0; k < K; ++k) {
        util::native<float> a_bcast(A_buf[row * K + k]);
        util::native<float> b_row;
        b_row.copy_from(&B_buf[k * N], util::stdx::vector_aligned);
        c_row = util::stdx::fma(a_bcast, b_row, c_row);
      }
      c_row.copy_to(&C_buf[row * N], util::stdx::vector_aligned);
    }
    // Scatter directly back to VGPRs (no Result staging vector).
    uint32_t *d_words = reinterpret_cast<uint32_t *>(cu.vgpr_data(dst));
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = wmma_output_loc_32(M, N, row, col);
        d_words[out.reg * wf + out.lane] = std::bit_cast<uint32_t>(C_buf[row * N + col]);
      }
  }
}

// CDNA3 uses the AMD FNUZ format for fp8/bf8
template <bool FP8, bool FNUZ> constexpr auto f8_extract_fn() {
  if constexpr (FP8) {
    if constexpr (FNUZ)
      return &extract_fp8_fnuz;
    else
      return &extract_fp8_ocp;
  } else {
    if constexpr (FNUZ)
      return &extract_bf8_fnuz;
    else
      return &extract_bf8_ocp;
  }
}

/// Compile-time fp8 (e4m3) vs bf8 (e5m2), OCP vs FNUZ bulk-convert selector for
/// the f8 spec kernels: converts `n` packed bytes starting at `words` to f32
/// through 256-entry LUTs matching the selected extractor.
template <bool FP8, bool FNUZ = false>
void f8_to_f32_block(const uint32_t *words, float *dst, size_t n) {
  if constexpr (FP8 && FNUZ)
    util::fp8_e4m3_fnuz_to_f32_block(reinterpret_cast<const uint8_t *>(words), dst, n);
  else if constexpr (FP8)
    util::fp8_e4m3_ocp_to_f32_block(reinterpret_cast<const uint8_t *>(words), dst, n);
  else if constexpr (FNUZ)
    util::bf8_e5m2_fnuz_to_f32_block(reinterpret_cast<const uint8_t *>(words), dst, n);
  else
    util::bf8_e5m2_ocp_to_f32_block(reinterpret_cast<const uint8_t *>(words), dst, n);
}

/// Fast path for the dense f32-out fp8/bf8-input WMMA shapes
/// (v_wmma_f32_16x16x{64,128}_{fp8,bf8}_{fp8,bf8}, gfx1250, wave32). Identical
/// to the f16/bf16 specializations except the bulk input convert is the f8 LUT
/// gather; A/B formats are compile-time selected. Sparse SWMMAC stays generic.
/// Falls back to the generic exec_wmma_f32_mixed without AVX-512 / under
/// force-scalar.
template <uint32_t M, uint32_t N, uint32_t K, bool A_FP8, bool B_FP8, bool FNUZ = false>
void exec_wmma_f32_f8_spec(amdgpu::ComputeUnitCore &cu, uint32_t dst, uint32_t s0, uint32_t s1,
                           uint32_t s2, uint32_t const_acc = ACC_FROM_VGPR,
                           uint32_t c_modifier = 0) {
  constexpr uint32_t in_bits = 8;
  static_assert(N % 16 == 0, "specialized f8 WMMA assumes N is a multiple of the zmm width");
  constexpr auto ea = f8_extract_fn<A_FP8, FNUZ>();
  constexpr auto eb = f8_extract_fn<B_FP8, FNUZ>();
  auto fallback = [&]() {
    exec_wmma_f32_mixed(cu, M, N, K, in_bits, in_bits, dst, s0, s1, s2, ea, eb, const_acc,
                        c_modifier);
  };
  if constexpr (!util::has_stdx_simd) {
    fallback();
    return;
  } else {
    if (util::force_scalar() || util::native<float>::size() != 16) {
      fallback();
      return;
    }
    require_wmma_wave32(cu);
    constexpr uint32_t W = 16;
    const uint32_t wf = cu.wf_size();
    const uint32_t *a_words = reinterpret_cast<const uint32_t *>(cu.vgpr_data(s0));
    const uint32_t *b_words = reinterpret_cast<const uint32_t *>(cu.vgpr_data(s1));
    const uint32_t *c_words = reinterpret_cast<const uint32_t *>(cu.vgpr_data(s2));
    uint32_t *d_words = reinterpret_cast<uint32_t *>(cu.vgpr_data(dst));
    alignas(64) float A_buf[M * K]; // A[row][k]
    alignas(64) float B_buf[K * N]; // B[k][col]
    alignas(64) float C_buf[M * N]; // C[row][col]
    // Bulk-convert each whole packed f8 region to f32 once, then the hoist is a
    // pure f32 index-shuffle (byte of word w lane l sub s -> (w*wf+l)*4+s).
    alignas(64) float A_f32[M * K];
    alignas(64) float B_f32[N * K];
    f8_to_f32_block<A_FP8, FNUZ>(a_words, A_f32, M * K);
    f8_to_f32_block<B_FP8, FNUZ>(b_words, B_f32, N * K);
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = wmma_output_loc_32(M, N, row, col);
        C_buf[row * N + col] = apply_wmma_c_modifier(
            (const_acc != ACC_FROM_VGPR) ? std::bit_cast<float>(const_acc)
                                         : std::bit_cast<float>(c_words[out.reg * wf + out.lane]),
            c_modifier);
      }
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t k = 0; k < K; ++k) {
        auto al = wmma_a_input_loc(M, K, row, k, in_bits, in_bits);
        A_buf[row * K + k] = A_f32[(al.vgpr_offset * wf + al.lane) * 4 + al.sub_element];
      }
    for (uint32_t k = 0; k < K; ++k)
      for (uint32_t col = 0; col < N; ++col) {
        auto bl = wmma_b_input_loc(N, K, col, k, in_bits, in_bits);
        B_buf[k * N + col] = B_f32[(bl.vgpr_offset * wf + bl.lane) * 4 + bl.sub_element];
      }
    // Dense MxKxN matmul, W-lane (zmm) stdx FMA over N (N/W chunks per row).
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t c0 = 0; c0 < N; c0 += W) {
        util::native<float> c_row;
        c_row.copy_from(&C_buf[row * N + c0], util::stdx::vector_aligned);
        for (uint32_t k = 0; k < K; ++k) {
          util::native<float> a_bcast(A_buf[row * K + k]);
          util::native<float> b_row;
          b_row.copy_from(&B_buf[k * N + c0], util::stdx::vector_aligned);
          c_row = util::stdx::fma(a_bcast, b_row, c_row);
        }
        c_row.copy_to(&C_buf[row * N + c0], util::stdx::vector_aligned);
      }
    // Scatter directly back to VGPRs (no Result staging vector).
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = wmma_output_loc_32(M, N, row, col);
        d_words[out.reg * wf + out.lane] = std::bit_cast<uint32_t>(C_buf[row * N + col]);
      }
  }
}

/// Fast path for the f32-input WMMA shapes (v_wmma_f32_*_f32). f32 inputs, so no
/// F16C convert — the hoist reads each operand word straight through vgpr_data.
/// Compile-time M/N/K fully unroll the matmul, looped over N in zmm-width chunks
/// (N a multiple of 16). Falls back to generic exec_wmma_f32 without AVX-512 /
/// under force-scalar.
template <uint32_t M, uint32_t N, uint32_t K>
void exec_wmma_f32_f32_spec(amdgpu::ComputeUnitCore &cu, uint32_t dst, uint32_t s0, uint32_t s1,
                            uint32_t s2, uint32_t const_acc = ACC_FROM_VGPR,
                            uint32_t c_modifier = 0) {
  constexpr uint32_t in_bits = 32;
  static_assert(N % 16 == 0, "specialized f32 WMMA assumes N is a multiple of the zmm width");
  if constexpr (!util::has_stdx_simd) {
    exec_wmma_f32(cu, M, N, K, in_bits, dst, s0, s1, s2, amdgpu::extract_f32, amdgpu::extract_f32,
                  const_acc, c_modifier);
    return;
  } else {
    if (util::force_scalar() || util::native<float>::size() != 16) {
      exec_wmma_f32(cu, M, N, K, in_bits, dst, s0, s1, s2, amdgpu::extract_f32, amdgpu::extract_f32,
                    const_acc, c_modifier);
      return;
    }
    require_wmma_wave32(cu);
    constexpr uint32_t W = 16;
    const uint32_t wf = cu.wf_size();
    const uint32_t *a_words = reinterpret_cast<const uint32_t *>(cu.vgpr_data(s0));
    const uint32_t *b_words = reinterpret_cast<const uint32_t *>(cu.vgpr_data(s1));
    const uint32_t *c_words = reinterpret_cast<const uint32_t *>(cu.vgpr_data(s2));
    uint32_t *d_words = reinterpret_cast<uint32_t *>(cu.vgpr_data(dst));
    alignas(64) float A_buf[M * K];
    alignas(64) float B_buf[K * N];
    alignas(64) float C_buf[M * N];
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = wmma_output_loc_32(M, N, row, col);
        C_buf[row * N + col] = apply_wmma_c_modifier(
            (const_acc != ACC_FROM_VGPR) ? std::bit_cast<float>(const_acc)
                                         : std::bit_cast<float>(c_words[out.reg * wf + out.lane]),
            c_modifier);
      }
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t k = 0; k < K; ++k) {
        auto al = wmma_input_loc(M, K, row, k, in_bits);
        A_buf[row * K + k] = std::bit_cast<float>(a_words[al.vgpr_offset * wf + al.lane]);
      }
    for (uint32_t k = 0; k < K; ++k)
      for (uint32_t col = 0; col < N; ++col) {
        auto bl = wmma_input_loc(N, K, col, k, in_bits);
        B_buf[k * N + col] = std::bit_cast<float>(b_words[bl.vgpr_offset * wf + bl.lane]);
      }
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t c0 = 0; c0 < N; c0 += W) {
        util::native<float> c_row;
        c_row.copy_from(&C_buf[row * N + c0], util::stdx::vector_aligned);
        for (uint32_t k = 0; k < K; ++k) {
          util::native<float> a_bcast(A_buf[row * K + k]);
          util::native<float> b_row;
          b_row.copy_from(&B_buf[k * N + c0], util::stdx::vector_aligned);
          c_row = util::stdx::fma(a_bcast, b_row, c_row);
        }
        c_row.copy_to(&C_buf[row * N + c0], util::stdx::vector_aligned);
      }
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = wmma_output_loc_32(M, N, row, col);
        d_words[out.reg * wf + out.lane] = std::bit_cast<uint32_t>(C_buf[row * N + col]);
      }
  }
}

template <typename ExtractA, typename ExtractB>
void exec_swmmac_f32_mixed(amdgpu::ComputeUnitCore &cu, uint32_t M, uint32_t N, uint32_t K,
                           uint32_t a_bits, uint32_t b_bits, uint32_t dst, uint32_t s0, uint32_t s1,
                           uint32_t acc_base, uint32_t index_base, uint32_t index_entries,
                           uint32_t index_key, ExtractA ea, ExtractB eb,
                           uint32_t const_acc = ACC_FROM_VGPR, uint32_t wave_size = WMMA_WAVE32) {
  require_gfx12_wmma_wave_size(wave_size);
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t val;
  };
  std::vector<Result> results;
  results.reserve(M * N);

  const uint32_t compressed_k = K / 2;

  // dense_k depends on (row, ck) via the row's metadata, not on col — so the
  // gathered B column is shared across all col for a fixed row. Resolve it once.
  auto dense_k_for = [&](uint32_t row, uint32_t ck) -> uint32_t {
    const auto index_loc = swmmac_index_loc(wave_size, M, K, a_bits, row, ck, index_entries);
    const uint64_t index_set =
        read_swmmac_index_set(cu, index_base, index_loc.lane, index_entries, index_key);
    return swmmac_dense_k(index_set, ck, index_loc.local_compressed_k);
  };

  auto run_scalar = [&]() {
    for (uint32_t row = 0; row < M; ++row) {
      for (uint32_t col = 0; col < N; ++col) {
        auto out = gfx12_wmma_output_loc_32(wave_size, M, N, row, col);
        float acc = (const_acc != ACC_FROM_VGPR)
                        ? std::bit_cast<float>(const_acc)
                        : std::bit_cast<float>(cu.read_vgpr(acc_base + out.reg, out.lane));
        for (uint32_t ck = 0; ck < compressed_k; ++ck) {
          auto al = swmmac_a_input_loc(wave_size, M, K, row, ck, a_bits);
          auto bl = swmmac_b_input_loc(wave_size, N, K, col, dense_k_for(row, ck), b_bits);
          acc += ea(cu, s0, al) * eb(cu, s1, bl);
        }
        results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(acc)});
      }
    }
  };

  // SIMD fast path: because the B gather is row-dependent, hoist each row's A
  // (compressed_k) and B (compressed_k x col, with dense_k folded in) into dense
  // buffers, then run that single row through the shared matmul core (M=1).
  if constexpr (util::has_stdx_simd) {
    constexpr uint32_t W = static_cast<uint32_t>(util::native<float>::size());
    const uint32_t stride = ((N + W - 1) / W) * W;
    if (util::force_scalar() || static_cast<size_t>(compressed_k) > WMMA_SIMD_MAX_AB ||
        static_cast<size_t>(compressed_k) * stride > WMMA_SIMD_MAX_BSTRIDE ||
        stride > WMMA_SIMD_MAX_C) {
      run_scalar();
    } else {
      alignas(64) float Abuf[WMMA_SIMD_MAX_AB] = {};
      alignas(64) float Bbuf[WMMA_SIMD_MAX_BSTRIDE] = {};
      alignas(64) float Cbuf[WMMA_SIMD_MAX_C] = {};
      for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t col = 0; col < N; ++col) {
          auto out = gfx12_wmma_output_loc_32(wave_size, M, N, row, col);
          Cbuf[col] = (const_acc != ACC_FROM_VGPR)
                          ? std::bit_cast<float>(const_acc)
                          : std::bit_cast<float>(cu.read_vgpr(acc_base + out.reg, out.lane));
        }
        for (uint32_t ck = 0; ck < compressed_k; ++ck) {
          Abuf[ck] = ea(cu, s0, swmmac_a_input_loc(wave_size, M, K, row, ck, a_bits));
          const uint32_t dense_k = dense_k_for(row, ck);
          for (uint32_t col = 0; col < N; ++col)
            Bbuf[ck * stride + col] =
                eb(cu, s1, swmmac_b_input_loc(wave_size, N, K, col, dense_k, b_bits));
        }
        wmma_simd_matmul<float>(1, N, compressed_k, W, stride, Abuf, Bbuf, Cbuf);
        for (uint32_t col = 0; col < N; ++col) {
          auto out = gfx12_wmma_output_loc_32(wave_size, M, N, row, col);
          results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(Cbuf[col])});
        }
      }
    }
  } else {
    run_scalar();
  }

  for (const auto &r : results)
    cu.write_vgpr(dst + r.reg, r.lane, r.val);
}

template <typename ExtractA, typename ExtractB>
void exec_swmmac_f32(amdgpu::ComputeUnitCore &cu, uint32_t M, uint32_t N, uint32_t K,
                     uint32_t in_bits, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t acc_base,
                     uint32_t index_base, uint32_t index_entries, uint32_t index_key, ExtractA ea,
                     ExtractB eb, uint32_t const_acc = ACC_FROM_VGPR,
                     uint32_t wave_size = WMMA_WAVE32) {
  exec_swmmac_f32_mixed(cu, M, N, K, in_bits, in_bits, dst, s0, s1, acc_base, index_base,
                        index_entries, index_key, ea, eb, const_acc, wave_size);
}

template <typename ExtractA, typename ExtractB, typename ReadAcc, typename PackResult>
void exec_wmma_packed16(amdgpu::ComputeUnitCore &cu, uint32_t M, uint32_t N, uint32_t K,
                        uint32_t in_bits, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                        ExtractA ea, ExtractB eb, ReadAcc read_acc, PackResult pack_result,
                        uint32_t const_acc = ACC_FROM_VGPR, uint32_t wave_size = WMMA_WAVE32) {
  require_gfx12_wmma_wave_size(wave_size);
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t sub_element;
    uint16_t val;
  };
  std::vector<Result> results;
  results.reserve(M * N);

  auto run_scalar = [&]() {
    for (uint32_t row = 0; row < M; ++row) {
      for (uint32_t col = 0; col < N; ++col) {
        auto out = gfx12_wmma_output_loc_16(wave_size, M, N, row, col);
        float acc = (const_acc != ACC_FROM_VGPR)
                        ? std::bit_cast<float>(const_acc)
                        : read_acc(cu, s2 + out.reg, out.lane, out.sub_element);
        for (uint32_t k = 0; k < K; ++k) {
          auto al = gfx12_wmma_input_loc(wave_size, M, K, row, k, in_bits);
          auto bl = gfx12_wmma_input_loc(wave_size, N, K, col, k, in_bits);
          acc += ea(cu, s0, al) * eb(cu, s1, bl);
        }
        results.push_back({out.reg, out.lane, out.sub_element, pack_result(acc)});
      }
    }
  };

  // SIMD fast path: run the f32 matmul vectorized over N into a dense grid, then
  // pack each result to 16 bits via the caller's pack_result. The masked 2-per-
  // word scatter below is unchanged.
  if constexpr (util::has_stdx_simd) {
    constexpr uint32_t W = static_cast<uint32_t>(util::native<float>::size());
    const uint32_t stride = ((N + W - 1) / W) * W;
    if (util::force_scalar() || static_cast<size_t>(M) * K > WMMA_SIMD_MAX_AB ||
        static_cast<size_t>(K) * stride > WMMA_SIMD_MAX_BSTRIDE ||
        static_cast<size_t>(M) * stride > WMMA_SIMD_MAX_C) {
      run_scalar();
    } else {
      alignas(64) float Abuf[WMMA_SIMD_MAX_AB] = {};
      alignas(64) float Bbuf[WMMA_SIMD_MAX_BSTRIDE] = {};
      alignas(64) float Cbuf[WMMA_SIMD_MAX_C] = {};
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t col = 0; col < N; ++col) {
          auto out = gfx12_wmma_output_loc_16(wave_size, M, N, row, col);
          Cbuf[row * stride + col] = (const_acc != ACC_FROM_VGPR)
                                         ? std::bit_cast<float>(const_acc)
                                         : read_acc(cu, s2 + out.reg, out.lane, out.sub_element);
        }
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t k = 0; k < K; ++k)
          Abuf[row * K + k] = ea(cu, s0, gfx12_wmma_input_loc(wave_size, M, K, row, k, in_bits));
      for (uint32_t k = 0; k < K; ++k)
        for (uint32_t col = 0; col < N; ++col)
          Bbuf[k * stride + col] =
              eb(cu, s1, gfx12_wmma_input_loc(wave_size, N, K, col, k, in_bits));
      wmma_simd_matmul<float>(M, N, K, W, stride, Abuf, Bbuf, Cbuf);
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t col = 0; col < N; ++col) {
          auto out = gfx12_wmma_output_loc_16(wave_size, M, N, row, col);
          results.push_back(
              {out.reg, out.lane, out.sub_element, pack_result(Cbuf[row * stride + col])});
        }
    }
  } else {
    run_scalar();
  }

  uint32_t dst_regs = ((M * N) / wave_size + 1) / 2;
  std::vector<uint32_t> words(dst_regs * wave_size, 0);
  std::vector<uint8_t> masks(dst_regs * wave_size, 0);
  for (const auto &r : results) {
    uint32_t idx = r.reg * wave_size + r.lane;
    uint32_t shift = r.sub_element * 16;
    words[idx] = (words[idx] & ~(0xFFFFu << shift)) | (static_cast<uint32_t>(r.val) << shift);
    masks[idx] |= 1u << r.sub_element;
  }
  for (uint32_t reg = 0; reg < dst_regs; ++reg) {
    for (uint32_t lane = 0; lane < wave_size; ++lane) {
      uint32_t idx = reg * wave_size + lane;
      uint32_t word = words[idx];
      if (masks[idx] != 0x3u) {
        uint32_t old = cu.read_vgpr(dst + reg, lane);
        if ((masks[idx] & 0x1u) == 0)
          word = (word & 0xFFFF0000u) | (old & 0x0000FFFFu);
        if ((masks[idx] & 0x2u) == 0)
          word = (word & 0x0000FFFFu) | (old & 0xFFFF0000u);
      }
      cu.write_vgpr(dst + reg, lane, word);
    }
  }
}

inline void exec_wmma_bf16f32_16x16x32_bf16(amdgpu::ComputeUnitCore &cu, uint32_t dst, uint32_t s0,
                                            uint32_t s1, uint32_t s2,
                                            uint32_t const_acc = ACC_FROM_VGPR,
                                            uint32_t c_modifier = 0) {
  constexpr uint32_t M = 16, N = 16, K = 32, in_bits = 16;
  require_wmma_wave32(cu);
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t sub_element;
    uint16_t val;
  };
  std::vector<Result> results;
  results.reserve(M * N);

  for (uint32_t row = 0; row < M; ++row) {
    for (uint32_t col = 0; col < N; ++col) {
      auto cout = wmma_output_loc_32(M, N, row, col);
      float acc = (const_acc != ACC_FROM_VGPR)
                      ? std::bit_cast<float>(const_acc)
                      : std::bit_cast<float>(cu.read_vgpr(s2 + cout.reg, cout.lane));
      acc = apply_wmma_c_modifier(acc, c_modifier);
      for (uint32_t k = 0; k < K; ++k) {
        auto al = wmma_input_loc(M, K, row, k, in_bits);
        auto bl = wmma_input_loc(N, K, col, k, in_bits);
        acc += extract_bf16(cu, s0, al) * extract_bf16(cu, s1, bl);
      }
      auto out = wmma_output_loc_16(M, N, row, col);
      results.push_back({out.reg, out.lane, out.sub_element, util::f32_to_bf16(acc)});
    }
  }

  uint32_t dst_regs = ((M * N) / WMMA_WAVE32 + 1) / 2;
  std::vector<uint32_t> words(dst_regs * WMMA_WAVE32, 0);
  std::vector<uint8_t> masks(dst_regs * WMMA_WAVE32, 0);
  for (const auto &r : results) {
    uint32_t idx = r.reg * WMMA_WAVE32 + r.lane;
    uint32_t shift = r.sub_element * 16;
    words[idx] = (words[idx] & ~(0xFFFFu << shift)) | (static_cast<uint32_t>(r.val) << shift);
    masks[idx] |= 1u << r.sub_element;
  }
  for (uint32_t reg = 0; reg < dst_regs; ++reg) {
    for (uint32_t lane = 0; lane < WMMA_WAVE32; ++lane) {
      uint32_t idx = reg * WMMA_WAVE32 + lane;
      uint32_t word = words[idx];
      if (masks[idx] != 0x3u) {
        uint32_t old = cu.read_vgpr(dst + reg, lane);
        if ((masks[idx] & 0x1u) == 0)
          word = (word & 0xFFFF0000u) | (old & 0x0000FFFFu);
        if ((masks[idx] & 0x2u) == 0)
          word = (word & 0x0000FFFFu) | (old & 0xFFFF0000u);
      }
      cu.write_vgpr(dst + reg, lane, word);
    }
  }
}

template <typename ExtractA, typename ExtractB, typename ReadAcc, typename PackResult>
void exec_swmmac_packed16(amdgpu::ComputeUnitCore &cu, uint32_t M, uint32_t N, uint32_t K,
                          uint32_t in_bits, uint32_t dst, uint32_t s0, uint32_t s1,
                          uint32_t acc_base, uint32_t index_base, uint32_t index_entries,
                          uint32_t index_key, ExtractA ea, ExtractB eb, ReadAcc read_acc,
                          PackResult pack_result, uint32_t const_acc = ACC_FROM_VGPR,
                          uint32_t wave_size = WMMA_WAVE32) {
  require_gfx12_wmma_wave_size(wave_size);
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t sub_element;
    uint16_t val;
  };
  std::vector<Result> results;
  results.reserve(M * N);

  const uint32_t compressed_k = K / 2;

  auto dense_k_for = [&](uint32_t row, uint32_t ck) -> uint32_t {
    const auto index_loc = swmmac_index_loc(wave_size, M, K, in_bits, row, ck, index_entries);
    const uint64_t index_set =
        read_swmmac_index_set(cu, index_base, index_loc.lane, index_entries, index_key);
    return swmmac_dense_k(index_set, ck, index_loc.local_compressed_k);
  };

  auto run_scalar = [&]() {
    for (uint32_t row = 0; row < M; ++row) {
      for (uint32_t col = 0; col < N; ++col) {
        auto out = gfx12_wmma_output_loc_16(wave_size, M, N, row, col);
        float acc = (const_acc != ACC_FROM_VGPR)
                        ? std::bit_cast<float>(const_acc)
                        : read_acc(cu, acc_base + out.reg, out.lane, out.sub_element);
        for (uint32_t ck = 0; ck < compressed_k; ++ck) {
          auto al = swmmac_a_input_loc(wave_size, M, K, row, ck, in_bits);
          auto bl = swmmac_b_input_loc(wave_size, N, K, col, dense_k_for(row, ck), in_bits);
          acc += ea(cu, s0, al) * eb(cu, s1, bl);
        }
        results.push_back({out.reg, out.lane, out.sub_element, pack_result(acc)});
      }
    }
  };

  // SIMD fast path: per-row gather (dense_k is row-dependent) into dense f32
  // buffers, single row through the matmul core (M=1), pack to 16 bits.
  if constexpr (util::has_stdx_simd) {
    constexpr uint32_t W = static_cast<uint32_t>(util::native<float>::size());
    const uint32_t stride = ((N + W - 1) / W) * W;
    if (util::force_scalar() || static_cast<size_t>(compressed_k) > WMMA_SIMD_MAX_AB ||
        static_cast<size_t>(compressed_k) * stride > WMMA_SIMD_MAX_BSTRIDE ||
        stride > WMMA_SIMD_MAX_C) {
      run_scalar();
    } else {
      alignas(64) float Abuf[WMMA_SIMD_MAX_AB] = {};
      alignas(64) float Bbuf[WMMA_SIMD_MAX_BSTRIDE] = {};
      alignas(64) float Cbuf[WMMA_SIMD_MAX_C] = {};
      for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t col = 0; col < N; ++col) {
          auto out = gfx12_wmma_output_loc_16(wave_size, M, N, row, col);
          Cbuf[col] = (const_acc != ACC_FROM_VGPR)
                          ? std::bit_cast<float>(const_acc)
                          : read_acc(cu, acc_base + out.reg, out.lane, out.sub_element);
        }
        for (uint32_t ck = 0; ck < compressed_k; ++ck) {
          Abuf[ck] = ea(cu, s0, swmmac_a_input_loc(wave_size, M, K, row, ck, in_bits));
          const uint32_t dense_k = dense_k_for(row, ck);
          for (uint32_t col = 0; col < N; ++col)
            Bbuf[ck * stride + col] =
                eb(cu, s1, swmmac_b_input_loc(wave_size, N, K, col, dense_k, in_bits));
        }
        wmma_simd_matmul<float>(1, N, compressed_k, W, stride, Abuf, Bbuf, Cbuf);
        for (uint32_t col = 0; col < N; ++col) {
          auto out = gfx12_wmma_output_loc_16(wave_size, M, N, row, col);
          results.push_back({out.reg, out.lane, out.sub_element, pack_result(Cbuf[col])});
        }
      }
    }
  } else {
    run_scalar();
  }

  uint32_t dst_regs = ((M * N) / wave_size + 1) / 2;
  std::vector<uint32_t> words(dst_regs * wave_size, 0);
  std::vector<uint8_t> masks(dst_regs * wave_size, 0);
  for (const auto &r : results) {
    uint32_t idx = r.reg * wave_size + r.lane;
    uint32_t shift = r.sub_element * 16;
    words[idx] = (words[idx] & ~(0xFFFFu << shift)) | (static_cast<uint32_t>(r.val) << shift);
    masks[idx] |= 1u << r.sub_element;
  }
  for (uint32_t reg = 0; reg < dst_regs; ++reg) {
    for (uint32_t lane = 0; lane < wave_size; ++lane) {
      uint32_t idx = reg * wave_size + lane;
      uint32_t word = words[idx];
      if (masks[idx] != 0x3u) {
        uint32_t old = cu.read_vgpr(dst + reg, lane);
        if ((masks[idx] & 0x1u) == 0)
          word = (word & 0xFFFF0000u) | (old & 0x0000FFFFu);
        if ((masks[idx] & 0x2u) == 0)
          word = (word & 0x0000FFFFu) | (old & 0xFFFF0000u);
      }
      cu.write_vgpr(dst + reg, lane, word);
    }
  }
}

template <typename ExtractA, typename ExtractB>
void exec_wmma_f16(amdgpu::ComputeUnitCore &cu, uint32_t M, uint32_t N, uint32_t K,
                   uint32_t in_bits, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                   ExtractA ea, ExtractB eb, uint32_t const_acc = ACC_FROM_VGPR,
                   uint32_t wave_size = WMMA_WAVE32) {
  exec_wmma_packed16(
      cu, M, N, K, in_bits, dst, s0, s1, s2, ea, eb,
      [](amdgpu::ComputeUnitCore &cu, uint32_t reg, uint32_t lane, uint32_t sub) {
        uint32_t raw = cu.read_vgpr(reg, lane);
        return util::f16_to_f32(static_cast<uint16_t>((raw >> (sub * 16)) & 0xFFFF));
      },
      [](float val) { return util::f32_to_f16(val); }, const_acc, wave_size);
}

/// Fast path for the f16-output WMMA shapes (v_wmma_f16_*_f16). Like the f32-out
/// f16 specialization for the input convert + matmul, but the result is packed
/// to 16 bits (2 per dst word) and written through the WMMA 16-bit output map.
/// Falls back to generic exec_wmma_f16 without AVX-512 / under force-scalar.
template <uint32_t M, uint32_t N, uint32_t K>
void exec_wmma_f16_spec(amdgpu::ComputeUnitCore &cu, uint32_t dst, uint32_t s0, uint32_t s1,
                        uint32_t s2, uint32_t const_acc = ACC_FROM_VGPR) {
  constexpr uint32_t in_bits = 16;
  static_assert(N % 16 == 0, "specialized f16 WMMA assumes N is a multiple of the zmm width");
  auto fallback = [&]() {
    exec_wmma_f16(cu, M, N, K, in_bits, dst, s0, s1, s2, amdgpu::extract_f16, amdgpu::extract_f16,
                  const_acc);
  };
  if constexpr (!util::has_stdx_simd) {
    fallback();
    return;
  } else {
    if (util::force_scalar() || util::native<float>::size() != 16) {
      fallback();
      return;
    }
    require_wmma_wave32(cu);
    constexpr uint32_t W = 16;
    const uint32_t wf = cu.wf_size();
    const uint32_t *a_words = reinterpret_cast<const uint32_t *>(cu.vgpr_data(s0));
    const uint32_t *b_words = reinterpret_cast<const uint32_t *>(cu.vgpr_data(s1));
    const uint32_t *c_words = reinterpret_cast<const uint32_t *>(cu.vgpr_data(s2));
    uint32_t *d_words = reinterpret_cast<uint32_t *>(cu.vgpr_data(dst));
    alignas(64) float A_buf[M * K];
    alignas(64) float B_buf[K * N];
    alignas(64) float C_buf[M * N];
    alignas(64) float A_f32[M * K];
    alignas(64) float B_f32[N * K];
    util::f16_to_f32_block(reinterpret_cast<const uint16_t *>(a_words), A_f32, M * K);
    util::f16_to_f32_block(reinterpret_cast<const uint16_t *>(b_words), B_f32, N * K);
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = wmma_output_loc_16(M, N, row, col);
        uint32_t raw = c_words[out.reg * wf + out.lane];
        C_buf[row * N + col] =
            (const_acc != ACC_FROM_VGPR)
                ? std::bit_cast<float>(const_acc)
                : util::f16_to_f32(static_cast<uint16_t>((raw >> (out.sub_element * 16)) & 0xFFFF));
      }
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t k = 0; k < K; ++k) {
        auto al = wmma_input_loc(M, K, row, k, in_bits);
        A_buf[row * K + k] = A_f32[(al.vgpr_offset * wf + al.lane) * 2 + al.sub_element];
      }
    for (uint32_t k = 0; k < K; ++k)
      for (uint32_t col = 0; col < N; ++col) {
        auto bl = wmma_input_loc(N, K, col, k, in_bits);
        B_buf[k * N + col] = B_f32[(bl.vgpr_offset * wf + bl.lane) * 2 + bl.sub_element];
      }
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t c0 = 0; c0 < N; c0 += W) {
        util::native<float> c_row;
        c_row.copy_from(&C_buf[row * N + c0], util::stdx::vector_aligned);
        for (uint32_t k = 0; k < K; ++k) {
          util::native<float> a_bcast(A_buf[row * K + k]);
          util::native<float> b_row;
          b_row.copy_from(&B_buf[k * N + c0], util::stdx::vector_aligned);
          c_row = util::stdx::fma(a_bcast, b_row, c_row);
        }
        c_row.copy_to(&C_buf[row * N + c0], util::stdx::vector_aligned);
      }
    // Pack f32 results to f16, two per dst word (the WMMA 16-bit output map).
    constexpr uint32_t DST_REGS = ((M * N) / WMMA_WAVE32 + 1) / 2;
    alignas(64) uint32_t words[DST_REGS * WMMA_WAVE32] = {};
    alignas(64) uint8_t masks[DST_REGS * WMMA_WAVE32] = {};
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = wmma_output_loc_16(M, N, row, col);
        uint32_t idx = out.reg * WMMA_WAVE32 + out.lane;
        uint32_t shift = out.sub_element * 16;
        uint16_t v = util::f32_to_f16(C_buf[row * N + col]);
        words[idx] = (words[idx] & ~(0xFFFFu << shift)) | (static_cast<uint32_t>(v) << shift);
        masks[idx] |= 1u << out.sub_element;
      }
    for (uint32_t reg = 0; reg < DST_REGS; ++reg)
      for (uint32_t lane = 0; lane < WMMA_WAVE32; ++lane) {
        uint32_t idx = reg * WMMA_WAVE32 + lane;
        uint32_t word = words[idx];
        if (masks[idx] != 0x3u) {
          uint32_t old = d_words[reg * wf + lane];
          if ((masks[idx] & 0x1u) == 0)
            word = (word & 0xFFFF0000u) | (old & 0x0000FFFFu);
          if ((masks[idx] & 0x2u) == 0)
            word = (word & 0x0000FFFFu) | (old & 0xFFFF0000u);
        }
        d_words[reg * wf + lane] = word;
      }
  }
}

template <typename ExtractA, typename ExtractB>
void exec_swmmac_f16(amdgpu::ComputeUnitCore &cu, uint32_t M, uint32_t N, uint32_t K,
                     uint32_t in_bits, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t acc_base,
                     uint32_t index_base, uint32_t index_entries, uint32_t index_key, ExtractA ea,
                     ExtractB eb, uint32_t const_acc = ACC_FROM_VGPR,
                     uint32_t wave_size = WMMA_WAVE32) {
  exec_swmmac_packed16(
      cu, M, N, K, in_bits, dst, s0, s1, acc_base, index_base, index_entries, index_key, ea, eb,
      [](amdgpu::ComputeUnitCore &cu, uint32_t reg, uint32_t lane, uint32_t sub) {
        uint32_t raw = cu.read_vgpr(reg, lane);
        return util::f16_to_f32(static_cast<uint16_t>((raw >> (sub * 16)) & 0xFFFF));
      },
      [](float val) { return util::f32_to_f16(val); }, const_acc, wave_size);
}

template <typename ExtractA, typename ExtractB>
void exec_wmma_bf16(amdgpu::ComputeUnitCore &cu, uint32_t M, uint32_t N, uint32_t K,
                    uint32_t in_bits, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                    ExtractA ea, ExtractB eb, uint32_t const_acc = ACC_FROM_VGPR,
                    uint32_t wave_size = WMMA_WAVE32) {
  exec_wmma_packed16(
      cu, M, N, K, in_bits, dst, s0, s1, s2, ea, eb,
      [](amdgpu::ComputeUnitCore &cu, uint32_t reg, uint32_t lane, uint32_t sub) {
        uint32_t raw = cu.read_vgpr(reg, lane);
        return util::bf16_to_f32(static_cast<uint16_t>((raw >> (sub * 16)) & 0xFFFF));
      },
      [](float val) { return util::f32_to_bf16(val); }, const_acc, wave_size);
}

/// Fast path for the bf16-output WMMA shapes (v_wmma_bf16_*_bf16). Like the
/// f16-out specialization but with the bf16 bulk input convert (zero-extend)
/// and bf16 truncation on output. Falls back to generic exec_wmma_bf16 without
/// AVX-512 / under force-scalar.
template <uint32_t M, uint32_t N, uint32_t K>
void exec_wmma_bf16_spec(amdgpu::ComputeUnitCore &cu, uint32_t dst, uint32_t s0, uint32_t s1,
                         uint32_t s2, uint32_t const_acc = ACC_FROM_VGPR) {
  constexpr uint32_t in_bits = 16;
  static_assert(N % 16 == 0, "specialized bf16 WMMA assumes N is a multiple of the zmm width");
  auto fallback = [&]() {
    exec_wmma_bf16(cu, M, N, K, in_bits, dst, s0, s1, s2, amdgpu::extract_bf16,
                   amdgpu::extract_bf16, const_acc);
  };
  if constexpr (!util::has_stdx_simd) {
    fallback();
    return;
  } else {
    if (util::force_scalar() || util::native<float>::size() != 16) {
      fallback();
      return;
    }
    require_wmma_wave32(cu);
    constexpr uint32_t W = 16;
    const uint32_t wf = cu.wf_size();
    const uint32_t *a_words = reinterpret_cast<const uint32_t *>(cu.vgpr_data(s0));
    const uint32_t *b_words = reinterpret_cast<const uint32_t *>(cu.vgpr_data(s1));
    const uint32_t *c_words = reinterpret_cast<const uint32_t *>(cu.vgpr_data(s2));
    uint32_t *d_words = reinterpret_cast<uint32_t *>(cu.vgpr_data(dst));
    alignas(64) float A_buf[M * K];
    alignas(64) float B_buf[K * N];
    alignas(64) float C_buf[M * N];
    alignas(64) float A_f32[M * K];
    alignas(64) float B_f32[N * K];
    util::bf16_to_f32_block(reinterpret_cast<const uint16_t *>(a_words), A_f32, M * K);
    util::bf16_to_f32_block(reinterpret_cast<const uint16_t *>(b_words), B_f32, N * K);
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = wmma_output_loc_16(M, N, row, col);
        uint32_t raw = c_words[out.reg * wf + out.lane];
        C_buf[row * N + col] = (const_acc != ACC_FROM_VGPR)
                                   ? std::bit_cast<float>(const_acc)
                                   : util::bf16_to_f32(static_cast<uint16_t>(
                                         (raw >> (out.sub_element * 16)) & 0xFFFF));
      }
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t k = 0; k < K; ++k) {
        auto al = wmma_input_loc(M, K, row, k, in_bits);
        A_buf[row * K + k] = A_f32[(al.vgpr_offset * wf + al.lane) * 2 + al.sub_element];
      }
    for (uint32_t k = 0; k < K; ++k)
      for (uint32_t col = 0; col < N; ++col) {
        auto bl = wmma_input_loc(N, K, col, k, in_bits);
        B_buf[k * N + col] = B_f32[(bl.vgpr_offset * wf + bl.lane) * 2 + bl.sub_element];
      }
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t c0 = 0; c0 < N; c0 += W) {
        util::native<float> c_row;
        c_row.copy_from(&C_buf[row * N + c0], util::stdx::vector_aligned);
        for (uint32_t k = 0; k < K; ++k) {
          util::native<float> a_bcast(A_buf[row * K + k]);
          util::native<float> b_row;
          b_row.copy_from(&B_buf[k * N + c0], util::stdx::vector_aligned);
          c_row = util::stdx::fma(a_bcast, b_row, c_row);
        }
        c_row.copy_to(&C_buf[row * N + c0], util::stdx::vector_aligned);
      }
    // Pack f32 results to bf16 (truncation), two per dst word.
    constexpr uint32_t DST_REGS = ((M * N) / WMMA_WAVE32 + 1) / 2;
    alignas(64) uint32_t words[DST_REGS * WMMA_WAVE32] = {};
    alignas(64) uint8_t masks[DST_REGS * WMMA_WAVE32] = {};
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = wmma_output_loc_16(M, N, row, col);
        uint32_t idx = out.reg * WMMA_WAVE32 + out.lane;
        uint32_t shift = out.sub_element * 16;
        uint16_t v = util::f32_to_bf16(C_buf[row * N + col]);
        words[idx] = (words[idx] & ~(0xFFFFu << shift)) | (static_cast<uint32_t>(v) << shift);
        masks[idx] |= 1u << out.sub_element;
      }
    for (uint32_t reg = 0; reg < DST_REGS; ++reg)
      for (uint32_t lane = 0; lane < WMMA_WAVE32; ++lane) {
        uint32_t idx = reg * WMMA_WAVE32 + lane;
        uint32_t word = words[idx];
        if (masks[idx] != 0x3u) {
          uint32_t old = d_words[reg * wf + lane];
          if ((masks[idx] & 0x1u) == 0)
            word = (word & 0xFFFF0000u) | (old & 0x0000FFFFu);
          if ((masks[idx] & 0x2u) == 0)
            word = (word & 0x0000FFFFu) | (old & 0xFFFF0000u);
        }
        d_words[reg * wf + lane] = word;
      }
  }
}

/// Fast path for the dense f16-out fp8/bf8-input WMMA shapes
/// (v_wmma_f16_16x16x{64,128}_{fp8,bf8}_{fp8,bf8}). Like the f16-out f16
/// specialization for the matmul + packed 16-bit output map, but the bulk
/// input convert is the f8 LUT gather; A/B formats are compile-time selected.
/// Falls back to the generic exec_wmma_f16 without AVX-512 / under
/// force-scalar.
template <uint32_t M, uint32_t N, uint32_t K, bool A_FP8, bool B_FP8, bool FNUZ = false>
void exec_wmma_f16_f8_spec(amdgpu::ComputeUnitCore &cu, uint32_t dst, uint32_t s0, uint32_t s1,
                           uint32_t s2, uint32_t const_acc = ACC_FROM_VGPR) {
  constexpr uint32_t in_bits = 8;
  static_assert(N % 16 == 0, "specialized f8 WMMA assumes N is a multiple of the zmm width");
  constexpr auto ea = f8_extract_fn<A_FP8, FNUZ>();
  constexpr auto eb = f8_extract_fn<B_FP8, FNUZ>();
  auto fallback = [&]() {
    exec_wmma_f16(cu, M, N, K, in_bits, dst, s0, s1, s2, ea, eb, const_acc);
  };
  if constexpr (!util::has_stdx_simd) {
    fallback();
    return;
  } else {
    if (util::force_scalar() || util::native<float>::size() != 16) {
      fallback();
      return;
    }
    require_wmma_wave32(cu);
    constexpr uint32_t W = 16;
    const uint32_t wf = cu.wf_size();
    const uint32_t *a_words = reinterpret_cast<const uint32_t *>(cu.vgpr_data(s0));
    const uint32_t *b_words = reinterpret_cast<const uint32_t *>(cu.vgpr_data(s1));
    const uint32_t *c_words = reinterpret_cast<const uint32_t *>(cu.vgpr_data(s2));
    uint32_t *d_words = reinterpret_cast<uint32_t *>(cu.vgpr_data(dst));
    alignas(64) float A_buf[M * K];
    alignas(64) float B_buf[K * N];
    alignas(64) float C_buf[M * N];
    alignas(64) float A_f32[M * K];
    alignas(64) float B_f32[N * K];
    f8_to_f32_block<A_FP8, FNUZ>(a_words, A_f32, M * K);
    f8_to_f32_block<B_FP8, FNUZ>(b_words, B_f32, N * K);
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = wmma_output_loc_16(M, N, row, col);
        uint32_t raw = c_words[out.reg * wf + out.lane];
        C_buf[row * N + col] =
            (const_acc != ACC_FROM_VGPR)
                ? std::bit_cast<float>(const_acc)
                : util::f16_to_f32(static_cast<uint16_t>((raw >> (out.sub_element * 16)) & 0xFFFF));
      }
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t k = 0; k < K; ++k) {
        auto al = wmma_a_input_loc(M, K, row, k, in_bits, in_bits);
        A_buf[row * K + k] = A_f32[(al.vgpr_offset * wf + al.lane) * 4 + al.sub_element];
      }
    for (uint32_t k = 0; k < K; ++k)
      for (uint32_t col = 0; col < N; ++col) {
        auto bl = wmma_b_input_loc(N, K, col, k, in_bits, in_bits);
        B_buf[k * N + col] = B_f32[(bl.vgpr_offset * wf + bl.lane) * 4 + bl.sub_element];
      }
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t c0 = 0; c0 < N; c0 += W) {
        util::native<float> c_row;
        c_row.copy_from(&C_buf[row * N + c0], util::stdx::vector_aligned);
        for (uint32_t k = 0; k < K; ++k) {
          util::native<float> a_bcast(A_buf[row * K + k]);
          util::native<float> b_row;
          b_row.copy_from(&B_buf[k * N + c0], util::stdx::vector_aligned);
          c_row = util::stdx::fma(a_bcast, b_row, c_row);
        }
        c_row.copy_to(&C_buf[row * N + c0], util::stdx::vector_aligned);
      }
    // Pack f32 results to f16, two per dst word (the WMMA 16-bit output map).
    constexpr uint32_t DST_REGS = ((M * N) / WMMA_WAVE32 + 1) / 2;
    alignas(64) uint32_t words[DST_REGS * WMMA_WAVE32] = {};
    alignas(64) uint8_t masks[DST_REGS * WMMA_WAVE32] = {};
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = wmma_output_loc_16(M, N, row, col);
        uint32_t idx = out.reg * WMMA_WAVE32 + out.lane;
        uint32_t shift = out.sub_element * 16;
        uint16_t v = util::f32_to_f16(C_buf[row * N + col]);
        words[idx] = (words[idx] & ~(0xFFFFu << shift)) | (static_cast<uint32_t>(v) << shift);
        masks[idx] |= 1u << out.sub_element;
      }
    for (uint32_t reg = 0; reg < DST_REGS; ++reg)
      for (uint32_t lane = 0; lane < WMMA_WAVE32; ++lane) {
        uint32_t idx = reg * WMMA_WAVE32 + lane;
        uint32_t word = words[idx];
        if (masks[idx] != 0x3u) {
          uint32_t old = d_words[reg * wf + lane];
          if ((masks[idx] & 0x1u) == 0)
            word = (word & 0xFFFF0000u) | (old & 0x0000FFFFu);
          if ((masks[idx] & 0x2u) == 0)
            word = (word & 0x0000FFFFu) | (old & 0xFFFF0000u);
        }
        d_words[reg * wf + lane] = word;
      }
  }
}

template <typename ExtractA, typename ExtractB>
void exec_swmmac_bf16(amdgpu::ComputeUnitCore &cu, uint32_t M, uint32_t N, uint32_t K,
                      uint32_t in_bits, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t acc_base,
                      uint32_t index_base, uint32_t index_entries, uint32_t index_key, ExtractA ea,
                      ExtractB eb, uint32_t const_acc = ACC_FROM_VGPR,
                      uint32_t wave_size = WMMA_WAVE32) {
  exec_swmmac_packed16(
      cu, M, N, K, in_bits, dst, s0, s1, acc_base, index_base, index_entries, index_key, ea, eb,
      [](amdgpu::ComputeUnitCore &cu, uint32_t reg, uint32_t lane, uint32_t sub) {
        uint32_t raw = cu.read_vgpr(reg, lane);
        return util::bf16_to_f32(static_cast<uint16_t>((raw >> (sub * 16)) & 0xFFFF));
      },
      [](float val) { return util::f32_to_bf16(val); }, const_acc, wave_size);
}

/// Scaled MFMA execute for f32 output with FP8/FP6/FP4 input (VOP3PX2).
///
/// Applies per-32-K-element-block E8M0 exponent biases from scale VGPRs.
/// Scale format: 8-bit biased exponent (bias=127), so 2^(scale - 127).
/// Each lane's scale VGPR holds packed 8-bit scale values (one byte per block).
///
/// @param scale_a_base  VGPR base for A-matrix scale values.
/// @param scale_b_base  VGPR base for B-matrix scale values.
template <typename ExtractA, typename ExtractB>
void exec_f32_scaled(amdgpu::ComputeUnitCore &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t B,
                     uint32_t in_bits, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                     ExtractA ea, ExtractB eb, uint32_t const_acc, uint32_t cbsz, uint32_t abid,
                     uint32_t blgp, uint32_t scale_a_base, uint32_t scale_b_base) {
  constexpr uint32_t BLOCK_K = 32;
  const uint32_t wf = cu.wf_size();
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t val;
  };
  std::vector<Result> results;
  results.reserve(M * N * B);
  uint32_t num_blocks = (K + BLOCK_K - 1) / BLOCK_K;

  // Per-block E8M0 scale factor for output (row,col,b) in K-block blk.
  auto scale_exp_for = [&](uint32_t row, uint32_t col, uint32_t b, uint32_t blk) -> int {
    auto out = physicalize_out(output_loc_32(M, N, row, col, b), wf);
    uint32_t sa_raw = cu.read_vgpr(scale_a_base, out.lane);
    uint32_t sb_raw = cu.read_vgpr(scale_b_base, out.lane);
    uint8_t sa_e8m0 = static_cast<uint8_t>((sa_raw >> (blk * 8)) & 0xFF);
    uint8_t sb_e8m0 = static_cast<uint8_t>((sb_raw >> (blk * 8)) & 0xFF);
    return static_cast<int>(sa_e8m0) + static_cast<int>(sb_e8m0) - 254;
  };

  auto run_scalar = [&]() {
    for (uint32_t b = 0; b < B; ++b) {
      for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t col = 0; col < N; ++col) {
          auto out = physicalize_out(output_loc_32(M, N, row, col, b), wf);
          float acc = (const_acc != ACC_FROM_VGPR)
                          ? std::bit_cast<float>(const_acc)
                          : std::bit_cast<float>(cu.read_vgpr(s2 + out.reg, out.lane));
          for (uint32_t blk = 0; blk < num_blocks; ++blk) {
            float block_sum = 0.0f;
            uint32_t k_start = blk * BLOCK_K;
            uint32_t k_end = std::min(k_start + BLOCK_K, K);
            for (uint32_t k = k_start; k < k_end; ++k) {
              auto al = input_loc(M, K, B, row, k, b, in_bits);
              auto bl = input_loc(N, K, B, col, k, b, in_bits);
              if (cbsz != 0)
                al.lane = permute_a_lane(al.lane, cbsz, abid);
              if (blgp != 0)
                bl.lane = permute_b_lane(bl.lane, blgp);
              block_sum +=
                  ea(cu, s0, physicalize_loc(al, wf)) * eb(cu, s1, physicalize_loc(bl, wf));
            }
            acc += std::ldexp(block_sum, scale_exp_for(row, col, b, blk));
          }
          results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(acc)});
        }
      }
    }
  };

  // SIMD fast path: hoist A/B into dense f32 buffers (lane permutation folded
  // in), then for each row accumulate each K-block's partial product as
  // native-width FMA rows over the N (column) dimension. The per-output E8M0
  // scale + ldexp accumulation stays scalar (cheap: O(num_blocks) per output
  // vs O(K) MACs). A scalar tail covers trailing N columns.
  if constexpr (util::has_stdx_simd) {
    // Column-padded, stack-allocated, aligned B loads. See exec_f32_mixed.
    // Cacc is touched scalar (per-output ldexp) so it keeps an N pitch.
    constexpr uint32_t W = static_cast<uint32_t>(util::native<float>::size());
    constexpr size_t MAX_AB = 2048;
    constexpr size_t MAX_BSTRIDE = 4096;
    constexpr size_t MAX_C = 1024;
    static_assert((MAX_AB + MAX_BSTRIDE + MAX_C) * sizeof(float) <= 48 * 1024,
                  "MFMA SIMD staging buffers exceed the 48 KiB stack budget");
    const uint32_t stride = ((N + W - 1) / W) * W;
    if (util::force_scalar() || static_cast<size_t>(M) * K > MAX_AB ||
        static_cast<size_t>(K) * stride > MAX_BSTRIDE || static_cast<size_t>(M) * N > MAX_C) {
      run_scalar();
    } else {
      // Zero-initialized staging buffers (uniform convention; see
      // wmma_simd_matmul). Cacc is pre-seeded below.
      alignas(64) float Abuf[MAX_AB] = {};
      alignas(64) float Bbuf[MAX_BSTRIDE] = {};
      alignas(64) float Cacc[MAX_C] = {};
      for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t row = 0; row < M; ++row)
          for (uint32_t k = 0; k < K; ++k) {
            auto al = input_loc(M, K, B, row, k, b, in_bits);
            if (cbsz != 0)
              al.lane = permute_a_lane(al.lane, cbsz, abid);
            Abuf[row * K + k] = ea(cu, s0, physicalize_loc(al, wf));
          }
        for (uint32_t k = 0; k < K; ++k)
          for (uint32_t col = 0; col < N; ++col) {
            auto bl = input_loc(N, K, B, col, k, b, in_bits);
            if (blgp != 0)
              bl.lane = permute_b_lane(bl.lane, blgp);
            Bbuf[k * stride + col] = eb(cu, s1, physicalize_loc(bl, wf));
          }
        for (uint32_t row = 0; row < M; ++row)
          for (uint32_t col = 0; col < N; ++col) {
            auto out = physicalize_out(output_loc_32(M, N, row, col, b), wf);
            Cacc[row * N + col] = (const_acc != ACC_FROM_VGPR)
                                      ? std::bit_cast<float>(const_acc)
                                      : std::bit_cast<float>(cu.read_vgpr(s2 + out.reg, out.lane));
          }
        for (uint32_t row = 0; row < M; ++row) {
          for (uint32_t blk = 0; blk < num_blocks; ++blk) {
            uint32_t k_start = blk * BLOCK_K;
            uint32_t k_end = std::min(k_start + BLOCK_K, K);
            uint32_t col = 0;
            alignas(64) float bs[64];
            for (; col + W <= N; col += W) {
              util::native<float> acc(0.0f);
              for (uint32_t k = k_start; k < k_end; ++k) {
                util::native<float> a(Abuf[row * K + k]);
                util::native<float> bv;
                bv.copy_from(&Bbuf[k * stride + col], util::stdx::vector_aligned);
                acc = util::stdx::fma(a, bv, acc);
              }
              acc.copy_to(bs, util::stdx::vector_aligned);
              for (uint32_t j = 0; j < W; ++j)
                Cacc[row * N + col + j] += std::ldexp(bs[j], scale_exp_for(row, col + j, b, blk));
            }
            for (; col < N; ++col) {
              float block_sum = 0.0f;
              for (uint32_t k = k_start; k < k_end; ++k)
                block_sum = std::fma(Abuf[row * K + k], Bbuf[k * stride + col], block_sum);
              Cacc[row * N + col] += std::ldexp(block_sum, scale_exp_for(row, col, b, blk));
            }
          }
        }
        for (uint32_t row = 0; row < M; ++row)
          for (uint32_t col = 0; col < N; ++col) {
            auto out = physicalize_out(output_loc_32(M, N, row, col, b), wf);
            results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(Cacc[row * N + col])});
          }
      }
    }
  } else {
    run_scalar();
  }

  for (const auto &r : results)
    cu.write_vgpr(dst + r.reg, r.lane, r.val);
}

/// Scaled MFMA for mixed-format f8f6f4: A and B may have different bit widths.
/// cbsz/blgp are used as format selectors (not lane permutations).
template <typename ExtractA, typename ExtractB>
void exec_f32_scaled_mixed(amdgpu::ComputeUnitCore &cu, uint32_t M, uint32_t N, uint32_t K,
                           uint32_t B, uint32_t a_bits, uint32_t b_bits, uint32_t dst, uint32_t s0,
                           uint32_t s1, uint32_t s2, ExtractA ea, ExtractB eb, uint32_t const_acc,
                           uint32_t scale_a_base, uint32_t scale_b_base) {
  constexpr uint32_t BLOCK_K = 32;
  const uint32_t wf = cu.wf_size();
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t val;
  };
  std::vector<Result> results;
  results.reserve(M * N * B);
  uint32_t num_blocks = (K + BLOCK_K - 1) / BLOCK_K;
  for (uint32_t b = 0; b < B; ++b) {
    for (uint32_t row = 0; row < M; ++row) {
      for (uint32_t col = 0; col < N; ++col) {
        auto out = physicalize_out(output_loc_32(M, N, row, col, b), wf);
        float acc = (const_acc != ACC_FROM_VGPR)
                        ? std::bit_cast<float>(const_acc)
                        : std::bit_cast<float>(cu.read_vgpr(s2 + out.reg, out.lane));
        for (uint32_t blk = 0; blk < num_blocks; ++blk) {
          float block_sum = 0.0f;
          uint32_t k_start = blk * BLOCK_K;
          uint32_t k_end = std::min(k_start + BLOCK_K, K);
          for (uint32_t k = k_start; k < k_end; ++k) {
            auto al = input_loc(M, K, B, row, k, b, a_bits);
            auto bl = input_loc(N, K, B, col, k, b, b_bits);
            block_sum += ea(cu, s0, physicalize_loc(al, wf)) * eb(cu, s1, physicalize_loc(bl, wf));
          }
          uint32_t sa_raw = cu.read_vgpr(scale_a_base, M * blk + row);
          uint32_t sb_raw = cu.read_vgpr(scale_b_base, N * blk + col);
          uint8_t sa_e8m0 = static_cast<uint8_t>(sa_raw & 0xFFu);
          uint8_t sb_e8m0 = static_cast<uint8_t>(sb_raw & 0xFFu);
          int scale_exp = static_cast<int>(sa_e8m0) + static_cast<int>(sb_e8m0) - 254;
          acc += std::ldexp(block_sum, scale_exp);
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
                        uint32_t const_acc = ACC_FROM_VGPR, uint32_t cbsz = 0, uint32_t abid = 0,
                        uint32_t blgp = 0) {
  const uint32_t wf = cu.wf_size();
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t val;
  };
  std::vector<Result> results;
  results.reserve(M * N * B);

  auto run_scalar = [&]() {
    for (uint32_t b = 0; b < B; ++b) {
      for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t col = 0; col < N; ++col) {
          auto out = physicalize_out(output_loc_32(M, N, row, col, b), wf);
          uint32_t acc =
              (const_acc != ACC_FROM_VGPR) ? const_acc : cu.read_vgpr(s2 + out.reg, out.lane);
          for (uint32_t k = 0; k < K; ++k) {
            auto al = input_loc(M, K, B, row, k, b, 8);
            auto bl = input_loc(N, K, B, col, k, b, 8);
            if (cbsz != 0)
              al.lane = permute_a_lane(al.lane, cbsz, abid);
            if (blgp != 0)
              bl.lane = permute_b_lane(bl.lane, blgp);
            acc += static_cast<uint32_t>(extract_i8(cu, s0, physicalize_loc(al, wf)) *
                                         extract_i8(cu, s1, physicalize_loc(bl, wf)));
          }
          results.push_back({out.reg, out.lane, acc});
        }
      }
    }
  };

  // SIMD fast path mirrors exec_f32_mixed: hoist A/B/C into dense int32
  // buffers, then run the matmul as native-width int32 multiply-accumulate
  // over the N dimension. Integer MAC is exact, so the SIMD and scalar paths
  // are bit-identical. A scalar tail handles trailing N columns.
  if constexpr (util::has_stdx_simd) {
    // Column-padded, stack-allocated, aligned. See exec_f32_mixed.
    constexpr uint32_t W = static_cast<uint32_t>(util::native<int32_t>::size());
    constexpr size_t MAX_AB = 2048;
    constexpr size_t MAX_BSTRIDE = 4096;
    constexpr size_t MAX_C = 1024;
    static_assert((MAX_AB + MAX_BSTRIDE + MAX_C) * sizeof(int32_t) <= 48 * 1024,
                  "MFMA SIMD staging buffers exceed the 48 KiB stack budget");
    const uint32_t stride = ((N + W - 1) / W) * W;
    if (util::force_scalar() || static_cast<size_t>(M) * K > MAX_AB ||
        static_cast<size_t>(K) * stride > MAX_BSTRIDE || static_cast<size_t>(M) * stride > MAX_C) {
      run_scalar();
    } else {
      // Zero-initialized staging buffers (uniform convention; see
      // wmma_simd_matmul). Cbuf is pre-seeded below.
      alignas(64) int32_t Abuf[MAX_AB] = {};
      alignas(64) int32_t Bbuf[MAX_BSTRIDE] = {};
      alignas(64) int32_t Cbuf[MAX_C] = {};
      for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t row = 0; row < M; ++row)
          for (uint32_t col = 0; col < N; ++col) {
            auto out = physicalize_out(output_loc_32(M, N, row, col, b), wf);
            Cbuf[row * stride + col] =
                (const_acc != ACC_FROM_VGPR)
                    ? static_cast<int32_t>(const_acc)
                    : static_cast<int32_t>(cu.read_vgpr(s2 + out.reg, out.lane));
          }
        for (uint32_t row = 0; row < M; ++row)
          for (uint32_t k = 0; k < K; ++k) {
            auto al = input_loc(M, K, B, row, k, b, 8);
            if (cbsz != 0)
              al.lane = permute_a_lane(al.lane, cbsz, abid);
            Abuf[row * K + k] = extract_i8(cu, s0, physicalize_loc(al, wf));
          }
        for (uint32_t k = 0; k < K; ++k)
          for (uint32_t col = 0; col < N; ++col) {
            auto bl = input_loc(N, K, B, col, k, b, 8);
            if (blgp != 0)
              bl.lane = permute_b_lane(bl.lane, blgp);
            Bbuf[k * stride + col] = extract_i8(cu, s1, physicalize_loc(bl, wf));
          }
        for (uint32_t row = 0; row < M; ++row) {
          uint32_t col = 0;
          for (; col + W <= N; col += W) {
            util::native<int32_t> c;
            c.copy_from(&Cbuf[row * stride + col], util::stdx::vector_aligned);
            for (uint32_t k = 0; k < K; ++k) {
              util::native<int32_t> a(Abuf[row * K + k]);
              util::native<int32_t> bv;
              bv.copy_from(&Bbuf[k * stride + col], util::stdx::vector_aligned);
              c += a * bv;
            }
            c.copy_to(&Cbuf[row * stride + col], util::stdx::vector_aligned);
          }
          for (; col < N; ++col) {
            uint32_t acc = static_cast<uint32_t>(Cbuf[row * stride + col]);
            for (uint32_t k = 0; k < K; ++k)
              acc += static_cast<uint32_t>(Abuf[row * K + k] * Bbuf[k * stride + col]);
            Cbuf[row * stride + col] = static_cast<int32_t>(acc);
          }
        }
        for (uint32_t row = 0; row < M; ++row)
          for (uint32_t col = 0; col < N; ++col) {
            auto out = physicalize_out(output_loc_32(M, N, row, col, b), wf);
            results.push_back({out.reg, out.lane, static_cast<uint32_t>(Cbuf[row * stride + col])});
          }
      }
    }
  } else {
    run_scalar();
  }

  for (const auto &r : results)
    cu.write_vgpr(dst + r.reg, r.lane, r.val);
}

template <typename ExtractA, typename ExtractB>
void exec_wmma_i32(amdgpu::ComputeUnitCore &cu, uint32_t M, uint32_t N, uint32_t K,
                   uint32_t in_bits, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                   ExtractA ea, ExtractB eb, bool clamp, uint32_t const_acc = ACC_FROM_VGPR,
                   uint32_t wave_size = WMMA_WAVE32) {
  require_gfx12_wmma_wave_size(wave_size);
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t val;
  };
  std::vector<Result> results;
  results.reserve(M * N);

  auto run_scalar = [&]() {
    for (uint32_t row = 0; row < M; ++row) {
      for (uint32_t col = 0; col < N; ++col) {
        auto out = gfx12_wmma_output_loc_32(wave_size, M, N, row, col);
        int64_t acc =
            (const_acc != ACC_FROM_VGPR)
                ? static_cast<int64_t>(static_cast<int32_t>(const_acc))
                : static_cast<int64_t>(static_cast<int32_t>(cu.read_vgpr(s2 + out.reg, out.lane)));
        for (uint32_t k = 0; k < K; ++k) {
          auto al = gfx12_wmma_input_loc(wave_size, M, K, row, k, in_bits);
          auto bl = gfx12_wmma_input_loc(wave_size, N, K, col, k, in_bits);
          acc += static_cast<int64_t>(ea(cu, s0, al)) * static_cast<int64_t>(eb(cu, s1, bl));
        }
        results.push_back({out.reg, out.lane, pack_i32_acc(acc, clamp)});
      }
    }
  };

  // SIMD fast path: the K-sum of products is exact in int32 for WMMA's i8/i4
  // inputs at K <= 128 (max |sum| ~2M << 2^31). The int32 accumulator is added
  // in 64-bit at pack time so pack_i32_acc saturates exactly like the scalar
  // int64 reference even when C sits near INT32_MAX/INT32_MIN.
  if constexpr (util::has_stdx_simd) {
    constexpr uint32_t W = static_cast<uint32_t>(util::native<int32_t>::size());
    const uint32_t stride = ((N + W - 1) / W) * W;
    if (util::force_scalar() || static_cast<size_t>(M) * K > WMMA_SIMD_MAX_AB ||
        static_cast<size_t>(K) * stride > WMMA_SIMD_MAX_BSTRIDE ||
        static_cast<size_t>(M) * stride > WMMA_SIMD_MAX_C) {
      run_scalar();
    } else {
      alignas(64) int32_t Abuf[WMMA_SIMD_MAX_AB] = {};
      alignas(64) int32_t Bbuf[WMMA_SIMD_MAX_BSTRIDE] = {};
      alignas(64) int32_t Cbuf[WMMA_SIMD_MAX_C] = {};
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t k = 0; k < K; ++k)
          Abuf[row * K + k] = ea(cu, s0, gfx12_wmma_input_loc(wave_size, M, K, row, k, in_bits));
      for (uint32_t k = 0; k < K; ++k)
        for (uint32_t col = 0; col < N; ++col)
          Bbuf[k * stride + col] =
              eb(cu, s1, gfx12_wmma_input_loc(wave_size, N, K, col, k, in_bits));
      wmma_simd_matmul<int32_t>(M, N, K, W, stride, Abuf, Bbuf, Cbuf);
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t col = 0; col < N; ++col) {
          auto out = gfx12_wmma_output_loc_32(wave_size, M, N, row, col);
          int64_t acc = (const_acc != ACC_FROM_VGPR)
                            ? static_cast<int64_t>(static_cast<int32_t>(const_acc))
                            : static_cast<int64_t>(
                                  static_cast<int32_t>(cu.read_vgpr(s2 + out.reg, out.lane)));
          acc += static_cast<int64_t>(Cbuf[row * stride + col]);
          results.push_back({out.reg, out.lane, pack_i32_acc(acc, clamp)});
        }
    }
  } else {
    run_scalar();
  }

  for (const auto &r : results)
    cu.write_vgpr(dst + r.reg, r.lane, r.val);
}

template <typename ExtractA, typename ExtractB>
void exec_gfx11_wmma_i32(amdgpu::ComputeUnitCore &cu, uint32_t wave_size, uint32_t M, uint32_t N,
                         uint32_t K, uint32_t in_bits, uint32_t dst, uint32_t s0, uint32_t s1,
                         uint32_t s2, ExtractA ea, ExtractB eb, bool clamp,
                         uint32_t const_acc = ACC_FROM_VGPR) {
  require_gfx11_wmma_wave_size(wave_size);
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t val;
  };
  std::vector<Result> results;
  results.reserve(M * N);

  for (uint32_t row = 0; row < M; ++row) {
    for (uint32_t col = 0; col < N; ++col) {
      auto out = gfx11_wmma_output_loc_32(wave_size, M, N, row, col);
      uint32_t lane_group = out.lane / N;
      int64_t acc =
          (const_acc != ACC_FROM_VGPR)
              ? static_cast<int64_t>(static_cast<int32_t>(const_acc))
              : static_cast<int64_t>(static_cast<int32_t>(cu.read_vgpr(s2 + out.reg, out.lane)));
      for (uint32_t k = 0; k < K; ++k) {
        auto al = gfx11_wmma_input_loc(M, K, row, k, in_bits, lane_group);
        auto bl = gfx11_wmma_input_loc(N, K, col, k, in_bits, lane_group);
        acc += static_cast<int64_t>(ea(cu, s0, al)) * static_cast<int64_t>(eb(cu, s1, bl));
      }
      results.push_back({out.reg, out.lane, pack_i32_acc(acc, clamp)});
    }
  }

  for (const auto &r : results)
    cu.write_vgpr(dst + r.reg, r.lane, r.val);
}

inline void exec_wmma_i32_i8(amdgpu::ComputeUnitCore &cu, uint32_t M, uint32_t N, uint32_t K,
                             uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                             uint32_t const_acc = ACC_FROM_VGPR) {
  exec_wmma_i32(cu, M, N, K, 8, dst, s0, s1, s2, extract_i8, extract_i8, false, const_acc);
}

/// Fast path for v_wmma_i32_16x16x64_iu8 (gfx1250, wave32, dense). Same recipe
/// as the f16/bf16 specializations: bulk sign-/zero-extend the packed i8
/// inputs (per the A/B sign selects from the neg bits) to int32 once, then run
/// a constexpr-unrolled int32 matmul with direct vgpr_data access. The K-sum
/// of products is exact in int32 (|sum| <= 64 * 16384) and is added to the
/// int32 accumulator in 64-bit at pack time, so clamp saturation matches the
/// scalar int64 reference bit for bit; with clamp off both wrap mod 2^32.
/// Falls back to the generic exec_wmma_i32 without AVX-512 / under
/// force-scalar.
inline void exec_wmma_i32_16x16x64_iu8(amdgpu::ComputeUnitCore &cu, uint32_t dst, uint32_t s0,
                                       uint32_t s1, uint32_t s2, bool a_signed, bool b_signed,
                                       bool clamp, uint32_t const_acc = ACC_FROM_VGPR) {
  constexpr uint32_t M = 16, N = 16, K = 64, in_bits = 8;
  auto fallback = [&]() {
    exec_wmma_i32(cu, M, N, K, in_bits, dst, s0, s1, s2, a_signed ? extract_i8 : extract_u8,
                  b_signed ? extract_i8 : extract_u8, clamp, const_acc);
  };
  if constexpr (!util::has_stdx_simd) {
    fallback();
    return;
  } else {
    if (util::force_scalar() || util::native<float>::size() != 16) {
      fallback();
      return;
    }
    require_wmma_wave32(cu);
    const uint32_t wf = cu.wf_size();
    const uint32_t *a_words = reinterpret_cast<const uint32_t *>(cu.vgpr_data(s0));
    const uint32_t *b_words = reinterpret_cast<const uint32_t *>(cu.vgpr_data(s1));
    const uint32_t *c_words = reinterpret_cast<const uint32_t *>(cu.vgpr_data(s2));
    // Accumulate in unsigned 32-bit (wrap is well-defined; identical mod 2^32
    // to the intended signed wrap), sign-restore via int32 cast at pack time.
    alignas(64) uint32_t A_buf[M * K]; // A[row][k] (sign-/zero-extended bits)
    alignas(64) uint32_t B_buf[K * N]; // B[k][col] (sign-/zero-extended bits)
    alignas(64) uint32_t S_buf[M * N]; // sum-of-products, accumulator added at pack
    // Bulk-extend the packed byte regions to int32 once, then the hoist is a
    // pure i32 index-shuffle (byte j of word w lane l sub s -> (w*wf+l)*4+s).
    alignas(64) int32_t A_i32[M * K];
    alignas(64) int32_t B_i32[N * K];
    if (a_signed)
      util::i8_to_i32_block(reinterpret_cast<const int8_t *>(a_words), A_i32, M * K);
    else
      util::u8_to_i32_block(reinterpret_cast<const uint8_t *>(a_words), A_i32, M * K);
    if (b_signed)
      util::i8_to_i32_block(reinterpret_cast<const int8_t *>(b_words), B_i32, N * K);
    else
      util::u8_to_i32_block(reinterpret_cast<const uint8_t *>(b_words), B_i32, N * K);
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t k = 0; k < K; ++k) {
        auto al = wmma_input_loc(M, K, row, k, in_bits);
        A_buf[row * K + k] = A_i32[(al.vgpr_offset * wf + al.lane) * 4 + al.sub_element];
      }
    for (uint32_t k = 0; k < K; ++k)
      for (uint32_t col = 0; col < N; ++col) {
        auto bl = wmma_input_loc(N, K, col, k, in_bits);
        B_buf[k * N + col] = B_i32[(bl.vgpr_offset * wf + bl.lane) * 4 + bl.sub_element];
      }
    // Dense 16x64 * 64x16 -> 16x16 product-sum, 16-lane stdx u32 MAC per row
    // (unsigned to avoid signed-overflow UB; bits match the signed math).
    for (uint32_t row = 0; row < M; ++row) {
      util::native<uint32_t> s_row(0);
      for (uint32_t k = 0; k < K; ++k) {
        util::native<uint32_t> a_bcast(A_buf[row * K + k]);
        util::native<uint32_t> b_row;
        b_row.copy_from(&B_buf[k * N], util::stdx::vector_aligned);
        s_row += a_bcast * b_row;
      }
      s_row.copy_to(&S_buf[row * N], util::stdx::vector_aligned);
    }
    // Add the accumulator in 64-bit and pack (saturating when clamp is set),
    // scattering directly back to VGPRs.
    uint32_t *d_words = reinterpret_cast<uint32_t *>(cu.vgpr_data(dst));
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = wmma_output_loc_32(M, N, row, col);
        int64_t acc =
            (const_acc != ACC_FROM_VGPR)
                ? static_cast<int64_t>(static_cast<int32_t>(const_acc))
                : static_cast<int64_t>(static_cast<int32_t>(c_words[out.reg * wf + out.lane]));
        acc += static_cast<int64_t>(static_cast<int32_t>(S_buf[row * N + col]));
        d_words[out.reg * wf + out.lane] = pack_i32_acc(acc, clamp);
      }
  }
}

template <typename ExtractA, typename ExtractB>
void exec_swmmac_i32(amdgpu::ComputeUnitCore &cu, uint32_t M, uint32_t N, uint32_t K,
                     uint32_t in_bits, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t acc_base,
                     uint32_t index_base, uint32_t index_entries, uint32_t index_key, ExtractA ea,
                     ExtractB eb, bool clamp, uint32_t const_acc = ACC_FROM_VGPR,
                     uint32_t wave_size = WMMA_WAVE32) {
  require_gfx12_wmma_wave_size(wave_size);
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t val;
  };
  std::vector<Result> results;
  results.reserve(M * N);

  const uint32_t compressed_k = K / 2;

  auto dense_k_for = [&](uint32_t row, uint32_t ck) -> uint32_t {
    const auto index_loc = swmmac_index_loc(wave_size, M, K, in_bits, row, ck, index_entries);
    const uint64_t index_set =
        read_swmmac_index_set(cu, index_base, index_loc.lane, index_entries, index_key);
    return swmmac_dense_k(index_set, ck, index_loc.local_compressed_k);
  };

  auto run_scalar = [&]() {
    for (uint32_t row = 0; row < M; ++row) {
      for (uint32_t col = 0; col < N; ++col) {
        auto out = gfx12_wmma_output_loc_32(wave_size, M, N, row, col);
        int64_t acc = (const_acc != ACC_FROM_VGPR)
                          ? static_cast<int64_t>(static_cast<int32_t>(const_acc))
                          : static_cast<int64_t>(
                                static_cast<int32_t>(cu.read_vgpr(acc_base + out.reg, out.lane)));
        for (uint32_t ck = 0; ck < compressed_k; ++ck) {
          auto al = swmmac_a_input_loc(wave_size, M, K, row, ck, in_bits);
          auto bl = swmmac_b_input_loc(wave_size, N, K, col, dense_k_for(row, ck), in_bits);
          acc += static_cast<int64_t>(ea(cu, s0, al)) * static_cast<int64_t>(eb(cu, s1, bl));
        }
        results.push_back({out.reg, out.lane, pack_i32_acc(acc, clamp)});
      }
    }
  };

  // SIMD fast path: per-row gather, single row through the int32 matmul core
  // (M=1). The K-sum of products is exact in int32 for i8/i4 inputs (no
  // overflow); the int32 accumulator is added in 64-bit at pack time so
  // pack_i32_acc saturates exactly like the scalar int64 reference even when
  // C sits near INT32_MAX/INT32_MIN.
  if constexpr (util::has_stdx_simd) {
    constexpr uint32_t W = static_cast<uint32_t>(util::native<int32_t>::size());
    const uint32_t stride = ((N + W - 1) / W) * W;
    if (util::force_scalar() || static_cast<size_t>(compressed_k) > WMMA_SIMD_MAX_AB ||
        static_cast<size_t>(compressed_k) * stride > WMMA_SIMD_MAX_BSTRIDE ||
        stride > WMMA_SIMD_MAX_C) {
      run_scalar();
    } else {
      alignas(64) int32_t Abuf[WMMA_SIMD_MAX_AB] = {};
      alignas(64) int32_t Bbuf[WMMA_SIMD_MAX_BSTRIDE] = {};
      alignas(64) int32_t Cbuf[WMMA_SIMD_MAX_C] = {};
      for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t col = 0; col < N; ++col)
          Cbuf[col] = 0;
        for (uint32_t ck = 0; ck < compressed_k; ++ck) {
          Abuf[ck] = ea(cu, s0, swmmac_a_input_loc(wave_size, M, K, row, ck, in_bits));
          const uint32_t dense_k = dense_k_for(row, ck);
          for (uint32_t col = 0; col < N; ++col)
            Bbuf[ck * stride + col] =
                eb(cu, s1, swmmac_b_input_loc(wave_size, N, K, col, dense_k, in_bits));
        }
        wmma_simd_matmul<int32_t>(1, N, compressed_k, W, stride, Abuf, Bbuf, Cbuf);
        for (uint32_t col = 0; col < N; ++col) {
          auto out = gfx12_wmma_output_loc_32(wave_size, M, N, row, col);
          int64_t acc = (const_acc != ACC_FROM_VGPR)
                            ? static_cast<int64_t>(static_cast<int32_t>(const_acc))
                            : static_cast<int64_t>(
                                  static_cast<int32_t>(cu.read_vgpr(acc_base + out.reg, out.lane)));
          acc += static_cast<int64_t>(Cbuf[col]);
          results.push_back({out.reg, out.lane, pack_i32_acc(acc, clamp)});
        }
      }
    }
  } else {
    run_scalar();
  }

  for (const auto &r : results)
    cu.write_vgpr(dst + r.reg, r.lane, r.val);
}

inline void exec_swmmac_i32_i8(amdgpu::ComputeUnitCore &cu, uint32_t M, uint32_t N, uint32_t K,
                               uint32_t dst, uint32_t s0, uint32_t s1, uint32_t acc_base,
                               uint32_t index_base, uint32_t index_entries, uint32_t index_key,
                               uint32_t const_acc = ACC_FROM_VGPR) {
  exec_swmmac_i32(cu, M, N, K, 8, dst, s0, s1, acc_base, index_base, index_entries, index_key,
                  extract_i8, extract_i8, false, const_acc);
}

/// MFMA execute for f64 output with f64 input: D = C + A x B.
inline void exec_f64(amdgpu::ComputeUnitCore &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t B,
                     uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                     uint32_t const_acc = ACC_FROM_VGPR, uint32_t neg = 0) {
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t lo;
    uint32_t hi;
  };
  std::vector<Result> results;
  results.reserve(M * N * B);
  auto apply_neg = [neg](double value, uint32_t bit) {
    return (neg & bit) ? std::bit_cast<double>(std::bit_cast<uint64_t>(value) ^ (uint64_t{1} << 63))
                       : value;
  };

  auto run_scalar = [&]() {
    for (uint32_t b = 0; b < B; ++b) {
      for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t col = 0; col < N; ++col) {
          // AMD convention: i=row (register dimension), j=col (lane dimension).
          auto out = output_loc_64(M, N, row, col, b);
          double acc;
          if (const_acc != ACC_FROM_VGPR) {
            acc = static_cast<double>(std::bit_cast<float>(const_acc));
          } else {
            uint32_t lo = cu.read_vgpr(s2 + out.reg, out.lane);
            uint32_t hi = cu.read_vgpr(s2 + out.reg + 1, out.lane);
            acc = std::bit_cast<double>(static_cast<uint64_t>(hi) << 32 | lo);
          }
          acc = apply_neg(acc, 0x4u);
          for (uint32_t k = 0; k < K; ++k) {
            auto al = input_loc(M, K, B, row, k, b, 64);
            auto bl = input_loc(N, K, B, col, k, b, 64);
            acc +=
                apply_neg(extract_f64(cu, s0, al), 0x1u) * apply_neg(extract_f64(cu, s1, bl), 0x2u);
          }
          uint64_t bits = std::bit_cast<uint64_t>(acc);
          results.push_back(
              {out.reg, out.lane, static_cast<uint32_t>(bits), static_cast<uint32_t>(bits >> 32)});
        }
      }
    }
  };

  // SIMD fast path mirrors exec_f32_mixed with native<double> lanes (8-wide on
  // AVX-512) and fused FMA, matching the GFX9 f64 MFMA single-rounding MACs.
  // A scalar (fused) tail covers the trailing N columns.
  if constexpr (util::has_stdx_simd) {
    // Column-padded, stack-allocated, aligned. See exec_f32_mixed. MAX_BSTRIDE
    // is half the f32 cap because native<double> packs half as many lanes.
    constexpr uint32_t W = static_cast<uint32_t>(util::native<double>::size());
    constexpr size_t MAX_AB = 2048;
    constexpr size_t MAX_BSTRIDE = 2048;
    constexpr size_t MAX_C = 1024;
    static_assert((MAX_AB + MAX_BSTRIDE + MAX_C) * sizeof(double) <= 48 * 1024,
                  "MFMA SIMD staging buffers exceed the 48 KiB stack budget");
    const uint32_t stride = ((N + W - 1) / W) * W;
    if (util::force_scalar() || static_cast<size_t>(M) * K > MAX_AB ||
        static_cast<size_t>(K) * stride > MAX_BSTRIDE || static_cast<size_t>(M) * stride > MAX_C) {
      run_scalar();
    } else {
      // Zero-initialized staging buffers (uniform convention; see
      // wmma_simd_matmul). Cbuf is pre-seeded below.
      alignas(64) double Abuf[MAX_AB] = {};
      alignas(64) double Bbuf[MAX_BSTRIDE] = {};
      alignas(64) double Cbuf[MAX_C] = {};
      for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t row = 0; row < M; ++row)
          for (uint32_t col = 0; col < N; ++col) {
            auto out = output_loc_64(M, N, row, col, b);
            if (const_acc != ACC_FROM_VGPR) {
              Cbuf[row * stride + col] = static_cast<double>(std::bit_cast<float>(const_acc));
            } else {
              uint32_t lo = cu.read_vgpr(s2 + out.reg, out.lane);
              uint32_t hi = cu.read_vgpr(s2 + out.reg + 1, out.lane);
              Cbuf[row * stride + col] =
                  std::bit_cast<double>(static_cast<uint64_t>(hi) << 32 | lo);
            }
            Cbuf[row * stride + col] = apply_neg(Cbuf[row * stride + col], 0x4u);
          }
        for (uint32_t row = 0; row < M; ++row)
          for (uint32_t k = 0; k < K; ++k) {
            auto al = input_loc(M, K, B, row, k, b, 64);
            Abuf[row * K + k] = apply_neg(extract_f64(cu, s0, al), 0x1u);
          }
        for (uint32_t k = 0; k < K; ++k)
          for (uint32_t col = 0; col < N; ++col) {
            auto bl = input_loc(N, K, B, col, k, b, 64);
            Bbuf[k * stride + col] = apply_neg(extract_f64(cu, s1, bl), 0x2u);
          }
        for (uint32_t row = 0; row < M; ++row) {
          uint32_t col = 0;
          for (; col + W <= N; col += W) {
            util::native<double> c;
            c.copy_from(&Cbuf[row * stride + col], util::stdx::vector_aligned);
            for (uint32_t k = 0; k < K; ++k) {
              util::native<double> a(Abuf[row * K + k]);
              util::native<double> bv;
              bv.copy_from(&Bbuf[k * stride + col], util::stdx::vector_aligned);
              c = util::stdx::fma(a, bv, c);
            }
            c.copy_to(&Cbuf[row * stride + col], util::stdx::vector_aligned);
          }
          for (; col < N; ++col) {
            double acc = Cbuf[row * stride + col];
            for (uint32_t k = 0; k < K; ++k)
              acc = std::fma(Abuf[row * K + k], Bbuf[k * stride + col], acc);
            Cbuf[row * stride + col] = acc;
          }
        }
        for (uint32_t row = 0; row < M; ++row)
          for (uint32_t col = 0; col < N; ++col) {
            auto out = output_loc_64(M, N, row, col, b);
            uint64_t bits = std::bit_cast<uint64_t>(Cbuf[row * stride + col]);
            results.push_back({out.reg, out.lane, static_cast<uint32_t>(bits),
                               static_cast<uint32_t>(bits >> 32)});
          }
      }
    }
  } else {
    run_scalar();
  }

  for (const auto &r : results) {
    cu.write_vgpr(dst + r.reg, r.lane, r.lo);
    cu.write_vgpr(dst + r.reg + 1, r.lane, r.hi);
  }
}

// ---------------------------------------------------------------------------
// SMFMAC (Sparse Matrix FMA) helpers and execution functions.
//
// Structured 2:4 sparsity: A is half-density (2 of every 4 K positions are
// nonzero). A per-lane index register selects which 2-of-4 positions are live.
// Each 4-bit nibble in the index encodes two 2-bit position selectors (p0, p1).
// ---------------------------------------------------------------------------

inline float smfmac_read_fp8_ocp(ComputeUnitCore &cu, uint32_t base, uint32_t byte_idx,
                                 uint32_t lane) {
  uint32_t raw = cu.read_vgpr(base + byte_idx / 4, lane);
  return util::fp8_e4m3_ocp_to_f32(static_cast<uint8_t>((raw >> ((byte_idx % 4) * 8)) & 0xFF));
}

inline float smfmac_read_bf8_ocp(ComputeUnitCore &cu, uint32_t base, uint32_t byte_idx,
                                 uint32_t lane) {
  uint32_t raw = cu.read_vgpr(base + byte_idx / 4, lane);
  return util::bf8_e5m2_ocp_to_f32(static_cast<uint8_t>((raw >> ((byte_idx % 4) * 8)) & 0xFF));
}

inline float smfmac_read_fp8_fnuz(ComputeUnitCore &cu, uint32_t base, uint32_t byte_idx,
                                  uint32_t lane) {
  uint32_t raw = cu.read_vgpr(base + byte_idx / 4, lane);
  return util::fp8_e4m3_fnuz_to_f32(static_cast<uint8_t>((raw >> ((byte_idx % 4) * 8)) & 0xFF));
}

inline float smfmac_read_bf8_fnuz(ComputeUnitCore &cu, uint32_t base, uint32_t byte_idx,
                                  uint32_t lane) {
  uint32_t raw = cu.read_vgpr(base + byte_idx / 4, lane);
  return util::bf8_e5m2_fnuz_to_f32(static_cast<uint8_t>((raw >> ((byte_idx % 4) * 8)) & 0xFF));
}

inline float smfmac_read_fp8(ComputeUnitCore &cu, uint32_t base, uint32_t byte_idx, uint32_t lane) {
  return smfmac_read_fp8_ocp(cu, base, byte_idx, lane);
}

inline float smfmac_read_bf8(ComputeUnitCore &cu, uint32_t base, uint32_t byte_idx, uint32_t lane) {
  return smfmac_read_bf8_ocp(cu, base, byte_idx, lane);
}

inline float smfmac_read_f16(ComputeUnitCore &cu, uint32_t base, uint32_t elem, uint32_t lane) {
  uint32_t raw = cu.read_vgpr(base + elem / 2, lane);
  return util::f16_to_f32(static_cast<uint16_t>((raw >> ((elem % 2) * 16)) & 0xFFFF));
}

inline float smfmac_read_bf16(ComputeUnitCore &cu, uint32_t base, uint32_t elem, uint32_t lane) {
  uint32_t raw = cu.read_vgpr(base + elem / 2, lane);
  return util::bf16_to_f32(static_cast<uint16_t>((raw >> ((elem % 2) * 16)) & 0xFFFF));
}

/// SMFMAC 16x16x32 f16/bf16 (CDNA3 mai-insts). K=32, 8 sparse groups.
/// A = v2 (4 halves/lane), B = v4 (8 halves/lane), D = v4 f32.
template <typename Extract>
void exec_smfmac_f32_16x16x32_f16(ComputeUnitCore &cu, uint32_t dst, uint32_t s0, uint32_t s1,
                                  uint32_t idx_base, Extract ex) {
  struct Result {
    uint32_t reg, lane, val;
  };
  std::vector<Result> results;
  results.reserve(16 * 16);
  for (uint32_t row = 0; row < 16; ++row) {
    for (uint32_t col = 0; col < 16; ++col) {
      auto out = output_loc_32(16, 16, row, col, 0);
      float acc = std::bit_cast<float>(cu.read_vgpr(dst + out.reg, out.lane));
      float Bcol[32];
      for (int g = 0; g < 4; ++g)
        for (int e = 0; e < 8; ++e)
          Bcol[8 * g + e] = ex(cu, s1, e, g * 16 + col);
      for (int q = 0; q < 8; ++q) {
        int laneA = (q / 2) * 16 + row;
        uint32_t idxval = cu.read_vgpr(idx_base, laneA);
        int field = (idxval >> (4 * (q % 2))) & 0xF;
        int p0 = field & 3, p1 = (field >> 2) & 3;
        for (int s = 0; s < 2; ++s) {
          float av = ex(cu, s0, (2 * q + s) % 4, laneA);
          acc += av * Bcol[4 * q + (s == 0 ? p0 : p1)];
        }
      }
      results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(acc)});
    }
  }
  for (const auto &r : results)
    cu.write_vgpr(dst + r.reg, r.lane, r.val);
}

/// SMFMAC 32x32x16 f16/bf16 (CDNA3 mai-insts). K=16, 4 sparse groups.
/// A = v2 (4 halves/lane), B = v4 (8 halves/lane), D = v16 f32.
template <typename Extract>
void exec_smfmac_f32_32x32x16_f16(ComputeUnitCore &cu, uint32_t dst, uint32_t s0, uint32_t s1,
                                  uint32_t idx_base, Extract ex) {
  struct Result {
    uint32_t reg, lane, val;
  };
  std::vector<Result> results;
  results.reserve(32 * 32);
  for (uint32_t row = 0; row < 32; ++row) {
    for (uint32_t col = 0; col < 32; ++col) {
      auto out = output_loc_32(32, 32, row, col, 0);
      float acc = std::bit_cast<float>(cu.read_vgpr(dst + out.reg, out.lane));
      uint32_t jlow = col % 16, jhi = col / 16;
      float Bcol[16];
      for (int kgrp = 0; kgrp < 2; ++kgrp) {
        uint32_t b_lane = 16 * (jhi + 2 * kgrp) + jlow;
        for (int e = 0; e < 8; ++e)
          Bcol[8 * kgrp + e] = ex(cu, s1, e, b_lane);
      }
      for (int q = 0; q < 4; ++q) {
        int laneA = (q / 2) * 32 + row;
        uint32_t idxval = cu.read_vgpr(idx_base, laneA);
        int field = (idxval >> (4 * (q % 2))) & 0xF;
        int p0 = field & 3, p1 = (field >> 2) & 3;
        for (int s = 0; s < 2; ++s) {
          float av = ex(cu, s0, (2 * q + s) % 4, laneA);
          acc += av * Bcol[4 * q + (s == 0 ? p0 : p1)];
        }
      }
      results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(acc)});
    }
  }
  for (const auto &r : results)
    cu.write_vgpr(dst + r.reg, r.lane, r.val);
}

/// SMFMAC 16x16x64 f16/bf16 (gfx950-insts). K=64, 16 sparse groups.
/// A = v4 (8 halves/lane), B = v8 (16 halves/lane), D = v4 f32.
template <typename Extract>
void exec_smfmac_f32_16x16x64_f16(ComputeUnitCore &cu, uint32_t dst, uint32_t s0, uint32_t s1,
                                  uint32_t idx_base, Extract ex) {
  struct Result {
    uint32_t reg, lane, val;
  };
  std::vector<Result> results;
  results.reserve(16 * 16);
  for (uint32_t row = 0; row < 16; ++row) {
    for (uint32_t col = 0; col < 16; ++col) {
      auto out = output_loc_32(16, 16, row, col, 0);
      float acc = std::bit_cast<float>(cu.read_vgpr(dst + out.reg, out.lane));
      float Bcol[64];
      for (int g = 0; g < 4; ++g)
        for (int e = 0; e < 16; ++e) {
          int k = 32 * (e / 8) + 8 * g + (e % 8);
          Bcol[k] = ex(cu, s1, e, g * 16 + col);
        }
      for (int q = 0; q < 16; ++q) {
        int laneA = (q / 4) * 16 + row;
        uint32_t idxval = cu.read_vgpr(idx_base, laneA);
        int field = (idxval >> (4 * (q % 4))) & 0xF;
        int p0 = field & 3, p1 = (field >> 2) & 3;
        for (int s = 0; s < 2; ++s) {
          float av = ex(cu, s0, (2 * q + s) % 8, laneA);
          acc += av * Bcol[4 * q + (s == 0 ? p0 : p1)];
        }
      }
      results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(acc)});
    }
  }
  for (const auto &r : results)
    cu.write_vgpr(dst + r.reg, r.lane, r.val);
}

/// SMFMAC 32x32x32 f16/bf16 (gfx950-insts). K=32, 8 sparse groups.
/// A = v4 (8 halves/lane), B = v8 (16 halves/lane), D = v16 f32.
template <typename Extract>
void exec_smfmac_f32_32x32x32_f16(ComputeUnitCore &cu, uint32_t dst, uint32_t s0, uint32_t s1,
                                  uint32_t idx_base, Extract ex) {
  struct Result {
    uint32_t reg, lane, val;
  };
  std::vector<Result> results;
  results.reserve(32 * 32);
  for (uint32_t row = 0; row < 32; ++row) {
    for (uint32_t col = 0; col < 32; ++col) {
      auto out = output_loc_32(32, 32, row, col, 0);
      float acc = std::bit_cast<float>(cu.read_vgpr(dst + out.reg, out.lane));
      float Bcol[32];
      for (int sl = 0; sl < 2; ++sl) {
        uint32_t src = col + 32 * sl;
        int kgrp = sl;
        for (int e = 0; e < 16; ++e) {
          int k = 16 * (e / 8) + 8 * kgrp + 2 * ((e / 2) % 4) + (e % 2);
          Bcol[k] = ex(cu, s1, e, src);
        }
      }
      for (int q = 0; q < 8; ++q) {
        int laneA = (q / 4) * 32 + row;
        uint32_t idxval = cu.read_vgpr(idx_base, laneA);
        int field = (idxval >> (4 * (q % 4))) & 0xF;
        int p0 = field & 3, p1 = (field >> 2) & 3;
        for (int s = 0; s < 2; ++s) {
          float av = ex(cu, s0, (2 * q + s) % 8, laneA);
          acc += av * Bcol[4 * q + (s == 0 ? p0 : p1)];
        }
      }
      results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(acc)});
    }
  }
  for (const auto &r : results)
    cu.write_vgpr(dst + r.reg, r.lane, r.val);
}

/// SMFMAC 16x16x64 fp8 (CDNA3 fp8-insts). K=64, 16 sparse groups.
/// A = v2 (8 bytes/lane), B = v4 (16 bytes/lane), D = v4 f32.
template <typename ExtractA, typename ExtractB>
void exec_smfmac_f32_16x16x64_fp8(ComputeUnitCore &cu, uint32_t dst, uint32_t s0, uint32_t s1,
                                  uint32_t idx_base, ExtractA ea, ExtractB eb) {
  struct Result {
    uint32_t reg, lane, val;
  };
  std::vector<Result> results;
  results.reserve(16 * 16);
  for (uint32_t row = 0; row < 16; ++row) {
    for (uint32_t col = 0; col < 16; ++col) {
      auto out = output_loc_32(16, 16, row, col, 0);
      float acc = std::bit_cast<float>(cu.read_vgpr(dst + out.reg, out.lane));
      float Bcol[64];
      for (int g = 0; g < 4; ++g)
        for (int e = 0; e < 16; ++e) {
          int k = 32 * (e / 8) + 8 * g + (e % 8);
          Bcol[k] = eb(cu, s1, e, g * 16 + col);
        }
      for (int q = 0; q < 16; ++q) {
        int ga = ((2 * q) % 16) / 4;
        int laneA = ga * 16 + row;
        int idxlane = ((q % 8) / 2) * 16 + row;
        int nb = 2 * (q / 8) + (q % 2);
        uint32_t idxval = cu.read_vgpr(idx_base, idxlane);
        int field = (idxval >> (4 * nb)) & 0xF;
        int p0 = field & 3, p1 = (field >> 2) & 3;
        for (int s = 0; s < 2; ++s) {
          int cc = 2 * q + s;
          int byte = 4 * (cc / 16) + (cc % 16) % 4;
          float av = ea(cu, s0, byte, laneA);
          acc += av * Bcol[4 * q + (s == 0 ? p0 : p1)];
        }
      }
      results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(acc)});
    }
  }
  for (const auto &r : results)
    cu.write_vgpr(dst + r.reg, r.lane, r.val);
}

/// SMFMAC 32x32x32 fp8 (CDNA3 fp8-insts). K=32, 8 sparse groups.
/// A = v2 (8 bytes/lane), B = v4 (16 bytes/lane), D = v16 f32.
template <typename ExtractA, typename ExtractB>
void exec_smfmac_f32_32x32x32_fp8(ComputeUnitCore &cu, uint32_t dst, uint32_t s0, uint32_t s1,
                                  uint32_t idx_base, ExtractA ea, ExtractB eb) {
  struct Result {
    uint32_t reg, lane, val;
  };
  std::vector<Result> results;
  results.reserve(32 * 32);
  for (uint32_t row = 0; row < 32; ++row) {
    for (uint32_t col = 0; col < 32; ++col) {
      auto out = output_loc_32(32, 32, row, col, 0);
      float acc = std::bit_cast<float>(cu.read_vgpr(dst + out.reg, out.lane));
      float Bcol[32];
      for (int kgrp = 0; kgrp < 2; ++kgrp) {
        uint32_t b_lane = col + 32 * kgrp;
        for (int e = 0; e < 16; ++e) {
          int k = 16 * (e / 8) + 8 * kgrp + 2 * ((e / 2) % 4) + (e % 2);
          Bcol[k] = eb(cu, s1, e, b_lane);
        }
      }
      for (int q = 0; q < 8; ++q) {
        int ga = ((2 * q) % 8) / 4;
        int laneA = ga * 32 + row;
        int idxlane = ((q % 4) / 2) * 32 + row;
        int nb = 2 * (q / 4) + (q % 2);
        uint32_t idxval = cu.read_vgpr(idx_base, idxlane);
        int field = (idxval >> (4 * nb)) & 0xF;
        int p0 = field & 3, p1 = (field >> 2) & 3;
        for (int s = 0; s < 2; ++s) {
          int cc = 2 * q + s;
          int byte = 4 * (cc / 8) + (cc % 8) % 4;
          float av = ea(cu, s0, byte, laneA);
          acc += av * Bcol[4 * q + (s == 0 ? p0 : p1)];
        }
      }
      results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(acc)});
    }
  }
  for (const auto &r : results)
    cu.write_vgpr(dst + r.reg, r.lane, r.val);
}

/// SMFMAC 16x16x128 fp8 (gfx950-insts). K=128, 32 sparse groups.
/// A = v4 (16 bytes/lane), B = v8 (32 bytes/lane), D = v4 f32.
template <typename ExtractA, typename ExtractB>
void exec_smfmac_f32_16x16x128_fp8(ComputeUnitCore &cu, uint32_t dst, uint32_t s0, uint32_t s1,
                                   uint32_t idx_base, ExtractA ea, ExtractB eb) {
  struct Result {
    uint32_t reg, lane, val;
  };
  std::vector<Result> results;
  results.reserve(16 * 16);
  for (uint32_t row = 0; row < 16; ++row) {
    for (uint32_t col = 0; col < 16; ++col) {
      auto out = output_loc_32(16, 16, row, col, 0);
      float acc = std::bit_cast<float>(cu.read_vgpr(dst + out.reg, out.lane));
      float Bcol[128];
      for (int g = 0; g < 4; ++g)
        for (int e = 0; e < 32; ++e) {
          int k = 32 * (e / 8) + 8 * g + (e % 8);
          Bcol[k] = eb(cu, s1, e, g * 16 + col);
        }
      for (int q = 0; q < 32; ++q) {
        int idxlane = 16 * (2 * (q / 16) + ((q / 4) % 2)) + row;
        int nb = 2 * ((q / 8) % 2) + 4 * ((q % 4) / 2) + ((q % 4) % 2);
        uint32_t idxval = cu.read_vgpr(idx_base, idxlane);
        int field = (idxval >> (4 * nb)) & 0xF;
        int p0 = field & 3, p1 = (field >> 2) & 3;
        for (int s = 0; s < 2; ++s) {
          int cc = 2 * q + s;
          int ga = 2 * ((cc >> 5) & 1) + ((cc >> 3) & 1);
          int hb = 2 * ((cc >> 2) & 1) + ((cc >> 4) & 1);
          int byte = 4 * hb + (cc & 3);
          float av = ea(cu, s0, byte, ga * 16 + row);
          acc += av * Bcol[4 * q + (s == 0 ? p0 : p1)];
        }
      }
      results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(acc)});
    }
  }
  for (const auto &r : results)
    cu.write_vgpr(dst + r.reg, r.lane, r.val);
}

/// SMFMAC 32x32x64 fp8 (gfx950-insts). K=64, 16 sparse groups.
/// A = v4 (16 bytes/lane), B = v8 (32 bytes/lane), D = v16 f32.
template <typename ExtractA, typename ExtractB>
void exec_smfmac_f32_32x32x64_fp8(ComputeUnitCore &cu, uint32_t dst, uint32_t s0, uint32_t s1,
                                  uint32_t idx_base, ExtractA ea, ExtractB eb) {
  struct Result {
    uint32_t reg, lane, val;
  };
  std::vector<Result> results;
  results.reserve(32 * 32);
  for (uint32_t row = 0; row < 32; ++row) {
    for (uint32_t col = 0; col < 32; ++col) {
      auto out = output_loc_32(32, 32, row, col, 0);
      float acc = std::bit_cast<float>(cu.read_vgpr(dst + out.reg, out.lane));
      float Bcol[64];
      for (int kgrp = 0; kgrp < 2; ++kgrp) {
        uint32_t b_lane = kgrp * 32 + col;
        for (int e = 0; e < 32; ++e) {
          int k = 16 * (e / 8) + 8 * kgrp + (e % 8);
          Bcol[k] = eb(cu, s1, e, b_lane);
        }
      }
      for (int q = 0; q < 16; ++q) {
        int idxlane = 32 * (q / 8) + row;
        int nb = (q % 2) + 2 * ((q / 4) % 2) + 4 * ((q / 2) % 2);
        uint32_t idxval = cu.read_vgpr(idx_base, idxlane);
        int field = (idxval >> (4 * nb)) & 0xF;
        int p0 = field & 3, p1 = (field >> 2) & 3;
        for (int s = 0; s < 2; ++s) {
          int cc = 2 * q + s;
          int ga = (cc >> 4) & 1;
          int hb = 2 * ((cc >> 2) & 1) + ((cc >> 3) & 1);
          int byte = 4 * hb + (cc & 3);
          float av = ea(cu, s0, byte, ga * 32 + row);
          acc += av * Bcol[4 * q + (s == 0 ? p0 : p1)];
        }
      }
      results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(acc)});
    }
  }
  for (const auto &r : results)
    cu.write_vgpr(dst + r.reg, r.lane, r.val);
}

/// Fast path for v_mfma_f32_16x16x32_f16. This single MFMA shape is the only
/// MFMA variant fired by OPT-125M fp16 eager forward (488k invocations per
/// forward; ~4B internal MACs). Kept as a dedicated specialization (rather
/// than forwarding to generic exec_f32) because the compile-time M/N/K/B let
/// the compiler fully unroll the 16-row x 32-K inner matmul into straight-line
/// AVX-512 FMAs — a runtime-dimension loop is materially slower on this hot
/// path. Hoists A and B into dense f32 buffers via extract_f16, runs the
/// matmul as 16 zmm-wide f32 FMA rows (512 zmm FMAs per MFMA), and scatters
/// directly back to VGPRs (no Result staging vector). VGPR access is batched
/// through vgpr_data (one base pointer per operand, no per-element virtual
/// read_vgpr/write_vgpr). Falls back to the generic exec_f32 when:
///   - <experimental/simd> is unavailable
///   - host native_simd<float> is not 16 lanes (i.e. no AVX-512)
///   - cbsz/blgp lane permutation is non-default
///   - RJ_FORCE_SCALAR is set
/// Fast path for the f32-input MFMA shapes (v_mfma_f32_*_f32). Like the f16
/// specialization but the inputs are already f32, so there is no F16C convert —
/// the hoist reads each operand word straight through vgpr_data. BATCH covers
/// the batched shapes (e.g. 32x32x1x2). Compile-time M/N/K/BATCH fully unroll
/// the matmul; it loops over N in zmm-width chunks (N must be a multiple of 16,
/// so the 4x4 shape stays on the generic path). Falls back to the generic
/// exec_f32 without AVX-512 / under force-scalar / with cbsz|blgp.
template <uint32_t M, uint32_t N, uint32_t K, uint32_t BATCH>
void exec_f32_mfma_f32_spec(amdgpu::ComputeUnitCore &cu, uint32_t dst, uint32_t s0, uint32_t s1,
                            uint32_t s2, uint32_t const_acc, uint32_t cbsz, uint32_t abid,
                            uint32_t blgp) {
  constexpr uint32_t in_bits = 32;
  static_assert(N % 16 == 0, "specialized f32 MFMA assumes N is a multiple of the zmm width");
  if constexpr (!util::has_stdx_simd) {
    exec_f32(cu, M, N, K, BATCH, in_bits, dst, s0, s1, s2, amdgpu::extract_f32, amdgpu::extract_f32,
             const_acc, cbsz, abid, blgp);
    return;
  } else {
    if (util::force_scalar() || cbsz != 0 || blgp != 0 || util::native<float>::size() != 16 ||
        cu.wf_size() != 64) {
      exec_f32(cu, M, N, K, BATCH, in_bits, dst, s0, s1, s2, amdgpu::extract_f32,
               amdgpu::extract_f32, const_acc, cbsz, abid, blgp);
      return;
    }
    constexpr uint32_t W = 16;
    const uint32_t wf = cu.wf_size();
    const uint32_t *a_words = reinterpret_cast<const uint32_t *>(cu.vgpr_data(s0));
    const uint32_t *b_words = reinterpret_cast<const uint32_t *>(cu.vgpr_data(s1));
    const uint32_t *c_words = reinterpret_cast<const uint32_t *>(cu.vgpr_data(s2));
    uint32_t *d_words = reinterpret_cast<uint32_t *>(cu.vgpr_data(dst));
    alignas(64) float A_buf[M * K];
    alignas(64) float B_buf[K * N];
    alignas(64) float C_buf[M * N];
    bool has_nan_or_inf = false;
    for (uint32_t b = 0; b < BATCH; ++b) {
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t col = 0; col < N; ++col) {
          auto out = output_loc_32(M, N, row, col, b);
          C_buf[row * N + col] = (const_acc != ACC_FROM_VGPR)
                                     ? std::bit_cast<float>(const_acc)
                                     : std::bit_cast<float>(c_words[out.reg * wf + out.lane]);
        }
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t k = 0; k < K; ++k) {
          auto al = input_loc(M, K, BATCH, row, k, b, in_bits);
          A_buf[row * K + k] = std::bit_cast<float>(a_words[al.vgpr_offset * wf + al.lane]);
        }
      for (uint32_t k = 0; k < K; ++k)
        for (uint32_t col = 0; col < N; ++col) {
          auto bl = input_loc(N, K, BATCH, col, k, b, in_bits);
          B_buf[k * N + col] = std::bit_cast<float>(b_words[bl.vgpr_offset * wf + bl.lane]);
        }
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t c0 = 0; c0 < N; c0 += W) {
          util::native<float> c_row;
          c_row.copy_from(&C_buf[row * N + c0], util::stdx::vector_aligned);
          for (uint32_t k = 0; k < K; ++k) {
            util::native<float> a_bcast(A_buf[row * K + k]);
            util::native<float> b_row;
            b_row.copy_from(&B_buf[k * N + c0], util::stdx::vector_aligned);
            c_row = util::stdx::fma(a_bcast, b_row, c_row);
          }
          c_row.copy_to(&C_buf[row * N + c0], util::stdx::vector_aligned);
        }
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t col = 0; col < N; ++col) {
          auto out = output_loc_32(M, N, row, col, b);
          float fv = C_buf[row * N + col];
          d_words[out.reg * wf + out.lane] = std::bit_cast<uint32_t>(fv);
          if (std::isnan(fv) || std::isinf(fv))
            has_nan_or_inf = true;
        }
    }
    if (has_nan_or_inf) {
      util::Logger::vm([&](auto &os) {
        os << std::format("MFMA_NAN_DETECTED (simd) dst=v{} s0=v{} s1=v{} s2=v{} {}x{}x{}_f32", dst,
                          s0, s1, s2, M, N, K);
      });
    }
  }
}

template <uint32_t M, uint32_t N, uint32_t K, uint32_t BATCH = 1>
void exec_f32_mfma_f16_spec(amdgpu::ComputeUnitCore &cu, uint32_t dst, uint32_t s0, uint32_t s1,
                            uint32_t s2, uint32_t const_acc, uint32_t cbsz, uint32_t abid,
                            uint32_t blgp) {
  constexpr uint32_t B = BATCH, in_bits = 16;
  static_assert(N % 16 == 0, "specialized f16 MFMA assumes N is a multiple of the zmm width");
  if constexpr (!util::has_stdx_simd) {
    exec_f32(cu, M, N, K, B, in_bits, dst, s0, s1, s2, amdgpu::extract_f16, amdgpu::extract_f16,
             const_acc, cbsz, abid, blgp);
    return;
  } else {
    if (util::force_scalar() || cbsz != 0 || blgp != 0 || util::native<float>::size() != 16 ||
        cu.wf_size() != 64) {
      exec_f32(cu, M, N, K, B, in_bits, dst, s0, s1, s2, amdgpu::extract_f16, amdgpu::extract_f16,
               const_acc, cbsz, abid, blgp);
      return;
    }
    constexpr uint32_t W = 16; // guaranteed by the native<float>::size()==16 guard above
    const uint32_t wf = cu.wf_size();
    const uint32_t *a_words = reinterpret_cast<const uint32_t *>(cu.vgpr_data(s0));
    const uint32_t *b_words = reinterpret_cast<const uint32_t *>(cu.vgpr_data(s1));
    const uint32_t *c_words = reinterpret_cast<const uint32_t *>(cu.vgpr_data(s2));
    uint32_t *d_words = reinterpret_cast<uint32_t *>(cu.vgpr_data(dst));
    alignas(64) float A_buf[M * K]; // A[row][k] (one batch block)
    alignas(64) float B_buf[K * N]; // B[k][col] (one batch block)
    alignas(64) float C_buf[M * N]; // C[row][col] (one batch block)
    // A/B occupy M*K*B and N*K*B packed f16 over their VGPRs. Bulk-convert each
    // whole region to f32 once with F16C (one vector op per 16 halves) instead
    // of branchy per-element f16_to_f32, then the hoist is a pure f32
    // index-shuffle (f16 of word w lane l sub s -> flat (w*wf+l)*2+s).
    alignas(64) float A_f32[M * K * B];
    alignas(64) float B_f32[N * K * B];
    util::f16_to_f32_block(reinterpret_cast<const uint16_t *>(a_words), A_f32, M * K * B);
    util::f16_to_f32_block(reinterpret_cast<const uint16_t *>(b_words), B_f32, N * K * B);
    bool has_nan_or_inf = false;
    for (uint32_t b = 0; b < B; ++b) {
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t col = 0; col < N; ++col) {
          auto out = output_loc_32(M, N, row, col, b);
          C_buf[row * N + col] = (const_acc != ACC_FROM_VGPR)
                                     ? std::bit_cast<float>(const_acc)
                                     : std::bit_cast<float>(c_words[out.reg * wf + out.lane]);
        }
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t k = 0; k < K; ++k) {
          auto al = input_loc(M, K, B, row, k, b, in_bits);
          A_buf[row * K + k] = A_f32[(al.vgpr_offset * wf + al.lane) * 2 + al.sub_element];
        }
      for (uint32_t k = 0; k < K; ++k)
        for (uint32_t col = 0; col < N; ++col) {
          auto bl = input_loc(N, K, B, col, k, b, in_bits);
          B_buf[k * N + col] = B_f32[(bl.vgpr_offset * wf + bl.lane) * 2 + bl.sub_element];
        }
      // Dense MxKxN matmul, W-lane (zmm) stdx FMA over N (N/W chunks per row).
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t c0 = 0; c0 < N; c0 += W) {
          util::native<float> c_row;
          c_row.copy_from(&C_buf[row * N + c0], util::stdx::vector_aligned);
          for (uint32_t k = 0; k < K; ++k) {
            util::native<float> a_bcast(A_buf[row * K + k]);
            util::native<float> b_row;
            b_row.copy_from(&B_buf[k * N + c0], util::stdx::vector_aligned);
            c_row = util::stdx::fma(a_bcast, b_row, c_row);
          }
          c_row.copy_to(&C_buf[row * N + c0], util::stdx::vector_aligned);
        }
      // Scatter directly back to VGPRs (no Result staging vector).
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t col = 0; col < N; ++col) {
          auto out = output_loc_32(M, N, row, col, b);
          float fv = C_buf[row * N + col];
          d_words[out.reg * wf + out.lane] = std::bit_cast<uint32_t>(fv);
          if (std::isnan(fv) || std::isinf(fv))
            has_nan_or_inf = true;
        }
    }
    if (has_nan_or_inf) {
      util::Logger::vm([&](auto &os) {
        os << std::format("MFMA_NAN_DETECTED (simd) dst=v{} s0=v{} s1=v{} s2=v{} {}x{}x{}_f16", dst,
                          s0, s1, s2, M, N, K);
      });
    }
  }
}

/// Fast path for the bf16-input MFMA shapes (v_mfma_f32_*_bf16). Identical to
/// the f16 specialization except the bulk input convert is the bf16 zero-extend
/// (no F16C needed). Falls back to the generic exec_f32 without AVX-512 / under
/// force-scalar / with cbsz|blgp.
template <uint32_t M, uint32_t N, uint32_t K, uint32_t BATCH = 1>
void exec_f32_mfma_bf16_spec(amdgpu::ComputeUnitCore &cu, uint32_t dst, uint32_t s0, uint32_t s1,
                             uint32_t s2, uint32_t const_acc, uint32_t cbsz, uint32_t abid,
                             uint32_t blgp) {
  constexpr uint32_t B = BATCH, in_bits = 16;
  static_assert(N % 16 == 0, "specialized bf16 MFMA assumes N is a multiple of the zmm width");
  if constexpr (!util::has_stdx_simd) {
    exec_f32(cu, M, N, K, B, in_bits, dst, s0, s1, s2, amdgpu::extract_bf16, amdgpu::extract_bf16,
             const_acc, cbsz, abid, blgp);
    return;
  } else {
    if (util::force_scalar() || cbsz != 0 || blgp != 0 || util::native<float>::size() != 16 ||
        cu.wf_size() != 64) {
      exec_f32(cu, M, N, K, B, in_bits, dst, s0, s1, s2, amdgpu::extract_bf16, amdgpu::extract_bf16,
               const_acc, cbsz, abid, blgp);
      return;
    }
    constexpr uint32_t W = 16; // guaranteed by the native<float>::size()==16 guard above
    const uint32_t wf = cu.wf_size();
    const uint32_t *a_words = reinterpret_cast<const uint32_t *>(cu.vgpr_data(s0));
    const uint32_t *b_words = reinterpret_cast<const uint32_t *>(cu.vgpr_data(s1));
    const uint32_t *c_words = reinterpret_cast<const uint32_t *>(cu.vgpr_data(s2));
    uint32_t *d_words = reinterpret_cast<uint32_t *>(cu.vgpr_data(dst));
    alignas(64) float A_buf[M * K]; // A[row][k] (one batch block)
    alignas(64) float B_buf[K * N]; // B[k][col] (one batch block)
    alignas(64) float C_buf[M * N]; // C[row][col] (one batch block)
    // Bulk-convert the packed bf16 regions to f32 once (zero-extend + shift),
    // then the hoist is a pure f32 index-shuffle.
    alignas(64) float A_f32[M * K * B];
    alignas(64) float B_f32[N * K * B];
    util::bf16_to_f32_block(reinterpret_cast<const uint16_t *>(a_words), A_f32, M * K * B);
    util::bf16_to_f32_block(reinterpret_cast<const uint16_t *>(b_words), B_f32, N * K * B);
    bool has_nan_or_inf = false;
    for (uint32_t b = 0; b < B; ++b) {
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t col = 0; col < N; ++col) {
          auto out = output_loc_32(M, N, row, col, b);
          C_buf[row * N + col] = (const_acc != ACC_FROM_VGPR)
                                     ? std::bit_cast<float>(const_acc)
                                     : std::bit_cast<float>(c_words[out.reg * wf + out.lane]);
        }
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t k = 0; k < K; ++k) {
          auto al = input_loc(M, K, B, row, k, b, in_bits);
          A_buf[row * K + k] = A_f32[(al.vgpr_offset * wf + al.lane) * 2 + al.sub_element];
        }
      for (uint32_t k = 0; k < K; ++k)
        for (uint32_t col = 0; col < N; ++col) {
          auto bl = input_loc(N, K, B, col, k, b, in_bits);
          B_buf[k * N + col] = B_f32[(bl.vgpr_offset * wf + bl.lane) * 2 + bl.sub_element];
        }
      // Dense MxKxN matmul, W-lane (zmm) stdx FMA over N (N/W chunks per row).
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t c0 = 0; c0 < N; c0 += W) {
          util::native<float> c_row;
          c_row.copy_from(&C_buf[row * N + c0], util::stdx::vector_aligned);
          for (uint32_t k = 0; k < K; ++k) {
            util::native<float> a_bcast(A_buf[row * K + k]);
            util::native<float> b_row;
            b_row.copy_from(&B_buf[k * N + c0], util::stdx::vector_aligned);
            c_row = util::stdx::fma(a_bcast, b_row, c_row);
          }
          c_row.copy_to(&C_buf[row * N + c0], util::stdx::vector_aligned);
        }
      // Scatter directly back to VGPRs (no Result staging vector).
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t col = 0; col < N; ++col) {
          auto out = output_loc_32(M, N, row, col, b);
          float fv = C_buf[row * N + col];
          d_words[out.reg * wf + out.lane] = std::bit_cast<uint32_t>(fv);
          if (std::isnan(fv) || std::isinf(fv))
            has_nan_or_inf = true;
        }
    }
    if (has_nan_or_inf) {
      util::Logger::vm([&](auto &os) {
        os << std::format("MFMA_NAN_DETECTED (simd) dst=v{} s0=v{} s1=v{} s2=v{} {}x{}x{}_bf16",
                          dst, s0, s1, s2, M, N, K);
      });
    }
  }
}

/// Fast path for the dense (B=1) fp8/bf8-input MFMA shapes
/// (v_mfma_f32_{16x16x32,32x32x16}_{fp8,bf8}_{fp8,bf8}). Identical to the
/// f16/bf16 specializations except the bulk input convert is the f8 LUT gather
/// (bit-exact with extract_fp8/extract_bf8 by construction); A/B formats are
/// compile-time selected. Falls back to the generic exec_f32 without AVX-512 /
/// under force-scalar / with cbsz|blgp.
template <uint32_t M, uint32_t N, uint32_t K, bool A_FP8, bool B_FP8, bool FNUZ = false>
void exec_f32_mfma_f8_spec(amdgpu::ComputeUnitCore &cu, uint32_t dst, uint32_t s0, uint32_t s1,
                           uint32_t s2, uint32_t const_acc, uint32_t cbsz, uint32_t abid,
                           uint32_t blgp) {
  constexpr uint32_t B = 1, in_bits = 8;
  static_assert(N % 16 == 0, "specialized f8 MFMA assumes N is a multiple of the zmm width");
  constexpr auto ea = f8_extract_fn<A_FP8, FNUZ>();
  constexpr auto eb = f8_extract_fn<B_FP8, FNUZ>();
  if constexpr (!util::has_stdx_simd) {
    exec_f32(cu, M, N, K, B, in_bits, dst, s0, s1, s2, ea, eb, const_acc, cbsz, abid, blgp);
    return;
  } else {
    if (util::force_scalar() || cbsz != 0 || blgp != 0 || util::native<float>::size() != 16 ||
        cu.wf_size() != 64) {
      exec_f32(cu, M, N, K, B, in_bits, dst, s0, s1, s2, ea, eb, const_acc, cbsz, abid, blgp);
      return;
    }
    constexpr uint32_t W = 16; // guaranteed by the native<float>::size()==16 guard above
    const uint32_t wf = cu.wf_size();
    const uint32_t *a_words = reinterpret_cast<const uint32_t *>(cu.vgpr_data(s0));
    const uint32_t *b_words = reinterpret_cast<const uint32_t *>(cu.vgpr_data(s1));
    const uint32_t *c_words = reinterpret_cast<const uint32_t *>(cu.vgpr_data(s2));
    alignas(64) float A_buf[M * K]; // A[row][k]
    alignas(64) float B_buf[K * N]; // B[k][col]
    alignas(64) float C_buf[M * N]; // C[row][col]
    // Bulk-convert the packed f8 regions to f32 once through the LUTs, then the
    // hoist is a pure f32 index-shuffle (byte of word w lane l sub s ->
    // (w*wf+l)*4+s).
    alignas(64) float A_f32[M * K];
    alignas(64) float B_f32[N * K];
    f8_to_f32_block<A_FP8, FNUZ>(a_words, A_f32, M * K);
    f8_to_f32_block<B_FP8, FNUZ>(b_words, B_f32, N * K);
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = output_loc_32(M, N, row, col, 0);
        C_buf[row * N + col] = (const_acc != ACC_FROM_VGPR)
                                   ? std::bit_cast<float>(const_acc)
                                   : std::bit_cast<float>(c_words[out.reg * wf + out.lane]);
      }
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t k = 0; k < K; ++k) {
        auto al = input_loc(M, K, B, row, k, 0, in_bits);
        A_buf[row * K + k] = A_f32[(al.vgpr_offset * wf + al.lane) * 4 + al.sub_element];
      }
    for (uint32_t k = 0; k < K; ++k)
      for (uint32_t col = 0; col < N; ++col) {
        auto bl = input_loc(N, K, B, col, k, 0, in_bits);
        B_buf[k * N + col] = B_f32[(bl.vgpr_offset * wf + bl.lane) * 4 + bl.sub_element];
      }
    // Dense MxKxN matmul, W-lane (zmm) stdx FMA over N (N/W chunks per row).
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t c0 = 0; c0 < N; c0 += W) {
        util::native<float> c_row;
        c_row.copy_from(&C_buf[row * N + c0], util::stdx::vector_aligned);
        for (uint32_t k = 0; k < K; ++k) {
          util::native<float> a_bcast(A_buf[row * K + k]);
          util::native<float> b_row;
          b_row.copy_from(&B_buf[k * N + c0], util::stdx::vector_aligned);
          c_row = util::stdx::fma(a_bcast, b_row, c_row);
        }
        c_row.copy_to(&C_buf[row * N + c0], util::stdx::vector_aligned);
      }
    // Scatter directly back to VGPRs (no Result staging vector).
    uint32_t *d_words = reinterpret_cast<uint32_t *>(cu.vgpr_data(dst));
    bool has_nan_or_inf = false;
    for (uint32_t row = 0; row < M; ++row)
      for (uint32_t col = 0; col < N; ++col) {
        auto out = output_loc_32(M, N, row, col, 0);
        float fv = C_buf[row * N + col];
        d_words[out.reg * wf + out.lane] = std::bit_cast<uint32_t>(fv);
        if (std::isnan(fv) || std::isinf(fv))
          has_nan_or_inf = true;
      }
    if (has_nan_or_inf) {
      util::Logger::vm([&](auto &os) {
        os << std::format("MFMA_NAN_DETECTED (simd) dst=v{} s0=v{} s1=v{} s2=v{} {}x{}x{}_f8", dst,
                          s0, s1, s2, M, N, K);
      });
    }
  }
}

/// Fast path for the dense i8-input MFMA shapes (v_mfma_i32_*_i8), including
/// the batched (BATCH>1) variants. Same structure as the f16 specialization
/// but integer: the packed i8 inputs are bulk sign-extended to int32 once,
/// the hoist is a pure i32 index-shuffle, and the matmul runs as
/// constexpr-unrolled zmm-wide int32 MACs. There is no clamp on the i8 MFMA
/// path: both scalar and SIMD accumulate (and wrap) in 32 bits, so they agree
/// bit for bit on every output (SIMD wraps in uint32 to keep the wrap
/// well-defined). Falls back to the generic exec_i32_i8 without AVX-512 /
/// under force-scalar.
template <uint32_t M, uint32_t N, uint32_t K, uint32_t BATCH = 1>
void exec_i32_mfma_i8_spec(amdgpu::ComputeUnitCore &cu, uint32_t dst, uint32_t s0, uint32_t s1,
                           uint32_t s2, uint32_t const_acc) {
  constexpr uint32_t B = BATCH, in_bits = 8;
  static_assert(N % 16 == 0, "specialized i8 MFMA assumes N is a multiple of the zmm width");
  if constexpr (!util::has_stdx_simd) {
    exec_i32_i8(cu, M, N, K, B, dst, s0, s1, s2, const_acc);
    return;
  } else {
    if (util::force_scalar() || util::native<float>::size() != 16 || cu.wf_size() != 64) {
      exec_i32_i8(cu, M, N, K, B, dst, s0, s1, s2, const_acc);
      return;
    }
    constexpr uint32_t W = 16; // guaranteed by the native<float>::size()==16 guard above
    const uint32_t wf = cu.wf_size();
    const uint32_t *a_words = reinterpret_cast<const uint32_t *>(cu.vgpr_data(s0));
    const uint32_t *b_words = reinterpret_cast<const uint32_t *>(cu.vgpr_data(s1));
    const uint32_t *c_words = reinterpret_cast<const uint32_t *>(cu.vgpr_data(s2));
    // Accumulate in unsigned 32-bit (wrap is well-defined and identical mod
    // 2^32 to the intended signed wrap), so the SIMD path has no signed-
    // overflow UB.
    uint32_t *d_words = reinterpret_cast<uint32_t *>(cu.vgpr_data(dst));
    alignas(64) uint32_t A_buf[M * K]; // A[row][k] (sign-extended bits, one batch block)
    alignas(64) uint32_t B_buf[K * N]; // B[k][col] (sign-extended bits, one batch block)
    alignas(64) uint32_t C_buf[M * N]; // C[row][col] (one batch block)
    // Bulk sign-extend the packed i8 regions to int32 once, then the hoist is
    // a pure i32 index-shuffle (byte of word w lane l sub s -> (w*wf+l)*4+s).
    alignas(64) int32_t A_i32[M * K * B];
    alignas(64) int32_t B_i32[N * K * B];
    util::i8_to_i32_block(reinterpret_cast<const int8_t *>(a_words), A_i32, M * K * B);
    util::i8_to_i32_block(reinterpret_cast<const int8_t *>(b_words), B_i32, N * K * B);
    for (uint32_t b = 0; b < B; ++b) {
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t col = 0; col < N; ++col) {
          auto out = output_loc_32(M, N, row, col, b);
          C_buf[row * N + col] =
              (const_acc != ACC_FROM_VGPR) ? const_acc : c_words[out.reg * wf + out.lane];
        }
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t k = 0; k < K; ++k) {
          auto al = input_loc(M, K, B, row, k, b, in_bits);
          A_buf[row * K + k] = A_i32[(al.vgpr_offset * wf + al.lane) * 4 + al.sub_element];
        }
      for (uint32_t k = 0; k < K; ++k)
        for (uint32_t col = 0; col < N; ++col) {
          auto bl = input_loc(N, K, B, col, k, b, in_bits);
          B_buf[k * N + col] = B_i32[(bl.vgpr_offset * wf + bl.lane) * 4 + bl.sub_element];
        }
      // Dense MxKxN matmul, W-lane (zmm) stdx u32 MAC over N (N/W chunks per
      // row); unsigned wrap matches the scalar signed accumulation mod 2^32.
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t c0 = 0; c0 < N; c0 += W) {
          util::native<uint32_t> c_row;
          c_row.copy_from(&C_buf[row * N + c0], util::stdx::vector_aligned);
          for (uint32_t k = 0; k < K; ++k) {
            util::native<uint32_t> a_bcast(A_buf[row * K + k]);
            util::native<uint32_t> b_row;
            b_row.copy_from(&B_buf[k * N + c0], util::stdx::vector_aligned);
            c_row += a_bcast * b_row;
          }
          c_row.copy_to(&C_buf[row * N + c0], util::stdx::vector_aligned);
        }
      // Scatter directly back to VGPRs (no Result staging vector).
      for (uint32_t row = 0; row < M; ++row)
        for (uint32_t col = 0; col < N; ++col) {
          auto out = output_loc_32(M, N, row, col, b);
          d_words[out.reg * wf + out.lane] = C_buf[row * N + col];
        }
    }
  }
}

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_EXEC_H_
