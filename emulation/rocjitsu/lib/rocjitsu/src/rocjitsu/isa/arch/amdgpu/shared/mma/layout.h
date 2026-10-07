// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_LAYOUT_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_LAYOUT_H_

#include "rocjitsu/isa/arch/amdgpu/shared/accvgpr_layout.h"
#include "util/except.h"
#include <cassert>
#include <cstdint>

namespace rocjitsu::amdgpu {

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
/// Encoding 768-1023 = acc[0-255] on CDNA1, which types MFMA/accvgpr destinations
/// as OPR_ACCVGPR (OPR_ACCVGPR_ACC_MIN = 768) rather than the 512-based range.
inline uint32_t dst_base(uint32_t vb, int ev, uint32_t acc_cd = 1) {
  if (ev >= 768)
    return vb + ACC_VGPR_OFFSET + static_cast<uint32_t>(ev - 768);
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

inline void require_gfx11_wmma_wave_size(uint32_t wave_size) {
  if (wave_size != WMMA_WAVE32 && wave_size != WMMA_WAVE64)
    throw util::ConfigError("gfx11 WMMA requires wave32 or wave64");
}

inline void require_gfx12_wmma_wave_size(uint32_t wave_size) {
  if (wave_size != WMMA_WAVE32 && wave_size != WMMA_WAVE64)
    throw util::ConfigError("gfx12 WMMA requires wave32 or wave64");
}

// CDNA5 WMMA is wave32-only. 16x16 WMMA v3 operands interleave the K
// dimension across the two lane groups in two register blocks.
inline InputLoc wmma_input_loc(uint32_t dim, uint32_t K, uint32_t i, uint32_t k,
                               uint32_t data_bits) {
  uint32_t lanes_per_group = WMMA_WAVE32 / dim;
  uint32_t elems_per_group = K / lanes_per_group;

  uint32_t local = k % elems_per_group;
  uint32_t lane = (k / elems_per_group) * dim + i;

  if (dim == 16 && K >= 32) {
    uint32_t block_elems = elems_per_group / 2;
    // CDNA5 K=128 dense WMMA uses 16-element K blocks. Do not change the
    // pre-existing K=64 family, which uses the default 16-element blocks too.
    if (data_bits == 8 && K == 128)
      block_elems = 16;
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

/// Compute an input element location for CDNA4 block-scale F8F6F4 MFMA.
///
/// Scaled MFMA uses the same dense operand layout as the corresponding
/// unscaled F8F6F4 operation. The supported shapes both use one matrix block
/// distributed across a wave64.
inline InputLoc mfma_scale_f8f6f4_input_loc(uint32_t dim, uint32_t K, uint32_t index, uint32_t k,
                                            uint32_t data_bits) {
  if (!((dim == 16 && K == 128) || (dim == 32 && K == 64)))
    throw util::UnimplementedInst("unsupported CDNA4 block-scale MFMA input shape");
  if (data_bits != 4 && data_bits != 6 && data_bits != 8)
    throw util::UnimplementedInst("unsupported CDNA4 block-scale MFMA input format");
  return input_loc(dim, K, /*B=*/1, index, k, /*b=*/0, data_bits);
}

inline InputLoc gfx12_wmma_input_loc(uint32_t wave_size, uint32_t dim, uint32_t K, uint32_t i,
                                     uint32_t k, uint32_t data_bits) {
  require_gfx12_wmma_wave_size(wave_size);
  if (wave_size == WMMA_WAVE32)
    return wmma_input_loc(dim, K, i, k, data_bits);
  if (dim == 16 && data_bits == 4 && (K == 16 || K == 32)) {
    const uint32_t block = k / 8u;
    const uint32_t lane_block = K == 16 ? block : ((block & 1u) << 1u) | (block >> 1u);
    return wmma_packed_input_loc(i + 16u * lane_block, k & 7u, data_bits);
  }
  if (dim == 16 && K == 16) {
    const uint32_t lane = i + 16u * ((k >> 2) & 1u) + 32u * ((k >> 3) & 1u);
    const uint32_t slot = 2u * ((k >> 1) & 1u) + (k & 1u);
    return wmma_packed_input_loc(lane, slot, data_bits);
  }
  throw util::UnimplementedInst("unsupported gfx12 wave64 WMMA input layout");
}

inline InputLoc wmma_f8f6f4_ab_input_loc(uint32_t dim, uint32_t K, uint32_t i, uint32_t k,
                                         uint32_t data_bits) {
  if (dim == 16 && K == 128 && data_bits < 8) {
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
  auto loc = wmma_f8f6f4_input_loc(16, 128, row % 16, k, 4, /*mixed_subbyte=*/false);
  loc.vgpr_offset += 8 * (row / 16);
  return loc;
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

/// Compute the CDNA5 block-scaled WMMA input layout. Unlike the ordinary
/// F8F6F4 WMMA layout, each operand format stores contiguous K ranges in each
/// lane group: 16 values for 8-bit inputs and 32 values for 4/6-bit inputs.
inline InputLoc wmma_block_scaled_input_loc(uint32_t dim, uint32_t K, uint32_t index, uint32_t k,
                                            uint32_t data_bits) {
  if (dim != 16 || K != 128 || (data_bits != 4 && data_bits != 6 && data_bits != 8))
    throw util::UnimplementedInst("unsupported CDNA5 block-scaled WMMA input shape");
  const uint32_t block_elems = data_bits == 8 ? 16u : 32u;
  const uint32_t lane = index + 16u * ((k / block_elems) & 1u);
  const uint32_t local = (k / (2u * block_elems)) * block_elems + (k % block_elems);
  return wmma_packed_input_loc(lane, local, data_bits);
}

inline InputLoc wmma_block_scaled_a_input_loc(uint32_t M, uint32_t K, uint32_t row, uint32_t k,
                                              uint32_t a_bits) {
  if (M == 32 && K == 128 && a_bits == 4) {
    auto loc = wmma_block_scaled_input_loc(16, K, row % 16, k, a_bits);
    loc.vgpr_offset += 8u * (row / 16u);
    return loc;
  }
  return wmma_block_scaled_input_loc(M, K, row, k, a_bits);
}

inline InputLoc wmma_block_scaled_b_input_loc(uint32_t N, uint32_t K, uint32_t col, uint32_t k,
                                              uint32_t b_bits) {
  return wmma_block_scaled_input_loc(N, K, col, k, b_bits);
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
  if (M == 32 && N == 16) {
    const uint32_t local_row = row % 16;
    const uint32_t lane = (local_row / 8) * N + col;
    const uint32_t reg = 8 * (row / 16) + local_row % 8;
    return {reg, lane};
  }
  uint32_t elems_per_lane = (M * N) / WMMA_WAVE32;
  uint32_t lane = (row / elems_per_lane) * N + col;
  uint32_t reg = row % elems_per_lane;
  return {reg, lane};
}

/// Map one logical F64 CDNA5 WMMA output to its pair of consecutive VGPRs.
///
/// The public CDNA5 ISA reference, section 7.12.2, maps the eight logical
/// 16x16 C/D components per lane with the same row/lane ordering represented
/// by wmma_output_loc_32(). The checked-in FMT_WMMA_DC_16X16_F64 data format
/// then maps each logical F64 component to two consecutive 32-bit VGPRs.
inline OutputLoc wmma_output_loc_64(uint32_t M, uint32_t N, uint32_t row, uint32_t col) {
  const auto logical = wmma_output_loc_32(M, N, row, col);
  return {2u * logical.reg, logical.lane};
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
  if (M == 16 && K == 64 && elem_bits == 4 && index_entries == 16)
    return {row + 16u * ((compressed_k / 8u) & 1u), 8u * (compressed_k / 16u) + compressed_k % 8u};
  // This generic routing is also intentional for gfx1250 K=128 8-bit
  // SWMMAC. Hardware-reference Tensile kernels require contiguous 32-entry
  // selector blocks even though sparse A changes lane halves every 16 packed
  // K elements. That differs from the association described by the public
  // CDNA5 ISA Sections 7.12.3 and 7.12.5; retain the validated behavior
  // pending specification clarification.
  return {row + (compressed_k / index_entries) * M, compressed_k % index_entries};
}

inline SwmmacIndexLoc swmmac_index_loc(uint32_t wave_size, uint32_t M, uint32_t K,
                                       uint32_t elem_bits, uint32_t row, uint32_t compressed_k,
                                       uint32_t index_entries) {
  require_gfx12_wmma_wave_size(wave_size);
  if (wave_size == WMMA_WAVE32)
    return swmmac_index_loc(M, K, elem_bits, row, compressed_k, index_entries);
  if (M == 16 && elem_bits == 4 && index_entries == 16 && (K == 32 || K == 64)) {
    const uint32_t block = compressed_k / 8u;
    const uint32_t lane_block = K == 32 ? block : ((block & 1u) << 1u) | (block >> 1u);
    return {row + 16u * lane_block, compressed_k & 7u};
  }
  if (M == 16 && K == 32 && index_entries == 16) {
    const uint32_t group = compressed_k / 2u;
    const uint32_t slot = compressed_k & 1u;
    const uint32_t block =
        (elem_bits <= 8) ? (2u * ((group >> 1) & 1u) + (group >> 2)) : (group >> 1);
    return {row + 16u * block, 2u * (group & 1u) + slot};
  }
  throw util::UnimplementedInst("unsupported gfx12 wave64 SWMMAC index layout");
}

[[gnu::always_inline]] inline InputLoc swmmac_a_input_loc(uint32_t M, uint32_t K, uint32_t row,
                                                          uint32_t compressed_k,
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

[[gnu::always_inline]] inline InputLoc swmmac_a_input_loc(uint32_t wave_size, uint32_t M,
                                                          uint32_t K, uint32_t row,
                                                          uint32_t compressed_k,
                                                          uint32_t elem_bits) {
  require_gfx12_wmma_wave_size(wave_size);
  if (wave_size == WMMA_WAVE32)
    return swmmac_a_input_loc(M, K, row, compressed_k, elem_bits);
  if (M == 16 && elem_bits == 4 && (K == 32 || K == 64)) {
    const uint32_t block = compressed_k / 8u;
    const uint32_t lane_block = K == 32 ? block : ((block & 1u) << 1u) | (block >> 1u);
    return wmma_packed_input_loc(row + 16u * lane_block, compressed_k & 7u, elem_bits);
  }
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

[[gnu::always_inline]] inline InputLoc swmmac_b_input_loc(uint32_t N, uint32_t K, uint32_t col,
                                                          uint32_t dense_k, uint32_t elem_bits) {
  if (N == 16 && K == 128 && elem_bits == 8) {
    // Hardware-reference gfx1250 Tensile kernels require this 32-element
    // SWMMAC B ordering. It differs from both dense K=128 WMMA and the public
    // CDNA5 ISA Section 7.12.5; retain the validated instruction-specific
    // layout pending specification clarification.
    const uint32_t lane = col + 16u * ((dense_k >> 5) & 1u);
    const uint32_t slot = (dense_k & 31u) + 32u * (dense_k >> 6);
    return wmma_packed_input_loc(lane, slot, elem_bits);
  }
  if (N == 16 && K == 32) {
    if (elem_bits == 4 || elem_bits == 8)
      return wmma_packed_input_loc(col + 16u * (dense_k / 16u), dense_k % 16u, elem_bits);
    if (elem_bits >= 16) {
      const uint32_t lane = col + 16u * ((dense_k / 8u) & 1u);
      const uint32_t slot = 8u * (dense_k / 16u) + 2u * ((dense_k / 2u) & 3u) + (dense_k & 1u);
      return wmma_packed_input_loc(lane, slot, elem_bits);
    }
  }
  return wmma_input_loc(N, K, col, dense_k, elem_bits);
}

[[gnu::always_inline]] inline InputLoc swmmac_b_input_loc(uint32_t wave_size, uint32_t N,
                                                          uint32_t K, uint32_t col,
                                                          uint32_t dense_k, uint32_t elem_bits) {
  require_gfx12_wmma_wave_size(wave_size);
  if (wave_size == WMMA_WAVE32)
    return swmmac_b_input_loc(N, K, col, dense_k, elem_bits);
  if (N == 16 && elem_bits == 4 && (K == 32 || K == 64)) {
    const uint32_t values_per_lane = K == 32 ? 8u : 16u;
    const uint32_t block = dense_k / values_per_lane;
    const uint32_t lane_block = ((block & 1u) << 1u) | (block >> 1u);
    return wmma_packed_input_loc(col + 16u * lane_block, dense_k % values_per_lane, elem_bits);
  }
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

inline uint32_t wmma_scale_lane(uint32_t index, uint32_t scale_select) {
  return index + ((scale_select & 0x1u) ? 16u : 0u);
}

inline uint32_t wmma_f4_32x16x128_a_scale_lane(uint32_t row) { return row; }

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

inline uint32_t wmma_block_scale_byte(uint32_t k, bool scale16) {
  return k / (scale16 ? 16u : 32u);
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

inline uint64_t mfma_full_lane_mask(uint32_t wf_size) {
  return wf_size >= 64 ? ~uint64_t{0} : ((uint64_t{1} << wf_size) - 1);
}

inline bool is_gfx1251_wmma_execution_state_valid(uint32_t wf_size, uint64_t exec_mask) {
  return wf_size == WMMA_WAVE32 && exec_mask == mfma_full_lane_mask(WMMA_WAVE32);
}

inline uint32_t mfma_dense_reg_count(uint64_t element_count, uint32_t element_bits,
                                     uint32_t wf_size) {
  uint64_t bits_per_reg = static_cast<uint64_t>(wf_size) * 32u;
  return static_cast<uint32_t>((element_count * element_bits + bits_per_reg - 1) / bits_per_reg);
}

/// Map f32 output position to packed 16-bit output position using GFX9 layout.
/// Two consecutive f32 register positions pack into one 32-bit VGPR as two
/// 16-bit sub-elements: reg/2 holds the VGPR offset, reg%2 the sub-element.
inline PackedOutputLoc output_loc_16(uint32_t M, uint32_t N, uint32_t i, uint32_t j, uint32_t b) {
  auto f32 = output_loc_32(M, N, i, j, b);
  return {f32.reg / 2, f32.lane, f32.reg % 2};
}

enum class SwmmacK64Input { F16, BF16 };
enum class SwmmacK64Accumulator { F32, F16, BF16 };
enum class SwmmacK64Result { F32, F16, BF16 };

enum class SwmmacK128Input { FP8, BF8 };
enum class SwmmacK128Result { F32, F16 };

enum class SmfmacLayout { Cdna3F16, Cdna4F16, Cdna3Fp8, Cdna4Fp8 };

struct SmfmacSparseLoc {
  uint32_t lane;
  uint32_t nibble;
  uint32_t element;
};

struct SmfmacDenseLoc {
  uint32_t lane;
  uint32_t element;
};

/// The sparse A element and its 2:4 selector occupy the same source lane.
/// Keep the four ISA layouts here so scalar/SIMD comparisons exercise exactly
/// the register permutations used by the original scalar helpers below.
template <SmfmacLayout Layout, uint32_t M, uint32_t K>
inline SmfmacSparseLoc smfmac_sparse_loc(uint32_t row, uint32_t q, uint32_t s) {
  if constexpr (Layout == SmfmacLayout::Cdna3F16 || Layout == SmfmacLayout::Cdna4F16) {
    constexpr uint32_t groups_per_lane = Layout == SmfmacLayout::Cdna3F16 ? 2 : 4;
    return {(q / groups_per_lane) * M + row, q % groups_per_lane,
            (2 * q + s) % (2 * groups_per_lane)};
  } else if constexpr (Layout == SmfmacLayout::Cdna3Fp8) {
    constexpr uint32_t half_groups = K / 8;
    return {((q % half_groups) / 2) * M + row, 2 * (q / half_groups) + (q % 2),
            4 * (q / half_groups) + (2 * q + s) % 4};
  } else if constexpr (M == 16) {
    const uint32_t lane_group = 2 * (q / 16) + ((q / 4) % 2);
    const uint32_t nibble = 2 * ((q / 8) % 2) + 4 * ((q % 4) / 2) + (q % 2);
    const uint32_t byte = 4 * (2 * ((q / 2) % 2) + ((q / 8) % 2)) + (2 * q + s) % 4;
    return {lane_group * M + row, nibble, byte};
  } else {
    static_assert(M == 32);
    const uint32_t lane_group = q / 8;
    const uint32_t nibble = (q % 2) + 2 * ((q / 4) % 2) + 4 * ((q / 2) % 2);
    const uint32_t byte = 4 * (2 * ((q / 2) % 2) + ((q / 4) % 2)) + (2 * q + s) % 4;
    return {lane_group * M + row, nibble, byte};
  }
}

template <SmfmacLayout Layout, uint32_t N>
inline SmfmacDenseLoc smfmac_dense_loc(uint32_t k, uint32_t col) {
  if constexpr (Layout == SmfmacLayout::Cdna3F16) {
    if constexpr (N == 16)
      return {(k / 8) * 16 + col, k % 8};
    else
      return {16 * ((col / 16) + 2 * (k / 8)) + (col % 16), k % 8};
  } else if constexpr (N == 16) {
    return {16 * ((k % 32) / 8) + col, 8 * (k / 32) + (k % 8)};
  } else {
    static_assert(N == 32);
    return {32 * ((k % 16) / 8) + col, 8 * (k / 16) + (k % 8)};
  }
}

/// Return whether an f32-accumulating matrix output row divides into complete
/// native SIMD chunks.
constexpr bool mma_f32_native_width_supported(uint32_t n, uint32_t width) {
  return width > 1 && n % width == 0;
}

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_LAYOUT_H_
