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

inline OutputLoc wmma_output_loc_32(uint32_t M, uint32_t N, uint32_t row, uint32_t col) {
  uint32_t elems_per_lane = (M * N) / WMMA_WAVE32;
  uint32_t lane = (row / elems_per_lane) * N + col;
  uint32_t reg = row % elems_per_lane;
  return {reg, lane};
}

inline PackedOutputLoc wmma_output_loc_16(uint32_t M, uint32_t N, uint32_t row, uint32_t col) {
  uint32_t elems_per_lane = (M * N) / WMMA_WAVE32;
  uint32_t lane = (row / elems_per_lane) * N + col;
  uint32_t elem = row % elems_per_lane;
  return {elem / 2, lane, elem % 2};
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

inline float extract_fp8(amdgpu::ComputeUnitCore &cu, uint32_t base, const InputLoc &loc) {
  uint32_t raw = cu.read_vgpr(base + loc.vgpr_offset, loc.lane);
  return util::fp8_e4m3_to_f32(static_cast<uint8_t>((raw >> (loc.sub_element * 8)) & 0xFF));
}

inline float extract_bf8(amdgpu::ComputeUnitCore &cu, uint32_t base, const InputLoc &loc) {
  uint32_t raw = cu.read_vgpr(base + loc.vgpr_offset, loc.lane);
  return util::bf8_e5m2_to_f32(static_cast<uint8_t>((raw >> (loc.sub_element * 8)) & 0xFF));
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

inline float decode_e8m0_scale(uint8_t raw) {
  if (raw == 0)
    return 0.0f;
  if (raw == 0xff)
    return std::numeric_limits<float>::quiet_NaN();
  return std::ldexp(1.0f, static_cast<int>(raw) - 127);
}

inline float decode_wmma_scale_byte(uint8_t raw, uint32_t fmt) {
  switch (fmt) {
  case 0:
    return decode_e8m0_scale(raw);
  default:
    throw util::UnimplementedInst("unsupported WMMA scale format");
  }
}

inline uint32_t wmma_scale_lane(uint32_t index, uint32_t scale_select) {
  return index + ((scale_select & 0x1u) ? 16u : 0u);
}

inline uint32_t wmma_scale_byte(const InputLoc &loc) {
  return (input_local_element(loc) / 16u) & 0x3u;
}

inline uint32_t wmma_b_fp4_scale_byte(uint32_t k) {
  return ((k / 16u) & 0x1u) | (((k / 64u) & 0x1u) << 1u);
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
        // AMD convention: i=row (register dimension), j=col (lane dimension).
        auto out = output_loc_32(M, N, row, col, b);
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
          float a_val = ea(cu, s0, al);
          float b_val = eb(cu, s1, bl);
          acc += a_val * b_val;
        }
        results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(acc)});
      }
    }
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

template <typename ExtractA, typename ExtractB>
void exec_wmma_f32_mixed(amdgpu::ComputeUnitCore &cu, uint32_t M, uint32_t N, uint32_t K,
                         uint32_t a_bits, uint32_t b_bits, uint32_t dst, uint32_t s0, uint32_t s1,
                         uint32_t s2, ExtractA ea, ExtractB eb,
                         uint32_t const_acc = ACC_FROM_VGPR) {
  require_wmma_wave32(cu);
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t val;
  };
  std::vector<Result> results;
  results.reserve(M * N);
  for (uint32_t row = 0; row < M; ++row) {
    for (uint32_t col = 0; col < N; ++col) {
      auto out = wmma_output_loc_32(M, N, row, col);
      float acc = (const_acc != ACC_FROM_VGPR)
                      ? std::bit_cast<float>(const_acc)
                      : std::bit_cast<float>(cu.read_vgpr(s2 + out.reg, out.lane));
      for (uint32_t k = 0; k < K; ++k) {
        auto al = wmma_input_loc(M, K, row, k, a_bits);
        auto bl = wmma_input_loc(N, K, col, k, b_bits);
        acc += ea(cu, s0, al) * eb(cu, s1, bl);
      }
      results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(acc)});
    }
  }
  for (const auto &r : results)
    cu.write_vgpr(dst + r.reg, r.lane, r.val);
}

template <typename ExtractA, typename ExtractB, typename ScaleAWord, typename ScaleBWord>
void exec_wmma_f32_scaled_mixed(amdgpu::ComputeUnitCore &cu, uint32_t M, uint32_t N, uint32_t K,
                                uint32_t a_bits, uint32_t b_bits, uint32_t dst, uint32_t s0,
                                uint32_t s1, uint32_t s2, ExtractA ea, ExtractB eb,
                                uint32_t const_acc, ScaleAWord scale_a_word,
                                ScaleBWord scale_b_word, uint32_t matrix_a_scale,
                                uint32_t matrix_b_scale, uint32_t matrix_a_scale_fmt,
                                uint32_t matrix_b_scale_fmt) {
  require_wmma_wave32(cu);
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t val;
  };
  std::vector<Result> results;
  results.reserve(M * N);
  for (uint32_t row = 0; row < M; ++row) {
    for (uint32_t col = 0; col < N; ++col) {
      auto out = wmma_output_loc_32(M, N, row, col);
      float acc = (const_acc != ACC_FROM_VGPR)
                      ? std::bit_cast<float>(const_acc)
                      : std::bit_cast<float>(cu.read_vgpr(s2 + out.reg, out.lane));
      const uint32_t a_scale_word = scale_a_word(wmma_scale_lane(row, matrix_a_scale));
      const uint32_t b_scale_word = scale_b_word(wmma_scale_lane(col, matrix_b_scale));
      for (uint32_t k = 0; k < K; ++k) {
        auto al = wmma_input_loc(M, K, row, k, a_bits);
        auto bl = wmma_input_loc(N, K, col, k, b_bits);
        const uint32_t a_scale_byte = wmma_scale_byte(al);
        const uint32_t b_scale_byte =
            (b_bits == 4) ? wmma_b_fp4_scale_byte(k) : wmma_scale_byte(bl);
        const float a_scale = decode_wmma_scale_byte(
            static_cast<uint8_t>((a_scale_word >> (a_scale_byte * 8)) & 0xffu), matrix_a_scale_fmt);
        const float b_scale = decode_wmma_scale_byte(
            static_cast<uint8_t>((b_scale_word >> (b_scale_byte * 8)) & 0xffu), matrix_b_scale_fmt);
        acc += ea(cu, s0, al) * eb(cu, s1, bl) * a_scale * b_scale;
      }
      results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(acc)});
    }
  }
  for (const auto &r : results)
    cu.write_vgpr(dst + r.reg, r.lane, r.val);
}

template <typename ExtractA, typename ExtractB>
void exec_wmma_f32(amdgpu::ComputeUnitCore &cu, uint32_t M, uint32_t N, uint32_t K,
                   uint32_t in_bits, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                   ExtractA ea, ExtractB eb, uint32_t const_acc = ACC_FROM_VGPR) {
  exec_wmma_f32_mixed(cu, M, N, K, in_bits, in_bits, dst, s0, s1, s2, ea, eb, const_acc);
}

template <typename ExtractA, typename ExtractB>
void exec_swmmac_f32_mixed(amdgpu::ComputeUnitCore &cu, uint32_t M, uint32_t N, uint32_t K,
                           uint32_t a_bits, uint32_t b_bits, uint32_t dst, uint32_t s0, uint32_t s1,
                           uint32_t acc_base, uint32_t index_base, uint32_t index_entries,
                           uint32_t index_key, ExtractA ea, ExtractB eb,
                           uint32_t const_acc = ACC_FROM_VGPR) {
  require_wmma_wave32(cu);
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t val;
  };
  std::vector<Result> results;
  results.reserve(M * N);

  const uint32_t compressed_k = K / 2;
  for (uint32_t row = 0; row < M; ++row) {
    for (uint32_t col = 0; col < N; ++col) {
      auto out = wmma_output_loc_32(M, N, row, col);
      float acc = (const_acc != ACC_FROM_VGPR)
                      ? std::bit_cast<float>(const_acc)
                      : std::bit_cast<float>(cu.read_vgpr(acc_base + out.reg, out.lane));
      for (uint32_t ck = 0; ck < compressed_k; ++ck) {
        auto al = wmma_input_loc(M, K / 2, row, ck, a_bits);
        const uint32_t metadata_lane = row + (ck / index_entries) * M;
        const uint32_t local_ck = ck % index_entries;
        const uint64_t index_set =
            read_swmmac_index_set(cu, index_base, metadata_lane, index_entries, index_key);
        const uint32_t dense_k = swmmac_dense_k(index_set, ck, local_ck);
        auto bl = wmma_input_loc(N, K, col, dense_k, b_bits);
        acc += ea(cu, s0, al) * eb(cu, s1, bl);
      }
      results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(acc)});
    }
  }
  for (const auto &r : results)
    cu.write_vgpr(dst + r.reg, r.lane, r.val);
}

template <typename ExtractA, typename ExtractB>
void exec_swmmac_f32(amdgpu::ComputeUnitCore &cu, uint32_t M, uint32_t N, uint32_t K,
                     uint32_t in_bits, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t acc_base,
                     uint32_t index_base, uint32_t index_entries, uint32_t index_key, ExtractA ea,
                     ExtractB eb, uint32_t const_acc = ACC_FROM_VGPR) {
  exec_swmmac_f32_mixed(cu, M, N, K, in_bits, in_bits, dst, s0, s1, acc_base, index_base,
                        index_entries, index_key, ea, eb, const_acc);
}

template <typename ExtractA, typename ExtractB, typename ReadAcc, typename PackResult>
void exec_wmma_packed16(amdgpu::ComputeUnitCore &cu, uint32_t M, uint32_t N, uint32_t K,
                        uint32_t in_bits, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                        ExtractA ea, ExtractB eb, ReadAcc read_acc, PackResult pack_result,
                        uint32_t const_acc = ACC_FROM_VGPR) {
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
      auto out = wmma_output_loc_16(M, N, row, col);
      float acc = (const_acc != ACC_FROM_VGPR)
                      ? std::bit_cast<float>(const_acc)
                      : read_acc(cu, s2 + out.reg, out.lane, out.sub_element);
      for (uint32_t k = 0; k < K; ++k) {
        auto al = wmma_input_loc(M, K, row, k, in_bits);
        auto bl = wmma_input_loc(N, K, col, k, in_bits);
        acc += ea(cu, s0, al) * eb(cu, s1, bl);
      }
      results.push_back({out.reg, out.lane, out.sub_element, pack_result(acc)});
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
                          PackResult pack_result, uint32_t const_acc = ACC_FROM_VGPR) {
  require_wmma_wave32(cu);
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t sub_element;
    uint16_t val;
  };
  std::vector<Result> results;
  results.reserve(M * N);

  const uint32_t compressed_k = K / 2;
  for (uint32_t row = 0; row < M; ++row) {
    for (uint32_t col = 0; col < N; ++col) {
      auto out = wmma_output_loc_16(M, N, row, col);
      float acc = (const_acc != ACC_FROM_VGPR)
                      ? std::bit_cast<float>(const_acc)
                      : read_acc(cu, acc_base + out.reg, out.lane, out.sub_element);
      for (uint32_t ck = 0; ck < compressed_k; ++ck) {
        auto al = wmma_input_loc(M, K / 2, row, ck, in_bits);
        const uint32_t metadata_lane = row + (ck / index_entries) * M;
        const uint32_t local_ck = ck % index_entries;
        const uint64_t index_set =
            read_swmmac_index_set(cu, index_base, metadata_lane, index_entries, index_key);
        const uint32_t dense_k = swmmac_dense_k(index_set, ck, local_ck);
        auto bl = wmma_input_loc(N, K, col, dense_k, in_bits);
        acc += ea(cu, s0, al) * eb(cu, s1, bl);
      }
      results.push_back({out.reg, out.lane, out.sub_element, pack_result(acc)});
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

template <typename ExtractA, typename ExtractB>
void exec_wmma_f16(amdgpu::ComputeUnitCore &cu, uint32_t M, uint32_t N, uint32_t K,
                   uint32_t in_bits, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                   ExtractA ea, ExtractB eb, uint32_t const_acc = ACC_FROM_VGPR) {
  exec_wmma_packed16(
      cu, M, N, K, in_bits, dst, s0, s1, s2, ea, eb,
      [](amdgpu::ComputeUnitCore &cu, uint32_t reg, uint32_t lane, uint32_t sub) {
        uint32_t raw = cu.read_vgpr(reg, lane);
        return util::f16_to_f32(static_cast<uint16_t>((raw >> (sub * 16)) & 0xFFFF));
      },
      [](float val) { return util::f32_to_f16(val); }, const_acc);
}

template <typename ExtractA, typename ExtractB>
void exec_swmmac_f16(amdgpu::ComputeUnitCore &cu, uint32_t M, uint32_t N, uint32_t K,
                     uint32_t in_bits, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t acc_base,
                     uint32_t index_base, uint32_t index_entries, uint32_t index_key, ExtractA ea,
                     ExtractB eb, uint32_t const_acc = ACC_FROM_VGPR) {
  exec_swmmac_packed16(
      cu, M, N, K, in_bits, dst, s0, s1, acc_base, index_base, index_entries, index_key, ea, eb,
      [](amdgpu::ComputeUnitCore &cu, uint32_t reg, uint32_t lane, uint32_t sub) {
        uint32_t raw = cu.read_vgpr(reg, lane);
        return util::f16_to_f32(static_cast<uint16_t>((raw >> (sub * 16)) & 0xFFFF));
      },
      [](float val) { return util::f32_to_f16(val); }, const_acc);
}

template <typename ExtractA, typename ExtractB>
void exec_wmma_bf16(amdgpu::ComputeUnitCore &cu, uint32_t M, uint32_t N, uint32_t K,
                    uint32_t in_bits, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                    ExtractA ea, ExtractB eb, uint32_t const_acc = ACC_FROM_VGPR) {
  exec_wmma_packed16(
      cu, M, N, K, in_bits, dst, s0, s1, s2, ea, eb,
      [](amdgpu::ComputeUnitCore &cu, uint32_t reg, uint32_t lane, uint32_t sub) {
        uint32_t raw = cu.read_vgpr(reg, lane);
        return util::bf16_to_f32(static_cast<uint16_t>((raw >> (sub * 16)) & 0xFFFF));
      },
      [](float val) { return util::f32_to_bf16(val); }, const_acc);
}

template <typename ExtractA, typename ExtractB>
void exec_swmmac_bf16(amdgpu::ComputeUnitCore &cu, uint32_t M, uint32_t N, uint32_t K,
                      uint32_t in_bits, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t acc_base,
                      uint32_t index_base, uint32_t index_entries, uint32_t index_key, ExtractA ea,
                      ExtractB eb, uint32_t const_acc = ACC_FROM_VGPR) {
  exec_swmmac_packed16(
      cu, M, N, K, in_bits, dst, s0, s1, acc_base, index_base, index_entries, index_key, ea, eb,
      [](amdgpu::ComputeUnitCore &cu, uint32_t reg, uint32_t lane, uint32_t sub) {
        uint32_t raw = cu.read_vgpr(reg, lane);
        return util::bf16_to_f32(static_cast<uint16_t>((raw >> (sub * 16)) & 0xFFFF));
      },
      [](float val) { return util::f32_to_bf16(val); }, const_acc);
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
        auto out = output_loc_32(M, N, row, col, b);
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
            block_sum += ea(cu, s0, al) * eb(cu, s1, bl);
          }
          uint32_t sa_raw = cu.read_vgpr(scale_a_base, out.lane);
          uint32_t sb_raw = cu.read_vgpr(scale_b_base, out.lane);
          uint8_t sa_e8m0 = static_cast<uint8_t>((sa_raw >> (blk * 8)) & 0xFF);
          uint8_t sb_e8m0 = static_cast<uint8_t>((sb_raw >> (blk * 8)) & 0xFF);
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

/// Scaled MFMA for mixed-format f8f6f4: A and B may have different bit widths.
/// cbsz/blgp are used as format selectors (not lane permutations).
template <typename ExtractA, typename ExtractB>
void exec_f32_scaled_mixed(amdgpu::ComputeUnitCore &cu, uint32_t M, uint32_t N, uint32_t K,
                           uint32_t B, uint32_t a_bits, uint32_t b_bits, uint32_t dst, uint32_t s0,
                           uint32_t s1, uint32_t s2, ExtractA ea, ExtractB eb, uint32_t const_acc,
                           uint32_t scale_a_base, uint32_t scale_b_base) {
  constexpr uint32_t BLOCK_K = 32;
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
        auto out = output_loc_32(M, N, row, col, b);
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
            block_sum += ea(cu, s0, al) * eb(cu, s1, bl);
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
        // AMD convention: i=row (register dimension), j=col (lane dimension).
        auto out = output_loc_32(M, N, row, col, b);
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

inline uint32_t pack_i32_acc(int64_t acc, bool clamp) {
  if (clamp) {
    acc = std::clamp(acc, static_cast<int64_t>(std::numeric_limits<int32_t>::min()),
                     static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
  }
  return static_cast<uint32_t>(acc);
}

template <typename ExtractA, typename ExtractB>
inline void exec_wmma_i32(amdgpu::ComputeUnitCore &cu, uint32_t M, uint32_t N, uint32_t K,
                          uint32_t in_bits, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                          ExtractA ea, ExtractB eb, bool clamp,
                          uint32_t const_acc = ACC_FROM_VGPR) {
  require_wmma_wave32(cu);
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t val;
  };
  std::vector<Result> results;
  results.reserve(M * N);
  for (uint32_t row = 0; row < M; ++row) {
    for (uint32_t col = 0; col < N; ++col) {
      auto out = wmma_output_loc_32(M, N, row, col);
      int64_t acc =
          (const_acc != ACC_FROM_VGPR)
              ? static_cast<int64_t>(static_cast<int32_t>(const_acc))
              : static_cast<int64_t>(static_cast<int32_t>(cu.read_vgpr(s2 + out.reg, out.lane)));
      for (uint32_t k = 0; k < K; ++k) {
        auto al = wmma_input_loc(M, K, row, k, in_bits);
        auto bl = wmma_input_loc(N, K, col, k, in_bits);
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

template <typename ExtractA, typename ExtractB>
inline void exec_swmmac_i32(amdgpu::ComputeUnitCore &cu, uint32_t M, uint32_t N, uint32_t K,
                            uint32_t in_bits, uint32_t dst, uint32_t s0, uint32_t s1,
                            uint32_t acc_base, uint32_t index_base, uint32_t index_entries,
                            uint32_t index_key, ExtractA ea, ExtractB eb, bool clamp,
                            uint32_t const_acc = ACC_FROM_VGPR) {
  require_wmma_wave32(cu);
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t val;
  };
  std::vector<Result> results;
  results.reserve(M * N);

  const uint32_t compressed_k = K / 2;
  for (uint32_t row = 0; row < M; ++row) {
    for (uint32_t col = 0; col < N; ++col) {
      auto out = wmma_output_loc_32(M, N, row, col);
      int64_t acc = (const_acc != ACC_FROM_VGPR)
                        ? static_cast<int64_t>(static_cast<int32_t>(const_acc))
                        : static_cast<int64_t>(
                              static_cast<int32_t>(cu.read_vgpr(acc_base + out.reg, out.lane)));
      for (uint32_t ck = 0; ck < compressed_k; ++ck) {
        auto al = wmma_input_loc(M, K / 2, row, ck, in_bits);
        const uint32_t metadata_lane = row + (ck / index_entries) * M;
        const uint32_t local_ck = ck % index_entries;
        const uint64_t index_set =
            read_swmmac_index_set(cu, index_base, metadata_lane, index_entries, index_key);
        const uint32_t dense_k = swmmac_dense_k(index_set, ck, local_ck);
        auto bl = wmma_input_loc(N, K, col, dense_k, in_bits);
        acc += static_cast<int64_t>(ea(cu, s0, al)) * static_cast<int64_t>(eb(cu, s1, bl));
      }
      results.push_back({out.reg, out.lane, pack_i32_acc(acc, clamp)});
    }
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

// ---------------------------------------------------------------------------
// SMFMAC (Sparse Matrix FMA) helpers and execution functions.
//
// Structured 2:4 sparsity: A is half-density (2 of every 4 K positions are
// nonzero). A per-lane index register selects which 2-of-4 positions are live.
// Each 4-bit nibble in the index encodes two 2-bit position selectors (p0, p1).
// ---------------------------------------------------------------------------

inline float smfmac_read_fp8(ComputeUnitCore &cu, uint32_t base, uint32_t byte_idx, uint32_t lane) {
  uint32_t raw = cu.read_vgpr(base + byte_idx / 4, lane);
  return util::fp8_e4m3_to_f32(static_cast<uint8_t>((raw >> ((byte_idx % 4) * 8)) & 0xFF));
}

inline float smfmac_read_bf8(ComputeUnitCore &cu, uint32_t base, uint32_t byte_idx, uint32_t lane) {
  uint32_t raw = cu.read_vgpr(base + byte_idx / 4, lane);
  return util::bf8_e5m2_to_f32(static_cast<uint8_t>((raw >> ((byte_idx % 4) * 8)) & 0xFF));
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

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_EXEC_H_
