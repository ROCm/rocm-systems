// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_EXEC_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_EXEC_H_

/// @file Matrix operand/register mapping shared by VM bindings and layout tests.
/// rj-arith owns numerical execution; these helpers resolve architectural operands
/// and lane/element locations without carrying an arithmetic implementation.
///
/// Implements the GFX9 MFMA register layout formulas from the AMD Matrix
/// Instruction Calculator (InstCalcGfx9). Shared across CDNA1-4 (all use the
/// same GFX9 encoding family for MFMA).
///
/// AccMode governs how the accumulator source (src2) is resolved:
///   - Unified:  CDNA3/4 — VGPR and AccVGPR share a single file; encoding
///               range 256-511 = VGPR, 768-1023 = AccVGPR (alias).
///   - Separate: CDNA1/2 — dedicated AccVGPR file, src2 ranges differ.
///
/// Key conventions:
///   - Output D[i][j]: i is the register dimension (matrix column),
///     j is the lane dimension (matrix row). Call output_loc with
///     i=column, j=row to get the physical (vgpr_offset, lane).
///   - Input A[row][k]: use input_loc(dim=M, K, B, i=row, k, b, bits).
///     Input B[col][k]: use input_loc(dim=N, K, B, i=col, k, b, bits).

#include "rocjitsu/isa/arch/amdgpu/shared/accvgpr_layout.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "util/log.h"
#include "util/meta_programming.h"

#include <cassert>
#include <cstdint>
#include <format>

namespace rocjitsu {
namespace amdgpu {
// Architectural matrix operands and register/lane layout helpers.

/// Accumulator register mode, determined by CDNA generation.
enum class AccMode {
  Unified,  ///< CDNA3/4: VGPR and AccVGPR are in a unified file.
  Separate, ///< CDNA1/2: dedicated AccVGPR file.
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
  if (ev >= 768) {
    return vb + ACC_VGPR_OFFSET + static_cast<uint32_t>(ev - 768);
  }
  if (ev >= 512) {
    return vb + ACC_VGPR_OFFSET + static_cast<uint32_t>(ev - 512);
  }
  if (acc_cd) {
    return vb + ACC_VGPR_OFFSET + static_cast<uint32_t>(ev);
  }
  return vb + static_cast<uint32_t>(ev);
}

/// Resolve VGPR base for an MFMA source operand (OPR_SRC_VGPR_OR_ACCVGPR).
/// Encoding: 256-511 = ArchVGPR (v0-v255), 768-1023 = AccVGPR (acc0-acc255).
inline uint32_t src_base(uint32_t vb, int ev) {
  if (ev >= 768) {
    return vb + ACC_VGPR_OFFSET + static_cast<uint32_t>(ev - 768);
  }
  return (ev >= 256) ? vb + static_cast<uint32_t>(ev - 256) : vb + static_cast<uint32_t>(ev);
}

/// Apply GPR_IDX only when an MMA operand resolves to the architectural VGPR
/// bank. AccVGPRs occupy the unified storage range above ACC_VGPR_OFFSET but
/// are not selected by MODE.GPR_IDX_EN.
inline uint32_t apply_gpr_idx_to_mma_base(const Wavefront &wf, uint32_t vb, uint32_t base,
                                          VgprMsbRole role) {
  if (base >= vb + ACC_VGPR_OFFSET) {
    return base;
  }
  const uint32_t index = apply_gpr_idx(wf, base - vb, role);
  // Matrix execution reads physical register bases, bypassing the logical VALU source accessor.
  if (role != VgprMsbRole::Dst && !wf.vgpr_bank(false).contains(index)) {
    return vb;
  }
  return vb + index;
}

/// Sentinel value indicating the accumulator comes from a register, not a constant.
constexpr uint32_t ACC_FROM_VGPR = UINT32_MAX;

/// A constant accumulator has no physical register base. Keeping that information
/// also distinguishes the legal inline integer -1 from the register sentinel.
constexpr uint32_t NO_ACC_REGISTER = UINT32_MAX;

constexpr uint32_t WMMA_WAVE32 = 32;
constexpr uint32_t WMMA_WAVE64 = 64;

inline uint32_t wmma_c_modifier(uint32_t neg, uint32_t neg_hi) {
  return ((neg >> 2) & 0x1u) | (((neg_hi >> 2) & 0x1u) << 1);
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
///        called when src2 is not a VGPR. Typically: [&]{ return
///        amdgpu::RegisterAccess(wf).read_scalar(src2); }
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

  if (data_bits == 64) {
    return {local * 2, lane, 0, 0, data_bits};
  }
  if (data_bits == 32) {
    return {local, lane, 0, 0, data_bits};
  }
  uint32_t bit = local * data_bits;
  uint32_t bit_in_word = bit % 32;
  uint32_t sub_element = (32 % data_bits == 0) ? (bit_in_word / data_bits) : 0;
  return {bit / 32, lane, sub_element, bit_in_word, data_bits};
}

/// Compute 32-bit output element location for the GFX9 MFMA register layout.
constexpr OutputLoc output_loc_32(uint32_t M, uint32_t N, uint32_t i, uint32_t j, uint32_t b) {
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

/// Whether input_loc places the dim x K x B elements of an MFMA operand on every lane of
/// each of its mfma_dense_reg_count registers and on no other register. The closed form is
/// conservative: it accepts every hardware MFMA operand layout, and some other dense layouts
/// may be rejected.
constexpr bool mfma_input_layout_is_dense(uint32_t dim, uint32_t K, uint32_t B, uint32_t bits) {
  const uint64_t lanes_per_element = uint64_t{dim} * B;
  if (K == 0 || bits == 0 || lanes_per_element == 0 || 64 % lanes_per_element != 0) {
    return false;
  }
  const uint32_t lanes_per_block = static_cast<uint32_t>(64 / lanes_per_element);
  if (K % lanes_per_block != 0) {
    return false;
  }
  const uint32_t elems_per_group = K / lanes_per_block;
  // input_loc splits more than 16 byte-or-wider elements per lane into 16-element chunks.
  if (elems_per_group > 16 && bits >= 8 && elems_per_group % 16 != 0) {
    return false;
  }
  return uint64_t{elems_per_group} * bits % 32 == 0;
}

/// Whether the M x N x B results of a wave64 MFMA, placed by output_loc_32 (one dword per
/// element) or output_loc_64 (two), fill every word of the first M*N*B*words_per_element/64
/// VGPRs exactly once. The closed form is conservative: it accepts every hardware MFMA output
/// layout, and some other dense layouts may be rejected.
constexpr bool mfma_output_layout_is_dense(uint32_t M, uint32_t N, uint32_t B,
                                           uint32_t words_per_element) {
  if (M == 0 || N == 0 || B == 0 || 64 % N != 0) {
    return false;
  }
  const uint32_t rows_per_reg = 64 / N;
  if (words_per_element == 1) {
    // Rows go to four registers at a time, rows_per_reg rows each, so a block of a multiple
    // of 4 * rows_per_reg rows fills whole registers. Otherwise a block of four rows spans N
    // lanes of four registers, and rows_per_reg such blocks fill them.
    return M % (4 * rows_per_reg) == 0 || (M == 4 && N < 16 && uint64_t{B} * N == 64);
  }
  if (words_per_element == 2) {
    // A block of a multiple of rows_per_reg rows fills whole register pairs. A smaller block
    // spans M * N lanes of one pair, and 64 / (M * N) such blocks fill it.
    return M % rows_per_reg == 0 || (uint64_t{M} * N < 64 && uint64_t{M} * N * B == 64);
  }
  return false;
}

inline void require_gfx12_wmma_wave_size([[maybe_unused]] uint32_t wave_size) {
  assert((wave_size == WMMA_WAVE32 || wave_size == WMMA_WAVE64) &&
         "gfx12 WMMA requires wave32 or wave64");
}

// CDNA5 WMMA is wave32-only. 16x16 WMMA v3 operands interleave the K
// dimension across the two lane groups in two register blocks.
template <uint32_t dim, uint32_t K>
inline InputLoc wmma_input_loc(uint32_t i, uint32_t k, uint32_t data_bits) {
  uint32_t lanes_per_group = WMMA_WAVE32 / dim;
  uint32_t elems_per_group = K / lanes_per_group;

  uint32_t local = k % elems_per_group;
  uint32_t lane = (k / elems_per_group) * dim + i;

  if (dim == 16 && K >= 32) {
    uint32_t block_elems = elems_per_group / 2;
    // CDNA5 K=128 dense WMMA uses 16-element K blocks. Do not change the
    // pre-existing K=64 family, which uses the default 16-element blocks too.
    if (data_bits == 8 && K == 128) {
      block_elems = 16;
    }
    if (data_bits == 4 && K == 128) {
      block_elems = 16;
    }
    if (block_elems != 0) {
      const uint32_t lane_group = (k / block_elems) % lanes_per_group;
      const uint32_t reg_group = k / (block_elems * lanes_per_group);
      local = reg_group * block_elems + (k % block_elems);
      lane = lane_group * dim + i;
    }
  }

  if (data_bits == 64) {
    return {local * 2, lane, 0, 0, data_bits};
  }
  if (data_bits == 32) {
    return {local, lane, 0, 0, data_bits};
  }
  uint32_t bit = local * data_bits;
  uint32_t bit_in_word = bit % 32;
  uint32_t sub_element = (32 % data_bits == 0) ? (bit_in_word / data_bits) : 0;
  return {bit / 32, lane, sub_element, bit_in_word, data_bits};
}

template <uint32_t dim, uint32_t K>
inline InputLoc gfx11_wmma_input_loc(uint32_t i, uint32_t k, uint32_t data_bits,
                                     uint32_t lane_group) {
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
  assert(((dim == 16 && K == 128) || (dim == 32 && K == 64)) &&
         "unsupported CDNA4 block-scale MFMA input shape");
  assert((data_bits == 4 || data_bits == 6 || data_bits == 8) &&
         "unsupported CDNA4 block-scale MFMA input format");
  return input_loc(dim, K, /*B=*/1, index, k, /*b=*/0, data_bits);
}

// Swap bits 2 and 3 of a four-bit product index for WMMA operand ordering.
inline constexpr uint32_t wmma_swap_k_bits_2_3(uint32_t k) {
  return (k & 3) | ((k & 4) << 1) | ((k & 8) >> 1);
}

template <uint32_t dim, uint32_t K>
inline InputLoc gfx12_wmma_input_loc(uint32_t wave_size, uint32_t i, uint32_t k,
                                     uint32_t data_bits) {
  require_gfx12_wmma_wave_size(wave_size);
  if (wave_size == WMMA_WAVE32) {
    return wmma_input_loc<dim, K>(i, k, data_bits);
  }
  if (dim == 16 && data_bits == 4 && (K == 16 || K == 32)) {
    const uint32_t block = k / 8u;
    const uint32_t lane_block = K == 16 ? block : ((block & 1u) << 1u) | (block >> 1u);
    return wmma_packed_input_loc(i + 16u * lane_block, k & 7u, data_bits);
  }
  assert(dim == 16 && K == 16 && "unsupported gfx12 wave64 WMMA input layout");
  const uint32_t lane = i + 16u * ((k >> 2) & 1u) + 32u * ((k >> 3) & 1u);
  const uint32_t slot = 2u * ((k >> 1) & 1u) + (k & 1u);
  return wmma_packed_input_loc(lane, slot, data_bits);
}

template <uint32_t dim, uint32_t K>
inline InputLoc wmma_f8f6f4_ab_input_loc(uint32_t i, uint32_t k, uint32_t data_bits) {
  if (dim == 16 && K == 128 && data_bits < 8) {
    const uint32_t lane = i + 16u * ((k >> 2) & 1u);
    const uint32_t reg = ((k >> 1) & 1u) + 2u * ((k >> 3) & 1u) + 4u * ((k >> 4) & 1u) +
                         8u * ((k >> 5) & 1u) + 16u * ((k >> 6) & 1u);
    return wmma_packed_input_loc(lane, 2u * reg + (k & 1u), data_bits);
  }
  return wmma_input_loc<dim, K>(i, k, data_bits);
}

template <uint32_t dim, uint32_t K>
inline InputLoc wmma_f8f6f4_mixed_subbyte_input_loc(uint32_t i, uint32_t k, uint32_t data_bits) {
  if (dim == 16 && K == 128 && data_bits < 8) {
    const uint32_t lane = i + 16u * ((k >> 5) & 1u);
    // Use the same logical K order as the 8-bit operand in wmma_input_loc:
    // contiguous groups of 16 there pair with groups of 32 here. The old
    // sub-byte permutation assumed an 8-bit layout split every four elements,
    // so it paired different K values when only one operand was sub-byte.
    const uint32_t slot = 32u * (k >> 6) + (k & 31u);
    return wmma_packed_input_loc(lane, slot, data_bits);
  }
  return wmma_f8f6f4_ab_input_loc<dim, K>(i, k, data_bits);
}

template <uint32_t dim, uint32_t K>
inline InputLoc wmma_f8f6f4_input_loc(uint32_t i, uint32_t k, uint32_t data_bits,
                                      bool mixed_subbyte) {
  if (mixed_subbyte) {
    return wmma_f8f6f4_mixed_subbyte_input_loc<dim, K>(i, k, data_bits);
  }
  return wmma_f8f6f4_ab_input_loc<dim, K>(i, k, data_bits);
}

inline InputLoc wmma_f4_32x16x128_a_input_loc(uint32_t row, uint32_t k) {
  auto loc = wmma_f8f6f4_input_loc<16, 128>(row % 16, k, 4, /*mixed_subbyte=*/false);
  loc.vgpr_offset += 8 * (row / 16);
  return loc;
}

template <uint32_t M, uint32_t K>
inline InputLoc wmma_a_input_loc(uint32_t row, uint32_t k, uint32_t a_bits, uint32_t b_bits) {
  if (M == 32 && K == 128 && a_bits == 4 && b_bits == 4) {
    return wmma_f4_32x16x128_a_input_loc(row, k);
  }
  return wmma_f8f6f4_input_loc<M, K>(row, k, a_bits, a_bits < 8 && b_bits == 8);
}

template <uint32_t N, uint32_t K>
inline InputLoc wmma_b_input_loc(uint32_t col, uint32_t k, uint32_t a_bits, uint32_t b_bits) {
  return wmma_f8f6f4_input_loc<N, K>(col, k, b_bits, b_bits < 8 && a_bits == 8);
}

/// Compute the CDNA5 block-scaled WMMA input layout. Unlike the same-format ordinary
/// F8F6F4 WMMA layout, each operand format stores contiguous K ranges in each
/// lane group: 16 values for 8-bit inputs and 32 values for 4/6-bit inputs.
template <uint32_t dim, uint32_t K>
inline InputLoc wmma_block_scaled_input_loc(uint32_t index, uint32_t k, uint32_t data_bits) {
  assert(dim == 16 && K == 128 && (data_bits == 4 || data_bits == 6 || data_bits == 8) &&
         "unsupported CDNA5 block-scaled WMMA input shape");
  const uint32_t block_elems = data_bits == 8 ? 16u : 32u;
  const uint32_t lane = index + 16u * ((k / block_elems) & 1u);
  const uint32_t local = (k / (2u * block_elems)) * block_elems + (k % block_elems);
  return wmma_packed_input_loc(lane, local, data_bits);
}

template <uint32_t M, uint32_t N> inline OutputLoc wmma_output_loc_32(uint32_t row, uint32_t col) {
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

template <uint32_t M, uint32_t N>
inline OutputLoc gfx12_wmma_output_loc_32(uint32_t wave_size, uint32_t row, uint32_t col) {
  require_gfx12_wmma_wave_size(wave_size);
  if (wave_size == WMMA_WAVE32) {
    return wmma_output_loc_32<M, N>(row, col);
  }
  assert(M == 16 && N == 16 && "unsupported gfx12 wave64 WMMA output layout");
  const uint32_t lane = col + 16u * ((row >> 3) & 1u) + 32u * ((row >> 2) & 1u);
  return {row & 3u, lane};
}

template <uint32_t M, uint32_t N>
inline OutputLoc gfx11_wmma_output_loc_32(uint32_t wave_size, uint32_t row, uint32_t col) {
  (void)M;
  uint32_t rows_per_reg = wave_size / N;
  uint32_t lane = (row % rows_per_reg) * N + col;
  uint32_t reg = row / rows_per_reg;
  return {reg, lane};
}

template <uint32_t M, uint32_t N>
inline PackedOutputLoc wmma_output_loc_16(uint32_t row, uint32_t col) {
  uint32_t elems_per_lane = (M * N) / WMMA_WAVE32;
  uint32_t lane = (row / elems_per_lane) * N + col;
  uint32_t elem = row % elems_per_lane;
  return {elem / 2, lane, elem % 2};
}

template <uint32_t M, uint32_t N>
inline PackedOutputLoc gfx12_wmma_output_loc_16(uint32_t wave_size, uint32_t row, uint32_t col) {
  require_gfx12_wmma_wave_size(wave_size);
  if (wave_size == WMMA_WAVE32) {
    return wmma_output_loc_16<M, N>(row, col);
  }
  assert(M == 16 && N == 16 && "unsupported gfx12 wave64 WMMA output layout");
  const uint32_t lane = col + 16u * ((row >> 3) & 1u) + 32u * ((row >> 2) & 1u);
  const uint32_t elem = row & 3u;
  return {elem / 2u, lane, elem & 1u};
}

struct SwmmacIndexLoc {
  uint32_t lane;
  uint32_t local_compressed_k;
};

template <uint32_t M, uint32_t K>
inline SwmmacIndexLoc swmmac_index_loc(uint32_t elem_bits, uint32_t row, uint32_t compressed_k,
                                       uint32_t index_entries) {
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
  if (M == 16 && K == 64 && elem_bits == 4 && index_entries == 16) {
    return {row + 16u * ((compressed_k / 8u) & 1u), 8u * (compressed_k / 16u) + compressed_k % 8u};
  }
  // This generic routing is also intentional for gfx1250 K=128 8-bit
  // SWMMAC. Hardware-reference Tensile kernels require contiguous 32-entry
  // selector blocks even though sparse A changes lane halves every 16 packed
  // K elements. That differs from the association described by the public
  // CDNA5 ISA Sections 7.12.3 and 7.12.5; retain the validated behavior
  // pending specification clarification.
  return {row + (compressed_k / index_entries) * M, compressed_k % index_entries};
}

template <uint32_t M, uint32_t K>
inline SwmmacIndexLoc swmmac_index_loc(uint32_t wave_size, uint32_t elem_bits, uint32_t row,
                                       uint32_t compressed_k, uint32_t index_entries) {
  require_gfx12_wmma_wave_size(wave_size);
  if (wave_size == WMMA_WAVE32) {
    return swmmac_index_loc<M, K>(elem_bits, row, compressed_k, index_entries);
  }
  if (M == 16 && elem_bits == 4 && index_entries == 16 && (K == 32 || K == 64)) {
    const uint32_t block = compressed_k / 8u;
    const uint32_t lane_block = K == 32 ? block : ((block & 1u) << 1u) | (block >> 1u);
    return {row + 16u * lane_block, compressed_k & 7u};
  }
  assert(M == 16 && K == 32 && index_entries == 16 &&
         "unsupported gfx12 wave64 SWMMAC index layout");
  const uint32_t group = compressed_k / 2u;
  const uint32_t slot = compressed_k & 1u;
  const uint32_t block =
      (elem_bits <= 8) ? (2u * ((group >> 1) & 1u) + (group >> 2)) : (group >> 1);
  return {row + 16u * block, 2u * (group & 1u) + slot};
}

template <uint32_t M, uint32_t K>
[[gnu::always_inline]] inline InputLoc swmmac_a_input_loc(uint32_t row, uint32_t compressed_k,
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
  return wmma_input_loc<M, K / 2>(row, compressed_k, elem_bits);
}

template <uint32_t M, uint32_t K>
[[gnu::always_inline]] inline InputLoc
swmmac_a_input_loc(uint32_t wave_size, uint32_t row, uint32_t compressed_k, uint32_t elem_bits) {
  require_gfx12_wmma_wave_size(wave_size);
  if (wave_size == WMMA_WAVE32) {
    return swmmac_a_input_loc<M, K>(row, compressed_k, elem_bits);
  }
  if (M == 16 && elem_bits == 4 && (K == 32 || K == 64)) {
    const uint32_t block = compressed_k / 8u;
    const uint32_t lane_block = K == 32 ? block : ((block & 1u) << 1u) | (block >> 1u);
    return wmma_packed_input_loc(row + 16u * lane_block, compressed_k & 7u, elem_bits);
  }
  assert(M == 16 && K == 32 && (elem_bits == 8 || elem_bits >= 16) &&
         "unsupported gfx12 wave64 SWMMAC A layout");
  const uint32_t group = compressed_k / 2u;
  const uint32_t slot = compressed_k & 1u;
  if (elem_bits == 8) {
    const uint32_t block = 2u * ((group >> 1) & 1u) + (group >> 2);
    return wmma_packed_input_loc(row + 16u * block, 2u * (group & 1u) + slot, elem_bits);
  }
  return wmma_packed_input_loc(row + 16u * (group >> 1), 2u * (group & 1u) + slot, elem_bits);
}

template <uint32_t N, uint32_t K>
[[gnu::always_inline]] inline InputLoc swmmac_b_input_loc(uint32_t col, uint32_t dense_k,
                                                          uint32_t elem_bits) {
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
    if (elem_bits == 4 || elem_bits == 8) {
      return wmma_packed_input_loc(col + 16u * (dense_k / 16u), dense_k % 16u, elem_bits);
    }
    if (elem_bits >= 16) {
      const uint32_t lane = col + 16u * ((dense_k / 8u) & 1u);
      const uint32_t slot = 8u * (dense_k / 16u) + 2u * ((dense_k / 2u) & 3u) + (dense_k & 1u);
      return wmma_packed_input_loc(lane, slot, elem_bits);
    }
  }
  return wmma_input_loc<N, K>(col, dense_k, elem_bits);
}

template <uint32_t N, uint32_t K>
[[gnu::always_inline]] inline InputLoc swmmac_b_input_loc(uint32_t wave_size, uint32_t col,
                                                          uint32_t dense_k, uint32_t elem_bits) {
  require_gfx12_wmma_wave_size(wave_size);
  if (wave_size == WMMA_WAVE32) {
    return swmmac_b_input_loc<N, K>(col, dense_k, elem_bits);
  }
  if (N == 16 && elem_bits == 4 && (K == 32 || K == 64)) {
    const uint32_t values_per_lane = K == 32 ? 8u : 16u;
    const uint32_t block = dense_k / values_per_lane;
    const uint32_t lane_block = ((block & 1u) << 1u) | (block >> 1u);
    return wmma_packed_input_loc(col + 16u * lane_block, dense_k % values_per_lane, elem_bits);
  }
  assert(N == 16 && K == 32 && (elem_bits == 8 || elem_bits >= 16) &&
         "unsupported gfx12 wave64 SWMMAC B layout");
  if (elem_bits == 8) {
    const uint32_t lane = col + 32u * ((dense_k >> 3) & 1u) + 16u * (dense_k >> 4);
    const uint32_t slot = 4u * ((dense_k >> 2) & 1u) + (dense_k & 3u);
    return wmma_packed_input_loc(lane, slot, elem_bits);
  }
  return wmma_packed_input_loc(col + 16u * (dense_k / 8u), dense_k % 8u, elem_bits);
}

// ---------------------------------------------------------------------------
// Lane permutation for cbsz/abid (A broadcast) and blgp (B permutation)
// ---------------------------------------------------------------------------

/// AMD's matrix calculator at 2ef91896 applies adjacent M-lane block groups
/// to gfx908, gfx90a and gfx942. CDNA2 (gfx90a) follows that mapping via
/// matrix_block_lanes = M. Calibrated batched gfx942 F16 and BF16 MFMA uses
/// rj_arith_gfx942_batched_mfma_a_lane (and single-batch gfx942 F16, BF16,
/// XF32, and FNUZ FP8/BF8 read all 64 lanes without permutation). gfx908,
/// gfx950, and gfx942 F32/I8 MFMA retain the full-wave formula in MI300
/// §7.1.5.1 and CDNA4 §7.1.6.1 pending a hardware comparison.
inline uint32_t permute_a_lane(uint32_t lane, uint32_t cbsz, uint32_t abid,
                               uint32_t matrix_block_lanes = 0) {
  if (cbsz == 0 || cbsz > 4) {
    return lane;
  }
  const uint32_t blocks = 1u << cbsz;
  const uint32_t width = matrix_block_lanes ? matrix_block_lanes : 64u / blocks;
  if (width > 64u / blocks) {
    return lane;
  }
  const uint32_t group_size = width * blocks;
  return (lane / group_size) * group_size + (abid % blocks) * width + (lane % width);
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

inline uint32_t wmma_scale_lane(uint32_t index, uint32_t scale_select) {
  return index + ((scale_select & 0x1u) ? 16u : 0u);
}

inline uint32_t wmma_f4_32x16x128_a_scale_lane(uint32_t row) { return row; }

template <uint32_t M, uint32_t K>
inline uint32_t wmma_a_scale_lane(uint32_t row, uint32_t scale_select, uint32_t a_bits,
                                  uint32_t b_bits) {
  if (M == 32 && K == 128 && a_bits == 4 && b_bits == 4) {
    return wmma_f4_32x16x128_a_scale_lane(row);
  }
  return wmma_scale_lane(row, scale_select);
}

inline uint32_t wmma_f8f6f4_scale_byte(uint32_t k, uint32_t data_bits, bool mixed_pair,
                                       bool scale16) {
  if (scale16) {
    if (mixed_pair || data_bits == 8) {
      return 2u * (k >> 5) + ((k >> 2) & 1u);
    }
    return 4u * (k >> 6) + 2u * ((k >> 2) & 1u) + ((k >> 5) & 1u);
  }
  if (mixed_pair || data_bits == 8) {
    return k >> 5;
  }
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
  if (wf_size >= 64) {
    return loc;
  }
  const uint32_t stride = 64 / wf_size;
  return {loc.vgpr_offset * stride + loc.lane / wf_size, loc.lane % wf_size, loc.sub_element,
          loc.bit_offset, loc.data_bits};
}

/// Adjust a GFX9 OutputLoc for wave32 physical VGPR addressing.
/// No-op when wf_size >= 64.
inline OutputLoc physicalize_out(const OutputLoc &loc, uint32_t wf_size) {
  if (wf_size >= 64) {
    return loc;
  }
  const uint32_t stride = 64 / wf_size;
  return {loc.reg * stride + loc.lane / wf_size, loc.lane % wf_size};
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

/// Lanes that an operand read through @p permute uses in each register of a layout that uses
/// every lane of every register.
inline uint64_t mfma_permuted_lane_mask(auto permute) {
  uint64_t lanes = 0;
  for (uint32_t lane = 0; lane < 64; ++lane) {
    lanes |= uint64_t{1} << permute(lane);
  }
  return lanes;
}

inline uint8_t mfma_inline_scale_e8m0(uint32_t selector) {
  switch (selector) {
  case 240: // +0.5f
  case 241: // -0.5f
    return 126;
  case 242: // +1.0f
  case 243: // -1.0f
    return 127;
  case 244: // +2.0f
  case 245: // -2.0f
    return 128;
  case 246: // +4.0f
  case 247: // -4.0f
    return 129;
  case 248: // 1/(2*pi)
    return 124;
  default:
    return 0;
  }
}

inline uint8_t read_mfma_scale_e8m0(auto &cu, uint32_t vgpr_base, uint32_t selector, uint32_t lane,
                                    uint32_t byte_index) {
  if (selector >= 240 && selector <= 248) {
    return mfma_inline_scale_e8m0(selector);
  }
  uint8_t byte_mask = static_cast<uint8_t>(1u << byte_index);
  uint32_t raw = RegisterAccess(cu).read_vgpr(src_base(vgpr_base, selector), lane, byte_mask);
  return static_cast<uint8_t>((raw >> (byte_index * 8)) & 0xffu);
}

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_EXEC_H_
