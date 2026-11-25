// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_EXEC_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_EXEC_H_

/// @file Shared matrix register mapping and remaining legacy execution.
/// Numerical kernels remain for CDNA1/2, gfx1150, and gfx1251 FP64 WMMA.
/// Migrated targets use rj-arith; their VM bridges still share these layouts.
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

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/isa/arch/amdgpu/shared/accvgpr_layout.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fp_mode.h"
#include "rocjitsu/isa/arch/amdgpu/shared/gfx11_dot2.h"
#include "rocjitsu/isa/isa_family.h"
#include "rocjitsu/result.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "util/data_types.h"
#include "util/host_target.h"
#include "util/log.h"
#include "util/meta_programming.h"
#include "util/simd.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <limits>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace amdgpu {
// MFMA register mapping, element extraction, and execution functions.

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
inline bool mfma_has_acc_register(uint32_t const_acc, uint32_t base) {
  return const_acc == ACC_FROM_VGPR && base != NO_ACC_REGISTER;
}

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

inline uint32_t input_local_element(const InputLoc &loc) {
  return (loc.vgpr_offset * 32u + loc.bit_offset) / loc.data_bits;
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

/// Whether output_loc_32 maps the M x N x B outputs of a wave64 MFMA one-to-one
/// onto every word of the first M*N*B/64 VGPRs.
template <uint32_t M, uint32_t N, uint32_t B>
consteval bool output_loc_32_fills_wave64_registers() {
  constexpr uint32_t words = M * N * B;
  if constexpr (words % 64 != 0) {
    return false;
  } else {
    std::array<bool, words> written{};
    for (uint32_t b = 0; b < B; ++b) {
      for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t col = 0; col < N; ++col) {
          const OutputLoc out = output_loc_32(M, N, row, col, b);
          const uint32_t word = out.reg * 64 + out.lane;
          if (out.lane >= 64 || word >= words || written[word]) {
            return false;
          }
          written[word] = true;
        }
      }
    }
    return true;
  }
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

inline void require_wmma_wave32([[maybe_unused]] const auto &cu) {
  assert(cu.wf_size() == WMMA_WAVE32 && "gfx1250 WMMA requires wave32");
}

inline void require_gfx11_wmma_wave_size([[maybe_unused]] uint32_t wave_size) {
  assert((wave_size == WMMA_WAVE32 || wave_size == WMMA_WAVE64) &&
         "gfx11 WMMA requires wave32 or wave64");
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

template <uint32_t M, uint32_t K>
inline InputLoc wmma_block_scaled_a_input_loc(uint32_t row, uint32_t k, uint32_t a_bits) {
  if (M == 32 && K == 128 && a_bits == 4) {
    auto loc = wmma_block_scaled_input_loc<16, K>(row % 16, k, a_bits);
    loc.vgpr_offset += 8u * (row / 16u);
    return loc;
  }
  return wmma_block_scaled_input_loc<M, K>(row, k, a_bits);
}

template <uint32_t N, uint32_t K>
inline InputLoc wmma_block_scaled_b_input_loc(uint32_t col, uint32_t k, uint32_t b_bits) {
  return wmma_block_scaled_input_loc<N, K>(col, k, b_bits);
}

template <uint32_t M, uint32_t K>
inline InputLoc gfx12_wmma_a_input_loc(uint32_t wave_size, uint32_t row, uint32_t k,
                                       uint32_t a_bits, uint32_t b_bits) {
  if (wave_size == WMMA_WAVE32) {
    return wmma_a_input_loc<M, K>(row, k, a_bits, b_bits);
  }
  return gfx12_wmma_input_loc<M, K>(wave_size, row, k, a_bits);
}

template <uint32_t N, uint32_t K>
inline InputLoc gfx12_wmma_b_input_loc(uint32_t wave_size, uint32_t col, uint32_t k,
                                       uint32_t a_bits, uint32_t b_bits) {
  if (wave_size == WMMA_WAVE32) {
    return wmma_b_input_loc<N, K>(col, k, a_bits, b_bits);
  }
  return gfx12_wmma_input_loc<N, K>(wave_size, col, k, b_bits);
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

/// Map one logical F64 CDNA5 WMMA output to its pair of consecutive VGPRs.
///
/// The public CDNA5 ISA reference, section 7.12.2, maps the eight logical
/// 16x16 C/D components per lane with the same row/lane ordering represented
/// by wmma_output_loc_32(). The checked-in FMT_WMMA_DC_16X16_F64 data format
/// then maps each logical F64 component to two consecutive 32-bit VGPRs.
template <uint32_t M, uint32_t N> inline OutputLoc wmma_output_loc_64(uint32_t row, uint32_t col) {
  const auto logical = wmma_output_loc_32<M, N>(row, col);
  return {2u * logical.reg, logical.lane};
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

template <uint32_t M, uint32_t N>
constexpr PackedOutputLoc gfx11_wmma_output_loc_16(uint32_t wave_size, uint32_t row, uint32_t col,
                                                   uint32_t opsel) {
  (void)M;
  uint32_t rows_per_reg = wave_size / N;
  uint32_t lane = (row % rows_per_reg) * N + col;
  uint32_t reg = row / rows_per_reg;
  return {reg, lane, opsel & 0x1u};
}

inline uint64_t read_swmmac_index_set(auto &cu, uint32_t index_base, uint32_t lane,
                                      uint32_t index_entries, uint32_t index_key) {
  // gfx1250 FMT_WMMA_INDEX_SET and FMT_WMMA_INDEX_SET2 contain 16 and 32
  // packed 2-bit entries respectively. LLVM names the matching SWMMAC index
  // selector by entry count, not by total bit count.
  assert((index_entries == 8 || index_entries == 16 || index_entries == 32) &&
         "unsupported SWMMAC sparse index width");
  switch (index_entries) {
  case 8:
    return (RegisterAccess(cu).read_vgpr(index_base, lane) >> (8 * (index_key & 0x3u))) & 0xFFu;
  case 16:
    if (index_key & 0x1u) {
      return (RegisterAccess(cu).read_vgpr(index_base, lane) >> 16) & 0xFFFFu;
    }
    return RegisterAccess(cu).read_vgpr(index_base, lane);
  default: {
    if (index_key & 0x1u) {
      return RegisterAccess(cu).read_vgpr(index_base + 1, lane);
    }
    uint64_t lo = RegisterAccess(cu).read_vgpr(index_base, lane);
    uint64_t hi = RegisterAccess(cu).read_vgpr(index_base + 1, lane);
    return lo | (hi << 32);
  }
  }
}

// Return the selected RDNA4 sparse index set in the low bits of one VGPR word.
// Selector bits above the number of available sets are ignored.
template <uint32_t InBits, uint32_t K>
inline uint32_t read_gfx12_swmmac_index_set(auto &cu, uint32_t wave_size, uint32_t index_base,
                                            uint32_t lane, uint32_t index_key) {
  static_assert((K == 32 && (InBits == 4 || InBits == 8 || InBits == 16)) ||
                (K == 64 && InBits == 4));
  uint32_t set_bits;
  if constexpr (K == 64) {
    set_bits = wave_size == 32 ? 32 : 16;
  } else {
    set_bits = (InBits == 4 || wave_size == 32) ? 16 : 8;
  }
  const uint32_t key = index_key & (32 / set_bits - 1);
  return RegisterAccess(cu).read_vgpr(index_base, lane) >> (key * set_bits);
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
RJ_ALWAYS_INLINE inline InputLoc swmmac_a_input_loc(uint32_t row, uint32_t compressed_k,
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
RJ_ALWAYS_INLINE inline InputLoc swmmac_a_input_loc(uint32_t wave_size, uint32_t row,
                                                    uint32_t compressed_k, uint32_t elem_bits) {
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
RJ_ALWAYS_INLINE inline InputLoc swmmac_b_input_loc(uint32_t col, uint32_t dense_k,
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
RJ_ALWAYS_INLINE inline InputLoc swmmac_b_input_loc(uint32_t wave_size, uint32_t col,
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
/// to gfx908, gfx90a and gfx942. CDNA2 follows that mapping. MI100 and MI200
/// describe the same fields. The gfx908 model retains full-wave broadcast
/// pending a hardware comparison. The gfx942/gfx950 model follows the
/// full-wave formula in MI300 §7.1.5.1 and CDNA4 §7.1.6.1. Both guides'
/// §13.3.6.1 ABID tables describe adjacent groups.
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

// ---------------------------------------------------------------------------
// Element extraction functions
// ---------------------------------------------------------------------------

inline uint32_t packed_mask(uint32_t bits) { return bits >= 32 ? UINT32_MAX : ((1u << bits) - 1u); }

inline uint32_t matrix_vgpr_word(const RegisterAccess::VgprReadRegion &region,
                                 uint32_t physical_reg, uint32_t lane) {
  assert(physical_reg >= region.base() && physical_reg - region.base() < region.reg_count() &&
         "matrix read exceeds acquired VGPR region");
  return region.lane(physical_reg - region.base(), lane);
}

inline uint32_t matrix_vgpr_word(RegisterAccess::VgprReadRegion &region, uint32_t physical_reg,
                                 uint32_t lane) {
  return matrix_vgpr_word(std::as_const(region), physical_reg, lane);
}

/// A copy of the wave64 VGPRs that hold one MFMA operand.
struct MfmaOperandWords {
  static constexpr uint32_t kMaxRegs = 16;
  uint32_t base = 0;
  uint32_t reg_count = 0;
  uint32_t words[kMaxRegs * 64];
};

inline uint32_t matrix_vgpr_word(const MfmaOperandWords &operand, uint32_t physical_reg,
                                 uint32_t lane) {
  assert(physical_reg >= operand.base && physical_reg - operand.base < operand.reg_count &&
         lane < 64 && "MFMA read exceeds the operand copy");
  return operand.words[(physical_reg - operand.base) * 64 + lane];
}

inline uint32_t matrix_vgpr_word(auto &cu, uint32_t physical_reg, uint32_t lane) {
  return RegisterAccess(cu).read_vgpr(physical_reg, lane);
}

inline uint32_t read_packed(auto &cu, uint32_t base, const InputLoc &loc) {
  uint32_t raw = matrix_vgpr_word(cu, base + loc.vgpr_offset, loc.lane) >> loc.bit_offset;
  if (loc.bit_offset + loc.data_bits > 32) {
    uint32_t next = matrix_vgpr_word(cu, base + loc.vgpr_offset + 1, loc.lane);
    raw |= next << (32 - loc.bit_offset);
  }
  return raw & packed_mask(loc.data_bits);
}

struct ExtractF32 {
  float operator()(auto &cu, uint32_t base, const InputLoc &loc) const {
    return std::bit_cast<float>(matrix_vgpr_word(cu, base + loc.vgpr_offset, loc.lane));
  }
};
inline constexpr ExtractF32 extract_f32{};

struct ExtractF16 {
  float operator()(auto &cu, uint32_t base, const InputLoc &loc) const {
    uint32_t raw = matrix_vgpr_word(cu, base + loc.vgpr_offset, loc.lane);
    return util::f16_to_f32(static_cast<uint16_t>((raw >> (loc.sub_element * 16)) & 0xFFFF));
  }
};
inline constexpr ExtractF16 extract_f16{};

struct ExtractBf16 {
  float operator()(auto &cu, uint32_t base, const InputLoc &loc) const {
    uint32_t raw = matrix_vgpr_word(cu, base + loc.vgpr_offset, loc.lane);
    return util::bf16_to_f32(static_cast<uint16_t>((raw >> (loc.sub_element * 16)) & 0xFFFF));
  }
};
inline constexpr ExtractBf16 extract_bf16{};

/// CDNA2 Table 25 fixes F16/BF16 matrix denormals to flush and takes F32
/// input/output control from MODE. Other callers retain their existing modes.
class MfmaF32Denorm {
public:
  uint32_t mode = 3;

  float input(float value, bool widened_f16 = false) const {
    if (mode & 1u) {
      return value;
    }
    if (widened_f16) {
      // The smallest normal F16 is 2^-14. Its subnormals are normal in F32.
      const uint32_t bits = std::bit_cast<uint32_t>(value);
      if ((bits & 0x7fffffffu) < 0x38800000u) {
        return std::bit_cast<float>(bits & 0x80000000u);
      }
      return value;
    }
    return denormal::flush_value(value);
  }

  float output(float value) const { return (mode & 2u) ? value : denormal::flush_value(value); }
};

struct ExtractI8 {
  int32_t operator()(auto &cu, uint32_t base, const InputLoc &loc) const {
    uint32_t raw = matrix_vgpr_word(cu, base + loc.vgpr_offset, loc.lane);
    return static_cast<int32_t>(static_cast<int8_t>((raw >> (loc.sub_element * 8)) & 0xFF));
  }
};
inline constexpr ExtractI8 extract_i8{};

struct ExtractU8 {
  int32_t operator()(auto &cu, uint32_t base, const InputLoc &loc) const {
    uint32_t raw = matrix_vgpr_word(cu, base + loc.vgpr_offset, loc.lane);
    return static_cast<int32_t>((raw >> (loc.sub_element * 8)) & 0xFF);
  }
};
inline constexpr ExtractU8 extract_u8{};

inline int32_t sign_extend_packed(uint32_t value, uint32_t bits) {
  uint32_t sign = 1u << (bits - 1);
  return static_cast<int32_t>((value ^ sign) - sign);
}

struct ExtractI4 {
  int32_t operator()(auto &cu, uint32_t base, const InputLoc &loc) const {
    return sign_extend_packed(read_packed(cu, base, loc), 4);
  }
};
inline constexpr ExtractI4 extract_i4{};

struct ExtractU4 {
  int32_t operator()(auto &cu, uint32_t base, const InputLoc &loc) const {
    return static_cast<int32_t>(read_packed(cu, base, loc));
  }
};
inline constexpr ExtractU4 extract_u4{};

struct ExtractFp8Ocp {
  float operator()(auto &cu, uint32_t base, const InputLoc &loc) const {
    uint32_t raw = matrix_vgpr_word(cu, base + loc.vgpr_offset, loc.lane);
    return util::fp8_e4m3_ocp_to_f32(static_cast<uint8_t>((raw >> (loc.sub_element * 8)) & 0xFF));
  }
};
inline constexpr ExtractFp8Ocp extract_fp8_ocp{};

struct ExtractBf8Ocp {
  float operator()(auto &cu, uint32_t base, const InputLoc &loc) const {
    uint32_t raw = matrix_vgpr_word(cu, base + loc.vgpr_offset, loc.lane);
    return util::bf8_e5m2_ocp_to_f32(static_cast<uint8_t>((raw >> (loc.sub_element * 8)) & 0xFF));
  }
};
inline constexpr ExtractBf8Ocp extract_bf8_ocp{};

struct ExtractFp8Fnuz {
  float operator()(auto &cu, uint32_t base, const InputLoc &loc) const {
    uint32_t raw = matrix_vgpr_word(cu, base + loc.vgpr_offset, loc.lane);
    return util::fp8_e4m3_fnuz_to_f32(static_cast<uint8_t>((raw >> (loc.sub_element * 8)) & 0xFF));
  }
};
inline constexpr ExtractFp8Fnuz extract_fp8_fnuz{};

struct ExtractBf8Fnuz {
  float operator()(auto &cu, uint32_t base, const InputLoc &loc) const {
    uint32_t raw = matrix_vgpr_word(cu, base + loc.vgpr_offset, loc.lane);
    return util::bf8_e5m2_fnuz_to_f32(static_cast<uint8_t>((raw >> (loc.sub_element * 8)) & 0xFF));
  }
};
inline constexpr ExtractBf8Fnuz extract_bf8_fnuz{};

struct ExtractFp8 {
  float operator()(auto &cu, uint32_t base, const InputLoc &loc) const {
    return extract_fp8_ocp(cu, base, loc);
  }
};
inline constexpr ExtractFp8 extract_fp8{};

struct ExtractBf8 {
  float operator()(auto &cu, uint32_t base, const InputLoc &loc) const {
    return extract_bf8_ocp(cu, base, loc);
  }
};
inline constexpr ExtractBf8 extract_bf8{};

struct ExtractFp4 {
  float operator()(auto &cu, uint32_t base, const InputLoc &loc) const {
    return util::fp4_e2m1_to_f32(static_cast<uint8_t>(read_packed(cu, base, loc)));
  }
};
inline constexpr ExtractFp4 extract_fp4{};

struct ExtractFp6 {
  float operator()(auto &cu, uint32_t base, const InputLoc &loc) const {
    return util::fp6_e2m3_to_f32(static_cast<uint8_t>(read_packed(cu, base, loc)));
  }
};
inline constexpr ExtractFp6 extract_fp6{};

struct ExtractBf6 {
  float operator()(auto &cu, uint32_t base, const InputLoc &loc) const {
    return util::bf6_e3m2_to_f32(static_cast<uint8_t>(read_packed(cu, base, loc)));
  }
};
inline constexpr ExtractBf6 extract_bf6{};

struct ExtractF64 {
  double operator()(auto &cu, uint32_t base, const InputLoc &loc) const {
    uint32_t lo = matrix_vgpr_word(cu, base + loc.vgpr_offset, loc.lane);
    uint32_t hi = matrix_vgpr_word(cu, base + loc.vgpr_offset + 1, loc.lane);
    return std::bit_cast<double>(static_cast<uint64_t>(hi) << 32 | lo);
  }
};
inline constexpr ExtractF64 extract_f64{};

inline float decode_e8m0_scale(uint8_t raw) { return util::e8m0_to_f32(raw); }

inline float decode_wmma_scale_byte(uint8_t raw, uint32_t fmt) {
  assert(fmt <= 2 && "unsupported WMMA scale format");
  switch (fmt) {
  case 0:
    return decode_e8m0_scale(raw);
  case 1:
    return util::fp8_e5m3_to_f32(raw);
  default:
    return util::fp8_e4m3_to_f32(raw);
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

inline uint32_t wmma_scale_byte(const InputLoc &loc) {
  return (input_local_element(loc) / 16u) & 0x3u;
}

inline uint32_t wmma_b_fp4_scale_byte(uint32_t k) {
  return ((k / 16u) & 0x1u) | (((k / 64u) & 0x1u) << 1u);
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
template <typename T, uint32_t StaticM = 0, uint32_t StaticN = 0, uint32_t StaticK = 0,
          uint32_t StaticWidth = 0, uint32_t StaticStride = 0>
void wmma_simd_matmul(uint32_t runtime_m, uint32_t runtime_n, uint32_t runtime_k,
                      uint32_t runtime_width, uint32_t runtime_stride, const T *Abuf, const T *Bbuf,
                      T *Cbuf) {
  const uint32_t M = StaticM == 0 ? runtime_m : StaticM;
  const uint32_t N = StaticN == 0 ? runtime_n : StaticN;
  const uint32_t K = StaticK == 0 ? runtime_k : StaticK;
  const uint32_t W = StaticWidth == 0 ? runtime_width : StaticWidth;
  const uint32_t stride = StaticStride == 0 ? runtime_stride : StaticStride;
  for (uint32_t row = 0; row < M; ++row) {
    uint32_t col = 0;
    for (; col + W <= N; col += W) {
      util::native<T> c;
      c.copy_from(&Cbuf[row * stride + col], util::stdx::vector_aligned);
      for (uint32_t k = 0; k < K; ++k) {
        util::native<T> a(Abuf[row * K + k]);
        util::native<T> bv;
        bv.copy_from(&Bbuf[k * stride + col], util::stdx::vector_aligned);
        if constexpr (std::is_floating_point_v<T>) {
          c = util::stdx::fma(a, bv, c);
        } else {
          c += a * bv;
        }
      }
      c.copy_to(&Cbuf[row * stride + col], util::stdx::vector_aligned);
    }
    for (; col < N; ++col) {
      T acc = Cbuf[row * stride + col];
      for (uint32_t k = 0; k < K; ++k) {
        if constexpr (std::is_floating_point_v<T>) {
          acc = std::fma(Abuf[row * K + k], Bbuf[k * stride + col], acc);
        } else {
          // Integer accumulation wraps; do it unsigned so overflow stays defined.
          acc = static_cast<T>(
              static_cast<std::make_unsigned_t<T>>(acc) +
              static_cast<std::make_unsigned_t<T>>(Abuf[row * K + k] * Bbuf[k * stride + col]));
        }
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

/// Adjust a GFX9 PackedOutputLoc for wave32 physical VGPR addressing.
/// No-op when wf_size >= 64.
inline PackedOutputLoc physicalize_packed_out(const PackedOutputLoc &loc, uint32_t wf_size) {
  if (wf_size >= 64) {
    return loc;
  }
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

// Matrix fast-path register access helpers.
//
// MFMA/WMMA specializations below read dense source regions through
// RegisterAccess views. Acquiring a read view observes the whole physical VGPR
// region. Logical traversal and copying remain independent of the backing-store
// layout. Keep the matrix-specific size calculations here so individual
// specializations do not open-code plugin notification or raw region sizing.
//
// This is still only the physical-register-region layer. The broader cleanup is
// to move operand-based SIMD paths onto the same facade and then make direct raw
// VGPR storage unavailable to instruction emulators.

inline void observe_contiguous_vgpr_reads(auto &cu, uint32_t base, uint32_t reg_count,
                                          uint32_t wf_size, uint8_t byte_mask = 0xF) {
  if (reg_count == 0) {
    return;
  }
  (void)RegisterAccess(cu).read_vgpr_region(base, reg_count, mfma_full_lane_mask(wf_size),
                                            byte_mask);
}

inline void observe_mfma_input_reads(auto &cu, uint32_t base, uint32_t dim, uint32_t K, uint32_t B,
                                     uint32_t data_bits, uint32_t wf_size) {
  uint64_t element_count = static_cast<uint64_t>(dim) * K * B;
  observe_contiguous_vgpr_reads(cu, base, mfma_dense_reg_count(element_count, data_bits, wf_size),
                                wf_size);
}

inline void observe_mfma_acc32_reads(auto &cu, uint32_t base, uint32_t M, uint32_t N, uint32_t B,
                                     uint32_t wf_size) {
  uint64_t element_count = static_cast<uint64_t>(M) * N * B;
  observe_contiguous_vgpr_reads(cu, base, mfma_dense_reg_count(element_count, 32, wf_size),
                                wf_size);
}

inline RegisterAccess::VgprWriteRegion write_mfma_acc32_region(auto &cu, uint32_t base, uint32_t M,
                                                               uint32_t N, uint32_t B,
                                                               uint32_t wf_size) {
  RegisterAccess regs(cu);
  uint64_t element_count = static_cast<uint64_t>(M) * N * B;
  uint32_t reg_count = mfma_dense_reg_count(element_count, 32, wf_size);
  return regs.write_vgpr_region(base, reg_count, mfma_full_lane_mask(wf_size));
}

template <typename Destination, typename Convert>
Result convert_packed_matrix_region(const RegisterAccess::VgprReadRegion &source,
                                    Destination *destination, size_t element_count,
                                    uint32_t elements_per_word, Convert convert) {
  const size_t elements_per_reg = static_cast<size_t>(source.wf_size()) * elements_per_word;
  size_t converted = 0;
  Result conversion_result = source.for_each([&](std::span<const uint32_t> lanes) {
    if (converted == element_count) {
      return;
    }
    const size_t count = std::min(elements_per_reg, element_count - converted);
    convert(lanes.data(), destination + converted, count);
    converted += count;
  });
  if (conversion_result.failed()) {
    return conversion_result;
  }
  assert(converted == element_count && "packed conversion exceeds matrix read region");
  return Result::success();
}

template <size_t Capacity>
Result copy_matrix_region_words(const RegisterAccess::VgprReadRegion &source,
                                uint32_t (&destination)[Capacity]) {
  const size_t word_count = static_cast<size_t>(source.reg_count()) * source.wf_size();
  assert(word_count <= Capacity && "matrix read region exceeds snapshot capacity");
  if (word_count > Capacity) {
    std::abort();
  }
  return source.copy_to({destination, word_count});
}

inline Result convert_f16_matrix_region(const RegisterAccess::VgprReadRegion &source,
                                        float *destination, size_t element_count) {
  return convert_packed_matrix_region(source, destination, element_count, /*elements_per_word=*/2,
                                      [](const uint32_t *words, float *out, size_t count) {
                                        util::f16_to_f32_block(
                                            reinterpret_cast<const uint16_t *>(words), out, count);
                                      });
}

inline Result convert_bf16_matrix_region(const RegisterAccess::VgprReadRegion &source,
                                         float *destination, size_t element_count) {
  return convert_packed_matrix_region(source, destination, element_count, /*elements_per_word=*/2,
                                      [](const uint32_t *words, float *out, size_t count) {
                                        util::bf16_to_f32_block(
                                            reinterpret_cast<const uint16_t *>(words), out, count);
                                      });
}

inline Result convert_i8_matrix_region(const RegisterAccess::VgprReadRegion &source,
                                       int32_t *destination, size_t element_count) {
  return convert_packed_matrix_region(source, destination, element_count, /*elements_per_word=*/4,
                                      [](const uint32_t *words, int32_t *out, size_t count) {
                                        util::i8_to_i32_block(
                                            reinterpret_cast<const int8_t *>(words), out, count);
                                      });
}

inline Result convert_u8_matrix_region(const RegisterAccess::VgprReadRegion &source,
                                       int32_t *destination, size_t element_count) {
  return convert_packed_matrix_region(source, destination, element_count, /*elements_per_word=*/4,
                                      [](const uint32_t *words, int32_t *out, size_t count) {
                                        util::u8_to_i32_block(
                                            reinterpret_cast<const uint8_t *>(words), out, count);
                                      });
}

class MatrixReadRegions {
public:
  explicit MatrixReadRegions(RegisterAccess::VgprReadRegion a, RegisterAccess::VgprReadRegion b,
                             std::optional<RegisterAccess::VgprReadRegion> acc)
      : a(std::move(a)), b(std::move(b)), acc(std::move(acc)) {}

  RegisterAccess::VgprReadRegion a;
  RegisterAccess::VgprReadRegion b;
  std::optional<RegisterAccess::VgprReadRegion> acc;
};

/// Whole-register copies of an MFMA's A, B and accumulator operands. Each register is read as
/// its own region, as the generic path reads it, so a register outside the wave's allocation
/// reads as zero without affecting the registers that are inside it. The A and B reads report
/// only the lanes the instruction uses, since a broadcast uses part of each register.
class MfmaOperandSnapshots {
public:
  template <typename ComputeUnit>
  MfmaOperandSnapshots(ComputeUnit &cu, uint32_t s0, uint32_t a_regs, uint64_t a_lanes, uint32_t s1,
                       uint32_t b_regs, uint64_t b_lanes, uint32_t s2, uint32_t acc_regs) {
    RegisterAccess regs(cu);
    failed_ = capture(regs, s0, a_regs, a_lanes, a_).failed() ||
              capture(regs, s1, b_regs, b_lanes, b_).failed() ||
              capture(regs, s2, acc_regs, mfma_full_lane_mask(64), acc_).failed();
  }

  [[nodiscard]] bool failed() const { return failed_; }
  [[nodiscard]] const MfmaOperandWords &a() const { return a_; }
  [[nodiscard]] const MfmaOperandWords &b() const { return b_; }
  [[nodiscard]] const MfmaOperandWords &acc() const { return acc_; }

private:
  static Result capture(RegisterAccess &regs, uint32_t base, uint32_t reg_count, uint64_t lanes,
                        MfmaOperandWords &operand) {
    assert(reg_count <= MfmaOperandWords::kMaxRegs && "MFMA operand exceeds its copy");
    operand.base = base;
    operand.reg_count = reg_count;
    // The copy holds every lane; the mask only limits what observers are told was read.
    for (uint32_t reg = 0; reg < reg_count; ++reg) {
      if (regs.read_vgpr_region(base + reg, 1, lanes)
              .copy_to({operand.words + reg * 64, 64})
              .failed()) {
        return Result::failure();
      }
    }
    return Result::success();
  }

  MfmaOperandWords a_;
  MfmaOperandWords b_;
  MfmaOperandWords acc_;
  bool failed_ = false;
};

/// Wave64 MFMA result registers, staged so that every operand is read before any result is
/// written. The caller sets every word of each register.
class MfmaResultWords {
public:
  static constexpr uint32_t kMaxRegs = MfmaOperandWords::kMaxRegs;

  explicit MfmaResultWords(uint32_t reg_count) : reg_count_(reg_count) {
    assert(reg_count <= kMaxRegs && "MFMA result exceeds the staging area");
  }

  void set(uint32_t reg, uint32_t lane, uint32_t word) {
    assert(reg < reg_count_ && lane < 64 && "MFMA result outside the staging area");
    words_[reg * 64 + lane] = word;
  }

  /// Writes each register as its own region, as the generic path writes it, so a register
  /// outside the wave's allocation is skipped without affecting the others.
  void publish(auto &cu, uint32_t dst) const {
    RegisterAccess regs(cu);
    for (uint32_t reg = 0; reg < reg_count_; ++reg) {
      regs.write_vgpr_region(dst + reg, 1, mfma_full_lane_mask(64))
          .set_lanes(0, std::span<const uint32_t>(words_ + reg * 64, 64));
    }
  }

private:
  uint32_t reg_count_;
  uint32_t words_[kMaxRegs * 64];
};

/// Staging capacity of try_exec_mfma_snapshot, in elements of T. The A, B and C buffers have
/// fixed byte sizes so that, with the operand and result copies, the frame stays within the
/// 48 KiB budget of the other matrix staging paths.
template <typename T> struct MfmaSnapshotStaging {
  static constexpr size_t kMaxA = 8 * 1024 / sizeof(T);
  static constexpr size_t kMaxB = 8 * 1024 / sizeof(T);
  static constexpr size_t kMaxC = 4 * 1024 / sizeof(T);
  static_assert((kMaxA + kMaxB + kMaxC) * sizeof(T) + sizeof(MfmaResultWords) +
                        sizeof(MfmaOperandSnapshots) <=
                    48 * 1024,
                "MFMA register-snapshot staging exceeds the 48 KiB stack budget");
};

/// Whether try_exec_mfma_snapshot handles a wave64 MFMA shape; other shapes use the generic
/// handler. The snapshot copies and publishes whole registers, so the operand and result
/// layouts must use every lane of every register they touch, and the copies and staging
/// buffers must hold them.
template <typename T>
bool mfma_snapshot_shape_fits(uint32_t M, uint32_t N, uint32_t K, uint32_t B, uint32_t a_bits,
                              uint32_t b_bits) {
  if constexpr (!util::has_stdx_simd) {
    return false;
  } else {
    using Staging = MfmaSnapshotStaging<T>;
    constexpr uint32_t W = static_cast<uint32_t>(util::native<T>::size());
    constexpr uint32_t kMaxRegs = MfmaOperandWords::kMaxRegs;
    if (!mfma_input_layout_is_dense(M, K, B, a_bits) ||
        !mfma_input_layout_is_dense(N, K, B, b_bits) ||
        !mfma_output_layout_is_dense(M, N, B, sizeof(T) / sizeof(uint32_t))) {
      return false;
    }
    const uint32_t stride = ((N + W - 1) / W) * W;
    return mfma_dense_reg_count(uint64_t{M} * K * B, a_bits, 64) <= kMaxRegs &&
           mfma_dense_reg_count(uint64_t{N} * K * B, b_bits, 64) <= kMaxRegs &&
           mfma_dense_reg_count(uint64_t{M} * N * B, sizeof(T) * 8, 64) <= kMaxRegs &&
           size_t{M} * K <= Staging::kMaxA && size_t{K} * stride <= Staging::kMaxB &&
           size_t{M} * stride <= Staging::kMaxC;
  }
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

template <typename Cu> bool force_scalar(const Cu &cu) {
  if constexpr (requires { cu.force_scalar(); }) {
    return cu.force_scalar();
  }
  return util::force_scalar();
}

/// Wave64 MFMA from register snapshots, published by register. Returns false without reading
/// registers when the shape or wave size needs the generic handler. Callers with fixed input
/// widths pass them as std::integral_constant so the register layout is known at compile time.
template <typename T, typename ExtractA, typename ExtractB>
bool try_exec_mfma_snapshot(auto &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t B, auto a_bits,
                            auto b_bits, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                            ExtractA ea, ExtractB eb, uint32_t const_acc, uint32_t cbsz = 0,
                            uint32_t abid = 0, uint32_t blgp = 0, uint32_t neg = 0,
                            MfmaF32Denorm denorm = {}) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double> ||
                std::is_same_v<T, int32_t>);
  if constexpr (!util::has_stdx_simd) {
    return false;
  } else {
    constexpr uint32_t W = static_cast<uint32_t>(util::native<T>::size());
    constexpr uint32_t kWordsPerElement = sizeof(T) / sizeof(uint32_t);
    using Staging = MfmaSnapshotStaging<T>;
    if (force_scalar(cu) || cu.wf_size() != 64 ||
        !mfma_snapshot_shape_fits<T>(M, N, K, B, a_bits, b_bits)) {
      return false;
    }
    const uint32_t stride = ((N + W - 1) / W) * W;
    const uint32_t d_regs = mfma_dense_reg_count(uint64_t{M} * N * B, sizeof(T) * 8, 64);
    const uint32_t a_block_lanes = cu.arch() == rocjitsu::IsaFamily::Cdna2 ? M : 0u;
    // cbsz/abid and blgp can broadcast part of each register.
    const uint64_t a_lanes =
        cbsz == 0 ? mfma_full_lane_mask(64) : mfma_permuted_lane_mask([&](uint32_t lane) {
          return permute_a_lane(lane, cbsz, abid, a_block_lanes);
        });
    const uint64_t b_lanes =
        blgp == 0
            ? mfma_full_lane_mask(64)
            : mfma_permuted_lane_mask([&](uint32_t lane) { return permute_b_lane(lane, blgp); });
    const MfmaOperandSnapshots snapshots(
        cu, s0, mfma_dense_reg_count(uint64_t{M} * K * B, a_bits, 64), a_lanes, s1,
        mfma_dense_reg_count(uint64_t{N} * K * B, b_bits, 64), b_lanes, s2,
        mfma_has_acc_register(const_acc, s2) ? d_regs : 0);
    // A reported read failure ends the instruction; it must not retry through the generic path.
    if (snapshots.failed()) {
      return true;
    }
    auto output = [=](uint32_t row, uint32_t col, uint32_t batch) {
      if constexpr (kWordsPerElement == 2) {
        return output_loc_64(M, N, row, col, batch);
      } else {
        return output_loc_32(M, N, row, col, batch);
      }
    };
    auto apply_neg = [=](T value, uint32_t bit) {
      if constexpr (std::is_same_v<T, double>) {
        return (neg & bit)
                   ? std::bit_cast<double>(std::bit_cast<uint64_t>(value) ^ (uint64_t{1} << 63))
                   : value;
      } else {
        return value;
      }
    };
    auto input = [&](T value, bool widened_f16 = false) -> T {
      if constexpr (std::is_same_v<T, float>) {
        return denorm.input(value, widened_f16);
      }
      return value;
    };
    MfmaResultWords results(d_regs);
    alignas(64) T Abuf[Staging::kMaxA] = {};
    alignas(64) T Bbuf[Staging::kMaxB] = {};
    alignas(64) T Cbuf[Staging::kMaxC] = {};
    bool has_nan_or_inf = false;
    for (uint32_t b = 0; b < B; ++b) {
      for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t k = 0; k < K; ++k) {
          auto al = input_loc(M, K, B, row, k, b, a_bits);
          if (cbsz != 0) {
            al.lane = permute_a_lane(al.lane, cbsz, abid, a_block_lanes);
          }
          Abuf[size_t{row} * K + k] = input(apply_neg(ea(snapshots.a(), s0, al), 0x1u),
                                            std::is_same_v<ExtractA, ExtractF16>);
        }
      }
      for (uint32_t k = 0; k < K; ++k) {
        for (uint32_t col = 0; col < N; ++col) {
          auto bl = input_loc(N, K, B, col, k, b, b_bits);
          if (blgp != 0) {
            bl.lane = permute_b_lane(bl.lane, blgp);
          }
          Bbuf[size_t{k} * stride + col] = input(apply_neg(eb(snapshots.b(), s1, bl), 0x2u),
                                                 std::is_same_v<ExtractB, ExtractF16>);
        }
      }
      for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t col = 0; col < N; ++col) {
          const auto out = output(row, col, b);
          T acc;
          if (!mfma_has_acc_register(const_acc, s2)) {
            if constexpr (std::is_same_v<T, double>) {
              acc = static_cast<double>(std::bit_cast<float>(const_acc));
            } else {
              acc = std::bit_cast<T>(const_acc);
            }
          } else if constexpr (kWordsPerElement == 2) {
            const uint64_t lo = matrix_vgpr_word(snapshots.acc(), s2 + out.reg, out.lane);
            const uint64_t hi = matrix_vgpr_word(snapshots.acc(), s2 + out.reg + 1, out.lane);
            acc = std::bit_cast<T>(hi << 32 | lo);
          } else {
            acc = std::bit_cast<T>(matrix_vgpr_word(snapshots.acc(), s2 + out.reg, out.lane));
          }
          Cbuf[row * stride + col] = input(apply_neg(acc, 0x4u));
        }
      }
      wmma_simd_matmul<T>(M, N, K, W, stride, Abuf, Bbuf, Cbuf);
      // Publish only after every batch has consumed its inputs, which may overlap the results.
      for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t col = 0; col < N; ++col) {
          const auto out = output(row, col, b);
          T value = Cbuf[row * stride + col];
          if constexpr (std::is_same_v<T, float>) {
            value = denorm.output(value);
          }
          if constexpr (kWordsPerElement == 2) {
            const uint64_t bits = std::bit_cast<uint64_t>(value);
            results.set(out.reg, out.lane, static_cast<uint32_t>(bits));
            results.set(out.reg + 1, out.lane, static_cast<uint32_t>(bits >> 32));
          } else {
            results.set(out.reg, out.lane, std::bit_cast<uint32_t>(value));
          }
          if constexpr (std::is_same_v<T, float>) {
            has_nan_or_inf = has_nan_or_inf || std::isnan(value) || std::isinf(value);
          }
        }
      }
    }
    results.publish(cu, dst);
    if (has_nan_or_inf) {
      util::Logger::vm([&](auto &os) {
        os << std::format("MFMA_NAN_DETECTED (regions) dst=v{} s0=v{} s1=v{} s2=v{} M={} N={} K={}",
                          dst, s0, s1, s2, M, N, K);
      });
    }
    return true;
  }
}

inline MatrixReadRegions read_mixed_matrix_fast_path_regions(auto &cu, uint32_t s0, uint32_t s1,
                                                             uint32_t s2, uint64_t a_elements,
                                                             uint32_t a_bits, uint64_t b_elements,
                                                             uint32_t b_bits, uint64_t acc_elements,
                                                             uint32_t acc_bits, uint32_t const_acc,
                                                             uint32_t wf_size) {
  RegisterAccess regs(cu);
  const uint64_t lane_mask = mfma_full_lane_mask(wf_size);
  const uint32_t a_regs = mfma_dense_reg_count(a_elements, a_bits, wf_size);
  const uint32_t b_regs = mfma_dense_reg_count(b_elements, b_bits, wf_size);
  std::optional<RegisterAccess::VgprReadRegion> acc_region;
  if (const_acc == ACC_FROM_VGPR) {
    const uint32_t acc_regs = mfma_dense_reg_count(acc_elements, acc_bits, wf_size);
    acc_region.emplace(regs.read_vgpr_region(s2, acc_regs, lane_mask));
  }
  return MatrixReadRegions(regs.read_vgpr_region(s0, a_regs, lane_mask),
                           regs.read_vgpr_region(s1, b_regs, lane_mask), std::move(acc_region));
}

inline MatrixReadRegions read_mfma_fast_path_regions(auto &cu, uint32_t s0, uint32_t s1,
                                                     uint32_t s2, uint32_t M, uint32_t N,
                                                     uint32_t K, uint32_t B, uint32_t data_bits,
                                                     uint32_t const_acc, uint32_t wf_size) {
  RegisterAccess regs(cu);
  uint64_t lane_mask = mfma_full_lane_mask(wf_size);
  uint32_t a_regs = mfma_dense_reg_count(static_cast<uint64_t>(M) * K * B, data_bits, wf_size);
  uint32_t b_regs = mfma_dense_reg_count(static_cast<uint64_t>(N) * K * B, data_bits, wf_size);
  std::optional<RegisterAccess::VgprReadRegion> acc_region;
  if (mfma_has_acc_register(const_acc, s2)) {
    uint32_t acc_regs = mfma_dense_reg_count(static_cast<uint64_t>(M) * N * B, 32, wf_size);
    acc_region.emplace(regs.read_vgpr_region(s2, acc_regs, lane_mask));
  }
  return MatrixReadRegions(regs.read_vgpr_region(s0, a_regs, lane_mask),
                           regs.read_vgpr_region(s1, b_regs, lane_mask), std::move(acc_region));
}

inline void observe_mfma_fast_path_reads(auto &cu, uint32_t s0, uint32_t s1, uint32_t s2,
                                         uint32_t M, uint32_t N, uint32_t K, uint32_t B,
                                         uint32_t data_bits, uint32_t const_acc, uint32_t wf_size) {
  (void)read_mfma_fast_path_regions(cu, s0, s1, s2, M, N, K, B, data_bits, const_acc, wf_size);
}

template <uint32_t M, uint32_t N, uint32_t K>
inline MatrixReadRegions
read_wmma_fast_path_regions(auto &cu, uint32_t s0, uint32_t s1, uint32_t s2, uint32_t data_bits,
                            uint32_t acc_bits, uint32_t const_acc, uint32_t wf_size) {
  RegisterAccess regs(cu);
  uint64_t lane_mask = mfma_full_lane_mask(wf_size);
  uint32_t a_regs = mfma_dense_reg_count(static_cast<uint64_t>(M) * K, data_bits, wf_size);
  uint32_t b_regs = mfma_dense_reg_count(static_cast<uint64_t>(N) * K, data_bits, wf_size);
  std::optional<RegisterAccess::VgprReadRegion> acc_region;
  if (const_acc == ACC_FROM_VGPR) {
    uint32_t acc_regs = mfma_dense_reg_count(static_cast<uint64_t>(M) * N, acc_bits, wf_size);
    acc_region.emplace(regs.read_vgpr_region(s2, acc_regs, lane_mask));
  }
  return MatrixReadRegions(regs.read_vgpr_region(s0, a_regs, lane_mask),
                           regs.read_vgpr_region(s1, b_regs, lane_mask), std::move(acc_region));
}

/// Map f32 output position to packed 16-bit output position using GFX9 layout.
/// Two consecutive f32 register positions pack into one 32-bit VGPR as two
/// 16-bit sub-elements: reg/2 holds the VGPR offset, reg%2 the sub-element.
inline PackedOutputLoc output_loc_16(uint32_t M, uint32_t N, uint32_t i, uint32_t j, uint32_t b) {
  auto f32 = output_loc_32(M, N, i, j, b);
  return {f32.reg / 2, f32.lane, f32.reg % 2};
}

/// exec_f32_mixed for shapes and modes that try_exec_mfma_snapshot does not handle. Kept out of
/// line so its staging frame is not live while the snapshot path runs.
template <uint32_t M, uint32_t N, uint32_t K, uint32_t B, typename ExtractA, typename ExtractB>
RJ_NOINLINE void exec_f32_mixed_generic(auto &cu, uint32_t a_bits, uint32_t b_bits, uint32_t dst,
                                        uint32_t s0, uint32_t s1, uint32_t s2, ExtractA ea,
                                        ExtractB eb, uint32_t const_acc, uint32_t cbsz,
                                        uint32_t abid, uint32_t blgp, MfmaF32Denorm denorm) {
  static_assert(M > 0 && N > 0 && K > 0);
  const uint32_t a_block_lanes = cu.arch() == rocjitsu::IsaFamily::Cdna2 ? M : 0u;
  const uint32_t wf = cu.wf_size();
  struct Result {
    uint32_t reg;
    uint32_t lane;
    uint32_t val;
  };
  std::vector<Result> results;
  results.reserve(M * N * B);

  constexpr size_t MAX_AB = 2048;      // max M*K over all MFMA shapes
  constexpr size_t MAX_BSTRIDE = 4096; // max K*stride over all MFMA shapes
  constexpr size_t MAX_C = 1024;       // max M*stride over all MFMA shapes
  static_assert((MAX_AB + MAX_BSTRIDE + MAX_C) * sizeof(float) <= 48 * 1024,
                "MFMA staging buffers exceed the 48 KiB stack budget");

  auto stage_operands = [&](uint32_t block, uint32_t b_stride, float *a_values, float *b_values) {
    for (uint32_t row = 0; row < M; ++row) {
      for (uint32_t k = 0; k < K; ++k) {
        auto al = input_loc(M, K, B, row, k, block, a_bits);
        if (cbsz != 0) {
          al.lane = permute_a_lane(al.lane, cbsz, abid, a_block_lanes);
        }
        a_values[static_cast<size_t>(row) * K + k] =
            denorm.input(ea(cu, s0, physicalize_loc(al, wf)), std::is_same_v<ExtractA, ExtractF16>);
      }
    }
    for (uint32_t k = 0; k < K; ++k) {
      for (uint32_t col = 0; col < N; ++col) {
        auto bl = input_loc(N, K, B, col, k, block, b_bits);
        if (blgp != 0) {
          bl.lane = permute_b_lane(bl.lane, blgp);
        }
        b_values[static_cast<size_t>(k) * b_stride + col] =
            denorm.input(eb(cu, s1, physicalize_loc(bl, wf)), std::is_same_v<ExtractB, ExtractF16>);
      }
    }
  };

  // Scalar reference: D[i][j] = C[i][j] + sum_k A[i][k] * B[k][j], accumulated
  // per output in K order (non-fused multiply-add).
  auto run_scalar = [&]() {
    const size_t a_count = static_cast<size_t>(M) * K;
    const size_t b_count = static_cast<size_t>(K) * N;
    // Real ISA shapes use bounded per-block stack storage. Preserve support
    // for direct callers with larger dimensions through one overflow buffer.
    alignas(64) float a_stack[MAX_AB];
    alignas(64) float b_stack[MAX_BSTRIDE];
    std::vector<float> overflow;
    float *a_values = a_stack;
    float *b_values = b_stack;
    if (a_count > MAX_AB || b_count > MAX_BSTRIDE) {
      overflow.resize(a_count + b_count);
      a_values = overflow.data();
      b_values = overflow.data() + a_count;
    }

    for (uint32_t b = 0; b < B; ++b) {
      stage_operands(b, N, a_values, b_values);
      for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t col = 0; col < N; ++col) {
          // AMD convention: i=row (register dimension), j=col (lane dimension).
          auto out = physicalize_out(output_loc_32(M, N, row, col, b), wf);
          float acc =
              (!mfma_has_acc_register(const_acc, s2))
                  ? std::bit_cast<float>(const_acc)
                  : std::bit_cast<float>(RegisterAccess(cu).read_vgpr(s2 + out.reg, out.lane));
          acc = denorm.input(acc);
          const float *a = a_values + static_cast<size_t>(row) * K;
          const float *bv = b_values + col;
          for (uint32_t k = 0; k < K; ++k) {
            acc += a[k] * bv[static_cast<size_t>(k) * N];
          }
          results.push_back({out.reg, out.lane, std::bit_cast<uint32_t>(denorm.output(acc))});
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
    const uint32_t stride = ((N + W - 1) / W) * W;
    if (force_scalar(cu) || static_cast<size_t>(M) * K > MAX_AB ||
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
        stage_operands(b, stride, Abuf, Bbuf);
        for (uint32_t row = 0; row < M; ++row) {
          for (uint32_t col = 0; col < N; ++col) {
            auto out = physicalize_out(output_loc_32(M, N, row, col, b), wf);
            Cbuf[row * stride + col] =
                (!mfma_has_acc_register(const_acc, s2))
                    ? std::bit_cast<float>(const_acc)
                    : std::bit_cast<float>(RegisterAccess(cu).read_vgpr(s2 + out.reg, out.lane));
            Cbuf[row * stride + col] = denorm.input(Cbuf[row * stride + col]);
          }
        }
        wmma_simd_matmul<float>(M, N, K, W, stride, Abuf, Bbuf, Cbuf);
        for (uint32_t row = 0; row < M; ++row) {
          for (uint32_t col = 0; col < N; ++col) {
            auto out = physicalize_out(output_loc_32(M, N, row, col, b), wf);
            results.push_back({out.reg, out.lane,
                               std::bit_cast<uint32_t>(denorm.output(Cbuf[row * stride + col]))});
          }
        }
      }
    }
  } else {
    run_scalar();
  }

  bool has_nan = false;
  for (const auto &r : results) {
    RegisterAccess(cu).write_vgpr(dst + r.reg, r.lane, r.val);
    float fval = std::bit_cast<float>(r.val);
    if (std::isnan(fval) || std::isinf(fval)) {
      has_nan = true;
    }
  }
  // The dumps below show staged results only: a register read here would be reported to
  // execution plugins as an access by the instruction.
  if (has_nan) {
    util::Logger::vm([&](auto &os) {
      os << std::format("MFMA_NAN_DETECTED dst=v{} s0=v{} s1=v{} s2=v{} M={} N={} K={}", dst, s0,
                        s1, s2, M, N, K);
      for (const auto &r : results) {
        float fval = std::bit_cast<float>(r.val);
        if (std::isnan(fval) || std::isinf(fval)) {
          os << std::format("\n[rj log VM]   reg={} lane={} val={:#x}({})", r.reg, r.lane, r.val,
                            fval);
        }
      }
    });
  }
  util::Logger::vm([&](auto &os) {
    static thread_local uint64_t mfma_count = 0;
    if (++mfma_count > 30) {
      return;
    }
    os << std::format("MFMA_F32 #{} M={} N={} K={} B={} dst=v{} s0=v{} s1=v{} s2=v{}", mfma_count,
                      M, N, K, B, dst, s0, s1, s2);
    std::array<uint32_t, 4 * 64> out{};
    for (const auto &r : results) {
      if (r.reg < 4) {
        out[r.reg * 64 + r.lane] = r.val;
      }
    }
    for (uint32_t ln : {0u, 1u, 4u, 8u, 16u, 31u, 32u, 48u, 63u}) {
      if (ln >= wf) {
        break;
      }
      os << std::format("\n[rj log VM]   L{}: out=[{:#x},{:#x},{:#x},{:#x}]", ln, out[ln],
                        out[64 + ln], out[128 + ln], out[192 + ln]);
    }
  });
}

/// Generic MFMA execute for f32 output: D = C + A x B.
///
/// All inputs are read before any outputs are written to avoid WAR hazards
/// when destination registers overlap source registers.
///
/// @param cbsz  A-matrix broadcast block size (0 = no broadcast).
/// @param abid  A-matrix broadcast source block ID.
/// @param blgp  B-matrix lane group permutation pattern.
template <uint32_t M, uint32_t N, uint32_t K, uint32_t B, typename ExtractA, typename ExtractB>
void exec_f32_mixed(auto &cu, uint32_t a_bits, uint32_t b_bits, uint32_t dst, uint32_t s0,
                    uint32_t s1, uint32_t s2, ExtractA ea, ExtractB eb,
                    uint32_t const_acc = ACC_FROM_VGPR, uint32_t cbsz = 0, uint32_t abid = 0,
                    uint32_t blgp = 0, MfmaF32Denorm denorm = {}) {
  if (!try_exec_mfma_snapshot<float>(cu, M, N, K, B, a_bits, b_bits, dst, s0, s1, s2, ea, eb,
                                     const_acc, cbsz, abid, blgp, 0, denorm)) {
    exec_f32_mixed_generic<M, N, K, B>(cu, a_bits, b_bits, dst, s0, s1, s2, ea, eb, const_acc, cbsz,
                                       abid, blgp, denorm);
  }
}

template <uint32_t M, uint32_t N, uint32_t K, uint32_t B, typename ExtractA, typename ExtractB>
void exec_f32(auto &cu, uint32_t in_bits, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
              ExtractA ea, ExtractB eb, uint32_t const_acc = ACC_FROM_VGPR, uint32_t cbsz = 0,
              uint32_t abid = 0, uint32_t blgp = 0, MfmaF32Denorm denorm = {}) {
  exec_f32_mixed<M, N, K, B>(cu, in_bits, in_bits, dst, s0, s1, s2, ea, eb, const_acc, cbsz, abid,
                             blgp, denorm);
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
void exec_i32_mixed(auto &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t B, uint32_t in_bits,
                    uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2, ExtractA ea, ExtractB eb,
                    uint32_t const_acc = ACC_FROM_VGPR, bool clamp = false, uint32_t cbsz = 0,
                    uint32_t abid = 0, uint32_t blgp = 0) {
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
          if (cbsz != 0) {
            al.lane = permute_a_lane(al.lane, cbsz, abid);
          }
          if (blgp != 0) {
            bl.lane = permute_b_lane(bl.lane, blgp);
          }
          return std::pair{ea(cu, s0, physicalize_loc(al, wf)),
                           eb(cu, s1, physicalize_loc(bl, wf))};
        };
        uint32_t val;
        if (clamp) {
          int64_t acc = (const_acc != ACC_FROM_VGPR)
                            ? static_cast<int64_t>(static_cast<int32_t>(const_acc))
                            : static_cast<int64_t>(static_cast<int32_t>(
                                  RegisterAccess(cu).read_vgpr(s2 + out.reg, out.lane)));
          for (uint32_t k = 0; k < K; ++k) {
            auto [a, b_val] = products(k);
            acc += static_cast<int64_t>(a) * static_cast<int64_t>(b_val);
          }
          val = pack_i32_acc(acc, true);
        } else {
          uint32_t acc = (const_acc != ACC_FROM_VGPR)
                             ? const_acc
                             : RegisterAccess(cu).read_vgpr(s2 + out.reg, out.lane);
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
  for (const auto &r : results) {
    RegisterAccess(cu).write_vgpr(dst + r.reg, r.lane, r.val);
  }
}

// GFX11 F16/BF16 WMMA evaluates adjacent pairs as DOT2 in increasing k order,
// starting at 0. Packed results narrow after each pair and preserve the other D half.
// Preserve raw NaN bits.
// Stage the whole result before writing D, which may overlap A, B, or C.
template <bool Bf16, bool Packed = false>
void exec_gfx11_wmma_dot2(auto &cu, uint32_t wave_size, uint32_t dst, uint32_t s0, uint32_t s1,
                          uint32_t s2, std::optional<uint32_t> const_acc, uint32_t neg,
                          uint32_t neg_hi, uint32_t opsel = 0, bool fp16_ovfl = false) {
  require_gfx11_wmma_wave_size(wave_size);
  std::array<uint32_t, 256> results;
  RegisterAccess regs(cu);
  for (uint32_t row = 0; row < 16; ++row) {
    for (uint32_t col = 0; col < 16; ++col) {
      const auto out = gfx11_wmma_output_loc_32<16, 16>(wave_size, row, col);
      const uint32_t lane_group = out.lane / 16;
      uint32_t acc = const_acc ? *const_acc : regs.read_vgpr(s2 + out.reg, out.lane);
      if constexpr (Packed) {
        acc = (acc >> (16 * opsel)) & 0xffff;
      }
      if (neg_hi & 4) {
        acc &= Packed ? 0x7fffu : 0x7fffffffu;
      }
      if (neg & 4) {
        acc ^= Packed ? 0x8000u : 0x80000000u;
      }
      for (uint32_t k = 0; k < 16; k += 2) {
        const auto al = gfx11_wmma_input_loc<16, 16>(row, k, 16, lane_group);
        const auto bl = gfx11_wmma_input_loc<16, 16>(col, k, 16, lane_group);
        uint32_t a = regs.read_vgpr(s0 + al.vgpr_offset, al.lane);
        uint32_t b = regs.read_vgpr(s1 + bl.vgpr_offset, bl.lane);
        a ^= ((neg & 1) << 15) | ((neg_hi & 1) << 31);
        b ^= ((neg & 2) << 14) | ((neg_hi & 2) << 30);
        if constexpr (Packed) {
          acc = gfx11_dot2_packed16<Bf16>(uint16_t(a), uint16_t(b), uint16_t(a >> 16),
                                          uint16_t(b >> 16), uint16_t(acc), fp16_ovfl);
        } else {
          acc = gfx11_dot2_f32<Bf16>(uint16_t(a), uint16_t(b), uint16_t(a >> 16), uint16_t(b >> 16),
                                     acc);
        }
      }
      results[row * 16 + col] = acc;
    }
  }
  for (uint32_t row = 0; row < 16; ++row) {
    for (uint32_t col = 0; col < 16; ++col) {
      const auto out = gfx11_wmma_output_loc_32<16, 16>(wave_size, row, col);
      uint32_t value = results[row * 16 + col];
      if constexpr (Packed) {
        const unsigned shift = 16 * opsel;
        value = (regs.read_vgpr(dst + out.reg, out.lane) & ~(0xffffu << shift)) | (value << shift);
      }
      regs.write_vgpr(dst + out.reg, out.lane, value);
    }
  }
}

/// Return whether an f32-accumulating matrix output row divides into complete
/// native SIMD chunks.
constexpr bool mma_f32_native_width_supported(uint32_t n, uint32_t width) {
  return width > 1 && n % width == 0;
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

/// exec_i32_i8 for shapes and modes that try_exec_mfma_snapshot does not handle. Kept out of line
/// so its staging frame is not live while the snapshot path runs.
RJ_NOINLINE inline void exec_i32_i8_generic(auto &cu, uint32_t M, uint32_t N, uint32_t K,
                                            uint32_t B, uint32_t dst, uint32_t s0, uint32_t s1,
                                            uint32_t s2, uint32_t const_acc, uint32_t cbsz,
                                            uint32_t abid, uint32_t blgp) {
  const uint32_t a_block_lanes = cu.arch() == rocjitsu::IsaFamily::Cdna2 ? M : 0u;
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
          uint32_t acc = (!mfma_has_acc_register(const_acc, s2))
                             ? const_acc
                             : RegisterAccess(cu).read_vgpr(s2 + out.reg, out.lane);
          for (uint32_t k = 0; k < K; ++k) {
            auto al = input_loc(M, K, B, row, k, b, 8);
            auto bl = input_loc(N, K, B, col, k, b, 8);
            if (cbsz != 0) {
              al.lane = permute_a_lane(al.lane, cbsz, abid, a_block_lanes);
            }
            if (blgp != 0) {
              bl.lane = permute_b_lane(bl.lane, blgp);
            }
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
    if (force_scalar(cu) || static_cast<size_t>(M) * K > MAX_AB ||
        static_cast<size_t>(K) * stride > MAX_BSTRIDE || static_cast<size_t>(M) * stride > MAX_C) {
      run_scalar();
    } else {
      // Zero-initialized staging buffers (uniform convention; see
      // wmma_simd_matmul). Cbuf is pre-seeded below.
      alignas(64) int32_t Abuf[MAX_AB] = {};
      alignas(64) int32_t Bbuf[MAX_BSTRIDE] = {};
      alignas(64) int32_t Cbuf[MAX_C] = {};
      for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t row = 0; row < M; ++row) {
          for (uint32_t col = 0; col < N; ++col) {
            auto out = physicalize_out(output_loc_32(M, N, row, col, b), wf);
            Cbuf[row * stride + col] =
                (!mfma_has_acc_register(const_acc, s2))
                    ? static_cast<int32_t>(const_acc)
                    : static_cast<int32_t>(RegisterAccess(cu).read_vgpr(s2 + out.reg, out.lane));
          }
        }
        for (uint32_t row = 0; row < M; ++row) {
          for (uint32_t k = 0; k < K; ++k) {
            auto al = input_loc(M, K, B, row, k, b, 8);
            if (cbsz != 0) {
              al.lane = permute_a_lane(al.lane, cbsz, abid, a_block_lanes);
            }
            Abuf[row * K + k] = extract_i8(cu, s0, physicalize_loc(al, wf));
          }
        }
        for (uint32_t k = 0; k < K; ++k) {
          for (uint32_t col = 0; col < N; ++col) {
            auto bl = input_loc(N, K, B, col, k, b, 8);
            if (blgp != 0) {
              bl.lane = permute_b_lane(bl.lane, blgp);
            }
            Bbuf[k * stride + col] = extract_i8(cu, s1, physicalize_loc(bl, wf));
          }
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
            for (uint32_t k = 0; k < K; ++k) {
              acc += static_cast<uint32_t>(Abuf[row * K + k] * Bbuf[k * stride + col]);
            }
            Cbuf[row * stride + col] = static_cast<int32_t>(acc);
          }
        }
        for (uint32_t row = 0; row < M; ++row) {
          for (uint32_t col = 0; col < N; ++col) {
            auto out = physicalize_out(output_loc_32(M, N, row, col, b), wf);
            results.push_back({out.reg, out.lane, static_cast<uint32_t>(Cbuf[row * stride + col])});
          }
        }
      }
    }
  } else {
    run_scalar();
  }

  for (const auto &r : results) {
    RegisterAccess(cu).write_vgpr(dst + r.reg, r.lane, r.val);
  }
}

/// MFMA execute for i32 output with i8 input: D = C + A x B.
inline void exec_i32_i8(auto &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t B, uint32_t dst,
                        uint32_t s0, uint32_t s1, uint32_t s2, uint32_t const_acc = ACC_FROM_VGPR,
                        uint32_t cbsz = 0, uint32_t abid = 0, uint32_t blgp = 0,
                        bool clamp = false) {
  if (clamp) {
    exec_i32_mixed(cu, M, N, K, B, 8, dst, s0, s1, s2, extract_i8, extract_i8, const_acc, true,
                   cbsz, abid, blgp);
    return;
  }
  constexpr std::integral_constant<uint32_t, 8> kBits;
  if (!try_exec_mfma_snapshot<int32_t>(cu, M, N, K, B, kBits, kBits, dst, s0, s1, s2, extract_i8,
                                       extract_i8, const_acc, cbsz, abid, blgp)) {
    exec_i32_i8_generic(cu, M, N, K, B, dst, s0, s1, s2, const_acc, cbsz, abid, blgp);
  }
}

// Integer WMMA saturates after architecture-specific product groups. Sparse forms
// use compressed K positions; physical_k maps the accumulation order to storage.
// See tests/fixtures/float_dot/README.md for the model's empirical basis.
template <bool Gfx12, uint32_t InBits, uint32_t K, bool Sparse = false> struct RdnaWmmaI32Policy {
  static_assert(InBits == 4 || InBits == 8);
  static_assert(Sparse ? (Gfx12 && (K == 32 || (K == 64 && InBits == 4)))
                       : (K == 16 || (Gfx12 && InBits == 4 && K == 32)));
  static constexpr uint32_t product_count = Sparse ? K / 2 : K;
  static constexpr uint32_t clamp_group = Gfx12 ? product_count / 2 : (InBits == 8 ? 2 : 4);

  // Map accumulation order to the K coordinate used by the register layout.
  static constexpr uint32_t physical_k(uint32_t wave_size, uint32_t k) {
    // In the model, IU4 with 32 A products (dense K32 or sparse K64) in wave32
    // consumes both registers of one lane group before the other lane group.
    if constexpr (Gfx12 && InBits == 4 && product_count == 32) {
      if (wave_size == 32) {
        return (k & 7) | ((k & 8) << 1) | ((k & 16) >> 1);
      }
    }
    if constexpr (Sparse) {
      return k;
    }
    // The dense K16 IU8 wave64 model swaps the four-product blocks.
    if (Gfx12 && InBits == 8 && wave_size == 64) {
      return wmma_swap_k_bits_2_3(k);
    }
    return k;
  }
};

// Compute a 16x16 integer matrix result, saturating at the policy boundaries
// when CLAMP is set. Destination storage may overlap any input, including indices.
template <bool Gfx12, uint32_t InBits, uint32_t K = 16, bool Sparse = false>
void exec_rdna_wmma_i32_impl(auto &cu, uint32_t wave_size, uint32_t dst, uint32_t s0, uint32_t s1,
                             uint32_t s2, std::optional<uint32_t> const_acc, bool a_signed,
                             bool b_signed, bool clamp, uint32_t index_base, uint32_t index_key) {
  using Policy = RdnaWmmaI32Policy<Gfx12, InBits, K, Sparse>;
  constexpr uint32_t kProductCount = Policy::product_count;
  if constexpr (Gfx12) {
    require_gfx12_wmma_wave_size(wave_size);
  } else {
    require_gfx11_wmma_wave_size(wave_size);
  }
  RegisterAccess regs(cu);
  std::array<uint32_t, 256> results;
  auto extract = [&](uint32_t src, InputLoc loc, bool is_signed) -> int64_t {
    if constexpr (InBits == 4) {
      return is_signed ? extract_i4(cu, src, loc) : extract_u4(cu, src, loc);
    } else {
      return is_signed ? extract_i8(cu, src, loc) : extract_u8(cu, src, loc);
    }
  };
  auto output_loc = [&](uint32_t row, uint32_t col) {
    if constexpr (Gfx12) {
      return gfx12_wmma_output_loc_32<16, 16>(wave_size, row, col);
    } else {
      return gfx11_wmma_output_loc_32<16, 16>(wave_size, row, col);
    }
  };
  for (uint32_t row = 0; row < 16; ++row) {
    // Gather one A row and all B columns for scalar or SIMD accumulation.
    // Signed 64-bit accumulators retain overflow until the next clamp boundary.
    alignas(64) int64_t a[kProductCount], b[kProductCount * 16], c[16];
    auto input_loc = [&](uint32_t index, uint32_t k) {
      if constexpr (Sparse) {
        return swmmac_a_input_loc<16, K>(wave_size, index, Policy::physical_k(wave_size, k),
                                         InBits);
      } else if constexpr (Gfx12) {
        return gfx12_wmma_input_loc<16, K>(wave_size, index, Policy::physical_k(wave_size, k),
                                           InBits);
      } else {
        return gfx11_wmma_input_loc<16, K>(index, k, InBits, output_loc(row, 0).lane / 16);
      }
    };
    for (uint32_t k = 0; k < kProductCount; ++k) {
      const auto al = input_loc(row, k);
      a[k] = extract(s0, al, a_signed);
      uint32_t dense_k = k;
      if constexpr (Sparse) {
        const uint32_t physical_k = Policy::physical_k(wave_size, k);
        const auto index = swmmac_index_loc<16, K>(wave_size, InBits, row, physical_k, 16);
        const uint32_t indices = read_gfx12_swmmac_index_set<InBits, K>(cu, wave_size, index_base,
                                                                        index.lane, index_key);
        dense_k = swmmac_dense_k(indices, physical_k, index.local_compressed_k);
      }
      for (uint32_t col = 0; col < 16; ++col) {
        InputLoc bl;
        if constexpr (Sparse) {
          bl = swmmac_b_input_loc<16, K>(wave_size, col, dense_k, InBits);
        } else {
          bl = input_loc(col, k);
        }
        b[k * 16 + col] = extract(s1, bl, b_signed);
      }
    }
    for (uint32_t col = 0; col < 16; ++col) {
      const auto out = output_loc(row, col);
      c[col] =
          std::bit_cast<int32_t>(const_acc ? *const_acc : regs.read_vgpr(s2 + out.reg, out.lane));
    }
    // CLAMP acts between product groups, not just on the final matrix result.
    const uint32_t step = clamp ? Policy::clamp_group : kProductCount;
    for (uint32_t k = 0; k < kProductCount; k += step) {
      auto scalar_group = [&]() {
        for (uint32_t j = k; j < k + step; ++j) {
          for (uint32_t col = 0; col < 16; ++col) {
            c[col] += a[j] * b[j * 16 + col];
          }
        }
      };
      if constexpr (util::has_stdx_simd) {
        if (force_scalar(cu)) {
          scalar_group();
        } else {
          wmma_simd_matmul<int64_t>(1, 16, step, util::native<int64_t>::size(), 16, a + k,
                                    b + k * 16, c);
        }
      } else {
        scalar_group();
      }
      if (clamp) {
        for (auto &value : c) {
          value = std::bit_cast<int32_t>(pack_i32_acc(value, true));
        }
      }
    }
    for (uint32_t col = 0; col < 16; ++col) {
      results[row * 16 + col] = static_cast<uint32_t>(c[col]);
    }
  }
  // Defer all writes until every source (including sparse indices) has been read.
  for (uint32_t row = 0; row < 16; ++row) {
    for (uint32_t col = 0; col < 16; ++col) {
      const auto out = output_loc(row, col);
      regs.write_vgpr(dst + out.reg, out.lane, results[row * 16 + col]);
    }
  }
}

// Dense integer WMMA with either a VGPR or inline accumulator.
template <bool Gfx12, uint32_t InBits, uint32_t K = 16>
void exec_rdna_wmma_i32(auto &cu, uint32_t wave_size, uint32_t dst, uint32_t s0, uint32_t s1,
                        uint32_t s2, std::optional<uint32_t> const_acc, bool a_signed,
                        bool b_signed, bool clamp) {
  exec_rdna_wmma_i32_impl<Gfx12, InBits, K, false>(cu, wave_size, dst, s0, s1, s2, const_acc,
                                                   a_signed, b_signed, clamp, 0, 0);
}

/// Execute gfx1251 wave32 V_WMMA_F64_16X16X4_F64.
///
/// Public evidence for the logical matrix operation, vector widths, modifiers,
/// and register tuple sizes comes from LLVM's F64_F64X4_WMMA_w32 profile and
/// gfx1251_asm_wmma_w32.s. The physical mapping combines section 7.12.2 of the
/// public CDNA5 ISA reference with the checked-in FMT_WMMA_AB_16X4_F64 and
/// FMT_WMMA_DC_16X16_F64 component layouts:
///   A[row][k] / B[col][k]: lane = row-or-col + 16*(k/2), VGPR = 2*(k%2)
///   C/D[row][col]: lane = col + 16*(row/8), VGPR = 2*(row%8)
///
/// Matrix reuse bits are scheduling hints and therefore do not enter this
/// functional helper. All source and accumulator values are staged before any
/// destination write so overlapping C/D and A/B tuples preserve read-before-
/// write behavior. CDNA5 also requires EXEC to contain all 32 wave lanes.
inline void exec_wmma_f64_16x16x4_f64(auto &cu, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                                      uint64_t const_acc, uint32_t neg, uint32_t neg_hi,
                                      [[maybe_unused]] uint64_t exec_mask) {
  // LLVM marks GFX12 WMMA as ReadsModeReg=0. Until physical gfx1251 validation
  // establishes a more specific fixed arithmetic policy, use the IEEE baseline:
  // round-to-nearest-even and preserve input/output denormals.
  constexpr uint32_t kRoundNearestEven = 0;
  constexpr uint32_t kPreserveInputOutputDenormals = 3;
  // TODO(hanchung): Validate gfx1251 F64 WMMA lane/register mapping and fixed
  // arithmetic behavior against a physical gfx1251 KFD/ROCm execution capture.
  constexpr uint32_t M = 16;
  constexpr uint32_t N = 16;
  constexpr uint32_t K = 4;
  const uint64_t kFullExec = mfma_full_lane_mask(WMMA_WAVE32);
  assert(is_gfx1251_wmma_execution_state_valid(cu.wf_size(), exec_mask) &&
         "gfx1251 F64 WMMA requires wave32 with EXEC set to all ones");

  auto toggle_sign = [](uint64_t value, bool toggle) {
    return toggle ? value ^ (uint64_t{1} << 63) : value;
  };
  auto modify_acc = [](uint64_t value, uint32_t modifier) {
    if (modifier & 0x2u) {
      value &= ~(uint64_t{1} << 63);
    }
    if (modifier & 0x1u) {
      value ^= uint64_t{1} << 63;
    }
    return value;
  };

  std::array<uint64_t, M * K> a{};
  std::array<uint64_t, K * N> b{};
  std::array<uint64_t, M * N> result{};
  {
    auto reads = read_wmma_fast_path_regions<M, N, K>(
        cu, s0, s1, s2, /*data_bits=*/64, /*acc_bits=*/64,
        const_acc == ACC_FROM_VGPR ? ACC_FROM_VGPR : 0u, WMMA_WAVE32);
    for (uint32_t row = 0; row < M; ++row) {
      for (uint32_t k = 0; k < K; ++k) {
        const auto loc = wmma_input_loc<M, K>(row, k, /*data_bits=*/64);
        a[row * K + k] = toggle_sign(reads.a.lane64(loc.vgpr_offset, loc.lane), (neg & 0x1u) != 0);
      }
    }
    for (uint32_t k = 0; k < K; ++k) {
      for (uint32_t col = 0; col < N; ++col) {
        const auto loc = wmma_input_loc<N, K>(col, k, /*data_bits=*/64);
        b[k * N + col] = toggle_sign(reads.b.lane64(loc.vgpr_offset, loc.lane), (neg & 0x2u) != 0);
      }
    }

    const uint32_t c_modifier = wmma_c_modifier(neg, neg_hi);
    for (uint32_t row = 0; row < M; ++row) {
      for (uint32_t col = 0; col < N; ++col) {
        const auto out = wmma_output_loc_64<M, N>(row, col);
        uint64_t acc =
            const_acc == ACC_FROM_VGPR ? reads.acc->lane64(out.reg, out.lane) : const_acc;
        acc = modify_acc(acc, c_modifier);
        for (uint32_t k = 0; k < K; ++k) {
          acc = fp_mode::fma_f64(a[row * K + k], b[k * N + col], acc, kRoundNearestEven,
                                 kPreserveInputOutputDenormals);
        }
        result[row * N + col] = acc;
      }
    }
  }

  RegisterAccess regs(cu);
  auto writes = regs.write_vgpr_region(dst, /*reg_count=*/16, kFullExec);
  for (uint32_t row = 0; row < M; ++row) {
    for (uint32_t col = 0; col < N; ++col) {
      const auto out = wmma_output_loc_64<M, N>(row, col);
      const uint64_t value = result[row * N + col];
      writes.set_lane(out.reg, out.lane, static_cast<uint32_t>(value));
      writes.set_lane(out.reg + 1, out.lane, static_cast<uint32_t>(value >> 32));
    }
  }
}

/// exec_f64 for shapes and modes that try_exec_mfma_snapshot does not handle. Kept out of line so
/// its staging frame is not live while the snapshot path runs.
RJ_NOINLINE inline void exec_f64_generic(auto &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t B,
                                         uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                                         uint32_t const_acc, uint32_t neg) {
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
          if (!mfma_has_acc_register(const_acc, s2)) {
            acc = static_cast<double>(std::bit_cast<float>(const_acc));
          } else {
            uint32_t lo = RegisterAccess(cu).read_vgpr(s2 + out.reg, out.lane);
            uint32_t hi = RegisterAccess(cu).read_vgpr(s2 + out.reg + 1, out.lane);
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
    if (force_scalar(cu) || static_cast<size_t>(M) * K > MAX_AB ||
        static_cast<size_t>(K) * stride > MAX_BSTRIDE || static_cast<size_t>(M) * stride > MAX_C) {
      run_scalar();
    } else {
      // Zero-initialized staging buffers (uniform convention; see
      // wmma_simd_matmul). Cbuf is pre-seeded below.
      alignas(64) double Abuf[MAX_AB] = {};
      alignas(64) double Bbuf[MAX_BSTRIDE] = {};
      alignas(64) double Cbuf[MAX_C] = {};
      for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t row = 0; row < M; ++row) {
          for (uint32_t col = 0; col < N; ++col) {
            auto out = output_loc_64(M, N, row, col, b);
            if (!mfma_has_acc_register(const_acc, s2)) {
              Cbuf[row * stride + col] = static_cast<double>(std::bit_cast<float>(const_acc));
            } else {
              uint32_t lo = RegisterAccess(cu).read_vgpr(s2 + out.reg, out.lane);
              uint32_t hi = RegisterAccess(cu).read_vgpr(s2 + out.reg + 1, out.lane);
              Cbuf[row * stride + col] =
                  std::bit_cast<double>(static_cast<uint64_t>(hi) << 32 | lo);
            }
            Cbuf[row * stride + col] = apply_neg(Cbuf[row * stride + col], 0x4u);
          }
        }
        for (uint32_t row = 0; row < M; ++row) {
          for (uint32_t k = 0; k < K; ++k) {
            auto al = input_loc(M, K, B, row, k, b, 64);
            Abuf[row * K + k] = apply_neg(extract_f64(cu, s0, al), 0x1u);
          }
        }
        for (uint32_t k = 0; k < K; ++k) {
          for (uint32_t col = 0; col < N; ++col) {
            auto bl = input_loc(N, K, B, col, k, b, 64);
            Bbuf[k * stride + col] = apply_neg(extract_f64(cu, s1, bl), 0x2u);
          }
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
            for (uint32_t k = 0; k < K; ++k) {
              acc = std::fma(Abuf[row * K + k], Bbuf[k * stride + col], acc);
            }
            Cbuf[row * stride + col] = acc;
          }
        }
        for (uint32_t row = 0; row < M; ++row) {
          for (uint32_t col = 0; col < N; ++col) {
            auto out = output_loc_64(M, N, row, col, b);
            uint64_t bits = std::bit_cast<uint64_t>(Cbuf[row * stride + col]);
            results.push_back({out.reg, out.lane, static_cast<uint32_t>(bits),
                               static_cast<uint32_t>(bits >> 32)});
          }
        }
      }
    }
  } else {
    run_scalar();
  }

  for (const auto &r : results) {
    RegisterAccess(cu).write_vgpr(dst + r.reg, r.lane, r.lo);
    RegisterAccess(cu).write_vgpr(dst + r.reg + 1, r.lane, r.hi);
  }
}

/// MFMA execute for f64 output with f64 input: D = C + A x B.
inline void exec_f64(auto &cu, uint32_t M, uint32_t N, uint32_t K, uint32_t B, uint32_t dst,
                     uint32_t s0, uint32_t s1, uint32_t s2, uint32_t const_acc = ACC_FROM_VGPR,
                     uint32_t neg = 0) {
  constexpr std::integral_constant<uint32_t, 64> kBits;
  if (!try_exec_mfma_snapshot<double>(cu, M, N, K, B, kBits, kBits, dst, s0, s1, s2, extract_f64,
                                      extract_f64, const_acc, 0, 0, 0, neg)) {
    exec_f64_generic(cu, M, N, K, B, dst, s0, s1, s2, const_acc, neg);
  }
}

/// Fast path for the f32-input MFMA shapes (v_mfma_f32_*_f32). Like the f16
/// specialization but the inputs are already f32, so there is no F16C convert —
/// the hoist reads each operand word straight through observed register-access
/// regions. BATCH covers the batched shapes (e.g. 32x32x1x2). Compile-time
/// M/N/K/BATCH fully unroll the matmul; it loops over N in native-width chunks.
/// Falls back to the generic exec_f32 without host SIMD, when the native width
/// is one or does not divide N, for non-wave64 execution, under force-scalar,
/// or with cbsz/blgp.
template <uint32_t M, uint32_t N, uint32_t K, uint32_t BATCH>
void exec_f32_mfma_f32_spec(auto &cu, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                            uint32_t const_acc, uint32_t cbsz, uint32_t abid, uint32_t blgp,
                            MfmaF32Denorm denorm = {}) {
  constexpr uint32_t in_bits = 32;
  if constexpr (!util::has_stdx_simd) {
    exec_f32<M, N, K, BATCH>(cu, in_bits, dst, s0, s1, s2, extract_f32, extract_f32, const_acc,
                             cbsz, abid, blgp, denorm);
    return;
  } else {
    constexpr uint32_t W = static_cast<uint32_t>(util::native<float>::size());
    if (force_scalar(cu) || cbsz != 0 || blgp != 0 || !mma_f32_native_width_supported(N, W) ||
        cu.wf_size() != 64) {
      exec_f32<M, N, K, BATCH>(cu, in_bits, dst, s0, s1, s2, extract_f32, extract_f32, const_acc,
                               cbsz, abid, blgp, denorm);
      return;
    }
    const uint32_t wf = cu.wf_size();
    auto reads =
        read_mfma_fast_path_regions(cu, s0, s1, s2, M, N, K, BATCH, in_bits, const_acc, wf);
    auto writes = write_mfma_acc32_region(cu, dst, M, N, BATCH, wf);
    alignas(64) float A_buf[M * K];
    alignas(64) float B_buf[K * N];
    static_assert(M * N * BATCH * sizeof(float) <= 8 * 1024,
                  "specialized MFMA result staging exceeds the stack budget");
    alignas(64) float C_buf[M * N * BATCH];
    alignas(64) uint32_t A_words[M * K * BATCH];
    alignas(64) uint32_t B_words[N * K * BATCH];
    alignas(64) uint32_t C_words[M * N * BATCH];
    if (copy_matrix_region_words(reads.a, A_words).failed()) {
      return;
    }
    if (copy_matrix_region_words(reads.b, B_words).failed()) {
      return;
    }
    if (reads.acc && copy_matrix_region_words(*reads.acc, C_words).failed()) {
      return;
    }
    bool has_nan_or_inf = false;
    for (uint32_t b = 0; b < BATCH; ++b) {
      float *C_batch = &C_buf[b * M * N];
      for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t col = 0; col < N; ++col) {
          auto out = output_loc_32(M, N, row, col, b);
          C_batch[row * N + col] = (!mfma_has_acc_register(const_acc, s2))
                                       ? std::bit_cast<float>(const_acc)
                                       : std::bit_cast<float>(C_words[out.reg * wf + out.lane]);
          C_batch[row * N + col] = denorm.input(C_batch[row * N + col]);
        }
      }
      for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t k = 0; k < K; ++k) {
          auto al = input_loc(M, K, BATCH, row, k, b, in_bits);
          A_buf[row * K + k] =
              denorm.input(std::bit_cast<float>(A_words[al.vgpr_offset * wf + al.lane]));
        }
      }
      for (uint32_t k = 0; k < K; ++k) {
        for (uint32_t col = 0; col < N; ++col) {
          auto bl = input_loc(N, K, BATCH, col, k, b, in_bits);
          B_buf[k * N + col] =
              denorm.input(std::bit_cast<float>(B_words[bl.vgpr_offset * wf + bl.lane]));
        }
      }
      for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t c0 = 0; c0 < N; c0 += W) {
          util::native<float> c_row;
          c_row.copy_from(&C_batch[row * N + c0], util::stdx::vector_aligned);
          for (uint32_t k = 0; k < K; ++k) {
            util::native<float> a_bcast(A_buf[row * K + k]);
            util::native<float> b_row;
            b_row.copy_from(&B_buf[k * N + c0], util::stdx::vector_aligned);
            c_row = util::stdx::fma(a_bcast, b_row, c_row);
          }
          c_row.copy_to(&C_batch[row * N + c0], util::stdx::vector_aligned);
        }
      }
    }
    // Publish only after every batch has consumed its inputs.
    for (uint32_t b = 0; b < BATCH; ++b) {
      for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t col = 0; col < N; ++col) {
          auto out = output_loc_32(M, N, row, col, b);
          float fv = denorm.output(C_buf[(b * M + row) * N + col]);
          C_words[out.reg * wf + out.lane] = std::bit_cast<uint32_t>(fv);
          if (std::isnan(fv) || std::isinf(fv)) {
            has_nan_or_inf = true;
          }
        }
      }
    }
    static_assert(output_loc_32_fills_wave64_registers<M, N, BATCH>(),
                  "f32 MFMA outputs must fill whole wave64 VGPRs exactly once");
    for (uint32_t reg = 0; reg < writes.reg_count(); ++reg) {
      writes.set_lanes(reg, std::span<const uint32_t>(C_words + reg * wf, wf));
    }
    if (has_nan_or_inf) {
      util::Logger::vm([&](auto &os) {
        os << std::format("MFMA_NAN_DETECTED (simd) dst=v{} s0=v{} s1=v{} s2=v{} {}x{}x{}_f32", dst,
                          s0, s1, s2, M, N, K);
      });
    }
  }
}

/// Fast path for the f16-input f32 MFMA shapes. The original target,
/// v_mfma_f32_16x16x32_f16, accounts for 488k invocations and about 4B
/// internal MACs per OPT-125M fp16 eager forward. Compile-time M/N/K/B let the
/// compiler fully unroll the inner matmul into straight-line native SIMD FMAs;
/// a runtime-dimension loop is materially slower on this hot path. The path
/// snapshots packed f16 inputs through observed register-access regions,
/// bulk-converts them into dense f32 buffers, and publishes staged results one
/// VGPR at a time only after every operand read. It falls back to the generic
/// exec_f32 when:
///   - <experimental/simd> is unavailable
///   - host native_simd<float> has no usable width for the output columns
///   - cbsz/blgp lane permutation is non-default
///   - RJ_FORCE_SCALAR is set
///   - the wave is not wave64
template <uint32_t M, uint32_t N, uint32_t K, uint32_t BATCH = 1>
void exec_f32_mfma_f16_spec(auto &cu, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                            uint32_t const_acc, uint32_t cbsz, uint32_t abid, uint32_t blgp,
                            MfmaF32Denorm denorm = {}) {
  constexpr uint32_t B = BATCH, in_bits = 16;
  if constexpr (!util::has_stdx_simd) {
    exec_f32<M, N, K, B>(cu, in_bits, dst, s0, s1, s2, amdgpu::extract_f16, amdgpu::extract_f16,
                         const_acc, cbsz, abid, blgp, denorm);
    return;
  } else {
    constexpr uint32_t W = static_cast<uint32_t>(util::native<float>::size());
    if (force_scalar(cu) || cbsz != 0 || blgp != 0 || !mma_f32_native_width_supported(N, W) ||
        cu.wf_size() != 64) {
      exec_f32<M, N, K, B>(cu, in_bits, dst, s0, s1, s2, amdgpu::extract_f16, amdgpu::extract_f16,
                           const_acc, cbsz, abid, blgp, denorm);
      return;
    }
    const uint32_t wf = cu.wf_size();
    auto reads = read_mfma_fast_path_regions(cu, s0, s1, s2, M, N, K, B, in_bits, const_acc, wf);
    auto writes = write_mfma_acc32_region(cu, dst, M, N, B, wf);
    alignas(64) float A_buf[M * K]; // A[row][k] (one batch block)
    alignas(64) float B_buf[K * N]; // B[k][col] (one batch block)
    static_assert(M * N * B * sizeof(float) <= 8 * 1024,
                  "specialized MFMA result staging exceeds the stack budget");
    alignas(64) float C_buf[M * N * B]; // C[batch][row][col]
    alignas(64) uint32_t C_words[M * N * B];
    if (reads.acc && copy_matrix_region_words(*reads.acc, C_words).failed()) {
      return;
    }
    // A/B occupy M*K*B and N*K*B packed f16 over their VGPRs. Bulk-convert each
    // whole region to f32 once with F16C (one vector op per 16 halves) instead
    // of branchy per-element f16_to_f32, then the hoist is a pure f32
    // index-shuffle (f16 of word w lane l sub s -> flat (w*wf+l)*2+s).
    alignas(64) float A_f32[M * K * B];
    alignas(64) float B_f32[N * K * B];
    if (convert_f16_matrix_region(reads.a, A_f32, M * K * B).failed()) {
      return;
    }
    if (convert_f16_matrix_region(reads.b, B_f32, N * K * B).failed()) {
      return;
    }
    if (!(denorm.mode & 1u)) {
      for (float &value : A_f32) {
        value = denorm.input(value, true);
      }
    }
    if (!(denorm.mode & 1u)) {
      for (float &value : B_f32) {
        value = denorm.input(value, true);
      }
    }
    bool has_nan_or_inf = false;
    for (uint32_t b = 0; b < B; ++b) {
      float *C_batch = &C_buf[b * M * N];
      for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t col = 0; col < N; ++col) {
          auto out = output_loc_32(M, N, row, col, b);
          C_batch[row * N + col] = (!mfma_has_acc_register(const_acc, s2))
                                       ? std::bit_cast<float>(const_acc)
                                       : std::bit_cast<float>(C_words[out.reg * wf + out.lane]);
          C_batch[row * N + col] = denorm.input(C_batch[row * N + col]);
        }
      }
      for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t k = 0; k < K; ++k) {
          auto al = input_loc(M, K, B, row, k, b, in_bits);
          A_buf[row * K + k] = A_f32[(al.vgpr_offset * wf + al.lane) * 2 + al.sub_element];
        }
      }
      for (uint32_t k = 0; k < K; ++k) {
        for (uint32_t col = 0; col < N; ++col) {
          auto bl = input_loc(N, K, B, col, k, b, in_bits);
          B_buf[k * N + col] = B_f32[(bl.vgpr_offset * wf + bl.lane) * 2 + bl.sub_element];
        }
      }
      // Dense MxKxN matmul, W-lane native SIMD FMA over N (N/W chunks per row).
      for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t c0 = 0; c0 < N; c0 += W) {
          util::native<float> c_row;
          c_row.copy_from(&C_batch[row * N + c0], util::stdx::vector_aligned);
          for (uint32_t k = 0; k < K; ++k) {
            util::native<float> a_bcast(A_buf[row * K + k]);
            util::native<float> b_row;
            b_row.copy_from(&B_buf[k * N + c0], util::stdx::vector_aligned);
            c_row = util::stdx::fma(a_bcast, b_row, c_row);
          }
          c_row.copy_to(&C_batch[row * N + c0], util::stdx::vector_aligned);
        }
      }
    }
    // Publish only after every batch has consumed its inputs. This preserves
    // instruction-level snapshot semantics when dst overlaps a later batch's
    // source or accumulator registers.
    for (uint32_t b = 0; b < B; ++b) {
      for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t col = 0; col < N; ++col) {
          auto out = output_loc_32(M, N, row, col, b);
          float fv = denorm.output(C_buf[(b * M + row) * N + col]);
          C_words[out.reg * wf + out.lane] = std::bit_cast<uint32_t>(fv);
          if (std::isnan(fv) || std::isinf(fv)) {
            has_nan_or_inf = true;
          }
        }
      }
    }
    // This path runs only for wave64, so the write region holds M*N*B/64 whole
    // VGPRs and each published span lies inside C_words. Every word of those
    // spans must also have been staged above; a gap would publish a stale
    // accumulator word or uninitialized stack memory.
    static_assert(output_loc_32_fills_wave64_registers<M, N, B>(),
                  "f16 MFMA outputs must fill whole wave64 VGPRs exactly once");
    for (uint32_t reg = 0; reg < writes.reg_count(); ++reg) {
      writes.set_lanes(reg, std::span<const uint32_t>(C_words + reg * wf, wf));
    }
    if (has_nan_or_inf) {
      util::Logger::vm([&](auto &os) {
        os << std::format("MFMA_NAN_DETECTED (simd) dst=v{} s0=v{} s1=v{} s2=v{} {}x{}x{}_f16", dst,
                          s0, s1, s2, M, N, K);
      });
    }
  }
}

/// Fast path for the bf16-input MFMA shapes (v_mfma_f32_*_bf16). Follows the
/// f16 specialization, except that the bulk input convert is the bf16
/// zero-extend (no F16C needed) and results are published one word at a time.
/// Falls back to the generic exec_f32 without usable host SIMD, under
/// force-scalar, or with cbsz/blgp.
template <uint32_t M, uint32_t N, uint32_t K, uint32_t BATCH = 1>
void exec_f32_mfma_bf16_spec(auto &cu, uint32_t dst, uint32_t s0, uint32_t s1, uint32_t s2,
                             uint32_t const_acc, uint32_t cbsz, uint32_t abid, uint32_t blgp,
                             MfmaF32Denorm denorm = {}) {
  constexpr uint32_t B = BATCH, in_bits = 16;
  if constexpr (!util::has_stdx_simd) {
    exec_f32<M, N, K, B>(cu, in_bits, dst, s0, s1, s2, amdgpu::extract_bf16, amdgpu::extract_bf16,
                         const_acc, cbsz, abid, blgp, denorm);
    return;
  } else {
    constexpr uint32_t W = static_cast<uint32_t>(util::native<float>::size());
    if (force_scalar(cu) || cbsz != 0 || blgp != 0 || !mma_f32_native_width_supported(N, W) ||
        cu.wf_size() != 64) {
      exec_f32<M, N, K, B>(cu, in_bits, dst, s0, s1, s2, amdgpu::extract_bf16, amdgpu::extract_bf16,
                           const_acc, cbsz, abid, blgp, denorm);
      return;
    }
    const uint32_t wf = cu.wf_size();
    auto reads = read_mfma_fast_path_regions(cu, s0, s1, s2, M, N, K, B, in_bits, const_acc, wf);
    auto writes = write_mfma_acc32_region(cu, dst, M, N, B, wf);
    alignas(64) float A_buf[M * K]; // A[row][k] (one batch block)
    alignas(64) float B_buf[K * N]; // B[k][col] (one batch block)
    static_assert(M * N * B * sizeof(float) <= 8 * 1024,
                  "specialized MFMA result staging exceeds the stack budget");
    alignas(64) float C_buf[M * N * B]; // C[batch][row][col]
    alignas(64) uint32_t C_words[M * N * B];
    if (reads.acc && copy_matrix_region_words(*reads.acc, C_words).failed()) {
      return;
    }
    // Bulk-convert the packed bf16 regions to f32 once (zero-extend + shift),
    // then the hoist is a pure f32 index-shuffle.
    alignas(64) float A_f32[M * K * B];
    alignas(64) float B_f32[N * K * B];
    if (convert_bf16_matrix_region(reads.a, A_f32, M * K * B).failed()) {
      return;
    }
    if (convert_bf16_matrix_region(reads.b, B_f32, N * K * B).failed()) {
      return;
    }
    if (!(denorm.mode & 1u)) {
      for (float &value : A_f32) {
        value = denorm.input(value, false);
      }
    }
    if (!(denorm.mode & 1u)) {
      for (float &value : B_f32) {
        value = denorm.input(value, false);
      }
    }
    bool has_nan_or_inf = false;
    for (uint32_t b = 0; b < B; ++b) {
      float *C_batch = &C_buf[b * M * N];
      for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t col = 0; col < N; ++col) {
          auto out = output_loc_32(M, N, row, col, b);
          C_batch[row * N + col] = (!mfma_has_acc_register(const_acc, s2))
                                       ? std::bit_cast<float>(const_acc)
                                       : std::bit_cast<float>(C_words[out.reg * wf + out.lane]);
          C_batch[row * N + col] = denorm.input(C_batch[row * N + col]);
        }
      }
      for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t k = 0; k < K; ++k) {
          auto al = input_loc(M, K, B, row, k, b, in_bits);
          A_buf[row * K + k] = A_f32[(al.vgpr_offset * wf + al.lane) * 2 + al.sub_element];
        }
      }
      for (uint32_t k = 0; k < K; ++k) {
        for (uint32_t col = 0; col < N; ++col) {
          auto bl = input_loc(N, K, B, col, k, b, in_bits);
          B_buf[k * N + col] = B_f32[(bl.vgpr_offset * wf + bl.lane) * 2 + bl.sub_element];
        }
      }
      // Dense MxKxN matmul, W-lane native SIMD FMA over N (N/W chunks per row).
      for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t c0 = 0; c0 < N; c0 += W) {
          util::native<float> c_row;
          c_row.copy_from(&C_batch[row * N + c0], util::stdx::vector_aligned);
          for (uint32_t k = 0; k < K; ++k) {
            util::native<float> a_bcast(A_buf[row * K + k]);
            util::native<float> b_row;
            b_row.copy_from(&B_buf[k * N + c0], util::stdx::vector_aligned);
            c_row = util::stdx::fma(a_bcast, b_row, c_row);
          }
          c_row.copy_to(&C_batch[row * N + c0], util::stdx::vector_aligned);
        }
      }
    }
    // Publish only after every batch has consumed its inputs.
    for (uint32_t b = 0; b < B; ++b) {
      for (uint32_t row = 0; row < M; ++row) {
        for (uint32_t col = 0; col < N; ++col) {
          auto out = output_loc_32(M, N, row, col, b);
          float fv = denorm.output(C_buf[(b * M + row) * N + col]);
          C_words[out.reg * wf + out.lane] = std::bit_cast<uint32_t>(fv);
          if (std::isnan(fv) || std::isinf(fv)) {
            has_nan_or_inf = true;
          }
        }
      }
    }
    static_assert(output_loc_32_fills_wave64_registers<M, N, B>(),
                  "bf16 MFMA outputs must fill whole wave64 VGPRs exactly once");
    for (uint32_t reg = 0; reg < writes.reg_count(); ++reg) {
      writes.set_lanes(reg, std::span<const uint32_t>(C_words + reg * wf, wf));
    }
    if (has_nan_or_inf) {
      util::Logger::vm([&](auto &os) {
        os << std::format("MFMA_NAN_DETECTED (simd) dst=v{} s0=v{} s1=v{} s2=v{} {}x{}x{}_bf16",
                          dst, s0, s1, s2, M, N, K);
      });
    }
  }
}

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_EXEC_H_
