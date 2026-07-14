// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file dpp_sdwa_ops.h
/// @brief DPP (Data-Parallel Primitives) and SDWA (Sub-Dword Access) helpers.
///
/// @details DPP modifies how VOP1/VOP2 instructions read src0 by applying a lane
/// permutation before the ALU operation. The permutation is controlled by
/// dpp_ctrl (9 bits), with row_mask/bank_mask disabling individual lanes.
///
/// SDWA selects sub-dword portions of source operands and merges results
/// into sub-dword positions of the destination. Available on GFX9 (CDNA)
/// and RDNA1/2; removed in RDNA3+.

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_DPP_SDWA_OPS_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_DPP_SDWA_OPS_H_

#include "rocjitsu/isa/operand.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <memory>

namespace rocjitsu {
namespace amdgpu {

/// @brief VOP1/VOP2 src0 encoding values that indicate a DPP or SDWA suffix.
constexpr uint32_t SRC_SDWA = 249;
constexpr uint32_t SRC_DPP = 250;
constexpr uint32_t SRC_DPP8_FI_0 = 233;
constexpr uint32_t SRC_DPP8_FI_1 = 234;
constexpr uint32_t SRC_DPP8_LO = SRC_DPP8_FI_0;
constexpr uint32_t SRC_DPP8_HI = SRC_DPP8_FI_1;

namespace dpp {

inline bool is_src_dpp8(uint32_t src0) { return src0 == SRC_DPP8_FI_0 || src0 == SRC_DPP8_FI_1; }

inline uint32_t src_dpp8_fi(uint32_t src0) { return src0 == SRC_DPP8_FI_1 ? 1u : 0u; }

/// Row size for DPP operations (16 lanes per row).
constexpr int ROW_SIZE = 16;
/// Number of banks per row (4 banks of 4 lanes each).
constexpr int NUM_BANKS = 4;

/// @brief DPP control value ranges.
enum DppCtrl : uint32_t {
  QUAD_PERM_MAX = 0xFF,
  ROW_SHL1 = 0x101, // row shift left 1..15
  ROW_SHL_MAX = 0x10F,
  ROW_SHR1 = 0x111, // row shift right 1..15
  ROW_SHR_MAX = 0x11F,
  ROW_ROR1 = 0x121, // row rotate right 1..15
  ROW_ROR_MAX = 0x12F,
  WF_SHL1 = 0x130,
  WF_ROL1 = 0x134, // wave rotate left 1
  WF_SRL1 = 0x138, // wave shift right 1
  WF_ROR1 = 0x13C, // wave rotate right 1
  ROW_MIRROR = 0x140,
  ROW_HALF_MIRROR = 0x141,
  ROW_BCAST15 = 0x142,    // broadcast lane 15 of each row to the following row
  ROW_BCAST31 = 0x143,    // broadcast lane 31 to the upper half-wave
  ROW_SHARE_BASE = 0x150, // row_share/row_newbcast: broadcast one lane within each row
  ROW_SHARE_MAX = 0x15F,
  ROW_XMASK_BASE = 0x160, // row_xmask (GFX10+)
  ROW_XMASK_MAX = 0x16F,
};

/// @brief Compute the source lane index for a DPP permutation.
///
/// @param dpp_ctrl 9-bit DPP control value.
/// @param lane Current lane index (0..wf_size-1).
/// @param wf_size Wavefront size (32 or 64).
/// @param[out] out_of_bounds Set to true if the source lane is invalid.
/// @returns Source lane to read from.
inline int dpp_permute(uint32_t dpp_ctrl, int lane, int wf_size, bool &out_of_bounds) {
  out_of_bounds = false;
  int row_num = lane / ROW_SIZE;
  int row_off = lane % ROW_SIZE;

  if (dpp_ctrl <= QUAD_PERM_MAX) {
    // Quad permute: 4 lanes per quad, 2-bit selector per lane position.
    int quad_base = lane & ~3;
    int quad_idx = lane & 3;
    int new_idx = (dpp_ctrl >> (2 * quad_idx)) & 3;
    return quad_base | new_idx;
  }

  if (dpp_ctrl >= ROW_SHL1 && dpp_ctrl <= ROW_SHL_MAX) {
    // row_shl N: data shifts left (toward lower lane indices).
    // Lane K reads from lane K+N (higher index).
    int shift = dpp_ctrl - ROW_SHL1 + 1;
    int new_off = row_off + shift;
    if (new_off >= ROW_SIZE) {
      out_of_bounds = true;
      return lane;
    }
    return row_num * ROW_SIZE + new_off;
  }

  if (dpp_ctrl >= ROW_SHR1 && dpp_ctrl <= ROW_SHR_MAX) {
    // row_shr N: data shifts right (toward higher lane indices).
    // Lane K reads from lane K-N (lower index).
    int shift = dpp_ctrl - ROW_SHR1 + 1;
    int new_off = row_off - shift;
    if (new_off < 0) {
      out_of_bounds = true;
      return lane;
    }
    return row_num * ROW_SIZE + new_off;
  }

  if (dpp_ctrl >= ROW_ROR1 && dpp_ctrl <= ROW_ROR_MAX) {
    // row_ror N: data rotates right within the row.
    // Lane K reads from lane (K-N+ROW_SIZE) % ROW_SIZE.
    int rot = dpp_ctrl - ROW_ROR1 + 1;
    int new_off = (row_off - rot + ROW_SIZE) % ROW_SIZE;
    return row_num * ROW_SIZE + new_off;
  }

  if (dpp_ctrl == WF_SHL1) {
    // Wave shift left 1: lane K reads from lane K+1.
    int src = lane + 1;
    if (src >= wf_size)
      out_of_bounds = true;
    return src < wf_size ? src : lane;
  }

  if (dpp_ctrl == WF_ROL1) {
    // Wave rotate left 1: lane K reads from lane (K+1) % wf_size.
    return (lane + 1) % wf_size;
  }

  if (dpp_ctrl == WF_SRL1) {
    // Wave shift right 1: lane K reads from lane K-1.
    int src = lane - 1;
    if (src < 0)
      out_of_bounds = true;
    return src >= 0 ? src : lane;
  }

  if (dpp_ctrl == WF_ROR1) {
    // Wave rotate right 1: lane K reads from lane (K-1+wf_size) % wf_size.
    return (lane - 1 + wf_size) % wf_size;
  }

  if (dpp_ctrl == ROW_MIRROR) {
    return row_num * ROW_SIZE + (ROW_SIZE - 1 - row_off);
  }

  if (dpp_ctrl == ROW_HALF_MIRROR) {
    int half_base = lane & ~7;
    int half_off = lane & 7;
    return half_base | (7 - half_off);
  }

  if (dpp_ctrl == ROW_BCAST15) {
    // Broadcast lane 15 of each row to the following row. Row 0 has invalid
    // shared data.
    if (lane < ROW_SIZE) {
      out_of_bounds = true;
      return lane;
    }
    return row_num * ROW_SIZE - 1;
  }

  if (dpp_ctrl == ROW_BCAST31) {
    // Broadcast lane 31 to lanes 32-63. Lanes 0-31 have invalid shared data.
    if (lane < 32 || wf_size <= 32) {
      out_of_bounds = true;
      return lane;
    }
    return 31;
  }

  if (dpp_ctrl >= ROW_SHARE_BASE && dpp_ctrl <= ROW_SHARE_MAX) {
    // row_share/row_newbcast: broadcast one selected source lane within the
    // destination row.
    int lane_sel = dpp_ctrl - ROW_SHARE_BASE;
    return row_num * ROW_SIZE + lane_sel;
  }

  if (dpp_ctrl >= ROW_XMASK_BASE && dpp_ctrl <= ROW_XMASK_MAX) {
    // row_xmask: XOR the lane offset within the row with a 4-bit mask.
    int mask = dpp_ctrl - ROW_XMASK_BASE;
    int new_off = row_off ^ mask;
    if (new_off >= ROW_SIZE) {
      out_of_bounds = true;
      return lane;
    }
    return row_num * ROW_SIZE + new_off;
  }

  // Unknown dpp_ctrl — identity.
  return lane;
}

/// @brief Check if a lane is disabled by DPP row/bank masks.
///
/// @param lane Lane index.
/// @param row_mask 4-bit row mask (bit N enables row N, 16 lanes/row).
///        For wave32, only bits 0-1 are meaningful (rows 0-1 cover lanes 0-31);
///        bits 2-3 have no effect since no lanes map to rows 2-3.
/// @param bank_mask 4-bit bank mask (bit N enables bank N, 4 lanes/bank).
/// @returns True if the lane is disabled (should not be written).
inline bool dpp_lane_masked(int lane, uint32_t row_mask, uint32_t bank_mask) {
  int row = lane / ROW_SIZE;
  int bank = (lane % ROW_SIZE) / NUM_BANKS;
  return ((row_mask & (1u << row)) == 0) || ((bank_mask & (1u << bank)) == 0);
}

/// @brief Check if a DPP instruction writes the destination lane.
///
/// Row/bank masks always disable writes. When the DPP permutation has invalid
/// shared data, BOUND_CTRL=0 disables the write and BOUND_CTRL=1 writes using a
/// zero source value.
inline bool dpp_lane_write_enabled(int lane, int wf_size, uint32_t dpp_ctrl, uint32_t row_mask,
                                   uint32_t bank_mask, uint32_t bound_ctrl) {
  if (dpp_lane_masked(lane, row_mask, bank_mask))
    return false;

  bool oob = false;
  (void)dpp_permute(dpp_ctrl, lane, wf_size, oob);
  return !oob || bound_ctrl != 0;
}

/// @brief Compute the destination write mask for a DPP instruction.
///
/// Includes only lanes enabled by row_mask/bank_mask and, when the DPP
/// permutation reads invalid shared data, only lanes whose BOUND_CTRL behavior
/// still writes a zero source value.
///
/// @param wf_size Wavefront size in lanes.
/// @param dpp_ctrl 9-bit DPP control value.
/// @param row_mask 4-bit row mask.
/// @param bank_mask 4-bit bank mask.
/// @param bound_ctrl If 1, invalid shared data writes zero; if 0, write is disabled.
/// @returns Bit mask with one bit per destination lane that should be written.
inline uint64_t dpp_write_mask(uint32_t wf_size, uint32_t dpp_ctrl, uint32_t row_mask,
                               uint32_t bank_mask, uint32_t bound_ctrl) {
  uint64_t mask = 0;
  for (uint32_t ln = 0; ln < wf_size; ++ln)
    if (dpp_lane_write_enabled(static_cast<int>(ln), static_cast<int>(wf_size), dpp_ctrl, row_mask,
                               bank_mask, bound_ctrl))
      mask |= (1ULL << ln);
  return mask;
}

/// @brief Apply a complete DPP read for one lane.
///
/// Reads the permuted source value from a VGPR across the wavefront.
/// Handles row/bank masking (returns old_val if masked), out-of-bounds
/// permutation (returns 0 if bound_ctrl=1, else returns old_val), and RDNA
/// fetch-inactive mode (returns 0 for inactive source lanes when fi=0).
/// Callers must still apply dpp_write_mask() after the ALU operation so lanes
/// disabled by masks or BOUND_CTRL=0 invalid shared data keep their old
/// destination values.
///
/// @param src_data Array of wf_size source values (one per lane).
/// @param lane Current lane index.
/// @param wf_size Wavefront size.
/// @param dpp_ctrl 9-bit DPP control value.
/// @param row_mask 4-bit row mask.
/// @param bank_mask 4-bit bank mask.
/// @param bound_ctrl If 1, out-of-bounds lanes read 0; if 0, unchanged.
/// @param fi If 1, inactive source lanes still read GPRs; if 0, they read 0.
/// @param old_val The lane's current value (used when masked/OOB with bound_ctrl=0).
/// @param exec_mask EXEC value used to classify inactive source lanes.
/// @returns The DPP-permuted source value for this lane.
inline uint32_t dpp_read(const uint32_t *src_data, int lane, int wf_size, uint32_t dpp_ctrl,
                         uint32_t row_mask, uint32_t bank_mask, uint32_t bound_ctrl, uint32_t fi,
                         uint32_t old_val, uint64_t exec_mask) {
  if (dpp_lane_masked(lane, row_mask, bank_mask))
    return old_val;

  bool oob = false;
  int src_lane = dpp_permute(dpp_ctrl, lane, wf_size, oob);

  if (oob)
    return bound_ctrl ? 0u : old_val;

  if (!fi && (exec_mask & (1ULL << src_lane)) == 0)
    return 0u;

  return src_data[src_lane];
}

/// @brief Pre-permute src0 for a DPP instruction.
///
/// Reads all src0 VGPR lanes, applies the DPP permutation, creates a
/// DppOperand with the permuted data, and swaps the src0 pointer.
/// Called from VOP1/VOP2 execute_impl() when src0 == 250.
///
/// @param[in,out] src0 Source operand pointer to replace.
/// @param dpp_ctrl 9-bit DPP control value.
/// @param row_mask 4-bit row mask.
/// @param bank_mask 4-bit bank mask.
/// @param bound_ctrl Bound control (1 = zero OOB, 0 = preserve).
/// @param fi Fetch-inactive control (1 = read inactive source lanes, 0 = zero).
/// @param[out] storage Owning pointer for the DppOperand lifetime.
/// @param wf Wavefront providing register state.
inline void apply_dpp(Operand *&src0, uint32_t dpp_ctrl, uint32_t row_mask, uint32_t bank_mask,
                      uint32_t bound_ctrl, uint32_t fi, std::unique_ptr<DppOperand> &storage,
                      amdgpu::Wavefront &wf) {
  auto &cu = wf.cu();
  uint32_t ws = wf.wf_size();
  uint32_t vbase = wf.vgpr_alloc().base + src0->encoding_value_;
  uint64_t exec_mask = wf.exec();
  uint32_t raw[64], result[64];
  for (uint32_t i = 0; i < ws; ++i)
    raw[i] = cu.read_vgpr(vbase, i);
  for (uint32_t i = 0; i < ws; ++i)
    result[i] = dpp_read(raw, static_cast<int>(i), static_cast<int>(ws), dpp_ctrl, row_mask,
                         bank_mask, bound_ctrl, fi, raw[i], exec_mask);
  storage = std::make_unique<DppOperand>(*src0, result, static_cast<int>(ws));
  src0 = storage.get();
}

inline uint32_t dpp8_src_lane(uint32_t lane, uint32_t lane_sel) {
  uint32_t sel = (lane_sel >> ((lane & 7u) * 3u)) & 7u;
  return (lane & ~7u) | sel;
}

inline uint32_t dpp8_read(const uint32_t *src_data, uint32_t lane, uint32_t wf_size,
                          uint32_t lane_sel, uint32_t fi, uint64_t exec_mask) {
  uint32_t src_lane = dpp8_src_lane(lane, lane_sel);
  if (src_lane >= wf_size)
    return 0u;
  if (!fi && (exec_mask & (1ULL << src_lane)) == 0)
    return 0u;
  return src_data[src_lane];
}

inline void apply_dpp8(Operand *&src0, uint32_t lane_sel, uint32_t fi,
                       std::unique_ptr<DppOperand> &storage, amdgpu::Wavefront &wf) {
  auto &cu = wf.cu();
  uint32_t ws = wf.wf_size();
  uint32_t vbase = wf.vgpr_alloc().base + src0->encoding_value_;
  uint64_t exec_mask = wf.exec();
  uint32_t raw[64], result[64];
  for (uint32_t i = 0; i < ws; ++i)
    raw[i] = cu.read_vgpr(vbase, i);
  for (uint32_t lane = 0; lane < ws; ++lane)
    result[lane] = dpp8_read(raw, lane, ws, lane_sel, fi, exec_mask);
  storage = std::make_unique<DppOperand>(*src0, result, static_cast<int>(ws));
  src0 = storage.get();
}

} // namespace dpp

namespace sdwa {

/// @brief SDWA sub-dword selection values.
enum SdwaSel : uint32_t {
  BYTE_0 = 0,
  BYTE_1 = 1,
  BYTE_2 = 2,
  BYTE_3 = 3,
  WORD_0 = 4,
  WORD_1 = 5,
  DWORD = 6,
};

/// @brief SDWA unused bits handling for destination.
enum SdwaUnused : uint32_t {
  UNUSED_PAD = 0,      ///< Zero-fill unused bytes/words.
  UNUSED_SEXT = 1,     ///< Sign-extend from the selected portion's MSB.
  UNUSED_PRESERVE = 2, ///< Keep the original destination register value.
};

/// @brief Extract a sub-dword from a source value per SDWA sel.
///
/// @param val The full 32-bit source value.
/// @param sel Sub-dword selection (BYTE_0..BYTE_3, WORD_0, WORD_1, DWORD).
/// @param sign_ext If true, sign-extend the extracted value to 32 bits.
/// @returns The extracted (and optionally sign-extended) value.
inline uint32_t sdwa_src_select(uint32_t val, uint32_t sel, bool sign_ext) {
  if (sel == DWORD)
    return val;

  if (sel <= BYTE_3) {
    uint32_t shift = sel * 8;
    uint32_t byte_val = (val >> shift) & 0xFF;
    if (sign_ext && (byte_val & 0x80))
      return byte_val | 0xFFFFFF00u;
    return byte_val;
  }

  // WORD_0 or WORD_1
  uint32_t shift = (sel & 1) * 16;
  uint32_t word_val = (val >> shift) & 0xFFFF;
  if (sign_ext && (word_val & 0x8000))
    return word_val | 0xFFFF0000u;
  return word_val;
}

/// @brief Merge an ALU result into a destination register per SDWA dst_sel.
///
/// @param result The 32-bit ALU result to merge.
/// @param old_dst The original destination register value (for PRESERVE mode).
/// @param dst_sel Destination sub-dword selection.
/// @param dst_unused How to handle unused bytes/words (PAD, SEXT, PRESERVE).
/// @returns The merged 32-bit destination value.
inline uint32_t sdwa_dst_merge(uint32_t result, uint32_t old_dst, uint32_t dst_sel,
                               uint32_t dst_unused) {
  if (dst_sel == DWORD)
    return result;

  if (dst_sel <= BYTE_3) {
    uint32_t shift = dst_sel * 8;
    uint32_t mask = 0xFFu << shift;
    uint32_t merged = (result & 0xFF) << shift;
    uint32_t upper_mask = static_cast<uint32_t>(~((uint64_t{1} << (shift + 8)) - 1));
    uint32_t fill;
    if (dst_unused == UNUSED_PRESERVE)
      fill = old_dst & ~mask;
    else if (dst_unused == UNUSED_SEXT && (result & 0x80))
      fill = upper_mask;
    else
      fill = 0;
    return fill | merged;
  }

  // WORD_0 or WORD_1
  uint32_t shift = (dst_sel & 1) * 16;
  uint32_t mask = 0xFFFFu << shift;
  uint32_t merged = (result & 0xFFFF) << shift;
  uint32_t upper_mask = static_cast<uint32_t>(~((uint64_t{1} << (shift + 16)) - 1));
  uint32_t fill;
  if (dst_unused == UNUSED_PRESERVE)
    fill = old_dst & ~mask;
  else if (dst_unused == UNUSED_SEXT && (result & 0x8000))
    fill = upper_mask;
  else
    fill = 0;
  return fill | merged;
}

/// @brief Apply SDWA clamp to an ALU result.
///
/// For floating-point operations, clamps the result to [0.0, 1.0].
/// The caller determines whether the operation is float or integer
/// based on the instruction's semantic type.
inline uint32_t sdwa_clamp_f32(uint32_t result) {
  float f = std::bit_cast<float>(result);
  f = std::fmin(std::fmax(f, 0.0f), 1.0f);
  return std::bit_cast<uint32_t>(f);
}

} // namespace sdwa
} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_DPP_SDWA_OPS_H_
