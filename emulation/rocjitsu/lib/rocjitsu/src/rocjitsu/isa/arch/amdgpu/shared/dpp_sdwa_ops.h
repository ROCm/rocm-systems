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

#include "rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <memory>

namespace rocjitsu {
namespace amdgpu {

namespace dpp {

/// Row size for DPP operations (16 lanes per row).
constexpr int ROW_SIZE = 16;
/// Number of banks per row (4 banks of 4 lanes each).
constexpr int NUM_BANKS = 4;

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

/// @brief Return destination lanes enabled by EXEC and instruction modifiers.
///
/// Applies DPP destination masking without changing architectural wave state.
template <typename Inst>
inline uint64_t execution_lane_mask(const Inst &inst, const amdgpu::Wavefront &wf) {
  uint64_t exec = wf.exec();
  if constexpr (requires {
                  inst.inst_.src0;
                  inst.dpp_ctrl_;
                  inst.dpp_row_mask_;
                  inst.dpp_bank_mask_;
                  inst.dpp_bound_ctrl_;
                }) {
    if (inst.inst_.src0 == amdgpu::SRC_DPP)
      exec &= dpp_write_mask(wf.wf_size(), inst.dpp_ctrl_, inst.dpp_row_mask_, inst.dpp_bank_mask_,
                             inst.dpp_bound_ctrl_);
  }
  return exec;
}

/// @brief Complete lane-access plan for a DPP source permutation.
struct DppAccessPlan {
  static constexpr int8_t kNoSourceLane = -1;

  uint64_t source_lane_mask = 0;
  std::array<int8_t, 64> source_lane_for_destination{};

  DppAccessPlan() { source_lane_for_destination.fill(kNoSourceLane); }
};

inline uint32_t dpp8_src_lane(uint32_t lane, uint32_t lane_sel);

inline uint8_t true16_source_byte_mask(uint32_t opsel, uint32_t source_index) {
  return (opsel & (1u << source_index)) ? rocjitsu::ExecutionPlugin::kHighHalfByteMask
                                        : rocjitsu::ExecutionPlugin::kLowHalfByteMask;
}

inline DppAccessPlan make_dpp_access_plan(uint32_t wf_size, uint32_t dpp_ctrl, uint32_t row_mask,
                                          uint32_t bank_mask, uint32_t bound_ctrl, uint32_t fi,
                                          uint64_t exec_mask) {
  DppAccessPlan plan;
  for (uint32_t lane = 0; lane < wf_size; ++lane) {
    const uint64_t lane_bit = uint64_t{1} << lane;
    if ((exec_mask & lane_bit) == 0 ||
        !dpp_lane_write_enabled(static_cast<int>(lane), static_cast<int>(wf_size), dpp_ctrl,
                                row_mask, bank_mask, bound_ctrl))
      continue;

    bool out_of_bounds = false;
    const int source_lane =
        dpp_permute(dpp_ctrl, static_cast<int>(lane), static_cast<int>(wf_size), out_of_bounds);
    if (out_of_bounds || (!fi && (exec_mask & (uint64_t{1} << source_lane)) == 0))
      continue;
    plan.source_lane_for_destination[lane] = static_cast<int8_t>(source_lane);
    plan.source_lane_mask |= uint64_t{1} << source_lane;
  }
  return plan;
}

inline DppAccessPlan make_dpp8_access_plan(uint32_t wf_size, uint32_t lane_sel, uint32_t fi,
                                           uint64_t exec_mask) {
  DppAccessPlan plan;
  for (uint32_t lane = 0; lane < wf_size; ++lane) {
    if ((exec_mask & (uint64_t{1} << lane)) == 0)
      continue;
    const uint32_t source_lane = dpp8_src_lane(lane, lane_sel);
    if (source_lane >= wf_size || (!fi && (exec_mask & (uint64_t{1} << source_lane)) == 0))
      continue;
    plan.source_lane_for_destination[lane] = static_cast<int8_t>(source_lane);
    plan.source_lane_mask |= uint64_t{1} << source_lane;
  }
  return plan;
}

inline void stage_dpp_operand(Operand *&src0, const DppAccessPlan &plan,
                              std::unique_ptr<StagedOperand> &storage, amdgpu::Wavefront &wf,
                              uint8_t source_byte_mask = 0) {
  RegisterAccess regs(wf);
  if (src0->size_bits_ > 32) {
    auto src_view = regs.read_operand64(*src0, plan.source_lane_mask);
    uint64_t result[64] = {};
    for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
      const int source_lane = plan.source_lane_for_destination[lane];
      if (source_lane >= 0)
        result[lane] = src_view.lane(static_cast<uint32_t>(source_lane));
    }
    storage = std::make_unique<StagedOperand>(*src0, result, static_cast<int>(wf.wf_size()));
  } else {
    if (source_byte_mask == 0)
      source_byte_mask = src0->size_bits_ == 16 ? rocjitsu::ExecutionPlugin::kLowHalfByteMask
                                                : rocjitsu::ExecutionPlugin::kFullByteMask;
    auto src_view = regs.read_operand(*src0, plan.source_lane_mask, source_byte_mask);
    uint32_t result[64] = {};
    for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
      const int source_lane = plan.source_lane_for_destination[lane];
      if (source_lane >= 0)
        result[lane] = src_view.lane(static_cast<uint32_t>(source_lane));
    }
    storage = std::make_unique<StagedOperand>(*src0, result, static_cast<int>(wf.wf_size()));
  }
  src0 = storage.get();
}

/// @brief Pre-permute src0 for a DPP instruction.
///
/// Reads all src0 VGPR lanes, applies the DPP permutation, creates a
/// StagedOperand with the permuted data, and swaps the src0 pointer.
/// Called from VOP1/VOP2 execute_impl() when src0 == 250.
///
/// @param[in,out] src0 Source operand pointer to replace.
/// @param dpp_ctrl 9-bit DPP control value.
/// @param row_mask 4-bit row mask.
/// @param bank_mask 4-bit bank mask.
/// @param bound_ctrl Bound control (1 = zero OOB, 0 = preserve).
/// @param fi Fetch-inactive control (1 = read inactive source lanes, 0 = zero).
/// @param[out] storage Owning pointer for the staged operand lifetime.
/// @param wf Wavefront providing register state.
inline void apply_dpp(Operand *&src0, uint32_t dpp_ctrl, uint32_t row_mask, uint32_t bank_mask,
                      uint32_t bound_ctrl, uint32_t fi, std::unique_ptr<StagedOperand> &storage,
                      amdgpu::Wavefront &wf, uint8_t source_byte_mask = 0) {
  const DppAccessPlan plan =
      make_dpp_access_plan(wf.wf_size(), dpp_ctrl, row_mask, bank_mask, bound_ctrl, fi, wf.exec());
  stage_dpp_operand(src0, plan, storage, wf, source_byte_mask);
}

inline uint32_t dpp8_src_lane(uint32_t lane, uint32_t lane_sel) {
  uint32_t sel = (lane_sel >> ((lane & 7u) * 3u)) & 7u;
  return (lane & ~7u) | sel;
}

inline void apply_dpp8(Operand *&src0, uint32_t lane_sel, uint32_t fi,
                       std::unique_ptr<StagedOperand> &storage, amdgpu::Wavefront &wf,
                       uint8_t source_byte_mask = 0) {
  const DppAccessPlan plan = make_dpp8_access_plan(wf.wf_size(), lane_sel, fi, wf.exec());
  stage_dpp_operand(src0, plan, storage, wf, source_byte_mask);
}

} // namespace dpp

namespace sdwa {

/// @brief Return the architectural source bytes selected by an SDWA selector.
inline uint8_t sdwa_src_byte_mask(uint32_t sel) {
  if (sel <= BYTE_3)
    return uint8_t{1} << sel;
  if (sel == WORD_0)
    return 0b0011;
  if (sel == WORD_1)
    return 0b1100;
  return rocjitsu::ExecutionPlugin::kFullByteMask;
}

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

inline uint8_t sdwa_dst_byte_mask(uint32_t dst_sel, uint32_t dst_unused) {
  if (dst_sel == DWORD || dst_unused != UNUSED_PRESERVE)
    return rocjitsu::ExecutionPlugin::kFullByteMask;
  return sdwa_src_byte_mask(dst_sel);
}

inline uint32_t sdwa_clamp_f32(uint32_t result);

/// @brief Whether a generated SIMD path can store its result without a
/// destination transform.
template <typename Inst> inline bool supports_direct_simd_store(const Inst &inst) {
  if constexpr (requires {
                  inst.inst_.src0;
                  inst.sdwa_dst_sel_;
                  inst.sdwa_clamp_;
                }) {
    return inst.inst_.src0 != amdgpu::SRC_SDWA ||
           (inst.sdwa_dst_sel_ == DWORD && !inst.sdwa_clamp_);
  }
  return true;
}

/// @brief Store one semantic result with destination modifiers applied.
///
/// Destination preservation and optional clamp are part of one architectural
/// write.
template <bool ApplyFloatClamp, typename Inst, typename Op>
inline void write_lane(Inst &inst, amdgpu::Wavefront &wf, const Op &op, uint32_t lane,
                       uint32_t value) {
  if constexpr (requires {
                  inst.inst_.src0;
                  inst.sdwa_dst_sel_;
                  inst.sdwa_dst_unused_;
                  inst.sdwa_clamp_;
                  inst.dst_operand(0);
                }) {
    if (inst.inst_.src0 == amdgpu::SRC_SDWA && op.is_vgpr() &&
        inst.dst_operand(0) == static_cast<const Operand *>(&op)) {
      const bool clamp = ApplyFloatClamp && inst.sdwa_clamp_;
      const uint8_t update_byte_mask =
          sdwa_dst_byte_mask(inst.sdwa_dst_sel_, inst.sdwa_dst_unused_);
      const uint8_t observed_byte_mask =
          clamp ? rocjitsu::ExecutionPlugin::kFullByteMask : update_byte_mask;
      const uint32_t placed = sdwa_dst_merge(value, 0, inst.sdwa_dst_sel_, inst.sdwa_dst_unused_);
      amdgpu::RegisterAccess(wf).write_lane_masked(op, lane, placed, update_byte_mask,
                                                   observed_byte_mask,
                                                   clamp ? &sdwa_clamp_f32 : nullptr);
      return;
    }
  }
  amdgpu::RegisterAccess(wf).write_lane(op, lane, value);
}

template <bool ApplyFloatClamp, typename Inst, typename Op>
inline void write_lane64(Inst &inst, amdgpu::Wavefront &wf, const Op &op, uint32_t lane,
                         uint64_t value) {
  (void)ApplyFloatClamp;
  (void)inst;
  amdgpu::RegisterAccess(wf).write_lane64(op, lane, value);
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
