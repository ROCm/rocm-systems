// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file dpp_sdwa_ops.h
/// @brief DPP (Data-Parallel Primitives) and SDWA (Sub-Dword Access) helpers.
///
/// @details DPP permutes the field-bearing vector source of supported
/// VOP1/VOP2/VOP3/VOP3P/VOPC forms before execution. Source validity and
/// BOUND_CTRL govern source-derived writes, while row/bank masks independently
/// filter destination commits; suppressed compare-result bits are zeroed.
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
#include "util/except.h"
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace rocjitsu {

/// @brief Allocation-free operand backed by instruction-local staged lane values.
///
/// @details DPP and SDWA construct this object only when staging is required.
/// Its two inline VGPR-shaped buffers preserve native 32-bit and 64-bit SIMD
/// source loads without adding storage to every decoded instruction or making
/// a heap allocation on the execution path.
class StagedOperand final : public Operand {
public:
  static constexpr int MAX_LANES = 64;

  StagedOperand(const Operand &base, int lane_count)
      : Operand(base.size_bits_, base.encoding_value_), lane_count_(lane_count) {}

  StagedOperand(const Operand &base, const uint32_t *data, int lane_count)
      : StagedOperand(base, lane_count) {
    for (int lane = 0; lane < lane_count && lane < MAX_LANES; ++lane)
      set_lane(lane, data[lane]);
  }

  StagedOperand(const Operand &base, const uint64_t *data, int lane_count)
      : StagedOperand(base, lane_count) {
    for (int lane = 0; lane < lane_count && lane < MAX_LANES; ++lane)
      set_lane64(lane, data[lane]);
  }

  std::string name() const override { return "staged_src"; }
  bool simd_capable() const override { return true; }

  void set_lane(uint32_t lane, uint32_t value) { lo_[lane] = value; }
  void set_lane64(uint32_t lane, uint64_t value) {
    lo_[lane] = static_cast<uint32_t>(value);
    hi_[lane] = static_cast<uint32_t>(value >> 32);
  }

private:
  uint32_t read_lane(const amdgpu::Wavefront &, uint32_t lane) const override {
    return lane < static_cast<uint32_t>(lane_count_) ? lo_[lane] : 0;
  }

  uint64_t read_lane64(const amdgpu::Wavefront &, uint32_t lane) const override {
    if (lane >= static_cast<uint32_t>(lane_count_))
      return 0;
    return uint64_t{lo_[lane]} | (uint64_t{hi_[lane]} << 32);
  }

  uint32_t read_scalar(const amdgpu::Wavefront &) const override { return lo_[0]; }
  uint64_t read_scalar64(const amdgpu::Wavefront &) const override {
    return uint64_t{lo_[0]} | (uint64_t{hi_[0]} << 32);
  }

  void read_lane_chunk(const amdgpu::Wavefront &, uint32_t lane_base, uint32_t count,
                       uint32_t *out) const override {
    const uint32_t lanes = static_cast<uint32_t>(lane_count_);
    for (uint32_t i = 0; i < count; ++i) {
      const uint32_t lane = lane_base + i;
      out[i] = lane < lanes ? lo_[lane] : 0u;
    }
  }

  amdgpu::ConstVgprStorage simd_vgpr_storage_impl(const amdgpu::Wavefront &) const override {
    return {reinterpret_cast<const uint32_t *>(&lo_), MAX_LANES};
  }

  amdgpu::ConstVgprStoragePair64
  simd_vgpr_storage64_impl(const amdgpu::Wavefront &) const override {
    return {{reinterpret_cast<const uint32_t *>(&lo_), MAX_LANES},
            {reinterpret_cast<const uint32_t *>(&hi_), MAX_LANES}};
  }

  simdojo::VectorReg<MAX_LANES, uint32_t> lo_{};
  simdojo::VectorReg<MAX_LANES, uint32_t> hi_{};
  int lane_count_ = 0;
};

using DppOperand = StagedOperand;

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

/// @brief Execution-local DPP source and destination analysis.
///
/// Keeping these masks separate is essential: row/bank masking is a
/// destination rule, while BOUND_CTRL decides whether an invalid source writes
/// a zero-derived result or suppresses the write.
struct DppPlan {
  static constexpr uint8_t INVALID_LANE = 0xFF;

  std::array<uint8_t, 64> source_lanes{};
  uint64_t physical_read_dest_mask = 0;
  uint64_t zero_source_mask = 0;
  uint64_t source_write_mask = 0;
  uint64_t row_bank_mask = 0;
};

inline uint64_t dpp_row_bank_mask(uint32_t wf_size, uint32_t row_mask, uint32_t bank_mask) {
  uint64_t mask = 0;
  for (uint32_t lane = 0; lane < wf_size; ++lane)
    if (!dpp_lane_masked(static_cast<int>(lane), row_mask, bank_mask))
      mask |= uint64_t{1} << lane;
  return mask;
}

inline DppPlan make_dpp_plan(uint32_t wf_size, uint32_t dpp_ctrl, uint32_t row_mask,
                             uint32_t bank_mask, uint32_t bound_ctrl, uint32_t fi,
                             uint64_t exec_mask, bool inactive_uses_bound_ctrl) {
  DppPlan plan;
  plan.row_bank_mask = dpp_row_bank_mask(wf_size, row_mask, bank_mask);
  plan.source_lanes.fill(DppPlan::INVALID_LANE);
  for (uint32_t lane = 0; lane < wf_size; ++lane) {
    bool out_of_bounds = false;
    const int source_lane =
        dpp_permute(dpp_ctrl, static_cast<int>(lane), static_cast<int>(wf_size), out_of_bounds);
    // FI applies only to inactive in-range source lanes. Out-of-range sources
    // are governed by BOUND_CTRL alone, irrespective of FI.
    const bool inactive = !out_of_bounds && !fi && (exec_mask & (uint64_t{1} << source_lane)) == 0;
    const uint64_t lane_bit = uint64_t{1} << lane;
    if (!out_of_bounds)
      plan.source_lanes[lane] = static_cast<uint8_t>(source_lane);
    if (!out_of_bounds && !inactive)
      plan.physical_read_dest_mask |= lane_bit;
    else if (bound_ctrl || (inactive && !inactive_uses_bound_ctrl))
      plan.zero_source_mask |= lane_bit;
    if (bound_ctrl || (!out_of_bounds && !(inactive && inactive_uses_bound_ctrl)))
      plan.source_write_mask |= lane_bit;
  }
  return plan;
}

inline uint64_t dpp_physical_source_mask(const DppPlan &plan, uint64_t destination_mask,
                                         uint32_t wf_size) {
  uint64_t source_mask = 0;
  const uint64_t read_destinations = destination_mask & plan.physical_read_dest_mask;
  for (uint32_t lane = 0; lane < wf_size; ++lane)
    if (read_destinations & (uint64_t{1} << lane))
      source_mask |= uint64_t{1} << plan.source_lanes[lane];
  return source_mask;
}

inline uint64_t dpp_source_write_mask(uint32_t wf_size, uint32_t dpp_ctrl, uint32_t bound_ctrl,
                                      uint32_t fi, uint64_t exec_mask,
                                      bool inactive_uses_bound_ctrl) {
  return make_dpp_plan(wf_size, dpp_ctrl, 0xF, 0xF, bound_ctrl, fi, exec_mask,
                       inactive_uses_bound_ctrl)
      .source_write_mask;
}

/// Legacy combined mask retained for architectures without the compare-specific rule.
inline uint64_t dpp_write_mask(uint32_t wf_size, uint32_t dpp_ctrl, uint32_t row_mask,
                               uint32_t bank_mask, uint32_t bound_ctrl) {
  const DppPlan plan =
      make_dpp_plan(wf_size, dpp_ctrl, row_mask, bank_mask, bound_ctrl, 1, ~0ULL, false);
  return plan.row_bank_mask & plan.source_write_mask;
}

inline uint64_t dpp_compare_result(uint64_t new_result, uint64_t old_exec, uint64_t row_bank_mask,
                                   uint64_t source_write_mask) {
  return new_result & old_exec & row_bank_mask & source_write_mask;
}

/// Preserve an active scalar side-result lane when BOUND_CTRL suppresses its
/// DPP source write. Row/bank masks are intentionally absent: they select the
/// vector destination only, not VCC or another scalar side result.
inline uint64_t dpp_source_suppressed_result(uint64_t new_result, uint64_t old_result,
                                             uint64_t old_exec, uint64_t source_write_mask) {
  const uint64_t preserve_mask = old_exec & ~source_write_mask;
  return (new_result & ~preserve_mask) | (old_result & preserve_mask);
}

/// SIMD executes every active lane; architectural destination filtering occurs at commit.
template <typename Inst>
inline uint64_t execution_lane_mask(const Inst &, const amdgpu::Wavefront &wf) {
  return wf.exec();
}

/// @brief Temporarily restrict architectural and observed VGPR writes.
///
/// @details bind() intersects the requested lanes with the wavefront's current
/// write mask, so independently created scopes nest correctly. restore() puts
/// back the mask that was active at bind time and is idempotent; destruction
/// also restores it for exception-safe execution. Generated DPP code may commit
/// scalar side results while this scope is bound: scalar writes do not consult
/// vgpr_write_mask, whose row/bank and invalid-source filtering applies only to
/// the vector destination.
class ScopedVgprWriteMask {
public:
  ScopedVgprWriteMask() = default;
  ScopedVgprWriteMask(const ScopedVgprWriteMask &) = delete;
  ScopedVgprWriteMask &operator=(const ScopedVgprWriteMask &) = delete;
  ~ScopedVgprWriteMask() { restore(); }

  /// @brief Bind this scope to a wavefront and intersect its VGPR write mask.
  void bind(amdgpu::Wavefront &wf, uint64_t mask) {
    restore();
    wf_ = &wf;
    previous_ = wf.vgpr_write_mask();
    wf.set_vgpr_write_mask(previous_ & mask);
  }

  /// @brief Restore the mask saved by bind(), if this scope is bound.
  void restore() {
    if (!wf_)
      return;
    wf_->set_vgpr_write_mask(previous_);
    wf_ = nullptr;
  }

private:
  amdgpu::Wavefront *wf_ = nullptr;
  uint64_t previous_ = 0;
};

inline uint32_t dpp8_src_lane(uint32_t lane, uint32_t lane_sel);

inline uint8_t true16_source_byte_mask(uint32_t opsel, uint32_t source_index) {
  return (opsel & (1u << source_index)) ? rocjitsu::ExecutionPlugin::kHighHalfByteMask
                                        : rocjitsu::ExecutionPlugin::kLowHalfByteMask;
}

inline DppPlan make_dpp8_plan(uint32_t wf_size, uint32_t lane_sel, uint32_t fi,
                              uint64_t exec_mask) {
  DppPlan plan;
  plan.source_lanes.fill(DppPlan::INVALID_LANE);
  plan.row_bank_mask = ~0ULL;
  for (uint32_t lane = 0; lane < wf_size; ++lane) {
    const uint32_t source_lane = dpp8_src_lane(lane, lane_sel);
    const uint64_t lane_bit = uint64_t{1} << lane;
    if (source_lane < wf_size)
      plan.source_lanes[lane] = static_cast<uint8_t>(source_lane);
    if (source_lane < wf_size && (fi || (exec_mask & (uint64_t{1} << source_lane))))
      plan.physical_read_dest_mask |= lane_bit;
    else
      plan.zero_source_mask |= lane_bit;
    plan.source_write_mask |= lane_bit;
  }
  return plan;
}

inline void stage_dpp_operand(const Operand &source, const DppPlan &plan,
                              uint64_t read_destinations, std::optional<StagedOperand> &storage,
                              amdgpu::Wavefront &wf, uint8_t source_byte_mask = 0) {
  RegisterAccess regs(wf);
  const uint64_t source_lane_mask = dpp_physical_source_mask(plan, read_destinations, wf.wf_size());
  storage.emplace(source, static_cast<int>(wf.wf_size()));
  if (source.size_bits_ > 32) {
    auto src_view = regs.read_operand64(source, source_lane_mask);
    for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
      const uint64_t lane_bit = uint64_t{1} << lane;
      if (plan.physical_read_dest_mask & lane_bit)
        storage->set_lane64(lane, src_view.lane(plan.source_lanes[lane]));
      else if ((plan.zero_source_mask & lane_bit) == 0)
        storage->set_lane64(lane, src_view.lane(lane));
    }
  } else {
    if (source_byte_mask == 0)
      source_byte_mask = source.size_bits_ == 16 ? rocjitsu::ExecutionPlugin::kLowHalfByteMask
                                                 : rocjitsu::ExecutionPlugin::kFullByteMask;
    auto src_view = regs.read_operand(source, source_lane_mask, source_byte_mask);
    for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
      const uint64_t lane_bit = uint64_t{1} << lane;
      if (plan.physical_read_dest_mask & lane_bit)
        storage->set_lane(lane, src_view.lane(plan.source_lanes[lane]));
      else if ((plan.zero_source_mask & lane_bit) == 0)
        storage->set_lane(lane, src_view.lane(lane));
    }
  }
}

/// @brief Pre-permute src0 for a DPP instruction.
///
/// Reads all src0 VGPR lanes, applies the DPP permutation, creates a
/// StagedOperand with the permuted data, and returns it through storage.
/// Called by generated VOP1, VOP2, VOPC, VOP3, and VOP3P modifier paths when
/// src0 uses SRC_DPP.
///
/// @param source Source operand to stage; the reference is not retained or replaced.
/// @param plan Precomputed source-lane and destination-write decisions.
/// @param read_destinations Destination lanes whose permuted sources must be read.
/// @param[out] storage Instruction-local staged operand storage.
/// @param wf Wavefront providing register state.
/// @param source_byte_mask Optional byte mask for a true16 source selection.
inline void apply_dpp(const Operand &source, const DppPlan &plan, uint64_t read_destinations,
                      std::optional<StagedOperand> &storage, amdgpu::Wavefront &wf,
                      uint8_t source_byte_mask = 0) {
  stage_dpp_operand(source, plan, read_destinations, storage, wf, source_byte_mask);
}

inline uint32_t dpp8_src_lane(uint32_t lane, uint32_t lane_sel) {
  uint32_t sel = (lane_sel >> ((lane & 7u) * 3u)) & 7u;
  return (lane & ~7u) | sel;
}

inline void apply_dpp8(const Operand &source, uint32_t lane_sel, uint32_t fi,
                       std::optional<StagedOperand> &storage, amdgpu::Wavefront &wf,
                       uint8_t source_byte_mask = 0) {
  if (source.size_bits() > 32)
    throw util::InvalidInst("DPP8 requires a source no wider than 32 bits", "");
  const DppPlan plan = make_dpp8_plan(wf.wf_size(), lane_sel, fi, wf.exec());
  stage_dpp_operand(source, plan, wf.exec(), storage, wf, source_byte_mask);
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

/// @brief Append an SDWA source, including modifiers encoded in the extension word.
/// The semantic source format selects which modifier family is meaningful:
/// integer-like sources use sign extension, while floating-point sources use
/// negate and absolute-value modifiers.
inline void append_source(std::string &out, const Operand &source, SourceModifierFormat format,
                          bool sign_extend, bool negate, bool absolute) {
  if (format == SourceModifierFormat::NONE && sign_extend)
    out += "sext(";
  if (format != SourceModifierFormat::NONE && negate)
    out += '-';
  if (format != SourceModifierFormat::NONE && absolute)
    out += '|';
  out += source.name();
  if (format != SourceModifierFormat::NONE && absolute)
    out += '|';
  if (format == SourceModifierFormat::NONE && sign_extend)
    out += ')';
}

inline std::string selection_name(uint32_t selection) {
  constexpr std::array<std::string_view, 7> names = {"BYTE_0", "BYTE_1", "BYTE_2", "BYTE_3",
                                                     "WORD_0", "WORD_1", "DWORD"};
  if (selection < names.size())
    return std::string(names[selection]);
  return "invalid(" + std::to_string(selection) + ")";
}

inline std::string destination_unused_name(uint32_t unused) {
  constexpr std::array<std::string_view, 3> names = {"UNUSED_PAD", "UNUSED_SEXT",
                                                     "UNUSED_PRESERVE"};
  if (unused < names.size())
    return std::string(names[unused]);
  return "invalid(" + std::to_string(unused) + ")";
}

inline void append_source_attributes(std::string &out, uint32_t src0_selection, const Operand *src1,
                                     uint32_t src1_selection) {
  out += " src0_sel:";
  out += selection_name(src0_selection);
  if (src1) {
    out += " src1_sel:";
    out += selection_name(src1_selection);
  }
}

inline void append_destination_attributes(std::string &out, bool clamp, uint32_t omod,
                                          uint32_t destination_selection,
                                          uint32_t destination_unused, uint32_t src0_selection,
                                          const Operand *src1, uint32_t src1_selection) {
  if (clamp)
    out += " clamp";
  switch (omod) {
  case 1:
    out += " mul:2";
    break;
  case 2:
    out += " mul:4";
    break;
  case 3:
    out += " div:2";
    break;
  default:
    break;
  }
  out += " dst_sel:";
  out += selection_name(destination_selection);
  out += " dst_unused:";
  out += destination_unused_name(destination_unused);
  append_source_attributes(out, src0_selection, src1, src1_selection);
}

inline uint32_t apply_source_modifiers(uint32_t value, SourceModifierFormat format, bool negate,
                                       bool absolute) {
  uint32_t sign_bit = 0;
  switch (format) {
  case SourceModifierFormat::F16:
  case SourceModifierFormat::BF16:
    sign_bit = uint32_t{1} << 15;
    break;
  case SourceModifierFormat::F32:
    sign_bit = uint32_t{1} << 31;
    break;
  case SourceModifierFormat::NONE:
    return value;
  }

  if (absolute)
    value &= ~sign_bit;
  if (negate)
    value ^= sign_bit;
  return value;
}

/// @brief Stage one SDWA source for semantic execution.
///
/// Reads exactly the selected source bytes from active lanes, applies SDWA
/// selection/sign extension and source abs/neg modifiers, and installs fresh
/// instruction-owned storage for later delegate binding. Unmodified DWORD
/// sources need no staging and clear any stale storage left by an earlier
/// execution.
inline void stage_source(Operand &source, uint32_t selection, bool sign_extend, bool negate,
                         bool absolute, SourceModifierFormat modifier_format,
                         std::optional<StagedOperand> &storage, Wavefront &wf) {
  storage.reset();
  const bool has_float_modifier =
      modifier_format != SourceModifierFormat::NONE && (absolute || negate);
  if (selection == DWORD && !has_float_modifier)
    return;

  const uint64_t exec = wf.exec();
  const auto source_view =
      RegisterAccess(wf).read_operand(source, exec, sdwa_src_byte_mask(selection));
  storage.emplace(source, static_cast<int>(wf.wf_size()));
  for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
    if ((exec & (uint64_t{1} << lane)) == 0)
      continue;

    uint32_t value = sdwa_src_select(source_view.lane(lane), selection, sign_extend);
    value = apply_source_modifiers(value, modifier_format, negate, absolute);
    storage->set_lane(lane, value);
  }
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

inline uint32_t sdwa_clamp_f32(uint32_t result, const Wavefront &wf);

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
/// NaN bits are preserved unless MODE.DX10_CLAMP requests conversion to zero.
/// The caller determines whether the operation is float or integer based on
/// the instruction's semantic type.
inline uint32_t sdwa_clamp_f32(uint32_t result, const Wavefront &wf) {
  float f = std::bit_cast<float>(result);
  if (std::isnan(f))
    return wf.dx10_clamp() ? std::bit_cast<uint32_t>(0.0f) : result;
  f = std::fmin(std::fmax(f, 0.0f), 1.0f);
  return std::bit_cast<uint32_t>(f);
}

} // namespace sdwa

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_DPP_SDWA_OPS_H_
