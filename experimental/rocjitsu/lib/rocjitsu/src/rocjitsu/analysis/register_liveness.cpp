// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file register_liveness.cpp
/// @brief Per-basic-block VGPR liveness analysis implementation.

#include "rocjitsu/analysis/register_liveness.h"

#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/isa/instruction.h"

#include <cassert>
#include <vector>

namespace rocjitsu {

namespace {

/// @brief Mark a range of VGPR indices in a live set.
inline void mark_range(VgprLiveSet &set, uint16_t base, uint16_t count) {
  for (uint16_t i = 0; i < count && base + i < kMaxVgprIndex; ++i)
    set.set(base + i);
}

/// @brief Clear a range of VGPR indices in a live set.
inline void clear_range(VgprLiveSet &set, uint16_t base, uint16_t count) {
  for (uint16_t i = 0; i < count && base + i < kMaxVgprIndex; ++i)
    set.reset(base + i);
}

} // namespace

RegisterLiveness RegisterLiveness::compute(BasicBlock &block) {
  RegisterLiveness result;

  struct InstInfo {
    uint64_t offset;
    const Instruction *inst;
  };
  std::vector<InstInfo> insts;
  uint64_t offset = block.start_offset();
  for (auto &inst : block.instructions()) {
    insts.push_back({offset, &inst});
    offset += inst.size();
  }

  // Intra-block liveness: the backward scan computes accurate gen/kill sets
  // within this block. Without CFG edges, registers that are live-out to
  // successor blocks are not tracked — find_free_run/find_free_sgpr callers
  // should search above the kernel's active register range (e.g., above
  // vdst + dst_vgprs) to avoid clobbering potentially live-out values.
  VgprLiveSet live;

  for (auto it = insts.rbegin(); it != insts.rend(); ++it) {
    const auto &[inst_offset, inst] = *it;

    for (int i = 0; i < inst->num_dst_operands(); ++i) {
      const auto *op = inst->dst_operand(i);
      if (op && op->is_vgpr())
        clear_range(live, op->unified_vgpr_index(), op->vgpr_count());
    }

    for (int i = 0; i < inst->num_src_operands(); ++i) {
      const auto *op = inst->src_operand(i);
      if (op && op->is_vgpr())
        mark_range(live, op->unified_vgpr_index(), op->vgpr_count());
    }

    result.live_sets_[inst_offset] = live;

    // Track max SGPR for conservative SGPR allocation.
    for (int i = 0; i < inst->num_src_operands(); ++i) {
      const auto *op = inst->src_operand(i);
      if (op) {
        int v = op->encoding_value();
        if (v >= 0 && v <= 105)
          result.max_sgpr_ = std::max(result.max_sgpr_, static_cast<uint16_t>(v));
      }
    }
    for (int i = 0; i < inst->num_dst_operands(); ++i) {
      const auto *op = inst->dst_operand(i);
      if (op) {
        int v = op->encoding_value();
        if (v >= 0 && v <= 105)
          result.max_sgpr_ = std::max(result.max_sgpr_, static_cast<uint16_t>(v));
      }
    }
  }

  return result;
}

const VgprLiveSet &RegisterLiveness::live_at(uint64_t offset) const {
  auto it = live_sets_.find(offset);
  return (it != live_sets_.end()) ? it->second : empty_set_;
}

bool RegisterLiveness::is_live(uint64_t offset, uint16_t vgpr_index) const {
  assert(vgpr_index < kMaxVgprIndex && "VGPR index out of range");
  return live_at(offset).test(vgpr_index);
}

std::optional<uint16_t> RegisterLiveness::find_free_run(uint64_t offset, uint16_t count,
                                                        uint16_t search_start) const {
  assert(count > 0 && "Must request at least one register");
  const auto &live = live_at(offset);

  for (uint16_t base = search_start; base + count <= kMaxVgprIndex; ++base) {
    bool all_free = true;
    for (uint16_t i = 0; i < count; ++i) {
      if (live.test(base + i)) {
        all_free = false;
        base = base + i;
        break;
      }
    }
    if (all_free)
      return base;
  }
  return std::nullopt;
}

std::optional<uint16_t> RegisterLiveness::find_free_sgpr_pair(uint16_t search_start) const {
  uint16_t base = std::max(search_start, static_cast<uint16_t>(max_sgpr_ + 1));
  if (base % 2 != 0)
    ++base; // even-align for s_mov_b64
  if (base + 1 < 106)
    return base;
  return std::nullopt;
}

std::optional<uint16_t> RegisterLiveness::find_free_sgpr(uint16_t search_start) const {
  uint16_t base = std::max(search_start, static_cast<uint16_t>(max_sgpr_ + 1));
  if (base < 106)
    return base;
  return std::nullopt;
}

} // namespace rocjitsu
