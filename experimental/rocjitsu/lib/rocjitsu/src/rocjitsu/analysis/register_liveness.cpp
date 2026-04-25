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

} // namespace rocjitsu
