// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <optional>

namespace rocjitsu::amdgpu {

struct OrderedCountLanes {
  uint64_t contribution;
  uint64_t result;
};

// Streamout ordered operations issue ADD(0) on lane zero for an empty EXEC.
constexpr uint64_t streamout_register_lane(uint64_t exec, bool ordered) {
  return exec ? exec & -exec : uint64_t{ordered};
}

// GDS accepts one wave request even if neither mask has any lanes.
inline OrderedCountLanes ordered_count_lanes(const Wavefront &wf, uint32_t logical_dst) {
  const uint64_t exec = wf.exec();
  if (wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA3_5)
    return {exec & -exec, logical_dst < wf.num_vgprs() ? streamout_register_lane(exec, true) : 0};
  if (logical_dst >= wf.num_vgprs())
    return {};
  return {exec & (uint64_t{0} - exec), ~uint64_t{0} >> (64 - wf.wf_size())};
}

inline std::optional<OrderedCountLanes> decoded_ordered_count_lanes(const Instruction &inst,
                                                                    const Wavefront &wf) {
  const auto *issue = inst.amdgpu_memory_issue_info();
  if (issue && !issue->exec_masked && inst.mnemonic() == "ds_ordered_count") {
    const auto *dst = inst.dst_operand(0);
    if (dst)
      if (const auto reg = dst->decoded_vgpr())
        return ordered_count_lanes(wf, reg->index);
  }
  return std::nullopt;
}

} // namespace rocjitsu::amdgpu
