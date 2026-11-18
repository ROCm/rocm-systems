// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MEMORY_RESULT_PREFLIGHT_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MEMORY_RESULT_PREFLIGHT_H_

#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <cstdint>

namespace rocjitsu::amdgpu::addr_calc {

/// CDNA5 and RDNA4 suppress operand reads for a load or returning atomic
/// whose destination is outside the wave allocation. The CU checks the full
/// result extent again before routing, after the instruction state exists.
inline bool nullify_unallocated_load_before_address(const Wavefront &wf, VectorMemState &state,
                                                    bool buffer_tfe = false) {
  // Generated block loads set is_block_transfer after address calculation, so
  // recognize their unique 32-DWORD shape here. Each enabled word has its own
  // allocation check at writeback.
  const bool block_shape =
      state.elem_size == sizeof(uint32_t) && state.num_elems == kBlockDwordCount;
  if (!state.is_load || state.lds_dst || state.is_block_transfer || block_shape)
    return false;
  const auto &allocation = wf.vgpr_alloc();
  const uint32_t words = state.destination_vgpr_count() + (state.buffer_tfe || buffer_tfe);
  if (allocation.contains(state.dst_reg_base, words) &&
      (!state.ds2_active || allocation.contains(state.ds2_dst_reg_base,
                                                state.ds2_destination_vgpr_count())))
    return false;
  state.nullified_result = true;
  state.exec_mask = state.lane_mask = 0;
  state.element_lane_masks.clear();
  return true;
}

} // namespace rocjitsu::amdgpu::addr_calc

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MEMORY_RESULT_PREFLIGHT_H_
