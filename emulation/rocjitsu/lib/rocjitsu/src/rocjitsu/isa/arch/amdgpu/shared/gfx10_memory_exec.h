// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

namespace rocjitsu::amdgpu {

inline void prepare_buffer_tfe(Wavefront &wf, VectorMemState &d) {
  d.check_texel_residency = true;
  d.texel_status = std::make_unique<TexelStatus>();
  d.texel_status->report_failure = true;
  if (!wf.vgpr_alloc().contains(d.dst_reg_base, d.destination_vgpr_count() + 1)) {
    d.exec_mask = 0;
    d.lane_mask = 0;
    d.element_lane_masks.clear();
  }
}

// RDNA1/2 issue a memory instruction with an unallocated result tuple as if
// EXEC were zero. Running the ordinary body preserves counter obligations and
// wave-level operations such as DS_ORDERED_COUNT release with no active lanes.
template <typename Inst> void execute_gfx10_vector_memory(Inst &inst, Wavefront &wf) {
  if (wf.vgpr_destinations_allocated(inst)) {
    inst.execute_impl(wf);
    return;
  }
  class RestoreExec {
  public:
    explicit RestoreExec(Wavefront &wave) : wave_(wave), saved_(wave.exec_raw()) {
      wave_.set_exec_raw(0);
    }
    ~RestoreExec() { wave_.set_exec_raw(saved_); }

  private:
    Wavefront &wave_;
    uint64_t saved_;
  } restore(wf);
  inst.execute_impl(wf);
}

} // namespace rocjitsu::amdgpu
