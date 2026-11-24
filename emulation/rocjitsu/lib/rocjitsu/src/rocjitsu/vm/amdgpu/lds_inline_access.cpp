// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/lds_inline_access.h"

#include "rocjitsu/isa/arch/amdgpu/shared/ds_transpose.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/lds_inline_access_steps.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <cassert>

namespace rocjitsu {
namespace amdgpu {

bool LdsInlineAccess::complete(const Instruction &inst, VectorMemState &d, Wavefront &wf) {
  ComputeUnitCore &cu = wf.raw_cu();
  if (!cu.completes_lds_inline() || d.lds_dst || d.gds || d.gds_m0 || d.ordered_count ||
      d.ds_memviol)
    return false;
  if (d.is_load &&
      (!cu.owns_vgpr_range(wf, d.dst_reg_base, d.destination_vgpr_count()) ||
       (d.ds2_active &&
        !cu.owns_vgpr_range(wf, d.ds2_dst_reg_base, d.ds2_destination_vgpr_count()))))
    return false;
  d.wf_size = wf.wf_size();
  const VmAccessOutcome outcome =
      LocalMemPipeline::initiate_lds_access(d, wf, transpose_request_lane_mask(d, d.wf_size));
  if (outcome != VmAccessOutcome::Complete)
    return false;
  const MemoryAccessCompletion completion =
      LocalMemPipeline::complete_lds_access(d, wf, MemoryAccessCompletionFactory{});
  assert(completion == MemoryAccessCompletion::Complete);
  (void)completion;
  cu.note_lds_inline_completion(inst);
  return true;
}

} // namespace amdgpu
} // namespace rocjitsu
