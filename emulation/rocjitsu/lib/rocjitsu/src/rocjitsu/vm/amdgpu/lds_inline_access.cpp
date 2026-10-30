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
  if (!cu.completes_lds_inline() || d.lds_dst)
    return false;
  d.wf_size = wf.wf_size();
  LocalMemPipeline::initiate_lds_access(d, wf, transpose_request_lane_mask(d, d.wf_size));
  const MemoryAccessCompletion completion =
      LocalMemPipeline::complete_lds_access(d, wf, MemoryAccessDeferredCompletion{});
  assert(completion == MemoryAccessCompletion::Complete);
  (void)completion;
  cu.note_lds_inline_completion(inst);
  return true;
}

} // namespace amdgpu
} // namespace rocjitsu
