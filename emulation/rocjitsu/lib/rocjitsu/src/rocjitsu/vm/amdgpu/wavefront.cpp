// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/wavefront.h"

#include "rocjitsu/vm/amdgpu/compute_unit.h"

namespace rocjitsu {
namespace amdgpu {

void Wavefront::halt() {
  state_ = WfState::HALTED;
  cu_.release_wf(dispatch_id_, wg_id_);
}

} // namespace amdgpu
} // namespace rocjitsu
