// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/register_access.h"

// Keep all four inline observer paths live in a separately loaded module. The
// loader test passes null; RTLD_NOW must still resolve every referenced helper.
extern "C" RJ_API_EXPORT void rj_probe_register_observer(rocjitsu::amdgpu::Wavefront *wf,
                                                         uint32_t sgpr, uint32_t vgpr) {
  if (!wf)
    return;
  const rocjitsu::amdgpu::RegisterAccess registers(*wf);
  registers.write_sgpr(sgpr, registers.read_sgpr(sgpr));
  const auto value = registers.read_vgpr_region(vgpr, 1, 1).lane(0, 0);
  registers.write_vgpr_region(vgpr, 1, 1).set_lane(0, 0, value);
}
