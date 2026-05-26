// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "plugins/kernel_logging_plugin.h"

namespace rocjitsu {
namespace amdgpu {

KernelLoggingPlugin::KernelLoggingPlugin() : ExecutionPlugin("logging") {}

KernelLoggingPlugin::~KernelLoggingPlugin() {}

void KernelLoggingPlugin::onAmdgpuDispatchPacketProcessed(const KernelDispatchInfo &info) {
  ++dispatch_count_;
  log("\n[rocjitsu] Kernel #%d dispatch\n"
      "  entry_pc=0x%lx  grid=[%u,%u,%u]  wg=[%u,%u,%u]\n"
      "  wgs=%u  wfs/wg=%u  sgprs=%u  vgprs=%u\n",
      dispatch_count_, info.entry_pc, info.grid_size_x, info.grid_size_y, info.grid_size_z,
      info.workgroup_size_x, info.workgroup_size_y, info.workgroup_size_z, info.workgroup_count,
      info.wfs_per_workgroup, info.sgprs_per_wf, info.vgprs_per_wf);
}

void KernelLoggingPlugin::onAmdgpuAfterExecuteInstruction(uint64_t /*pc*/, const Instruction &inst,
                                                          Wavefront &wf) {
  auto dispatch_id = wf.dispatch_id();
  bool is_mfma = inst.is_mfma() || inst.mnemonic().starts_with("v_wmma_");
  if (is_mfma && !mfma_printed_.contains(dispatch_id)) {
    mfma_printed_.insert(dispatch_id);
    log("[rocjitsu] mfma detected in dispatch %u\n", dispatch_id);
  }
}

} // namespace amdgpu
} // namespace rocjitsu
