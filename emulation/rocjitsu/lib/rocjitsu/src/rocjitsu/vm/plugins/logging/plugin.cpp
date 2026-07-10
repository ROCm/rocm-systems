// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/plugins/logging/plugin.h"

#include <format>

namespace rocjitsu {
namespace amdgpu {

KernelLoggingPlugin::KernelLoggingPlugin() : ExecutionPlugin("logging") {}

KernelLoggingPlugin::~KernelLoggingPlugin() {}

void KernelLoggingPlugin::onAmdgpuDispatchPacketProcessed(const KernelDispatchInfo &info) {
  std::lock_guard<std::mutex> lock(mutex_);
  ++dispatch_count_;
  sink().write(std::format("\n[rocjitsu] Kernel #{} dispatch\n"
                           "  entry_pc={:#x}  grid=[{},{},{}]  wg=[{},{},{}]\n"
                           "  wgs={}  wfs/wg={}  sgprs={}  vgprs={}\n",
                           dispatch_count_, info.entry_pc, info.grid_size_x, info.grid_size_y,
                           info.grid_size_z, info.workgroup_size_x, info.workgroup_size_y,
                           info.workgroup_size_z, info.workgroup_count, info.wfs_per_workgroup,
                           info.sgprs_per_wf, info.vgprs_per_wf));
}

void KernelLoggingPlugin::onAmdgpuDispatchExecutionBegin(uint32_t dispatch_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  sink().write(std::format("[rocjitsu] dispatch {} execution begin\n", dispatch_id));
}

void KernelLoggingPlugin::onAmdgpuDispatchExecutionEnd(uint32_t dispatch_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  sink().write(std::format("[rocjitsu] dispatch {} execution end\n", dispatch_id));
}

void KernelLoggingPlugin::onAmdgpuWorkgroupCompleted(uint32_t dispatch_id, uint32_t wg_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  sink().write(std::format("[rocjitsu] dispatch {} workgroup {} completed\n", dispatch_id, wg_id));
}

void KernelLoggingPlugin::onAmdgpuWavefrontDispatched(Wavefront &wf) {
  std::lock_guard<std::mutex> lock(mutex_);
  sink().write(std::format("[rocjitsu] dispatch {} workgroup {} wave {} dispatched\n",
                           wf.dispatch_id(), wf.wg_id(), wf.wf_id()));
}

void KernelLoggingPlugin::onAmdgpuWavefrontHalted(Wavefront &wf) {
  std::lock_guard<std::mutex> lock(mutex_);
  sink().write(std::format("[rocjitsu] dispatch {} workgroup {} wave {} halted\n", wf.dispatch_id(),
                           wf.wg_id(), wf.wf_id()));
}

void KernelLoggingPlugin::onAmdgpuAfterExecuteInstruction(uint64_t /*pc*/, const Instruction &inst,
                                                          Wavefront &wf) {
  auto dispatch_id = wf.dispatch_id();
  bool is_mfma = inst.is_mfma() || inst.mnemonic().starts_with("v_wmma_");
  if (is_mfma) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (mfma_printed_.insert(dispatch_id).second)
      sink().write(std::format("[rocjitsu] mfma detected in dispatch {}\n", dispatch_id));
  }
}

} // namespace amdgpu
} // namespace rocjitsu

// extern "C" + raw new: needed for planned dlsym-based dynamic loading (#6628).
extern "C" rocjitsu::ExecutionPlugin *createKernelLoggingPlugin() {
  return new rocjitsu::amdgpu::KernelLoggingPlugin();
}
