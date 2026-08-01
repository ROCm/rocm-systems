// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/plugins/logging/plugin.h"

#include <algorithm>
#include <format>

namespace rocjitsu {
namespace amdgpu {

KernelLoggingPlugin::KernelLoggingPlugin(const char * /*config_json*/)
    : ExecutionPlugin("logging") {}

KernelLoggingPlugin::~KernelLoggingPlugin() {}

void KernelLoggingPlugin::onAmdgpuDispatchPacketProcessed(const KernelDispatchInfo &info) {
  std::lock_guard<std::mutex> lock(mutex_);
  ++dispatch_count_;
  dispatch_progress_[info.dispatch_id] = {dispatch_count_, info.workgroup_count, 0};
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
  auto it = dispatch_progress_.find(dispatch_id);
  if (it != dispatch_progress_.end())
    sink().write(std::format("[rocjitsu] Kernel #{} execution begin\n", it->second.kernel_number));
}

void KernelLoggingPlugin::onAmdgpuWorkgroupCompleted(uint32_t dispatch_id, uint32_t /*wg_id*/) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = dispatch_progress_.find(dispatch_id);
  if (it == dispatch_progress_.end())
    return;

  auto &progress = it->second;
  ++progress.completed_workgroups;
  constexpr uint32_t kLargeDispatchWorkgroups = 8192;
  if (progress.total_workgroups < kLargeDispatchWorkgroups)
    return;

  const uint32_t interval = std::max(1u, progress.total_workgroups / 8);
  if (progress.completed_workgroups != 1 &&
      progress.completed_workgroups != progress.total_workgroups &&
      progress.completed_workgroups % interval != 0)
    return;

  const double percent = 100.0 * static_cast<double>(progress.completed_workgroups) /
                         static_cast<double>(progress.total_workgroups);
  sink().write(std::format("[rocjitsu] Kernel #{} progress {}/{} ({:.1f}%)\n",
                           progress.kernel_number, progress.completed_workgroups,
                           progress.total_workgroups, percent));
}

void KernelLoggingPlugin::onAmdgpuDispatchExecutionEnd(uint32_t dispatch_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = dispatch_progress_.find(dispatch_id);
  if (it != dispatch_progress_.end())
    sink().write(std::format("[rocjitsu] Kernel #{} execution end\n", it->second.kernel_number));
  dispatch_progress_.erase(dispatch_id);
  mfma_printed_.erase(dispatch_id);
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
