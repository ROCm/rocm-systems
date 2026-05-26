// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file execution_plugin_group.h
/// @brief Collection of plugins that delegates to each member.

#pragma once

#include "plugins/execution_plugin.h"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace rocjitsu {

/// @brief Collection of plugins that delegates to each member.
///
/// Dispatches directly to plugins with no overhead. Subclass
/// ProfiledExecutionPluginGroup adds adaptive-sampling timing.
class ExecutionPluginGroup {
public:
  ExecutionPluginGroup() = default;
  virtual ~ExecutionPluginGroup() = default;

  bool add(std::unique_ptr<ExecutionPlugin> p) {
    if (!p)
      return false;
    for (const auto &existing : plugins_)
      if (existing->name() == p->name())
        return false;
    p->slot_index_ = static_cast<uint32_t>(plugins_.size());
    plugins_.push_back(std::move(p));
    return true;
  }

  uint32_t num_plugins() const { return static_cast<uint32_t>(plugins_.size()); }
  bool empty() const { return plugins_.empty(); }

  // -- Lifecycle (non-virtual) --
  void onInit() {
    for (auto &p : plugins_)
      p->onInit();
  }

  void onShutdown() {
    for (auto &p : plugins_)
      p->onShutdown();
  }

  // -- RISC-V (non-virtual) --
  void onRiscvExecuteInstruction(uint64_t pc, const Instruction &inst) {
    for (auto &p : plugins_)
      p->onRiscvExecuteInstruction(pc, inst);
  }

  // -- AMDGPU (virtual) --
  virtual void beforeAmdgpuExecuteInstruction(uint64_t pc, const Instruction &inst,
                                              amdgpu::Wavefront &wf) {
    for (auto &p : plugins_)
      p->beforeAmdgpuExecuteInstruction(pc, inst, wf);
  }

  virtual void afterAmdgpuExecuteInstruction(uint64_t pc, const Instruction &inst,
                                             amdgpu::Wavefront &wf) {
    for (auto &p : plugins_)
      p->afterAmdgpuExecuteInstruction(pc, inst, wf);
  }

  virtual void onAmdgpuRouteMemoryInstruction(const Instruction &inst, amdgpu::Wavefront &wf) {
    for (auto &p : plugins_)
      p->onAmdgpuRouteMemoryInstruction(inst, wf);
  }

  virtual void onAmdgpuDispatchPacketProcessed(const KernelDispatchInfo &info) {
    for (auto &p : plugins_)
      p->onAmdgpuDispatchPacketProcessed(info);
  }

  virtual void onAmdgpuDispatchExecutionBegin(uint32_t dispatch_id) {
    for (auto &p : plugins_)
      p->onAmdgpuDispatchExecutionBegin(dispatch_id);
  }

  virtual void onAmdgpuDispatchExecutionEnd(uint32_t dispatch_id) {
    for (auto &p : plugins_)
      p->onAmdgpuDispatchExecutionEnd(dispatch_id);
  }

  virtual void onAmdgpuWorkgroupDispatched(uint32_t wg_id, uint32_t dispatch_id,
                                           uint32_t vgpr_count, uint32_t sgpr_count,
                                           std::span<amdgpu::Wavefront *> wavefronts) {
    for (auto &p : plugins_)
      p->onAmdgpuWorkgroupDispatched(wg_id, dispatch_id, vgpr_count, sgpr_count, wavefronts);
  }

  virtual void onAmdgpuWorkgroupCompleted(uint32_t dispatch_id, uint32_t wg_id) {
    for (auto &p : plugins_)
      p->onAmdgpuWorkgroupCompleted(dispatch_id, wg_id);
  }

  virtual void onAmdgpuWavefrontDispatched(amdgpu::Wavefront &wf) {
    for (auto &p : plugins_)
      p->onAmdgpuWavefrontDispatched(wf);
  }

  virtual void onAmdgpuWavefrontHalted(amdgpu::Wavefront &wf) {
    for (auto &p : plugins_)
      p->onAmdgpuWavefrontHalted(wf);
  }

  virtual void onAmdgpuReadVgpr(amdgpu::Wavefront *wf, uint32_t physical_reg, uint32_t lane,
                                uint8_t byteMask = 0xF) {
    for (auto &p : plugins_)
      p->onAmdgpuReadVgpr(wf, physical_reg, lane, byteMask);
  }

  virtual void onAmdgpuReadSgpr(amdgpu::Wavefront *wf, uint32_t physical_reg) {
    for (auto &p : plugins_)
      p->onAmdgpuReadSgpr(wf, physical_reg);
  }

  virtual void onAmdgpuBarrierResolved(std::span<amdgpu::Wavefront *> wavefronts) {
    for (auto &p : plugins_)
      p->onAmdgpuBarrierResolved(wavefronts);
  }

  /// A shared empty group used as the default when no plugins are attached.
  static std::shared_ptr<ExecutionPluginGroup> empty_group() {
    static auto instance = std::make_shared<ExecutionPluginGroup>();
    return instance;
  }

private:
  std::vector<std::unique_ptr<ExecutionPlugin>> plugins_;
};

} // namespace rocjitsu
