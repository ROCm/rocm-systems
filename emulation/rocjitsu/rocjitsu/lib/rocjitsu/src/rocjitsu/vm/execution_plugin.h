// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file execution_plugin.h
/// @brief Architecture-neutral plugin interface for execution hooks.
///
/// A single plugin class serves both AMDGPU and RISC-V execution paths.
/// Hooks are prefixed with their architecture name (onAmdgpu*, onRiscv*).

#pragma once

#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace rocjitsu {

/// @brief Abstract plugin interface for execution hooks.
///
/// Plugins receive callbacks at key points during simulation execution.
/// All hooks have empty default implementations so plugins only override
/// what they need. The no-plugin path has near-zero overhead (the
/// default empty group's delegation loops iterate an empty vector).
///
/// Ownership: plugins are owned by an ExecutionPluginGroup via
/// unique_ptr. The group itself is shared (via shared_ptr) between
/// the SoC and all components that fire hooks (CommandProcessor,
/// ComputeUnit, Hart). A static empty group is used as the default
/// so the plugin group pointer is never null.
class ExecutionPlugin {
public:
  virtual ~ExecutionPlugin() = default;

  // -- AMDGPU hooks --------------------------------------------------------

  /// Called before every AMDGPU instruction is executed.
  virtual void onAmdgpuExecuteInstruction(uint64_t /*pc*/, const Instruction & /*inst*/) {}

  /// Called when an AMDGPU memory instruction is routed to a pipeline.
  virtual void onAmdgpuRouteMemoryInstruction(const Instruction & /*inst*/) {}

  /// Called when a new AMDGPU kernel dispatch begins.
  virtual void onAmdgpuKernelDispatch(uint64_t /*kernel_object*/, uint64_t /*entry_pc*/) {}

  /// Called after a workgroup's wavefronts have been dispatched to a CU.
  virtual void onAmdgpuDispatchWorkgroup(uint32_t /*wg_id*/, uint32_t /*vgpr_count*/,
                                         uint32_t /*sgpr_count*/,
                                         std::span<amdgpu::Wavefront *> /*wavefronts*/) {}

  /// Called when a VGPR is read during instruction execution.
  /// Not yet wired — to be connected when race detection lands.
  virtual void onAmdgpuReadVgpr(amdgpu::Wavefront * /*wf*/, uint32_t /*logical_reg*/,
                                uint32_t /*lane*/) {}

  /// Called when an SGPR is read during instruction execution.
  /// Not yet wired — to be connected when race detection lands.
  virtual void onAmdgpuReadSgpr(amdgpu::Wavefront * /*wf*/, uint32_t /*logical_reg*/) {}

  /// Called when s_waitcnt sets counter thresholds.
  virtual void onAmdgpuSetWaitTarget(amdgpu::Wavefront * /*wf*/, int /*vmcnt*/, int /*lgkmcnt*/) {}

  /// Called when all waves in a workgroup have reached s_barrier.
  virtual void onAmdgpuBarrierResolved(uint32_t /*wg_id*/) {}

  // -- RISC-V hooks --------------------------------------------------------

  /// Called before every RISC-V instruction is executed.
  virtual void onRiscvExecuteInstruction(uint64_t /*pc*/, const Instruction & /*inst*/) {}
};

/// @brief Collection of plugins that delegates to each member.
class ExecutionPluginGroup {
public:
  void add(std::unique_ptr<ExecutionPlugin> p) {
    assert(p && "cannot add a null plugin");
    plugins_.push_back(std::move(p));
  }

  // -- AMDGPU --
  void onAmdgpuExecuteInstruction(uint64_t pc, const Instruction &inst) {
    for (auto &p : plugins_)
      p->onAmdgpuExecuteInstruction(pc, inst);
  }

  void onAmdgpuRouteMemoryInstruction(const Instruction &inst) {
    for (auto &p : plugins_)
      p->onAmdgpuRouteMemoryInstruction(inst);
  }

  void onAmdgpuKernelDispatch(uint64_t kernel_object, uint64_t entry_pc) {
    for (auto &p : plugins_)
      p->onAmdgpuKernelDispatch(kernel_object, entry_pc);
  }

  void onAmdgpuDispatchWorkgroup(uint32_t wg_id, uint32_t vgpr_count, uint32_t sgpr_count,
                                 std::span<amdgpu::Wavefront *> wavefronts) {
    for (auto &p : plugins_)
      p->onAmdgpuDispatchWorkgroup(wg_id, vgpr_count, sgpr_count, wavefronts);
  }

  void onAmdgpuReadVgpr(amdgpu::Wavefront *wf, uint32_t logical_reg, uint32_t lane) {
    for (auto &p : plugins_)
      p->onAmdgpuReadVgpr(wf, logical_reg, lane);
  }

  void onAmdgpuReadSgpr(amdgpu::Wavefront *wf, uint32_t logical_reg) {
    for (auto &p : plugins_)
      p->onAmdgpuReadSgpr(wf, logical_reg);
  }

  void onAmdgpuSetWaitTarget(amdgpu::Wavefront *wf, int vmcnt, int lgkmcnt) {
    for (auto &p : plugins_)
      p->onAmdgpuSetWaitTarget(wf, vmcnt, lgkmcnt);
  }

  void onAmdgpuBarrierResolved(uint32_t wg_id) {
    for (auto &p : plugins_)
      p->onAmdgpuBarrierResolved(wg_id);
  }

  // -- RISC-V --
  void onRiscvExecuteInstruction(uint64_t pc, const Instruction &inst) {
    for (auto &p : plugins_)
      p->onRiscvExecuteInstruction(pc, inst);
  }

  bool empty() const { return plugins_.empty(); }

  /// A shared empty group used as the default when no plugins are attached.
  static std::shared_ptr<ExecutionPluginGroup> empty_group() {
    static auto instance = std::make_shared<ExecutionPluginGroup>();
    return instance;
  }

private:
  std::vector<std::unique_ptr<ExecutionPlugin>> plugins_;
};

} // namespace rocjitsu
