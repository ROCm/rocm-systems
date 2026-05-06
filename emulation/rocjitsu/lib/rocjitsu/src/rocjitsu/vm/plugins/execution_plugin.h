// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file execution_plugin.h
/// @brief Architecture-neutral plugin interface for execution hooks.
///
/// A single plugin class serves both AMDGPU and RISC-V execution paths.
/// Hooks are prefixed with their architecture name (onAmdgpu*, onRiscv*).

#pragma once

#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/plugins/kernel_dispatch_info.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
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
  explicit ExecutionPlugin(std::string name) : name_(std::move(name)) {}
  virtual ~ExecutionPlugin() = default;

  const std::string &name() const { return name_; }

  // -- AMDGPU hooks --------------------------------------------------------

  /// Called before every AMDGPU instruction is executed.
  /// Wavefront state reflects the state prior to the instruction's effects.
  virtual void beforeAmdgpuExecuteInstruction(uint64_t /*pc*/, const Instruction& /*inst*/,
                                              amdgpu::Wavefront& /*wf*/) {}

  /// Called after every AMDGPU instruction is executed.
  /// Wavefront state (wait targets, PC, etc.) reflects the instruction's effects.
  virtual void afterAmdgpuExecuteInstruction(uint64_t /*pc*/, const Instruction& /*inst*/,
                                           amdgpu::Wavefront& /*wf*/) {}

  /// Called when an AMDGPU memory instruction is routed to a pipeline.
  virtual void onAmdgpuRouteMemoryInstruction(const Instruction & /*inst*/,
                                              amdgpu::Wavefront & /*wf*/) {}

  /// Called when a new AMDGPU kernel dispatch begins.
  virtual void onAmdgpuKernelDispatch(const KernelDispatchInfo & /*info*/) {}

  /// Called after a workgroup's wavefronts have been dispatched to a CU.
  virtual void onAmdgpuDispatchWorkgroup(
      uint32_t /*wg_id*/, uint32_t /*n_dispatched*/,
      uint32_t /*vgpr_count*/, uint32_t /*sgpr_count*/,
      std::span<amdgpu::Wavefront *> /*wavefronts*/) {}

  /// Called when a VGPR is read during instruction execution.
  /// @param wf Owning wavefront, or nullptr if the register is unallocated.
  /// @param physical_reg Physical register index in the VGPR file.
  /// @param lane Lane index within the wavefront.
  virtual void onAmdgpuReadVgpr(amdgpu::Wavefront * /*wf*/,
                                uint32_t /*physical_reg*/, uint32_t /*lane*/,
                                uint8_t /*byteMask*/ = 0xF) {}

  /// Called when an SGPR is read during instruction execution.
  /// @param wf Owning wavefront, or nullptr if the register is unallocated.
  /// @param physical_reg Physical register index in the SGPR file.
  virtual void onAmdgpuReadSgpr(amdgpu::Wavefront * /*wf*/,
                                uint32_t /*physical_reg*/) {}

  /// Called when all waves in a workgroup have reached s_barrier.
  virtual void onAmdgpuBarrierResolved(std::span<amdgpu::Wavefront *> /*wavefronts*/) {}

  // -- RISC-V hooks --------------------------------------------------------

  /// Called before every RISC-V instruction is executed.
  virtual void onRiscvExecuteInstruction(uint64_t /*pc*/, const Instruction& /*inst*/) {}

private:
  std::string name_;
};

/// @brief Collection of plugins that delegates to each member.
class ExecutionPluginGroup {
public:
  bool add(std::unique_ptr<ExecutionPlugin> p) {
    if (!p) return false;
    for (const auto &existing : plugins_)
      if (existing->name() == p->name()) return false;
    plugins_.push_back(std::move(p));
    return true;
  }

  // -- AMDGPU --
  void beforeAmdgpuExecuteInstruction(uint64_t pc, const Instruction &inst,
                                      amdgpu::Wavefront &wf) {
    for (auto &p : plugins_)
      p->beforeAmdgpuExecuteInstruction(pc, inst, wf);
  }

  void afterAmdgpuExecuteInstruction(uint64_t pc, const Instruction &inst,
                                   amdgpu::Wavefront &wf) {
    for (auto &p : plugins_)
      p->afterAmdgpuExecuteInstruction(pc, inst, wf);
  }

  void onAmdgpuRouteMemoryInstruction(const Instruction &inst,
                                       amdgpu::Wavefront &wf) {
    for (auto &p : plugins_)
      p->onAmdgpuRouteMemoryInstruction(inst, wf);
  }

  void onAmdgpuKernelDispatch(const KernelDispatchInfo &info) {
    for (auto &p : plugins_)
      p->onAmdgpuKernelDispatch(info);
  }

  void onAmdgpuDispatchWorkgroup(uint32_t wg_id, uint32_t n_dispatched,
                                   uint32_t vgpr_count, uint32_t sgpr_count,
                                   std::span<amdgpu::Wavefront *> wavefronts) {
    for (auto &p : plugins_)
      p->onAmdgpuDispatchWorkgroup(wg_id, n_dispatched, vgpr_count,
                                     sgpr_count, wavefronts);
  }

  void onAmdgpuReadVgpr(amdgpu::Wavefront *wf, uint32_t physical_reg,
                         uint32_t lane, uint8_t byteMask = 0xF) {
    for (auto &p : plugins_)
      p->onAmdgpuReadVgpr(wf, physical_reg, lane, byteMask);
  }

  void onAmdgpuReadSgpr(amdgpu::Wavefront *wf, uint32_t physical_reg) {
    for (auto &p : plugins_)
      p->onAmdgpuReadSgpr(wf, physical_reg);
  }

  void onAmdgpuBarrierResolved(std::span<amdgpu::Wavefront *> wavefronts) {
    for (auto &p : plugins_)
      p->onAmdgpuBarrierResolved(wavefronts);
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
