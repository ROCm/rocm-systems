// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace rocjitsu {
class Instruction;
namespace amdgpu {
class Wavefront;

/// @brief Abstract plugin interface for execution hooks.
///
/// Plugins receive callbacks at key points during kernel dispatch and
/// execution. All hooks have empty default implementations.
class ExecutionPlugin {
public:
  virtual ~ExecutionPlugin() = default;

  /// Called after a workgroup's wavefronts have been dispatched to a CU.
  virtual void onWorkgroupDispatched(uint32_t /*wg_id*/, uint32_t /*lds_size*/,
                                     uint32_t /*vgpr_count*/,
                                     uint32_t /*sgpr_count*/,
                                     std::span<Wavefront *> /*wavefronts*/) {}

  /// Called when a memory instruction is routed to a pipeline.
  virtual void onMemoryInstruction(Instruction * /*inst*/, Wavefront & /*wf*/) {}

  /// Called when a VGPR is read during instruction execution.
  virtual void onVgprRead(Wavefront * /*wf*/, uint32_t /*logical_reg*/,
                          uint32_t /*lane*/) {}

  /// Called when an SGPR is read during instruction execution.
  virtual void onSgprRead(Wavefront * /*wf*/, uint32_t /*logical_reg*/) {}

  /// Called when s_waitcnt sets counter thresholds.
  virtual void onWaitcnt(Wavefront * /*wf*/, int /*vmcnt*/, int /*lgkmcnt*/) {}
};

/// @brief Collection of plugins that delegates to each member.
class ExecutionPluginGroup {
public:
  void add(std::unique_ptr<ExecutionPlugin> p) {
    plugins_.push_back(std::move(p));
  }

  void onWorkgroupDispatched(uint32_t wg_id, uint32_t lds_size,
                             uint32_t vgpr_count, uint32_t sgpr_count,
                             std::span<Wavefront *> wavefronts) {
    for (auto &p : plugins_)
      p->onWorkgroupDispatched(wg_id, lds_size, vgpr_count, sgpr_count,
                               wavefronts);
  }

  void onMemoryInstruction(Instruction *inst, Wavefront &wf) {
    for (auto &p : plugins_)
      p->onMemoryInstruction(inst, wf);
  }

  void onVgprRead(Wavefront *wf, uint32_t logical_reg, uint32_t lane) {
    for (auto &p : plugins_)
      p->onVgprRead(wf, logical_reg, lane);
  }

  void onSgprRead(Wavefront *wf, uint32_t logical_reg) {
    for (auto &p : plugins_)
      p->onSgprRead(wf, logical_reg);
  }

  void onWaitcnt(Wavefront *wf, int vmcnt, int lgkmcnt) {
    for (auto &p : plugins_)
      p->onWaitcnt(wf, vmcnt, lgkmcnt);
  }

  bool empty() const { return plugins_.empty(); }

private:
  std::vector<std::unique_ptr<ExecutionPlugin>> plugins_;
};

} // namespace amdgpu
} // namespace rocjitsu
