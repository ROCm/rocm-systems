// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file execution_plugin.h
/// @brief Architecture-neutral plugin interface for execution hooks.
///
/// Hooks are prefixed with their target architecture name (onAmdgpu*).
///
/// Plugins produce diagnostic output via sink().write("message") rather
/// than writing to stderr directly. The sink is assigned by the
/// ExecutionPluginGroup when the plugin is added. See plugin_sink.h.

#pragma once

#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/plugins/kernel_dispatch_info.h"
#include "rocjitsu/vm/plugins/plugin_sink.h"
#include "rocjitsu/vm/plugins/wavefront_state.h"

#include <cstdint>
#include <span>
#include <string>

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
  static constexpr uint8_t kFullByteMask = 0xF;

  explicit ExecutionPlugin(std::string name) : name_(std::move(name)) {}
  virtual ~ExecutionPlugin() = default;

  const std::string &name() const { return name_; }

  /// Index into Wavefront::plugin_states_, assigned by the group on add().
  uint32_t slot_index() const { return slot_index_; }

  /// Output sink for this plugin. Use sink().write("msg") for all output.
  PluginSink &sink() { return *sink_; }

  // -- Lifecycle hooks ------------------------------------------------------

  /// Called when the emulated driver opens (simulation is ready to accept work).
  virtual void onInit() {}

  /// Called when the emulated driver closes (simulation is shutting down).
  /// All simulation state is still valid during this callback.
  virtual void onShutdown() {}

  // -- AMDGPU hooks --------------------------------------------------------

  /// Called before every AMDGPU instruction is executed.
  /// Wavefront state reflects the state prior to the instruction's effects.
  virtual void onAmdgpuBeforeExecuteInstruction(uint64_t /*pc*/, const Instruction & /*inst*/,
                                                amdgpu::Wavefront & /*wf*/) {}

  /// Called after every AMDGPU instruction is executed.
  /// Wavefront state (wait targets, PC, etc.) reflects the instruction's effects.
  virtual void onAmdgpuAfterExecuteInstruction(uint64_t /*pc*/, const Instruction & /*inst*/,
                                               amdgpu::Wavefront & /*wf*/) {}

  /// Called when an AMDGPU memory instruction is routed to a pipeline.
  virtual void onAmdgpuRouteMemoryInstruction(const Instruction & /*inst*/,
                                              amdgpu::Wavefront & /*wf*/) {}

  /// Called when the command processor has parsed an AQL kernel dispatch packet
  /// and created a DispatchEntry. Fires during packet fetching, before any
  /// workgroups are placed. Multiple packets may be parsed in a single fetch.
  virtual void onAmdgpuDispatchPacketProcessed(const KernelDispatchInfo & /*info*/) {}

  /// Called when the command processor begins executing a dispatch — barriers
  /// are satisfied and workgroup placement is about to start.
  virtual void onAmdgpuDispatchExecutionBegin(uint32_t /*dispatch_id*/) {}

  /// Called when all workgroups of a dispatch have completed execution.
  virtual void onAmdgpuDispatchExecutionEnd(uint32_t /*dispatch_id*/) {}

  /// Called after a workgroup's wavefronts have been dispatched to a CU.
  virtual void onAmdgpuWorkgroupDispatched(uint32_t /*dispatch_id*/, uint32_t /*wg_id*/,
                                           uint32_t /*vgpr_count*/, uint32_t /*sgpr_count*/,
                                           std::span<amdgpu::Wavefront *> /*wavefronts*/) {}

  /// Called when the last wavefront of a workgroup has halted.
  virtual void onAmdgpuWorkgroupCompleted(uint32_t /*dispatch_id*/, uint32_t /*wg_id*/) {}

  /// Called after a wavefront is initialized and before its first instruction.
  virtual void onAmdgpuWavefrontDispatched(amdgpu::Wavefront & /*wf*/) {}

  /// Called when a wavefront halts, before its resources are freed.
  virtual void onAmdgpuWavefrontHalted(amdgpu::Wavefront & /*wf*/) {}

  /// Called when a VGPR is read during instruction execution.
  /// @param wf Owning wavefront, or nullptr if the register is unallocated.
  /// @param physical_reg Physical register index in the VGPR file.
  /// @param lane_begin First lane in the read range.
  /// @param lane_end One past the last lane in the read range.
  /// @param byte_mask Sub-dword byte mask (kFullByteMask = full dword).
  virtual void onAmdgpuReadVgprs(const amdgpu::Wavefront * /*wf*/, uint32_t /*physical_reg*/,
                                 uint32_t /*lane_begin*/, uint32_t /*lane_end*/,
                                 uint8_t /*byte_mask*/ = kFullByteMask) {}

  /// Called when an SGPR is read during instruction execution.
  /// @param wf Owning wavefront, or nullptr if the register is unallocated.
  /// @param physical_reg Physical register index in the SGPR file.
  virtual void onAmdgpuReadSgpr(const amdgpu::Wavefront * /*wf*/, uint32_t /*physical_reg*/) {}

  /// Called when all waves in a workgroup have reached s_barrier.
  virtual void onAmdgpuBarrierResolved(std::span<amdgpu::Wavefront *> /*wavefronts*/) {}

private:
  friend class ExecutionPluginGroup;
  std::string name_;
  uint32_t slot_index_ = 0;
  PluginSink *sink_ = &StderrSink::instance();
};

} // namespace rocjitsu
