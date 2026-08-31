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
#include "rocjitsu/isa/register_set.h"
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
/// default empty group returns before locking or dispatch).
///
/// Ownership: plugins are owned by an ExecutionPluginGroup via
/// unique_ptr. The group itself is shared (via shared_ptr) between
/// the SoC and all components that fire hooks (CommandProcessor,
/// ComputeUnit, Hart). A static empty group is used as the default
/// so the plugin group pointer is never null.
///
/// Lifecycle callbacks are host-ordered, not part of the hot/infrequent hook
/// locking policy: onInit() completes before simulation callbacks begin, and
/// onShutdown() begins only after all simulation callbacks have stopped. The
/// host must enforce this ordering; lifecycle callbacks do not synchronize
/// with concurrently executing simulation hooks.
class ExecutionPlugin {
public:
  static constexpr uint8_t kFullByteMask = 0xF;
  static constexpr uint8_t kLowHalfByteMask = 0b0011u;
  static constexpr uint8_t kHighHalfByteMask = 0b1100u;

  explicit ExecutionPlugin(std::string name) : name_(std::move(name)) {}
  virtual ~ExecutionPlugin() = default;

  const std::string &name() const { return name_; }

  /// Index into Wavefront::plugin_states_, assigned by the group on add().
  uint32_t slot_index() const { return slot_index_; }

  /// Output sink for this plugin. Use sink().write("msg") for all output.
  PluginSink &sink() { return *sink_; }

  /// Whether high-frequency instruction, memory-routing, and register callbacks
  /// must acquire the group's callback lock. By default, infrequent callbacks
  /// exclude only other infrequent callbacks; high-frequency callbacks may
  /// overlap both one another and an infrequent callback. Returning true makes
  /// all callbacks share the same lock. Override after identifying shared mutable
  /// state that cannot be protected within the plugin. The group samples this
  /// policy once when the plugin is added, so implementations must return a stable
  /// value from construction onward.
  virtual bool requires_serial_hot_hooks() const { return false; }

  // -- Lifecycle hooks ------------------------------------------------------

  /// Called when the emulated driver opens (simulation is ready to accept work).
  /// The host completes this callback before starting simulation callbacks.
  virtual void onInit() {}

  /// Called when the emulated driver closes (simulation is shutting down).
  /// All simulation state is still valid during this callback.
  /// The host starts this callback only after simulation callbacks have stopped.
  virtual void onShutdown() {}

  // -- AMDGPU hooks --------------------------------------------------------

  /// Called before every AMDGPU instruction is executed.
  /// Wavefront state reflects the state prior to the instruction's effects.
  /// May run concurrently across simulation partitions unless
  /// requires_serial_hot_hooks() returns true.
  virtual void onAmdgpuBeforeExecuteInstruction(uint64_t /*pc*/, const Instruction & /*inst*/,
                                                amdgpu::Wavefront & /*wf*/) {}

  /// Called after every AMDGPU instruction is executed.
  /// Wavefront state (wait targets, PC, etc.) reflects the instruction's effects.
  /// May run concurrently across simulation partitions unless
  /// requires_serial_hot_hooks() returns true.
  virtual void onAmdgpuAfterExecuteInstruction(uint64_t /*pc*/, const Instruction & /*inst*/,
                                               amdgpu::Wavefront & /*wf*/) {}

  /// Called when an AMDGPU memory instruction is routed to a pipeline.
  /// May run concurrently across simulation partitions unless
  /// requires_serial_hot_hooks() returns true.
  virtual void onAmdgpuRouteMemoryInstruction(const Instruction & /*inst*/,
                                              amdgpu::Wavefront & /*wf*/) {}

  /// Called when the command processor has parsed an AQL kernel dispatch packet
  /// and created a DispatchEntry. Fires during packet fetching, before any
  /// workgroups are placed. Multiple packets may be parsed in a single fetch.
  /// Infrequent hook; see the concurrency contract on requires_serial_hot_hooks().
  virtual void onAmdgpuDispatchPacketProcessed(const KernelDispatchInfo & /*info*/) {}

  /// Called when the command processor begins executing a dispatch — barriers
  /// are satisfied and workgroup placement is about to start.
  /// Infrequent hook; see the concurrency contract on requires_serial_hot_hooks().
  virtual void onAmdgpuDispatchExecutionBegin(uint32_t /*dispatch_id*/) {}

  /// Called when all workgroups of a dispatch have completed execution.
  /// Infrequent hook; see the concurrency contract on requires_serial_hot_hooks().
  virtual void onAmdgpuDispatchExecutionEnd(uint32_t /*dispatch_id*/) {}

  /// Called after a workgroup's wavefronts have been dispatched to a CU.
  /// @param physical_vgpr_count Physical VGPR allocation block size per wavefront.
  /// @param physical_sgpr_count Physical SGPR allocation block size per wavefront.
  /// Infrequent hook; see the concurrency contract on requires_serial_hot_hooks().
  virtual void onAmdgpuWorkgroupDispatched(uint32_t /*dispatch_id*/, uint32_t /*wg_id*/,
                                           uint32_t /*physical_vgpr_count*/,
                                           uint32_t /*physical_sgpr_count*/,
                                           std::span<amdgpu::Wavefront *> /*wavefronts*/) {}

  /// Called when the last wavefront of a workgroup has halted.
  /// Infrequent hook; see the concurrency contract on requires_serial_hot_hooks().
  virtual void onAmdgpuWorkgroupCompleted(uint32_t /*dispatch_id*/, uint32_t /*wg_id*/) {}

  /// Called after a wavefront is initialized and before its first instruction.
  /// Infrequent hook; see the concurrency contract on requires_serial_hot_hooks().
  virtual void onAmdgpuWavefrontDispatched(amdgpu::Wavefront & /*wf*/) {}

  /// Called when a wavefront halts, before its resources are freed.
  /// Infrequent hook; see the concurrency contract on requires_serial_hot_hooks().
  virtual void onAmdgpuWavefrontHalted(amdgpu::Wavefront & /*wf*/) {}

  /// Called when a VGPR is read during instruction execution.
  /// @param wf Owning wavefront, or nullptr if the register is unallocated.
  /// @param physical_reg Physical register index in the VGPR file.
  /// @param lane_mask Bit mask of lanes read by the instruction.
  /// @param byte_mask Sub-dword byte mask (kFullByteMask = full dword).
  /// May run concurrently across simulation partitions unless
  /// requires_serial_hot_hooks() returns true.
  virtual void onAmdgpuReadVgprLanes(const amdgpu::Wavefront * /*wf*/, uint32_t /*physical_reg*/,
                                     uint64_t /*lane_mask*/,
                                     uint8_t /*byte_mask*/ = kFullByteMask) {}

  /// Called when a VGPR is written during instruction execution.
  /// Memory pipeline completions write raw VGPR storage and do not fire this
  /// hook; this is for instruction-level destination writes.
  ///
  /// Internal storage operations do not produce additional architectural
  /// callbacks.
  /// @param wf Owning wavefront, or nullptr if the register is unallocated.
  /// @param physical_reg Physical register index in the VGPR file.
  /// @param lane_mask Bit mask of lanes written by the instruction.
  /// @param byte_mask Sub-dword byte mask (kFullByteMask = full dword).
  /// May run concurrently across simulation partitions unless
  /// requires_serial_hot_hooks() returns true.
  virtual void onAmdgpuWriteVgprLanes(const amdgpu::Wavefront * /*wf*/, uint32_t /*physical_reg*/,
                                      uint64_t /*lane_mask*/,
                                      uint8_t /*byte_mask*/ = kFullByteMask) {}

  /// Called when an SGPR is read during instruction execution.
  /// @param wf Owning wavefront, or nullptr if the register is unallocated.
  /// @param physical_reg Physical register index in the SGPR file.
  /// May run concurrently across simulation partitions unless
  /// requires_serial_hot_hooks() returns true.
  virtual void onAmdgpuReadSgpr(const amdgpu::Wavefront * /*wf*/, uint32_t /*physical_reg*/) {}

  /// Called when an architectural scalar register is read during instruction execution.
  /// @param wf Owning wavefront.
  /// @param reg Logical register identity. The class distinguishes ordinary
  ///        SGPRs from TTMPs without exposing their encoded selector values or
  ///        physical storage layout.
  /// May run concurrently across simulation partitions unless
  /// requires_serial_hot_hooks() returns true.
  virtual void onAmdgpuReadScalarRegister(const amdgpu::Wavefront *wf, RegisterRef reg) {
    if (!wf || reg.cls != RegClass::SGPR)
      return;
    for (uint32_t offset = 0; offset < reg.width; ++offset)
      onAmdgpuReadSgpr(wf, wf->sgpr_alloc().base + reg.index + offset);
  }

  /// Called when an architectural scalar register is written during instruction execution.
  /// Memory completion and runtime initialization remain unobserved storage operations.
  virtual void onAmdgpuWriteScalarRegister(const amdgpu::Wavefront * /*wf*/, RegisterRef /*reg*/) {}

  /// Called with the waves synchronized by a completed barrier domain.
  /// Infrequent hook; see the concurrency contract on requires_serial_hot_hooks().
  virtual void onAmdgpuBarrierResolved(std::span<amdgpu::Wavefront *> /*wavefronts*/) {}

  /// Whether this plugin needs typed scalar-register reads or legacy SGPR reads.
  /// The group samples this policy once when the plugin is added. The
  /// conservative default preserves existing plugin behavior: callbacks
  /// continue unless a plugin explicitly opts out.
  virtual bool observes_sgpr_reads() const { return true; }

private:
  friend class ExecutionPluginGroup;
  std::string name_;
  uint32_t slot_index_ = 0;
  PluginSink *sink_ = &StderrSink::instance();
};

} // namespace rocjitsu
