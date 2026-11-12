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
#include "rocjitsu/vm/plugins/kernel_dispatch_info.h"
#include "rocjitsu/vm/plugins/memory_access_observation.h"
#include "rocjitsu/vm/plugins/plugin_sink.h"
#include "rocjitsu/vm/plugins/register_hazard_observation.h"
#include "rocjitsu/vm/plugins/tensor_dma_issue_observation.h"
#include "rocjitsu/vm/plugins/tensor_dma_memory_access_observation.h"
#include "rocjitsu/vm/plugins/wavefront_state.h"

#include <cstdint>
#include <span>
#include <string>

namespace rocjitsu {

namespace amdgpu {
class Wavefront;
}

/// The kind of barrier passed to ExecutionPlugin::onAmdgpuBarrierResolved.
enum class BarrierKind : uint8_t {
  /// A workgroup barrier or workgroup trap barrier. It resolves only after
  /// every wave of the workgroup has arrived or ended.
  Workgroup,
  /// A named barrier. It counts signals against a member count set by the
  /// program and notifies the waves that joined it.
  Named,
  /// A cluster barrier or cluster trap barrier. One signal from any wave of
  /// a workgroup counts for the whole workgroup.
  Cluster,
};

/// Optional notification when an implicit instruction wait retires its memory producers.
class AmdgpuImplicitMemoryWaitObserver {
public:
  virtual ~AmdgpuImplicitMemoryWaitObserver() = default;
  virtual void onAmdgpuImplicitMemoryWait(amdgpu::Wavefront &wf,
                                          const amdgpu::WaitFields &fields) = 0;
};

/// @brief Abstract plugin interface for execution hooks.
///
/// Plugins receive callbacks at key points during simulation execution.
/// Plugin callbacks, policy queries, construction and destruction must not throw.
/// Expected instruction failures use Wavefront::report_instruction_execution_error();
/// callbacks must still preserve issue/completion accounting and cancellation.
/// This is a semantic contract; callback types do not impose noexcept.
/// All hooks have empty default implementations. High-frequency hooks also
/// require an explicit interest declaration. The no-plugin path has near-zero overhead (the
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

  /// Whether high-frequency instruction, memory-routing, tensor-DMA, and register
  /// callbacks must acquire the group's callback lock. By default, infrequent callbacks
  /// exclude only other infrequent callbacks; high-frequency callbacks may
  /// overlap both one another and an infrequent callback. Returning true makes
  /// all callbacks share the same lock. Override after identifying shared mutable
  /// state that cannot be protected within the plugin. The group samples this
  /// policy once when the plugin is added, so implementations must return a stable
  /// value from construction onward.
  virtual bool requires_serial_hot_hooks() const { return false; }

  /// Whether this plugin consumes high-frequency callbacks for @p wf.
  ///
  /// Wavefronts are observed by default. A plugin may
  /// return false after its wavefront-dispatch callback has decided that the
  /// wave is outside its observation scope. The group samples the decision once
  /// after every plugin's wavefront-dispatch callback and reuses it until halt
  /// or cancellation, including rollback before a retry in the same slot.
  /// Reentrant hooks during dispatch, halt or cancellation, hooks without a
  /// wavefront, and hooks after live plugin-group replacement use a live query,
  /// so this method must remain lock-free and thread-safe. The decision must be
  /// stable between dispatch completion and the matching halt or cancellation.
  /// Lifecycle and dispatch/workgroup/wavefront callbacks are unaffected.
  virtual bool observes_hot_hooks_for_wavefront(const amdgpu::Wavefront * /*wf*/) const {
    return true;
  }

  /// High-frequency hooks are opt-in: override the corresponding interest
  /// method as well as the callback. Unrequested hooks do not enable observation
  /// work in the emulator. The group samples each interest once on add(), so
  /// values must be stable from construction onward. Lifecycle hooks need no
  /// interest declaration; wave selection is a separate filter above.
  virtual bool observes_before_execute_instruction() const { return false; }
  virtual bool observes_after_execute_instruction() const { return false; }
  virtual bool observes_async_instruction_issued() const { return false; }
  virtual bool observes_memory_instruction_routing() const { return false; }
  virtual bool observes_vgpr_reads() const { return false; }
  virtual bool observes_vgpr_writes() const { return false; }
  virtual bool observes_scalar_register_reads() const { return false; }
  virtual bool observes_scalar_register_writes() const { return false; }
  virtual bool observes_memory_routing() const { return false; }
  virtual bool observes_tensor_dma_memory_access() const { return false; }
  virtual bool observes_tensor_dma_issue() const { return false; }

  /// Request core register-dependency checking, independently of stderr warning
  /// settings. Findings retain pending dependencies for later consumers; the
  /// observer owns suppression. This does not request register-access hooks.
  virtual bool observes_register_hazards() const { return false; }

  /// Called on the issuer before the consuming instruction executes, including
  /// before its ordinary before-execute or async-issue notification. The record
  /// owns its instruction evidence; the wave is borrowed only for this callback.
  /// Does not apply the core stderr budget. The observer must not mutate the
  /// scoreboard or execute/replay instructions from this callback.
  virtual void onAmdgpuRegisterHazard(const amdgpu::RegisterHazardObservation & /*hazard*/,
                                      const amdgpu::Wavefront & /*wf*/) {}

  /// Whether this plugin tolerates asynchronous arithmetic. The default leaves
  /// offload enabled. Return false when inspecting synchronous instruction state
  /// or requiring the before/after pair for every instruction: any such plugin
  /// disables offload for its group. This capability does not subscribe to hooks.
  /// Instruction observers that support async must handle both accepted-issue
  /// and synchronous callbacks, since offload can fall back to synchronous execution.
  /// Sampled once on add(), with a stable value from construction onward.
  /// Register hooks may run on helpers concurrently with issuer hooks for the
  /// SAME wave, and may precede the issue notification. At issue, inspect only
  /// instruction/dispatch metadata and synchronized plugin state: even sources
  /// may alias a destination currently being written. Serializing hooks alone
  /// does not meet this contract.
  virtual bool supports_async_instructions() const { return true; }

  // -- Lifecycle hooks ------------------------------------------------------

  /// Called when the emulated driver opens (simulation is ready to accept work).
  /// The host completes this callback before starting simulation callbacks.
  virtual void onInit() {}

  /// Called when the emulated driver closes (simulation is shutting down).
  /// All simulation state is still valid during this callback.
  /// The host starts this callback only after simulation callbacks have stopped.
  virtual void onShutdown() {}

  // -- AMDGPU hooks --------------------------------------------------------

  /// Called before a synchronously executed AMDGPU instruction.
  /// Wavefront state reflects the state prior to the instruction's effects.
  /// Memory instructions expose their decoded issue metadata through
  /// Instruction::amdgpu_memory_issue_info() at this point.
  /// The CU supplies four DWORDs starting at the instruction PC, including
  /// words following a shorter instruction. For a GFX10 MIMG instruction
  /// with NSA=3, the window also contains the fifth DWORD. An empty span
  /// means no fetch window is available. The instruction, wave and fetch
  /// window are borrowed only for this callback.
  /// May run concurrently across simulation partitions unless
  /// requires_serial_hot_hooks() returns true.
  virtual void onAmdgpuBeforeExecuteInstruction(uint64_t /*pc*/, const Instruction & /*inst*/,
                                                amdgpu::Wavefront & /*wf*/,
                                                std::span<const uint32_t> /*fetch_window*/) {}

  /// Called after a synchronously executed AMDGPU instruction.
  /// Wavefront state (wait targets, PC, etc.) reflects the instruction's effects.
  /// May run concurrently across simulation partitions unless
  /// requires_serial_hot_hooks() returns true.
  virtual void onAmdgpuAfterExecuteInstruction(uint64_t /*pc*/, const Instruction & /*inst*/,
                                               amdgpu::Wavefront & /*wf*/) {}

  /// Called on the issuer when an async job is accepted, before PC advance.
  /// The job may already be running or finished; no register snapshot is safe
  /// to inspect here. Replaces the ordinary before/after pair.
  /// Inspect metadata only and do not retain the instruction or wave reference.
  virtual void onAmdgpuAsyncInstructionIssued(uint64_t /*pc*/, const Instruction & /*inst*/,
                                              amdgpu::Wavefront & /*wf*/) {}

  /// Called when an AMDGPU memory instruction is routed to a pipeline.
  /// Fires BEFORE routing decides anything: the instruction still carries the
  /// space it decoded as and the addresses it computed. For what the memory
  /// system will actually see, use onAmdgpuMemoryAccessRouted().
  /// May run concurrently across simulation partitions unless
  /// requires_serial_hot_hooks() returns true.
  virtual void onAmdgpuRouteMemoryInstruction(const Instruction & /*inst*/,
                                              amdgpu::Wavefront & /*wf*/) {}

  /// Called once routing has settled, with the pipeline the access was issued
  /// to and the addresses it was issued with. Fires for every routed memory
  /// instruction, including one no pipeline accepted, which is reported with
  /// an UNKNOWN route rather than dropped.
  ///
  /// The instruction and wave already reflect the selected route. They and
  /// the observation's execution-owned spans are borrowed only for this callback.
  /// May run concurrently across simulation partitions unless
  /// requires_serial_hot_hooks() returns true.
  virtual void onAmdgpuMemoryAccessRouted(const amdgpu::MemoryAccessObservation & /*access*/,
                                          const Instruction & /*inst*/,
                                          amdgpu::Wavefront & /*wf*/) {}

  /// Called after one tensor DMA instruction and any descriptor-requested
  /// atomic-barrier arrival return normally, when the transfer made at least one
  /// global request, so its transfer fields come from a validated descriptor.
  /// The observation contains only in-bounds global requests attempted, in
  /// execution order. It intentionally does not filter memory access outcomes,
  /// matching the observer boundary.
  /// The wave and deferred address view are borrowed only for this callback.
  /// May run concurrently across simulation partitions unless
  /// requires_serial_hot_hooks() returns true.
  virtual void
  onAmdgpuTensorDmaMemoryAccess(const amdgpu::TensorDmaMemoryAccessObservation & /*access*/,
                                const amdgpu::Wavefront & /*wf*/) {}

  /// Called once when a tensor DMA instruction issues, after its descriptor is
  /// read and, unless it is NULL, validated and prepared. A store has already
  /// sampled its LDS source bytes, so changing LDS in the callback does not
  /// change what it writes to global memory. No global request has been made
  /// and a load has written no LDS. This is the tensor DMA counterpart of
  /// onAmdgpuMemoryAccessRouted():
  /// it fires in the issuing wave's instruction step, also when the transfer
  /// later waits for memory, and also for an instruction that moves no data,
  /// such as a NULL descriptor, a NOP tile, a gather with no valid indices or a
  /// store wholly outside the tensor. A transfer deferred until its memory is
  /// available is not reported again when it resumes. The observation's LDS
  /// ranges are byte ranges that can end mid-element; see
  /// TensorDmaIssueObservation. They are
  /// borrowed and valid only during this callback.
  /// May run concurrently across simulation partitions unless
  /// requires_serial_hot_hooks() returns true.
  virtual void onAmdgpuTensorDmaIssued(const amdgpu::TensorDmaIssueObservation & /*issue*/,
                                       const amdgpu::Wavefront & /*wf*/) {}

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
  /// Cancelled dispatches do not receive this callback, even if
  /// onAmdgpuDispatchExecutionBegin was called.
  /// Infrequent hook; see the concurrency contract on requires_serial_hot_hooks().
  virtual void onAmdgpuDispatchExecutionEnd(uint32_t /*dispatch_id*/) {}

  /// Called after all of a workgroup's waves are initialized on a CU, before
  /// onAmdgpuWavefrontDispatched() runs for any of them. The span is valid only
  /// during this callback. Register allocation metadata is available through
  /// each wave and its CU.
  /// Infrequent hook; see the concurrency contract on requires_serial_hot_hooks().
  virtual void onAmdgpuWorkgroupDispatched(uint32_t /*dispatch_id*/, uint32_t /*wg_id*/,
                                           std::span<amdgpu::Wavefront *> /*wavefronts*/) {}

  /// Called when the last wavefront of a workgroup has completed normally.
  /// Cancelled workgroups do not receive this callback.
  /// Infrequent hook; see the concurrency contract on requires_serial_hot_hooks().
  virtual void onAmdgpuWorkgroupCompleted(uint32_t /*dispatch_id*/, uint32_t /*wg_id*/) {}

  /// Called after a wavefront is initialized and before its first instruction.
  /// Infrequent hook; see the concurrency contract on requires_serial_hot_hooks().
  virtual void onAmdgpuWavefrontDispatched(amdgpu::Wavefront & /*wf*/) {}

  /// Called when a wavefront terminates normally, after its memory operations
  /// have drained and before its resources are freed.
  /// This is a terminal notification; report problems through sink().write().
  /// Instruction-execution errors recorded here are discarded with the wave.
  /// Dispatch cancellation uses onAmdgpuWavefrontCancelled instead.
  /// Infrequent hook; see the concurrency contract on requires_serial_hot_hooks().
  virtual void onAmdgpuWavefrontHalted(amdgpu::Wavefront & /*wf*/) {}

  /// Called once when a dispatched wave incarnation is cancelled, while its
  /// registers and plugin state are still available. The wave's instruction-execution
  /// error, if present, is still available. May precede its first instruction.
  /// Cancellation also covers committed cluster placements rolled back after
  /// a later peer fails setup. That dispatch may retry and complete normally;
  /// cancellation alone does not imply permanent dispatch failure.
  /// This is a terminal notification for this incarnation, not successful
  /// workgroup or dispatch completion; it does not also call onAmdgpuWavefrontHalted.
  /// Infrequent hook; see the concurrency contract on requires_serial_hot_hooks().
  virtual void onAmdgpuWavefrontCancelled(amdgpu::Wavefront & /*wf*/) {}

  /// Called when a VGPR is read during instruction execution.
  /// @param wf Owning wavefront, or nullptr if the register is unallocated.
  /// @param physical_reg Physical register index in the VGPR file.
  /// @param lane_mask Bit mask of lanes read by the instruction.
  /// @param byte_mask Sub-dword byte mask (kFullByteMask = full dword).
  /// One callback may report multiple lanes of the same architectural access;
  /// plugins must consume @p lane_mask rather than relying on callback count or
  /// per-lane callback ordering.
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

  /// Called when an architectural scalar register is read during instruction execution.
  /// @param wf Owning wavefront.
  /// @param reg Logical register range, reported in one callback. The class
  ///        distinguishes ordinary SGPRs from TTMPs without exposing their
  ///        encoded selector values or physical storage layout.
  /// May run concurrently across simulation partitions unless
  /// requires_serial_hot_hooks() returns true.
  virtual void onAmdgpuReadScalarRegister(const amdgpu::Wavefront * /*wf*/, RegisterRef /*reg*/) {}

  /// Called when an architectural scalar register is written during instruction execution.
  /// Memory completion and runtime initialization remain unobserved storage operations.
  virtual void onAmdgpuWriteScalarRegister(const amdgpu::Wavefront * /*wf*/, RegisterRef /*reg*/) {}

  /// Called with the waves synchronized by a completed barrier domain.
  /// Workgroup callbacks include all surviving waves of one workgroup.
  /// Waves that ended before the barrier resolved are not included.
  /// Infrequent hook; see the concurrency contract on requires_serial_hot_hooks().
  virtual void onAmdgpuBarrierResolved(std::span<amdgpu::Wavefront *> /*wavefronts*/,
                                       BarrierKind /*kind*/) {}

  /// Opt in to reads of the LDS-direct VALU source. Sampled when the plugin is registered.
  virtual bool observes_lds_direct_reads() const { return false; }
  virtual void onAmdgpuLdsDirectRead(const amdgpu::Wavefront &, uint32_t, uint32_t, uint32_t) {}

private:
  friend class ExecutionPluginGroup;
  std::string name_;
  uint32_t slot_index_ = 0;
  PluginSink *sink_ = &StderrSink::instance();
};

/// Optional interface for plugins that observe a wave entering its trap handler.
/// Separate from ExecutionPlugin to preserve its vtable for loadable plugins.
class AmdgpuTrapEntryObserver {
public:
  virtual ~AmdgpuTrapEntryObserver() = default;
  virtual void onAmdgpuTrapEntered(amdgpu::Wavefront &wf) = 0;
};

} // namespace rocjitsu
