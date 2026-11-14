// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/plugins/execution_plugin.h"
#include "util/diagnostic.h"

#include <memory>

namespace rocjitsu::plugins::perfsim {

/// Adapts RocJITsu execution observations to the FFM observer ABI consumed by
/// Perfsim. The Perfsim backend remains a separate shared object and owns all of its
/// configuration and report output.
class PerfsimPlugin final : public ExecutionPlugin {
public:
  PerfsimPlugin();
  ~PerfsimPlugin() override;

  /// @brief Configure the adapter before adding it to a plugin group.
  /// Expected admission/configuration/backend failures return failure with a diagnostic.
  /// A failed call releases its singleton claim and may be retried. The backend
  /// API accessor, callbacks, and diagnostic sink must not throw. Standard-library
  /// allocation failures are fatal when exceptions are disabled.
  /// @param config_json Resolved plugin configuration containing the required
  ///        non-empty string `library_path`; optional positive integers
  ///        `max_staged_bytes` and `max_observed_workgroups`; and the optional
  ///        non-empty string `dispatch_name`.
  util::Result initialize(const char *config_json, util::DiagnosticEmitter diagnostic = {});

  PerfsimPlugin(const PerfsimPlugin &) = delete;
  PerfsimPlugin &operator=(const PerfsimPlugin &) = delete;

  bool requires_serial_hot_hooks() const override;
  bool observes_hot_hooks_for_wavefront(const amdgpu::Wavefront *wf) const override;
  bool observes_after_execute_instruction() const override { return false; }
  bool observes_async_instruction_issued() const override { return false; }
  bool observes_memory_instruction_routing() const override { return false; }
  bool observes_vgpr_reads() const override { return false; }
  bool observes_vgpr_writes() const override { return false; }
  bool observes_scalar_register_writes() const override { return false; }
  bool observes_memory_routing() const override { return true; }
  bool observes_tensor_dma_memory_access() const override { return true; }
  bool observes_sgpr_reads() const override { return false; }

  /// @brief Activate the configured backend, whose callbacks must not throw.
  /// The FFM on_init callback returns no status and cannot report activation failure.
  void onInit() override;
  void onShutdown() override;
  void onAmdgpuDispatchPacketProcessed(const KernelDispatchInfo &info) override;
  void onAmdgpuDispatchExecutionBegin(uint32_t dispatch_id) override;
  void onAmdgpuDispatchExecutionEnd(uint32_t dispatch_id) override;
  void onAmdgpuWavefrontDispatched(amdgpu::Wavefront &wf) override;
  void onAmdgpuWavefrontHalted(amdgpu::Wavefront &wf) override;
  void onAmdgpuWavefrontCancelled(amdgpu::Wavefront &wf) override;
  void onAmdgpuBeforeExecuteInstruction(uint64_t pc, const Instruction &inst,
                                        amdgpu::Wavefront &wf) override;
  void onAmdgpuBeforeExecuteInstruction(uint64_t pc, const Instruction &inst, amdgpu::Wavefront &wf,
                                        std::span<const uint32_t> fetch_window) override;
  void onAmdgpuMemoryAccessRouted(const amdgpu::MemoryAccessObservation &access) override;
  void onAmdgpuMemoryAccessRouted(const amdgpu::MemoryAccessObservation &access,
                                  const amdgpu::Wavefront &wf) override;
  void
  onAmdgpuTensorDmaMemoryAccess(const amdgpu::TensorDmaMemoryAccessObservation &access) override;
  void onAmdgpuTensorDmaMemoryAccess(const amdgpu::TensorDmaMemoryAccessObservation &access,
                                     const amdgpu::Wavefront &wf) override;

private:
  void retire_wavefront(amdgpu::Wavefront &wf, bool cancelled);
  void record_instruction(uint64_t pc, const Instruction &inst, amdgpu::Wavefront &wf,
                          std::span<const uint32_t> fetch_window);

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace rocjitsu::plugins::perfsim
