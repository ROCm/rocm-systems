// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/plugins/execution_plugin.h"

#include <memory>

namespace rocjitsu::plugins::pffm {

/// Adapts RocJITsu execution observations to the FFM observer ABI consumed by
/// pFFM. The pFFM backend remains a separate shared object and owns all of its
/// configuration and report output.
class PffmPlugin final : public ExecutionPlugin {
public:
  /// @param config_json Resolved plugin configuration containing the required
  ///        string field `library_path`.
  explicit PffmPlugin(const char *config_json);
  ~PffmPlugin() override;

  PffmPlugin(const PffmPlugin &) = delete;
  PffmPlugin &operator=(const PffmPlugin &) = delete;

  bool requires_serial_hot_hooks() const override { return true; }
  bool observes_memory_routing() const override { return true; }
  bool observes_tensor_dma_memory_access() const override { return true; }
  bool observes_sgpr_reads() const override { return false; }

  void onInit() override;
  void onShutdown() override;
  void onAmdgpuDispatchPacketProcessed(const KernelDispatchInfo &info) override;
  void onAmdgpuDispatchExecutionBegin(uint32_t dispatch_id) override;
  void onAmdgpuDispatchExecutionEnd(uint32_t dispatch_id) override;
  void onAmdgpuWavefrontDispatched(amdgpu::Wavefront &wf) override;
  void onAmdgpuWavefrontHalted(amdgpu::Wavefront &wf) override;
  void onAmdgpuBeforeExecuteInstruction(uint64_t pc, const Instruction &inst,
                                        amdgpu::Wavefront &wf) override;
  void onAmdgpuBeforeExecuteInstruction(uint64_t pc, const Instruction &inst, amdgpu::Wavefront &wf,
                                        std::span<const uint32_t> fetch_window) override;
  void onAmdgpuMemoryAccessRouted(const amdgpu::MemoryAccessObservation &access) override;
  void
  onAmdgpuTensorDmaMemoryAccess(const amdgpu::TensorDmaMemoryAccessObservation &access) override;

private:
  void record_instruction(uint64_t pc, const Instruction &inst, amdgpu::Wavefront &wf,
                          std::span<const uint32_t> fetch_window);

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace rocjitsu::plugins::pffm
