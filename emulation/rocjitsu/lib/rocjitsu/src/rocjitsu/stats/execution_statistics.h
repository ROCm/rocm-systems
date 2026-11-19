// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/stats/statistics.h"
#include "rocjitsu/vm/plugins/execution_plugin.h"
#include "rocjitsu/vm/plugins/instruction_family.h"

#include <array>
#include <chrono>
#include <unordered_map>

namespace rocjitsu {

/// @brief Execution observation adapter sharing the VM's statistics session.
class ExecutionStatistics final : public ExecutionPlugin {
public:
  ExecutionStatistics(std::shared_ptr<StatisticsSession> session, uint32_t gpu);
  static bool requested(const StatisticsSession &session);
  bool observes_before_execute_instruction() const override { return count_instructions_; }
  bool observes_async_instruction_issued() const override { return count_instructions_; }
  bool observes_after_execute_instruction() const override { return false; }
  bool observes_memory_instruction_routing() const override { return false; }
  bool observes_vgpr_reads() const override { return false; }
  bool observes_vgpr_writes() const override { return false; }
  bool observes_sgpr_reads() const override { return false; }
  bool observes_scalar_register_writes() const override { return false; }
  bool supports_async_instructions() const override { return true; }

  void onAmdgpuDispatchPacketProcessed(const KernelDispatchInfo &info) override;
  void onAmdgpuDispatchExecutionBegin(uint32_t dispatch_id) override;
  void onAmdgpuDispatchExecutionEnd(uint32_t dispatch_id) override;
  void onAmdgpuWavefrontDispatched(amdgpu::Wavefront &wave) override;
  void onAmdgpuWavefrontHalted(amdgpu::Wavefront &wave) override;
  void onAmdgpuWavefrontCancelled(amdgpu::Wavefront &wave) override;
  void onAmdgpuBeforeExecuteInstruction(uint64_t, const Instruction &instruction,
                                        amdgpu::Wavefront &wave) override;
  void onAmdgpuAsyncInstructionIssued(uint64_t, const Instruction &instruction,
                                      amdgpu::Wavefront &wave) override;
  void onShutdown() override;

private:
  using Counts =
      std::array<std::shared_ptr<stats::RegisteredCounter>, plugins::kInstructionFamilyCount>;
  using Clock = std::chrono::steady_clock;
  class WaveState final : public WavefrontState {
  public:
    const ExecutionStatistics *owner = nullptr;
    Counts counts;
  };
  class DispatchState {
  public:
    Clock::time_point begin;
    bool begun = false;
    Counts counts;
    std::shared_ptr<stats::Scope> scope;
    std::shared_ptr<stats::RegisteredHistogram> duration;
    std::shared_ptr<stats::RegisteredCounter> started;
    std::shared_ptr<stats::RegisteredCounter> ended;
    std::shared_ptr<stats::RegisteredCounter> incomplete;
  };

  static Counts bind_counts(const std::shared_ptr<stats::Scope> &scope);
  static std::shared_ptr<stats::RegisteredHistogram>
  bind_duration(const std::shared_ptr<stats::Scope> &scope);
  void count(const Instruction &instruction, amdgpu::Wavefront &wave);

  std::shared_ptr<StatisticsSession> session_;
  uint32_t gpu_;
  bool count_instructions_ = false;
  bool shutdown_ = false;
  Counts counts_;
  std::shared_ptr<stats::RegisteredCounter> started_;
  std::shared_ptr<stats::RegisteredCounter> ended_;
  std::shared_ptr<stats::RegisteredCounter> incomplete_;
  std::shared_ptr<stats::RegisteredCounter> cancelled_waves_;
  std::shared_ptr<stats::RegisteredCounter> halted_waves_;
  std::shared_ptr<stats::RegisteredHistogram> duration_;
  std::unordered_map<uint32_t, DispatchState> dispatches_;
};

} // namespace rocjitsu
