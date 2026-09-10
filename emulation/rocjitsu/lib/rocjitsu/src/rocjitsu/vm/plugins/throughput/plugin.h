// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/plugins/execution_plugin.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <unordered_map>

namespace rocjitsu::plugins::throughput {

/// Exclusive instruction families used by the throughput report. The order is
/// part of the JSONL schema and should remain stable.
enum class InstructionFamily : size_t {
  Scalar,
  Vector,
  Matrix,
  Lds,
  Global,
  Control,
  Other,
  Count,
};

inline constexpr size_t kInstructionFamilyCount = static_cast<size_t>(InstructionFamily::Count);
using InstructionCounts = std::array<uint64_t, kInstructionFamilyCount>;
using InstructionNanoseconds = std::array<uint64_t, kInstructionFamilyCount>;

struct ThroughputWavefrontState final : WavefrontState {
  InstructionCounts counts{};
  InstructionNanoseconds execution_nanoseconds{};
  std::chrono::steady_clock::time_point instruction_begin{};
  InstructionFamily active_family = InstructionFamily::Other;
  bool instruction_active = false;
};

/// Reports simulator throughput in executed wave instructions per host second.
///
/// One instruction is counted each time a wavefront reaches the before-execute
/// hook. It is not multiplied by the number of active lanes.
class ThroughputPlugin final : public ExecutionPlugin {
public:
  /// @param config_json Plugin configuration object as a JSON string (unused;
  ///        this plugin takes no configuration). May be null.
  explicit ThroughputPlugin(const char *config_json = nullptr);
  ~ThroughputPlugin() override;

  bool observes_sgpr_reads() const override { return false; }

  void onShutdown() override;
  void onAmdgpuDispatchPacketProcessed(const KernelDispatchInfo &info) override;
  void onAmdgpuDispatchExecutionBegin(uint32_t dispatch_id) override;
  void onAmdgpuDispatchExecutionEnd(uint32_t dispatch_id) override;
  void onAmdgpuWavefrontDispatched(amdgpu::Wavefront &wf) override;
  void onAmdgpuWavefrontHalted(amdgpu::Wavefront &wf) override;
  void onAmdgpuBeforeExecuteInstruction(uint64_t pc, const Instruction &inst,
                                        amdgpu::Wavefront &wf) override;
  void onAmdgpuAfterExecuteInstruction(uint64_t pc, const Instruction &inst,
                                       amdgpu::Wavefront &wf) override;

  static InstructionFamily classify(const Instruction &inst);
  static std::string_view family_name(InstructionFamily family);

private:
  using Clock = std::chrono::steady_clock;

  struct DispatchState {
    KernelDispatchInfo info;
    Clock::time_point begin{};
    bool begun = false;
    InstructionCounts counts{};
    InstructionNanoseconds execution_nanoseconds{};
  };

  static uint64_t total(const InstructionCounts &counts);
  static void add(InstructionCounts &destination, const InstructionCounts &source);
  static void finish_instruction(ThroughputWavefrontState &state, Clock::time_point end);
  void emit_record(std::string_view record, const KernelDispatchInfo *info,
                   const InstructionCounts &counts,
                   const InstructionNanoseconds &execution_nanoseconds, double wall_seconds,
                   double dispatch_seconds_sum = 0.0);

  std::unordered_map<uint32_t, DispatchState> dispatches_;
  InstructionCounts aggregate_counts_{};
  InstructionNanoseconds aggregate_execution_nanoseconds_{};
  Clock::time_point first_begin_{};
  Clock::time_point last_end_{};
  double dispatch_seconds_sum_ = 0.0;
  uint64_t completed_dispatches_ = 0;
  bool have_active_window_ = false;
  bool summary_emitted_ = false;
};

} // namespace rocjitsu::plugins::throughput
