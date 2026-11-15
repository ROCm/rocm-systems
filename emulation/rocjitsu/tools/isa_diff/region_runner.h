// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file region_runner.h
/// @brief Execute only a region's instructions through RocJITsu's instruction execution.
///
/// @details Each runner owns a private GpuMemory, L2Cache, ComputeUnitCore and one
/// dispatched wave64, so an old and a new runner share no mutable state. The region is
/// decoded once, from its recorded words, when the runner is created. Each case then:
///   1. load(): writes every tracked location and every frame field, and clears the
///      per-instruction error state;
///   2. run(): sets the PC to the recorded address of each instruction and calls
///      ComputeUnitCore::execute_instruction, the same entry point the CU's own issue
///      loop uses, so no semantics are re-implemented here;
///   3. capture(): reads the state back.
/// A failure is classified and never treated as a no-op or a success.
#pragma once

#include "machine_state.h"
#include "region.h"

#include <memory>
#include <string>
#include <vector>

namespace rocjitsu {
class Decoder;
class Instruction;
namespace amdgpu {
class ComputeUnitCore;
class GpuMemory;
class L2Cache;
class Wavefront;
} // namespace amdgpu
} // namespace rocjitsu

namespace rocjitsu::isa_diff {

enum class RunStatus {
  Ok,
  DecodeFailure,
  MissingExecutionBackend,
  Unimplemented,
  UnsupportedOperand,
  InvalidRegisterAccess,
  IllegalInstruction,
  ExecutionError,
  HostException,
  UndefinedBehavior,
  PcWritten,
  WaveStateChanged,
  BudgetExceeded,
};
[[nodiscard]] const char *run_status_name(RunStatus s);

struct RunOutcome {
  RunStatus status = RunStatus::Ok;
  size_t executed = 0;     ///< Instructions executed this run.
  size_t failed_index = 0; ///< Region index of the failing instruction.
  std::string detail;
};

struct TraceStep {
  size_t index = 0;
  uint64_t pc = 0;
  std::string text;
  LocSet changed;
  std::vector<std::string> frame_changed;
  MachineState after;
};

class RegionRunner {
public:
  /// @brief Build a runner. Decode failures do not fail creation; run() reports them.
  static std::unique_ptr<RegionRunner> create(const std::vector<RegionInst> &insts,
                                              const std::string &label, std::string &error);
  ~RegionRunner();
  RegionRunner(const RegionRunner &) = delete;
  RegionRunner &operator=(const RegionRunner &) = delete;

  /// Frame values right after dispatch; contracts fix the frame to these.
  [[nodiscard]] const Frame &baseline_frame() const { return baseline_; }

  void load(const MachineState &state);
  RunOutcome run(size_t budget, std::vector<TraceStep> *trace = nullptr);
  [[nodiscard]] MachineState capture() const;

  [[nodiscard]] size_t size() const { return pcs_.size(); }
  [[nodiscard]] uint64_t total_executed() const { return total_executed_; }
  /// Test seam: drop an instruction's execution callback (simulates a model-only provider).
  void simulate_model_only_for_test(size_t index);

private:
  RegionRunner();

  std::unique_ptr<amdgpu::GpuMemory> memory_;
  std::unique_ptr<amdgpu::L2Cache> l2_;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu_;
  amdgpu::Wavefront *wave_ = nullptr;
  std::unique_ptr<Decoder> decoder_;
  std::vector<std::unique_ptr<Instruction>> insts_;
  std::vector<bool> model_only_;
  std::vector<uint64_t> pcs_;
  std::vector<std::string> texts_;
  bool decode_failed_ = false;
  size_t decode_failed_index_ = 0;
  std::string decode_error_;
  Frame baseline_;
  uint64_t total_executed_ = 0;
};

} // namespace rocjitsu::isa_diff
