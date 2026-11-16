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
#include <unordered_map>
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
  MemoryFault, ///< A memory access faulted (the wave halted or trapped).
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
  VSet vchanged;
  std::vector<std::string> frame_changed;
  MachineState after;
};

/// Initial memory content of one case, shared by the old and new runners so both read
/// the same input. Words are generated on first read (data or pointer-shaped) and kept.
class MemoryImage {
public:
  /// Start a case. The content is kept while the seed is unchanged (re-executions of the
  /// same case: reduction, traces), otherwise cleared; preset words are added if absent.
  void begin(uint64_t seed, const std::vector<std::pair<uint64_t, uint32_t>> &preset);
  /// Initial value of word @p w; @p pointer_half is -1 for data, 0/1 for the low/high half
  /// of a pointer (canonical, away from the region's code).
  uint32_t word(uint64_t w, int pointer_half);
  [[nodiscard]] std::vector<std::pair<uint64_t, uint32_t>> snapshot() const;

private:
  uint64_t seed_ = 0;
  bool started_ = false;
  std::unordered_map<uint64_t, uint32_t> words_;
};

class RegionRunner {
public:
  /// @brief Build a runner. Decode failures do not fail creation; run() reports them.
  /// @param vgpr_count VGPRs v0..v{n-1} the wave allocates, loads and captures; any
  /// access beyond them is an invalid register access. 0: no VGPR state (SALU region).
  static std::unique_ptr<RegionRunner> create(const std::vector<RegionInst> &insts,
                                              const std::string &label, std::string &error,
                                              unsigned vgpr_count = 0);
  ~RegionRunner();
  RegionRunner(const RegionRunner &) = delete;
  RegionRunner &operator=(const RegionRunner &) = delete;

  /// Frame values right after dispatch; contracts fix the frame to these.
  [[nodiscard]] const Frame &baseline_frame() const { return baseline_; }

  void load(const MachineState &state);
  /// Share the initial memory image with the other side's runner (one per case).
  void set_memory_image(std::shared_ptr<MemoryImage> image) { image_ = std::move(image); }
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
  unsigned vgpr_count_ = 0;
  /// Memory model: memory instructions are issued through ComputeUnitCore::step() (the
  /// full fetch/execute/route path, synchronous in functional mode); a plugin seeds the
  /// bytes a load reads first from the initial content and records stored bytes.
  class MemoryModel;
  bool has_memory_ = false;
  std::vector<bool> is_mem_;
  MemoryModel *mem_ = nullptr;
  std::shared_ptr<void> plugin_group_;
  std::shared_ptr<MemoryImage> image_ = std::make_shared<MemoryImage>();
  /// Per instruction: dwords of a scalar load that later serve as a memory base
  /// (bit k: dword k is a pointer's low half / high half).
  std::vector<uint32_t> pointer_lo_, pointer_hi_;
  uint64_t entry_pc_ = 0;
  uint32_t vgpr_alloc_ = 0;
  uint64_t total_executed_ = 0;
};

} // namespace rocjitsu::isa_diff
