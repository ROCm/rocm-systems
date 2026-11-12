// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/plugins/execution_plugin.h"

#include "rocjitsu/vm/plugins/race_detector/core/race_detector.h"
#include "rocjitsu/vm/plugins/race_detector/core/wave_race_state.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace rocjitsu::plugins::race_detector {

struct MarkedInstruction {
  InstructionSnapshot instruction;
  int wave;
  int lane; // -1 for no lane: a scalar access or a tensor DMA transfer.
};

/// Retains the last N pushed values. Elements are never default-constructed.
template <typename T, size_t N> class RingBuffer {
public:
  void push(const T &val) {
    if (data_.size() < N)
      data_.push_back(val);
    else
      data_[head_] = val;
    head_ = (head_ + 1) % N;
  }

  /// Element i in insertion order; 0 is the oldest retained element.
  /// Entries start at index 0 while filling; once full, head_ is the oldest entry.
  const T &operator[](size_t i) const { return data_[data_.size() < N ? i : (head_ + i) % N]; }
  size_t size() const { return data_.size(); }

private:
  std::vector<T> data_;
  size_t head_ = 0;
};

/// Return the memory instruction recorded as the exact conflict.
/// Returns failure if the detector no longer holds the conflicting event.
util::FailureOr<MarkedInstruction> findConflict(const RaceViolation &, RaceDetector &);

/// Format the instructions that one wave executed between a conflicting memory
/// instruction and the access that raced with it.
///
/// Marks trace entries whose PC and encoding match `conflict` or `read` with `==>` and
/// annotates them with wave and lane. Drops entries before the first marked one
/// and after the first match of `read` that follows it, and elides the middle of
/// a remaining span longer than 32 rows. If the trace contains no entry at the
/// conflict's PC and encoding, prints the conflict first with a "(before trace window)"
/// note. Each trace row prints its own captured bytes. Calls `disassemble` only
/// for printed rows. Repeated executions of identical bytes at the same PC are
/// not distinguished. Returns failure when `disassemble` fails. The callback must not throw.
util::FailureOr<std::string> formatTrace(
    const RingBuffer<InstructionSnapshot, 256> &trace,
    const std::function<util::FailureOr<std::string>(const InstructionSnapshot &)> &disassemble,
    std::optional<MarkedInstruction> conflict, MarkedInstruction read);

struct RaceWavefrontState : WavefrontState {
  /// The wave's most recently issued instructions, oldest first. Each entry owns
  /// a copy of its encoding, so reports can disassemble it after the code is
  /// rewritten or unmapped.
  RingBuffer<InstructionSnapshot, 256> trace;
  // Aliases this wave's state while owning the whole workgroup detector. Peers
  // keep an ended wave's LDS history alive until the last wave releases it.
  std::shared_ptr<WaveRaceState> race_state;
  struct LdsOperandRead {
    uint32_t address;
    uint32_t bytes;
    uint32_t lane;
  };
  std::optional<LdsOperandRead> lds_operand_read;
};

class RaceDetectorPlugin : public ExecutionPlugin,
                           public AmdgpuTrapEntryObserver,
                           public AmdgpuImplicitMemoryWaitObserver {
public:
  /// Reports the core's register findings and checks LDS ordering.
  explicit RaceDetectorPlugin(const char *config_json = nullptr);
  ~RaceDetectorPlugin() override;

  bool observes_before_execute_instruction() const override { return true; }
  bool observes_after_execute_instruction() const override { return true; }
  bool observes_async_instruction_issued() const override { return true; }
  bool observes_register_hazards() const override { return true; }
  bool observes_memory_routing() const override { return true; }
  bool observes_tensor_dma_issue() const override { return true; }
  /// Skip slots without a live race-analysis state.
  bool observes_hot_hooks_for_wavefront(const amdgpu::Wavefront *wf) const override;

  void onAmdgpuDispatchPacketProcessed(const KernelDispatchInfo &info) override;

  void onAmdgpuRegisterHazard(const amdgpu::RegisterHazardObservation &hazard,
                              const amdgpu::Wavefront &wf) override;

  void onAmdgpuWorkgroupDispatched(uint32_t dispatch_id, uint32_t wg_id,
                                   std::span<amdgpu::Wavefront *> wavefronts) override;

  void onAmdgpuWavefrontHalted(amdgpu::Wavefront &wf) override;

  void onAmdgpuWavefrontCancelled(amdgpu::Wavefront &wf) override;

  void onAmdgpuMemoryAccessRouted(const amdgpu::MemoryAccessObservation &access,
                                  const Instruction &inst, amdgpu::Wavefront &wf) override;

  bool observes_lds_direct_reads() const override { return true; }
  void onAmdgpuLdsDirectRead(const amdgpu::Wavefront &wf, uint32_t address, uint32_t bytes,
                             uint32_t lane) override;

  /// Register an issued tensor DMA instruction's LDS side as an event on TENSORCNT.
  void onAmdgpuTensorDmaIssued(const amdgpu::TensorDmaIssueObservation &issue,
                               const amdgpu::Wavefront &wf) override;

  void onAmdgpuBeforeExecuteInstruction(uint64_t pc, const Instruction &inst, amdgpu::Wavefront &wf,
                                        std::span<const uint32_t>) override;

  void onAmdgpuAsyncInstructionIssued(uint64_t pc, const Instruction &inst,
                                      amdgpu::Wavefront &wf) override;

  void onAmdgpuAfterExecuteInstruction(uint64_t pc, const Instruction &inst,
                                       amdgpu::Wavefront &wf) override;

  void onAmdgpuTrapEntered(amdgpu::Wavefront &wf) override;
  void onAmdgpuImplicitMemoryWait(amdgpu::Wavefront &wf, const amdgpu::WaitFields &fields) override;

  void onAmdgpuBarrierResolved(std::span<amdgpu::Wavefront *> wavefronts,
                               BarrierKind kind) override;

  std::string getSummary() const;

private:
  struct KernelNames {
    std::string name;
    std::string symbol;
  };

  struct ReportLocation {
    std::string_view space;
    int index;
    int wave;
    int lane;
    bool write;
    Dim3d workgroup;
    uint64_t producer_pc;
    uint64_t consumer_pc;
    rj_code_arch_t arch;
  };
  // Called with report_mutex_ held, after selecting a report for emission.
  void emit_report(uint32_t dispatch, const ReportLocation &location, std::string_view details,
                   const amdgpu::RegisterHazardObservation *hazard = nullptr);

  RaceWavefrontState *get_state(const amdgpu::Wavefront &wf) {
    return static_cast<RaceWavefrontState *>(wf.plugin_state(slot_index()));
  }
  RaceWavefrontState *get_state(const amdgpu::Wavefront *wf) {
    return wf ? get_state(*wf) : nullptr;
  }
  void release_wave_state(amdgpu::Wavefront &wf) { wf.clear_plugin_state(slot_index()); }

  mutable std::mutex report_mutex_;
  std::set<std::pair<uint32_t, uint64_t>> observed_races_;
  using RegisterSite =
      std::tuple<uint32_t, uint64_t, uint64_t, RegClass, uint16_t, bool, amdgpu::WaitCounterKind>;
  std::set<RegisterSite> observed_register_races_;
  uint64_t reported_races_ = 0;
  uint64_t repeated_register_races_ = 0;
  std::unordered_map<uint32_t, KernelNames> dispatch_kernel_names_;
};

} // namespace rocjitsu::plugins::race_detector
