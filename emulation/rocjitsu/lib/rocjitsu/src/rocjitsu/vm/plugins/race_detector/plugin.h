// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/vm/plugins/execution_plugin.h"

#include "rocjitsu/vm/plugins/race_detector/core/event_registry.h"
#include "rocjitsu/vm/plugins/race_detector/core/instruction_snapshot.h"
#include "rocjitsu/vm/plugins/race_detector/core/race_detector.h"
#include "rocjitsu/vm/plugins/race_detector/core/types.h"
#include "rocjitsu/vm/plugins/race_detector/core/wave_race_state.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
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

/// @brief Return the memory instruction recorded as the exact conflict.
/// @brief Returns failure if the detector no longer holds the conflicting event.
util::FailureOr<MarkedInstruction> findConflict(const RaceViolation &, RaceDetector &);

inline constexpr size_t kTraceHistorySize = 256;
/// Default instruction context before formatTrace elides the middle of a report.
inline constexpr size_t kDefaultTraceContext = 128;

/// Instruction context for another wave's conflict when the reporting wave never
/// ran the conflicting instruction.
inline constexpr size_t kOtherWaveTraceContext = 16;

/// @brief Format the reporting wave's trace in execution order.
/// @details Mark the first matching same-wave conflict and the last matching
/// current access by PC and encoding. Identical
/// producer executions are not distinguished. A conflict from another wave, or
/// one no longer retained, is printed separately before the trace. For another
/// wave's conflict, the trace starts at the reporting wave's latest execution of
/// the same instruction, or shows the last min(kOtherWaveTraceContext, context_instructions)
/// instructions when it has none. Longer spans keep their beginning and end,
/// plus synchronization and control-flow instructions. Context counts executed
/// instructions before consecutive identical unmarked s_nops are compacted.
/// Disassemble only printed rows. Returns failure when `disassemble` fails.
/// The callback must not throw.
/// @pre A same-wave conflict must not follow the current access in the trace.
/// @pre context_instructions is between 2 and kTraceHistorySize, inclusive.
util::FailureOr<std::string> formatTrace(
    const RingBuffer<InstructionSnapshot, kTraceHistorySize> &trace,
    const std::function<util::FailureOr<std::string>(const InstructionSnapshot &)> &disassemble,
    std::optional<MarkedInstruction> conflict, MarkedInstruction access,
    size_t context_instructions = kDefaultTraceContext);

/// @brief Describe a violation: the hazard direction, the current access at
/// `access_pc`, the conflicting event, and that event's completion and
/// synchronization state, with counter names for `arch`.
/// @throws std::out_of_range if `events` no longer holds the conflicting event.
/// @throws std::invalid_argument for an architecture without a wait-counter model.
/// @throws std::logic_error if the event is already retired.
std::string formatRaceDescription(const RaceViolation &violation, const EventRegistry &events,
                                  uint64_t access_pc, rj_code_arch_t arch);

struct RaceWavefrontState : WavefrontState {
  /// The wave's most recently issued instructions, oldest first. Each entry owns
  /// a copy of its encoding, so reports can disassemble it after the code is
  /// rewritten or unmapped.
  RingBuffer<InstructionSnapshot, kTraceHistorySize> trace;
  WaveRaceState *race_state = nullptr;
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
  /// @param config_json Plugin JSON with an optional integer trace_context in
  ///        [2, kTraceHistorySize]. Null or empty uses the default context.
  /// @throws std::invalid_argument for malformed configuration.
  explicit RaceDetectorPlugin(const char *config_json = nullptr);
  ~RaceDetectorPlugin() override;

  bool observes_sgpr_reads() const override { return true; }
  bool observes_memory_routing() const override { return true; }
  bool observes_tensor_dma_issue() const override { return true; }
  /// Skip waves without race state, including slots between a halt and the
  /// next workgroup dispatch, when other plugins may read their registers.
  bool observes_hot_hooks_for_wavefront(const amdgpu::Wavefront *wf) const override;

  void onAmdgpuDispatchPacketProcessed(const KernelDispatchInfo &info) override;

  void onAmdgpuWorkgroupDispatched(uint32_t dispatch_id, uint32_t wg_id,
                                   uint32_t physical_vgpr_count, uint32_t physical_sgpr_count,
                                   std::span<amdgpu::Wavefront *> wavefronts) override;

  /// Release the workgroup's detector. Reported races are still counted.
  void onAmdgpuWorkgroupCompleted(uint32_t dispatch_id, uint32_t wg_id) override;

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

  void onAmdgpuReadVgprLanes(const amdgpu::Wavefront *wf, uint32_t physical_reg, uint64_t lane_mask,
                             uint8_t byte_mask = ExecutionPlugin::kFullByteMask) override;

  void onAmdgpuWriteVgprLanes(const amdgpu::Wavefront *wf, uint32_t physical_reg,
                              uint64_t lane_mask,
                              uint8_t byte_mask = ExecutionPlugin::kFullByteMask) override;

  void onAmdgpuReadScalarRegister(const amdgpu::Wavefront *wf, RegisterRef reg) override;

  void onAmdgpuWriteScalarRegister(const amdgpu::Wavefront *wf, RegisterRef reg) override;

  void onAmdgpuBeforeExecuteInstruction(uint64_t pc, const Instruction &inst,
                                        amdgpu::Wavefront &wf) override;

  void onAmdgpuAfterExecuteInstruction(uint64_t pc, const Instruction &inst,
                                       amdgpu::Wavefront &wf) override;

  void onAmdgpuTrapEntered(amdgpu::Wavefront &wf) override;
  void onAmdgpuImplicitMemoryWait(amdgpu::Wavefront &wf, const amdgpu::WaitFields &fields) override;

  void onAmdgpuBarrierResolved(std::span<amdgpu::Wavefront *> wavefronts,
                               BarrierKind kind) override;

  std::string getSummary() const;

  /// Number of workgroups whose detector state is still held.
  size_t retainedWorkgroupCount() const;

private:
  size_t trace_context_ = kDefaultTraceContext;

  struct WorkgroupKey {
    uint32_t dispatch_id;
    uint32_t workgroup_id;
    bool operator<(const WorkgroupKey &o) const {
      return std::tie(dispatch_id, workgroup_id) < std::tie(o.dispatch_id, o.workgroup_id);
    }
  };

  struct KernelNames {
    std::string name;
    std::string symbol;
  };

  RaceWavefrontState *get_state(const amdgpu::Wavefront &wf) {
    return static_cast<RaceWavefrontState *>(wf.plugin_state(slot_index()));
  }
  RaceWavefrontState *get_state(const amdgpu::Wavefront *wf) {
    return wf ? get_state(*wf) : nullptr;
  }
  void release_wave_state(amdgpu::Wavefront &wf) { wf.clear_plugin_state(slot_index()); }

  mutable std::mutex dispatch_mutex_;
  std::map<WorkgroupKey, std::unique_ptr<RaceDetector>> detectors_;

  mutable std::mutex report_mutex_;
  uint64_t reports_emitted_ = 0;
  uint64_t observations_suppressed_ = 0;
  std::set<std::pair<uint32_t, uint64_t>> observed_races_;
  std::unordered_map<uint32_t, KernelNames> dispatch_kernel_names_;
};

} // namespace rocjitsu::plugins::race_detector
