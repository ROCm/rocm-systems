// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/plugins/execution_plugin.h"

#include "rocjitsu/vm/plugins/race_detector/core/race_detector.h"
#include "rocjitsu/vm/plugins/race_detector/core/wave_race_state.h"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace rocjitsu::plugins::race_detector {

struct MarkedInstruction {
  InstructionSnapshot instruction;
  int wave;
  int lane; // -1 for scalar
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
/// Throws std::out_of_range if the detector no longer holds the conflicting event.
MarkedInstruction findConflict(const RaceViolation &, RaceDetector &);

/// Format the reporting wave's trace, marking the first matching same-wave
/// conflict and the last matching current access by PC and encoding. Identical
/// producer executions are not distinguished. Other-wave and missing records
/// are printed separately. Retain synchronization and control-flow rows even
/// when the context limit elides other rows. Disassemble only printed records.
std::string formatTrace(const RingBuffer<InstructionSnapshot, 256> &trace,
                        const std::function<std::string(const InstructionSnapshot &)> &disassemble,
                        std::optional<MarkedInstruction> conflict, MarkedInstruction read,
                        size_t trace_context_instructions = 128);

/// Explain the exact conflicting event and its synchronization state.
std::string formatRaceDescription(const RaceViolation &violation, const EventRegistry &events,
                                  uint64_t access_pc, rj_code_arch_t arch);

struct RaceWavefrontState : WavefrontState {
  /// The wave's most recently issued instructions, oldest first. Each entry owns
  /// a copy of its encoding, so reports can disassemble it after the code is
  /// rewritten or unmapped.
  RingBuffer<InstructionSnapshot, 256> trace;
  WaveRaceState *race_state = nullptr;
};

class RaceDetectorPlugin : public ExecutionPlugin {
public:
  /// @param config_json Optional JSON object with trace_context_instructions (2..256).
  explicit RaceDetectorPlugin(const char *config_json = nullptr);
  ~RaceDetectorPlugin() override;

  bool observes_sgpr_reads() const override { return true; }
  bool observes_memory_routing() const override { return true; }

  void onAmdgpuDispatchPacketProcessed(const KernelDispatchInfo &info) override;

  void onAmdgpuWorkgroupDispatched(uint32_t dispatch_id, uint32_t wg_id,
                                   uint32_t physical_vgpr_count, uint32_t physical_sgpr_count,
                                   std::span<amdgpu::Wavefront *> wavefronts) override;

  void onAmdgpuMemoryAccessRouted(const amdgpu::MemoryAccessObservation &access,
                                  const Instruction &inst, amdgpu::Wavefront &wf) override;

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

  void onAmdgpuBarrierResolved(std::span<amdgpu::Wavefront *> wavefronts) override;

  std::string getSummary() const;

private:
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

  std::mutex dispatch_mutex_;
  std::map<WorkgroupKey, std::unique_ptr<RaceDetector>> detectors_;

  const size_t trace_context_instructions_;
  mutable std::mutex report_mutex_;
  uint64_t reports_emitted_ = 0;
  uint64_t observations_suppressed_ = 0;
  std::set<std::pair<uint32_t, uint64_t>> observed_races_;
  std::unordered_map<uint32_t, KernelNames> dispatch_kernel_names_;
};

} // namespace rocjitsu::plugins::race_detector
