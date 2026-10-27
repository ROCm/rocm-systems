// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/plugins/execution_plugin.h"

#include "rocjitsu/vm/plugins/race_detector/core/race_detector.h"
#include "rocjitsu/vm/plugins/race_detector/core/wave_race_state.h"

#include <array>
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

namespace rocjitsu::plugins::race_detector {

struct MarkedInstruction {
  InstructionSnapshot instruction;
  int wave;
  int lane; // -1 for scalar
};

template <typename T, size_t N> struct RingBuffer {
  std::array<T, N> data{};
  size_t head = 0;
  size_t len = 0;

  void push(T val) {
    data[head] = val;
    head = (head + 1) % N;
    if (len < N)
      ++len;
  }

  /// Element i in insertion order; 0 is the oldest retained element.
  const T &operator[](size_t i) const { return data[(head - len + i) % N]; }
  size_t size() const { return len; }
};

/// Return the memory instruction recorded as the exact conflict.
MarkedInstruction findConflict(const RaceViolation &, RaceDetector &);

/// Format the instructions that one wave executed between a conflicting memory
/// instruction and the access that raced with it.
///
/// Marks trace entries whose PC matches `conflict` or `read` with `==>` and
/// annotates them with wave and lane. Drops entries before the first marked one
/// and after the first match of `read` that follows it, and elides the middle of
/// a remaining span longer than 32 rows. If the trace contains no entry at the
/// conflict's PC, prints the conflict first with a "(before trace window)" note.
/// Marked rows print the marked instruction's bytes; other rows print the trace
/// entry's bytes. Calls `disassemble` only for printed rows.
std::string formatTrace(const RingBuffer<InstructionSnapshot, 256> &trace,
                        const std::function<std::string(const InstructionSnapshot &)> &disassemble,
                        std::optional<MarkedInstruction> conflict, MarkedInstruction read);

struct RaceWavefrontState : WavefrontState {
  /// The wave's most recently issued instructions, oldest first. Each entry owns
  /// a copy of its encoding, so reports can disassemble it after the code is
  /// rewritten or unmapped.
  RingBuffer<InstructionSnapshot, 256> trace;
  WaveRaceState *race_state = nullptr;
};

class RaceDetectorPlugin : public ExecutionPlugin {
public:
  /// @param config_json Plugin configuration object as a JSON string (unused;
  ///        this plugin takes no configuration). May be null.
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

  std::mutex report_mutex_;
  std::set<std::pair<uint32_t, uint64_t>> observed_races_;
  std::unordered_map<uint32_t, KernelNames> dispatch_kernel_names_;
};

} // namespace rocjitsu::plugins::race_detector
