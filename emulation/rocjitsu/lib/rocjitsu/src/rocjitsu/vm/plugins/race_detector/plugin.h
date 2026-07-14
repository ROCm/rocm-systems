// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/plugins/execution_plugin.h"

#include "rocjitsu/vm/plugins/race_detector/core/race_detector.h"
#include "rocjitsu/vm/plugins/race_detector/core/wave_race_state.h"

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>

namespace rocjitsu::plugins::race_detector {

struct MarkedPc {
  uint64_t pc;
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

  T operator[](size_t i) const { return data[(head - len + i) % N]; }
  size_t size() const { return len; }
};

/// Find the memory instruction whose outstanding result conflicts with the
/// racy read described by @p v, returning its PC and wave.
std::optional<MarkedPc> findConflict(const RaceViolation &v, RaceDetector &detector);

/// Format a trace with ==> markers and wave/lane annotations.
std::string formatTrace(const RingBuffer<uint64_t, 256> &trace,
                        const std::unordered_map<uint64_t, std::string> &disasm,
                        std::optional<MarkedPc> conflict, MarkedPc read);

/// PC-to-disassembly cache, shared across all wavefronts in a dispatch.
///
/// Records disassembly on first encounter of each PC. An alternative would
/// be lazy decode at race-report time (code bytes are in GPU memory), but
/// this approach avoids needing access to the CU's decoder from the plugin.
///
/// Shared per-dispatch because all wavefronts execute the same kernel code.
/// Per-wavefront caches caused cache thrashing under round-robin scheduling.
///
/// Keyed by absolute PC. This avoids assuming that a dispatch executes a single
/// compact, monotonic text range; helper/trampoline code can be far away from
/// the first PC observed for the dispatch.
struct DisasmCache {
  void record(uint64_t pc, const Instruction &inst) { record(pc, inst.disassemble()); }

  void record(uint64_t pc, std::string disasm) {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.try_emplace(pc, std::move(disasm));
  }

  std::unordered_map<uint64_t, std::string> to_map() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_;
  }

private:
  mutable std::mutex mutex_;
  std::unordered_map<uint64_t, std::string> entries_;
};

struct RaceWavefrontState : WavefrontState {
  RingBuffer<uint64_t, 256> trace;
  std::shared_ptr<DisasmCache> disasm;
  WaveRaceState *race_state = nullptr;
};

class RaceDetectorPlugin : public ExecutionPlugin {
public:
  RaceDetectorPlugin();
  ~RaceDetectorPlugin() override;

  void onAmdgpuDispatchPacketProcessed(const KernelDispatchInfo &info) override;

  void onAmdgpuWorkgroupDispatched(uint32_t dispatch_id, uint32_t wg_id,
                                   uint32_t physical_vgpr_count, uint32_t sgpr_count,
                                   std::span<amdgpu::Wavefront *> wavefronts) override;

  void onAmdgpuRouteMemoryInstruction(const Instruction &inst, amdgpu::Wavefront &wf) override;

  void onAmdgpuReadVgprs(const amdgpu::Wavefront *wf, uint32_t physical_reg, uint32_t lane_begin,
                         uint32_t lane_end, uint8_t byte_mask = 0xF) override;

  void onAmdgpuReadSgpr(const amdgpu::Wavefront *wf, uint32_t physical_reg) override;

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

  RaceWavefrontState *get_state(const amdgpu::Wavefront &wf) {
    return static_cast<RaceWavefrontState *>(wf.plugin_state(slot_index()));
  }
  RaceWavefrontState *get_state(const amdgpu::Wavefront *wf) {
    return wf ? get_state(*wf) : nullptr;
  }

  std::mutex dispatch_mutex_;
  std::map<WorkgroupKey, std::unique_ptr<RaceDetector>> detectors_;
  std::map<uint32_t, std::shared_ptr<DisasmCache>> dispatch_disasm_;

  std::mutex report_mutex_;
  std::set<std::pair<uint32_t, uint64_t>> observed_races_;
};

} // namespace rocjitsu::plugins::race_detector
