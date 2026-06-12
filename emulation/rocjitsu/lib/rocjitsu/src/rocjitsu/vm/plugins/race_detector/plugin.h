// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/plugins/execution_plugin.h"

#include "rocjitsu/vm/plugins/race_detector/core/race_detector.h"
#include "rocjitsu/vm/plugins/race_detector/core/wave_race_state.h"

#include <array>
#include <atomic>
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
/// Indexed by (pc - base) / 4, avoiding hash-map overhead on the hot path.
/// The mutex is only taken for vector resizes (rare) and first writes
/// (~100 unique PCs per kernel).
struct DisasmCache {
  void record(uint64_t pc, const Instruction &inst) {
    size_t idx = pc_to_idx(pc);
    if (idx < size_.load(std::memory_order_acquire) && !entries_[idx].empty())
      return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (entries_.empty())
      base_ = pc;
    idx = pc_to_idx(pc);
    if (idx >= entries_.size())
      entries_.resize(std::max(idx + 1, entries_.size() * 2));
    if (entries_[idx].empty())
      entries_[idx] = inst.disassemble();
    size_.store(entries_.size(), std::memory_order_release);
  }

  std::unordered_map<uint64_t, std::string> to_map() const {
    std::unordered_map<uint64_t, std::string> m;
    for (size_t i = 0; i < entries_.size(); ++i)
      if (!entries_[i].empty())
        m[base_ + i * 4] = entries_[i];
    return m;
  }

private:
  size_t pc_to_idx(uint64_t pc) const { return (pc - base_) / 4; }

  std::mutex mutex_;
  uint64_t base_ = 0;
  std::vector<std::string> entries_;
  std::atomic<size_t> size_{0};
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
