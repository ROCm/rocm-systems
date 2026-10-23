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
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <vector>

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

  void clear() {
    head = 0;
    len = 0;
  }

  T operator[](size_t i) const { return data[(head - len + i) % N]; }
  size_t size() const { return len; }
};

/// Return the memory instruction recorded as the exact conflict.
MarkedPc findConflict(const RaceViolation &, RaceDetector &);

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
  void record(uint64_t pc, const Instruction &inst) {
    // Hot path: every executed instruction calls this, from every simulation
    // thread, and all but the first execution of a PC are no-ops. Probe a
    // lock-free table of already-recorded PCs so the shared mutex is only taken
    // the first time a PC is seen. A miss caused by a full table or a lost race
    // falls through to the locked path, which remains authoritative.
    const uint64_t tag = pc + 1; // 0 marks an empty slot; PCs are never UINT64_MAX.
    size_t slot = probe_start(pc);
    for (size_t i = 0; i < kMaxProbes; ++i, slot = (slot + 1) & (kSlots - 1)) {
      const uint64_t seen = seen_[slot].load(std::memory_order_acquire);
      if (seen == tag)
        return;
      if (seen == 0)
        break;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (entries_.contains(pc))
      return;
    entries_.emplace(pc, inst.disassemble());
    slot = probe_start(pc);
    for (size_t i = 0; i < kMaxProbes; ++i, slot = (slot + 1) & (kSlots - 1)) {
      if (seen_[slot].load(std::memory_order_relaxed) == 0) {
        seen_[slot].store(tag, std::memory_order_release);
        return;
      }
    }
  }

  std::unordered_map<uint64_t, std::string> to_map() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_;
  }

private:
  static constexpr size_t kSlots = 1 << 14;
  static constexpr size_t kMaxProbes = 8;
  static size_t probe_start(uint64_t pc) {
    return static_cast<size_t>(((pc >> 2) * 0x9E3779B97F4A7C15ULL) >> 50) & (kSlots - 1);
  }

  std::array<std::atomic<uint64_t>, kSlots> seen_{};
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
  /// @param config_json Plugin configuration object as a JSON string (unused;
  ///        this plugin takes no configuration). May be null.
  explicit RaceDetectorPlugin(const char *config_json = nullptr);
  ~RaceDetectorPlugin() override;

  bool observes_sgpr_reads() const override { return true; }
  bool observes_memory_routing() const override { return true; }

  void onAmdgpuDispatchPacketProcessed(const KernelDispatchInfo &info) override;
  void onAmdgpuDispatchExecutionEnd(uint32_t dispatch_id) override;

  void onAmdgpuWorkgroupDispatched(uint32_t dispatch_id, uint32_t wg_id,
                                   uint32_t physical_vgpr_count, uint32_t physical_sgpr_count,
                                   std::span<amdgpu::Wavefront *> wavefronts) override;

  void onAmdgpuWorkgroupCompleted(uint32_t dispatch_id, uint32_t wg_id) override;

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

  /// Everything that fixes the allocation shape of a RaceDetector.
  struct DetectorShape {
    uint32_t num_waves;
    uint32_t vgpr_count;
    uint32_t sgpr_count;
    int vmcnt;
    int lgkmcnt;
    size_t lds_size_bytes;
  };
  struct PooledDetector {
    DetectorShape shape;
    std::unique_ptr<RaceDetector> detector;
  };
  static constexpr size_t kMaxPooledDetectors = 4096;

  std::mutex dispatch_mutex_;
  std::map<WorkgroupKey, std::unique_ptr<RaceDetector>> detectors_;
  std::map<WorkgroupKey, DetectorShape> detector_shapes_;
  std::vector<PooledDetector> free_detectors_;
  std::map<uint32_t, std::shared_ptr<DisasmCache>> dispatch_disasm_;

  std::mutex report_mutex_;
  std::set<std::pair<uint32_t, uint64_t>> observed_races_;
  std::unordered_map<uint32_t, KernelNames> dispatch_kernel_names_;
};

} // namespace rocjitsu::plugins::race_detector
