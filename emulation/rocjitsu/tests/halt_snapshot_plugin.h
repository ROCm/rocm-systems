// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file halt_snapshot_plugin.h
/// @brief Test-only ExecutionPlugin that captures each wavefront's final register
/// state at halt.
///
/// @details Under the hardware-accurate model a wavefront frees its SGPR/VGPR
/// allocations the instant it reaches s_endpgm, so its register file is no longer
/// readable through the (now-free) slot after a kernel finishes. Tests that need to
/// observe a wave's final state install this plugin, which snapshots the full
/// register file from `onAmdgpuWavefrontHalted` — fired while the registers are
/// still live, before they are freed.

#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/plugins/execution_plugin.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"

#include "simdojo/sim/component.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace rocjitsu::test {

/// @brief Captured final state of a single halted wavefront.
struct WavefrontSnapshot {
  uint32_t wf_id = 0;
  uint32_t wg_id = 0;
  uint32_t dispatch_id = 0;
  uint32_t wf_size = 0;
  uint32_t num_sgprs = 0;
  uint32_t num_vgprs = 0;  ///< Architectural VGPR count requested by the dispatch.
  uint32_t vgpr_block = 0; ///< Physical VGPR block size (includes the AccVGPR bank).
  uint64_t exec = 0;
  uint64_t vcc = 0;
  uint32_t status = 0;
  uint32_t mode_raw = 0;          ///< MODE register at halt.
  uint8_t vgpr_msb_mode = 0;      ///< Decoded s_set_vgpr_msb layout at halt.
  uint64_t lds_size_bytes = 0;    ///< Size of the LDS region visible to this wave.
  simdojo::ComponentID cu_id = 0; ///< Originating CU component id (for per-CU grouping).
  std::vector<uint32_t> sgprs;    ///< Full physical SGPR block (sgprs_per_wf), so TTMP
                                  ///< slots past num_sgprs remain visible to tests.
  std::vector<uint32_t> vgprs;    ///< [vgpr_block * wf_size], row-major by physical reg.

  /// @brief Read a captured SGPR by architectural index.
  uint32_t sgpr(uint32_t idx) const { return sgprs.at(idx); }

  /// @brief Read a captured 64-bit SGPR pair (lo at idx, hi at idx+1).
  uint64_t sgpr64(uint32_t idx) const {
    return (static_cast<uint64_t>(sgprs.at(idx + 1)) << 32) | sgprs.at(idx);
  }

  /// @brief Read a captured VGPR lane by physical register index and lane.
  /// @details The physical index space includes the AccVGPR bank on CDNA, so
  /// callers may pass `ACC_VGPR_OFFSET + acc_reg` to read accumulator registers.
  uint32_t vgpr(uint32_t reg, uint32_t lane) const {
    return vgprs.at(static_cast<size_t>(reg) * wf_size + lane);
  }
};

/// @brief Records a WavefrontSnapshot for every wavefront that halts.
/// @details onAmdgpuWavefrontHalted can fire concurrently from multiple partition
/// engine threads in a multi-threaded simulation, so the append is serialized by
/// mutex_. The read accessors below are for POST-RUN inspection only — call them
/// after the engine has drained (no waves still executing); they return references
/// into snapshots_ and must not race a concurrent halt.
class HaltSnapshotPlugin : public ExecutionPlugin {
public:
  HaltSnapshotPlugin() : ExecutionPlugin("halt_snapshot") {}

  const std::vector<WavefrontSnapshot> &snapshots() const { return snapshots_; }

  /// @brief Snapshots taken on a specific CU, in halt order.
  std::vector<const WavefrontSnapshot *> for_cu(const amdgpu::ComputeUnitCore *cu) const {
    std::vector<const WavefrontSnapshot *> out;
    for (const auto &s : snapshots_)
      if (cu && s.cu_id == cu->id())
        out.push_back(&s);
    return out;
  }

  /// @brief First snapshot matching a wavefront slot id, or nullptr.
  const WavefrontSnapshot *by_wf_id(uint32_t wf_id) const {
    for (const auto &s : snapshots_)
      if (s.wf_id == wf_id)
        return &s;
    return nullptr;
  }

  /// @brief First snapshot matching a workgroup id, or nullptr.
  const WavefrontSnapshot *by_wg_id(uint32_t wg_id) const {
    for (const auto &s : snapshots_)
      if (s.wg_id == wg_id)
        return &s;
    return nullptr;
  }

  void onAmdgpuWavefrontHalted(amdgpu::Wavefront &wf) override {
    WavefrontSnapshot s;
    s.wf_id = wf.wf_id();
    s.wg_id = wf.wg_id();
    s.dispatch_id = wf.dispatch_id();
    s.wf_size = wf.wf_size();
    s.num_sgprs = wf.num_sgprs();
    s.num_vgprs = wf.num_vgprs();
    s.exec = wf.exec();
    s.vcc = wf.vcc();
    s.status = wf.status_raw();
    s.mode_raw = wf.mode_raw();
    s.vgpr_msb_mode = wf.vgpr_msb_mode();
    s.lds_size_bytes = wf.lds().size_bytes();
    s.cu_id = wf.cu().id();

    // Read the live register file through the instruction-facing facade rather
    // than the CU's raw storage: the hardware-accurate model hides the full
    // ComputeUnitCore behind Wavefront, so RegisterAccess is the supported path
    // to observe physical registers while the wave is still resident at halt.
    const amdgpu::RegisterAccess regs(wf);

    // Capture the full physical SGPR block (sgprs_per_wf), not just the requested
    // count: TTMP registers alias into the high slots of the block and tests read
    // them by physical index (e.g. TTMP7 at 115, TTMP9 at 117).
    const uint32_t sbase = wf.sgpr_alloc().base;
    const uint32_t sgpr_block = wf.cu().sgprs_per_wf();
    s.sgprs.reserve(sgpr_block);
    for (uint32_t i = 0; i < sgpr_block; ++i)
      s.sgprs.push_back(regs.read_sgpr(sbase + i));

    // Capture the full physical VGPR block (normal + AccVGPR bank) so tests can
    // inspect accumulator registers as well as ordinary VGPRs.
    s.vgpr_block = wf.cu().vgpr_allocation_block_size();
    const uint32_t vbase = wf.vgpr_alloc().base;
    s.vgprs.reserve(static_cast<size_t>(s.vgpr_block) * s.wf_size);
    for (uint32_t r = 0; r < s.vgpr_block; ++r)
      for (uint32_t lane = 0; lane < s.wf_size; ++lane)
        s.vgprs.push_back(regs.read_vgpr(vbase + r, lane));

    // Serialize the append: concurrent halts from different partition threads
    // would otherwise race the vector.
    std::lock_guard<std::mutex> lk(mutex_);
    snapshots_.push_back(std::move(s));
  }

private:
  mutable std::mutex mutex_;
  std::vector<WavefrontSnapshot> snapshots_;
};

/// @brief Build a plugin group holding a HaltSnapshotPlugin, keep a raw handle.
/// @details Returns the group (attach with `soc->set_plugin_group(group)`), and
/// sets @p out to the contained plugin for later inspection.
inline std::shared_ptr<ExecutionPluginGroup> make_halt_snapshot_group(HaltSnapshotPlugin **out) {
  auto group = std::make_shared<ExecutionPluginGroup>();
  auto plugin = std::make_unique<HaltSnapshotPlugin>();
  *out = plugin.get();
  [[maybe_unused]] const bool added = group->add(std::move(plugin));
  assert(added && "HaltSnapshotPlugin add() must not fail on a fresh group");
  return group;
}

/// @brief Counts wavefront dispatches per CU at the moment they are placed.
/// @details Fires from onAmdgpuWavefrontDispatched (before execution), so it
/// observes workgroup-to-CU distribution independent of when waves later halt and
/// free themselves.
class DispatchCountPlugin : public ExecutionPlugin {
public:
  DispatchCountPlugin() : ExecutionPlugin("dispatch_count") {}

  void onAmdgpuWavefrontDispatched(amdgpu::Wavefront &wf) override {
    // Serialize: onAmdgpuWavefrontDispatched can fire from multiple partition
    // engine threads in a multi-threaded simulation.
    std::lock_guard<std::mutex> lk(mutex_);
    counts_[wf.cu().id()]++;
    total_++;
  }

  uint32_t total() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return total_;
  }
  uint32_t for_cu(const amdgpu::ComputeUnitCore *cu) const {
    std::lock_guard<std::mutex> lk(mutex_);
    if (!cu)
      return 0u;
    auto it = counts_.find(cu->id());
    return it == counts_.end() ? 0u : it->second;
  }

private:
  mutable std::mutex mutex_;
  std::unordered_map<simdojo::ComponentID, uint32_t> counts_;
  uint32_t total_ = 0;
};

/// @brief Build a plugin group holding a DispatchCountPlugin, keep a raw handle.
inline std::shared_ptr<ExecutionPluginGroup> make_dispatch_count_group(DispatchCountPlugin **out) {
  auto group = std::make_shared<ExecutionPluginGroup>();
  auto plugin = std::make_unique<DispatchCountPlugin>();
  *out = plugin.get();
  [[maybe_unused]] const bool added = group->add(std::move(plugin));
  assert(added && "DispatchCountPlugin add() must not fail on a fresh group");
  return group;
}

} // namespace rocjitsu::test
