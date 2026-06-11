// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file spi.h
/// @brief Shader Processor Input (SPI) — workgroup scheduler within a Shader Engine.
///
/// @details The SPI receives workgroups from the Asynchronous Compute Engine (ACE)
/// and assigns them to Compute Units based on resource availability (WF slots,
/// VGPR, SGPR, LDS). It maintains per-queue ordering (strict WG ID order per
/// AQL queue) and selects the oldest ready WG from the least recently used queue
/// for dispatch. All wavefronts of a workgroup land on the same CU.

#ifndef ROCJITSU_VM_AMDGPU_SPI_H_
#define ROCJITSU_VM_AMDGPU_SPI_H_

#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/cpu_dispatch_pool.h"
#include "rocjitsu/vm/amdgpu/dispatch_entry.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace rocjitsu {
namespace amdgpu {

/// @brief Shader Processor Input — queues workgroups and dispatches to CUs.
class ShaderProcessorInput {
public:
  using WfInitFn = std::function<void(ComputeUnitCore *, Wavefront *, const DispatchEntry &,
                                      uint32_t wg_id, uint32_t wf_idx)>;
  using WgDispatchFn =
      std::function<void(uint32_t wg_id, const DispatchEntry &, std::span<Wavefront *> wfs)>;

  /// @brief A queued workgroup request from an ACE pipe.
  struct WgRequest {
    uint32_t pipe_id;
    uint32_t global_wg_id;
    DispatchEntry *entry;
  };

  explicit ShaderProcessorInput(std::vector<ComputeUnitCore *> cus) : cus_(std::move(cus)) {}

  void set_apertures(uint64_t shared_base, uint64_t shared_limit, uint64_t private_base,
                     uint64_t private_limit) {
    for (auto *cu : cus_)
      cu->set_apertures(shared_base, shared_limit, private_base, private_limit);
  }

  /// @brief Enqueue a workgroup from an ACE pipe.
  void enqueue_wg(uint32_t pipe_id, uint32_t global_wg_id, DispatchEntry *entry) {
    if (pipe_id >= pipe_queues_.size())
      pipe_queues_.resize(pipe_id + 1);
    pipe_queues_[pipe_id].push_back({pipe_id, global_wg_id, entry});
  }

  /// @brief Try to dispatch the next ready WG to a CU.
  ///
  /// @details Selects the oldest WG from the least recently used queue
  /// that has a CU with sufficient resources. Creates WFs on the selected CU.
  bool try_dispatch(const WfInitFn &init_wf, const WgDispatchFn &on_wg_dispatch = {}) {
    size_t num_pipes = pipe_queues_.size();
    if (num_pipes == 0)
      return false;

    for (size_t attempt = 0; attempt < num_pipes; ++attempt) {
      size_t idx = (next_pipe_ + attempt) % num_pipes;
      auto &q = pipe_queues_[idx];
      if (q.empty())
        continue;

      auto &wg = q.front();
      ComputeUnitCore *cu = select_cu(*wg.entry);
      if (!cu)
        continue;

      uint32_t lds_base = cu->allocate_lds(wg.entry->group_segment_fixed_size);
      cu->begin_workgroup(wg.entry->dispatch_id, wg.global_wg_id, wg.entry->wfs_per_workgroup);
      std::vector<Wavefront *> wg_wfs;
      wg_wfs.reserve(wg.entry->wfs_per_workgroup);
      for (uint32_t w = 0; w < wg.entry->wfs_per_workgroup; ++w) {
        Wavefront *wf = cu->dispatch_wf(wg.global_wg_id, wg.entry->kernel_entry_pc,
                                        wg.entry->sgprs_per_wf, wg.entry->vgprs_per_wf);
        assert(wf && "dispatch_wf failed after select_cu returned a CU");
        wf->set_lds_base(lds_base);
        wf->set_dispatch_id(wg.entry->dispatch_id);
        wf->set_process_id(wg.entry->process_id);
        init_wf(cu, wf, *wg.entry, wg.global_wg_id, w);
        wg_wfs.push_back(wf);
      }
      if (on_wg_dispatch)
        on_wg_dispatch(wg.global_wg_id, *wg.entry, std::span<Wavefront *>(wg_wfs));
      q.pop_front();
      next_pipe_ = (idx + 1) % num_pipes;
      return true;
    }
    return false;
  }

  /// @brief Step each CU once (one round-robin pass within this SE).
  bool step() {
    bool any_active = false;
    for (auto *cu : cus_) {
      if (cu->has_active_wfs()) {
        cu->step();
        any_active = true;
      }
    }
    return any_active;
  }

  /// @brief Check whether any CU in this SE has active wavefronts.
  bool has_active_cus() const {
    for (auto *cu : cus_)
      if (cu->has_active_wfs())
        return true;
    return false;
  }

  /// @brief Count CUs in this SE with active wavefronts.
  uint32_t active_cu_count() const {
    uint32_t count = 0;
    for (auto *cu : cus_)
      if (cu->has_active_wfs())
        ++count;
    return count;
  }

  /// @brief Append this SE's active CUs to @p out for CP-wide co-scheduling.
  ///
  /// @details Lets the command processor gather every active CU across all of
  /// its SEs into a single dispatch-pool fork-join, instead of one (often
  /// under-filled) fork-join per SE. Larger task sets balance better across
  /// host threads and cut the number of barriers per CP step.
  void collect_active_cus(std::vector<ComputeUnitCore *> &out) const {
    for (auto *cu : cus_)
      if (cu->has_active_wfs())
        out.push_back(cu);
  }

  /// @brief Execute one functional quantum on each active CU.
  ///
  /// @details Drives wavefront execution for this SE's CUs, fanning out
  /// across up to @p threads host threads via an SPI-owned worker pool.
  /// @returns true if any CU executed instructions.
  bool run_active_cus_once(uint32_t threads) {
    std::vector<ComputeUnitCore *> active;
    active.reserve(cus_.size());
    for (auto *cu : cus_) {
      if (cu->has_active_wfs())
        active.push_back(cu);
    }
    if (active.empty())
      return false;

    uint32_t effective_threads = std::min<uint32_t>(threads, static_cast<uint32_t>(active.size()));
    if (effective_threads > 1) {
      if (!dispatch_pool_ || dispatch_pool_->thread_count() < effective_threads)
        dispatch_pool_ = std::make_unique<CpuDispatchPool>(effective_threads);
      dispatch_pool_->run(active, effective_threads);
    } else {
      for (auto *cu : active)
        cu->run_quantum();
    }
    return true;
  }

  /// @brief Tear down the worker pool (e.g., on thread-count change).
  void reset_dispatch_pool() { dispatch_pool_.reset(); }

  /// @brief Check if any WGs are queued or any CU is active.
  bool has_pending() const {
    for (auto &q : pipe_queues_)
      if (!q.empty())
        return true;
    for (auto *cu : cus_)
      if (cu->has_active_wfs())
        return true;
    return false;
  }

  /// @brief Run all CUs to idle (functional mode, for test harness use).
  void run_to_idle() {
    bool progress = true;
    while (progress) {
      progress = false;
      for (auto *cu : cus_) {
        if (cu->has_active_wfs()) {
          cu->step();
          progress = true;
        }
      }
    }
  }

  /// @brief Legacy: select a CU with capacity for direct dispatch.
  ///
  /// @details Used by dispatch_workgroups() fallback path when SPIs are not
  /// used for queuing. Renamed from dispatch_workgroup().
  ComputeUnitCore *select_cu(const DispatchEntry &entry) {
    for (size_t attempt = 0; attempt < cus_.size(); ++attempt) {
      size_t idx = (next_cu_ + attempt) % cus_.size();
      auto *cu = cus_[idx];
      cu->retire_halted_wfs();
      if (!cu->can_accept_workgroup(entry.wfs_per_workgroup, entry.group_segment_fixed_size))
        continue;
      next_cu_ = (idx + 1) % cus_.size();
      return cu;
    }
    return nullptr;
  }

  /// @brief Legacy: direct dispatch (kept for backward compatibility).
  ComputeUnitCore *dispatch_workgroup(const DispatchEntry &entry) { return select_cu(entry); }

  const std::vector<ComputeUnitCore *> &compute_units() const { return cus_; }

private:
  std::vector<ComputeUnitCore *> cus_;
  size_t next_cu_ = 0;
  std::vector<std::deque<WgRequest>> pipe_queues_;
  size_t next_pipe_ = 0;
  std::unique_ptr<CpuDispatchPool> dispatch_pool_; ///< Drives wavefront execution on host threads.
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_SPI_H_
