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
#include "rocjitsu/vm/amdgpu/dispatch_entry.h"
#include "rocjitsu/vm/amdgpu/workgroup_key.h"
#include "util/bit.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>
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

  /// @brief Resources selected for one workgroup.
  struct WorkgroupPlacement {
    ComputeUnitCore *cu = nullptr;
    Lds *lds = nullptr;
    uint32_t lds_base = 0;
  };

  explicit ShaderProcessorInput(std::vector<ComputeUnitCore *> cus) : cus_(std::move(cus)) {
    cu_to_wgp_.assign(cus_.size(), std::numeric_limits<size_t>::max());
    for (size_t i = 0; i + 1 < cus_.size(); i += 2) {
      auto wgp = std::make_unique<WgpResource>(cus_[i], cus_[i + 1]);
      cu_to_wgp_[i] = wgps_.size();
      cu_to_wgp_[i + 1] = wgps_.size();
      wgps_.push_back(std::move(wgp));
    }
  }

  void set_apertures(uint64_t shared_base, uint64_t shared_limit, uint64_t private_base,
                     uint64_t private_limit) {
    for (auto *cu : cus_)
      cu->set_apertures(shared_base, shared_limit, private_base, private_limit);
  }

  /// @brief Enqueue a workgroup from an ACE pipe.
  void enqueue_wg(uint32_t pipe_id, uint32_t global_wg_id, DispatchEntry *entry) {
    if (entry->wgp_mode)
      throw std::invalid_argument("legacy SPI queue path does not support WGP-mode workgroups");
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
      auto placement = allocate_workgroup(*wg.entry, wg.global_wg_id);
      if (!placement)
        continue;

      ComputeUnitCore *cu = placement->cu;
      uint32_t lds_base = placement->lds_base;
      cu->begin_workgroup(wg.entry->dispatch_id, wg.global_wg_id, wg.entry->wfs_per_workgroup);
      std::vector<Wavefront *> wg_wfs;
      wg_wfs.reserve(wg.entry->wfs_per_workgroup);
      for (uint32_t w = 0; w < wg.entry->wfs_per_workgroup; ++w) {
        Wavefront *wf = cu->dispatch_wf(wg.global_wg_id, wg.entry->kernel_entry_pc,
                                        wg.entry->sgprs_per_wf, wg.entry->vgprs_per_wf);
        assert(wf && "dispatch_wf failed after select_cu returned a CU");
        wf->set_lds_base(lds_base);
        wf->set_lds(placement->lds);
        wf->set_dispatch_id(wg.entry->dispatch_id);
        wf->set_process_id(wg.entry->process_id);
        wf->set_exec(initial_exec_mask_for_wave(*wg.entry, wg.global_wg_id, w, cu->wf_size()));
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

  /// @brief Legacy: select a CU with capacity for direct dispatch.
  ///
  /// @details Used by dispatch_workgroups() fallback path when SPIs are not
  /// used for queuing. Renamed from dispatch_workgroup().
  ComputeUnitCore *select_cu(const DispatchEntry &entry) {
    if (entry.wgp_mode)
      return nullptr;
    for (size_t attempt = 0; attempt < cus_.size(); ++attempt) {
      size_t idx = (next_cu_ + attempt) % cus_.size();
      auto *cu = cus_[idx];
      cu->retire_halted_wfs();
      const size_t wgp_index = cu_to_wgp_[idx];
      if (wgp_index != std::numeric_limits<size_t>::max() &&
          wgps_[wgp_index]->active_workgroups != 0)
        continue;
      if (!cu->can_accept_workgroup(entry.wfs_per_workgroup, entry.group_segment_fixed_size))
        continue;
      next_cu_ = (idx + 1) % cus_.size();
      return cu;
    }
    return nullptr;
  }

  /// @brief Legacy: direct dispatch (kept for backward compatibility).
  ComputeUnitCore *dispatch_workgroup(const DispatchEntry &entry) { return select_cu(entry); }

  /// @brief Select and reserve all compute/LDS resources for one workgroup.
  std::optional<WorkgroupPlacement> allocate_workgroup(const DispatchEntry &entry,
                                                       uint32_t global_wg_id) {
    if (!entry.wgp_mode) {
      auto *cu = select_cu(entry);
      if (!cu)
        return std::nullopt;
      return WorkgroupPlacement{cu, &cu->lds(), cu->allocate_lds(entry.group_segment_fixed_size)};
    }

    if (wgps_.empty())
      return std::nullopt;
    const uint32_t aligned = util::align_up(entry.group_segment_fixed_size, 256u);
    for (size_t attempt = 0; attempt < wgps_.size(); ++attempt) {
      size_t wgp_index = (next_wgp_ + attempt) % wgps_.size();
      auto &wgp = *wgps_[wgp_index];
      wgp.cu0->retire_halted_wfs();
      wgp.cu1->retire_halted_wfs();

      // A WGP allocation cannot overlap CU-mode residents or cluster-pinned
      // CU-local LDS state. Existing WGP-mode workgroups may share the pool.
      if (wgp.active_workgroups == 0 &&
          (wgp.cu0->has_active_wfs() || wgp.cu1->has_active_wfs() ||
           wgp.cu0->lds_allocation_pinned() || wgp.cu1->lds_allocation_pinned()))
        continue;
      if (static_cast<uint64_t>(wgp.next_lds_alloc) + aligned > wgp.lds.size_bytes())
        continue;

      ComputeUnitCore *selected = nullptr;
      for (uint32_t half = 0; half < 2; ++half) {
        auto *candidate = ((next_wgp_half_ + half) & 1u) == 0 ? wgp.cu0 : wgp.cu1;
        if (candidate->can_accept_workgroup(entry.wfs_per_workgroup, 0)) {
          selected = candidate;
          next_wgp_half_ = (next_wgp_half_ + half + 1) & 1u;
          break;
        }
      }
      if (!selected)
        continue;

      const uint32_t lds_base = wgp.next_lds_alloc;
      wgp.lds.zero_range(lds_base, aligned);
      wgp.next_lds_alloc += aligned;
      ++wgp.active_workgroups;
      resident_wgp_workgroups_[wg_key(entry.dispatch_id, global_wg_id)] = WgpReservation{wgp_index};
      next_wgp_ = (wgp_index + 1) % wgps_.size();
      return WorkgroupPlacement{selected, &wgp.lds, lds_base};
    }
    return std::nullopt;
  }

  /// @brief Release a workgroup's paired-WGP reservation.
  bool release_wgp_workgroup(uint32_t dispatch_id, uint32_t global_wg_id) {
    auto resident = resident_wgp_workgroups_.find(wg_key(dispatch_id, global_wg_id));
    if (resident == resident_wgp_workgroups_.end())
      return false;
    const size_t wgp_index = resident->second.wgp_index;
    resident_wgp_workgroups_.erase(resident);
    assert(wgp_index < wgps_.size());
    auto &wgp = *wgps_[wgp_index];
    assert(wgp.active_workgroups != 0);
    if (--wgp.active_workgroups == 0)
      wgp.next_lds_alloc = 0;
    return true;
  }

  uint32_t max_wgp_lds_bytes() const {
    uint64_t max_bytes = 0;
    for (const auto &wgp : wgps_)
      max_bytes = std::max(max_bytes, static_cast<uint64_t>(wgp->lds.size_bytes()));
    return static_cast<uint32_t>(
        std::min<uint64_t>(max_bytes, std::numeric_limits<uint32_t>::max()));
  }

  const std::vector<ComputeUnitCore *> &compute_units() const { return cus_; }

private:
  struct WgpResource {
    WgpResource(ComputeUnitCore *first, ComputeUnitCore *second)
        : cu0(first), cu1(second), lds(first->config().lds_size_kb + second->config().lds_size_kb) {
    }

    ComputeUnitCore *cu0 = nullptr;
    ComputeUnitCore *cu1 = nullptr;
    Lds lds;
    uint32_t next_lds_alloc = 0;
    uint32_t active_workgroups = 0;
  };

  struct WgpReservation {
    size_t wgp_index = 0;
  };

  std::vector<ComputeUnitCore *> cus_;
  size_t next_cu_ = 0;
  std::vector<std::unique_ptr<WgpResource>> wgps_;
  std::vector<size_t> cu_to_wgp_;
  size_t next_wgp_ = 0;
  uint32_t next_wgp_half_ = 0;
  std::unordered_map<uint64_t, WgpReservation> resident_wgp_workgroups_;
  std::vector<std::deque<WgRequest>> pipe_queues_;
  size_t next_pipe_ = 0;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_SPI_H_
