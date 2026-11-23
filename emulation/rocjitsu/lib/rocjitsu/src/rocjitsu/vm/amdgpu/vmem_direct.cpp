// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/vmem_direct.h"

#include "rocjitsu/vm/amdgpu/coalesced_lane_runs.h"
#include "rocjitsu/vm/amdgpu/l1_vector_cache.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/vm_direct_backing.h"

#include <array>
#include <cstdlib>
#include <cstring>

namespace rocjitsu::amdgpu {

bool vmem_direct_enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("RJ_VMEM_DIRECT");
    return !value || std::strcmp(value, "0") != 0;
  }();
  return enabled;
}

namespace {

struct ResolvedRun {
  CoalescedLaneRun run;
  uint8_t *host = nullptr;
  // Only driver-sealed RAM is pinned by the mapping lease.
  bool guarded = true;
};

struct Transfer {
  const DirectVectorRequest &request;
  const VmDirectBacking &backing;
  std::span<const ResolvedRun> runs;
  uint32_t stride = 0;

  // False when a guarded copy faulted; earlier transactions may have moved.
  bool run() const {
    // A reordered run fits one transaction and moves through a staging line.
    std::array<VmDirectBacking::Copy, 64> copies;
    std::array<std::array<uint8_t, kMaxMemoryTransactionBytes>, 64> staged;
    bool guarded = false;
    for (size_t i = 0; i < runs.size(); ++i) {
      const ResolvedRun &resolved = runs[i];
      const CoalescedLaneRun &run = resolved.run;
      guarded |= resolved.guarded;
      void *local = request.data + run.first_lane * stride;
      if (!run.lane_ordered) {
        local = staged[i].data();
        if (!request.is_load)
          gather_lane_run(run.address, run.lane_mask, run.size, request.addrs, request.data, stride,
                          staged[i].data());
      }
      copies[i] = {run.address, resolved.host, local, run.size};
    }
    if (!backing.copy(std::span(copies.data(), runs.size()), !request.is_load, guarded))
      return false;
    if (request.is_load)
      for (size_t i = 0; i < runs.size(); ++i)
        if (!runs[i].run.lane_ordered)
          scatter_lane_run(runs[i].run.address, runs[i].run.lane_mask, runs[i].run.size,
                           request.addrs, staged[i].data(), stride, request.data);
    return true;
  }
  bool completed = false;
};

} // namespace

std::optional<VmAccessOutcome> try_direct_vector_access(const DirectVectorRequest &request,
                                                        const VmDirectBacking &backing,
                                                        const L1VectorCache &l1, L2Cache &l2) {
  if (request.lanes == 0)
    return VmAccessOutcome::Complete;
  const uint32_t stride = request.elem_size * request.num_elems;
  // Resolve every transaction before any effect. Resolution may take
  // page-table locks, so it must finish before the admission is held.
  const uint64_t generation = backing.generation();
  std::array<ResolvedRun, 64> runs;
  uint32_t count = 0;
  std::optional<LegacyLocalExtent> extent;
  uint64_t lo = UINT64_MAX, hi = 0;
  const bool resolved = for_each_coalesced_lane_run(
      request.addrs, request.lanes, request.wf_size, stride, request.transactions,
      [&](const CoalescedLaneRun &run) {
        if (count == runs.size() || run.size == 0)
          return false;
        if (!extent || !extent->contains(run.address, run.size)) {
          extent = backing.extent(run.address, generation);
          if (!extent || !extent->contains(run.address, run.size))
            return false;
          const Mtype mtype = effective_mtype(request.instruction_mtype, extent->mtype);
          if (mtype == Mtype::UC || mtype == Mtype::CC)
            return false;
        }
        if (!request.is_load &&
            (l1.may_hold(run.address, run.size) || !l2.may_bypass(run.address, run.size)))
          return false;
        runs[count++] = {run, extent->host(run.address),
                         extent->owner != LegacyHostExtentOwner::DriverSealedRam};
        lo = std::min(lo, run.address);
        hi = std::max(hi, run.address + run.size);
        return true;
      });
  if (!resolved || count == 0 || l2.has_unpublished_lines())
    return std::nullopt;

  std::optional<VmRawBackingWriteGate::Permit> permit;
  if (!request.is_load) {
    permit = backing.begin_write(lo, hi - lo);
    if (!permit)
      return std::nullopt;
  }
  Transfer transfer{request, backing, std::span(runs.data(), count), stride};
  if (!backing.run_admitted(
          generation,
          [](void *context) {
            auto &transfer = *static_cast<Transfer *>(context);
            transfer.completed = transfer.run();
          },
          &transfer))
    return std::nullopt;
  // Plain transfers are idempotent: after a host fault, the routed path
  // repeats the request from its start and reports the fault itself.
  if (!transfer.completed)
    return std::nullopt;
  if (!request.is_load)
    for (const ResolvedRun &resolved_run : transfer.runs)
      if (l2.bypass_raced_fill(resolved_run.run.address, resolved_run.run.size))
        l2.drop_bypassed_copies(resolved_run.run.address, resolved_run.run.size);
  return VmAccessOutcome::Complete;
}

} // namespace rocjitsu::amdgpu
