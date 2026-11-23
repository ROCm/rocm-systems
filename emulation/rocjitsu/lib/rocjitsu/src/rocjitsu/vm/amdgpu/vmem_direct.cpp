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

namespace {

// A run of equal-size spans, consecutive in memory, whose payload bytes lie
// at a constant stride.
struct StridedRun {
  uint64_t address = 0;
  uint32_t piece = 0;
  uint32_t count = 0;
  uint32_t data_offset = 0;
  uint32_t data_stride = 0;
  uint8_t *host = nullptr;
  bool guarded = true;

  [[nodiscard]] uint32_t bytes() const { return piece * count; }
  [[nodiscard]] bool contiguous() const { return count == 1 || data_stride == piece; }
};

constexpr size_t kMaxSwizzledRuns = 512;
constexpr size_t kMaxSwizzledBytes = 64 * 16;

struct SwizzledTransfer {
  std::span<const StridedRun> runs;
  uint8_t *data = nullptr;
  bool is_load = true;
  const VmDirectBacking &backing;
  bool completed = false;

  bool run() const {
    std::array<VmDirectBacking::Copy, kMaxSwizzledRuns> copies;
    std::array<uint8_t, kMaxSwizzledBytes> staged;
    uint32_t staged_used = 0;
    std::array<uint32_t, kMaxSwizzledRuns> stage_offset;
    bool guarded = false;
    for (size_t i = 0; i < runs.size(); ++i) {
      const StridedRun &run = runs[i];
      guarded |= run.guarded;
      void *local = data + run.data_offset;
      if (!run.contiguous()) {
        stage_offset[i] = staged_used;
        local = staged.data() + staged_used;
        staged_used += run.bytes();
        if (!is_load)
          for (uint32_t k = 0; k < run.count; ++k)
            std::memcpy(static_cast<uint8_t *>(local) + k * run.piece,
                        data + run.data_offset + k * run.data_stride, run.piece);
      }
      copies[i] = {run.address, run.host, local, run.bytes()};
    }
    if (!backing.copy(std::span(copies.data(), runs.size()), !is_load, guarded))
      return false;
    if (is_load)
      for (size_t i = 0; i < runs.size(); ++i)
        if (!runs[i].contiguous())
          for (uint32_t k = 0; k < runs[i].count; ++k)
            std::memcpy(data + runs[i].data_offset + k * runs[i].data_stride,
                        staged.data() + stage_offset[i] + k * runs[i].piece, runs[i].piece);
    return true;
  }
};

} // namespace

std::optional<VmAccessOutcome> try_direct_swizzled_access(const VectorMemState &state,
                                                          uint64_t lanes, uint8_t *data,
                                                          const VmDirectBacking &backing,
                                                          const L1VectorCache &l1, L2Cache &l2) {
  if (lanes == 0)
    return VmAccessOutcome::Complete;
  if (size_t{state.wf_size} * state.num_elems * state.elem_size > kMaxSwizzledBytes)
    return std::nullopt;
  std::array<StridedRun, kMaxSwizzledRuns> runs;
  uint32_t count = 0;
  bool overflow = false;
  state.for_each_swizzled_span(lanes, [&](uint64_t address, uint32_t data_offset, uint32_t size) {
    if (overflow)
      return;
    if (count) {
      StridedRun &run = runs[count - 1];
      const uint32_t stride = run.count == 1 ? data_offset - run.data_offset : run.data_stride;
      if (size == run.piece && address == run.address + run.bytes() &&
          data_offset > run.data_offset && stride >= run.piece &&
          data_offset == run.data_offset + run.count * stride &&
          run.bytes() + size <= kMaxMemoryTransactionBytes * 8) {
        run.data_stride = stride;
        ++run.count;
        return;
      }
    }
    if (count == runs.size()) {
      overflow = true;
      return;
    }
    runs[count++] = {.address = address, .piece = size, .count = 1, .data_offset = data_offset};
  });
  if (overflow || count == 0)
    return std::nullopt;

  const uint64_t generation = backing.generation();
  std::optional<LegacyLocalExtent> extent;
  uint64_t lo = UINT64_MAX, hi = 0;
  uint32_t staged_bytes = 0;
  for (uint32_t i = 0; i < count; ++i) {
    StridedRun &run = runs[i];
    const uint64_t size = run.bytes();
    if (!extent || !extent->contains(run.address, size)) {
      extent = backing.extent(run.address, generation);
      if (!extent || !extent->contains(run.address, size))
        return std::nullopt;
      const Mtype mtype = effective_mtype(state.mtype, extent->mtype);
      if (mtype == Mtype::UC || mtype == Mtype::CC)
        return std::nullopt;
    }
    if (!state.is_load && (l1.may_hold(run.address, size) || !l2.may_bypass(run.address, size)))
      return std::nullopt;
    run.host = extent->host(run.address);
    run.guarded = extent->owner != LegacyHostExtentOwner::DriverSealedRam;
    if (!run.contiguous())
      staged_bytes += run.bytes();
    lo = std::min(lo, run.address);
    hi = std::max(hi, run.address + size);
  }
  if (staged_bytes > kMaxSwizzledBytes || l2.has_unpublished_lines())
    return std::nullopt;

  std::optional<VmRawBackingWriteGate::Permit> permit;
  if (!state.is_load) {
    permit = backing.begin_write(lo, hi - lo);
    if (!permit)
      return std::nullopt;
  }
  SwizzledTransfer transfer{std::span<const StridedRun>(runs.data(), count), data, state.is_load,
                            backing};
  if (!backing.run_admitted(
          generation,
          [](void *context) {
            auto &transfer = *static_cast<SwizzledTransfer *>(context);
            transfer.completed = transfer.run();
          },
          &transfer))
    return std::nullopt;
  if (!transfer.completed)
    return std::nullopt;
  if (!state.is_load)
    for (const StridedRun &run : transfer.runs)
      if (l2.bypass_raced_fill(run.address, run.bytes()))
        l2.drop_bypassed_copies(run.address, run.bytes());
  return VmAccessOutcome::Complete;
}

} // namespace rocjitsu::amdgpu
