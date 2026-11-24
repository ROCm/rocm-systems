// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/vmem_direct.h"

#include "rocjitsu/vm/amdgpu/coalesced_lane_runs.h"
#include "rocjitsu/vm/amdgpu/l1_vector_cache.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/vm_direct_backing.h"

#include <array>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace rocjitsu::amdgpu {

bool vmem_line_groups_enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("RJ_VMEM_LINE_GROUPS");
    return !value || std::strcmp(value, "0") != 0;
  }();
  return enabled;
}

bool vmem_scalar_direct_enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("RJ_VMEM_SCALAR_DIRECT");
    return vmem_direct_enabled() && (!value || std::strcmp(value, "0") != 0);
  }();
  return enabled;
}

bool vmem_tdm_direct_enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("RJ_VMEM_TDM_DIRECT");
    return vmem_direct_enabled() && !(value && value[0] == '0');
  }();
  return enabled;
}

bool vmem_direct_enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("RJ_VMEM_DIRECT");
    return !value || std::strcmp(value, "0") != 0;
  }();
  return enabled;
}

namespace {

// Per-request scratch arrays sized for a whole wave; only the used prefix is
// ever written, so constructing every element would be wasted work.
template <typename T, size_t N> union Uninitialized {
  Uninitialized() {}
  T items[N];
  T &operator[](size_t i) { return items[i]; }
  const T &operator[](size_t i) const { return items[i]; }
  T *data() { return items; }
  const T *data() const { return items; }
  static constexpr size_t size() { return N; }
};

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
    Uninitialized<VmDirectBacking::Copy, 64> copies;
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

// Lanes of a load that name the same operand read it once (uniform and
// group-broadcast loads); the coalescer would otherwise emit one overlapping
// run, and one extent check and stripe lock, per lane.
constexpr uint32_t kMaxSharedOperands = 4;
constexpr uint32_t kMaxSharedOperandBytes = 64;

struct SharedOperands {
  std::array<uint64_t, kMaxSharedOperands> address;
  std::array<uint64_t, kMaxSharedOperands> lanes;
  uint32_t count = 0;
};

// Whether at most kMaxSharedOperands distinct addresses cover `lanes`, with at
// least one of them shared. Unshared loads usually stop at the fifth lane.
bool find_shared_operands(const uint64_t *addrs, uint64_t lanes, SharedOperands &shared) {
  for (uint64_t remaining = lanes; remaining; remaining &= remaining - 1) {
    const uint32_t lane = std::countr_zero(remaining);
    const uint64_t bit = uint64_t{1} << lane;
    uint32_t i = 0;
    while (i < shared.count && shared.address[i] != addrs[lane])
      ++i;
    if (i == shared.count) {
      if (shared.count == kMaxSharedOperands)
        return false;
      shared.address[i] = addrs[lane];
      shared.lanes[i] = 0;
      ++shared.count;
    }
    shared.lanes[i] |= bit;
  }
  return shared.count < static_cast<uint32_t>(std::popcount(lanes));
}

struct SharedTransfer {
  const VmDirectBacking &backing;
  std::span<const VmDirectBacking::Copy> copies;
  bool guarded = true;
  bool completed = false;
};

std::optional<VmAccessOutcome> try_direct_shared_load(const DirectVectorRequest &request,
                                                      const SharedOperands &shared, uint32_t stride,
                                                      const VmDirectBacking &backing,
                                                      const L2Cache &l2) {
  const uint64_t generation = backing.generation();
  std::array<std::array<uint8_t, kMaxSharedOperandBytes>, kMaxSharedOperands> staged;
  std::array<VmDirectBacking::Copy, kMaxSharedOperands> copies;
  bool guarded = false;
  std::optional<LegacyLocalExtent> extent;
  for (uint32_t i = 0; i < shared.count; ++i) {
    const uint64_t address = shared.address[i];
    if (address > UINT64_MAX - stride)
      return std::nullopt;
    if (!extent || !extent->contains(address, stride)) {
      extent = backing.extent(address, generation);
      if (!extent || !extent->contains(address, stride))
        return std::nullopt;
      const Mtype mtype = effective_mtype(request.instruction_mtype, extent->mtype);
      if (mtype == Mtype::UC || mtype == Mtype::CC)
        return std::nullopt;
    }
    guarded |= extent->owner != LegacyHostExtentOwner::DriverSealedRam;
    copies[i] = {address, extent->host(address), staged[i].data(), stride};
  }
  if (l2.has_unpublished_lines())
    return std::nullopt;
  SharedTransfer transfer{backing, std::span(copies.data(), shared.count), guarded};
  if (!backing.run_admitted(
          generation,
          [](void *context) {
            auto &transfer = *static_cast<SharedTransfer *>(context);
            transfer.completed = transfer.backing.copy(transfer.copies, false, transfer.guarded);
          },
          &transfer) ||
      !transfer.completed)
    return std::nullopt;
  for (uint32_t i = 0; i < shared.count; ++i)
    for (uint64_t lanes = shared.lanes[i]; lanes; lanes &= lanes - 1)
      std::memcpy(request.data + std::countr_zero(lanes) * stride, staged[i].data(), stride);
  return VmAccessOutcome::Complete;
}

// Loads whose lanes leave gaps or repeat inside a few transactions (stride-2
// dwords, grouped gathers, overlapping half-waves) read each touched
// transaction's span once and fill lanes from it; the coalescer would emit
// one run, extent check and stripe lock per lane.
constexpr uint32_t kMaxLineGroups = 8;

struct LineGroups {
  std::array<uint64_t, kMaxLineGroups> base;
  std::array<uint32_t, kMaxLineGroups> lo;
  std::array<uint32_t, kMaxLineGroups> hi;
  uint32_t count = 0;
};

// Whether every active operand lies inside one of at most kMaxLineGroups
// transactions and the lane-ordered runs the coalescer would emit number at
// least twice the transactions; otherwise its direct runs are cheaper than a
// staged copy plus a per-lane fill.
bool find_line_groups(const uint64_t *addrs, uint64_t lanes, uint32_t wf_size, uint32_t stride,
                      uint32_t transaction_bytes, LineGroups &groups) {
  // Gapped and repeated operands break contiguity at the first lane pair;
  // contiguous waves and row tiles do not, so they skip the lane scan.
  const uint32_t first = std::countr_zero(lanes);
  const uint64_t rest = lanes & (lanes - 1);
  if (!rest || addrs[std::countr_zero(rest)] == addrs[first] + stride)
    return false;
  if constexpr (detail::kVectorizedStrideCheck) {
    const bool full_wave =
        (wf_size == 32 && lanes == UINT32_MAX) || (wf_size == 64 && lanes == UINT64_MAX);
    if (full_wave && detail::is_strided_lane_addresses(addrs, wf_size, stride))
      return false;
  }
  const uint64_t mask = ~uint64_t{transaction_bytes - 1};
  uint32_t runs = 0;
  uint64_t previous = 0;
  for (uint64_t remaining = lanes; remaining; remaining &= remaining - 1) {
    const uint64_t address = addrs[std::countr_zero(remaining)];
    if (address > UINT64_MAX - stride || (address & mask) != ((address + stride - 1) & mask))
      return false;
    runs += runs == 0 || address != previous + stride;
    previous = address;
    const uint64_t base = address & mask;
    const uint32_t offset = static_cast<uint32_t>(address - base);
    uint32_t i = 0;
    while (i < groups.count && groups.base[i] != base)
      ++i;
    if (i == groups.count) {
      if (groups.count == kMaxLineGroups)
        return false;
      groups.base[i] = base;
      groups.lo[i] = offset;
      groups.hi[i] = offset + stride;
      ++groups.count;
      continue;
    }
    groups.lo[i] = std::min(groups.lo[i], offset);
    groups.hi[i] = std::max(groups.hi[i], offset + stride);
  }
  return runs >= 2 * groups.count;
}

std::optional<VmAccessOutcome>
try_direct_line_grouped_load(const DirectVectorRequest &request, const LineGroups &groups,
                             uint32_t stride, const VmDirectBacking &backing, const L2Cache &l2) {
  const uint64_t generation = backing.generation();
  std::array<std::array<uint8_t, kMaxMemoryTransactionBytes>, kMaxLineGroups> staged;
  std::array<VmDirectBacking::Copy, kMaxLineGroups> copies;
  bool guarded = false;
  std::optional<LegacyLocalExtent> extent;
  for (uint32_t i = 0; i < groups.count; ++i) {
    const uint64_t address = groups.base[i] + groups.lo[i];
    const uint32_t size = groups.hi[i] - groups.lo[i];
    if (!extent || !extent->contains(address, size)) {
      extent = backing.extent(address, generation);
      if (!extent || !extent->contains(address, size))
        return std::nullopt;
      const Mtype mtype = effective_mtype(request.instruction_mtype, extent->mtype);
      if (mtype == Mtype::UC || mtype == Mtype::CC)
        return std::nullopt;
    }
    guarded |= extent->owner != LegacyHostExtentOwner::DriverSealedRam;
    copies[i] = {address, extent->host(address), staged[i].data() + groups.lo[i], size};
  }
  if (l2.has_unpublished_lines())
    return std::nullopt;
  SharedTransfer transfer{backing, std::span(copies.data(), groups.count), guarded};
  if (!backing.run_admitted(
          generation,
          [](void *context) {
            auto &transfer = *static_cast<SharedTransfer *>(context);
            transfer.completed = transfer.backing.copy(transfer.copies, false, transfer.guarded);
          },
          &transfer) ||
      !transfer.completed)
    return std::nullopt;
  const uint64_t mask = ~uint64_t{request.transactions.bytes() - 1};
  for (uint64_t remaining = request.lanes; remaining; remaining &= remaining - 1) {
    const uint32_t lane = std::countr_zero(remaining);
    const uint64_t base = request.addrs[lane] & mask;
    uint32_t i = 0;
    while (groups.base[i] != base)
      ++i;
    std::memcpy(request.data + lane * stride, staged[i].data() + (request.addrs[lane] - base),
                stride);
  }
  return VmAccessOutcome::Complete;
}

} // namespace

std::optional<VmAccessOutcome> try_direct_vector_access(const DirectVectorRequest &request,
                                                        const VmDirectBacking &backing,
                                                        const L1VectorCache &l1, L2Cache &l2) {
  if (request.lanes == 0)
    return VmAccessOutcome::Complete;
  const uint32_t stride = request.elem_size * request.num_elems;
  if (request.is_load && stride <= kMaxSharedOperandBytes) {
    SharedOperands shared;
    if (find_shared_operands(request.addrs, request.lanes, shared))
      return try_direct_shared_load(request, shared, stride, backing, l2);
    LineGroups groups;
    if (vmem_line_groups_enabled() &&
        find_line_groups(request.addrs, request.lanes, request.wf_size, stride,
                         request.transactions.bytes(), groups))
      return try_direct_line_grouped_load(request, groups, stride, backing, l2);
  }
  // Resolve every transaction before any effect. Resolution may take
  // page-table locks, so it must finish before the admission is held.
  const uint64_t generation = backing.generation();
  Uninitialized<ResolvedRun, 64> runs;
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

std::optional<VmAccessOutcome> try_direct_spans(std::span<const DirectSpan> spans, bool is_load,
                                                Mtype instruction_mtype,
                                                const VmDirectBacking &backing,
                                                const L1VectorCache &l1, L2Cache &l2) {
  if (spans.empty())
    return VmAccessOutcome::Complete;
  // Resolve every span before any effect; resolution may take page-table
  // locks, so it must finish before the admission is held.
  const uint64_t generation = backing.generation();
  std::vector<VmDirectBacking::Copy> copies;
  copies.reserve(spans.size());
  std::optional<LegacyLocalExtent> extent;
  uint64_t lo = UINT64_MAX, hi = 0;
  bool guarded = false;
  for (const DirectSpan &span : spans) {
    if (span.size == 0)
      continue;
    if (!extent || !extent->contains(span.address, span.size)) {
      extent = backing.extent(span.address, generation);
      if (!extent || !extent->contains(span.address, span.size))
        return std::nullopt;
      const Mtype mtype = effective_mtype(instruction_mtype, extent->mtype);
      if (mtype == Mtype::UC || mtype == Mtype::CC)
        return std::nullopt;
    }
    if (!is_load &&
        (l1.may_hold(span.address, span.size) || !l2.may_bypass(span.address, span.size)))
      return std::nullopt;
    guarded |= extent->owner != LegacyHostExtentOwner::DriverSealedRam;
    copies.push_back({span.address, extent->host(span.address), span.data, span.size});
    lo = std::min(lo, span.address);
    hi = std::max(hi, span.address + span.size);
  }
  if (copies.empty())
    return VmAccessOutcome::Complete;
  if (l2.has_unpublished_lines())
    return std::nullopt;

  std::optional<VmRawBackingWriteGate::Permit> permit;
  if (!is_load) {
    permit = backing.begin_write(lo, hi - lo);
    if (!permit)
      return std::nullopt;
  }
  struct SpanTransfer {
    std::span<const VmDirectBacking::Copy> copies;
    const VmDirectBacking &backing;
    bool into_memory = false;
    bool guarded = true;
    bool completed = false;
  } transfer{copies, backing, !is_load, guarded};
  if (!backing.run_admitted(
          generation,
          [](void *context) {
            auto &t = *static_cast<SpanTransfer *>(context);
            t.completed = t.backing.copy(t.copies, t.into_memory, t.guarded);
          },
          &transfer))
    return std::nullopt;
  // Plain transfers are idempotent: after a host fault, the caller's ordinary
  // path repeats the request from its start and reports the fault itself.
  if (!transfer.completed)
    return std::nullopt;
  if (!is_load)
    for (const VmDirectBacking::Copy &copy : copies)
      if (l2.bypass_raced_fill(copy.address, copy.size))
        l2.drop_bypassed_copies(copy.address, copy.size);
  return VmAccessOutcome::Complete;
}

} // namespace rocjitsu::amdgpu
