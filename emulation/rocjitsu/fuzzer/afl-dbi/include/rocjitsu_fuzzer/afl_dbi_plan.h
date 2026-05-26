#pragma once

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/patch/instruction_builder.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace rocjitsu::fuzzer::afl_dbi {

inline constexpr uint32_t kMapSize = 65536;
inline constexpr uint32_t kDeviceStart = kMapSize / 2;
inline constexpr uint32_t kCoverageSlots = kMapSize / 2;
inline constexpr uint32_t kPreviousBbSlots = 1u << 20;
inline constexpr uint32_t kDeviceStateSlots = kCoverageSlots + kPreviousBbSlots;
inline constexpr uint32_t kPreviousBbByteOffset = kCoverageSlots * sizeof(uint32_t);
inline constexpr uint32_t kEntryCounterSlot = 0;
inline constexpr uint32_t kFirstEdgeCounterSlot = 1;
inline constexpr uint32_t kHashEdgeSlots = kCoverageSlots;
inline constexpr uint32_t kHashEdgeSlotMask = kHashEdgeSlots - 1;
inline constexpr uint32_t kUnknownDeviceImageIndex = std::numeric_limits<uint32_t>::max();
inline constexpr uint32_t kMaxInlinePositiveImm = 64;
inline constexpr uint32_t kMaxInlineCounterSlot = kMaxInlinePositiveImm / sizeof(uint32_t);
inline constexpr uint32_t kMaxFixedCounterSlot = kCoverageSlots - 1;
inline constexpr uint32_t kMaxLogicalBranchEdgesPerSite = 2;
// Baseline product cap for auto-selected non-previous-BB branch probes. This is
// a logical edge budget, not a bitmap limit: a conditional branch site consumes
// two edges and an unconditional branch site consumes one. Previous-BB branch
// plans use the candidate-derived writer budget below and rely on per-site
// liveness, descriptor, scratch, placement, and fallback checks for safety.
inline constexpr uint32_t kAdaptiveBranchEdgeSiteLimit = 100;
// Auto-selected previous-BB branch probes no longer use a fixed writer-site
// product cap. The planner first classifies EXEC-conditioned sites into the
// fixed-counter safety fallback, then attempts every remaining direct branch
// candidate. Individual sites still fail closed through liveness/resource
// selection, descriptor/private-segment patching, and trampoline placement; when
// the smaller fixed counter is safe, the adaptive fallback keeps harvesting
// coverage for rejected previous-BB sites.
inline constexpr uint32_t kAdaptivePreviousBbBranchSiteLimit =
    std::numeric_limits<uint32_t>::max();
inline constexpr uint32_t kFixedCounterBranchEdgeFallbackMapBudget =
    kMaxFixedCounterSlot - kFirstEdgeCounterSlot + 1;
// When the safer previous-BB branch budget is exhausted, the default strategy
// keeps harvesting branch feedback with smaller fixed-counter probes. The auto
// budget is derived from the per-kernel candidate branch edge count and capped
// by the fixed-counter map capacity; placement and descriptor planning still
// fail closed if the selected fallback set cannot be installed.
inline constexpr uint32_t kAdaptiveFixedCounterBranchEdgeFallbackLimit =
    kFixedCounterBranchEdgeFallbackMapBudget;

inline constexpr uint32_t
adaptive_fixed_counter_branch_edge_fallback_budget(uint32_t candidate_edges) {
  return std::min(candidate_edges, kFixedCounterBranchEdgeFallbackMapBudget);
}
inline constexpr uint16_t kScalarExecLo = 126;
inline constexpr uint16_t kScalarExecHi = kScalarExecLo + 1;
inline constexpr uint16_t kScalarVccLo = 106;
inline constexpr uint16_t kScalarTtmp9 = 117;
inline constexpr uint16_t kScalarLiteral = 255;

inline constexpr uint32_t register_count_to_granulated(uint32_t registers,
                                                       uint32_t granularity) {
  if (registers == 0)
    return 0;
  return (registers + granularity - 1) / granularity - 1;
}

inline constexpr uint32_t granulated_to_register_count(uint32_t granulated,
                                                       uint32_t granularity) {
  return (granulated + 1) * granularity;
}

enum class EdgeCounterPolicy {
  FirstActiveLane,
  EveryActiveLane,
  DistinctPreviousBb,
};

enum class PreviousBbBranchAggregateLimitKind {
  NotPreviousBbPolicy,
  DebugConfiguredBudget,
  None,
  EdgeCap,
  SiteCap,
  EdgeAndSiteCap,
};

inline constexpr const char *previous_bb_branch_aggregate_limit_kind_name(
    PreviousBbBranchAggregateLimitKind kind) {
  switch (kind) {
  case PreviousBbBranchAggregateLimitKind::NotPreviousBbPolicy:
    return "not-previous-bb-policy";
  case PreviousBbBranchAggregateLimitKind::DebugConfiguredBudget:
    return "debug-configured-budget";
  case PreviousBbBranchAggregateLimitKind::None:
    return "none";
  case PreviousBbBranchAggregateLimitKind::EdgeCap:
    return "edge-cap";
  case PreviousBbBranchAggregateLimitKind::SiteCap:
    return "site-cap";
  case PreviousBbBranchAggregateLimitKind::EdgeAndSiteCap:
    return "edge-and-site-cap";
  }
  return "unknown";
}

inline constexpr uint32_t budget_overage(uint32_t candidate, uint32_t budget) {
  return candidate > budget ? candidate - budget : 0;
}

inline constexpr uint32_t
previous_bb_branch_site_derived_edge_budget(uint32_t site_budget) {
  if (site_budget >
      std::numeric_limits<uint32_t>::max() / kMaxLogicalBranchEdgesPerSite)
    return std::numeric_limits<uint32_t>::max();
  return site_budget * kMaxLogicalBranchEdgesPerSite;
}

// Central budget decision for the adaptive branch strategy. The current auto
// path still uses conservative caps, but future DBI-side aggregate proofs should
// evolve this decision instead of adding another user-visible coverage mode.
struct PreviousBbBranchAggregateBudget {
  uint32_t candidate_edges = 0;
  uint32_t candidate_sites = 0;
  uint32_t edge_budget = 0;
  uint32_t site_budget = std::numeric_limits<uint32_t>::max();
  uint32_t edge_over_budget = 0;
  uint32_t site_over_budget = 0;
  uint32_t fixed_counter_fallback_budget = 0;
  bool previous_bb_policy = false;
  bool edge_budget_auto = false;
  bool site_budget_auto = false;
  bool fixed_counter_fallback_enabled = false;

  constexpr PreviousBbBranchAggregateLimitKind limit_kind() const {
    if (!previous_bb_policy)
      return PreviousBbBranchAggregateLimitKind::NotPreviousBbPolicy;
    if (!edge_budget_auto && !site_budget_auto)
      return PreviousBbBranchAggregateLimitKind::DebugConfiguredBudget;
    const bool edge_limited = edge_over_budget != 0;
    const bool site_limited = site_over_budget != 0;
    if (edge_limited && site_limited)
      return PreviousBbBranchAggregateLimitKind::EdgeAndSiteCap;
    if (edge_limited)
      return PreviousBbBranchAggregateLimitKind::EdgeCap;
    if (site_limited)
      return PreviousBbBranchAggregateLimitKind::SiteCap;
    return PreviousBbBranchAggregateLimitKind::None;
  }

  constexpr bool edge_budget_exhausted(uint32_t selected_edges,
                                       uint32_t site_edge_count) const {
    return site_edge_count > edge_budget ||
           selected_edges > edge_budget - site_edge_count;
  }

  constexpr bool previous_bb_site_budget_exhausted(
      uint32_t selected_sites) const {
    return previous_bb_policy && selected_sites >= site_budget;
  }

  constexpr bool fixed_counter_fallback_available(
      uint32_t selected_fallback_edges, uint32_t site_edge_count) const {
    return previous_bb_policy && fixed_counter_fallback_enabled &&
           site_edge_count <= fixed_counter_fallback_budget &&
           selected_fallback_edges <=
               fixed_counter_fallback_budget - site_edge_count;
  }
};

inline constexpr PreviousBbBranchAggregateBudget
make_previous_bb_branch_aggregate_budget(
    uint32_t candidate_edges, uint32_t candidate_sites, uint32_t edge_budget,
    uint32_t site_budget, bool edge_budget_auto, bool site_budget_auto,
    bool previous_bb_policy, bool fixed_counter_fallback_enabled,
    uint32_t fixed_counter_fallback_budget) {
  PreviousBbBranchAggregateBudget budget;
  budget.candidate_edges = candidate_edges;
  budget.candidate_sites = candidate_sites;
  budget.edge_budget = edge_budget;
  budget.site_budget = site_budget;
  budget.previous_bb_policy = previous_bb_policy;
  budget.edge_budget_auto = edge_budget_auto;
  budget.site_budget_auto = site_budget_auto;
  budget.fixed_counter_fallback_enabled = fixed_counter_fallback_enabled;
  budget.fixed_counter_fallback_budget =
      fixed_counter_fallback_enabled ? fixed_counter_fallback_budget : 0;
  if (previous_bb_policy) {
    budget.edge_over_budget = budget_overage(candidate_edges, edge_budget);
    budget.site_over_budget = budget_overage(candidate_sites, site_budget);
  }
  return budget;
}

enum class ProbeMemoryModel {
  Unsupported,
  Gfx9FlatGlobal,
  Gfx10FlatGlobal,
  Gfx11GlobalSaddr,
  Gfx12VglobalSaddr,
};

enum class ProbeInstrumentationTier {
  Unsupported,
  EntryCounterOnly,
  PreviousBbEdges,
};

struct ProbeTarget {
  uint32_t elf_mach;
  const char *gfxip;
  rj_code_arch_t arch;
  ProbeMemoryModel memory_model;
  ProbeInstrumentationTier tier;
};

// Keep this table constrained to gfxip machine IDs rocjitsu can decode today.
// TODO: Add newly generated rocjitsu targets here as their EF_AMDGPU_MACH
// constants and decoders land, rather than guessing unsupported gfxips.
inline constexpr ProbeTarget kProbeTargets[] = {
    {rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX90A, "gfx90a", ROCJITSU_CODE_ARCH_CDNA2,
     ProbeMemoryModel::Gfx9FlatGlobal, ProbeInstrumentationTier::EntryCounterOnly},
    {rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX940, "gfx940", ROCJITSU_CODE_ARCH_CDNA3,
     ProbeMemoryModel::Gfx9FlatGlobal, ProbeInstrumentationTier::EntryCounterOnly},
    {rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX941, "gfx941", ROCJITSU_CODE_ARCH_CDNA3,
     ProbeMemoryModel::Gfx9FlatGlobal, ProbeInstrumentationTier::EntryCounterOnly},
    {rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX942, "gfx942", ROCJITSU_CODE_ARCH_CDNA3,
     ProbeMemoryModel::Gfx9FlatGlobal, ProbeInstrumentationTier::EntryCounterOnly},
    {rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX950, "gfx950", ROCJITSU_CODE_ARCH_CDNA4,
     ProbeMemoryModel::Gfx9FlatGlobal, ProbeInstrumentationTier::EntryCounterOnly},
    {rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1010, "gfx1010", ROCJITSU_CODE_ARCH_RDNA1,
     ProbeMemoryModel::Gfx10FlatGlobal, ProbeInstrumentationTier::EntryCounterOnly},
    {rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1030, "gfx1030", ROCJITSU_CODE_ARCH_RDNA2,
     ProbeMemoryModel::Gfx10FlatGlobal, ProbeInstrumentationTier::EntryCounterOnly},
    {rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1100, "gfx1100", ROCJITSU_CODE_ARCH_RDNA3,
     ProbeMemoryModel::Gfx11GlobalSaddr, ProbeInstrumentationTier::PreviousBbEdges},
    {rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1150, "gfx1150", ROCJITSU_CODE_ARCH_RDNA3_5,
     ProbeMemoryModel::Gfx11GlobalSaddr, ProbeInstrumentationTier::PreviousBbEdges},
    {rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1200, "gfx1200", ROCJITSU_CODE_ARCH_RDNA4,
     ProbeMemoryModel::Gfx12VglobalSaddr, ProbeInstrumentationTier::PreviousBbEdges},
    {rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1201, "gfx1201", ROCJITSU_CODE_ARCH_RDNA4,
     ProbeMemoryModel::Gfx12VglobalSaddr, ProbeInstrumentationTier::PreviousBbEdges},
};

inline constexpr const ProbeTarget *probe_target_for_elf_mach(uint32_t elf_mach) {
  for (const ProbeTarget &target : kProbeTargets) {
    if (target.elf_mach == elf_mach)
      return &target;
  }
  return nullptr;
}

inline constexpr bool probe_target_supports_previous_bb_edges(const ProbeTarget &target) {
  return target.tier == ProbeInstrumentationTier::PreviousBbEdges;
}

inline constexpr bool probe_target_supports_entry_counter(const ProbeTarget &target) {
  return target.tier == ProbeInstrumentationTier::EntryCounterOnly ||
         target.tier == ProbeInstrumentationTier::PreviousBbEdges;
}

inline constexpr const char *probe_target_edge_support_reason(const ProbeTarget &target) {
  switch (target.tier) {
  case ProbeInstrumentationTier::Unsupported:
    return "unsupported-probe-target";
  case ProbeInstrumentationTier::PreviousBbEdges:
    return "previous-bb-edge-probes-supported";
  case ProbeInstrumentationTier::EntryCounterOnly:
    return "missing-previous-bb-probe-helpers";
  }
  return "unknown-probe-tier";
}

struct Rdna4ProbeRegisters {
  uint8_t state_sgpr = 100;
  uint8_t saved_exec_sgpr = 102;
  // The wave64 first-active-lane helper writes a 64-bit compare result to
  // tmp1_sgpr:tmp1_sgpr+1. Keep the defaults away from VCC special registers.
  uint8_t tmp0_sgpr = 104;
  uint8_t tmp1_sgpr = 104;
  uint8_t scc_sgpr = 108;
  uint8_t workitem_vgpr = 120;
  uint8_t tmp0_vgpr = 121;
  uint8_t tmp1_vgpr = 122;
  uint8_t tmp2_vgpr = 123;
};

struct ProbeRegisterRequirements {
  uint32_t sgprs = 0;
  uint32_t vgprs = 0;
  // Total per-lane private_segment_fixed_size required by instrumentation.
  // Future spill-backed probes should feed this from SpillManager::total_private_bytes().
  uint32_t private_segment_bytes = 0;
};

inline constexpr ProbeRegisterRequirements
probe_register_requirements(const Rdna4ProbeRegisters &regs) {
  ProbeRegisterRequirements requirements;
  requirements.sgprs = static_cast<uint32_t>(regs.state_sgpr) + 2;
  requirements.sgprs =
      std::max(requirements.sgprs, static_cast<uint32_t>(regs.saved_exec_sgpr) + 2);
  requirements.sgprs =
      std::max(requirements.sgprs, static_cast<uint32_t>(regs.tmp0_sgpr) + 1);
  requirements.sgprs =
      std::max(requirements.sgprs, static_cast<uint32_t>(regs.tmp1_sgpr) + 2);
  requirements.vgprs = static_cast<uint32_t>(regs.workitem_vgpr) + 1;
  requirements.vgprs =
      std::max(requirements.vgprs, static_cast<uint32_t>(regs.tmp0_vgpr) + 1);
  requirements.vgprs =
      std::max(requirements.vgprs, static_cast<uint32_t>(regs.tmp1_vgpr) + 1);
  requirements.vgprs =
      std::max(requirements.vgprs, static_cast<uint32_t>(regs.tmp2_vgpr) + 1);
  return requirements;
}

inline constexpr ProbeRegisterRequirements
flagless_counter_probe_register_requirements(const Rdna4ProbeRegisters &regs) {
  ProbeRegisterRequirements requirements;
  requirements.sgprs = static_cast<uint32_t>(regs.state_sgpr) + 2;
  requirements.vgprs = static_cast<uint32_t>(regs.workitem_vgpr) + 1;
  requirements.vgprs =
      std::max(requirements.vgprs, static_cast<uint32_t>(regs.tmp0_vgpr) + 1);
  return requirements;
}

inline constexpr ProbeRegisterRequirements
forced_lane0_counter_probe_register_requirements(const Rdna4ProbeRegisters &regs) {
  ProbeRegisterRequirements requirements =
      flagless_counter_probe_register_requirements(regs);
  requirements.sgprs =
      std::max(requirements.sgprs, static_cast<uint32_t>(regs.saved_exec_sgpr) + 2);
  return requirements;
}

inline constexpr ProbeRegisterRequirements
first_active_counter_probe_register_requirements(const Rdna4ProbeRegisters &regs) {
  ProbeRegisterRequirements requirements =
      flagless_counter_probe_register_requirements(regs);
  requirements.sgprs =
      std::max(requirements.sgprs, static_cast<uint32_t>(regs.saved_exec_sgpr) + 2);
  requirements.sgprs =
      std::max(requirements.sgprs, static_cast<uint32_t>(regs.tmp0_sgpr) + 1);
  requirements.sgprs =
      std::max(requirements.sgprs, static_cast<uint32_t>(regs.tmp1_sgpr) + 2);
  return requirements;
}

inline constexpr ProbeRegisterRequirements
previous_bb_probe_register_requirements(const Rdna4ProbeRegisters &regs) {
  ProbeRegisterRequirements requirements;
  requirements.sgprs = static_cast<uint32_t>(regs.state_sgpr) + 2;
  requirements.sgprs =
      std::max(requirements.sgprs, static_cast<uint32_t>(regs.saved_exec_sgpr) + 2);
  requirements.sgprs =
      std::max(requirements.sgprs, static_cast<uint32_t>(regs.tmp0_sgpr) + 1);
  requirements.sgprs =
      std::max(requirements.sgprs, static_cast<uint32_t>(regs.tmp1_sgpr) + 2);
  requirements.sgprs =
      std::max(requirements.sgprs, static_cast<uint32_t>(regs.scc_sgpr) + 1);
  requirements.vgprs = static_cast<uint32_t>(regs.workitem_vgpr) + 1;
  requirements.vgprs =
      std::max(requirements.vgprs, static_cast<uint32_t>(regs.tmp0_vgpr) + 1);
  requirements.vgprs =
      std::max(requirements.vgprs, static_cast<uint32_t>(regs.tmp1_vgpr) + 1);
  requirements.vgprs =
      std::max(requirements.vgprs, static_cast<uint32_t>(regs.tmp2_vgpr) + 1);
  return requirements;
}

enum class EdgeSlotPolicyKind {
  PreviousBbHash,
  FixedCounter,
};

inline const char *edge_slot_policy_name(EdgeSlotPolicyKind kind) {
  switch (kind) {
  case EdgeSlotPolicyKind::PreviousBbHash:
    return "previous-bb-hash";
  case EdgeSlotPolicyKind::FixedCounter:
    return "fixed-counter";
  }
  return "unknown";
}

struct InstrumentationPlanOptions {
  uint32_t block_entry_site_limit = 12;
  uint32_t branch_edge_site_limit = 8;
  bool branch_edge_site_limit_auto = false;
  uint32_t previous_bb_branch_site_limit = std::numeric_limits<uint32_t>::max();
  bool previous_bb_branch_site_limit_auto = false;
  EdgeSlotPolicyKind block_entry_slot_policy = EdgeSlotPolicyKind::PreviousBbHash;
  // Product branch coverage prefers previous-BB hashing. Fixed counters are a
  // per-site fallback or explicit debug override, not the planner default.
  EdgeSlotPolicyKind branch_terminator_slot_policy = EdgeSlotPolicyKind::PreviousBbHash;
  Rdna4ProbeRegisters probe_registers;
  bool fixed_edge_slots = false;
  bool branch_edge_slots = false;
  bool liveness_registers = true;
  bool require_liveness_registers = false;
  bool fixed_counter_fallback_for_branch_liveness = true;
  bool fixed_counter_fallback_for_branch_budget = false;
  uint32_t fixed_counter_branch_edge_fallback_limit = 0;
  bool allow_vgpr_scratch_spills = false;
  bool force_fresh_sgprs = false;
  bool force_fresh_vgprs = false;
  bool allow_opaque_fresh_registers = false;
  bool self_contained_edge_probes = false;
  bool verbose = false;
};

class FixedEdgeSlotTracker {
public:
  bool record(uint32_t slot) {
    for (uint32_t used : used_slots_) {
      if (used == slot)
        return true;
    }
    used_slots_.push_back(slot);
    return false;
  }

  uint32_t used() const {
    return static_cast<uint32_t>(
        std::min<size_t>(used_slots_.size(), std::numeric_limits<uint32_t>::max()));
  }

private:
  std::vector<uint32_t> used_slots_;
};

class FixedEdgeSlotAllocator {
public:
  struct StableReservation {
    uint32_t primary_slot = 0;
    uint32_t secondary_slot = 0;
    uint32_t slots_reserved = 0;
    uint32_t collisions = 0;
  };

  FixedEdgeSlotAllocator() = default;

  explicit FixedEdgeSlotAllocator(FixedEdgeSlotTracker *stable_slot_tracker) {
    if (stable_slot_tracker != nullptr)
      stable_slot_tracker_ = stable_slot_tracker;
  }

  std::optional<uint32_t> reserve(uint32_t count) {
    if (count == 0)
      return std::nullopt;
    if (next_slot_ > kMaxFixedCounterSlot || count - 1 > kMaxFixedCounterSlot - next_slot_)
      return std::nullopt;
    const uint32_t first = next_slot_;
    next_slot_ += count;
    return first;
  }

  uint32_t used() const { return next_slot_ - kFirstEdgeCounterSlot; }

  std::optional<StableReservation> reserve_stable(uint32_t primary_id,
                                                  uint32_t secondary_id = 0,
                                                  uint32_t count = 1) {
    if (count == 0 || count > 2)
      return std::nullopt;

    StableReservation reservation;
    reservation.primary_slot = fixed_counter_slot_for_id(primary_id);
    reservation.slots_reserved = count;
    reservation.collisions += record_stable_slot(reservation.primary_slot) ? 1 : 0;
    if (count == 2) {
      reservation.secondary_slot = fixed_counter_slot_for_id(secondary_id);
      if (reservation.secondary_slot == reservation.primary_slot)
        reservation.secondary_slot = next_stable_slot(reservation.secondary_slot);
      reservation.collisions += record_stable_slot(reservation.secondary_slot) ? 1 : 0;
    }
    return reservation;
  }

  uint32_t stable_used() const { return stable_slot_tracker_->used(); }

  static constexpr uint32_t fixed_slot_budget() {
    return kMaxFixedCounterSlot - kFirstEdgeCounterSlot + 1;
  }

  static constexpr uint32_t fixed_counter_slot_for_id(uint32_t id) {
    return kFirstEdgeCounterSlot + (id % fixed_slot_budget());
  }

private:
  static constexpr uint32_t next_stable_slot(uint32_t slot) {
    return slot == kMaxFixedCounterSlot ? kFirstEdgeCounterSlot : slot + 1;
  }

  bool record_stable_slot(uint32_t slot) {
    return stable_slot_tracker_->record(slot);
  }

  uint32_t next_slot_ = kFirstEdgeCounterSlot;
  FixedEdgeSlotTracker local_stable_slot_tracker_;
  FixedEdgeSlotTracker *stable_slot_tracker_ = &local_stable_slot_tracker_;
};

struct EdgeSlotAssignment {
  EdgeSlotPolicyKind policy = EdgeSlotPolicyKind::PreviousBbHash;
  uint32_t primary_slot = 0;
  uint32_t secondary_slot = 0;
  uint32_t fixed_slots_reserved = 0;
  uint32_t fixed_slot_collisions = 0;
  uint32_t inline_slots_reserved = 0;
};

struct EdgeSlotPolicySummary {
  uint32_t hashed_edge_sites = 0;
  uint32_t fixed_edge_sites = 0;
  uint32_t fixed_slot_requests = 0;
  uint32_t fixed_slots_reserved = 0;
  uint32_t fixed_slot_exhaustions = 0;
  uint32_t fixed_slot_collisions = 0;
  // Compatibility aliases for older report consumers. Fixed-counter probes can
  // now materialize literal offsets beyond the inline-immediate slot range.
  uint32_t inline_slot_requests = 0;
  uint32_t inline_slots_reserved = 0;
  uint32_t inline_slot_exhaustions = 0;
};

class EdgeSlotPolicy {
public:
  EdgeSlotPolicy(EdgeSlotPolicyKind block_entry_policy,
                 EdgeSlotPolicyKind branch_terminator_policy,
                 FixedEdgeSlotTracker *fixed_slot_tracker = nullptr)
      : block_entry_policy_(block_entry_policy),
        branch_terminator_policy_(branch_terminator_policy),
        fixed_slots_(fixed_slot_tracker) {}

  std::optional<EdgeSlotAssignment> assign_block_entry() {
    return assign(block_entry_policy_, /*edge_count=*/1);
  }

  std::optional<EdgeSlotAssignment> assign_block_entry(uint32_t stable_id) {
    return assign(block_entry_policy_, /*edge_count=*/1, stable_id);
  }

  std::optional<EdgeSlotAssignment> assign_branch_terminator(uint32_t edge_count) {
    return assign(branch_terminator_policy_, edge_count);
  }

  std::optional<EdgeSlotAssignment> assign_branch_terminator(uint32_t edge_count,
                                                            uint32_t primary_edge_id,
                                                            uint32_t secondary_edge_id = 0) {
    return assign(branch_terminator_policy_, edge_count, primary_edge_id,
                  secondary_edge_id);
  }

  std::optional<EdgeSlotAssignment>
  assign_branch_terminator_fallback(EdgeSlotPolicyKind policy, uint32_t edge_count) {
    return assign(policy, edge_count);
  }

  std::optional<EdgeSlotAssignment>
  assign_branch_terminator_fallback(EdgeSlotPolicyKind policy, uint32_t edge_count,
                                    uint32_t primary_edge_id,
                                    uint32_t secondary_edge_id = 0) {
    return assign(policy, edge_count, primary_edge_id, secondary_edge_id);
  }

  const EdgeSlotPolicySummary &summary() const { return summary_; }

private:
  std::optional<EdgeSlotAssignment> assign(EdgeSlotPolicyKind policy, uint32_t edge_count) {
    if (edge_count == 0)
      return std::nullopt;

    EdgeSlotAssignment assignment;
    assignment.policy = policy;
    if (policy == EdgeSlotPolicyKind::PreviousBbHash) {
      ++summary_.hashed_edge_sites;
      return assignment;
    }

    summary_.fixed_slot_requests += edge_count;
    summary_.inline_slot_requests += edge_count;
    std::optional<uint32_t> reserved = fixed_slots_.reserve(edge_count);
    if (!reserved) {
      ++summary_.fixed_slot_exhaustions;
      ++summary_.inline_slot_exhaustions;
      return std::nullopt;
    }

    ++summary_.fixed_edge_sites;
    summary_.fixed_slots_reserved += edge_count;
    summary_.inline_slots_reserved += edge_count;
    assignment.primary_slot = *reserved;
    assignment.secondary_slot = edge_count > 1 ? *reserved + 1 : 0;
    assignment.fixed_slots_reserved = edge_count;
    assignment.inline_slots_reserved = edge_count;
    return assignment;
  }

  std::optional<EdgeSlotAssignment> assign(EdgeSlotPolicyKind policy, uint32_t edge_count,
                                           uint32_t primary_edge_id,
                                           uint32_t secondary_edge_id = 0) {
    if (edge_count == 0)
      return std::nullopt;

    EdgeSlotAssignment assignment;
    assignment.policy = policy;
    if (policy == EdgeSlotPolicyKind::PreviousBbHash) {
      ++summary_.hashed_edge_sites;
      return assignment;
    }

    summary_.fixed_slot_requests += edge_count;
    summary_.inline_slot_requests += edge_count;
    std::optional<FixedEdgeSlotAllocator::StableReservation> reserved =
        fixed_slots_.reserve_stable(primary_edge_id, secondary_edge_id, edge_count);
    if (!reserved) {
      ++summary_.fixed_slot_exhaustions;
      ++summary_.inline_slot_exhaustions;
      return std::nullopt;
    }

    ++summary_.fixed_edge_sites;
    summary_.fixed_slots_reserved += edge_count;
    summary_.fixed_slot_collisions += reserved->collisions;
    summary_.inline_slots_reserved += edge_count;
    assignment.primary_slot = reserved->primary_slot;
    assignment.secondary_slot = reserved->secondary_slot;
    assignment.fixed_slots_reserved = edge_count;
    assignment.fixed_slot_collisions = reserved->collisions;
    assignment.inline_slots_reserved = edge_count;
    return assignment;
  }

  EdgeSlotPolicyKind block_entry_policy_;
  EdgeSlotPolicyKind branch_terminator_policy_;
  FixedEdgeSlotAllocator fixed_slots_;
  EdgeSlotPolicySummary summary_;
};

struct KernelSite {
  std::string name;
  uint64_t descriptor_file_offset = 0;
  uint64_t entry_text_offset = 0;
  uint32_t elf_mach = 0;
  uint32_t descriptor_sgpr_count = 0;
  uint32_t metadata_sgpr_count = 0;
  uint32_t allocated_sgpr_count = 0;
  uint32_t allocated_vgpr_count = 0;
  uint32_t private_segment_fixed_size = 0;
  bool has_metadata_sgpr_count = false;
  bool descriptor_sgpr_count_effective = true;
  bool fresh_sgpr_growth_supported = true;
  bool wave32 = false;
};

enum class EdgePatchKind {
  BlockEntry,
  ConditionalBlockEntry,
  BranchTerminator,
  ConditionalBranchTerminator,
};

inline const char *edge_patch_kind_name(EdgePatchKind kind) {
  switch (kind) {
  case EdgePatchKind::BlockEntry:
    return "block";
  case EdgePatchKind::ConditionalBlockEntry:
    return "cond-block";
  case EdgePatchKind::BranchTerminator:
    return "branch";
  case EdgePatchKind::ConditionalBranchTerminator:
    return "cond-branch";
  }
  return "unknown";
}

inline constexpr bool edge_patch_kind_is_block_entry(EdgePatchKind kind) {
  return kind == EdgePatchKind::BlockEntry || kind == EdgePatchKind::ConditionalBlockEntry;
}

inline constexpr bool edge_patch_kind_is_conditional_dispatch(EdgePatchKind kind) {
  return kind == EdgePatchKind::ConditionalBlockEntry ||
         kind == EdgePatchKind::ConditionalBranchTerminator;
}

struct ProbeScratchSpillSlot {
  uint8_t vgpr = 0;
  uint32_t byte_offset = 0;
};

struct ProbeScratchSgprSpillSlot {
  uint8_t sgpr = 0;
  uint32_t byte_offset = 0;
};

struct ProbeScratchSpillPlan {
  uint8_t address_vgpr = 0;
  uint32_t private_segment_bytes = 0;
  bool wave32 = false;
  std::vector<ProbeScratchSpillSlot> vgpr_spills;
  std::vector<ProbeScratchSgprSpillSlot> sgpr_spills;
};

struct EdgeSite {
  EdgePatchKind kind = EdgePatchKind::BlockEntry;
  std::string kernel_name;
  uint64_t pred_text_offset = 0;
  uint64_t block_text_offset = 0;
  uint64_t patch_text_offset = 0;
  uint64_t return_text_offset = 0;
  uint32_t first_inst_size = 0;
  uint32_t predecessor_count = 0;
  uint32_t bb_id = 0;
  uint32_t fallthrough_bb_id = 0;
  EdgeSlotPolicyKind slot_policy = EdgeSlotPolicyKind::PreviousBbHash;
  Rdna4ProbeRegisters probe_registers;
  uint32_t fixed_slot = 0;
  uint32_t fallthrough_slot = 0;
  uint32_t fixed_slot_collisions = 0;
  uint16_t branch_opcode = 0;
  // Trampoline emission must use this per-site decision so a hybrid plan can
  // mix entry-backed and self-contained probes in the same code object.
  bool self_contained_probe = false;
  // Used only for fixed-counter fallback edges whose branch outcome has
  // EXEC==0. The planner sets this after selecting fresh VGPR temporaries and a
  // distinct saved-EXEC SGPR pair, so inactive-lane VGPR state is not clobbered.
  bool force_lane0_exec_for_fixed_counter = false;
  std::optional<ProbeScratchSpillPlan> scratch_spill_plan;
};

inline constexpr ProbeRegisterRequirements
edge_site_probe_register_requirements(const EdgeSite &site) {
  ProbeRegisterRequirements requirements;
  if (site.slot_policy == EdgeSlotPolicyKind::PreviousBbHash)
    requirements = previous_bb_probe_register_requirements(site.probe_registers);
  else {
    const bool flagless_counter =
        site.self_contained_probe || !edge_patch_kind_is_block_entry(site.kind);
    requirements =
        flagless_counter
            ? (site.force_lane0_exec_for_fixed_counter
                   ? forced_lane0_counter_probe_register_requirements(site.probe_registers)
                   : flagless_counter_probe_register_requirements(site.probe_registers))
            : first_active_counter_probe_register_requirements(site.probe_registers);
  }
  if (site.scratch_spill_plan) {
    requirements.vgprs =
        std::max(requirements.vgprs,
                 static_cast<uint32_t>(site.scratch_spill_plan->address_vgpr) + 1);
    requirements.private_segment_bytes =
        std::max(requirements.private_segment_bytes,
                 site.scratch_spill_plan->private_segment_bytes);
  }
  return requirements;
}

struct EdgeSiteSkipSample {
  std::string kind;
  uint64_t text_offset = 0;
  std::string reason;
  std::string mnemonic;
  uint32_t instruction_size = 0;
  uint64_t instruction_flags = 0;
  std::vector<uint32_t> words;
};

struct EdgeSiteSkipReasonCount {
  std::string kind;
  std::string reason;
  uint32_t count = 0;
};

struct OpaqueInstructionSample {
  std::string mnemonic;
  uint64_t text_offset = 0;
  std::vector<uint32_t> words;
  bool liveness_modeled = false;
};

struct OpaqueFreshRegisterCandidateSample {
  std::string kind;
  uint64_t patch_text_offset = 0;
  std::string mnemonic;
  std::vector<uint32_t> words;
  uint32_t required_sgprs = 0;
  uint32_t required_vgprs = 0;
  uint32_t allocated_sgprs = 0;
  uint32_t allocated_vgprs = 0;
  bool sgpr_growth = false;
  bool vgpr_growth = false;
  bool previous_bb_probe_registers = false;
  bool stable_state_sgpr = false;
  std::string slot_policy;
  uint32_t state_sgpr = 0;
  uint32_t saved_exec_sgpr = 0;
  uint32_t tmp0_sgpr = 0;
  uint32_t tmp1_sgpr = 0;
  uint32_t scc_sgpr = 0;
  uint32_t workitem_vgpr = 0;
  uint32_t tmp0_vgpr = 0;
  uint32_t tmp1_vgpr = 0;
  uint32_t tmp2_vgpr = 0;
};

struct KernelEdgeSelectionSummary {
  std::string kernel_name;
  std::string coverage_strategy;
  uint32_t reachable_blocks = 0;
  uint32_t block_candidates = 0;
  uint32_t block_selected = 0;
  uint32_t inline_slots_reserved = 0;
  uint32_t skipped_unsafe = 0;
  uint32_t skipped_liveness = 0;
  uint32_t skipped_limit = 0;
  uint32_t skipped_fixed_slot = 0;
  uint32_t branch_candidates = 0;
  uint32_t branch_edge_candidate_edges = 0;
  uint32_t previous_bb_branch_edge_candidate_edges = 0;
  uint32_t previous_bb_branch_site_candidate_sites = 0;
  uint32_t branch_edge_budget = 0;
  uint32_t previous_bb_branch_site_budget = 0;
  uint32_t previous_bb_branch_edge_over_budget = 0;
  uint32_t previous_bb_branch_site_over_budget = 0;
  uint32_t fixed_counter_branch_edge_fallback_budget = 0;
  // Per-cause fixed-counter fallback accounting. The aggregate fallback budget
  // is still shared by the product planner, but reports should distinguish
  // safety degradation from aggregate-cap and placement degradation so the next
  // DBI proof target is visible.
  uint32_t fixed_counter_branch_edge_aggregate_fallback_used = 0;
  uint32_t fixed_counter_branch_edge_safety_fallback_used = 0;
  uint32_t fixed_counter_branch_edge_liveness_fallback_used = 0;
  uint32_t fixed_counter_branch_edge_placement_fallback_used = 0;
  uint32_t fixed_counter_branch_edge_fallback_used = 0;
  uint32_t exec_empty_fixed_counter_edges = 0;
  uint32_t branch_edges_selected = 0;
  uint32_t previous_bb_branch_edges_selected = 0;
  uint32_t branch_edges_degraded_to_fixed = 0;
  uint32_t previous_bb_branch_sites_selected = 0;
  uint32_t previous_bb_branch_sites_degraded_to_fixed = 0;
  uint32_t edge_trampolines_planned = 0;
  uint32_t previous_bb_branch_edge_trampolines_planned = 0;
  uint32_t planned_appended_edge_trampolines = 0;
  uint32_t planned_local_edge_trampolines = 0;
  uint32_t previous_bb_branch_planned_appended_edge_trampolines = 0;
  uint32_t previous_bb_branch_planned_local_edge_trampolines = 0;
  uint64_t planned_edge_trampoline_bytes = 0;
  uint64_t previous_bb_branch_edge_trampoline_bytes = 0;
  uint64_t planned_appended_edge_trampoline_bytes = 0;
  uint64_t planned_local_edge_trampoline_bytes = 0;
  uint64_t previous_bb_branch_planned_appended_edge_trampoline_bytes = 0;
  uint64_t previous_bb_branch_planned_local_edge_trampoline_bytes = 0;
  uint64_t largest_edge_trampoline_bytes = 0;
  uint64_t largest_previous_bb_branch_edge_trampoline_bytes = 0;
  uint32_t previous_bb_branch_afl_map_budget = kHashEdgeSlots;
  uint32_t previous_bb_branch_afl_map_pressure_ppm = 0;
  uint32_t previous_bb_branch_trampoline_avg_bytes_x100 = 0;
  uint32_t previous_bb_branch_appended_trampoline_ratio_ppm = 0;
  uint32_t previous_bb_branch_local_trampoline_ratio_ppm = 0;
  uint32_t skipped_branch_unsafe = 0;
  uint32_t skipped_branch_liveness = 0;
  uint32_t skipped_branch_limit = 0;
  uint32_t opaque_instruction_count = 0;
  uint32_t unmodeled_opaque_instruction_count = 0;
  uint32_t liveness_probe_points = 0;
  uint32_t fresh_register_probe_points = 0;
  uint32_t opaque_fresh_register_candidate_probe_points = 0;
  uint32_t opaque_fresh_register_candidate_sgpr_growth_probe_points = 0;
  uint32_t opaque_fresh_register_candidate_vgpr_growth_probe_points = 0;
  uint32_t opaque_fresh_register_candidate_required_sgprs = 0;
  uint32_t opaque_fresh_register_candidate_required_vgprs = 0;
  uint32_t scratch_spill_probe_points = 0;
  uint32_t vgpr_scratch_spill_probe_points = 0;
  uint32_t sgpr_scratch_spill_probe_points = 0;
  uint32_t fresh_register_growth_disabled_by_opaque_probe_points = 0;
  uint32_t sgpr_scratch_spill_disabled_by_opaque_probe_points = 0;
  uint32_t sgpr_scratch_spill_disabled_by_exec_condition_probe_points = 0;
  uint32_t direct_exec_fixed_scratch_disabled_by_opaque_probe_points = 0;
  bool liveness_registers = false;
  bool fresh_registers = false;
  bool self_contained_probe = false;
  std::string branch_edge_budget_reason;
  std::string previous_bb_branch_site_budget_reason;
  std::string fixed_counter_branch_edge_fallback_budget_reason;
  std::string previous_bb_branch_aggregate_limit_kind;
  std::string previous_bb_branch_aggregate_safety;
  std::string previous_bb_branch_aggregate_safety_reason;
  std::string branch_edge_slot_policy_reason;
  std::string previous_bb_branch_overhead_status;
  std::string previous_bb_branch_overhead_reason;
  EdgeSlotPolicySummary slot_policy_summary;
  std::vector<EdgeSiteSkipReasonCount> degradation_reason_counts;
  std::vector<EdgeSiteSkipReasonCount> skip_reason_counts;
  std::vector<EdgeSiteSkipSample> sampled_skips;
  std::vector<OpaqueInstructionSample> sampled_opaque_instructions;
  std::vector<OpaqueFreshRegisterCandidateSample>
      sampled_opaque_fresh_register_candidates;
};

struct InstrumentationPlan {
  std::vector<EdgeSite> sites;
  std::vector<KernelEdgeSelectionSummary> kernel_summaries;
  EdgeSlotPolicySummary slot_policy_summary;
  std::string failure_reason;
};

struct EdgePatchFailure {
  std::string kernel_name;
  std::string kind;
  uint64_t patch_text_offset = 0;
  uint64_t return_text_offset = 0;
  std::string reason;
};

struct EdgeSiteSelectionSample {
  std::string kernel_name;
  std::string kind;
  uint64_t pred_text_offset = 0;
  uint64_t block_text_offset = 0;
  uint64_t patch_text_offset = 0;
  uint64_t return_text_offset = 0;
  uint64_t cave_text_offset = 0;
  uint64_t trampoline_bytes = 0;
  uint32_t bb_id = 0;
  uint32_t fallthrough_bb_id = 0;
  uint32_t fixed_slot = 0;
  uint32_t fallthrough_slot = 0;
  uint32_t fixed_slot_collisions = 0;
  bool self_contained_probe = false;
  bool force_lane0_exec_for_fixed_counter = false;
  bool scratch_spill = false;
  bool vgpr_scratch_spill = false;
  bool sgpr_scratch_spill = false;
  uint32_t state_sgpr = 0;
  uint32_t saved_exec_sgpr = 0;
  uint32_t tmp0_sgpr = 0;
  uint32_t tmp1_sgpr = 0;
  uint32_t scc_sgpr = 0;
  uint32_t workitem_vgpr = 0;
  uint32_t tmp0_vgpr = 0;
  uint32_t tmp1_vgpr = 0;
  uint32_t tmp2_vgpr = 0;
  uint32_t scratch_address_vgpr = 0;
  std::vector<uint8_t> scratch_spilled_vgprs;
  std::vector<uint8_t> scratch_spilled_sgprs;
  std::string slot_policy;
  std::string scratch_address_exec_source;
  std::string placement;
};

struct LocalTextCaveSummary {
  uint32_t range_count = 0;
  uint64_t total_bytes = 0;
  uint64_t largest_range_bytes = 0;
};

struct KernelDescriptorResourceSummary {
  std::string kernel_name;
  uint64_t descriptor_file_offset = 0;
  bool wave32 = false;
  bool has_metadata_sgpr_count = false;
  bool descriptor_sgpr_count_effective = true;
  bool fresh_sgpr_growth_supported = true;
  bool old_private_segment_enabled = false;
  bool patched_private_segment_enabled = false;
  uint32_t vgpr_granularity = 0;
  uint32_t sgpr_granularity = 8;
  uint32_t old_vgpr_granulated = 0;
  uint32_t old_sgpr_granulated = 0;
  uint32_t patched_vgpr_granulated = 0;
  uint32_t patched_sgpr_granulated = 0;
  uint32_t descriptor_sgpr_count = 0;
  uint32_t metadata_sgpr_count = 0;
  uint32_t old_vgpr_count = 0;
  uint32_t old_sgpr_count = 0;
  uint32_t patched_vgpr_count = 0;
  uint32_t patched_sgpr_count = 0;
  uint32_t old_private_segment_fixed_size = 0;
  uint32_t patched_private_segment_fixed_size = 0;
  uint32_t spill_bytes = 0;
  std::string sgpr_count_metadata_patch = "none";
  std::string private_segment_metadata_patch = "none";
  bool resource_fields_changed = false;
};

struct KernelPatchabilitySkipSummary {
  std::string kernel_name;
  std::string reason;
  bool entry_probe_safe = false;
  bool self_contained_probe_safe = false;
  bool branch_probe_safe = false;
  bool prefers_self_contained_edge_probes = false;
  bool prefers_fixed_branch_counters = false;
};

class LocalTextCaveAllocator {
public:
  explicit LocalTextCaveAllocator(std::span<const uint8_t> text) {
    uint64_t pos = 0;
    while (pos < text.size()) {
      while (pos < text.size() && text[static_cast<size_t>(pos)] != 0)
        ++pos;
      const uint64_t run_start = pos;
      while (pos < text.size() && text[static_cast<size_t>(pos)] == 0)
        ++pos;
      const uint64_t run_end = pos;

      const uint64_t aligned_start = align_up(run_start, sizeof(uint32_t));
      const uint64_t aligned_end = align_down(run_end, sizeof(uint32_t));
      if (aligned_end > aligned_start && aligned_end - aligned_start >= kMinCaveBytes)
        ranges_.push_back({aligned_start, aligned_end});
    }
  }

  template <typename CanUse>
  std::optional<uint64_t> allocate(uint64_t ideal_offset, uint64_t size_bytes, CanUse can_use) {
    if (size_bytes == 0)
      return std::nullopt;
    size_bytes = align_up(size_bytes, sizeof(uint32_t));

    size_t best_index = ranges_.size();
    uint64_t best_start = 0;
    uint64_t best_distance = std::numeric_limits<uint64_t>::max();

    for (size_t i = 0; i < ranges_.size(); ++i) {
      const Range &range = ranges_[i];
      if (range.end <= range.start || range.end - range.start < size_bytes)
        continue;

      const uint64_t last = range.end - size_bytes;
      for (uint64_t candidate = range.start; candidate <= last;
           candidate += sizeof(uint32_t)) {
        if (!can_use(candidate))
          continue;
        const uint64_t distance = candidate > ideal_offset ? candidate - ideal_offset
                                                           : ideal_offset - candidate;
        if (distance < best_distance ||
            (distance == best_distance && candidate < best_start)) {
          best_distance = distance;
          best_index = i;
          best_start = candidate;
        }
      }
    }

    if (best_index == ranges_.size())
      return std::nullopt;

    consume(best_index, best_start, size_bytes);
    return best_start;
  }

  LocalTextCaveSummary summary() const {
    LocalTextCaveSummary out;
    out.range_count = static_cast<uint32_t>(
        std::min<size_t>(ranges_.size(), std::numeric_limits<uint32_t>::max()));
    for (const Range &range : ranges_) {
      const uint64_t size = range.end - range.start;
      out.total_bytes += size;
      out.largest_range_bytes = std::max(out.largest_range_bytes, size);
    }
    return out;
  }

private:
  struct Range {
    uint64_t start = 0;
    uint64_t end = 0;
  };

  static constexpr uint64_t kMinCaveBytes = 32;

  static uint64_t align_up(uint64_t value, uint64_t alignment) {
    const uint64_t remainder = value % alignment;
    return remainder == 0 ? value : value + alignment - remainder;
  }

  static uint64_t align_down(uint64_t value, uint64_t alignment) {
    return value - (value % alignment);
  }

  void consume(size_t index, uint64_t start, uint64_t size_bytes) {
    const Range old = ranges_[index];
    const uint64_t end = start + size_bytes;
    ranges_.erase(ranges_.begin() + static_cast<std::ptrdiff_t>(index));
    if (end < old.end && old.end - end >= kMinCaveBytes)
      ranges_.insert(ranges_.begin() + static_cast<std::ptrdiff_t>(index), {end, old.end});
    if (start > old.start && start - old.start >= kMinCaveBytes)
      ranges_.insert(ranges_.begin() + static_cast<std::ptrdiff_t>(index), {old.start, start});
  }

  std::vector<Range> ranges_;
};

struct PatchDeviceElfReport {
  std::string context;
  std::string device_image_id;
  uint32_t device_image_index = kUnknownDeviceImageIndex;
  std::string kernel_filter;
  std::string gfxip;
  std::string arch;
  std::string reason;
  std::string failure_phase;
  std::string edge_instrumentation_reason;
  std::string coverage_strategy;
  std::string coverage_strategy_reason;
  std::string cfg_failure_reason;
  size_t input_bytes = 0;
  size_t output_bytes = 0;
  uint32_t kernel_descriptor_count = 0;
  uint32_t entry_candidate_count = 0;
  uint32_t descriptor_updates = 0;
  uint32_t entry_patched = 0;
  uint32_t edge_sites_selected = 0;
  uint32_t edge_sites_patched = 0;
  uint32_t edge_patch_failures = 0;
  uint32_t local_text_caves = 0;
  uint32_t local_text_cave_ranges = 0;
  uint32_t appended_caves = 0;
  uint32_t branch_range_failures = 0;
  uint32_t hashed_edge_sites = 0;
  uint32_t fixed_edge_sites = 0;
  uint32_t fixed_slot_requests = 0;
  uint32_t fixed_slots_reserved = 0;
  uint32_t fixed_slot_exhaustions = 0;
  uint32_t fixed_slot_collisions = 0;
  uint32_t inline_slot_requests = 0;
  uint32_t inline_slot_exhaustions = 0;
  uint32_t branch_edges_degraded_to_fixed = 0;
  uint32_t fixed_counter_branch_edge_aggregate_fallback_used = 0;
  uint32_t fixed_counter_branch_edge_safety_fallback_used = 0;
  uint32_t fixed_counter_branch_edge_liveness_fallback_used = 0;
  uint32_t fixed_counter_branch_edge_placement_fallback_used = 0;
  uint32_t exec_empty_fixed_counter_edges = 0;
  uint32_t previous_bb_branch_edges_selected = 0;
  uint32_t previous_bb_branch_sites_selected = 0;
  uint32_t previous_bb_branch_sites_degraded_to_fixed = 0;
  uint32_t edge_trampolines_planned = 0;
  uint32_t previous_bb_branch_edge_trampolines_planned = 0;
  uint32_t planned_appended_edge_trampolines = 0;
  uint32_t planned_local_edge_trampolines = 0;
  uint32_t previous_bb_branch_planned_appended_edge_trampolines = 0;
  uint32_t previous_bb_branch_planned_local_edge_trampolines = 0;
  uint64_t planned_edge_trampoline_bytes = 0;
  uint64_t previous_bb_branch_edge_trampoline_bytes = 0;
  uint64_t planned_appended_edge_trampoline_bytes = 0;
  uint64_t planned_local_edge_trampoline_bytes = 0;
  uint64_t previous_bb_branch_planned_appended_edge_trampoline_bytes = 0;
  uint64_t previous_bb_branch_planned_local_edge_trampoline_bytes = 0;
  uint64_t largest_edge_trampoline_bytes = 0;
  uint64_t largest_previous_bb_branch_edge_trampoline_bytes = 0;
  uint32_t previous_bb_branch_afl_map_budget = kHashEdgeSlots;
  uint32_t previous_bb_branch_afl_map_pressure_ppm = 0;
  uint32_t previous_bb_branch_trampoline_avg_bytes_x100 = 0;
  uint32_t previous_bb_branch_appended_trampoline_ratio_ppm = 0;
  uint32_t previous_bb_branch_local_trampoline_ratio_ppm = 0;
  uint32_t previous_bb_branch_code_growth_pressure_ppm = 0;
  uint32_t probe_required_sgprs = 0;
  uint32_t probe_required_vgprs = 0;
  uint32_t probe_required_private_segment_bytes = 0;
  uint32_t spill_bytes = 0;
  uint32_t entry_liveness_register_kernels = 0;
  uint32_t entry_liveness_probe_points = 0;
  uint32_t entry_backed_edge_kernels = 0;
  uint32_t self_contained_edge_kernels = 0;
  uint32_t edge_site_limit = 0;
  uint32_t branch_edge_site_limit = 0;
  uint32_t previous_bb_branch_site_limit = 0;
  uint64_t local_text_cave_bytes = 0;
  uint64_t largest_local_text_cave_bytes = 0;
  uint64_t appended_cave_bytes = 0;
  uintptr_t runtime_shadow_registration = 0;
  bool success = false;
  bool supports_previous_bb_edges = false;
  bool edge_instrumentation_enabled = false;
  bool skip_entry_probe = false;
  bool fixed_edge_slots = false;
  bool branch_edge_slots = false;
  bool hybrid_edge_probes = false;
  bool require_liveness_registers = false;
  bool allow_opaque_fresh_registers = false;
  bool vgpr_scratch_spills_requested = false;
  bool vgpr_scratch_spills_enabled = false;
  bool branch_edge_site_limit_auto = false;
  bool previous_bb_branch_site_limit_auto = false;
  EdgeSlotPolicyKind branch_edge_slot_policy = EdgeSlotPolicyKind::PreviousBbHash;
  std::string vgpr_scratch_spill_reason;
  std::string previous_bb_branch_overhead_status;
  std::string previous_bb_branch_overhead_reason;
  std::string descriptor_resource_failure_reason;
  std::vector<KernelEdgeSelectionSummary> kernel_summaries;
  std::vector<KernelDescriptorResourceSummary> descriptor_resources;
  std::vector<KernelPatchabilitySkipSummary> skipped_kernels;
  std::vector<EdgePatchFailure> sampled_failures;
  std::vector<EdgeSiteSelectionSample> sampled_selected_edges;
};

inline constexpr uint16_t amdgpu_positive_inline_const(uint32_t value) {
  return static_cast<uint16_t>(128u + value);
}

inline constexpr uint16_t amdgpu_negative_inline_const(uint32_t absolute_value) {
  return static_cast<uint16_t>(192u + absolute_value);
}

inline constexpr uint32_t build_rdna4_v_mov_b32(uint8_t vdst, uint16_t src0) {
  return (0x3Fu << 25) | (static_cast<uint32_t>(vdst) << 17) | (1u << 9) | (src0 & 0x1FFu);
}

inline constexpr uint16_t amdgpu_vgpr_src(uint8_t vgpr) {
  return static_cast<uint16_t>(256u + vgpr);
}

inline constexpr uint8_t kScratchNoScalarOffset = 0x7f;

inline constexpr bool is_power_of_two_u32(uint32_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

inline constexpr uint32_t log2_power_of_two_u32(uint32_t value) {
  uint32_t log = 0;
  while (value > 1) {
    value >>= 1;
    ++log;
  }
  return log;
}

inline constexpr uint32_t build_rdna4_vop2(uint8_t op, uint8_t vdst, uint16_t src0,
                                           uint8_t vsrc1) {
  return (static_cast<uint32_t>(op) << 25) | (static_cast<uint32_t>(vdst) << 17) |
         (static_cast<uint32_t>(vsrc1) << 9) | (src0 & 0x1FFu);
}

inline constexpr uint32_t build_rdna4_v_lshlrev_b32(uint8_t vdst, uint16_t src0,
                                                    uint8_t vsrc1) {
  return build_rdna4_vop2(/*op=*/0x18, vdst, src0, vsrc1);
}

inline constexpr uint32_t build_rdna4_v_add_nc_u32(uint8_t vdst, uint16_t src0,
                                                   uint8_t vsrc1) {
  return build_rdna4_vop2(/*op=*/0x25, vdst, src0, vsrc1);
}

inline constexpr uint32_t build_rdna4_v_xor_b32(uint8_t vdst, uint16_t src0, uint8_t vsrc1) {
  return build_rdna4_vop2(/*op=*/0x1d, vdst, src0, vsrc1);
}

inline constexpr uint32_t build_rdna4_v_and_b32(uint8_t vdst, uint16_t src0, uint8_t vsrc1) {
  return build_rdna4_vop2(/*op=*/0x1b, vdst, src0, vsrc1);
}

inline constexpr uint32_t build_rdna4_v_mbcnt_lo_u32_b32_word0(uint8_t vdst) {
  return 0xD71F0000u | vdst;
}

inline constexpr uint32_t build_rdna4_v_mbcnt_hi_u32_b32_word0(uint8_t vdst) {
  return 0xD7200000u | vdst;
}

inline constexpr uint32_t
build_rdna4_v_mbcnt_u32_b32_word1(uint16_t src0,
                                   uint16_t src1 = amdgpu_positive_inline_const(0)) {
  return (src0 & 0x1FFu) | ((src1 & 0x1FFu) << 9);
}

inline constexpr uint32_t
build_rdna4_v_mbcnt_lo_u32_b32_word1(uint16_t src0,
                                      uint16_t src1 = amdgpu_positive_inline_const(0)) {
  return build_rdna4_v_mbcnt_u32_b32_word1(src0, src1);
}

inline constexpr uint32_t
build_rdna4_v_mbcnt_hi_u32_b32_word1(uint16_t src0, uint16_t src1) {
  return build_rdna4_v_mbcnt_u32_b32_word1(src0, src1);
}

inline constexpr uint32_t build_rdna4_v_cmp_eq_u32_vcc(uint16_t src0, uint8_t vsrc1) {
  return 0x7C940000u | (static_cast<uint32_t>(vsrc1) << 9) | (src0 & 0x1FFu);
}

inline constexpr uint32_t build_rdna4_v_cmp_eq_u32_sdst_word0(uint8_t sdst) {
  return 0xD44A0000u | sdst;
}

inline constexpr uint32_t build_rdna4_v_cmp_eq_u32_sdst_word1(uint16_t src0, uint16_t src1) {
  return (src0 & 0x1FFu) | ((src1 & 0x1FFu) << 9);
}

inline constexpr uint32_t build_rdna4_v_cmpx_eq_u32_exec(uint16_t src0, uint8_t vsrc1) {
  return 0x7D940000u | (static_cast<uint32_t>(vsrc1) << 9) | (src0 & 0x1FFu);
}

inline constexpr uint32_t build_rdna4_v_readfirstlane_b32(uint8_t sdst, uint8_t vsrc) {
  return (0x3Fu << 25) | (static_cast<uint32_t>(sdst) << 17) | (2u << 9) |
         amdgpu_vgpr_src(vsrc);
}

inline constexpr uint32_t build_rdna4_s_load_b64_s0_word0(uint8_t sdst_lo) {
  return 0xF4002000u | (static_cast<uint32_t>(sdst_lo) << 6);
}

inline constexpr uint32_t build_rdna4_s_load_b64_s0_word1(uint32_t byte_offset) {
  return 0xF8000000u | byte_offset;
}

inline constexpr uint32_t build_rdna4_s_load_b64_s100_s0_word1(uint32_t byte_offset) {
  return build_rdna4_s_load_b64_s0_word1(byte_offset);
}

inline constexpr uint32_t build_rdna4_sopp(uint8_t op, uint16_t imm) {
  return rocjitsu::pack_sopp(op, imm);
}

inline constexpr uint32_t build_rdna4_sop1(uint8_t op, uint8_t sdst, uint16_t ssrc0) {
  return rocjitsu::pack_sop1(op, sdst, ssrc0);
}

inline constexpr uint32_t build_rdna4_sop2(uint8_t op, uint8_t sdst, uint16_t ssrc0,
                                           uint16_t ssrc1) {
  return rocjitsu::pack_sop2(op, sdst, ssrc0, ssrc1);
}

inline constexpr uint32_t build_rdna4_sopc(uint8_t op, uint16_t ssrc0, uint16_t ssrc1) {
  return (0x17Eu << 23) | (static_cast<uint32_t>(op) << 16) |
         ((ssrc1 & 0xFFu) << 8) | (ssrc0 & 0xFFu);
}

inline constexpr uint8_t sop1_op_s_mov_b32(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_RDNA1:
  case ROCJITSU_CODE_ARCH_RDNA2:
    return 3;
  default:
    return 0;
  }
}

inline constexpr uint32_t build_s_mov_b32_word(uint8_t sdst, uint16_t src, rj_code_arch_t arch) {
  return build_rdna4_sop1(sop1_op_s_mov_b32(arch), sdst, src);
}

inline constexpr uint32_t
build_rdna4_s_mov_b32_literal_word(uint8_t sdst, rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA4) {
  return build_s_mov_b32_word(sdst, kScalarLiteral, arch);
}

inline void append_rdna4_s_mov_b64_literal(std::vector<uint32_t> &words, uint8_t sdst_lo,
                                           uint64_t value,
                                           rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA4) {
  words.push_back(build_rdna4_s_mov_b32_literal_word(sdst_lo, arch));
  words.push_back(static_cast<uint32_t>(value));
  words.push_back(build_rdna4_s_mov_b32_literal_word(sdst_lo + 1, arch));
  words.push_back(static_cast<uint32_t>(value >> 32));
}

class Rdna4ProbeBuilder {
public:
  explicit Rdna4ProbeBuilder(Rdna4ProbeRegisters regs = {},
                             rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA4)
      : regs_(regs), arch_(arch) {}

  const Rdna4ProbeRegisters &regs() const { return regs_; }

  std::vector<uint32_t> take() { return std::move(words_); }

  void word(uint32_t value) { words_.push_back(value); }

  void s_load_b64_state_from_kernarg(uint32_t byte_offset) {
    words_.push_back(build_rdna4_s_load_b64_s0_word0(regs_.state_sgpr));
    words_.push_back(build_rdna4_s_load_b64_s0_word1(byte_offset));
  }

  void s_mov_b64_literal(uint8_t sdst_lo, uint64_t value) {
    append_rdna4_s_mov_b64_literal(words_, sdst_lo, value, arch_);
  }

  void s_mov_b32(uint8_t sdst, uint16_t src) {
    words_.push_back(build_s_mov_b32_word(sdst, src, arch_));
  }

  void s_bcnt1_i32_b32(uint8_t sdst, uint16_t src) {
    words_.push_back(build_rdna4_sop1(/*op=*/0x18, sdst, src));
  }

  void s_bcnt1_i32_b64(uint8_t sdst, uint16_t src) {
    words_.push_back(build_rdna4_sop1(/*op=*/0x19, sdst, src));
  }

  void s_and_b32(uint8_t sdst, uint16_t src0, uint16_t src1) {
    words_.push_back(build_rdna4_sop2(/*op=*/22, sdst, src0, src1));
  }

  void s_or_b32(uint8_t sdst, uint16_t src0, uint16_t src1) {
    words_.push_back(build_rdna4_sop2(/*op=*/24, sdst, src0, src1));
  }

  void s_cselect_b32(uint8_t sdst, uint16_t src0, uint16_t src1) {
    words_.push_back(build_rdna4_sop2(/*op=*/0x30, sdst, src0, src1));
  }

  void s_cmp_lg_u32(uint16_t src0, uint16_t src1) {
    words_.push_back(build_rdna4_sopc(/*op=*/7, src0, src1));
  }

  void s_delay_alu(uint16_t imm) { words_.push_back(build_rdna4_sopp(/*op=*/7, imm)); }

  void s_wait_alu(uint16_t imm) { words_.push_back(build_rdna4_sopp(/*op=*/8, imm)); }

  void valu_dep_1_barrier() {
    if (supports_s_delay_alu()) {
      s_delay_alu(/*VALU_DEP_1=*/1);
      return;
    }
    s_nop();
  }

  void valu_dep_2_barrier() {
    if (supports_s_delay_alu()) {
      s_delay_alu(/*VALU_DEP_2=*/2);
      return;
    }
    s_nop();
  }

  void s_wait_kmcnt(uint16_t imm = 0) {
    if (uses_gfx9_or_gfx10_waitcnt()) {
      words_.push_back(build_rdna4_sopp(/*s_waitcnt=*/12, /*all counters complete=*/0));
      return;
    }
    if (uses_gfx11_global_encoding()) {
      words_.push_back(build_rdna4_sopp(/*s_waitcnt=*/9, /*all counters complete=*/0));
      return;
    }
    words_.push_back(build_rdna4_sopp(/*s_wait_kmcnt=*/0x47, imm));
  }

  void s_wait_scratch_loadcnt(uint16_t imm = 0) {
    if (!uses_gfx12_scratch_encoding()) {
      s_wait_kmcnt();
      return;
    }
    words_.push_back(build_rdna4_sopp(/*s_wait_loadcnt=*/0x40, imm));
  }

  void s_wait_scratch_storecnt(uint16_t imm = 0) {
    if (!uses_gfx12_scratch_encoding()) {
      s_wait_kmcnt();
      return;
    }
    words_.push_back(build_rdna4_sopp(/*s_wait_storecnt=*/0x41, imm));
  }

  void s_cbranch_execz(uint16_t offset_dwords) {
    words_.push_back(build_rdna4_sopp(/*op=*/0x25, offset_dwords));
  }

  void s_nop() { words_.push_back(build_rdna4_sopp(/*op=*/0, /*imm=*/0)); }

  void v_mov_b32(uint8_t vdst, uint16_t src0) {
    words_.push_back(build_rdna4_v_mov_b32(vdst, src0));
  }

  void v_mov_b32_literal(uint8_t vdst, uint32_t literal) {
    v_mov_b32(vdst, kScalarLiteral);
    words_.push_back(literal);
  }

  void v_mov_workitem_id(uint8_t vdst) {
    v_mbcnt_lo_u32_b32(vdst, amdgpu_negative_inline_const(1));
    valu_dep_2_barrier();
    v_mbcnt_hi_u32_b32(vdst, amdgpu_negative_inline_const(1), amdgpu_vgpr_src(vdst));
    valu_dep_2_barrier();
  }

  void v_lshlrev_b32(uint8_t vdst, uint16_t src0, uint8_t vsrc1) {
    words_.push_back(build_rdna4_v_lshlrev_b32(vdst, src0, vsrc1));
  }

  void v_add_nc_u32(uint8_t vdst, uint16_t src0, uint8_t vsrc1) {
    words_.push_back(build_rdna4_v_add_nc_u32(vdst, src0, vsrc1));
  }

  void v_xor_b32(uint8_t vdst, uint16_t src0, uint8_t vsrc1) {
    words_.push_back(build_rdna4_v_xor_b32(vdst, src0, vsrc1));
  }

  void v_and_b32(uint8_t vdst, uint16_t src0, uint8_t vsrc1) {
    words_.push_back(build_rdna4_v_and_b32(vdst, src0, vsrc1));
  }

  void v_mbcnt_lo_u32_b32(uint8_t vdst, uint16_t src0,
                           uint16_t src1 = amdgpu_positive_inline_const(0)) {
    words_.push_back(build_rdna4_v_mbcnt_lo_u32_b32_word0(vdst));
    words_.push_back(build_rdna4_v_mbcnt_lo_u32_b32_word1(src0, src1));
  }

  void v_mbcnt_hi_u32_b32(uint8_t vdst, uint16_t src0, uint16_t src1) {
    words_.push_back(build_rdna4_v_mbcnt_hi_u32_b32_word0(vdst));
    words_.push_back(build_rdna4_v_mbcnt_hi_u32_b32_word1(src0, src1));
  }

  void v_cmp_eq_u32_vcc(uint16_t src0, uint8_t vsrc1) {
    words_.push_back(build_rdna4_v_cmp_eq_u32_vcc(src0, vsrc1));
  }

  void v_cmp_eq_u32_sdst(uint8_t sdst, uint16_t src0, uint16_t src1) {
    words_.push_back(build_rdna4_v_cmp_eq_u32_sdst_word0(sdst));
    words_.push_back(build_rdna4_v_cmp_eq_u32_sdst_word1(src0, src1));
  }

  void v_cmpx_eq_u32_exec(uint16_t src0, uint8_t vsrc1) {
    words_.push_back(build_rdna4_v_cmpx_eq_u32_exec(src0, vsrc1));
  }

  void v_readfirstlane_b32(uint8_t sdst, uint8_t vsrc) {
    words_.push_back(build_rdna4_v_readfirstlane_b32(sdst, vsrc));
  }

  void global_load_b32(uint8_t vdst, uint8_t vaddr, uint8_t saddr_lo) {
    if (uses_gfx9_flat_global_encoding()) {
      words_.push_back(0xDC508000u);
      words_.push_back((static_cast<uint32_t>(vdst) << 24) |
                       (static_cast<uint32_t>(saddr_lo) << 16) | vaddr);
      return;
    }
    if (uses_gfx10_flat_global_encoding()) {
      words_.push_back(0xDC308000u);
      words_.push_back((static_cast<uint32_t>(vdst) << 24) |
                       (static_cast<uint32_t>(saddr_lo) << 16) | vaddr);
      return;
    }
    if (uses_gfx11_memory_encoding()) {
      words_.push_back(0xDC520000u);
      words_.push_back((static_cast<uint32_t>(vdst) << 24) |
                       (static_cast<uint32_t>(saddr_lo) << 16) | vaddr);
      s_nop();
      return;
    }
    words_.push_back(0xEE050000u | saddr_lo);
    words_.push_back(vdst);
    words_.push_back(vaddr);
  }

  void global_store_b32(uint8_t vaddr, uint8_t vdata, uint8_t saddr_lo) {
    if (uses_gfx9_flat_global_encoding()) {
      words_.push_back(0xDC708000u);
      words_.push_back((static_cast<uint32_t>(saddr_lo) << 16) |
                       (static_cast<uint32_t>(vdata) << 8) | vaddr);
      return;
    }
    if (uses_gfx10_flat_global_encoding()) {
      words_.push_back(0xDC708000u);
      words_.push_back((static_cast<uint32_t>(saddr_lo) << 16) |
                       (static_cast<uint32_t>(vdata) << 8) | vaddr);
      return;
    }
    if (uses_gfx11_memory_encoding()) {
      words_.push_back(0xDC6A0000u);
      words_.push_back((static_cast<uint32_t>(saddr_lo) << 16) |
                       (static_cast<uint32_t>(vdata) << 8) | vaddr);
      s_nop();
      return;
    }
    words_.push_back(0xEE068000u | saddr_lo);
    words_.push_back(static_cast<uint32_t>(vdata) << 23);
    words_.push_back(vaddr);
  }

  // Scratch encodings are pinned in ProbeDecodeUnit against llvm-mc-emitted
  // gfx11/gfx12 words. Spill probes still need a separate address builder.
  bool scratch_load_b32(uint8_t vdst, uint8_t vaddr, uint8_t saddr, uint32_t byte_offset) {
    if (!scratch_offset_fits(byte_offset))
      return false;
    if (uses_gfx11_memory_encoding()) {
      words_.push_back(0xDC510000u | (byte_offset & 0x1FFFu));
      words_.push_back(static_cast<uint32_t>(vaddr) | (static_cast<uint32_t>(saddr) << 16) |
                       0x00800000u | (static_cast<uint32_t>(vdst) << 24));
      return true;
    }
    if (uses_gfx12_scratch_encoding()) {
      words_.push_back(0xED050000u | (saddr & 0x7Fu));
      words_.push_back(0x00020000u | (vdst & 0xFFu));
      words_.push_back(static_cast<uint32_t>(vaddr) | ((byte_offset & 0xFFFFFFu) << 8));
      return true;
    }
    return false;
  }

  bool scratch_store_b32(uint8_t vaddr, uint8_t vdata, uint8_t saddr, uint32_t byte_offset) {
    if (!scratch_offset_fits(byte_offset))
      return false;
    if (uses_gfx11_memory_encoding()) {
      words_.push_back(0xDC690000u | (byte_offset & 0x1FFFu));
      words_.push_back(static_cast<uint32_t>(vaddr) | (static_cast<uint32_t>(vdata) << 8) |
                       (static_cast<uint32_t>(saddr) << 16) | 0x00800000u);
      return true;
    }
    if (uses_gfx12_scratch_encoding()) {
      words_.push_back(0xED068000u | (saddr & 0x7Fu));
      words_.push_back(0x00020000u | (static_cast<uint32_t>(vdata) << 23));
      words_.push_back(static_cast<uint32_t>(vaddr) | ((byte_offset & 0xFFFFFFu) << 8));
      return true;
    }
    return false;
  }

  void global_atomic_add_u32(uint8_t vaddr, uint8_t vdata, uint8_t saddr_lo,
                             bool rdna4_scope_dev = false) {
    if (uses_gfx9_flat_global_encoding()) {
      words_.push_back(0xDD088000u);
      words_.push_back((static_cast<uint32_t>(saddr_lo) << 16) |
                       (static_cast<uint32_t>(vdata) << 8) | vaddr);
      return;
    }
    if (uses_gfx10_flat_global_encoding()) {
      words_.push_back(0xDCC88000u);
      words_.push_back((static_cast<uint32_t>(saddr_lo) << 16) |
                       (static_cast<uint32_t>(vdata) << 8) | vaddr);
      return;
    }
    if (uses_gfx11_memory_encoding()) {
      words_.push_back(0xDCD60000u);
      words_.push_back((static_cast<uint32_t>(saddr_lo) << 16) |
                       (static_cast<uint32_t>(vdata) << 8) | vaddr);
      s_nop();
      return;
    }
    words_.push_back(0xEE0D4000u | saddr_lo);
    words_.push_back((static_cast<uint32_t>(vdata) << 23) | (rdna4_scope_dev ? 0x00080000u : 0));
    words_.push_back(vaddr);
  }

  void save_exec64() {
    s_mov_b32(regs_.saved_exec_sgpr, kScalarExecLo);
    s_mov_b32(static_cast<uint8_t>(regs_.saved_exec_sgpr + 1), kScalarExecHi);
  }

  void save_scc() {
    s_cselect_b32(regs_.scc_sgpr, amdgpu_positive_inline_const(1),
                  amdgpu_positive_inline_const(0));
  }

  void restore_scc() {
    s_cmp_lg_u32(regs_.scc_sgpr, amdgpu_positive_inline_const(0));
  }

  void restore_exec64() {
    s_or_b32(static_cast<uint8_t>(kScalarExecLo), kScalarExecLo, regs_.saved_exec_sgpr);
    s_or_b32(static_cast<uint8_t>(kScalarExecHi), kScalarExecHi,
             static_cast<uint16_t>(regs_.saved_exec_sgpr + 1));
  }

  void restore_exec64_exact() {
    s_mov_b32(static_cast<uint8_t>(kScalarExecLo), regs_.saved_exec_sgpr);
    s_mov_b32(static_cast<uint8_t>(kScalarExecHi),
              static_cast<uint16_t>(regs_.saved_exec_sgpr + 1));
  }

  void force_exec_lane0() {
    s_mov_b32(static_cast<uint8_t>(kScalarExecLo), amdgpu_positive_inline_const(1));
    s_mov_b32(static_cast<uint8_t>(kScalarExecHi), amdgpu_positive_inline_const(0));
  }

  void mask_exec_to_first_active_lane() {
    s_mov_b32(regs_.tmp0_sgpr, regs_.saved_exec_sgpr);
    v_mbcnt_lo_u32_b32(regs_.tmp0_vgpr, regs_.tmp0_sgpr);
    valu_dep_2_barrier();
    s_mov_b32(regs_.tmp0_sgpr, static_cast<uint16_t>(regs_.saved_exec_sgpr + 1));
    v_mbcnt_hi_u32_b32(regs_.tmp0_vgpr, regs_.tmp0_sgpr, amdgpu_vgpr_src(regs_.tmp0_vgpr));
    v_cmp_eq_u32_sdst(regs_.tmp1_sgpr, amdgpu_positive_inline_const(0),
                      amdgpu_vgpr_src(regs_.tmp0_vgpr));
    s_and_b32(regs_.tmp1_sgpr, regs_.saved_exec_sgpr, regs_.tmp1_sgpr);
    s_and_b32(static_cast<uint8_t>(regs_.tmp1_sgpr + 1),
              static_cast<uint16_t>(regs_.saved_exec_sgpr + 1),
              static_cast<uint16_t>(regs_.tmp1_sgpr + 1));
    s_wait_alu(0xfffe);
    s_mov_b32(static_cast<uint8_t>(kScalarExecLo), regs_.tmp1_sgpr);
    s_mov_b32(static_cast<uint8_t>(kScalarExecHi), static_cast<uint16_t>(regs_.tmp1_sgpr + 1));
  }

  void mask_exec_to_first_active_lane_cmpx() {
    v_mbcnt_lo_u32_b32(regs_.tmp0_vgpr, regs_.saved_exec_sgpr);
    valu_dep_2_barrier();
    v_mbcnt_hi_u32_b32(regs_.tmp0_vgpr, static_cast<uint16_t>(regs_.saved_exec_sgpr + 1),
                       amdgpu_vgpr_src(regs_.tmp0_vgpr));
    v_cmpx_eq_u32_exec(amdgpu_positive_inline_const(0), regs_.tmp0_vgpr);
    s_wait_alu(0xfffe);
  }

private:
  bool uses_gfx9_flat_global_encoding() const {
    return arch_ == ROCJITSU_CODE_ARCH_CDNA2 || arch_ == ROCJITSU_CODE_ARCH_CDNA3 ||
           arch_ == ROCJITSU_CODE_ARCH_CDNA4;
  }

  bool uses_gfx10_flat_global_encoding() const {
    return arch_ == ROCJITSU_CODE_ARCH_RDNA1 || arch_ == ROCJITSU_CODE_ARCH_RDNA2;
  }

  bool uses_gfx9_or_gfx10_waitcnt() const {
    return uses_gfx9_flat_global_encoding() || uses_gfx10_flat_global_encoding();
  }

  bool supports_s_delay_alu() const { return !uses_gfx9_or_gfx10_waitcnt(); }

  bool uses_gfx11_global_encoding() const {
    return arch_ == ROCJITSU_CODE_ARCH_RDNA3 || arch_ == ROCJITSU_CODE_ARCH_RDNA3_5;
  }

  bool uses_gfx11_memory_encoding() const { return uses_gfx11_global_encoding(); }

  bool uses_gfx12_scratch_encoding() const { return arch_ == ROCJITSU_CODE_ARCH_RDNA4; }

  bool scratch_offset_fits(uint32_t byte_offset) const {
    if (uses_gfx11_memory_encoding())
      return byte_offset <= 0x0FFFu;
    if (uses_gfx12_scratch_encoding())
      return byte_offset <= 0x7FFFFFu;
    return false;
  }

  Rdna4ProbeRegisters regs_;
  rj_code_arch_t arch_ = ROCJITSU_CODE_ARCH_RDNA4;
  std::vector<uint32_t> words_;
};

inline std::optional<std::vector<uint32_t>>
wrap_probe_with_vgpr_scratch_spills(std::span<const uint32_t> probe,
                                    const ProbeScratchSpillPlan &spill_plan,
                                    const Rdna4ProbeRegisters &probe_registers,
                                    rj_code_arch_t arch,
                                    bool use_saved_exec_for_address = true) {
  if (spill_plan.vgpr_spills.empty() && spill_plan.sgpr_spills.empty())
    return std::vector<uint32_t>(probe.begin(), probe.end());
  if (!is_power_of_two_u32(spill_plan.private_segment_bytes))
    return std::nullopt;
  if (!spill_plan.sgpr_spills.empty() && spill_plan.vgpr_spills.empty())
    return std::nullopt;

  for (const ProbeScratchSpillSlot &slot : spill_plan.vgpr_spills) {
    if (slot.vgpr == spill_plan.address_vgpr)
      return std::nullopt;
  }

  const uint32_t private_segment_shift =
      log2_power_of_two_u32(spill_plan.private_segment_bytes);
  if (private_segment_shift > kMaxInlinePositiveImm)
    return std::nullopt;

  auto build_scratch_address = [&](Rdna4ProbeBuilder &builder) {
    const uint16_t exec_lo =
        use_saved_exec_for_address ? probe_registers.saved_exec_sgpr : kScalarExecLo;
    const uint16_t exec_hi =
        use_saved_exec_for_address
            ? static_cast<uint16_t>(probe_registers.saved_exec_sgpr + 1)
            : kScalarExecHi;
    if (use_saved_exec_for_address)
      builder.save_exec64();
    builder.v_mbcnt_lo_u32_b32(spill_plan.address_vgpr, exec_lo);
    builder.valu_dep_1_barrier();
    if (use_saved_exec_for_address || !spill_plan.wave32) {
      builder.v_mbcnt_hi_u32_b32(
          spill_plan.address_vgpr, exec_hi, amdgpu_vgpr_src(spill_plan.address_vgpr));
      builder.valu_dep_1_barrier();
    }
    builder.v_lshlrev_b32(spill_plan.address_vgpr,
                          amdgpu_positive_inline_const(private_segment_shift),
                          spill_plan.address_vgpr);
    builder.valu_dep_1_barrier();
  };

  Rdna4ProbeBuilder save(probe_registers, arch);
  // Scratch backing is per wave. Fixed-counter probes do not change EXEC, so
  // they can derive the compacted per-lane scratch offset from EXEC directly.
  // Previous-BB probes can use their saved-EXEC pair for address formation when
  // that pair is not itself scratch-spilled. If the wrapper spills SGPRs, it
  // derives the address from EXEC before the probe and again after the probe has
  // restored EXEC.
  build_scratch_address(save);
  for (const ProbeScratchSpillSlot &slot : spill_plan.vgpr_spills) {
    if (!save.scratch_store_b32(spill_plan.address_vgpr, slot.vgpr,
                                kScratchNoScalarOffset, slot.byte_offset)) {
      return std::nullopt;
    }
  }
  const uint8_t sgpr_data_vgpr = spill_plan.vgpr_spills.front().vgpr;
  for (const ProbeScratchSgprSpillSlot &slot : spill_plan.sgpr_spills) {
    save.v_mov_b32(sgpr_data_vgpr, slot.sgpr);
    save.valu_dep_1_barrier();
    if (!save.scratch_store_b32(spill_plan.address_vgpr, sgpr_data_vgpr,
                                kScratchNoScalarOffset, slot.byte_offset)) {
      return std::nullopt;
    }
  }
  save.s_wait_scratch_storecnt();

  std::vector<uint32_t> words = save.take();
  words.insert(words.end(), probe.begin(), probe.end());

  Rdna4ProbeBuilder fill(probe_registers, arch);
  fill.s_wait_scratch_storecnt();
  build_scratch_address(fill);
  for (const ProbeScratchSgprSpillSlot &slot : spill_plan.sgpr_spills) {
    if (!fill.scratch_load_b32(sgpr_data_vgpr, spill_plan.address_vgpr,
                               kScratchNoScalarOffset, slot.byte_offset)) {
      return std::nullopt;
    }
    fill.s_wait_scratch_loadcnt();
    fill.v_readfirstlane_b32(slot.sgpr, sgpr_data_vgpr);
    fill.s_wait_alu(0xfffe);
  }
  for (const ProbeScratchSpillSlot &slot : spill_plan.vgpr_spills) {
    if (!fill.scratch_load_b32(slot.vgpr, spill_plan.address_vgpr,
                               kScratchNoScalarOffset, slot.byte_offset)) {
      return std::nullopt;
    }
  }
  fill.s_wait_scratch_loadcnt();
  std::vector<uint32_t> fill_words = fill.take();
  words.insert(words.end(), fill_words.begin(), fill_words.end());
  if (!spill_plan.sgpr_spills.empty()) {
    if (words.size() > std::numeric_limits<uint16_t>::max())
      return std::nullopt;
    Rdna4ProbeBuilder guard(probe_registers, arch);
    guard.s_cbranch_execz(static_cast<uint16_t>(words.size()));
    std::vector<uint32_t> guarded = guard.take();
    guarded.insert(guarded.end(), words.begin(), words.end());
    return guarded;
  }
  return words;
}

inline std::optional<std::vector<uint32_t>>
wrap_forced_lane0_probe_with_vgpr_scratch_spills(
    std::span<const uint32_t> probe, const ProbeScratchSpillPlan &spill_plan,
    const Rdna4ProbeRegisters &probe_registers, rj_code_arch_t arch) {
  if (spill_plan.vgpr_spills.empty() || !spill_plan.sgpr_spills.empty())
    return std::nullopt;
  if (!is_power_of_two_u32(spill_plan.private_segment_bytes))
    return std::nullopt;

  for (const ProbeScratchSpillSlot &slot : spill_plan.vgpr_spills) {
    if (slot.vgpr == spill_plan.address_vgpr)
      return std::nullopt;
  }

  const uint32_t private_segment_shift =
      log2_power_of_two_u32(spill_plan.private_segment_bytes);
  if (private_segment_shift > kMaxInlinePositiveImm)
    return std::nullopt;

  auto build_lane0_scratch_address = [&](Rdna4ProbeBuilder &builder) {
    builder.v_mbcnt_lo_u32_b32(spill_plan.address_vgpr, kScalarExecLo);
    builder.valu_dep_1_barrier();
    if (!spill_plan.wave32) {
      builder.v_mbcnt_hi_u32_b32(spill_plan.address_vgpr, kScalarExecHi,
                                 amdgpu_vgpr_src(spill_plan.address_vgpr));
      builder.valu_dep_1_barrier();
    }
    builder.v_lshlrev_b32(spill_plan.address_vgpr,
                          amdgpu_positive_inline_const(private_segment_shift),
                          spill_plan.address_vgpr);
    builder.valu_dep_1_barrier();
  };

  Rdna4ProbeBuilder save(probe_registers, arch);
  save.save_exec64();
  save.force_exec_lane0();
  build_lane0_scratch_address(save);
  for (const ProbeScratchSpillSlot &slot : spill_plan.vgpr_spills) {
    if (!save.scratch_store_b32(spill_plan.address_vgpr, slot.vgpr,
                                kScratchNoScalarOffset, slot.byte_offset)) {
      return std::nullopt;
    }
  }
  save.s_wait_scratch_storecnt();
  save.restore_exec64_exact();

  std::vector<uint32_t> words = save.take();
  words.insert(words.end(), probe.begin(), probe.end());

  Rdna4ProbeBuilder fill(probe_registers, arch);
  fill.save_exec64();
  fill.force_exec_lane0();
  build_lane0_scratch_address(fill);
  for (const ProbeScratchSpillSlot &slot : spill_plan.vgpr_spills) {
    if (!fill.scratch_load_b32(slot.vgpr, spill_plan.address_vgpr,
                               kScratchNoScalarOffset, slot.byte_offset)) {
      return std::nullopt;
    }
  }
  fill.s_wait_scratch_loadcnt();
  fill.restore_exec64_exact();
  std::vector<uint32_t> fill_words = fill.take();
  words.insert(words.end(), fill_words.begin(), fill_words.end());
  return words;
}

inline std::optional<uint32_t> counter_slot_byte_offset(uint32_t slot) {
  if (slot >= kCoverageSlots)
    return std::nullopt;
  const uint64_t byte_offset = static_cast<uint64_t>(slot) * sizeof(uint32_t);
  if (byte_offset > std::numeric_limits<uint32_t>::max())
    return std::nullopt;
  return static_cast<uint32_t>(byte_offset);
}

inline bool counter_slot_uses_inline_offset(uint32_t slot) {
  auto byte_offset = counter_slot_byte_offset(slot);
  return byte_offset && *byte_offset <= kMaxInlinePositiveImm;
}

inline std::optional<int16_t> s_branch_offset_dwords(uint64_t branch_pc, uint64_t target) {
  const int64_t delta_bytes =
      static_cast<int64_t>(target) - static_cast<int64_t>(branch_pc + sizeof(uint32_t));
  if (delta_bytes % static_cast<int64_t>(sizeof(uint32_t)) != 0)
    return std::nullopt;

  const int64_t delta_dwords = delta_bytes / static_cast<int64_t>(sizeof(uint32_t));
  if (delta_dwords < std::numeric_limits<int16_t>::min() ||
      delta_dwords > std::numeric_limits<int16_t>::max())
    return std::nullopt;
  return static_cast<int16_t>(delta_dwords);
}

inline constexpr uint32_t previous_bb_after(uint32_t bb_id) { return bb_id >> 1; }

inline constexpr uint32_t hashed_edge_counter_slot(uint32_t previous_bb, uint32_t bb_id) {
  return (previous_bb ^ bb_id) & kHashEdgeSlotMask;
}

inline bool active_lane(uint64_t active_mask, size_t lane) {
  return lane < 64 && ((active_mask >> lane) & 1u) != 0;
}

inline void increment_counter(std::vector<uint32_t> &counters, uint32_t slot) {
  if (slot < counters.size())
    ++counters[slot];
}

inline void model_record_edge(uint32_t bb_id, uint64_t active_mask,
                              std::vector<uint32_t> &previous_bb, std::vector<uint32_t> &counters,
                              EdgeCounterPolicy policy = EdgeCounterPolicy::FirstActiveLane) {
  bool first_active_lane_counted = false;
  std::vector<uint32_t> distinct_slots;

  for (size_t lane = 0; lane < previous_bb.size() && lane < 64; ++lane) {
    if (!active_lane(active_mask, lane))
      continue;

    const uint32_t slot = hashed_edge_counter_slot(previous_bb[lane], bb_id);
    if (policy == EdgeCounterPolicy::EveryActiveLane) {
      increment_counter(counters, slot);
    } else if (policy == EdgeCounterPolicy::FirstActiveLane) {
      if (!first_active_lane_counted) {
        increment_counter(counters, slot);
        first_active_lane_counted = true;
      }
    } else {
      bool seen = false;
      for (uint32_t existing : distinct_slots) {
        if (existing == slot) {
          seen = true;
          break;
        }
      }
      if (!seen) {
        distinct_slots.push_back(slot);
        increment_counter(counters, slot);
      }
    }

    previous_bb[lane] = previous_bb_after(bb_id);
  }
}

inline std::vector<uint32_t>
rdna4_edge_entry_probe(rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA4,
                       Rdna4ProbeRegisters regs = {}) {
  Rdna4ProbeBuilder b(regs, arch);
  const auto &r = b.regs();
  b.s_load_b64_state_from_kernarg(/*byte_offset=*/0);
  b.save_exec64();
  b.mask_exec_to_first_active_lane();
  b.s_cbranch_execz(/*offset_dwords=*/8);
  b.s_bcnt1_i32_b64(r.tmp0_sgpr, r.saved_exec_sgpr);
  b.s_wait_alu(0xfffe);
  b.v_mov_b32(r.workitem_vgpr, amdgpu_positive_inline_const(0));
  b.v_mov_b32(r.tmp0_vgpr, r.tmp0_sgpr);
  b.s_wait_kmcnt();
  b.global_atomic_add_u32(r.workitem_vgpr, r.tmp0_vgpr, r.state_sgpr);
  b.restore_exec64();
  b.s_wait_kmcnt();
  return b.take();
}

inline std::vector<uint32_t>
rdna4_previous_bb_edge_probe(uint32_t bb_id, bool load_state_base,
                             uint32_t state_pointer_kernarg_offset = 0,
                             rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA4,
                             Rdna4ProbeRegisters regs = {}) {
  Rdna4ProbeBuilder b(regs, arch);
  const auto &r = b.regs();
  if (load_state_base)
    b.s_load_b64_state_from_kernarg(state_pointer_kernarg_offset);

  b.v_mov_workitem_id(r.workitem_vgpr);
  b.save_scc();
  b.save_exec64();
  b.v_lshlrev_b32(r.tmp0_vgpr, amdgpu_positive_inline_const(2), r.workitem_vgpr);
  b.v_mov_b32_literal(r.tmp1_vgpr, kPreviousBbByteOffset);
  b.v_add_nc_u32(r.tmp0_vgpr, amdgpu_vgpr_src(r.tmp1_vgpr), r.tmp0_vgpr);
  b.s_wait_kmcnt();
  b.global_load_b32(r.tmp1_vgpr, r.tmp0_vgpr, r.state_sgpr);
  b.s_wait_kmcnt();

  b.v_mov_b32_literal(r.tmp2_vgpr, bb_id);
  b.v_xor_b32(r.tmp1_vgpr, amdgpu_vgpr_src(r.tmp1_vgpr), r.tmp2_vgpr);
  b.v_and_b32(r.tmp1_vgpr, kScalarLiteral, r.tmp1_vgpr);
  b.word(kHashEdgeSlotMask);
  b.v_lshlrev_b32(r.workitem_vgpr, amdgpu_positive_inline_const(2), r.tmp1_vgpr);
  b.v_mov_b32_literal(r.tmp2_vgpr, previous_bb_after(bb_id));
  b.global_store_b32(r.tmp0_vgpr, r.tmp2_vgpr, r.state_sgpr);
  b.mask_exec_to_first_active_lane();
  b.s_cbranch_execz(/*offset_dwords=*/5);
  b.v_mov_b32(r.tmp0_vgpr, amdgpu_positive_inline_const(1));
  b.s_wait_kmcnt();
  b.global_atomic_add_u32(r.workitem_vgpr, r.tmp0_vgpr, r.state_sgpr);
  b.restore_exec64_exact();
  b.s_wait_kmcnt();
  b.restore_scc();
  return b.take();
}

inline std::vector<uint32_t>
rdna4_previous_bb_edge_probe_with_state_pointer(uint32_t bb_id, uint64_t state_pointer,
                                                rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA4,
                                                Rdna4ProbeRegisters regs = {}) {
  Rdna4ProbeBuilder b(regs, arch);
  b.s_mov_b64_literal(b.regs().state_sgpr, state_pointer);
  std::vector<uint32_t> words = b.take();
  std::vector<uint32_t> tail =
      rdna4_previous_bb_edge_probe(bb_id, /*load_state_base=*/false,
                                   /*state_pointer_kernarg_offset=*/0, arch, regs);
  words.insert(words.end(), tail.begin(), tail.end());
  return words;
}

inline void replace_rdna4_initial_s_load_b64_with_state_pointer(std::vector<uint32_t> &words,
                                                                uint8_t sdst_lo,
                                                                uint64_t state_pointer) {
  if (words.size() < 2)
    return;
  words.erase(words.begin(), words.begin() + 2);
  std::vector<uint32_t> prefix;
  append_rdna4_s_mov_b64_literal(prefix, sdst_lo, state_pointer);
  words.insert(words.begin(), prefix.begin(), prefix.end());
}

inline std::optional<std::vector<uint32_t>>
rdna4_counter_probe(uint32_t slot, rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA4,
                    Rdna4ProbeRegisters regs = {}) {
  auto slot_byte_offset = counter_slot_byte_offset(slot);
  if (!slot_byte_offset)
    return std::nullopt;

  if (slot == kEntryCounterSlot) {
    Rdna4ProbeRegisters regs;
    regs.state_sgpr = 4;
    regs.saved_exec_sgpr = 2;
    regs.tmp0_sgpr = 6;
    regs.tmp1_sgpr = 6;
    regs.workitem_vgpr = 1;
    regs.tmp0_vgpr = 2;

    Rdna4ProbeBuilder b(regs, arch);
    const auto &r = b.regs();
    b.s_load_b64_state_from_kernarg(/*byte_offset=*/0);
    b.save_exec64();
    b.mask_exec_to_first_active_lane();
    b.s_cbranch_execz(/*offset_dwords=*/8);
    b.s_bcnt1_i32_b64(r.tmp0_sgpr, r.saved_exec_sgpr);
    b.s_wait_alu(0xfffe);
    b.v_mov_b32(r.workitem_vgpr, amdgpu_positive_inline_const(0));
    b.v_mov_b32(r.tmp0_vgpr, r.tmp0_sgpr);
    b.s_wait_kmcnt();
    b.global_atomic_add_u32(r.workitem_vgpr, r.tmp0_vgpr, r.state_sgpr,
                            /*rdna4_scope_dev=*/true);
    b.restore_exec64();
    b.s_wait_kmcnt();
    return b.take();
  }

  Rdna4ProbeBuilder b(regs, arch);
  const auto &r = b.regs();
  b.save_exec64();
  b.mask_exec_to_first_active_lane();
  b.s_cbranch_execz(/*offset_dwords=*/10);
  b.s_bcnt1_i32_b64(r.tmp0_sgpr, r.saved_exec_sgpr);
  b.s_wait_alu(0xfffe);
  b.v_mov_b32(r.workitem_vgpr, amdgpu_positive_inline_const(0));
  b.v_mov_b32(r.tmp0_vgpr, r.tmp0_sgpr);
  if (*slot_byte_offset <= kMaxInlinePositiveImm) {
    b.v_mov_b32(r.workitem_vgpr, amdgpu_positive_inline_const(*slot_byte_offset));
  } else {
    b.v_mov_b32_literal(r.workitem_vgpr, *slot_byte_offset);
  }
  b.s_delay_alu(/*VALU_DEP_1=*/1);
  b.s_wait_kmcnt();
  b.global_atomic_add_u32(r.workitem_vgpr, r.tmp0_vgpr, r.state_sgpr);
  b.restore_exec64();
  b.s_wait_kmcnt();
  return b.take();
}

inline std::optional<std::vector<uint32_t>>
rdna4_flagless_counter_probe(uint32_t slot, rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA4,
                             Rdna4ProbeRegisters regs = {}) {
  if (slot >= kCoverageSlots)
    return std::nullopt;

  Rdna4ProbeBuilder b(regs, arch);
  const auto &r = b.regs();
  const uint32_t slot_byte_offset = slot * sizeof(uint32_t);
  if (slot_byte_offset <= kMaxInlinePositiveImm) {
    b.v_mov_b32(r.workitem_vgpr, amdgpu_positive_inline_const(slot_byte_offset));
  } else {
    b.v_mov_b32_literal(r.workitem_vgpr, slot_byte_offset);
  }
  b.v_mov_b32(r.tmp0_vgpr, amdgpu_positive_inline_const(1));
  b.valu_dep_1_barrier();
  b.global_atomic_add_u32(r.workitem_vgpr, r.tmp0_vgpr, r.state_sgpr);
  b.s_wait_kmcnt();
  return b.take();
}

inline std::optional<std::vector<uint32_t>>
rdna4_flagless_counter_probe_force_lane0(uint32_t slot,
                                         rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA4,
                                         Rdna4ProbeRegisters regs = {}) {
  if (slot >= kCoverageSlots)
    return std::nullopt;

  Rdna4ProbeBuilder b(regs, arch);
  const auto &r = b.regs();
  const uint32_t slot_byte_offset = slot * sizeof(uint32_t);
  b.save_exec64();
  b.force_exec_lane0();
  if (slot_byte_offset <= kMaxInlinePositiveImm) {
    b.v_mov_b32(r.workitem_vgpr, amdgpu_positive_inline_const(slot_byte_offset));
  } else {
    b.v_mov_b32_literal(r.workitem_vgpr, slot_byte_offset);
  }
  b.v_mov_b32(r.tmp0_vgpr, amdgpu_positive_inline_const(1));
  b.valu_dep_1_barrier();
  b.global_atomic_add_u32(r.workitem_vgpr, r.tmp0_vgpr, r.state_sgpr);
  b.restore_exec64_exact();
  b.s_wait_kmcnt();
  return b.take();
}

inline std::optional<std::vector<uint32_t>>
rdna4_flagless_counter_probe_with_state_pointer(uint32_t slot, uint64_t state_pointer,
                                                rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA4,
                                                Rdna4ProbeRegisters regs = {}) {
  auto counter = rdna4_flagless_counter_probe(slot, arch, regs);
  if (!counter)
    return std::nullopt;

  Rdna4ProbeBuilder b(regs, arch);
  b.s_mov_b64_literal(b.regs().state_sgpr, state_pointer);
  std::vector<uint32_t> words = b.take();
  words.insert(words.end(), counter->begin(), counter->end());
  return words;
}

inline std::optional<std::vector<uint32_t>>
rdna4_flagless_counter_probe_force_lane0_with_state_pointer(
    uint32_t slot, uint64_t state_pointer, rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA4,
    Rdna4ProbeRegisters regs = {}) {
  auto counter = rdna4_flagless_counter_probe_force_lane0(slot, arch, regs);
  if (!counter)
    return std::nullopt;

  Rdna4ProbeBuilder b(regs, arch);
  b.s_mov_b64_literal(b.regs().state_sgpr, state_pointer);
  std::vector<uint32_t> words = b.take();
  words.insert(words.end(), counter->begin(), counter->end());
  return words;
}

} // namespace rocjitsu::fuzzer::afl_dbi
