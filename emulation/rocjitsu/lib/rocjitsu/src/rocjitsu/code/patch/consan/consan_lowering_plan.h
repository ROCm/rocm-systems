// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_lowering_plan.h
/// @brief Target-neutral planning facts and resource demand for ConSan.

#pragma once

#include "rocjitsu/code/patch/consan/consan_access_target.h"
#include "rocjitsu/code/patch/consan/consan_register_allocation.h"
#include "rocjitsu/code/patch/consan/consan_report_contract.h"

#include <cstddef>
#include <span>
#include <string_view>

namespace rocjitsu::consan::detail {

/// Admitted access and atomic consumers used to size ConSan instrumentation.
struct ObjectFacts {
  bool has_access_candidate = false;
  size_t admitted_atomic_count = 0;
};

/// Effective ConSan options and report geometry for one code object.
struct ObjectModePlan {
  OwnerSource owner_source = OwnerSource::Automatic;
  bool track_atomics = false;
  ObjectModeSemantics semantics;
  /// Empty when planning needs no user-visible warning; otherwise a static message.
  std::string_view warning;
};

static_assert(sizeof(ObjectModePlan) <= 96);

/// Operational sites that survived semantic admission and resource planning.
/// ConSan derives persistent-state requirements from these consumers.
struct PersistentStateFacts {
  size_t access_count = 0;
  size_t atomic_count = 0;
  size_t barrier_count = 0;
  bool has_operational_dynamic_stack_owner = false;
};

struct PersistentStateDemand {
  bool needs_entry_workgroup_tuple = false;
  bool needs_persistent_state = false;
  bool synchronization_requires_persistent_owner = false;
  bool private_workgroup_tuple_supported = false;
  bool private_state_supported = false;
  bool scalar_state_required_for_private_or_overflow = false;
};

/// Mode-owned permission to retry an otherwise unplaceable site by spilling a
/// scratch window that overlaps short-lived guest operands. Common placement
/// executes the retry; the selected mode owns whether its emission ordering
/// can recover the operands and which guest result must remain disjoint.
struct OperandOverlapSpillPolicy {
  bool supported = false;
  std::optional<uint16_t> protected_vgpr;
  uint8_t protected_vgpr_count = 0;
};

struct OperandOverlapSpillContext {
  const Request &request;
  const AccessResourceFacts &resource_facts;
  const Candidate *access_candidate = nullptr;
  ResourceSiteKind site_kind = ResourceSiteKind::Access;
};

/// Mode-owned request for a smaller spill-backed transaction after ordinary
/// access placement. Common placement retries with the returned scratch size.
struct AccessSpillFallbackContext {
  const Request &request;
  const AccessResourceFacts &resource_facts;
  bool no_ordinary_window = false;
  bool initial_spill_overlaps_guest = false;
};

/// Normalized facts used by a mode to declare its dispatch-identity demand.
/// Target-family inspection and site traversal remain common solver work.
struct DispatchIdentityFacts {
  bool access_reports_need_explicit_identity = true;
  bool has_access_or_atomic_consumer = false;
};

/// Complete scalar preservation ABI selected for ConSan.
struct ScalarAbiPlan {
  std::optional<detail::SpecialStateSgprs> special_state;
};

/// Caller and target bounds used to plan ConSan evidence capacity.
struct EvidencePlanningContext {
  const ProgramInventory &program_inventory;
  const ObservationPlan &observation_plan;
  uint64_t requested_report_buffer_size = 0;
  std::optional<uint64_t> maximum_access_probe_count;
  std::optional<uint32_t> maximum_workgroup_lds_bytes;
  uint32_t watchpoint_banks = 0;
};

/// Exact target-neutral facts used by ConSan to size one barrier probe. The
/// common resource solver projects these facts but does not interpret them.
struct BarrierScratchFacts {
  bool automatic_private_epoch = false;
  bool persistent_scalar_state_complete = false;
};

[[nodiscard]] inline BarrierScratchFacts
project_barrier_scratch_facts(const OperatingPoint &point) {
  return {point.automatic_private_epoch, point.persistent_sgprs.complete()};
}

[[nodiscard]] EvidenceRequirements
plan_evidence_requirements(const EvidencePlanningContext &context);

ObjectModePlan plan_object_mode(const Request &, const BoundRuntimeResources &,
                                const TransformPolicy &, const ObjectFacts &);

void apply_mode_patches(std::span<const uint8_t>, const Options &, OperatingPoint &, rj_code_arch_t,
                        ResourcePlanningState &, std::span<const Candidate>,
                        const ObjectModeSemantics &, TransformArtifacts &);

uint16_t barrier_scratch_vgpr_count(const BarrierScratchFacts &);

PersistentStateDemand plan_persistent_state_demand(const Request &, const OperatingPoint &,
                                                   const PersistentStateFacts &);

OperandOverlapSpillPolicy operand_overlap_spill(const OperandOverlapSpillContext &);

std::optional<uint16_t> access_spill_fallback(const AccessSpillFallbackContext &);

bool requires_dispatch_identity(const Request &, const DispatchIdentityFacts &);

ScalarAbiPlan plan_scalar_abi(const ScalarPreservationState &);

bool plan_report_layout(const AutoReportInventory &, AutoReportPlan &, uint64_t &cursor);

std::optional<AutoReportInventory> reconstruct_report_inventory(const ReportBufferLayout &);
inline constexpr uint16_t kDynamicStackFrameSaveSgprOffset = 8u;

} // namespace rocjitsu::consan::detail
