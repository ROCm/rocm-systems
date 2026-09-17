// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_report_planning.h"

namespace rocjitsu::consan {

bool SuperColliderEvidenceRequirements::well_formed() const {
  if (reason != EvidenceRequirementReason::None)
    return false;
  if (mode == SuperColliderEvidenceMode::TrapOnly)
    return marker_bytes == 0u && runtime_requirements == RuntimeCapabilityRequirements{};
  return mode == SuperColliderEvidenceMode::StickyMarker &&
         (marker_bytes == 0u || marker_bytes == sizeof(uint32_t)) &&
         runtime_requirements.host_device_visible_memory &&
         runtime_requirements.host_device_coherent_memory &&
         !runtime_requirements.device_atomic_publication &&
         runtime_requirements.minimum_report_allocation_bytes == marker_bytes &&
         !runtime_requirements.max_workgroup_lds_bytes && runtime_requirements.executable_binding &&
         !runtime_requirements.dispatch_segment_binding;
}

SuperColliderEvidenceRequirements
plan_supercollider_evidence(const ObservationPlan &observation_plan,
                            SuperColliderEvidenceMode mode) {
  SuperColliderEvidenceRequirements requirements;
  requirements.mode = mode;
  requirements.reason = detail::validate_evidence_intents(observation_plan, Mode::SuperCollider);
  if (requirements.reason != EvidenceRequirementReason::None)
    return requirements;
  if (mode != SuperColliderEvidenceMode::TrapOnly &&
      mode != SuperColliderEvidenceMode::StickyMarker) {
    requirements.reason = EvidenceRequirementReason::InvalidObservationPlan;
    return requirements;
  }

  if (mode == SuperColliderEvidenceMode::TrapOnly) {
    requirements.reason = EvidenceRequirementReason::None;
    return requirements;
  }

  requirements.marker_bytes = observation_plan.probe_intents.empty() ? 0u : sizeof(uint32_t);
  requirements.runtime_requirements.host_device_visible_memory = true;
  requirements.runtime_requirements.host_device_coherent_memory = true;
  requirements.runtime_requirements.minimum_report_allocation_bytes = requirements.marker_bytes;
  requirements.runtime_requirements.executable_binding = true;
  requirements.reason = EvidenceRequirementReason::None;
  return requirements;
}

} // namespace rocjitsu::consan
