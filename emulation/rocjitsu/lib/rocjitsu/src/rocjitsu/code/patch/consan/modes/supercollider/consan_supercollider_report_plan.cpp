// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi_report_planning.h"

namespace rocjitsu {

bool ConSanSuperColliderEvidenceRequirements::well_formed() const {
  if (reason != ConSanEvidenceRequirementReason::None)
    return false;
  if (mode == ConSanSuperColliderEvidenceMode::TrapOnly)
    return marker_bytes == 0u && runtime_requirements == RuntimeCapabilityRequirements{};
  return mode == ConSanSuperColliderEvidenceMode::StickyMarker &&
         (marker_bytes == 0u || marker_bytes == sizeof(uint32_t)) &&
         runtime_requirements.host_device_visible_memory &&
         runtime_requirements.host_device_coherent_memory &&
         !runtime_requirements.device_atomic_publication &&
         runtime_requirements.minimum_report_allocation_bytes == marker_bytes &&
         !runtime_requirements.max_workgroup_lds_bytes && runtime_requirements.executable_binding &&
         !runtime_requirements.dispatch_segment_binding;
}

ConSanSuperColliderEvidenceRequirements
plan_consan_supercollider_evidence(const ConSanEvidenceIntentPlan &evidence_intents,
                                   ConSanSuperColliderEvidenceMode mode) {
  ConSanSuperColliderEvidenceRequirements requirements;
  requirements.mode = mode;
  requirements.reason = consan_moi_impl::validate_moi_evidence_intents(
      evidence_intents, ConSanCapabilityEngine::SuperCollider);
  if (requirements.reason != ConSanEvidenceRequirementReason::None)
    return requirements;
  if (mode != ConSanSuperColliderEvidenceMode::TrapOnly &&
      mode != ConSanSuperColliderEvidenceMode::StickyMarker) {
    requirements.reason = ConSanEvidenceRequirementReason::InvalidObservationPlan;
    return requirements;
  }

  if (mode == ConSanSuperColliderEvidenceMode::TrapOnly) {
    requirements.reason = ConSanEvidenceRequirementReason::None;
    return requirements;
  }

  requirements.marker_bytes = evidence_intents.intents.empty() ? 0u : sizeof(uint32_t);
  requirements.runtime_requirements.host_device_visible_memory = true;
  requirements.runtime_requirements.host_device_coherent_memory = true;
  requirements.runtime_requirements.minimum_report_allocation_bytes = requirements.marker_bytes;
  requirements.runtime_requirements.executable_binding = true;
  requirements.reason = ConSanEvidenceRequirementReason::None;
  return requirements;
}

} // namespace rocjitsu
