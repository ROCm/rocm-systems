// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_contracts.h"

#include "rocjitsu/code/patch/consan/consan_access_target.h"

namespace rocjitsu::consan::detail {

bool access_can_emit_spill_over_guest_operands(const AccessResourceFacts &resource_facts,
                                               const Candidate &candidate) {
  return candidate.site().lowering.form && !candidate.site().lowering.form->destination_vgpr &&
         !candidate.is_direct_to_lds() && resource_facts.has_persistent_owner_vgpr &&
         !resource_facts.uses_private_epoch && !resource_facts.has_complete_persistent_sgprs &&
         !candidate_requires_flat_address_materialization(candidate);
}

bool access_can_plan_spill_over_guest_operands(const Request &request,
                                               const AccessResourceFacts &resource_facts,
                                               const Candidate &candidate) {
  if (access_can_emit_spill_over_guest_operands(resource_facts, candidate))
    return true;
  if (!candidate.site().lowering.form || candidate.site().lowering.form->destination_vgpr ||
      candidate.is_direct_to_lds() || resource_facts.uses_private_epoch ||
      resource_facts.has_complete_persistent_sgprs ||
      candidate_requires_flat_address_materialization(candidate)) {
    return false;
  }
  // Automatic persistent-state placement runs after the first resource pass.
  // Admit the overlap provisionally when entry-persistent owner state will be
  // resolved later; emission still requires a concrete persistent owner VGPR.
  return resource_facts.initialize_owner_epoch || request.track_atomics || request.track_barriers ||
         (request.runtime_sample_stride > 1u || request.cell_selector().stride > 1u);
}

uint16_t direct_scratch_count(const Request &request, const AccessResourceFacts &resource_facts) {
  const uint16_t base_scratch_count = request.device_conflict_check ? 7u : 5u;
  return static_cast<uint16_t>(base_scratch_count + 1u + resource_facts.address_scratch_vgpr_count +
                               (resource_facts.uses_private_epoch
                                    ? (request.owner_source == OwnerSource::WorkitemId ? 2u : 1u)
                                : resource_facts.has_complete_persistent_sgprs ? 2u
                                                                               : 0u) +
                               resource_facts.two_address_replay_vgpr_count);
}

std::optional<PublicationStateSgprs>
publication_state_sgprs(std::optional<uint16_t> exec_save_sgpr) {
  if (!exec_save_sgpr)
    return std::nullopt;
  const uint16_t base = *exec_save_sgpr;
  return PublicationStateSgprs{
      .original_exec_save_sgpr = base,
      .selection_vcc_save_sgpr = static_cast<uint16_t>(base + 2u),
      .publication_exec_save_sgpr = static_cast<uint16_t>(base + 4u),
      .guest_scc_snapshot_sgpr = static_cast<uint16_t>(base + 6u),
  };
}

} // namespace rocjitsu::consan::detail
