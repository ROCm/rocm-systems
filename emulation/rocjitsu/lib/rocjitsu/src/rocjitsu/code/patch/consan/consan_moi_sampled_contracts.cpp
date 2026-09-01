// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi_sampled_contracts.h"

#include "rocjitsu/code/patch/consan/consan_moi_access_target.h"

namespace rocjitsu::consan_moi_impl {

bool sampled_access_can_emit_spill_over_guest_operands(const MoiAccessResourceFacts &resource_facts,
                                                       const ConSanMoiCandidate &candidate) {
  return candidate.lowering.form && !candidate.lowering.form->destination_vgpr &&
         !candidate.is_direct_to_lds() && resource_facts.has_persistent_owner_vgpr &&
         !resource_facts.uses_private_epoch && !resource_facts.has_complete_persistent_sgprs &&
         !candidate_requires_flat_address_materialization(candidate);
}

bool sampled_access_can_plan_spill_over_guest_operands(const ConSanRequest &request,
                                                       const MoiAccessResourceFacts &resource_facts,
                                                       const ConSanMoiCandidate &candidate) {
  if (sampled_access_can_emit_spill_over_guest_operands(resource_facts, candidate))
    return true;
  if (!candidate.lowering.form || candidate.lowering.form->destination_vgpr ||
      candidate.is_direct_to_lds() || resource_facts.uses_private_epoch ||
      resource_facts.has_complete_persistent_sgprs ||
      candidate_requires_flat_address_materialization(candidate)) {
    return false;
  }
  // Automatic persistent-state placement runs after the first resource pass.
  // Admit the overlap provisionally when entry-persistent owner state will be
  // resolved later; emission still requires a concrete persistent owner VGPR.
  return resource_facts.initialize_owner_epoch || request.moi_track_atomics ||
         request.moi_track_barriers || request.moi_runtime_sample_stride > 1u;
}

uint16_t direct_sampled_scratch_count(const ConSanRequest &request,
                                      const MoiAccessResourceFacts &resource_facts) {
  const uint16_t base_scratch_count = request.moi_sampled_check ? 7u : 5u;
  return static_cast<uint16_t>(
      base_scratch_count + 1u + resource_facts.address_scratch_vgpr_count +
      (resource_facts.uses_private_epoch
           ? (request.moi_owner_source == ConSanMoiOwnerSource::WorkitemId ? 2u : 1u)
       : resource_facts.has_complete_persistent_sgprs ? 2u
                                                      : 0u) +
      resource_facts.two_address_replay_vgpr_count);
}

std::optional<MoiSampledPublicationStateSgprs>
moi_sampled_publication_state_sgprs(const ConSanRequest &request,
                                    const ConSanMoiOperatingPoint &point) {
  if (request.moi_engine != ConSanMoiEngine::Sampled || !point.moi_exec_save_sgpr)
    return std::nullopt;
  const uint16_t base = *point.moi_exec_save_sgpr;
  return MoiSampledPublicationStateSgprs{
      .original_exec_save_sgpr = base,
      .selection_vcc_save_sgpr = static_cast<uint16_t>(base + 2u),
      .publication_exec_save_sgpr = static_cast<uint16_t>(base + 4u),
      .guest_scc_snapshot_sgpr = static_cast<uint16_t>(base + 6u),
  };
}

} // namespace rocjitsu::consan_moi_impl
