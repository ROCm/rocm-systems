// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi_sampled_contracts.h"

#include "rocjitsu/code/patch/consan/consan_capability_contract.h"
#include "rocjitsu/code/patch/consan/consan_moi_access_target.h"

namespace rocjitsu::consan_moi_impl {

bool sampled_access_supports_spill_backed_operand_recovery(const ConSanRequest &request,
                                                           const ConSanMoiCandidate &candidate,
                                                           rj_code_arch_t arch) {
  return request.moi_engine == ConSanMoiEngine::Sampled && candidate.is_native_lds() &&
         (consan_arch_is_rdna(arch) || consan_uses_gfx9_cdna_encoding(arch)) &&
         !candidate_requires_flat_address_materialization(candidate);
}

bool sampled_access_can_emit_spill_over_guest_operands(const ConSanMoiOperatingPoint &point,
                                                       const ConSanMoiCandidate &candidate) {
  return candidate.lowering.form && !candidate.lowering.form->destination_vgpr &&
         !candidate.is_direct_to_lds() && moi_owner_vgpr(point) &&
         !point.automatic_moi_private_epoch && !point.moi_persistent_sgprs.complete() &&
         !candidate_requires_flat_address_materialization(candidate);
}

bool sampled_access_can_plan_spill_over_guest_operands(const ConSanRequest &request,
                                                       const ConSanMoiOperatingPoint &point,
                                                       const ConSanMoiCandidate &candidate) {
  if (request.moi_engine != ConSanMoiEngine::Sampled)
    return false;
  if (sampled_access_can_emit_spill_over_guest_operands(point, candidate))
    return true;
  if (!candidate.lowering.form || candidate.lowering.form->destination_vgpr ||
      candidate.is_direct_to_lds() || point.automatic_moi_private_epoch ||
      point.moi_persistent_sgprs.complete() ||
      candidate_requires_flat_address_materialization(candidate)) {
    return false;
  }
  // Automatic persistent-state placement runs after the first resource pass.
  // Admit the overlap provisionally when entry-persistent owner state will be
  // resolved later; emission still requires a concrete persistent owner VGPR.
  return moi_initializes_owner_epoch(request, point) || request.moi_track_atomics ||
         request.moi_track_barriers || request.moi_runtime_sample_stride > 1u;
}

uint16_t direct_sampled_scratch_count(const ConSanRequest &request,
                                      const ConSanMoiOperatingPoint &point,
                                      const ConSanMoiCandidate &candidate, rj_code_arch_t arch) {
  const uint16_t base_scratch_count = request.moi_sampled_check ? 7u : 5u;
  const uint16_t two_address_replay_count = consan_uses_gfx12_cdna_execution(arch) &&
                                                    candidate.is_native_two_range() &&
                                                    candidate.encoded_offset_scale_bytes() > 8u
                                                ? 1u
                                                : 0u;
  return static_cast<uint16_t>(
      base_scratch_count + 1u +
      (flat_access_address_scratch_count(candidate) != 0u
           ? flat_access_address_scratch_count(candidate)
       : candidate.is_direct_to_lds() || moi_load_clobbers_address(candidate) ||
               moi_access_requires_high_bank_address_capture(candidate, arch)
           ? 1u
           : 0u) +
      (point.automatic_moi_private_epoch
           ? (request.moi_owner_source == ConSanMoiOwnerSource::WorkitemId ? 2u : 1u)
       : point.moi_persistent_sgprs.complete() ? 2u
                                               : 0u) +
      two_address_replay_count);
}

uint16_t sampled_spill_backed_scratch_count(const ConSanRequest &request,
                                            const ConSanMoiOperatingPoint &point,
                                            const ConSanMoiCandidate &candidate,
                                            rj_code_arch_t arch) {
  return direct_sampled_scratch_count(request, point, candidate, arch);
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
