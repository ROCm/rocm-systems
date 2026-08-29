// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi_candidate_projection.h"

#include <map>
#include <ranges>

namespace rocjitsu::consan_detail {
namespace {

[[nodiscard]] ConSanMoiCandidate make_moi_candidate(const ConSanAccessInventorySite &access,
                                                    ConSanProbeIntentId intent_id) {
  ConSanMoiCandidate candidate;
  static_cast<ConSanAccessInventorySite &>(candidate) = access;
  candidate.intent_ids.push_back(intent_id);
  if (access.execution_owner_descriptor_file_offsets.size() == 1u) {
    candidate.kernel_descriptor_file_offset =
        access.execution_owner_descriptor_file_offsets.front();
  }
  return candidate;
}

} // namespace

std::vector<ConSanMoiCandidate> build_moi_candidates(const ProgramInventory &inventory,
                                                     const ConSanObservationPlan &plan,
                                                     std::vector<std::string> &errors) {
  std::vector<ConSanMoiCandidate> candidates;
  std::map<uint64_t, size_t> candidate_index_by_offset;
  for (const ConSanProbeIntent &intent : plan.probe_intents) {
    if (intent.kind != ConSanProbeIntentKind::AccessRecord &&
        intent.kind != ConSanProbeIntentKind::SampledAccess &&
        intent.kind != ConSanProbeIntentKind::ExactShadowAccess) {
      continue;
    }
    const auto access = std::ranges::find_if(inventory.access_sites(), [&](const auto &candidate) {
      return candidate.physical_id == intent.physical_site;
    });
    if (access == inventory.access_sites().end()) {
      errors.emplace_back("ConSan MOI access intent lost its inventory site");
      return {};
    }
    if (!access->lowering.normalized() || !access->lowering.replay_guest_access.available()) {
      errors.emplace_back("ConSan MOI access intent was not backed by a replay lowering form");
      return {};
    }
    const auto [position, inserted] = candidate_index_by_offset.try_emplace(
        access->physical_id.original_text_offset, candidates.size());
    if (inserted) {
      candidates.push_back(make_moi_candidate(*access, intent.id));
    } else {
      ConSanMoiCandidate &candidate = candidates[position->second];
      if (candidate.physical_id != access->physical_id) {
        errors.emplace_back("ConSan MOI coalesced distinct physical access identities");
        return {};
      }
      candidate.intent_ids.push_back(intent.id);
    }
  }
  return candidates;
}

std::vector<const ConSanMoiCandidate *>
make_moi_candidate_pointer_view(std::span<const ConSanMoiCandidate> admitted) {
  std::vector<const ConSanMoiCandidate *> selected;
  selected.reserve(admitted.size());
  for (const ConSanMoiCandidate &candidate : admitted)
    selected.push_back(&candidate);
  return selected;
}

} // namespace rocjitsu::consan_detail
