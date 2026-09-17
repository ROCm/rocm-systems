// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_candidate_projection.h"

#include <map>
#include <ranges>

namespace rocjitsu::consan::detail {
namespace {

[[nodiscard]] Candidate make_candidate(const ProgramSite &access, ProbeIntentId intent_id) {
  Candidate candidate(access);
  candidate.intent_ids.push_back(intent_id);
  return candidate;
}

} // namespace

std::vector<Candidate> build_candidates(const ProgramInventory &inventory,
                                        const ObservationPlan &plan,
                                        std::vector<std::string> &errors) {
  std::vector<Candidate> candidates;
  std::map<uint64_t, size_t> candidate_index_by_offset;
  for (const ProbeIntent &intent : plan.probe_intents) {
    if (intent.kind != ProbeIntentKind::Access) {
      continue;
    }
    const ProgramSite *access = inventory.program_site(intent.source_site);
    if (access == nullptr || !access->has_access() || access->physical_id != intent.physical_site) {
      errors.emplace_back("ConSan access intent lost its inventory site");
      return {};
    }
    if (!access->lowering.normalized() || !access->lowering.replay_guest_access.available()) {
      errors.emplace_back("ConSan access intent was not backed by a replay lowering form");
      return {};
    }
    const auto [position, inserted] = candidate_index_by_offset.try_emplace(
        access->physical_id.original_text_offset, candidates.size());
    if (inserted) {
      candidates.push_back(make_candidate(*access, intent.id));
    } else {
      Candidate &candidate = candidates[position->second];
      if (candidate.site().physical_id != access->physical_id) {
        errors.emplace_back("ConSan coalesced distinct physical access identities");
        return {};
      }
      candidate.intent_ids.push_back(intent.id);
    }
  }
  return candidates;
}

std::vector<const Candidate *> make_candidate_pointer_view(std::span<const Candidate> admitted) {
  std::vector<const Candidate *> selected;
  selected.reserve(admitted.size());
  for (const Candidate &candidate : admitted)
    selected.push_back(&candidate);
  return selected;
}

} // namespace rocjitsu::consan::detail
