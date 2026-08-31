// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"

#include <array>
#include <optional>
#include <string>
#include <utility>

namespace rocjitsu {

/// Publish one synthetic lowering transaction for a policy-oriented test.
///
/// Production has no outcome-only mutation API: every finalized intent is
/// owned by a complete `ConSanCommittedLowering`. Tests that do not exercise
/// machine-code geometry use this adapter to create the smallest valid
/// transaction, including the runtime attribution required by access intents.
[[nodiscard]] inline bool publish_test_lowering_outcome(
    ConSanCoverageLedger &ledger, ConSanProbeIntentId id, ConSanLoweringOutcomeKind outcome,
    std::string detail = {},
    std::optional<ConSanRegisterPlanReason> resource_rejection_reason = std::nullopt) {
  const ConSanObservationPlan &plan = ledger.observation_plan();
  const ConSanProbeIntent *intent = plan.intent(id);
  if (intent == nullptr)
    return false;
  if (outcome == ConSanLoweringOutcomeKind::Pending) {
    const ConSanIntentCoverageEntry *entry = ledger.intent_entry(id);
    return entry != nullptr && entry->lowering == ConSanLoweringOutcomeKind::Pending &&
           detail.empty() && !resource_rejection_reason;
  }

  std::array<ConSanCommittedLoweringLocation, 1> locations = {
      ConSanCommittedLoweringLocation{
          .original_site = intent->physical_site,
          .emitted_text_offset = intent->physical_site.original_text_offset,
          .emitted_size = 4u,
          .relocated_guest_text_offset = std::nullopt,
      },
  };
  const bool instrumented = outcome == ConSanLoweringOutcomeKind::Instrumented;
  ConSanRuntimeStaticMapping runtime_mapping;
  if (instrumented) {
    ConSanStaticAccessAttribution attribution{
        .intent_ids = {id},
        .original_site = intent->physical_site,
        .original_semantic_sites = intent->covered_semantic_sites,
        .execution_owner_descriptor_file_offsets = {},
        .owner_provenance_complete = false,
    };
    switch (intent->kind) {
    case ConSanProbeIntentKind::AccessRecord:
      runtime_mapping.record_replay_accesses.push_back({.access = std::move(attribution)});
      break;
    case ConSanProbeIntentKind::SampledAccess:
      runtime_mapping.sampled_accesses.push_back({
          .access = std::move(attribution),
          .first_slot = 0u,
          .range_count = 1u,
          .bank_count = 1u,
          .emitted_probe_text_offset = locations.front().emitted_text_offset,
          .relocated_guest_text_offset = std::nullopt,
          .scratch_vgpr = std::nullopt,
      });
      break;
    case ConSanProbeIntentKind::ExactShadowAccess:
      attribution.execution_owner_descriptor_file_offsets = {0u};
      attribution.owner_provenance_complete = true;
      runtime_mapping.inline_compact_accesses.push_back({
          .access = std::move(attribution),
          .token = 1u,
      });
      break;
    default:
      break;
    }
  }

  const std::array intent_ids = {id};
  const std::span<const ConSanCommittedLoweringLocation> committed_locations =
      instrumented ? std::span<const ConSanCommittedLoweringLocation>(locations)
                   : std::span<const ConSanCommittedLoweringLocation>{};
  auto commit = make_consan_committed_lowering(plan, intent_ids, committed_locations, outcome,
                                               std::move(detail), std::move(runtime_mapping),
                                               resource_rejection_reason);
  return commit && ledger.publish_lowering_commit(std::move(*commit));
}

} // namespace rocjitsu
