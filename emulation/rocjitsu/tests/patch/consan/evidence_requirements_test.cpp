// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"

#include <set>

namespace rocjitsu {
namespace {

template <typename Enum, size_t N, typename NameFunction>
void expect_evidence_enum_contract(const std::array<Enum, N> &values, Enum count, NameFunction name,
                                   std::string_view invalid_name) {
  EXPECT_EQ(values.size(), static_cast<size_t>(count));
  std::set<std::string_view> names;
  for (size_t ordinal = 0; ordinal < values.size(); ++ordinal) {
    EXPECT_EQ(static_cast<size_t>(values[ordinal]), ordinal);
    EXPECT_NE(name(values[ordinal]), invalid_name);
    EXPECT_TRUE(names.insert(name(values[ordinal])).second);
  }
  EXPECT_EQ(name(count), invalid_name);
  EXPECT_EQ(name(static_cast<Enum>(255)), invalid_name);
}

/// Builder for a valid policy-level Record/Replay plan whose intent mix can be
/// varied without constructing an ELF or any target-specific operand data.
class RecordReplayObservationPlanBuilder {
public:
  RecordReplayObservationPlanBuilder() {
    plan_.engine = ConSanCapabilityEngine::RecordReplay;
    physical_template_.code_object = make_consan_code_object_id(std::array<uint8_t, 4>{1, 2, 3, 4});
  }

  RecordReplayObservationPlanBuilder &add_access(size_t range_count) {
    add_intent(ConSanProbeIntentKind::AccessRecord, ConSanSemanticSiteDomain::Access, range_count,
               /*associated=*/false);
    return *this;
  }

  RecordReplayObservationPlanBuilder &add_barrier() {
    add_intent(ConSanProbeIntentKind::BarrierRecord, ConSanSemanticSiteDomain::SynchronizationEvent,
               1, /*associated=*/false);
    return *this;
  }

  RecordReplayObservationPlanBuilder &add_atomic() {
    add_intent(ConSanProbeIntentKind::AtomicRecord, ConSanSemanticSiteDomain::SynchronizationEvent,
               1, /*associated=*/true);
    return *this;
  }

  RecordReplayObservationPlanBuilder &add_fence() {
    add_intent(ConSanProbeIntentKind::FenceRecord, ConSanSemanticSiteDomain::SynchronizationEvent,
               1, /*associated=*/true);
    return *this;
  }

  [[nodiscard]] ConSanObservationPlan build() const { return plan_; }

private:
  void add_intent(ConSanProbeIntentKind kind, ConSanSemanticSiteDomain domain,
                  size_t semantic_count, bool associated) {
    PhysicalSiteId physical = physical_template_;
    physical.original_text_offset = 16u + plan_.probe_intents.size() * 16u;
    ConSanProbeIntent intent;
    intent.id = {static_cast<uint32_t>(plan_.probe_intents.size())};
    intent.engine = plan_.engine;
    intent.physical_site = physical;
    intent.kind = kind;
    intent.position = kind == ConSanProbeIntentKind::AccessRecord ? ConSanProbePosition::Before
                                                                  : ConSanProbePosition::After;
    intent.lane_mask = ConSanLaneMaskPolicy::ActiveExecutionMask;
    intent.requirement = ConSanProbeRequirement::Required;
    if (associated)
      intent.synchronization_association = {"sequence-" + std::to_string(intent.id.value)};
    for (size_t ordinal = 0; ordinal < semantic_count; ++ordinal) {
      intent.covered_semantic_sites.push_back({
          .physical = physical,
          .domain = domain,
          .range_ordinal = static_cast<uint32_t>(ordinal),
      });
    }
    plan_.probe_intents.push_back(std::move(intent));
  }

  ConSanObservationPlan plan_;
  PhysicalSiteId physical_template_;
};

TEST(ConSanEvidenceRequirements, EnumContractsAreExhaustiveNamedAndRejectInvalidValues) {
  expect_evidence_enum_contract(kConSanEvidenceSchemas, ConSanEvidenceSchema::Count,
                                consan_evidence_schema_name, "invalid-evidence-schema");
  expect_evidence_enum_contract(kConSanEvidenceBoundednessValues, ConSanEvidenceBoundedness::Count,
                                consan_evidence_boundedness_name, "invalid-evidence-boundedness");
  expect_evidence_enum_contract(kConSanEvidenceLossSeverities, ConSanEvidenceLossSeverity::Count,
                                consan_evidence_loss_severity_name,
                                "invalid-evidence-loss-severity");
  expect_evidence_enum_contract(
      kConSanEvidenceRequirementReasons, ConSanEvidenceRequirementReason::Count,
      consan_evidence_requirement_reason_name, "invalid-evidence-requirement-reason");

  constexpr std::array outcomes = {
      ConSanMoiAutoReportPlanOutcome::Complete,
      ConSanMoiAutoReportPlanOutcome::InsufficientReportCapacity,
      ConSanMoiAutoReportPlanOutcome::Overflow,
  };
  expect_evidence_enum_contract(outcomes, ConSanMoiAutoReportPlanOutcome::Count,
                                consan_moi_auto_report_plan_outcome_name,
                                "invalid_auto_report_plan_outcome");
  constexpr std::array reasons = {
      ConSanMoiAutoReportPlanReason::None,
      ConSanMoiAutoReportPlanReason::PerBufferCeiling,
      ConSanMoiAutoReportPlanReason::AbiCapacityOverflow,
      ConSanMoiAutoReportPlanReason::AbiGeometryCapacityOverflow,
      ConSanMoiAutoReportPlanReason::ByteSizeOverflow,
  };
  expect_evidence_enum_contract(reasons, ConSanMoiAutoReportPlanReason::Count,
                                consan_moi_auto_report_plan_reason_name,
                                "invalid_auto_report_plan_reason");
}

TEST(ConSanEvidenceRequirements, EmptyRecordReplayPlanProducesOneAddressFreeHeaderContract) {
  ConSanObservationPlan plan;
  plan.engine = ConSanCapabilityEngine::RecordReplay;
  ASSERT_TRUE(plan.valid());

  const ConSanRecordReplayEvidenceRequirements requirements =
      plan_consan_record_replay_evidence(plan);
  ASSERT_TRUE(requirements.well_formed());
  ASSERT_TRUE(requirements.complete());
  EXPECT_EQ(requirements.schema, ConSanEvidenceSchema::RecordReplay);
  EXPECT_EQ(requirements.boundedness, ConSanEvidenceBoundedness::BoundedFirstLight);
  EXPECT_EQ(requirements.loss_severity, ConSanEvidenceLossSeverity::InvalidatesCompleteness);
  EXPECT_EQ(requirements.delivery_scope, ConSanRuntimeResourceScope::Executable);
  EXPECT_EQ(requirements.sizing_inventory.access_range_count, 0u);
  EXPECT_EQ(requirements.abi_plan.required_bytes, sizeof(ConSanMoiReportHeader));
  EXPECT_EQ(requirements.runtime_requirements.minimum_report_allocation_bytes,
            requirements.abi_plan.required_bytes);
  EXPECT_TRUE(requirements.runtime_requirements.host_device_visible_memory);
  EXPECT_TRUE(requirements.runtime_requirements.host_device_coherent_memory);
  EXPECT_TRUE(requirements.runtime_requirements.device_atomic_publication);
  EXPECT_TRUE(requirements.runtime_requirements.executable_binding);
  EXPECT_FALSE(requirements.runtime_requirements.max_workgroup_lds_bytes);
  EXPECT_FALSE(requirements.runtime_requirements.dispatch_segment_binding);
}

TEST(ConSanEvidenceRequirements, IntentKindsMapToIndependentRecordReplaySizingDimensions) {
  const ConSanObservationPlan plan = RecordReplayObservationPlanBuilder()
                                         .add_access(2)
                                         .add_access(1)
                                         .add_barrier()
                                         .add_atomic()
                                         .add_fence()
                                         .build();
  ASSERT_TRUE(plan.valid());
  const auto requirements = plan_consan_record_replay_evidence(plan);
  ASSERT_TRUE(requirements.complete());
  const auto &sizing = requirements.sizing_inventory;
  EXPECT_EQ(sizing.access_range_count, 3u);
  EXPECT_EQ(sizing.barrier_event_count, kConSanMoiRecordReplayDynamicEventHeadroom);
  EXPECT_EQ(sizing.atomic_event_count, kConSanMoiRecordReplayDynamicLaneEventHeadroom);
  EXPECT_EQ(sizing.fence_event_count, kConSanMoiRecordReplayDynamicLaneEventHeadroom);
  EXPECT_EQ(sizing.record_replay_access_dispatch_bank_count,
            kConSanMoiRecordReplayMaximumDispatchBankCount);
  EXPECT_EQ(sizing.record_replay_access_owner_bank_count,
            kConSanMoiRecordReplayMaximumOwnerBankCount);
  EXPECT_EQ(sizing.record_replay_address_group_headroom,
            kConSanMoiRecordReplayMaximumAddressGroupsPerWave);
  EXPECT_EQ(sizing.diagnostic_count, 3u * kConSanMoiRecordReplayMaximumDispatchBankCount *
                                         kConSanMoiRecordReplayMaximumOwnerBankCount *
                                         kConSanMoiRecordReplayMaximumAddressGroupsPerWave);
}

TEST(ConSanEvidenceRequirements, ExpertAccessLimitCountsPhysicalIntentsButNotSyncIntents) {
  const ConSanObservationPlan plan = RecordReplayObservationPlanBuilder()
                                         .add_access(2)
                                         .add_access(3)
                                         .add_barrier()
                                         .add_atomic()
                                         .add_fence()
                                         .build();
  const auto requirements =
      plan_consan_record_replay_evidence(plan, {.maximum_access_probe_count = 1u});
  ASSERT_TRUE(requirements.complete());
  EXPECT_EQ(requirements.sizing_inventory.access_range_count, 2u);
  EXPECT_EQ(requirements.sizing_inventory.barrier_event_count,
            kConSanMoiRecordReplayDynamicEventHeadroom);
  EXPECT_EQ(requirements.sizing_inventory.atomic_event_count,
            kConSanMoiRecordReplayDynamicLaneEventHeadroom);
  EXPECT_EQ(requirements.sizing_inventory.fence_event_count,
            kConSanMoiRecordReplayDynamicLaneEventHeadroom);

  ConSanOptions options;
  options.max_patches = 7;
  options.max_patches_is_expert_limit = false;
  EXPECT_EQ(make_consan_record_replay_capacity_policy(options, 4096),
            (ConSanRecordReplayCapacityPolicy{.caller_ceiling_bytes = 4096,
                                              .maximum_access_probe_count = std::nullopt}));
  options.max_patches_is_expert_limit = true;
  EXPECT_EQ(make_consan_record_replay_capacity_policy(options, 4096),
            (ConSanRecordReplayCapacityPolicy{.caller_ceiling_bytes = 4096,
                                              .maximum_access_probe_count = 7}));
}

TEST(ConSanEvidenceRequirements, InvalidAndWrongEnginePlansHaveDistinctTypedReasons) {
  ConSanObservationPlan invalid;
  EXPECT_FALSE(invalid.valid());
  const auto invalid_requirements = plan_consan_record_replay_evidence(invalid);
  EXPECT_EQ(invalid_requirements.reason, ConSanEvidenceRequirementReason::InvalidObservationPlan);
  EXPECT_FALSE(invalid_requirements.well_formed());
  EXPECT_FALSE(invalid_requirements.complete());

  ConSanObservationPlan sampled;
  sampled.engine = ConSanCapabilityEngine::Sampled;
  ASSERT_TRUE(sampled.valid());
  const auto wrong_engine = plan_consan_record_replay_evidence(sampled);
  EXPECT_EQ(wrong_engine.reason, ConSanEvidenceRequirementReason::WrongEngine);
  EXPECT_FALSE(wrong_engine.well_formed());
}

TEST(ConSanEvidenceRequirements, ForeignIntentKindsAndPayloadDomainsFailClosed) {
  ConSanObservationPlan foreign = RecordReplayObservationPlanBuilder().add_access(1).build();
  foreign.probe_intents.front().kind = ConSanProbeIntentKind::SampledAccess;
  ASSERT_TRUE(foreign.valid());
  EXPECT_EQ(plan_consan_record_replay_evidence(foreign).reason,
            ConSanEvidenceRequirementReason::UnexpectedIntentKind);

  ConSanObservationPlan malformed = RecordReplayObservationPlanBuilder().add_access(1).build();
  malformed.probe_intents.front().covered_semantic_sites.front().domain =
      ConSanSemanticSiteDomain::SynchronizationEvent;
  ASSERT_TRUE(malformed.valid());
  EXPECT_EQ(plan_consan_record_replay_evidence(malformed).reason,
            ConSanEvidenceRequirementReason::InvalidIntentPayload);
}

TEST(ConSanEvidenceRequirements, RuntimeContractRequiresEveryPublicationAndLifetimeFact) {
  const auto requirements = plan_consan_record_replay_evidence(
      RecordReplayObservationPlanBuilder().add_access(1).build());
  ASSERT_TRUE(requirements.complete());
  RuntimeCapabilities capabilities{
      .backend = ConSanRuntimeBackend::PhysicalHsa,
      .host_device_visible_memory = true,
      .host_device_coherent_memory = true,
      .device_atomic_publication = true,
      .max_report_allocation_bytes = requirements.abi_plan.required_bytes,
      .max_workgroup_lds_bytes = std::nullopt,
      .executable_binding = true,
      .dispatch_segment_binding = false,
  };
  EXPECT_EQ(validate_runtime_capabilities(capabilities, requirements.runtime_requirements),
            ConSanContractIssue::None);
  capabilities.host_device_coherent_memory = false;
  EXPECT_EQ(validate_runtime_capabilities(capabilities, requirements.runtime_requirements),
            ConSanContractIssue::MissingCoherentMemory);
  capabilities.host_device_coherent_memory = true;
  capabilities.device_atomic_publication = false;
  EXPECT_EQ(validate_runtime_capabilities(capabilities, requirements.runtime_requirements),
            ConSanContractIssue::MissingAtomicPublication);
  capabilities.device_atomic_publication = true;
  capabilities.executable_binding = false;
  EXPECT_EQ(validate_runtime_capabilities(capabilities, requirements.runtime_requirements),
            ConSanContractIssue::MissingExecutableBinding);
}

TEST(ConSanEvidenceRequirements, CapacityFailureRemainsAWellFormedAddressFreeRequirement) {
  const ConSanObservationPlan plan = RecordReplayObservationPlanBuilder().add_access(1).build();
  const auto requirements = plan_consan_record_replay_evidence(
      plan, {.caller_ceiling_bytes = sizeof(ConSanMoiReportHeader),
             .maximum_access_probe_count = std::nullopt});
  EXPECT_TRUE(requirements.well_formed());
  EXPECT_FALSE(requirements.complete());
  EXPECT_EQ(requirements.reason, ConSanEvidenceRequirementReason::None);
  EXPECT_EQ(requirements.abi_plan.outcome,
            ConSanMoiAutoReportPlanOutcome::InsufficientReportCapacity);
  EXPECT_EQ(requirements.abi_plan.reason, ConSanMoiAutoReportPlanReason::PerBufferCeiling);
  ASSERT_TRUE(requirements.runtime_requirements.minimum_report_allocation_bytes);
  EXPECT_GT(*requirements.runtime_requirements.minimum_report_allocation_bytes,
            sizeof(ConSanMoiReportHeader));
}

TEST(ConSanEvidenceRequirements, WellFormedRejectsEveryCrossTypeContractMismatch) {
  const auto good = plan_consan_record_replay_evidence(
      RecordReplayObservationPlanBuilder().add_access(1).build());
  ASSERT_TRUE(good.well_formed());

  const auto expect_rejected = [&](auto mutate) {
    ConSanRecordReplayEvidenceRequirements broken = good;
    mutate(broken);
    EXPECT_FALSE(broken.well_formed());
    EXPECT_FALSE(broken.complete());
  };
  expect_rejected([](auto &value) { value.schema = ConSanEvidenceSchema::Sampled; });
  expect_rejected(
      [](auto &value) { value.boundedness = ConSanEvidenceBoundedness::ExactOnDevice; });
  expect_rejected([](auto &value) { value.loss_severity = ConSanEvidenceLossSeverity::Optional; });
  expect_rejected([](auto &value) { value.delivery_scope = ConSanRuntimeResourceScope::Dispatch; });
  expect_rejected(
      [](auto &value) { value.runtime_requirements.host_device_visible_memory = false; });
  expect_rejected(
      [](auto &value) { value.runtime_requirements.host_device_coherent_memory = false; });
  expect_rejected(
      [](auto &value) { value.runtime_requirements.device_atomic_publication = false; });
  expect_rejected([](auto &value) { value.runtime_requirements.executable_binding = false; });
  expect_rejected([](auto &value) { value.runtime_requirements.max_workgroup_lds_bytes = true; });
  expect_rejected([](auto &value) { value.runtime_requirements.dispatch_segment_binding = true; });
  expect_rejected(
      [](auto &value) { ++*value.runtime_requirements.minimum_report_allocation_bytes; });
  expect_rejected([](auto &value) { value.sizing_inventory.engine = ConSanMoiEngine::Sampled; });
  expect_rejected([](auto &value) { value.abi_plan.engine = ConSanMoiEngine::Sampled; });
  expect_rejected(
      [](auto &value) { value.abi_plan.outcome = ConSanMoiAutoReportPlanOutcome::Count; });
  expect_rejected(
      [](auto &value) { value.abi_plan.reason = ConSanMoiAutoReportPlanReason::Count; });
  expect_rejected([](auto &value) { ++value.abi_plan.layout.required_bytes; });
}

TEST(ConSanEvidenceRequirements,
     LegacyInventoryAdapterConsumesOnlyTheObservationPlanForRecordReplay) {
  ConSanResult result;
  result.observation_plan = RecordReplayObservationPlanBuilder()
                                .add_access(2)
                                .add_barrier()
                                .add_atomic()
                                .add_fence()
                                .build();
  // Contradictory mutable telemetry proves that the compatibility entry point
  // no longer rescans resource plans, dispositions, patches, or object bytes.
  result.site_dispositions.resize(19);
  result.resource_plans.resize(23);
  result.patches.resize(29);
  ConSanOptions options;
  options.moi_engine = ConSanMoiEngine::RecordReplay;
  const ConSanMoiAutoReportInventory inventory =
      inventory_consan_moi_auto_report(result, options, {});
  EXPECT_EQ(inventory,
            plan_consan_record_replay_evidence(result.observation_plan).sizing_inventory);
  EXPECT_EQ(inventory.access_range_count, 2u);
  EXPECT_EQ(inventory.barrier_event_count, kConSanMoiRecordReplayDynamicEventHeadroom);
  EXPECT_EQ(inventory.atomic_event_count, kConSanMoiRecordReplayDynamicLaneEventHeadroom);
  EXPECT_EQ(inventory.fence_event_count, kConSanMoiRecordReplayDynamicLaneEventHeadroom);
}

TEST(ConSanEvidenceRequirements, PlanningIsDeterministicAndDoesNotMutateItsInputs) {
  const ConSanObservationPlan plan = RecordReplayObservationPlanBuilder()
                                         .add_access(2)
                                         .add_barrier()
                                         .add_atomic()
                                         .add_fence()
                                         .build();
  const ConSanObservationPlan before = plan;
  const ConSanRecordReplayCapacityPolicy policy{.caller_ceiling_bytes = 16u * 1024u * 1024u,
                                                .maximum_access_probe_count = 1};
  const auto first = plan_consan_record_replay_evidence(plan, policy);
  const auto second = plan_consan_record_replay_evidence(plan, policy);
  EXPECT_EQ(first, second);
  EXPECT_EQ(plan, before);
}

} // namespace
} // namespace rocjitsu
