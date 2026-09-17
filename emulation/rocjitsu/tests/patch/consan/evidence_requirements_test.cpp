// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"

#include <set>

namespace rocjitsu::consan {
namespace {

[[nodiscard]] const ObservationPlan &evidence_intents(const ObservationPlan &plan) { return plan; }

template <typename Values, typename Enum, typename NameFunction>
void expect_evidence_enum_contract(const Values &values, Enum count, NameFunction name,
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

/// Builder for valid mode-specific intent mixtures that do not need an
/// accompanying program inventory.
class EvidenceObservationPlanBuilder {
public:
  explicit EvidenceObservationPlanBuilder(Mode mode) {
    plan_.mode = mode;
    physical_template_.code_object = make_code_object_id(std::array<uint8_t, 4>{4, 3, 2, 1});
  }

  EvidenceObservationPlanBuilder &add(ProbeIntentKind kind, SemanticSiteDomain domain,
                                      size_t semantic_count = 1, bool associated = false) {
    PhysicalSiteId physical = physical_template_;
    physical.original_text_offset = 32u + plan_.probe_intents.size() * 16u;
    ProbeIntent intent;
    intent.id = {static_cast<uint32_t>(plan_.probe_intents.size())};
    intent.mode = plan_.mode;
    intent.source_site = {intent.id.value};
    intent.physical_site = physical;
    intent.kind = kind;
    intent.position = kind == ProbeIntentKind::AtomicAddressCapture ? ProbePosition::Before
                                                                    : ProbePosition::After;
    if (kind == ProbeIntentKind::AtomicAddressCapture) {
      AtomicLoweringForm &form = intent.atomic_lowering_form.emplace();
      form.kind = AtomicLoweringFormKind::GlobalVectorAddress;
      form.instruction_size = 8;
      form.value_width_bits = 32;
      form.value_register_count = 1;
      form.data_register_count = 1;
      form.address_vgpr_count = 2;
    }
    if (associated)
      intent.synchronization_association = {"evidence-sequence-" + std::to_string(intent.id.value)};
    for (size_t ordinal = 0; ordinal < semantic_count; ++ordinal) {
      intent.covered_semantic_sites.push_back({
          .physical = physical,
          .domain = domain,
          .range_ordinal = static_cast<uint32_t>(ordinal),
      });
    }
    plan_.probe_intents.push_back(std::move(intent));
    return *this;
  }

  [[nodiscard]] ObservationPlan build() const { return plan_; }

private:
  ObservationPlan plan_;
  PhysicalSiteId physical_template_;
};

TEST(EvidenceRequirements, EnumContractsAreExhaustiveNamedAndRejectInvalidValues) {
  constexpr std::array outcomes = {
      AutoReportPlanOutcome::Complete,
      AutoReportPlanOutcome::InsufficientReportCapacity,
      AutoReportPlanOutcome::Overflow,
  };
  expect_evidence_enum_contract(outcomes, AutoReportPlanOutcome::Count,
                                auto_report_plan_outcome_name, "invalid_auto_report_plan_outcome");
  constexpr std::array reasons = {
      AutoReportPlanReason::None,
      AutoReportPlanReason::PerBufferCeiling,
      AutoReportPlanReason::AbiCapacityOverflow,
      AutoReportPlanReason::ByteSizeOverflow,
  };
  expect_evidence_enum_contract(reasons, AutoReportPlanReason::Count, auto_report_plan_reason_name,
                                "invalid_auto_report_plan_reason");
}

TEST(EvidenceRequirements, ObservationIntentsExposeEveryModeReportRole) {
  const auto verify = [](const ObservationPlan &observation,
                         std::initializer_list<EvidenceIntentKind> kinds,
                         std::initializer_list<uint64_t> element_counts) {
    ASSERT_TRUE(observation.valid());
    ASSERT_EQ(observation.probe_intents.size(), kinds.size());
    ASSERT_EQ(observation.probe_intents.size(), element_counts.size());
    size_t index = 0;
    for (EvidenceIntentKind kind : kinds) {
      EXPECT_EQ(evidence_intent_kind(observation.mode, observation.probe_intents[index].kind),
                kind);
      ++index;
    }
    index = 0;
    for (uint64_t element_count : element_counts)
      EXPECT_EQ(evidence_element_count(observation.mode, observation.probe_intents[index++]),
                element_count);
  };

  verify(
      EvidenceObservationPlanBuilder(Mode::Default)
          .add(ProbeIntentKind::Access, SemanticSiteDomain::Access, 3)
          .add(ProbeIntentKind::BarrierEpoch, SemanticSiteDomain::SynchronizationEvent, 2)
          .add(ProbeIntentKind::AtomicAddressCapture, SemanticSiteDomain::SynchronizationEvent, 1,
               true)
          .add(ProbeIntentKind::AtomicOrdering, SemanticSiteDomain::SynchronizationEvent, 1, true)
          .build(),
      {EvidenceIntentKind::Access, EvidenceIntentKind::Barrier, EvidenceIntentKind::AddressCapture,
       EvidenceIntentKind::Atomic},
      {3, 2, 0, 1});
  verify(EvidenceObservationPlanBuilder(Mode::SuperCollider)
             .add(ProbeIntentKind::RedundantAccessObservation, SemanticSiteDomain::Access, 2)
             .build(),
         {EvidenceIntentKind::StickyMarker}, {1});
}

TEST(EvidenceRequirements, EvidencePlanningRejectsMalformedOrCrossModeObservations) {
  const ObservationPlan observation =
      EvidenceObservationPlanBuilder(Mode::Default)
          .add(ProbeIntentKind::Access, SemanticSiteDomain::Access, 2)
          .build();
  ASSERT_TRUE(observation.valid());

  ObservationPlan malformed = observation;
  malformed.probe_intents.front().id = {1};
  EXPECT_EQ(plan_evidence(malformed).reason, EvidenceRequirementReason::InvalidObservationPlan);
  malformed = observation;
  malformed.probe_intents.front().source_site = {};
  EXPECT_EQ(plan_evidence(malformed).reason, EvidenceRequirementReason::InvalidObservationPlan);
  malformed = observation;
  malformed.probe_intents.front().covered_semantic_sites.front().domain =
      SemanticSiteDomain::SynchronizationEvent;
  EXPECT_EQ(plan_evidence(malformed).reason, EvidenceRequirementReason::InvalidIntentPayload);
  malformed = observation;
  malformed.mode = Mode::SuperCollider;
  malformed.probe_intents.front().mode = malformed.mode;
  EXPECT_EQ(plan_evidence(malformed).reason, EvidenceRequirementReason::WrongMode);

  ObservationPlan foreign = observation;
  foreign.probe_intents.front().kind = ProbeIntentKind::RedundantAccessObservation;
  ASSERT_TRUE(foreign.valid());
  EXPECT_EQ(plan_evidence(foreign).reason, EvidenceRequirementReason::UnexpectedIntentKind);
}

TEST(EvidenceRequirements, ObservationPlansAreTheOnlyEvidenceSizingInputs) {
  const ObservationPlan sampled =
      EvidenceObservationPlanBuilder(Mode::Default)
          .add(ProbeIntentKind::Access, SemanticSiteDomain::Access, 2)
          .add(ProbeIntentKind::AtomicOrdering, SemanticSiteDomain::SynchronizationEvent, 1, true)
          .build();
  EXPECT_TRUE(plan_evidence(sampled).complete());

  const ObservationPlan supercollider =
      EvidenceObservationPlanBuilder(Mode::SuperCollider)
          .add(ProbeIntentKind::RedundantAccessObservation, SemanticSiteDomain::Access)
          .build();
  EXPECT_TRUE(plan_supercollider_evidence(supercollider).complete());

  ObservationPlan malformed = sampled;
  malformed.probe_intents.front().covered_semantic_sites.front().domain =
      SemanticSiteDomain::SynchronizationEvent;
  EXPECT_EQ(plan_evidence(malformed).reason, EvidenceRequirementReason::InvalidIntentPayload);
}

TEST(EvidenceRequirements, EmptyPlanProducesOneAddressFreeHeaderContract) {
  ObservationPlan plan;
  plan.mode = Mode::Default;
  ASSERT_TRUE(plan.valid());

  const ReportRequirements requirements = plan_evidence(plan);
  ASSERT_TRUE(requirements.well_formed());
  ASSERT_TRUE(requirements.complete());
  EXPECT_EQ(requirements.sizing_inventory.access_range_count, 0u);
  EXPECT_EQ(requirements.abi_plan.required_bytes, sizeof(ReportHeader));
  EXPECT_EQ(requirements.runtime_requirements.minimum_report_allocation_bytes,
            requirements.abi_plan.required_bytes);
  EXPECT_TRUE(requirements.runtime_requirements.host_device_visible_memory);
  EXPECT_TRUE(requirements.runtime_requirements.host_device_coherent_memory);
  EXPECT_TRUE(requirements.runtime_requirements.device_atomic_publication);
  EXPECT_TRUE(requirements.runtime_requirements.executable_binding);
  EXPECT_FALSE(requirements.runtime_requirements.max_workgroup_lds_bytes);
  EXPECT_FALSE(requirements.runtime_requirements.dispatch_segment_binding);
}

TEST(EvidenceRequirements, BindingRequiresSemanticObservationsAndACompleteAbiPlan) {
  AutoReportInventory inventory;
  EXPECT_FALSE(inventory.has_semantic_observations());
  for (const AutoReportCapacityMember member : {
           &AutoReportInventory::access_range_count,
           &AutoReportInventory::barrier_event_count,
           &AutoReportInventory::atomic_event_count,
       }) {
    AutoReportInventory observed;
    observed.*member = 1u;
    EXPECT_TRUE(observed.has_semantic_observations());
  }

  ObservationPlan empty_plan;
  empty_plan.mode = Mode::Default;
  const auto empty = plan_evidence(evidence_intents(empty_plan));
  ASSERT_TRUE(empty.complete());
  EXPECT_FALSE(empty.sizing_inventory.has_semantic_observations());
  EXPECT_FALSE(empty.requires_binding());
  const auto observed = plan_evidence(evidence_intents(
      EvidenceObservationPlanBuilder(Mode::Default)
          .add(ProbeIntentKind::BarrierEpoch, SemanticSiteDomain::SynchronizationEvent)
          .build()));
  ASSERT_TRUE(observed.complete());
  EXPECT_TRUE(observed.sizing_inventory.has_semantic_observations());
  EXPECT_TRUE(observed.requires_binding());

  EXPECT_FALSE(ReportRequirements{}.sizing_inventory.has_semantic_observations());
  EXPECT_FALSE(ReportRequirements{}.requires_binding());
}

TEST(EvidenceRequirements, RuntimeContractRequiresEveryPublicationAndLifetimeFact) {
  const auto requirements =
      plan_evidence(evidence_intents(EvidenceObservationPlanBuilder(Mode::Default)
                                         .add(ProbeIntentKind::Access, SemanticSiteDomain::Access)
                                         .build()));
  ASSERT_TRUE(requirements.complete());
  RuntimeCapabilities capabilities{
      .backend = RuntimeBackend::PhysicalHsa,
      .host_device_visible_memory = true,
      .host_device_coherent_memory = true,
      .device_atomic_publication = true,
      .max_report_allocation_bytes = requirements.abi_plan.required_bytes,
      .max_workgroup_lds_bytes = std::nullopt,
      .executable_binding = true,
      .dispatch_segment_binding = false,
  };
  EXPECT_EQ(validate_runtime_capabilities(capabilities, requirements.runtime_requirements),
            ContractIssue::None);
  capabilities.host_device_coherent_memory = false;
  EXPECT_EQ(validate_runtime_capabilities(capabilities, requirements.runtime_requirements),
            ContractIssue::MissingCoherentMemory);
  capabilities.host_device_coherent_memory = true;
  capabilities.device_atomic_publication = false;
  EXPECT_EQ(validate_runtime_capabilities(capabilities, requirements.runtime_requirements),
            ContractIssue::MissingAtomicPublication);
  capabilities.device_atomic_publication = true;
  capabilities.executable_binding = false;
  EXPECT_EQ(validate_runtime_capabilities(capabilities, requirements.runtime_requirements),
            ContractIssue::MissingExecutableBinding);
}

TEST(EvidenceRequirements, CapacityFailureRemainsAWellFormedAddressFreeRequirement) {
  const ObservationPlan plan = EvidenceObservationPlanBuilder(Mode::Default)
                                   .add(ProbeIntentKind::Access, SemanticSiteDomain::Access)
                                   .build();
  const auto requirements =
      plan_evidence(evidence_intents(plan), {.caller_ceiling_bytes = sizeof(ReportHeader),
                                             .maximum_access_probe_count = std::nullopt});
  EXPECT_TRUE(requirements.well_formed());
  EXPECT_FALSE(requirements.complete());
  EXPECT_EQ(requirements.reason, EvidenceRequirementReason::None);
  EXPECT_EQ(requirements.abi_plan.outcome, AutoReportPlanOutcome::InsufficientReportCapacity);
  EXPECT_EQ(requirements.abi_plan.reason, AutoReportPlanReason::PerBufferCeiling);
  ASSERT_TRUE(requirements.runtime_requirements.minimum_report_allocation_bytes);
  EXPECT_GT(*requirements.runtime_requirements.minimum_report_allocation_bytes,
            sizeof(ReportHeader));
}

TEST(EvidenceRequirements, CapacityPoliciesHaveNeutralAddressFreeDefaults) {
  EXPECT_EQ(CapacityPolicy{}, (CapacityPolicy{.caller_ceiling_bytes = 0,
                                              .maximum_access_probe_count = std::nullopt}));
}

TEST(EvidenceRequirements, IntentKindsMapToIndependentTypedCapacities) {
  const ObservationPlan plan =
      EvidenceObservationPlanBuilder(Mode::Default)
          .add(ProbeIntentKind::Access, SemanticSiteDomain::Access, 2)
          .add(ProbeIntentKind::Access, SemanticSiteDomain::Access, 1)
          .add(ProbeIntentKind::BarrierEpoch, SemanticSiteDomain::SynchronizationEvent, 3)
          .add(ProbeIntentKind::AtomicAddressCapture, SemanticSiteDomain::SynchronizationEvent, 1,
               true)
          .add(ProbeIntentKind::AtomicOrdering, SemanticSiteDomain::SynchronizationEvent, 1, true)
          .build();
  ASSERT_TRUE(plan.valid());
  const ReportRequirements requirements = plan_evidence(evidence_intents(plan));
  ASSERT_TRUE(requirements.complete());
  EXPECT_EQ(requirements.sizing_inventory.access_range_count, 3u);
  EXPECT_EQ(requirements.sizing_inventory.barrier_event_count, 3u);
  EXPECT_EQ(requirements.sizing_inventory.atomic_event_count, 1u);
  EXPECT_EQ(requirements.sizing_inventory.range_bank_count, 24u);
  EXPECT_EQ(requirements.sizing_inventory.sync_slot_count, 25u);
  EXPECT_EQ(requirements.sizing_inventory.watchpoint_count, 25u);
  EXPECT_TRUE(requirements.runtime_requirements.host_device_visible_memory);
  EXPECT_TRUE(requirements.runtime_requirements.host_device_coherent_memory);
  EXPECT_TRUE(requirements.runtime_requirements.device_atomic_publication);
  EXPECT_TRUE(requirements.runtime_requirements.executable_binding);
  EXPECT_EQ(requirements.runtime_requirements.minimum_report_allocation_bytes,
            requirements.abi_plan.required_bytes);
}

TEST(EvidenceRequirements, ExpertLimitCountsPhysicalAccessIntentsWithoutTruncatingSync) {
  const ObservationPlan plan =
      EvidenceObservationPlanBuilder(Mode::Default)
          .add(ProbeIntentKind::Access, SemanticSiteDomain::Access, 2)
          .add(ProbeIntentKind::Access, SemanticSiteDomain::Access, 3)
          .add(ProbeIntentKind::BarrierEpoch, SemanticSiteDomain::SynchronizationEvent, 2)
          .add(ProbeIntentKind::AtomicOrdering, SemanticSiteDomain::SynchronizationEvent, 1, true)
          .build();
  const auto requirements =
      plan_evidence(evidence_intents(plan), {.maximum_access_probe_count = 1});
  ASSERT_TRUE(requirements.complete());
  EXPECT_EQ(requirements.sizing_inventory.access_range_count, 2u);
  EXPECT_EQ(requirements.sizing_inventory.range_bank_count, 16u);
  EXPECT_EQ(requirements.sizing_inventory.sync_slot_count, 17u);
  EXPECT_EQ(requirements.sizing_inventory.barrier_event_count, 2u);
  EXPECT_EQ(requirements.sizing_inventory.atomic_event_count, 1u);
}

TEST(EvidenceRequirements, RequestedBanksAdaptToCapWithoutDroppingRanges) {
  const auto observation =
      EvidenceObservationPlanBuilder(Mode::Default)
          .add(ProbeIntentKind::Access, SemanticSiteDomain::Access, 2)
          .add(ProbeIntentKind::AtomicOrdering, SemanticSiteDomain::SynchronizationEvent, 1, true)
          .build();
  ProgramInventory inventory;
  const auto plan = [&](uint32_t banks, uint64_t ceiling) {
    return std::get<ReportRequirements>(
        detail::plan_evidence_requirements({.program_inventory = inventory,
                                            .observation_plan = observation,
                                            .requested_report_buffer_size = ceiling,
                                            .maximum_access_probe_count = std::nullopt,
                                            .maximum_workgroup_lds_bytes = std::nullopt,
                                            .watchpoint_banks = banks}));
  };
  const auto two = plan(2, 0);
  ASSERT_TRUE(two.complete());
  for (uint32_t banks : {0u, 1u, 2u, 4u, 8u, 16u, 256u, 1024u}) {
    const uint32_t expected = banks == 0 ? 8u : banks;
    const auto full = plan(banks, 0);
    ASSERT_TRUE(full.complete());
    EXPECT_EQ(full.sizing_inventory.access_range_count, 2u);
    EXPECT_EQ(full.sizing_inventory.range_bank_count, 2u * expected);
    EXPECT_EQ(full.sizing_inventory.sync_slot_count, 2u * expected + 1u);
    const auto capped = plan(banks, two.abi_plan.required_bytes);
    ASSERT_TRUE(capped.complete());
    EXPECT_EQ(capped.sizing_inventory.access_range_count, 2u);
    EXPECT_EQ(capped.sizing_inventory.range_bank_count, 2u * std::min(expected, 2u));
    EXPECT_EQ(capped.sizing_inventory.atomic_event_count, 1u);
  }
}

TEST(EvidenceRequirements, RejectsInvalidWrongForeignAndMalformedPlans) {
  ObservationPlan invalid;
  EXPECT_EQ(plan_evidence(evidence_intents(invalid)).reason,
            EvidenceRequirementReason::InvalidObservationPlan);

  ObservationPlan wrong;
  wrong.mode = Mode::SuperCollider;
  ASSERT_TRUE(wrong.valid());
  EXPECT_EQ(plan_evidence(evidence_intents(wrong)).reason, EvidenceRequirementReason::WrongMode);

  ObservationPlan foreign = EvidenceObservationPlanBuilder(Mode::Default)
                                .add(ProbeIntentKind::Access, SemanticSiteDomain::Access)
                                .build();
  foreign.probe_intents.front().kind = ProbeIntentKind::RedundantAccessObservation;
  ASSERT_TRUE(foreign.valid());
  EXPECT_EQ(plan_evidence(evidence_intents(foreign)).reason,
            EvidenceRequirementReason::UnexpectedIntentKind);

  ObservationPlan malformed =
      EvidenceObservationPlanBuilder(Mode::Default)
          .add(ProbeIntentKind::Access, SemanticSiteDomain::SynchronizationEvent)
          .build();
  ASSERT_TRUE(malformed.valid());
  EXPECT_EQ(plan_evidence(evidence_intents(malformed)).reason,
            EvidenceRequirementReason::InvalidIntentPayload);
}

TEST(EvidenceRequirements, WellFormedChecksEveryCrossTypeInvariant) {
  const auto good =
      plan_evidence(evidence_intents(EvidenceObservationPlanBuilder(Mode::Default)
                                         .add(ProbeIntentKind::Access, SemanticSiteDomain::Access)
                                         .build()));
  ASSERT_TRUE(good.well_formed());
  const auto expect_rejected = [&](auto mutate) {
    ReportRequirements broken = good;
    mutate(broken);
    EXPECT_FALSE(broken.well_formed());
    EXPECT_FALSE(broken.complete());
  };
  expect_rejected(
      [](auto &value) { value.runtime_requirements.host_device_visible_memory = false; });
  expect_rejected(
      [](auto &value) { value.runtime_requirements.host_device_coherent_memory = false; });
  expect_rejected(
      [](auto &value) { value.runtime_requirements.device_atomic_publication = false; });
  expect_rejected([](auto &value) { value.runtime_requirements.executable_binding = false; });
  expect_rejected([](auto &value) { value.runtime_requirements.max_workgroup_lds_bytes = true; });
  expect_rejected([](auto &value) { value.runtime_requirements.dispatch_segment_binding = true; });
  expect_rejected([](auto &value) { value.sizing_inventory.access_range_count += 2u; });
  expect_rejected([](auto &value) { value.sizing_inventory.bank_count_adaptive = false; });
  expect_rejected([](auto &value) { ++value.sizing_inventory.range_bank_count; });
  expect_rejected([](auto &value) { ++value.sizing_inventory.sync_slot_count; });
  expect_rejected([](auto &value) { ++value.sizing_inventory.watchpoint_count; });
  expect_rejected([](auto &value) { value.abi_plan.outcome = AutoReportPlanOutcome::Count; });
  expect_rejected([](auto &value) { value.abi_plan.reason = AutoReportPlanReason::Count; });
  expect_rejected([](auto &value) { ++value.abi_plan.layout.required_bytes; });
  expect_rejected(
      [](auto &value) { ++*value.runtime_requirements.minimum_report_allocation_bytes; });
}

TEST(EvidenceRequirements, CapacityFailureRemainsWellFormedButIncomplete) {
  const auto requirements = plan_evidence(
      evidence_intents(EvidenceObservationPlanBuilder(Mode::Default)
                           .add(ProbeIntentKind::Access, SemanticSiteDomain::Access)
                           .build()),
      {.caller_ceiling_bytes = sizeof(ReportHeader), .maximum_access_probe_count = std::nullopt});
  EXPECT_TRUE(requirements.well_formed());
  EXPECT_FALSE(requirements.complete());
  EXPECT_EQ(requirements.abi_plan.outcome, AutoReportPlanOutcome::InsufficientReportCapacity);
  EXPECT_EQ(requirements.abi_plan.reason, AutoReportPlanReason::PerBufferCeiling);
}

TEST(EvidenceRequirements, SuperColliderMarkerContractCoversEmptyAndObservedPlans) {
  ObservationPlan empty;
  empty.mode = Mode::SuperCollider;
  const auto no_marker = plan_supercollider_evidence(evidence_intents(empty));
  ASSERT_TRUE(no_marker.complete());
  EXPECT_EQ(no_marker.marker_bytes, 0u);
  EXPECT_FALSE(no_marker.requires_binding());

  const ObservationPlan observed =
      EvidenceObservationPlanBuilder(Mode::SuperCollider)
          .add(ProbeIntentKind::RedundantAccessObservation, SemanticSiteDomain::Access, 2)
          .build();
  const auto marker = plan_supercollider_evidence(evidence_intents(observed));
  ASSERT_TRUE(marker.complete());
  EXPECT_TRUE(marker.requires_binding());
  EXPECT_EQ(marker.marker_bytes, sizeof(uint32_t));
  EXPECT_EQ(marker.runtime_requirements.minimum_report_allocation_bytes, sizeof(uint32_t));
  EXPECT_TRUE(marker.runtime_requirements.host_device_visible_memory);
  EXPECT_TRUE(marker.runtime_requirements.host_device_coherent_memory);
  EXPECT_FALSE(marker.runtime_requirements.device_atomic_publication);
  EXPECT_TRUE(marker.runtime_requirements.executable_binding);

  const auto trap_only =
      plan_supercollider_evidence(evidence_intents(observed), SuperColliderEvidenceMode::TrapOnly);
  ASSERT_TRUE(trap_only.complete());
  EXPECT_EQ(trap_only.mode, SuperColliderEvidenceMode::TrapOnly);
  EXPECT_FALSE(trap_only.requires_binding());
  EXPECT_EQ(trap_only.marker_bytes, 0u);
  EXPECT_EQ(trap_only.runtime_requirements, RuntimeCapabilityRequirements{});
}

TEST(EvidenceRequirements, SuperColliderRejectsInvalidWrongForeignAndMalformedPlans) {
  ObservationPlan invalid;
  EXPECT_EQ(plan_supercollider_evidence(evidence_intents(invalid)).reason,
            EvidenceRequirementReason::InvalidObservationPlan);
  ObservationPlan wrong;
  wrong.mode = Mode::Default;
  ASSERT_TRUE(wrong.valid());
  EXPECT_EQ(plan_supercollider_evidence(evidence_intents(wrong)).reason,
            EvidenceRequirementReason::WrongMode);

  ObservationPlan foreign =
      EvidenceObservationPlanBuilder(Mode::SuperCollider)
          .add(ProbeIntentKind::RedundantAccessObservation, SemanticSiteDomain::Access)
          .build();
  foreign.probe_intents.front().kind = ProbeIntentKind::Access;
  ASSERT_TRUE(foreign.valid());
  EXPECT_EQ(plan_supercollider_evidence(evidence_intents(foreign)).reason,
            EvidenceRequirementReason::UnexpectedIntentKind);
  foreign.probe_intents.front().kind = ProbeIntentKind::RedundantAccessObservation;
  foreign.probe_intents.front().covered_semantic_sites.front().domain =
      SemanticSiteDomain::SynchronizationEvent;
  ASSERT_TRUE(foreign.valid());
  EXPECT_EQ(plan_supercollider_evidence(evidence_intents(foreign)).reason,
            EvidenceRequirementReason::InvalidIntentPayload);
}

TEST(EvidenceRequirements, SuperColliderWellFormedChecksEveryMarkerInvariant) {
  const auto good = plan_supercollider_evidence(evidence_intents(
      EvidenceObservationPlanBuilder(Mode::SuperCollider)
          .add(ProbeIntentKind::RedundantAccessObservation, SemanticSiteDomain::Access)
          .build()));
  ASSERT_TRUE(good.well_formed());
  const auto expect_rejected = [&](auto mutate) {
    SuperColliderEvidenceRequirements broken = good;
    mutate(broken);
    EXPECT_FALSE(broken.well_formed());
    EXPECT_FALSE(broken.complete());
    EXPECT_FALSE(broken.requires_binding());
  };
  expect_rejected([](auto &value) { value.marker_bytes = 8; });
  expect_rejected(
      [](auto &value) { value.runtime_requirements.host_device_visible_memory = false; });
  expect_rejected(
      [](auto &value) { value.runtime_requirements.host_device_coherent_memory = false; });
  expect_rejected([](auto &value) { value.runtime_requirements.device_atomic_publication = true; });
  expect_rejected([](auto &value) { value.runtime_requirements.executable_binding = false; });
  expect_rejected(
      [](auto &value) { ++*value.runtime_requirements.minimum_report_allocation_bytes; });
  expect_rejected([](auto &value) { value.mode = SuperColliderEvidenceMode::Count; });
}

TEST(EvidenceRequirements, ClosedVariantPreservesEachAlternativeAndWellFormedContract) {
  const std::array<EvidenceRequirements, 2> requirements = {
      plan_evidence(evidence_intents(EvidenceObservationPlanBuilder(Mode::Default)
                                         .add(ProbeIntentKind::Access, SemanticSiteDomain::Access)
                                         .build())),
      plan_supercollider_evidence(evidence_intents(
          EvidenceObservationPlanBuilder(Mode::SuperCollider)
              .add(ProbeIntentKind::RedundantAccessObservation, SemanticSiteDomain::Access)
              .build())),
  };
  EXPECT_TRUE(std::holds_alternative<ReportRequirements>(requirements[0]));
  EXPECT_TRUE(std::holds_alternative<SuperColliderEvidenceRequirements>(requirements[1]));
  for (const EvidenceRequirements &requirement : requirements)
    EXPECT_TRUE(evidence_requirements_well_formed(requirement));
  ReportRequirements broken = std::get<ReportRequirements>(requirements[0]);
  broken.runtime_requirements.executable_binding = false;
  EXPECT_FALSE(evidence_requirements_well_formed(EvidenceRequirements{broken}));
}

TEST(EvidenceRequirements, EveryAlternativePublishesAnIndependentlyValidatedRuntimeContract) {
  const std::array<EvidenceRequirements, 2> requirements = {
      plan_evidence(evidence_intents(EvidenceObservationPlanBuilder(Mode::Default)
                                         .add(ProbeIntentKind::Access, SemanticSiteDomain::Access)
                                         .build())),
      plan_supercollider_evidence(evidence_intents(
          EvidenceObservationPlanBuilder(Mode::SuperCollider)
              .add(ProbeIntentKind::RedundantAccessObservation, SemanticSiteDomain::Access)
              .build())),
  };

  for (const EvidenceRequirements &requirement : requirements) {
    SCOPED_TRACE(requirement.index());
    const RuntimeCapabilityRequirements runtime =
        std::visit([](const auto &typed) { return typed.runtime_requirements; }, requirement);
    ASSERT_TRUE(runtime.minimum_report_allocation_bytes);
    RuntimeCapabilities capabilities{
        .backend = RuntimeBackend::RocJitsuSimulator,
        .host_device_visible_memory = true,
        .host_device_coherent_memory = true,
        .device_atomic_publication = true,
        .max_report_allocation_bytes = *runtime.minimum_report_allocation_bytes,
        .max_workgroup_lds_bytes = std::nullopt,
        .executable_binding = true,
        .dispatch_segment_binding = false,
    };
    EXPECT_EQ(validate_runtime_capabilities(capabilities, runtime), ContractIssue::None);

    capabilities.host_device_visible_memory = false;
    EXPECT_EQ(validate_runtime_capabilities(capabilities, runtime),
              ContractIssue::MissingVisibleMemory);
    capabilities.host_device_visible_memory = true;
    capabilities.host_device_coherent_memory = false;
    EXPECT_EQ(validate_runtime_capabilities(capabilities, runtime),
              ContractIssue::MissingCoherentMemory);
    capabilities.host_device_coherent_memory = true;
    capabilities.max_report_allocation_bytes = *runtime.minimum_report_allocation_bytes - 1u;
    EXPECT_EQ(validate_runtime_capabilities(capabilities, runtime),
              ContractIssue::InsufficientReportAllocation);
    capabilities.max_report_allocation_bytes = *runtime.minimum_report_allocation_bytes;
    capabilities.executable_binding = false;
    EXPECT_EQ(validate_runtime_capabilities(capabilities, runtime),
              ContractIssue::MissingExecutableBinding);
  }

  for (size_t index = 0; index < requirements.size(); ++index) {
    const RuntimeCapabilityRequirements runtime = std::visit(
        [](const auto &typed) { return typed.runtime_requirements; }, requirements[index]);
    EXPECT_EQ(runtime.device_atomic_publication, index != 1u);
    RuntimeCapabilities capabilities{
        .backend = RuntimeBackend::RocJitsuSimulator,
        .host_device_visible_memory = true,
        .host_device_coherent_memory = true,
        .device_atomic_publication = false,
        .max_report_allocation_bytes = *runtime.minimum_report_allocation_bytes,
        .max_workgroup_lds_bytes = std::nullopt,
        .executable_binding = true,
        .dispatch_segment_binding = false,
    };
    EXPECT_EQ(validate_runtime_capabilities(capabilities, runtime),
              index == 1u ? ContractIssue::None : ContractIssue::MissingAtomicPublication);
  }
}

TEST(EvidenceRequirements, EveryPlannerIsDeterministicAndPreservesTypedInputs) {
  const ObservationPlan plan = EvidenceObservationPlanBuilder(Mode::Default)
                                   .add(ProbeIntentKind::Access, SemanticSiteDomain::Access, 2)
                                   .build();
  const ObservationPlan before = plan;
  const CapacityPolicy policy{.caller_ceiling_bytes = 32u * 1024u * 1024u,
                              .maximum_access_probe_count = 1};
  const ObservationPlan &intents = plan;
  EXPECT_EQ(plan_evidence(intents, policy), plan_evidence(intents, policy));
  EXPECT_EQ(plan, before);

  const ObservationPlan supercollider_plan =
      EvidenceObservationPlanBuilder(Mode::SuperCollider)
          .add(ProbeIntentKind::RedundantAccessObservation, SemanticSiteDomain::Access)
          .build();
  const ObservationPlan supercollider_before = supercollider_plan;
  const ObservationPlan &supercollider_intents = supercollider_plan;
  EXPECT_EQ(plan_supercollider_evidence(supercollider_intents),
            plan_supercollider_evidence(supercollider_intents));
  EXPECT_EQ(supercollider_plan, supercollider_before);
}

} // namespace
} // namespace rocjitsu::consan
