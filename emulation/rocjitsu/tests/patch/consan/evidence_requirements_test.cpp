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

/// Builder for valid engine-specific intent mixtures that do not need an
/// accompanying program inventory.
class EvidenceObservationPlanBuilder {
public:
  explicit EvidenceObservationPlanBuilder(ConSanCapabilityEngine engine) {
    plan_.engine = engine;
    physical_template_.code_object = make_consan_code_object_id(std::array<uint8_t, 4>{4, 3, 2, 1});
  }

  EvidenceObservationPlanBuilder &add(ConSanProbeIntentKind kind, ConSanSemanticSiteDomain domain,
                                      size_t semantic_count = 1, bool associated = false) {
    PhysicalSiteId physical = physical_template_;
    physical.original_text_offset = 32u + plan_.probe_intents.size() * 16u;
    ConSanProbeIntent intent;
    intent.id = {static_cast<uint32_t>(plan_.probe_intents.size())};
    intent.engine = plan_.engine;
    intent.physical_site = physical;
    intent.kind = kind;
    intent.position = kind == ConSanProbeIntentKind::AtomicAddressCapture
                          ? ConSanProbePosition::Before
                          : ConSanProbePosition::After;
    intent.lane_mask = ConSanLaneMaskPolicy::ActiveExecutionMask;
    intent.requirement = ConSanProbeRequirement::Required;
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

  [[nodiscard]] ConSanObservationPlan build() const { return plan_; }

private:
  ConSanObservationPlan plan_;
  PhysicalSiteId physical_template_;
};

/// Immutable inventory plus a matching InlineShadow access plan used to test
/// joins between policy intent identities and original LDS facts.
struct InlineEvidenceFixture {
  ProgramInventory inventory;
  ConSanObservationPlan plan;
};

InlineEvidenceFixture make_inline_evidence_fixture(bool flat, bool dynamic_lds,
                                                   uint32_t declared_lds_bytes,
                                                   uint32_t encoded_static_offset = 3u) {
  const std::array<uint8_t, 8> bytes = {static_cast<uint8_t>(encoded_static_offset),
                                        static_cast<uint8_t>(encoded_static_offset >> 8u),
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0};
  ProgramInventoryBuilder builder(bytes);
  builder.set_code_object_facts(true, 0, ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_TARGET_GFX950);
  ConSanKernelInfo kernel;
  kernel.name = "inline_owner";
  kernel.descriptor_file_offset = 512;
  kernel.declared_group_segment_bytes = declared_lds_bytes;
  kernel.has_dynamic_lds = dynamic_lds;
  if (flat) {
    ConSanFlatSite site;
    site.kind = ConSanLdsAccessKind::Write;
    site.text_offset = 16;
    site.file_offset = 0;
    site.size = 8;
    site.width_bits = 32;
    site.addr_vgpr = 2;
    site.data_vgpr = 3;
    site.raw_ioffset = 0;
    site.address_space_hint = ConSanFlatAddressSpaceHint::Group;
    site.owner_descriptor_file_offsets = {512};
    site.mnemonic = "flat_store_b32";
    kernel.flat_sites.push_back(std::move(site));
  } else {
    ConSanLdsSite site;
    site.kind = ConSanLdsAccessKind::Write;
    site.supported_mvp = true;
    site.text_offset = 16;
    site.file_offset = 0;
    site.size = 8;
    site.width_bits = 32;
    site.addr_vgpr = 2;
    site.data_vgpr = 3;
    site.owner_descriptor_file_offsets = {512};
    site.mnemonic = "ds_store_b32";
    kernel.lds_sites.push_back(std::move(site));
  }
  builder.kernels().push_back(std::move(kernel));
  builder.rebuild_access_inventory(bytes);

  InlineEvidenceFixture fixture;
  fixture.inventory = builder.view();
  const ConSanAccessInventorySite &site = fixture.inventory.access_sites().front();
  fixture.plan.engine = ConSanCapabilityEngine::InlineShadow;
  ConSanProbeIntent intent;
  intent.id = {0};
  intent.engine = fixture.plan.engine;
  intent.physical_site = site.physical_id;
  intent.kind = ConSanProbeIntentKind::ExactShadowAccess;
  intent.position = ConSanProbePosition::Before;
  intent.lane_mask = ConSanLaneMaskPolicy::ActiveExecutionMask;
  intent.requirement = ConSanProbeRequirement::Required;
  for (const ConSanAccessRange &range : site.ranges)
    intent.covered_semantic_sites.push_back(range.id);
  fixture.plan.probe_intents.push_back(std::move(intent));
  return fixture;
}

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
  expect_rejected([](auto &value) { ++value.sizing_inventory.diagnostic_count; });
  expect_rejected([](auto &value) { value.abi_plan.engine = ConSanMoiEngine::Sampled; });
  expect_rejected(
      [](auto &value) { value.abi_plan.outcome = ConSanMoiAutoReportPlanOutcome::Count; });
  expect_rejected(
      [](auto &value) { value.abi_plan.reason = ConSanMoiAutoReportPlanReason::Count; });
  expect_rejected([](auto &value) { ++value.abi_plan.layout.required_bytes; });
  expect_rejected(
      [](auto &value) { ++*value.runtime_requirements.minimum_report_allocation_bytes; });
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
  // no longer rescans resource plans, patches, or object bytes.
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

TEST(ConSanEvidenceRequirements,
     LegacyInventoryAdapterConsumesTypedPlansForSampledAndInlineShadow) {
  ConSanResult sampled_result;
  sampled_result.observation_plan =
      EvidenceObservationPlanBuilder(ConSanCapabilityEngine::Sampled)
          .add(ConSanProbeIntentKind::SampledAccess, ConSanSemanticSiteDomain::Access, 2)
          .add(ConSanProbeIntentKind::SampledAtomicOrdering,
               ConSanSemanticSiteDomain::SynchronizationEvent, 1, true)
          .build();
  sampled_result.resource_plans.resize(17);
  sampled_result.patches.resize(19);
  ConSanOptions sampled_options;
  sampled_options.moi_engine = ConSanMoiEngine::Sampled;
  EXPECT_EQ(inventory_consan_moi_auto_report(sampled_result, sampled_options, {}),
            plan_consan_sampled_evidence(sampled_result.observation_plan).sizing_inventory);

  const InlineEvidenceFixture fixture =
      make_inline_evidence_fixture(/*flat=*/false, /*dynamic_lds=*/false, 4096);
  ConSanResult inline_result;
  inline_result.program_inventory = fixture.inventory;
  inline_result.observation_plan = fixture.plan;
  inline_result.resource_plans.resize(29);
  inline_result.patches.resize(31);
  ConSanOptions inline_options;
  inline_options.moi_engine = ConSanMoiEngine::InlineShadow;
  EXPECT_EQ(inventory_consan_moi_auto_report(inline_result, inline_options, {}),
            plan_consan_inline_shadow_evidence(fixture.inventory, fixture.plan).sizing_inventory);
}

TEST(ConSanEvidenceRequirements, CapacityPolicyAdaptersCarryOnlyExplicitLegacyBounds) {
  ConSanOptions options;
  options.max_patches = 11;
  options.max_patches_is_expert_limit = false;
  options.moi_max_workgroup_lds_bytes = 96u * 1024u;
  EXPECT_EQ(make_consan_sampled_capacity_policy(options, 8192),
            (ConSanSampledCapacityPolicy{.caller_ceiling_bytes = 8192,
                                         .maximum_access_probe_count = std::nullopt}));
  EXPECT_EQ(make_consan_inline_shadow_capacity_policy(options, 16384),
            (ConSanInlineShadowCapacityPolicy{.caller_ceiling_bytes = 16384,
                                              .maximum_access_probe_count = std::nullopt,
                                              .maximum_workgroup_lds_bytes = 96u * 1024u}));

  options.max_patches_is_expert_limit = true;
  EXPECT_EQ(
      make_consan_sampled_capacity_policy(options),
      (ConSanSampledCapacityPolicy{.caller_ceiling_bytes = 0, .maximum_access_probe_count = 11}));
  EXPECT_EQ(make_consan_inline_shadow_capacity_policy(options),
            (ConSanInlineShadowCapacityPolicy{.caller_ceiling_bytes = 0,
                                              .maximum_access_probe_count = 11,
                                              .maximum_workgroup_lds_bytes = 96u * 1024u}));
}

TEST(ConSanEvidenceRequirements, CapacityPoliciesHaveNeutralAddressFreeDefaults) {
  EXPECT_EQ(ConSanRecordReplayCapacityPolicy{},
            (ConSanRecordReplayCapacityPolicy{.caller_ceiling_bytes = 0,
                                              .maximum_access_probe_count = std::nullopt}));
  EXPECT_EQ(ConSanSampledCapacityPolicy{},
            (ConSanSampledCapacityPolicy{.caller_ceiling_bytes = 0,
                                         .maximum_access_probe_count = std::nullopt}));
  EXPECT_EQ(ConSanInlineShadowCapacityPolicy{},
            (ConSanInlineShadowCapacityPolicy{.caller_ceiling_bytes = 0,
                                              .maximum_access_probe_count = std::nullopt,
                                              .maximum_workgroup_lds_bytes = std::nullopt}));
}

TEST(ConSanEvidenceRequirements, SampledIntentKindsMapToIndependentTypedCapacities) {
  const ConSanObservationPlan plan =
      EvidenceObservationPlanBuilder(ConSanCapabilityEngine::Sampled)
          .add(ConSanProbeIntentKind::SampledAccess, ConSanSemanticSiteDomain::Access, 2)
          .add(ConSanProbeIntentKind::SampledAccess, ConSanSemanticSiteDomain::Access, 1)
          .add(ConSanProbeIntentKind::SampledBarrierEpoch,
               ConSanSemanticSiteDomain::SynchronizationEvent, 3)
          .add(ConSanProbeIntentKind::AtomicAddressCapture,
               ConSanSemanticSiteDomain::SynchronizationEvent, 1, true)
          .add(ConSanProbeIntentKind::SampledAtomicOrdering,
               ConSanSemanticSiteDomain::SynchronizationEvent, 1, true)
          .build();
  ASSERT_TRUE(plan.valid());
  const ConSanSampledEvidenceRequirements requirements = plan_consan_sampled_evidence(plan);
  ASSERT_TRUE(requirements.complete());
  EXPECT_EQ(requirements.schema, ConSanEvidenceSchema::Sampled);
  EXPECT_EQ(requirements.boundedness, ConSanEvidenceBoundedness::BoundedSampled);
  EXPECT_EQ(requirements.loss_severity, ConSanEvidenceLossSeverity::InvalidatesCompleteness);
  EXPECT_EQ(requirements.delivery_scope, ConSanRuntimeResourceScope::Executable);
  EXPECT_EQ(requirements.sizing_inventory.access_range_count, 3u);
  EXPECT_EQ(requirements.sizing_inventory.barrier_event_count, 3u);
  EXPECT_EQ(requirements.sizing_inventory.atomic_event_count, 1u);
  EXPECT_EQ(requirements.sizing_inventory.sampled_range_bank_count, 24u);
  EXPECT_EQ(requirements.sizing_inventory.sampled_sync_slot_count, 25u);
  EXPECT_EQ(requirements.sizing_inventory.sampled_watchpoint_count, 25u);
  EXPECT_EQ(requirements.sizing_inventory.diagnostic_count, 3u);
  EXPECT_TRUE(requirements.runtime_requirements.host_device_visible_memory);
  EXPECT_TRUE(requirements.runtime_requirements.host_device_coherent_memory);
  EXPECT_TRUE(requirements.runtime_requirements.device_atomic_publication);
  EXPECT_TRUE(requirements.runtime_requirements.executable_binding);
  EXPECT_EQ(requirements.runtime_requirements.minimum_report_allocation_bytes,
            requirements.abi_plan.required_bytes);
}

TEST(ConSanEvidenceRequirements,
     SampledExpertLimitCountsPhysicalAccessIntentsWithoutTruncatingSync) {
  const ConSanObservationPlan plan =
      EvidenceObservationPlanBuilder(ConSanCapabilityEngine::Sampled)
          .add(ConSanProbeIntentKind::SampledAccess, ConSanSemanticSiteDomain::Access, 2)
          .add(ConSanProbeIntentKind::SampledAccess, ConSanSemanticSiteDomain::Access, 3)
          .add(ConSanProbeIntentKind::SampledBarrierEpoch,
               ConSanSemanticSiteDomain::SynchronizationEvent, 2)
          .add(ConSanProbeIntentKind::SampledAtomicOrdering,
               ConSanSemanticSiteDomain::SynchronizationEvent, 1, true)
          .build();
  const auto requirements = plan_consan_sampled_evidence(plan, {.maximum_access_probe_count = 1});
  ASSERT_TRUE(requirements.complete());
  EXPECT_EQ(requirements.sizing_inventory.access_range_count, 2u);
  EXPECT_EQ(requirements.sizing_inventory.sampled_range_bank_count, 16u);
  EXPECT_EQ(requirements.sizing_inventory.sampled_sync_slot_count, 17u);
  EXPECT_EQ(requirements.sizing_inventory.barrier_event_count, 2u);
  EXPECT_EQ(requirements.sizing_inventory.atomic_event_count, 1u);
}

TEST(ConSanEvidenceRequirements, SampledRejectsInvalidWrongForeignAndMalformedPlans) {
  ConSanObservationPlan invalid;
  EXPECT_EQ(plan_consan_sampled_evidence(invalid).reason,
            ConSanEvidenceRequirementReason::InvalidObservationPlan);

  ConSanObservationPlan wrong;
  wrong.engine = ConSanCapabilityEngine::InlineShadow;
  ASSERT_TRUE(wrong.valid());
  EXPECT_EQ(plan_consan_sampled_evidence(wrong).reason,
            ConSanEvidenceRequirementReason::WrongEngine);

  ConSanObservationPlan foreign =
      EvidenceObservationPlanBuilder(ConSanCapabilityEngine::Sampled)
          .add(ConSanProbeIntentKind::SampledAccess, ConSanSemanticSiteDomain::Access)
          .build();
  foreign.probe_intents.front().kind = ConSanProbeIntentKind::ExactShadowAccess;
  ASSERT_TRUE(foreign.valid());
  EXPECT_EQ(plan_consan_sampled_evidence(foreign).reason,
            ConSanEvidenceRequirementReason::UnexpectedIntentKind);

  ConSanObservationPlan malformed =
      EvidenceObservationPlanBuilder(ConSanCapabilityEngine::Sampled)
          .add(ConSanProbeIntentKind::SampledAccess, ConSanSemanticSiteDomain::SynchronizationEvent)
          .build();
  ASSERT_TRUE(malformed.valid());
  EXPECT_EQ(plan_consan_sampled_evidence(malformed).reason,
            ConSanEvidenceRequirementReason::InvalidIntentPayload);
}

TEST(ConSanEvidenceRequirements, SampledWellFormedChecksEveryCrossTypeInvariant) {
  const auto good = plan_consan_sampled_evidence(
      EvidenceObservationPlanBuilder(ConSanCapabilityEngine::Sampled)
          .add(ConSanProbeIntentKind::SampledAccess, ConSanSemanticSiteDomain::Access)
          .build());
  ASSERT_TRUE(good.well_formed());
  const auto expect_rejected = [&](auto mutate) {
    ConSanSampledEvidenceRequirements broken = good;
    mutate(broken);
    EXPECT_FALSE(broken.well_formed());
    EXPECT_FALSE(broken.complete());
  };
  expect_rejected([](auto &value) { value.schema = ConSanEvidenceSchema::RecordReplay; });
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
      [](auto &value) { value.sizing_inventory.engine = ConSanMoiEngine::InlineShadow; });
  expect_rejected([](auto &value) { ++value.sizing_inventory.access_range_count; });
  expect_rejected([](auto &value) { ++value.sizing_inventory.diagnostic_count; });
  expect_rejected([](auto &value) { value.sizing_inventory.sampled_bank_count_adaptive = false; });
  expect_rejected([](auto &value) { ++value.sizing_inventory.sampled_range_bank_count; });
  expect_rejected([](auto &value) { ++value.sizing_inventory.sampled_sync_slot_count; });
  expect_rejected([](auto &value) { ++value.sizing_inventory.sampled_watchpoint_count; });
  expect_rejected([](auto &value) { value.abi_plan.engine = ConSanMoiEngine::InlineShadow; });
  expect_rejected(
      [](auto &value) { value.abi_plan.outcome = ConSanMoiAutoReportPlanOutcome::Count; });
  expect_rejected(
      [](auto &value) { value.abi_plan.reason = ConSanMoiAutoReportPlanReason::Count; });
  expect_rejected([](auto &value) { ++value.abi_plan.layout.required_bytes; });
  expect_rejected(
      [](auto &value) { ++*value.runtime_requirements.minimum_report_allocation_bytes; });
}

TEST(ConSanEvidenceRequirements, SampledCapacityFailureRemainsWellFormedButIncomplete) {
  const auto requirements = plan_consan_sampled_evidence(
      EvidenceObservationPlanBuilder(ConSanCapabilityEngine::Sampled)
          .add(ConSanProbeIntentKind::SampledAccess, ConSanSemanticSiteDomain::Access)
          .build(),
      {.caller_ceiling_bytes = sizeof(ConSanMoiReportHeader),
       .maximum_access_probe_count = std::nullopt});
  EXPECT_TRUE(requirements.well_formed());
  EXPECT_FALSE(requirements.complete());
  EXPECT_EQ(requirements.abi_plan.outcome,
            ConSanMoiAutoReportPlanOutcome::InsufficientReportCapacity);
  EXPECT_EQ(requirements.abi_plan.reason, ConSanMoiAutoReportPlanReason::PerBufferCeiling);
}

TEST(ConSanEvidenceRequirements, InlineNativeAccessUsesImmutableDescriptorAndRangeFacts) {
  const InlineEvidenceFixture fixture =
      make_inline_evidence_fixture(/*flat=*/false, /*dynamic_lds=*/false,
                                   /*declared_lds_bytes=*/4096, /*encoded_static_offset=*/17);
  ASSERT_TRUE(fixture.plan.valid());
  const ConSanInlineShadowEvidenceRequirements requirements =
      plan_consan_inline_shadow_evidence(fixture.inventory, fixture.plan);
  ASSERT_TRUE(requirements.complete());
  EXPECT_EQ(requirements.schema, ConSanEvidenceSchema::InlineShadow);
  EXPECT_EQ(requirements.boundedness, ConSanEvidenceBoundedness::ExactOnDevice);
  EXPECT_EQ(requirements.required_lds_aperture_bytes, 4096u);
  EXPECT_EQ(requirements.sizing_inventory.inline_lds_bytes, 4096u);
  EXPECT_EQ(requirements.sizing_inventory.access_range_count, 1u);
  EXPECT_EQ(requirements.sizing_inventory.inline_compact_token_mapping_count, 1u);
  EXPECT_EQ(requirements.sizing_inventory.inline_atomic_release_count,
            kConSanMoiInlineShadowAtomicReleaseSlotCapacity);
  EXPECT_EQ(fixture.inventory.access_sites().front().ranges.front().static_byte_offset, 17);
}

TEST(ConSanEvidenceRequirements,
     InlineFlatDynamicAndOpaqueNativeAccessesUseTheConfiguredFullAperture) {
  constexpr uint32_t kFullAperture = 80u * 1024u;
  for (const InlineEvidenceFixture &fixture : {
           make_inline_evidence_fixture(/*flat=*/true, /*dynamic_lds=*/false, 1024),
           make_inline_evidence_fixture(/*flat=*/false, /*dynamic_lds=*/true, 1024),
           make_inline_evidence_fixture(/*flat=*/false, /*dynamic_lds=*/false, 2,
                                        /*encoded_static_offset=*/7),
       }) {
    const auto requirements =
        plan_consan_inline_shadow_evidence(fixture.inventory, fixture.plan,
                                           {.caller_ceiling_bytes = 0,
                                            .maximum_access_probe_count = std::nullopt,
                                            .maximum_workgroup_lds_bytes = kFullAperture});
    ASSERT_TRUE(requirements.complete());
    EXPECT_EQ(requirements.required_lds_aperture_bytes, kFullAperture);
    EXPECT_EQ(requirements.sizing_inventory.inline_lds_bytes, kFullAperture);
  }
}

TEST(ConSanEvidenceRequirements, InlineIntentCountsAndExpertLimitRemainIndependent) {
  InlineEvidenceFixture fixture =
      make_inline_evidence_fixture(/*flat=*/false, /*dynamic_lds=*/false, 4096);
  ConSanProbeIntent second = fixture.plan.probe_intents.front();
  second.id = {1};
  second.physical_site.original_text_offset += 16;
  for (SemanticSiteId &id : second.covered_semantic_sites)
    id.physical = second.physical_site;
  fixture.plan.probe_intents.push_back(second);
  // The second identity is intentionally not in inventory; an expert limit of
  // one proves that rejected physical access intents are never joined or
  // allowed to truncate barrier/atomic requirements.
  fixture.plan.probe_intents.push_back(
      EvidenceObservationPlanBuilder(ConSanCapabilityEngine::InlineShadow)
          .add(ConSanProbeIntentKind::ExactAtomicOrdering,
               ConSanSemanticSiteDomain::SynchronizationEvent, 1, true)
          .build()
          .probe_intents.front());
  fixture.plan.probe_intents.back().id = {2};
  fixture.plan.probe_intents.back().engine = ConSanCapabilityEngine::InlineShadow;
  ASSERT_TRUE(fixture.plan.valid());
  const auto requirements =
      plan_consan_inline_shadow_evidence(fixture.inventory, fixture.plan,
                                         {.caller_ceiling_bytes = 0,
                                          .maximum_access_probe_count = 1,
                                          .maximum_workgroup_lds_bytes = std::nullopt});
  ASSERT_TRUE(requirements.complete());
  EXPECT_EQ(requirements.sizing_inventory.access_range_count, 1u);
  EXPECT_EQ(requirements.sizing_inventory.inline_compact_token_mapping_count, 1u);
  EXPECT_EQ(requirements.sizing_inventory.atomic_event_count, 1u);
}

TEST(ConSanEvidenceRequirements, InlineMissingInventoryRelationshipsFailClosedWithTypedReason) {
  InlineEvidenceFixture fixture =
      make_inline_evidence_fixture(/*flat=*/false, /*dynamic_lds=*/false, 4096);
  fixture.plan.probe_intents.front().covered_semantic_sites.front().range_ordinal = 99;
  ASSERT_TRUE(fixture.plan.valid());
  EXPECT_EQ(plan_consan_inline_shadow_evidence(fixture.inventory, fixture.plan).reason,
            ConSanEvidenceRequirementReason::MissingInventoryFact);

  fixture = make_inline_evidence_fixture(/*flat=*/false, /*dynamic_lds=*/false, 4096);
  ProgramInventoryBuilder missing_descriptor(fixture.inventory);
  missing_descriptor.kernels().front().declared_group_segment_bytes.reset();
  EXPECT_EQ(plan_consan_inline_shadow_evidence(missing_descriptor.view(), fixture.plan).reason,
            ConSanEvidenceRequirementReason::MissingInventoryFact);
}

TEST(ConSanEvidenceRequirements, InlineRejectsWrongForeignAndMalformedPlans) {
  const InlineEvidenceFixture fixture =
      make_inline_evidence_fixture(/*flat=*/false, /*dynamic_lds=*/false, 4096);
  ConSanObservationPlan wrong;
  wrong.engine = ConSanCapabilityEngine::Sampled;
  ASSERT_TRUE(wrong.valid());
  EXPECT_EQ(plan_consan_inline_shadow_evidence(fixture.inventory, wrong).reason,
            ConSanEvidenceRequirementReason::WrongEngine);

  ConSanObservationPlan foreign = fixture.plan;
  foreign.probe_intents.front().kind = ConSanProbeIntentKind::SampledAccess;
  ASSERT_TRUE(foreign.valid());
  EXPECT_EQ(plan_consan_inline_shadow_evidence(fixture.inventory, foreign).reason,
            ConSanEvidenceRequirementReason::UnexpectedIntentKind);

  ConSanObservationPlan malformed = fixture.plan;
  malformed.probe_intents.front().covered_semantic_sites.front().domain =
      ConSanSemanticSiteDomain::SynchronizationEvent;
  ASSERT_TRUE(malformed.valid());
  EXPECT_EQ(plan_consan_inline_shadow_evidence(fixture.inventory, malformed).reason,
            ConSanEvidenceRequirementReason::InvalidIntentPayload);
}

TEST(ConSanEvidenceRequirements, InlineWellFormedChecksEverySchemaSpecificInvariant) {
  const InlineEvidenceFixture fixture =
      make_inline_evidence_fixture(/*flat=*/false, /*dynamic_lds=*/false, 4096);
  const auto good = plan_consan_inline_shadow_evidence(fixture.inventory, fixture.plan);
  ASSERT_TRUE(good.well_formed());
  const auto expect_rejected = [&](auto mutate) {
    ConSanInlineShadowEvidenceRequirements broken = good;
    mutate(broken);
    EXPECT_FALSE(broken.well_formed());
    EXPECT_FALSE(broken.complete());
  };
  expect_rejected([](auto &value) { value.schema = ConSanEvidenceSchema::Sampled; });
  expect_rejected(
      [](auto &value) { value.boundedness = ConSanEvidenceBoundedness::BoundedSampled; });
  expect_rejected([](auto &value) { value.loss_severity = ConSanEvidenceLossSeverity::Optional; });
  expect_rejected([](auto &value) { value.delivery_scope = ConSanRuntimeResourceScope::Dispatch; });
  expect_rejected([](auto &value) { ++value.required_lds_aperture_bytes; });
  expect_rejected(
      [](auto &value) { value.runtime_requirements.host_device_visible_memory = false; });
  expect_rejected(
      [](auto &value) { value.runtime_requirements.host_device_coherent_memory = false; });
  expect_rejected(
      [](auto &value) { value.runtime_requirements.device_atomic_publication = false; });
  expect_rejected([](auto &value) { value.runtime_requirements.executable_binding = false; });
  expect_rejected([](auto &value) { value.runtime_requirements.max_workgroup_lds_bytes = true; });
  expect_rejected([](auto &value) { value.runtime_requirements.dispatch_segment_binding = true; });
  expect_rejected([](auto &value) { value.sizing_inventory.engine = ConSanMoiEngine::Sampled; });
  expect_rejected([](auto &value) { ++value.sizing_inventory.diagnostic_count; });
  expect_rejected([](auto &value) { ++value.sizing_inventory.inline_atomic_release_count; });
  expect_rejected([](auto &value) { ++value.sizing_inventory.inline_causal_snapshot_count; });
  expect_rejected([](auto &value) { ++value.sizing_inventory.inline_acquired_epoch_token_count; });
  expect_rejected(
      [](auto &value) { value.sizing_inventory.inline_compact_token_mapping_count = 2; });
  expect_rejected(
      [](auto &value) { value.sizing_inventory.inline_diagnostic_count_adaptive = false; });
  expect_rejected([](auto &value) { value.abi_plan.engine = ConSanMoiEngine::Sampled; });
  expect_rejected(
      [](auto &value) { value.abi_plan.outcome = ConSanMoiAutoReportPlanOutcome::Count; });
  expect_rejected(
      [](auto &value) { value.abi_plan.reason = ConSanMoiAutoReportPlanReason::Count; });
  expect_rejected([](auto &value) { ++value.abi_plan.layout.required_bytes; });
  expect_rejected(
      [](auto &value) { ++*value.runtime_requirements.minimum_report_allocation_bytes; });
}

TEST(ConSanEvidenceRequirements, InlineCapacityFailureRemainsWellFormedButIncomplete) {
  const InlineEvidenceFixture fixture =
      make_inline_evidence_fixture(/*flat=*/false, /*dynamic_lds=*/false, 4096);
  const auto requirements =
      plan_consan_inline_shadow_evidence(fixture.inventory, fixture.plan,
                                         {.caller_ceiling_bytes = sizeof(ConSanMoiReportHeader),
                                          .maximum_access_probe_count = std::nullopt,
                                          .maximum_workgroup_lds_bytes = std::nullopt});
  EXPECT_TRUE(requirements.well_formed());
  EXPECT_FALSE(requirements.complete());
  EXPECT_EQ(requirements.abi_plan.outcome,
            ConSanMoiAutoReportPlanOutcome::InsufficientReportCapacity);
  EXPECT_EQ(requirements.abi_plan.reason, ConSanMoiAutoReportPlanReason::PerBufferCeiling);
}

TEST(ConSanEvidenceRequirements, SuperColliderMarkerContractCoversEmptyAndObservedPlans) {
  ConSanObservationPlan empty;
  empty.engine = ConSanCapabilityEngine::SuperCollider;
  const auto no_marker = plan_consan_supercollider_evidence(empty);
  ASSERT_TRUE(no_marker.complete());
  EXPECT_EQ(no_marker.schema, ConSanEvidenceSchema::SuperCollider);
  EXPECT_EQ(no_marker.boundedness, ConSanEvidenceBoundedness::StickyMarker);
  EXPECT_EQ(no_marker.loss_severity, ConSanEvidenceLossSeverity::InvalidatesCompleteness);
  EXPECT_EQ(no_marker.marker_bytes, 0u);
  EXPECT_FALSE(no_marker.requires_binding());

  const ConSanObservationPlan observed =
      EvidenceObservationPlanBuilder(ConSanCapabilityEngine::SuperCollider)
          .add(ConSanProbeIntentKind::RedundantAccessObservation, ConSanSemanticSiteDomain::Access,
               2)
          .build();
  const auto marker = plan_consan_supercollider_evidence(observed);
  ASSERT_TRUE(marker.complete());
  EXPECT_TRUE(marker.requires_binding());
  EXPECT_EQ(marker.marker_bytes, sizeof(uint32_t));
  EXPECT_EQ(marker.runtime_requirements.minimum_report_allocation_bytes, sizeof(uint32_t));
  EXPECT_TRUE(marker.runtime_requirements.host_device_visible_memory);
  EXPECT_TRUE(marker.runtime_requirements.host_device_coherent_memory);
  EXPECT_FALSE(marker.runtime_requirements.device_atomic_publication);
  EXPECT_TRUE(marker.runtime_requirements.executable_binding);
}

TEST(ConSanEvidenceRequirements, SuperColliderRejectsInvalidWrongForeignAndMalformedPlans) {
  ConSanObservationPlan invalid;
  EXPECT_EQ(plan_consan_supercollider_evidence(invalid).reason,
            ConSanEvidenceRequirementReason::InvalidObservationPlan);
  ConSanObservationPlan wrong;
  wrong.engine = ConSanCapabilityEngine::RecordReplay;
  ASSERT_TRUE(wrong.valid());
  EXPECT_EQ(plan_consan_supercollider_evidence(wrong).reason,
            ConSanEvidenceRequirementReason::WrongEngine);

  ConSanObservationPlan foreign =
      EvidenceObservationPlanBuilder(ConSanCapabilityEngine::SuperCollider)
          .add(ConSanProbeIntentKind::RedundantAccessObservation, ConSanSemanticSiteDomain::Access)
          .build();
  foreign.probe_intents.front().kind = ConSanProbeIntentKind::AccessRecord;
  ASSERT_TRUE(foreign.valid());
  EXPECT_EQ(plan_consan_supercollider_evidence(foreign).reason,
            ConSanEvidenceRequirementReason::UnexpectedIntentKind);
  foreign.probe_intents.front().kind = ConSanProbeIntentKind::RedundantAccessObservation;
  foreign.probe_intents.front().covered_semantic_sites.front().domain =
      ConSanSemanticSiteDomain::SynchronizationEvent;
  ASSERT_TRUE(foreign.valid());
  EXPECT_EQ(plan_consan_supercollider_evidence(foreign).reason,
            ConSanEvidenceRequirementReason::InvalidIntentPayload);
}

TEST(ConSanEvidenceRequirements, SuperColliderWellFormedChecksEveryMarkerInvariant) {
  const auto good = plan_consan_supercollider_evidence(
      EvidenceObservationPlanBuilder(ConSanCapabilityEngine::SuperCollider)
          .add(ConSanProbeIntentKind::RedundantAccessObservation, ConSanSemanticSiteDomain::Access)
          .build());
  ASSERT_TRUE(good.well_formed());
  const auto expect_rejected = [&](auto mutate) {
    ConSanSuperColliderEvidenceRequirements broken = good;
    mutate(broken);
    EXPECT_FALSE(broken.well_formed());
    EXPECT_FALSE(broken.complete());
    EXPECT_FALSE(broken.requires_binding());
  };
  expect_rejected([](auto &value) { value.schema = ConSanEvidenceSchema::InlineShadow; });
  expect_rejected(
      [](auto &value) { value.boundedness = ConSanEvidenceBoundedness::ExactOnDevice; });
  expect_rejected([](auto &value) { value.loss_severity = ConSanEvidenceLossSeverity::Optional; });
  expect_rejected([](auto &value) { value.delivery_scope = ConSanRuntimeResourceScope::Dispatch; });
  expect_rejected([](auto &value) { value.marker_bytes = 8; });
  expect_rejected(
      [](auto &value) { value.runtime_requirements.host_device_visible_memory = false; });
  expect_rejected(
      [](auto &value) { value.runtime_requirements.host_device_coherent_memory = false; });
  expect_rejected([](auto &value) { value.runtime_requirements.device_atomic_publication = true; });
  expect_rejected([](auto &value) { value.runtime_requirements.executable_binding = false; });
  expect_rejected(
      [](auto &value) { ++*value.runtime_requirements.minimum_report_allocation_bytes; });
}

TEST(ConSanEvidenceRequirements, ClosedVariantPreservesEachSchemaAndWellFormedContract) {
  const InlineEvidenceFixture inline_fixture =
      make_inline_evidence_fixture(/*flat=*/false, /*dynamic_lds=*/false, 4096);
  const std::array<ConSanEvidenceRequirements, 4> requirements = {
      plan_consan_record_replay_evidence(
          RecordReplayObservationPlanBuilder().add_access(1).build()),
      plan_consan_sampled_evidence(
          EvidenceObservationPlanBuilder(ConSanCapabilityEngine::Sampled)
              .add(ConSanProbeIntentKind::SampledAccess, ConSanSemanticSiteDomain::Access)
              .build()),
      plan_consan_inline_shadow_evidence(inline_fixture.inventory, inline_fixture.plan),
      plan_consan_supercollider_evidence(
          EvidenceObservationPlanBuilder(ConSanCapabilityEngine::SuperCollider)
              .add(ConSanProbeIntentKind::RedundantAccessObservation,
                   ConSanSemanticSiteDomain::Access)
              .build()),
  };
  for (size_t index = 0; index < requirements.size(); ++index) {
    EXPECT_EQ(consan_evidence_requirements_schema(requirements[index]),
              kConSanEvidenceSchemas[index]);
    EXPECT_TRUE(consan_evidence_requirements_well_formed(requirements[index]));
    EXPECT_EQ(requirements[index].index(), index);
  }
  ConSanSampledEvidenceRequirements broken = std::get<1>(requirements[1]);
  broken.schema = ConSanEvidenceSchema::RecordReplay;
  EXPECT_FALSE(consan_evidence_requirements_well_formed(ConSanEvidenceRequirements{broken}));
}

TEST(ConSanEvidenceRequirements, EverySchemaPublishesAnIndependentlyValidatedRuntimeContract) {
  const InlineEvidenceFixture inline_fixture =
      make_inline_evidence_fixture(/*flat=*/false, /*dynamic_lds=*/false, 4096);
  const std::array<ConSanEvidenceRequirements, 4> requirements = {
      plan_consan_record_replay_evidence(
          RecordReplayObservationPlanBuilder().add_access(1).build()),
      plan_consan_sampled_evidence(
          EvidenceObservationPlanBuilder(ConSanCapabilityEngine::Sampled)
              .add(ConSanProbeIntentKind::SampledAccess, ConSanSemanticSiteDomain::Access)
              .build()),
      plan_consan_inline_shadow_evidence(inline_fixture.inventory, inline_fixture.plan),
      plan_consan_supercollider_evidence(
          EvidenceObservationPlanBuilder(ConSanCapabilityEngine::SuperCollider)
              .add(ConSanProbeIntentKind::RedundantAccessObservation,
                   ConSanSemanticSiteDomain::Access)
              .build()),
  };

  for (const ConSanEvidenceRequirements &requirement : requirements) {
    const RuntimeCapabilityRequirements runtime =
        std::visit([](const auto &typed) { return typed.runtime_requirements; }, requirement);
    ASSERT_TRUE(runtime.minimum_report_allocation_bytes);
    RuntimeCapabilities capabilities{
        .backend = ConSanRuntimeBackend::RocJitsuSimulator,
        .host_device_visible_memory = true,
        .host_device_coherent_memory = true,
        .device_atomic_publication = true,
        .max_report_allocation_bytes = *runtime.minimum_report_allocation_bytes,
        .max_workgroup_lds_bytes = std::nullopt,
        .executable_binding = true,
        .dispatch_segment_binding = false,
    };
    EXPECT_EQ(validate_runtime_capabilities(capabilities, runtime), ConSanContractIssue::None)
        << consan_evidence_schema_name(consan_evidence_requirements_schema(requirement));

    capabilities.host_device_visible_memory = false;
    EXPECT_EQ(validate_runtime_capabilities(capabilities, runtime),
              ConSanContractIssue::MissingVisibleMemory);
    capabilities.host_device_visible_memory = true;
    capabilities.host_device_coherent_memory = false;
    EXPECT_EQ(validate_runtime_capabilities(capabilities, runtime),
              ConSanContractIssue::MissingCoherentMemory);
    capabilities.host_device_coherent_memory = true;
    capabilities.max_report_allocation_bytes = *runtime.minimum_report_allocation_bytes - 1u;
    EXPECT_EQ(validate_runtime_capabilities(capabilities, runtime),
              ConSanContractIssue::InsufficientReportAllocation);
    capabilities.max_report_allocation_bytes = *runtime.minimum_report_allocation_bytes;
    capabilities.executable_binding = false;
    EXPECT_EQ(validate_runtime_capabilities(capabilities, runtime),
              ConSanContractIssue::MissingExecutableBinding);
  }

  for (size_t index = 0; index < requirements.size(); ++index) {
    const RuntimeCapabilityRequirements runtime = std::visit(
        [](const auto &typed) { return typed.runtime_requirements; }, requirements[index]);
    EXPECT_EQ(runtime.device_atomic_publication, index != 3u);
    RuntimeCapabilities capabilities{
        .backend = ConSanRuntimeBackend::RocJitsuSimulator,
        .host_device_visible_memory = true,
        .host_device_coherent_memory = true,
        .device_atomic_publication = false,
        .max_report_allocation_bytes = *runtime.minimum_report_allocation_bytes,
        .max_workgroup_lds_bytes = std::nullopt,
        .executable_binding = true,
        .dispatch_segment_binding = false,
    };
    EXPECT_EQ(validate_runtime_capabilities(capabilities, runtime),
              index == 3u ? ConSanContractIssue::None
                          : ConSanContractIssue::MissingAtomicPublication);
  }
}

TEST(ConSanEvidenceRequirements, EveryPlannerIsDeterministicAndPreservesTypedInputs) {
  const ConSanObservationPlan sampled_plan =
      EvidenceObservationPlanBuilder(ConSanCapabilityEngine::Sampled)
          .add(ConSanProbeIntentKind::SampledAccess, ConSanSemanticSiteDomain::Access, 2)
          .build();
  const ConSanObservationPlan sampled_before = sampled_plan;
  const ConSanSampledCapacityPolicy sampled_policy{.caller_ceiling_bytes = 32u * 1024u * 1024u,
                                                   .maximum_access_probe_count = 1};
  EXPECT_EQ(plan_consan_sampled_evidence(sampled_plan, sampled_policy),
            plan_consan_sampled_evidence(sampled_plan, sampled_policy));
  EXPECT_EQ(sampled_plan, sampled_before);

  const InlineEvidenceFixture inline_fixture =
      make_inline_evidence_fixture(/*flat=*/false, /*dynamic_lds=*/false, 4096);
  const ConSanObservationPlan inline_before = inline_fixture.plan;
  const std::vector<ConSanAccessInventorySite> inventory_before(
      inline_fixture.inventory.access_sites().begin(),
      inline_fixture.inventory.access_sites().end());
  const ConSanInlineShadowCapacityPolicy inline_policy{.caller_ceiling_bytes = 64u * 1024u * 1024u,
                                                       .maximum_access_probe_count = 1,
                                                       .maximum_workgroup_lds_bytes = 64u * 1024u};
  EXPECT_EQ(plan_consan_inline_shadow_evidence(inline_fixture.inventory, inline_fixture.plan,
                                               inline_policy),
            plan_consan_inline_shadow_evidence(inline_fixture.inventory, inline_fixture.plan,
                                               inline_policy));
  EXPECT_EQ(inline_fixture.plan, inline_before);
  EXPECT_TRUE(std::ranges::equal(inline_fixture.inventory.access_sites(), inventory_before));

  const ConSanObservationPlan supercollider_plan =
      EvidenceObservationPlanBuilder(ConSanCapabilityEngine::SuperCollider)
          .add(ConSanProbeIntentKind::RedundantAccessObservation, ConSanSemanticSiteDomain::Access)
          .build();
  const ConSanObservationPlan supercollider_before = supercollider_plan;
  EXPECT_EQ(plan_consan_supercollider_evidence(supercollider_plan),
            plan_consan_supercollider_evidence(supercollider_plan));
  EXPECT_EQ(supercollider_plan, supercollider_before);
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
