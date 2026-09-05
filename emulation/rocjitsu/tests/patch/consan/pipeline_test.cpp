// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"
#include "transform_result_test_access.h"

#include "rocjitsu/code/patch/consan/consan_moi_mode_planning.h"
#include "rocjitsu/code/patch/consan/consan_moi_pipeline.h"
#include "rocjitsu/code/patch/consan/consan_pipeline.h"

#include <gtest/gtest.h>

#include <array>
#include <concepts>
#include <cstdint>
#include <string_view>
#include <utility>

namespace rocjitsu {
namespace {

static_assert(!std::derived_from<TransformResult, ConSanTransformArtifacts>);

template <typename T>
concept HasPublicTransformDiagnosticReport =
    requires(const T &result) { result.diagnostic_report(); };

static_assert(!HasPublicTransformDiagnosticReport<TransformResult>);

template <typename T>
concept HasPatchValidationProof = requires(const T &patch) {
  patch.owner_descriptor_file_offsets;
  patch.fault_sequence_identity;
};

static_assert(HasPatchValidationProof<ConSanPatchInfo>);
static_assert(!HasPatchValidationProof<ConSanPatchDiagnostic>);

template <typename T>
concept HasCommittedPatchGeometry = requires(const T &patch) {
  patch.kind;
  patch.anchor_offset;
  patch.trampoline_offset;
  patch.relocated_guest_instruction_offset;
};

template <typename T>
concept HasPatchRoutingProof =
    requires(const T &patch) { patch.dispatch_id_primary_prologue_offset; };

template <typename T>
concept HasPatchMutationProof = requires(const T &patch) {
  patch.perturbation_edge;
  patch.barrier_move_cfg_contract;
};

template <typename T>
concept HasPatchAbiEffects = requires(const T &patch) {
  patch.required_private_segment_size;
  patch.owner_descriptor_file_offsets;
};

template <typename T>
concept HasPatchFaultProof = requires(const T &patch) {
  patch.fault_sequence_identity;
  patch.fault_target_address_vgpr;
};

static_assert(std::derived_from<ConSanPatchInfo, ConSanCommittedPatchGeometry>);
static_assert(std::derived_from<ConSanPatchInfo, ConSanPatchRoutingProof>);
static_assert(std::derived_from<ConSanPatchInfo, ConSanPatchMutationProof>);
static_assert(std::derived_from<ConSanPatchInfo, ConSanPatchAbiEffects>);
static_assert(std::derived_from<ConSanPatchInfo, ConSanPatchFaultProof>);
static_assert(std::derived_from<ConSanPatchInfo, ConSanPatchLoweringProduct>);
static_assert(std::derived_from<ConSanPatchInfo, ConSanPatchMutationProduct>);
static_assert(std::derived_from<ConSanPatchMutationProduct, ConSanPatchLoweringProduct>);
static_assert(std::derived_from<ConSanPatchMutationProduct, ConSanPatchMutationProof>);
static_assert(std::derived_from<ConSanPatchLoweringProduct, ConSanCommittedPatchGeometry>);
static_assert(std::derived_from<ConSanPatchLoweringProduct, ConSanPatchPlacementEffects>);
static_assert(std::derived_from<ConSanPatchPlacementEffects, ConSanPatchRoutingProof>);
static_assert(std::derived_from<ConSanPatchPlacementEffects, ConSanPatchAbiEffects>);

static_assert(HasCommittedPatchGeometry<ConSanCommittedPatchGeometry>);
static_assert(!HasPatchRoutingProof<ConSanCommittedPatchGeometry>);
static_assert(!HasPatchAbiEffects<ConSanCommittedPatchGeometry>);
static_assert(!HasPatchFaultProof<ConSanCommittedPatchGeometry>);
static_assert(HasPatchRoutingProof<ConSanPatchRoutingProof>);
static_assert(!HasCommittedPatchGeometry<ConSanPatchRoutingProof>);
static_assert(HasPatchMutationProof<ConSanPatchMutationProof>);
static_assert(!HasPatchAbiEffects<ConSanPatchMutationProof>);
static_assert(HasPatchAbiEffects<ConSanPatchAbiEffects>);
static_assert(!HasCommittedPatchGeometry<ConSanPatchAbiEffects>);
static_assert(HasPatchFaultProof<ConSanPatchFaultProof>);
static_assert(!HasPatchMutationProof<ConSanPatchFaultProof>);
static_assert(!HasPatchMutationProof<ConSanPatchLoweringProduct>);
static_assert(!HasPatchFaultProof<ConSanPatchLoweringProduct>);
static_assert(!HasPatchFaultProof<ConSanPatchMutationProduct>);
static_assert(!HasCommittedPatchGeometry<ConSanPatchPlacementEffects>);
static_assert(!HasPatchMutationProof<ConSanPatchPlacementEffects>);
static_assert(!HasPatchFaultProof<ConSanPatchPlacementEffects>);

template <typename T>
concept IsMoiLoweringSummaryInventory = requires(T inventory) {
  consan_moi_impl::summarize_moi_lowering(ConSanMoiEngine::RecordReplay, true, inventory);
};

static_assert(IsMoiLoweringSummaryInventory<std::span<const ConSanPatchKind>>);
static_assert(!IsMoiLoweringSummaryInventory<std::span<const ConSanPatchInfo>>);
static_assert(!IsMoiLoweringSummaryInventory<const ConSanTransformArtifacts &>);

TEST(ConSanPipeline, BarrierScratchSizingIsModeOwned) {
  BoundRuntimeResources resources;
  ConSanMoiOperatingPoint point;
  consan_moi_impl::MoiObjectModeSemantics mode_semantics;
  const auto scratch_count = [&](ConSanMoiEngine engine) {
    return consan_moi_impl::moi_mode_operations(engine).barrier_scratch_vgpr_count(
        consan_moi_impl::project_moi_barrier_scratch_facts(resources, point, mode_semantics));
  };

  EXPECT_EQ(scratch_count(ConSanMoiEngine::RecordReplay), 6u);
  EXPECT_EQ(scratch_count(ConSanMoiEngine::Sampled), 7u);
  EXPECT_EQ(scratch_count(ConSanMoiEngine::InlineShadow), 1u);

  resources.moi_report_buffer_address = 0x123456780000ull;
  EXPECT_EQ(scratch_count(ConSanMoiEngine::InlineShadow), 3u);
  mode_semantics.inline_access_present = true;
  EXPECT_EQ(scratch_count(ConSanMoiEngine::InlineShadow), 1u);

  point.moi_persistent_sgprs.set_owner_epoch(20u, 21u);
  EXPECT_EQ(scratch_count(ConSanMoiEngine::Sampled), 9u);
  point.moi_persistent_sgprs = {};
  point.automatic_moi_private_epoch = true;
  EXPECT_EQ(scratch_count(ConSanMoiEngine::Sampled), 9u);
}

TEST(ConSanPipeline, AtomicScratchSizingComposesModeAndNormalizedTarget) {
  ConSanAtomicLoweringForm form;
  consan_moi_impl::MoiTargetFacts target;
  const auto scratch_count = [&](ConSanMoiEngine engine) {
    return consan_moi_impl::moi_mode_operations(engine).atomic_scratch_vgpr_count(form, target);
  };

  EXPECT_EQ(scratch_count(ConSanMoiEngine::RecordReplay), 7u);
  EXPECT_EQ(scratch_count(ConSanMoiEngine::Sampled), 11u);
  EXPECT_EQ(scratch_count(ConSanMoiEngine::InlineShadow), 26u);
  form.compare_exchange = true;
  EXPECT_EQ(scratch_count(ConSanMoiEngine::RecordReplay), 8u);
  target.requires_aligned_flat_compare_swap_data_pair = true;
  EXPECT_EQ(scratch_count(ConSanMoiEngine::InlineShadow), 28u);
}

TEST(ConSanPipeline, MoiLoweringSummaryConsumesOnlyTypedPatchKindInventory) {
  const std::vector patches{
      ConSanPatchKind::InlineMoiAccessRecordStore,
      ConSanPatchKind::TrampolineMoiAccessRecordStore,
      ConSanPatchKind::InlineMoiExactShadowStore,
      ConSanPatchKind::TrampolineMoiExactShadowStore,
      ConSanPatchKind::InlineMoiSampledWatchpointStore,
      ConSanPatchKind::TrampolineMoiSampledWatchpointStore,
      ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue,
      ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue,
      ConSanPatchKind::TrampolineMoiBarrierRecord,
      ConSanPatchKind::TrampolineMoiBarrierRecord,
      ConSanPatchKind::TrampolineMoiInlineEpochBarrier,
      ConSanPatchKind::TrampolineMoiInlineAtomicOrdering,
      ConSanPatchKind::TrampolineMoiAtomicRecord,
      ConSanPatchKind::TrampolineMoiSampledSyncMetadata,
      ConSanPatchKind::TrampolineMoiFenceRecord,
  };

  EXPECT_EQ(consan_moi_impl::summarize_moi_lowering(ConSanMoiEngine::RecordReplay, true, patches),
            (std::vector<std::string>{
                "ConSan MOI record_replay engine emitted a first-light access record probe",
                "ConSan MOI record_replay engine emitted an appended-cave first-light access "
                "record probe",
                "ConSan MOI inline-shadow engine emitted an exact-shadow publish probe",
                "ConSan MOI inline-shadow engine emitted an appended-cave exact-shadow publish "
                "probe",
                "ConSan MOI sampled engine emitted a direct sampled watchpoint probe",
                "ConSan MOI sampled engine emitted an appended-cave direct sampled watchpoint "
                "probe",
                "ConSan MOI initialized owner/epoch VGPRs with a kernel-entry prologue",
                "ConSan MOI initialized private epoch state with a kernel-entry prologue",
                "ConSan MOI emitted 2 barrier record probe(s)",
                "ConSan MOI record_replay engine emitted 1 barrier epoch probe(s)",
                "ConSan MOI inline-shadow engine emitted 1 inline atomic ordering probe(s)",
                "ConSan MOI emitted 1 atomic record probe(s)",
                "ConSan MOI sampled engine emitted 1 typed synchronization probe(s)",
                "ConSan MOI emitted 1 fence record probe(s)",
            }));
  EXPECT_TRUE(
      consan_moi_impl::summarize_moi_lowering(ConSanMoiEngine::InlineShadow, true, {}).empty());
  EXPECT_EQ(consan_moi_impl::summarize_moi_lowering(ConSanMoiEngine::Sampled, false, {}),
            (std::vector<std::string>{"ConSan MOI sampled engine is an inventory-only stub"}));
}

[[nodiscard]] RuntimeCapabilities complete_runtime_capabilities() {
  return {
      .backend = ConSanRuntimeBackend::PhysicalHsa,
      .host_device_visible_memory = true,
      .host_device_coherent_memory = true,
      .device_atomic_publication = true,
      .max_report_allocation_bytes = 512u * 1024u * 1024u,
      .max_workgroup_lds_bytes = 64u * 1024u,
      .executable_binding = true,
      .dispatch_segment_binding = true,
  };
}

[[nodiscard]] ConSanRequest moi_request(ConSanMoiEngine engine) {
  ConSanRequest request;
  request.flavor = ConSanFlavor::Moi;
  request.moi_engine = engine;
  request.moi_auto_report_buffer_size = 128u * 1024u * 1024u;
  return request;
}

[[nodiscard]] ConSanRequest supercollider_request(bool observe_lds = true) {
  ConSanRequest request;
  request.flavor = ConSanFlavor::SuperCollider;
  request.probe_lds_check_trap = observe_lds;
  return request;
}

[[nodiscard]] RuntimePolicy enabled_runtime_policy() {
  RuntimePolicy policy;
  policy.enabled = true;
  return policy;
}

[[nodiscard]] TransformResult record_replay_inventory_result() {
  return transform_consan(make_rdna4_supported_lds_code_object(),
                          moi_request(ConSanMoiEngine::RecordReplay), TransformPolicy{},
                          enabled_runtime_policy(), ConSanDebugOverrides{},
                          complete_runtime_capabilities(), BoundRuntimeResources{});
}

TEST(ConSanPipeline, DefaultResultFailsClosed) {
  const TransformResult result;
  EXPECT_FALSE(result.well_formed());
  EXPECT_EQ(result.install_action(/*fail_closed=*/false), ConSanInstallAction::LoadOriginal);
  EXPECT_EQ(result.install_action(/*fail_closed=*/true), ConSanInstallAction::Reject);
}

TEST(ConSanPipeline, MutationTallyDistinguishesSelectionAmbiguityFromApplication) {
  ConSanMutationTally tally;
  EXPECT_FALSE(tally.has_plan());
  EXPECT_FALSE(tally.has_ambiguous_plan());
  EXPECT_FALSE(tally.has_application());

  tally.requested = 1u;
  tally.planned = 1u;
  EXPECT_TRUE(tally.has_plan());
  EXPECT_FALSE(tally.has_ambiguous_plan());
  EXPECT_FALSE(tally.has_application());

  tally.planned = 2u;
  EXPECT_TRUE(tally.has_plan());
  EXPECT_TRUE(tally.has_ambiguous_plan());
  EXPECT_FALSE(tally.has_application());

  tally.applied = 2u;
  EXPECT_TRUE(tally.has_application());
  EXPECT_EQ(tally, (ConSanMutationTally{.requested = 1u, .planned = 2u, .applied = 2u}));
}

TEST(ConSanPipeline, MutationOutcomeKeepsFaultAndPerturbationDomainsSeparate) {
  const ConSanMutationOutcome outcome = {
      .fault = {.requested = 1u, .planned = 2u, .applied = 0u},
      .perturbation = {.requested = 3u, .planned = 1u, .applied = 1u},
      .applied_fault_logical_identity = "fault-site",
  };

  EXPECT_TRUE(outcome.fault.has_ambiguous_plan());
  EXPECT_FALSE(outcome.fault.has_application());
  EXPECT_FALSE(outcome.perturbation.has_ambiguous_plan());
  EXPECT_TRUE(outcome.perturbation.has_application());
  EXPECT_EQ(outcome.applied_fault_logical_identity, "fault-site");
  EXPECT_NE(outcome, ConSanMutationOutcome{});
}

TEST(ConSanPipeline, DispatchRequirementsValidatePayloadOrderingAndPacketInterception) {
  ConSanDispatchRequirements requirements;
  EXPECT_TRUE(requirements.well_formed());
  EXPECT_FALSE(requirements.requires_packet_interception());

  requirements.kernels = {{
      .kernel_name = "kernel_a",
      .required_private_bytes = 64u,
      .dynamic_private_addend = 16u,
      .required_group_bytes = 128u,
      .has_instrumented_probe = true,
  }};
  EXPECT_TRUE(requirements.kernels.front().has_segment_requirement());
  EXPECT_TRUE(requirements.kernels.front().well_formed());
  EXPECT_TRUE(requirements.well_formed());
  EXPECT_TRUE(requirements.requires_packet_interception());

  ConSanDispatchRequirements malformed = requirements;
  malformed.kernels.front().kernel_name.clear();
  EXPECT_FALSE(malformed.well_formed());
  malformed = requirements;
  malformed.kernels.front().dynamic_private_addend = 65u;
  EXPECT_FALSE(malformed.well_formed());
  malformed = requirements;
  malformed.kernels = {{.kernel_name = "kernel_b", .has_instrumented_probe = true},
                       {.kernel_name = "kernel_a", .has_instrumented_probe = true}};
  EXPECT_FALSE(malformed.well_formed());
  malformed.kernels[1].kernel_name = "kernel_b";
  EXPECT_FALSE(malformed.well_formed());

  const ConSanKernelDispatchRequirement attribution_only = {
      .kernel_name = "kernel_c",
      .has_instrumented_probe = true,
  };
  EXPECT_FALSE(attribution_only.has_segment_requirement());
  EXPECT_TRUE(attribution_only.well_formed());
  EXPECT_NE(attribution_only, ConSanKernelDispatchRequirement{});
}

TEST(ConSanPipeline, PublicationJoinsTypedCoverageAndSegmentGrowthOncePerKernel) {
  constexpr std::array<uint8_t, 24> bytes{};
  ProgramInventoryBuilder inventory_builder(bytes);
  inventory_builder.set_code_object_facts(true, 0u, ROCJITSU_CODE_ARCH_CDNA4,
                                          ROCJITSU_CODE_TARGET_GFX950);
  ConSanProgramContainer kernel_a{ConSanProgramContainerKind::Kernel};
  kernel_a.name = "kernel_a";
  kernel_a.descriptor_file_offset = 64u;
  kernel_a.entry_text_offset = 0u;
  kernel_a.code_size = 8u;
  kernel_a.has_text_range = true;
  ConSanProgramSite shared_access;
  shared_access.origin = ConSanAccessOrigin::NativeLds;
  shared_access.kind = ConSanLdsAccessKind::Read;
  shared_access.physical_id.original_text_offset = 0u;
  shared_access.decoded_site().text_offset = 0u;
  shared_access.decoded_site().file_offset = 0u;
  shared_access.decoded_site().size = sizeof(uint32_t);
  shared_access.decoded_width_bits = 32u;
  shared_access.decoded_site().mnemonic = "ds_read_b32";
  shared_access.operands.address_vgpr = 0u;
  shared_access.container = kernel_a.id;
  inventory_builder.add_access_site(shared_access);
  inventory_builder.add_kernel(kernel_a);
  ConSanProgramContainer kernel_b{ConSanProgramContainerKind::Kernel};
  kernel_b.name = "kernel_b";
  kernel_b.descriptor_file_offset = 128u;
  kernel_b.entry_text_offset = 8u;
  kernel_b.code_size = 8u;
  kernel_b.has_text_range = true;
  inventory_builder.add_kernel(kernel_b);
  ConSanProgramContainer kernel_c{ConSanProgramContainerKind::Kernel};
  kernel_c.name = "kernel_c";
  kernel_c.descriptor_file_offset = 192u;
  kernel_c.entry_text_offset = 16u;
  kernel_c.code_size = 8u;
  kernel_c.has_text_range = true;
  inventory_builder.add_kernel(kernel_c);
  inventory_builder.publish_decoded_accesses(bytes);
  inventory_builder.access_sites().front().execution_owners = {
      {.kernel = inventory_builder.kernels()[0].id}, {.kernel = inventory_builder.kernels()[1].id}};
  ConSanSyncEvent barrier;
  barrier.semantic_id = {
      .physical = {.code_object = inventory_builder.view().code_object_id(),
                   .original_text_offset = 16u},
      .domain = ConSanSemanticSiteDomain::SynchronizationEvent,
  };
  barrier.kind = ConSanSyncKind::Barrier;
  barrier.operation = ConSanSyncOperation::BarrierFull;
  barrier.identity = "event:24";
  ConSanProgramSite fault_source;
  fault_source.physical_id = barrier.semantic_id.physical;
  fault_source.container = inventory_builder.kernels().back().id;
  fault_source.execution_owners.push_back({.kernel = inventory_builder.kernels()[2].id});
  ConSanAtomicSite atomic_source;
  atomic_source.text_offset = 16u;
  atomic_source.file_offset = 16u;
  atomic_source.size = 8u;
  atomic_source.width_bits = 64u;
  atomic_source.mnemonic = "flat_atomic_add_u64";
  atomic_source.address_vgpr = 2u;
  fault_source.payload = std::move(atomic_source);
  barrier.source_site = {static_cast<uint32_t>(inventory_builder.program_sites().size())};
  inventory_builder.add_semantic_site(std::move(fault_source));
  inventory_builder.synchronization().sync_events.push_back(barrier);
  ConSanSyncSequence barrier_sequence;
  barrier_sequence.identity = "sequence:24";
  barrier_sequence.member_event_ids = {{0}};
  barrier_sequence.operation = ConSanSyncOperation::BarrierFull;
  barrier_sequence.confidence = ConSanSemanticConfidence::Exact;
  barrier_sequence.memory_role = ConSanSyncMemoryRole::Release;
  inventory_builder.synchronization().sync_sequences.push_back(std::move(barrier_sequence));
  const ProgramInventory inventory = inventory_builder.view();
  ASSERT_EQ(inventory.access_sites().size(), 1u);
  ASSERT_EQ(inventory.access_sites().front().ranges.size(), 1u);

  ConSanObservationPlan plan;
  plan.engine = ConSanCapabilityEngine::RecordReplay;
  const SemanticSiteId semantic = inventory.access_sites().front().ranges.front().id;
  plan.probe_intents.push_back({
      .id = {0u},
      .engine = ConSanCapabilityEngine::RecordReplay,
      .physical_site = semantic.physical,
      .covered_semantic_sites = {semantic},
      .kind = ConSanProbeIntentKind::AccessRecord,
      .position = ConSanProbePosition::Before,
      .synchronization_association = std::nullopt,
      .dynamic_result = ConSanDynamicResultRequirement::None,
  });
  plan.site_decisions.push_back({
      .engine = ConSanCapabilityEngine::RecordReplay,
      .semantic_site = semantic,
      .kind = ConSanSiteDecisionKind::Admitted,
      .reason = ConSanAccessPolicyReason::None,
      .intent_ids = {{0u}},
  });
  plan.probe_intents.push_back({
      .id = {1u},
      .engine = ConSanCapabilityEngine::RecordReplay,
      .physical_site = barrier.semantic_id.physical,
      .covered_semantic_sites = {barrier.semantic_id},
      .kind = ConSanProbeIntentKind::BarrierRecord,
      .position = ConSanProbePosition::Before,
      .synchronization_association = std::nullopt,
      .dynamic_result = ConSanDynamicResultRequirement::None,
  });
  plan.barrier_site_decisions.push_back({
      .engine = ConSanCapabilityEngine::RecordReplay,
      .semantic_site = barrier.semantic_id,
      .kind = ConSanSiteDecisionKind::Admitted,
      .reason = ConSanBarrierPolicyReason::None,
      .intent_ids = {{1u}},
  });
  ASSERT_TRUE(plan.valid());
  ConSanCoverageLedger coverage(plan);
  ASSERT_TRUE(
      publish_test_lowering_outcome(coverage, {0u}, ConSanLoweringOutcomeKind::Instrumented));
  ASSERT_TRUE(
      publish_test_lowering_outcome(coverage, {1u}, ConSanLoweringOutcomeKind::Instrumented));

  ConSanTransformArtifacts mechanism;
  mechanism.program_inventory = inventory;
  mechanism.coverage_ledger = coverage;
  mechanism.outcome = ConSanTransformOutcome::ModifiedValid;
  mechanism.replacement = {0x7f, 'E', 'L', 'F'};
  ConSanFaultSite published_fault_site;
  published_fault_site.kind = ConSanFaultSiteKind::Atomic;
  published_fault_site.identity = "published-fault-site";
  published_fault_site.occurrence = 3u;
  published_fault_site.source_site = barrier.source_site;
  published_fault_site.selectable_vgpr_bank_mode = 1u;
  mechanism.fault_sites.push_back(std::move(published_fault_site));

  ConSanBarrierMoveDestination published_destination;
  published_destination.identity = "published-destination";
  published_destination.container_name = "kernel_c";
  published_destination.basic_block_index = 2u;
  published_destination.text_offset = 32u;
  published_destination.file_offset = 48u;
  published_destination.size = 4u;
  published_destination.mnemonic = "v_add_f32";
  published_destination.execution_owners = {
      {.kernel = {1}, .proof = ConSanOwnerProofKind::RecoveredIndirectCall}};
  const ConSanBarrierMoveDestinationPresentation expected_destination = published_destination;
  mechanism.barrier_move_destinations.push_back(std::move(published_destination));

  ConSanFaultMutationPlan published_fault_plan;
  published_fault_plan.kind = ConSanFaultMutationKind::DropBarrier;
  published_fault_plan.primary_identity = "published-fault-plan";
  const ConSanFaultMutationPresentation expected_fault_plan = published_fault_plan;
  mechanism.fault_plans.push_back(std::move(published_fault_plan));
  mechanism.mutation.fault = {.requested = 1u, .planned = 1u, .applied = 1u};
  mechanism.resource_plans.emplace_back().candidate_index = 7u;
  mechanism.warnings.emplace_back("published-warning");
  ConSanPatchInfo shared_segments;
  shared_segments.kind = ConSanPatchKind::InlineNopRewrite;
  shared_segments.required_private_segment_size = 40u;
  shared_segments.dynamic_private_segment_addend = 8u;
  shared_segments.workgroup_shadow.emplace().required_group_segment_size = 100u;
  shared_segments.owner_descriptor_file_offsets = {64u, 128u};
  mechanism.patches.push_back(shared_segments);
  ConSanPatchInfo kernel_a_segments = shared_segments;
  kernel_a_segments.required_private_segment_size = 64u;
  kernel_a_segments.dynamic_private_segment_addend = 16u;
  kernel_a_segments.workgroup_shadow->required_group_segment_size = 80u;
  kernel_a_segments.owner_descriptor_file_offsets = {64u};
  mechanism.patches.push_back(kernel_a_segments);
  ConSanPatchInfo legacy_kernel_b_segments = shared_segments;
  legacy_kernel_b_segments.anchor_offset = 12u;
  legacy_kernel_b_segments.required_private_segment_size = 56u;
  legacy_kernel_b_segments.dynamic_private_segment_addend = 0u;
  legacy_kernel_b_segments.workgroup_shadow->required_group_segment_size = 120u;
  legacy_kernel_b_segments.owner_descriptor_file_offsets.clear();
  mechanism.patches.push_back(legacy_kernel_b_segments);

  BoundRuntimeResources publication_resources;
  publication_resources.scope = ConSanRuntimeResourceScope::Executable;
  publication_resources.moi_report_buffer_address = 0x123456780000ull;
  publication_resources.moi_report_buffer_size = 128u * 1024u * 1024u;
  ConSanTransformArtifacts invalid_fault_source = mechanism;
  invalid_fault_source.fault_sites.front().source_site = {};
  ConSanTransformArtifacts invalid_execution_owner = mechanism;
  ProgramInventoryBuilder invalid_owner_builder(inventory);
  invalid_owner_builder.access_sites().front().execution_owners.front().kernel = {99};
  invalid_execution_owner.program_inventory = invalid_owner_builder.view();
  const TransformResult published = TransformResultTestAccess::publish(
      bytes, moi_request(ConSanMoiEngine::RecordReplay), TransformPolicy{},
      enabled_runtime_policy(), ConSanDebugOverrides{}, MutationRequest{},
      complete_runtime_capabilities(), publication_resources, std::move(mechanism));

  ASSERT_TRUE(published.well_formed()) << testing::PrintToString(published.errors);
  EXPECT_EQ(published.replacement, (std::vector<uint8_t>{0x7f, 'E', 'L', 'F'}));
  const ConSanTransformDiagnosticReport published_diagnostics =
      TransformResultTestAccess::diagnostic_report(published);
  ASSERT_EQ(published_diagnostics.fault_sites.size(), 1u);
  const ConSanFaultSiteDiagnostic &fault_diagnostic = published_diagnostics.fault_sites.front();
  EXPECT_EQ(fault_diagnostic.kind, ConSanFaultSiteKind::Atomic);
  EXPECT_EQ(fault_diagnostic.identity, "published-fault-site");
  EXPECT_EQ(fault_diagnostic.container_name, "kernel_c");
  EXPECT_TRUE(fault_diagnostic.in_kernel);
  EXPECT_EQ(fault_diagnostic.occurrence, 3u);
  EXPECT_EQ(fault_diagnostic.text_offset, 16u);
  EXPECT_EQ(fault_diagnostic.file_offset, 16u);
  EXPECT_EQ(fault_diagnostic.size, 8u);
  EXPECT_EQ(fault_diagnostic.width_bits, 64u);
  EXPECT_EQ(fault_diagnostic.mnemonic, "flat_atomic_add_u64");
  EXPECT_EQ(fault_diagnostic.semantic_role, "workgroup-barrier");
  EXPECT_EQ(fault_diagnostic.decoded_operands, "addr_vgpr=2");
  EXPECT_EQ(fault_diagnostic.sync_confidence, ConSanSemanticConfidence::Exact);
  EXPECT_EQ(fault_diagnostic.sync_memory_role, ConSanSyncMemoryRole::Release);
  EXPECT_EQ(fault_diagnostic.execution_owners, std::vector<ConSanExecutionOwner>{{.kernel = {2}}});
  EXPECT_EQ(fault_diagnostic.sync_event_identity, "event:24");
  EXPECT_EQ(fault_diagnostic.sync_sequence_identity, "sequence:24");
  const TransformResult malformed_fault_source = TransformResultTestAccess::publish(
      bytes, moi_request(ConSanMoiEngine::RecordReplay), TransformPolicy{},
      enabled_runtime_policy(), ConSanDebugOverrides{}, MutationRequest{},
      complete_runtime_capabilities(), publication_resources, std::move(invalid_fault_source));
  EXPECT_FALSE(malformed_fault_source.well_formed());
  const TransformResult malformed_execution_owner = TransformResultTestAccess::publish(
      bytes, moi_request(ConSanMoiEngine::RecordReplay), TransformPolicy{},
      enabled_runtime_policy(), ConSanDebugOverrides{}, MutationRequest{},
      complete_runtime_capabilities(), publication_resources, std::move(invalid_execution_owner));
  EXPECT_FALSE(malformed_execution_owner.well_formed());
  ASSERT_EQ(published_diagnostics.barrier_move_destinations.size(), 1u);
  EXPECT_EQ(published_diagnostics.barrier_move_destinations.front(), expected_destination);
  ASSERT_EQ(published_diagnostics.fault_mutations.size(), 1u);
  EXPECT_EQ(published_diagnostics.fault_mutations.front(), expected_fault_plan);
  TransformResult malformed_fault_plan = published;
  malformed_fault_plan.mutation.fault.planned = 0u;
  EXPECT_FALSE(malformed_fault_plan.well_formed());
  EXPECT_EQ(published_diagnostics.resource_summary.unsupported_plans, 1u);
  EXPECT_EQ(published.mutation.fault,
            (ConSanMutationTally{.requested = 1u, .planned = 1u, .applied = 1u}));
  EXPECT_EQ(published.warnings, std::vector<std::string>{"published-warning"});
  ASSERT_EQ(published_diagnostics.patches.size(), 3u);
  EXPECT_EQ(published_diagnostics.patches.front().kind, "inline-nop-rewrite");
  EXPECT_EQ(published_diagnostics.patches[1].required_private_segment_size, 64u);
  EXPECT_EQ(published_diagnostics.patches[1].dynamic_private_segment_addend, 16u);
  EXPECT_EQ(published_diagnostics.patches[2].anchor_offset, 12u);
  EXPECT_EQ(published_diagnostics.resource_summary.emitted_spill_patches, 0u);
  ASSERT_EQ(published.dispatch_requirements.kernels.size(), 3u);
  EXPECT_EQ(published.dispatch_requirements.kernels[0], (ConSanKernelDispatchRequirement{
                                                            .kernel_name = "kernel_a",
                                                            .required_private_bytes = 64u,
                                                            .dynamic_private_addend = 16u,
                                                            .required_group_bytes = 100u,
                                                            .has_instrumented_probe = true,
                                                        }));
  EXPECT_EQ(published.dispatch_requirements.kernels[1], (ConSanKernelDispatchRequirement{
                                                            .kernel_name = "kernel_b",
                                                            .required_private_bytes = 56u,
                                                            .dynamic_private_addend = 8u,
                                                            .required_group_bytes = 120u,
                                                            .has_instrumented_probe = true,
                                                        }));
  EXPECT_EQ(published.dispatch_requirements.kernels[2], (ConSanKernelDispatchRequirement{
                                                            .kernel_name = "kernel_c",
                                                            .has_instrumented_probe = true,
                                                        }));
  EXPECT_TRUE(published.dispatch_requirements.requires_packet_interception());
}

TEST(ConSanPipeline, InvalidConfigurationStopsBeforeTargetLoweringWithTypedIssue) {
  ConSanRequest request = moi_request(ConSanMoiEngine::RecordReplay);
  request.moi_sample_stride = 0;
  constexpr std::array<uint8_t, 4> bytes = {0x7f, 'E', 'L', 'F'};
  TransformResult result =
      transform_consan(bytes, request, TransformPolicy{}, enabled_runtime_policy(),
                       ConSanDebugOverrides{}, complete_runtime_capabilities(), {});

  ASSERT_TRUE(result.well_formed());
  EXPECT_EQ(result.contract_issue, ConSanContractIssue::InvalidSampleStride);
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::Invalid);
  EXPECT_TRUE(result.errors.empty());

  EXPECT_TRUE(result.program_inventory.empty());
  EXPECT_FALSE(result.program_inventory.code_object_parsed());
}

TEST(ConSanPipeline, MissingRuntimeBackendStopsAtCapabilityBoundary) {
  TransformResult result = transform_consan(
      make_rdna4_supported_lds_code_object(), moi_request(ConSanMoiEngine::RecordReplay),
      TransformPolicy{}, enabled_runtime_policy(), ConSanDebugOverrides{}, RuntimeCapabilities{},
      BoundRuntimeResources{});

  ASSERT_TRUE(result.well_formed());
  EXPECT_EQ(result.contract_issue, ConSanContractIssue::MissingRuntimeBackend);
  EXPECT_TRUE(result.errors.empty());
}

TEST(ConSanPipeline, InvalidCodeObjectRetainsOneResultIdentity) {
  constexpr std::array<uint8_t, 4> bytes = {0x7f, 'E', 'L', 'F'};
  TransformResult result = transform_consan(
      bytes, moi_request(ConSanMoiEngine::RecordReplay), TransformPolicy{},
      enabled_runtime_policy(), ConSanDebugOverrides{}, complete_runtime_capabilities(), {});

  ASSERT_TRUE(result.well_formed());
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::Invalid);
  EXPECT_TRUE(result.code_object.valid());
  EXPECT_FALSE(result.errors.empty());
}

TEST(ConSanPipeline, RuntimeFailurePolicyDoesNotChangeStaticTransform) {
  constexpr std::array<uint8_t, 4> bytes = {0x7f, 'E', 'L', 'F'};
  RuntimePolicy fail_open = enabled_runtime_policy();
  RuntimePolicy fail_closed = fail_open;
  fail_closed.fail_closed = true;

  const TransformResult open =
      transform_consan(bytes, moi_request(ConSanMoiEngine::RecordReplay), TransformPolicy{},
                       fail_open, ConSanDebugOverrides{}, complete_runtime_capabilities(), {});
  const TransformResult closed =
      transform_consan(bytes, moi_request(ConSanMoiEngine::RecordReplay), TransformPolicy{},
                       fail_closed, ConSanDebugOverrides{}, complete_runtime_capabilities(), {});

  ASSERT_TRUE(open.well_formed()) << testing::PrintToString(open.errors);
  ASSERT_TRUE(closed.well_formed()) << testing::PrintToString(closed.errors);
  EXPECT_EQ(open.code_object, closed.code_object);
  EXPECT_EQ(open.program_inventory.code_object_id(), closed.program_inventory.code_object_id());
  EXPECT_EQ(open.program_inventory.code_object_parsed(),
            closed.program_inventory.code_object_parsed());
  EXPECT_EQ(open.program_inventory.arch(), closed.program_inventory.arch());
  EXPECT_EQ(open.program_inventory.target(), closed.program_inventory.target());
  EXPECT_EQ(open.observation_plan(), closed.observation_plan());
  EXPECT_EQ(open.coverage_ledger, closed.coverage_ledger);
  EXPECT_EQ(open.replacement, closed.replacement);
  EXPECT_EQ(open.outcome, closed.outcome);
  EXPECT_EQ(open.errors, closed.errors);
  EXPECT_EQ(open.warnings, closed.warnings);
  EXPECT_EQ(open.mutation, closed.mutation);
  EXPECT_EQ(open.install_action(/*fail_closed=*/false), ConSanInstallAction::LoadOriginal);
  EXPECT_EQ(closed.install_action(/*fail_closed=*/true), ConSanInstallAction::Reject);
}

TEST(ConSanPipeline, EveryEnginePublishesItsTypedEvidenceContractBeforeBinding) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  const auto check = [&]<typename ExpectedEvidence>(const ConSanRequest &request) {
    TransformResult result = transform_consan(
        bytes, request, TransformPolicy{}, enabled_runtime_policy(), ConSanDebugOverrides{},
        complete_runtime_capabilities(), BoundRuntimeResources{});
    ASSERT_TRUE(result.well_formed()) << testing::PrintToString(result.errors);
    EXPECT_TRUE(result.observation_plan().valid());
    ASSERT_TRUE(result.evidence_requirements) << testing::PrintToString(result.errors);
    EXPECT_TRUE(std::holds_alternative<ExpectedEvidence>(*result.evidence_requirements));
  };

  check.operator()<ConSanRecordReplayEvidenceRequirements>(
      moi_request(ConSanMoiEngine::RecordReplay));
  check.operator()<ConSanSampledEvidenceRequirements>(moi_request(ConSanMoiEngine::Sampled));
  check.operator()<ConSanInlineShadowEvidenceRequirements>(
      moi_request(ConSanMoiEngine::InlineShadow));
  check.operator()<ConSanSuperColliderEvidenceRequirements>(supercollider_request());
}

TEST(ConSanPipeline, MoiEvidenceCapacityComesDirectlyFromTypedRequestPolicyAndCapabilities) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  RuntimeCapabilities capabilities = complete_runtime_capabilities();
  capabilities.max_workgroup_lds_bytes = 96u * 1024u;

  for (const bool expert_limit : {false, true}) {
    TransformPolicy transform_policy;
    transform_policy.max_patches = 7u;
    transform_policy.max_patches_is_expert_limit = expert_limit;
    const std::optional<uint64_t> maximum_access_probe_count =
        expert_limit ? std::optional<uint64_t>{7u} : std::nullopt;

    for (const ConSanMoiEngine engine :
         {ConSanMoiEngine::RecordReplay, ConSanMoiEngine::Sampled, ConSanMoiEngine::InlineShadow}) {
      SCOPED_TRACE(consan_moi_engine_name(engine));
      SCOPED_TRACE(expert_limit);
      ConSanRequest request = moi_request(engine);
      request.moi_auto_report_buffer_size = 4096u;
      const TransformResult result =
          transform_consan(bytes, request, transform_policy, enabled_runtime_policy(),
                           ConSanDebugOverrides{}, capabilities, BoundRuntimeResources{});

      ASSERT_TRUE(result.evidence_requirements) << testing::PrintToString(result.errors);
      const ConSanEvidenceIntentPlan evidence_intents =
          plan_consan_evidence_intents(result.observation_plan());
      switch (engine) {
      case ConSanMoiEngine::RecordReplay:
        EXPECT_EQ(
            std::get<ConSanRecordReplayEvidenceRequirements>(*result.evidence_requirements),
            plan_consan_record_replay_evidence(
                evidence_intents, {.caller_ceiling_bytes = 4096u,
                                   .maximum_access_probe_count = maximum_access_probe_count}));
        break;
      case ConSanMoiEngine::Sampled:
        EXPECT_EQ(std::get<ConSanSampledEvidenceRequirements>(*result.evidence_requirements),
                  plan_consan_sampled_evidence(evidence_intents, {.caller_ceiling_bytes = 4096u,
                                                                  .maximum_access_probe_count =
                                                                      maximum_access_probe_count}));
        break;
      case ConSanMoiEngine::InlineShadow:
        EXPECT_EQ(std::get<ConSanInlineShadowEvidenceRequirements>(*result.evidence_requirements),
                  plan_consan_inline_shadow_evidence(
                      result.program_inventory, evidence_intents,
                      {.caller_ceiling_bytes = 4096u,
                       .maximum_access_probe_count = maximum_access_probe_count,
                       .maximum_workgroup_lds_bytes = 96u * 1024u}));
        break;
      }
    }
  }
}

TEST(ConSanPipeline, EmptyPlansNeedNoRuntimeBindingForAnyEngine) {
  constexpr std::array<uint32_t, 1> kTextWords = {0xBFB00000u};
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(kTextWords, "typed_empty_observation_plan");
  const std::array requests = {
      moi_request(ConSanMoiEngine::RecordReplay),
      moi_request(ConSanMoiEngine::Sampled),
      moi_request(ConSanMoiEngine::InlineShadow),
      supercollider_request(),
  };
  for (const ConSanRequest &request : requests) {
    TransformResult result = transform_consan(
        bytes, request, TransformPolicy{}, enabled_runtime_policy(), ConSanDebugOverrides{},
        complete_runtime_capabilities(), BoundRuntimeResources{});
    ASSERT_TRUE(result.well_formed()) << testing::PrintToString(result.errors);
    ASSERT_TRUE(result.evidence_requirements);
    EXPECT_FALSE(
        std::visit([](const auto &requirements) { return requirements.requires_binding(); },
                   *result.evidence_requirements));
    EXPECT_EQ(result.outcome, ConSanTransformOutcome::Unchanged);
  }
}

TEST(ConSanPipeline, ConcreteBindingChecksRuntimeFactsAndLifetimeScope) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  const ConSanRequest request = moi_request(ConSanMoiEngine::RecordReplay);
  TransformResult inventory =
      transform_consan(bytes, request, TransformPolicy{}, enabled_runtime_policy(),
                       ConSanDebugOverrides{}, complete_runtime_capabilities(), {});
  ASSERT_TRUE(inventory.evidence_requirements);
  const auto *requirements =
      std::get_if<ConSanRecordReplayEvidenceRequirements>(&*inventory.evidence_requirements);
  ASSERT_NE(requirements, nullptr);
  ASSERT_TRUE(requirements->complete());

  BoundRuntimeResources bound;
  bound.scope = ConSanRuntimeResourceScope::Executable;
  bound.moi_report_buffer_address = 0x123456780000ull;
  bound.moi_report_buffer_size = requirements->abi_plan.required_bytes;
  bound.moi_report_layout = requirements->abi_plan.layout;
  TransformResult complete =
      transform_consan(bytes, request, TransformPolicy{}, enabled_runtime_policy(),
                       ConSanDebugOverrides{}, complete_runtime_capabilities(), bound);
  ASSERT_TRUE(complete.well_formed()) << testing::PrintToString(complete.errors);
  EXPECT_EQ(complete.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_EQ(complete.contract_issue, ConSanContractIssue::None);

  RuntimeCapabilities missing_visibility = complete_runtime_capabilities();
  missing_visibility.host_device_visible_memory = false;
  TransformResult rejected =
      transform_consan(bytes, request, TransformPolicy{}, enabled_runtime_policy(),
                       ConSanDebugOverrides{}, missing_visibility, bound);
  ASSERT_TRUE(rejected.well_formed()) << testing::PrintToString(rejected.errors);
  EXPECT_EQ(rejected.outcome, ConSanTransformOutcome::Unsupported);
  EXPECT_EQ(rejected.contract_issue, ConSanContractIssue::MissingVisibleMemory);

  BoundRuntimeResources undersized = bound;
  --undersized.moi_report_buffer_size;
  TransformResult rejected_size =
      transform_consan(bytes, request, TransformPolicy{}, enabled_runtime_policy(),
                       ConSanDebugOverrides{}, complete_runtime_capabilities(), undersized);
  ASSERT_TRUE(rejected_size.well_formed()) << testing::PrintToString(rejected_size.errors);
  EXPECT_EQ(rejected_size.contract_issue, ConSanContractIssue::InvalidResourceSize);

  BoundRuntimeResources explicit_fixed;
  explicit_fixed.scope = ConSanRuntimeResourceScope::CodeObject;
  explicit_fixed.moi_report_buffer_address = 0x123456780000ull;
  explicit_fixed.moi_report_buffer_size =
      sizeof(ConSanMoiReportHeader) + sizeof(ConSanMoiAccessRecord);
  TransformResult fixed_complete =
      transform_consan(bytes, request, TransformPolicy{}, enabled_runtime_policy(),
                       ConSanDebugOverrides{}, complete_runtime_capabilities(), explicit_fixed);
  ASSERT_TRUE(fixed_complete.well_formed()) << testing::PrintToString(fixed_complete.errors);
  EXPECT_EQ(fixed_complete.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_EQ(fixed_complete.contract_issue, ConSanContractIssue::None);

  BoundRuntimeResources dispatch_scoped = explicit_fixed;
  dispatch_scoped.scope = ConSanRuntimeResourceScope::Dispatch;
  TransformResult rejected_scope =
      transform_consan(bytes, request, TransformPolicy{}, enabled_runtime_policy(),
                       ConSanDebugOverrides{}, complete_runtime_capabilities(), dispatch_scoped);
  ASSERT_TRUE(rejected_scope.well_formed()) << testing::PrintToString(rejected_scope.errors);
  EXPECT_EQ(rejected_scope.contract_issue, ConSanContractIssue::InvalidResourceScope);

  BoundRuntimeResources wrong_schema;
  wrong_schema.scope = ConSanRuntimeResourceScope::Executable;
  wrong_schema.report_buffer_address = 0x123456780000ull;
  TransformResult rejected_schema =
      transform_consan(bytes, request, TransformPolicy{}, enabled_runtime_policy(),
                       ConSanDebugOverrides{}, complete_runtime_capabilities(), wrong_schema);
  ASSERT_TRUE(rejected_schema.well_formed()) << testing::PrintToString(rejected_schema.errors);
  EXPECT_EQ(rejected_schema.contract_issue, ConSanContractIssue::InvalidResourceAddress);
}

TEST(ConSanPipeline, SuperColliderBindingRequiresItsStickyMarkerAddress) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  const ConSanRequest request = supercollider_request();

  BoundRuntimeResources marker;
  marker.scope = ConSanRuntimeResourceScope::Executable;
  marker.report_buffer_address = 0x123456780000ull;
  TransformResult complete =
      transform_consan(bytes, request, TransformPolicy{}, enabled_runtime_policy(),
                       ConSanDebugOverrides{}, complete_runtime_capabilities(), marker);
  ASSERT_TRUE(complete.well_formed()) << testing::PrintToString(complete.errors);
  EXPECT_EQ(complete.contract_issue, ConSanContractIssue::None);

  BoundRuntimeResources code_object_marker = marker;
  code_object_marker.scope = ConSanRuntimeResourceScope::CodeObject;
  TransformResult code_object_complete =
      transform_consan(bytes, request, TransformPolicy{}, enabled_runtime_policy(),
                       ConSanDebugOverrides{}, complete_runtime_capabilities(), code_object_marker);
  ASSERT_TRUE(code_object_complete.well_formed())
      << testing::PrintToString(code_object_complete.errors);
  EXPECT_EQ(code_object_complete.contract_issue, ConSanContractIssue::None);

  BoundRuntimeResources wrong_schema;
  wrong_schema.scope = ConSanRuntimeResourceScope::Executable;
  wrong_schema.moi_report_buffer_address = 0x123456780000ull;
  wrong_schema.moi_report_buffer_size = sizeof(ConSanMoiReportHeader);
  TransformResult rejected =
      transform_consan(bytes, request, TransformPolicy{}, enabled_runtime_policy(),
                       ConSanDebugOverrides{}, complete_runtime_capabilities(), wrong_schema);
  ASSERT_TRUE(rejected.well_formed()) << testing::PrintToString(rejected.errors);
  EXPECT_EQ(rejected.contract_issue, ConSanContractIssue::InvalidResourceAddress);
}

TEST(ConSanPipeline, ResultValidatorRejectsEveryOwnedCrossTypeInvariant) {
  const TransformResult good = record_replay_inventory_result();
  ASSERT_TRUE(good.well_formed()) << testing::PrintToString(good.errors);

  TransformResult malformed = good;
  malformed.contract_issue = ConSanContractIssue::MissingFlavor;
  EXPECT_FALSE(malformed.well_formed());
  malformed = good;
  ProgramInventoryBuilder foreign(std::array<uint8_t, 2>{7, 8});
  malformed.program_inventory = foreign.view();
  EXPECT_FALSE(malformed.well_formed());
  malformed = good;
  malformed.evidence_requirements.reset();
  EXPECT_FALSE(malformed.well_formed());
  malformed = good;
  ASSERT_TRUE(malformed.evidence_requirements);
  std::get<ConSanRecordReplayEvidenceRequirements>(*malformed.evidence_requirements)
      .runtime_requirements.executable_binding = false;
  EXPECT_FALSE(malformed.well_formed());
  malformed = good;
  malformed.errors.emplace_back();
  EXPECT_FALSE(malformed.well_formed());
  malformed = good;
  malformed.dispatch_requirements.kernels.push_back({});
  EXPECT_FALSE(malformed.well_formed());
  malformed = good;
  malformed.outcome = ConSanTransformOutcome::ModifiedValid;
  EXPECT_FALSE(malformed.well_formed());
  malformed = good;
  malformed.replacement.push_back(1);
  EXPECT_FALSE(malformed.well_formed());
}

TEST(ConSanPipeline, InstallActionTruthTableUsesOnlySplitStaticResult) {
  TransformResult result;
  result.outcome = ConSanTransformOutcome::Unchanged;
  EXPECT_EQ(result.install_action(false), ConSanInstallAction::LoadOriginal);
  EXPECT_EQ(result.install_action(true), ConSanInstallAction::LoadOriginal);

  result.outcome = ConSanTransformOutcome::Unsupported;
  EXPECT_EQ(result.install_action(false), ConSanInstallAction::LoadOriginal);
  EXPECT_EQ(result.install_action(true), ConSanInstallAction::Reject);
  result.outcome = ConSanTransformOutcome::Invalid;
  EXPECT_EQ(result.install_action(false), ConSanInstallAction::LoadOriginal);
  EXPECT_EQ(result.install_action(true), ConSanInstallAction::Reject);

  result.outcome = ConSanTransformOutcome::ModifiedValid;
  result.replacement = {1};
  result.replacement.clear();
  EXPECT_EQ(result.install_action(false), ConSanInstallAction::Reject);
  result.replacement = {1};
  EXPECT_EQ(result.install_action(false), ConSanInstallAction::LoadReplacement);
  EXPECT_EQ(result.install_action(true), ConSanInstallAction::LoadReplacement);
}

TEST(ConSanPipeline, ProductionResultOwnsAllPublishedTransformArtifacts) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  const ConSanRequest request = moi_request(ConSanMoiEngine::RecordReplay);
  const TransformPolicy transform_policy;
  const RuntimePolicy runtime_policy = enabled_runtime_policy();
  const ConSanDebugOverrides debug;
  const MutationRequest mutation;
  const RuntimeCapabilities capabilities = complete_runtime_capabilities();
  BoundRuntimeResources resources;
  resources.scope = ConSanRuntimeResourceScope::Executable;
  resources.moi_report_buffer_address = 0x123456780000ull;
  resources.moi_report_buffer_size = 64u * 1024u * 1024u;

  TransformResult split = transform_consan(bytes, request, transform_policy, runtime_policy, debug,
                                           capabilities, resources);
  ASSERT_TRUE(split.well_formed()) << testing::PrintToString(split.errors);

  EXPECT_TRUE(split.program_inventory.code_object_parsed());
  EXPECT_EQ(split.program_inventory.code_object_id(), split.code_object);
  EXPECT_FALSE(split.observation_plan().probe_intents.empty());
  EXPECT_EQ(split.coverage_ledger.intent_entries().size(),
            split.observation_plan().probe_intents.size());
  EXPECT_FALSE(split.replacement.empty());
  EXPECT_FALSE(TransformResultTestAccess::diagnostic_report(split).patches.empty());
  EXPECT_NE(TransformResultTestAccess::diagnostic_report(split).resource_summary,
            ConSanResourcePlanSummary{});
  EXPECT_EQ(split.install_action(false), split.outcome == ConSanTransformOutcome::ModifiedValid
                                             ? ConSanInstallAction::LoadReplacement
                                             : ConSanInstallAction::LoadOriginal);
}

TEST(ConSanPipeline, ExtendedBarrierInventoryShapeUsesOnlyTypedSemanticInputs) {
  ConSanRequest request = moi_request(ConSanMoiEngine::RecordReplay);
  ConSanDebugOverrides debug;
  MutationRequest mutation;
  EXPECT_FALSE(consan_requires_extended_barrier_pairs(request, debug, mutation));

  mutation.fault_drop_barrier = true;
  EXPECT_TRUE(consan_requires_extended_barrier_pairs(request, debug, mutation));
  mutation = {};
  mutation.fault_move_barrier = true;
  EXPECT_TRUE(consan_requires_extended_barrier_pairs(request, debug, mutation));
  mutation = {};
  mutation.fault_mutate_barrier_id_scope = true;
  EXPECT_TRUE(consan_requires_extended_barrier_pairs(request, debug, mutation));
  mutation = {};
  mutation.fault_mutate_barrier_participants = true;
  EXPECT_TRUE(consan_requires_extended_barrier_pairs(request, debug, mutation));

  mutation = {};
  debug.abort_unmatched_barrier_wait = true;
  EXPECT_TRUE(consan_requires_extended_barrier_pairs(request, debug, mutation));
  debug = {};
  request.moi_track_barriers = true;
  EXPECT_FALSE(consan_requires_extended_barrier_pairs(request, debug, mutation));
  request.moi_engine = ConSanMoiEngine::Sampled;
  EXPECT_TRUE(consan_requires_extended_barrier_pairs(request, debug, mutation));
  request.moi_track_barriers = false;
  EXPECT_FALSE(consan_requires_extended_barrier_pairs(request, debug, mutation));
}

TEST(ConSanPipeline, AutomaticMoiPreparationUsesTypedRequestShapeWithoutBinding) {
  std::array<uint32_t, 17> text_words{};
  text_words[0] = 0xBE804EC1u; // s_barrier_signal -1
  std::fill(text_words.begin() + 1, text_words.begin() + 15,
            build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words[15] = 0xBF94FFFFu; // s_barrier_wait -1
  text_words[16] = 0xBFB00000u;
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "typed_pristine_extended_barrier_pair");

  ConSanRequest request = moi_request(ConSanMoiEngine::RecordReplay);
  request.moi_track_barriers = true;
  TransformPolicy transform_policy;
  transform_policy.max_patches = 16;

  ConSanAutomaticTransformPreparation preparation = prepare_consan_automatic_transform(
      bytes, request, transform_policy, enabled_runtime_policy(), ConSanDebugOverrides{},
      MutationRequest{}, complete_runtime_capabilities());
  ASSERT_TRUE(std::holds_alternative<ConSanDeferredBinding>(preparation));
  const ConSanDeferredBinding &inventory = std::get<ConSanDeferredBinding>(preparation);
  ASSERT_TRUE(inventory.well_formed());

  const auto sync = inventory.program_inventory().sync().sync_sequences;
  EXPECT_EQ(
      std::ranges::count(sync, ConSanSyncOperation::BarrierFull, &ConSanSyncSequence::operation),
      0u);
  ASSERT_FALSE(sync.empty());
  EXPECT_TRUE(inventory.evidence_requirements());
}

TEST(ConSanPipeline, AutomaticMoiResumeRemainsInsideTypedPipelineBoundary) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  const ConSanRequest request = moi_request(ConSanMoiEngine::RecordReplay);
  const TransformPolicy transform_policy;
  const RuntimePolicy runtime_policy = enabled_runtime_policy();
  const ConSanDebugOverrides debug;
  const MutationRequest mutation;
  const RuntimeCapabilities capabilities = complete_runtime_capabilities();

  ConSanAutomaticTransformPreparation preparation = prepare_consan_automatic_transform(
      bytes, request, transform_policy, runtime_policy, debug, mutation, capabilities);
  ASSERT_TRUE(std::holds_alternative<ConSanDeferredBinding>(preparation));
  const ConSanDeferredBinding &inventory = std::get<ConSanDeferredBinding>(preparation);
  ASSERT_TRUE(inventory.well_formed());
  ASSERT_TRUE(inventory.evidence_requirements());
  const auto &requirements =
      std::get<ConSanRecordReplayEvidenceRequirements>(*inventory.evidence_requirements());

  BoundRuntimeResources resources;
  resources.scope = ConSanRuntimeResourceScope::Executable;
  resources.moi_report_buffer_address = 0x123456780000ull;
  resources.moi_report_buffer_size = requirements.abi_plan.required_bytes;
  TransformResult retried = resume_consan_automatic_transform(
      bytes, resources, std::move(std::get<ConSanDeferredBinding>(preparation)));
  const TransformResult direct = transform_consan(bytes, request, transform_policy, runtime_policy,
                                                  debug, capabilities, resources);

  ASSERT_TRUE(retried.well_formed()) << testing::PrintToString(retried.errors);
  ASSERT_EQ(retried.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_EQ(retried.install_action(false), ConSanInstallAction::LoadReplacement);
  EXPECT_EQ(retried.code_object, direct.code_object);
  EXPECT_EQ(retried.observation_plan(), direct.observation_plan());
  EXPECT_EQ(retried.coverage_ledger, direct.coverage_ledger);
  EXPECT_EQ(retried.replacement, direct.replacement);
  const ConSanTransformDiagnosticReport retried_diagnostics =
      TransformResultTestAccess::diagnostic_report(retried);
  const ConSanTransformDiagnosticReport direct_diagnostics =
      TransformResultTestAccess::diagnostic_report(direct);
  EXPECT_EQ(retried_diagnostics.patches.size(), direct_diagnostics.patches.size());
  EXPECT_EQ(retried_diagnostics.resource_summary, direct_diagnostics.resource_summary);
  EXPECT_EQ(retried_diagnostics.fault_sites.size(), direct_diagnostics.fault_sites.size());
  EXPECT_EQ(retried_diagnostics.barrier_move_destinations.size(),
            direct_diagnostics.barrier_move_destinations.size());
  EXPECT_EQ(retried_diagnostics.fault_mutations.size(), direct_diagnostics.fault_mutations.size());
}

TEST(ConSanPipeline, AutomaticDynamicReplayTreatsExplicitCapAsRuntimeRingCapacity) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  ConSanRequest request = moi_request(ConSanMoiEngine::RecordReplay);
  request.moi_dynamic_access_records = true;
  request.moi_auto_report_buffer_size =
      consan_moi_report_buffer_min_bytes(/*access_count=*/1u, /*diagnostic_count=*/0u,
                                         /*barrier_count=*/0u, /*atomic_count=*/0u);
  TransformPolicy transform_policy;
  transform_policy.max_patches = 2u;
  transform_policy.max_patches_is_expert_limit = true;

  ConSanAutomaticTransformPreparation preparation = prepare_consan_automatic_transform(
      bytes, request, transform_policy, enabled_runtime_policy(), ConSanDebugOverrides{},
      MutationRequest{}, complete_runtime_capabilities());
  ASSERT_TRUE(std::holds_alternative<ConSanDeferredBinding>(preparation));
  const ConSanDeferredBinding &deferred = std::get<ConSanDeferredBinding>(preparation);
  ASSERT_TRUE(deferred.well_formed());
  ASSERT_TRUE(deferred.evidence_requirements());
  const auto &requirements =
      std::get<ConSanRecordReplayEvidenceRequirements>(*deferred.evidence_requirements());
  EXPECT_TRUE(requirements.complete());
  EXPECT_GT(requirements.abi_plan.required_bytes, request.moi_auto_report_buffer_size);

  BoundRuntimeResources resources;
  resources.scope = ConSanRuntimeResourceScope::Executable;
  resources.moi_report_buffer_address = 0x123456780000ull;
  resources.moi_report_buffer_size = request.moi_auto_report_buffer_size;
  const TransformResult resumed = resume_consan_automatic_transform(
      bytes, resources, std::move(std::get<ConSanDeferredBinding>(preparation)));

  ASSERT_TRUE(resumed.well_formed()) << testing::PrintToString(resumed.errors);
  EXPECT_EQ(resumed.outcome, ConSanTransformOutcome::ModifiedValid)
      << testing::PrintToString(resumed.warnings);
  EXPECT_EQ(resumed.install_action(false), ConSanInstallAction::LoadReplacement);
  EXPECT_EQ(resumed.contract_issue, ConSanContractIssue::None);
}

TEST(ConSanPipeline, AutomaticMoiBindingPublishesImmutableTokenAndLibraryOwnedResume) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  const ConSanRequest request = moi_request(ConSanMoiEngine::RecordReplay);
  const TransformPolicy transform_policy;
  const RuntimePolicy runtime_policy = enabled_runtime_policy();
  const ConSanDebugOverrides debug;
  const MutationRequest mutation;
  const RuntimeCapabilities capabilities = complete_runtime_capabilities();

  ConSanAutomaticTransformPreparation preparation = prepare_consan_automatic_transform(
      bytes, request, transform_policy, runtime_policy, debug, mutation, capabilities);
  ASSERT_TRUE(std::holds_alternative<ConSanDeferredBinding>(preparation));
  ConSanDeferredBinding &deferred = std::get<ConSanDeferredBinding>(preparation);
  ASSERT_TRUE(deferred.well_formed());
  EXPECT_EQ(deferred.code_object(), make_consan_code_object_id(bytes));
  EXPECT_EQ(deferred.requested_mutation(), mutation);
  EXPECT_EQ(deferred.inventory_mutation(), mutation);
  EXPECT_EQ(deferred.program_inventory().code_object_id(), deferred.code_object());
  EXPECT_TRUE(deferred.observation_plan().valid());
  ASSERT_TRUE(deferred.evidence_requirements());

  const auto &requirements =
      std::get<ConSanRecordReplayEvidenceRequirements>(*deferred.evidence_requirements());
  BoundRuntimeResources resources;
  resources.scope = ConSanRuntimeResourceScope::Executable;
  resources.moi_report_buffer_address = 0x123456780000ull;
  resources.moi_report_buffer_size = requirements.abi_plan.required_bytes;
  resources.moi_report_layout = requirements.abi_plan.complete_layout();
  const TransformResult resumed = resume_consan_automatic_transform(
      bytes, resources, std::move(std::get<ConSanDeferredBinding>(preparation)));
  const TransformResult direct = transform_consan(bytes, request, transform_policy, runtime_policy,
                                                  debug, capabilities, resources);
  ASSERT_TRUE(resumed.well_formed()) << testing::PrintToString(resumed.errors);
  EXPECT_EQ(resumed.outcome, direct.outcome);
  EXPECT_EQ(resumed.observation_plan(), direct.observation_plan());
  EXPECT_EQ(resumed.coverage_ledger, direct.coverage_ledger);
  EXPECT_EQ(resumed.runtime_static_mapping(), direct.runtime_static_mapping());
  EXPECT_EQ(resumed.replacement, direct.replacement);
}

TEST(ConSanPipeline, AutomaticBindingTokenStatesPristineAndRequestedMutationProvenance) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  MutationRequest mutation;
  mutation.fault_lds_wrong_address = true;
  mutation.fault_lds_address_vgpr = 4u;
  ConSanAutomaticTransformPreparation preparation = prepare_consan_automatic_transform(
      bytes, moi_request(ConSanMoiEngine::RecordReplay), TransformPolicy{},
      enabled_runtime_policy(), ConSanDebugOverrides{}, mutation, complete_runtime_capabilities());
  ASSERT_TRUE(std::holds_alternative<ConSanDeferredBinding>(preparation));
  const ConSanDeferredBinding &deferred = std::get<ConSanDeferredBinding>(preparation);
  ASSERT_TRUE(deferred.well_formed());
  EXPECT_EQ(deferred.requested_mutation(), mutation);
  EXPECT_EQ(deferred.inventory_mutation(), without_consan_fault_mutations(mutation));
  EXPECT_FALSE(deferred.inventory_mutation().has_fault_mutation());
}

TEST(ConSanPipeline, AutomaticSuperColliderBindingRelowersThroughLibraryStrategy) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  const ConSanRequest request = supercollider_request();
  const RuntimeCapabilities capabilities = complete_runtime_capabilities();
  ConSanAutomaticTransformPreparation preparation = prepare_consan_automatic_transform(
      bytes, request, TransformPolicy{}, enabled_runtime_policy(), ConSanDebugOverrides{},
      MutationRequest{}, capabilities);
  ASSERT_TRUE(std::holds_alternative<ConSanDeferredBinding>(preparation));
  ConSanDeferredBinding &deferred = std::get<ConSanDeferredBinding>(preparation);
  ASSERT_TRUE(deferred.well_formed());
  ASSERT_TRUE(deferred.evidence_requirements());
  EXPECT_TRUE(std::holds_alternative<ConSanSuperColliderEvidenceRequirements>(
      *deferred.evidence_requirements()));

  BoundRuntimeResources resources;
  resources.scope = ConSanRuntimeResourceScope::Executable;
  resources.report_buffer_address = 0x123456780000ull;
  const TransformResult resumed = resume_consan_automatic_transform(
      bytes, resources, std::move(std::get<ConSanDeferredBinding>(preparation)));
  const TransformResult direct =
      transform_consan(bytes, request, TransformPolicy{}, enabled_runtime_policy(),
                       ConSanDebugOverrides{}, capabilities, resources);
  ASSERT_TRUE(resumed.well_formed()) << testing::PrintToString(resumed.errors);
  EXPECT_EQ(resumed.outcome, direct.outcome);
  EXPECT_EQ(resumed.observation_plan(), direct.observation_plan());
  EXPECT_EQ(resumed.coverage_ledger, direct.coverage_ledger);
  EXPECT_EQ(resumed.replacement, direct.replacement);
}

TEST(ConSanPipeline, AutomaticResumeRejectsBindingBeforeNativeLowering) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  ConSanAutomaticTransformPreparation preparation = prepare_consan_automatic_transform(
      bytes, moi_request(ConSanMoiEngine::RecordReplay), TransformPolicy{},
      enabled_runtime_policy(), ConSanDebugOverrides{}, MutationRequest{},
      complete_runtime_capabilities());
  ASSERT_TRUE(std::holds_alternative<ConSanDeferredBinding>(preparation));

  BoundRuntimeResources invalid;
  invalid.scope = ConSanRuntimeResourceScope::Dispatch;
  invalid.moi_report_buffer_address = 0x123456780000ull;
  invalid.moi_report_buffer_size = sizeof(ConSanMoiReportHeader);
  const TransformResult rejected = resume_consan_automatic_transform(
      bytes, invalid, std::move(std::get<ConSanDeferredBinding>(preparation)));

  ASSERT_TRUE(rejected.well_formed()) << testing::PrintToString(rejected.errors);
  EXPECT_EQ(rejected.outcome, ConSanTransformOutcome::Unsupported);
  EXPECT_EQ(rejected.contract_issue, ConSanContractIssue::InvalidResourceScope);
}

TEST(ConSanPipeline, AutomaticResumeRejectsDifferentInputIdentity) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  ConSanAutomaticTransformPreparation preparation = prepare_consan_automatic_transform(
      bytes, moi_request(ConSanMoiEngine::RecordReplay), TransformPolicy{},
      enabled_runtime_policy(), ConSanDebugOverrides{}, MutationRequest{},
      complete_runtime_capabilities());
  ASSERT_TRUE(std::holds_alternative<ConSanDeferredBinding>(preparation));
  std::vector<uint8_t> different = bytes;
  different.back() ^= 1u;
  BoundRuntimeResources resources;
  resources.scope = ConSanRuntimeResourceScope::Executable;
  resources.moi_report_buffer_address = 0x123456780000ull;
  resources.moi_report_buffer_size = 128u * 1024u * 1024u;
  const TransformResult rejected = resume_consan_automatic_transform(
      different, resources, std::move(std::get<ConSanDeferredBinding>(preparation)));
  EXPECT_EQ(rejected.outcome, ConSanTransformOutcome::Invalid);
  EXPECT_TRUE(rejected.replacement.empty());
  EXPECT_TRUE(std::ranges::any_of(rejected.errors, [](const std::string &error) {
    return error.find("prepared input image") != std::string::npos;
  }));
}

TEST(ConSanPipeline, AutomaticMoiCancellationPublishesCoherentNonInstallableResult) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  const ConSanRequest request = moi_request(ConSanMoiEngine::RecordReplay);
  const RuntimePolicy runtime_policy = enabled_runtime_policy();
  const RuntimeCapabilities capabilities = complete_runtime_capabilities();
  ConSanAutomaticTransformPreparation preparation =
      prepare_consan_automatic_transform(bytes, request, TransformPolicy{}, runtime_policy,
                                         ConSanDebugOverrides{}, MutationRequest{}, capabilities);
  ASSERT_TRUE(std::holds_alternative<ConSanDeferredBinding>(preparation));
  const TransformResult cancelled = cancel_consan_automatic_transform(
      std::move(std::get<ConSanDeferredBinding>(preparation)), "test allocation failure");

  ASSERT_TRUE(cancelled.well_formed()) << testing::PrintToString(cancelled.errors);
  EXPECT_EQ(cancelled.outcome, ConSanTransformOutcome::Unsupported);
  EXPECT_EQ(cancelled.install_action(false), ConSanInstallAction::LoadOriginal);
  EXPECT_TRUE(cancelled.replacement.empty());
  EXPECT_TRUE(std::ranges::any_of(cancelled.warnings, [](const std::string &warning) {
    return warning.find("test allocation failure") != std::string::npos;
  }));
}

TEST(ConSanPipeline, OrdinaryAndMutationEntryPointsAreSeparateAndDeterministic) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  const ConSanRequest request = supercollider_request();
  const TransformPolicy transform_policy;
  const RuntimePolicy runtime_policy = enabled_runtime_policy();
  const ConSanDebugOverrides debug;
  const RuntimeCapabilities capabilities = complete_runtime_capabilities();
  const BoundRuntimeResources resources;
  const ConSanRequest request_before = request;
  const RuntimeCapabilities capabilities_before = capabilities;

  const TransformResult first = transform_consan(bytes, request, transform_policy, runtime_policy,
                                                 debug, capabilities, resources);
  const TransformResult second = transform_consan(bytes, request, transform_policy, runtime_policy,
                                                  debug, capabilities, resources);
  ASSERT_TRUE(first.well_formed()) << testing::PrintToString(first.errors);
  ASSERT_TRUE(second.well_formed()) << testing::PrintToString(second.errors);
  EXPECT_EQ(first.code_object, second.code_object);
  EXPECT_EQ(first.observation_plan(), second.observation_plan());
  EXPECT_EQ(first.evidence_requirements, second.evidence_requirements);
  EXPECT_EQ(first.outcome, second.outcome);
  EXPECT_EQ(first.replacement, second.replacement);
  EXPECT_EQ(first.errors, second.errors);
  EXPECT_EQ(first.warnings, second.warnings);
  EXPECT_EQ(first.mutation, second.mutation);
  EXPECT_EQ(first.dispatch_requirements, second.dispatch_requirements);
  EXPECT_EQ(request, request_before);
  EXPECT_EQ(capabilities, capabilities_before);

  MutationRequest mutation;
  mutation.fault_lds_wrong_address = true;
  mutation.fault_lds_address_vgpr = 4;
  mutation.fault_dry_run = true;
  TransformResult mutated = transform_consan_with_mutation(
      bytes, request, transform_policy, runtime_policy, debug, mutation, capabilities, resources);
  ASSERT_TRUE(mutated.well_formed()) << testing::PrintToString(mutated.errors);
  EXPECT_GT(mutated.mutation.fault.requested, 0u);
  EXPECT_NE(mutated.mutation, ConSanMutationOutcome{});
  const ConSanTransformDiagnosticReport mutated_diagnostics =
      TransformResultTestAccess::diagnostic_report(mutated);
  EXPECT_FALSE(mutated_diagnostics.fault_sites.empty());
  EXPECT_FALSE(mutated_diagnostics.fault_mutations.empty());
  EXPECT_EQ(mutated_diagnostics.fault_mutations.front().target_address_vgpr, 4u);
  EXPECT_EQ(first.code_object, mutated.program_inventory.code_object_id());
}

TEST(ConSanPipeline, RuntimeDiscardClearsInstallableTypedArtifacts) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  const ConSanRequest request = moi_request(ConSanMoiEngine::RecordReplay);
  TransformResult inventory =
      transform_consan(bytes, request, TransformPolicy{}, enabled_runtime_policy(),
                       ConSanDebugOverrides{}, complete_runtime_capabilities(), {});
  ASSERT_TRUE(inventory.evidence_requirements);
  const auto &requirements =
      std::get<ConSanRecordReplayEvidenceRequirements>(*inventory.evidence_requirements);

  BoundRuntimeResources resources;
  resources.scope = ConSanRuntimeResourceScope::Executable;
  resources.moi_report_buffer_address = 0x123456780000ull;
  resources.moi_report_buffer_size = requirements.abi_plan.required_bytes;
  TransformResult result =
      transform_consan(bytes, request, TransformPolicy{}, enabled_runtime_policy(),
                       ConSanDebugOverrides{}, complete_runtime_capabilities(), resources);
  ASSERT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  ASSERT_FALSE(result.replacement.empty());
  ASSERT_FALSE(TransformResultTestAccess::diagnostic_report(result).patches.empty());
  ASSERT_FALSE(result.dispatch_requirements.kernels.empty());

  result.discard_replacement("runtime report allocation failed");

  ASSERT_TRUE(result.well_formed()) << testing::PrintToString(result.errors);
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::Unsupported);
  EXPECT_TRUE(result.replacement.empty());
  EXPECT_TRUE(result.dispatch_requirements.kernels.empty());
  EXPECT_EQ(result.install_action(false), ConSanInstallAction::LoadOriginal);
  EXPECT_EQ(result.install_action(true), ConSanInstallAction::Reject);
  const ConSanTransformDiagnosticReport discarded_diagnostics =
      TransformResultTestAccess::diagnostic_report(result);
  EXPECT_TRUE(discarded_diagnostics.patches.empty());
  EXPECT_TRUE(std::ranges::all_of(
      result.coverage_ledger.intent_entries(), [](const ConSanIntentCoverageEntry &entry) {
        return entry.lowering == ConSanLoweringOutcomeKind::ResourceRejected ||
               entry.lowering == ConSanLoweringOutcomeKind::PlacementRejected;
      }));
  EXPECT_EQ(result.warnings.back(), "runtime report allocation failed");
  EXPECT_EQ(result.contract_issue, ConSanContractIssue::None);
}

} // namespace
} // namespace rocjitsu
