// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"

#include <functional>

namespace rocjitsu::consan {
namespace {

const std::vector<uint8_t> &barrier_policy_bytes() {
  static const std::vector<uint8_t> bytes(256, 0);
  return bytes;
}

SyncEvent
make_barrier_event(uint64_t offset, SyncOperation operation = SyncOperation::BarrierFull,
                   [[maybe_unused]] BarrierSite::Scope scope = BarrierSite::Scope::Workgroup,
                   std::string container = "barrier_kernel") {
  SyncEvent event;
  event.semantic_id = {
      .physical =
          {
              .code_object = make_code_object_id(barrier_policy_bytes()),
              .original_text_offset = offset,
          },
      .domain = SemanticSiteDomain::SynchronizationEvent,
      .member_ordinal = 0,
      .range_ordinal = 0,
  };
  event.kind = SyncKind::Barrier;
  event.operation = operation;
  event.memory_role = operation == SyncOperation::BarrierSignal ? SyncMemoryRole::Release
                      : operation == SyncOperation::BarrierWait ? SyncMemoryRole::Acquire
                                                                : SyncMemoryRole::AcquireRelease;
  event.confidence = SemanticConfidence::Exact;
  event.memory_role_confidence = SemanticConfidence::Exact;
  event.identity = container + "|barrier=" + std::to_string(offset);
  return event;
}

SyncSequence make_barrier_sequence(std::span<const SyncEvent> events,
                                   BarrierSite::Scope scope = BarrierSite::Scope::Workgroup) {
  SyncSequence sequence;
  sequence.kind = SyncKind::Barrier;
  sequence.operation = SyncOperation::BarrierFull;
  sequence.memory_role = SyncMemoryRole::AcquireRelease;
  sequence.confidence = SemanticConfidence::Exact;
  sequence.memory_role_confidence = SemanticConfidence::Exact;
  sequence.identity = "barrier-sequence";
  sequence.begin_text_offset = events.front().text_offset();
  sequence.end_text_offset = events.back().text_offset() + sizeof(uint32_t);
  sequence.basic_block_index = 0;
  sequence.barrier_id = 0;
  sequence.barrier_operand_source = BarrierSite::OperandSource::Immediate;
  sequence.barrier_scope = scope;
  for (size_t index = 0; index < events.size(); ++index)
    sequence.member_event_ids.push_back({static_cast<uint32_t>(index)});
  return sequence;
}

ProgramInventory build_barrier_inventory(std::vector<SyncEvent> events,
                                         std::vector<SyncSequence> sequences,
                                         rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA4,
                                         rj_code_target_id_t target = ROCJITSU_CODE_TARGET_GFX1201,
                                         std::span<const uint32_t> source_sizes = {}) {
  ProgramInventoryBuilder builder(barrier_policy_bytes());
  builder.set_code_object_facts(true, 0, arch, target);
  ProgramContainer kernel{ProgramContainerKind::Kernel};
  kernel.name = "barrier_kernel";
  kernel.descriptor_file_offset = 192;
  kernel.entry_text_offset = 0;
  builder.add_kernel(std::move(kernel));
  for (const SyncEvent &event : events) {
    const std::string name = event.identity.substr(0, event.identity.find('|'));
    if (std::ranges::find(builder.kernels(), name, &ProgramContainer::name) !=
        builder.kernels().end())
      continue;
    ProgramContainer event_kernel{ProgramContainerKind::Kernel};
    event_kernel.name = name;
    event_kernel.descriptor_file_offset = 192u + 64u * builder.kernels().size();
    event_kernel.entry_text_offset = 0;
    builder.add_kernel(std::move(event_kernel));
  }
  for (size_t index = 0; index < events.size(); ++index) {
    SyncEvent &event = events[index];
    BarrierSite site;
    site.operation = event.operation == SyncOperation::BarrierSignal
                         ? BarrierSite::Operation::Signal
                     : event.operation == SyncOperation::BarrierWait ? BarrierSite::Operation::Wait
                                                                     : BarrierSite::Operation::Full;
    site.text_offset = event.text_offset();
    site.file_offset = event.text_offset();
    site.size = index < source_sizes.size() ? source_sizes[index] : sizeof(uint32_t);
    site.barrier_id = 0;
    site.operand_source = BarrierSite::OperandSource::Immediate;
    site.scope = BarrierSite::Scope::Workgroup;
    for (const SyncSequence &sequence : sequences) {
      const SyncEventId member{static_cast<uint32_t>(index)};
      if (std::ranges::find(sequence.member_event_ids, member) != sequence.member_event_ids.end()) {
        site.scope = sequence.barrier_scope;
        break;
      }
    }
    site.mnemonic = "s_barrier";
    event.source_site = {static_cast<uint32_t>(builder.program_sites().size())};
    const std::string_view container_name =
        std::string_view(event.identity).substr(0, event.identity.find('|'));
    const auto container =
        std::ranges::find(builder.kernels(), container_name, &ProgramContainer::name);
    EXPECT_NE(container, builder.kernels().end());
    stage_decoded_site(
        builder, container == builder.kernels().end() ? builder.kernels().front() : *container,
        std::move(site));
    builder.program_sites().back().execution_owners.push_back({});
  }
  SynchronizationInventoryBuildView synchronization = builder.synchronization();
  synchronization.sync_events = std::move(events);
  synchronization.sync_sequences = std::move(sequences);
  return builder.view();
}

ProgramInventory
one_full_barrier_inventory(rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA4,
                           rj_code_target_id_t target = ROCJITSU_CODE_TARGET_GFX1201) {
  std::vector events{make_barrier_event(32)};
  std::vector sequences{make_barrier_sequence(events)};
  return build_barrier_inventory(std::move(events), std::move(sequences), arch, target);
}

BarrierPolicyRequest barrier_request(Mode mode) {
  return {
      .mode = mode,
      .tracking_enabled = true,
      .container_filter = {},
      .kernel_name_allowlist = {},
  };
}

TEST(ConSanBarrierPolicy, InvalidModeFailsValidationButEmptyInventoryIsAValidEmptyPlan) {
  const BarrierPolicyResult invalid =
      plan_barrier_observation(one_full_barrier_inventory(), barrier_request(Mode::None));
  EXPECT_FALSE(invalid.valid());
  EXPECT_FALSE(invalid.plan.valid());

  const BarrierPolicyResult empty =
      plan_barrier_observation(ProgramInventory{}, barrier_request(Mode::Default));
  EXPECT_TRUE(empty.valid());
  EXPECT_TRUE(empty.plan.barrier_site_decisions.empty());
  EXPECT_TRUE(empty.plan.probe_intents.empty());
}

TEST(ConSanBarrierPolicy, AllModesExpressTheirBarrierContract) {
  const ProgramInventory inventory = one_full_barrier_inventory();
  const BarrierPolicyResult supercollider =
      plan_barrier_observation(inventory, barrier_request(Mode::SuperCollider));
  ASSERT_TRUE(supercollider.valid());
  ASSERT_EQ(supercollider.plan.barrier_site_decisions.size(), 1u);
  EXPECT_EQ(supercollider.plan.barrier_site_decisions.front().kind,
            SiteDecisionKind::NotApplicable);
  EXPECT_EQ(supercollider.plan.barrier_site_decisions.front().reason,
            BarrierPolicyReason::ModeMutationOnly);
  EXPECT_TRUE(supercollider.plan.probe_intents.empty());

  const BarrierPolicyResult policy =
      plan_barrier_observation(inventory, barrier_request(Mode::Default));
  ASSERT_TRUE(policy.valid());
  ASSERT_EQ(policy.plan.barrier_site_decisions.size(), 1u);
  ASSERT_EQ(policy.plan.probe_intents.size(), 1u);
  EXPECT_EQ(policy.plan.barrier_site_decisions.front().kind, SiteDecisionKind::Admitted);
  EXPECT_EQ(policy.plan.barrier_site_decisions.front().reason, BarrierPolicyReason::None);
  EXPECT_EQ(policy.plan.probe_intents.front().kind, ProbeIntentKind::BarrierEpoch);
  EXPECT_EQ(policy.plan.probe_intents.front().position, ProbePosition::After);
}

TEST(ConSanBarrierPolicy, PairedSequenceHasOneEpochIntent) {
  std::vector events{
      make_barrier_event(32, SyncOperation::BarrierSignal),
      make_barrier_event(48, SyncOperation::BarrierWait),
  };
  std::vector sequences{make_barrier_sequence(events)};
  const ProgramInventory inventory = build_barrier_inventory(events, sequences);

  const BarrierPolicyResult epoch =
      plan_barrier_observation(inventory, barrier_request(Mode::Default));
  ASSERT_TRUE(epoch.valid());
  ASSERT_EQ(epoch.plan.barrier_site_decisions.size(), 2u);
  ASSERT_EQ(epoch.plan.probe_intents.size(), 1u);
  EXPECT_EQ(epoch.plan.probe_intents.front().physical_site.original_text_offset, 48u);
  EXPECT_EQ(epoch.plan.probe_intents.front().covered_semantic_sites.size(), 2u);
}

TEST(ConSanBarrierPolicy, SignalWithoutACompletionIsUnsupported) {
  std::vector events{make_barrier_event(32, SyncOperation::BarrierSignal)};
  SyncSequence sequence = make_barrier_sequence(events);
  sequence.operation = SyncOperation::BarrierSignal;
  sequence.memory_role = SyncMemoryRole::Release;
  const BarrierPolicyResult policy =
      plan_barrier_observation(build_barrier_inventory(std::move(events), {std::move(sequence)}),
                               barrier_request(Mode::Default));
  ASSERT_TRUE(policy.valid());
  ASSERT_EQ(policy.plan.barrier_site_decisions.size(), 1u);
  EXPECT_EQ(policy.plan.barrier_site_decisions.front().kind, SiteDecisionKind::Unsupported);
  EXPECT_EQ(policy.plan.barrier_site_decisions.front().reason,
            BarrierPolicyReason::UnqualifiedSyncSequence);
  EXPECT_TRUE(policy.plan.probe_intents.empty());
}

TEST(ConSanBarrierPolicy, DisabledFilteredAndRuntimeEventsRemainExplicitlyOutsideContract) {
  const ProgramInventory inventory = one_full_barrier_inventory();
  BarrierPolicyRequest disabled = barrier_request(Mode::Default);
  disabled.tracking_enabled = false;
  EXPECT_EQ(
      plan_barrier_observation(inventory, disabled).plan.barrier_site_decisions.front().reason,
      BarrierPolicyReason::TrackingDisabled);

  BarrierPolicyRequest filtered = barrier_request(Mode::Default);
  filtered.container_filter = "different";
  EXPECT_EQ(
      plan_barrier_observation(inventory, filtered).plan.barrier_site_decisions.front().reason,
      BarrierPolicyReason::ContainerFilterExcluded);

  std::vector runtime_events{make_barrier_event(
      32, SyncOperation::BarrierFull, BarrierSite::Scope::Workgroup, "__amd_rocclr_runtime")};
  const ProgramInventory runtime =
      build_barrier_inventory(runtime_events, {make_barrier_sequence(runtime_events)});
  EXPECT_EQ(plan_barrier_observation(runtime, barrier_request(Mode::Default))
                .plan.barrier_site_decisions.front()
                .reason,
            BarrierPolicyReason::RuntimeKernelExcluded);
}

TEST(ConSanBarrierPolicy, ClusterScopeUsesTheTargetCapabilityContract) {
  std::vector events{
      make_barrier_event(32, SyncOperation::BarrierFull, BarrierSite::Scope::Cluster),
  };
  std::vector sequences{make_barrier_sequence(events, BarrierSite::Scope::Cluster)};
  const BarrierPolicyResult gfx1201 = plan_barrier_observation(
      build_barrier_inventory(events, sequences), barrier_request(Mode::Default));
  ASSERT_TRUE(gfx1201.valid());
  EXPECT_EQ(gfx1201.plan.barrier_site_decisions.front().kind, SiteDecisionKind::NotApplicable);
  EXPECT_EQ(gfx1201.plan.barrier_site_decisions.front().reason,
            BarrierPolicyReason::TargetCapabilityUnavailable);

  const BarrierPolicyResult gfx1250 = plan_barrier_observation(
      build_barrier_inventory(std::move(events), std::move(sequences), ROCJITSU_CODE_ARCH_CDNA5,
                              ROCJITSU_CODE_TARGET_GFX1250),
      barrier_request(Mode::Default));
  ASSERT_TRUE(gfx1250.valid());
  EXPECT_EQ(gfx1250.plan.barrier_site_decisions.front().kind, SiteDecisionKind::Admitted);
}

TEST(ConSanBarrierPolicy, MismatchedCompletionAndRedundantFullBarrierHaveTypedReasons) {
  std::vector invalid_events{make_barrier_event(32)};
  std::vector invalid_sequences{make_barrier_sequence(invalid_events)};
  const BarrierPolicyResult invalid = plan_barrier_observation(
      build_barrier_inventory(std::move(invalid_events), std::move(invalid_sequences),
                              ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_TARGET_GFX1201,
                              std::array<uint32_t, 1>{8u}),
      barrier_request(Mode::Default));
  ASSERT_TRUE(invalid.valid());
  EXPECT_EQ(invalid.plan.barrier_site_decisions.front().kind, SiteDecisionKind::Unsupported);
  EXPECT_EQ(invalid.plan.barrier_site_decisions.front().reason,
            BarrierPolicyReason::MissingCompletingEvent);

  std::vector adjacent_events{make_barrier_event(32), make_barrier_event(36)};
  std::vector adjacent_sequences{
      make_barrier_sequence(std::span<const SyncEvent>(adjacent_events).first(1))};
  adjacent_sequences.push_back(
      make_barrier_sequence(std::span<const SyncEvent>(adjacent_events).subspan(1)));
  adjacent_sequences.back().member_event_ids = {{1}};
  const BarrierPolicyResult adjacent = plan_barrier_observation(
      build_barrier_inventory(std::move(adjacent_events), std::move(adjacent_sequences)),
      barrier_request(Mode::Default));
  ASSERT_TRUE(adjacent.valid());
  ASSERT_EQ(adjacent.plan.barrier_site_decisions.size(), 2u);
  EXPECT_EQ(adjacent.plan.barrier_site_decisions[0].kind, SiteDecisionKind::Admitted);
  EXPECT_EQ(adjacent.plan.barrier_site_decisions[1].reason,
            BarrierPolicyReason::RedundantAdjacentFullBarrier);
}

TEST(ConSanBarrierPolicy, QualificationRejectsEveryRequiredSemanticFact) {
  using Mutation = std::function<void(SyncSequence &)>;
  const std::vector<std::pair<std::string_view, Mutation>> cases = {
      {"kind", [](auto &s) { s.kind = SyncKind::Fence; }},
      {"operation", [](auto &s) { s.operation = SyncOperation::BarrierWait; }},
      {"role", [](auto &s) { s.memory_role = SyncMemoryRole::Acquire; }},
      {"confidence", [](auto &s) { s.confidence = SemanticConfidence::Unsupported; }},
      {"role-confidence",
       [](auto &s) { s.memory_role_confidence = SemanticConfidence::Unsupported; }},
      {"block", [](auto &s) { s.basic_block_index.reset(); }},
      {"clause", [](auto &s) { s.inside_scalar_clause = true; }},
      {"static-id",
       [](auto &s) { s.barrier_operand_source = BarrierSite::OperandSource::DynamicM0; }},
      {"id", [](auto &s) { s.barrier_id.reset(); }},
      {"scope", [](auto &s) { s.barrier_scope = BarrierSite::Scope::Unknown; }},
      {"members", [](auto &s) { s.member_event_ids.clear(); }},
      {"bounds", [](auto &s) { s.end_text_offset = s.begin_text_offset; }},
  };
  for (const auto &[name, mutate] : cases) {
    SCOPED_TRACE(name);
    std::vector events{make_barrier_event(32)};
    std::vector sequences{make_barrier_sequence(events)};
    mutate(sequences.front());
    const BarrierPolicyResult policy =
        plan_barrier_observation(build_barrier_inventory(std::move(events), std::move(sequences)),
                                 barrier_request(Mode::Default));
    ASSERT_TRUE(policy.valid());
    ASSERT_EQ(policy.plan.barrier_site_decisions.size(), 1u);
    EXPECT_EQ(policy.plan.barrier_site_decisions.front().kind, SiteDecisionKind::Unsupported);
    EXPECT_EQ(policy.plan.barrier_site_decisions.front().reason,
              BarrierPolicyReason::UnqualifiedSyncSequence);
    EXPECT_TRUE(policy.plan.probe_intents.empty());
  }

  std::vector ownerless_events{make_barrier_event(32)};
  std::vector ownerless_sequences{make_barrier_sequence(ownerless_events)};
  ProgramInventoryBuilder ownerless(
      build_barrier_inventory(std::move(ownerless_events), std::move(ownerless_sequences)));
  ownerless.program_sites().front().execution_owners.clear();
  const BarrierPolicyResult ownerless_policy =
      plan_barrier_observation(ownerless.view(), barrier_request(Mode::Default));
  ASSERT_TRUE(ownerless_policy.valid());
  ASSERT_EQ(ownerless_policy.plan.barrier_site_decisions.size(), 1u);
  EXPECT_EQ(ownerless_policy.plan.barrier_site_decisions.front().kind,
            SiteDecisionKind::Unsupported);
  EXPECT_EQ(ownerless_policy.plan.barrier_site_decisions.front().reason,
            BarrierPolicyReason::UnqualifiedSyncSequence);
  EXPECT_TRUE(ownerless_policy.plan.probe_intents.empty());
}

TEST(ConSanBarrierPolicy, AmbiguousAndIncompleteSequencesFailClosedWithDistinctReasons) {
  std::vector events{make_barrier_event(32), make_barrier_event(48)};
  std::vector sequences{
      make_barrier_sequence(std::span<const SyncEvent>(events).first(1)),
      make_barrier_sequence(events),
  };
  const BarrierPolicyResult ambiguous = plan_barrier_observation(
      build_barrier_inventory(events, sequences), barrier_request(Mode::Default));
  ASSERT_TRUE(ambiguous.valid());
  EXPECT_EQ(ambiguous.plan.barrier_site_decisions.front().reason,
            BarrierPolicyReason::AmbiguousSequenceMembership);

  std::vector incomplete_events{make_barrier_event(32)};
  std::vector incomplete_sequences{make_barrier_sequence(incomplete_events)};
  incomplete_sequences.front().end_text_offset = 40;
  const BarrierPolicyResult incomplete = plan_barrier_observation(
      build_barrier_inventory(std::move(incomplete_events), std::move(incomplete_sequences)),
      barrier_request(Mode::Default));
  ASSERT_TRUE(incomplete.valid());
  EXPECT_EQ(incomplete.plan.barrier_site_decisions.front().reason,
            BarrierPolicyReason::MissingCompletingEvent);
}

TEST(ConSanBarrierPolicy, IdenticalAliasesCoalesceAndConflictingAliasesAreFatal) {
  SyncEvent first = make_barrier_event(32);
  SyncEvent second = first;
  second.identity = "shared_alias|barrier=32";
  const std::array alias_events{first, second};
  const ProgramInventory coalesced_inventory =
      build_barrier_inventory({first, second}, {make_barrier_sequence(alias_events)});
  const BarrierPolicyResult coalesced =
      plan_barrier_observation(coalesced_inventory, barrier_request(Mode::Default));
  ASSERT_TRUE(coalesced.valid());
  ASSERT_EQ(coalesced.plan.barrier_site_decisions.size(), 1u);
  EXPECT_EQ(coalesced_inventory.source_container_names(
                coalesced.plan.barrier_site_decisions.front().semantic_site.physical),
            (std::vector<std::string>{"barrier_kernel", "shared_alias"}));
  EXPECT_EQ(coalesced.plan.probe_intents.size(), 1u);

  const BarrierPolicyResult conflicting = plan_barrier_observation(
      build_barrier_inventory({first, second}, {}, ROCJITSU_CODE_ARCH_RDNA4,
                              ROCJITSU_CODE_TARGET_GFX1201, std::array<uint32_t, 2>{4u, 8u}),
      barrier_request(Mode::Default));
  EXPECT_FALSE(conflicting.valid());
  EXPECT_EQ(conflicting.errors, (std::vector{BarrierPolicyReason::ConflictingPhysicalAliases}));
  EXPECT_EQ(conflicting.plan.barrier_site_decisions.front().reason,
            BarrierPolicyReason::ConflictingPhysicalAliases);
}

TEST(ConSanBarrierPolicy, PolicyIsDeterministicAndDoesNotMutateInventory) {
  const ProgramInventory inventory = one_full_barrier_inventory();
  const SynchronizationInventoryView view = inventory.sync();
  const std::vector<SyncEvent> events(view.sync_events.begin(), view.sync_events.end());
  const std::vector<SyncSequence> sequences(view.sync_sequences.begin(), view.sync_sequences.end());
  const BarrierPolicyRequest request = barrier_request(Mode::Default);
  EXPECT_EQ(plan_barrier_observation(inventory, request),
            plan_barrier_observation(inventory, request));
  ASSERT_EQ(view.sync_events.size(), events.size());
  ASSERT_EQ(view.sync_sequences.size(), sequences.size());
  EXPECT_EQ(view.sync_events.front().identity, events.front().identity);
  EXPECT_EQ(view.sync_events.front().semantic_id, events.front().semantic_id);
  EXPECT_EQ(view.sync_events.front().operation, events.front().operation);
  EXPECT_EQ(view.sync_sequences.front().identity, sequences.front().identity);
  EXPECT_EQ(view.sync_sequences.front().member_event_ids, sequences.front().member_event_ids);
  EXPECT_EQ(view.sync_sequences.front().operation, sequences.front().operation);
}

} // namespace
} // namespace rocjitsu::consan
