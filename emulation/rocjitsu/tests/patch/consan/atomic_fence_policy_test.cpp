// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"
#include "rocjitsu/code/patch/consan/consan_evidence_planning.h"

#include <functional>

namespace rocjitsu::consan {
namespace {

const std::vector<uint8_t> &atomic_policy_bytes() {
  static const std::vector<uint8_t> bytes(512, 0);
  return bytes;
}

/// Architecture/processor pair used to transport one semantic policy fixture
/// across the five supported targets without letting their encoding widths
/// leak into individual tests.
struct AtomicPolicyTarget {
  rj_code_arch_t arch = ROCJITSU_CODE_ARCH_RDNA4;
  rj_code_target_id_t target = ROCJITSU_CODE_TARGET_GFX1201;
};

AtomicSite make_global_atomic_site(const AtomicPolicyTarget &target = {}, uint64_t offset = 32,
                                   std::string mnemonic = "global_atomic_add_u32",
                                   SyncRmwOutcome outcome = SyncRmwOutcome::ReturnsOldValue) {
  AtomicSite site;
  site.text_offset = offset;
  site.file_offset = offset;
  site.size = target_profile(target.arch)->vector_memory.instruction_word_count * sizeof(uint32_t);
  site.width_bits = 32;
  site.destination_vgpr = 1;
  site.address_vgpr = 0;
  site.data_vgpr = 2;
  site.scalar_address_sgpr = 4;
  site.raw_saddr = 4;
  if (target.arch == ROCJITSU_CODE_ARCH_CDNA5)
    site.raw_scale_offset = false;
  site.raw_vaddr = 0;
  site.raw_vdata = 2;
  site.raw_ioffset = 0;
  site.scope = MemoryScope::Agent;
  site.raw_th = 0;
  site.returns_old_value = outcome != SyncRmwOutcome::NoReturn;
  site.mnemonic = std::move(mnemonic);
  return site;
}

AtomicSite make_lds_atomic_site(uint64_t offset = 32) {
  AtomicSite site;
  site.address_space_hint = AtomicAddressSpaceHint::Lds;
  site.text_offset = offset;
  site.file_offset = offset;
  site.size = 8;
  site.width_bits = 32;
  site.destination_vgpr = 1;
  site.address_vgpr = 0;
  site.data_vgpr = 2;
  site.scope = MemoryScope::Workgroup;
  site.raw_addr = 0;
  site.raw_data0 = 2;
  site.raw_ioffset = 0;
  site.returns_old_value = true;
  site.mnemonic = "ds_add_u32";
  return site;
}

OrdinaryMemorySite make_global_store_site(const AtomicPolicyTarget &target, uint64_t offset = 32) {
  OrdinaryMemorySite site;
  site.operation = OrdinaryMemoryOperation::Store;
  site.support_reason = OrdinaryMemorySupportReason::SupportedSynchronizationOnly;
  site.text_offset = offset;
  site.file_offset = offset;
  site.size = target_profile(target.arch)->vector_memory.instruction_word_count * sizeof(uint32_t);
  site.width_bits = 32;
  site.address_vgpr = 0;
  site.scalar_address_sgpr = 4;
  site.data_vgpr = 2;
  site.raw_saddr = 4;
  if (target.arch == ROCJITSU_CODE_ARCH_CDNA5)
    site.raw_scale_offset = false;
  site.raw_vaddr = 0;
  site.raw_vsrc = 2;
  site.raw_ioffset = 0;
  site.scope = MemoryScope::Agent;
  site.raw_th = 0;
  site.mnemonic = "global_store_b32";
  return site;
}

OrdinaryMemorySite make_cdna5_buffer_load_site(uint64_t offset = 32) {
  OrdinaryMemorySite site;
  site.operation = OrdinaryMemoryOperation::Load;
  site.support_reason = OrdinaryMemorySupportReason::SupportedSynchronizationOnly;
  site.text_offset = offset;
  site.file_offset = offset;
  site.size = 12;
  site.width_bits = 32;
  site.destination_vgpr = 4;
  site.address_vgpr = 5;
  site.scalar_address_sgpr = 24;
  site.raw_rsrc = 24;
  site.raw_soffset = 0x7c;
  site.raw_vaddr = 5;
  site.raw_vdst = 4;
  site.raw_ioffset = 0;
  site.scope = MemoryScope::Agent;
  site.raw_offen = true;
  site.raw_idxen = false;
  site.mnemonic = "buffer_load_b32";
  return site;
}

SyncEvent
make_atomic_event(uint64_t offset = 32, SyncRmwOutcome outcome = SyncRmwOutcome::ReturnsOldValue,
                  SyncAddressSource address_source = SyncAddressSource::GlobalScalarVector,
                  std::string mnemonic = "global_atomic_add_u32",
                  std::string container = "atomic_kernel") {
  SyncEvent event;
  event.semantic_id = {
      .physical =
          {
              .code_object = make_code_object_id(atomic_policy_bytes()),
              .original_text_offset = offset,
          },
      .domain = SemanticSiteDomain::SynchronizationEvent,
  };
  event.kind = SyncKind::Atomic;
  event.operation = mnemonic.find("cmp") == std::string::npos
                        ? SyncOperation::AtomicRmw
                        : SyncOperation::AtomicCompareExchange;
  event.address_source = address_source;
  event.memory_role = SyncMemoryRole::AcquireRelease;
  event.rmw_outcome = outcome;
  event.confidence = SemanticConfidence::Exact;
  event.memory_role_confidence = SemanticConfidence::Exact;
  event.identity = container + "|atomic=" + std::to_string(offset);
  event.scope =
      address_source == SyncAddressSource::LdsVector ? MemoryScope::Workgroup : MemoryScope::Agent;
  return event;
}

SyncEvent make_ordinary_store_event(uint64_t offset = 32) {
  SyncEvent event = make_atomic_event(offset, SyncRmwOutcome::NotApplicable,
                                      SyncAddressSource::GlobalScalarVector, "global_store_b32");
  event.kind = SyncKind::OrdinaryMemory;
  event.operation = SyncOperation::OrdinaryStore;
  event.memory_role = SyncMemoryRole::Release;
  event.identity = "atomic_kernel|ordinary-store=" + std::to_string(offset);
  return event;
}

SyncEvent make_fence_event(uint64_t offset = 48) {
  SyncEvent event = make_atomic_event(offset, SyncRmwOutcome::NotApplicable,
                                      SyncAddressSource::NotApplicable, "global_wb");
  event.kind = SyncKind::Fence;
  event.operation = SyncOperation::Fence;
  event.memory_role = SyncMemoryRole::Release;
  event.identity = "atomic_kernel|fence=" + std::to_string(offset);
  event.scope.reset();
  return event;
}

SyncSequence make_atomic_sequence(const SyncEvent &event,
                                  std::string identity = "atomic-sequence") {
  SyncSequence sequence;
  sequence.kind =
      event.kind == SyncKind::OrdinaryMemory ? SyncKind::OrdinaryMemory : SyncKind::Atomic;
  sequence.operation = event.operation;
  sequence.address_source = event.address_source;
  sequence.memory_role = event.memory_role;
  sequence.rmw_outcome = event.rmw_outcome;
  sequence.confidence = SemanticConfidence::Exact;
  sequence.memory_role_confidence = SemanticConfidence::Exact;
  sequence.identity = std::move(identity);
  sequence.begin_text_offset = event.text_offset();
  sequence.end_text_offset = event.text_offset() + 12u;
  sequence.basic_block_index = 0;
  sequence.member_event_ids.push_back({0});
  sequence.scope = event.scope;
  return sequence;
}

FenceCandidate make_fence_candidate([[maybe_unused]] const SyncEvent &communication,
                                    [[maybe_unused]] const SyncEvent &fence,
                                    [[maybe_unused]] const SyncSequence &sequence,
                                    FenceAssociation association = FenceAssociation::Qualified) {
  return {
      .fence_event = {1},
      .sequence = {0},
      .communication_event = SyncEventId{0},
      .memory_role = fence.memory_role,
      .association = association,
  };
}

ProgramInventory build_atomic_inventory(
    std::vector<SyncEvent> events, std::vector<SyncSequence> sequences,
    std::vector<AtomicSite> atomic_sites, std::vector<OrdinaryMemorySite> ordinary_sites = {},
    std::vector<FenceCandidate> fences = {}, const AtomicPolicyTarget &target = {},
    std::vector<uint64_t> unowned_offsets = {}, std::vector<ProgramSite> access_sites = {}) {
  ProgramInventoryBuilder builder(atomic_policy_bytes());
  builder.set_code_object_facts(true, 0, target.arch, target.target);
  ProgramContainer kernel{ProgramContainerKind::Kernel};
  kernel.name = "atomic_kernel";
  kernel.descriptor_file_offset = 384;
  kernel.entry_text_offset = 0;
  builder.add_kernel(std::move(kernel));
  for (const SyncEvent &event : events) {
    const std::string name = event.identity.substr(0, event.identity.find('|'));
    if (std::ranges::find(builder.kernels(), name, &ProgramContainer::name) !=
        builder.kernels().end())
      continue;
    ProgramContainer event_kernel{ProgramContainerKind::Kernel};
    event_kernel.name = name;
    event_kernel.descriptor_file_offset = 384u + 64u * builder.kernels().size();
    event_kernel.entry_text_offset = 0;
    builder.add_kernel(std::move(event_kernel));
  }
  for (ProgramSite &site : access_sites) {
    site.container = builder.kernels().back().id;
    builder.add_access_site(std::move(site));
  }
  for (AtomicSite &site : atomic_sites)
    stage_decoded_site(builder, builder.kernels().back(), std::move(site));
  for (OrdinaryMemorySite &site : ordinary_sites)
    stage_decoded_site(builder, builder.kernels().back(), std::move(site));
  for (const SyncEvent &event : events) {
    if (event.kind != SyncKind::Fence)
      continue;
    FenceSite site;
    site.text_offset = event.text_offset();
    site.file_offset = event.text_offset();
    site.size =
        target_profile(target.arch)->vector_memory.instruction_word_count * sizeof(uint32_t);
    site.cache_operation = event.memory_role == SyncMemoryRole::Acquire ? CacheOperation::Acquire
                                                                        : CacheOperation::Release;
    site.mnemonic = site.cache_operation == CacheOperation::Acquire ? "global_inv" : "global_wb";
    stage_decoded_site(builder, builder.kernels().back(), std::move(site));
  }
  builder.publish_decoded_accesses(atomic_policy_bytes());
  const std::span<const ProgramSite> program_sites = builder.view().program_sites();
  for (SyncEvent &event : events) {
    if (event.source_site.valid())
      continue;
    std::optional<ProgramSiteId> match;
    for (size_t index = 0; index < program_sites.size(); ++index) {
      const ProgramSite &decoded = program_sites[index];
      const bool kind_matches =
          (event.kind == SyncKind::Atomic && decoded.get_if<AtomicSite>() != nullptr) ||
          (event.kind == SyncKind::OrdinaryMemory &&
           decoded.get_if<OrdinaryMemorySite>() != nullptr) ||
          (event.kind == SyncKind::Fence && decoded.get_if<FenceSite>() != nullptr);
      if (!kind_matches || decoded.text_offset() != event.text_offset())
        continue;
      if (match) {
        match.reset();
        break;
      }
      match = ProgramSiteId{static_cast<uint32_t>(index)};
    }
    if (match)
      event.source_site = *match;
  }
  for (ProgramSite &site : builder.program_sites()) {
    if (std::ranges::find(unowned_offsets, site.text_offset()) == unowned_offsets.end())
      site.execution_owners.push_back({});
  }
  for (const SyncEvent &event : events) {
    if (!event.source_site.valid() || event.source_site.ordinal >= builder.program_sites().size())
      continue;
    const std::string_view name =
        std::string_view(event.identity).substr(0, event.identity.find('|'));
    const auto container = std::ranges::find(builder.kernels(), name, &ProgramContainer::name);
    if (container != builder.kernels().end())
      builder.program_sites()[event.source_site.ordinal].container = container->id;
  }
  SynchronizationInventoryBuildView synchronization = builder.synchronization();
  synchronization.sync_events = std::move(events);
  synchronization.sync_sequences = std::move(sequences);
  synchronization.fence_candidates = std::move(fences);
  return builder.view();
}

ProgramSite make_lds_access(LdsAccessKind kind, uint64_t offset = 96) {
  ProgramSite site;
  site.origin = AccessOrigin::NativeLds;
  site.kind = kind;
  site.physical_id.original_text_offset = offset;
  site.decoded_site().text_offset = offset;
  site.decoded_site().file_offset = offset;
  site.decoded_site().size = 8;
  site.decoded_width_bits = 32;
  site.operands.address_vgpr = 3;
  site.operands.data_vgpr = 4;
  site.decoded_site().mnemonic = kind == LdsAccessKind::Read ? "ds_load_b32" : "ds_store_b32";
  return site;
}

ProgramInventory
one_atomic_inventory(const AtomicPolicyTarget &target = {},
                     SyncRmwOutcome outcome = SyncRmwOutcome::ReturnsOldValue,
                     std::string mnemonic = "global_atomic_add_u32",
                     SyncAddressSource address_source = SyncAddressSource::GlobalScalarVector) {
  std::vector events{make_atomic_event(32, outcome, address_source, mnemonic)};
  std::vector sequences{make_atomic_sequence(events.front())};
  AtomicSite site = address_source == SyncAddressSource::LdsVector
                        ? make_lds_atomic_site()
                        : make_global_atomic_site(target, 32, std::move(mnemonic), outcome);
  return build_atomic_inventory(std::move(events), std::move(sequences), {std::move(site)}, {}, {},
                                target);
}

ProgramInventory
ordinary_fence_inventory(FenceAssociation association = FenceAssociation::Qualified,
                         const AtomicPolicyTarget &target = {}) {
  std::vector events{make_ordinary_store_event(), make_fence_event()};
  std::vector sequences{make_atomic_sequence(events.front())};
  std::vector fences{make_fence_candidate(events[0], events[1], sequences[0], association)};
  return build_atomic_inventory(std::move(events), std::move(sequences), {},
                                {make_global_store_site(target)}, std::move(fences), target);
}

AtomicFencePolicyRequest atomic_request(Mode mode) {
  static constexpr std::array kWindows = {
      DirectionalAccessAvailability{
          .owner = ProgramContainerId{},
          .read = true,
          .write = true,
      },
  };
  return {
      .mode = mode,
      .tracking_enabled = true,
      .directional_access_windows = kWindows,
      .container_filter = {},
      .kernel_name_allowlist = {},
  };
}

TEST(ConSanAtomicFencePolicy, InvalidModeFailsValidationButEmptyInventoryIsAValidEmptyPlan) {
  const AtomicFencePolicyResult invalid =
      plan_atomic_fence_observation(one_atomic_inventory(), atomic_request(Mode::None));
  EXPECT_FALSE(invalid.valid());
  EXPECT_FALSE(invalid.plan.valid());

  const AtomicFencePolicyResult empty =
      plan_atomic_fence_observation(ProgramInventory{}, atomic_request(Mode::Default));
  EXPECT_TRUE(empty.valid());
  EXPECT_TRUE(empty.plan.atomic_site_decisions.empty());
  EXPECT_TRUE(empty.plan.fence_site_decisions.empty());
  EXPECT_TRUE(empty.plan.probe_intents.empty());
}

TEST(ConSanAtomicFencePolicy, AllModesExpressTheirAtomicObservationContract) {
  const ProgramInventory inventory = one_atomic_inventory();
  const AtomicFencePolicyResult supercollider =
      plan_atomic_fence_observation(inventory, atomic_request(Mode::SuperCollider));
  ASSERT_TRUE(supercollider.valid());
  ASSERT_EQ(supercollider.plan.atomic_site_decisions.size(), 1u);
  EXPECT_EQ(supercollider.plan.atomic_site_decisions.front().kind, SiteDecisionKind::NotApplicable);
  EXPECT_EQ(supercollider.plan.atomic_site_decisions.front().reason,
            AtomicPolicyReason::ModeMutationOnly);
  EXPECT_TRUE(supercollider.plan.probe_intents.empty());

  const AtomicFencePolicyResult policy =
      plan_atomic_fence_observation(inventory, atomic_request(Mode::Default));
  ASSERT_TRUE(policy.valid());
  ASSERT_EQ(policy.plan.atomic_site_decisions.size(), 1u);
  ASSERT_EQ(policy.plan.probe_intents.size(), 2u);
  const AtomicSiteDecision &decision = policy.plan.atomic_site_decisions.front();
  EXPECT_EQ(decision.kind, SiteDecisionKind::Admitted);
  EXPECT_EQ(decision.reason, AtomicPolicyReason::None);
  ASSERT_TRUE(policy.plan.probe_intents[0].synchronization_association.has_value());
  ASSERT_TRUE(policy.plan.probe_intents[0].atomic_lowering_form.has_value());
  EXPECT_EQ(policy.plan.probe_intents[0].atomic_lowering_form->kind,
            AtomicLoweringFormKind::GlobalScalarVectorAddress);
  EXPECT_EQ(policy.plan.probe_intents[0].kind, ProbeIntentKind::AtomicAddressCapture);
  EXPECT_EQ(policy.plan.probe_intents[0].position, ProbePosition::Before);
  EXPECT_EQ(policy.plan.probe_intents[0].dynamic_result, DynamicResultRequirement::None);
  EXPECT_EQ(policy.plan.probe_intents[1].kind, ProbeIntentKind::AtomicOrdering);
  EXPECT_EQ(policy.plan.probe_intents[1].position, ProbePosition::After);
  EXPECT_EQ(policy.plan.probe_intents[1].dynamic_result,
            DynamicResultRequirement::ReturnedOldValue);
  EXPECT_EQ(policy.plan.probe_intents[0].synchronization_association,
            policy.plan.probe_intents[1].synchronization_association);
}

TEST(ConSanAtomicFencePolicy, DynamicAtomicOutcomesBecomeExplicitAfterIntentRequirements) {
  constexpr std::array cases = {
      std::tuple{SyncRmwOutcome::NoReturn, "global_atomic_add_u32", DynamicResultRequirement::None},
      std::tuple{SyncRmwOutcome::ReturnsOldValue, "global_atomic_add_u32",
                 DynamicResultRequirement::ReturnedOldValue},
      std::tuple{SyncRmwOutcome::CompareExchange, "global_atomic_cmpswap_u32",
                 DynamicResultRequirement::CompareExchangeSuccess},
  };
  for (const auto &[outcome, mnemonic, expected] : cases) {
    SCOPED_TRACE(mnemonic);
    const AtomicFencePolicyResult policy = plan_atomic_fence_observation(
        one_atomic_inventory({}, outcome, mnemonic), atomic_request(Mode::Default));
    ASSERT_TRUE(policy.valid());
    ASSERT_EQ(policy.plan.probe_intents.size(), 2u);
    EXPECT_EQ(policy.plan.probe_intents[1].dynamic_result, expected);
  }
}

TEST(ConSanAtomicFencePolicy, RequestExclusionsRemainTypedAndDoNotCreateIntents) {
  const ProgramInventory inventory = one_atomic_inventory();
  using Mutation = std::function<void(AtomicFencePolicyRequest &)>;
  const std::vector<std::tuple<std::string_view, Mode, Mutation, AtomicPolicyReason>> cases = {
      {"disabled", Mode::Default, [](auto &request) { request.tracking_enabled = false; },
       AtomicPolicyReason::TrackingDisabled},
      {"filtered", Mode::Default, [](auto &request) { request.container_filter = "different"; },
       AtomicPolicyReason::ContainerFilterExcluded},
      {"sampled-window", Mode::Default,
       [](auto &request) { request.directional_access_windows = {}; },
       AtomicPolicyReason::MissingDirectionalAccessWindow},
  };
  for (const auto &[name, mode, mutate, expected] : cases) {
    SCOPED_TRACE(name);
    AtomicFencePolicyRequest request = atomic_request(mode);
    mutate(request);
    const AtomicFencePolicyResult policy = plan_atomic_fence_observation(inventory, request);
    ASSERT_TRUE(policy.valid());
    ASSERT_EQ(policy.plan.atomic_site_decisions.size(), 1u);
    EXPECT_EQ(policy.plan.atomic_site_decisions.front().kind, SiteDecisionKind::NotApplicable);
    EXPECT_EQ(policy.plan.atomic_site_decisions.front().reason, expected);
    EXPECT_TRUE(policy.plan.probe_intents.empty());
  }
}

TEST(ConSanAtomicFencePolicy, ExcludesReleaseWithoutOwnerLocalWriteWindow) {
  static constexpr std::array kReadOnlyWindows = {
      DirectionalAccessAvailability{
          .owner = ProgramContainerId{},
          .read = true,
          .write = false,
      },
  };
  AtomicFencePolicyRequest request = atomic_request(Mode::Default);
  request.directional_access_windows = kReadOnlyWindows;

  const AtomicFencePolicyResult policy =
      plan_atomic_fence_observation(ordinary_fence_inventory(), request);

  ASSERT_TRUE(policy.valid());
  ASSERT_EQ(policy.plan.atomic_site_decisions.size(), 1u);
  EXPECT_EQ(policy.plan.atomic_site_decisions.front().kind, SiteDecisionKind::NotApplicable);
  EXPECT_EQ(policy.plan.atomic_site_decisions.front().reason,
            AtomicPolicyReason::MissingDirectionalAccessWindow);
  ASSERT_EQ(policy.plan.fence_site_decisions.size(), 1u);
  EXPECT_EQ(policy.plan.fence_site_decisions.front().kind, SiteDecisionKind::NotApplicable);
  EXPECT_EQ(policy.plan.fence_site_decisions.front().reason,
            FencePolicyReason::CommunicationNotApplicable);
  EXPECT_TRUE(policy.plan.probe_intents.empty());
}

TEST(ConSanAtomicFencePolicy, AssembledPolicyDerivesDirectionalOwnerLocalWindows) {
  const auto assemble = [](LdsAccessKind access_kind) {
    std::vector events{make_ordinary_store_event(), make_fence_event()};
    std::vector sequences{make_atomic_sequence(events.front())};
    std::vector fences{make_fence_candidate(events[0], events[1], sequences[0])};
    ProgramInventoryBuilder builder(build_atomic_inventory(
        std::move(events), std::move(sequences), {}, {make_global_store_site({})},
        std::move(fences), {}, {}, {make_lds_access(access_kind)}));
    for (ProgramSite &site : builder.program_sites()) {
      site.execution_owners.clear();
      site.execution_owners.push_back({.kernel = builder.kernels().front().id});
    }
    return assemble_observation_product(builder.view(),
                                        {.mode = Mode::Default,
                                         .native_lds_enabled = true,
                                         .group_flat_enabled = true,
                                         .flat_provenance_mode = FlatProvenanceMode::Likely,
                                         .barrier_tracking_enabled = true,
                                         .include_atomic_fence_policy = true,
                                         .atomic_fence_tracking_enabled = true,
                                         .container_filter = {},
                                         .kernel_name_allowlist = {},
                                         .reserved_for_synchronization = {}});
  };

  const ObservationProduct read_only = assemble(LdsAccessKind::Read);
  ASSERT_TRUE(read_only.valid());
  ASSERT_EQ(read_only.plan().site_decisions.size(), 1u);
  EXPECT_EQ(read_only.plan().site_decisions.front().kind, SiteDecisionKind::Admitted);
  ASSERT_EQ(read_only.plan().atomic_site_decisions.size(), 1u);
  EXPECT_EQ(read_only.plan().atomic_site_decisions.front().kind, SiteDecisionKind::NotApplicable);
  EXPECT_EQ(read_only.plan().atomic_site_decisions.front().reason,
            AtomicPolicyReason::MissingDirectionalAccessWindow);
  ASSERT_EQ(read_only.plan().fence_site_decisions.size(), 1u);
  EXPECT_EQ(read_only.plan().fence_site_decisions.front().kind, SiteDecisionKind::NotApplicable);
  EXPECT_EQ(read_only.plan().fence_site_decisions.front().reason,
            FencePolicyReason::CommunicationNotApplicable);
  EXPECT_EQ(std::ranges::count(read_only.plan().probe_intents, ProbeIntentKind::Access,
                               &ProbeIntent::kind),
            1u);
  EXPECT_EQ(read_only.plan().probe_intents.size(), 1u);

  const ObservationProduct write = assemble(LdsAccessKind::Write);
  ASSERT_TRUE(write.valid());
  ASSERT_EQ(write.plan().atomic_site_decisions.size(), 1u);
  EXPECT_EQ(write.plan().atomic_site_decisions.front().kind, SiteDecisionKind::Admitted);
  ASSERT_EQ(write.plan().fence_site_decisions.size(), 1u);
  EXPECT_EQ(write.plan().fence_site_decisions.front().kind, SiteDecisionKind::Admitted);
  EXPECT_EQ(write.plan().probe_intents.size(), 3u);
}

TEST(ConSanAtomicFencePolicy, GlobalAtomicContractTransportsAcrossEverySupportedArchitecture) {
  constexpr std::array targets = {
      AtomicPolicyTarget{ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_TARGET_GFX942},
      AtomicPolicyTarget{ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_TARGET_GFX950},
      AtomicPolicyTarget{ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_TARGET_GFX1100},
      AtomicPolicyTarget{ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_TARGET_GFX1201},
      AtomicPolicyTarget{ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_TARGET_GFX1250},
  };
  for (const AtomicPolicyTarget &target : targets) {
    SCOPED_TRACE(rj_code_target_name(target.target));
    const AtomicFencePolicyResult policy =
        plan_atomic_fence_observation(one_atomic_inventory(target), atomic_request(Mode::Default));
    ASSERT_TRUE(policy.valid());
    const AtomicSiteDecision &decision = policy.plan.atomic_site_decisions.front();
    EXPECT_EQ(decision.kind, SiteDecisionKind::Admitted);
    EXPECT_EQ(decision.capability, CapabilityDisposition::Supported);
    ASSERT_TRUE(policy.plan.probe_intents.front().atomic_lowering_form.has_value());
    EXPECT_EQ(policy.plan.probe_intents.front().atomic_lowering_form->kind,
              AtomicLoweringFormKind::GlobalScalarVectorAddress);
  }
}

TEST(ConSanAtomicFencePolicy, OrderedLdsAtomicIsTargetGatedToGfx1250) {
  constexpr AtomicPolicyTarget gfx1201{ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_TARGET_GFX1201};
  const AtomicFencePolicyResult unsupported = plan_atomic_fence_observation(
      one_atomic_inventory(gfx1201, SyncRmwOutcome::ReturnsOldValue, "ds_add_u32",
                           SyncAddressSource::LdsVector),
      atomic_request(Mode::Default));
  ASSERT_TRUE(unsupported.valid());
  EXPECT_EQ(unsupported.plan.atomic_site_decisions.front().kind, SiteDecisionKind::NotApplicable);
  EXPECT_EQ(unsupported.plan.atomic_site_decisions.front().reason,
            AtomicPolicyReason::TargetCapabilityUnavailable);

  constexpr AtomicPolicyTarget gfx1250{ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_TARGET_GFX1250};
  const AtomicFencePolicyResult supported = plan_atomic_fence_observation(
      one_atomic_inventory(gfx1250, SyncRmwOutcome::ReturnsOldValue, "ds_add_u32",
                           SyncAddressSource::LdsVector),
      atomic_request(Mode::Default));
  ASSERT_TRUE(supported.valid());
  EXPECT_EQ(supported.plan.atomic_site_decisions.front().kind, SiteDecisionKind::Admitted);
  EXPECT_EQ(supported.plan.atomic_site_decisions.front().capability,
            CapabilityDisposition::Supported);
  ASSERT_TRUE(supported.plan.probe_intents.front().atomic_lowering_form.has_value());
  EXPECT_EQ(supported.plan.probe_intents.front().atomic_lowering_form->kind,
            AtomicLoweringFormKind::LdsVectorOffset);
}

TEST(ConSanAtomicFencePolicy, Gfx1250OrderedLdsRequiresGraphNormalizedWorkgroupScope) {
  constexpr AtomicPolicyTarget gfx1250{ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_TARGET_GFX1250};
  std::vector events{make_atomic_event(32, SyncRmwOutcome::ReturnsOldValue,
                                       SyncAddressSource::LdsVector, "ds_add_u32")};
  std::vector sequences{make_atomic_sequence(events.front())};
  events.front().scope.reset();
  sequences.front().scope.reset();
  AtomicSite site = make_lds_atomic_site();
  site.scope.reset();
  const AtomicFencePolicyResult policy =
      plan_atomic_fence_observation(build_atomic_inventory(std::move(events), std::move(sequences),
                                                           {std::move(site)}, {}, {}, gfx1250),
                                    atomic_request(Mode::Default));
  ASSERT_TRUE(policy.valid());
  ASSERT_EQ(policy.plan.atomic_site_decisions.size(), 1u);
  EXPECT_EQ(policy.plan.atomic_site_decisions.front().kind, SiteDecisionKind::Unsupported);
  EXPECT_EQ(policy.plan.atomic_site_decisions.front().reason, AtomicPolicyReason::MissingScope);
}

TEST(ConSanAtomicFencePolicy, Gfx1250OrdinaryAcquireUsesItsDerivedWorkgroupScope) {
  constexpr AtomicPolicyTarget gfx1250{ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_TARGET_GFX1250};
  SyncEvent event = make_ordinary_store_event();
  event.operation = SyncOperation::OrdinaryLoad;
  event.memory_role = SyncMemoryRole::Acquire;
  event.scope = MemoryScope::Wavefront;
  SyncSequence sequence = make_atomic_sequence(event);
  sequence.memory_role = SyncMemoryRole::Acquire;
  sequence.scope = MemoryScope::Workgroup;
  OrdinaryMemorySite site = make_global_store_site(gfx1250);
  site.operation = OrdinaryMemoryOperation::Load;
  site.destination_vgpr = 2;
  site.data_vgpr.reset();
  site.raw_vsrc.reset();
  site.raw_vdst = 2;
  site.raw_saddr = 0x7cu;
  site.scope = MemoryScope::Wavefront;
  site.mnemonic = "flat_load_b32";

  const AtomicFencePolicyResult policy = plan_atomic_fence_observation(
      build_atomic_inventory({std::move(event)}, {std::move(sequence)}, {}, {std::move(site)}, {},
                             gfx1250),
      atomic_request(Mode::Default));
  ASSERT_TRUE(policy.valid());
  ASSERT_EQ(policy.plan.atomic_site_decisions.size(), 1u);
  EXPECT_EQ(policy.plan.atomic_site_decisions.front().kind, SiteDecisionKind::Admitted);
  EXPECT_EQ(policy.plan.atomic_site_decisions.front().reason, AtomicPolicyReason::None);
}

TEST(ConSanAtomicFencePolicy, CommunicationMaterializationUsesCanonicalSiteAndSequenceHandles) {
  SyncEvent event = make_ordinary_store_event();
  SyncSequence sequence = make_atomic_sequence(event);
  sequence.scope = MemoryScope::Workgroup;
  const ProgramInventory inventory =
      build_atomic_inventory({event}, {sequence}, {}, {make_global_store_site({})});
  const SynchronizationInventoryView sync = inventory.sync();
  const SyncEvent &published_event = sync.sync_events.front();

  const std::optional<AtomicSite> communication = detail::materialize_communication_site(
      inventory, published_event.source_site, SyncSequenceId{0u});
  ASSERT_TRUE(communication.has_value());
  EXPECT_EQ(communication->text_offset, event.text_offset());
  EXPECT_EQ(communication->width_bits, 32u);
  EXPECT_EQ(communication->scope, MemoryScope::Workgroup);

  detail::AtomicEvidenceSitePlan plan;
  plan.event = {0u};
  plan.sequence = {0u};
  plan.source_site = published_event.source_site;
  plan.address_capture_intent = {.value = 1u};
  plan.evidence_intent = {.value = 2u};
  plan.lowering_form.kind = AtomicLoweringFormKind::FlatVectorAddress;
  const std::optional<detail::AtomicEvidenceSourceView> source =
      detail::resolve_atomic_evidence_source(inventory, plan);
  ASSERT_TRUE(source.has_value());
  EXPECT_EQ(source->event, &published_event);
  EXPECT_EQ(source->sequence, &inventory.sync().sync_sequences.front());
  EXPECT_FALSE(source->is_rmw());
  EXPECT_EQ(source->site, *communication);

  detail::AtomicEvidenceSitePlan mismatched = plan;
  mismatched.event = {1u};
  EXPECT_FALSE(detail::resolve_atomic_evidence_source(inventory, mismatched).has_value());
  EXPECT_FALSE(
      detail::materialize_communication_site(inventory, {}, SyncSequenceId{0u}).has_value());
  EXPECT_FALSE(detail::materialize_communication_site(inventory, published_event.source_site, {})
                   .has_value());

  ProgramInventoryBuilder unsupported(inventory);
  OrdinaryMemorySite *ordinary =
      unsupported.program_sites()[published_event.source_site.ordinal].get_if<OrdinaryMemorySite>();
  ASSERT_NE(ordinary, nullptr);
  ordinary->support_reason = OrdinaryMemorySupportReason::MissingAddressVgpr;
  EXPECT_FALSE(detail::materialize_communication_site(
                   unsupported.view(), published_event.source_site, SyncSequenceId{0u})
                   .has_value());
}

TEST(ConSanAtomicFencePolicy, Gfx1250AdmitsExactBufferOrdinaryFenceCommunication) {
  constexpr AtomicPolicyTarget gfx1250{ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_TARGET_GFX1250};
  SyncEvent communication = make_ordinary_store_event();
  communication.operation = SyncOperation::OrdinaryLoad;
  communication.address_source = SyncAddressSource::BufferResource;
  communication.memory_role = SyncMemoryRole::Acquire;
  SyncEvent fence = make_fence_event();
  fence.memory_role = SyncMemoryRole::Acquire;
  SyncSequence sequence = make_atomic_sequence(communication);
  sequence.memory_role = SyncMemoryRole::Acquire;
  FenceCandidate candidate = make_fence_candidate(communication, fence, sequence);
  candidate.memory_role = SyncMemoryRole::Acquire;

  const auto make_inventory = [&](OrdinaryMemorySite site) {
    return build_atomic_inventory({communication, fence}, {sequence}, {}, {std::move(site)},
                                  {candidate}, gfx1250);
  };

  const AtomicFencePolicyResult supported = plan_atomic_fence_observation(
      make_inventory(make_cdna5_buffer_load_site()), atomic_request(Mode::Default));
  ASSERT_TRUE(supported.valid());
  ASSERT_EQ(supported.plan.atomic_site_decisions.size(), 1u);
  ASSERT_EQ(supported.plan.fence_site_decisions.size(), 1u);
  EXPECT_EQ(supported.plan.atomic_site_decisions.front().kind, SiteDecisionKind::Admitted);
  EXPECT_EQ(supported.plan.fence_site_decisions.front().kind, SiteDecisionKind::Admitted);

  OrdinaryMemorySite malformed = make_cdna5_buffer_load_site();
  malformed.raw_rsrc.reset();
  const AtomicFencePolicyResult rejected = plan_atomic_fence_observation(
      make_inventory(std::move(malformed)), atomic_request(Mode::Default));
  ASSERT_TRUE(rejected.valid());
  EXPECT_EQ(rejected.plan.atomic_site_decisions.front().kind, SiteDecisionKind::Unsupported);
  EXPECT_EQ(rejected.plan.atomic_site_decisions.front().reason,
            AtomicPolicyReason::UnsupportedEncoding);
  EXPECT_EQ(rejected.plan.fence_site_decisions.front().kind, SiteDecisionKind::Unsupported);
  EXPECT_EQ(rejected.plan.fence_site_decisions.front().reason,
            FencePolicyReason::MissingCommunicationEvent);
}

TEST(ConSanAtomicFencePolicy, EverySemanticQualificationFailureHasADistinctTypedReason) {
  using Mutation = std::function<void(SyncEvent &, SyncSequence &)>;
  const std::vector<std::tuple<std::string_view, Mutation, AtomicPolicyReason>> cases = {
      {"owner", [](auto &, auto &) {}, AtomicPolicyReason::MissingExecutionOwner},
      {"confidence",
       [](auto &, auto &sequence) { sequence.confidence = SemanticConfidence::Unsupported; },
       AtomicPolicyReason::UnqualifiedSyncSequence},
      {"role",
       [](auto &, auto &sequence) {
         sequence.memory_role = SyncMemoryRole::SequentiallyConsistent;
       },
       AtomicPolicyReason::MissingDirectionalAccessWindow},
      {"missing-scope", [](auto &, auto &sequence) { sequence.scope.reset(); },
       AtomicPolicyReason::MissingScope},
      {"scope", [](auto &, auto &sequence) { sequence.scope = MemoryScope::Wavefront; },
       AtomicPolicyReason::UnsupportedScope},
      {"outcome",
       [](auto &event, auto &sequence) {
         event.rmw_outcome = SyncRmwOutcome::Unknown;
         sequence.rmw_outcome = SyncRmwOutcome::Unknown;
       },
       AtomicPolicyReason::UnsupportedDynamicOutcome},
  };
  for (const auto &[name, mutate, expected] : cases) {
    SCOPED_TRACE(name);
    std::vector events{make_atomic_event()};
    std::vector sequences{make_atomic_sequence(events.front())};
    mutate(events.front(), sequences.front());
    const AtomicFencePolicyResult policy = plan_atomic_fence_observation(
        build_atomic_inventory(
            std::move(events), std::move(sequences), {make_global_atomic_site()}, {}, {}, {},
            name == "owner" ? std::vector<uint64_t>{32u} : std::vector<uint64_t>{}),
        atomic_request(Mode::Default));
    ASSERT_TRUE(policy.valid());
    ASSERT_EQ(policy.plan.atomic_site_decisions.size(), 1u);
    EXPECT_EQ(policy.plan.atomic_site_decisions.front().kind,
              expected == AtomicPolicyReason::MissingExecutionOwner ||
                      expected == AtomicPolicyReason::UnqualifiedSyncSequence ||
                      expected == AtomicPolicyReason::UnsupportedScope ||
                      expected == AtomicPolicyReason::MissingDirectionalAccessWindow
                  ? SiteDecisionKind::NotApplicable
                  : SiteDecisionKind::Unsupported);
    EXPECT_EQ(policy.plan.atomic_site_decisions.front().reason, expected);
    EXPECT_TRUE(policy.plan.probe_intents.empty());
  }
}

TEST(ConSanAtomicFencePolicy, AmbiguousSequenceMembershipFailsClosed) {
  std::vector events{make_atomic_event()};
  std::vector sequences{make_atomic_sequence(events.front(), "first"),
                        make_atomic_sequence(events.front(), "second")};
  const AtomicFencePolicyResult policy = plan_atomic_fence_observation(
      build_atomic_inventory(std::move(events), std::move(sequences), {make_global_atomic_site()}),
      atomic_request(Mode::Default));
  ASSERT_TRUE(policy.valid());
  EXPECT_EQ(policy.plan.atomic_site_decisions.front().kind, SiteDecisionKind::Unsupported);
  EXPECT_EQ(policy.plan.atomic_site_decisions.front().reason,
            AtomicPolicyReason::AmbiguousSequenceMembership);
}

TEST(ConSanAtomicFencePolicy, EncodingAndOperandFailuresRemainPolicyNotLoweringFacts) {
  using Mutation = std::function<void(AtomicSite &)>;
  const std::vector<std::tuple<std::string_view, Mutation, AtomicPolicyReason>> cases = {
      {"address-source", [](auto &site) { site.mnemonic = "buffer_atomic_add_u32"; },
       AtomicPolicyReason::UnsupportedAddressSource},
      {"width-zero", [](auto &site) { site.width_bits = 0; },
       AtomicPolicyReason::InvalidAccessWidth},
      {"width-bits", [](auto &site) { site.width_bits = 31; },
       AtomicPolicyReason::InvalidAccessWidth},
      {"size", [](auto &site) { site.size = 4; }, AtomicPolicyReason::UnsupportedEncoding},
      {"offset", [](auto &site) { site.raw_ioffset.reset(); },
       AtomicPolicyReason::UnsupportedEncoding},
      {"address", [](auto &site) { site.address_vgpr.reset(); },
       AtomicPolicyReason::MissingOperands},
      {"scope", [](auto &site) { site.scope.reset(); }, AtomicPolicyReason::MissingScope},
  };
  for (const auto &[name, mutate, expected] : cases) {
    SCOPED_TRACE(name);
    std::vector events{make_atomic_event()};
    std::vector sequences{make_atomic_sequence(events.front())};
    if (expected == AtomicPolicyReason::MissingScope) {
      events.front().scope.reset();
      sequences.front().scope.reset();
    }
    AtomicSite site = make_global_atomic_site();
    mutate(site);
    const AtomicFencePolicyResult policy = plan_atomic_fence_observation(
        build_atomic_inventory(std::move(events), std::move(sequences), {std::move(site)}),
        atomic_request(Mode::Default));
    ASSERT_TRUE(policy.valid());
    EXPECT_EQ(policy.plan.atomic_site_decisions.front().kind, SiteDecisionKind::Unsupported);
    EXPECT_EQ(policy.plan.atomic_site_decisions.front().reason, expected);
    EXPECT_TRUE(policy.plan.probe_intents.empty());
  }
}

TEST(ConSanAtomicFencePolicy, PlanValidationEnforcesAtomicAndFenceTypeRelationships) {
  const AtomicFencePolicyResult policy =
      plan_atomic_fence_observation(ordinary_fence_inventory(), atomic_request(Mode::Default));
  ASSERT_TRUE(policy.valid());

  ObservationPlan broken = policy.plan;
  broken.probe_intents.front().synchronization_association.reset();
  EXPECT_FALSE(broken.valid());
  broken = policy.plan;
  broken.probe_intents.front().synchronization_association = SynchronizationAssociationId{};
  EXPECT_FALSE(broken.valid());
  broken = policy.plan;
  broken.probe_intents.front().position = ProbePosition::After;
  EXPECT_FALSE(broken.valid());
  broken = policy.plan;
  broken.probe_intents.front().dynamic_result = DynamicResultRequirement::ReturnedOldValue;
  EXPECT_FALSE(broken.valid());
  broken = policy.plan;
  broken.probe_intents.back().position = ProbePosition::Before;
  EXPECT_FALSE(broken.valid());
  broken = policy.plan;
  broken.probe_intents.front().atomic_lowering_form.reset();
  EXPECT_FALSE(broken.valid());
  broken = policy.plan;
  broken.probe_intents.front().atomic_lowering_form->kind = AtomicLoweringFormKind::Count;
  EXPECT_FALSE(broken.valid());
  broken = policy.plan;
  broken.atomic_site_decisions.front().capability = static_cast<CapabilityDisposition>(255);
  EXPECT_FALSE(broken.valid());
  broken = policy.plan;
  broken.fence_site_decisions.front().inventory_association = FenceAssociation::Count;
  EXPECT_FALSE(broken.valid());
}

TEST(ConSanAtomicFencePolicy, AppendAndCoverageLedgerOwnDecisionsAndRebaseIntents) {
  const AtomicFencePolicyResult policy =
      plan_atomic_fence_observation(ordinary_fence_inventory(), atomic_request(Mode::Default));
  ASSERT_TRUE(policy.valid());
  ObservationPlan combined = policy.plan;
  ASSERT_TRUE(combined.append(policy.plan));
  ASSERT_TRUE(combined.valid());
  ASSERT_EQ(combined.probe_intents.size(), 4u);
  ASSERT_EQ(combined.atomic_site_decisions.size(), 2u);
  ASSERT_EQ(combined.fence_site_decisions.size(), 2u);
  EXPECT_EQ(combined.probe_intents[2].id, ProbeIntentId{2});
  EXPECT_EQ(combined.probe_intents[3].id, ProbeIntentId{3});

  CoverageLedger ledger(combined);
  EXPECT_TRUE(std::ranges::equal(ledger.atomic_site_decisions(), combined.atomic_site_decisions));
  EXPECT_TRUE(std::ranges::equal(ledger.fence_site_decisions(), combined.fence_site_decisions));
  ASSERT_EQ(ledger.intent_entries().size(), combined.probe_intents.size());
  EXPECT_TRUE(publish_test_lowering_outcome(ledger, {3}, LoweringOutcomeKind::Instrumented));
  ASSERT_NE(ledger.intent({3}), nullptr);
  EXPECT_EQ(ledger.intent({3})->kind, ProbeIntentKind::AtomicOrdering);
}

TEST(ConSanAtomicFencePolicy, OrdinaryFenceAssociationDefinesEachModeEvidenceContract) {
  const ProgramInventory inventory = ordinary_fence_inventory();

  const AtomicFencePolicyResult policy =
      plan_atomic_fence_observation(inventory, atomic_request(Mode::Default));
  ASSERT_TRUE(policy.valid());
  ASSERT_EQ(policy.plan.atomic_site_decisions.size(), 1u);
  ASSERT_EQ(policy.plan.fence_site_decisions.size(), 1u);
  const AtomicSiteDecision &atomic = policy.plan.atomic_site_decisions.front();
  const FenceSiteDecision &fence = policy.plan.fence_site_decisions.front();
  EXPECT_EQ(atomic.kind, SiteDecisionKind::Admitted);
  EXPECT_EQ(fence.kind, SiteDecisionKind::Admitted);
  EXPECT_EQ(policy.plan.probe_intents[0].synchronization_association,
            policy.plan.probe_intents[1].synchronization_association);
  ASSERT_TRUE(policy.plan.probe_intents.front().atomic_lowering_form.has_value());
  EXPECT_EQ(fence.inventory_association, FenceAssociation::Qualified);
  EXPECT_EQ(fence.capability, CapabilityDisposition::AssociatedOnly);
  ASSERT_EQ(policy.plan.probe_intents.size(), 2u);
  EXPECT_EQ(policy.plan.probe_intents[0].kind, ProbeIntentKind::AtomicAddressCapture);
  EXPECT_EQ(policy.plan.probe_intents[1].kind, ProbeIntentKind::AtomicOrdering);
}

TEST(ConSanAtomicFencePolicy, EveryFenceAssociationRejectionRemainsTypedInventoryEvidence) {
  for (uint8_t value = 0; value < static_cast<uint8_t>(FenceAssociation::Count); ++value) {
    const auto association = static_cast<FenceAssociation>(value);
    SCOPED_TRACE(value);
    const AtomicFencePolicyResult policy = plan_atomic_fence_observation(
        ordinary_fence_inventory(association), atomic_request(Mode::Default));
    ASSERT_TRUE(policy.valid());
    ASSERT_EQ(policy.plan.fence_site_decisions.size(), 1u);
    const FenceSiteDecision &decision = policy.plan.fence_site_decisions.front();
    EXPECT_EQ(decision.inventory_association, association);
    if (association == FenceAssociation::Qualified) {
      EXPECT_EQ(decision.kind, SiteDecisionKind::Admitted);
      EXPECT_EQ(decision.reason, FencePolicyReason::None);
    } else {
      EXPECT_EQ(decision.kind, SiteDecisionKind::NotApplicable);
      EXPECT_EQ(decision.reason, FencePolicyReason::AssociationUnavailable);
    }
  }
}

TEST(ConSanAtomicFencePolicy, FenceRequestExclusionsAndMissingFactsRemainTyped) {
  const ProgramInventory inventory = ordinary_fence_inventory();

  AtomicFencePolicyRequest disabled = atomic_request(Mode::Default);
  disabled.tracking_enabled = false;
  EXPECT_EQ(
      plan_atomic_fence_observation(inventory, disabled).plan.fence_site_decisions.front().reason,
      FencePolicyReason::TrackingDisabled);

  EXPECT_EQ(plan_atomic_fence_observation(inventory, atomic_request(Mode::SuperCollider))
                .plan.fence_site_decisions.front()
                .reason,
            FencePolicyReason::ModeMutationOnly);

  AtomicFencePolicyRequest filtered = atomic_request(Mode::Default);
  filtered.container_filter = "different";
  EXPECT_EQ(
      plan_atomic_fence_observation(inventory, filtered).plan.fence_site_decisions.front().reason,
      FencePolicyReason::ContainerFilterExcluded);

  std::vector owner_events{make_ordinary_store_event(), make_fence_event()};
  std::vector owner_sequences{make_atomic_sequence(owner_events.front())};
  std::vector owner_fences{
      make_fence_candidate(owner_events[0], owner_events[1], owner_sequences[0])};
  const AtomicFencePolicyResult missing_owner = plan_atomic_fence_observation(
      build_atomic_inventory(std::move(owner_events), std::move(owner_sequences), {},
                             {make_global_store_site({})}, std::move(owner_fences), {}, {48u}),
      atomic_request(Mode::Default));
  ASSERT_TRUE(missing_owner.valid());
  EXPECT_EQ(missing_owner.plan.fence_site_decisions.front().kind, SiteDecisionKind::NotApplicable);
  EXPECT_EQ(missing_owner.plan.fence_site_decisions.front().reason,
            FencePolicyReason::MissingExecutionOwner);

  std::vector missing_events{make_ordinary_store_event(), make_fence_event()};
  std::vector missing_sequences{make_atomic_sequence(missing_events.front())};
  FenceCandidate missing_fence =
      make_fence_candidate(missing_events[0], missing_events[1], missing_sequences[0]);
  missing_fence.communication_event = SyncEventId{999u};
  const AtomicFencePolicyResult missing_communication = plan_atomic_fence_observation(
      build_atomic_inventory(std::move(missing_events), std::move(missing_sequences), {},
                             {make_global_store_site({})}, {std::move(missing_fence)}),
      atomic_request(Mode::Default));
  ASSERT_TRUE(missing_communication.valid());
  EXPECT_EQ(missing_communication.plan.fence_site_decisions.front().kind,
            SiteDecisionKind::Unsupported);
  EXPECT_EQ(missing_communication.plan.fence_site_decisions.front().reason,
            FencePolicyReason::MissingCommunicationEvent);
}

TEST(ConSanAtomicFencePolicy, ConflictingFenceAliasesProduceTypedFatalError) {
  std::vector events{make_ordinary_store_event(), make_fence_event()};
  std::vector sequences{make_atomic_sequence(events.front())};
  events[1].source_site = {1};
  SyncEvent fence_alias = events[1];
  fence_alias.source_site = {2};
  fence_alias.identity = "aliased_kernel|fence=48";
  events.push_back(std::move(fence_alias));
  FenceCandidate first = make_fence_candidate(events[0], events[1], sequences[0]);
  FenceCandidate conflict = first;
  conflict.memory_role = SyncMemoryRole::Acquire;
  const ProgramInventory inventory = build_atomic_inventory(
      std::move(events), std::move(sequences), {}, {make_global_store_site({})}, {first, conflict});
  const AtomicFencePolicyResult policy =
      plan_atomic_fence_observation(inventory, atomic_request(Mode::Default));
  EXPECT_FALSE(policy.valid());
  EXPECT_EQ(policy.fence_errors, (std::vector{FencePolicyReason::ConflictingPhysicalAliases}));
  ASSERT_EQ(policy.plan.fence_site_decisions.size(), 1u);
  EXPECT_EQ(policy.plan.fence_site_decisions.front().kind, SiteDecisionKind::Unsupported);
  EXPECT_EQ(policy.plan.fence_site_decisions.front().reason,
            FencePolicyReason::ConflictingPhysicalAliases);
  EXPECT_EQ(inventory.source_container_names(
                policy.plan.fence_site_decisions.front().semantic_site.physical),
            (std::vector<std::string>{"aliased_kernel", "atomic_kernel"}));
}

TEST(ConSanAtomicFencePolicy, ConflictingPhysicalAliasesProduceTypedFatalError) {
  std::vector events{make_atomic_event()};
  SyncEvent alias = events.front();
  alias.identity = "aliased_kernel|atomic=32";
  events.front().source_site = {0};
  alias.source_site = {1};
  events.push_back(std::move(alias));
  std::vector sequences{make_atomic_sequence(events.front())};
  AtomicSite conflicting_site = make_global_atomic_site();
  conflicting_site.width_bits = 64;
  const ProgramInventory inventory =
      build_atomic_inventory(std::move(events), std::move(sequences),
                             {make_global_atomic_site(), std::move(conflicting_site)});
  const AtomicFencePolicyResult policy =
      plan_atomic_fence_observation(inventory, atomic_request(Mode::Default));
  EXPECT_FALSE(policy.valid());
  EXPECT_EQ(policy.atomic_errors, (std::vector{AtomicPolicyReason::ConflictingPhysicalAliases}));
  ASSERT_EQ(policy.plan.atomic_site_decisions.size(), 1u);
  EXPECT_EQ(policy.plan.atomic_site_decisions.front().kind, SiteDecisionKind::Unsupported);
  EXPECT_EQ(inventory.source_container_names(
                policy.plan.atomic_site_decisions.front().semantic_site.physical),
            (std::vector<std::string>{"aliased_kernel", "atomic_kernel"}));
  EXPECT_TRUE(policy.plan.probe_intents.empty());
}

TEST(ConSanAtomicFencePolicy, PolicyIsDeterministicAndDoesNotMutatePublishedInventory) {
  const ProgramInventory inventory = ordinary_fence_inventory();
  const SynchronizationInventoryView before = inventory.sync();
  std::vector<std::string> event_identities_before;
  for (const SyncEvent &event : before.sync_events)
    event_identities_before.push_back(event.identity);
  std::vector<std::string> sequence_identities_before;
  for (const SyncSequence &sequence : before.sync_sequences)
    sequence_identities_before.push_back(sequence.identity);
  const AtomicFencePolicyRequest request = atomic_request(Mode::Default);
  const AtomicFencePolicyResult first = plan_atomic_fence_observation(inventory, request);
  const AtomicFencePolicyResult second = plan_atomic_fence_observation(inventory, request);
  EXPECT_EQ(first, second);
  EXPECT_TRUE(first.valid());
  const SynchronizationInventoryView after = inventory.sync();
  std::vector<std::string> event_identities_after;
  for (const SyncEvent &event : after.sync_events)
    event_identities_after.push_back(event.identity);
  std::vector<std::string> sequence_identities_after;
  for (const SyncSequence &sequence : after.sync_sequences)
    sequence_identities_after.push_back(sequence.identity);
  EXPECT_EQ(event_identities_after, event_identities_before);
  EXPECT_EQ(sequence_identities_after, sequence_identities_before);
}

} // namespace
} // namespace rocjitsu::consan
