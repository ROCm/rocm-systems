// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"

#include <functional>

namespace rocjitsu {
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

ConSanAtomicSite
make_global_atomic_site(const AtomicPolicyTarget &target = {}, uint64_t offset = 32,
                        std::string mnemonic = "global_atomic_add_u32",
                        ConSanSyncRmwOutcome outcome = ConSanSyncRmwOutcome::ReturnsOldValue) {
  ConSanAtomicSite site;
  site.text_offset = offset;
  site.file_offset = offset;
  site.size = consan_uses_gfx12_encoding(target.arch) ? 12u : 8u;
  site.width_bits = 32;
  site.dst_vgpr = 1;
  site.addr_vgpr = 0;
  site.data_vgpr = 2;
  site.saddr_sgpr = 4;
  site.raw_saddr = 4;
  site.raw_vaddr = 0;
  site.raw_vdata = 2;
  site.raw_ioffset = 0;
  site.raw_scope = 2;
  site.raw_th = 0;
  site.returns_old_value = outcome != ConSanSyncRmwOutcome::NoReturn;
  site.mnemonic = std::move(mnemonic);
  return site;
}

ConSanAtomicSite make_lds_atomic_site(uint64_t offset = 32) {
  ConSanAtomicSite site;
  site.address_space_hint = ConSanAtomicAddressSpaceHint::Lds;
  site.text_offset = offset;
  site.file_offset = offset;
  site.size = 8;
  site.width_bits = 32;
  site.dst_vgpr = 1;
  site.addr_vgpr = 0;
  site.data_vgpr = 2;
  site.raw_scope = 1;
  site.returns_old_value = true;
  site.mnemonic = "ds_add_u32";
  return site;
}

ConSanOrdinaryMemorySite make_global_store_site(const AtomicPolicyTarget &target,
                                                uint64_t offset = 32) {
  ConSanOrdinaryMemorySite site;
  site.operation = ConSanOrdinaryMemoryOperation::Store;
  site.support_reason = ConSanOrdinaryMemorySupportReason::SupportedSynchronizationOnly;
  site.text_offset = offset;
  site.file_offset = offset;
  site.size = consan_uses_gfx12_encoding(target.arch) ? 12u : 8u;
  site.width_bits = 32;
  site.address_vgpr = 0;
  site.address_sgpr = 4;
  site.value_vgpr = 2;
  site.raw_saddr = 4;
  site.raw_vaddr = 0;
  site.raw_vsrc = 2;
  site.raw_ioffset = 0;
  site.raw_scope = 2;
  site.raw_th = 0;
  site.mnemonic = "global_store_b32";
  return site;
}

ConSanOrdinaryMemorySite make_cdna5_buffer_load_site(uint64_t offset = 32) {
  ConSanOrdinaryMemorySite site;
  site.operation = ConSanOrdinaryMemoryOperation::Load;
  site.support_reason = ConSanOrdinaryMemorySupportReason::SupportedSynchronizationOnly;
  site.text_offset = offset;
  site.file_offset = offset;
  site.size = 12;
  site.width_bits = 32;
  site.destination_vgpr = 4;
  site.address_vgpr = 5;
  site.address_sgpr = 24;
  site.raw_rsrc = 24;
  site.raw_soffset = 0x7c;
  site.raw_vaddr = 5;
  site.raw_vdst = 4;
  site.raw_ioffset = 0;
  site.raw_scope = 2;
  site.raw_offen = true;
  site.raw_idxen = false;
  site.mnemonic = "buffer_load_b32";
  return site;
}

ConSanSyncEvent make_atomic_event(
    uint64_t offset = 32, ConSanSyncRmwOutcome outcome = ConSanSyncRmwOutcome::ReturnsOldValue,
    ConSanSyncAddressSource address_source = ConSanSyncAddressSource::GlobalScalarVector,
    std::string mnemonic = "global_atomic_add_u32", std::string container = "atomic_kernel") {
  ConSanSyncEvent event;
  event.semantic_id = {
      .physical =
          {
              .code_object = make_consan_code_object_id(atomic_policy_bytes()),
              .original_text_offset = offset,
          },
      .domain = ConSanSemanticSiteDomain::SynchronizationEvent,
  };
  event.kind = ConSanSyncEventKind::Atomic;
  event.operation = mnemonic.find("cmp") == std::string::npos
                        ? ConSanSyncOperation::AtomicRmw
                        : ConSanSyncOperation::AtomicCompareExchange;
  event.address_source = address_source;
  event.memory_role = ConSanSyncMemoryRole::AcquireRelease;
  event.rmw_outcome = outcome;
  event.confidence = ConSanSemanticConfidence::Exact;
  event.memory_role_confidence = ConSanSemanticConfidence::Exact;
  event.identity = container + "|atomic=" + std::to_string(offset);
  event.code_object_fingerprint = event.semantic_id.physical.code_object.fingerprint;
  event.container_name = std::move(container);
  event.in_kernel = true;
  event.text_offset = offset;
  event.file_offset = offset;
  event.size = 12;
  event.width_bits = 32;
  event.mnemonic = std::move(mnemonic);
  event.raw_scope = address_source == ConSanSyncAddressSource::LdsVector ? 1u : 2u;
  event.execution_owners.push_back({});
  return event;
}

ConSanSyncEvent make_ordinary_store_event(uint64_t offset = 32) {
  ConSanSyncEvent event =
      make_atomic_event(offset, ConSanSyncRmwOutcome::NotApplicable,
                        ConSanSyncAddressSource::GlobalScalarVector, "global_store_b32");
  event.kind = ConSanSyncEventKind::OrdinaryMemory;
  event.operation = ConSanSyncOperation::OrdinaryStore;
  event.memory_role = ConSanSyncMemoryRole::Release;
  event.identity = "atomic_kernel|ordinary-store=" + std::to_string(offset);
  return event;
}

ConSanSyncEvent make_fence_event(uint64_t offset = 48) {
  ConSanSyncEvent event = make_atomic_event(offset, ConSanSyncRmwOutcome::NotApplicable,
                                            ConSanSyncAddressSource::NotApplicable, "global_wb");
  event.kind = ConSanSyncEventKind::Fence;
  event.operation = ConSanSyncOperation::Fence;
  event.memory_role = ConSanSyncMemoryRole::Release;
  event.identity = "atomic_kernel|fence=" + std::to_string(offset);
  event.width_bits = 0;
  event.raw_scope.reset();
  return event;
}

ConSanSyncSequence make_atomic_sequence(const ConSanSyncEvent &event,
                                        std::string identity = "atomic-sequence") {
  ConSanSyncSequence sequence;
  sequence.kind = event.kind == ConSanSyncEventKind::OrdinaryMemory
                      ? ConSanSyncSequenceKind::OrdinaryMemory
                      : ConSanSyncSequenceKind::Atomic;
  sequence.operation = event.operation;
  sequence.address_source = event.address_source;
  sequence.memory_role = event.memory_role;
  sequence.rmw_outcome = event.rmw_outcome;
  sequence.confidence = ConSanSemanticConfidence::Exact;
  sequence.memory_role_confidence = ConSanSemanticConfidence::Exact;
  sequence.identity = std::move(identity);
  sequence.container_name = event.container_name;
  sequence.in_kernel = event.in_kernel;
  sequence.begin_text_offset = event.text_offset;
  sequence.end_text_offset = event.text_offset + event.size;
  sequence.basic_block_index = 0;
  SemanticSiteId member = event.semantic_id;
  member.domain = ConSanSemanticSiteDomain::SynchronizationSequenceMember;
  sequence.member_semantic_ids.push_back(member);
  sequence.member_event_identities.push_back(event.identity);
  sequence.width_bits = event.width_bits;
  sequence.raw_scope = event.raw_scope;
  sequence.execution_owners = event.execution_owners;
  return sequence;
}

ConSanMoiFenceCandidate
make_fence_candidate(const ConSanSyncEvent &communication, const ConSanSyncEvent &fence,
                     const ConSanSyncSequence &sequence,
                     ConSanFenceAssociation association = ConSanFenceAssociation::Qualified) {
  return {
      .fence_event = fence.semantic_id,
      .sequence_identity = sequence.identity,
      .communication_event = communication.semantic_id,
      .memory_role = fence.memory_role,
      .association = association,
  };
}

ProgramInventory build_atomic_inventory(std::vector<ConSanSyncEvent> events,
                                        std::vector<ConSanSyncSequence> sequences,
                                        std::vector<ConSanAtomicSite> atomic_sites,
                                        std::vector<ConSanOrdinaryMemorySite> ordinary_sites = {},
                                        std::vector<ConSanMoiFenceCandidate> fences = {},
                                        const AtomicPolicyTarget &target = {}) {
  ProgramInventoryBuilder builder(atomic_policy_bytes());
  builder.set_code_object_facts(true, 0, target.arch, target.target);
  ConSanKernelInfo kernel;
  kernel.name = "atomic_kernel";
  kernel.descriptor_file_offset = 384;
  kernel.entry_text_offset = 0;
  kernel.atomic_sites = std::move(atomic_sites);
  kernel.ordinary_memory_sites = std::move(ordinary_sites);
  builder.kernels().push_back(std::move(kernel));
  SynchronizationInventoryBuildView synchronization = builder.synchronization();
  synchronization.sync_events = std::move(events);
  synchronization.sync_sequences = std::move(sequences);
  synchronization.moi_fence_candidates = std::move(fences);
  return builder.view();
}

ProgramInventory one_atomic_inventory(
    const AtomicPolicyTarget &target = {},
    ConSanSyncRmwOutcome outcome = ConSanSyncRmwOutcome::ReturnsOldValue,
    std::string mnemonic = "global_atomic_add_u32",
    ConSanSyncAddressSource address_source = ConSanSyncAddressSource::GlobalScalarVector) {
  std::vector events{make_atomic_event(32, outcome, address_source, mnemonic)};
  std::vector sequences{make_atomic_sequence(events.front())};
  ConSanAtomicSite site = address_source == ConSanSyncAddressSource::LdsVector
                              ? make_lds_atomic_site()
                              : make_global_atomic_site(target, 32, std::move(mnemonic), outcome);
  return build_atomic_inventory(std::move(events), std::move(sequences), {std::move(site)}, {}, {},
                                target);
}

ProgramInventory
ordinary_fence_inventory(ConSanFenceAssociation association = ConSanFenceAssociation::Qualified,
                         const AtomicPolicyTarget &target = {}) {
  std::vector events{make_ordinary_store_event(), make_fence_event()};
  std::vector sequences{make_atomic_sequence(events.front())};
  std::vector fences{make_fence_candidate(events[0], events[1], sequences[0], association)};
  return build_atomic_inventory(std::move(events), std::move(sequences), {},
                                {make_global_store_site(target)}, std::move(fences), target);
}

ConSanAtomicFencePolicyRequest atomic_request(ConSanCapabilityEngine engine) {
  return {
      .engine = engine,
      .tracking_enabled = true,
      .sampled_access_window_available = true,
      .container_filter = {},
  };
}

TEST(ConSanAtomicFencePolicy, InvalidEngineFailsValidationButEmptyInventoryIsAValidEmptyPlan) {
  const ConSanAtomicFencePolicyResult invalid = plan_consan_atomic_fence_observation(
      one_atomic_inventory(), atomic_request(ConSanCapabilityEngine::Count));
  EXPECT_FALSE(invalid.valid());
  EXPECT_FALSE(invalid.plan.valid());

  const ConSanAtomicFencePolicyResult empty = plan_consan_atomic_fence_observation(
      ProgramInventory{}, atomic_request(ConSanCapabilityEngine::RecordReplay));
  EXPECT_TRUE(empty.valid());
  EXPECT_TRUE(empty.plan.atomic_site_decisions.empty());
  EXPECT_TRUE(empty.plan.fence_site_decisions.empty());
  EXPECT_TRUE(empty.plan.probe_intents.empty());
}

TEST(ConSanAtomicFencePolicy, AllEnginesExpressTheirAtomicObservationContract) {
  const ProgramInventory inventory = one_atomic_inventory();
  const ConSanAtomicFencePolicyResult supercollider = plan_consan_atomic_fence_observation(
      inventory, atomic_request(ConSanCapabilityEngine::SuperCollider));
  ASSERT_TRUE(supercollider.valid());
  ASSERT_EQ(supercollider.plan.atomic_site_decisions.size(), 1u);
  EXPECT_EQ(supercollider.plan.atomic_site_decisions.front().kind,
            ConSanSiteDecisionKind::NotApplicable);
  EXPECT_EQ(supercollider.plan.atomic_site_decisions.front().reason,
            ConSanAtomicPolicyReason::EngineMutationOnly);
  EXPECT_TRUE(supercollider.plan.probe_intents.empty());

  constexpr std::array expected = {
      std::pair{ConSanCapabilityEngine::RecordReplay, ConSanProbeIntentKind::AtomicRecord},
      std::pair{ConSanCapabilityEngine::Sampled, ConSanProbeIntentKind::SampledAtomicOrdering},
      std::pair{ConSanCapabilityEngine::InlineShadow, ConSanProbeIntentKind::ExactAtomicOrdering},
  };
  for (const auto &[engine, evidence_kind] : expected) {
    SCOPED_TRACE(consan_capability_engine_name(engine));
    const ConSanAtomicFencePolicyResult policy =
        plan_consan_atomic_fence_observation(inventory, atomic_request(engine));
    ASSERT_TRUE(policy.valid());
    ASSERT_EQ(policy.plan.atomic_site_decisions.size(), 1u);
    ASSERT_EQ(policy.plan.probe_intents.size(), 2u);
    const ConSanAtomicSiteDecision &decision = policy.plan.atomic_site_decisions.front();
    EXPECT_EQ(decision.kind, ConSanSiteDecisionKind::Admitted);
    EXPECT_EQ(decision.reason, ConSanAtomicPolicyReason::None);
    ASSERT_TRUE(decision.association.has_value());
    EXPECT_EQ(decision.intent_ids, (std::vector{ConSanProbeIntentId{0}, ConSanProbeIntentId{1}}));
    EXPECT_EQ(policy.plan.probe_intents[0].kind, ConSanProbeIntentKind::AtomicAddressCapture);
    EXPECT_EQ(policy.plan.probe_intents[0].position, ConSanProbePosition::Before);
    EXPECT_EQ(policy.plan.probe_intents[0].dynamic_result, ConSanDynamicResultRequirement::None);
    EXPECT_EQ(policy.plan.probe_intents[1].kind, evidence_kind);
    EXPECT_EQ(policy.plan.probe_intents[1].position, ConSanProbePosition::After);
    EXPECT_EQ(policy.plan.probe_intents[1].dynamic_result,
              ConSanDynamicResultRequirement::ReturnedOldValue);
    EXPECT_EQ(policy.plan.probe_intents[0].synchronization_association,
              policy.plan.probe_intents[1].synchronization_association);
  }
}

TEST(ConSanAtomicFencePolicy, DynamicAtomicOutcomesBecomeExplicitAfterIntentRequirements) {
  constexpr std::array cases = {
      std::tuple{ConSanSyncRmwOutcome::NoReturn, "global_atomic_add_u32",
                 ConSanDynamicResultRequirement::None},
      std::tuple{ConSanSyncRmwOutcome::ReturnsOldValue, "global_atomic_add_u32",
                 ConSanDynamicResultRequirement::ReturnedOldValue},
      std::tuple{ConSanSyncRmwOutcome::CompareExchange, "global_atomic_cmpswap_u32",
                 ConSanDynamicResultRequirement::CompareExchangeSuccess},
  };
  for (const auto &[outcome, mnemonic, expected] : cases) {
    SCOPED_TRACE(mnemonic);
    const ConSanAtomicFencePolicyResult policy =
        plan_consan_atomic_fence_observation(one_atomic_inventory({}, outcome, mnemonic),
                                             atomic_request(ConSanCapabilityEngine::InlineShadow));
    ASSERT_TRUE(policy.valid());
    ASSERT_EQ(policy.plan.probe_intents.size(), 2u);
    EXPECT_EQ(policy.plan.atomic_site_decisions.front().dynamic_result, expected);
    EXPECT_EQ(policy.plan.probe_intents[1].dynamic_result, expected);
  }
}

TEST(ConSanAtomicFencePolicy, RequestExclusionsRemainTypedAndDoNotCreateIntents) {
  const ProgramInventory inventory = one_atomic_inventory();
  using Mutation = std::function<void(ConSanAtomicFencePolicyRequest &)>;
  const std::vector<
      std::tuple<std::string_view, ConSanCapabilityEngine, Mutation, ConSanAtomicPolicyReason>>
      cases = {
          {"disabled", ConSanCapabilityEngine::RecordReplay,
           [](auto &request) { request.tracking_enabled = false; },
           ConSanAtomicPolicyReason::TrackingDisabled},
          {"filtered", ConSanCapabilityEngine::RecordReplay,
           [](auto &request) { request.container_filter = "different"; },
           ConSanAtomicPolicyReason::ContainerFilterExcluded},
          {"sampled-window", ConSanCapabilityEngine::Sampled,
           [](auto &request) { request.sampled_access_window_available = false; },
           ConSanAtomicPolicyReason::MissingSampledAccessWindow},
      };
  for (const auto &[name, engine, mutate, expected] : cases) {
    SCOPED_TRACE(name);
    ConSanAtomicFencePolicyRequest request = atomic_request(engine);
    mutate(request);
    const ConSanAtomicFencePolicyResult policy =
        plan_consan_atomic_fence_observation(inventory, request);
    ASSERT_TRUE(policy.valid());
    ASSERT_EQ(policy.plan.atomic_site_decisions.size(), 1u);
    EXPECT_EQ(policy.plan.atomic_site_decisions.front().kind,
              ConSanSiteDecisionKind::NotApplicable);
    EXPECT_EQ(policy.plan.atomic_site_decisions.front().reason, expected);
    EXPECT_TRUE(policy.plan.probe_intents.empty());
  }
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
    const ConSanAtomicFencePolicyResult policy = plan_consan_atomic_fence_observation(
        one_atomic_inventory(target), atomic_request(ConSanCapabilityEngine::InlineShadow));
    ASSERT_TRUE(policy.valid());
    EXPECT_EQ(policy.plan.atomic_site_decisions.front().kind, ConSanSiteDecisionKind::Admitted);
    EXPECT_EQ(policy.plan.atomic_site_decisions.front().capability,
              ConSanCapabilityDisposition::Supported);
  }
}

TEST(ConSanAtomicFencePolicy, OrderedLdsAtomicIsTargetGatedToGfx1250) {
  constexpr AtomicPolicyTarget gfx1201{ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_TARGET_GFX1201};
  const ConSanAtomicFencePolicyResult unsupported = plan_consan_atomic_fence_observation(
      one_atomic_inventory(gfx1201, ConSanSyncRmwOutcome::ReturnsOldValue, "ds_add_u32",
                           ConSanSyncAddressSource::LdsVector),
      atomic_request(ConSanCapabilityEngine::RecordReplay));
  ASSERT_TRUE(unsupported.valid());
  EXPECT_EQ(unsupported.plan.atomic_site_decisions.front().kind,
            ConSanSiteDecisionKind::NotApplicable);
  EXPECT_EQ(unsupported.plan.atomic_site_decisions.front().reason,
            ConSanAtomicPolicyReason::TargetCapabilityUnavailable);

  constexpr AtomicPolicyTarget gfx1250{ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_TARGET_GFX1250};
  const ConSanAtomicFencePolicyResult supported = plan_consan_atomic_fence_observation(
      one_atomic_inventory(gfx1250, ConSanSyncRmwOutcome::ReturnsOldValue, "ds_add_u32",
                           ConSanSyncAddressSource::LdsVector),
      atomic_request(ConSanCapabilityEngine::RecordReplay));
  ASSERT_TRUE(supported.valid());
  EXPECT_EQ(supported.plan.atomic_site_decisions.front().kind, ConSanSiteDecisionKind::Admitted);
  EXPECT_EQ(supported.plan.atomic_site_decisions.front().capability,
            ConSanCapabilityDisposition::Supported);
}

TEST(ConSanAtomicFencePolicy, Gfx1250OrderedLdsDerivesItsImplicitWorkgroupScope) {
  constexpr AtomicPolicyTarget gfx1250{ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_TARGET_GFX1250};
  std::vector events{make_atomic_event(32, ConSanSyncRmwOutcome::ReturnsOldValue,
                                       ConSanSyncAddressSource::LdsVector, "ds_add_u32")};
  std::vector sequences{make_atomic_sequence(events.front())};
  events.front().raw_scope.reset();
  sequences.front().raw_scope.reset();
  ConSanAtomicSite site = make_lds_atomic_site();
  site.raw_scope.reset();
  const ConSanAtomicFencePolicyResult policy = plan_consan_atomic_fence_observation(
      build_atomic_inventory(std::move(events), std::move(sequences), {std::move(site)}, {}, {},
                             gfx1250),
      atomic_request(ConSanCapabilityEngine::RecordReplay));
  ASSERT_TRUE(policy.valid());
  EXPECT_EQ(policy.plan.atomic_site_decisions.front().kind, ConSanSiteDecisionKind::Admitted);
}

TEST(ConSanAtomicFencePolicy, Gfx1250OrdinaryAcquireUsesItsDerivedWorkgroupScope) {
  constexpr AtomicPolicyTarget gfx1250{ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_TARGET_GFX1250};
  ConSanSyncEvent event = make_ordinary_store_event();
  event.operation = ConSanSyncOperation::OrdinaryLoad;
  event.memory_role = ConSanSyncMemoryRole::Acquire;
  event.mnemonic = "flat_load_b32";
  event.raw_scope = 0u;
  ConSanSyncSequence sequence = make_atomic_sequence(event);
  sequence.memory_role = ConSanSyncMemoryRole::Acquire;
  sequence.raw_scope = 1u;
  ConSanOrdinaryMemorySite site = make_global_store_site(gfx1250);
  site.operation = ConSanOrdinaryMemoryOperation::Load;
  site.destination_vgpr = 2;
  site.value_vgpr.reset();
  site.raw_vsrc.reset();
  site.raw_vdst = 2;
  site.raw_saddr = 0x7cu;
  site.raw_scope = 0u;
  site.mnemonic = "flat_load_b32";

  const ConSanAtomicFencePolicyResult policy = plan_consan_atomic_fence_observation(
      build_atomic_inventory({std::move(event)}, {std::move(sequence)}, {}, {std::move(site)}, {},
                             gfx1250),
      atomic_request(ConSanCapabilityEngine::RecordReplay));
  ASSERT_TRUE(policy.valid());
  ASSERT_EQ(policy.plan.atomic_site_decisions.size(), 1u);
  EXPECT_EQ(policy.plan.atomic_site_decisions.front().kind, ConSanSiteDecisionKind::Admitted);
  EXPECT_EQ(policy.plan.atomic_site_decisions.front().reason, ConSanAtomicPolicyReason::None);
}

TEST(ConSanAtomicFencePolicy, Gfx1250RecordReplayAdmitsExactBufferOrdinaryFenceCommunication) {
  constexpr AtomicPolicyTarget gfx1250{ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_TARGET_GFX1250};
  ConSanSyncEvent communication = make_ordinary_store_event();
  communication.operation = ConSanSyncOperation::OrdinaryLoad;
  communication.address_source = ConSanSyncAddressSource::BufferResource;
  communication.memory_role = ConSanSyncMemoryRole::Acquire;
  communication.mnemonic = "buffer_load_b32";
  ConSanSyncEvent fence = make_fence_event();
  fence.memory_role = ConSanSyncMemoryRole::Acquire;
  fence.mnemonic = "global_inv";
  ConSanSyncSequence sequence = make_atomic_sequence(communication);
  sequence.memory_role = ConSanSyncMemoryRole::Acquire;
  ConSanMoiFenceCandidate candidate = make_fence_candidate(communication, fence, sequence);
  candidate.memory_role = ConSanSyncMemoryRole::Acquire;

  const auto make_inventory = [&](ConSanOrdinaryMemorySite site) {
    return build_atomic_inventory({communication, fence}, {sequence}, {}, {std::move(site)},
                                  {candidate}, gfx1250);
  };
  const ConSanAtomicFencePolicyResult supported =
      plan_consan_atomic_fence_observation(make_inventory(make_cdna5_buffer_load_site()),
                                           atomic_request(ConSanCapabilityEngine::RecordReplay));
  ASSERT_TRUE(supported.valid());
  ASSERT_EQ(supported.plan.atomic_site_decisions.size(), 1u);
  ASSERT_EQ(supported.plan.fence_site_decisions.size(), 1u);
  EXPECT_EQ(supported.plan.atomic_site_decisions.front().kind, ConSanSiteDecisionKind::Admitted);
  EXPECT_EQ(supported.plan.fence_site_decisions.front().kind, ConSanSiteDecisionKind::Admitted);

  ConSanOrdinaryMemorySite malformed = make_cdna5_buffer_load_site();
  malformed.raw_rsrc.reset();
  const ConSanAtomicFencePolicyResult rejected = plan_consan_atomic_fence_observation(
      make_inventory(std::move(malformed)), atomic_request(ConSanCapabilityEngine::RecordReplay));
  ASSERT_TRUE(rejected.valid());
  EXPECT_EQ(rejected.plan.atomic_site_decisions.front().kind, ConSanSiteDecisionKind::Unsupported);
  EXPECT_EQ(rejected.plan.atomic_site_decisions.front().reason,
            ConSanAtomicPolicyReason::UnsupportedEncoding);
  EXPECT_EQ(rejected.plan.fence_site_decisions.front().kind, ConSanSiteDecisionKind::Unsupported);
  EXPECT_EQ(rejected.plan.fence_site_decisions.front().reason,
            ConSanFencePolicyReason::MissingCommunicationEvent);
}

TEST(ConSanAtomicFencePolicy, EverySemanticQualificationFailureHasADistinctTypedReason) {
  using Mutation = std::function<void(ConSanSyncEvent &, ConSanSyncSequence &)>;
  const std::vector<std::tuple<std::string_view, Mutation, ConSanAtomicPolicyReason>> cases = {
      {"owner", [](auto &event, auto &) { event.execution_owners.clear(); },
       ConSanAtomicPolicyReason::MissingExecutionOwner},
      {"confidence",
       [](auto &, auto &sequence) { sequence.confidence = ConSanSemanticConfidence::Unsupported; },
       ConSanAtomicPolicyReason::UnqualifiedSyncSequence},
      {"role",
       [](auto &, auto &sequence) {
         sequence.memory_role = ConSanSyncMemoryRole::SequentiallyConsistent;
       },
       ConSanAtomicPolicyReason::UnsupportedMemoryRole},
      {"missing-scope", [](auto &, auto &sequence) { sequence.raw_scope.reset(); },
       ConSanAtomicPolicyReason::MissingScope},
      {"scope", [](auto &, auto &sequence) { sequence.raw_scope = 0; },
       ConSanAtomicPolicyReason::UnsupportedScope},
      {"width-zero", [](auto &, auto &sequence) { sequence.width_bits = 0; },
       ConSanAtomicPolicyReason::InvalidAccessWidth},
      {"width-bits", [](auto &, auto &sequence) { sequence.width_bits = 31; },
       ConSanAtomicPolicyReason::InvalidAccessWidth},
      {"outcome",
       [](auto &event, auto &sequence) {
         event.rmw_outcome = ConSanSyncRmwOutcome::Unknown;
         sequence.rmw_outcome = ConSanSyncRmwOutcome::Unknown;
       },
       ConSanAtomicPolicyReason::UnsupportedDynamicOutcome},
  };
  for (const auto &[name, mutate, expected] : cases) {
    SCOPED_TRACE(name);
    std::vector events{make_atomic_event()};
    std::vector sequences{make_atomic_sequence(events.front())};
    mutate(events.front(), sequences.front());
    const ConSanAtomicFencePolicyResult policy = plan_consan_atomic_fence_observation(
        build_atomic_inventory(std::move(events), std::move(sequences),
                               {make_global_atomic_site()}),
        atomic_request(ConSanCapabilityEngine::RecordReplay));
    ASSERT_TRUE(policy.valid());
    ASSERT_EQ(policy.plan.atomic_site_decisions.size(), 1u);
    EXPECT_EQ(policy.plan.atomic_site_decisions.front().kind,
              expected == ConSanAtomicPolicyReason::MissingExecutionOwner ||
                      expected == ConSanAtomicPolicyReason::UnqualifiedSyncSequence ||
                      expected == ConSanAtomicPolicyReason::UnsupportedScope
                  ? ConSanSiteDecisionKind::NotApplicable
                  : ConSanSiteDecisionKind::Unsupported);
    EXPECT_EQ(policy.plan.atomic_site_decisions.front().reason, expected);
    EXPECT_TRUE(policy.plan.probe_intents.empty());
  }
}

TEST(ConSanAtomicFencePolicy, AmbiguousSequenceMembershipFailsClosed) {
  std::vector events{make_atomic_event()};
  std::vector sequences{make_atomic_sequence(events.front(), "first"),
                        make_atomic_sequence(events.front(), "second")};
  const ConSanAtomicFencePolicyResult policy = plan_consan_atomic_fence_observation(
      build_atomic_inventory(std::move(events), std::move(sequences), {make_global_atomic_site()}),
      atomic_request(ConSanCapabilityEngine::RecordReplay));
  ASSERT_TRUE(policy.valid());
  EXPECT_EQ(policy.plan.atomic_site_decisions.front().kind, ConSanSiteDecisionKind::Unsupported);
  EXPECT_EQ(policy.plan.atomic_site_decisions.front().reason,
            ConSanAtomicPolicyReason::AmbiguousSequenceMembership);
}

TEST(ConSanAtomicFencePolicy, EncodingAndOperandFailuresRemainPolicyNotLoweringFacts) {
  using Mutation = std::function<void(ConSanAtomicSite &)>;
  const std::vector<std::tuple<std::string_view, Mutation, ConSanAtomicPolicyReason>> cases = {
      {"address-source", [](auto &site) { site.mnemonic = "buffer_atomic_add_u32"; },
       ConSanAtomicPolicyReason::UnsupportedAddressSource},
      {"width", [](auto &site) { site.width_bits = 64; },
       ConSanAtomicPolicyReason::InvalidAccessWidth},
      {"size", [](auto &site) { site.size = 4; }, ConSanAtomicPolicyReason::UnsupportedEncoding},
      {"offset", [](auto &site) { site.raw_ioffset.reset(); },
       ConSanAtomicPolicyReason::UnsupportedEncoding},
      {"address", [](auto &site) { site.addr_vgpr.reset(); },
       ConSanAtomicPolicyReason::MissingOperands},
      {"scope", [](auto &site) { site.raw_scope.reset(); }, ConSanAtomicPolicyReason::MissingScope},
  };
  for (const auto &[name, mutate, expected] : cases) {
    SCOPED_TRACE(name);
    std::vector events{make_atomic_event()};
    std::vector sequences{make_atomic_sequence(events.front())};
    ConSanAtomicSite site = make_global_atomic_site();
    mutate(site);
    const ConSanAtomicFencePolicyResult policy = plan_consan_atomic_fence_observation(
        build_atomic_inventory(std::move(events), std::move(sequences), {std::move(site)}),
        atomic_request(ConSanCapabilityEngine::InlineShadow));
    ASSERT_TRUE(policy.valid());
    EXPECT_EQ(policy.plan.atomic_site_decisions.front().kind, ConSanSiteDecisionKind::Unsupported);
    EXPECT_EQ(policy.plan.atomic_site_decisions.front().reason, expected);
    EXPECT_TRUE(policy.plan.probe_intents.empty());
  }
}

TEST(ConSanAtomicFencePolicy, PlanValidationEnforcesAtomicAndFenceTypeRelationships) {
  const ConSanAtomicFencePolicyResult policy = plan_consan_atomic_fence_observation(
      ordinary_fence_inventory(), atomic_request(ConSanCapabilityEngine::RecordReplay));
  ASSERT_TRUE(policy.valid());

  ConSanObservationPlan broken = policy.plan;
  broken.probe_intents.front().synchronization_association.reset();
  EXPECT_FALSE(broken.valid());
  broken = policy.plan;
  broken.probe_intents.front().synchronization_association = ConSanSynchronizationAssociationId{};
  EXPECT_FALSE(broken.valid());
  broken = policy.plan;
  broken.probe_intents.front().position = ConSanProbePosition::After;
  EXPECT_FALSE(broken.valid());
  broken = policy.plan;
  broken.probe_intents.front().dynamic_result = ConSanDynamicResultRequirement::ReturnedOldValue;
  EXPECT_FALSE(broken.valid());
  broken = policy.plan;
  broken.probe_intents.back().position = ConSanProbePosition::Before;
  EXPECT_FALSE(broken.valid());
  broken = policy.plan;
  broken.atomic_site_decisions.front().association.reset();
  EXPECT_FALSE(broken.valid());
  broken = policy.plan;
  broken.atomic_site_decisions.front().dynamic_result = ConSanDynamicResultRequirement::Count;
  EXPECT_FALSE(broken.valid());
  broken = policy.plan;
  broken.atomic_site_decisions.front().capability = static_cast<ConSanCapabilityDisposition>(255);
  EXPECT_FALSE(broken.valid());
  broken = policy.plan;
  broken.atomic_site_decisions.front().intent_ids = {{99}};
  EXPECT_FALSE(broken.valid());
  broken = policy.plan;
  broken.fence_site_decisions.front().inventory_association = ConSanFenceAssociation::Count;
  EXPECT_FALSE(broken.valid());
  broken = policy.plan;
  broken.fence_site_decisions.front().association.reset();
  EXPECT_FALSE(broken.valid());
  broken = policy.plan;
  broken.fence_site_decisions.front().intent_ids = {{99}};
  EXPECT_FALSE(broken.valid());
}

TEST(ConSanAtomicFencePolicy, AppendAndCoverageLedgerOwnAndRebaseBothDecisionFamilies) {
  const ConSanAtomicFencePolicyResult policy = plan_consan_atomic_fence_observation(
      ordinary_fence_inventory(), atomic_request(ConSanCapabilityEngine::RecordReplay));
  ASSERT_TRUE(policy.valid());
  ConSanObservationPlan combined = policy.plan;
  ASSERT_TRUE(combined.append(policy.plan));
  ASSERT_TRUE(combined.valid());
  ASSERT_EQ(combined.probe_intents.size(), 4u);
  ASSERT_EQ(combined.atomic_site_decisions.size(), 2u);
  ASSERT_EQ(combined.fence_site_decisions.size(), 2u);
  EXPECT_EQ(combined.atomic_site_decisions[1].intent_ids,
            (std::vector{ConSanProbeIntentId{2}, ConSanProbeIntentId{3}}));
  EXPECT_EQ(combined.fence_site_decisions[1].intent_ids,
            (std::vector{ConSanProbeIntentId{2}, ConSanProbeIntentId{3}}));

  ConSanCoverageLedger ledger(combined);
  EXPECT_TRUE(std::ranges::equal(ledger.atomic_site_decisions(), combined.atomic_site_decisions));
  EXPECT_TRUE(std::ranges::equal(ledger.fence_site_decisions(), combined.fence_site_decisions));
  ASSERT_EQ(ledger.intent_entries().size(), combined.probe_intents.size());
  EXPECT_TRUE(ledger.set_lowering_outcome({3}, ConSanLoweringOutcomeKind::Instrumented));
  EXPECT_EQ(ledger.intent_entry({3})->intent.kind, ConSanProbeIntentKind::FenceRecord);
}

TEST(ConSanAtomicFencePolicy, OrdinaryFenceAssociationDefinesEachEngineEvidenceContract) {
  const ProgramInventory inventory = ordinary_fence_inventory();
  for (ConSanCapabilityEngine engine :
       {ConSanCapabilityEngine::RecordReplay, ConSanCapabilityEngine::Sampled,
        ConSanCapabilityEngine::InlineShadow}) {
    SCOPED_TRACE(consan_capability_engine_name(engine));
    const ConSanAtomicFencePolicyResult policy =
        plan_consan_atomic_fence_observation(inventory, atomic_request(engine));
    ASSERT_TRUE(policy.valid());
    ASSERT_EQ(policy.plan.atomic_site_decisions.size(), 1u);
    ASSERT_EQ(policy.plan.fence_site_decisions.size(), 1u);
    const ConSanAtomicSiteDecision &atomic = policy.plan.atomic_site_decisions.front();
    const ConSanFenceSiteDecision &fence = policy.plan.fence_site_decisions.front();
    EXPECT_EQ(atomic.kind, ConSanSiteDecisionKind::Admitted);
    EXPECT_EQ(fence.kind, ConSanSiteDecisionKind::Admitted);
    EXPECT_EQ(atomic.association, fence.association);
    EXPECT_EQ(fence.inventory_association, ConSanFenceAssociation::Qualified);
    EXPECT_EQ(fence.capability, engine == ConSanCapabilityEngine::RecordReplay
                                    ? ConSanCapabilityDisposition::Supported
                                    : ConSanCapabilityDisposition::AssociatedOnly);
    EXPECT_EQ(atomic.intent_ids, fence.intent_ids);
    ASSERT_EQ(policy.plan.probe_intents.size(), 2u);
    EXPECT_EQ(policy.plan.probe_intents[0].kind, ConSanProbeIntentKind::AtomicAddressCapture);
    EXPECT_EQ(policy.plan.probe_intents[1].kind, engine == ConSanCapabilityEngine::RecordReplay
                                                     ? ConSanProbeIntentKind::FenceRecord
                                                 : engine == ConSanCapabilityEngine::Sampled
                                                     ? ConSanProbeIntentKind::SampledAtomicOrdering
                                                     : ConSanProbeIntentKind::ExactAtomicOrdering);
  }
}

TEST(ConSanAtomicFencePolicy, EveryFenceAssociationRejectionRemainsTypedInventoryEvidence) {
  for (ConSanFenceAssociation association : kConSanFenceAssociations) {
    SCOPED_TRACE(consan_fence_association_name(association));
    const ConSanAtomicFencePolicyResult policy =
        plan_consan_atomic_fence_observation(ordinary_fence_inventory(association),
                                             atomic_request(ConSanCapabilityEngine::RecordReplay));
    ASSERT_TRUE(policy.valid());
    ASSERT_EQ(policy.plan.fence_site_decisions.size(), 1u);
    const ConSanFenceSiteDecision &decision = policy.plan.fence_site_decisions.front();
    EXPECT_EQ(decision.inventory_association, association);
    if (association == ConSanFenceAssociation::Qualified) {
      EXPECT_EQ(decision.kind, ConSanSiteDecisionKind::Admitted);
      EXPECT_EQ(decision.reason, ConSanFencePolicyReason::None);
    } else {
      EXPECT_EQ(decision.kind, ConSanSiteDecisionKind::NotApplicable);
      EXPECT_EQ(decision.reason, ConSanFencePolicyReason::AssociationUnavailable);
    }
  }
}

TEST(ConSanAtomicFencePolicy, FenceRequestExclusionsAndMissingFactsRemainTyped) {
  const ProgramInventory inventory = ordinary_fence_inventory();

  ConSanAtomicFencePolicyRequest disabled = atomic_request(ConSanCapabilityEngine::RecordReplay);
  disabled.tracking_enabled = false;
  EXPECT_EQ(plan_consan_atomic_fence_observation(inventory, disabled)
                .plan.fence_site_decisions.front()
                .reason,
            ConSanFencePolicyReason::TrackingDisabled);

  EXPECT_EQ(plan_consan_atomic_fence_observation(
                inventory, atomic_request(ConSanCapabilityEngine::SuperCollider))
                .plan.fence_site_decisions.front()
                .reason,
            ConSanFencePolicyReason::EngineMutationOnly);

  ConSanAtomicFencePolicyRequest filtered = atomic_request(ConSanCapabilityEngine::RecordReplay);
  filtered.container_filter = "different";
  EXPECT_EQ(plan_consan_atomic_fence_observation(inventory, filtered)
                .plan.fence_site_decisions.front()
                .reason,
            ConSanFencePolicyReason::ContainerFilterExcluded);

  std::vector owner_events{make_ordinary_store_event(), make_fence_event()};
  owner_events.back().execution_owners.clear();
  std::vector owner_sequences{make_atomic_sequence(owner_events.front())};
  std::vector owner_fences{
      make_fence_candidate(owner_events[0], owner_events[1], owner_sequences[0])};
  const ConSanAtomicFencePolicyResult missing_owner = plan_consan_atomic_fence_observation(
      build_atomic_inventory(std::move(owner_events), std::move(owner_sequences), {},
                             {make_global_store_site({})}, std::move(owner_fences)),
      atomic_request(ConSanCapabilityEngine::RecordReplay));
  ASSERT_TRUE(missing_owner.valid());
  EXPECT_EQ(missing_owner.plan.fence_site_decisions.front().kind,
            ConSanSiteDecisionKind::NotApplicable);
  EXPECT_EQ(missing_owner.plan.fence_site_decisions.front().reason,
            ConSanFencePolicyReason::MissingExecutionOwner);

  std::vector missing_events{make_ordinary_store_event(), make_fence_event()};
  std::vector missing_sequences{make_atomic_sequence(missing_events.front())};
  ConSanMoiFenceCandidate missing_fence =
      make_fence_candidate(missing_events[0], missing_events[1], missing_sequences[0]);
  missing_fence.communication_event->physical.original_text_offset = 999u;
  const ConSanAtomicFencePolicyResult missing_communication = plan_consan_atomic_fence_observation(
      build_atomic_inventory(std::move(missing_events), std::move(missing_sequences), {},
                             {make_global_store_site({})}, {std::move(missing_fence)}),
      atomic_request(ConSanCapabilityEngine::RecordReplay));
  ASSERT_TRUE(missing_communication.valid());
  EXPECT_EQ(missing_communication.plan.fence_site_decisions.front().kind,
            ConSanSiteDecisionKind::Unsupported);
  EXPECT_EQ(missing_communication.plan.fence_site_decisions.front().reason,
            ConSanFencePolicyReason::MissingCommunicationEvent);
}

TEST(ConSanAtomicFencePolicy, ConflictingFenceAliasesProduceTypedFatalError) {
  std::vector events{make_ordinary_store_event(), make_fence_event()};
  std::vector sequences{make_atomic_sequence(events.front())};
  events[1].source_containers = {"atomic_kernel", "aliased_kernel"};
  ConSanMoiFenceCandidate first = make_fence_candidate(events[0], events[1], sequences[0]);
  ConSanMoiFenceCandidate conflict = first;
  conflict.memory_role = ConSanSyncMemoryRole::Acquire;
  const ConSanAtomicFencePolicyResult policy = plan_consan_atomic_fence_observation(
      build_atomic_inventory(std::move(events), std::move(sequences), {},
                             {make_global_store_site({})}, {std::move(first), std::move(conflict)}),
      atomic_request(ConSanCapabilityEngine::RecordReplay));
  EXPECT_FALSE(policy.valid());
  EXPECT_EQ(policy.fence_errors,
            (std::vector{ConSanFencePolicyReason::ConflictingPhysicalAliases}));
  ASSERT_EQ(policy.plan.fence_site_decisions.size(), 1u);
  EXPECT_EQ(policy.plan.fence_site_decisions.front().kind, ConSanSiteDecisionKind::Unsupported);
  EXPECT_EQ(policy.plan.fence_site_decisions.front().reason,
            ConSanFencePolicyReason::ConflictingPhysicalAliases);
  EXPECT_EQ(policy.plan.fence_site_decisions.front().source_containers,
            (std::vector<std::string>{"aliased_kernel", "atomic_kernel"}));
}

TEST(ConSanAtomicFencePolicy, ConflictingPhysicalAliasesProduceTypedFatalError) {
  std::vector events{make_atomic_event()};
  ConSanSyncEvent alias = events.front();
  alias.container_name = "aliased_kernel";
  alias.width_bits = 64;
  events.push_back(std::move(alias));
  std::vector sequences{make_atomic_sequence(events.front())};
  const ConSanAtomicFencePolicyResult policy = plan_consan_atomic_fence_observation(
      build_atomic_inventory(std::move(events), std::move(sequences), {make_global_atomic_site()}),
      atomic_request(ConSanCapabilityEngine::RecordReplay));
  EXPECT_FALSE(policy.valid());
  EXPECT_EQ(policy.atomic_errors,
            (std::vector{ConSanAtomicPolicyReason::ConflictingPhysicalAliases}));
  ASSERT_EQ(policy.plan.atomic_site_decisions.size(), 1u);
  EXPECT_EQ(policy.plan.atomic_site_decisions.front().kind, ConSanSiteDecisionKind::Unsupported);
  EXPECT_EQ(policy.plan.atomic_site_decisions.front().source_containers,
            (std::vector<std::string>{"aliased_kernel", "atomic_kernel"}));
  EXPECT_TRUE(policy.plan.probe_intents.empty());
}

TEST(ConSanAtomicFencePolicy, PolicyIsDeterministicAndDoesNotMutatePublishedInventory) {
  const ProgramInventory inventory = ordinary_fence_inventory();
  const SynchronizationInventoryView before = inventory.sync();
  std::vector<std::string> event_identities_before;
  for (const ConSanSyncEvent &event : before.sync_events)
    event_identities_before.push_back(event.identity);
  std::vector<std::string> sequence_identities_before;
  for (const ConSanSyncSequence &sequence : before.sync_sequences)
    sequence_identities_before.push_back(sequence.identity);
  const ConSanAtomicFencePolicyRequest request =
      atomic_request(ConSanCapabilityEngine::InlineShadow);
  const ConSanAtomicFencePolicyResult first =
      plan_consan_atomic_fence_observation(inventory, request);
  const ConSanAtomicFencePolicyResult second =
      plan_consan_atomic_fence_observation(inventory, request);
  EXPECT_EQ(first, second);
  EXPECT_TRUE(first.valid());
  const SynchronizationInventoryView after = inventory.sync();
  std::vector<std::string> event_identities_after;
  for (const ConSanSyncEvent &event : after.sync_events)
    event_identities_after.push_back(event.identity);
  std::vector<std::string> sequence_identities_after;
  for (const ConSanSyncSequence &sequence : after.sync_sequences)
    sequence_identities_after.push_back(sequence.identity);
  EXPECT_EQ(event_identities_after, event_identities_before);
  EXPECT_EQ(sequence_identities_after, sequence_identities_before);
}

} // namespace
} // namespace rocjitsu
