// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"

#include <concepts>
#include <set>
#include <tuple>
#include <type_traits>

namespace rocjitsu {
namespace {

template <typename Enum, size_t N, typename NameFunction>
void expect_complete_enum_contract(const std::array<Enum, N> &values, Enum count, NameFunction name,
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

ConSanAccessInventorySite make_inventory_lds_site(std::string mnemonic, uint64_t text_offset = 16,
                                                  uint64_t file_offset = 0,
                                                  uint32_t width_bits = 32) {
  ConSanAccessInventorySite site;
  site.origin = ConSanAccessOrigin::NativeLds;
  site.kind = ConSanLdsAccessKind::Write;
  site.supported_mvp = true;
  site.physical_id.original_text_offset = text_offset;
  site.file_offset = file_offset;
  site.instruction_size = 8;
  site.decoded_width_bits = width_bits;
  site.operands.address_vgpr = 3;
  site.operands.data_vgpr = 7;
  site.mnemonic = std::move(mnemonic);
  return site;
}

ConSanAccessInventorySite make_inventory_flat_site(ConSanFlatAddressSpaceHint hint,
                                                   uint64_t text_offset = 16) {
  ConSanAccessInventorySite site;
  site.origin = ConSanAccessOrigin::Flat;
  site.kind = ConSanLdsAccessKind::Read;
  site.physical_id.original_text_offset = text_offset;
  site.instruction_size = 8;
  site.decoded_width_bits = 32;
  site.operands.destination_vgpr = 4;
  site.operands.address_vgpr = 6;
  site.operands.raw_ioffset = -8;
  site.flat_address_space_hint = hint;
  site.mnemonic = "flat_load_b32";
  return site;
}

ConSanKernelInfo make_inventory_kernel(std::string name = "inventory_kernel") {
  ConSanKernelInfo kernel;
  kernel.name = std::move(name);
  kernel.descriptor_file_offset = 512;
  kernel.declared_group_segment_bytes = 4096;
  kernel.entry_text_offset = 32;
  kernel.text_file_offset = 1024;
  kernel.code_size_inferred_from_zero = true;
  kernel.uses_gfx1250_cluster_workgroup_id = true;
  return kernel;
}

std::vector<ConSanInventoryExclusionReason>
exclusion_reasons(const ConSanAccessInventorySite &site) {
  std::vector<ConSanInventoryExclusionReason> reasons;
  for (const ConSanInventoryExclusion &exclusion : site.exclusions)
    reasons.push_back(exclusion.reason);
  return reasons;
}

TEST(ConSanProgramInventory, EnumContractsAreExhaustiveNamedAndRejectInvalidValues) {
  expect_complete_enum_contract(kConSanSemanticSiteDomains, ConSanSemanticSiteDomain::Count,
                                consan_semantic_site_domain_name, "invalid-semantic-site-domain");
  expect_complete_enum_contract(kConSanAccessOrigins, ConSanAccessOrigin::Count,
                                consan_access_origin_name, "invalid-access-origin");
  expect_complete_enum_contract(kConSanAccessAddressSpaces, ConSanAccessAddressSpace::Count,
                                consan_access_address_space_name, "invalid-access-address-space");
  expect_complete_enum_contract(kConSanAccessProvenances, ConSanAccessProvenance::Count,
                                consan_access_provenance_name, "invalid-access-provenance");
  expect_complete_enum_contract(
      kConSanInventoryExclusionReasons, ConSanInventoryExclusionReason::Count,
      consan_inventory_exclusion_reason_name, "invalid-inventory-exclusion-reason");
  expect_complete_enum_contract(kConSanFenceAssociations, ConSanFenceAssociation::Count,
                                consan_fence_association_name, "invalid-fence-association");
}

TEST(ConSanProgramInventory, FenceCandidateEligibilityExactlyMatchesQualifiedAssociation) {
  ConSanMoiFenceCandidate candidate;
  for (ConSanFenceAssociation association : kConSanFenceAssociations) {
    SCOPED_TRACE(consan_fence_association_name(association));
    candidate.association = association;
    EXPECT_EQ(candidate.eligible(), association == ConSanFenceAssociation::Qualified);
  }
}

TEST(ConSanProgramInventory, CodeObjectIdentityIsStableCompatibleAndCollisionAware) {
  const std::array<uint8_t, 5> bytes = {1, 2, 3, 4, 5};
  const ConSanCodeObjectId first = make_consan_code_object_id(bytes);
  const ConSanCodeObjectId again = make_consan_code_object_id(bytes);
  EXPECT_TRUE(first.valid());
  EXPECT_EQ(first, again);
  EXPECT_EQ(first.fingerprint, "fnv1a64:0f66dcbf4f6b7d88");
  EXPECT_EQ(first.byte_size, bytes.size());

  const std::array<uint8_t, 5> reversed = {5, 4, 3, 2, 1};
  EXPECT_NE(first, make_consan_code_object_id(reversed));
  ConSanCodeObjectId simulated_primary_collision = first;
  ++simulated_primary_collision.collision_verifier;
  EXPECT_NE(first, simulated_primary_collision);
  ConSanCodeObjectId simulated_size_collision = first;
  ++simulated_size_collision.byte_size;
  EXPECT_NE(first, simulated_size_collision);
  EXPECT_FALSE(ConSanCodeObjectId{}.valid());
  EXPECT_TRUE(make_consan_code_object_id(std::span<const uint8_t>{}).valid());
}

TEST(ConSanProgramInventory, PhysicalAndSemanticIdentitiesHaveExplicitValidityAndOrdinals) {
  PhysicalSiteId physical;
  EXPECT_FALSE(physical.valid());
  physical.code_object = make_consan_code_object_id(std::array<uint8_t, 1>{42});
  physical.original_text_offset = 24;
  EXPECT_TRUE(physical.valid());

  SemanticSiteId semantic;
  semantic.physical = physical;
  EXPECT_FALSE(semantic.valid());
  semantic.domain = ConSanSemanticSiteDomain::Access;
  EXPECT_TRUE(semantic.valid());

  SemanticSiteId other_range = semantic;
  other_range.range_ordinal = 1;
  EXPECT_NE(semantic, other_range);
  SemanticSiteId other_member = semantic;
  other_member.member_ordinal = 1;
  EXPECT_NE(semantic, other_member);
  SemanticSiteId other_domain = semantic;
  other_domain.domain = ConSanSemanticSiteDomain::SynchronizationEvent;
  EXPECT_NE(semantic, other_domain);
}

TEST(ConSanProgramInventory, ValueRecordsPreserveTypedFactsAndCompleteness) {
  ConSanProgramContainerRef container;
  container.kind = ConSanProgramContainerKind::Kernel;
  container.name = "kernel";
  container.entry_text_offset = 64;
  container.kernel_descriptor_file_offset = 128;
  container.uses_gfx1250_cluster_workgroup_id = true;
  EXPECT_TRUE(container.is_kernel());
  EXPECT_EQ(container, container);
  ConSanProgramContainerRef function = container;
  function.kind = ConSanProgramContainerKind::Function;
  function.kernel_descriptor_file_offset.reset();
  EXPECT_FALSE(function.is_kernel());
  EXPECT_NE(container, function);

  ConSanAccessOperandFacts operands;
  operands.destination_vgpr = 1;
  operands.destination_accvgpr = 2;
  operands.address_vgpr = 3;
  operands.data_vgpr = 4;
  operands.second_data_vgpr = 5;
  operands.raw_op = 6;
  operands.raw_saddr = 7;
  operands.raw_scale_offset = true;
  operands.raw_vaddr = 8;
  operands.raw_vsrc = 9;
  operands.raw_vdst = 10;
  operands.raw_ioffset = -11;
  operands.raw_segment = 12;
  operands.raw_scope = 13;
  operands.raw_th = 14;
  EXPECT_EQ(operands, operands);
  ConSanAccessOperandFacts changed_operands = operands;
  changed_operands.raw_th = 15;
  EXPECT_NE(operands, changed_operands);

  ConSanAccessInventorySite site;
  EXPECT_FALSE(site.complete());
  ConSanAccessRange range;
  range.byte_width = 4;
  site.ranges.push_back(range);
  EXPECT_TRUE(site.complete());
  ConSanInventoryExclusion exclusion;
  exclusion.reason = ConSanInventoryExclusionReason::InvalidAccessWidth;
  exclusion.detail = "test";
  site.exclusions.push_back(exclusion);
  EXPECT_FALSE(site.complete());
  EXPECT_EQ(site.exclusions.front(), exclusion);
  EXPECT_EQ(site.ranges.front(), range);
}

TEST(ConSanProgramInventory, ImmutableViewsRetainFactsAcrossCopyMoveAndBuilderLifetime) {
  static_assert(std::same_as<decltype(std::declval<const ProgramInventory &>().access_sites()),
                             std::span<const ConSanAccessInventorySite>>);
  static_assert(std::same_as<decltype(std::declval<const ProgramInventory &>().kernels()),
                             std::span<const ConSanKernelInfo>>);
  static_assert(std::is_const_v<std::remove_reference_t<
                    decltype(std::declval<const ProgramInventory &>().kernels().front())>>);

  ProgramInventory empty;
  EXPECT_TRUE(empty.empty());
  EXPECT_FALSE(empty.code_object_id().valid());
  EXPECT_FALSE(empty.code_object_parsed());
  EXPECT_FALSE(empty.kernel_metadata_trustworthy());
  EXPECT_EQ(empty.malformed_kernel_metadata_note_count(), 0u);
  EXPECT_EQ(empty.arch(), ROCJITSU_CODE_ARCH_INVALID);
  EXPECT_EQ(empty.target(), ROCJITSU_CODE_TARGET_INVALID);
  EXPECT_FALSE(empty.semantic_arch_required());
  EXPECT_TRUE(empty.access_sites().empty());
  EXPECT_TRUE(empty.text_sections().empty());
  EXPECT_TRUE(empty.kernels().empty());
  EXPECT_TRUE(empty.functions().empty());
  ProgramInventoryBuilder empty_content_builder;
  EXPECT_FALSE(empty_content_builder.view().empty());
  EXPECT_FALSE(empty_content_builder.view().code_object_parsed());
  EXPECT_TRUE(empty_content_builder.view().text_sections().empty());
  EXPECT_TRUE(empty_content_builder.view().kernels().empty());
  EXPECT_TRUE(empty_content_builder.view().functions().empty());

  const auto make_result = [] {
    const std::array<uint8_t, 4> bytes = {7, 8, 9, 10};
    ProgramInventoryBuilder builder(bytes);
    builder.set_code_object_facts(true, 3, ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_TARGET_GFX1201);
    builder.set_semantic_arch_required(true);
    ConSanTextSection section;
    section.name = ".text";
    builder.text_sections().push_back(section);
    ConSanKernelInfo kernel = make_inventory_kernel();
    kernel.access_sites.push_back(make_inventory_lds_site("ds_store_b32"));
    builder.kernels().push_back(std::move(kernel));
    ConSanFunctionInfo function;
    function.name = "helper";
    builder.functions().push_back(std::move(function));
    builder.rebuild_access_inventory(bytes);

    ConSanResult result;
    result.program_inventory = builder.view();
    return result;
  };

  ConSanResult result = make_result();
  ConSanResult copied = result;
  ConSanResult moved = std::move(copied);
  EXPECT_FALSE(result.program_inventory.empty());
  EXPECT_TRUE(moved.program_inventory.code_object_parsed());
  EXPECT_TRUE(moved.program_inventory.kernel_metadata_trustworthy());
  EXPECT_EQ(moved.program_inventory.malformed_kernel_metadata_note_count(), 3u);
  EXPECT_EQ(moved.program_inventory.arch(), ROCJITSU_CODE_ARCH_RDNA4);
  EXPECT_EQ(moved.program_inventory.target(), ROCJITSU_CODE_TARGET_GFX1201);
  EXPECT_TRUE(moved.program_inventory.semantic_arch_required());
  ASSERT_EQ(moved.program_inventory.text_sections().size(), 1u);
  ASSERT_EQ(moved.program_inventory.kernels().size(), 1u);
  ASSERT_EQ(moved.program_inventory.functions().size(), 1u);
  ASSERT_EQ(moved.program_inventory.access_sites().size(), 1u);
  EXPECT_EQ(moved.program_inventory.kernels().front().name, "inventory_kernel");
  EXPECT_EQ(moved.program_inventory.kernels().front().declared_group_segment_bytes, 4096u);
  EXPECT_EQ(moved.program_inventory.access_sites().front().container.name, "inventory_kernel");
}

TEST(ConSanProgramInventory, SynchronizationViewIsConstCompleteAndLifetimeSafe) {
  static_assert(
      std::same_as<typename decltype(SynchronizationInventoryView{}.sync_events)::element_type,
                   const ConSanSyncEvent>);
  static_assert(
      std::same_as<typename decltype(SynchronizationInventoryView{}.sync_sequences)::element_type,
                   const ConSanSyncSequence>);
  static_assert(std::same_as<typename decltype(SynchronizationInventoryView{}
                                                   .barrier_lifecycle_groups)::element_type,
                             const ConSanBarrierLifecycleGroup>);
  static_assert(std::same_as<typename decltype(SynchronizationInventoryView{}
                                                   .communication_address_recipes)::element_type,
                             const ConSanCommunicationAddressRecipe>);
  static_assert(std::same_as<typename decltype(SynchronizationInventoryView{}
                                                   .moi_fence_candidates)::element_type,
                             const ConSanMoiFenceCandidate>);
  static_assert(std::same_as<decltype(std::declval<const ProgramInventory &>().sync().sync_events),
                             std::span<const ConSanSyncEvent>>);
  static_assert(
      std::same_as<decltype(std::declval<const ProgramInventory &>().sync().sync_sequences),
                   std::span<const ConSanSyncSequence>>);
  static_assert(std::same_as<
                decltype(std::declval<const ProgramInventory &>().sync().barrier_lifecycle_groups),
                std::span<const ConSanBarrierLifecycleGroup>>);
  static_assert(
      std::same_as<
          decltype(std::declval<const ProgramInventory &>().sync().communication_address_recipes),
          std::span<const ConSanCommunicationAddressRecipe>>);
  static_assert(
      std::same_as<decltype(std::declval<const ProgramInventory &>().sync().moi_fence_candidates),
                   std::span<const ConSanMoiFenceCandidate>>);
  static_assert(!std::is_base_of_v<SynchronizationInventoryView, ConSanResult>);

  ProgramInventory empty;
  EXPECT_TRUE(empty.sync().empty());
  EXPECT_TRUE(empty.sync().sync_events.empty());
  EXPECT_TRUE(empty.sync().sync_sequences.empty());
  EXPECT_TRUE(empty.sync().barrier_lifecycle_groups.empty());
  EXPECT_TRUE(empty.sync().communication_address_recipes.empty());
  EXPECT_TRUE(empty.sync().moi_fence_candidates.empty());

  const std::array<uint8_t, 4> bytes = {4, 3, 2, 1};
  ProgramInventoryBuilder builder(bytes);
  SynchronizationInventoryBuildView build = builder.synchronization();
  ConSanSyncEvent event;
  event.identity = "event";
  event.semantic_id = {
      .physical = {.code_object = make_consan_code_object_id(bytes), .original_text_offset = 16},
      .domain = ConSanSemanticSiteDomain::SynchronizationEvent,
  };
  build.sync_events.push_back(event);
  ConSanSyncSequence sequence;
  sequence.identity = "sequence";
  SemanticSiteId member = event.semantic_id;
  member.domain = ConSanSemanticSiteDomain::SynchronizationSequenceMember;
  sequence.member_semantic_ids.push_back(member);
  sequence.member_event_identities.push_back(event.identity);
  build.sync_sequences.push_back(sequence);
  ConSanBarrierLifecycleGroup lifecycle;
  lifecycle.identity = "lifecycle";
  lifecycle.member_semantic_ids = {member};
  build.barrier_lifecycle_groups.push_back(lifecycle);
  ConSanCommunicationAddressRecipe recipe;
  recipe.support = ConSanCommunicationAddressSupport::Supported;
  recipe.sequence_identity = sequence.identity;
  build.communication_address_recipes.push_back(recipe);
  ConSanMoiFenceCandidate fence;
  fence.identity = "fence";
  fence.sequence_identity = sequence.identity;
  fence.association = ConSanFenceAssociation::Qualified;
  build.moi_fence_candidates.push_back(fence);

  ProgramInventory published = builder.view();
  const SynchronizationInventoryView view = published.sync();
  EXPECT_FALSE(view.empty());
  EXPECT_EQ(view.sync_events.data(), published.sync().sync_events.data());
  EXPECT_EQ(view.sync_sequences.data(), published.sync().sync_sequences.data());
  EXPECT_EQ(view.barrier_lifecycle_groups.data(), published.sync().barrier_lifecycle_groups.data());
  EXPECT_EQ(view.communication_address_recipes.data(),
            published.sync().communication_address_recipes.data());
  EXPECT_EQ(view.moi_fence_candidates.data(), published.sync().moi_fence_candidates.data());
  ASSERT_EQ(view.sync_events.size(), 1u);
  ASSERT_EQ(view.sync_sequences.size(), 1u);
  ASSERT_EQ(view.barrier_lifecycle_groups.size(), 1u);
  ASSERT_EQ(view.communication_address_recipes.size(), 1u);
  ASSERT_EQ(view.moi_fence_candidates.size(), 1u);
  EXPECT_TRUE(view.sync_events.front().semantic_id.valid());
  EXPECT_TRUE(view.sync_sequences.front().member_semantic_ids.front().valid());
  EXPECT_TRUE(view.communication_address_recipes.front().supported());
  EXPECT_TRUE(view.moi_fence_candidates.front().eligible());

  ConSanResult result;
  result.program_inventory = published;
  ConSanResult copied = result;
  ConSanResult moved = std::move(copied);
  EXPECT_EQ(moved.program_inventory.sync().sync_events.front().identity, "event");
  EXPECT_EQ(moved.program_inventory.sync().sync_sequences.front().identity, "sequence");
  EXPECT_EQ(moved.program_inventory.sync().barrier_lifecycle_groups.front().identity, "lifecycle");
  EXPECT_EQ(moved.program_inventory.sync().communication_address_recipes.front().sequence_identity,
            "sequence");
  EXPECT_EQ(moved.program_inventory.sync().moi_fence_candidates.front().identity, "fence");
}

TEST(ConSanProgramInventory, MutableRevisionIsDeepCopiedFromPublishedInventory) {
  ProgramInventoryBuilder original(std::array<uint8_t, 1>{7});
  original.kernels().push_back(make_inventory_kernel("original"));
  ConSanSyncEvent event;
  event.identity = "original-event";
  original.synchronization().sync_events.push_back(event);
  ConSanSyncSequence sequence;
  sequence.identity = "original-sequence";
  original.synchronization().sync_sequences.push_back(sequence);
  ConSanBarrierLifecycleGroup group;
  group.identity = "original-group";
  original.synchronization().barrier_lifecycle_groups.push_back(group);
  ConSanCommunicationAddressRecipe recipe;
  recipe.sequence_identity = "original-sequence";
  original.synchronization().communication_address_recipes.push_back(recipe);
  ConSanMoiFenceCandidate fence;
  fence.identity = "original-fence";
  original.synchronization().moi_fence_candidates.push_back(fence);
  const ProgramInventory published = original.view();

  ProgramInventoryBuilder revision(published);
  revision.kernels().front().name = "revision";
  SynchronizationInventoryBuildView revised = revision.synchronization();
  revised.sync_events.front().identity = "revision-event";
  revised.sync_sequences.clear();
  revised.barrier_lifecycle_groups.clear();
  revised.communication_address_recipes.clear();
  revised.moi_fence_candidates.front().association = ConSanFenceAssociation::Qualified;

  const SynchronizationInventoryView unchanged = published.sync();
  EXPECT_EQ(published.kernels().front().name, "original");
  EXPECT_EQ(unchanged.sync_events.front().identity, "original-event");
  EXPECT_EQ(unchanged.sync_sequences.size(), 1u);
  EXPECT_EQ(unchanged.barrier_lifecycle_groups.size(), 1u);
  EXPECT_EQ(unchanged.communication_address_recipes.size(), 1u);
  EXPECT_FALSE(unchanged.moi_fence_candidates.front().eligible());

  const ProgramInventory revised_inventory = revision.view();
  EXPECT_EQ(revised_inventory.kernels().front().name, "revision");
  EXPECT_EQ(revised_inventory.sync().sync_events.front().identity, "revision-event");
  EXPECT_TRUE(revised_inventory.sync().sync_sequences.empty());
  EXPECT_TRUE(revised_inventory.sync().barrier_lifecycle_groups.empty());
  EXPECT_TRUE(revised_inventory.sync().communication_address_recipes.empty());
  EXPECT_TRUE(revised_inventory.sync().moi_fence_candidates.front().eligible());
}

TEST(ConSanProgramInventory, RealSynchronizationInventoryUsesTypedStableMemberIdentities) {
  const std::array<uint32_t, 6> text_words = {
      0xBE805181u, // s_barrier_init 1
      0xBE805281u, // s_barrier_join 1
      0xBE804E81u, // s_barrier_signal 1
      0xBF940001u, // s_barrier_wait 1
      0xBF950000u, // s_barrier_leave
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_gfx1250_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.fault_dry_run = true;
  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_FALSE(result.program_inventory.sync().sync_events.empty());
  for (const ConSanSyncEvent &event : result.program_inventory.sync().sync_events) {
    EXPECT_TRUE(event.semantic_id.valid());
    EXPECT_EQ(event.semantic_id.domain, ConSanSemanticSiteDomain::SynchronizationEvent);
    EXPECT_EQ(event.semantic_id.physical.code_object, result.program_inventory.code_object_id());
    EXPECT_EQ(event.semantic_id.physical.original_text_offset, event.text_offset);
  }
  for (const ConSanSyncSequence &sequence : result.program_inventory.sync().sync_sequences) {
    ASSERT_EQ(sequence.member_semantic_ids.size(), sequence.member_event_identities.size());
    for (size_t index = 0; index < sequence.member_semantic_ids.size(); ++index) {
      const auto event =
          std::ranges::find(result.program_inventory.sync().sync_events,
                            sequence.member_event_identities[index], &ConSanSyncEvent::identity);
      ASSERT_NE(event, result.program_inventory.sync().sync_events.end());
      SemanticSiteId expected = event->semantic_id;
      expected.domain = ConSanSemanticSiteDomain::SynchronizationSequenceMember;
      EXPECT_EQ(sequence.member_semantic_ids[index], expected);
    }
  }
  ASSERT_EQ(result.program_inventory.sync().barrier_lifecycle_groups.size(), 1u);
  const ConSanBarrierLifecycleGroup &group =
      result.program_inventory.sync().barrier_lifecycle_groups.front();
  EXPECT_TRUE(group.admissible);
  EXPECT_EQ(group.member_semantic_ids.size(), group.member_event_identities.size());
  EXPECT_EQ(group.member_semantic_ids.size(), 5u);
  EXPECT_TRUE(std::ranges::all_of(group.member_semantic_ids, [](const SemanticSiteId &member_id) {
    return member_id.valid() &&
           member_id.domain == ConSanSemanticSiteDomain::SynchronizationSequenceMember;
  }));
}

TEST(ConSanProgramInventory, NativeLdsFactsAndSubwordRangesAreNormalizedWithoutPolicy) {
  const std::array<uint8_t, 8> bytes = {};
  ProgramInventoryBuilder builder(bytes);
  ConSanKernelInfo kernel = make_inventory_kernel();
  ConSanAccessInventorySite byte_site = make_inventory_lds_site("ds_store_b8", 16, 0, 8);
  byte_site.operands.destination_vgpr = 2;
  byte_site.operands.destination_accvgpr = 4;
  byte_site.operands.second_data_vgpr = 8;
  kernel.access_sites.push_back(byte_site);
  kernel.access_sites.push_back(make_inventory_lds_site("ds_store_b16", 24, 0, 16));
  ConSanAccessInventorySite direct_to_lds =
      make_inventory_lds_site("global_load_lds_b32", 32, 0, 32);
  direct_to_lds.origin = ConSanAccessOrigin::DirectToLds;
  direct_to_lds.operands.address_vgpr.reset();
  kernel.access_sites.push_back(std::move(direct_to_lds));
  builder.kernels().push_back(std::move(kernel));
  builder.rebuild_access_inventory(bytes);
  builder.rebuild_access_inventory(bytes);

  const auto sites = builder.view().access_sites();
  ASSERT_EQ(sites.size(), 3u);
  const ConSanAccessInventorySite &byte = sites[0];
  EXPECT_EQ(byte.origin, ConSanAccessOrigin::NativeLds);
  EXPECT_EQ(byte.address_space, ConSanAccessAddressSpace::Group);
  EXPECT_EQ(byte.provenance, ConSanAccessProvenance::NativeLdsOpcode);
  EXPECT_EQ(byte.confidence, ConSanSemanticConfidence::Exact);
  EXPECT_TRUE(byte.supported_mvp);
  EXPECT_EQ(byte.container.kind, ConSanProgramContainerKind::Kernel);
  EXPECT_EQ(byte.container.kernel_descriptor_file_offset, 512u);
  EXPECT_EQ(byte.container.text_file_offset, 1024u);
  EXPECT_TRUE(byte.container.code_size_inferred_from_zero);
  EXPECT_TRUE(byte.container.uses_gfx1250_cluster_workgroup_id);
  EXPECT_EQ(byte.operands.destination_vgpr, 2u);
  EXPECT_EQ(byte.operands.destination_accvgpr, 4u);
  EXPECT_EQ(byte.operands.address_vgpr, 3u);
  EXPECT_EQ(byte.operands.data_vgpr, 7u);
  EXPECT_EQ(byte.operands.second_data_vgpr, 8u);
  EXPECT_TRUE(byte.execution_owner_descriptor_file_offsets.empty());
  ASSERT_EQ(byte.ranges.size(), 1u);
  EXPECT_EQ(byte.ranges[0].byte_width, 1u);
  EXPECT_EQ(byte.ranges[0].static_byte_offset, 0);
  EXPECT_EQ(byte.ranges[0].id.physical, byte.physical_id);
  EXPECT_EQ(byte.ranges[0].id.domain, ConSanSemanticSiteDomain::Access);
  EXPECT_EQ(byte.ranges[0].id.range_ordinal, 0u);
  EXPECT_TRUE(byte.complete());

  ASSERT_EQ(sites[1].ranges.size(), 1u);
  EXPECT_EQ(sites[1].ranges[0].byte_width, 2u);
  EXPECT_EQ(sites[2].origin, ConSanAccessOrigin::DirectToLds);
  EXPECT_FALSE(sites[2].operands.address_vgpr);
  ASSERT_EQ(sites[2].ranges.size(), 1u);
  EXPECT_EQ(sites[2].ranges[0].static_byte_offset, 0);
  EXPECT_TRUE(sites[2].complete());
}

TEST(ConSanProgramInventory, StagedRangeReattributesAndMergesNormalizedAccesses) {
  const std::array<uint8_t, 128> bytes = {};
  ProgramInventoryBuilder builder(bytes);
  ConSanKernelInfo owner = make_inventory_kernel("owner");
  owner.descriptor_file_offset = 512;
  builder.kernels().push_back(std::move(owner));
  ConSanKernelInfo overlapping = make_inventory_kernel("overlapping");
  overlapping.descriptor_file_offset = 768;
  overlapping.access_sites.push_back(make_inventory_lds_site("ds_store_b32", 80));
  overlapping.access_sites.push_back(
      make_inventory_flat_site(ConSanFlatAddressSpaceHint::Group, 88));
  overlapping.access_sites.push_back(make_inventory_lds_site("ds_store_b32", 120));
  builder.kernels().push_back(std::move(overlapping));
  builder.rebuild_access_inventory(bytes);

  ConSanKernelInfo decoded_range = make_inventory_kernel("temporary_range");
  decoded_range.access_sites.push_back(make_inventory_lds_site("ds_store_b32", 80));
  decoded_range.access_sites.push_back(make_inventory_lds_site("ds_store_b32", 96));
  decoded_range.access_sites.push_back(
      make_inventory_flat_site(ConSanFlatAddressSpaceHint::Group, 88));
  decoded_range.access_sites.push_back(
      make_inventory_flat_site(ConSanFlatAddressSpaceHint::Group, 104));
  builder.reattribute_access_range(80, 32, builder.kernels().front(), decoded_range, bytes);

  const auto accesses = builder.view().access_sites();
  ASSERT_EQ(accesses.size(), 5u);
  for (const uint64_t offset : {80u, 88u, 96u, 104u}) {
    const auto access = std::ranges::find(accesses, offset, [](const auto &candidate) {
      return candidate.physical_id.original_text_offset;
    });
    ASSERT_NE(access, accesses.end());
    EXPECT_EQ(access->container.name, "owner");
    EXPECT_EQ(access->container.kernel_descriptor_file_offset, 512u);
  }
  const auto outside = std::ranges::find(accesses, 120u, [](const auto &candidate) {
    return candidate.physical_id.original_text_offset;
  });
  ASSERT_NE(outside, accesses.end());
  EXPECT_EQ(outside->container.name, "overlapping");
  EXPECT_EQ(outside->container.kernel_descriptor_file_offset, 768u);
}

TEST(ConSanProgramInventory, SingleRangeNativeOffsetsPreserveArchitectureSpecificEncoding) {
  const std::array<uint8_t, 8> bytes = {0x34, 0x12, 0, 0, 0, 0, 0, 0};
  for (const auto [arch, kind, expected] : {
           std::tuple{ROCJITSU_CODE_ARCH_CDNA3, ConSanLdsAccessKind::Atomic, int64_t{0x34}},
           std::tuple{ROCJITSU_CODE_ARCH_CDNA3, ConSanLdsAccessKind::Read, int64_t{0x1234}},
           std::tuple{ROCJITSU_CODE_ARCH_CDNA4, ConSanLdsAccessKind::Atomic, int64_t{0x34}},
           std::tuple{ROCJITSU_CODE_ARCH_CDNA4, ConSanLdsAccessKind::Read, int64_t{0x1234}},
           std::tuple{ROCJITSU_CODE_ARCH_CDNA5, ConSanLdsAccessKind::Atomic, int64_t{0x1234}},
           std::tuple{ROCJITSU_CODE_ARCH_CDNA5, ConSanLdsAccessKind::Read, int64_t{0x1234}},
           std::tuple{ROCJITSU_CODE_ARCH_RDNA3, ConSanLdsAccessKind::Atomic, int64_t{0x1234}},
           std::tuple{ROCJITSU_CODE_ARCH_RDNA3, ConSanLdsAccessKind::Read, int64_t{0x1234}},
           std::tuple{ROCJITSU_CODE_ARCH_RDNA4, ConSanLdsAccessKind::Atomic, int64_t{0x1234}},
           std::tuple{ROCJITSU_CODE_ARCH_RDNA4, ConSanLdsAccessKind::Write, int64_t{0x1234}},
       }) {
    ProgramInventoryBuilder builder(bytes);
    builder.set_code_object_facts(true, 0, arch, ROCJITSU_CODE_TARGET_INVALID);
    ConSanKernelInfo kernel = make_inventory_kernel();
    ConSanAccessInventorySite site = make_inventory_lds_site("ds_single", 16, 0, 32);
    site.kind = kind;
    kernel.access_sites.push_back(std::move(site));
    builder.kernels().push_back(std::move(kernel));
    builder.rebuild_access_inventory(bytes);
    ASSERT_EQ(builder.view().access_sites().front().ranges.size(), 1u);
    EXPECT_EQ(builder.view().access_sites().front().ranges.front().static_byte_offset, expected);
  }
}

TEST(ConSanProgramInventory, UnreadableSingleRangeNativeOffsetRemainsAnExplicitMissingFact) {
  const std::array<uint8_t, 3> truncated_bytes = {0x34, 0x12, 0};
  ProgramInventoryBuilder builder(truncated_bytes);
  builder.set_code_object_facts(true, 0, ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_TARGET_GFX950);
  ConSanKernelInfo kernel = make_inventory_kernel();
  ConSanAccessInventorySite site = make_inventory_lds_site("ds_truncated", 16, 0, 32);
  kernel.access_sites.push_back(std::move(site));
  builder.kernels().push_back(std::move(kernel));
  builder.rebuild_access_inventory(truncated_bytes);

  ASSERT_EQ(builder.view().access_sites().front().ranges.size(), 1u);
  EXPECT_FALSE(builder.view().access_sites().front().ranges.front().static_byte_offset);
}

TEST(ConSanProgramInventory, TwoAddressRangesDecodeElementWidthScaleAndStableOrdinals) {
  /// One table row for an ISA spelling and its normalized per-range geometry.
  struct Case {
    std::string_view mnemonic;
    uint32_t width_bits;
    uint32_t scale;
  };
  constexpr std::array<Case, 16> cases = {
      Case{"ds_store_2addr_b32", 32, 4},
      Case{"ds_store_2addr_b64", 64, 8},
      Case{"ds_store_2addr_stride64_b32", 32, 256},
      Case{"ds_store_2addr_stride64_b64", 64, 512},
      Case{"ds_load_2addr_b32", 32, 4},
      Case{"ds_load_2addr_b64", 64, 8},
      Case{"ds_load_2addr_stride64_b32", 32, 256},
      Case{"ds_load_2addr_stride64_b64", 64, 512},
      Case{"ds_write2_b32", 32, 4},
      Case{"ds_write2_b64", 64, 8},
      Case{"ds_write2st64_b32", 32, 256},
      Case{"ds_write2st64_b64", 64, 512},
      Case{"ds_read2_b32", 32, 4},
      Case{"ds_read2_b64", 64, 8},
      Case{"ds_read2st64_b32", 32, 256},
      Case{"ds_read2st64_b64", 64, 512},
  };
  for (const Case &test : cases) {
    SCOPED_TRACE(test.mnemonic);
    const std::array<uint8_t, 8> bytes = {3, 5, 0, 0, 0, 0, 0, 0};
    ProgramInventoryBuilder builder(bytes);
    ConSanKernelInfo kernel = make_inventory_kernel();
    kernel.access_sites.push_back(
        make_inventory_lds_site(std::string(test.mnemonic), 40, 0, 2 * test.width_bits));
    builder.kernels().push_back(std::move(kernel));
    builder.rebuild_access_inventory(bytes);

    const ConSanAccessInventorySite &site = builder.view().access_sites().front();
    ASSERT_EQ(site.ranges.size(), 2u);
    EXPECT_EQ(site.ranges[0].static_byte_offset, 3 * test.scale);
    EXPECT_EQ(site.ranges[1].static_byte_offset, 5 * test.scale);
    EXPECT_EQ(site.ranges[0].byte_width, test.width_bits / 8u);
    EXPECT_EQ(site.ranges[1].byte_width, test.width_bits / 8u);
    EXPECT_EQ(site.ranges[0].id.physical, site.ranges[1].id.physical);
    EXPECT_EQ(site.ranges[0].id.member_ordinal, 0u);
    EXPECT_EQ(site.ranges[1].id.member_ordinal, 0u);
    EXPECT_EQ(site.ranges[0].id.range_ordinal, 0u);
    EXPECT_EQ(site.ranges[1].id.range_ordinal, 1u);
    EXPECT_NE(site.ranges[0].id, site.ranges[1].id);
    EXPECT_TRUE(site.complete());
  }
}

TEST(ConSanProgramInventory, FlatHintsBecomeTypedAddressSpaceProvenanceAndConfidence) {
  /// Expected target-neutral semantic facts for one legacy FLAT hint.
  struct Case {
    ConSanFlatAddressSpaceHint hint;
    ConSanAccessAddressSpace address_space;
    ConSanAccessProvenance provenance;
    ConSanSemanticConfidence confidence;
  };
  constexpr std::array<Case, 6> cases = {
      Case{ConSanFlatAddressSpaceHint::Group, ConSanAccessAddressSpace::Group,
           ConSanAccessProvenance::EncodedFlatSegment, ConSanSemanticConfidence::Exact},
      Case{ConSanFlatAddressSpaceHint::Private, ConSanAccessAddressSpace::NonGroup,
           ConSanAccessProvenance::EncodedFlatSegment, ConSanSemanticConfidence::Exact},
      Case{ConSanFlatAddressSpaceHint::Global, ConSanAccessAddressSpace::NonGroup,
           ConSanAccessProvenance::EncodedFlatSegment, ConSanSemanticConfidence::Exact},
      Case{ConSanFlatAddressSpaceHint::MaybeGroup, ConSanAccessAddressSpace::Group,
           ConSanAccessProvenance::PropagatedFlatPointer, ConSanSemanticConfidence::Conservative},
      Case{ConSanFlatAddressSpaceHint::MaybePrivate, ConSanAccessAddressSpace::NonGroup,
           ConSanAccessProvenance::PropagatedFlatPointer, ConSanSemanticConfidence::Conservative},
      Case{ConSanFlatAddressSpaceHint::Unknown, ConSanAccessAddressSpace::Unresolved,
           ConSanAccessProvenance::UnresolvedFlatPointer, ConSanSemanticConfidence::Ambiguous},
  };

  ProgramInventoryBuilder builder;
  ConSanKernelInfo kernel = make_inventory_kernel();
  for (size_t index = 0; index < cases.size(); ++index)
    kernel.access_sites.push_back(make_inventory_flat_site(cases[index].hint, 64 + index * 8));
  kernel.access_sites.front().operands.raw_op = 1;
  kernel.access_sites.front().operands.raw_saddr = 2;
  kernel.access_sites.front().operands.raw_scale_offset = true;
  kernel.access_sites.front().operands.raw_vaddr = 3;
  kernel.access_sites.front().operands.raw_vsrc = 4;
  kernel.access_sites.front().operands.raw_vdst = 5;
  kernel.access_sites.front().operands.raw_segment = 6;
  kernel.access_sites.front().operands.raw_scope = 7;
  kernel.access_sites.front().operands.raw_th = 8;
  builder.kernels().push_back(std::move(kernel));
  builder.rebuild_access_inventory({});

  const auto sites = builder.view().access_sites();
  ASSERT_EQ(sites.size(), cases.size());
  for (size_t index = 0; index < cases.size(); ++index) {
    SCOPED_TRACE(index);
    EXPECT_EQ(sites[index].origin, ConSanAccessOrigin::Flat);
    EXPECT_EQ(sites[index].flat_address_space_hint, cases[index].hint);
    EXPECT_EQ(sites[index].address_space, cases[index].address_space);
    EXPECT_EQ(sites[index].provenance, cases[index].provenance);
    EXPECT_EQ(sites[index].confidence, cases[index].confidence);
    ASSERT_EQ(sites[index].ranges.size(), 1u);
    EXPECT_EQ(sites[index].ranges[0].static_byte_offset, -8);
    EXPECT_EQ(sites[index].ranges[0].byte_width, 4u);
    EXPECT_TRUE(sites[index].complete());
  }
  EXPECT_EQ(sites.front().operands.raw_op, 1u);
  EXPECT_EQ(sites.front().operands.raw_saddr, 2u);
  EXPECT_EQ(sites.front().operands.raw_scale_offset, true);
  EXPECT_EQ(sites.front().operands.raw_vaddr, 3u);
  EXPECT_EQ(sites.front().operands.raw_vsrc, 4u);
  EXPECT_EQ(sites.front().operands.raw_vdst, 5u);
  EXPECT_EQ(sites.front().operands.raw_segment, 6u);
  EXPECT_EQ(sites.front().operands.raw_scope, 7u);
  EXPECT_EQ(sites.front().operands.raw_th, 8u);
}

TEST(ConSanProgramInventory, TypedExclusionsDescribeEveryInventoryConstructionFailure) {
  ProgramInventoryBuilder builder;
  ConSanKernelInfo kernel = make_inventory_kernel();

  ConSanAccessInventorySite malformed;
  malformed.origin = ConSanAccessOrigin::NativeLds;
  malformed.kind = ConSanLdsAccessKind::Other;
  malformed.physical_id.original_text_offset = 8;
  malformed.mnemonic = "ds_other";
  kernel.access_sites.push_back(malformed);

  ConSanAccessInventorySite missing_address = make_inventory_lds_site("ds_store_b32", 16);
  missing_address.operands.address_vgpr.reset();
  kernel.access_sites.push_back(missing_address);

  ConSanAccessInventorySite unavailable_range =
      make_inventory_lds_site("ds_store_2addr_b32", 24, 128, 64);
  kernel.access_sites.push_back(unavailable_range);
  builder.kernels().push_back(std::move(kernel));
  builder.rebuild_access_inventory({});

  const auto sites = builder.view().access_sites();
  ASSERT_EQ(sites.size(), 3u);
  EXPECT_EQ(exclusion_reasons(sites[0]),
            (std::vector<ConSanInventoryExclusionReason>{
                ConSanInventoryExclusionReason::NonAccessInstruction,
                ConSanInventoryExclusionReason::InvalidInstructionSize,
                ConSanInventoryExclusionReason::InvalidAccessWidth,
                ConSanInventoryExclusionReason::MissingAddressOperand}));
  EXPECT_FALSE(sites[0].complete());
  EXPECT_EQ(exclusion_reasons(sites[1]),
            (std::vector<ConSanInventoryExclusionReason>{
                ConSanInventoryExclusionReason::MissingAddressOperand}));
  EXPECT_FALSE(sites[1].complete());
  EXPECT_EQ(exclusion_reasons(sites[2]),
            (std::vector<ConSanInventoryExclusionReason>{
                ConSanInventoryExclusionReason::RangeEncodingUnavailable}));
  EXPECT_TRUE(sites[2].ranges.empty());
  EXPECT_FALSE(sites[2].complete());
}

TEST(ConSanProgramInventory, SymbolAliasesSharePhysicalAndRangeIdentityButKeepAttribution) {
  const std::array<uint8_t, 8> bytes = {};
  ProgramInventoryBuilder builder(bytes);
  ConSanKernelInfo kernel = make_inventory_kernel("kernel_alias");
  kernel.access_sites.push_back(make_inventory_lds_site("ds_store_b32", 80));
  builder.kernels().push_back(std::move(kernel));
  ConSanFunctionInfo function;
  function.name = "function_alias";
  function.entry_text_offset = 72;
  function.access_sites.push_back(make_inventory_lds_site("ds_store_b32", 80));
  builder.functions().push_back(std::move(function));
  builder.rebuild_access_inventory(bytes);

  const auto sites = builder.view().access_sites();
  ASSERT_EQ(sites.size(), 2u);
  EXPECT_EQ(sites[0].physical_id, sites[1].physical_id);
  ASSERT_EQ(sites[0].ranges.size(), 1u);
  ASSERT_EQ(sites[1].ranges.size(), 1u);
  EXPECT_EQ(sites[0].ranges[0].id, sites[1].ranges[0].id);
  EXPECT_NE(sites[0].container, sites[1].container);
  EXPECT_EQ(sites[0].container.kind, ConSanProgramContainerKind::Kernel);
  EXPECT_EQ(sites[1].container.kind, ConSanProgramContainerKind::Function);
}

TEST(ConSanProgramInventory, RealCodeObjectPublishesDecodedContainersAndNormalizedAccesses) {
  std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  mutate_first_kernel_descriptor(
      bytes, [](KD &descriptor) { descriptor.group_segment_fixed_size = 1234u; });
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;
  options.moi_engine = ConSanMoiEngine::RecordReplay;
  options.fault_dry_run = true;
  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_EQ(result.program_inventory.code_object_id(), make_consan_code_object_id(bytes));
  EXPECT_EQ(result.program_inventory.code_object_id().byte_size, bytes.size());
  EXPECT_EQ(result.program_inventory.arch(), result.program_inventory.arch());
  EXPECT_EQ(result.program_inventory.target(), ROCJITSU_CODE_TARGET_GFX1201);
  EXPECT_TRUE(result.program_inventory.kernel_metadata_trustworthy());
  EXPECT_EQ(result.program_inventory.malformed_kernel_metadata_note_count(), 0u);
  EXPECT_TRUE(result.program_inventory.semantic_arch_required());

  size_t decoded_access_count = 0;
  for (const ConSanKernelInfo &kernel : result.program_inventory.kernels())
    decoded_access_count += kernel.access_sites.size();
  for (const ConSanFunctionInfo &function : result.program_inventory.functions())
    decoded_access_count += function.access_sites.size();
  EXPECT_EQ(result.program_inventory.access_sites().size(), decoded_access_count);

  ASSERT_EQ(result.program_inventory.kernels().size(), 1u);
  EXPECT_EQ(result.program_inventory.kernels().front().declared_group_segment_bytes, 1234u);
  ASSERT_EQ(result.program_inventory.access_sites().size(), 2u);
  for (const ConSanAccessInventorySite &site : result.program_inventory.access_sites()) {
    const auto decoded_site =
        std::ranges::find_if(result.program_inventory.kernels().front().access_sites,
                             [&](const ConSanAccessInventorySite &candidate) {
                               return candidate.physical_id.original_text_offset ==
                                      site.physical_id.original_text_offset;
                             });
    ASSERT_NE(decoded_site, result.program_inventory.kernels().front().access_sites.end());
    EXPECT_EQ(site.file_offset, decoded_site->file_offset);
    EXPECT_EQ(site.instruction_size, decoded_site->instruction_size);
    EXPECT_EQ(site.decoded_width_bits, decoded_site->decoded_width_bits);
    EXPECT_EQ(site.kind, decoded_site->kind);
    EXPECT_EQ(site.mnemonic, decoded_site->mnemonic);
    EXPECT_EQ(site.operands.destination_vgpr, decoded_site->operands.destination_vgpr);
    EXPECT_EQ(site.operands.address_vgpr, decoded_site->operands.address_vgpr);
    EXPECT_EQ(site.operands.data_vgpr, decoded_site->operands.data_vgpr);
    EXPECT_EQ(
        site.execution_owner_descriptor_file_offsets,
        (std::vector<uint64_t>{result.program_inventory.kernels().front().descriptor_file_offset}));
  }
}

} // namespace
} // namespace rocjitsu
