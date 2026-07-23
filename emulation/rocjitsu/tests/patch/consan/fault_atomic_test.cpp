// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"

namespace rocjitsu {
namespace {

TEST(ConSan, AtomicAddressFaultCarriesPristinePerturbationPlan) {
  const auto wait_store = build_s_wait_storecnt0(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(wait_store);
  const std::vector<uint32_t> words = {0xEE0B0000u, 0u,          0u, *wait_store,
                                       0xEE158004u, 0x00980000u, 2u, 0xBFB00000u};
  const auto bytes = make_rdna4_lds_code_object(words);
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  inventory_options.fault_dry_run = true;
  const ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  ASSERT_EQ(inventory.fault_sites.size(), 1u);
  ConSanOptions select = inventory_options;
  select.sc_perturb_kind = ConSanPerturbationKind::Atomic;
  select.sc_perturb_edge = ConSanPerturbationEdge::Release;
  select.sc_perturb_required_count = 1;
  const ConSanResult selected = try_patch_consan(bytes, select);
  ASSERT_EQ(selected.perturbation_plans.size(), 1u);
  ConSanOptions options = select;
  options.fault_dry_run = false;
  options.fault_atomic_wrong_address = true;
  options.fault_atomic_address_delta = 4;
  options.fault_require_exactly_one = true;
  options.fault_site_identity = inventory.fault_sites.front().identity;
  options.sc_perturb_identity = selected.perturbation_plans.front().candidate_identity;
  const ConSanResult result = try_patch_consan(bytes, options);
  ASSERT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid)
      << testing::PrintToString(result.errors);
  EXPECT_TRUE(result.staged_composition_validated);
  EXPECT_EQ(result.planned_fault_mutations, 1u);
  EXPECT_EQ(result.applied_fault_mutations, 1u);
  ASSERT_EQ(result.fault_plans.size(), 1u);
  EXPECT_EQ(result.fault_plans.front().primary_identity, inventory.fault_sites.front().identity);
  EXPECT_EQ(result.applied_perturbations, 1u);
  const auto mutation = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::InlineAtomicAddressRewrite;
  });
  const auto perturbation = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineScPerturbation, &ConSanPatchInfo::kind);
  ASSERT_NE(mutation, result.patches.end());
  ASSERT_NE(perturbation, result.patches.end());
  EXPECT_NE(mutation->anchor_offset, perturbation->anchor_offset);
  EXPECT_EQ(perturbation->perturbation_source_candidate_identity,
            selected.perturbation_plans.front().candidate_identity);
  ASSERT_EQ(perturbation->owner_descriptor_file_offsets.size(), 1u);
  EXPECT_EQ(perturbation->owner_descriptor_file_offsets.front(),
            result.kernels.front().descriptor_file_offset);

  ConSanResult wrong_owner = result;
  auto &owner_patch = *std::ranges::find(
      wrong_owner.patches, ConSanPatchKind::TrampolineScPerturbation, &ConSanPatchInfo::kind);
  ASSERT_TRUE(owner_patch.perturbation_source_owner_descriptor_file_offset);
  *owner_patch.perturbation_source_owner_descriptor_file_offset += sizeof(uint32_t);
  const auto owner_errors = validate_consan_modified_elf(bytes, wrong_owner);
  EXPECT_TRUE(std::ranges::any_of(owner_errors, [](const std::string &error) {
    return error.find("pristine kernel owner") != std::string::npos;
  }));
  ConSanResult stale = result;
  auto &stale_patch = *std::ranges::find(stale.patches, ConSanPatchKind::TrampolineScPerturbation,
                                         &ConSanPatchInfo::kind);
  stale_patch.perturbation_source_sequence_identity += "|stale";
  const auto identity_errors = validate_consan_modified_elf(bytes, stale);
  EXPECT_TRUE(std::ranges::any_of(identity_errors, [](const std::string &error) {
    return error.find("pristine admitted sequence edge") != std::string::npos;
  }));
  ConSanResult stale_anchor = result;
  auto &stale_anchor_patch = *std::ranges::find(
      stale_anchor.patches, ConSanPatchKind::TrampolineScPerturbation, &ConSanPatchInfo::kind);
  stale_anchor_patch.perturbation_source_anchor_identity += "|stale";
  const auto anchor_errors = validate_consan_modified_elf(bytes, stale_anchor);
  EXPECT_TRUE(std::ranges::any_of(anchor_errors, [](const std::string &error) {
    return error.find("outer sequence member") != std::string::npos;
  }));
}

TEST(ConSan, AtomicOrderFaultComposesWithRemovedReleaseBoundary) {
  const auto wait_store = build_s_wait_storecnt0(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(wait_store);
  const std::vector<uint32_t> words = {0xEE0B0000u, 0u,          0u, *wait_store,
                                       0xEE158004u, 0x00980000u, 2u, 0xBFB00000u};
  const auto bytes = make_rdna4_lds_code_object(words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.fault_atomic_weaken_order = true;
  options.fault_require_exactly_one = true;
  options.sc_perturb_kind = ConSanPerturbationKind::Atomic;
  options.sc_perturb_edge = ConSanPerturbationEdge::Release;
  options.sc_perturb_required_count = 1;
  options.sc_perturb_sleep = 6;
  const ConSanResult result = try_patch_consan(bytes, options);
  ASSERT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid)
      << testing::PrintToString(result.errors);
  EXPECT_TRUE(result.staged_composition_validated);
  const auto mutation = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::InlineAtomicOrderRewrite;
  });
  const auto perturbation = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineScPerturbation, &ConSanPatchInfo::kind);
  ASSERT_NE(mutation, result.patches.end());
  ASSERT_NE(perturbation, result.patches.end());
  EXPECT_EQ(mutation->anchor_offset, perturbation->anchor_offset);
  EXPECT_TRUE(perturbation->perturbation_composite_atomic_overlap);
  EXPECT_TRUE(perturbation->perturbation_composite_removed_boundary);
  AmdGpuCodeObject replacement(result.elf_bytes.data(), result.elf_bytes.size());
  const Section *text = replacement.text_sections().front();
  std::array<uint32_t, 4> body{};
  std::memcpy(body.data(), text->data() + perturbation->trampoline_offset, sizeof(body));
  EXPECT_EQ(body[0], build_s_sleep(6, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_TRUE(std::ranges::all_of(std::span(body).subspan(1), [](uint32_t word) {
    return word == build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
  }));
  ConSanResult corrupted = result;
  const uint32_t cache = 0xEE0B0000u;
  std::memcpy(corrupted.elf_bytes.data() + text->sectionOffset() + perturbation->trampoline_offset +
                  sizeof(uint32_t),
              &cache, sizeof(cache));
  const auto errors = validate_consan_modified_elf(bytes, corrupted);
  EXPECT_TRUE(std::ranges::any_of(errors, [](const std::string &error) {
    return error.find("resurrected a removed cache operation") != std::string::npos;
  }));
}

TEST(ConSan, AtomicFaultRollsBackWhenCarriedPerturbationIsUnreachable) {
  const auto wait_store = build_s_wait_storecnt0(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(wait_store);
  std::vector<uint32_t> words(40000u, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  words[0] = 0xEE0B0000u;
  words[1] = words[2] = 0u;
  words[3] = *wait_store;
  words[4] = 0xEE158004u;
  words[5] = 0x00980000u;
  words[6] = 2u;
  words.back() = 0xBFB00000u;
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.fault_atomic_wrong_address = true;
  options.fault_require_exactly_one = true;
  options.sc_perturb_kind = ConSanPerturbationKind::Atomic;
  options.sc_perturb_edge = ConSanPerturbationEdge::Release;
  options.sc_perturb_required_count = 1;
  const ConSanResult result =
      try_patch_consan(make_rdna4_lds_code_object(words, "atomic_composite_far"), options);
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::Unsupported);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
  EXPECT_TRUE(result.patches.empty());
  EXPECT_EQ(result.applied_fault_mutations, 0u);
  EXPECT_EQ(result.applied_perturbations, 0u);

  ConSanOptions discovery = options;
  discovery.fault_dry_run = true;
  discovery.probe_lds_check_trap = true;
  discovery.max_patches = 2;
  const ConSanResult dry_run =
      try_patch_consan(make_rdna4_lds_code_object(words, "atomic_composite_far"), discovery);
  EXPECT_FALSE(dry_run.modified);
  EXPECT_TRUE(dry_run.access_plans.empty());
  EXPECT_FALSE(dry_run.composite_proof);
}

TEST(ConSan, AtomicAddressFaultComposesWithExactFlatUnknownCasReleaseEdge) {
  const std::vector<uint8_t> bytes = make_rdna4_ordered_flat_cas_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.fault_atomic_wrong_address = true;
  options.fault_require_exactly_one = true;
  options.sc_perturb_kind = ConSanPerturbationKind::Atomic;
  options.sc_perturb_edge = ConSanPerturbationEdge::Release;
  options.sc_perturb_required_count = 1;
  options.max_patches = 2;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_TRUE(result.staged_composition_validated);
  EXPECT_EQ(result.applied_fault_mutations, 1u);
  EXPECT_EQ(result.applied_perturbations, 1u);
  EXPECT_TRUE(std::ranges::any_of(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::InlineAtomicAddressRewrite;
  }));
  EXPECT_TRUE(std::ranges::any_of(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineScPerturbation;
  }));
}

TEST(ConSan, FaultInventoryIncludesAtomicOperandsAndRoles) {
  const std::vector<uint8_t> bytes = make_rdna4_flat_atomic_release_acquire_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_EQ(result.fault_sites.size(), 2u);
  EXPECT_EQ(result.fault_sites[0].kind, ConSanFaultSiteKind::Atomic);
  EXPECT_EQ(result.fault_sites[0].occurrence, 0u);
  EXPECT_EQ(result.fault_sites[1].occurrence, 1u);
  EXPECT_EQ(result.fault_sites[0].semantic_role, "atomic-order-unknown");
  EXPECT_EQ(result.fault_sites[1].semantic_role, "atomic-order-unknown");
  ASSERT_TRUE(result.fault_sites[0].sync_event_identity);
  ASSERT_TRUE(result.fault_sites[0].sync_sequence_identity);
  EXPECT_EQ(result.fault_sites[0].sync_confidence, ConSanSemanticConfidence::Unsupported);
  EXPECT_EQ(result.fault_sites[0].sync_memory_role, ConSanSyncMemoryRole::Unknown);
  EXPECT_NE(result.fault_sites[1].decoded_operands.find("raw_ioffset=0"), std::string::npos);
  EXPECT_NE(result.fault_sites[1].decoded_operands.find("raw_scope=2"), std::string::npos);
  EXPECT_NE(result.fault_sites[1].decoded_operands.find("raw_th=1"), std::string::npos);
  EXPECT_NE(result.fault_sites[1].decoded_operands.find("returns_old=1"), std::string::npos);
}

TEST(ConSan, FaultAtomicExactIdentitySupersedesGlobalIndex) {
  const std::vector<uint8_t> bytes = make_rdna4_flat_atomic_release_acquire_code_object();
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  ASSERT_EQ(inventory.fault_sites.size(), 2u);

  ConSanOptions options = inventory_options;
  options.fault_atomic_wrong_address = true;
  options.fault_atomic_address_delta = 8;
  options.fault_atomic_index = 0;
  options.fault_site_identity = inventory.fault_sites[1].identity;
  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().anchor_offset, inventory.fault_sites[1].text_offset);
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.kernels.front().atomic_sites.size(), 2u);
  ASSERT_TRUE(result.kernels.front().atomic_sites[0].raw_ioffset);
  ASSERT_TRUE(result.kernels.front().atomic_sites[1].raw_ioffset);
  EXPECT_EQ(*result.kernels.front().atomic_sites[0].raw_ioffset, 0);
  EXPECT_EQ(*result.kernels.front().atomic_sites[1].raw_ioffset, 8);
  EXPECT_EQ(result.requested_fault_mutations, 1u);
  EXPECT_EQ(result.applied_fault_mutations, 1u);
}

TEST(ConSan, FaultAtomicWrongAddressRejectsUnalignedDelta) {
  const std::vector<uint8_t> bytes = make_rdna4_flat_atomic_release_acquire_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.fault_atomic_wrong_address = true;
  options.fault_atomic_address_delta = 2;

  const ConSanResult result = try_patch_consan(bytes, options);

  EXPECT_EQ(result.outcome, ConSanTransformOutcome::Invalid);
  EXPECT_EQ(result.applied_fault_mutations, 0u);
  EXPECT_TRUE(std::ranges::any_of(result.errors, [](const std::string &error) {
    return error.find("positive aligned signed-24-bit offset") != std::string::npos;
  }));
}

TEST(ConSan, FaultAtomicDryRunPreservesBytesAndReportsExactPlans) {
  const std::vector<uint8_t> bytes = make_rdna4_ordered_flat_atomic_release_acquire_code_object();
  const std::vector<uint8_t> original = bytes;
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  ASSERT_EQ(inventory.fault_sites.size(), 2u);

  ConSanOptions options = inventory_options;
  options.fault_atomic_wrong_address = true;
  options.fault_atomic_weaken_order = true;
  options.fault_dry_run = true;
  options.fault_site_identity = inventory.fault_sites[1].identity;
  const ConSanResult result = try_patch_consan(bytes, options);

  EXPECT_EQ(bytes, original);
  EXPECT_FALSE(result.modified);
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::Unchanged);
  EXPECT_TRUE(result.elf_bytes.empty());
  EXPECT_TRUE(result.patches.empty());
  ASSERT_EQ(result.fault_plans.size(), 2u);
  EXPECT_EQ(result.fault_plans[0].kind, ConSanFaultMutationKind::AtomicWrongAddress);
  EXPECT_EQ(result.fault_plans[0].primary_identity, inventory.fault_sites[1].identity);
  EXPECT_FALSE(result.fault_plans[0].companion_identity);
  EXPECT_EQ(result.requested_fault_mutations, 2u);
  EXPECT_EQ(result.fault_plans[1].kind, ConSanFaultMutationKind::AtomicWeakenOrder);
  EXPECT_EQ(result.fault_plans[1].primary_identity, inventory.fault_sites[1].identity);
  EXPECT_TRUE(result.fault_plans[1].companion_identity);
  EXPECT_EQ(result.planned_fault_mutations, 2u);
  EXPECT_EQ(result.applied_fault_mutations, 0u);
}

TEST(ConSan, FaultAtomicWeakenOrderDryRunRejectsThAsOrderingField) {
  const std::vector<uint8_t> bytes = make_rdna4_ordered_flat_atomic_release_acquire_code_object();
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_track_atomics = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.max_patches = 8;
  options.fault_atomic_weaken_order = true;
  options.fault_dry_run = true;
  options.fault_atomic_index = 1;
  const ConSanResult result = try_patch_consan(bytes, options);

  EXPECT_EQ(result.outcome, ConSanTransformOutcome::Unchanged);
  EXPECT_FALSE(result.modified);
  ASSERT_EQ(result.fault_plans.size(), 1u);
  EXPECT_EQ(result.fault_plans.front().kind, ConSanFaultMutationKind::AtomicWeakenOrder);
  EXPECT_TRUE(result.fault_plans.front().companion_identity);
}

TEST(ConSan, FaultAtomicWeakenOrderSelectsExplicitReleaseAndAcquireEdges) {
  const std::vector<uint8_t> bytes = make_rdna4_ordered_flat_atomic_release_acquire_code_object();

  ConSanOptions release_options;
  release_options.flavor = ConSanFlavor::SuperCollider;
  release_options.fault_atomic_weaken_order = true;
  release_options.fault_atomic_order_edge = ConSanAtomicOrderEdge::Release;
  release_options.fault_atomic_index = 0;
  const ConSanResult release = try_patch_consan(bytes, release_options);
  ASSERT_TRUE(release.errors.empty()) << testing::PrintToString(release.errors);
  EXPECT_EQ(release.applied_fault_mutations, 1u);
  EXPECT_TRUE(std::ranges::any_of(release.warnings, [](const std::string &warning) {
    return warning.find("removed associated global_wb") != std::string::npos;
  }));

  ConSanOptions acquire_options = release_options;
  acquire_options.fault_atomic_order_edge = ConSanAtomicOrderEdge::Acquire;
  acquire_options.fault_atomic_index = 1;
  const ConSanResult acquire = try_patch_consan(bytes, acquire_options);
  ASSERT_TRUE(acquire.errors.empty()) << testing::PrintToString(acquire.errors);
  EXPECT_EQ(acquire.applied_fault_mutations, 1u);
  EXPECT_TRUE(std::ranges::any_of(acquire.warnings, [](const std::string &warning) {
    return warning.find("removed associated global_inv") != std::string::npos;
  }));
}

TEST(ConSan, FaultAtomicWeakenOrderSupportsCdna4CompilerSequence) {
  const auto release = cdna4::build_mubuf(cdna4::kBufferWbl2Mubuf, {.sc1 = 1});
  const auto acquire = cdna4::build_mubuf(cdna4::kBufferInvMubuf, {.sc1 = 1});
  const auto atomic = build_cdna4_flat_atomic_add_u32(
      /*vaddr=*/2, /*vsrc=*/4, /*vdst=*/5, /*return_old_value=*/true,
      /*scope=*/2, ROCJITSU_CODE_ARCH_CDNA4);
  const auto wait = build_cdna4_s_wait_flat0(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(atomic && wait);
  std::vector<uint32_t> words;
  words.insert(words.end(), release.begin(), release.end());
  words.push_back(*wait);
  words.insert(words.end(), atomic->begin(), atomic->end());
  words.push_back(*wait);
  words.insert(words.end(), acquire.begin(), acquire.end());
  words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4));
  const std::vector<uint8_t> bytes = make_cdna4_lds_code_object(words, "atomic_order_fault");

  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  ASSERT_TRUE(consan_patch_succeeded(inventory));
  ASSERT_EQ(inventory.fault_sites.size(), 1u);

  ConSanOptions order_options = inventory_options;
  order_options.fault_atomic_weaken_order = true;
  order_options.fault_atomic_order_edge = ConSanAtomicOrderEdge::Release;
  order_options.fault_site_identity = inventory.fault_sites.front().identity;
  order_options.fault_require_exactly_one = true;
  const ConSanResult order = try_patch_consan(bytes, order_options);
  ASSERT_TRUE(order.errors.empty()) << testing::PrintToString(order.errors);
  EXPECT_EQ(order.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_EQ(order.applied_fault_mutations, 1u);
  EXPECT_TRUE(std::ranges::any_of(order.warnings, [](const std::string &warning) {
    return warning.find("removed associated buffer_wbl2") != std::string::npos;
  }));
  ASSERT_FALSE(order.elf_bytes.empty());
  const uint64_t text_file_offset = inventory.kernels.front().text_file_offset;
  const uint32_t nop = build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4);
  for (uint64_t offset = 0; offset < release.size() * sizeof(uint32_t);
       offset += sizeof(uint32_t)) {
    uint32_t staged_word = 0;
    std::memcpy(&staged_word, order.elf_bytes.data() + text_file_offset + offset,
                sizeof(staged_word));
    EXPECT_EQ(staged_word, nop);
  }
  const uint64_t preserved_begin = release.size() * sizeof(uint32_t);
  const uint64_t preserved_size = (words.size() - release.size() - 1u) * sizeof(uint32_t);
  EXPECT_TRUE(std::equal(bytes.begin() + text_file_offset + preserved_begin,
                         bytes.begin() + text_file_offset + preserved_begin + preserved_size,
                         order.elf_bytes.begin() + text_file_offset + preserved_begin));

  ConSanOptions scope_options = inventory_options;
  scope_options.fault_atomic_weaken_scope = true;
  scope_options.fault_dry_run = true;
  scope_options.fault_site_identity = inventory.fault_sites.front().identity;
  const ConSanResult scope = try_patch_consan(bytes, scope_options);
  EXPECT_TRUE(scope.fault_plans.empty());
  EXPECT_TRUE(std::ranges::any_of(scope.warnings, [](const std::string &warning) {
    return warning.find("cannot weaken scope") != std::string::npos;
  }));

  scope_options.fault_dry_run = false;
  const ConSanResult live_scope = try_patch_consan(bytes, scope_options);
  EXPECT_EQ(live_scope.outcome, ConSanTransformOutcome::Invalid);
  EXPECT_FALSE(live_scope.modified);
  EXPECT_TRUE(std::ranges::any_of(live_scope.errors, [](const std::string &error) {
    return error.find("scope fault is unsupported") != std::string::npos;
  }));

  ConSanOptions address_options = inventory_options;
  address_options.fault_atomic_wrong_address = true;
  address_options.fault_atomic_address_delta = 4;
  address_options.fault_site_identity = inventory.fault_sites.front().identity;
  const ConSanResult address = try_patch_consan(bytes, address_options);
  EXPECT_EQ(address.outcome, ConSanTransformOutcome::Invalid);
  EXPECT_FALSE(address.modified);
  EXPECT_TRUE(std::ranges::any_of(address.errors, [](const std::string &error) {
    return error.find("address fault is unsupported") != std::string::npos;
  }));
}

TEST(ConSan, FaultAtomicWeakenOrderExplicitEdgeFailsClosedWhenAbsent) {
  const std::vector<uint8_t> bytes = make_rdna4_ordered_flat_atomic_release_acquire_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.fault_atomic_weaken_order = true;
  options.fault_atomic_order_edge = ConSanAtomicOrderEdge::Acquire;
  options.fault_atomic_index = 0;
  options.fault_dry_run = true;
  const ConSanResult result = try_patch_consan(bytes, options);

  EXPECT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.fault_plans.empty());
  EXPECT_EQ(result.planned_fault_mutations, 0u);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("no qualified atomic ordering boundary") != std::string::npos;
  }));
}

TEST(ConSan, FaultAtomicWeakenScopeDryRunPreservesBytesAndIdentity) {
  const std::vector<uint8_t> bytes = make_rdna4_ordered_flat_atomic_release_acquire_code_object();
  const std::vector<uint8_t> original = bytes;
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  ASSERT_EQ(inventory.fault_sites.size(), 2u);

  ConSanOptions options = inventory_options;
  options.fault_atomic_weaken_scope = true;
  options.fault_dry_run = true;
  options.fault_site_identity = inventory.fault_sites[1].identity;
  const ConSanResult result = try_patch_consan(bytes, options);

  EXPECT_EQ(bytes, original);
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::Unchanged);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
  EXPECT_TRUE(result.patches.empty());
  ASSERT_EQ(result.fault_plans.size(), 1u);
  EXPECT_EQ(result.fault_plans.front().kind, ConSanFaultMutationKind::AtomicWeakenScope);
  EXPECT_EQ(result.fault_plans.front().primary_identity, inventory.fault_sites[1].identity);
  EXPECT_FALSE(result.fault_plans.front().companion_identity);
  EXPECT_EQ(result.requested_fault_mutations, 1u);
  EXPECT_EQ(result.planned_fault_mutations, 1u);
  EXPECT_EQ(result.applied_fault_mutations, 0u);
}

TEST(ConSan, FaultAtomicWrongAddressIsVisibleToSubsequentInventory) {
  const std::vector<uint8_t> bytes = make_rdna4_flat_atomic_release_acquire_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.fault_atomic_wrong_address = true;
  options.fault_atomic_index = 1;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches[0].kind, ConSanPatchKind::InlineAtomicAddressRewrite);
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.kernels[0].atomic_sites.size(), 2u);
  ASSERT_TRUE(result.kernels[0].atomic_sites[1].raw_ioffset);
  EXPECT_EQ(*result.kernels[0].atomic_sites[1].raw_ioffset, 4);
}

TEST(ConSan, FaultAtomicWeakenOrderLeavesThReturnBehaviorUntouched) {
  const std::vector<uint8_t> bytes = make_rdna4_ordered_flat_atomic_release_acquire_code_object();
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_track_atomics = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.max_patches = 8;
  options.fault_atomic_weaken_order = true;
  options.fault_atomic_index = 1;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(std::ranges::any_of(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.phase == ConSanPatchPhase::Mutation &&
           patch.kind == ConSanPatchKind::InlineAtomicOrderRewrite;
  }));
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.kernels[0].atomic_sites.size(), 2u);
  ASSERT_TRUE(result.kernels[0].atomic_sites[1].raw_th);
  ASSERT_TRUE(result.kernels[0].atomic_sites[1].returns_old_value);
  EXPECT_EQ(*result.kernels[0].atomic_sites[1].raw_th, 1u);
  EXPECT_TRUE(*result.kernels[0].atomic_sites[1].returns_old_value);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("removed associated global_inv") != std::string::npos;
  }));
}

TEST(ConSan, FaultAtomicWeakenOrderPreservesReturningCasDataOperation) {
  const std::vector<uint8_t> bytes = make_rdna4_ordered_flat_cas_code_object();
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_track_atomics = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.max_patches = 8;
  options.fault_atomic_weaken_order = true;
  options.fault_require_exactly_one = true;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_EQ(result.requested_fault_mutations, 1u);
  EXPECT_EQ(result.applied_fault_mutations, 1u);
  EXPECT_TRUE(std::ranges::any_of(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.phase == ConSanPatchPhase::Mutation &&
           patch.kind == ConSanPatchKind::InlineAtomicOrderRewrite;
  }));
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.kernels[0].atomic_sites.size(), 1u);
  const ConSanAtomicSite &atomic = result.kernels[0].atomic_sites.front();
  EXPECT_EQ(atomic.mnemonic, "flat_atomic_cmpswap_b32");
  ASSERT_TRUE(atomic.returns_old_value);
  ASSERT_TRUE(atomic.raw_th);
  ASSERT_TRUE(atomic.raw_scope);
  EXPECT_TRUE(*atomic.returns_old_value);
  EXPECT_EQ(*atomic.raw_th, 1u);
  EXPECT_EQ(*atomic.raw_scope, 2u);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("removed associated global_wb") != std::string::npos;
  }));
}

TEST(ConSan, FaultAtomicWeakenScopeLeavesThReturnAndAddressUntouched) {
  const std::vector<uint8_t> bytes = make_rdna4_ordered_flat_atomic_release_acquire_code_object();
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_track_atomics = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.max_patches = 8;
  options.fault_atomic_weaken_scope = true;
  options.fault_atomic_index = 1;
  options.fault_require_exactly_one = true;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_EQ(result.requested_fault_mutations, 1u);
  EXPECT_EQ(result.applied_fault_mutations, 1u);
  EXPECT_TRUE(std::ranges::any_of(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.phase == ConSanPatchPhase::Mutation &&
           patch.kind == ConSanPatchKind::InlineAtomicScopeRewrite;
  }));
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.kernels[0].atomic_sites.size(), 2u);
  const ConSanAtomicSite &atomic = result.kernels[0].atomic_sites[1];
  ASSERT_TRUE(atomic.raw_scope);
  ASSERT_TRUE(atomic.raw_th);
  ASSERT_TRUE(atomic.returns_old_value);
  ASSERT_TRUE(atomic.raw_ioffset);
  EXPECT_EQ(*atomic.raw_scope, 0u);
  EXPECT_EQ(*atomic.raw_th, 1u);
  EXPECT_TRUE(*atomic.returns_old_value);
  EXPECT_EQ(*atomic.raw_ioffset, 0);
}

TEST(ConSan, FaultGlobalAtomicWrongAddressIsVisibleToSubsequentInventory) {
  const std::vector<uint8_t> bytes = make_rdna4_global_atomic_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.fault_atomic_wrong_address = true;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches[0].kind, ConSanPatchKind::InlineAtomicAddressRewrite);
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.kernels[0].atomic_sites.size(), 1u);
  ASSERT_TRUE(result.kernels[0].atomic_sites[0].raw_ioffset);
  ASSERT_TRUE(result.kernels[0].atomic_sites[0].raw_scope);
  ASSERT_TRUE(result.kernels[0].atomic_sites[0].raw_th);
  ASSERT_TRUE(result.kernels[0].atomic_sites[0].returns_old_value);
  EXPECT_EQ(*result.kernels[0].atomic_sites[0].raw_ioffset, 4);
  EXPECT_EQ(*result.kernels[0].atomic_sites[0].raw_scope, 2u);
  EXPECT_EQ(*result.kernels[0].atomic_sites[0].raw_th, 1u);
  EXPECT_TRUE(*result.kernels[0].atomic_sites[0].returns_old_value);
}

TEST(ConSan, FaultGlobalAtomicWeakenScopePreservesReturnedValueAndAddress) {
  const std::vector<uint8_t> bytes = make_rdna4_global_atomic_code_object();
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.fault_atomic_weaken_scope = true;
  options.fault_require_exactly_one = true;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_EQ(result.applied_fault_mutations, 1u);
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.kernels[0].atomic_sites.size(), 1u);
  const ConSanAtomicSite &atomic = result.kernels[0].atomic_sites.front();
  EXPECT_EQ(atomic.mnemonic, "global_atomic_add_f32");
  ASSERT_TRUE(atomic.raw_ioffset);
  ASSERT_TRUE(atomic.raw_scope);
  ASSERT_TRUE(atomic.raw_th);
  ASSERT_TRUE(atomic.returns_old_value);
  EXPECT_EQ(*atomic.raw_ioffset, 0);
  EXPECT_EQ(*atomic.raw_scope, 0u);
  EXPECT_EQ(*atomic.raw_th, 1u);
  EXPECT_TRUE(*atomic.returns_old_value);
}

TEST(ConSan, FaultGlobalAtomicWeakenOrderPreservesReturnedValue) {
  const std::vector<uint8_t> bytes = make_rdna4_ordered_global_atomic_code_object();
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.fault_atomic_weaken_order = true;
  options.fault_require_exactly_one = true;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_EQ(result.applied_fault_mutations, 1u);
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.kernels[0].atomic_sites.size(), 1u);
  const ConSanAtomicSite &atomic = result.kernels[0].atomic_sites.front();
  EXPECT_EQ(atomic.mnemonic, "global_atomic_add_f32");
  ASSERT_TRUE(atomic.raw_scope);
  ASSERT_TRUE(atomic.raw_th);
  ASSERT_TRUE(atomic.returns_old_value);
  EXPECT_EQ(*atomic.raw_scope, 2u);
  EXPECT_EQ(*atomic.raw_th, 1u);
  EXPECT_TRUE(*atomic.returns_old_value);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("removed associated global_wb") != std::string::npos;
  }));
}

TEST(ConSan, FaultNoReturnAtomicWeakenOrderRemovesExactReleaseWait) {
  const std::vector<uint8_t> bytes = make_rdna4_release_wait_no_return_bitwise_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.fault_atomic_weaken_order = true;
  options.fault_require_exactly_one = true;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_EQ(result.applied_fault_mutations, 1u);
  const auto mutation = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.phase == ConSanPatchPhase::Mutation &&
           patch.kind == ConSanPatchKind::InlineAtomicOrderRewrite;
  });
  ASSERT_NE(mutation, result.patches.end());
  EXPECT_EQ(mutation->anchor_offset, 0u);
  EXPECT_EQ(mutation->original_size, sizeof(uint32_t));
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("removed associated s_wait_storecnt_dscnt") != std::string::npos;
  }));
  AmdGpuCodeObject replacement(result.elf_bytes.data(), result.elf_bytes.size());
  const uint64_t text_file_offset = replacement.text_sections().front()->sectionOffset();
  uint32_t first_word = 0;
  std::memcpy(&first_word, result.elf_bytes.data() + text_file_offset, sizeof(first_word));
  EXPECT_EQ(first_word, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(ConSan, FaultAtomicWeakenOrderRemovesReleaseWaitBeforeWaitAlu) {
  const std::array<uint32_t, 6> text_words = {
      0xBFC10000u, // s_wait_storecnt 0
      0xBF88FF9Eu, // s_wait_alu
      0xEE0F0006u, 0x00980000u,
      0x00000000u, // global_atomic_and_b32 v0, v1, s[6:7], return old, device
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.fault_atomic_weaken_order = true;
  options.fault_atomic_order_edge = ConSanAtomicOrderEdge::Release;
  options.fault_require_exactly_one = true;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_EQ(result.applied_fault_mutations, 1u);
  const auto mutation = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.phase == ConSanPatchPhase::Mutation &&
           patch.kind == ConSanPatchKind::InlineAtomicOrderRewrite;
  });
  ASSERT_NE(mutation, result.patches.end());
  EXPECT_EQ(mutation->anchor_offset, 0u);
  AmdGpuCodeObject replacement(result.elf_bytes.data(), result.elf_bytes.size());
  const uint64_t text_file_offset = replacement.text_sections().front()->sectionOffset();
  uint32_t first_word = 0;
  uint32_t second_word = 0;
  std::memcpy(&first_word, result.elf_bytes.data() + text_file_offset, sizeof(first_word));
  std::memcpy(&second_word, result.elf_bytes.data() + text_file_offset + sizeof(uint32_t),
              sizeof(second_word));
  EXPECT_EQ(first_word, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(second_word, text_words[1]);
}

TEST(ConSan, FaultGlobalAtomicExactIdentityNoTargetFailsCardinalityWithoutMutation) {
  const std::vector<uint8_t> bytes = make_rdna4_global_atomic_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.fault_atomic_wrong_address = true;
  options.fault_site_identity = "fnv1a64:missing|kernel=missing|kind=atomic";
  options.fault_require_exactly_one = true;

  const ConSanResult result = try_patch_consan(bytes, options);

  EXPECT_EQ(result.outcome, ConSanTransformOutcome::Invalid);
  EXPECT_FALSE(result.modified);
  EXPECT_EQ(result.requested_fault_mutations, 1u);
  EXPECT_EQ(result.applied_fault_mutations, 0u);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("found no site with exact identity") != std::string::npos;
  }));
}

TEST(ConSan, FaultInventoryIncludesBufferAndDsAtomicEncodings) {
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const ConSanResult buffer = try_patch_consan(make_rdna4_buffer_atomic_code_object(), options);
  ASSERT_TRUE(buffer.errors.empty()) << testing::PrintToString(buffer.errors);
  ASSERT_EQ(buffer.fault_sites.size(), 1u);
  EXPECT_EQ(buffer.fault_sites.front().mnemonic, "buffer_atomic_add_u32");
  EXPECT_EQ(buffer.fault_sites.front().size, 12u);
  EXPECT_EQ(buffer.fault_sites.front().occurrence, 0u);
  EXPECT_NE(buffer.fault_sites.front().decoded_operands.find("raw_ioffset=0"), std::string::npos);
  EXPECT_NE(buffer.fault_sites.front().decoded_operands.find("raw_scope=2"), std::string::npos);
  EXPECT_NE(buffer.fault_sites.front().decoded_operands.find("returns_old=1"), std::string::npos);

  const ConSanResult ds = try_patch_consan(make_rdna4_ds_atomic_code_object(), options);
  ASSERT_TRUE(ds.errors.empty()) << testing::PrintToString(ds.errors);
  ASSERT_EQ(ds.fault_sites.size(), 1u);
  EXPECT_EQ(ds.fault_sites.front().mnemonic, "ds_add_u32");
  EXPECT_EQ(ds.fault_sites.front().size, 8u);
  EXPECT_EQ(ds.fault_sites.front().occurrence, 0u);
  EXPECT_NE(ds.fault_sites.front().decoded_operands.find("raw_addr=0"), std::string::npos);
}

TEST(ConSan, FaultBufferAtomicWrongAddressPreservesScopeAndReturnedValue) {
  const std::vector<uint8_t> bytes = make_rdna4_buffer_atomic_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.fault_atomic_wrong_address = true;
  options.fault_atomic_address_delta = 8;
  options.fault_require_exactly_one = true;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_EQ(result.applied_fault_mutations, 1u);
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.kernels.front().atomic_sites.size(), 1u);
  const ConSanAtomicSite &atomic = result.kernels.front().atomic_sites.front();
  EXPECT_EQ(atomic.mnemonic, "buffer_atomic_add_u32");
  ASSERT_TRUE(atomic.raw_ioffset);
  ASSERT_TRUE(atomic.raw_scope);
  ASSERT_TRUE(atomic.raw_th);
  ASSERT_TRUE(atomic.returns_old_value);
  EXPECT_EQ(*atomic.raw_ioffset, 8);
  EXPECT_EQ(*atomic.raw_scope, 2u);
  EXPECT_EQ(*atomic.raw_th, 1u);
  EXPECT_TRUE(*atomic.returns_old_value);
}

TEST(ConSan, FaultBufferAtomicWeakenScopePreservesAddressAndReturnedValue) {
  const std::vector<uint8_t> bytes = make_rdna4_buffer_atomic_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.fault_atomic_weaken_scope = true;
  options.fault_require_exactly_one = true;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_EQ(result.applied_fault_mutations, 1u);
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.kernels.front().atomic_sites.size(), 1u);
  const ConSanAtomicSite &atomic = result.kernels.front().atomic_sites.front();
  ASSERT_TRUE(atomic.raw_ioffset);
  ASSERT_TRUE(atomic.raw_scope);
  ASSERT_TRUE(atomic.raw_th);
  ASSERT_TRUE(atomic.returns_old_value);
  EXPECT_EQ(*atomic.raw_ioffset, 0);
  EXPECT_EQ(*atomic.raw_scope, 0u);
  EXPECT_EQ(*atomic.raw_th, 1u);
  EXPECT_TRUE(*atomic.returns_old_value);
}

TEST(ConSan, FaultBufferAtomicWeakenOrderPreservesDataOperationAndReturnedValue) {
  const std::vector<uint8_t> bytes = make_rdna4_ordered_buffer_atomic_code_object();
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.fault_atomic_weaken_order = true;
  options.fault_require_exactly_one = true;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_EQ(result.applied_fault_mutations, 1u);
  EXPECT_TRUE(std::ranges::any_of(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.phase == ConSanPatchPhase::Mutation &&
           patch.kind == ConSanPatchKind::InlineAtomicOrderRewrite;
  }));
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.kernels.front().atomic_sites.size(), 1u);
  const ConSanAtomicSite &atomic = result.kernels.front().atomic_sites.front();
  EXPECT_EQ(atomic.mnemonic, "buffer_atomic_add_u32");
  ASSERT_TRUE(atomic.raw_scope);
  ASSERT_TRUE(atomic.raw_th);
  ASSERT_TRUE(atomic.returns_old_value);
  EXPECT_EQ(*atomic.raw_scope, 2u);
  EXPECT_EQ(*atomic.raw_th, 1u);
  EXPECT_TRUE(*atomic.returns_old_value);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("removed associated global_wb") != std::string::npos;
  }));
}

TEST(ConSan, FaultDsAtomicWrongAddressUsesAlignedOffset0AndExactCardinality) {
  const std::vector<uint8_t> bytes = make_rdna4_ds_atomic_code_object();
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.fault_atomic_wrong_address = true;
  options.fault_atomic_address_delta = 4;
  options.fault_require_exactly_one = true;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_EQ(result.applied_fault_mutations, 1u);
  ASSERT_EQ(result.text_sections.size(), 1u);
  uint32_t mutated_word0 = 0;
  std::memcpy(&mutated_word0, result.elf_bytes.data() + result.text_sections.front().file_offset,
              sizeof(mutated_word0));
  EXPECT_EQ(mutated_word0 & 0xffu, 4u);
  EXPECT_EQ(mutated_word0 & ~0xffu, 0xD8000000u);
}

TEST(ConSan, FaultDsAtomicScopeAndOrderFailClosedBeforeStagingBytes) {
  const std::vector<uint8_t> bytes = make_rdna4_ds_atomic_code_object();
  for (const bool weaken_scope : {false, true}) {
    ConSanOptions options;
    options.flavor = ConSanFlavor::SuperCollider;
    options.fault_atomic_weaken_scope = weaken_scope;
    options.fault_atomic_weaken_order = !weaken_scope;

    const ConSanResult result = try_patch_consan(bytes, options);

    EXPECT_EQ(result.outcome, ConSanTransformOutcome::Invalid);
    EXPECT_FALSE(result.modified);
    EXPECT_TRUE(result.elf_bytes.empty());
    EXPECT_TRUE(result.patches.empty());
    EXPECT_TRUE(std::ranges::any_of(result.errors, [&](const std::string &error) {
      return error.find(weaken_scope ? "scope fault is unsupported"
                                     : "order fault is unsupported") != std::string::npos;
    }));
  }
}

} // namespace
} // namespace rocjitsu
