// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"

namespace rocjitsu {
namespace {

TEST(ConSan, FaultInventoryProvesDirectSharedHelperOwnersAndFiltersExactDispatch) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.helper_has_ordered_atomic = true;
  const std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object(fixture);
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::Moi;
  inventory_options.fault_dry_run = true;
  const ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  const auto site = std::ranges::find_if(inventory.fault_sites, [](const ConSanFaultSite &item) {
    return item.kind == ConSanFaultSiteKind::Atomic && item.container_name == "shared_lds_helper";
  });
  ASSERT_NE(site, inventory.fault_sites.end());
  ASSERT_EQ(site->execution_owners.size(), 2u);
  for (const ConSanExecutionOwner &owner : site->execution_owners)
    EXPECT_EQ(owner.proof, ConSanOwnerProofKind::DirectCall);

  for (std::string_view owner_name : {"shared_owner_0", "shared_owner_1"}) {
    ConSanOptions options = inventory_options;
    options.fault_atomic_wrong_address = true;
    options.fault_site_identity = site->identity;
    options.test_kernel_name_filter = owner_name;
    const ConSanResult selected = try_patch_consan(bytes, options);
    ASSERT_EQ(selected.fault_plans.size(), 1u) << testing::PrintToString(selected.warnings);
    EXPECT_EQ(selected.fault_plans.front().primary_identity, site->identity);
  }

  for (std::string_view rejected_filter : {"unrelated_kernel", "shared_owner_"}) {
    ConSanOptions options = inventory_options;
    options.fault_atomic_wrong_address = true;
    options.fault_site_identity = site->identity;
    options.test_kernel_name_filter = rejected_filter;
    const ConSanResult rejected = try_patch_consan(bytes, options);
    EXPECT_TRUE(rejected.fault_plans.empty());
  }
}

TEST(ConSan, FaultInventoryProvesRecoveredIndirectSharedHelperOwners) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.helper_has_ordered_atomic = true;
  fixture.use_indirect_calls = true;
  const std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object(fixture);
  ConSanOptions options = moi_options();
  options.fault_dry_run = true;
  const ConSanResult result = try_patch_consan(bytes, options);
  const auto site = std::ranges::find_if(result.fault_sites, [](const ConSanFaultSite &item) {
    return item.kind == ConSanFaultSiteKind::Atomic && item.container_name == "shared_lds_helper";
  });
  ASSERT_NE(site, result.fault_sites.end());
  ASSERT_EQ(site->execution_owners.size(), 2u);
  for (const ConSanExecutionOwner &owner : site->execution_owners)
    EXPECT_EQ(owner.proof, ConSanOwnerProofKind::RecoveredIndirectCall);
}

TEST(ConSan, FaultInventoryMarksKernelLocalOwner) {
  const std::vector<uint8_t> bytes = make_rdna4_global_atomic_code_object();
  ConSanOptions options = moi_options();
  options.fault_dry_run = true;
  const ConSanResult result = try_patch_consan(bytes, options);
  ASSERT_EQ(result.fault_sites.size(), 1u);
  ASSERT_EQ(result.fault_sites.front().execution_owners.size(), 1u);
  EXPECT_EQ(result.fault_sites.front().execution_owners.front().proof,
            ConSanOwnerProofKind::KernelLocal);
}

TEST(ConSan, FaultLoadSelectorSelectsOneStableOneBasedOccurrence) {
  ConSanFaultLoadSelector selector(/*requested_occurrence=*/2);
  const ConSanFaultLoadSelection first = selector.observe();
  const ConSanFaultLoadSelection second = selector.observe();
  const ConSanFaultLoadSelection third = selector.observe();
  EXPECT_EQ(first.occurrence, 1u);
  EXPECT_FALSE(first.selected);
  EXPECT_EQ(second.occurrence, 2u);
  EXPECT_TRUE(second.selected);
  EXPECT_EQ(third.occurrence, 3u);
  EXPECT_FALSE(third.selected);
  EXPECT_EQ(selector.observed(), 3u);
  EXPECT_EQ(selector.selected(), 1u);
  EXPECT_TRUE(selector.accepted());
}

TEST(ConSan, FaultLoadSelectorFailsClosedWhenOccurrenceIsAbsent) {
  ConSanFaultLoadSelector selector(/*requested_occurrence=*/2);
  EXPECT_FALSE(selector.accepted());
  EXPECT_FALSE(selector.observe().selected);
  EXPECT_FALSE(selector.accepted());
}

TEST(ConSan, FaultMutationCardinalityReportsZeroAndEnforcesGuard) {
  const std::vector<uint8_t> bytes = make_rdna4_flat_atomic_release_acquire_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.fault_atomic_wrong_address = true;
  options.fault_atomic_index = 99;

  const ConSanResult unguarded = try_patch_consan(bytes, options);
  EXPECT_EQ(unguarded.outcome, ConSanTransformOutcome::Unchanged);
  EXPECT_EQ(unguarded.requested_fault_mutations, 1u);
  EXPECT_EQ(unguarded.applied_fault_mutations, 0u);

  options.fault_require_exactly_one = true;
  const ConSanResult guarded = try_patch_consan(bytes, options);
  EXPECT_EQ(guarded.outcome, ConSanTransformOutcome::Invalid);
  EXPECT_EQ(guarded.requested_fault_mutations, 1u);
  EXPECT_EQ(guarded.applied_fault_mutations, 0u);
  EXPECT_TRUE(std::ranges::any_of(guarded.errors, [](const std::string &error) {
    return error.find("required exactly one applied mutation, got 0") != std::string::npos;
  }));
}

TEST(ConSan, FaultMutationCardinalityExcludesRetiredThOrderMutation) {
  const std::vector<uint8_t> bytes = make_rdna4_ordered_flat_atomic_release_acquire_code_object();
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_track_atomics = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.max_patches = 8;
  options.fault_atomic_wrong_address = true;
  options.fault_atomic_weaken_order = true;
  options.fault_atomic_index = 1;

  const ConSanResult unguarded = try_patch_consan(bytes, options);
  EXPECT_EQ(unguarded.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_EQ(unguarded.requested_fault_mutations, 2u);
  EXPECT_EQ(unguarded.applied_fault_mutations, 2u);
  EXPECT_TRUE(std::ranges::any_of(unguarded.warnings, [](const std::string &warning) {
    return warning.find("removed associated global_inv") != std::string::npos;
  }));

  options.fault_require_exactly_one = true;
  const ConSanResult guarded = try_patch_consan(bytes, options);
  EXPECT_EQ(guarded.outcome, ConSanTransformOutcome::Invalid);
  EXPECT_EQ(guarded.requested_fault_mutations, 2u);
  EXPECT_EQ(guarded.applied_fault_mutations, 2u);
}

TEST(ConSan, BarrierMutationQualificationSeparatesHostLiveAndTopologyEvidence) {
  const auto scope_crossing = consan_barrier_mutation_qualification(
      "gfx1201", ConSanBarrierMutationForm::PairScopeCrossing);
  EXPECT_EQ(scope_crossing.host, ConSanQualificationEvidence::Proven);
  EXPECT_EQ(scope_crossing.live_gpu, ConSanQualificationEvidence::None);
  EXPECT_EQ(scope_crossing.cluster_or_multi_device, ConSanQualificationEvidence::None);

  const auto completing_id =
      consan_barrier_mutation_qualification("gfx1201", ConSanBarrierMutationForm::PairCompletingId);
  EXPECT_EQ(completing_id.host, ConSanQualificationEvidence::None);
  EXPECT_EQ(completing_id.live_gpu, ConSanQualificationEvidence::None);
  EXPECT_EQ(completing_id.cluster_or_multi_device, ConSanQualificationEvidence::None);

  const auto lifecycle = consan_barrier_mutation_qualification(
      "gfx1250", ConSanBarrierMutationForm::StaticNamedLifecycle);
  EXPECT_EQ(lifecycle.host, ConSanQualificationEvidence::Proven);
  EXPECT_EQ(lifecycle.live_gpu, ConSanQualificationEvidence::DeferredA1);
  EXPECT_EQ(lifecycle.cluster_or_multi_device, ConSanQualificationEvidence::None);

  const auto unknown = consan_barrier_mutation_qualification(
      "gfx9999", ConSanBarrierMutationForm::PairScopeCrossing);
  EXPECT_EQ(unknown.host, ConSanQualificationEvidence::None);
  EXPECT_EQ(unknown.live_gpu, ConSanQualificationEvidence::None);
  EXPECT_EQ(unknown.cluster_or_multi_device, ConSanQualificationEvidence::None);
}

TEST(ConSan, FinalValidationRejectsUnprovenBarrierMutation) {
  const std::array<uint32_t, 2> text_words = {
      0xBF940000u, // s_barrier_wait
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.fault_drop_barrier = true;
  const ConSanResult valid = try_patch_consan(bytes, options);
  ASSERT_EQ(valid.outcome, ConSanTransformOutcome::ModifiedValid);
  ASSERT_EQ(valid.text_sections.size(), 1u);

  ConSanResult corrupted = valid;
  std::memcpy(corrupted.elf_bytes.data() + valid.text_sections.front().file_offset,
              text_words.data(), sizeof(uint32_t));
  const std::vector<std::string> errors = validate_consan_modified_elf(bytes, corrupted);

  EXPECT_TRUE(std::ranges::any_of(errors, [](const std::string &error) {
    return error.find("mutation proof did not replace the selected barrier") != std::string::npos;
  }));
}

TEST(ConSan, FinalValidationRejectsWrongAtomicMutationDisplacement) {
  const std::vector<uint8_t> bytes = make_rdna4_flat_atomic_release_acquire_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.fault_atomic_wrong_address = true;
  const ConSanResult valid = try_patch_consan(bytes, options);
  ASSERT_EQ(valid.outcome, ConSanTransformOutcome::ModifiedValid);
  ASSERT_EQ(valid.patches.size(), 1u);
  ASSERT_EQ(valid.text_sections.size(), 1u);

  ConSanResult corrupted = valid;
  const size_t word2_file_offset = valid.text_sections.front().file_offset +
                                   valid.patches.front().anchor_offset + 2 * sizeof(uint32_t);
  uint32_t word2 = 0;
  std::memcpy(&word2, corrupted.elf_bytes.data() + word2_file_offset, sizeof(word2));
  word2 = (word2 & 0xffu) | (2u << 8u);
  std::memcpy(corrupted.elf_bytes.data() + word2_file_offset, &word2, sizeof(word2));
  const std::vector<std::string> errors = validate_consan_modified_elf(bytes, corrupted);

  EXPECT_TRUE(std::ranges::any_of(errors, [](const std::string &error) {
    return error.find("mutation proof found the wrong atomic address displacement") !=
           std::string::npos;
  }));
}

TEST(ConSan, FinalValidationRejectsScopeMutationThatChangesTh) {
  const std::vector<uint8_t> bytes = make_rdna4_ordered_flat_atomic_release_acquire_code_object();
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_track_atomics = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.max_patches = 8;
  options.fault_atomic_weaken_scope = true;
  options.fault_atomic_index = 0;
  const ConSanResult valid = try_patch_consan(bytes, options);
  ASSERT_EQ(valid.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_EQ(valid.requested_fault_mutations, 1u);
  EXPECT_EQ(valid.planned_fault_mutations, 1u);
  EXPECT_EQ(valid.applied_fault_mutations, 1u);
  ASSERT_EQ(valid.fault_plans.size(), 1u);
  EXPECT_EQ(valid.fault_plans.front().kind, ConSanFaultMutationKind::AtomicWeakenScope);
  ASSERT_EQ(valid.text_sections.size(), 1u);
  const auto scope_patch = std::ranges::find_if(valid.patches, [](const ConSanPatchInfo &patch) {
    return patch.phase == ConSanPatchPhase::Mutation &&
           patch.kind == ConSanPatchKind::InlineAtomicScopeRewrite;
  });
  ASSERT_NE(scope_patch, valid.patches.end());

  ConSanResult corrupted = valid;
  const size_t word1_file_offset =
      valid.text_sections.front().file_offset + scope_patch->anchor_offset + sizeof(uint32_t);
  uint32_t word1 = 0;
  std::memcpy(&word1, corrupted.elf_bytes.data() + word1_file_offset, sizeof(word1));
  word1 ^= 1u << 16u;
  std::memcpy(corrupted.elf_bytes.data() + word1_file_offset, &word1, sizeof(word1));
  const std::vector<std::string> errors = validate_consan_modified_elf(bytes, corrupted);

  EXPECT_TRUE(std::ranges::any_of(errors, [](const std::string &error) {
    return error.find("changed fields other than the selected atomic scope") != std::string::npos;
  }));
}

TEST(ConSan, FinalValidationRejectsCorruptedDsAtomicAddressMutation) {
  const std::vector<uint8_t> bytes = make_rdna4_ds_atomic_code_object();
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.fault_atomic_wrong_address = true;
  options.fault_atomic_address_delta = 4;
  const ConSanResult valid = try_patch_consan(bytes, options);
  ASSERT_EQ(valid.outcome, ConSanTransformOutcome::ModifiedValid);
  ASSERT_EQ(valid.patches.size(), 1u);

  ConSanResult corrupted = valid;
  const size_t word1_file_offset = valid.text_sections.front().file_offset + sizeof(uint32_t);
  uint32_t word1 = 0;
  std::memcpy(&word1, corrupted.elf_bytes.data() + word1_file_offset, sizeof(word1));
  word1 ^= 1u;
  std::memcpy(corrupted.elf_bytes.data() + word1_file_offset, &word1, sizeof(word1));
  const std::vector<std::string> errors = validate_consan_modified_elf(bytes, corrupted);

  EXPECT_TRUE(std::ranges::any_of(errors, [](const std::string &error) {
    return error.find("wrong atomic address displacement") != std::string::npos;
  }));
}

} // namespace
} // namespace rocjitsu
