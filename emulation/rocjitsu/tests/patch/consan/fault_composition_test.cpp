// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"

namespace rocjitsu {
namespace {

TEST(ConSan, PerturbationPlansStableBarrierReleaseAndAcquireEdges) {
  const std::array<uint32_t, 3> text_words = {
      0xBE804EC1u, // s_barrier_signal -1
      0xBF94FFFFu, // s_barrier_wait -1
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions release_options;
  release_options.flavor = ConSanFlavor::SuperCollider;
  release_options.sc_perturb_kind = ConSanPerturbationKind::Barrier;
  release_options.sc_perturb_edge = ConSanPerturbationEdge::Release;
  release_options.sc_perturb_sleep = 7;
  release_options.sc_perturb_required_count = 1;
  release_options.fault_dry_run = true;
  const ConSanResult release = try_patch_consan(bytes, release_options);

  ASSERT_TRUE(release.errors.empty()) << (release.errors.empty() ? "" : release.errors.front());
  ASSERT_EQ(release.perturbation_candidates.size(), 2u);
  ASSERT_EQ(release.perturbation_plans.size(), 1u);
  const ConSanPerturbationCandidate &release_candidate = release.perturbation_candidates[0];
  EXPECT_TRUE(release_candidate.eligible);
  EXPECT_EQ(release_candidate.edge, ConSanPerturbationEdge::Release);
  ASSERT_EQ(release_candidate.ordered_member_identities.size(), 2u);
  EXPECT_EQ(release_candidate.anchor_event_identity,
            release_candidate.ordered_member_identities.front());
  EXPECT_EQ(release.perturbation_plans[0].candidate_identity, release_candidate.identity);
  EXPECT_EQ(release.perturbation_plans[0].anchor_text_offset, 0u);
  EXPECT_EQ(release.perturbation_plans[0].sleep_imm, 7u);
  EXPECT_FALSE(release.modified);
  EXPECT_TRUE(release.elf_bytes.empty());

  ConSanOptions acquire_options = release_options;
  acquire_options.sc_perturb_edge = ConSanPerturbationEdge::Acquire;
  const ConSanResult acquire = try_patch_consan(bytes, acquire_options);
  ASSERT_TRUE(acquire.errors.empty()) << (acquire.errors.empty() ? "" : acquire.errors.front());
  ASSERT_EQ(acquire.perturbation_plans.size(), 1u);
  const ConSanPerturbationCandidate &acquire_candidate = acquire.perturbation_candidates[1];
  EXPECT_TRUE(acquire_candidate.eligible);
  EXPECT_EQ(acquire_candidate.anchor_event_identity,
            acquire_candidate.ordered_member_identities.back());
  EXPECT_EQ(acquire.perturbation_plans[0].anchor_text_offset, 4u);

  ConSanOptions identity_options = acquire_options;
  identity_options.sc_perturb_identity = acquire_candidate.identity;
  identity_options.sc_perturb_index = 99;
  const ConSanResult identity_selected = try_patch_consan(bytes, identity_options);
  ASSERT_TRUE(identity_selected.errors.empty())
      << (identity_selected.errors.empty() ? "" : identity_selected.errors.front());
  ASSERT_EQ(identity_selected.perturbation_plans.size(), 1u);
  EXPECT_EQ(identity_selected.perturbation_plans[0].candidate_identity, acquire_candidate.identity);
  const ConSanResult repeated = try_patch_consan(bytes, acquire_options);
  ASSERT_EQ(repeated.perturbation_plans.size(), 1u);
  EXPECT_EQ(repeated.perturbation_plans[0].candidate_identity,
            acquire.perturbation_plans[0].candidate_identity);
}

TEST(ConSan, PerturbationPlansOrderedAtomicOuterEdgesOnly) {
  const auto wait_store = build_s_wait_storecnt0(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(wait_store);
  const std::vector<uint32_t> text_words = {
      0xEE0B0000u, 0x00000000u, 0x00000000u, // global_wb
      *wait_store, 0xEE158004u, 0x00980000u,
      0x00000002u,                                        // global_atomic_add_f32 scope:device
      *wait_store, 0xEE0AC000u, 0x00000000u, 0x00000000u, // global_inv
      0xBFB00000u,                                        // s_endpgm
  };
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.sc_perturb_kind = ConSanPerturbationKind::Atomic;
  options.sc_perturb_edge = ConSanPerturbationEdge::Release;
  options.sc_perturb_required_count = 1;
  options.fault_dry_run = true;
  const ConSanResult release = try_patch_consan(make_rdna4_lds_code_object(text_words), options);

  ASSERT_TRUE(release.errors.empty()) << (release.errors.empty() ? "" : release.errors.front());
  ASSERT_EQ(release.perturbation_plans.size(), 1u);
  ASSERT_EQ(release.sync_sequences.size(), 1u);
  EXPECT_EQ(release.perturbation_plans[0].anchor_event_identity,
            release.sync_sequences[0].member_event_identities.front());
  EXPECT_EQ(release.perturbation_plans[0].anchor_text_offset, 0u);

  options.sc_perturb_edge = ConSanPerturbationEdge::Acquire;
  const ConSanResult acquire = try_patch_consan(make_rdna4_lds_code_object(text_words), options);
  ASSERT_TRUE(acquire.errors.empty()) << (acquire.errors.empty() ? "" : acquire.errors.front());
  ASSERT_EQ(acquire.perturbation_plans.size(), 1u);
  EXPECT_EQ(acquire.perturbation_plans[0].anchor_event_identity,
            acquire.sync_sequences[0].member_event_identities.back());
  EXPECT_EQ(acquire.perturbation_plans[0].anchor_text_offset, 32u);
}

TEST(ConSan, PerturbationEmissionOrdersBarrierSleepAtSelectedEdge) {
  const std::array<uint32_t, 3> text_words = {
      0xBE804EC1u, // s_barrier_signal -1
      0xBF94FFFFu, // s_barrier_wait -1
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.sc_perturb_kind = ConSanPerturbationKind::Barrier;
  options.sc_perturb_edge = ConSanPerturbationEdge::Release;
  options.sc_perturb_sleep = 7;
  options.sc_perturb_required_count = 1;

  const ConSanResult release = try_patch_consan(bytes, options);
  ASSERT_TRUE(release.errors.empty()) << (release.errors.empty() ? "" : release.errors.front());
  EXPECT_EQ(release.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_TRUE(release.modified);
  EXPECT_TRUE(release.final_validation_passed);
  EXPECT_EQ(release.planned_perturbations, 1u);
  EXPECT_EQ(release.applied_perturbations, 1u);
  ASSERT_EQ(release.patches.size(), 1u);
  const ConSanPatchInfo &release_patch = release.patches.front();
  EXPECT_EQ(release_patch.kind, ConSanPatchKind::TrampolineScPerturbation);
  EXPECT_EQ(release_patch.anchor_offset, 0u);
  EXPECT_EQ(release_patch.trampoline_offset, 12u);
  EXPECT_EQ(release_patch.original_size, 4u);
  EXPECT_EQ(release_patch.trampoline_size, 12u);
  ASSERT_TRUE(release_patch.perturbation_edge);
  EXPECT_EQ(*release_patch.perturbation_edge, ConSanPerturbationEdge::Release);
  EXPECT_FALSE(release_patch.perturbation_sequence_identity.empty());
  EXPECT_FALSE(release_patch.scratch_vgpr);
  EXPECT_EQ(release_patch.required_private_segment_size, 0u);
  ASSERT_EQ(release_patch.owner_descriptor_file_offsets.size(), 1u);
  ASSERT_EQ(release.kernels.size(), 1u);
  EXPECT_EQ(release_patch.owner_descriptor_file_offsets.front(),
            release.kernels.front().descriptor_file_offset);

  AmdGpuCodeObject release_object(release.elf_bytes.data(), release.elf_bytes.size());
  ASSERT_EQ(release_object.text_sections().size(), 1u);
  std::array<uint32_t, 6> release_words{};
  std::memcpy(release_words.data(), release_object.text_sections().front()->data(),
              release_words.size() * sizeof(uint32_t));
  const std::array<uint32_t, 6> expected_release = {
      build_s_branch(2, ROCJITSU_CODE_ARCH_RDNA4),
      0xBF94FFFFu,
      0xBFB00000u,
      build_s_sleep(7, ROCJITSU_CODE_ARCH_RDNA4),
      0xBE804EC1u,
      build_s_branch(-5, ROCJITSU_CODE_ARCH_RDNA4),
  };
  EXPECT_EQ(release_words, expected_release);

  options.sc_perturb_edge = ConSanPerturbationEdge::Acquire;
  const ConSanResult acquire = try_patch_consan(bytes, options);
  ASSERT_TRUE(acquire.errors.empty()) << (acquire.errors.empty() ? "" : acquire.errors.front());
  ASSERT_EQ(acquire.patches.size(), 1u);
  EXPECT_EQ(acquire.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_EQ(acquire.patches.front().anchor_offset, 4u);
  ASSERT_TRUE(acquire.patches.front().perturbation_edge);
  EXPECT_EQ(*acquire.patches.front().perturbation_edge, ConSanPerturbationEdge::Acquire);
  AmdGpuCodeObject acquire_object(acquire.elf_bytes.data(), acquire.elf_bytes.size());
  ASSERT_EQ(acquire_object.text_sections().size(), 1u);
  std::array<uint32_t, 6> acquire_words{};
  std::memcpy(acquire_words.data(), acquire_object.text_sections().front()->data(),
              acquire_words.size() * sizeof(uint32_t));
  const std::array<uint32_t, 6> expected_acquire = {
      0xBE804EC1u,
      build_s_branch(1, ROCJITSU_CODE_ARCH_RDNA4),
      0xBFB00000u,
      0xBF94FFFFu,
      build_s_sleep(7, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_branch(-4, ROCJITSU_CODE_ARCH_RDNA4),
  };
  EXPECT_EQ(acquire_words, expected_acquire);
}

TEST(ConSan, PerturbationEmissionRelocatesTwelveAndEightByteAtomicEdges) {
  const auto wait_store = build_s_wait_storecnt0(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(wait_store);
  const std::vector<uint32_t> release_words = {
      0xEE0B0000u, 0x00000000u, 0x00000000u, // global_wb
      *wait_store, 0xEE158004u, 0x00980000u,
      0x00000002u, // global_atomic_add_f32 scope:device
      0xBFB00000u,
  };
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.sc_perturb_kind = ConSanPerturbationKind::Atomic;
  options.sc_perturb_edge = ConSanPerturbationEdge::Release;
  options.sc_perturb_sleep = 3;
  options.sc_perturb_required_count = 1;
  const ConSanResult release = try_patch_consan(make_rdna4_lds_code_object(release_words), options);
  ASSERT_TRUE(release.errors.empty()) << (release.errors.empty() ? "" : release.errors.front());
  ASSERT_EQ(release.patches.size(), 1u);
  EXPECT_EQ(release.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_EQ(release.patches.front().anchor_offset, 0u);
  EXPECT_EQ(release.patches.front().original_size, 12u);
  EXPECT_EQ(release.patches.front().trampoline_size, 20u);
  AmdGpuCodeObject release_object(release.elf_bytes.data(), release.elf_bytes.size());
  const Section *release_text = release_object.text_sections().front();
  std::array<uint32_t, 4> release_body{};
  std::memcpy(release_body.data(), release_text->data() + release.patches.front().trampoline_offset,
              release_body.size() * sizeof(uint32_t));
  EXPECT_EQ(release_body[0], build_s_sleep(3, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(release_body[1], 0xEE0B0000u);
  EXPECT_EQ(release_body[2], 0x00000000u);
  EXPECT_EQ(release_body[3], 0x00000000u);

  const std::vector<uint32_t> acquire_words = {
      0xEE158004u, 0x00980000u, 0x00000002u, // global_atomic_add_f32 scope:device
      *wait_store, 0xF4042000u, 0x00000000u, // s_dcache_inv
      0xBFB00000u,
  };
  options.sc_perturb_edge = ConSanPerturbationEdge::Acquire;
  const ConSanResult acquire = try_patch_consan(make_rdna4_lds_code_object(acquire_words), options);
  ASSERT_TRUE(acquire.errors.empty()) << (acquire.errors.empty() ? "" : acquire.errors.front());
  ASSERT_EQ(acquire.patches.size(), 1u);
  EXPECT_EQ(acquire.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_EQ(acquire.patches.front().anchor_offset, 16u);
  EXPECT_EQ(acquire.patches.front().original_size, 8u);
  EXPECT_EQ(acquire.patches.front().trampoline_size, 16u);
  AmdGpuCodeObject acquire_object(acquire.elf_bytes.data(), acquire.elf_bytes.size());
  const Section *acquire_text = acquire_object.text_sections().front();
  std::array<uint32_t, 3> acquire_body{};
  std::memcpy(acquire_body.data(), acquire_text->data() + acquire.patches.front().trampoline_offset,
              acquire_body.size() * sizeof(uint32_t));
  EXPECT_EQ(acquire_body[0], 0xF4042000u);
  EXPECT_EQ(acquire_body[1], 0x00000000u);
  EXPECT_EQ(acquire_body[2], build_s_sleep(3, ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(ConSan, AtomicCompositeDryRunExportsOnlyValidatedAccessAndProof) {
  const auto wait_store = build_s_wait_storecnt0(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(wait_store);
  const std::vector<uint32_t> words = {
      0xEE0B0000u, 0u,          0u,              // release cache operation
      *wait_store, 0xEE158004u, 0x00980000u, 2u, // returning global atomic
      0xD8D80000u, 0u,                           // selectable LDS load
      0xBFB00000u,
  };
  const auto bytes = make_rdna4_lds_code_object(words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.fault_dry_run = true;
  options.fault_atomic_weaken_order = true;
  options.fault_require_exactly_one = true;
  options.sc_perturb_kind = ConSanPerturbationKind::Atomic;
  options.sc_perturb_edge = ConSanPerturbationEdge::Release;
  options.sc_perturb_required_count = 1;
  options.probe_lds_check_trap = true;
  options.max_patches = 2;
  ConSanOptions live_options = options;
  live_options.fault_dry_run = false;
  const ConSanResult live = try_patch_consan(bytes, live_options);
  ASSERT_EQ(live.outcome, ConSanTransformOutcome::ModifiedValid)
      << testing::PrintToString(live.errors) << testing::PrintToString(live.warnings);
  const ConSanResult result = try_patch_consan(bytes, options);
  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_FALSE(result.modified);
  ASSERT_EQ(result.access_plans.size(), 1u);
  EXPECT_EQ(result.access_plans.front().kind, "lds-check-trap");
  EXPECT_TRUE(result.access_plans.front().in_kernel);
  ASSERT_TRUE(result.composite_proof);
  const ConSanCompositeProof &proof = *result.composite_proof;
  EXPECT_TRUE(proof.atomic_overlap);
  EXPECT_TRUE(proof.removed_cache_boundary);
  EXPECT_TRUE(proof.removed_cache_non_resurrection_applicable);
  EXPECT_TRUE(proof.removed_cache_non_resurrection_validated);
  EXPECT_EQ(proof.anchor_relation, "removed-cache-boundary");
  EXPECT_NE(proof.cache_companion_identity, "-");
  ASSERT_TRUE(proof.atomic_mutation_anchor_text_offset);
  EXPECT_EQ(*proof.atomic_mutation_anchor_text_offset, proof.translated_anchor_text_offset);
}

TEST(ConSan, AtomicScopeCompositeDryRunExportsValidatedAccessAndProof) {
  const auto wait_store = build_s_wait_storecnt0(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(wait_store);
  const std::vector<uint32_t> words = {
      0xEE0B0000u, 0u,          0u,              // release cache operation
      *wait_store, 0xEE158004u, 0x00980000u, 2u, // returning global atomic, device scope
      0xD8D80000u, 0u,                           // selectable LDS load
      0xBFB00000u,
  };
  const auto bytes = make_rdna4_lds_code_object(words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.fault_dry_run = true;
  options.fault_atomic_weaken_scope = true;
  options.fault_require_exactly_one = true;
  options.sc_perturb_kind = ConSanPerturbationKind::Atomic;
  options.sc_perturb_edge = ConSanPerturbationEdge::Release;
  options.sc_perturb_required_count = 1;
  options.probe_lds_check_trap = true;
  options.max_patches = 2;
  ConSanOptions live_options = options;
  live_options.fault_dry_run = false;
  const ConSanResult live = try_patch_consan(bytes, live_options);
  ASSERT_EQ(live.outcome, ConSanTransformOutcome::ModifiedValid)
      << testing::PrintToString(live.errors) << testing::PrintToString(live.warnings);
  const ConSanResult result = try_patch_consan(bytes, options);
  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.access_plans.size(), 1u) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.composite_proof) << testing::PrintToString(result.warnings);
  EXPECT_FALSE(result.composite_proof->atomic_overlap);
  EXPECT_FALSE(result.composite_proof->removed_cache_boundary);
  EXPECT_EQ(result.composite_proof->anchor_relation, "unchanged");
  ASSERT_TRUE(result.composite_proof->atomic_mutation_anchor_text_offset);
  EXPECT_EQ(*result.composite_proof->atomic_mutation_anchor_text_offset, 4u * sizeof(uint32_t));
}

TEST(ConSan, PerturbationEmissionPrefersLocalCaveAndReservesMultipleAppendedBodies) {
  const std::array<uint32_t, 3> kernel_words = {
      0xBE804EC1u,
      0xBF94FFFFu,
      0xBFB00000u,
  };
  const std::array<uint32_t, 1> function_words = {0xBFB00000u};
  const std::array<uint32_t, 3> tail_words = {
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint8_t> local_bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words, tail_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.sc_perturb_kind = ConSanPerturbationKind::Barrier;
  options.sc_perturb_edge = ConSanPerturbationEdge::Release;
  options.sc_perturb_required_count = 1;
  const ConSanResult local = try_patch_consan(local_bytes, options);
  ASSERT_TRUE(local.errors.empty()) << (local.errors.empty() ? "" : local.errors.front());
  ASSERT_EQ(local.patches.size(), 1u);
  EXPECT_EQ(local.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_EQ(local.patches.front().trampoline_offset, 16u);
  EXPECT_EQ(local.elf_bytes.size(), local_bytes.size());

  const std::array<uint32_t, 5> two_pair_words = {
      0xBE804EC3u, 0xBF94FFFDu, 0xBE804E81u, 0xBF940001u, 0xBFB00000u,
  };
  options.sc_perturb_max = 2;
  options.sc_perturb_required_count = 2;
  options.max_patches = 2;
  const ConSanResult multiple =
      try_patch_consan(make_rdna4_lds_code_object(two_pair_words), options);
  ASSERT_TRUE(multiple.errors.empty()) << (multiple.errors.empty() ? "" : multiple.errors.front());
  EXPECT_EQ(multiple.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_EQ(multiple.applied_perturbations, 2u);
  ASSERT_EQ(multiple.patches.size(), 2u);
  EXPECT_EQ(multiple.patches[0].trampoline_offset, 20u);
  EXPECT_EQ(multiple.patches[1].trampoline_offset, 32u);
  EXPECT_EQ(multiple.patches[0].trampoline_size, 12u);
  EXPECT_EQ(multiple.patches[1].trampoline_size, 12u);
}

TEST(ConSan, PerturbationEmissionFailsClosedWithoutReachableCaveAndRollsBackComposition) {
  std::vector<uint32_t> text_words(40000u, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words[0] = 0xBE804EC1u;
  text_words[1] = 0xBF94FFFFu;
  text_words.back() = 0xBFB00000u;
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words, "sc_perturb_far");
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.sc_perturb_kind = ConSanPerturbationKind::Barrier;
  options.sc_perturb_edge = ConSanPerturbationEdge::Release;
  options.sc_perturb_required_count = 1;
  const ConSanResult unreachable = try_patch_consan(bytes, options);
  EXPECT_EQ(unreachable.outcome, ConSanTransformOutcome::Unsupported);
  EXPECT_FALSE(unreachable.modified);
  EXPECT_TRUE(unreachable.elf_bytes.empty());
  EXPECT_TRUE(unreachable.errors.empty());
  EXPECT_EQ(unreachable.planned_perturbations, 1u);
  EXPECT_EQ(unreachable.applied_perturbations, 0u);
  EXPECT_TRUE(std::ranges::any_of(unreachable.warnings, [](const std::string &warning) {
    return warning.find("no reachable local or appended cave") != std::string::npos;
  }));

  options.fault_drop_barrier = true;
  const ConSanResult composed = try_patch_consan(bytes, options);
  EXPECT_EQ(composed.outcome, ConSanTransformOutcome::Unsupported);
  EXPECT_FALSE(composed.modified);
  EXPECT_TRUE(composed.elf_bytes.empty());
  EXPECT_TRUE(std::ranges::any_of(composed.warnings, [](const std::string &warning) {
    return warning.find("composite planning did not select exact fault and perturbation plans") !=
           std::string::npos;
  })) << (composed.warnings.empty() ? "no warning" : composed.warnings.back());
}

TEST(ConSan, BarrierCompositeRollsBackWhenDropDestroysSelectedEdge) {
  const std::array<uint32_t, 3> text_words = {
      0xBE804EC1u, // s_barrier_signal -1
      0xBF94FFFFu, // s_barrier_wait -1
      0xBFB00000u,
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  ASSERT_EQ(inventory.fault_sites.size(), 2u);
  const auto perturb = std::ranges::find_if(
      inventory.perturbation_candidates, [](const ConSanPerturbationCandidate &candidate) {
        return candidate.eligible && candidate.kind == ConSanPerturbationKind::Barrier &&
               candidate.edge == ConSanPerturbationEdge::Release;
      });
  ASSERT_NE(perturb, inventory.perturbation_candidates.end());

  ConSanOptions options = inventory_options;
  options.fault_drop_barrier = true;
  options.fault_allow_destructive_incomplete_barrier_drop = true;
  options.fault_require_exactly_one = true;
  options.fault_site_identity = inventory.fault_sites.front().identity;
  options.sc_perturb_kind = ConSanPerturbationKind::Barrier;
  options.sc_perturb_edge = ConSanPerturbationEdge::Release;
  options.sc_perturb_identity = perturb->identity;
  options.sc_perturb_required_count = 1;
  const ConSanResult result = try_patch_consan(bytes, options);

  EXPECT_EQ(result.outcome, ConSanTransformOutcome::Unsupported);
  EXPECT_FALSE(result.modified);
  EXPECT_FALSE(result.final_validation_passed);
  EXPECT_FALSE(result.staged_composition_validated);
  EXPECT_TRUE(result.elf_bytes.empty());
  EXPECT_TRUE(result.patches.empty());
  EXPECT_EQ(result.applied_fault_mutations, 0u);
  EXPECT_EQ(result.applied_perturbations, 0u);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("selected perturbation edge was destroyed") != std::string::npos;
  }));
}

TEST(ConSan, FinalValidationRejectsCorruptedBarrierCompositeIdentityOwnershipAndBytes) {
  const std::array<uint32_t, 7> text_words = {
      0xD8340000u, 0x00000000u, // earlier ds_store_b32 destination
      0xBE804EC1u,              // s_barrier_signal -1
      0xBF94FFFFu,              // s_barrier_wait -1
      0xD8D80000u, 0x00000000u, // later ds_load_b32 destination
      0xBFB00000u,
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  const auto destination = std::ranges::find(inventory.barrier_move_destinations, 0u,
                                             &ConSanBarrierMoveDestination::text_offset);
  const auto perturb = std::ranges::find_if(
      inventory.perturbation_candidates, [](const ConSanPerturbationCandidate &candidate) {
        return candidate.eligible && candidate.kind == ConSanPerturbationKind::Barrier &&
               candidate.edge == ConSanPerturbationEdge::Release;
      });
  ASSERT_NE(destination, inventory.barrier_move_destinations.end());
  ASSERT_NE(perturb, inventory.perturbation_candidates.end());

  ConSanOptions options = inventory_options;
  options.fault_move_barrier = true;
  options.fault_require_exactly_one = true;
  options.fault_site_identity = inventory.fault_sites.front().identity;
  options.fault_barrier_move_direction = ConSanBarrierMoveDirection::Earlier;
  options.fault_barrier_destination_identity = destination->identity;
  options.sc_perturb_kind = ConSanPerturbationKind::Barrier;
  options.sc_perturb_edge = ConSanPerturbationEdge::Release;
  options.sc_perturb_identity = perturb->identity;
  options.sc_perturb_required_count = 1;
  const ConSanResult valid = try_patch_consan(bytes, options);
  ASSERT_EQ(valid.outcome, ConSanTransformOutcome::ModifiedValid)
      << testing::PrintToString(valid.errors);
  ASSERT_EQ(valid.patches.size(), 4u);
  AmdGpuCodeObject replacement(valid.elf_bytes.data(), valid.elf_bytes.size());
  ASSERT_EQ(replacement.text_sections().size(), 1u);
  const uint64_t text_file_offset = replacement.text_sections().front()->sectionOffset();

  const auto expect_rejected = [&](ConSanResult corrupted, std::string_view expected_error) {
    const std::vector<std::string> errors = validate_consan_modified_elf(bytes, corrupted);
    EXPECT_TRUE(std::ranges::any_of(
        errors,
        [&](const std::string &error) { return error.find(expected_error) != std::string::npos; }))
        << expected_error << testing::PrintToString(errors);
  };

  ConSanResult stale_candidate = valid;
  stale_candidate.patches.back().perturbation_source_candidate_identity += "|stale";
  expect_rejected(std::move(stale_candidate), "match one pristine admitted sequence edge");

  ConSanResult wrong_owner = valid;
  wrong_owner.patches.back().perturbation_source_container_name += "_other";
  expect_rejected(std::move(wrong_owner), "match one pristine admitted sequence edge");

  ConSanResult wrong_source_anchor = valid;
  wrong_source_anchor.patches.back().perturbation_source_anchor_offset += sizeof(uint32_t);
  expect_rejected(std::move(wrong_source_anchor), "rederived outer sequence member");

  ConSanResult missing_stage_proof = valid;
  missing_stage_proof.staged_composition_validated = false;
  expect_rejected(std::move(missing_stage_proof), "out-of-range patch");

  ConSanResult corrupted_sleep = valid;
  const ConSanPatchInfo &patch = corrupted_sleep.patches.back();
  const uint32_t nop = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
  std::memcpy(corrupted_sleep.elf_bytes.data() + text_file_offset + patch.trampoline_offset, &nop,
              sizeof(nop));
  expect_rejected(std::move(corrupted_sleep), "single bounded sleep on the declared side");

  ConSanResult corrupted_mutation = valid;
  std::memcpy(corrupted_mutation.elf_bytes.data() + text_file_offset +
                  valid.patches.front().anchor_offset,
              &text_words[2], sizeof(uint32_t));
  expect_rejected(std::move(corrupted_mutation), "replace the selected barrier with s_nop 0");
}

TEST(ConSan, FinalValidationProvesPerturbationBytesAndPristineSequenceSemantics) {
  const std::array<uint32_t, 3> text_words = {
      0xBE804EC1u, // s_barrier_signal -1
      0xBF94FFFFu, // s_barrier_wait -1
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.sc_perturb_kind = ConSanPerturbationKind::Barrier;
  options.sc_perturb_edge = ConSanPerturbationEdge::Release;
  options.sc_perturb_sleep = 7;
  options.sc_perturb_required_count = 1;
  const ConSanResult valid = try_patch_consan(bytes, options);
  ASSERT_EQ(valid.outcome, ConSanTransformOutcome::ModifiedValid)
      << (valid.errors.empty() ? "" : valid.errors.front());
  ASSERT_EQ(valid.patches.size(), 1u);
  const ConSanPatchInfo &patch = valid.patches.front();
  AmdGpuCodeObject replacement(valid.elf_bytes.data(), valid.elf_bytes.size());
  const uint64_t text_file_offset = replacement.text_sections().front()->sectionOffset();

  ConSanResult rederived = valid;
  rederived.sync_events.clear();
  rederived.sync_sequences.clear();
  rederived.perturbation_candidates.clear();
  rederived.perturbation_plans.clear();
  EXPECT_TRUE(validate_consan_modified_elf(bytes, rederived).empty());

  const auto expect_rejected = [&](ConSanResult corrupted, std::string_view expected_error) {
    const std::vector<std::string> errors = validate_consan_modified_elf(bytes, corrupted);
    EXPECT_TRUE(std::ranges::any_of(errors, [&](const std::string &error) {
      return error.find(expected_error) != std::string::npos;
    })) << expected_error;
  };
  const auto replace_body_word = [&](uint64_t body_offset, uint32_t word) {
    ConSanResult corrupted = valid;
    std::memcpy(corrupted.elf_bytes.data() + text_file_offset + patch.trampoline_offset +
                    body_offset,
                &word, sizeof(word));
    return corrupted;
  };

  expect_rejected(replace_body_word(sizeof(uint32_t), build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4)),
                  "preserve the exact original boundary bytes");
  expect_rejected(replace_body_word(0u, build_s_sleep(0, ROCJITSU_CODE_ARCH_RDNA4)),
                  "single bounded sleep on the declared side");

  ConSanResult wrong_side = valid;
  const uint32_t original = text_words.front();
  const uint32_t sleep = build_s_sleep(7, ROCJITSU_CODE_ARCH_RDNA4);
  std::memcpy(wrong_side.elf_bytes.data() + text_file_offset + patch.trampoline_offset, &original,
              sizeof(original));
  std::memcpy(wrong_side.elf_bytes.data() + text_file_offset + patch.trampoline_offset +
                  sizeof(uint32_t),
              &sleep, sizeof(sleep));
  expect_rejected(std::move(wrong_side), "single bounded sleep on the declared side");

  ConSanResult stale_identity = valid;
  stale_identity.patches.front().perturbation_sequence_identity += "|stale";
  expect_rejected(std::move(stale_identity), "one pristine admitted sequence edge");

  ConSanResult wrong_count = valid;
  wrong_count.applied_perturbations = 0;
  expect_rejected(std::move(wrong_count), "inconsistent applied patch count");

  ConSanResult duplicate = valid;
  duplicate.patches.push_back(duplicate.patches.front());
  duplicate.applied_perturbations = 2;
  expect_rejected(std::move(duplicate), "duplicate sequence edge");

  expect_rejected(replace_body_word(patch.trampoline_size - sizeof(uint32_t),
                                    build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4)),
                  "invalid exact boundary return");

  ConSanResult out_of_range = valid;
  out_of_range.patches.front().trampoline_offset =
      replacement.text_sections().front()->size() + sizeof(uint32_t);
  expect_rejected(std::move(out_of_range), "out-of-range patch");

  ConSanResult bad_anchor = valid;
  const uint32_t nop = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
  std::memcpy(bad_anchor.elf_bytes.data() + text_file_offset + patch.anchor_offset, &nop,
              sizeof(nop));
  expect_rejected(std::move(bad_anchor), "invalid anchor branch");

  ConSanResult wrong_outer_edge = valid;
  wrong_outer_edge.patches.front().perturbation_edge = ConSanPerturbationEdge::Acquire;
  expect_rejected(std::move(wrong_outer_edge), "rederived outer sequence member");
}

TEST(ConSan, FinalValidationRejectsPerturbationNonNopWideAnchorTail) {
  const auto wait_store = build_s_wait_storecnt0(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(wait_store);
  const std::vector<uint32_t> text_words = {
      0xEE0B0000u, 0x00000000u, 0x00000000u, // global_wb
      *wait_store, 0xEE158004u, 0x00980000u,
      0x00000002u, // global_atomic_add_f32 scope:device
      0xBFB00000u,
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.sc_perturb_kind = ConSanPerturbationKind::Atomic;
  options.sc_perturb_edge = ConSanPerturbationEdge::Release;
  options.sc_perturb_required_count = 1;
  const ConSanResult valid = try_patch_consan(bytes, options);
  ASSERT_EQ(valid.outcome, ConSanTransformOutcome::ModifiedValid)
      << (valid.errors.empty() ? "" : valid.errors.front());
  ASSERT_EQ(valid.patches.size(), 1u);
  ASSERT_EQ(valid.patches.front().original_size, 12u);
  AmdGpuCodeObject replacement(valid.elf_bytes.data(), valid.elf_bytes.size());
  const uint64_t text_file_offset = replacement.text_sections().front()->sectionOffset();
  ConSanResult corrupted = valid;
  const uint32_t non_nop = build_s_nop(1, ROCJITSU_CODE_ARCH_RDNA4);
  std::memcpy(corrupted.elf_bytes.data() + text_file_offset + sizeof(uint32_t), &non_nop,
              sizeof(non_nop));
  const std::vector<std::string> errors = validate_consan_modified_elf(bytes, corrupted);
  EXPECT_TRUE(std::ranges::any_of(errors, [](const std::string &error) {
    return error.find("non-NOP anchor tail") != std::string::npos;
  }));
}

TEST(ConSan, PerturbationCompositionSharesBudgetWithOneRedundantLdsAccess) {
  const std::array<uint32_t, 15> text_words = {
      0xBE804EC1u, // s_barrier_signal -1
      0xBF94FFFFu, // s_barrier_wait -1
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBFB00000u,
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3;
  options.delay_nops = 2;
  options.sc_perturb_kind = ConSanPerturbationKind::Barrier;
  options.sc_perturb_edge = ConSanPerturbationEdge::Release;
  options.sc_perturb_required_count = 1;
  options.max_patches = 2;
  const ConSanResult composed = try_patch_consan(bytes, options);
  ASSERT_TRUE(composed.errors.empty()) << testing::PrintToString(composed.errors);
  EXPECT_EQ(composed.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_TRUE(composed.final_validation_passed);
  EXPECT_EQ(composed.applied_perturbations, 1u);
  ASSERT_EQ(composed.patches.size(), 2u);
  EXPECT_EQ(composed.patches[0].kind, ConSanPatchKind::InlineLdsLoadCheckTrap);
  EXPECT_EQ(composed.patches[0].anchor_offset, 8u);
  EXPECT_EQ(composed.patches[1].kind, ConSanPatchKind::TrampolineScPerturbation);
  EXPECT_EQ(composed.patches[1].anchor_offset, 0u);
  EXPECT_EQ(composed.patches[1].trampoline_offset, text_words.size() * sizeof(uint32_t));
  EXPECT_TRUE(composed.patches[0].anchor_offset + composed.patches[0].original_size <=
                  composed.patches[1].anchor_offset ||
              composed.patches[1].anchor_offset + composed.patches[1].original_size <=
                  composed.patches[0].anchor_offset);

  options.max_patches = 1;
  const ConSanResult capped = try_patch_consan(bytes, options);
  ASSERT_TRUE(capped.errors.empty()) << testing::PrintToString(capped.errors);
  EXPECT_EQ(capped.outcome, ConSanTransformOutcome::ModifiedValid);
  ASSERT_EQ(capped.patches.size(), 1u);
  EXPECT_EQ(capped.patches.front().kind, ConSanPatchKind::TrampolineScPerturbation);
  EXPECT_TRUE(std::ranges::any_of(capped.warnings, [](const std::string &warning) {
    return warning.find("shared patch budget consumed") != std::string::npos;
  })) << testing::PrintToString(capped.warnings);

  options.max_patches_is_expert_limit = false;
  const ConSanResult all_supported = try_patch_consan(bytes, options);
  ASSERT_TRUE(all_supported.errors.empty()) << testing::PrintToString(all_supported.errors);
  EXPECT_EQ(all_supported.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_TRUE(all_supported.final_validation_passed);
  EXPECT_EQ(all_supported.applied_perturbations, 1u);
  ASSERT_EQ(all_supported.patches.size(), 2u);
  EXPECT_EQ(all_supported.patches[0].kind, ConSanPatchKind::InlineLdsLoadCheckTrap);
  EXPECT_EQ(all_supported.patches[1].kind, ConSanPatchKind::TrampolineScPerturbation);
}

TEST(ConSan, PerturbationCompositionReservesLocalCaveAndRollsBackUnreachablePlan) {
  const std::array<uint32_t, 5> kernel_words = {
      0xBE804EC1u, 0xBF94FFFFu, 0xD8D80000u, 0x01000002u, 0xBFB00000u,
  };
  const std::array<uint32_t, 1> function_words = {0xBFB00000u};
  std::array<uint32_t, 20> tail_words{};
  tail_words.fill(build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words, tail_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3;
  options.delay_nops = 2;
  options.sc_perturb_kind = ConSanPerturbationKind::Barrier;
  options.sc_perturb_edge = ConSanPerturbationEdge::Release;
  options.sc_perturb_required_count = 1;
  options.max_patches = 2;
  const ConSanResult local = try_patch_consan(bytes, options);
  ASSERT_TRUE(local.errors.empty()) << testing::PrintToString(local.errors);
  EXPECT_EQ(local.outcome, ConSanTransformOutcome::ModifiedValid);
  ASSERT_EQ(local.patches.size(), 2u);
  EXPECT_EQ(local.patches[0].kind, ConSanPatchKind::LocalCaveLdsLoadCheckTrap);
  EXPECT_EQ(local.patches[1].kind, ConSanPatchKind::TrampolineScPerturbation);
  EXPECT_TRUE(local.patches[0].trampoline_offset + local.patches[0].trampoline_size <=
                  local.patches[1].trampoline_offset ||
              local.patches[1].trampoline_offset + local.patches[1].trampoline_size <=
                  local.patches[0].trampoline_offset);

  std::vector<uint32_t> far_words(40000u, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  far_words[0] = 0xBE804EC1u;
  far_words[1] = 0xBF94FFFFu;
  far_words[2] = 0xD8D80000u;
  far_words[3] = 0x01000002u;
  far_words.back() = 0xBFB00000u;
  const ConSanResult unreachable =
      try_patch_consan(make_rdna4_lds_code_object(far_words, "composed_far"), options);
  EXPECT_EQ(unreachable.outcome, ConSanTransformOutcome::Unsupported);
  EXPECT_FALSE(unreachable.modified);
  EXPECT_TRUE(unreachable.elf_bytes.empty());
  EXPECT_TRUE(unreachable.patches.empty());
  EXPECT_EQ(unreachable.applied_perturbations, 0u);
}

TEST(ConSan, PerturbationCompositionSharesTransactionWithFlatRedundantAccess) {
  const std::array<uint32_t, 3> kernel_words = {
      0xBE804EC1u,
      0xBF94FFFFu,
      0xBFB00000u,
  };
  const std::array<uint32_t, 19> function_words = {
      0xBE8001EBu,                           // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u, 0x00000000u,              // v_mov_b32_e64 v0, s0
      0xD5810001u, 0x00000001u,              // v_mov_b32_e64 v1, s1
      0xEC05007Cu, 0x00000002u, 0x00000000u, // flat_load_b32 v2, v[0:1]
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBFB00000u,
  };
  const std::vector<uint8_t> bytes = make_rdna4_code_object_with_local_function(
      kernel_words, function_words, {}, kRdna4Wave64AllVgprsGranulated,
      /*function_is_kernel=*/true);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_flat_check_trap = true;
  options.flat_provenance_mode = ConSanFlatProvenanceMode::Strict;
  options.scratch_vgpr = 5;
  options.delay_nops = 1;
  options.sc_perturb_kind = ConSanPerturbationKind::Barrier;
  options.sc_perturb_edge = ConSanPerturbationEdge::Acquire;
  options.sc_perturb_required_count = 1;
  options.max_patches = 2;
  const ConSanResult result = try_patch_consan(bytes, options);
  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  ASSERT_EQ(result.patches.size(), 2u);
  EXPECT_EQ(result.patches[0].kind, ConSanPatchKind::InlineFlatLoadCheckTrap);
  EXPECT_EQ(result.patches[0].anchor_offset, 32u);
  EXPECT_EQ(result.patches[1].kind, ConSanPatchKind::TrampolineScPerturbation);
  EXPECT_EQ(result.patches[1].anchor_offset, 4u);
  EXPECT_EQ(result.patches[1].trampoline_offset,
            (kernel_words.size() + function_words.size()) * sizeof(uint32_t));
}

TEST(ConSan, PerturbationEmissionAcceptsReturningOrderedCasReleaseEdge) {
  const auto atomic = build_flat_atomic_cmpswap_b32_vaddr_vsrc_vdst(
      /*vaddr=*/4, /*vsrc=*/1, /*vdst=*/0, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(atomic);
  const std::array<uint32_t, 12> text_words = {
      0xBE8001EBu,                             // s_mov_b64 s[0:1], src_shared_base
      0xD5810004u,  0x00000000u,               // v_mov_b32_e64 v4, s0
      0xD5810005u,  0x00000001u,               // v_mov_b32_e64 v5, s1
      0xEE0B0000u,  0x00000000u,  0x00000000u, // global_wb
      (*atomic)[0], (*atomic)[1], (*atomic)[2], 0xBFB00000u,
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.sc_perturb_kind = ConSanPerturbationKind::Atomic;
  options.sc_perturb_edge = ConSanPerturbationEdge::Release;
  options.sc_perturb_required_count = 1;
  const ConSanResult result = try_patch_consan(bytes, options);
  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::TrampolineScPerturbation);
  EXPECT_EQ(result.patches.front().anchor_offset, 20u);
  EXPECT_EQ(result.patches.front().original_size, 12u);
  ASSERT_EQ(result.sync_sequences.size(), 1u);
  EXPECT_EQ(result.sync_sequences.front().operation, ConSanSyncOperation::AtomicCompareExchange);
  EXPECT_EQ(result.sync_sequences.front().rmw_outcome, ConSanSyncRmwOutcome::CompareExchange);
  EXPECT_EQ(result.sync_sequences.front().address_source, ConSanSyncAddressSource::FlatVector);
  EXPECT_TRUE(consan_sync_confidence_meets(result.sync_sequences.front().confidence,
                                           ConSanSemanticConfidence::Conservative));
}

TEST(ConSan, PerturbationAcceptsExactOrderedCasWithoutStaticFlatProvenance) {
  const std::vector<uint8_t> bytes = make_rdna4_ordered_flat_cas_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.sc_perturb_kind = ConSanPerturbationKind::Atomic;
  options.sc_perturb_edge = ConSanPerturbationEdge::Release;
  options.sc_perturb_required_count = 1;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  ASSERT_EQ(result.sync_events.size(), 2u);
  EXPECT_EQ(result.sync_events[1].operation, ConSanSyncOperation::AtomicCompareExchange);
  EXPECT_EQ(result.sync_events[1].confidence, ConSanSemanticConfidence::Unsupported);
  ASSERT_EQ(result.sync_sequences.size(), 1u);
  EXPECT_EQ(result.sync_sequences.front().address_source, ConSanSyncAddressSource::FlatVector);
  EXPECT_EQ(result.sync_sequences.front().confidence, ConSanSemanticConfidence::Conservative);
  ASSERT_EQ(result.perturbation_plans.size(), 1u);
  EXPECT_EQ(result.perturbation_plans.front().anchor_text_offset, 0u);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::TrampolineScPerturbation);
}

TEST(ConSan, PerturbationHostControlsAreStableExactAndFailClosed) {
  const std::array<uint32_t, 5> two_pair_words = {
      0xBE804EC3u, 0xBF94FFFDu, // cluster barrier -3
      0xBE804E81u, 0xBF940001u, // workgroup barrier 1
      0xBFB00000u,
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(two_pair_words);
  ConSanOptions disabled_options;
  disabled_options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult disabled_first = try_patch_consan(bytes, disabled_options);
  const ConSanResult disabled_second = try_patch_consan(bytes, disabled_options);
  ASSERT_TRUE(disabled_first.errors.empty()) << testing::PrintToString(disabled_first.errors);
  ASSERT_TRUE(disabled_second.errors.empty()) << testing::PrintToString(disabled_second.errors);
  EXPECT_EQ(disabled_first.outcome, ConSanTransformOutcome::Unchanged);
  EXPECT_FALSE(disabled_first.modified);
  EXPECT_TRUE(disabled_first.elf_bytes.empty());
  EXPECT_TRUE(disabled_first.patches.empty());
  EXPECT_EQ(disabled_first.planned_perturbations, 0u);
  EXPECT_EQ(disabled_first.applied_perturbations, 0u);
  ASSERT_EQ(disabled_first.perturbation_candidates.size(),
            disabled_second.perturbation_candidates.size());
  for (size_t i = 0; i < disabled_first.perturbation_candidates.size(); ++i) {
    EXPECT_EQ(disabled_first.perturbation_candidates[i].identity,
              disabled_second.perturbation_candidates[i].identity);
    EXPECT_EQ(disabled_first.perturbation_candidates[i].eligible,
              disabled_second.perturbation_candidates[i].eligible);
  }

  ConSanOptions enabled = disabled_options;
  enabled.sc_perturb_kind = ConSanPerturbationKind::Barrier;
  enabled.sc_perturb_edge = ConSanPerturbationEdge::Release;
  enabled.sc_perturb_required_count = 1;
  enabled.max_patches = 1;
  const ConSanResult one = try_patch_consan(bytes, enabled);
  ASSERT_EQ(one.outcome, ConSanTransformOutcome::ModifiedValid)
      << testing::PrintToString(one.errors);
  EXPECT_EQ(one.planned_perturbations, 1u);
  EXPECT_EQ(one.applied_perturbations, 1u);
  EXPECT_EQ(one.patches.size(), 1u);

  enabled.sc_perturb_max = 2;
  enabled.sc_perturb_required_count = 2;
  enabled.max_patches = 2;
  const ConSanResult two = try_patch_consan(bytes, enabled);
  ASSERT_EQ(two.outcome, ConSanTransformOutcome::ModifiedValid)
      << testing::PrintToString(two.errors);
  EXPECT_EQ(two.planned_perturbations, 2u);
  EXPECT_EQ(two.applied_perturbations, 2u);
  EXPECT_EQ(two.patches.size(), 2u);

  std::vector<uint32_t> far_words(40000u, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  far_words[0] = 0xBE804EC1u;
  far_words[1] = 0xBF94FFFFu;
  far_words.back() = 0xBFB00000u;
  ConSanOptions unreachable_options = disabled_options;
  unreachable_options.sc_perturb_kind = ConSanPerturbationKind::Barrier;
  unreachable_options.sc_perturb_edge = ConSanPerturbationEdge::Release;
  unreachable_options.sc_perturb_required_count = 1;
  const ConSanResult unreachable = try_patch_consan(
      make_rdna4_lds_code_object(far_words, "sc2b1_unreachable"), unreachable_options);
  EXPECT_EQ(unreachable.outcome, ConSanTransformOutcome::Unsupported);
  EXPECT_TRUE(unreachable.errors.empty());
  EXPECT_FALSE(unreachable.modified);
  EXPECT_TRUE(unreachable.elf_bytes.empty());
  EXPECT_TRUE(unreachable.patches.empty());
  EXPECT_FALSE(unreachable.final_validation_passed);
  EXPECT_EQ(unreachable.planned_perturbations, 1u);
  EXPECT_EQ(unreachable.applied_perturbations, 0u);
  EXPECT_TRUE(std::ranges::any_of(unreachable.warnings, [](const std::string &warning) {
    return warning.find("no reachable local or appended cave") != std::string::npos;
  }));
}

TEST(ConSan, PerturbationControlsAreBoundedAndRequiredCountFailsClosed) {
  const std::array<uint32_t, 3> text_words = {
      0xBE804EC1u,
      0xBF94FFFFu,
      0xBFB00000u,
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.sc_perturb_kind = ConSanPerturbationKind::Barrier;
  options.fault_dry_run = true;

  ConSanOptions wrong_flavor = options;
  wrong_flavor.flavor = ConSanFlavor::Moi;
  EXPECT_FALSE(try_patch_consan(bytes, wrong_flavor).errors.empty());

  options.sc_perturb_max = 3;
  EXPECT_FALSE(try_patch_consan(bytes, options).errors.empty());
  options.sc_perturb_max = 1;
  options.sc_perturb_sleep = 0;
  EXPECT_FALSE(try_patch_consan(bytes, options).errors.empty());
  options.sc_perturb_sleep = 16;
  EXPECT_FALSE(try_patch_consan(bytes, options).errors.empty());
  options.sc_perturb_sleep = 1;
  options.sc_perturb_identity = "stale-sequence-identity";
  options.sc_perturb_required_count = 1;
  const ConSanResult stale = try_patch_consan(bytes, options);
  ASSERT_FALSE(stale.errors.empty());
  EXPECT_TRUE(stale.perturbation_plans.empty());
  EXPECT_FALSE(stale.modified);

  const std::array<uint32_t, 5> two_pair_words = {
      0xBE804EC3u, 0xBF94FFFDu, // cluster barrier -3
      0xBE804E81u, 0xBF940001u, // workgroup barrier 1
      0xBFB00000u,
  };
  options.sc_perturb_identity.clear();
  options.sc_perturb_max = 2;
  options.sc_perturb_required_count = 2;
  const ConSanResult two = try_patch_consan(make_rdna4_lds_code_object(two_pair_words), options);
  ASSERT_TRUE(two.errors.empty()) << (two.errors.empty() ? "" : two.errors.front());
  ASSERT_EQ(two.perturbation_plans.size(), 2u);
  EXPECT_NE(two.perturbation_plans[0].candidate_identity,
            two.perturbation_plans[1].candidate_identity);

  options.sc_perturb_index = 1;
  options.sc_perturb_max = 1;
  options.sc_perturb_required_count = 1;
  const ConSanResult indexed =
      try_patch_consan(make_rdna4_lds_code_object(two_pair_words), options);
  ASSERT_TRUE(indexed.errors.empty()) << (indexed.errors.empty() ? "" : indexed.errors.front());
  ASSERT_EQ(indexed.perturbation_plans.size(), 1u);
  EXPECT_EQ(indexed.perturbation_plans[0].candidate_identity,
            two.perturbation_plans[1].candidate_identity);
}

TEST(ConSan, PerturbationRejectsUnpairedDynamicAmbiguousAndCyclicSequences) {
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.sc_perturb_kind = ConSanPerturbationKind::Barrier;
  options.sc_perturb_required_count = 1;

  const std::array<uint32_t, 3> dynamic_words = {
      0xBE804E7Du, // s_barrier_signal m0
      0xBF94FFFFu, // s_barrier_wait -1
      0xBFB00000u,
  };
  const ConSanResult dynamic = try_patch_consan(make_rdna4_lds_code_object(dynamic_words), options);
  EXPECT_TRUE(dynamic.perturbation_plans.empty());
  EXPECT_FALSE(dynamic.errors.empty());

  const std::array<uint32_t, 2> unpaired_words = {
      0xBE804EC1u, // s_barrier_signal -1
      0xBFB00000u,
  };
  const ConSanResult unpaired =
      try_patch_consan(make_rdna4_lds_code_object(unpaired_words), options);
  EXPECT_TRUE(unpaired.perturbation_plans.empty());
  EXPECT_FALSE(unpaired.errors.empty());

  const std::array<uint32_t, 3> runtime_words = {
      0xBE804EC1u,
      0xBF94FFFFu,
      0xBFB00000u,
  };
  const ConSanResult runtime = try_patch_consan(
      make_rdna4_lds_code_object(runtime_words, "__amd_rocclr_runtime_helper"), options);
  EXPECT_TRUE(runtime.perturbation_candidates.empty());
  EXPECT_TRUE(runtime.perturbation_plans.empty());
  EXPECT_FALSE(runtime.errors.empty());

  const std::array<uint32_t, 4> cyclic_words = {
      0xBE804EC1u,
      0xBF94FFFFu,
      build_s_branch(-3, ROCJITSU_CODE_ARCH_RDNA4), // back to signal
      0xBFB00000u,
  };
  const ConSanResult cyclic = try_patch_consan(make_rdna4_lds_code_object(cyclic_words), options);
  ASSERT_FALSE(cyclic.perturbation_candidates.empty());
  EXPECT_TRUE(
      std::ranges::none_of(cyclic.perturbation_candidates, &ConSanPerturbationCandidate::eligible));
  EXPECT_EQ(cyclic.perturbation_candidates.front().rejection_reason, "cyclic-cfg-component");
  EXPECT_TRUE(cyclic.perturbation_plans.empty());
  EXPECT_FALSE(cyclic.errors.empty());

  options.sc_perturb_kind = ConSanPerturbationKind::Atomic;
  const ConSanResult ambiguous = try_patch_consan(make_rdna4_flat_atomic_code_object(), options);
  ASSERT_FALSE(ambiguous.perturbation_candidates.empty());
  EXPECT_TRUE(std::ranges::none_of(ambiguous.perturbation_candidates,
                                   &ConSanPerturbationCandidate::eligible));
  EXPECT_TRUE(ambiguous.perturbation_plans.empty());
  EXPECT_FALSE(ambiguous.errors.empty());
}

TEST(ConSan, PerturbationRejectsClausesUnknownRolesAndWaveScope) {
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.sc_perturb_kind = ConSanPerturbationKind::Atomic;
  options.sc_perturb_edge = ConSanPerturbationEdge::Release;
  options.sc_perturb_required_count = 1;

  const std::array<uint32_t, 8> clause_words = {
      0xBF850000u,                           // s_clause 0
      0xEE0B0000u, 0x00000000u, 0x00000000u, // global_wb
      0xEE158004u, 0x00980000u,
      0x00000002u, // global_atomic_add_f32 scope:device
      0xBFB00000u,
  };
  const ConSanResult clause = try_patch_consan(make_rdna4_lds_code_object(clause_words), options);
  ASSERT_FALSE(clause.perturbation_candidates.empty());
  EXPECT_TRUE(
      std::ranges::none_of(clause.perturbation_candidates, &ConSanPerturbationCandidate::eligible));
  EXPECT_EQ(clause.perturbation_candidates.front().rejection_reason, "inside-s-clause");
  EXPECT_FALSE(clause.errors.empty());

  const ConSanResult unknown_role =
      try_patch_consan(make_rdna4_global_atomic_code_object(), options);
  ASSERT_FALSE(unknown_role.perturbation_candidates.empty());
  EXPECT_TRUE(std::ranges::none_of(unknown_role.perturbation_candidates,
                                   &ConSanPerturbationCandidate::eligible));
  EXPECT_EQ(unknown_role.perturbation_candidates.front().rejection_reason,
            "unknown-or-inapplicable-memory-role");

  const std::array<uint32_t, 7> wave_scope_words = {
      0xEE0B0000u, 0x00000000u, 0x00000000u, // global_wb
      0xEE158004u, 0x00900000u,
      0x00000002u, // global_atomic_add_f32 scope:wave
      0xBFB00000u,
  };
  const ConSanResult wave_scope =
      try_patch_consan(make_rdna4_lds_code_object(wave_scope_words), options);
  ASSERT_EQ(wave_scope.sync_events.size(), 2u);
  const auto atomic = std::ranges::find(wave_scope.sync_events, ConSanSyncEventKind::Atomic,
                                        &ConSanSyncEvent::kind);
  ASSERT_NE(atomic, wave_scope.sync_events.end());
  ASSERT_TRUE(atomic->raw_scope);
  EXPECT_EQ(*atomic->raw_scope, 0u);
  ASSERT_EQ(wave_scope.sync_sequences.size(), 1u);
  ASSERT_TRUE(wave_scope.sync_sequences.front().raw_scope);
  EXPECT_EQ(*wave_scope.sync_sequences.front().raw_scope, 0u);
  ASSERT_FALSE(wave_scope.perturbation_candidates.empty());
  EXPECT_TRUE(std::ranges::none_of(wave_scope.perturbation_candidates,
                                   &ConSanPerturbationCandidate::eligible));
  EXPECT_EQ(wave_scope.perturbation_candidates.front().rejection_reason,
            "unsupported-atomic-scope");
}

TEST(ConSan, FaultCompositionRollsBackMutationWhenInstrumentationIsInvalid) {
  const std::array<uint32_t, 12> text_words = {
      0xD8D80000u, 0x00000000u, // ds_load_b32
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF940000u, // s_barrier_wait
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  const std::vector<uint8_t> original = bytes;
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.fault_drop_barrier = true;
  options.delay_mode = ConSanDelayMode::Sleep;
  options.delay_nops = 65536;

  const ConSanResult result = try_patch_consan(bytes, options);

  EXPECT_EQ(result.outcome, ConSanTransformOutcome::Invalid);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
  EXPECT_TRUE(result.patches.empty());
  ASSERT_FALSE(result.errors.empty());
  EXPECT_NE(result.errors.front().find("16-bit s_sleep"), std::string::npos);
  ASSERT_FALSE(result.warnings.empty());
  EXPECT_NE(result.warnings.back().find("rolled back staged fault mutation"), std::string::npos);
  EXPECT_EQ(result.applied_fault_mutations, 0u);
  EXPECT_EQ(bytes, original);
}

TEST(ConSan, ProbeLdsCheckTrapModeUsesWideCompositeRelayDonor) {
  constexpr size_t kTextWords = 33010u;
  constexpr uint64_t kOriginalTextSize = kTextWords * sizeof(uint32_t);
  constexpr uint64_t kAnchorOffset = 22012u;
  constexpr uint64_t kHostOffset = 132012u;
  const std::array<uint32_t, 2> anchor_original = {0x7E2602FFu,
                                                   0x00000C0Du}; // v_mov_b32_e32 v19, literal 0xc0d
  const std::array<uint32_t, 6> host_original = {
      0xEC05007Cu, 0x00000002u, 0x00000000u,
      0xD581000Bu, 0x00000001u, build_s_mov_b32(100, 100, ROCJITSU_CODE_ARCH_RDNA4)};

  std::vector<uint32_t> text_words(kTextWords, build_s_mov_b32(100, 100, ROCJITSU_CODE_ARCH_RDNA4));
  text_words[0] = 0xD8D80000u;
  text_words[1] = 0x01000002u; // ds_load_b32 v1, v2
  std::ranges::copy(anchor_original,
                    text_words.begin() + static_cast<ptrdiff_t>(kAnchorOffset / sizeof(uint32_t)));
  std::ranges::copy(host_original,
                    text_words.begin() + static_cast<ptrdiff_t>(kHostOffset / sizeof(uint32_t)));
  text_words.back() = 0xBFB00000u; // s_endpgm

  ASSERT_FALSE(compute_sopp_branch_simm16(0u, kOriginalTextSize));
  ASSERT_EQ(kHostOffset - kAnchorOffset, 110000u);
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.max_patches = 1;
  options.scratch_vgpr = 6;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.modified);
  std::vector<const ConSanPatchInfo *> donors;
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.kind == ConSanPatchKind::TrampolineScBranchRelayDonor)
      donors.push_back(&patch);
  }
  ASSERT_EQ(donors.size(), 2u);
  const ConSanPatchInfo *anchor_patch =
      donors[0]->original_size == anchor_original.size() * sizeof(uint32_t) ? donors[0] : donors[1];
  const ConSanPatchInfo *host_patch = anchor_patch == donors[0] ? donors[1] : donors[0];
  EXPECT_EQ(anchor_patch->anchor_offset, kAnchorOffset);
  EXPECT_EQ(anchor_patch->trampoline_offset, kHostOffset + sizeof(uint32_t));
  EXPECT_EQ(anchor_patch->original_size, 2u * sizeof(uint32_t));
  EXPECT_EQ(anchor_patch->trampoline_size, 3u * sizeof(uint32_t));
  EXPECT_EQ(host_patch->anchor_offset, kHostOffset);
  EXPECT_EQ(host_patch->trampoline_offset, kOriginalTextSize);
  EXPECT_EQ(host_patch->original_size, 6u * sizeof(uint32_t));
  EXPECT_EQ(host_patch->trampoline_size, 7u * sizeof(uint32_t));

  const auto island = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineScIndirectBranchIsland, &ConSanPatchInfo::kind);
  ASSERT_NE(island, result.patches.end());
  EXPECT_EQ(island->anchor_offset, 0u);
  EXPECT_EQ(island->trampoline_offset, kOriginalTextSize + 7u * sizeof(uint32_t));

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const Section *text = patched.text_sections().front();
  ASSERT_GE(text->size(), kOriginalTextSize + 7u * sizeof(uint32_t));
  const auto text_bytes =
      std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(text->data()), text->size());
  std::array<uint32_t, 2> patched_anchor{};
  std::array<uint32_t, 6> patched_host{};
  std::array<uint32_t, 7> appended_host{};
  std::memcpy(patched_anchor.data(), text_bytes.data() + kAnchorOffset, sizeof(patched_anchor));
  std::memcpy(patched_host.data(), text_bytes.data() + kHostOffset, sizeof(patched_host));
  std::memcpy(appended_host.data(), text_bytes.data() + kOriginalTextSize, sizeof(appended_host));

  const auto anchor_to_host =
      compute_sopp_branch_simm16(kAnchorOffset, kHostOffset + sizeof(uint32_t));
  const auto host_to_body = compute_sopp_branch_simm16(kHostOffset, kOriginalTextSize);
  const auto anchor_return = compute_sopp_branch_simm16(kHostOffset + 3u * sizeof(uint32_t),
                                                        kAnchorOffset + 2u * sizeof(uint32_t));
  const auto host_return = compute_sopp_branch_simm16(kOriginalTextSize + 6u * sizeof(uint32_t),
                                                      kHostOffset + 6u * sizeof(uint32_t));
  ASSERT_TRUE(anchor_to_host && host_to_body && anchor_return && host_return);
  EXPECT_EQ(patched_anchor[0], build_s_branch(*anchor_to_host, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(patched_host[0], build_s_branch(*host_to_body, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_TRUE(std::ranges::equal(std::span<const uint32_t>(patched_host).subspan(1u, 2u),
                                 std::span<const uint32_t>(anchor_original)));
  EXPECT_EQ(patched_host[3], build_s_branch(*anchor_return, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_TRUE(std::ranges::equal(std::span<const uint32_t>(appended_host).first(6u),
                                 std::span<const uint32_t>(host_original)));
  EXPECT_EQ(appended_host[6], build_s_branch(*host_return, ROCJITSU_CODE_ARCH_RDNA4));

  ConSanResult corrupted = result;
  const uint64_t text_file_offset = text->sectionOffset();
  corrupted.elf_bytes[text_file_offset + kOriginalTextSize + sizeof(uint32_t)] ^= 1u;
  const std::vector<std::string> validation_errors = validate_consan_modified_elf(bytes, corrupted);
  ASSERT_FALSE(validation_errors.empty());
  EXPECT_TRUE(std::ranges::any_of(validation_errors, [](const std::string &error) {
    return error.find("corrupted displaced") != std::string::npos;
  }));
}

} // namespace
} // namespace rocjitsu
