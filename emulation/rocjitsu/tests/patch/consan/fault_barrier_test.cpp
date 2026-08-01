// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"

namespace rocjitsu {
namespace {

TEST(ConSan, BarrierMoveExactHelperIdentityCannotBypassDispatchOwnership) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.helper_has_barrier = true;
  const std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object(fixture);
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  inventory_options.fault_dry_run = true;
  const ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  const auto source = std::ranges::find_if(inventory.fault_sites, [](const ConSanFaultSite &item) {
    return item.kind == ConSanFaultSiteKind::Barrier && item.container_name == "shared_lds_helper";
  });
  ASSERT_NE(source, inventory.fault_sites.end());
  ASSERT_TRUE(source->sync_sequence_identity);
  const auto destination =
      std::ranges::find_if(inventory.barrier_move_destinations, [](const auto &item) {
        return item.container_name == "shared_lds_helper" && item.suitable &&
               item.mnemonic == "ds_store_b32";
      });
  ASSERT_NE(destination, inventory.barrier_move_destinations.end());

  ConSanOptions selected_options = inventory_options;
  selected_options.fault_move_barrier = true;
  selected_options.fault_site_identity = source->identity;
  selected_options.fault_barrier_sequence_identity = *source->sync_sequence_identity;
  selected_options.fault_barrier_move_direction = ConSanBarrierMoveDirection::Earlier;
  selected_options.fault_barrier_destination_identity = destination->identity;
  selected_options.test_kernel_name_filter = "shared_owner_0";
  const ConSanResult selected = try_patch_consan(bytes, selected_options);
  ASSERT_EQ(selected.fault_plans.size(), 1u) << testing::PrintToString(selected.warnings);

  selected_options.test_kernel_name_filter = "unrelated_kernel";
  const ConSanResult rejected = try_patch_consan(bytes, selected_options);
  EXPECT_TRUE(rejected.fault_plans.empty());
}

TEST(ConSan, BarrierDropCarriesDistinctPristinePerturbationIdentityAcrossReinventory) {
  const std::array<uint32_t, 5> text_words = {
      0xBE804EC1u, // s_barrier_signal -1 (fault pair)
      0xBF94FFFFu, // s_barrier_wait -1
      0xBE804E81u, // s_barrier_signal 1 (perturbed pair)
      0xBF940001u, // s_barrier_wait 1
      0xBFB00000u,
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  ASSERT_EQ(std::ranges::count(inventory.fault_sites, ConSanFaultSiteKind::Barrier,
                               &ConSanFaultSite::kind),
            4u);
  const auto perturb = std::ranges::find_if(
      inventory.perturbation_candidates, [](const ConSanPerturbationCandidate &candidate) {
        return candidate.eligible && candidate.kind == ConSanPerturbationKind::Barrier &&
               candidate.edge == ConSanPerturbationEdge::Release &&
               candidate.anchor_text_offset == 2u * sizeof(uint32_t);
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
  options.max_patches = 1;
  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid)
      << testing::PrintToString(result.errors) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_TRUE(result.staged_composition_validated);
  EXPECT_EQ(result.planned_fault_mutations, 1u);
  EXPECT_EQ(result.applied_fault_mutations, 1u);
  EXPECT_EQ(result.applied_perturbations, 1u);
  ASSERT_EQ(result.patches.size(), 2u);
  EXPECT_EQ(result.patches[0].kind, ConSanPatchKind::InlineBarrierNopRewrite);
  const ConSanPatchInfo &patch = result.patches[1];
  EXPECT_EQ(patch.kind, ConSanPatchKind::TrampolineScPerturbation);
  EXPECT_EQ(patch.anchor_offset, 2u * sizeof(uint32_t));
  EXPECT_EQ(patch.perturbation_source_candidate_identity, perturb->identity);
  EXPECT_EQ(patch.perturbation_source_sequence_identity, perturb->sequence_identity);
  EXPECT_EQ(patch.perturbation_source_container_name, perturb->container_name);
  EXPECT_EQ(patch.perturbation_source_in_kernel, perturb->in_kernel);
  EXPECT_EQ(patch.perturbation_source_anchor_offset, perturb->anchor_text_offset);
  EXPECT_NE(patch.perturbation_sequence_identity, patch.perturbation_source_sequence_identity);
  EXPECT_TRUE(validate_consan_modified_elf(bytes, result).empty());
}

TEST(ConSan, BarrierMoveCarriesSelectedEdgeIntoOwnedWholePairTrampoline) {
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
  ASSERT_EQ(std::ranges::count(inventory.fault_sites, ConSanFaultSiteKind::Barrier,
                               &ConSanFaultSite::kind),
            2u);
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
  options.max_patches = 1;
  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid)
      << testing::PrintToString(result.errors) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_TRUE(result.staged_composition_validated);
  EXPECT_EQ(result.planned_fault_mutations, 1u);
  EXPECT_EQ(result.applied_fault_mutations, 1u);
  EXPECT_EQ(result.applied_perturbations, 1u);
  ASSERT_EQ(result.patches.size(), 4u);
  const ConSanPatchInfo &move_target = result.patches[2];
  const ConSanPatchInfo &perturb_patch = result.patches[3];
  ASSERT_EQ(move_target.kind, ConSanPatchKind::InlineBarrierMoveTargetRewrite);
  ASSERT_EQ(perturb_patch.kind, ConSanPatchKind::TrampolineScPerturbation);
  EXPECT_EQ(perturb_patch.anchor_offset, move_target.trampoline_offset);
  EXPECT_EQ(perturb_patch.perturbation_source_anchor_offset, perturb->anchor_text_offset);
  EXPECT_EQ(perturb_patch.perturbation_source_container_name, perturb->container_name);
  EXPECT_NE(perturb_patch.perturbation_sequence_identity,
            perturb_patch.perturbation_source_sequence_identity);
  EXPECT_TRUE(validate_consan_modified_elf(bytes, result).empty());

  ConSanOptions discovery = options;
  discovery.fault_dry_run = true;
  discovery.probe_lds_check_trap = true;
  discovery.max_patches = 2;
  const ConSanResult dry_run = try_patch_consan(bytes, discovery);
  ASSERT_TRUE(dry_run.errors.empty()) << testing::PrintToString(dry_run.errors);
  EXPECT_FALSE(dry_run.modified);
  ASSERT_EQ(dry_run.access_plans.size(), 1u);
  ASSERT_TRUE(dry_run.composite_proof);
  EXPECT_EQ(dry_run.composite_proof->pristine_identity, perturb->identity);
  EXPECT_EQ(dry_run.composite_proof->pristine_sequence, perturb->sequence_identity);
  EXPECT_EQ(dry_run.composite_proof->pristine_anchor, perturb->anchor_event_identity);
  EXPECT_EQ(dry_run.composite_proof->anchor_relation, "move-target+0");
  EXPECT_EQ(dry_run.composite_proof->translated_anchor_text_offset, destination->text_offset);
  EXPECT_FALSE(dry_run.composite_proof->atomic_overlap);
  EXPECT_FALSE(dry_run.composite_proof->removed_cache_non_resurrection_applicable);
}

TEST(ConSan, SyncSequencesPairLiteral32ButRejectLiteral64BarrierId) {
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const std::array<uint32_t, 4> literal32_words = {
      0xBE804EFFu,
      0xFFFFFFFFu, // s_barrier_signal literal32(-1)
      0xBF94FFFFu, // s_barrier_wait -1
      0xBFB00000u, // s_endpgm
  };
  const ConSanResult literal32 =
      try_patch_consan(make_gfx1250_code_object(literal32_words), options);
  ASSERT_TRUE(literal32.errors.empty())
      << (literal32.errors.empty() ? "" : literal32.errors.front());
  ASSERT_EQ(literal32.sync_sequences.size(), 1u);
  EXPECT_EQ(literal32.sync_sequences[0].operation, ConSanSyncOperation::BarrierFull);
  EXPECT_EQ(literal32.sync_sequences[0].barrier_operand_source,
            ConSanBarrierSite::OperandSource::Literal32);
  EXPECT_EQ(literal32.sync_sequences[0].barrier_id, -1);
  EXPECT_EQ(literal32.sync_sequences[0].barrier_literal_value, 0xFFFFFFFFu);

  const std::array<uint32_t, 5> literal64_words = {
      0xBE804EFEu, 0xFFFFFFFFu, 0xFFFFFFFFu, // s_barrier_signal literal64
      0xBF94FFFFu,                           // s_barrier_wait -1
      0xBFB00000u,                           // s_endpgm
  };
  const ConSanResult literal64 =
      try_patch_consan(make_gfx1250_code_object(literal64_words), options);
  ASSERT_TRUE(literal64.errors.empty())
      << (literal64.errors.empty() ? "" : literal64.errors.front());
  ASSERT_EQ(literal64.sync_sequences.size(), 2u);
  EXPECT_EQ(literal64.sync_sequences[0].operation, ConSanSyncOperation::BarrierSignal);
  EXPECT_EQ(literal64.sync_sequences[0].barrier_operand_source,
            ConSanBarrierSite::OperandSource::Literal64);
  EXPECT_FALSE(literal64.sync_sequences[0].barrier_id);
  EXPECT_EQ(literal64.sync_sequences[0].confidence, ConSanSemanticConfidence::Unsupported);
  EXPECT_EQ(literal64.sync_sequences[1].operation, ConSanSyncOperation::BarrierWait);
  EXPECT_EQ(literal64.sync_sequences[1].confidence, ConSanSemanticConfidence::Ambiguous);
}

TEST(ConSan, FaultBarrierIdScopeRewritesCompleteStaticLifecycleAsOneMutation) {
  const std::array<uint32_t, 6> text_words = {
      0xBE805181u, // s_barrier_init 1
      0xBE805281u, // s_barrier_join 1
      0xBE804E81u, // s_barrier_signal 1
      0xBF940001u, // s_barrier_wait 1
      0xBF950000u, // s_barrier_leave
      0xBFB00000u,
  };
  const std::vector<uint8_t> bytes = make_gfx1250_code_object(text_words);
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  inventory_options.fault_dry_run = true;
  const ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  const auto barrier = std::ranges::find(inventory.sync_sequences, ConSanSyncOperation::BarrierFull,
                                         &ConSanSyncSequence::operation);
  ASSERT_NE(barrier, inventory.sync_sequences.end());

  ConSanOptions options = inventory_options;
  options.fault_mutate_barrier_id_scope = true;
  options.fault_barrier_sequence_identity = barrier->identity;
  options.fault_barrier_target_id = 2;
  options.fault_require_exactly_one = true;
  const ConSanResult dry_run = try_patch_consan(bytes, options);
  ASSERT_TRUE(dry_run.errors.empty()) << (dry_run.errors.empty() ? "" : dry_run.errors.front());
  ASSERT_EQ(dry_run.fault_plans.size(), 1u);
  EXPECT_EQ(dry_run.fault_plans[0].logical_sequence_identity,
            inventory.barrier_lifecycle_groups[0].identity);
  EXPECT_EQ(dry_run.fault_plans[0].ordered_member_identities.size(), 5u);

  options.fault_dry_run = false;
  const ConSanResult execution = try_patch_consan(bytes, options);
  ASSERT_TRUE(execution.errors.empty())
      << (execution.errors.empty() ? "" : execution.errors.front());
  EXPECT_TRUE(execution.modified);
  EXPECT_EQ(execution.applied_fault_mutations, 1u);
  EXPECT_TRUE(execution.final_validation_passed);
  ASSERT_EQ(execution.patches.size(), 4u);
  EXPECT_TRUE(std::ranges::all_of(execution.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::InlineBarrierIdScopeRewrite;
  }));

  AmdGpuCodeObject patched(execution.elf_bytes.data(), execution.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const auto *text = reinterpret_cast<const uint32_t *>(patched.text_sections().front()->data());
  EXPECT_EQ(text[0], 0xBE805182u);   // s_barrier_init 2
  EXPECT_EQ(text[1], 0xBE805282u);   // s_barrier_join 2
  EXPECT_EQ(text[2], 0xBE804E82u);   // s_barrier_signal 2
  EXPECT_EQ(text[3], 0xBF940002u);   // s_barrier_wait 2
  EXPECT_EQ(text[4], text_words[4]); // fixed-zero leave stays byte-identical
  EXPECT_EQ(text[5], text_words[5]);
  const auto qualification = consan_barrier_mutation_qualification(
      "gfx1250", ConSanBarrierMutationForm::StaticNamedLifecycle);
  EXPECT_EQ(qualification.host, ConSanQualificationEvidence::Proven);
  EXPECT_EQ(qualification.live_gpu, ConSanQualificationEvidence::DeferredA1);
  EXPECT_EQ(qualification.cluster_or_multi_device, ConSanQualificationEvidence::None);
}

TEST(ConSan, FaultBarrierIdScopeRewritesEveryLiteralLifecycleMember) {
  const std::array<uint32_t, 10> text_words = {
      0xBE8051FFu, 0x00000001u, // s_barrier_init literal32(1)
      0xBE8052FFu, 0x00000001u, // s_barrier_join literal32(1)
      0xBE805281u,              // s_barrier_join 1
      0xBE804EFFu, 0x00000001u, // s_barrier_signal literal32(1)
      0xBF940001u,              // s_barrier_wait 1
      0xBF950000u,              // s_barrier_leave
      0xBFB00000u,
  };
  const std::vector<uint8_t> bytes = make_gfx1250_code_object(text_words);
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  inventory_options.fault_dry_run = true;
  const ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  ASSERT_EQ(inventory.barrier_lifecycle_groups.size(), 1u);
  ASSERT_TRUE(inventory.barrier_lifecycle_groups[0].admissible);
  const auto barrier = std::ranges::find(inventory.sync_sequences, ConSanSyncOperation::BarrierFull,
                                         &ConSanSyncSequence::operation);
  ASSERT_NE(barrier, inventory.sync_sequences.end());

  ConSanOptions options = inventory_options;
  options.fault_dry_run = false;
  options.fault_mutate_barrier_id_scope = true;
  options.fault_barrier_sequence_identity = barrier->identity;
  options.fault_barrier_target_id = 16;
  options.fault_require_exactly_one = true;
  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_EQ(result.applied_fault_mutations, 1u);
  ASSERT_EQ(result.patches.size(), 5u);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const auto *text = reinterpret_cast<const uint32_t *>(patched.text_sections().front()->data());
  EXPECT_EQ(text[0], text_words[0]);
  EXPECT_EQ(text[1], 16u);
  EXPECT_EQ(text[2], text_words[2]);
  EXPECT_EQ(text[3], 16u);
  EXPECT_EQ(text[4], 0xBE805290u);
  EXPECT_EQ(text[5], text_words[5]);
  EXPECT_EQ(text[6], 16u);
  EXPECT_EQ(text[7], 0xBF940010u);
  EXPECT_EQ(text[8], text_words[8]);
  EXPECT_EQ(text[9], text_words[9]);
}

TEST(ConSan, FaultBarrierParticipantCountRewritesProvenLiteralM0LifecycleSetup) {
  constexpr uint32_t kSetupWord =
      build_s_mov_b32(/*sdst=*/125, /*ssrc0=*/255, ROCJITSU_CODE_ARCH_GFX1250);
  static_assert(kSetupWord == 0xBEFD00FFu);
  const std::array<uint32_t, 8> text_words = {
      kSetupWord,  0x000C0001u, // s_mov_b32 m0, count=12 | named barrier ID=1
      0xBE80517Du,              // s_barrier_init m0
      0xBE805281u,              // s_barrier_join 1
      0xBE804E81u,              // s_barrier_signal 1
      0xBF940001u,              // s_barrier_wait 1
      0xBF950000u,              // s_barrier_leave
      0xBFB00000u,
  };
  const std::vector<uint8_t> bytes = make_gfx1250_code_object(text_words);
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  inventory_options.fault_dry_run = true;
  const ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  ASSERT_EQ(inventory.barrier_lifecycle_groups.size(), 1u);
  ASSERT_TRUE(inventory.barrier_lifecycle_groups.front().admissible);
  const auto init = std::ranges::find(inventory.sync_events, ConSanSyncOperation::BarrierInit,
                                      &ConSanSyncEvent::operation);
  ASSERT_NE(init, inventory.sync_events.end());
  EXPECT_EQ(init->barrier_operand_source, ConSanBarrierSite::OperandSource::StaticM0Literal32);
  EXPECT_EQ(init->barrier_id, 1);
  EXPECT_EQ(init->participant_count, 12u);
  EXPECT_FALSE(init->participant_mask);
  const auto barrier = std::ranges::find(inventory.sync_sequences, ConSanSyncOperation::BarrierFull,
                                         &ConSanSyncSequence::operation);
  ASSERT_NE(barrier, inventory.sync_sequences.end());

  ConSanOptions options = inventory_options;
  options.fault_mutate_barrier_participants = true;
  options.fault_barrier_sequence_identity = barrier->identity;
  options.fault_barrier_target_participant_count = 8;
  options.fault_require_exactly_one = true;
  const ConSanResult dry_run = try_patch_consan(bytes, options);
  ASSERT_EQ(dry_run.outcome, ConSanTransformOutcome::Unchanged);
  ASSERT_EQ(dry_run.fault_plans.size(), 1u);
  EXPECT_EQ(dry_run.fault_plans.front().kind, ConSanFaultMutationKind::BarrierParticipantCount);
  EXPECT_EQ(dry_run.fault_plans.front().original_participant_count, 12u);
  EXPECT_EQ(dry_run.fault_plans.front().target_participant_count, 8u);
  EXPECT_EQ(dry_run.fault_plans.front().logical_sequence_identity,
            inventory.barrier_lifecycle_groups.front().identity);

  options.fault_dry_run = false;
  const ConSanResult result = try_patch_consan(bytes, options);
  ASSERT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid)
      << testing::PrintToString(result.errors);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_EQ(result.requested_fault_mutations, 1u);
  EXPECT_EQ(result.applied_fault_mutations, 1u);
  EXPECT_EQ(result.applied_fault_logical_identity,
            inventory.barrier_lifecycle_groups.front().identity);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineBarrierParticipantCountRewrite);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().original_size, 8u);
  EXPECT_EQ(result.patches.front().original_participant_count, 12u);
  EXPECT_EQ(result.patches.front().target_participant_count, 8u);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const auto *text = reinterpret_cast<const uint32_t *>(patched.text_sections().front()->data());
  EXPECT_EQ(text[0], text_words[0]);
  EXPECT_EQ(text[1], 0x00080001u);
  EXPECT_TRUE(std::ranges::equal(text_words.begin() + 2, text_words.end(), text + 2, text + 8));
}

TEST(ConSan, FaultBarrierParticipantsReturnTypedUnsupportedWithoutProvenEncoding) {
  const std::array<uint32_t, 6> immediate_init_words = {
      0xBE805181u, 0xBE805281u, 0xBE804E81u, 0xBF940001u, 0xBF950000u, 0xBFB00000u,
  };
  const std::vector<uint8_t> immediate_bytes = make_gfx1250_code_object(immediate_init_words);
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  inventory_options.fault_dry_run = true;
  const ConSanResult inventory = try_patch_consan(immediate_bytes, inventory_options);
  const auto barrier = std::ranges::find(inventory.sync_sequences, ConSanSyncOperation::BarrierFull,
                                         &ConSanSyncSequence::operation);
  ASSERT_NE(barrier, inventory.sync_sequences.end());

  ConSanOptions options = inventory_options;
  options.fault_mutate_barrier_participants = true;
  options.fault_barrier_sequence_identity = barrier->identity;
  options.fault_barrier_target_participant_count = 8;
  options.fault_require_exactly_one = true;
  const ConSanResult unavailable = try_patch_consan(immediate_bytes, options);
  EXPECT_EQ(unavailable.outcome, ConSanTransformOutcome::Unsupported);
  EXPECT_TRUE(unavailable.errors.empty());
  EXPECT_TRUE(unavailable.fault_plans.empty());
  EXPECT_EQ(unavailable.requested_fault_mutations, 1u);
  EXPECT_TRUE(std::ranges::any_of(unavailable.warnings, [](const std::string &warning) {
    return warning == "ConSan barrier participant mutation is unsupported: no complete lifecycle "
                      "has an immediately preceding literal M0 count setup";
  }));

  constexpr uint32_t kSetupWord =
      build_s_mov_b32(/*sdst=*/125, /*ssrc0=*/255, ROCJITSU_CODE_ARCH_GFX1250);
  const std::array<uint32_t, 8> counted_words = {
      kSetupWord,  0x000C0001u, 0xBE80517Du, 0xBE805281u,
      0xBE804E81u, 0xBF940001u, 0xBF950000u, 0xBFB00000u,
  };
  const std::vector<uint8_t> counted_bytes = make_gfx1250_code_object(counted_words);
  const ConSanResult counted_inventory = try_patch_consan(counted_bytes, inventory_options);
  const auto counted_barrier =
      std::ranges::find(counted_inventory.sync_sequences, ConSanSyncOperation::BarrierFull,
                        &ConSanSyncSequence::operation);
  ASSERT_NE(counted_barrier, counted_inventory.sync_sequences.end());
  options.fault_barrier_sequence_identity = counted_barrier->identity;
  options.fault_barrier_target_participant_count.reset();
  options.fault_barrier_target_participant_mask = 0x3;
  const ConSanResult mask = try_patch_consan(counted_bytes, options);
  EXPECT_EQ(mask.outcome, ConSanTransformOutcome::Unsupported);
  EXPECT_TRUE(mask.errors.empty());
  EXPECT_TRUE(mask.fault_plans.empty());
  EXPECT_TRUE(std::ranges::any_of(mask.warnings, [](const std::string &warning) {
    return warning == "ConSan barrier participant mutation is unsupported: the GFX12.5 lifecycle "
                      "exposes a six-bit count in M0, not a participant mask";
  }));
}

TEST(ConSan, FaultBarrierLifecycleComposesWithMoiAsOneRetainedMutation) {
  const std::array<uint32_t, 6> text_words = {
      0xBE805181u, // s_barrier_init 1
      0xBE805281u, // s_barrier_join 1
      0xBE804E81u, // s_barrier_signal 1
      0xBF940001u, // s_barrier_wait 1
      0xBF950000u, // s_barrier_leave
      0xBFB00000u,
  };
  const std::vector<uint8_t> bytes = make_gfx1250_code_object(text_words);
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  inventory_options.fault_dry_run = true;
  const ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  ASSERT_EQ(inventory.barrier_lifecycle_groups.size(), 1u);
  const auto barrier = std::ranges::find(inventory.sync_sequences, ConSanSyncOperation::BarrierFull,
                                         &ConSanSyncSequence::operation);
  ASSERT_NE(barrier, inventory.sync_sequences.end());

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_track_barriers = true;
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 30;
  options.moi_owner_vgpr = 14;
  options.moi_epoch_vgpr = 15;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(5, 0, 0, 0, 5);
  options.max_patches = 8;
  options.fault_mutate_barrier_id_scope = true;
  options.fault_barrier_sequence_identity = barrier->identity;
  options.fault_barrier_target_id = 2;
  options.fault_require_exactly_one = true;
  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid)
      << testing::PrintToString(result.errors);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_TRUE(result.staged_composition_validated);
  EXPECT_EQ(result.requested_fault_mutations, 1u);
  EXPECT_EQ(result.applied_fault_mutations, 1u);
  ASSERT_TRUE(result.applied_fault_logical_identity);
  EXPECT_EQ(*result.applied_fault_logical_identity,
            inventory.barrier_lifecycle_groups.front().identity);
  ASSERT_EQ(result.barrier_lifecycle_groups.size(), 1u);
  EXPECT_EQ(result.barrier_lifecycle_groups.front().barrier_id, 2);

  const size_t mutation_count =
      std::ranges::count_if(result.patches, [](const ConSanPatchInfo &patch) {
        return patch.phase == ConSanPatchPhase::Mutation &&
               patch.kind == ConSanPatchKind::InlineBarrierIdScopeRewrite;
      });
  EXPECT_EQ(mutation_count, 4u);
  const auto init = std::ranges::find(inventory.sync_events, ConSanSyncOperation::BarrierInit,
                                      &ConSanSyncEvent::operation);
  ASSERT_NE(init, inventory.sync_events.end());
  const auto owner = result.kernels.front().descriptor_file_offset;
  size_t barrier_record_count = 0;
  size_t nested_instrumentation_count = 0;
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.phase != ConSanPatchPhase::Instrumentation ||
        patch.kind != ConSanPatchKind::TrampolineMoiBarrierRecord)
      continue;
    ++barrier_record_count;
    EXPECT_NE(std::ranges::find(patch.owner_descriptor_file_offsets, owner),
              patch.owner_descriptor_file_offsets.end());
    if (std::ranges::any_of(result.patches, [&](const ConSanPatchInfo &mutation) {
          return mutation.phase == ConSanPatchPhase::Mutation &&
                 mutation.anchor_offset == patch.anchor_offset;
        })) {
      ++nested_instrumentation_count;
    }
  }
  // Every lifecycle event is recorded. Init, join, signal, and wait are also
  // rewritten, so those four records nest over mutation-owned anchors; leave
  // is recorded while its required fixed-zero instruction remains unchanged.
  EXPECT_EQ(barrier_record_count, 5u);
  EXPECT_EQ(nested_instrumentation_count, 4u);
  EXPECT_TRUE(std::ranges::any_of(result.patches, [&](const ConSanPatchInfo &patch) {
    return patch.phase == ConSanPatchPhase::Mutation && patch.anchor_offset == init->text_offset;
  }));
  EXPECT_TRUE(std::ranges::any_of(result.patches, [&](const ConSanPatchInfo &patch) {
    return patch.phase == ConSanPatchPhase::Instrumentation &&
           patch.kind == ConSanPatchKind::TrampolineMoiBarrierRecord &&
           patch.anchor_offset == init->text_offset;
  }));
}

TEST(ConSan, FaultBarrierLifecycleRollsBackWhenMoiResourcesAreUnsupported) {
  const std::array<uint32_t, 6> text_words = {
      0xBE805181u, 0xBE805281u, 0xBE804E81u, 0xBF940001u, 0xBF950000u, 0xBFB00000u,
  };
  const std::vector<uint8_t> bytes = make_gfx1250_code_object(text_words);
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  inventory_options.fault_dry_run = true;
  const ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  const auto barrier = std::ranges::find(inventory.sync_sequences, ConSanSyncOperation::BarrierFull,
                                         &ConSanSyncSequence::operation);
  ASSERT_NE(barrier, inventory.sync_sequences.end());

  ConSanOptions options = moi_options();
  options.moi_track_barriers = true;
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 30;
  options.moi_owner_vgpr = 9; // Deliberately overlaps the six-VGPR scratch window.
  options.moi_epoch_vgpr = 15;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(5, 0, 0, 0, 5);
  options.max_patches = 8;
  options.fault_mutate_barrier_id_scope = true;
  options.fault_barrier_sequence_identity = barrier->identity;
  options.fault_barrier_target_id = 2;
  options.fault_require_exactly_one = true;
  const ConSanResult result = try_patch_consan(bytes, options);

  EXPECT_EQ(result.outcome, ConSanTransformOutcome::Unsupported);
  EXPECT_FALSE(result.modified);
  EXPECT_FALSE(result.final_validation_passed);
  EXPECT_FALSE(result.staged_composition_validated);
  EXPECT_TRUE(result.elf_bytes.empty());
  EXPECT_TRUE(result.patches.empty());
  EXPECT_TRUE(result.errors.empty());
  EXPECT_EQ(result.requested_fault_mutations, 1u);
  EXPECT_EQ(result.applied_fault_mutations, 0u);
  EXPECT_FALSE(result.applied_fault_logical_identity);
  EXPECT_EQ(result.resource_plans.size(), 5u);
  EXPECT_EQ(result.resource_plan_summary.unsupported_plans, 5u);
  EXPECT_TRUE(std::ranges::all_of(result.resource_plans, [](const auto &plan) {
    return plan.source == ConSanRegisterAllocationSource::Unsupported &&
           plan.reason == ConSanRegisterPlanReason::ForbiddenOverlap;
  }));
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning ==
           "ConSan rolled back staged fault mutation because instrumentation is unsupported";
  }));
}

TEST(ConSan, FaultDropBarrierModeRewritesSelectedBarrier) {
  const std::array<uint32_t, 5> text_words = {
      0xBF940000u,              // s_barrier_wait
      0xD8340000u, 0x00000000u, // ds_store_b32
      0xBF940000u,              // s_barrier_wait
      0xBFB00000u,              // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.fault_drop_barrier = true;
  options.fault_barrier_index = 1;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().phase, ConSanPatchPhase::Mutation);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineBarrierNopRewrite);
  EXPECT_EQ(result.patches.front().anchor_offset, 12u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 12u);
  EXPECT_EQ(result.patches.front().original_size, 4u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  uint32_t first_barrier = 0;
  uint32_t rewritten_barrier = 0;
  std::memcpy(&first_barrier, result.elf_bytes.data() + 0x100, sizeof(first_barrier));
  std::memcpy(&rewritten_barrier, result.elf_bytes.data() + 0x100 + 12, sizeof(rewritten_barrier));
  EXPECT_EQ(first_barrier, 0xBF940000u);
  EXPECT_EQ(rewritten_barrier, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(ConSan, FaultDropInventoryCanSkipUnneededMoveDestinations) {
  const std::array<uint32_t, 5> text_words = {
      0xD8340000u, 0x00000000u, // ds_store_b32
      0xBE804EC1u,              // s_barrier_signal -1
      0xBF94FFFFu,              // s_barrier_wait -1
      0xBFB00000u,              // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "drop_inventory_without_move_destinations");
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.fault_drop_barrier = true;
  options.fault_dry_run = true;
  options.collect_barrier_move_destinations = false;
  const ConSanResult inventory = try_patch_consan(bytes, options);
  EXPECT_FALSE(inventory.fault_sites.empty());
  EXPECT_FALSE(inventory.sync_sequences.empty());
  EXPECT_TRUE(inventory.barrier_move_destinations.empty());
}

TEST(ConSan, FaultDropBarrierRejectsQualifiedPairHalfWithoutDestructiveOptIn) {
  const std::array<uint32_t, 3> text_words = {
      0xBE804EC1u, // s_barrier_signal -1
      0xBF94FFFFu, // s_barrier_wait -1
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words, "guarded_pair_drop");
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  ASSERT_EQ(std::ranges::count(inventory.fault_sites, ConSanFaultSiteKind::Barrier,
                               &ConSanFaultSite::kind),
            2u);

  ConSanOptions options = inventory_options;
  options.fault_drop_barrier = true;
  options.fault_site_identity = inventory.fault_sites.front().identity;
  const ConSanResult rejected = try_patch_consan(bytes, options);
  EXPECT_EQ(rejected.outcome, ConSanTransformOutcome::Unchanged);
  EXPECT_FALSE(rejected.modified);
  EXPECT_TRUE(rejected.patches.empty());
  EXPECT_EQ(rejected.requested_fault_mutations, 1u);
  EXPECT_EQ(rejected.applied_fault_mutations, 0u);
  EXPECT_TRUE(std::ranges::any_of(rejected.warnings, [](const std::string &warning) {
    return warning.find("dropping one half of a qualified logical barrier pair") !=
           std::string::npos;
  }));

  options.fault_allow_destructive_incomplete_barrier_drop = true;
  const ConSanResult allowed = try_patch_consan(bytes, options);
  ASSERT_EQ(allowed.outcome, ConSanTransformOutcome::ModifiedValid)
      << (allowed.errors.empty() ? "" : allowed.errors.front());
  ASSERT_EQ(allowed.patches.size(), 1u);
  EXPECT_EQ(allowed.patches.front().anchor_offset, inventory.fault_sites.front().text_offset);
  EXPECT_EQ(allowed.applied_fault_mutations, 1u);
}

TEST(ConSan, FaultDropBarrierDryRunRequiresDestructiveOptInForQualifiedPairHalf) {
  const std::array<uint32_t, 3> text_words = {
      0xBE804EC1u, // s_barrier_signal -1
      0xBF94FFFFu, // s_barrier_wait -1
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words, "dry_guarded_drop");
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  ASSERT_EQ(std::ranges::count(inventory.fault_sites, ConSanFaultSiteKind::Barrier,
                               &ConSanFaultSite::kind),
            2u);

  ConSanOptions options = inventory_options;
  options.fault_drop_barrier = true;
  options.fault_dry_run = true;
  options.fault_site_identity = inventory.fault_sites.back().identity;
  const ConSanResult rejected = try_patch_consan(bytes, options);
  EXPECT_TRUE(rejected.fault_plans.empty());
  EXPECT_EQ(rejected.planned_fault_mutations, 0u);

  options.fault_allow_destructive_incomplete_barrier_drop = true;
  const ConSanResult allowed = try_patch_consan(bytes, options);
  ASSERT_EQ(allowed.fault_plans.size(), 1u);
  EXPECT_EQ(allowed.fault_plans.front().kind, ConSanFaultMutationKind::DropBarrier);
  EXPECT_EQ(allowed.fault_plans.front().primary_identity, inventory.fault_sites.back().identity);
  EXPECT_EQ(allowed.planned_fault_mutations, 1u);
}

TEST(ConSan, FaultDropBarrierExactSequenceRewritesBothMembersAsOneMutation) {
  const std::array<uint32_t, 3> text_words = {
      0xBE804EC1u, // s_barrier_signal -1
      0xBF94FFFFu, // s_barrier_wait -1
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "exact_whole_barrier_drop");
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  ASSERT_EQ(std::ranges::count(inventory.fault_sites, ConSanFaultSiteKind::Barrier,
                               &ConSanFaultSite::kind),
            2u);
  const auto sequence = std::ranges::find(
      inventory.sync_sequences, ConSanSyncOperation::BarrierFull, &ConSanSyncSequence::operation);
  ASSERT_NE(sequence, inventory.sync_sequences.end());

  ConSanOptions options = inventory_options;
  options.fault_drop_barrier = true;
  options.fault_site_identity = inventory.fault_sites.front().identity;
  options.fault_barrier_sequence_identity = sequence->identity;
  options.fault_require_exactly_one = true;
  options.fault_dry_run = true;
  const ConSanResult dry_run = try_patch_consan(bytes, options);
  ASSERT_TRUE(dry_run.errors.empty()) << testing::PrintToString(dry_run.errors);
  ASSERT_EQ(dry_run.fault_plans.size(), 1u);
  EXPECT_EQ(dry_run.planned_fault_mutations, 1u);
  EXPECT_EQ(dry_run.fault_plans.front().primary_identity, inventory.fault_sites.front().identity);
  EXPECT_EQ(dry_run.fault_plans.front().companion_identity, inventory.fault_sites.back().identity);
  EXPECT_EQ(dry_run.fault_plans.front().logical_sequence_identity, sequence->identity);
  EXPECT_EQ(dry_run.fault_plans.front().ordered_member_identities,
            sequence->member_event_identities);

  options.fault_dry_run = false;
  const ConSanResult execution = try_patch_consan(bytes, options);
  ASSERT_EQ(execution.outcome, ConSanTransformOutcome::ModifiedValid)
      << testing::PrintToString(execution.errors);
  EXPECT_TRUE(execution.final_validation_passed);
  EXPECT_EQ(execution.requested_fault_mutations, 1u);
  EXPECT_EQ(execution.planned_fault_mutations, 1u);
  EXPECT_EQ(execution.applied_fault_mutations, 1u);
  ASSERT_EQ(execution.fault_plans.size(), 1u);
  EXPECT_EQ(execution.fault_plans.front().logical_sequence_identity, sequence->identity);
  EXPECT_EQ(execution.applied_fault_logical_identity, sequence->identity);
  ASSERT_EQ(execution.patches.size(), 2u);
  EXPECT_TRUE(std::ranges::all_of(execution.patches, [&](const ConSanPatchInfo &patch) {
    return patch.phase == ConSanPatchPhase::Mutation &&
           patch.kind == ConSanPatchKind::InlineBarrierNopRewrite &&
           patch.fault_primary_identity == inventory.fault_sites.front().identity &&
           patch.fault_companion_identity == inventory.fault_sites.back().identity &&
           patch.fault_sequence_identity == sequence->identity;
  }));
  AmdGpuCodeObject patched(execution.elf_bytes.data(), execution.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const auto *text = reinterpret_cast<const uint32_t *>(patched.text_sections().front()->data());
  EXPECT_EQ(text[0], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(text[1], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(text[2], text_words[2]);

  ConSanOptions no_target_options = options;
  no_target_options.flavor = ConSanFlavor::Moi;
  no_target_options.moi_engine = ConSanMoiEngine::RecordReplay;
  const ConSanResult no_target_execution = try_patch_consan(bytes, no_target_options);
  ASSERT_EQ(no_target_execution.outcome, ConSanTransformOutcome::ModifiedValid)
      << testing::PrintToString(no_target_execution.errors);
  EXPECT_EQ(no_target_execution.requested_fault_mutations, 1u);
  EXPECT_EQ(no_target_execution.planned_fault_mutations, 1u);
  EXPECT_EQ(no_target_execution.applied_fault_mutations, 1u);
  ASSERT_EQ(no_target_execution.fault_plans.size(), 1u);
  EXPECT_EQ(no_target_execution.fault_plans.front().logical_sequence_identity, sequence->identity);
  EXPECT_EQ(no_target_execution.applied_fault_logical_identity, sequence->identity);
  EXPECT_EQ(no_target_execution.patches.size(), 2u);

  ConSanResult corrupted = execution;
  corrupted.patches.back().fault_companion_identity = "stale-companion";
  const std::vector<std::string> validation_errors = validate_consan_modified_elf(bytes, corrupted);
  EXPECT_TRUE(std::ranges::any_of(validation_errors, [](const std::string &error) {
    return error.find("complete exact whole-barrier drop") != std::string::npos;
  }));
}

TEST(ConSan, FaultDropBarrierExactSequenceRejectsStaleAndIncompleteIdentities) {
  const std::array<uint32_t, 3> pair_words = {
      0xBE804EC1u,
      0xBF94FFFFu,
      0xBFB00000u,
  };
  const std::vector<uint8_t> pair_bytes =
      make_rdna4_lds_code_object(pair_words, "stale_whole_barrier_drop");
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult inventory = try_patch_consan(pair_bytes, inventory_options);
  ASSERT_EQ(std::ranges::count(inventory.fault_sites, ConSanFaultSiteKind::Barrier,
                               &ConSanFaultSite::kind),
            2u);
  const auto sequence = std::ranges::find(
      inventory.sync_sequences, ConSanSyncOperation::BarrierFull, &ConSanSyncSequence::operation);
  ASSERT_NE(sequence, inventory.sync_sequences.end());

  ConSanOptions options = inventory_options;
  options.fault_drop_barrier = true;
  options.fault_require_exactly_one = true;
  options.fault_site_identity = inventory.fault_sites.front().identity;
  options.fault_barrier_sequence_identity = "missing-sequence";
  const ConSanResult stale_sequence = try_patch_consan(pair_bytes, options);
  EXPECT_EQ(stale_sequence.outcome, ConSanTransformOutcome::Invalid);
  EXPECT_FALSE(stale_sequence.modified);
  EXPECT_TRUE(stale_sequence.patches.empty());
  EXPECT_EQ(stale_sequence.requested_fault_mutations, 1u);
  EXPECT_EQ(stale_sequence.planned_fault_mutations, 0u);
  EXPECT_EQ(stale_sequence.applied_fault_mutations, 0u);
  EXPECT_TRUE(stale_sequence.fault_plans.empty());

  options.fault_barrier_sequence_identity = sequence->identity;
  options.fault_site_identity = "missing-site";
  const ConSanResult stale_site = try_patch_consan(pair_bytes, options);
  EXPECT_EQ(stale_site.outcome, ConSanTransformOutcome::Invalid);
  EXPECT_FALSE(stale_site.modified);
  EXPECT_TRUE(stale_site.patches.empty());
  EXPECT_EQ(stale_site.applied_fault_mutations, 0u);

  const std::array<uint32_t, 2> partial_words = {
      0xBE804EC1u, // s_barrier_signal -1 without a completing wait
      0xBFB00000u,
  };
  const std::vector<uint8_t> partial_bytes =
      make_rdna4_lds_code_object(partial_words, "partial_whole_barrier_drop");
  const ConSanResult partial_inventory = try_patch_consan(partial_bytes, inventory_options);
  ASSERT_EQ(partial_inventory.fault_sites.size(), 1u);
  ASSERT_EQ(partial_inventory.sync_sequences.size(), 1u);
  options.fault_site_identity = partial_inventory.fault_sites.front().identity;
  options.fault_barrier_sequence_identity = partial_inventory.sync_sequences.front().identity;
  const ConSanResult partial = try_patch_consan(partial_bytes, options);
  EXPECT_EQ(partial.outcome, ConSanTransformOutcome::Invalid);
  EXPECT_FALSE(partial.modified);
  EXPECT_TRUE(partial.patches.empty());
  EXPECT_EQ(partial.applied_fault_mutations, 0u);
}

TEST(ConSan, FaultDropBarrierExactSequenceAcceptsBoundedQwenStylePairOnlyInFaultMode) {
  std::array<uint32_t, 17> text_words{};
  text_words[0] = 0xBE804EC1u; // s_barrier_signal -1 at 0x00.
  std::fill(text_words.begin() + 1, text_words.begin() + 15,
            build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words[15] = 0xBF94FFFFu; // s_barrier_wait -1 at 0x3c.
  text_words[16] = 0xBFB00000u; // s_endpgm
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "qwen_style_whole_barrier_drop");

  ConSanOptions ordinary_options;
  ordinary_options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult ordinary = try_patch_consan(bytes, ordinary_options);
  EXPECT_EQ(std::ranges::count(ordinary.sync_sequences, ConSanSyncOperation::BarrierFull,
                               &ConSanSyncSequence::operation),
            0u);

  ConSanOptions inventory_options = ordinary_options;
  inventory_options.fault_drop_barrier = true;
  inventory_options.fault_dry_run = true;
  const ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  const auto sequence = std::ranges::find(
      inventory.sync_sequences, ConSanSyncOperation::BarrierFull, &ConSanSyncSequence::operation);
  ASSERT_NE(sequence, inventory.sync_sequences.end());
  ASSERT_EQ(sequence->member_event_identities.size(), 2u);
  EXPECT_EQ(sequence->begin_text_offset, 0u);
  EXPECT_EQ(sequence->end_text_offset, 16u * sizeof(uint32_t));
  EXPECT_NE(sequence->confidence_reason.find("same-owner same-block"), std::string::npos);
  EXPECT_NE(sequence->confidence_reason.find("no intervening barrier"), std::string::npos);

  const auto primary = std::ranges::find_if(inventory.fault_sites, [&](const auto &site) {
    return site.sync_sequence_identity == sequence->identity && site.mnemonic == "s_barrier_signal";
  });
  ASSERT_NE(primary, inventory.fault_sites.end());
  ConSanOptions execution_options = inventory_options;
  execution_options.fault_dry_run = false;
  execution_options.fault_site_identity = primary->identity;
  execution_options.fault_barrier_sequence_identity = sequence->identity;
  execution_options.fault_require_exactly_one = true;
  const ConSanResult execution = try_patch_consan(bytes, execution_options);
  ASSERT_EQ(execution.outcome, ConSanTransformOutcome::ModifiedValid)
      << testing::PrintToString(execution.errors);
  EXPECT_TRUE(execution.final_validation_passed);
  EXPECT_EQ(execution.applied_fault_mutations, 1u);
  EXPECT_EQ(execution.applied_fault_logical_identity, sequence->identity);
  ASSERT_EQ(execution.patches.size(), 2u);
  EXPECT_EQ(execution.patches[0].anchor_offset, 0u);
  EXPECT_EQ(execution.patches[1].anchor_offset, 15u * sizeof(uint32_t));
}

TEST(ConSan, FaultDropBarrierExactSequenceAcceptsLongExactPairsAndRejectsUnsafeShapes) {
  const auto full_pair_count = [](std::span<const uint32_t> words) {
    ConSanOptions options;
    options.flavor = ConSanFlavor::SuperCollider;
    options.fault_drop_barrier = true;
    options.fault_dry_run = true;
    const ConSanResult result = try_patch_consan(make_rdna4_lds_code_object(words), options);
    return std::ranges::count(result.sync_sequences, ConSanSyncOperation::BarrierFull,
                              &ConSanSyncSequence::operation);
  };

  std::array<uint32_t, 20> long_straight_line_pair{};
  long_straight_line_pair[0] = 0xBE804EC1u;
  std::fill(long_straight_line_pair.begin() + 1, long_straight_line_pair.begin() + 18,
            build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  long_straight_line_pair[18] = 0xBF94FFFFu;
  long_straight_line_pair[19] = 0xBFB00000u;
  EXPECT_EQ(full_pair_count(long_straight_line_pair), 1u);

  std::array<uint32_t, 36> excessive_distance{};
  excessive_distance[0] = 0xBE804EC1u;
  std::fill(excessive_distance.begin() + 1, excessive_distance.begin() + 34,
            build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  excessive_distance[34] = 0xBF94FFFFu;
  excessive_distance[35] = 0xBFB00000u;
  EXPECT_EQ(full_pair_count(excessive_distance), 1u);

  const std::array<uint32_t, 4> intervening_barrier = {
      0xBE804EC1u, // s_barrier_signal -1
      0xBF940001u, // intervening s_barrier_wait 1
      0xBF94FFFFu, // s_barrier_wait -1
      0xBFB00000u,
  };
  EXPECT_EQ(full_pair_count(intervening_barrier), 0u);

  const std::array<uint32_t, 4> mismatched_id = {
      0xBE804EC1u, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4), 0xBF94FFFEu, 0xBFB00000u};
  EXPECT_EQ(full_pair_count(mismatched_id), 0u);

  const std::array<uint32_t, 4> different_basic_blocks = {
      0xBE804EC1u, build_s_branch(0, ROCJITSU_CODE_ARCH_RDNA4), 0xBF94FFFFu, 0xBFB00000u};
  EXPECT_EQ(full_pair_count(different_basic_blocks), 0u);
}

TEST(ConSan, FaultDropBarrierExactGroupRewritesTwoCompletePairsAsOneMutation) {
  const std::array<uint32_t, 7> text_words = {
      0xBE804EC1u,              // first s_barrier_signal -1
      0xBF94FFFFu,              // first s_barrier_wait -1
      0xD8340000u, 0x00000000u, // ds_store_b32 separates the pairs
      0xBE804EC1u,              // second s_barrier_signal -1
      0xBF94FFFFu,              // second s_barrier_wait -1
      0xBFB00000u,              // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "exact_grouped_barrier_drop");
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  std::vector<const ConSanSyncSequence *> sequences;
  for (const ConSanSyncSequence &sequence : inventory.sync_sequences)
    if (sequence.operation == ConSanSyncOperation::BarrierFull)
      sequences.push_back(&sequence);
  ASSERT_EQ(sequences.size(), 2u);
  const auto site_for = [&](const ConSanSyncSequence &sequence) {
    return std::ranges::find_if(inventory.fault_sites, [&](const ConSanFaultSite &site) {
      return site.sync_sequence_identity == sequence.identity;
    });
  };
  const auto first_site = site_for(*sequences[0]);
  const auto second_site = site_for(*sequences[1]);
  ASSERT_NE(first_site, inventory.fault_sites.end());
  ASSERT_NE(second_site, inventory.fault_sites.end());

  ConSanOptions options = inventory_options;
  options.fault_drop_barrier = true;
  options.fault_require_exactly_one = true;
  options.fault_site_identity = first_site->identity;
  options.fault_barrier_sequence_identity = sequences[0]->identity;
  options.fault_barrier_companion_site_identity = second_site->identity;
  options.fault_barrier_companion_sequence_identity = sequences[1]->identity;
  options.fault_dry_run = true;
  const ConSanResult dry_run = try_patch_consan(bytes, options);
  ASSERT_TRUE(dry_run.errors.empty()) << testing::PrintToString(dry_run.errors);
  ASSERT_EQ(dry_run.fault_plans.size(), 1u);
  EXPECT_EQ(dry_run.planned_fault_mutations, 1u);
  ASSERT_EQ(dry_run.fault_plans.front().ordered_member_identities.size(), 4u);
  EXPECT_TRUE(dry_run.fault_plans.front().logical_sequence_identity->starts_with("barrier-group["));

  options.fault_dry_run = false;
  const ConSanResult execution = try_patch_consan(bytes, options);
  ASSERT_EQ(execution.outcome, ConSanTransformOutcome::ModifiedValid)
      << testing::PrintToString(execution.errors);
  EXPECT_TRUE(execution.final_validation_passed);
  EXPECT_EQ(execution.applied_fault_mutations, 1u);
  ASSERT_EQ(execution.patches.size(), 4u);
  EXPECT_TRUE(std::ranges::all_of(execution.patches, [](const ConSanPatchInfo &patch) {
    return patch.phase == ConSanPatchPhase::Mutation &&
           patch.kind == ConSanPatchKind::InlineBarrierNopRewrite;
  }));
  AmdGpuCodeObject patched(execution.elf_bytes.data(), execution.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const auto *text = reinterpret_cast<const uint32_t *>(patched.text_sections().front()->data());
  EXPECT_EQ(text[0], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(text[1], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(text[2], text_words[2]);
  EXPECT_EQ(text[3], text_words[3]);
  EXPECT_EQ(text[4], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(text[5], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));

  ConSanResult corrupted = execution;
  corrupted.patches.pop_back();
  const std::vector<std::string> validation_errors = validate_consan_modified_elf(bytes, corrupted);
  EXPECT_TRUE(std::ranges::any_of(validation_errors, [](const std::string &error) {
    return error.find("complete exact whole-barrier drop group") != std::string::npos;
  }));
}

TEST(ConSan, FaultDropBarrierExactGroupRejectsDuplicateReversedAndPartialGroupsWithoutWrites) {
  const std::array<uint32_t, 7> text_words = {
      0xBE804EC1u, 0xBF94FFFFu, 0xD8340000u, 0x00000000u, 0xBE804EC1u, 0xBF94FFFFu, 0xBFB00000u,
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "rejected_grouped_barrier_drop");
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  std::vector<const ConSanSyncSequence *> sequences;
  for (const ConSanSyncSequence &sequence : inventory.sync_sequences)
    if (sequence.operation == ConSanSyncOperation::BarrierFull)
      sequences.push_back(&sequence);
  ASSERT_EQ(sequences.size(), 2u);
  const auto site_for = [&](const ConSanSyncSequence &sequence) {
    return std::ranges::find_if(inventory.fault_sites, [&](const ConSanFaultSite &site) {
      return site.sync_sequence_identity == sequence.identity;
    });
  };
  const auto first_site = site_for(*sequences[0]);
  const auto second_site = site_for(*sequences[1]);
  ASSERT_NE(first_site, inventory.fault_sites.end());
  ASSERT_NE(second_site, inventory.fault_sites.end());

  ConSanOptions options = inventory_options;
  options.fault_drop_barrier = true;
  options.fault_require_exactly_one = true;
  options.fault_site_identity = first_site->identity;
  options.fault_barrier_sequence_identity = sequences[0]->identity;
  options.fault_barrier_companion_site_identity = first_site->identity;
  options.fault_barrier_companion_sequence_identity = sequences[0]->identity;
  const ConSanResult duplicate = try_patch_consan(bytes, options);
  EXPECT_FALSE(duplicate.modified);
  EXPECT_TRUE(duplicate.patches.empty());
  EXPECT_EQ(duplicate.applied_fault_mutations, 0u);

  options.fault_site_identity = second_site->identity;
  options.fault_barrier_sequence_identity = sequences[1]->identity;
  options.fault_barrier_companion_site_identity = first_site->identity;
  options.fault_barrier_companion_sequence_identity = sequences[0]->identity;
  const ConSanResult reversed = try_patch_consan(bytes, options);
  EXPECT_FALSE(reversed.modified);
  EXPECT_TRUE(reversed.patches.empty());

  options.fault_site_identity = first_site->identity;
  options.fault_barrier_sequence_identity = sequences[0]->identity;
  options.fault_barrier_companion_site_identity = "missing-site";
  options.fault_barrier_companion_sequence_identity = sequences[1]->identity;
  const ConSanResult partial = try_patch_consan(bytes, options);
  EXPECT_FALSE(partial.modified);
  EXPECT_TRUE(partial.patches.empty());

  options.fault_site_identity.clear();
  options.fault_barrier_sequence_identity.clear();
  options.fault_barrier_companion_site_identity = second_site->identity;
  options.fault_barrier_companion_sequence_identity = sequences[1]->identity;
  const ConSanResult missing_primary = try_patch_consan(bytes, options);
  EXPECT_FALSE(missing_primary.modified);
  EXPECT_TRUE(missing_primary.patches.empty());
  EXPECT_EQ(missing_primary.applied_fault_mutations, 0u);
}

TEST(ConSan, FaultInventoryAssignsStableBarrierIdentities) {
  const std::array<uint32_t, 3> text_words = {
      0xBF940000u, // s_barrier_wait
      0xBF940000u, // s_barrier_wait
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words, "stable_barriers");
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const ConSanResult first = try_patch_consan(bytes, options);
  const ConSanResult second = try_patch_consan(bytes, options);

  ASSERT_EQ(first.fault_sites.size(), 2u);
  ASSERT_EQ(second.fault_sites.size(), first.fault_sites.size());
  EXPECT_EQ(first.fault_sites[0].identity, second.fault_sites[0].identity);
  EXPECT_EQ(first.fault_sites[1].identity, second.fault_sites[1].identity);
  EXPECT_NE(first.fault_sites[0].identity, first.fault_sites[1].identity);
  EXPECT_TRUE(first.fault_sites[0].code_object_fingerprint.starts_with("fnv1a64:"));
  EXPECT_EQ(first.fault_sites[0].kind, ConSanFaultSiteKind::Barrier);
  EXPECT_EQ(first.fault_sites[0].container_name, "stable_barriers");
  EXPECT_TRUE(first.fault_sites[0].in_kernel);
  EXPECT_EQ(first.fault_sites[0].occurrence, 0u);
  EXPECT_EQ(first.fault_sites[1].occurrence, 1u);
  EXPECT_EQ(first.fault_sites[0].text_offset, 0u);
  EXPECT_EQ(first.fault_sites[1].text_offset, 4u);
  EXPECT_EQ(first.fault_sites[0].semantic_role, "barrier-wait");
  EXPECT_EQ(first.fault_sites[0].decoded_operands,
            "encoding=0xbf940000,barrier_id=0,operand_source=immediate,scope=unknown,"
            "raw_selector=-,literal_width_bits=-,literal_value=-,raw_simm16=0");
  EXPECT_NE(first.fault_sites[0].identity.find("|kernel=stable_barriers|kind=barrier|"),
            std::string::npos);
}

TEST(ConSan, FaultDropBarrierExactIdentitySupersedesGlobalIndex) {
  const std::array<uint32_t, 3> text_words = {
      0xBF940000u, // s_barrier_wait
      0xBF940000u, // s_barrier_wait
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words, "exact_barriers");
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  ASSERT_EQ(std::ranges::count(inventory.fault_sites, ConSanFaultSiteKind::Barrier,
                               &ConSanFaultSite::kind),
            2u);

  ConSanOptions options = inventory_options;
  options.fault_drop_barrier = true;
  options.fault_barrier_index = 0;
  options.fault_site_identity = inventory.fault_sites[1].identity;
  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().anchor_offset, inventory.fault_sites[1].text_offset);
  EXPECT_EQ(result.patches.front().anchor_offset, 4u);
}

TEST(ConSan, FaultBarrierMoveDryRunReportsCompletingPairWithoutChangingBytes) {
  const std::array<uint32_t, 3> text_words = {
      0xBE804EC1u, // s_barrier_signal -1
      0xBF94FFFFu, // s_barrier_wait -1
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words, "dry_run_pair");
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  ASSERT_EQ(std::ranges::count(inventory.fault_sites, ConSanFaultSiteKind::Barrier,
                               &ConSanFaultSite::kind),
            2u);

  ConSanOptions options = inventory_options;
  options.fault_move_barrier = true;
  options.fault_dry_run = true;
  options.fault_site_identity = inventory.fault_sites[0].identity;
  const ConSanResult result = try_patch_consan(bytes, options);

  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
  EXPECT_TRUE(result.patches.empty());
  ASSERT_EQ(result.fault_plans.size(), 1u);
  EXPECT_EQ(result.fault_plans[0].kind, ConSanFaultMutationKind::MoveBarrierPair);
  EXPECT_EQ(result.fault_plans[0].primary_identity, inventory.fault_sites[0].identity);
  ASSERT_TRUE(result.fault_plans[0].companion_identity);
  EXPECT_EQ(*result.fault_plans[0].companion_identity, inventory.fault_sites[1].identity);
  ASSERT_TRUE(result.fault_plans[0].logical_sequence_identity);
  ASSERT_EQ(result.fault_plans[0].ordered_member_identities.size(), 2u);
  EXPECT_FALSE(result.fault_plans[0].destination_identity);
  EXPECT_EQ(result.fault_plans[0].barrier_move_direction, ConSanBarrierMoveDirection::LegacyMarker);
}

TEST(ConSan, FaultBarrierIdScopeDryRunSelectsExactLogicalSequence) {
  const std::array<uint32_t, 3> text_words = {
      0xBE804EC1u, // s_barrier_signal -1
      0xBF94FFFFu, // s_barrier_wait -1
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "barrier_id_scope_plan");
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  inventory_options.fault_dry_run = true;
  const ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  ASSERT_TRUE(inventory.errors.empty())
      << (inventory.errors.empty() ? "" : inventory.errors.front());
  ASSERT_EQ(inventory.sync_sequences.size(), 1u);
  ASSERT_EQ(std::ranges::count(inventory.fault_sites, ConSanFaultSiteKind::Barrier,
                               &ConSanFaultSite::kind),
            2u);

  ConSanOptions options = inventory_options;
  options.fault_mutate_barrier_id_scope = true;
  options.fault_barrier_sequence_identity = inventory.sync_sequences[0].identity;
  options.fault_barrier_target_id = -3;
  options.fault_require_exactly_one = true;
  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_EQ(result.requested_fault_mutations, 1u);
  EXPECT_EQ(result.planned_fault_mutations, 1u);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
  ASSERT_EQ(result.fault_plans.size(), 1u);
  const ConSanFaultMutationPlan &plan = result.fault_plans.front();
  EXPECT_EQ(plan.kind, ConSanFaultMutationKind::BarrierIdScope);
  EXPECT_EQ(plan.primary_identity, inventory.fault_sites[0].identity);
  EXPECT_EQ(plan.companion_identity, inventory.fault_sites[1].identity);
  EXPECT_EQ(plan.logical_sequence_identity, inventory.sync_sequences[0].identity);
  EXPECT_EQ(plan.ordered_member_identities, inventory.sync_sequences[0].member_event_identities);
  EXPECT_EQ(plan.original_barrier_id, -1);
  EXPECT_EQ(plan.target_barrier_id, -3);
  EXPECT_EQ(plan.original_barrier_scope, ConSanBarrierSite::Scope::Workgroup);
  EXPECT_EQ(plan.target_barrier_scope, ConSanBarrierSite::Scope::Cluster);
  const auto qualification = consan_barrier_mutation_qualification(
      "gfx1201", ConSanBarrierMutationForm::PairScopeCrossing);
  EXPECT_EQ(qualification.host, ConSanQualificationEvidence::Proven);
  EXPECT_EQ(qualification.live_gpu, ConSanQualificationEvidence::None);
  EXPECT_EQ(qualification.cluster_or_multi_device, ConSanQualificationEvidence::None);
}

TEST(ConSan, FaultBarrierIdScopePairsBoundedSignalWaitWithInterveningInstructions) {
  const std::array<uint32_t, 5> text_words = {
      0xBE804EC1u, // s_barrier_signal -1
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
      0xBF94FFFFu, // s_barrier_wait -1
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "barrier_id_scope_bounded_pair");
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.fault_dry_run = true;
  options.fault_mutate_barrier_id_scope = true;
  options.fault_barrier_target_id = -3;

  const ConSanResult discovery = try_patch_consan(bytes, options);
  const auto sequence = std::ranges::find(
      discovery.sync_sequences, ConSanSyncOperation::BarrierFull, &ConSanSyncSequence::operation);
  ASSERT_NE(sequence, discovery.sync_sequences.end()) << testing::PrintToString(discovery.warnings);

  options.fault_barrier_sequence_identity = sequence->identity;
  options.fault_require_exactly_one = true;
  const ConSanResult selected = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(selected)) << testing::PrintToString(selected.errors);
  EXPECT_EQ(selected.requested_fault_mutations, 1u);
  EXPECT_EQ(selected.planned_fault_mutations, 1u);
  ASSERT_EQ(selected.fault_plans.size(), 1u);
  EXPECT_EQ(selected.fault_plans.front().kind, ConSanFaultMutationKind::BarrierIdScope);

  options.fault_dry_run = false;
  const ConSanResult applied = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(applied)) << testing::PrintToString(applied.errors);
  EXPECT_TRUE(applied.modified);
  EXPECT_EQ(applied.applied_fault_mutations, 1u);
  EXPECT_TRUE(applied.final_validation_passed);
}

TEST(ConSan, FaultBarrierIdScopeDryRunRejectsInexactAndInvalidTargets) {
  const std::array<uint32_t, 3> text_words = {
      0xBE804EC1u, // s_barrier_signal -1
      0xBF94FFFFu, // s_barrier_wait -1
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  inventory_options.fault_dry_run = true;
  const ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  ASSERT_EQ(inventory.sync_sequences.size(), 1u);

  ConSanOptions options = inventory_options;
  options.fault_mutate_barrier_id_scope = true;
  options.fault_require_exactly_one = true;
  options.fault_barrier_target_id = -3;
  const ConSanResult missing_identity = try_patch_consan(bytes, options);
  EXPECT_TRUE(missing_identity.fault_plans.empty());
  EXPECT_FALSE(missing_identity.errors.empty());
  EXPECT_NE(missing_identity.warnings.front().find("exact logical sequence identity"),
            std::string::npos);

  options.fault_barrier_sequence_identity = "missing-sequence";
  const ConSanResult wrong_identity = try_patch_consan(bytes, options);
  EXPECT_TRUE(wrong_identity.fault_plans.empty());
  EXPECT_FALSE(wrong_identity.errors.empty());

  options.fault_barrier_sequence_identity = inventory.sync_sequences[0].identity;
  options.fault_barrier_target_id = -1;
  const ConSanResult unchanged = try_patch_consan(bytes, options);
  EXPECT_TRUE(unchanged.fault_plans.empty());
  EXPECT_FALSE(unchanged.errors.empty());

  options.fault_barrier_target_id = 0;
  const ConSanResult unknown_scope = try_patch_consan(bytes, options);
  EXPECT_TRUE(unknown_scope.fault_plans.empty());
  EXPECT_FALSE(unknown_scope.errors.empty());
}

TEST(ConSan, FaultBarrierIdScopeRewritesInlinePairAsOneMutation) {
  const std::array<uint32_t, 3> text_words = {
      0xBE804EC1u, // s_barrier_signal -1
      0xBF94FFFFu, // s_barrier_wait -1
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  inventory_options.fault_dry_run = true;
  const ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  ASSERT_EQ(inventory.sync_sequences.size(), 1u);

  ConSanOptions options = inventory_options;
  options.fault_dry_run = false;
  options.fault_mutate_barrier_id_scope = true;
  options.fault_barrier_target_id = -3;
  options.fault_barrier_sequence_identity = inventory.sync_sequences[0].identity;
  options.fault_require_exactly_one = true;
  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  EXPECT_FALSE(result.elf_bytes.empty());
  EXPECT_EQ(result.requested_fault_mutations, 1u);
  EXPECT_EQ(result.applied_fault_mutations, 1u);
  EXPECT_TRUE(result.final_validation_passed);
  ASSERT_EQ(result.patches.size(), 2u);
  EXPECT_EQ(result.patches[0].kind, ConSanPatchKind::InlineBarrierIdScopeRewrite);
  EXPECT_EQ(result.patches[1].kind, ConSanPatchKind::InlineBarrierIdScopeRewrite);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const auto *text = reinterpret_cast<const uint32_t *>(patched.text_sections().front()->data());
  EXPECT_EQ(text[0], 0xBE804EC3u); // s_barrier_signal -3
  EXPECT_EQ(text[1], 0xBF94FFFDu); // s_barrier_wait -3
  EXPECT_EQ(text[2], text_words[2]);
  const auto qualification = consan_barrier_mutation_qualification(
      "gfx1201", ConSanBarrierMutationForm::PairScopeCrossing);
  EXPECT_EQ(qualification.host, ConSanQualificationEvidence::Proven);
  EXPECT_EQ(qualification.live_gpu, ConSanQualificationEvidence::None);
}

TEST(ConSan, FaultBarrierIdScopePreservesLiteral32SignalEncoding) {
  const std::array<uint32_t, 4> text_words = {
      0xBE804EFFu,
      0xFFFFFFFFu, // s_barrier_signal literal32(-1)
      0xBF94FFFFu, // s_barrier_wait -1
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_gfx1250_code_object(text_words);
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  inventory_options.fault_dry_run = true;
  const ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  ASSERT_EQ(inventory.sync_sequences.size(), 1u);

  ConSanOptions options = inventory_options;
  options.fault_dry_run = false;
  options.fault_mutate_barrier_id_scope = true;
  options.fault_barrier_target_id = -3;
  options.fault_barrier_sequence_identity = inventory.sync_sequences[0].identity;
  options.fault_require_exactly_one = true;
  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_EQ(result.applied_fault_mutations, 1u);
  EXPECT_TRUE(result.final_validation_passed);
  ASSERT_EQ(result.patches.size(), 2u);
  EXPECT_EQ(result.patches[0].original_size, 8u);
  EXPECT_EQ(result.patches[1].original_size, 4u);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const auto *text = reinterpret_cast<const uint32_t *>(patched.text_sections().front()->data());
  EXPECT_EQ(text[0], text_words[0]);
  EXPECT_EQ(text[1], 0xFFFFFFFDu);
  EXPECT_EQ(text[2], 0xBF94FFFDu);
  EXPECT_EQ(text[3], text_words[3]);
}

TEST(ConSan, FaultBarrierIdScopeExecutionRejectsUnsupportedForms) {
  const std::array<uint32_t, 3> dynamic_words = {
      0xBE804E7Du, // s_barrier_signal m0
      0xBF94FFFFu, // s_barrier_wait -1
      0xBFB00000u, // s_endpgm
  };
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.fault_mutate_barrier_id_scope = true;
  options.fault_barrier_target_id = -3;
  options.fault_barrier_sequence_identity = "not-a-qualified-sequence";
  const ConSanResult dynamic = try_patch_consan(make_rdna4_lds_code_object(dynamic_words), options);
  EXPECT_FALSE(dynamic.errors.empty());
  EXPECT_FALSE(dynamic.modified);
  EXPECT_EQ(dynamic.applied_fault_mutations, 0u);
}

TEST(ConSan, FaultBarrierMoveDryRunPlansStableRealEarlierAndLaterDestinations) {
  const std::array<uint32_t, 7> text_words = {
      0xD8340000u, 0x00000000u, // ds_store_b32
      0xBE804EC1u,              // s_barrier_signal -1
      0xBF94FFFFu,              // s_barrier_wait -1
      0xD8D80000u, 0x00000000u, // ds_load_b32
      0xBFB00000u,              // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "markerless_destinations");
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  const ConSanResult repeated = try_patch_consan(bytes, inventory_options);
  ASSERT_EQ(std::ranges::count(inventory.fault_sites, ConSanFaultSiteKind::Barrier,
                               &ConSanFaultSite::kind),
            2u);
  ASSERT_EQ(inventory.barrier_move_destinations.size(), repeated.barrier_move_destinations.size());

  const auto earlier = std::ranges::find(inventory.barrier_move_destinations, 0u,
                                         &ConSanBarrierMoveDestination::text_offset);
  const auto later = std::ranges::find(inventory.barrier_move_destinations, 16u,
                                       &ConSanBarrierMoveDestination::text_offset);
  ASSERT_NE(earlier, inventory.barrier_move_destinations.end());
  ASSERT_NE(later, inventory.barrier_move_destinations.end());
  EXPECT_TRUE(earlier->suitable);
  EXPECT_TRUE(later->suitable);
  const auto repeated_earlier = std::ranges::find(repeated.barrier_move_destinations, 0u,
                                                  &ConSanBarrierMoveDestination::text_offset);
  ASSERT_NE(repeated_earlier, repeated.barrier_move_destinations.end());
  EXPECT_EQ(repeated_earlier->identity, earlier->identity);

  const auto plan = [&](ConSanBarrierMoveDirection direction,
                        const ConSanBarrierMoveDestination &destination) {
    ConSanOptions options = inventory_options;
    options.fault_move_barrier = true;
    options.fault_dry_run = true;
    options.fault_site_identity = inventory.fault_sites[0].identity;
    options.fault_barrier_move_direction = direction;
    options.fault_barrier_destination_identity = destination.identity;
    return try_patch_consan(bytes, options);
  };
  const ConSanResult early_plan = plan(ConSanBarrierMoveDirection::Earlier, *earlier);
  const ConSanResult late_plan = plan(ConSanBarrierMoveDirection::Later, *later);
  for (const auto &[result, direction, destination] :
       {std::tuple<const ConSanResult &, ConSanBarrierMoveDirection,
                   const ConSanBarrierMoveDestination &>{
            early_plan, ConSanBarrierMoveDirection::Earlier, *earlier},
        std::tuple<const ConSanResult &, ConSanBarrierMoveDirection,
                   const ConSanBarrierMoveDestination &>{
            late_plan, ConSanBarrierMoveDirection::Later, *later}}) {
    ASSERT_EQ(result.fault_plans.size(), 1u);
    const ConSanFaultMutationPlan &mutation = result.fault_plans.front();
    ASSERT_TRUE(mutation.logical_sequence_identity);
    EXPECT_EQ(*mutation.logical_sequence_identity,
              *inventory.fault_sites[0].sync_sequence_identity);
    ASSERT_EQ(mutation.ordered_member_identities.size(), 2u);
    EXPECT_EQ(mutation.ordered_member_identities[0], *inventory.fault_sites[0].sync_event_identity);
    EXPECT_EQ(mutation.ordered_member_identities[1], *inventory.fault_sites[1].sync_event_identity);
    ASSERT_TRUE(mutation.destination_identity);
    EXPECT_EQ(*mutation.destination_identity, destination.identity);
    EXPECT_EQ(mutation.barrier_move_direction, direction);
  }
}

TEST(ConSan, FaultBarrierMoveDryRunRejectsOverlapBoundaryUnsuitableAndNoTarget) {
  const std::array<uint32_t, 8> text_words = {
      0xD8340000u,
      0x00000000u,                                 // ds_store_b32
      build_s_branch(0, ROCJITSU_CODE_ARCH_RDNA4), // block boundary
      0x06040F06u,                                 // v_add_f32_e32
      0xBE804EC1u,                                 // s_barrier_signal -1
      0xBF94FFFFu,                                 // s_barrier_wait -1
      0xD8D80000u,
      0x00000000u, // ds_load_b32
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "rejected_destinations");
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  ASSERT_EQ(std::ranges::count(inventory.fault_sites, ConSanFaultSiteKind::Barrier,
                               &ConSanFaultSite::kind),
            2u);
  const auto boundary = std::ranges::find(inventory.barrier_move_destinations, 0u,
                                          &ConSanBarrierMoveDestination::text_offset);
  const auto unsuitable = std::ranges::find(inventory.barrier_move_destinations, 12u,
                                            &ConSanBarrierMoveDestination::text_offset);
  const auto overlap = std::ranges::find(inventory.barrier_move_destinations, 16u,
                                         &ConSanBarrierMoveDestination::text_offset);
  ASSERT_NE(boundary, inventory.barrier_move_destinations.end());
  ASSERT_NE(unsuitable, inventory.barrier_move_destinations.end());
  ASSERT_NE(overlap, inventory.barrier_move_destinations.end());

  const auto reject = [&](std::string identity) {
    ConSanOptions options = inventory_options;
    options.fault_move_barrier = true;
    options.fault_dry_run = true;
    options.fault_site_identity = inventory.fault_sites[0].identity;
    options.fault_barrier_move_direction = ConSanBarrierMoveDirection::Earlier;
    options.fault_barrier_destination_identity = std::move(identity);
    return try_patch_consan(bytes, options);
  };
  const ConSanResult boundary_result = reject(boundary->identity);
  const ConSanResult unsuitable_result = reject(unsuitable->identity);
  const ConSanResult overlap_result = reject(overlap->identity);
  const ConSanResult no_target_result = reject("missing-destination");
  EXPECT_TRUE(boundary_result.fault_plans.empty());
  EXPECT_TRUE(unsuitable_result.fault_plans.empty());
  EXPECT_TRUE(overlap_result.fault_plans.empty());
  EXPECT_TRUE(no_target_result.fault_plans.empty());
  EXPECT_TRUE(std::ranges::any_of(boundary_result.warnings, [](const std::string &warning) {
    return warning.find("control-flow or container boundary") != std::string::npos;
  }));
  EXPECT_TRUE(std::ranges::any_of(unsuitable_result.warnings, [](const std::string &warning) {
    return warning.find("not-memory-operation") != std::string::npos;
  }));
  EXPECT_TRUE(std::ranges::any_of(overlap_result.warnings, [](const std::string &warning) {
    return warning.find("overlapping the selected logical barrier") != std::string::npos;
  }));
  EXPECT_TRUE(std::ranges::any_of(no_target_result.warnings, [](const std::string &warning) {
    return warning.find("no barrier-move destination") != std::string::npos;
  }));
}

TEST(ConSan, FaultBarrierMoveDryRunRejectsDestinationInsideScalarClause) {
  const std::array<uint32_t, 6> text_words = {
      0xBF850000u,              // s_clause 0: one following instruction
      0xD8340000u, 0x00000000u, // ds_store_b32
      0xBE804EC1u,              // s_barrier_signal -1
      0xBF94FFFFu,              // s_barrier_wait -1
      0xBFB00000u,              // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words, "clause_destination");
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  const auto destination = std::ranges::find(inventory.barrier_move_destinations, 4u,
                                             &ConSanBarrierMoveDestination::text_offset);
  ASSERT_NE(destination, inventory.barrier_move_destinations.end());
  EXPECT_FALSE(destination->suitable);
  EXPECT_EQ(destination->rejection_reason, "inside-s-clause");

  ConSanOptions options = inventory_options;
  options.fault_move_barrier = true;
  options.fault_dry_run = true;
  options.fault_site_identity = inventory.fault_sites[0].identity;
  options.fault_barrier_move_direction = ConSanBarrierMoveDirection::Earlier;
  options.fault_barrier_destination_identity = destination->identity;
  const ConSanResult result = try_patch_consan(bytes, options);
  EXPECT_TRUE(result.fault_plans.empty());
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("inside-s-clause") != std::string::npos;
  }));
}

TEST(ConSan, FaultBarrierMarkerlessExecutionRelocatesExactPairEarlierAndLater) {
  const std::array<uint32_t, 7> text_words = {
      0xD8340000u, 0x00000000u, // ds_store_b32
      0xBE804EC1u,              // s_barrier_signal -1
      0xBF94FFFFu,              // s_barrier_wait -1
      0xD8D80000u, 0x00000000u, // ds_load_b32
      0xBFB00000u,              // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  for (const auto &[direction, destination_offset] :
       {std::pair{ConSanBarrierMoveDirection::Earlier, 0u},
        std::pair{ConSanBarrierMoveDirection::Later, 16u}}) {
    const auto destination =
        std::ranges::find(inventory.barrier_move_destinations, destination_offset,
                          &ConSanBarrierMoveDestination::text_offset);
    ASSERT_NE(destination, inventory.barrier_move_destinations.end());

    ConSanOptions options = inventory_options;
    options.fault_move_barrier = true;
    options.fault_site_identity = inventory.fault_sites[0].identity;
    options.fault_barrier_move_direction = direction;
    options.fault_barrier_destination_identity = destination->identity;
    const ConSanResult result = try_patch_consan(bytes, options);
    ASSERT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid)
        << (result.errors.empty() ? "" : result.errors.front());
    ASSERT_EQ(result.patches.size(), 3u);
    EXPECT_EQ(result.applied_fault_mutations, 1u);
    EXPECT_EQ(result.patches[0].anchor_offset, 8u);
    EXPECT_EQ(result.patches[1].anchor_offset, 12u);
    const ConSanPatchInfo &target = result.patches[2];
    EXPECT_EQ(target.anchor_offset, destination_offset);
    EXPECT_EQ(target.original_size, 8u);
    EXPECT_EQ(target.trampoline_size, 20u);
    ASSERT_TRUE(target.barrier_move_direction);
    EXPECT_EQ(*target.barrier_move_direction, direction);

    AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_EQ(patched.text_sections().size(), 1u);
    std::array<uint32_t, 5> cave{};
    std::memcpy(cave.data(), patched.text_sections().front()->data() + target.trampoline_offset,
                sizeof(cave));
    const std::array<uint32_t, 2> pair = {text_words[2], text_words[3]};
    const std::array<uint32_t, 2> displaced = destination_offset == 0u
                                                  ? std::array{text_words[0], text_words[1]}
                                                  : std::array{text_words[4], text_words[5]};
    if (direction == ConSanBarrierMoveDirection::Earlier) {
      EXPECT_TRUE(std::equal(pair.begin(), pair.end(), cave.begin()));
      EXPECT_TRUE(std::equal(displaced.begin(), displaced.end(), cave.begin() + 2));
    } else {
      EXPECT_TRUE(std::equal(displaced.begin(), displaced.end(), cave.begin()));
      EXPECT_TRUE(std::equal(pair.begin(), pair.end(), cave.begin() + 2));
    }
  }
}

TEST(ConSan, FaultBarrierMarkerlessExecutionPreservesTwelveByteDestination) {
  const std::array<uint32_t, 6> text_words = {
      0xEE158004u, 0x00980000u,
      0x00000002u, // global_atomic_add_f32 (12 bytes)
      0xBE804EC1u, // s_barrier_signal -1
      0xBF94FFFFu, // s_barrier_wait -1
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words, "wide_destination");
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  const auto destination = std::ranges::find(inventory.barrier_move_destinations, 0u,
                                             &ConSanBarrierMoveDestination::text_offset);
  ASSERT_NE(destination, inventory.barrier_move_destinations.end());
  ASSERT_TRUE(destination->suitable) << destination->rejection_reason;
  ASSERT_EQ(destination->size, 12u);

  ConSanOptions options = inventory_options;
  options.fault_move_barrier = true;
  options.fault_site_identity = inventory.fault_sites[0].identity;
  options.fault_barrier_move_direction = ConSanBarrierMoveDirection::Earlier;
  options.fault_barrier_destination_identity = destination->identity;
  const ConSanResult result = try_patch_consan(bytes, options);
  ASSERT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid)
      << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_EQ(result.patches.size(), 3u);
  EXPECT_EQ(result.patches.back().original_size, 12u);
  EXPECT_EQ(result.patches.back().trampoline_size, 24u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  std::array<uint32_t, 6> cave{};
  std::memcpy(cave.data(),
              patched.text_sections().front()->data() + result.patches.back().trampoline_offset,
              sizeof(cave));
  EXPECT_EQ(cave[0], text_words[3]);
  EXPECT_EQ(cave[1], text_words[4]);
  EXPECT_TRUE(std::equal(text_words.begin(), text_words.begin() + 3, cave.begin() + 2));
}

TEST(ConSan, FaultBarrierConditionalMoveAdmitsProvenCompletingStructuredDiamond) {
  const std::array<uint32_t, 10> text_words = {
      0xD8340000u,
      0x00000000u, // guard-block ds_store_b32 destination
      pack_sopp(/*s_cbranch_scc0=*/33, /*simm16=*/2),
      build_v_mov_b32_e32(/*vdst=*/1, vector_source_vgpr(1), ROCJITSU_CODE_ARCH_RDNA4),
      pack_sopp(/*s_branch=*/32, /*simm16=*/2),
      build_v_mov_b32_e32(/*vdst=*/2, vector_source_vgpr(2), ROCJITSU_CODE_ARCH_RDNA4),
      pack_sopp(/*s_branch=*/32, /*simm16=*/0),
      0xBE804EC1u, // reconverged s_barrier_signal -1
      0xBF94FFFFu, // s_barrier_wait -1
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "completing_conditional_move");
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  ASSERT_EQ(std::ranges::count(inventory.fault_sites, ConSanFaultSiteKind::Barrier,
                               &ConSanFaultSite::kind),
            2u);
  const auto destination = std::ranges::find(inventory.barrier_move_destinations, 0u,
                                             &ConSanBarrierMoveDestination::text_offset);
  ASSERT_NE(destination, inventory.barrier_move_destinations.end());
  ASSERT_TRUE(destination->suitable) << destination->rejection_reason;
  EXPECT_EQ(destination->cfg_contract, ConSanBarrierMoveCfgContract::CompletingStructuredDiamond);
  ASSERT_TRUE(destination->structured_guard_block_index);
  ASSERT_TRUE(destination->structured_source_block_index);
  EXPECT_EQ(destination->structured_guard_offset, 8u);
  EXPECT_EQ(destination->structured_source_offset, 28u);

  ConSanOptions options = inventory_options;
  options.fault_move_barrier = true;
  options.fault_dry_run = true;
  options.fault_site_identity = inventory.fault_sites.front().identity;
  options.fault_barrier_move_direction = ConSanBarrierMoveDirection::Earlier;
  options.fault_barrier_destination_identity = destination->identity;
  EXPECT_TRUE(try_patch_consan(bytes, options).fault_plans.empty());

  // The destructive divergence opt-in cannot authorize a completing contract.
  options.fault_allow_destructive_divergent_barrier_move = true;
  EXPECT_TRUE(try_patch_consan(bytes, options).fault_plans.empty());
  options.fault_allow_destructive_divergent_barrier_move = false;
  options.fault_allow_completing_conditional_barrier_move = true;
  const ConSanResult dry_run = try_patch_consan(bytes, options);
  ASSERT_EQ(dry_run.fault_plans.size(), 1u);
  const ConSanFaultMutationPlan &plan = dry_run.fault_plans.front();
  EXPECT_EQ(plan.barrier_move_cfg_contract,
            ConSanBarrierMoveCfgContract::CompletingStructuredDiamond);
  EXPECT_EQ(plan.structured_guard_block_index, destination->structured_guard_block_index);
  EXPECT_EQ(plan.structured_destination_block_index, destination->basic_block_index);
  EXPECT_EQ(plan.structured_source_block_index, destination->structured_source_block_index);

  options.fault_dry_run = false;
  const ConSanResult result = try_patch_consan(bytes, options);
  ASSERT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid)
      << testing::PrintToString(result.errors);
  ASSERT_EQ(result.patches.size(), 3u);
  const ConSanPatchInfo &target = result.patches.back();
  EXPECT_EQ(target.barrier_move_cfg_contract,
            ConSanBarrierMoveCfgContract::CompletingStructuredDiamond);
  EXPECT_EQ(target.structured_guard_offset, 8u);
  EXPECT_EQ(target.structured_destination_offset, 0u);
  EXPECT_EQ(target.structured_source_offset, 28u);

  ConSanResult wrong_contract = result;
  wrong_contract.patches.back().barrier_move_cfg_contract =
      ConSanBarrierMoveCfgContract::DestructiveStructuredExecDiamond;
  const auto proof_errors = validate_consan_modified_elf(bytes, wrong_contract);
  EXPECT_TRUE(std::ranges::any_of(proof_errors, [](const std::string &error) {
    return error.find("could not rederive its structured CFG contract") != std::string::npos;
  }));

  auto cyclic_words = text_words;
  cyclic_words[6] = pack_sopp(/*s_branch=*/32, /*simm16=*/-1);
  const ConSanResult cyclic_inventory = try_patch_consan(
      make_rdna4_lds_code_object(cyclic_words, "cyclic_conditional_move"), inventory_options);
  const auto cyclic_destination = std::ranges::find(cyclic_inventory.barrier_move_destinations, 0u,
                                                    &ConSanBarrierMoveDestination::text_offset);
  ASSERT_NE(cyclic_destination, cyclic_inventory.barrier_move_destinations.end());
  EXPECT_EQ(cyclic_destination->cfg_contract, ConSanBarrierMoveCfgContract::SameBlock);
}

TEST(ConSan, FaultBarrierDivergentMoveAdmitsOnlyProvenStructuredExecDiamond) {
  const auto save_exec =
      build_s_and_saveexec_b64(/*sdst=*/30, /*ssrc0=*/106, ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_exec = build_s_mov_b64(/*sdst=*/126, /*ssrc0=*/30, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(save_exec);
  ASSERT_TRUE(restore_exec);
  const std::array<uint32_t, 8> text_words = {
      *save_exec,    pack_sopp(/*s_cbranch_execz=*/37, /*simm16=*/2), 0xD8340000u,
      0x00000000u, // optional-arm ds_store_b32
      *restore_exec,
      0xBE804EC1u, // reconverged s_barrier_signal -1
      0xBF94FFFFu, // s_barrier_wait -1
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "structured_divergent_move");
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  ASSERT_EQ(std::ranges::count(inventory.fault_sites, ConSanFaultSiteKind::Barrier,
                               &ConSanFaultSite::kind),
            2u);
  const auto destination = std::ranges::find(inventory.barrier_move_destinations, 8u,
                                             &ConSanBarrierMoveDestination::text_offset);
  ASSERT_NE(destination, inventory.barrier_move_destinations.end());
  ASSERT_TRUE(destination->suitable) << destination->rejection_reason;
  EXPECT_EQ(destination->cfg_contract,
            ConSanBarrierMoveCfgContract::DestructiveStructuredExecDiamond);
  ASSERT_TRUE(destination->structured_guard_block_index);
  ASSERT_TRUE(destination->structured_source_block_index);
  EXPECT_EQ(*destination->structured_guard_offset, 4u);
  EXPECT_EQ(*destination->structured_source_offset, 16u);

  ConSanOptions options = inventory_options;
  options.fault_move_barrier = true;
  options.fault_dry_run = true;
  options.fault_site_identity = inventory.fault_sites.front().identity;
  options.fault_barrier_move_direction = ConSanBarrierMoveDirection::Earlier;
  options.fault_barrier_destination_identity = destination->identity;
  const ConSanResult contained = try_patch_consan(bytes, options);
  EXPECT_TRUE(contained.fault_plans.empty());
  EXPECT_TRUE(std::ranges::any_of(contained.warnings, [](const std::string &warning) {
    return warning.find("matching proven completing or destructive structured diamond opt-in") !=
           std::string::npos;
  }));

  options.fault_allow_completing_conditional_barrier_move = true;
  EXPECT_TRUE(try_patch_consan(bytes, options).fault_plans.empty());
  options.fault_allow_completing_conditional_barrier_move = false;
  options.fault_allow_destructive_divergent_barrier_move = true;
  const ConSanResult dry_run = try_patch_consan(bytes, options);
  ASSERT_EQ(dry_run.fault_plans.size(), 1u);
  const ConSanFaultMutationPlan &plan = dry_run.fault_plans.front();
  EXPECT_EQ(plan.barrier_move_cfg_contract,
            ConSanBarrierMoveCfgContract::DestructiveStructuredExecDiamond);
  EXPECT_EQ(plan.structured_guard_block_index, destination->structured_guard_block_index);
  EXPECT_EQ(plan.structured_destination_block_index, destination->basic_block_index);
  EXPECT_EQ(plan.structured_source_block_index, destination->structured_source_block_index);
  EXPECT_EQ(plan.structured_guard_offset, 4u);
  EXPECT_EQ(plan.structured_destination_offset, 8u);
  EXPECT_EQ(plan.structured_source_offset, 16u);

  options.fault_dry_run = false;
  const ConSanResult result = try_patch_consan(bytes, options);
  ASSERT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid)
      << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_EQ(result.patches.size(), 3u);
  const ConSanPatchInfo &target = result.patches.back();
  EXPECT_EQ(target.barrier_move_cfg_contract,
            ConSanBarrierMoveCfgContract::DestructiveStructuredExecDiamond);
  EXPECT_EQ(target.structured_guard_block_index, destination->structured_guard_block_index);
  EXPECT_EQ(target.structured_destination_block_index, destination->basic_block_index);
  EXPECT_EQ(target.structured_source_block_index, destination->structured_source_block_index);
  EXPECT_EQ(target.structured_guard_offset, 4u);
  EXPECT_EQ(target.structured_destination_offset, 8u);
  EXPECT_EQ(target.structured_source_offset, 16u);
}

TEST(ConSan, FaultBarrierDivergentMoveRejectsLaterAndBrokenExecRestore) {
  const auto save_exec =
      build_s_and_saveexec_b64(/*sdst=*/30, /*ssrc0=*/106, ROCJITSU_CODE_ARCH_RDNA4);
  const auto wrong_restore = build_s_mov_b64(/*sdst=*/126, /*ssrc0=*/32, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(save_exec);
  ASSERT_TRUE(wrong_restore);
  const std::array<uint32_t, 8> text_words = {
      *save_exec,     pack_sopp(/*s_cbranch_execz=*/37, /*simm16=*/2),
      0xD8340000u,    0x00000000u, // optional-arm ds_store_b32
      *wrong_restore, 0xBE804EC1u,
      0xBF94FFFFu,    0xBFB00000u,
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "broken_structured_divergent_move");
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  const auto destination = std::ranges::find(inventory.barrier_move_destinations, 8u,
                                             &ConSanBarrierMoveDestination::text_offset);
  ASSERT_NE(destination, inventory.barrier_move_destinations.end());
  EXPECT_FALSE(destination->structured_guard_block_index);

  ConSanOptions options = inventory_options;
  options.fault_move_barrier = true;
  options.fault_dry_run = true;
  options.fault_allow_destructive_divergent_barrier_move = true;
  options.fault_site_identity = inventory.fault_sites.front().identity;
  options.fault_barrier_move_direction = ConSanBarrierMoveDirection::Earlier;
  options.fault_barrier_destination_identity = destination->identity;
  EXPECT_TRUE(try_patch_consan(bytes, options).fault_plans.empty());

  options.fault_barrier_move_direction = ConSanBarrierMoveDirection::Later;
  EXPECT_TRUE(try_patch_consan(bytes, options).fault_plans.empty());
}

TEST(ConSan, FaultBarrierDivergentMoveAdmitsWave32CmpxExecDiamond) {
  const std::array<uint32_t, 9> text_words = {
      0xBE82007Eu, // s_mov_b32 s2, exec_lo
      0x7D9800A0u, // v_cmpx_gt_u32_e32 32, v0
      pack_sopp(/*s_cbranch_execz=*/37, /*simm16=*/2),
      0xD8340000u,
      0x00000102u, // optional-arm ds_store_b32
      0x8C7E027Eu, // s_or_b32 exec_lo, exec_lo, s2
      0xBE804EC1u, // reconverged s_barrier_signal -1
      0xBF94FFFFu, // s_barrier_wait -1
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "wave32_cmpx_divergent_move");
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult inventory = try_patch_consan(bytes, options);
  const auto destination = std::ranges::find(inventory.barrier_move_destinations, 12u,
                                             &ConSanBarrierMoveDestination::text_offset);
  ASSERT_NE(destination, inventory.barrier_move_destinations.end());
  ASSERT_TRUE(destination->structured_guard_block_index);
  ASSERT_TRUE(destination->structured_source_block_index);
  EXPECT_EQ(destination->structured_guard_offset, 8u);
  EXPECT_EQ(destination->structured_source_offset, 20u);
}

TEST(ConSan, FaultBarrierMarkerlessExecutionPrefersReachableUncoveredLocalCave) {
  const std::array<uint32_t, 7> kernel_words = {
      0xD8340000u, 0x00000000u, // ds_store_b32
      0xBE804EC1u,              // s_barrier_signal -1
      0xBF94FFFFu,              // s_barrier_wait -1
      0xD8D80000u, 0x00000000u, // ds_load_b32
      0xBFB00000u,              // s_endpgm
  };
  const std::array<uint32_t, 1> function_words = {0xBFB00000u};
  const std::array<uint32_t, 5> tail_words = {
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4), build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4), build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words, tail_words);
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  inventory_options.test_kernel_name_filter = "lds_probe";
  const ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  ASSERT_EQ(std::ranges::count(inventory.fault_sites, ConSanFaultSiteKind::Barrier,
                               &ConSanFaultSite::kind),
            2u);
  const auto destination = std::ranges::find(inventory.barrier_move_destinations, 16u,
                                             &ConSanBarrierMoveDestination::text_offset);
  ASSERT_NE(destination, inventory.barrier_move_destinations.end());

  ConSanOptions options = inventory_options;
  options.fault_move_barrier = true;
  options.fault_site_identity = inventory.fault_sites[0].identity;
  options.fault_barrier_move_direction = ConSanBarrierMoveDirection::Later;
  options.fault_barrier_destination_identity = destination->identity;
  const ConSanResult result = try_patch_consan(bytes, options);
  ASSERT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid)
      << (result.errors.empty() ? "" : result.errors.front());
  ASSERT_EQ(result.patches.size(), 3u);
  EXPECT_EQ(result.patches.back().trampoline_offset,
            (kernel_words.size() + function_words.size()) * sizeof(uint32_t));

  AmdGpuCodeObject original(bytes.data(), bytes.size());
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  EXPECT_EQ(patched.text_sections().front()->size(), original.text_sections().front()->size());
}

TEST(ConSan, FaultBarrierMarkerlessExecutionReportsUnreachableCaveAsUnsupported) {
  std::vector<uint32_t> text_words(40000u, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000000u; // ds_store_b32
  text_words[2] = 0xBE804EC1u;
  text_words[3] = 0xBF94FFFFu;
  text_words.back() = 0xBFB00000u;
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words, "unreachable_cave");
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  ASSERT_EQ(std::ranges::count(inventory.fault_sites, ConSanFaultSiteKind::Barrier,
                               &ConSanFaultSite::kind),
            2u);
  const auto destination = std::ranges::find(inventory.barrier_move_destinations, 0u,
                                             &ConSanBarrierMoveDestination::text_offset);
  ASSERT_NE(destination, inventory.barrier_move_destinations.end());

  ConSanOptions options = inventory_options;
  options.fault_move_barrier = true;
  options.fault_site_identity = inventory.fault_sites[0].identity;
  options.fault_barrier_move_direction = ConSanBarrierMoveDirection::Earlier;
  options.fault_barrier_destination_identity = destination->identity;
  const ConSanResult result = try_patch_consan(bytes, options);
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::Unsupported);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.errors.empty());
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("no reachable local or appended cave") != std::string::npos;
  }));
}

TEST(ConSan, FinalValidationRejectsCorruptedMarkerlessBarrierMoveComponents) {
  const std::array<uint32_t, 7> text_words = {
      0xD8340000u, 0x00000000u, // ds_store_b32
      0xBE804EC1u,              // s_barrier_signal -1
      0xBF94FFFFu,              // s_barrier_wait -1
      0xD8D80000u, 0x00000000u, // ds_load_b32
      0xBFB00000u,              // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words, "corrupt_move");
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  const auto destination = std::ranges::find(inventory.barrier_move_destinations, 16u,
                                             &ConSanBarrierMoveDestination::text_offset);
  ASSERT_NE(destination, inventory.barrier_move_destinations.end());
  ConSanOptions options = inventory_options;
  options.fault_move_barrier = true;
  options.fault_site_identity = inventory.fault_sites[0].identity;
  options.fault_barrier_move_direction = ConSanBarrierMoveDirection::Later;
  options.fault_barrier_destination_identity = destination->identity;
  const ConSanResult valid = try_patch_consan(bytes, options);
  ASSERT_EQ(valid.outcome, ConSanTransformOutcome::ModifiedValid);
  ASSERT_EQ(valid.patches.size(), 3u);
  const ConSanPatchInfo &target = valid.patches.back();
  AmdGpuCodeObject patched(valid.elf_bytes.data(), valid.elf_bytes.size());
  const uint64_t text_file_offset = patched.text_sections().front()->sectionOffset();

  const auto expect_rejected = [&](uint64_t body_offset, uint32_t replacement,
                                   std::string_view expected_error) {
    ConSanResult corrupted = valid;
    std::memcpy(corrupted.elf_bytes.data() + text_file_offset + target.trampoline_offset +
                    body_offset,
                &replacement, sizeof(replacement));
    const std::vector<std::string> errors = validate_consan_modified_elf(bytes, corrupted);
    EXPECT_TRUE(std::ranges::any_of(errors, [&](const std::string &error) {
      return error.find(expected_error) != std::string::npos;
    })) << expected_error;
  };
  const auto expect_anchor_rejected = [&](uint64_t anchor_offset, uint32_t replacement,
                                          std::string_view expected_error) {
    ConSanResult corrupted = valid;
    std::memcpy(corrupted.elf_bytes.data() + text_file_offset + anchor_offset, &replacement,
                sizeof(replacement));
    const std::vector<std::string> errors = validate_consan_modified_elf(bytes, corrupted);
    EXPECT_TRUE(std::ranges::any_of(errors, [&](const std::string &error) {
      return error.find(expected_error) != std::string::npos;
    })) << expected_error;
  };
  expect_anchor_rejected(valid.patches[0].anchor_offset, text_words[2],
                         "replace the selected barrier with s_nop 0");
  expect_anchor_rejected(target.anchor_offset, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
                         "invalid branch to the whole-pair trampoline");
  expect_anchor_rejected(target.anchor_offset + sizeof(uint32_t),
                         build_s_nop(1, ROCJITSU_CODE_ARCH_RDNA4),
                         "fully replace the displaced destination anchor");
  expect_rejected(2u * sizeof(uint32_t), build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
                  "byte-identical ordered barrier pair");
  expect_rejected(sizeof(uint32_t), 1u, "preserve the displaced destination semantics");
  expect_rejected(target.trampoline_size - sizeof(uint32_t),
                  build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4), "invalid whole-pair trampoline return");

  ConSanResult wrong_direction = valid;
  wrong_direction.patches.back().barrier_move_direction = ConSanBarrierMoveDirection::Earlier;
  const std::vector<std::string> direction_errors =
      validate_consan_modified_elf(bytes, wrong_direction);
  EXPECT_TRUE(std::ranges::any_of(direction_errors, [](const std::string &error) {
    return error.find("relocation target on the wrong side") != std::string::npos;
  }));
}

TEST(ConSan, FaultDropBarrierModeSkipsRocclrRuntimeHelpers) {
  const std::array<uint32_t, 3> text_words = {
      0xBF940000u, // s_barrier_wait
      0x06040F06u, // v_add_f32_e32 v2, v6, v7
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "__amd_rocclr_fillBufferAligned");
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.fault_drop_barrier = true;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
  EXPECT_TRUE(result.patches.empty());

  bool saw_skip_warning = false;
  for (const std::string &warning : result.warnings)
    saw_skip_warning |= warning.find("skipped ROCclr runtime helper") != std::string::npos;
  EXPECT_TRUE(saw_skip_warning);
}

TEST(ConSan, FaultDropBarrierModeComposesWithLdsCheckTrapPatch) {
  const std::array<uint32_t, 12> text_words = {
      0xD8D80000u, 0x00000000u, // ds_load_b32
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF940000u, // s_barrier_wait
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.fault_drop_barrier = true;
  options.scratch_vgpr = 5;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 2u);
  EXPECT_EQ(result.patches[0].phase, ConSanPatchPhase::Mutation);
  EXPECT_EQ(result.patches[1].phase, ConSanPatchPhase::Instrumentation);
  EXPECT_EQ(result.patches[0].kind, ConSanPatchKind::InlineBarrierNopRewrite);
  EXPECT_EQ(result.patches[1].kind, ConSanPatchKind::LocalCaveLdsLoadCheckTrap);
  EXPECT_EQ(result.patches[0].anchor_offset, 40u);
  EXPECT_GT(result.elf_bytes.size(), bytes.size());

  uint32_t rewritten_barrier = 0;
  std::memcpy(&rewritten_barrier, result.elf_bytes.data() + 0x100 + 40, sizeof(rewritten_barrier));
  EXPECT_EQ(rewritten_barrier, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));

  const ConSanPatchInfo &instrumentation = result.patches[1];
  ASSERT_GT(instrumentation.trampoline_size, 0u);
  size_t relocated_access_count = 0;
  for (uint64_t offset = instrumentation.trampoline_offset;
       offset + sizeof(uint32_t) <=
       instrumentation.trampoline_offset + instrumentation.trampoline_size;
       offset += sizeof(uint32_t)) {
    uint32_t word = 0;
    std::memcpy(&word, result.elf_bytes.data() + 0x100 + offset, sizeof(word));
    relocated_access_count += word == 0xD8D80000u;
  }
  EXPECT_EQ(relocated_access_count, 2u);
}

TEST(ConSan, FaultMoveBarrierRelocatesBarrierToMarker) {
  const std::array<uint32_t, 8> text_words = {
      0xBE804EC1u, // s_barrier_signal -1
      0xBF94FFFFu, // s_barrier_wait -1
      0xD8340000u,
      0x00000000u, // ds_store_b32
      build_s_nop(42, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_nop(43, ROCJITSU_CODE_ARCH_RDNA4),
      0x06040F06u, // v_add_f32_e32 v2, v6, v7
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  ASSERT_GE(inventory.fault_sites.size(), 2u);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.fault_move_barrier = true;
  options.fault_barrier_index = 99;
  options.fault_site_identity = inventory.fault_sites.front().identity;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 4u);
  EXPECT_EQ(result.patches[0].kind, ConSanPatchKind::InlineBarrierMoveSourceRewrite);
  EXPECT_EQ(result.patches[1].kind, ConSanPatchKind::InlineBarrierMoveSourceRewrite);
  EXPECT_EQ(result.patches[2].kind, ConSanPatchKind::InlineBarrierMoveTargetRewrite);
  EXPECT_EQ(result.patches[3].kind, ConSanPatchKind::InlineBarrierMoveTargetRewrite);
  uint32_t source = 0;
  uint32_t second_source = 0;
  uint32_t target = 0;
  uint32_t second_target = 0;
  std::memcpy(&source, result.elf_bytes.data() + 0x100, sizeof(source));
  std::memcpy(&second_source, result.elf_bytes.data() + 0x100 + 4, sizeof(second_source));
  std::memcpy(&target, result.elf_bytes.data() + 0x100 + 16, sizeof(target));
  std::memcpy(&second_target, result.elf_bytes.data() + 0x100 + 20, sizeof(second_target));
  EXPECT_EQ(source, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(second_source, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(target, 0xBE804EC1u);
  EXPECT_EQ(second_target, 0xBF94FFFFu);
}

} // namespace
} // namespace rocjitsu
