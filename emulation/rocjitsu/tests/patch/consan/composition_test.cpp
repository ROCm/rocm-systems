// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"
#include "rocjitsu/code/major_image_ownership.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_transform_memory.h"

namespace rocjitsu {
namespace {

[[nodiscard]] constexpr uint64_t ownership_mask(major_image_ownership::OwnerKind kind) {
  return uint64_t{1} << static_cast<size_t>(kind);
}

[[nodiscard]] ConSanOptions release_last_record_replay_options(bool with_atomic_fault) {
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_track_atomics = true;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 15;
  options.moi_epoch_vgpr = 16;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 0, 1, 1);
  options.fault_atomic_wrong_address = with_atomic_fault;
  options.fault_atomic_address_delta = 4;
  options.fault_require_exactly_one = with_atomic_fault;
  return options;
}

TEST(ConSanMoi, AtomicWrongAddressComposesWithReleaseLastRecordProbe) {
  const std::vector<uint8_t> bytes = make_rdna4_release_wait_no_return_bitwise_code_object();
  const ConSanOptions options = release_last_record_replay_options(/*with_atomic_fault=*/true);
  const ConSanResult valid = try_patch_consan(bytes, options);

  ASSERT_EQ(valid.outcome, ConSanTransformOutcome::ModifiedValid)
      << testing::PrintToString(valid.errors) << testing::PrintToString(valid.warnings);
  EXPECT_TRUE(valid.staged_composition_validated);
  EXPECT_TRUE(valid.final_validation_passed);
  EXPECT_EQ(valid.applied_fault_mutations, 1u);
  ASSERT_GE(valid.elf_bytes.size(), bytes.size());
  const auto mutation = std::ranges::find(
      valid.patches, ConSanPatchKind::InlineAtomicAddressRewrite, &ConSanPatchInfo::kind);
  const auto record = std::ranges::find(valid.patches, ConSanPatchKind::TrampolineMoiAtomicRecord,
                                        &ConSanPatchInfo::kind);
  ASSERT_NE(mutation, valid.patches.end());
  ASSERT_NE(record, valid.patches.end());
  ASSERT_TRUE(record->relocated_guest_instruction_offset);
  EXPECT_EQ(*record->relocated_guest_instruction_offset + mutation->original_size +
                sizeof(uint32_t),
            record->trampoline_offset + record->trampoline_size);

  AmdGpuCodeObject replacement(valid.elf_bytes.data(), valid.elf_bytes.size());
  ASSERT_TRUE(replacement.is_valid());
  const uint64_t text_file_offset = replacement.text_sections().front()->sectionOffset();
  uint32_t relocated_word2 = 0;
  std::memcpy(&relocated_word2,
              valid.elf_bytes.data() + text_file_offset +
                  *record->relocated_guest_instruction_offset + 2u * sizeof(uint32_t),
              sizeof(relocated_word2));
  EXPECT_EQ((relocated_word2 >> 8u) & 0xffffffu, 4u);

  ConSanResult negative_drift = valid;
  relocated_word2 = (relocated_word2 & 0xffu) | (0xfffffcu << 8u);
  std::memcpy(negative_drift.elf_bytes.data() + text_file_offset +
                  *record->relocated_guest_instruction_offset + 2u * sizeof(uint32_t),
              &relocated_word2, sizeof(relocated_word2));
  const std::vector<std::string> errors = validate_consan_modified_elf(bytes, negative_drift);
  EXPECT_TRUE(std::ranges::any_of(errors, [](const std::string &error) {
    return error.find("mutation proof found the wrong atomic address displacement") !=
           std::string::npos;
  })) << testing::PrintToString(errors);
}

TEST(ConSanOwnership, CompositePeakFitsAdmissionAcrossAllPhases) {
  const std::vector<uint8_t> bytes = make_rdna4_release_wait_no_return_bitwise_code_object();
  const ConSanOptions options = release_last_record_replay_options(/*with_atomic_fault=*/true);

  major_image_ownership::ScopedMeasurement measurement;
  const ConSanResult result = try_patch_consan(bytes, options);
  const major_image_ownership::Measurement observed = measurement.snapshot();

  ASSERT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid)
      << testing::PrintToString(result.errors) << testing::PrintToString(result.warnings);
  ASSERT_GE(result.elf_bytes.size(), bytes.size());
  const ConSanPatchedImageGrowthLimit exact_growth = {
      .kind = ConSanPatchedImageGrowthLimitKind::AbsoluteBytes,
      .absolute_bytes = result.elf_bytes.size() - bytes.size(),
  };
  const auto ownership_estimate =
      consan_hook::consan_transform_major_image_reservation(bytes.size(), exact_growth);
  ASSERT_TRUE(ownership_estimate);
  const auto composite_phase =
      std::ranges::find(consan_hook::kConSanTransformOwnershipPhases,
                        consan_hook::ConSanTransformOwnershipPhase::CompositeIncrementalPatch,
                        &consan_hook::ConSanTransformOwnership::phase);
  ASSERT_NE(composite_phase, consan_hook::kConSanTransformOwnershipPhases.end());
  const auto composite_reservation = consan_hook::consan_transform_phase_reservation_bytes(
      *composite_phase, bytes.size(), result.elf_bytes.size());
  ASSERT_TRUE(composite_reservation);
  EXPECT_EQ(*composite_reservation, bytes.size() + 12u * result.elf_bytes.size())
      << "modified composite transforms must retain their explicit parser-complete phase";
  EXPECT_GE(ownership_estimate->reservation_bytes, *composite_reservation);

  struct MeasuredPhase {
    major_image_ownership::Phase measured;
    consan_hook::ConSanTransformOwnershipPhase admitted;
  };
  constexpr std::array measured_phases{
      MeasuredPhase{major_image_ownership::Phase::IncrementalPatch,
                    consan_hook::ConSanTransformOwnershipPhase::IncrementalPatch},
      MeasuredPhase{major_image_ownership::Phase::CompositeIncrementalPatch,
                    consan_hook::ConSanTransformOwnershipPhase::CompositeIncrementalPatch},
      MeasuredPhase{major_image_ownership::Phase::FinalValidation,
                    consan_hook::ConSanTransformOwnershipPhase::FinalValidation},
  };
  EXPECT_FALSE(observed.overflowed);
  EXPECT_FALSE(observed.bookkeeping_error);
  // Admission is a conservative all-input upper bound, not a fixture-specific
  // utilization target. Record the representative peaks without imposing a
  // lower-bound ratio that would couple production coefficients to this ELF.
  for (const MeasuredPhase &phase : measured_phases) {
    const auto admitted =
        std::ranges::find(consan_hook::kConSanTransformOwnershipPhases, phase.admitted,
                          &consan_hook::ConSanTransformOwnership::phase);
    ASSERT_NE(admitted, consan_hook::kConSanTransformOwnershipPhases.end());
    const auto reservation = consan_hook::consan_transform_phase_reservation_bytes(
        *admitted, bytes.size(), result.elf_bytes.size());
    ASSERT_TRUE(reservation);
    const auto &observation = observed.phase(phase.measured);
    EXPECT_GT(observation.peak_bytes, 0u);
    EXPECT_LE(observation.peak_bytes, *reservation)
        << consan_hook::consan_transform_ownership_phase_name(phase.admitted);
    EXPECT_NE(observation.observed_owner_mask &
                  ownership_mask(major_image_ownership::OwnerKind::Parser),
              0u);
  }
  const auto &incremental = observed.phase(major_image_ownership::Phase::IncrementalPatch);
  EXPECT_NE(incremental.observed_owner_mask &
                ownership_mask(major_image_ownership::OwnerKind::PatcherImage),
            0u);
  const auto &composite = observed.phase(major_image_ownership::Phase::CompositeIncrementalPatch);
  EXPECT_NE(composite.observed_owner_mask &
                ownership_mask(major_image_ownership::OwnerKind::CompositeImage),
            0u);
  EXPECT_NE(composite.observed_owner_mask &
                ownership_mask(major_image_ownership::OwnerKind::CompactIndex),
            0u);
  EXPECT_NE(composite.observed_owner_mask &
                ownership_mask(major_image_ownership::OwnerKind::ReplacementBytes),
            0u);
  EXPECT_NE(composite.observed_owner_mask &
                ownership_mask(major_image_ownership::OwnerKind::TransactionImage),
            0u);
  const auto &validation = observed.phase(major_image_ownership::Phase::FinalValidation);
  EXPECT_NE(validation.observed_owner_mask &
                ownership_mask(major_image_ownership::OwnerKind::ResultImage),
            0u);
  RecordProperty("composite_peak_bytes", composite.peak_bytes);
  RecordProperty("final_validation_peak_bytes", validation.peak_bytes);
  RecordProperty("input_image_bytes", bytes.size());
  RecordProperty("replacement_image_bytes", result.elf_bytes.size());
}

TEST(ConSanOwnership, OrdinaryIncrementalPeakFitsAdmission) {
  const std::vector<uint8_t> bytes = make_rdna4_release_wait_no_return_bitwise_code_object();
  const ConSanOptions options = release_last_record_replay_options(/*with_atomic_fault=*/false);

  major_image_ownership::ScopedMeasurement measurement;
  const ConSanResult result = try_patch_consan(bytes, options);
  const major_image_ownership::Measurement observed = measurement.snapshot();

  ASSERT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid)
      << testing::PrintToString(result.errors) << testing::PrintToString(result.warnings);
  const auto phase = std::ranges::find(consan_hook::kConSanTransformOwnershipPhases,
                                       consan_hook::ConSanTransformOwnershipPhase::IncrementalPatch,
                                       &consan_hook::ConSanTransformOwnership::phase);
  ASSERT_NE(phase, consan_hook::kConSanTransformOwnershipPhases.end());
  const auto reservation = consan_hook::consan_transform_phase_reservation_bytes(
      *phase, bytes.size(), result.elf_bytes.size());
  ASSERT_TRUE(reservation);
  const auto &incremental = observed.phase(major_image_ownership::Phase::IncrementalPatch);
  EXPECT_FALSE(observed.overflowed);
  EXPECT_FALSE(observed.bookkeeping_error);
  EXPECT_GT(incremental.peak_bytes, 0u);
  EXPECT_LE(incremental.peak_bytes, *reservation);
  for (const major_image_ownership::OwnerKind kind : {
           major_image_ownership::OwnerKind::InputImage,
           major_image_ownership::OwnerKind::Parser,
           major_image_ownership::OwnerKind::ResultImage,
           major_image_ownership::OwnerKind::PatcherImage,
           major_image_ownership::OwnerKind::ReplacementBytes,
           major_image_ownership::OwnerKind::TransactionImage,
       }) {
    EXPECT_NE(incremental.observed_owner_mask & ownership_mask(kind), 0u)
        << static_cast<unsigned>(kind);
  }
  EXPECT_EQ(observed.phase(major_image_ownership::Phase::CompositeIncrementalPatch).peak_bytes, 0u);
  RecordProperty("incremental_peak_bytes", incremental.peak_bytes);
  RecordProperty("input_image_bytes", bytes.size());
  RecordProperty("replacement_image_bytes", result.elf_bytes.size());
}

TEST(ConSanMoi, AtomicWrongAddressComposesWithRetainedInlineShadowProbe) {
  const std::vector<uint8_t> bytes = make_rdna4_ordered_global_atomic_release_acquire_code_object();
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_track_atomics = true;
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 80;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.fault_atomic_wrong_address = true;
  options.fault_atomic_address_delta = 4;
  options.fault_require_exactly_one = true;
  options.max_patches = 2;

  const ConSanResult valid = try_patch_consan(bytes, options);

  ASSERT_EQ(valid.outcome, ConSanTransformOutcome::ModifiedValid)
      << testing::PrintToString(valid.errors) << testing::PrintToString(valid.warnings);
  EXPECT_TRUE(valid.staged_composition_validated);
  EXPECT_TRUE(valid.final_validation_passed);
  EXPECT_EQ(valid.applied_fault_mutations, 1u);
  const auto mutation = std::ranges::find(
      valid.patches, ConSanPatchKind::InlineAtomicAddressRewrite, &ConSanPatchInfo::kind);
  ASSERT_NE(mutation, valid.patches.end());
  const auto inline_shadow = std::ranges::find_if(valid.patches, [&](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiInlineAtomicOrdering &&
           patch.anchor_offset == mutation->anchor_offset &&
           patch.original_size == mutation->original_size;
  });
  const auto any_inline = std::ranges::find(
      valid.patches, ConSanPatchKind::TrampolineMoiInlineAtomicOrdering, &ConSanPatchInfo::kind);
  ASSERT_NE(inline_shadow, valid.patches.end())
      << "mutation anchor=" << mutation->anchor_offset << " first inline anchor="
      << (any_inline == valid.patches.end() ? UINT64_MAX : any_inline->anchor_offset);
  ASSERT_TRUE(inline_shadow->relocated_guest_instruction_offset);
  EXPECT_GE(*inline_shadow->relocated_guest_instruction_offset, inline_shadow->trampoline_offset);
  EXPECT_LE(*inline_shadow->relocated_guest_instruction_offset + mutation->original_size,
            inline_shadow->trampoline_offset + inline_shadow->trampoline_size);

  AmdGpuCodeObject replacement(valid.elf_bytes.data(), valid.elf_bytes.size());
  ASSERT_TRUE(replacement.is_valid());
  const uint64_t text_file_offset = replacement.text_sections().front()->sectionOffset();
  uint32_t relocated_word2 = 0;
  std::memcpy(&relocated_word2,
              valid.elf_bytes.data() + text_file_offset +
                  *inline_shadow->relocated_guest_instruction_offset + 2u * sizeof(uint32_t),
              sizeof(relocated_word2));
  EXPECT_EQ((relocated_word2 >> 8u) & 0xffffffu, 4u);

  ConSanResult negative_drift = valid;
  relocated_word2 = (relocated_word2 & 0xffu) | (0xfffffcu << 8u);
  std::memcpy(negative_drift.elf_bytes.data() + text_file_offset +
                  *inline_shadow->relocated_guest_instruction_offset + 2u * sizeof(uint32_t),
              &relocated_word2, sizeof(relocated_word2));
  const std::vector<std::string> errors = validate_consan_modified_elf(bytes, negative_drift);
  EXPECT_TRUE(std::ranges::any_of(errors, [](const std::string &error) {
    return error.find("mutation proof found the wrong atomic address displacement") !=
           std::string::npos;
  })) << testing::PrintToString(errors);
}

TEST(ConSanMoi, PristineAutoReportInventoryCoversLiveBarrierMoveComposition) {
  const std::array<uint32_t, 7> text_words = {
      0xD8340000u, 0x00000000u, // earlier ds_store_b32 destination
      0xBE804EC1u,              // s_barrier_signal -1
      0xBF94FFFFu,              // s_barrier_wait -1
      0xD8D80000u, 0x00000000u, // later ds_load_b32 destination
      0xBFB00000u,
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions selection_options;
  selection_options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult selection = try_patch_consan(bytes, selection_options);
  ASSERT_EQ(selection.fault_sites.size(), 2u);
  const auto destination = std::ranges::find(selection.barrier_move_destinations, 0u,
                                             &ConSanBarrierMoveDestination::text_offset);
  ASSERT_NE(destination, selection.barrier_move_destinations.end());

  ConSanOptions live_options = moi_options(ConSanMoiEngine::RecordReplay);
  live_options.moi_track_barriers = true;
  live_options.moi_track_atomics = true;
  live_options.moi_report_buffer_address = 0x123456780000ull;
  live_options.moi_report_buffer_size = 64u * 1024u * 1024u;
  live_options.max_patches = 16;
  live_options.fault_move_barrier = true;
  live_options.fault_require_exactly_one = true;
  live_options.fault_site_identity = selection.fault_sites.front().identity;
  live_options.fault_barrier_move_direction = ConSanBarrierMoveDirection::Earlier;
  live_options.fault_barrier_destination_identity = destination->identity;

  ConSanOptions pristine_options = live_options;
  pristine_options.fault_move_barrier = false;
  pristine_options.fault_require_exactly_one = false;
  pristine_options.moi_report_buffer_address.reset();
  pristine_options.moi_report_buffer_size = 0;
  const ConSanResult pristine = try_patch_consan(bytes, pristine_options);
  const ConSanMoiAutoReportInventory pristine_inventory =
      inventory_consan_moi_auto_report(pristine, pristine_options, bytes);

  const ConSanResult live = try_patch_consan(bytes, live_options);
  ASSERT_EQ(live.outcome, ConSanTransformOutcome::ModifiedValid)
      << testing::PrintToString(live.errors) << testing::PrintToString(live.warnings);
  ASSERT_EQ(live.applied_fault_mutations, 1u);
  const ConSanMoiAutoReportInventory live_inventory =
      inventory_consan_moi_auto_report(live, live_options, bytes);

  EXPECT_TRUE(consan_moi_auto_report_inventory_covers(pristine_inventory, live_inventory));
  EXPECT_EQ(pristine_inventory.access_range_count, live_inventory.access_range_count);
  EXPECT_GE(pristine_inventory.barrier_event_count, live_inventory.barrier_event_count);
  EXPECT_GE(pristine_inventory.atomic_event_count, live_inventory.atomic_event_count);
  EXPECT_GE(pristine_inventory.fence_event_count, live_inventory.fence_event_count);
}

TEST(ConSanMoi, FaultBarrierMarkerlessUncoveredLocalCaveComposesWithInlineShadow) {
  const std::array<uint32_t, 9> kernel_words = {
      0xD8340000u, 0x00000000u, // ds_store_b32
      0xBE804EC1u,              // s_barrier_signal -1
      0xBF94FFFFu,              // s_barrier_wait -1
      0xD8D80000u, 0x00000000u, // ds_load_b32
      0xD8340000u, 0x00000000u, // ds_store_b32 after the move return
      0xBFB00000u,              // s_endpgm
  };
  const std::array<uint32_t, 1> tail_function_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 5> uncovered_cave_words = {
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4), build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4), build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_code_object_with_local_function(
      kernel_words, tail_function_words, uncovered_cave_words, kRdna4Wave64AllVgprsGranulated,
      /*function_is_kernel=*/true);

  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  inventory_options.test_kernel_name_filter = "lds_probe";
  const ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  ASSERT_EQ(inventory.fault_sites.size(), 2u);
  const auto destination = std::ranges::find(inventory.barrier_move_destinations, 16u,
                                             &ConSanBarrierMoveDestination::text_offset);
  ASSERT_NE(destination, inventory.barrier_move_destinations.end());

  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_track_barriers = true;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.max_patches = 16;
  options.test_kernel_name_filter = "lds_probe";
  options.fault_move_barrier = true;
  options.fault_require_exactly_one = true;
  options.fault_site_identity = inventory.fault_sites[0].identity;
  options.fault_barrier_move_direction = ConSanBarrierMoveDirection::Later;
  options.fault_barrier_destination_identity = destination->identity;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid)
      << (result.errors.empty() ? "" : result.errors.front());
  EXPECT_TRUE(result.modified);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_EQ(result.applied_fault_mutations, 1u);
  EXPECT_TRUE(std::ranges::any_of(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.phase == ConSanPatchPhase::Mutation &&
           patch.kind == ConSanPatchKind::InlineBarrierMoveTargetRewrite;
  }));
  EXPECT_TRUE(std::ranges::any_of(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.phase == ConSanPatchPhase::Instrumentation &&
           patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  }));
  EXPECT_TRUE(std::ranges::any_of(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.phase == ConSanPatchPhase::Instrumentation &&
           patch.kind == ConSanPatchKind::TrampolineMoiInlineEpochBarrier;
  }));
  const auto post_return_candidate =
      std::ranges::find(result.moi_candidates, 24u, &ConSanMoiCandidate::text_offset);
  ASSERT_NE(post_return_candidate, result.moi_candidates.end());
  EXPECT_TRUE(post_return_candidate->kernel_descriptor_file_offset.has_value());

  AmdGpuCodeObject original(bytes.data(), bytes.size());
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_EQ(original.kernels().size(), 2u);
  ASSERT_EQ(patched.kernels().size(), 2u);
  const auto original_owner =
      std::ranges::find(original.kernels(), "lds_probe", &AmdGpuKernelInfo::name);
  const auto original_tail =
      std::ranges::find(original.kernels(), "lds_helper", &AmdGpuKernelInfo::name);
  const auto patched_owner =
      std::ranges::find(patched.kernels(), "lds_probe", &AmdGpuKernelInfo::name);
  const auto patched_tail =
      std::ranges::find(patched.kernels(), "lds_helper", &AmdGpuKernelInfo::name);
  ASSERT_NE(original_owner, original.kernels().end());
  ASSERT_NE(original_tail, original.kernels().end());
  ASSERT_NE(patched_owner, patched.kernels().end());
  ASSERT_NE(patched_tail, patched.kernels().end());
  EXPECT_EQ(patched_owner->code_size, original_owner->code_size);

  const auto mutation_target = std::ranges::find_if(result.patches, [](const auto &patch) {
    return patch.phase == ConSanPatchPhase::Mutation &&
           patch.kind == ConSanPatchKind::InlineBarrierMoveTargetRewrite;
  });
  ASSERT_NE(mutation_target, result.patches.end());
  const auto nested_instrumentation =
      std::ranges::find_if(result.patches, [&](const ConSanPatchInfo &patch) {
        return patch.phase == ConSanPatchPhase::Instrumentation &&
               patch.anchor_offset >= mutation_target->trampoline_offset &&
               patch.anchor_offset <
                   mutation_target->trampoline_offset + mutation_target->trampoline_size;
      });
  ASSERT_NE(nested_instrumentation, result.patches.end());
  size_t nested_patch_count = 0;
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.phase != ConSanPatchPhase::Instrumentation ||
        patch.anchor_offset < mutation_target->trampoline_offset ||
        patch.anchor_offset >=
            mutation_target->trampoline_offset + mutation_target->trampoline_size)
      continue;
    ++nested_patch_count;
    EXPECT_NE(std::ranges::find(patch.owner_descriptor_file_offsets,
                                original_owner->descriptor_file_offset),
              patch.owner_descriptor_file_offsets.end());
    EXPECT_EQ(std::ranges::find(patch.owner_descriptor_file_offsets,
                                original_tail->descriptor_file_offset),
              patch.owner_descriptor_file_offsets.end());
  }
  EXPECT_GE(nested_patch_count, 2u);
  const auto post_return_disposition =
      std::ranges::find(result.site_dispositions, 24u, &ConSanSiteDispositionRecord::text_offset);
  ASSERT_NE(post_return_disposition, result.site_dispositions.end());
  EXPECT_EQ(post_return_disposition->lowering_outcome, ConSanSiteLoweringOutcome::Patched)
      << "lowering reason=" << static_cast<int>(post_return_disposition->lowering_reason)
      << " resource reason=" << static_cast<int>(post_return_disposition->resource_reason);
  const auto prologue = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue;
  });
  ASSERT_NE(prologue, result.patches.end());
  EXPECT_EQ(std::ranges::count_if(result.patches,
                                  [](const ConSanPatchInfo &patch) {
                                    return patch.kind ==
                                           ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue;
                                  }),
            1);
  EXPECT_EQ(prologue->anchor_offset, original_owner->entry_text_offset);

  ConSanResult corrupted = result;
  AmdGpuCodeObject composed(corrupted.elf_bytes.data(), corrupted.elf_bytes.size());
  ASSERT_EQ(composed.text_sections().size(), 1u);
  const uint64_t text_file_offset = composed.text_sections().front()->sectionOffset();
  ASSERT_GE(nested_instrumentation->trampoline_size, sizeof(uint32_t));
  const uint32_t invalid_opcode = 0;
  std::memcpy(corrupted.elf_bytes.data() + text_file_offset +
                  nested_instrumentation->trampoline_offset +
                  nested_instrumentation->trampoline_size - sizeof(uint32_t),
              &invalid_opcode, sizeof(invalid_opcode));
  EXPECT_FALSE(validate_consan_modified_elf(bytes, corrupted).empty());
}

TEST(ConSanMoi, Gfx1250DenseInlineHostPreservesPreappliedBarrierDrop) {
  constexpr uint32_t kAccessCount = 9u;
  constexpr size_t kLargeTextWords = 33000u;
  const uint32_t filler = build_s_mov_b32(/*sdst=*/100, /*ssrc0=*/100, ROCJITSU_CODE_ARCH_GFX1250);
  std::vector<uint32_t> text_words(kLargeTextWords, filler);
  size_t cursor = 32u;
  const size_t dropped_pair_word = cursor;
  text_words[cursor++] = *build_s_barrier_signal_all(ROCJITSU_CODE_ARCH_GFX1250);
  text_words[cursor++] = *build_s_barrier_wait_all(ROCJITSU_CODE_ARCH_GFX1250);
  text_words[cursor++] = build_s_mov_b32(/*sdst=*/0, /*ssrc0=*/0, ROCJITSU_CODE_ARCH_GFX1250);
  text_words[cursor++] = build_s_mov_b32(/*sdst=*/1, /*ssrc0=*/1, ROCJITSU_CODE_ARCH_GFX1250);
  text_words[cursor++] = *build_s_barrier_signal_all(ROCJITSU_CODE_ARCH_GFX1250);
  text_words[cursor++] = *build_s_barrier_wait_all(ROCJITSU_CODE_ARCH_GFX1250);
  for (uint32_t index = 0; index < kAccessCount; ++index) {
    text_words[cursor++] = 0xD8340000u | index * sizeof(uint32_t);
    text_words[cursor++] = 0x00000000u; // ds_store_b32 v0, v0 offset:index*4
  }
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250);
  const std::vector<uint8_t> bytes =
      make_gfx1250_code_object(text_words, "gfx1250_dense_inline_fault");

  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  inventory_options.fault_drop_barrier = true;
  inventory_options.fault_dry_run = true;
  const ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  ASSERT_TRUE(inventory.errors.empty()) << testing::PrintToString(inventory.errors);
  ASSERT_EQ(inventory.sync_sequences.size(), 2u);
  ASSERT_EQ(inventory.fault_sites.size(), 4u);

  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.scratch_vgpr = 82;
  options.moi_owner_vgpr = 80;
  options.moi_epoch_vgpr = 81;
  options.moi_exec_save_sgpr = 60;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.moi_track_barriers = true;
  options.moi_track_atomics = false;
  options.max_patches = 64;
  options.fault_drop_barrier = true;
  options.fault_require_exactly_one = true;
  options.fault_site_identity = inventory.fault_sites.front().identity;
  options.fault_barrier_sequence_identity = inventory.sync_sequences.front().identity;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid)
      << testing::PrintToString(result.errors) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.staged_composition_validated);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_EQ(result.applied_fault_mutations, 1u);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::InlineBarrierNopRewrite,
                               &ConSanPatchInfo::kind),
            2u);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiExactShadowStore,
                               &ConSanPatchInfo::kind),
            kAccessCount);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiInlineEpochBarrier,
                               &ConSanPatchInfo::kind),
            1u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  std::array<uint32_t, 2> dropped{};
  std::memcpy(dropped.data(),
              patched.text_sections().front()->data() + dropped_pair_word * sizeof(uint32_t),
              sizeof(dropped));
  EXPECT_EQ(dropped[0], build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250));
  EXPECT_EQ(dropped[1], build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250));
}

} // namespace
} // namespace rocjitsu
