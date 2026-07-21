// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"

namespace rocjitsu {
namespace {

TEST(ConSanMoi, AtomicWrongAddressComposesWithReleaseLastRecordProbe) {
  const std::vector<uint8_t> bytes = make_rdna4_release_wait_no_return_bitwise_code_object();
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_track_atomics = true;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 15;
  options.moi_epoch_vgpr = 16;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 0, 1, 1);
  options.fault_atomic_wrong_address = true;
  options.fault_atomic_address_delta = 4;
  options.fault_require_exactly_one = true;

  const ConSanResult valid = try_patch_consan(bytes, options);

  ASSERT_EQ(valid.outcome, ConSanTransformOutcome::ModifiedValid)
      << testing::PrintToString(valid.errors) << testing::PrintToString(valid.warnings);
  EXPECT_TRUE(valid.staged_composition_validated);
  EXPECT_TRUE(valid.final_validation_passed);
  EXPECT_EQ(valid.applied_fault_mutations, 1u);
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
  std::vector<uint32_t> text_words = {
      *build_s_barrier_signal_all(ROCJITSU_CODE_ARCH_GFX1250),
      *build_s_barrier_wait_all(ROCJITSU_CODE_ARCH_GFX1250),
  };
  text_words.resize(9u, build_s_mov_b32(/*sdst=*/0, /*ssrc0=*/0, ROCJITSU_CODE_ARCH_GFX1250));
  for (uint32_t index = 0; index < kAccessCount; ++index) {
    text_words.push_back(0xD8340000u | index * sizeof(uint32_t));
    text_words.push_back(0x00000000u); // ds_store_b32 v0, v0 offset:index*4
  }
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250));
  const std::vector<uint8_t> bytes =
      make_gfx1250_code_object(text_words, "gfx1250_dense_inline_fault");

  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  inventory_options.fault_drop_barrier = true;
  inventory_options.fault_dry_run = true;
  const ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  ASSERT_TRUE(inventory.errors.empty()) << testing::PrintToString(inventory.errors);
  ASSERT_EQ(inventory.sync_sequences.size(), 1u);
  ASSERT_EQ(inventory.fault_sites.size(), 2u);

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

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  std::array<uint32_t, 2> dropped{};
  std::memcpy(dropped.data(), patched.text_sections().front()->data(), sizeof(dropped));
  EXPECT_EQ(dropped[0], build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250));
  EXPECT_EQ(dropped[1], build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250));
}

} // namespace
} // namespace rocjitsu
