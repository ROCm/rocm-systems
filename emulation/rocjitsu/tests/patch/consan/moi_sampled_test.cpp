// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"

namespace rocjitsu {
namespace {

struct SampledCdnaTarget {
  rj_code_arch_t arch;
  std::string_view label;
};

constexpr std::array<SampledCdnaTarget, 2> kSampledCdnaTargets = {{
    {ROCJITSU_CODE_ARCH_CDNA3, "gfx942/cdna3"},
    {ROCJITSU_CODE_ARCH_CDNA4, "gfx950/cdna4"},
}};

TEST(ConSanMoi, SampledEngineInventoriesCodeObjectWithoutModification) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_FALSE(result.modified);
  EXPECT_EQ(result.flavor, ConSanFlavor::Moi);
  EXPECT_EQ(result.moi_engine, ConSanMoiEngine::Sampled);
  EXPECT_TRUE(result.elf_bytes.empty());
  ASSERT_EQ(result.kernels.size(), 1u);
  EXPECT_TRUE(result.kernels.front().decoded);
  ASSERT_EQ(result.resource_plans.size(), 2u);
  std::vector<uint16_t> scratch_counts;
  for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
    EXPECT_EQ(plan.source, ConSanRegisterAllocationSource::LivenessDead);
    EXPECT_EQ(plan.scratch_vgpr, 1);
    scratch_counts.push_back(plan.scratch_vgpr_count);
  }
  std::ranges::sort(scratch_counts);
  EXPECT_EQ(scratch_counts, (std::vector<uint16_t>{5u, 6u}));
  EXPECT_EQ(result.kernels.front().preflight_action, ConSanPreflightAction::NotRun);
  bool saw_stub_warning = false;
  for (const std::string &warning : result.warnings)
    saw_stub_warning |= warning.find("sampled") != std::string::npos;
  EXPECT_TRUE(saw_stub_warning);
}

TEST(ConSanMoi, DirectSampledProbeWritesPackedWatchpointEntry) {
  std::array<uint32_t, 420> text_words{};
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000000u; // ds_store_b32 v0, v0
  for (size_t i = 2; i + 1 < text_words.size(); ++i)
    text_words[i] = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
  text_words.back() = 0xBFB00000u; // s_endpgm

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 13;
  options.moi_epoch_vgpr = 14;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_generation = 7;
  options.moi_report_dispatch_id = 0x1122334455667788ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(2);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineMoiSampledWatchpointStore);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 8u);

  bool saw_sampled_warning = false;
  bool saw_access_warning = false;
  for (const std::string &warning : result.warnings) {
    saw_sampled_warning |= warning.find("direct sampled watchpoint") != std::string::npos;
    saw_access_warning |= warning.find("access record") != std::string::npos;
  }
  EXPECT_TRUE(saw_sampled_warning);
  EXPECT_FALSE(saw_access_warning);

  const std::vector<uint32_t> rewritten_words =
      patched_words_at_file_offset(result, 0x100, result.patches.front().original_size);
  const auto atomic_publish = build_flat_atomic_swap_b64_vaddr_vsrc_vdst(
      /*vaddr=*/8, /*vsrc=*/10, /*vdst=*/10, /*return_old_value=*/true,
      /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(atomic_publish);
  EXPECT_EQ(count_subsequence(rewritten_words, *atomic_publish), 1u);
  const auto atomic_claim = build_flat_atomic_cmpswap_b32_vaddr_vsrc_vdst(
      /*vaddr=*/8, /*vsrc=*/10, /*vdst=*/10, /*return_old_value=*/true,
      /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(atomic_claim);
  EXPECT_EQ(count_subsequence(rewritten_words, *atomic_claim), 1u);
  ASSERT_TRUE(result.resolved_moi_exec_save_sgpr);
  const auto save_guest_exec =
      build_s_mov_b64(*result.resolved_moi_exec_save_sgpr, kRdna4ExecLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_guest_exec =
      build_s_mov_b64(kRdna4ExecLo, *result.resolved_moi_exec_save_sgpr, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(save_guest_exec);
  ASSERT_TRUE(restore_guest_exec);
  EXPECT_EQ(std::count(rewritten_words.begin(), rewritten_words.end(), *save_guest_exec), 1);
  EXPECT_EQ(std::count(rewritten_words.begin(), rewritten_words.end(), *restore_guest_exec), 1);
  // Claim/publication/counters and the exact duplicate-identity loads all wait
  // before consuming returned device memory.
  EXPECT_EQ(std::count(rewritten_words.begin(), rewritten_words.end(), 0xBFC00000u), 16);
  const uint64_t causal_window = *options.moi_report_buffer_address + sizeof(ConSanMoiReportHeader);
  const auto causal_window_address = build_v_mov_b32_e64_literal(
      8, static_cast<uint32_t>(causal_window), ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(causal_window_address);
  EXPECT_TRUE(contains_subsequence(rewritten_words, *causal_window_address));
  EXPECT_EQ(
      count_subsequence(
          rewritten_words,
          make_expected_literal_offset_store_words(
              offsetof(ConSanMoiSampledCausalWindow, publication_state),
              static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Publishing), 8, 12)),
      0u);
  EXPECT_EQ(count_subsequence(
                rewritten_words,
                make_expected_literal_offset_store_words(
                    offsetof(ConSanMoiSampledCausalWindow, publication_state),
                    static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Ready), 8, 12)),
            1u);
  EXPECT_EQ(count_subsequence(rewritten_words,
                              make_expected_literal_offset_store_words(
                                  offsetof(ConSanMoiSampledCausalWindow, generation), 7, 8, 12)),
            1u);
  EXPECT_EQ(
      count_subsequence(rewritten_words, make_expected_literal_offset_store_words(
                                             offsetof(ConSanMoiSampledCausalWindow, dispatch_id),
                                             0x55667788u, 8, 12)),
      1u);
  const auto window_counter = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/8, /*vsrc=*/12, /*vdst=*/12, /*return_old_value=*/true,
      /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(window_counter);
  EXPECT_EQ(count_subsequence(rewritten_words, *window_counter), 2u);
  ASSERT_TRUE(result.resolved_moi_exec_save_sgpr);
  const uint16_t state_base = *result.resolved_moi_exec_save_sgpr;
  const auto save_scc = build_rdna4_s_cselect_b32(
      static_cast<uint16_t>(state_base + 6u), scalar_positive_inline_u32(1),
      scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_scc =
      build_rdna4_s_cmp_lg_u32(static_cast<uint16_t>(state_base + 6u),
                               scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
  const auto narrow_exec = build_s_and_saveexec_b64(static_cast<uint16_t>(state_base + 4u),
                                                    kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_exec = build_s_mov_b64(kRdna4ExecLo, static_cast<uint16_t>(state_base + 4u),
                                            ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(save_scc);
  ASSERT_TRUE(restore_scc);
  ASSERT_TRUE(narrow_exec);
  ASSERT_TRUE(restore_exec);
  EXPECT_EQ(std::count(rewritten_words.begin(), rewritten_words.end(), *save_scc), 1);
  EXPECT_EQ(std::count(rewritten_words.begin(), rewritten_words.end(), *restore_scc), 1);
  EXPECT_EQ(std::count(rewritten_words.begin(), rewritten_words.end(), *narrow_exec), 2);
  // Winner, exact duplicate, and different-identity collision paths restore
  // the original EXEC mask independently.
  EXPECT_EQ(std::count(rewritten_words.begin(), rewritten_words.end(), *restore_exec), 3);
}

TEST(ConSanMoi, Gfx1250DirectSampledProbePassesFinalValidation) {
  std::vector<uint32_t> text_words(600, build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250));
  constexpr auto store = gfx1250::build_vds(gfx1250::kDsStoreB32Vds, {.addr = 0, .data0 = 1});
  text_words[0] = store[0];
  text_words[1] = store[1];
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250);

  const std::vector<uint8_t> bytes = make_gfx1250_code_object(text_words);
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.scratch_vgpr = 80;
  options.moi_owner_vgpr = 85;
  options.moi_epoch_vgpr = 86;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_generation = 7;
  options.moi_report_dispatch_id = 0x1122334455667788ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(2);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_TRUE(result.final_validation_passed);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineMoiSampledWatchpointStore);
  EXPECT_EQ(result.patches.front().scratch_vgpr, 80u);
}

TEST(ConSanMoi, Gfx1250DenseSampledAccessesPartitionRelayWindowsAcrossLargeKernel) {
  constexpr uint32_t kAccessesPerWindow = 9u;
  constexpr uint32_t kSecondWindowWord = 65'580u;
  const uint32_t filler = build_s_mov_b32(/*sdst=*/0, /*ssrc0=*/0, ROCJITSU_CODE_ARCH_GFX1250);
  std::vector<uint32_t> text_words(8u, filler);
  for (uint32_t index = 0; index < kAccessesPerWindow; ++index) {
    text_words.push_back(0xD8340000u | index * sizeof(uint32_t));
    text_words.push_back(0x00000000u); // ds_store_b32 v0, v0 offset:index*4
  }
  text_words.resize(kSecondWindowWord, filler);
  for (uint32_t index = 0; index < kAccessesPerWindow; ++index) {
    text_words.push_back(0xD8340000u | index * sizeof(uint32_t));
    text_words.push_back(0x00000000u); // ds_store_b32 v0, v0 offset:index*4
  }
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250));
  const std::vector<uint8_t> bytes =
      make_gfx1250_code_object(text_words, "gfx1250_partitioned_dense_sampled");

  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 80;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(2u * kAccessesPerWindow);
  options.moi_runtime_sample_stride = 1u;
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.max_patches = 2u * kAccessesPerWindow;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiSampledWatchpointStore,
                               &ConSanPatchInfo::kind),
            2u * kAccessesPerWindow);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiIndirectBranchIsland,
                               &ConSanPatchInfo::kind),
            4u); // One host relay and one appended dispatcher per reachability window.
  EXPECT_TRUE(std::ranges::none_of(result.warnings, [](const std::string &warning) {
    return warning.find("inside a relocated prefix") != std::string::npos;
  }));
}

TEST(ConSanMoi, Gfx1250DenseSampledAccessesPreserveGuestVgprMsbMode) {
  constexpr uint32_t kAccessCount = 9u;
  constexpr uint16_t kGuestVgprMsbTransition = 0x4004u;
  constexpr uint8_t kGuestVgprMsbMode = 0x04u;
  std::vector<uint32_t> text_words(
      8u, build_s_mov_b32(/*sdst=*/0, /*ssrc0=*/0, ROCJITSU_CODE_ARCH_GFX1250));
  text_words.push_back(
      *build_gfx1250_s_set_vgpr_msb(kGuestVgprMsbTransition, ROCJITSU_CODE_ARCH_GFX1250));
  for (uint32_t index = 0; index < kAccessCount; ++index) {
    text_words.push_back(0xD8340000u | index * sizeof(uint32_t));
    text_words.push_back(0x00000000u); // ds_store_b32 v0, v0 offset:index*4
  }
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250));

  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 80;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(kAccessCount);
  options.moi_runtime_sample_stride = 1u;
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.max_patches = kAccessCount;

  const ConSanResult result = try_patch_consan(
      make_gfx1250_code_object(text_words, "gfx1250_dense_sampled_vgpr_msb"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const uint32_t select_low =
      *build_gfx1250_s_set_vgpr_msb_transition(kGuestVgprMsbMode, 0u, ROCJITSU_CODE_ARCH_GFX1250);
  const uint32_t restore_guest =
      *build_gfx1250_s_set_vgpr_msb_transition(0u, kGuestVgprMsbMode, ROCJITSU_CODE_ARCH_GFX1250);
  uint32_t checked = 0u;
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.kind != ConSanPatchKind::TrampolineMoiSampledWatchpointStore)
      continue;
    const std::vector<uint32_t> words =
        text_words_at_offset(patched, patch.trampoline_offset, patch.trampoline_size);
    // Four pairs cover the surrounding sampled trampoline, appended-body
    // entry/exit, embedded guest, and appended spill restore.
    EXPECT_EQ(std::ranges::count(words, select_low), 4u);
    EXPECT_EQ(std::ranges::count(words, restore_guest), 4u);
    ASSERT_TRUE(patch.relocated_guest_instruction_offset);
    ASSERT_GE(*patch.relocated_guest_instruction_offset,
              patch.trampoline_offset + sizeof(uint32_t));
    const size_t guest_word = static_cast<size_t>(
        (*patch.relocated_guest_instruction_offset - patch.trampoline_offset) / sizeof(uint32_t));
    EXPECT_EQ(words[guest_word - 1u], restore_guest);
    constexpr size_t kGuestWordCount = 2u;
    ASSERT_LT(guest_word + kGuestWordCount, words.size());
    EXPECT_EQ(words[guest_word + kGuestWordCount], select_low);
    ++checked;
  }
  EXPECT_EQ(checked, kAccessCount);
}

TEST(ConSanMoi, Cdna4DirectSampledProbeEmitsNativePublicationRecipes) {
  std::vector<uint32_t> text_words(600, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  text_words[0] = 0xd81a0004u;
  text_words[1] = 0x00000302u; // ds_write_b32 v2, v3 offset:4
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);

  const std::vector<uint8_t> bytes = make_cdna4_lds_code_object(text_words);
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 13;
  options.moi_epoch_vgpr = 14;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_generation = 7;
  options.moi_report_dispatch_id = 0x1122334455667788ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(2);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_TRUE(result.final_validation_passed);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineMoiSampledWatchpointStore);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 8u);

  const std::vector<uint32_t> rewritten_words =
      patched_words_at_file_offset(result, 0x100, result.patches.front().original_size);
  const auto atomic_publish = build_cdna4_flat_atomic_swap_b64(
      /*vaddr=*/8, /*vsrc=*/10, /*vdst=*/10, /*return_old_value=*/true,
      /*scope=*/2, ROCJITSU_CODE_ARCH_CDNA4);
  const auto atomic_claim = build_cdna4_flat_atomic_cmpswap_b32(
      /*vaddr=*/8, /*vsrc=*/10, /*vdst=*/10, /*return_old_value=*/true,
      /*scope=*/2, ROCJITSU_CODE_ARCH_CDNA4);
  const auto window_counter = build_cdna4_flat_atomic_add_u32(
      /*vaddr=*/8, /*vsrc=*/12, /*vdst=*/12, /*return_old_value=*/true,
      /*scope=*/2, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(atomic_publish && atomic_claim && window_counter);
  EXPECT_EQ(count_subsequence(rewritten_words, *atomic_publish), 1u);
  EXPECT_EQ(count_subsequence(rewritten_words, *atomic_claim), 1u);
  EXPECT_EQ(count_subsequence(rewritten_words, *window_counter), 2u);
  EXPECT_GE(std::count(rewritten_words.begin(), rewritten_words.end(), 0xbf8c0070u), 1);
  EXPECT_TRUE(
      contains_subsequence(rewritten_words, std::array<uint32_t, 2>{text_words[0], text_words[1]}));
}

TEST(ConSanMoi, DirectSampledProbeAddsNativeLdsImmediateOffsets) {
  std::vector<uint32_t> text_words(1300, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  constexpr std::array<size_t, 3> site_words = {0u, 400u, 800u};
  constexpr std::array<uint32_t, 3> byte_offsets = {0u, 4u, 8u};
  for (size_t i = 0; i < site_words.size(); ++i) {
    text_words[site_words[i]] = 0xD8340000u | byte_offsets[i];
    text_words[site_words[i] + 1u] = 0x01000000u; // ds_store_b32 v0, v1 offset:N
  }
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.scratch_vgpr = 20;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(3);
  options.max_patches = 3;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.patches.size(), 3u);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const auto *text = patched.text_sections().front();
  std::vector<uint32_t> patched_words(text->size() / sizeof(uint32_t));
  std::memcpy(patched_words.data(), text->data(), text->size());

  constexpr uint16_t tmp_vgpr = 24;
  const auto zero_shift = build_v_lshrrev_b32_e32(
      tmp_vgpr, scalar_positive_inline_u32(consan_moi_sampled_watchpoint::granule_shift),
      /*vsrc1=*/0, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(zero_shift);
  EXPECT_TRUE(contains_subsequence(patched_words, std::array<uint32_t, 1>{*zero_shift}));
  for (uint32_t byte_offset : std::array<uint32_t, 2>{4u, 8u}) {
    const auto mov_offset =
        build_v_mov_b32_e64_literal(tmp_vgpr, byte_offset, ROCJITSU_CODE_ARCH_RDNA4);
    const auto add_offset = build_v_add_nc_u32_e32(tmp_vgpr, vector_source_vgpr(/*address=*/0),
                                                   tmp_vgpr, ROCJITSU_CODE_ARCH_RDNA4);
    const auto shift = build_v_lshrrev_b32_e32(
        tmp_vgpr, scalar_positive_inline_u32(consan_moi_sampled_watchpoint::granule_shift),
        tmp_vgpr, ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_TRUE(mov_offset);
    ASSERT_TRUE(add_offset);
    ASSERT_TRUE(shift);
    std::vector<uint32_t> expected(mov_offset->begin(), mov_offset->end());
    expected.push_back(*add_offset);
    expected.push_back(*shift);
    EXPECT_TRUE(contains_subsequence(patched_words, expected)) << "byte offset " << byte_offset;
  }
}

TEST(ConSanMoi, DirectSampledProbeSnapshotsOverlappingLoadAddress) {
  std::array<uint32_t, 420> text_words{};
  text_words[0] = 0xD8D80000u;
  text_words[1] = 0x00000000u; // ds_load_b32 v0, v0
  for (size_t i = 2; i + 1u < text_words.size(); ++i)
    text_words[i] = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.scratch_vgpr = 20;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(2);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().scratch_vgpr_count, 6u);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const auto *text = patched.text_sections().front();
  std::vector<uint32_t> patched_words(text->size() / sizeof(uint32_t));
  std::memcpy(patched_words.data(), text->data(), text->size());
  const uint32_t snapshot =
      build_v_mov_b32_e32(/*vdst=*/25, vector_source_vgpr(/*vsrc=*/0), ROCJITSU_CODE_ARCH_RDNA4);
  EXPECT_TRUE(contains_subsequence(
      patched_words, std::array<uint32_t, 3>{snapshot, text_words[0], text_words[1]}));
  const auto start_cell = build_v_lshrrev_b32_e32(
      /*vdst=*/24, scalar_positive_inline_u32(consan_moi_sampled_watchpoint::granule_shift),
      /*saved address=*/25, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(start_cell);
  EXPECT_TRUE(contains_subsequence(patched_words, std::array<uint32_t, 1>{*start_cell}));
}

TEST(ConSanMoi, DirectSampledProbePublishesMultipleLdsAccessRanges) {
  std::vector<uint32_t> text_words(420, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words[0] = 0xD8380201u;
  text_words[1] = 0x00000000u; // ds_store_2addr_b32 v0, v0, v0 offset0:1 offset1:2
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.scratch_vgpr = 20;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(2);
  options.max_patches = 2;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.patches.size(), 1u);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const auto *text = patched.text_sections().front();
  std::vector<uint32_t> patched_words(text->size() / sizeof(uint32_t));
  std::memcpy(patched_words.data(), text->data(), text->size());

  const auto atomic_publish = build_flat_atomic_swap_b64_vaddr_vsrc_vdst(
      /*vaddr=*/20, /*vsrc=*/22, /*vdst=*/22, /*return_old_value=*/true,
      /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(atomic_publish);
  EXPECT_EQ(count_subsequence(patched_words, *atomic_publish), 2u);
  EXPECT_EQ(count_subsequence(patched_words, std::array<uint32_t, 2>{text_words[0], text_words[1]}),
            1u);

  constexpr uint16_t tmp_vgpr = 24;
  for (uint32_t byte_offset : std::array<uint32_t, 2>{4u, 8u}) {
    const auto mov_offset =
        build_v_mov_b32_e64_literal(tmp_vgpr, byte_offset, ROCJITSU_CODE_ARCH_RDNA4);
    const auto add_offset = build_v_add_nc_u32_e32(tmp_vgpr, vector_source_vgpr(/*address=*/0),
                                                   tmp_vgpr, ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_TRUE(mov_offset);
    ASSERT_TRUE(add_offset);
    std::vector<uint32_t> expected(mov_offset->begin(), mov_offset->end());
    expected.push_back(*add_offset);
    EXPECT_TRUE(contains_subsequence(patched_words, expected)) << "byte offset " << byte_offset;
  }
}

TEST(ConSanMoi, DirectSampledProbeRequiresCapacityForEveryLdsAccessRange) {
  std::vector<uint32_t> text_words(420, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words[0] = 0xD8380201u;
  text_words[1] = 0x00000000u; // ds_store_2addr_b32 v0, v0, v0 offset0:1 offset1:2
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.scratch_vgpr = 20;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(1);

  const auto result = try_patch_consan(make_rdna4_lds_code_object(text_words), options);

  EXPECT_TRUE(consan_patch_succeeded(result));
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.patches.empty());
  ASSERT_EQ(std::ranges::count(result.site_dispositions, ConSanResourceSiteKind::Access,
                               &ConSanSiteDispositionRecord::site_kind),
            1u);
  const auto access_site =
      std::ranges::find(result.site_dispositions, ConSanResourceSiteKind::Access,
                        &ConSanSiteDispositionRecord::site_kind);
  ASSERT_NE(access_site, result.site_dispositions.end());
  EXPECT_EQ(access_site->disposition, ConSanSiteDisposition::Supported);
  EXPECT_EQ(access_site->reason, ConSanSiteDispositionReason::None);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("cannot retain every range") != std::string::npos;
  }));
}

TEST(ConSanMoi, SampledAtomicTrackingPublishesQualifiedTypedMetadata) {
  const std::vector<uint8_t> bytes = make_rdna4_sampled_lds_and_ordered_flat_atomic_code_object();
  ASSERT_FALSE(bytes.empty());
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.moi_track_atomics = true;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 20;
  options.moi_epoch_vgpr = 21;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(2);
  options.max_patches = 2;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_EQ(result.moi_engine, ConSanMoiEngine::Sampled);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.kernels.front().atomic_sites.size(), 1u);
  const auto patch = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiSampledSyncMetadata, &ConSanPatchInfo::kind);
  ASSERT_NE(patch, result.patches.end());
  EXPECT_EQ(patch->anchor_offset, 20u);
  ASSERT_TRUE(patch->scratch_vgpr);
  EXPECT_EQ(*patch->scratch_vgpr, 8u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const auto *text = patched.text_sections().front();
  std::vector<uint32_t> trampoline(patch->trampoline_size / sizeof(uint32_t));
  std::memcpy(trampoline.data(), text->data() + patch->trampoline_offset, patch->trampoline_size);
  const ConSanMoiReportBufferLayout layout =
      consan_moi_direct_sampled_report_buffer_layout_for_bytes(options.moi_report_buffer_size);
  const uint64_t metadata =
      *options.moi_report_buffer_address + layout.sampled_sync_metadata_offset;
  const auto encoded = encode_consan_moi_sampled_sync_metadata({
      .address = 1,
      .byte_count = 4,
      .kind = ConSanMoiSampledSyncKind::Atomic,
      .role = ConSanMoiSampledSyncRole::RmwRelease,
      .scope = ConSanMoiSampledSyncScope::Agent,
      .outcome = ConSanMoiSampledSyncOutcome::RmwNoReturn,
  });
  ASSERT_EQ(encoded.classification, ConSanMoiSampledSyncClassification::Valid);
  const auto claim = build_flat_atomic_cmpswap_b32_vaddr_vsrc_vdst(
      /*vaddr=*/8, /*vsrc=*/10, /*vdst=*/10, /*return_old_value=*/true,
      /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  const auto commit = build_flat_atomic_cmpswap_b32_vaddr_vsrc_vdst(
      /*vaddr=*/8, /*vsrc=*/10, /*vdst=*/10, /*return_old_value=*/true,
      /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(claim);
  ASSERT_TRUE(commit);
  EXPECT_EQ(*claim, *commit);
  EXPECT_EQ(count_subsequence(trampoline, *claim), 2u);
  const auto payload_wait = build_s_wait_storecnt0(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(payload_wait);
  EXPECT_NE(std::find(trampoline.begin(), trampoline.end(), *payload_wait), trampoline.end());
  EXPECT_EQ(count_subsequence(
                trampoline,
                make_expected_literal_store_words(
                    metadata + offsetof(ConSanMoiSampledSyncMetadataPacked, byte_count), 4, 8)),
            1u);
  EXPECT_EQ(
      count_subsequence(
          trampoline, make_expected_vgpr_store_words(
                          metadata + offsetof(ConSanMoiSampledSyncMetadataPacked, address), 16, 8)),
      1u);
  EXPECT_EQ(count_subsequence(
                trampoline,
                make_expected_vgpr_store_words(
                    metadata + offsetof(ConSanMoiSampledSyncMetadataPacked, address) + 4u, 17, 8)),
            1u);
  ASSERT_TRUE(result.resolved_moi_exec_save_sgpr);
  const uint16_t state = *result.resolved_moi_exec_save_sgpr;
  const auto save_vcc =
      build_s_mov_b64(static_cast<uint16_t>(state + 2u), kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_vcc =
      build_s_mov_b64(kRdna4VccLo, static_cast<uint16_t>(state + 2u), ROCJITSU_CODE_ARCH_RDNA4);
  const auto save_scc =
      build_rdna4_s_cselect_b32(static_cast<uint16_t>(state + 4u), scalar_positive_inline_u32(1),
                                scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_scc = build_rdna4_s_cmp_lg_u32(
      static_cast<uint16_t>(state + 4u), scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(save_vcc);
  ASSERT_TRUE(restore_vcc);
  ASSERT_TRUE(save_scc);
  ASSERT_TRUE(restore_scc);
  EXPECT_NE(std::find(trampoline.begin(), trampoline.end(), *save_vcc), trampoline.end());
  EXPECT_NE(std::find(trampoline.begin(), trampoline.end(), *restore_vcc), trampoline.end());
  EXPECT_NE(std::find(trampoline.begin(), trampoline.end(), *save_scc), trampoline.end());
  EXPECT_NE(std::find(trampoline.begin(), trampoline.end(), *restore_scc), trampoline.end());

  // Every identity test must intersect the surviving lane mask. Merely
  // branching on VCCZ lets different lanes satisfy different fields and was
  // the regression this test is intended to catch.
  const auto cumulative_narrow =
      build_s_and_saveexec_b64(state, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(cumulative_narrow);
  EXPECT_GE(std::count(trampoline.begin(), trampoline.end(), *cumulative_narrow), 20);
  const auto restore_collision_exec =
      build_s_mov_b64(kRdna4ExecLo, static_cast<uint16_t>(state + 6u), ROCJITSU_CODE_ARCH_RDNA4);
  const auto collision_mask =
      build_s_mov_b64(state, static_cast<uint16_t>(state + 6u), ROCJITSU_CODE_ARCH_RDNA4);
  const auto collision_mbcnt_lo =
      build_v_mbcnt_lo_u32_b32(10, state, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
  const auto collision_mbcnt_hi = build_v_mbcnt_hi_u32_b32(
      10, static_cast<uint16_t>(state + 1u), vector_source_vgpr(10), ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(restore_collision_exec);
  ASSERT_TRUE(collision_mask);
  ASSERT_TRUE(collision_mbcnt_lo);
  ASSERT_TRUE(collision_mbcnt_hi);
  EXPECT_NE(std::find(trampoline.begin(), trampoline.end(), *restore_collision_exec),
            trampoline.end());
  EXPECT_NE(std::find(trampoline.begin(), trampoline.end(), *collision_mask), trampoline.end());
  EXPECT_NE(std::search(trampoline.begin(), trampoline.end(), collision_mbcnt_lo->begin(),
                        collision_mbcnt_lo->end()),
            trampoline.end());
  EXPECT_NE(std::search(trampoline.begin(), trampoline.end(), collision_mbcnt_hi->begin(),
                        collision_mbcnt_hi->end()),
            trampoline.end());
}

TEST(ConSanMoi, Gfx1250OrderedLdsAtomicComposesSampledAccessAndOrderingMetadata) {
  const std::array<uint32_t, 7> words = {
      0x360202ffu, 0x000000ffu, // release wait setup
      0xbf94ffffu,              // s_barrier_wait -1
      0xbfc10000u,              // release ordering completion
      0xd8000000u, 0x00001210u, // ds_add_u32 v0, v18, no return
      0xbfb00000u,              // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_gfx1250_code_object(words, "sampled_ordered_lds_atomic");
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.moi_track_atomics = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(8, 8, 0, 0, 0, 8, 1);
  options.max_patches = 8;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result))
      << testing::PrintToString(result.errors) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::InlineMoiSampledWatchpointStore ||
           patch.kind == ConSanPatchKind::TrampolineMoiSampledWatchpointStore;
  });
  const auto atomic = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiSampledSyncMetadata, &ConSanPatchInfo::kind);
  ASSERT_NE(access, result.patches.end());
  ASSERT_NE(atomic, result.patches.end()) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(access->relocated_guest_instruction_offset);
  ASSERT_TRUE(atomic->relocated_guest_instruction_offset);
  EXPECT_EQ(access->anchor_offset, atomic->anchor_offset);
  EXPECT_NE(*access->relocated_guest_instruction_offset,
            *atomic->relocated_guest_instruction_offset);
  EXPECT_TRUE(std::ranges::any_of(result.site_dispositions, [](const auto &site) {
    return site.site_kind == ConSanResourceSiteKind::Atomic &&
           site.lowering_outcome == ConSanSiteLoweringOutcome::Patched;
  }));
}

TEST(ConSanMoi, Gfx1250SampledPublishesIsolatedLdsReleaseOrdering) {
  constexpr auto store = gfx1250::build_vds(gfx1250::kDsStoreB32Vds, {.addr = 4u, .data0 = 5u});
  constexpr auto atomic =
      gfx1250::build_vds(gfx1250::kDsAddU32Vds, {.offset0 = 12u, .addr = 2u, .data0 = 1u});
  const std::array<uint32_t, 6> words = {
      store[0],  store[1],  0xBFC90000u, // s_wait_storecnt_dscnt 0
      atomic[0], atomic[1], build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250)};
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.moi_track_atomics = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(3);
  options.max_patches = 3;

  const ConSanResult result = try_patch_consan(
      make_gfx1250_code_object(words, "gfx1250_sampled_isolated_lds_release"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  EXPECT_FALSE(std::ranges::any_of(result.site_dispositions, [](const auto &site) {
    return site.site_kind == ConSanResourceSiteKind::Atomic &&
           site.disposition == ConSanSiteDisposition::NotApplicable &&
           site.reason == ConSanSiteDispositionReason::NoAtomicAcquireConsumer &&
           site.lowering_outcome == ConSanSiteLoweringOutcome::NotApplicable;
  }));
  EXPECT_EQ(std::ranges::count_if(result.patches,
                                  [](const ConSanPatchInfo &patch) {
                                    return patch.kind ==
                                               ConSanPatchKind::InlineMoiSampledWatchpointStore ||
                                           patch.kind ==
                                               ConSanPatchKind::TrampolineMoiSampledWatchpointStore;
                                  }),
            2u);
  // The qualified release is dual-role: ordinary atomic access observation
  // and workgroup-scoped ordering metadata remain independently visible.
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiSampledSyncMetadata,
                               &ConSanPatchInfo::kind),
            1u);
}

TEST(ConSanMoi, Cdna4SampledAtomicTracksCacheAssociatedOrdering) {
  const auto release = cdna4::build_mubuf(cdna4::kBufferWbl2Mubuf, {.sc1 = 1});
  const auto acquire = cdna4::build_mubuf(cdna4::kBufferInvMubuf, {.sc1 = 1});
  const auto atomic = build_cdna4_flat_atomic_add_u32(
      /*vaddr=*/4, /*vsrc=*/6, /*vdst=*/7, /*return_old_value=*/true,
      /*scope=*/2, ROCJITSU_CODE_ARCH_CDNA4);
  const auto wait = build_cdna4_s_wait_flat0(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(atomic && wait);
  std::vector<uint32_t> text_words = {
      0xd81a0004u,
      0x00000302u, // ds_write_b32 v2, v3 offset:4
  };
  text_words.insert(text_words.end(), release.begin(), release.end());
  text_words.push_back(*wait);
  text_words.insert(text_words.end(), atomic->begin(), atomic->end());
  text_words.push_back(*wait);
  text_words.insert(text_words.end(), acquire.begin(), acquire.end());
  text_words.resize(800, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);
  const std::vector<uint8_t> bytes =
      make_cdna4_lds_code_object(text_words, "sampled_atomic_acquire_release");
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.moi_track_atomics = true;
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 80;
  options.moi_owner_vgpr = 20;
  options.moi_epoch_vgpr = 21;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(2);
  options.max_patches = 2;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << "warnings=" << testing::PrintToString(result.warnings)
                               << " errors=" << testing::PrintToString(result.errors);
  EXPECT_NE(std::ranges::find(result.patches, ConSanPatchKind::TrampolineMoiSampledSyncMetadata,
                              &ConSanPatchInfo::kind),
            result.patches.end());
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, Cdna4SampledBarrierPublishesSelectedEpochTransition) {
  const auto barrier = build_cdna4_s_barrier(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(barrier);
  std::vector<uint32_t> text_words(540, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  text_words[0] = 0xd81a0004u;
  text_words[1] = 0x00000302u; // ds_write_b32 v2, v3 offset:4
  text_words[400] = *barrier;
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.moi_track_barriers = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(2);
  options.max_patches = 3;

  const ConSanResult result =
      try_patch_consan(make_cdna4_lds_code_object(text_words, "sampled_barrier"), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified) << "warnings=" << testing::PrintToString(result.warnings)
                               << " errors=" << testing::PrintToString(result.errors);
  EXPECT_NE(std::ranges::find(result.patches, ConSanPatchKind::TrampolineMoiSampledSyncMetadata,
                              &ConSanPatchInfo::kind),
            result.patches.end());
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, CdnaSampledAtomicUsesPrivatePersistentStateAtAccvgprBoundary) {
  constexpr uint16_t kCasCompareOffset = 4u;
  constexpr uint16_t kCasResultOffset = 5u;
  constexpr uint16_t kSavedAddressOffset = 8u;
  for (const SampledCdnaTarget &target : kSampledCdnaTargets) {
    for (bool is_cas : {false, true}) {
      for (bool deferred_acquire : {false, true}) {
        SCOPED_TRACE(testing::Message() << target.label << (is_cas ? " cas" : " add")
                                        << (deferred_acquire ? " deferred-acquire" : " release"));
        std::vector<uint32_t> release_words;
        std::vector<uint32_t> acquire_words;
        std::vector<uint32_t> atomic_words;
        // The release layouts cover an in-spill address plus mixed CAS evidence
        // on CDNA3, and an out-of-spill address on CDNA4. The deferred layout
        // keeps both evidence words in spill slots. Together they lock
        // address/evidence private reloads and direct register copies.
        uint16_t cas_address_vgpr = 8u;
        uint16_t cas_result_vgpr = 4u;
        if (!deferred_acquire) {
          if (target.arch == ROCJITSU_CODE_ARCH_CDNA3) {
            cas_address_vgpr = 4u;
            cas_result_vgpr = 11u;
          } else {
            cas_address_vgpr = 10u;
          }
        }
        uint32_t wait_word = 0u;
        if (target.arch == ROCJITSU_CODE_ARCH_CDNA3) {
          const auto release = cdna3::build_mubuf(cdna3::kBufferWbl2Mubuf, {.sc1 = 1});
          const auto acquire = build_cdna3_buffer_inv_sc1(target.arch);
          const auto atomic =
              is_cas ? instrumentation::build_flat_atomic_cmpswap_b32(cas_address_vgpr, /*vsrc=*/6,
                                                                      cas_result_vgpr,
                                                                      /*return_old_value=*/true,
                                                                      /*scope=*/2, target.arch)
                     : instrumentation::build_flat_atomic_add_u32(
                           /*vaddr=*/4, /*vsrc=*/6, /*vdst=*/7, /*return_old_value=*/true,
                           /*scope=*/2, target.arch);
          const auto wait = build_cdna3_s_wait_vmcnt_lgkmcnt0(target.arch);
          ASSERT_TRUE(acquire && atomic && wait);
          release_words.assign(release.begin(), release.end());
          acquire_words.assign(acquire->begin(), acquire->end());
          atomic_words.assign(atomic->begin(), atomic->end());
          wait_word = *wait;
        } else {
          const auto release = cdna4::build_mubuf(cdna4::kBufferWbl2Mubuf, {.sc1 = 1});
          const auto acquire = cdna4::build_mubuf(cdna4::kBufferInvMubuf, {.sc1 = 1});
          const auto atomic =
              is_cas ? instrumentation::build_flat_atomic_cmpswap_b32(cas_address_vgpr, /*vsrc=*/6,
                                                                      cas_result_vgpr,
                                                                      /*return_old_value=*/true,
                                                                      /*scope=*/2, target.arch)
                     : instrumentation::build_flat_atomic_add_u32(
                           /*vaddr=*/4, /*vsrc=*/6, /*vdst=*/7, /*return_old_value=*/true,
                           /*scope=*/2, target.arch);
          const auto wait = build_cdna4_s_wait_flat0(target.arch);
          ASSERT_TRUE(atomic && wait);
          release_words.assign(release.begin(), release.end());
          acquire_words.assign(acquire.begin(), acquire.end());
          atomic_words.assign(atomic->begin(), atomic->end());
          wait_word = *wait;
        }
        std::vector<uint32_t> access_words;
        if (deferred_acquire) {
          if (target.arch == ROCJITSU_CODE_ARCH_CDNA3) {
            const auto access = build_cdna3_ds_load_b32(
                /*vdst=*/11, /*vaddr=*/10, /*byte_offset=*/0, target.arch);
            ASSERT_TRUE(access);
            access_words.assign(access->begin(), access->end());
          } else {
            access_words = {
                0xd86c0000u,
                0x0b00000au, // ds_read_b32 v11, v10
            };
          }
        } else {
          const auto access = target.arch == ROCJITSU_CODE_ARCH_CDNA3
                                  ? build_cdna3_ds_store_b32(
                                        /*vaddr=*/10, /*vdata=*/11, /*byte_offset=*/0, target.arch)
                                  : build_cdna4_ds_store_b32(
                                        /*vaddr=*/10, /*vdata=*/11, /*byte_offset=*/0, target.arch);
          ASSERT_TRUE(access);
          access_words.assign(access->begin(), access->end());
        }

        std::vector<uint32_t> text_words;
        if (!deferred_acquire)
          text_words.insert(text_words.end(), access_words.begin(), access_words.end());
        text_words.insert(text_words.end(), release_words.begin(), release_words.end());
        text_words.push_back(wait_word);
        text_words.insert(text_words.end(), atomic_words.begin(), atomic_words.end());
        text_words.push_back(wait_word);
        text_words.insert(text_words.end(), acquire_words.begin(), acquire_words.end());
        if (deferred_acquire)
          text_words.insert(text_words.end(), access_words.begin(), access_words.end());
        text_words.resize(800u, build_s_nop(0, target.arch));
        text_words.back() = build_s_endpgm(target.arch);
        std::vector<uint8_t> bytes =
            target.arch == ROCJITSU_CODE_ARCH_CDNA3
                ? make_cdna3_lds_code_object(text_words, "sampled_atomic_accvgpr_boundary",
                                             /*vgpr_granulated=*/3u)
                : make_cdna4_lds_code_object(text_words, "sampled_atomic_accvgpr_boundary",
                                             /*vgpr_granulated=*/3u);
        mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
          AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET,
                          2u);
        });

        ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
        options.moi_runtime_sample_stride = 2u;
        options.moi_track_barriers = false;
        options.moi_track_atomics = true;
        options.moi_report_buffer_address = 0x123456780000ull;
        options.moi_report_buffer_size = direct_sampled_report_bytes(2);
        options.max_patches = 3u;

        const ConSanResult result = try_patch_consan(bytes, options);

        ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
        ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
        EXPECT_TRUE(result.moi_private_epoch_automatic);
        const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
          return patch.kind == ConSanPatchKind::InlineMoiSampledWatchpointStore ||
                 patch.kind == ConSanPatchKind::TrampolineMoiSampledWatchpointStore;
        });
        const auto atomic =
            std::ranges::find(result.patches, ConSanPatchKind::TrampolineMoiSampledSyncMetadata,
                              &ConSanPatchInfo::kind);
        const auto prologue =
            std::ranges::find(result.patches, ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue,
                              &ConSanPatchInfo::kind);
        ASSERT_NE(access, result.patches.end());
        ASSERT_NE(atomic, result.patches.end()) << testing::PrintToString(result.warnings);
        ASSERT_NE(prologue, result.patches.end());
        EXPECT_EQ(atomic->persistent_epoch_private_offset, access->persistent_epoch_private_offset);
        EXPECT_EQ(atomic->persistent_owner_private_offset, access->persistent_owner_private_offset);
        EXPECT_EQ(atomic->persistent_sample_sequence_private_offset,
                  access->persistent_sample_sequence_private_offset);
        EXPECT_EQ(prologue->persistent_private_state_end, access->persistent_private_state_end);

        ASSERT_TRUE(atomic->scratch_vgpr);
        ASSERT_TRUE(atomic->persistent_owner_private_offset);
        EXPECT_EQ(atomic->spilled_vgpr_count, 10u);
        EXPECT_EQ(atomic->relocated_guest_instruction_offset, atomic->trampoline_offset);
        AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
        ASSERT_TRUE(patched.is_valid());
        const std::vector<uint32_t> cave =
            text_words_at_offset(patched, atomic->trampoline_offset, atomic->trampoline_size);
        const uint16_t owner_vgpr = static_cast<uint16_t>(*atomic->scratch_vgpr + 6u);
        const auto owner_load = instrumentation::build_private_load_b32(
            owner_vgpr, *atomic->persistent_owner_private_offset, target.arch);
        ASSERT_TRUE(owner_load);
        EXPECT_TRUE(contains_subsequence(cave, *owner_load));
        ASSERT_TRUE(atomic->persistent_private_state_end);
        ASSERT_LE(*atomic->scratch_vgpr, 4u);
        ASSERT_GT(static_cast<uint32_t>(*atomic->scratch_vgpr) + atomic->spilled_vgpr_count, 7u);
        const uint32_t scratch_end =
            static_cast<uint32_t>(*atomic->scratch_vgpr) + atomic->spilled_vgpr_count;
        std::vector<uint32_t> snapshot_words;
        bool address_loaded_from_private = false;
        const uint16_t atomic_address_vgpr = is_cas ? cas_address_vgpr : 4u;
        for (uint16_t word = 0; word < 2u; ++word) {
          const uint16_t source = static_cast<uint16_t>(atomic_address_vgpr + word);
          const uint16_t destination =
              static_cast<uint16_t>(*atomic->scratch_vgpr + kSavedAddressOffset + word);
          if (source >= *atomic->scratch_vgpr && source < scratch_end) {
            const auto load = instrumentation::build_private_load_b32(
                destination,
                *atomic->persistent_private_state_end +
                    static_cast<uint32_t>(source - *atomic->scratch_vgpr) * sizeof(uint32_t),
                target.arch);
            ASSERT_TRUE(load);
            snapshot_words.insert(snapshot_words.end(), load->begin(), load->end());
            address_loaded_from_private = true;
          } else {
            snapshot_words.push_back(
                build_v_mov_b32_e32(destination, vector_source_vgpr(source), target.arch));
          }
        }
        const auto wait = instrumentation::build_s_wait_private_load0(target.arch);
        ASSERT_TRUE(wait);
        if (address_loaded_from_private)
          snapshot_words.push_back(*wait);
        if (is_cas) {
          const auto compare_load = instrumentation::build_private_load_b32(
              static_cast<uint16_t>(*atomic->scratch_vgpr + kCasCompareOffset),
              *atomic->persistent_private_state_end +
                  static_cast<uint32_t>(7u - *atomic->scratch_vgpr) * sizeof(uint32_t),
              target.arch);
          ASSERT_TRUE(compare_load);
          snapshot_words.insert(snapshot_words.end(), compare_load->begin(), compare_load->end());
          if (cas_result_vgpr >= *atomic->scratch_vgpr && cas_result_vgpr < scratch_end) {
            const auto result_load = instrumentation::build_private_load_b32(
                static_cast<uint16_t>(*atomic->scratch_vgpr + kCasResultOffset),
                *atomic->persistent_private_state_end +
                    static_cast<uint32_t>(cas_result_vgpr - *atomic->scratch_vgpr) *
                        sizeof(uint32_t),
                target.arch);
            ASSERT_TRUE(result_load);
            snapshot_words.insert(snapshot_words.end(), result_load->begin(), result_load->end());
          } else {
            snapshot_words.push_back(
                build_v_mov_b32_e32(static_cast<uint16_t>(*atomic->scratch_vgpr + kCasResultOffset),
                                    vector_source_vgpr(cas_result_vgpr), target.arch));
          }
          snapshot_words.push_back(*wait);
        }
        EXPECT_TRUE(contains_subsequence(cave, snapshot_words));
        EXPECT_FALSE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
          return warning.find("sampled atomic sync could not lower associated") !=
                 std::string::npos;
        }));
        EXPECT_TRUE(result.final_validation_passed);
      }
    }
  }
}

TEST(ConSanMoi, SignExtendsCdnaVglobalOffsetAt13BitBoundaries) {
  EXPECT_EQ(sign_extend_13_bit_offset(0x0000u), 0);
  EXPECT_EQ(sign_extend_13_bit_offset(0x0014u), 20);
  EXPECT_EQ(sign_extend_13_bit_offset(0x0fffu), 4095);
  EXPECT_EQ(sign_extend_13_bit_offset(0x1000u), -4096);
  EXPECT_EQ(sign_extend_13_bit_offset(0x1fffu), -1);
  EXPECT_EQ(sign_extend_13_bit_offset(0xffffe014u), 20);
}

TEST(ConSanMoi, CdnaVglobalMaterializationSelectsSafeSignScratch) {
  ConSanMoiAtomicAddressPlan plan;
  plan.kind = ConSanMoiAtomicAddressKind::VglobalMaterialized;
  plan.support = ConSanMoiAtomicAddressSupport::Supported;
  plan.input_address_vgpr = 4u;
  plan.input_address_vgpr_count = 1u;
  plan.scalar_base_sgpr = 20u;
  plan.signed_byte_offset = 20;
  plan.result_address_vgpr = 7u;
  plan.result_address_vgpr_count = 2u;
  plan.scratch_vgpr = 4u;
  plan.scratch_vgpr_count = 5u;
  plan.resource_source = ConSanRegisterAllocationSource::SpillRequired;
  for (const SampledCdnaTarget &target : kSampledCdnaTargets) {
    SCOPED_TRACE(target.label);
    const auto expect_sign_scratch = [&](const ConSanMoiAtomicAddressPlan &candidate,
                                         uint16_t sign_vgpr) {
      const auto materialization = build_consan_moi_atomic_address_materialization(
          candidate, /*vcc_save_sgpr=*/82u, /*scc_save_sgpr=*/84u, target.arch);
      const auto signed_add = instrumentation::build_v_add_u64_signed_vgpr_offset(
          candidate.result_address_vgpr, candidate.input_address_vgpr, sign_vgpr, target.arch);
      ASSERT_TRUE(materialization && signed_add);
      ASSERT_GE(materialization->size(), 4u + signed_add->size());
      EXPECT_TRUE(
          std::equal(signed_add->begin(), signed_add->end(), materialization->begin() + 4u));
    };

    // The first scratch aliases the input, so materialization uses the next one.
    expect_sign_scratch(plan, /*sign_vgpr=*/5u);

    // A non-aliasing first scratch can be used directly.
    ConSanMoiAtomicAddressPlan direct = plan;
    direct.scratch_vgpr = 5u;
    direct.result_address_vgpr = 8u;
    expect_sign_scratch(direct, /*sign_vgpr=*/5u);

    // Exactly two words before the pair is the minimum valid spacing.
    ConSanMoiAtomicAddressPlan boundary = plan;
    boundary.result_address_vgpr = 6u;
    expect_sign_scratch(boundary, /*sign_vgpr=*/5u);

    // One word before the pair cannot hold the aliased input and sign scratch.
    ConSanMoiAtomicAddressPlan under_reserved = plan;
    under_reserved.result_address_vgpr = 5u;
    EXPECT_FALSE(build_consan_moi_atomic_address_materialization(
        under_reserved, /*vcc_save_sgpr=*/82u, /*scc_save_sgpr=*/84u, target.arch));

    // The declared scratch window must contain the chosen temporary and pair.
    ConSanMoiAtomicAddressPlan truncated_window = plan;
    truncated_window.scratch_vgpr_count = 1u;
    EXPECT_FALSE(build_consan_moi_atomic_address_materialization(
        truncated_window, /*vcc_save_sgpr=*/82u, /*scc_save_sgpr=*/84u, target.arch));
  }
}

TEST(ConSanMoi, CdnaSampledVglobalMaterializesVectorAndScalarAddressesInScratchTail) {
  struct VglobalCase {
    std::string_view label;
    std::array<uint32_t, 2> words;
    ConSanMoiAtomicAddressKind expected_kind;
    uint16_t input_vgpr_count;
    std::optional<uint16_t> scalar_base_sgpr;
    int32_t signed_byte_offset;
  };
  constexpr std::array<VglobalCase, 2> kVglobalCases = {{
      {
          "vector-only/off",
          {
              0xdf089fecu,
              0x007f0604u, // global_atomic_add v[4:5], v6, off offset:-20 sc1
          },
          ConSanMoiAtomicAddressKind::VglobalGuestPairMaterialized,
          2u,
          std::nullopt,
          -20,
      },
      {
          "scalar-base",
          {
              0xdf088014u,
              0x00140604u, // global_atomic_add v4, v6, s[20:21] offset:20 sc1
          },
          ConSanMoiAtomicAddressKind::VglobalMaterialized,
          1u,
          20u,
          20,
      },
  }};
  for (const SampledCdnaTarget &target : kSampledCdnaTargets) {
    SCOPED_TRACE(target.label);
    const auto access = target.arch == ROCJITSU_CODE_ARCH_CDNA3
                            ? build_cdna3_ds_store_b32(
                                  /*vaddr=*/10, /*vdata=*/11, /*byte_offset=*/0, target.arch)
                            : build_cdna4_ds_store_b32(
                                  /*vaddr=*/10, /*vdata=*/11, /*byte_offset=*/0, target.arch);
    ASSERT_TRUE(access);
    std::vector<uint32_t> release_words;
    std::vector<uint32_t> acquire_words;
    uint32_t wait_word = 0u;
    if (target.arch == ROCJITSU_CODE_ARCH_CDNA3) {
      const auto release = cdna3::build_mubuf(cdna3::kBufferWbl2Mubuf, {.sc1 = 1});
      const auto acquire = build_cdna3_buffer_inv_sc1(target.arch);
      const auto wait = build_cdna3_s_wait_vmcnt_lgkmcnt0(target.arch);
      ASSERT_TRUE(acquire && wait);
      release_words.assign(release.begin(), release.end());
      acquire_words.assign(acquire->begin(), acquire->end());
      wait_word = *wait;
    } else {
      const auto release = cdna4::build_mubuf(cdna4::kBufferWbl2Mubuf, {.sc1 = 1});
      const auto acquire = cdna4::build_mubuf(cdna4::kBufferInvMubuf, {.sc1 = 1});
      const auto wait = build_cdna4_s_wait_flat0(target.arch);
      ASSERT_TRUE(wait);
      release_words.assign(release.begin(), release.end());
      acquire_words.assign(acquire.begin(), acquire.end());
      wait_word = *wait;
    }

    for (const VglobalCase &test_case : kVglobalCases) {
      SCOPED_TRACE(test_case.label);
      std::vector<uint32_t> text_words;
      text_words.insert(text_words.end(), access->begin(), access->end());
      text_words.insert(text_words.end(), release_words.begin(), release_words.end());
      text_words.push_back(wait_word);
      text_words.insert(text_words.end(), test_case.words.begin(), test_case.words.end());
      text_words.push_back(wait_word);
      text_words.insert(text_words.end(), acquire_words.begin(), acquire_words.end());
      text_words.resize(800u, build_s_nop(0, target.arch));
      text_words.back() = build_s_endpgm(target.arch);
      std::vector<uint8_t> bytes =
          target.arch == ROCJITSU_CODE_ARCH_CDNA3
              ? make_cdna3_lds_code_object(text_words, "sampled_vglobal_materialized_spill",
                                           /*vgpr_granulated=*/3u)
              : make_cdna4_lds_code_object(text_words, "sampled_vglobal_materialized_spill",
                                           /*vgpr_granulated=*/3u);
      mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
        AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET,
                        2u);
      });

      ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
      options.moi_runtime_sample_stride = 2u;
      options.moi_track_barriers = false;
      options.moi_track_atomics = true;
      options.moi_exec_save_sgpr = 80u;
      options.moi_report_buffer_address = 0x123456780000ull;
      options.moi_report_buffer_size = direct_sampled_report_bytes(2);
      options.max_patches = 3u;

      const ConSanResult result = try_patch_consan(bytes, options);

      ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
      ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
      ASSERT_EQ(result.kernels.size(), 1u);
      const auto site = std::ranges::find_if(
          result.kernels.front().atomic_sites,
          [](const ConSanAtomicSite &item) { return item.mnemonic.starts_with("global_atomic_"); });
      ASSERT_NE(site, result.kernels.front().atomic_sites.end())
          << testing::PrintToString(result.kernels.front().atomic_sites);
      ASSERT_TRUE(site->raw_scope) << testing::PrintToString(*site);
      EXPECT_EQ(site->width_bits, 32u);
      EXPECT_EQ(site->saddr_sgpr, test_case.scalar_base_sgpr);
      const auto patch =
          std::ranges::find(result.patches, ConSanPatchKind::TrampolineMoiSampledSyncMetadata,
                            &ConSanPatchInfo::kind);
      ASSERT_NE(patch, result.patches.end()) << testing::PrintToString(result.warnings);
      ASSERT_TRUE(patch->scratch_vgpr);
      ASSERT_TRUE(patch->relocated_guest_instruction_offset);
      EXPECT_EQ(patch->spilled_vgpr_count, 10u);
      EXPECT_EQ(patch->relocated_guest_instruction_offset, patch->trampoline_offset);

      const ConSanMoiAtomicAddressPlan address_plan =
          plan_consan_moi_atomic_address(*site, *patch->scratch_vgpr, patch->spilled_vgpr_count,
                                         ConSanRegisterAllocationSource::SpillRequired, target.arch,
                                         /*allow_post_guest_spill_operand_overlap=*/true);
      ASSERT_TRUE(address_plan.supported());
      EXPECT_EQ(address_plan.kind, test_case.expected_kind);
      EXPECT_EQ(address_plan.input_address_vgpr_count, test_case.input_vgpr_count);
      EXPECT_EQ(address_plan.scalar_base_sgpr, test_case.scalar_base_sgpr);
      EXPECT_EQ(address_plan.signed_byte_offset, test_case.signed_byte_offset);
      const uint16_t saved_address = static_cast<uint16_t>(*patch->scratch_vgpr + 8u);
      EXPECT_EQ(address_plan.result_address_vgpr, saved_address);
      for (int32_t boundary : {-(1 << 12), (1 << 12) - 1}) {
        ConSanAtomicSite boundary_site = *site;
        boundary_site.raw_ioffset = boundary;
        EXPECT_TRUE(plan_consan_moi_atomic_address(
                        boundary_site, *patch->scratch_vgpr, patch->spilled_vgpr_count,
                        ConSanRegisterAllocationSource::SpillRequired, target.arch,
                        /*allow_post_guest_spill_operand_overlap=*/true)
                        .supported());
      }
      for (int32_t out_of_range : {-(1 << 12) - 1, 1 << 12}) {
        ConSanAtomicSite boundary_site = *site;
        boundary_site.raw_ioffset = out_of_range;
        EXPECT_EQ(plan_consan_moi_atomic_address(
                      boundary_site, *patch->scratch_vgpr, patch->spilled_vgpr_count,
                      ConSanRegisterAllocationSource::SpillRequired, target.arch,
                      /*allow_post_guest_spill_operand_overlap=*/true)
                      .support,
                  ConSanMoiAtomicAddressSupport::UnsupportedOffset);
      }

      const auto materialization = build_consan_moi_atomic_address_materialization(
          address_plan, /*vcc_save_sgpr=*/82u, /*scc_save_sgpr=*/84u, target.arch);
      ASSERT_TRUE(materialization);
      if (test_case.scalar_base_sgpr) {
        EXPECT_NE(*patch->scratch_vgpr, address_plan.input_address_vgpr);
        const uint16_t sign_vgpr = *patch->scratch_vgpr;
        const auto signed_add = instrumentation::build_v_add_u64_signed_vgpr_offset(
            address_plan.result_address_vgpr, address_plan.input_address_vgpr, sign_vgpr,
            target.arch);
        const auto zero_extended_add = instrumentation::build_v_add_u64_vgpr_offset(
            address_plan.result_address_vgpr, address_plan.input_address_vgpr, target.arch);
        ASSERT_TRUE(signed_add && zero_extended_add);
        ASSERT_GE(materialization->size(), 4u + signed_add->size());
        EXPECT_TRUE(
            std::equal(signed_add->begin(), signed_add->end(), materialization->begin() + 4u));
        EXPECT_FALSE(contains_subsequence(*materialization, *zero_extended_add));
      }
      AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
      ASSERT_TRUE(patched.is_valid());
      const std::vector<uint32_t> cave =
          text_words_at_offset(patched, patch->trampoline_offset, patch->trampoline_size);
      EXPECT_TRUE(contains_subsequence(cave, *materialization));
      EXPECT_EQ(std::ranges::count(cave, build_v_mov_b32_e32(saved_address,
                                                             vector_source_vgpr(saved_address),
                                                             target.arch)),
                0u);
      EXPECT_EQ(std::ranges::count(
                    cave, build_v_mov_b32_e32(
                              static_cast<uint16_t>(saved_address + 1u),
                              vector_source_vgpr(static_cast<uint16_t>(saved_address + 1u)),
                              target.arch)),
                0u);
      EXPECT_TRUE(result.final_validation_passed);
    }
  }
}

TEST(ConSanMoi, Cdna4SharedSampledAtomicSeparatesPersistentStateFromSpills) {
  const std::vector<uint8_t> bytes = make_cdna4_two_kernel_shared_sampled_atomic_code_object(
      /*first_vgpr_granulated=*/3u, /*second_vgpr_granulated=*/3u,
      /*first_private_bytes=*/0u, /*second_private_bytes=*/20u);
  ASSERT_FALSE(bytes.empty());

  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.moi_runtime_sample_stride = 2u;
  options.moi_track_barriers = false;
  options.moi_track_atomics = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(2);
  options.max_patches = 3u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.moi_private_epoch_automatic);
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::InlineMoiSampledWatchpointStore ||
           patch.kind == ConSanPatchKind::TrampolineMoiSampledWatchpointStore;
  });
  const auto atomic = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiSampledSyncMetadata, &ConSanPatchInfo::kind);
  ASSERT_NE(access, result.patches.end());
  ASSERT_NE(atomic, result.patches.end()) << testing::PrintToString(result.warnings);
  ASSERT_EQ(atomic->owner_descriptor_file_offsets.size(), 2u);
  ASSERT_TRUE(atomic->persistent_owner_private_offset);
  ASSERT_TRUE(atomic->scratch_vgpr);
  ASSERT_TRUE(atomic->persistent_private_state_end);
  EXPECT_EQ(atomic->persistent_epoch_private_offset, access->persistent_epoch_private_offset);
  EXPECT_EQ(atomic->persistent_owner_private_offset, access->persistent_owner_private_offset);
  EXPECT_EQ(atomic->persistent_sample_sequence_private_offset,
            access->persistent_sample_sequence_private_offset);
  EXPECT_EQ(atomic->persistent_private_state_end, 48u);
  EXPECT_EQ(atomic->required_private_segment_size, 96u);
  EXPECT_EQ(atomic->spilled_vgpr_count, 10u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> cave =
      text_words_at_offset(patched, atomic->trampoline_offset, atomic->trampoline_size);
  const auto first_spill = instrumentation::build_private_store_b32(
      *atomic->scratch_vgpr, *atomic->persistent_private_state_end, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(first_spill);
  EXPECT_TRUE(contains_subsequence(cave, *first_spill));
  for (const AmdGpuKernelInfo &kernel : patched.kernels()) {
    if (kernel.name != "shared_owner_0" && kernel.name != "shared_owner_1")
      continue;
    KD descriptor{};
    std::memcpy(&descriptor, result.elf_bytes.data() + kernel.descriptor_file_offset,
                sizeof(descriptor));
    EXPECT_EQ(descriptor.private_segment_fixed_size, 96u);
  }
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, Cdna4SampledAtomicRejectsSpillWhenResultOverwritesGuestAddress) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  const auto access =
      build_cdna4_ds_store_b32(/*vaddr=*/10, /*vdata=*/11, /*byte_offset=*/0, kArch);
  const auto release = cdna4::build_mubuf(cdna4::kBufferWbl2Mubuf, {.sc1 = 1});
  const auto atomic = instrumentation::build_flat_atomic_cmpswap_b32(
      /*vaddr=*/4, /*vsrc=*/6, /*vdst=*/4, /*return_old_value=*/true,
      /*scope=*/2, kArch);
  const auto wait = build_cdna4_s_wait_flat0(kArch);
  ASSERT_TRUE(access && atomic && wait);
  std::vector<uint32_t> text_words;
  text_words.insert(text_words.end(), access->begin(), access->end());
  text_words.insert(text_words.end(), release.begin(), release.end());
  text_words.push_back(*wait);
  text_words.insert(text_words.end(), atomic->begin(), atomic->end());
  text_words.push_back(*wait);
  text_words.resize(800u, build_s_nop(0, kArch));
  text_words.back() = build_s_endpgm(kArch);
  std::vector<uint8_t> bytes =
      make_cdna4_lds_code_object(text_words, "sampled_atomic_result_address_alias",
                                 /*vgpr_granulated=*/3u);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 2u);
  });

  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.moi_runtime_sample_stride = 2u;
  options.moi_track_barriers = false;
  options.moi_track_atomics = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(2);
  options.max_patches = 3u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  EXPECT_EQ(std::ranges::find(result.patches, ConSanPatchKind::TrampolineMoiSampledSyncMetadata,
                              &ConSanPatchInfo::kind),
            result.patches.end());
  // The result/address alias disables guest-first spilling, so the strict
  // scratch-overlap rejection is expected before CAS snapshot emission.
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("scratch-operand-alias") != std::string::npos;
  })) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, Cdna4SampledAtomicRejectsFlatScratchScalarAlias) {
  const auto atomic = build_cdna4_flat_atomic_add_u32(
      /*vaddr=*/4, /*vsrc=*/6, /*vdst=*/7, /*return_old_value=*/true,
      /*scope=*/2, ROCJITSU_CODE_ARCH_CDNA4);
  const auto wait = build_cdna4_s_wait_flat0(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(atomic && wait);
  std::vector<uint32_t> text_words = {
      0xd81a0004u,
      0x00000302u, // ds_write_b32 v2, v3 offset:4
  };
  text_words.insert(text_words.end(), atomic->begin(), atomic->end());
  text_words.push_back(*wait);
  text_words.resize(800, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);
  const std::vector<uint8_t> bytes =
      make_cdna4_lds_code_object(text_words, "sampled_atomic_special_sgpr_alias");
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.moi_track_atomics = true;
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 96;
  options.moi_owner_vgpr = 20;
  options.moi_epoch_vgpr = 21;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(2);
  options.max_patches = 2;

  const ConSanResult result = try_patch_consan(bytes, options);

  EXPECT_EQ(result.outcome, ConSanTransformOutcome::Unsupported);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("architectural special SGPR") != std::string::npos;
  }));
}

TEST(ConSanMoi, SampledAtomicTrackingAdmitsWorkgroupScope) {
  const std::vector<uint8_t> bytes = make_rdna4_sampled_lds_and_ordered_flat_atomic_code_object(
      /*compare_exchange=*/false, /*scope=*/1u);
  ASSERT_FALSE(bytes.empty());
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.moi_track_atomics = true;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 20;
  options.moi_epoch_vgpr = 21;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(2);
  options.max_patches = 2;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_NE(std::ranges::find(result.patches, ConSanPatchKind::TrampolineMoiSampledSyncMetadata,
                              &ConSanPatchInfo::kind),
            result.patches.end())
      << testing::PrintToString(result.warnings);
}

TEST(ConSanMoi, SampledAtomicCasPublishesOnlyEvidenceDerivedOutcomes) {
  const std::vector<uint8_t> bytes =
      make_rdna4_sampled_lds_and_ordered_flat_atomic_code_object(/*compare_exchange=*/true);
  ASSERT_FALSE(bytes.empty());
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.moi_track_atomics = true;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 20;
  options.moi_epoch_vgpr = 21;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(2);
  options.max_patches = 2;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto patch = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiSampledSyncMetadata, &ConSanPatchInfo::kind);
  ASSERT_NE(patch, result.patches.end());
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> trampoline =
      text_words_at_offset(patched, patch->trampoline_offset, patch->trampoline_size);
  const auto descriptor = [&](ConSanMoiSampledSyncOutcome outcome) {
    return encode_consan_moi_sampled_sync_metadata({
        .address = 1,
        .byte_count = 4,
        .kind = ConSanMoiSampledSyncKind::Atomic,
        .role = ConSanMoiSampledSyncRole::RmwRelease,
        .scope = ConSanMoiSampledSyncScope::Agent,
        .outcome = outcome,
    });
  };
  const auto success = descriptor(ConSanMoiSampledSyncOutcome::CasSuccess);
  const auto failure = descriptor(ConSanMoiSampledSyncOutcome::CasFailure);
  ASSERT_EQ(success.classification, ConSanMoiSampledSyncClassification::Valid);
  ASSERT_EQ(failure.classification, ConSanMoiSampledSyncClassification::Valid);
  const auto success_literal =
      build_v_mov_b32_e64_literal(/*vdst=*/10, success.packed.descriptor, ROCJITSU_CODE_ARCH_RDNA4);
  const auto failure_literal =
      build_v_mov_b32_e64_literal(/*vdst=*/10, failure.packed.descriptor, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(success_literal);
  ASSERT_TRUE(failure_literal);
  EXPECT_EQ(count_subsequence(trampoline, *success_literal), 1u);
  EXPECT_EQ(count_subsequence(trampoline, *failure_literal), 1u);
}

TEST(ConSanMoi, SampledAtomicCasWithoutReturnedOldValueFailsClosed) {
  const std::vector<uint8_t> bytes = make_rdna4_ordered_flat_cas_code_object(false);
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.moi_track_atomics = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(1);

  const auto result = try_patch_consan(bytes, options);

  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.patches.empty());
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("no selected LDS access candidates") != std::string::npos;
  }));
}

TEST(ConSanMoi, SampledAtomicForcedSpillPreservesTenVgprWindow) {
  const std::vector<uint8_t> bytes = make_rdna4_sampled_lds_and_ordered_flat_atomic_code_object();
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.moi_track_atomics = true;
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(2);
  options.max_patches = 2;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto patch = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiSampledSyncMetadata, &ConSanPatchInfo::kind);
  ASSERT_NE(patch, result.patches.end());
  EXPECT_EQ(patch->spilled_vgpr_count, 10u);
  EXPECT_GT(patch->required_private_segment_size, 0u);
  EXPECT_GE(result.resource_plan_summary.emitted_spill_patches, 1u);
}

TEST(ConSanMoi, SampledAtomicEdgesAssociateWithTheirOrderedAccessWindows) {
  const std::vector<uint8_t> bytes = make_rdna4_lds_and_ordered_flat_atomic_handoff_code_object();
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.moi_track_atomics = true;
  options.scratch_vgpr = 24;
  options.moi_owner_vgpr = 20;
  options.moi_epoch_vgpr = 21;
  options.moi_report_buffer_address = 0x123456780000ull;
  constexpr uint64_t slot_bytes = direct_sampled_report_bytes(1) - sizeof(ConSanMoiReportHeader);
  options.moi_report_buffer_size = sizeof(ConSanMoiReportHeader) + 4u * slot_bytes;
  options.max_patches = 4;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  std::vector<const ConSanPatchInfo *> sync_patches;
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.kind == ConSanPatchKind::TrampolineMoiSampledSyncMetadata)
      sync_patches.push_back(&patch);
  }
  ASSERT_EQ(sync_patches.size(), 2u) << testing::PrintToString(result.warnings);
  ASSERT_EQ(sync_patches[0]->anchor_offset, 20u);
  ASSERT_EQ(sync_patches[1]->anchor_offset, 32u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const ConSanMoiReportBufferLayout layout =
      consan_moi_direct_sampled_report_buffer_layout_for_bytes(options.moi_report_buffer_size);
  for (uint32_t slot = 0; slot < sync_patches.size(); ++slot) {
    const ConSanPatchInfo &patch = *sync_patches[slot];
    ASSERT_TRUE(patch.scratch_vgpr);
    const uint64_t descriptor_address =
        *options.moi_report_buffer_address +
        (slot == 1 ? layout.sampled_pending_acquires_offset +
                         static_cast<uint64_t>(slot) * sizeof(ConSanMoiSampledPendingAcquireSlot) +
                         offsetof(ConSanMoiSampledPendingAcquireSlot, metadata)
                   : layout.sampled_sync_metadata_offset +
                         static_cast<uint64_t>(slot) * sizeof(ConSanMoiSampledSyncMetadataPacked)) +
        offsetof(ConSanMoiSampledSyncMetadataPacked, descriptor);
    // The pending-acquire publisher now keeps the whole record base in its
    // address VGPR and uses field offsets; the barrier publisher addresses the
    // descriptor directly for its atomic claim.
    const uint64_t materialized_address =
        slot == 1 ? *options.moi_report_buffer_address + layout.sampled_pending_acquires_offset +
                        static_cast<uint64_t>(slot) * sizeof(ConSanMoiSampledPendingAcquireSlot)
                  : descriptor_address;
    const auto materialize = build_v_mov_b32_e64_literal(
        *patch.scratch_vgpr, static_cast<uint32_t>(materialized_address), ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_TRUE(materialize);
    const std::vector<uint32_t> trampoline =
        text_words_at_offset(patched, patch.trampoline_offset, patch.trampoline_size);
    EXPECT_TRUE(contains_subsequence(trampoline, *materialize));
    if (slot == 1) {
      EXPECT_TRUE(contains_subsequence(
          trampoline, make_expected_offset_store_words(
                          offsetof(ConSanMoiSampledPendingAcquireSlot, metadata) +
                              offsetof(ConSanMoiSampledSyncMetadataPacked, descriptor),
                          static_cast<uint16_t>(*patch.scratch_vgpr + 2u), *patch.scratch_vgpr)));
      const uint64_t contention_address =
          *options.moi_report_buffer_address +
          offsetof(ConSanMoiReportHeader, sampled_pending_acquire_contention_count);
      const auto contention = build_v_mov_b32_e64_literal(
          *patch.scratch_vgpr, static_cast<uint32_t>(contention_address), ROCJITSU_CODE_ARCH_RDNA4);
      ASSERT_TRUE(contention);
      EXPECT_TRUE(contains_subsequence(trampoline, *contention));
    }
  }
}

TEST(ConSanMoi, SampledAtomicWeakenedReleaseIsReinventoriedBeforeInstrumentation) {
  const std::vector<uint8_t> bytes = make_rdna4_lds_and_ordered_flat_atomic_handoff_code_object();
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.moi_track_atomics = true;
  options.scratch_vgpr = 24;
  options.moi_owner_vgpr = 20;
  options.moi_epoch_vgpr = 21;
  options.moi_report_buffer_address = 0x123456780000ull;
  constexpr uint64_t slot_bytes = direct_sampled_report_bytes(1) - sizeof(ConSanMoiReportHeader);
  options.moi_report_buffer_size = sizeof(ConSanMoiReportHeader) + 4u * slot_bytes;
  options.max_patches = 4;

  const ConSanResult clean = try_patch_consan(bytes, options);
  ASSERT_TRUE(clean.errors.empty()) << testing::PrintToString(clean.errors);
  EXPECT_EQ(std::ranges::count(clean.patches, ConSanPatchKind::TrampolineMoiSampledSyncMetadata,
                               &ConSanPatchInfo::kind),
            2u);

  options.fault_atomic_weaken_order = true;
  options.fault_atomic_order_edge = ConSanAtomicOrderEdge::Release;
  options.fault_atomic_index = 0;
  options.fault_require_exactly_one = true;
  const ConSanResult fault = try_patch_consan(bytes, options);

  ASSERT_TRUE(fault.errors.empty()) << testing::PrintToString(fault.errors);
  EXPECT_EQ(fault.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_EQ(fault.applied_fault_mutations, 1u);
  EXPECT_TRUE(std::ranges::any_of(fault.patches, [](const ConSanPatchInfo &patch) {
    return patch.phase == ConSanPatchPhase::Mutation &&
           patch.kind == ConSanPatchKind::InlineAtomicOrderRewrite;
  }));
  std::vector<const ConSanPatchInfo *> fault_sync;
  for (const ConSanPatchInfo &patch : fault.patches) {
    if (patch.kind == ConSanPatchKind::TrampolineMoiSampledSyncMetadata)
      fault_sync.push_back(&patch);
  }
  ASSERT_EQ(fault_sync.size(), 1u) << testing::PrintToString(fault.warnings);
  EXPECT_EQ(fault_sync.front()->anchor_offset, 32u);
  EXPECT_EQ(std::ranges::count_if(fault.sync_sequences,
                                  [](const ConSanSyncSequence &sequence) {
                                    return sequence.kind == ConSanSyncSequenceKind::Atomic &&
                                           sequence.memory_role == ConSanSyncMemoryRole::Release;
                                  }),
            0u);
}

TEST(ConSanMoi, DirectSampledProbeAutomaticallyUsesDeadVgprs) {
  std::array<uint32_t, 420> text_words{};
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000000u; // ds_store_b32 v0, v0
  for (size_t i = 2; i + 1 < text_words.size(); ++i)
    text_words[i] = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
  text_words.back() = 0xBFB00000u; // s_endpgm

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(2);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().source, ConSanRegisterAllocationSource::LivenessDead);
  EXPECT_EQ(result.resource_plans.front().scratch_vgpr, 1);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineMoiSampledWatchpointStore);
  EXPECT_EQ(result.patches.front().scratch_vgpr, 1);
}

TEST(ConSanMoi, DirectSampledProbeDoesNotTrustLivenessWithRelativeVgprAccess) {
  std::array<uint32_t, 420> text_words{};
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000000u; // ds_store_b32 v0, v0
  text_words[2] = 0x7E028708u; // v_movrels_b32_e32 v1, v8
  for (size_t i = 3; i + 1 < text_words.size(); ++i)
    text_words[i] = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);

  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT, 0);
  });
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(2);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().source, ConSanRegisterAllocationSource::DescriptorGrowth);
  EXPECT_EQ(result.resource_plans.front().current_vgpr_count, 4u);
  EXPECT_EQ(result.resource_plans.front().scratch_vgpr, 9u);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().scratch_vgpr, 9u);
}

TEST(ConSanMoi, Gfx1250SampledAutomaticExecSaveUsesOwnerLocalWindow) {
  const auto make_owner = [](uint16_t first_live, uint16_t last_live, uint16_t dead_destination) {
    std::vector<uint32_t> words = {
        0xD8340000u,
        0x00000000u, // ds_store_b32 v0, v0
    };
    for (uint16_t sgpr = first_live; sgpr <= last_live; ++sgpr) {
      words.push_back(build_s_mov_b32(dead_destination, sgpr, ROCJITSU_CODE_ARCH_GFX1250));
    }
    words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250));
    return words;
  };
  const std::vector<uint32_t> first_words =
      make_owner(/*first_live=*/8u, /*last_live=*/105u, /*dead_destination=*/0u);
  const std::vector<uint32_t> second_words =
      make_owner(/*first_live=*/0u, /*last_live=*/97u, /*dead_destination=*/97u);
  const std::vector<uint8_t> bytes = make_gfx1250_code_object_with_local_function(
      first_words, second_words, {}, kRdna4Wave64AllVgprsGranulated,
      /*function_is_kernel=*/true);

  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(2);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.max_patches = 2;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  ASSERT_TRUE(result.resolved_moi_exec_save_sgpr);
  ASSERT_FALSE(result.resolved_moi_transient_sgpr_assignments.empty());
  EXPECT_EQ(std::ranges::count_if(result.patches,
                                  [](const ConSanPatchInfo &patch) {
                                    return patch.kind ==
                                               ConSanPatchKind::InlineMoiSampledWatchpointStore ||
                                           patch.kind ==
                                               ConSanPatchKind::TrampolineMoiSampledWatchpointStore;
                                  }),
            2u);
  EXPECT_TRUE(std::ranges::all_of(result.resource_plans, [](const auto &plan) {
    return plan.site_kind != ConSanResourceSiteKind::Access ||
           plan.source != ConSanRegisterAllocationSource::Unsupported;
  }));
}

TEST(ConSanMoi, Gfx1250SampledSpillsExecVccStateWithSeparateDeadDenseRouter) {
  const std::array<uint16_t, 4> dead = {0u, 1u, 4u, 6u};
  std::vector<uint32_t> words;
  for (uint32_t index = 0; index < 9u; ++index) {
    words.push_back(0xD8340000u);
    words.push_back(0x00000000u); // ds_store_b32 v0, v0
  }
  for (uint16_t sgpr = 0; sgpr < 106u; ++sgpr) {
    if (std::ranges::find(dead, sgpr) == dead.end())
      words.push_back(build_s_mov_b32(/*sdst=*/0u, sgpr, ROCJITSU_CODE_ARCH_GFX1250));
  }
  words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250));
  const std::vector<uint8_t> bytes = make_gfx1250_code_object(words);

  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(9);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.max_patches = 9u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.resolved_moi_transient_sgpr_assignments.size(), 1u);
  const ConSanMoiTransientSgprAssignment &assignment =
      result.resolved_moi_transient_sgpr_assignments.front();
  EXPECT_TRUE(assignment.spill_backed);
  EXPECT_EQ(assignment.indirect_pc_sgpr, 0u);
  EXPECT_EQ(assignment.indirect_scc_sgpr, 4u);
  EXPECT_EQ(assignment.dispatch_key_sgpr, 6u);
  EXPECT_EQ(assignment.call_return_sgpr, assignment.indirect_pc_sgpr);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiSampledWatchpointStore,
                               &ConSanPatchInfo::kind),
            9u);
  EXPECT_TRUE(std::ranges::all_of(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind != ConSanPatchKind::TrampolineMoiSampledWatchpointStore ||
           patch.required_private_segment_size > 0u;
  }));
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, Cdna4Wave64AccvgprBoundarySampledUsesPrivatePersistentState) {
  const auto guest = build_cdna4_ds_store_b32(
      /*vaddr=*/10, /*vdata=*/11, /*byte_offset=*/0, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(guest);
  std::vector<uint32_t> text_words(320u, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  std::copy(guest->begin(), guest->end(), text_words.begin());
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);
  std::vector<uint8_t> bytes =
      make_cdna4_lds_code_object(text_words, "sampled_accvgpr_boundary", /*vgpr_granulated=*/3u);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    // Preserve the compiler-defined split: v12 is the first accumulator while
    // ordinary guest code reaches v11.
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 2u);
  });

  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.moi_runtime_sample_stride = 2u;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(2);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.moi_private_epoch_automatic);
  EXPECT_FALSE(result.moi_persistent_vgprs_automatic);
  const auto sampled_patch = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiSampledWatchpointStore, &ConSanPatchInfo::kind);
  ASSERT_NE(sampled_patch, result.patches.end());
  EXPECT_TRUE(sampled_patch->persistent_epoch_private_offset);
  EXPECT_TRUE(sampled_patch->persistent_owner_private_offset);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue,
                               &ConSanPatchInfo::kind),
            1u);
  AmdGpuCodeObject original(bytes.data(), bytes.size());
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(original.is_valid());
  ASSERT_TRUE(patched.is_valid());
  KD original_descriptor{};
  KD patched_descriptor{};
  std::memcpy(&original_descriptor,
              bytes.data() + original.kernels().front().descriptor_file_offset,
              sizeof(original_descriptor));
  std::memcpy(&patched_descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(patched_descriptor));
  EXPECT_EQ(AMDHSA_BITS_GET(patched_descriptor.compute_pgm_rsrc3,
                            kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET),
            AMDHSA_BITS_GET(original_descriptor.compute_pgm_rsrc3,
                            kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET));
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, CdnaSampledBarrierUsesPrivatePersistentStateAtAccvgprBoundary) {
  for (const SampledCdnaTarget &target : kSampledCdnaTargets) {
    for (uint32_t sample_stride : {1u, 2u}) {
      SCOPED_TRACE(testing::Message() << target.label << " sample_stride=" << sample_stride);
      const auto guest =
          target.arch == ROCJITSU_CODE_ARCH_CDNA3
              ? build_cdna3_ds_store_b32(/*vaddr=*/10, /*vdata=*/11, /*byte_offset=*/0, target.arch)
              : build_cdna4_ds_store_b32(/*vaddr=*/10, /*vdata=*/11, /*byte_offset=*/0,
                                         target.arch);
      const auto barrier = target.arch == ROCJITSU_CODE_ARCH_CDNA3
                               ? build_cdna3_s_barrier(target.arch)
                               : build_cdna4_s_barrier(target.arch);
      ASSERT_TRUE(guest && barrier);
      std::vector<uint32_t> text_words(540u, build_s_nop(0, target.arch));
      std::copy(guest->begin(), guest->end(), text_words.begin());
      text_words[400] = *barrier;
      text_words.back() = build_s_endpgm(target.arch);
      std::vector<uint8_t> bytes =
          target.arch == ROCJITSU_CODE_ARCH_CDNA3
              ? make_cdna3_lds_code_object(text_words, "sampled_barrier_accvgpr_boundary",
                                           /*vgpr_granulated=*/3u)
              : make_cdna4_lds_code_object(text_words, "sampled_barrier_accvgpr_boundary",
                                           /*vgpr_granulated=*/3u);
      mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
        // v12 is the first accumulator while ordinary guest code reaches v11.
        AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET,
                        2u);
      });

      ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
      options.moi_runtime_sample_stride = sample_stride;
      options.moi_report_buffer_address = 0x123456780000ull;
      options.moi_report_buffer_size = direct_sampled_report_bytes(2);
      options.moi_track_barriers = true;
      options.moi_track_atomics = false;
      options.max_patches = 3u;

      const ConSanResult result = try_patch_consan(bytes, options);

      ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
      ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
      EXPECT_TRUE(result.moi_private_epoch_automatic);
      EXPECT_FALSE(result.moi_persistent_vgprs_automatic);
      const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
        return patch.kind == ConSanPatchKind::InlineMoiSampledWatchpointStore ||
               patch.kind == ConSanPatchKind::TrampolineMoiSampledWatchpointStore;
      });
      const auto barrier_patch =
          std::ranges::find(result.patches, ConSanPatchKind::TrampolineMoiSampledSyncMetadata,
                            &ConSanPatchInfo::kind);
      const auto prologue =
          std::ranges::find(result.patches, ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue,
                            &ConSanPatchInfo::kind);
      ASSERT_NE(access, result.patches.end());
      ASSERT_NE(barrier_patch, result.patches.end());
      ASSERT_NE(prologue, result.patches.end());
      EXPECT_TRUE(access->persistent_epoch_private_offset);
      EXPECT_TRUE(access->persistent_owner_private_offset);
      EXPECT_EQ(access->persistent_sample_sequence_private_offset.has_value(), sample_stride > 1u);
      EXPECT_EQ(barrier_patch->persistent_epoch_private_offset,
                access->persistent_epoch_private_offset);
      EXPECT_EQ(barrier_patch->persistent_owner_private_offset,
                access->persistent_owner_private_offset);
      EXPECT_EQ(barrier_patch->persistent_sample_sequence_private_offset,
                access->persistent_sample_sequence_private_offset);
      EXPECT_EQ(prologue->persistent_epoch_private_offset, access->persistent_epoch_private_offset);
      EXPECT_EQ(prologue->persistent_owner_private_offset, access->persistent_owner_private_offset);
      EXPECT_EQ(prologue->persistent_sample_sequence_private_offset,
                access->persistent_sample_sequence_private_offset);

      ASSERT_TRUE(barrier_patch->scratch_vgpr);
      ASSERT_TRUE(barrier_patch->persistent_owner_private_offset);
      AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
      ASSERT_TRUE(patched.is_valid());
      const std::vector<uint32_t> cave = text_words_at_offset(
          patched, barrier_patch->trampoline_offset, barrier_patch->trampoline_size);
      const uint16_t owner_vgpr = static_cast<uint16_t>(*barrier_patch->scratch_vgpr + 7u);
      const auto owner_load = instrumentation::build_private_load_b32(
          owner_vgpr, *barrier_patch->persistent_owner_private_offset, target.arch);
      const auto captured_owner = instrumentation::build_v_lshrrev_b32(
          owner_vgpr, scalar_positive_inline_u32(6u), owner_vgpr, target.arch);
      const auto live_owner = instrumentation::build_v_lshrrev_b32(
          owner_vgpr, scalar_positive_inline_u32(6u), /*workitem_id_x=*/0u, target.arch);
      ASSERT_TRUE(owner_load && captured_owner && live_owner);
      EXPECT_TRUE(contains_subsequence(cave, *owner_load));
      EXPECT_NE(std::ranges::find(cave, *captured_owner), cave.end());
      EXPECT_EQ(std::ranges::find(cave, *live_owner), cave.end());
      EXPECT_TRUE(result.final_validation_passed);
    }
  }
}

TEST(ConSanMoi, DirectSampledProbeSpillsFiveVgprsInAppendedCave) {
  const std::array<uint32_t, 3> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32 v0, v0
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    descriptor.private_segment_fixed_size = 32;
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT, 1u);
  });
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(2);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().source, ConSanRegisterAllocationSource::SpillRequired);
  EXPECT_EQ(result.resource_plans.front().scratch_vgpr, 1);
  ASSERT_EQ(result.patches.size(), 1u);
  const ConSanPatchInfo &patch = result.patches.front();
  EXPECT_EQ(patch.kind, ConSanPatchKind::TrampolineMoiSampledWatchpointStore);
  EXPECT_EQ(patch.scratch_vgpr, 1);
  EXPECT_EQ(patch.spilled_vgpr_count, 5u);
  EXPECT_EQ(patch.required_private_segment_size, 52u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const auto *text = patched.text_sections().front();
  std::vector<uint32_t> actual(text->size() / sizeof(uint32_t));
  std::memcpy(actual.data(), text->data(), text->size());
  const std::vector<uint32_t> save =
      expected_vgpr_spill_words(1, 5, /*restore=*/false, /*slot_base=*/32);
  const std::vector<uint32_t> restore =
      expected_vgpr_spill_words(1, 5, /*restore=*/true, /*slot_base=*/32);
  ASSERT_FALSE(save.empty());
  ASSERT_FALSE(restore.empty());
  const size_t cave = patch.trampoline_offset / sizeof(uint32_t);
  const auto owner_init =
      build_v_lshrrev_b32_e32(5, scalar_positive_inline_u32(6), 0, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(owner_init);
  ASSERT_LE(cave + save.size() + restore.size() + 4u, actual.size());
  EXPECT_TRUE(std::equal(save.begin(), save.end(), actual.begin() + cave));
  EXPECT_EQ(actual[cave + save.size()], *owner_init);
  EXPECT_EQ(actual[cave + save.size() + 1u], text_words[0]);
  EXPECT_EQ(actual[cave + save.size() + 2u], text_words[1]);
  EXPECT_TRUE(std::equal(restore.begin(), restore.end(), actual.end() - 1u - restore.size()));

  ASSERT_EQ(patched.kernels().size(), 1u);
  KD descriptor{};
  const uint64_t descriptor_offset = patched.kernels().front().descriptor_file_offset;
  std::memcpy(&descriptor, result.elf_bytes.data() + descriptor_offset, sizeof(descriptor));
  EXPECT_EQ(descriptor.private_segment_fixed_size, 52u);
  EXPECT_EQ(
      AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT),
      1u);
}

TEST(ConSanMoi, DirectSampledProbeCanUseSleepDelay) {
  std::array<uint32_t, 420> text_words{};
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000000u; // ds_store_b32 v0, v0
  for (size_t i = 2; i + 1 < text_words.size(); ++i)
    text_words[i] = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 13;
  options.moi_epoch_vgpr = 14;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(2);
  options.delay_mode = ConSanDelayMode::Sleep;
  options.delay_nops = 7;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineMoiSampledWatchpointStore);

  const std::vector<uint32_t> rewritten_words =
      patched_words_at_file_offset(result, 0x100, result.patches.front().original_size);
  EXPECT_NE(std::find(rewritten_words.begin(), rewritten_words.end(),
                      build_s_sleep(7, ROCJITSU_CODE_ARCH_RDNA4)),
            rewritten_words.end());
}

TEST(ConSanMoi, DirectSampledProbeCanUseSleepVarDelay) {
  std::array<uint32_t, 420> text_words{};
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000000u; // ds_store_b32 v0, v0
  for (size_t i = 2; i + 1 < text_words.size(); ++i)
    text_words[i] = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 13;
  options.moi_epoch_vgpr = 14;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(2);
  options.delay_mode = ConSanDelayMode::SleepVar;
  options.delay_nops = 1;
  options.delay_var_ssrc = kRdna4VccLo;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineMoiSampledWatchpointStore);

  const std::vector<uint32_t> rewritten_words =
      patched_words_at_file_offset(result, 0x100, result.patches.front().original_size);
  EXPECT_NE(std::find(rewritten_words.begin(), rewritten_words.end(),
                      build_s_sleep_var(kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4)),
            rewritten_words.end());
}

TEST(ConSanMoi, DirectSampledProbeRejectsOversizedSleepVarSource) {
  std::array<uint32_t, 420> text_words{};
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000000u; // ds_store_b32 v0, v0
  for (size_t i = 2; i + 1 < text_words.size(); ++i)
    text_words[i] = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 13;
  options.moi_epoch_vgpr = 14;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(2);
  options.delay_mode = ConSanDelayMode::SleepVar;
  options.delay_nops = 1;
  options.delay_var_ssrc = 300;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_FALSE(result.modified);

  bool saw_sleep_var_warning = false;
  for (const std::string &warning : result.warnings)
    saw_sleep_var_warning |=
        warning.find("sleep_var source exceeds the 8-bit scalar source field") != std::string::npos;
  EXPECT_TRUE(saw_sleep_var_warning);
}

TEST(ConSanMoi, DirectSampledProbeCanStrideCandidateSelection) {
  constexpr uint32_t kSecondSiteWord = 350;
  std::vector<uint32_t> text_words(800, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000000u; // ds_store_b32 v0, v0
  text_words[kSecondSiteWord] = 0xD8D80000u;
  text_words[kSecondSiteWord + 1] = 0x01000000u; // ds_load_b32 v1, v0
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.scratch_vgpr = 20;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(2);
  options.max_patches = 2;
  options.moi_sample_stride = 2;
  options.moi_sample_offset = 1;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineMoiSampledWatchpointStore);
  EXPECT_EQ(result.patches.front().anchor_offset, kSecondSiteWord * sizeof(uint32_t));
}

TEST(ConSanMoi, DirectSampledProbeRuntimeAddressSelectionKeepsAllSitesPatchable) {
  constexpr uint32_t kSecondSiteWord = 350;
  std::vector<uint32_t> text_words(800, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000000u; // ds_store_b32 v0, v0
  text_words[kSecondSiteWord] = 0xD8D80000u;
  text_words[kSecondSiteWord + 1] = 0x01000000u; // ds_load_b32 v1, v0
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "lds_probe", kRdna4Wave64AllVgprsGranulated, false, false,
      /*workgroup_id_dimension_mask=*/7);
  AmdGpuCodeObject original(bytes.data(), bytes.size());
  ASSERT_EQ(original.kernels().size(), 1u);
  KD original_descriptor{};
  std::memcpy(&original_descriptor,
              bytes.data() + original.kernels().front().descriptor_file_offset,
              sizeof(original_descriptor));
  ASSERT_EQ(AMDHSA_BITS_GET(original_descriptor.compute_pgm_rsrc2,
                            kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X),
            1u);
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(16);
  options.max_patches = 16;
  options.moi_runtime_sample_stride = 4;
  options.moi_runtime_sample_offset = 1;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 3u);
  EXPECT_EQ(result.patches[0].anchor_offset, 0u);
  EXPECT_EQ(result.patches[1].anchor_offset, kSecondSiteWord * sizeof(uint32_t));
  EXPECT_EQ(result.patches[0].sampled_first_slot, 0u);
  EXPECT_EQ(result.patches[0].sampled_window_bank_count, 8u);
  EXPECT_EQ(result.patches[1].sampled_first_slot, 8u);
  EXPECT_EQ(result.patches[1].sampled_window_bank_count, 8u);
  EXPECT_TRUE(result.moi_exec_save_sgprs_automatic);
  ASSERT_TRUE(result.resolved_moi_exec_save_sgpr);
  EXPECT_TRUE(result.moi_dispatch_id_sgprs_automatic);
  EXPECT_TRUE(result.resolved_moi_dispatch_id_sgpr.has_value());
  EXPECT_TRUE(result.moi_persistent_vgprs_automatic);
  EXPECT_FALSE(result.moi_private_epoch_automatic);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue,
                               &ConSanPatchInfo::kind),
            1u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const auto *text_section = patched.text_sections().front();
  ASSERT_EQ(text_section->size() % sizeof(uint32_t), 0u);
  std::vector<uint32_t> patched_words(text_section->size() / sizeof(uint32_t));
  std::memcpy(patched_words.data(), text_section->data(), text_section->size());

  const uint16_t save_sgpr = static_cast<uint16_t>(*result.resolved_moi_exec_save_sgpr + 2u);
  const auto save_vcc = build_s_mov_b64(save_sgpr, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_vcc = build_s_mov_b64(kRdna4VccLo, save_sgpr, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(save_vcc);
  ASSERT_TRUE(restore_vcc);
  EXPECT_EQ(std::count(patched_words.begin(), patched_words.end(), *save_vcc), 2);
  EXPECT_EQ(std::count(patched_words.begin(), patched_words.end(), *restore_vcc), 2);
  const uint16_t scc_save_sgpr = static_cast<uint16_t>(*result.resolved_moi_exec_save_sgpr + 6u);
  const auto save_scc =
      build_rdna4_s_cselect_b32(scc_save_sgpr, scalar_positive_inline_u32(1),
                                scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_scc = build_rdna4_s_cmp_lg_u32(scc_save_sgpr, scalar_positive_inline_u32(0),
                                                    ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(save_scc);
  ASSERT_TRUE(restore_scc);
  EXPECT_EQ(std::count(patched_words.begin(), patched_words.end(), *save_scc), 2);
  EXPECT_EQ(std::count(patched_words.begin(), patched_words.end(), *restore_scc), 2);
  const uint16_t publication_exec_save_sgpr =
      static_cast<uint16_t>(*result.resolved_moi_exec_save_sgpr + 4u);
  const auto narrow_claim =
      build_s_and_saveexec_b64(publication_exec_save_sgpr, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_exec =
      build_s_mov_b64(kRdna4ExecLo, publication_exec_save_sgpr, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(narrow_claim);
  ASSERT_TRUE(restore_exec);
  // Each site first narrows to the selected LDS-cell residue, then narrows for
  // the winning publisher and collision-accounting paths.
  EXPECT_EQ(std::count(patched_words.begin(), patched_words.end(), *narrow_claim), 7);
  EXPECT_EQ(std::count(patched_words.begin(), patched_words.end(), *restore_exec), 6);
  const auto selected_origin = build_v_cmp_eq_u32_e32_vcc(
      scalar_positive_inline_u32(0),
      static_cast<uint16_t>(*result.patches.front().scratch_vgpr + 2u), ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(selected_origin);
  // Every site compares the atomic-claim result with zero. Entry-captured
  // owner/epoch initialization may use the same encoding elsewhere in the
  // object, so do not mistake that prologue for workgroup preselection.
  EXPECT_GE(std::count(patched_words.begin(), patched_words.end(), *selected_origin), 2);
  for (const ConSanPatchInfo &patch : std::span<const ConSanPatchInfo>(result.patches).first(2u)) {
    ASSERT_TRUE(patch.scratch_vgpr);
    const uint16_t scratch = *patch.scratch_vgpr;
    const uint16_t low_vgpr = static_cast<uint16_t>(*patch.scratch_vgpr + 2u);
    const auto claim = build_flat_atomic_cmpswap_b32_vaddr_vsrc_vdst(
        scratch, low_vgpr, low_vgpr, /*return_old_value=*/true, /*scope=*/2,
        ROCJITSU_CODE_ARCH_RDNA4);
    const uint16_t high_vgpr = static_cast<uint16_t>(*patch.scratch_vgpr + 3u);
    const auto select_cell = build_v_lshrrev_b32_e32(
        low_vgpr, scalar_positive_inline_u32(consan_moi_sampled_watchpoint::granule_shift),
        low_vgpr, ROCJITSU_CODE_ARCH_RDNA4);
    const auto select_residue =
        build_v_and_b32_e32_literal(low_vgpr, 3u, low_vgpr, ROCJITSU_CODE_ARCH_RDNA4);
    const auto selected_value =
        build_v_mov_b32_e64_literal(high_vgpr, /*literal=*/1, ROCJITSU_CODE_ARCH_RDNA4);
    const auto selected = build_v_cmp_eq_u32_e32_vcc(vector_source_vgpr(high_vgpr), low_vgpr,
                                                     ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_TRUE(claim);
    ASSERT_TRUE(select_cell);
    ASSERT_TRUE(select_residue);
    ASSERT_TRUE(selected_value);
    ASSERT_TRUE(selected);
    EXPECT_TRUE(contains_subsequence(patched_words, *claim));
    EXPECT_NE(std::find(patched_words.begin(), patched_words.end(), *select_cell),
              patched_words.end());
    EXPECT_TRUE(contains_subsequence(patched_words, *select_residue));
    EXPECT_TRUE(contains_subsequence(patched_words, *selected_value));
    EXPECT_NE(std::find(patched_words.begin(), patched_words.end(), *selected),
              patched_words.end());
  }
}

TEST(ConSanMoi, Gfx1250RuntimeSamplingUsesLiteralDispatchIdAtFullScalarPressure) {
  std::vector<uint32_t> text_words(800, build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250));
  constexpr auto store = gfx1250::build_vds(gfx1250::kDsStoreB32Vds, {.addr = 0, .data0 = 1});
  text_words[0] = store[0];
  text_words[1] = store[1];
  text_words[2] = build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/106u, ROCJITSU_CODE_ARCH_GFX1250);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250);
  const std::vector<uint8_t> bytes = make_gfx1250_code_object(text_words);

  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_generation = 7u;
  options.moi_report_dispatch_id = 0x1122334455667788ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(16u);
  options.moi_runtime_sample_stride = 4u;
  options.max_patches = 16u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_FALSE(result.resolved_moi_dispatch_id_sgpr);
  EXPECT_EQ(std::ranges::count_if(result.patches,
                                  [](const ConSanPatchInfo &patch) {
                                    return patch.kind ==
                                               ConSanPatchKind::InlineMoiSampledWatchpointStore ||
                                           patch.kind ==
                                               ConSanPatchKind::TrampolineMoiSampledWatchpointStore;
                                  }),
            1u);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_NE(result.resource_plans.front().source, ConSanRegisterAllocationSource::Unsupported);
}

TEST(ConSanMoi, DirectSampledProbeCanCheckPriorSlotInKernel) {
  constexpr uint32_t kSecondSiteWord = 500;
  std::vector<uint32_t> text_words(1100, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000000u; // ds_store_b32 v0, v0
  text_words[kSecondSiteWord] = 0xD8D80000u;
  text_words[kSecondSiteWord + 1] = 0x01000000u; // ds_load_b32 v1, v0
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.moi_sampled_check = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(2);
  options.max_patches = 2;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 2u);
  EXPECT_TRUE(result.moi_exec_save_sgprs_automatic);
  ASSERT_TRUE(result.patches[1].scratch_vgpr);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const auto *text_section = patched.text_sections().front();
  ASSERT_EQ(text_section->size() % sizeof(uint32_t), 0u);
  std::vector<uint32_t> patched_words(text_section->size() / sizeof(uint32_t));
  std::memcpy(patched_words.data(), text_section->data(), text_section->size());

  const uint16_t scratch = *result.patches[1].scratch_vgpr;
  const auto diagnostic_increment = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      scratch, static_cast<uint16_t>(scratch + 4u), static_cast<uint16_t>(scratch + 4u),
      /*return_old_value=*/true, /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(diagnostic_increment);
  // The immediate-conflict increment shares its opcode/register shape with
  // this site's claimed-window and dropped-claim counters.
  EXPECT_EQ(count_subsequence(patched_words, *diagnostic_increment), 3u);
  const auto atomic_snapshot = build_flat_atomic_add_u64_vaddr_vsrc_vdst(
      scratch, static_cast<uint16_t>(scratch + 5u), static_cast<uint16_t>(scratch + 5u),
      /*return_old_value=*/true, /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(atomic_snapshot);
  EXPECT_EQ(count_subsequence(patched_words, *atomic_snapshot), 1u);
}

TEST(ConSanMoi, Gfx1250SampledFastGatesKeepThousandPackedSitesReachable) {
  constexpr uint32_t kSiteCount = 1000;
  constexpr auto store = gfx1250::build_vds(gfx1250::kDsStoreB32Vds, {.addr = 0, .data0 = 1});
  std::vector<uint32_t> text_words;
  text_words.reserve(2u * kSiteCount + 1u);
  for (uint32_t i = 0; i < kSiteCount; ++i) {
    text_words.push_back(store[0]);
    text_words.push_back(store[1]);
  }
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250));

  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.scratch_vgpr = 20;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(kSiteCount);
  options.max_patches = kSiteCount;
  options.moi_runtime_sample_stride = 16384;
  options.moi_runtime_sample_offset = 0;

  const ConSanResult result = try_patch_consan(
      make_gfx1250_code_object(text_words, "gfx1250_thousand_sampled_sites"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_EQ(std::ranges::count_if(result.patches,
                                  [](const ConSanPatchInfo &patch) {
                                    return patch.kind ==
                                           ConSanPatchKind::TrampolineMoiSampledWatchpointStore;
                                  }),
            kSiteCount)
      << testing::PrintToString(result.warnings);
}

TEST(ConSanMoi, DirectSampledProbeCanCheckCorrespondingPriorBank) {
  constexpr uint32_t kSecondSiteWord = 500;
  std::vector<uint32_t> text_words(1100, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000000u; // ds_store_b32 v0, v0
  text_words[kSecondSiteWord] = 0xD8D80000u;
  text_words[kSecondSiteWord + 1] = 0x01000000u; // ds_load_b32 v1, v0
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "lds_probe", kRdna4Wave64AllVgprsGranulated, false, false,
      /*workgroup_id_dimension_mask=*/7);
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.moi_sampled_check = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(16);
  options.max_patches = 16;
  options.moi_runtime_sample_stride = 4;
  options.moi_runtime_sample_offset = 1;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 3u);
  EXPECT_EQ(result.patches[0].sampled_first_slot, 0u);
  EXPECT_EQ(result.patches[0].sampled_window_bank_count, 8u);
  EXPECT_EQ(result.patches[1].sampled_first_slot, 8u);
  EXPECT_EQ(result.patches[1].sampled_window_bank_count, 8u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const auto *text_section = patched.text_sections().front();
  ASSERT_EQ(text_section->size() % sizeof(uint32_t), 0u);
  std::vector<uint32_t> patched_words(text_section->size() / sizeof(uint32_t));
  std::memcpy(patched_words.data(), text_section->data(), text_section->size());

  ASSERT_TRUE(result.patches[1].scratch_vgpr);
  const uint16_t scratch = *result.patches[1].scratch_vgpr;
  const auto atomic_snapshot = build_flat_atomic_add_u64_vaddr_vsrc_vdst(
      scratch, static_cast<uint16_t>(scratch + 5u), static_cast<uint16_t>(scratch + 5u),
      /*return_old_value=*/true, /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  const auto bank_offset = build_v_mul_lo_u32_vop3_literal(
      static_cast<uint16_t>(scratch + 4u), sizeof(uint64_t), static_cast<uint16_t>(scratch + 7u),
      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(atomic_snapshot);
  ASSERT_TRUE(bank_offset);
  EXPECT_EQ(count_subsequence(patched_words, *atomic_snapshot), 1u);
  EXPECT_TRUE(contains_subsequence(patched_words, *bank_offset));
}

TEST(ConSanMoi, DirectSampledProbeChecksEveryPriorMultiAddressRange) {
  constexpr uint32_t kSecondSiteWord = 500;
  std::vector<uint32_t> text_words(1200, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words[0] = 0xD8380201u;
  text_words[1] = 0x00000000u; // ds_store_2addr_b32 v0, v0, v0 offset0:1 offset1:2
  text_words[kSecondSiteWord] = 0xD9DC0201u;
  text_words[kSecondSiteWord + 1] = 0x01000009u; // ds_load_2addr_b64 v[1:4], v9
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "lds_probe", kRdna4Wave64AllVgprsGranulated, false, false,
      /*workgroup_id_dimension_mask=*/7);
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.moi_sampled_check = true;
  options.scratch_vgpr = 20;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(32);
  options.max_patches = 32;
  options.moi_runtime_sample_stride = 4;
  options.moi_runtime_sample_offset = 1;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.patches.size(), 2u) << testing::PrintToString(result.warnings);
  EXPECT_EQ(result.patches[0].sampled_access_range_count, 2u);
  EXPECT_EQ(result.patches[1].sampled_access_range_count, 2u);
  EXPECT_EQ(result.patches[0].sampled_first_slot, 0u);
  EXPECT_EQ(result.patches[1].sampled_first_slot, 16u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const auto *text = patched.text_sections().front();
  std::vector<uint32_t> patched_words(text->size() / sizeof(uint32_t));
  std::memcpy(patched_words.data(), text->data(), text->size());

  ASSERT_TRUE(result.patches[1].scratch_vgpr);
  const uint16_t scratch = *result.patches[1].scratch_vgpr;
  const auto atomic_snapshot = build_flat_atomic_add_u64_vaddr_vsrc_vdst(
      scratch, static_cast<uint16_t>(scratch + 5u), static_cast<uint16_t>(scratch + 5u),
      /*return_old_value=*/true, /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(atomic_snapshot);
  // Each of the load's two address ranges checks both ranges from the prior
  // store. This covers transposes whose conflicting pair crosses range ordinals.
  EXPECT_EQ(count_subsequence(patched_words, *atomic_snapshot), 4u);
}

TEST(ConSanMoi, DirectSampledProbeWarnsWhenReportCapacityLimitsPatches) {
  std::vector<uint32_t> text_words(800, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000000u; // ds_store_b32 v0, v0
  text_words[350] = 0xD8D80000u;
  text_words[351] = 0x01000000u; // ds_load_b32 v1, v0
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.scratch_vgpr = 20;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(1);
  options.max_patches = 2;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_TRUE(result.patches.front().kind == ConSanPatchKind::InlineMoiSampledWatchpointStore ||
              result.patches.front().kind == ConSanPatchKind::TrampolineMoiSampledWatchpointStore);

  bool saw_limit_warning = false;
  for (const std::string &warning : result.warnings) {
    saw_limit_warning |= warning.find("limited sampled patches to 1 of 2") != std::string::npos;
  }
  EXPECT_TRUE(saw_limit_warning);
}

TEST(ConSanMoi, SampledRuntimeGateUsesExpandedBranchIslands) {
  std::vector<uint32_t> text_words;
  for (uint32_t i = 0; i < 9u; ++i) {
    text_words.push_back(0xD8340000u); // ds_store_b32 v0, v0
    text_words.push_back(0x00000000u);
  }
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4));
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "lds_probe", kRdna4Wave64AllVgprsGranulated, /*wave32=*/false,
      /*uses_dynamic_stack=*/false, /*workgroup_id_dimension_mask=*/1u);
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 80;
  options.moi_dispatch_id_sgpr = 70;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(18);
  options.moi_runtime_sample_stride = 16384;
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.max_patches = 18;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiSampledWatchpointStore,
                               &ConSanPatchInfo::kind),
            9);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiIndirectBranchIsland,
                               &ConSanPatchInfo::kind),
            9);
  const auto access_patch = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiSampledWatchpointStore, &ConSanPatchInfo::kind);
  ASSERT_NE(access_patch, result.patches.end());
  ASSERT_TRUE(access_patch->scratch_vgpr);
  EXPECT_EQ(access_patch->sampled_window_bank_count, 2u);
  std::vector<uint32_t> patched_words(patched.text_sections().front()->size() / sizeof(uint32_t));
  std::memcpy(patched_words.data(), patched.text_sections().front()->data(),
              patched_words.size() * sizeof(uint32_t));
  const uint16_t bank_vgpr = static_cast<uint16_t>(*access_patch->scratch_vgpr + 5u);
  EXPECT_NE(
      std::ranges::find(patched_words, build_v_mov_b32_e32(bank_vgpr, *options.moi_dispatch_id_sgpr,
                                                           ROCJITSU_CODE_ARCH_RDNA4)),
      patched_words.end());
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.kind == ConSanPatchKind::TrampolineMoiIndirectBranchIsland) {
      // Offset zero needs the 21-word fixed gate plus the displaced two-word
      // instruction, rather than the former unconditional 40-word reservation.
      EXPECT_EQ(patch.trampoline_size, 23u * sizeof(uint32_t));
      uint32_t workgroup_mix = 0;
      std::memcpy(&workgroup_mix,
                  patched.text_sections().front()->data() + patch.trampoline_offset +
                      2u * sizeof(uint32_t),
                  sizeof(workgroup_mix));
      EXPECT_EQ(workgroup_mix,
                *build_s_sub_u32(/*sdst=*/86, /*ssrc0=*/86, ttmp_scalar_operand(kTtmpRdna4GridX),
                                 ROCJITSU_CODE_ARCH_RDNA4));
    }
  }
}

TEST(ConSanMoi, SampledSyncMetadataAbiHasStablePackedLayout) {
  static_assert(sizeof(ConSanMoiSampledSyncMetadataPacked) == 24);
  static_assert(alignof(ConSanMoiSampledSyncMetadataPacked) == 8);
  EXPECT_EQ(offsetof(ConSanMoiSampledSyncMetadataPacked, address), 0u);
  EXPECT_EQ(offsetof(ConSanMoiSampledSyncMetadataPacked, byte_count), 8u);
  EXPECT_EQ(offsetof(ConSanMoiSampledSyncMetadataPacked, descriptor), 12u);
  EXPECT_EQ(offsetof(ConSanMoiSampledSyncMetadataPacked, epoch_before), 16u);
  EXPECT_EQ(offsetof(ConSanMoiSampledSyncMetadataPacked, epoch_after), 20u);
  EXPECT_EQ(consan_moi_sampled_sync_abi::version, 1u);
  EXPECT_EQ(consan_moi_sampled_sync_abi::reserved_mask, 0xff000000u);
}

TEST(ConSanMoi, SampledSyncMetadataRoundTripsTypedAtomicRolesAndOutcomes) {
  constexpr std::array cases = {
      std::pair{ConSanMoiSampledSyncRole::Release, ConSanMoiSampledSyncOutcome::NotApplicable},
      std::pair{ConSanMoiSampledSyncRole::Acquire, ConSanMoiSampledSyncOutcome::NotApplicable},
      std::pair{ConSanMoiSampledSyncRole::Rmw, ConSanMoiSampledSyncOutcome::RmwNoReturn},
      std::pair{ConSanMoiSampledSyncRole::RmwRelease, ConSanMoiSampledSyncOutcome::RmwReturnsOld},
      std::pair{ConSanMoiSampledSyncRole::RmwAcquire, ConSanMoiSampledSyncOutcome::CasFailure},
      std::pair{ConSanMoiSampledSyncRole::RmwAcquireRelease,
                ConSanMoiSampledSyncOutcome::CasSuccess},
  };
  for (const auto &[role, outcome] : cases) {
    const ConSanMoiSampledSyncMetadata metadata{
        .address = 0xfedcba9876543210ull,
        .byte_count = 16,
        .kind = ConSanMoiSampledSyncKind::Atomic,
        .role = role,
        .scope = ConSanMoiSampledSyncScope::Agent,
        .outcome = outcome,
        .epoch_before = 0x12345678u,
        .epoch_after = 0x12345678u,
    };
    const ConSanMoiSampledSyncEncodeResult encoded =
        encode_consan_moi_sampled_sync_metadata(metadata);
    EXPECT_EQ(encoded.classification, ConSanMoiSampledSyncClassification::Valid);
    EXPECT_EQ(encoded.packed.address, metadata.address);
    EXPECT_EQ(encoded.packed.byte_count, metadata.byte_count);
    const ConSanMoiSampledSyncDecodeResult decoded =
        decode_consan_moi_sampled_sync_metadata(encoded.packed);
    EXPECT_EQ(decoded.classification, ConSanMoiSampledSyncClassification::Valid);
    EXPECT_EQ(decoded.metadata, metadata);
  }
}

TEST(ConSanMoi, SampledSyncMetadataRoundTripsBarrierEpochTransition) {
  const ConSanMoiSampledSyncMetadata metadata{
      .kind = ConSanMoiSampledSyncKind::Barrier,
      .role = ConSanMoiSampledSyncRole::AcquireRelease,
      .scope = ConSanMoiSampledSyncScope::Workgroup,
      .epoch_before = 41,
      .epoch_after = 42,
  };
  const ConSanMoiSampledSyncEncodeResult encoded =
      encode_consan_moi_sampled_sync_metadata(metadata);
  ASSERT_EQ(encoded.classification, ConSanMoiSampledSyncClassification::Valid);
  const ConSanMoiSampledSyncDecodeResult decoded =
      decode_consan_moi_sampled_sync_metadata(encoded.packed);
  EXPECT_EQ(decoded.classification, ConSanMoiSampledSyncClassification::Valid);
  EXPECT_EQ(decoded.metadata, metadata);
}

TEST(ConSanMoi, SampledSyncMetadataRejectsVersionKindRoleScopeAndOutcome) {
  const ConSanMoiSampledSyncMetadata valid{
      .address = 0x1000,
      .byte_count = 4,
      .kind = ConSanMoiSampledSyncKind::Atomic,
      .role = ConSanMoiSampledSyncRole::RmwAcquire,
      .scope = ConSanMoiSampledSyncScope::Workgroup,
      .outcome = ConSanMoiSampledSyncOutcome::RmwReturnsOld,
      .epoch_before = 7,
      .epoch_after = 7,
  };
  const auto classify = [](ConSanMoiSampledSyncMetadata metadata) {
    const ConSanMoiSampledSyncEncodeResult encoded =
        encode_consan_moi_sampled_sync_metadata(metadata);
    EXPECT_EQ(encoded.packed, ConSanMoiSampledSyncMetadataPacked{});
    return encoded.classification;
  };
  auto candidate = valid;
  candidate.version = 2;
  EXPECT_EQ(classify(candidate), ConSanMoiSampledSyncClassification::UnsupportedVersion);
  candidate = valid;
  candidate.kind = static_cast<ConSanMoiSampledSyncKind>(15);
  EXPECT_EQ(classify(candidate), ConSanMoiSampledSyncClassification::UnsupportedKind);
  candidate = valid;
  candidate.role = static_cast<ConSanMoiSampledSyncRole>(15);
  EXPECT_EQ(classify(candidate), ConSanMoiSampledSyncClassification::UnsupportedRole);
  candidate = valid;
  candidate.scope = static_cast<ConSanMoiSampledSyncScope>(15);
  EXPECT_EQ(classify(candidate), ConSanMoiSampledSyncClassification::UnsupportedScope);
  candidate = valid;
  candidate.outcome = static_cast<ConSanMoiSampledSyncOutcome>(15);
  EXPECT_EQ(classify(candidate), ConSanMoiSampledSyncClassification::UnsupportedOutcome);
}

TEST(ConSanMoi, SampledSyncMetadataRejectsInvalidAndOverflowingRanges) {
  ConSanMoiSampledSyncMetadata metadata{
      .address = 0x1000,
      .byte_count = 4,
      .kind = ConSanMoiSampledSyncKind::Atomic,
      .role = ConSanMoiSampledSyncRole::Acquire,
      .scope = ConSanMoiSampledSyncScope::System,
      .epoch_before = 9,
      .epoch_after = 9,
  };
  metadata.address = 0;
  EXPECT_EQ(classify_consan_moi_sampled_sync_metadata(metadata),
            ConSanMoiSampledSyncClassification::InvalidRange);
  metadata.address = 0x1000;
  metadata.byte_count = 0;
  EXPECT_EQ(classify_consan_moi_sampled_sync_metadata(metadata),
            ConSanMoiSampledSyncClassification::InvalidRange);
  metadata.address = std::numeric_limits<uint64_t>::max() - 2u;
  metadata.byte_count = 4;
  EXPECT_EQ(classify_consan_moi_sampled_sync_metadata(metadata),
            ConSanMoiSampledSyncClassification::RangeOverflow);
}

TEST(ConSanMoi, SampledSyncMetadataRejectsUnsupportedAtomicAndBarrierSequences) {
  ConSanMoiSampledSyncMetadata atomic{
      .address = 0x1000,
      .byte_count = 4,
      .kind = ConSanMoiSampledSyncKind::Atomic,
      .role = ConSanMoiSampledSyncRole::RmwAcquire,
      .scope = ConSanMoiSampledSyncScope::Workgroup,
      .outcome = ConSanMoiSampledSyncOutcome::RmwReturnsOld,
      .epoch_before = 9,
      .epoch_after = 9,
  };
  atomic.outcome = ConSanMoiSampledSyncOutcome::NotApplicable;
  EXPECT_EQ(classify_consan_moi_sampled_sync_metadata(atomic),
            ConSanMoiSampledSyncClassification::UnsupportedSequence);
  atomic.role = ConSanMoiSampledSyncRole::Acquire;
  atomic.outcome = ConSanMoiSampledSyncOutcome::CasSuccess;
  EXPECT_EQ(classify_consan_moi_sampled_sync_metadata(atomic),
            ConSanMoiSampledSyncClassification::UnsupportedSequence);
  atomic.role = ConSanMoiSampledSyncRole::AcquireRelease;
  atomic.outcome = ConSanMoiSampledSyncOutcome::NotApplicable;
  EXPECT_EQ(classify_consan_moi_sampled_sync_metadata(atomic),
            ConSanMoiSampledSyncClassification::UnsupportedSequence);
  atomic.role = ConSanMoiSampledSyncRole::Acquire;
  atomic.epoch_after = 10;
  EXPECT_EQ(classify_consan_moi_sampled_sync_metadata(atomic),
            ConSanMoiSampledSyncClassification::UnsupportedSequence);

  ConSanMoiSampledSyncMetadata barrier{
      .kind = ConSanMoiSampledSyncKind::Barrier,
      .role = ConSanMoiSampledSyncRole::AcquireRelease,
      .scope = ConSanMoiSampledSyncScope::Workgroup,
      .epoch_before = 10,
      .epoch_after = 11,
  };
  barrier.address = 0x1000;
  barrier.byte_count = 4;
  EXPECT_EQ(classify_consan_moi_sampled_sync_metadata(barrier),
            ConSanMoiSampledSyncClassification::InvalidRange);
  barrier.address = 0;
  barrier.byte_count = 0;
  barrier.role = ConSanMoiSampledSyncRole::Release;
  EXPECT_EQ(classify_consan_moi_sampled_sync_metadata(barrier),
            ConSanMoiSampledSyncClassification::UnsupportedSequence);
  barrier.role = ConSanMoiSampledSyncRole::AcquireRelease;
  barrier.scope = ConSanMoiSampledSyncScope::Agent;
  EXPECT_EQ(classify_consan_moi_sampled_sync_metadata(barrier),
            ConSanMoiSampledSyncClassification::UnsupportedSequence);
  barrier.scope = ConSanMoiSampledSyncScope::Workgroup;
  barrier.epoch_after = 12;
  EXPECT_EQ(classify_consan_moi_sampled_sync_metadata(barrier),
            ConSanMoiSampledSyncClassification::UnsupportedSequence);
  barrier.epoch_before = std::numeric_limits<uint32_t>::max();
  barrier.epoch_after = 0;
  EXPECT_EQ(classify_consan_moi_sampled_sync_metadata(barrier),
            ConSanMoiSampledSyncClassification::EpochOverflow);
}

TEST(ConSanMoi, SampledSyncMetadataDecodeFailsClosedOnEmptyAndMalformedWords) {
  EXPECT_EQ(decode_consan_moi_sampled_sync_metadata({}).classification,
            ConSanMoiSampledSyncClassification::Empty);
  ConSanMoiSampledSyncMetadataPacked publishing{};
  publishing.descriptor = kConSanMoiSampledSyncPublishingDescriptor;
  EXPECT_EQ(decode_consan_moi_sampled_sync_metadata(publishing).classification,
            ConSanMoiSampledSyncClassification::Publishing);
  ConSanMoiSampledSyncMetadataPacked malformed{};
  malformed.descriptor = consan_moi_sampled_sync_abi::reserved_mask;
  EXPECT_EQ(decode_consan_moi_sampled_sync_metadata(malformed).classification,
            ConSanMoiSampledSyncClassification::Malformed);
  malformed = {};
  malformed.descriptor = 2u << consan_moi_sampled_sync_abi::version_shift;
  EXPECT_EQ(decode_consan_moi_sampled_sync_metadata(malformed).classification,
            ConSanMoiSampledSyncClassification::UnsupportedVersion);
  malformed = {};
  malformed.descriptor =
      consan_moi_sampled_sync_abi::version | (15u << consan_moi_sampled_sync_abi::kind_shift);
  EXPECT_EQ(decode_consan_moi_sampled_sync_metadata(malformed).classification,
            ConSanMoiSampledSyncClassification::UnsupportedKind);
}

TEST(ConSanMoi, SampledPendingAcquireRequiresStableExactIdentityAndPastEpoch) {
  const ConSanMoiSampledPendingAcquireKey key{
      .generation = 11,
      .dispatch_id = 13,
      .workgroup_x = 2,
      .workgroup_y = 3,
      .workgroup_z = 5,
      .owner_id = 7,
      .source_epoch = 9,
      .selected_slot = 4,
  };
  const auto metadata = encode_consan_moi_sampled_sync_metadata({
      .address = 0x123456780000ull,
      .byte_count = 4,
      .kind = ConSanMoiSampledSyncKind::Atomic,
      .role = ConSanMoiSampledSyncRole::RmwAcquire,
      .scope = ConSanMoiSampledSyncScope::Agent,
      .outcome = ConSanMoiSampledSyncOutcome::RmwReturnsOld,
      .epoch_before = key.source_epoch,
      .epoch_after = key.source_epoch,
  });
  ASSERT_EQ(metadata.classification, ConSanMoiSampledSyncClassification::Valid);
  ConSanMoiSampledPendingAcquireSlot slot{
      .version = 2,
      .selected_slot = key.selected_slot,
      .generation = key.generation,
      .dispatch_id = key.dispatch_id,
      .workgroup_x = key.workgroup_x,
      .workgroup_y = key.workgroup_y,
      .workgroup_z = key.workgroup_z,
      .owner_id = key.owner_id,
      .source_epoch = key.source_epoch,
      .reserved = 0,
      .metadata = metadata.packed,
  };
  const auto classify = [&](const ConSanMoiSampledPendingAcquireSlot &value, uint32_t before = 2,
                            uint32_t after = 2, uint32_t window_epoch = 10) {
    return classify_consan_moi_sampled_pending_acquire({before, value, after}, key, window_epoch);
  };
  EXPECT_EQ(classify(slot), ConSanMoiSampledPendingAcquireState::Ready);
  EXPECT_EQ(classify(slot, 2, 4), ConSanMoiSampledPendingAcquireState::ChangedDuringRead);
  EXPECT_EQ(classify(slot, 1, 1), ConSanMoiSampledPendingAcquireState::Publishing);
  EXPECT_EQ(classify(slot, 2, 2, 8), ConSanMoiSampledPendingAcquireState::FutureEpoch);

  const auto reject_identity = [&](auto mutate) {
    auto changed = slot;
    mutate(changed);
    EXPECT_EQ(classify(changed), ConSanMoiSampledPendingAcquireState::IdentityMismatch);
  };
  reject_identity([](auto &value) { ++value.selected_slot; });
  reject_identity([](auto &value) { ++value.generation; });
  reject_identity([](auto &value) { ++value.dispatch_id; });
  reject_identity([](auto &value) { ++value.workgroup_x; });
  reject_identity([](auto &value) { ++value.workgroup_y; });
  reject_identity([](auto &value) { ++value.workgroup_z; });
  reject_identity([](auto &value) { ++value.owner_id; });
  reject_identity([](auto &value) { ++value.source_epoch; });

  auto malformed = slot;
  malformed.reserved = 1;
  EXPECT_EQ(classify(malformed), ConSanMoiSampledPendingAcquireState::Malformed);
  malformed = slot;
  malformed.metadata.descriptor = consan_moi_sampled_sync_abi::reserved_mask;
  EXPECT_EQ(classify(malformed), ConSanMoiSampledPendingAcquireState::Malformed);
  malformed = slot;
  malformed.metadata = encode_consan_moi_sampled_sync_metadata(
                           {
                               .address = 0x123456780000ull,
                               .byte_count = 4,
                               .kind = ConSanMoiSampledSyncKind::Atomic,
                               .role = ConSanMoiSampledSyncRole::RmwRelease,
                               .scope = ConSanMoiSampledSyncScope::Agent,
                               .outcome = ConSanMoiSampledSyncOutcome::RmwNoReturn,
                               .epoch_before = key.source_epoch,
                               .epoch_after = key.source_epoch,
                           })
                           .packed;
  EXPECT_EQ(classify(malformed), ConSanMoiSampledPendingAcquireState::Malformed);
}

TEST(ConSanMoi, DeferredSampledAcquireJoinsOnlyItsExactWaveWindow) {
  using State = ConSanMoiSampledPendingAcquireState;
  constexpr uint64_t generation = 17;
  constexpr uint64_t dispatch = 0x1122334455667788ull;
  constexpr uint32_t slot_index = 3;
  constexpr uint32_t owner = 5;
  constexpr uint32_t source_epoch = 2;
  constexpr uint32_t window_epoch = 4;
  const auto encoded = encode_consan_moi_sampled_sync_metadata({
      .address = 0x123456780000ull,
      .byte_count = 4,
      .kind = ConSanMoiSampledSyncKind::Atomic,
      .role = ConSanMoiSampledSyncRole::RmwAcquire,
      .scope = ConSanMoiSampledSyncScope::Agent,
      .outcome = ConSanMoiSampledSyncOutcome::RmwReturnsOld,
      .epoch_before = source_epoch,
      .epoch_after = source_epoch,
  });
  ASSERT_EQ(encoded.classification, ConSanMoiSampledSyncClassification::Valid);
  ConSanMoiSampledPendingAcquireSlot slot{
      .version = 2,
      .selected_slot = slot_index,
      .generation = generation,
      .dispatch_id = dispatch,
      .workgroup_x = 7,
      .workgroup_y = 8,
      .workgroup_z = 9,
      .owner_id = owner,
      .source_epoch = source_epoch,
      .metadata = encoded.packed,
  };
  ConSanMoiSampledPendingAcquireView pending{2, slot, 2};
  const ConSanMoiSampledCausalWindow window{
      .generation = generation,
      .dispatch_id = dispatch,
      .workgroup_x = 7,
      .workgroup_y = 8,
      .workgroup_z = 9,
      .epoch = window_epoch,
      .first_entry = slot_index,
      .entry_count = 1,
      .publication_state = static_cast<uint32_t>(ConSanMoiSampledCausalPublicationState::Ready),
  };
  const auto watchpoint = [&](uint32_t wave_owner) {
    return pack_consan_moi_sampled_watchpoint_entry(ConSanMoiShadowAccessKind::Read, wave_owner,
                                                    window_epoch, static_cast<uint32_t>(generation),
                                                    12, 1);
  };

  const auto joined =
      consan_moi_sampled_join_pending_acquire(pending, window, watchpoint(owner), slot_index);
  ASSERT_EQ(joined.state, State::Ready);
  ASSERT_EQ(joined.sync.classification, ConSanMoiSampledSyncClassification::Valid);
  EXPECT_EQ(joined.sync.metadata.address, encoded.packed.address);
  EXPECT_EQ(joined.sync.metadata.epoch_before, window_epoch);
  EXPECT_EQ(joined.sync.metadata.epoch_after, window_epoch);
  EXPECT_EQ(
      consan_moi_sampled_join_pending_acquire(pending, window, watchpoint(owner + 1u), slot_index)
          .state,
      State::IdentityMismatch);
  pending.slot.reserved = 1;
  EXPECT_EQ(
      consan_moi_sampled_join_pending_acquire(pending, window, watchpoint(owner), slot_index).state,
      State::Malformed);
  pending.slot.reserved = 0;
  pending.version_after = 4;
  EXPECT_EQ(
      consan_moi_sampled_join_pending_acquire(pending, window, watchpoint(owner), slot_index).state,
      State::ChangedDuringRead);
}

TEST(ConSanMoi, SampledSyncSnapshotRejectsConcurrentHybridAndWrongPairedEpoch) {
  const auto encoded = encode_consan_moi_sampled_sync_metadata({
      .address = 0x123456780000ull,
      .byte_count = 4,
      .kind = ConSanMoiSampledSyncKind::Atomic,
      .role = ConSanMoiSampledSyncRole::RmwRelease,
      .scope = ConSanMoiSampledSyncScope::Agent,
      .outcome = ConSanMoiSampledSyncOutcome::RmwNoReturn,
      .epoch_before = 7,
      .epoch_after = 7,
  });
  ASSERT_EQ(encoded.classification, ConSanMoiSampledSyncClassification::Valid);
  EXPECT_EQ(
      classify_consan_moi_sampled_sync_snapshot({0u, encoded.packed, encoded.packed.descriptor}, 7)
          .classification,
      ConSanMoiSampledSyncClassification::ChangedDuringRead);
  auto publishing = encoded.packed;
  publishing.descriptor = kConSanMoiSampledSyncPublishingDescriptor;
  EXPECT_EQ(classify_consan_moi_sampled_sync_snapshot({kConSanMoiSampledSyncPublishingDescriptor,
                                                       publishing,
                                                       kConSanMoiSampledSyncPublishingDescriptor},
                                                      7)
                .classification,
            ConSanMoiSampledSyncClassification::Publishing);
  EXPECT_EQ(classify_consan_moi_sampled_sync_snapshot(
                {encoded.packed.descriptor, encoded.packed, encoded.packed.descriptor}, 8)
                .classification,
            ConSanMoiSampledSyncClassification::UnsupportedSequence);
  EXPECT_EQ(classify_consan_moi_sampled_sync_snapshot(
                {encoded.packed.descriptor, encoded.packed, encoded.packed.descriptor}, 7)
                .classification,
            ConSanMoiSampledSyncClassification::Valid);
}

TEST(ConSanMoi, SampledSyncMetadataPublicationIsBoundedAndCollisionSafe) {
  const ConSanMoiSampledSyncMetadata first{
      .address = 0x1000,
      .byte_count = 4,
      .kind = ConSanMoiSampledSyncKind::Atomic,
      .role = ConSanMoiSampledSyncRole::Release,
      .scope = ConSanMoiSampledSyncScope::Agent,
      .epoch_before = 3,
      .epoch_after = 3,
  };
  auto second = first;
  second.address = 0x2000;
  std::array<ConSanMoiSampledSyncMetadataPacked, 2> slots{};

  auto result = consan_moi_sampled_publish_sync_metadata(slots, 0, first);
  EXPECT_EQ(result.outcome, ConSanMoiSampledSyncPublishOutcome::Published);
  EXPECT_EQ(result.classification, ConSanMoiSampledSyncClassification::Valid);
  const auto retained = slots[0];
  result = consan_moi_sampled_publish_sync_metadata(slots, 0, first);
  EXPECT_EQ(result.outcome, ConSanMoiSampledSyncPublishOutcome::Existing);
  result = consan_moi_sampled_publish_sync_metadata(slots, 0, second);
  EXPECT_EQ(result.outcome, ConSanMoiSampledSyncPublishOutcome::Collision);
  EXPECT_EQ(slots[0], retained);
  EXPECT_EQ(slots[1], ConSanMoiSampledSyncMetadataPacked{});

  result = consan_moi_sampled_publish_sync_metadata(slots, 2, second);
  EXPECT_EQ(result.outcome, ConSanMoiSampledSyncPublishOutcome::CapacityExhausted);
  result = consan_moi_sampled_publish_sync_metadata({}, 0, second);
  EXPECT_EQ(result.outcome, ConSanMoiSampledSyncPublishOutcome::CapacityExhausted);

  slots[1].descriptor = consan_moi_sampled_sync_abi::reserved_mask;
  result = consan_moi_sampled_publish_sync_metadata(slots, 1, second);
  EXPECT_EQ(result.outcome, ConSanMoiSampledSyncPublishOutcome::MalformedSlot);
  EXPECT_EQ(result.classification, ConSanMoiSampledSyncClassification::Malformed);
  EXPECT_EQ(slots[1].descriptor, consan_moi_sampled_sync_abi::reserved_mask);

  auto invalid = second;
  invalid.byte_count = 0;
  result = consan_moi_sampled_publish_sync_metadata(slots, 1, invalid);
  EXPECT_EQ(result.outcome, ConSanMoiSampledSyncPublishOutcome::Rejected);
  EXPECT_EQ(result.classification, ConSanMoiSampledSyncClassification::InvalidRange);
  EXPECT_EQ(slots[1].descriptor, consan_moi_sampled_sync_abi::reserved_mask);
}

TEST(ConSanMoi, SampledBarrierQualificationAcceptsCompleteStaticOwnedSequence) {
  ConSanSyncSequence sequence;
  sequence.kind = ConSanSyncSequenceKind::Barrier;
  sequence.operation = ConSanSyncOperation::BarrierFull;
  sequence.memory_role = ConSanSyncMemoryRole::AcquireRelease;
  sequence.confidence = ConSanSemanticConfidence::Conservative;
  sequence.memory_role_confidence = ConSanSemanticConfidence::Conservative;
  sequence.in_kernel = true;
  sequence.basic_block_index = 0;
  sequence.begin_text_offset = 8;
  sequence.end_text_offset = 16;
  sequence.member_event_identities = {"signal", "wait"};
  sequence.barrier_id = -1;
  sequence.barrier_operand_source = ConSanBarrierSite::OperandSource::Immediate;
  sequence.barrier_scope = ConSanBarrierSite::Scope::Workgroup;
  sequence.execution_owners.push_back({.descriptor_file_offset = 0x80});
  EXPECT_TRUE(consan_moi_sampled_qualifies_barrier_sequence(sequence));

  const auto rejects = [&](auto mutate) {
    ConSanSyncSequence candidate = sequence;
    mutate(candidate);
    EXPECT_FALSE(consan_moi_sampled_qualifies_barrier_sequence(candidate));
  };
  rejects([](auto &item) { item.operation = ConSanSyncOperation::BarrierSignal; });
  rejects([](auto &item) { item.operation = ConSanSyncOperation::BarrierWait; });
  rejects([](auto &item) {
    item.barrier_operand_source = ConSanBarrierSite::OperandSource::DynamicM0;
  });
  rejects([](auto &item) { item.barrier_id.reset(); });
  rejects([](auto &item) { item.barrier_scope = ConSanBarrierSite::Scope::Unknown; });
  rejects([](auto &item) { item.confidence = ConSanSemanticConfidence::Ambiguous; });
  sequence.in_cyclic_cfg_component = true;
  EXPECT_TRUE(consan_moi_sampled_qualifies_barrier_sequence(sequence));
  sequence.in_cyclic_cfg_component = false;
  sequence.barrier_scope = ConSanBarrierSite::Scope::Cluster;
  EXPECT_TRUE(consan_moi_sampled_qualifies_barrier_sequence(sequence));
  sequence.barrier_scope = ConSanBarrierSite::Scope::Workgroup;
  rejects([](auto &item) { item.inside_scalar_clause = true; });
  rejects([](auto &item) { item.execution_owners.clear(); });
  ConSanSyncSequence callable = sequence;
  callable.in_kernel = false;
  callable.execution_owners.push_back({.descriptor_file_offset = 0x90});
  EXPECT_TRUE(consan_moi_sampled_qualifies_barrier_sequence(callable));
  rejects([](auto &item) { item.member_event_identities.push_back("ambiguous-third-member"); });
}

TEST(ConSanMoi, SampledBarrierSnapshotMustStartAtPairedSelectedEpoch) {
  const auto encoded = encode_consan_moi_sampled_sync_metadata({
      .kind = ConSanMoiSampledSyncKind::Barrier,
      .role = ConSanMoiSampledSyncRole::AcquireRelease,
      .scope = ConSanMoiSampledSyncScope::Workgroup,
      .epoch_before = 7,
      .epoch_after = 8,
  });
  ASSERT_EQ(encoded.classification, ConSanMoiSampledSyncClassification::Valid);
  EXPECT_EQ(classify_consan_moi_sampled_sync_snapshot(
                {encoded.packed.descriptor, encoded.packed, encoded.packed.descriptor}, 7)
                .classification,
            ConSanMoiSampledSyncClassification::Valid);
  EXPECT_EQ(classify_consan_moi_sampled_sync_snapshot(
                {encoded.packed.descriptor, encoded.packed, encoded.packed.descriptor}, 6)
                .classification,
            ConSanMoiSampledSyncClassification::UnsupportedSequence);
}

TEST(ConSanMoi, SampledClusterBarrierMetadataRoundTrips) {
  const auto encoded = encode_consan_moi_sampled_sync_metadata({
      .kind = ConSanMoiSampledSyncKind::Barrier,
      .role = ConSanMoiSampledSyncRole::AcquireRelease,
      .scope = ConSanMoiSampledSyncScope::Cluster,
      .epoch_before = 7,
      .epoch_after = 8,
  });
  ASSERT_EQ(encoded.classification, ConSanMoiSampledSyncClassification::Valid);
  const auto decoded = decode_consan_moi_sampled_sync_metadata(encoded.packed);
  EXPECT_EQ(decoded.classification, ConSanMoiSampledSyncClassification::Valid);
  EXPECT_EQ(decoded.metadata.scope, ConSanMoiSampledSyncScope::Cluster);
}

TEST(ConSanMoi, SampledQualifiedBarrierPublishesSelectedEpochTransition) {
  std::vector<uint32_t> words(540, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  words[0] = 0xD8340000u;
  words[1] = 0x00000000u; // ds_store_b32 v0, v0
  constexpr size_t kSignal = 400;
  constexpr size_t kWait = 401;
  words[kSignal] = 0xBE804EC1u; // s_barrier_signal -1
  words[kWait] = 0xBF94FFFFu;   // s_barrier_wait -1
  words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(words, "sampled_barrier");
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.moi_track_barriers = true;
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 80;
  options.moi_owner_vgpr = 20;
  options.moi_epoch_vgpr = 21;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(2);
  options.max_patches = 3;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord,
                               &ConSanPatchInfo::kind),
            0u);
  const auto patch = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiSampledSyncMetadata, &ConSanPatchInfo::kind);
  ASSERT_NE(patch, result.patches.end()) << testing::PrintToString(result.warnings);
  EXPECT_EQ(patch->anchor_offset, kWait * sizeof(uint32_t));
  EXPECT_EQ(patch->covered_sync_event_count, 2u);
  EXPECT_EQ(result.sampled_barrier_applicable_event_count, 2u);
  std::vector<ConSanSiteDispositionRecord> barrier_dispositions;
  std::ranges::copy_if(result.site_dispositions, std::back_inserter(barrier_dispositions),
                       [](const ConSanSiteDispositionRecord &site) {
                         return site.site_kind == ConSanResourceSiteKind::Barrier;
                       });
  ASSERT_EQ(barrier_dispositions.size(), 2u);
  EXPECT_TRUE(std::ranges::all_of(barrier_dispositions, [](const auto &site) {
    return site.disposition == ConSanSiteDisposition::Supported &&
           site.reason == ConSanSiteDispositionReason::None &&
           site.lowering_outcome == ConSanSiteLoweringOutcome::Patched &&
           site.lowering_reason == ConSanSiteLoweringReason::None;
  }));
  EXPECT_EQ(barrier_dispositions[0].text_offset, kSignal * sizeof(uint32_t));
  EXPECT_EQ(barrier_dispositions[1].text_offset, kWait * sizeof(uint32_t));
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> trampoline =
      text_words_at_offset(patched, patch->trampoline_offset, patch->trampoline_size);
  EXPECT_NE(std::ranges::find(trampoline, words[kWait]), trampoline.end());
  const auto advance =
      build_v_add_nc_u32_e32(*options.moi_epoch_vgpr, scalar_positive_inline_u32(1),
                             *options.moi_epoch_vgpr, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(advance);
  EXPECT_NE(std::ranges::find(trampoline, *advance), trampoline.end());
  const auto descriptor = encode_consan_moi_sampled_sync_metadata({
      .kind = ConSanMoiSampledSyncKind::Barrier,
      .role = ConSanMoiSampledSyncRole::AcquireRelease,
      .scope = ConSanMoiSampledSyncScope::Workgroup,
      .epoch_before = 0,
      .epoch_after = 1,
  });
  ASSERT_EQ(descriptor.classification, ConSanMoiSampledSyncClassification::Valid);
  const auto descriptor_literal = build_v_mov_b32_e64_literal(
      /*vdst=*/10, descriptor.packed.descriptor, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(descriptor_literal);
  EXPECT_TRUE(contains_subsequence(trampoline, *descriptor_literal));

  const auto select_owner_zero = build_v_cmp_eq_u32_e32_vcc(
      scalar_positive_inline_u32(0), *options.moi_owner_vgpr, ROCJITSU_CODE_ARCH_RDNA4);
  const auto narrow_owner_wave =
      build_s_and_saveexec_b64(*options.moi_exec_save_sgpr, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(select_owner_zero);
  ASSERT_TRUE(narrow_owner_wave);
  EXPECT_TRUE(contains_subsequence(
      trampoline, std::array<uint32_t, 2>{*select_owner_zero, *narrow_owner_wave}));

  const auto save_scc = build_rdna4_s_cselect_b32(
      /*sdst=*/84, scalar_positive_inline_u32(1), scalar_positive_inline_u32(0),
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto save_vcc = build_s_mov_b64(/*sdst=*/82, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto save_exec = build_s_mov_b64(/*sdst=*/86, kRdna4ExecLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_exec = build_s_mov_b64(kRdna4ExecLo, /*ssrc0=*/86, ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_vcc = build_s_mov_b64(kRdna4VccLo, /*ssrc0=*/82, ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_scc = build_rdna4_s_cmp_lg_u32(
      /*ssrc0=*/84, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(save_scc);
  ASSERT_TRUE(save_vcc);
  ASSERT_TRUE(save_exec);
  ASSERT_TRUE(restore_exec);
  ASSERT_TRUE(restore_vcc);
  ASSERT_TRUE(restore_scc);
  EXPECT_TRUE(
      contains_subsequence(trampoline, std::array<uint32_t, 3>{*save_scc, *save_vcc, *save_exec}));
  EXPECT_TRUE(contains_subsequence(
      trampoline, std::array<uint32_t, 3>{*restore_exec, *restore_vcc, *restore_scc}));
}

TEST(ConSanMoi, SampledStraightLineSeparatedBarriersPublishTwoMemberSequences) {
  std::vector<uint32_t> words(540, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  words[0] = 0xD8340000u;
  words[1] = 0x00000000u; // ds_store_b32 v0, v0
  constexpr size_t kSignal = 390;
  // Thirty-one intervening instructions exercise the extended structural
  // association path rather than the ordinary four-instruction path.
  constexpr size_t kWait = 422;
  words[kSignal] = 0xBE804EC1u; // s_barrier_signal -1
  words[kWait] = 0xBF94FFFFu;   // s_barrier_wait -1
  words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.moi_track_barriers = true;
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 80;
  options.moi_owner_vgpr = 20;
  options.moi_epoch_vgpr = 21;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(2);
  options.max_patches = 3;

  const ConSanResult result =
      try_patch_consan(make_rdna4_lds_code_object(words, "sampled_separated_barrier"), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  const auto patch = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiSampledSyncMetadata, &ConSanPatchInfo::kind);
  ASSERT_NE(patch, result.patches.end()) << testing::PrintToString(result.warnings);
  EXPECT_EQ(patch->anchor_offset, kWait * sizeof(uint32_t));
  EXPECT_EQ(patch->covered_sync_event_count, 2u);
  EXPECT_EQ(result.sampled_barrier_applicable_event_count, 2u);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiSampledSyncMetadata,
                               &ConSanPatchInfo::kind),
            1u);

  std::vector<uint32_t> beyond = words;
  beyond[kWait] = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
  constexpr size_t kBeyondWait = 424;
  beyond[kBeyondWait] = 0xBF94FFFFu;
  const ConSanResult farther =
      try_patch_consan(make_rdna4_lds_code_object(beyond, "sampled_overlong_barrier"), options);
  ASSERT_TRUE(consan_patch_succeeded(farther));
  const auto farther_patch = std::ranges::find(
      farther.patches, ConSanPatchKind::TrampolineMoiSampledSyncMetadata, &ConSanPatchInfo::kind);
  ASSERT_NE(farther_patch, farther.patches.end()) << testing::PrintToString(farther.warnings);
  EXPECT_EQ(farther_patch->anchor_offset, kBeyondWait * sizeof(uint32_t));
  EXPECT_EQ(farther_patch->covered_sync_event_count, 2u);
}

TEST(ConSanMoi, SampledIncompleteAndDynamicBarriersCannotAdvanceEpoch) {
  for (const std::vector<uint32_t> &barrier_words :
       {std::vector<uint32_t>{0xBE804EC1u},                 // unmatched signal
        std::vector<uint32_t>{0xBF94FFFFu},                 // unmatched wait
        std::vector<uint32_t>{0xBE804E7Du, 0xBF940001u},    // dynamic signal, static wait
        std::vector<uint32_t>{0xBE804EC1u, 0xBF940001u}}) { // mismatched IDs
    std::vector<uint32_t> words(540, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
    words[0] = 0xD8340000u;
    words[1] = 0x00000000u;
    std::ranges::copy(barrier_words, words.begin() + 400);
    words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);
    ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
    options.moi_track_barriers = true;
    options.scratch_vgpr = 8;
    options.moi_exec_save_sgpr = 80;
    options.moi_owner_vgpr = 20;
    options.moi_epoch_vgpr = 21;
    options.moi_report_buffer_address = 0x123456780000ull;
    options.moi_report_buffer_size = direct_sampled_report_bytes(2);
    options.max_patches = 3;
    const ConSanResult result =
        try_patch_consan(make_rdna4_lds_code_object(words, "rejected_barrier"), options);
    ASSERT_TRUE(consan_patch_succeeded(result));
    EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiSampledSyncMetadata,
                                 &ConSanPatchInfo::kind),
              0u)
        << testing::PrintToString(result.warnings);
    EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord,
                                 &ConSanPatchInfo::kind),
              0u);
    std::vector<ConSanSiteDispositionRecord> barrier_dispositions;
    std::ranges::copy_if(result.site_dispositions, std::back_inserter(barrier_dispositions),
                         [](const ConSanSiteDispositionRecord &site) {
                           return site.site_kind == ConSanResourceSiteKind::Barrier;
                         });
    ASSERT_EQ(barrier_dispositions.size(), barrier_words.size());
    EXPECT_TRUE(std::ranges::all_of(barrier_dispositions, [](const auto &site) {
      return site.disposition == ConSanSiteDisposition::Unsupported &&
             site.reason == ConSanSiteDispositionReason::UnqualifiedSyncSequence;
    }));
  }
}

TEST(ConSanMoi, SampledBarrierWithoutPrecedingSelectedWindowIsTypedNotApplicable) {
  std::vector<uint32_t> words(540, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  words[100] = 0xBE804EC1u;
  words[101] = 0xBF94FFFFu;
  words[200] = 0xD8340000u;
  words[201] = 0x00000000u;
  words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.moi_track_barriers = true;
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 80;
  options.moi_owner_vgpr = 20;
  options.moi_epoch_vgpr = 21;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(2);
  options.max_patches = 3;

  const ConSanResult result =
      try_patch_consan(make_rdna4_lds_code_object(words, "sampled_barrier_before_window"), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiSampledSyncMetadata,
                               &ConSanPatchInfo::kind),
            0u);
  std::vector<ConSanSiteDispositionRecord> barriers;
  std::ranges::copy_if(result.site_dispositions, std::back_inserter(barriers),
                       [](const ConSanSiteDispositionRecord &site) {
                         return site.site_kind == ConSanResourceSiteKind::Barrier;
                       });
  ASSERT_EQ(barriers.size(), 2u);
  EXPECT_TRUE(std::ranges::all_of(barriers, [](const auto &site) {
    return site.disposition == ConSanSiteDisposition::NotApplicable &&
           site.reason == ConSanSiteDispositionReason::NoPrecedingSampledWindow;
  }));
}

TEST(ConSanMoi, SampledRejectedBarriersStillPreserveAccessOwnerAtEntry) {
  std::vector<uint32_t> words(540, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  words[0] = 0xD8340000u;
  words[1] = 0x00000000u;   // ds_store_b32 v0, v0
  words[400] = 0xBE804EC1u; // unmatched s_barrier_signal
  words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.moi_init_owner_epoch = true;
  options.moi_track_barriers = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(2);
  options.max_patches = 3;

  const ConSanResult result =
      try_patch_consan(make_rdna4_lds_code_object(words, "rejected_barrier_auto_state"), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  EXPECT_TRUE(result.resolved_moi_owner_vgpr);
  EXPECT_TRUE(result.resolved_moi_epoch_vgpr);
  EXPECT_TRUE(result.moi_persistent_vgprs_automatic);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue,
                               &ConSanPatchInfo::kind),
            1u);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiSampledSyncMetadata,
                               &ConSanPatchInfo::kind),
            0u);
  EXPECT_TRUE(std::ranges::any_of(result.site_dispositions, [](const auto &site) {
    return site.site_kind == ConSanResourceSiteKind::Barrier &&
           site.disposition == ConSanSiteDisposition::Unsupported &&
           site.reason == ConSanSiteDispositionReason::UnqualifiedSyncSequence;
  }));
}

TEST(ConSanMoi, SampledQualifiedBarrierPersistsOwnerAcrossAccessAndSync) {
  std::vector<uint32_t> words(540, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  words[0] = 0xD8340000u;
  words[1] = 0x00000000u;   // ds_store_b32 v0, v0
  words[400] = 0xBE804EC1u; // s_barrier_signal -1
  words[401] = 0xBF94FFFFu; // s_barrier_wait -1
  words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.moi_track_barriers = true;
  options.moi_runtime_sample_stride = 64;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(2);
  options.max_patches = 3;

  const ConSanResult result = try_patch_consan(
      make_rdna4_lds_code_object(words, "sampled_barrier_persistent_owner"), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.moi_persistent_vgprs_automatic);
  EXPECT_FALSE(result.moi_private_epoch_automatic);
  ASSERT_TRUE(result.resolved_moi_owner_vgpr);
  ASSERT_TRUE(result.resolved_moi_epoch_vgpr);
  EXPECT_NE(std::ranges::find(result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue,
                              &ConSanPatchInfo::kind),
            result.patches.end());
  EXPECT_NE(std::ranges::find(result.patches, ConSanPatchKind::TrampolineMoiSampledSyncMetadata,
                              &ConSanPatchInfo::kind),
            result.patches.end());
}

TEST(ConSanMoi, SampledFarBarrierUsesReachableLocalIndirectEntryIsland) {
  std::vector<uint32_t> kernel_words;
  for (uint32_t i = 0; i < 9u; ++i) {
    kernel_words.push_back(0xD8340000u); // ds_store_b32 v0, v0
    kernel_words.push_back(0x00000000u);
  }
  const uint64_t barrier_offset = kernel_words.size() * sizeof(uint32_t);
  kernel_words.push_back(0xBE804EC1u); // s_barrier_signal -1
  kernel_words.push_back(0xBF94FFFFu); // s_barrier_wait -1
  kernel_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4));
  const std::array function_words = {build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4)};
  std::vector<uint32_t> far_tail(40000u, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words, far_tail);
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.moi_track_barriers = true;
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 80;
  options.moi_dispatch_id_sgpr = 70;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(18);
  options.moi_runtime_sample_stride = 16384;
  options.max_patches = 20;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  EXPECT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto sync_patch = std::ranges::find_if(result.patches, [&](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiSampledSyncMetadata &&
           patch.anchor_offset == barrier_offset + sizeof(uint32_t);
  });
  ASSERT_NE(sync_patch, result.patches.end()) << testing::PrintToString(result.warnings);
  const auto island = std::ranges::find_if(result.patches, [&](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiIndirectBranchIsland &&
           patch.anchor_offset == sync_patch->anchor_offset;
  });
  ASSERT_NE(island, result.patches.end());
  EXPECT_TRUE(compute_sopp_branch_simm16(sync_patch->anchor_offset, island->trampoline_offset));
  EXPECT_FALSE(
      compute_sopp_branch_simm16(sync_patch->anchor_offset, sync_patch->trampoline_offset));
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, Rdna4DenseSampledAccessesShareExplicitKeyRelay) {
  constexpr uint32_t kAccessCount = 9u;
  std::vector<uint32_t> text_words(
      8u, build_s_mov_b32(/*sdst=*/0, /*ssrc0=*/0, ROCJITSU_CODE_ARCH_RDNA4));
  for (uint32_t index = 0; index < kAccessCount; ++index) {
    text_words.push_back(0xD8340000u | index * sizeof(uint32_t));
    text_words.push_back(0x00000000u); // ds_store_b32 v0, v0 offset:index*4
  }
  // Keep the sites near the entry and the per-site appended relays outside
  // SOPP reach. The shared explicit-key router must retain every dense site
  // instead of relocating neighboring guest accesses without probes.
  text_words.resize(33'000u, build_s_mov_b32(/*sdst=*/0, /*ssrc0=*/0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4));

  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 80;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(kAccessCount);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.max_patches = kAccessCount;

  const ConSanResult result =
      try_patch_consan(make_rdna4_lds_code_object(text_words, "rdna4_dense_sampled"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiSampledWatchpointStore,
                               &ConSanPatchInfo::kind),
            kAccessCount);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiIndirectBranchIsland,
                               &ConSanPatchInfo::kind),
            2u); // One local relay plus one appended explicit-key dispatcher.
  EXPECT_TRUE(std::ranges::none_of(result.warnings, [](const std::string &warning) {
    return warning.find("inside a relocated prefix") != std::string::npos;
  }));
}

TEST(ConSanMoi, SampledSharedAccessRelayPreservesEntryIslandForFarBarrier) {
  std::vector<uint32_t> kernel_words;
  for (uint32_t i = 0; i < 9u; ++i) {
    kernel_words.push_back(0xD8340000u); // ds_store_b32 v0, v0
    kernel_words.push_back(0x00000000u);
  }
  const uint64_t barrier_offset = kernel_words.size() * sizeof(uint32_t);
  kernel_words.push_back(0xBE804EC1u); // s_barrier_signal -1
  kernel_words.push_back(0xBF94FFFFu); // s_barrier_wait -1
  kernel_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4));
  const std::array function_words = {build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4)};
  std::vector<uint32_t> far_tail(40000u, build_s_nop(1, ROCJITSU_CODE_ARCH_RDNA4));
  // There are only eight local islands for nine accesses plus the barrier.
  // Dense access routing must share one relay so the qualified barrier still
  // receives an island instead of becoming a placement gap.
  std::fill_n(far_tail.begin(), 8u * 8u, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words, far_tail);
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.moi_track_barriers = true;
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 80;
  options.moi_dispatch_id_sgpr = 70;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(18);
  options.moi_runtime_sample_stride = 16384;
  options.max_patches = 20;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  EXPECT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiSampledSyncMetadata,
                               &ConSanPatchInfo::kind),
            1u);
  EXPECT_TRUE(std::ranges::none_of(result.warnings, [](const std::string &warning) {
    return warning.find("no reachable entry island") != std::string::npos;
  })) << testing::PrintToString(result.warnings);
  const auto barrier = std::ranges::find_if(result.site_dispositions, [&](const auto &site) {
    return site.site_kind == ConSanResourceSiteKind::Barrier &&
           site.text_offset == barrier_offset + sizeof(uint32_t);
  });
  ASSERT_NE(barrier, result.site_dispositions.end());
  EXPECT_EQ(barrier->lowering_outcome, ConSanSiteLoweringOutcome::Patched);
  EXPECT_EQ(barrier->lowering_reason, ConSanSiteLoweringReason::None);
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, Rdna4SampledDenseBarrierRelayKeepsManyFarPairsReachable) {
  constexpr uint32_t kBarrierCount = 10;
  std::vector<uint32_t> words;
  for (uint32_t i = 0; i < kBarrierCount; ++i) {
    words.push_back(0xD8340000u); // ds_store_b32 v0, v0
    words.push_back(0x00000000u);
    words.push_back(0xBE804EC1u); // s_barrier_signal -1
    words.push_back(0xBF94FFFFu); // s_barrier_wait -1
  }
  // Keep every barrier near the entry while forcing its appended body beyond
  // SOPP reach. The dense relay must relocate one ordinary host and route all
  // ten pairs through it instead of consuming scarce local NOP islands.
  words.resize(33'000u, build_s_mov_b32(/*sdst=*/0, /*ssrc0=*/0, ROCJITSU_CODE_ARCH_RDNA4));
  words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4));

  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.moi_track_barriers = true;
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 80;
  options.moi_dispatch_id_sgpr = 70;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(2u * kBarrierCount);
  options.moi_runtime_sample_stride = 16384;
  options.max_patches = 3u * kBarrierCount;

  const ConSanResult result =
      try_patch_consan(make_rdna4_lds_code_object(words, "rdna4_dense_sampled_barriers"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  EXPECT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiSampledSyncMetadata,
                               &ConSanPatchInfo::kind),
            kBarrierCount);
  EXPECT_TRUE(std::ranges::none_of(result.warnings, [](const std::string &warning) {
    return warning.find("no reachable entry island") != std::string::npos;
  })) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, SampledQualifiedBarrierAdmitsLongStraightLinePair) {
  std::vector<uint32_t> words(540, build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250));
  constexpr auto store = gfx1250::build_vds(gfx1250::kDsStoreB32Vds, {.addr = 0, .data0 = 0});
  words[0] = store[0];
  words[1] = store[1];
  words[300] = 0xBE804EC1u; // s_barrier_signal -1
  words[380] = 0xBF94FFFFu; // s_barrier_wait -1, 320 bytes later
  words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250);
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.moi_track_barriers = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(2);
  options.max_patches = 3;

  const ConSanResult result = try_patch_consan(
      make_gfx1250_code_object(words, "sampled_long_straight_line_barrier"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  const auto patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &item) {
    return item.kind == ConSanPatchKind::TrampolineMoiSampledSyncMetadata &&
           item.anchor_offset == 380u * sizeof(uint32_t);
  });
  ASSERT_NE(patch, result.patches.end()) << testing::PrintToString(result.warnings);
  EXPECT_EQ(patch->covered_sync_event_count, 2u);
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, SampledQualifiedBarrierForcedSpillPreservesSevenVgprWindow) {
  std::vector<uint32_t> words(540, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  words[0] = 0xD8340000u;
  words[1] = 0x00000000u;
  words[400] = 0xBE804EC1u;
  words[401] = 0xBF94FFFFu;
  words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.moi_track_barriers = true;
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(2);
  options.max_patches = 3;

  const ConSanResult result =
      try_patch_consan(make_rdna4_lds_code_object(words, "sampled_barrier_spill"), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  const auto patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &item) {
    return item.kind == ConSanPatchKind::TrampolineMoiSampledSyncMetadata &&
           item.anchor_offset == 401u * sizeof(uint32_t);
  });
  ASSERT_NE(patch, result.patches.end()) << testing::PrintToString(result.warnings);
  EXPECT_EQ(patch->spilled_vgpr_count, 7u);
  EXPECT_GT(patch->required_private_segment_size, 0u);
}

TEST(ConSanMoi, Gfx1250SampledQualifiedBarrierUsesSpill) {
  std::vector<uint32_t> words(540, build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250));
  constexpr auto store = gfx1250::build_vds(gfx1250::kDsStoreB32Vds, {.addr = 0, .data0 = 0});
  words[0] = store[0];
  words[1] = store[1];
  words[400] = 0xBE804EC1u; // s_barrier_signal -1
  words[401] = 0xBF94FFFFu; // s_barrier_wait -1
  words[402] = 0xBF860001u; // next block selects a nonzero VGPR bank
  words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250);
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.moi_track_barriers = true;
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(2);
  options.max_patches = 3;

  const ConSanResult result =
      try_patch_consan(make_gfx1250_code_object(words, "gfx1250_sampled_barrier_spill"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  const auto patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &item) {
    return item.kind == ConSanPatchKind::TrampolineMoiSampledSyncMetadata &&
           item.anchor_offset == 401u * sizeof(uint32_t);
  });
  ASSERT_NE(patch, result.patches.end()) << testing::PrintToString(result.warnings);
  EXPECT_EQ(patch->covered_sync_event_count, 2u);
  EXPECT_EQ(patch->spilled_vgpr_count, 7u);
  EXPECT_GT(patch->required_private_segment_size, 0u);
  EXPECT_TRUE(result.final_validation_passed);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> trampoline =
      text_words_at_offset(patched, patch->trampoline_offset, patch->trampoline_size);
  ASSERT_FALSE(trampoline.empty());
  EXPECT_EQ(trampoline.front(), 0xBF860000u);
}

TEST(ConSanMoi, Gfx1250SampledBarrierDoesNotGateWorkgroupsForAddressSampling) {
  std::vector<uint32_t> words(540, build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250));
  constexpr auto store = gfx1250::build_vds(gfx1250::kDsStoreB32Vds, {.addr = 0, .data0 = 0});
  words[0] = store[0];
  words[1] = store[1];
  words[400] = 0xBE804EC1u; // s_barrier_signal -1
  words[401] = 0xBF94FFFFu; // s_barrier_wait -1
  words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250);
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.moi_track_barriers = true;
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 80;
  options.moi_owner_vgpr = 20;
  options.moi_epoch_vgpr = 21;
  options.moi_runtime_sample_stride = 64;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(2);
  options.max_patches = 3;

  const ConSanResult result = try_patch_consan(
      make_gfx1250_code_object(words, "gfx1250_sampled_barrier_runtime_gate"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  const auto patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &item) {
    return item.kind == ConSanPatchKind::TrampolineMoiSampledSyncMetadata &&
           item.anchor_offset == 401u * sizeof(uint32_t);
  });
  ASSERT_NE(patch, result.patches.end()) << testing::PrintToString(result.warnings);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> trampoline =
      text_words_at_offset(patched, patch->trampoline_offset, patch->trampoline_size);
  const auto obsolete_workgroup_gate_shift = build_s_lshr_b32(
      /*sdst=*/85u, /*ssrc0=*/86u, scalar_positive_inline_u32(6u), ROCJITSU_CODE_ARCH_GFX1250);
  const auto owner_election = build_v_cmp_eq_u32_e32_vcc(
      scalar_positive_inline_u32(0), *options.moi_owner_vgpr, ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(owner_election);
  const auto barrier = std::ranges::find(trampoline, words[401]);
  const auto owner = std::ranges::find(trampoline, *owner_election);
  ASSERT_NE(barrier, trampoline.end());
  ASSERT_NE(owner, trampoline.end());
  EXPECT_LT(barrier, owner);
  EXPECT_EQ(std::ranges::find(trampoline, obsolete_workgroup_gate_shift), trampoline.end());
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, Gfx1250SampledBarriersPartitionRelayWindowsAcrossLargeKernel) {
  constexpr size_t kAccessesPerWindow = 9u;
  constexpr size_t kFirstStore = 32u;
  constexpr size_t kFirstSignal = 400u;
  constexpr size_t kFirstWait = 401u;
  constexpr size_t kSecondStore = 65'580u;
  constexpr size_t kSecondSignal = 65'980u;
  constexpr size_t kSecondWait = 65'981u;
  const uint32_t filler = build_s_mov_b32(/*sdst=*/0, /*ssrc0=*/0, ROCJITSU_CODE_ARCH_GFX1250);
  std::vector<uint32_t> words(kSecondWait + 140u, filler);
  for (size_t index = 0; index < kAccessesPerWindow; ++index) {
    words[kFirstStore + 2u * index] = 0xD8340000u | static_cast<uint32_t>(index * sizeof(uint32_t));
    words[kFirstStore + 2u * index + 1u] = 0x00000000u;
  }
  words[kFirstSignal] = 0xBE804EC1u; // s_barrier_signal -1
  words[kFirstWait] = 0xBF94FFFFu;   // s_barrier_wait -1
  for (size_t index = 0; index < kAccessesPerWindow; ++index) {
    words[kSecondStore + 2u * index] =
        0xD8340000u | static_cast<uint32_t>(index * sizeof(uint32_t));
    words[kSecondStore + 2u * index + 1u] = 0x00000000u;
  }
  words[kSecondSignal] = 0xBE804EC1u; // s_barrier_signal -1
  words[kSecondWait] = 0xBF94FFFFu;   // s_barrier_wait -1
  words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250);

  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.moi_track_barriers = true;
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 80;
  options.moi_owner_vgpr = 20;
  options.moi_epoch_vgpr = 21;
  options.moi_runtime_sample_stride = 1;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(2u * kAccessesPerWindow);
  options.max_patches = 2u * kAccessesPerWindow + 4u;

  const ConSanResult result = try_patch_consan(
      make_gfx1250_code_object(words, "gfx1250_partitioned_sampled_barriers"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiSampledSyncMetadata,
                               &ConSanPatchInfo::kind),
            2u)
      << testing::PrintToString(result.warnings);
}

TEST(ConSanMoi, Gfx1250SampledClusterBarrierPublishesClusterScope) {
  std::vector<uint32_t> words(540, build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250));
  constexpr auto store = gfx1250::build_vds(gfx1250::kDsStoreB32Vds, {.addr = 0, .data0 = 0});
  words[0] = store[0];
  words[1] = store[1];
  const auto bypass_signal = build_s_cbranch_scc1(/*offset_dwords=*/1, ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(bypass_signal);
  words[400] = *bypass_signal;
  words[401] = 0xBE804EC3u; // s_barrier_signal -3
  words[402] = 0x3600009Fu; // v_and_b32_e32 v0, 31, v0
  words[403] = 0xBF048475u; // s_cmp_lt_i32 ttmp9, 4
  words[404] = 0xBF94FFFDu; // s_barrier_wait -3
  words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250);
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.moi_track_barriers = true;
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 80;
  options.moi_owner_vgpr = 20;
  options.moi_epoch_vgpr = 21;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(2);
  options.max_patches = 3;

  const ConSanResult result =
      try_patch_consan(make_gfx1250_code_object(words, "gfx1250_sampled_cluster_barrier"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  const auto patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &item) {
    return item.kind == ConSanPatchKind::TrampolineMoiSampledSyncMetadata &&
           item.anchor_offset == 404u * sizeof(uint32_t);
  });
  ASSERT_NE(patch, result.patches.end()) << testing::PrintToString(result.warnings);
  EXPECT_EQ(patch->covered_sync_event_count, 2u);
  EXPECT_EQ(result.sampled_barrier_applicable_event_count, 2u);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> trampoline =
      text_words_at_offset(patched, patch->trampoline_offset, patch->trampoline_size);
  const auto descriptor = encode_consan_moi_sampled_sync_metadata({
      .kind = ConSanMoiSampledSyncKind::Barrier,
      .role = ConSanMoiSampledSyncRole::AcquireRelease,
      .scope = ConSanMoiSampledSyncScope::Cluster,
      .epoch_before = 0,
      .epoch_after = 1,
  });
  ASSERT_EQ(descriptor.classification, ConSanMoiSampledSyncClassification::Valid);
  const auto descriptor_literal = build_v_mov_b32_e64_literal(
      /*vdst=*/10, descriptor.packed.descriptor, ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(descriptor_literal);
  EXPECT_TRUE(contains_subsequence(trampoline, *descriptor_literal));
}

TEST(ConSanMoi, Gfx1250SampledComposesWithAdjacentClusterBarrierDrop) {
  std::vector<uint32_t> words(43, build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250));
  constexpr auto store = gfx1250::build_vds(gfx1250::kDsStoreB32Vds, {.addr = 1, .data0 = 2});
  words[21] = store[0];
  words[22] = store[1];
  const auto bypass_signal = build_s_cbranch_scc1(/*offset_dwords=*/1, ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(bypass_signal);
  words[23] = *bypass_signal;
  words[24] = 0xBE804EC3u; // s_barrier_signal -3
  words[25] = 0xBF94FFFDu; // s_barrier_wait -3
  words[26] = 0xBFC60000u; // s_wait_dscnt 0
  words[27] = 0xBE804EC1u; // s_barrier_signal -1
  words[33] = 0xBF94FFFFu; // s_barrier_wait -1
  constexpr auto load = gfx1250::build_vds(gfx1250::kDsLoadB32Vds, {.addr = 1, .vdst = 2});
  words[34] = load[0];
  words[35] = load[1];
  words[42] = build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250);
  const std::vector<uint8_t> bytes =
      make_gfx1250_code_object(words, "gfx1250_sampled_cluster_drop_composition");

  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.moi_track_barriers = true;
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 80;
  options.moi_owner_vgpr = 20;
  options.moi_epoch_vgpr = 21;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(2);
  options.max_patches = 8;
  options.fault_dry_run = true;
  const ConSanResult inventory = try_patch_consan(bytes, options);
  const auto cluster_sequence =
      std::ranges::find_if(inventory.sync_sequences, [](const ConSanSyncSequence &sequence) {
        return sequence.operation == ConSanSyncOperation::BarrierFull &&
               sequence.barrier_scope == ConSanBarrierSite::Scope::Cluster;
      });
  ASSERT_NE(cluster_sequence, inventory.sync_sequences.end());
  const auto primary =
      std::ranges::find_if(inventory.fault_sites, [&](const ConSanFaultSite &site) {
        return site.sync_sequence_identity == cluster_sequence->identity &&
               site.mnemonic == "s_barrier_signal";
      });
  ASSERT_NE(primary, inventory.fault_sites.end());

  options.fault_drop_barrier = true;
  options.fault_site_identity = primary->identity;
  options.fault_barrier_sequence_identity = cluster_sequence->identity;
  options.fault_require_exactly_one = true;
  options.fault_dry_run = false;
  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid)
      << testing::PrintToString(result.errors);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_TRUE(result.staged_composition_validated);
  EXPECT_EQ(result.applied_fault_mutations, 1u);
  for (const ConSanPatchInfo &mutation : result.patches) {
    if (mutation.phase != ConSanPatchPhase::Mutation)
      continue;
    EXPECT_TRUE(std::ranges::none_of(result.patches, [&](const ConSanPatchInfo &instrumentation) {
      return instrumentation.phase == ConSanPatchPhase::Instrumentation &&
             instrumentation.original_size != 0u &&
             instrumentation.anchor_offset < mutation.anchor_offset + mutation.original_size &&
             mutation.anchor_offset < instrumentation.anchor_offset + instrumentation.original_size;
    }));
  }
}

TEST(ConSanMoi, SampledQualifiedBarrierUsesSpillBackedPersistentEpoch) {
  std::vector<uint32_t> words(540, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  words[0] = 0xD8340000u;
  words[1] = 0x00000000u;   // ds_store_b32 v0, v0
  words[400] = 0xBE804EC1u; // s_barrier_signal -1
  words[401] = 0xBF94FFFFu; // s_barrier_wait -1
  words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.moi_track_barriers = true;
  options.force_private_epoch = true;
  options.scratch_vgpr = 56;
  options.moi_runtime_sample_stride = 64;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(2);
  options.max_patches = 3;

  const ConSanResult result =
      try_patch_consan(make_rdna4_lds_code_object(words, "sampled_barrier_private_epoch"), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.moi_private_epoch_automatic);
  EXPECT_FALSE(result.resolved_moi_owner_vgpr);
  EXPECT_FALSE(result.resolved_moi_epoch_vgpr);
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &item) {
    return item.kind == ConSanPatchKind::TrampolineMoiSampledWatchpointStore ||
           item.kind == ConSanPatchKind::InlineMoiSampledWatchpointStore;
  });
  const auto barrier = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiSampledSyncMetadata, &ConSanPatchInfo::kind);
  const auto prologue = std::ranges::find(
      result.patches, ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue, &ConSanPatchInfo::kind);
  ASSERT_NE(access, result.patches.end()) << testing::PrintToString(result.warnings);
  ASSERT_NE(barrier, result.patches.end()) << testing::PrintToString(result.warnings);
  ASSERT_NE(prologue, result.patches.end()) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(access->persistent_epoch_private_offset);
  EXPECT_EQ(barrier->persistent_epoch_private_offset, access->persistent_epoch_private_offset);
  EXPECT_EQ(prologue->persistent_epoch_private_offset, access->persistent_epoch_private_offset);
  EXPECT_GT(barrier->required_private_segment_size, 0u);
}

TEST(ConSanMoi, SampledConditionallyExecutedBarrierPublishesOnlyWhenExecuted) {
  std::vector<uint32_t> words(540, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  words[0] = 0xD8340000u;
  words[1] = 0x00000000u;
  const auto bypass = build_s_cbranch_scc0(/*offset_dwords=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(bypass);
  words[400] = *bypass;
  words[401] = 0xBE804EC1u;
  words[402] = 0xBF94FFFFu;
  words[403] = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.moi_track_barriers = true;
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 80;
  options.moi_owner_vgpr = 20;
  options.moi_epoch_vgpr = 21;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(2);
  options.max_patches = 3;

  const ConSanResult result =
      try_patch_consan(make_rdna4_lds_code_object(words, "bypassable_barrier"), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiSampledSyncMetadata,
                               &ConSanPatchInfo::kind),
            1u);
  EXPECT_EQ(result.sampled_barrier_applicable_event_count, 2u);
}

TEST(ConSanMoi, SampledBarrierInConditionallyExitingLoopPublishesEveryIteration) {
  std::vector<uint32_t> words(540, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  words[0] = 0xD8340000u;
  words[1] = 0x00000000u;
  words[400] = 0xBE804EC1u;
  words[401] = 0xBF94FFFFu;
  const auto exit = build_s_cbranch_scc1(/*offset_dwords=*/1, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(exit);
  words[402] = *exit;
  words[403] = build_s_branch(-4, ROCJITSU_CODE_ARCH_RDNA4);
  words[404] = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);
  ConSanOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.moi_track_barriers = true;
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 80;
  options.moi_owner_vgpr = 20;
  options.moi_epoch_vgpr = 21;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(2);
  options.max_patches = 3;

  const ConSanResult result =
      try_patch_consan(make_rdna4_lds_code_object(words, "barrier_loop"), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiSampledSyncMetadata,
                               &ConSanPatchInfo::kind),
            1u);
  EXPECT_EQ(result.sampled_barrier_applicable_event_count, 2u);
}

TEST(ConSanMoi, SampledWatchpointRoundTripsRangeFields) {
  constexpr uint64_t packed =
      pack_consan_moi_sampled_watchpoint_entry(ConSanMoiShadowAccessKind::Read,
                                               /*owner_id=*/0x512,
                                               /*epoch=*/0x4aa,
                                               /*generation=*/0x1abcde,
                                               /*start_cell=*/0x4567,
                                               /*cell_count=*/32,
                                               /*consumed=*/true);
  constexpr ConSanMoiSampledWatchpointEntry decoded =
      decode_consan_moi_sampled_watchpoint_entry(packed);

  EXPECT_TRUE(decoded.valid);
  EXPECT_TRUE(decoded.consumed);
  EXPECT_EQ(decoded.kind, ConSanMoiShadowAccessKind::Read);
  EXPECT_EQ(decoded.owner_id, 0x112u);
  EXPECT_EQ(decoded.epoch, 0xaau);
  EXPECT_EQ(decoded.generation, 0xabcdeu);
  EXPECT_EQ(decoded.start_cell, 0x567u);
  EXPECT_EQ(decoded.cell_count, 32u);
}

TEST(ConSanMoi, SampledAtomicPublicationHasNoStableHybridSnapshot) {
  constexpr uint64_t old_entry = pack_consan_moi_sampled_watchpoint_entry(
      ConSanMoiShadowAccessKind::Write, /*owner_id=*/1, /*epoch=*/2,
      /*generation=*/6, /*start_cell=*/10, /*cell_count=*/1);
  constexpr uint64_t new_entry = pack_consan_moi_sampled_watchpoint_entry(
      ConSanMoiShadowAccessKind::Read, /*owner_id=*/7, /*epoch=*/9,
      /*generation=*/7, /*start_cell=*/31, /*cell_count=*/4);
  constexpr uint32_t old_low = static_cast<uint32_t>(old_entry);
  constexpr uint32_t old_high = static_cast<uint32_t>(old_entry >> 32u);
  constexpr uint32_t new_low = static_cast<uint32_t>(new_entry);
  constexpr uint32_t new_high = static_cast<uint32_t>(new_entry >> 32u);
  // The atomic exchange changes both words at one point. A low/high/low host
  // observation can therefore see the old entry, the new entry, or detect that
  // the exchange happened during the observation; it cannot accept a hybrid.
  constexpr std::array<uint32_t, 2> low_by_stage = {old_low, new_low};
  constexpr std::array<uint32_t, 2> high_by_stage = {old_high, new_high};

  for (size_t low_before_stage = 0; low_before_stage < low_by_stage.size(); ++low_before_stage) {
    for (size_t high_stage = low_before_stage; high_stage < high_by_stage.size(); ++high_stage) {
      for (size_t low_after_stage = high_stage; low_after_stage < low_by_stage.size();
           ++low_after_stage) {
        SCOPED_TRACE(::testing::Message() << "stages=" << low_before_stage << ',' << high_stage
                                          << ',' << low_after_stage);
        const ConSanMoiSampledSnapshot snapshot = classify_consan_moi_sampled_snapshot(
            {low_by_stage[low_before_stage], high_by_stage[high_stage],
             low_by_stage[low_after_stage]},
            /*active_generation=*/7);
        if (snapshot.state == ConSanMoiSampledSnapshotState::Stable) {
          EXPECT_EQ(low_before_stage, 1u);
          EXPECT_EQ(high_stage, 1u);
          EXPECT_EQ(low_after_stage, 1u);
          EXPECT_EQ(snapshot.entry.generation, 7u);
          EXPECT_EQ(snapshot.entry.owner_id, 7u);
          EXPECT_EQ(snapshot.entry.start_cell, 31u);
        } else if (low_before_stage == 0 && high_stage == 0 && low_after_stage == 0) {
          EXPECT_EQ(snapshot.state, ConSanMoiSampledSnapshotState::StaleGeneration);
        } else {
          EXPECT_EQ(snapshot.state, ConSanMoiSampledSnapshotState::ChangedDuringRead);
        }
      }
    }
  }
}

TEST(ConSanMoi, SampledReplayClassifiesUnusableSnapshotsWithoutConflicts) {
  constexpr uint64_t stale =
      pack_consan_moi_sampled_watchpoint_entry(ConSanMoiShadowAccessKind::Write, 1, 2, 6, 10, 1);
  constexpr uint64_t incomplete =
      pack_consan_moi_sampled_watchpoint_entry(ConSanMoiShadowAccessKind::Write, 2, 2, 7, 10, 1) &
      ~consan_moi_sampled_watchpoint::valid_mask;
  constexpr uint64_t malformed =
      pack_consan_moi_sampled_watchpoint_entry(ConSanMoiShadowAccessKind::Atomic, 3, 2, 7, 10, 1);
  constexpr uint64_t stable =
      pack_consan_moi_sampled_watchpoint_entry(ConSanMoiShadowAccessKind::Read, 4, 2, 7, 20, 1);
  const auto words = [](uint64_t packed) {
    const uint32_t low = static_cast<uint32_t>(packed);
    return ConSanMoiSampledSnapshotWords{low, static_cast<uint32_t>(packed >> 32u), low};
  };
  std::array<ConSanMoiSampledSnapshotWords, 6> snapshots = {words(0),
                                                            words(stale),
                                                            words(incomplete),
                                                            {static_cast<uint32_t>(stable),
                                                             static_cast<uint32_t>(stable >> 32u),
                                                             static_cast<uint32_t>(stable) ^ 0x20u},
                                                            words(malformed),
                                                            words(stable)};
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/0,
      /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/0,
      /*sampled_watchpoint_capacity=*/snapshots.size());
  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};

  const ConSanMoiSampledReplayResult replay =
      consan_moi_sampled_replay_snapshots(header, snapshots, diagnostics);

  EXPECT_EQ(replay.processed_entry_count, snapshots.size());
  EXPECT_EQ(replay.empty_entry_count, 1u);
  EXPECT_EQ(replay.stale_generation_entry_count, 1u);
  EXPECT_EQ(replay.incomplete_publication_entry_count, 1u);
  EXPECT_EQ(replay.changed_during_read_entry_count, 1u);
  EXPECT_EQ(replay.malformed_entry_count, 1u);
  EXPECT_FALSE(replay.conflict);
  EXPECT_EQ(replay.emitted_diagnostic_count, 0u);
}

TEST(ConSanMoi, SampledWatchpointPublishesPackedAccessRecords) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/2,
      /*diagnostic_capacity=*/0, /*exact_shadow_entry_capacity=*/0,
      /*sampled_watchpoint_capacity=*/2);
  header.access_record_count = 2;

  std::array<ConSanMoiAccessRecord, 2> records{};
  records[0].wave_id = 1;
  records[0].epoch = 3;
  records[0].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
  records[0].lds_byte_offset = 8;
  records[0].lds_byte_count = 4;

  records[1].generation = 9;
  records[1].wave_id = 2;
  records[1].epoch = 4;
  records[1].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read);
  records[1].start_cell = 5;
  records[1].cell_count = 2;

  std::array<uint64_t, 2> sampled{};

  const ConSanMoiSampledPublishResult result =
      consan_moi_sampled_publish_access_records(header, records, sampled);

  EXPECT_EQ(result.processed_access_count, 2u);
  EXPECT_EQ(result.published_entry_count, 2u);
  EXPECT_FALSE(result.sampled_capacity_exhausted);

  const ConSanMoiSampledWatchpointEntry first =
      decode_consan_moi_sampled_watchpoint_entry(sampled[0]);
  EXPECT_TRUE(first.valid);
  EXPECT_EQ(first.kind, ConSanMoiShadowAccessKind::Write);
  EXPECT_EQ(first.owner_id, 1u);
  EXPECT_EQ(first.epoch, 3u);
  EXPECT_EQ(first.generation, 7u);
  EXPECT_EQ(first.start_cell, 2u);
  EXPECT_EQ(first.cell_count, 1u);

  const ConSanMoiSampledWatchpointEntry second =
      decode_consan_moi_sampled_watchpoint_entry(sampled[1]);
  EXPECT_TRUE(second.valid);
  EXPECT_EQ(second.kind, ConSanMoiShadowAccessKind::Read);
  EXPECT_EQ(second.owner_id, 2u);
  EXPECT_EQ(second.epoch, 4u);
  EXPECT_EQ(second.generation, 9u);
  EXPECT_EQ(second.start_cell, 5u);
  EXPECT_EQ(second.cell_count, 2u);
}

TEST(ConSanMoi, SampledWatchpointPublishReportsCapacityExhaustion) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/2,
      /*diagnostic_capacity=*/0, /*exact_shadow_entry_capacity=*/0,
      /*sampled_watchpoint_capacity=*/1);
  header.access_record_count = 2;

  std::array<ConSanMoiAccessRecord, 2> records{};
  records[0].wave_id = 1;
  records[0].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
  records[0].cell_count = 1;
  records[1].wave_id = 2;
  records[1].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read);
  records[1].cell_count = 1;

  std::array<uint64_t, 1> sampled{};

  const ConSanMoiSampledPublishResult result =
      consan_moi_sampled_publish_access_records(header, records, sampled);

  EXPECT_EQ(result.processed_access_count, 2u);
  EXPECT_EQ(result.published_entry_count, 1u);
  EXPECT_TRUE(result.sampled_capacity_exhausted);
}

TEST(ConSanMoi, SampledWatchpointReplayEmitsLowerFidelityConflictDiagnostic) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/0,
      /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/0,
      /*sampled_watchpoint_capacity=*/2);
  std::array<uint64_t, 2> sampled = {
      pack_consan_moi_sampled_watchpoint_entry(ConSanMoiShadowAccessKind::Write,
                                               /*owner_id=*/1, /*epoch=*/3, /*generation=*/7,
                                               /*start_cell=*/2, /*cell_count=*/1),
      pack_consan_moi_sampled_watchpoint_entry(ConSanMoiShadowAccessKind::Read,
                                               /*owner_id=*/2, /*epoch=*/3, /*generation=*/7,
                                               /*start_cell=*/2, /*cell_count=*/1),
  };
  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};

  const ConSanMoiSampledReplayResult replay =
      consan_moi_sampled_replay_entries(header, sampled, diagnostics);

  EXPECT_EQ(replay.processed_entry_count, 2u);
  EXPECT_EQ(replay.emitted_diagnostic_count, 1u);
  EXPECT_FALSE(replay.diagnostic_capacity_exhausted);
  EXPECT_TRUE(replay.conflict);
  ASSERT_EQ(header.diagnostic_count, 1u);
  EXPECT_EQ(diagnostics[0].kind, static_cast<uint32_t>(ConSanMoiDiagnosticKind::AccessConflict));
  EXPECT_EQ(diagnostics[0].backend, static_cast<uint32_t>(ConSanMoiEngine::Sampled));
  EXPECT_EQ(diagnostics[0].generation, 7u);
  EXPECT_EQ(diagnostics[0].epoch, 3u);
  EXPECT_EQ(diagnostics[0].first_owner_id, 1u);
  EXPECT_EQ(diagnostics[0].second_owner_id, 2u);
  EXPECT_EQ(diagnostics[0].first_access_kind,
            static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write));
  EXPECT_EQ(diagnostics[0].second_access_kind,
            static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read));
}

TEST(ConSanMoi, SampledWatchpointReplayDoesNotTreatCleanSnapshotAsProof) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/0,
      /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/0,
      /*sampled_watchpoint_capacity=*/2);
  std::array<uint64_t, 2> sampled = {
      pack_consan_moi_sampled_watchpoint_entry(ConSanMoiShadowAccessKind::Write,
                                               /*owner_id=*/1, /*epoch=*/3, /*generation=*/7,
                                               /*start_cell=*/2, /*cell_count=*/1),
      pack_consan_moi_sampled_watchpoint_entry(ConSanMoiShadowAccessKind::Read,
                                               /*owner_id=*/2, /*epoch=*/3, /*generation=*/7,
                                               /*start_cell=*/3, /*cell_count=*/1),
  };
  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};

  const ConSanMoiSampledReplayResult replay =
      consan_moi_sampled_replay_entries(header, sampled, diagnostics);

  EXPECT_EQ(replay.processed_entry_count, 2u);
  EXPECT_EQ(replay.emitted_diagnostic_count, 0u);
  EXPECT_FALSE(replay.conflict);
  EXPECT_EQ(header.diagnostic_count, 0u);
}

TEST(ConSanMoi, SampledWatchpointReplayIgnoresStaleGenerations) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/8, /*dispatch_id=*/11, /*access_record_capacity=*/0,
      /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/0,
      /*sampled_watchpoint_capacity=*/2);
  std::array<uint64_t, 2> sampled = {
      pack_consan_moi_sampled_watchpoint_entry(ConSanMoiShadowAccessKind::Write,
                                               /*owner_id=*/1, /*epoch=*/3, /*generation=*/7,
                                               /*start_cell=*/2, /*cell_count=*/1),
      pack_consan_moi_sampled_watchpoint_entry(ConSanMoiShadowAccessKind::Read,
                                               /*owner_id=*/2, /*epoch=*/3, /*generation=*/7,
                                               /*start_cell=*/2, /*cell_count=*/1),
  };
  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};

  const ConSanMoiSampledReplayResult replay =
      consan_moi_sampled_replay_entries(header, sampled, diagnostics);

  EXPECT_EQ(replay.processed_entry_count, 2u);
  EXPECT_EQ(replay.emitted_diagnostic_count, 0u);
  EXPECT_FALSE(replay.conflict);
  EXPECT_EQ(header.diagnostic_count, 0u);
}

TEST(ConSanMoi, SampledWatchpointConflictRequiresExactRangeOverlap) {
  constexpr ConSanMoiSampledWatchpointEntry current{
      /*valid=*/true,
      /*consumed=*/false, ConSanMoiShadowAccessKind::Write,
      /*owner_id=*/3,
      /*epoch=*/11,
      /*generation=*/42,
      /*start_cell=*/20,
      /*cell_count=*/4,
  };
  constexpr ConSanMoiSampledWatchpointEntry overlapping_read{
      /*valid=*/true,
      /*consumed=*/false, ConSanMoiShadowAccessKind::Read,
      /*owner_id=*/1,
      /*epoch=*/11,
      /*generation=*/42,
      /*start_cell=*/23,
      /*cell_count=*/3,
  };
  constexpr ConSanMoiSampledWatchpointEntry adjacent_read{
      /*valid=*/true,
      /*consumed=*/false, ConSanMoiShadowAccessKind::Read,
      /*owner_id=*/1,
      /*epoch=*/11,
      /*generation=*/42,
      /*start_cell=*/24,
      /*cell_count=*/2,
  };
  constexpr ConSanMoiSampledWatchpointEntry consumed_read{
      /*valid=*/true,
      /*consumed=*/true, ConSanMoiShadowAccessKind::Read,
      /*owner_id=*/1,
      /*epoch=*/11,
      /*generation=*/42,
      /*start_cell=*/23,
      /*cell_count=*/3,
  };
  constexpr ConSanMoiSampledWatchpointEntry old_generation{
      /*valid=*/true,
      /*consumed=*/false, ConSanMoiShadowAccessKind::Read,
      /*owner_id=*/1,
      /*epoch=*/11,
      /*generation=*/41,
      /*start_cell=*/23,
      /*cell_count=*/3,
  };

  EXPECT_TRUE(consan_moi_sampled_watchpoints_conflict(current, overlapping_read));
  EXPECT_FALSE(consan_moi_sampled_watchpoints_conflict(current, adjacent_read));
  EXPECT_FALSE(consan_moi_sampled_watchpoints_conflict(current, consumed_read));
  EXPECT_FALSE(consan_moi_sampled_watchpoints_conflict(current, old_generation));
}

} // namespace
} // namespace rocjitsu
