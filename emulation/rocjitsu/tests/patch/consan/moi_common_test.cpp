// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"
#include "rocjitsu/code/patch/consan/consan_moi_internal.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"

namespace rocjitsu {
namespace {

TEST(ConSanMoi, ScalarPersistentTemporaryValidationIsNoopWhenDisabled) {
  ConSanOptions disabled;
  std::vector<std::string> errors;
  EXPECT_TRUE(consan_detail::validate_scalar_state_temporaries(disabled, "test consumer", errors));
  EXPECT_TRUE(errors.empty());
}

TEST(ConSanMoi, ScalarPersistentTemporaryValidationFailsClosed) {
  for (uint32_t present_mask = 0u; present_mask < 3u; ++present_mask) {
    SCOPED_TRACE(present_mask);
    ConSanOptions options;
    options.moi_persistent_owner_sgpr = 40u;
    options.moi_persistent_epoch_sgpr = 41u;
    if (present_mask & 1u)
      options.moi_owner_vgpr = 6u;
    if (present_mask & 2u)
      options.moi_epoch_vgpr = 7u;
    std::vector<std::string> errors;

    EXPECT_FALSE(
        consan_detail::validate_scalar_state_temporaries(options, "test consumer", errors));
    ASSERT_EQ(errors.size(), 1u);
    EXPECT_NE(errors.front().find("test consumer has no scalar-state VGPR temporaries"),
              std::string::npos);
  }

  ConSanOptions valid;
  valid.moi_persistent_owner_sgpr = 40u;
  valid.moi_persistent_epoch_sgpr = 41u;
  valid.moi_owner_vgpr = 6u;
  valid.moi_epoch_vgpr = 7u;
  std::vector<std::string> errors;
  EXPECT_TRUE(consan_detail::validate_scalar_state_temporaries(valid, "test consumer", errors));
  EXPECT_TRUE(errors.empty());
}

TEST(ConSanMoi, ScalarOwnerContextResolutionFailsClosedAndComputesTailFloor) {
  using Summary = consan_detail::ScalarOwnerContextSummary;
  const std::array contexts = {
      Summary{.descriptor_file_offset = 0x10u,
              .current_sgpr_count = 40u,
              .max_referenced_sgpr_count = 48u,
              .descriptor_valid = true},
      Summary{.descriptor_file_offset = 0x20u,
              .current_sgpr_count = 80u,
              .max_referenced_sgpr_count = 72u,
              .descriptor_valid = true},
  };
  constexpr std::array<uint64_t, 2> kOwners = {0x20u, 0x10u};

  const auto resolved = consan_detail::resolve_scalar_owner_contexts(true, contexts, kOwners);

  ASSERT_TRUE(resolved);
  EXPECT_EQ(resolved->context_indices, (std::vector<size_t>{1u, 0u}));
  EXPECT_EQ(resolved->tail_floor, 80u);

  EXPECT_FALSE(consan_detail::resolve_scalar_owner_contexts(false, contexts, kOwners));
  EXPECT_FALSE(
      consan_detail::resolve_scalar_owner_contexts(true, contexts, std::span<const uint64_t>{}));
  constexpr std::array<uint64_t, 1> kMissingOwner = {0x30u};
  EXPECT_FALSE(consan_detail::resolve_scalar_owner_contexts(true, contexts, kMissingOwner));

  std::array invalid_contexts = contexts;
  invalid_contexts[1].descriptor_valid = false;
  EXPECT_FALSE(consan_detail::resolve_scalar_owner_contexts(true, invalid_contexts, kOwners));
  invalid_contexts[1].descriptor_valid = true;
  invalid_contexts[1].descriptor_file_offset = std::nullopt;
  EXPECT_FALSE(consan_detail::resolve_scalar_owner_contexts(true, invalid_contexts, kOwners));

  const std::array contexts_with_unrelated_invalid = {
      Summary{.descriptor_file_offset = std::nullopt,
              .current_sgpr_count = 200u,
              .max_referenced_sgpr_count = 200u,
              .descriptor_valid = false},
      contexts[0],
  };
  constexpr std::array<uint64_t, 1> kValidOwner = {0x10u};
  // Invalid contexts that are not named owners must not affect owner resolution
  // or raise the scalar tail floor.
  const auto skipped_invalid = consan_detail::resolve_scalar_owner_contexts(
      true, contexts_with_unrelated_invalid, kValidOwner);
  ASSERT_TRUE(skipped_invalid);
  EXPECT_EQ(skipped_invalid->context_indices, (std::vector<size_t>{1u}));
  EXPECT_EQ(skipped_invalid->tail_floor, 48u);
}

TEST(ConSanMoi, Cdna4HeterogeneousOwnersKeepUsableComponentAcrossMoiEngines) {
  constexpr uint64_t kHighPressureEntry = 320u * sizeof(uint32_t);
  for (ConSanMoiEngine engine :
       {ConSanMoiEngine::RecordReplay, ConSanMoiEngine::Sampled, ConSanMoiEngine::InlineShadow}) {
    for (uint16_t live_sgpr_count : {96u, 98u}) {
      SCOPED_TRACE(testing::PrintToString(engine) +
                   " live_sgprs=" + std::to_string(live_sgpr_count));
      const std::vector<uint8_t> bytes =
          make_cdna4_disconnected_scalar_pressure_code_object(live_sgpr_count);
      ASSERT_FALSE(bytes.empty());
      ConSanOptions options = moi_options(engine);
      options.moi_track_barriers = false;
      options.moi_track_atomics = false;
      options.moi_runtime_sample_stride = 2u;
      options.moi_report_buffer_address = 0x123456780000ull;
      options.moi_report_buffer_size = engine == ConSanMoiEngine::InlineShadow
                                           ? kInlineShadowFullLdsReportBufferSize
                                           : 64u * 1024u * 1024u;

      const ConSanResult result = try_patch_consan(bytes, options);

      ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
      ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
      EXPECT_TRUE(result.final_validation_passed);
      ASSERT_TRUE(result.resolved_moi_dispatch_id_sgpr);
      ASSERT_TRUE(result.resolved_moi_exec_save_sgpr);
      EXPECT_EQ(*result.resolved_moi_dispatch_id_sgpr, 96u);
      EXPECT_LT(*result.resolved_moi_exec_save_sgpr, 96u);

      const auto low = std::ranges::find(result.site_dispositions, 0u,
                                         &ConSanSiteDispositionRecord::text_offset);
      const auto high = std::ranges::find(result.site_dispositions, kHighPressureEntry,
                                          &ConSanSiteDispositionRecord::text_offset);
      ASSERT_NE(low, result.site_dispositions.end());
      ASSERT_NE(high, result.site_dispositions.end());
      EXPECT_EQ(low->lowering_outcome, ConSanSiteLoweringOutcome::Patched);
      EXPECT_EQ(high->lowering_outcome, ConSanSiteLoweringOutcome::ResourceFailed);
      EXPECT_EQ(high->resource_reason, ConSanRegisterPlanReason::ForbiddenOverlap);
      EXPECT_TRUE(std::ranges::any_of(result.patches, [](const ConSanPatchInfo &patch) {
        return patch.phase == ConSanPatchPhase::Instrumentation && patch.anchor_offset == 0u;
      }));
      EXPECT_TRUE(std::ranges::none_of(result.patches, [=](const ConSanPatchInfo &patch) {
        return patch.phase == ConSanPatchPhase::Instrumentation &&
               patch.anchor_offset == kHighPressureEntry;
      }));
      if (live_sgpr_count == 98u) {
        EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
          return warning.find("reserved a high dispatch-ID SGPR pair") != std::string::npos;
        }));
      }
    }
  }
}

TEST(ConSanMoi, Gfx1250Wave32DescriptorUsesSixteenVgprGranules) {
  constexpr auto store = gfx1250::build_vds(gfx1250::kDsStoreB32Vds, {.addr = 0, .data0 = 1});
  const std::array<uint32_t, 3> text_words = {store[0], store[1],
                                              build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250)};
  const std::vector<uint8_t> bytes =
      make_gfx1250_code_object(text_words, "lds_probe", /*vgpr_granulated=*/4);
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().current_vgpr_count, 80u);
}

TEST(ConSanMoi, Cdna3Wave64DescriptorUsesEightVgprGranules) {
  constexpr auto store = cdna3::build_ds(cdna3::kDsWriteB32Ds, {.addr = 0, .data0 = 1});
  const std::array<uint32_t, 3> text_words = {store[0], store[1],
                                              build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA3)};
  const std::vector<uint8_t> bytes =
      make_cdna3_lds_code_object(text_words, "lds_probe", /*vgpr_granulated=*/3);
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().current_vgpr_count, 32u);
}

TEST(ConSanMoi, Gfx1250TwoAddressLoadScratchAvoidsCompleteDestinationPair) {
  constexpr auto load =
      gfx1250::build_vds(gfx1250::kDsLoad2addrStride64B32Vds, {.addr = 0, .vdst = 1});
  const std::array<uint32_t, 3> text_words = {load[0], load[1],
                                              build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250)};
  const std::vector<uint8_t> bytes = make_gfx1250_code_object(text_words, "two_address_load");
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(2, 0, 0, 0);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  ASSERT_TRUE(result.resource_plans.front().scratch_vgpr);
  EXPECT_GE(*result.resource_plans.front().scratch_vgpr, 3u);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::TrampolineMoiAccessRecordStore);
}

struct NativeB96Access {
  std::array<uint32_t, 2> words;
  std::string_view mnemonic;
  bool load_clobbers_address = false;
};

template <size_t AccessCount, typename CodeObjectFactory>
void expect_moi_engines_admit_native_b96_accesses(
    rj_code_arch_t arch, const std::array<NativeB96Access, AccessCount> &accesses,
    CodeObjectFactory code_object_factory) {
  for (ConSanMoiEngine engine :
       {ConSanMoiEngine::RecordReplay, ConSanMoiEngine::InlineShadow, ConSanMoiEngine::Sampled}) {
    for (const auto &[access, mnemonic, load_clobbers_address] : accesses) {
      SCOPED_TRACE(std::string(consan_moi_engine_name(engine)) + " " + std::string(mnemonic));
      const std::array<uint32_t, 3> text_words = {access[0], access[1], build_s_endpgm(arch)};
      const std::vector<uint8_t> bytes = code_object_factory(text_words);
      ConSanOptions options = moi_options(engine);
      options.moi_track_atomics = false;
      options.moi_track_barriers = false;
      options.moi_runtime_sample_stride = engine == ConSanMoiEngine::Sampled ? 2u : 1u;
      options.moi_report_buffer_address = 0x123456780000ull;
      options.moi_report_buffer_size = engine == ConSanMoiEngine::InlineShadow
                                           ? kInlineShadowFullLdsReportBufferSize
                                           : 64u * 1024u * 1024u;

      const ConSanResult result = try_patch_consan(bytes, options);

      ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
      ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
      ASSERT_EQ(result.moi_candidates.size(), 1u);
      EXPECT_EQ(result.moi_candidates.front().mnemonic, mnemonic);
      EXPECT_EQ(result.moi_candidates.front().width_bits, 96u);
      const ConSanPatchKind expected_patch_kind =
          engine == ConSanMoiEngine::RecordReplay ? ConSanPatchKind::TrampolineMoiAccessRecordStore
          : engine == ConSanMoiEngine::InlineShadow
              ? ConSanPatchKind::TrampolineMoiExactShadowStore
              : ConSanPatchKind::TrampolineMoiSampledWatchpointStore;
      const auto access_patch =
          std::ranges::find(result.patches, expected_patch_kind, &ConSanPatchInfo::kind);
      ASSERT_NE(access_patch, result.patches.end());
      ASSERT_TRUE(access_patch->scratch_vgpr);
      AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
      ASSERT_TRUE(patched.is_valid());
      const std::vector<uint32_t> body = text_words_at_offset(
          patched, access_patch->trampoline_offset, access_patch->trampoline_size);
      if (engine == ConSanMoiEngine::RecordReplay) {
        EXPECT_TRUE(contains_subsequence(
            body, make_expected_literal_offset_store_words(
                      offsetof(ConSanMoiAccessRecord, lds_byte_count), 3u * sizeof(uint32_t),
                      *access_patch->scratch_vgpr,
                      static_cast<uint16_t>(*access_patch->scratch_vgpr + 2u))));
      } else if (engine == ConSanMoiEngine::InlineShadow) {
        const uint16_t loop_counter_vgpr = consan_detail::inline_shadow_loop_counter_vgpr(
            *access_patch->scratch_vgpr, result.resolved_moi_exec_save_sgpr.has_value(),
            options.moi_track_atomics);
        const auto resource_plan = std::ranges::find_if(
            result.resource_plans, [&](const ConSanCandidateResourcePlan &plan) {
              return plan.site_kind == ConSanResourceSiteKind::Access &&
                     plan.candidate_index == 0u && plan.scratch_vgpr == access_patch->scratch_vgpr;
            });
        ASSERT_NE(resource_plan, result.resource_plans.end());
        const uint16_t reserved_end =
            static_cast<uint16_t>(*resource_plan->scratch_vgpr + resource_plan->scratch_vgpr_count);
        EXPECT_LT(loop_counter_vgpr, reserved_end);
        const auto three_cell_bound = instrumentation::build_v_cmp_gt_u32_vcc(
            scalar_positive_inline_u32(3u), loop_counter_vgpr, arch);
        ASSERT_TRUE(three_cell_bound);
        EXPECT_NE(std::ranges::find(body, *three_cell_bound), body.end());
        if (load_clobbers_address) {
          const uint16_t saved_address_vgpr = static_cast<uint16_t>(
              loop_counter_vgpr + consan_detail::inline_shadow_loop_scratch_count(
                                      result.moi_candidates.front().width_bits,
                                      consan_moi_exact_shadow::granule_bytes));
          EXPECT_LT(saved_address_vgpr, reserved_end);
          EXPECT_NE(std::ranges::find(body, build_v_mov_b32_e32(saved_address_vgpr,
                                                                vector_source_vgpr(0u), arch)),
                    body.end());
        }
      } else {
        const uint16_t high_vgpr = static_cast<uint16_t>(*access_patch->scratch_vgpr + 3u);
        const uint16_t tmp_vgpr = static_cast<uint16_t>(*access_patch->scratch_vgpr + 4u);
        const uint32_t encoded_three_cell_count =
            encode_consan_moi_sampled_cell_count(3u)
            << (consan_moi_sampled_watchpoint::count_shift - 32u);
        const auto cell_count_literal =
            instrumentation::build_v_mov_b32_literal(tmp_vgpr, encoded_three_cell_count, arch);
        const auto add_cell_count = instrumentation::build_v_add_u32(
            high_vgpr, vector_source_vgpr(high_vgpr), tmp_vgpr, arch);
        ASSERT_TRUE(cell_count_literal && add_cell_count);
        std::vector<uint32_t> expected = *cell_count_literal;
        expected.insert(expected.end(), add_cell_count->begin(), add_cell_count->end());
        EXPECT_TRUE(contains_subsequence(body, expected));
      }
      EXPECT_EQ(std::ranges::count_if(result.site_dispositions,
                                      [](const ConSanSiteDispositionRecord &site) {
                                        return site.site_kind == ConSanResourceSiteKind::Access &&
                                               site.disposition ==
                                                   ConSanSiteDisposition::Supported &&
                                               site.lowering_outcome ==
                                                   ConSanSiteLoweringOutcome::Patched;
                                      }),
                1);
    }
  }
}

TEST(ConSanMoi, Gfx1250MoiEnginesAdmitNativeB96Accesses) {
  constexpr auto store =
      gfx1250::build_vds(gfx1250::kDsStoreB96Vds, {.offset0 = 12, .addr = 0, .data0 = 1});
  constexpr auto load =
      gfx1250::build_vds(gfx1250::kDsLoadB96Vds, {.offset0 = 12, .addr = 0, .vdst = 4});
  constexpr auto aliasing_load =
      gfx1250::build_vds(gfx1250::kDsLoadB96Vds, {.offset0 = 12, .addr = 0, .vdst = 0});
  constexpr std::array<NativeB96Access, 3> accesses = {
      NativeB96Access{store, "ds_store_b96", false},
      NativeB96Access{load, "ds_load_b96", false},
      NativeB96Access{aliasing_load, "ds_load_b96", true},
  };

  expect_moi_engines_admit_native_b96_accesses(
      ROCJITSU_CODE_ARCH_GFX1250, accesses, [](const auto &text_words) {
        return make_gfx1250_code_object(text_words, "native_b96_access");
      });
}

TEST(ConSanMoi, Gfx1250RelaxedLdsAtomicIsAccessButNotSynchronization) {
  constexpr auto atomic = gfx1250::build_vds(
      gfx1250::kDsCmpstoreRtnB32Vds, {.offset0 = 12, .addr = 0, .data0 = 1, .data1 = 2, .vdst = 3});
  const std::array<uint32_t, 3> text_words = {atomic[0], atomic[1],
                                              build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250)};
  const std::vector<uint8_t> bytes = make_gfx1250_code_object(text_words, "relaxed_lds_atomic");
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_track_atomics = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 0, 1, 1);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << "warnings=" << testing::PrintToString(result.warnings)
                               << " dispositions="
                               << testing::PrintToString(result.site_dispositions)
                               << " plans=" << testing::PrintToString(result.resource_plans);
  ASSERT_EQ(result.moi_candidates.size(), 1u);
  EXPECT_EQ(result.moi_candidates.front().kind, ConSanLdsAccessKind::Atomic);
  EXPECT_EQ(result.moi_candidates.front().mnemonic, "ds_cmpstore_rtn_b32");
  EXPECT_TRUE(std::ranges::any_of(result.site_dispositions, [](const auto &site) {
    return site.site_kind == ConSanResourceSiteKind::Access &&
           site.disposition == ConSanSiteDisposition::Supported &&
           site.lowering_outcome == ConSanSiteLoweringOutcome::Patched;
  }));
  EXPECT_TRUE(std::ranges::any_of(result.site_dispositions, [](const auto &site) {
    return site.site_kind == ConSanResourceSiteKind::Atomic &&
           site.disposition == ConSanSiteDisposition::NotApplicable &&
           site.reason == ConSanSiteDispositionReason::UnqualifiedSyncSequence;
  }));
}

TEST(ConSanMoi, Cdna4HistogramLdsAtomicsAreAccessesButNotSynchronization) {
  constexpr auto add_u32 =
      cdna4::build_ds(cdna4::kDsAddU32Ds, {.offset0 = 4, .addr = 3, .data0 = 7});
  constexpr auto add_u64 =
      cdna4::build_ds(cdna4::kDsAddU64Ds, {.offset0 = 8, .addr = 5, .data0 = 8});
  constexpr auto add_f32 =
      cdna4::build_ds(cdna4::kDsAddF32Ds, {.offset0 = 12, .addr = 7, .data0 = 3});
  constexpr auto add_f64 =
      cdna4::build_ds(cdna4::kDsAddF64Ds, {.offset0 = 16, .addr = 9, .data0 = 14});
  constexpr auto cmpst = cdna4::build_ds(
      cdna4::kDsCmpstRtnB32Ds, {.offset0 = 20, .addr = 12, .data0 = 11, .data1 = 13, .vdst = 13});
  const std::array<uint32_t, 11> text_words = {
      add_u32[0],
      add_u32[1],
      add_u64[0],
      add_u64[1],
      add_f32[0],
      add_f32[1],
      add_f64[0],
      add_f64[1],
      cmpst[0],
      cmpst[1],
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  const std::vector<uint8_t> bytes =
      make_cdna4_lds_code_object(text_words, "cdna4_histogram_lds_atomics");
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_track_barriers = false;
  options.moi_track_atomics = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(5, 0, 0, 0, 0, 1, 1);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.moi_candidates.size(), 5u);
  EXPECT_TRUE(std::ranges::all_of(result.moi_candidates, [](const auto &candidate) {
    return candidate.kind == ConSanLdsAccessKind::Atomic;
  }));
  EXPECT_EQ(std::ranges::count_if(result.site_dispositions,
                                  [](const auto &site) {
                                    return site.site_kind == ConSanResourceSiteKind::Access &&
                                           site.disposition == ConSanSiteDisposition::Supported;
                                  }),
            5u);
  EXPECT_TRUE(std::ranges::any_of(result.site_dispositions, [](const auto &site) {
    return site.site_kind == ConSanResourceSiteKind::Access &&
           site.lowering_outcome == ConSanSiteLoweringOutcome::Patched;
  }));
  EXPECT_TRUE(std::ranges::none_of(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiAtomicRecord;
  }));

  EXPECT_FALSE(consan_moi_shadow_kind_conflicts(ConSanMoiShadowAccessKind::Atomic,
                                                ConSanMoiShadowAccessKind::Atomic));
  EXPECT_TRUE(consan_moi_shadow_kind_conflicts(ConSanMoiShadowAccessKind::Atomic,
                                               ConSanMoiShadowAccessKind::Read));
  EXPECT_TRUE(consan_moi_shadow_kind_conflicts(ConSanMoiShadowAccessKind::Atomic,
                                               ConSanMoiShadowAccessKind::Write));
}

TEST(ConSanMoi, UnassociatedFenceIsNotApplicableOnEverySupportedTarget) {
  const std::array<uint32_t, 3> text_words = {
      0xF4042000u,
      0x00000000u, // s_dcache_inv
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::array<std::vector<uint8_t>, 2> objects = {
      make_rdna4_lds_code_object(text_words, "unassociated_fence"),
      make_gfx1250_code_object(text_words, "unassociated_fence"),
  };
  for (const std::vector<uint8_t> &bytes : objects) {
    ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
    options.moi_track_atomics = true;

    const ConSanResult result = try_patch_consan(bytes, options);

    ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
    ASSERT_EQ(result.moi_fence_candidates.size(), 1u);
    EXPECT_FALSE(result.moi_fence_candidates.front().eligible);
    EXPECT_TRUE(std::ranges::any_of(result.site_dispositions, [](const auto &site) {
      return site.site_kind == ConSanResourceSiteKind::Fence &&
             site.disposition == ConSanSiteDisposition::NotApplicable &&
             site.reason == ConSanSiteDispositionReason::IneligibleFence &&
             site.lowering_outcome == ConSanSiteLoweringOutcome::NotApplicable;
    }));
  }
}

TEST(ConSanMoi, Cdna4UnassociatedFenceIsNotApplicable) {
  const auto fence = build_cdna4_s_dcache_inv_vol(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(fence);
  std::vector<uint32_t> text_words(fence->begin(), fence->end());
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4));
  const std::vector<uint8_t> bytes = make_cdna4_lds_code_object(text_words, "unassociated_fence");
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_track_atomics = true;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_EQ(result.moi_fence_candidates.size(), 1u);
  EXPECT_FALSE(result.moi_fence_candidates.front().eligible);
  EXPECT_TRUE(std::ranges::any_of(result.site_dispositions, [](const auto &site) {
    return site.site_kind == ConSanResourceSiteKind::Fence &&
           site.disposition == ConSanSiteDisposition::NotApplicable &&
           site.reason == ConSanSiteDispositionReason::IneligibleFence &&
           site.lowering_outcome == ConSanSiteLoweringOutcome::NotApplicable;
  }));
}

TEST(ConSanMoi, InventoriesDynamicStackMarker) {
  const std::array<uint32_t, 3> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32 v0, v0
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "dynamic_stack_kernel", kRdna4Wave64AllVgprsGranulated,
                                 /*wave32=*/false, /*uses_dynamic_stack=*/true);
  ConSanOptions options = moi_options();

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_TRUE(result.kernels.front().uses_dynamic_stack.has_value());
  EXPECT_TRUE(*result.kernels.front().uses_dynamic_stack);
}

TEST(ConSanMoi, DynamicStackMetadataOverridesZeroValuedMarker) {
  constexpr std::string_view kernel_name = "metadata_dynamic_stack_kernel";
  const std::array<uint32_t, 3> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32 v0, v0
      0xBFB00000u, // s_endpgm
  };
  std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, kernel_name, kRdna4Wave64AllVgprsGranulated,
                                 /*wave32=*/false, /*uses_dynamic_stack=*/false);
  append_kernel_metadata_note(bytes, kernel_name, /*uses_dynamic_stack=*/true,
                              /*sgpr_count=*/83u);
  ConSanOptions options = moi_options();

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_TRUE(result.kernels.front().uses_dynamic_stack.has_value());
  EXPECT_TRUE(*result.kernels.front().uses_dynamic_stack);
  ASSERT_TRUE(result.kernels.front().sgpr_count.has_value());
  EXPECT_EQ(*result.kernels.front().sgpr_count, 83u);
  ASSERT_FALSE(result.resource_plans.empty());
  for (const ConSanCandidateResourcePlan &plan : result.resource_plans)
    EXPECT_EQ(plan.max_referenced_sgpr_count, 83u);
}

TEST(ConSanMoi, InventoriesHiddenDynamicLdsArgument) {
  constexpr std::string_view kernel_name = "dynamic_lds_kernel";
  const std::array<uint32_t, 3> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32 v0, v0
      0xBFB00000u, // s_endpgm
  };
  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, kernel_name, kRdna4Wave64AllVgprsGranulated, /*wave32=*/false);
  append_kernel_metadata_note(bytes, kernel_name, /*uses_dynamic_stack=*/false,
                              /*sgpr_count=*/0u, std::nullopt, std::nullopt,
                              /*has_dynamic_lds=*/true);

  const ConSanResult result = try_patch_consan(bytes, moi_options());

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.kernels.size(), 1u);
  EXPECT_TRUE(result.kernels.front().has_dynamic_lds);
}

TEST(ConSanMoi, InventoryIncludesLikelyGroupFlatSitesFromLocalFunctions) {
  const std::array<uint32_t, 1> kernel_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 9> function_words = {
      0xBE8001EBu,                           // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u, 0x00000000u,              // v_mov_b32_e64 v0, s0
      0xD5810001u, 0x00000001u,              // v_mov_b32_e64 v1, s1
      0xEC05007Cu, 0x00000002u, 0x00000000u, // flat_load_b32 v2, v[0:1]
      0xBFB00000u,                           // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words);
  ConSanOptions options = moi_options();

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.functions.size(), 1u);
  ASSERT_EQ(result.functions.front().flat_sites.size(), 1u);
  ASSERT_EQ(result.moi_candidates.size(), 1u);
  const ConSanMoiCandidate &candidate = result.moi_candidates.front();
  EXPECT_EQ(candidate.source, ConSanMoiCandidateSource::FlatGroup);
  EXPECT_EQ(candidate.kind, ConSanLdsAccessKind::Read);
  EXPECT_EQ(candidate.flat_address_space_hint, ConSanFlatAddressSpaceHint::Group);
  EXPECT_FALSE(candidate.in_kernel);
  EXPECT_EQ(candidate.container_name, "lds_helper");
  EXPECT_EQ(candidate.mnemonic, "flat_load_b32");
  EXPECT_EQ(candidate.text_offset, 24u);
  EXPECT_EQ(candidate.file_offset, 0x118u);
  EXPECT_EQ(candidate.size, 3u * sizeof(uint32_t));
  EXPECT_EQ(candidate.width_bits, 32u);
  ASSERT_TRUE(candidate.dst_vgpr);
  EXPECT_EQ(*candidate.dst_vgpr, 2u);
  ASSERT_TRUE(candidate.addr_vgpr);
  EXPECT_EQ(*candidate.addr_vgpr, 0u);
  ASSERT_TRUE(candidate.raw_vaddr);
  EXPECT_EQ(*candidate.raw_vaddr, 0u);
  ASSERT_TRUE(candidate.raw_vdst);
  EXPECT_EQ(*candidate.raw_vdst, 2u);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_TRUE(result.resource_plans.front().owner_descriptor_file_offsets.empty());
  EXPECT_EQ(result.resource_plans.front().source, ConSanRegisterAllocationSource::Unsupported);
  EXPECT_EQ(result.resource_plans.front().reason, ConSanRegisterPlanReason::MissingOwner);
  EXPECT_EQ(result.resource_plan_summary.unsupported_plans, 1u);
}

TEST(ConSanMoi, SharedHelperPlanUsesCommonDeadWindowAcrossTwoOwners) {
  const std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object();
  ConSanOptions options = moi_options();

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.kernels.size(), 3u);
  ASSERT_EQ(result.functions.size(), 1u);
  ASSERT_EQ(result.moi_candidates.size(), 1u);
  EXPECT_FALSE(result.moi_candidates.front().in_kernel);
  EXPECT_EQ(result.moi_candidates.front().container_name, "shared_lds_helper");
  ASSERT_EQ(result.resource_plans.size(), 1u);
  const ConSanCandidateResourcePlan &plan = result.resource_plans.front();
  ASSERT_EQ(plan.owner_descriptor_file_offsets.size(), 2u);
  EXPECT_EQ(plan.source, ConSanRegisterAllocationSource::LivenessDead);
  EXPECT_EQ(plan.reason, ConSanRegisterPlanReason::None);
  EXPECT_EQ(plan.scratch_vgpr, 1);
  EXPECT_EQ(plan.scratch_vgpr_count, 6u);
}

TEST(ConSanMoi, Cdna4ScalarStateClearsEverySharedOwnerAllocation) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  const auto access =
      build_cdna4_ds_store_b32(/*vaddr=*/10, /*vdata=*/11, /*byte_offset=*/0, kArch);
  ASSERT_TRUE(access);
  std::vector<uint32_t> helper(access->begin(), access->end());
  helper.push_back(0xBE802A02u); // s_movrels_b32 s0, s2

  TwoKernelSharedFixtureOptions fixture;
  fixture.first_vgpr_granulated = 3u;
  fixture.second_vgpr_granulated = 3u;
  fixture.entry_nop_words = 1u;
  std::vector<uint8_t> bytes = make_two_kernel_shared_helper_code_object(fixture, kArch, helper);
  ASSERT_FALSE(bytes.empty());
  mutate_kernel_descriptor(bytes, "shared_owner_0", [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 4u);
  });
  mutate_kernel_descriptor(bytes, "shared_owner_1", [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 9u);
  });
  constexpr std::array<std::string_view, 1> kAdditionalOwners = {"shared_owner_1"};
  append_kernel_metadata_note(bytes, "shared_owner_0", /*uses_dynamic_stack=*/true,
                              /*sgpr_count=*/0u, std::nullopt, std::nullopt,
                              /*has_dynamic_lds=*/false, kAdditionalOwners);

  // The per-owner context path is shared by the Sampled/Inline dynamic-stack
  // fallback and Record/Replay's scalar tail. InlineShadow's scalar entry and
  // emission path is covered by
  // Cdna4InlineScalarPersistencePlansEntryScratchForEveryComponent in
  // moi_inline_shadow_test.cpp.
  for (ConSanMoiEngine engine : {ConSanMoiEngine::Sampled, ConSanMoiEngine::RecordReplay}) {
    SCOPED_TRACE(testing::PrintToString(engine));
    ConSanOptions options = moi_options(engine);
    options.force_vgpr_spill = true;
    options.moi_runtime_sample_stride = 2u;
    options.moi_track_barriers = false;
    // Record/Replay's persistent access epoch exercises its scalar tail
    // without relying on an unrelated atomic-tracking request.
    options.moi_track_atomics = true;
    options.moi_init_owner_epoch = engine == ConSanMoiEngine::RecordReplay;
    options.moi_report_buffer_address = 0x123456780000ull;
    options.moi_report_buffer_size =
        engine == ConSanMoiEngine::Sampled ? direct_sampled_report_bytes(2) : 64u * 1024u * 1024u;
    options.max_patches = 1u;

    const ConSanResult result = try_patch_consan(bytes, options);

    ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
    ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
    ASSERT_TRUE(result.resolved_moi_persistent_owner_sgpr)
        << testing::PrintToString(result.warnings);
    ASSERT_TRUE(result.resolved_moi_persistent_epoch_sgpr);
    EXPECT_TRUE(result.moi_persistent_sgprs_automatic);
    EXPECT_EQ(*result.resolved_moi_persistent_owner_sgpr, 80u);
    if (engine == ConSanMoiEngine::RecordReplay) {
      EXPECT_FALSE(result.resolved_moi_persistent_workgroup_key_sgpr);
    }
    const auto access_patch =
        std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
          return patch.phase == ConSanPatchPhase::Instrumentation &&
                 patch.owner_descriptor_file_offsets.size() == 2u;
        });
    ASSERT_NE(access_patch, result.patches.end());
    EXPECT_TRUE(result.final_validation_passed);
  }
}

TEST(ConSanMoi, SharedHelperAtomicUsesCommonOwnerResourcePlan) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.helper_has_ordered_atomic = true;
  const std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object(fixture);
  ASSERT_FALSE(bytes.empty());
  ConSanOptions options = moi_options();
  options.moi_track_atomics = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 0, 1, 1);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.functions.size(), 1u);
  ASSERT_EQ(result.functions.front().atomic_sites.size(), 1u);
  ASSERT_EQ(result.resource_plans.size(), 2u);
  const auto plan_it = std::ranges::find_if(result.resource_plans, [](const auto &item) {
    return item.site_kind == ConSanResourceSiteKind::Atomic;
  });
  ASSERT_NE(plan_it, result.resource_plans.end());
  const ConSanCandidateResourcePlan &plan = *plan_it;
  EXPECT_EQ(plan.site_kind, ConSanResourceSiteKind::Atomic);
  EXPECT_EQ(plan.source, ConSanRegisterAllocationSource::LivenessDead);
  ASSERT_EQ(plan.owner_descriptor_file_offsets.size(), 2u);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_FALSE(result.moi_persistent_vgprs_automatic);
  EXPECT_FALSE(result.resolved_moi_owner_vgpr);
  EXPECT_FALSE(result.resolved_moi_epoch_vgpr);
  EXPECT_TRUE(result.moi_exec_save_sgprs_automatic);
  ASSERT_TRUE(result.resolved_moi_exec_save_sgpr);
  const auto atomic_patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiAtomicRecord;
  });
  ASSERT_NE(atomic_patch, result.patches.end());
  EXPECT_EQ(atomic_patch->owner_descriptor_file_offsets, plan.owner_descriptor_file_offsets);
  EXPECT_EQ(std::count_if(result.patches.begin(), result.patches.end(),
                          [](const ConSanPatchInfo &patch) {
                            return patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue;
                          }),
            0);
  EXPECT_NE(
      std::ranges::find(
          result.warnings,
          "ConSan MOI record/replay uses probe-local owner derivation without access records"),
      result.warnings.end());
}

TEST(ConSanMoi, SharedHelperAtomicSpillUsesOneLayoutForEveryOwner) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.helper_has_ordered_atomic = true;
  fixture.first_private_bytes = 0;
  fixture.second_private_bytes = 20;
  const std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object(fixture);
  ASSERT_FALSE(bytes.empty());
  ConSanOptions options = moi_options();
  options.moi_track_atomics = true;
  options.force_vgpr_spill = true;
  options.moi_owner_vgpr = 10;
  options.moi_epoch_vgpr = 11;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 0, 1, 1);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.resource_plans.size(), 2u);
  const auto plan_it = std::ranges::find_if(result.resource_plans, [](const auto &item) {
    return item.site_kind == ConSanResourceSiteKind::Atomic;
  });
  ASSERT_NE(plan_it, result.resource_plans.end());
  const ConSanCandidateResourcePlan &plan = *plan_it;
  EXPECT_EQ(plan.site_kind, ConSanResourceSiteKind::Atomic);
  EXPECT_EQ(plan.source, ConSanRegisterAllocationSource::SpillRequired);
  EXPECT_EQ(plan.original_private_segment_size, 20u);
  ASSERT_EQ(plan.owner_descriptor_file_offsets.size(), 2u);
  const auto patch_it = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiAtomicRecord;
  });
  ASSERT_NE(patch_it, result.patches.end());
  const ConSanPatchInfo &patch = *patch_it;
  EXPECT_EQ(patch.kind, ConSanPatchKind::TrampolineMoiAtomicRecord);
  EXPECT_EQ(patch.spilled_vgpr_count, 7u);
  EXPECT_EQ(patch.required_private_segment_size, 60u);
  EXPECT_EQ(patch.owner_descriptor_file_offsets, plan.owner_descriptor_file_offsets);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiFenceRecord,
                               &ConSanPatchInfo::kind),
            1u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  for (const AmdGpuKernelInfo &kernel : patched.kernels()) {
    KD descriptor{};
    std::memcpy(&descriptor, result.elf_bytes.data() + kernel.descriptor_file_offset,
                sizeof(descriptor));
    if (kernel.name == "shared_owner_0" || kernel.name == "shared_owner_1") {
      EXPECT_EQ(descriptor.private_segment_fixed_size, 76u);
    } else if (kernel.name == "unrelated_kernel") {
      EXPECT_EQ(descriptor.private_segment_fixed_size, 0u);
    }
  }
}

TEST(ConSanMoi, SharedHelperPatchNamesEveryOwnerAndLeavesUnrelatedDescriptorUnchanged) {
  const std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object();
  AmdGpuCodeObject original(bytes.data(), bytes.size());
  ASSERT_TRUE(original.is_valid());
  const auto original_unrelated =
      std::ranges::find_if(original.kernels(), [](const AmdGpuKernelInfo &kernel) {
        return kernel.name == "unrelated_kernel";
      });
  ASSERT_NE(original_unrelated, original.kernels().end());
  KD original_unrelated_descriptor{};
  std::memcpy(&original_unrelated_descriptor,
              bytes.data() + original_unrelated->descriptor_file_offset,
              sizeof(original_unrelated_descriptor));

  ConSanOptions options = moi_options();
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);
  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  const ConSanPatchInfo &patch = result.patches.front();
  EXPECT_EQ(patch.kind, ConSanPatchKind::TrampolineMoiAccessRecordStore);
  EXPECT_EQ(patch.anchor_offset, 20u);
  ASSERT_EQ(patch.owner_descriptor_file_offsets.size(), 2u);
  EXPECT_EQ(patch.owner_descriptor_file_offsets,
            result.resource_plans.front().owner_descriptor_file_offsets);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const auto patched_unrelated =
      std::ranges::find_if(patched.kernels(), [](const AmdGpuKernelInfo &kernel) {
        return kernel.name == "unrelated_kernel";
      });
  ASSERT_NE(patched_unrelated, patched.kernels().end());
  KD patched_unrelated_descriptor{};
  std::memcpy(&patched_unrelated_descriptor,
              result.elf_bytes.data() + patched_unrelated->descriptor_file_offset,
              sizeof(patched_unrelated_descriptor));
  // Text growth legitimately adjusts KD-relative entry offsets. Resource and
  // ABI fields for a kernel that cannot reach the helper stay unchanged.
  EXPECT_EQ(patched_unrelated_descriptor.compute_pgm_rsrc1,
            original_unrelated_descriptor.compute_pgm_rsrc1);
  EXPECT_EQ(patched_unrelated_descriptor.compute_pgm_rsrc2,
            original_unrelated_descriptor.compute_pgm_rsrc2);
  EXPECT_EQ(patched_unrelated_descriptor.private_segment_fixed_size,
            original_unrelated_descriptor.private_segment_fixed_size);
  EXPECT_EQ(patched_unrelated_descriptor.kernel_code_properties,
            original_unrelated_descriptor.kernel_code_properties);
}

TEST(ConSanMoi, SharedHelperPlanGrowsEveryOwnerForOneFreshWindow) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.first_vgpr_granulated = 0;
  fixture.second_vgpr_granulated = 0;
  fixture.helper_keeps_v1_v3_live = true;
  const std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object(fixture);
  ConSanOptions options = moi_options();
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  const ConSanCandidateResourcePlan &plan = result.resource_plans.front();
  EXPECT_EQ(plan.source, ConSanRegisterAllocationSource::DescriptorGrowth);
  EXPECT_EQ(plan.scratch_vgpr, 4);
  EXPECT_EQ(plan.required_vgpr_count, 10u);
  ASSERT_EQ(plan.owner_descriptor_file_offsets.size(), 2u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  for (const AmdGpuKernelInfo &kernel : patched.kernels()) {
    KD descriptor{};
    std::memcpy(&descriptor, result.elf_bytes.data() + kernel.descriptor_file_offset,
                sizeof(descriptor));
    const uint32_t granulated = AMDHSA_BITS_GET(
        descriptor.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
    if (kernel.name == "shared_owner_0" || kernel.name == "shared_owner_1") {
      EXPECT_EQ(granulated, 2u);
    } else if (kernel.name == "unrelated_kernel") {
      EXPECT_EQ(granulated, kRdna4Wave64AllVgprsGranulated);
    }
  }
}

TEST(ConSanMoi, SharedHelperSpillUsesOneLayoutAndGrowsEveryOwner) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.first_private_bytes = 0;
  fixture.second_private_bytes = 20;
  const std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object(fixture);
  ConSanOptions options = moi_options();
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().source, ConSanRegisterAllocationSource::SpillRequired);
  EXPECT_EQ(result.resource_plans.front().original_private_segment_size, 20u);
  ASSERT_EQ(result.patches.size(), 1u);
  const ConSanPatchInfo &patch = result.patches.front();
  EXPECT_EQ(patch.spilled_vgpr_count, 6u);
  EXPECT_EQ(patch.required_private_segment_size, 56u);
  ASSERT_EQ(patch.owner_descriptor_file_offsets.size(), 2u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const std::vector<uint32_t> expected_save =
      expected_vgpr_spill_words(/*base=*/1, /*count=*/6, /*restore=*/false, /*slot_base=*/32);
  ASSERT_FALSE(expected_save.empty());
  const std::vector<uint32_t> trampoline_words =
      text_words_at_offset(patched, patch.trampoline_offset, patch.trampoline_size);
  ASSERT_GE(trampoline_words.size(), expected_save.size());
  EXPECT_TRUE(std::equal(expected_save.begin(), expected_save.end(), trampoline_words.begin()));

  for (const AmdGpuKernelInfo &kernel : patched.kernels()) {
    KD descriptor{};
    std::memcpy(&descriptor, result.elf_bytes.data() + kernel.descriptor_file_offset,
                sizeof(descriptor));
    if (kernel.name == "shared_owner_0" || kernel.name == "shared_owner_1") {
      EXPECT_EQ(descriptor.private_segment_fixed_size, 56u);
    } else if (kernel.name == "unrelated_kernel") {
      EXPECT_EQ(descriptor.private_segment_fixed_size, 0u);
    }
  }
}

TEST(ConSanMoi, IndirectSharedHelperSpillUsesEveryRecoveredOwner) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.use_indirect_calls = true;
  const std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object(fixture);
  ConSanOptions options = moi_options();
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  const ConSanCandidateResourcePlan &plan = result.resource_plans.front();
  EXPECT_EQ(plan.source, ConSanRegisterAllocationSource::SpillRequired);
  EXPECT_EQ(plan.reason, ConSanRegisterPlanReason::None);
  EXPECT_EQ(plan.owner_descriptor_file_offsets.size(), 2u);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().owner_descriptor_file_offsets,
            plan.owner_descriptor_file_offsets);
}

TEST(ConSanMoi, ScopedSpillPlanningExcludesUnselectedFullVgprCandidate) {
  TwoKernelSharedFixtureOptions fixture;
  // Keep the selected owners large enough for the six-VGPR static
  // Record/Replay spill window. The unrelated full-VGPR candidate remains the
  // part this test proves is excluded from scoped planning.
  fixture.first_vgpr_granulated = 1;
  fixture.second_vgpr_granulated = 1;
  fixture.first_private_bytes = 20;
  fixture.unrelated_has_lds = true;
  const std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object(fixture);
  ConSanOptions options = moi_options();
  options.force_vgpr_spill = true;
  options.test_kernel_name_filter = "shared_lds_helper";
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().required_private_segment_size, 56u);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().text_offset, result.patches.front().anchor_offset);
  EXPECT_LT(result.resource_plans.front().required_vgpr_count, 256u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  for (const AmdGpuKernelInfo &kernel : patched.kernels()) {
    KD descriptor{};
    std::memcpy(&descriptor, result.elf_bytes.data() + kernel.descriptor_file_offset,
                sizeof(descriptor));
    if (kernel.name == "shared_owner_0" || kernel.name == "shared_owner_1") {
      EXPECT_EQ(descriptor.private_segment_fixed_size, 56u);
    } else if (kernel.name == "unrelated_kernel") {
      EXPECT_EQ(descriptor.private_segment_fixed_size, 0u);
    }
  }
}

TEST(ConSanMoi, SharedHelperRejectsAssignmentLiveInAnyOwnerScope) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.first_continuation_uses_v1 = true;
  const std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object(fixture);
  ConSanOptions options = moi_options();
  options.scratch_vgpr = 1;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  EXPECT_TRUE(result.errors.empty());
  EXPECT_FALSE(result.modified);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().source, ConSanRegisterAllocationSource::Unsupported);
  EXPECT_EQ(result.resource_plans.front().reason, ConSanRegisterPlanReason::ExplicitLive);
  EXPECT_EQ(result.resource_plans.front().owner_descriptor_file_offsets.size(), 2u);
}

TEST(ConSanMoi, SharedPrivateOwnerRejectsIncompatibleWaveSizes) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.first_wave32 = true;
  fixture.second_wave32 = false;
  const std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object(fixture);
  AmdGpuCodeObject original(bytes.data(), bytes.size());
  ASSERT_TRUE(original.is_valid());
  for (const AmdGpuKernelInfo &kernel : original.kernels()) {
    KD descriptor{};
    std::memcpy(&descriptor, bytes.data() + kernel.descriptor_file_offset, sizeof(descriptor));
    if (kernel.name == "shared_owner_0") {
      EXPECT_EQ(AMDHSA_BITS_GET(descriptor.kernel_code_properties,
                                kd::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32),
                1u);
    } else if (kernel.name == "shared_owner_1") {
      EXPECT_EQ(AMDHSA_BITS_GET(descriptor.kernel_code_properties,
                                kd::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32),
                0u);
    }
  }
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.force_private_epoch = true;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const auto result = try_patch_consan(bytes, options);

  EXPECT_TRUE(result.errors.empty());
  EXPECT_FALSE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().owner_descriptor_file_offsets.size(), 2u);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("incompatible owner wave sizes") != std::string::npos;
  }));
}

TEST(ConSanMoi, InventorySkipsUnknownFlatSites) {
  const std::vector<uint8_t> bytes = make_rdna4_flat_memory_code_object();
  ConSanOptions options = moi_options();

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.kernels.front().flat_sites.size(), 2u);
  EXPECT_TRUE(result.moi_candidates.empty());

  bool saw_skip_warning = false;
  for (const std::string &warning : result.warnings) {
    saw_skip_warning |= warning.find("skipped flat sites") != std::string::npos &&
                        warning.find("unknown=2") != std::string::npos;
  }
  EXPECT_TRUE(saw_skip_warning);
}

TEST(ConSanMoi, DispatchIdPreloadPlanPreservesShiftedGuestSgprs) {
  EXPECT_EQ(consan_moi_amdhsa_dispatch_id_prefix_sgpr_count(
                /*private_segment_buffer=*/true, /*dispatch_ptr=*/true,
                /*queue_ptr=*/true, /*kernarg_segment_ptr=*/true),
            10u);
  const auto plan = consan_moi_plan_dispatch_id_preload(
      /*original_user_sgpr_count=*/12, /*system_sgpr_count=*/3,
      /*dispatch_id_prefix_sgpr_count=*/10, /*dispatch_id_already_enabled=*/false);
  ASSERT_TRUE(plan.supported());
  EXPECT_TRUE(plan.descriptor_change_required());
  EXPECT_EQ(plan.support, ConSanMoiDispatchIdPreloadSupport::SupportedInsert);
  EXPECT_EQ(plan.dispatch_id_sgpr, 10u);
  EXPECT_EQ(plan.expanded_user_sgpr_count, 14u);
  EXPECT_EQ(plan.first_shifted_guest_sgpr, 10u);
  EXPECT_EQ(plan.shifted_guest_sgpr_count, 5u);
  EXPECT_EQ(plan.required_sgpr_count, 17u);

  std::array<uint32_t, 17> expanded{};
  for (uint16_t guest_sgpr = 0; guest_sgpr < 15; ++guest_sgpr) {
    const auto source = consan_moi_dispatch_id_restore_source(plan, guest_sgpr);
    expanded[source.value_or(guest_sgpr)] = 0x1000u + guest_sgpr;
  }
  expanded[plan.dispatch_id_sgpr] = 0xD15A7C01u;
  expanded[plan.dispatch_id_sgpr + 1u] = 0xD15A7C02u;
  const std::array<uint32_t, 2> captured_dispatch = {expanded[plan.dispatch_id_sgpr],
                                                     expanded[plan.dispatch_id_sgpr + 1u]};
  for (uint16_t destination = plan.first_shifted_guest_sgpr;
       destination < plan.first_shifted_guest_sgpr + plan.shifted_guest_sgpr_count; ++destination) {
    const auto source = consan_moi_dispatch_id_restore_source(plan, destination);
    ASSERT_TRUE(source.has_value());
    expanded[destination] = expanded[*source];
  }
  EXPECT_EQ(captured_dispatch[0], 0xD15A7C01u);
  EXPECT_EQ(captured_dispatch[1], 0xD15A7C02u);
  for (uint16_t guest_sgpr = 0; guest_sgpr < 15; ++guest_sgpr)
    EXPECT_EQ(expanded[guest_sgpr], 0x1000u + guest_sgpr);
}

TEST(ConSanMoi, DispatchIdPreloadPlanRejectsTruncationAndInvalidLayouts) {
  const auto existing = consan_moi_plan_dispatch_id_preload(
      /*original_user_sgpr_count=*/14, /*system_sgpr_count=*/4,
      /*dispatch_id_prefix_sgpr_count=*/8, /*dispatch_id_already_enabled=*/true);
  ASSERT_TRUE(existing.supported());
  EXPECT_EQ(existing.support, ConSanMoiDispatchIdPreloadSupport::SupportedAlreadyEnabled);
  EXPECT_FALSE(existing.descriptor_change_required());
  EXPECT_FALSE(consan_moi_dispatch_id_restore_source(existing, 8).has_value());

  EXPECT_EQ(consan_moi_plan_dispatch_id_preload(15, 1, 10, false).support,
            ConSanMoiDispatchIdPreloadSupport::UserSgprInitializationLimit);
  EXPECT_EQ(consan_moi_plan_dispatch_id_preload(8, 1, 9, false).support,
            ConSanMoiDispatchIdPreloadSupport::InvalidDispatchPosition);
  EXPECT_EQ(consan_moi_plan_dispatch_id_preload(14, 91, 10, false).support,
            ConSanMoiDispatchIdPreloadSupport::SgprAllocationLimit);
  EXPECT_EQ(consan_moi_plan_dispatch_id_preload(17, 0, 10, true).support,
            ConSanMoiDispatchIdPreloadSupport::UserSgprInitializationLimit);
}

TEST(ConSanMoi, DispatchIdPreloadPlanSupportsGfx1250UserSgprLimit) {
  const auto plan = consan_moi_plan_dispatch_id_preload(
      /*original_user_sgpr_count=*/29, /*system_sgpr_count=*/1,
      /*dispatch_id_prefix_sgpr_count=*/2, /*dispatch_id_already_enabled=*/false,
      /*sgpr_limit=*/106, /*user_sgpr_initialization_limit=*/32);
  ASSERT_TRUE(plan.supported());
  EXPECT_EQ(plan.support, ConSanMoiDispatchIdPreloadSupport::SupportedInsert);
  EXPECT_EQ(plan.expanded_user_sgpr_count, 31u);
  EXPECT_EQ(plan.shifted_guest_sgpr_count, 28u);
  EXPECT_EQ(plan.required_sgpr_count, 32u);
}

TEST(ConSanMoi, DispatchPreloadDescriptorPermutationsUseExactAmdhsaPrefix) {
  for (uint32_t mask = 0; mask < 16u; ++mask) {
    std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
    const uint16_t prefix =
        consan_moi_amdhsa_dispatch_id_prefix_sgpr_count(mask & 1u, mask & 2u, mask & 4u, mask & 8u);
    mutate_first_kernel_descriptor(bytes, [&](KD &descriptor) {
      AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                      kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER, (mask & 1u));
      AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                      kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR, ((mask >> 1u) & 1u));
      AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                      kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR, ((mask >> 2u) & 1u));
      AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                      kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR,
                      ((mask >> 3u) & 1u));
      AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                      kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_FLAT_SCRATCH_INIT, 1u);
      AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT,
                      (prefix + 2u));
      AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2,
                      kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X, 1u);
      AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2,
                      kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_INFO, 1u);
    });

    ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
    options.moi_report_buffer_address = 0x100000000ull;
    options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
    const ConSanResult result = try_patch_consan(bytes, options);

    ASSERT_TRUE(result.errors.empty())
        << "mask=" << mask << " " << (result.errors.empty() ? "" : result.errors.front());
    ASSERT_TRUE(result.modified) << "mask=" << mask;
    const auto prologue = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
      return patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue;
    });
    ASSERT_NE(prologue, result.patches.end());
    ASSERT_TRUE(prologue->dispatch_id_capture_sgpr);
    EXPECT_EQ(prologue->dispatch_id_source_sgpr, prefix);
    EXPECT_EQ(prologue->dispatch_id_original_user_sgpr_count, prefix + 2u);
    EXPECT_EQ(prologue->dispatch_id_expanded_user_sgpr_count, prefix + 4u);
    EXPECT_EQ(prologue->dispatch_id_system_sgpr_count, 2u);
    EXPECT_TRUE(prologue->dispatch_id_preload_inserted);

    AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(patched.is_valid());
    KD descriptor{};
    std::memcpy(&descriptor,
                result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
                sizeof(descriptor));
    EXPECT_EQ(AMDHSA_BITS_GET(descriptor.kernel_code_properties,
                              kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID),
              1u);
    EXPECT_EQ(AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT),
              prefix + 4u);
  }
}

TEST(ConSanMoi, DispatchPrologueCapturesBeforeAscendingRestoreAtBothKernargEntries) {
  std::vector<uint32_t> text_words(80u, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000000u; // ds_store_b32 v0, v0
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);
  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                    kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER, 1u);
    AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                    kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR, 1u);
    AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                    kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR, 1u);
    AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                    kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 1u);
    AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                    kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_FLAT_SCRATCH_INIT, 1u);
    AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                    kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE, 1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 14u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_INFO,
                    1u);
    AMDHSA_BITS_SET(descriptor.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_LENGTH, 1u);
    AMDHSA_BITS_SET(descriptor.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_OFFSET, 3u);
  });

  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified);
  const auto prologue = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue;
  });
  ASSERT_NE(prologue, result.patches.end());
  ASSERT_TRUE(prologue->dispatch_id_capture_sgpr);
  EXPECT_EQ(prologue->dispatch_id_source_sgpr, 10u);
  EXPECT_EQ(prologue->dispatch_id_original_user_sgpr_count, 14u);
  EXPECT_EQ(prologue->dispatch_id_expanded_user_sgpr_count, 16u);
  EXPECT_EQ(prologue->dispatch_id_system_sgpr_count, 4u);
  EXPECT_GE(prologue->trampoline_size, 256u + 20u * sizeof(uint32_t));

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  KD descriptor{};
  std::memcpy(&descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  EXPECT_EQ(AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT),
            16u);
  EXPECT_EQ(AMDHSA_BITS_GET(descriptor.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_LENGTH), 1u);
  EXPECT_EQ(AMDHSA_BITS_GET(descriptor.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_OFFSET), 3u);

  const auto verify_entry = [&](uint64_t entry_offset) {
    const char *text = patched.text_sections().front()->data();
    uint64_t cursor = entry_offset;
    const auto expect_write = [&](uint32_t expected) {
      uint32_t word = 0;
      std::memcpy(&word, text + cursor, sizeof(word));
      EXPECT_EQ(word, expected);
      cursor += sizeof(word);
      std::memcpy(&word, text + cursor, sizeof(word));
      EXPECT_EQ(word, build_s_delay_alu(kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4));
      cursor += sizeof(word);
    };
    const uint16_t persistent = *prologue->dispatch_id_capture_sgpr;
    expect_write(build_s_mov_b32(persistent, 10u, ROCJITSU_CODE_ARCH_RDNA4));
    expect_write(
        build_s_mov_b32(static_cast<uint16_t>(persistent + 1u), 11u, ROCJITSU_CODE_ARCH_RDNA4));
    expect_write(build_s_add_u32(persistent, persistent, scalar_positive_inline_u32(1),
                                 ROCJITSU_CODE_ARCH_RDNA4));
    expect_write(build_s_addc_u32(static_cast<uint16_t>(persistent + 1u),
                                  static_cast<uint16_t>(persistent + 1u),
                                  scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4));
    for (uint16_t destination = 10u; destination < 18u; ++destination) {
      expect_write(build_s_mov_b32(destination, static_cast<uint16_t>(destination + 2u),
                                   ROCJITSU_CODE_ARCH_RDNA4));
    }
  };
  verify_entry(prologue->trampoline_offset);
  verify_entry(prologue->trampoline_offset + 256u);
}

TEST(ConSanMoi, AlreadyEnabledDispatchPreloadIsCapturedWithoutGuestShuffle) {
  std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                    kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 1u);
    AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                    kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID, 1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 4u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X,
                    1u);
  });
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified);
  const auto prologue = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.dispatch_id_capture_sgpr.has_value();
  });
  ASSERT_NE(prologue, result.patches.end());
  EXPECT_EQ(prologue->dispatch_id_source_sgpr, 2u);
  EXPECT_EQ(prologue->dispatch_id_original_user_sgpr_count, 4u);
  EXPECT_EQ(prologue->dispatch_id_expanded_user_sgpr_count, 4u);
  EXPECT_FALSE(prologue->dispatch_id_preload_inserted);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  KD descriptor{};
  std::memcpy(&descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  EXPECT_EQ(AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT),
            4u);
  uint32_t first_owner_word = 0;
  std::memcpy(&first_owner_word,
              patched.text_sections().front()->data() + prologue->trampoline_offset +
                  8u * sizeof(uint32_t),
              sizeof(first_owner_word));
  const auto owner_init = build_v_lshrrev_b32_e32(
      *result.resolved_moi_owner_vgpr, scalar_positive_inline_u32(6), 0, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(owner_init);
  EXPECT_EQ(first_owner_word, *owner_init);
}

TEST(ConSanMoi, SharedHelperDispatchCaptureUsesPerKernelLayoutsAndOnePersistentPair) {
  auto make_fixture = [] {
    std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object();
    mutate_kernel_descriptor(bytes, "shared_owner_0", [](KD &descriptor) {
      AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                      kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 1u);
      AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 2u);
      AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2,
                      kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X, 1u);
    });
    mutate_kernel_descriptor(bytes, "shared_owner_1", [](KD &descriptor) {
      AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                      kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER, 1u);
      AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                      kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR, 1u);
      AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                      kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_FLAT_SCRATCH_INIT, 1u);
      AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 8u);
      AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2,
                      kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X, 1u);
    });
    return bytes;
  };
  std::vector<uint8_t> bytes = make_fixture();
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  std::vector<const ConSanPatchInfo *> prologues;
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.dispatch_id_capture_sgpr)
      prologues.push_back(&patch);
  }
  ASSERT_EQ(prologues.size(), 2u);
  ASSERT_TRUE(result.resolved_moi_dispatch_id_sgpr);
  EXPECT_EQ(prologues[0]->dispatch_id_capture_sgpr, result.resolved_moi_dispatch_id_sgpr);
  EXPECT_EQ(prologues[1]->dispatch_id_capture_sgpr, result.resolved_moi_dispatch_id_sgpr);
  std::array<uint16_t, 2> sources = {prologues[0]->dispatch_id_source_sgpr,
                                     prologues[1]->dispatch_id_source_sgpr};
  std::ranges::sort(sources);
  EXPECT_EQ(sources, (std::array<uint16_t, 2>{2u, 6u}));

  AmdGpuCodeObject original(bytes.data(), bytes.size());
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(original.is_valid());
  ASSERT_TRUE(patched.is_valid());
  const auto original_unrelated =
      std::ranges::find(original.kernels(), "unrelated_kernel", &AmdGpuKernelInfo::name);
  const auto patched_unrelated =
      std::ranges::find(patched.kernels(), "unrelated_kernel", &AmdGpuKernelInfo::name);
  ASSERT_NE(original_unrelated, original.kernels().end());
  ASSERT_NE(patched_unrelated, patched.kernels().end());
  KD original_descriptor{};
  KD patched_descriptor{};
  std::memcpy(&original_descriptor, bytes.data() + original_unrelated->descriptor_file_offset,
              sizeof(KD));
  std::memcpy(&patched_descriptor,
              result.elf_bytes.data() + patched_unrelated->descriptor_file_offset, sizeof(KD));
  EXPECT_EQ(patched_descriptor.compute_pgm_rsrc1, original_descriptor.compute_pgm_rsrc1);
  EXPECT_EQ(patched_descriptor.compute_pgm_rsrc2, original_descriptor.compute_pgm_rsrc2);
  EXPECT_EQ(patched_descriptor.kernel_code_properties, original_descriptor.kernel_code_properties);
  EXPECT_EQ(patched_descriptor.kernarg_preload, original_descriptor.kernarg_preload);
  EXPECT_EQ(patched_descriptor.private_segment_fixed_size,
            original_descriptor.private_segment_fixed_size);

  std::vector<uint8_t> rejected_bytes = make_fixture();
  mutate_kernel_descriptor(rejected_bytes, "shared_owner_1", [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 15u);
  });
  const ConSanResult rejected = try_patch_consan(rejected_bytes, options);
  EXPECT_EQ(rejected.outcome, ConSanTransformOutcome::Unsupported);
  EXPECT_FALSE(rejected.modified);
  EXPECT_TRUE(rejected.elf_bytes.empty());
  EXPECT_TRUE(rejected.patches.empty());
}

TEST(ConSanMoi, DispatchPreloadUnsupportedLayoutsRollbackTransactionally) {
  const auto run = [](const auto &mutator, std::optional<uint16_t> explicit_pair = std::nullopt) {
    std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
    mutate_first_kernel_descriptor(bytes, mutator);
    ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
    options.moi_dispatch_id_sgpr = explicit_pair;
    options.moi_report_buffer_address = 0x100000000ull;
    options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
    return try_patch_consan(bytes, options);
  };

  const ConSanResult user_limit = run([](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 15u);
  });
  EXPECT_EQ(user_limit.outcome, ConSanTransformOutcome::Unsupported);
  EXPECT_FALSE(user_limit.modified);
  EXPECT_TRUE(user_limit.elf_bytes.empty());
  EXPECT_TRUE(user_limit.patches.empty());

  const ConSanResult malformed_prefix = run([](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                    kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER, 1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 2u);
  });
  EXPECT_EQ(malformed_prefix.outcome, ConSanTransformOutcome::Unsupported);
  EXPECT_FALSE(malformed_prefix.modified);
  EXPECT_TRUE(malformed_prefix.elf_bytes.empty());
  EXPECT_TRUE(malformed_prefix.patches.empty());

  const ConSanResult invalid_pair = run([](KD &) {}, 105u);
  EXPECT_EQ(invalid_pair.outcome, ConSanTransformOutcome::Unsupported);
  EXPECT_FALSE(invalid_pair.modified);
  EXPECT_TRUE(invalid_pair.elf_bytes.empty());
  EXPECT_TRUE(invalid_pair.patches.empty());
}

TEST(ConSanMoi, FinalValidationPinsDispatchDescriptorAndCaptureSequence) {
  std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                    kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 2u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X,
                    1u);
  });
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  const ConSanResult valid = try_patch_consan(bytes, options);
  ASSERT_TRUE(valid.errors.empty()) << (valid.errors.empty() ? "" : valid.errors.front());
  ASSERT_TRUE(valid.modified);
  EXPECT_TRUE(validate_consan_modified_elf(bytes, valid).empty());

  ConSanResult descriptor_corruption = valid;
  AmdGpuCodeObject descriptor_object(descriptor_corruption.elf_bytes.data(),
                                     descriptor_corruption.elf_bytes.size());
  ASSERT_TRUE(descriptor_object.is_valid());
  const uint64_t descriptor_offset = descriptor_object.kernels().front().descriptor_file_offset;
  KD descriptor{};
  std::memcpy(&descriptor, descriptor_corruption.elf_bytes.data() + descriptor_offset,
              sizeof(descriptor));
  AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                  kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID, 0u);
  std::memcpy(descriptor_corruption.elf_bytes.data() + descriptor_offset, &descriptor,
              sizeof(descriptor));
  EXPECT_FALSE(validate_consan_modified_elf(bytes, descriptor_corruption).empty());

  ConSanResult capture_corruption = valid;
  const auto prologue =
      std::ranges::find_if(capture_corruption.patches, [](const ConSanPatchInfo &patch) {
        return patch.dispatch_id_capture_sgpr.has_value();
      });
  ASSERT_NE(prologue, capture_corruption.patches.end());
  const size_t capture_file_offset =
      capture_corruption.text_sections.front().file_offset + prologue->trampoline_offset;
  const uint32_t wrong_capture = build_s_mov_b32(
      *prologue->dispatch_id_capture_sgpr,
      static_cast<uint16_t>(prologue->dispatch_id_source_sgpr + 1u), ROCJITSU_CODE_ARCH_RDNA4);
  std::memcpy(capture_corruption.elf_bytes.data() + capture_file_offset, &wrong_capture,
              sizeof(wrong_capture));
  EXPECT_FALSE(validate_consan_modified_elf(bytes, capture_corruption).empty());
}

TEST(ConSanMoi, WarnsWhenReportBufferIsSmallerThanHeader) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  ConSanOptions options = moi_options();
  options.moi_report_buffer_address = 0x1000;
  options.moi_report_buffer_size = sizeof(ConSanMoiReportHeader) - 1;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  bool saw_small_buffer_warning = false;
  for (const std::string &warning : result.warnings)
    saw_small_buffer_warning |=
        warning.find("smaller than the report ABI header") != std::string::npos;
  EXPECT_TRUE(saw_small_buffer_warning);
}

TEST(ConSanMoi, InventorySkipsUnsupportedNativeLdsSites) {
  const std::vector<uint8_t> bytes = make_rdna4_unsupported_lds_code_object();
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.moi_candidates.size(), 2u);
  EXPECT_EQ(result.moi_candidates[0].mnemonic, "ds_store_b32");
  EXPECT_EQ(result.moi_candidates[1].mnemonic, "ds_load_b32");
  bool saw_skipped_lds_warning = false;
  for (const std::string &warning : result.warnings) {
    saw_skipped_lds_warning |= warning.find("skipped native LDS sites") != std::string::npos &&
                               warning.find("unsupported_kind=1") != std::string::npos &&
                               warning.find("unsupported_mnemonic=0") != std::string::npos;
  }
  EXPECT_TRUE(saw_skipped_lds_warning);
}

TEST(ConSanMoi, Gfx1201MoiEnginesAdmitNativeB96Accesses) {
  constexpr auto store =
      rdna4::build_vds(rdna4::kDsStoreB96Vds, {.offset0 = 12, .addr = 0, .data0 = 1});
  constexpr auto load =
      rdna4::build_vds(rdna4::kDsLoadB96Vds, {.offset0 = 12, .addr = 0, .vdst = 4});
  constexpr auto aliasing_load =
      rdna4::build_vds(rdna4::kDsLoadB96Vds, {.offset0 = 12, .addr = 0, .vdst = 0});
  constexpr std::array<NativeB96Access, 3> accesses = {
      NativeB96Access{store, "ds_store_b96", false},
      NativeB96Access{load, "ds_load_b96", false},
      NativeB96Access{aliasing_load, "ds_load_b96", true},
  };

  expect_moi_engines_admit_native_b96_accesses(
      ROCJITSU_CODE_ARCH_RDNA4, accesses, [](const auto &text_words) {
        return make_rdna4_lds_code_object(text_words, "native_b96_access");
      });
}

TEST(ConSanMoi, InventoryUsesTypedArchNotDisplayName) {
  constexpr std::array<uint32_t, 3> text_words = {0xDB78000Cu,
                                                  0x00000100u, // ds_store_b96 v0, v[1:3] offset:12
                                                  build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4)};
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words, "typed_arch_inventory");
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_track_atomics = false;
  options.moi_track_barriers = false;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = 64u * 1024u * 1024u;
  ConSanResult result = try_patch_consan(bytes, options);
  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_EQ(result.arch, ROCJITSU_CODE_ARCH_RDNA4);

  result.arch_display_name = "stale-display-name";
  EXPECT_EQ(inventory_consan_moi_auto_report(result, options, bytes).access_range_count, 1u);
}

TEST(ConSanMoi, NativeB96CapabilityMatchesArchitectureBoundary) {
  for (std::string_view mnemonic : {"ds_load_b96", "ds_store_b96"}) {
    SCOPED_TRACE(mnemonic);
    EXPECT_TRUE(consan_moi_supports_native_lds_mnemonic(mnemonic, ROCJITSU_CODE_ARCH_RDNA4));
    EXPECT_TRUE(consan_moi_supports_native_lds_mnemonic(mnemonic, ROCJITSU_CODE_ARCH_GFX1250));
    // RDNA3 and RDNA3.5 hardware encode B96, but they are intentionally
    // outside ConSan's supported target set.
    EXPECT_FALSE(consan_moi_supports_native_lds_mnemonic(mnemonic, ROCJITSU_CODE_ARCH_RDNA3));
    EXPECT_FALSE(consan_moi_supports_native_lds_mnemonic(mnemonic, ROCJITSU_CODE_ARCH_RDNA3_5));
    EXPECT_FALSE(consan_moi_supports_native_lds_mnemonic(mnemonic, ROCJITSU_CODE_ARCH_CDNA3));
    EXPECT_FALSE(consan_moi_supports_native_lds_mnemonic(mnemonic, ROCJITSU_CODE_ARCH_CDNA4));
  }
}

TEST(ConSanMoi, CdnaNativeB96AccessesRemainUnsupportedAtTransformBoundary) {
  constexpr auto cdna3_store = cdna3::build_ds(cdna3::kDsWriteB96Ds, {.addr = 0, .data0 = 1});
  constexpr auto cdna3_load = cdna3::build_ds(cdna3::kDsReadB96Ds, {.addr = 0, .vdst = 1});
  constexpr auto cdna4_store = cdna4::build_ds(cdna4::kDsWriteB96Ds, {.addr = 0, .data0 = 1});
  constexpr auto cdna4_load = cdna4::build_ds(cdna4::kDsReadB96Ds, {.addr = 0, .vdst = 1});
  const std::array cases = {
      std::pair{make_cdna3_lds_code_object(std::array{cdna3_store[0], cdna3_store[1],
                                                      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA3)},
                                           "cdna3_b96"),
                std::string_view{"ds_write_b96"}},
      std::pair{make_cdna3_lds_code_object(std::array{cdna3_load[0], cdna3_load[1],
                                                      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA3)},
                                           "cdna3_b96"),
                std::string_view{"ds_read_b96"}},
      std::pair{make_cdna4_lds_code_object(std::array{cdna4_store[0], cdna4_store[1],
                                                      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4)},
                                           "cdna4_b96"),
                std::string_view{"ds_write_b96"}},
      std::pair{make_cdna4_lds_code_object(std::array{cdna4_load[0], cdna4_load[1],
                                                      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4)},
                                           "cdna4_b96"),
                std::string_view{"ds_read_b96"}},
  };

  for (const auto &[bytes, mnemonic] : cases) {
    SCOPED_TRACE(mnemonic);
    const ConSanResult result = try_patch_consan(bytes, moi_options(ConSanMoiEngine::RecordReplay));
    const auto disposition = std::ranges::find(result.site_dispositions, mnemonic,
                                               &ConSanSiteDispositionRecord::mnemonic);
    ASSERT_NE(disposition, result.site_dispositions.end());
    EXPECT_EQ(disposition->disposition, ConSanSiteDisposition::Unsupported);
    EXPECT_EQ(disposition->reason, ConSanSiteDispositionReason::UnsupportedMnemonic);
  }
}

TEST(ConSanMoi, UnsupportedOnlyAccessRemainsApplicableInPreFilterLedger) {
  const std::array<uint32_t, 3> text_words = {
      0xDAC40000u,
      0x00000000u, // ds_load_addtid_b32 (implicit address, unsupported by MOI)
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);

  for (ConSanMoiEngine engine :
       {ConSanMoiEngine::RecordReplay, ConSanMoiEngine::InlineShadow, ConSanMoiEngine::Sampled}) {
    SCOPED_TRACE(consan_moi_engine_name(engine));
    ConSanOptions options = moi_options();
    options.moi_engine = engine;
    const ConSanResult result = try_patch_consan(bytes, options);

    ASSERT_TRUE(consan_patch_succeeded(result));
    EXPECT_TRUE(result.moi_candidates.empty());
    ASSERT_EQ(result.site_dispositions.size(), 1u);
    const ConSanSiteDispositionRecord &site = result.site_dispositions.front();
    EXPECT_EQ(site.site_kind, ConSanResourceSiteKind::Access);
    EXPECT_EQ(site.disposition, ConSanSiteDisposition::Unsupported);
    EXPECT_EQ(site.reason, ConSanSiteDispositionReason::MissingAddressOperand);
    EXPECT_EQ(site.lowering_outcome, ConSanSiteLoweringOutcome::Unsupported);
    EXPECT_EQ(site.lowering_reason, ConSanSiteLoweringReason::SemanticUnsupported);
    EXPECT_EQ(site.mnemonic, "ds_load_addtid_b32");
    EXPECT_STREQ(consan_site_disposition_name(site.disposition), "unsupported");
    EXPECT_STREQ(consan_site_disposition_reason_name(site.reason), "missing_address_operand");
  }
}

TEST(ConSanMoi, MixedAccessLedgerRetainsSupportedAndUnsupportedFinalCodeSites) {
  const std::array<uint32_t, 5> text_words = {
      0xD8340000u, 0x00000102u, // ds_store_b32 v2, v1
      0xDAC40000u, 0x00000000u, // ds_load_addtid_b32
      0xBFB00000u,              // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options = moi_options();

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.site_dispositions.size(), 2u);
  EXPECT_EQ(result.site_dispositions[0].disposition, ConSanSiteDisposition::Supported);
  EXPECT_EQ(result.site_dispositions[0].reason, ConSanSiteDispositionReason::None);
  EXPECT_EQ(result.site_dispositions[0].mnemonic, "ds_store_b32");
  EXPECT_EQ(result.site_dispositions[0].lowering_outcome,
            ConSanSiteLoweringOutcome::PlacementOrLoweringFailed);
  EXPECT_EQ(result.site_dispositions[0].lowering_reason,
            ConSanSiteLoweringReason::InstrumentationPatchMissing);
  EXPECT_EQ(result.site_dispositions[1].disposition, ConSanSiteDisposition::Unsupported);
  EXPECT_EQ(result.site_dispositions[1].reason, ConSanSiteDispositionReason::MissingAddressOperand);
  EXPECT_EQ(result.site_dispositions[1].mnemonic, "ds_load_addtid_b32");
  EXPECT_EQ(result.site_dispositions[1].lowering_outcome, ConSanSiteLoweringOutcome::Unsupported);
}

TEST(ConSanMoi, AutoReportInventoryCountsAdmittedLogicalRangesBeforeAllocation) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  for (ConSanMoiEngine engine :
       {ConSanMoiEngine::RecordReplay, ConSanMoiEngine::Sampled, ConSanMoiEngine::InlineShadow}) {
    SCOPED_TRACE(consan_moi_engine_name(engine));
    ConSanOptions options = moi_options();
    options.moi_engine = engine;
    options.max_patches = 1u << 20u;
    options.moi_runtime_sample_stride = engine == ConSanMoiEngine::Sampled ? 256u : 1u;
    const ConSanResult result = try_patch_consan(bytes, options);

    ASSERT_TRUE(consan_patch_succeeded(result));
    const ConSanMoiAutoReportInventory inventory =
        inventory_consan_moi_auto_report(result, options, bytes);
    EXPECT_EQ(inventory.engine, engine);
    EXPECT_EQ(inventory.access_range_count, 2u);
    EXPECT_GE(inventory.diagnostic_count, 2u);
    if (engine == ConSanMoiEngine::Sampled) {
      EXPECT_EQ(inventory.sampled_range_bank_count, 16u);
      EXPECT_EQ(inventory.sampled_watchpoint_count, 16u);
    } else if (engine == ConSanMoiEngine::InlineShadow) {
      EXPECT_EQ(inventory.inline_lds_bytes, kConSanMoiInlineShadowConservativeExactShadowEntries *
                                                consan_moi_exact_shadow::granule_bytes);
      EXPECT_EQ(inventory.inline_atomic_release_count,
                kConSanMoiInlineShadowAtomicReleaseSlotCapacity);
      EXPECT_EQ(inventory.inline_causal_snapshot_count,
                kConSanMoiInlineShadowAtomicReleaseSlotCapacity);
      EXPECT_EQ(inventory.inline_acquired_epoch_token_count,
                kConSanMoiInlineShadowAcquiredEpochTokenSlotCapacity);
    }
    EXPECT_TRUE(plan_consan_moi_auto_report(inventory).complete());
  }
}

TEST(ConSanMoi, Gfx1250AutoReportUsesRuntimeApertureForDescriptorOpaqueLds) {
  constexpr uint32_t kRuntimeLdsBytes = 96u * 1024u;
  constexpr auto store = gfx1250::build_vds(gfx1250::kDsStoreB32Vds, {.addr = 0, .data0 = 1});
  const std::array<uint32_t, 3> text_words = {store[0], store[1],
                                              build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250)};
  const std::vector<uint8_t> bytes =
      make_gfx1250_code_object(text_words, "opaque_lds", /*vgpr_granulated=*/4);
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_max_workgroup_lds_bytes = kRuntimeLdsBytes;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  const ConSanMoiAutoReportInventory inventory =
      inventory_consan_moi_auto_report(result, options, bytes);
  EXPECT_EQ(inventory.inline_lds_bytes, kRuntimeLdsBytes);
  const ConSanMoiAutoReportPlan plan = plan_consan_moi_auto_report(inventory);
  ASSERT_TRUE(plan.complete());
  EXPECT_EQ(plan.layout.inline_exact_dispatch_bank_count,
            consan_moi_inline_exact_dispatch_bank_count_for_lds(inventory.inline_lds_bytes));
  EXPECT_EQ(plan.layout.exact_shadow_entry_capacity,
            kRuntimeLdsBytes / consan_moi_exact_shadow::granule_bytes *
                plan.layout.inline_exact_dispatch_bank_count);
}

TEST(ConSanMoi, Gfx1250AutoReportCoversFullApertureForDynamicLds) {
  constexpr auto store = gfx1250::build_vds(gfx1250::kDsStoreB32Vds, {.addr = 0, .data0 = 1});
  const std::array<uint32_t, 3> text_words = {store[0], store[1],
                                              build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250)};
  constexpr std::string_view kernel_name = "dynamic_lds_auto_report";
  std::vector<uint8_t> bytes = make_gfx1250_code_object(text_words, kernel_name);
  mutate_first_kernel_descriptor(bytes,
                                 [](KD &descriptor) { descriptor.group_segment_fixed_size = 4u; });
  append_kernel_metadata_note(bytes, kernel_name, /*uses_dynamic_stack=*/false,
                              /*sgpr_count=*/0u, std::nullopt, std::nullopt,
                              /*has_dynamic_lds=*/true);
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.kernels.size(), 1u);
  EXPECT_TRUE(result.kernels.front().has_dynamic_lds);
  const ConSanMoiAutoReportInventory inventory =
      inventory_consan_moi_auto_report(result, options, bytes);
  EXPECT_EQ(inventory.inline_lds_bytes,
            consan_moi_max_workgroup_lds_bytes(ROCJITSU_CODE_ARCH_GFX1250));
  const ConSanMoiAutoReportPlan plan = plan_consan_moi_auto_report(inventory);
  ASSERT_TRUE(plan.complete());
  EXPECT_EQ(plan.layout.inline_exact_dispatch_bank_count,
            consan_moi_inline_exact_dispatch_bank_count_for_lds(inventory.inline_lds_bytes));
  EXPECT_EQ(plan.layout.exact_shadow_entry_capacity,
            consan_moi_max_workgroup_lds_bytes(ROCJITSU_CODE_ARCH_GFX1250) /
                consan_moi_exact_shadow::granule_bytes *
                plan.layout.inline_exact_dispatch_bank_count);
}

TEST(ConSanMoi, AutoReportInventoryCoversFullLdsApertureForFlatGroupAccess) {
  const std::array<uint32_t, 1> kernel_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 9> function_words = {
      0xBE8001EBu,                           // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u, 0x00000000u,              // v_mov_b32_e64 v0, s0
      0xD5810001u, 0x00000001u,              // v_mov_b32_e64 v1, s1
      0xEC05007Cu, 0x00000002u, 0x00000000u, // flat_load_b32 v2, v[0:1]
      0xBFB00000u,                           // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words);
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.moi_candidates.size(), 1u);
  ASSERT_EQ(result.moi_candidates.front().source, ConSanMoiCandidateSource::FlatGroup);
  const ConSanMoiAutoReportInventory inventory =
      inventory_consan_moi_auto_report(result, options, bytes);
  EXPECT_EQ(inventory.access_range_count, 1u);
  EXPECT_EQ(inventory.inline_lds_bytes, kConSanMoiInlineShadowConservativeExactShadowEntries *
                                            consan_moi_exact_shadow::granule_bytes);
  const ConSanMoiAutoReportPlan plan = plan_consan_moi_auto_report(inventory);
  ASSERT_TRUE(plan.complete());
  EXPECT_EQ(plan.layout.inline_exact_dispatch_bank_count,
            consan_moi_inline_exact_dispatch_bank_count_for_lds(inventory.inline_lds_bytes));
  EXPECT_EQ(plan.layout.exact_shadow_entry_capacity,
            kConSanMoiInlineShadowConservativeExactShadowEntries *
                plan.layout.inline_exact_dispatch_bank_count);
}

TEST(ConSanMoi, OwnerEpochPrologueRedirectsKernelDescriptorEntry) {
  constexpr uint64_t kExpectedPrologueOffset = 256;
  const std::array<uint32_t, 2> text_words = {
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options = moi_options();
  options.moi_init_owner_epoch = true;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().trampoline_offset, kExpectedPrologueOffset);
  EXPECT_EQ(result.patches.front().original_size, 0u);
  EXPECT_EQ(result.patches.front().trampoline_size, 3u * sizeof(uint32_t));

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.kernels().size(), 1u);
  ASSERT_EQ(patched.text_sections().size(), 1u);
  EXPECT_EQ(patched.text_sections().front()->size(),
            kExpectedPrologueOffset + 3u * sizeof(uint32_t));

  KD descriptor{};
  std::memcpy(&descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  const uint64_t descriptor_vaddr = patched.kernel_descriptor_offset("lds_probe");
  ASSERT_NE(descriptor_vaddr, 0u);
  const int64_t descriptor_entry_vaddr =
      static_cast<int64_t>(descriptor_vaddr) + descriptor.kernel_code_entry_byte_offset;
  const int64_t expected_entry_vaddr =
      static_cast<int64_t>(patched.text_sections().front()->vaddr() + kExpectedPrologueOffset);
  EXPECT_EQ(descriptor_entry_vaddr, expected_entry_vaddr);

  ASSERT_EQ(kExpectedPrologueOffset % sizeof(uint32_t), 0u);
  const auto text_word_count = patched.text_sections().front()->size() / sizeof(uint32_t);
  std::vector<uint32_t> actual_words(text_word_count);
  std::memcpy(actual_words.data(), patched.text_sections().front()->data(),
              actual_words.size() * sizeof(uint32_t));

  ASSERT_GE(actual_words.size(), text_words.size());
  EXPECT_TRUE(std::equal(text_words.begin(), text_words.end(), actual_words.begin()));
  const size_t prologue_word_offset = kExpectedPrologueOffset / sizeof(uint32_t);
  ASSERT_GE(actual_words.size(), prologue_word_offset + 3u);
  for (size_t i = text_words.size(); i < prologue_word_offset; ++i)
    EXPECT_EQ(actual_words[i], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));

  std::vector<uint32_t> expected_prologue_words;
  const auto owner_shift =
      build_v_lshrrev_b32_e32(11, scalar_positive_inline_u32(6), 0, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(owner_shift);
  expected_prologue_words.push_back(*owner_shift);
  expected_prologue_words.push_back(
      build_v_mov_b32_e32(12, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4));
  const auto branch =
      compute_sopp_branch_simm16(kExpectedPrologueOffset + 2u * sizeof(uint32_t), 0);
  ASSERT_TRUE(branch);
  expected_prologue_words.push_back(build_s_branch(*branch, ROCJITSU_CODE_ARCH_RDNA4));

  const std::span<const uint32_t> actual_prologue_words(actual_words.data() + prologue_word_offset,
                                                        expected_prologue_words.size());
  EXPECT_TRUE(std::equal(expected_prologue_words.begin(), expected_prologue_words.end(),
                         actual_prologue_words.begin()));
}

TEST(ConSanMoi, Cdna4OwnerEpochPrologueRedirectsKernelDescriptorEntry) {
  const std::array<uint32_t, 2> text_words = {
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  const std::vector<uint8_t> bytes = make_cdna4_lds_code_object(text_words, "owner_epoch");
  ConSanOptions options = moi_options();
  options.moi_init_owner_epoch = true;
  options.moi_owner_vgpr = 10;
  options.moi_epoch_vgpr = 11;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_EQ(result.target_name, "gfx950");
  EXPECT_EQ(result.arch_display_name, "cdna4");
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.kernels().size(), 1u);
  KD descriptor{};
  std::memcpy(&descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  const uint64_t descriptor_vaddr = patched.kernel_descriptor_offset("owner_epoch");
  ASSERT_NE(descriptor_vaddr, 0u);
  const int64_t descriptor_entry_vaddr =
      static_cast<int64_t>(descriptor_vaddr) + descriptor.kernel_code_entry_byte_offset;
  EXPECT_EQ(descriptor_entry_vaddr, static_cast<int64_t>(patched.text_sections().front()->vaddr() +
                                                         result.patches.front().trampoline_offset));
}

TEST(ConSanMoi, OwnerEpochPrologueUsesIndirectReturnBeyondSoppRange) {
  std::vector<uint32_t> text_words(33000u, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options = moi_options();
  options.moi_init_owner_epoch = true;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;
  options.moi_exec_save_sgpr = 30;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  const ConSanPatchInfo &patch = result.patches.front();
  EXPECT_EQ(patch.kind, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue);
  EXPECT_EQ(patch.trampoline_size, 8u * sizeof(uint32_t));

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const std::vector<uint32_t> actual_words =
      text_words_at_offset(patched, patch.trampoline_offset, patch.trampoline_size);

  const auto owner_init =
      build_v_lshrrev_b32_e32(11, scalar_positive_inline_u32(6), 0, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(owner_init);
  EXPECT_EQ(actual_words[0], *owner_init);
  EXPECT_EQ(actual_words[1],
            build_v_mov_b32_e32(12, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(actual_words[2], pack_sop1(/*s_getpc_b64=*/0x47, kRdna4VccLo, 0));

  std::vector<uint32_t> expected_builder;
  const uint64_t pc_after_getpc = patch.trampoline_offset + 3u * sizeof(uint32_t);
  ASSERT_TRUE(append_pc_delta_builder(expected_builder, ROCJITSU_CODE_ARCH_RDNA4, kRdna4VccLo,
                                      -static_cast<int64_t>(pc_after_getpc)));
  ASSERT_EQ(expected_builder.size(), 3u);
  EXPECT_TRUE(
      std::equal(expected_builder.begin(), expected_builder.end(), actual_words.begin() + 3));
  EXPECT_EQ(actual_words[6], *build_s_wait_alu_sa_sdst0(ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(actual_words[7], pack_sop1(/*s_setpc_b64=*/0x48, 0, kRdna4VccLo));
}

TEST(ConSanMoi, OwnerEpochPrologueUsesWave32DescriptorForOwnerShift) {
  constexpr uint64_t kExpectedPrologueOffset = 256;
  const std::array<uint32_t, 2> text_words = {
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "lds_probe", kRdna4Wave64AllVgprsGranulated, /*wave32=*/true);
  AmdGpuCodeObject input(bytes.data(), bytes.size());
  ASSERT_TRUE(input.is_valid());
  ASSERT_EQ(input.kernels().size(), 1u);
  KD input_descriptor{};
  std::memcpy(&input_descriptor, bytes.data() + input.kernels().front().descriptor_file_offset,
              sizeof(input_descriptor));
  ASSERT_NE(AMDHSA_BITS_GET(input_descriptor.kernel_code_properties,
                            kd::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32),
            0u);
  ConSanOptions options = moi_options();
  options.moi_init_owner_epoch = true;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);

  uint32_t owner_init = 0;
  std::memcpy(&owner_init, patched.text_sections().front()->data() + kExpectedPrologueOffset,
              sizeof(owner_init));
  const auto expected_owner_init =
      build_v_lshrrev_b32_e32(11, scalar_positive_inline_u32(5), 0, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(expected_owner_init);
  EXPECT_EQ(owner_init, *expected_owner_init);
}

TEST(ConSanMoi, OwnerEpochPrologueGrowsKernelDescriptorVgprAllocation) {
  const std::array<uint32_t, 2> text_words = {
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };

  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "lds_probe", /*vgpr_granulated=*/0);
  ConSanOptions options = moi_options();
  options.moi_init_owner_epoch = true;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.kernels().size(), 1u);
  KD descriptor{};
  std::memcpy(&descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  const uint32_t granulated = AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc1,
                                              kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
  EXPECT_EQ(granulated, 3u);
}

TEST(ConSanMoi, OwnerEpochPrologueHwIdOwnerSourceRequiresOwnerSgpr) {
  const std::array<uint32_t, 2> text_words = {
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options = moi_options();
  options.moi_init_owner_epoch = true;
  options.moi_owner_source = ConSanMoiOwnerSource::HwId;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;

  const auto result = try_patch_consan(bytes, options);

  EXPECT_FALSE(result.errors.empty());
  EXPECT_FALSE(result.modified);
  bool saw_missing_sgpr = false;
  for (const std::string &error : result.errors)
    saw_missing_sgpr |= error.find("RJ_CONSAN_MOI_OWNER_SGPR") != std::string::npos;
  EXPECT_TRUE(saw_missing_sgpr);
}

TEST(ConSanMoi, OwnerEpochPrologueCanUseHwIdOwnerSource) {
  constexpr uint64_t kExpectedPrologueOffset = 256;
  const std::array<uint32_t, 2> text_words = {
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };

  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "lds_probe", /*vgpr_granulated=*/0);
  ConSanOptions options = moi_options();
  options.moi_init_owner_epoch = true;
  options.moi_owner_source = ConSanMoiOwnerSource::HwId;
  options.moi_owner_sgpr = 20;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue);
  EXPECT_EQ(result.patches.front().trampoline_size, 5u * sizeof(uint32_t));

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.kernels().size(), 1u);
  ASSERT_EQ(patched.text_sections().size(), 1u);

  KD descriptor{};
  std::memcpy(&descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  const uint32_t vgpr_granulated = AMDHSA_BITS_GET(
      descriptor.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
  EXPECT_EQ(vgpr_granulated, 3u);
  const uint32_t sgpr_granulated = AMDHSA_BITS_GET(
      descriptor.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT);
  EXPECT_EQ(sgpr_granulated, 2u);

  const auto hwreg = build_hwreg_imm(/*reg_id=*/23, /*offset=*/0, /*size_bits=*/10);
  ASSERT_TRUE(hwreg);
  const auto get_hw_id = build_s_getreg_b32(/*sdst=*/20, *hwreg, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(get_hw_id);
  std::vector<uint32_t> expected_prologue_words;
  expected_prologue_words.push_back(*get_hw_id);
  expected_prologue_words.push_back(build_s_delay_alu(kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4));
  expected_prologue_words.push_back(build_v_mov_b32_e32(11, /*src0=*/20, ROCJITSU_CODE_ARCH_RDNA4));
  expected_prologue_words.push_back(
      build_v_mov_b32_e32(12, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4));
  const auto branch =
      compute_sopp_branch_simm16(kExpectedPrologueOffset + 4u * sizeof(uint32_t), 0);
  ASSERT_TRUE(branch);
  expected_prologue_words.push_back(build_s_branch(*branch, ROCJITSU_CODE_ARCH_RDNA4));

  ASSERT_EQ(kExpectedPrologueOffset % sizeof(uint32_t), 0u);
  const auto text_word_count = patched.text_sections().front()->size() / sizeof(uint32_t);
  std::vector<uint32_t> actual_words(text_word_count);
  std::memcpy(actual_words.data(), patched.text_sections().front()->data(),
              actual_words.size() * sizeof(uint32_t));
  const size_t prologue_word_offset = kExpectedPrologueOffset / sizeof(uint32_t);
  ASSERT_GE(actual_words.size(), prologue_word_offset + expected_prologue_words.size());
  const std::span<const uint32_t> actual_prologue_words(actual_words.data() + prologue_word_offset,
                                                        expected_prologue_words.size());
  EXPECT_TRUE(std::equal(expected_prologue_words.begin(), expected_prologue_words.end(),
                         actual_prologue_words.begin()));
}

TEST(ConSanMoi, AutomaticPersistentProloguesOnlyTargetEmittedProbeOwners) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.unrelated_has_lds = true;
  const std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object(fixture);

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_init_owner_epoch = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(8, 0, 0, 0, 1);
  options.max_patches = 1;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto access_patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::InlineMoiAccessRecordStore ||
           patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore;
  });
  ASSERT_NE(access_patch, result.patches.end());
  ASSERT_FALSE(access_patch->owner_descriptor_file_offsets.empty());

  const auto owns_access_patch = [&](uint64_t descriptor_offset) {
    return std::ranges::find(access_patch->owner_descriptor_file_offsets, descriptor_offset) !=
           access_patch->owner_descriptor_file_offsets.end();
  };
  size_t prologue_count = 0;
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.kind != ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue)
      continue;
    ++prologue_count;
    ASSERT_EQ(patch.owner_descriptor_file_offsets.size(), 1u);
    EXPECT_TRUE(owns_access_patch(patch.owner_descriptor_file_offsets.front()));
  }
  EXPECT_EQ(prologue_count, access_patch->owner_descriptor_file_offsets.size());

  const auto omitted_planned_owner =
      std::ranges::find_if(result.resource_plans, [&](const ConSanCandidateResourcePlan &plan) {
        return std::ranges::any_of(plan.owner_descriptor_file_offsets,
                                   [&](uint64_t owner) { return !owns_access_patch(owner); });
      });
  ASSERT_NE(omitted_planned_owner, result.resource_plans.end());
  for (uint64_t owner : omitted_planned_owner->owner_descriptor_file_offsets) {
    if (owns_access_patch(owner))
      continue;
    EXPECT_FALSE(std::ranges::any_of(result.patches, [&](const ConSanPatchInfo &patch) {
      return patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue &&
             std::ranges::find(patch.owner_descriptor_file_offsets, owner) !=
                 patch.owner_descriptor_file_offsets.end();
    }));
  }
}

TEST(ConSanMoi, AutomaticScalarPersistentStatePreservesGuestVgprAllocation) {
  const std::array<uint32_t, 4> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32 v0, v0
      build_v_mov_b32_e32(/*vdst=*/62, vector_source_vgpr(62), ROCJITSU_CODE_ARCH_RDNA4),
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  constexpr uint32_t kWave64Vgpr64Granulated = 15;
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "vgpr_pressure", kWave64Vgpr64Granulated);

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_init_owner_epoch = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(2, 0, 0, 0, 1);
  options.max_patches = 1;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.moi_persistent_sgprs_automatic);
  EXPECT_FALSE(result.moi_private_epoch_automatic);
  EXPECT_FALSE(result.moi_persistent_vgprs_automatic);
  EXPECT_FALSE(result.resolved_moi_owner_vgpr);
  EXPECT_FALSE(result.resolved_moi_epoch_vgpr);
  ASSERT_TRUE(result.resolved_moi_persistent_owner_sgpr);
  ASSERT_TRUE(result.resolved_moi_persistent_epoch_sgpr);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue,
                               &ConSanPatchInfo::kind),
            1);
  EXPECT_TRUE(std::ranges::any_of(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::InlineMoiAccessRecordStore ||
           patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore;
  }));
  EXPECT_TRUE(std::ranges::all_of(result.resource_plans, [](const auto &plan) {
    return plan.required_vgpr_count <= plan.current_vgpr_count;
  }));
}

TEST(ConSanMoi, AtomicConsumersRejectUnqualifiedStandaloneMemoryRole) {
  const std::vector<uint8_t> bytes = make_rdna4_flat_atomic_code_object();
  ConSanOptions options = moi_options();
  options.moi_track_atomics = true;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 0, 1, 1);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.sync_sequences.size(), 1u);
  EXPECT_EQ(result.sync_sequences.front().memory_role, ConSanSyncMemoryRole::Unknown);
  EXPECT_EQ(result.sync_sequences.front().memory_role_confidence,
            ConSanSemanticConfidence::Unsupported);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.patches.empty());
  EXPECT_TRUE(result.resource_plans.empty());
}

TEST(ConSanMoi, LdsCellRangesRoundUnalignedBytesToFourByteGranules) {
  constexpr ConSanMoiLdsCellRange byte_0_to_3 = consan_moi_lds_cell_range_for_bytes(0, 4);
  constexpr ConSanMoiLdsCellRange byte_3_to_4 = consan_moi_lds_cell_range_for_bytes(3, 2);
  constexpr ConSanMoiLdsCellRange byte_8_to_11 = consan_moi_lds_cell_range_for_bytes(8, 4);
  constexpr ConSanMoiLdsCellRange adjacent = consan_moi_lds_cell_range_for_bytes(12, 4);

  EXPECT_EQ(byte_0_to_3.start_cell, 0u);
  EXPECT_EQ(byte_0_to_3.cell_count, 1u);
  EXPECT_EQ(byte_3_to_4.start_cell, 0u);
  EXPECT_EQ(byte_3_to_4.cell_count, 2u);
  EXPECT_EQ(byte_8_to_11.start_cell, 2u);
  EXPECT_EQ(byte_8_to_11.cell_count, 1u);
  EXPECT_TRUE(consan_moi_cell_ranges_overlap(byte_3_to_4, byte_0_to_3));
  EXPECT_FALSE(consan_moi_cell_ranges_overlap(byte_8_to_11, adjacent));
}

TEST(ConSanMoi, StrictFlatProvenanceExcludesMaybeGroupCandidates) {
  const std::array<uint32_t, 9> text_words = {
      0xBE8001EBu,                           // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u, 0x00000080u,              // v_mov_b32_e64 v0, 0
      0xD5810001u, 0x00000001u,              // v_mov_b32_e64 v1, s1
      0xEC05007Cu, 0x00000002u, 0x00000000u, // flat_load_b32 v2, v[0:1]
      0xBFB00000u,                           // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);

  ConSanOptions likely_options;
  likely_options.flavor = ConSanFlavor::Moi;
  const auto likely_result = try_patch_consan(bytes, likely_options);
  ASSERT_TRUE(likely_result.errors.empty());
  ASSERT_EQ(likely_result.moi_candidates.size(), 1u);
  EXPECT_EQ(likely_result.moi_candidates.front().source, ConSanMoiCandidateSource::FlatMaybeGroup);

  ConSanOptions strict_options = likely_options;
  strict_options.flat_provenance_mode = ConSanFlatProvenanceMode::Strict;
  const auto strict_result = try_patch_consan(bytes, strict_options);
  ASSERT_TRUE(strict_result.errors.empty());
  EXPECT_TRUE(strict_result.moi_candidates.empty());
  EXPECT_TRUE(std::ranges::any_of(strict_result.warnings, [](const std::string &warning) {
    return warning.find("excluded_maybe_group=1") != std::string::npos;
  }));
}

} // namespace
} // namespace rocjitsu
