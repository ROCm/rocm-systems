// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"
#include "rocjitsu/code/patch/gfx1250_instrumentation_builder.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"
#include "test_paths.h"

#include <fstream>
#include <iterator>

namespace rocjitsu {
namespace {

std::vector<uint8_t> load_consan_fixture_hsaco(std::string_view name) {
  std::ifstream file(test::kernel_hsaco_path(std::string(name).c_str()), std::ios::binary);
  if (!file)
    return {};
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(file),
                              std::istreambuf_iterator<char>());
}

ConSanResult patch_record_replay_fixture_with_auto_report(std::span<const uint8_t> bytes) {
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_track_barriers = true;
  options.moi_track_atomics = false;
  options.max_patches = 65536u;
  options.max_patches_is_expert_limit = false;
  ConSanResult inventory = try_patch_consan(bytes, options);
  if (!inventory.errors.empty())
    return inventory;
  const ConSanMoiAutoReportInventory report_inventory =
      inventory_consan_moi_auto_report(inventory, options, bytes);
  const ConSanMoiAutoReportPlan report_plan = plan_consan_moi_auto_report(report_inventory);
  if (!report_plan.complete())
    return inventory;
  const std::optional<ConSanMoiReportLayoutOverride> layout =
      consan_moi_auto_report_layout_override(report_plan);
  if (!layout)
    return inventory;
  const ConSanMoiReportRetryConfig report{
      .buffer_address = 0x123456780000ull,
      .buffer_size = report_plan.required_bytes,
      .layout = *layout,
      .generation = 1u,
      .dispatch_id = 1u,
  };
  return retry_patch_consan_moi_from_inventory(
      std::move(inventory), ConSanMoiInventoryRetryConfig{.report = report, .fault = std::nullopt},
      bytes);
}

TEST(ConSanMoi, Gfx950TritonAttentionRecordReplayLowersEverySupportedSite) {
  constexpr std::array<std::string_view, 3> kFixtures = {
      "triton_cdna4_matmul_buffer_async_1024",
      "triton_cdna4_flash_attention_no_async_1024",
      "triton_cdna4_flash_attention_buffer_async_1024",
  };
  for (const std::string_view fixture : kFixtures) {
    SCOPED_TRACE(fixture);
    const std::vector<uint8_t> bytes = load_consan_fixture_hsaco(fixture);
    ASSERT_FALSE(bytes.empty());
    const ConSanResult result = patch_record_replay_fixture_with_auto_report(bytes);
    SCOPED_TRACE(testing::PrintToString(result.warnings));
    ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
    ASSERT_TRUE(result.modified);
    EXPECT_TRUE(result.final_validation_passed);
    const auto supported = [&](ConSanResourceSiteKind kind) {
      return std::ranges::count_if(result.site_dispositions, [&](const auto &site) {
        return site.site_kind == kind && site.disposition == ConSanSiteDisposition::Supported;
      });
    };
    const auto patched = [&](ConSanResourceSiteKind kind) {
      return std::ranges::count_if(result.site_dispositions, [&](const auto &site) {
        return site.site_kind == kind && site.disposition == ConSanSiteDisposition::Supported &&
               site.lowering_outcome == ConSanSiteLoweringOutcome::Patched;
      });
    };
    EXPECT_EQ(patched(ConSanResourceSiteKind::Access), supported(ConSanResourceSiteKind::Access));
    EXPECT_EQ(patched(ConSanResourceSiteKind::Barrier), supported(ConSanResourceSiteKind::Barrier));
  }
}

TEST(ConSanMoi, KernelAllowlistUsesExactEntryNamesAndLeavesOtherKernelsUntouched) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.entry_nop_words = 8u;
  fixture.unrelated_has_lds = true;
  const std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object(fixture);

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.kernel_name_allowlist = {"unrelated_kernel.kd"};
  options.scratch_vgpr = 8u;
  options.moi_exec_save_sgpr = 30u;
  options.moi_owner_vgpr = 14u;
  options.moi_epoch_vgpr = 15u;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1u, 0u, 0u, 0u);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.max_patches = 1u;

  const ConSanResult selected = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(selected)) << testing::PrintToString(selected.errors);
  ASSERT_TRUE(selected.modified) << testing::PrintToString(selected.warnings);
  ASSERT_EQ(selected.moi_candidates.size(), 1u);
  EXPECT_EQ(selected.moi_candidates.front().container_name, "unrelated_kernel");
  ASSERT_EQ(selected.site_dispositions.size(), 1u);
  EXPECT_EQ(selected.site_dispositions.front().container_name, "unrelated_kernel");
  const auto unrelated =
      std::ranges::find(selected.kernels, "unrelated_kernel", &ConSanKernelInfo::name);
  ASSERT_NE(unrelated, selected.kernels.end());
  const auto access_patch = std::ranges::find(
      selected.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore, &ConSanPatchInfo::kind);
  ASSERT_NE(access_patch, selected.patches.end());
  ASSERT_EQ(access_patch->owner_descriptor_file_offsets.size(), 1u);
  EXPECT_EQ(access_patch->owner_descriptor_file_offsets.front(), unrelated->descriptor_file_offset);
  EXPECT_TRUE(std::ranges::all_of(selected.patches, [&](const ConSanPatchInfo &patch) {
    return patch.owner_descriptor_file_offsets.empty() ||
           (patch.owner_descriptor_file_offsets.size() == 1u &&
            patch.owner_descriptor_file_offsets.front() == unrelated->descriptor_file_offset);
  }));

  options.kernel_name_allowlist = {"unrelated"};
  const ConSanResult no_substring_match = try_patch_consan(bytes, options);
  ASSERT_TRUE(consan_patch_succeeded(no_substring_match))
      << testing::PrintToString(no_substring_match.errors);
  EXPECT_FALSE(no_substring_match.modified);
  EXPECT_TRUE(no_substring_match.moi_candidates.empty());
  EXPECT_TRUE(no_substring_match.site_dispositions.empty());
  EXPECT_TRUE(no_substring_match.patches.empty());

  options.kernel_name_allowlist = {"shared_owner_0"};
  const ConSanResult partial_shared_ownership = try_patch_consan(bytes, options);
  ASSERT_TRUE(consan_patch_succeeded(partial_shared_ownership))
      << testing::PrintToString(partial_shared_ownership.errors);
  EXPECT_FALSE(partial_shared_ownership.modified);
  EXPECT_TRUE(partial_shared_ownership.moi_candidates.empty());

  options.kernel_name_allowlist = {"shared_owner_0", "shared_owner_1"};
  const ConSanResult complete_shared_ownership = try_patch_consan(bytes, options);
  ASSERT_TRUE(consan_patch_succeeded(complete_shared_ownership))
      << testing::PrintToString(complete_shared_ownership.errors);
  ASSERT_TRUE(complete_shared_ownership.modified)
      << testing::PrintToString(complete_shared_ownership.warnings);
  ASSERT_EQ(complete_shared_ownership.moi_candidates.size(), 1u);
  EXPECT_EQ(complete_shared_ownership.moi_candidates.front().container_name, "shared_lds_helper");
  const auto shared_patch =
      std::ranges::find(complete_shared_ownership.patches,
                        ConSanPatchKind::TrampolineMoiAccessRecordStore, &ConSanPatchInfo::kind);
  ASSERT_NE(shared_patch, complete_shared_ownership.patches.end());
  EXPECT_EQ(shared_patch->owner_descriptor_file_offsets.size(), 2u);
}

std::vector<uint32_t> make_expected_fetch_add_one_words(uint64_t address, uint16_t result_vgpr,
                                                        uint16_t scratch_vgpr) {
  std::vector<uint32_t> words;
  const auto mov_address_lo = build_v_mov_b32_e64_literal(
      scratch_vgpr, static_cast<uint32_t>(address), ROCJITSU_CODE_ARCH_RDNA4);
  const auto mov_address_hi =
      build_v_mov_b32_e64_literal(static_cast<uint16_t>(scratch_vgpr + 1u),
                                  static_cast<uint32_t>(address >> 32u), ROCJITSU_CODE_ARCH_RDNA4);
  const auto mov_one = build_v_mov_b32_e64_literal(result_vgpr, 1u, ROCJITSU_CODE_ARCH_RDNA4);
  const auto atomic_add = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      scratch_vgpr, result_vgpr, result_vgpr, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  if (!mov_address_lo || !mov_address_hi || !mov_one || !atomic_add)
    return words;
  words.insert(words.end(), mov_address_lo->begin(), mov_address_lo->end());
  words.insert(words.end(), mov_address_hi->begin(), mov_address_hi->end());
  words.insert(words.end(), mov_one->begin(), mov_one->end());
  words.insert(words.end(), atomic_add->begin(), atomic_add->end());
  const auto wait = instrumentation::build_s_wait_global_load0(ROCJITSU_CODE_ARCH_RDNA4);
  const auto store_wait = instrumentation::build_s_wait_global_store0(ROCJITSU_CODE_ARCH_RDNA4);
  if (!wait || !store_wait)
    return {};
  words.push_back(*wait);
  words.push_back(*store_wait);
  return words;
}

std::vector<uint32_t> record_access_patch_words(const ConSanResult &result,
                                                const ConSanPatchInfo &patch,
                                                uint64_t text_file_offset) {
  if (patch.kind == ConSanPatchKind::InlineMoiAccessRecordStore) {
    return patched_words_at_file_offset(result, text_file_offset + patch.anchor_offset,
                                        patch.original_size);
  }

  EXPECT_EQ(patch.kind, ConSanPatchKind::TrampolineMoiAccessRecordStore);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  EXPECT_TRUE(patched.is_valid());
  if (!patched.is_valid())
    return {};
  return text_words_at_offset(patched, patch.trampoline_offset, patch.trampoline_size);
}

TEST(ConSanMoi, RecordReplayEngineInventoriesCodeObjectWithoutModification) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.visited_code_object);
  EXPECT_FALSE(result.modified);
  EXPECT_EQ(result.flavor, ConSanFlavor::Moi);
  EXPECT_EQ(result.moi_engine, ConSanMoiEngine::RecordReplay);
  EXPECT_TRUE(result.elf_bytes.empty());
  ASSERT_EQ(result.kernels.size(), 1u);
  const ConSanKernelInfo &kernel = result.kernels.front();
  ASSERT_TRUE(kernel.uses_dynamic_stack.has_value());
  EXPECT_FALSE(*kernel.uses_dynamic_stack);
  EXPECT_TRUE(kernel.decoded);
  EXPECT_EQ(kernel.preflight_action, ConSanPreflightAction::NotRun);
  EXPECT_EQ(kernel.stats.lds_read_count, 1u);
  EXPECT_EQ(kernel.stats.lds_write_count, 1u);
  ASSERT_EQ(kernel.lds_sites.size(), 2u);
  ASSERT_EQ(result.moi_candidates.size(), 2u);
  EXPECT_EQ(result.moi_candidates[0].source, ConSanMoiCandidateSource::NativeLds);
  EXPECT_EQ(result.moi_candidates[0].kind, ConSanLdsAccessKind::Write);
  EXPECT_TRUE(result.moi_candidates[0].in_kernel);
  EXPECT_EQ(result.moi_candidates[0].container_name, "lds_probe");
  EXPECT_EQ(result.moi_candidates[0].mnemonic, "ds_store_b32");
  EXPECT_EQ(result.moi_candidates[0].text_offset, 0u);
  EXPECT_EQ(result.moi_candidates[0].file_offset, 0x100u);
  ASSERT_TRUE(result.moi_candidates[0].addr_vgpr);
  EXPECT_EQ(*result.moi_candidates[0].addr_vgpr, 0u);
  ASSERT_TRUE(result.moi_candidates[0].data_vgpr);
  EXPECT_EQ(*result.moi_candidates[0].data_vgpr, 0u);
  EXPECT_EQ(result.moi_candidates[1].kind, ConSanLdsAccessKind::Read);
  EXPECT_EQ(result.moi_candidates[1].mnemonic, "ds_load_b32");
  ASSERT_TRUE(result.moi_candidates[1].dst_vgpr);
  EXPECT_EQ(*result.moi_candidates[1].dst_vgpr, 0u);
  ASSERT_EQ(result.resource_plans.size(), 2u);
  for (size_t plan_index = 0; plan_index < result.resource_plans.size(); ++plan_index) {
    const ConSanCandidateResourcePlan &plan = result.resource_plans[plan_index];
    ASSERT_EQ(plan.owner_descriptor_file_offsets.size(), 1u);
    EXPECT_EQ(plan.owner_descriptor_file_offsets.front(), kernel.descriptor_file_offset);
    EXPECT_EQ(plan.source, ConSanRegisterAllocationSource::LivenessDead);
    EXPECT_EQ(plan.reason, ConSanRegisterPlanReason::None);
    EXPECT_EQ(plan.scratch_vgpr, 1);
    EXPECT_EQ(plan.scratch_vgpr_count, plan_index == 0 ? 6u : 7u);
    EXPECT_EQ(plan.current_vgpr_count, 256);
    EXPECT_EQ(plan.max_referenced_vgpr_count, 1);
    EXPECT_EQ(plan.required_vgpr_count, 256);
    EXPECT_EQ(plan.original_private_segment_size, 0u);
  }
  EXPECT_FALSE(result.warnings.empty());
}

TEST(ConSanMoi, Gfx1250RecordReplayZerosUnusedPersistentClusterCoordinate) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA5;
  constexpr uint16_t kClusterCoordinateSgpr = 53u;
  std::vector<uint32_t> words(64u, build_s_nop(0u, kArch));
  words[8] = 0xD8340000u;
  words[9] = 0x00000000u; // ds_store_b32 v0, v0
  words.back() = build_s_endpgm(kArch);

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 8u;
  options.moi_exec_save_sgpr = 80u;
  options.moi_owner_vgpr = 40u;
  options.moi_epoch_vgpr = 41u;
  options.moi_init_owner_epoch = true;
  options.moi_record_replay_workgroup_sgprs = {
      .x = 50u,
      .y = 51u,
      .z = 52u,
      .cluster_workgroup_id = kClusterCoordinateSgpr,
  };
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1u, 0u, 0u, 0u);
  options.moi_runtime_sample_stride = 1u;
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.max_patches = 1u;

  const ConSanResult result = try_patch_consan(
      make_gfx1250_code_object(words, "gfx1250_zero_unused_cluster_coordinate"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.kernels.size(), 1u);
  EXPECT_FALSE(result.kernels.front().uses_gfx1250_cluster_workgroup_id);
  const auto prologue = std::ranges::find(
      result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue, &ConSanPatchInfo::kind);
  ASSERT_NE(prologue, result.patches.end());
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> prologue_words =
      text_words_at_offset(patched, prologue->trampoline_offset, prologue->trampoline_size);
  EXPECT_NE(
      std::ranges::find(prologue_words, build_s_mov_b32(kClusterCoordinateSgpr,
                                                        scalar_positive_inline_u32(0), kArch)),
      prologue_words.end());
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, RecordReplayPatchesAliasedAccessAndBarrierOnceForEveryOwner) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.entry_nop_words = 2u;
  std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object(fixture);
  AmdGpuCodeObject original(bytes.data(), bytes.size());
  ASSERT_TRUE(original.is_valid());
  const auto first =
      std::ranges::find(original.kernels(), "shared_owner_0", &AmdGpuKernelInfo::name);
  const auto alias =
      std::ranges::find(original.kernels(), "shared_owner_1", &AmdGpuKernelInfo::name);
  ASSERT_NE(first, original.kernels().end());
  ASSERT_NE(alias, original.kernels().end());
  ASSERT_EQ(first->code_size, 4u * sizeof(uint32_t));
  ASSERT_FALSE(original.text_sections().empty());

  const uint64_t target_entry_address =
      original.text_sections().front()->vaddr() + first->entry_text_offset;
  const uint64_t alias_descriptor_address = original.kernel_descriptor_offset(alias->name);
  mutate_kernel_descriptor(bytes, alias->name, [&](KD &descriptor) {
    descriptor.kernel_code_entry_byte_offset =
        static_cast<int64_t>(target_entry_address) - static_cast<int64_t>(alias_descriptor_address);
  });
  mutate_elf_symbol_by_name(bytes, alias->name, [&](Elf64_Sym &symbol) {
    symbol.st_value = target_entry_address;
    symbol.st_size = first->code_size;
  });
  const std::array<uint32_t, 4> aliased_body = {
      0xD8340000u,
      0x00000000u, // ds_store_b32 v0, v0
      0xBF940000u, // s_barrier_wait -1
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const uint64_t body_file_offset = first->text_file_offset + first->entry_text_offset;
  ASSERT_LE(body_file_offset, bytes.size());
  ASSERT_LE(sizeof(aliased_body), bytes.size() - body_file_offset);
  std::memcpy(bytes.data() + body_file_offset, aliased_body.data(), sizeof(aliased_body));

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.test_kernel_name_filter = "shared_owner";
  options.scratch_vgpr = 8u;
  options.moi_exec_save_sgpr = 30u;
  options.moi_owner_vgpr = 14u;
  options.moi_epoch_vgpr = 15u;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 1);
  options.moi_track_barriers = true;
  options.moi_track_atomics = false;
  options.max_patches = 1u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result))
      << "warnings=" << testing::PrintToString(result.warnings)
      << " errors=" << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.moi_candidates.size(), 1u);
  EXPECT_EQ(result.moi_candidates.front().container_name, "shared_owner_0");
  EXPECT_FALSE(result.moi_candidates.front().kernel_descriptor_file_offset);
  ASSERT_EQ(result.sync_events.size(), 1u);
  EXPECT_EQ(result.sync_events.front().kind, ConSanSyncEventKind::Barrier);
  ASSERT_EQ(result.site_dispositions.size(), 4u);
  EXPECT_TRUE(std::ranges::all_of(result.site_dispositions, [](const auto &disposition) {
    return (disposition.site_kind == ConSanResourceSiteKind::Access ||
            disposition.site_kind == ConSanResourceSiteKind::Barrier) &&
           disposition.disposition == ConSanSiteDisposition::Supported;
  }));
  ASSERT_EQ(result.resource_plans.size(), 2u);
  for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
    ASSERT_EQ(plan.owner_descriptor_file_offsets.size(), 2u);
    EXPECT_NE(std::ranges::find(plan.owner_descriptor_file_offsets, first->descriptor_file_offset),
              plan.owner_descriptor_file_offsets.end());
    EXPECT_NE(std::ranges::find(plan.owner_descriptor_file_offsets, alias->descriptor_file_offset),
              plan.owner_descriptor_file_offsets.end());
  }
  const auto is_access_patch = [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::InlineMoiAccessRecordStore ||
           patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore;
  };
  EXPECT_EQ(std::ranges::count_if(result.patches, is_access_patch), 1u);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord,
                               &ConSanPatchInfo::kind),
            1u);
  const auto access_patch = std::ranges::find_if(result.patches, is_access_patch);
  ASSERT_NE(access_patch, result.patches.end());
  EXPECT_EQ(access_patch->owner_descriptor_file_offsets.size(), 2u);
  const auto barrier_patch = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord, &ConSanPatchInfo::kind);
  ASSERT_NE(barrier_patch, result.patches.end());
  EXPECT_EQ(barrier_patch->owner_descriptor_file_offsets.size(), 2u);
}

TEST(ConSanMoi, RecordReplayPatchesAliasedAtomicOnceForEveryOwner) {
  const std::vector<uint8_t> bytes = make_rdna4_two_kernel_aliased_ordered_atomic_code_object();
  ASSERT_FALSE(bytes.empty());
  AmdGpuCodeObject original(bytes.data(), bytes.size());
  ASSERT_TRUE(original.is_valid());
  const auto first =
      std::ranges::find(original.kernels(), "shared_owner_0", &AmdGpuKernelInfo::name);
  const auto alias =
      std::ranges::find(original.kernels(), "shared_owner_1", &AmdGpuKernelInfo::name);
  ASSERT_NE(first, original.kernels().end());
  ASSERT_NE(alias, original.kernels().end());

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_track_barriers = false;
  options.moi_track_atomics = true;
  options.scratch_vgpr = 16u;
  options.moi_exec_save_sgpr = 80u;
  options.moi_dispatch_id_sgpr = 70u;
  options.moi_owner_vgpr = 40u;
  options.moi_epoch_vgpr = 41u;
  options.moi_init_owner_epoch = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(
      /*access_record_capacity=*/1u, /*diagnostic_capacity=*/0u,
      /*exact_shadow_entry_capacity=*/0u, /*sampled_watchpoint_capacity=*/0u,
      /*barrier_record_capacity=*/0u, /*atomic_record_capacity=*/1u,
      /*fence_record_capacity=*/1u);
  options.max_patches = 3u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto atomic_patch = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiAtomicRecord, &ConSanPatchInfo::kind);
  ASSERT_NE(atomic_patch, result.patches.end()) << testing::PrintToString(result.warnings);
  ASSERT_EQ(atomic_patch->owner_descriptor_file_offsets.size(), 2u);
  EXPECT_NE(
      std::ranges::find(atomic_patch->owner_descriptor_file_offsets, first->descriptor_file_offset),
      atomic_patch->owner_descriptor_file_offsets.end());
  EXPECT_NE(
      std::ranges::find(atomic_patch->owner_descriptor_file_offsets, alias->descriptor_file_offset),
      atomic_patch->owner_descriptor_file_offsets.end());
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAtomicRecord,
                               &ConSanPatchInfo::kind),
            1u);
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, RecordReplayDenseAliasedKernelBarriersUseKernelRelayBounds) {
  constexpr uint32_t kSiteCount = 17u;
  TwoKernelSharedFixtureOptions fixture;
  fixture.entry_nop_words = 33'000u;
  std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object(fixture);
  AmdGpuCodeObject original(bytes.data(), bytes.size());
  ASSERT_TRUE(original.is_valid());
  const auto first =
      std::ranges::find(original.kernels(), "shared_owner_0", &AmdGpuKernelInfo::name);
  const auto alias =
      std::ranges::find(original.kernels(), "shared_owner_1", &AmdGpuKernelInfo::name);
  ASSERT_NE(first, original.kernels().end());
  ASSERT_NE(alias, original.kernels().end());
  ASSERT_EQ(first->code_size, alias->code_size);
  ASSERT_FALSE(original.text_sections().empty());

  const uint64_t target_entry_address =
      original.text_sections().front()->vaddr() + first->entry_text_offset;
  const uint64_t alias_descriptor_address = original.kernel_descriptor_offset(alias->name);
  mutate_kernel_descriptor(bytes, alias->name, [&](KD &descriptor) {
    descriptor.kernel_code_entry_byte_offset =
        static_cast<int64_t>(target_entry_address) - static_cast<int64_t>(alias_descriptor_address);
  });
  mutate_elf_symbol_by_name(bytes, alias->name, [&](Elf64_Sym &symbol) {
    symbol.st_value = target_entry_address;
    symbol.st_size = first->code_size;
  });

  const uint32_t filler = build_s_mov_b32(/*sdst=*/20u, /*ssrc0=*/20u, ROCJITSU_CODE_ARCH_RDNA4);
  std::vector<uint32_t> body(first->code_size / sizeof(uint32_t), filler);
  size_t cursor = 32u;
  for (uint32_t index = 0u; index < kSiteCount; ++index) {
    body[cursor++] = 0xD8340000u | index * sizeof(uint32_t);
    body[cursor++] = 0x00000000u; // ds_store_b32 v0, v0 offset:index*4
    body[cursor++] = 0xBE804EC1u; // s_barrier_signal -1
    body[cursor++] = 0xBF94FFFFu; // s_barrier_wait -1
  }
  body.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);
  const uint64_t body_file_offset = first->text_file_offset + first->entry_text_offset;
  ASSERT_LE(body_file_offset, bytes.size());
  ASSERT_LE(body.size() * sizeof(uint32_t), bytes.size() - body_file_offset);
  std::memcpy(bytes.data() + body_file_offset, body.data(), body.size() * sizeof(uint32_t));

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.test_kernel_name_filter = "shared_owner";
  options.scratch_vgpr = 8u;
  options.moi_exec_save_sgpr = 80u;
  options.moi_owner_vgpr = 40u;
  options.moi_epoch_vgpr = 41u;
  options.moi_init_owner_epoch = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size =
      consan_moi_report_buffer_min_bytes(kSiteCount, 0, 0, 0, 2u * kSiteCount);
  options.moi_track_barriers = true;
  options.moi_track_atomics = false;
  options.max_patches = 4u * kSiteCount;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               &ConSanPatchInfo::kind),
            kSiteCount);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord,
                               &ConSanPatchInfo::kind),
            2u * kSiteCount)
      << testing::PrintToString(result.warnings);
  EXPECT_EQ(std::ranges::count_if(result.site_dispositions,
                                  [](const ConSanSiteDispositionRecord &site) {
                                    return site.site_kind == ConSanResourceSiteKind::Barrier &&
                                           site.lowering_outcome ==
                                               ConSanSiteLoweringOutcome::Patched;
                                  }),
            4u * kSiteCount);
  EXPECT_TRUE(std::ranges::all_of(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind != ConSanPatchKind::TrampolineMoiBarrierRecord ||
           patch.owner_descriptor_file_offsets.size() == 2u;
  }));
}

TEST(ConSanMoi, RejectsPhysicalSyncAliasWithConflictingContainerSemantics) {
  const std::array<uint32_t, 4> kernel_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32 v0, v0
      0xBF940000u, // s_barrier_wait -1
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::array<uint32_t, 1> function_words = {
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words);
  AmdGpuCodeObject original(bytes.data(), bytes.size());
  ASSERT_TRUE(original.is_valid());
  ASSERT_FALSE(original.text_sections().empty());
  const auto kernel = std::ranges::find(original.kernels(), "lds_probe", &AmdGpuKernelInfo::name);
  ASSERT_NE(kernel, original.kernels().end());
  const uint64_t kernel_entry_address =
      original.text_sections().front()->vaddr() + kernel->entry_text_offset;
  mutate_elf_symbol_by_name(bytes, "lds_helper", [&](Elf64_Sym &symbol) {
    symbol.st_value = kernel_entry_address;
    symbol.st_size = kernel->code_size;
  });

  const ConSanResult result = try_patch_consan(bytes, moi_options(ConSanMoiEngine::RecordReplay));

  EXPECT_EQ(result.outcome, ConSanTransformOutcome::Invalid);
  EXPECT_TRUE(std::ranges::any_of(result.errors, [](const std::string &error) {
    return error.find("synchronization site") != std::string::npos &&
           error.find("decoded inconsistently through aliases") != std::string::npos;
  })) << testing::PrintToString(result.errors);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
}

TEST(ConSanMoi, Rdna4RecordReplayRecordsHardwareDispatchIdentity) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(2, 0, 0, 0);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.moi_dispatch_id_sgprs_automatic);
  ASSERT_TRUE(result.resolved_moi_dispatch_id_sgpr);
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::InlineMoiAccessRecordStore ||
           patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore;
  });
  ASSERT_NE(access, result.patches.end());
  ASSERT_TRUE(access->scratch_vgpr);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> access_words =
      access->kind == ConSanPatchKind::InlineMoiAccessRecordStore
          ? patched_words_at_file_offset(result, 0x100 + access->anchor_offset,
                                         access->original_size)
          : text_words_at_offset(patched, access->trampoline_offset, access->trampoline_size);
  EXPECT_TRUE(contains_subsequence(access_words, make_expected_scalar_offset_store_words(
                                                     offsetof(ConSanMoiAccessRecord, generation),
                                                     *result.resolved_moi_dispatch_id_sgpr,
                                                     *access->scratch_vgpr)));
  EXPECT_TRUE(contains_subsequence(
      access_words, make_expected_scalar_offset_store_words(
                        offsetof(ConSanMoiAccessRecord, generation) + sizeof(uint32_t),
                        static_cast<uint16_t>(*result.resolved_moi_dispatch_id_sgpr + 1u),
                        *access->scratch_vgpr)));
}

TEST(ConSanMoi, ReportBufferRetryMatchesFreshRecordReplayTransform) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  ConSanOptions inventory_options = moi_options(ConSanMoiEngine::RecordReplay);
  ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  ASSERT_TRUE(inventory.errors.empty()) << testing::PrintToString(inventory.errors);
  ASSERT_FALSE(inventory.modified);

  ConSanOptions options = inventory_options;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(2, 0, 0, 0);
  const ConSanResult fresh = try_patch_consan(bytes, options);
  const ConSanMoiReportRetryConfig report{
      .buffer_address = options.moi_report_buffer_address,
      .buffer_size = options.moi_report_buffer_size,
      .layout = options.moi_report_layout,
      .generation = options.moi_report_generation,
      .dispatch_id = options.moi_report_dispatch_id,
  };
  const ConSanResult retried = retry_patch_consan_moi_from_inventory(
      std::move(inventory), ConSanMoiInventoryRetryConfig{.report = report, .fault = std::nullopt},
      bytes);

  ASSERT_TRUE(consan_patch_succeeded(fresh)) << testing::PrintToString(fresh.errors);
  ASSERT_TRUE(consan_patch_succeeded(retried)) << testing::PrintToString(retried.errors);
  EXPECT_EQ(retried.outcome, fresh.outcome);
  EXPECT_EQ(retried.modified, fresh.modified);
  EXPECT_EQ(retried.final_validation_passed, fresh.final_validation_passed);
  EXPECT_EQ(retried.elf_bytes, fresh.elf_bytes);
  EXPECT_EQ(retried.warnings, fresh.warnings);
  ASSERT_EQ(retried.patches.size(), fresh.patches.size());
  for (size_t index = 0; index < fresh.patches.size(); ++index) {
    EXPECT_EQ(retried.patches[index].kind, fresh.patches[index].kind);
    EXPECT_EQ(retried.patches[index].anchor_offset, fresh.patches[index].anchor_offset);
    EXPECT_EQ(retried.patches[index].trampoline_offset, fresh.patches[index].trampoline_offset);
  }
}

TEST(ConSanMoi, ReportBufferRetryHandlesRecordReplaySyncInventory) {
  const std::array<uint32_t, 5> text_words = {
      0xD8340000u, 0x00000000u, // ds_store_b32
      0xBE804EC1u,              // s_barrier_signal -1
      0xBF94FFFFu,              // s_barrier_wait -1
      0xBFB00000u,              // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "record_replay_retry_sync_inventory");
  const ConSanMoiAutoReportPlan report_plan = plan_consan_moi_auto_report(
      {.engine = ConSanMoiEngine::RecordReplay, .access_range_count = 1, .barrier_event_count = 1});
  ASSERT_TRUE(report_plan.complete());
  const auto layout_override = consan_moi_auto_report_layout_override(report_plan);
  ASSERT_TRUE(layout_override);

  ConSanOptions inventory_options = moi_options(ConSanMoiEngine::RecordReplay);
  inventory_options.moi_track_barriers = true;
  ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  ASSERT_TRUE(inventory.errors.empty()) << testing::PrintToString(inventory.errors);
  ASSERT_FALSE(inventory.modified);

  ConSanOptions options = inventory_options;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = report_plan.required_bytes;
  options.moi_report_layout = *layout_override;
  const ConSanResult fresh = try_patch_consan(bytes, options);
  const ConSanMoiReportRetryConfig report{
      .buffer_address = options.moi_report_buffer_address,
      .buffer_size = options.moi_report_buffer_size,
      .layout = options.moi_report_layout,
      .generation = options.moi_report_generation,
      .dispatch_id = options.moi_report_dispatch_id,
  };
  const ConSanResult retried = retry_patch_consan_moi_from_inventory(
      std::move(inventory), ConSanMoiInventoryRetryConfig{.report = report, .fault = std::nullopt},
      bytes);

  ASSERT_TRUE(consan_patch_succeeded(fresh)) << testing::PrintToString(fresh.errors);
  ASSERT_TRUE(consan_patch_succeeded(retried)) << testing::PrintToString(retried.errors);
  EXPECT_TRUE(fresh.modified);
  EXPECT_TRUE(retried.modified);
  EXPECT_EQ(retried.outcome, fresh.outcome);
  EXPECT_EQ(retried.final_validation_passed, fresh.final_validation_passed);
  EXPECT_EQ(retried.elf_bytes, fresh.elf_bytes);
  EXPECT_EQ(retried.warnings, fresh.warnings);
  EXPECT_NE(std::ranges::find(retried.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue,
                              &ConSanPatchInfo::kind),
            retried.patches.end());
  EXPECT_TRUE(std::ranges::any_of(retried.site_dispositions, [](const auto &site) {
    return site.site_kind == ConSanResourceSiteKind::Barrier &&
           site.disposition == ConSanSiteDisposition::Supported;
  }));
}

TEST(ConSanMoi, ReportBufferRetryRejectsMismatchedOrMutableInventory) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  ConSanOptions inventory_options = moi_options(ConSanMoiEngine::RecordReplay);
  const ConSanResult pristine = try_patch_consan(bytes, inventory_options);
  ASSERT_TRUE(pristine.errors.empty()) << testing::PrintToString(pristine.errors);
  ASSERT_FALSE(pristine.modified);
  ConSanMoiReportRetryConfig report;
  report.buffer_address = 0x123456780000ull;
  report.buffer_size = consan_moi_report_buffer_min_bytes(2, 0, 0, 0);
  const auto expect_invalid = [&](ConSanResult inventory, std::span<const uint8_t> retry_bytes,
                                  std::string_view expected_error) {
    const ConSanResult result = retry_patch_consan_moi_from_inventory(
        std::move(inventory),
        ConSanMoiInventoryRetryConfig{.report = report, .fault = std::nullopt}, retry_bytes);
    EXPECT_EQ(result.outcome, ConSanTransformOutcome::Invalid);
    EXPECT_TRUE(std::ranges::any_of(result.errors, [&](const std::string &error) {
      return error.find(expected_error) != std::string::npos;
    })) << testing::PrintToString(result.errors);
    EXPECT_FALSE(result.modified);
    EXPECT_TRUE(result.elf_bytes.empty());
  };

  std::vector<uint8_t> wrong_size = bytes;
  wrong_size.push_back(0u);
  expect_invalid(pristine, wrong_size, "code-object size");

  std::vector<uint8_t> wrong_bytes = bytes;
  ASSERT_GT(wrong_bytes.size(), 0x100u);
  wrong_bytes[0x100u] ^= 1u;
  expect_invalid(pristine, wrong_bytes, "code-object bytes");

  ConSanResult wrong_engine = pristine;
  wrong_engine.moi_engine = ConSanMoiEngine::Sampled;
  expect_invalid(std::move(wrong_engine), bytes, "requested engine");

  ConSanResult wrong_target = pristine;
  wrong_target.target = ROCJITSU_CODE_TARGET_GFX1250;
  expect_invalid(std::move(wrong_target), bytes, "code-object target");

  ConSanResult wrong_arch = pristine;
  wrong_arch.arch = ROCJITSU_CODE_ARCH_CDNA5;
  expect_invalid(std::move(wrong_arch), bytes, "code-object architecture");

  ConSanResult modified = pristine;
  modified.modified = true;
  expect_invalid(std::move(modified), bytes, "unmodified semantic inventory");
}

TEST(ConSanMoi, AutoRecordReplaySelectsBoundedSlotFromFullAccessIdentity) {
  const std::array<uint32_t, 6> text_words = {
      0xD8340000u, 0x00000000u, // ds_store_b32
      0xD8D80000u, 0x00000000u, // ds_load_b32
      0xBFC60000u,              // s_wait_dscnt
      0xBFB00000u,              // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "record_replay_banked_identity", kRdna4Wave64AllVgprsGranulated,
      /*wave32=*/false, /*uses_dynamic_stack=*/false,
      /*workgroup_id_dimension_mask=*/7u);
  const ConSanMoiAutoReportPlan report_plan = plan_consan_moi_auto_report(
      {.engine = ConSanMoiEngine::RecordReplay, .access_range_count = 2});
  ASSERT_TRUE(report_plan.complete());
  const auto layout_override = consan_moi_auto_report_layout_override(report_plan);
  ASSERT_TRUE(layout_override);

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = report_plan.required_bytes;
  options.moi_report_layout = *layout_override;
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.resolved_moi_dispatch_id_sgpr);
  ASSERT_TRUE(result.resolved_moi_record_replay_workgroup_vgprs.complete())
      << "RDNA4 automatic banked replay must capture its exact tuple in VGPRs";
  EXPECT_TRUE(result.resolved_moi_record_replay_workgroup_sgprs.empty());
  EXPECT_FALSE(result.resolved_moi_workgroup_key_vgpr);
  EXPECT_FALSE(result.resolved_moi_persistent_workgroup_key_sgpr);
  EXPECT_NE(std::ranges::find(result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue,
                              &ConSanPatchInfo::kind),
            result.patches.end());
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::InlineMoiAccessRecordStore ||
           patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore;
  });
  ASSERT_NE(access, result.patches.end());
  ASSERT_TRUE(access->scratch_vgpr);
  const uint16_t scratch = *access->scratch_vgpr;
  const uint32_t access_identity_capacity = report_plan.layout.access_record_capacity;

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> access_words =
      access->kind == ConSanPatchKind::InlineMoiAccessRecordStore
          ? patched_words_at_file_offset(result, 0x100 + access->anchor_offset,
                                         access->original_size)
          : text_words_at_offset(patched, access->trampoline_offset, access->trampoline_size);
  const auto hash_dispatch = instrumentation::build_v_xor_b32(
      static_cast<uint16_t>(scratch + 6u), vector_source_vgpr(static_cast<uint16_t>(scratch + 2u)),
      static_cast<uint16_t>(scratch + 3u), ROCJITSU_CODE_ARCH_RDNA4);
  const auto bound_dispatch_directory = instrumentation::build_v_and_b32_literal(
      static_cast<uint16_t>(scratch + 6u), kConSanMoiRecordReplayMaximumDispatchTokenCount - 1u,
      static_cast<uint16_t>(scratch + 6u), ROCJITSU_CODE_ARCH_RDNA4);
  const auto scale_dispatch_slot = instrumentation::build_v_mul_lo_u32_literal(
      static_cast<uint16_t>(scratch + 4u), static_cast<uint16_t>(scratch + 5u), sizeof(uint64_t),
      static_cast<uint16_t>(scratch + 6u), ROCJITSU_CODE_ARCH_RDNA4);
  const auto add_dispatch_slot = instrumentation::build_v_add_u64_vgpr_offset(
      scratch, static_cast<uint16_t>(scratch + 4u), ROCJITSU_CODE_ARCH_RDNA4);
  const auto increment_dispatch_probe = instrumentation::build_v_add_u32_literal(
      static_cast<uint16_t>(scratch + 8u), static_cast<uint16_t>(scratch + 4u), 1u,
      static_cast<uint16_t>(scratch + 8u), ROCJITSU_CODE_ARCH_RDNA4);
  const auto materialize_dispatch_capacity = instrumentation::build_v_mov_b32_literal(
      static_cast<uint16_t>(scratch + 4u), kConSanMoiRecordReplayProbeLimit,
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto dispatch_probe_in_range = instrumentation::build_v_cmp_gt_u32_vcc(
      vector_source_vgpr(static_cast<uint16_t>(scratch + 4u)), static_cast<uint16_t>(scratch + 8u),
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto increment_dispatch_bank = instrumentation::build_v_add_u32(
      static_cast<uint16_t>(scratch + 6u), vector_source_vgpr(static_cast<uint16_t>(scratch + 6u)),
      static_cast<uint16_t>(scratch + 8u), ROCJITSU_CODE_ARCH_RDNA4);
  const auto wrap_dispatch_directory = instrumentation::build_v_and_b32_literal(
      static_cast<uint16_t>(scratch + 6u), kConSanMoiRecordReplayMaximumDispatchTokenCount - 1u,
      static_cast<uint16_t>(scratch + 6u), ROCJITSU_CODE_ARCH_RDNA4);
  const auto owner_mix = instrumentation::build_v_mul_lo_u32_literal(
      static_cast<uint16_t>(scratch + 7u), static_cast<uint16_t>(scratch + 3u),
      kConSanMoiRecordReplayIdentityHashMultiplier, static_cast<uint16_t>(scratch + 7u),
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto owner_combine = instrumentation::build_v_xor_b32(
      static_cast<uint16_t>(scratch + 7u), vector_source_vgpr(static_cast<uint16_t>(scratch + 7u)),
      static_cast<uint16_t>(scratch + 2u), ROCJITSU_CODE_ARCH_RDNA4);
  const auto combine_dispatch_identity = instrumentation::build_v_xor_b32(
      static_cast<uint16_t>(scratch + 7u), vector_source_vgpr(static_cast<uint16_t>(scratch + 7u)),
      static_cast<uint16_t>(scratch + 6u), ROCJITSU_CODE_ARCH_RDNA4);
  const auto bound_owner = instrumentation::build_v_and_b32_literal(
      static_cast<uint16_t>(scratch + 7u), access_identity_capacity - 1u,
      static_cast<uint16_t>(scratch + 7u), ROCJITSU_CODE_ARCH_RDNA4);
  const auto increment_owner_probe = instrumentation::build_v_add_u32_literal(
      static_cast<uint16_t>(scratch + 8u), static_cast<uint16_t>(scratch + 4u), 1u,
      static_cast<uint16_t>(scratch + 8u), ROCJITSU_CODE_ARCH_RDNA4);
  const auto materialize_identity_capacity = instrumentation::build_v_mov_b32_literal(
      static_cast<uint16_t>(scratch + 4u),
      std::min(access_identity_capacity, kConSanMoiRecordReplayProbeLimit),
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto owner_probe_in_range = instrumentation::build_v_cmp_gt_u32_vcc(
      vector_source_vgpr(static_cast<uint16_t>(scratch + 4u)), static_cast<uint16_t>(scratch + 8u),
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto increment_owner_bank = instrumentation::build_v_add_u32(
      static_cast<uint16_t>(scratch + 7u), vector_source_vgpr(static_cast<uint16_t>(scratch + 7u)),
      static_cast<uint16_t>(scratch + 8u), ROCJITSU_CODE_ARCH_RDNA4);
  const auto wrap_owner_bank = instrumentation::build_v_and_b32_literal(
      static_cast<uint16_t>(scratch + 7u), access_identity_capacity - 1u,
      static_cast<uint16_t>(scratch + 7u), ROCJITSU_CODE_ARCH_RDNA4);
  const auto scale_bank = instrumentation::build_v_mul_lo_u32_literal(
      static_cast<uint16_t>(scratch + 2u), static_cast<uint16_t>(scratch + 3u),
      sizeof(ConSanMoiAccessRecord), static_cast<uint16_t>(scratch + 7u), ROCJITSU_CODE_ARCH_RDNA4);
  const auto add_bank = instrumentation::build_v_add_u64_vgpr_offset(
      scratch, static_cast<uint16_t>(scratch + 2u), ROCJITSU_CODE_ARCH_RDNA4);
  const auto claim = instrumentation::build_flat_atomic_cmpswap_b64(
      scratch, static_cast<uint16_t>(scratch + 2u), static_cast<uint16_t>(scratch + 2u),
      /*return_old_value=*/true, /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  const auto commit = instrumentation::build_flat_atomic_cmpswap_b32(
      scratch, static_cast<uint16_t>(scratch + 2u), static_cast<uint16_t>(scratch + 2u),
      /*return_old_value=*/true, /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  const auto branch_on_incomplete =
      instrumentation::build_s_cbranch_vccnz(/*offset_dwords=*/0, ROCJITSU_CODE_ARCH_RDNA4);
  const auto load_workgroup_x = instrumentation::build_flat_load_b32(
      scratch, static_cast<uint16_t>(scratch + 2u), ROCJITSU_CODE_ARCH_RDNA4,
      offsetof(ConSanMoiAccessRecord, workgroup_x));
  const auto load_workgroup_y = instrumentation::build_flat_load_b32(
      scratch, static_cast<uint16_t>(scratch + 2u), ROCJITSU_CODE_ARCH_RDNA4,
      offsetof(ConSanMoiAccessRecord, workgroup_y));
  const auto load_workgroup_z = instrumentation::build_flat_load_b32(
      scratch, static_cast<uint16_t>(scratch + 2u), ROCJITSU_CODE_ARCH_RDNA4,
      offsetof(ConSanMoiAccessRecord, workgroup_z));
  const auto load_site_token = instrumentation::build_flat_load_b32(
      scratch, static_cast<uint16_t>(scratch + 2u), ROCJITSU_CODE_ARCH_RDNA4,
      offsetof(ConSanMoiAccessRecord, site_token));
  const auto transient_workgroup_x =
      build_v_mov_b32_e32(static_cast<uint16_t>(scratch + 4u), ttmp_scalar_operand(kTtmpRdna4GridX),
                          ROCJITSU_CODE_ARCH_RDNA4);
  const auto load_persistent_workgroup_x =
      build_v_mov_b32_e32(static_cast<uint16_t>(scratch + 2u),
                          vector_source_vgpr(*result.resolved_moi_record_replay_workgroup_vgprs.x),
                          ROCJITSU_CODE_ARCH_RDNA4);
  const auto load_persistent_workgroup_y =
      build_v_mov_b32_e32(static_cast<uint16_t>(scratch + 2u),
                          vector_source_vgpr(*result.resolved_moi_record_replay_workgroup_vgprs.y),
                          ROCJITSU_CODE_ARCH_RDNA4);
  const auto load_persistent_workgroup_z =
      build_v_mov_b32_e32(static_cast<uint16_t>(scratch + 2u),
                          vector_source_vgpr(*result.resolved_moi_record_replay_workgroup_vgprs.z),
                          ROCJITSU_CODE_ARCH_RDNA4);
  const auto compare_workgroup_x = instrumentation::build_v_cmp_ne_u32_vcc(
      vector_source_vgpr(static_cast<uint16_t>(scratch + 4u)), static_cast<uint16_t>(scratch + 2u),
      ROCJITSU_CODE_ARCH_RDNA4);
  const uint64_t flags_address =
      *options.moi_report_buffer_address + offsetof(ConSanMoiReportHeader, flags);
  const auto flags_low = instrumentation::build_v_mov_b32_literal(
      scratch, static_cast<uint32_t>(flags_address), ROCJITSU_CODE_ARCH_RDNA4);
  const auto flags_high = instrumentation::build_v_mov_b32_literal(
      static_cast<uint16_t>(scratch + 1u), static_cast<uint32_t>(flags_address >> 32u),
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto dispatch_saturation_flag = instrumentation::build_v_mov_b32_literal(
      static_cast<uint16_t>(scratch + 2u),
      kConSanMoiReportFlagRecordReplayBankSaturated |
          kConSanMoiReportFlagRecordReplayDispatchBankSaturated,
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto owner_saturation_flag = instrumentation::build_v_mov_b32_literal(
      static_cast<uint16_t>(scratch + 2u),
      kConSanMoiReportFlagRecordReplayBankSaturated |
          kConSanMoiReportFlagRecordReplayOwnerBankSaturated,
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto saturation_or = instrumentation::build_flat_atomic_or_u32(
      scratch, static_cast<uint16_t>(scratch + 2u), static_cast<uint16_t>(scratch + 2u),
      /*return_old_value=*/true, /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(hash_dispatch && bound_dispatch_directory && scale_dispatch_slot &&
              add_dispatch_slot && increment_dispatch_probe && materialize_dispatch_capacity &&
              dispatch_probe_in_range && increment_dispatch_bank && wrap_dispatch_directory &&
              owner_mix && owner_combine && combine_dispatch_identity && bound_owner &&
              increment_owner_probe && materialize_identity_capacity && owner_probe_in_range &&
              increment_owner_bank && wrap_owner_bank && scale_bank && add_bank && claim &&
              commit && branch_on_incomplete && load_workgroup_x && load_workgroup_y &&
              load_workgroup_z && load_site_token && flags_low && flags_high &&
              dispatch_saturation_flag && owner_saturation_flag && saturation_or &&
              compare_workgroup_x);
  EXPECT_NE(std::ranges::find(access_words, *hash_dispatch), access_words.end());
  EXPECT_TRUE(contains_subsequence(access_words, *bound_dispatch_directory));
  EXPECT_TRUE(contains_subsequence(access_words, *scale_dispatch_slot));
  EXPECT_TRUE(contains_subsequence(access_words, *add_dispatch_slot));
  EXPECT_TRUE(contains_subsequence(access_words, *increment_dispatch_probe));
  EXPECT_TRUE(contains_subsequence(access_words, *materialize_dispatch_capacity));
  EXPECT_NE(std::ranges::find(access_words, *dispatch_probe_in_range), access_words.end());
  EXPECT_TRUE(contains_subsequence(access_words, *increment_dispatch_bank));
  EXPECT_TRUE(contains_subsequence(access_words, *wrap_dispatch_directory));
  EXPECT_TRUE(contains_subsequence(access_words, *owner_mix));
  EXPECT_NE(std::ranges::find(access_words, *owner_combine), access_words.end());
  EXPECT_NE(std::ranges::find(access_words, *combine_dispatch_identity), access_words.end());
  EXPECT_TRUE(contains_subsequence(access_words, *bound_owner));
  EXPECT_TRUE(contains_subsequence(access_words, *increment_owner_probe));
  EXPECT_TRUE(contains_subsequence(access_words, *materialize_identity_capacity));
  EXPECT_NE(std::ranges::find(access_words, *owner_probe_in_range), access_words.end());
  EXPECT_TRUE(contains_subsequence(access_words, *increment_owner_bank));
  EXPECT_TRUE(contains_subsequence(access_words, *wrap_owner_bank));
  EXPECT_TRUE(contains_subsequence(access_words, *scale_bank));
  EXPECT_TRUE(contains_subsequence(access_words, *add_bank));
  EXPECT_TRUE(contains_subsequence(access_words, *claim));
  EXPECT_TRUE(contains_subsequence(access_words, *commit));
  const auto last_commit_read =
      std::find_end(access_words.begin(), access_words.end(), commit->begin(), commit->end());
  ASSERT_NE(last_commit_read, access_words.end());
  const auto incomplete_collision_branch = std::find_if(
      last_commit_read + static_cast<ptrdiff_t>(commit->size()), access_words.end(),
      [&](uint32_t word) { return (word & 0xffff0000u) == (*branch_on_incomplete & 0xffff0000u); });
  ASSERT_NE(incomplete_collision_branch, access_words.end());
  EXPECT_GT(static_cast<int16_t>(*incomplete_collision_branch & 0xffffu), 0)
      << "an incomplete automatic access claim must probe forward instead of "
         "waiting on a colliding publisher";
  EXPECT_TRUE(contains_subsequence(access_words, *load_workgroup_x));
  EXPECT_TRUE(contains_subsequence(access_words, *load_workgroup_y));
  EXPECT_TRUE(contains_subsequence(access_words, *load_workgroup_z));
  EXPECT_TRUE(contains_subsequence(access_words, *load_site_token));
  EXPECT_NE(std::ranges::find(access_words, load_persistent_workgroup_x), access_words.end());
  EXPECT_NE(std::ranges::find(access_words, load_persistent_workgroup_y), access_words.end());
  EXPECT_NE(std::ranges::find(access_words, load_persistent_workgroup_z), access_words.end());
  EXPECT_EQ(std::ranges::find(access_words, transient_workgroup_x), access_words.end());
  EXPECT_NE(std::ranges::find(access_words, *compare_workgroup_x), access_words.end());
  EXPECT_TRUE(contains_subsequence(access_words, *dispatch_saturation_flag));
  EXPECT_TRUE(contains_subsequence(access_words, *owner_saturation_flag));
  EXPECT_TRUE(contains_subsequence(access_words, *saturation_or));
  ASSERT_TRUE(result.resolved_moi_exec_save_sgpr);
  const uint16_t exec_save = *result.resolved_moi_exec_save_sgpr;
  const uint16_t original_exec = static_cast<uint16_t>(exec_save + 8u);
  const uint16_t address_key = static_cast<uint16_t>(exec_save + 10u);
  const uint16_t address_group_exec = static_cast<uint16_t>(exec_save + 12u);
  const auto save_original_exec =
      instrumentation::build_s_mov_b64(original_exec, kRdna4ExecLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto read_address = instrumentation::build_v_readfirstlane_b32(
      address_key, /*address_vgpr=*/0u, ROCJITSU_CODE_ARCH_RDNA4);
  const auto select_address = instrumentation::build_v_cmp_eq_u32_vcc(
      address_key, /*address_vgpr=*/0u, ROCJITSU_CODE_ARCH_RDNA4);
  const auto narrow_address =
      instrumentation::build_s_and_saveexec_b64(exec_save, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto save_address_group =
      instrumentation::build_s_mov_b64(address_group_exec, kRdna4ExecLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto rank_address_group_lo = instrumentation::build_v_mbcnt_lo_u32_b32(
      static_cast<uint16_t>(scratch + 2u), address_group_exec, scalar_positive_inline_u32(0),
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto rank_address_group_hi = instrumentation::build_v_mbcnt_hi_u32_b32(
      static_cast<uint16_t>(scratch + 2u), static_cast<uint16_t>(address_group_exec + 1u),
      vector_source_vgpr(static_cast<uint16_t>(scratch + 2u)), ROCJITSU_CODE_ARCH_RDNA4);
  const auto narrow_representative = instrumentation::build_s_and_saveexec_b64(
      address_group_exec, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto remove_address_group = instrumentation::build_s_xor_b64(
      exec_save, exec_save, address_group_exec, ROCJITSU_CODE_ARCH_RDNA4);
  const auto select_remaining =
      instrumentation::build_s_mov_b64(kRdna4ExecLo, exec_save, ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_original =
      instrumentation::build_s_mov_b64(kRdna4ExecLo, original_exec, ROCJITSU_CODE_ARCH_RDNA4);
  const auto combine_address_identity = instrumentation::build_v_xor_b32(
      static_cast<uint16_t>(scratch + 7u), address_key, static_cast<uint16_t>(scratch + 7u),
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto loop_branch =
      instrumentation::build_s_cbranch_execnz(/*offset_dwords=*/0, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(save_original_exec && read_address && select_address && narrow_address &&
              save_address_group && rank_address_group_lo && rank_address_group_hi &&
              narrow_representative && remove_address_group && select_remaining &&
              restore_original && combine_address_identity && loop_branch);
  const auto save_original_exec_position = std::ranges::find(access_words, *save_original_exec);
  ASSERT_NE(save_original_exec_position, access_words.end());
  const auto read_address_position = std::ranges::find(access_words, *read_address);
  ASSERT_NE(read_address_position, access_words.end());
  EXPECT_NE(std::ranges::find(access_words, *select_address), access_words.end());
  EXPECT_NE(std::ranges::find(access_words, *narrow_address), access_words.end());
  EXPECT_NE(std::ranges::find(access_words, *save_address_group), access_words.end());
  EXPECT_TRUE(contains_subsequence(access_words, *rank_address_group_lo));
  EXPECT_TRUE(contains_subsequence(access_words, *rank_address_group_hi));
  EXPECT_NE(std::ranges::find(access_words, *narrow_representative), access_words.end());
  EXPECT_NE(std::ranges::find(access_words, *remove_address_group), access_words.end());
  EXPECT_NE(std::ranges::find(access_words, *select_remaining), access_words.end());
  EXPECT_NE(std::ranges::find(access_words, *restore_original), access_words.end());
  EXPECT_NE(std::ranges::find(access_words, *combine_address_identity), access_words.end());
  const auto backward_group_branch = std::ranges::find_if(access_words, [&](uint32_t word) {
    return (word & 0xffff0000u) == (*loop_branch & 0xffff0000u) &&
           static_cast<int16_t>(word & 0xffffu) < 0;
  });
  ASSERT_NE(backward_group_branch, access_words.end());
  EXPECT_EQ(std::ranges::count_if(access_words,
                                  [&](uint32_t word) {
                                    return (word & 0xffff0000u) == (*loop_branch & 0xffff0000u) &&
                                           static_cast<int16_t>(word & 0xffffu) < 0;
                                  }),
            1u);
  const int64_t loop_head = std::distance(access_words.begin(), read_address_position);
  const int64_t branch_index = std::distance(access_words.begin(), backward_group_branch);
  const int64_t branch_target =
      branch_index + 1 + static_cast<int16_t>(*backward_group_branch & 0xffffu);
  EXPECT_EQ(branch_target, loop_head);
  EXPECT_TRUE(contains_subsequence(
      access_words, make_expected_scalar_offset_store_words(
                        offsetof(ConSanMoiAccessRecord, lane_mask), address_group_exec, scratch)));
  EXPECT_TRUE(contains_subsequence(
      access_words, make_expected_scalar_offset_store_words(
                        offsetof(ConSanMoiAccessRecord, lane_mask) + sizeof(uint32_t),
                        static_cast<uint16_t>(address_group_exec + 1u), scratch)));
  EXPECT_TRUE(contains_subsequence(
      access_words,
      make_expected_literal_offset_store_words(offsetof(ConSanMoiAccessRecord, flags),
                                               kConSanMoiAccessRecordFlagExactAddressGroupMask,
                                               scratch, static_cast<uint16_t>(scratch + 2u))));
  EXPECT_EQ(report_plan.layout.record_replay_dispatch_token_capacity,
            kConSanMoiRecordReplayMaximumDispatchTokenCount);
  EXPECT_EQ(report_plan.layout.record_replay_logical_access_range_count, 2u);
  EXPECT_EQ(report_plan.layout.access_record_capacity, 65536u);
}

TEST(ConSanMoi, AutoRecordReplayOneByOneHeadroomStillAddressesTheHashedSlot) {
  const std::array<uint32_t, 4> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32
      0xBFC60000u, // s_wait_dscnt
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "record_replay_one_by_one_headroom", kRdna4Wave64AllVgprsGranulated,
      /*wave32=*/false, /*uses_dynamic_stack=*/false,
      /*workgroup_id_dimension_mask=*/7u);
  ConSanMoiAutoReportInventory inventory;
  inventory.engine = ConSanMoiEngine::RecordReplay;
  inventory.access_range_count = 1u;
  inventory.diagnostic_count = 1u;
  inventory.record_replay_access_dispatch_bank_count = 1u;
  inventory.record_replay_access_owner_bank_count = 1u;
  inventory.record_replay_address_group_headroom = 1u;
  const ConSanMoiAutoReportPlan report_plan = plan_consan_moi_auto_report(inventory);
  ASSERT_TRUE(report_plan.complete());
  const auto layout_override = consan_moi_auto_report_layout_override(report_plan);
  ASSERT_TRUE(layout_override);

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = report_plan.required_bytes;
  options.moi_report_layout = *layout_override;
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::InlineMoiAccessRecordStore ||
           patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore;
  });
  ASSERT_NE(access, result.patches.end());
  ASSERT_TRUE(access->scratch_vgpr);
  const uint16_t scratch = *access->scratch_vgpr;
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> access_words =
      access->kind == ConSanPatchKind::InlineMoiAccessRecordStore
          ? patched_words_at_file_offset(result, 0x100 + access->anchor_offset,
                                         access->original_size)
          : text_words_at_offset(patched, access->trampoline_offset, access->trampoline_size);
  const auto scale_slot = instrumentation::build_v_mul_lo_u32_literal(
      static_cast<uint16_t>(scratch + 2u), static_cast<uint16_t>(scratch + 3u),
      sizeof(ConSanMoiAccessRecord), static_cast<uint16_t>(scratch + 7u), ROCJITSU_CODE_ARCH_RDNA4);
  const auto add_slot = instrumentation::build_v_add_u64_vgpr_offset(
      scratch, static_cast<uint16_t>(scratch + 2u), ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(scale_slot && add_slot);
  EXPECT_TRUE(contains_subsequence(access_words, *scale_slot));
  EXPECT_TRUE(contains_subsequence(access_words, *add_slot));
}

TEST(ConSanMoi, AutoRecordReplayCapturesDispatchIdentityInPersistentVgprsAtScalarPressure) {
  std::vector<uint32_t> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32
      0xBFC60000u, // s_wait_dscnt
  };
  constexpr uint16_t kExecSaveSgpr = 88u;
  constexpr uint16_t kExecSaveSgprCount = 14u;
  for (uint16_t sgpr = 0u; sgpr < REGISTER_SET_ALLOCATABLE_SGPRS; ++sgpr) {
    if (sgpr >= kExecSaveSgpr && sgpr < kExecSaveSgpr + kExecSaveSgprCount)
      continue;
    const auto use =
        build_s_cmp_eq_u32(sgpr, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_TRUE(use);
    text_words.push_back(*use);
  }
  for (uint16_t sgpr = 102u; sgpr < 106u; ++sgpr) {
    const auto use =
        build_s_cmp_eq_u32(sgpr, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_TRUE(use);
    text_words.push_back(*use);
  }
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4));
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "record_replay_vgpr_dispatch_capture",
                                 /*vgpr_granulated=*/0u);
  const ConSanMoiAutoReportPlan report_plan = plan_consan_moi_auto_report(
      {.engine = ConSanMoiEngine::RecordReplay, .access_range_count = 1u});
  ASSERT_TRUE(report_plan.complete());
  const auto layout_override = consan_moi_auto_report_layout_override(report_plan);
  ASSERT_TRUE(layout_override);

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = report_plan.required_bytes;
  options.moi_report_layout = *layout_override;
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.moi_exec_save_sgpr = kExecSaveSgpr;

  const ConSanResult result = try_patch_consan(bytes, options);

  SCOPED_TRACE(testing::PrintToString(result.warnings));
  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_FALSE(result.resolved_moi_dispatch_id_sgpr);
  ASSERT_TRUE(result.resolved_moi_dispatch_id_vgpr);
  EXPECT_TRUE(result.moi_dispatch_id_vgprs_automatic);
  EXPECT_TRUE(result.final_validation_passed);
  const auto prologue = std::ranges::find(
      result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue, &ConSanPatchInfo::kind);
  ASSERT_NE(prologue, result.patches.end());
  EXPECT_FALSE(prologue->dispatch_id_capture_sgpr);
  EXPECT_EQ(prologue->dispatch_id_capture_vgpr, result.resolved_moi_dispatch_id_vgpr);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.kernels().size(), 1u);
  KD descriptor{};
  std::memcpy(&descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  const uint32_t allocated_vgprs =
      (AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc1,
                       kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT) +
       1u) *
      4u;
  EXPECT_GE(allocated_vgprs, static_cast<uint32_t>(*result.resolved_moi_dispatch_id_vgpr) + 2u);
  EXPECT_GT(allocated_vgprs, 4u);
}

TEST(ConSanMoi, Gfx1100AutoRecordReplayCapturesDispatchIdentityInPersistentVgprs) {
  constexpr auto store = rdna3::build_ds(rdna3::kDsStoreB32Ds, {.addr = 0u, .data0 = 1u});
  std::vector<uint32_t> text_words = {store[0], store[1]};
  text_words.insert(text_words.end(), 128u, build_s_nop(0u, ROCJITSU_CODE_ARCH_RDNA3));
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA3));
  std::vector<uint8_t> bytes =
      make_rdna3_lds_code_object(text_words, "gfx1100_record_replay_dispatch_capture",
                                 /*vgpr_granulated=*/0u, /*wave32=*/true,
                                 /*uses_dynamic_stack=*/false,
                                 /*workgroup_id_dimension_mask=*/1u);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                    kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 15u);
  });
  const ConSanMoiAutoReportPlan report_plan = plan_consan_moi_auto_report(
      {.engine = ConSanMoiEngine::RecordReplay, .access_range_count = 1u});
  ASSERT_TRUE(report_plan.complete());
  const auto layout_override = consan_moi_auto_report_layout_override(report_plan);
  ASSERT_TRUE(layout_override);

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = report_plan.required_bytes;
  options.moi_report_layout = *layout_override;
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;

  const ConSanResult result = try_patch_consan(bytes, options);

  SCOPED_TRACE(testing::PrintToString(result.warnings));
  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified);
  EXPECT_FALSE(result.resolved_moi_dispatch_id_sgpr);
  EXPECT_TRUE(result.resolved_moi_dispatch_id_vgpr);
  EXPECT_TRUE(result.moi_dispatch_id_vgprs_automatic);
  EXPECT_TRUE(result.final_validation_passed);
  const auto prologue = std::ranges::find(
      result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue, &ConSanPatchInfo::kind);
  ASSERT_NE(prologue, result.patches.end());
  EXPECT_EQ(prologue->dispatch_id_source_sgpr, 2u);
  EXPECT_EQ(prologue->dispatch_id_original_user_sgpr_count, 15u);
  EXPECT_EQ(prologue->dispatch_id_expanded_user_sgpr_count, 15u);
  EXPECT_EQ(prologue->dispatch_id_kernarg_reload_count, 0u);
  EXPECT_EQ(prologue->dispatch_id_shifted_guest_sgpr_count, 0u);
  EXPECT_EQ(prologue->dispatch_id_shifted_system_sgpr_count, 0u);
  EXPECT_EQ(prologue->dispatch_id_system_sgpr_shift, 0u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  KD descriptor{};
  std::memcpy(&descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  EXPECT_EQ(AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT),
            15u);
  EXPECT_EQ(AMDHSA_BITS_GET(descriptor.kernel_code_properties,
                            kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID),
            1u);
  EXPECT_EQ(AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2,
                            kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X),
            1u);
  EXPECT_EQ(AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2,
                            kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y),
            1u);
  EXPECT_EQ(AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2,
                            kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z),
            1u);
}

TEST(ConSanMoi, Gfx1100FullVgprRecordReplayUsesPersistentScalarState) {
  std::vector<uint32_t> text_words;
  text_words.reserve(2u * 256u + 131u);
  for (uint16_t vgpr = 0u; vgpr < 256u; ++vgpr) {
    text_words.push_back(
        build_v_mov_b32_e32(vgpr, vector_source_vgpr(vgpr), ROCJITSU_CODE_ARCH_RDNA3));
  }
  constexpr auto store = rdna3::build_ds(rdna3::kDsStoreB32Ds, {.addr = 0u, .data0 = 1u});
  text_words.insert(text_words.end(), store.begin(), store.end());
  text_words.insert(text_words.end(), 128u, build_s_nop(0u, ROCJITSU_CODE_ARCH_RDNA3));
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA3));
  std::vector<uint8_t> bytes =
      make_rdna3_lds_code_object(text_words, "gfx1100_full_vgpr_record_replay_scalar_state",
                                 kRdna4Wave64AllVgprsGranulated, /*wave32=*/true,
                                 /*uses_dynamic_stack=*/true, /*workgroup_id_dimension_mask=*/1u);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                    kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 15u);
  });
  const ConSanMoiAutoReportPlan report_plan = plan_consan_moi_auto_report(
      {.engine = ConSanMoiEngine::RecordReplay, .access_range_count = 1u});
  ASSERT_TRUE(report_plan.complete());
  const auto layout_override = consan_moi_auto_report_layout_override(report_plan);
  ASSERT_TRUE(layout_override);

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = report_plan.required_bytes;
  options.moi_report_layout = *layout_override;
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;

  const ConSanResult result = try_patch_consan(bytes, options);

  SCOPED_TRACE(testing::PrintToString(result.warnings));
  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_TRUE(result.moi_dispatch_id_sgprs_automatic);
  ASSERT_TRUE(result.resolved_moi_dispatch_id_sgpr);
  EXPECT_FALSE(result.resolved_moi_dispatch_id_vgpr);
  EXPECT_FALSE(result.moi_dispatch_id_vgprs_automatic);
  EXPECT_TRUE(result.moi_persistent_sgprs_automatic);
  EXPECT_FALSE(result.moi_persistent_vgprs_automatic);
  ASSERT_TRUE(result.resolved_moi_persistent_owner_sgpr);
  ASSERT_TRUE(result.resolved_moi_persistent_epoch_sgpr);
  ASSERT_TRUE(result.resolved_moi_record_replay_workgroup_sgprs.complete());
  EXPECT_TRUE(result.resolved_moi_record_replay_workgroup_vgprs.empty());
  EXPECT_EQ(*result.resolved_moi_persistent_epoch_sgpr,
            *result.resolved_moi_persistent_owner_sgpr + 1u);
  EXPECT_TRUE(
      *result.resolved_moi_dispatch_id_sgpr + 2u <= *result.resolved_moi_persistent_owner_sgpr ||
      *result.resolved_moi_persistent_owner_sgpr + 5u <= *result.resolved_moi_dispatch_id_sgpr);

  const auto prologue = std::ranges::find(
      result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue, &ConSanPatchInfo::kind);
  ASSERT_NE(prologue, result.patches.end());
  EXPECT_EQ(prologue->dispatch_id_capture_sgpr, result.resolved_moi_dispatch_id_sgpr);
  EXPECT_FALSE(prologue->dispatch_id_capture_vgpr);
  EXPECT_EQ(prologue->dispatch_id_source_sgpr, 2u);
}

TEST(ConSanMoi, Gfx1100RecordReplayRouteKeyDoesNotAliasPersistentEpoch) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA3;
  constexpr auto store = rdna3::build_ds(rdna3::kDsStoreB32Ds, {.addr = 1u, .data0 = 5u});
  constexpr auto load = rdna3::build_ds(rdna3::kDsLoadB32Ds, {.addr = 1u, .vdst = 5u});
  std::vector<uint32_t> text_words;
  text_words.reserve(101'000u);
  for (uint16_t vgpr = 1u; vgpr < 256u; vgpr += 2u)
    text_words.push_back(build_v_mov_b32_e32(vgpr, vector_source_vgpr(vgpr), kArch));
  text_words.insert(text_words.end(), store.begin(), store.end());
  text_words.insert(text_words.end(), 100'000u,
                    build_s_mov_b32(/*sdst=*/20u, /*ssrc0=*/20u, kArch));
  text_words.insert(text_words.end(), load.begin(), load.end());
  text_words.insert(text_words.end(), store.begin(), store.end());
  text_words.insert(text_words.end(), load.begin(), load.end());
  for (uint16_t sgpr = 0u; sgpr < 92u; ++sgpr)
    text_words.push_back(build_s_mov_b32(/*sdst=*/0u, sgpr, kArch));
  text_words.push_back(build_s_endpgm(kArch));

  std::vector<uint8_t> bytes =
      make_rdna3_lds_code_object(text_words, "gfx1100_record_replay_route_key_persistent_epoch",
                                 kRdna4Wave64AllVgprsGranulated, /*wave32=*/true,
                                 /*uses_dynamic_stack=*/true, /*workgroup_id_dimension_mask=*/1u);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                    kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 15u);
  });
  const ConSanMoiAutoReportPlan report_plan = plan_consan_moi_auto_report(
      {.engine = ConSanMoiEngine::RecordReplay, .access_range_count = 4u});
  ASSERT_TRUE(report_plan.complete());
  const auto layout_override = consan_moi_auto_report_layout_override(report_plan);
  ASSERT_TRUE(layout_override);

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = report_plan.required_bytes;
  options.moi_report_layout = *layout_override;
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;

  const ConSanResult result = try_patch_consan(bytes, options);

  SCOPED_TRACE(testing::PrintToString(result.warnings));
  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified);
  EXPECT_TRUE(result.final_validation_passed);
  ASSERT_TRUE(result.resolved_moi_persistent_owner_sgpr);
  ASSERT_TRUE(result.resolved_moi_persistent_epoch_sgpr);
  ASSERT_TRUE(result.resolved_moi_record_replay_workgroup_sgprs.complete());
  ASSERT_EQ(result.resolved_moi_transient_sgpr_assignments.size(), 1u);
  const ConSanMoiTransientSgprAssignment &assignment =
      result.resolved_moi_transient_sgpr_assignments.front();
  ASSERT_TRUE(assignment.dispatch_key_sgpr);

  const uint16_t persistent_begin = *result.resolved_moi_persistent_owner_sgpr;
  const uint16_t persistent_end =
      static_cast<uint16_t>(*result.resolved_moi_record_replay_workgroup_sgprs.z + 1u);
  const auto persistent_overlaps = [&](std::optional<uint16_t> begin, uint16_t width = 1u) {
    return begin && *begin < persistent_end && persistent_begin < *begin + width;
  };
  EXPECT_FALSE(persistent_overlaps(assignment.exec_save_sgpr, 4u));
  EXPECT_FALSE(persistent_overlaps(assignment.owner_sgpr));
  EXPECT_FALSE(persistent_overlaps(assignment.dispatch_id_sgpr, 2u));
  EXPECT_FALSE(persistent_overlaps(assignment.indirect_pc_sgpr, 2u));
  EXPECT_FALSE(persistent_overlaps(assignment.indirect_scc_sgpr));
  EXPECT_FALSE(persistent_overlaps(assignment.dispatch_key_sgpr));
  EXPECT_FALSE(persistent_overlaps(assignment.call_return_sgpr, 2u));
  EXPECT_FALSE(persistent_overlaps(assignment.visible_evidence_sgpr));
  EXPECT_FALSE(persistent_overlaps(assignment.dynamic_stack_borrowed_sgpr, 2u));
}

TEST(ConSanMoi, AutoRecordReplaySpillsWideAddressGroupStateAtScalarPressure) {
  for (const rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    SCOPED_TRACE(arch);
    constexpr std::array<uint16_t, 4> kRouterState = {0u, 1u, 4u, 6u};
    std::vector<uint32_t> words = {
        0xD8340000u,
        0x00000000u, // ds_store_b32 v0, v0
    };
    for (uint16_t sgpr = 0; sgpr < 106u; ++sgpr) {
      if (std::ranges::find(kRouterState, sgpr) == kRouterState.end())
        words.push_back(build_s_mov_b32(/*M0=*/125u, sgpr, arch));
    }
    words.push_back(build_s_endpgm(arch));

    const std::vector<uint8_t> bytes =
        arch == ROCJITSU_CODE_ARCH_CDNA5
            ? make_gfx1250_code_object(words, "auto_record_replay_scalar_spill",
                                       kRdna4Wave64AllVgprsGranulated, /*wave32=*/true)
            : make_rdna4_lds_code_object(words, "auto_record_replay_scalar_spill",
                                         kRdna4Wave64AllVgprsGranulated);
    const ConSanMoiAutoReportPlan report_plan = plan_consan_moi_auto_report(
        {.engine = ConSanMoiEngine::RecordReplay, .access_range_count = 1u});
    ASSERT_TRUE(report_plan.complete());
    const auto layout_override = consan_moi_auto_report_layout_override(report_plan);
    ASSERT_TRUE(layout_override);

    ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
    options.scratch_vgpr = 8;
    options.moi_owner_vgpr = 40;
    options.moi_epoch_vgpr = 41;
    options.moi_report_buffer_address = 0x123456780000ull;
    options.moi_report_buffer_size = report_plan.required_bytes;
    options.moi_report_layout = *layout_override;
    options.moi_track_barriers = false;
    options.moi_track_atomics = false;
    options.max_patches = 1;

    const ConSanResult result = try_patch_consan(bytes, options);

    ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
    ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
    ASSERT_EQ(result.resolved_moi_transient_sgpr_assignments.size(), 1u);
    const ConSanMoiTransientSgprAssignment &assignment =
        result.resolved_moi_transient_sgpr_assignments.front();
    EXPECT_TRUE(assignment.spill_backed);
    const auto access_patch = std::ranges::find(
        result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore, &ConSanPatchInfo::kind);
    ASSERT_NE(access_patch, result.patches.end());
    EXPECT_EQ(access_patch->required_private_segment_size, 0u);
    AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(patched.is_valid());
    expect_lane_backed_scalar_spill(result, patched, *access_patch, assignment.exec_save_sgpr,
                                    arch);
    EXPECT_TRUE(result.final_validation_passed);
  }
}

TEST(ConSanMoi, RejectsAmbiguousOrInvalidPersistentDispatchVgprOverrides) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  const auto expect_unsupported = [&](ConSanOptions options, std::string_view diagnostic) {
    options.moi_report_buffer_address = 0x123456780000ull;
    options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(2, 0, 0, 0);
    options.moi_owner_vgpr = 40u;
    options.moi_epoch_vgpr = 41u;
    options.moi_init_owner_epoch = true;
    const ConSanResult result = try_patch_consan(bytes, options);
    EXPECT_EQ(result.outcome, ConSanTransformOutcome::Unsupported);
    EXPECT_FALSE(result.modified);
    EXPECT_TRUE(std::ranges::any_of(result.warnings, [&](const std::string &warning) {
      return warning.find(diagnostic) != std::string::npos;
    })) << testing::PrintToString(result.warnings);
  };

  ConSanOptions ambiguous = moi_options(ConSanMoiEngine::RecordReplay);
  ambiguous.moi_dispatch_id_sgpr = 80u;
  ambiguous.moi_dispatch_id_vgpr = 42u;
  expect_unsupported(std::move(ambiguous),
                     "multiple persistent hardware dispatch-ID representations");

  ConSanOptions overlaps_owner = moi_options(ConSanMoiEngine::RecordReplay);
  overlaps_owner.moi_dispatch_id_vgpr = 40u;
  expect_unsupported(std::move(overlaps_owner),
                     "overlaps persistent state or exceeds the architectural VGPR file");

  ConSanOptions exceeds_file = moi_options(ConSanMoiEngine::RecordReplay);
  exceeds_file.moi_dispatch_id_vgpr = 255u;
  expect_unsupported(std::move(exceeds_file),
                     "overlaps persistent state or exceeds the architectural VGPR file");
}

TEST(ConSanMoi, AutoRecordReplayRejectsDispatchIdentityWithoutPersistentVgprPair) {
  std::vector<uint32_t> text_words;
  for (uint16_t vgpr = 0u; vgpr < 256u; ++vgpr) {
    text_words.push_back(
        build_v_mov_b32_e32(vgpr, vector_source_vgpr(vgpr), ROCJITSU_CODE_ARCH_RDNA4));
  }
  text_words.push_back(0xD8340000u);
  text_words.push_back(0x00000000u); // ds_store_b32
  text_words.push_back(0xBE804EC1u); // s_barrier_signal -1
  text_words.push_back(0xBF94FFFFu); // s_barrier_wait -1
  for (uint16_t sgpr = 0u; sgpr < REGISTER_SET_ALLOCATABLE_SGPRS; ++sgpr) {
    if (sgpr >= 90u)
      continue;
    const auto use =
        build_s_cmp_eq_u32(sgpr, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_TRUE(use);
    text_words.push_back(*use);
  }
  for (uint16_t sgpr = 102u; sgpr < 106u; ++sgpr) {
    const auto use =
        build_s_cmp_eq_u32(sgpr, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_TRUE(use);
    text_words.push_back(*use);
  }
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4));
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "record_replay_no_dispatch_vgpr_pair");
  const ConSanMoiAutoReportPlan report_plan = plan_consan_moi_auto_report(
      {.engine = ConSanMoiEngine::RecordReplay, .access_range_count = 1u});
  ASSERT_TRUE(report_plan.complete());
  const auto layout_override = consan_moi_auto_report_layout_override(report_plan);
  ASSERT_TRUE(layout_override);
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = report_plan.required_bytes;
  options.moi_report_layout = *layout_override;
  options.moi_track_barriers = true;
  options.moi_track_atomics = false;
  options.moi_exec_save_sgpr = 90u;

  const ConSanResult result = try_patch_consan(bytes, options);

  EXPECT_EQ(result.outcome, ConSanTransformOutcome::Unsupported);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find(
               "cannot retain automatic Record/Replay dispatch identity without a persistent "
               "VGPR pair") != std::string::npos;
  })) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(std::ranges::any_of(result.site_dispositions, [](const auto &site) {
    return site.site_kind == ConSanResourceSiteKind::Barrier &&
           site.disposition == ConSanSiteDisposition::Supported &&
           site.lowering_outcome == ConSanSiteLoweringOutcome::ResourceFailed &&
           site.lowering_reason == ConSanSiteLoweringReason::UnsupportedResourcePlan &&
           site.resource_reason == ConSanRegisterPlanReason::NoLegalWindow;
  })) << testing::PrintToString(result.site_dispositions);
  EXPECT_TRUE(std::ranges::all_of(result.site_dispositions, [](const auto &site) {
    return site.disposition != ConSanSiteDisposition::Supported ||
           (site.lowering_outcome == ConSanSiteLoweringOutcome::ResourceFailed &&
            site.lowering_reason == ConSanSiteLoweringReason::UnsupportedResourcePlan &&
            site.resource_reason == ConSanRegisterPlanReason::NoLegalWindow);
  })) << testing::PrintToString(result.site_dispositions);
}

TEST(ConSanMoi, AutoRecordReplayAddsExactTupleToExplicitOwnerEpoch) {
  const std::array<uint32_t, 4> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32
      0xBFC60000u, // s_wait_dscnt
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "record_replay_explicit_owner_epoch", kRdna4Wave64AllVgprsGranulated,
      /*wave32=*/false, /*uses_dynamic_stack=*/false,
      /*workgroup_id_dimension_mask=*/7u);
  const ConSanMoiAutoReportPlan report_plan = plan_consan_moi_auto_report(
      {.engine = ConSanMoiEngine::RecordReplay, .access_range_count = 1});
  ASSERT_TRUE(report_plan.complete());
  const auto layout_override = consan_moi_auto_report_layout_override(report_plan);
  ASSERT_TRUE(layout_override);

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_owner_vgpr = 40u;
  options.moi_epoch_vgpr = 41u;
  options.moi_init_owner_epoch = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = report_plan.required_bytes;
  options.moi_report_layout = *layout_override;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_EQ(result.resolved_moi_owner_vgpr, 40u);
  EXPECT_EQ(result.resolved_moi_epoch_vgpr, 41u);
  ASSERT_TRUE(result.resolved_moi_record_replay_workgroup_vgprs.complete());
  EXPECT_GT(*result.resolved_moi_record_replay_workgroup_vgprs.x, 41u);
  EXPECT_GT(*result.resolved_moi_record_replay_workgroup_vgprs.y, 41u);
  EXPECT_GT(*result.resolved_moi_record_replay_workgroup_vgprs.z, 41u);
  EXPECT_TRUE(result.resolved_moi_record_replay_workgroup_sgprs.empty());
  EXPECT_NE(std::ranges::find(result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue,
                              &ConSanPatchInfo::kind),
            result.patches.end());
}

TEST(ConSanMoi, RecordReplayRejectsAmbiguousPersistentWorkgroupRepresentations) {
  std::vector<uint32_t> text_words(320u, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000000u; // ds_store_b32 v0, v0
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_record_replay_workgroup_sgprs = {.x = 40u, .y = 41u, .z = 42u};
  options.moi_record_replay_workgroup_vgprs = {.x = 50u, .y = 51u, .z = 52u};
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const ConSanResult result =
      try_patch_consan(make_rdna4_lds_code_object(text_words, "ambiguous_tuple"), options);

  EXPECT_EQ(result.outcome, ConSanTransformOutcome::Unsupported);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("multiple persistent workgroup-tuple representations") != std::string::npos;
  })) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(std::ranges::none_of(result.warnings, [](const std::string &warning) {
    return warning.find("no common entry workgroup assignment") != std::string::npos;
  }));
}

TEST(ConSanMoi, RecordReplayRuntimeGateUsesPersistentWorkgroupTuple) {
  std::vector<uint32_t> text_words;
  for (uint32_t i = 0; i < 9u; ++i) {
    text_words.push_back(0xD8340000u); // ds_store_b32 v0, v0
    text_words.push_back(0x00000000u);
  }
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4));
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "record_replay_runtime_gate", kRdna4Wave64AllVgprsGranulated,
      /*wave32=*/false, /*uses_dynamic_stack=*/false,
      /*workgroup_id_dimension_mask=*/7u);
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 8u;
  options.moi_exec_save_sgpr = 80u;
  options.moi_dispatch_id_sgpr = 70u;
  options.moi_owner_vgpr = 30u;
  options.moi_epoch_vgpr = 31u;
  options.moi_record_replay_workgroup_sgprs = {.x = 40u, .y = 41u, .z = 42u};
  options.moi_init_owner_epoch = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(9, 0, 0, 0);
  options.moi_runtime_sample_stride = 65536u;
  options.moi_runtime_sample_offset = 1234u;
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.max_patches = 9u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result))
      << "errors=" << testing::PrintToString(result.errors)
      << " warnings=" << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               &ConSanPatchInfo::kind),
            9);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  std::vector<uint32_t> patched_words(patched.text_sections().front()->size() / sizeof(uint32_t));
  std::memcpy(patched_words.data(), patched.text_sections().front()->data(),
              patched_words.size() * sizeof(uint32_t));
  const uint16_t quotient_sgpr = 85u;
  const uint16_t residue_sgpr = 86u;
  EXPECT_NE(std::ranges::find(patched_words, build_s_lshr_b32(quotient_sgpr, residue_sgpr,
                                                              scalar_positive_inline_u32(16u),
                                                              ROCJITSU_CODE_ARCH_RDNA4)),
            patched_words.end());
  ASSERT_NE(std::ranges::find(patched_words, *build_s_sub_u32(residue_sgpr, residue_sgpr, 40u,
                                                              ROCJITSU_CODE_ARCH_RDNA4)),
            patched_words.end());
  const auto literal_move = std::ranges::search(
      patched_words,
      std::array<uint32_t, 2>{
          build_s_mov_b32(quotient_sgpr, /*literal source=*/255u, ROCJITSU_CODE_ARCH_RDNA4),
          1234u,
      });
  EXPECT_NE(literal_move.begin(), patched_words.end());
}

TEST(ConSanMoi, AutomaticRecordReplayPreservesRuntimeWorkgroupTupleAcrossTargets) {
  constexpr uint32_t kAccessCount = 9u;
  for (const rj_code_arch_t arch : {
           ROCJITSU_CODE_ARCH_CDNA3,
           ROCJITSU_CODE_ARCH_CDNA4,
           ROCJITSU_CODE_ARCH_CDNA5,
           ROCJITSU_CODE_ARCH_RDNA3,
           ROCJITSU_CODE_ARCH_RDNA4,
       }) {
    SCOPED_TRACE("arch=" + std::to_string(arch));
    std::vector<uint32_t> text_words(8u, build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/0u, arch));
    if (arch == ROCJITSU_CODE_ARCH_CDNA5) {
      // Qwen/TP2 kernels consume this cluster-local coordinate. It must be
      // mixed into the cached predicate before the persistent x slot is reused.
      text_words[0] = build_s_mov_b32(/*sdst=*/10u, ttmp_scalar_operand(/*ttmp=*/6u), arch);
    }
    for (uint32_t index = 0u; index < kAccessCount; ++index) {
      if (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4) {
        const auto access = arch == ROCJITSU_CODE_ARCH_CDNA3
                                ? build_cdna3_ds_store_b32(/*vaddr=*/0u, /*vdata=*/1u,
                                                           index * sizeof(uint32_t), arch)
                                : build_cdna4_ds_store_b32(/*vaddr=*/0u, /*vdata=*/1u,
                                                           index * sizeof(uint32_t), arch);
        ASSERT_TRUE(access);
        text_words.insert(text_words.end(), access->begin(), access->end());
      } else {
        text_words.push_back(0xD8340000u | index * sizeof(uint32_t));
        text_words.push_back(0x00000000u); // ds_store_b32 v0, v0 offset:index*4
      }
    }
    text_words.push_back(build_s_endpgm(arch));

    const std::vector<uint8_t> bytes = [&]() {
      switch (arch) {
      case ROCJITSU_CODE_ARCH_CDNA3:
        return make_cdna3_lds_code_object(text_words, "cdna3_cached_runtime_selection");
      case ROCJITSU_CODE_ARCH_CDNA4:
        return make_cdna4_lds_code_object(text_words, "cdna4_cached_runtime_selection");
      case ROCJITSU_CODE_ARCH_CDNA5:
        return make_gfx1250_code_object(text_words, "gfx1250_cached_runtime_selection");
      case ROCJITSU_CODE_ARCH_RDNA3:
        return make_rdna3_lds_code_object(text_words, "rdna3_cached_runtime_selection");
      case ROCJITSU_CODE_ARCH_RDNA4:
        return make_rdna4_lds_code_object(text_words, "rdna4_cached_runtime_selection");
      default:
        return std::vector<uint8_t>{};
      }
    }();
    ASSERT_FALSE(bytes.empty());

    ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
    options.moi_init_owner_epoch = true;
    options.moi_report_buffer_address = 0x123456780000ull;
    options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(kAccessCount, 0u, 0u, 0u);
    options.moi_runtime_sample_stride = 65'536u;
    options.moi_runtime_sample_offset = 1'234u;
    options.moi_track_barriers = false;
    options.moi_track_atomics = false;
    options.max_patches = kAccessCount;

    const ConSanResult result = try_patch_consan(bytes, options);

    ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
    ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
    EXPECT_TRUE(result.final_validation_passed);
    const bool has_dispatch_register = result.resolved_moi_dispatch_id_sgpr.has_value() ||
                                       result.resolved_moi_dispatch_id_vgpr.has_value();
    if (has_dispatch_register) {
      EXPECT_TRUE(result.moi_dispatch_id_sgprs_automatic || result.moi_dispatch_id_vgprs_automatic);
      ASSERT_NE(result.resolved_moi_dispatch_id_sgpr.has_value(),
                result.resolved_moi_dispatch_id_vgpr.has_value());
    } else {
      EXPECT_EQ(arch, ROCJITSU_CODE_ARCH_RDNA3);
    }
    ASSERT_TRUE(result.resolved_moi_exec_save_sgpr);
    const bool scalar_selection = result.resolved_moi_record_replay_workgroup_sgprs.complete();
    const bool vector_selection = result.resolved_moi_record_replay_workgroup_vgprs.complete();
    ASSERT_NE(scalar_selection, vector_selection);
    const uint16_t workgroup_x_register =
        scalar_selection ? *result.resolved_moi_record_replay_workgroup_sgprs.x
                         : *result.resolved_moi_record_replay_workgroup_vgprs.x;
    if (arch == ROCJITSU_CODE_ARCH_CDNA5) {
      ASSERT_EQ(result.kernels.size(), 1u);
      EXPECT_TRUE(result.kernels.front().uses_gfx1250_cluster_workgroup_id);
      EXPECT_TRUE(result.resolved_moi_record_replay_workgroup_sgprs.cluster_workgroup_id ||
                  result.resolved_moi_record_replay_workgroup_vgprs.cluster_workgroup_id);
    }

    AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(patched.is_valid());
    const std::vector<uint32_t> patched_words =
        text_words_at_offset(patched, /*text_offset=*/0u,
                             static_cast<uint32_t>(patched.text_sections().front()->size()));
    const uint16_t residue = arch == ROCJITSU_CODE_ARCH_CDNA5
                                 ? *result.resolved_moi_exec_save_sgpr
                                 : static_cast<uint16_t>(*result.resolved_moi_exec_save_sgpr + 6u);
    const uint32_t initialize_workgroup_hash =
        build_s_mov_b32(residue, scalar_positive_inline_u32(0u), arch);
    EXPECT_GE(std::ranges::count(patched_words, initialize_workgroup_hash), kAccessCount);
    const auto overwrite_workgroup_x = instrumentation::build_s_cselect_b32(
        workgroup_x_register, scalar_positive_inline_u32(1u), scalar_positive_inline_u32(0u), arch);
    ASSERT_TRUE(overwrite_workgroup_x);
    EXPECT_EQ(std::ranges::count(patched_words, *overwrite_workgroup_x), 0u)
        << "the sampling predicate must not replace exact workgroup X";

    uint16_t selected_scalar = workgroup_x_register;
    if (vector_selection) {
      selected_scalar = static_cast<uint16_t>(*result.resolved_moi_exec_save_sgpr + 5u);
      const auto read =
          instrumentation::build_v_readfirstlane_b32(selected_scalar, workgroup_x_register, arch);
      ASSERT_TRUE(read);
      EXPECT_GE(std::ranges::count(patched_words, *read), kAccessCount);
    }
    const auto mix_workgroup_x =
        instrumentation::build_s_sub_u32(residue, residue, selected_scalar, arch);
    ASSERT_TRUE(mix_workgroup_x);
    EXPECT_GE(std::ranges::count(patched_words, *mix_workgroup_x), kAccessCount);
  }
}

TEST(ConSanMoi, RecordReplayRuntimeGateCoversBarrierRecords) {
  constexpr uint32_t kBarrierWait = 0xBF940000u;
  const std::array<uint32_t, 2> text_words = {
      kBarrierWait,
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 8u;
  options.moi_exec_save_sgpr = 80u;
  options.moi_dispatch_id_sgpr = 70u;
  options.moi_owner_vgpr = 30u;
  options.moi_epoch_vgpr = 31u;
  options.moi_record_replay_workgroup_sgprs = {.x = 40u, .y = 41u, .z = 42u};
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 1);
  options.moi_runtime_sample_stride = 65536u;
  options.moi_runtime_sample_offset = 1234u;
  options.moi_track_barriers = true;
  options.moi_track_atomics = false;
  options.max_patches = 1u;

  const ConSanResult result =
      try_patch_consan(make_rdna4_lds_code_object(text_words, "barrier_runtime_gate"), options);

  ASSERT_TRUE(consan_patch_succeeded(result))
      << "errors=" << testing::PrintToString(result.errors)
      << " warnings=" << testing::PrintToString(result.warnings);
  const auto patch = std::ranges::find(result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord,
                                       &ConSanPatchInfo::kind);
  ASSERT_NE(patch, result.patches.end());
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> words =
      text_words_at_offset(patched, patch->trampoline_offset, patch->trampoline_size);
  const auto gate_entry = instrumentation::build_s_cselect_b32(
      /*sdst=*/84u, scalar_positive_inline_u32(1u), scalar_positive_inline_u32(0u),
      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(gate_entry);
  ASSERT_FALSE(words.empty());
  EXPECT_EQ(words.front(), *gate_entry);
  EXPECT_EQ(std::ranges::count(words, kBarrierWait), 2u);
}

TEST(ConSanMoi, RecordReplayRuntimeGateCoversStandaloneAtomicAndFenceRecords) {
  const std::vector<uint8_t> bytes = make_rdna4_atomic_fence_sequence_code_object();
  ASSERT_FALSE(bytes.empty());
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 16u;
  options.moi_exec_save_sgpr = 80u;
  options.moi_dispatch_id_sgpr = 70u;
  options.moi_owner_vgpr = 30u;
  options.moi_epoch_vgpr = 31u;
  options.moi_record_replay_workgroup_sgprs = {.x = 40u, .y = 41u, .z = 42u};
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(
      /*access_record_capacity=*/1u, /*diagnostic_capacity=*/0u,
      /*exact_shadow_entry_capacity=*/0u, /*sampled_watchpoint_capacity=*/0u,
      /*barrier_record_capacity=*/0u, /*atomic_record_capacity=*/1u,
      /*fence_record_capacity=*/2u);
  options.moi_runtime_sample_stride = 65536u;
  options.moi_runtime_sample_offset = 1234u;
  options.moi_track_barriers = false;
  options.moi_track_atomics = true;
  options.max_patches = 3u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result))
      << "errors=" << testing::PrintToString(result.errors)
      << " warnings=" << testing::PrintToString(result.warnings);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAtomicRecord,
                               &ConSanPatchInfo::kind),
            1u);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiFenceRecord,
                               &ConSanPatchInfo::kind),
            1u);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const auto gate_entry = instrumentation::build_s_cselect_b32(
      /*sdst=*/84u, scalar_positive_inline_u32(1u), scalar_positive_inline_u32(0u),
      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(gate_entry);
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.kind != ConSanPatchKind::TrampolineMoiAtomicRecord &&
        patch.kind != ConSanPatchKind::TrampolineMoiFenceRecord)
      continue;
    const std::vector<uint32_t> words =
        text_words_at_offset(patched, patch.trampoline_offset, patch.trampoline_size);
    ASSERT_FALSE(words.empty());
    EXPECT_EQ(words.front(), *gate_entry);
  }
}

TEST(ConSanMoi, CdnaRecordReplayEnablesAndCapturesEveryLaunchCoordinate) {
  for (const rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    for (const uint32_t dimension_mask : {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u}) {
      SCOPED_TRACE("arch=" + std::to_string(arch) +
                   " dimension_mask=" + std::to_string(dimension_mask));
      const auto guest =
          arch == ROCJITSU_CODE_ARCH_CDNA3
              ? build_cdna3_ds_store_b32(/*vaddr=*/0u, /*vdata=*/1u, /*byte_offset=*/0u, arch)
              : build_cdna4_ds_store_b32(/*vaddr=*/0u, /*vdata=*/1u, /*byte_offset=*/0u, arch);
      ASSERT_TRUE(guest);
      std::vector<uint32_t> text_words(320u, build_s_nop(0, arch));
      std::ranges::copy(*guest, text_words.begin());
      text_words.back() = build_s_endpgm(arch);
      std::vector<uint8_t> bytes =
          arch == ROCJITSU_CODE_ARCH_CDNA3
              ? make_cdna3_lds_code_object(text_words, "incomplete_launch_tuple",
                                           /*vgpr_granulated=*/0u,
                                           /*uses_dynamic_stack=*/false, dimension_mask)
              : make_cdna4_lds_code_object(text_words, "incomplete_launch_tuple",
                                           /*vgpr_granulated=*/0u,
                                           /*uses_dynamic_stack=*/false, dimension_mask);
      const bool has_guest_workgroup_info = dimension_mask == 2u;
      if (has_guest_workgroup_info) {
        mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
          AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2,
                          kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_INFO, 1u);
        });
      }
      ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
      options.moi_track_barriers = false;
      options.moi_track_atomics = false;
      options.moi_report_buffer_address = 0x123456780000ull;
      options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

      const ConSanResult result = try_patch_consan(bytes, options);

      ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
      ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
      AmdGpuCodeObject original(bytes.data(), bytes.size());
      AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
      ASSERT_TRUE(original.is_valid());
      ASSERT_TRUE(patched.is_valid());
      ASSERT_EQ(original.kernels().size(), 1u);
      ASSERT_EQ(patched.kernels().size(), 1u);
      KD original_descriptor{};
      KD patched_descriptor{};
      std::memcpy(&original_descriptor,
                  bytes.data() + original.kernels().front().descriptor_file_offset,
                  sizeof(original_descriptor));
      std::memcpy(&patched_descriptor,
                  result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
                  sizeof(patched_descriptor));
      EXPECT_NE(AMDHSA_BITS_GET(patched_descriptor.compute_pgm_rsrc2,
                                kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X),
                0u);
      EXPECT_NE(AMDHSA_BITS_GET(patched_descriptor.compute_pgm_rsrc2,
                                kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y),
                0u);
      EXPECT_NE(AMDHSA_BITS_GET(patched_descriptor.compute_pgm_rsrc2,
                                kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z),
                0u);

      const auto prologue =
          std::ranges::find(result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue,
                            &ConSanPatchInfo::kind);
      ASSERT_NE(prologue, result.patches.end());
      const std::vector<uint32_t> prologue_words =
          text_words_at_offset(patched, prologue->trampoline_offset, prologue->trampoline_size);
      const uint16_t full_payload_base = static_cast<uint16_t>(AMDHSA_BITS_GET(
          patched_descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT));
      if (result.resolved_moi_record_replay_workgroup_sgprs.complete()) {
        const std::array<uint16_t, 3> destinations = {
            *result.resolved_moi_record_replay_workgroup_sgprs.x,
            *result.resolved_moi_record_replay_workgroup_sgprs.y,
            *result.resolved_moi_record_replay_workgroup_sgprs.z,
        };
        for (uint16_t dimension = 0; dimension < 3u; ++dimension) {
          EXPECT_NE(
              std::ranges::find(
                  prologue_words,
                  build_s_mov_b32(destinations[dimension],
                                  static_cast<uint16_t>(full_payload_base + dimension), arch)),
              prologue_words.end());
        }
      } else {
        ASSERT_TRUE(prologue->persistent_record_replay_workgroup_vgprs.complete());
        const std::array<uint16_t, 3> destinations = {
            *prologue->persistent_record_replay_workgroup_vgprs.x,
            *prologue->persistent_record_replay_workgroup_vgprs.y,
            *prologue->persistent_record_replay_workgroup_vgprs.z,
        };
        for (uint16_t dimension = 0; dimension < 3u; ++dimension) {
          EXPECT_NE(
              std::ranges::find(
                  prologue_words,
                  build_v_mov_b32_e32(destinations[dimension],
                                      static_cast<uint16_t>(full_payload_base + dimension), arch)),
              prologue_words.end());
        }
      }

      uint16_t guest_destination = static_cast<uint16_t>(AMDHSA_BITS_GET(
          original_descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT));
      for (uint16_t dimension = 0; dimension < 3u; ++dimension) {
        if ((dimension_mask & (1u << dimension)) == 0u)
          continue;
        const uint16_t full_source = static_cast<uint16_t>(full_payload_base + dimension);
        if (guest_destination != full_source) {
          EXPECT_EQ(std::ranges::count(prologue_words,
                                       build_s_mov_b32(guest_destination, full_source, arch)),
                    1u)
              << "the semantic workgroup-payload repair must be the sole copy; a preceding "
                 "positional system-SGPR repair would consume overlapping sources";
        }
        ++guest_destination;
      }
      if (has_guest_workgroup_info) {
        EXPECT_NE(
            std::ranges::find(prologue_words,
                              build_s_mov_b32(guest_destination,
                                              static_cast<uint16_t>(full_payload_base + 3u), arch)),
            prologue_words.end());
      }
    }
  }
}

TEST(ConSanMoi, RecordReplayRejectsPartialPersistentOwnerEpochOverride) {
  std::vector<uint32_t> text_words(320u, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000000u; // ds_store_b32 v0, v0
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);
  for (bool owner_only : {true, false}) {
    SCOPED_TRACE(owner_only ? "owner only" : "epoch only");
    ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
    if (owner_only)
      options.moi_owner_vgpr = 14u;
    else
      options.moi_epoch_vgpr = 15u;
    options.moi_report_buffer_address = 0x123456780000ull;
    options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

    const ConSanResult result =
        try_patch_consan(make_rdna4_lds_code_object(text_words, "partial_owner_epoch"), options);

    EXPECT_EQ(result.outcome, ConSanTransformOutcome::Unsupported);
    EXPECT_FALSE(result.modified);
    EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
      return warning.find("persistent VGPR overrides require both owner and epoch registers") !=
             std::string::npos;
    })) << testing::PrintToString(result.warnings);
  }
}

TEST(ConSanMoi, AutoRecordReplayBarrierRecordsCapturedWorkgroupIdentity) {
  const std::array<uint32_t, 5> text_words = {
      0xD8340000u, 0x00000000u, // ds_store_b32
      0xBE804EC1u,              // s_barrier_signal -1
      0xBF94FFFFu,              // s_barrier_wait -1
      0xBFB00000u,              // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "record_replay_exact_barrier_tuple", kRdna4Wave64AllVgprsGranulated,
      /*wave32=*/false, /*uses_dynamic_stack=*/false,
      /*workgroup_id_dimension_mask=*/7u);
  const ConSanMoiAutoReportPlan report_plan = plan_consan_moi_auto_report(
      {.engine = ConSanMoiEngine::RecordReplay, .access_range_count = 1, .barrier_event_count = 1});
  ASSERT_TRUE(report_plan.complete());
  const auto layout_override = consan_moi_auto_report_layout_override(report_plan);
  ASSERT_TRUE(layout_override);

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = report_plan.required_bytes;
  options.moi_report_layout = *layout_override;
  options.moi_track_barriers = true;
  options.max_patches = 4u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.resolved_moi_record_replay_workgroup_vgprs.complete());
  const auto barrier = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord, &ConSanPatchInfo::kind);
  ASSERT_NE(barrier, result.patches.end());
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> barrier_words =
      text_words_at_offset(patched, barrier->trampoline_offset, barrier->trampoline_size);
  // Access and barrier records both use the exact tuple captured at entry;
  // the barrier must not reload launch TTMPs at this arbitrary post-entry
  // site.
  for (uint16_t destination = 0; destination < 16u; ++destination) {
    const uint32_t transient_x = build_v_mov_b32_e32(
        destination, ttmp_scalar_operand(kTtmpRdna4GridX), ROCJITSU_CODE_ARCH_RDNA4);
    EXPECT_EQ(std::ranges::find(barrier_words, transient_x), barrier_words.end());
  }
}

TEST(ConSanMoi, RecordReplayExcludesUnreachableTailOfFinalZeroSizedSymbol) {
  constexpr auto live_store = cdna5::build_vds(cdna5::kDsStoreB16Vds, {.addr = 4u, .data0 = 5u});
  constexpr auto dead_load = cdna5::build_vds(cdna5::kDsLoadB32Vds, {.addr = 6u, .vdst = 7u});
  const std::array<uint32_t, 6> text_words = {
      live_store[0], live_store[1], build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5),
      dead_load[0],  dead_load[1],  build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5),
  };
  std::vector<uint8_t> bytes = make_gfx1250_code_object(text_words, "zero_sized_final_symbol");
  mutate_elf_symbol(bytes, 1u, [](Elf64_Sym &symbol) { symbol.st_size = 0; });

  const ConSanResult result = try_patch_consan(bytes, moi_options(ConSanMoiEngine::RecordReplay));

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_EQ(result.kernels.size(), 1u);
  EXPECT_TRUE(result.kernels.front().code_size_inferred_from_zero);
  ASSERT_EQ(result.moi_candidates.size(), 1u);
  EXPECT_EQ(result.moi_candidates.front().mnemonic, "ds_store_b16");
  EXPECT_EQ(result.moi_candidates.front().text_offset, 0u);
  const auto access_dispositions =
      std::ranges::count_if(result.site_dispositions, [](const ConSanSiteDispositionRecord &site) {
        return site.site_kind == ConSanResourceSiteKind::Access;
      });
  EXPECT_EQ(access_dispositions, 1u);
  EXPECT_EQ(result.site_dispositions.front().disposition, ConSanSiteDisposition::Supported);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().reason, ConSanRegisterPlanReason::None);
  EXPECT_EQ(result.resource_plans.front().owner_descriptor_file_offsets.size(), 1u);
}

TEST(ConSanMoi, RecordReplayExcludesUnreachableTailOfBoundedZeroSizedSymbol) {
  constexpr auto live_store = cdna5::build_vds(cdna5::kDsStoreB16Vds, {.addr = 4u, .data0 = 5u});
  constexpr auto dead_load = cdna5::build_vds(cdna5::kDsLoadB32Vds, {.addr = 6u, .vdst = 7u});
  const std::array<uint32_t, 6> first_kernel_words = {
      live_store[0], live_store[1], build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5),
      dead_load[0],  dead_load[1],  build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5),
  };
  const std::array<uint32_t, 1> second_kernel_words = {
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5),
  };
  std::vector<uint8_t> bytes = make_gfx1250_code_object_with_local_function(
      first_kernel_words, second_kernel_words, {}, kRdna4Wave64AllVgprsGranulated,
      /*function_is_kernel=*/true);
  mutate_elf_symbol(bytes, 1u, [](Elf64_Sym &symbol) { symbol.st_size = 0; });

  const ConSanResult result = try_patch_consan(bytes, moi_options(ConSanMoiEngine::RecordReplay));

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_EQ(result.kernels.size(), 2u);
  const auto first_kernel = std::ranges::find(result.kernels, "lds_probe", &ConSanKernelInfo::name);
  ASSERT_NE(first_kernel, result.kernels.end());
  EXPECT_TRUE(first_kernel->code_size_inferred_from_zero);
  ASSERT_EQ(result.moi_candidates.size(), 1u);
  EXPECT_EQ(result.moi_candidates.front().container_name, "lds_probe");
  EXPECT_EQ(result.moi_candidates.front().mnemonic, "ds_store_b16");
  EXPECT_EQ(result.moi_candidates.front().text_offset, 0u);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().reason, ConSanRegisterPlanReason::None);
}

TEST(ConSanMoi, RecordReplayDoesNotPruneExplicitSizedUnreachableTail) {
  constexpr auto live_store = cdna5::build_vds(cdna5::kDsStoreB16Vds, {.addr = 4u, .data0 = 5u});
  constexpr auto dead_load = cdna5::build_vds(cdna5::kDsLoadB32Vds, {.addr = 6u, .vdst = 7u});
  const std::array<uint32_t, 6> text_words = {
      live_store[0], live_store[1], build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5),
      dead_load[0],  dead_load[1],  build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5),
  };
  const std::vector<uint8_t> bytes =
      make_gfx1250_code_object(text_words, "explicit_sized_unreachable_tail");

  const ConSanResult result = try_patch_consan(bytes, moi_options(ConSanMoiEngine::RecordReplay));

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_EQ(result.kernels.size(), 1u);
  EXPECT_FALSE(result.kernels.front().code_size_inferred_from_zero);
  ASSERT_EQ(result.moi_candidates.size(), 2u);
  EXPECT_EQ(result.moi_candidates[0].mnemonic, "ds_store_b16");
  EXPECT_EQ(result.moi_candidates[1].mnemonic, "ds_load_b32");
  ASSERT_EQ(result.resource_plans.size(), 2u);
  EXPECT_EQ(result.resource_plans[0].reason, ConSanRegisterPlanReason::None);
  EXPECT_EQ(result.resource_plans[1].reason, ConSanRegisterPlanReason::MissingOwner);
}

TEST(ConSanMoi, RecordReplayIgnoresUnpublishedSparseAtomicAndFenceSlots) {
  std::array<ConSanMoiRecordReplayAtomicEvent, 2> atomics{};
  atomics[0].kind = static_cast<ConSanMoiAtomicEventKind>(0);
  atomics[0].operation = static_cast<ConSanMoiAtomicOperation>(0);
  atomics[1].generation = 7;
  atomics[1].atomic_address = 0x4000;
  atomics[1].instruction_offset = 0x100;
  atomics[1].event_index = 1;
  atomics[1].kind = ConSanMoiAtomicEventKind::Release;

  std::array<ConSanMoiRecordReplayFenceEvent, 2> fences{};
  fences[0].kind = static_cast<ConSanMoiFenceEventKind>(0);
  fences[1].generation = 7;
  fences[1].instruction_offset = 0x200;
  fences[1].event_index = 2;
  fences[1].kind = ConSanMoiFenceEventKind::Release;
  fences[1].scope = 1;
  fences[1].communication_token = 0x5000;

  std::array<ConSanMoiRecordReplayPcEntry, 2> dictionary{};
  std::array<ConSanMoiRecordReplayWorkgroupRun, 1> runs{};
  std::array<ConSanMoiRecordReplayCompactEvent, 2> events{};
  const auto trace = consan_moi_compact_record_replay_trace(
      7, 11, std::span<const ConSanMoiAccessRecord>{}, std::span<const ConSanMoiBarrierRecord>{},
      atomics, fences, dictionary, runs, events);
  EXPECT_EQ(trace.flags, 0u);
  EXPECT_EQ(trace.rejected_event_count, 0u);
  EXPECT_EQ(trace.dictionary_count, 2u);
  EXPECT_EQ(trace.event_count, 2u);

  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/0,
      /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/0,
      /*sampled_watchpoint_capacity=*/0);
  std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
  const ConSanMoiRecordReplayResult replay = consan_moi_record_replay_access_records(
      header, std::span<const ConSanMoiAccessRecord>{}, std::span<const ConSanMoiBarrierRecord>{},
      atomics, fences, diagnostics, std::span<uint64_t>{});
  EXPECT_EQ(replay.processed_atomic_count, 1u);
  EXPECT_EQ(replay.unsupported_atomic_count, 0u);
  EXPECT_EQ(replay.processed_fence_count, 1u);
  EXPECT_EQ(replay.unsupported_fence_count, 0u);
  EXPECT_FALSE(replay.metadata_full);
}

TEST(ConSanMoi, ReportAbiHeaderCarriesVersionedLayout) {
  constexpr ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7,
      /*dispatch_id=*/9,
      /*access_record_capacity=*/11,
      /*diagnostic_capacity=*/13,
      /*exact_shadow_entry_capacity=*/17,
      /*sampled_watchpoint_capacity=*/19,
      /*barrier_record_capacity=*/23,
      /*atomic_record_capacity=*/29,
      /*inline_atomic_release_capacity=*/31,
      /*fence_record_capacity=*/0,
      /*inline_acquired_epoch_token_capacity=*/31,
      /*inline_causal_snapshot_capacity=*/31, ConSanMoiEngine::InlineShadow);

  EXPECT_EQ(header.magic, kConSanMoiReportMagic);
  EXPECT_EQ(header.abi_version, kConSanMoiReportAbiVersion);
  EXPECT_EQ(header.header_size, sizeof(ConSanMoiReportHeader));
  EXPECT_EQ(header.generation, 7u);
  EXPECT_EQ(header.dispatch_id, 9u);
  EXPECT_EQ(header.engine, static_cast<uint32_t>(ConSanMoiEngine::InlineShadow));
  EXPECT_EQ(header.layout_flags, kConSanMoiReportKnownLayoutFlags);
  EXPECT_EQ(header.access_record_capacity, 11u);
  EXPECT_EQ(header.barrier_record_capacity, 23u);
  EXPECT_EQ(header.atomic_record_capacity, 29u);
  EXPECT_EQ(header.diagnostic_capacity, 13u);
  EXPECT_EQ(header.exact_shadow_entry_capacity, 17u);
  EXPECT_EQ(header.sampled_watchpoint_capacity, 19u);
  EXPECT_EQ(header.sampled_sync_metadata_capacity, 19u);
  EXPECT_EQ(header.sampled_pending_acquire_capacity, 19u);
  EXPECT_EQ(header.access_record_count, 0u);
  EXPECT_EQ(header.barrier_record_count, 0u);
  EXPECT_EQ(header.atomic_record_count, 0u);
  EXPECT_EQ(header.diagnostic_count, 0u);
  EXPECT_EQ(header.event_counter, 0u);
  EXPECT_EQ(header.inline_atomic_release_capacity, 31u);
  EXPECT_EQ(header.inline_acquired_epoch_token_capacity, 31u);
  EXPECT_EQ(header.inline_causal_snapshot_capacity, 31u);
  EXPECT_TRUE(consan_moi_report_header_is_current(header));
  ConSanMoiReportHeader stale_v4 = header;
  stale_v4.abi_version = 4;
  EXPECT_FALSE(consan_moi_report_header_is_current(stale_v4));
  ConSanMoiReportHeader stale_v5 = header;
  stale_v5.abi_version = 5;
  EXPECT_FALSE(consan_moi_report_header_is_current(stale_v5));
  ConSanMoiReportHeader short_v6 = header;
  short_v6.header_size -= sizeof(uint32_t);
  EXPECT_FALSE(consan_moi_report_header_is_current(short_v6));
  ConSanMoiReportHeader unknown_engine = header;
  unknown_engine.engine = 99;
  EXPECT_FALSE(consan_moi_report_header_is_current(unknown_engine));
  ConSanMoiReportHeader unknown_layout = header;
  unknown_layout.layout_flags |= 1u << 31u;
  EXPECT_FALSE(consan_moi_report_header_is_current(unknown_layout));

  constexpr size_t expected_bytes =
      sizeof(ConSanMoiReportHeader) + 11u * sizeof(ConSanMoiAccessRecord) +
      23u * sizeof(ConSanMoiBarrierRecord) + 29u * sizeof(ConSanMoiAtomicRecord) +
      13u * sizeof(ConSanMoiDiagnosticRecord) + 17u * sizeof(uint64_t) +
      19u * (sizeof(uint64_t) + sizeof(ConSanMoiSampledSyncMetadataPacked) +
             sizeof(ConSanMoiSampledPendingAcquireSlot));
  EXPECT_EQ(consan_moi_report_buffer_min_bytes(11, 13, 17, 19, 23, 29), expected_bytes);

  constexpr ConSanMoiReportHeader fence_header =
      make_consan_moi_report_header(7, 9, 2, 0, 0, 0, 0, 2, 0, 2);
  EXPECT_EQ(fence_header.fence_record_capacity, 2u);
  EXPECT_EQ(fence_header.fence_record_count, 0u);

  EXPECT_EQ(consan_moi_default_auto_report_buffer_size(ConSanMoiEngine::RecordReplay), 64u * 1024u);
  EXPECT_EQ(consan_moi_default_auto_report_buffer_size(ConSanMoiEngine::Sampled), 64u * 1024u);
  EXPECT_EQ(consan_moi_default_auto_report_buffer_size(ConSanMoiEngine::InlineShadow),
            512u * 1024u);

  constexpr ConSanMoiReportBufferLayout default_record_layout =
      consan_moi_report_buffer_layout_for_bytes(
          consan_moi_default_auto_report_buffer_size(ConSanMoiEngine::RecordReplay), true, true);
  EXPECT_GT(default_record_layout.access_record_capacity, 0u);
  EXPECT_EQ(default_record_layout.access_record_capacity,
            default_record_layout.barrier_record_capacity);
  EXPECT_EQ(default_record_layout.access_record_capacity,
            default_record_layout.atomic_record_capacity);

  constexpr ConSanMoiReportBufferLayout fence_layout = consan_moi_report_buffer_layout_for_bytes(
      consan_moi_report_buffer_min_bytes(2, 0, 0, 0, 0, 2, 2),
      /*include_barriers=*/false, /*include_atomics=*/true, /*include_fences=*/true);
  EXPECT_EQ(fence_layout.access_record_capacity, 2u);
  EXPECT_EQ(fence_layout.atomic_record_capacity, 2u);
  EXPECT_EQ(fence_layout.fence_record_capacity, 2u);
  EXPECT_EQ(fence_layout.fence_records_offset,
            fence_layout.atomic_records_offset + 2u * sizeof(ConSanMoiAtomicRecord));
  EXPECT_EQ(fence_layout.diagnostic_records_offset,
            fence_layout.fence_records_offset + 2u * sizeof(ConSanMoiFenceRecord));

  constexpr ConSanMoiReportBufferLayout default_sampled_layout =
      consan_moi_direct_sampled_report_buffer_layout_for_bytes(
          consan_moi_default_auto_report_buffer_size(ConSanMoiEngine::Sampled));
  EXPECT_GT(default_sampled_layout.sampled_watchpoint_capacity, 0u);
  EXPECT_EQ(default_sampled_layout.barrier_record_capacity, 0u);
  EXPECT_EQ(default_sampled_layout.atomic_record_capacity, 0u);
  EXPECT_TRUE(consan_moi_report_layout_has_required_capacities(
      default_sampled_layout, ConSanMoiEngine::Sampled, /*track_barriers=*/true,
      /*track_atomics=*/false));
  EXPECT_TRUE(consan_moi_report_layout_has_required_capacities(
      default_sampled_layout, ConSanMoiEngine::Sampled, /*track_barriers=*/false,
      /*track_atomics=*/true));

  constexpr ConSanMoiReportBufferLayout default_inline_layout =
      consan_moi_inline_shadow_report_buffer_layout_for_bytes(
          consan_moi_default_auto_report_buffer_size(ConSanMoiEngine::InlineShadow));
  EXPECT_EQ(default_inline_layout.diagnostic_capacity,
            kConSanMoiInlineShadowDefaultDiagnosticCapacity);
  EXPECT_GE(default_inline_layout.exact_shadow_entry_capacity,
            kConSanMoiInlineShadowConservativeExactShadowEntries);

  constexpr ConSanMoiReportBufferLayout access_only_layout =
      consan_moi_report_buffer_layout_for_bytes(consan_moi_report_buffer_min_bytes(5, 0, 0, 0),
                                                /*include_barriers=*/false);
  EXPECT_EQ(access_only_layout.access_record_capacity, 5u);
  EXPECT_EQ(access_only_layout.barrier_record_capacity, 0u);
  EXPECT_EQ(access_only_layout.atomic_record_capacity, 0u);
  EXPECT_EQ(access_only_layout.diagnostic_capacity, 0u);
  EXPECT_EQ(access_only_layout.exact_shadow_entry_capacity, 0u);
  EXPECT_EQ(access_only_layout.sampled_watchpoint_capacity, 0u);
  EXPECT_EQ(access_only_layout.access_records_offset, sizeof(ConSanMoiReportHeader));
  EXPECT_EQ(access_only_layout.barrier_records_offset,
            sizeof(ConSanMoiReportHeader) + 5u * sizeof(ConSanMoiAccessRecord));
  EXPECT_EQ(access_only_layout.atomic_records_offset, access_only_layout.barrier_records_offset);
  EXPECT_EQ(access_only_layout.diagnostic_records_offset, access_only_layout.atomic_records_offset);
  EXPECT_EQ(access_only_layout.exact_shadow_entries_offset,
            access_only_layout.diagnostic_records_offset);
  EXPECT_EQ(access_only_layout.inline_atomic_release_slots_offset,
            access_only_layout.exact_shadow_entries_offset);
  EXPECT_EQ(access_only_layout.sampled_watchpoints_offset,
            access_only_layout.exact_shadow_entries_offset);

  constexpr ConSanMoiReportBufferLayout barrier_layout = consan_moi_report_buffer_layout_for_bytes(
      consan_moi_report_buffer_min_bytes(3, 0, 0, 0, 3), /*include_barriers=*/true);
  EXPECT_EQ(barrier_layout.access_record_capacity, 3u);
  EXPECT_EQ(barrier_layout.barrier_record_capacity, 3u);
  EXPECT_EQ(barrier_layout.atomic_record_capacity, 0u);
  EXPECT_EQ(barrier_layout.diagnostic_capacity, 0u);
  EXPECT_EQ(barrier_layout.exact_shadow_entry_capacity, 0u);
  EXPECT_EQ(barrier_layout.sampled_watchpoint_capacity, 0u);
  EXPECT_EQ(barrier_layout.access_records_offset, sizeof(ConSanMoiReportHeader));
  EXPECT_EQ(barrier_layout.barrier_records_offset,
            sizeof(ConSanMoiReportHeader) + 3u * sizeof(ConSanMoiAccessRecord));
  EXPECT_EQ(barrier_layout.atomic_records_offset,
            barrier_layout.barrier_records_offset + 3u * sizeof(ConSanMoiBarrierRecord));
  EXPECT_EQ(barrier_layout.diagnostic_records_offset, barrier_layout.atomic_records_offset);
  EXPECT_EQ(barrier_layout.exact_shadow_entries_offset, barrier_layout.diagnostic_records_offset);
  EXPECT_EQ(barrier_layout.inline_atomic_release_slots_offset,
            barrier_layout.exact_shadow_entries_offset);
  EXPECT_EQ(barrier_layout.sampled_watchpoints_offset, barrier_layout.exact_shadow_entries_offset);

  constexpr ConSanMoiReportBufferLayout atomic_layout = consan_moi_report_buffer_layout_for_bytes(
      consan_moi_report_buffer_min_bytes(2, 0, 0, 0, 0, 2),
      /*include_barriers=*/false,
      /*include_atomics=*/true);
  EXPECT_EQ(atomic_layout.access_record_capacity, 2u);
  EXPECT_EQ(atomic_layout.barrier_record_capacity, 0u);
  EXPECT_EQ(atomic_layout.atomic_record_capacity, 2u);
  EXPECT_EQ(atomic_layout.diagnostic_capacity, 0u);
  EXPECT_EQ(atomic_layout.exact_shadow_entry_capacity, 0u);
  EXPECT_EQ(atomic_layout.sampled_watchpoint_capacity, 0u);
  EXPECT_EQ(atomic_layout.access_records_offset, sizeof(ConSanMoiReportHeader));
  EXPECT_EQ(atomic_layout.barrier_records_offset,
            sizeof(ConSanMoiReportHeader) + 2u * sizeof(ConSanMoiAccessRecord));
  EXPECT_EQ(atomic_layout.atomic_records_offset, atomic_layout.barrier_records_offset);
  EXPECT_EQ(atomic_layout.diagnostic_records_offset,
            atomic_layout.atomic_records_offset + 2u * sizeof(ConSanMoiAtomicRecord));
  EXPECT_EQ(atomic_layout.exact_shadow_entries_offset, atomic_layout.diagnostic_records_offset);
  EXPECT_EQ(atomic_layout.inline_atomic_release_slots_offset,
            atomic_layout.exact_shadow_entries_offset);
  EXPECT_EQ(atomic_layout.sampled_watchpoints_offset, atomic_layout.exact_shadow_entries_offset);

  constexpr ConSanMoiReportBufferLayout combined_layout = consan_moi_report_buffer_layout_for_bytes(
      consan_moi_report_buffer_min_bytes(4, 0, 0, 0, 4, 4),
      /*include_barriers=*/true,
      /*include_atomics=*/true);
  EXPECT_EQ(combined_layout.access_record_capacity, 4u);
  EXPECT_EQ(combined_layout.barrier_record_capacity, 4u);
  EXPECT_EQ(combined_layout.atomic_record_capacity, 4u);
  EXPECT_EQ(combined_layout.diagnostic_capacity, 0u);
  EXPECT_EQ(combined_layout.exact_shadow_entry_capacity, 0u);
  EXPECT_EQ(combined_layout.sampled_watchpoint_capacity, 0u);
  EXPECT_EQ(combined_layout.access_records_offset, sizeof(ConSanMoiReportHeader));
  EXPECT_EQ(combined_layout.barrier_records_offset,
            sizeof(ConSanMoiReportHeader) + 4u * sizeof(ConSanMoiAccessRecord));
  EXPECT_EQ(combined_layout.atomic_records_offset,
            combined_layout.barrier_records_offset + 4u * sizeof(ConSanMoiBarrierRecord));
  EXPECT_EQ(combined_layout.diagnostic_records_offset,
            combined_layout.atomic_records_offset + 4u * sizeof(ConSanMoiAtomicRecord));
  EXPECT_EQ(combined_layout.exact_shadow_entries_offset, combined_layout.diagnostic_records_offset);
  EXPECT_EQ(combined_layout.inline_atomic_release_slots_offset,
            combined_layout.exact_shadow_entries_offset);
  EXPECT_EQ(combined_layout.sampled_watchpoints_offset,
            combined_layout.exact_shadow_entries_offset);

  constexpr ConSanMoiReportBufferLayout direct_sampled_layout =
      consan_moi_direct_sampled_report_buffer_layout_for_bytes(
          sizeof(ConSanMoiReportHeader) +
          6u * (sizeof(ConSanMoiSampledCausalWindow) + sizeof(uint64_t) +
                sizeof(ConSanMoiSampledSyncMetadataPacked) +
                sizeof(ConSanMoiSampledPendingAcquireSlot)));
  EXPECT_EQ(direct_sampled_layout.access_record_capacity, 0u);
  EXPECT_EQ(direct_sampled_layout.barrier_record_capacity, 0u);
  EXPECT_EQ(direct_sampled_layout.atomic_record_capacity, 0u);
  EXPECT_EQ(direct_sampled_layout.diagnostic_capacity, 0u);
  EXPECT_EQ(direct_sampled_layout.exact_shadow_entry_capacity, 0u);
  EXPECT_EQ(direct_sampled_layout.sampled_watchpoint_capacity, 6u);
  EXPECT_EQ(direct_sampled_layout.sampled_causal_window_capacity, 6u);
  EXPECT_EQ(direct_sampled_layout.sampled_sync_metadata_capacity, 6u);
  EXPECT_EQ(direct_sampled_layout.sampled_pending_acquire_capacity, 6u);
  EXPECT_EQ(direct_sampled_layout.diagnostic_records_offset, sizeof(ConSanMoiReportHeader));
  EXPECT_EQ(direct_sampled_layout.exact_shadow_entries_offset, sizeof(ConSanMoiReportHeader));
  EXPECT_EQ(direct_sampled_layout.inline_atomic_release_slots_offset,
            sizeof(ConSanMoiReportHeader));
  EXPECT_EQ(direct_sampled_layout.sampled_causal_windows_offset, sizeof(ConSanMoiReportHeader));
  EXPECT_EQ(direct_sampled_layout.sampled_watchpoints_offset,
            sizeof(ConSanMoiReportHeader) + 6u * sizeof(ConSanMoiSampledCausalWindow));
  EXPECT_EQ(direct_sampled_layout.sampled_sync_metadata_offset,
            direct_sampled_layout.sampled_watchpoints_offset + 6u * sizeof(uint64_t));
  EXPECT_EQ(direct_sampled_layout.sampled_pending_acquires_offset,
            direct_sampled_layout.sampled_sync_metadata_offset +
                6u * sizeof(ConSanMoiSampledSyncMetadataPacked));
  constexpr uint64_t sampled_slot_bytes = sizeof(ConSanMoiSampledCausalWindow) + sizeof(uint64_t) +
                                          sizeof(ConSanMoiSampledSyncMetadataPacked) +
                                          sizeof(ConSanMoiSampledPendingAcquireSlot);
  constexpr auto exact_one_sampled_slot = consan_moi_direct_sampled_report_buffer_layout_for_bytes(
      sizeof(ConSanMoiReportHeader) + sampled_slot_bytes);
  constexpr auto truncated_sampled_slot = consan_moi_direct_sampled_report_buffer_layout_for_bytes(
      sizeof(ConSanMoiReportHeader) + sampled_slot_bytes - 1u);
  EXPECT_EQ(exact_one_sampled_slot.sampled_sync_metadata_capacity, 1u);
  EXPECT_EQ(truncated_sampled_slot.sampled_watchpoint_capacity, 0u);
  EXPECT_EQ(truncated_sampled_slot.sampled_causal_window_capacity, 0u);
  EXPECT_EQ(truncated_sampled_slot.sampled_sync_metadata_capacity, 0u);
  EXPECT_EQ(truncated_sampled_slot.sampled_pending_acquire_capacity, 0u);

  constexpr ConSanMoiReportBufferLayout inline_shadow_layout =
      consan_moi_inline_shadow_report_buffer_layout_for_bytes(
          sizeof(ConSanMoiReportHeader) + 4u * sizeof(ConSanMoiDiagnosticRecord) +
          sizeof(ConSanMoiInlineAtomicReleaseSlot) + sizeof(ConSanMoiInlineCausalSnapshot) +
          sizeof(ConSanMoiInlineAcquiredEpochTokenSlot) +
          32u * sizeof(ConSanMoiInlineExactShadowSlot));
  EXPECT_EQ(inline_shadow_layout.access_record_capacity, 0u);
  EXPECT_EQ(inline_shadow_layout.barrier_record_capacity, 0u);
  EXPECT_EQ(inline_shadow_layout.atomic_record_capacity, 0u);
  EXPECT_EQ(inline_shadow_layout.diagnostic_capacity, 4u);
  EXPECT_EQ(inline_shadow_layout.exact_shadow_entry_capacity, 32u);
  EXPECT_EQ(inline_shadow_layout.inline_atomic_release_capacity, 1u);
  EXPECT_EQ(inline_shadow_layout.inline_acquired_epoch_token_capacity, 1u);
  EXPECT_EQ(inline_shadow_layout.inline_causal_snapshot_capacity, 1u);
  EXPECT_EQ(inline_shadow_layout.sampled_watchpoint_capacity, 0u);
  EXPECT_EQ(inline_shadow_layout.diagnostic_records_offset, sizeof(ConSanMoiReportHeader));
  EXPECT_EQ(inline_shadow_layout.exact_shadow_entries_offset,
            sizeof(ConSanMoiReportHeader) + 4u * sizeof(ConSanMoiDiagnosticRecord));
  EXPECT_EQ(inline_shadow_layout.inline_atomic_release_slots_offset,
            inline_shadow_layout.exact_shadow_entries_offset +
                32u * sizeof(ConSanMoiInlineExactShadowSlot));
  EXPECT_EQ(inline_shadow_layout.inline_causal_snapshots_offset,
            inline_shadow_layout.inline_atomic_release_slots_offset +
                sizeof(ConSanMoiInlineAtomicReleaseSlot));
  EXPECT_EQ(inline_shadow_layout.inline_acquired_epoch_token_slots_offset,
            inline_shadow_layout.inline_causal_snapshots_offset +
                sizeof(ConSanMoiInlineCausalSnapshot));
  EXPECT_EQ(inline_shadow_layout.sampled_watchpoints_offset,
            inline_shadow_layout.inline_acquired_epoch_token_slots_offset +
                sizeof(ConSanMoiInlineAcquiredEpochTokenSlot));

  constexpr ConSanMoiReportBufferLayout small_inline_shadow_layout =
      consan_moi_inline_shadow_report_buffer_layout_for_bytes(sizeof(ConSanMoiReportHeader) +
                                                              sizeof(ConSanMoiDiagnosticRecord));
  EXPECT_EQ(small_inline_shadow_layout.diagnostic_capacity, 0u);
  EXPECT_EQ(small_inline_shadow_layout.inline_atomic_release_capacity, 0u);
  EXPECT_EQ(small_inline_shadow_layout.inline_acquired_epoch_token_capacity, 0u);
  EXPECT_EQ(small_inline_shadow_layout.inline_causal_snapshot_capacity, 0u);
  EXPECT_EQ(small_inline_shadow_layout.exact_shadow_entry_capacity,
            sizeof(ConSanMoiDiagnosticRecord) / sizeof(ConSanMoiInlineExactShadowSlot));
}

TEST(ConSanMoi, FirstLightProbeAutomaticallyUsesDeadVgprs) {
  std::array<uint32_t, 320> text_words{};
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000000u; // ds_store_b32 v0, v0
  for (size_t i = 2; i + 1 < text_words.size(); ++i)
    text_words[i] = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
  text_words.back() = 0xBFB00000u; // s_endpgm

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options = moi_options();
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().source, ConSanRegisterAllocationSource::LivenessDead);
  EXPECT_EQ(result.resource_plans.front().scratch_vgpr, 1);
  ASSERT_EQ(result.site_dispositions.size(), 1u);
  EXPECT_EQ(result.site_dispositions.front().lowering_outcome, ConSanSiteLoweringOutcome::Patched);
  EXPECT_EQ(result.site_dispositions.front().lowering_reason, ConSanSiteLoweringReason::None);
  EXPECT_EQ(result.site_dispositions.front().resource_reason, ConSanRegisterPlanReason::None);
  ASSERT_EQ(non_entry_prologue_patch_count(result), 1u);
  EXPECT_EQ(only_non_entry_prologue_patch(result).kind,
            ConSanPatchKind::InlineMoiAccessRecordStore);
  EXPECT_EQ(only_non_entry_prologue_patch(result).scratch_vgpr, 1);
}

TEST(ConSanMoi, Cdna4GprIndexedKernelSpillsInsteadOfTrustingPointLiveness) {
  const auto guest = build_cdna4_ds_store_b32(
      /*vaddr=*/38, /*vdata=*/50, /*byte_offset=*/0, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(guest);
  std::array<uint32_t, 420> text_words{};
  text_words[0] = 0xBF11082Eu; // s_set_gpr_idx_on s46, gpr_idx(DST)
  text_words[1] = 0x7E000332u; // v_mov_b32_e32 v0, v50 (indexed destination)
  text_words[2] = 0xBF9C0000u; // s_set_gpr_idx_off
  std::copy(guest->begin(), guest->end(), text_words.begin() + 3);
  for (size_t i = 5; i + 1 < text_words.size(); ++i)
    text_words[i] = build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);

  constexpr uint32_t kCdna4Wave64AllVgprsGranulated = 31u;
  const std::vector<uint8_t> bytes = make_cdna4_lds_code_object(
      text_words, "gpr_indexed_record_replay", kCdna4Wave64AllVgprsGranulated);
  ConSanOptions options = moi_options();
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result))
      << "warnings=" << testing::PrintToString(result.warnings)
      << " errors=" << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  const ConSanCandidateResourcePlan &plan = result.resource_plans.front();
  EXPECT_TRUE(plan.has_indirect_vgpr_access);
  EXPECT_EQ(plan.source, ConSanRegisterAllocationSource::SpillRequired);
  const auto patch = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore, &ConSanPatchInfo::kind);
  ASSERT_NE(patch, result.patches.end());
  EXPECT_GT(patch->spilled_vgpr_count, 0u);
}

TEST(ConSanMoi, FirstLightProbeAutomaticallyGrowsOwningDescriptor) {
  std::array<uint32_t, 170> text_words{};
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000102u; // ds_store_b32 v2, v1
  for (size_t i = 2; i + 1 < text_words.size(); ++i)
    text_words[i] = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
  text_words.back() = 0xBFB00000u; // s_endpgm

  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT, 0);
  });
  ConSanOptions options = moi_options();
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  const ConSanCandidateResourcePlan &plan = result.resource_plans.front();
  EXPECT_EQ(plan.source, ConSanRegisterAllocationSource::DescriptorGrowth);
  EXPECT_EQ(plan.current_vgpr_count, 4);
  EXPECT_EQ(plan.max_referenced_vgpr_count, 3);
  EXPECT_EQ(plan.scratch_vgpr, 4);
  EXPECT_EQ(plan.required_vgpr_count, 10);
  EXPECT_EQ(result.resource_plan_summary.descriptor_growth_plans, 1u);
  ASSERT_EQ(non_entry_prologue_patch_count(result), 1u);
  EXPECT_EQ(only_non_entry_prologue_patch(result).scratch_vgpr, 4);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.kernels().size(), 1u);
  const uint64_t descriptor_offset = patched.kernels().front().descriptor_file_offset;
  KD descriptor{};
  std::memcpy(&descriptor, result.elf_bytes.data() + descriptor_offset, sizeof(descriptor));
  EXPECT_EQ(AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc1,
                            kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT),
            2u);
}

TEST(ConSanMoi, FirstLightProbeSpillsVictimWindowInAppendedCave) {
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
  ConSanOptions options = moi_options();
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().source, ConSanRegisterAllocationSource::SpillRequired);
  EXPECT_EQ(result.resource_plans.front().scratch_vgpr, 1);
  ASSERT_EQ(non_entry_prologue_patch_count(result), 1u);
  const ConSanPatchInfo &patch = only_non_entry_prologue_patch(result);
  EXPECT_EQ(patch.kind, ConSanPatchKind::TrampolineMoiAccessRecordStore);
  EXPECT_EQ(patch.scratch_vgpr, 1);
  EXPECT_EQ(patch.spilled_vgpr_count, 6u);
  EXPECT_EQ(patch.required_private_segment_size, 56u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const std::vector<uint32_t> trampoline_words =
      text_words_at_offset(patched, patch.trampoline_offset, patch.trampoline_size);
  const std::vector<uint32_t> save =
      expected_vgpr_spill_words(1, 6, /*restore=*/false, /*slot_base=*/32);
  const std::vector<uint32_t> restore =
      expected_vgpr_spill_words(1, 6, /*restore=*/true, /*slot_base=*/32);
  ASSERT_FALSE(save.empty());
  ASSERT_FALSE(restore.empty());
  const auto save_at = std::ranges::search(trampoline_words, save).begin();
  const auto guest_at =
      std::ranges::search(trampoline_words,
                          std::span<const uint32_t>(text_words.data(), text_words.size() - 1u))
          .begin();
  const auto restore_at = std::ranges::search(trampoline_words, restore).begin();
  ASSERT_NE(save_at, trampoline_words.end());
  ASSERT_NE(guest_at, trampoline_words.end());
  ASSERT_NE(restore_at, trampoline_words.end());
  EXPECT_LT(save_at, guest_at);
  EXPECT_LT(guest_at, restore_at);

  ASSERT_EQ(patched.kernels().size(), 1u);
  KD descriptor{};
  const uint64_t descriptor_offset = patched.kernels().front().descriptor_file_offset;
  std::memcpy(&descriptor, result.elf_bytes.data() + descriptor_offset, sizeof(descriptor));
  EXPECT_EQ(descriptor.private_segment_fixed_size, 56u);
  EXPECT_EQ(
      AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT),
      1u);

  ConSanResult insufficient_descriptor = result;
  descriptor.private_segment_fixed_size = 32;
  std::memcpy(insufficient_descriptor.elf_bytes.data() + descriptor_offset, &descriptor,
              sizeof(descriptor));
  const std::vector<std::string> validation_errors =
      validate_consan_modified_elf(bytes, insufficient_descriptor);
  ASSERT_FALSE(validation_errors.empty());
  EXPECT_TRUE(std::ranges::any_of(validation_errors, [](const std::string &error) {
    return error.find("insufficient spill descriptor state") != std::string::npos;
  }));
}

TEST(ConSanMoi, AutomaticPersistentStatePreservesStaticRecordReplaySpillWindow) {
  std::vector<uint32_t> text_words(64u, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000304u; // ds_store_b32 v4, v3
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);
  std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "persistent_state_spill_window");
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    descriptor.private_segment_fixed_size = 48u;
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT, 1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT, 1u);
  });
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_init_owner_epoch = true;
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_FALSE(result.resolved_moi_owner_vgpr);
  EXPECT_FALSE(result.resolved_moi_epoch_vgpr);
  ASSERT_TRUE(result.resolved_moi_persistent_owner_sgpr) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.resolved_moi_persistent_epoch_sgpr);
  ASSERT_TRUE(result.resolved_moi_record_replay_workgroup_sgprs.complete());
  EXPECT_EQ(*result.resolved_moi_persistent_epoch_sgpr,
            *result.resolved_moi_persistent_owner_sgpr + 1u);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().source, ConSanRegisterAllocationSource::SpillRequired);
  EXPECT_EQ(result.resource_plans.front().scratch_vgpr, 0u);
  const auto access = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore, &ConSanPatchInfo::kind);
  ASSERT_NE(access, result.patches.end());
  EXPECT_EQ(access->spilled_vgpr_count, 6u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> cave =
      text_words_at_offset(patched, access->trampoline_offset, access->trampoline_size);
  const auto reload_address = build_address_free_scratch_load_b32(
      /*vdst=*/5u, /*byte_offset=*/64u, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(reload_address);
  EXPECT_TRUE(contains_subsequence(cave, *reload_address));
}

TEST(ConSanMoi, FirstLightProbeSupportsZeroToNonzeroDispatchScratch) {
  const std::array<uint32_t, 3> text_words = {
      0xD8340000u,
      0x00000000u,
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options = moi_options();
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified);
  ASSERT_EQ(non_entry_prologue_patch_count(result), 1u);
  EXPECT_EQ(only_non_entry_prologue_patch(result).spilled_vgpr_count, 6u);
  EXPECT_EQ(only_non_entry_prologue_patch(result).required_private_segment_size, 24u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.kernels().size(), 1u);
  KD descriptor{};
  std::memcpy(&descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  EXPECT_EQ(descriptor.private_segment_fixed_size, 24u);
  EXPECT_EQ(
      AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT),
      1u);
}

TEST(ConSanMoi, Rdna4RecordReplaySpillsThroughSiteLocalDynamicStackFrame) {
  const std::array<uint32_t, 3> text_words = {
      0xD8340000u,
      0x00000000u,
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "dynamic_spill", kRdna4Wave64AllVgprsGranulated, /*wave32=*/false,
      /*uses_dynamic_stack=*/true);
  ConSanOptions options = moi_options();
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_TRUE(result.final_validation_passed);
  ASSERT_TRUE(result.resolved_moi_exec_save_sgpr);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().source, ConSanRegisterAllocationSource::SpillRequired);
  EXPECT_EQ(result.resource_plans.front().reason, ConSanRegisterPlanReason::None);
  ASSERT_EQ(result.site_dispositions.size(), 1u);
  EXPECT_EQ(result.site_dispositions.front().lowering_outcome, ConSanSiteLoweringOutcome::Patched);
  EXPECT_EQ(result.site_dispositions.front().lowering_reason, ConSanSiteLoweringReason::None);
  const auto patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &item) {
    return item.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore;
  });
  ASSERT_NE(patch, result.patches.end());
  EXPECT_EQ(patch->spilled_vgpr_count, 6u);
  EXPECT_EQ(patch->required_private_segment_size, 24u);
  EXPECT_EQ(patch->dynamic_private_segment_addend, 24u);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> cave_words =
      text_words_at_offset(patched, patch->trampoline_offset, patch->trampoline_size);
  const uint16_t saved_frame_sgpr = static_cast<uint16_t>(*result.resolved_moi_exec_save_sgpr + 5u);
  EXPECT_NE(std::ranges::find(cave_words, build_s_mov_b32(saved_frame_sgpr, /*frame base=*/33,
                                                          ROCJITSU_CODE_ARCH_RDNA4)),
            cave_words.end());
}

TEST(ConSanMoi, Gfx1250RecordReplaySpillsDynamicStackAccessInBothWaveModes) {
  constexpr auto guest = cdna5::build_vds(cdna5::kDsStoreB32Vds, {.addr = 0u, .data0 = 0u});
  for (bool wave32 : {false, true}) {
    SCOPED_TRACE(wave32 ? "wave32" : "wave64");
    std::vector<uint32_t> text_words(guest.begin(), guest.end());
    text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5));
    const std::vector<uint8_t> bytes = make_gfx1250_code_object(
        text_words, "dynamic_spill", kRdna4Wave64AllVgprsGranulated, wave32,
        /*uses_dynamic_stack=*/true);
    ConSanOptions options = moi_options();
    options.force_vgpr_spill = true;
    options.moi_report_buffer_address = 0x123456780000ull;
    options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

    const ConSanResult result = try_patch_consan(bytes, options);

    ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
    ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
    EXPECT_TRUE(result.final_validation_passed);
    const auto patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &item) {
      return item.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore;
    });
    ASSERT_NE(patch, result.patches.end());
    EXPECT_EQ(patch->spilled_vgpr_count, 6u);
    EXPECT_EQ(patch->required_private_segment_size, 24u);
    EXPECT_EQ(patch->dynamic_private_segment_addend, 24u);
  }
}

TEST(ConSanMoi, Rdna4RecordReplayRejectsMixedStackOwnersWhenSharedHelperNeedsSpill) {
  std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object();
  append_kernel_metadata_note(bytes, "shared_owner_0", /*uses_dynamic_stack=*/true,
                              /*sgpr_count=*/0u);
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);
  options.max_patches = 1;

  const ConSanResult result = try_patch_consan(bytes, options);

  EXPECT_FALSE(result.modified);
  const auto shared_plan =
      std::ranges::find_if(result.resource_plans, [](const ConSanCandidateResourcePlan &plan) {
        return plan.site_kind == ConSanResourceSiteKind::Access &&
               plan.owner_descriptor_file_offsets.size() == 2u;
      });
  ASSERT_NE(shared_plan, result.resource_plans.end());
  EXPECT_EQ(shared_plan->source, ConSanRegisterAllocationSource::Unsupported);
  EXPECT_EQ(shared_plan->reason, ConSanRegisterPlanReason::DynamicStack);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("dynamic-stack owning kernel") != std::string::npos;
  })) << testing::PrintToString(result.warnings);
}

TEST(ConSanMoi, Cdna4RecordReplaySpillsThroughSiteLocalDynamicStackFrame) {
  const auto guest = build_cdna4_ds_store_b32(
      /*vaddr=*/0, /*vdata=*/0, /*byte_offset=*/0, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(guest);
  std::vector<uint32_t> text_words(guest->begin(), guest->end());
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4));
  constexpr uint32_t kCdna4Wave64AllVgprsGranulated = 31u;
  const std::vector<uint8_t> bytes =
      make_cdna4_lds_code_object(text_words, "dynamic_spill", kCdna4Wave64AllVgprsGranulated,
                                 /*uses_dynamic_stack=*/true);
  ConSanOptions options = moi_options();
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << "warnings=" << testing::PrintToString(result.warnings)
                               << " errors=" << testing::PrintToString(result.errors);
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_TRUE(result.final_validation_passed);
  ASSERT_TRUE(result.resolved_moi_exec_save_sgpr);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().source, ConSanRegisterAllocationSource::SpillRequired);
  EXPECT_EQ(result.resource_plans.front().reason, ConSanRegisterPlanReason::None);
  ASSERT_EQ(result.site_dispositions.size(), 1u);
  EXPECT_EQ(result.site_dispositions.front().lowering_outcome, ConSanSiteLoweringOutcome::Patched);
  EXPECT_EQ(result.site_dispositions.front().lowering_reason, ConSanSiteLoweringReason::None);
  const auto patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &item) {
    return item.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore;
  });
  ASSERT_NE(patch, result.patches.end());
  EXPECT_EQ(patch->spilled_vgpr_count, 6u);
  EXPECT_EQ(patch->required_private_segment_size, 32u);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> cave_words =
      text_words_at_offset(patched, patch->trampoline_offset, patch->trampoline_size);
  const uint16_t saved_scc_sgpr = static_cast<uint16_t>(*result.resolved_moi_exec_save_sgpr + 4u);
  const uint16_t saved_frame_sgpr = static_cast<uint16_t>(*result.resolved_moi_exec_save_sgpr + 5u);
  EXPECT_NE(
      std::find(cave_words.begin(), cave_words.end(),
                build_s_mov_b32(saved_frame_sgpr, /*frame base=*/33, ROCJITSU_CODE_ARCH_CDNA4)),
      cave_words.end());
  EXPECT_NE(std::find(cave_words.begin(), cave_words.end(),
                      *build_cdna4_s_cselect_b32(saved_scc_sgpr, scalar_positive_inline_u32(1),
                                                 scalar_positive_inline_u32(0),
                                                 ROCJITSU_CODE_ARCH_CDNA4)),
            cave_words.end());
}

TEST(ConSanMoi, Cdna4DynamicStackRejectsRecordReplayScalarSpillWithoutBootstrap) {
  const auto guest = build_cdna4_ds_store_b32(
      /*vaddr=*/0, /*vdata=*/0, /*byte_offset=*/0, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(guest);
  std::vector<uint32_t> text_words(guest->begin(), guest->end());
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4));
  constexpr uint32_t kCdna4Wave64AllVgprsGranulated = 31u;
  const std::vector<uint8_t> bytes =
      make_cdna4_lds_code_object(text_words, "dynamic_spill", kCdna4Wave64AllVgprsGranulated,
                                 /*uses_dynamic_stack=*/true);
  ConSanOptions options = moi_options();
  options.force_vgpr_spill = true;
  options.automatic_moi_record_replay_sgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const ConSanResult result = try_patch_consan(bytes, options);

  EXPECT_FALSE(result.modified);
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::Unchanged);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().source, ConSanRegisterAllocationSource::SpillRequired);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("no SCC-save register") != std::string::npos;
  })) << testing::PrintToString(result.warnings);
}

TEST(ConSanMoi, Cdna3RecordReplaySpillsThroughSiteLocalDynamicStackFrame) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA3;
  const auto guest = build_cdna3_ds_store_b32(/*vaddr=*/0, /*vdata=*/0, /*byte_offset=*/0, kArch);
  ASSERT_TRUE(guest);
  std::vector<uint32_t> text_words(guest->begin(), guest->end());
  text_words.push_back(build_s_endpgm(kArch));
  constexpr uint32_t kCdna3Wave64AllVgprsGranulated = 31u;
  const std::vector<uint8_t> bytes =
      make_cdna3_lds_code_object(text_words, "dynamic_spill", kCdna3Wave64AllVgprsGranulated,
                                 /*uses_dynamic_stack=*/true);
  ConSanOptions options = moi_options();
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << "warnings=" << testing::PrintToString(result.warnings)
                               << " errors=" << testing::PrintToString(result.errors);
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_TRUE(result.final_validation_passed);
  ASSERT_TRUE(result.resolved_moi_exec_save_sgpr);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().source, ConSanRegisterAllocationSource::SpillRequired);
  EXPECT_EQ(result.resource_plans.front().reason, ConSanRegisterPlanReason::None);
  const auto patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &item) {
    return item.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore;
  });
  ASSERT_NE(patch, result.patches.end());
  EXPECT_EQ(patch->spilled_vgpr_count, 6u);
  EXPECT_EQ(patch->required_private_segment_size, 32u);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> cave_words =
      text_words_at_offset(patched, patch->trampoline_offset, patch->trampoline_size);
  const uint16_t saved_scc_sgpr = static_cast<uint16_t>(*result.resolved_moi_exec_save_sgpr + 4u);
  const uint16_t saved_frame_sgpr = static_cast<uint16_t>(*result.resolved_moi_exec_save_sgpr + 5u);
  EXPECT_NE(std::find(cave_words.begin(), cave_words.end(),
                      build_s_mov_b32(saved_frame_sgpr, /*frame base=*/33, kArch)),
            cave_words.end());
  EXPECT_NE(std::find(cave_words.begin(), cave_words.end(),
                      *build_cdna3_s_cselect_b32(saved_scc_sgpr, scalar_positive_inline_u32(1),
                                                 scalar_positive_inline_u32(0), kArch)),
            cave_words.end());
}

TEST(ConSanMoi, FirstLightProbeWritesOneNativeLdsAccessRecord) {
  std::array<uint32_t, 320> text_words{};
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000000u; // ds_store_b32 v0, v0
  for (size_t i = 2; i + 1 < text_words.size(); ++i)
    text_words[i] = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
  text_words.back() = 0xBFB00000u; // s_endpgm

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options = moi_options();
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 15;
  options.moi_epoch_vgpr = 16;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(non_entry_prologue_patch_count(result), 1u);
  EXPECT_EQ(only_non_entry_prologue_patch(result).kind,
            ConSanPatchKind::InlineMoiAccessRecordStore);
  EXPECT_EQ(only_non_entry_prologue_patch(result).anchor_offset, 0u);
  EXPECT_EQ(only_non_entry_prologue_patch(result).trampoline_offset, 8u);
  expect_bounded_static_record_replay_probe_size(
      only_non_entry_prologue_patch(result).original_size);
  ASSERT_TRUE(only_non_entry_prologue_patch(result).scratch_vgpr);
  EXPECT_EQ(*only_non_entry_prologue_patch(result).scratch_vgpr, 8u);
  EXPECT_EQ(result.resource_plan_summary.explicit_plans, 1u);

  const uint64_t base = *options.moi_report_buffer_address;
  const uint64_t access_record_base = base + sizeof(ConSanMoiReportHeader);
  const std::vector<uint32_t> rewritten_words = patched_words_at_file_offset(
      result, 0x100, only_non_entry_prologue_patch(result).original_size);
  ASSERT_GE(rewritten_words.size(), 2u);
  EXPECT_TRUE(
      contains_subsequence(rewritten_words, std::span<const uint32_t>(text_words.data(), 2u)));
  EXPECT_EQ(std::count(rewritten_words.begin(), rewritten_words.end(), 0xBFC60000u), 0u);

  const auto atomic = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      8, 12, 12, /*return_old_value=*/true, /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  const uint64_t publication_address =
      access_record_base + offsetof(ConSanMoiAccessRecord, claim_token);
  const auto publication_address_lo = build_v_mov_b32_e64_literal(
      8, static_cast<uint32_t>(publication_address), ROCJITSU_CODE_ARCH_RDNA4);
  const auto publication_address_hi = build_v_mov_b32_e64_literal(
      9, static_cast<uint32_t>(publication_address >> 32u), ROCJITSU_CODE_ARCH_RDNA4);
  const auto publication_claim = build_flat_atomic_cmpswap_b64_vaddr_vsrc_vdst(
      8, 10, 10, /*return_old_value=*/true, /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  const auto canonical_owner =
      build_v_cmp_eq_u32_e32_vcc(scalar_positive_inline_u32(0), 11, ROCJITSU_CODE_ARCH_RDNA4);
  const auto record_address_lo = build_v_mov_b32_e64_literal(
      8, static_cast<uint32_t>(access_record_base), ROCJITSU_CODE_ARCH_RDNA4);
  const auto record_address_hi = build_v_mov_b32_e64_literal(
      9, static_cast<uint32_t>(access_record_base >> 32u), ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(atomic);
  ASSERT_TRUE(publication_address_lo);
  ASSERT_TRUE(publication_address_hi);
  ASSERT_TRUE(publication_claim);
  ASSERT_TRUE(canonical_owner);
  ASSERT_TRUE(record_address_lo);
  ASSERT_TRUE(record_address_hi);
  EXPECT_TRUE(contains_subsequence(rewritten_words, *atomic));
  EXPECT_TRUE(contains_subsequence(rewritten_words, *publication_address_lo));
  EXPECT_TRUE(contains_subsequence(rewritten_words, *publication_address_hi));
  EXPECT_EQ(count_subsequence(rewritten_words, *publication_claim), 1u);
  // Fixed slots are claimed by the first lane that actually executes the
  // site; do not hard-code a workgroup/wave-zero owner gate.
  EXPECT_FALSE(
      contains_subsequence(rewritten_words, std::span<const uint32_t>(&*canonical_owner, 1u)));
  EXPECT_GE(count_subsequence(rewritten_words, *record_address_lo), 1u);
  EXPECT_GE(count_subsequence(rewritten_words, *record_address_hi), 1u)
      << "the shared high half can also match header-field addresses";
  EXPECT_TRUE(contains_subsequence(
      rewritten_words,
      make_expected_offset_store_words(offsetof(ConSanMoiAccessRecord, wave_id), 15, 8)));
  EXPECT_TRUE(contains_subsequence(
      rewritten_words,
      make_expected_offset_store_words(offsetof(ConSanMoiAccessRecord, epoch), 16, 8)));
  EXPECT_TRUE(contains_subsequence(
      rewritten_words,
      make_expected_offset_store_words(offsetof(ConSanMoiAccessRecord, lds_byte_offset), 0, 8)));
  EXPECT_TRUE(contains_subsequence(
      rewritten_words,
      make_expected_offset_store_words(offsetof(ConSanMoiAccessRecord, start_cell), 10, 8)));
  const auto commit_kind = instrumentation::build_v_mov_b32_literal(
      10, static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write), ROCJITSU_CODE_ARCH_RDNA4);
  const auto commit_expected = instrumentation::build_v_mov_b32_literal(
      11, static_cast<uint32_t>(ConSanMoiShadowAccessKind::Empty), ROCJITSU_CODE_ARCH_RDNA4);
  const auto commit_address = instrumentation::build_v_add_u64_signed_i24(
      8, offsetof(ConSanMoiAccessRecord, access_kind), ROCJITSU_CODE_ARCH_RDNA4);
  const auto commit = instrumentation::build_flat_atomic_cmpswap_b32(
      8, 10, 10, /*return_old_value=*/true, /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(commit_kind && commit_expected && commit_address && commit);
  std::vector<uint32_t> commit_words(commit_address->begin(), commit_address->end());
  commit_words.insert(commit_words.end(), commit_kind->begin(), commit_kind->end());
  commit_words.insert(commit_words.end(), commit_expected->begin(), commit_expected->end());
  commit_words.insert(commit_words.end(), commit->begin(), commit->end());
  EXPECT_TRUE(contains_subsequence(rewritten_words, commit_words));
  ASSERT_TRUE(result.resolved_moi_exec_save_sgpr);
  const uint16_t exec_save = *result.resolved_moi_exec_save_sgpr;
  const auto save_incoming_exec =
      build_s_mov_b64(exec_save, kRdna4ExecLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto lane_rank_lo = build_v_mbcnt_lo_u32_b32(10, exec_save, scalar_positive_inline_u32(0),
                                                     ROCJITSU_CODE_ARCH_RDNA4);
  const auto lane_rank_hi = build_v_mbcnt_hi_u32_b32(
      10, static_cast<uint16_t>(exec_save + 1u), vector_source_vgpr(10), ROCJITSU_CODE_ARCH_RDNA4);
  const auto first_active =
      build_v_cmp_eq_u32_e32_vcc(scalar_positive_inline_u32(0), 10, ROCJITSU_CODE_ARCH_RDNA4);
  const auto narrow = build_s_and_saveexec_b64(exec_save, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore = build_s_mov_b64(kRdna4ExecLo, exec_save, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(save_incoming_exec);
  ASSERT_TRUE(lane_rank_lo);
  ASSERT_TRUE(lane_rank_hi);
  ASSERT_TRUE(first_active);
  ASSERT_TRUE(narrow);
  ASSERT_TRUE(restore);
  EXPECT_NE(std::ranges::find(rewritten_words, *save_incoming_exec), rewritten_words.end());
  EXPECT_TRUE(contains_subsequence(rewritten_words, *lane_rank_lo));
  EXPECT_TRUE(contains_subsequence(rewritten_words, *lane_rank_hi));
  EXPECT_TRUE(contains_subsequence(rewritten_words, std::span<const uint32_t>(&*first_active, 1u)));
  const auto narrow_position = std::ranges::find(rewritten_words, *narrow);
  const auto atomic_position =
      std::search(rewritten_words.begin(), rewritten_words.end(), atomic->begin(), atomic->end());
  const auto publication_claim_position =
      std::search(rewritten_words.begin(), rewritten_words.end(), publication_claim->begin(),
                  publication_claim->end());
  const auto restore_position = std::ranges::find(rewritten_words, *restore);
  ASSERT_NE(narrow_position, rewritten_words.end());
  ASSERT_NE(atomic_position, rewritten_words.end());
  ASSERT_NE(publication_claim_position, rewritten_words.end());
  ASSERT_NE(restore_position, rewritten_words.end());
  EXPECT_LT(narrow_position, publication_claim_position);
  EXPECT_LT(publication_claim_position, atomic_position);
  EXPECT_LT(atomic_position, restore_position);
  EXPECT_TRUE(contains_subsequence(
      rewritten_words,
      make_expected_scalar_offset_store_words(offsetof(ConSanMoiAccessRecord, lane_mask), exec_save,
                                              /*address_vgpr=*/8)));
  EXPECT_TRUE(contains_subsequence(
      rewritten_words, make_expected_scalar_offset_store_words(
                           offsetof(ConSanMoiAccessRecord, lane_mask) + sizeof(uint32_t),
                           static_cast<uint16_t>(exec_save + 1u),
                           /*address_vgpr=*/8)));
}

TEST(ConSanMoi, Cdna4FirstLightProbeEmitsNativeVariableLengthRecipes) {
  std::vector<uint32_t> text_words(260, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  text_words[0] = 0xd81a0004u;
  text_words[1] = 0x00000302u; // ds_write_b32 v2, v3 offset:4
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);

  const std::vector<uint8_t> bytes = make_cdna4_lds_code_object(text_words);
  ConSanOptions options = moi_options();
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 15;
  options.moi_epoch_vgpr = 16;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_EQ(result.target, ROCJITSU_CODE_TARGET_GFX950);
  EXPECT_EQ(result.arch, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_TRUE(result.final_validation_passed);
  ASSERT_EQ(result.moi_candidates.size(), 1u);
  EXPECT_EQ(result.moi_candidates.front().source, ConSanMoiCandidateSource::NativeLds);
  EXPECT_EQ(result.moi_candidates.front().kind, ConSanLdsAccessKind::Write);
  ASSERT_EQ(non_entry_prologue_patch_count(result), 1u);
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::InlineMoiAccessRecordStore ||
           patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore;
  });
  ASSERT_NE(access, result.patches.end());
  EXPECT_EQ(access->anchor_offset, 0u);
  ASSERT_TRUE(access->scratch_vgpr);
  EXPECT_EQ(*access->scratch_vgpr, 8u);

  const std::vector<uint32_t> rewritten_words = record_access_patch_words(result, *access, 0x100u);
  ASSERT_GE(rewritten_words.size(), 2u);
  EXPECT_TRUE(
      contains_subsequence(rewritten_words, std::span<const uint32_t>(text_words.data(), 2u)));

  const auto publication_claim = build_cdna4_flat_atomic_cmpswap_b64(
      8, 10, 10, /*return_old_value=*/true, /*scope=*/2, ROCJITSU_CODE_ARCH_CDNA4);
  const auto event_index = build_cdna4_flat_atomic_add_u32(8, 12, 12, /*return_old_value=*/true,
                                                           /*scope=*/2, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(result.resolved_moi_exec_save_sgpr);
  const uint16_t exec_save = *result.resolved_moi_exec_save_sgpr;
  const auto save_incoming_exec =
      instrumentation::build_s_mov_b64(exec_save, kRdna4ExecLo, ROCJITSU_CODE_ARCH_CDNA4);
  const auto lane_rank_lo = build_cdna4_v_mbcnt_lo_u32_b32(
      10, exec_save, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_CDNA4);
  const auto lane_rank_hi = build_cdna4_v_mbcnt_hi_u32_b32(
      10, static_cast<uint16_t>(exec_save + 1u), vector_source_vgpr(10), ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(publication_claim && event_index && save_incoming_exec && lane_rank_lo &&
              lane_rank_hi);
  EXPECT_EQ(count_subsequence(rewritten_words, *publication_claim), 1u);
  EXPECT_TRUE(contains_subsequence(rewritten_words, *event_index));
  EXPECT_NE(std::ranges::find(rewritten_words, *save_incoming_exec), rewritten_words.end());
  EXPECT_TRUE(contains_subsequence(rewritten_words, *lane_rank_lo));
  EXPECT_TRUE(contains_subsequence(rewritten_words, *lane_rank_hi));
}

TEST(ConSanMoi, Cdna4RecordReplayNormalizesTransposeAndTwoAddressLdsRanges) {
  const auto check = [](uint32_t word0, uint32_t word1, std::string_view expected_mnemonic,
                        uint32_t expected_width_bits,
                        std::initializer_list<uint32_t> expected_byte_offsets) {
    std::vector<uint32_t> text_words(520, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
    text_words[0] = word0;
    text_words[1] = word1;
    text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);

    const std::vector<uint8_t> bytes = make_cdna4_lds_code_object(text_words);
    ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
    options.scratch_vgpr = 20;
    options.moi_owner_vgpr = 40;
    options.moi_epoch_vgpr = 41;
    options.moi_report_buffer_address = 0x123456780000ull;
    options.moi_report_buffer_size =
        consan_moi_report_buffer_min_bytes(expected_byte_offsets.size(), 0, 0, 0);

    const ConSanResult result = try_patch_consan(bytes, options);

    ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
    ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
    EXPECT_TRUE(result.final_validation_passed);
    ASSERT_EQ(result.moi_candidates.size(), 1u);
    EXPECT_EQ(result.moi_candidates.front().mnemonic, expected_mnemonic);
    EXPECT_EQ(result.moi_candidates.front().width_bits, expected_width_bits);
    ASSERT_EQ(result.site_dispositions.size(), 1u);
    EXPECT_EQ(result.site_dispositions.front().disposition, ConSanSiteDisposition::Supported);
    EXPECT_EQ(result.site_dispositions.front().lowering_outcome,
              ConSanSiteLoweringOutcome::Patched);

    const ConSanMoiAutoReportInventory inventory =
        inventory_consan_moi_auto_report(result, options, bytes);
    EXPECT_EQ(inventory.access_range_count, expected_byte_offsets.size());
    ASSERT_EQ(non_entry_prologue_patch_count(result), 1u);
    const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
      return patch.kind == ConSanPatchKind::InlineMoiAccessRecordStore ||
             patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore;
    });
    ASSERT_NE(access, result.patches.end());
    const std::vector<uint32_t> rewritten_words =
        record_access_patch_words(result, *access, 0x100u);
    for (uint32_t byte_offset : expected_byte_offsets) {
      const auto mov_offset = instrumentation::build_v_mov_b32_literal(
          /*vdst=*/22, byte_offset, ROCJITSU_CODE_ARCH_CDNA4);
      ASSERT_TRUE(mov_offset);
      EXPECT_TRUE(contains_subsequence(rewritten_words, *mov_offset))
          << "missing byte offset " << byte_offset;
    }
  };

  check(0xD9C60800u, 0x0E000001u, "ds_read_b64_tr_b16", 64u, {2048u});
  check(0xD8EE0400u, 0x0800000Bu, "ds_read2_b64", 64u, {0u, 32u});
  check(0xD8700400u, 0x00000001u, "ds_read2st64_b32", 32u, {0u, 1024u});
  check(0xD81E0400u, 0x00000201u, "ds_write2st64_b32", 32u, {0u, 1024u});
  check(0xD8F00400u, 0x00000001u, "ds_read2st64_b64", 64u, {0u, 2048u});
  check(0xD89E0400u, 0x0006080Bu, "ds_write2st64_b64", 64u, {0u, 2048u});
}

TEST(ConSanMoi, Cdna4RecordReplaySupportsSubwordNativeLdsSites) {
  static_assert(cdna4::build_ds(cdna4::kDsReadI8Ds) == std::array<uint32_t, 2>{0xD8720000u, 0u});
  static_assert(cdna4::build_ds(cdna4::kDsWriteB8D16HiDs) ==
                std::array<uint32_t, 2>{0xD8A80000u, 0u});
  static_assert(cdna4::build_ds(cdna4::kDsWriteB16D16HiDs) ==
                std::array<uint32_t, 2>{0xD8AA0000u, 0u});

  const auto check = [](uint32_t word0, uint32_t word1, std::string_view expected_mnemonic,
                        ConSanLdsAccessKind expected_kind, uint32_t expected_width_bits,
                        uint32_t expected_byte_offset, uint16_t expected_addr_vgpr,
                        uint16_t expected_value_vgpr) {
    std::vector<uint32_t> text_words(520, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
    text_words[0] = word0;
    text_words[1] = word1;
    text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);

    const std::vector<uint8_t> bytes = make_cdna4_lds_code_object(text_words);
    ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
    options.scratch_vgpr = 20;
    options.moi_owner_vgpr = 40;
    options.moi_epoch_vgpr = 41;
    options.moi_report_buffer_address = 0x123456780000ull;
    options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

    const ConSanResult result = try_patch_consan(bytes, options);

    ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
    ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
    EXPECT_TRUE(result.final_validation_passed);
    ASSERT_EQ(result.moi_candidates.size(), 1u);
    EXPECT_EQ(result.moi_candidates.front().mnemonic, expected_mnemonic);
    EXPECT_EQ(result.moi_candidates.front().kind, expected_kind);
    EXPECT_EQ(result.moi_candidates.front().width_bits, expected_width_bits);
    ASSERT_TRUE(result.moi_candidates.front().addr_vgpr);
    EXPECT_EQ(*result.moi_candidates.front().addr_vgpr, expected_addr_vgpr);
    if (expected_kind == ConSanLdsAccessKind::Read) {
      ASSERT_TRUE(result.moi_candidates.front().dst_vgpr);
      EXPECT_EQ(*result.moi_candidates.front().dst_vgpr, expected_value_vgpr);
    } else {
      ASSERT_TRUE(result.moi_candidates.front().data_vgpr);
      EXPECT_EQ(*result.moi_candidates.front().data_vgpr, expected_value_vgpr);
    }
    ASSERT_EQ(result.site_dispositions.size(), 1u);
    EXPECT_EQ(result.site_dispositions.front().disposition, ConSanSiteDisposition::Supported);
    EXPECT_EQ(result.site_dispositions.front().lowering_outcome,
              ConSanSiteLoweringOutcome::Patched);

    const ConSanMoiAutoReportInventory inventory =
        inventory_consan_moi_auto_report(result, options, bytes);
    EXPECT_EQ(inventory.access_range_count, 1u);
    ASSERT_EQ(non_entry_prologue_patch_count(result), 1u);
    const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
      return patch.kind == ConSanPatchKind::InlineMoiAccessRecordStore ||
             patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore;
    });
    ASSERT_NE(access, result.patches.end());
    const std::vector<uint32_t> rewritten_words =
        record_access_patch_words(result, *access, 0x100u);
    const auto mov_offset = instrumentation::build_v_mov_b32_literal(
        /*vdst=*/22, expected_byte_offset, ROCJITSU_CODE_ARCH_CDNA4);
    ASSERT_TRUE(mov_offset);
    EXPECT_TRUE(contains_subsequence(rewritten_words, *mov_offset));
  };

  check(0xD8740020u, 0x07000004u, "ds_read_u8", ConSanLdsAccessKind::Read, 8u, 32u,
        /*addr=*/4u, /*vdst=*/7u);
  check(0xD8780020u, 0x07000004u, "ds_read_u16", ConSanLdsAccessKind::Read, 16u, 32u,
        /*addr=*/4u, /*vdst=*/7u);
  check(0xD83C0010u, 0x00000704u, "ds_write_b8", ConSanLdsAccessKind::Write, 8u, 16u,
        /*addr=*/4u, /*data0=*/7u);
  check(0xD83E0010u, 0x00000704u, "ds_write_b16", ConSanLdsAccessKind::Write, 16u, 16u,
        /*addr=*/4u, /*data0=*/7u);

  constexpr auto read_i8 =
      cdna4::build_ds(cdna4::kDsReadI8Ds, {.offset0 = 0x21, .addr = 8, .vdst = 12});
  constexpr auto read_u8_d16 =
      cdna4::build_ds(cdna4::kDsReadU8D16Ds, {.offset0 = 0x22, .addr = 8, .vdst = 12});
  constexpr auto read_u8_d16_hi =
      cdna4::build_ds(cdna4::kDsReadU8D16HiDs, {.offset0 = 0x23, .addr = 8, .vdst = 12});
  constexpr auto read_i8_d16 =
      cdna4::build_ds(cdna4::kDsReadI8D16Ds, {.offset0 = 0x24, .addr = 8, .vdst = 12});
  constexpr auto read_i8_d16_hi =
      cdna4::build_ds(cdna4::kDsReadI8D16HiDs, {.offset0 = 0x25, .addr = 8, .vdst = 12});
  constexpr auto read_u16_d16 =
      cdna4::build_ds(cdna4::kDsReadU16D16Ds, {.offset0 = 0x26, .addr = 8, .vdst = 12});
  constexpr auto read_u16_d16_hi =
      cdna4::build_ds(cdna4::kDsReadU16D16HiDs, {.offset0 = 0x27, .addr = 8, .vdst = 12});
  constexpr auto write_b8_d16_hi =
      cdna4::build_ds(cdna4::kDsWriteB8D16HiDs, {.offset0 = 0x12, .addr = 5, .data0 = 9});
  constexpr auto write_b16_d16_hi =
      cdna4::build_ds(cdna4::kDsWriteB16D16HiDs, {.offset0 = 0x34, .addr = 6, .data0 = 10});
  check(read_i8[0], read_i8[1], "ds_read_i8", ConSanLdsAccessKind::Read, 8u, 0x21u,
        /*addr=*/8u, /*vdst=*/12u);
  check(read_u8_d16[0], read_u8_d16[1], "ds_read_u8_d16", ConSanLdsAccessKind::Read, 8u, 0x22u,
        /*addr=*/8u, /*vdst=*/12u);
  check(read_u8_d16_hi[0], read_u8_d16_hi[1], "ds_read_u8_d16_hi", ConSanLdsAccessKind::Read, 8u,
        0x23u, /*addr=*/8u, /*vdst=*/12u);
  check(read_i8_d16[0], read_i8_d16[1], "ds_read_i8_d16", ConSanLdsAccessKind::Read, 8u, 0x24u,
        /*addr=*/8u, /*vdst=*/12u);
  check(read_i8_d16_hi[0], read_i8_d16_hi[1], "ds_read_i8_d16_hi", ConSanLdsAccessKind::Read, 8u,
        0x25u, /*addr=*/8u, /*vdst=*/12u);
  check(read_u16_d16[0], read_u16_d16[1], "ds_read_u16_d16", ConSanLdsAccessKind::Read, 16u, 0x26u,
        /*addr=*/8u, /*vdst=*/12u);
  check(read_u16_d16_hi[0], read_u16_d16_hi[1], "ds_read_u16_d16_hi", ConSanLdsAccessKind::Read,
        16u, 0x27u, /*addr=*/8u, /*vdst=*/12u);
  check(write_b8_d16_hi[0], write_b8_d16_hi[1], "ds_write_b8_d16_hi", ConSanLdsAccessKind::Write,
        8u, 0x12u, /*addr=*/5u, /*data0=*/9u);
  check(write_b16_d16_hi[0], write_b16_d16_hi[1], "ds_write_b16_d16_hi", ConSanLdsAccessKind::Write,
        16u, 0x34u, /*addr=*/6u, /*data0=*/10u);
}

TEST(ConSanMoi, Cdna4RecordReplayRecordsDispatchIdentity) {
  const auto guest = build_cdna4_ds_store_b32(
      /*vaddr=*/2, /*vdata=*/3, /*byte_offset=*/0, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(guest);
  std::vector<uint32_t> text_words(320, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  std::copy(guest->begin(), guest->end(), text_words.begin());
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_init_owner_epoch = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(8, 0, 0, 0, 8);

  const ConSanResult result = try_patch_consan(
      make_cdna4_lds_code_object(text_words, "record_replay_dispatch_identity"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.resolved_moi_dispatch_id_sgpr);
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::InlineMoiAccessRecordStore ||
           patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore;
  });
  ASSERT_NE(access, result.patches.end());
  ASSERT_TRUE(access->scratch_vgpr);

  const std::vector<uint32_t> access_words =
      access->kind == ConSanPatchKind::InlineMoiAccessRecordStore
          ? patched_words_at_file_offset(result, 0x100 + access->anchor_offset,
                                         access->original_size)
          : text_words_at_offset(AmdGpuCodeObject(result.elf_bytes.data(), result.elf_bytes.size()),
                                 access->trampoline_offset, access->trampoline_size);
  const auto expected_dispatch_store = [&](uint32_t byte_offset, uint16_t scalar_src) {
    const uint16_t value_vgpr = static_cast<uint16_t>(*access->scratch_vgpr + 2u);
    std::vector<uint32_t> words = {
        build_v_mov_b32_e32(value_vgpr, scalar_src, ROCJITSU_CODE_ARCH_CDNA4)};
    const auto store =
        build_cdna4_flat_store_b32(*access->scratch_vgpr, value_vgpr,
                                   static_cast<uint16_t>(byte_offset), ROCJITSU_CODE_ARCH_CDNA4);
    EXPECT_TRUE(store);
    if (store)
      words.insert(words.end(), store->begin(), store->end());
    return words;
  };
  EXPECT_TRUE(contains_subsequence(
      access_words, expected_dispatch_store(offsetof(ConSanMoiAccessRecord, generation),
                                            *result.resolved_moi_dispatch_id_sgpr)));
  EXPECT_TRUE(contains_subsequence(
      access_words,
      expected_dispatch_store(offsetof(ConSanMoiAccessRecord, generation) + sizeof(uint32_t),
                              static_cast<uint16_t>(*result.resolved_moi_dispatch_id_sgpr + 1u))));
}

TEST(ConSanMoi, Cdna3PrivateEpochRecordReplayLoadsEntryOwner) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA3;
  const auto guest = build_cdna3_ds_store_b32(/*vaddr=*/2, /*vdata=*/3, /*byte_offset=*/0, kArch);
  ASSERT_TRUE(guest);
  std::vector<uint32_t> text_words(320, build_s_nop(0, kArch));
  text_words[0] = build_v_mov_b32_e32(/*vdst=*/0, vector_source_vgpr(7), kArch);
  std::ranges::copy(*guest, text_words.begin() + 1);
  text_words.back() = build_s_endpgm(kArch);

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_dynamic_access_records = true;
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.moi_init_owner_epoch = true;
  options.force_private_epoch = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(8, 0, 0, 0, 8);

  const ConSanResult result =
      try_patch_consan(make_cdna3_lds_code_object(text_words, "private_entry_owner"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.moi_private_epoch_automatic);
  const auto access = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore, &ConSanPatchInfo::kind);
  const auto prologue = std::ranges::find(
      result.patches, ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue, &ConSanPatchInfo::kind);
  ASSERT_NE(access, result.patches.end());
  ASSERT_NE(prologue, result.patches.end());
  ASSERT_TRUE(access->scratch_vgpr);
  ASSERT_TRUE(access->persistent_epoch_private_offset);
  ASSERT_TRUE(access->persistent_owner_private_offset);
  EXPECT_EQ(prologue->persistent_epoch_private_offset, access->persistent_epoch_private_offset);
  EXPECT_EQ(prologue->persistent_owner_private_offset, access->persistent_owner_private_offset);
  EXPECT_EQ(prologue->persistent_private_state_end, access->persistent_private_state_end);
  ASSERT_TRUE(prologue->scratch_vgpr);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  KD patched_descriptor{};
  std::memcpy(&patched_descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(patched_descriptor));
  const uint16_t full_payload_base = static_cast<uint16_t>(
      AMDHSA_BITS_GET(patched_descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT));
  const std::vector<uint32_t> prologue_words =
      text_words_at_offset(patched, prologue->trampoline_offset, prologue->trampoline_size);
  const uint32_t capture_entry_owner =
      build_v_mov_b32_e32(*prologue->scratch_vgpr, vector_source_vgpr(/*workitem_id_x=*/0u), kArch);
  EXPECT_NE(std::ranges::find(prologue_words, capture_entry_owner), prologue_words.end());
  for (uint16_t dimension = 0; dimension < 3u; ++dimension) {
    const uint32_t capture_coordinate = build_v_mov_b32_e32(
        *prologue->scratch_vgpr, static_cast<uint16_t>(full_payload_base + dimension), kArch);
    EXPECT_NE(std::ranges::find(prologue_words, capture_coordinate), prologue_words.end());
  }
  const std::vector<uint32_t> cave =
      text_words_at_offset(patched, access->trampoline_offset, access->trampoline_size);
  const uint16_t owner_vgpr = static_cast<uint16_t>(*access->scratch_vgpr + 4u);
  const auto owner_load = instrumentation::build_private_load_b32(
      owner_vgpr, *access->persistent_owner_private_offset, kArch);
  const auto captured_owner = instrumentation::build_v_lshrrev_b32(
      owner_vgpr, scalar_positive_inline_u32(6u), owner_vgpr, kArch);
  const auto live_owner = instrumentation::build_v_lshrrev_b32(
      owner_vgpr, scalar_positive_inline_u32(6u), /*workitem_id_x=*/0u, kArch);
  ASSERT_TRUE(owner_load && captured_owner && live_owner);
  EXPECT_TRUE(contains_subsequence(cave, *owner_load));
  EXPECT_NE(std::ranges::find(cave, *captured_owner), cave.end());
  EXPECT_EQ(std::ranges::find(cave, *live_owner), cave.end());
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, Gfx1250PrivateEpochAtomicAndFenceRecordsLoadEntryOwner) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA5;
  constexpr auto guest = cdna5::build_vds(cdna5::kDsStoreB32Vds, {.addr = 2u, .data0 = 3u});
  const auto atomic = build_gfx1250_flat_atomic_add_u32(
      /*vaddr=*/4, /*vsrc=*/6, /*vdst=*/7, /*return_old_value=*/true, /*scope=*/2, kArch);
  ASSERT_TRUE(atomic);
  std::vector<uint32_t> text_words(320, build_s_nop(0, kArch));
  text_words[0] = build_v_mov_b32_e32(/*vdst=*/0, vector_source_vgpr(8), kArch);
  size_t cursor = 1u;
  std::ranges::copy(guest, text_words.begin() + cursor);
  cursor += guest.size();
  const std::array<uint32_t, 3> release = {0xEE0B0000u, 0x00000000u, 0x00000000u}; // global_wb
  std::ranges::copy(release, text_words.begin() + cursor);
  cursor += release.size();
  std::ranges::copy(*atomic, text_words.begin() + cursor);
  text_words.back() = build_s_endpgm(kArch);

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_dynamic_access_records = true;
  options.moi_track_barriers = false;
  options.moi_track_atomics = true;
  options.moi_init_owner_epoch = true;
  options.force_private_epoch = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(8, 0, 0, 0, 0, 8, 8);

  const ConSanResult result =
      try_patch_consan(make_gfx1250_code_object(text_words, "private_sync_entry_owner"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.moi_private_epoch_automatic);
  const auto access = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore, &ConSanPatchInfo::kind);
  const auto atomic_patch = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiAtomicRecord, &ConSanPatchInfo::kind);
  const auto fence_patch = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiFenceRecord, &ConSanPatchInfo::kind);
  ASSERT_NE(access, result.patches.end());
  ASSERT_NE(atomic_patch, result.patches.end()) << testing::PrintToString(result.warnings);
  ASSERT_NE(fence_patch, result.patches.end()) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(access->persistent_owner_private_offset);
  EXPECT_EQ(atomic_patch->persistent_owner_private_offset, access->persistent_owner_private_offset);
  EXPECT_EQ(fence_patch->persistent_owner_private_offset, access->persistent_owner_private_offset);
  ASSERT_TRUE(atomic_patch->scratch_vgpr);
  ASSERT_TRUE(fence_patch->scratch_vgpr);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const auto expect_captured_owner = [&](const ConSanPatchInfo &patch, uint16_t owner_vgpr) {
    const std::vector<uint32_t> words =
        text_words_at_offset(patched, patch.trampoline_offset, patch.trampoline_size);
    const auto load = instrumentation::build_private_load_b32(
        owner_vgpr, *patch.persistent_owner_private_offset, kArch);
    const auto captured = instrumentation::build_v_lshrrev_b32(
        owner_vgpr, scalar_positive_inline_u32(5u), owner_vgpr, kArch);
    const auto live = instrumentation::build_v_lshrrev_b32(
        owner_vgpr, scalar_positive_inline_u32(5u), /*workitem_id_x=*/0u, kArch);
    ASSERT_TRUE(load && captured && live);
    EXPECT_TRUE(contains_subsequence(words, *load));
    EXPECT_NE(std::ranges::find(words, *captured), words.end());
    EXPECT_EQ(std::ranges::find(words, *live), words.end());
  };
  expect_captured_owner(*atomic_patch, static_cast<uint16_t>(*atomic_patch->scratch_vgpr + 4u));
  expect_captured_owner(*fence_patch, static_cast<uint16_t>(*fence_patch->scratch_vgpr + 5u));
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, Gfx1250PrivateOwnerSpillsBeginAfterCapturedState) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA5;
  constexpr auto guest = cdna5::build_vds(cdna5::kDsStoreB32Vds, {.addr = 2u, .data0 = 3u});
  const auto atomic = build_gfx1250_flat_atomic_add_u32(
      /*vaddr=*/4, /*vsrc=*/6, /*vdst=*/7, /*return_old_value=*/true, /*scope=*/2, kArch);
  ASSERT_TRUE(atomic);
  std::vector<uint32_t> text_words(320, build_s_nop(0, kArch));
  std::ranges::copy(guest, text_words.begin() + 1);
  size_t cursor = 1u + guest.size();
  const std::array<uint32_t, 3> release = {0xEE0B0000u, 0x00000000u, 0x00000000u}; // global_wb
  std::ranges::copy(release, text_words.begin() + cursor);
  cursor += release.size();
  std::ranges::copy(*atomic, text_words.begin() + cursor);
  text_words.back() = build_s_endpgm(kArch);

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_dynamic_access_records = true;
  options.moi_track_barriers = false;
  options.moi_track_atomics = true;
  options.moi_init_owner_epoch = true;
  options.force_private_epoch = true;
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(8, 0, 0, 0, 0, 8, 8);

  const ConSanResult result =
      try_patch_consan(make_gfx1250_code_object(text_words, "private_owner_spill"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  const auto access = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore, &ConSanPatchInfo::kind);
  ASSERT_NE(access, result.patches.end()) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(access->persistent_owner_private_offset);
  ASSERT_TRUE(access->persistent_private_state_end);
  ASSERT_GT(*access->persistent_private_state_end, 0u);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  for (ConSanPatchKind kind :
       {ConSanPatchKind::TrampolineMoiAtomicRecord, ConSanPatchKind::TrampolineMoiFenceRecord}) {
    SCOPED_TRACE(testing::PrintToString(kind));
    const auto patch = std::ranges::find(result.patches, kind, &ConSanPatchInfo::kind);
    ASSERT_NE(patch, result.patches.end()) << testing::PrintToString(result.warnings);
    ASSERT_TRUE(patch->scratch_vgpr);
    ASSERT_GT(patch->spilled_vgpr_count, 0u);
    EXPECT_EQ(patch->persistent_owner_private_offset, access->persistent_owner_private_offset);
    EXPECT_EQ(patch->persistent_private_state_end, access->persistent_private_state_end);
    const std::vector<uint32_t> cave =
        text_words_at_offset(patched, patch->trampoline_offset, patch->trampoline_size);
    bool found_spill_after_persistent_state = false;
    for (uint16_t index = 0; index < patch->spilled_vgpr_count; ++index) {
      const uint16_t vgpr = static_cast<uint16_t>(*patch->scratch_vgpr + index);
      for (uint32_t offset = 0; offset < *access->persistent_private_state_end;
           offset += sizeof(uint32_t)) {
        const auto overlapping_spill =
            instrumentation::build_private_store_b32(vgpr, offset, kArch);
        ASSERT_TRUE(overlapping_spill);
        EXPECT_FALSE(contains_subsequence(cave, *overlapping_spill));
      }
      for (uint32_t offset = *access->persistent_private_state_end;
           offset < patch->required_private_segment_size; offset += sizeof(uint32_t)) {
        const auto spill = instrumentation::build_private_store_b32(vgpr, offset, kArch);
        ASSERT_TRUE(spill);
        found_spill_after_persistent_state |= contains_subsequence(cave, *spill);
      }
    }
    EXPECT_TRUE(found_spill_after_persistent_state);
  }
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, Gfx1250PrivateOwnerScalarSpillsDoNotOverlapCapturedState) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA5;
  constexpr uint32_t kAccessCount = 9u;
  const auto atomic = build_gfx1250_flat_atomic_add_u32(
      /*vaddr=*/4, /*vsrc=*/6, /*vdst=*/7, /*return_old_value=*/true, /*scope=*/2, kArch);
  ASSERT_TRUE(atomic);
  std::vector<uint32_t> text_words;
  for (uint32_t index = 0; index < kAccessCount; ++index) {
    text_words.push_back(0xD8340000u | index * sizeof(uint32_t));
    text_words.push_back(0x00000000u); // ds_store_b32 v0, v0 offset:index*4
  }
  const std::array<uint32_t, 3> release = {0xEE0B0000u, 0x00000000u, 0x00000000u}; // global_wb
  text_words.insert(text_words.end(), release.begin(), release.end());
  text_words.insert(text_words.end(), atomic->begin(), atomic->end());
  const std::array<uint16_t, 4> dead_router_sgprs = {0u, 1u, 4u, 6u};
  for (uint16_t sgpr = 0; sgpr < 106u; ++sgpr) {
    if (std::ranges::find(dead_router_sgprs, sgpr) == dead_router_sgprs.end())
      text_words.push_back(build_s_mov_b32(/*sdst=*/20u, sgpr, kArch));
  }
  text_words.push_back(build_s_endpgm(kArch));

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 8u;
  options.moi_dynamic_access_records = true;
  options.moi_track_barriers = false;
  options.moi_track_atomics = true;
  options.moi_init_owner_epoch = true;
  options.force_private_epoch = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size =
      consan_moi_report_buffer_min_bytes(kAccessCount, 0, 0, 0, 0, 1, 1);
  options.max_patches = 32u;

  const ConSanResult result =
      try_patch_consan(make_gfx1250_code_object(text_words, "private_owner_scalar_spill"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.resolved_moi_transient_sgpr_assignments.size(), 1u);
  EXPECT_TRUE(result.resolved_moi_transient_sgpr_assignments.front().spill_backed);
  const auto access = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore, &ConSanPatchInfo::kind);
  ASSERT_NE(access, result.patches.end()) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(access->persistent_private_state_end);
  ASSERT_GT(*access->persistent_private_state_end, 0u);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  for (ConSanPatchKind kind :
       {ConSanPatchKind::TrampolineMoiAtomicRecord, ConSanPatchKind::TrampolineMoiFenceRecord}) {
    SCOPED_TRACE(testing::PrintToString(kind));
    const auto patch = std::ranges::find(result.patches, kind, &ConSanPatchInfo::kind);
    ASSERT_NE(patch, result.patches.end()) << testing::PrintToString(result.warnings);
    ASSERT_TRUE(patch->scratch_vgpr);
    EXPECT_EQ(patch->persistent_private_state_end, access->persistent_private_state_end);
    EXPECT_EQ(patch->required_private_segment_size, *access->persistent_private_state_end);
    const std::vector<uint32_t> cave =
        text_words_at_offset(patched, patch->trampoline_offset, patch->trampoline_size);
    for (uint32_t offset = 0; offset < patch->required_private_segment_size;
         offset += sizeof(uint32_t)) {
      const auto spill =
          instrumentation::build_private_store_b32(*patch->scratch_vgpr, offset, kArch);
      ASSERT_TRUE(spill);
      EXPECT_FALSE(contains_subsequence(cave, *spill));
    }
    expect_lane_backed_scalar_spill(
        result, patched, *patch,
        result.resolved_moi_transient_sgpr_assignments.front().exec_save_sgpr, kArch);
  }
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, Gfx1250PrivateOwnerFallbackRetainsAtomicRecord) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA5;
  constexpr auto guest = cdna5::build_vds(cdna5::kDsStoreB32Vds, {.addr = 2u, .data0 = 3u});
  const auto atomic = build_gfx1250_flat_atomic_add_u32(
      /*vaddr=*/4, /*vsrc=*/6, /*vdst=*/7, /*return_old_value=*/true, /*scope=*/2, kArch);
  ASSERT_TRUE(atomic);
  std::vector<uint32_t> text_words(
      40000u, build_v_mov_b32_e32(/*vdst=*/20u, vector_source_vgpr(20u), kArch));
  std::ranges::copy(guest, text_words.begin());
  size_t cursor = text_words.size() - atomic->size() - 4u;
  const std::array<uint32_t, 3> release = {0xEE0B0000u, 0x00000000u, 0x00000000u}; // global_wb
  std::ranges::copy(release, text_words.begin() + cursor);
  cursor += release.size();
  std::ranges::copy(*atomic, text_words.begin() + cursor);
  text_words.back() = build_s_endpgm(kArch);

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_dynamic_access_records = true;
  options.moi_track_barriers = false;
  options.moi_track_atomics = true;
  options.moi_init_owner_epoch = true;
  options.force_private_epoch = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(8, 0, 0, 0, 0, 8, 8);

  const ConSanResult result =
      try_patch_consan(make_gfx1250_code_object(text_words, "private_owner_fallback"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  const auto atomic_patch = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiAtomicRecord, &ConSanPatchInfo::kind);
  const auto prologue = std::ranges::find(
      result.patches, ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue, &ConSanPatchInfo::kind);
  ASSERT_NE(atomic_patch, result.patches.end()) << testing::PrintToString(result.warnings);
  ASSERT_NE(prologue, result.patches.end()) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(atomic_patch->persistent_record_replay_workgroup_private_offsets.complete());
  EXPECT_EQ(atomic_patch->persistent_record_replay_workgroup_private_offsets,
            prologue->persistent_record_replay_workgroup_private_offsets);
  EXPECT_FALSE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("probe-local") != std::string::npos ||
           warning.find("private owner has no captured state") != std::string::npos;
  }));
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, UnrelatedDynamicStackKernelDoesNotDisablePrivateRecordReplayState) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA5;
  constexpr auto guest = cdna5::build_vds(cdna5::kDsStoreB32Vds, {.addr = 2u, .data0 = 3u});
  constexpr auto unsupported_guest =
      cdna5::build_vds(cdna5::kDsStoreB32Vds, {.addr = 8u, .data0 = 9u});
  std::vector<uint32_t> owner_words(320u, build_s_nop(0, kArch));
  std::ranges::copy(guest, owner_words.begin());
  owner_words.back() = build_s_endpgm(kArch);
  const std::array<uint32_t, 3> unrelated_words = {unsupported_guest[0], unsupported_guest[1],
                                                   build_s_endpgm(kArch)};
  std::vector<uint8_t> bytes = make_gfx1250_code_object_with_local_function(
      owner_words, unrelated_words, {}, /*vgpr_granulated=*/0u, /*function_is_kernel=*/true);
  append_kernel_metadata_note(bytes, "lds_helper", /*uses_dynamic_stack=*/true,
                              /*sgpr_count=*/0u);

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.moi_init_owner_epoch = true;
  options.force_private_epoch = true;
  // The second kernel aliases this explicit scratch window with its guest
  // operands, producing an owned but unsupported resource plan.
  options.scratch_vgpr = 8u;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.kernels.size(), 2u);
  const auto unrelated = std::ranges::find(result.kernels, "lds_helper", &ConSanKernelInfo::name);
  const auto owner = std::ranges::find(result.kernels, "lds_probe", &ConSanKernelInfo::name);
  ASSERT_NE(unrelated, result.kernels.end());
  ASSERT_NE(owner, result.kernels.end());
  ASSERT_TRUE(unrelated->uses_dynamic_stack);
  EXPECT_TRUE(*unrelated->uses_dynamic_stack);
  EXPECT_TRUE(std::ranges::any_of(result.resource_plans, [&](const auto &plan) {
    return plan.source == ConSanRegisterAllocationSource::Unsupported &&
           plan.owner_descriptor_file_offsets ==
               std::vector<uint64_t>{unrelated->descriptor_file_offset};
  })) << testing::PrintToString(result.resource_plans);
  EXPECT_TRUE(result.moi_private_epoch_automatic);
  const auto access = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore, &ConSanPatchInfo::kind);
  const auto prologue = std::ranges::find(
      result.patches, ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue, &ConSanPatchInfo::kind);
  ASSERT_NE(access, result.patches.end());
  ASSERT_NE(prologue, result.patches.end());
  EXPECT_EQ(access->owner_descriptor_file_offsets,
            std::vector<uint64_t>{owner->descriptor_file_offset});
  EXPECT_EQ(prologue->owner_descriptor_file_offsets, access->owner_descriptor_file_offsets);
  EXPECT_TRUE(access->persistent_record_replay_workgroup_private_offsets.complete());
  EXPECT_EQ(prologue->persistent_record_replay_workgroup_private_offsets,
            access->persistent_record_replay_workgroup_private_offsets);
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, OwningDynamicStackKernelRejectsForcedPrivateRecordReplayState) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA5;
  constexpr auto guest = cdna5::build_vds(cdna5::kDsStoreB32Vds, {.addr = 2u, .data0 = 3u});
  std::vector<uint32_t> text_words(320u, build_s_nop(0, kArch));
  std::ranges::copy(guest, text_words.begin());
  text_words.back() = build_s_endpgm(kArch);
  const std::vector<uint8_t> bytes =
      make_gfx1250_code_object(text_words, "owning_dynamic_stack", kRdna4Wave64AllVgprsGranulated,
                               /*wave32=*/true, /*uses_dynamic_stack=*/true);
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.force_private_epoch = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const ConSanResult result = try_patch_consan(bytes, options);

  EXPECT_EQ(result.outcome, ConSanTransformOutcome::Unsupported);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("dynamic-stack owner without a persistent register window") !=
           std::string::npos;
  })) << testing::PrintToString(result.warnings);
}

TEST(ConSanMoi, Cdna3PrivateEpochAtomicAndFenceRecordsLoadEntryOwner) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA3;
  const auto guest = build_cdna3_ds_store_b32(/*vaddr=*/2, /*vdata=*/3, /*byte_offset=*/0, kArch);
  const auto release = cdna3::build_mubuf(cdna3::kBufferWbl2Mubuf, {.sc1 = 1});
  const auto acquire = build_cdna3_buffer_inv_sc1(kArch);
  const auto atomic = build_cdna3_flat_atomic_add_u32(
      /*vaddr=*/2, /*vsrc=*/4, /*vdst=*/5, /*return_old_value=*/true, /*scope=*/2, kArch);
  const auto wait = build_cdna3_s_wait_vmcnt_lgkmcnt0(kArch);
  ASSERT_TRUE(guest && acquire && atomic && wait);
  std::vector<uint32_t> text_words(320, build_s_nop(0, kArch));
  text_words[0] = build_v_mov_b32_e32(/*vdst=*/0, vector_source_vgpr(7), kArch);
  size_t cursor = 1u;
  std::ranges::copy(*guest, text_words.begin() + cursor);
  cursor += guest->size();
  std::ranges::copy(release, text_words.begin() + cursor);
  cursor += release.size();
  text_words[cursor++] = *wait;
  std::ranges::copy(*atomic, text_words.begin() + cursor);
  cursor += atomic->size();
  text_words[cursor++] = *wait;
  std::ranges::copy(*acquire, text_words.begin() + cursor);
  text_words.back() = build_s_endpgm(kArch);

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_dynamic_access_records = true;
  options.moi_track_barriers = false;
  options.moi_track_atomics = true;
  options.moi_init_owner_epoch = true;
  options.force_private_epoch = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(8, 0, 0, 0, 0, 8, 8);

  const std::vector<uint8_t> bytes =
      make_cdna3_lds_code_object(text_words, "private_sync_entry_owner");
  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.moi_private_epoch_automatic);
  const auto access = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore, &ConSanPatchInfo::kind);
  const auto atomic_patch = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiAtomicRecord, &ConSanPatchInfo::kind);
  const auto fence_patch = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiFenceRecord, &ConSanPatchInfo::kind);
  const auto prologue = std::ranges::find(
      result.patches, ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue, &ConSanPatchInfo::kind);
  ASSERT_NE(access, result.patches.end());
  ASSERT_NE(atomic_patch, result.patches.end()) << testing::PrintToString(result.warnings);
  ASSERT_NE(fence_patch, result.patches.end()) << testing::PrintToString(result.warnings);
  ASSERT_NE(prologue, result.patches.end()) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(access->persistent_owner_private_offset);
  EXPECT_EQ(atomic_patch->persistent_owner_private_offset, access->persistent_owner_private_offset);
  EXPECT_EQ(fence_patch->persistent_owner_private_offset, access->persistent_owner_private_offset);
  ASSERT_TRUE(access->persistent_record_replay_workgroup_private_offsets.complete());
  EXPECT_EQ(atomic_patch->persistent_record_replay_workgroup_private_offsets,
            access->persistent_record_replay_workgroup_private_offsets);
  EXPECT_EQ(fence_patch->persistent_record_replay_workgroup_private_offsets,
            access->persistent_record_replay_workgroup_private_offsets);
  EXPECT_EQ(prologue->persistent_record_replay_workgroup_private_offsets,
            access->persistent_record_replay_workgroup_private_offsets);
  ASSERT_TRUE(access->scratch_vgpr);
  ASSERT_TRUE(atomic_patch->scratch_vgpr);
  ASSERT_TRUE(fence_patch->scratch_vgpr);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const auto expect_captured_owner = [&](const ConSanPatchInfo &patch, uint16_t owner_vgpr) {
    const std::vector<uint32_t> words =
        text_words_at_offset(patched, patch.trampoline_offset, patch.trampoline_size);
    const auto load = instrumentation::build_private_load_b32(
        owner_vgpr, *patch.persistent_owner_private_offset, kArch);
    const auto captured = instrumentation::build_v_lshrrev_b32(
        owner_vgpr, scalar_positive_inline_u32(6u), owner_vgpr, kArch);
    const auto live = instrumentation::build_v_lshrrev_b32(
        owner_vgpr, scalar_positive_inline_u32(6u), /*workitem_id_x=*/0u, kArch);
    ASSERT_TRUE(load && captured && live);
    EXPECT_TRUE(contains_subsequence(words, *load));
    EXPECT_NE(std::ranges::find(words, *captured), words.end());
    EXPECT_EQ(std::ranges::find(words, *live), words.end());
  };
  expect_captured_owner(*atomic_patch, static_cast<uint16_t>(*atomic_patch->scratch_vgpr + 4u));
  expect_captured_owner(*fence_patch, static_cast<uint16_t>(*fence_patch->scratch_vgpr + 5u));
  const auto expect_captured_workgroup = [&](const ConSanPatchInfo &patch, uint16_t value_vgpr) {
    const std::vector<uint32_t> words =
        text_words_at_offset(patched, patch.trampoline_offset, patch.trampoline_size);
    for (uint32_t offset : {*patch.persistent_record_replay_workgroup_private_offsets.x,
                            *patch.persistent_record_replay_workgroup_private_offsets.y,
                            *patch.persistent_record_replay_workgroup_private_offsets.z}) {
      const auto load = instrumentation::build_private_load_b32(value_vgpr, offset, kArch);
      ASSERT_TRUE(load);
      EXPECT_TRUE(contains_subsequence(words, *load));
    }
  };
  expect_captured_workgroup(*access, static_cast<uint16_t>(*access->scratch_vgpr + 5u));
  expect_captured_workgroup(*atomic_patch, static_cast<uint16_t>(*atomic_patch->scratch_vgpr + 4u));
  expect_captured_workgroup(*fence_patch, static_cast<uint16_t>(*fence_patch->scratch_vgpr + 5u));
  EXPECT_TRUE(result.final_validation_passed);

  ConSanResult incomplete_tuple = result;
  const auto incomplete_atomic = std::ranges::find(
      incomplete_tuple.patches, ConSanPatchKind::TrampolineMoiAtomicRecord, &ConSanPatchInfo::kind);
  ASSERT_NE(incomplete_atomic, incomplete_tuple.patches.end());
  incomplete_atomic->persistent_record_replay_workgroup_private_offsets.z.reset();
  const std::vector<std::string> incomplete_errors =
      validate_consan_modified_elf(bytes, incomplete_tuple);
  EXPECT_TRUE(std::ranges::any_of(incomplete_errors, [](const std::string &error) {
    return error.find("incomplete private Record/Replay workgroup tuple") != std::string::npos;
  })) << testing::PrintToString(incomplete_errors);

  ConSanResult out_of_bounds_tuple = result;
  const auto out_of_bounds_atomic =
      std::ranges::find(out_of_bounds_tuple.patches, ConSanPatchKind::TrampolineMoiAtomicRecord,
                        &ConSanPatchInfo::kind);
  ASSERT_NE(out_of_bounds_atomic, out_of_bounds_tuple.patches.end());
  ASSERT_TRUE(out_of_bounds_atomic->persistent_private_state_end);
  out_of_bounds_atomic->persistent_record_replay_workgroup_private_offsets.z =
      *out_of_bounds_atomic->persistent_private_state_end;
  const std::vector<std::string> out_of_bounds_errors =
      validate_consan_modified_elf(bytes, out_of_bounds_tuple);
  EXPECT_TRUE(std::ranges::any_of(out_of_bounds_errors, [](const std::string &error) {
    return error.find("coordinate outside persistent state") != std::string::npos;
  })) << testing::PrintToString(out_of_bounds_errors);

  const auto expect_mutated_atomic_error =
      [&](std::string_view expected, const std::function<void(ConSanPatchInfo &)> &mutate) {
        ConSanResult invalid = result;
        const auto patch = std::ranges::find(
            invalid.patches, ConSanPatchKind::TrampolineMoiAtomicRecord, &ConSanPatchInfo::kind);
        ASSERT_NE(patch, invalid.patches.end());
        mutate(*patch);
        const std::vector<std::string> errors = validate_consan_modified_elf(bytes, invalid);
        EXPECT_TRUE(std::ranges::any_of(errors, [&](const std::string &error) {
          return error.find(expected) != std::string::npos;
        })) << testing::PrintToString(errors);
      };
  expect_mutated_atomic_error("overlapping private Record/Replay", [](ConSanPatchInfo &patch) {
    patch.persistent_record_replay_workgroup_private_offsets.y =
        patch.persistent_record_replay_workgroup_private_offsets.x;
  });
  expect_mutated_atomic_error("overlapping private Record/Replay", [](ConSanPatchInfo &patch) {
    patch.persistent_record_replay_workgroup_private_offsets.z =
        patch.persistent_record_replay_workgroup_private_offsets.x;
  });
  expect_mutated_atomic_error("persistent-state boundary", [](ConSanPatchInfo &patch) {
    patch.persistent_private_state_end.reset();
  });
  expect_mutated_atomic_error("beyond the patch", [](ConSanPatchInfo &patch) {
    ASSERT_TRUE(patch.persistent_private_state_end);
    patch.required_private_segment_size = *patch.persistent_private_state_end - 1u;
  });
}

TEST(ConSanMoi, Cdna4FirstLightProbeDescriptorGrowthUsesEightVgprGranules) {
  const auto guest = build_cdna4_ds_store_b32(
      /*vaddr=*/6, /*vdata=*/7, /*byte_offset=*/4, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(guest);
  std::vector<uint32_t> text_words(260, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  std::copy(guest->begin(), guest->end(), text_words.begin());
  for (uint16_t vgpr = 1; vgpr < 8; ++vgpr)
    text_words[static_cast<size_t>(vgpr + 1u)] =
        build_v_mov_b32_e32(/*vdst=*/0, vector_source_vgpr(vgpr), ROCJITSU_CODE_ARCH_CDNA4);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);
  const std::vector<uint8_t> bytes = make_cdna4_lds_code_object(text_words, "descriptor_growth");
  ConSanOptions options = moi_options();
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.resource_plans.size(), 1u);
  const ConSanCandidateResourcePlan &plan = result.resource_plans.front();
  EXPECT_EQ(plan.source, ConSanRegisterAllocationSource::DescriptorGrowth);
  EXPECT_EQ(plan.current_vgpr_count, 8u);
  EXPECT_EQ(plan.max_referenced_vgpr_count, 8u);
  EXPECT_EQ(plan.scratch_vgpr, 8u);
  EXPECT_EQ(plan.required_vgpr_count, 14u);
  ASSERT_TRUE(result.modified) << "warnings=" << testing::PrintToString(result.warnings)
                               << " errors=" << testing::PrintToString(result.errors);
  ASSERT_EQ(non_entry_prologue_patch_count(result), 1u);
  EXPECT_EQ(only_non_entry_prologue_patch(result).scratch_vgpr, 8u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.kernels().size(), 1u);
  KD descriptor{};
  std::memcpy(&descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  EXPECT_EQ(AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc1,
                            kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT),
            1u);

  options.scratch_vgpr = 1;
  const ConSanResult odd_scratch = try_patch_consan(bytes, options);
  EXPECT_TRUE(odd_scratch.errors.empty());
  EXPECT_FALSE(odd_scratch.modified);
  ASSERT_EQ(odd_scratch.resource_plans.size(), 1u);
  EXPECT_EQ(odd_scratch.resource_plans.front().source, ConSanRegisterAllocationSource::Unsupported);
  EXPECT_EQ(odd_scratch.resource_plans.front().reason,
            ConSanRegisterPlanReason::ExplicitMisaligned);
}

TEST(ConSanMoi, Cdna4FirstLightTransientSgprsAvoidOldAndGrownPhysicalVcc) {
  const auto guest = build_cdna4_ds_store_b32(
      /*vaddr=*/2, /*vdata=*/3, /*byte_offset=*/4, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(guest);
  std::vector<uint32_t> text_words(300, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  std::copy(guest->begin(), guest->end(), text_words.begin());
  text_words[2] = build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/33u, ROCJITSU_CODE_ARCH_CDNA4);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);
  std::vector<uint8_t> bytes = make_cdna4_lds_code_object(text_words, "physical_vcc_growth");
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    // Five allocation granules give 40 SGPRs and place VCC at s34:s35.
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 4u);
  });
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.resolved_moi_dispatch_id_sgpr);
  ASSERT_TRUE(result.resolved_moi_exec_save_sgpr);
  const uint16_t dispatch_id = *result.resolved_moi_dispatch_id_sgpr;
  const uint16_t exec_save = *result.resolved_moi_exec_save_sgpr;
  const auto overlaps = [](uint16_t lhs_base, uint16_t lhs_width, uint16_t rhs_base,
                           uint16_t rhs_width) {
    return lhs_base < static_cast<uint32_t>(rhs_base) + rhs_width &&
           rhs_base < static_cast<uint32_t>(lhs_base) + lhs_width;
  };
  constexpr uint16_t kOriginalVcc = 34u;
  EXPECT_FALSE(overlaps(dispatch_id, 2u, kOriginalVcc, 2u));
  EXPECT_FALSE(overlaps(exec_save, 5u, kOriginalVcc, 2u));
  EXPECT_FALSE(overlaps(dispatch_id, 2u, exec_save, 5u));
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.kernels().size(), 1u);
  KD descriptor{};
  std::memcpy(&descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  const uint16_t allocated_sgprs = static_cast<uint16_t>(
      (AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc1,
                       kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT) +
       1u) *
      8u);
  ASSERT_GE(allocated_sgprs, 6u);
  const uint16_t grown_vcc = static_cast<uint16_t>(allocated_sgprs - 6u);
  EXPECT_GE(allocated_sgprs, static_cast<uint16_t>(dispatch_id + 2u));
  EXPECT_GE(allocated_sgprs, static_cast<uint16_t>(exec_save + 5u));
  EXPECT_FALSE(overlaps(dispatch_id, 2u, grown_vcc, 2u));
  EXPECT_FALSE(overlaps(exec_save, 5u, grown_vcc, 2u));
}

TEST(ConSanMoi, Cdna3TransientSgprsAvoidOldAndGrownPhysicalVcc) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA3;
  const auto guest = build_cdna3_ds_store_b32(/*vaddr=*/2, /*vdata=*/3, /*byte_offset=*/4, kArch);
  ASSERT_TRUE(guest);
  std::vector<uint32_t> text_words(300, build_s_nop(0, kArch));
  std::ranges::copy(*guest, text_words.begin());
  text_words[2] = build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/33u, kArch);
  text_words.back() = build_s_endpgm(kArch);
  std::vector<uint8_t> bytes = make_cdna3_lds_code_object(text_words, "physical_vcc_growth");
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    // Five allocation granules give 40 SGPRs and place VCC at s34:s35.
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 4u);
  });
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.resolved_moi_dispatch_id_sgpr);
  ASSERT_TRUE(result.resolved_moi_exec_save_sgpr);
  const uint16_t dispatch_id = *result.resolved_moi_dispatch_id_sgpr;
  const uint16_t exec_save = *result.resolved_moi_exec_save_sgpr;
  const auto overlaps = [](uint16_t lhs_base, uint16_t lhs_width, uint16_t rhs_base,
                           uint16_t rhs_width) {
    return lhs_base < static_cast<uint32_t>(rhs_base) + rhs_width &&
           rhs_base < static_cast<uint32_t>(lhs_base) + lhs_width;
  };
  constexpr uint16_t kOriginalVcc = 34u;
  EXPECT_FALSE(overlaps(dispatch_id, 2u, kOriginalVcc, 2u));
  EXPECT_FALSE(overlaps(exec_save, 5u, kOriginalVcc, 2u));
  EXPECT_FALSE(overlaps(dispatch_id, 2u, exec_save, 5u));
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.kernels().size(), 1u);
  KD descriptor{};
  std::memcpy(&descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  const uint16_t allocated_sgprs = static_cast<uint16_t>(
      (AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc1,
                       kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT) +
       1u) *
      8u);
  ASSERT_GE(allocated_sgprs, 6u);
  const uint16_t grown_vcc = static_cast<uint16_t>(allocated_sgprs - 6u);
  EXPECT_GE(allocated_sgprs, static_cast<uint16_t>(dispatch_id + 2u));
  EXPECT_GE(allocated_sgprs, static_cast<uint16_t>(exec_save + 5u));
  EXPECT_FALSE(overlaps(dispatch_id, 2u, grown_vcc, 2u));
  EXPECT_FALSE(overlaps(exec_save, 5u, grown_vcc, 2u));
}

TEST(ConSanMoi, Cdna4FirstLightProbeForcedSpillUsesNativePrivateWindow) {
  std::vector<uint32_t> text_words(260, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  text_words[0] = 0xd81a0004u;
  text_words[1] = 0x00000302u; // ds_write_b32 v2, v3 offset:4
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);
  std::vector<uint8_t> bytes = make_cdna4_lds_code_object(text_words, "forced_spill");
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    descriptor.private_segment_fixed_size = 32;
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT, 1u);
  });
  ConSanOptions options = moi_options();
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().source, ConSanRegisterAllocationSource::SpillRequired);
  EXPECT_EQ(result.resource_plans.front().reason, ConSanRegisterPlanReason::None);
  EXPECT_EQ(result.resource_plans.front().scratch_vgpr_count, 6u);
  EXPECT_EQ(result.resource_plans.front().current_vgpr_count, 8u);
  EXPECT_EQ(result.resource_plans.front().original_private_segment_size, 32u);
  ASSERT_TRUE(result.modified) << "warnings=" << testing::PrintToString(result.warnings)
                               << " errors=" << testing::PrintToString(result.errors)
                               << " dispositions="
                               << testing::PrintToString(result.site_dispositions)
                               << " resources=" << testing::PrintToString(result.resource_plans);
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_TRUE(result.final_validation_passed);
  ASSERT_EQ(non_entry_prologue_patch_count(result), 1u);
  const auto access = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore, &ConSanPatchInfo::kind);
  ASSERT_NE(access, result.patches.end());
  const ConSanPatchInfo &patch = *access;
  EXPECT_EQ(patch.kind, ConSanPatchKind::TrampolineMoiAccessRecordStore);
  ASSERT_TRUE(patch.scratch_vgpr);
  EXPECT_EQ(patch.spilled_vgpr_count, 6u);
  ASSERT_TRUE(patch.persistent_private_state_end);
  EXPECT_TRUE(patch.persistent_record_replay_workgroup_private_offsets.complete());
  EXPECT_LT(*patch.persistent_record_replay_workgroup_private_offsets.x,
            *patch.persistent_private_state_end);
  EXPECT_LT(*patch.persistent_record_replay_workgroup_private_offsets.y,
            *patch.persistent_private_state_end);
  EXPECT_LT(*patch.persistent_record_replay_workgroup_private_offsets.z,
            *patch.persistent_private_state_end);
  EXPECT_EQ(*patch.persistent_private_state_end, 64u);
  EXPECT_EQ(patch.required_private_segment_size, 96u);
  const uint32_t unaligned_private_end =
      *patch.persistent_private_state_end + patch.spilled_vgpr_count * sizeof(uint32_t);
  EXPECT_GE(patch.required_private_segment_size, unaligned_private_end);
  EXPECT_LT(patch.required_private_segment_size, unaligned_private_end + 32u);
  EXPECT_EQ(patch.required_private_segment_size % 32u, 0u);
  EXPECT_EQ(result.resource_plan_summary.spill_plans, 1u);
  // The access probe owns a six-VGPR transient spill window, while the
  // entry-capture prologue independently spills its one-VGPR temporary.
  EXPECT_EQ(result.resource_plan_summary.emitted_spill_patches, 2u);
  EXPECT_EQ(result.resource_plan_summary.emitted_spill_slot_bytes, 28u);

  const auto wait = build_cdna4_s_wait_vmcnt0(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(wait);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const std::vector<uint32_t> patched_words =
      text_words_at_offset(patched, patch.trampoline_offset, patch.trampoline_size);
  const auto guest_at =
      std::ranges::search(patched_words, std::array<uint32_t, 2>{text_words[0], text_words[1]})
          .begin();
  ASSERT_NE(guest_at, patched_words.end());
  const auto first_wait = std::ranges::find(patched_words, *wait);
  ASSERT_NE(first_wait, patched_words.end());
  EXPECT_LT(first_wait, guest_at);
  for (uint16_t i = 0; i < patch.spilled_vgpr_count; ++i) {
    const uint16_t vgpr = static_cast<uint16_t>(*patch.scratch_vgpr + i);
    const uint32_t offset = *patch.persistent_private_state_end + sizeof(uint32_t) * i;
    const auto store =
        build_cdna4_address_free_scratch_store_b32(vgpr, offset, ROCJITSU_CODE_ARCH_CDNA4);
    ASSERT_TRUE(store);
    const auto save_at = std::ranges::search(patched_words, *store).begin();
    ASSERT_NE(save_at, patched_words.end());
    EXPECT_LT(first_wait, save_at);
    EXPECT_LT(save_at, guest_at);
    const auto load =
        build_cdna4_address_free_scratch_load_b32(vgpr, offset, ROCJITSU_CODE_ARCH_CDNA4);
    ASSERT_TRUE(load);
    const auto restore_at = std::ranges::search(patched_words, *load).begin();
    ASSERT_NE(restore_at, patched_words.end());
    EXPECT_LT(save_at, restore_at);
    EXPECT_LT(restore_at, guest_at)
        << "CDNA restores the complete borrowed window before executing the relocated guest";
  }

  ASSERT_EQ(patched.kernels().size(), 1u);
  KD descriptor{};
  std::memcpy(&descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  EXPECT_EQ(descriptor.private_segment_fixed_size, patch.required_private_segment_size);
  EXPECT_EQ(
      AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT),
      1u);
}

TEST(ConSanMoi, Cdna4StaticRecordReplayRestoresOverlappingStoreOperandsBeforeGuest) {
  const auto guest = build_cdna4_ds_store_b32(
      /*vaddr=*/6, /*vdata=*/10, /*byte_offset=*/0, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(guest);
  std::vector<uint32_t> text_words(guest->begin(), guest->end());
  for (uint16_t vgpr = 1; vgpr < 16; ++vgpr)
    text_words.push_back(
        build_v_mov_b32_e32(/*vdst=*/15, vector_source_vgpr(vgpr), ROCJITSU_CODE_ARCH_CDNA4));
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4));
  std::vector<uint8_t> bytes = make_cdna4_lds_code_object(
      text_words, "record_replay_store_operand_overlap", /*vgpr_granulated=*/2u);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    // Bound ordinary VGPRs to v0..v15. Every aligned ten-VGPR window intersects
    // address v6 or payload v10, forcing the overlap-safe spill path.
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 3u);
  });
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  const ConSanMoiAutoReportPlan report_plan = plan_consan_moi_auto_report(
      {.engine = ConSanMoiEngine::RecordReplay, .access_range_count = 1});
  ASSERT_TRUE(report_plan.complete());
  const auto layout_override = consan_moi_auto_report_layout_override(report_plan);
  ASSERT_TRUE(layout_override);
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = report_plan.required_bytes;
  options.moi_report_layout = *layout_override;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_NE(result.resolved_moi_record_replay_workgroup_vgprs.complete(),
            result.resolved_moi_record_replay_workgroup_sgprs.complete())
      << "CDNA4 automatic banked replay must resolve one exact persistent workgroup tuple";
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().source, ConSanRegisterAllocationSource::SpillRequired);
  EXPECT_EQ(result.resource_plans.front().reason, ConSanRegisterPlanReason::None);
  const auto patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &item) {
    return item.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore;
  });
  ASSERT_NE(patch, result.patches.end());
  ASSERT_TRUE(patch->scratch_vgpr);
  ASSERT_EQ(patch->spilled_vgpr_count, 10u);
  const uint16_t scratch = *patch->scratch_vgpr;
  ASSERT_LE(scratch, 6u);
  ASSERT_GT(static_cast<uint32_t>(scratch) + patch->spilled_vgpr_count, 6u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> cave_words =
      text_words_at_offset(patched, patch->trampoline_offset, patch->trampoline_size);
  const auto reload_address = build_cdna4_address_free_scratch_load_b32(
      static_cast<uint16_t>(scratch + 5u), 4u * (6u - scratch), ROCJITSU_CODE_ARCH_CDNA4);
  const auto combine_dispatch_identity = instrumentation::build_v_xor_b32(
      static_cast<uint16_t>(scratch + 7u), vector_source_vgpr(static_cast<uint16_t>(scratch + 7u)),
      static_cast<uint16_t>(scratch + 6u), ROCJITSU_CODE_ARCH_CDNA4);
  const auto range_store = build_cdna4_flat_store_b32(
      scratch, static_cast<uint16_t>(scratch + 5u),
      offsetof(ConSanMoiAccessRecord, lds_byte_offset), ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(reload_address && combine_dispatch_identity && range_store);
  const auto identity_selection_position =
      std::ranges::find(cave_words, *combine_dispatch_identity);
  const auto address_group_reload_position = std::search(
      cave_words.begin(), cave_words.end(), reload_address->begin(), reload_address->end());
  const auto publication_reload_position =
      std::search(identity_selection_position, cave_words.end(), reload_address->begin(),
                  reload_address->end());
  const auto range_store_position =
      std::search(cave_words.begin(), cave_words.end(), range_store->begin(), range_store->end());
  ASSERT_NE(identity_selection_position, cave_words.end());
  ASSERT_NE(address_group_reload_position, cave_words.end());
  ASSERT_NE(publication_reload_position, cave_words.end());
  ASSERT_NE(range_store_position, cave_words.end());
  EXPECT_LT(address_group_reload_position, identity_selection_position)
      << "address-group selection must recover the authoritative overlapping LDS address";
  EXPECT_LT(identity_selection_position, publication_reload_position)
      << "identity-bank selection may reuse the recovered address VGPR";
  EXPECT_LT(publication_reload_position, range_store_position)
      << "the static record must recover the spilled LDS address before publishing its range";
  std::vector<uint32_t> restore;
  for (uint16_t index = 0; index < patch->spilled_vgpr_count; ++index) {
    const auto load = build_cdna4_address_free_scratch_load_b32(
        static_cast<uint16_t>(*patch->scratch_vgpr + index), 4u * index, ROCJITSU_CODE_ARCH_CDNA4);
    ASSERT_TRUE(load);
    restore.insert(restore.end(), load->begin(), load->end());
  }
  const auto wait = build_cdna4_s_wait_vmcnt0(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(wait);
  restore.push_back(*wait);
  const auto restore_position =
      std::search(cave_words.begin(), cave_words.end(), restore.begin(), restore.end());
  ASSERT_NE(restore_position, cave_words.end());
  const auto guest_position =
      std::search(cave_words.begin(), cave_words.end(), guest->begin(), guest->end());
  ASSERT_NE(guest_position, cave_words.end());
  EXPECT_EQ(restore_position + restore.size(), guest_position)
      << "the store must consume restored address and payload operands";
}

TEST(ConSanMoi, Cdna4StaticRecordReplayRestoresOverlappingLoadOperandsBeforeGuest) {
  const auto guest = cdna4::build_ds(cdna4::kDsReadB32Ds, {.addr = 0, .vdst = 4});
  std::vector<uint32_t> text_words(guest.begin(), guest.end());
  for (uint16_t vgpr = 1; vgpr < 8; ++vgpr)
    text_words.push_back(
        build_v_mov_b32_e32(/*vdst=*/7, vector_source_vgpr(vgpr), ROCJITSU_CODE_ARCH_CDNA4));
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4));
  std::vector<uint8_t> bytes = make_cdna4_lds_code_object(
      text_words, "record_replay_load_operand_overlap", /*vgpr_granulated=*/1u);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 1u);
  });
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().source, ConSanRegisterAllocationSource::SpillRequired);
  EXPECT_EQ(result.resource_plans.front().reason, ConSanRegisterPlanReason::None);
  const auto patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &item) {
    return item.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore;
  });
  ASSERT_NE(patch, result.patches.end());
  ASSERT_EQ(patch->scratch_vgpr, 0u);
  ASSERT_EQ(patch->spilled_vgpr_count, 6u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> cave_words =
      text_words_at_offset(patched, patch->trampoline_offset, patch->trampoline_size);
  std::vector<uint32_t> restore;
  for (uint16_t vgpr = 0; vgpr < patch->spilled_vgpr_count; ++vgpr) {
    const auto load =
        build_cdna4_address_free_scratch_load_b32(vgpr, 4u * vgpr, ROCJITSU_CODE_ARCH_CDNA4);
    ASSERT_TRUE(load);
    restore.insert(restore.end(), load->begin(), load->end());
  }
  const auto wait = build_cdna4_s_wait_vmcnt0(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(wait);
  restore.push_back(*wait);
  const auto restore_position =
      std::search(cave_words.begin(), cave_words.end(), restore.begin(), restore.end());
  ASSERT_NE(restore_position, cave_words.end());
  const auto guest_position =
      std::search(cave_words.begin(), cave_words.end(), guest.begin(), guest.end());
  ASSERT_NE(guest_position, cave_words.end());
  EXPECT_EQ(restore_position + restore.size(), guest_position)
      << "the load must consume the restored address before defining its destination";
}

TEST(ConSanMoi, DynamicAccessRecordProbeAppendsPerLaneRecords) {
  std::array<uint32_t, 260> text_words{};
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000000u; // ds_store_b32 v0, v0
  for (size_t i = 2; i + 1 < text_words.size(); ++i)
    text_words[i] = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options = moi_options();
  options.scratch_vgpr = 16;
  options.moi_exec_save_sgpr = 30;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_dynamic_access_records = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(8, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(non_entry_prologue_patch_count(result), 1u);
  const ConSanPatchInfo &patch = only_non_entry_prologue_patch(result);
  ASSERT_TRUE(patch.kind == ConSanPatchKind::InlineMoiAccessRecordStore ||
              patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore);
  ASSERT_TRUE(patch.scratch_vgpr);
  EXPECT_EQ(*patch.scratch_vgpr, 16u);

  const std::vector<uint32_t> rewritten_words =
      record_access_patch_words(result, patch, /*text_file_offset=*/0x100u);

  const uint64_t base = *options.moi_report_buffer_address;
  EXPECT_TRUE(contains_subsequence(
      rewritten_words,
      make_expected_fetch_add_one_words(base + offsetof(ConSanMoiReportHeader, access_record_count),
                                        /*result_vgpr=*/18, /*scratch_vgpr=*/16)));
  EXPECT_TRUE(contains_subsequence(
      rewritten_words,
      make_expected_fetch_add_one_words(base + offsetof(ConSanMoiReportHeader, event_counter),
                                        /*result_vgpr=*/21, /*scratch_vgpr=*/16)));

  const auto compare_capacity =
      build_v_cmp_gt_u32_e32_vcc(vector_source_vgpr(21), /*vsrc1=*/18, ROCJITSU_CODE_ARCH_RDNA4);
  const auto save_exec =
      build_s_and_saveexec_b64(/*sdst=*/30, /*ssrc0=*/kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_exec = build_s_mov_b64(/*sdst=*/126, /*ssrc0=*/30, ROCJITSU_CODE_ARCH_RDNA4);
  const auto save_scc = build_rdna4_s_cselect_b32(
      /*sdst=*/34, scalar_positive_inline_u32(1), scalar_positive_inline_u32(0),
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto save_vcc = build_s_mov_b64(/*sdst=*/32, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_vcc = build_s_mov_b64(kRdna4VccLo, /*ssrc0=*/32, ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_scc = build_rdna4_s_cmp_lg_u32(
      /*ssrc0=*/34, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(compare_capacity);
  ASSERT_TRUE(save_exec);
  ASSERT_TRUE(restore_exec);
  ASSERT_TRUE(save_scc);
  ASSERT_TRUE(save_vcc);
  ASSERT_TRUE(restore_vcc);
  ASSERT_TRUE(restore_scc);
  EXPECT_TRUE(std::find(rewritten_words.begin(), rewritten_words.end(), *save_scc) !=
              rewritten_words.end());
  EXPECT_TRUE(std::find(rewritten_words.begin(), rewritten_words.end(), *save_vcc) !=
              rewritten_words.end());
  EXPECT_TRUE(std::find(rewritten_words.begin(), rewritten_words.end(), *compare_capacity) !=
              rewritten_words.end());
  EXPECT_TRUE(std::find(rewritten_words.begin(), rewritten_words.end(), *save_exec) !=
              rewritten_words.end());
  EXPECT_TRUE(std::find(rewritten_words.begin(), rewritten_words.end(), *restore_exec) !=
              rewritten_words.end());
  EXPECT_TRUE(std::find(rewritten_words.begin(), rewritten_words.end(), *restore_vcc) !=
              rewritten_words.end());
  EXPECT_TRUE(std::find(rewritten_words.begin(), rewritten_words.end(), *restore_scc) !=
              rewritten_words.end());
  const auto save_scc_it = std::find(rewritten_words.begin(), rewritten_words.end(), *save_scc);
  const auto save_vcc_it = std::find(rewritten_words.begin(), rewritten_words.end(), *save_vcc);
  const auto save_exec_it = std::find(rewritten_words.begin(), rewritten_words.end(), *save_exec);
  const auto restore_exec_it =
      std::find(rewritten_words.begin(), rewritten_words.end(), *restore_exec);
  const auto restore_vcc_it =
      std::find(rewritten_words.begin(), rewritten_words.end(), *restore_vcc);
  const auto restore_scc_it =
      std::find(rewritten_words.begin(), rewritten_words.end(), *restore_scc);
  // VCC and SCC use scalar snapshots, so this sequence remains valid even if
  // the incoming EXEC mask has no active lane.
  EXPECT_LT(save_scc_it, save_vcc_it);
  EXPECT_LT(save_vcc_it, save_exec_it);
  EXPECT_LT(restore_exec_it, restore_vcc_it);
  EXPECT_LT(restore_vcc_it, restore_scc_it);
}

TEST(ConSanMoi, DynamicRecordKindsUseLiteralDispatchIdentityAtFullScalarPressure) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  const auto atomic = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/2, /*vsrc=*/1, /*vdst=*/0, /*return_old_value=*/false, /*scope=*/2, kArch);
  const auto wait_store = build_s_wait_storecnt0(kArch);
  ASSERT_TRUE(atomic && wait_store);

  std::vector<uint32_t> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32 v0, v0
      0xBF940000u, // s_barrier_wait -1
      0xEE0B0000u, 0x00000000u,
      0x00000000u, // global_wb
      *wait_store, (*atomic)[0], (*atomic)[1], (*atomic)[2], *wait_store, 0xEE0AC000u, 0x00000000u,
      0x00000000u, // global_inv
  };
  text_words.resize(800u, build_s_nop(0, kArch));

  // This is fixture pressure, not a production register convention. Leave
  // one explicitly configured transient window dead while keeping every
  // other ordinary SGPR live across every record-producing site.
  constexpr uint16_t kOrdinarySgprCount = 106u;
  constexpr uint16_t kExecSaveSgpr = 90u;
  constexpr uint16_t kExecSaveSgprCount = 7u;
  for (uint16_t sgpr = 0; sgpr < kOrdinarySgprCount; ++sgpr) {
    if (sgpr >= kExecSaveSgpr && sgpr < kExecSaveSgpr + kExecSaveSgprCount)
      continue;
    const auto use = build_s_cmp_eq_u32(sgpr, scalar_positive_inline_u32(0), kArch);
    ASSERT_TRUE(use);
    text_words.push_back(*use);
  }
  text_words.push_back(build_s_endpgm(kArch));

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 8u;
  options.moi_exec_save_sgpr = kExecSaveSgpr;
  options.moi_owner_vgpr = 20u;
  options.moi_epoch_vgpr = 21u;
  options.moi_dynamic_access_records = true;
  options.moi_track_barriers = true;
  options.moi_track_atomics = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_dispatch_id = 0x1122334455667788ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(
      /*access_record_capacity=*/8, /*diagnostic_capacity=*/0,
      /*exact_shadow_entry_capacity=*/0, /*sampled_watchpoint_capacity=*/0,
      /*barrier_record_capacity=*/2, /*atomic_record_capacity=*/2,
      /*fence_record_capacity=*/2);
  options.max_patches = 8u;

  const ConSanResult result =
      try_patch_consan(make_rdna4_lds_code_object(text_words, "literal_dynamic_records"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_FALSE(result.resolved_moi_dispatch_id_sgpr);
  EXPECT_TRUE(result.final_validation_passed);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const auto expect_literals = [&](const ConSanPatchInfo &patch, uint16_t value_vgpr_offset) {
    ASSERT_TRUE(patch.scratch_vgpr);
    const std::vector<uint32_t> words =
        patch.kind == ConSanPatchKind::InlineMoiAccessRecordStore
            ? patched_words_at_file_offset(result, 0x100u + patch.anchor_offset,
                                           patch.original_size)
            : text_words_at_offset(patched, patch.trampoline_offset, patch.trampoline_size);
    for (const bool high_word : {false, true}) {
      const uint32_t literal = high_word
                                   ? static_cast<uint32_t>(options.moi_report_dispatch_id >> 32u)
                                   : static_cast<uint32_t>(options.moi_report_dispatch_id);
      const uint16_t value_vgpr = static_cast<uint16_t>(*patch.scratch_vgpr + value_vgpr_offset);
      const auto materialize = instrumentation::build_v_mov_b32_literal(value_vgpr, literal, kArch);
      ASSERT_TRUE(materialize);
      EXPECT_TRUE(contains_subsequence(words, *materialize));
    }
  };

  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::InlineMoiAccessRecordStore ||
           patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore;
  });
  const auto barrier = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord, &ConSanPatchInfo::kind);
  const auto atomic_patch = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiAtomicRecord, &ConSanPatchInfo::kind);
  ASSERT_NE(access, result.patches.end());
  ASSERT_NE(barrier, result.patches.end());
  ASSERT_NE(atomic_patch, result.patches.end());
  expect_literals(*access, /*value_vgpr_offset=*/5u);
  expect_literals(*barrier, /*value_vgpr_offset=*/5u);
  expect_literals(*atomic_patch, /*value_vgpr_offset=*/4u);
}

TEST(ConSanMoi, DynamicAccessRecordReportsBoundedFullSgprFileFailure) {
  std::vector<uint32_t> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32 v0, v0
  };
  // Keep every allocatable SGPR live across the access. A single transient
  // high-register reference is intentionally no longer enough to reject an
  // otherwise dead lower window.
  for (uint16_t sgpr = 0; sgpr < 106u; ++sgpr) {
    const auto use =
        build_s_cmp_eq_u32(sgpr, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_TRUE(use);
    text_words.push_back(*use);
  }
  text_words.resize(360, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);
  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 15u);
  });
  ConSanOptions options = moi_options();
  options.moi_dynamic_access_records = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(8, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  EXPECT_TRUE(result.errors.empty());
  EXPECT_FALSE(result.modified);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().max_referenced_sgpr_count, 106u);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("could not place a fresh automatic EXEC-save SGPR window") !=
           std::string::npos;
  }));
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("requires RJ_CONSAN_MOI_EXEC_SAVE_SGPR") != std::string::npos;
  }));
}

TEST(ConSanMoi, DynamicAccessRecordPreservesWave32AndWave64SpecialState) {
  for (bool wave32 : {false, true}) {
    std::vector<uint32_t> text_words(360, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
    text_words[0] = 0xD8340000u;
    text_words[1] = 0x00000000u; // ds_store_b32 v0, v0
    text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);
    const std::vector<uint8_t> bytes =
        make_rdna4_lds_code_object(text_words, wave32 ? "dynamic_wave32" : "dynamic_wave64",
                                   kRdna4Wave64AllVgprsGranulated, wave32);
    ConSanOptions options = moi_options();
    options.moi_dynamic_access_records = true;
    options.moi_report_buffer_address = 0x123456780000ull;
    options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(8, 0, 0, 0);

    const auto result = try_patch_consan(bytes, options);

    ASSERT_TRUE(consan_patch_succeeded(result));
    ASSERT_TRUE(result.modified);
    ASSERT_TRUE(result.resolved_moi_exec_save_sgpr);
    ASSERT_EQ(non_entry_prologue_patch_count(result), 1u);
    AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(patched.is_valid());
    ASSERT_EQ(patched.kernels().size(), 1u);
    KD descriptor{};
    std::memcpy(&descriptor,
                result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
                sizeof(descriptor));
    EXPECT_EQ(AMDHSA_BITS_GET(descriptor.kernel_code_properties,
                              kd::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32),
              wave32 ? 1u : 0u);

    std::vector<uint32_t> words(only_non_entry_prologue_patch(result).original_size /
                                sizeof(uint32_t));
    std::memcpy(words.data(), result.elf_bytes.data() + 0x100, words.size() * sizeof(uint32_t));
    const uint16_t base = *result.resolved_moi_exec_save_sgpr;
    const auto save_exec = build_s_and_saveexec_b64(base, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
    const auto save_vcc =
        build_s_mov_b64(static_cast<uint16_t>(base + 2u), kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
    const auto save_scc =
        build_rdna4_s_cselect_b32(static_cast<uint16_t>(base + 4u), scalar_positive_inline_u32(1),
                                  scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_TRUE(save_exec);
    ASSERT_TRUE(save_vcc);
    ASSERT_TRUE(save_scc);
    EXPECT_TRUE(std::find(words.begin(), words.end(), *save_exec) != words.end());
    EXPECT_TRUE(contains_subsequence(words, std::array<uint32_t, 2>{*save_scc, *save_vcc}));
  }
}

TEST(ConSanMoi, FirstLightProbeCapturesOwnerFromWorkitemIdAtEntryWhenOwnerVgprIsUnset) {
  std::array<uint32_t, 320> text_words{};
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000000u; // ds_store_b32 v0, v0
  for (size_t i = 2; i + 1 < text_words.size(); ++i)
    text_words[i] = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "lds_probe", kRdna4Wave64AllVgprsGranulated, /*wave32=*/true);
  ConSanOptions options = moi_options();
  options.moi_owner_source = ConSanMoiOwnerSource::Automatic;
  options.scratch_vgpr = 8;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(non_entry_prologue_patch_count(result), 1u);
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::InlineMoiAccessRecordStore ||
           patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore;
  });
  const auto prologue = std::ranges::find(
      result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue, &ConSanPatchInfo::kind);
  ASSERT_NE(access, result.patches.end());
  ASSERT_NE(prologue, result.patches.end());
  EXPECT_TRUE(result.resolved_moi_owner_vgpr || result.resolved_moi_persistent_owner_sgpr);
  EXPECT_NE(result.resolved_moi_record_replay_workgroup_vgprs.complete(),
            result.resolved_moi_record_replay_workgroup_sgprs.complete());

  const std::vector<uint32_t> access_words = record_access_patch_words(result, *access, 0x100u);
  const std::vector<uint32_t> prologue_words =
      text_words_at_offset(AmdGpuCodeObject(result.elf_bytes.data(), result.elf_bytes.size()),
                           prologue->trampoline_offset, prologue->trampoline_size);
  ASSERT_TRUE(result.resolved_moi_owner_vgpr);
  const uint16_t owner_vgpr = *result.resolved_moi_owner_vgpr;
  const auto owner_init = build_v_lshrrev_b32_e32(owner_vgpr, scalar_positive_inline_u32(5), 0,
                                                  ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(owner_init);
  EXPECT_TRUE(contains_subsequence(prologue_words, std::span<const uint32_t>(&*owner_init, 1u)));
  // Entry workitem_id_x is at most 1023; after the wave32 shift this owner is
  // already within the 10-bit record field and needs no late-probe mask.
  EXPECT_TRUE(contains_subsequence(
      access_words,
      make_expected_offset_store_words(offsetof(ConSanMoiAccessRecord, wave_id), owner_vgpr, 8)));
  const auto late_owner_init =
      build_v_lshrrev_b32_e32(10, scalar_positive_inline_u32(5), 0, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(late_owner_init);
  EXPECT_EQ(std::ranges::find(access_words, *late_owner_init), access_words.end());
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, FirstLightProbeCapturesCompleteHardwareGridAtEntry) {
  std::array<uint32_t, 320> text_words{};
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000000u; // ds_store_b32 v0, v0
  for (size_t i = 2; i + 1 < text_words.size(); ++i)
    text_words[i] = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);

  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 5u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X,
                    0u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y,
                    0u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z,
                    0u);
  });

  ConSanOptions options = moi_options();
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 15;
  options.moi_epoch_vgpr = 16;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(non_entry_prologue_patch_count(result), 1u);
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::InlineMoiAccessRecordStore ||
           patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore;
  });
  const auto prologue = std::ranges::find(
      result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue, &ConSanPatchInfo::kind);
  ASSERT_NE(access, result.patches.end());
  ASSERT_NE(prologue, result.patches.end());
  EXPECT_NE(result.resolved_moi_record_replay_workgroup_vgprs.complete(),
            result.resolved_moi_record_replay_workgroup_sgprs.complete());

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> prologue_words =
      text_words_at_offset(patched, prologue->trampoline_offset, prologue->trampoline_size);
  const std::vector<uint32_t> access_words = record_access_patch_words(result, *access, 0x100u);
  const ConSanMoiPersistentWorkgroupRegisters tuple =
      result.resolved_moi_record_replay_workgroup_vgprs.complete()
          ? result.resolved_moi_record_replay_workgroup_vgprs
          : result.resolved_moi_record_replay_workgroup_sgprs;
  ASSERT_TRUE(tuple.complete());
  for (const auto &[destination, source] :
       {std::pair{*tuple.x, ttmp_scalar_operand(kTtmpRdna4GridX)},
        std::pair{*tuple.y, ttmp_scalar_operand(kTtmpRdna4GridYz)},
        std::pair{*tuple.z, ttmp_scalar_operand(kTtmpRdna4GridYz)}}) {
    const uint32_t capture =
        result.resolved_moi_record_replay_workgroup_vgprs.complete()
            ? build_v_mov_b32_e32(destination, source, ROCJITSU_CODE_ARCH_RDNA4)
            : build_s_mov_b32(destination, source, ROCJITSU_CODE_ARCH_RDNA4);
    EXPECT_NE(std::ranges::find(prologue_words, capture), prologue_words.end());
  }
  const std::array<size_t, 3> fields = {
      offsetof(ConSanMoiAccessRecord, workgroup_x),
      offsetof(ConSanMoiAccessRecord, workgroup_y),
      offsetof(ConSanMoiAccessRecord, workgroup_z),
  };
  const std::array<uint16_t, 3> registers = {*tuple.x, *tuple.y, *tuple.z};
  ASSERT_TRUE(access->scratch_vgpr);
  for (size_t index = 0; index < fields.size(); ++index) {
    std::vector<uint32_t> store;
    if (result.resolved_moi_record_replay_workgroup_vgprs.complete()) {
      const uint16_t value_vgpr = static_cast<uint16_t>(*access->scratch_vgpr + 2u);
      store.push_back(build_v_mov_b32_e32(value_vgpr, vector_source_vgpr(registers[index]),
                                          ROCJITSU_CODE_ARCH_RDNA4));
      const std::vector<uint32_t> field_store =
          make_expected_offset_store_words(fields[index], value_vgpr, *access->scratch_vgpr);
      store.insert(store.end(), field_store.begin(), field_store.end());
    } else {
      store = make_expected_scalar_offset_store_words(fields[index], registers[index],
                                                      *access->scratch_vgpr);
    }
    EXPECT_TRUE(contains_subsequence(access_words, store))
        << "scratch=" << *access->scratch_vgpr << " register=" << registers[index]
        << " expected=" << testing::PrintToString(store)
        << " actual=" << testing::PrintToString(access_words);
  }
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, FirstLightProbeSupportsMultiWidthNativeLdsSites) {
  {
    SCOPED_TRACE("ds_load_u8");
    expect_moi_first_light_width(0xD8E80000u, 0x01000009u, 8u, ConSanLdsAccessKind::Read);
  }
  {
    SCOPED_TRACE("ds_load_u8_d16");
    expect_moi_first_light_width(0xDA880000u, 0x01000009u, 8u, ConSanLdsAccessKind::Read);
  }
  {
    SCOPED_TRACE("ds_load_u16");
    expect_moi_first_light_width(0xD8F00000u, 0x01000009u, 16u, ConSanLdsAccessKind::Read);
  }
  {
    SCOPED_TRACE("ds_load_b64");
    expect_moi_first_light_width(0xD9D80000u, 0x01000009u, 64u, ConSanLdsAccessKind::Read);
  }
  {
    SCOPED_TRACE("ds_store_b8");
    expect_moi_first_light_width(0xD8780000u, 0x00000109u, 8u, ConSanLdsAccessKind::Write);
  }
  {
    SCOPED_TRACE("ds_store_b16");
    expect_moi_first_light_width(0xD87C0000u, 0x00000109u, 16u, ConSanLdsAccessKind::Write);
  }
  {
    SCOPED_TRACE("ds_store_b128");
    expect_moi_first_light_width(0xDB7C0000u, 0x00000109u, 128u, ConSanLdsAccessKind::Write);
  }
  {
    SCOPED_TRACE("ds_load_u16_d16");
    expect_moi_first_light_width(0xDA980000u, 0x01000002u, 16u, ConSanLdsAccessKind::Read);
  }
}

TEST(ConSanMoi, AutoReportInventoryReservesRecordReplaySyncHeadroom) {
  // A representative batched validation executes each static synchronization
  // site 32 times per dispatch and measures ten dispatches in one process.
  constexpr uint64_t kBatchedEventsPerStaticSite = 32u * 10u;
  static_assert(kConSanMoiRecordReplayDynamicEventHeadroom >= kBatchedEventsPerStaticSite);
  const std::array<uint32_t, 4> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32 v0, v0
      0xBF940000u, // s_barrier_wait -1
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_track_barriers = true;

  const ConSanResult result = try_patch_consan(bytes, options);
  ASSERT_TRUE(consan_patch_succeeded(result));
  const ConSanMoiAutoReportInventory inventory =
      inventory_consan_moi_auto_report(result, options, bytes);

  EXPECT_EQ(inventory.barrier_event_count, kConSanMoiRecordReplayDynamicEventHeadroom);
  const ConSanMoiAutoReportPlan plan = plan_consan_moi_auto_report(inventory);
  ASSERT_TRUE(plan.complete());
  EXPECT_EQ(plan.layout.barrier_record_capacity, kConSanMoiRecordReplayDynamicEventHeadroom);
}

TEST(ConSanMoi, AutoReportInventoryReservesLaneScaledRecordReplayFenceHeadroom) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  const std::vector<uint32_t> text_words = {
      0xD81A0004u,
      0x00000302u, // ds_write_b32 v2, v3 offset:4
      0xE0A08000u,
      0x00000000u, // buffer_wbl2 sc1
      0xBF8C0F70u, // s_waitcnt vmcnt(0)
      0xDE708004u,
      0x00000100u, // global_store_dword v0, v1, s[0:1] offset:4 sc1
      0xDE508004u,
      0x027F0000u, // global_load_dword v2, v[0:1], off offset:4 sc1
      0xBF8C0F70u, // s_waitcnt vmcnt(0)
      0xE0A48000u,
      0x00000000u, // buffer_inv sc1
      build_s_endpgm(kArch),
  };
  const std::vector<uint8_t> bytes =
      make_cdna4_lds_code_object(text_words, "lane_scaled_fence_headroom");
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_track_barriers = false;
  options.moi_track_atomics = true;

  const ConSanResult result = try_patch_consan(bytes, options);
  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  const ConSanMoiAutoReportInventory inventory =
      inventory_consan_moi_auto_report(result, options, bytes);

  ASSERT_EQ(result.moi_fence_candidates.size(), 2u);
  EXPECT_EQ(inventory.fence_event_count, 2u * kConSanMoiRecordReplayDynamicLaneEventHeadroom);
  const ConSanMoiAutoReportPlan plan = plan_consan_moi_auto_report(inventory);
  ASSERT_TRUE(plan.complete());
  EXPECT_EQ(plan.layout.fence_record_capacity, 2u * kConSanMoiRecordReplayDynamicLaneEventHeadroom);
}

TEST(ConSanMoi, AutoReportInventoryAdaptsRecordReplayGridAndEventHeadroomForFatObjects) {
  ConSanMoiAutoReportInventory inventory;
  inventory.engine = ConSanMoiEngine::RecordReplay;
  // Retain a large static inventory while accounting for the worst case of
  // one distinct dynamic address group per lane.
  inventory.access_range_count = 741u;
  inventory.diagnostic_count = 741u;
  inventory.barrier_event_count = 10000u;
  inventory.record_replay_bank_count_adaptive = true;

  const ConSanMoiAutoReportInventory fitted =
      fit_consan_moi_record_replay_auto_report_inventory(inventory);
  const ConSanMoiAutoReportPlan plan = plan_consan_moi_auto_report(fitted);

  ASSERT_TRUE(plan.complete());
  ASSERT_EQ(fitted.barrier_event_count % 10000u, 0u);
  const uint64_t fitted_event_headroom = fitted.barrier_event_count / 10000u;
  EXPECT_GE(fitted_event_headroom, 1u);
  EXPECT_EQ(fitted_event_headroom & (fitted_event_headroom - 1u), 0u);
  EXPECT_LT(fitted.barrier_event_count, 10000u * kConSanMoiRecordReplayDynamicEventHeadroom);
  EXPECT_LT(fitted.record_replay_access_dispatch_bank_count,
            inventory.record_replay_access_dispatch_bank_count);
  EXPECT_LT(fitted.record_replay_access_owner_bank_count,
            inventory.record_replay_access_owner_bank_count);
  EXPECT_EQ(fitted.record_replay_access_dispatch_bank_count,
            fitted.record_replay_access_owner_bank_count);
  EXPECT_EQ(fitted.diagnostic_count, fitted.access_range_count *
                                         fitted.record_replay_access_dispatch_bank_count *
                                         fitted.record_replay_access_owner_bank_count *
                                         fitted.record_replay_address_group_headroom);
  EXPECT_LE(plan.required_bytes, kConSanMoiRecordReplayAutoReportBufferCeilingBytes);

  auto exact = inventory;
  exact.record_replay_bank_count_adaptive = false;
  const ConSanMoiAutoReportInventory unfitted =
      fit_consan_moi_record_replay_auto_report_inventory(exact);
  EXPECT_EQ(unfitted.record_replay_access_dispatch_bank_count,
            exact.record_replay_access_dispatch_bank_count);
  EXPECT_EQ(unfitted.record_replay_access_owner_bank_count,
            exact.record_replay_access_owner_bank_count);
  EXPECT_EQ(plan_consan_moi_auto_report(unfitted).outcome,
            ConSanMoiAutoReportPlanOutcome::InsufficientReportCapacity);
}

TEST(ConSanMoi, AutoReportInventoryAdaptsAddressGroupHeadroomForVeryLargeObjects) {
  ConSanMoiAutoReportInventory inventory;
  inventory.engine = ConSanMoiEngine::RecordReplay;
  inventory.access_range_count = 50000u;
  inventory.diagnostic_count = inventory.access_range_count;
  inventory.barrier_event_count = 65000u;
  inventory.record_replay_bank_count_adaptive = true;

  const ConSanMoiAutoReportInventory fitted =
      fit_consan_moi_record_replay_auto_report_inventory(inventory);
  const ConSanMoiAutoReportPlan plan = plan_consan_moi_auto_report(fitted);

  ASSERT_TRUE(plan.complete());
  EXPECT_EQ(fitted.access_range_count, inventory.access_range_count);
  EXPECT_EQ(fitted.record_replay_access_dispatch_bank_count, 1u);
  EXPECT_EQ(fitted.record_replay_access_owner_bank_count, 1u);
  EXPECT_LT(fitted.record_replay_address_group_headroom,
            kConSanMoiRecordReplayMaximumAddressGroupsPerWave);
  EXPECT_GE(fitted.record_replay_address_group_headroom, 1u);
  EXPECT_EQ(fitted.diagnostic_count,
            fitted.access_range_count * fitted.record_replay_address_group_headroom);
  EXPECT_GE(plan.layout.access_record_capacity,
            fitted.access_range_count * fitted.record_replay_address_group_headroom);
  EXPECT_LE(plan.required_bytes, kConSanMoiRecordReplayAutoReportBufferCeilingBytes);
}

TEST(ConSanMoi, AutoReportInventoryAdaptsBeforeLargeAccessGeometryExceedsAbiCapacity) {
  ConSanMoiAutoReportInventory inventory;
  inventory.engine = ConSanMoiEngine::RecordReplay;
  inventory.access_range_count = 68190u;
  inventory.diagnostic_count = inventory.access_range_count;
  inventory.barrier_event_count = 6138u;
  inventory.record_replay_bank_count_adaptive = true;

  auto expanded = inventory;
  expanded.barrier_event_count *= kConSanMoiRecordReplayDynamicEventHeadroom;
  expanded.diagnostic_count = expanded.access_range_count *
                              expanded.record_replay_access_dispatch_bank_count *
                              expanded.record_replay_access_owner_bank_count *
                              expanded.record_replay_address_group_headroom;
  const ConSanMoiAutoReportPlan expanded_plan = plan_consan_moi_auto_report(expanded);
  ASSERT_EQ(expanded_plan.outcome, ConSanMoiAutoReportPlanOutcome::Overflow);
  ASSERT_EQ(expanded_plan.reason, ConSanMoiAutoReportPlanReason::AbiGeometryCapacityOverflow);

  const ConSanMoiAutoReportInventory fitted =
      fit_consan_moi_record_replay_auto_report_inventory(inventory);
  const ConSanMoiAutoReportPlan plan = plan_consan_moi_auto_report(fitted);

  ASSERT_TRUE(plan.complete());
  EXPECT_EQ(fitted.access_range_count, inventory.access_range_count);
  EXPECT_GE(fitted.barrier_event_count, inventory.barrier_event_count);
  EXPECT_EQ(fitted.barrier_event_count % inventory.barrier_event_count, 0u);
  EXPECT_LT(fitted.record_replay_access_dispatch_bank_count,
            inventory.record_replay_access_dispatch_bank_count);
  EXPECT_LT(fitted.record_replay_access_owner_bank_count,
            inventory.record_replay_access_owner_bank_count);
  EXPECT_LE(plan.required_bytes, kConSanMoiRecordReplayAutoReportBufferCeilingBytes);
}

TEST(ConSanMoi, AutoReportInventoryDoesNotReduceGeometryForFixedAbiCapacityOverflow) {
  ConSanMoiAutoReportInventory inventory;
  inventory.engine = ConSanMoiEngine::RecordReplay;
  inventory.access_range_count = uint64_t{UINT32_MAX} + 1u;
  inventory.record_replay_bank_count_adaptive = true;

  const ConSanMoiAutoReportPlan initial_plan = plan_consan_moi_auto_report(inventory);
  ASSERT_EQ(initial_plan.outcome, ConSanMoiAutoReportPlanOutcome::Overflow);
  ASSERT_EQ(initial_plan.reason, ConSanMoiAutoReportPlanReason::AbiCapacityOverflow);

  const ConSanMoiAutoReportInventory fitted =
      fit_consan_moi_record_replay_auto_report_inventory(inventory);
  EXPECT_EQ(fitted.access_range_count, inventory.access_range_count);
  EXPECT_EQ(fitted.record_replay_access_dispatch_bank_count,
            inventory.record_replay_access_dispatch_bank_count);
  EXPECT_EQ(fitted.record_replay_access_owner_bank_count,
            inventory.record_replay_access_owner_bank_count);
  EXPECT_EQ(fitted.record_replay_address_group_headroom,
            inventory.record_replay_address_group_headroom);
}

TEST(ConSanMoi, FirstLightProbeAddsNativeLdsImmediateOffset) {
  std::array<uint32_t, 320> text_words{};
  text_words[0] = 0xDA980480u;
  text_words[1] = 0x01000002u; // ds_load_u16_d16 v1, v2 offset:1152
  for (size_t i = 2; i + 1 < text_words.size(); ++i)
    text_words[i] = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options = moi_options();
  options.scratch_vgpr = 20;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(non_entry_prologue_patch_count(result), 1u);
  expect_bounded_static_record_replay_probe_size(
      only_non_entry_prologue_patch(result).original_size);

  const std::vector<uint32_t> rewritten_words = patched_words_at_file_offset(
      result, 0x100, only_non_entry_prologue_patch(result).original_size);

  const auto mov_offset = build_v_mov_b32_e64_literal(22, 1152u, ROCJITSU_CODE_ARCH_RDNA4);
  const auto add_offset =
      build_v_add_nc_u32_e32(22, vector_source_vgpr(2), 22, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(mov_offset);
  ASSERT_TRUE(add_offset);
  std::vector<uint32_t> expected_offset = {mov_offset->at(0), mov_offset->at(1), mov_offset->at(2),
                                           *add_offset};
  EXPECT_TRUE(contains_subsequence(rewritten_words, expected_offset));

  EXPECT_TRUE(contains_subsequence(
      rewritten_words,
      make_expected_offset_store_words(offsetof(ConSanMoiAccessRecord, lds_byte_offset),
                                       /*value_vgpr=*/22, *options.scratch_vgpr)));
  EXPECT_TRUE(contains_subsequence(
      rewritten_words, make_expected_offset_store_words(offsetof(ConSanMoiAccessRecord, start_cell),
                                                        /*value_vgpr=*/22, *options.scratch_vgpr)));
}

TEST(ConSanMoi, FirstLightProbeLowersTwoAddressNativeLdsSitesToTwoRecords) {
  std::vector<uint32_t> text_words(520, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words[0] = 0xD8380201u;
  text_words[1] = 0x00000000u; // ds_store_2addr_b32 v0, v0, v0 offset0:1 offset1:2
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options = moi_options();
  options.scratch_vgpr = 20;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(2, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.moi_candidates.size(), 1u);
  EXPECT_EQ(result.moi_candidates.front().mnemonic, "ds_store_2addr_b32");
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(non_entry_prologue_patch_count(result), 1u);
  const ConSanPatchInfo &patch = only_non_entry_prologue_patch(result);
  ASSERT_TRUE(patch.kind == ConSanPatchKind::InlineMoiAccessRecordStore ||
              patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore);

  const std::vector<uint32_t> rewritten_words =
      record_access_patch_words(result, patch, /*text_file_offset=*/0x100u);

  const std::vector<uint32_t> offset_store = make_expected_offset_store_words(
      offsetof(ConSanMoiAccessRecord, lds_byte_offset), /*value_vgpr=*/22, *options.scratch_vgpr);
  EXPECT_EQ(count_subsequence(rewritten_words, offset_store), 2u);

  const auto mov_offset0 = build_v_mov_b32_e64_literal(22, 4u, ROCJITSU_CODE_ARCH_RDNA4);
  const auto mov_offset1 = build_v_mov_b32_e64_literal(22, 8u, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(mov_offset0);
  ASSERT_TRUE(mov_offset1);
  EXPECT_TRUE(contains_subsequence(rewritten_words, *mov_offset0));
  EXPECT_TRUE(contains_subsequence(rewritten_words, *mov_offset1));
}

TEST(ConSanMoi, FirstLightRecordLinearizesBeforeDisplacedTwoAddressLoad) {
  const std::array<uint32_t, 3> text_words = {
      0xD9DCA1A0u,
      0x10000004u, // ds_load_2addr_b64 v[16:19], v4 offset0:160 offset1:161
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 21;
  options.moi_owner_vgpr = 70;
  options.moi_epoch_vgpr = 71;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(2, 0, 0, 0);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().scratch_vgpr_count, 6u);
  ASSERT_EQ(non_entry_prologue_patch_count(result), 1u);
  const ConSanPatchInfo &patch = only_non_entry_prologue_patch(result);
  ASSERT_EQ(patch.kind, ConSanPatchKind::TrampolineMoiAccessRecordStore);
  // Include the complete hardware workgroup identity while keeping the two-range
  // Record/Replay probe within a tightly bounded append-cave footprint.
  EXPECT_LE(patch.trampoline_size, 2144u)
      << "two-range dispatch/workgroup-qualified Record/Replay probes must remain within the "
         "2144-byte append-cave budget including complete returning-atomic waits";

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const std::vector<uint32_t> cave =
      text_words_at_offset(patched, patch.trampoline_offset, patch.trampoline_size);
  ASSERT_GE(cave.size(), 3u);
  EXPECT_EQ(cave[cave.size() - 3u], text_words[0]);
  EXPECT_EQ(cave[cave.size() - 2u], text_words[1]);
  EXPECT_EQ(std::count(cave.begin(), cave.end(), 0xBFC60000u), 0u)
      << "the guest's following wait retains ownership of LDS completion";
}

TEST(ConSanMoi, DynamicAccessRecordProbeLowersTwoAddressNativeLdsSites) {
  std::vector<uint32_t> text_words(760, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words[0] = 0xD8380201u;
  text_words[1] = 0x00000000u; // ds_store_2addr_b32 v0, v0, v0 offset0:1 offset1:2
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options = moi_options();
  options.scratch_vgpr = 20;
  options.moi_exec_save_sgpr = 30;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_dynamic_access_records = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(8, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(non_entry_prologue_patch_count(result), 1u);
  ASSERT_TRUE(result.resolved_moi_record_replay_workgroup_vgprs.complete());
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::InlineMoiAccessRecordStore ||
           patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore;
  });
  const auto prologue = std::ranges::find(
      result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue, &ConSanPatchInfo::kind);
  ASSERT_NE(access, result.patches.end());
  ASSERT_NE(prologue, result.patches.end());

  const std::vector<uint32_t> rewritten_words = record_access_patch_words(result, *access, 0x100u);
  const auto wait_store = instrumentation::build_s_wait_global_store0(ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_exec =
      build_s_mov_b64(/*sdst=*/126, *options.moi_exec_save_sgpr, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(wait_store);
  ASSERT_TRUE(restore_exec);
  const auto first_wait = std::ranges::find(rewritten_words, *wait_store);
  const auto first_restore = std::ranges::find(rewritten_words, *restore_exec);
  ASSERT_NE(first_wait, rewritten_words.end());
  ASSERT_NE(first_restore, rewritten_words.end());
  EXPECT_LT(first_wait, first_restore);

  ASSERT_TRUE(access->scratch_vgpr);
  const uint16_t workgroup_value_vgpr = static_cast<uint16_t>(*access->scratch_vgpr + 5u);
  for (uint16_t coordinate : {*result.resolved_moi_record_replay_workgroup_vgprs.x,
                              *result.resolved_moi_record_replay_workgroup_vgprs.y,
                              *result.resolved_moi_record_replay_workgroup_vgprs.z}) {
    EXPECT_NE(std::ranges::find(rewritten_words, build_v_mov_b32_e32(workgroup_value_vgpr,
                                                                     vector_source_vgpr(coordinate),
                                                                     ROCJITSU_CODE_ARCH_RDNA4)),
              rewritten_words.end());
  }
  for (uint16_t destination = 0; destination < 256u; ++destination) {
    for (uint16_t ttmp : {kTtmpRdna4GridX, kTtmpRdna4GridYz}) {
      EXPECT_EQ(std::ranges::find(rewritten_words,
                                  build_v_mov_b32_e32(destination, ttmp_scalar_operand(ttmp),
                                                      ROCJITSU_CODE_ARCH_RDNA4)),
                rewritten_words.end());
    }
  }
}

TEST(ConSanMoi, DynamicAccessRecordProbeDrainsTerminalAppendedCaveBeforeEndpgm) {
  const std::array<uint32_t, 3> text_words = {
      0xD8D80000u,
      0x01000000u, // ds_load_b32 v1, v0
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options = moi_options();
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 30;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_dynamic_access_records = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(8, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified);
  ASSERT_EQ(non_entry_prologue_patch_count(result), 1u);
  const ConSanPatchInfo &patch = only_non_entry_prologue_patch(result);
  EXPECT_EQ(patch.kind, ConSanPatchKind::TrampolineMoiAccessRecordStore);
  EXPECT_EQ(patch.anchor_offset, 0u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  std::vector<uint32_t> actual_words(patched.text_sections().front()->size() / sizeof(uint32_t));
  std::memcpy(actual_words.data(), patched.text_sections().front()->data(),
              patched.text_sections().front()->size());
  EXPECT_EQ(actual_words[2], build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4));

  const auto wait_store = instrumentation::build_s_wait_global_store0(ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_exec =
      build_s_mov_b64(/*sdst=*/126, *options.moi_exec_save_sgpr, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(wait_store);
  ASSERT_TRUE(restore_exec);
  const auto cave_begin = actual_words.begin() + patch.trampoline_offset / sizeof(uint32_t);
  const auto wait = std::find(cave_begin, actual_words.end(), *wait_store);
  const auto restore = std::find(cave_begin, actual_words.end(), *restore_exec);
  ASSERT_NE(wait, actual_words.end());
  ASSERT_NE(restore, actual_words.end());
  EXPECT_LT(wait, restore);
  EXPECT_LT(restore, actual_words.end() - 1);
}

TEST(ConSanMoi, DynamicAccessRecordProbePreservesOverlappingLoadAddress) {
  const std::array<uint32_t, 3> text_words = {
      0xD8D80000u,
      0x00000000u, // ds_load_b32 v0, v0
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 30;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_dynamic_access_records = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(8, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified);
  ASSERT_EQ(non_entry_prologue_patch_count(result), 1u);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().scratch_vgpr_count, 7u);
  const ConSanPatchInfo &patch = only_non_entry_prologue_patch(result);
  EXPECT_EQ(patch.kind, ConSanPatchKind::TrampolineMoiAccessRecordStore);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const std::vector<uint32_t> cave =
      text_words_at_offset(patched, patch.trampoline_offset, patch.trampoline_size);
  const uint16_t saved_address_vgpr = 14;
  const uint32_t save_address = build_v_mov_b32_e32(
      saved_address_vgpr, vector_source_vgpr(/*vsrc=*/0), ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_GE(cave.size(), 3u);
  EXPECT_EQ(cave[0], save_address);
  EXPECT_EQ(cave[1], text_words[0]);
  EXPECT_EQ(cave[2], text_words[1]);
  const auto start_cell = build_v_lshrrev_b32_e32(
      /*vdst=*/13, scalar_positive_inline_u32(consan_moi_exact_shadow::granule_shift),
      saved_address_vgpr, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(start_cell);
  EXPECT_NE(std::find(cave.begin() + 3, cave.end(), *start_cell), cave.end());
}

TEST(ConSanMoi, DynamicAccessRecordProbeSkipsImmediateSaveexecRegion) {
  std::vector<uint32_t> text_words(360, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000000u; // ds_store_b32 v0, v0
  const auto save_exec =
      build_s_and_saveexec_b64(/*sdst=*/4, /*ssrc0=*/8, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(save_exec);
  text_words[2] = *save_exec;
  text_words[3] = build_v_mov_b32_e32(/*vdst=*/1, /*src=*/scalar_positive_inline_u32(0),
                                      ROCJITSU_CODE_ARCH_RDNA4);
  text_words[4] = 0xD8340020u;
  text_words[5] = 0x00000901u; // ds_store_b32 v1, v9 offset:32
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options = moi_options();
  options.scratch_vgpr = 20;
  options.moi_exec_save_sgpr = 30;
  options.moi_dynamic_access_records = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(8, 0, 0, 0);
  options.max_patches = 2;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(non_entry_prologue_patch_count(result), 1u);
  EXPECT_EQ(only_non_entry_prologue_patch(result).anchor_offset, 0u);
  bool saw_saveexec_warning = false;
  for (const std::string &warning : result.warnings)
    saw_saveexec_warning |= warning.find("immediately after s_*_saveexec") != std::string::npos;
  EXPECT_TRUE(saw_saveexec_warning);
}

TEST(ConSanMoi, FirstLightProbeCanPatchTwoNativeLdsAccessRecords) {
  constexpr uint32_t kSecondSiteWord = 320;
  std::vector<uint32_t> text_words(680, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000000u; // ds_store_b32 v0, v0
  text_words[kSecondSiteWord] = 0xD8D80000u;
  text_words[kSecondSiteWord + 1] = 0x01000000u; // ds_load_b32 v1, v0
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options = moi_options();
  options.scratch_vgpr = 20;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(2, 0, 0, 0);
  options.max_patches = 2;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(non_entry_prologue_patch_count(result), 2u);
  EXPECT_EQ(result.patches[0].kind, ConSanPatchKind::InlineMoiAccessRecordStore);
  EXPECT_EQ(result.patches[0].anchor_offset, 0u);
  expect_bounded_static_record_replay_probe_size(result.patches[0].original_size);
  EXPECT_LT(result.patches[0].original_size,
            static_cast<uint64_t>(kSecondSiteWord) * sizeof(uint32_t));
  EXPECT_EQ(result.patches[1].kind, ConSanPatchKind::InlineMoiAccessRecordStore);
  EXPECT_EQ(result.patches[1].anchor_offset, kSecondSiteWord * sizeof(uint32_t));
  expect_bounded_static_record_replay_probe_size(result.patches[1].original_size);

  std::vector<uint32_t> first_words(result.patches[0].original_size / sizeof(uint32_t));
  std::vector<uint32_t> second_words(result.patches[1].original_size / sizeof(uint32_t));
  std::memcpy(first_words.data(), result.elf_bytes.data() + 0x100,
              first_words.size() * sizeof(uint32_t));
  std::memcpy(second_words.data(),
              result.elf_bytes.data() + 0x100 +
                  static_cast<uint64_t>(kSecondSiteWord) * sizeof(uint32_t),
              second_words.size() * sizeof(uint32_t));
  ASSERT_GE(first_words.size(), 2u);
  ASSERT_GE(second_words.size(), 2u);
  EXPECT_EQ(first_words[first_words.size() - 2u], 0xD8340000u);
  EXPECT_EQ(first_words.back(), 0x00000000u);
  EXPECT_EQ(second_words[second_words.size() - 2u], 0xD8D80000u);
  EXPECT_EQ(second_words.back(), 0x01000000u);
  EXPECT_EQ(std::count(first_words.begin(), first_words.end(), 0xBFC60000u), 0u);
  EXPECT_EQ(std::count(second_words.begin(), second_words.end(), 0xBFC60000u), 0u);
}

TEST(ConSanMoi, FirstLightProbeCanPatchTwoAppendedCaveAccessRecords) {
  const std::array<uint32_t, 5> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32 v0, v0
      0xD8D80000u,
      0x01000000u, // ds_load_b32 v1, v0
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options = moi_options();
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 14;
  options.moi_epoch_vgpr = 15;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(2, 0, 0, 0);
  options.max_patches = 2;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(non_entry_prologue_patch_count(result), 2u);
  EXPECT_EQ(result.patches[0].kind, ConSanPatchKind::TrampolineMoiAccessRecordStore);
  EXPECT_EQ(result.patches[0].anchor_offset, 0u);
  EXPECT_EQ(result.patches[0].original_size, 2u * sizeof(uint32_t));
  EXPECT_EQ(result.patches[1].kind, ConSanPatchKind::TrampolineMoiAccessRecordStore);
  EXPECT_EQ(result.patches[1].anchor_offset, 2u * sizeof(uint32_t));
  EXPECT_EQ(result.patches[1].original_size, 2u * sizeof(uint32_t));

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  EXPECT_GT(patched.text_sections().front()->size(), text_words.size() * sizeof(uint32_t));
}

TEST(ConSanMoi, RecordReplayFindsDeadSgprsBelowAHighTransientReference) {
  const std::array<uint32_t, 4> text_words = {
      build_s_mov_b32(/*sdst=*/104, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4),
      0xD8340000u,
      0x00000000u, // ds_store_b32 v0, v0
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_EQ(result.resolved_moi_exec_save_sgpr, 0u);
  EXPECT_TRUE(result.moi_exec_save_sgprs_automatic);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("liveness-dead EXEC-save SGPRs s0:s4") != std::string::npos;
  }));
}

TEST(ConSanMoi, RecordReplayDeadSgprWindowRejectsAnyLiveLane) {
  const std::array<uint32_t, 5> text_words = {
      build_s_mov_b32(/*sdst=*/104, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4),
      0xD8340000u,
      0x00000000u, // ds_store_b32 v0, v0
      build_s_mov_b32(/*sdst=*/0, /*ssrc0=*/2, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_EQ(result.resolved_moi_exec_save_sgpr, 4u);
  EXPECT_TRUE(result.moi_exec_save_sgprs_automatic);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("liveness-dead EXEC-save SGPRs s4:s8") != std::string::npos;
  }));
}

TEST(ConSanMoi, RecordReplayAutomaticExecSaveUsesSafePerOwnerWindows) {
  const auto make_owner = [](uint16_t first_live, uint16_t last_live, uint16_t dead_destination) {
    std::vector<uint32_t> words = {
        0xD8340000u,
        0x00000000u, // ds_store_b32 v0, v0
    };
    for (uint16_t sgpr = first_live; sgpr <= last_live; ++sgpr) {
      words.push_back(build_s_mov_b32(dead_destination, sgpr, ROCJITSU_CODE_ARCH_CDNA5));
    }
    words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5));
    return words;
  };
  // The first owner leaves only s0:s7 dead at its access. The second leaves
  // only s97:s105 dead. Their union has no code-object-wide five-SGPR
  // window, but each independent owner has a safe transient window below the
  // architectural aliases at s102:s105.
  const std::vector<uint32_t> first_words =
      make_owner(/*first_live=*/8u, /*last_live=*/105u, /*dead_destination=*/0u);
  const std::vector<uint32_t> second_words =
      make_owner(/*first_live=*/0u, /*last_live=*/96u, /*dead_destination=*/96u);
  const std::vector<uint8_t> bytes = make_gfx1250_code_object_with_local_function(
      first_words, second_words, {}, kRdna4Wave64AllVgprsGranulated,
      /*function_is_kernel=*/true);

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(2, 0, 0, 0);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.max_patches = 2;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  ASSERT_TRUE(result.resolved_moi_exec_save_sgpr);
  ASSERT_EQ(result.resolved_moi_transient_sgpr_assignments.size(), 2u);
  for (const ConSanMoiTransientSgprAssignment &assignment :
       result.resolved_moi_transient_sgpr_assignments) {
    SCOPED_TRACE(assignment.descriptor_file_offset);
    EXPECT_LE(assignment.exec_save_sgpr + 5u, 102u);
  }
  EXPECT_EQ(std::ranges::count_if(result.resolved_moi_transient_sgpr_assignments,
                                  [](const auto &assignment) {
                                    return assignment.spill_backed &&
                                           assignment.dispatch_id_sgpr.has_value();
                                  }),
            1u);
  EXPECT_EQ(std::ranges::count_if(result.resolved_moi_transient_sgpr_assignments,
                                  [](const auto &assignment) {
                                    return !assignment.spill_backed &&
                                           !assignment.dispatch_id_sgpr.has_value();
                                  }),
            1u);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               &ConSanPatchInfo::kind),
            2u);
  EXPECT_TRUE(std::ranges::all_of(result.resource_plans, [](const auto &plan) {
    return plan.site_kind != ConSanResourceSiteKind::Access ||
           plan.source != ConSanRegisterAllocationSource::Unsupported;
  }));
}

TEST(ConSanMoi, Gfx1250SparseRecordReplaySpillPreservesKernargPreloadScalarWindow) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA5;
  constexpr uint16_t kExecWindowSize = 7u;
  constexpr uint32_t kAccessCount = 9u;
  constexpr std::array<uint16_t, 4> kRouterScalars = {66u, 67u, 75u, 76u};

  // This is the compact scalar shape distilled from Stream-K's preloaded
  // GFX1250 kernels. Every eight-SGPR window is live at the dense LDS sites,
  // while one PC pair and two independent scalars remain available to route
  // a spill-backed probe. Sparse workgroup selection also needs that borrowed
  // window at kernel entry, before any site-local spill can protect it.
  std::vector<uint32_t> words;
  for (uint32_t access = 0u; access < kAccessCount; ++access) {
    words.push_back(0xD8340000u | access * sizeof(uint32_t));
    words.push_back(0x00000000u); // ds_store_b32 v0, v0 offset:access*4
  }
  for (uint16_t sgpr = 0u; sgpr < 106u; ++sgpr) {
    if (std::ranges::find(kRouterScalars, sgpr) == kRouterScalars.end())
      words.push_back(build_s_mov_b32(/*M0=*/125u, sgpr, kArch));
  }
  words.push_back(build_s_endpgm(kArch));
  std::vector<uint8_t> bytes =
      make_gfx1250_code_object(words, "gfx1250_sparse_spill_kernarg_preload");
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                    kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 32u);
    AMDHSA_BITS_SET(descriptor.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_LENGTH, 8u);
    AMDHSA_BITS_SET(descriptor.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_OFFSET, 0u);
  });

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 8u;
  options.moi_owner_vgpr = 40u;
  options.moi_epoch_vgpr = 41u;
  options.moi_init_owner_epoch = true;
  options.moi_runtime_sample_stride = 65'536u;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(kAccessCount, 0u, 0u, 0u);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.max_patches = kAccessCount;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.resolved_moi_transient_sgpr_assignments.size(), 1u);
  const ConSanMoiTransientSgprAssignment &assignment =
      result.resolved_moi_transient_sgpr_assignments.front();
  ASSERT_TRUE(assignment.spill_backed);
  ASSERT_FALSE(assignment.branch_only_scalar_spill);
  ASSERT_TRUE(assignment.indirect_pc_sgpr);
  ASSERT_TRUE(assignment.indirect_scc_sgpr);
  ASSERT_TRUE(assignment.dispatch_key_sgpr);
  ASSERT_TRUE(assignment.call_return_sgpr);

  const auto prologue = std::ranges::find(
      result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue, &ConSanPatchInfo::kind);
  ASSERT_NE(prologue, result.patches.end());
  ASSERT_TRUE(prologue->entry_scalar_backup_vgpr);
  EXPECT_EQ(prologue->entry_scalar_backup_sgpr_base, assignment.exec_save_sgpr);
  EXPECT_EQ(prologue->entry_scalar_backup_sgpr_count, kExecWindowSize);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> prologue_words =
      text_words_at_offset(patched, prologue->trampoline_offset, prologue->trampoline_size);
  const auto first_save = instrumentation::build_v_writelane_b32(
      *prologue->entry_scalar_backup_vgpr, assignment.exec_save_sgpr, /*lane=*/0u, kArch);
  const auto last_restore = instrumentation::build_v_readlane_b32(
      static_cast<uint16_t>(assignment.exec_save_sgpr + kExecWindowSize - 1u),
      *prologue->entry_scalar_backup_vgpr, kExecWindowSize - 1u, kArch);
  ASSERT_TRUE(first_save);
  ASSERT_TRUE(last_restore);
  EXPECT_TRUE(contains_subsequence(prologue_words, *first_save));
  EXPECT_TRUE(contains_subsequence(prologue_words, *last_restore));
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, SparseRecordReplaySpillPreservesEntryHashWindowAcrossTargets) {
  constexpr uint16_t kExecSaveSgpr = 80u;
  constexpr uint16_t kExecWindowSize = 7u;
  for (const rj_code_arch_t arch : {
           ROCJITSU_CODE_ARCH_CDNA3,
           ROCJITSU_CODE_ARCH_CDNA4,
           ROCJITSU_CODE_ARCH_CDNA5,
           ROCJITSU_CODE_ARCH_RDNA3,
           ROCJITSU_CODE_ARCH_RDNA4,
       }) {
    SCOPED_TRACE("arch=" + std::to_string(arch));
    std::vector<uint32_t> words;
    if (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4) {
      const auto access = arch == ROCJITSU_CODE_ARCH_CDNA3
                              ? build_cdna3_ds_store_b32(/*vaddr=*/0u, /*vdata=*/1u,
                                                         /*byte_offset=*/0u, arch)
                              : build_cdna4_ds_store_b32(/*vaddr=*/0u, /*vdata=*/1u,
                                                         /*byte_offset=*/0u, arch);
      ASSERT_TRUE(access);
      words.insert(words.end(), access->begin(), access->end());
    } else {
      words.push_back(0xD8340000u);
      words.push_back(0x00000000u); // ds_store_b32 v0, v0
    }
    words.push_back(build_s_endpgm(arch));

    const std::vector<uint8_t> bytes = [&]() {
      switch (arch) {
      case ROCJITSU_CODE_ARCH_CDNA3:
        return make_cdna3_lds_code_object(words, "cdna3_sparse_spill_entry_hash");
      case ROCJITSU_CODE_ARCH_CDNA4:
        return make_cdna4_lds_code_object(words, "cdna4_sparse_spill_entry_hash");
      case ROCJITSU_CODE_ARCH_CDNA5:
        return make_gfx1250_code_object(words, "gfx1250_sparse_spill_entry_hash");
      case ROCJITSU_CODE_ARCH_RDNA3:
        return make_rdna3_lds_code_object(words, "rdna3_sparse_spill_entry_hash");
      case ROCJITSU_CODE_ARCH_RDNA4:
        return make_rdna4_lds_code_object(words, "rdna4_sparse_spill_entry_hash");
      default:
        return std::vector<uint8_t>{};
      }
    }();
    ASSERT_FALSE(bytes.empty());

    ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
    options.scratch_vgpr = 8u;
    options.moi_owner_vgpr = 40u;
    options.moi_epoch_vgpr = 41u;
    options.moi_init_owner_epoch = true;
    options.moi_exec_save_sgpr = kExecSaveSgpr;
    options.automatic_moi_record_replay_sgpr_spill = true;
    options.moi_dispatch_id_sgpr = 60u;
    options.moi_record_replay_workgroup_sgprs = {.x = 50u, .y = 51u, .z = 52u};
    options.moi_runtime_sample_stride = 65'536u;
    options.moi_report_buffer_address = 0x123456780000ull;
    options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1u, 0u, 0u, 0u);
    options.moi_track_barriers = false;
    options.moi_track_atomics = false;
    options.max_patches = 1u;

    const ConSanResult result = try_patch_consan(bytes, options);

    ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
    ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
    const auto prologue = std::ranges::find(
        result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue, &ConSanPatchInfo::kind);
    ASSERT_NE(prologue, result.patches.end());
    ASSERT_TRUE(prologue->entry_scalar_backup_vgpr);
    EXPECT_EQ(prologue->entry_scalar_backup_sgpr_base, kExecSaveSgpr);
    EXPECT_EQ(prologue->entry_scalar_backup_sgpr_count, kExecWindowSize);

    AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(patched.is_valid());
    const std::vector<uint32_t> prologue_words =
        text_words_at_offset(patched, prologue->trampoline_offset, prologue->trampoline_size);
    const auto first_save = instrumentation::build_v_writelane_b32(
        *prologue->entry_scalar_backup_vgpr, kExecSaveSgpr, /*lane=*/0u, arch);
    const auto last_restore = instrumentation::build_v_readlane_b32(
        kExecSaveSgpr + kExecWindowSize - 1u, *prologue->entry_scalar_backup_vgpr,
        kExecWindowSize - 1u, arch);
    ASSERT_TRUE(first_save);
    ASSERT_TRUE(last_restore);
    EXPECT_TRUE(contains_subsequence(prologue_words, *first_save));
    EXPECT_TRUE(contains_subsequence(prologue_words, *last_restore));
    EXPECT_TRUE(result.final_validation_passed);
  }
}

TEST(ConSanMoi, Gfx1250RecordReplayKeepsDispatchOnlyFullPressureOwner) {
  std::vector<uint32_t> low_pressure_words(320u, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5));
  low_pressure_words[0] = 0xD8340000u;
  low_pressure_words[1] = 0x00000000u; // ds_store_b32 v0, v0
  low_pressure_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5);

  std::vector<uint32_t> full_pressure_words = low_pressure_words;
  full_pressure_words[2] = build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/105u, ROCJITSU_CODE_ARCH_CDNA5);
  const std::vector<uint8_t> bytes = make_gfx1250_code_object_with_local_function(
      low_pressure_words, full_pressure_words, {}, kRdna4Wave64AllVgprsGranulated,
      /*function_is_kernel=*/true);

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(2, 0, 0, 0);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.max_patches = 2;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  ASSERT_TRUE(result.resolved_moi_dispatch_id_sgpr);
  // The full-pressure owner references the architectural s105 alias. The
  // allocator must remain below the complete s102:s105 special range and use
  // the highest safe ordinary pair.
  EXPECT_EQ(*result.resolved_moi_dispatch_id_sgpr, 100u);
  EXPECT_EQ(
      std::ranges::count_if(result.patches,
                            [](const ConSanPatchInfo &patch) {
                              return patch.kind == ConSanPatchKind::InlineMoiAccessRecordStore ||
                                     patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore;
                            }),
      2u);
  EXPECT_TRUE(std::ranges::all_of(result.resource_plans, [](const auto &plan) {
    return plan.site_kind != ConSanResourceSiteKind::Access ||
           plan.source != ConSanRegisterAllocationSource::Unsupported;
  }));
  const auto full_pressure_kernel =
      std::ranges::find(result.kernels, "lds_helper", &ConSanKernelInfo::name);
  ASSERT_NE(full_pressure_kernel, result.kernels.end());
  const auto full_pressure_assignment = std::ranges::find_if(
      result.resolved_moi_transient_sgpr_assignments, [&](const auto &assignment) {
        return assignment.descriptor_file_offset == full_pressure_kernel->descriptor_file_offset;
      });
  ASSERT_NE(full_pressure_assignment, result.resolved_moi_transient_sgpr_assignments.end());
  EXPECT_FALSE(full_pressure_assignment->dispatch_id_sgpr);
  EXPECT_EQ(result.resolved_moi_transient_sgpr_assignments.size(), 1u);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("owner-local zero-generation records") != std::string::npos;
  }));
}

TEST(ConSanMoi, RecordReplaySharedHelperUsesOneSpillBackedWindowAcrossOwners) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.first_wave32 = true;
  fixture.second_wave32 = true;
  fixture.unrelated_has_lds = true;
  for (uint16_t sgpr = 8; sgpr <= 105; ++sgpr)
    fixture.first_continuation_live_sgprs.push_back(sgpr);
  for (uint16_t sgpr = 0; sgpr <= 97; ++sgpr)
    fixture.second_continuation_live_sgprs.push_back(sgpr);
  std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object(fixture);
  mutate_elf_header(bytes,
                    [](Elf64_Ehdr &header) { header.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX1250; });

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(2, 0, 0, 0);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.max_patches = 2;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  const auto shared_plan = std::ranges::find_if(result.resource_plans, [](const auto &plan) {
    return plan.site_kind == ConSanResourceSiteKind::Access &&
           plan.owner_descriptor_file_offsets.size() == 2u;
  });
  ASSERT_NE(shared_plan, result.resource_plans.end());
  EXPECT_NE(shared_plan->source, ConSanRegisterAllocationSource::Unsupported);
  EXPECT_EQ(shared_plan->reason, ConSanRegisterPlanReason::None);

  // The shared helper has one text body. Its callers cannot use incompatible
  // liveness-dead windows, but they can share one borrowed window because the
  // branch-only route preserves that complete window before using it.
  const ConSanMoiTransientSgprAssignment *common = nullptr;
  for (uint64_t owner : shared_plan->owner_descriptor_file_offsets) {
    const auto assignment =
        std::ranges::find(result.resolved_moi_transient_sgpr_assignments, owner,
                          &ConSanMoiTransientSgprAssignment::descriptor_file_offset);
    ASSERT_NE(assignment, result.resolved_moi_transient_sgpr_assignments.end());
    EXPECT_TRUE(assignment->spill_backed);
    EXPECT_TRUE(assignment->branch_only_scalar_spill);
    EXPECT_TRUE(assignment->indirect_pc_sgpr);
    EXPECT_TRUE(assignment->indirect_scc_sgpr);
    if (common)
      EXPECT_EQ(assignment->exec_save_sgpr, common->exec_save_sgpr);
    else
      common = &*assignment;
  }
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               &ConSanPatchInfo::kind),
            2u);
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, RecordReplaySpillsExecVccStateOnRdna) {
  for (const rj_code_arch_t arch :
       {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    for (const bool uses_dynamic_stack : {false, true}) {
      SCOPED_TRACE(arch);
      SCOPED_TRACE(uses_dynamic_stack ? "dynamic-stack" : "fixed-stack");
      const std::array<uint16_t, 4> dead = {0u, 1u, 4u, 6u};
      const uint32_t access_count = arch == ROCJITSU_CODE_ARCH_CDNA5 ? 9u : 1u;
      std::vector<uint32_t> words;
      for (uint32_t index = 0; index < access_count; ++index) {
        words.push_back(0xD8340000u);
        words.push_back(0x00000000u); // ds_store_b32 v0, v0
      }
      if (arch == ROCJITSU_CODE_ARCH_RDNA4) {
        words.push_back(0xBF940000u); // s_barrier_wait -1
        words.push_back(0xBF940000u); // s_barrier_wait -1
      }
      if (arch == ROCJITSU_CODE_ARCH_RDNA4) {
        for (uint32_t padding = 0; padding < 16u; ++padding)
          words.push_back(build_s_nop(0, arch));
      }
      for (uint16_t sgpr = 0; sgpr < 106u; ++sgpr) {
        if (std::ranges::find(dead, sgpr) == dead.end())
          words.push_back(build_s_mov_b32(/*sdst=*/0u, sgpr, arch));
      }
      words.push_back(build_s_endpgm(arch));
      const std::vector<uint8_t> bytes =
          arch == ROCJITSU_CODE_ARCH_CDNA5
              ? make_gfx1250_code_object(words, "record_replay_scalar_spill",
                                         kRdna4Wave64AllVgprsGranulated, /*wave32=*/true,
                                         uses_dynamic_stack)
          : arch == ROCJITSU_CODE_ARCH_RDNA3
              ? make_rdna3_lds_code_object(words, "record_replay_scalar_spill",
                                           kRdna4Wave64AllVgprsGranulated,
                                           /*wave32=*/false, uses_dynamic_stack,
                                           /*workgroup_id_dimension_mask=*/7u)
              : make_rdna4_lds_code_object(words, "record_replay_scalar_spill",
                                           kRdna4Wave64AllVgprsGranulated,
                                           /*wave32=*/false, uses_dynamic_stack);

      ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
      options.scratch_vgpr = 8;
      options.moi_owner_vgpr = 40;
      options.moi_epoch_vgpr = 41;
      options.moi_report_buffer_address = 0x123456780000ull;
      options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(
          access_count, 0, 0, 0, arch == ROCJITSU_CODE_ARCH_RDNA4 ? 2u : 0u);
      options.moi_track_barriers = arch == ROCJITSU_CODE_ARCH_RDNA4;
      options.moi_track_atomics = false;
      options.max_patches = access_count;

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
      EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                                   &ConSanPatchInfo::kind),
                access_count);
      if (uses_dynamic_stack) {
        EXPECT_TRUE(std::ranges::any_of(result.patches, [](const ConSanPatchInfo &patch) {
          return patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore &&
                 patch.dynamic_private_segment_addend > 0u;
        }));
      } else {
        const auto access_patch =
            std::ranges::find(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                              &ConSanPatchInfo::kind);
        ASSERT_NE(access_patch, result.patches.end());
        AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
        ASSERT_TRUE(patched.is_valid());
        expect_lane_backed_scalar_spill(result, patched, *access_patch, assignment.exec_save_sgpr,
                                        arch);
      }
      EXPECT_TRUE(result.final_validation_passed);
    }
  }
}

TEST(ConSanMoi, Gfx1100RecordReplayRoutesDenseBarrierInventory) {
  constexpr uint32_t kSiteCount = 33u;
  constexpr auto store = rdna3::build_ds(rdna3::kDsStoreB32Ds, {.addr = 0u, .data0 = 1u});
  const auto barrier = build_rdna3_s_barrier(ROCJITSU_CODE_ARCH_RDNA3);
  ASSERT_TRUE(barrier);
  std::vector<uint32_t> words;
  words.reserve(kSiteCount * 3u + 1u);
  for (uint32_t index = 0u; index < kSiteCount; ++index) {
    words.insert(words.end(), store.begin(), store.end());
    words.push_back(*barrier);
  }
  words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA3));

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 8u;
  options.moi_exec_save_sgpr = 80u;
  options.moi_owner_vgpr = 40u;
  options.moi_epoch_vgpr = 41u;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size =
      consan_moi_report_buffer_min_bytes(kSiteCount, 0u, 0u, 0u, kSiteCount);
  options.moi_track_barriers = true;
  options.moi_track_atomics = false;
  options.max_patches = 2u * kSiteCount;

  const ConSanResult result =
      try_patch_consan(make_rdna3_lds_code_object(words, "gfx1100_dense_barriers",
                                                  kRdna4Wave64AllVgprsGranulated, /*wave32=*/false,
                                                  /*uses_dynamic_stack=*/false,
                                                  /*workgroup_id_dimension_mask=*/7u),
                       options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               &ConSanPatchInfo::kind),
            kSiteCount);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord,
                               &ConSanPatchInfo::kind),
            kSiteCount);
  EXPECT_TRUE(std::ranges::any_of(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiIndirectBranchIsland;
  }));
}

TEST(ConSanMoi, RecordReplayDynamicStackKernelEntryRelayUsesSpecialStateOnly) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  constexpr size_t kAccessWord = 64u;
  std::vector<uint32_t> words(33000u, build_s_nop(0, kArch));
  words.front() = build_s_mov_b32(/*sdst=*/33u, scalar_positive_inline_u32(0u), kArch);
  words[kAccessWord] = 0xD8340000u;
  words[kAccessWord + 1u] = 0x00000000u; // ds_store_b32 v0, v0
  words.back() = build_s_endpgm(kArch);
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      words, "entry_safe_dynamic_stack", kRdna4Wave64AllVgprsGranulated, /*wave32=*/false,
      /*uses_dynamic_stack=*/true);

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 8u;
  options.moi_exec_save_sgpr = 80u;
  options.moi_owner_vgpr = 40u;
  options.moi_epoch_vgpr = 41u;
  options.moi_init_owner_epoch = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1u, 0u, 0u, 0u);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.max_patches = 1u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.final_validation_passed);
  const auto prologue = std::ranges::find(
      result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue, &ConSanPatchInfo::kind);
  ASSERT_NE(prologue, result.patches.end());
  ASSERT_FALSE(compute_sopp_branch_simm16(/*branch_pc=*/0u, prologue->trampoline_offset))
      << "prologue_offset=" << prologue->trampoline_offset;
  const auto entry_relay = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiIndirectBranchIsland &&
           patch.anchor_offset == 0u && patch.trampoline_size == 7u * sizeof(uint32_t);
  });
  const uint64_t relay_offset =
      entry_relay == result.patches.end() ? 0u : entry_relay->trampoline_offset;
  if (entry_relay == result.patches.end()) {
    ASSERT_EQ(prologue->original_size, 7u * sizeof(uint32_t));
  }

  std::vector<uint32_t> expected = {build_s_getpc_b64(kRdna4VccLo, kArch)};
  const uint64_t pc_after_getpc = relay_offset + sizeof(uint32_t);
  ASSERT_TRUE(append_pc_delta_builder(expected, kArch, kRdna4VccLo,
                                      static_cast<int64_t>(prologue->trampoline_offset) -
                                          static_cast<int64_t>(pc_after_getpc)));
  const auto dependency_delay = instrumentation::build_salu_dependency_delay(kArch);
  ASSERT_TRUE(dependency_delay);
  expected.push_back(*dependency_delay);
  expected.push_back(build_s_nop(0, kArch));
  expected.push_back(build_s_setpc_b64(kRdna4VccLo, kArch));

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  EXPECT_EQ(text_words_at_offset(patched, relay_offset, 7u * sizeof(uint32_t)), expected);
}

TEST(ConSanMoi, Gfx1250OwnerPrologueUsesOwnerLocalEntryAndReturnRelays) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA5;
  constexpr size_t kFirstAccessWord = 27'750u;
  constexpr size_t kOwnerCodeWords = kFirstAccessWord + 3u;
  constexpr size_t kTotalTextWords = 60'000u;
  constexpr auto store = cdna5::build_vds(cdna5::kDsStoreB32Vds, {.addr = 0u, .data0 = 1u});
  const uint32_t nop = build_s_nop(0u, kArch);

  std::vector<uint32_t> text_words(kOwnerCodeWords, build_s_mov_b32(0u, 0u, kArch));
  std::ranges::copy(store, text_words.begin() + kFirstAccessWord);
  text_words.back() = build_s_endpgm(kArch);
  text_words.resize(kTotalTextWords, nop);
  std::vector<uint8_t> bytes = make_gfx1250_code_object(
      text_words, "owner_local_prologue_relay", kRdna4Wave64AllVgprsGranulated,
      /*wave32=*/true, /*uses_dynamic_stack=*/true);
  mutate_elf_symbol_by_name(bytes, "owner_local_prologue_relay", [](Elf64_Sym &symbol) {
    symbol.st_size = kOwnerCodeWords * sizeof(uint32_t);
  });

  AmdGpuCodeObject original(bytes.data(), bytes.size());
  ASSERT_TRUE(original.is_valid());
  ASSERT_EQ(original.kernels().size(), 1u);
  const AmdGpuKernelInfo &owner = original.kernels().front();
  ASSERT_EQ(owner.entry_text_offset, 0u);
  ASSERT_EQ(owner.code_size, kOwnerCodeWords * sizeof(uint32_t));

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 8u;
  options.moi_exec_save_sgpr = 80u;
  options.moi_owner_vgpr = 40u;
  options.moi_epoch_vgpr = 41u;
  options.moi_init_owner_epoch = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1u, 0u, 0u, 0u);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.max_patches = 1u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.final_validation_passed);
  const auto access = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore, &ConSanPatchInfo::kind);
  ASSERT_NE(access, result.patches.end());
  EXPECT_EQ(access->owner_descriptor_file_offsets,
            std::vector<uint64_t>{owner.descriptor_file_offset});

  const uint64_t owner_gap_begin = owner.code_size;
  const uint64_t owner_gap_end = kTotalTextWords * sizeof(uint32_t);
  const std::vector<uint64_t> descriptor = {owner.descriptor_file_offset};
  const auto prologue = std::ranges::find_if(result.patches, [&](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue &&
           patch.anchor_offset == owner.entry_text_offset &&
           patch.owner_descriptor_file_offsets == descriptor;
  });
  ASSERT_NE(prologue, result.patches.end()) << testing::PrintToString(result.patches);
  EXPECT_FALSE(compute_sopp_branch_simm16(prologue->trampoline_offset,
                                          owner.entry_text_offset + prologue->original_size));

  const auto entry_island = std::ranges::find_if(result.patches, [&](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiIndirectBranchIsland &&
           patch.anchor_offset == owner.entry_text_offset &&
           patch.trampoline_size == 7u * sizeof(uint32_t) &&
           patch.owner_descriptor_file_offsets == descriptor;
  });
  ASSERT_NE(entry_island, result.patches.end());
  EXPECT_GE(entry_island->trampoline_offset, owner_gap_begin);
  EXPECT_LT(entry_island->trampoline_offset, owner_gap_end);

  const auto return_relay = std::ranges::find_if(result.patches, [&](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiIndirectBranchIsland &&
           patch.anchor_offset == owner.entry_text_offset &&
           patch.trampoline_size == sizeof(uint32_t) &&
           patch.owner_descriptor_file_offsets == descriptor;
  });
  ASSERT_NE(return_relay, result.patches.end());
  EXPECT_GE(return_relay->trampoline_offset, owner_gap_begin);
  EXPECT_LT(return_relay->trampoline_offset, owner_gap_end);

  const uint64_t return_branch_offset =
      prologue->trampoline_offset + prologue->trampoline_size - sizeof(uint32_t);
  const uint64_t return_target = owner.entry_text_offset + prologue->original_size;
  const auto to_relay =
      compute_sopp_branch_simm16(return_branch_offset, return_relay->trampoline_offset);
  const auto from_relay =
      compute_sopp_branch_simm16(return_relay->trampoline_offset, return_target);
  ASSERT_TRUE(to_relay);
  ASSERT_TRUE(from_relay);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  EXPECT_EQ(text_words_at_offset(patched, return_branch_offset, sizeof(uint32_t)),
            std::vector<uint32_t>{build_s_branch(*to_relay, kArch)});
  EXPECT_EQ(text_words_at_offset(patched, return_relay->trampoline_offset, sizeof(uint32_t)),
            std::vector<uint32_t>{build_s_branch(*from_relay, kArch)});
}

TEST(ConSanMoi, RecordReplaySpillBackedLocalEntryIslandsDoNotOverlap) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  constexpr uint32_t kAccessCount = 65u;
  constexpr std::array<uint16_t, 4> kDeadSgprs = {0u, 1u, 4u, 6u};
  const auto make_owner_words = [&] {
    std::vector<uint32_t> words = {
        build_v_mov_b32_e32(/*vdst=*/0u, scalar_positive_inline_u32(0u), kArch),
        build_v_mov_b32_e32(/*vdst=*/1u, scalar_positive_inline_u32(0u), kArch),
    };
    for (uint32_t index = 0u; index < kAccessCount; ++index) {
      words.push_back(0xD8340000u);
      words.push_back(0x00000000u); // ds_store_b32 v0, v0
    }
    for (uint16_t sgpr = 0u; sgpr < 106u; ++sgpr) {
      if (std::ranges::find(kDeadSgprs, sgpr) == kDeadSgprs.end())
        words.push_back(build_s_mov_b32(/*sdst=*/0u, sgpr, kArch));
    }
    words.push_back(build_s_endpgm(kArch));
    return words;
  };
  const std::vector<uint32_t> owner_words = make_owner_words();
  std::vector<uint32_t> kernel_container_words = owner_words;
  kernel_container_words.resize(kernel_container_words.size() + 16u, build_s_nop(0, kArch));
  const std::array<uint32_t, 1> function_words = {build_s_endpgm(kArch)};
  // The unowned suffix keeps the default appended relays out of SOPP reach and
  // a bounded gap between the kernel and function supplies one local-island
  // pool for two dense access groups.
  std::vector<uint32_t> tail_words(33000u, build_s_nop(0, kArch));
  std::vector<uint8_t> bytes = make_rdna4_code_object_with_local_function(
      kernel_container_words, function_words, tail_words, kRdna4Wave64AllVgprsGranulated);
  mutate_elf_symbol_by_name(bytes, "lds_probe", [&](Elf64_Sym &symbol) {
    symbol.st_size = owner_words.size() * sizeof(uint32_t);
  });
  append_kernel_metadata_note(bytes, "lds_probe", /*uses_dynamic_stack=*/false,
                              /*sgpr_count=*/106u, std::nullopt, std::nullopt,
                              /*has_dynamic_lds=*/false);

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 8u;
  options.moi_owner_vgpr = 40u;
  options.moi_epoch_vgpr = 41u;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(kAccessCount, 0u, 0u, 0u);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.max_patches = kAccessCount;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  ASSERT_EQ(result.resolved_moi_transient_sgpr_assignments.size(), 1u);
  EXPECT_TRUE(std::ranges::all_of(result.resolved_moi_transient_sgpr_assignments,
                                  &ConSanMoiTransientSgprAssignment::spill_backed));

  std::vector<const ConSanPatchInfo *> entry_islands;
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.kind == ConSanPatchKind::TrampolineMoiIndirectBranchIsland &&
        patch.trampoline_size != 0u) {
      entry_islands.push_back(&patch);
    }
  }
  ASSERT_GE(entry_islands.size(), 2u);
  std::ranges::sort(entry_islands, [](const ConSanPatchInfo *lhs, const ConSanPatchInfo *rhs) {
    return lhs->trampoline_offset < rhs->trampoline_offset;
  });
  EXPECT_TRUE(std::ranges::any_of(entry_islands, [](const ConSanPatchInfo *patch) {
    return patch->trampoline_size == 8u * sizeof(uint32_t);
  }));
  for (size_t index = 1u; index < entry_islands.size(); ++index) {
    EXPECT_LE(entry_islands[index - 1u]->trampoline_offset +
                  entry_islands[index - 1u]->trampoline_size,
              entry_islands[index]->trampoline_offset);
  }
}

TEST(ConSanMoi, Cdna4RecordReplaySpillsTransientStateAcrossAccessAndBarrier) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  const auto access = build_cdna4_ds_store_b32(/*vaddr=*/2, /*vdata=*/3, /*byte_offset=*/0, kArch);
  const auto barrier = build_cdna4_s_barrier(kArch);
  ASSERT_TRUE(access && barrier);

  std::vector<uint32_t> words(access->begin(), access->end());
  words.push_back(*barrier);
  // Leave only the router state, one dispatch-ID pair, and physical VCC
  // unused. The test gives the kernel a 104-register allocation so s98:s99
  // are physical VCC and must remain unavailable even without guest uses.
  constexpr std::array<uint16_t, 8> kUnreferenced = {
      0u, 1u, 4u, 6u, 20u, 21u, 98u, 99u,
  };
  for (uint16_t sgpr = 0u; sgpr < 102u; ++sgpr) {
    if (std::ranges::find(kUnreferenced, sgpr) == kUnreferenced.end())
      words.push_back(build_s_mov_b32(/*sdst=*/10u, sgpr, kArch));
  }
  words.push_back(build_s_endpgm(kArch));

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 8u;
  options.moi_owner_vgpr = 40u;
  options.moi_epoch_vgpr = 41u;
  options.moi_dispatch_id_sgpr = 20u;
  options.moi_init_owner_epoch = true;
  options.moi_track_barriers = true;
  options.moi_track_atomics = false;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 1);
  options.max_patches = 8u;

  std::vector<uint8_t> bytes = make_cdna4_lds_code_object(words, "cdna4_scalar_spill_barrier");
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 12u);
  });
  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.resolved_moi_transient_sgpr_assignments.size(), 1u)
      << testing::PrintToString(result.warnings);
  const ConSanMoiTransientSgprAssignment &assignment =
      result.resolved_moi_transient_sgpr_assignments.front();
  EXPECT_TRUE(assignment.spill_backed);
  ASSERT_TRUE(assignment.indirect_pc_sgpr);
  ASSERT_TRUE(assignment.indirect_scc_sgpr);
  ASSERT_TRUE(assignment.dispatch_key_sgpr);
  const auto avoids_physical_vcc = [](uint16_t base, uint16_t width) {
    return static_cast<uint32_t>(base) + width <= 98u || base >= 100u;
  };
  EXPECT_TRUE(avoids_physical_vcc(*assignment.indirect_pc_sgpr, 2u));
  EXPECT_TRUE(avoids_physical_vcc(*assignment.indirect_scc_sgpr, 1u));
  EXPECT_TRUE(avoids_physical_vcc(*assignment.dispatch_key_sgpr, 1u));

  for (ConSanPatchKind kind : {ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               ConSanPatchKind::TrampolineMoiBarrierRecord}) {
    const auto patch = std::ranges::find(result.patches, kind, &ConSanPatchInfo::kind);
    ASSERT_NE(patch, result.patches.end()) << testing::PrintToString(result.warnings);
    EXPECT_EQ(patch->required_private_segment_size, 0u);
    AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(patched.is_valid());
    expect_lane_backed_scalar_spill(result, patched, *patch, assignment.exec_save_sgpr, kArch);
  }
  EXPECT_TRUE(result.final_validation_passed);

  constexpr uint32_t kDenseBarrierCount = 80u;
  std::vector<uint32_t> dense_words;
  for (uint32_t index = 0u; index < kDenseBarrierCount; ++index) {
    dense_words.insert(dense_words.end(), access->begin(), access->end());
    dense_words.push_back(*barrier);
  }
  for (uint16_t sgpr = 0u; sgpr < 102u; ++sgpr) {
    if (std::ranges::find(kUnreferenced, sgpr) == kUnreferenced.end())
      dense_words.push_back(build_s_mov_b32(/*sdst=*/10u, sgpr, kArch));
  }
  dense_words.push_back(build_s_endpgm(kArch));
  std::vector<uint8_t> dense_bytes =
      make_cdna4_lds_code_object(dense_words, "cdna4_dense_scalar_spill_barrier");
  mutate_first_kernel_descriptor(dense_bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 12u);
  });
  ConSanOptions dense_options = options;
  dense_options.moi_report_buffer_size =
      consan_moi_report_buffer_min_bytes(kDenseBarrierCount, 0, 0, 0, kDenseBarrierCount);
  dense_options.max_patches = 96u;

  const ConSanResult dense_result = try_patch_consan(dense_bytes, dense_options);

  ASSERT_TRUE(consan_patch_succeeded(dense_result)) << testing::PrintToString(dense_result.errors);
  ASSERT_TRUE(dense_result.modified) << testing::PrintToString(dense_result.warnings);
  EXPECT_TRUE(dense_result.final_validation_passed);
  ASSERT_EQ(dense_result.resolved_moi_transient_sgpr_assignments.size(), 1u);
  const ConSanMoiTransientSgprAssignment &dense_assignment =
      dense_result.resolved_moi_transient_sgpr_assignments.front();
  EXPECT_TRUE(dense_assignment.spill_backed);
  for (ConSanPatchKind kind : {ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               ConSanPatchKind::TrampolineMoiBarrierRecord}) {
    const auto patch = std::ranges::find(dense_result.patches, kind, &ConSanPatchInfo::kind);
    ASSERT_NE(patch, dense_result.patches.end()) << testing::PrintToString(dense_result.warnings);
    // This fixture supplies persistent owner/epoch VGPRs explicitly. The
    // transient scalar window remains lane-backed and therefore needs no
    // private segment or automatic private epoch.
    EXPECT_FALSE(patch->persistent_private_state_end);
    EXPECT_EQ(patch->required_private_segment_size, 0u);
    AmdGpuCodeObject patched(dense_result.elf_bytes.data(), dense_result.elf_bytes.size());
    ASSERT_TRUE(patched.is_valid());
    expect_lane_backed_scalar_spill(dense_result, patched, *patch, dense_assignment.exec_save_sgpr,
                                    kArch);
  }
  ASSERT_TRUE(dense_assignment.indirect_pc_sgpr);
  const auto indirect_barrier = std::ranges::find_if(dense_result.patches, [&](const auto &patch) {
    if (patch.kind != ConSanPatchKind::TrampolineMoiBarrierRecord)
      return false;
    return std::ranges::any_of(dense_result.patches, [&](const auto &island) {
      return island.kind == ConSanPatchKind::TrampolineMoiIndirectBranchIsland &&
             island.anchor_offset == patch.anchor_offset;
    });
  });
  ASSERT_NE(indirect_barrier, dense_result.patches.end());
  const auto barrier_island =
      std::ranges::find_if(dense_result.patches, [&](const ConSanPatchInfo &patch) {
        return patch.kind == ConSanPatchKind::TrampolineMoiIndirectBranchIsland &&
               patch.anchor_offset == indirect_barrier->anchor_offset;
      });
  ASSERT_NE(barrier_island, dense_result.patches.end());
  AmdGpuCodeObject dense_patched(dense_result.elf_bytes.data(), dense_result.elf_bytes.size());
  ASSERT_TRUE(dense_patched.is_valid());
  const std::vector<uint32_t> island_words = text_words_at_offset(
      dense_patched, barrier_island->trampoline_offset, barrier_island->trampoline_size);
  ASSERT_GE(island_words.size(), 2u);
  // The local relay runs before the scalar spill body. It must use the
  // descriptor-local dead PC pair, not the unrelated root fallback.
  EXPECT_EQ(island_words[1], build_s_getpc_b64(*dense_assignment.indirect_pc_sgpr, kArch));
}

TEST(ConSanMoi, RecordReplayUsesBranchOnlySpillWithoutDeadRouterScalars) {
  const auto run = [](std::span<const uint16_t> dead) {
    std::vector<uint32_t> words;
    for (uint32_t index = 0; index < 9u; ++index) {
      words.push_back(0xD8340000u);
      words.push_back(0x00000000u); // ds_store_b32 v0, v0
    }
    for (uint16_t sgpr = 0; sgpr < 106u; ++sgpr) {
      if (std::ranges::find(dead, sgpr) == dead.end())
        words.push_back(build_s_mov_b32(/*M0=*/125u, sgpr, ROCJITSU_CODE_ARCH_CDNA5));
    }
    words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5));

    ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
    options.scratch_vgpr = 8;
    options.moi_owner_vgpr = 40;
    options.moi_epoch_vgpr = 41;
    options.moi_report_buffer_address = 0x123456780000ull;
    options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(9, 0, 0, 0);
    options.moi_track_barriers = false;
    options.moi_track_atomics = false;
    options.max_patches = 9u;
    return try_patch_consan(make_gfx1250_code_object(words), options);
  };

  const auto expect_branch_only = [](const ConSanResult &result) {
    EXPECT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
    EXPECT_TRUE(result.modified) << testing::PrintToString(result.warnings);
    ASSERT_EQ(result.resolved_moi_transient_sgpr_assignments.size(), 1u);
    const ConSanMoiTransientSgprAssignment &assignment =
        result.resolved_moi_transient_sgpr_assignments.front();
    EXPECT_TRUE(assignment.spill_backed);
    EXPECT_TRUE(assignment.branch_only_scalar_spill);
    EXPECT_TRUE(assignment.indirect_pc_sgpr);
    EXPECT_TRUE(assignment.indirect_scc_sgpr);
    EXPECT_FALSE(assignment.dispatch_key_sgpr);
    EXPECT_FALSE(assignment.call_return_sgpr);
    EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                                 &ConSanPatchInfo::kind),
              9u);
    EXPECT_TRUE(std::ranges::all_of(result.patches, [](const ConSanPatchInfo &patch) {
      return patch.kind != ConSanPatchKind::TrampolineMoiAccessRecordStore ||
             patch.branch_only_continuation;
    }));
    EXPECT_TRUE(result.final_validation_passed);
  };

  // No aligned pair is dead, so the owner-local scalar-spill window supplies
  // the optional borrowed entry PC/SCC tuple. These adjacent accesses cannot
  // expand their anchors without overlapping one another, and therefore keep
  // the pure branch-only continuation shape.
  const std::array<uint16_t, 5> no_pc_pair = {0u, 2u, 4u, 6u, 8u};
  const ConSanResult no_pc = run(no_pc_pair);
  expect_branch_only(no_pc);

  // One dead pair and one scalar cannot retain both guest SCC and the dense
  // dispatch key across an indirect jump. The fixed-stack tuple remains
  // available, while these adjacent sites use the same pure branch-only path.
  const std::array<uint16_t, 3> no_key_scalar = {0u, 1u, 4u};
  const ConSanResult no_key = run(no_key_scalar);
  expect_branch_only(no_key);
}

TEST(ConSanMoi, Rdna4SpillBackedTwoSiteDenseDispatcherIncludesSharedHostArm) {
  constexpr uint32_t kAccessCount = 2u;
  const std::array<uint16_t, 4> dead = {0u, 1u, 4u, 6u};
  const uint32_t filler = build_s_mov_b32(/*sdst=*/20u, /*ssrc0=*/20u, ROCJITSU_CODE_ARCH_RDNA4);
  std::vector<uint32_t> words(8u, filler);
  for (uint32_t index = 0; index < kAccessCount; ++index) {
    words.push_back(0xD8340000u | index * sizeof(uint32_t));
    words.push_back(0x00000000u); // ds_store_b32 v0, v0 offset:index*4
  }
  for (uint16_t sgpr = 0; sgpr < 106u; ++sgpr) {
    if (std::ranges::find(dead, sgpr) == dead.end())
      words.push_back(build_s_mov_b32(/*sdst=*/20u, sgpr, ROCJITSU_CODE_ARCH_RDNA4));
  }
  // Keep the two access anchors outside SOPP reach of the appended bodies,
  // with no local NOP island, so they share the spill-backed dense host whose
  // dispatcher exposed the missing terminator reservation.
  words.resize(33'000u, filler);
  words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4));

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(kAccessCount, 0, 0, 0);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.max_patches = kAccessCount;

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(words, "rdna4_spill_dense_two");
  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.resolved_moi_transient_sgpr_assignments.size(), 1u);
  const ConSanMoiTransientSgprAssignment &assignment =
      result.resolved_moi_transient_sgpr_assignments.front();
  EXPECT_TRUE(assignment.spill_backed);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               &ConSanPatchInfo::kind),
            kAccessCount);
  const auto dense_host = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiIndirectBranchIsland &&
           patch.original_size != 0u;
  });
  ASSERT_NE(dense_host, result.patches.end());
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> entry_island = text_words_at_offset(
      patched, dense_host->anchor_offset + sizeof(uint32_t), 8u * sizeof(uint32_t));
  ASSERT_EQ(entry_island.size(), 8u);
  EXPECT_EQ(entry_island.back(), build_s_nop(0u, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, Gfx1250SpillBackedSingleSiteDenseDispatcherReservesSpillArm) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA5;
  const std::array<uint16_t, 4> dead = {0u, 1u, 4u, 6u};
  const uint32_t filler = build_s_mov_b32(/*sdst=*/20u, /*ssrc0=*/20u, kArch);
  std::vector<uint32_t> words(8u, filler);
  words.push_back(0xD8340000u);
  words.push_back(0x00000000u); // ds_store_b32 v0, v0
  for (uint16_t sgpr = 0u; sgpr < 106u; ++sgpr) {
    if (std::ranges::find(dead, sgpr) == dead.end())
      words.push_back(build_s_mov_b32(/*sdst=*/20u, sgpr, kArch));
  }
  // A single remote site is the tight reservation case: its dense dispatcher
  // has fixed routing overhead and one extra spill-backed entry arm, without a
  // second site's slot to mask an off-by-one relay reservation.
  words.resize(33'000u, filler);
  words.push_back(build_s_endpgm(kArch));

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 8u;
  options.moi_owner_vgpr = 40u;
  options.moi_epoch_vgpr = 41u;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1u, 0u, 0u, 0u);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.max_patches = 1u;

  const ConSanResult result =
      try_patch_consan(make_gfx1250_code_object(words, "gfx1250_spill_dense_single"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.resolved_moi_transient_sgpr_assignments.size(), 1u);
  EXPECT_TRUE(result.resolved_moi_transient_sgpr_assignments.front().spill_backed);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               &ConSanPatchInfo::kind),
            1u);
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, Rdna4ModerateSpillBackedDispatcherKeepsCompactRelaySpacing) {
  constexpr uint32_t kAccessCount = 3u;
  const std::array<uint16_t, 4> dead = {0u, 1u, 4u, 6u};
  const uint32_t filler = build_s_mov_b32(/*sdst=*/20u, /*ssrc0=*/20u, ROCJITSU_CODE_ARCH_RDNA4);
  std::vector<uint32_t> words(8u, filler);
  for (uint32_t index = 0; index < kAccessCount; ++index) {
    words.push_back(0xD8340000u | index * sizeof(uint32_t));
    words.push_back(0x00000000u); // ds_store_b32 v0, v0 offset:index*4
  }
  for (uint16_t sgpr = 0; sgpr < 106u; ++sgpr) {
    if (std::ranges::find(dead, sgpr) == dead.end())
      words.push_back(build_s_mov_b32(/*sdst=*/20u, sgpr, ROCJITSU_CODE_ARCH_RDNA4));
  }
  words.resize(33'000u, filler);
  words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4));

  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(words, "rdna4_spill_dense_moderate");
  AmdGpuCodeObject original(bytes.data(), bytes.size());
  ASSERT_TRUE(original.is_valid());
  ASSERT_EQ(original.text_sections().size(), 1u);
  const uint64_t original_text_size = original.text_sections().front()->size();
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(kAccessCount, 0, 0, 0);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.max_patches = kAccessCount;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.resolved_moi_transient_sgpr_assignments.size(), 1u);
  EXPECT_TRUE(result.resolved_moi_transient_sgpr_assignments.front().spill_backed);
  const auto first_access = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore, &ConSanPatchInfo::kind);
  ASSERT_NE(first_access, result.patches.end());
  constexpr uint64_t kCompactSpillBackedRelayWords = 11u;
  EXPECT_EQ(first_access->trampoline_offset,
            original_text_size + kAccessCount * kCompactSpillBackedRelayWords * sizeof(uint32_t));
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, RecordReplaySpillBackedDenseHostPreservesBarrierAndCapturedCallKey) {
  constexpr uint32_t kAccessCount = 9u;
  const std::array<uint16_t, 6> dead = {0u, 1u, 4u, 6u, 8u, 9u};
  std::vector<uint32_t> words(
      9u, build_s_mov_b32(/*sdst=*/20u, /*ssrc0=*/20u, ROCJITSU_CODE_ARCH_CDNA5));
  for (uint32_t index = 0; index < kAccessCount; ++index) {
    words.push_back(0xD8340000u | index * sizeof(uint32_t));
    words.push_back(0x00000000u); // ds_store_b32 v0, v0 offset:index*4
    if (index == 1u)
      words.push_back(0xBF940000u); // s_barrier_wait -1
  }
  for (uint16_t sgpr = 0; sgpr < 106u; ++sgpr) {
    if (std::ranges::find(dead, sgpr) == dead.end())
      words.push_back(build_s_mov_b32(/*sdst=*/20u, sgpr, ROCJITSU_CODE_ARCH_CDNA5));
  }
  words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5));
  const std::vector<uint8_t> bytes = make_gfx1250_code_object(words);

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_init_owner_epoch = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size =
      consan_moi_report_buffer_min_bytes(kAccessCount, 0, 0, 0, kAccessCount);
  options.moi_track_barriers = true;
  options.moi_track_atomics = false;
  options.max_patches = 32u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.resolved_moi_transient_sgpr_assignments.size(), 1u);
  const ConSanMoiTransientSgprAssignment &assignment =
      result.resolved_moi_transient_sgpr_assignments.front();
  EXPECT_TRUE(assignment.spill_backed);
  ASSERT_TRUE(assignment.indirect_pc_sgpr);
  ASSERT_TRUE(assignment.dispatch_key_sgpr);
  ASSERT_TRUE(assignment.call_return_sgpr);
  ASSERT_EQ(assignment.call_return_sgpr, assignment.indirect_pc_sgpr);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               &ConSanPatchInfo::kind),
            kAccessCount);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord,
                               &ConSanPatchInfo::kind),
            1u);
  EXPECT_TRUE(std::ranges::any_of(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiIndirectBranchIsland &&
           patch.original_size == 9u * sizeof(uint32_t);
  }));
  const auto dispatcher = std::ranges::find_if(result.patches, [&](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiIndirectBranchIsland &&
           patch.original_size == 0u &&
           patch.anchor_offset == result.moi_candidates.front().text_offset &&
           patch.trampoline_size != 0u;
  });
  ASSERT_NE(dispatcher, result.patches.end());
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> dispatcher_words =
      text_words_at_offset(patched, dispatcher->trampoline_offset, dispatcher->trampoline_size);
  const auto captured_key_compare = build_s_cmp_eq_u32(
      *assignment.dispatch_key_sgpr, *assignment.indirect_pc_sgpr, ROCJITSU_CODE_ARCH_CDNA5);
  const auto tautological_pc_compare = build_s_cmp_eq_u32(
      *assignment.indirect_pc_sgpr, *assignment.indirect_pc_sgpr, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_TRUE(captured_key_compare);
  ASSERT_TRUE(tautological_pc_compare);
  EXPECT_GT(std::ranges::count(dispatcher_words, *captured_key_compare), 0u);
  EXPECT_EQ(std::ranges::count(dispatcher_words, *tautological_pc_compare), 0u);
  EXPECT_TRUE(std::ranges::all_of(result.site_dispositions, [](const auto &site) {
    return site.disposition != ConSanSiteDisposition::Supported ||
           site.lowering_outcome == ConSanSiteLoweringOutcome::Patched;
  }));
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, Gfx1250RecordReplaySpillBackedDenseBarrierUsesCapturedEntryKey) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA5;
  constexpr uint32_t kSiteCount = 17u;
  const std::array<uint16_t, 6> dead = {0u, 1u, 4u, 6u, 8u, 9u};
  std::vector<uint32_t> words(9u, build_s_mov_b32(/*sdst=*/20u, /*ssrc0=*/20u, kArch));
  for (uint32_t index = 0u; index < kSiteCount; ++index) {
    words.push_back(0xBE804EC1u); // s_barrier_signal -1
    words.push_back(0xBF94FFFFu); // s_barrier_wait -1
  }
  for (uint16_t sgpr = 0u; sgpr < 106u; ++sgpr) {
    if (std::ranges::find(dead, sgpr) == dead.end())
      words.push_back(build_s_mov_b32(/*sdst=*/20u, sgpr, kArch));
  }
  words.resize(33'000u, build_s_mov_b32(/*sdst=*/20u, /*ssrc0=*/20u, kArch));
  words.push_back(build_s_endpgm(kArch));

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 8u;
  options.moi_owner_vgpr = 40u;
  options.moi_epoch_vgpr = 41u;
  options.moi_init_owner_epoch = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size =
      consan_moi_report_buffer_min_bytes(0u, 0u, 0u, 0u, 2u * kSiteCount);
  options.moi_track_barriers = true;
  options.moi_track_atomics = false;
  options.max_patches = 4u * kSiteCount;

  const ConSanResult result =
      try_patch_consan(make_gfx1250_code_object(words, "gfx1250_spill_dense_barriers"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.resolved_moi_transient_sgpr_assignments.size(), 1u);
  const ConSanMoiTransientSgprAssignment &assignment =
      result.resolved_moi_transient_sgpr_assignments.front();
  ASSERT_TRUE(assignment.spill_backed);
  ASSERT_TRUE(assignment.indirect_pc_sgpr);
  ASSERT_TRUE(assignment.dispatch_key_sgpr);
  ASSERT_TRUE(assignment.call_return_sgpr);
  ASSERT_EQ(assignment.call_return_sgpr, assignment.indirect_pc_sgpr);
  const auto first_barrier = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord, &ConSanPatchInfo::kind);
  ASSERT_NE(first_barrier, result.patches.end()) << testing::PrintToString(result.warnings);
  const auto dispatcher = std::ranges::find_if(result.patches, [&](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiIndirectBranchIsland &&
           patch.original_size == 0u && patch.anchor_offset == first_barrier->anchor_offset &&
           patch.trampoline_size != 0u;
  });
  ASSERT_NE(dispatcher, result.patches.end()) << testing::PrintToString(result.warnings);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> dispatcher_words =
      text_words_at_offset(patched, dispatcher->trampoline_offset, dispatcher->trampoline_size);
  const auto captured_key_compare =
      build_s_cmp_eq_u32(*assignment.dispatch_key_sgpr, /*literal=*/255u, kArch);
  const auto tautological_pc_compare =
      build_s_cmp_eq_u32(*assignment.indirect_pc_sgpr, *assignment.indirect_pc_sgpr, kArch);
  ASSERT_TRUE(captured_key_compare);
  ASSERT_TRUE(tautological_pc_compare);
  EXPECT_GT(std::ranges::count(dispatcher_words, *captured_key_compare), 0u);
  EXPECT_EQ(std::ranges::count(dispatcher_words, *tautological_pc_compare), 0u);
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, Gfx1250DenseBarrierHostAvoidsImplicitHighVgprBank) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA5;
  constexpr uint32_t kSiteCount = 17u;
  constexpr uint32_t kFillerWords = 20u;
  constexpr std::array<uint16_t, 6> kDead = {0u, 1u, 4u, 6u, 8u, 9u};
  const uint32_t filler = build_s_mov_b32(/*sdst=*/20u, /*ssrc0=*/20u, kArch);
  std::vector<uint32_t> words(kFillerWords, filler);
  for (uint32_t index = 0u; index < kSiteCount; ++index) {
    words.push_back(0xBE804EC1u); // s_barrier_signal -1
    words.push_back(0xBF94FFFFu); // s_barrier_wait -1
  }
  // Give the midpoint-biased host search an otherwise ideal relocation range
  // that begins with a high-bank transition.  The gfx1250 MXF4 workload had
  // this exact shape followed by WMMA instructions.  Moving the transition
  // made either the appended WMMA operands or the original continuation use
  // v[0+] instead of v[256+].
  words.push_back(*build_gfx1250_s_set_vgpr_msb(/*mode=*/4u, kArch));
  words.insert(words.end(), kFillerWords, filler);
  words.push_back(*build_gfx1250_s_set_vgpr_msb(/*previous_mode=*/4u << 8u, kArch));
  words.insert(words.end(), kFillerWords, filler);
  for (uint16_t sgpr = 0u; sgpr < 106u; ++sgpr) {
    if (std::ranges::find(kDead, sgpr) == kDead.end())
      words.push_back(build_s_mov_b32(/*sdst=*/20u, sgpr, kArch));
  }
  words.resize(33'000u, filler);
  words.push_back(build_s_endpgm(kArch));

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 8u;
  options.moi_owner_vgpr = 40u;
  options.moi_epoch_vgpr = 41u;
  options.moi_init_owner_epoch = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size =
      consan_moi_report_buffer_min_bytes(0u, 0u, 0u, 0u, 2u * kSiteCount);
  options.moi_track_barriers = true;
  options.moi_track_atomics = false;
  options.max_patches = 4u * kSiteCount;

  const std::vector<uint8_t> bytes =
      make_gfx1250_code_object(words, "gfx1250_dense_barrier_high_vgpr_bank");
  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord,
                               &ConSanPatchInfo::kind),
            2u * kSiteCount);
  const auto host = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiIndirectBranchIsland &&
           patch.original_size != 0u;
  });
  ASSERT_NE(host, result.patches.end()) << testing::PrintToString(result.warnings);
  AmdGpuCodeObject original(bytes.data(), bytes.size());
  ASSERT_TRUE(original.is_valid());
  ASSERT_EQ(original.text_sections().size(), 1u);
  const Section *text = original.text_sections().front();
  const std::optional<uint16_t> host_mode = consan_gfx1250_vgpr_msb_mode_at(
      bytes, text->sectionOffset(), /*container_entry_text_offset=*/0u,
      text->sectionOffset() + host->anchor_offset);
  ASSERT_TRUE(host_mode);
  EXPECT_EQ(*host_mode, 0u);
  const std::optional<uint16_t> continuation_mode = consan_gfx1250_vgpr_msb_mode_at(
      bytes, text->sectionOffset(), /*container_entry_text_offset=*/0u,
      text->sectionOffset() + host->anchor_offset + host->original_size);
  ASSERT_TRUE(continuation_mode);
  EXPECT_EQ(*continuation_mode, 0u);
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, RecordReplaySpillBackedDenseHostAvoidsTransientSgprLiveRange) {
  constexpr uint32_t kAccessCount = 9u;
  constexpr std::array<uint16_t, 4> kTransientWindow = {0u, 1u, 4u, 6u};
  std::vector<uint32_t> words;
  for (uint32_t index = 0; index < kAccessCount; ++index) {
    words.push_back(0xD8340000u | index * sizeof(uint32_t));
    words.push_back(0x00000000u); // ds_store_b32 v0, v0 offset:index*4
  }

  // The automatic spill router selects these four registers because the
  // definitions keep them dead at every access anchor. They are nevertheless
  // live across the following NOP run, which is an attractive relocatable
  // host unless the post-planning liveness state includes arbitrary hosts.
  for (uint16_t sgpr : kTransientWindow) {
    words.push_back(build_s_mov_b32(sgpr, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_CDNA5));
  }
  words.resize(words.size() + 32u, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5));
  for (uint16_t sgpr : kTransientWindow)
    words.push_back(build_s_mov_b32(/*sdst=*/20u, sgpr, ROCJITSU_CODE_ARCH_CDNA5));
  const uint64_t first_safe_host_offset = words.size() * sizeof(uint32_t);

  for (uint16_t sgpr = 0; sgpr < 106u; ++sgpr) {
    if (std::ranges::find(kTransientWindow, sgpr) == kTransientWindow.end())
      words.push_back(build_s_mov_b32(/*sdst=*/20u, sgpr, ROCJITSU_CODE_ARCH_CDNA5));
  }
  words.resize(words.size() + 16u, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5));
  words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5));

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(kAccessCount, 0, 0, 0);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.max_patches = kAccessCount;

  const ConSanResult result = try_patch_consan(make_gfx1250_code_object(words), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.resolved_moi_transient_sgpr_assignments.size(), 1u);
  const ConSanMoiTransientSgprAssignment &assignment =
      result.resolved_moi_transient_sgpr_assignments.front();
  EXPECT_TRUE(assignment.spill_backed);
  EXPECT_EQ(assignment.indirect_pc_sgpr, kTransientWindow[0]);
  EXPECT_EQ(assignment.indirect_scc_sgpr, kTransientWindow[2]);
  EXPECT_EQ(assignment.dispatch_key_sgpr, kTransientWindow[3]);
  bool saw_relocated_host = false;
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.kind != ConSanPatchKind::TrampolineMoiIndirectBranchIsland ||
        patch.original_size == 0u) {
      continue;
    }
    saw_relocated_host = true;
    EXPECT_GE(patch.anchor_offset, first_safe_host_offset);
  }
  EXPECT_TRUE(saw_relocated_host);
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, RecordReplayReservedRelaySpaceComposesWithBarrierEpochs) {
  std::vector<uint32_t> text_words;
  for (uint32_t i = 0; i < 9u; ++i) {
    text_words.push_back(0xD8340000u); // ds_store_b32 v0, v0
    text_words.push_back(0x00000000u);
  }
  text_words.push_back(0xBF940000u); // s_barrier_wait -1
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4));
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 80;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_init_owner_epoch = true;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(9, 0, 0, 0, 9);
  options.moi_track_barriers = true;
  options.moi_track_atomics = false;
  options.max_patches = 32;

  const auto result = try_patch_consan(bytes, options);

  SCOPED_TRACE(testing::PrintToString(result.warnings));
  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               &ConSanPatchInfo::kind),
            9);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord,
                               &ConSanPatchInfo::kind),
            1);
  // The epoch body may fit directly in its reserved local relay space. Keep
  // this test about the composition contract instead of requiring the larger
  // indirect-entry layout.
  EXPECT_GE(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiIndirectBranchIsland,
                               &ConSanPatchInfo::kind),
            9);

  const auto barrier_patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiBarrierRecord &&
           patch.anchor_offset == 9u * 2u * sizeof(uint32_t);
  });
  ASSERT_NE(barrier_patch, result.patches.end());
  ASSERT_TRUE(barrier_patch->relocated_guest_instruction_offset);
  EXPECT_GE(*barrier_patch->relocated_guest_instruction_offset, barrier_patch->trampoline_offset);
  EXPECT_LT(*barrier_patch->relocated_guest_instruction_offset,
            barrier_patch->trampoline_offset + barrier_patch->trampoline_size);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  uint64_t entry_target = barrier_patch->trampoline_offset;
  const auto direct_branch = compute_sopp_branch_simm16(barrier_patch->anchor_offset, entry_target);
  ASSERT_TRUE(direct_branch);
  const uint32_t encoded_entry =
      text_words_at_offset(patched, barrier_patch->anchor_offset, sizeof(uint32_t)).front();
  if (encoded_entry != build_s_branch(*direct_branch, ROCJITSU_CODE_ARCH_RDNA4)) {
    const auto island = std::ranges::find_if(result.patches, [&](const ConSanPatchInfo &patch) {
      return patch.kind == ConSanPatchKind::TrampolineMoiIndirectBranchIsland &&
             patch.anchor_offset == barrier_patch->anchor_offset;
    });
    ASSERT_NE(island, result.patches.end());
    entry_target = island->trampoline_offset;
  }
  const auto branch = compute_sopp_branch_simm16(barrier_patch->anchor_offset, entry_target);
  ASSERT_TRUE(branch);
  EXPECT_EQ(encoded_entry, build_s_branch(*branch, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, RecordReplayAllowsScratchBetweenDisjointGfx1250StoreTuples) {
  constexpr auto store =
      cdna5::build_vds(cdna5::kDsStore2addrB64Vds,
                       {.offset0 = 0u, .offset1 = 1u, .addr = 0u, .data0 = 29u, .data1 = 38u});
  const std::array<uint32_t, 3> text_words = {
      store[0],
      store[1],
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5),
  };
  const std::vector<uint8_t> bytes =
      make_gfx1250_code_object(text_words, "disjoint_store_tuple_scratch");
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 32;
  options.moi_exec_save_sgpr = 60;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(2, 0, 0, 0);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.moi_candidates.size(), 1u);
  ASSERT_TRUE(result.moi_candidates.front().data_vgpr);
  EXPECT_EQ(*result.moi_candidates.front().data_vgpr, 29u);
  ASSERT_TRUE(result.moi_candidates.front().second_data_vgpr);
  EXPECT_EQ(*result.moi_candidates.front().second_data_vgpr, 38u);
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore ||
           patch.kind == ConSanPatchKind::InlineMoiAccessRecordStore;
  });
  ASSERT_NE(access, result.patches.end());
  ASSERT_TRUE(access->scratch_vgpr);
  EXPECT_EQ(*access->scratch_vgpr, 32u);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> body =
      text_words_at_offset(patched, access->trampoline_offset, access->trampoline_size);
  const auto first = cdna5::build_vds(cdna5::kDsStoreB64Vds, {.addr = 0u, .data0 = 29u});
  const auto second =
      cdna5::build_vds(cdna5::kDsStoreB64Vds, {.offset0 = 8u, .addr = 0u, .data0 = 38u});
  EXPECT_TRUE(contains_subsequence(body, first));
  EXPECT_TRUE(contains_subsequence(body, second));
  EXPECT_FALSE(contains_subsequence(body, store));
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, RecordReplaySplitsLargeGfx1250TwoAddressOffsetsWithPlannedScratch) {
  constexpr auto store =
      cdna5::build_vds(cdna5::kDsStore2addrStride64B64Vds,
                       {.offset0 = 1u, .offset1 = 255u, .addr = 0u, .data0 = 2u, .data1 = 4u});
  const std::array<uint32_t, 3> text_words = {
      store[0],
      store[1],
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5),
  };
  const std::vector<uint8_t> bytes =
      make_gfx1250_code_object(text_words, "large_two_address_record_replay");
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 32;
  options.moi_exec_save_sgpr = 60;
  options.moi_owner_vgpr = 50;
  options.moi_epoch_vgpr = 51;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(2, 0, 0, 0);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore;
  });
  ASSERT_NE(access, result.patches.end());
  ASSERT_TRUE(access->scratch_vgpr);
  const auto plan = std::ranges::find_if(result.resource_plans, [](const auto &resource_plan) {
    return resource_plan.site_kind == ConSanResourceSiteKind::Access;
  });
  ASSERT_NE(plan, result.resource_plans.end());
  ASSERT_GE(plan->scratch_vgpr_count, 1u);
  const uint16_t adjusted_address =
      static_cast<uint16_t>(*access->scratch_vgpr + plan->scratch_vgpr_count - 1u);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> body =
      text_words_at_offset(patched, access->trampoline_offset, access->trampoline_size);
  const auto first =
      cdna5::build_vds(cdna5::kDsStoreB64Vds, {.offset1 = 2u, .addr = 0u, .data0 = 2u});
  const auto second = cdna5::build_vds(
      cdna5::kDsStoreB64Vds, {.addr = static_cast<uint8_t>(adjusted_address), .data0 = 4u});
  EXPECT_TRUE(contains_subsequence(body, first));
  EXPECT_TRUE(contains_subsequence(body, second));
  EXPECT_FALSE(contains_subsequence(body, store));
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, FirstLightProbeWritesOneLikelyGroupFlatAccessRecord) {
  const std::array<uint32_t, 2> kernel_words = {
      pack_sopk(/*s_call_b64=*/0x14, /*sdst=*/30, /*simm16=*/1),
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint32_t> function_words = make_padded_moi_flat_first_light_function_words();
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words);
  ConSanOptions options = moi_options();
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 14;
  options.moi_epoch_vgpr = 15;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.moi_candidates.size(), 1u);
  EXPECT_EQ(result.moi_candidates.front().source, ConSanMoiCandidateSource::FlatGroup);
  EXPECT_EQ(result.moi_candidates.front().kind, ConSanLdsAccessKind::Read);
  EXPECT_EQ(result.moi_candidates.front().mnemonic, "flat_load_b32");
  EXPECT_EQ(result.moi_candidates.front().text_offset, 28u);
  ASSERT_EQ(non_entry_prologue_patch_count(result), 1u);
  EXPECT_EQ(only_non_entry_prologue_patch(result).kind,
            ConSanPatchKind::InlineMoiAccessRecordStore);
  EXPECT_EQ(only_non_entry_prologue_patch(result).anchor_offset, 28u);
  EXPECT_EQ(only_non_entry_prologue_patch(result).trampoline_offset, 40u);
  expect_bounded_static_record_replay_probe_size(
      only_non_entry_prologue_patch(result).original_size);

  const std::vector<uint32_t> rewritten_words = patched_words_at_file_offset(
      result, 0x11c, only_non_entry_prologue_patch(result).original_size);
  ASSERT_GE(rewritten_words.size(), 3u);
  EXPECT_EQ(rewritten_words[rewritten_words.size() - 3u], 0xEC05007Cu);
  EXPECT_EQ(rewritten_words[rewritten_words.size() - 2u], 0x00000002u);
  EXPECT_EQ(rewritten_words.back(), 0x00000000u);
  EXPECT_EQ(std::count(rewritten_words.begin(), rewritten_words.end(), 0xBFC60000u), 0u);
}

TEST(ConSanMoi, RecordReplayUsesBranchIslandForFunctionOwnedAccess) {
  const std::array<uint32_t, 2> kernel_words = {
      pack_sopk(/*s_call_b64=*/0x14, /*sdst=*/30, /*simm16=*/1),
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::array<uint32_t, 9> function_words = {
      0xBE8001EBu, // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u,
      0x00000000u, // v_mov_b32_e64 v0, s0
      0xD5810001u,
      0x00000001u, // v_mov_b32_e64 v1, s1
      0xEC05007Cu,
      0x00000002u,
      0x00000000u, // flat_load_b32 v2, v[0:1]
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  std::vector<uint32_t> tail_words(40000u, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words, tail_words);
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 80;
  options.moi_owner_vgpr = 14;
  options.moi_epoch_vgpr = 15;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               &ConSanPatchInfo::kind),
            1);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiIndirectBranchIsland,
                               &ConSanPatchInfo::kind),
            1);
}

TEST(ConSanMoi, Rdna4DenseFunctionAccessesUseRelocatableHost) {
  constexpr uint32_t kAccessCount = 9u;
  const std::array<uint32_t, 2> kernel_words = {
      pack_sopk(/*s_call_b64=*/0x14, /*sdst=*/30, /*simm16=*/1),
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const uint32_t filler = build_s_mov_b32(/*sdst=*/20u, /*ssrc0=*/20u, ROCJITSU_CODE_ARCH_RDNA4);
  std::vector<uint32_t> function_words(8u, filler);
  for (uint32_t index = 0; index < kAccessCount; ++index) {
    function_words.push_back(0xD8340000u | index * sizeof(uint32_t));
    function_words.push_back(0x00000000u); // ds_store_b32 v0, v0 offset:index*4
  }
  // Keep the function-owned sites outside SOPP reach of their appended
  // bodies and provide no NOP island. The relocatable host must be selected
  // from the function itself rather than from its owning kernel.
  function_words.resize(33'000u, filler);
  function_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4));

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 80;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(kAccessCount, 0, 0, 0);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.max_patches = kAccessCount;

  const ConSanResult result = try_patch_consan(
      make_rdna4_code_object_with_local_function(kernel_words, function_words), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               &ConSanPatchInfo::kind),
            kAccessCount);
  EXPECT_EQ(std::ranges::count_if(result.site_dispositions,
                                  [](const auto &site) {
                                    return site.site_kind == ConSanResourceSiteKind::Access &&
                                           !site.in_kernel &&
                                           site.lowering_outcome ==
                                               ConSanSiteLoweringOutcome::Patched;
                                  }),
            kAccessCount);
}

TEST(ConSanMoi, FirstLightProbeRejectsScratchVgprsOverlappingFlatAddressPair) {
  const std::array<uint32_t, 2> kernel_words = {
      pack_sopk(/*s_call_b64=*/0x14, /*sdst=*/30, /*simm16=*/1),
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint32_t> function_words = make_padded_moi_flat_first_light_function_words();
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words);
  ConSanOptions options = moi_options();
  options.scratch_vgpr = 1;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  EXPECT_TRUE(result.errors.empty());
  EXPECT_FALSE(result.modified);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().reason, ConSanRegisterPlanReason::ForbiddenOverlap);
}

TEST(ConSanMoi, BarrierRecordPatchTrampolinesBarrierAndWritesRecord) {
  constexpr uint32_t kBarrierWait = 0xBF940000u;
  const std::array<uint32_t, 2> text_words = {
      kBarrierWait,
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options = moi_options();
  options.moi_track_barriers = true;
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 30;
  options.moi_owner_vgpr = 14;
  options.moi_epoch_vgpr = 15;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 1);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.kernels.front().barrier_sites.size(), 1u);
  ASSERT_EQ(non_entry_prologue_patch_count(result), 1u);
  EXPECT_EQ(only_non_entry_prologue_patch(result).kind,
            ConSanPatchKind::TrampolineMoiBarrierRecord);
  EXPECT_EQ(only_non_entry_prologue_patch(result).anchor_offset, 0u);
  EXPECT_EQ(only_non_entry_prologue_patch(result).trampoline_offset,
            text_words.size() * sizeof(uint32_t));
  EXPECT_EQ(only_non_entry_prologue_patch(result).original_size, sizeof(uint32_t));

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);

  std::vector<uint32_t> expected_prefix;
  const auto fwd = compute_sopp_branch_simm16(0, text_words.size() * sizeof(uint32_t));
  ASSERT_TRUE(fwd);
  expected_prefix.push_back(build_s_branch(*fwd, ROCJITSU_CODE_ARCH_RDNA4));
  expected_prefix.push_back(text_words[1]);
  std::vector<uint32_t> actual_words(patched.text_sections().front()->size() / sizeof(uint32_t));
  std::memcpy(actual_words.data(), patched.text_sections().front()->data(),
              actual_words.size() * sizeof(uint32_t));
  ASSERT_GE(actual_words.size(), expected_prefix.size());
  EXPECT_TRUE(std::equal(expected_prefix.begin(), expected_prefix.end(), actual_words.begin()));

  const ConSanPatchInfo &patch = only_non_entry_prologue_patch(result);
  const std::vector<uint32_t> trampoline_words =
      text_words_at_offset(patched, patch.trampoline_offset, patch.trampoline_size);

  const auto mbcnt_lo = build_v_mbcnt_lo_u32_b32(
      /*vdst=*/13, /*src0=*/0xC1, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
  const auto mbcnt_hi = build_v_mbcnt_hi_u32_b32(
      /*vdst=*/13, /*src0=*/0xC1, vector_source_vgpr(13), ROCJITSU_CODE_ARCH_RDNA4);
  const auto first_active_lane = build_v_cmp_eq_u32_e32_vcc(scalar_positive_inline_u32(0),
                                                            /*vsrc1=*/13, ROCJITSU_CODE_ARCH_RDNA4);
  const auto save_exec =
      build_s_and_saveexec_b64(/*sdst=*/30, /*ssrc0=*/kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_exec = build_s_mov_b64(/*sdst=*/126, /*ssrc0=*/30, ROCJITSU_CODE_ARCH_RDNA4);
  const auto save_scc = build_rdna4_s_cselect_b32(
      /*sdst=*/34, scalar_positive_inline_u32(1), scalar_positive_inline_u32(0),
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto save_vcc = build_s_mov_b64(/*sdst=*/32, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_vcc = build_s_mov_b64(kRdna4VccLo, /*ssrc0=*/32, ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_scc = build_rdna4_s_cmp_lg_u32(
      /*ssrc0=*/34, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
  const auto skip_overflow = build_s_cbranch_vccz(/*offset_dwords=*/0, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(mbcnt_lo);
  ASSERT_TRUE(mbcnt_hi);
  ASSERT_TRUE(first_active_lane);
  ASSERT_TRUE(save_exec);
  ASSERT_TRUE(restore_exec);
  ASSERT_TRUE(save_scc);
  ASSERT_TRUE(save_vcc);
  ASSERT_TRUE(restore_vcc);
  ASSERT_TRUE(restore_scc);
  ASSERT_TRUE(skip_overflow);
  EXPECT_TRUE(contains_subsequence(trampoline_words, *mbcnt_lo));
  EXPECT_TRUE(contains_subsequence(trampoline_words, *mbcnt_hi));
  EXPECT_TRUE(contains_subsequence(trampoline_words,
                                   std::array<uint32_t, 2>{*first_active_lane, *save_exec}));
  EXPECT_TRUE(std::find(trampoline_words.begin(), trampoline_words.end(),
                        build_v_mov_b32_e32(/*vdst=*/13, /*src0=*/30, ROCJITSU_CODE_ARCH_RDNA4)) !=
              trampoline_words.end());
  EXPECT_TRUE(std::find(trampoline_words.begin(), trampoline_words.end(),
                        build_v_mov_b32_e32(/*vdst=*/13, /*src0=*/31, ROCJITSU_CODE_ARCH_RDNA4)) !=
              trampoline_words.end());
  EXPECT_TRUE(std::find(trampoline_words.begin(), trampoline_words.end(), *restore_exec) !=
              trampoline_words.end());
  EXPECT_TRUE(
      contains_subsequence(trampoline_words, std::array<uint32_t, 2>{*save_scc, *save_vcc}));
  EXPECT_TRUE(contains_subsequence(
      trampoline_words, std::array<uint32_t, 3>{*restore_exec, *restore_vcc, *restore_scc}));
  EXPECT_TRUE(std::find(trampoline_words.begin(), trampoline_words.end(), kBarrierWait) !=
              trampoline_words.end());
  EXPECT_TRUE(std::any_of(trampoline_words.begin(), trampoline_words.end(),
                          [](uint32_t word) { return (word & 0xFFFF0000u) == 0xBFA30000u; }));

  const uint64_t base = *options.moi_report_buffer_address;
  const auto mov_barrier_count_lo = build_v_mov_b32_e64_literal(
      8, static_cast<uint32_t>(base + offsetof(ConSanMoiReportHeader, barrier_record_count)),
      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(mov_barrier_count_lo);
  EXPECT_TRUE(contains_subsequence(trampoline_words, *mov_barrier_count_lo));
  EXPECT_GT(only_non_entry_prologue_patch(result).trampoline_size, 0u);
}

TEST(ConSanMoi, Cdna4AdjacentFullBarriersShareOneRecordProbe) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  const auto barrier = build_cdna4_s_barrier(kArch);
  ASSERT_TRUE(barrier);
  const std::array<uint32_t, 3> text_words = {
      *barrier,
      *barrier,
      build_s_endpgm(kArch),
  };

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_track_barriers = true;
  options.moi_track_atomics = false;
  options.scratch_vgpr = 8u;
  options.moi_exec_save_sgpr = 30u;
  options.moi_owner_vgpr = 14u;
  options.moi_epoch_vgpr = 15u;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1u, 0u, 0u, 0u, 1u);
  options.max_patches = 2u;

  const ConSanResult result =
      try_patch_consan(make_cdna4_lds_code_object(text_words, "adjacent_barriers"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  std::vector<const ConSanPatchInfo *> barriers;
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.kind == ConSanPatchKind::TrampolineMoiBarrierRecord)
      barriers.push_back(&patch);
  }
  ASSERT_EQ(barriers.size(), 1u);
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.kernels[0].barrier_sites.size(), 2u);
  EXPECT_EQ(barriers[0]->anchor_offset, result.kernels[0].barrier_sites[0].text_offset);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> second_barrier = text_words_at_offset(
      patched, result.kernels[0].barrier_sites[1].text_offset, sizeof(uint32_t));
  ASSERT_EQ(second_barrier.size(), 1u);
  EXPECT_EQ(second_barrier.front(), *barrier);
  const auto redundant_disposition = std::ranges::find(
      result.site_dispositions, ConSanSiteDispositionReason::RedundantAdjacentBarrier,
      &ConSanSiteDispositionRecord::reason);
  ASSERT_NE(redundant_disposition, result.site_dispositions.end());
  EXPECT_EQ(redundant_disposition->disposition, ConSanSiteDisposition::NotApplicable);
  EXPECT_EQ(redundant_disposition->lowering_outcome, ConSanSiteLoweringOutcome::NotApplicable);
  EXPECT_STREQ(consan_site_disposition_reason_name(redundant_disposition->reason),
               "redundant_adjacent_barrier");
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, BarrierRecordUsesLocalIndirectIslandForFarAppendedHelper) {
  constexpr size_t kLargeTextWords = 33000u;
  std::vector<uint32_t> text_words = {
      0xBF940000u, // s_barrier_wait -1
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  text_words.resize(10u, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words.resize(kLargeTextWords - 1u, build_s_mov_b32(100, 100, ROCJITSU_CODE_ARCH_RDNA4));
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4));
  const uint64_t original_text_size = text_words.size() * sizeof(uint32_t);
  ASSERT_FALSE(compute_sopp_branch_simm16(0u, original_text_size));
  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  mutate_elf_symbol(bytes, 1, [](Elf64_Sym &symbol) { symbol.st_size = 2u * sizeof(uint32_t); });

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_track_barriers = true;
  options.moi_track_atomics = false;
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 30;
  options.moi_owner_vgpr = 14;
  options.moi_epoch_vgpr = 15;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 1);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto island = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiIndirectBranchIsland, &ConSanPatchInfo::kind);
  const auto body = std::ranges::find(result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord,
                                      &ConSanPatchInfo::kind);
  ASSERT_NE(island, result.patches.end());
  ASSERT_NE(body, result.patches.end());
  EXPECT_EQ(island->anchor_offset, 0u);
  EXPECT_EQ(island->trampoline_offset, 2u * sizeof(uint32_t));
  EXPECT_EQ(island->trampoline_size, 8u * sizeof(uint32_t));
  EXPECT_EQ(body->trampoline_offset, original_text_size);
  EXPECT_TRUE(compute_sopp_branch_simm16(island->anchor_offset, island->trampoline_offset));
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  EXPECT_TRUE(patched.is_valid());
}

TEST(ConSanMoi, BarrierRecordUsesEntryCapturedWorkgroupIds) {
  constexpr uint32_t kBarrierWait = 0xBF940000u;
  const std::array<uint32_t, 2> text_words = {
      kBarrierWait,
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);

  ConSanOptions options = moi_options();
  options.moi_track_barriers = true;
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 30;
  options.moi_owner_vgpr = 14;
  options.moi_epoch_vgpr = 15;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 1);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(non_entry_prologue_patch_count(result), 1u);
  const auto barrier = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord, &ConSanPatchInfo::kind);
  const auto prologue = std::ranges::find(
      result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue, &ConSanPatchInfo::kind);
  ASSERT_NE(barrier, result.patches.end());
  ASSERT_NE(prologue, result.patches.end());
  ASSERT_TRUE(result.resolved_moi_record_replay_workgroup_vgprs.complete());

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);

  const std::vector<uint32_t> barrier_words =
      text_words_at_offset(patched, barrier->trampoline_offset, barrier->trampoline_size);
  const std::vector<uint32_t> prologue_words =
      text_words_at_offset(patched, prologue->trampoline_offset, prologue->trampoline_size);

  const std::vector<uint32_t> expected_x = {
      build_v_mov_b32_e32(*result.resolved_moi_record_replay_workgroup_vgprs.x,
                          ttmp_scalar_operand(kTtmpRdna4GridX), ROCJITSU_CODE_ARCH_RDNA4),
  };
  const auto y_shift_left = build_v_lshlrev_b32_e32(
      *result.resolved_moi_record_replay_workgroup_vgprs.y, scalar_positive_inline_u32(16),
      *result.resolved_moi_record_replay_workgroup_vgprs.y, ROCJITSU_CODE_ARCH_RDNA4);
  const auto y_shift_right = build_v_lshrrev_b32_e32(
      *result.resolved_moi_record_replay_workgroup_vgprs.y, scalar_positive_inline_u32(16),
      *result.resolved_moi_record_replay_workgroup_vgprs.y, ROCJITSU_CODE_ARCH_RDNA4);
  const auto z_shift = build_v_lshrrev_b32_e32(
      *result.resolved_moi_record_replay_workgroup_vgprs.z, scalar_positive_inline_u32(16),
      *result.resolved_moi_record_replay_workgroup_vgprs.z, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(y_shift_left);
  ASSERT_TRUE(y_shift_right);
  ASSERT_TRUE(z_shift);
  const std::vector<uint32_t> expected_y = {
      build_v_mov_b32_e32(*result.resolved_moi_record_replay_workgroup_vgprs.y,
                          ttmp_scalar_operand(kTtmpRdna4GridYz), ROCJITSU_CODE_ARCH_RDNA4),
      *y_shift_left,
      *y_shift_right,
  };
  const std::vector<uint32_t> expected_z = {
      build_v_mov_b32_e32(*result.resolved_moi_record_replay_workgroup_vgprs.z,
                          ttmp_scalar_operand(kTtmpRdna4GridYz), ROCJITSU_CODE_ARCH_RDNA4),
      *z_shift,
  };
  EXPECT_TRUE(contains_subsequence(prologue_words, expected_x));
  EXPECT_TRUE(contains_subsequence(prologue_words, expected_y));
  EXPECT_TRUE(contains_subsequence(prologue_words, expected_z));
  constexpr uint16_t kBarrierValueVgpr = 13u;
  for (uint16_t coordinate : {*result.resolved_moi_record_replay_workgroup_vgprs.x,
                              *result.resolved_moi_record_replay_workgroup_vgprs.y,
                              *result.resolved_moi_record_replay_workgroup_vgprs.z}) {
    EXPECT_NE(std::ranges::find(barrier_words, build_v_mov_b32_e32(kBarrierValueVgpr,
                                                                   vector_source_vgpr(coordinate),
                                                                   ROCJITSU_CODE_ARCH_RDNA4)),
              barrier_words.end());
  }
}

TEST(ConSanMoi, BarrierRecordAutomaticallyPlansScratchAndScalarState) {
  constexpr uint32_t kBarrierWait = 0xBF940000u;
  const std::array<uint32_t, 2> text_words = {
      kBarrierWait,
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options = moi_options();
  options.moi_track_barriers = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 1);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.moi_exec_save_sgprs_automatic);
  const auto patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &item) {
    return item.kind == ConSanPatchKind::TrampolineMoiBarrierRecord;
  });
  ASSERT_NE(patch, result.patches.end());
  ASSERT_TRUE(patch->scratch_vgpr);
  EXPECT_LT(*patch->scratch_vgpr, 256u);
  const auto plan = std::ranges::find_if(result.resource_plans, [](const auto &item) {
    return item.site_kind == ConSanResourceSiteKind::Barrier;
  });
  ASSERT_NE(plan, result.resource_plans.end());
  EXPECT_EQ(plan->scratch_vgpr, patch->scratch_vgpr);
  EXPECT_EQ(result.resource_plan_summary.dead_plans, 1u);
  EXPECT_EQ(std::ranges::count(result.resource_plans, ConSanResourceSiteKind::Fence,
                               &ConSanCandidateResourcePlan::site_kind),
            0);
}

TEST(ConSanMoi, BarrierRecordForcedSpillUsesPlannedPrivateWindow) {
  constexpr uint32_t kBarrierWait = 0xBF940000u;
  const std::array<uint32_t, 2> text_words = {
      kBarrierWait,
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options = moi_options();
  options.moi_track_barriers = true;
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 1);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &item) {
    return item.kind == ConSanPatchKind::TrampolineMoiBarrierRecord;
  });
  ASSERT_NE(patch, result.patches.end());
  EXPECT_EQ(patch->spilled_vgpr_count, 6u);
  EXPECT_GT(patch->required_private_segment_size, 0u);
  EXPECT_EQ(result.resource_plan_summary.spill_plans, 1u);
  EXPECT_EQ(result.resource_plan_summary.emitted_spill_patches, 1u);
  EXPECT_EQ(result.resource_plan_summary.emitted_spill_slot_bytes, 24u);
}

TEST(ConSanMoi, Rdna4DynamicStackBarrierRecordUsesSiteLocalSpillFrames) {
  constexpr uint32_t kBarrierWait = 0xBF940000u;
  const std::array<uint32_t, 2> text_words = {
      kBarrierWait,
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "dynamic_barrier_spill", kRdna4Wave64AllVgprsGranulated,
      /*wave32=*/false, /*uses_dynamic_stack=*/true);
  ConSanOptions options = moi_options();
  options.moi_track_barriers = true;
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 1);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto barrier = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord, &ConSanPatchInfo::kind);
  ASSERT_NE(barrier, result.patches.end());
  EXPECT_EQ(barrier->spilled_vgpr_count, 6u);
  EXPECT_EQ(barrier->dynamic_private_segment_addend, 24u);
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, Cdna4BarrierRecordForcedSpillUsesNativePrivateWindows) {
  const auto barrier = build_cdna4_s_barrier(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(barrier);
  std::vector<uint32_t> text_words(320, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  text_words[0] = *barrier;
  text_words[1] =
      build_v_mov_b32_e32(/*vdst=*/0, vector_source_vgpr(255), ROCJITSU_CODE_ARCH_CDNA4);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);
  constexpr uint32_t kCdna4Wave64AllVgprsGranulated = 31u;
  const std::vector<uint8_t> bytes = make_cdna4_lds_code_object(text_words, "barrier_forced_spill",
                                                                kCdna4Wave64AllVgprsGranulated);
  ConSanOptions options = moi_options();
  options.moi_track_barriers = true;
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 1);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << "warnings=" << testing::PrintToString(result.warnings)
                               << " errors=" << testing::PrintToString(result.errors);
  EXPECT_EQ(result.target, ROCJITSU_CODE_TARGET_GFX950);
  EXPECT_EQ(result.arch, ROCJITSU_CODE_ARCH_CDNA4);
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_TRUE(result.final_validation_passed);
  const auto patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &item) {
    return item.kind == ConSanPatchKind::TrampolineMoiBarrierRecord;
  });
  ASSERT_NE(patch, result.patches.end()) << "patches=" << testing::PrintToString(result.patches)
                                         << " warnings=" << testing::PrintToString(result.warnings)
                                         << " kernels=" << testing::PrintToString(result.kernels);
  EXPECT_EQ(patch->spilled_vgpr_count, 6u);
  EXPECT_EQ(patch->required_private_segment_size, 32u);
  const auto plan = std::ranges::find_if(result.resource_plans, [](const auto &item) {
    return item.site_kind == ConSanResourceSiteKind::Barrier;
  });
  ASSERT_NE(plan, result.resource_plans.end());
  EXPECT_EQ(plan->source, ConSanRegisterAllocationSource::SpillRequired);
  EXPECT_EQ(plan->reason, ConSanRegisterPlanReason::None);
  EXPECT_EQ(plan->scratch_vgpr_count, 6u);
  ASSERT_TRUE(plan->scratch_vgpr);
  EXPECT_EQ(*plan->scratch_vgpr % 2u, 0u);
  EXPECT_EQ(result.resource_plan_summary.spill_plans, 1u);
  EXPECT_EQ(result.resource_plan_summary.emitted_spill_patches, 1u);
  EXPECT_EQ(result.resource_plan_summary.emitted_spill_slot_bytes, 24u);
}

TEST(ConSanMoi, RecordReplayPersistentEpochStillEmitsBarrierRecords) {
  constexpr uint32_t kBarrierWait = 0xBF940000u;
  const std::array<uint32_t, 4> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32
      kBarrierWait,
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_dynamic_access_records = true;
  options.moi_track_barriers = true;
  options.moi_init_owner_epoch = true;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 24;
  options.moi_epoch_vgpr = 25;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(8, 0, 0, 0, 8);

  const auto result = try_patch_consan(make_rdna4_lds_code_object(text_words), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord,
                               &ConSanPatchInfo::kind),
            1);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue,
                               &ConSanPatchInfo::kind),
            1);
}

TEST(ConSanMoi, Cdna4RecordReplayPersistentEpochEmitsBarrierRecord) {
  const auto guest = build_cdna4_ds_store_b32(
      /*vaddr=*/2, /*vdata=*/3, /*byte_offset=*/0, ROCJITSU_CODE_ARCH_CDNA4);
  const auto barrier = build_cdna4_s_barrier(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(guest && barrier);
  std::vector<uint32_t> text_words(320, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  std::copy(guest->begin(), guest->end(), text_words.begin());
  text_words[guest->size()] = *barrier;
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_dynamic_access_records = true;
  options.moi_track_barriers = true;
  options.moi_init_owner_epoch = true;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 24;
  options.moi_epoch_vgpr = 25;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(8, 0, 0, 0, 8);

  const ConSanResult result =
      try_patch_consan(make_cdna4_lds_code_object(text_words, "persistent_epoch_barrier"), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << "warnings=" << testing::PrintToString(result.warnings)
                               << " errors=" << testing::PrintToString(result.errors);
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord,
                               &ConSanPatchInfo::kind),
            1);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue,
                               &ConSanPatchInfo::kind),
            1);
}

TEST(ConSanMoi, Gfx1201RecordReplayAvoidsPrivateEpochOnHotAccesses) {
  constexpr uint32_t kBarrierWait = 0xBF940000u;
  std::vector<uint32_t> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32 v0, v0
  };
  text_words.insert(text_words.end(), 33u, kBarrierWait);
  text_words.push_back(
      build_v_mov_b32_e32(/*vdst=*/62, vector_source_vgpr(62), ROCJITSU_CODE_ARCH_RDNA4));
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4));
  constexpr uint32_t kWave64Vgpr64Granulated = 15;
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "record_replay_large_barrier_pressure", kWave64Vgpr64Granulated);

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_dynamic_access_records = true;
  options.moi_track_barriers = true;
  options.moi_init_owner_epoch = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(64, 1, 0, 0, 64);

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
  EXPECT_EQ(*result.resolved_moi_persistent_epoch_sgpr,
            *result.resolved_moi_persistent_owner_sgpr + 1u);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord,
                               &ConSanPatchInfo::kind),
            1);
  const auto access = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore, &ConSanPatchInfo::kind);
  const auto barrier = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord, &ConSanPatchInfo::kind);
  const auto prologue = std::ranges::find(
      result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue, &ConSanPatchInfo::kind);
  ASSERT_NE(access, result.patches.end());
  ASSERT_NE(barrier, result.patches.end());
  ASSERT_NE(prologue, result.patches.end());
  EXPECT_FALSE(access->persistent_epoch_private_offset);
  EXPECT_FALSE(barrier->persistent_epoch_private_offset);
  EXPECT_FALSE(prologue->persistent_epoch_private_offset);
}

TEST(ConSanMoi, Rdna4ScalarRelativeVariantsUseFixedWaveSgprBound) {
  struct Target {
    rj_code_arch_t arch;
    std::string_view label;
  };
  constexpr std::array kTargets = {
      Target{ROCJITSU_CODE_ARCH_RDNA4, "gfx1201"},
      Target{ROCJITSU_CODE_ARCH_CDNA5, "gfx1250"},
  };
  constexpr std::array kIndirectInstructions = {
      0xBE804002u, // s_movrels_b32 s0, s2
      0xBE804202u, // s_movreld_b32 s0, s2
      0xBE804402u, // s_movrelsd_2_b32 s0, s2
  };
  constexpr uint32_t kBarrierWait = 0xBF940000u;

  for (const Target &target : kTargets) {
    for (uint32_t indirect_instruction : kIndirectInstructions) {
      SCOPED_TRACE(std::string(target.label) + ":" + std::to_string(indirect_instruction));
      std::vector<uint32_t> text_words = {
          0xD8340000u,
          0x00000000u, // ds_store_b32 v0, v0
      };
      text_words.insert(text_words.end(), 33u, kBarrierWait);
      text_words.push_back(0xBED00000u); // s_mov_b32 s80, s0
      text_words.push_back(indirect_instruction);
      text_words.push_back(build_s_endpgm(target.arch));
      const std::vector<uint8_t> bytes =
          target.arch == ROCJITSU_CODE_ARCH_CDNA5
              ? make_gfx1250_code_object(text_words, "rdna_indirect_sgpr",
                                         /*vgpr_granulated=*/3u)
              : make_rdna4_lds_code_object(text_words, "rdna_indirect_sgpr",
                                           /*vgpr_granulated=*/3u);

      ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
      options.moi_dynamic_access_records = true;
      options.moi_track_barriers = true;
      options.moi_init_owner_epoch = true;
      options.moi_report_buffer_address = 0x123456780000ull;
      options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(64, 1, 0, 0, 64);

      const ConSanResult result = try_patch_consan(bytes, options);

      ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
      ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
      ASSERT_FALSE(result.resource_plans.empty());
      EXPECT_TRUE(std::ranges::all_of(result.resource_plans, [](const auto &plan) {
        return plan.current_sgpr_count == 106u && plan.max_referenced_sgpr_count >= 81u &&
               plan.scalar_tail_floor == 106u && plan.has_indirect_sgpr_access &&
               plan.sgpr_reference_coverage_complete;
      }));
      EXPECT_FALSE(result.moi_persistent_sgprs_automatic);
      EXPECT_FALSE(result.resolved_moi_persistent_owner_sgpr);
      EXPECT_FALSE(result.resolved_moi_persistent_epoch_sgpr);
      EXPECT_TRUE(result.final_validation_passed);
    }
  }
}

TEST(ConSanMoi, Gfx1250FullVgprRecordReplayUsesScalarEpochCoalescing) {
  constexpr uint32_t kBarrierWait = 0xBF94FFFFu;
  std::vector<uint32_t> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32 v0, v0
  };
  text_words.insert(text_words.end(), 33u, kBarrierWait);
  text_words.push_back(
      build_v_mov_b32_e32(/*vdst=*/255, vector_source_vgpr(255), ROCJITSU_CODE_ARCH_CDNA5));
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5));

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_track_barriers = true;
  options.moi_init_owner_epoch = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(64, 1, 0, 0, 64);

  const ConSanResult result = try_patch_consan(
      make_gfx1250_code_object(text_words, "gfx1250_scalar_epoch_coalescing"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.moi_persistent_sgprs_automatic);
  EXPECT_FALSE(result.moi_private_epoch_automatic);
  ASSERT_TRUE(result.resolved_moi_persistent_owner_sgpr);
  ASSERT_TRUE(result.resolved_moi_persistent_epoch_sgpr);
  EXPECT_EQ(*result.resolved_moi_persistent_epoch_sgpr,
            *result.resolved_moi_persistent_owner_sgpr + 1u);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord,
                               &ConSanPatchInfo::kind),
            1u);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue,
                               &ConSanPatchInfo::kind),
            1u);
  const auto access = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore, &ConSanPatchInfo::kind);
  ASSERT_NE(access, result.patches.end());
  ASSERT_TRUE(access->scratch_vgpr);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> access_words =
      text_words_at_offset(patched, access->trampoline_offset, access->trampoline_size);
  ASSERT_TRUE(result.resolved_moi_dispatch_id_sgpr);
  EXPECT_TRUE(contains_subsequence(access_words, make_expected_scalar_offset_store_words(
                                                     offsetof(ConSanMoiAccessRecord, generation),
                                                     *result.resolved_moi_dispatch_id_sgpr,
                                                     *access->scratch_vgpr)));
  EXPECT_TRUE(contains_subsequence(
      access_words, make_expected_scalar_offset_store_words(
                        offsetof(ConSanMoiAccessRecord, generation) + sizeof(uint32_t),
                        static_cast<uint16_t>(*result.resolved_moi_dispatch_id_sgpr + 1u),
                        *access->scratch_vgpr)));
  const uint16_t record_value_vgpr = static_cast<uint16_t>(*access->scratch_vgpr + 2u);
  EXPECT_NE(std::ranges::find(access_words,
                              build_v_mov_b32_e32(record_value_vgpr,
                                                  *result.resolved_moi_persistent_epoch_sgpr,
                                                  ROCJITSU_CODE_ARCH_CDNA5)),
            access_words.end());
}

TEST(ConSanMoi, Gfx1250HighSgprPressureSkipsArchitecturalAliasesForPersistentEpoch) {
  constexpr uint32_t kBarrierWait = 0xBF94FFFFu;
  std::vector<uint32_t> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32 v0, v0
      kBarrierWait,
      build_s_mov_b32(/*sdst=*/0u, /*ssrc=*/101u, ROCJITSU_CODE_ARCH_CDNA5),
      build_v_mov_b32_e32(/*vdst=*/255, vector_source_vgpr(255), ROCJITSU_CODE_ARCH_CDNA5),
  };
  text_words.resize(320u, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5));
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5);

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_dynamic_access_records = true;
  options.moi_track_barriers = true;
  options.moi_init_owner_epoch = true;
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(64, 64, 0, 0, 64);

  const ConSanResult result = try_patch_consan(
      make_gfx1250_code_object(text_words, "gfx1250_flat_scratch_alias_pressure"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  ASSERT_TRUE(result.resolved_moi_dispatch_id_sgpr);
  EXPECT_EQ(*result.resolved_moi_dispatch_id_sgpr, 100u);
  EXPECT_TRUE(result.moi_private_epoch_automatic);
  EXPECT_FALSE(result.resolved_moi_persistent_owner_sgpr);
  EXPECT_FALSE(result.resolved_moi_persistent_epoch_sgpr);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("retaining it in entry-captured private state") != std::string::npos;
  })) << testing::PrintToString(result.warnings);
  const auto access = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore, &ConSanPatchInfo::kind);
  const auto prologue = std::ranges::find(
      result.patches, ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue, &ConSanPatchInfo::kind);
  ASSERT_NE(access, result.patches.end());
  ASSERT_NE(prologue, result.patches.end());
  EXPECT_TRUE(access->persistent_record_replay_workgroup_private_offsets.complete());
  EXPECT_EQ(access->persistent_record_replay_workgroup_private_offsets,
            prologue->persistent_record_replay_workgroup_private_offsets);
  EXPECT_NE(std::ranges::find(result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord,
                              &ConSanPatchInfo::kind),
            result.patches.end());
}

TEST(ConSanMoi, Gfx1250RejectsExplicitPersistentStateInFlatScratch) {
  std::vector<uint32_t> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32 v0, v0
  };
  text_words.resize(128u, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5));
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5);

  const auto patch_with = [&](ConSanOptions options) {
    options.moi_report_buffer_address = 0x123456780000ull;
    options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(64, 0, 0, 0);
    return try_patch_consan(
        make_gfx1250_code_object(text_words, "gfx1250_explicit_flat_scratch_state"), options);
  };

  const auto expect_special_alias_rejected = [&](const ConSanOptions &options,
                                                 std::string_view diagnostic) {
    const ConSanResult result = patch_with(options);
    EXPECT_EQ(result.outcome, ConSanTransformOutcome::Unsupported);
    EXPECT_FALSE(result.modified);
    EXPECT_TRUE(std::ranges::any_of(result.warnings, [&](const std::string &warning) {
      return warning.find(diagnostic) != std::string::npos;
    })) << testing::PrintToString(result.warnings);
  };

  ConSanOptions dispatch_options = moi_options(ConSanMoiEngine::RecordReplay);
  dispatch_options.moi_dispatch_id_sgpr = 102u;
  expect_special_alias_rejected(dispatch_options, "violates architectural reservation");

  ConSanOptions owner_options = moi_options(ConSanMoiEngine::RecordReplay);
  owner_options.moi_persistent_owner_sgpr = 102u;
  owner_options.moi_persistent_epoch_sgpr = 81u;
  owner_options.moi_init_owner_epoch = true;
  expect_special_alias_rejected(owner_options, "architectural special SGPR");

  ConSanOptions epoch_options = moi_options(ConSanMoiEngine::RecordReplay);
  epoch_options.moi_persistent_owner_sgpr = 80u;
  epoch_options.moi_persistent_epoch_sgpr = 103u;
  epoch_options.moi_init_owner_epoch = true;
  expect_special_alias_rejected(epoch_options, "architectural special SGPR");

  ConSanOptions workgroup_options = moi_options(ConSanMoiEngine::RecordReplay);
  workgroup_options.moi_persistent_owner_sgpr = 80u;
  workgroup_options.moi_persistent_epoch_sgpr = 81u;
  workgroup_options.moi_persistent_workgroup_key_sgpr = 102u;
  workgroup_options.moi_init_owner_epoch = true;
  expect_special_alias_rejected(workgroup_options, "architectural special SGPR");
}

TEST(ConSanMoi, Gfx1250RejectsConfiguredPersistentStateInXnackMask) {
  std::vector<uint32_t> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32 v0, v0
  };
  text_words.resize(128u, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5));
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5);

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  // LLVM reserves s104:s105 as the gfx1250 XNACK_MASK pair even though the
  // target does not expose a persistent XNACK selector to ordinary code.
  options.moi_persistent_owner_sgpr = 104u;
  options.moi_persistent_epoch_sgpr = 105u;
  options.moi_init_owner_epoch = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(64, 0, 0, 0);

  const ConSanResult result = try_patch_consan(
      make_gfx1250_code_object(text_words, "gfx1250_explicit_state_above_flat_scratch"), options);

  EXPECT_EQ(result.outcome, ConSanTransformOutcome::Unsupported);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("architectural special SGPR") != std::string::npos;
  })) << testing::PrintToString(result.warnings);
}

TEST(ConSanMoi, Gfx1250RejectsConfiguredPersistentStateAtOrdinarySgprLimit) {
  std::vector<uint32_t> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32 v0, v0
  };
  text_words.resize(128u, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5));
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5);

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  // s106:s107 are VCC and begin immediately after the ordinary s0:s105 file.
  // This locks the limit boundary; the adjacent tests cover both architectural
  // alias pairs immediately below it.
  options.moi_persistent_owner_sgpr = 106u;
  options.moi_persistent_epoch_sgpr = 107u;
  options.moi_init_owner_epoch = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(64, 0, 0, 0);

  const ConSanResult result = try_patch_consan(
      make_gfx1250_code_object(text_words, "gfx1250_explicit_state_at_ordinary_limit"), options);

  EXPECT_EQ(result.outcome, ConSanTransformOutcome::Unsupported);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("architectural special SGPR") != std::string::npos;
  })) << testing::PrintToString(result.warnings);
}

TEST(ConSanMoi, SupportedCdnaTargetsHonorConfiguredPersistentStateSgprLimit) {
  using GuestBuilder =
      std::optional<std::array<uint32_t, 2>> (*)(uint16_t, uint16_t, uint8_t, rj_code_arch_t);
  using ObjectBuilder = std::vector<uint8_t> (*)(std::span<const uint32_t>, std::string_view);
  struct Target {
    rj_code_arch_t arch;
    std::string_view label;
    std::string_view object_name;
    GuestBuilder build_guest;
    ObjectBuilder make_object;
  };
  constexpr std::array<Target, 2> kTargets = {{
      {ROCJITSU_CODE_ARCH_CDNA3, "gfx942/cdna3", "cdna3_explicit_state_at_ordinary_limit",
       &build_cdna3_ds_store_b32,
       +[](std::span<const uint32_t> words, std::string_view name) {
         return make_cdna3_lds_code_object(words, name);
       }},
      {ROCJITSU_CODE_ARCH_CDNA4, "gfx950/cdna4", "cdna4_explicit_state_at_ordinary_limit",
       &build_cdna4_ds_store_b32,
       +[](std::span<const uint32_t> words, std::string_view name) {
         return make_cdna4_lds_code_object(words, name);
       }},
  }};
  for (const Target &target : kTargets) {
    SCOPED_TRACE(target.label);
    const auto guest = target.build_guest(/*vaddr=*/2, /*vdata=*/3, /*byte_offset=*/0, target.arch);
    ASSERT_TRUE(guest);
    std::vector<uint32_t> text_words(128u, build_s_nop(0, target.arch));
    std::copy(guest->begin(), guest->end(), text_words.begin());
    text_words.back() = build_s_endpgm(target.arch);
    const std::vector<uint8_t> bytes = target.make_object(text_words, target.object_name);

    const auto patch_with = [&](uint16_t owner) {
      ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
      options.moi_persistent_owner_sgpr = owner;
      options.moi_persistent_epoch_sgpr = static_cast<uint16_t>(owner + 1u);
      options.moi_init_owner_epoch = true;
      options.moi_report_buffer_address = 0x123456780000ull;
      options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(64, 0, 0, 0);
      return try_patch_consan(bytes, options);
    };

    // s100:s101 are the final pair inside the CDNA ordinary scalar file.
    const ConSanResult below_limit = patch_with(100u);
    ASSERT_TRUE(consan_patch_succeeded(below_limit)) << testing::PrintToString(below_limit.errors);
    EXPECT_TRUE(below_limit.modified) << testing::PrintToString(below_limit.warnings);
    EXPECT_TRUE(below_limit.final_validation_passed);

    // s102:s103 start at the limit and must be rejected.
    const ConSanResult at_limit = patch_with(102u);
    EXPECT_EQ(at_limit.outcome, ConSanTransformOutcome::Unsupported);
    EXPECT_FALSE(at_limit.modified);
    EXPECT_TRUE(std::ranges::any_of(at_limit.warnings, [](const std::string &warning) {
      return warning.find("architectural special SGPR") != std::string::npos;
    })) << testing::PrintToString(at_limit.warnings);
  }
}

TEST(ConSanMoi, Cdna4AccvgprBoundaryRecordReplayUsesScalarEpochCoalescing) {
  const auto guest = build_cdna4_ds_store_b32(
      /*vaddr=*/2, /*vdata=*/3, /*byte_offset=*/0, ROCJITSU_CODE_ARCH_CDNA4);
  const auto barrier = build_cdna4_s_barrier(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(guest && barrier);
  std::vector<uint32_t> text_words(320, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  std::copy(guest->begin(), guest->end(), text_words.begin());
  text_words[guest->size()] = *barrier;
  text_words[guest->size() + 1u] =
      build_v_mov_b32_e32(/*vdst=*/0, vector_source_vgpr(7), ROCJITSU_CODE_ARCH_CDNA4);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);
  std::vector<uint8_t> bytes =
      make_cdna4_lds_code_object(text_words, "accvgpr_scalar_epoch", /*vgpr_granulated=*/3u);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    // Encoded 1 makes v8 the first accumulator register. The guest references
    // through v7, leaving no ordinary-VGPR room for the persistent pair.
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 1u);
  });

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_track_barriers = true;
  // The runtime enables atomic tracking generically. An object with no
  // admitted atomic or fence site still needs the access record's exact
  // entry-captured workgroup tuple, but no separate compact atomic key.
  options.moi_track_atomics = true;
  options.moi_init_owner_epoch = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(64, 1, 0, 0, 64);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.moi_persistent_sgprs_automatic);
  EXPECT_FALSE(result.moi_persistent_vgprs_automatic);
  EXPECT_FALSE(result.moi_private_epoch_automatic);
  EXPECT_FALSE(result.resolved_moi_workgroup_key_vgpr);
  EXPECT_FALSE(result.resolved_moi_persistent_workgroup_key_sgpr);
  EXPECT_TRUE(result.resolved_moi_record_replay_workgroup_vgprs.empty());
  ASSERT_TRUE(result.resolved_moi_persistent_owner_sgpr);
  ASSERT_TRUE(result.resolved_moi_persistent_epoch_sgpr);
  ASSERT_TRUE(result.resolved_moi_record_replay_workgroup_sgprs.complete());
  EXPECT_EQ(*result.resolved_moi_persistent_epoch_sgpr,
            *result.resolved_moi_persistent_owner_sgpr + 1u);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord,
                               &ConSanPatchInfo::kind),
            1u);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue,
                               &ConSanPatchInfo::kind),
            1u);

  AmdGpuCodeObject original(bytes.data(), bytes.size());
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(original.is_valid());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(original.kernels().size(), 1u);
  ASSERT_EQ(patched.kernels().size(), 1u);
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

  const auto barrier_patch = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord, &ConSanPatchInfo::kind);
  const auto access_patch = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore, &ConSanPatchInfo::kind);
  const auto prologue_patch = std::ranges::find(
      result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue, &ConSanPatchInfo::kind);
  ASSERT_NE(barrier_patch, result.patches.end());
  ASSERT_NE(access_patch, result.patches.end());
  ASSERT_NE(prologue_patch, result.patches.end());
  ASSERT_TRUE(access_patch->scratch_vgpr);
  ASSERT_TRUE(result.resolved_moi_exec_save_sgpr);
  const std::vector<uint32_t> access_words =
      text_words_at_offset(patched, access_patch->trampoline_offset, access_patch->trampoline_size);
  const std::vector<uint32_t> prologue_words = text_words_at_offset(
      patched, prologue_patch->trampoline_offset, prologue_patch->trampoline_size);
  const uint16_t record_value_vgpr = static_cast<uint16_t>(*access_patch->scratch_vgpr + 2u);
  const std::array<uint16_t, 3> captured_coordinates = {
      *result.resolved_moi_record_replay_workgroup_sgprs.x,
      *result.resolved_moi_record_replay_workgroup_sgprs.y,
      *result.resolved_moi_record_replay_workgroup_sgprs.z,
  };
  for (size_t dimension = 0; dimension < captured_coordinates.size(); ++dimension) {
    const uint16_t coordinate = captured_coordinates[dimension];
    EXPECT_NE(std::ranges::find(access_words, build_v_mov_b32_e32(record_value_vgpr, coordinate,
                                                                  ROCJITSU_CODE_ARCH_CDNA4)),
              access_words.end());
    const uint16_t source =
        static_cast<uint16_t>(AMDHSA_BITS_GET(patched_descriptor.compute_pgm_rsrc2,
                                              kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT) +
                              dimension);
    EXPECT_NE(std::ranges::find(prologue_words,
                                build_s_mov_b32(coordinate, source, ROCJITSU_CODE_ARCH_CDNA4)),
              prologue_words.end());
  }
  const std::vector<uint32_t> barrier_words = text_words_at_offset(
      patched, barrier_patch->trampoline_offset, barrier_patch->trampoline_size);
  EXPECT_NE(std::ranges::find(barrier_words, *barrier), barrier_words.end());
}

TEST(ConSanMoi, Cdna4AutomaticBankedReplaySkipsOccupiedExactTupleScalarHole) {
  const auto guest = build_cdna4_ds_store_b32(
      /*vaddr=*/2, /*vdata=*/3, /*byte_offset=*/0, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(guest);
  std::vector<uint32_t> text_words(320u, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  std::copy(guest->begin(), guest->end(), text_words.begin());
  size_t cursor = guest->size();
  text_words[cursor++] =
      build_v_mov_b32_e32(/*vdst=*/0, vector_source_vgpr(15), ROCJITSU_CODE_ARCH_CDNA4);
  // Reference every scalar below s72 and s74 after the access. A five-register
  // automatic owner/epoch/x/y/z tuple beginning at s72 would therefore
  // overlap its third register even though s72:s73 themselves are untouched.
  // The next complete owner-scope hole is s75:s79; s80:s93 and s94:s95 are
  // reserved explicitly for transient and dispatch state.
  for (uint16_t sgpr = 0u; sgpr < 72u; ++sgpr)
    text_words[cursor++] = build_s_mov_b32(/*sdst=*/87u, sgpr, ROCJITSU_CODE_ARCH_CDNA4);
  text_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/74u, ROCJITSU_CODE_ARCH_CDNA4);
  text_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/87u, ROCJITSU_CODE_ARCH_CDNA4);
  text_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/96u, ROCJITSU_CODE_ARCH_CDNA4);
  text_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/97u, ROCJITSU_CODE_ARCH_CDNA4);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);
  // A separate linked kernel does use s75:s79. It is not an owner of the LDS
  // site and therefore must not poison the complete owner-scope proof.
  const std::array<uint32_t, 6> unrelated_words = {
      build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/75u, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/76u, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/77u, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/78u, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/79u, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  std::vector<uint8_t> bytes = make_cdna4_code_object_with_local_function(
      text_words, unrelated_words, {}, /*vgpr_granulated=*/3u, /*function_is_kernel=*/true);
  mutate_kernel_descriptor(bytes, "lds_probe", [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 3u);
    // 104 decoded SGPRs place physical VCC at s98:s99. Dispatch and transient
    // state already fit below it; the selected hole must not grow the
    // allocation or move VCC.
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 12u);
  });

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.moi_exec_save_sgpr = 80u;
  options.moi_dispatch_id_sgpr = 94u;
  const ConSanMoiAutoReportPlan report_plan = plan_consan_moi_auto_report(
      {.engine = ConSanMoiEngine::RecordReplay, .access_range_count = 1});
  ASSERT_TRUE(report_plan.complete());
  const auto layout_override = consan_moi_auto_report_layout_override(report_plan);
  ASSERT_TRUE(layout_override);
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = report_plan.required_bytes;
  options.moi_report_layout = *layout_override;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_TRUE(result.moi_persistent_sgprs_automatic);
  ASSERT_TRUE(result.resolved_moi_persistent_owner_sgpr);
  ASSERT_TRUE(result.resolved_moi_persistent_epoch_sgpr);
  ASSERT_TRUE(result.resolved_moi_record_replay_workgroup_sgprs.complete());
  EXPECT_TRUE(result.resolved_moi_record_replay_workgroup_vgprs.empty());
  EXPECT_EQ(*result.resolved_moi_persistent_owner_sgpr, 75u);
  EXPECT_EQ(*result.resolved_moi_persistent_epoch_sgpr, 76u);
  EXPECT_EQ(*result.resolved_moi_record_replay_workgroup_sgprs.x, 77u);
  EXPECT_EQ(*result.resolved_moi_record_replay_workgroup_sgprs.y, 78u);
  EXPECT_EQ(*result.resolved_moi_record_replay_workgroup_sgprs.z, 79u);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue,
                               &ConSanPatchInfo::kind),
            1u);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
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
  EXPECT_EQ(patched_descriptor.compute_pgm_rsrc1, original_descriptor.compute_pgm_rsrc1);
  EXPECT_EQ(AMDHSA_BITS_GET(patched_descriptor.compute_pgm_rsrc3,
                            kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET),
            AMDHSA_BITS_GET(original_descriptor.compute_pgm_rsrc3,
                            kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET));
  const auto access = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore, &ConSanPatchInfo::kind);
  ASSERT_NE(access, result.patches.end());
  ASSERT_TRUE(access->scratch_vgpr);
  const std::vector<uint32_t> access_words =
      text_words_at_offset(patched, access->trampoline_offset, access->trampoline_size);
  EXPECT_NE(std::ranges::find(access_words,
                              build_v_mov_b32_e32(static_cast<uint16_t>(*access->scratch_vgpr + 2u),
                                                  *result.resolved_moi_persistent_owner_sgpr,
                                                  ROCJITSU_CODE_ARCH_CDNA4)),
            access_words.end());
}

TEST(ConSanMoi, Cdna4ScalarHoleAcceptsCallContextOnlySetpcReturn) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  constexpr uint16_t kReturnSreg = 30u;
  const auto guest = build_cdna4_ds_store_b32(/*vaddr=*/2, /*vdata=*/3, /*byte_offset=*/0, kArch);
  ASSERT_TRUE(guest);

  std::vector<uint32_t> kernel_words(320u, build_s_nop(0, kArch));
  size_t cursor = 0u;
  kernel_words[cursor++] =
      build_s_call_b64(kReturnSreg, static_cast<int16_t>(kernel_words.size() - 1u), kArch);
  std::copy(guest->begin(), guest->end(), kernel_words.begin() + cursor);
  cursor += guest->size();
  kernel_words[cursor++] = build_v_mov_b32_e32(/*vdst=*/0u, vector_source_vgpr(7u), kArch);
  for (uint16_t sgpr = 0u; sgpr < 72u; ++sgpr)
    kernel_words[cursor++] = build_s_mov_b32(/*sdst=*/87u, sgpr, kArch);
  kernel_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/87u, kArch);
  kernel_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/90u, kArch);
  kernel_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/91u, kArch);
  kernel_words.back() = build_s_endpgm(kArch);

  const std::array<uint32_t, 2> helper_words = {
      build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/71u, kArch),
      build_s_setpc_b64(kReturnSreg, kArch),
  };
  // Keep complete-text fallback unavailable. This test succeeds only if the
  // call-owned helper return is accepted by the owner-scope closure proof.
  const std::array<uint32_t, 3> uncovered_tail = {
      build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/72u, kArch),
      build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/73u, kArch),
      build_s_endpgm(kArch),
  };
  std::vector<uint8_t> bytes = make_cdna4_code_object_with_local_function(
      kernel_words, helper_words, uncovered_tail, /*vgpr_granulated=*/3u);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 12u);
  });

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.moi_init_owner_epoch = true;
  options.moi_exec_save_sgpr = 92u;
  options.moi_dispatch_id_sgpr = 88u;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_TRUE(result.moi_persistent_sgprs_automatic);
  ASSERT_TRUE(result.resolved_moi_persistent_owner_sgpr);
  ASSERT_TRUE(result.resolved_moi_persistent_epoch_sgpr);
  EXPECT_EQ(*result.resolved_moi_persistent_owner_sgpr, 72u);
  EXPECT_EQ(*result.resolved_moi_persistent_epoch_sgpr, 73u);
  EXPECT_TRUE(std::ranges::none_of(result.warnings, [](const std::string &warning) {
    return warning.find("using complete-text coverage") != std::string::npos;
  })) << testing::PrintToString(result.warnings);
}

TEST(ConSanMoi, Cdna4ScalarHoleUsesCompleteTextCoverageForUnresolvedCall) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  const auto guest = build_cdna4_ds_store_b32(/*vaddr=*/2, /*vdata=*/3, /*byte_offset=*/0, kArch);
  ASSERT_TRUE(guest);

  std::vector<uint32_t> text_words(320u, build_s_nop(0, kArch));
  size_t cursor = 0u;
  // Model a generated local-label call whose target arithmetic is unavailable
  // to CFG recovery. Its call fallthrough remains reachable, while the single
  // declared function range still proves complete instruction coverage.
  text_words[cursor++] = build_s_swappc_b64(/*sdst=*/30u, /*ssrc0=*/2u, kArch);
  std::copy(guest->begin(), guest->end(), text_words.begin() + cursor);
  cursor += guest->size();
  text_words[cursor++] = build_v_mov_b32_e32(/*vdst=*/0u, vector_source_vgpr(7u), kArch);
  // Make s72:s73 the only untouched persistent pair below the explicitly
  // reserved dispatch and transient windows.
  for (uint16_t sgpr = 0u; sgpr < 72u; ++sgpr)
    text_words[cursor++] = build_s_mov_b32(/*sdst=*/87u, sgpr, kArch);
  text_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/87u, kArch);
  text_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/90u, kArch);
  text_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/91u, kArch);
  text_words.back() = build_s_endpgm(kArch);

  std::vector<uint8_t> bytes =
      make_cdna4_lds_code_object(text_words, "unresolved_call_complete_text",
                                 /*vgpr_granulated=*/3u);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 12u);
  });

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.moi_init_owner_epoch = true;
  options.moi_exec_save_sgpr = 92u;
  options.moi_dispatch_id_sgpr = 88u;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_TRUE(result.moi_persistent_sgprs_automatic);
  ASSERT_TRUE(result.resolved_moi_persistent_owner_sgpr);
  ASSERT_TRUE(result.resolved_moi_persistent_epoch_sgpr);
  EXPECT_EQ(*result.resolved_moi_persistent_owner_sgpr, 72u);
  EXPECT_EQ(*result.resolved_moi_persistent_epoch_sgpr, 73u);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("using complete-text coverage") != std::string::npos;
  })) << testing::PrintToString(result.warnings);
}

TEST(ConSanMoi, Cdna4ScalarHoleAllowsVerifiedPaddingOutsideFunctionRanges) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  const auto guest = build_cdna4_ds_store_b32(/*vaddr=*/2, /*vdata=*/3, /*byte_offset=*/0, kArch);
  ASSERT_TRUE(guest);

  std::vector<uint32_t> kernel_words(320u, build_s_nop(0, kArch));
  size_t cursor = 0u;
  kernel_words[cursor++] = build_s_swappc_b64(/*sdst=*/30u, /*ssrc0=*/2u, kArch);
  std::copy(guest->begin(), guest->end(), kernel_words.begin() + cursor);
  cursor += guest->size();
  kernel_words[cursor++] = build_v_mov_b32_e32(/*vdst=*/0u, vector_source_vgpr(7u), kArch);
  for (uint16_t sgpr = 0u; sgpr < 72u; ++sgpr)
    kernel_words[cursor++] = build_s_mov_b32(/*sdst=*/87u, sgpr, kArch);
  kernel_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/87u, kArch);
  kernel_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/90u, kArch);
  kernel_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/91u, kArch);
  kernel_words.back() = build_s_endpgm(kArch);

  const std::array<uint32_t, 1> helper_words = {build_s_endpgm(kArch)};
  // Real CDNA objects align adjacent functions with a mixture of s_nop and
  // zero words that are intentionally absent from the function symbols.
  // Such padding cannot clobber scalar state and must not prevent a
  // whole-object reference proof.
  const std::array<uint32_t, 4> uncovered_padding = {
      build_s_nop(0, kArch),
      0u,
      build_s_nop(0, kArch),
      0u,
  };
  std::vector<uint8_t> bytes = make_cdna4_code_object_with_local_function(
      kernel_words, helper_words, uncovered_padding, /*vgpr_granulated=*/3u);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 12u);
  });

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.moi_init_owner_epoch = true;
  options.moi_exec_save_sgpr = 92u;
  options.moi_dispatch_id_sgpr = 88u;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_TRUE(result.moi_persistent_sgprs_automatic);
  ASSERT_TRUE(result.resolved_moi_persistent_owner_sgpr);
  ASSERT_TRUE(result.resolved_moi_persistent_epoch_sgpr);
  ASSERT_TRUE(result.resolved_moi_record_replay_workgroup_sgprs.complete());
  EXPECT_EQ(*result.resolved_moi_persistent_owner_sgpr, 72u);
  EXPECT_EQ(*result.resolved_moi_persistent_epoch_sgpr, 73u);
  EXPECT_EQ(*result.resolved_moi_record_replay_workgroup_sgprs.x, 74u);
  EXPECT_EQ(*result.resolved_moi_record_replay_workgroup_sgprs.y, 75u);
  EXPECT_EQ(*result.resolved_moi_record_replay_workgroup_sgprs.z, 76u);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("using complete-text coverage") != std::string::npos;
  })) << testing::PrintToString(result.warnings);
}

TEST(ConSanMoi, Cdna4FullPressureUsesProvenHybridPersistentRegisterHoles) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  const auto guest = build_cdna4_ds_store_b32(/*vaddr=*/2, /*vdata=*/3, /*byte_offset=*/0, kArch);
  ASSERT_TRUE(guest);

  std::vector<uint32_t> text_words(384u, build_s_nop(0, kArch));
  size_t cursor = 0u;
  text_words[cursor++] = build_s_swappc_b64(/*sdst=*/30u, /*ssrc0=*/2u, kArch);
  std::copy(guest->begin(), guest->end(), text_words.begin() + cursor);
  cursor += guest->size();
  // v8:v9 are the only unused ordinary pair below the v16 AccVGPR boundary.
  for (const uint16_t vgpr : {0u, 1u, 4u, 5u, 6u, 7u, 10u, 11u, 12u, 13u, 14u, 15u})
    text_words[cursor++] = build_v_mov_b32_e32(/*vdst=*/3u, vector_source_vgpr(vgpr), kArch);
  // s72:s74 are the only three consecutive unreferenced ordinary SGPRs after
  // excluding the dispatch, transient, dynamic-stack, and physical-VCC state.
  for (uint16_t sgpr = 0u; sgpr < 72u; ++sgpr)
    text_words[cursor++] = build_s_mov_b32(/*sdst=*/87u, sgpr, kArch);
  for (uint16_t sgpr = 75u; sgpr < 88u; ++sgpr)
    text_words[cursor++] = build_s_mov_b32(/*sdst=*/87u, sgpr, kArch);
  text_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/90u, kArch);
  text_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/91u, kArch);
  text_words.back() = build_s_endpgm(kArch);

  std::vector<uint8_t> bytes = make_cdna4_lds_code_object(text_words, "hybrid_persistent_holes",
                                                          /*vgpr_granulated=*/7u);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 3u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 12u);
  });

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.moi_init_owner_epoch = true;
  options.moi_exec_save_sgpr = 92u;
  options.moi_dispatch_id_sgpr = 88u;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << "warnings=" << testing::PrintToString(result.warnings)
                               << " plans=" << testing::PrintToString(result.resource_plans)
                               << " dispositions="
                               << testing::PrintToString(result.site_dispositions);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_TRUE(result.moi_persistent_vgprs_automatic);
  EXPECT_TRUE(result.moi_persistent_sgprs_automatic);
  EXPECT_FALSE(result.moi_private_epoch_automatic);
  ASSERT_TRUE(result.resolved_moi_owner_vgpr);
  ASSERT_TRUE(result.resolved_moi_epoch_vgpr);
  ASSERT_TRUE(result.resolved_moi_record_replay_workgroup_sgprs.complete());
  EXPECT_EQ(*result.resolved_moi_owner_vgpr, 8u);
  EXPECT_EQ(*result.resolved_moi_epoch_vgpr, 9u);
  EXPECT_EQ(*result.resolved_moi_record_replay_workgroup_sgprs.x, 72u);
  EXPECT_EQ(*result.resolved_moi_record_replay_workgroup_sgprs.y, 73u);
  EXPECT_EQ(*result.resolved_moi_record_replay_workgroup_sgprs.z, 74u);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("hybrid persistent vector owner/epoch") != std::string::npos &&
           warning.find("using complete-text coverage") != std::string::npos;
  })) << testing::PrintToString(result.warnings);

  const auto access = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore, &ConSanPatchInfo::kind);
  const auto prologue = std::ranges::find(
      result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue, &ConSanPatchInfo::kind);
  ASSERT_NE(access, result.patches.end());
  ASSERT_NE(prologue, result.patches.end());
  ASSERT_TRUE(access->scratch_vgpr);
  const auto plan = std::ranges::find_if(result.resource_plans, [](const auto &item) {
    return item.site_kind == ConSanResourceSiteKind::Access;
  });
  ASSERT_NE(plan, result.resource_plans.end());
  const uint32_t scratch_begin = *access->scratch_vgpr;
  const uint32_t scratch_end = scratch_begin + plan->scratch_vgpr_count;
  EXPECT_TRUE(*result.resolved_moi_epoch_vgpr < scratch_begin ||
              *result.resolved_moi_owner_vgpr >= scratch_end);
  EXPECT_EQ(prologue->persistent_owner_vgpr, result.resolved_moi_owner_vgpr);
  EXPECT_EQ(prologue->persistent_epoch_vgpr, result.resolved_moi_epoch_vgpr);

  ConSanOptions colliding_options = options;
  colliding_options.scratch_vgpr = 8u;
  colliding_options.force_vgpr_spill = true;
  const ConSanResult colliding = try_patch_consan(bytes, colliding_options);
  ASSERT_TRUE(consan_patch_succeeded(colliding)) << testing::PrintToString(colliding.errors);
  EXPECT_FALSE(colliding.modified);
  ASSERT_EQ(colliding.resource_plans.size(), 1u);
  EXPECT_EQ(colliding.resource_plans.front().source, ConSanRegisterAllocationSource::Unsupported);
  EXPECT_EQ(colliding.resource_plans.front().reason, ConSanRegisterPlanReason::ExplicitLive);
  EXPECT_FALSE(colliding.resolved_moi_owner_vgpr == 8u && colliding.resolved_moi_epoch_vgpr == 9u);
  EXPECT_TRUE(std::ranges::none_of(colliding.warnings, [](const std::string &warning) {
    return warning.find("hybrid persistent vector owner/epoch") != std::string::npos;
  })) << testing::PrintToString(colliding.warnings);
}

TEST(ConSanMoi, Cdna4RegisterSaturatedRecordReplayUsesPrivateDispatchIdentity) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  const auto access =
      build_cdna4_ds_store_b32(/*vaddr=*/2u, /*vdata=*/3u, /*byte_offset=*/0u, kArch);
  const auto barrier = build_cdna4_s_barrier(kArch);
  ASSERT_TRUE(access && barrier);

  const auto make_fixture = [&](bool scalar_pressure) {
    std::vector<uint32_t> words(768u, build_s_nop(0u, kArch));
    size_t cursor = 0u;
    for (size_t access_index = 0u; access_index < 5u; ++access_index) {
      std::ranges::copy(*access, words.begin() + static_cast<ptrdiff_t>(cursor));
      cursor += access->size();
      if (access_index < 3u)
        words[cursor++] = *barrier;
    }
    if (scalar_pressure) {
      // Mirror the issue reproducer's empty-asm clobber: the descriptor declares
      // all 106 scalar resources, preventing a persistent dispatch-ID pair, but
      // those registers are not simultaneously live at the LDS/barrier sites.
      // Site-local transient instrumentation state may therefore still use a
      // liveness-proven scalar window.
      words[cursor++] = build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/99u, kArch);
    }
    words.back() = build_s_endpgm(kArch);
    std::vector<uint8_t> bytes = make_cdna4_lds_code_object(
        words, scalar_pressure ? "gfx950_sgpr_pressure" : "gfx950_sgpr_control",
        /*vgpr_granulated=*/0u);
    mutate_first_kernel_descriptor(bytes, [&](KD &descriptor) {
      AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                      kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                      scalar_pressure ? 13u : 1u);
    });
    return bytes;
  };

  const ConSanMoiAutoReportPlan report_plan =
      plan_consan_moi_auto_report({.engine = ConSanMoiEngine::RecordReplay,
                                   .access_range_count = 5u,
                                   .barrier_event_count = 3u});
  ASSERT_TRUE(report_plan.complete());
  const auto layout = consan_moi_auto_report_layout_override(report_plan);
  ASSERT_TRUE(layout);

  const auto instrument = [&](const std::vector<uint8_t> &bytes) {
    ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
    options.max_patches = 8u;
    options.moi_track_barriers = true;
    options.moi_track_atomics = false;
    options.moi_report_buffer_address = 0x123456780000ull;
    options.moi_report_buffer_size = report_plan.required_bytes;
    options.moi_report_layout = *layout;
    return try_patch_consan(bytes, options);
  };

  const ConSanResult control = instrument(make_fixture(false));
  ASSERT_TRUE(consan_patch_succeeded(control)) << testing::PrintToString(control.errors);
  ASSERT_TRUE(control.modified) << testing::PrintToString(control.warnings);
  EXPECT_TRUE(control.resolved_moi_dispatch_id_sgpr);

  const ConSanResult pressure = instrument(make_fixture(true));
  ASSERT_TRUE(consan_patch_succeeded(pressure))
      << "warnings=" << testing::PrintToString(pressure.warnings)
      << " errors=" << testing::PrintToString(pressure.errors)
      << " plans=" << testing::PrintToString(pressure.resource_plans);
  for (const ConSanCandidateResourcePlan &plan : pressure.resource_plans) {
    EXPECT_NE(plan.source, ConSanRegisterAllocationSource::Unsupported)
        << "site=" << static_cast<unsigned>(plan.site_kind) << " offset=" << plan.text_offset
        << " reason=" << consan_register_plan_reason_name(plan.reason)
        << " scratch_count=" << plan.scratch_vgpr_count
        << " current_vgprs=" << plan.current_vgpr_count
        << " max_referenced_vgprs=" << plan.max_referenced_vgpr_count
        << " ordinary_limit=" << plan.ordinary_vgpr_limit;
  }
  ASSERT_TRUE(pressure.modified) << "warnings=" << testing::PrintToString(pressure.warnings)
                                 << " plans=" << testing::PrintToString(pressure.resource_plans)
                                 << " dispositions="
                                 << testing::PrintToString(pressure.site_dispositions);
  EXPECT_TRUE(pressure.final_validation_passed);
  EXPECT_FALSE(pressure.resolved_moi_dispatch_id_sgpr);
  EXPECT_TRUE(pressure.moi_private_epoch_automatic);
  EXPECT_EQ(std::ranges::count_if(pressure.site_dispositions,
                                  [](const auto &site) {
                                    return site.site_kind == ConSanResourceSiteKind::Access &&
                                           site.lowering_outcome ==
                                               ConSanSiteLoweringOutcome::Patched;
                                  }),
            5u);
  EXPECT_EQ(std::ranges::count_if(pressure.site_dispositions,
                                  [](const auto &site) {
                                    return site.site_kind == ConSanResourceSiteKind::Barrier &&
                                           site.lowering_outcome ==
                                               ConSanSiteLoweringOutcome::Patched;
                                  }),
            3u);
  EXPECT_TRUE(std::ranges::none_of(pressure.site_dispositions, [](const auto &site) {
    return site.resource_reason == ConSanRegisterPlanReason::NoLegalWindow;
  }));
  const auto private_prologue =
      std::ranges::find(pressure.patches, ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue,
                        &ConSanPatchInfo::kind);
  ASSERT_NE(private_prologue, pressure.patches.end());
  EXPECT_TRUE(private_prologue->persistent_dispatch_id_private_offset);
  EXPECT_TRUE(std::ranges::all_of(pressure.patches, [](const ConSanPatchInfo &patch) {
    if (patch.kind != ConSanPatchKind::InlineMoiAccessRecordStore &&
        patch.kind != ConSanPatchKind::TrampolineMoiAccessRecordStore &&
        patch.kind != ConSanPatchKind::TrampolineMoiBarrierRecord)
      return true;
    return patch.persistent_dispatch_id_private_offset.has_value();
  }));
}

TEST(ConSanMoi, Cdna4PrivateDispatchSpillKeepsScalarTupleBelowOrdinaryLimit) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  constexpr std::string_view kKernelName = "gfx950_private_dispatch_spill_pressure";
  const auto access =
      build_cdna4_ds_store_b32(/*vaddr=*/2u, /*vdata=*/3u, /*byte_offset=*/0u, kArch);
  ASSERT_TRUE(access);

  std::vector<uint32_t> words(768u, build_s_nop(0u, kArch));
  size_t cursor = 0u;
  std::ranges::copy(*access, words.begin() + static_cast<ptrdiff_t>(cursor));
  cursor += access->size();
  for (uint16_t vgpr = 0u; vgpr < 8u; ++vgpr)
    words[cursor++] = build_v_mov_b32_e32(/*vdst=*/0u, vector_source_vgpr(vgpr), kArch);
  for (uint16_t sgpr = 0u; sgpr < 100u; ++sgpr)
    words[cursor++] = build_s_mov_b32(/*sdst=*/99u, sgpr, kArch);
  words.back() = build_s_endpgm(kArch);

  std::vector<uint8_t> bytes =
      make_cdna4_lds_code_object(words, kKernelName, /*vgpr_granulated=*/0u);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    // v8 is the first AccVGPR and s0:s99 are the compiler's complete ordinary
    // scalar tail. The descriptor's 104-register scalar allocation leaves its
    // reserved CDNA tail above that point, but does not make s102+ addressable.
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 12u);
  });
  append_kernel_metadata_note(bytes, kKernelName, /*uses_dynamic_stack=*/false,
                              /*sgpr_count=*/100u,
                              /*private_segment_fixed_size=*/std::nullopt,
                              /*required_workgroup_size=*/std::nullopt,
                              /*has_dynamic_lds=*/false,
                              /*additional_kernel_names=*/{}, /*agpr_count=*/1u,
                              /*vgpr_count=*/8u);

  const ConSanMoiAutoReportPlan report_plan = plan_consan_moi_auto_report(
      {.engine = ConSanMoiEngine::RecordReplay, .access_range_count = 1u});
  ASSERT_TRUE(report_plan.complete());
  const auto layout = consan_moi_auto_report_layout_override(report_plan);
  ASSERT_TRUE(layout);

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.force_vgpr_spill = true;
  options.max_patches = 1u;
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = report_plan.required_bytes;
  options.moi_report_layout = *layout;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result))
      << "warnings=" << testing::PrintToString(result.warnings)
      << " errors=" << testing::PrintToString(result.errors)
      << " plans=" << testing::PrintToString(result.resource_plans);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_FALSE(result.resolved_moi_dispatch_id_sgpr);
  EXPECT_FALSE(result.resolved_moi_persistent_owner_sgpr);
  EXPECT_FALSE(result.resolved_moi_persistent_epoch_sgpr);
  EXPECT_TRUE(result.resolved_moi_record_replay_workgroup_sgprs.empty());
  EXPECT_TRUE(result.moi_private_epoch_automatic);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().source, ConSanRegisterAllocationSource::SpillRequired);
  EXPECT_GT(result.resource_plan_summary.emitted_spill_patches, 0u);
  EXPECT_GT(result.resource_plan_summary.emitted_spill_slot_bytes, 0u);

  const auto access_patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::InlineMoiAccessRecordStore ||
           patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore;
  });
  ASSERT_NE(access_patch, result.patches.end());
  EXPECT_TRUE(access_patch->persistent_dispatch_id_private_offset);
  EXPECT_TRUE(access_patch->persistent_record_replay_workgroup_private_offsets.complete());
  EXPECT_GT(access_patch->spilled_vgpr_count, 0u);
}

TEST(ConSanMoi, CdnaRecordReplayMovesOnlyEmptyAccumulatorBoundaryForDynamicStackState) {
  for (const rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    for (const uint8_t agpr_count : {0u, 1u}) {
      SCOPED_TRACE(arch);
      SCOPED_TRACE(agpr_count);
      const auto store = arch == ROCJITSU_CODE_ARCH_CDNA3
                             ? build_cdna3_ds_store_b32(/*vaddr=*/2u, /*vdata=*/3u,
                                                        /*byte_offset=*/0u, arch)
                             : build_cdna4_ds_store_b32(/*vaddr=*/2u, /*vdata=*/3u,
                                                        /*byte_offset=*/0u, arch);
      const auto barrier = arch == ROCJITSU_CODE_ARCH_CDNA3 ? build_cdna3_s_barrier(arch)
                                                            : build_cdna4_s_barrier(arch);
      ASSERT_TRUE(store && barrier);

      constexpr std::string_view kKernelName = "record_replay_dynamic_stack_empty_accumulator";
      std::vector<uint32_t> text_words(1200u, build_s_nop(0u, arch));
      text_words[0] = build_v_mov_b32_e32(/*vdst=*/0u, vector_source_vgpr(125u), arch);
      std::ranges::copy(*store, text_words.begin() + 1u);
      text_words[3] = *barrier;
      size_t cursor = 4u;
      // Leave no persistent pair below the encoded ordinary-VGPR boundary.
      for (uint16_t vgpr = 0u; vgpr < 126u; ++vgpr)
        text_words[cursor++] = build_v_mov_b32_e32(/*vdst=*/0u, vector_source_vgpr(vgpr), arch);
      // The only scalar holes are the explicitly reserved transient dispatch
      // and EXEC-save windows. They cannot also carry state that survives guest
      // code between entry and every Record/Replay probe.
      for (uint16_t sgpr = 0u; sgpr < 104u; ++sgpr) {
        if ((sgpr >= 88u && sgpr < 90u) || (sgpr >= 92u && sgpr < 97u))
          continue;
        text_words[cursor++] = build_s_mov_b32(/*sdst=*/87u, sgpr, arch);
      }
      text_words.back() = build_s_endpgm(arch);
      std::vector<uint8_t> bytes =
          arch == ROCJITSU_CODE_ARCH_CDNA3
              ? make_cdna3_lds_code_object(text_words, kKernelName,
                                           /*vgpr_granulated=*/agpr_count == 0u ? 15u : 16u,
                                           /*uses_dynamic_stack=*/true,
                                           /*workgroup_id_dimension_mask=*/7u,
                                           /*group_segment_fixed_size=*/4u)
              : make_cdna4_lds_code_object(text_words, kKernelName,
                                           /*vgpr_granulated=*/agpr_count == 0u ? 15u : 16u,
                                           /*uses_dynamic_stack=*/true,
                                           /*workgroup_id_dimension_mask=*/7u,
                                           /*group_segment_fixed_size=*/4u);
      mutate_first_kernel_descriptor(bytes, [agpr_count](KD &descriptor) {
        // The guest reaches v125 in a 128-VGPR allocation. ACCUM_OFFSET names
        // v128, but trusted metadata proves that no accumulator is allocated.
        // Record/Replay's owner/epoch/exact-workgroup tuple must move that empty
        // boundary rather than rejecting a dynamic-stack owner.
        AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET,
                        31u);
        AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                        kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                        agpr_count == 0u ? 15u : 16u);
        AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                        kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 13u);
      });
      append_kernel_metadata_note(bytes, kKernelName, /*uses_dynamic_stack=*/true,
                                  /*sgpr_count=*/104u, /*private_segment_fixed_size=*/48u,
                                  /*required_workgroup_size=*/std::nullopt,
                                  /*has_dynamic_lds=*/false,
                                  /*additional_kernel_names=*/{}, agpr_count,
                                  /*vgpr_count=*/126u);

      ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
      options.moi_track_barriers = true;
      options.moi_track_atomics = false;
      options.moi_init_owner_epoch = true;
      options.moi_dispatch_id_sgpr = 88u;
      options.moi_exec_save_sgpr = 92u;
      options.moi_report_buffer_address = 0x123456780000ull;
      options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 1);

      const ConSanResult result = try_patch_consan(bytes, options);

      if (agpr_count != 0u) {
        EXPECT_FALSE(result.modified);
        EXPECT_EQ(result.outcome, ConSanTransformOutcome::Unsupported);
        EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
          return warning.find("connected CDNA AccVGPR boundary") != std::string::npos;
        })) << testing::PrintToString(result.warnings);
        continue;
      }

      ASSERT_TRUE(consan_patch_succeeded(result))
          << "warnings=" << testing::PrintToString(result.warnings)
          << " errors=" << testing::PrintToString(result.errors)
          << " plans=" << testing::PrintToString(result.resource_plans);
      ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
      EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
      EXPECT_TRUE(result.final_validation_passed);
      EXPECT_TRUE(result.moi_persistent_vgprs_automatic);
      EXPECT_FALSE(result.moi_persistent_sgprs_automatic);
      EXPECT_FALSE(result.moi_private_epoch_automatic);
      ASSERT_EQ(result.resolved_moi_persistent_vgpr_assignments.size(), 1u);
      const ConSanMoiPersistentVgprAssignment &assignment =
          result.resolved_moi_persistent_vgpr_assignments.front();
      EXPECT_EQ(assignment.owner_vgpr, 126u);
      EXPECT_EQ(assignment.epoch_vgpr, 127u);
      ASSERT_TRUE(assignment.record_replay_workgroup_vgprs.complete());
      EXPECT_EQ(*assignment.record_replay_workgroup_vgprs.x, 128u);
      EXPECT_EQ(*assignment.record_replay_workgroup_vgprs.y, 129u);
      EXPECT_EQ(*assignment.record_replay_workgroup_vgprs.z, 130u);

      AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
      ASSERT_TRUE(patched.is_valid());
      KD descriptor{};
      std::memcpy(&descriptor,
                  result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
                  sizeof(descriptor));
      EXPECT_GT(
          AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET),
          31u);
      EXPECT_EQ(std::ranges::count_if(result.site_dispositions,
                                      [](const auto &site) {
                                        return site.disposition ==
                                                   ConSanSiteDisposition::Supported &&
                                               site.lowering_outcome ==
                                                   ConSanSiteLoweringOutcome::Patched;
                                      }),
                2u);
    }
  }
}

TEST(ConSanMoi, Cdna4RecordReplayMixesPrivateAndDynamicStackPersistentStateByOwner) {
  const std::vector<uint8_t> bytes = make_cdna4_mixed_accvgpr_persistence_code_object();
  ASSERT_FALSE(bytes.empty());

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_track_barriers = true;
  options.moi_track_atomics = false;
  options.moi_init_owner_epoch = true;
  options.max_patches = 2u;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(2, 0, 0, 0, 2);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result))
      << "warnings=" << testing::PrintToString(result.warnings)
      << " errors=" << testing::PrintToString(result.errors)
      << " plans=" << testing::PrintToString(result.resource_plans);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_TRUE(result.moi_private_epoch_automatic);
  EXPECT_TRUE(result.moi_persistent_vgprs_automatic);

  const auto fixed = std::ranges::find(result.kernels, "lds_probe", &ConSanKernelInfo::name);
  const auto dynamic = std::ranges::find(result.kernels, "lds_helper", &ConSanKernelInfo::name);
  ASSERT_NE(fixed, result.kernels.end());
  ASSERT_NE(dynamic, result.kernels.end());
  ASSERT_TRUE(fixed->uses_dynamic_stack);
  ASSERT_TRUE(dynamic->uses_dynamic_stack);
  EXPECT_FALSE(*fixed->uses_dynamic_stack);
  EXPECT_TRUE(*dynamic->uses_dynamic_stack);
  ASSERT_EQ(result.resolved_moi_persistent_vgpr_assignments.size(), 1u);
  const ConSanMoiPersistentVgprAssignment &dynamic_assignment =
      result.resolved_moi_persistent_vgpr_assignments.front();
  EXPECT_EQ(dynamic_assignment.descriptor_file_offset, dynamic->descriptor_file_offset);
  EXPECT_TRUE(dynamic_assignment.record_replay_workgroup_vgprs.complete());
  for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
    EXPECT_NE(plan.source, ConSanRegisterAllocationSource::Unsupported)
        << "site=" << static_cast<uint32_t>(plan.site_kind) << " offset=" << plan.text_offset
        << " reason=" << consan_register_plan_reason_name(plan.reason)
        << " owners=" << testing::PrintToString(plan.owner_descriptor_file_offsets);
  }

  const auto private_prologue = std::ranges::find(
      result.patches, ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue, &ConSanPatchInfo::kind);
  const auto register_prologue = std::ranges::find(
      result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue, &ConSanPatchInfo::kind);
  ASSERT_NE(private_prologue, result.patches.end());
  ASSERT_NE(register_prologue, result.patches.end())
      << "warnings=" << testing::PrintToString(result.warnings)
      << " patches=" << testing::PrintToString(result.patches)
      << " dispositions=" << testing::PrintToString(result.site_dispositions);
  EXPECT_EQ(private_prologue->owner_descriptor_file_offsets,
            std::vector<uint64_t>{fixed->descriptor_file_offset});
  EXPECT_EQ(register_prologue->owner_descriptor_file_offsets,
            std::vector<uint64_t>{dynamic->descriptor_file_offset});
  EXPECT_TRUE(private_prologue->persistent_record_replay_workgroup_private_offsets.complete());
  ASSERT_EQ(std::ranges::count(result.site_dispositions, ConSanSiteDisposition::Supported,
                               &ConSanSiteDispositionRecord::disposition),
            4u);
  for (const ConSanSiteDispositionRecord &site : result.site_dispositions) {
    if (site.disposition != ConSanSiteDisposition::Supported)
      continue;
    EXPECT_EQ(site.lowering_outcome, ConSanSiteLoweringOutcome::Patched)
        << "kind=" << static_cast<uint32_t>(site.site_kind) << " container=" << site.container_name
        << " offset=" << site.text_offset << " mnemonic=" << site.mnemonic
        << " outcome=" << consan_site_lowering_outcome_name(site.lowering_outcome)
        << " reason=" << consan_site_lowering_reason_name(site.lowering_reason)
        << " warnings=" << testing::PrintToString(result.warnings)
        << " plans=" << testing::PrintToString(result.resource_plans);
  }
}

TEST(ConSanMoi, Cdna4RecordReplayRecoversDynamicOwnerAfterPrivateAbiResourceFailure) {
  const std::vector<uint8_t> bytes = make_cdna4_mixed_accvgpr_persistence_code_object(
      /*full_sampled_pressure=*/false, /*dynamic_flat_group_load=*/false,
      /*dynamic_has_accvgpr_bank=*/false,
      /*second_fixed_stack_full_scalar_pressure=*/false,
      /*dynamic_ordered_atomic=*/false,
      /*record_replay_recovery_pressure=*/true);
  ASSERT_FALSE(bytes.empty());

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_track_barriers = true;
  options.moi_track_atomics = false;
  options.moi_init_owner_epoch = true;
  options.max_patches = 2u;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(2, 0, 0, 0, 2);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result))
      << "warnings=" << testing::PrintToString(result.warnings)
      << " errors=" << testing::PrintToString(result.errors)
      << " plans=" << testing::PrintToString(result.resource_plans);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_TRUE(result.moi_private_epoch_automatic);
  EXPECT_TRUE(result.moi_persistent_vgprs_automatic);

  const auto dynamic = std::ranges::find(result.kernels, "lds_helper", &ConSanKernelInfo::name);
  ASSERT_NE(dynamic, result.kernels.end());
  ASSERT_EQ(result.resolved_moi_persistent_vgpr_assignments.size(), 1u);
  EXPECT_EQ(result.resolved_moi_persistent_vgpr_assignments.front().descriptor_file_offset,
            dynamic->descriptor_file_offset);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("deferred dynamic-stack Record/Replay state") != std::string::npos;
  })) << testing::PrintToString(result.warnings);

  ASSERT_EQ(result.resource_plans.size(), 4u);
  for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
    EXPECT_NE(plan.source, ConSanRegisterAllocationSource::Unsupported)
        << "site=" << static_cast<uint32_t>(plan.site_kind) << " offset=" << plan.text_offset
        << " reason=" << consan_register_plan_reason_name(plan.reason)
        << " scratch_count=" << plan.scratch_vgpr_count
        << " owners=" << testing::PrintToString(plan.owner_descriptor_file_offsets);
  }
  for (const ConSanSiteDispositionRecord &site : result.site_dispositions) {
    ASSERT_EQ(site.disposition, ConSanSiteDisposition::Supported);
    EXPECT_EQ(site.lowering_outcome, ConSanSiteLoweringOutcome::Patched)
        << "kind=" << static_cast<uint32_t>(site.site_kind) << " container=" << site.container_name
        << " offset=" << site.text_offset << " mnemonic=" << site.mnemonic
        << " reason=" << consan_site_lowering_reason_name(site.lowering_reason)
        << " warnings=" << testing::PrintToString(result.warnings);
  }
}

TEST(ConSanMoi, Cdna4RecordReplayCarriesMixedPersistentStateAcrossAtomicAndFenceRecords) {
  const std::vector<uint8_t> bytes = make_cdna4_mixed_accvgpr_persistence_code_object(
      /*full_sampled_pressure=*/false, /*dynamic_flat_group_load=*/false,
      /*dynamic_has_accvgpr_bank=*/false,
      /*second_fixed_stack_full_scalar_pressure=*/false,
      /*dynamic_ordered_atomic=*/true);
  ASSERT_FALSE(bytes.empty());

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_dynamic_access_records = true;
  options.moi_track_barriers = false;
  options.moi_track_atomics = true;
  options.moi_init_owner_epoch = true;
  options.max_patches = 8u;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(8, 0, 0, 0, 0, 8, 8);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result))
      << "warnings=" << testing::PrintToString(result.warnings)
      << " errors=" << testing::PrintToString(result.errors)
      << " plans=" << testing::PrintToString(result.resource_plans);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_TRUE(result.moi_private_epoch_automatic);
  EXPECT_TRUE(result.moi_persistent_vgprs_automatic);

  const auto fixed = std::ranges::find(result.kernels, "lds_probe", &ConSanKernelInfo::name);
  const auto dynamic = std::ranges::find(result.kernels, "lds_helper", &ConSanKernelInfo::name);
  ASSERT_NE(fixed, result.kernels.end());
  ASSERT_NE(dynamic, result.kernels.end());
  ASSERT_EQ(result.resolved_moi_persistent_vgpr_assignments.size(), 1u);
  EXPECT_EQ(result.resolved_moi_persistent_vgpr_assignments.front().descriptor_file_offset,
            dynamic->descriptor_file_offset);

  for (const ConSanPatchKind kind :
       {ConSanPatchKind::TrampolineMoiAtomicRecord, ConSanPatchKind::TrampolineMoiFenceRecord}) {
    SCOPED_TRACE(testing::PrintToString(kind));
    const auto patch = std::ranges::find(result.patches, kind, &ConSanPatchInfo::kind);
    ASSERT_NE(patch, result.patches.end()) << testing::PrintToString(result.warnings);
    EXPECT_EQ(patch->owner_descriptor_file_offsets,
              std::vector<uint64_t>{dynamic->descriptor_file_offset});
    EXPECT_FALSE(patch->persistent_epoch_private_offset);
    EXPECT_FALSE(patch->persistent_owner_private_offset);
    EXPECT_FALSE(patch->persistent_record_replay_workgroup_private_offsets.complete());
  }
  for (const ConSanSiteDispositionRecord &site : result.site_dispositions) {
    if (site.site_kind != ConSanResourceSiteKind::Atomic &&
        site.site_kind != ConSanResourceSiteKind::Fence)
      continue;
    ASSERT_EQ(site.disposition, ConSanSiteDisposition::Supported)
        << "container=" << site.container_name << " offset=" << site.text_offset
        << " mnemonic=" << site.mnemonic;
    EXPECT_EQ(site.lowering_outcome, ConSanSiteLoweringOutcome::Patched)
        << "container=" << site.container_name << " offset=" << site.text_offset
        << " mnemonic=" << site.mnemonic
        << " reason=" << consan_site_lowering_reason_name(site.lowering_reason)
        << " warnings=" << testing::PrintToString(result.warnings);
  }
}

TEST(ConSanMoi, Cdna4RecordReplayRoutesWithoutDeadTransientScalarRegisters) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  const auto guest =
      build_cdna4_ds_store_b32(/*vaddr=*/0u, /*vdata=*/0u, /*byte_offset=*/0u, kArch);
  ASSERT_TRUE(guest);

  // Keep every allocated ordinary scalar register live across the access.
  // This matches compiler-generated full-pressure MFMA kernels: a spill-backed
  // instrumentation window exists, but there is no liveness-dead PC/SCC pair
  // with which to reach an appended body.
  std::vector<uint32_t> text_words(1200u, build_s_nop(0u, kArch));
  std::copy(guest->begin(), guest->end(), text_words.begin() + 1u);
  size_t cursor = 1u + guest->size();
  for (uint16_t sgpr = 0u; sgpr < 102u; ++sgpr)
    text_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, sgpr, kArch);
  text_words.back() = build_s_endpgm(kArch);

  constexpr uint32_t kCdna4Wave64AllVgprsGranulated = 31u;
  std::vector<uint8_t> bytes = make_cdna4_lds_code_object(
      text_words, "record_replay_no_dead_transient_sgprs", kCdna4Wave64AllVgprsGranulated);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 13u);
  });

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.force_vgpr_spill = true;
  options.moi_dispatch_id_sgpr = 88u;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1u, 0u, 0u, 0u);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.max_patches = 1u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.resolved_moi_transient_sgpr_assignments.size(), 1u);
  const ConSanMoiTransientSgprAssignment &assignment =
      result.resolved_moi_transient_sgpr_assignments.front();
  EXPECT_TRUE(assignment.spill_backed);
  EXPECT_TRUE(assignment.branch_only_scalar_spill);
  EXPECT_TRUE(assignment.indirect_pc_sgpr);
  EXPECT_TRUE(assignment.indirect_scc_sgpr);
  EXPECT_FALSE(assignment.dispatch_id_sgpr);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               &ConSanPatchInfo::kind),
            1u);
}

TEST(ConSanMoi, Gfx1250RecordReplayRoutesWithoutDeadTransientScalarRegisters) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA5;

  // Keep every allocatable ordinary scalar register live across the access.
  // PyTorch TopK has independent fixed-stack owners with this shape: private
  // spill storage is available, but no dead PC/SCC/key tuple exists for an
  // indirect route to the appended Record/Replay body.
  std::vector<uint32_t> text_words(1200u, build_s_nop(0u, kArch));
  text_words[1] = 0xD8340000u;
  text_words[2] = 0x00000000u; // ds_store_b32 v0, v0
  size_t cursor = 3u;
  for (uint16_t sgpr = 0u; sgpr < 102u; ++sgpr)
    text_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, sgpr, kArch);
  text_words.back() = build_s_endpgm(kArch);

  std::vector<uint8_t> bytes = make_gfx1250_code_object(
      text_words, "gfx1250_record_replay_no_dead_transient_sgprs", kRdna4Wave64AllVgprsGranulated);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 13u);
  });

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.force_vgpr_spill = true;
  options.moi_dispatch_id_sgpr = 88u;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1u, 0u, 0u, 0u);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.max_patches = 1u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.resolved_moi_transient_sgpr_assignments.size(), 1u);
  const ConSanMoiTransientSgprAssignment &assignment =
      result.resolved_moi_transient_sgpr_assignments.front();
  EXPECT_TRUE(assignment.spill_backed);
  EXPECT_TRUE(assignment.branch_only_scalar_spill);
  EXPECT_TRUE(assignment.indirect_pc_sgpr);
  EXPECT_TRUE(assignment.indirect_scc_sgpr);
  EXPECT_FALSE(assignment.dispatch_id_sgpr);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               &ConSanPatchInfo::kind),
            1u);
}

TEST(ConSanMoi, Cdna4RecordReplayRoutesFarAccessWithoutDeadTransientScalarRegisters) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  constexpr uint32_t kAccessCount = 33u;
  const auto guest =
      build_cdna4_ds_store_b32(/*vaddr=*/0u, /*vdata=*/0u, /*byte_offset=*/0u, kArch);
  ASSERT_TRUE(guest);

  // Keep the body beyond direct SOPP reach while retaining abundant pristine
  // NOP capacity between the anchor and the appended relay prefix. This is
  // the shape of large generated libraries whose full-pressure kernels must
  // borrow and spill transient scalar state before executing a probe.
  std::vector<uint32_t> text_words(40'000u, build_s_nop(0u, kArch));
  size_t cursor = 1u;
  for (uint32_t access = 0u; access < kAccessCount; ++access) {
    std::copy(guest->begin(), guest->end(), text_words.begin() + cursor);
    cursor += guest->size();
  }
  for (uint16_t sgpr = 0u; sgpr < 102u; ++sgpr)
    text_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, sgpr, kArch);
  text_words.back() = build_s_endpgm(kArch);

  constexpr uint32_t kCdna4Wave64AllVgprsGranulated = 31u;
  std::vector<uint8_t> bytes = make_cdna4_lds_code_object(
      text_words, "record_replay_far_no_dead_transient_sgprs", kCdna4Wave64AllVgprsGranulated);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 13u);
  });

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.force_vgpr_spill = true;
  options.moi_dispatch_id_sgpr = 88u;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(kAccessCount, 0u, 0u, 0u);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.max_patches = kAccessCount;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.resolved_moi_transient_sgpr_assignments.size(), 1u);
  EXPECT_TRUE(result.resolved_moi_transient_sgpr_assignments.front().branch_only_scalar_spill);
  const auto access_count = std::ranges::count(
      result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore, &ConSanPatchInfo::kind);
  ASSERT_EQ(access_count, kAccessCount) << testing::PrintToString(result.warnings);
  size_t borrowed_entry_count = 0u;
  for (const ConSanPatchInfo &access : result.patches) {
    if (access.kind != ConSanPatchKind::TrampolineMoiAccessRecordStore)
      continue;
    EXPECT_FALSE(compute_sopp_branch_simm16(access.anchor_offset, access.trampoline_offset));
    EXPECT_TRUE(access.branch_only_continuation);
    if (access.branch_only_borrowed_indirect_entry) {
      ++borrowed_entry_count;
      EXPECT_TRUE(access.branch_only_entry_relay_offsets.empty());
    } else {
      EXPECT_FALSE(access.branch_only_entry_relay_offsets.empty());
    }
    EXPECT_FALSE(access.branch_only_return_relay_offsets.empty());
  }
  // The densely adjacent prefix cannot expand without consuming another
  // access. Its final site is followed by a relocatable scalar continuation
  // and therefore safely halves only that site's far-route demand.
  EXPECT_EQ(borrowed_entry_count, 1u);
  EXPECT_GT(std::ranges::count(result.patches, ConSanPatchKind::TrampolineBranchRelayReservoir,
                               &ConSanPatchInfo::kind),
            0u);
  EXPECT_GT(result.moi_branch_only_reservoir_telemetry.used_reservoir_count, 0u);
  EXPECT_EQ(result.moi_branch_only_placement_failure_count, 0u);
  // A many-site object must qualify and index its shared relay inventory once,
  // rather than rebuilding it for every far access.
  EXPECT_EQ(result.moi_branch_only_routing_telemetry.pair_attempt_count, kAccessCount);
  EXPECT_EQ(result.moi_branch_only_routing_telemetry.plan_call_count, 1u);
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, Cdna4RecordReplayRoutesFarAccessThroughSelectedAnchorTails) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  constexpr size_t kTextWords = 40'000u;
  constexpr std::array<size_t, 4> kAccessWords = {1u, 12'000u, 24'000u, 36'000u};
  const auto guest =
      build_cdna4_ds_store_b32(/*vaddr=*/0u, /*vdata=*/0u, /*byte_offset=*/0u, kArch);
  ASSERT_TRUE(guest);
  ASSERT_GT(guest->size(), 1u);

  // This object deliberately has neither NOP caves nor 16-word relocatable
  // donor runs. The last access can reach appended text directly; after it is
  // selected, the dead tail of its multiword instruction must extend the
  // frontier to the preceding access, and so on. This isolates the reusable
  // multiword-anchor contract without depending on a particular E2E layout.
  std::vector<uint32_t> text_words(kTextWords, build_s_branch(/*simm16=*/0, kArch));
  for (size_t access_word : kAccessWords) {
    ASSERT_LE(access_word + guest->size(), text_words.size());
    std::copy(guest->begin(), guest->end(), text_words.begin() + access_word);
  }
  size_t cursor = kTextWords - 256u;
  for (uint16_t sgpr = 0u; sgpr < 102u; ++sgpr) {
    text_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, sgpr, kArch);
    text_words[cursor++] = build_s_branch(/*simm16=*/0, kArch);
  }
  text_words.back() = build_s_endpgm(kArch);

  constexpr uint32_t kCdna4Wave64AllVgprsGranulated = 31u;
  std::vector<uint8_t> bytes = make_cdna4_lds_code_object(
      text_words, "record_replay_far_selected_anchor_tails", kCdna4Wave64AllVgprsGranulated);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 13u);
  });

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.force_vgpr_spill = true;
  options.moi_dispatch_id_sgpr = 88u;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size =
      consan_moi_report_buffer_min_bytes(kAccessWords.size(), 0u, 0u, 0u);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.max_patches = kAccessWords.size();

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.resolved_moi_transient_sgpr_assignments.size(), 1u);
  EXPECT_TRUE(result.resolved_moi_transient_sgpr_assignments.front().branch_only_scalar_spill);
  const auto access_count = std::ranges::count(
      result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore, &ConSanPatchInfo::kind);
  ASSERT_EQ(access_count, kAccessWords.size()) << testing::PrintToString(result.warnings);
  EXPECT_EQ(result.moi_branch_only_reservoir_telemetry.planned_reservoir_count, 0u);
  EXPECT_EQ(result.moi_branch_only_routing_telemetry.pair_attempt_count, kAccessWords.size());
  EXPECT_EQ(result.moi_branch_only_routing_telemetry.plan_call_count, 1u);
  EXPECT_EQ(result.moi_branch_only_placement_failure_count, 0u);

  std::set<uint64_t> anchor_tail_offsets;
  for (size_t access_word : kAccessWords) {
    for (size_t word = 1u; word < guest->size(); ++word)
      anchor_tail_offsets.insert((access_word + word) * sizeof(uint32_t));
  }
  EXPECT_TRUE(std::ranges::any_of(result.patches, [&](const ConSanPatchInfo &patch) {
    if (patch.kind != ConSanPatchKind::TrampolineMoiAccessRecordStore)
      return false;
    const auto uses_anchor_tail = [&](uint64_t relay) {
      return anchor_tail_offsets.contains(relay);
    };
    return std::ranges::any_of(patch.branch_only_entry_relay_offsets, uses_anchor_tail) ||
           std::ranges::any_of(patch.branch_only_return_relay_offsets, uses_anchor_tail);
  }));
  EXPECT_TRUE(std::ranges::any_of(result.patches, [&](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore &&
           std::ranges::any_of(patch.branch_relay_offsets,
                               [&](uint64_t relay) { return anchor_tail_offsets.contains(relay); });
  }));
  EXPECT_TRUE(result.final_validation_passed);

  ConSanResult stale_provenance = result;
  const auto relay_owner = std::ranges::find_if(stale_provenance.patches, [](const auto &patch) {
    return !patch.branch_relay_offsets.empty();
  });
  ASSERT_NE(relay_owner, stale_provenance.patches.end());
  relay_owner->branch_relay_offsets.front() = std::numeric_limits<uint64_t>::max() - 3u;
  const std::vector<std::string> validation_errors =
      validate_consan_modified_elf(bytes, stale_provenance);
  EXPECT_TRUE(std::ranges::any_of(validation_errors, [](const std::string &error) {
    return error.find("stale branch-relay provenance") != std::string::npos;
  })) << testing::PrintToString(validation_errors);
}

TEST(ConSanMoi, Cdna4RecordReplayRecursivelyRoutesInstructionReservoirs) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  constexpr size_t kTextWords = 100'000u;
  constexpr std::array<size_t, 3> kDonorWords = {10'000u, 40'000u, 70'000u};
  constexpr size_t kDonorWordCount = 16u;
  const auto guest =
      build_cdna4_ds_store_b32(/*vaddr=*/0u, /*vdata=*/0u, /*byte_offset=*/0u, kArch);
  ASSERT_TRUE(guest);

  // One-word branch blocks provide no incidental caves. Three explicit
  // ordinary-instruction runs are each less than one SOPP hop apart, but only
  // the last can directly reach appended text. The two earlier reservoirs
  // must route through already-materialized later reservoirs, and selecting
  // the earliest tail must retain that complete dependency chain.
  std::vector<uint32_t> text_words(kTextWords, build_s_branch(/*simm16=*/0, kArch));
  std::copy(guest->begin(), guest->end(), text_words.begin() + 1u);
  const uint32_t donor = build_v_mov_b32_e32(/*vdst=*/31u, vector_source_vgpr(31u), kArch);
  for (size_t donor_word : kDonorWords)
    std::fill_n(text_words.begin() + donor_word, kDonorWordCount, donor);
  size_t cursor = kTextWords - 128u;
  for (uint16_t sgpr = 0u; sgpr < 102u; ++sgpr)
    text_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, sgpr, kArch);
  text_words.back() = build_s_endpgm(kArch);

  constexpr uint32_t kCdna4Wave64AllVgprsGranulated = 31u;
  std::vector<uint8_t> bytes = make_cdna4_lds_code_object(
      text_words, "record_replay_recursive_instruction_reservoirs", kCdna4Wave64AllVgprsGranulated);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 13u);
  });

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.force_vgpr_spill = true;
  options.moi_dispatch_id_sgpr = 88u;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1u, 0u, 0u, 0u);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.max_patches = 1u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.resolved_moi_transient_sgpr_assignments.size(), 1u);
  EXPECT_TRUE(result.resolved_moi_transient_sgpr_assignments.front().branch_only_scalar_spill);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               &ConSanPatchInfo::kind),
            1u);
  const auto routed_reservoir_count =
      std::ranges::count_if(result.patches, [](const ConSanPatchInfo &patch) {
        return patch.kind == ConSanPatchKind::TrampolineBranchRelayReservoir &&
               patch.original_size != 0u && patch.branch_only_continuation;
      });
  EXPECT_GE(routed_reservoir_count, 2u);
  EXPECT_GE(result.moi_branch_only_reservoir_telemetry.used_reservoir_count, kDonorWords.size());
  EXPECT_EQ(result.moi_branch_only_placement_failure_count, 0u);
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, Cdna4RecordReplayBarrierConsumesReservoirOmittedByAccessSelection) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  constexpr size_t kTextWords = 100'000u;
  constexpr size_t kAccessWord = 45'000u;
  constexpr size_t kBarrierWord = 1u;
  constexpr std::array<size_t, 3> kDonorWords = {10'000u, 40'000u, 70'000u};
  constexpr size_t kDonorWordCount = 16u;
  const auto branch_only_access =
      build_cdna4_ds_store_b32(/*vaddr=*/0u, /*vdata=*/0u, /*byte_offset=*/0u, kArch);
  const auto ordinary_access =
      build_cdna4_ds_store_b32(/*vaddr=*/2u, /*vdata=*/3u, /*byte_offset=*/0u, kArch);
  const auto barrier = build_cdna4_s_barrier(kArch);
  ASSERT_TRUE(branch_only_access && ordinary_access && barrier);

  // The full-pressure kernel contributes a barrier more than one SOPP hop
  // before its branch-only access, but the one-access expert limit selects the
  // ordinary access in the other kernel first. Reservoir discovery must use
  // the barrier as its earliest downstream frontier and preserve that route
  // even though no selected access claims it. This is the reduced contract
  // for the TopK object in which 400 reservoirs were planned only as far back
  // as the omitted accesses, zero were access-used, and 4,680 earlier barriers
  // consequently lost every original-text foothold.
  std::vector<uint32_t> branch_only_words(kTextWords, build_s_branch(/*simm16=*/0, kArch));
  std::copy(branch_only_access->begin(), branch_only_access->end(),
            branch_only_words.begin() + kAccessWord);
  branch_only_words[kBarrierWord] = *barrier;
  const uint32_t donor = build_v_mov_b32_e32(/*vdst=*/31u, vector_source_vgpr(31u), kArch);
  for (size_t donor_word : kDonorWords)
    std::fill_n(branch_only_words.begin() + donor_word, kDonorWordCount, donor);
  size_t cursor = kTextWords - 128u;
  for (uint16_t sgpr = 0u; sgpr < 102u; ++sgpr)
    branch_only_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, sgpr, kArch);
  branch_only_words.back() = build_s_endpgm(kArch);

  std::vector<uint32_t> ordinary_words(320u, build_s_nop(0u, kArch));
  std::copy(ordinary_access->begin(), ordinary_access->end(), ordinary_words.begin() + 1u);
  ordinary_words.back() = build_s_endpgm(kArch);

  constexpr uint32_t kCdna4Wave64AllVgprsGranulated = 31u;
  std::vector<uint8_t> bytes = make_cdna4_code_object_with_local_function(
      ordinary_words, branch_only_words, {}, kCdna4Wave64AllVgprsGranulated,
      /*function_is_kernel=*/true);
  mutate_kernel_descriptor(bytes, "lds_probe", [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 0u);
  });
  mutate_kernel_descriptor(bytes, "lds_helper", [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 13u);
  });
  AmdGpuCodeObject original(bytes.data(), bytes.size());
  ASSERT_TRUE(original.is_valid());
  const auto branch_only_kernel =
      std::ranges::find(original.kernels(), "lds_helper", &AmdGpuKernelInfo::name);
  ASSERT_NE(branch_only_kernel, original.kernels().end());
  const uint64_t branch_only_access_offset =
      branch_only_kernel->entry_text_offset + kAccessWord * sizeof(uint32_t);
  const uint64_t barrier_offset =
      branch_only_kernel->entry_text_offset + kBarrierWord * sizeof(uint32_t);

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.force_vgpr_spill = true;
  options.moi_dispatch_id_sgpr = 88u;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(2u, 0u, 0u, 0u, 1u);
  options.moi_track_barriers = true;
  options.moi_track_atomics = false;
  options.max_patches = 1u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               &ConSanPatchInfo::kind),
            1u);
  EXPECT_TRUE(std::ranges::none_of(result.patches, [&](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore &&
           patch.anchor_offset == branch_only_access_offset;
  }));
  ASSERT_GT(result.moi_branch_only_reservoir_telemetry.planned_reservoir_count, 0u);
  EXPECT_GT(result.moi_branch_only_reservoir_telemetry.used_reservoir_count, 0u);
  EXPECT_GT(result.moi_branch_only_reservoir_telemetry.unused_reservoir_count, 0u);
  const auto barrier_patch = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord, &ConSanPatchInfo::kind);
  ASSERT_NE(barrier_patch, result.patches.end()) << testing::PrintToString(result.warnings);
  EXPECT_EQ(barrier_patch->anchor_offset, barrier_offset);
  EXPECT_TRUE(barrier_patch->branch_only_continuation);
  EXPECT_FALSE(barrier_patch->branch_only_entry_relay_offsets.empty());
  EXPECT_FALSE(barrier_patch->branch_only_return_relay_offsets.empty());
  const auto relay_uses_direct_reservoir = [&](uint64_t relay) {
    return std::ranges::any_of(result.patches, [&](const ConSanPatchInfo &patch) {
      return patch.kind == ConSanPatchKind::TrampolineBranchRelayReservoir &&
             patch.original_size >= 2u * sizeof(uint32_t) && relay > patch.anchor_offset &&
             relay < patch.anchor_offset + patch.original_size;
    });
  };
  EXPECT_TRUE(std::ranges::any_of(barrier_patch->branch_only_entry_relay_offsets,
                                  relay_uses_direct_reservoir) ||
              std::ranges::any_of(barrier_patch->branch_only_return_relay_offsets,
                                  relay_uses_direct_reservoir));
  EXPECT_EQ(result.moi_branch_only_placement_failure_count, 0u);
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, Cdna4RecordReplayBarrierReturnRoutesThroughSelectedAccessAnchorTails) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  constexpr size_t kAccessKernelWords = 40'000u;
  constexpr std::array<size_t, 10> kAccessWords = {1u,      3u,      12'000u, 12'002u, 12'004u,
                                                   24'000u, 24'002u, 24'004u, 36'000u, 36'002u};
  const auto access =
      build_cdna4_ds_store_b32(/*vaddr=*/2u, /*vdata=*/3u, /*byte_offset=*/0u, kArch);
  const auto barrier = build_cdna4_s_barrier(kArch);
  ASSERT_TRUE(access && barrier);

  // Both independent owners have full scalar pressure. Clustered multiword
  // accesses less than one SOPP hop apart across the second kernel route
  // through their own selected dead anchor tails. Two middle tails are
  // consumed by access routing while their adjacent peer remains available.
  // The first owner's barrier must import that remaining paid frontier as its
  // only capacity-one original-text round trip to the appended body: no direct
  // instruction reservoir is planned.
  std::vector<uint32_t> barrier_words(320u, build_s_nop(0u, kArch));
  barrier_words[1] = *barrier;
  size_t scalar_cursor = 32u;
  for (uint16_t sgpr = 0u; sgpr < 102u; ++sgpr)
    barrier_words[scalar_cursor++] = build_s_mov_b32(/*sdst=*/0u, sgpr, kArch);
  barrier_words.back() = build_s_endpgm(kArch);

  std::vector<uint32_t> access_words(kAccessKernelWords, build_s_branch(/*simm16=*/0, kArch));
  for (size_t access_word : kAccessWords)
    std::copy(access->begin(), access->end(), access_words.begin() + access_word);
  size_t access_scalar_cursor = kAccessKernelWords - 256u;
  for (uint16_t sgpr = 0u; sgpr < 102u; ++sgpr) {
    access_words[access_scalar_cursor++] = build_s_mov_b32(/*sdst=*/0u, sgpr, kArch);
    access_words[access_scalar_cursor++] = build_s_branch(/*simm16=*/0, kArch);
  }
  access_words.back() = build_s_endpgm(kArch);

  constexpr uint32_t kCdna4Wave64AllVgprsGranulated = 31u;
  std::vector<uint8_t> bytes = make_cdna4_code_object_with_local_function(
      barrier_words, access_words, {}, kCdna4Wave64AllVgprsGranulated,
      /*function_is_kernel=*/true);
  mutate_kernel_descriptor(bytes, "lds_probe", [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 13u);
  });
  mutate_kernel_descriptor(bytes, "lds_helper", [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 13u);
  });

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.force_vgpr_spill = true;
  options.moi_dispatch_id_sgpr = 88u;
  options.moi_report_buffer_address = 0x123456780000ull;
  // Leave alignment headroom so the barrier partition cannot reduce the ten
  // exact access-record slots in this synthetic mixed-owner object.
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(32u, 0u, 0u, 0u, 1u);
  options.moi_track_barriers = true;
  options.moi_track_atomics = false;
  options.max_patches = kAccessWords.size();

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               &ConSanPatchInfo::kind),
            kAccessWords.size());
  EXPECT_EQ(result.moi_branch_only_reservoir_telemetry.planned_reservoir_count, 0u);
  size_t published_access_tail_count = 0u;
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore)
      published_access_tail_count += patch.branch_relay_offsets.size();
  }
  EXPECT_EQ(published_access_tail_count, kAccessWords.size());
  testing::Message access_tail_details;
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore) {
      access_tail_details << "access anchor=" << patch.anchor_offset
                          << " body=" << patch.trampoline_offset
                          << " original_size=" << patch.original_size << " tails=";
      for (uint64_t offset : patch.branch_relay_offsets)
        access_tail_details << offset << ",";
      access_tail_details << "\n";
    }
  }
  const auto barrier_patch = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord, &ConSanPatchInfo::kind);
  ASSERT_NE(barrier_patch, result.patches.end())
      << access_tail_details << testing::PrintToString(result.warnings);
  EXPECT_TRUE(barrier_patch->branch_only_continuation);
  std::set<uint64_t> access_anchor_tails;
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore)
      access_anchor_tails.insert(patch.branch_relay_offsets.begin(),
                                 patch.branch_relay_offsets.end());
  }
  const auto is_access_anchor_tail = [&](uint64_t relay) {
    return access_anchor_tails.contains(relay);
  };
  EXPECT_TRUE(barrier_patch->branch_only_borrowed_indirect_entry);
  EXPECT_TRUE(barrier_patch->branch_only_entry_relay_offsets.empty());
  EXPECT_TRUE(
      std::ranges::any_of(barrier_patch->branch_only_return_relay_offsets, is_access_anchor_tail));
  EXPECT_EQ(result.moi_branch_only_placement_failure_count, 0u);
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, Cdna4RecordReplayRelaySpineCrossesBarrierPrefixBeyondSoppReach) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  constexpr size_t kTextWords = 100'000u;
  constexpr std::array<size_t, 3> kDonorWords = {10'000u, 40'000u, 70'000u};
  constexpr size_t kDonorWordCount = 16u;
  constexpr size_t kBarrierBeginWord = 20'000u;
  constexpr size_t kBarrierIslandWords = 7u;
  constexpr size_t kBarrierRelaySlotWords = kBarrierIslandWords + 2u;
  constexpr size_t kBarrierCount =
      kSoppBranchMaximumForwardReachBytes / (kBarrierIslandWords * sizeof(uint32_t)) + 1u;
  constexpr std::array<size_t, 3> kIndependentBarrierWords = {9'000u, 39'000u, 69'000u};
  constexpr size_t kReservedBarrierCount = kBarrierCount + kIndependentBarrierWords.size();
  constexpr size_t kRoutedBarrierCount = 1u + kIndependentBarrierWords.size();
  static_assert(kBarrierCount * kBarrierIslandWords * sizeof(uint32_t) >
                kSoppBranchMaximumForwardReachBytes);
  static_assert(kBarrierBeginWord + kBarrierCount < kTextWords);
  const auto guest =
      build_cdna4_ds_store_b32(/*vaddr=*/0u, /*vdata=*/0u, /*byte_offset=*/0u, kArch);
  const auto barrier = build_cdna4_s_barrier(kArch);
  ASSERT_TRUE(guest && barrier);

  // This is the recursive-reservoir contract with an independent barrier
  // inventory whose reserved island prefix is itself longer than one SOPP
  // hop. The seven-word barrier islands remain pristine; their dedicated
  // eighth and ninth words must bridge pristine text to the generated access
  // bank, while unused tails in the already-materialized direct reservoirs
  // provide the first and last footholds from the original barrier.
  std::vector<uint32_t> text_words(kTextWords, build_s_branch(/*simm16=*/0, kArch));
  std::copy(guest->begin(), guest->end(), text_words.begin() + 1u);
  const uint32_t donor = build_v_mov_b32_e32(/*vdst=*/31u, vector_source_vgpr(31u), kArch);
  for (size_t donor_word : kDonorWords)
    std::fill_n(text_words.begin() + donor_word, kDonorWordCount, donor);
  // The adjacent inventory reserves a generated prefix longer than one SOPP
  // hop and canonicalizes to one routed barrier. Additional independent sites
  // near the three materialized donor runs prove that the later barrier pass
  // routes all of its real demands in one batch without turning this contract
  // into a relay-capacity test for an unrealistic access-free object.
  std::fill_n(text_words.begin() + kBarrierBeginWord, kBarrierCount, *barrier);
  for (size_t barrier_word : kIndependentBarrierWords)
    text_words[barrier_word] = *barrier;
  size_t cursor = kTextWords - 128u;
  for (uint16_t sgpr = 0u; sgpr < 102u; ++sgpr)
    text_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, sgpr, kArch);
  text_words.back() = build_s_endpgm(kArch);

  constexpr uint32_t kCdna4Wave64AllVgprsGranulated = 31u;
  std::vector<uint8_t> bytes = make_cdna4_lds_code_object(
      text_words, "record_replay_barrier_prefix_relay_spine", kCdna4Wave64AllVgprsGranulated);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 13u);
  });
  AmdGpuCodeObject original(bytes.data(), bytes.size());
  ASSERT_TRUE(original.is_valid());
  ASSERT_EQ(original.text_sections().size(), 1u);
  const uint64_t original_text_size = original.text_sections().front()->size();

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.force_vgpr_spill = true;
  options.moi_dispatch_id_sgpr = 88u;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size =
      consan_moi_report_buffer_min_bytes(1u, 0u, 0u, 0u, kReservedBarrierCount);
  options.moi_track_barriers = true;
  options.moi_track_atomics = false;
  options.max_patches = kReservedBarrierCount + 1u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.resolved_moi_transient_sgpr_assignments.size(), 1u);
  EXPECT_TRUE(result.resolved_moi_transient_sgpr_assignments.front().branch_only_scalar_spill);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               &ConSanPatchInfo::kind),
            1u);
  const size_t routed_barrier_count = std::ranges::count(
      result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord, &ConSanPatchInfo::kind);
  ASSERT_EQ(routed_barrier_count, kRoutedBarrierCount) << testing::PrintToString(result.warnings);
  const auto routed_barrier = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord, &ConSanPatchInfo::kind);
  ASSERT_NE(routed_barrier, result.patches.end()) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(routed_barrier->branch_only_continuation);
  EXPECT_FALSE(routed_barrier->branch_only_entry_relay_offsets.empty());
  EXPECT_FALSE(routed_barrier->branch_only_return_relay_offsets.empty());
  const auto is_donor_tail = [&](uint64_t offset) {
    return std::ranges::any_of(kDonorWords, [&](size_t donor_word) {
      const uint64_t begin = donor_word * sizeof(uint32_t) + sizeof(uint32_t);
      const uint64_t end = (donor_word + kDonorWordCount) * sizeof(uint32_t);
      return offset >= begin && offset < end;
    });
  };
  EXPECT_TRUE(std::ranges::any_of(routed_barrier->branch_only_entry_relay_offsets, is_donor_tail));
  EXPECT_TRUE(std::ranges::any_of(routed_barrier->branch_only_return_relay_offsets, is_donor_tail));
  EXPECT_EQ(result.moi_branch_only_placement_failure_count, 0u);
  // The access pass and the barrier pass each route their entire inventory in
  // one batch. A per-barrier plan call would make large E2E objects scale with
  // the product of barrier count and relay-index construction cost.
  EXPECT_EQ(result.moi_branch_only_routing_telemetry.pair_attempt_count, kRoutedBarrierCount + 1u);
  EXPECT_EQ(result.moi_branch_only_routing_telemetry.plan_call_count, 2u);
  EXPECT_EQ(result.moi_branch_only_routing_telemetry.work_budget_failure_count, 0u);
  EXPECT_GT(result.moi_branch_only_reservoir_telemetry.used_reservoir_count, 0u);
  EXPECT_TRUE(std::ranges::any_of(result.patches, [&](const ConSanPatchInfo &patch) {
    if (patch.kind != ConSanPatchKind::TrampolineBranchRelayReservoir ||
        patch.original_size != 0u || patch.trampoline_offset < original_text_size) {
      return false;
    }
    const uint64_t relative = patch.trampoline_offset - original_text_size;
    const uint64_t slot_offset = relative % (kBarrierRelaySlotWords * sizeof(uint32_t));
    return slot_offset >= kBarrierIslandWords * sizeof(uint32_t);
  })) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, Cdna4RejectedRecordReplayBarrierDoesNotMutateItsOwnerDescriptor) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  constexpr size_t kUnreachableKernelWords =
      kSoppBranchMaximumForwardReachBytes / sizeof(uint32_t) + 128u;
  const auto barrier = build_cdna4_s_barrier(kArch);
  const auto access =
      build_cdna4_ds_store_b32(/*vaddr=*/0u, /*vdata=*/1u, /*byte_offset=*/0u, kArch);
  ASSERT_TRUE(barrier && access);

  // The first kernel has a resource-admitted barrier but no relay foothold
  // before the second kernel, which begins beyond one SOPP hop.  The nearby
  // access in the second kernel is independently patchable and keeps the
  // overall transform live.  Rejecting the first barrier must not leave its
  // descriptor with speculative VGPR, SGPR, or private-segment growth.
  std::vector<uint32_t> unreachable_kernel(kUnreachableKernelWords,
                                           build_s_branch(/*simm16=*/0, kArch));
  unreachable_kernel.front() = *barrier;
  unreachable_kernel.back() = build_s_endpgm(kArch);
  std::vector<uint32_t> patchable_kernel(320u, build_s_nop(0u, kArch));
  std::copy(access->begin(), access->end(), patchable_kernel.begin() + 1u);
  patchable_kernel.back() = build_s_endpgm(kArch);

  constexpr uint32_t kCdna4Wave64AllVgprsGranulated = 31u;
  const std::vector<uint8_t> bytes = make_cdna4_code_object_with_local_function(
      unreachable_kernel, patchable_kernel, {}, kCdna4Wave64AllVgprsGranulated,
      /*function_is_kernel=*/true);
  AmdGpuCodeObject original(bytes.data(), bytes.size());
  ASSERT_TRUE(original.is_valid());
  const auto original_rejected =
      std::ranges::find(original.kernels(), "lds_probe", &AmdGpuKernelInfo::name);
  ASSERT_NE(original_rejected, original.kernels().end());
  KD original_rejected_descriptor{};
  std::memcpy(&original_rejected_descriptor,
              bytes.data() + original_rejected->descriptor_file_offset,
              sizeof(original_rejected_descriptor));

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.force_vgpr_spill = true;
  options.moi_dispatch_id_sgpr = 88u;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(8u, 0u, 0u, 0u, 8u);
  options.moi_track_barriers = true;
  options.moi_track_atomics = false;
  options.max_patches = 8u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               &ConSanPatchInfo::kind),
            1u);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord,
                               &ConSanPatchInfo::kind),
            0u);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("no reachable indirect entry island") != std::string::npos;
  }));

  AmdGpuCodeObject replacement(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(replacement.is_valid());
  const auto replacement_rejected =
      std::ranges::find(replacement.kernels(), "lds_probe", &AmdGpuKernelInfo::name);
  ASSERT_NE(replacement_rejected, replacement.kernels().end());
  KD replacement_rejected_descriptor{};
  std::memcpy(&replacement_rejected_descriptor,
              result.elf_bytes.data() + replacement_rejected->descriptor_file_offset,
              sizeof(replacement_rejected_descriptor));
  EXPECT_EQ(std::memcmp(&replacement_rejected_descriptor, &original_rejected_descriptor,
                        sizeof(original_rejected_descriptor)),
            0);
}

TEST(ConSanMoi, Cdna4RecordReplayRejectsFarAccessWithoutFirstHopBeforeReservingBody) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  constexpr size_t kTextWords = 40'000u;
  const auto guest =
      build_cdna4_ds_store_b32(/*vaddr=*/0u, /*vdata=*/0u, /*byte_offset=*/0u, kArch);
  ASSERT_TRUE(guest);

  // One-word branch blocks are neither pristine NOPs nor admissible relocation
  // donors. With the access and generated bank more than one SOPP hop apart,
  // routing is intrinsically impossible and must be rejected before reserving
  // a large unreachable appended body.
  std::vector<uint32_t> text_words(kTextWords, build_s_branch(/*simm16=*/0, kArch));
  std::copy(guest->begin(), guest->end(), text_words.begin() + 1u);
  size_t cursor = kTextWords - 128u;
  for (uint16_t sgpr = 0u; sgpr < 102u; ++sgpr)
    text_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, sgpr, kArch);
  text_words.back() = build_s_endpgm(kArch);

  constexpr uint32_t kCdna4Wave64AllVgprsGranulated = 31u;
  std::vector<uint8_t> bytes = make_cdna4_lds_code_object(
      text_words, "record_replay_far_without_first_hop", kCdna4Wave64AllVgprsGranulated);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 13u);
  });

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.force_vgpr_spill = true;
  options.moi_dispatch_id_sgpr = 88u;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1u, 0u, 0u, 0u);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.max_patches = 1u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  EXPECT_FALSE(result.modified);
  EXPECT_EQ(result.moi_branch_only_placement_failure_count, 1u);
  EXPECT_EQ(result.moi_branch_only_routing_telemetry.plan_call_count, 0u);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               &ConSanPatchInfo::kind),
            0u);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("has no first-hop relay") != std::string::npos;
  })) << testing::PrintToString(result.warnings);
}

TEST(ConSanMoi, Cdna4RecordReplayBorrowsEntryTupleToHalveFarRelayDemand) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  constexpr size_t kTextWords = 100'000u;
  const auto guest =
      build_cdna4_ds_store_b32(/*vaddr=*/0u, /*vdata=*/0u, /*byte_offset=*/0u, kArch);
  ASSERT_TRUE(guest);

  // Sparse movable runs connect the access to appended text. The access is
  // followed by a separate relocatable owner-local run, so the expanded anchor
  // can preserve a borrowed PC/SCC tuple in its dead scratch VGPR, enter
  // indirectly, and consume the corridor only on return. This directly checks
  // the reduced-demand contract from physical gfx950 TopK: one routed path per
  // site, rather than an entry/return pair.
  std::vector<uint32_t> text_words(kTextWords, build_s_branch(/*simm16=*/0, kArch));
  std::copy(guest->begin(), guest->end(), text_words.begin() + 1u);
  const uint32_t movable = build_v_mov_b32_e32(/*vdst=*/31u, vector_source_vgpr(31u), kArch);
  std::fill_n(text_words.begin() + 1u + guest->size(), 24u, movable);
  for (size_t reservoir_word : {20'000u, 40'000u, 60'000u, 80'000u, 95'000u})
    std::fill_n(text_words.begin() + reservoir_word, 16u, movable);
  size_t cursor = kTextWords - 128u;
  for (uint16_t sgpr = 0u; sgpr < 102u; ++sgpr)
    text_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, sgpr, kArch);
  text_words.back() = build_s_endpgm(kArch);

  constexpr uint32_t kCdna4Wave64AllVgprsGranulated = 31u;
  std::vector<uint8_t> bytes = make_cdna4_lds_code_object(
      text_words, "record_replay_borrowed_entry_tuple", kCdna4Wave64AllVgprsGranulated);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 13u);
  });

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.force_vgpr_spill = true;
  options.moi_dispatch_id_sgpr = 88u;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1u, 0u, 0u, 0u);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.max_patches = 1u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.resolved_moi_transient_sgpr_assignments.size(), 1u);
  const ConSanMoiTransientSgprAssignment &assignment =
      result.resolved_moi_transient_sgpr_assignments.front();
  EXPECT_TRUE(assignment.branch_only_scalar_spill);
  EXPECT_TRUE(assignment.indirect_pc_sgpr);
  EXPECT_TRUE(assignment.indirect_scc_sgpr);
  ASSERT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               &ConSanPatchInfo::kind),
            1u)
      << testing::PrintToString(result.warnings);
  const auto access = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore, &ConSanPatchInfo::kind);
  ASSERT_NE(access, result.patches.end());
  EXPECT_GT(access->original_size, guest->size() * sizeof(uint32_t));
  EXPECT_TRUE(access->branch_only_entry_relay_offsets.empty());
  EXPECT_FALSE(access->branch_only_return_relay_offsets.empty());
  EXPECT_TRUE(access->branch_only_borrowed_backup_vgpr);
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, Cdna4RecordReplayDoesNotBorrowAcrossReconvergenceEntry) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  constexpr size_t kTextWords = 100'000u;
  constexpr size_t kAccessWord = 1u;
  constexpr size_t kReconvergenceWord = 12u;
  const auto guest =
      build_cdna4_ds_store_b32(/*vaddr=*/0u, /*vdata=*/0u, /*byte_offset=*/0u, kArch);
  const auto bypass =
      instrumentation::build_s_cbranch_execz(static_cast<int16_t>(kReconvergenceWord - 1u), kArch);
  ASSERT_TRUE(guest && bypass);

  // The fallthrough reaches the access, while an empty wave enters a later
  // instruction directly. Relocating a 20-word entry window here would leave
  // that original branch targeting the middle of the synthetic island. This
  // is the compact host analogue of the TopK-derived device regression: the
  // access itself is relocatable, but the expanded window is not single-entry.
  std::vector<uint32_t> text_words(kTextWords, build_s_branch(/*simm16=*/0, kArch));
  text_words.front() = *bypass;
  std::copy(guest->begin(), guest->end(), text_words.begin() + kAccessWord);
  const uint32_t movable = build_v_mov_b32_e32(/*vdst=*/31u, vector_source_vgpr(31u), kArch);
  std::fill_n(text_words.begin() + kAccessWord + guest->size(), 24u, movable);
  for (size_t reservoir_word : {20'000u, 40'000u, 60'000u, 80'000u, 95'000u})
    std::fill_n(text_words.begin() + reservoir_word, 16u, movable);
  size_t cursor = kTextWords - 128u;
  for (uint16_t sgpr = 0u; sgpr < 102u; ++sgpr)
    text_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, sgpr, kArch);
  text_words.back() = build_s_endpgm(kArch);

  constexpr uint32_t kCdna4Wave64AllVgprsGranulated = 31u;
  std::vector<uint8_t> bytes = make_cdna4_lds_code_object(
      text_words, "record_replay_borrowed_entry_reconvergence", kCdna4Wave64AllVgprsGranulated);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 13u);
  });

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.force_vgpr_spill = true;
  options.moi_dispatch_id_sgpr = 88u;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1u, 0u, 0u, 0u);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.max_patches = 1u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto access = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore, &ConSanPatchInfo::kind);
  ASSERT_NE(access, result.patches.end()) << testing::PrintToString(result.warnings);
  EXPECT_EQ(access->original_size, guest->size() * sizeof(uint32_t));
  EXPECT_FALSE(access->branch_only_borrowed_indirect_entry);
  EXPECT_FALSE(access->branch_only_borrowed_backup_vgpr);
  EXPECT_FALSE(access->branch_only_entry_relay_offsets.empty());
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, Cdna4RecordReplayBarrierBorrowsEntryTupleBeforeScalarProbe) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  constexpr size_t kBarrierWord = 300u;
  const auto barrier = build_cdna4_s_barrier(kArch);
  ASSERT_TRUE(barrier);

  std::vector<uint32_t> text_words(1200u, build_s_nop(0u, kArch));
  text_words[kBarrierWord] = *barrier;
  const uint32_t movable = build_v_mov_b32_e32(/*vdst=*/31u, vector_source_vgpr(31u), kArch);
  std::fill_n(text_words.begin() + kBarrierWord + 1u, 24u, movable);
  size_t cursor = text_words.size() - 128u;
  for (uint16_t sgpr = 0u; sgpr < 102u; ++sgpr)
    text_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, sgpr, kArch);
  text_words.back() = build_s_endpgm(kArch);

  constexpr uint32_t kCdna4Wave64AllVgprsGranulated = 31u;
  std::vector<uint8_t> bytes = make_cdna4_lds_code_object(
      text_words, "record_replay_barrier_borrowed_entry_tuple", kCdna4Wave64AllVgprsGranulated);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 13u);
  });

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.force_vgpr_spill = true;
  options.moi_dispatch_id_sgpr = 88u;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1u, 0u, 0u, 0u, 1u);
  options.moi_track_barriers = true;
  options.moi_track_atomics = false;
  options.max_patches = 1u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto patch = std::ranges::find(result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord,
                                       &ConSanPatchInfo::kind);
  ASSERT_NE(patch, result.patches.end()) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(patch->branch_only_continuation);
  EXPECT_TRUE(patch->branch_only_borrowed_indirect_entry);
  EXPECT_GT(patch->original_size, sizeof(uint32_t));
  EXPECT_TRUE(patch->branch_only_entry_relay_offsets.empty());
  EXPECT_TRUE(patch->branch_only_borrowed_backup_vgpr);
  EXPECT_FALSE(patch->branch_only_borrowed_continuation_offset);
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, Cdna4RecordReplayBarrierDoesNotBorrowAcrossFollowingFence) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  constexpr size_t kBarrierWord = 300u;
  const auto barrier = build_cdna4_s_barrier(kArch);
  const auto fence = build_cdna4_s_dcache_inv_vol(kArch);
  ASSERT_TRUE(barrier && fence);

  std::vector<uint32_t> text_words(1200u, build_s_nop(0u, kArch));
  text_words[kBarrierWord] = *barrier;
  std::ranges::copy(*fence, text_words.begin() + kBarrierWord + 1u);
  const uint32_t movable = build_v_mov_b32_e32(/*vdst=*/31u, vector_source_vgpr(31u), kArch);
  std::fill_n(text_words.begin() + kBarrierWord + 1u + fence->size(), 24u, movable);
  size_t cursor = text_words.size() - 128u;
  for (uint16_t sgpr = 0u; sgpr < 102u; ++sgpr)
    text_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, sgpr, kArch);
  text_words.back() = build_s_endpgm(kArch);

  constexpr uint32_t kCdna4Wave64AllVgprsGranulated = 31u;
  std::vector<uint8_t> bytes = make_cdna4_lds_code_object(
      text_words, "record_replay_barrier_followed_by_fence", kCdna4Wave64AllVgprsGranulated);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 13u);
  });

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.force_vgpr_spill = true;
  options.moi_dispatch_id_sgpr = 88u;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size =
      consan_moi_report_buffer_min_bytes(32u, 0u, 0u, 0u, 32u, 32u, 32u);
  options.moi_track_barriers = true;
  options.moi_track_atomics = true;
  options.max_patches = 1u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto barrier_patch = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord, &ConSanPatchInfo::kind);
  ASSERT_NE(barrier_patch, result.patches.end()) << testing::PrintToString(result.warnings);
  EXPECT_EQ(barrier_patch->original_size, sizeof(uint32_t));
  EXPECT_FALSE(barrier_patch->branch_only_borrowed_indirect_entry);
  EXPECT_FALSE(result.moi_fence_candidates.empty());
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, Cdna4RecordReplayBarrierDoesNotBorrowAcrossReconvergenceEntry) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  constexpr size_t kBarrierWord = 300u;
  constexpr size_t kReconvergenceWord = kBarrierWord + 8u;
  const auto barrier = build_cdna4_s_barrier(kArch);
  const auto bypass = instrumentation::build_s_cbranch_execz(
      static_cast<int16_t>(kReconvergenceWord - kBarrierWord), kArch);
  ASSERT_TRUE(barrier && bypass);

  std::vector<uint32_t> text_words(1200u, build_s_nop(0u, kArch));
  text_words[kBarrierWord - 1u] = *bypass;
  text_words[kBarrierWord] = *barrier;
  const uint32_t movable = build_v_mov_b32_e32(/*vdst=*/31u, vector_source_vgpr(31u), kArch);
  std::fill_n(text_words.begin() + kBarrierWord + 1u, 24u, movable);
  size_t cursor = text_words.size() - 128u;
  for (uint16_t sgpr = 0u; sgpr < 102u; ++sgpr)
    text_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, sgpr, kArch);
  text_words.back() = build_s_endpgm(kArch);

  constexpr uint32_t kCdna4Wave64AllVgprsGranulated = 31u;
  std::vector<uint8_t> bytes = make_cdna4_lds_code_object(
      text_words, "record_replay_barrier_reconvergence", kCdna4Wave64AllVgprsGranulated);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 13u);
  });

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.force_vgpr_spill = true;
  options.moi_dispatch_id_sgpr = 88u;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1u, 0u, 0u, 0u, 1u);
  options.moi_track_barriers = true;
  options.moi_track_atomics = false;
  options.max_patches = 1u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto patch = std::ranges::find(result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord,
                                       &ConSanPatchInfo::kind);
  ASSERT_NE(patch, result.patches.end()) << testing::PrintToString(result.warnings);
  EXPECT_EQ(patch->original_size, sizeof(uint32_t));
  EXPECT_FALSE(patch->branch_only_borrowed_indirect_entry);
  EXPECT_FALSE(patch->branch_only_borrowed_backup_vgpr);
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, Cdna4RecordReplayRoutesBarrierWithoutDeadTransientScalarRegisters) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  constexpr uint32_t kAccessCount = 39u;
  constexpr size_t kBarrierWord = 1024u;
  constexpr size_t kTextWords = 11'200u;
  const auto guest =
      build_cdna4_ds_store_b32(/*vaddr=*/0u, /*vdata=*/0u, /*byte_offset=*/0u, kArch);
  const auto barrier = build_cdna4_s_barrier(kArch);
  ASSERT_TRUE(guest && barrier);

  std::vector<uint32_t> text_words(kTextWords, build_s_nop(0u, kArch));
  size_t cursor = 0u;
  for (uint32_t access = 0u; access < kAccessCount; ++access) {
    std::copy(guest->begin(), guest->end(), text_words.begin() + cursor);
    cursor += guest->size();
  }
  ASSERT_LT(cursor, kBarrierWord);
  text_words[kBarrierWord] = *barrier;
  cursor = kBarrierWord + 1u;
  // Keep every allocated ordinary scalar register live across both the
  // accesses and barrier. The access bodies push the barrier cave beyond a
  // direct SOPP branch, while their pre-reserved local island remains close.
  for (uint16_t sgpr = 0u; sgpr < 102u; ++sgpr)
    text_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, sgpr, kArch);
  text_words.back() = build_s_endpgm(kArch);

  constexpr uint32_t kCdna4Wave64AllVgprsGranulated = 31u;
  std::vector<uint8_t> bytes = make_cdna4_lds_code_object(
      text_words, "record_replay_no_dead_transient_barrier", kCdna4Wave64AllVgprsGranulated);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 13u);
  });

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.force_vgpr_spill = true;
  options.moi_dispatch_id_sgpr = 88u;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(kAccessCount, 0u, 0u, 0u, 1u);
  options.moi_track_barriers = true;
  options.moi_track_atomics = false;
  options.max_patches = kAccessCount;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.resolved_moi_transient_sgpr_assignments.size(), 1u);
  const ConSanMoiTransientSgprAssignment &assignment =
      result.resolved_moi_transient_sgpr_assignments.front();
  EXPECT_TRUE(assignment.spill_backed);
  EXPECT_TRUE(assignment.branch_only_scalar_spill);
  EXPECT_TRUE(assignment.indirect_pc_sgpr);
  EXPECT_TRUE(assignment.indirect_scc_sgpr);

  const auto barrier_patch = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord, &ConSanPatchInfo::kind);
  ASSERT_NE(barrier_patch, result.patches.end()) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(barrier_patch->branch_only_continuation);
  EXPECT_TRUE(barrier_patch->branch_only_borrowed_indirect_entry);
  EXPECT_TRUE(barrier_patch->branch_only_entry_relay_offsets.empty());
  EXPECT_EQ(result.moi_branch_only_placement_failure_count, 0u);
  EXPECT_GT(result.moi_branch_only_routing_telemetry.plan_call_count, 0u);
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, Cdna4ScalarHoleUsesCompleteTextCoverageForUnresolvedBranch) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  const auto guest = build_cdna4_ds_store_b32(/*vaddr=*/2, /*vdata=*/3, /*byte_offset=*/0, kArch);
  ASSERT_TRUE(guest);

  std::vector<uint32_t> text_words(320u, build_s_nop(0, kArch));
  size_t cursor = 0u;
  std::copy(guest->begin(), guest->end(), text_words.begin() + cursor);
  cursor += guest->size();
  text_words[cursor++] = build_v_mov_b32_e32(/*vdst=*/0u, vector_source_vgpr(7u), kArch);
  for (uint16_t sgpr = 0u; sgpr < 72u; ++sgpr)
    text_words[cursor++] = build_s_mov_b32(/*sdst=*/87u, sgpr, kArch);
  text_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/87u, kArch);
  text_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/90u, kArch);
  text_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/91u, kArch);
  // This target cannot be recovered and has no fallthrough. The LDS site is
  // still owned, but owner-local references are not a closed execution scope.
  text_words[cursor++] = build_s_setpc_b64(/*ssrc0=*/2u, kArch);
  text_words.back() = build_s_endpgm(kArch);

  std::vector<uint8_t> bytes =
      make_cdna4_lds_code_object(text_words, "unresolved_branch_complete_text",
                                 /*vgpr_granulated=*/3u);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 12u);
  });

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.moi_init_owner_epoch = true;
  options.moi_exec_save_sgpr = 92u;
  options.moi_dispatch_id_sgpr = 88u;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_TRUE(result.moi_persistent_sgprs_automatic);
  ASSERT_TRUE(result.resolved_moi_persistent_owner_sgpr);
  ASSERT_TRUE(result.resolved_moi_persistent_epoch_sgpr);
  EXPECT_EQ(*result.resolved_moi_persistent_owner_sgpr, 72u);
  EXPECT_EQ(*result.resolved_moi_persistent_epoch_sgpr, 73u);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("using complete-text coverage") != std::string::npos;
  })) << testing::PrintToString(result.warnings);
}

TEST(ConSanMoi, Cdna4ScalarHoleFailsClosedWithoutCompleteTextCoverage) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  const auto guest = build_cdna4_ds_store_b32(/*vaddr=*/2, /*vdata=*/3, /*byte_offset=*/0, kArch);
  ASSERT_TRUE(guest);

  std::vector<uint32_t> kernel_words(320u, build_s_nop(0, kArch));
  size_t cursor = 0u;
  kernel_words[cursor++] = build_s_swappc_b64(/*sdst=*/30u, /*ssrc0=*/2u, kArch);
  std::copy(guest->begin(), guest->end(), kernel_words.begin() + cursor);
  cursor += guest->size();
  kernel_words[cursor++] = build_v_mov_b32_e32(/*vdst=*/0u, vector_source_vgpr(7u), kArch);
  for (uint16_t sgpr = 0u; sgpr < 72u; ++sgpr)
    kernel_words[cursor++] = build_s_mov_b32(/*sdst=*/87u, sgpr, kArch);
  kernel_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/87u, kArch);
  kernel_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/90u, kArch);
  kernel_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/91u, kArch);
  kernel_words.back() = build_s_endpgm(kArch);
  const std::array<uint32_t, 1> helper_words = {build_s_endpgm(kArch)};
  // This executable tail is intentionally outside every declared function
  // range. Complete-text coverage must therefore fail rather than blessing an
  // owner-local hole across the unresolved call. The omitted destination uses
  // scalar-relative addressing, which is intentionally invisible to the
  // recovered owner scope.
  const std::array<uint32_t, 3> uncovered_tail = {
      0xBE802A02u, // s_movrels_b32 s0, s2
      build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/73u, kArch),
      build_s_endpgm(kArch),
  };
  std::vector<uint8_t> bytes = make_cdna4_code_object_with_local_function(
      kernel_words, helper_words, uncovered_tail, /*vgpr_granulated=*/3u);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 12u);
  });

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.moi_init_owner_epoch = true;
  options.moi_exec_save_sgpr = 92u;
  options.moi_dispatch_id_sgpr = 88u;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const ConSanResult result = try_patch_consan(bytes, options);

  EXPECT_FALSE(result.moi_persistent_sgprs_automatic);
  EXPECT_FALSE(result.resolved_moi_persistent_owner_sgpr);
  EXPECT_FALSE(result.resolved_moi_persistent_epoch_sgpr);
  ASSERT_FALSE(result.resource_plans.empty());
  EXPECT_TRUE(std::ranges::all_of(result.resource_plans, [](const auto &plan) {
    return !plan.has_indirect_sgpr_access && !plan.sgpr_reference_coverage_complete &&
           plan.scalar_tail_floor >= 104u;
  }));
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("unresolved guest call s_swappc_b64") != std::string::npos;
  })) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(std::ranges::none_of(result.warnings, [](const std::string &warning) {
    return warning.find("using complete-text coverage") != std::string::npos;
  }));
}

TEST(ConSanMoi, Cdna4ScalarHoleFailsClosedWhenDirectCallTargetIsUndecoded) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  constexpr uint16_t kReturnSreg = 30u;
  const auto guest = build_cdna4_ds_store_b32(/*vaddr=*/2, /*vdata=*/3, /*byte_offset=*/0, kArch);
  ASSERT_TRUE(guest);

  std::vector<uint32_t> kernel_words(320u, build_s_nop(0, kArch));
  const std::array<uint32_t, 1> helper_words = {build_s_endpgm(kArch)};
  size_t cursor = 0u;
  // The encoded direct target is the executable tail after the symbolized
  // helper. BasicBlock decoding deliberately omits that tail, so the explicit
  // static-successor completeness contract must make owner-local placement
  // fail closed.
  kernel_words[cursor++] = build_s_call_b64(
      kReturnSreg, static_cast<int16_t>(kernel_words.size() + helper_words.size() - 1u), kArch);
  std::copy(guest->begin(), guest->end(), kernel_words.begin() + cursor);
  cursor += guest->size();
  kernel_words[cursor++] = build_v_mov_b32_e32(/*vdst=*/0u, vector_source_vgpr(7u), kArch);
  for (uint16_t sgpr = 0u; sgpr < 72u; ++sgpr)
    kernel_words[cursor++] = build_s_mov_b32(/*sdst=*/87u, sgpr, kArch);
  kernel_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/87u, kArch);
  kernel_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/90u, kArch);
  kernel_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/91u, kArch);
  kernel_words.back() = build_s_endpgm(kArch);
  const std::array<uint32_t, 3> uncovered_tail = {
      build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/72u, kArch),
      build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/73u, kArch),
      build_s_endpgm(kArch),
  };
  std::vector<uint8_t> bytes = make_cdna4_code_object_with_local_function(
      kernel_words, helper_words, uncovered_tail, /*vgpr_granulated=*/3u);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 12u);
  });

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.moi_init_owner_epoch = true;
  options.moi_exec_save_sgpr = 92u;
  options.moi_dispatch_id_sgpr = 88u;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const ConSanResult result = try_patch_consan(bytes, options);

  EXPECT_FALSE(result.moi_persistent_sgprs_automatic);
  EXPECT_FALSE(result.resolved_moi_persistent_owner_sgpr);
  EXPECT_FALSE(result.resolved_moi_persistent_epoch_sgpr);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("unresolved guest call s_call_b64") != std::string::npos;
  })) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(std::ranges::none_of(result.warnings, [](const std::string &warning) {
    return warning.find("using complete-text coverage") != std::string::npos;
  }));
}

TEST(ConSanMoi, Cdna4ScalarHoleReportsUndecodedHelperFallthrough) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  constexpr uint16_t kReturnSreg = 30u;
  const auto guest = build_cdna4_ds_store_b32(/*vaddr=*/2, /*vdata=*/3, /*byte_offset=*/0, kArch);
  ASSERT_TRUE(guest);

  std::vector<uint32_t> kernel_words(320u, build_s_nop(0, kArch));
  size_t cursor = 0u;
  kernel_words[cursor++] =
      build_s_call_b64(kReturnSreg, static_cast<int16_t>(kernel_words.size() - 1u), kArch);
  std::copy(guest->begin(), guest->end(), kernel_words.begin() + cursor);
  cursor += guest->size();
  kernel_words[cursor++] = build_v_mov_b32_e32(/*vdst=*/0u, vector_source_vgpr(7u), kArch);
  for (uint16_t sgpr = 0u; sgpr < 72u; ++sgpr)
    kernel_words[cursor++] = build_s_mov_b32(/*sdst=*/87u, sgpr, kArch);
  kernel_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/87u, kArch);
  kernel_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/90u, kArch);
  kernel_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/91u, kArch);
  kernel_words.back() = build_s_endpgm(kArch);

  // The symbolized helper has no terminator, while the executable tail after
  // it is deliberately outside every declared function range. Its missing
  // fallthrough must be reported as such rather than inferred to be a branch
  // or silently treated as a closed helper body.
  const std::array<uint32_t, 1> helper_words = {build_s_nop(0, kArch)};
  const std::array<uint32_t, 3> uncovered_tail = {
      build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/72u, kArch),
      build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/73u, kArch),
      build_s_endpgm(kArch),
  };
  std::vector<uint8_t> bytes = make_cdna4_code_object_with_local_function(
      kernel_words, helper_words, uncovered_tail, /*vgpr_granulated=*/3u);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 12u);
  });

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.moi_init_owner_epoch = true;
  options.moi_exec_save_sgpr = 92u;
  options.moi_dispatch_id_sgpr = 88u;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const ConSanResult result = try_patch_consan(bytes, options);

  EXPECT_FALSE(result.moi_persistent_sgprs_automatic);
  EXPECT_FALSE(result.resolved_moi_persistent_owner_sgpr);
  EXPECT_FALSE(result.resolved_moi_persistent_epoch_sgpr);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("unresolved guest fallthrough after s_nop") != std::string::npos;
  })) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(std::ranges::none_of(result.warnings, [](const std::string &warning) {
    return warning.find("using complete-text coverage") != std::string::npos;
  }));
}

TEST(ConSanMoi, Cdna4ScalarHoleClassifiesResolvedBranchWithMissingFallthrough) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  constexpr uint16_t kReturnSreg = 30u;
  const auto guest = build_cdna4_ds_store_b32(/*vaddr=*/2, /*vdata=*/3, /*byte_offset=*/0, kArch);
  const auto branch = instrumentation::build_s_cbranch_scc1(/*offset_dwords=*/-2, kArch);
  ASSERT_TRUE(guest);
  ASSERT_TRUE(branch);

  std::vector<uint32_t> kernel_words(320u, build_s_nop(0, kArch));
  size_t cursor = 0u;
  kernel_words[cursor++] =
      build_s_call_b64(kReturnSreg, static_cast<int16_t>(kernel_words.size() - 1u), kArch);
  std::copy(guest->begin(), guest->end(), kernel_words.begin() + cursor);
  cursor += guest->size();
  kernel_words[cursor++] = build_v_mov_b32_e32(/*vdst=*/0u, vector_source_vgpr(7u), kArch);
  for (uint16_t sgpr = 0u; sgpr < 72u; ++sgpr)
    kernel_words[cursor++] = build_s_mov_b32(/*sdst=*/87u, sgpr, kArch);
  kernel_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/87u, kArch);
  kernel_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/90u, kArch);
  kernel_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/91u, kArch);
  kernel_words.back() = build_s_endpgm(kArch);

  // The conditional branch target is the decoded start of this helper, while
  // its fallthrough lies in the deliberately uncovered executable tail. The
  // diagnostic must describe the missing fallthrough, not the resolved branch
  // target.
  const std::array<uint32_t, 2> helper_words = {
      build_s_nop(0, kArch),
      *branch,
  };
  const std::array<uint32_t, 3> uncovered_tail = {
      build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/72u, kArch),
      build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/73u, kArch),
      build_s_endpgm(kArch),
  };
  std::vector<uint8_t> bytes = make_cdna4_code_object_with_local_function(
      kernel_words, helper_words, uncovered_tail, /*vgpr_granulated=*/3u);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 12u);
  });

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.moi_init_owner_epoch = true;
  options.moi_exec_save_sgpr = 92u;
  options.moi_dispatch_id_sgpr = 88u;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const ConSanResult result = try_patch_consan(bytes, options);

  EXPECT_FALSE(result.moi_persistent_sgprs_automatic);
  EXPECT_FALSE(result.resolved_moi_persistent_owner_sgpr);
  EXPECT_FALSE(result.resolved_moi_persistent_epoch_sgpr);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("unresolved guest fallthrough after s_cbranch") != std::string::npos;
  })) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(std::ranges::none_of(result.warnings, [](const std::string &warning) {
    return warning.find("unresolved guest branch s_cbranch") != std::string::npos;
  })) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(std::ranges::none_of(result.warnings, [](const std::string &warning) {
    return warning.find("using complete-text coverage") != std::string::npos;
  }));
}

TEST(ConSanMoi, Gfx1250PrivateEpochBarrierPreservesGuestVgprMsbMode) {
  constexpr uint32_t kBarrierSignal = 0xBE804EC1u;
  constexpr uint32_t kBarrierWait = 0xBF94FFFFu;
  constexpr uint16_t kGuestVgprMsbTransition = 0x4004u;
  constexpr uint8_t kGuestVgprMsbMode = 0x04u;
  std::vector<uint32_t> text_words = {
      *build_gfx1250_s_set_vgpr_msb(kGuestVgprMsbTransition, ROCJITSU_CODE_ARCH_CDNA5),
      0xD8340000u,
      0x00000000u, // ds_store_b32 v0, v0
      kBarrierSignal,
      kBarrierWait,
  };
  text_words.resize(320u, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5));
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5);

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_dynamic_access_records = true;
  options.moi_track_barriers = true;
  options.moi_init_owner_epoch = true;
  options.force_private_epoch = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(8, 0, 0, 0, 8);

  const ConSanResult result = try_patch_consan(
      make_gfx1250_code_object(text_words, "gfx1250_private_epoch_vgpr_msb"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.moi_private_epoch_automatic);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord,
                               &ConSanPatchInfo::kind),
            1u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const uint32_t select_low =
      *build_gfx1250_s_set_vgpr_msb_transition(kGuestVgprMsbMode, 0u, ROCJITSU_CODE_ARCH_CDNA5);
  const uint32_t restore_guest =
      *build_gfx1250_s_set_vgpr_msb_transition(0u, kGuestVgprMsbMode, ROCJITSU_CODE_ARCH_CDNA5);
  const auto barrier = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord, &ConSanPatchInfo::kind);
  ASSERT_NE(barrier, result.patches.end());
  EXPECT_EQ(barrier->anchor_offset, 3u * sizeof(uint32_t));
  const std::vector<uint32_t> words =
      text_words_at_offset(patched, barrier->trampoline_offset, barrier->trampoline_size);
  EXPECT_EQ(std::ranges::count(words, select_low), 1u);
  EXPECT_EQ(std::ranges::count(words, restore_guest), 1u);
  EXPECT_LT(std::ranges::find(words, select_low), std::ranges::find(words, restore_guest));
}

TEST(ConSanMoi, AtomicRecordPatchTrampolinesFlatAtomicAndWritesRecord) {
  const std::vector<uint8_t> bytes = make_rdna4_ordered_flat_atomic_code_object();
  ASSERT_FALSE(bytes.empty());
  ConSanOptions options = moi_options();
  options.moi_track_atomics = true;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 15;
  options.moi_epoch_vgpr = 16;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 0, 1, 1);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.kernels.front().atomic_sites.size(), 1u);
  const auto atomic_patch = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiAtomicRecord, &ConSanPatchInfo::kind);
  ASSERT_NE(atomic_patch, result.patches.end())
      << testing::PrintToString(result.warnings) << testing::PrintToString(result.errors)
      << testing::PrintToString(result.resource_plans);
  EXPECT_EQ(atomic_patch->anchor_offset, 12u);
  EXPECT_EQ(atomic_patch->original_size, 3u * sizeof(uint32_t));
  // Non-CAS RMWs have no meaningful success mask. Include the complete hardware
  // workgroup identity while keeping the dynamically indexed record body tightly
  // bounded.
  EXPECT_LE(atomic_patch->trampoline_size, 1200u)
      << "dynamically indexed atomic Record/Replay probes must remain within the 1200-byte "
         "append-cave budget";

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);

  const ConSanPatchInfo &patch = *atomic_patch;
  std::vector<uint32_t> anchor_words(patch.original_size / sizeof(uint32_t));
  std::memcpy(anchor_words.data(), patched.text_sections().front()->data() + patch.anchor_offset,
              anchor_words.size() * sizeof(uint32_t));
  EXPECT_EQ(anchor_words[0] >> 23u, kSoppEncodingPrefix);
  EXPECT_EQ(anchor_words[1], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(anchor_words[2], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));

  const std::vector<uint32_t> trampoline_words =
      text_words_at_offset(patched, patch.trampoline_offset, patch.trampoline_size);

  const auto original_atomic = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/2, /*vsrc=*/1, /*vdst=*/0, /*return_old_value=*/false, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(original_atomic);
  ASSERT_GE(trampoline_words.size(), original_atomic->size() + 1u);
  const auto guest_atomic_begin = trampoline_words.end() - original_atomic->size() - 1u;
  ASSERT_TRUE(patch.relocated_guest_instruction_offset);
  EXPECT_EQ(*patch.relocated_guest_instruction_offset,
            patch.trampoline_offset +
                static_cast<uint64_t>(std::distance(trampoline_words.begin(), guest_atomic_begin)) *
                    sizeof(uint32_t));
  EXPECT_TRUE(std::equal(original_atomic->begin(), original_atomic->end(), guest_atomic_begin));
  EXPECT_EQ(trampoline_words.back() >> 23u, kSoppEncodingPrefix);
  const auto flat_load_wait = instrumentation::build_s_wait_flat_load0(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(flat_load_wait);
  const std::array<uint32_t, 4> post_atomic_wait = {(*original_atomic)[0], (*original_atomic)[1],
                                                    (*original_atomic)[2], *flat_load_wait};
  EXPECT_FALSE(contains_subsequence(trampoline_words, post_atomic_wait));

  const uint64_t base = *options.moi_report_buffer_address;
  const ConSanMoiReportBufferLayout layout = consan_moi_report_buffer_layout_for_bytes(
      options.moi_report_buffer_size, /*include_barriers=*/false, /*include_atomics=*/true,
      /*include_fences=*/true);
  const std::vector<uint32_t> reserve_record = make_expected_fetch_add_one_words(
      base + offsetof(ConSanMoiReportHeader, atomic_record_count),
      static_cast<uint16_t>(*options.scratch_vgpr + 2u), *options.scratch_vgpr);
  EXPECT_TRUE(contains_subsequence(trampoline_words, reserve_record));
  EXPECT_FALSE(contains_subsequence(
      trampoline_words,
      make_expected_literal_store_words(base + offsetof(ConSanMoiReportHeader, atomic_record_count),
                                        1u, *options.scratch_vgpr)));
  const auto mov_capacity = build_v_mov_b32_e64_literal(
      *options.scratch_vgpr, layout.atomic_record_capacity, ROCJITSU_CODE_ARCH_RDNA4);
  const auto compare_capacity = build_v_cmp_gt_u32_e32_vcc(
      vector_source_vgpr(*options.scratch_vgpr), static_cast<uint16_t>(*options.scratch_vgpr + 2u),
      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(mov_capacity && compare_capacity && result.resolved_moi_exec_save_sgpr);
  EXPECT_TRUE(contains_subsequence(trampoline_words, *mov_capacity));
  EXPECT_NE(std::find(trampoline_words.begin(), trampoline_words.end(), *compare_capacity),
            trampoline_words.end());
  const auto narrow_exec = build_s_and_saveexec_b64(*result.resolved_moi_exec_save_sgpr,
                                                    kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_exec =
      build_s_mov_b64(kRdna4ExecLo, *result.resolved_moi_exec_save_sgpr, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(narrow_exec && restore_exec);
  EXPECT_NE(std::find(trampoline_words.begin(), trampoline_words.end(), *narrow_exec),
            trampoline_words.end());
  EXPECT_NE(std::find(trampoline_words.begin(), trampoline_words.end(), *restore_exec),
            trampoline_words.end());
  const auto save_active_exec =
      build_s_mov_b64(*result.resolved_moi_exec_save_sgpr, kRdna4ExecLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto lane_rank_lo =
      build_v_mbcnt_lo_u32_b32(*options.scratch_vgpr, *result.resolved_moi_exec_save_sgpr,
                               scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
  const auto lane_rank_hi = build_v_mbcnt_hi_u32_b32(
      *options.scratch_vgpr, static_cast<uint16_t>(*result.resolved_moi_exec_save_sgpr + 1u),
      vector_source_vgpr(*options.scratch_vgpr), ROCJITSU_CODE_ARCH_RDNA4);
  const auto select_first_lane = build_v_cmp_eq_u32_e32_vcc(
      scalar_positive_inline_u32(0), *options.scratch_vgpr, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(save_active_exec && lane_rank_lo && lane_rank_hi && select_first_lane);
  const auto saved_active_exec =
      std::find(trampoline_words.begin(), trampoline_words.end(), *save_active_exec);
  EXPECT_TRUE(contains_subsequence(trampoline_words, *lane_rank_lo));
  EXPECT_TRUE(contains_subsequence(trampoline_words, *lane_rank_hi));
  const auto ranked = std::search(trampoline_words.begin(), trampoline_words.end(),
                                  lane_rank_lo->begin(), lane_rank_lo->end());
  const auto selected =
      std::find(trampoline_words.begin(), trampoline_words.end(), *select_first_lane);
  const auto reserved = std::search(trampoline_words.begin(), trampoline_words.end(),
                                    reserve_record.begin(), reserve_record.end());
  ASSERT_NE(saved_active_exec, trampoline_words.end());
  ASSERT_NE(ranked, trampoline_words.end());
  ASSERT_NE(selected, trampoline_words.end());
  ASSERT_NE(reserved, trampoline_words.end());
  EXPECT_LT(saved_active_exec, ranked);
  EXPECT_LT(ranked, selected);
  EXPECT_LT(selected, reserved);

  EXPECT_EQ(
      std::ranges::count(trampoline_words,
                         build_v_mov_b32_e32(static_cast<uint16_t>(*options.scratch_vgpr + 5u),
                                             vector_source_vgpr(2), ROCJITSU_CODE_ARCH_RDNA4)),
      1u);
  EXPECT_EQ(
      std::ranges::count(trampoline_words,
                         build_v_mov_b32_e32(static_cast<uint16_t>(*options.scratch_vgpr + 6u),
                                             vector_source_vgpr(3), ROCJITSU_CODE_ARCH_RDNA4)),
      1u);
}

TEST(ConSanMoi, Gfx1250AtomicRecordPatchesOrderedFlatAtomic) {
  const auto atomic = build_gfx1250_flat_atomic_add_u32(
      /*vaddr=*/2, /*vsrc=*/1, /*vdst=*/2, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_TRUE(atomic);
  const std::array<uint32_t, 7> text_words = {
      0xEE0B0000u,  0x00000000u,  0x00000000u, // global_wb
      (*atomic)[0], (*atomic)[1], (*atomic)[2],
      0xBFB00000u, // s_endpgm
  };
  ConSanOptions options = moi_options();
  options.moi_track_atomics = true;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 15;
  options.moi_epoch_vgpr = 16;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 0, 1, 1);

  const ConSanResult result =
      try_patch_consan(make_gfx1250_code_object(text_words, "gfx1250_atomic_record"), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.kernels.front().atomic_sites.size(), 1u);
  const ConSanAtomicSite &site = result.kernels.front().atomic_sites.front();
  EXPECT_EQ(site.raw_saddr, kGfx1250FlatNoSaddrEncoding);
  EXPECT_EQ(site.raw_scope, 2u);
  EXPECT_EQ(site.raw_ioffset, 0);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAtomicRecord,
                               &ConSanPatchInfo::kind),
            1u);
}

TEST(ConSanMoi, Gfx1250RecordReplayMaterializesSignedFlatAccessOffset) {
  constexpr auto load = cdna5::build_vflat(
      cdna5::kFlatLoadB128Vflat,
      {.saddr = static_cast<uint8_t>(cdna5::OPR_SREG_NULL), .vdst = 8, .vaddr = 4, .ioffset = 16});
  const std::array<uint32_t, 8> text_words = {
      0xBE8001EBu, // s_mov_b64 s[0:1], src_shared_base
      build_v_mov_b32_e32(/*vdst=*/4, /*src=*/0, ROCJITSU_CODE_ARCH_CDNA5),
      build_v_mov_b32_e32(/*vdst=*/5, /*src=*/1, ROCJITSU_CODE_ARCH_CDNA5),
      load[0],
      load[1],
      load[2],
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5),
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5),
  };
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.flat_provenance_mode = ConSanFlatProvenanceMode::Strict;
  options.scratch_vgpr = 16;
  options.moi_exec_save_sgpr = 60;
  options.moi_owner_vgpr = 24;
  options.moi_epoch_vgpr = 25;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;

  const ConSanResult result =
      try_patch_consan(make_gfx1250_code_object(text_words, "gfx1250_flat_offset"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.moi_candidates.size(), 1u);
  EXPECT_EQ(result.moi_candidates.front().source, ConSanMoiCandidateSource::FlatGroup);
  EXPECT_EQ(result.moi_candidates.front().raw_ioffset, 16);
  EXPECT_TRUE(std::ranges::any_of(result.site_dispositions, [](const auto &site) {
    return site.site_kind == ConSanResourceSiteKind::Access &&
           site.disposition == ConSanSiteDisposition::Supported &&
           site.lowering_outcome == ConSanSiteLoweringOutcome::Patched;
  }));

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  std::vector<uint32_t> patched_words(patched.text_sections().front()->size() / sizeof(uint32_t));
  std::memcpy(patched_words.data(), patched.text_sections().front()->data(),
              patched.text_sections().front()->size());
  const auto add = build_v_add_u64_signed_i24(/*address_vgpr=*/22, /*displacement=*/16,
                                              ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_TRUE(add);
  EXPECT_TRUE(contains_subsequence(patched_words, *add));
}

TEST(ConSanMoi, Gfx1250WaveScopeAtomicIsTypedNotApplicable) {
  const auto atomic = build_gfx1250_flat_atomic_add_u32(
      /*vaddr=*/2, /*vsrc=*/1, /*vdst=*/2, /*return_old_value=*/true, /*scope=*/0,
      ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_TRUE(atomic);
  const std::array<uint32_t, 7> text_words = {
      0xEE0B0000u,  0x00000000u,  0x00000000u, // global_wb
      (*atomic)[0], (*atomic)[1], (*atomic)[2],
      0xBFB00000u, // s_endpgm
  };
  ConSanOptions options = moi_options();
  options.moi_track_atomics = true;

  const ConSanResult result =
      try_patch_consan(make_gfx1250_code_object(text_words, "gfx1250_wave_atomic"), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(std::ranges::any_of(result.site_dispositions, [](const auto &site) {
    return site.site_kind == ConSanResourceSiteKind::Atomic &&
           site.disposition == ConSanSiteDisposition::NotApplicable &&
           site.reason == ConSanSiteDispositionReason::UnsupportedScope &&
           site.lowering_outcome == ConSanSiteLoweringOutcome::NotApplicable;
  }));
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAtomicRecord,
                               &ConSanPatchInfo::kind),
            0u);
}

TEST(ConSanMoi, Gfx1250IsolatedLdsReleaseIsRetainedWithAccessReplay) {
  constexpr auto store = cdna5::build_vds(cdna5::kDsStoreB32Vds, {.addr = 4u, .data0 = 5u});
  constexpr auto atomic =
      cdna5::build_vds(cdna5::kDsAddU32Vds, {.offset0 = 12u, .addr = 2u, .data0 = 1u});
  const std::array<uint32_t, 6> text_words = {
      store[0],  store[1],  0xBFC90000u, // s_wait_storecnt_dscnt 0
      atomic[0], atomic[1], build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5)};
  ConSanOptions options = moi_options();
  options.moi_track_atomics = true;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 15;
  options.moi_epoch_vgpr = 16;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(2, 0, 0, 0, 0, 1, 1);
  options.max_patches = 2;

  const ConSanResult result =
      try_patch_consan(make_gfx1250_code_object(text_words, "gfx1250_lds_atomic_record"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.kernels.front().atomic_sites.size(), 1u);
  const ConSanAtomicSite &site = result.kernels.front().atomic_sites.front();
  EXPECT_EQ(site.mnemonic, "ds_add_u32");
  EXPECT_EQ(site.raw_ioffset, 12);
  EXPECT_FALSE(site.raw_scope);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAtomicRecord,
                               &ConSanPatchInfo::kind),
            1u);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               &ConSanPatchInfo::kind),
            1u);
  EXPECT_TRUE(std::ranges::any_of(result.site_dispositions, [](const auto &disposition) {
    return disposition.site_kind == ConSanResourceSiteKind::Atomic &&
           disposition.disposition == ConSanSiteDisposition::Supported &&
           disposition.reason == ConSanSiteDispositionReason::None;
  }));
}

TEST(ConSanMoi, Rdna4FamilyDenseAccessesShareOneWordCallRelay) {
  for (const rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    SCOPED_TRACE(arch);
    constexpr uint32_t kAccessCount = 9u;
    std::vector<uint32_t> text_words(8u, build_s_mov_b32(/*sdst=*/0, /*ssrc0=*/0, arch));
    for (uint32_t index = 0; index < kAccessCount; ++index) {
      text_words.push_back(0xD8340000u | index * sizeof(uint32_t));
      text_words.push_back(0x00000000u); // ds_store_b32 v0, v0 offset:index*4
    }
    if (arch == ROCJITSU_CODE_ARCH_RDNA4) {
      // Keep the sites near the entry and the appended relay beyond SOPP
      // reach, requiring the RDNA4 dense-relay recovery path.
      text_words.resize(33'000u, build_s_mov_b32(/*sdst=*/0, /*ssrc0=*/0, arch));
    }
    text_words.push_back(build_s_endpgm(arch));
    std::vector<uint8_t> bytes =
        arch == ROCJITSU_CODE_ARCH_CDNA5
            ? make_gfx1250_code_object(text_words, "gfx1250_dense_record_replay")
            : make_rdna4_lds_code_object(text_words, "rdna4_dense_record_replay");

    ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
    options.scratch_vgpr = 8;
    options.moi_exec_save_sgpr = 80;
    options.moi_owner_vgpr = 40;
    options.moi_epoch_vgpr = 41;
    options.moi_report_buffer_address = 0x123456780000ull;
    options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(kAccessCount, 0, 0, 0);
    options.moi_track_barriers = false;
    options.moi_track_atomics = false;
    options.max_patches = kAccessCount;

    const ConSanResult result = try_patch_consan(bytes, options);

    ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
    ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
    EXPECT_TRUE(result.final_validation_passed);
    EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                                 &ConSanPatchInfo::kind),
              kAccessCount);
    EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiIndirectBranchIsland,
                                 &ConSanPatchInfo::kind),
              2u); // One local relay plus one appended return-PC dispatcher.
  }
}

TEST(ConSanMoi, Gfx1250FarOrdinaryAcquireUsesOwnerLocalEntryWithAutomaticExecSave) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA5;
  constexpr uint32_t kAccessCount = 9u;
  constexpr size_t kOwnerWordCount = 33'000u;
  const auto make_owner = [](uint16_t first_live, uint16_t last_live, uint16_t dead_destination) {
    std::vector<uint32_t> words(9u, build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/0u, kArch));
    for (uint32_t index = 0; index < kAccessCount; ++index) {
      words.push_back(0xD8340000u | index * sizeof(uint32_t));
      words.push_back(0x00000000u); // ds_store_b32 v0, v0 offset:index*4
    }
    words.insert(words.end(), {
                                  0xC4050018u, 0x40883804u,
                                  0x00000005u, // buffer_load_b32 v4, v5, device
                                  0xBFC00000u, // s_wait_loadcnt 0
                                  0xEE0AC07Cu, 0x00080000u,
                                  0x00000000u, // global_inv scope:device
                                  0xBFC00000u, // s_wait_loadcnt 0
                                  0xBE804EC1u, // s_barrier_signal -1
                                  0xBF94FFFFu, // s_barrier_wait -1
                              });
    const size_t live_word_count = last_live - first_live + 1u;
    words.resize(kOwnerWordCount - live_word_count - 1u,
                 build_s_mov_b32(/*sdst=*/0u, scalar_positive_inline_u32(0u), kArch));
    for (uint16_t sgpr = first_live; sgpr <= last_live; ++sgpr)
      words.push_back(build_s_mov_b32(dead_destination, sgpr, kArch));
    words.push_back(build_s_endpgm(kArch));
    return words;
  };
  const std::vector<uint32_t> first_words =
      make_owner(/*first_live=*/16u, /*last_live=*/105u, /*dead_destination=*/0u);
  const std::vector<uint32_t> second_words =
      make_owner(/*first_live=*/0u, /*last_live=*/87u, /*dead_destination=*/88u);
  const std::vector<uint8_t> bytes = make_gfx1250_code_object_with_local_function(
      first_words, second_words, {}, kRdna4Wave64AllVgprsGranulated,
      /*function_is_kernel=*/true);

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_init_owner_epoch = true;
  options.moi_track_barriers = true;
  options.moi_track_atomics = true;
  options.moi_runtime_sample_stride = 65'536u;
  options.scratch_vgpr = 8u;
  options.moi_owner_vgpr = 40u;
  options.moi_epoch_vgpr = 41u;
  options.max_patches = 128u;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(
      /*access_record_capacity=*/128u, 0u, 0u, 0u, /*barrier_record_capacity=*/8u,
      /*atomic_record_capacity=*/0u, /*fence_record_capacity=*/4u);

  ASSERT_FALSE(options.moi_exec_save_sgpr);
  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.resolved_moi_exec_save_sgpr);
  ASSERT_EQ(result.resolved_moi_transient_sgpr_assignments.size(), 1u);
  EXPECT_NE(result.resolved_moi_transient_sgpr_assignments.front().exec_save_sgpr,
            *result.resolved_moi_exec_save_sgpr);
  EXPECT_TRUE(std::ranges::none_of(result.resolved_moi_transient_sgpr_assignments,
                                   &ConSanMoiTransientSgprAssignment::spill_backed));
  ASSERT_EQ(result.moi_fence_candidates.size(), 2u);
  EXPECT_TRUE(std::ranges::all_of(
      result.moi_fence_candidates,
      [](const ConSanMoiFenceCandidate &candidate) { return candidate.eligible; }));
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               &ConSanPatchInfo::kind),
            2u * kAccessCount);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord,
                               &ConSanPatchInfo::kind),
            4u);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiFenceRecord,
                               &ConSanPatchInfo::kind),
            2u)
      << testing::PrintToString(result.warnings);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  for (const ConSanPatchInfo &fence : result.patches) {
    if (fence.kind != ConSanPatchKind::TrampolineMoiFenceRecord)
      continue;
    ASSERT_EQ(fence.owner_descriptor_file_offsets.size(), 1u);
    const uint64_t owner = fence.owner_descriptor_file_offsets.front();
    const auto assignment =
        std::ranges::find(result.resolved_moi_transient_sgpr_assignments, owner,
                          &ConSanMoiTransientSgprAssignment::descriptor_file_offset);
    const uint16_t owner_exec_save_sgpr =
        assignment == result.resolved_moi_transient_sgpr_assignments.end()
            ? *result.resolved_moi_exec_save_sgpr
            : assignment->exec_save_sgpr;
    EXPECT_LT(owner_exec_save_sgpr, 128u);
    const auto local_entry =
        std::ranges::find_if(result.patches, [&](const ConSanPatchInfo &relay) {
          return relay.kind == ConSanPatchKind::TrampolineMoiIndirectBranchIsland &&
                 relay.anchor_offset == fence.anchor_offset &&
                 relay.trampoline_offset == fence.anchor_offset && relay.trampoline_size != 0u &&
                 std::ranges::find(relay.owner_descriptor_file_offsets, owner) !=
                     relay.owner_descriptor_file_offsets.end();
        });
    ASSERT_NE(local_entry, result.patches.end());
    // The complete qualified load/wait/invalidate sequence supplies the
    // relocatable owner-local entry. It is larger than the semantic cache
    // instruction and returns after the entire acquire suffix.
    EXPECT_GE(fence.original_size, 7u * sizeof(uint32_t));
    EXPECT_EQ(local_entry->trampoline_size, fence.original_size);
  }
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, Gfx1250DenseRuntimeGatePreservesCallReturnPair) {
  constexpr uint32_t kAccessCount = 9u;
  constexpr uint16_t kExecSaveSgpr = 80u;
  constexpr uint16_t kDispatchIdSgpr = 70u;
  constexpr uint16_t kCallReturnSgpr = kExecSaveSgpr + 6u;
  std::vector<uint32_t> text_words(
      8u, build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/0u, ROCJITSU_CODE_ARCH_CDNA5));
  for (uint32_t index = 0; index < kAccessCount; ++index) {
    text_words.push_back(0xD8340000u | index * sizeof(uint32_t));
    text_words.push_back(0x00000000u); // ds_store_b32 v0, v0 offset:index*4
  }
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5));

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 8u;
  options.moi_exec_save_sgpr = kExecSaveSgpr;
  options.moi_dispatch_id_sgpr = kDispatchIdSgpr;
  options.moi_owner_vgpr = 40u;
  options.moi_epoch_vgpr = 41u;
  options.moi_record_replay_workgroup_sgprs = {.x = 50u, .y = 51u, .z = 52u};
  options.moi_init_owner_epoch = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(kAccessCount, 0, 0, 0);
  options.moi_runtime_sample_stride = 256u;
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.max_patches = kAccessCount;

  const ConSanResult result =
      try_patch_consan(make_gfx1250_code_object(text_words, "gfx1250_dense_runtime_gate"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  const auto access_patch = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore, &ConSanPatchInfo::kind);
  ASSERT_NE(access_patch, result.patches.end());
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> words =
      text_words_at_offset(patched, access_patch->trampoline_offset, access_patch->trampoline_size);
  const auto save_scc = instrumentation::build_s_cselect_b32(
      /*sdst=*/kExecSaveSgpr + 4u, scalar_positive_inline_u32(1u), scalar_positive_inline_u32(0u),
      ROCJITSU_CODE_ARCH_CDNA5);
  const uint32_t initialize_residue = build_s_mov_b32(
      /*sdst=*/kExecSaveSgpr, scalar_positive_inline_u32(0u), ROCJITSU_CODE_ARCH_CDNA5);
  const uint32_t clobber_call_return = build_s_mov_b32(
      /*sdst=*/kCallReturnSgpr, scalar_positive_inline_u32(0u), ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_TRUE(save_scc);
  ASSERT_GE(words.size(), 2u);
  EXPECT_EQ(words[0], *save_scc);
  EXPECT_EQ(words[1], initialize_residue);
  EXPECT_EQ(std::ranges::find(words, clobber_call_return), words.end());
  EXPECT_NE(std::ranges::find(words, build_s_setpc_b64(kCallReturnSgpr, ROCJITSU_CODE_ARCH_CDNA5)),
            words.end());
}

TEST(ConSanMoi, Cdna4DenseRecordReplayAccessesDoNotRequireBarrierRouter) {
  constexpr uint32_t kAccessCount = 9u;
  constexpr size_t kLargeTextWords = 33'000u;
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  const uint32_t filler = build_s_mov_b32(/*sdst=*/0, /*ssrc0=*/0, kArch);
  std::vector<uint32_t> text_words(kLargeTextWords, filler);
  size_t cursor = 8u;
  for (uint32_t index = 0; index < kAccessCount; ++index) {
    const auto access = build_cdna4_ds_store_b32(
        /*vaddr=*/2, /*vdata=*/3, index * sizeof(uint32_t), kArch);
    ASSERT_TRUE(access);
    std::copy(access->begin(), access->end(), text_words.begin() + cursor);
    cursor += access->size();
  }
  text_words.back() = build_s_endpgm(kArch);

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 80;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(kAccessCount, 0, 0, 0);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.max_patches = kAccessCount;

  const ConSanResult result = try_patch_consan(
      make_cdna4_lds_code_object(text_words, "cdna4_dense_record_replay"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_EQ(result.target, ROCJITSU_CODE_TARGET_GFX950);
  EXPECT_EQ(result.arch, ROCJITSU_CODE_ARCH_CDNA4);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               &ConSanPatchInfo::kind),
            kAccessCount);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiIndirectBranchIsland,
                               &ConSanPatchInfo::kind),
            2u);
  EXPECT_FALSE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("entry island is unreachable") != std::string::npos;
  }));
}

TEST(ConSanMoi, Cdna4DenseRecordReplayRouteRestoresGuestSccBeforeAccessBody) {
  constexpr uint32_t kAccessCount = 18u;
  constexpr size_t kLargeTextWords = 33'000u;
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  constexpr uint16_t kExecSaveSgpr = 80u;
  constexpr uint16_t kSccSaveSgpr = kExecSaveSgpr + 4u;
  constexpr uint16_t kDispatchKeySgpr = kExecSaveSgpr + 5u;
  const uint32_t filler = build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/0u, kArch);
  std::vector<uint32_t> text_words(kLargeTextWords, filler);
  size_t cursor = 8u;
  for (uint32_t index = 0u; index < kAccessCount; ++index) {
    if (index + 1u == kAccessCount) {
      const auto seed_scc = instrumentation::build_s_cmp_eq_u32(
          /*ssrc0=*/0u, scalar_positive_inline_u32(0u), kArch);
      ASSERT_TRUE(seed_scc);
      text_words[cursor++] = *seed_scc;
    }
    const auto access = cdna4::build_ds(
        cdna4::kDsWriteB16Ds,
        {.offset0 = static_cast<uint8_t>(index * sizeof(uint16_t)), .addr = 2u, .data0 = 3u});
    std::copy(access.begin(), access.end(), text_words.begin() + cursor);
    cursor += access.size();
    if (index + 1u == kAccessCount) {
      const auto consume_scc = instrumentation::build_s_cbranch_scc1(/*displacement=*/0, kArch);
      ASSERT_TRUE(consume_scc);
      text_words[cursor++] = *consume_scc;
    }
  }
  text_words.back() = build_s_endpgm(kArch);

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 8u;
  options.moi_exec_save_sgpr = kExecSaveSgpr;
  options.moi_owner_vgpr = 40u;
  options.moi_epoch_vgpr = 41u;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(kAccessCount, 0u, 0u, 0u);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.max_patches = kAccessCount;

  const ConSanResult result = try_patch_consan(
      make_cdna4_lds_code_object(text_words, "cdna4_dense_record_replay_live_scc"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               &ConSanPatchInfo::kind),
            kAccessCount);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const auto last_route_compare = instrumentation::build_s_cmp_eq_u32(
      kDispatchKeySgpr, scalar_positive_inline_u32(kAccessCount), kArch);
  const auto restore_guest_scc =
      instrumentation::build_s_cmp_lg_u32(kSccSaveSgpr, scalar_positive_inline_u32(0u), kArch);
  ASSERT_TRUE(last_route_compare);
  ASSERT_TRUE(restore_guest_scc);

  bool found_last_route = false;
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.kind != ConSanPatchKind::TrampolineMoiIndirectBranchIsland)
      continue;
    const std::vector<uint32_t> words =
        text_words_at_offset(patched, patch.trampoline_offset, patch.trampoline_size);
    const auto compare = std::ranges::find(words, *last_route_compare);
    if (compare == words.end())
      continue;
    const size_t compare_index = static_cast<size_t>(compare - words.begin());
    ASSERT_GE(words.size(), compare_index + 4u);
    EXPECT_EQ(words[compare_index + 2u], *restore_guest_scc);
    found_last_route = true;
  }
  EXPECT_TRUE(found_last_route) << testing::PrintToString(result.patches);
}

TEST(ConSanMoi, Cdna4PartitionedDenseHostsAvoidEveryAccessCandidate) {
  constexpr uint32_t kAccessCount = 65u;
  constexpr size_t kLargeTextWords = 33'000u;
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  const uint32_t filler = build_s_mov_b32(/*sdst=*/0, /*ssrc0=*/0, kArch);
  std::vector<uint32_t> text_words(kLargeTextWords, filler);
  size_t cursor = 8u;
  for (uint32_t index = 0; index < kAccessCount; ++index) {
    const auto access = build_cdna4_ds_store_b32(
        /*vaddr=*/2, /*vdata=*/3, (index % 64u) * sizeof(uint32_t), kArch);
    ASSERT_TRUE(access);
    std::copy(access->begin(), access->end(), text_words.begin() + cursor);
    cursor += access->size();
  }
  text_words.back() = build_s_endpgm(kArch);

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 80;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(kAccessCount, 0, 0, 0);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.max_patches = kAccessCount;

  const ConSanResult result = try_patch_consan(
      make_cdna4_lds_code_object(text_words, "cdna4_partitioned_dense_record_replay"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               &ConSanPatchInfo::kind),
            kAccessCount);
  std::vector<const ConSanPatchInfo *> relocated_hosts;
  for (const ConSanPatchInfo &host : result.patches) {
    if (host.kind != ConSanPatchKind::TrampolineMoiIndirectBranchIsland || host.original_size == 0u)
      continue;
    relocated_hosts.push_back(&host);
    for (const ConSanPatchInfo &access : result.patches) {
      if (access.kind != ConSanPatchKind::TrampolineMoiAccessRecordStore)
        continue;
      EXPECT_FALSE(host.anchor_offset < access.anchor_offset + access.original_size &&
                   access.anchor_offset < host.anchor_offset + host.original_size)
          << "host=" << host.anchor_offset << "+" << host.original_size
          << " access=" << access.anchor_offset << "+" << access.original_size;
    }
  }
  ASSERT_GE(relocated_hosts.size(), 2u);
  for (size_t lhs = 0; lhs < relocated_hosts.size(); ++lhs) {
    for (size_t rhs = lhs + 1u; rhs < relocated_hosts.size(); ++rhs) {
      const ConSanPatchInfo &left = *relocated_hosts[lhs];
      const ConSanPatchInfo &right = *relocated_hosts[rhs];
      EXPECT_FALSE(left.anchor_offset < right.anchor_offset + right.original_size &&
                   right.anchor_offset < left.anchor_offset + left.original_size)
          << "left=" << left.anchor_offset << "+" << left.original_size
          << " right=" << right.anchor_offset << "+" << right.original_size;
    }
  }
}

TEST(ConSanMoi, Cdna4CompactRecordReplayBarriersReserveEightWordEntryIslands) {
  constexpr uint32_t kAccessCount = 9u;
  constexpr uint32_t kBarrierCount = 8u;
  constexpr size_t kLargeTextWords = 32'000u;
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  const uint32_t filler = build_s_mov_b32(/*sdst=*/0, /*ssrc0=*/0, kArch);
  std::vector<uint32_t> text_words(kLargeTextWords, filler);
  std::vector<uint64_t> barrier_offsets;
  size_t cursor = 32u;
  for (uint32_t index = 0; index < kAccessCount; ++index) {
    const auto access = build_cdna4_ds_store_b32(
        /*vaddr=*/2, /*vdata=*/3, index * sizeof(uint32_t), kArch);
    ASSERT_TRUE(access);
    std::copy(access->begin(), access->end(), text_words.begin() + cursor);
    cursor += access->size();
    if (index < kBarrierCount) {
      const auto barrier = build_cdna4_s_barrier(kArch);
      ASSERT_TRUE(barrier);
      barrier_offsets.push_back(cursor * sizeof(uint32_t));
      text_words[cursor++] = *barrier;
    }
  }
  text_words.back() = build_s_endpgm(kArch);

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 80;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_init_owner_epoch = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size =
      consan_moi_report_buffer_min_bytes(2u * kAccessCount, 0, 0, 0, 2u * kBarrierCount);
  options.moi_track_barriers = true;
  options.moi_track_atomics = false;
  options.max_patches = kAccessCount + kBarrierCount;

  const ConSanResult result = try_patch_consan(
      make_cdna4_lds_code_object(text_words, "cdna4_compact_record_replay_barriers"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               &ConSanPatchInfo::kind),
            kAccessCount);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord,
                               &ConSanPatchInfo::kind),
            kBarrierCount)
      << testing::PrintToString(result.warnings);
  EXPECT_EQ(std::ranges::count_if(result.site_dispositions,
                                  [](const auto &disposition) {
                                    return disposition.site_kind ==
                                               ConSanResourceSiteKind::Barrier &&
                                           disposition.lowering_outcome ==
                                               ConSanSiteLoweringOutcome::Patched;
                                  }),
            kBarrierCount);
  ASSERT_EQ(barrier_offsets.size(), kBarrierCount);
  for (uint64_t barrier_offset : barrier_offsets) {
    EXPECT_EQ(std::ranges::count_if(
                  result.patches,
                  [&](const ConSanPatchInfo &patch) {
                    return patch.kind == ConSanPatchKind::TrampolineMoiIndirectBranchIsland &&
                           patch.anchor_offset == barrier_offset;
                  }),
              1u)
        << "barrier_offset=" << barrier_offset;
  }
  EXPECT_FALSE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("no reachable indirect entry island") != std::string::npos;
  }));
}

TEST(ConSanMoi, Rdna4DenseRecordReplayBarriersUseRelocatedRouter) {
  // Seventeen split barriers contribute 34 supported member instructions,
  // exceeding the compact operating point and reserving the dense router.
  constexpr uint32_t kAccessCount = 17u;
  constexpr size_t kLargeTextWords = 33'000u;
  const uint32_t filler = build_s_mov_b32(/*sdst=*/0, /*ssrc0=*/0, ROCJITSU_CODE_ARCH_RDNA4);
  std::vector<uint32_t> text_words(kLargeTextWords, filler);
  size_t cursor = 32u;
  for (uint32_t index = 0; index < kAccessCount; ++index) {
    text_words[cursor++] = 0xD8340000u | index * sizeof(uint32_t);
    text_words[cursor++] = 0x00000000u; // ds_store_b32 v0, v0 offset:index*4
    text_words[cursor++] = 0xBE804EC1u; // s_barrier_signal -1
    text_words[cursor++] = 0xBF94FFFFu; // s_barrier_wait -1
  }
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 80;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_init_owner_epoch = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size =
      consan_moi_report_buffer_min_bytes(kAccessCount, 0, 0, 0, 2u * kAccessCount);
  options.moi_track_barriers = true;
  options.moi_track_atomics = false;
  options.max_patches = 64;

  const ConSanResult result = try_patch_consan(
      make_rdna4_lds_code_object(text_words, "rdna4_dense_record_replay_barriers"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               &ConSanPatchInfo::kind),
            kAccessCount);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord,
                               &ConSanPatchInfo::kind),
            2u * kAccessCount);
}

TEST(ConSanMoi, Rdna4AccessHeavyCompactSplitBarrierReservesEveryMemberEntry) {
  // Qwen/TP-style kernels can remain compact in pristine text while many LDS
  // access bodies push the later barrier bodies beyond a direct SOPP branch.
  // Unlike a dense router, compact lowering needs an independent entry for
  // both members of an RDNA4 signal/wait pair.
  constexpr uint32_t kAccessCount = 130u;
  constexpr uint32_t kBarrierCount = 2u;
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  std::vector<uint32_t> text_words(8u, build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/0u, kArch));
  for (uint32_t index = 0u; index < kAccessCount; ++index) {
    text_words.push_back(0xD8340000u | (index % 64u) * sizeof(uint32_t));
    text_words.push_back(0x00000000u); // ds_store_b32 v0, v0 offset:(index % 64)*4
  }
  text_words.push_back(0xBE804EC1u); // s_barrier_signal -1
  text_words.push_back(0xBF94FFFFu); // s_barrier_wait -1
  text_words.push_back(build_s_endpgm(kArch));

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_init_owner_epoch = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size =
      consan_moi_report_buffer_min_bytes(2u * kAccessCount, 0u, 0u, 0u, kBarrierCount);
  options.moi_runtime_sample_stride = 2u;
  options.moi_track_barriers = true;
  options.moi_track_atomics = false;
  options.max_patches = kAccessCount + kBarrierCount;

  const ConSanResult result = try_patch_consan(
      make_rdna4_lds_code_object(text_words, "rdna4_access_heavy_compact_split_barrier"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_EQ(result.moi_reserved_barrier_island_count, kBarrierCount);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               &ConSanPatchInfo::kind),
            kAccessCount)
      << testing::PrintToString(result.warnings);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord,
                               &ConSanPatchInfo::kind),
            kBarrierCount)
      << testing::PrintToString(result.warnings);
  EXPECT_FALSE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("no reachable indirect entry island") != std::string::npos;
  }));
}

TEST(ConSanMoi, CdnaSparseAccessDenseSpillRetainsEveryAccessAcrossTargets) {
  constexpr uint32_t kAccessCount = 130u;
  for (const rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    SCOPED_TRACE("arch=" + std::to_string(arch));
    std::vector<uint32_t> text_words(8u, build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/0u, arch));
    for (uint32_t index = 0u; index < kAccessCount; ++index) {
      const auto access = arch == ROCJITSU_CODE_ARCH_CDNA3
                              ? build_cdna3_ds_store_b32(/*vaddr=*/0u, /*vdata=*/1u,
                                                         (index % 64u) * sizeof(uint32_t), arch)
                              : build_cdna4_ds_store_b32(/*vaddr=*/0u, /*vdata=*/1u,
                                                         (index % 64u) * sizeof(uint32_t), arch);
      ASSERT_TRUE(access);
      text_words.insert(text_words.end(), access->begin(), access->end());
    }
    text_words.push_back(build_s_endpgm(arch));

    ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
    options.force_vgpr_spill = true;
    options.moi_init_owner_epoch = true;
    options.moi_report_buffer_address = 0x123456780000ull;
    options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(256u, 0u, 0u, 0u);
    options.moi_runtime_sample_stride = 2u;
    options.moi_track_barriers = false;
    options.moi_track_atomics = false;
    options.max_patches = kAccessCount;

    const std::vector<uint8_t> bytes =
        arch == ROCJITSU_CODE_ARCH_CDNA3
            ? make_cdna3_lds_code_object(text_words, "cdna3_sparse_access_dense_spill")
            : make_cdna4_lds_code_object(text_words, "cdna4_sparse_access_dense_spill");
    const ConSanResult result = try_patch_consan(bytes, options);

    ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
    ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
    EXPECT_TRUE(result.final_validation_passed);
    EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                                 &ConSanPatchInfo::kind),
              kAccessCount)
        << testing::PrintToString(result.warnings);
    EXPECT_TRUE(
        std::ranges::all_of(result.resource_plans, [](const ConSanCandidateResourcePlan &plan) {
          return plan.site_kind != ConSanResourceSiteKind::Access ||
                 plan.source == ConSanRegisterAllocationSource::SpillRequired;
        }));
  }
}

TEST(ConSanMoi, AmdhsaScalarPressureAccessBodiesRetainEverySiteAcrossTargets) {
  constexpr uint32_t kAccessCount = 130u;
  for (const rj_code_arch_t arch :
       {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    SCOPED_TRACE("arch=" + std::to_string(arch));
    std::vector<uint32_t> text_words(8u, build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/0u, arch));
    for (uint32_t index = 0u; index < kAccessCount; ++index) {
      const auto access = instrumentation::build_ds_store_b32(
          /*vaddr=*/0u, /*vdata=*/1u, static_cast<uint8_t>((index % 64u) * sizeof(uint32_t)), arch);
      ASSERT_TRUE(access);
      text_words.insert(text_words.end(), access->begin(), access->end());
    }
    // Keep every ordinary scalar live across every access. This removes the
    // indirect PC/key window and forces the bounded, branch-only continuation
    // used by full-pressure TopK-style kernels.
    for (uint16_t sgpr = 0u; sgpr < 102u; ++sgpr)
      text_words.push_back(build_s_mov_b32(/*sdst=*/0u, sgpr, arch));
    text_words.push_back(build_s_endpgm(arch));

    std::vector<uint8_t> bytes =
        arch == ROCJITSU_CODE_ARCH_RDNA3
            ? make_rdna3_lds_code_object(text_words, "rdna3_branch_only_access_pressure",
                                         /*vgpr_granulated=*/0u, /*wave32=*/false,
                                         /*uses_dynamic_stack=*/false,
                                         /*workgroup_id_dimension_mask=*/7u)
        : arch == ROCJITSU_CODE_ARCH_CDNA3
            ? make_cdna3_lds_code_object(text_words, "cdna3_branch_only_access_pressure")
            : make_cdna4_lds_code_object(text_words, "cdna4_branch_only_access_pressure");
    mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
      AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                      kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 13u);
    });
    AmdGpuCodeObject original(bytes.data(), bytes.size());
    ASSERT_TRUE(original.is_valid());
    ASSERT_EQ(original.text_sections().size(), 1u);
    const uint64_t original_text_size = original.text_sections().front()->size();

    ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
    options.force_vgpr_spill = true;
    options.force_private_epoch = true;
    options.moi_dispatch_id_sgpr = 88u;
    options.moi_init_owner_epoch = true;
    options.moi_report_buffer_address = 0x123456780000ull;
    options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(256u, 0u, 0u, 0u);
    options.moi_runtime_sample_stride = 2u;
    options.moi_track_barriers = false;
    options.moi_track_atomics = false;
    options.max_patches = kAccessCount;

    const ConSanResult result = try_patch_consan(bytes, options);

    ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
    ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
    EXPECT_TRUE(result.final_validation_passed);
    ASSERT_EQ(result.resolved_moi_transient_sgpr_assignments.size(), 1u);
    const bool uses_branch_only_scalar_spill = arch != ROCJITSU_CODE_ARCH_RDNA3;
    EXPECT_EQ(result.resolved_moi_transient_sgpr_assignments.front().branch_only_scalar_spill,
              uses_branch_only_scalar_spill);
    EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                                 &ConSanPatchInfo::kind),
              kAccessCount)
        << testing::PrintToString(result.warnings);
    ASSERT_TRUE(result.resolved_moi_exec_save_sgpr);
    const auto first_access = std::ranges::find(
        result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore, &ConSanPatchInfo::kind);
    ASSERT_NE(first_access, result.patches.end());
    ASSERT_TRUE(first_access->scratch_vgpr);
    const auto first_plan =
        std::ranges::find_if(result.resource_plans, [](const ConSanCandidateResourcePlan &plan) {
          return plan.site_kind == ConSanResourceSiteKind::Access;
        });
    ASSERT_NE(first_plan, result.resource_plans.end());
    ASSERT_EQ(first_access->spilled_vgpr_count,
              static_cast<uint16_t>(first_plan->scratch_vgpr_count + 1u));
    const uint16_t reservoir =
        static_cast<uint16_t>(*first_access->scratch_vgpr + first_access->spilled_vgpr_count - 1u);
    const auto save_first = instrumentation::build_v_writelane_b32(
        reservoir, *result.resolved_moi_exec_save_sgpr, /*lane=*/0u, arch);
    const auto restore_first = instrumentation::build_v_readlane_b32(
        *result.resolved_moi_exec_save_sgpr, reservoir, /*lane=*/0u, arch);
    ASSERT_TRUE(save_first && restore_first);
    AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(patched.is_valid());
    ASSERT_EQ(original.kernels().size(), 1u);
    ASSERT_EQ(patched.kernels().size(), 1u);
    KD original_descriptor{};
    KD patched_descriptor{};
    std::memcpy(&original_descriptor,
                bytes.data() + original.kernels().front().descriptor_file_offset,
                sizeof(original_descriptor));
    std::memcpy(&patched_descriptor,
                result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
                sizeof(patched_descriptor));
    const std::vector<uint32_t> first_body = text_words_at_offset(
        patched, first_access->trampoline_offset, first_access->trampoline_size);
    EXPECT_NE(std::ranges::search(first_body, *save_first).begin(), first_body.end());
    EXPECT_NE(std::ranges::search(first_body, *restore_first).begin(), first_body.end());
    const auto entry_prologue =
        std::ranges::find(result.patches, ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue,
                          &ConSanPatchInfo::kind);
    ASSERT_NE(entry_prologue, result.patches.end());
    EXPECT_EQ(entry_prologue->spilled_vgpr_count, first_access->spilled_vgpr_count);
    const std::vector<uint32_t> entry_body = text_words_at_offset(
        patched, entry_prologue->trampoline_offset, entry_prologue->trampoline_size);
    EXPECT_NE(std::ranges::search(entry_body, *save_first).begin(), entry_body.end());
    EXPECT_NE(std::ranges::search(entry_body, *restore_first).begin(), entry_body.end());
    const uint16_t guest_workgroup_x = static_cast<uint16_t>(AMDHSA_BITS_GET(
        original_descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT));
    const uint16_t full_workgroup_x = static_cast<uint16_t>(AMDHSA_BITS_GET(
        patched_descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT));
    ASSERT_LE(*result.resolved_moi_exec_save_sgpr, guest_workgroup_x);
    const uint16_t guest_slot =
        static_cast<uint16_t>(guest_workgroup_x - *result.resolved_moi_exec_save_sgpr);
    const auto save_semantic_workgroup_x =
        instrumentation::build_v_writelane_b32(reservoir, full_workgroup_x, guest_slot, arch);
    ASSERT_TRUE(save_semantic_workgroup_x);
    EXPECT_NE(std::ranges::search(entry_body, *save_semantic_workgroup_x).begin(), entry_body.end())
        << "the private prologue must back up guest-semantic workgroup X, not the raw "
           "dispatch-preload value occupying its ABI slot";
    EXPECT_EQ(result.moi_branch_only_placement_failure_count, 0u);
    EXPECT_EQ(result.moi_branch_only_routing_telemetry.pair_attempt_count,
              uses_branch_only_scalar_spill ? kAccessCount : 0u);
    EXPECT_EQ(result.moi_branch_only_routing_telemetry.plan_call_count,
              uses_branch_only_scalar_spill ? 1u : 0u);
    const auto uses_appended_relay = [&](const ConSanPatchInfo &patch) {
      return patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore &&
             (std::ranges::any_of(patch.branch_only_entry_relay_offsets,
                                  [&](uint64_t relay) { return relay >= original_text_size; }) ||
              std::ranges::any_of(patch.branch_only_return_relay_offsets,
                                  [&](uint64_t relay) { return relay >= original_text_size; }));
    };
    EXPECT_EQ(std::ranges::any_of(result.patches, uses_appended_relay),
              uses_branch_only_scalar_spill);
  }
}

TEST(ConSanMoi, Cdna4OrdinaryBodiesPreserveLaterBranchOnlyRelaySpine) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  constexpr uint32_t kAccessCountPerKernel = 512u;
  const auto access = build_cdna4_ds_store_b32(/*vaddr=*/0u, /*vdata=*/1u,
                                               /*byte_offset=*/0u, kArch);
  const auto barrier = build_cdna4_s_barrier(kArch);
  ASSERT_TRUE(access && barrier);
  const auto make_accesses = [&] {
    std::vector<uint32_t> words(8u, build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/0u, kArch));
    for (uint32_t index = 0u; index < kAccessCountPerKernel; ++index)
      words.insert(words.end(), access->begin(), access->end());
    return words;
  };
  std::vector<uint32_t> ordinary_words = make_accesses();
  ordinary_words.push_back(build_s_endpgm(kArch));
  std::vector<uint32_t> branch_only_words = make_accesses();
  branch_only_words.push_back(*barrier);
  for (uint16_t sgpr = 0u; sgpr < 102u; ++sgpr)
    branch_only_words.push_back(build_s_mov_b32(/*sdst=*/0u, sgpr, kArch));
  branch_only_words.push_back(build_s_endpgm(kArch));

  constexpr uint32_t kCdna4Wave64AllVgprsGranulated = 31u;
  std::vector<uint8_t> bytes = make_cdna4_code_object_with_local_function(
      ordinary_words, branch_only_words, {}, kCdna4Wave64AllVgprsGranulated,
      /*function_is_kernel=*/true);
  mutate_kernel_descriptor(bytes, "lds_probe", [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 0u);
  });
  mutate_kernel_descriptor(bytes, "lds_helper", [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 13u);
  });
  AmdGpuCodeObject original(bytes.data(), bytes.size());
  ASSERT_TRUE(original.is_valid());
  ASSERT_EQ(original.text_sections().size(), 1u);
  const uint64_t original_text_size = original.text_sections().front()->size();
  ConSanMoiAutoReportInventory report_inventory;
  report_inventory.engine = ConSanMoiEngine::RecordReplay;
  report_inventory.access_range_count = 2u * kAccessCountPerKernel;
  report_inventory.barrier_event_count = 1u;
  report_inventory.diagnostic_count = 1u;
  report_inventory.record_replay_access_dispatch_bank_count = 8u;
  report_inventory.record_replay_access_owner_bank_count = 8u;
  report_inventory.record_replay_bank_count_adaptive = true;
  report_inventory = fit_consan_moi_record_replay_auto_report_inventory(report_inventory);
  const ConSanMoiAutoReportPlan report_plan = plan_consan_moi_auto_report(report_inventory);
  ASSERT_TRUE(report_plan.complete());
  const auto layout_override = consan_moi_auto_report_layout_override(report_plan);
  ASSERT_TRUE(layout_override);

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.force_vgpr_spill = true;
  options.moi_dispatch_id_sgpr = 88u;
  options.moi_init_owner_epoch = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = report_plan.required_bytes;
  options.moi_report_layout = *layout_override;
  options.moi_runtime_sample_stride = 2u;
  options.moi_track_barriers = true;
  options.moi_track_atomics = false;
  options.max_patches = 2u * kAccessCountPerKernel + 1u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               &ConSanPatchInfo::kind),
            2u * kAccessCountPerKernel)
      << testing::PrintToString(result.warnings);
  EXPECT_EQ(result.moi_branch_only_placement_failure_count, 0u);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord,
                               &ConSanPatchInfo::kind),
            1u)
      << testing::PrintToString(result.warnings);
  EXPECT_EQ(result.moi_branch_only_routing_telemetry.pair_attempt_count,
            kAccessCountPerKernel + 1u);
  EXPECT_EQ(result.moi_branch_only_routing_telemetry.plan_call_count, 2u);
  EXPECT_TRUE(std::ranges::any_of(result.patches, [&](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore &&
           patch.branch_only_continuation &&
           (std::ranges::any_of(patch.branch_only_entry_relay_offsets,
                                [&](uint64_t relay) { return relay >= original_text_size; }) ||
            std::ranges::any_of(patch.branch_only_return_relay_offsets,
                                [&](uint64_t relay) { return relay >= original_text_size; }));
  }));

  const uint64_t relay_bank_bytes = 2u * (kAccessCountPerKernel + 1u) * sizeof(uint32_t);
  std::vector<std::pair<uint64_t, uint64_t>> relay_banks;
  std::ranges::copy_if(result.moi_generated_branch_relay_ranges, std::back_inserter(relay_banks),
                       [&](const auto &range) {
                         return range.first >= original_text_size &&
                                range.second - range.first == relay_bank_bytes;
                       });
  ASSERT_GE(relay_banks.size(), 2u);
  std::ranges::sort(relay_banks);
  uint64_t maximum_reserved_body_bytes = 0u;
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore) {
      maximum_reserved_body_bytes = std::max<uint64_t>(
          maximum_reserved_body_bytes, patch.trampoline_size + 9u * sizeof(uint32_t));
    }
  }
  ASSERT_GT(maximum_reserved_body_bytes, 0u);
  constexpr uint64_t kRelaySpineSpacing = kSoppBranchMaximumForwardReachBytes / 2u;
  for (size_t bank = 1u; bank < relay_banks.size(); ++bank) {
    ASSERT_LE(relay_banks[bank - 1u].second, relay_banks[bank].first);
    EXPECT_GE(relay_banks[bank].first - relay_banks[bank - 1u].second + maximum_reserved_body_bytes,
              kRelaySpineSpacing)
        << "a demand-scaled bank must not consume its own following spacing interval";
  }
}

TEST(ConSanMoi, Cdna4FarRecordReplayBarriersUseRelocatedRouterBelowCompactCountLimit) {
  // A generated object can strand a small barrier inventory too. Keep this
  // below the count-based compact threshold so branch reach, rather than the
  // number of sites, is what reserves the relocated router.
  constexpr uint32_t kSiteCount = 28u;
  constexpr size_t kLargeTextWords = 33'000u;
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  const uint32_t filler = build_s_mov_b32(/*sdst=*/0, /*ssrc0=*/0, kArch);
  std::vector<uint32_t> text_words(kLargeTextWords, filler);
  size_t cursor = 32u;
  for (uint32_t index = 0; index < kSiteCount; ++index) {
    const auto access = build_cdna4_ds_store_b32(
        /*vaddr=*/2, /*vdata=*/3, index * sizeof(uint32_t), kArch);
    const auto barrier = build_cdna4_s_barrier(kArch);
    ASSERT_TRUE(access && barrier);
    std::copy(access->begin(), access->end(), text_words.begin() + cursor);
    cursor += access->size();
    text_words[cursor++] = *barrier;
  }
  text_words.back() = build_s_endpgm(kArch);

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 80;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_init_owner_epoch = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size =
      consan_moi_report_buffer_min_bytes(kSiteCount, 0, 0, 0, 2u * kSiteCount);
  options.moi_track_barriers = true;
  options.moi_track_atomics = false;
  options.max_patches = 96u;

  const ConSanResult result = try_patch_consan(
      make_cdna4_lds_code_object(text_words, "cdna4_far_record_replay_barriers"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_EQ(result.target, ROCJITSU_CODE_TARGET_GFX950);
  EXPECT_EQ(result.arch, ROCJITSU_CODE_ARCH_CDNA4);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               &ConSanPatchInfo::kind),
            kSiteCount);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord,
                               &ConSanPatchInfo::kind),
            kSiteCount)
      << testing::PrintToString(result.warnings);
  EXPECT_FALSE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("no reachable indirect entry island") != std::string::npos;
  }));
}

TEST(ConSanMoi, Cdna4AccessHeavyRecordReplayPreservesReservedBarrierAndFenceRelays) {
  // Stream-K-like generated kernels can have many LDS accesses but only a
  // handful of phase barriers and language-level release/acquire operations.
  // The access pass must reserve independent compact relays for both later
  // synchronization passes; barrier lowering must not consume the fence
  // reservation.
  constexpr uint32_t kAccessCount = 80u;
  constexpr uint32_t kBarrierCount = 5u;
  constexpr size_t kTextWords = 5'000u;
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  const uint32_t filler = build_s_mov_b32(/*sdst=*/0, /*ssrc0=*/0, kArch);
  std::vector<uint32_t> text_words(kTextWords, filler);
  size_t cursor = 32u;
  for (uint32_t index = 0; index < kAccessCount; ++index) {
    text_words[cursor++] = 0xD8EE0400u;
    text_words[cursor++] = 0x0800000Bu; // ds_read2_b64 v[8:11], v11 offset1:4
    if ((index + 1u) % (kAccessCount / kBarrierCount) == 0u) {
      const auto barrier = build_cdna4_s_barrier(kArch);
      ASSERT_TRUE(barrier);
      text_words[cursor++] = *barrier;
    }
  }
  const std::array<uint32_t, 5> release_sequence = {
      0xE0A08000u,
      0x00000000u, // buffer_wbl2 sc1
      0xBF8C0F70u, // s_waitcnt vmcnt(0)
      0xDE708004u,
      0x00000100u, // global_store_dword v0, v1, s[0:1] offset:4 sc1
  };
  const std::array<uint32_t, 5> acquire_sequence = {
      0xDE508004u,
      0x027F0000u, // global_load_dword v2, v[0:1], off offset:4 sc1
      0xBF8C0F70u, // s_waitcnt vmcnt(0)
      0xE0A48000u,
      0x00000000u, // buffer_inv sc1
  };
  std::ranges::copy(release_sequence, text_words.begin() + cursor);
  cursor += release_sequence.size();
  std::ranges::copy(acquire_sequence, text_words.begin() + cursor);
  text_words.back() = build_s_endpgm(kArch);

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.force_vgpr_spill = true;
  options.moi_init_owner_epoch = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  // Leave independent headroom for both record domains; the production
  // layout partitions one shared backing buffer between them.
  options.moi_report_buffer_size =
      consan_moi_report_buffer_min_bytes(256u, 0, 0, 0, 256u, 0u, 256u);
  options.moi_track_barriers = true;
  options.moi_track_atomics = true;
  options.moi_runtime_sample_stride = 65'536u;
  options.max_patches = 96u;

  constexpr uint32_t kCdna4Wave64AllVgprsGranulated = 31u;
  const ConSanResult result =
      try_patch_consan(make_cdna4_lds_code_object(text_words, "cdna4_access_heavy_record_replay",
                                                  kCdna4Wave64AllVgprsGranulated),
                       options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               &ConSanPatchInfo::kind),
            kAccessCount);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord,
                               &ConSanPatchInfo::kind),
            kBarrierCount)
      << testing::PrintToString(result.warnings);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiFenceRecord,
                               &ConSanPatchInfo::kind),
            2u)
      << testing::PrintToString(result.warnings);
  EXPECT_FALSE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("no reachable indirect entry island") != std::string::npos;
  }));
}

TEST(ConSanMoi, Rdna4DenseFunctionBarriersUseRelocatableRouter) {
  constexpr uint32_t kSiteCount = 17u;
  const std::array<uint32_t, 2> kernel_words = {
      pack_sopk(/*s_call_b64=*/0x14, /*sdst=*/30, /*simm16=*/1),
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const uint32_t filler = build_s_mov_b32(/*sdst=*/20u, /*ssrc0=*/20u, ROCJITSU_CODE_ARCH_RDNA4);
  std::vector<uint32_t> function_words(32u, filler);
  for (uint32_t index = 0; index < kSiteCount; ++index) {
    function_words.push_back(0xD8340000u | index * sizeof(uint32_t));
    function_words.push_back(0x00000000u); // ds_store_b32 v0, v0 offset:index*4
    function_words.push_back(0xBE804EC1u); // s_barrier_signal -1
    function_words.push_back(0xBF94FFFFu); // s_barrier_wait -1
  }
  // Keep every function-local relay target beyond SOPP reach and leave no
  // NOP island. Access and barrier dispatchers must each use a relocatable
  // host proven against every kernel that owns this function.
  function_words.resize(33'000u, filler);
  function_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4));

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 80;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_init_owner_epoch = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size =
      consan_moi_report_buffer_min_bytes(kSiteCount, 0, 0, 0, 2u * kSiteCount);
  options.moi_track_barriers = true;
  options.moi_track_atomics = false;
  options.max_patches = 64u;

  const ConSanResult result = try_patch_consan(
      make_rdna4_code_object_with_local_function(kernel_words, function_words), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               &ConSanPatchInfo::kind),
            kSiteCount);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord,
                               &ConSanPatchInfo::kind),
            2u * kSiteCount);
  EXPECT_EQ(std::ranges::count_if(result.site_dispositions,
                                  [](const auto &site) {
                                    return site.site_kind == ConSanResourceSiteKind::Barrier &&
                                           !site.in_kernel &&
                                           site.lowering_outcome ==
                                               ConSanSiteLoweringOutcome::Patched;
                                  }),
            2u * kSiteCount);
}

TEST(ConSanMoi, Gfx1250DenseAccessesPartitionRelayWindowsAcrossLargeKernel) {
  constexpr uint32_t kAccessesPerWindow = 9u;
  constexpr uint32_t kSecondWindowWord = 65'580u;
  const uint32_t filler = build_s_mov_b32(/*sdst=*/0, /*ssrc0=*/0, ROCJITSU_CODE_ARCH_CDNA5);
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
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5));
  const std::vector<uint8_t> bytes =
      make_gfx1250_code_object(text_words, "gfx1250_partitioned_dense_record_replay");

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 80;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size =
      consan_moi_report_buffer_min_bytes(2u * kAccessesPerWindow, 0, 0, 0);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.max_patches = 2u * kAccessesPerWindow;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               &ConSanPatchInfo::kind),
            2u * kAccessesPerWindow);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiIndirectBranchIsland,
                               &ConSanPatchInfo::kind),
            4u); // One host relay and one appended dispatcher per reachability window.
  EXPECT_TRUE(std::ranges::none_of(result.warnings, [](const std::string &warning) {
    return warning.find("inside a relocated prefix") != std::string::npos;
  }));
}

TEST(ConSanMoi, RecordReplayAllSupportedPolicyIgnoresNominalPatchLimit) {
  constexpr uint32_t kAccessCount = 9u;
  std::vector<uint32_t> text_words(
      8u, build_s_mov_b32(/*sdst=*/0, /*ssrc0=*/0, ROCJITSU_CODE_ARCH_CDNA5));
  for (uint32_t index = 0; index < kAccessCount; ++index) {
    text_words.push_back(0xD8340000u | index * sizeof(uint32_t));
    text_words.push_back(0x00000000u); // ds_store_b32 v0, v0 offset:index*4
  }
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5));
  const std::vector<uint8_t> bytes =
      make_gfx1250_code_object(text_words, "gfx1250_all_supported_record_replay");

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 80;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(kAccessCount, 0, 0, 0);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.max_patches = 1u;
  options.max_patches_is_expert_limit = false;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               &ConSanPatchInfo::kind),
            kAccessCount);

  const ConSanMoiAutoReportInventory inventory =
      inventory_consan_moi_auto_report(result, options, bytes);
  EXPECT_EQ(inventory.access_range_count, kAccessCount);
}

TEST(ConSanMoi, Gfx1250DenseAccessesUseRelocatableHostPastKernelEntry) {
  constexpr uint32_t kAccessCount = 9u;
  std::vector<uint32_t> text_words = {
      build_s_branch(/*simm16=*/8, ROCJITSU_CODE_ARCH_CDNA5),
  };
  text_words.insert(text_words.end(), 8u,
                    build_s_mov_b32(/*sdst=*/0, /*ssrc0=*/0, ROCJITSU_CODE_ARCH_CDNA5));
  for (uint32_t index = 0; index < kAccessCount; ++index) {
    text_words.push_back(0xD8340000u | index * sizeof(uint32_t));
    text_words.push_back(0x00000000u); // ds_store_b32 v0, v0 offset:index*4
  }
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5));
  std::vector<uint8_t> bytes =
      make_gfx1250_code_object(text_words, "gfx1250_dense_record_replay_late_host");

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 80;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(kAccessCount, 0, 0, 0);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.max_patches = kAccessCount;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               &ConSanPatchInfo::kind),
            kAccessCount);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiIndirectBranchIsland,
                               &ConSanPatchInfo::kind),
            2u);
}

TEST(ConSanMoi, Gfx1250DenseAccessRouterPreservesOrdinaryAcquireSequence) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA5;
  constexpr uint32_t kAccessCount = 9u;
  constexpr auto acquire = cdna5::build_vflat(
      cdna5::kFlatLoadB32Vflat,
      {.saddr = static_cast<uint8_t>(cdna5::OPR_SREG_NULL), .vdst = 10u, .scope = 0u, .vaddr = 0u});
  constexpr uint64_t kAcquireOffset = (2u * kAccessCount + 5u) * sizeof(uint32_t);
  const auto wait = build_s_wait_loadcnt_dscnt0(kArch);
  ASSERT_TRUE(wait);

  std::vector<uint32_t> text_words;
  for (uint32_t index = 0; index < kAccessCount; ++index) {
    text_words.push_back(0xD8340000u | index * sizeof(uint32_t));
    text_words.push_back(0x00000000u); // ds_store_b32 v0, v0 offset:index*4
  }
  // This nine-word tail is attractive as a safe dense access-router host. Its
  // final four words are nevertheless a compiler workgroup-acquire sequence
  // and belong to the downstream synchronization pass; router placement must
  // not consume them as generic relocation space.
  text_words.insert(text_words.end(), {
                                          0xBE8001EBu, // s_mov_b64 s[0:1], src_shared_base
                                          0xD5810000u,
                                          0x00000000u, // v_mov_b32_e64 v0, s0
                                          0xD5810001u,
                                          0x00000001u, // v_mov_b32_e64 v1, s1
                                          acquire[0],
                                          acquire[1],
                                          acquire[2],
                                          *wait,
                                      });
  text_words.push_back(build_s_endpgm(kArch));

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 16u;
  options.moi_exec_save_sgpr = 80u;
  options.moi_owner_vgpr = 40u;
  options.moi_epoch_vgpr = 41u;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size =
      consan_moi_report_buffer_min_bytes(kAccessCount, 0, 0, 0, 0, kAccessCount, kAccessCount);
  options.moi_track_barriers = false;
  options.moi_track_atomics = true;
  options.max_patches = kAccessCount;

  const ConSanResult result = try_patch_consan(
      make_gfx1250_code_object(text_words, "gfx1250_dense_preserve_ordinary_acquire"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.sync_sequences.size(), 1u);
  EXPECT_EQ(result.sync_sequences.front().kind, ConSanSyncSequenceKind::OrdinaryMemory);
  EXPECT_EQ(result.sync_sequences.front().memory_role, ConSanSyncMemoryRole::Acquire);
  EXPECT_EQ(result.sync_sequences.front().begin_text_offset, kAcquireOffset);
  EXPECT_EQ(result.sync_sequences.front().end_text_offset, kAcquireOffset + 4u * sizeof(uint32_t));
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               &ConSanPatchInfo::kind),
            kAccessCount);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAtomicRecord,
                               &ConSanPatchInfo::kind),
            1u)
      << testing::PrintToString(result.warnings);
  EXPECT_TRUE(std::ranges::none_of(result.patches, [&](const ConSanPatchInfo &patch) {
    if (patch.kind != ConSanPatchKind::TrampolineMoiIndirectBranchIsland ||
        patch.original_size == 0u)
      return false;
    const uint64_t patch_end = patch.anchor_offset + patch.original_size;
    return patch.anchor_offset < result.sync_sequences.front().end_text_offset &&
           kAcquireOffset < patch_end;
  })) << testing::PrintToString(result.patches);
  EXPECT_TRUE(std::ranges::any_of(result.site_dispositions, [&](const auto &site) {
    return site.site_kind == ConSanResourceSiteKind::Atomic && site.text_offset == kAcquireOffset &&
           site.lowering_outcome == ConSanSiteLoweringOutcome::Patched;
  })) << testing::PrintToString(result.site_dispositions);
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, Gfx1250DenseAccessesRejectUnreachableHostWhenOwnerUsesRouterState) {
  constexpr uint32_t kAccessCount = 9u;
  constexpr size_t kUnreachableBeginWord = 1u;
  constexpr size_t kUnreachableEndWord = 9u;
  std::vector<uint32_t> text_words = {
      build_s_branch(/*simm16=*/8, ROCJITSU_CODE_ARCH_CDNA5),
  };
  text_words.insert(text_words.end(), kUnreachableEndWord - kUnreachableBeginWord,
                    build_s_mov_b32(/*sdst=*/0, /*ssrc0=*/0, ROCJITSU_CODE_ARCH_CDNA5));
  // The reachable owner explicitly references the record/replay router base.
  // Its unreachable tail therefore cannot be treated as scalar-state-free
  // relocation storage merely because it has no live-before CFG snapshot.
  text_words.push_back(build_s_mov_b32(/*sdst=*/80, /*ssrc0=*/80, ROCJITSU_CODE_ARCH_CDNA5));
  for (uint32_t index = 0; index < kAccessCount; ++index) {
    text_words.push_back(0xD8340000u | index * sizeof(uint32_t));
    text_words.push_back(0x00000000u); // ds_store_b32 v0, v0 offset:index*4
  }
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5));

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 80;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(kAccessCount, 0, 0, 0);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.max_patches = kAccessCount;

  const ConSanResult result = try_patch_consan(
      make_gfx1250_code_object(text_words, "gfx1250_dense_reject_unsafe_unreachable_host"),
      options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  EXPECT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               &ConSanPatchInfo::kind),
            kAccessCount);
  constexpr uint64_t kUnreachableBegin = kUnreachableBeginWord * sizeof(uint32_t);
  constexpr uint64_t kUnreachableEnd = kUnreachableEndWord * sizeof(uint32_t);
  EXPECT_TRUE(std::ranges::none_of(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiIndirectBranchIsland &&
           patch.original_size != 0u && patch.anchor_offset >= kUnreachableBegin &&
           patch.anchor_offset < kUnreachableEnd;
  })) << testing::PrintToString(result.patches);
}

TEST(ConSanMoi, Gfx1250DenseAccessesPreserveGuestVgprMsbMode) {
  constexpr uint32_t kAccessCount = 9u;
  constexpr uint16_t kGuestVgprMsbTransition = 0x4004u;
  constexpr uint8_t kGuestVgprMsbMode = 0x04u;
  std::vector<uint32_t> text_words(
      8u, build_s_mov_b32(/*sdst=*/0, /*ssrc0=*/0, ROCJITSU_CODE_ARCH_CDNA5));
  text_words.push_back(
      *build_gfx1250_s_set_vgpr_msb(kGuestVgprMsbTransition, ROCJITSU_CODE_ARCH_CDNA5));
  for (uint32_t index = 0; index < kAccessCount; ++index) {
    text_words.push_back(0xD8340000u | index * sizeof(uint32_t));
    text_words.push_back(0x00000000u); // ds_store_b32 v0, v0 offset:index*4
  }
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5));
  std::vector<uint8_t> bytes =
      make_gfx1250_code_object(text_words, "gfx1250_dense_record_replay_vgpr_msb");

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 80;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_dynamic_access_records = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(kAccessCount, 0, 0, 0);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.max_patches = kAccessCount;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  ASSERT_EQ(result.moi_candidates.size(), kAccessCount);
  for (const ConSanMoiCandidate &candidate : result.moi_candidates)
    EXPECT_EQ(candidate.gfx1250_vgpr_msb_mode, kGuestVgprMsbMode);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const uint32_t select_low =
      *build_gfx1250_s_set_vgpr_msb_transition(kGuestVgprMsbMode, 0u, ROCJITSU_CODE_ARCH_CDNA5);
  const uint32_t restore_guest =
      *build_gfx1250_s_set_vgpr_msb_transition(0u, kGuestVgprMsbMode, ROCJITSU_CODE_ARCH_CDNA5);
  uint32_t checked = 0;
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.kind != ConSanPatchKind::TrampolineMoiAccessRecordStore)
      continue;
    const std::vector<uint32_t> words =
        text_words_at_offset(patched, patch.trampoline_offset, patch.trampoline_size);
    ASSERT_GE(words.size(), 5u);
    EXPECT_EQ(words.front(), select_low);
    EXPECT_EQ(std::ranges::count(words, select_low), 2u);
    EXPECT_EQ(std::ranges::count(words, restore_guest), 2u);
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

TEST(ConSanMoi, Gfx1250RecordReplayCapturesHighBankLdsAddressBeforeSelectingScratchBank) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA5;
  constexpr uint8_t kGuestVgprMsbMode = 0x01u;
  constexpr uint16_t kEncodedAddressVgpr = 30u;
  std::vector<uint32_t> text_words = {
      *build_gfx1250_s_set_vgpr_msb(kGuestVgprMsbMode, kArch), 0xDBFC0000u,
      0x0000001Eu, // ds_load_b128 v[0:3], v30 /* physical v286 */
  };
  // Even when an inline body would fit in nearby padding, capturing an
  // address from a high SRC0 bank requires an appended body that can copy the
  // encoded operand before selecting the low scratch bank.
  text_words.insert(text_words.end(), 512u, build_s_nop(0u, kArch));
  text_words.push_back(build_s_endpgm(kArch));

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.scratch_vgpr = 8u;
  options.moi_exec_save_sgpr = 30u;
  options.moi_owner_vgpr = 40u;
  options.moi_epoch_vgpr = 41u;
  options.moi_dynamic_access_records = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(8u, 0u, 0u, 0u);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.max_patches = 1u;

  const ConSanResult result = try_patch_consan(
      make_gfx1250_code_object(text_words, "gfx1250_record_replay_high_bank_address"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.moi_candidates.size(), 1u);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.moi_candidates.front().gfx1250_vgpr_msb_mode, kGuestVgprMsbMode);
  EXPECT_EQ(result.resource_plans.front().scratch_vgpr_count, 7u);
  const ConSanPatchInfo &patch = only_non_entry_prologue_patch(result);
  ASSERT_EQ(patch.kind, ConSanPatchKind::TrampolineMoiAccessRecordStore);
  ASSERT_TRUE(patch.scratch_vgpr);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> cave =
      text_words_at_offset(patched, patch.trampoline_offset, patch.trampoline_size);
  const uint16_t captured_address_vgpr = static_cast<uint16_t>(*patch.scratch_vgpr + 6u);
  const uint32_t capture =
      build_v_mov_b32_e32(captured_address_vgpr, vector_source_vgpr(kEncodedAddressVgpr), kArch);
  const uint32_t select_low =
      *build_gfx1250_s_set_vgpr_msb_transition(kGuestVgprMsbMode, 0u, kArch);
  const uint32_t restore_guest =
      *build_gfx1250_s_set_vgpr_msb_transition(0u, kGuestVgprMsbMode, kArch);
  ASSERT_GE(cave.size(), 6u);
  EXPECT_EQ(cave[0], capture);
  EXPECT_EQ(cave[1], select_low);
  EXPECT_EQ(cave[2], restore_guest);
  EXPECT_EQ(cave[3], text_words[1]);
  EXPECT_EQ(cave[4], text_words[2]);
  EXPECT_EQ(cave[5], select_low);

  const uint16_t value_vgpr = static_cast<uint16_t>(*patch.scratch_vgpr + 5u);
  const auto start_cell = build_v_lshrrev_b32_e32(
      value_vgpr, scalar_positive_inline_u32(consan_moi_exact_shadow::granule_shift),
      captured_address_vgpr, kArch);
  ASSERT_TRUE(start_cell);
  EXPECT_NE(std::ranges::find(cave, *start_cell), cave.end());
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, Gfx1250RecordReplaySavesSpillBeforeCompositeHighBankAddressCapture) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA5;
  constexpr uint8_t kGuestVgprMsbMode = 0x41u;
  constexpr uint8_t kAddressCaptureMode = 0x01u;
  constexpr uint16_t kEncodedAddressVgpr = 30u;
  const std::vector<uint32_t> text_words = {
      *build_gfx1250_s_set_vgpr_msb(kGuestVgprMsbMode, kArch),
      0xDBFC0000u,
      0x0000001Eu, // ds_load_b128 v[0:3] /* v[256:259] */, v30 /* v286 */
      build_s_endpgm(kArch),
  };

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_exec_save_sgpr = 30u;
  options.moi_owner_vgpr = 40u;
  options.moi_epoch_vgpr = 41u;
  options.moi_dynamic_access_records = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(8u, 0u, 0u, 0u);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.force_vgpr_spill = true;
  options.max_patches = 1u;

  const ConSanResult result = try_patch_consan(
      make_gfx1250_code_object(text_words, "gfx1250_record_replay_composite_high_bank_address"),
      options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.moi_candidates.size(), 1u);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.moi_candidates.front().gfx1250_vgpr_msb_mode, kGuestVgprMsbMode);
  EXPECT_EQ(result.resource_plans.front().source, ConSanRegisterAllocationSource::SpillRequired);
  const ConSanPatchInfo &patch = only_non_entry_prologue_patch(result);
  ASSERT_EQ(patch.kind, ConSanPatchKind::TrampolineMoiAccessRecordStore);
  ASSERT_TRUE(patch.scratch_vgpr);
  ASSERT_EQ(patch.spilled_vgpr_count, 7u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> cave =
      text_words_at_offset(patched, patch.trampoline_offset, patch.trampoline_size);
  const uint16_t captured_address_vgpr = static_cast<uint16_t>(*patch.scratch_vgpr + 6u);
  const uint32_t select_low_for_spill =
      *build_gfx1250_s_set_vgpr_msb_transition(kGuestVgprMsbMode, 0u, kArch);
  const uint32_t select_address_for_capture =
      *build_gfx1250_s_set_vgpr_msb_transition(0u, kAddressCaptureMode, kArch);
  const uint32_t capture =
      build_v_mov_b32_e32(captured_address_vgpr, vector_source_vgpr(kEncodedAddressVgpr), kArch);
  const uint32_t clear_address_after_capture =
      *build_gfx1250_s_set_vgpr_msb_transition(kAddressCaptureMode, 0u, kArch);
  const uint32_t restore_guest =
      *build_gfx1250_s_set_vgpr_msb_transition(0u, kGuestVgprMsbMode, kArch);
  const uint32_t clear_guest =
      *build_gfx1250_s_set_vgpr_msb_transition(kGuestVgprMsbMode, 0u, kArch);
  const std::vector<uint32_t> save =
      expected_vgpr_spill_words(*patch.scratch_vgpr, 7u, /*restore=*/false, /*slot_base=*/0u);
  const std::vector<uint32_t> restore =
      expected_vgpr_spill_words(*patch.scratch_vgpr, 7u, /*restore=*/true, /*slot_base=*/0u);
  ASSERT_FALSE(save.empty());
  ASSERT_FALSE(restore.empty());
  ASSERT_GE(cave.size(), save.size() + 4u);
  EXPECT_EQ(cave.front(), select_low_for_spill);
  EXPECT_TRUE(std::equal(save.begin(), save.end(), cave.begin() + 1u));
  const size_t capture_begin = 1u + save.size();
  EXPECT_EQ(cave[capture_begin], select_address_for_capture);
  EXPECT_EQ(cave[capture_begin + 1u], capture);
  EXPECT_EQ(cave[capture_begin + 2u], clear_address_after_capture);

  const std::array<uint32_t, 2> guest_words = {text_words[1], text_words[2]};
  const auto guest = std::ranges::search(cave, guest_words).begin();
  const auto restored_spill = std::ranges::search(cave, restore).begin();
  ASSERT_NE(guest, cave.end());
  ASSERT_NE(restored_spill, cave.end());
  ASSERT_NE(guest, cave.begin());
  ASSERT_LT(guest + guest_words.size(), cave.end());
  EXPECT_EQ(*(guest - 1u), restore_guest);
  EXPECT_EQ(*(guest + guest_words.size()), clear_guest);
  EXPECT_LT(guest, restored_spill);
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, Gfx1250DenseAccessesIgnorePreviousGuestVgprMsbMode) {
  constexpr uint16_t kPreviousOnlyTransition = 0x4400u;
  std::vector<uint32_t> text_words(
      8u, build_s_mov_b32(/*sdst=*/0, /*ssrc0=*/0, ROCJITSU_CODE_ARCH_CDNA5));
  text_words.push_back(
      *build_gfx1250_s_set_vgpr_msb(kPreviousOnlyTransition, ROCJITSU_CODE_ARCH_CDNA5));
  text_words.push_back(0xD8340000u);
  text_words.push_back(0x00000000u); // ds_store_b32 v0, v0
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5));

  const ConSanResult result =
      try_patch_consan(make_gfx1250_code_object(text_words, "gfx1250_previous_vgpr_msb_mode"),
                       moi_options(ConSanMoiEngine::RecordReplay));

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_EQ(result.moi_candidates.size(), 1u);
  EXPECT_EQ(result.moi_candidates.front().gfx1250_vgpr_msb_mode, 0u);
}

TEST(ConSanMoi, AtomicRecordUsesLocalIndirectIslandForFarAppendedHelper) {
  constexpr size_t kLargeTextWords = 33000u;
  const auto atomic = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/2, /*vsrc=*/1, /*vdst=*/0, /*return_old_value=*/false, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(atomic);
  std::vector<uint32_t> text_words = {
      0xEE0B0000u,  0x00000000u,  0x00000000u, // global_wb
      (*atomic)[0], (*atomic)[1], (*atomic)[2], build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  text_words.resize(15u, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words.resize(kLargeTextWords - 1u, build_s_mov_b32(100, 100, ROCJITSU_CODE_ARCH_RDNA4));
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4));
  const uint64_t original_text_size = text_words.size() * sizeof(uint32_t);
  ASSERT_FALSE(compute_sopp_branch_simm16(3u * sizeof(uint32_t), original_text_size));
  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  mutate_elf_symbol(bytes, 1, [](Elf64_Sym &symbol) { symbol.st_size = 7u * sizeof(uint32_t); });

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_track_barriers = false;
  options.moi_track_atomics = true;
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 30;
  options.moi_owner_vgpr = 16;
  options.moi_epoch_vgpr = 17;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 0, 1, 1);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result))
      << "errors=" << testing::PrintToString(result.errors)
      << " warnings=" << testing::PrintToString(result.warnings)
      << " patches=" << testing::PrintToString(result.patches)
      << " fences=" << testing::PrintToString(result.moi_fence_candidates);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto island = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiIndirectBranchIsland, &ConSanPatchInfo::kind);
  const auto body = std::ranges::find(result.patches, ConSanPatchKind::TrampolineMoiAtomicRecord,
                                      &ConSanPatchInfo::kind);
  ASSERT_NE(island, result.patches.end());
  ASSERT_NE(body, result.patches.end());
  EXPECT_EQ(island->anchor_offset, 3u * sizeof(uint32_t));
  EXPECT_EQ(island->trampoline_offset, 7u * sizeof(uint32_t));
  EXPECT_EQ(island->trampoline_size, 8u * sizeof(uint32_t));
  EXPECT_EQ(body->trampoline_offset, original_text_size);
  EXPECT_TRUE(compute_sopp_branch_simm16(island->anchor_offset, island->trampoline_offset));
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  EXPECT_TRUE(patched.is_valid());
}

TEST(ConSanMoi, FenceRecordRelocatesSccDeadPrefixForCompactFarEntry) {
  constexpr size_t kLargeTextWords = 33000u;
  const auto atomic = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/2, /*vsrc=*/1, /*vdst=*/0, /*return_old_value=*/true, /*scope=*/3,
      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(atomic);
  std::vector<uint32_t> text_words = {
      0xEE0B007Cu,  0x000C0000u,  0x00000000u, // global_wb scope:system
      0xBFC30000u,                             // s_wait_bvhcnt 0
      0xBFC20000u,                             // s_wait_samplecnt 0
      0xBFC10000u,                             // s_wait_storecnt 0
      0xBFC80000u,                             // s_wait_loadcnt_dscnt 0
      (*atomic)[0], (*atomic)[1], (*atomic)[2],
  };
  // The atomic record patch consumes the only ordinary local indirect island.
  text_words.resize(18u, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words.resize(kLargeTextWords - 1u, build_s_mov_b32(100, 100, ROCJITSU_CODE_ARCH_RDNA4));
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4));
  const uint64_t original_text_size = text_words.size() * sizeof(uint32_t);
  ASSERT_FALSE(compute_sopp_branch_simm16(0u, original_text_size));
  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  mutate_elf_symbol(bytes, 1, [](Elf64_Sym &symbol) { symbol.st_size = 10u * sizeof(uint32_t); });

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_track_barriers = false;
  options.moi_track_atomics = true;
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 30;
  options.moi_owner_vgpr = 16;
  options.moi_epoch_vgpr = 17;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 0, 1, 1);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto fence = std::ranges::find(result.patches, ConSanPatchKind::TrampolineMoiFenceRecord,
                                       &ConSanPatchInfo::kind);
  ASSERT_NE(fence, result.patches.end()) << "warnings=" << testing::PrintToString(result.warnings)
                                         << " errors=" << testing::PrintToString(result.errors);
  EXPECT_EQ(fence->anchor_offset, 0u);
  EXPECT_EQ(fence->original_size, 7u * sizeof(uint32_t));
  const auto island = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiIndirectBranchIsland &&
           patch.anchor_offset == 0u && patch.trampoline_offset == 0u;
  });
  ASSERT_NE(island, result.patches.end());
  EXPECT_EQ(island->trampoline_size, fence->original_size);
  EXPECT_GE(fence->trampoline_offset, original_text_size);
  EXPECT_FALSE(compute_sopp_branch_simm16(fence->anchor_offset, fence->trampoline_offset));
  EXPECT_TRUE(result.final_validation_passed);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  EXPECT_TRUE(patched.is_valid());
}

TEST(ConSanMoi, AtomicRecordKeepsAcquireResultBeforeReporting) {
  const std::vector<uint8_t> bytes = make_rdna4_ordered_flat_atomic_release_acquire_code_object();
  ASSERT_FALSE(bytes.empty());
  ConSanOptions options = moi_options();
  options.moi_track_atomics = true;
  options.max_patches = 2;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 15;
  options.moi_epoch_vgpr = 16;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(2, 0, 0, 0, 0, 2, 2);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAtomicRecord,
                               &ConSanPatchInfo::kind),
            2);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const auto acquire_patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiAtomicRecord &&
           patch.anchor_offset == 6u * sizeof(uint32_t);
  });
  ASSERT_NE(acquire_patch, result.patches.end());
  const std::vector<uint32_t> words = text_words_at_offset(
      patched, acquire_patch->trampoline_offset, acquire_patch->trampoline_size);
  const auto original_acquire = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/4, /*vsrc=*/1, /*vdst=*/0, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(original_acquire);
  ASSERT_TRUE(acquire_patch->relocated_guest_instruction_offset);
  const size_t guest_word =
      (*acquire_patch->relocated_guest_instruction_offset - acquire_patch->trampoline_offset) /
      sizeof(uint32_t);
  ASSERT_GE(words.size(), guest_word + original_acquire->size() + 2u);
  EXPECT_TRUE(std::equal(original_acquire->begin(), original_acquire->end(),
                         words.begin() + static_cast<ptrdiff_t>(guest_word)));
  EXPECT_EQ(words[guest_word + original_acquire->size()],
            *instrumentation::build_s_wait_flat_load0(ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(ConSanMoi, RecordReplayNeutralizesWorkgroupReleaseScalarClauseAcrossRdnaTargets) {
  for (const rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA4}) {
    SCOPED_TRACE(arch == ROCJITSU_CODE_ARCH_RDNA3 ? "gfx1100" : "gfx1201");
    const RdnaWorkgroupClauseReleaseFixture fixture =
        make_rdna_workgroup_clause_release_code_object(arch);
    ASSERT_FALSE(fixture.bytes.empty());
    ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
    options.moi_track_barriers = false;
    options.moi_track_atomics = true;
    options.moi_init_owner_epoch = true;
    options.scratch_vgpr = 16u;
    options.moi_exec_save_sgpr = 30u;
    options.moi_owner_vgpr = 40u;
    options.moi_epoch_vgpr = 41u;
    options.moi_report_buffer_address = 0x123456780000ull;
    options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1u, 0u, 0u, 0u, 0u, 2u, 2u);
    options.max_patches = 1u;

    const ConSanResult result = try_patch_consan(fixture.bytes, options);

    ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
    ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
    EXPECT_TRUE(result.final_validation_passed);
    const auto clause_patch = std::ranges::find(
        result.patches, ConSanPatchKind::InlineScalarClauseNopRewrite, &ConSanPatchInfo::kind);
    const auto atomic_patch = std::ranges::find(
        result.patches, ConSanPatchKind::TrampolineMoiAtomicRecord, &ConSanPatchInfo::kind);
    ASSERT_NE(clause_patch, result.patches.end()) << testing::PrintToString(result.patches);
    ASSERT_NE(atomic_patch, result.patches.end()) << testing::PrintToString(result.warnings);
    EXPECT_EQ(clause_patch->anchor_offset, fixture.clause_text_offset);
    EXPECT_EQ(clause_patch->original_size, sizeof(uint32_t));
    EXPECT_EQ(atomic_patch->anchor_offset, fixture.release_text_offset);
    AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(patched.is_valid());
    const std::vector<uint32_t> clause =
        text_words_at_offset(patched, fixture.clause_text_offset, sizeof(uint32_t));
    ASSERT_EQ(clause.size(), 1u);
    EXPECT_EQ(clause.front(), build_s_nop(0, arch));
  }
}

TEST(ConSanMoi, AtomicRecordKeepsReturningReleaseAtEndOfProbe) {
  const std::vector<uint8_t> bytes =
      make_rdna4_ordered_flat_atomic_code_object(/*return_old_value=*/true);
  ASSERT_FALSE(bytes.empty());
  ConSanOptions options = moi_options();
  options.moi_track_atomics = true;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 15;
  options.moi_epoch_vgpr = 16;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 0, 1, 1);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  const auto patch = std::ranges::find(result.patches, ConSanPatchKind::TrampolineMoiAtomicRecord,
                                       &ConSanPatchInfo::kind);
  ASSERT_NE(patch, result.patches.end());
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> words =
      text_words_at_offset(patched, patch->trampoline_offset, patch->trampoline_size);
  const auto original_release = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/2, /*vsrc=*/1, /*vdst=*/0, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(original_release);
  ASSERT_GE(words.size(), original_release->size() + 1u);
  EXPECT_TRUE(std::equal(original_release->begin(), original_release->end(),
                         words.end() - original_release->size() - 1u));
}

TEST(ConSanMoi, FenceRecordPatchCardinalityIsBoundedAndPrefixComplete) {
  const std::vector<uint8_t> bytes = make_rdna4_atomic_fence_sequence_code_object();
  ASSERT_FALSE(bytes.empty());
  ConSanOptions options = moi_options();
  options.moi_track_atomics = true;
  options.max_patches = 1;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 30;
  options.moi_epoch_vgpr = 31;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(2, 0, 0, 0, 0, 2, 2);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result))
      << "errors=" << testing::PrintToString(result.errors)
      << " warnings=" << testing::PrintToString(result.warnings)
      << " patches=" << testing::PrintToString(result.patches)
      << " fences=" << testing::PrintToString(result.moi_fence_candidates);
  const auto fence = std::ranges::find(result.patches, ConSanPatchKind::TrampolineMoiFenceRecord,
                                       &ConSanPatchInfo::kind);
  ASSERT_NE(fence, result.patches.end())
      << "warnings=" << testing::PrintToString(result.warnings)
      << " patches=" << testing::PrintToString(result.patches)
      << " fences=" << testing::PrintToString(result.moi_fence_candidates)
      << " plans=" << testing::PrintToString(result.resource_plans);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiFenceRecord,
                               &ConSanPatchInfo::kind),
            1);
  EXPECT_EQ(fence->anchor_offset, result.moi_fence_candidates.front().text_offset);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> words =
      text_words_at_offset(patched, fence->trampoline_offset, fence->trampoline_size);
  EXPECT_TRUE(contains_subsequence(
      words,
      make_expected_fetch_add_one_words(
          *options.moi_report_buffer_address + offsetof(ConSanMoiReportHeader, fence_record_count),
          static_cast<uint16_t>(*options.scratch_vgpr + 2u), *options.scratch_vgpr)));
}

TEST(ConSanMoi, FenceRecordAcceptsSupportedRdna4OrdinaryAcquireAddress) {
  const std::vector<uint8_t> bytes = make_rdna4_ordinary_acquire_code_object();
  ASSERT_FALSE(bytes.empty());
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_track_barriers = false;
  options.moi_track_atomics = true;
  options.scratch_vgpr = 16;
  options.moi_owner_vgpr = 30;
  options.moi_epoch_vgpr = 31;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 0, 1, 1);

  const ConSanResult result = try_patch_consan(bytes, options);

  SCOPED_TRACE(testing::PrintToString(result.warnings));
  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.kernels.front().ordinary_memory_sites.size(), 1u);
  EXPECT_EQ(result.moi_fence_candidates.size(), 1u);
  EXPECT_EQ(result.site_dispositions.size(), 1u);
  EXPECT_EQ(result.kernels.front().ordinary_memory_sites.front().support_reason,
            ConSanOrdinaryMemorySupportReason::Supported);
  EXPECT_EQ(result.site_dispositions.front().disposition, ConSanSiteDisposition::Supported);
  EXPECT_EQ(result.site_dispositions.front().reason, ConSanSiteDispositionReason::None);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiFenceRecord,
                               &ConSanPatchInfo::kind),
            1);
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, Cdna4FenceRecordUsesCacheOrderingWithoutRdnaTh) {
  const auto release = cdna4::build_mubuf(cdna4::kBufferWbl2Mubuf, {.sc1 = 1});
  const auto acquire = cdna4::build_mubuf(cdna4::kBufferInvMubuf, {.sc1 = 1});
  const auto atomic = build_cdna4_flat_atomic_add_u32(
      /*vaddr=*/2, /*vsrc=*/4, /*vdst=*/5, /*return_old_value=*/true,
      /*scope=*/2, ROCJITSU_CODE_ARCH_CDNA4);
  const auto wait = build_cdna4_s_wait_flat0(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(atomic && wait);
  std::vector<uint32_t> text_words;
  text_words.insert(text_words.end(), release.begin(), release.end());
  text_words.push_back(*wait);
  text_words.insert(text_words.end(), atomic->begin(), atomic->end());
  text_words.push_back(*wait);
  text_words.insert(text_words.end(), acquire.begin(), acquire.end());
  text_words.resize(800, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4));
  const std::vector<uint8_t> bytes =
      make_cdna4_lds_code_object(text_words, "atomic_acquire_release");
  ConSanOptions options = moi_options();
  options.moi_track_atomics = true;
  options.max_patches = 1;
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 80;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 0, 1, 1);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << "warnings=" << testing::PrintToString(result.warnings)
                               << " errors=" << testing::PrintToString(result.errors);
  const auto fence = std::ranges::find(result.patches, ConSanPatchKind::TrampolineMoiFenceRecord,
                                       &ConSanPatchInfo::kind);
  ASSERT_NE(fence, result.patches.end()) << "warnings=" << testing::PrintToString(result.warnings)
                                         << " errors=" << testing::PrintToString(result.errors);
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, FenceRecordPatchRejectsStaleCommunicationIdentityWithoutGuessing) {
  const std::vector<uint8_t> bytes = make_rdna4_atomic_fence_sequence_code_object();
  ASSERT_FALSE(bytes.empty());
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::Moi;
  ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  ASSERT_TRUE(inventory.errors.empty());
  ASSERT_EQ(inventory.moi_fence_candidates.size(), 2u);
  for (ConSanMoiFenceCandidate &candidate : inventory.moi_fence_candidates) {
    ASSERT_TRUE(candidate.eligible);
    candidate.communication_event_identity += "|stale";
  }

  ConSanOptions options = moi_options();
  options.moi_track_atomics = true;
  options.max_patches = 3;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(3, 0, 0, 0, 0, 3, 3);

  const ConSanResult result =
      try_patch_consan_moi(std::move(inventory), options, bytes, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiFenceRecord,
                               &ConSanPatchInfo::kind),
            0);
  EXPECT_EQ(std::ranges::count(result.resource_plans, ConSanResourceSiteKind::Fence,
                               &ConSanCandidateResourcePlan::site_kind),
            0);
  EXPECT_EQ(std::ranges::count(result.site_dispositions, ConSanResourceSiteKind::Fence,
                               &ConSanSiteDispositionRecord::site_kind),
            2);
  EXPECT_TRUE(std::ranges::all_of(result.site_dispositions, [](const auto &site) {
    return site.site_kind != ConSanResourceSiteKind::Fence ||
           (site.disposition == ConSanSiteDisposition::Unsupported &&
            site.reason == ConSanSiteDispositionReason::MissingCommunicationEvent);
  }));
  EXPECT_NE(std::ranges::find(result.warnings,
                              "ConSan MOI fence record patch rejected all qualified "
                              "communication events"),
            result.warnings.end());
}

TEST(ConSanMoi, FenceRecordTreatsUnownedRuntimeCommunicationAsNotApplicable) {
  const std::vector<uint8_t> bytes = make_rdna4_atomic_fence_sequence_code_object();
  ASSERT_FALSE(bytes.empty());
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::Moi;
  ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  ASSERT_TRUE(inventory.errors.empty());
  ASSERT_EQ(inventory.moi_fence_candidates.size(), 2u);
  for (const ConSanMoiFenceCandidate &candidate : inventory.moi_fence_candidates) {
    ASSERT_TRUE(candidate.eligible);
    const auto event = std::ranges::find(
        inventory.sync_events, candidate.communication_event_identity, &ConSanSyncEvent::identity);
    ASSERT_NE(event, inventory.sync_events.end());
    event->execution_owners.clear();
  }

  ConSanOptions options = moi_options();
  options.moi_track_atomics = true;
  options.max_patches = 3;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(3, 0, 0, 0, 0, 3, 3);

  const ConSanResult result =
      try_patch_consan_moi(std::move(inventory), options, bytes, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiFenceRecord,
                               &ConSanPatchInfo::kind),
            0);
  EXPECT_EQ(std::ranges::count(result.resource_plans, ConSanResourceSiteKind::Fence,
                               &ConSanCandidateResourcePlan::site_kind),
            0);
  EXPECT_EQ(std::ranges::count(result.site_dispositions, ConSanResourceSiteKind::Fence,
                               &ConSanSiteDispositionRecord::site_kind),
            2);
  EXPECT_TRUE(std::ranges::all_of(result.site_dispositions, [](const auto &site) {
    return site.site_kind != ConSanResourceSiteKind::Fence ||
           (site.disposition == ConSanSiteDisposition::NotApplicable &&
            site.reason == ConSanSiteDispositionReason::MissingCommunicationEvent);
  }));
}

TEST(ConSanMoi, AtomicRecordMarksCompareExchangeOutcomeUnavailableUntilCaptured) {
  const std::vector<uint8_t> bytes = make_rdna4_ordered_flat_cas_code_object();
  ASSERT_FALSE(bytes.empty());
  ConSanOptions options = moi_options();
  options.moi_track_atomics = true;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 16;
  options.moi_epoch_vgpr = 17;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 0, 1, 1);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto atomic_patch = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiAtomicRecord, &ConSanPatchInfo::kind);
  ASSERT_NE(atomic_patch, result.patches.end())
      << testing::PrintToString(result.warnings) << testing::PrintToString(result.errors)
      << testing::PrintToString(result.resource_plans);
  const ConSanPatchInfo &patch = *atomic_patch;
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> words =
      text_words_at_offset(patched, patch.trampoline_offset, patch.trampoline_size);
  const uint64_t base = *options.moi_report_buffer_address;
  EXPECT_TRUE(contains_subsequence(
      words, make_expected_fetch_add_one_words(
                 base + offsetof(ConSanMoiReportHeader, atomic_record_count),
                 static_cast<uint16_t>(*options.scratch_vgpr + 2u), *options.scratch_vgpr)));
  const auto compare = build_v_cmp_eq_u32_e32_vcc(vector_source_vgpr(/*compare_vgpr=*/2),
                                                  /*old_value_vgpr=*/0, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(compare);
  const auto original_cas = build_flat_atomic_cmpswap_b32_vaddr_vsrc_vdst(
      /*vaddr=*/4, /*vsrc=*/1, /*vdst=*/0, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(original_cas);
  const auto guest =
      std::search(words.begin(), words.end(), original_cas->begin(), original_cas->end());
  const std::vector<uint32_t> reserve_event = make_expected_fetch_add_one_words(
      base + offsetof(ConSanMoiReportHeader, event_counter),
      static_cast<uint16_t>(*options.scratch_vgpr + 7u), *options.scratch_vgpr);
  const auto event_reservation =
      std::search(words.begin(), words.end(), reserve_event.begin(), reserve_event.end());
  const auto outcome_compare = std::find(words.begin(), words.end(), *compare);
  ASSERT_NE(guest, words.end());
  ASSERT_NE(event_reservation, words.end());
  ASSERT_NE(outcome_compare, words.end());
  EXPECT_LT(event_reservation + reserve_event.size(), guest);
  EXPECT_LT(guest + original_cas->size(), outcome_compare);
  const uint32_t capture_success_lo = build_v_mov_b32_e32(
      static_cast<uint16_t>(*options.scratch_vgpr + 3u), kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
  const uint32_t capture_success_hi =
      build_v_mov_b32_e32(static_cast<uint16_t>(*options.scratch_vgpr + 4u), kRdna4VccLo + 1u,
                          ROCJITSU_CODE_ARCH_RDNA4);
  const auto captured_lo = std::find(outcome_compare, words.end(), capture_success_lo);
  const auto captured_hi = std::find(outcome_compare, words.end(), capture_success_hi);
  ASSERT_NE(captured_lo, words.end());
  ASSERT_NE(captured_hi, words.end());
  EXPECT_LT(outcome_compare, captured_lo);
  EXPECT_LT(captured_lo, captured_hi);
  const auto store_success_lo = instrumentation::build_flat_store_b32(
      *options.scratch_vgpr, static_cast<uint16_t>(*options.scratch_vgpr + 3u),
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto store_success_hi = instrumentation::build_flat_store_b32(
      *options.scratch_vgpr, static_cast<uint16_t>(*options.scratch_vgpr + 4u),
      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(store_success_lo && store_success_hi);
  EXPECT_NE(
      std::search(captured_hi, words.end(), store_success_lo->begin(), store_success_lo->end()),
      words.end());
  EXPECT_NE(
      std::search(captured_hi, words.end(), store_success_hi->begin(), store_success_hi->end()),
      words.end());
}

TEST(ConSanMoi, AtomicRecordRejectsNoReturnCasWithTypedOutcomeReason) {
  const std::vector<uint8_t> bytes =
      make_rdna4_ordered_flat_cas_code_object(/*return_old_value=*/false);
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_track_atomics = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 0, 1, 1);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_EQ(std::ranges::count_if(result.patches,
                                  [](const ConSanPatchInfo &patch) {
                                    return patch.kind == ConSanPatchKind::TrampolineMoiAtomicRecord;
                                  }),
            0);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("compare-exchange-outcome-unavailable") != std::string::npos;
  }));
}

TEST(ConSanMoi, AtomicRecordAutomaticallyPlansScratch) {
  const std::vector<uint8_t> bytes = make_rdna4_ordered_flat_atomic_code_object();
  ConSanOptions options = moi_options();
  options.moi_track_atomics = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 0, 1, 1);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &item) {
    return item.kind == ConSanPatchKind::TrampolineMoiAtomicRecord;
  });
  ASSERT_NE(patch, result.patches.end());
  ASSERT_TRUE(patch->scratch_vgpr);
  EXPECT_GE(*patch->scratch_vgpr, 4u);
  const auto plan = std::ranges::find_if(result.resource_plans, [](const auto &item) {
    return item.site_kind == ConSanResourceSiteKind::Atomic;
  });
  ASSERT_NE(plan, result.resource_plans.end());
  EXPECT_EQ(plan->scratch_vgpr, patch->scratch_vgpr);
  const auto fence_patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &item) {
    return item.kind == ConSanPatchKind::TrampolineMoiFenceRecord;
  });
  ASSERT_NE(fence_patch, result.patches.end());
  ASSERT_TRUE(fence_patch->scratch_vgpr);
  const auto fence_plan =
      std::ranges::find_if(result.resource_plans, [](const ConSanCandidateResourcePlan &item) {
        return item.site_kind == ConSanResourceSiteKind::Fence;
      });
  ASSERT_NE(fence_plan, result.resource_plans.end());
  EXPECT_EQ(fence_plan->scratch_vgpr, fence_patch->scratch_vgpr);
  EXPECT_EQ(result.resource_plan_summary.dead_plans, 2u);
}

TEST(ConSanMoi, AtomicRecordForcedSpillUsesPlannedPrivateWindow) {
  const std::vector<uint8_t> bytes = make_rdna4_ordered_flat_atomic_code_object();
  ConSanOptions options = moi_options();
  options.moi_track_atomics = true;
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 0, 1, 1);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &item) {
    return item.kind == ConSanPatchKind::TrampolineMoiAtomicRecord;
  });
  ASSERT_NE(patch, result.patches.end());
  EXPECT_EQ(patch->spilled_vgpr_count, 7u);
  EXPECT_GT(patch->required_private_segment_size, 0u);
  const auto fence_patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &item) {
    return item.kind == ConSanPatchKind::TrampolineMoiFenceRecord;
  });
  EXPECT_NE(fence_patch, result.patches.end());
  EXPECT_EQ(std::ranges::find_if(
                result.warnings,
                [](const std::string &warning) {
                  return warning.find(
                             "does not support spill resources across the second text-growth "
                             "pass") != std::string::npos;
                }),
            result.warnings.end());
  EXPECT_FALSE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("fence record patch rejected communication address: "
                        "scratch-operand-alias") != std::string::npos;
  }));
  EXPECT_EQ(result.resource_plan_summary.spill_plans, 2u);
  EXPECT_EQ(result.resource_plan_summary.emitted_spill_patches, 2u);
  size_t emitted_spill_slot_bytes = 0u;
  for (const ConSanPatchInfo &item : result.patches) {
    emitted_spill_slot_bytes += static_cast<size_t>(item.spilled_vgpr_count) * sizeof(uint32_t);
  }
  EXPECT_EQ(result.resource_plan_summary.emitted_spill_slot_bytes, emitted_spill_slot_bytes);
}

TEST(ConSanMoi, Rdna4DynamicStackAtomicRecordUsesSiteLocalSpillFrames) {
  std::vector<uint8_t> bytes = make_rdna4_ordered_flat_atomic_code_object();
  append_kernel_metadata_note(bytes, "lds_probe", /*uses_dynamic_stack=*/true,
                              /*sgpr_count=*/0u);
  ConSanOptions options = moi_options();
  options.moi_track_atomics = true;
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 0, 1, 1);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result))
      << "warnings=" << testing::PrintToString(result.warnings)
      << " errors=" << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto atomic = std::ranges::find(result.patches, ConSanPatchKind::TrampolineMoiAtomicRecord,
                                        &ConSanPatchInfo::kind);
  ASSERT_NE(atomic, result.patches.end());
  EXPECT_EQ(atomic->spilled_vgpr_count, 7u);
  EXPECT_EQ(atomic->dynamic_private_segment_addend, 28u);
  const auto fence = std::ranges::find(result.patches, ConSanPatchKind::TrampolineMoiFenceRecord,
                                       &ConSanPatchInfo::kind);
  ASSERT_NE(fence, result.patches.end());
  EXPECT_GT(fence->spilled_vgpr_count, 0u);
  EXPECT_EQ(fence->dynamic_private_segment_addend, fence->spilled_vgpr_count * sizeof(uint32_t));
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, AtomicRecordSpillsSpecialStateOnRdna4) {
  std::vector<uint32_t> words = {
      0xBFC90000u, // s_wait_storecnt_dscnt 0
      0xEE0F0006u, 0x00880000u,
      0x00000000u, // global_atomic_and_b32 v0, v1, s[6:7], no-return, device
      0xEE0EC07Cu, 0x05080000u,
      0x00001408u, // global_atomic_max_u32 v[8:9], v10, off offset:20, device
      0xEE0AC000u, 0x00000000u,
      0x00000000u, // global_inv scope:device
  };
  const std::array<uint16_t, 4> dead_router_sgprs = {0u, 1u, 4u, 5u};
  for (uint16_t sgpr = 0; sgpr < 106u; ++sgpr) {
    if (std::ranges::find(dead_router_sgprs, sgpr) != dead_router_sgprs.end())
      continue;
    const auto use =
        build_s_cmp_eq_u32(sgpr, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_TRUE(use);
    words.push_back(*use);
  }
  words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4));

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_track_barriers = false;
  options.moi_track_atomics = true;
  options.max_patches = 2u;
  options.scratch_vgpr = 16u;
  options.moi_owner_vgpr = 30u;
  options.moi_epoch_vgpr = 31u;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(2, 0, 0, 0, 0, 2, 2);

  const ConSanResult result =
      try_patch_consan(make_rdna4_lds_code_object(words, "rdna4_atomic_scalar_spill"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_EQ(result.resolved_moi_transient_sgpr_assignments.size(), 1u);
  const ConSanMoiTransientSgprAssignment &assignment =
      result.resolved_moi_transient_sgpr_assignments.front();
  EXPECT_TRUE(assignment.spill_backed);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAtomicRecord,
                               &ConSanPatchInfo::kind),
            2u);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiFenceRecord,
                               &ConSanPatchInfo::kind),
            1u);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.kind != ConSanPatchKind::TrampolineMoiAtomicRecord &&
        patch.kind != ConSanPatchKind::TrampolineMoiFenceRecord)
      continue;
    EXPECT_EQ(patch.required_private_segment_size, 0u);
    expect_lane_backed_scalar_spill(result, patched, patch, assignment.exec_save_sgpr,
                                    ROCJITSU_CODE_ARCH_RDNA4);
  }
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, Cdna3RecordReplayAtomicEmitsValidatedNativeTransaction) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA3;
  const auto release = cdna3::build_mubuf(cdna3::kBufferWbl2Mubuf, {.sc1 = 1});
  const auto acquire = build_cdna3_buffer_inv_sc1(kArch);
  const auto atomic = build_cdna3_flat_atomic_add_u32(
      /*vaddr=*/2, /*vsrc=*/4, /*vdst=*/5, /*return_old_value=*/true,
      /*scope=*/2, kArch);
  const auto wait = build_cdna3_s_wait_vmcnt_lgkmcnt0(kArch);
  ASSERT_TRUE(acquire && atomic && wait);
  std::vector<uint32_t> text_words;
  text_words.insert(text_words.end(), release.begin(), release.end());
  text_words.push_back(*wait);
  text_words.insert(text_words.end(), atomic->begin(), atomic->end());
  text_words.push_back(*wait);
  text_words.insert(text_words.end(), acquire->begin(), acquire->end());
  text_words.resize(800, build_s_nop(0, kArch));
  text_words.push_back(build_s_endpgm(kArch));
  const std::vector<uint8_t> bytes = make_cdna3_lds_code_object(text_words, "atomic_record_native");
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_track_barriers = false;
  options.moi_track_atomics = true;
  options.scratch_vgpr = 16u;
  options.moi_owner_vgpr = 30u;
  options.moi_epoch_vgpr = 31u;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 0, 1, 1);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_EQ(result.target, ROCJITSU_CODE_TARGET_GFX942);
  EXPECT_EQ(result.arch, ROCJITSU_CODE_ARCH_CDNA3);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAtomicRecord,
                               &ConSanPatchInfo::kind),
            1u);
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, Cdna4AtomicRecordForcedSpillUsesNativePrivateWindow) {
  const auto release = cdna4::build_mubuf(cdna4::kBufferWbl2Mubuf, {.sc1 = 1});
  const auto acquire = cdna4::build_mubuf(cdna4::kBufferInvMubuf, {.sc1 = 1});
  const auto atomic = build_cdna4_flat_atomic_add_u32(
      /*vaddr=*/2, /*vsrc=*/4, /*vdst=*/5, /*return_old_value=*/true,
      /*scope=*/2, ROCJITSU_CODE_ARCH_CDNA4);
  const auto wait = build_cdna4_s_wait_flat0(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(atomic && wait);
  std::vector<uint32_t> text_words;
  text_words.insert(text_words.end(), release.begin(), release.end());
  text_words.push_back(*wait);
  text_words.insert(text_words.end(), atomic->begin(), atomic->end());
  text_words.push_back(*wait);
  text_words.insert(text_words.end(), acquire.begin(), acquire.end());
  text_words.resize(800, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  text_words.push_back(
      build_v_mov_b32_e32(/*vdst=*/0, vector_source_vgpr(255), ROCJITSU_CODE_ARCH_CDNA4));
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4));
  constexpr uint32_t kCdna4Wave64AllVgprsGranulated = 31u;
  const std::vector<uint8_t> bytes = make_cdna4_lds_code_object(
      text_words, "atomic_record_forced_spill", kCdna4Wave64AllVgprsGranulated);
  ConSanOptions options = moi_options();
  options.moi_track_atomics = true;
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 0, 1, 1);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << "warnings=" << testing::PrintToString(result.warnings)
                               << " errors=" << testing::PrintToString(result.errors);
  EXPECT_EQ(result.target, ROCJITSU_CODE_TARGET_GFX950);
  EXPECT_EQ(result.arch, ROCJITSU_CODE_ARCH_CDNA4);
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_TRUE(result.final_validation_passed);
  const auto patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &item) {
    return item.kind == ConSanPatchKind::TrampolineMoiAtomicRecord;
  });
  ASSERT_NE(patch, result.patches.end()) << "patches=" << testing::PrintToString(result.patches)
                                         << " warnings=" << testing::PrintToString(result.warnings);
  EXPECT_EQ(patch->spilled_vgpr_count, 7u);
  EXPECT_EQ(patch->required_private_segment_size, 32u);
  const auto plan = std::ranges::find_if(result.resource_plans, [](const auto &item) {
    return item.site_kind == ConSanResourceSiteKind::Atomic;
  });
  ASSERT_NE(plan, result.resource_plans.end());
  EXPECT_EQ(plan->source, ConSanRegisterAllocationSource::SpillRequired);
  EXPECT_EQ(plan->reason, ConSanRegisterPlanReason::None);
  EXPECT_EQ(plan->scratch_vgpr_count, 7u);
  ASSERT_TRUE(plan->scratch_vgpr);
  EXPECT_EQ(*plan->scratch_vgpr % 2u, 0u);
  EXPECT_EQ(result.resource_plan_summary.spill_plans, 3u);
  EXPECT_EQ(result.resource_plan_summary.emitted_spill_patches, 2u);
  EXPECT_EQ(result.resource_plan_summary.emitted_spill_slot_bytes, 60u);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiFenceRecord,
                               &ConSanPatchInfo::kind),
            1u);
  EXPECT_FALSE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("does not support spill resources across the second text-growth pass") !=
           std::string::npos;
  }));
  EXPECT_FALSE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("fence record patch rejected communication address: "
                        "scratch-operand-alias") != std::string::npos;
  }));
  EXPECT_EQ(std::ranges::count(result.site_dispositions, ConSanResourceSiteKind::Fence,
                               &ConSanSiteDispositionRecord::site_kind),
            2);
  EXPECT_EQ(
      std::ranges::count_if(result.site_dispositions,
                            [](const auto &site) {
                              return site.site_kind == ConSanResourceSiteKind::Fence &&
                                     site.lowering_outcome ==
                                         ConSanSiteLoweringOutcome::PlacementOrLoweringFailed &&
                                     site.lowering_reason ==
                                         ConSanSiteLoweringReason::InstrumentationPatchMissing &&
                                     site.resource_reason == ConSanRegisterPlanReason::None;
                            }),
      1);
}

TEST(ConSanMoi, Cdna4AtomicRecordSpillsThroughSiteLocalDynamicStackFrame) {
  const auto release = cdna4::build_mubuf(cdna4::kBufferWbl2Mubuf, {.sc1 = 1});
  const auto acquire = cdna4::build_mubuf(cdna4::kBufferInvMubuf, {.sc1 = 1});
  const auto atomic = build_cdna4_flat_atomic_add_u32(
      /*vaddr=*/2, /*vsrc=*/4, /*vdst=*/5, /*return_old_value=*/true,
      /*scope=*/2, ROCJITSU_CODE_ARCH_CDNA4);
  const auto wait = build_cdna4_s_wait_flat0(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(atomic && wait);
  std::vector<uint32_t> text_words;
  text_words.push_back(build_s_mov_b32(/*sdst=*/18u, /*ssrc=*/33u, ROCJITSU_CODE_ARCH_CDNA4));
  text_words.push_back(build_s_mov_b32(/*sdst=*/33u, /*ssrc=*/32u, ROCJITSU_CODE_ARCH_CDNA4));
  text_words.insert(text_words.end(), release.begin(), release.end());
  text_words.push_back(*wait);
  text_words.insert(text_words.end(), atomic->begin(), atomic->end());
  text_words.push_back(*wait);
  text_words.insert(text_words.end(), acquire.begin(), acquire.end());
  text_words.push_back(build_s_mov_b32(/*sdst=*/33u, /*ssrc=*/18u, ROCJITSU_CODE_ARCH_CDNA4));
  text_words.resize(800, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  text_words.push_back(
      build_v_mov_b32_e32(/*vdst=*/0, vector_source_vgpr(255), ROCJITSU_CODE_ARCH_CDNA4));
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4));
  constexpr uint32_t kCdna4Wave64AllVgprsGranulated = 31u;
  std::vector<uint8_t> bytes = make_cdna4_lds_code_object(text_words, "atomic_record_dynamic_spill",
                                                          kCdna4Wave64AllVgprsGranulated,
                                                          /*uses_dynamic_stack=*/true);
  append_kernel_metadata_note(bytes, "atomic_record_dynamic_spill",
                              /*uses_dynamic_stack=*/true, /*sgpr_count=*/0u,
                              /*private_segment_fixed_size=*/20u);
  mutate_kernel_descriptor(bytes, "atomic_record_dynamic_spill", [](KD &descriptor) {
    descriptor.private_segment_fixed_size = 20u;
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT, 1u);
  });
  ConSanOptions options = moi_options();
  options.moi_track_atomics = true;
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 0, 1, 1);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << "warnings=" << testing::PrintToString(result.warnings)
                               << " errors=" << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.resolved_moi_exec_save_sgpr);
  ASSERT_TRUE(result.resolved_moi_persistent_owner_sgpr);
  ASSERT_TRUE(result.resolved_moi_persistent_epoch_sgpr);
  EXPECT_FALSE(result.resolved_moi_persistent_workgroup_key_sgpr);
  ASSERT_TRUE(result.resolved_moi_record_replay_workgroup_sgprs.complete());
  EXPECT_TRUE(result.moi_persistent_sgprs_automatic);
  EXPECT_TRUE(*result.resolved_moi_exec_save_sgpr + 6u <= 18u ||
              *result.resolved_moi_exec_save_sgpr > 18u);
  const auto patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &item) {
    return item.kind == ConSanPatchKind::TrampolineMoiAtomicRecord;
  });
  ASSERT_NE(patch, result.patches.end());
  EXPECT_EQ(patch->spilled_vgpr_count, 7u);
  EXPECT_EQ(patch->required_private_segment_size, 48u);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> cave_words =
      text_words_at_offset(patched, patch->trampoline_offset, patch->trampoline_size);
  const uint16_t saved_frame_sgpr = static_cast<uint16_t>(*result.resolved_moi_exec_save_sgpr + 5u);
  EXPECT_NE(
      std::find(cave_words.begin(), cave_words.end(),
                build_s_mov_b32(saved_frame_sgpr, /*frame base=*/33, ROCJITSU_CODE_ARCH_CDNA4)),
      cave_words.end());
  const uint16_t record_value_vgpr = static_cast<uint16_t>(patch->scratch_vgpr.value() + 4u);
  EXPECT_NE(
      std::find(cave_words.begin(), cave_words.end(),
                build_v_mov_b32_e32(record_value_vgpr, *result.resolved_moi_persistent_owner_sgpr,
                                    ROCJITSU_CODE_ARCH_CDNA4)),
      cave_words.end());
  EXPECT_NE(
      std::find(cave_words.begin(), cave_words.end(),
                build_v_mov_b32_e32(record_value_vgpr, *result.resolved_moi_persistent_epoch_sgpr,
                                    ROCJITSU_CODE_ARCH_CDNA4)),
      cave_words.end());
  const auto fence = std::ranges::find(result.patches, ConSanPatchKind::TrampolineMoiFenceRecord,
                                       &ConSanPatchInfo::kind);
  ASSERT_NE(fence, result.patches.end());
  ASSERT_TRUE(fence->scratch_vgpr);
  const std::vector<uint32_t> fence_words =
      text_words_at_offset(patched, fence->trampoline_offset, fence->trampoline_size);
  const uint16_t fence_value_vgpr = static_cast<uint16_t>(*fence->scratch_vgpr + 5u);
  EXPECT_NE(
      std::ranges::find(fence_words, build_v_mov_b32_e32(fence_value_vgpr,
                                                         *result.resolved_moi_persistent_owner_sgpr,
                                                         ROCJITSU_CODE_ARCH_CDNA4)),
      fence_words.end());
  EXPECT_NE(
      std::ranges::find(fence_words, build_v_mov_b32_e32(fence_value_vgpr,
                                                         *result.resolved_moi_persistent_epoch_sgpr,
                                                         ROCJITSU_CODE_ARCH_CDNA4)),
      fence_words.end());
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, AtomicRecordRetainsIsolatedNoReturnReleaseAndAccessReplay) {
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_store_and_release_wait_no_return_bitwise_code_object();
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_track_atomics = true;
  options.moi_dynamic_access_records = true;
  options.scratch_vgpr = 16;
  options.moi_exec_save_sgpr = 30;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(8, 0, 0, 0, 0, 1, 1);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAtomicRecord,
                               &ConSanPatchInfo::kind),
            1);
  EXPECT_EQ(
      std::ranges::count_if(result.patches,
                            [](const ConSanPatchInfo &patch) {
                              return patch.kind == ConSanPatchKind::InlineMoiAccessRecordStore ||
                                     patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore;
                            }),
      1);
  EXPECT_FALSE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("pruned isolated no-return release metadata") != std::string::npos;
  }));
}

TEST(ConSanMoi, Gfx1250OrderedLdsAtomicComposesAccessAndOrderingRecords) {
  const std::array<uint32_t, 7> words = {
      0x360202ffu, 0x000000ffu, // release wait setup
      0xbf94ffffu,              // s_barrier_wait -1
      0xbfc10000u,              // release ordering completion
      0xd8000000u, 0x00001210u, // ds_add_u32 v0, v18, no return
      0xbfb00000u,              // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_gfx1250_code_object(words, "ordered_lds_atomic");
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_track_atomics = true;
  options.moi_track_barriers = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(8, 8, 0, 0, 0, 8, 1);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result))
      << testing::PrintToString(result.errors) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::InlineMoiAccessRecordStore ||
           patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore;
  });
  const auto atomic = std::ranges::find(result.patches, ConSanPatchKind::TrampolineMoiAtomicRecord,
                                        &ConSanPatchInfo::kind);
  ASSERT_NE(access, result.patches.end());
  ASSERT_NE(atomic, result.patches.end());
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

TEST(ConSanMoi, FirstLightProbeRejectsScratchVgprsOverlappingLdsAddress) {
  std::array<uint32_t, 76> text_words{};
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000000u; // ds_store_b32 v0, v0
  for (size_t i = 2; i + 1 < text_words.size(); ++i)
    text_words[i] = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
  text_words.back() = 0xBFB00000u; // s_endpgm

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options = moi_options();
  options.scratch_vgpr = 0;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.errors.empty());
  bool saw_overlap_warning = false;
  for (const std::string &warning : result.warnings)
    saw_overlap_warning |= warning.find("scratch VGPRs overlap") != std::string::npos;
  EXPECT_TRUE(saw_overlap_warning);
}

TEST(ConSanMoi, RecordReplayAcquireReleaseImportsAndPublishesOrdering) {
  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/4,
      /*diagnostic_capacity=*/2, /*exact_shadow_entry_capacity=*/2,
      /*sampled_watchpoint_capacity=*/0);
  header.access_record_count = 4;

  std::array<ConSanMoiAccessRecord, 4> records{};
  records[0].wave_id = 0;
  records[0].event_index = 0;
  records[0].instruction_offset = 0x10;
  records[0].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
  records[0].lds_byte_count = 4;
  records[0].cell_count = 1;

  records[1].wave_id = 1;
  records[1].event_index = 2;
  records[1].instruction_offset = 0x20;
  records[1].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
  records[1].lds_byte_offset = 4;
  records[1].lds_byte_count = 4;
  records[1].start_cell = 1;
  records[1].cell_count = 1;

  records[2].wave_id = 1;
  records[2].event_index = 4;
  records[2].instruction_offset = 0x30;
  records[2].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read);
  records[2].lds_byte_count = 4;
  records[2].cell_count = 1;

  records[3].wave_id = 2;
  records[3].event_index = 6;
  records[3].instruction_offset = 0x40;
  records[3].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read);
  records[3].lds_byte_offset = 4;
  records[3].lds_byte_count = 4;
  records[3].start_cell = 1;
  records[3].cell_count = 1;

  std::array<ConSanMoiRecordReplayAtomicEvent, 3> atomics{};
  atomics[0].owner_id = 0;
  atomics[0].atomic_address = 0x4000;
  atomics[0].instruction_offset = 0x100;
  atomics[0].event_index = 1;
  atomics[0].kind = ConSanMoiAtomicEventKind::Release;

  atomics[1].owner_id = 1;
  atomics[1].atomic_address = 0x4000;
  atomics[1].instruction_offset = 0x200;
  atomics[1].event_index = 3;
  atomics[1].kind = ConSanMoiAtomicEventKind::AcquireRelease;

  atomics[2].owner_id = 2;
  atomics[2].atomic_address = 0x4000;
  atomics[2].instruction_offset = 0x300;
  atomics[2].event_index = 5;
  atomics[2].kind = ConSanMoiAtomicEventKind::Acquire;

  std::array<ConSanMoiDiagnosticRecord, 2> diagnostics{};
  std::array<uint64_t, 2> shadow{};

  const ConSanMoiRecordReplayResult replay = consan_moi_record_replay_access_records(
      header, records, std::span<const ConSanMoiBarrierRecord>{}, atomics, diagnostics, shadow);

  EXPECT_EQ(replay.processed_access_count, 4u);
  EXPECT_EQ(replay.processed_atomic_count, 3u);
  EXPECT_EQ(replay.unsupported_atomic_count, 0u);
  EXPECT_FALSE(replay.metadata_full);
  EXPECT_FALSE(replay.conflict);
  EXPECT_EQ(header.diagnostic_count, 0u);
}

TEST(ConSanMoi, RecordReplayCompareExchangePublishesOnlyOnSuccess) {
  auto replay_outcome = [](ConSanMoiAtomicOutcome outcome, uint64_t lane_mask = 0,
                           uint64_t success_lane_mask = 0) {
    ConSanMoiReportHeader header = make_consan_moi_report_header(
        /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/2,
        /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/1,
        /*sampled_watchpoint_capacity=*/0);
    header.access_record_count = 2;

    std::array<ConSanMoiAccessRecord, 2> records{};
    records[0].wave_id = 1;
    records[0].event_index = 0;
    records[0].instruction_offset = 0x10;
    records[0].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
    records[0].lds_byte_count = 4;
    records[0].cell_count = 1;
    records[1].wave_id = 2;
    records[1].event_index = 3;
    records[1].instruction_offset = 0x20;
    records[1].access_kind = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read);
    records[1].lds_byte_count = 4;
    records[1].cell_count = 1;

    std::array<ConSanMoiRecordReplayAtomicEvent, 2> atomics{};
    atomics[0].owner_id = 1;
    atomics[0].atomic_address = 0x4000;
    atomics[0].event_index = 1;
    atomics[0].kind = ConSanMoiAtomicEventKind::AcquireRelease;
    atomics[0].operation = ConSanMoiAtomicOperation::CompareExchange;
    atomics[0].outcome = outcome;
    atomics[0].lane_mask = lane_mask;
    atomics[0].success_lane_mask = success_lane_mask;
    atomics[1].owner_id = 2;
    atomics[1].atomic_address = 0x4000;
    atomics[1].event_index = 2;
    atomics[1].kind = ConSanMoiAtomicEventKind::Acquire;

    std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
    std::array<uint64_t, 1> shadow{};
    return consan_moi_record_replay_access_records(
        header, records, std::span<const ConSanMoiBarrierRecord>{}, atomics, diagnostics, shadow);
  };

  const ConSanMoiRecordReplayResult success = replay_outcome(ConSanMoiAtomicOutcome::Success);
  EXPECT_EQ(success.unsupported_atomic_count, 0u);
  EXPECT_FALSE(success.conflict);

  const ConSanMoiRecordReplayResult failure = replay_outcome(ConSanMoiAtomicOutcome::Failure);
  EXPECT_EQ(failure.unsupported_atomic_count, 0u);
  EXPECT_TRUE(failure.conflict);

  const ConSanMoiRecordReplayResult unavailable =
      replay_outcome(ConSanMoiAtomicOutcome::Unavailable);
  EXPECT_EQ(unavailable.unsupported_atomic_count, 1u);
  EXPECT_TRUE(unavailable.conflict);

  const ConSanMoiRecordReplayResult captured_success =
      replay_outcome(ConSanMoiAtomicOutcome::Unavailable, 0x3u, 0x3u);
  EXPECT_EQ(captured_success.unsupported_atomic_count, 0u);
  EXPECT_FALSE(captured_success.conflict);

  const ConSanMoiRecordReplayResult captured_failure =
      replay_outcome(ConSanMoiAtomicOutcome::Unavailable, 0x3u, 0u);
  EXPECT_EQ(captured_failure.unsupported_atomic_count, 0u);
  EXPECT_TRUE(captured_failure.conflict);

  const ConSanMoiRecordReplayResult mixed =
      replay_outcome(ConSanMoiAtomicOutcome::Unavailable, 0x3u, 0x1u);
  EXPECT_EQ(mixed.unsupported_atomic_count, 1u);
  EXPECT_TRUE(mixed.conflict);
}

} // namespace
} // namespace rocjitsu
