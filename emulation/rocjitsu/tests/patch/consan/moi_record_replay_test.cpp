// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"
#include "rocjitsu/code/patch/gfx1250_instrumentation_builder.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"

namespace rocjitsu {
namespace {

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
  words.push_back(0xBFC00000u);
  return words;
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
    EXPECT_EQ(plan.scratch_vgpr_count, plan_index == 0 ? 3u : 4u);
    EXPECT_EQ(plan.current_vgpr_count, 256);
    EXPECT_EQ(plan.max_referenced_vgpr_count, 1);
    EXPECT_EQ(plan.required_vgpr_count, 256);
    EXPECT_EQ(plan.original_private_segment_size, 0u);
  }
  EXPECT_FALSE(result.warnings.empty());
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

TEST(ConSanMoi, RecordReplayExcludesUnreachableTailOfFinalZeroSizedSymbol) {
  constexpr auto live_store =
      gfx1250::build_vds(gfx1250::kDsStoreB16Vds, {.addr = 4u, .data0 = 5u});
  constexpr auto dead_load = gfx1250::build_vds(gfx1250::kDsLoadB32Vds, {.addr = 6u, .vdst = 7u});
  const std::array<uint32_t, 6> text_words = {
      live_store[0], live_store[1], build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250),
      dead_load[0],  dead_load[1],  build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250),
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
  constexpr auto live_store =
      gfx1250::build_vds(gfx1250::kDsStoreB16Vds, {.addr = 4u, .data0 = 5u});
  constexpr auto dead_load = gfx1250::build_vds(gfx1250::kDsLoadB32Vds, {.addr = 6u, .vdst = 7u});
  const std::array<uint32_t, 6> first_kernel_words = {
      live_store[0], live_store[1], build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250),
      dead_load[0],  dead_load[1],  build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250),
  };
  const std::array<uint32_t, 1> second_kernel_words = {
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250),
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
  constexpr auto live_store =
      gfx1250::build_vds(gfx1250::kDsStoreB16Vds, {.addr = 4u, .data0 = 5u});
  constexpr auto dead_load = gfx1250::build_vds(gfx1250::kDsLoadB32Vds, {.addr = 6u, .vdst = 7u});
  const std::array<uint32_t, 6> text_words = {
      live_store[0], live_store[1], build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250),
      dead_load[0],  dead_load[1],  build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250),
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
  std::array<uint32_t, 170> text_words{};
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
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineMoiAccessRecordStore);
  EXPECT_EQ(result.patches.front().scratch_vgpr, 1);
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
  EXPECT_EQ(plan.required_vgpr_count, 7);
  EXPECT_EQ(result.resource_plan_summary.descriptor_growth_plans, 1u);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().scratch_vgpr, 4);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.kernels().size(), 1u);
  const uint64_t descriptor_offset = patched.kernels().front().descriptor_file_offset;
  KD descriptor{};
  std::memcpy(&descriptor, result.elf_bytes.data() + descriptor_offset, sizeof(descriptor));
  EXPECT_EQ(AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc1,
                            kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT),
            1u);
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
  ASSERT_EQ(result.patches.size(), 1u);
  const ConSanPatchInfo &patch = result.patches.front();
  EXPECT_EQ(patch.kind, ConSanPatchKind::TrampolineMoiAccessRecordStore);
  EXPECT_EQ(patch.scratch_vgpr, 1);
  EXPECT_EQ(patch.spilled_vgpr_count, 3u);
  EXPECT_EQ(patch.required_private_segment_size, 44u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const auto *text = patched.text_sections().front();
  std::vector<uint32_t> actual(text->size() / sizeof(uint32_t));
  std::memcpy(actual.data(), text->data(), text->size());
  const std::vector<uint32_t> save =
      expected_vgpr_spill_words(1, 3, /*restore=*/false, /*slot_base=*/32);
  const std::vector<uint32_t> restore =
      expected_vgpr_spill_words(1, 3, /*restore=*/true, /*slot_base=*/32);
  ASSERT_FALSE(save.empty());
  ASSERT_FALSE(restore.empty());
  const size_t cave = patch.trampoline_offset / sizeof(uint32_t);
  const auto owner_init =
      build_v_lshrrev_b32_e32(3, scalar_positive_inline_u32(6), 0, ROCJITSU_CODE_ARCH_RDNA4);
  const auto owner_mask = build_v_and_b32_e32_literal(3, consan_moi_exact_shadow::max_owner, 3,
                                                      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(owner_init);
  ASSERT_TRUE(owner_mask);
  ASSERT_LE(cave + save.size() + owner_mask->size() + restore.size() + 4u, actual.size());
  EXPECT_TRUE(std::equal(save.begin(), save.end(), actual.begin() + cave));
  EXPECT_TRUE(contains_subsequence(actual, std::span<const uint32_t>(&*owner_init, 1u)));
  EXPECT_TRUE(contains_subsequence(actual, *owner_mask));
  const size_t guest_offset = actual.size() - 1u - restore.size() - 2u;
  EXPECT_EQ(actual[guest_offset], text_words[0]);
  EXPECT_EQ(actual[guest_offset + 1u], text_words[1]);
  EXPECT_TRUE(std::equal(restore.begin(), restore.end(), actual.end() - 1u - restore.size()));

  ASSERT_EQ(patched.kernels().size(), 1u);
  KD descriptor{};
  const uint64_t descriptor_offset = patched.kernels().front().descriptor_file_offset;
  std::memcpy(&descriptor, result.elf_bytes.data() + descriptor_offset, sizeof(descriptor));
  EXPECT_EQ(descriptor.private_segment_fixed_size, 44u);
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
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().spilled_vgpr_count, 3u);
  EXPECT_EQ(result.patches.front().required_private_segment_size, 12u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.kernels().size(), 1u);
  KD descriptor{};
  std::memcpy(&descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  EXPECT_EQ(descriptor.private_segment_fixed_size, 12u);
  EXPECT_EQ(
      AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT),
      1u);
}

TEST(ConSanMoi, FirstLightProbeRejectsSpillingDynamicStackKernel) {
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

  EXPECT_TRUE(result.errors.empty());
  EXPECT_FALSE(result.modified);
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::Unsupported);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().source, ConSanRegisterAllocationSource::Unsupported);
  EXPECT_EQ(result.resource_plans.front().reason, ConSanRegisterPlanReason::DynamicStack);
  ASSERT_EQ(result.site_dispositions.size(), 1u);
  EXPECT_EQ(result.site_dispositions.front().lowering_outcome,
            ConSanSiteLoweringOutcome::ResourceFailed);
  EXPECT_EQ(result.site_dispositions.front().lowering_reason,
            ConSanSiteLoweringReason::UnsupportedResourcePlan);
  EXPECT_EQ(result.site_dispositions.front().resource_reason,
            ConSanRegisterPlanReason::DynamicStack);
  EXPECT_STREQ(consan_site_lowering_outcome_name(result.site_dispositions.front().lowering_outcome),
               "resource_failed");
  EXPECT_STREQ(consan_site_lowering_reason_name(result.site_dispositions.front().lowering_reason),
               "unsupported_resource_plan");
  EXPECT_EQ(result.resource_plan_summary.unsupported_plans, 1u);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("dynamic-stack") != std::string::npos;
  }));
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
  EXPECT_EQ(patch->spilled_vgpr_count, 3u);
  EXPECT_EQ(patch->required_private_segment_size, 16u);
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

TEST(ConSanMoi, Cdna4DynamicStackRejectsShortRecordReplayScalarSpillWindow) {
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
    return warning.find("no reserved frame-base save slot") != std::string::npos;
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
  EXPECT_EQ(patch->spilled_vgpr_count, 3u);
  EXPECT_EQ(patch->required_private_segment_size, 16u);
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
  std::array<uint32_t, 170> text_words{};
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
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineMoiAccessRecordStore);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 8u);
  EXPECT_EQ(result.patches.front().original_size, 116u * sizeof(uint32_t));
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 8u);
  EXPECT_EQ(result.resource_plan_summary.explicit_plans, 1u);

  const uint64_t base = *options.moi_report_buffer_address;
  const uint64_t access_record_base = base + sizeof(ConSanMoiReportHeader);
  const std::vector<uint32_t> rewritten_words =
      patched_words_at_file_offset(result, 0x100, result.patches.front().original_size);
  ASSERT_GE(rewritten_words.size(), 2u);
  EXPECT_EQ(rewritten_words[rewritten_words.size() - 2u], text_words[0]);
  EXPECT_EQ(rewritten_words.back(), text_words[1]);
  EXPECT_EQ(std::count(rewritten_words.begin(), rewritten_words.end(), 0xBFC60000u), 0u);

  const auto atomic = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      8, 10, 10, /*return_old_value=*/true, /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  const uint64_t publication_address =
      access_record_base + offsetof(ConSanMoiAccessRecord, access_kind);
  const auto publication_address_lo = build_v_mov_b32_e64_literal(
      8, static_cast<uint32_t>(publication_address), ROCJITSU_CODE_ARCH_RDNA4);
  const auto publication_address_hi = build_v_mov_b32_e64_literal(
      9, static_cast<uint32_t>(publication_address >> 32u), ROCJITSU_CODE_ARCH_RDNA4);
  const auto publication_load =
      build_flat_load_b32_vaddr_vdst(8, 10, ROCJITSU_CODE_ARCH_RDNA4, /*byte_offset=*/0);
  const auto publication_claim = build_flat_atomic_or_u32_vaddr_vsrc_vdst(
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
  ASSERT_TRUE(publication_load);
  ASSERT_TRUE(publication_claim);
  ASSERT_TRUE(canonical_owner);
  ASSERT_TRUE(record_address_lo);
  ASSERT_TRUE(record_address_hi);
  EXPECT_TRUE(contains_subsequence(rewritten_words, *atomic));
  EXPECT_TRUE(contains_subsequence(rewritten_words, *publication_address_lo));
  EXPECT_TRUE(contains_subsequence(rewritten_words, *publication_address_hi));
  EXPECT_TRUE(contains_subsequence(rewritten_words, *publication_load));
  EXPECT_EQ(count_subsequence(rewritten_words, *publication_claim), 1u);
  // Fixed slots are claimed by the first lane that actually executes the
  // site; do not hard-code a workgroup/wave-zero owner gate.
  EXPECT_FALSE(
      contains_subsequence(rewritten_words, std::span<const uint32_t>(&*canonical_owner, 1u)));
  EXPECT_EQ(count_subsequence(rewritten_words, *record_address_lo), 1u);
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
  ASSERT_TRUE(result.resolved_moi_exec_save_sgpr);
  const uint16_t exec_save = *result.resolved_moi_exec_save_sgpr;
  const auto lane_rank_lo = build_v_mbcnt_lo_u32_b32(
      10, /*src0=*/0xC1, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
  const auto lane_rank_hi =
      build_v_mbcnt_hi_u32_b32(10, /*src0=*/0xC1, vector_source_vgpr(10), ROCJITSU_CODE_ARCH_RDNA4);
  const auto first_active =
      build_v_cmp_eq_u32_e32_vcc(scalar_positive_inline_u32(0), 10, ROCJITSU_CODE_ARCH_RDNA4);
  const auto narrow = build_s_and_saveexec_b64(exec_save, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore = build_s_mov_b64(kRdna4ExecLo, exec_save, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(lane_rank_lo);
  ASSERT_TRUE(lane_rank_hi);
  ASSERT_TRUE(first_active);
  ASSERT_TRUE(narrow);
  ASSERT_TRUE(restore);
  EXPECT_TRUE(contains_subsequence(rewritten_words, *lane_rank_lo));
  EXPECT_TRUE(contains_subsequence(rewritten_words, *lane_rank_hi));
  EXPECT_TRUE(contains_subsequence(rewritten_words, std::span<const uint32_t>(&*first_active, 1u)));
  const auto narrow_position = std::ranges::find(rewritten_words, *narrow);
  const auto atomic_position =
      std::search(rewritten_words.begin(), rewritten_words.end(), atomic->begin(), atomic->end());
  const auto publication_load_position =
      std::search(rewritten_words.begin(), rewritten_words.end(), publication_load->begin(),
                  publication_load->end());
  const auto publication_claim_position =
      std::search(rewritten_words.begin(), rewritten_words.end(), publication_claim->begin(),
                  publication_claim->end());
  const auto restore_position = std::ranges::find(rewritten_words, *restore);
  ASSERT_NE(narrow_position, rewritten_words.end());
  ASSERT_NE(atomic_position, rewritten_words.end());
  ASSERT_NE(publication_load_position, rewritten_words.end());
  ASSERT_NE(publication_claim_position, rewritten_words.end());
  ASSERT_NE(restore_position, rewritten_words.end());
  EXPECT_LT(narrow_position, atomic_position);
  EXPECT_LT(narrow_position, publication_load_position);
  EXPECT_LT(publication_load_position, publication_claim_position);
  EXPECT_LT(publication_claim_position, atomic_position);
  EXPECT_LT(atomic_position, restore_position);
  EXPECT_TRUE(contains_subsequence(rewritten_words, make_expected_scalar_offset_store_words(
                                                        offsetof(ConSanMoiAccessRecord, lane_mask),
                                                        exec_save, /*address_vgpr=*/8)));
  EXPECT_TRUE(contains_subsequence(
      rewritten_words, make_expected_scalar_offset_store_words(
                           offsetof(ConSanMoiAccessRecord, lane_mask) + sizeof(uint32_t),
                           static_cast<uint16_t>(exec_save + 1u), /*address_vgpr=*/8)));
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
  EXPECT_EQ(result.target_name, "gfx950");
  EXPECT_EQ(result.arch_name, "cdna4");
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_TRUE(result.final_validation_passed);
  ASSERT_EQ(result.moi_candidates.size(), 1u);
  EXPECT_EQ(result.moi_candidates.front().source, ConSanMoiCandidateSource::NativeLds);
  EXPECT_EQ(result.moi_candidates.front().kind, ConSanLdsAccessKind::Write);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineMoiAccessRecordStore);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 8u);

  const std::vector<uint32_t> rewritten_words =
      patched_words_at_file_offset(result, 0x100, result.patches.front().original_size);
  ASSERT_GE(rewritten_words.size(), 2u);
  EXPECT_EQ(rewritten_words[rewritten_words.size() - 2u], text_words[0]);
  EXPECT_EQ(rewritten_words.back(), text_words[1]);

  const auto publication_load = build_cdna4_flat_load_b32(8, 10, 0, ROCJITSU_CODE_ARCH_CDNA4);
  const auto publication_claim = build_cdna4_flat_atomic_or_u32(
      8, 10, 10, /*return_old_value=*/true, /*scope=*/2, ROCJITSU_CODE_ARCH_CDNA4);
  const auto event_index = build_cdna4_flat_atomic_add_u32(8, 10, 10, /*return_old_value=*/true,
                                                           /*scope=*/2, ROCJITSU_CODE_ARCH_CDNA4);
  const auto lane_rank_lo = build_cdna4_v_mbcnt_lo_u32_b32(
      10, /*src0=*/0xc1, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_CDNA4);
  const auto lane_rank_hi = build_cdna4_v_mbcnt_hi_u32_b32(
      10, /*src0=*/0xc1, vector_source_vgpr(10), ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(publication_load && publication_claim && event_index && lane_rank_lo && lane_rank_hi);
  EXPECT_TRUE(contains_subsequence(rewritten_words, *publication_load));
  EXPECT_EQ(count_subsequence(rewritten_words, *publication_claim), 1u);
  EXPECT_TRUE(contains_subsequence(rewritten_words, *event_index));
  EXPECT_TRUE(contains_subsequence(rewritten_words, *lane_rank_lo));
  EXPECT_TRUE(contains_subsequence(rewritten_words, *lane_rank_hi));
}

TEST(ConSanMoi, Cdna4RecordReplayNormalizesTransposeAndTwoAddressLdsRanges) {
  const auto check = [](uint32_t word0, uint32_t word1, std::string_view expected_mnemonic,
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
    ASSERT_EQ(result.site_dispositions.size(), 1u);
    EXPECT_EQ(result.site_dispositions.front().disposition, ConSanSiteDisposition::Supported);
    EXPECT_EQ(result.site_dispositions.front().lowering_outcome,
              ConSanSiteLoweringOutcome::Patched);

    const ConSanMoiAutoReportInventory inventory =
        inventory_consan_moi_auto_report(result, options, bytes);
    EXPECT_EQ(inventory.access_range_count, expected_byte_offsets.size());
    ASSERT_EQ(result.patches.size(), 1u);
    const std::vector<uint32_t> rewritten_words =
        patched_words_at_file_offset(result, 0x100, result.patches.front().original_size);
    for (uint32_t byte_offset : expected_byte_offsets) {
      const auto mov_offset = instrumentation::build_v_mov_b32_literal(
          /*vdst=*/22, byte_offset, ROCJITSU_CODE_ARCH_CDNA4);
      ASSERT_TRUE(mov_offset);
      EXPECT_TRUE(contains_subsequence(rewritten_words, *mov_offset))
          << "missing byte offset " << byte_offset;
    }
  };

  check(0xD9C60800u, 0x0E000001u, "ds_read_b64_tr_b16", {2048u});
  check(0xD8EE0400u, 0x0800000Bu, "ds_read2_b64", {0u, 32u});
  check(0xD89E0400u, 0x0006080Bu, "ds_write2st64_b64", {0u, 2048u});
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
    ASSERT_EQ(result.patches.size(), 1u);
    const std::vector<uint32_t> rewritten_words =
        patched_words_at_file_offset(result, 0x100, result.patches.front().original_size);
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
  constexpr auto write_b8_d16_hi =
      cdna4::build_ds(cdna4::kDsWriteB8D16HiDs, {.offset0 = 0x12, .addr = 5, .data0 = 9});
  constexpr auto write_b16_d16_hi =
      cdna4::build_ds(cdna4::kDsWriteB16D16HiDs, {.offset0 = 0x34, .addr = 6, .data0 = 10});
  check(read_i8[0], read_i8[1], "ds_read_i8", ConSanLdsAccessKind::Read, 8u, 0x21u,
        /*addr=*/8u, /*vdst=*/12u);
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
  const std::vector<uint32_t> prologue_words =
      text_words_at_offset(patched, prologue->trampoline_offset, prologue->trampoline_size);
  const uint32_t capture_entry_owner =
      build_v_mov_b32_e32(*prologue->scratch_vgpr, vector_source_vgpr(/*workitem_id_x=*/0u), kArch);
  const uint32_t capture_scalar_zero =
      build_v_mov_b32_e32(*prologue->scratch_vgpr, /*scalar_src=*/0u, kArch);
  EXPECT_NE(std::ranges::find(prologue_words, capture_entry_owner), prologue_words.end());
  EXPECT_EQ(std::ranges::find(prologue_words, capture_scalar_zero), prologue_words.end());
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
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_GFX1250;
  constexpr auto guest = gfx1250::build_vds(gfx1250::kDsStoreB32Vds, {.addr = 2u, .data0 = 3u});
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
  expect_captured_owner(*fence_patch, static_cast<uint16_t>(*fence_patch->scratch_vgpr + 2u));
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, Gfx1250PrivateOwnerSpillsBeginAfterCapturedState) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_GFX1250;
  constexpr auto guest = gfx1250::build_vds(gfx1250::kDsStoreB32Vds, {.addr = 2u, .data0 = 3u});
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

TEST(ConSanMoi, Gfx1250PrivateOwnerFallbackRetainsAtomicRecord) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_GFX1250;
  constexpr auto guest = gfx1250::build_vds(gfx1250::kDsStoreB32Vds, {.addr = 2u, .data0 = 3u});
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
  EXPECT_NE(std::ranges::find(
                result.warnings,
                "ConSan MOI record/replay reverted to probe-local state after all access probes "
                "failed placement"),
            result.warnings.end())
      << testing::PrintToString(result.warnings);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAtomicRecord,
                               &ConSanPatchInfo::kind),
            1u)
      << testing::PrintToString(result.warnings);
  EXPECT_FALSE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("private owner has no captured state") != std::string::npos;
  }));
  EXPECT_TRUE(result.final_validation_passed);
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
  // Supplying an already-persistent key lets this test isolate the private
  // owner path without asking the automatic allocator for CDNA scalar state.
  options.moi_persistent_workgroup_key_sgpr = 40u;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(8, 0, 0, 0, 0, 8, 8);

  const ConSanResult result =
      try_patch_consan(make_cdna3_lds_code_object(text_words, "private_sync_entry_owner"), options);

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
  EXPECT_TRUE(result.final_validation_passed);
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
  EXPECT_EQ(plan.required_vgpr_count, 11u);
  ASSERT_TRUE(result.modified) << "warnings=" << testing::PrintToString(result.warnings)
                               << " errors=" << testing::PrintToString(result.errors);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().scratch_vgpr, 8u);

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
  EXPECT_EQ(result.resource_plans.front().scratch_vgpr_count, 3u);
  EXPECT_EQ(result.resource_plans.front().current_vgpr_count, 8u);
  EXPECT_EQ(result.resource_plans.front().original_private_segment_size, 32u);
  ASSERT_TRUE(result.modified) << "warnings=" << testing::PrintToString(result.warnings)
                               << " errors=" << testing::PrintToString(result.errors)
                               << " dispositions="
                               << testing::PrintToString(result.site_dispositions)
                               << " resources=" << testing::PrintToString(result.resource_plans);
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_TRUE(result.final_validation_passed);
  ASSERT_EQ(result.patches.size(), 1u);
  const ConSanPatchInfo &patch = result.patches.front();
  EXPECT_EQ(patch.kind, ConSanPatchKind::TrampolineMoiAccessRecordStore);
  ASSERT_TRUE(patch.scratch_vgpr);
  EXPECT_EQ(patch.spilled_vgpr_count, 3u);
  EXPECT_EQ(patch.required_private_segment_size, 48u);
  EXPECT_EQ(result.resource_plan_summary.spill_plans, 1u);
  EXPECT_EQ(result.resource_plan_summary.emitted_spill_patches, 1u);
  EXPECT_EQ(result.resource_plan_summary.emitted_spill_slot_bytes, 12u);

  const auto wait = build_cdna4_s_wait_vmcnt0(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(wait);
  std::vector<uint32_t> save{*wait};
  std::vector<uint32_t> restore;
  for (uint16_t i = 0; i < patch.spilled_vgpr_count; ++i) {
    const uint16_t vgpr = static_cast<uint16_t>(*patch.scratch_vgpr + i);
    const uint32_t offset = 32u + 4u * i;
    const auto store =
        build_cdna4_address_free_scratch_store_b32(vgpr, offset, ROCJITSU_CODE_ARCH_CDNA4);
    const auto load =
        build_cdna4_address_free_scratch_load_b32(vgpr, offset, ROCJITSU_CODE_ARCH_CDNA4);
    ASSERT_TRUE(store && load);
    save.insert(save.end(), store->begin(), store->end());
    restore.insert(restore.end(), load->begin(), load->end());
  }
  save.push_back(*wait);
  restore.push_back(*wait);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const Section *text = patched.text_sections().front();
  std::vector<uint32_t> patched_words(text->size() / sizeof(uint32_t));
  std::memcpy(patched_words.data(), text->data(), text->size());
  EXPECT_TRUE(contains_subsequence(patched_words, save));
  EXPECT_TRUE(contains_subsequence(patched_words, restore));
  EXPECT_TRUE(
      contains_subsequence(patched_words, std::array<uint32_t, 2>{text_words[0], text_words[1]}));

  ASSERT_EQ(patched.kernels().size(), 1u);
  KD descriptor{};
  std::memcpy(&descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  EXPECT_EQ(descriptor.private_segment_fixed_size, 48u);
  EXPECT_EQ(
      AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT),
      1u);
}

TEST(ConSanMoi, Cdna4StaticRecordReplayRestoresOverlappingStoreOperandsBeforeGuest) {
  const auto guest = build_cdna4_ds_store_b32(
      /*vaddr=*/0, /*vdata=*/4, /*byte_offset=*/0, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(guest);
  std::vector<uint32_t> text_words(guest->begin(), guest->end());
  for (uint16_t vgpr = 1; vgpr < 8; ++vgpr)
    text_words.push_back(
        build_v_mov_b32_e32(/*vdst=*/7, vector_source_vgpr(vgpr), ROCJITSU_CODE_ARCH_CDNA4));
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4));
  std::vector<uint8_t> bytes =
      make_cdna4_lds_code_object(text_words, "record_replay_store_operand_overlap");
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    // Bound ordinary VGPRs to v0..v7. Every aligned three-VGPR window then
    // intersects either the address v0 or store payload v4.
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
  ASSERT_EQ(patch->spilled_vgpr_count, 3u);

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
  std::vector<uint8_t> bytes =
      make_cdna4_lds_code_object(text_words, "record_replay_load_operand_overlap");
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
  ASSERT_EQ(patch->spilled_vgpr_count, 3u);

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
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineMoiAccessRecordStore);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 16u);

  const std::vector<uint32_t> rewritten_words =
      patched_words_at_file_offset(result, 0x100, result.patches.front().original_size);

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
    ASSERT_EQ(result.patches.size(), 1u);
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

    std::vector<uint32_t> words(result.patches.front().original_size / sizeof(uint32_t));
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

TEST(ConSanMoi, FirstLightProbeDerivesOwnerFromWorkitemIdWhenOwnerVgprIsUnset) {
  std::array<uint32_t, 170> text_words{};
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000000u; // ds_store_b32 v0, v0
  for (size_t i = 2; i + 1 < text_words.size(); ++i)
    text_words[i] = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "lds_probe", kRdna4Wave64AllVgprsGranulated, /*wave32=*/true);
  ConSanOptions options = moi_options();
  options.scratch_vgpr = 8;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineMoiAccessRecordStore);
  EXPECT_EQ(result.patches.front().original_size, 116u * sizeof(uint32_t));

  const auto owner_init =
      build_v_lshrrev_b32_e32(10, scalar_positive_inline_u32(5), 0, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(owner_init);
  const auto owner_mask = build_v_and_b32_e32_literal(10, consan_moi_exact_shadow::max_owner, 10,
                                                      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(owner_mask);
  const std::vector<uint32_t> rewritten_words =
      patched_words_at_file_offset(result, 0x100, result.patches.front().original_size);
  EXPECT_TRUE(contains_subsequence(rewritten_words, std::span<const uint32_t>(&*owner_init, 1u)));
  EXPECT_TRUE(contains_subsequence(rewritten_words, *owner_mask));
  EXPECT_TRUE(contains_subsequence(
      rewritten_words,
      make_expected_offset_store_words(offsetof(ConSanMoiAccessRecord, wave_id), 10, 8)));
  ASSERT_GE(rewritten_words.size(), 2u);
  EXPECT_EQ(rewritten_words[rewritten_words.size() - 2u], text_words[0]);
  EXPECT_EQ(rewritten_words.back(), text_words[1]);
}

TEST(ConSanMoi, FirstLightProbeStoresDescriptorWorkgroupIds) {
  std::array<uint32_t, 220> text_words{};
  text_words[0] = 0xD8340000u;
  text_words[1] = 0x00000000u; // ds_store_b32 v0, v0
  for (size_t i = 2; i + 1 < text_words.size(); ++i)
    text_words[i] = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);

  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 5u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z,
                    1u);
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
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineMoiAccessRecordStore);

  const std::vector<uint32_t> rewritten_words = patched_words_at_file_offset(
      result, 0x100 + result.patches.front().anchor_offset, result.patches.front().original_size);

  const std::vector<uint32_t> expected_x = make_expected_scalar_offset_store_words(
      offsetof(ConSanMoiAccessRecord, workgroup_x), ttmp_scalar_operand(kTtmpRdna4GridX),
      *options.scratch_vgpr);
  const std::vector<uint32_t> expected_y = make_expected_scalar_offset_store_words(
      offsetof(ConSanMoiAccessRecord, workgroup_y), ttmp_scalar_operand(kTtmpRdna4GridYz),
      *options.scratch_vgpr, /*shift_right_16=*/false, /*mask_low_16=*/true);
  const std::vector<uint32_t> expected_z = make_expected_scalar_offset_store_words(
      offsetof(ConSanMoiAccessRecord, workgroup_z), ttmp_scalar_operand(kTtmpRdna4GridYz),
      *options.scratch_vgpr, /*shift_right_16=*/true);
  EXPECT_TRUE(contains_subsequence(rewritten_words, expected_x));
  EXPECT_TRUE(contains_subsequence(rewritten_words, expected_y));
  EXPECT_TRUE(contains_subsequence(rewritten_words, expected_z));
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

TEST(ConSanMoi, AutoReportInventoryAdaptsRecordReplayHeadroomForFatObjects) {
  ConSanMoiAutoReportInventory inventory;
  inventory.engine = ConSanMoiEngine::RecordReplay;
  inventory.access_range_count = 47428u;
  inventory.diagnostic_count = 47428u;
  inventory.barrier_event_count = 10000u;

  const ConSanMoiAutoReportInventory fitted =
      fit_consan_moi_record_replay_auto_report_inventory(inventory);
  const ConSanMoiAutoReportPlan plan = plan_consan_moi_auto_report(fitted);

  ASSERT_TRUE(plan.complete());
  EXPECT_EQ(fitted.barrier_event_count, 10000u * 256u);
  EXPECT_LT(fitted.barrier_event_count, 10000u * kConSanMoiRecordReplayDynamicEventHeadroom);
  EXPECT_LE(plan.required_bytes, kConSanMoiAutoReportBufferCeilingBytes);
}

TEST(ConSanMoi, FirstLightProbeAddsNativeLdsImmediateOffset) {
  std::array<uint32_t, 180> text_words{};
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
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().original_size, 120u * sizeof(uint32_t));

  const std::vector<uint32_t> rewritten_words =
      patched_words_at_file_offset(result, 0x100, result.patches.front().original_size);

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
  std::vector<uint32_t> text_words(420, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
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
  ASSERT_EQ(result.patches.size(), 1u);

  const std::vector<uint32_t> rewritten_words =
      patched_words_at_file_offset(result, 0x100, result.patches.front().original_size);

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
  EXPECT_EQ(result.resource_plans.front().scratch_vgpr_count, 3u);
  ASSERT_EQ(result.patches.size(), 1u);
  const ConSanPatchInfo &patch = result.patches.front();
  ASSERT_EQ(patch.kind, ConSanPatchKind::TrampolineMoiAccessRecordStore);
  EXPECT_LT(patch.trampoline_size, 932u)
      << "dispatch-qualified static Record/Replay probes must retain the CLIP size bound";

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
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineMoiAccessRecordStore);

  const std::vector<uint32_t> rewritten_words =
      patched_words_at_file_offset(result, 0x100, result.patches.front().original_size);
  const auto wait_store = build_s_wait_storecnt0(ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_exec =
      build_s_mov_b64(/*sdst=*/126, *options.moi_exec_save_sgpr, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(wait_store);
  ASSERT_TRUE(restore_exec);
  const auto first_wait = std::ranges::find(rewritten_words, *wait_store);
  const auto first_restore = std::ranges::find(rewritten_words, *restore_exec);
  ASSERT_NE(first_wait, rewritten_words.end());
  ASSERT_NE(first_restore, rewritten_words.end());
  EXPECT_LT(first_wait, first_restore);
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
  ASSERT_EQ(result.patches.size(), 1u);
  const ConSanPatchInfo &patch = result.patches.front();
  EXPECT_EQ(patch.kind, ConSanPatchKind::TrampolineMoiAccessRecordStore);
  EXPECT_EQ(patch.anchor_offset, 0u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  std::vector<uint32_t> actual_words(patched.text_sections().front()->size() / sizeof(uint32_t));
  std::memcpy(actual_words.data(), patched.text_sections().front()->data(),
              patched.text_sections().front()->size());
  EXPECT_EQ(actual_words[2], build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4));

  const auto wait_store = build_s_wait_storecnt0(ROCJITSU_CODE_ARCH_RDNA4);
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
  ASSERT_EQ(result.patches.size(), 1u);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().scratch_vgpr_count, 7u);
  const ConSanPatchInfo &patch = result.patches.front();
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
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  bool saw_saveexec_warning = false;
  for (const std::string &warning : result.warnings)
    saw_saveexec_warning |= warning.find("immediately after s_*_saveexec") != std::string::npos;
  EXPECT_TRUE(saw_saveexec_warning);
}

TEST(ConSanMoi, FirstLightProbeCanPatchTwoNativeLdsAccessRecords) {
  constexpr uint32_t kSecondSiteWord = 170;
  std::vector<uint32_t> text_words(360, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
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
  ASSERT_EQ(result.patches.size(), 2u);
  EXPECT_EQ(result.patches[0].kind, ConSanPatchKind::InlineMoiAccessRecordStore);
  EXPECT_EQ(result.patches[0].anchor_offset, 0u);
  EXPECT_EQ(result.patches[0].original_size, 116u * sizeof(uint32_t));
  EXPECT_EQ(result.patches[1].kind, ConSanPatchKind::InlineMoiAccessRecordStore);
  EXPECT_EQ(result.patches[1].anchor_offset, kSecondSiteWord * sizeof(uint32_t));
  EXPECT_EQ(result.patches[1].original_size, 116u * sizeof(uint32_t));

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
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(2, 0, 0, 0);
  options.max_patches = 2;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 2u);
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

TEST(ConSanMoi, RecordReplayAutomaticExecSaveOverridesOnlyIncompatibleOwner) {
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
  // The first owner leaves only s0:s7 dead at its access. The second leaves
  // only s98:s105 dead. Their union has no code-object-wide five-SGPR
  // window, but each independent owner has a safe transient window.
  const std::vector<uint32_t> first_words =
      make_owner(/*first_live=*/8u, /*last_live=*/105u, /*dead_destination=*/0u);
  const std::vector<uint32_t> second_words =
      make_owner(/*first_live=*/0u, /*last_live=*/97u, /*dead_destination=*/97u);
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
  ASSERT_EQ(result.resolved_moi_transient_sgpr_assignments.size(), 1u);
  EXPECT_FALSE(result.resolved_moi_transient_sgpr_assignments.front().dispatch_id_sgpr);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               &ConSanPatchInfo::kind),
            2u);
  EXPECT_TRUE(std::ranges::all_of(result.resource_plans, [](const auto &plan) {
    return plan.site_kind != ConSanResourceSiteKind::Access ||
           plan.source != ConSanRegisterAllocationSource::Unsupported;
  }));
}

TEST(ConSanMoi, Gfx1250RecordReplayKeepsDispatchOnlyFullPressureOwner) {
  std::vector<uint32_t> low_pressure_words(320u, build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250));
  low_pressure_words[0] = 0xD8340000u;
  low_pressure_words[1] = 0x00000000u; // ds_store_b32 v0, v0
  low_pressure_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250);

  std::vector<uint32_t> full_pressure_words = low_pressure_words;
  full_pressure_words[2] = build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/105u, ROCJITSU_CODE_ARCH_GFX1250);
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
  // s101 is the highest guest reference, so the unguarded aligned allocator
  // chooses s102:s103. The architectural-alias guard must skip directly to
  // the only remaining ordinary pair.
  EXPECT_EQ(*result.resolved_moi_dispatch_id_sgpr, 104u);
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

TEST(ConSanMoi, RecordReplayOwnerLocalExecSaveRequiresCommonWindowForSharedHelper) {
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
  EXPECT_EQ(shared_plan->source, ConSanRegisterAllocationSource::Unsupported);
  EXPECT_EQ(shared_plan->reason, ConSanRegisterPlanReason::ForbiddenOverlap);
  // The shared helper has one text body. Its callers' disjoint safe windows
  // must not be represented as two incompatible per-owner assignments.
  EXPECT_TRUE(std::ranges::none_of(
      result.resolved_moi_transient_sgpr_assignments, [&](const auto &assignment) {
        return std::ranges::find(shared_plan->owner_descriptor_file_offsets,
                                 assignment.descriptor_file_offset) !=
               shared_plan->owner_descriptor_file_offsets.end();
      }));
}

TEST(ConSanMoi, RecordReplaySpillsExecVccStateOnGfx12) {
  for (const rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_GFX1250}) {
    SCOPED_TRACE(arch);
    const std::array<uint16_t, 4> dead = {0u, 1u, 4u, 6u};
    const uint32_t access_count = arch == ROCJITSU_CODE_ARCH_GFX1250 ? 9u : 1u;
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
    const std::vector<uint8_t> bytes = arch == ROCJITSU_CODE_ARCH_GFX1250
                                           ? make_gfx1250_code_object(words)
                                           : make_rdna4_lds_code_object(words);

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
    EXPECT_TRUE(std::ranges::any_of(result.patches, [](const ConSanPatchInfo &patch) {
      return patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore &&
             patch.required_private_segment_size > 0u;
    }));
    EXPECT_TRUE(result.final_validation_passed);
  }
}

TEST(ConSanMoi, RecordReplayRejectsSpillRouterWithoutDeadPairAndScalars) {
  const auto run = [](std::span<const uint16_t> dead) {
    std::vector<uint32_t> words;
    for (uint32_t index = 0; index < 9u; ++index) {
      words.push_back(0xD8340000u);
      words.push_back(0x00000000u); // ds_store_b32 v0, v0
    }
    for (uint16_t sgpr = 0; sgpr < 106u; ++sgpr) {
      if (std::ranges::find(dead, sgpr) == dead.end())
        words.push_back(build_s_mov_b32(/*M0=*/125u, sgpr, ROCJITSU_CODE_ARCH_GFX1250));
    }
    words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250));

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

  // No aligned pair is dead, so the router cannot hold a PC.
  const std::array<uint16_t, 5> no_pc_pair = {0u, 2u, 4u, 6u, 8u};
  const ConSanResult no_pc = run(no_pc_pair);
  EXPECT_TRUE(consan_patch_succeeded(no_pc)) << testing::PrintToString(no_pc.errors);
  EXPECT_TRUE(no_pc.resolved_moi_transient_sgpr_assignments.empty());
  EXPECT_EQ(std::ranges::count(no_pc.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               &ConSanPatchInfo::kind),
            0u);

  // One dead pair and one scalar cannot retain both guest SCC and the dense
  // dispatch key across the long jump.
  const std::array<uint16_t, 3> no_key_scalar = {0u, 1u, 4u};
  const ConSanResult no_key = run(no_key_scalar);
  EXPECT_TRUE(consan_patch_succeeded(no_key)) << testing::PrintToString(no_key.errors);
  EXPECT_TRUE(no_key.resolved_moi_transient_sgpr_assignments.empty());
  EXPECT_EQ(std::ranges::count(no_key.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               &ConSanPatchInfo::kind),
            0u);
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

  const ConSanResult result =
      try_patch_consan(make_rdna4_lds_code_object(words, "rdna4_spill_dense_two"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.resolved_moi_transient_sgpr_assignments.size(), 1u);
  EXPECT_TRUE(result.resolved_moi_transient_sgpr_assignments.front().spill_backed);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               &ConSanPatchInfo::kind),
            kAccessCount);
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

TEST(ConSanMoi, RecordReplaySpillBackedDenseHostDoesNotConsumeNearbyBarrier) {
  constexpr uint32_t kAccessCount = 9u;
  const std::array<uint16_t, 6> dead = {0u, 1u, 4u, 6u, 8u, 9u};
  std::vector<uint32_t> words(
      9u, build_s_mov_b32(/*sdst=*/20u, /*ssrc0=*/20u, ROCJITSU_CODE_ARCH_GFX1250));
  for (uint32_t index = 0; index < kAccessCount; ++index) {
    words.push_back(0xD8340000u | index * sizeof(uint32_t));
    words.push_back(0x00000000u); // ds_store_b32 v0, v0 offset:index*4
    if (index == 1u)
      words.push_back(0xBF940000u); // s_barrier_wait -1
  }
  for (uint16_t sgpr = 0; sgpr < 106u; ++sgpr) {
    if (std::ranges::find(dead, sgpr) == dead.end())
      words.push_back(build_s_mov_b32(/*sdst=*/20u, sgpr, ROCJITSU_CODE_ARCH_GFX1250));
  }
  words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250));
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
  EXPECT_TRUE(result.resolved_moi_transient_sgpr_assignments.front().spill_backed);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               &ConSanPatchInfo::kind),
            kAccessCount);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiInlineEpochBarrier,
                               &ConSanPatchInfo::kind),
            1u);
  EXPECT_TRUE(std::ranges::any_of(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiIndirectBranchIsland &&
           patch.original_size == 9u * sizeof(uint32_t);
  }));
  EXPECT_TRUE(std::ranges::all_of(result.site_dispositions, [](const auto &site) {
    return site.disposition != ConSanSiteDisposition::Supported ||
           site.lowering_outcome == ConSanSiteLoweringOutcome::Patched;
  }));
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
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiInlineEpochBarrier,
                               &ConSanPatchInfo::kind),
            1);
  // The epoch body may fit directly in its reserved local relay space. Keep
  // this test about the composition contract instead of requiring the larger
  // indirect-entry layout.
  EXPECT_GE(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiIndirectBranchIsland,
                               &ConSanPatchInfo::kind),
            9);

  const auto barrier_patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiInlineEpochBarrier &&
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
      gfx1250::build_vds(gfx1250::kDsStore2addrB64Vds,
                         {.offset0 = 0u, .offset1 = 1u, .addr = 0u, .data0 = 29u, .data1 = 35u});
  const std::array<uint32_t, 3> text_words = {
      store[0],
      store[1],
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250),
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
  EXPECT_EQ(*result.moi_candidates.front().second_data_vgpr, 35u);
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore ||
           patch.kind == ConSanPatchKind::InlineMoiAccessRecordStore;
  });
  ASSERT_NE(access, result.patches.end());
  ASSERT_TRUE(access->scratch_vgpr);
  EXPECT_EQ(*access->scratch_vgpr, 32u);
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
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;
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
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineMoiAccessRecordStore);
  EXPECT_EQ(result.patches.front().anchor_offset, 28u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 40u);
  EXPECT_EQ(result.patches.front().original_size, 117u * sizeof(uint32_t));

  const std::vector<uint32_t> rewritten_words =
      patched_words_at_file_offset(result, 0x11c, result.patches.front().original_size);
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
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;
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
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 1);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.kernels.front().barrier_sites.size(), 1u);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::TrampolineMoiBarrierRecord);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().trampoline_offset, text_words.size() * sizeof(uint32_t));
  EXPECT_EQ(result.patches.front().original_size, sizeof(uint32_t));

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

  const ConSanPatchInfo &patch = result.patches.front();
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
  EXPECT_GT(result.patches.front().trampoline_size, 0u);
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

TEST(ConSanMoi, BarrierRecordPatchStoresDescriptorWorkgroupIds) {
  constexpr uint32_t kBarrierWait = 0xBF940000u;
  const std::array<uint32_t, 2> text_words = {
      kBarrierWait,
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };

  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 5u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z,
                    1u);
  });

  ConSanOptions options = moi_options();
  options.moi_track_barriers = true;
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = 30;
  options.moi_owner_vgpr = 14;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 1);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::TrampolineMoiBarrierRecord);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);

  const ConSanPatchInfo &patch = result.patches.front();
  const std::vector<uint32_t> trampoline_words =
      text_words_at_offset(patched, patch.trampoline_offset, patch.trampoline_size);

  const std::vector<uint32_t> expected_x = {
      build_v_mov_b32_e32(/*vdst=*/13, ttmp_scalar_operand(kTtmpRdna4GridX),
                          ROCJITSU_CODE_ARCH_RDNA4),
  };
  const auto y_shift_left = build_v_lshlrev_b32_e32(
      /*vdst=*/13, scalar_positive_inline_u32(16), /*vsrc1=*/13, ROCJITSU_CODE_ARCH_RDNA4);
  const auto y_shift_right = build_v_lshrrev_b32_e32(
      /*vdst=*/13, scalar_positive_inline_u32(16), /*vsrc1=*/13, ROCJITSU_CODE_ARCH_RDNA4);
  const auto z_shift = build_v_lshrrev_b32_e32(
      /*vdst=*/13, scalar_positive_inline_u32(16), /*vsrc1=*/13, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(y_shift_left);
  ASSERT_TRUE(y_shift_right);
  ASSERT_TRUE(z_shift);
  const std::vector<uint32_t> expected_y = {
      build_v_mov_b32_e32(/*vdst=*/13, ttmp_scalar_operand(kTtmpRdna4GridYz),
                          ROCJITSU_CODE_ARCH_RDNA4),
      *y_shift_left,
      *y_shift_right,
  };
  const std::vector<uint32_t> expected_z = {
      build_v_mov_b32_e32(/*vdst=*/13, ttmp_scalar_operand(kTtmpRdna4GridYz),
                          ROCJITSU_CODE_ARCH_RDNA4),
      *z_shift,
  };
  EXPECT_TRUE(contains_subsequence(trampoline_words, expected_x));
  EXPECT_TRUE(contains_subsequence(trampoline_words, expected_y));
  EXPECT_TRUE(contains_subsequence(trampoline_words, expected_z));
}

TEST(ConSanMoi, BarrierRecordAutomaticallyPlansScratchAndScalarState) {
  constexpr uint32_t kBarrierWait = 0xBF940000u;
  const std::array<uint32_t, 4> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32 v0, v0
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
  EXPECT_GT(*patch->scratch_vgpr, 0u);
  const auto plan = std::ranges::find_if(result.resource_plans, [](const auto &item) {
    return item.site_kind == ConSanResourceSiteKind::Barrier;
  });
  ASSERT_NE(plan, result.resource_plans.end());
  EXPECT_EQ(plan->scratch_vgpr, patch->scratch_vgpr);
  EXPECT_EQ(result.resource_plan_summary.dead_plans, 2u);
  EXPECT_EQ(std::ranges::count(result.resource_plans, ConSanResourceSiteKind::Fence,
                               &ConSanCandidateResourcePlan::site_kind),
            0);
}

TEST(ConSanMoi, BarrierRecordForcedSpillUsesPlannedPrivateWindow) {
  constexpr uint32_t kBarrierWait = 0xBF940000u;
  const std::array<uint32_t, 4> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32 v0, v0
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
  EXPECT_EQ(result.resource_plan_summary.spill_plans, 2u);
  EXPECT_EQ(result.resource_plan_summary.emitted_spill_patches, 2u);
  EXPECT_EQ(result.resource_plan_summary.emitted_spill_slot_bytes, 36u);
}

TEST(ConSanMoi, Cdna4BarrierRecordForcedSpillUsesNativePrivateWindows) {
  const auto guest = build_cdna4_ds_store_b32(
      /*vaddr=*/2, /*vdata=*/3, /*byte_offset=*/0, ROCJITSU_CODE_ARCH_CDNA4);
  const auto barrier = build_cdna4_s_barrier(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(guest && barrier);
  std::vector<uint32_t> text_words(320, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  std::copy(guest->begin(), guest->end(), text_words.begin());
  text_words[guest->size()] = *barrier;
  text_words[guest->size() + 1u] =
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
  EXPECT_EQ(result.target_name, "gfx950");
  EXPECT_EQ(result.arch_name, "cdna4");
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
  EXPECT_EQ(result.resource_plan_summary.spill_plans, 2u);
  EXPECT_EQ(result.resource_plan_summary.emitted_spill_patches, 2u);
  EXPECT_EQ(result.resource_plan_summary.emitted_spill_slot_bytes, 36u);
}

TEST(ConSanMoi, RecordReplayPersistentEpochAvoidsDynamicBarrierRecords) {
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
            0);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiInlineEpochBarrier,
                               &ConSanPatchInfo::kind),
            1);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue,
                               &ConSanPatchInfo::kind),
            1);
}

TEST(ConSanMoi, Cdna4RecordReplayPersistentEpochAdvancesAtBarrier) {
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
            0);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiInlineEpochBarrier,
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
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiInlineEpochBarrier,
                               &ConSanPatchInfo::kind),
            1);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiBarrierRecord,
                               &ConSanPatchInfo::kind),
            0);
  const auto access = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore, &ConSanPatchInfo::kind);
  const auto barrier = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiInlineEpochBarrier, &ConSanPatchInfo::kind);
  const auto prologue = std::ranges::find(
      result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue, &ConSanPatchInfo::kind);
  ASSERT_NE(access, result.patches.end());
  ASSERT_NE(barrier, result.patches.end());
  ASSERT_NE(prologue, result.patches.end());
  EXPECT_FALSE(access->persistent_epoch_private_offset);
  EXPECT_FALSE(barrier->persistent_epoch_private_offset);
  EXPECT_FALSE(prologue->persistent_epoch_private_offset);
}

TEST(ConSanMoi, Gfx1250FullVgprRecordReplayUsesScalarEpochCoalescing) {
  constexpr uint32_t kBarrierWait = 0xBF94FFFFu;
  std::vector<uint32_t> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32 v0, v0
  };
  text_words.insert(text_words.end(), 33u, kBarrierWait);
  text_words.push_back(
      build_v_mov_b32_e32(/*vdst=*/255, vector_source_vgpr(255), ROCJITSU_CODE_ARCH_GFX1250));
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250));

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
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiInlineEpochBarrier,
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
                                                  ROCJITSU_CODE_ARCH_GFX1250)),
            access_words.end());
}

TEST(ConSanMoi, Gfx1250HighSgprPressureSkipsFlatScratchForPersistentEpoch) {
  constexpr uint32_t kBarrierWait = 0xBF94FFFFu;
  std::vector<uint32_t> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32 v0, v0
      kBarrierWait,
      build_s_mov_b32(/*sdst=*/0u, /*ssrc=*/101u, ROCJITSU_CODE_ARCH_GFX1250),
      build_v_mov_b32_e32(/*vdst=*/255, vector_source_vgpr(255), ROCJITSU_CODE_ARCH_GFX1250),
  };
  text_words.resize(320u, build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250));
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250);

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
  EXPECT_EQ(*result.resolved_moi_dispatch_id_sgpr, 104u);
  EXPECT_TRUE(result.moi_private_epoch_automatic);
  EXPECT_FALSE(result.resolved_moi_persistent_owner_sgpr);
  EXPECT_FALSE(result.resolved_moi_persistent_epoch_sgpr);
  EXPECT_NE(std::ranges::find(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                              &ConSanPatchInfo::kind),
            result.patches.end());
  EXPECT_NE(std::ranges::find(result.patches, ConSanPatchKind::TrampolineMoiInlineEpochBarrier,
                              &ConSanPatchInfo::kind),
            result.patches.end());
}

TEST(ConSanMoi, Gfx1250RejectsExplicitPersistentStateInFlatScratch) {
  std::vector<uint32_t> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32 v0, v0
  };
  text_words.resize(128u, build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250));
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250);

  const auto patch_with = [&](ConSanOptions options) {
    options.moi_report_buffer_address = 0x123456780000ull;
    options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(64, 0, 0, 0);
    return try_patch_consan(
        make_gfx1250_code_object(text_words, "gfx1250_explicit_flat_scratch_state"), options);
  };

  const auto expect_special_alias_rejected = [&](const ConSanOptions &options) {
    const ConSanResult result = patch_with(options);
    EXPECT_EQ(result.outcome, ConSanTransformOutcome::Unsupported);
    EXPECT_FALSE(result.modified);
    EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
      return warning.find("architectural special SGPR") != std::string::npos;
    })) << testing::PrintToString(result.warnings);
  };

  ConSanOptions dispatch_options = moi_options(ConSanMoiEngine::RecordReplay);
  dispatch_options.moi_dispatch_id_sgpr = 102u;
  expect_special_alias_rejected(dispatch_options);

  ConSanOptions owner_options = moi_options(ConSanMoiEngine::RecordReplay);
  owner_options.moi_persistent_owner_sgpr = 102u;
  owner_options.moi_persistent_epoch_sgpr = 81u;
  owner_options.moi_init_owner_epoch = true;
  expect_special_alias_rejected(owner_options);

  ConSanOptions epoch_options = moi_options(ConSanMoiEngine::RecordReplay);
  epoch_options.moi_persistent_owner_sgpr = 80u;
  epoch_options.moi_persistent_epoch_sgpr = 103u;
  epoch_options.moi_init_owner_epoch = true;
  expect_special_alias_rejected(epoch_options);

  ConSanOptions workgroup_options = moi_options(ConSanMoiEngine::RecordReplay);
  workgroup_options.moi_persistent_owner_sgpr = 80u;
  workgroup_options.moi_persistent_epoch_sgpr = 81u;
  workgroup_options.moi_persistent_workgroup_key_sgpr = 102u;
  workgroup_options.moi_init_owner_epoch = true;
  expect_special_alias_rejected(workgroup_options);
}

TEST(ConSanMoi, Gfx1250AcceptsConfiguredPersistentStateAboveFlatScratch) {
  std::vector<uint32_t> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32 v0, v0
  };
  text_words.resize(128u, build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250));
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250);

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  // gfx1250 has no persistent XNACK_MASK selector at s104:s105, so this pair
  // remains ordinary scalar state above the aliased FLAT_SCRATCH selectors.
  options.moi_persistent_owner_sgpr = 104u;
  options.moi_persistent_epoch_sgpr = 105u;
  options.moi_init_owner_epoch = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(64, 0, 0, 0);

  const ConSanResult result = try_patch_consan(
      make_gfx1250_code_object(text_words, "gfx1250_explicit_state_above_flat_scratch"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_TRUE(std::ranges::none_of(result.warnings, [](const std::string &warning) {
    return warning.find("architectural special SGPR") != std::string::npos;
  })) << testing::PrintToString(result.warnings);
}

TEST(ConSanMoi, Gfx1250RejectsConfiguredPersistentStateAtOrdinarySgprLimit) {
  std::vector<uint32_t> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32 v0, v0
  };
  text_words.resize(128u, build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250));
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250);

  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  // s106:s107 are VCC and begin immediately after the ordinary s0:s105 file.
  // This locks the limit boundary; the adjacent tests cover the reserved
  // FLAT_SCRATCH subrange and the valid s104:s105 range below it.
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
  options.moi_init_owner_epoch = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(64, 1, 0, 0, 64);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.moi_persistent_sgprs_automatic);
  EXPECT_FALSE(result.moi_persistent_vgprs_automatic);
  EXPECT_FALSE(result.moi_private_epoch_automatic);
  ASSERT_TRUE(result.resolved_moi_persistent_owner_sgpr);
  ASSERT_TRUE(result.resolved_moi_persistent_epoch_sgpr);
  EXPECT_EQ(*result.resolved_moi_persistent_epoch_sgpr,
            *result.resolved_moi_persistent_owner_sgpr + 1u);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiInlineEpochBarrier,
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
      result.patches, ConSanPatchKind::TrampolineMoiInlineEpochBarrier, &ConSanPatchInfo::kind);
  ASSERT_NE(barrier_patch, result.patches.end());
  ASSERT_TRUE(result.resolved_moi_exec_save_sgpr);
  const uint16_t scc_save = static_cast<uint16_t>(*result.resolved_moi_exec_save_sgpr + 4u);
  const std::vector<uint32_t> barrier_words = text_words_at_offset(
      patched, barrier_patch->trampoline_offset, barrier_patch->trampoline_size);
  const std::vector<uint32_t> expected_epoch_update = {
      *instrumentation::build_s_cselect_b32(scc_save, scalar_positive_inline_u32(1),
                                            scalar_positive_inline_u32(0),
                                            ROCJITSU_CODE_ARCH_CDNA4),
      *barrier,
      *instrumentation::build_s_cmp_eq_u32(*result.resolved_moi_persistent_epoch_sgpr,
                                           /*literal=*/255u, ROCJITSU_CODE_ARCH_CDNA4),
      consan_moi_exact_shadow::max_epoch,
      *instrumentation::build_s_cbranch_scc1(/*offset_dwords=*/2, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_add_u32(*result.resolved_moi_persistent_epoch_sgpr,
                      *result.resolved_moi_persistent_epoch_sgpr, scalar_positive_inline_u32(1),
                      ROCJITSU_CODE_ARCH_CDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      *instrumentation::build_s_cmp_lg_u32(scc_save, scalar_positive_inline_u32(0),
                                           ROCJITSU_CODE_ARCH_CDNA4),
  };
  EXPECT_TRUE(contains_subsequence(barrier_words, expected_epoch_update));
}

TEST(ConSanMoi, Cdna4AccvgprBoundaryRecordReplayUsesProvenUnusedScalarHole) {
  const auto guest = build_cdna4_ds_store_b32(
      /*vaddr=*/2, /*vdata=*/3, /*byte_offset=*/0, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(guest);
  std::vector<uint32_t> text_words(320u, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  std::copy(guest->begin(), guest->end(), text_words.begin());
  size_t cursor = guest->size();
  text_words[cursor++] =
      build_v_mov_b32_e32(/*vdst=*/0, vector_source_vgpr(7), ROCJITSU_CODE_ARCH_CDNA4);
  // Reference every scalar below s72 and s87 after the access. The only
  // untouched pair below the high guest references is s72:s73; s88:s89 and
  // s92:s96 are reserved explicitly for dispatch and transient state.
  for (uint16_t sgpr = 0u; sgpr < 72u; ++sgpr)
    text_words[cursor++] = build_s_mov_b32(/*sdst=*/87u, sgpr, ROCJITSU_CODE_ARCH_CDNA4);
  text_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/87u, ROCJITSU_CODE_ARCH_CDNA4);
  text_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/90u, ROCJITSU_CODE_ARCH_CDNA4);
  text_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/91u, ROCJITSU_CODE_ARCH_CDNA4);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);
  std::vector<uint8_t> bytes =
      make_cdna4_lds_code_object(text_words, "accvgpr_scalar_hole", /*vgpr_granulated=*/3u);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 1u);
    // 104 decoded SGPRs place physical VCC at s98:s99. Dispatch and transient
    // state already fit below it; the selected hole must not grow the
    // allocation or move VCC.
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

TEST(ConSanMoi, Gfx1250PrivateEpochBarrierPreservesGuestVgprMsbMode) {
  constexpr uint32_t kBarrierSignal = 0xBE804EC1u;
  constexpr uint32_t kBarrierWait = 0xBF94FFFFu;
  constexpr uint16_t kGuestVgprMsbTransition = 0x4004u;
  constexpr uint8_t kGuestVgprMsbMode = 0x04u;
  std::vector<uint32_t> text_words = {
      *build_gfx1250_s_set_vgpr_msb(kGuestVgprMsbTransition, ROCJITSU_CODE_ARCH_GFX1250),
      0xD8340000u,
      0x00000000u, // ds_store_b32 v0, v0
      kBarrierSignal,
      kBarrierWait,
  };
  text_words.resize(320u, build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250));
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250);

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
  const auto barrier = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiInlineEpochBarrier, &ConSanPatchInfo::kind);
  ASSERT_NE(barrier, result.patches.end());
  EXPECT_EQ(barrier->anchor_offset, 4u * sizeof(uint32_t));
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiInlineEpochBarrier,
                               &ConSanPatchInfo::kind),
            1u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> words =
      text_words_at_offset(patched, barrier->trampoline_offset, barrier->trampoline_size);
  const uint32_t select_low =
      *build_gfx1250_s_set_vgpr_msb_transition(kGuestVgprMsbMode, 0u, ROCJITSU_CODE_ARCH_GFX1250);
  const uint32_t restore_guest =
      *build_gfx1250_s_set_vgpr_msb_transition(0u, kGuestVgprMsbMode, ROCJITSU_CODE_ARCH_GFX1250);
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
  // Non-CAS RMWs have no meaningful success mask. Keep the dynamically
  // indexed record body within a bounded append-cave footprint.
  EXPECT_LT(atomic_patch->trampoline_size, 1000u);

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
  const std::array<uint32_t, 4> post_atomic_wait = {(*original_atomic)[0], (*original_atomic)[1],
                                                    (*original_atomic)[2], 0xBFC00000u};
  EXPECT_FALSE(contains_subsequence(trampoline_words, post_atomic_wait));

  const uint64_t base = *options.moi_report_buffer_address;
  const ConSanMoiReportBufferLayout layout = consan_moi_report_buffer_layout_for_bytes(
      options.moi_report_buffer_size, /*include_barriers=*/false, /*include_atomics=*/true,
      /*include_fences=*/true);
  const uint64_t atomic_record_base = base + layout.atomic_records_offset;
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

  const auto expect_dynamic_field_address = [&](uint32_t offset) {
    const auto materialize = build_v_mov_b32_e64_literal(
        *options.scratch_vgpr, static_cast<uint32_t>(atomic_record_base + offset),
        ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_TRUE(materialize);
    EXPECT_TRUE(contains_subsequence(trampoline_words, *materialize));
  };
  expect_dynamic_field_address(offsetof(ConSanMoiAtomicRecord, owner_id));
  expect_dynamic_field_address(offsetof(ConSanMoiAtomicRecord, epoch));
  expect_dynamic_field_address(offsetof(ConSanMoiAtomicRecord, atomic_address));
  expect_dynamic_field_address(offsetof(ConSanMoiAtomicRecord, kind));
  expect_dynamic_field_address(offsetof(ConSanMoiAtomicRecord, operation));
  expect_dynamic_field_address(offsetof(ConSanMoiAtomicRecord, outcome));
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
      ROCJITSU_CODE_ARCH_GFX1250);
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
  constexpr auto load = gfx1250::build_vflat(gfx1250::kFlatLoadB128Vflat,
                                             {.saddr = static_cast<uint8_t>(gfx1250::OPR_SREG_NULL),
                                              .vdst = 8,
                                              .vaddr = 4,
                                              .ioffset = 16});
  const std::array<uint32_t, 8> text_words = {
      0xBE8001EBu, // s_mov_b64 s[0:1], src_shared_base
      build_v_mov_b32_e32(/*vdst=*/4, /*src=*/0, ROCJITSU_CODE_ARCH_GFX1250),
      build_v_mov_b32_e32(/*vdst=*/5, /*src=*/1, ROCJITSU_CODE_ARCH_GFX1250),
      load[0],
      load[1],
      load[2],
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250),
      build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250),
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
  const auto add = build_v_add_u64_signed_i24(/*address_vgpr=*/19, /*displacement=*/16,
                                              ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(add);
  EXPECT_TRUE(contains_subsequence(patched_words, *add));
}

TEST(ConSanMoi, Gfx1250WaveScopeAtomicIsTypedNotApplicable) {
  const auto atomic = build_gfx1250_flat_atomic_add_u32(
      /*vaddr=*/2, /*vsrc=*/1, /*vdst=*/2, /*return_old_value=*/true, /*scope=*/0,
      ROCJITSU_CODE_ARCH_GFX1250);
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
  constexpr auto store = gfx1250::build_vds(gfx1250::kDsStoreB32Vds, {.addr = 4u, .data0 = 5u});
  constexpr auto atomic =
      gfx1250::build_vds(gfx1250::kDsAddU32Vds, {.offset0 = 12u, .addr = 2u, .data0 = 1u});
  const std::array<uint32_t, 6> text_words = {
      store[0],  store[1],  0xBFC90000u, // s_wait_storecnt_dscnt 0
      atomic[0], atomic[1], build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250)};
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
  for (const rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_GFX1250}) {
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
        arch == ROCJITSU_CODE_ARCH_GFX1250
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
  EXPECT_EQ(result.target_name, "gfx950");
  EXPECT_EQ(result.arch_name, "cdna4");
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
  for (const ConSanPatchInfo &host : result.patches) {
    if (host.kind != ConSanPatchKind::TrampolineMoiIndirectBranchIsland || host.original_size == 0u)
      continue;
    for (const ConSanPatchInfo &access : result.patches) {
      if (access.kind != ConSanPatchKind::TrampolineMoiAccessRecordStore)
        continue;
      EXPECT_FALSE(host.anchor_offset < access.anchor_offset + access.original_size &&
                   access.anchor_offset < host.anchor_offset + host.original_size)
          << "host=" << host.anchor_offset << "+" << host.original_size
          << " access=" << access.anchor_offset << "+" << access.original_size;
    }
  }
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
      consan_moi_report_buffer_min_bytes(kAccessCount, 0, 0, 0, kAccessCount);
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
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiInlineEpochBarrier,
                               &ConSanPatchInfo::kind),
            kAccessCount);
}

TEST(ConSanMoi, Cdna4DenseRecordReplayBarriersUseRelocatedRouter) {
  constexpr uint32_t kSiteCount = 33u;
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
      consan_moi_report_buffer_min_bytes(kSiteCount, 0, 0, 0, kSiteCount);
  options.moi_track_barriers = true;
  options.moi_track_atomics = false;
  options.max_patches = 96u;

  const ConSanResult result = try_patch_consan(
      make_cdna4_lds_code_object(text_words, "cdna4_dense_record_replay_barriers"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_EQ(result.target_name, "gfx950");
  EXPECT_EQ(result.arch_name, "cdna4");
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAccessRecordStore,
                               &ConSanPatchInfo::kind),
            kSiteCount);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiInlineEpochBarrier,
                               &ConSanPatchInfo::kind),
            kSiteCount);
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
      consan_moi_report_buffer_min_bytes(kSiteCount, 0, 0, 0, kSiteCount);
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
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiInlineEpochBarrier,
                               &ConSanPatchInfo::kind),
            kSiteCount);
  EXPECT_EQ(std::ranges::count_if(result.site_dispositions,
                                  [](const auto &site) {
                                    return site.site_kind == ConSanResourceSiteKind::Barrier &&
                                           !site.in_kernel &&
                                           site.lowering_outcome ==
                                               ConSanSiteLoweringOutcome::Patched;
                                  }),
            kSiteCount);
}

TEST(ConSanMoi, Gfx1250DenseAccessesPartitionRelayWindowsAcrossLargeKernel) {
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
      8u, build_s_mov_b32(/*sdst=*/0, /*ssrc0=*/0, ROCJITSU_CODE_ARCH_GFX1250));
  for (uint32_t index = 0; index < kAccessCount; ++index) {
    text_words.push_back(0xD8340000u | index * sizeof(uint32_t));
    text_words.push_back(0x00000000u); // ds_store_b32 v0, v0 offset:index*4
  }
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250));
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
      build_s_branch(/*simm16=*/8, ROCJITSU_CODE_ARCH_GFX1250),
  };
  text_words.insert(text_words.end(), 8u,
                    build_s_mov_b32(/*sdst=*/0, /*ssrc0=*/0, ROCJITSU_CODE_ARCH_GFX1250));
  for (uint32_t index = 0; index < kAccessCount; ++index) {
    text_words.push_back(0xD8340000u | index * sizeof(uint32_t));
    text_words.push_back(0x00000000u); // ds_store_b32 v0, v0 offset:index*4
  }
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250));
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

TEST(ConSanMoi, Gfx1250DenseAccessesPreserveGuestVgprMsbMode) {
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
      *build_gfx1250_s_set_vgpr_msb_transition(kGuestVgprMsbMode, 0u, ROCJITSU_CODE_ARCH_GFX1250);
  const uint32_t restore_guest =
      *build_gfx1250_s_set_vgpr_msb_transition(0u, kGuestVgprMsbMode, ROCJITSU_CODE_ARCH_GFX1250);
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

TEST(ConSanMoi, Gfx1250DenseAccessesIgnorePreviousGuestVgprMsbMode) {
  constexpr uint16_t kPreviousOnlyTransition = 0x4400u;
  std::vector<uint32_t> text_words(
      8u, build_s_mov_b32(/*sdst=*/0, /*ssrc0=*/0, ROCJITSU_CODE_ARCH_GFX1250));
  text_words.push_back(
      *build_gfx1250_s_set_vgpr_msb(kPreviousOnlyTransition, ROCJITSU_CODE_ARCH_GFX1250));
  text_words.push_back(0xD8340000u);
  text_words.push_back(0x00000000u); // ds_store_b32 v0, v0
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250));

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

  ASSERT_TRUE(consan_patch_succeeded(result));
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
  ASSERT_GE(words.size(), original_acquire->size() + 2u);
  EXPECT_TRUE(std::equal(original_acquire->begin(), original_acquire->end(), words.begin()));
  EXPECT_EQ(words[original_acquire->size()], 0xBFC00000u);
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
  options.moi_owner_vgpr = 15;
  options.moi_epoch_vgpr = 16;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(2, 0, 0, 0, 0, 2, 2);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  const auto fence = std::ranges::find(result.patches, ConSanPatchKind::TrampolineMoiFenceRecord,
                                       &ConSanPatchInfo::kind);
  ASSERT_NE(fence, result.patches.end());
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
      make_expected_literal_store_words(*options.moi_report_buffer_address +
                                            offsetof(ConSanMoiReportHeader, fence_record_count),
                                        1u, *options.scratch_vgpr)));
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
  options.moi_owner_vgpr = 15;
  options.moi_epoch_vgpr = 16;
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
  const ConSanMoiReportBufferLayout layout = consan_moi_report_buffer_layout_for_bytes(
      options.moi_report_buffer_size, /*include_barriers=*/false, /*include_atomics=*/true,
      /*include_fences=*/true);
  const uint64_t base = *options.moi_report_buffer_address;
  const uint64_t record = base + layout.atomic_records_offset;
  EXPECT_TRUE(contains_subsequence(
      words, make_expected_fetch_add_one_words(
                 base + offsetof(ConSanMoiReportHeader, atomic_record_count),
                 static_cast<uint16_t>(*options.scratch_vgpr + 2u), *options.scratch_vgpr)));
  const auto expect_dynamic_field_address = [&](uint32_t offset) {
    const auto materialize = build_v_mov_b32_e64_literal(
        *options.scratch_vgpr, static_cast<uint32_t>(record + offset), ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_TRUE(materialize);
    EXPECT_TRUE(contains_subsequence(words, *materialize));
  };
  expect_dynamic_field_address(offsetof(ConSanMoiAtomicRecord, operation));
  expect_dynamic_field_address(offsetof(ConSanMoiAtomicRecord, outcome));
  expect_dynamic_field_address(offsetof(ConSanMoiAtomicRecord, lane_mask));
  expect_dynamic_field_address(offsetof(ConSanMoiAtomicRecord, success_lane_mask));
  const auto compare = build_v_cmp_eq_u32_e32_vcc(vector_source_vgpr(/*compare_vgpr=*/2),
                                                  /*old_value_vgpr=*/0, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(compare);
  const auto original_cas = build_flat_atomic_cmpswap_b32_vaddr_vsrc_vdst(
      /*vaddr=*/4, /*vsrc=*/1, /*vdst=*/0, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(original_cas);
  const auto guest =
      std::search(words.begin(), words.end(), original_cas->begin(), original_cas->end());
  const auto outcome_compare = std::find(words.begin(), words.end(), *compare);
  ASSERT_NE(guest, words.end());
  ASSERT_NE(outcome_compare, words.end());
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
  EXPECT_EQ(result.resource_plan_summary.emitted_spill_slot_bytes, 40u);
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
  EXPECT_TRUE(result.resolved_moi_transient_sgpr_assignments.front().spill_backed);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiAtomicRecord,
                               &ConSanPatchInfo::kind),
            2u);
  EXPECT_TRUE(std::ranges::all_of(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind != ConSanPatchKind::TrampolineMoiAtomicRecord ||
           patch.required_private_segment_size > 0u;
  }));
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiFenceRecord,
                               &ConSanPatchInfo::kind),
            1u);
  EXPECT_TRUE(std::ranges::all_of(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind != ConSanPatchKind::TrampolineMoiFenceRecord ||
           patch.required_private_segment_size > 0u;
  }));
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
  EXPECT_EQ(result.target_name, "gfx942");
  EXPECT_EQ(result.arch_name, "cdna3");
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
  EXPECT_EQ(result.target_name, "gfx950");
  EXPECT_EQ(result.arch_name, "cdna4");
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
  ASSERT_TRUE(result.resolved_moi_persistent_workgroup_key_sgpr);
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
