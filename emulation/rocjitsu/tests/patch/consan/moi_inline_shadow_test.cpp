// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"
#include "embedded_schema.h"
#include "rocjitsu/code/patch/consan/consan_moi_internal.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"
#include "rocjitsu/config/config_loader.h"

namespace rocjitsu {
namespace {

namespace ib = instrumentation;

TEST(ConSanMoi, InlineShadowProbePublishesNativeLdsStoreToExactShadow) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 24;
  options.moi_epoch_vgpr = 25;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_EQ(result.flavor, ConSanFlavor::Moi);
  EXPECT_EQ(result.moi_engine, ConSanMoiEngine::InlineShadow);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::TrampolineMoiExactShadowStore);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 8u);
  const auto access_plan = std::ranges::find(result.resource_plans, ConSanResourceSiteKind::Access,
                                             &ConSanCandidateResourcePlan::site_kind);
  ASSERT_NE(access_plan, result.resource_plans.end());
  EXPECT_EQ(access_plan->scratch_vgpr_count, 16u);
  ASSERT_EQ(result.kernels.size(), 1u);
  EXPECT_TRUE(result.kernels.front().decoded);
  bool saw_inline_shadow_warning = false;
  for (const std::string &warning : result.warnings)
    saw_inline_shadow_warning |= warning.find("exact-shadow publish probe") != std::string::npos;
  EXPECT_TRUE(saw_inline_shadow_warning);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const auto *text_section = patched.text_sections().front();
  ASSERT_EQ(text_section->size() % sizeof(uint32_t), 0u);
  std::vector<uint32_t> text_words(text_section->size() / sizeof(uint32_t));
  std::memcpy(text_words.data(), text_section->data(), text_section->size());

  const ConSanMoiReportBufferLayout layout =
      consan_moi_inline_shadow_report_buffer_layout_for_bytes(options.moi_report_buffer_size);
  const uint64_t exact_shadow_base =
      *options.moi_report_buffer_address + layout.exact_shadow_entries_offset;
  const uint32_t low_literal = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Write);
  const std::array<uint32_t, 2> expected_original_access = {
      0xD8340000u,
      0x00000000u,
  };
  EXPECT_TRUE(contains_subsequence(text_words, expected_original_access));
  std::vector<uint32_t> expected_publish_prefix;
  const auto mov_low = build_v_mov_b32_e64_literal(10, low_literal, ROCJITSU_CODE_ARCH_RDNA4);
  const auto owner_mask = build_v_and_b32_e32_literal(12, consan_moi_exact_shadow::max_owner, 24,
                                                      ROCJITSU_CODE_ARCH_RDNA4);
  const auto owner_shift =
      build_v_lshlrev_b32_e32(12, scalar_positive_inline_u32(consan_moi_exact_shadow::owner_shift),
                              12, ROCJITSU_CODE_ARCH_RDNA4);
  const auto owner_add =
      build_v_add_nc_u32_e32(10, vector_source_vgpr(10), 12, ROCJITSU_CODE_ARCH_RDNA4);
  const auto epoch_mask = build_v_and_b32_e32_literal(12, consan_moi_exact_shadow::max_epoch, 25,
                                                      ROCJITSU_CODE_ARCH_RDNA4);
  const auto epoch_shift =
      build_v_lshlrev_b32_e32(12, scalar_positive_inline_u32(consan_moi_exact_shadow::epoch_shift),
                              12, ROCJITSU_CODE_ARCH_RDNA4);
  const auto epoch_add =
      build_v_add_nc_u32_e32(10, vector_source_vgpr(10), 12, ROCJITSU_CODE_ARCH_RDNA4);
  const auto mov_high = build_v_mov_b32_e64_literal(11, 0, ROCJITSU_CODE_ARCH_RDNA4);
  const auto mov_address_lo = build_v_mov_b32_e64_literal(
      8, static_cast<uint32_t>(exact_shadow_base), ROCJITSU_CODE_ARCH_RDNA4);
  const auto mov_address_hi = build_v_mov_b32_e64_literal(
      9, static_cast<uint32_t>(exact_shadow_base >> 32u), ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(result.resolved_moi_dispatch_id_sgpr);
  const uint32_t dispatch_bank_stride =
      (layout.exact_shadow_entry_capacity / layout.inline_exact_dispatch_bank_count) *
      sizeof(ConSanMoiInlineExactShadowSlot);
  const uint32_t copy_dispatch_id =
      build_v_mov_b32_e32(12, *result.resolved_moi_dispatch_id_sgpr, ROCJITSU_CODE_ARCH_RDNA4);
  const auto mix_workgroup_key = build_v_xor_b32_e32(
      12, vector_source_vgpr(/*transaction workgroup key=*/13u), 12, ROCJITSU_CODE_ARCH_RDNA4);
  const auto select_dispatch_bank = build_v_and_b32_e32_literal(
      12, layout.inline_exact_dispatch_bank_count - 1u, 12, ROCJITSU_CODE_ARCH_RDNA4);
  const auto scale_dispatch_bank =
      build_v_mul_lo_u32_vop3_literal(12, dispatch_bank_stride, 12, ROCJITSU_CODE_ARCH_RDNA4);
  const auto start_cell_shift = build_v_lshrrev_b32_e32(
      12, scalar_positive_inline_u32(consan_moi_exact_shadow::granule_shift), 0,
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto byte_index_shift =
      build_v_lshlrev_b32_e32(12, scalar_positive_inline_u32(3), 12, ROCJITSU_CODE_ARCH_RDNA4);
  const auto address_add = build_v_add_u64_vgpr_offset(8, 12, ROCJITSU_CODE_ARCH_RDNA4);
  const auto high_cell_shift =
      build_v_lshlrev_b32_e32(12, scalar_positive_inline_u32(1), 12, ROCJITSU_CODE_ARCH_RDNA4);
  const auto atomic_swap = build_flat_atomic_swap_b64_vaddr_vsrc_vdst(
      8, 10, 13, /*return_old_value=*/true, /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(mov_low);
  ASSERT_TRUE(owner_mask);
  ASSERT_TRUE(owner_shift);
  ASSERT_TRUE(owner_add);
  ASSERT_TRUE(epoch_mask);
  ASSERT_TRUE(epoch_shift);
  ASSERT_TRUE(epoch_add);
  ASSERT_TRUE(mov_high);
  ASSERT_TRUE(mov_address_lo);
  ASSERT_TRUE(mov_address_hi);
  ASSERT_TRUE(mix_workgroup_key);
  ASSERT_TRUE(select_dispatch_bank);
  ASSERT_TRUE(scale_dispatch_bank);
  ASSERT_TRUE(start_cell_shift);
  ASSERT_TRUE(byte_index_shift);
  ASSERT_TRUE(address_add);
  ASSERT_TRUE(high_cell_shift);
  ASSERT_TRUE(atomic_swap);
  expected_publish_prefix.insert(expected_publish_prefix.end(), mov_low->begin(), mov_low->end());
  expected_publish_prefix.insert(expected_publish_prefix.end(), owner_mask->begin(),
                                 owner_mask->end());
  expected_publish_prefix.push_back(*owner_shift);
  expected_publish_prefix.push_back(*owner_add);
  expected_publish_prefix.insert(expected_publish_prefix.end(), epoch_mask->begin(),
                                 epoch_mask->end());
  expected_publish_prefix.push_back(*epoch_shift);
  expected_publish_prefix.push_back(*epoch_add);
  expected_publish_prefix.insert(expected_publish_prefix.end(), mov_high->begin(), mov_high->end());
  EXPECT_TRUE(contains_subsequence(text_words, expected_publish_prefix));
  std::vector<uint32_t> expected_address;
  expected_address.insert(expected_address.end(), mov_address_lo->begin(), mov_address_lo->end());
  expected_address.insert(expected_address.end(), mov_address_hi->begin(), mov_address_hi->end());
  expected_address.push_back(copy_dispatch_id);
  expected_address.push_back(*mix_workgroup_key);
  expected_address.insert(expected_address.end(), select_dispatch_bank->begin(),
                          select_dispatch_bank->end());
  expected_address.insert(expected_address.end(), scale_dispatch_bank->begin(),
                          scale_dispatch_bank->end());
  expected_address.insert(expected_address.end(), address_add->begin(), address_add->end());
  expected_address.push_back(*start_cell_shift);
  expected_address.push_back(*byte_index_shift);
  expected_address.insert(expected_address.end(), address_add->begin(), address_add->end());
  expected_address.push_back(*high_cell_shift);
  expected_address.insert(expected_address.end(), address_add->begin(), address_add->end());
  EXPECT_TRUE(contains_subsequence(text_words, expected_address));
  const auto second_cell_offset = build_v_add_u64_signed_i24(
      /*address_vgpr=*/8, sizeof(ConSanMoiInlineExactShadowSlot), ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(second_cell_offset);
  EXPECT_EQ(count_subsequence(text_words, *second_cell_offset), 1u)
      << "a potentially unaligned dword must publish its second possible cell explicitly";
  const auto second_cell_bytes = ib::build_v_mov_b32_literal(
      /*vdst=*/12u, consan_moi_exact_shadow::granule_bytes, ROCJITSU_CODE_ARCH_RDNA4);
  const auto clamp_second_cell =
      ib::build_v_min_u32(/*vdst=*/12u, vector_source_vgpr(/*access_end=*/11u),
                          /*vsrc1=*/12u, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(second_cell_bytes);
  ASSERT_TRUE(clamp_second_cell);
  std::vector<uint32_t> second_cell_provenance = *second_cell_bytes;
  second_cell_provenance.push_back(*clamp_second_cell);
  EXPECT_EQ(count_subsequence(text_words, second_cell_provenance), 2u)
      << "both versioned publication paths must derive an empty or partial second-cell mask";
  const uint16_t loop_counter_vgpr = consan_detail::inline_shadow_loop_counter_vgpr(
      /*scratch_vgpr=*/8u, /*has_exec_save=*/true, options.moi_track_atomics);
  EXPECT_EQ(std::ranges::count(text_words,
                               build_v_mov_b32_e32(loop_counter_vgpr, scalar_positive_inline_u32(0),
                                                   ROCJITSU_CODE_ARCH_RDNA4)),
            0u)
      << "narrow unaligned accesses must not reference unplanned loop registers";
  EXPECT_EQ(count_subsequence(text_words, *atomic_swap), 0u);

  const auto version_load = build_flat_load_b32_vaddr_vdst(
      /*vaddr=*/8, /*vdst=*/21, ROCJITSU_CODE_ARCH_RDNA4,
      offsetof(ConSanMoiInlineExactShadowSlot, version));
  const auto version_cas = build_flat_atomic_cmpswap_b32_vaddr_vsrc_vdst(
      /*vaddr=*/8, /*vsrc=*/22, /*vdst=*/22, /*return_old_value=*/true,
      /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  const auto dispatch_store_low = build_flat_store_b32_vaddr_vsrc(
      /*vaddr=*/8, /*vsrc=*/10, ROCJITSU_CODE_ARCH_RDNA4,
      offsetof(ConSanMoiInlineExactShadowSlot, dispatch_id));
  const auto dispatch_store_high = build_flat_store_b32_vaddr_vsrc(
      /*vaddr=*/8, /*vsrc=*/11, ROCJITSU_CODE_ARCH_RDNA4,
      offsetof(ConSanMoiInlineExactShadowSlot, dispatch_id) + sizeof(uint32_t));
  ASSERT_TRUE(version_load);
  ASSERT_TRUE(version_cas);
  ASSERT_TRUE(dispatch_store_low);
  ASSERT_TRUE(dispatch_store_high);
  const auto wait_load = instrumentation::build_s_wait_global_load0(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(wait_load);
  EXPECT_EQ(count_subsequence(text_words, *version_load), 4u)
      << "both possible cells and both transaction paths must read prior versions";
  EXPECT_EQ(count_subsequence(text_words, *version_cas), 8u)
      << "both possible cells and both transaction paths must claim odd and commit even";
  std::vector<uint32_t> drained_version_cas(version_cas->begin(), version_cas->end());
  drained_version_cas.push_back(*wait_load);
  EXPECT_EQ(count_subsequence(text_words, drained_version_cas), 8u)
      << "every gfx12 version claim and commit must wait for its returned value";
  EXPECT_EQ(count_subsequence(text_words, *dispatch_store_low), 4u);
  EXPECT_EQ(count_subsequence(text_words, *dispatch_store_high), 4u);
  EXPECT_TRUE(contains_subsequence(
      text_words,
      std::array<uint32_t, 2>{
          build_v_mov_b32_e32(10, *result.resolved_moi_dispatch_id_sgpr, ROCJITSU_CODE_ARCH_RDNA4),
          build_v_mov_b32_e32(11, *result.resolved_moi_dispatch_id_sgpr + 1u,
                              ROCJITSU_CODE_ARCH_RDNA4)}));
}

TEST(ConSanMoi, Cdna4InlineShadowProbeEmitsNativeTransactions) {
  // Leave enough dense padding for the complete exact-byte transaction so
  // this fixture continues to exercise the native inline placement.
  std::vector<uint32_t> text_words(2400, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  text_words[0] = 0xd81a0004u;
  text_words[1] = 0x00000302u; // ds_write_b32 v2, v3 offset:4
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);

  const std::vector<uint8_t> bytes = make_cdna4_lds_code_object(text_words);
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 24;
  options.moi_epoch_vgpr = 25;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_TRUE(result.final_validation_passed);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineMoiExactShadowStore);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 8u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const Section *text = patched.text_sections().front();
  std::vector<uint32_t> patched_words(text->size() / sizeof(uint32_t));
  std::memcpy(patched_words.data(), text->data(), text->size());

  const auto version_load = build_cdna4_flat_load_b32(
      /*vaddr=*/8, /*vdst=*/21, offsetof(ConSanMoiInlineExactShadowSlot, version),
      ROCJITSU_CODE_ARCH_CDNA4);
  const auto version_cas = build_cdna4_flat_atomic_cmpswap_b32(
      /*vaddr=*/8, /*vsrc=*/22, /*vdst=*/22, /*return_old_value=*/true,
      /*scope=*/2, ROCJITSU_CODE_ARCH_CDNA4);
  const auto retry_invalidate = build_cdna4_buffer_inv_sc1(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(version_load && version_cas && retry_invalidate);
  EXPECT_EQ(count_subsequence(patched_words, *version_load), 4u);
  EXPECT_EQ(count_subsequence(patched_words, *version_cas), 8u);
  EXPECT_EQ(count_subsequence(patched_words, *retry_invalidate), 4u)
      << "both transaction paths for both possible cells must invalidate stale payload on retry";
  EXPECT_GE(std::count(patched_words.begin(), patched_words.end(), 0xbf8c0f70u), 1);
  const std::array<uint32_t, 2> guest_store = {text_words[0], text_words[1]};
  const auto first_shadow_transaction = std::search(patched_words.begin(), patched_words.end(),
                                                    version_load->begin(), version_load->end());
  const auto displaced_store = std::search(patched_words.begin(), patched_words.end(),
                                           guest_store.begin(), guest_store.end());
  ASSERT_NE(first_shadow_transaction, patched_words.end());
  ASSERT_NE(displaced_store, patched_words.end());
  EXPECT_LT(first_shadow_transaction, displaced_store)
      << "Inline shadow publication must precede the displaced guest store";
}

TEST(ConSanMoi, CdnaInlineShadowMovesOnlyAnEmptyAccumulatorBoundaryForScratchGrowth) {
  for (const rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    SCOPED_TRACE(arch);
    std::vector<uint32_t> text_words(1200u, build_s_nop(0, arch));
    if (arch == ROCJITSU_CODE_ARCH_CDNA3) {
      const auto store =
          build_cdna3_ds_store_b32(/*vaddr=*/2u, /*vdata=*/3u, /*byte_offset=*/0u, arch);
      ASSERT_TRUE(store);
      std::copy(store->begin(), store->end(), text_words.begin());
    } else {
      const auto store =
          build_cdna4_ds_store_b32(/*vaddr=*/2u, /*vdata=*/3u, /*byte_offset=*/0u, arch);
      ASSERT_TRUE(store);
      std::copy(store->begin(), store->end(), text_words.begin());
    }
    text_words.back() = build_s_endpgm(arch);
    std::vector<uint8_t> bytes =
        arch == ROCJITSU_CODE_ARCH_CDNA3
            ? make_cdna3_lds_code_object(text_words, "empty_accumulator_boundary",
                                         /*vgpr_granulated=*/0u)
            : make_cdna4_lds_code_object(text_words, "empty_accumulator_boundary",
                                         /*vgpr_granulated=*/0u);
    mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
      // The eight-register unified allocation ends exactly at v8. There is no
      // allocated accumulator bank whose mapping needs to be preserved.
      AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 1u);
    });

    ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
    options.moi_track_atomics = false;
    options.moi_track_barriers = false;
    options.moi_report_buffer_address = 0x100000000ull;
    options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

    const ConSanResult result = try_patch_consan(bytes, options);

    ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
    ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
    ASSERT_EQ(result.resource_plans.size(), 1u);
    EXPECT_EQ(result.resource_plans.front().source,
              ConSanRegisterAllocationSource::DescriptorGrowth);
    EXPECT_EQ(result.resource_plans.front().scratch_vgpr, 8u);
    EXPECT_EQ(result.resource_plans.front().scratch_vgpr_count, 16u);
    EXPECT_EQ(result.resource_plans.front().required_vgpr_count, 24u);

    AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(patched.is_valid());
    ASSERT_EQ(patched.kernels().size(), 1u);
    KD descriptor{};
    std::memcpy(&descriptor,
                result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
                sizeof(descriptor));
    EXPECT_EQ(AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc1,
                              kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT),
              2u);
    EXPECT_EQ(
        AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET),
        5u);
    EXPECT_TRUE(result.final_validation_passed);
  }
}

TEST(ConSanMoi, Cdna4InlineShadowRecoversFullWindowKernargPreloadTail) {
  std::vector<uint32_t> text_words(1200, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  text_words[0] = 0xd81a0004u;
  text_words[1] = 0x00000302u; // ds_write_b32 v2, v3 offset:4
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);
  std::vector<uint8_t> bytes =
      make_cdna4_lds_code_object(text_words, "full_window_kernarg_preload",
                                 /*vgpr_granulated=*/0u, /*uses_dynamic_stack=*/false,
                                 /*workgroup_id_dimension_mask=*/0u);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                    kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 15u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X,
                    1u);
    AMDHSA_BITS_SET(descriptor.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_LENGTH, 13u);
    AMDHSA_BITS_SET(descriptor.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_OFFSET, 3u);
  });
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 24;
  options.moi_epoch_vgpr = 25;
  options.moi_init_owner_epoch = true;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  const auto prologue = std::ranges::find(
      result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue, &ConSanPatchInfo::kind);
  ASSERT_NE(prologue, result.patches.end());
  EXPECT_EQ(prologue->dispatch_id_source_sgpr, 2u);
  EXPECT_EQ(prologue->dispatch_id_original_user_sgpr_count, 15u);
  EXPECT_EQ(prologue->dispatch_id_expanded_user_sgpr_count, 16u);
  EXPECT_EQ(prologue->dispatch_id_kernarg_reload_sgpr, 14u);
  EXPECT_EQ(prologue->dispatch_id_kernarg_reload_base_sgpr, 0u);
  EXPECT_EQ(prologue->dispatch_id_kernarg_reload_offset_dwords, 15u);
  EXPECT_EQ(prologue->dispatch_id_kernarg_reload_count, 1u);
  EXPECT_EQ(prologue->dispatch_id_shifted_system_sgpr_count, 1u);
  EXPECT_EQ(prologue->dispatch_id_system_sgpr_shift, 1u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  KD descriptor{};
  std::memcpy(&descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  EXPECT_EQ(AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT),
            16u);
  EXPECT_EQ(AMDHSA_BITS_GET(descriptor.kernel_code_properties,
                            kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID),
            1u);
  EXPECT_EQ(AMDHSA_BITS_GET(descriptor.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_LENGTH), 12u);
  EXPECT_EQ(AMDHSA_BITS_GET(descriptor.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_OFFSET), 3u);
}

TEST(ConSanMoi, Cdna3InlineShadowRecoversFullWindowKernargPreloadTail) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA3;
  std::vector<uint32_t> text_words(1200, build_s_nop(0, kArch));
  const auto store = build_cdna3_ds_store_b32(/*vaddr=*/2, /*vdata=*/3, /*byte_offset=*/4, kArch);
  ASSERT_TRUE(store);
  std::ranges::copy(*store, text_words.begin());
  text_words.back() = build_s_endpgm(kArch);
  std::vector<uint8_t> bytes =
      make_cdna3_lds_code_object(text_words, "full_window_kernarg_preload",
                                 /*vgpr_granulated=*/0u, /*uses_dynamic_stack=*/false,
                                 /*workgroup_id_dimension_mask=*/0u);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                    kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 15u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X,
                    1u);
    AMDHSA_BITS_SET(descriptor.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_LENGTH, 13u);
    AMDHSA_BITS_SET(descriptor.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_OFFSET, 3u);
  });
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 24;
  options.moi_epoch_vgpr = 25;
  options.moi_init_owner_epoch = true;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  const auto prologue = std::ranges::find(
      result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue, &ConSanPatchInfo::kind);
  ASSERT_NE(prologue, result.patches.end());
  EXPECT_EQ(prologue->dispatch_id_source_sgpr, 2u);
  EXPECT_EQ(prologue->dispatch_id_original_user_sgpr_count, 15u);
  EXPECT_EQ(prologue->dispatch_id_expanded_user_sgpr_count, 16u);
  EXPECT_EQ(prologue->dispatch_id_kernarg_reload_sgpr, 14u);
  EXPECT_EQ(prologue->dispatch_id_kernarg_reload_base_sgpr, 0u);
  EXPECT_EQ(prologue->dispatch_id_kernarg_reload_offset_dwords, 15u);
  EXPECT_EQ(prologue->dispatch_id_kernarg_reload_count, 1u);
  EXPECT_EQ(prologue->dispatch_id_shifted_system_sgpr_count, 1u);
  EXPECT_EQ(prologue->dispatch_id_system_sgpr_shift, 1u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  KD descriptor{};
  std::memcpy(&descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  EXPECT_EQ(AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT),
            16u);
  EXPECT_EQ(AMDHSA_BITS_GET(descriptor.kernel_code_properties,
                            kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID),
            1u);
  EXPECT_EQ(AMDHSA_BITS_GET(descriptor.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_LENGTH), 12u);
  EXPECT_EQ(AMDHSA_BITS_GET(descriptor.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_OFFSET), 3u);
}

TEST(ConSanMoi, Cdna4InlineShadowPreservesDsWorkgroupKeyFromKernelEntry) {
  std::vector<uint32_t> text_words(1200, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  const auto long_entry = ib::build_v_mov_b32_literal(
      /*vdst=*/4u, /*literal=*/0x12345678u, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(long_entry);
  ASSERT_EQ(long_entry->size(), 2u);
  std::ranges::copy(*long_entry, text_words.begin());
  text_words[20] = 0xd81a0004u;
  text_words[21] = 0x00000302u; // ds_write_b32 v2, v3 offset:4
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);
  std::vector<uint8_t> bytes =
      make_cdna4_lds_code_object(text_words, "persistent_ds_workgroup", /*vgpr_granulated=*/0u,
                                 /*uses_dynamic_stack=*/false, /*workgroup_id_dimension_mask=*/0x7u,
                                 /*group_segment_fixed_size=*/4352u);
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
    AMDHSA_BITS_SET(descriptor.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_LENGTH, 1u);
    AMDHSA_BITS_SET(descriptor.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_OFFSET, 3u);
  });
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_inline_workgroup_shadow = true;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_TRUE(result.moi_persistent_vgprs_automatic);
  ASSERT_TRUE(result.resolved_moi_epoch_vgpr);
  ASSERT_TRUE(result.resolved_moi_workgroup_key_vgpr);
  EXPECT_EQ(*result.resolved_moi_workgroup_key_vgpr, *result.resolved_moi_epoch_vgpr + 1u);
  const auto prologue = std::ranges::find(
      result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue, &ConSanPatchInfo::kind);
  ASSERT_NE(prologue, result.patches.end());
  EXPECT_FALSE(prologue->workgroup_shadow_lazy_initialization);
  EXPECT_EQ(prologue->workgroup_shadow_validity_size, 0u);
  ASSERT_TRUE(result.resolved_moi_exec_save_sgpr);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> prologue_words =
      text_words_at_offset(patched, prologue->trampoline_offset, prologue->trampoline_size);
  const auto select_invalid = instrumentation::build_s_andn2_b64(
      kRdna4ExecLo, static_cast<uint16_t>(*result.resolved_moi_exec_save_sgpr + 20u),
      static_cast<uint16_t>(*result.resolved_moi_exec_save_sgpr + 8u), ROCJITSU_CODE_ARCH_CDNA4);
  const auto zero_invalid = instrumentation::build_v_mov_b32_literal(
      *result.resolved_moi_workgroup_key_vgpr, 0u, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(select_invalid && zero_invalid);
  EXPECT_NE(std::ranges::find(prologue_words, *select_invalid), prologue_words.end());
  EXPECT_TRUE(contains_subsequence(prologue_words, *zero_invalid));
  // Capturing and normalizing the entry workgroup key grows the paired
  // kernarg-preload bodies beyond their 256-byte hardware entry windows.
  // Both hardware entries therefore remain short relays to equivalent
  // expanded bodies.
  ASSERT_TRUE(prologue->dispatch_id_primary_prologue_offset);
  ASSERT_TRUE(prologue->dispatch_id_secondary_prologue_offset);
  EXPECT_LT(*prologue->dispatch_id_primary_prologue_offset,
            *prologue->dispatch_id_secondary_prologue_offset);
}

TEST(ConSanMoi, Cdna4InlineShadowAvoidsOriginalPhysicalVccPair) {
  std::vector<uint32_t> text_words(1200, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  text_words[0] = 0xd81a0004u;
  text_words[1] = 0x00000302u; // ds_write_b32 v2, v3 offset:4
  const auto reference_s67 =
      ib::build_s_cmp_eq_u32(/*ssrc0=*/67u, /*ssrc1=*/0u, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(reference_s67);
  text_words[2] = *reference_s67;
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);
  std::vector<uint8_t> bytes = make_cdna4_lds_code_object(text_words, "physical_vcc_boundary");
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    // Nine allocation quanta give the original kernel 72 SGPRs. CDNA4 keeps
    // a six-register allocation tail and therefore places VCC at s66:s67.
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 8u);
  });
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.resolved_moi_dispatch_id_sgpr);
  ASSERT_TRUE(result.resolved_moi_exec_save_sgpr);
  // Growing to 80 SGPRs moves VCC to s74:s75, so s72:s73 is a safe dispatch
  // pair. The large transient window remains inside the original allocation.
  EXPECT_EQ(*result.resolved_moi_dispatch_id_sgpr, 72u);
  EXPECT_EQ(*result.resolved_moi_exec_save_sgpr, 2u);
}

TEST(ConSanMoi, Cdna4InlineShadowForcedSpillRotatesLocalExchangeTuple) {
  std::vector<uint32_t> text_words(1200, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  text_words[0] = 0xd81a0004u;
  text_words[1] = 0x00000302u; // ds_write_b32 v2, v3 offset:4
  text_words[2] =
      build_v_mov_b32_e32(/*vdst=*/0, vector_source_vgpr(255), ROCJITSU_CODE_ARCH_CDNA4);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);
  constexpr uint32_t kCdna4Wave64AllVgprsGranulated = 31u;
  const std::vector<uint8_t> bytes =
      make_cdna4_lds_code_object(text_words, "forced_spill", kCdna4Wave64AllVgprsGranulated,
                                 /*uses_dynamic_stack=*/false, /*workgroup_id_dimension_mask=*/0u,
                                 /*group_segment_fixed_size=*/4u);
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.force_vgpr_spill = true;
  options.moi_inline_workgroup_shadow = true;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().source, ConSanRegisterAllocationSource::SpillRequired);
  EXPECT_EQ(result.resource_plans.front().scratch_vgpr_count, 16u);
  ASSERT_TRUE(result.resource_plans.front().scratch_vgpr);
  EXPECT_EQ(*result.resource_plans.front().scratch_vgpr, 4u);

  const auto patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &item) {
    return item.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  });
  ASSERT_NE(patch, result.patches.end());
  ASSERT_TRUE(patch->scratch_vgpr);
  EXPECT_EQ(patch->spilled_vgpr_count, 16u);
  ASSERT_EQ(patch->persistent_epoch_private_offset, 0u);
  EXPECT_FALSE(patch->persistent_owner_private_offset);
  ASSERT_EQ(patch->persistent_workgroup_key_private_offset, 4u);
  ASSERT_EQ(patch->persistent_private_state_end, 8u);
  EXPECT_GT(patch->required_private_segment_size, 0u);
  EXPECT_GT(patch->workgroup_shadow_size, 0u);
  EXPECT_FALSE(patch->workgroup_shadow_lazy_initialization);
  EXPECT_EQ(patch->workgroup_shadow_validity_size, 0u);

  const auto prologue = std::ranges::find(
      result.patches, ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue, &ConSanPatchInfo::kind);
  ASSERT_NE(prologue, result.patches.end());
  ASSERT_EQ(prologue->persistent_workgroup_key_private_offset, 4u);
  EXPECT_EQ(prologue->persistent_private_state_end, 8u);
  EXPECT_EQ(prologue->spilled_vgpr_count, 3u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  std::vector<uint32_t> patch_words(patch->trampoline_size / sizeof(uint32_t));
  std::memcpy(patch_words.data(),
              patched.text_sections().front()->data() + patch->trampoline_offset,
              patch->trampoline_size);
  const auto private_key_load = ib::build_private_load_b32(
      static_cast<uint16_t>(*patch->scratch_vgpr + 5u), 8u, ROCJITSU_CODE_ARCH_CDNA4);
  const auto private_key_wait = ib::build_s_wait_private_load0(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(private_key_load && private_key_wait);
  std::vector<uint32_t> expected_private_key_load = *private_key_load;
  expected_private_key_load.push_back(*private_key_wait);
  EXPECT_FALSE(contains_subsequence(patch_words, expected_private_key_load))
      << "an eagerly initialized local mirror uses its generation field for exact bytes";
  const auto local_exchange = build_cdna4_ds_storexchg_rtn_b64(
      static_cast<uint16_t>(*patch->scratch_vgpr + 6u), *patch->scratch_vgpr,
      static_cast<uint16_t>(*patch->scratch_vgpr + 2u), /*byte_offset=*/0u,
      ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(local_exchange);
  EXPECT_EQ(count_subsequence(patch_words, *local_exchange), 2u)
      << "a potentially unaligned dword must exchange both possible local cells";

  const auto pack_lane = ib::build_v_lshlrev_b32(
      static_cast<uint16_t>(*patch->scratch_vgpr + 3u),
      scalar_positive_inline_u32(consan_moi_exact_byte_cell::lane_plus_one_shift),
      static_cast<uint16_t>(*patch->scratch_vgpr + 3u), ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(pack_lane);
  EXPECT_NE(std::ranges::find(patch_words, *pack_lane), patch_words.end())
      << "the full local cell must carry exact-byte representative-lane evidence";

  // The exact-byte provenance inputs occupy the preceding transaction
  // registers; the diagnostic compare-swap tuple is pinned to v14:v15.
  const uint16_t diagnostic_slot_vgpr = static_cast<uint16_t>(*patch->scratch_vgpr + 14u);
  const auto diagnostic_claim = build_cdna4_flat_atomic_cmpswap_b32(
      *patch->scratch_vgpr, diagnostic_slot_vgpr, diagnostic_slot_vgpr,
      /*return_old_value=*/true, /*scope=*/2u, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(diagnostic_claim);
  std::vector<uint32_t> expected_diagnostic_claim = {
      build_v_mov_b32_e32(diagnostic_slot_vgpr, scalar_positive_inline_u32(1u),
                          ROCJITSU_CODE_ARCH_CDNA4),
      build_v_mov_b32_e32(static_cast<uint16_t>(diagnostic_slot_vgpr + 1u),
                          scalar_positive_inline_u32(0u), ROCJITSU_CODE_ARCH_CDNA4),
  };
  expected_diagnostic_claim.insert(expected_diagnostic_claim.end(), diagnostic_claim->begin(),
                                   diagnostic_claim->end());
  EXPECT_EQ(count_subsequence(patch_words, expected_diagnostic_claim), 2u)
      << "each possible cell must initialize both diagnostic compare-swap tuple words";

  const uint16_t loop_counter_vgpr = consan_detail::inline_shadow_loop_counter_vgpr(
      *patch->scratch_vgpr, /*has_exec_save=*/true, options.moi_track_atomics);
  const auto safe_cell_scale = ib::build_v_mul_lo_u32_literal(
      static_cast<uint16_t>(*patch->scratch_vgpr + 4u), diagnostic_slot_vgpr,
      consan_moi_exact_shadow::granule_bytes, loop_counter_vgpr, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(safe_cell_scale);
  EXPECT_FALSE(contains_subsequence(patch_words, *safe_cell_scale))
      << "narrow accesses must unroll rather than use unplanned loop state";

  // The guest address v2 lies below the spill window and remains live until
  // the displaced store executes. No phase-local address recovery is needed.
  const uint16_t address_snapshot = static_cast<uint16_t>(*patch->scratch_vgpr + 15u);
  EXPECT_EQ(
      std::ranges::find(patch_words, build_v_mov_b32_e32(address_snapshot, vector_source_vgpr(2u),
                                                         ROCJITSU_CODE_ARCH_CDNA4)),
      patch_words.end());
}

TEST(ConSanMoi, CdnaInlineShadowClobberingLoadFitsBelowAccumulatorBoundary) {
  for (const rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    SCOPED_TRACE(arch);
    std::vector<uint32_t> guest;
    if (arch == ROCJITSU_CODE_ARCH_CDNA3) {
      const auto load = build_cdna3_ds_load_b32(
          /*vdst=*/6u, /*vaddr=*/6u, /*byte_offset=*/0u, arch);
      ASSERT_TRUE(load);
      guest.assign(load->begin(), load->end());
    } else {
      constexpr auto load = cdna4::build_ds(cdna4::kDsReadB32Ds, {.addr = 6u, .vdst = 6u});
      guest.assign(load.begin(), load.end());
    }
    std::vector<uint32_t> text_words(1200u, build_s_nop(0, arch));
    std::copy(guest.begin(), guest.end(), text_words.begin());
    size_t cursor = guest.size();
    for (uint16_t vgpr = 0u; vgpr < 16u; ++vgpr) {
      text_words[cursor++] = build_v_mov_b32_e32(/*vdst=*/15u, vector_source_vgpr(vgpr), arch);
    }
    text_words.back() = build_s_endpgm(arch);
    std::vector<uint8_t> bytes =
        arch == ROCJITSU_CODE_ARCH_CDNA3
            ? make_cdna3_lds_code_object(text_words, "sixteen_vgpr_inline_shadow",
                                         /*vgpr_granulated=*/2u)
            : make_cdna4_lds_code_object(text_words, "sixteen_vgpr_inline_shadow",
                                         /*vgpr_granulated=*/2u);
    mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
      // Encoded 3 makes v16 the first accumulator register. The access probe
      // must preserve that compiler-defined split.
      AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 3u);
    });
    append_kernel_metadata_note(bytes, "sixteen_vgpr_inline_shadow",
                                /*uses_dynamic_stack=*/false, /*sgpr_count=*/0u,
                                /*private_segment_fixed_size=*/std::nullopt,
                                /*required_workgroup_size=*/std::nullopt,
                                /*has_dynamic_lds=*/true);

    ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
    options.moi_track_atomics = false;
    options.moi_track_barriers = false;
    options.moi_report_buffer_address = 0x100000000ull;
    options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

    const ConSanResult result = try_patch_consan(bytes, options);

    ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
    ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
    ASSERT_EQ(result.resource_plans.size(), 1u);
    EXPECT_EQ(result.resource_plans.front().source, ConSanRegisterAllocationSource::SpillRequired);
    EXPECT_EQ(result.resource_plans.front().scratch_vgpr, 0u);
    EXPECT_EQ(result.resource_plans.front().scratch_vgpr_count, 16u);
    const auto patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &item) {
      return item.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
    });
    ASSERT_NE(patch, result.patches.end());
    ASSERT_EQ(patch->scratch_vgpr, 0u);
    ASSERT_EQ(patch->spilled_vgpr_count, 16u);
    ASSERT_TRUE(patch->persistent_private_state_end);
    EXPECT_EQ(patch->workgroup_shadow_size, 0u);

    AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(patched.is_valid());
    ASSERT_EQ(patched.kernels().size(), 1u);
    KD patched_descriptor{};
    std::memcpy(&patched_descriptor,
                result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
                sizeof(patched_descriptor));
    EXPECT_EQ(AMDHSA_BITS_GET(patched_descriptor.compute_pgm_rsrc3,
                              kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET),
              3u)
        << "live accumulator storage must never be relocated for instrumentation scratch";
    const std::vector<uint32_t> cave_words =
        text_words_at_offset(patched, patch->trampoline_offset, patch->trampoline_size);
    const auto spill_base =
        normalize_address_free_scratch_private_size(arch, *patch->persistent_private_state_end);
    ASSERT_TRUE(spill_base);
    const uint32_t address_slot = *spill_base + 6u * sizeof(uint32_t);
    auto reload_address =
        arch == ROCJITSU_CODE_ARCH_CDNA3
            ? build_cdna3_address_free_scratch_load_b32(/*vdst=*/15u, address_slot, arch)
            : build_cdna4_address_free_scratch_load_b32(/*vdst=*/15u, address_slot, arch);
    const auto wait = ib::build_s_wait_private_load0(arch);
    auto version_cas =
        arch == ROCJITSU_CODE_ARCH_CDNA3
            ? build_cdna3_flat_atomic_cmpswap_b32(
                  /*vaddr=*/0u, /*vsrc=*/14u, /*vdst=*/14u, /*return_old_value=*/true,
                  /*scope=*/2u, arch)
            : build_cdna4_flat_atomic_cmpswap_b32(
                  /*vaddr=*/0u, /*vsrc=*/14u, /*vdst=*/14u, /*return_old_value=*/true,
                  /*scope=*/2u, arch);
    ASSERT_TRUE(reload_address && wait && version_cas);
    std::vector<uint32_t> reload_sequence(reload_address->begin(), reload_address->end());
    reload_sequence.push_back(*wait);
    const auto first_reload = std::ranges::search(cave_words, reload_sequence).begin();
    const auto first_cas = std::ranges::search(cave_words, *version_cas).begin();
    ASSERT_NE(first_reload, cave_words.end());
    ASSERT_NE(first_cas, cave_words.end());
    EXPECT_LT(first_reload, first_cas);
    const auto second_reload =
        std::search(first_reload + static_cast<ptrdiff_t>(reload_sequence.size()), cave_words.end(),
                    reload_sequence.begin(), reload_sequence.end());
    ASSERT_NE(second_reload, cave_words.end());
    EXPECT_LT(first_cas, second_reload);
    EXPECT_GE(count_subsequence(cave_words, reload_sequence), 2u)
        << "external publication must recover the guest address before indexing and diagnostics";
    EXPECT_TRUE(result.final_validation_passed);
  }
}

TEST(ConSanMoi, Cdna4InlineShadowRecordsEveryRejectedFallbackAttempt) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  constexpr auto load = cdna4::build_ds(cdna4::kDsReadB32Ds, {.addr = 6u, .vdst = 6u});
  std::vector<uint32_t> text_words(1200u, build_s_nop(0, kArch));
  std::copy(load.begin(), load.end(), text_words.begin());
  text_words.back() = build_s_endpgm(kArch);
  std::vector<uint8_t> bytes = make_cdna4_lds_code_object(
      text_words, "rejected_spill_backed_window", /*vgpr_granulated=*/2u);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 3u);
  });
  append_kernel_metadata_note(bytes, "rejected_spill_backed_window",
                              /*uses_dynamic_stack=*/false, /*sgpr_count=*/0u,
                              /*private_segment_fixed_size=*/std::nullopt,
                              /*required_workgroup_size=*/std::nullopt,
                              /*has_dynamic_lds=*/true);

  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  // The only 16-register ordinary window contains this persistent tuple.
  // The alternative is therefore considered and rejected rather than silently
  // disappearing behind the retained 17-register failure.
  options.moi_owner_vgpr = 7u;
  options.moi_epoch_vgpr = 8u;
  options.moi_workgroup_key_vgpr = 9u;
  options.moi_exec_save_sgpr = 40u;
  options.moi_track_atomics = false;
  options.moi_track_barriers = false;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  const ConSanCandidateResourcePlan &plan = result.resource_plans.front();
  EXPECT_EQ(plan.source, ConSanRegisterAllocationSource::Unsupported);
  EXPECT_EQ(plan.reason, ConSanRegisterPlanReason::NoLegalWindow);
  ASSERT_EQ(plan.alternatives.size(), 3u);
  EXPECT_EQ(plan.alternatives[0].kind, ConSanResourcePlanAlternativeKind::GuestOperandOverlapSpill);
  EXPECT_EQ(plan.alternatives[0].scratch_vgpr_count, 17u);
  EXPECT_EQ(plan.alternatives[1].kind,
            ConSanResourcePlanAlternativeKind::SpillBackedOperandRecovery);
  EXPECT_EQ(plan.alternatives[1].scratch_vgpr_count, 16u);
  EXPECT_EQ(plan.alternatives[2].kind, ConSanResourcePlanAlternativeKind::GuestOperandOverlapSpill);
  EXPECT_EQ(plan.alternatives[2].scratch_vgpr_count, 16u);
  for (const ConSanResourcePlanAlternative &alternative : plan.alternatives) {
    EXPECT_EQ(alternative.source, ConSanRegisterAllocationSource::Unsupported);
    EXPECT_EQ(alternative.reason, ConSanRegisterPlanReason::NoLegalWindow);
    EXPECT_EQ(consan_resource_plan_alternative_outcome(plan, alternative),
              ConSanResourcePlanAlternativeOutcome::Rejected);
  }
  EXPECT_EQ(result.resource_plan_summary.alternative_attempts, 3u);
  EXPECT_EQ(result.resource_plan_summary.alternative_selected, 0u);
  EXPECT_EQ(result.resource_plan_summary.alternative_rejected, 3u);
  EXPECT_EQ(result.resource_plan_summary.alternative_superseded, 0u);
  EXPECT_EQ(result.resource_plan_summary.alternative_contributed, 0u);
  EXPECT_EQ(result.resource_plan_summary.alternative_vetoed, 0u);
  EXPECT_EQ(result.resource_plan_summary.alternative_attempts,
            result.resource_plan_summary.alternative_selected +
                result.resource_plan_summary.alternative_rejected +
                result.resource_plan_summary.alternative_superseded +
                result.resource_plan_summary.alternative_contributed +
                result.resource_plan_summary.alternative_vetoed);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiExactShadowStore,
                               &ConSanPatchInfo::kind),
            0u);
}

TEST(ConSanMoi, Cdna4InlineShadowReloadsOverlappingDynamicStackAddress) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  const auto guest = build_cdna4_ds_store_b32(
      /*vaddr=*/0u, /*vdata=*/16u, /*byte_offset=*/0u, kArch);
  ASSERT_TRUE(guest);
  std::vector<uint32_t> text_words(1200u, build_s_nop(0, kArch));
  std::copy(guest->begin(), guest->end(), text_words.begin() + 1u);
  text_words.back() = build_s_endpgm(kArch);
  std::vector<uint8_t> bytes = make_cdna4_lds_code_object(
      text_words, "overlapping_dynamic_stack_address", /*vgpr_granulated=*/4u,
      /*uses_dynamic_stack=*/true, /*workgroup_id_dimension_mask=*/0u,
      /*group_segment_fixed_size=*/4u);
  append_kernel_metadata_note(bytes, "overlapping_dynamic_stack_address",
                              /*uses_dynamic_stack=*/true, /*sgpr_count=*/0u,
                              /*private_segment_fixed_size=*/0u);

  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.force_vgpr_spill = true;
  // These persistent values split the 40-register allocation so no disjoint
  // 16-register spill window exists. Operand-overlap fallback deterministically
  // selects v0:v15 while leaving persistent state outside it.
  options.moi_owner_vgpr = 24u;
  options.moi_epoch_vgpr = 25u;
  options.moi_workgroup_key_vgpr = 26u;
  options.moi_track_atomics = false;
  options.moi_track_barriers = false;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  const ConSanCandidateResourcePlan &plan = result.resource_plans.front();
  EXPECT_EQ(plan.source, ConSanRegisterAllocationSource::SpillRequired);
  EXPECT_EQ(plan.scratch_vgpr, 0u);
  EXPECT_EQ(plan.scratch_vgpr_count, 16u);
  ASSERT_EQ(plan.alternatives.size(), 1u);
  EXPECT_EQ(plan.alternatives.front().kind,
            ConSanResourcePlanAlternativeKind::GuestOperandOverlapSpill);
  EXPECT_EQ(consan_resource_plan_alternative_outcome(plan, plan.alternatives.front()),
            ConSanResourcePlanAlternativeOutcome::Selected);
  EXPECT_EQ(result.resource_plan_summary.alternative_attempts, 1u);
  EXPECT_EQ(result.resource_plan_summary.alternative_selected, 1u);
  const auto patch = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiExactShadowStore, &ConSanPatchInfo::kind);
  ASSERT_NE(patch, result.patches.end());
  ASSERT_EQ(patch->scratch_vgpr, 0u);
  ASSERT_EQ(patch->spilled_vgpr_count, 16u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> cave_words =
      text_words_at_offset(patched, patch->trampoline_offset, patch->trampoline_size);
  const auto reload_address = build_cdna4_scratch_load_b32_saddr(
      /*vdst=*/15u, /*saddr=*/33u, /*byte_offset=*/0u, kArch);
  const auto wait = ib::build_s_wait_private_load0(kArch);
  ASSERT_TRUE(reload_address && wait);
  std::vector<uint32_t> reload_sequence(reload_address->begin(), reload_address->end());
  reload_sequence.push_back(*wait);
  EXPECT_EQ(count_subsequence(cave_words, reload_sequence), 4u)
      << "both transaction paths for both possible cells must recover the spilled address";
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, CdnaInlineShadowAtomicTrackingFitsSpillBackedTransactionWindow) {
  for (const rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    SCOPED_TRACE(arch);
    std::vector<uint32_t> guest;
    if (arch == ROCJITSU_CODE_ARCH_CDNA3) {
      const auto load = build_cdna3_ds_load_b32(
          /*vdst=*/0u, /*vaddr=*/0u, /*byte_offset=*/0u, arch);
      ASSERT_TRUE(load);
      guest.assign(load->begin(), load->end());
    } else {
      constexpr auto load = cdna4::build_ds(cdna4::kDsReadB32Ds, {.addr = 0u, .vdst = 0u});
      guest.assign(load.begin(), load.end());
    }
    std::vector<uint32_t> text_words(1200u, build_s_nop(0, arch));
    std::copy(guest.begin(), guest.end(), text_words.begin());
    size_t cursor = guest.size();
    if (arch == ROCJITSU_CODE_ARCH_CDNA3) {
      const auto release = cdna3::build_mubuf(cdna3::kBufferWbl2Mubuf, {.sc1 = 1});
      const auto wait = build_cdna3_s_wait_vmcnt_lgkmcnt0(arch);
      const auto atomic = build_cdna3_flat_atomic_add_u32(
          /*vaddr=*/2u, /*vsrc=*/4u, /*vdst=*/5u, /*return_old_value=*/true,
          /*scope=*/2u, arch);
      ASSERT_TRUE(wait && atomic);
      std::copy(release.begin(), release.end(), text_words.begin() + cursor);
      cursor += release.size();
      text_words[cursor++] = *wait;
      std::copy(atomic->begin(), atomic->end(), text_words.begin() + cursor);
    } else {
      const auto release = cdna4::build_mubuf(cdna4::kBufferWbl2Mubuf, {.sc1 = 1});
      const auto wait = build_cdna4_s_wait_flat0(arch);
      const auto atomic = build_cdna4_flat_atomic_add_u32(
          /*vaddr=*/2u, /*vsrc=*/4u, /*vdst=*/5u, /*return_old_value=*/true,
          /*scope=*/2u, arch);
      ASSERT_TRUE(wait && atomic);
      std::copy(release.begin(), release.end(), text_words.begin() + cursor);
      cursor += release.size();
      text_words[cursor++] = *wait;
      std::copy(atomic->begin(), atomic->end(), text_words.begin() + cursor);
    }
    text_words.back() = build_s_endpgm(arch);
    std::vector<uint8_t> bytes =
        arch == ROCJITSU_CODE_ARCH_CDNA3
            ? make_cdna3_lds_code_object(text_words, "atomic_tracking_spill_pressure",
                                         /*vgpr_granulated=*/3u)
            : make_cdna4_lds_code_object(text_words, "atomic_tracking_spill_pressure",
                                         /*vgpr_granulated=*/3u);
    mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
      AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 7u);
    });

    ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
    options.force_vgpr_spill = true;
    // Both fallback shapes fit through guest overlap, while no disjoint
    // window exists. The compact 24-register recovery supersedes the initial
    // 25-register attempt and its nested overlap contributes to that result.
    options.moi_owner_vgpr = 25u;
    options.moi_epoch_vgpr = 26u;
    options.moi_workgroup_key_vgpr = 27u;
    options.moi_exec_save_sgpr = 40u;
    options.moi_track_atomics = true;
    options.moi_track_barriers = false;
    options.moi_report_buffer_address = 0x100000000ull;
    options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

    const ConSanResult result = try_patch_consan(bytes, options);

    ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
    ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
    const auto access_plan =
        std::ranges::find_if(result.resource_plans, [](const ConSanCandidateResourcePlan &plan) {
          return plan.site_kind == ConSanResourceSiteKind::Access && plan.text_offset == 0u;
        });
    ASSERT_NE(access_plan, result.resource_plans.end());
    const ConSanCandidateResourcePlan &plan = *access_plan;
    EXPECT_EQ(plan.source, ConSanRegisterAllocationSource::SpillRequired);
    EXPECT_EQ(plan.scratch_vgpr, 0u);
    EXPECT_EQ(plan.scratch_vgpr_count, 24u);
    ASSERT_EQ(plan.alternatives.size(), 3u);
    EXPECT_EQ(plan.alternatives[0].kind,
              ConSanResourcePlanAlternativeKind::GuestOperandOverlapSpill);
    EXPECT_EQ(plan.alternatives[0].scratch_vgpr_count, 25u);
    EXPECT_EQ(consan_resource_plan_alternative_outcome(plan, plan.alternatives[0]),
              ConSanResourcePlanAlternativeOutcome::Superseded);
    EXPECT_EQ(plan.alternatives[1].kind,
              ConSanResourcePlanAlternativeKind::SpillBackedOperandRecovery);
    EXPECT_EQ(plan.alternatives[1].scratch_vgpr_count, 24u);
    EXPECT_EQ(consan_resource_plan_alternative_outcome(plan, plan.alternatives[1]),
              ConSanResourcePlanAlternativeOutcome::Selected);
    EXPECT_EQ(plan.alternatives[2].kind,
              ConSanResourcePlanAlternativeKind::GuestOperandOverlapSpill);
    EXPECT_EQ(plan.alternatives[2].scratch_vgpr_count, 24u);
    EXPECT_EQ(plan.alternatives[2].source, ConSanRegisterAllocationSource::SpillRequired);
    EXPECT_EQ(plan.alternatives[2].reason, plan.reason);
    EXPECT_EQ(consan_resource_plan_alternative_outcome(plan, plan.alternatives[2]),
              ConSanResourcePlanAlternativeOutcome::Contributed);
    EXPECT_EQ(result.resource_plan_summary.alternative_attempts, 3u);
    EXPECT_EQ(result.resource_plan_summary.alternative_selected, 1u);
    EXPECT_EQ(result.resource_plan_summary.alternative_rejected, 0u);
    EXPECT_EQ(result.resource_plan_summary.alternative_superseded, 1u);
    EXPECT_EQ(result.resource_plan_summary.alternative_contributed, 1u);
    EXPECT_EQ(result.resource_plan_summary.alternative_vetoed, 0u);
    const auto patch = std::ranges::find(
        result.patches, ConSanPatchKind::TrampolineMoiExactShadowStore, &ConSanPatchInfo::kind);
    ASSERT_NE(patch, result.patches.end());
    EXPECT_EQ(patch->spilled_vgpr_count, 24u);
    EXPECT_TRUE(result.final_validation_passed);
  }
}

TEST(ConSanMoi, CdnaInlineShadowWideAccessReloadsAddressOutsideCellLoopState) {
  for (const rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    SCOPED_TRACE(arch);
    std::vector<uint32_t> guest;
    if (arch == ROCJITSU_CODE_ARCH_CDNA3) {
      const auto store = build_cdna3_ds_store_b128(
          /*vaddr=*/0u, /*vdata=*/16u, /*byte_offset=*/0u, arch);
      ASSERT_TRUE(store);
      guest.assign(store->begin(), store->end());
    } else {
      constexpr auto store = cdna4::build_ds(cdna4::kDsWriteB128Ds, {.addr = 0u, .data0 = 16u});
      guest.assign(store.begin(), store.end());
    }
    std::vector<uint32_t> text_words(1200u, build_s_nop(0, arch));
    std::copy(guest.begin(), guest.end(), text_words.begin());
    text_words.back() = build_s_endpgm(arch);
    std::vector<uint8_t> bytes =
        arch == ROCJITSU_CODE_ARCH_CDNA3
            ? make_cdna3_lds_code_object(text_words, "wide_access_spill_pressure",
                                         /*vgpr_granulated=*/3u)
            : make_cdna4_lds_code_object(text_words, "wide_access_spill_pressure",
                                         /*vgpr_granulated=*/3u);
    mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
      AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 7u);
    });

    ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
    options.force_vgpr_spill = true;
    options.moi_owner_vgpr = 24u;
    options.moi_epoch_vgpr = 25u;
    options.moi_workgroup_key_vgpr = 26u;
    options.moi_exec_save_sgpr = 40u;
    options.moi_track_atomics = false;
    options.moi_track_barriers = false;
    options.moi_report_buffer_address = 0x100000000ull;
    options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

    const ConSanResult result = try_patch_consan(bytes, options);

    ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
    ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
    ASSERT_EQ(result.moi_candidates.size(), 1u);
    EXPECT_EQ(result.moi_candidates.front().width_bits, 128u);
    ASSERT_EQ(result.resource_plans.size(), 1u);
    EXPECT_EQ(result.resource_plans.front().source, ConSanRegisterAllocationSource::SpillRequired);
    EXPECT_EQ(result.resource_plans.front().scratch_vgpr, 0u);
    EXPECT_EQ(result.resource_plans.front().scratch_vgpr_count, 18u);
    const auto patch = std::ranges::find(
        result.patches, ConSanPatchKind::TrampolineMoiExactShadowStore, &ConSanPatchInfo::kind);
    ASSERT_NE(patch, result.patches.end());
    EXPECT_EQ(patch->spilled_vgpr_count, 18u);

    AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(patched.is_valid());
    const std::vector<uint32_t> cave_words =
        text_words_at_offset(patched, patch->trampoline_offset, patch->trampoline_size);
    constexpr uint32_t kSpillBase = 0u;
    constexpr uint16_t kPhaseSharedAddressVgpr = 15u;
    auto reload_address =
        arch == ROCJITSU_CODE_ARCH_CDNA3
            ? build_cdna3_address_free_scratch_load_b32(kPhaseSharedAddressVgpr, kSpillBase, arch)
            : build_cdna4_address_free_scratch_load_b32(kPhaseSharedAddressVgpr, kSpillBase, arch);
    const auto wait = ib::build_s_wait_private_load0(arch);
    ASSERT_TRUE(reload_address && wait);
    std::vector<uint32_t> reload_sequence(reload_address->begin(), reload_address->end());
    reload_sequence.push_back(*wait);
    EXPECT_EQ(count_subsequence(cave_words, reload_sequence), 2u);
    const uint16_t loop_counter = consan_detail::inline_shadow_loop_counter_vgpr(
        /*scratch_vgpr=*/0u, /*has_exec_save=*/true, /*track_atomics=*/false);
    EXPECT_LT(kPhaseSharedAddressVgpr, loop_counter);
    EXPECT_TRUE(result.final_validation_passed);
  }
}

TEST(ConSanMoi, Cdna4InlineShadowReloadsBothSpilledMaybeGroupAddressHalves) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  std::vector<uint32_t> text_words;
  text_words.push_back(0xBE8001EBu); // s_mov_b64 s[0:1], src_shared_base
  text_words.push_back(build_v_mov_b32_e32(/*vdst=*/0u, vector_source_vgpr(30u), kArch));
  text_words.push_back(build_v_mov_b32_e32(/*vdst=*/1u, /*s1=*/1u, kArch));
  constexpr auto load = cdna4::build_flat(cdna4::kFlatLoadDwordFlat, {.addr = 0u, .vdst = 16u});
  text_words.insert(text_words.end(), load.begin(), load.end());
  text_words.resize(1200u, build_s_nop(0, kArch));
  text_words.back() = build_s_endpgm(kArch);
  std::vector<uint8_t> bytes =
      make_cdna4_lds_code_object(text_words, "maybe_group_spill_overlap", /*vgpr_granulated=*/4u);

  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.force_vgpr_spill = true;
  options.moi_owner_vgpr = 24u;
  options.moi_epoch_vgpr = 25u;
  options.moi_workgroup_key_vgpr = 26u;
  options.moi_exec_save_sgpr = 40u;
  options.moi_track_atomics = false;
  options.moi_track_barriers = false;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.moi_candidates.size(), 1u);
  EXPECT_EQ(result.moi_candidates.front().source, ConSanMoiCandidateSource::FlatMaybeGroup);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().source, ConSanRegisterAllocationSource::SpillRequired);
  EXPECT_EQ(result.resource_plans.front().scratch_vgpr, 0u);
  const auto patch = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiExactShadowStore, &ConSanPatchInfo::kind);
  ASSERT_NE(patch, result.patches.end());
  ASSERT_EQ(patch->scratch_vgpr, 0u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> cave_words =
      text_words_at_offset(patched, patch->trampoline_offset, patch->trampoline_size);
  constexpr uint32_t kSpillBase = 0u;
  const auto reload_low =
      build_cdna4_address_free_scratch_load_b32(/*vdst=*/15u, kSpillBase, kArch);
  const auto reload_high = build_cdna4_address_free_scratch_load_b32(
      /*vdst=*/1u, kSpillBase + sizeof(uint32_t), kArch);
  const auto wait = ib::build_s_wait_private_load0(kArch);
  ASSERT_TRUE(reload_low && reload_high && wait);
  std::vector<uint32_t> low_sequence(reload_low->begin(), reload_low->end());
  low_sequence.push_back(*wait);
  std::vector<uint32_t> high_sequence(reload_high->begin(), reload_high->end());
  high_sequence.push_back(*wait);
  EXPECT_EQ(count_subsequence(cave_words, low_sequence), 4u);
  EXPECT_EQ(count_subsequence(cave_words, high_sequence), 1u);
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, Cdna4ExternalInlineShadowRetainsPrivateWorkgroupKey) {
  std::vector<uint32_t> text_words(1200, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  text_words[0] = 0xd81a0004u;
  text_words[1] = 0x00000302u; // ds_write_b32 v2, v3 offset:4
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);
  const std::vector<uint8_t> bytes =
      make_cdna4_lds_code_object(text_words, "external_private_workgroup_key");
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.force_private_epoch = true;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &item) {
    return item.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  });
  ASSERT_NE(access, result.patches.end());
  ASSERT_EQ(access->persistent_epoch_private_offset, 0u);
  EXPECT_FALSE(access->persistent_owner_private_offset);
  ASSERT_EQ(access->persistent_workgroup_key_private_offset, 4u);
  EXPECT_EQ(access->persistent_private_state_end, 8u);
  EXPECT_EQ(access->workgroup_shadow_size, 0u);
}

TEST(ConSanMoi, WorkgroupShadowLayoutUsesOneEightByteSlotPerFourByteLdsCell) {
  const auto qwen = plan_consan_moi_workgroup_shadow(4352u);
  ASSERT_TRUE(qwen);
  EXPECT_EQ(qwen->base, 4352u);
  EXPECT_EQ(qwen->size, 8704u);
  EXPECT_EQ(qwen->required_group_segment_size, 13056u);

  const auto unaligned = plan_consan_moi_workgroup_shadow(5u);
  ASSERT_TRUE(unaligned);
  EXPECT_EQ(unaligned->base, 8u);
  EXPECT_EQ(unaligned->size, 16u);
  EXPECT_EQ(unaligned->required_group_segment_size, 24u);

  EXPECT_FALSE(plan_consan_moi_workgroup_shadow(0u));
  EXPECT_TRUE(plan_consan_moi_workgroup_shadow(21840u));
  EXPECT_FALSE(plan_consan_moi_workgroup_shadow(21848u));
}

TEST(ConSanMoi, Gfx1250WorkgroupShadowUsesConfiguredAperture) {
  constexpr uint32_t kConfiguredLdsBytes =
      consan_moi_max_workgroup_lds_bytes(ROCJITSU_CODE_ARCH_GFX1250);
  constexpr uint32_t kOriginalLdsBytes = kConfiguredLdsBytes / 4u;
  const auto layout =
      plan_consan_moi_compact_workgroup_shadow(kOriginalLdsBytes, kConfiguredLdsBytes);

  ASSERT_TRUE(layout);
  EXPECT_EQ(layout->base, kOriginalLdsBytes);
  EXPECT_LE(layout->required_group_segment_size, kConfiguredLdsBytes);
  EXPECT_TRUE(layout->compact);
  EXPECT_TRUE(layout->lazy_initialization);
  EXPECT_FALSE(plan_consan_moi_compact_workgroup_shadow(kConfiguredLdsBytes, kConfiguredLdsBytes));
}

TEST(ConSanMoi, Gfx1250WorkgroupLdsMatchesSimulatorConfig) {
  const auto loaded =
      config::load_config(std::string(CONFIG_DIR) + "/gfx1250.json", rocjitsu::kEmbeddedSchema);

  ASSERT_TRUE(loaded.device.present);
  EXPECT_EQ(consan_moi_max_workgroup_lds_bytes(ROCJITSU_CODE_ARCH_GFX1250),
            loaded.device.lds_size_kb * 1024u);
}

TEST(ConSanMoi, RuntimeLdsApertureOverridesConfiguredDefault) {
  ConSanOptions options;
  options.moi_max_workgroup_lds_bytes = 96u * 1024u;

  EXPECT_EQ(consan_moi_max_workgroup_lds_bytes(options, ROCJITSU_CODE_ARCH_GFX1250), 96u * 1024u);
  options.moi_max_workgroup_lds_bytes.reset();
  EXPECT_EQ(consan_moi_max_workgroup_lds_bytes(options, ROCJITSU_CODE_ARCH_GFX1250),
            consan_moi_max_workgroup_lds_bytes(ROCJITSU_CODE_ARCH_GFX1250));
}

TEST(ConSanMoi, InlineExactDispatchBankSelectionCoversFitBoundaries) {
  constexpr uint64_t kExactShadowBudget = kConSanMoiAutoReportBufferCeilingBytes / 2u;
  constexpr uint64_t kLargestFittingCellCount =
      kExactShadowBudget / sizeof(ConSanMoiInlineExactShadowSlot);
  constexpr uint32_t kLargestFittingLdsBytes = static_cast<uint32_t>(kLargestFittingCellCount * 4u);

  EXPECT_EQ(consan_moi_inline_exact_dispatch_bank_count_for_lds(4u), 256u);
  EXPECT_EQ(consan_moi_inline_exact_dispatch_bank_count_for_lds(64u * 1024u), 128u);
  EXPECT_EQ(consan_moi_inline_exact_dispatch_bank_count_for_lds(1024u * 1024u), 8u);
  EXPECT_EQ(consan_moi_inline_exact_dispatch_bank_count_for_lds(kLargestFittingLdsBytes), 1u);
  EXPECT_EQ(consan_moi_inline_exact_dispatch_bank_count_for_lds(kLargestFittingLdsBytes + 4u), 0u);
}

TEST(ConSanMoi, InlineExactDispatchBanksTrackConfiguredLdsAperture) {
  constexpr uint64_t kConfiguredLdsBytes =
      consan_moi_max_workgroup_lds_bytes(ROCJITSU_CODE_ARCH_GFX1250);
  constexpr uint64_t kExactBytesPerBank =
      ((kConfiguredLdsBytes + consan_moi_exact_shadow::granule_bytes - 1u) /
       consan_moi_exact_shadow::granule_bytes) *
      sizeof(ConSanMoiInlineExactShadowSlot);
  constexpr uint64_t kExactShadowBudget = kConSanMoiAutoReportBufferCeilingBytes / 2u;
  constexpr uint32_t kConfiguredDispatchBankCount =
      consan_moi_inline_exact_dispatch_bank_count_for_lds(kConfiguredLdsBytes);
  const ConSanMoiAutoReportInventory inventory{
      .engine = ConSanMoiEngine::InlineShadow,
      .inline_lds_bytes = kConfiguredLdsBytes,
  };
  const ConSanMoiAutoReportPlan plan = plan_consan_moi_auto_report(inventory);

  ASSERT_TRUE(plan.complete());
  EXPECT_EQ(plan.layout.inline_exact_dispatch_bank_count, kConfiguredDispatchBankCount);
  EXPECT_LE(kExactBytesPerBank * kConfiguredDispatchBankCount, kExactShadowBudget);
  if (kConfiguredDispatchBankCount < kConSanMoiInlineMaximumDispatchBankCount) {
    EXPECT_GT(kExactBytesPerBank * (kConfiguredDispatchBankCount * 2u), kExactShadowBudget);
  }
}

TEST(ConSanMoi, LazyWorkgroupShadowAddsTwoBitValidityState) {
  const auto qwen = plan_consan_moi_lazy_workgroup_shadow(21120u);
  ASSERT_TRUE(qwen);
  EXPECT_EQ(qwen->base, 21120u);
  EXPECT_EQ(qwen->size, 42240u);
  EXPECT_EQ(qwen->validity_base, 63360u);
  EXPECT_EQ(qwen->validity_size, 1328u);
  EXPECT_EQ(qwen->required_group_segment_size, 64688u);
  EXPECT_TRUE(qwen->lazy_initialization);

  EXPECT_FALSE(plan_consan_moi_lazy_workgroup_shadow(0u));
  EXPECT_FALSE(plan_consan_moi_lazy_workgroup_shadow(21840u));
}

TEST(ConSanMoi, GenerationTaggedWorkgroupShadowNeedsNoInitializationState) {
  const auto layout = plan_consan_moi_generation_tagged_workgroup_shadow(
      21120u, consan_moi_max_workgroup_lds_bytes(ROCJITSU_CODE_ARCH_GFX1250));
  ASSERT_TRUE(layout);
  EXPECT_EQ(layout->base, 21120u);
  EXPECT_EQ(layout->size, 42240u);
  EXPECT_EQ(layout->validity_base, 0u);
  EXPECT_EQ(layout->validity_size, 0u);
  EXPECT_EQ(layout->required_group_segment_size, 63360u);
  EXPECT_TRUE(layout->lazy_initialization);
  EXPECT_FALSE(layout->compact);
  EXPECT_TRUE(consan_moi_generation_tagged_workgroup_shadow(*layout));
}

TEST(ConSanMoi, Gfx1250GenerationTaggedShadowUsesCompactCellsOnlyAsCapacityFallback) {
  EXPECT_TRUE(consan_moi_prefer_generation_tagged_workgroup_shadow(21120u));
  EXPECT_TRUE(consan_moi_prefer_generation_tagged_workgroup_shadow(22u * 1024u));
  EXPECT_TRUE(consan_moi_prefer_generation_tagged_workgroup_shadow(24576u));
  EXPECT_TRUE(consan_moi_prefer_generation_tagged_workgroup_shadow(49152u));
  EXPECT_TRUE(consan_moi_prefer_generation_tagged_workgroup_shadow(64u * 1024u));
  EXPECT_FALSE(consan_moi_prefer_generation_tagged_workgroup_shadow(96u * 1024u));
  EXPECT_FALSE(consan_moi_prefer_generation_tagged_workgroup_shadow(128u * 1024u));
  EXPECT_TRUE(consan_moi_prefer_generation_tagged_workgroup_shadow(
      24576u, /*compact_cells_available=*/false));
  EXPECT_TRUE(consan_moi_prefer_generation_tagged_workgroup_shadow(
      49152u, /*compact_cells_available=*/false));
}

TEST(ConSanMoi, CompactWorkgroupShadowUsesOneWordPerGuestCell) {
  const auto layout = plan_consan_moi_compact_workgroup_shadow(21120u);
  ASSERT_TRUE(layout);
  EXPECT_EQ(layout->base, 21120u);
  EXPECT_EQ(layout->size, 21120u);
  EXPECT_EQ(layout->validity_base, 42240u);
  EXPECT_EQ(layout->validity_size, 1328u);
  EXPECT_EQ(layout->required_group_segment_size, 43568u);
  EXPECT_TRUE(layout->lazy_initialization);
  EXPECT_TRUE(layout->compact);
}

TEST(ConSanMoi, Rdna4AccessOnlyInlineShadowUsesInitializedWorkgroupLocalLdsMirror) {
  const std::array<uint32_t, 3> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32
      0xBFB00000u, // s_endpgm
  };
  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "workgroup_shadow", kRdna4Wave64AllVgprsGranulated, false, false, 0, 4352u);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_VGPR_WORKITEM_ID,
                    /*x_y_and_z=*/2u);
  });
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_inline_workgroup_shadow = true;
  options.moi_track_barriers = true;
  options.moi_track_atomics = true;
  options.moi_init_owner_epoch = true;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 24;
  options.moi_epoch_vgpr = 25;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  });
  const auto prologue = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue;
  });
  ASSERT_NE(access, result.patches.end());
  ASSERT_NE(prologue, result.patches.end());
  for (const ConSanPatchInfo *patch : {&*access, &*prologue}) {
    EXPECT_EQ(patch->workgroup_shadow_base, 4352u);
    EXPECT_EQ(patch->workgroup_shadow_size, 8704u);
    EXPECT_EQ(patch->workgroup_shadow_validity_base, 13056u);
    EXPECT_EQ(patch->workgroup_shadow_validity_size, 272u);
    EXPECT_EQ(patch->required_group_segment_size, 13328u);
    EXPECT_TRUE(patch->workgroup_shadow_lazy_initialization);
    EXPECT_FALSE(patch->workgroup_shadow_compact);
  }

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.kernels().size(), 1u);
  const uint64_t descriptor_offset = patched.kernels().front().descriptor_file_offset;
  ASSERT_LE(descriptor_offset + sizeof(KD), result.elf_bytes.size());
  KD descriptor{};
  std::memcpy(&descriptor, result.elf_bytes.data() + descriptor_offset, sizeof(descriptor));
  EXPECT_EQ(descriptor.group_segment_fixed_size, 13328u);

  ASSERT_EQ(patched.text_sections().size(), 1u);
  const auto *text = patched.text_sections().front();
  ASSERT_LE(access->trampoline_offset + access->trampoline_size, text->size());
  ASSERT_TRUE(access->scratch_vgpr);
  const uint16_t access_scratch = *access->scratch_vgpr;
  std::vector<uint32_t> access_words(access->trampoline_size / sizeof(uint32_t));
  std::memcpy(access_words.data(), text->data() + access->trampoline_offset,
              access->trampoline_size);
  const auto local_exchange = build_ds_storexchg_rtn_b64(
      static_cast<uint16_t>(access_scratch + 5u), access_scratch,
      static_cast<uint16_t>(access_scratch + 2u), /*byte_offset=*/0, ROCJITSU_CODE_ARCH_RDNA4);
  const auto diagnostic_count_claim = build_flat_atomic_cmpswap_b32_vaddr_vsrc_vdst(
      access_scratch, static_cast<uint16_t>(access_scratch + 14u),
      static_cast<uint16_t>(access_scratch + 14u), /*return_old_value=*/true,
      /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(local_exchange);
  ASSERT_TRUE(diagnostic_count_claim);
  EXPECT_EQ(count_subsequence(access_words, *local_exchange), 2u);
  EXPECT_EQ(count_subsequence(access_words, *diagnostic_count_claim), 2u)
      << "each possible local cell has one cold diagnostic claim";
  const auto second_cell_bytes =
      ib::build_v_mov_b32_literal(static_cast<uint16_t>(access_scratch + 4u),
                                  consan_moi_exact_shadow::granule_bytes, ROCJITSU_CODE_ARCH_RDNA4);
  const auto clamp_second_cell =
      ib::build_v_min_u32(static_cast<uint16_t>(access_scratch + 4u),
                          vector_source_vgpr(static_cast<uint16_t>(access_scratch + 8u)),
                          static_cast<uint16_t>(access_scratch + 4u), ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(second_cell_bytes);
  ASSERT_TRUE(clamp_second_cell);
  std::vector<uint32_t> second_cell_provenance = *second_cell_bytes;
  second_cell_provenance.push_back(*clamp_second_cell);
  EXPECT_EQ(count_subsequence(access_words, second_cell_provenance), 1u)
      << "the local mirror must derive an empty or partial second-cell mask";
  const auto shadow_capacity = build_v_mov_b32_e64_literal(
      access_scratch, /*4352 guest bytes / 4-byte cells=*/1088u, ROCJITSU_CODE_ARCH_RDNA4);
  const auto bounds_check = build_v_cmp_gt_u32_e32_vcc(vector_source_vgpr(access_scratch),
                                                       static_cast<uint16_t>(access_scratch + 4u),
                                                       ROCJITSU_CODE_ARCH_RDNA4);
  const auto undercoverage_address = build_v_mov_b32_e64_literal(
      access_scratch,
      static_cast<uint32_t>(options.moi_report_buffer_address.value() +
                            offsetof(ConSanMoiReportHeader, inline_undercoverage_count)),
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto event_counter_address = build_v_mov_b32_e64_literal(
      access_scratch,
      static_cast<uint32_t>(options.moi_report_buffer_address.value() +
                            offsetof(ConSanMoiReportHeader, event_counter)),
      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(shadow_capacity);
  ASSERT_TRUE(bounds_check);
  ASSERT_TRUE(undercoverage_address);
  ASSERT_TRUE(event_counter_address);
  EXPECT_TRUE(contains_subsequence(access_words, *shadow_capacity));
  EXPECT_NE(std::find(access_words.begin(), access_words.end(), *bounds_check), access_words.end());
  EXPECT_TRUE(contains_subsequence(access_words, *undercoverage_address));
  EXPECT_TRUE(contains_subsequence(access_words, *event_counter_address));

  // Empty conflict masks must skip the cold diagnostic writer instead of
  // walking it under EXEC=0. Those diagnostic branches converge on the common
  // application-state restoration; the local undercoverage branch has its
  // own earlier restoration target.
  const auto execz = build_s_cbranch_execz(0, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(execz);
  ASSERT_TRUE(result.resolved_moi_exec_save_sgpr);
  const auto save_original_exec =
      build_s_mov_b64(static_cast<uint16_t>(*result.resolved_moi_exec_save_sgpr +
                                            kConSanMoiInlineOriginalExecSaveOffset),
                      /*exec=*/126u, ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_exec =
      build_s_mov_b64(/*sdst=*/126u, *result.resolved_moi_exec_save_sgpr, ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_bounded_exec = build_s_mov_b64(
      /*sdst=*/126u, static_cast<uint16_t>(*result.resolved_moi_exec_save_sgpr + 16u),
      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(save_original_exec);
  ASSERT_TRUE(restore_exec);
  ASSERT_TRUE(restore_bounded_exec);
  EXPECT_NE(std::ranges::find(access_words, *save_original_exec), access_words.end());
  size_t diagnostic_cold_path_branch_count = 0;
  size_t undercoverage_branch_count = 0;
  std::vector<size_t> diagnostic_restore_indices;
  for (size_t i = 0; i < access_words.size(); ++i) {
    if ((access_words[i] & 0xffff0000u) != (*execz & 0xffff0000u))
      continue;
    const int16_t delta = static_cast<int16_t>(access_words[i]);
    if (delta <= 0)
      continue;
    const size_t target = i + 1u + static_cast<size_t>(delta);
    ASSERT_LT(target, access_words.size());
    if (access_words[target] == *restore_bounded_exec) {
      ++undercoverage_branch_count;
      continue;
    }
    EXPECT_EQ(access_words[target], *restore_exec);
    if (std::ranges::find(diagnostic_restore_indices, target) == diagnostic_restore_indices.end())
      diagnostic_restore_indices.push_back(target);
    ++diagnostic_cold_path_branch_count;
  }
  EXPECT_EQ(diagnostic_restore_indices.size(), 2u)
      << "each unrolled cell must converge on its own application-state restoration";
  EXPECT_GE(diagnostic_cold_path_branch_count, 8u);
  EXPECT_GE(undercoverage_branch_count, 2u);

  ASSERT_LE(prologue->trampoline_offset + prologue->trampoline_size, text->size());
  std::vector<uint32_t> prologue_words(prologue->trampoline_size / sizeof(uint32_t));
  std::memcpy(prologue_words.data(), text->data() + prologue->trampoline_offset,
              prologue->trampoline_size);
  const auto extract_x = ib::build_v_and_b32_literal(
      /*vdst=*/26u, 0x3ffu, /*packed_workitem_id=*/0u, ROCJITSU_CODE_ARCH_RDNA4);
  const auto extract_y_shift = ib::build_v_lshrrev_b32(
      /*vdst=*/24u, scalar_positive_inline_u32(10u), /*packed_workitem_id=*/0u,
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto extract_y_mask =
      ib::build_v_and_b32_literal(/*vdst=*/24u, 0x3ffu, /*vsrc1=*/24u, ROCJITSU_CODE_ARCH_RDNA4);
  const auto extract_z_shift = ib::build_v_lshrrev_b32(
      /*vdst=*/24u, scalar_positive_inline_u32(20u), /*packed_workitem_id=*/0u,
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto select_y_zero = ib::build_v_cmp_eq_u32_vcc(
      scalar_positive_inline_u32(0u), /*extracted_y=*/24u, ROCJITSU_CODE_ARCH_RDNA4);
  const auto scale_x = ib::build_v_lshlrev_b32(
      /*vdst=*/26u, scalar_positive_inline_u32(3u), /*extracted_x=*/26u, ROCJITSU_CODE_ARCH_RDNA4);
  const auto store_wide =
      build_ds_store_b64(/*vaddr=*/24, /*vdata=*/25, /*byte_offset=*/0, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(extract_x);
  ASSERT_TRUE(extract_y_shift);
  ASSERT_TRUE(extract_y_mask);
  ASSERT_TRUE(extract_z_shift);
  ASSERT_TRUE(select_y_zero);
  ASSERT_TRUE(scale_x);
  ASSERT_TRUE(store_wide);
  EXPECT_TRUE(contains_subsequence(prologue_words, *extract_x));
  EXPECT_NE(std::ranges::find(prologue_words, *extract_y_shift), prologue_words.end());
  EXPECT_NE(std::ranges::find(prologue_words, *extract_z_shift), prologue_words.end());
  EXPECT_EQ(count_subsequence(prologue_words, *extract_y_mask), 2u);
  EXPECT_EQ(std::ranges::count(prologue_words, *select_y_zero), 2u);
  EXPECT_NE(std::ranges::find(prologue_words, *scale_x), prologue_words.end())
      << "the initializer must scale the extracted x coordinate, not packed v0";
  EXPECT_TRUE(contains_subsequence(prologue_words, *store_wide));
  EXPECT_EQ(std::count(prologue_words.begin(), prologue_words.end(),
                       *build_s_barrier_signal_all(ROCJITSU_CODE_ARCH_RDNA4)),
            1);
  EXPECT_EQ(std::count(prologue_words.begin(), prologue_words.end(),
                       *build_s_barrier_wait_all(ROCJITSU_CODE_ARCH_RDNA4)),
            1);
}

TEST(ConSanMoi, InvalidWorkitemDimensionsFailClosed) {
  std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_VGPR_WORKITEM_ID,
                    /*reserved=*/3u);
  });
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_inline_workgroup_shadow = true;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  EXPECT_FALSE(consan_patch_succeeded(result));
  EXPECT_FALSE(result.modified);
  ASSERT_FALSE(result.errors.empty());
  EXPECT_TRUE(std::ranges::any_of(result.errors, [](const std::string &error) {
    return error.find("invalid workitem-ID dimensions") != std::string::npos;
  }));
}

TEST(ConSanMoi, Rdna4AtomicTrackingUsesInitializedWorkgroupLocalLdsMirror) {
  const auto release = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/2, /*vsrc=*/1, /*vdst=*/0, /*return_old_value=*/false, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto acquire = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/4, /*vsrc=*/1, /*vdst=*/0, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(release && acquire);
  const std::array<uint32_t, 15> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32
      0xEE0B0000u,   0x00000000u,
      0x00000000u, // global_wb
      (*release)[0], (*release)[1], (*release)[2], (*acquire)[0],
      (*acquire)[1], (*acquire)[2], 0xEE0AC000u,   0x00000000u,
      0x00000000u, // global_inv
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "atomic_workgroup_shadow",
                                 kRdna4Wave64AllVgprsGranulated, false, false, 0, 4352u);
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_inline_workgroup_shadow = true;
  options.moi_track_atomics = true;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.max_patches = 16;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  });
  const auto prologue = std::ranges::find(
      result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue, &ConSanPatchInfo::kind);
  ASSERT_NE(access, result.patches.end());
  ASSERT_NE(prologue, result.patches.end());
  EXPECT_EQ(access->workgroup_shadow_base, 4352u);
  EXPECT_EQ(access->workgroup_shadow_size, 8704u);
  EXPECT_EQ(access->workgroup_shadow_validity_base, 13056u);
  EXPECT_EQ(access->workgroup_shadow_validity_size, 272u);
  EXPECT_EQ(access->required_group_segment_size, 13328u);
  EXPECT_TRUE(access->workgroup_shadow_lazy_initialization);
  EXPECT_FALSE(access->workgroup_shadow_compact);
  EXPECT_TRUE(prologue->workgroup_shadow_lazy_initialization);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const auto *text = patched.text_sections().front();
  std::vector<uint32_t> prologue_words(prologue->trampoline_size / sizeof(uint32_t));
  std::memcpy(prologue_words.data(), text->data() + prologue->trampoline_offset,
              prologue->trampoline_size);
  EXPECT_EQ(std::count(prologue_words.begin(), prologue_words.end(),
                       *build_s_barrier_signal_all(ROCJITSU_CODE_ARCH_RDNA4)),
            1);
  EXPECT_EQ(std::count(prologue_words.begin(), prologue_words.end(),
                       *build_s_barrier_wait_all(ROCJITSU_CODE_ARCH_RDNA4)),
            1);

  ASSERT_TRUE(result.resolved_moi_dispatch_id_sgpr);
  ASSERT_TRUE(access->scratch_vgpr);
  const std::vector<uint32_t> access_words =
      text_words_at_offset(patched, access->trampoline_offset, access->trampoline_size);
  const uint16_t generation_vgpr = static_cast<uint16_t>(*access->scratch_vgpr + 5u);
  const auto mix_dispatch_low =
      ib::build_v_xor_b32(generation_vgpr, *result.resolved_moi_dispatch_id_sgpr, generation_vgpr,
                          ROCJITSU_CODE_ARCH_RDNA4);
  const auto mix_dispatch_high = ib::build_v_xor_b32(
      generation_vgpr, static_cast<uint16_t>(*result.resolved_moi_dispatch_id_sgpr + 1u),
      generation_vgpr, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(mix_dispatch_low);
  ASSERT_TRUE(mix_dispatch_high);
  EXPECT_EQ(std::ranges::find(access_words, *mix_dispatch_low), access_words.end());
  EXPECT_EQ(std::ranges::find(access_words, *mix_dispatch_high), access_words.end());
}

TEST(ConSanMoi, Rdna4InlineShadowEagerlyClearsExactMirrorWhenValidityBitmapDoesNotFit) {
  const std::array<uint32_t, 3> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words, "eager_exact_mirror");
  mutate_first_kernel_descriptor(
      bytes, [](KD &descriptor) { descriptor.group_segment_fixed_size = 21840u; });
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_inline_workgroup_shadow = true;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto access = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiExactShadowStore, &ConSanPatchInfo::kind);
  const auto prologue = std::ranges::find(
      result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue, &ConSanPatchInfo::kind);
  ASSERT_NE(access, result.patches.end());
  ASSERT_NE(prologue, result.patches.end());
  for (const ConSanPatchInfo *patch : {&*access, &*prologue}) {
    EXPECT_EQ(patch->workgroup_shadow_base, 21840u);
    EXPECT_EQ(patch->workgroup_shadow_size, 43680u);
    EXPECT_EQ(patch->workgroup_shadow_validity_size, 0u);
    EXPECT_EQ(patch->required_group_segment_size, 65520u);
    EXPECT_FALSE(patch->workgroup_shadow_lazy_initialization);
    EXPECT_FALSE(patch->workgroup_shadow_compact);
  }

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> prologue_words =
      text_words_at_offset(patched, prologue->trampoline_offset, prologue->trampoline_size);
  EXPECT_EQ(
      std::ranges::count(prologue_words, *build_s_barrier_signal_all(ROCJITSU_CODE_ARCH_RDNA4)), 1);
  EXPECT_EQ(std::ranges::count(prologue_words, *build_s_barrier_wait_all(ROCJITSU_CODE_ARCH_RDNA4)),
            1);
}

TEST(ConSanMoi, Rdna4BarrierTrackingUsesInitializedWorkgroupLocalLdsMirror) {
  const auto signal = build_s_barrier_signal_all(ROCJITSU_CODE_ARCH_RDNA4);
  const auto wait = build_s_barrier_wait_all(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(signal && wait);
  const std::array<uint32_t, 5> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32
      *signal,     *wait,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "barrier_workgroup_shadow",
                                 kRdna4Wave64AllVgprsGranulated, false, false, 0, 4352u);
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_inline_workgroup_shadow = true;
  options.moi_track_barriers = true;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.max_patches = 16;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  });
  ASSERT_NE(access, result.patches.end());
  EXPECT_EQ(access->workgroup_shadow_base, 4352u);
  EXPECT_EQ(access->workgroup_shadow_size, 8704u);
  EXPECT_EQ(access->workgroup_shadow_validity_base, 13056u);
  EXPECT_EQ(access->workgroup_shadow_validity_size, 272u);
  EXPECT_EQ(access->required_group_segment_size, 13328u);
  EXPECT_TRUE(access->workgroup_shadow_lazy_initialization);
  EXPECT_FALSE(access->workgroup_shadow_compact);

  const auto prologue = std::ranges::find(
      result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue, &ConSanPatchInfo::kind);
  ASSERT_NE(prologue, result.patches.end());
  EXPECT_TRUE(prologue->workgroup_shadow_lazy_initialization);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const auto *text = patched.text_sections().front();
  std::vector<uint32_t> prologue_words(prologue->trampoline_size / sizeof(uint32_t));
  std::memcpy(prologue_words.data(), text->data() + prologue->trampoline_offset,
              prologue->trampoline_size);
  EXPECT_EQ(std::count(prologue_words.begin(), prologue_words.end(),
                       *build_s_barrier_signal_all(ROCJITSU_CODE_ARCH_RDNA4)),
            1);
  EXPECT_EQ(std::count(prologue_words.begin(), prologue_words.end(),
                       *build_s_barrier_wait_all(ROCJITSU_CODE_ARCH_RDNA4)),
            1);
}

TEST(ConSanMoi, InlineShadowFallsBackToExternalMirrorWhenLocalMirrorDoesNotFit) {
  const std::array<uint32_t, 3> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "external_shadow_fallback",
                                 kRdna4Wave64AllVgprsGranulated, false, false, 0, 21848u);
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_inline_workgroup_shadow = true;
  options.moi_init_owner_epoch = true;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 24;
  options.moi_epoch_vgpr = 25;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  });
  ASSERT_NE(access, result.patches.end());
  EXPECT_EQ(access->workgroup_shadow_base, 0u);
  EXPECT_EQ(access->workgroup_shadow_size, 0u);
  EXPECT_EQ(access->required_group_segment_size, 0u);
  const auto prologue = std::ranges::find(
      result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue, &ConSanPatchInfo::kind);
  ASSERT_NE(prologue, result.patches.end());
  EXPECT_EQ(prologue->workgroup_shadow_size, 0u);
  EXPECT_NE(std::ranges::find(
                result.warnings,
                "ConSan MOI inline-shadow access falls back to the external exact-shadow table"),
            result.warnings.end());
}

TEST(ConSanMoi, InlineShadowUsesExternalMirrorForDynamicLdsKernel) {
  constexpr auto store = gfx1250::build_vds(gfx1250::kDsStoreB32Vds, {.addr = 0, .data0 = 1});
  const std::array<uint32_t, 3> text_words = {
      store[0],
      store[1],
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250),
  };
  constexpr std::string_view kernel_name = "dynamic_lds_external_shadow";
  std::vector<uint8_t> bytes = make_gfx1250_code_object(text_words, kernel_name);
  mutate_first_kernel_descriptor(bytes,
                                 [](KD &descriptor) { descriptor.group_segment_fixed_size = 2u; });
  append_kernel_metadata_note(bytes, kernel_name, /*uses_dynamic_stack=*/false,
                              /*sgpr_count=*/0u, std::nullopt, std::array<uint8_t, 3>{64u, 1u, 1u},
                              /*has_dynamic_lds=*/true);
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_inline_workgroup_shadow = true;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.kernels.size(), 1u);
  EXPECT_TRUE(result.kernels.front().has_dynamic_lds);
  EXPECT_FALSE(result.moi_private_epoch_automatic);
  EXPECT_TRUE(result.moi_persistent_vgprs_automatic);
  EXPECT_TRUE(result.resolved_moi_owner_vgpr);
  EXPECT_TRUE(result.resolved_moi_epoch_vgpr);
  EXPECT_TRUE(result.resolved_moi_workgroup_key_vgpr);
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  });
  ASSERT_NE(access, result.patches.end());
  EXPECT_FALSE(access->persistent_epoch_private_offset);
  EXPECT_FALSE(access->persistent_owner_private_offset);
  EXPECT_FALSE(access->persistent_workgroup_key_private_offset);
  EXPECT_EQ(access->workgroup_shadow_size, 0u);
  EXPECT_EQ(access->required_group_segment_size, 0u);
  EXPECT_NE(std::ranges::find(
                result.warnings,
                "ConSan MOI inline-shadow dynamic-LDS owner uses the external exact-shadow table"),
            result.warnings.end());
}

TEST(ConSanMoi, InlineShadowLoopsOverEveryWideWorkgroupLocalCellCompactly) {
  const std::array<uint32_t, 3> text_words = {
      0xDB7C0000u,
      0x00000109u, // ds_store_b128 v0, v[1:4]
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "wide_workgroup_shadow", kRdna4Wave64AllVgprsGranulated, false, false, 0, 4352u);
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_inline_workgroup_shadow = true;
  options.moi_init_owner_epoch = true;
  options.scratch_vgpr = 16;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.moi_candidates.size(), 1u);
  EXPECT_EQ(result.moi_candidates.front().width_bits, 128u);
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  });
  ASSERT_NE(access, result.patches.end());
  ASSERT_TRUE(access->scratch_vgpr);
  // The former unrolled implementation emitted roughly 6.8 KiB per b128
  // site. One exact transaction plus a bounded cell loop keeps the complete
  // probe below half that size while covering the possible fifth cell of an
  // unaligned sixteen-byte access.
  EXPECT_LT(access->trampoline_size, 3000u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const auto *text = patched.text_sections().front();
  ASSERT_LE(access->trampoline_offset + access->trampoline_size, text->size());
  std::vector<uint32_t> body(access->trampoline_size / sizeof(uint32_t));
  std::memcpy(body.data(), text->data() + access->trampoline_offset, access->trampoline_size);

  const uint16_t scratch = *access->scratch_vgpr;
  const auto local_exchange = build_ds_storexchg_rtn_b64(
      static_cast<uint16_t>(scratch + 5u), scratch, static_cast<uint16_t>(scratch + 2u),
      /*byte_offset=*/0, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(local_exchange);
  EXPECT_EQ(count_subsequence(body, *local_exchange), 1u);
  const auto exec_loop_encoding = build_s_cbranch_execnz(0, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(exec_loop_encoding);
  ASSERT_TRUE(result.resolved_moi_exec_save_sgpr);
  const auto masked_predicate =
      build_s_and_saveexec_b64(static_cast<uint16_t>(*result.resolved_moi_exec_save_sgpr + 14u),
                               /*vcc=*/106, ROCJITSU_CODE_ARCH_RDNA4);
  const uint16_t loop_counter_vgpr =
      consan_detail::inline_shadow_loop_counter_vgpr(scratch, /*has_exec_save=*/true,
                                                     /*track_atomics=*/false);
  const auto saturate_cell_start = ib::build_v_min_u32(
      static_cast<uint16_t>(scratch + 4u), vector_source_vgpr(static_cast<uint16_t>(scratch + 8u)),
      static_cast<uint16_t>(scratch + 4u), ROCJITSU_CODE_ARCH_RDNA4);
  const auto advance_counter =
      ib::build_v_add_u32(loop_counter_vgpr, scalar_positive_inline_u32(1u), loop_counter_vgpr,
                          ROCJITSU_CODE_ARCH_RDNA4);
  const auto more_cells = ib::build_v_cmp_gt_u32_vcc(
      // An unaligned sixteen-byte access touches at most five four-byte cells.
      scalar_positive_inline_u32(5u), loop_counter_vgpr, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(masked_predicate);
  ASSERT_TRUE(saturate_cell_start);
  ASSERT_TRUE(advance_counter);
  ASSERT_TRUE(more_cells);
  EXPECT_EQ(std::ranges::count(body, *saturate_cell_start), 1u)
      << "the worst-case fifth cell must saturate at the access end";
  std::vector<uint32_t> expected_loop_control(advance_counter->begin(), advance_counter->end());
  expected_loop_control.push_back(*more_cells);
  expected_loop_control.push_back(*masked_predicate);
  const auto loop_control = std::search(body.begin(), body.end(), expected_loop_control.begin(),
                                        expected_loop_control.end());
  ASSERT_NE(loop_control, body.end());
  const auto state_index =
      ib::build_v_and_b32_literal(static_cast<uint16_t>(scratch + 8u), 15u,
                                  static_cast<uint16_t>(scratch + 7u), ROCJITSU_CODE_ARCH_RDNA4);
  const auto scale_state_index =
      ib::build_v_lshlrev_b32(static_cast<uint16_t>(scratch + 8u), scalar_positive_inline_u32(1u),
                              static_cast<uint16_t>(scratch + 8u), ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(state_index);
  ASSERT_TRUE(scale_state_index);
  std::vector<uint32_t> expected_two_bit_state(state_index->begin(), state_index->end());
  expected_two_bit_state.push_back(*scale_state_index);
  EXPECT_TRUE(contains_subsequence(body, expected_two_bit_state))
      << "adjacent shadow cells must use disjoint initializing/ready bit pairs";
  const auto loop_branch = std::next(loop_control, expected_loop_control.size());
  ASSERT_NE(loop_branch, body.end());
  EXPECT_EQ(*loop_branch & 0xffff0000u, *exec_loop_encoding & 0xffff0000u);
  EXPECT_LT(static_cast<int16_t>(*loop_branch), 0);
  EXPECT_EQ(*(loop_branch - 1), *masked_predicate);
  EXPECT_EQ(std::ranges::count_if(body,
                                  [&](uint32_t word) {
                                    return (word & 0xffff0000u) ==
                                               (*exec_loop_encoding & 0xffff0000u) &&
                                           static_cast<int16_t>(word) < 0;
                                  }),
            2u)
      << "wide lazy shadow requires one readiness poll and one publication loop";
  EXPECT_TRUE(validate_consan_modified_elf(bytes, result).empty());

  ConSanResult broken_loop = result;
  const size_t loop_byte_offset =
      static_cast<size_t>(std::distance(body.begin(), loop_branch)) * sizeof(uint32_t);
  const uint32_t nop = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
  std::memcpy(broken_loop.elf_bytes.data() + result.text_sections.front().file_offset +
                  access->trampoline_offset + loop_byte_offset,
              &nop, sizeof(nop));
  const std::vector<std::string> loop_errors = validate_consan_modified_elf(bytes, broken_loop);
  EXPECT_TRUE(std::ranges::any_of(loop_errors, [](const std::string &error) {
    return error.find("workgroup-local exact-shadow publication semantics") != std::string::npos;
  })) << testing::PrintToString(loop_errors);
}

TEST(ConSanMoi, InlineWorkgroupShadowPublishesVisibleEvidenceOncePerAccess) {
  const std::array<uint32_t, 3> text_words = {
      0xD9DC0201u,
      0x01000009u, // ds_load_2addr_b64: two disjoint ranges, four exact cells
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "multi_range_visible_evidence",
                                 kRdna4Wave64AllVgprsGranulated, false, false, 0, 4352u);
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_inline_workgroup_shadow = true;
  options.moi_init_owner_epoch = true;
  options.scratch_vgpr = 16;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.moi_candidates.size(), 1u);
  EXPECT_EQ(result.moi_candidates.front().mnemonic, "ds_load_2addr_b64");
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  });
  ASSERT_NE(access, result.patches.end());
  ASSERT_EQ(access->scratch_vgpr, 16);
  EXPECT_LT(access->trampoline_size, 3400u)
      << "two-range local diagnostics must retain one bounded loop per range, coherent exact-byte "
         "provenance, and the bitmap cold path without returning to per-cell unrolling";

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const auto *text = patched.text_sections().front();
  ASSERT_LE(access->trampoline_offset + access->trampoline_size, text->size());
  std::vector<uint32_t> body(access->trampoline_size / sizeof(uint32_t));
  std::memcpy(body.data(), text->data() + access->trampoline_offset, access->trampoline_size);

  const auto visible_load = build_flat_load_b32_vaddr_vdst(
      /*vaddr=*/16, /*vdst=*/20, ROCJITSU_CODE_ARCH_RDNA4);
  const auto visible_atomic = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/16, /*vsrc=*/20, /*vdst=*/20, /*return_old_value=*/true,
      /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  const auto event_counter_address = build_v_mov_b32_e64_literal(
      /*vdst=*/16,
      static_cast<uint32_t>(*options.moi_report_buffer_address +
                            offsetof(ConSanMoiReportHeader, event_counter)),
      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(visible_load);
  ASSERT_TRUE(visible_atomic);
  ASSERT_TRUE(event_counter_address);
  EXPECT_EQ(count_subsequence(body, *visible_load), 1u)
      << "one invocation should inspect aggregate evidence once, not once per range or cell";
  EXPECT_EQ(count_subsequence(body, *event_counter_address), 2u)
      << "one invocation should form the event-counter address once for its load and once for "
         "its conditional atomic, not once per range or cell";
  EXPECT_GE(count_subsequence(body, *visible_atomic), 1u);
}

TEST(ConSanMoi, Gfx1250InlineWorkgroupShadowCachesEvidenceInFreshScalarState) {
  constexpr auto store0 = gfx1250::build_vds(gfx1250::kDsStoreB32Vds, {.addr = 0, .data0 = 0});
  constexpr auto store1 = gfx1250::build_vds(gfx1250::kDsStoreB32Vds, {.addr = 0, .data0 = 1});
  const std::array<uint32_t, 5> text_words = {
      store0[0], store0[1], store1[0], store1[1], build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250),
  };
  std::vector<uint8_t> bytes =
      make_gfx1250_code_object(text_words, "workgroup_visible_evidence_latch");
  append_kernel_metadata_note(bytes, "workgroup_visible_evidence_latch",
                              /*uses_dynamic_stack=*/false, /*sgpr_count=*/0u, std::nullopt,
                              std::array<uint8_t, 3>{64u, 1u, 1u});
  mutate_first_kernel_descriptor(
      bytes, [](KD &descriptor) { descriptor.group_segment_fixed_size = 4360u; });
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_inline_workgroup_shadow = true;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.max_patches = 2;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.resolved_moi_exec_save_sgpr);
  const uint16_t evidence_latch = static_cast<uint16_t>(*result.resolved_moi_exec_save_sgpr + 24u);
  const auto prologue = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue;
  });
  ASSERT_NE(prologue, result.patches.end());
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> prologue_words =
      text_words_at_offset(patched, prologue->trampoline_offset, prologue->trampoline_size);
  ASSERT_TRUE(result.resolved_moi_owner_vgpr);
  ASSERT_TRUE(result.resolved_moi_epoch_vgpr);
  const uint16_t initializer_address = *result.resolved_moi_owner_vgpr;
  ASSERT_TRUE(result.kernels.front().required_workgroup_size);
  EXPECT_EQ(*result.kernels.front().required_workgroup_size,
            (std::array<uint32_t, 3>{64u, 1u, 1u}));
  const uint16_t initializer_offset = static_cast<uint16_t>(*result.resolved_moi_epoch_vgpr + 1u);
  const auto extract_x = ib::build_v_and_b32_literal(
      initializer_offset, 0x3ffu, /*packed_workitem_id=*/0u, ROCJITSU_CODE_ARCH_GFX1250);
  const auto select_initialization_x_lanes = ib::build_v_cmp_gt_u32_vcc(
      scalar_positive_inline_u32(64u), initializer_offset, ROCJITSU_CODE_ARCH_GFX1250);
  const auto legacy_first_wave = ib::build_v_cmp_gt_u32_vcc(
      scalar_positive_inline_u32(32u), /*workitem_id_x=*/0u, ROCJITSU_CODE_ARCH_GFX1250);
  const auto scale_x = ib::build_v_lshlrev_b32(initializer_offset, scalar_positive_inline_u32(4u),
                                               initializer_offset, ROCJITSU_CODE_ARCH_GFX1250);
  EXPECT_EQ(prologue->workgroup_shadow_validity_base, 13080u);
  EXPECT_EQ(prologue->workgroup_shadow_validity_size, 288u);
  const auto add_shadow_base =
      ib::build_v_add_u32_literal(initializer_address, /*validity base=*/13080u, initializer_offset,
                                  ROCJITSU_CODE_ARCH_GFX1250);
  const auto advance_row = ib::build_v_add_u32(
      initializer_address, static_cast<uint16_t>(*result.resolved_moi_exec_save_sgpr + 4u),
      initializer_address, ROCJITSU_CODE_ARCH_GFX1250);
  const auto select_x_zero = ib::build_v_cmp_eq_u32_vcc(
      scalar_positive_inline_u32(0u), /*workitem_id_x=*/0u, ROCJITSU_CODE_ARCH_GFX1250);
  const auto end_address = ib::build_v_cmp_gt_u32_literal_vcc(
      /*13080-byte base + 288-byte validity state=*/13368u, initializer_address,
      ROCJITSU_CODE_ARCH_GFX1250);
  const auto store_wide = ib::build_ds_store_b128(
      initializer_address, *result.resolved_moi_epoch_vgpr, 0u, ROCJITSU_CODE_ARCH_GFX1250);
  const auto count_lanes =
      ib::build_s_bcnt1_i32_b64(static_cast<uint16_t>(*result.resolved_moi_exec_save_sgpr + 4u),
                                /*exec=*/126u, ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(extract_x);
  ASSERT_TRUE(select_initialization_x_lanes);
  ASSERT_TRUE(legacy_first_wave);
  ASSERT_TRUE(scale_x);
  ASSERT_TRUE(add_shadow_base);
  ASSERT_TRUE(advance_row);
  ASSERT_TRUE(select_x_zero);
  ASSERT_TRUE(end_address);
  ASSERT_TRUE(store_wide);
  ASSERT_TRUE(count_lanes);
  EXPECT_TRUE(contains_subsequence(prologue_words, *extract_x));
  EXPECT_NE(std::ranges::find(prologue_words, *select_initialization_x_lanes),
            prologue_words.end());
  EXPECT_EQ(std::ranges::count(prologue_words, *legacy_first_wave), 0u)
      << "a proven 64-workitem x-row must distribute initialization across both wave32 waves";
  EXPECT_NE(std::ranges::find(prologue_words, *scale_x), prologue_words.end());
  EXPECT_TRUE(contains_subsequence(prologue_words, *add_shadow_base));
  EXPECT_TRUE(contains_subsequence(prologue_words, *advance_row));
  EXPECT_EQ(std::ranges::count(prologue_words, *select_x_zero), 0u)
      << "one-dimensional shadows must use all 32 lanes of the first x-wave";
  EXPECT_TRUE(contains_subsequence(prologue_words, *end_address));
  EXPECT_TRUE(contains_subsequence(prologue_words, *store_wide));
  for (uint16_t i = 0u; i < 4u; ++i) {
    EXPECT_NE(std::ranges::find(
                  prologue_words,
                  build_v_mov_b32_e32(static_cast<uint16_t>(*result.resolved_moi_epoch_vgpr + i),
                                      scalar_positive_inline_u32(0u), ROCJITSU_CODE_ARCH_GFX1250)),
              prologue_words.end());
  }
  EXPECT_NE(std::ranges::find(prologue_words, *count_lanes), prologue_words.end())
      << "the runtime x-row width must determine the exact shadow stride";
  EXPECT_NE(std::find(prologue_words.begin(), prologue_words.end(),
                      build_s_mov_b32(evidence_latch, scalar_positive_inline_u32(0),
                                      ROCJITSU_CODE_ARCH_GFX1250)),
            prologue_words.end())
      << testing::PrintToString(result.warnings);

  const auto latch_nonzero = build_rdna4_s_cmp_lg_u32(evidence_latch, scalar_positive_inline_u32(0),
                                                      ROCJITSU_CODE_ARCH_GFX1250);
  const auto skip_published = build_s_cbranch_scc1(0, ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(latch_nonzero);
  ASSERT_TRUE(skip_published);
  size_t access_count = 0;
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.kind != ConSanPatchKind::TrampolineMoiExactShadowStore)
      continue;
    ++access_count;
    const std::vector<uint32_t> body =
        text_words_at_offset(patched, patch.trampoline_offset, patch.trampoline_size);
    const auto compare = std::find(body.begin(), body.end(), *latch_nonzero);
    ASSERT_NE(compare, body.end());
    ASSERT_NE(compare + 1, body.end());
    EXPECT_EQ((*(compare + 1)) & 0xffff0000u, (*skip_published) & 0xffff0000u);
    EXPECT_NE(std::find(compare + 2, body.end(),
                        build_s_mov_b32(evidence_latch, scalar_positive_inline_u32(1),
                                        ROCJITSU_CODE_ARCH_GFX1250)),
              body.end());
  }
  EXPECT_EQ(access_count, 2u);
  EXPECT_TRUE(validate_consan_modified_elf(bytes, result).empty());
}

TEST(ConSanMoi, Gfx1250InlineWorkgroupShadowValidatesAtomicAccessCandidate) {
  constexpr auto atomic =
      gfx1250::build_vds(gfx1250::kDsAddU32Vds, {.offset0 = 12u, .addr = 2u, .data0 = 1u});
  const std::array<uint32_t, 3> text_words = {atomic[0], atomic[1],
                                              build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250)};
  std::vector<uint8_t> bytes = make_gfx1250_code_object(text_words, "local_atomic_shadow");
  append_kernel_metadata_note(bytes, "local_atomic_shadow",
                              /*uses_dynamic_stack=*/false, /*sgpr_count=*/0u, std::nullopt,
                              std::array<uint8_t, 3>{64u, 1u, 1u});
  mutate_first_kernel_descriptor(bytes,
                                 [](KD &descriptor) { descriptor.group_segment_fixed_size = 16u; });

  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_inline_workgroup_shadow = true;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.max_patches = 1u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.final_validation_passed);
  ASSERT_EQ(result.moi_candidates.size(), 1u);
  EXPECT_EQ(result.moi_candidates.front().kind, ConSanLdsAccessKind::Atomic);
  const auto exact_patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::InlineMoiExactShadowStore ||
           patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  });
  ASSERT_NE(exact_patch, result.patches.end());
  EXPECT_NE(exact_patch->workgroup_shadow_size, 0u);
  EXPECT_TRUE(validate_consan_modified_elf(bytes, result).empty());
}

TEST(ConSanMoi, Gfx1250InlineGlobalShadowUsesLiteralDispatchIdAtFullScalarPressure) {
  std::vector<uint32_t> text_words(800, build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250));
  constexpr auto store = gfx1250::build_vds(gfx1250::kDsStoreB32Vds, {.addr = 0, .data0 = 1});
  text_words[0] = store[0];
  text_words[1] = store[1];
  text_words[2] = build_s_mov_b32(/*sdst=*/0u, /*ssrc0=*/106u, ROCJITSU_CODE_ARCH_GFX1250);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250);
  const std::vector<uint8_t> bytes = make_gfx1250_code_object(text_words);

  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_dispatch_id = 0x1122334455667788ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.max_patches = 16u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_FALSE(result.resolved_moi_dispatch_id_sgpr);
  EXPECT_EQ(result.moi_report_dispatch_id, options.moi_report_dispatch_id);
  const auto exact_patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::InlineMoiExactShadowStore ||
           patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  });
  EXPECT_NE(exact_patch, result.patches.end());
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_NE(result.resource_plans.front().source, ConSanRegisterAllocationSource::Unsupported);
}

TEST(ConSanMoi, Rdna4InlineGlobalShadowSpillsFullScalarPressure) {
  std::vector<uint32_t> text_words(800, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  size_t cursor = 0;
  // Reference the top ordinary SGPR, while keeping only scattered scalar
  // values live across the access. There is no 28-register dead window, but
  // there are enough dead registers for the external-shadow router.
  text_words[cursor++] = build_s_mov_b32(0, 105u, ROCJITSU_CODE_ARCH_RDNA4);
  constexpr std::array<uint16_t, 6> live_sgprs = {15u, 31u, 47u, 63u, 79u, 95u};
  for (uint16_t sgpr : live_sgprs) {
    text_words[cursor++] =
        build_s_mov_b32(sgpr, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
  }
  const size_t access_word = cursor;
  text_words[cursor++] = 0xD8340000u;
  text_words[cursor++] = 0x00000000u; // ds_store_b32 v0, v0
  for (uint16_t sgpr : live_sgprs)
    text_words[cursor++] = build_s_mov_b32(0, sgpr, ROCJITSU_CODE_ARCH_RDNA4);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);
  for (const bool uses_dynamic_stack : {false, true}) {
    SCOPED_TRACE(uses_dynamic_stack ? "dynamic-stack" : "fixed-stack");
    const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
        text_words, "inline_scalar_spill", kRdna4Wave64AllVgprsGranulated, /*wave32=*/false,
        uses_dynamic_stack);

    ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
    options.moi_report_buffer_address = 0x123456780000ull;
    options.moi_report_dispatch_id = 0x1122334455667788ull;
    options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
    options.max_patches = 16u;

    const ConSanResult result = try_patch_consan(bytes, options);

    ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
    ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
    EXPECT_TRUE(result.final_validation_passed);
    // Complete owner coverage proves that this pair is absent from guest
    // code, so preserve runtime dispatch identity instead of falling back to
    // the configured literal merely because the live scalar set is sparse.
    EXPECT_TRUE(result.resolved_moi_dispatch_id_sgpr);
    EXPECT_EQ(result.moi_report_dispatch_id, options.moi_report_dispatch_id);
    EXPECT_TRUE(std::ranges::any_of(result.patches, [&](const ConSanPatchInfo &patch) {
      return patch.anchor_offset == access_word * sizeof(uint32_t) &&
             (patch.kind == ConSanPatchKind::InlineMoiExactShadowStore ||
              patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore);
    }));
    ASSERT_EQ(result.resource_plans.size(), 1u);
    EXPECT_NE(result.resource_plans.front().source, ConSanRegisterAllocationSource::Unsupported);
    EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
      return warning.find("automatically assigned spill-backed Inline SGPRs") != std::string::npos;
    })) << testing::PrintToString(result.warnings);
    EXPECT_TRUE(std::ranges::any_of(result.patches, [](const ConSanPatchInfo &patch) {
      return patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore &&
             patch.required_private_segment_size != 0u;
    }));
    if (uses_dynamic_stack) {
      EXPECT_TRUE(std::ranges::any_of(result.patches, [](const ConSanPatchInfo &patch) {
        return patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore &&
               patch.dynamic_private_segment_addend > 0u;
      }));
    }
  }
}

TEST(ConSanMoi, Rdna4InlineBranchOnlyDynamicStackPreservesEntryScalarInputs) {
  std::vector<uint32_t> text_words(800u, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  size_t cursor = 0u;
  text_words[cursor++] = 0xD8340000u;
  text_words[cursor++] = 0x00000000u; // ds_store_b32 v0, v0
  // Make every ordinary SGPR live at the access without assuming any
  // preserved scalar range. Consume s105 before reusing it as the destination
  // for the remaining reads.
  text_words[cursor++] = build_s_mov_b32(105u, 105u, ROCJITSU_CODE_ARCH_RDNA4);
  for (uint16_t sgpr = 0u; sgpr < 105u; ++sgpr)
    text_words[cursor++] = build_s_mov_b32(105u, sgpr, ROCJITSU_CODE_ARCH_RDNA4);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "inline_branch_only_dynamic_stack",
                                 kRdna4Wave64AllVgprsGranulated, /*wave32=*/false,
                                 /*uses_dynamic_stack=*/true);

  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_dispatch_id = 0x1122334455667788ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.max_patches = 1u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.final_validation_passed);
  ASSERT_EQ(result.resolved_moi_transient_sgpr_assignments.size(), 1u)
      << testing::PrintToString(result.warnings);
  const ConSanMoiTransientSgprAssignment &assignment =
      result.resolved_moi_transient_sgpr_assignments.front();
  ASSERT_TRUE(assignment.spill_backed);
  ASSERT_TRUE(assignment.branch_only_scalar_spill);
  ASSERT_TRUE(assignment.dynamic_stack_borrowed_sgpr);
  EXPECT_FALSE(assignment.indirect_pc_sgpr);
  EXPECT_FALSE(assignment.indirect_scc_sgpr);
  EXPECT_FALSE(assignment.dispatch_key_sgpr);
  EXPECT_FALSE(assignment.call_return_sgpr);

  const auto prologue = std::ranges::find(
      result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue, &ConSanPatchInfo::kind);
  ASSERT_NE(prologue, result.patches.end());
  ASSERT_TRUE(prologue->entry_scalar_backup_vgpr);
  EXPECT_EQ(prologue->entry_scalar_backup_sgpr_base, assignment.exec_save_sgpr);
  EXPECT_EQ(prologue->entry_scalar_backup_sgpr_count, kConSanMoiInlineExecSaveSgprCount);
  ASSERT_EQ(prologue->owner_descriptor_file_offsets.size(), 1u);
  EXPECT_EQ(prologue->owner_descriptor_file_offsets.front(), assignment.descriptor_file_offset);
  const auto branch_only =
      std::ranges::find(result.patches, true, &ConSanPatchInfo::branch_only_continuation);
  ASSERT_NE(branch_only, result.patches.end());
  EXPECT_EQ(branch_only->anchor_offset, 0u);
  EXPECT_EQ(branch_only->branch_only_entry_prologue_offset, prologue->trampoline_offset);
  EXPECT_EQ(prologue->entry_prologue_chained_trampoline_offset, branch_only->trampoline_offset);
  EXPECT_TRUE(branch_only->branch_only_entry_relay_offsets.empty());
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineNopBranchRelay,
                               &ConSanPatchInfo::kind),
            0u);
  EXPECT_EQ(result.moi_branch_only_routing_telemetry.pair_attempt_count, 1u);
  EXPECT_EQ(result.moi_branch_only_routing_telemetry.plan_call_count, 1u);
  EXPECT_EQ(result.moi_branch_only_routing_telemetry.work_budget_exhaustion_count, 0u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> prologue_words =
      text_words_at_offset(patched, prologue->trampoline_offset, prologue->trampoline_size);
  const auto first_save = instrumentation::build_v_writelane_b32(
      *prologue->entry_scalar_backup_vgpr, assignment.exec_save_sgpr, /*lane=*/0u,
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto last_save = instrumentation::build_v_writelane_b32(
      *prologue->entry_scalar_backup_vgpr,
      static_cast<uint16_t>(assignment.exec_save_sgpr + kConSanMoiInlineExecSaveSgprCount - 1u),
      kConSanMoiInlineExecSaveSgprCount - 1u, ROCJITSU_CODE_ARCH_RDNA4);
  const auto first_restore = instrumentation::build_v_readlane_b32(
      assignment.exec_save_sgpr, *prologue->entry_scalar_backup_vgpr, /*lane=*/0u,
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto last_restore = instrumentation::build_v_readlane_b32(
      static_cast<uint16_t>(assignment.exec_save_sgpr + kConSanMoiInlineExecSaveSgprCount - 1u),
      *prologue->entry_scalar_backup_vgpr, kConSanMoiInlineExecSaveSgprCount - 1u,
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_wait = instrumentation::build_s_wait_alu_va_sdst0(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(first_save);
  ASSERT_TRUE(last_save);
  ASSERT_TRUE(first_restore);
  ASSERT_TRUE(last_restore);
  ASSERT_TRUE(restore_wait);
  ASSERT_GE(prologue_words.size(), first_save->size());
  EXPECT_TRUE(std::ranges::equal(std::span(prologue_words).first<2>(), *first_save));
  EXPECT_TRUE(contains_subsequence(prologue_words, *last_save));
  EXPECT_TRUE(contains_subsequence(prologue_words, *first_restore));
  const std::array<uint32_t, 3> final_restore = {
      (*last_restore)[0],
      (*last_restore)[1],
      *restore_wait,
  };
  EXPECT_TRUE(contains_subsequence(prologue_words, final_restore));

  ConSanResult corrupted = result;
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const uint64_t backup_body_offset =
      prologue->dispatch_id_primary_prologue_offset.value_or(prologue->trampoline_offset);
  corrupted.elf_bytes[patched.text_sections().front()->sectionOffset() + backup_body_offset] ^= 1u;
  const std::vector<std::string> validation_errors = validate_consan_modified_elf(bytes, corrupted);
  EXPECT_TRUE(std::ranges::any_of(validation_errors, [](const std::string &error) {
    return error.find("invalid entry scalar backup save") != std::string::npos;
  })) << testing::PrintToString(validation_errors);

  ConSanResult invalid_resources = result;
  const auto invalid_prologue =
      std::ranges::find(invalid_resources.patches,
                        ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue, &ConSanPatchInfo::kind);
  ASSERT_NE(invalid_prologue, invalid_resources.patches.end());
  invalid_prologue->entry_scalar_backup_vgpr = kConSanOrdinaryVgprLimit;
  const std::vector<std::string> resource_errors =
      validate_consan_modified_elf(bytes, invalid_resources);
  EXPECT_TRUE(std::ranges::any_of(resource_errors, [](const std::string &error) {
    return error.find("invalid entry scalar backup resources") != std::string::npos;
  })) << testing::PrintToString(resource_errors);
}

void check_inline_branch_only_fixed_stack_preserves_entry_scalar_inputs(rj_code_arch_t arch) {
  std::vector<uint32_t> text_words(800u, build_s_nop(0, arch));
  size_t cursor = 0u;
  if (arch == ROCJITSU_CODE_ARCH_GFX1250) {
    constexpr auto store = gfx1250::build_vds(gfx1250::kDsStoreB32Vds, {.addr = 0u, .data0 = 0u});
    text_words[cursor++] = store[0];
    text_words[cursor++] = store[1];
  } else {
    text_words[cursor++] = 0xD8340000u;
    text_words[cursor++] = 0x00000000u; // ds_store_b32 v0, v0
  }
  text_words[cursor++] = build_s_mov_b32(105u, 105u, arch);
  for (uint16_t sgpr = 0u; sgpr < 105u; ++sgpr)
    text_words[cursor++] = build_s_mov_b32(105u, sgpr, arch);
  text_words.back() = build_s_endpgm(arch);
  const std::vector<uint8_t> bytes =
      arch == ROCJITSU_CODE_ARCH_GFX1250
          ? make_gfx1250_code_object(text_words, "inline_branch_only_fixed_stack_gfx1250",
                                     kRdna4Wave64AllVgprsGranulated, /*wave32=*/true,
                                     /*uses_dynamic_stack=*/false)
          : make_rdna4_lds_code_object(text_words, "inline_branch_only_fixed_stack_rdna4",
                                       kRdna4Wave64AllVgprsGranulated, /*wave32=*/false,
                                       /*uses_dynamic_stack=*/false);

  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_dispatch_id = 0x1122334455667788ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.max_patches = 1u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.final_validation_passed);
  ASSERT_EQ(result.resolved_moi_transient_sgpr_assignments.size(), 1u)
      << testing::PrintToString(result.warnings);
  const ConSanMoiTransientSgprAssignment &assignment =
      result.resolved_moi_transient_sgpr_assignments.front();
  ASSERT_TRUE(assignment.spill_backed);
  ASSERT_TRUE(assignment.branch_only_scalar_spill);
  EXPECT_FALSE(assignment.dynamic_stack_borrowed_sgpr);
  EXPECT_FALSE(assignment.indirect_pc_sgpr);
  EXPECT_FALSE(assignment.indirect_scc_sgpr);
  EXPECT_FALSE(assignment.dispatch_key_sgpr);
  EXPECT_FALSE(assignment.call_return_sgpr);

  const auto prologue = std::ranges::find(
      result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue, &ConSanPatchInfo::kind);
  ASSERT_NE(prologue, result.patches.end());
  ASSERT_TRUE(prologue->entry_scalar_backup_vgpr);
  EXPECT_EQ(prologue->entry_scalar_backup_sgpr_base, assignment.exec_save_sgpr);
  EXPECT_EQ(prologue->entry_scalar_backup_sgpr_count, kConSanMoiInlineExecSaveSgprCount);
  const auto branch_only =
      std::ranges::find(result.patches, true, &ConSanPatchInfo::branch_only_continuation);
  ASSERT_NE(branch_only, result.patches.end());
  EXPECT_EQ(branch_only->branch_only_entry_prologue_offset, prologue->trampoline_offset);
  EXPECT_EQ(prologue->entry_prologue_chained_trampoline_offset, branch_only->trampoline_offset);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> prologue_words =
      text_words_at_offset(patched, prologue->trampoline_offset, prologue->trampoline_size);
  const auto first_save = instrumentation::build_v_writelane_b32(
      *prologue->entry_scalar_backup_vgpr, assignment.exec_save_sgpr, /*lane=*/0u, arch);
  const auto last_save = instrumentation::build_v_writelane_b32(
      *prologue->entry_scalar_backup_vgpr,
      static_cast<uint16_t>(assignment.exec_save_sgpr + kConSanMoiInlineExecSaveSgprCount - 1u),
      kConSanMoiInlineExecSaveSgprCount - 1u, arch);
  const auto first_restore = instrumentation::build_v_readlane_b32(
      assignment.exec_save_sgpr, *prologue->entry_scalar_backup_vgpr, /*lane=*/0u, arch);
  const auto last_restore = instrumentation::build_v_readlane_b32(
      static_cast<uint16_t>(assignment.exec_save_sgpr + kConSanMoiInlineExecSaveSgprCount - 1u),
      *prologue->entry_scalar_backup_vgpr, kConSanMoiInlineExecSaveSgprCount - 1u, arch);
  ASSERT_TRUE(first_save);
  ASSERT_TRUE(last_save);
  ASSERT_TRUE(first_restore);
  ASSERT_TRUE(last_restore);
  ASSERT_GE(prologue_words.size(), first_save->size());
  EXPECT_TRUE(std::ranges::equal(std::span(prologue_words).first<2>(), *first_save));
  EXPECT_TRUE(contains_subsequence(prologue_words, *last_save));
  EXPECT_TRUE(contains_subsequence(prologue_words, *first_restore));
  EXPECT_TRUE(contains_subsequence(prologue_words, *last_restore));
}

TEST(ConSanMoi, Rdna4InlineBranchOnlyFixedStackPreservesEntryScalarInputs) {
  check_inline_branch_only_fixed_stack_preserves_entry_scalar_inputs(ROCJITSU_CODE_ARCH_RDNA4);
}

TEST(ConSanMoi, Gfx1250InlineBranchOnlyFixedStackPreservesEntryScalarInputs) {
  check_inline_branch_only_fixed_stack_preserves_entry_scalar_inputs(ROCJITSU_CODE_ARCH_GFX1250);
}

TEST(ConSanMoi, Rdna4InlineUnknownStackDoesNotSelectFixedStackBranchOnlySpill) {
  std::vector<uint32_t> text_words(800u, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  size_t cursor = 0u;
  text_words[cursor++] = 0xD8340000u;
  text_words[cursor++] = 0x00000000u; // ds_store_b32 v0, v0
  text_words[cursor++] = build_s_mov_b32(105u, 105u, ROCJITSU_CODE_ARCH_RDNA4);
  for (uint16_t sgpr = 0u; sgpr < 105u; ++sgpr)
    text_words[cursor++] = build_s_mov_b32(105u, sgpr, ROCJITSU_CODE_ARCH_RDNA4);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);
  std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "inline_branch_only_unknown_stack",
                                 kRdna4Wave64AllVgprsGranulated, /*wave32=*/false,
                                 /*uses_dynamic_stack=*/false);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT, 1u);
  });
  mutate_elf_symbol_by_name(bytes, "inline_branch_only_unknown_stack.has_dyn_sized_stack",
                            [](Elf64_Sym &symbol) { symbol.st_name = 0u; });

  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.max_patches = 1u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  ASSERT_EQ(result.kernels.size(), 1u);
  EXPECT_FALSE(result.kernels.front().uses_dynamic_stack.has_value());
  EXPECT_FALSE(std::ranges::any_of(result.resolved_moi_transient_sgpr_assignments,
                                   &ConSanMoiTransientSgprAssignment::branch_only_scalar_spill))
      << testing::PrintToString(result.warnings);
  EXPECT_FALSE(std::ranges::any_of(result.patches, &ConSanPatchInfo::branch_only_continuation))
      << testing::PrintToString(result.warnings);
}

TEST(ConSanMoi, Rdna4BranchOnlyDynamicStackRoutesThroughIsolatedNopWords) {
  constexpr size_t kSegmentWords = 20'000u;
  constexpr size_t kEntryRelayWord = 15'000u;
  constexpr size_t kUnusedEntryRelayWord = kEntryRelayWord + 1u;
  constexpr size_t kReturnRelayWord = 5'000u;
  const uint32_t filler = build_s_mov_b32(/*sdst=*/105u, /*ssrc0=*/105u, ROCJITSU_CODE_ARCH_RDNA4);
  std::vector<uint32_t> kernel_words(7u, filler);
  kernel_words.reserve(kSegmentWords);
  const uint64_t access_offset = kernel_words.size() * sizeof(uint32_t);
  kernel_words.push_back(0xD8340000u);
  kernel_words.push_back(0x00000000u); // ds_store_b32 v0, v0
  kernel_words.push_back(filler);
  for (uint16_t sgpr = 0u; sgpr < 105u; ++sgpr)
    kernel_words.push_back(build_s_mov_b32(/*sdst=*/105u, sgpr, ROCJITSU_CODE_ARCH_RDNA4));
  kernel_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4));
  const uint64_t kernel_code_size = kernel_words.size() * sizeof(uint32_t);
  kernel_words.resize(kSegmentWords, filler);
  kernel_words[kEntryRelayWord] = build_s_nop(0u, ROCJITSU_CODE_ARCH_RDNA4);
  kernel_words[kUnusedEntryRelayWord] = build_s_nop(0u, ROCJITSU_CODE_ARCH_RDNA4);

  std::vector<uint32_t> function_words(kSegmentWords, filler);
  function_words.front() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);
  function_words[kReturnRelayWord] = build_s_nop(0u, ROCJITSU_CODE_ARCH_RDNA4);
  std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words);
  mutate_elf_symbol_by_name(bytes, "lds_probe",
                            [&](Elf64_Sym &symbol) { symbol.st_size = kernel_code_size; });
  mutate_elf_symbol_by_name(bytes, "lds_helper",
                            [&](Elf64_Sym &symbol) { symbol.st_size = sizeof(uint32_t); });
  append_kernel_metadata_note(bytes, "lds_probe", /*uses_dynamic_stack=*/true,
                              /*sgpr_count=*/106u);

  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_dispatch_id = 0x1122334455667788ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.max_patches = 1u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.final_validation_passed);
  const auto patch = std::ranges::find_if(result.patches, [&](const ConSanPatchInfo &candidate) {
    return candidate.branch_only_continuation && candidate.anchor_offset == access_offset;
  });
  ASSERT_NE(patch, result.patches.end()) << testing::PrintToString(result.patches);
  ASSERT_EQ(patch->branch_only_entry_relay_offsets.size(), 1u);
  ASSERT_EQ(patch->branch_only_return_relay_offsets.size(), 1u);
  const std::array<uint64_t, 3> relay_candidates = {
      kEntryRelayWord * sizeof(uint32_t),
      kUnusedEntryRelayWord * sizeof(uint32_t),
      (kSegmentWords + kReturnRelayWord) * sizeof(uint32_t),
  };
  EXPECT_NE(std::ranges::find(relay_candidates, patch->branch_only_entry_relay_offsets.front()),
            relay_candidates.end());
  EXPECT_NE(std::ranges::find(relay_candidates, patch->branch_only_return_relay_offsets.front()),
            relay_candidates.end());
  EXPECT_NE(patch->branch_only_entry_relay_offsets.front(),
            patch->branch_only_return_relay_offsets.front());
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineNopBranchRelay,
                               &ConSanPatchInfo::kind),
            2u);

  const auto unused_relay = std::ranges::find_if(relay_candidates, [&](uint64_t candidate) {
    return candidate != patch->branch_only_entry_relay_offsets.front() &&
           candidate != patch->branch_only_return_relay_offsets.front();
  });
  ASSERT_NE(unused_relay, relay_candidates.end());
  ConSanResult unused = result;
  ConSanPatchInfo unused_info;
  unused_info.kind = ConSanPatchKind::TrampolineNopBranchRelay;
  unused_info.anchor_offset = *unused_relay;
  unused_info.trampoline_offset = *unused_relay;
  unused_info.original_size = sizeof(uint32_t);
  unused_info.trampoline_size = sizeof(uint32_t);
  unused.patches.push_back(std::move(unused_info));
  const std::vector<std::string> unused_errors = validate_consan_modified_elf(bytes, unused);
  EXPECT_TRUE(std::ranges::any_of(unused_errors, [](const std::string &error) {
    return error.find("unused original-NOP relay") != std::string::npos;
  })) << testing::PrintToString(unused_errors);
}

ConSanResult run_rdna4_branch_only_instruction_reservoir(uint32_t filler,
                                                         std::string_view kernel_name) {
  constexpr size_t kTextWords = 45'000u;
  constexpr size_t kAccessWord = 7u;
  constexpr size_t kScalarUseWord = 40'000u;
  std::vector<uint32_t> text_words(kTextWords, filler);
  text_words[kAccessWord] = 0xD8340000u;
  text_words[kAccessWord + 1u] = 0x00000000u; // ds_store_b32 v0, v0
  size_t cursor = kScalarUseWord;
  for (uint16_t sgpr = 0u; sgpr < 105u; ++sgpr)
    text_words[cursor++] = build_s_mov_b32(105u, sgpr, ROCJITSU_CODE_ARCH_RDNA4);
  text_words[cursor] = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, kernel_name, kRdna4Wave64AllVgprsGranulated,
                                 /*wave32=*/false, /*uses_dynamic_stack=*/true);
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_dispatch_id = 0x1122334455667788ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.max_patches = 1u;
  return try_patch_consan(bytes, options);
}

TEST(ConSanMoi, Rdna4BranchOnlyDynamicStackRelocatesInstructionReservoirs) {
  const ConSanResult relocated = run_rdna4_branch_only_instruction_reservoir(
      build_s_mov_b32(/*sdst=*/105u, /*ssrc0=*/105u, ROCJITSU_CODE_ARCH_RDNA4),
      "branch_only_instruction_reservoir");
  ASSERT_TRUE(consan_patch_succeeded(relocated)) << testing::PrintToString(relocated.errors);
  ASSERT_TRUE(relocated.modified) << testing::PrintToString(relocated.warnings);
  ASSERT_TRUE(relocated.final_validation_passed);
  const auto branch_only =
      std::ranges::find(relocated.patches, true, &ConSanPatchInfo::branch_only_continuation);
  ASSERT_NE(branch_only, relocated.patches.end());
  const auto reservoir = std::ranges::find(
      relocated.patches, ConSanPatchKind::TrampolineBranchRelayReservoir, &ConSanPatchInfo::kind);
  ASSERT_NE(reservoir, relocated.patches.end());
  // Direct reservoirs contain at least the router's 16-word donor minimum.
  ASSERT_GE(reservoir->original_size, 16u * sizeof(uint32_t));
  const auto in_reservoir = [&](uint64_t relay) {
    return relay > reservoir->anchor_offset &&
           relay < reservoir->anchor_offset + reservoir->original_size;
  };
  EXPECT_TRUE(std::ranges::any_of(branch_only->branch_only_entry_relay_offsets, in_reservoir));
  EXPECT_TRUE(std::ranges::any_of(branch_only->branch_only_return_relay_offsets, in_reservoir));
  const ConSanBranchOnlyReservoirTelemetry &inventory =
      relocated.moi_branch_only_reservoir_telemetry;
  const size_t emitted_reservoir_count = std::ranges::count(
      relocated.patches, ConSanPatchKind::TrampolineBranchRelayReservoir, &ConSanPatchInfo::kind);
  EXPECT_GE(inventory.planned_reservoir_count, 1u);
  EXPECT_EQ(inventory.used_reservoir_count, emitted_reservoir_count);
  EXPECT_EQ(inventory.planned_reservoir_count,
            inventory.used_reservoir_count + inventory.unused_reservoir_count);
  EXPECT_EQ(inventory.planned_appended_bytes,
            inventory.used_appended_bytes + inventory.unused_appended_bytes);
  EXPECT_GT(inventory.used_appended_bytes, 0u);
}

TEST(ConSanMoi, Rdna4BranchOnlyDynamicStackFailsClosedWithoutAdmissibleReservoir) {
  // s_delay_alu is intentionally excluded from donor discovery, leaving no
  // instruction run that the router may safely relocate.
  const ConSanResult rejected = run_rdna4_branch_only_instruction_reservoir(
      build_s_delay_alu(kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4),
      "branch_only_without_admissible_reservoir");
  EXPECT_TRUE(rejected.errors.empty()) << testing::PrintToString(rejected.errors);
  EXPECT_EQ(std::ranges::find(rejected.patches, true, &ConSanPatchInfo::branch_only_continuation),
            rejected.patches.end());
  EXPECT_TRUE(std::ranges::any_of(rejected.warnings, [](const std::string &warning) {
    return warning.find("could not route its branch-only scalar-spill body") != std::string::npos;
  })) << testing::PrintToString(rejected.warnings);
}

TEST(ConSanMoi, Rdna4BranchOnlyDynamicStackRoutesThroughSelectedAnchorTails) {
  constexpr size_t kTextWords = 45'000u;
  constexpr size_t kEarlyAccessWord = 7u;
  constexpr size_t kMiddleAccessWord = 25'000u;
  constexpr size_t kLateAccessWord = 30'000u;
  constexpr size_t kScalarUseWord = 40'000u;
  const uint32_t filler = build_s_mov_b32(/*sdst=*/105u, /*ssrc0=*/105u, ROCJITSU_CODE_ARCH_RDNA4);
  std::vector<uint32_t> text_words(kTextWords, filler);
  const auto write_access = [&](size_t word) {
    text_words[word] = 0xD8340000u;
    text_words[word + 1u] = 0x00000000u; // ds_store_b32 v0, v0
  };
  write_access(kEarlyAccessWord);
  write_access(kMiddleAccessWord);
  write_access(kLateAccessWord);
  size_t cursor = kScalarUseWord;
  for (uint16_t sgpr = 0u; sgpr < 105u; ++sgpr)
    text_words[cursor++] = build_s_mov_b32(/*sdst=*/105u, sgpr, ROCJITSU_CODE_ARCH_RDNA4);
  text_words[cursor] = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);

  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "inline_branch_only_anchor_relays",
                                 kRdna4Wave64AllVgprsGranulated, /*wave32=*/false,
                                 /*uses_dynamic_stack=*/true);
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_dispatch_id = 0x1122334455667788ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.max_patches = 3u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.final_validation_passed);
  const auto branch_patch_at = [&](size_t word) {
    return std::ranges::find_if(result.patches, [&](const ConSanPatchInfo &patch) {
      return patch.branch_only_continuation && patch.anchor_offset == word * sizeof(uint32_t);
    });
  };
  const auto early = branch_patch_at(kEarlyAccessWord);
  const auto middle = branch_patch_at(kMiddleAccessWord);
  const auto late = branch_patch_at(kLateAccessWord);
  ASSERT_NE(early, result.patches.end()) << testing::PrintToString(result.warnings);
  ASSERT_NE(middle, result.patches.end()) << testing::PrintToString(result.warnings);
  ASSERT_NE(late, result.patches.end()) << testing::PrintToString(result.warnings);
  ASSERT_EQ(early->branch_only_entry_relay_offsets.size(), 1u);
  ASSERT_EQ(early->branch_only_return_relay_offsets.size(), 1u);
  EXPECT_TRUE(middle->branch_only_entry_relay_offsets.empty());
  EXPECT_TRUE(middle->branch_only_return_relay_offsets.empty());
  EXPECT_TRUE(late->branch_only_entry_relay_offsets.empty());
  EXPECT_TRUE(late->branch_only_return_relay_offsets.empty());
  std::array<uint64_t, 2> relays = {
      early->branch_only_entry_relay_offsets.front(),
      early->branch_only_return_relay_offsets.front(),
  };
  std::ranges::sort(relays);
  const std::array<uint64_t, 2> expected = {
      (kMiddleAccessWord + 1u) * sizeof(uint32_t),
      (kLateAccessWord + 1u) * sizeof(uint32_t),
  };
  EXPECT_EQ(relays, expected);
}

TEST(ConSanMoi, Gfx1250InlineOddShadowSlotCountClearsOnlyItsValidityState) {
  constexpr auto store = gfx1250::build_vds(gfx1250::kDsStoreB32Vds, {.addr = 0, .data0 = 0});
  const std::array<uint32_t, 3> text_words = {
      store[0],
      store[1],
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250),
  };
  std::vector<uint8_t> bytes = make_gfx1250_code_object(text_words, "odd_shadow_slot_count");
  mutate_first_kernel_descriptor(bytes,
                                 [](KD &descriptor) { descriptor.group_segment_fixed_size = 4u; });
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_inline_workgroup_shadow = true;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.resolved_moi_owner_vgpr);
  ASSERT_TRUE(result.resolved_moi_epoch_vgpr);
  const auto prologue = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue;
  });
  ASSERT_NE(prologue, result.patches.end());
  EXPECT_EQ(prologue->workgroup_shadow_size, 8u);
  EXPECT_EQ(prologue->workgroup_shadow_validity_base, 16u);
  EXPECT_EQ(prologue->workgroup_shadow_validity_size, 16u);
  EXPECT_TRUE(prologue->workgroup_shadow_lazy_initialization);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> prologue_words =
      text_words_at_offset(patched, prologue->trampoline_offset, prologue->trampoline_size);
  const auto store_pair =
      ib::build_ds_store_b64(*result.resolved_moi_owner_vgpr, *result.resolved_moi_epoch_vgpr, 0u,
                             ROCJITSU_CODE_ARCH_GFX1250);
  const auto store_quad =
      ib::build_ds_store_b128(*result.resolved_moi_owner_vgpr, *result.resolved_moi_epoch_vgpr, 0u,
                              ROCJITSU_CODE_ARCH_GFX1250);
  const auto scale_x = ib::build_v_lshlrev_b32(
      static_cast<uint16_t>(*result.resolved_moi_epoch_vgpr + 1u), scalar_positive_inline_u32(3u),
      /*workitem_id_x=*/0u, ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(store_pair);
  ASSERT_TRUE(store_quad);
  ASSERT_TRUE(scale_x);
  EXPECT_FALSE(contains_subsequence(prologue_words, *store_pair));
  EXPECT_TRUE(contains_subsequence(prologue_words, *store_quad));
  EXPECT_EQ(std::ranges::find(prologue_words, *scale_x), prologue_words.end());
}

TEST(ConSanMoi, Gfx1250InlineLargeLocalMirrorUsesFullExactCellsAndValidityState) {
  constexpr auto store = gfx1250::build_vds(gfx1250::kDsStoreB32Vds, {.addr = 0, .data0 = 0});
  const std::array<uint32_t, 3> text_words = {
      store[0],
      store[1],
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250),
  };
  std::vector<uint8_t> bytes = make_gfx1250_code_object(text_words, "large_local_mirror");
  mutate_first_kernel_descriptor(
      bytes, [](KD &descriptor) { descriptor.group_segment_fixed_size = 21120u; });
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_inline_workgroup_shadow = true;
  options.moi_dispatch_id_sgpr = 80u;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  });
  ASSERT_NE(access, result.patches.end());
  ASSERT_TRUE(access->scratch_vgpr);
  EXPECT_EQ(access->workgroup_shadow_base, 21120u);
  EXPECT_EQ(access->workgroup_shadow_size, 42240u);
  EXPECT_EQ(access->workgroup_shadow_validity_base, 63360u);
  EXPECT_EQ(access->workgroup_shadow_validity_size, 1328u);
  EXPECT_TRUE(access->workgroup_shadow_lazy_initialization);
  EXPECT_FALSE(access->workgroup_shadow_compact);
  EXPECT_EQ(access->workgroup_shadow_compact_token, 0u);
  EXPECT_EQ(access->required_group_segment_size, 64688u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> body =
      text_words_at_offset(patched, access->trampoline_offset, access->trampoline_size);
  const uint16_t scratch = *access->scratch_vgpr;
  const auto exchange = ib::build_ds_storexchg_rtn_b64(static_cast<uint16_t>(scratch + 5u), scratch,
                                                       static_cast<uint16_t>(scratch + 2u), 0u,
                                                       ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(exchange);
  EXPECT_TRUE(contains_subsequence(body, *exchange));
  const auto claim = ib::build_ds_or_rtn_b32(
      static_cast<uint16_t>(scratch + 5u), static_cast<uint16_t>(scratch + 12u),
      static_cast<uint16_t>(scratch + 8u), 0u, ROCJITSU_CODE_ARCH_GFX1250);
  const auto observe =
      ib::build_ds_load_b32(static_cast<uint16_t>(scratch + 5u),
                            static_cast<uint16_t>(scratch + 12u), 0u, ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(claim);
  ASSERT_TRUE(observe);
  EXPECT_TRUE(contains_subsequence(body, *claim));
  EXPECT_TRUE(contains_subsequence(body, *observe));

  const auto prologue = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue;
  });
  ASSERT_NE(prologue, result.patches.end());
  EXPECT_TRUE(prologue->workgroup_shadow_lazy_initialization);
  EXPECT_FALSE(prologue->workgroup_shadow_compact);
  EXPECT_EQ(prologue->workgroup_shadow_validity_base, 63360u);
  EXPECT_EQ(prologue->workgroup_shadow_validity_size, 1328u);
  EXPECT_TRUE(validate_consan_modified_elf(bytes, result).empty());
}

TEST(ConSanMoi, Gfx1250LargeFullLocalMirrorUsesDisjointTwoBitValidityState) {
  constexpr auto store = gfx1250::build_vds(gfx1250::kDsStoreB32Vds, {.addr = 0, .data0 = 0});
  const std::array<uint32_t, 3> text_words = {
      store[0],
      store[1],
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250),
  };
  std::vector<uint8_t> bytes = make_gfx1250_code_object(text_words, "compact_validity_state");
  mutate_first_kernel_descriptor(
      bytes, [](KD &descriptor) { descriptor.group_segment_fixed_size = 68u * 1024u; });
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_inline_workgroup_shadow = true;
  options.moi_max_workgroup_lds_bytes = 256u * 1024u;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto access = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiExactShadowStore, &ConSanPatchInfo::kind);
  ASSERT_NE(access, result.patches.end());
  ASSERT_TRUE(access->scratch_vgpr);
  EXPECT_FALSE(access->workgroup_shadow_compact);
  EXPECT_TRUE(access->workgroup_shadow_lazy_initialization);
  EXPECT_EQ(access->workgroup_shadow_validity_size, 4352u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> body =
      text_words_at_offset(patched, access->trampoline_offset, access->trampoline_size);
  const uint16_t scratch = *access->scratch_vgpr;
  const auto state_index =
      ib::build_v_and_b32_literal(static_cast<uint16_t>(scratch + 8u), 15u,
                                  static_cast<uint16_t>(scratch + 7u), ROCJITSU_CODE_ARCH_GFX1250);
  const auto scale_state_index =
      ib::build_v_lshlrev_b32(static_cast<uint16_t>(scratch + 8u), scalar_positive_inline_u32(1u),
                              static_cast<uint16_t>(scratch + 8u), ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(state_index);
  ASSERT_TRUE(scale_state_index);
  std::vector<uint32_t> two_bit_state(state_index->begin(), state_index->end());
  two_bit_state.push_back(*scale_state_index);
  EXPECT_TRUE(contains_subsequence(body, two_bit_state))
      << "adjacent gfx1250 shadow cells must use disjoint initializing/ready bit pairs";
}

TEST(ConSanMoi, Gfx1250FullExactShadowValidatesAtomicTokenWithWorkgroupKey) {
  constexpr auto store = gfx1250::build_vds(gfx1250::kDsStoreB32Vds, {.addr = 0, .data0 = 0});
  const auto release = build_gfx1250_flat_atomic_add_u32(
      /*vaddr=*/2, /*vsrc=*/1, /*vdst=*/0, /*return_old_value=*/false, /*scope=*/2,
      ROCJITSU_CODE_ARCH_GFX1250);
  const auto acquire = build_gfx1250_flat_atomic_add_u32(
      /*vaddr=*/4, /*vsrc=*/1, /*vdst=*/4, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(release && acquire);
  const std::array<uint32_t, 15> text_words = {
      store[0],      store[1],      0xEE0B0000u,
      0x00000000u,   0x00000000u,   (*release)[0],
      (*release)[1], (*release)[2], (*acquire)[0],
      (*acquire)[1], (*acquire)[2], 0xEE0AC000u,
      0x00000000u,   0x00000000u,   build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250),
  };
  std::vector<uint8_t> bytes = make_gfx1250_code_object(text_words, "compact_atomic_workgroup_key");
  mutate_first_kernel_descriptor(
      bytes, [](KD &descriptor) { descriptor.group_segment_fixed_size = 21120u; });
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_inline_workgroup_shadow = true;
  options.moi_track_atomics = true;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.resolved_moi_workgroup_key_vgpr);
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  });
  ASSERT_NE(access, result.patches.end());
  ASSERT_FALSE(access->workgroup_shadow_compact);
  ASSERT_TRUE(access->workgroup_shadow_lazy_initialization);
  ASSERT_EQ(access->workgroup_shadow_validity_size, 1328u);
  ASSERT_TRUE(access->scratch_vgpr);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> body =
      text_words_at_offset(patched, access->trampoline_offset, access->trampoline_size);
  bool copies_workgroup_key = false;
  for (uint16_t destination = *access->scratch_vgpr;
       destination < static_cast<uint16_t>(*access->scratch_vgpr + 25u); ++destination) {
    const uint32_t copy = build_v_mov_b32_e32(
        destination, vector_source_vgpr(*result.resolved_moi_workgroup_key_vgpr),
        ROCJITSU_CODE_ARCH_GFX1250);
    copies_workgroup_key |= std::ranges::find(body, copy) != body.end();
  }
  EXPECT_TRUE(copies_workgroup_key);
  ASSERT_TRUE(result.resolved_moi_exec_save_sgpr);
  const uint16_t exec_base = *result.resolved_moi_exec_save_sgpr;
  const uint16_t original_exec =
      static_cast<uint16_t>(exec_base + kConSanMoiInlineOriginalExecSaveOffset);
  const auto save_original =
      ib::build_s_mov_b64(original_exec, kRdna4ExecLo, ROCJITSU_CODE_ARCH_GFX1250);
  const auto restore_original =
      ib::build_s_mov_b64(kRdna4ExecLo, original_exec, ROCJITSU_CODE_ARCH_GFX1250);
  const auto authorize_stable_access_token = ib::build_v_cmp_gt_u32_vcc(
      scalar_positive_inline_u32(
          static_cast<uint32_t>(ConSanMoiInlineTokenEvidenceKind::ReleaseSequence)),
      static_cast<uint16_t>(*access->scratch_vgpr + 16u), ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(save_original && restore_original && authorize_stable_access_token);
  EXPECT_NE(std::ranges::find(body, *save_original), body.end());
  EXPECT_NE(std::ranges::find(body, *restore_original), body.end());
  EXPECT_NE(std::ranges::find(body, *authorize_stable_access_token), body.end());
}

TEST(ConSanMoi, Gfx1250FullLocalShadowValidatesAtomicTokenWithPersistentWorkgroupKey) {
  constexpr auto store = gfx1250::build_vds(gfx1250::kDsStoreB32Vds, {.addr = 0, .data0 = 0});
  const auto release = build_gfx1250_flat_atomic_add_u32(
      /*vaddr=*/2, /*vsrc=*/1, /*vdst=*/0, /*return_old_value=*/false, /*scope=*/2,
      ROCJITSU_CODE_ARCH_GFX1250);
  const auto acquire = build_gfx1250_flat_atomic_add_u32(
      /*vaddr=*/4, /*vsrc=*/1, /*vdst=*/4, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(release && acquire);
  const std::array<uint32_t, 15> text_words = {
      store[0],      store[1],      0xEE0B0000u,
      0x00000000u,   0x00000000u,   (*release)[0],
      (*release)[1], (*release)[2], (*acquire)[0],
      (*acquire)[1], (*acquire)[2], 0xEE0AC000u,
      0x00000000u,   0x00000000u,   build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250),
  };
  std::vector<uint8_t> bytes =
      make_gfx1250_code_object(text_words, "full_local_atomic_workgroup_key");
  mutate_first_kernel_descriptor(
      bytes, [](KD &descriptor) { descriptor.group_segment_fixed_size = 6144u; });
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_inline_workgroup_shadow = true;
  options.moi_track_atomics = true;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.resolved_moi_workgroup_key_vgpr);
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  });
  ASSERT_NE(access, result.patches.end());
  ASSERT_FALSE(access->workgroup_shadow_compact);
  ASSERT_GT(access->workgroup_shadow_size, 0u);
  ASSERT_TRUE(access->scratch_vgpr);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> body =
      text_words_at_offset(patched, access->trampoline_offset, access->trampoline_size);
  bool copies_workgroup_key = false;
  for (uint16_t destination = *access->scratch_vgpr;
       destination < static_cast<uint16_t>(*access->scratch_vgpr + 25u); ++destination) {
    const uint32_t copy = build_v_mov_b32_e32(
        destination, vector_source_vgpr(*result.resolved_moi_workgroup_key_vgpr),
        ROCJITSU_CODE_ARCH_GFX1250);
    copies_workgroup_key |= std::ranges::find(body, copy) != body.end();
  }
  EXPECT_TRUE(copies_workgroup_key);
}

TEST(ConSanMoi, Gfx1250FullExactShadowCoversEveryWideAccessCell) {
  constexpr auto store =
      gfx1250::build_vds(gfx1250::kDsStore2addrB64Vds,
                         {.offset0 = 0u, .offset1 = 1u, .addr = 0u, .data0 = 2u, .data1 = 4u});
  const std::array<uint32_t, 3> text_words = {
      store[0],
      store[1],
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250),
  };
  std::vector<uint8_t> bytes = make_gfx1250_code_object(text_words, "paired_compact_cells");
  mutate_first_kernel_descriptor(
      bytes, [](KD &descriptor) { descriptor.group_segment_fixed_size = 21120u; });
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_inline_workgroup_shadow = true;
  options.moi_dispatch_id_sgpr = 80u;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.moi_candidates.size(), 1u);
  EXPECT_EQ(result.moi_candidates.front().mnemonic, "ds_store_2addr_b64");
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  });
  ASSERT_NE(access, result.patches.end());
  ASSERT_TRUE(access->scratch_vgpr);
  EXPECT_FALSE(access->workgroup_shadow_compact);
  EXPECT_TRUE(access->workgroup_shadow_lazy_initialization);
  EXPECT_EQ(access->workgroup_shadow_validity_size, 1328u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> body =
      text_words_at_offset(patched, access->trampoline_offset, access->trampoline_size);
  const uint16_t scratch = *access->scratch_vgpr;
  const auto paired_exchange = ib::build_ds_storexchg_rtn_b64(
      static_cast<uint16_t>(scratch + 5u), scratch, static_cast<uint16_t>(scratch + 2u), 0u,
      ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(paired_exchange);
  EXPECT_EQ(count_subsequence(body, *paired_exchange), 2u)
      << "each disjoint b64 range should have an exact cell-exchange loop";
  EXPECT_TRUE(validate_consan_modified_elf(bytes, result).empty());
}

TEST(ConSanMoi, InlineShadowAutomaticallyAllocatesPersistentOwnerEpochVgprs) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_owner_source = ConSanMoiOwnerSource::Automatic;
  options.scratch_vgpr = 8;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << "candidates=" << result.moi_candidates.size()
                               << " plans=" << result.resource_plans.size()
                               << " patches=" << result.patches.size()
                               << " warnings=" << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.moi_persistent_vgprs_automatic);
  EXPECT_EQ(result.resolved_moi_owner_vgpr, 1);
  EXPECT_EQ(result.resolved_moi_epoch_vgpr, 2);
  EXPECT_EQ(result.resolved_moi_workgroup_key_vgpr, 3);
  ASSERT_EQ(result.patches.size(), 2u);
  const auto prologue = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue;
  });
  ASSERT_NE(prologue, result.patches.end());
  EXPECT_EQ(std::ranges::count_if(result.patches,
                                  [](const ConSanPatchInfo &patch) {
                                    return patch.kind ==
                                           ConSanPatchKind::TrampolineMoiExactShadowStore;
                                  }),
            1);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.kernels().size(), 1u);
  ASSERT_EQ(patched.text_sections().size(), 1u);
  KD descriptor{};
  std::memcpy(&descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  EXPECT_GE(AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc1,
                            kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT),
            4u);
  EXPECT_EQ(AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2,
                            kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X),
            1u);
  EXPECT_EQ(AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2,
                            kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y),
            1u);
  EXPECT_EQ(AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2,
                            kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z),
            1u);

  const std::vector<uint32_t> prologue_words =
      text_words_at_offset(patched, prologue->trampoline_offset, prologue->trampoline_size);
  ASSERT_TRUE(result.resolved_moi_owner_sgpr);
  const auto hwreg = build_hwreg_imm(/*reg_id=*/23, /*offset=*/0, /*size_bits=*/10);
  ASSERT_TRUE(hwreg);
  const auto owner_init =
      build_s_getreg_b32(*result.resolved_moi_owner_sgpr, *hwreg, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(owner_init);
  ASSERT_TRUE(prologue->dispatch_id_capture_sgpr);
  ASSERT_GE(prologue_words.size(), 11u);
  ASSERT_TRUE(prologue->entry_scalar_backup_vgpr);
  ASSERT_TRUE(prologue->entry_scalar_backup_sgpr_base);
  const auto entry_owner_backup = instrumentation::build_v_writelane_b32(
      *prologue->entry_scalar_backup_vgpr, *prologue->entry_scalar_backup_sgpr_base,
      /*lane=*/0u, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(entry_owner_backup);
  EXPECT_TRUE(std::ranges::equal(std::span(prologue_words).first<2>(), *entry_owner_backup));
  const std::array<uint32_t, 6> owner_sequence = {
      *owner_init,
      build_s_delay_alu(kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_add_u32(*result.resolved_moi_owner_sgpr, *result.resolved_moi_owner_sgpr,
                      scalar_positive_inline_u32(1), ROCJITSU_CODE_ARCH_RDNA4),
      build_s_delay_alu(kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4),
      *instrumentation::build_salu_to_valu_dependency_wait(ROCJITSU_CODE_ARCH_RDNA4),
      build_v_mov_b32_e32(1, *result.resolved_moi_owner_sgpr, ROCJITSU_CODE_ARCH_RDNA4),
  };
  EXPECT_NE(std::ranges::find(prologue_words, *owner_init), prologue_words.end());
  EXPECT_TRUE(contains_subsequence(prologue_words, owner_sequence));
  EXPECT_NE(std::ranges::find(prologue_words, build_v_mov_b32_e32(2, scalar_positive_inline_u32(0),
                                                                  ROCJITSU_CODE_ARCH_RDNA4)),
            prologue_words.end());
}

TEST(ConSanMoi, InlineShadowRejectsExplicitWorkitemOwnerSource) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_owner_source = ConSanMoiOwnerSource::WorkitemId;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  EXPECT_EQ(result.outcome, ConSanTransformOutcome::Invalid);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
  EXPECT_TRUE(result.patches.empty());
  ASSERT_EQ(result.errors.size(), 1u);
  EXPECT_NE(result.errors.front().find("requires resident-wave ownership"), std::string::npos);
}

TEST(ConSanMoi, InlineShadowEntryPrologueRelocatesCompleteScalarClause) {
  const std::array<uint32_t, 8> text_words = {
      0xBF850001u, // s_clause 1: the following two instructions
      0xF4006100u,
      0xF8000000u, // s_load_b256 s[4:11], s[0:1], 0
      0xF400A300u,
      0xF8000020u, // s_load_b96 s[12:14], s[0:1], 0x20
      0xD8340000u,
      0x00000100u, // ds_store_b32 v0, v1
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "clause_entry", kRdna4Wave64AllVgprsGranulated,
      /*wave32=*/false, /*uses_dynamic_stack=*/false,
      /*workgroup_id_dimension_mask=*/0, /*group_segment_fixed_size=*/768u);
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto prologue = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue;
  });
  ASSERT_NE(prologue, result.patches.end());
  constexpr size_t kClauseRunWords = 5u;
  EXPECT_EQ(prologue->original_size, kClauseRunWords * sizeof(uint32_t));

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> prologue_words =
      text_words_at_offset(patched, prologue->trampoline_offset, prologue->trampoline_size);
  const std::span<const uint32_t> clause_run(text_words.data(), kClauseRunWords);
  const auto relocated = std::search(prologue_words.begin(), prologue_words.end(),
                                     clause_run.begin(), clause_run.end());
  ASSERT_NE(relocated, prologue_words.end());
  ASSERT_NE(relocated + kClauseRunWords, prologue_words.end());
  EXPECT_EQ((*(relocated + kClauseRunWords)) & 0xffff0000u, 0xBFA00000u)
      << "the intact scalar clause must precede the return branch";

  ASSERT_EQ(patched.text_sections().size(), 1u);
  const uint32_t *entry =
      reinterpret_cast<const uint32_t *>(patched.text_sections().front()->data());
  EXPECT_EQ(entry[0] & 0xffff0000u, 0xBFA00000u);
  for (size_t index = 1; index < kClauseRunWords; ++index)
    EXPECT_EQ(entry[index], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(ConSanMoi, InlineShadowAutomaticallyAllocatesScratchAndPersistentVgprs) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.moi_persistent_vgprs_automatic);
  EXPECT_EQ(result.resolved_moi_owner_vgpr, 1);
  EXPECT_EQ(result.resolved_moi_epoch_vgpr, 2);
  EXPECT_EQ(result.resolved_moi_workgroup_key_vgpr, 3);
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  });
  ASSERT_NE(access, result.patches.end());
  EXPECT_EQ(access->scratch_vgpr, 4);
  EXPECT_EQ(access->spilled_vgpr_count, 0u);
}

TEST(ConSanMoi, InlineShadowGrowsPersistentVgprsInsteadOfReloadingHotOwnerState) {
  const std::array<uint32_t, 4> text_words = {
      build_v_mov_b32_e32(/*vdst=*/11, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4),
      0xD8340000u,
      0x00000000u, // ds_store_b32
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "persistent_owner_descriptor_growth", /*vgpr_granulated=*/2);
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.moi_persistent_vgprs_automatic);
  EXPECT_FALSE(result.moi_private_epoch_automatic);
  ASSERT_TRUE(result.resolved_moi_owner_vgpr);
  ASSERT_TRUE(result.resolved_moi_epoch_vgpr);
  EXPECT_GT(*result.resolved_moi_owner_vgpr, 11u);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue,
                               &ConSanPatchInfo::kind),
            1u);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue,
                               &ConSanPatchInfo::kind),
            0u);
}

TEST(ConSanMoi, InlineShadowGrowsPersistentVgprsForDynamicStackOwner) {
  const std::array<uint32_t, 4> text_words = {
      build_v_mov_b32_e32(/*vdst=*/11, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4),
      0xD8340000u,
      0x00000000u, // ds_store_b32
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "dynamic_stack_descriptor_growth", /*vgpr_granulated=*/2,
      /*wave32=*/false, /*uses_dynamic_stack=*/true);
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.moi_persistent_vgprs_automatic);
  EXPECT_FALSE(result.moi_private_epoch_automatic);
  ASSERT_TRUE(result.resolved_moi_owner_vgpr);
  ASSERT_TRUE(result.resolved_moi_epoch_vgpr);
  EXPECT_GT(*result.resolved_moi_owner_vgpr, 11u);
  EXPECT_EQ(*result.resolved_moi_epoch_vgpr, *result.resolved_moi_owner_vgpr + 1u);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue,
                               &ConSanPatchInfo::kind),
            1u);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiExactShadowStore,
                               &ConSanPatchInfo::kind),
            1u);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.kernels().size(), 1u);
  KD descriptor{};
  std::memcpy(&descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  EXPECT_GT(AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc1,
                            kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT),
            2u);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("descriptor growth because private epoch storage is incompatible") !=
           std::string::npos;
  }));
}

TEST(ConSanMoi, InlineShadowSpillsThroughSiteLocalDynamicStackFrame) {
  const std::array<uint32_t, 4> text_words = {
      build_v_mov_b32_e32(/*vdst=*/11, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4),
      0xD8340000u,
      0x00000000u, // ds_store_b32
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "dynamic_stack_inline_spill", kRdna4Wave64AllVgprsGranulated,
      /*wave32=*/false, /*uses_dynamic_stack=*/true);
  append_kernel_metadata_note(bytes, "dynamic_stack_inline_spill",
                              /*uses_dynamic_stack=*/true, /*sgpr_count=*/0u,
                              /*private_segment_fixed_size=*/0u);
  const std::vector<uint8_t> original_note = first_note_segment_bytes(bytes);
  ASSERT_FALSE(original_note.empty());
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.resolved_moi_exec_save_sgpr);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().source, ConSanRegisterAllocationSource::SpillRequired);
  const auto patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &item) {
    return item.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  });
  ASSERT_NE(patch, result.patches.end());
  EXPECT_GT(patch->spilled_vgpr_count, 0u);
  EXPECT_EQ(patch->required_private_segment_size, patch->spilled_vgpr_count * sizeof(uint32_t));
  EXPECT_EQ(patch->dynamic_private_segment_addend, patch->spilled_vgpr_count * sizeof(uint32_t));

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.kernels().size(), 1u);
  KD descriptor{};
  std::memcpy(&descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  EXPECT_EQ(descriptor.private_segment_fixed_size, patch->spilled_vgpr_count * sizeof(uint32_t));
  EXPECT_EQ(first_note_segment_bytes(result.elf_bytes), original_note)
      << "ROCR ignores the duplicated MessagePack private size, so instrumentation must not "
         "rewrite it";
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const std::vector<uint32_t> cave_words =
      text_words_at_offset(patched, patch->trampoline_offset, patch->trampoline_size);
  ASSERT_GE(cave_words.size(), 4u);
  const uint16_t saved_frame_sgpr =
      static_cast<uint16_t>(*result.resolved_moi_exec_save_sgpr + 24u);
  EXPECT_EQ(cave_words[2],
            build_s_mov_b32(saved_frame_sgpr, /*frame base=*/33, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(cave_words[3],
            build_s_mov_b32(/*frame base=*/33, /*stack top=*/32, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_NE(std::find(cave_words.begin(), cave_words.end(), 0xed068021u), cave_words.end());
  EXPECT_NE(std::find(cave_words.begin(), cave_words.end(), 0xed050021u), cave_words.end());
}

TEST(ConSanMoi, Cdna4InlineShadowSpillsThroughSiteLocalDynamicStackFrame) {
  const auto guest = build_cdna4_ds_store_b32(
      /*vaddr=*/2, /*vdata=*/3, /*byte_offset=*/0, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(guest);
  std::vector<uint32_t> text_words(1200, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  // Keep a relocatable scalar instruction at the hardware entry. Real
  // dynamic-stack kernels have an ABI setup prefix here; the owner/epoch
  // prologue displaces that instruction before reaching the first LDS site.
  std::copy(guest->begin(), guest->end(), text_words.begin() + 1u);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);
  constexpr uint32_t kCdna4Wave64AllVgprsGranulated = 31u;
  std::vector<uint8_t> bytes = make_cdna4_lds_code_object(
      text_words, "dynamic_stack_inline_spill", kCdna4Wave64AllVgprsGranulated,
      /*uses_dynamic_stack=*/true, /*workgroup_id_dimension_mask=*/0u,
      /*group_segment_fixed_size=*/4u);
  append_kernel_metadata_note(bytes, "dynamic_stack_inline_spill",
                              /*uses_dynamic_stack=*/true, /*sgpr_count=*/0u,
                              /*private_segment_fixed_size=*/0u);
  const std::vector<uint8_t> original_note = first_note_segment_bytes(bytes);
  ASSERT_FALSE(original_note.empty());
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << "warnings=" << testing::PrintToString(result.warnings)
                               << " errors=" << testing::PrintToString(result.errors);
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_TRUE(result.final_validation_passed);
  ASSERT_TRUE(result.resolved_moi_exec_save_sgpr);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().source, ConSanRegisterAllocationSource::SpillRequired);
  const auto patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &item) {
    return item.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  });
  ASSERT_NE(patch, result.patches.end());
  ASSERT_TRUE(patch->scratch_vgpr);
  EXPECT_EQ(*patch->scratch_vgpr, 8u);
  EXPECT_EQ(patch->spilled_vgpr_count, 16u);
  EXPECT_EQ(patch->required_private_segment_size, 64u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.kernels().size(), 1u);
  KD descriptor{};
  std::memcpy(&descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  EXPECT_EQ(descriptor.private_segment_fixed_size, 64u);
  EXPECT_EQ(first_note_segment_bytes(result.elf_bytes), original_note);
  const std::vector<uint32_t> cave_words =
      text_words_at_offset(patched, patch->trampoline_offset, patch->trampoline_size);
  const uint16_t saved_frame_sgpr =
      static_cast<uint16_t>(*result.resolved_moi_exec_save_sgpr + 24u);
  const auto store = build_cdna4_scratch_store_b32_saddr(
      *patch->scratch_vgpr, /*frame base=*/33, /*byte_offset=*/0, ROCJITSU_CODE_ARCH_CDNA4);
  const auto load = build_cdna4_scratch_load_b32_saddr(*patch->scratch_vgpr, /*frame base=*/33,
                                                       /*byte_offset=*/0, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(store && load);
  EXPECT_TRUE(contains_subsequence(cave_words, *store));
  EXPECT_TRUE(contains_subsequence(cave_words, *load));
  const uint16_t address_snapshot = static_cast<uint16_t>(*patch->scratch_vgpr + 15u);
  EXPECT_EQ(
      std::ranges::find(cave_words, build_v_mov_b32_e32(address_snapshot, vector_source_vgpr(2u),
                                                        ROCJITSU_CODE_ARCH_CDNA4)),
      cave_words.end());
  EXPECT_NE(std::find(cave_words.begin(), cave_words.end(), 0xbf8c0f70u), cave_words.end());
  EXPECT_NE(
      std::find(cave_words.begin(), cave_words.end(),
                build_s_mov_b32(saved_frame_sgpr, /*frame base=*/33, ROCJITSU_CODE_ARCH_CDNA4)),
      cave_words.end());
}

TEST(ConSanMoi, Cdna4InlineScalarPersistencePlansEntryScratchForEveryComponent) {
  const auto guest = build_cdna4_ds_store_b32(
      /*vaddr=*/0, /*vdata=*/0, /*byte_offset=*/0, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(guest);
  std::vector<uint32_t> probe_words(320u, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  probe_words[1] =
      build_v_mov_b32_e32(/*vdst=*/0, vector_source_vgpr(255), ROCJITSU_CODE_ARCH_CDNA4);
  std::copy(guest->begin(), guest->end(), probe_words.begin() + 2u);
  probe_words[8] = 0xBE802A02u; // s_movrels_b32 s0, s2
  probe_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);
  std::vector<uint32_t> helper_words(320u, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  std::copy(guest->begin(), guest->end(), helper_words.begin() + 1u);
  helper_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);

  std::vector<uint8_t> bytes = make_rdna4_code_object_with_local_function(
      probe_words, helper_words, {}, kRdna4Wave64AllVgprsGranulated,
      /*function_is_kernel=*/true, /*wave32=*/false);
  mutate_elf_header(bytes,
                    [](Elf64_Ehdr &header) { header.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX950; });
  mutate_kernel_descriptor(bytes, "lds_probe", [](KD &descriptor) {
    // v256 is the first accumulator register, while the guest reaches v255.
    // This component therefore needs scalar persistent state without moving
    // the compiler's ordinary/accumulator boundary.
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 63u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 4u);
  });
  mutate_kernel_descriptor(bytes, "lds_helper", [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 4u);
  });
  append_kernel_metadata_note(bytes, "lds_probe", /*uses_dynamic_stack=*/true,
                              /*sgpr_count=*/0u);

  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.force_vgpr_spill = true;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result))
      << "warnings=" << testing::PrintToString(result.warnings)
      << " errors=" << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << "warnings=" << testing::PrintToString(result.warnings)
                               << " errors=" << testing::PrintToString(result.errors);
  EXPECT_TRUE(result.moi_persistent_sgprs_automatic);
  EXPECT_FALSE(result.moi_persistent_vgprs_automatic);
  EXPECT_FALSE(result.moi_private_epoch_automatic);
  EXPECT_TRUE(result.resolved_moi_persistent_vgpr_assignments.empty());
  ASSERT_TRUE(result.resolved_moi_persistent_owner_sgpr);
  EXPECT_GE(*result.resolved_moi_persistent_owner_sgpr, 40u);
  EXPECT_EQ(result.resolved_moi_prologue_scratch_vgpr_assignments.size(), 2u);
  // The full-bank owner drives the code-object-wide scalar choice but its
  // forced-spill access is filtered during the rebuilt resource plan.  The
  // other component still emits, and planning must retain an entry assignment
  // for both so a later selection change cannot expose a partial scalar mode.
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiExactShadowStore,
                               &ConSanPatchInfo::kind),
            1u);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue,
                               &ConSanPatchInfo::kind),
            1u);
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, InlineShadowSpillingWorksWithoutMetadata) {
  const std::array<uint32_t, 4> text_words = {
      build_v_mov_b32_e32(/*vdst=*/11, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4),
      0xD8340000u,
      0x00000000u, // ds_store_b32
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "dynamic_stack_without_metadata_note", kRdna4Wave64AllVgprsGranulated,
      /*wave32=*/false, /*uses_dynamic_stack=*/true);
  // The malformed-note counterpart is rejected in
  // ConSan.RejectsCodeObjectWithMalformedKernelMetadataNote.
  ASSERT_TRUE(first_note_segment_bytes(bytes).empty());
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &item) {
    return item.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  });
  ASSERT_NE(patch, result.patches.end());
  EXPECT_GT(patch->spilled_vgpr_count, 0u);
}

TEST(ConSanMoi, InlineShadowAutomaticallyAllocatesHwIdOwnerAndSpecialStateSgprs) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_owner_source = ConSanMoiOwnerSource::HwId;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified);
  EXPECT_TRUE(result.moi_owner_sgpr_automatic);
  EXPECT_TRUE(result.moi_exec_save_sgprs_automatic);
  EXPECT_TRUE(result.moi_dispatch_id_sgprs_automatic);
  EXPECT_EQ(result.resolved_moi_owner_sgpr, result.resolved_moi_exec_save_sgpr);
  EXPECT_EQ(result.resolved_moi_dispatch_id_sgpr, 20);
  EXPECT_EQ(result.resolved_moi_exec_save_sgpr, 22);

  const auto prologue = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue;
  });
  ASSERT_NE(prologue, result.patches.end());
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const std::vector<uint32_t> prologue_words =
      text_words_at_offset(patched, prologue->trampoline_offset, prologue->trampoline_size);
  const auto hwreg = build_hwreg_imm(/*reg_id=*/23, /*offset=*/0, /*size_bits=*/10);
  ASSERT_TRUE(hwreg);
  ASSERT_TRUE(result.resolved_moi_owner_sgpr);
  const auto get_hw_id =
      build_s_getreg_b32(*result.resolved_moi_owner_sgpr, *hwreg, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(get_hw_id);
  EXPECT_TRUE(std::find(prologue_words.begin(), prologue_words.end(), *get_hw_id) !=
              prologue_words.end());

  ASSERT_EQ(patched.kernels().size(), 1u);
  KD descriptor{};
  std::memcpy(&descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  const uint32_t sgpr_granulated = AMDHSA_BITS_GET(
      descriptor.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT);
  // RDNA's zero field already denotes the complete fixed per-wave SGPR pool.
  EXPECT_EQ(sgpr_granulated, 0u);
}

TEST(ConSanMoi, InlineShadowPrivateEpochUsesDimensionIndependentResidentWaveOwner) {
  const std::array<uint32_t, 3> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "wave32_private_epoch", kRdna4Wave64AllVgprsGranulated, /*wave32=*/true);
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.force_private_epoch = true;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified);
  ASSERT_TRUE(result.moi_private_epoch_automatic);
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  });
  ASSERT_NE(access, result.patches.end());
  ASSERT_EQ(access->scratch_vgpr, 1);
  EXPECT_FALSE(access->persistent_owner_private_offset);
  ASSERT_TRUE(result.resolved_moi_owner_sgpr);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const std::vector<uint32_t> words =
      text_words_at_offset(patched, access->trampoline_offset, access->trampoline_size);
  const auto hwreg = build_hwreg_imm(/*reg_id=*/23, /*offset=*/0, /*size_bits=*/10);
  const auto get_hw_id =
      hwreg ? build_s_getreg_b32(*result.resolved_moi_owner_sgpr, *hwreg, ROCJITSU_CODE_ARCH_RDNA4)
            : std::nullopt;
  const auto save_owner = instrumentation::build_v_writelane_b32(
      /*owner backup=*/4u, *result.resolved_moi_owner_sgpr, 0u, ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_owner = instrumentation::build_v_readlane_b32(
      *result.resolved_moi_owner_sgpr, /*owner backup=*/4u, 0u, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(get_hw_id);
  ASSERT_TRUE(save_owner);
  ASSERT_TRUE(restore_owner);
  std::vector<uint32_t> expected_owner;
  expected_owner.insert(expected_owner.end(), save_owner->begin(), save_owner->end());
  expected_owner.push_back(*get_hw_id);
  expected_owner.push_back(build_s_delay_alu(kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4));
  expected_owner.push_back(
      build_s_add_u32(*result.resolved_moi_owner_sgpr, *result.resolved_moi_owner_sgpr,
                      scalar_positive_inline_u32(1), ROCJITSU_CODE_ARCH_RDNA4));
  expected_owner.push_back(build_s_delay_alu(kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4));
  expected_owner.push_back(
      *instrumentation::build_salu_to_valu_dependency_wait(ROCJITSU_CODE_ARCH_RDNA4));
  expected_owner.push_back(build_v_mov_b32_e32(/*owner field=*/5u, *result.resolved_moi_owner_sgpr,
                                               ROCJITSU_CODE_ARCH_RDNA4));
  expected_owner.insert(expected_owner.end(), restore_owner->begin(), restore_owner->end());
  expected_owner.push_back(
      *instrumentation::build_valu_to_salu_dependency_wait(ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_TRUE(contains_subsequence(words, expected_owner));
}

TEST(ConSanMoi, InlineShadowPrivateHwIdOwnerBiasesBeforeProbeValuRead) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.force_private_epoch = true;
  options.moi_owner_source = ConSanMoiOwnerSource::HwId;
  options.moi_owner_sgpr = 20u;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto access = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiExactShadowStore, &ConSanPatchInfo::kind);
  ASSERT_NE(access, result.patches.end());
  ASSERT_TRUE(access->scratch_vgpr);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const auto hwreg = build_hwreg_imm(/*reg_id=*/23, /*offset=*/0, /*size_bits=*/10);
  const auto get_hw_id =
      hwreg ? build_s_getreg_b32(*options.moi_owner_sgpr, *hwreg, ROCJITSU_CODE_ARCH_RDNA4)
            : std::nullopt;
  ASSERT_TRUE(get_hw_id);
  const std::array<uint32_t, 6> expected_owner = {
      *get_hw_id,
      build_s_delay_alu(kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_add_u32(*options.moi_owner_sgpr, *options.moi_owner_sgpr,
                      scalar_positive_inline_u32(1), ROCJITSU_CODE_ARCH_RDNA4),
      build_s_delay_alu(kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4),
      *instrumentation::build_salu_to_valu_dependency_wait(ROCJITSU_CODE_ARCH_RDNA4),
      build_v_mov_b32_e32(static_cast<uint16_t>(*access->scratch_vgpr + 4u),
                          *options.moi_owner_sgpr, ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint32_t> access_words =
      text_words_at_offset(patched, access->trampoline_offset, access->trampoline_size);
  EXPECT_TRUE(contains_subsequence(access_words, expected_owner));
}

TEST(ConSanMoi, Rdna4InlinePrivateEpochUsesInitializedLocalMirror) {
  const std::array<uint32_t, 3> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "private_epoch_workgroup_shadow",
                                 kRdna4Wave64AllVgprsGranulated, false, false, 0, 4352u);
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_inline_workgroup_shadow = true;
  options.force_private_epoch = true;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.moi_private_epoch_automatic);
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  });
  const auto prologue = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue;
  });
  ASSERT_NE(access, result.patches.end());
  ASSERT_NE(prologue, result.patches.end());
  ASSERT_TRUE(prologue->scratch_vgpr);
  EXPECT_EQ(access->persistent_workgroup_key_private_offset, 4u);
  EXPECT_EQ(prologue->persistent_workgroup_key_private_offset, 4u);
  EXPECT_EQ(access->persistent_private_state_end, 8u);
  EXPECT_EQ(prologue->persistent_private_state_end, 8u);
  EXPECT_EQ(prologue->spilled_vgpr_count, 3u);
  EXPECT_EQ(access->workgroup_shadow_base, 4352u);
  EXPECT_EQ(access->workgroup_shadow_size, 8704u);
  EXPECT_EQ(access->workgroup_shadow_validity_base, 13056u);
  EXPECT_EQ(access->workgroup_shadow_validity_size, 272u);
  EXPECT_EQ(access->required_group_segment_size, 13328u);
  EXPECT_TRUE(access->workgroup_shadow_lazy_initialization);
  EXPECT_FALSE(access->workgroup_shadow_compact);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const auto *text = patched.text_sections().front();
  ASSERT_LE(prologue->trampoline_offset + prologue->trampoline_size, text->size());
  std::vector<uint32_t> words(prologue->trampoline_size / sizeof(uint32_t));
  std::memcpy(words.data(), text->data() + prologue->trampoline_offset, prologue->trampoline_size);

  const uint16_t scratch = *prologue->scratch_vgpr;
  const auto extract_x =
      ib::build_v_and_b32_literal(static_cast<uint16_t>(scratch + 2u), 0x3ffu,
                                  /*packed_workitem_id=*/0u, ROCJITSU_CODE_ARCH_RDNA4);
  const auto shadow_offset =
      ib::build_v_lshlrev_b32(static_cast<uint16_t>(scratch + 2u), scalar_positive_inline_u32(3u),
                              static_cast<uint16_t>(scratch + 2u), ROCJITSU_CODE_ARCH_RDNA4);
  const auto shadow_base = ib::build_v_add_u32_literal(
      scratch, 13056u, static_cast<uint16_t>(scratch + 2u), ROCJITSU_CODE_ARCH_RDNA4);
  const auto store_wide = build_ds_store_b64(scratch, static_cast<uint16_t>(scratch + 1u),
                                             /*byte_offset=*/0, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(extract_x);
  ASSERT_TRUE(shadow_offset);
  ASSERT_TRUE(shadow_base);
  ASSERT_TRUE(store_wide);
  EXPECT_TRUE(contains_subsequence(words, *extract_x));
  EXPECT_NE(std::ranges::find(words, *shadow_offset), words.end());
  EXPECT_TRUE(contains_subsequence(words, *shadow_base));
  EXPECT_TRUE(contains_subsequence(words, *store_wide));
  EXPECT_EQ(
      std::count(words.begin(), words.end(), *build_s_barrier_signal_all(ROCJITSU_CODE_ARCH_RDNA4)),
      1);
  EXPECT_EQ(
      std::count(words.begin(), words.end(), *build_s_barrier_wait_all(ROCJITSU_CODE_ARCH_RDNA4)),
      1);
}

TEST(ConSanMoi, InlineShadowDescriptorFullUsesPrivateEpochWithoutSpillOverlap) {
  constexpr uint32_t kBarrierWait = 0xBF940000u;
  const std::array<uint32_t, 5> text_words = {
      build_v_mov_b32_e32(/*vdst=*/255, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4),
      0xD8340000u,
      0x00000000u, // ds_store_b32
      kBarrierWait,
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "descriptor_full_private_epoch", kRdna4Wave64AllVgprsGranulated);
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_track_barriers = true;
  options.force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.moi_private_epoch_automatic);
  EXPECT_FALSE(result.moi_persistent_vgprs_automatic);
  EXPECT_FALSE(result.resolved_moi_owner_vgpr);
  EXPECT_FALSE(result.resolved_moi_epoch_vgpr);

  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  });
  const auto barrier = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiInlineEpochBarrier;
  });
  const auto prologue = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue;
  });
  ASSERT_NE(access, result.patches.end());
  ASSERT_NE(barrier, result.patches.end());
  ASSERT_NE(prologue, result.patches.end());
  ASSERT_TRUE(access->scratch_vgpr);
  EXPECT_EQ(access->scratch_vgpr, 1);
  EXPECT_EQ(access->spilled_vgpr_count, 16u);
  EXPECT_EQ(access->persistent_epoch_private_offset, 0u);
  EXPECT_EQ(access->required_private_segment_size, 80u);
  EXPECT_EQ(barrier->scratch_vgpr, access->scratch_vgpr);
  EXPECT_EQ(barrier->spilled_vgpr_count, 1u);
  EXPECT_EQ(barrier->persistent_epoch_private_offset, 0u);
  EXPECT_EQ(barrier->required_private_segment_size, 80u);
  EXPECT_EQ(prologue->scratch_vgpr, access->scratch_vgpr);
  EXPECT_EQ(prologue->spilled_vgpr_count, 3u);
  EXPECT_EQ(prologue->persistent_epoch_private_offset, 0u);
  EXPECT_EQ(prologue->required_private_segment_size, 80u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.kernels().size(), 1u);
  ASSERT_EQ(patched.text_sections().size(), 1u);
  KD descriptor{};
  std::memcpy(&descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  EXPECT_EQ(descriptor.private_segment_fixed_size, 80u);
  EXPECT_EQ(
      AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT),
      1u);

  const auto patch_words = [&](const ConSanPatchInfo &patch) {
    const std::vector<uint32_t> words =
        text_words_at_offset(patched, patch.trampoline_offset, patch.trampoline_size);
    return words;
  };
  const std::vector<uint32_t> access_words = patch_words(*access);
  const std::vector<uint32_t> barrier_words = patch_words(*barrier);
  const std::vector<uint32_t> prologue_words = patch_words(*prologue);
  const auto private_increment = build_v_add_nc_u32_e32(
      /*vdst=*/1, scalar_positive_inline_u32(1), /*vsrc1=*/1, ROCJITSU_CODE_ARCH_RDNA4);
  const auto private_saturate = build_v_min_u32_e32_literal(
      /*vdst=*/1, consan_moi_exact_shadow::max_epoch, /*vsrc1=*/1, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(private_increment);
  ASSERT_TRUE(private_saturate);
  std::vector<uint32_t> expected_private_saturating_add = {*private_increment};
  expected_private_saturating_add.insert(expected_private_saturating_add.end(),
                                         private_saturate->begin(), private_saturate->end());
  EXPECT_TRUE(contains_subsequence(barrier_words, expected_private_saturating_add));
  const auto epoch_load = build_address_free_scratch_load_b32(
      /*vdst=*/5, /*byte_offset=*/0, ROCJITSU_CODE_ARCH_RDNA4);
  const auto epoch_barrier_load = build_address_free_scratch_load_b32(
      /*vdst=*/1, /*byte_offset=*/0, ROCJITSU_CODE_ARCH_RDNA4);
  const auto epoch_store = build_address_free_scratch_store_b32(
      /*vsrc=*/1, /*byte_offset=*/0, ROCJITSU_CODE_ARCH_RDNA4);
  const auto spill_store = build_address_free_scratch_store_b32(
      /*vsrc=*/1, /*byte_offset=*/16, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(epoch_load);
  ASSERT_TRUE(epoch_barrier_load);
  ASSERT_TRUE(epoch_store);
  ASSERT_TRUE(spill_store);
  EXPECT_TRUE(contains_subsequence(access_words, *spill_store));
  EXPECT_TRUE(contains_subsequence(access_words, *epoch_load));
  EXPECT_TRUE(contains_subsequence(barrier_words, *spill_store));
  EXPECT_TRUE(contains_subsequence(barrier_words, *epoch_barrier_load));
  EXPECT_TRUE(contains_subsequence(barrier_words, *epoch_store));
  EXPECT_TRUE(contains_subsequence(prologue_words, *spill_store));
  EXPECT_TRUE(contains_subsequence(prologue_words, *epoch_store));
}

TEST(ConSanMoi, InlineShadowProbePublishesMultiCellNativeLdsStore) {
  const std::array<uint32_t, 3> input_words = {
      0xDB7C0000u,
      0x00000109u, // ds_store_b128 v0, v[1:4]
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(input_words);
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.scratch_vgpr = 16;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified) << "errors=" << result.errors.size()
                               << " warnings=" << result.warnings.size()
                               << " candidates=" << result.moi_candidates.size()
                               << " warnings=" << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.moi_candidates.size(), 1u);
  EXPECT_EQ(result.moi_candidates.front().width_bits, 128u);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::TrampolineMoiExactShadowStore);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const auto *text_section = patched.text_sections().front();
  ASSERT_EQ(text_section->size() % sizeof(uint32_t), 0u);
  std::vector<uint32_t> text_words(text_section->size() / sizeof(uint32_t));
  std::memcpy(text_words.data(), text_section->data(), text_section->size());

  const auto atomic_swap = build_flat_atomic_swap_b64_vaddr_vsrc_vdst(
      /*vaddr=*/16, /*vsrc=*/18, /*vdst=*/21, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto version_cas = build_flat_atomic_cmpswap_b32_vaddr_vsrc_vdst(
      /*vaddr=*/16, /*vsrc=*/30, /*vdst=*/30, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(atomic_swap);
  ASSERT_TRUE(version_cas);
  EXPECT_EQ(count_subsequence(text_words, *atomic_swap), 0u);
  // The wide-cell loop has one static uniform/lane-wise publication body,
  // with odd claim and even commit in each path. Exactly one path executes per
  // cell at runtime.
  EXPECT_EQ(count_subsequence(text_words, *version_cas), 4u);

  const auto scale_cell = instrumentation::build_v_mul_lo_u32_literal(
      /*vdst=*/20, /*sdst_unused=*/21, sizeof(ConSanMoiInlineExactShadowSlot), /*vsrc0=*/32,
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto add_cell = build_v_add_u64_vgpr_offset(/*address_vgpr=*/16, /*offset_vgpr=*/20,
                                                    ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(scale_cell);
  ASSERT_TRUE(add_cell);
  EXPECT_TRUE(contains_subsequence(text_words, *scale_cell));
  EXPECT_TRUE(contains_subsequence(text_words, *add_cell));
}

TEST(ConSanMoi, InlineShadowProbeCoversNativeWidthAndTwoAddressFamilies) {
  const auto expect_cell_publications =
      [](uint32_t word0, uint32_t word1, std::string_view expected_mnemonic,
         uint32_t expected_width_bits, uint32_t expected_static_publication_sites) {
        const std::array<uint32_t, 3> input_words = {
            word0,
            word1,
            build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
        };
        const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(input_words);
        ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
        options.scratch_vgpr = 16;
        options.moi_owner_vgpr = 40;
        options.moi_epoch_vgpr = 41;
        options.moi_report_buffer_address = 0x100000000ull;
        options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

        const auto result = try_patch_consan(bytes, options);

        ASSERT_TRUE(consan_patch_succeeded(result));
        ASSERT_TRUE(result.modified);
        ASSERT_EQ(result.moi_candidates.size(), 1u);
        EXPECT_EQ(result.moi_candidates.front().mnemonic, expected_mnemonic);
        EXPECT_EQ(result.moi_candidates.front().width_bits, expected_width_bits);

        AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
        ASSERT_TRUE(patched.is_valid());
        ASSERT_EQ(patched.text_sections().size(), 1u);
        const auto *text_section = patched.text_sections().front();
        ASSERT_EQ(text_section->size() % sizeof(uint32_t), 0u);
        std::vector<uint32_t> text_words(text_section->size() / sizeof(uint32_t));
        std::memcpy(text_words.data(), text_section->data(), text_section->size());

        const auto version_cas = build_flat_atomic_cmpswap_b32_vaddr_vsrc_vdst(
            /*vaddr=*/16, /*vsrc=*/30, /*vdst=*/30, /*return_old_value=*/true, /*scope=*/2,
            ROCJITSU_CODE_ARCH_RDNA4);
        ASSERT_TRUE(version_cas);
        const uint32_t unrolled_cell_count =
            expected_width_bits <= consan_moi_exact_shadow::granule_bytes * 8u
                ? consan_moi_maximum_cell_count_for_unaligned_bytes(expected_width_bits / 8u)
                : 1u;
        EXPECT_EQ(count_subsequence(text_words, *version_cas),
                  4u * expected_static_publication_sites * unrolled_cell_count);
      };

  expect_cell_publications(0xD9D80000u, 0x01000009u, "ds_load_b64", 64u, 1u);
  expect_cell_publications(0xDA980000u, 0x01000002u, "ds_load_u16_d16", 16u, 1u);
  expect_cell_publications(0xD8380201u, 0x00000000u, "ds_store_2addr_b32", 32u, 2u);
  expect_cell_publications(0xD9DC0201u, 0x01000009u, "ds_load_2addr_b64", 64u, 2u);
  constexpr auto load_stride64_b32 = rdna4::build_vds(
      rdna4::kDsLoad2addrStride64B32Vds, {.offset0 = 0, .offset1 = 1, .addr = 0, .vdst = 1});
  constexpr auto store_stride64_b32 =
      rdna4::build_vds(rdna4::kDsStore2addrStride64B32Vds,
                       {.offset0 = 0, .offset1 = 1, .addr = 0, .data0 = 1, .data1 = 2});
  constexpr auto load_stride64_b64 = rdna4::build_vds(
      rdna4::kDsLoad2addrStride64B64Vds, {.offset0 = 0, .offset1 = 1, .addr = 0, .vdst = 1});
  constexpr auto store_stride64_b64 =
      rdna4::build_vds(rdna4::kDsStore2addrStride64B64Vds,
                       {.offset0 = 0, .offset1 = 1, .addr = 0, .data0 = 1, .data1 = 3});
  expect_cell_publications(load_stride64_b32[0], load_stride64_b32[1], "ds_load_2addr_stride64_b32",
                           32u, 2u);
  expect_cell_publications(store_stride64_b32[0], store_stride64_b32[1],
                           "ds_store_2addr_stride64_b32", 32u, 2u);
  expect_cell_publications(load_stride64_b64[0], load_stride64_b64[1], "ds_load_2addr_stride64_b64",
                           64u, 2u);
  expect_cell_publications(store_stride64_b64[0], store_stride64_b64[1],
                           "ds_store_2addr_stride64_b64", 64u, 2u);
}

TEST(ConSanMoi, InlineShadowProbeCanEmitGpuConflictDiagnostic) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 24;
  options.moi_epoch_vgpr = 25;
  options.moi_exec_save_sgpr = 30;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::TrampolineMoiExactShadowStore);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const auto *text_section = patched.text_sections().front();
  ASSERT_EQ(text_section->size() % sizeof(uint32_t), 0u);
  std::vector<uint32_t> text_words(text_section->size() / sizeof(uint32_t));
  std::memcpy(text_words.data(), text_section->data(), text_section->size());

  const uint64_t report_base = *options.moi_report_buffer_address;

  // The exact-shadow update retains incoming and pending masks, partitions the
  // latter by address, and then proves metadata uniformity within that group.
  // Nonuniform metadata takes the lane-wise fallback for that address only.
  const auto save_incoming_exec =
      build_s_mov_b64(/*sdst=*/42, kRdna4ExecLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto initialize_pending =
      build_s_mov_b64(/*sdst=*/44, kRdna4ExecLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto select_pending = build_s_mov_b64(kRdna4ExecLo, /*ssrc0=*/44, ROCJITSU_CODE_ARCH_RDNA4);
  const auto save_address_lo =
      build_v_mov_b32_e32(/*vdst=*/15, vector_source_vgpr(8), ROCJITSU_CODE_ARCH_RDNA4);
  const auto save_address_hi =
      build_v_mov_b32_e32(/*vdst=*/16, vector_source_vgpr(9), ROCJITSU_CODE_ARCH_RDNA4);
  const auto save_current_low =
      build_v_mov_b32_e32(/*vdst=*/17, vector_source_vgpr(10), ROCJITSU_CODE_ARCH_RDNA4);
  const auto save_current_high =
      build_v_mov_b32_e32(/*vdst=*/18, vector_source_vgpr(11), ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_address_lo =
      build_v_mov_b32_e32(/*vdst=*/8, vector_source_vgpr(15), ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_address_hi =
      build_v_mov_b32_e32(/*vdst=*/9, vector_source_vgpr(16), ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_current_low =
      build_v_mov_b32_e32(/*vdst=*/10, vector_source_vgpr(17), ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_current_high =
      build_v_mov_b32_e32(/*vdst=*/11, vector_source_vgpr(18), ROCJITSU_CODE_ARCH_RDNA4);
  const auto read_address =
      build_v_readfirstlane_b32(/*sdst=*/48, /*vsrc=*/8, ROCJITSU_CODE_ARCH_RDNA4);
  const auto read_metadata =
      build_v_readfirstlane_b32(/*sdst=*/49, /*vsrc=*/10, ROCJITSU_CODE_ARCH_RDNA4);
  const auto address_uniform =
      build_v_cmp_eq_u32_e32_vcc(/*src0=*/48, /*vsrc1=*/8, ROCJITSU_CODE_ARCH_RDNA4);
  const auto narrow_address =
      build_s_and_saveexec_b64(/*sdst=*/32, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto save_group = build_s_mov_b64(/*sdst=*/46, kRdna4ExecLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto metadata_uniform =
      build_v_cmp_eq_u32_e32_vcc(/*src0=*/49, /*vsrc1=*/10, ROCJITSU_CODE_ARCH_RDNA4);
  const auto narrow_metadata =
      build_s_and_saveexec_b64(/*sdst=*/32, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto low_mask_uniform =
      build_s_cmp_eq_u32(kRdna4ExecLo, /*ssrc1=*/46, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(save_incoming_exec);
  ASSERT_TRUE(initialize_pending);
  ASSERT_TRUE(select_pending);
  ASSERT_TRUE(read_address);
  ASSERT_TRUE(read_metadata);
  ASSERT_TRUE(address_uniform);
  ASSERT_TRUE(narrow_address);
  ASSERT_TRUE(save_group);
  ASSERT_TRUE(metadata_uniform);
  ASSERT_TRUE(narrow_metadata);
  ASSERT_TRUE(low_mask_uniform);
  const std::array<uint32_t, 19> expected_uniform_admission = {
      *save_incoming_exec, initialize_pending.value(), save_address_lo,      save_address_hi,
      save_current_low,    save_current_high,          *select_pending,      restore_address_lo,
      restore_address_hi,  restore_current_low,        restore_current_high, *read_address,
      *read_metadata,      *address_uniform,           *narrow_address,      *save_group,
      *metadata_uniform,   *narrow_metadata,           *low_mask_uniform,
  };
  EXPECT_TRUE(contains_subsequence(text_words, expected_uniform_admission));

  // EXEC is first restored from pending, then intersected with the address
  // equality mask before being copied to group. This pins group ⊆ pending,
  // the invariant that makes pending XOR group an exact set subtraction.
  const std::array<uint32_t, 4> expected_group_subset_construction = {
      *select_pending,
      restore_address_lo,
      restore_address_hi,
      restore_current_low,
  };
  const auto subset_prefix =
      std::search(text_words.begin(), text_words.end(), expected_group_subset_construction.begin(),
                  expected_group_subset_construction.end());
  ASSERT_NE(subset_prefix, text_words.end());
  const auto address_intersection = std::find(subset_prefix, text_words.end(), *narrow_address);
  ASSERT_NE(address_intersection, text_words.end());
  const auto group_capture = std::find(address_intersection, text_words.end(), *save_group);
  ASSERT_NE(group_capture, text_words.end());

  const auto remove_group = build_s_xor_b64(
      /*sdst=*/44, /*ssrc0=*/44, /*ssrc1=*/46, ROCJITSU_CODE_ARCH_RDNA4);
  const auto select_remaining =
      build_s_mov_b64(kRdna4ExecLo, /*ssrc0=*/44, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(remove_group);
  ASSERT_TRUE(select_remaining);
  const std::array<uint32_t, 1> expected_group_removal = {*remove_group};
  EXPECT_TRUE(contains_subsequence(text_words, expected_group_removal));
  const auto group_removal_position =
      std::search(text_words.begin(), text_words.end(), expected_group_removal.begin(),
                  expected_group_removal.end());
  ASSERT_NE(group_removal_position, text_words.end());
  ASSERT_LT(group_removal_position + expected_group_removal.size() + 3, text_words.end());
  const auto loop_control = group_removal_position + expected_group_removal.size();
  EXPECT_EQ(loop_control[0], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(loop_control[1], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(loop_control[2], *select_remaining);
  EXPECT_EQ(loop_control[3] & 0xffff0000u,
            pack_sopp(/*s_cbranch_execnz=*/0x26, /*simm16=*/0) & 0xffff0000u);

  const auto shadow_swap = build_flat_atomic_swap_b64_vaddr_vsrc_vdst(
      /*vaddr=*/8, /*vsrc=*/10, /*vdst=*/13, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(shadow_swap);
  EXPECT_EQ(count_subsequence(text_words, *shadow_swap), 0u);

  const auto version_cas = build_flat_atomic_cmpswap_b32_vaddr_vsrc_vdst(
      /*vaddr=*/8, /*vsrc=*/22, /*vdst=*/22, /*return_old_value=*/true,
      /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(version_cas);
  EXPECT_EQ(count_subsequence(text_words, *version_cas), 8u)
      << "both possible cells have uniform and lane-wise odd/even publication paths";

  // Both uniform representatives and metadata-distinct lane representatives
  // retry bounded cross-wave reservation contention. The outer partition loop
  // supplies the other backward EXEC branch.
  // s38:s39 preserve the guest VCC for an EXEC-save base of s30. The retry
  // counter must use the later diagnostic-temporary region instead.
  const uint16_t retry_count_sgpr = 50u;
  const std::array<uint32_t, 2> initialize_retries = {
      build_s_mov_b32(retry_count_sgpr, /*literal source=*/255u, ROCJITSU_CODE_ARCH_RDNA4),
      2048u,
  };
  EXPECT_TRUE(contains_subsequence(text_words, initialize_retries));
  const std::array<uint32_t, 2> clobber_saved_vcc = {
      build_s_mov_b32(/*sdst=*/38, /*literal source=*/255u, ROCJITSU_CODE_ARCH_RDNA4),
      2048u,
  };
  EXPECT_FALSE(contains_subsequence(text_words, clobber_saved_vcc));
  EXPECT_NE(std::find(text_words.begin(), text_words.end(),
                      build_s_sleep(/*delay=*/1, ROCJITSU_CODE_ARCH_RDNA4)),
            text_words.end());
  const auto decrement_retry = build_s_sub_u32(
      retry_count_sgpr, retry_count_sgpr, scalar_positive_inline_u32(1), ROCJITSU_CODE_ARCH_RDNA4);
  const auto retry_nonzero = build_rdna4_s_cmp_lg_u32(
      retry_count_sgpr, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
  const auto retry_exhausted = build_s_cbranch_scc0(1, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(decrement_retry);
  ASSERT_TRUE(retry_nonzero);
  ASSERT_TRUE(retry_exhausted);
  EXPECT_NE(std::find(text_words.begin(), text_words.end(), *decrement_retry), text_words.end());
  EXPECT_NE(std::find(text_words.begin(), text_words.end(), *retry_nonzero), text_words.end());
  EXPECT_NE(std::find(text_words.begin(), text_words.end(), *retry_exhausted), text_words.end());
  const uint32_t exec_branch_opcode =
      pack_sopp(/*s_cbranch_execnz=*/0x26, /*simm16=*/0) & 0xffff0000u;
  EXPECT_GE(std::ranges::count_if(text_words,
                                  [&](uint32_t word) {
                                    return (word & 0xffff0000u) == exec_branch_opcode &&
                                           static_cast<int16_t>(word) < 0;
                                  }),
            2u);

  const auto partition_rank_lo = build_v_mbcnt_lo_u32_b32(
      /*vdst=*/14, /*src0=*/46, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
  const auto partition_rank_hi = build_v_mbcnt_hi_u32_b32(
      /*vdst=*/14, /*src0=*/47, vector_source_vgpr(14), ROCJITSU_CODE_ARCH_RDNA4);
  const auto first_group_lane = build_v_cmp_eq_u32_e32_vcc(scalar_positive_inline_u32(0),
                                                           /*vsrc1=*/14, ROCJITSU_CODE_ARCH_RDNA4);
  const auto narrow_partition_representative =
      build_s_and_saveexec_b64(/*sdst=*/32, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_group = build_s_mov_b64(kRdna4ExecLo, /*ssrc0=*/46, ROCJITSU_CODE_ARCH_RDNA4);
  const auto save_publishers = build_s_mov_b64(/*sdst=*/34, kRdna4ExecLo, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(partition_rank_lo);
  ASSERT_TRUE(partition_rank_hi);
  ASSERT_TRUE(first_group_lane);
  const auto partition_wait = instrumentation::build_s_wait_global_load0(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(narrow_partition_representative);
  ASSERT_TRUE(restore_group);
  ASSERT_TRUE(save_publishers);
  ASSERT_TRUE(partition_wait);
  const uint64_t exchange_counter_address =
      *options.moi_report_buffer_address + offsetof(ConSanMoiReportHeader, event_counter);
  const auto counter_address_lo = build_v_mov_b32_e64_literal(
      /*vdst=*/8, static_cast<uint32_t>(exchange_counter_address), ROCJITSU_CODE_ARCH_RDNA4);
  const auto counter_address_hi = build_v_mov_b32_e64_literal(
      /*vdst=*/9, static_cast<uint32_t>(exchange_counter_address >> 32u), ROCJITSU_CODE_ARCH_RDNA4);
  const auto counter_one = build_v_mov_b32_e64_literal(/*vdst=*/12, 1u, ROCJITSU_CODE_ARCH_RDNA4);
  const auto count_exchange = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/8, /*vsrc=*/12, /*vdst=*/12, /*return_old_value=*/true,
      /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(counter_address_lo);
  ASSERT_TRUE(counter_address_hi);
  ASSERT_TRUE(counter_one);
  ASSERT_TRUE(count_exchange);
  std::vector<uint32_t> expected_exchange_count;
  expected_exchange_count.insert(expected_exchange_count.end(), counter_address_lo->begin(),
                                 counter_address_lo->end());
  expected_exchange_count.insert(expected_exchange_count.end(), counter_address_hi->begin(),
                                 counter_address_hi->end());
  expected_exchange_count.insert(expected_exchange_count.end(), counter_one->begin(),
                                 counter_one->end());
  expected_exchange_count.insert(expected_exchange_count.end(), count_exchange->begin(),
                                 count_exchange->end());
  expected_exchange_count.push_back(*partition_wait);
  std::vector<uint32_t> expected_uniform_publish;
  expected_uniform_publish.insert(expected_uniform_publish.end(), partition_rank_lo->begin(),
                                  partition_rank_lo->end());
  expected_uniform_publish.insert(expected_uniform_publish.end(), partition_rank_hi->begin(),
                                  partition_rank_hi->end());
  expected_uniform_publish.push_back(*first_group_lane);
  expected_uniform_publish.push_back(*narrow_partition_representative);
  expected_uniform_publish.push_back(*save_publishers);
  EXPECT_TRUE(contains_subsequence(text_words, expected_uniform_publish));

  const auto fallback_restore = std::find(text_words.begin(), text_words.end(), *restore_group);
  ASSERT_NE(fallback_restore, text_words.end());
  std::vector<uint32_t> expected_lane_wise_fallback;
  expected_lane_wise_fallback.insert(expected_lane_wise_fallback.end(), partition_rank_lo->begin(),
                                     partition_rank_lo->end());
  expected_lane_wise_fallback.insert(expected_lane_wise_fallback.end(), partition_rank_hi->begin(),
                                     partition_rank_hi->end());
  expected_lane_wise_fallback.push_back(*first_group_lane);
  expected_lane_wise_fallback.push_back(*narrow_partition_representative);
  expected_lane_wise_fallback.push_back(*save_group);
  EXPECT_NE(std::search(std::next(fallback_restore), text_words.end(),
                        expected_lane_wise_fallback.begin(), expected_lane_wise_fallback.end()),
            text_words.end())
      << "the fallback may compute byte provenance between restoring the address group and "
         "selecting its lane representative";
  EXPECT_GE(std::ranges::count(text_words, *save_publishers), 2u)
      << "both the uniform and lane-wise paths must preserve their publisher masks";
  const auto use_uniform_group_mask =
      build_s_mov_b64(/*sdst=*/34, /*ssrc0=*/46, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(use_uniform_group_mask);
  EXPECT_NE(std::find(text_words.begin(), text_words.end(), *use_uniform_group_mask),
            text_words.end())
      << "uniform publication may attribute its representative to the metadata-identical group";
  EXPECT_EQ(count_subsequence(text_words, expected_exchange_count), 4u)
      << "each generated transaction for both possible cells counts only a committed publication";

  const auto save_scc = build_rdna4_s_cselect_b32(
      /*sdst=*/40, scalar_positive_inline_u32(1), scalar_positive_inline_u32(0),
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto save_vcc = build_s_mov_b64(/*sdst=*/38, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(save_scc);
  ASSERT_TRUE(save_vcc);
  EXPECT_TRUE(contains_subsequence(text_words, std::array<uint32_t, 2>{*save_scc, *save_vcc}));
  const uint32_t zero = build_v_mov_b32_e32(
      /*vdst=*/12, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
  const auto nonempty =
      build_v_cmp_gt_u32_e32_vcc(vector_source_vgpr(13), /*vsrc1=*/12, ROCJITSU_CODE_ARCH_RDNA4);
  const auto narrow_nonempty =
      build_s_and_saveexec_b64(/*sdst=*/30, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto prior_owner = build_v_lshrrev_b32_e32(
      /*vdst=*/12, scalar_positive_inline_u32(consan_moi_exact_shadow::owner_shift), /*vsrc1=*/13,
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto owner_mask = build_v_and_b32_e32_literal(
      /*vdst=*/12, consan_moi_exact_shadow::max_owner, /*vsrc1=*/12, ROCJITSU_CODE_ARCH_RDNA4);
  const auto current_owner = build_v_lshrrev_b32_e32(
      /*vdst=*/11, scalar_positive_inline_u32(consan_moi_exact_shadow::owner_shift), /*vsrc1=*/10,
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto current_owner_mask = build_v_and_b32_e32_literal(
      /*vdst=*/11, consan_moi_exact_shadow::max_owner, /*vsrc1=*/11, ROCJITSU_CODE_ARCH_RDNA4);
  const auto owner_ne =
      build_v_cmp_ne_u32_e32_vcc(vector_source_vgpr(11), /*vsrc1=*/12, ROCJITSU_CODE_ARCH_RDNA4);
  const auto save_conflict_candidates =
      build_s_mov_b64(/*sdst=*/32, kRdna4ExecLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto narrow_different_owner =
      build_s_and_saveexec_b64(/*sdst=*/36, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto save_different_owner =
      build_s_mov_b64(/*sdst=*/34, kRdna4ExecLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto union_owner_conflicts =
      ib::build_s_xor_b64(kRdna4ExecLo, /*ssrc0=*/34, kRdna4ExecLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto prior_epoch = build_v_lshrrev_b32_e32(
      /*vdst=*/12, scalar_positive_inline_u32(consan_moi_exact_shadow::epoch_shift),
      /*vsrc1=*/13, ROCJITSU_CODE_ARCH_RDNA4);
  const auto epoch_mask = build_v_and_b32_e32_literal(
      /*vdst=*/12, consan_moi_exact_shadow::max_epoch, /*vsrc1=*/12, ROCJITSU_CODE_ARCH_RDNA4);
  const auto current_epoch = build_v_lshrrev_b32_e32(
      /*vdst=*/11, scalar_positive_inline_u32(consan_moi_exact_shadow::epoch_shift), /*vsrc1=*/10,
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto current_epoch_mask = build_v_and_b32_e32_literal(
      /*vdst=*/11, consan_moi_exact_shadow::max_epoch, /*vsrc1=*/11, ROCJITSU_CODE_ARCH_RDNA4);
  const auto epoch_eq =
      build_v_cmp_eq_u32_e32_vcc(vector_source_vgpr(11), /*vsrc1=*/12, ROCJITSU_CODE_ARCH_RDNA4);
  const auto narrow_same_epoch =
      build_s_and_saveexec_b64(/*sdst=*/32, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(nonempty);
  ASSERT_TRUE(narrow_nonempty);
  ASSERT_TRUE(prior_owner);
  ASSERT_TRUE(owner_mask);
  ASSERT_TRUE(current_owner);
  ASSERT_TRUE(current_owner_mask);
  ASSERT_TRUE(owner_ne);
  ASSERT_TRUE(save_conflict_candidates);
  ASSERT_TRUE(narrow_different_owner);
  ASSERT_TRUE(save_different_owner);
  ASSERT_TRUE(union_owner_conflicts);
  ASSERT_TRUE(prior_epoch);
  ASSERT_TRUE(epoch_mask);
  ASSERT_TRUE(current_epoch);
  ASSERT_TRUE(current_epoch_mask);
  ASSERT_TRUE(epoch_eq);
  ASSERT_TRUE(narrow_same_epoch);
  const std::array<uint32_t, 3> nonempty_predicate = {
      zero,
      *nonempty,
      *narrow_nonempty,
  };
  EXPECT_TRUE(contains_subsequence(text_words, nonempty_predicate));
  std::vector<uint32_t> owner_predicate;
  owner_predicate.push_back(*prior_owner);
  owner_predicate.insert(owner_predicate.end(), owner_mask->begin(), owner_mask->end());
  owner_predicate.push_back(*current_owner);
  owner_predicate.insert(owner_predicate.end(), current_owner_mask->begin(),
                         current_owner_mask->end());
  owner_predicate.push_back(*save_conflict_candidates);
  owner_predicate.push_back(*owner_ne);
  owner_predicate.push_back(*narrow_different_owner);
  owner_predicate.push_back(*save_different_owner);
  std::vector<uint32_t> epoch_predicate;
  epoch_predicate.push_back(*prior_epoch);
  epoch_predicate.insert(epoch_predicate.end(), epoch_mask->begin(), epoch_mask->end());
  epoch_predicate.push_back(*current_epoch);
  epoch_predicate.insert(epoch_predicate.end(), current_epoch_mask->begin(),
                         current_epoch_mask->end());
  epoch_predicate.push_back(*epoch_eq);
  epoch_predicate.push_back(*narrow_same_epoch);
  EXPECT_TRUE(contains_subsequence(text_words, owner_predicate));
  EXPECT_NE(std::ranges::find(text_words, *union_owner_conflicts), text_words.end())
      << "different-owner and same-owner/different-lane candidates form one exact conflict mask";
  EXPECT_TRUE(contains_subsequence(text_words, epoch_predicate));
  const auto epoch_position = std::search(text_words.begin(), text_words.end(),
                                          epoch_predicate.begin(), epoch_predicate.end());
  const auto owner_position = std::search(text_words.begin(), text_words.end(),
                                          owner_predicate.begin(), owner_predicate.end());
  ASSERT_NE(epoch_position, text_words.end());
  ASSERT_NE(owner_position, text_words.end());
  EXPECT_LT(owner_position, epoch_position)
      << "exact owner/lane conflict formation must precede epoch extraction";

  const auto count_address_lo = build_v_mov_b32_e64_literal(
      /*vdst=*/8,
      static_cast<uint32_t>(report_base + offsetof(ConSanMoiReportHeader, diagnostic_count)),
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto count_address_hi = build_v_mov_b32_e64_literal(
      /*vdst=*/9,
      static_cast<uint32_t>((report_base + offsetof(ConSanMoiReportHeader, diagnostic_count)) >>
                            32u),
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto count_one = build_v_mov_b32_e64_literal(/*vdst=*/22, 1u, ROCJITSU_CODE_ARCH_RDNA4);
  const auto count_add = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/8, /*vsrc=*/22, /*vdst=*/22, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto count_wait = instrumentation::build_s_wait_global_load0(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(count_address_lo);
  ASSERT_TRUE(count_address_hi);
  ASSERT_TRUE(count_one);
  ASSERT_TRUE(count_add);
  ASSERT_TRUE(count_wait);
  std::vector<uint32_t> expected_slot_reservation;
  expected_slot_reservation.insert(expected_slot_reservation.end(), count_address_lo->begin(),
                                   count_address_lo->end());
  expected_slot_reservation.insert(expected_slot_reservation.end(), count_address_hi->begin(),
                                   count_address_hi->end());
  expected_slot_reservation.insert(expected_slot_reservation.end(), count_one->begin(),
                                   count_one->end());
  expected_slot_reservation.insert(expected_slot_reservation.end(), count_add->begin(),
                                   count_add->end());
  expected_slot_reservation.push_back(*count_wait);
  EXPECT_TRUE(contains_subsequence(text_words, expected_slot_reservation));
  EXPECT_EQ(count_subsequence(text_words, *count_add), 2u);

  // Diagnostic reservation is wave-coalesced: preserve the complete conflict
  // mask for the record, compute each active lane's rank, and let only rank
  // zero reserve and populate a diagnostic slot. Shadow publication remains a
  // separate IS3 stage because it must partition lanes by shadow address.
  const auto save_conflict_exec =
      build_s_mov_b64(/*sdst=*/32, kRdna4ExecLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto lane_rank_lo = build_v_mbcnt_lo_u32_b32(
      /*vdst=*/12, /*src0=*/32, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
  const auto lane_rank_hi = build_v_mbcnt_hi_u32_b32(
      /*vdst=*/12, /*src0=*/33, vector_source_vgpr(12), ROCJITSU_CODE_ARCH_RDNA4);
  const auto first_active_lane = build_v_cmp_eq_u32_e32_vcc(scalar_positive_inline_u32(0),
                                                            /*vsrc1=*/12, ROCJITSU_CODE_ARCH_RDNA4);
  const auto narrow_representative =
      build_s_and_saveexec_b64(/*sdst=*/34, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
  const auto saved_exec_wait =
      instrumentation::build_salu_to_valu_dependency_wait(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(save_conflict_exec);
  ASSERT_TRUE(saved_exec_wait);
  ASSERT_TRUE(lane_rank_lo);
  ASSERT_TRUE(lane_rank_hi);
  ASSERT_TRUE(first_active_lane);
  ASSERT_TRUE(narrow_representative);
  std::vector<uint32_t> expected_wave_coalesced_reservation = {*save_conflict_exec};
  expected_wave_coalesced_reservation.push_back(*saved_exec_wait);
  expected_wave_coalesced_reservation.insert(expected_wave_coalesced_reservation.end(),
                                             lane_rank_lo->begin(), lane_rank_lo->end());
  expected_wave_coalesced_reservation.insert(expected_wave_coalesced_reservation.end(),
                                             lane_rank_hi->begin(), lane_rank_hi->end());
  expected_wave_coalesced_reservation.push_back(*first_active_lane);
  expected_wave_coalesced_reservation.push_back(*narrow_representative);
  expected_wave_coalesced_reservation.insert(expected_wave_coalesced_reservation.end(),
                                             expected_slot_reservation.begin(),
                                             expected_slot_reservation.end());
  EXPECT_TRUE(contains_subsequence(text_words, expected_wave_coalesced_reservation));

  const auto restore_vcc = build_s_mov_b64(kRdna4VccLo, /*ssrc0=*/38, ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_scc = build_rdna4_s_cmp_lg_u32(
      /*ssrc0=*/40, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(restore_vcc);
  ASSERT_TRUE(restore_scc);
  const std::array<uint32_t, 2> expected_restore = {
      *restore_vcc,
      *restore_scc,
  };
  EXPECT_TRUE(contains_subsequence(text_words, expected_restore));

  const auto restore_original_exec =
      build_s_mov_b64(kRdna4ExecLo, /*ssrc0=*/42, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(restore_original_exec);
  const std::array<uint32_t, 5> expected_partition_restore = {
      *restore_original_exec, restore_current_low, restore_current_high, *restore_vcc, *restore_scc,
  };
  EXPECT_TRUE(contains_subsequence(text_words, expected_partition_restore));

  const auto remove_position = std::find(text_words.begin(), text_words.end(), *remove_group);
  ASSERT_NE(remove_position, text_words.end());
  EXPECT_LT(owner_position, remove_position)
      << "the full address-group diagnostic must run before its lanes leave pending EXEC";
}

TEST(ConSanMoi, InlineShadowWaveCoalescingRejectsScalarSaveWindowOverSpecialRegisters) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 24;
  options.moi_epoch_vgpr = 25;
  options.moi_exec_save_sgpr = 92;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const auto result = try_patch_consan(bytes, options);

  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("scalar instrumentation state aliases an architectural special SGPR") !=
           std::string::npos;
  })) << testing::PrintToString(result.warnings);
}

TEST(ConSanMoi, InlineShadowPartitionMaskDebugIsBoundedAndExplicit) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 24;
  options.moi_epoch_vgpr = 25;
  options.moi_exec_save_sgpr = 30;
  options.moi_partition_mask_debug = true;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const auto result = try_patch_consan(bytes, options);
  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const auto *text_section = patched.text_sections().front();
  std::vector<uint32_t> text_words(text_section->size() / sizeof(uint32_t));
  std::memcpy(text_words.data(), text_section->data(), text_section->size());

  const std::array<size_t, 7> offsets = {
      offsetof(ConSanMoiReportHeader, dispatch_id),
      offsetof(ConSanMoiReportHeader, dispatch_id) + sizeof(uint32_t),
      offsetof(ConSanMoiReportHeader, flags),
      offsetof(ConSanMoiReportHeader, access_record_count),
      offsetof(ConSanMoiReportHeader, barrier_record_count),
      offsetof(ConSanMoiReportHeader, atomic_record_count),
      offsetof(ConSanMoiReportHeader, event_counter),
  };
  for (size_t index = 0; index < offsets.size(); ++index) {
    const size_t offset = offsets[index];
    ASSERT_LE(offset + sizeof(uint32_t), sizeof(ConSanMoiReportHeader));
    const auto address = build_v_mov_b32_e64_literal(
        /*vdst=*/index == 2 || index == 5 ? 13 : 8,
        static_cast<uint32_t>(*options.moi_report_buffer_address + offset),
        ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_TRUE(address);
    EXPECT_TRUE(contains_subsequence(text_words, *address));
  }
  const std::array<size_t, 7> protected_capacity_offsets = {
      offsetof(ConSanMoiReportHeader, access_record_capacity),
      offsetof(ConSanMoiReportHeader, barrier_record_capacity),
      offsetof(ConSanMoiReportHeader, atomic_record_capacity),
      offsetof(ConSanMoiReportHeader, diagnostic_capacity),
      offsetof(ConSanMoiReportHeader, exact_shadow_entry_capacity),
      offsetof(ConSanMoiReportHeader, sampled_watchpoint_capacity),
      offsetof(ConSanMoiReportHeader, inline_atomic_release_capacity),
  };
  for (size_t offset : protected_capacity_offsets) {
    for (uint16_t address_vgpr : {8u, 15u}) {
      const auto address = build_v_mov_b32_e64_literal(
          address_vgpr, static_cast<uint32_t>(*options.moi_report_buffer_address + offset),
          ROCJITSU_CODE_ARCH_RDNA4);
      ASSERT_TRUE(address);
      EXPECT_FALSE(contains_subsequence(text_words, *address));
    }
  }

  // MBCNT ranks a lane against the active subset supplied in src0. Using the
  // inline -1 source computes the physical lane id instead, so only lane zero
  // can represent a singleton divergent-address group or conflict mask.
  const auto diagnostic_rank_lo = build_v_mbcnt_lo_u32_b32(
      /*vdst=*/12, /*src0=*/32, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
  const auto diagnostic_rank_hi = build_v_mbcnt_hi_u32_b32(
      /*vdst=*/12, /*src0=*/33, vector_source_vgpr(12), ROCJITSU_CODE_ARCH_RDNA4);
  const auto group_rank_lo = build_v_mbcnt_lo_u32_b32(
      /*vdst=*/14, /*src0=*/46, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
  const auto group_rank_hi = build_v_mbcnt_hi_u32_b32(
      /*vdst=*/14, /*src0=*/47, vector_source_vgpr(14), ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(diagnostic_rank_lo);
  ASSERT_TRUE(diagnostic_rank_hi);
  ASSERT_TRUE(group_rank_lo);
  ASSERT_TRUE(group_rank_hi);
  EXPECT_TRUE(contains_subsequence(text_words, *diagnostic_rank_lo));
  EXPECT_TRUE(contains_subsequence(text_words, *diagnostic_rank_hi));
  EXPECT_TRUE(contains_subsequence(text_words, *group_rank_lo));
  EXPECT_TRUE(contains_subsequence(text_words, *group_rank_hi));
}

TEST(ConSanMoi, InlineShadowProbeCanPatchTwoAppendedCaveSites) {
  const std::array<uint32_t, 5> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32 v0, v0
      0xD8D80000u,
      0x00000000u, // ds_load_b32 v0, v0
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 25;
  options.moi_epoch_vgpr = 26;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.max_patches = 2;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 2u);
  EXPECT_EQ(result.patches[0].kind, ConSanPatchKind::TrampolineMoiExactShadowStore);
  EXPECT_EQ(result.patches[0].anchor_offset, 0u);
  EXPECT_EQ(result.patches[0].original_size, 2u * sizeof(uint32_t));
  EXPECT_EQ(result.patches[1].kind, ConSanPatchKind::TrampolineMoiExactShadowStore);
  EXPECT_EQ(result.patches[1].anchor_offset, 2u * sizeof(uint32_t));
  EXPECT_EQ(result.patches[1].original_size, 2u * sizeof(uint32_t));

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  EXPECT_GT(patched.text_sections().front()->size(), text_words.size() * sizeof(uint32_t));

  // The load overwrites its address VGPR. Its trampoline must snapshot v0,
  // derive the same cell index as the store, and defer the application load
  // until all shadow bookkeeping is complete.
  const ConSanPatchInfo &load_patch = result.patches[1];
  const std::vector<uint32_t> load_cave =
      text_words_at_offset(patched, load_patch.trampoline_offset, load_patch.trampoline_size);
  const uint32_t save_address =
      build_v_mov_b32_e32(/*vdst=*/24, vector_source_vgpr(/*vsrc=*/0), ROCJITSU_CODE_ARCH_RDNA4);
  const auto load_cell = build_v_lshrrev_b32_e32(
      /*vdst=*/12, scalar_positive_inline_u32(consan_moi_exact_shadow::granule_shift),
      /*vsrc1=*/24, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(load_cell);
  ASSERT_GE(load_cave.size(), 3u);
  EXPECT_EQ(load_cave[0], save_address);
  EXPECT_TRUE(contains_subsequence(load_cave, std::array<uint32_t, 1>{*load_cell}));
  const std::array<uint32_t, 2> guest_load = {text_words[2], text_words[3]};
  const auto load_position =
      std::search(load_cave.begin(), load_cave.end(), guest_load.begin(), guest_load.end());
  const auto shadow_position = std::find(load_cave.begin(), load_cave.end(), *load_cell);
  ASSERT_NE(load_position, load_cave.end());
  ASSERT_NE(shadow_position, load_cave.end());
  EXPECT_LT(shadow_position, load_position);

  ASSERT_EQ(result.resource_plans.size(), 2u);
  EXPECT_EQ(result.resource_plans[0].scratch_vgpr_count, 16u);
  EXPECT_EQ(result.resource_plans[1].scratch_vgpr_count, 17u);
}

TEST(ConSanMoi, Gfx1250InlineShadowDefersLoadBeforeAnchorIslandContinuation) {
  constexpr size_t kLargeTextWords = 33000u;
  constexpr size_t kOwnerWords = 65u;
  constexpr auto load = gfx1250::build_vds(gfx1250::kDsLoadB128Vds, {.addr = 4, .vdst = 8});
  const uint32_t filler = build_s_mov_b32(100, 100, ROCJITSU_CODE_ARCH_GFX1250);
  std::vector<uint32_t> text_words(kLargeTextWords, filler);
  for (size_t site = 0; site < 8u; ++site) {
    const size_t offset = site * 8u;
    text_words[offset] = load[0];
    text_words[offset + 1u] = load[1];
  }
  const uint32_t address_consumer =
      build_v_mov_b32_e32(/*vdst=*/4, vector_source_vgpr(11), ROCJITSU_CODE_ARCH_GFX1250);
  text_words[2] = address_consumer;
  text_words[kOwnerWords - 1u] = build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250);

  std::vector<uint8_t> bytes = make_gfx1250_code_object(text_words, "far_vds_load");
  mutate_elf_symbol(bytes, 1,
                    [](Elf64_Sym &symbol) { symbol.st_size = kOwnerWords * sizeof(uint32_t); });
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.scratch_vgpr = 82;
  options.moi_owner_vgpr = 80;
  options.moi_epoch_vgpr = 81;
  options.moi_exec_save_sgpr = 60;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.max_patches = 1;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto patch = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiExactShadowStore, &ConSanPatchInfo::kind);
  ASSERT_NE(patch, result.patches.end());
  ASSERT_EQ(patch->original_size, 8u * sizeof(uint32_t));
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> cave =
      text_words_at_offset(patched, patch->trampoline_offset, patch->trampoline_size);
  const auto load_position = std::search(cave.begin(), cave.end(), load.begin(), load.end());
  const auto consumer_position = std::find(cave.begin(), cave.end(), address_consumer);
  ASSERT_NE(load_position, cave.end());
  ASSERT_NE(consumer_position, cave.end());
  EXPECT_LT(load_position, consumer_position);
}

TEST(ConSanMoi, Gfx1250InlineShadowPreservesGuestVgprBankForDeferredLoad) {
  constexpr uint32_t kGuestMode = 0x40u;
  constexpr uint32_t kSelectLow = 0xBF864000u;
  constexpr uint32_t kRestoreGuest = 0xBF860040u;
  constexpr auto load = gfx1250::build_vds(gfx1250::kDsLoadB128Vds, {.addr = 4, .vdst = 8});
  std::vector<uint32_t> text_words = {0xBF860000u | kGuestMode, load[0], load[1]};
  text_words.resize(48u, build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250));
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250);
  const std::vector<uint8_t> bytes = make_gfx1250_code_object(text_words, "banked_vds_load");
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.scratch_vgpr = 82;
  options.moi_owner_vgpr = 80;
  options.moi_epoch_vgpr = 81;
  options.moi_exec_save_sgpr = 60;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.max_patches = 1;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto patch = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiExactShadowStore, &ConSanPatchInfo::kind);
  ASSERT_NE(patch, result.patches.end());
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> cave =
      text_words_at_offset(patched, patch->trampoline_offset, patch->trampoline_size);
  const auto select_low = std::find(cave.begin(), cave.end(), kSelectLow);
  const auto restore_guest = std::find(cave.begin(), cave.end(), kRestoreGuest);
  const auto guest_load = std::search(cave.begin(), cave.end(), load.begin(), load.end());
  ASSERT_NE(select_low, cave.end());
  ASSERT_NE(restore_guest, cave.end());
  ASSERT_NE(guest_load, cave.end());
  EXPECT_LT(select_low, restore_guest);
  EXPECT_LT(restore_guest, guest_load);
}

TEST(ConSanMoi, Gfx1250DenseInlineShadowAccessesShareOneWordCallRelay) {
  constexpr uint32_t kAccessCount = 9u;
  std::vector<uint32_t> text_words(
      9u, build_s_mov_b32(/*sdst=*/0, /*ssrc0=*/0, ROCJITSU_CODE_ARCH_GFX1250));
  for (uint32_t index = 0; index < kAccessCount; ++index) {
    text_words.push_back(0xD8340000u | index * sizeof(uint32_t));
    text_words.push_back(0x00000000u); // ds_store_b32 v0, v0 offset:index*4
  }
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250));
  const std::vector<uint8_t> bytes =
      make_gfx1250_code_object(text_words, "gfx1250_dense_inline_shadow");

  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.scratch_vgpr = 82;
  options.moi_owner_vgpr = 80;
  options.moi_epoch_vgpr = 81;
  options.moi_exec_save_sgpr = 60;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.max_patches = kAccessCount;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiExactShadowStore,
                               &ConSanPatchInfo::kind),
            kAccessCount);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiIndirectBranchIsland,
                               &ConSanPatchInfo::kind),
            2u); // One relocatable host plus one appended return-PC dispatcher.
}

TEST(ConSanMoi, Rdna4DenseInlineShadowAccessesShareExplicitKeyRelay) {
  // Sixty-four large Inline bodies force the later dispatcher targets outside
  // direct SOPP reach and exercise the worst-case long-jump reservation.
  constexpr uint32_t kAccessCount = 64u;
  std::vector<uint32_t> text_words(
      8u, build_s_mov_b32(/*sdst=*/0, /*ssrc0=*/0, ROCJITSU_CODE_ARCH_RDNA4));
  for (uint32_t index = 0; index < kAccessCount; ++index) {
    text_words.push_back(0xD8340000u | index * sizeof(uint32_t));
    text_words.push_back(0x00000000u); // ds_store_b32 v0, v0 offset:index*4
  }
  text_words.resize(33'000u, build_s_mov_b32(/*sdst=*/0, /*ssrc0=*/0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4));

  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.scratch_vgpr = 82;
  options.moi_owner_vgpr = 80;
  options.moi_epoch_vgpr = 81;
  options.moi_exec_save_sgpr = 60;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.max_patches = kAccessCount;

  const ConSanResult result = try_patch_consan(
      make_rdna4_lds_code_object(text_words, "rdna4_dense_inline_shadow"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiExactShadowStore,
                               &ConSanPatchInfo::kind),
            kAccessCount);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiIndirectBranchIsland,
                               &ConSanPatchInfo::kind),
            2u); // One local relay plus one appended explicit-key dispatcher.
  EXPECT_TRUE(std::ranges::none_of(result.warnings, [](const std::string &warning) {
    return warning.find("inside a relocated prefix") != std::string::npos;
  }));

  constexpr uint16_t kExecSaveBase = 60u;
  constexpr uint16_t kExplicitKeySgpr = kExecSaveBase + 28u;
  constexpr uint16_t kIndirectPcSgpr = kExecSaveBase + 12u;
  constexpr uint64_t kFirstAccessOffset = 8u * sizeof(uint32_t);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> anchor_words =
      text_words_at_offset(patched, kFirstAccessOffset, 2u * sizeof(uint32_t));
  ASSERT_EQ(anchor_words.size(), 2u);
  EXPECT_EQ(anchor_words.front(), build_s_mov_b32(kExplicitKeySgpr, scalar_positive_inline_u32(1u),
                                                  ROCJITSU_CODE_ARCH_RDNA4));

  const auto first_access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore &&
           patch.anchor_offset == kFirstAccessOffset;
  });
  ASSERT_NE(first_access, result.patches.end());
  const std::vector<uint32_t> body_words =
      text_words_at_offset(patched, first_access->trampoline_offset, first_access->trampoline_size);
  ASSERT_FALSE(body_words.empty());
  EXPECT_EQ(body_words.back(), build_s_setpc_b64(kIndirectPcSgpr, ROCJITSU_CODE_ARCH_RDNA4));

  const auto compare_key = ib::build_s_cmp_eq_u32(kExplicitKeySgpr, scalar_positive_inline_u32(1u),
                                                  ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(compare_key);
  EXPECT_TRUE(std::ranges::any_of(result.patches, [&](const ConSanPatchInfo &patch) {
    if (patch.kind != ConSanPatchKind::TrampolineMoiIndirectBranchIsland ||
        patch.trampoline_size == 0u) {
      return false;
    }
    const std::vector<uint32_t> words =
        text_words_at_offset(patched, patch.trampoline_offset, patch.trampoline_size);
    return std::ranges::find(words, *compare_key) != words.end();
  }));
}

TEST(ConSanMoi, Rdna4LargeInlineShadowCompositionUsesGeneralDenseRouting) {
  constexpr uint32_t kAccessCount = 136u;
  std::vector<uint32_t> text_words(
      8u, build_s_mov_b32(/*sdst=*/0, /*ssrc0=*/0, ROCJITSU_CODE_ARCH_RDNA4));
  for (uint32_t index = 0; index < kAccessCount; ++index) {
    text_words.push_back(0xD8340000u | index * sizeof(uint32_t));
    text_words.push_back(0x00000000u); // ds_store_b32 v0, v0 offset:index*4
  }
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4));

  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.scratch_vgpr = 82;
  options.moi_owner_vgpr = 80;
  options.moi_epoch_vgpr = 81;
  options.moi_exec_save_sgpr = 60;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.max_patches = kAccessCount;

  const ConSanResult result = try_patch_consan(
      make_rdna4_lds_code_object(text_words, "rdna4_large_inline_shadow"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiExactShadowStore,
                               &ConSanPatchInfo::kind),
            kAccessCount);
  EXPECT_TRUE(std::ranges::none_of(result.warnings, [](const std::string &warning) {
    return warning.find("skipped access site") != std::string::npos;
  })) << testing::PrintToString(result.warnings);
}

TEST(ConSanMoi, AutomaticTransientPlanningScalesAcrossIndependentKernels) {
  constexpr uint32_t kKernelCount = 2048u;
  constexpr uint32_t kAccessesPerKernel = 8u;
  constexpr size_t kAccessCount =
      static_cast<size_t>(kKernelCount) * static_cast<size_t>(kAccessesPerKernel);

  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_owner_vgpr = 80u;
  options.moi_epoch_vgpr = 81u;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  // Planning still has to classify every site and owner. Keeping emission to
  // one probe makes this a focused scaling regression instead of a file-growth
  // stress test.
  options.max_patches = 1u;

  const ConSanResult result = try_patch_consan(
      make_rdna4_many_kernel_lds_code_object(kKernelCount, kAccessesPerKernel), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  EXPECT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_EQ(result.kernels.size(), kKernelCount);
  EXPECT_EQ(std::ranges::count(result.site_dispositions, ConSanResourceSiteKind::Access,
                               &ConSanSiteDispositionRecord::site_kind),
            kAccessCount);
  EXPECT_EQ(std::ranges::count(result.site_dispositions, ConSanSiteDisposition::Supported,
                               &ConSanSiteDispositionRecord::disposition),
            kAccessCount);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiExactShadowStore,
                               &ConSanPatchInfo::kind),
            1u);
}

TEST(ConSanMoi, AutomaticTransientEmissionScalesAcrossLargeKernel) {
  constexpr uint32_t kAccessCount = 512u;
  std::vector<uint32_t> text_words(
      8u, build_s_mov_b32(/*sdst=*/0, /*ssrc0=*/0, ROCJITSU_CODE_ARCH_RDNA4));
  for (uint32_t index = 0; index < kAccessCount; ++index) {
    text_words.push_back(0xD8340000u | (index % 64u) * sizeof(uint32_t));
    text_words.push_back(0x00000000u); // ds_store_b32 v0, v0
  }
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4));

  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_owner_vgpr = 80u;
  options.moi_epoch_vgpr = 81u;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.max_patches = kAccessCount;

  const ConSanResult result = try_patch_consan(
      make_rdna4_lds_code_object(text_words, "rdna4_large_automatic_inline_shadow"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiExactShadowStore,
                               &ConSanPatchInfo::kind),
            kAccessCount);
  EXPECT_TRUE(std::ranges::none_of(result.warnings, [](const std::string &warning) {
    return warning.find("skipped access site") != std::string::npos;
  })) << testing::PrintToString(result.warnings);
}

TEST(ConSanMoi, Rdna4FarAccessAndAdjacentBarrierUseIndependentDenseRoutes) {
  // Keep the first of many sites farther than direct SOPP reach from the
  // appended body and put a full barrier immediately after it, matching the
  // shape of large generated kernels. Access and barrier calls must share only
  // the relay, retaining separate anchors and bodies.
  constexpr uint32_t kAccessCount = 136u;
  constexpr size_t kLargeTextWords = 33'000u;
  constexpr size_t kFirstAccessWord = 32u;
  const uint32_t filler = build_s_mov_b32(0, 0, ROCJITSU_CODE_ARCH_RDNA4);
  std::vector<uint32_t> text_words(kLargeTextWords, filler);
  text_words[kFirstAccessWord] = 0xD8340000u;
  text_words[kFirstAccessWord + 1u] = 0x00000000u; // ds_store_b32 v0, v0
  text_words[kFirstAccessWord + 2u] = 0xBE804EC1u; // s_barrier_signal -1
  text_words[kFirstAccessWord + 3u] = 0xBF94FFFFu; // s_barrier_wait -1
  for (uint32_t index = 1u; index < kAccessCount; ++index) {
    const size_t word = kFirstAccessWord + 4u + index * 2u;
    text_words[word] = 0xD8340000u | index * sizeof(uint32_t);
    text_words[word + 1u] = 0x00000000u; // ds_store_b32 v0, v0 offset:index*4
  }
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);

  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.scratch_vgpr = 82;
  options.moi_owner_vgpr = 80;
  options.moi_epoch_vgpr = 81;
  options.moi_exec_save_sgpr = 60;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.moi_track_barriers = true;
  options.moi_track_atomics = false;
  options.max_patches = 256;

  const ConSanResult result = try_patch_consan(
      make_rdna4_lds_code_object(text_words, "rdna4_far_access_adjacent_barrier"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  const uint64_t first_access_offset = kFirstAccessWord * sizeof(uint32_t);
  const uint64_t barrier_signal_offset = (kFirstAccessWord + 2u) * sizeof(uint32_t);
  const uint64_t barrier_wait_offset = (kFirstAccessWord + 3u) * sizeof(uint32_t);
  EXPECT_TRUE(std::ranges::any_of(result.patches, [&](const ConSanPatchInfo &patch) {
    return (patch.kind == ConSanPatchKind::InlineMoiExactShadowStore ||
            patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore) &&
           patch.anchor_offset == first_access_offset;
  }));
  EXPECT_TRUE(std::ranges::none_of(result.patches, [&](const ConSanPatchInfo &patch) {
    return (patch.kind == ConSanPatchKind::InlineMoiExactShadowStore ||
            patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore) &&
           patch.anchor_offset < barrier_signal_offset &&
           patch.anchor_offset + patch.original_size > barrier_signal_offset;
  }));
  EXPECT_TRUE(std::ranges::any_of(result.patches, [&](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiInlineEpochBarrier &&
           patch.anchor_offset == barrier_wait_offset;
  }));
}

TEST(ConSanMoi, Gfx1250DenseInlineShadowBarriersUseSpillBackedRouter) {
  constexpr uint32_t kAccessCount = 9u;
  constexpr size_t kLargeTextWords = 33000u;
  const uint32_t filler = build_s_mov_b32(100, 100, ROCJITSU_CODE_ARCH_GFX1250);
  std::vector<uint32_t> text_words(kLargeTextWords, filler);
  size_t cursor = 32u;
  for (uint32_t index = 0; index < kAccessCount; ++index) {
    text_words[cursor++] = 0xD8340000u | index * sizeof(uint32_t);
    text_words[cursor++] = 0x00000000u; // ds_store_b32 v0, v0 offset:index*4
    text_words[cursor++] = *build_s_barrier_signal_all(ROCJITSU_CODE_ARCH_GFX1250);
    text_words[cursor++] = *build_s_barrier_wait_all(ROCJITSU_CODE_ARCH_GFX1250);
  }
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250);
  const std::vector<uint8_t> bytes =
      make_gfx1250_code_object(text_words, "gfx1250_dense_inline_shadow_barriers");

  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.scratch_vgpr = 82;
  options.moi_owner_vgpr = 80;
  options.moi_epoch_vgpr = 81;
  options.moi_exec_save_sgpr = 60;
  options.automatic_moi_exec_save_sgprs = true;
  options.automatic_moi_inline_sgpr_spill = true;
  options.moi_inline_visible_evidence_sgpr = 28;
  options.moi_inline_indirect_pc_sgpr = 30;
  options.moi_inline_call_return_sgpr = 26;
  options.moi_inline_dispatch_key_sgpr = 25;
  options.moi_inline_indirect_scc_sgpr = 29;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.moi_track_barriers = true;
  options.moi_track_atomics = false;
  options.max_patches = 64;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiExactShadowStore,
                               &ConSanPatchInfo::kind),
            kAccessCount);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiInlineEpochBarrier,
                               &ConSanPatchInfo::kind),
            kAccessCount) // One epoch advance after each signal/wait pair completes.
      << testing::PrintToString(result.warnings);
  const auto access_dispatcher =
      std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
        return patch.kind == ConSanPatchKind::TrampolineMoiIndirectBranchIsland &&
               patch.moi_dense_entry_island_offset.has_value();
      });
  ASSERT_NE(access_dispatcher, result.patches.end());

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const uint32_t external_return = build_s_setpc_b64(/*sdst=*/26, ROCJITSU_CODE_ARCH_GFX1250);
  const uint32_t spill_window_return = build_s_setpc_b64(/*sdst=*/88, ROCJITSU_CODE_ARCH_GFX1250);
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.kind != ConSanPatchKind::TrampolineMoiExactShadowStore)
      continue;
    const std::vector<uint32_t> cave =
        text_words_at_offset(patched, patch.trampoline_offset, patch.trampoline_size);
    EXPECT_NE(std::ranges::find(cave, external_return), cave.end());
    EXPECT_EQ(std::ranges::find(cave, spill_window_return), cave.end());
  }
}

TEST(ConSanMoi, Gfx1250DenseInlineShadowBarrierReusesAccessDispatcherWhenItFits) {
  constexpr uint32_t kAccessCount = 9u;
  constexpr size_t kLargeTextWords = 33'000u;
  const uint32_t filler = build_s_mov_b32(100, 100, ROCJITSU_CODE_ARCH_GFX1250);
  std::vector<uint32_t> text_words(kLargeTextWords, filler);
  size_t cursor = 32u;
  for (uint32_t index = 0; index < kAccessCount; ++index) {
    text_words[cursor++] = 0xD8340000u | index * sizeof(uint32_t);
    text_words[cursor++] = 0x00000000u; // ds_store_b32 v0, v0
  }
  text_words[cursor++] = *build_s_barrier_signal_all(ROCJITSU_CODE_ARCH_GFX1250);
  text_words[cursor++] = *build_s_barrier_wait_all(ROCJITSU_CODE_ARCH_GFX1250);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250);

  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.scratch_vgpr = 82;
  options.moi_owner_vgpr = 80;
  options.moi_epoch_vgpr = 81;
  options.moi_exec_save_sgpr = 60;
  options.automatic_moi_exec_save_sgprs = true;
  options.automatic_moi_inline_sgpr_spill = true;
  options.moi_inline_visible_evidence_sgpr = 28;
  options.moi_inline_indirect_pc_sgpr = 30;
  options.moi_inline_call_return_sgpr = 26;
  options.moi_inline_dispatch_key_sgpr = 25;
  options.moi_inline_indirect_scc_sgpr = 29;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.moi_track_barriers = true;
  options.moi_track_atomics = false;
  options.max_patches = kAccessCount + 1u;

  const ConSanResult result = try_patch_consan(
      make_gfx1250_code_object(text_words, "gfx1250_dense_barrier_reuses_access_dispatcher"),
      options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  const auto access_dispatcher =
      std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
        return patch.kind == ConSanPatchKind::TrampolineMoiIndirectBranchIsland &&
               patch.moi_dense_entry_island_offset.has_value();
      });
  const auto barrier = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiInlineEpochBarrier, &ConSanPatchInfo::kind);
  ASSERT_NE(access_dispatcher, result.patches.end());
  ASSERT_NE(barrier, result.patches.end());
  const auto call_delta = compute_sopp_branch_simm16(
      barrier->anchor_offset, *access_dispatcher->moi_dense_entry_island_offset);
  ASSERT_TRUE(call_delta);
  const auto reused_call =
      instrumentation::build_s_call_i64(/*sdst=*/26u, *call_delta, ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(reused_call);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> anchor =
      text_words_at_offset(patched, barrier->anchor_offset, sizeof(uint32_t));
  ASSERT_EQ(anchor.size(), 1u);
  EXPECT_EQ(anchor.front(), *reused_call);
}

TEST(ConSanMoi, Rdna4DenseInlineShadowBarriersUseRelocatedRouter) {
  constexpr uint32_t kAccessCount = 9u;
  constexpr size_t kLargeTextWords = 33'000u;
  const uint32_t filler = build_s_mov_b32(0, 0, ROCJITSU_CODE_ARCH_RDNA4);
  std::vector<uint32_t> text_words(kLargeTextWords, filler);
  size_t cursor = 32u;
  for (uint32_t index = 0; index < kAccessCount; ++index) {
    text_words[cursor++] = 0xD8340000u | index * sizeof(uint32_t);
    text_words[cursor++] = 0x00000000u; // ds_store_b32 v0, v0 offset:index*4
    text_words[cursor++] = 0xBE804EC1u; // s_barrier_signal -1
    text_words[cursor++] = 0xBF94FFFFu; // s_barrier_wait -1
  }
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4);

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

  const ConSanResult result = try_patch_consan(
      make_rdna4_lds_code_object(text_words, "rdna4_dense_inline_shadow_barriers"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiExactShadowStore,
                               &ConSanPatchInfo::kind),
            kAccessCount);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiInlineEpochBarrier,
                               &ConSanPatchInfo::kind),
            kAccessCount);
  EXPECT_EQ(std::ranges::count_if(result.patches,
                                  [](const ConSanPatchInfo &patch) {
                                    return patch.kind ==
                                               ConSanPatchKind::TrampolineMoiIndirectBranchIsland &&
                                           patch.moi_dense_entry_island_offset.has_value();
                                  }),
            1u);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiIndirectBranchIsland,
                               &ConSanPatchInfo::kind),
            4u); // Access and barrier routing remain structurally independent on RDNA4.
}

TEST(ConSanMoi, Gfx1250DenseBarrierFallsBackWhenAccessDispatcherReservationIsFull) {
  constexpr uint32_t kAccessCount = 9u;
  constexpr uint32_t kBarrierCount = 64u;
  constexpr size_t kLargeTextWords = 33'000u;
  const uint32_t filler = build_s_mov_b32(100, 100, ROCJITSU_CODE_ARCH_GFX1250);
  std::vector<uint32_t> text_words(kLargeTextWords, filler);
  size_t cursor = 32u;
  for (uint32_t index = 0; index < kAccessCount; ++index) {
    text_words[cursor++] = 0xD8340000u | index * sizeof(uint32_t);
    text_words[cursor++] = 0x00000000u; // ds_store_b32 v0, v0
  }
  for (uint32_t index = 0; index < kBarrierCount; ++index) {
    text_words[cursor++] = *build_s_barrier_signal_all(ROCJITSU_CODE_ARCH_GFX1250);
    text_words[cursor++] = *build_s_barrier_wait_all(ROCJITSU_CODE_ARCH_GFX1250);
  }
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250);

  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.scratch_vgpr = 82;
  options.moi_owner_vgpr = 80;
  options.moi_epoch_vgpr = 81;
  options.moi_exec_save_sgpr = 60;
  options.automatic_moi_exec_save_sgprs = true;
  options.automatic_moi_inline_sgpr_spill = true;
  options.moi_inline_visible_evidence_sgpr = 28;
  options.moi_inline_indirect_pc_sgpr = 30;
  options.moi_inline_call_return_sgpr = 26;
  options.moi_inline_dispatch_key_sgpr = 25;
  options.moi_inline_indirect_scc_sgpr = 29;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.moi_track_barriers = true;
  options.moi_track_atomics = false;
  options.max_patches = kAccessCount + kBarrierCount;

  const ConSanResult result = try_patch_consan(
      make_gfx1250_code_object(text_words, "gfx1250_dense_barrier_reuse_fallback"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiExactShadowStore,
                               &ConSanPatchInfo::kind),
            kAccessCount);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiInlineEpochBarrier,
                               &ConSanPatchInfo::kind),
            kBarrierCount);
  const auto first_barrier = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiInlineEpochBarrier, &ConSanPatchInfo::kind);
  ASSERT_NE(first_barrier, result.patches.end());
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> anchor =
      text_words_at_offset(patched, first_barrier->anchor_offset, sizeof(uint32_t));
  ASSERT_EQ(anchor.size(), 1u);
  EXPECT_EQ((anchor.front() >> 16u) & 0x7Fu,
            30u); // The independent spill-backed barrier router uses the indirect PC pair.
}

TEST(ConSanMoi, Rdna4SharedHelperBarrierUsesCommonPrivateEpochState) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.helper_has_barrier = true;
  const std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object(fixture);

  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.moi_inline_workgroup_shadow = true;
  options.moi_track_barriers = true;
  options.moi_track_atomics = false;
  options.max_patches = 8u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  const auto access = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiExactShadowStore, &ConSanPatchInfo::kind);
  const auto barrier = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiInlineEpochBarrier, &ConSanPatchInfo::kind);
  ASSERT_NE(access, result.patches.end()) << testing::PrintToString(result.warnings);
  ASSERT_NE(barrier, result.patches.end()) << testing::PrintToString(result.warnings);
  EXPECT_EQ(access->owner_descriptor_file_offsets.size(), 2u);
  EXPECT_EQ(barrier->owner_descriptor_file_offsets, access->owner_descriptor_file_offsets);
  EXPECT_EQ(barrier->persistent_epoch_private_offset, access->persistent_epoch_private_offset);
}

TEST(ConSanMoi, Gfx1250DenseInlineShadowBarriersPartitionRelayWindowsAcrossLargeKernel) {
  constexpr uint32_t kBarriersPerWindow = 9u;
  constexpr size_t kSecondWindowWord = 65'580u;
  const uint32_t filler = build_s_mov_b32(100, 100, ROCJITSU_CODE_ARCH_GFX1250);
  std::vector<uint32_t> text_words(kSecondWindowWord + 64u, filler);
  const auto append_window = [&](size_t cursor, bool include_accesses) {
    for (uint32_t index = 0; index < kBarriersPerWindow; ++index) {
      if (include_accesses) {
        text_words[cursor++] = 0xD8340000u | index * sizeof(uint32_t);
        text_words[cursor++] = 0x00000000u; // ds_store_b32 v0, v0 offset:index*4
      }
      text_words[cursor++] = *build_s_barrier_signal_all(ROCJITSU_CODE_ARCH_GFX1250);
      text_words[cursor++] = *build_s_barrier_wait_all(ROCJITSU_CODE_ARCH_GFX1250);
    }
  };
  append_window(32u, /*include_accesses=*/true);
  append_window(kSecondWindowWord, /*include_accesses=*/false);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250);
  const std::vector<uint8_t> bytes =
      make_gfx1250_code_object(text_words, "gfx1250_partitioned_dense_inline_shadow_barriers");

  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.scratch_vgpr = 82;
  options.moi_owner_vgpr = 80;
  options.moi_epoch_vgpr = 81;
  options.moi_exec_save_sgpr = 60;
  options.automatic_moi_exec_save_sgprs = true;
  options.automatic_moi_inline_sgpr_spill = true;
  options.moi_inline_visible_evidence_sgpr = 28;
  options.moi_inline_indirect_pc_sgpr = 30;
  options.moi_inline_call_return_sgpr = 26;
  options.moi_inline_dispatch_key_sgpr = 25;
  options.moi_inline_indirect_scc_sgpr = 29;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.moi_track_barriers = true;
  options.moi_track_atomics = false;
  options.max_patches = 4u * kBarriersPerWindow;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiExactShadowStore,
                               &ConSanPatchInfo::kind),
            kBarriersPerWindow);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiInlineEpochBarrier,
                               &ConSanPatchInfo::kind),
            2u * kBarriersPerWindow)
      << testing::PrintToString(result.warnings);
  EXPECT_TRUE(std::ranges::none_of(result.warnings, [](const std::string &warning) {
    return warning.find("no relocatable dense relay host") != std::string::npos;
  }));
}

TEST(ConSanMoi, Gfx1250InlineBarrierEstablishesLowBankBeforeAdjacentGuestTransition) {
  constexpr auto store = gfx1250::build_vds(gfx1250::kDsStoreB32Vds, {.addr = 0, .data0 = 1});
  constexpr uint32_t kBarrierWait = 0xBF94FFFFu;
  const std::vector<uint32_t> text_words = {
      store[0],
      store[1],
      0xBE804EC1u, // s_barrier_signal -1
      kBarrierWait,
      0xBF860001u, // The next guest block selects a nonzero VGPR bank.
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250),
  };

  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.scratch_vgpr = 82;
  options.moi_owner_vgpr = 80;
  options.moi_epoch_vgpr = 81;
  options.moi_exec_save_sgpr = 60;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.moi_track_barriers = true;
  options.moi_track_atomics = false;

  const ConSanResult result = try_patch_consan(
      make_gfx1250_code_object(text_words, "gfx1250_inline_adjacent_vgpr_bank"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  const auto barrier = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiInlineEpochBarrier, &ConSanPatchInfo::kind);
  ASSERT_NE(barrier, result.patches.end());
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> cave =
      text_words_at_offset(patched, barrier->trampoline_offset, barrier->trampoline_size);
  ASSERT_GE(cave.size(), 2u);
  EXPECT_EQ(cave[0], kBarrierWait);
  EXPECT_EQ(cave[1], *build_gfx1250_s_set_vgpr_msb_transition(0u, 0u, ROCJITSU_CODE_ARCH_GFX1250));
}

TEST(ConSanMoi, Gfx1250InlineUsesComponentLocalScalarSpillForMixedPressureOwners) {
  const auto make_owner = [](std::span<const uint16_t> live_sgprs) {
    const uint16_t dead_destination = live_sgprs.size() > 3u ? 0u : 5u;
    std::vector<uint32_t> words = {
        0xD8340000u,
        0x00000000u, // ds_store_b32 v0, v0
    };
    for (uint16_t sgpr : live_sgprs)
      words.push_back(build_s_mov_b32(dead_destination, sgpr, ROCJITSU_CODE_ARCH_GFX1250));
    words.push_back(*build_s_barrier_signal_all(ROCJITSU_CODE_ARCH_GFX1250));
    words.push_back(*build_s_barrier_wait_all(ROCJITSU_CODE_ARCH_GFX1250));
    for (uint16_t sgpr : live_sgprs)
      words.push_back(build_s_mov_b32(dead_destination, sgpr, ROCJITSU_CODE_ARCH_GFX1250));
    words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250));
    return words;
  };

  // The low-pressure owner admits the code-object-wide high window.  The
  // second owner reaches s101 and has no dead 30-SGPR Inline window, while
  // retaining a dead PC pair and SCC slot plus four fresh high registers for
  // visible evidence, dispatch key, and call return.  The union fills those
  // low holes, so only component-scoped spill can instrument both owners.
  const std::array<uint16_t, 3> low_live = {0u, 1u, 4u};
  std::vector<uint16_t> high_live;
  for (uint16_t sgpr = 0; sgpr <= 101u; ++sgpr) {
    if (sgpr != 0u && sgpr != 1u && sgpr != 4u)
      high_live.push_back(sgpr);
  }
  const std::vector<uint8_t> bytes = make_gfx1250_code_object_with_local_function(
      make_owner(low_live), make_owner(high_live), {}, kRdna4Wave64AllVgprsGranulated,
      /*function_is_kernel=*/true);

  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.moi_inline_workgroup_shadow = true;
  options.moi_track_barriers = true;
  options.moi_track_atomics = false;
  options.max_patches = 8u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.resolved_moi_transient_sgpr_assignments.size(), 1u)
      << testing::PrintToString(result.warnings);
  const ConSanMoiTransientSgprAssignment &assignment =
      result.resolved_moi_transient_sgpr_assignments.front();
  EXPECT_TRUE(assignment.spill_backed);
  EXPECT_TRUE(assignment.visible_evidence_sgpr);
  EXPECT_TRUE(assignment.indirect_pc_sgpr);
  EXPECT_TRUE(assignment.indirect_scc_sgpr);
  EXPECT_TRUE(assignment.dispatch_key_sgpr);
  EXPECT_TRUE(assignment.call_return_sgpr);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiExactShadowStore,
                               &ConSanPatchInfo::kind),
            2u);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiInlineEpochBarrier,
                               &ConSanPatchInfo::kind),
            2u);
  EXPECT_TRUE(std::ranges::all_of(result.resource_plans, [](const auto &plan) {
    return (plan.site_kind != ConSanResourceSiteKind::Access &&
            plan.site_kind != ConSanResourceSiteKind::Barrier) ||
           plan.source != ConSanRegisterAllocationSource::Unsupported;
  }));
  EXPECT_TRUE(std::ranges::any_of(result.patches, [](const ConSanPatchInfo &patch) {
    return (patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore ||
            patch.kind == ConSanPatchKind::TrampolineMoiInlineEpochBarrier) &&
           patch.required_private_segment_size > 0u;
  }));
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, Cdna4InlineUsesComponentLocalScalarSpillOutsidePreloadsAndPhysicalVcc) {
  const auto guest = build_cdna4_ds_store_b32(
      /*vaddr=*/0, /*vdata=*/0, /*byte_offset=*/0, ROCJITSU_CODE_ARCH_CDNA4);
  const auto barrier = build_cdna4_s_barrier(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(guest && barrier);
  const auto make_owner = [&](std::span<const uint16_t> live_sgprs) {
    std::vector<uint32_t> words(guest->begin(), guest->end());
    for (uint16_t sgpr : live_sgprs)
      words.push_back(build_s_mov_b32(/*sdst=*/52u, sgpr, ROCJITSU_CODE_ARCH_CDNA4));
    words.push_back(*barrier);
    for (uint16_t sgpr : live_sgprs)
      words.push_back(build_s_mov_b32(/*sdst=*/52u, sgpr, ROCJITSU_CODE_ARCH_CDNA4));
    words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4));
    return words;
  };

  const std::array<uint16_t, 2> low_live = {8u, 9u};
  std::vector<uint16_t> high_live;
  for (uint16_t sgpr = 8u; sgpr <= 85u; ++sgpr) {
    // Leave one transient PC pair and SCC slot outside every viable 30-SGPR
    // spill window. Fresh s86:s89 carry persistent dense-router state.
    if (sgpr != 52u && sgpr != 53u && sgpr != 54u)
      high_live.push_back(sgpr);
  }
  std::vector<uint8_t> bytes = make_rdna4_code_object_with_local_function(
      make_owner(low_live), make_owner(high_live), {}, kRdna4Wave64AllVgprsGranulated,
      /*function_is_kernel=*/true, /*wave32=*/false);
  mutate_elf_header(bytes,
                    [](Elf64_Ehdr &header) { header.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX950; });
  const auto configure_descriptor = [](KD &descriptor) {
    // Twelve allocation quanta give 96 decoded SGPRs and physical VCC at
    // s90:s91. The first eight SGPRs include a kernarg pointer and preload and
    // must not be borrowed even when decoded instruction liveness is empty.
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 11u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 8u);
    AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                    kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 1u);
    AMDHSA_BITS_SET(descriptor.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_LENGTH, 4u);
    AMDHSA_BITS_SET(descriptor.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_OFFSET, 0u);
  };
  mutate_kernel_descriptor(bytes, "lds_probe", configure_descriptor);
  mutate_kernel_descriptor(bytes, "lds_helper", configure_descriptor);

  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.moi_inline_workgroup_shadow = true;
  options.moi_track_barriers = true;
  options.moi_track_atomics = false;
  options.max_patches = 8u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.resolved_moi_transient_sgpr_assignments.size(), 1u)
      << testing::PrintToString(result.warnings);
  const ConSanMoiTransientSgprAssignment &assignment =
      result.resolved_moi_transient_sgpr_assignments.front();
  ASSERT_TRUE(assignment.spill_backed);
  ASSERT_TRUE(assignment.visible_evidence_sgpr);
  ASSERT_TRUE(assignment.indirect_pc_sgpr);
  ASSERT_TRUE(assignment.indirect_scc_sgpr);
  ASSERT_TRUE(assignment.dispatch_key_sgpr);
  ASSERT_TRUE(assignment.call_return_sgpr);

  constexpr uint16_t kInitializedEnd = 8u;
  constexpr uint16_t kOriginalPhysicalVcc = 90u;
  constexpr uint16_t kGrownPhysicalVcc = 98u;
  const std::array ranges = {
      std::pair{assignment.exec_save_sgpr, kConSanMoiInlineExecSaveSgprCount},
      std::pair{*assignment.visible_evidence_sgpr, uint16_t{1u}},
      std::pair{*assignment.indirect_pc_sgpr, uint16_t{2u}},
      std::pair{*assignment.indirect_scc_sgpr, uint16_t{1u}},
      std::pair{*assignment.dispatch_key_sgpr, uint16_t{1u}},
      std::pair{*assignment.call_return_sgpr, uint16_t{2u}},
  };
  for (const auto &[base, width] : ranges) {
    EXPECT_GE(base, kInitializedEnd);
    EXPECT_LE(static_cast<uint32_t>(base) + width, kGrownPhysicalVcc);
    EXPECT_FALSE(base < kOriginalPhysicalVcc + 2u && kOriginalPhysicalVcc < base + width);
    EXPECT_FALSE(base < kGrownPhysicalVcc + 2u && kGrownPhysicalVcc < base + width);
  }
  EXPECT_TRUE(std::ranges::all_of(result.resource_plans, [](const auto &plan) {
    return (plan.site_kind != ConSanResourceSiteKind::Access &&
            plan.site_kind != ConSanResourceSiteKind::Barrier) ||
           plan.source != ConSanRegisterAllocationSource::Unsupported;
  }));
  EXPECT_TRUE(std::ranges::any_of(result.patches, [](const ConSanPatchInfo &patch) {
    return (patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore ||
            patch.kind == ConSanPatchKind::TrampolineMoiInlineEpochBarrier) &&
           patch.required_private_segment_size > 0u;
  }));
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, Cdna4InlineSpillsMixedVgprSourcesThroughDynamicStackFrames) {
  const auto guest = build_cdna4_ds_store_b32(
      /*vaddr=*/0, /*vdata=*/0, /*byte_offset=*/0, ROCJITSU_CODE_ARCH_CDNA4);
  const auto barrier = build_cdna4_s_barrier(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(guest && barrier);
  const std::array<uint16_t, 3> dead_bootstrap_sgprs = {0u, 1u, 4u};
  std::vector<uint32_t> text_words(1600u, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  std::copy(guest->begin(), guest->end(), text_words.begin() + 1u);
  size_t cursor = 1u + guest->size();
  // Keep 253 of 256 ordinary VGPRs live at the first site so it must spill
  // while leaving v253:v255 available for persistent state. Place the second site
  // after those uses, where a liveness-dead scratch window is available. Both
  // sites still need the same scalar spill frame.
  for (uint16_t vgpr = 0u; vgpr < 253u; ++vgpr) {
    text_words[cursor++] =
        build_v_mov_b32_e32(/*vdst=*/0u, vector_source_vgpr(vgpr), ROCJITSU_CODE_ARCH_CDNA4);
  }
  const uint64_t dead_scratch_access_text_offset = cursor * sizeof(uint32_t);
  std::copy(guest->begin(), guest->end(), text_words.begin() + cursor);
  cursor += guest->size();
  text_words[cursor++] = *barrier;
  for (uint16_t sgpr = 0u; sgpr < 100u; ++sgpr) {
    if (std::ranges::find(dead_bootstrap_sgprs, sgpr) == dead_bootstrap_sgprs.end()) {
      text_words[cursor++] = build_s_mov_b32(/*sdst=*/0u, sgpr, ROCJITSU_CODE_ARCH_CDNA4);
    }
  }
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);
  constexpr uint32_t kCdna4Wave64AllVgprsGranulated = 31u;
  std::vector<uint8_t> bytes = make_cdna4_lds_code_object(
      text_words, "inline_full_pressure_dynamic_stack", kCdna4Wave64AllVgprsGranulated,
      /*uses_dynamic_stack=*/true, /*workgroup_id_dimension_mask=*/0u);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 13u);
    // No accumulator operand is present; v256 is the empty boundary at the
    // end of the unified allocation.
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 63u);
  });

  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.moi_inline_workgroup_shadow = true;
  options.moi_track_barriers = true;
  options.moi_track_atomics = false;
  options.max_patches = 3u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(std::ranges::any_of(result.resource_plans, [](const auto &plan) {
    return plan.site_kind == ConSanResourceSiteKind::Access &&
           plan.source == ConSanRegisterAllocationSource::SpillRequired;
  }));
  EXPECT_TRUE(std::ranges::any_of(result.resource_plans, [](const auto &plan) {
    return plan.site_kind == ConSanResourceSiteKind::Access &&
           plan.source == ConSanRegisterAllocationSource::LivenessDead;
  }));
  ASSERT_EQ(result.resolved_moi_transient_sgpr_assignments.size(), 1u);
  const ConSanMoiTransientSgprAssignment &assignment =
      result.resolved_moi_transient_sgpr_assignments.front();
  ASSERT_TRUE(assignment.spill_backed);
  EXPECT_FALSE(assignment.visible_evidence_sgpr);
  ASSERT_TRUE(assignment.indirect_pc_sgpr);
  ASSERT_TRUE(assignment.indirect_scc_sgpr);
  EXPECT_EQ(assignment.dispatch_key_sgpr, assignment.indirect_scc_sgpr);
  EXPECT_EQ(assignment.call_return_sgpr, assignment.indirect_pc_sgpr);
  const auto overlaps_dynamic_stack = [](uint16_t base, uint16_t width) {
    return base < 34u && 32u < static_cast<uint32_t>(base) + width;
  };
  EXPECT_FALSE(
      overlaps_dynamic_stack(assignment.exec_save_sgpr, kConSanMoiInlineExecSaveSgprCount));
  EXPECT_FALSE(overlaps_dynamic_stack(*assignment.indirect_pc_sgpr, 2u));
  EXPECT_FALSE(overlaps_dynamic_stack(*assignment.indirect_scc_sgpr, 1u));

  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiExactShadowStore,
                               &ConSanPatchInfo::kind),
            2u);
  const auto access = std::ranges::find_if(result.patches, [&](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore &&
           patch.anchor_offset == dead_scratch_access_text_offset;
  });
  const auto barrier_patch = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiInlineEpochBarrier, &ConSanPatchInfo::kind);
  ASSERT_NE(access, result.patches.end());
  ASSERT_NE(barrier_patch, result.patches.end());
  EXPECT_EQ(access->workgroup_shadow_size, 0u);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_TRUE(access->scratch_vgpr);
  EXPECT_GT(access->spilled_vgpr_count, 0u);
  EXPECT_GT(access->required_private_segment_size, access->spilled_vgpr_count * sizeof(uint32_t));
  const std::vector<uint32_t> access_cave =
      text_words_at_offset(patched, access->trampoline_offset, access->trampoline_size);
  EXPECT_NE(
      std::ranges::find(access_cave, build_s_mov_b32(*assignment.indirect_pc_sgpr,
                                                     /*frame base=*/33u, ROCJITSU_CODE_ARCH_CDNA4)),
      access_cave.end());
  const uint32_t scalar_slot = access->spilled_vgpr_count * sizeof(uint32_t);
  const auto scalar_store = build_cdna4_scratch_store_b32_saddr(
      *access->scratch_vgpr, /*frame base=*/33u, scalar_slot, ROCJITSU_CODE_ARCH_CDNA4);
  const auto scalar_load = build_cdna4_scratch_load_b32_saddr(
      *access->scratch_vgpr, /*frame base=*/33u, scalar_slot, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(scalar_store && scalar_load);
  EXPECT_TRUE(contains_subsequence(access_cave, *scalar_store));
  EXPECT_TRUE(contains_subsequence(access_cave, *scalar_load));
  // External-shadow barriers use only the dead bootstrap PC/SCC state and
  // persistent epoch VGPR. They do not borrow the spill-backed access window.
  EXPECT_EQ(barrier_patch->spilled_vgpr_count, 0u);
  EXPECT_EQ(barrier_patch->required_private_segment_size, 0u);
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, Cdna4InlineExcludesOnlyOwnerWithoutSpillRouter) {
  const auto guest = build_cdna4_ds_store_b32(
      /*vaddr=*/0, /*vdata=*/0, /*byte_offset=*/0, ROCJITSU_CODE_ARCH_CDNA4);
  const auto barrier = build_cdna4_s_barrier(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(guest && barrier);
  const auto make_owner = [&](std::span<const uint16_t> live_sgprs) {
    std::vector<uint32_t> words(guest->begin(), guest->end());
    for (uint16_t sgpr : live_sgprs)
      words.push_back(build_s_mov_b32(/*sdst=*/89u, sgpr, ROCJITSU_CODE_ARCH_CDNA4));
    words.push_back(*barrier);
    for (uint16_t sgpr : live_sgprs)
      words.push_back(build_s_mov_b32(/*sdst=*/89u, sgpr, ROCJITSU_CODE_ARCH_CDNA4));
    words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4));
    return words;
  };

  const std::array<uint16_t, 2> low_live = {8u, 9u};
  std::vector<uint16_t> high_live;
  for (uint16_t sgpr = 8u; sgpr <= 88u; ++sgpr)
    high_live.push_back(sgpr);
  std::vector<uint8_t> bytes = make_rdna4_code_object_with_local_function(
      make_owner(low_live), make_owner(high_live), {}, kRdna4Wave64AllVgprsGranulated,
      /*function_is_kernel=*/true, /*wave32=*/false);
  mutate_elf_header(bytes,
                    [](Elf64_Ehdr &header) { header.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX950; });
  const auto configure_descriptor = [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 11u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 8u);
    AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                    kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 1u);
    AMDHSA_BITS_SET(descriptor.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_LENGTH, 4u);
  };
  mutate_kernel_descriptor(bytes, "lds_probe", configure_descriptor);
  mutate_kernel_descriptor(bytes, "lds_helper", configure_descriptor);

  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.moi_inline_workgroup_shadow = true;
  options.moi_track_barriers = true;
  options.moi_track_atomics = false;
  options.max_patches = 8u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_EQ(result.resolved_moi_transient_sgpr_assignments.size(), 1u);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("admitted full-pressure owners with component-local scalar state") !=
           std::string::npos;
  })) << testing::PrintToString(result.warnings);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiExactShadowStore,
                               &ConSanPatchInfo::kind),
            1u);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiInlineEpochBarrier,
                               &ConSanPatchInfo::kind),
            2u);
  EXPECT_TRUE(std::ranges::any_of(result.resource_plans, [](const auto &plan) {
    return plan.site_kind == ConSanResourceSiteKind::Access &&
           plan.source == ConSanRegisterAllocationSource::Unsupported;
  }));
}

TEST(ConSanMoi, InlineShadowPreservesTwoAddressLoadAddressAliasedBySecondResult) {
  const std::array<uint32_t, 3> text_words = {
      0xD9DC1D1Cu,
      0x6800006Bu, // ds_load_2addr_b64 v[104:107], v107 offset0:28 offset1:29
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.scratch_vgpr = 138;
  options.moi_owner_vgpr = 160;
  options.moi_epoch_vgpr = 161;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.patches.size(), 1u);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().scratch_vgpr_count, 19u);

  const ConSanPatchInfo &patch = result.patches.front();
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> cave_words =
      text_words_at_offset(patched, patch.trampoline_offset, patch.trampoline_size);
  ASSERT_GE(cave_words.size(), 3u);
  EXPECT_EQ(cave_words[0], build_v_mov_b32_e32(/*vdst=*/156, vector_source_vgpr(/*vsrc=*/107),
                                               ROCJITSU_CODE_ARCH_RDNA4));
  const auto cell = build_v_lshrrev_b32_e32(
      /*vdst=*/142, scalar_positive_inline_u32(consan_moi_exact_shadow::granule_shift),
      /*vsrc1=*/142, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(cell);
  const std::array<uint32_t, 2> guest_load = {text_words[0], text_words[1]};
  const auto load_position =
      std::search(cave_words.begin(), cave_words.end(), guest_load.begin(), guest_load.end());
  const auto cell_position = std::find(cave_words.begin(), cave_words.end(), *cell);
  ASSERT_NE(load_position, cave_words.end());
  ASSERT_NE(cell_position, cave_words.end());
  EXPECT_LT(cell_position, load_position);
}

TEST(ConSanMoi, InlineShadowProbePublishesNativeLdsLoadAndSuppressesReadRead) {
  const std::array<uint32_t, 4> input_words = {
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xBFC60000u, // s_wait_dscnt
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(input_words);
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 24;
  options.moi_epoch_vgpr = 25;
  options.moi_exec_save_sgpr = 30;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_FALSE(result.elf_bytes.empty());
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const auto *text_section = patched.text_sections().front();
  ASSERT_EQ(text_section->size() % sizeof(uint32_t), 0u);
  std::vector<uint32_t> text_words(text_section->size() / sizeof(uint32_t));
  std::memcpy(text_words.data(), text_section->data(), text_section->size());

  const uint32_t read_low_literal = static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read);
  const auto mov_read_low =
      build_v_mov_b32_e64_literal(10, read_low_literal, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(mov_read_low);
  EXPECT_TRUE(contains_subsequence(text_words, *mov_read_low));

  const auto prior_kind = build_v_and_b32_e32_literal(
      /*vdst=*/12, static_cast<uint32_t>(consan_moi_exact_shadow::access_kind_mask), /*vsrc1=*/13,
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto kind_ne = build_v_cmp_ne_u32_e32_vcc(
      scalar_positive_inline_u32(static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read)),
      /*vsrc1=*/12, ROCJITSU_CODE_ARCH_RDNA4);
  const auto narrow_kind_conflict =
      build_s_and_saveexec_b64(/*sdst=*/32, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(prior_kind);
  ASSERT_TRUE(kind_ne);
  ASSERT_TRUE(narrow_kind_conflict);
  std::vector<uint32_t> expected_read_kind_filter;
  expected_read_kind_filter.insert(expected_read_kind_filter.end(), prior_kind->begin(),
                                   prior_kind->end());
  expected_read_kind_filter.push_back(*kind_ne);
  expected_read_kind_filter.push_back(*narrow_kind_conflict);
  EXPECT_TRUE(contains_subsequence(text_words, expected_read_kind_filter));
}

TEST(ConSanMoi, InlineShadowLoadPreservesConditionStateBeforeMetadataSetup) {
  // Exercises the production-like shape with the expanded version-transaction
  // scratch window and an s60 special-state window.
  const std::array<uint32_t, 4> input_words = {
      0xD8D80000u,
      0x00000000u, // ds_load_b32 v0, v0
      0xBFC60000u, // s_wait_dscnt 0
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(input_words);
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.scratch_vgpr = 24;
  options.moi_owner_vgpr = 41;
  options.moi_epoch_vgpr = 42;
  options.moi_exec_save_sgpr = 60;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.patches.size(), 1u);
  const ConSanPatchInfo &patch = result.patches.front();
  ASSERT_EQ(patch.kind, ConSanPatchKind::TrampolineMoiExactShadowStore);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const std::vector<uint32_t> cave_words =
      text_words_at_offset(patched, patch.trampoline_offset, patch.trampoline_size);

  const auto save_scc =
      build_rdna4_s_cselect_b32(/*sdst=*/70, scalar_positive_inline_u32(1),
                                scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4);
  const auto save_vcc = build_s_mov_b64(/*sdst=*/68, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
  const uint32_t save_address =
      build_v_mov_b32_e32(/*vdst=*/40, vector_source_vgpr(/*vsrc=*/0), ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(save_scc);
  ASSERT_TRUE(save_vcc);
  ASSERT_GE(cave_words.size(), 6u);
  EXPECT_EQ(cave_words[0], save_address);
  EXPECT_EQ(cave_words[1], *save_scc);
  EXPECT_EQ(cave_words[2], *save_vcc);
  EXPECT_EQ(cave_words[3], input_words[2]);
  const std::array<uint32_t, 2> guest_load = {input_words[0], input_words[1]};
  const auto load_position =
      std::search(cave_words.begin(), cave_words.end(), guest_load.begin(), guest_load.end());
  ASSERT_NE(load_position, cave_words.end());
  EXPECT_LT(cave_words.begin() + 2, load_position);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().scratch_vgpr_count, 17u);
}

TEST(ConSanMoi, InlineShadowProbeRejectsSmallExactShadowCapacity) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 20;
  options.moi_epoch_vgpr = 21;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = sizeof(ConSanMoiReportHeader) +
                                   4u * sizeof(ConSanMoiDiagnosticRecord) + 64u * sizeof(uint64_t);

  const auto result = try_patch_consan(bytes, options);

  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(consan_patch_succeeded(result));
  bool saw_capacity_warning = false;
  for (const std::string &warning : result.warnings)
    saw_capacity_warning |= warning.find("full 64 KiB LDS address range") != std::string::npos;
  EXPECT_TRUE(saw_capacity_warning) << testing::PrintToString(result.warnings);
}

TEST(ConSanMoi, SharedInlineShadowUsesOnePersistentPairForEveryOwner) {
  const std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object();
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified);
  EXPECT_TRUE(result.moi_persistent_vgprs_automatic);
  ASSERT_TRUE(result.resolved_moi_owner_vgpr);
  ASSERT_TRUE(result.resolved_moi_epoch_vgpr);
  EXPECT_EQ(std::count_if(result.patches.begin(), result.patches.end(),
                          [](const ConSanPatchInfo &patch) {
                            return patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
                          }),
            1);
  EXPECT_EQ(std::count_if(result.patches.begin(), result.patches.end(),
                          [](const ConSanPatchInfo &patch) {
                            return patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue;
                          }),
            2);
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  });
  ASSERT_NE(access, result.patches.end());
  EXPECT_EQ(access->owner_descriptor_file_offsets.size(), 2u);
  std::vector<uint64_t> prologue_anchors;
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue)
      prologue_anchors.push_back(patch.anchor_offset);
  }
  std::ranges::sort(prologue_anchors);
  EXPECT_EQ(prologue_anchors, (std::vector<uint64_t>{0u, 8u}));
}

TEST(ConSanMoi, SharedInlineShadowCapturesFullKeyAcrossDescriptorCoordinateViews) {
  std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object();
  mutate_kernel_descriptor(bytes, "shared_owner_0", [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X,
                    1u);
  });
  mutate_kernel_descriptor(bytes, "shared_owner_1", [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z,
                    1u);
  });
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.resolved_moi_workgroup_key_vgpr);
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  });
  ASSERT_NE(access, result.patches.end());
  EXPECT_EQ(access->owner_descriptor_file_offsets.size(), 2u);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue,
                               &ConSanPatchInfo::kind),
            2);
  EXPECT_FALSE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("incompatible descriptor ABI inputs") != std::string::npos;
  })) << testing::PrintToString(result.warnings);
}

TEST(ConSanMoi, SharedPrivateEpochCapturesFullKeyAcrossDescriptorCoordinateViews) {
  std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object();
  mutate_kernel_descriptor(bytes, "shared_owner_0", [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X,
                    1u);
  });
  mutate_kernel_descriptor(bytes, "shared_owner_1", [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z,
                    1u);
  });
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.force_private_epoch = true;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  });
  ASSERT_NE(access, result.patches.end());
  EXPECT_EQ(access->owner_descriptor_file_offsets.size(), 2u);
  ASSERT_TRUE(access->persistent_workgroup_key_private_offset);
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.kind != ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue)
      continue;
    EXPECT_EQ(patch.persistent_workgroup_key_private_offset,
              access->persistent_workgroup_key_private_offset);
  }
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue,
                               &ConSanPatchInfo::kind),
            2);
}

TEST(ConSanMoi, InlineShadowPlanningExcludesUnselectedResourcePlans) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.first_vgpr_granulated = 0;
  fixture.second_vgpr_granulated = 0;
  fixture.unrelated_has_lds = true;
  const std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object(fixture);
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.test_kernel_name_filter = "shared_lds_helper";
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_LT(result.resource_plans.front().required_vgpr_count, 256u);
  EXPECT_TRUE(std::ranges::any_of(result.patches, [&](const ConSanPatchInfo &patch) {
    return patch.anchor_offset == result.resource_plans.front().text_offset &&
           patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  }));
}

TEST(ConSanMoi, SharedInlineShadowUsesOnePrivateEpochLayoutForEveryOwner) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.first_private_bytes = 0;
  fixture.second_private_bytes = 20;
  const std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object(fixture);
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.force_private_epoch = true;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << "candidates=" << result.moi_candidates.size()
                               << " plans=" << result.resource_plans.size()
                               << " patches=" << result.patches.size()
                               << " warnings=" << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.moi_private_epoch_automatic);
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  });
  ASSERT_NE(access, result.patches.end());
  EXPECT_EQ(access->persistent_epoch_private_offset, 32u);
  EXPECT_FALSE(access->persistent_owner_private_offset);
  EXPECT_EQ(access->persistent_workgroup_key_private_offset, 36u);
  EXPECT_EQ(access->owner_descriptor_file_offsets.size(), 2u);
  EXPECT_EQ(std::count_if(result.patches.begin(), result.patches.end(),
                          [](const ConSanPatchInfo &patch) {
                            return patch.kind ==
                                   ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue;
                          }),
            2);
  std::vector<uint64_t> prologue_anchors;
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.kind == ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue) {
      prologue_anchors.push_back(patch.anchor_offset);
      EXPECT_EQ(patch.persistent_epoch_private_offset, 32u);
      EXPECT_FALSE(patch.persistent_owner_private_offset);
      EXPECT_EQ(patch.persistent_workgroup_key_private_offset, 36u);
      EXPECT_EQ(patch.required_private_segment_size, 60u);
    }
  }
  std::ranges::sort(prologue_anchors);
  EXPECT_EQ(prologue_anchors, (std::vector<uint64_t>{0u, 8u}));

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  for (const AmdGpuKernelInfo &kernel : patched.kernels()) {
    KD descriptor{};
    std::memcpy(&descriptor, result.elf_bytes.data() + kernel.descriptor_file_offset,
                sizeof(descriptor));
    if (kernel.name == "shared_owner_0" || kernel.name == "shared_owner_1") {
      EXPECT_EQ(descriptor.private_segment_fixed_size, 60u);
    } else if (kernel.name == "unrelated_kernel") {
      EXPECT_EQ(descriptor.private_segment_fixed_size, 0u);
    }
  }
}

TEST(ConSanMoi, InlineAbiV6LayoutIsCheckedBoundedAndNonAliasing) {
  static_assert(sizeof(ConSanMoiReportHeader) == 184);
  static_assert(sizeof(ConSanMoiInlineExactShadowSlot) == 24);
  static_assert(offsetof(ConSanMoiInlineExactShadowSlot, packed_access) == 0);
  static_assert(offsetof(ConSanMoiInlineExactShadowSlot, dispatch_id) == 8);
  static_assert(offsetof(ConSanMoiInlineExactShadowSlot, version) == 16);
  static_assert(sizeof(ConSanMoiInlineAtomicReleaseSlot) == 32);
  static_assert(sizeof(ConSanMoiInlineAcquiredEpochTokenSlot) == 56);
  static_assert(alignof(ConSanMoiInlineAcquiredEpochTokenSlot) == 8);
  static_assert(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, version) == 0);
  static_assert(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, consumer_owner_id) == 4);
  static_assert(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, producer_owner_id) == 8);
  static_assert(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, producer_epoch_plus_one) == 12);
  static_assert(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, workgroup_key) == 16);
  static_assert(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, kind) == 20);
  static_assert(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, dispatch_id) == 24);
  static_assert(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, source_release_address) == 32);
  static_assert(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, source_release_version) == 40);
  static_assert(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, consumer_epoch_plus_one) == 44);
  static_assert(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, reservation_version) == 48);
  static_assert(sizeof(ConSanMoiInlineCausalSnapshot) == 40);

  constexpr uint64_t metadata_bytes = sizeof(ConSanMoiInlineAtomicReleaseSlot) +
                                      sizeof(ConSanMoiInlineCausalSnapshot) +
                                      sizeof(ConSanMoiInlineAcquiredEpochTokenSlot);
  constexpr uint64_t one_slot_bytes =
      sizeof(ConSanMoiReportHeader) + metadata_bytes + sizeof(ConSanMoiInlineExactShadowSlot);
  constexpr auto one =
      consan_moi_inline_shadow_report_buffer_layout_for_bytes(one_slot_bytes,
                                                              /*requested_diagnostics=*/0);
  static_assert(one.valid);
  EXPECT_EQ(one.inline_atomic_release_capacity, 1u);
  EXPECT_EQ(one.inline_causal_snapshot_capacity, 1u);
  EXPECT_EQ(one.inline_acquired_epoch_token_capacity, 1u);
  EXPECT_EQ(one.exact_shadow_entry_capacity, 1u);
  EXPECT_EQ(one.inline_atomic_release_slots_offset,
            sizeof(ConSanMoiReportHeader) + sizeof(ConSanMoiInlineExactShadowSlot));
  EXPECT_EQ(one.inline_causal_snapshots_offset,
            one.inline_atomic_release_slots_offset + sizeof(ConSanMoiInlineAtomicReleaseSlot));
  EXPECT_EQ(one.inline_acquired_epoch_token_slots_offset,
            one.inline_causal_snapshots_offset + sizeof(ConSanMoiInlineCausalSnapshot));
  EXPECT_EQ(one.required_bytes, one_slot_bytes);

  constexpr auto below = consan_moi_inline_shadow_report_buffer_layout_for_bytes(
      one_slot_bytes - 1u, /*requested_diagnostics=*/0);
  EXPECT_TRUE(below.valid);
  EXPECT_EQ(below.inline_atomic_release_capacity, 0u);
  EXPECT_EQ(below.inline_causal_snapshot_capacity, 0u);
  EXPECT_EQ(below.inline_acquired_epoch_token_capacity, 0u);

  constexpr uint64_t full_metadata_bytes =
      kConSanMoiInlineShadowAtomicReleaseSlotCapacity * metadata_bytes;
  constexpr auto full = consan_moi_inline_shadow_report_buffer_layout_for_bytes(
      sizeof(ConSanMoiReportHeader) + full_metadata_bytes + sizeof(ConSanMoiInlineExactShadowSlot),
      /*requested_diagnostics=*/0);
  EXPECT_EQ(full.inline_atomic_release_capacity, kConSanMoiInlineShadowAtomicReleaseSlotCapacity);
  EXPECT_EQ(full.inline_causal_snapshot_capacity, kConSanMoiInlineShadowAtomicReleaseSlotCapacity);
  EXPECT_EQ(full.inline_acquired_epoch_token_capacity,
            kConSanMoiInlineShadowAtomicReleaseSlotCapacity);
  EXPECT_EQ(full.exact_shadow_entry_capacity, 1u);

  constexpr auto defaults = consan_moi_inline_shadow_report_buffer_layout_for_bytes(
      kConSanMoiInlineShadowDefaultReportBufferBytes);
  EXPECT_TRUE(defaults.valid);
  EXPECT_GE(defaults.exact_shadow_entry_capacity,
            kConSanMoiInlineShadowConservativeExactShadowEntries);
  EXPECT_LE(defaults.required_bytes, kConSanMoiInlineShadowDefaultReportBufferBytes);

  ConSanMoiReportHeader header = make_consan_moi_report_header(
      /*generation=*/1, /*dispatch_id=*/2, one.access_record_capacity, one.diagnostic_capacity,
      one.exact_shadow_entry_capacity, one.sampled_watchpoint_capacity, one.barrier_record_capacity,
      one.atomic_record_capacity, one.inline_atomic_release_capacity, one.fence_record_capacity,
      one.inline_acquired_epoch_token_capacity, one.inline_causal_snapshot_capacity,
      ConSanMoiEngine::InlineShadow);
  EXPECT_TRUE(consan_moi_report_layout_matches_header(header, one, ConSanMoiEngine::InlineShadow,
                                                      one_slot_bytes));
  header.inline_causal_snapshot_capacity += 1u;
  EXPECT_FALSE(consan_moi_report_layout_matches_header(header, one, ConSanMoiEngine::InlineShadow,
                                                       one_slot_bytes));
  header.inline_causal_snapshot_capacity -= 1u;
  EXPECT_FALSE(consan_moi_report_layout_matches_header(header, one, ConSanMoiEngine::InlineShadow,
                                                       one_slot_bytes - 1u));
  header.layout_flags = 0;
  EXPECT_FALSE(consan_moi_report_layout_matches_header(header, one, ConSanMoiEngine::InlineShadow,
                                                       one_slot_bytes));
}

TEST(ConSanMoi, InlineExactSnapshotRequiresStableVersionedDispatchIdentity) {
  constexpr uint64_t packed = pack_consan_moi_exact_shadow_entry(
      ConSanMoiShadowAccessKind::Write, /*owner_id=*/0, /*epoch=*/7,
      /*generation=*/19, /*instruction_offset=*/0x1234);
  constexpr uint32_t byte_provenance =
      pack_consan_moi_exact_byte_cell_provenance(/*byte_mask=*/0x6, /*representative_lane=*/63);
  constexpr uint32_t out_of_range_lane_provenance =
      (byte_provenance & ~consan_moi_exact_byte_cell::lane_plus_one_mask) |
      (65u << consan_moi_exact_byte_cell::lane_plus_one_shift);
  constexpr uint64_t dispatch = 0x123456789abcdef0ull;
  const auto classify = [](uint32_t before, uint64_t access, uint64_t dispatch_id,
                           uint32_t provenance, uint32_t after) {
    return classify_consan_moi_inline_exact_snapshot(
        {before, access, dispatch_id, provenance, after});
  };

  EXPECT_EQ(classify(0, 0, 0, 0, 0).state, ConSanMoiInlineExactSnapshotState::Empty);

  const auto stable = classify(2, packed, dispatch, byte_provenance, 2);
  EXPECT_EQ(stable.state, ConSanMoiInlineExactSnapshotState::Stable);
  EXPECT_EQ(stable.dispatch_id, dispatch);
  EXPECT_EQ(stable.version, 2u);
  EXPECT_EQ(stable.entry.kind, ConSanMoiShadowAccessKind::Write);
  EXPECT_EQ(stable.entry.owner_id, 0u);
  EXPECT_EQ(stable.entry.epoch, 7u);
  EXPECT_EQ(stable.entry.generation, 19u);
  EXPECT_EQ(stable.entry.instruction_offset, 0x1234u);
  EXPECT_EQ(stable.byte_provenance.byte_mask, 0x6u);
  EXPECT_EQ(stable.byte_provenance.byte_offset, 1u);
  EXPECT_EQ(stable.byte_provenance.byte_count, 2u);
  EXPECT_EQ(stable.byte_provenance.representative_lane, 63u);

  for (uint32_t version : {1u, 3u, std::numeric_limits<uint32_t>::max()}) {
    EXPECT_EQ(classify(version, packed, dispatch, byte_provenance, version).state,
              ConSanMoiInlineExactSnapshotState::Publishing);
  }
  for (const auto &[before, after] : {std::pair{2u, 4u}, std::pair{2u, 3u}, std::pair{1u, 2u}}) {
    EXPECT_EQ(classify(before, packed, dispatch, byte_provenance, after).state,
              ConSanMoiInlineExactSnapshotState::ChangedDuringRead);
  }
  for (const auto malformed :
       {classify(0, packed, dispatch, byte_provenance, 0), classify(2, 0, dispatch, 0, 2),
        classify(2, packed, 0, byte_provenance, 2), classify(2, packed, dispatch, 0, 2),
        classify(2, packed, dispatch, byte_provenance | (1u << 31u), 2),
        classify(2, packed, dispatch, out_of_range_lane_provenance, 2),
        classify(2, packed, dispatch,
                 pack_consan_moi_exact_byte_cell_provenance(0x6, 63) ^ (1u << 4u), 2),
        classify(2,
                 pack_consan_moi_exact_shadow_entry(ConSanMoiShadowAccessKind::Empty, 0, 0, 19, 0),
                 dispatch, byte_provenance, 2),
        classify(2,
                 pack_consan_moi_exact_shadow_entry(static_cast<ConSanMoiShadowAccessKind>(7), 0, 0,
                                                    19, 0),
                 dispatch, byte_provenance, 2),
        classify(2, pack_consan_moi_exact_shadow_entry(ConSanMoiShadowAccessKind::Read, 0, 0, 0, 0),
                 dispatch, byte_provenance, 2)}) {
    EXPECT_EQ(malformed.state, ConSanMoiInlineExactSnapshotState::Malformed);
  }

  const auto high_word_distinct = classify(4, packed, dispatch ^ (1ull << 48u), byte_provenance, 4);
  EXPECT_EQ(high_word_distinct.state, ConSanMoiInlineExactSnapshotState::Stable);
  EXPECT_NE(high_word_distinct.dispatch_id, stable.dispatch_id);
  EXPECT_EQ(classify(kConSanMoiInlineExactMaxReadyVersion, packed, dispatch, byte_provenance,
                     kConSanMoiInlineExactMaxReadyVersion)
                .state,
            ConSanMoiInlineExactSnapshotState::Stable);
}

TEST(ConSanMoi, InlineVersionedReleaseClaimIsStableAndFailClosed) {
  constexpr ConSanMoiInlineVersionedReleaseIdentity identity{/*dispatch_id=*/0x123456789ABCDEF0ull,
                                                             /*atomic_address=*/0x4000,
                                                             /*workgroup_key=*/19};
  ConSanMoiInlineVersionedReleaseState state;
  auto claim = consan_moi_inline_plan_release_claim(state, identity, /*version_after=*/0);
  ASSERT_TRUE(claim.can_publish());
  EXPECT_EQ(claim.claim, ConSanMoiInlineReleaseClaim::Empty);
  EXPECT_EQ(claim.publishing_version, 1u);
  EXPECT_EQ(consan_moi_inline_restore_release_version(claim), 0u);
  state = {consan_moi_inline_commit_release_version(claim), identity};
  EXPECT_EQ(state.version, 2u);
  EXPECT_TRUE(consan_moi_inline_release_snapshot_is_stable(2, 2));
  EXPECT_FALSE(consan_moi_inline_release_snapshot_is_stable(1, 1));
  EXPECT_FALSE(consan_moi_inline_release_snapshot_is_stable(2, 4));

  claim = consan_moi_inline_plan_release_claim(state, identity, /*version_after=*/2);
  ASSERT_TRUE(claim.can_publish());
  EXPECT_EQ(claim.claim, ConSanMoiInlineReleaseClaim::ExactReady);
  EXPECT_EQ(claim.expected_version, 2u);
  EXPECT_EQ(claim.publishing_version, 3u);
  EXPECT_EQ(consan_moi_inline_commit_release_version(claim), 4u);
  EXPECT_EQ(consan_moi_inline_restore_release_version(claim), 2u);

  auto different = identity;
  different.atomic_address += 4;
  EXPECT_EQ(consan_moi_inline_plan_release_claim(state, different, /*version_after=*/2).claim,
            ConSanMoiInlineReleaseClaim::Collision);
  EXPECT_EQ(consan_moi_inline_plan_release_claim(state, identity, /*version_after=*/4).claim,
            ConSanMoiInlineReleaseClaim::UnstableRead);
  state.version = 3;
  EXPECT_EQ(consan_moi_inline_plan_release_claim(state, identity, /*version_after=*/3).claim,
            ConSanMoiInlineReleaseClaim::Publishing);
  state.version = std::numeric_limits<uint32_t>::max() - 1u;
  EXPECT_EQ(consan_moi_inline_plan_release_claim(state, identity, state.version).claim,
            ConSanMoiInlineReleaseClaim::VersionExhausted);
  EXPECT_EQ(consan_moi_inline_plan_release_claim({},
                                                 {/*dispatch_id=*/0,
                                                  /*atomic_address=*/0x4000,
                                                  /*workgroup_key=*/19},
                                                 /*version_after=*/0)
                .claim,
            ConSanMoiInlineReleaseClaim::InvalidIdentity);

  EXPECT_TRUE(consan_moi_inline_release_claim_cas_succeeded(
      {/*claim=*/ConSanMoiInlineReleaseClaim::ExactReady,
       /*expected_version=*/2,
       /*publishing_version=*/3},
      /*returned_version=*/2));
  EXPECT_FALSE(consan_moi_inline_release_claim_cas_succeeded(
      {/*claim=*/ConSanMoiInlineReleaseClaim::ExactReady,
       /*expected_version=*/2,
       /*publishing_version=*/3},
      /*returned_version=*/4));
  for (const auto malformed :
       {ConSanMoiInlineReleaseClaimResult{/*claim=*/ConSanMoiInlineReleaseClaim::ExactReady,
                                          /*expected_version=*/2,
                                          /*publishing_version=*/5},
        ConSanMoiInlineReleaseClaimResult{
            /*claim=*/ConSanMoiInlineReleaseClaim::ExactReady,
            /*expected_version=*/std::numeric_limits<uint32_t>::max() - 1u,
            /*publishing_version=*/std::numeric_limits<uint32_t>::max()}}) {
    EXPECT_FALSE(consan_moi_inline_release_claim_is_well_formed(malformed));
    EXPECT_EQ(consan_moi_inline_commit_release_version(malformed), 0u);
    EXPECT_EQ(consan_moi_inline_restore_release_version(malformed), 0u);
  }
}

TEST(ConSanMoi, InlineVersionedReleaseSnapshotRejectsOddChangedAndMalformedEvidence) {
  ConSanMoiInlineReleaseSnapshotWords words;
  EXPECT_EQ(classify_consan_moi_inline_release_snapshot(words).state,
            ConSanMoiInlineReleaseSnapshotState::Empty);

  words.version_before = words.slot.version = words.version_after = 2;
  words.slot.owner_id = 3;
  words.slot.epoch_plus_one = 12;
  words.slot.workgroup_key = 19;
  words.slot.atomic_address = 0x4000;
  words.slot.dispatch_id = 0x123456789abcdef0ull;
  auto classified = classify_consan_moi_inline_release_snapshot(words);
  ASSERT_EQ(classified.state, ConSanMoiInlineReleaseSnapshotState::Stable);
  EXPECT_EQ(classified.release.identity.dispatch_id, words.slot.dispatch_id);
  EXPECT_EQ(classified.release.releaser_epoch_plus_one, 12u);

  words.version_before = words.slot.version = words.version_after = 3;
  EXPECT_EQ(classify_consan_moi_inline_release_snapshot(words).state,
            ConSanMoiInlineReleaseSnapshotState::Publishing);
  words.version_after = 4;
  EXPECT_EQ(classify_consan_moi_inline_release_snapshot(words).state,
            ConSanMoiInlineReleaseSnapshotState::ChangedDuringRead);

  words = {};
  words.slot.owner_id = 1;
  EXPECT_EQ(classify_consan_moi_inline_release_snapshot(words).state,
            ConSanMoiInlineReleaseSnapshotState::Malformed);
  words = {};
  words.version_before = words.slot.version = words.version_after = 2;
  words.slot.owner_id = 3;
  words.slot.epoch_plus_one = 12;
  words.slot.workgroup_key = 19;
  words.slot.atomic_address = 0x4000;
  words.slot.dispatch_id = 0x123456789abcdef0ull;
  words.snapshot.flags =
      consan_moi_inline_causal_snapshot_flag(ConSanMoiInlineCausalSnapshotFlag::SourceIncomplete);
  EXPECT_EQ(classify_consan_moi_inline_release_snapshot(words).state,
            ConSanMoiInlineReleaseSnapshotState::SourceIncomplete);
}

TEST(ConSanMoi, InlineVersionedReleaseTransactionPinsLinearizationOrder) {
  using Event = ConSanMoiInlineReleaseTransactionEvent;
  constexpr std::array release = {Event::Reserve, Event::Metadata, Event::CausalSnapshot,
                                  Event::GuestAtomic, Event::CommitReady};
  EXPECT_TRUE(consan_moi_inline_release_transaction_is_sound(release, /*claim_succeeded=*/true,
                                                             /*dynamic_acquire_semantics=*/false,
                                                             /*release_outcome=*/true,
                                                             /*outcome_dependent_release=*/false));

  constexpr std::array acquire_release = {
      Event::PriorSnapshot, Event::Reserve,        Event::GuestAtomic, Event::AcquireImport,
      Event::Metadata,      Event::CausalSnapshot, Event::CommitReady,
  };
  EXPECT_TRUE(consan_moi_inline_release_transaction_is_sound(
      acquire_release, /*claim_succeeded=*/true, /*dynamic_acquire_semantics=*/true,
      /*release_outcome=*/true, /*outcome_dependent_release=*/false));

  constexpr std::array successful_cas = {Event::Reserve, Event::GuestAtomic, Event::Metadata,
                                         Event::CausalSnapshot, Event::CommitReady};
  EXPECT_TRUE(consan_moi_inline_release_transaction_is_sound(
      successful_cas, /*claim_succeeded=*/true, /*dynamic_acquire_semantics=*/false,
      /*release_outcome=*/true, /*outcome_dependent_release=*/true));

  constexpr std::array failed_acquire_cas = {Event::PriorSnapshot, Event::Reserve,
                                             Event::GuestAtomic, Event::AcquireImport,
                                             Event::RestorePrior};
  EXPECT_TRUE(consan_moi_inline_release_transaction_is_sound(
      failed_acquire_cas, /*claim_succeeded=*/true, /*dynamic_acquire_semantics=*/true,
      /*release_outcome=*/false, /*outcome_dependent_release=*/true));
  constexpr std::array failed_relaxed_cas = {Event::Reserve, Event::GuestAtomic,
                                             Event::RestorePrior};
  EXPECT_TRUE(consan_moi_inline_release_transaction_is_sound(
      failed_relaxed_cas, /*claim_succeeded=*/true, /*dynamic_acquire_semantics=*/false,
      /*release_outcome=*/false, /*outcome_dependent_release=*/true));

  constexpr std::array failed_claim = {Event::PoisonCoverage, Event::GuestAtomic};
  EXPECT_TRUE(consan_moi_inline_release_transaction_is_sound(
      failed_claim, /*claim_succeeded=*/false, /*dynamic_acquire_semantics=*/false,
      /*release_outcome=*/true, /*outcome_dependent_release=*/false));
  constexpr std::array lost_acquire_claim = {Event::PriorSnapshot, Event::PoisonCoverage,
                                             Event::GuestAtomic};
  EXPECT_TRUE(consan_moi_inline_release_transaction_is_sound(
      lost_acquire_claim, /*claim_succeeded=*/false, /*dynamic_acquire_semantics=*/true,
      /*release_outcome=*/true, /*outcome_dependent_release=*/false));

  constexpr std::array reserve_after_guest = {Event::GuestAtomic, Event::Reserve, Event::Metadata,
                                              Event::CausalSnapshot, Event::CommitReady};
  EXPECT_FALSE(consan_moi_inline_release_transaction_is_sound(
      reserve_after_guest, true, false, true, /*outcome_dependent_release=*/false));
  constexpr std::array snapshot_before_import = {
      Event::PriorSnapshot, Event::Reserve,  Event::GuestAtomic, Event::CausalSnapshot,
      Event::AcquireImport, Event::Metadata, Event::CommitReady};
  EXPECT_FALSE(consan_moi_inline_release_transaction_is_sound(
      snapshot_before_import, true, true, true, /*outcome_dependent_release=*/false));
  constexpr std::array commit_before_metadata = {Event::Reserve, Event::GuestAtomic,
                                                 Event::CommitReady, Event::Metadata,
                                                 Event::CausalSnapshot};
  EXPECT_FALSE(consan_moi_inline_release_transaction_is_sound(
      commit_before_metadata, true, false, true, /*outcome_dependent_release=*/true));
  constexpr std::array unpoisoned_failure = {Event::GuestAtomic};
  EXPECT_FALSE(consan_moi_inline_release_transaction_is_sound(
      unpoisoned_failure, false, false, true, /*outcome_dependent_release=*/false));
  constexpr std::array failed_claim_with_metadata = {Event::PoisonCoverage, Event::Metadata,
                                                     Event::GuestAtomic};
  EXPECT_FALSE(consan_moi_inline_release_transaction_is_sound(
      failed_claim_with_metadata, false, false, true, /*outcome_dependent_release=*/false));
  EXPECT_FALSE(consan_moi_inline_release_transaction_is_sound(
      failed_relaxed_cas, true, false, false, /*outcome_dependent_release=*/false));
}

TEST(ConSanMoi, FinalValidationPinsVersionedExactShadowPublication) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  const ConSanResult valid = try_patch_consan(bytes, options);
  ASSERT_TRUE(valid.errors.empty()) << (valid.errors.empty() ? "" : valid.errors.front());
  ASSERT_TRUE(valid.modified);
  ASSERT_TRUE(valid.resolved_moi_dispatch_id_sgpr);
  ASSERT_FALSE(valid.text_sections.empty());
  const auto patch = std::ranges::find_if(valid.patches, [](const ConSanPatchInfo &item) {
    return item.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  });
  ASSERT_NE(patch, valid.patches.end());
  ASSERT_TRUE(patch->scratch_vgpr);
  EXPECT_TRUE(validate_consan_modified_elf(bytes, valid).empty());

  const size_t body_file_offset =
      valid.text_sections.front().file_offset + patch->trampoline_offset;
  ASSERT_LE(body_file_offset + patch->trampoline_size, valid.elf_bytes.size());
  std::vector<uint32_t> body(patch->trampoline_size / sizeof(uint32_t));
  std::memcpy(body.data(), valid.elf_bytes.data() + body_file_offset, patch->trampoline_size);
  const uint16_t scratch = *patch->scratch_vgpr;
  const auto version_cas = build_flat_atomic_cmpswap_b32_vaddr_vsrc_vdst(
      scratch, static_cast<uint16_t>(scratch + 14u), static_cast<uint16_t>(scratch + 14u),
      /*return_old_value=*/true, /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(version_cas);
  const auto cas_position =
      std::search(body.begin(), body.end(), version_cas->begin(), version_cas->end());
  ASSERT_NE(cas_position, body.end());

  ConSanResult wrong_atomic = valid;
  const auto atomic_add = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      scratch, static_cast<uint16_t>(scratch + 14u), static_cast<uint16_t>(scratch + 14u),
      /*return_old_value=*/true, /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(atomic_add);
  const size_t cas_byte_offset =
      static_cast<size_t>(std::distance(body.begin(), cas_position)) * sizeof(uint32_t);
  std::memcpy(wrong_atomic.elf_bytes.data() + body_file_offset + cas_byte_offset,
              atomic_add->data(), sizeof(*atomic_add));
  const std::vector<std::string> atomic_errors = validate_consan_modified_elf(bytes, wrong_atomic);
  EXPECT_TRUE(std::ranges::any_of(atomic_errors, [](const std::string &error) {
    return error.find("versioned exact-shadow publication semantics") != std::string::npos;
  }));

  const uint32_t capture_low =
      build_v_mov_b32_e32(static_cast<uint16_t>(scratch + 2u), *valid.resolved_moi_dispatch_id_sgpr,
                          ROCJITSU_CODE_ARCH_RDNA4);
  const auto capture_position = std::find(body.begin(), body.end(), capture_low);
  ASSERT_NE(capture_position, body.end());
  ConSanResult wrong_dispatch = valid;
  const uint32_t wrong_capture = build_v_mov_b32_e32(
      static_cast<uint16_t>(scratch + 2u),
      static_cast<uint16_t>(*valid.resolved_moi_dispatch_id_sgpr + 2u), ROCJITSU_CODE_ARCH_RDNA4);
  const size_t capture_byte_offset =
      static_cast<size_t>(std::distance(body.begin(), capture_position)) * sizeof(uint32_t);
  std::memcpy(wrong_dispatch.elf_bytes.data() + body_file_offset + capture_byte_offset,
              &wrong_capture, sizeof(wrong_capture));
  const std::vector<std::string> dispatch_errors =
      validate_consan_modified_elf(bytes, wrong_dispatch);
  EXPECT_TRUE(std::ranges::any_of(dispatch_errors, [](const std::string &error) {
    return error.find("versioned exact-shadow publication semantics") != std::string::npos;
  }));
}

TEST(ConSanMoi, InlineWorkgroupKeyIsExactInsideBoundedShapes) {
  const auto one_dimensional = consan_moi_inline_workgroup_key(
      12345, 0, 0, ConSanMoiInlineWorkgroupKeyShape::OneDimensional);
  ASSERT_TRUE(one_dimensional.valid);
  EXPECT_EQ(one_dimensional.value, 12346u);
  EXPECT_FALSE(consan_moi_inline_workgroup_key(consan_moi_exact_shadow::max_generation, 0, 0,
                                               ConSanMoiInlineWorkgroupKeyShape::OneDimensional)
                   .valid);
  EXPECT_FALSE(
      consan_moi_inline_workgroup_key(0, 1, 0, ConSanMoiInlineWorkgroupKeyShape::OneDimensional)
          .valid);

  const auto two_dimensional =
      consan_moi_inline_workgroup_key(17, 29, 0, ConSanMoiInlineWorkgroupKeyShape::TwoDimensional);
  ASSERT_TRUE(two_dimensional.valid);
  EXPECT_EQ(two_dimensional.value, (17u | (29u << 10u)) + 1u);
  EXPECT_NE(two_dimensional.value, consan_moi_inline_workgroup_key(
                                       18, 29, 0, ConSanMoiInlineWorkgroupKeyShape::TwoDimensional)
                                       .value);
  EXPECT_FALSE(
      consan_moi_inline_workgroup_key(1024, 0, 0, ConSanMoiInlineWorkgroupKeyShape::TwoDimensional)
          .valid);

  const auto three_dimensional = consan_moi_inline_workgroup_key(
      7, 11, 13, ConSanMoiInlineWorkgroupKeyShape::ThreeDimensional);
  ASSERT_TRUE(three_dimensional.valid);
  EXPECT_EQ(three_dimensional.value, (7u | (11u << 8u) | (13u << 14u)) + 1u);
  EXPECT_FALSE(
      consan_moi_inline_workgroup_key(0, 0, 64, ConSanMoiInlineWorkgroupKeyShape::ThreeDimensional)
          .valid);
}

TEST(ConSanMoi, FirstLightProbeUsesAppendedCaveWhenInlinePaddingIsUnavailable) {
  const std::array<uint32_t, 3> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32 v0, v0
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options = moi_options();
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 14;
  options.moi_epoch_vgpr = 15;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(non_entry_prologue_patch_count(result), 1u);
  const ConSanPatchInfo &patch = result.patches.front();
  EXPECT_EQ(patch.kind, ConSanPatchKind::TrampolineMoiAccessRecordStore);
  EXPECT_EQ(patch.anchor_offset, 0u);
  EXPECT_EQ(patch.trampoline_offset, text_words.size() * sizeof(uint32_t));
  EXPECT_EQ(patch.original_size, 2u * sizeof(uint32_t));
  EXPECT_GT(patch.trampoline_size, 2u * sizeof(uint32_t));

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  EXPECT_GE(patched.text_sections().front()->size(),
            text_words.size() * sizeof(uint32_t) + patch.trampoline_size);

  std::vector<uint32_t> actual_words(patched.text_sections().front()->size() / sizeof(uint32_t));
  std::memcpy(actual_words.data(), patched.text_sections().front()->data(),
              patched.text_sections().front()->size());
  const auto fwd =
      compute_sopp_branch_simm16(/*branch_pc=*/0, text_words.size() * sizeof(uint32_t));
  ASSERT_TRUE(fwd);
  EXPECT_EQ(actual_words[0], build_s_branch(*fwd, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(actual_words[1], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(actual_words[2], build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4));
  const std::vector<uint32_t> trampoline_words =
      text_words_at_offset(patched, patch.trampoline_offset, patch.trampoline_size);
  ASSERT_GE(trampoline_words.size(), 3u);
  EXPECT_EQ(trampoline_words[trampoline_words.size() - 3u], 0xD8340000u);
  EXPECT_EQ(trampoline_words[trampoline_words.size() - 2u], 0x00000000u);
  EXPECT_EQ(std::count(actual_words.begin(), actual_words.end(), 0xBFC60000u), 0u);
  const uint64_t return_branch_pc =
      patch.trampoline_offset + patch.trampoline_size - sizeof(uint32_t);
  const auto ret = compute_sopp_branch_simm16(return_branch_pc, 2u * sizeof(uint32_t));
  ASSERT_TRUE(ret);
  EXPECT_EQ(trampoline_words.back(), build_s_branch(*ret, ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(ConSanMoi, InlineShadowPublishesStronglyClassifiedFlatLdsCell) {
  const std::array<uint32_t, 2> kernel_words = {
      pack_sopk(/*s_call_b64=*/0x14, /*sdst=*/30, /*simm16=*/1),
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint32_t> function_words = make_padded_moi_flat_first_light_function_words();
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words);
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.flat_provenance_mode = ConSanFlatProvenanceMode::Strict;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 24;
  options.moi_epoch_vgpr = 25;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.moi_candidates.size(), 1u);
  EXPECT_EQ(result.moi_candidates.front().source, ConSanMoiCandidateSource::FlatGroup);
  const auto access_patch = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiExactShadowStore, &ConSanPatchInfo::kind);
  ASSERT_NE(access_patch, result.patches.end());

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  std::vector<uint32_t> text_words(patched.text_sections().front()->size() / sizeof(uint32_t));
  std::memcpy(text_words.data(), patched.text_sections().front()->data(),
              patched.text_sections().front()->size());
  const auto start_cell_shift = build_v_lshrrev_b32_e32(
      /*vdst=*/12, scalar_positive_inline_u32(consan_moi_exact_shadow::granule_shift),
      /*vsrc=*/0, ROCJITSU_CODE_ARCH_RDNA4);
  const auto atomic_swap = build_flat_atomic_swap_b64_vaddr_vsrc_vdst(
      /*vaddr=*/8, /*vsrc=*/10, /*vdst=*/13, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(start_cell_shift);
  ASSERT_TRUE(atomic_swap);
  EXPECT_TRUE(contains_subsequence(text_words, std::span<const uint32_t>(&*start_cell_shift, 1)));
  EXPECT_EQ(count_subsequence(text_words, *atomic_swap), 0u);
  const auto version_cas = build_flat_atomic_cmpswap_b32_vaddr_vsrc_vdst(
      /*vaddr=*/8, /*vsrc=*/22, /*vdst=*/22, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(version_cas);
  EXPECT_EQ(count_subsequence(text_words, *version_cas), 8u);
}

TEST(ConSanMoi, InlineBarrierOnlyObjectPatchesBarrierWithoutEntryPrologue) {
  constexpr uint32_t kBarrierWait = 0xBF940000u;
  const std::array<uint32_t, 2> text_words = {
      kBarrierWait,
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_track_barriers = true;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(make_rdna4_lds_code_object(text_words), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.moi_persistent_vgprs_automatic);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiExactShadowStore,
                               &ConSanPatchInfo::kind),
            0);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::InlineMoiExactShadowStore,
                               &ConSanPatchInfo::kind),
            0);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiInlineEpochBarrier,
                               &ConSanPatchInfo::kind),
            1);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue,
                               &ConSanPatchInfo::kind),
            0);
  const auto barrier_patch = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineMoiInlineEpochBarrier, &ConSanPatchInfo::kind);
  ASSERT_NE(barrier_patch, result.patches.end());
  ASSERT_TRUE(barrier_patch->scratch_vgpr);
  const auto barrier_plan =
      std::ranges::find(result.resource_plans, ConSanResourceSiteKind::Barrier,
                        &ConSanCandidateResourcePlan::site_kind);
  ASSERT_NE(barrier_plan, result.resource_plans.end());
  EXPECT_EQ(barrier_plan->scratch_vgpr_count, 3u);
  EXPECT_EQ(barrier_plan->scratch_vgpr, barrier_patch->scratch_vgpr);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> barrier_words = text_words_at_offset(
      patched, barrier_patch->trampoline_offset, barrier_patch->trampoline_size);
  const uint16_t scratch = *barrier_patch->scratch_vgpr;
  const auto visible_atomic = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/static_cast<uint16_t>(scratch + 1u), /*vsrc=*/scratch, /*vdst=*/scratch,
      /*return_old_value=*/true, /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(visible_atomic);
  EXPECT_GE(count_subsequence(barrier_words, *visible_atomic), 1u);
  EXPECT_TRUE(validate_consan_modified_elf(make_rdna4_lds_code_object(text_words), result).empty());
  const auto barrier_disposition =
      std::ranges::find(result.site_dispositions, ConSanResourceSiteKind::Barrier,
                        &ConSanSiteDispositionRecord::site_kind);
  ASSERT_NE(barrier_disposition, result.site_dispositions.end());
  EXPECT_EQ(barrier_disposition->lowering_outcome, ConSanSiteLoweringOutcome::Patched);
  EXPECT_EQ(barrier_disposition->lowering_reason, ConSanSiteLoweringReason::None);
}

TEST(ConSanMoi, InlineBarrierOnlySharedOwnerSkipsUnobservedEntryPrologue) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.unrelated_has_barrier = true;
  fixture.group_bytes = 4352u;
  const std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object(fixture);
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_inline_workgroup_shadow = true;
  options.moi_track_barriers = true;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiInlineEpochBarrier,
                               &ConSanPatchInfo::kind),
            1);
  const auto unrelated =
      std::ranges::find(result.kernels, "unrelated_kernel", &ConSanKernelInfo::name);
  ASSERT_NE(unrelated, result.kernels.end());
  const auto prologue = std::ranges::find_if(result.patches, [&](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue &&
           std::ranges::find(patch.owner_descriptor_file_offsets,
                             unrelated->descriptor_file_offset) !=
               patch.owner_descriptor_file_offsets.end();
  });
  EXPECT_EQ(prologue, result.patches.end());
  const auto barrier_disposition =
      std::ranges::find(result.site_dispositions, ConSanResourceSiteKind::Barrier,
                        &ConSanSiteDispositionRecord::site_kind);
  ASSERT_NE(barrier_disposition, result.site_dispositions.end());
  EXPECT_EQ(barrier_disposition->lowering_outcome, ConSanSiteLoweringOutcome::Patched);
  EXPECT_EQ(barrier_disposition->lowering_reason, ConSanSiteLoweringReason::None);
}

TEST(ConSanMoi, InlineShadowBarrierEpochPatchTrampolinesBarrierAndSaturatesEpoch) {
  constexpr uint32_t kBarrierWait = 0xBF940000u;
  const std::array<uint32_t, 4> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32
      kBarrierWait,
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_track_barriers = true;
  options.moi_init_owner_epoch = true;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 24;
  options.moi_epoch_vgpr = 25;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  const auto epoch_patch_it =
      std::find_if(result.patches.begin(), result.patches.end(), [](const ConSanPatchInfo &patch) {
        return patch.kind == ConSanPatchKind::TrampolineMoiInlineEpochBarrier;
      });
  ASSERT_NE(epoch_patch_it, result.patches.end());
  EXPECT_EQ(epoch_patch_it->anchor_offset, 2u * sizeof(uint32_t));
  EXPECT_EQ(epoch_patch_it->original_size, sizeof(uint32_t));
  EXPECT_EQ(epoch_patch_it->trampoline_size, 5u * sizeof(uint32_t));

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const auto *text_section = patched.text_sections().front();
  ASSERT_GE(text_section->size(),
            epoch_patch_it->trampoline_offset + epoch_patch_it->trampoline_size);

  uint32_t rewritten_barrier = 0;
  std::memcpy(&rewritten_barrier, text_section->data() + epoch_patch_it->anchor_offset,
              sizeof(rewritten_barrier));
  const auto fwd =
      compute_sopp_branch_simm16(epoch_patch_it->anchor_offset, epoch_patch_it->trampoline_offset);
  ASSERT_TRUE(fwd);
  EXPECT_EQ(rewritten_barrier, build_s_branch(*fwd, ROCJITSU_CODE_ARCH_RDNA4));

  std::array<uint32_t, 5> trampoline_words{};
  std::memcpy(trampoline_words.data(), text_section->data() + epoch_patch_it->trampoline_offset,
              epoch_patch_it->trampoline_size);
  const auto increment_epoch = build_v_add_nc_u32_e32(
      /*vdst=*/25, scalar_positive_inline_u32(1), /*vsrc1=*/25, ROCJITSU_CODE_ARCH_RDNA4);
  const auto saturate_epoch = build_v_min_u32_e32_literal(
      /*vdst=*/25, consan_moi_exact_shadow::max_epoch, /*vsrc1=*/25, ROCJITSU_CODE_ARCH_RDNA4);
  const auto ret =
      compute_sopp_branch_simm16(epoch_patch_it->trampoline_offset + 4u * sizeof(uint32_t),
                                 epoch_patch_it->anchor_offset + sizeof(uint32_t));
  ASSERT_TRUE(increment_epoch);
  ASSERT_TRUE(saturate_epoch);
  ASSERT_TRUE(ret);
  EXPECT_EQ(trampoline_words[0], kBarrierWait);
  EXPECT_EQ(trampoline_words[1], *increment_epoch);
  EXPECT_TRUE(
      std::equal(saturate_epoch->begin(), saturate_epoch->end(), trampoline_words.begin() + 2));
  EXPECT_EQ(trampoline_words[4], build_s_branch(*ret, ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(ConSanMoi, ExactShadowEntryRoundTripsMaskedFields) {
  constexpr uint64_t packed = pack_consan_moi_exact_shadow_entry(ConSanMoiShadowAccessKind::Write,
                                                                 /*owner_id=*/0x413,
                                                                 /*epoch=*/0x477,
                                                                 /*generation=*/0x1f1234,
                                                                 /*instruction_offset=*/0x9abcdef);
  constexpr ConSanMoiExactShadowEntry decoded = decode_consan_moi_exact_shadow_entry(packed);

  EXPECT_EQ(decoded.kind, ConSanMoiShadowAccessKind::Write);
  EXPECT_EQ(decoded.owner_id, 0x13u);
  EXPECT_EQ(decoded.epoch, 0x77u);
  EXPECT_EQ(decoded.generation, 0xf1234u);
  EXPECT_EQ(decoded.instruction_offset, 0xbcdefu);
}

TEST(ConSanMoi, ExactShadowConflictPredicateMatchesSubgroupContract) {
  constexpr ConSanMoiExactShadowEntry current{
      ConSanMoiShadowAccessKind::Read,
      /*owner_id=*/2,
      /*epoch=*/17,
      /*generation=*/99,
      /*instruction_offset=*/0x100,
  };
  constexpr ConSanMoiExactShadowEntry prior_write{
      ConSanMoiShadowAccessKind::Write,
      /*owner_id=*/1,
      /*epoch=*/17,
      /*generation=*/99,
      /*instruction_offset=*/0x120,
  };
  constexpr ConSanMoiExactShadowEntry same_owner{
      ConSanMoiShadowAccessKind::Write,
      /*owner_id=*/2,
      /*epoch=*/17,
      /*generation=*/99,
      /*instruction_offset=*/0x120,
  };
  constexpr ConSanMoiExactShadowEntry old_epoch{
      ConSanMoiShadowAccessKind::Write,
      /*owner_id=*/1,
      /*epoch=*/16,
      /*generation=*/99,
      /*instruction_offset=*/0x120,
  };
  constexpr ConSanMoiExactShadowEntry prior_read{
      ConSanMoiShadowAccessKind::Read,
      /*owner_id=*/1,
      /*epoch=*/17,
      /*generation=*/99,
      /*instruction_offset=*/0x120,
  };

  EXPECT_TRUE(consan_moi_exact_shadow_entries_conflict(current, prior_write));
  EXPECT_FALSE(consan_moi_exact_shadow_entries_conflict(current, same_owner));
  EXPECT_FALSE(consan_moi_exact_shadow_entries_conflict(current, old_epoch));
  EXPECT_FALSE(consan_moi_exact_shadow_entries_conflict(current, prior_read));
}

TEST(ConSanMoi, ExactByteConflictModelDistinguishesAdjacentAndOverlappingGroups) {
  ConSanMoiExactByteAccess prior{
      /*generation=*/7,
      /*owner_id=*/1,
      /*epoch=*/3,
      ConSanMoiShadowAccessKind::Write,
      /*lds_byte_offset=*/0,
      /*lds_byte_count=*/2,
      /*instruction_offset=*/0x10,
      /*lane_mask=*/0x1,
      /*exact_address_group=*/true,
  };
  ConSanMoiExactByteAccess current = prior;
  current.owner_id = 2;
  current.lds_byte_offset = 2;
  current.instruction_offset = 0x20;
  current.lane_mask = 0x2;

  EXPECT_TRUE(consan_moi_exact_byte_overlap(current, prior).empty());
  EXPECT_FALSE(consan_moi_exact_byte_accesses_conflict(current, prior));

  current.lds_byte_offset = 1;
  const ConSanMoiExactByteOverlap overlap = consan_moi_exact_byte_overlap(current, prior);
  EXPECT_EQ(overlap.byte_offset, 1u);
  EXPECT_EQ(overlap.byte_count, 1u);
  EXPECT_TRUE(consan_moi_exact_byte_accesses_conflict(current, prior));

  current.owner_id = prior.owner_id;
  current.instruction_offset = prior.instruction_offset;
  EXPECT_TRUE(consan_moi_exact_byte_accesses_conflict(current, prior));
  current.exact_address_group = false;
  EXPECT_FALSE(consan_moi_exact_byte_accesses_conflict(current, prior));
  current.exact_address_group = true;
  current.instruction_offset = 0x28;
  EXPECT_FALSE(consan_moi_exact_byte_accesses_conflict(current, prior));
}

TEST(ConSanMoi, ExactByteCellProvenanceRoundTripsBoundariesAndRejectsCorruption) {
  for (const uint32_t mask : {0x1u, 0x2u, 0x3u, 0x4u, 0x6u, 0x7u, 0x8u, 0xcu, 0xeu, 0xfu}) {
    for (const uint32_t lane : {0u, 63u}) {
      const uint32_t packed = pack_consan_moi_exact_byte_cell_provenance(mask, lane);
      const ConSanMoiExactByteCellProvenance decoded =
          decode_consan_moi_exact_byte_cell_provenance(packed);
      EXPECT_TRUE(decoded.valid);
      EXPECT_EQ(decoded.byte_mask, mask);
      EXPECT_EQ(decoded.representative_lane, lane);
      EXPECT_EQ(((1u << decoded.byte_count) - 1u) << decoded.byte_offset, mask);
      EXPECT_EQ((consan_moi_exact_byte_cell::byte_offset_by_mask_lookup >> (2u * mask)) & 0x3u,
                decoded.byte_offset);
      EXPECT_EQ((consan_moi_exact_byte_cell::byte_end_minus_one_by_mask_lookup >> (2u * mask)) &
                    0x3u,
                decoded.byte_offset + decoded.byte_count - 1u);
    }
  }

  EXPECT_EQ(pack_consan_moi_exact_byte_cell_provenance(0u, 0u), 0u);
  EXPECT_EQ(pack_consan_moi_exact_byte_cell_provenance(0x5u, 0u), 0u);
  EXPECT_EQ(pack_consan_moi_exact_byte_cell_provenance(0xfu, 64u), 0u);
  EXPECT_FALSE(decode_consan_moi_exact_byte_cell_provenance(0u).valid);
  EXPECT_FALSE(decode_consan_moi_exact_byte_cell_provenance(
                   pack_consan_moi_exact_byte_cell_provenance(0x6u, 7u) | (1u << 31u))
                   .valid);
  EXPECT_FALSE(decode_consan_moi_exact_byte_cell_provenance(
                   pack_consan_moi_exact_byte_cell_provenance(0x6u, 7u) ^ (1u << 4u))
                   .valid);
  EXPECT_FALSE(decode_consan_moi_exact_byte_cell_provenance(
                   pack_consan_moi_exact_byte_cell_provenance(0x6u, 7u) ^ (1u << 6u))
                   .valid);
  constexpr uint32_t valid_lane_63 = pack_consan_moi_exact_byte_cell_provenance(0x6u, 63u);
  constexpr uint32_t encoded_lane_64 =
      (valid_lane_63 & ~consan_moi_exact_byte_cell::lane_plus_one_mask) |
      (65u << consan_moi_exact_byte_cell::lane_plus_one_shift);
  EXPECT_FALSE(decode_consan_moi_exact_byte_cell_provenance(encoded_lane_64).valid);
}

TEST(ConSanMoi, ExactByteCellMasksCoverEveryUnalignedBoundary) {
  EXPECT_EQ(consan_moi_maximum_cell_count_for_unaligned_bytes(0u), 0u);
  EXPECT_EQ(consan_moi_exact_byte_mask_for_cell(/*byte_offset=*/0u, /*byte_count=*/0u,
                                                /*relative_cell_index=*/0u),
            0u);
  EXPECT_EQ(consan_moi_maximum_cell_count_for_unaligned_bytes(std::numeric_limits<uint32_t>::max()),
            1073741825u);
  for (uint32_t byte_count = 1u; byte_count <= 16u; ++byte_count) {
    const uint32_t maximum_cell_count =
        consan_moi_maximum_cell_count_for_unaligned_bytes(byte_count);
    for (uint32_t byte_offset = 0u; byte_offset < 4u; ++byte_offset) {
      uint32_t reconstructed_count = 0u;
      for (uint32_t cell = 0u; cell < maximum_cell_count; ++cell) {
        const uint32_t mask = consan_moi_exact_byte_mask_for_cell(byte_offset, byte_count, cell);
        if (mask != 0u) {
          EXPECT_TRUE(consan_moi_exact_byte_cell_mask_is_contiguous(mask));
          reconstructed_count += std::popcount(mask);
        }
      }
      EXPECT_EQ(reconstructed_count, byte_count);
      EXPECT_EQ(consan_moi_exact_byte_mask_for_cell(byte_offset, byte_count, maximum_cell_count),
                0u);
    }
  }
}

TEST(ConSanMoi, InlineShadowLoopScratchKeepsNarrowUnalignedCrossingsUnrolled) {
  constexpr uint32_t kGranuleBytes = consan_moi_exact_shadow::granule_bytes;
  EXPECT_EQ(consan_detail::inline_shadow_loop_scratch_count(/*width_bits=*/0u, kGranuleBytes), 0u);
  EXPECT_EQ(consan_detail::inline_shadow_loop_scratch_count(/*width_bits=*/8u, kGranuleBytes), 0u);
  EXPECT_EQ(consan_detail::inline_shadow_loop_scratch_count(/*width_bits=*/16u, kGranuleBytes), 0u);
  EXPECT_EQ(consan_detail::inline_shadow_loop_scratch_count(/*width_bits=*/32u, kGranuleBytes), 0u);
  EXPECT_EQ(consan_detail::inline_shadow_loop_scratch_count(/*width_bits=*/128u, kGranuleBytes),
            2u);
}

TEST(ConSanMoi, ExactByteCellConflictMatchesSharedRangeContract) {
  constexpr ConSanMoiExactShadowEntry prior_access{
      ConSanMoiShadowAccessKind::Write,
      /*owner_id=*/1,
      /*epoch=*/3,
      /*generation=*/7,
      /*instruction_offset=*/0x10,
  };
  ConSanMoiExactShadowEntry current_access = prior_access;
  current_access.owner_id = 2;
  current_access.instruction_offset = 0x20;
  const auto prior_byte = decode_consan_moi_exact_byte_cell_provenance(
      pack_consan_moi_exact_byte_cell_provenance(0x3u, 0u));
  auto current_byte = decode_consan_moi_exact_byte_cell_provenance(
      pack_consan_moi_exact_byte_cell_provenance(0xcu, 1u));

  const ConSanMoiExactByteCellProvenance invalid_byte{};
  EXPECT_FALSE(
      consan_moi_exact_byte_cells_conflict(current_access, invalid_byte, prior_access, prior_byte));
  EXPECT_FALSE(consan_moi_exact_byte_cells_conflict(current_access, current_byte, prior_access,
                                                    invalid_byte));
  EXPECT_FALSE(
      consan_moi_exact_byte_cells_conflict(current_access, current_byte, prior_access, prior_byte));
  current_byte = decode_consan_moi_exact_byte_cell_provenance(
      pack_consan_moi_exact_byte_cell_provenance(0x6u, 1u));
  EXPECT_TRUE(
      consan_moi_exact_byte_cells_conflict(current_access, current_byte, prior_access, prior_byte));
  current_access.kind = ConSanMoiShadowAccessKind::Read;
  EXPECT_TRUE(
      consan_moi_exact_byte_cells_conflict(current_access, current_byte, prior_access, prior_byte));
  current_access.kind = ConSanMoiShadowAccessKind::Write;
  current_access.owner_id = prior_access.owner_id;
  current_access.instruction_offset = prior_access.instruction_offset;
  EXPECT_TRUE(
      consan_moi_exact_byte_cells_conflict(current_access, current_byte, prior_access, prior_byte));
  current_byte.representative_lane = prior_byte.representative_lane;
  EXPECT_FALSE(
      consan_moi_exact_byte_cells_conflict(current_access, current_byte, prior_access, prior_byte));
  current_access.owner_id = 2;
  current_access.epoch = 4;
  EXPECT_FALSE(
      consan_moi_exact_byte_cells_conflict(current_access, current_byte, prior_access, prior_byte));
}

TEST(ConSanMoi, SparseExactByteShadowReportsSameEpochConflicts) {
  ConSanMoiSparseExactByteShadow model(/*byte_capacity=*/4, /*maximum_access_count=*/2);
  const ConSanMoiExactByteAccess writer{
      /*generation=*/7,
      /*owner_id=*/1,
      /*epoch=*/3,
      ConSanMoiShadowAccessKind::Write,
      /*lds_byte_offset=*/0,
      /*lds_byte_count=*/4,
      /*instruction_offset=*/0x10,
      /*lane_mask=*/0x1,
  };
  const ConSanMoiExactByteAccess reader{
      /*generation=*/7,
      /*owner_id=*/2,
      /*epoch=*/3,
      ConSanMoiShadowAccessKind::Read,
      /*lds_byte_offset=*/0,
      /*lds_byte_count=*/4,
      /*instruction_offset=*/0x20,
      /*lane_mask=*/0x2,
  };

  const auto first = model.access(writer);
  EXPECT_FALSE(first.conflict);

  const auto second = model.access(reader);
  EXPECT_TRUE(second.conflict);
  EXPECT_FALSE(second.capacity_exhausted);
  ASSERT_TRUE(second.prior);
  EXPECT_EQ(second.prior->generation, 7u);
  EXPECT_EQ(second.prior->epoch, 3u);
  EXPECT_EQ(second.prior->owner_id, 1u);
  EXPECT_EQ(second.prior->lane_mask, 0x1u);
  EXPECT_EQ(second.prior->instruction_offset, 0x10u);
  EXPECT_EQ(second.prior->kind, ConSanMoiShadowAccessKind::Write);
}

TEST(ConSanMoi, SparseExactByteShadowTreatsDifferentEpochAsOrdered) {
  ConSanMoiSparseExactByteShadow model(/*byte_capacity=*/4, /*maximum_access_count=*/2);
  ConSanMoiExactByteAccess writer{
      /*generation=*/7,
      /*owner_id=*/1,
      /*epoch=*/3,
      ConSanMoiShadowAccessKind::Write,
      /*lds_byte_offset=*/0,
      /*lds_byte_count=*/4,
      /*instruction_offset=*/0x10,
      /*lane_mask=*/0x1,
  };
  ConSanMoiExactByteAccess reader = writer;
  reader.owner_id = 2;
  reader.epoch = 4;
  reader.kind = ConSanMoiShadowAccessKind::Read;
  reader.instruction_offset = 0x20;
  reader.lane_mask = 0x2;

  EXPECT_FALSE(model.access(writer).conflict);
  const auto second = model.access(reader);
  EXPECT_FALSE(second.conflict);
  EXPECT_FALSE(second.capacity_exhausted);
}

TEST(ConSanMoi, SparseExactByteShadowRejectsOutOfRangeAccess) {
  ConSanMoiSparseExactByteShadow model(/*byte_capacity=*/4, /*maximum_access_count=*/1);
  const ConSanMoiExactByteAccess access{
      /*generation=*/9,
      /*owner_id=*/3,
      /*epoch=*/5,
      ConSanMoiShadowAccessKind::Write,
      /*lds_byte_offset=*/8,
      /*lds_byte_count=*/4,
      /*instruction_offset=*/0x30,
      /*lane_mask=*/0x4,
  };

  const auto result = model.access(access);

  EXPECT_FALSE(result.conflict);
  EXPECT_TRUE(result.capacity_exhausted);
  EXPECT_FALSE(result.prior);
}

TEST(ConSanMoi, RecordReplaySeparatesExactShadowByWorkgroup) {
  auto make_record = [](uint32_t workgroup_x, uint32_t wave_id, ConSanMoiShadowAccessKind kind,
                        uint32_t instruction_offset) {
    ConSanMoiAccessRecord record{};
    record.workgroup_x = workgroup_x;
    record.wave_id = wave_id;
    record.lane_mask = uint64_t{1} << wave_id;
    record.instruction_offset = instruction_offset;
    record.access_kind = static_cast<uint32_t>(kind);
    record.lds_byte_count = 4;
    record.cell_count = 1;
    return record;
  };

  {
    SCOPED_TRACE("same workgroup accesses conflict");
    ConSanMoiReportHeader header = make_consan_moi_report_header(
        /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/2,
        /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/1,
        /*sampled_watchpoint_capacity=*/0);
    header.access_record_count = 2;
    std::array<ConSanMoiAccessRecord, 2> records = {
        make_record(/*workgroup_x=*/3, /*wave_id=*/1, ConSanMoiShadowAccessKind::Write,
                    /*instruction_offset=*/0x10),
        make_record(/*workgroup_x=*/3, /*wave_id=*/2, ConSanMoiShadowAccessKind::Read,
                    /*instruction_offset=*/0x20),
    };
    std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
    std::array<uint64_t, 1> shadow{};

    const ConSanMoiRecordReplayResult replay =
        consan_moi_record_replay_access_records(header, records, diagnostics, shadow);

    EXPECT_TRUE(replay.conflict);
    ASSERT_EQ(header.diagnostic_count, 1u);
    EXPECT_EQ(diagnostics[0].first_instruction_offset, 0x10u);
    EXPECT_EQ(diagnostics[0].second_instruction_offset, 0x20u);
  }

  {
    SCOPED_TRACE("different workgroup accesses do not conflict");
    ConSanMoiReportHeader header = make_consan_moi_report_header(
        /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/2,
        /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/1,
        /*sampled_watchpoint_capacity=*/0);
    header.access_record_count = 2;
    std::array<ConSanMoiAccessRecord, 2> records = {
        make_record(/*workgroup_x=*/3, /*wave_id=*/1, ConSanMoiShadowAccessKind::Write,
                    /*instruction_offset=*/0x10),
        make_record(/*workgroup_x=*/4, /*wave_id=*/2, ConSanMoiShadowAccessKind::Read,
                    /*instruction_offset=*/0x20),
    };
    std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
    std::array<uint64_t, 1> shadow{};

    const ConSanMoiRecordReplayResult replay =
        consan_moi_record_replay_access_records(header, records, diagnostics, shadow);

    EXPECT_FALSE(replay.conflict);
    EXPECT_EQ(header.diagnostic_count, 0u);
  }
}

TEST(ConSanMoi, RecordReplayMatchesHipMoiExactShadowSeeds) {
  constexpr uint32_t kProducerSite = 0x101;
  constexpr uint32_t kConsumerSite = 0x202;
  constexpr uint32_t kOverflowSite = 0x303;

  auto make_record = [](uint32_t owner, uint32_t epoch, ConSanMoiShadowAccessKind kind,
                        uint32_t lds_byte_offset, uint32_t instruction_offset) {
    ConSanMoiAccessRecord record{};
    record.generation = 7;
    record.wave_id = owner;
    record.epoch = epoch;
    record.lane_mask = uint64_t{1} << owner;
    record.instruction_offset = instruction_offset;
    record.access_kind = static_cast<uint32_t>(kind);
    record.lds_byte_offset = lds_byte_offset;
    record.lds_byte_count = 4;
    return record;
  };

  {
    SCOPED_TRACE("synchronized write/read is ordered by an epoch advance");
    ConSanMoiReportHeader header = make_consan_moi_report_header(
        /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/2,
        /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/1,
        /*sampled_watchpoint_capacity=*/0);
    header.access_record_count = 2;
    std::array<ConSanMoiAccessRecord, 2> records = {
        make_record(/*owner=*/0, /*epoch=*/0, ConSanMoiShadowAccessKind::Write,
                    /*lds_byte_offset=*/0, kProducerSite),
        make_record(/*owner=*/1, /*epoch=*/1, ConSanMoiShadowAccessKind::Read,
                    /*lds_byte_offset=*/0, kConsumerSite),
    };
    std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
    std::array<uint64_t, 1> shadow{};

    const ConSanMoiRecordReplayResult replay =
        consan_moi_record_replay_access_records(header, records, diagnostics, shadow);

    EXPECT_EQ(replay.processed_access_count, 2u);
    EXPECT_FALSE(replay.conflict);
    EXPECT_EQ(header.diagnostic_count, 0u);
    const ConSanMoiExactShadowEntry final = decode_consan_moi_exact_shadow_entry(shadow[0]);
    EXPECT_EQ(final.kind, ConSanMoiShadowAccessKind::Read);
    EXPECT_EQ(final.owner_id, 1u);
    EXPECT_EQ(final.epoch, 1u);
    EXPECT_EQ(final.instruction_offset, kConsumerSite);
  }

  {
    SCOPED_TRACE("same-epoch write/read reports an access conflict");
    ConSanMoiReportHeader header = make_consan_moi_report_header(
        /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/2,
        /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/1,
        /*sampled_watchpoint_capacity=*/0);
    header.access_record_count = 2;
    std::array<ConSanMoiAccessRecord, 2> records = {
        make_record(/*owner=*/0, /*epoch=*/0, ConSanMoiShadowAccessKind::Write,
                    /*lds_byte_offset=*/0, kProducerSite),
        make_record(/*owner=*/1, /*epoch=*/0, ConSanMoiShadowAccessKind::Read,
                    /*lds_byte_offset=*/0, kConsumerSite),
    };
    std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
    std::array<uint64_t, 1> shadow{};

    const ConSanMoiRecordReplayResult replay =
        consan_moi_record_replay_access_records(header, records, diagnostics, shadow);

    EXPECT_TRUE(replay.conflict);
    EXPECT_FALSE(replay.metadata_full);
    ASSERT_EQ(header.diagnostic_count, 1u);
    EXPECT_EQ(diagnostics[0].kind, static_cast<uint32_t>(ConSanMoiDiagnosticKind::AccessConflict));
    EXPECT_EQ(diagnostics[0].first_instruction_offset, kProducerSite);
    EXPECT_EQ(diagnostics[0].second_instruction_offset, kConsumerSite);
    EXPECT_EQ(diagnostics[0].first_owner_id, 0u);
    EXPECT_EQ(diagnostics[0].second_owner_id, 1u);
  }

  {
    SCOPED_TRACE("out-of-range offset reports metadata saturation");
    ConSanMoiReportHeader header = make_consan_moi_report_header(
        /*generation=*/7, /*dispatch_id=*/11, /*access_record_capacity=*/1,
        /*diagnostic_capacity=*/1, /*exact_shadow_entry_capacity=*/1,
        /*sampled_watchpoint_capacity=*/0);
    header.access_record_count = 1;
    std::array<ConSanMoiAccessRecord, 1> records = {
        make_record(/*owner=*/0, /*epoch=*/0, ConSanMoiShadowAccessKind::Write,
                    /*lds_byte_offset=*/8, kOverflowSite),
    };
    std::array<ConSanMoiDiagnosticRecord, 1> diagnostics{};
    std::array<uint64_t, 1> shadow{};

    const ConSanMoiRecordReplayResult replay =
        consan_moi_record_replay_access_records(header, records, diagnostics, shadow);

    EXPECT_TRUE(replay.conflict);
    EXPECT_TRUE(replay.metadata_full);
    ASSERT_EQ(header.diagnostic_count, 1u);
    EXPECT_EQ(diagnostics[0].kind, static_cast<uint32_t>(ConSanMoiDiagnosticKind::MetadataFull));
    EXPECT_EQ(diagnostics[0].second_instruction_offset, kOverflowSite);
    EXPECT_EQ(diagnostics[0].second_lds_byte_offset, 8u);
  }
}

} // namespace
} // namespace rocjitsu
