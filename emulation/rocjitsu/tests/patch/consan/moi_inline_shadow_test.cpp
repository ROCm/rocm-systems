// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"
#include "embedded_schema.h"
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
  const auto mix_workgroup_key = build_v_xor_b32_e32(12, vector_source_vgpr(/*workgroup key=*/13),
                                                     12, ROCJITSU_CODE_ARCH_RDNA4);
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
  EXPECT_EQ(count_subsequence(text_words, *version_load), 2u)
      << "both transaction paths must bracket their prior snapshot with version reads";
  EXPECT_EQ(count_subsequence(text_words, *version_cas), 4u)
      << "both transaction paths must claim odd and commit even";
  EXPECT_EQ(count_subsequence(text_words, *dispatch_store_low), 2u);
  EXPECT_EQ(count_subsequence(text_words, *dispatch_store_high), 2u);
  EXPECT_TRUE(contains_subsequence(
      text_words,
      std::array<uint32_t, 2>{
          build_v_mov_b32_e32(10, *result.resolved_moi_dispatch_id_sgpr, ROCJITSU_CODE_ARCH_RDNA4),
          build_v_mov_b32_e32(11, *result.resolved_moi_dispatch_id_sgpr + 1u,
                              ROCJITSU_CODE_ARCH_RDNA4)}));
}

TEST(ConSanMoi, Cdna4InlineShadowProbeEmitsNativeTransactions) {
  std::vector<uint32_t> text_words(1200, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
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
  EXPECT_EQ(count_subsequence(patched_words, *version_load), 2u);
  EXPECT_EQ(count_subsequence(patched_words, *version_cas), 4u);
  EXPECT_EQ(count_subsequence(patched_words, *retry_invalidate), 2u)
      << "both exact-shadow transaction paths must invalidate stale CDNA4 payload on retry";
  EXPECT_GE(std::count(patched_words.begin(), patched_words.end(), 0xbf8c0070u), 1);
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

TEST(ConSanMoi, Cdna4InlineShadowRecoversFullWindowKernargPreloadTail) {
  std::vector<uint32_t> text_words(1200, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  text_words[0] = 0xd81a0004u;
  text_words[1] = 0x00000302u; // ds_write_b32 v2, v3 offset:4
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);
  std::vector<uint8_t> bytes =
      make_cdna4_lds_code_object(text_words, "full_window_kernarg_preload");
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
      make_cdna3_lds_code_object(text_words, "full_window_kernarg_preload");
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
  EXPECT_TRUE(prologue->workgroup_shadow_lazy_initialization);
  EXPECT_EQ(prologue->workgroup_shadow_validity_size, 0u);
  // Generation-tagged CDNA4 shadows need no eager LDS clear, so both entry
  // variants now fit directly in their aligned entry windows.
  EXPECT_FALSE(prologue->dispatch_id_primary_prologue_offset);
  EXPECT_FALSE(prologue->dispatch_id_secondary_prologue_offset);
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
  EXPECT_EQ(result.resource_plans.front().scratch_vgpr_count, 18u);
  ASSERT_TRUE(result.resource_plans.front().scratch_vgpr);
  EXPECT_EQ(*result.resource_plans.front().scratch_vgpr % 2u, 0u);

  const auto patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &item) {
    return item.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  });
  ASSERT_NE(patch, result.patches.end());
  ASSERT_TRUE(patch->scratch_vgpr);
  EXPECT_EQ(patch->spilled_vgpr_count, 18u);
  ASSERT_EQ(patch->persistent_epoch_private_offset, 0u);
  ASSERT_EQ(patch->persistent_owner_private_offset, 4u);
  ASSERT_EQ(patch->persistent_workgroup_key_private_offset, 8u);
  ASSERT_EQ(patch->persistent_private_state_end, 12u);
  EXPECT_GT(patch->required_private_segment_size, 0u);
  EXPECT_GT(patch->workgroup_shadow_size, 0u);
  EXPECT_TRUE(patch->workgroup_shadow_lazy_initialization);
  EXPECT_EQ(patch->workgroup_shadow_validity_size, 0u);

  const auto prologue = std::ranges::find(
      result.patches, ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue, &ConSanPatchInfo::kind);
  ASSERT_NE(prologue, result.patches.end());
  ASSERT_EQ(prologue->persistent_workgroup_key_private_offset, 8u);
  EXPECT_EQ(prologue->persistent_private_state_end, 12u);
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
  EXPECT_TRUE(contains_subsequence(patch_words, expected_private_key_load));
  const auto local_exchange = build_cdna4_ds_storexchg_rtn_b64(
      static_cast<uint16_t>(*patch->scratch_vgpr + 6u), *patch->scratch_vgpr,
      static_cast<uint16_t>(*patch->scratch_vgpr + 2u), /*byte_offset=*/0u,
      ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(local_exchange);
  EXPECT_TRUE(contains_subsequence(patch_words, *local_exchange));

  ASSERT_TRUE(result.resolved_moi_dispatch_id_sgpr);
  const uint16_t generation_vgpr = static_cast<uint16_t>(*patch->scratch_vgpr + 5u);
  const auto mix_dispatch_low =
      ib::build_v_xor_b32(generation_vgpr, *result.resolved_moi_dispatch_id_sgpr, generation_vgpr,
                          ROCJITSU_CODE_ARCH_CDNA4);
  const auto mix_dispatch_high = ib::build_v_xor_b32(
      generation_vgpr, static_cast<uint16_t>(*result.resolved_moi_dispatch_id_sgpr + 1u),
      generation_vgpr, ROCJITSU_CODE_ARCH_CDNA4);
  const auto bound_generation =
      ib::build_v_and_b32_literal(generation_vgpr, consan_moi_exact_shadow::max_generation - 1u,
                                  generation_vgpr, ROCJITSU_CODE_ARCH_CDNA4);
  const auto make_nonzero_generation = ib::build_v_add_u32(
      generation_vgpr, scalar_positive_inline_u32(1u), generation_vgpr, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(mix_dispatch_low && mix_dispatch_high && bound_generation && make_nonzero_generation);
  EXPECT_NE(std::ranges::find(patch_words, *mix_dispatch_low), patch_words.end());
  EXPECT_NE(std::ranges::find(patch_words, *mix_dispatch_high), patch_words.end());
  EXPECT_TRUE(contains_subsequence(patch_words, *bound_generation));
  EXPECT_TRUE(contains_subsequence(patch_words, *make_nonzero_generation));

  const uint16_t first_snapshot = static_cast<uint16_t>(*patch->scratch_vgpr + 16u);
  const uint16_t address_snapshot = first_snapshot == 2u ? first_snapshot + 1u : first_snapshot;
  EXPECT_NE(address_snapshot, 2u);
  EXPECT_NE(
      std::ranges::find(patch_words, build_v_mov_b32_e32(address_snapshot, vector_source_vgpr(2u),
                                                         ROCJITSU_CODE_ARCH_CDNA4)),
      patch_words.end());
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
  ASSERT_EQ(access->persistent_owner_private_offset, 4u);
  ASSERT_EQ(access->persistent_workgroup_key_private_offset, 8u);
  EXPECT_EQ(access->persistent_private_state_end, 12u);
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

TEST(ConSanMoi, Rdna4AccessOnlyInlineShadowUsesGenerationTaggedWorkgroupLocalLdsMirror) {
  const std::array<uint32_t, 3> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32
      0xBFB00000u, // s_endpgm
  };
  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(
      text_words, "workgroup_shadow", kRdna4Wave64AllVgprsGranulated, false, false, 0, 4352u);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_VGPR_WORKITEM_ID,
                    /*x_and_y=*/1u);
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
    EXPECT_EQ(patch->required_group_segment_size, 13056u);
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
  EXPECT_EQ(descriptor.group_segment_fixed_size, 13056u);

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
  const auto global_version_claim = build_flat_atomic_cmpswap_b32_vaddr_vsrc_vdst(
      access_scratch, static_cast<uint16_t>(access_scratch + 14u),
      static_cast<uint16_t>(access_scratch + 14u), /*return_old_value=*/true,
      /*scope=*/2, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(local_exchange);
  ASSERT_TRUE(global_version_claim);
  EXPECT_TRUE(contains_subsequence(access_words, *local_exchange));
  EXPECT_EQ(count_subsequence(access_words, *global_version_claim), 0u);
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
  std::optional<size_t> common_restore_index;
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
    if (common_restore_index)
      EXPECT_EQ(target, *common_restore_index);
    else
      common_restore_index = target;
    ++diagnostic_cold_path_branch_count;
  }
  EXPECT_GE(diagnostic_cold_path_branch_count, 4u);
  EXPECT_GE(undercoverage_branch_count, 1u);

  ASSERT_LE(prologue->trampoline_offset + prologue->trampoline_size, text->size());
  std::vector<uint32_t> prologue_words(prologue->trampoline_size / sizeof(uint32_t));
  std::memcpy(prologue_words.data(), text->data() + prologue->trampoline_offset,
              prologue->trampoline_size);
  const auto store_wide =
      build_ds_store_b64(/*vaddr=*/24, /*vdata=*/25, /*byte_offset=*/0, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(store_wide);
  EXPECT_FALSE(contains_subsequence(prologue_words, *store_wide));
  EXPECT_EQ(std::count(prologue_words.begin(), prologue_words.end(),
                       *build_s_barrier_signal_all(ROCJITSU_CODE_ARCH_RDNA4)),
            0);
  EXPECT_EQ(std::count(prologue_words.begin(), prologue_words.end(),
                       *build_s_barrier_wait_all(ROCJITSU_CODE_ARCH_RDNA4)),
            0);
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
  EXPECT_EQ(access->required_group_segment_size, 13056u);
  EXPECT_FALSE(access->workgroup_shadow_lazy_initialization);
  EXPECT_FALSE(access->workgroup_shadow_compact);
  EXPECT_FALSE(prologue->workgroup_shadow_lazy_initialization);

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
  EXPECT_EQ(access->required_group_segment_size, 13056u);
  EXPECT_FALSE(access->workgroup_shadow_lazy_initialization);
  EXPECT_FALSE(access->workgroup_shadow_compact);

  const auto prologue = std::ranges::find(
      result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue, &ConSanPatchInfo::kind);
  ASSERT_NE(prologue, result.patches.end());
  EXPECT_FALSE(prologue->workgroup_shadow_lazy_initialization);
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
  EXPECT_TRUE(result.moi_private_epoch_automatic);
  EXPECT_FALSE(result.resolved_moi_owner_vgpr);
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  });
  ASSERT_NE(access, result.patches.end());
  EXPECT_TRUE(access->persistent_epoch_private_offset);
  EXPECT_TRUE(access->persistent_owner_private_offset);
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
  // site. One exact transaction plus a four-cell loop keeps the complete
  // probe below half that size without reducing the covered LDS width.
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
  const auto loop_branch = std::ranges::find_if(body, [&](uint32_t word) {
    return (word & 0xffff0000u) == (*exec_loop_encoding & 0xffff0000u) &&
           static_cast<int16_t>(word) < 0;
  });
  ASSERT_NE(loop_branch, body.end());
  ASSERT_NE(loop_branch, body.begin());
  ASSERT_TRUE(result.resolved_moi_exec_save_sgpr);
  const auto masked_predicate =
      build_s_and_saveexec_b64(static_cast<uint16_t>(*result.resolved_moi_exec_save_sgpr + 14u),
                               /*vcc=*/106, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(masked_predicate);
  EXPECT_EQ(*(loop_branch - 1), *masked_predicate);
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
  }));
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
  EXPECT_LT(access->trampoline_size, 2100u)
      << "two-range local diagnostics must retain the compact record writer; cumulative growth "
         "moves large-code-object entry prologues outside the executable operating range";

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
  const auto select_initialization_x_lanes = ib::build_v_cmp_gt_u32_vcc(
      scalar_positive_inline_u32(64u), /*workitem_id_x=*/0u, ROCJITSU_CODE_ARCH_GFX1250);
  const auto legacy_first_wave = ib::build_v_cmp_gt_u32_vcc(
      scalar_positive_inline_u32(32u), /*workitem_id_x=*/0u, ROCJITSU_CODE_ARCH_GFX1250);
  const uint16_t initializer_offset = static_cast<uint16_t>(*result.resolved_moi_epoch_vgpr + 1u);
  const auto scale_x = ib::build_v_lshlrev_b32(initializer_offset, scalar_positive_inline_u32(4u),
                                               /*workitem_id_x=*/0u, ROCJITSU_CODE_ARCH_GFX1250);
  const auto add_shadow_base = ib::build_v_add_u32_literal(
      initializer_address, /*base=*/4360u, initializer_offset, ROCJITSU_CODE_ARCH_GFX1250);
  const auto advance_row = ib::build_v_add_u32(
      initializer_address, static_cast<uint16_t>(*result.resolved_moi_exec_save_sgpr + 4u),
      initializer_address, ROCJITSU_CODE_ARCH_GFX1250);
  const auto select_x_zero = ib::build_v_cmp_eq_u32_vcc(
      scalar_positive_inline_u32(0u), /*workitem_id_x=*/0u, ROCJITSU_CODE_ARCH_GFX1250);
  const auto end_address = ib::build_v_cmp_gt_u32_literal_vcc(
      /*4360-byte base + 8720-byte shadow=*/13080u, initializer_address,
      ROCJITSU_CODE_ARCH_GFX1250);
  const auto store_wide = ib::build_ds_store_b128(
      initializer_address, *result.resolved_moi_epoch_vgpr, 0u, ROCJITSU_CODE_ARCH_GFX1250);
  const auto count_lanes =
      ib::build_s_bcnt1_i32_b64(static_cast<uint16_t>(*result.resolved_moi_exec_save_sgpr + 4u),
                                /*exec=*/126u, ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(select_initialization_x_lanes);
  ASSERT_TRUE(legacy_first_wave);
  ASSERT_TRUE(scale_x);
  ASSERT_TRUE(add_shadow_base);
  ASSERT_TRUE(advance_row);
  ASSERT_TRUE(select_x_zero);
  ASSERT_TRUE(end_address);
  ASSERT_TRUE(store_wide);
  ASSERT_TRUE(count_lanes);
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
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);

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
}

TEST(ConSanMoi, Gfx1250InlineOddShadowSlotCountFallsBackToExactB64Clear) {
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
  EXPECT_TRUE(contains_subsequence(prologue_words, *store_pair));
  EXPECT_FALSE(contains_subsequence(prologue_words, *store_quad));
  EXPECT_NE(std::ranges::find(prologue_words, *scale_x), prologue_words.end());
}

TEST(ConSanMoi, Gfx1250InlineLargeLocalMirrorUsesGenerationTaggedExactCells) {
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
  EXPECT_EQ(access->workgroup_shadow_validity_base, 0u);
  EXPECT_EQ(access->workgroup_shadow_validity_size, 0u);
  EXPECT_TRUE(access->workgroup_shadow_lazy_initialization);
  EXPECT_FALSE(access->workgroup_shadow_compact);
  EXPECT_EQ(access->workgroup_shadow_compact_token, 0u);
  EXPECT_EQ(access->required_group_segment_size, 63360u);

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
  EXPECT_FALSE(contains_subsequence(body, *claim));
  EXPECT_FALSE(contains_subsequence(body, *observe));

  const auto prologue = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue;
  });
  ASSERT_NE(prologue, result.patches.end());
  EXPECT_TRUE(prologue->workgroup_shadow_lazy_initialization);
  EXPECT_FALSE(prologue->workgroup_shadow_compact);
  EXPECT_EQ(prologue->workgroup_shadow_validity_base, 0u);
  EXPECT_EQ(prologue->workgroup_shadow_validity_size, 0u);
  EXPECT_TRUE(validate_consan_modified_elf(bytes, result).empty());
}

TEST(ConSanMoi, Gfx1250GenerationTaggedShadowValidatesAtomicTokenWithWorkgroupKey) {
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
  ASSERT_EQ(access->workgroup_shadow_validity_size, 0u);
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
      static_cast<uint16_t>(*access->scratch_vgpr + 8u), ROCJITSU_CODE_ARCH_GFX1250);
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

TEST(ConSanMoi, Gfx1250GenerationTaggedShadowCoversEveryWideAccessCell) {
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
  EXPECT_EQ(access->workgroup_shadow_validity_size, 0u);

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

  const std::vector<uint32_t> prologue_words =
      text_words_at_offset(patched, prologue->trampoline_offset, prologue->trampoline_size);
  const auto owner_init =
      build_v_lshrrev_b32_e32(1, scalar_positive_inline_u32(6), 0, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(owner_init);
  ASSERT_TRUE(prologue->dispatch_id_capture_sgpr);
  ASSERT_GE(prologue_words.size(), 11u);
  EXPECT_EQ(prologue_words[0],
            build_s_mov_b32(*prologue->dispatch_id_capture_sgpr, prologue->dispatch_id_source_sgpr,
                            ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(prologue_words[1], build_s_delay_alu(kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(prologue_words[4],
            build_s_add_u32(*prologue->dispatch_id_capture_sgpr,
                            *prologue->dispatch_id_capture_sgpr, scalar_positive_inline_u32(1),
                            ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(prologue_words[6],
            build_s_addc_u32(static_cast<uint16_t>(*prologue->dispatch_id_capture_sgpr + 1u),
                             static_cast<uint16_t>(*prologue->dispatch_id_capture_sgpr + 1u),
                             scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(prologue_words[8], *owner_init);
  const auto owner_bias =
      build_v_add_nc_u32_e32(1, scalar_positive_inline_u32(1), 1, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(owner_bias);
  EXPECT_EQ(prologue_words[9], *owner_bias);
  EXPECT_EQ(prologue_words[10],
            build_v_mov_b32_e32(2, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4));
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
  const auto access = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  });
  ASSERT_NE(access, result.patches.end());
  EXPECT_EQ(access->scratch_vgpr, 3);
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
  EXPECT_EQ(patch->spilled_vgpr_count, 18u);
  EXPECT_EQ(patch->required_private_segment_size, 80u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.kernels().size(), 1u);
  KD descriptor{};
  std::memcpy(&descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  EXPECT_EQ(descriptor.private_segment_fixed_size, 80u);
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

TEST(ConSanMoi, InlineShadowSpillingIgnoresAbsentAndMalformedMetadata) {
  const std::array<uint32_t, 4> text_words = {
      build_v_mov_b32_e32(/*vdst=*/11, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4),
      0xD8340000u,
      0x00000000u, // ds_store_b32
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const auto make_bytes = [&] {
    return make_rdna4_lds_code_object(text_words, "dynamic_stack_metadata_independence",
                                      kRdna4Wave64AllVgprsGranulated,
                                      /*wave32=*/false, /*uses_dynamic_stack=*/true);
  };
  std::vector<std::pair<std::string_view, std::vector<uint8_t>>> cases;
  cases.emplace_back("absent", make_bytes());
  auto malformed = make_bytes();
  append_kernel_metadata_note(malformed, "dynamic_stack_metadata_independence",
                              /*uses_dynamic_stack=*/true, /*sgpr_count=*/0u,
                              /*private_segment_fixed_size=*/0u);
  Elf64_Ehdr header{};
  std::memcpy(&header, malformed.data(), sizeof(header));
  Elf64_Phdr note_segment{};
  std::memcpy(&note_segment, malformed.data() + header.e_phoff, sizeof(note_segment));
  malformed[note_segment.p_offset + sizeof(Elf64_Nhdr) + 8u] = 0xc1u;
  cases.emplace_back("malformed", std::move(malformed));

  for (auto &[name, bytes] : cases) {
    SCOPED_TRACE(name);
    const std::vector<uint8_t> original_note = first_note_segment_bytes(bytes);
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
    EXPECT_EQ(first_note_segment_bytes(result.elf_bytes), original_note);
  }
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
  EXPECT_EQ(result.resolved_moi_owner_sgpr, 0);
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
  const auto get_hw_id = build_s_getreg_b32(/*sdst=*/0, *hwreg, ROCJITSU_CODE_ARCH_RDNA4);
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
  EXPECT_GE(sgpr_granulated, 1u);
}

TEST(ConSanMoi, InlineShadowPrivateEpochUsesWave32OwnerShift) {
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
  ASSERT_EQ(access->persistent_owner_private_offset, 4u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const std::vector<uint32_t> words =
      text_words_at_offset(patched, access->trampoline_offset, access->trampoline_size);
  const auto owner_load = build_address_free_scratch_load_b32(
      /*vdst=*/5, *access->persistent_owner_private_offset, ROCJITSU_CODE_ARCH_RDNA4);
  const auto wait = build_s_wait_loadcnt0(ROCJITSU_CODE_ARCH_RDNA4);
  const auto owner = build_v_lshrrev_b32_e32(
      /*vdst=*/5, scalar_positive_inline_u32(5), /*vsrc1=*/5, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(owner_load);
  ASSERT_TRUE(wait);
  ASSERT_TRUE(owner);
  std::vector<uint32_t> expected_owner(owner_load->begin(), owner_load->end());
  expected_owner.push_back(*wait);
  expected_owner.push_back(*owner);
  EXPECT_TRUE(contains_subsequence(words, expected_owner));
}

TEST(ConSanMoi, Rdna4InlinePrivateEpochUsesGenerationTaggedLocalMirror) {
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
  EXPECT_FALSE(access->persistent_workgroup_key_private_offset);
  EXPECT_FALSE(prologue->persistent_workgroup_key_private_offset);
  EXPECT_EQ(access->persistent_private_state_end, 8u);
  EXPECT_EQ(prologue->persistent_private_state_end, 8u);
  EXPECT_EQ(prologue->spilled_vgpr_count, 2u);
  EXPECT_EQ(access->workgroup_shadow_base, 4352u);
  EXPECT_EQ(access->workgroup_shadow_size, 8704u);
  EXPECT_EQ(access->required_group_segment_size, 13056u);
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
  const auto shadow_offset =
      ib::build_v_lshlrev_b32(static_cast<uint16_t>(scratch + 2u), scalar_positive_inline_u32(3u),
                              /*workitem_id_x=*/0u, ROCJITSU_CODE_ARCH_RDNA4);
  const auto shadow_base = ib::build_v_add_u32_literal(
      scratch, 4352u, static_cast<uint16_t>(scratch + 2u), ROCJITSU_CODE_ARCH_RDNA4);
  const auto store_wide = build_ds_store_b64(scratch, static_cast<uint16_t>(scratch + 1u),
                                             /*byte_offset=*/0, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(shadow_offset);
  ASSERT_TRUE(shadow_base);
  ASSERT_TRUE(store_wide);
  EXPECT_EQ(std::ranges::find(words, *shadow_offset), words.end());
  EXPECT_FALSE(contains_subsequence(words, *shadow_base));
  EXPECT_FALSE(contains_subsequence(words, *store_wide));
  EXPECT_EQ(
      std::count(words.begin(), words.end(), *build_s_barrier_signal_all(ROCJITSU_CODE_ARCH_RDNA4)),
      0);
  EXPECT_EQ(
      std::count(words.begin(), words.end(), *build_s_barrier_wait_all(ROCJITSU_CODE_ARCH_RDNA4)),
      0);
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
  EXPECT_EQ(prologue->spilled_vgpr_count, 1u);
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
  const auto expect_cell_publications = [](uint32_t word0, uint32_t word1,
                                           std::string_view expected_mnemonic,
                                           uint32_t expected_static_publication_sites) {
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
    EXPECT_EQ(count_subsequence(text_words, *version_cas), 4u * expected_static_publication_sites);
  };

  expect_cell_publications(0xD9D80000u, 0x01000009u, "ds_load_b64", 1u);
  expect_cell_publications(0xDA980000u, 0x01000002u, "ds_load_u16_d16", 1u);
  expect_cell_publications(0xD8380201u, 0x00000000u, "ds_store_2addr_b32", 2u);
  expect_cell_publications(0xD9DC0201u, 0x01000009u, "ds_load_2addr_b64", 2u);
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
  EXPECT_EQ(count_subsequence(text_words, *version_cas), 4u)
      << "uniform and lane-wise publication paths must each claim odd and commit even";

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
  const auto partition_wait = build_s_wait_loadcnt0(ROCJITSU_CODE_ARCH_RDNA4);
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

  std::vector<uint32_t> expected_lane_wise_fallback = {*restore_group};
  expected_lane_wise_fallback.insert(expected_lane_wise_fallback.end(), partition_rank_lo->begin(),
                                     partition_rank_lo->end());
  expected_lane_wise_fallback.insert(expected_lane_wise_fallback.end(), partition_rank_hi->begin(),
                                     partition_rank_hi->end());
  expected_lane_wise_fallback.push_back(*first_group_lane);
  expected_lane_wise_fallback.push_back(*narrow_partition_representative);
  expected_lane_wise_fallback.push_back(*save_group);
  expected_lane_wise_fallback.push_back(*save_publishers);
  EXPECT_TRUE(contains_subsequence(text_words, expected_lane_wise_fallback));
  const auto use_uniform_group_mask =
      build_s_mov_b64(/*sdst=*/34, /*ssrc0=*/46, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(use_uniform_group_mask);
  EXPECT_NE(std::find(text_words.begin(), text_words.end(), *use_uniform_group_mask),
            text_words.end())
      << "uniform publication may attribute its representative to the metadata-identical group";
  EXPECT_EQ(count_subsequence(text_words, expected_exchange_count), 2u)
      << "each generated transaction counts only a successful committed publication";

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
  const auto narrow_conflict =
      build_s_and_saveexec_b64(/*sdst=*/32, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4);
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
  ASSERT_TRUE(narrow_conflict);
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
  owner_predicate.push_back(*owner_ne);
  owner_predicate.push_back(*narrow_conflict);
  std::vector<uint32_t> epoch_predicate;
  epoch_predicate.push_back(*prior_epoch);
  epoch_predicate.insert(epoch_predicate.end(), epoch_mask->begin(), epoch_mask->end());
  epoch_predicate.push_back(*current_epoch);
  epoch_predicate.insert(epoch_predicate.end(), current_epoch_mask->begin(),
                         current_epoch_mask->end());
  epoch_predicate.push_back(*epoch_eq);
  epoch_predicate.push_back(*narrow_same_epoch);
  EXPECT_TRUE(contains_subsequence(text_words, owner_predicate));
  EXPECT_TRUE(contains_subsequence(text_words, epoch_predicate));
  const auto epoch_position = std::search(text_words.begin(), text_words.end(),
                                          epoch_predicate.begin(), epoch_predicate.end());
  const auto owner_position = std::search(text_words.begin(), text_words.end(),
                                          owner_predicate.begin(), owner_predicate.end());
  ASSERT_NE(epoch_position, text_words.end());
  ASSERT_NE(owner_position, text_words.end());
  EXPECT_LT(owner_position, epoch_position)
      << "same-owner rejection must precede epoch extraction for conflict candidates";

  const auto count_address_lo = build_v_mov_b32_e64_literal(
      /*vdst=*/8,
      static_cast<uint32_t>(report_base + offsetof(ConSanMoiReportHeader, diagnostic_count)),
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto count_address_hi = build_v_mov_b32_e64_literal(
      /*vdst=*/9,
      static_cast<uint32_t>((report_base + offsetof(ConSanMoiReportHeader, diagnostic_count)) >>
                            32u),
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto count_one = build_v_mov_b32_e64_literal(/*vdst=*/11, 1u, ROCJITSU_CODE_ARCH_RDNA4);
  const auto count_add = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/8, /*vsrc=*/11, /*vdst=*/11, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  const auto count_wait = build_s_wait_loadcnt0(ROCJITSU_CODE_ARCH_RDNA4);
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
  EXPECT_EQ(count_subsequence(text_words, *count_add), 1u);

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
  ASSERT_TRUE(save_conflict_exec);
  ASSERT_TRUE(lane_rank_lo);
  ASSERT_TRUE(lane_rank_hi);
  ASSERT_TRUE(first_active_lane);
  ASSERT_TRUE(narrow_representative);
  std::vector<uint32_t> expected_wave_coalesced_reservation = {*save_conflict_exec};
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

  const auto slot_times_16 = build_v_lshlrev_b32_e32(
      /*vdst=*/9, scalar_positive_inline_u32(4), /*vsrc1=*/11, ROCJITSU_CODE_ARCH_RDNA4);
  const auto slot_times_64 = build_v_lshlrev_b32_e32(
      /*vdst=*/8, scalar_positive_inline_u32(6), /*vsrc1=*/11, ROCJITSU_CODE_ARCH_RDNA4);
  const auto slot_times_80 = build_v_add_nc_u32_e32(
      /*vdst=*/9, vector_source_vgpr(8), /*vsrc1=*/9, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(slot_times_16);
  ASSERT_TRUE(slot_times_64);
  ASSERT_TRUE(slot_times_80);
  const std::array<uint32_t, 3> expected_slot_stride = {
      *slot_times_16,
      *slot_times_64,
      *slot_times_80,
  };
  EXPECT_TRUE(contains_subsequence(text_words, expected_slot_stride));

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
  const std::array<uint32_t, 3> expected_partition_restore = {
      *restore_original_exec,
      *restore_vcc,
      *restore_scc,
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
}

TEST(ConSanMoi, Rdna4LargeInlineShadowCompositionRetainsPerSitePlacement) {
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
  options.max_patches = 1u;

  const ConSanResult result = try_patch_consan(
      make_rdna4_lds_code_object(text_words, "rdna4_large_inline_shadow"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("outside the qualified dense-routing composition envelope") !=
           std::string::npos;
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
            kAccessCount); // One epoch advance after each signal/wait pair completes.

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
            2u * kBarriersPerWindow);
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
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = sizeof(ConSanMoiReportHeader) +
                                   4u * sizeof(ConSanMoiDiagnosticRecord) + 64u * sizeof(uint64_t);

  const auto result = try_patch_consan(bytes, options);

  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(consan_patch_succeeded(result));
  bool saw_capacity_warning = false;
  for (const std::string &warning : result.warnings)
    saw_capacity_warning |= warning.find("full 64 KiB LDS address range") != std::string::npos;
  EXPECT_TRUE(saw_capacity_warning);
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
      EXPECT_EQ(patch.required_private_segment_size, 52u);
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
      EXPECT_EQ(descriptor.private_segment_fixed_size, 52u);
    } else if (kernel.name == "unrelated_kernel") {
      EXPECT_EQ(descriptor.private_segment_fixed_size, 0u);
    }
  }
}

TEST(ConSanMoi, InlineAbiV6LayoutIsCheckedBoundedAndNonAliasing) {
  static_assert(sizeof(ConSanMoiReportHeader) == 176);
  static_assert(sizeof(ConSanMoiInlineExactShadowSlot) == 24);
  static_assert(offsetof(ConSanMoiInlineExactShadowSlot, packed_access) == 0);
  static_assert(offsetof(ConSanMoiInlineExactShadowSlot, dispatch_id) == 8);
  static_assert(offsetof(ConSanMoiInlineExactShadowSlot, version) == 16);
  static_assert(sizeof(ConSanMoiInlineAtomicReleaseSlot) == 32);
  static_assert(sizeof(ConSanMoiInlineAcquiredEpochTokenSlot) == 48);
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
  static_assert(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, reserved) == 44);
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
  constexpr uint64_t dispatch = 0x123456789abcdef0ull;
  const auto classify = [](uint32_t before, uint64_t access, uint64_t dispatch_id,
                           uint32_t reserved, uint32_t after) {
    return classify_consan_moi_inline_exact_snapshot(
        {before, access, dispatch_id, reserved, after});
  };

  EXPECT_EQ(classify(0, 0, 0, 0, 0).state, ConSanMoiInlineExactSnapshotState::Empty);

  const auto stable = classify(2, packed, dispatch, 0, 2);
  EXPECT_EQ(stable.state, ConSanMoiInlineExactSnapshotState::Stable);
  EXPECT_EQ(stable.dispatch_id, dispatch);
  EXPECT_EQ(stable.version, 2u);
  EXPECT_EQ(stable.entry.kind, ConSanMoiShadowAccessKind::Write);
  EXPECT_EQ(stable.entry.owner_id, 0u);
  EXPECT_EQ(stable.entry.epoch, 7u);
  EXPECT_EQ(stable.entry.generation, 19u);
  EXPECT_EQ(stable.entry.instruction_offset, 0x1234u);

  for (uint32_t version : {1u, 3u, std::numeric_limits<uint32_t>::max()}) {
    EXPECT_EQ(classify(version, packed, dispatch, 0, version).state,
              ConSanMoiInlineExactSnapshotState::Publishing);
  }
  for (const auto &[before, after] : {std::pair{2u, 4u}, std::pair{2u, 3u}, std::pair{1u, 2u}}) {
    EXPECT_EQ(classify(before, packed, dispatch, 0, after).state,
              ConSanMoiInlineExactSnapshotState::ChangedDuringRead);
  }
  for (const auto malformed :
       {classify(0, packed, dispatch, 0, 0), classify(2, 0, dispatch, 0, 2),
        classify(2, packed, 0, 0, 2), classify(2, packed, dispatch, 1, 2),
        classify(2,
                 pack_consan_moi_exact_shadow_entry(ConSanMoiShadowAccessKind::Empty, 0, 0, 19, 0),
                 dispatch, 0, 2),
        classify(2,
                 pack_consan_moi_exact_shadow_entry(static_cast<ConSanMoiShadowAccessKind>(7), 0, 0,
                                                    19, 0),
                 dispatch, 0, 2),
        classify(2, pack_consan_moi_exact_shadow_entry(ConSanMoiShadowAccessKind::Read, 0, 0, 0, 0),
                 dispatch, 0, 2)}) {
    EXPECT_EQ(malformed.state, ConSanMoiInlineExactSnapshotState::Malformed);
  }

  const auto high_word_distinct = classify(4, packed, dispatch ^ (1ull << 48u), 0, 4);
  EXPECT_EQ(high_word_distinct.state, ConSanMoiInlineExactSnapshotState::Stable);
  EXPECT_NE(high_word_distinct.dispatch_id, stable.dispatch_id);
  EXPECT_EQ(classify(kConSanMoiInlineExactMaxReadyVersion, packed, dispatch, 0,
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
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  const ConSanPatchInfo &patch = result.patches.front();
  EXPECT_EQ(patch.kind, ConSanPatchKind::TrampolineMoiAccessRecordStore);
  EXPECT_EQ(patch.anchor_offset, 0u);
  EXPECT_EQ(patch.trampoline_offset, text_words.size() * sizeof(uint32_t));
  EXPECT_EQ(patch.original_size, 2u * sizeof(uint32_t));
  EXPECT_GT(patch.trampoline_size, 2u * sizeof(uint32_t));

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  EXPECT_EQ(patched.text_sections().front()->size(),
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
  EXPECT_EQ(actual_words[actual_words.size() - 3u], 0xD8340000u);
  EXPECT_EQ(actual_words[actual_words.size() - 2u], 0x00000000u);
  EXPECT_EQ(std::count(actual_words.begin(), actual_words.end(), 0xBFC60000u), 0u);
  const uint64_t return_branch_pc =
      patch.trampoline_offset + patch.trampoline_size - sizeof(uint32_t);
  const auto ret = compute_sopp_branch_simm16(return_branch_pc, 2u * sizeof(uint32_t));
  ASSERT_TRUE(ret);
  EXPECT_EQ(actual_words.back(), build_s_branch(*ret, ROCJITSU_CODE_ARCH_RDNA4));
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
  EXPECT_EQ(count_subsequence(text_words, *version_cas), 4u);
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

TEST(ConSanMoi, RecordReplayExactShadowReportsSameEpochConflicts) {
  std::array<uint64_t, 1> shadow{};
  const ConSanMoiRecordReplayAccess writer{
      /*generation=*/7,
      /*owner_id=*/1,
      /*epoch=*/3,
      ConSanMoiShadowAccessKind::Write,
      /*lds_byte_offset=*/0,
      /*lds_byte_count=*/4,
      /*start_cell=*/0,
      /*cell_count=*/1,
      /*instruction_offset=*/0x10,
      /*lane_mask=*/0x1,
  };
  const ConSanMoiRecordReplayAccess reader{
      /*generation=*/7,
      /*owner_id=*/2,
      /*epoch=*/3,
      ConSanMoiShadowAccessKind::Read,
      /*lds_byte_offset=*/0,
      /*lds_byte_count=*/4,
      /*start_cell=*/0,
      /*cell_count=*/1,
      /*instruction_offset=*/0x20,
      /*lane_mask=*/0x2,
  };

  const auto first = consan_moi_record_replay_access(shadow, writer);
  EXPECT_FALSE(first.conflict);
  EXPECT_NE(shadow[0], 0u);

  const auto second = consan_moi_record_replay_access(shadow, reader);
  EXPECT_TRUE(second.conflict);
  EXPECT_FALSE(second.metadata_full);
  EXPECT_EQ(second.diagnostic.kind, static_cast<uint32_t>(ConSanMoiDiagnosticKind::AccessConflict));
  EXPECT_EQ(second.diagnostic.backend, static_cast<uint32_t>(ConSanMoiEngine::RecordReplay));
  EXPECT_EQ(second.diagnostic.generation, 7u);
  EXPECT_EQ(second.diagnostic.epoch, 3u);
  EXPECT_EQ(second.diagnostic.first_owner_id, 1u);
  EXPECT_EQ(second.diagnostic.second_owner_id, 2u);
  EXPECT_EQ(second.diagnostic.second_lane_mask, 0x2u);
  EXPECT_EQ(second.diagnostic.first_instruction_offset, 0x10u);
  EXPECT_EQ(second.diagnostic.second_instruction_offset, 0x20u);
  EXPECT_EQ(second.diagnostic.second_access_kind,
            static_cast<uint32_t>(ConSanMoiShadowAccessKind::Read));
}

TEST(ConSanMoi, RecordReplayExactShadowTreatsDifferentEpochAsOrdered) {
  std::array<uint64_t, 1> shadow{};
  ConSanMoiRecordReplayAccess writer{
      /*generation=*/7,
      /*owner_id=*/1,
      /*epoch=*/3,
      ConSanMoiShadowAccessKind::Write,
      /*lds_byte_offset=*/0,
      /*lds_byte_count=*/4,
      /*start_cell=*/0,
      /*cell_count=*/1,
      /*instruction_offset=*/0x10,
      /*lane_mask=*/0x1,
  };
  ConSanMoiRecordReplayAccess reader = writer;
  reader.owner_id = 2;
  reader.epoch = 4;
  reader.kind = ConSanMoiShadowAccessKind::Read;
  reader.instruction_offset = 0x20;
  reader.lane_mask = 0x2;

  EXPECT_FALSE(consan_moi_record_replay_access(shadow, writer).conflict);
  const auto second = consan_moi_record_replay_access(shadow, reader);
  EXPECT_FALSE(second.conflict);
  const ConSanMoiExactShadowEntry updated = decode_consan_moi_exact_shadow_entry(shadow[0]);
  EXPECT_EQ(updated.kind, ConSanMoiShadowAccessKind::Read);
  EXPECT_EQ(updated.owner_id, 2u);
  EXPECT_EQ(updated.epoch, 4u);
}

TEST(ConSanMoi, RecordReplayExactShadowReportsMetadataFullForOutOfRangeAccess) {
  std::array<uint64_t, 1> shadow{};
  const ConSanMoiRecordReplayAccess access{
      /*generation=*/9,
      /*owner_id=*/3,
      /*epoch=*/5,
      ConSanMoiShadowAccessKind::Write,
      /*lds_byte_offset=*/8,
      /*lds_byte_count=*/4,
      /*start_cell=*/2,
      /*cell_count=*/1,
      /*instruction_offset=*/0x30,
      /*lane_mask=*/0x4,
  };

  const auto result = consan_moi_record_replay_access(shadow, access);

  EXPECT_TRUE(result.conflict);
  EXPECT_TRUE(result.metadata_full);
  EXPECT_EQ(result.diagnostic.kind, static_cast<uint32_t>(ConSanMoiDiagnosticKind::MetadataFull));
  EXPECT_EQ(result.diagnostic.second_owner_id, 3u);
  EXPECT_EQ(result.diagnostic.second_instruction_offset, 0x30u);
  EXPECT_EQ(shadow[0], 0u);
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
