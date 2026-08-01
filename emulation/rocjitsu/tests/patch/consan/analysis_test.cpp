// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

namespace rocjitsu {
namespace {

TEST(ConSan, InventoriesGfx1250VflatRawFields) {
  constexpr auto store = gfx1250::build_vflat(gfx1250::kFlatStoreB128Vflat, {.saddr = 124,
                                                                             .nv = 1,
                                                                             .scale_offset = 1,
                                                                             .sve = 1,
                                                                             .scope = 2,
                                                                             .th = 3,
                                                                             .vsrc = 7,
                                                                             .vaddr = 5,
                                                                             .ioffset = 0xFFFFFC});
  const std::array<uint32_t, 4> text_words = {store[0], store[1], store[2],
                                              build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250)};
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const ConSanResult result = try_patch_consan(make_gfx1250_code_object(text_words), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.kernels.front().flat_sites.size(), 1u);
  const ConSanFlatSite &site = result.kernels.front().flat_sites.front();
  EXPECT_EQ(site.raw_op, gfx1250::kFlatStoreB128Vflat);
  EXPECT_EQ(site.raw_saddr, 124u);
  EXPECT_EQ(site.raw_vaddr, 5u);
  EXPECT_EQ(site.raw_vsrc, 7u);
  EXPECT_EQ(site.raw_vdst, 0u);
  EXPECT_EQ(site.raw_ioffset, -4);
  EXPECT_EQ(site.raw_scope, 2u);
  EXPECT_EQ(site.raw_th, 3u);
}

TEST(ConSan, RecoversGfx1250DirectCallOwnerForSharedVflatHelper) {
  constexpr auto call = gfx1250::build_sopk(gfx1250::kSCallI64Sopk, {.simm16 = 1, .sdst = 30});
  const std::array<uint32_t, 2> kernel_words = {call[0],
                                                build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250)};
  constexpr auto store =
      gfx1250::build_vflat(gfx1250::kFlatStoreB32Vflat, {.saddr = 124, .vsrc = 2, .vaddr = 0});
  constexpr auto return_to_caller = gfx1250::build_sop1(gfx1250::kSSetPcI64Sop1, {.ssrc0 = 30});
  const std::array<uint32_t, 4> function_words = {store[0], store[1], store[2],
                                                  return_to_caller[0]};
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const ConSanResult result = try_patch_consan(
      make_gfx1250_code_object_with_local_function(kernel_words, function_words), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.functions.size(), 1u);
  ASSERT_EQ(result.functions.front().flat_sites.size(), 1u);
  const ConSanFlatSite &site = result.functions.front().flat_sites.front();
  ASSERT_EQ(site.owner_descriptor_file_offsets.size(), 1u);
  EXPECT_EQ(site.owner_descriptor_file_offsets.front(),
            result.kernels.front().descriptor_file_offset);
}

TEST(ConSan, RecoversGfx1250WideLiteralIndirectCallOwnerForSharedVflatHelper) {
  constexpr auto get_pc = gfx1250::build_sop1(gfx1250::kSGetPcI64Sop1, {.sdst = 0});
  constexpr auto add_pc =
      gfx1250::build_sop2(gfx1250::kSAddNcU64Sop2, {.ssrc0 = 0, .ssrc1 = 254, .sdst = 0});
  constexpr auto call = gfx1250::build_sop1(gfx1250::kSSwapPcI64Sop1, {.ssrc0 = 0, .sdst = 30});
  // The local function starts at byte 24. s_get_pc_i64 produces byte 4, so the
  // literal delta is 20 bytes.
  const std::array<uint32_t, 6> kernel_words = {
      get_pc[0], add_pc[0], 20, 0, call[0], build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250)};
  constexpr auto store =
      gfx1250::build_vflat(gfx1250::kFlatStoreB32Vflat, {.saddr = 124, .vsrc = 2, .vaddr = 0});
  constexpr auto return_to_caller = gfx1250::build_sop1(gfx1250::kSSetPcI64Sop1, {.ssrc0 = 30});
  const std::array<uint32_t, 4> function_words = {store[0], store[1], store[2],
                                                  return_to_caller[0]};
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const ConSanResult result = try_patch_consan(
      make_gfx1250_code_object_with_local_function(kernel_words, function_words), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.functions.size(), 1u);
  ASSERT_EQ(result.functions.front().flat_sites.size(), 1u);
  const ConSanFlatSite &site = result.functions.front().flat_sites.front();
  ASSERT_EQ(site.owner_descriptor_file_offsets.size(), 1u);
  EXPECT_EQ(site.owner_descriptor_file_offsets.front(),
            result.kernels.front().descriptor_file_offset);
}

TEST(ConSan, Gfx1250SuperColliderPreflightAllowsInventoriedCacheOperations) {
  constexpr auto load = gfx1250::build_vds(gfx1250::kDsLoadB32Vds, {.addr = 2, .vdst = 1});
  const std::array<uint32_t, 9> text_words = {
      load[0],
      load[1],
      0xEE0B007Cu,
      0x00080000u,
      0x00000000u, // global_wb scope:device
      0xEE0AC07Cu,
      0x00080000u,
      0x00000000u, // global_inv scope:device
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250),
  };
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const ConSanResult result =
      try_patch_consan(make_gfx1250_code_object(text_words, "cache_and_lds"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.kernels.size(), 1u);
  const ConSanKernelInfo &kernel = result.kernels.front();
  EXPECT_EQ(kernel.stats.lds_read_count, 1u);
  EXPECT_EQ(kernel.stats.fence_like_count, 2u);
  EXPECT_EQ(kernel.fence_sites.size(), 2u);
  EXPECT_EQ(kernel.preflight_action, ConSanPreflightAction::Candidate);
}

TEST(ConSan, Gfx1250PreflightIgnoresRegisterLaneBpermute) {
  constexpr auto load = gfx1250::build_vds(gfx1250::kDsLoadB32Vds, {.addr = 2, .vdst = 1});
  constexpr auto bpermute =
      gfx1250::build_vds(gfx1250::kDsBpermuteB32Vds, {.addr = 3, .data0 = 4, .vdst = 5});
  const std::array<uint32_t, 5> text_words = {load[0], load[1], bpermute[0], bpermute[1],
                                              build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250)};
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const ConSanResult result = try_patch_consan(make_gfx1250_code_object(text_words), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.kernels.size(), 1u);
  const ConSanKernelInfo &kernel = result.kernels.front();
  EXPECT_EQ(kernel.stats.lds_read_count, 1u);
  EXPECT_EQ(kernel.stats.ds_other_count, 0u);
  ASSERT_EQ(kernel.lds_sites.size(), 1u);
  EXPECT_EQ(kernel.lds_sites.front().mnemonic, "ds_load_b32");
  EXPECT_EQ(kernel.preflight_action, ConSanPreflightAction::Candidate);
}

TEST(ConSan, CountsFlatGlobalAndScratchMemoryInstructions) {
  const std::vector<uint8_t> bytes = make_rdna4_flat_memory_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_FALSE(result.warnings.empty());
  ASSERT_EQ(result.kernels.size(), 1u);

  const ConSanKernelInfo &kernel = result.kernels.front();
  EXPECT_TRUE(kernel.decoded);
  EXPECT_EQ(kernel.code_size, 52u);
  EXPECT_EQ(kernel.stats.instruction_count, 5u);
  EXPECT_EQ(kernel.stats.lds_read_count, 0u);
  EXPECT_EQ(kernel.stats.lds_write_count, 0u);
  EXPECT_EQ(kernel.stats.lds_atomic_count, 0u);
  EXPECT_EQ(kernel.stats.ds_other_count, 0u);
  EXPECT_EQ(kernel.stats.flat_read_count, 1u);
  EXPECT_EQ(kernel.stats.flat_write_count, 1u);
  EXPECT_EQ(kernel.stats.flat_atomic_count, 0u);
  EXPECT_EQ(kernel.stats.flat_group_hint_count, 0u);
  EXPECT_EQ(kernel.stats.flat_private_hint_count, 0u);
  EXPECT_EQ(kernel.stats.flat_maybe_group_hint_count, 0u);
  EXPECT_EQ(kernel.stats.flat_maybe_private_hint_count, 0u);
  EXPECT_EQ(kernel.stats.flat_global_hint_count, 0u);
  EXPECT_EQ(kernel.stats.flat_unknown_hint_count, 2u);
  EXPECT_EQ(kernel.stats.global_memory_count, 1u);
  EXPECT_EQ(kernel.stats.scratch_memory_count, 1u);
  EXPECT_EQ(kernel.preflight_action, ConSanPreflightAction::Skip);
  ASSERT_GE(kernel.preflight_reasons.size(), 4u);
  EXPECT_EQ(kernel.preflight_reasons[0], "no supported non-atomic LDS reads or writes");
  EXPECT_EQ(kernel.preflight_reasons[1], "flat/generic memory instructions observed: 2");
  EXPECT_EQ(kernel.preflight_reasons[2], "global memory instructions observed: 1");
  EXPECT_EQ(kernel.preflight_reasons[3], "scratch memory instructions observed: 1");
  EXPECT_TRUE(kernel.lds_sites.empty());
  ASSERT_EQ(kernel.flat_sites.size(), 2u);
  EXPECT_EQ(kernel.flat_sites[0].kind, ConSanLdsAccessKind::Read);
  EXPECT_EQ(kernel.flat_sites[0].mnemonic, "flat_load_b32");
  EXPECT_EQ(kernel.flat_sites[0].text_offset, 0u);
  EXPECT_EQ(kernel.flat_sites[0].file_offset, 0x100u);
  EXPECT_EQ(kernel.flat_sites[0].size, 12u);
  EXPECT_EQ(kernel.flat_sites[0].width_bits, 32u);
  EXPECT_EQ(kernel.flat_sites[0].address_space_hint, ConSanFlatAddressSpaceHint::Unknown);
  ASSERT_TRUE(kernel.flat_sites[0].dst_vgpr);
  ASSERT_TRUE(kernel.flat_sites[0].addr_vgpr);
  EXPECT_FALSE(kernel.flat_sites[0].data_vgpr);
  EXPECT_EQ(*kernel.flat_sites[0].dst_vgpr, 0u);
  EXPECT_LT(*kernel.flat_sites[0].addr_vgpr, 256u);
  ASSERT_TRUE(kernel.flat_sites[0].raw_saddr);
  ASSERT_TRUE(kernel.flat_sites[0].raw_vaddr);
  ASSERT_TRUE(kernel.flat_sites[0].raw_vdst);
  ASSERT_TRUE(kernel.flat_sites[0].raw_ioffset);
  EXPECT_EQ(*kernel.flat_sites[0].raw_saddr, 0u);
  EXPECT_EQ(*kernel.flat_sites[0].raw_vdst, 0u);
  EXPECT_EQ(*kernel.flat_sites[0].raw_ioffset, 0);
  EXPECT_EQ(kernel.flat_sites[1].kind, ConSanLdsAccessKind::Write);
  EXPECT_EQ(kernel.flat_sites[1].mnemonic, "flat_store_b32");
  EXPECT_EQ(kernel.flat_sites[1].text_offset, 12u);
  EXPECT_EQ(kernel.flat_sites[1].file_offset, 0x10cu);
  EXPECT_EQ(kernel.flat_sites[1].size, 12u);
  EXPECT_EQ(kernel.flat_sites[1].width_bits, 32u);
  EXPECT_EQ(kernel.flat_sites[1].address_space_hint, ConSanFlatAddressSpaceHint::Unknown);
  EXPECT_FALSE(kernel.flat_sites[1].dst_vgpr);
  ASSERT_TRUE(kernel.flat_sites[1].addr_vgpr);
  ASSERT_TRUE(kernel.flat_sites[1].data_vgpr);
  EXPECT_LT(*kernel.flat_sites[1].addr_vgpr, 256u);
  EXPECT_EQ(*kernel.flat_sites[1].data_vgpr, 0u);
  ASSERT_TRUE(kernel.flat_sites[1].raw_saddr);
  ASSERT_TRUE(kernel.flat_sites[1].raw_vaddr);
  ASSERT_TRUE(kernel.flat_sites[1].raw_vsrc);
  ASSERT_TRUE(kernel.flat_sites[1].raw_ioffset);
  EXPECT_EQ(*kernel.flat_sites[1].raw_saddr, 0u);
  EXPECT_EQ(*kernel.flat_sites[1].raw_vsrc, 0u);
  EXPECT_EQ(*kernel.flat_sites[1].raw_ioffset, 0);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
}

TEST(ConSan, ClassifiesObviousSharedBaseFlatLoad) {
  const std::array<uint32_t, 9> text_words = {
      0xBE8001EBu,                           // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u, 0x00000000u,              // v_mov_b32_e64 v0, s0
      0xD5810001u, 0x00000001u,              // v_mov_b32_e64 v1, s1
      0xEC05007Cu, 0x00000002u, 0x00000000u, // flat_load_b32 v2, v[0:1]
      0xBFB00000u,                           // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_FALSE(result.warnings.empty());
  ASSERT_EQ(result.kernels.size(), 1u);

  const ConSanKernelInfo &kernel = result.kernels.front();
  EXPECT_TRUE(kernel.decoded);
  EXPECT_EQ(kernel.code_size, 36u);
  EXPECT_EQ(kernel.stats.instruction_count, 5u);
  EXPECT_EQ(kernel.stats.flat_read_count, 1u);
  EXPECT_EQ(kernel.stats.flat_write_count, 0u);
  EXPECT_EQ(kernel.stats.flat_group_hint_count, 1u);
  EXPECT_EQ(kernel.stats.flat_maybe_group_hint_count, 0u);
  EXPECT_EQ(kernel.stats.flat_private_hint_count, 0u);
  EXPECT_EQ(kernel.stats.flat_unknown_hint_count, 0u);
  ASSERT_EQ(kernel.flat_sites.size(), 1u);
  EXPECT_EQ(kernel.flat_sites.front().kind, ConSanLdsAccessKind::Read);
  EXPECT_EQ(kernel.flat_sites.front().mnemonic, "flat_load_b32");
  EXPECT_EQ(kernel.flat_sites.front().address_space_hint, ConSanFlatAddressSpaceHint::Group);
  ASSERT_TRUE(kernel.flat_sites.front().addr_vgpr);
  EXPECT_EQ(*kernel.flat_sites.front().addr_vgpr, 0u);
  ASSERT_TRUE(kernel.flat_sites.front().dst_vgpr);
  EXPECT_EQ(*kernel.flat_sites.front().dst_vgpr, 2u);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
}

TEST(ConSan, ClassifiesHighHalfSharedBaseFlatLoadAsMaybeGroup) {
  const std::array<uint32_t, 9> text_words = {
      0xBE8001EBu,                           // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u, 0x00000080u,              // v_mov_b32_e64 v0, 0
      0xD5810001u, 0x00000001u,              // v_mov_b32_e64 v1, s1
      0xEC05007Cu, 0x00000002u, 0x00000000u, // flat_load_b32 v2, v[0:1]
      0xBFB00000u,                           // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_FALSE(result.warnings.empty());
  ASSERT_EQ(result.kernels.size(), 1u);

  const ConSanKernelInfo &kernel = result.kernels.front();
  EXPECT_EQ(kernel.stats.flat_read_count, 1u);
  EXPECT_EQ(kernel.stats.flat_group_hint_count, 0u);
  EXPECT_EQ(kernel.stats.flat_maybe_group_hint_count, 1u);
  EXPECT_EQ(kernel.stats.flat_unknown_hint_count, 0u);
  ASSERT_EQ(kernel.flat_sites.size(), 1u);
  EXPECT_EQ(kernel.flat_sites.front().address_space_hint, ConSanFlatAddressSpaceHint::MaybeGroup);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
}

TEST(ConSan, PropagatesSharedBaseThroughVectorAddCarryAddressConstruction) {
  const std::array<uint32_t, 11> text_words = {
      0xBE8E01EBu,                           // s_mov_b64 s[14:15], src_shared_base
      0xBE81000Fu,                           // s_mov_b32 s1, s15
      0xD5810000u, 0x00000080u,              // v_mov_b32_e64 v0, 0
      0xD5200100u, 0x00220001u,              // v_add_co_ci_u32_e64 v0, s1, s1, v0, s8
      0x7E020300u,                           // v_mov_b32_e32 v1, v0
      0xEC05007Cu, 0x00000002u, 0x00000000u, // flat_load_b32 v2, v[0:1]
      0xBFB00000u,                           // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.kernels.size(), 1u);

  const ConSanKernelInfo &kernel = result.kernels.front();
  EXPECT_TRUE(kernel.decoded);
  EXPECT_EQ(kernel.stats.flat_read_count, 1u);
  EXPECT_EQ(kernel.stats.flat_group_hint_count, 0u);
  EXPECT_EQ(kernel.stats.flat_maybe_group_hint_count, 1u);
  EXPECT_EQ(kernel.stats.flat_unknown_hint_count, 0u);
  ASSERT_EQ(kernel.flat_sites.size(), 1u);
  EXPECT_EQ(kernel.flat_sites.front().address_space_hint, ConSanFlatAddressSpaceHint::MaybeGroup);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
}

TEST(ConSan, ClassifiesCdnaSharedBaseAfterLowHalfVectorAdd) {
  const std::array<uint32_t, 7> text_words = {
      0xBE8E01EBu,              // s_mov_b64 s[14:15], src_shared_base
      0x7E00020Eu,              // v_mov_b32_e32 v0, s14
      0x7E02020Fu,              // v_mov_b32_e32 v1, s15
      0x68000700u,              // v_add_u32_e32 v0, v0, v3
      0xDC500000u, 0x04000000u, // flat_load_dword v4, v[0:1]
      0xBF810000u,              // s_endpgm
  };
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  for (const rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    SCOPED_TRACE(static_cast<uint32_t>(arch));
    const std::vector<uint8_t> bytes =
        arch == ROCJITSU_CODE_ARCH_CDNA3
            ? make_cdna3_lds_code_object(text_words, "cdna_low_half_add")
            : make_cdna4_lds_code_object(text_words, "cdna_low_half_add");
    const ConSanResult result = try_patch_consan(bytes, options);

    ASSERT_TRUE(consan_patch_succeeded(result));
    ASSERT_EQ(result.kernels.size(), 1u);
    ASSERT_EQ(result.kernels.front().flat_sites.size(), 1u);
    EXPECT_EQ(result.kernels.front().flat_sites.front().address_space_hint,
              ConSanFlatAddressSpaceHint::Group);
    EXPECT_EQ(result.kernels.front().stats.flat_group_hint_count, 1u);
    EXPECT_EQ(result.kernels.front().stats.flat_maybe_group_hint_count, 0u);
  }
}

TEST(ConSan, DoesNotTreatCdnaDppAddAsLaneLocalPointerArithmetic) {
  const std::array<uint32_t, 8> text_words = {
      0xBE8E01EBu,              // s_mov_b64 s[14:15], src_shared_base
      0x7E00020Eu,              // v_mov_b32_e32 v0, s14
      0x7E02020Fu,              // v_mov_b32_e32 v1, s15
      0x680006FAu, 0xFF011100u, // v_add_u32_dpp v0, v0, v3 row_shr:1
      0xDC500000u, 0x04000000u, // flat_load_dword v4, v[0:1]
      0xBF810000u,              // s_endpgm
  };
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  for (const rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    SCOPED_TRACE(static_cast<uint32_t>(arch));
    const std::vector<uint8_t> bytes =
        arch == ROCJITSU_CODE_ARCH_CDNA3
            ? make_cdna3_lds_code_object(text_words, "cdna_dpp_low_half_add")
            : make_cdna4_lds_code_object(text_words, "cdna_dpp_low_half_add");
    const ConSanResult result = try_patch_consan(bytes, options);

    ASSERT_TRUE(consan_patch_succeeded(result));
    ASSERT_EQ(result.kernels.size(), 1u);
    ASSERT_EQ(result.kernels.front().flat_sites.size(), 1u);
    EXPECT_NE(result.kernels.front().flat_sites.front().address_space_hint,
              ConSanFlatAddressSpaceHint::Group);
    EXPECT_EQ(result.kernels.front().stats.flat_group_hint_count, 0u);
  }
}

TEST(ConSan, PropagatesSharedPointerThroughExactScratchSlot) {
  const std::array<uint32_t, 13> text_words = {
      0xBE9201EBu,                           // s_mov_b64 s[18:19], src_shared_base
      0x7E020212u,                           // v_mov_b32_e32 v1, s18
      0x7E040213u,                           // v_mov_b32_e32 v2, s19
      0xED06C021u, 0x00800000u, 0x00061400u, // scratch_store_b64 off, v[1:2], s33
      0xED054021u, 0x00000003u, 0x00061400u, // scratch_load_b64 v[3:4], off, s33
      0xEC05007Cu, 0x00000005u, 0x00000003u, // flat_load_b32 v5, v[3:4]
      0xBFB00000u,                           // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.kernels.size(), 1u);
  const ConSanKernelInfo &kernel = result.kernels.front();
  ASSERT_EQ(kernel.flat_sites.size(), 1u);
  EXPECT_EQ(kernel.flat_sites.front().address_space_hint, ConSanFlatAddressSpaceHint::Group);
  EXPECT_EQ(kernel.stats.flat_group_hint_count, 1u);
  EXPECT_EQ(kernel.stats.flat_unknown_hint_count, 0u);
}

TEST(ConSan, PropagatesGfx1250SharedPointerThroughExactScratchSlot) {
  const std::array<uint32_t, 13> text_words = {
      0xBE9201EBu, // s_mov_b64 s[18:19], src_shared_base
      0x7E020212u, // v_mov_b32_e32 v1, s18
      0x7E040213u, // v_mov_b32_e32 v2, s19
      0xED06C021u,
      0x00800000u,
      0x00061400u, // scratch_store_b64 off, v[1:2], s33
      0xED054021u,
      0x00000003u,
      0x00061400u, // scratch_load_b64 v[3:4], off, s33
      0xEC05007Cu,
      0x00000005u,
      0x00000003u, // flat_load_b32 v5, v[3:4]
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250),
  };
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const ConSanResult result = try_patch_consan(make_gfx1250_code_object(text_words), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.kernels.front().flat_sites.size(), 1u);
  EXPECT_EQ(result.kernels.front().flat_sites.front().address_space_hint,
            ConSanFlatAddressSpaceHint::Group);
  EXPECT_EQ(result.kernels.front().stats.flat_group_hint_count, 1u);
  EXPECT_EQ(result.kernels.front().stats.flat_unknown_hint_count, 0u);
}

TEST(ConSan, PropagatesCdnaSharedPointerThroughExactScratchSlot) {
  const std::array<uint32_t, 14> text_words = {
      0xbe8001ebu, // s_mov_b64 s[0:1], src_shared_base
      0x7e000200u, // v_mov_b32_e32 v0, s0
      0x7e020201u, // v_mov_b32_e32 v1, s1
      0xdc744000u,
      0x00080000u, // scratch_store_dwordx2 off, v[0:1], s8
      0xdc544000u,
      0x02080000u, // scratch_load_dwordx2 v[2:3], off, s8
      0xdc500000u,
      0x04000002u, // flat_load_dword v4, v[2:3]
      0xdc546000u,
      0x06080004u, // scratch_load_dwordx2 v[6:7], v4, s8
      0xdc500000u,
      0x08000006u, // flat_load_dword v8, v[6:7]
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  for (const rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    SCOPED_TRACE(static_cast<uint32_t>(arch));
    const std::vector<uint8_t> bytes =
        arch == ROCJITSU_CODE_ARCH_CDNA3
            ? make_cdna3_lds_code_object(text_words, "cdna_scratch_slot")
            : make_cdna4_lds_code_object(text_words, "cdna_scratch_slot");
    const ConSanResult result = try_patch_consan(bytes, options);

    ASSERT_TRUE(consan_patch_succeeded(result));
    ASSERT_EQ(result.kernels.size(), 1u);
    ASSERT_EQ(result.kernels.front().flat_sites.size(), 2u);
    EXPECT_EQ(result.kernels.front().flat_sites[0].address_space_hint,
              ConSanFlatAddressSpaceHint::Group);
    EXPECT_EQ(result.kernels.front().flat_sites[1].address_space_hint,
              ConSanFlatAddressSpaceHint::Unknown);
    EXPECT_EQ(result.kernels.front().stats.flat_group_hint_count, 1u);
    EXPECT_EQ(result.kernels.front().stats.flat_unknown_hint_count, 1u);
  }
}

TEST(ConSan, PropagatesGfx1250SharedHighHalfThroughVectorAddU64) {
  const std::vector<uint32_t> text_words = {
      0xBE8801EBu, // s_mov_b64 s[8:9], src_shared_base
      0xBE810009u, // s_mov_b32 s1, s9
      0x98010301u, // s_cselect_b32 s1, s1, s3
      0xBE830001u, // s_mov_b32 s3, s1
      0xD5280000u,
      0x02020002u, // v_add_nc_u64_e64 v[0:1], s[2:3], v[0:1]
      0xEC05007Cu,
      0x00000004u,
      0x00000000u, // flat_load_b32 v4, v[0:1]
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250),
  };
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const ConSanResult result = try_patch_consan(make_gfx1250_code_object(text_words), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.kernels.front().flat_sites.size(), 1u);
  EXPECT_EQ(result.kernels.front().flat_sites.front().address_space_hint,
            ConSanFlatAddressSpaceHint::MaybeGroup);
  EXPECT_EQ(result.kernels.front().stats.flat_maybe_group_hint_count, 1u);
  EXPECT_EQ(result.kernels.front().stats.flat_unknown_hint_count, 0u);
}

TEST(ConSan, PropagatesSharedHighHalfThroughD128ScratchSequence) {
  const std::array<uint32_t, 20> text_words = {
      0xBE9201EBu,                           // s_mov_b64 s[18:19], src_shared_base
      0xBE820013u,                           // s_mov_b32 s2, s19
      0x98020302u,                           // s_cselect_b32 s2, s2, s3
      0xBE910002u,                           // s_mov_b32 s17, s2
      0xBE830011u,                           // s_mov_b32 s3, s17
      0xD5200303u, 0x003A0403u,              // v_add_co_ci_u32_e64 v3, s3, s3, v2, s14
      0x7E040303u,                           // v_mov_b32_e32 v2, v3
      0xED06C021u, 0x00800000u, 0x00061400u, // scratch_store_b64 off, v[1:2], s33
      0xED054021u, 0x00000003u, 0x00061400u, // scratch_load_b64 v[3:4], off, s33
      0xD73D0003u, 0x02020603u,              // v_lshrrev_b64 v[3:4], s3, v[3:4]
      0xEC05007Cu, 0x00000005u, 0x00000003u, // flat_load_b32 v5, v[3:4]
      0xBFB00000u,                           // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.kernels.size(), 1u);
  const ConSanKernelInfo &kernel = result.kernels.front();
  ASSERT_EQ(kernel.flat_sites.size(), 1u);
  EXPECT_EQ(kernel.flat_sites.front().address_space_hint, ConSanFlatAddressSpaceHint::MaybeGroup);
  EXPECT_EQ(kernel.stats.flat_maybe_group_hint_count, 1u);
  EXPECT_EQ(kernel.stats.flat_unknown_hint_count, 0u);
}

TEST(ConSan, PropagatesSharedHighHalfThroughD128AddressConstruction) {
  const std::vector<uint32_t> text_words = {
      0xBE820180u,                           // s_mov_b64 s[2:3], 0
      0xBE900002u,                           // s_mov_b32 s16, s2
      0x8B117E0Eu,                           // s_and_b32 s17, s14, exec_lo
      0x9810100Fu,                           // s_cselect_b32 s16, s15, s16
      0xBE9201EBu,                           // s_mov_b64 s[18:19], src_shared_base
      0xBE820013u,                           // s_mov_b32 s2, s19
      0x8B0E7E0Eu,                           // s_and_b32 s14, s14, exec_lo
      0x98020302u,                           // s_cselect_b32 s2, s2, s3
      0xBE910002u,                           // s_mov_b32 s17, s2
      0xBE820084u,                           // s_mov_b32 s2, 4
      0xD51F0002u, 0x02020202u,              // v_lshlrev_b64_e64 v[2:3], s2, v[1:2]
      0xBE8E0010u,                           // s_mov_b32 s14, s16
      0x7E020302u,                           // v_mov_b32_e32 v1, v2
      0xBE830011u,                           // s_mov_b32 s3, s17
      0x7E040303u,                           // v_mov_b32_e32 v2, v3
      0xD7000E01u, 0x0202020Eu,              // v_add_co_u32 v1, s14, s14, v1
      0xD5200303u, 0x003A0403u,              // v_add_co_ci_u32_e64 v3, s3, s3, v2, s14
      0x7E040303u,                           // v_mov_b32_e32 v2, v3
      0xEC05007Cu, 0x00000005u, 0x00000001u, // flat_load_b32 v5, v[1:2]
      0xBFB00000u,                           // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.kernels.size(), 1u);
  const ConSanKernelInfo &kernel = result.kernels.front();
  ASSERT_EQ(kernel.flat_sites.size(), 1u);
  EXPECT_EQ(kernel.flat_sites.front().address_space_hint, ConSanFlatAddressSpaceHint::MaybeGroup);
}

TEST(ConSan, PropagatesSharedPointerThroughExactPrivateFrameSlot) {
  const std::vector<uint32_t> text_words = {
      0xBE9201EBu,                           // s_mov_b64 s[18:19], src_shared_base
      0x7E100212u,                           // v_mov_b32_e32 v8, s18
      0x7E120213u,                           // v_mov_b32_e32 v9, s19
      0xBE8201EDu,                           // s_mov_b64 s[2:3], src_private_base
      0x7E000202u,                           // v_mov_b32_e32 v0, s2
      0x7E020203u,                           // v_mov_b32_e32 v1, s3
      0xEC06C07Cu, 0x04000000u, 0x00000000u, // flat_store_b64 v[0:1], v[8:9]
      0x7E080202u,                           // v_mov_b32_e32 v4, s2
      0x7E0A0203u,                           // v_mov_b32_e32 v5, s3
      0xEC05407Cu, 0x00000006u, 0x00000004u, // flat_load_b64 v[6:7], v[4:5]
      0xEC05007Cu, 0x0000000Au, 0x00000006u, // flat_load_b32 v10, v[6:7]
      0xBFB00000u,                           // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.kernels.size(), 1u);
  const ConSanKernelInfo &kernel = result.kernels.front();
  ASSERT_EQ(kernel.flat_sites.size(), 3u);
  EXPECT_EQ(kernel.flat_sites[0].address_space_hint, ConSanFlatAddressSpaceHint::Private);
  EXPECT_EQ(kernel.flat_sites[1].address_space_hint, ConSanFlatAddressSpaceHint::Private);
  EXPECT_EQ(kernel.flat_sites[2].address_space_hint, ConSanFlatAddressSpaceHint::Group);
}

TEST(ConSan, PropagatesSharedPointerThroughScalarLaneReservoir) {
  const std::vector<uint32_t> text_words = {
      0xBE9201EBu,                           // s_mov_b64 s[18:19], src_shared_base
      0xD7610028u, 0x02010012u,              // v_writelane_b32 v40, s18, 0
      0xD7610028u, 0x02010213u,              // v_writelane_b32 v40, s19, 1
      0xD7600002u, 0x02010128u,              // v_readlane_b32 s2, v40, 0
      0xD7600003u, 0x02010328u,              // v_readlane_b32 s3, v40, 1
      0x7E000202u,                           // v_mov_b32_e32 v0, s2
      0x7E020203u,                           // v_mov_b32_e32 v1, s3
      0xEC05007Cu, 0x00000004u, 0x00000000u, // flat_load_b32 v4, v[0:1]
      0xBFB00000u,                           // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.kernels.front().flat_sites.size(), 1u);
  EXPECT_EQ(result.kernels.front().flat_sites.front().address_space_hint,
            ConSanFlatAddressSpaceHint::Group);
}

TEST(ConSan, PropagatesGfx1250SharedPointerThroughScalarLaneReservoir) {
  const std::vector<uint32_t> text_words = {
      0xBE9201EBu, // s_mov_b64 s[18:19], src_shared_base
      0xD7610028u,
      0x02010012u, // v_writelane_b32 v40, s18, 0
      0xD7610028u,
      0x02010213u, // v_writelane_b32 v40, s19, 1
      0xD7600002u,
      0x02010128u, // v_readlane_b32 s2, v40, 0
      0xD7600003u,
      0x02010328u, // v_readlane_b32 s3, v40, 1
      0x7E000202u, // v_mov_b32_e32 v0, s2
      0x7E020203u, // v_mov_b32_e32 v1, s3
      0xEC05007Cu,
      0x00000004u,
      0x00000000u, // flat_load_b32 v4, v[0:1]
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250),
  };
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const ConSanResult result = try_patch_consan(make_gfx1250_code_object(text_words), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.kernels.front().flat_sites.size(), 1u);
  EXPECT_EQ(result.kernels.front().flat_sites.front().address_space_hint,
            ConSanFlatAddressSpaceHint::Group);
}

TEST(ConSan, InventoriesLocalFunctionFlatSharedAccesses) {
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
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.kernels.size(), 1u);
  EXPECT_EQ(result.kernels.front().name, "lds_probe");
  EXPECT_EQ(result.kernels.front().code_size, 4u);
  EXPECT_EQ(result.kernels.front().stats.flat_group_hint_count, 0u);
  ASSERT_EQ(result.functions.size(), 1u);

  const ConSanFunctionInfo &function = result.functions.front();
  EXPECT_EQ(function.name, "lds_helper");
  EXPECT_TRUE(function.decoded);
  EXPECT_EQ(function.entry_text_offset, 4u);
  EXPECT_EQ(function.text_file_offset, 0x100u);
  EXPECT_EQ(function.code_size, 36u);
  EXPECT_EQ(function.stats.instruction_count, 5u);
  EXPECT_EQ(function.stats.flat_read_count, 1u);
  EXPECT_EQ(function.stats.flat_group_hint_count, 1u);
  EXPECT_EQ(function.stats.flat_unknown_hint_count, 0u);
  ASSERT_EQ(function.flat_sites.size(), 1u);
  EXPECT_EQ(function.flat_sites.front().address_space_hint, ConSanFlatAddressSpaceHint::Group);
  EXPECT_EQ(function.flat_sites.front().text_offset, 24u);
  EXPECT_EQ(function.flat_sites.front().file_offset, 0x118u);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
}

TEST(ConSan, PropagatesCdna4LaneStateThroughAccVgpr) {
  const std::vector<uint32_t> text_words = {
      0xbe8001ebu, // s_mov_b64 s[0:1], src_shared_base
      0xd28a000au,
      0x00011800u, // v_writelane_b32 v10, s0, 12
      0xd28a000au,
      0x00011a01u, // v_writelane_b32 v10, s1, 13
      0xd3d9400fu,
      0x1800010au, // v_accvgpr_write_b32 a15, v10
      0x7e140280u, // v_mov_b32_e32 v10, 0
      0xd3d8400au,
      0x1800010fu, // v_accvgpr_read_b32 v10, a15
      0xd2890002u,
      0x0001190au, // v_readlane_b32 s2, v10, 12
      0xd2890003u,
      0x00011b0au, // v_readlane_b32 s3, v10, 13
      0x7e000202u, // v_mov_b32_e32 v0, s2
      0x7e020203u, // v_mov_b32_e32 v1, s3
      0xdc500000u,
      0x04000000u, // flat_load_dword v4, v[0:1]
      0xd3d9400fu,
      0x1800010bu, // v_accvgpr_write_b32 a15, v11
      0xd3d8400au,
      0x1800010fu, // v_accvgpr_read_b32 v10, a15
      0xd2890002u,
      0x0001190au, // v_readlane_b32 s2, v10, 12
      0xd2890003u,
      0x00011b0au, // v_readlane_b32 s3, v10, 13
      0x7e000202u, // v_mov_b32_e32 v0, s2
      0x7e020203u, // v_mov_b32_e32 v1, s3
      0xdc500000u,
      0x04000000u, // flat_load_dword v4, v[0:1]
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const ConSanResult result =
      try_patch_consan(make_cdna4_lds_code_object(text_words, "accvgpr_lane_state"), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.kernels.front().flat_sites.size(), 2u);
  EXPECT_EQ(result.kernels.front().flat_sites[0].address_space_hint,
            ConSanFlatAddressSpaceHint::Group);
  EXPECT_EQ(result.kernels.front().flat_sites[1].address_space_hint,
            ConSanFlatAddressSpaceHint::Unknown);
}

TEST(ConSan, RejectsCdnaDynamicLaneSelectorProvenance) {
  const std::array<uint32_t, 14> text_words = {
      0xbe8001ebu, // s_mov_b64 s[0:1], src_shared_base
      0xd28a000au,
      0x00011800u, // v_writelane_b32 v10, s0, 12
      0xd28a000au,
      0x00011a01u, // v_writelane_b32 v10, s1, 13
      0xd2890002u,
      0x0000190au, // v_readlane_b32 s2, v10, s12
      0xd2890003u,
      0x00001b0au, // v_readlane_b32 s3, v10, s13
      0x7e000202u, // v_mov_b32_e32 v0, s2
      0x7e020203u, // v_mov_b32_e32 v1, s3
      0xdc500000u,
      0x04000000u, // flat_load_dword v4, v[0:1]
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  for (const rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    SCOPED_TRACE(static_cast<uint32_t>(arch));
    const std::vector<uint8_t> bytes =
        arch == ROCJITSU_CODE_ARCH_CDNA3
            ? make_cdna3_lds_code_object(text_words, "cdna_dynamic_lane")
            : make_cdna4_lds_code_object(text_words, "cdna_dynamic_lane");
    const ConSanResult result = try_patch_consan(bytes, options);

    ASSERT_TRUE(consan_patch_succeeded(result));
    ASSERT_EQ(result.kernels.size(), 1u);
    ASSERT_EQ(result.kernels.front().flat_sites.size(), 1u);
    EXPECT_EQ(result.kernels.front().flat_sites.front().address_space_hint,
              ConSanFlatAddressSpaceHint::Unknown);
    EXPECT_EQ(result.kernels.front().stats.flat_unknown_hint_count, 1u);
  }
}

TEST(ConSan, RejectsRdna4AndGfx1250DynamicLaneSelectorProvenance) {
  const std::array<uint32_t, 15> text_words = {
      0xbe8001ebu, // s_mov_b64 s[0:1], src_shared_base
      0xd761000au,
      0x02011800u, // v_writelane_b32 v10, s0, 12
      0xd761000au,
      0x02011a01u, // v_writelane_b32 v10, s1, 13
      0xd7600002u,
      0x0200190au, // v_readlane_b32 s2, v10, s12
      0xd7600003u,
      0x02001b0au, // v_readlane_b32 s3, v10, s13
      0x7e000202u, // v_mov_b32_e32 v0, s2
      0x7e020203u, // v_mov_b32_e32 v1, s3
      0xec05007cu,
      0x00000004u,
      0x00000000u, // flat_load_b32 v4, v[0:1]
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250),
  };
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  for (const rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_GFX1250}) {
    SCOPED_TRACE(static_cast<uint32_t>(arch));
    const std::vector<uint8_t> bytes =
        arch == ROCJITSU_CODE_ARCH_RDNA4
            ? make_rdna4_lds_code_object(text_words, "rdna4_dynamic_lane")
            : make_gfx1250_code_object(text_words, "gfx1250_dynamic_lane");
    const ConSanResult result = try_patch_consan(bytes, options);

    ASSERT_TRUE(consan_patch_succeeded(result));
    ASSERT_EQ(result.kernels.size(), 1u);
    ASSERT_EQ(result.kernels.front().flat_sites.size(), 1u);
    EXPECT_EQ(result.kernels.front().flat_sites.front().address_space_hint,
              ConSanFlatAddressSpaceHint::Unknown);
    EXPECT_EQ(result.kernels.front().stats.flat_unknown_hint_count, 1u);
  }
}

TEST(ConSan, ClearsRdna4AndGfx1250LaneProvenanceOnWideLiteralWrite) {
  const std::array<uint32_t, 21> text_words = {
      0xbe8001ebu, // s_mov_b64 s[0:1], src_shared_base
      0xd761000au,
      0x02011800u, // v_writelane_b32 v10, s0, 12
      0xd761000au,
      0x02011a01u, // v_writelane_b32 v10, s1, 13
      0xd761000au,
      0x020118ffu,
      0x12345678u, // v_writelane_b32 v10, 0x12345678, 12
      0xd761000au,
      0x02011affu,
      0x87654321u, // v_writelane_b32 v10, 0x87654321, 13
      0xd7600002u,
      0x0201190au, // v_readlane_b32 s2, v10, 12
      0xd7600003u,
      0x02011b0au, // v_readlane_b32 s3, v10, 13
      0x7e000202u, // v_mov_b32_e32 v0, s2
      0x7e020203u, // v_mov_b32_e32 v1, s3
      0xec05007cu,
      0x00000004u,
      0x00000000u, // flat_load_b32 v4, v[0:1]
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250),
  };
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  for (const rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_GFX1250}) {
    SCOPED_TRACE(static_cast<uint32_t>(arch));
    const std::vector<uint8_t> bytes =
        arch == ROCJITSU_CODE_ARCH_RDNA4
            ? make_rdna4_lds_code_object(text_words, "rdna4_wide_lane")
            : make_gfx1250_code_object(text_words, "gfx1250_wide_lane");
    const ConSanResult result = try_patch_consan(bytes, options);

    ASSERT_TRUE(consan_patch_succeeded(result));
    ASSERT_EQ(result.kernels.size(), 1u);
    ASSERT_EQ(result.kernels.front().flat_sites.size(), 1u);
    EXPECT_EQ(result.kernels.front().flat_sites.front().address_space_hint,
              ConSanFlatAddressSpaceHint::Unknown);
    EXPECT_EQ(result.kernels.front().stats.flat_unknown_hint_count, 1u);
  }
}

TEST(ConSan, SelectsCdnaSemanticValueForVectorShift) {
  const std::array<uint32_t, 11> text_words = {
      0xbe8401edu, // s_mov_b64 s[4:5], src_private_base
      0xbe8001ebu, // s_mov_b64 s[0:1], src_shared_base
      0x7e040200u, // v_mov_b32_e32 v2, s0
      0x7e060201u, // v_mov_b32_e32 v3, s1
      0xd2900000u,
      0x00020404u, // v_lshrrev_b64 v[0:1], s4, v[2:3]
      0x7e020300u, // v_mov_b32_e32 v1, v0
      0x7e000302u, // v_mov_b32_e32 v0, v2
      0xdc500000u,
      0x04000000u, // flat_load_dword v4, v[0:1]
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const ConSanResult result =
      try_patch_consan(make_cdna4_lds_code_object(text_words, "cdna_vector_shift"), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.kernels.front().flat_sites.size(), 1u);
  EXPECT_EQ(result.kernels.front().flat_sites.front().address_space_hint,
            ConSanFlatAddressSpaceHint::MaybeGroup);
  EXPECT_EQ(result.kernels.front().stats.flat_maybe_group_hint_count, 1u);
  EXPECT_EQ(result.kernels.front().stats.flat_private_hint_count, 0u);
  EXPECT_EQ(result.kernels.front().stats.flat_maybe_private_hint_count, 0u);
}

TEST(ConSan, SelectsCdnaSemanticValueForVectorLeftShift) {
  const std::array<uint32_t, 9> text_words = {
      0xbe8401edu, // s_mov_b64 s[4:5], src_private_base
      0xbe8001ebu, // s_mov_b64 s[0:1], src_shared_base
      0x7e040200u, // v_mov_b32_e32 v2, s0
      0x7e060201u, // v_mov_b32_e32 v3, s1
      0xd28f0000u,
      0x00020404u, // v_lshlrev_b64 v[0:1], s4, v[2:3]
      0xdc500000u,
      0x04000000u, // flat_load_dword v4, v[0:1]
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const ConSanResult result =
      try_patch_consan(make_cdna4_lds_code_object(text_words, "cdna_vector_left_shift"), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.kernels.front().flat_sites.size(), 1u);
  EXPECT_EQ(result.kernels.front().flat_sites.front().address_space_hint,
            ConSanFlatAddressSpaceHint::MaybeGroup);
  EXPECT_EQ(result.kernels.front().stats.flat_maybe_group_hint_count, 1u);
  EXPECT_EQ(result.kernels.front().stats.flat_private_hint_count, 0u);
  EXPECT_EQ(result.kernels.front().stats.flat_maybe_private_hint_count, 0u);
}

TEST(ConSan, PropagatesCdna4AccVgprPointerAcrossHelperCall) {
  constexpr uint32_t kFunctionDelta = 20u;
  const std::array<uint32_t, 13> kernel_words = {
      0xbe8001ebu, // s_mov_b64 s[0:1], src_shared_base
      0x7e000200u, // v_mov_b32_e32 v0, s0
      0x7e020201u, // v_mov_b32_e32 v1, s1
      0xd3d94000u,
      0x18000100u, // v_accvgpr_write_b32 a0, v0
      0xd3d94001u,
      0x18000101u, // v_accvgpr_write_b32 a1, v1
      build_s_getpc_b64(/*sdst=*/2, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_add_u32(/*sdst=*/2, /*ssrc0=*/2, /*literal=*/255, ROCJITSU_CODE_ARCH_CDNA4),
      kFunctionDelta,
      build_s_addc_u32(/*sdst=*/3, /*ssrc0=*/3, /*inline 0=*/128, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_swappc_b64(/*sdst=*/30, /*ssrc0=*/2, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  const std::array<uint32_t, 7> function_words = {
      0xd3d84000u,
      0x18000100u, // v_accvgpr_read_b32 v0, a0
      0xd3d84001u,
      0x18000101u, // v_accvgpr_read_b32 v1, a1
      0xdc500000u,
      0x04000000u, // flat_load_dword v4, v[0:1]
      build_s_setpc_b64(/*ssrc0=*/30, ROCJITSU_CODE_ARCH_CDNA4),
  };
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const ConSanResult result = try_patch_consan(
      make_cdna4_code_object_with_local_function(kernel_words, function_words), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.functions.size(), 1u);
  ASSERT_EQ(result.functions.front().flat_sites.size(), 1u);
  EXPECT_EQ(result.functions.front().flat_sites.front().address_space_hint,
            ConSanFlatAddressSpaceHint::Group);
  EXPECT_EQ(result.functions.front().stats.flat_group_hint_count, 1u);
  EXPECT_EQ(result.functions.front().stats.flat_unknown_hint_count, 0u);
}

TEST(ConSan, RelaysCdna4SharedPointerThroughPrivateHelperFrame) {
  constexpr uint32_t kFunctionDelta = 20u;
  const std::array<uint32_t, 21> kernel_words = {
      0xbe8001ebu, // s_mov_b64 s[0:1], src_shared_base
      build_v_mov_b32_e32(/*vdst=*/0, /*scalar s0=*/0, ROCJITSU_CODE_ARCH_CDNA4),
      build_v_mov_b32_e32(/*vdst=*/1, /*scalar s1=*/1, ROCJITSU_CODE_ARCH_CDNA4),
      0xd3d94000u,
      0x18000100u, // v_accvgpr_write_b32 a0, v0
      0xd3d94001u,
      0x18000101u, // v_accvgpr_write_b32 a1, v1
      0xd3d84002u,
      0x18000100u, // v_accvgpr_read_b32 v2, a0
      0xd3d84003u,
      0x18000101u, // v_accvgpr_read_b32 v3, a1
      0xd2900000u,
      0x00020400u, // v_lshrrev_b64 v[0:1], s0, v[2:3]
      0x7e020300u, // v_mov_b32_e32 v1, v0
      0x7e000302u, // v_mov_b32_e32 v0, v2
      build_s_getpc_b64(/*sdst=*/2, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_add_u32(/*sdst=*/2, /*ssrc0=*/2, /*literal=*/255, ROCJITSU_CODE_ARCH_CDNA4),
      kFunctionDelta,
      build_s_addc_u32(/*sdst=*/3, /*ssrc0=*/3, /*inline 0=*/128, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_swappc_b64(/*sdst=*/30, /*ssrc0=*/2, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  const std::array<uint32_t, 28> function_words = {
      0xbe8801edu, // s_mov_b64 s[8:9], src_private_base
      0xbe8a0009u, // s_mov_b32 s10, s9
      0x7e040208u, // v_mov_b32_e32 v2, s8
      0x7e06020au, // v_mov_b32_e32 v3, s10
      0xdc740000u,
      0x00000002u, // flat_store_dwordx2 v[2:3], v[0:1]
      0xd28a000au,
      0x00011808u, // v_writelane_b32 v10, s8, 12
      0xd28a000au,
      0x00011a0au, // v_writelane_b32 v10, s10, 13
      0xbe880080u, // s_mov_b32 s8, 0
      0xbe8a0080u, // s_mov_b32 s10, 0
      0xd2890008u,
      0x0001190au, // v_readlane_b32 s8, v10, 12
      0xd289000au,
      0x00011b0au, // v_readlane_b32 s10, v10, 13
      0x7e040208u, // v_mov_b32_e32 v2, s8
      0x7e06020au, // v_mov_b32_e32 v3, s10
      0xdc540000u,
      0x04000002u, // flat_load_dwordx2 v[4:5], v[2:3]
      0x7e000208u, // v_mov_b32_e32 v0, s8
      0x7e020209u, // v_mov_b32_e32 v1, s9
      0xbe800082u, // s_mov_b32 s0, 2
      0xd2080000u,
      0x04100100u, // v_lshl_add_u64 v[0:1], v[0:1], s0, v[4:5]
      0xdc700000u,
      0x00000600u, // flat_store_dword v[0:1], v6
      build_s_setpc_b64(/*ssrc0=*/30, ROCJITSU_CODE_ARCH_CDNA4),
  };
  const std::vector<uint8_t> bytes =
      make_cdna4_code_object_with_local_function(kernel_words, function_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.functions.size(), 1u);
  const ConSanFunctionInfo &function = result.functions.front();
  ASSERT_EQ(function.flat_sites.size(), 3u);
  EXPECT_EQ(function.flat_sites[0].address_space_hint, ConSanFlatAddressSpaceHint::Private);
  EXPECT_EQ(function.flat_sites[1].address_space_hint, ConSanFlatAddressSpaceHint::Private);
  EXPECT_EQ(function.flat_sites[2].address_space_hint, ConSanFlatAddressSpaceHint::MaybeGroup);
  EXPECT_EQ(function.stats.flat_private_hint_count, 2u);
  EXPECT_EQ(function.stats.flat_maybe_group_hint_count, 1u);
  EXPECT_EQ(function.stats.flat_unknown_hint_count, 0u);
}

TEST(ConSan, CountsRdna4LdsAndSynchronizationInstructions) {
  const std::vector<uint8_t> bytes = make_rdna4_unsupported_lds_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.warnings.empty());
  ASSERT_EQ(result.kernels.size(), 1u);
  EXPECT_EQ(result.target, ROCJITSU_CODE_TARGET_GFX1201);
  EXPECT_EQ(result.arch, ROCJITSU_CODE_ARCH_RDNA4);

  const ConSanKernelInfo &kernel = result.kernels.front();
  EXPECT_EQ(kernel.name, "lds_probe");
  EXPECT_TRUE(kernel.has_text_range);
  EXPECT_TRUE(kernel.decoded);
  EXPECT_EQ(kernel.code_size, 44u);
  EXPECT_EQ(kernel.stats.instruction_count, 7u);
  EXPECT_EQ(kernel.stats.lds_read_count, 1u);
  EXPECT_EQ(kernel.stats.lds_write_count, 1u);
  EXPECT_EQ(kernel.stats.lds_atomic_count, 1u);
  EXPECT_EQ(kernel.stats.ds_other_count, 0u);
  EXPECT_EQ(kernel.stats.barrier_count, 1u);
  EXPECT_EQ(kernel.stats.wait_count, 1u);
  EXPECT_EQ(kernel.stats.fence_like_count, 1u);
  EXPECT_EQ(kernel.stats.decode_error_count, 0u);
  ASSERT_EQ(kernel.lds_sites.size(), 3u);
  EXPECT_EQ(kernel.lds_sites[0].kind, ConSanLdsAccessKind::Write);
  EXPECT_TRUE(kernel.lds_sites[0].supported_mvp);
  EXPECT_EQ(kernel.lds_sites[0].mnemonic, "ds_store_b32");
  EXPECT_EQ(kernel.lds_sites[0].text_offset, 0u);
  EXPECT_EQ(kernel.lds_sites[0].file_offset, 0x100u);
  EXPECT_EQ(kernel.lds_sites[0].size, 8u);
  EXPECT_EQ(kernel.lds_sites[0].width_bits, 32u);
  ASSERT_TRUE(kernel.lds_sites[0].addr_vgpr);
  ASSERT_TRUE(kernel.lds_sites[0].data_vgpr);
  EXPECT_FALSE(kernel.lds_sites[0].dst_vgpr);
  EXPECT_EQ(*kernel.lds_sites[0].addr_vgpr, 0u);
  EXPECT_EQ(*kernel.lds_sites[0].data_vgpr, 0u);
  EXPECT_EQ(kernel.lds_sites[1].kind, ConSanLdsAccessKind::Read);
  EXPECT_TRUE(kernel.lds_sites[1].supported_mvp);
  EXPECT_EQ(kernel.lds_sites[1].mnemonic, "ds_load_b32");
  EXPECT_EQ(kernel.lds_sites[1].text_offset, 8u);
  EXPECT_EQ(kernel.lds_sites[1].file_offset, 0x108u);
  EXPECT_EQ(kernel.lds_sites[1].size, 8u);
  EXPECT_EQ(kernel.lds_sites[1].width_bits, 32u);
  ASSERT_TRUE(kernel.lds_sites[1].dst_vgpr);
  ASSERT_TRUE(kernel.lds_sites[1].addr_vgpr);
  EXPECT_FALSE(kernel.lds_sites[1].data_vgpr);
  EXPECT_EQ(*kernel.lds_sites[1].dst_vgpr, 0u);
  EXPECT_EQ(*kernel.lds_sites[1].addr_vgpr, 0u);
  EXPECT_EQ(kernel.lds_sites[2].kind, ConSanLdsAccessKind::Atomic);
  EXPECT_FALSE(kernel.lds_sites[2].supported_mvp);
  EXPECT_EQ(kernel.lds_sites[2].mnemonic, "ds_add_u32");
  EXPECT_EQ(kernel.lds_sites[2].text_offset, 16u);
  EXPECT_EQ(kernel.lds_sites[2].file_offset, 0x110u);
  EXPECT_EQ(kernel.lds_sites[2].size, 8u);
  EXPECT_EQ(kernel.lds_sites[2].width_bits, 32u);
  ASSERT_TRUE(kernel.lds_sites[2].addr_vgpr);
  ASSERT_TRUE(kernel.lds_sites[2].data_vgpr);
  EXPECT_EQ(*kernel.lds_sites[2].addr_vgpr, 0u);
  EXPECT_EQ(*kernel.lds_sites[2].data_vgpr, 0u);
  ASSERT_EQ(kernel.barrier_sites.size(), 1u);
  EXPECT_EQ(kernel.barrier_sites[0].mnemonic, "s_barrier_wait");
  EXPECT_EQ(kernel.barrier_sites[0].text_offset, 24u);
  ASSERT_TRUE(kernel.barrier_sites[0].barrier_id);
  EXPECT_EQ(*kernel.barrier_sites[0].barrier_id, 0);
  EXPECT_EQ(kernel.barrier_sites[0].operand_source, ConSanBarrierSite::OperandSource::Immediate);
  EXPECT_EQ(kernel.barrier_sites[0].scope, ConSanBarrierSite::Scope::Unknown);
  ASSERT_EQ(kernel.fence_sites.size(), 1u);
  EXPECT_EQ(kernel.fence_sites[0].mnemonic, "s_dcache_inv");
  EXPECT_EQ(kernel.fence_sites[0].text_offset, 32u);
  EXPECT_EQ(kernel.fence_sites[0].file_offset, 0x120u);
  EXPECT_EQ(kernel.fence_sites[0].size, 8u);
  ASSERT_EQ(kernel.atomic_sites.size(), 1u);
  const ConSanAtomicSite &atomic = kernel.atomic_sites.front();
  EXPECT_EQ(atomic.address_space_hint, ConSanAtomicAddressSpaceHint::Lds);
  EXPECT_EQ(atomic.mnemonic, "ds_add_u32");
  EXPECT_EQ(atomic.text_offset, 16u);
  EXPECT_EQ(atomic.file_offset, 0x110u);
  EXPECT_EQ(atomic.size, 8u);
  EXPECT_EQ(atomic.width_bits, 32u);
  ASSERT_TRUE(atomic.addr_vgpr);
  ASSERT_TRUE(atomic.data_vgpr);
  EXPECT_EQ(*atomic.addr_vgpr, 0u);
  EXPECT_EQ(*atomic.data_vgpr, 0u);
  ASSERT_TRUE(atomic.raw_op);
  ASSERT_TRUE(atomic.raw_addr);
  ASSERT_TRUE(atomic.raw_data0);
  ASSERT_TRUE(atomic.raw_data1);
  ASSERT_TRUE(atomic.raw_vdst);
  EXPECT_EQ(*atomic.raw_op, 0u);
  EXPECT_EQ(*atomic.raw_addr, 0u);
  EXPECT_EQ(*atomic.raw_data0, 0u);
  EXPECT_EQ(*atomic.raw_data1, 0u);
  EXPECT_EQ(*atomic.raw_vdst, 0u);
  EXPECT_EQ(atomic.raw_ioffset, 0);
  EXPECT_FALSE(atomic.raw_scope);
  EXPECT_FALSE(atomic.raw_th);
  ASSERT_TRUE(atomic.returns_old_value);
  EXPECT_FALSE(*atomic.returns_old_value);
  EXPECT_EQ(kernel.preflight_action, ConSanPreflightAction::Candidate);
  EXPECT_TRUE(std::ranges::any_of(kernel.preflight_reasons, [](const std::string &reason) {
    return reason == "atomic LDS accesses excluded: 1";
  }));
  ASSERT_EQ(result.sync_events.size(), 3u);
  const ConSanSyncEvent &atomic_event = result.sync_events[0];
  EXPECT_EQ(atomic_event.kind, ConSanSyncEventKind::Atomic);
  EXPECT_EQ(atomic_event.operation, ConSanSyncOperation::AtomicRmw);
  EXPECT_EQ(atomic_event.address_source, ConSanSyncAddressSource::LdsVector);
  EXPECT_EQ(atomic_event.memory_role, ConSanSyncMemoryRole::Unknown);
  EXPECT_EQ(atomic_event.rmw_outcome, ConSanSyncRmwOutcome::NoReturn);
  EXPECT_EQ(atomic_event.confidence, ConSanSemanticConfidence::Conservative);
  EXPECT_FALSE(atomic_event.confidence_reason.empty());
  EXPECT_EQ(atomic_event.text_offset, 16u);
  EXPECT_EQ(atomic_event.width_bits, 32u);
  EXPECT_EQ(atomic_event.static_byte_offset, 0);
  EXPECT_FALSE(atomic_event.raw_scope);

  const ConSanSyncEvent &barrier_event = result.sync_events[1];
  EXPECT_EQ(barrier_event.kind, ConSanSyncEventKind::Barrier);
  EXPECT_EQ(barrier_event.operation, ConSanSyncOperation::BarrierWait);
  EXPECT_EQ(barrier_event.memory_role, ConSanSyncMemoryRole::Acquire);
  EXPECT_EQ(barrier_event.confidence, ConSanSemanticConfidence::Unsupported);
  EXPECT_EQ(barrier_event.text_offset, 24u);
  ASSERT_TRUE(barrier_event.barrier_id);
  EXPECT_EQ(*barrier_event.barrier_id, 0);
  EXPECT_EQ(barrier_event.barrier_operand_source, ConSanBarrierSite::OperandSource::Immediate);
  EXPECT_EQ(barrier_event.barrier_scope, ConSanBarrierSite::Scope::Unknown);
  EXPECT_FALSE(barrier_event.participant_count);
  EXPECT_FALSE(barrier_event.participant_mask);

  const ConSanSyncEvent &fence_event = result.sync_events[2];
  EXPECT_EQ(fence_event.kind, ConSanSyncEventKind::Fence);
  EXPECT_EQ(fence_event.operation, ConSanSyncOperation::Fence);
  EXPECT_EQ(fence_event.memory_role, ConSanSyncMemoryRole::Unknown);
  EXPECT_EQ(fence_event.confidence, ConSanSemanticConfidence::Conservative);
  EXPECT_EQ(fence_event.text_offset, 32u);
  EXPECT_EQ(fence_event.code_object_fingerprint, atomic_event.code_object_fingerprint);
  EXPECT_NE(fence_event.identity.find("|kernel=lds_probe|event=fence|"), std::string::npos);
  ASSERT_EQ(result.sync_sequences.size(), result.sync_events.size());
  for (size_t i = 0; i < result.sync_sequences.size(); ++i) {
    const ConSanSyncSequence &sequence = result.sync_sequences[i];
    ASSERT_TRUE(sequence.basic_block_index);
    ASSERT_EQ(sequence.member_event_identities.size(), 1u);
    EXPECT_EQ(sequence.member_event_identities.front(), result.sync_events[i].identity);
    EXPECT_EQ(sequence.begin_text_offset, result.sync_events[i].text_offset);
    EXPECT_EQ(sequence.end_text_offset,
              result.sync_events[i].text_offset + result.sync_events[i].size);
  }
  EXPECT_EQ(result.sync_sequences[0].kind, ConSanSyncSequenceKind::Atomic);
  EXPECT_EQ(result.sync_sequences[1].kind, ConSanSyncSequenceKind::Barrier);
  EXPECT_EQ(result.sync_sequences[2].kind, ConSanSyncSequenceKind::Fence);
  EXPECT_EQ(result.sync_sequences[0].confidence, ConSanSemanticConfidence::Conservative);
  EXPECT_EQ(result.sync_sequences[1].confidence, ConSanSemanticConfidence::Unsupported);
  EXPECT_EQ(result.sync_sequences[2].confidence, ConSanSemanticConfidence::Conservative);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
}

TEST(ConSan, RetainsTypedIdentityForEverySupportedTarget) {
  struct TargetCase {
    rj_code_target_id_t target;
    rj_code_arch_t arch;
    std::vector<uint8_t> bytes;
  };
  const std::array<TargetCase, 5> cases = {
      TargetCase{
          .target = ROCJITSU_CODE_TARGET_GFX942,
          .arch = ROCJITSU_CODE_ARCH_CDNA3,
          .bytes = make_cdna3_lds_code_object(std::array{build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA3)}),
      },
      TargetCase{
          .target = ROCJITSU_CODE_TARGET_GFX950,
          .arch = ROCJITSU_CODE_ARCH_CDNA4,
          .bytes = make_cdna4_lds_code_object(std::array{build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4)}),
      },
      TargetCase{
          .target = ROCJITSU_CODE_TARGET_GFX1100,
          .arch = ROCJITSU_CODE_ARCH_RDNA3,
          .bytes = make_rdna3_lds_code_object(std::array{build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA3)}),
      },
      TargetCase{
          .target = ROCJITSU_CODE_TARGET_GFX1201,
          .arch = ROCJITSU_CODE_ARCH_RDNA4,
          .bytes = make_rdna4_lds_code_object(std::array{build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4)}),
      },
      TargetCase{
          .target = ROCJITSU_CODE_TARGET_GFX1250,
          .arch = ROCJITSU_CODE_ARCH_GFX1250,
          .bytes = make_gfx1250_code_object(std::array{build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250)}),
      },
  };
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  for (const TargetCase &target_case : cases) {
    SCOPED_TRACE(rj_code_target_name(target_case.target));
    EXPECT_EQ(consan_arch_for_target(target_case.target), target_case.arch);
    const ConSanResult result = try_patch_consan(target_case.bytes, options);
    ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
    EXPECT_TRUE(result.parsed_code_object);
    EXPECT_EQ(result.target, target_case.target);
    EXPECT_EQ(result.arch, target_case.arch);
  }
  EXPECT_EQ(consan_capability_engine(ConSanFlavor::SuperCollider, ConSanMoiEngine::RecordReplay),
            ConSanCapabilityEngine::SuperCollider);
  EXPECT_EQ(consan_capability_engine(ConSanFlavor::Moi, ConSanMoiEngine::RecordReplay),
            ConSanCapabilityEngine::RecordReplay);
  EXPECT_EQ(consan_capability_engine(ConSanFlavor::Moi, ConSanMoiEngine::Sampled),
            ConSanCapabilityEngine::Sampled);
  EXPECT_EQ(consan_capability_engine(ConSanFlavor::Moi, ConSanMoiEngine::InlineShadow),
            ConSanCapabilityEngine::InlineShadow);
}

TEST(ConSan, CountsCdna4LdsAccessesFromNativeInstructionShapes) {
  const std::array<uint32_t, 5> text_words = {
      0xd81a0004u,
      0x00000302u, // ds_write_b32 v2, v3 offset:4
      0xd86c0004u,
      0x04000002u, // ds_read_b32 v4, v2 offset:4
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  const std::vector<uint8_t> bytes = make_cdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.target, ROCJITSU_CODE_TARGET_GFX950);
  ASSERT_EQ(result.arch, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_EQ(result.kernels.size(), 1u);
  const ConSanKernelInfo &kernel = result.kernels.front();
  EXPECT_TRUE(kernel.decoded);
  EXPECT_EQ(kernel.stats.instruction_count, 3u);
  EXPECT_EQ(kernel.stats.lds_write_count, 1u);
  EXPECT_EQ(kernel.stats.lds_read_count, 1u);
  EXPECT_EQ(kernel.stats.decode_error_count, 0u);
  ASSERT_EQ(kernel.lds_sites.size(), 2u);
  EXPECT_EQ(kernel.lds_sites[0].kind, ConSanLdsAccessKind::Write);
  EXPECT_EQ(kernel.lds_sites[0].mnemonic, "ds_write_b32");
  EXPECT_EQ(kernel.lds_sites[0].text_offset, 0u);
  EXPECT_EQ(kernel.lds_sites[0].size, 8u);
  ASSERT_TRUE(kernel.lds_sites[0].addr_vgpr);
  ASSERT_TRUE(kernel.lds_sites[0].data_vgpr);
  EXPECT_EQ(*kernel.lds_sites[0].addr_vgpr, 2u);
  EXPECT_EQ(*kernel.lds_sites[0].data_vgpr, 3u);
  EXPECT_TRUE(kernel.lds_sites[0].supported_mvp);
  EXPECT_EQ(kernel.lds_sites[1].kind, ConSanLdsAccessKind::Read);
  EXPECT_EQ(kernel.lds_sites[1].mnemonic, "ds_read_b32");
  EXPECT_EQ(kernel.lds_sites[1].text_offset, 8u);
  EXPECT_EQ(kernel.lds_sites[1].size, 8u);
  ASSERT_TRUE(kernel.lds_sites[1].addr_vgpr);
  ASSERT_TRUE(kernel.lds_sites[1].dst_vgpr);
  EXPECT_EQ(*kernel.lds_sites[1].addr_vgpr, 2u);
  EXPECT_EQ(*kernel.lds_sites[1].dst_vgpr, 4u);
  EXPECT_TRUE(kernel.lds_sites[1].supported_mvp);
  EXPECT_EQ(kernel.preflight_action, ConSanPreflightAction::Candidate);
  EXPECT_FALSE(result.modified);
}

TEST(ConSan, InventoriesCdna4HistogramLdsAtomics) {
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
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const ConSanResult result = try_patch_consan(
      make_cdna4_lds_code_object(text_words, "cdna4_histogram_lds_atomics"), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.kernels.size(), 1u);
  const ConSanKernelInfo &kernel = result.kernels.front();
  EXPECT_EQ(kernel.stats.lds_atomic_count, 5u);
  ASSERT_EQ(kernel.lds_sites.size(), 5u);
  const std::array<std::string_view, 5> expected_mnemonics = {
      "ds_add_u32", "ds_add_u64", "ds_add_f32", "ds_add_f64", "ds_cmpst_rtn_b32"};
  const std::array<uint32_t, 5> expected_widths = {32u, 64u, 32u, 64u, 32u};
  for (size_t i = 0; i < kernel.lds_sites.size(); ++i) {
    SCOPED_TRACE(i);
    EXPECT_EQ(kernel.lds_sites[i].kind, ConSanLdsAccessKind::Atomic);
    EXPECT_EQ(kernel.lds_sites[i].mnemonic, expected_mnemonics[i]);
    EXPECT_EQ(kernel.lds_sites[i].width_bits, expected_widths[i]);
    EXPECT_TRUE(kernel.lds_sites[i].addr_vgpr);
    EXPECT_TRUE(kernel.lds_sites[i].data_vgpr);
  }
  ASSERT_TRUE(kernel.lds_sites.back().dst_vgpr);
  ASSERT_TRUE(kernel.lds_sites.back().second_data_vgpr);
  EXPECT_EQ(*kernel.lds_sites.back().dst_vgpr, 13u);
  EXPECT_EQ(*kernel.lds_sites.back().addr_vgpr, 12u);
  EXPECT_EQ(*kernel.lds_sites.back().data_vgpr, 11u);
  EXPECT_EQ(*kernel.lds_sites.back().second_data_vgpr, 13u);
}

TEST(ConSan, InventoriesCdna4FlatRawFieldsAndExplicitSharedBase) {
  const auto load = build_cdna4_flat_load_b32(
      /*vaddr=*/0, /*vdst=*/2, /*byte_offset=*/0, ROCJITSU_CODE_ARCH_CDNA4);
  const auto store = build_cdna4_flat_store_b32(
      /*vaddr=*/4, /*vsrc=*/5, /*byte_offset=*/0, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(load && store);
  const std::array<uint32_t, 8> text_words = {
      0xbe8001ebu, // s_mov_b64 s[0:1], SRC_SHARED_BASE
      build_v_mov_b32_e32(/*vdst=*/0, /*scalar s0=*/0, ROCJITSU_CODE_ARCH_CDNA4),
      build_v_mov_b32_e32(/*vdst=*/1, /*scalar s1=*/1, ROCJITSU_CODE_ARCH_CDNA4),
      (*load)[0],
      (*load)[1],
      (*store)[0],
      (*store)[1],
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  const std::vector<uint8_t> bytes =
      make_cdna4_lds_code_object(text_words, "gfx950_flat_inventory");
  ConSanOptions options;
  options.flavor = ConSanFlavor::Moi;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.kernels.front().flat_sites.size(), 2u);
  const ConSanFlatSite &group_load = result.kernels.front().flat_sites[0];
  EXPECT_EQ(group_load.address_space_hint, ConSanFlatAddressSpaceHint::Group);
  EXPECT_EQ(group_load.size, 2u * sizeof(uint32_t));
  EXPECT_EQ(group_load.raw_vaddr, 0u);
  EXPECT_EQ(group_load.raw_vdst, 2u);
  EXPECT_EQ(group_load.raw_segment, 0u);
  EXPECT_EQ(group_load.raw_ioffset, 0);
  EXPECT_TRUE(group_load.raw_op.has_value());
  const ConSanFlatSite &unknown_store = result.kernels.front().flat_sites[1];
  EXPECT_EQ(unknown_store.address_space_hint, ConSanFlatAddressSpaceHint::Unknown);
  EXPECT_EQ(unknown_store.raw_vaddr, 4u);
  EXPECT_EQ(unknown_store.raw_vsrc, 5u);
  EXPECT_EQ(unknown_store.raw_segment, 0u);
  ASSERT_EQ(result.moi_candidates.size(), 1u);
  EXPECT_EQ(result.moi_candidates.front().source, ConSanMoiCandidateSource::FlatGroup);
  EXPECT_EQ(result.moi_candidates.front().raw_segment, 0u);
  ASSERT_EQ(result.site_dispositions.size(), 1u);
  EXPECT_EQ(result.site_dispositions.front().disposition, ConSanSiteDisposition::Supported);
  EXPECT_EQ(result.site_dispositions.front().reason, ConSanSiteDispositionReason::None);
}

std::vector<uint8_t> make_cdna4_padded_group_flat_code_object() {
  const auto load = build_cdna4_flat_load_b32(
      /*vaddr=*/0, /*vdst=*/2, /*byte_offset=*/0, ROCJITSU_CODE_ARCH_CDNA4);
  if (!load)
    return {};
  std::vector<uint32_t> text_words = {
      0xbe8001ebu, // s_mov_b64 s[0:1], SRC_SHARED_BASE
      build_v_mov_b32_e32(/*vdst=*/0, /*scalar s0=*/0, ROCJITSU_CODE_ARCH_CDNA4),
      build_v_mov_b32_e32(/*vdst=*/1, /*scalar s1=*/1, ROCJITSU_CODE_ARCH_CDNA4),
      (*load)[0],
      (*load)[1],
  };
  text_words.resize(1200, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);
  return make_cdna4_lds_code_object(text_words, "gfx950_flat_group_emission");
}

struct FlatSubwordTarget {
  rj_code_arch_t arch;
  std::string_view label;
  size_t target_index;
};

constexpr std::array<FlatSubwordTarget, 4> kFlatSubwordTargets = {{
    {ROCJITSU_CODE_ARCH_RDNA4, "gfx1201", 0},
    {ROCJITSU_CODE_ARCH_GFX1250, "gfx1250", 1},
    {ROCJITSU_CODE_ARCH_CDNA3, "gfx942", 2},
    {ROCJITSU_CODE_ARCH_CDNA4, "gfx950", 3},
}};

struct FlatD16LoadForm {
  std::array<uint16_t, 4> opcodes;
  std::array<std::string_view, 4> mnemonics;
  uint32_t memory_width_bits;
  ConSanFlatSubwordPlacement placement;
};

constexpr std::array<FlatD16LoadForm, 6> kFlatD16LoadForms = {{
    {{rdna4::kFlatLoadD16U8Vflat, gfx1250::kFlatLoadD16U8Vflat, cdna3::kFlatLoadUbyteD16Flat,
      cdna4::kFlatLoadUbyteD16Flat},
     {"flat_load_d16_u8", "flat_load_d16_u8", "flat_load_ubyte_d16", "flat_load_ubyte_d16"},
     8,
     ConSanFlatSubwordPlacement::Low16},
    {{rdna4::kFlatLoadD16I8Vflat, gfx1250::kFlatLoadD16I8Vflat, cdna3::kFlatLoadSbyteD16Flat,
      cdna4::kFlatLoadSbyteD16Flat},
     {"flat_load_d16_i8", "flat_load_d16_i8", "flat_load_sbyte_d16", "flat_load_sbyte_d16"},
     8,
     ConSanFlatSubwordPlacement::Low16},
    {{rdna4::kFlatLoadD16B16Vflat, gfx1250::kFlatLoadD16B16Vflat, cdna3::kFlatLoadShortD16Flat,
      cdna4::kFlatLoadShortD16Flat},
     {"flat_load_d16_b16", "flat_load_d16_b16", "flat_load_short_d16", "flat_load_short_d16"},
     16,
     ConSanFlatSubwordPlacement::Low16},
    {{rdna4::kFlatLoadD16HiU8Vflat, gfx1250::kFlatLoadD16HiU8Vflat, cdna3::kFlatLoadUbyteD16HiFlat,
      cdna4::kFlatLoadUbyteD16HiFlat},
     {"flat_load_d16_hi_u8", "flat_load_d16_hi_u8", "flat_load_ubyte_d16_hi",
      "flat_load_ubyte_d16_hi"},
     8,
     ConSanFlatSubwordPlacement::High16},
    {{rdna4::kFlatLoadD16HiI8Vflat, gfx1250::kFlatLoadD16HiI8Vflat, cdna3::kFlatLoadSbyteD16HiFlat,
      cdna4::kFlatLoadSbyteD16HiFlat},
     {"flat_load_d16_hi_i8", "flat_load_d16_hi_i8", "flat_load_sbyte_d16_hi",
      "flat_load_sbyte_d16_hi"},
     8,
     ConSanFlatSubwordPlacement::High16},
    {{rdna4::kFlatLoadD16HiB16Vflat, gfx1250::kFlatLoadD16HiB16Vflat,
      cdna3::kFlatLoadShortD16HiFlat, cdna4::kFlatLoadShortD16HiFlat},
     {"flat_load_d16_hi_b16", "flat_load_d16_hi_b16", "flat_load_short_d16_hi",
      "flat_load_short_d16_hi"},
     16,
     ConSanFlatSubwordPlacement::High16},
}};

struct FlatSubwordStoreForm {
  std::array<uint16_t, 4> opcodes;
  std::array<std::string_view, 4> mnemonics;
  uint32_t memory_width_bits;
  ConSanFlatSubwordPlacement placement;
};

constexpr std::array<FlatSubwordStoreForm, 4> kFlatSubwordStoreForms = {{
    {{rdna4::kFlatStoreB8Vflat, gfx1250::kFlatStoreB8Vflat, cdna3::kFlatStoreByteFlat,
      cdna4::kFlatStoreByteFlat},
     {"flat_store_b8", "flat_store_b8", "flat_store_byte", "flat_store_byte"},
     8,
     ConSanFlatSubwordPlacement::Low16},
    {{rdna4::kFlatStoreB16Vflat, gfx1250::kFlatStoreB16Vflat, cdna3::kFlatStoreShortFlat,
      cdna4::kFlatStoreShortFlat},
     {"flat_store_b16", "flat_store_b16", "flat_store_short", "flat_store_short"},
     16,
     ConSanFlatSubwordPlacement::Low16},
    {{rdna4::kFlatStoreD16HiB8Vflat, gfx1250::kFlatStoreD16HiB8Vflat,
      cdna3::kFlatStoreByteD16HiFlat, cdna4::kFlatStoreByteD16HiFlat},
     {"flat_store_d16_hi_b8", "flat_store_d16_hi_b8", "flat_store_byte_d16_hi",
      "flat_store_byte_d16_hi"},
     8,
     ConSanFlatSubwordPlacement::High16},
    {{rdna4::kFlatStoreD16HiB16Vflat, gfx1250::kFlatStoreD16HiB16Vflat,
      cdna3::kFlatStoreShortD16HiFlat, cdna4::kFlatStoreShortD16HiFlat},
     {"flat_store_d16_hi_b16", "flat_store_d16_hi_b16", "flat_store_short_d16_hi",
      "flat_store_short_d16_hi"},
     16,
     ConSanFlatSubwordPlacement::High16},
}};

std::vector<uint8_t> make_group_flat_load_code_object(const FlatSubwordTarget &target,
                                                      uint16_t opcode) {
  std::vector<uint32_t> text_words = {
      0xbe8001ebu, // s_mov_b64 s[0:1], SRC_SHARED_BASE
      build_v_mov_b32_e32(/*vdst=*/0, /*scalar s0=*/0, target.arch),
      build_v_mov_b32_e32(/*vdst=*/1, /*scalar s1=*/1, target.arch),
  };
  switch (target.arch) {
  case ROCJITSU_CODE_ARCH_RDNA4: {
    const auto load = rdna4::build_vflat(opcode, {.saddr = 124, .vdst = 2, .vaddr = 0});
    text_words.insert(text_words.end(), load.begin(), load.end());
    break;
  }
  case ROCJITSU_CODE_ARCH_GFX1250: {
    const auto load = gfx1250::build_vflat(opcode, {.saddr = 124, .vdst = 2, .vaddr = 0});
    text_words.insert(text_words.end(), load.begin(), load.end());
    break;
  }
  case ROCJITSU_CODE_ARCH_CDNA3: {
    const auto load = cdna3::build_flat(opcode, {.addr = 0, .vdst = 2});
    text_words.insert(text_words.end(), load.begin(), load.end());
    break;
  }
  case ROCJITSU_CODE_ARCH_CDNA4: {
    const auto load = cdna4::build_flat(opcode, {.addr = 0, .vdst = 2});
    text_words.insert(text_words.end(), load.begin(), load.end());
    break;
  }
  default:
    ADD_FAILURE() << "unsupported group-FLAT load test architecture";
    return {};
  }
  text_words.resize(1200, build_s_nop(0, target.arch));
  text_words.back() = build_s_endpgm(target.arch);

  switch (target.arch) {
  case ROCJITSU_CODE_ARCH_RDNA4:
    return make_rdna4_lds_code_object(text_words, "gfx1201_flat_d16_load");
  case ROCJITSU_CODE_ARCH_GFX1250:
    return make_gfx1250_code_object(text_words, "gfx1250_flat_d16_load");
  case ROCJITSU_CODE_ARCH_CDNA3:
    return make_cdna3_lds_code_object(text_words, "gfx942_flat_d16_load",
                                      /*vgpr_granulated=*/1u);
  case ROCJITSU_CODE_ARCH_CDNA4:
    return make_cdna4_lds_code_object(text_words, "gfx950_flat_d16_load",
                                      /*vgpr_granulated=*/1u);
  default:
    return {};
  }
}

std::vector<uint8_t> make_group_flat_d16_load_code_object(const FlatSubwordTarget &target,
                                                          const FlatD16LoadForm &form) {
  return make_group_flat_load_code_object(target, form.opcodes[target.target_index]);
}

std::vector<uint8_t> make_group_flat_d16_store_code_object(const FlatSubwordTarget &target,
                                                           const FlatSubwordStoreForm &form) {
  std::vector<uint32_t> text_words = {
      0xbe8001ebu, // s_mov_b64 s[0:1], SRC_SHARED_BASE
      build_v_mov_b32_e32(/*vdst=*/0, /*scalar s0=*/0, target.arch),
      build_v_mov_b32_e32(/*vdst=*/1, /*scalar s1=*/1, target.arch),
  };
  const uint16_t opcode = form.opcodes[target.target_index];
  switch (target.arch) {
  case ROCJITSU_CODE_ARCH_RDNA4: {
    const auto store = rdna4::build_vflat(opcode, {.saddr = 124, .vsrc = 2, .vaddr = 0});
    text_words.insert(text_words.end(), store.begin(), store.end());
    break;
  }
  case ROCJITSU_CODE_ARCH_GFX1250: {
    const auto store = gfx1250::build_vflat(opcode, {.saddr = 124, .vsrc = 2, .vaddr = 0});
    text_words.insert(text_words.end(), store.begin(), store.end());
    break;
  }
  case ROCJITSU_CODE_ARCH_CDNA3: {
    const auto store = cdna3::build_flat(opcode, {.addr = 0, .data = 2});
    text_words.insert(text_words.end(), store.begin(), store.end());
    break;
  }
  case ROCJITSU_CODE_ARCH_CDNA4: {
    const auto store = cdna4::build_flat(opcode, {.addr = 0, .data = 2});
    text_words.insert(text_words.end(), store.begin(), store.end());
    break;
  }
  default:
    ADD_FAILURE() << "unsupported group-FLAT store test architecture";
    return {};
  }
  text_words.resize(1200, build_s_nop(0, target.arch));
  text_words.back() = build_s_endpgm(target.arch);

  switch (target.arch) {
  case ROCJITSU_CODE_ARCH_RDNA4:
    return make_rdna4_lds_code_object(text_words, "gfx1201_flat_d16_store");
  case ROCJITSU_CODE_ARCH_GFX1250:
    return make_gfx1250_code_object(text_words, "gfx1250_flat_d16_store");
  case ROCJITSU_CODE_ARCH_CDNA3:
    return make_cdna3_lds_code_object(text_words, "gfx942_flat_d16_store",
                                      /*vgpr_granulated=*/1u);
  case ROCJITSU_CODE_ARCH_CDNA4:
    return make_cdna4_lds_code_object(text_words, "gfx950_flat_d16_store",
                                      /*vgpr_granulated=*/1u);
  default:
    return {};
  }
}

std::vector<uint32_t> expected_group_flat_store_readback(const FlatSubwordTarget &target,
                                                         uint32_t memory_width_bits,
                                                         uint16_t scratch_vgpr) {
  switch (target.arch) {
  case ROCJITSU_CODE_ARCH_RDNA4: {
    const uint16_t opcode =
        memory_width_bits == 8u ? rdna4::kFlatLoadU8Vflat : rdna4::kFlatLoadU16Vflat;
    const auto load = rdna4::build_vflat(
        opcode, {.saddr = 124, .vdst = static_cast<uint8_t>(scratch_vgpr), .vaddr = 0});
    return {load.begin(), load.end()};
  }
  case ROCJITSU_CODE_ARCH_GFX1250: {
    const uint16_t opcode =
        memory_width_bits == 8u ? gfx1250::kFlatLoadU8Vflat : gfx1250::kFlatLoadU16Vflat;
    const auto load = gfx1250::build_vflat(
        opcode, {.saddr = 124, .vdst = static_cast<uint8_t>(scratch_vgpr), .vaddr = 0});
    return {load.begin(), load.end()};
  }
  case ROCJITSU_CODE_ARCH_CDNA3: {
    const uint16_t opcode =
        memory_width_bits == 8u ? cdna3::kFlatLoadUbyteFlat : cdna3::kFlatLoadUshortFlat;
    const auto load =
        cdna3::build_flat(opcode, {.addr = 0, .vdst = static_cast<uint8_t>(scratch_vgpr)});
    return {load.begin(), load.end()};
  }
  case ROCJITSU_CODE_ARCH_CDNA4: {
    const uint16_t opcode =
        memory_width_bits == 8u ? cdna4::kFlatLoadUbyteFlat : cdna4::kFlatLoadUshortFlat;
    const auto load =
        cdna4::build_flat(opcode, {.addr = 0, .vdst = static_cast<uint8_t>(scratch_vgpr)});
    return {load.begin(), load.end()};
  }
  default:
    return {};
  }
}

std::optional<uint16_t> flat_check_trap_vcc_save_sgpr(const ConSanPatchInfo &patch) {
  if (patch.required_sgpr_count < 2u)
    return std::nullopt;
  return static_cast<uint16_t>(patch.required_sgpr_count - 2u);
}

constexpr uint64_t kFlatMismatchReportAddress = 0x8000u;
constexpr uint32_t kFlatMismatchReportMarker = 0xC05A4D16u;
constexpr uint32_t kFlatMismatchOriginal = 0x1234ABCDu;

struct FlatMismatchExecutionCase {
  uint32_t original;
  uint32_t duplicate;
  uint32_t expected_marker;
};

void execute_emitted_flat_mismatch_sequence(const FlatSubwordTarget &target,
                                            std::string_view case_label,
                                            std::span<const uint32_t> patched_words,
                                            std::span<const uint32_t> normalization,
                                            const ConSanPatchInfo &patch, uint16_t original_vgpr,
                                            uint16_t compare_lhs_vgpr, uint16_t duplicate_vgpr,
                                            std::span<const FlatMismatchExecutionCase> cases) {
  const auto sequence_begin = std::search(patched_words.begin(), patched_words.end(),
                                          normalization.begin(), normalization.end());
  ASSERT_NE(sequence_begin, patched_words.end());
  const auto compare = instrumentation::build_v_cmp_ne_u16_vcc(vector_source_vgpr(compare_lhs_vgpr),
                                                               duplicate_vgpr, target.arch);
  ASSERT_TRUE(compare);
  const auto compare_word = sequence_begin + static_cast<ptrdiff_t>(normalization.size());
  ASSERT_NE(compare_word, patched_words.end());
  EXPECT_EQ(*compare_word, *compare);
  const auto branch_word = std::next(compare_word);
  ASSERT_NE(branch_word, patched_words.end());

  const auto vcc_save = flat_check_trap_vcc_save_sgpr(patch);
  ASSERT_TRUE(vcc_save);
  const auto restore_vcc = instrumentation::build_s_mov_b64(kRdna4VccLo, *vcc_save, target.arch);
  ASSERT_TRUE(restore_vcc);
  auto sequence_end = std::next(branch_word);
  for (; sequence_end != patched_words.end(); ++sequence_end) {
    if (*sequence_end != *restore_vcc)
      continue;
    const ptrdiff_t action_words = sequence_end - std::next(branch_word);
    if (action_words > std::numeric_limits<int16_t>::max())
      continue;
    const auto skip_action =
        instrumentation::build_s_cbranch_vccz(static_cast<int16_t>(action_words), target.arch);
    if (skip_action && *skip_action == *branch_word)
      break;
  }
  ASSERT_NE(sequence_end, patched_words.end());
  const std::vector<uint32_t> sequence(sequence_begin, sequence_end);

  const std::string component_prefix =
      std::string(target.label) + "_consan_" + std::string(case_label);
  amdgpu::GpuMemory gpu_mem(component_prefix + "_mem");
  amdgpu::L2Cache l2(component_prefix + "_l2");
  l2.set_backing_memory(&gpu_mem);
  amdgpu::ComputeUnitCore::Config config{};
  config.arch = target.arch;
  config.num_wf_slots = 1;
  config.sgprs_per_wf = 106;
  config.vgprs_per_wf = 256;
  config.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create(component_prefix, config, &gpu_mem, &l2);
  ASSERT_NE(cu, nullptr);
  amdgpu::Wavefront *wave = cu->dispatch_wf(0, 0, config.sgprs_per_wf, config.vgprs_per_wf);
  ASSERT_NE(wave, nullptr);
  for (size_t i = 0; i < sequence.size(); ++i)
    gpu_mem.write32(i * sizeof(uint32_t), sequence[i]);

  const uint32_t vgpr_base = wave->vgpr_alloc().base;
  for (const FlatMismatchExecutionCase &test_case : cases) {
    SCOPED_TRACE("original=" + std::to_string(test_case.original) +
                 " duplicate=" + std::to_string(test_case.duplicate));
    gpu_mem.write32(kFlatMismatchReportAddress, 0u);
    wave->pc = 0u;
    wave->set_exec(1u);
    wave->set_vcc(0u);
    cu->write_vgpr(vgpr_base + original_vgpr, 0u, test_case.original);
    cu->write_vgpr(vgpr_base + duplicate_vgpr, 0u, test_case.duplicate);

    size_t steps = 0;
    while (wave->pc < sequence.size() * sizeof(uint32_t)) {
      ASSERT_LT(steps, sequence.size());
      ++steps;
      cu->step();
    }
    cu->flush_all();
    EXPECT_EQ(gpu_mem.read32(kFlatMismatchReportAddress), test_case.expected_marker);
  }

  if (!wave->is_halted())
    wave->halt();
}

TEST(ConSan, SuperColliderHighHalfGroupFlatMismatchActionExecutesOnEveryTarget) {
  constexpr std::array load_cases = {
      FlatMismatchExecutionCase{kFlatMismatchOriginal, 0x1234DCBAu, 0u},
      FlatMismatchExecutionCase{kFlatMismatchOriginal, 0x5678ABCDu, kFlatMismatchReportMarker},
  };

  for (const FlatSubwordTarget &target : kFlatSubwordTargets) {
    for (const FlatD16LoadForm &form : kFlatD16LoadForms) {
      if (form.placement != ConSanFlatSubwordPlacement::High16)
        continue;
      const std::string_view mnemonic = form.mnemonics[target.target_index];
      SCOPED_TRACE(std::string(target.label) + " " + std::string(mnemonic));
      const std::vector<uint8_t> bytes = make_group_flat_d16_load_code_object(target, form);
      ASSERT_FALSE(bytes.empty());
      ConSanOptions options;
      options.flavor = ConSanFlavor::SuperCollider;
      options.probe_flat_check_trap = true;
      options.flat_provenance_mode = ConSanFlatProvenanceMode::Strict;
      options.report_buffer_address = kFlatMismatchReportAddress;
      options.report_marker = kFlatMismatchReportMarker;
      options.max_patches = 1;

      const ConSanResult result = try_patch_consan(bytes, options);

      ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
      ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
      ASSERT_TRUE(result.final_validation_passed);
      ASSERT_EQ(result.kernels.size(), 1u);
      ASSERT_EQ(result.kernels.front().flat_sites.size(), 1u);
      const ConSanFlatSite &site = result.kernels.front().flat_sites.front();
      ASSERT_TRUE(site.dst_vgpr);
      const auto patch = std::ranges::find(result.patches, ConSanPatchKind::InlineFlatLoadCheckTrap,
                                           &ConSanPatchInfo::kind);
      ASSERT_NE(patch, result.patches.end());
      ASSERT_TRUE(patch->scratch_vgpr);
      ASSERT_LT(*patch->scratch_vgpr, 255u);

      AmdGpuCodeObject replacement(result.elf_bytes.data(), result.elf_bytes.size());
      ASSERT_TRUE(replacement.is_valid());
      ASSERT_EQ(replacement.text_sections().size(), 1u);
      const Section *text = replacement.text_sections().front();
      const auto words = std::span<const uint32_t>(reinterpret_cast<const uint32_t *>(text->data()),
                                                   text->size() / sizeof(uint32_t));
      const uint16_t comparison_scratch = static_cast<uint16_t>(*patch->scratch_vgpr + 1u);
      const auto select_original_high = instrumentation::build_v_lshrrev_b32(
          comparison_scratch, scalar_positive_inline_u32(16u), *site.dst_vgpr, target.arch);
      const auto select_duplicate_high = instrumentation::build_v_lshrrev_b32(
          *patch->scratch_vgpr, scalar_positive_inline_u32(16u), *patch->scratch_vgpr, target.arch);
      ASSERT_TRUE(select_original_high);
      ASSERT_TRUE(select_duplicate_high);
      const std::array normalization = {*select_original_high, *select_duplicate_high};
      execute_emitted_flat_mismatch_sequence(target, mnemonic, words, normalization, *patch,
                                             *site.dst_vgpr, comparison_scratch,
                                             *patch->scratch_vgpr, load_cases);
    }

    for (const FlatSubwordStoreForm &form : kFlatSubwordStoreForms) {
      if (form.placement != ConSanFlatSubwordPlacement::High16)
        continue;
      const std::string_view mnemonic = form.mnemonics[target.target_index];
      SCOPED_TRACE(std::string(target.label) + " " + std::string(mnemonic));
      const std::vector<uint8_t> bytes = make_group_flat_d16_store_code_object(target, form);
      ASSERT_FALSE(bytes.empty());
      ConSanOptions options;
      options.flavor = ConSanFlavor::SuperCollider;
      options.probe_flat_check_trap = true;
      options.flat_provenance_mode = ConSanFlatProvenanceMode::Strict;
      options.report_buffer_address = kFlatMismatchReportAddress;
      options.report_marker = kFlatMismatchReportMarker;
      options.max_patches = 1;

      const ConSanResult result = try_patch_consan(bytes, options);

      ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
      ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
      ASSERT_TRUE(result.final_validation_passed);
      ASSERT_EQ(result.kernels.size(), 1u);
      ASSERT_EQ(result.kernels.front().flat_sites.size(), 1u);
      const ConSanFlatSite &site = result.kernels.front().flat_sites.front();
      ASSERT_TRUE(site.data_vgpr);
      const auto patch = std::ranges::find(
          result.patches, ConSanPatchKind::InlineFlatStoreCheckTrap, &ConSanPatchInfo::kind);
      ASSERT_NE(patch, result.patches.end());
      ASSERT_TRUE(patch->scratch_vgpr);
      ASSERT_LT(*patch->scratch_vgpr, 255u);

      AmdGpuCodeObject replacement(result.elf_bytes.data(), result.elf_bytes.size());
      ASSERT_TRUE(replacement.is_valid());
      ASSERT_EQ(replacement.text_sections().size(), 1u);
      const Section *text = replacement.text_sections().front();
      const auto words = std::span<const uint32_t>(reinterpret_cast<const uint32_t *>(text->data()),
                                                   text->size() / sizeof(uint32_t));
      const uint16_t comparison_scratch = static_cast<uint16_t>(*patch->scratch_vgpr + 1u);
      const auto select_original_high = instrumentation::build_v_lshrrev_b32(
          comparison_scratch, scalar_positive_inline_u32(16u), *site.data_vgpr, target.arch);
      ASSERT_TRUE(select_original_high);
      std::vector<uint32_t> normalization = {*select_original_high};
      if (form.memory_width_bits == 8u) {
        const auto mask_original_byte = instrumentation::build_v_and_b32_literal(
            comparison_scratch, 0xffu, comparison_scratch, target.arch);
        ASSERT_TRUE(mask_original_byte);
        normalization.insert(normalization.end(), mask_original_byte->begin(),
                             mask_original_byte->end());
      }
      const uint32_t readback = form.memory_width_bits == 8u
                                    ? (kFlatMismatchOriginal >> 16u) & 0xffu
                                    : kFlatMismatchOriginal >> 16u;
      const std::array store_cases = {
          FlatMismatchExecutionCase{0x1234DCBAu, readback, 0u},
          FlatMismatchExecutionCase{kFlatMismatchOriginal, readback ^ 1u,
                                    kFlatMismatchReportMarker},
      };
      execute_emitted_flat_mismatch_sequence(target, mnemonic, words, normalization, *patch,
                                             *site.data_vgpr, comparison_scratch,
                                             *patch->scratch_vgpr, store_cases);
    }
  }
}

TEST(ConSan, SuperColliderSupportsEveryD16GroupFlatLoadOnEveryTarget) {
  for (const FlatSubwordTarget &target : kFlatSubwordTargets) {
    for (const FlatD16LoadForm &form : kFlatD16LoadForms) {
      const std::string_view expected_mnemonic = form.mnemonics[target.target_index];
      SCOPED_TRACE(std::string(target.label) + " " + std::string(expected_mnemonic));
      const std::vector<uint8_t> bytes = make_group_flat_d16_load_code_object(target, form);
      ASSERT_FALSE(bytes.empty());
      ConSanOptions options;
      options.flavor = ConSanFlavor::SuperCollider;
      options.probe_flat_check_trap = true;
      options.flat_provenance_mode = ConSanFlatProvenanceMode::Strict;
      options.report_buffer_address = 0x100000000ull;
      options.max_patches = 1;

      const ConSanResult result = try_patch_consan(bytes, options);

      ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
      ASSERT_EQ(result.kernels.size(), 1u);
      ASSERT_EQ(result.kernels.front().flat_sites.size(), 1u);
      const ConSanFlatSite &site = result.kernels.front().flat_sites.front();
      EXPECT_EQ(site.mnemonic, expected_mnemonic);
      EXPECT_EQ(site.kind, ConSanLdsAccessKind::Read);
      EXPECT_EQ(site.width_bits, form.memory_width_bits);
      EXPECT_EQ(site.address_space_hint, ConSanFlatAddressSpaceHint::Group);
      const auto semantics = consan_flat_load_subword_semantics(site.mnemonic);
      ASSERT_TRUE(semantics);
      EXPECT_EQ(semantics->memory_width_bits, form.memory_width_bits);
      EXPECT_EQ(semantics->placement, form.placement);
      EXPECT_TRUE(consan_supercollider_supports_flat_site(site, ConSanFlatProvenanceMode::Strict));
      ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
      EXPECT_TRUE(result.final_validation_passed);
      const auto patch = std::ranges::find(result.patches, ConSanPatchKind::InlineFlatLoadCheckTrap,
                                           &ConSanPatchInfo::kind);
      ASSERT_NE(patch, result.patches.end());
      ASSERT_TRUE(patch->scratch_vgpr);
      ASSERT_TRUE(site.dst_vgpr);
      ASSERT_FALSE(result.elf_bytes.empty());
      AmdGpuCodeObject replacement(result.elf_bytes.data(), result.elf_bytes.size());
      ASSERT_EQ(replacement.text_sections().size(), 1u);
      const Section *text = replacement.text_sections().front();
      const auto words = std::span<const uint32_t>(reinterpret_cast<const uint32_t *>(text->data()),
                                                   text->size() / sizeof(uint32_t));
      const auto vcc_save = flat_check_trap_vcc_save_sgpr(*patch);
      ASSERT_TRUE(vcc_save);
      const auto save_vcc = instrumentation::build_s_mov_b64(*vcc_save, kRdna4VccLo, target.arch);
      const auto restore_vcc =
          instrumentation::build_s_mov_b64(kRdna4VccLo, *vcc_save, target.arch);
      ASSERT_TRUE(save_vcc);
      ASSERT_TRUE(restore_vcc);
      EXPECT_NE(std::ranges::find(words, *save_vcc), words.end());
      EXPECT_NE(std::ranges::find(words, *restore_vcc), words.end());

      uint16_t compare_lhs = *site.dst_vgpr;
      if (form.placement == ConSanFlatSubwordPlacement::High16) {
        ASSERT_LT(*patch->scratch_vgpr, 255u);
        compare_lhs = static_cast<uint16_t>(*patch->scratch_vgpr + 1u);
        const auto select_original_high = instrumentation::build_v_lshrrev_b32(
            compare_lhs, scalar_positive_inline_u32(16u), *site.dst_vgpr, target.arch);
        const auto select_duplicate_high = instrumentation::build_v_lshrrev_b32(
            *patch->scratch_vgpr, scalar_positive_inline_u32(16u), *patch->scratch_vgpr,
            target.arch);
        ASSERT_TRUE(select_original_high);
        ASSERT_TRUE(select_duplicate_high);
        EXPECT_NE(std::ranges::find(words, *select_original_high), words.end());
        EXPECT_NE(std::ranges::find(words, *select_duplicate_high), words.end());
      }
      const auto compare = instrumentation::build_v_cmp_ne_u16_vcc(
          vector_source_vgpr(compare_lhs), *patch->scratch_vgpr, target.arch);
      ASSERT_TRUE(compare);
      EXPECT_NE(std::ranges::find(words, *compare), words.end());
    }
  }
}

TEST(ConSanMoi, EveryEngineSupportsEveryD16GroupFlatLoadOnEveryTarget) {
  constexpr std::array<ConSanMoiEngine, 3> kEngines = {
      ConSanMoiEngine::RecordReplay,
      ConSanMoiEngine::Sampled,
      ConSanMoiEngine::InlineShadow,
  };
  for (const FlatSubwordTarget &target : kFlatSubwordTargets) {
    for (const FlatD16LoadForm &form : kFlatD16LoadForms) {
      const std::string_view expected_mnemonic = form.mnemonics[target.target_index];
      const std::vector<uint8_t> bytes = make_group_flat_d16_load_code_object(target, form);
      ASSERT_FALSE(bytes.empty());
      for (ConSanMoiEngine engine : kEngines) {
        SCOPED_TRACE(std::string(target.label) + " " + std::string(expected_mnemonic) +
                     " engine=" + std::to_string(static_cast<uint32_t>(engine)));
        ConSanOptions options = moi_options(engine);
        options.flat_provenance_mode = ConSanFlatProvenanceMode::Strict;
        options.scratch_vgpr = 8;
        options.moi_exec_save_sgpr = engine == ConSanMoiEngine::InlineShadow ? 60u : 80u;
        options.moi_owner_vgpr = 40;
        options.moi_epoch_vgpr = 41;
        options.moi_report_buffer_address = 0x100000000ull;
        options.moi_report_buffer_size =
            engine == ConSanMoiEngine::RecordReplay ? consan_moi_report_buffer_min_bytes(1, 0, 0, 0)
            : engine == ConSanMoiEngine::Sampled    ? direct_sampled_report_bytes(1)
                                                    : kInlineShadowFullLdsReportBufferSize;
        options.moi_track_barriers = false;
        options.moi_track_atomics = false;
        options.max_patches = 1;

        const ConSanResult result = try_patch_consan(bytes, options);

        ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
        ASSERT_EQ(result.moi_candidates.size(), 1u);
        const ConSanMoiCandidate &candidate = result.moi_candidates.front();
        EXPECT_EQ(candidate.mnemonic, expected_mnemonic);
        EXPECT_EQ(candidate.source, ConSanMoiCandidateSource::FlatGroup);
        EXPECT_EQ(candidate.kind, ConSanLdsAccessKind::Read);
        EXPECT_EQ(candidate.width_bits, form.memory_width_bits);
        EXPECT_TRUE(consan_moi_supports_flat_access_mnemonic(candidate.mnemonic));
        ASSERT_EQ(result.site_dispositions.size(), 1u);
        EXPECT_EQ(result.site_dispositions.front().disposition, ConSanSiteDisposition::Supported);
        EXPECT_EQ(result.site_dispositions.front().reason, ConSanSiteDispositionReason::None);
        EXPECT_EQ(result.site_dispositions.front().lowering_outcome,
                  ConSanSiteLoweringOutcome::Patched);
        ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
        EXPECT_TRUE(result.final_validation_passed);
      }
    }
  }
}

TEST(ConSan, SuperColliderSupportsEverySubwordGroupFlatStoreOnEveryTarget) {
  for (const FlatSubwordTarget &target : kFlatSubwordTargets) {
    for (const FlatSubwordStoreForm &form : kFlatSubwordStoreForms) {
      const std::string_view expected_mnemonic = form.mnemonics[target.target_index];
      SCOPED_TRACE(std::string(target.label) + " " + std::string(expected_mnemonic));
      const std::vector<uint8_t> bytes = make_group_flat_d16_store_code_object(target, form);
      ASSERT_FALSE(bytes.empty());
      ConSanOptions options;
      options.flavor = ConSanFlavor::SuperCollider;
      options.probe_flat_check_trap = true;
      options.flat_provenance_mode = ConSanFlatProvenanceMode::Strict;
      options.report_buffer_address = 0x100000000ull;
      options.max_patches = 1;

      const ConSanResult result = try_patch_consan(bytes, options);

      ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
      ASSERT_EQ(result.kernels.size(), 1u);
      ASSERT_EQ(result.kernels.front().flat_sites.size(), 1u);
      const ConSanFlatSite &site = result.kernels.front().flat_sites.front();
      EXPECT_EQ(site.mnemonic, expected_mnemonic);
      EXPECT_EQ(site.kind, ConSanLdsAccessKind::Write);
      EXPECT_EQ(site.width_bits, form.memory_width_bits);
      EXPECT_EQ(site.address_space_hint, ConSanFlatAddressSpaceHint::Group);
      ASSERT_TRUE(site.data_vgpr);
      EXPECT_EQ(*site.data_vgpr, 2u);
      const auto semantics = consan_flat_store_subword_semantics(site.mnemonic);
      ASSERT_TRUE(semantics);
      EXPECT_EQ(semantics->memory_width_bits, form.memory_width_bits);
      EXPECT_EQ(semantics->placement, form.placement);
      EXPECT_TRUE(consan_supercollider_supports_flat_site(site, ConSanFlatProvenanceMode::Strict));
      ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
      EXPECT_TRUE(result.final_validation_passed);
      const auto patch = std::ranges::find(
          result.patches, ConSanPatchKind::InlineFlatStoreCheckTrap, &ConSanPatchInfo::kind);
      ASSERT_NE(patch, result.patches.end());
      ASSERT_TRUE(patch->scratch_vgpr);
      ASSERT_FALSE(result.elf_bytes.empty());
      AmdGpuCodeObject replacement(result.elf_bytes.data(), result.elf_bytes.size());
      ASSERT_EQ(replacement.text_sections().size(), 1u);
      const Section *text = replacement.text_sections().front();
      const auto words = std::span<const uint32_t>(reinterpret_cast<const uint32_t *>(text->data()),
                                                   text->size() / sizeof(uint32_t));
      const auto vcc_save = flat_check_trap_vcc_save_sgpr(*patch);
      ASSERT_TRUE(vcc_save);
      const auto save_vcc = instrumentation::build_s_mov_b64(*vcc_save, kRdna4VccLo, target.arch);
      const auto restore_vcc =
          instrumentation::build_s_mov_b64(kRdna4VccLo, *vcc_save, target.arch);
      ASSERT_TRUE(save_vcc);
      ASSERT_TRUE(restore_vcc);
      EXPECT_NE(std::ranges::find(words, *save_vcc), words.end());
      EXPECT_NE(std::ranges::find(words, *restore_vcc), words.end());

      const std::vector<uint32_t> readback =
          expected_group_flat_store_readback(target, form.memory_width_bits, *patch->scratch_vgpr);
      ASSERT_FALSE(readback.empty());
      EXPECT_TRUE(contains_subsequence(words, readback));

      uint16_t compare_lhs = *site.data_vgpr;
      if (form.placement == ConSanFlatSubwordPlacement::High16 || form.memory_width_bits == 8u) {
        ASSERT_LT(*patch->scratch_vgpr, 255u);
        const uint16_t comparison_scratch = static_cast<uint16_t>(*patch->scratch_vgpr + 1u);
        if (form.placement == ConSanFlatSubwordPlacement::High16) {
          const auto select_high = instrumentation::build_v_lshrrev_b32(
              comparison_scratch, scalar_positive_inline_u32(16u), *site.data_vgpr, target.arch);
          ASSERT_TRUE(select_high);
          EXPECT_NE(std::ranges::find(words, *select_high), words.end());
          compare_lhs = comparison_scratch;
        }
        if (form.memory_width_bits == 8u) {
          const auto mask_byte = instrumentation::build_v_and_b32_literal(comparison_scratch, 0xffu,
                                                                          compare_lhs, target.arch);
          ASSERT_TRUE(mask_byte);
          EXPECT_TRUE(contains_subsequence(words, *mask_byte));
          compare_lhs = comparison_scratch;
        }
      }
      const auto compare = instrumentation::build_v_cmp_ne_u16_vcc(
          vector_source_vgpr(compare_lhs), *patch->scratch_vgpr, target.arch);
      ASSERT_TRUE(compare);
      EXPECT_NE(std::ranges::find(words, *compare), words.end());
    }
  }
}

TEST(ConSanMoi, EveryEngineSupportsEverySubwordGroupFlatStoreOnEveryTarget) {
  constexpr std::array<ConSanMoiEngine, 3> kEngines = {
      ConSanMoiEngine::RecordReplay,
      ConSanMoiEngine::Sampled,
      ConSanMoiEngine::InlineShadow,
  };
  for (const FlatSubwordTarget &target : kFlatSubwordTargets) {
    for (const FlatSubwordStoreForm &form : kFlatSubwordStoreForms) {
      const std::string_view expected_mnemonic = form.mnemonics[target.target_index];
      const std::vector<uint8_t> bytes = make_group_flat_d16_store_code_object(target, form);
      ASSERT_FALSE(bytes.empty());
      for (ConSanMoiEngine engine : kEngines) {
        SCOPED_TRACE(std::string(target.label) + " " + std::string(expected_mnemonic) +
                     " engine=" + std::to_string(static_cast<uint32_t>(engine)));
        ConSanOptions options = moi_options(engine);
        options.flat_provenance_mode = ConSanFlatProvenanceMode::Strict;
        options.scratch_vgpr = 8;
        options.moi_exec_save_sgpr = engine == ConSanMoiEngine::InlineShadow ? 60u : 80u;
        options.moi_owner_vgpr = 40;
        options.moi_epoch_vgpr = 41;
        options.moi_report_buffer_address = 0x100000000ull;
        options.moi_report_buffer_size =
            engine == ConSanMoiEngine::RecordReplay ? consan_moi_report_buffer_min_bytes(1, 0, 0, 0)
            : engine == ConSanMoiEngine::Sampled    ? direct_sampled_report_bytes(1)
                                                    : kInlineShadowFullLdsReportBufferSize;
        options.moi_track_barriers = false;
        options.moi_track_atomics = false;
        options.max_patches = 1;

        const ConSanResult result = try_patch_consan(bytes, options);

        ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
        ASSERT_EQ(result.moi_candidates.size(), 1u);
        const ConSanMoiCandidate &candidate = result.moi_candidates.front();
        EXPECT_EQ(candidate.mnemonic, expected_mnemonic);
        EXPECT_EQ(candidate.source, ConSanMoiCandidateSource::FlatGroup);
        EXPECT_EQ(candidate.kind, ConSanLdsAccessKind::Write);
        EXPECT_EQ(candidate.width_bits, form.memory_width_bits);
        const auto semantics = consan_flat_store_subword_semantics(candidate.mnemonic);
        ASSERT_TRUE(semantics);
        EXPECT_EQ(semantics->placement, form.placement);
        EXPECT_TRUE(consan_moi_supports_flat_access_mnemonic(candidate.mnemonic));
        ASSERT_EQ(result.site_dispositions.size(), 1u);
        EXPECT_EQ(result.site_dispositions.front().disposition, ConSanSiteDisposition::Supported);
        EXPECT_EQ(result.site_dispositions.front().reason, ConSanSiteDispositionReason::None);
        EXPECT_EQ(result.site_dispositions.front().lowering_outcome,
                  ConSanSiteLoweringOutcome::Patched);
        ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
        EXPECT_TRUE(result.final_validation_passed);
      }
    }
  }
}

TEST(ConSanMoi, UnsupportedGroupFlatLoadRemainsInPreFilterLedger) {
  const FlatSubwordTarget &target = kFlatSubwordTargets.back();
  const std::vector<uint8_t> bytes =
      make_group_flat_load_code_object(target, cdna4::kFlatLoadDwordx3Flat);
  ASSERT_FALSE(bytes.empty());
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.flat_provenance_mode = ConSanFlatProvenanceMode::Strict;
  options.max_patches = 1;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  EXPECT_FALSE(result.modified);
  ASSERT_EQ(result.moi_candidates.size(), 1u);
  const ConSanMoiCandidate &candidate = result.moi_candidates.front();
  EXPECT_EQ(candidate.mnemonic, "flat_load_dwordx3");
  EXPECT_EQ(candidate.source, ConSanMoiCandidateSource::FlatGroup);
  EXPECT_EQ(candidate.kind, ConSanLdsAccessKind::Read);
  EXPECT_EQ(candidate.width_bits, 96u);
  EXPECT_FALSE(consan_moi_supports_flat_access_mnemonic("flat_load_dwordx3"));
  ASSERT_EQ(result.site_dispositions.size(), 1u);
  const ConSanSiteDispositionRecord &site = result.site_dispositions.front();
  EXPECT_EQ(site.site_kind, ConSanResourceSiteKind::Access);
  EXPECT_EQ(site.mnemonic, "flat_load_dwordx3");
  EXPECT_EQ(site.disposition, ConSanSiteDisposition::Unsupported);
  EXPECT_EQ(site.reason, ConSanSiteDispositionReason::UnsupportedMnemonic);
  EXPECT_EQ(site.lowering_outcome, ConSanSiteLoweringOutcome::Unsupported);
  EXPECT_EQ(site.lowering_reason, ConSanSiteLoweringReason::SemanticUnsupported);
}

TEST(ConSan, Cdna4SuperColliderEmitsGroupFlatCheckAndReport) {
  const std::vector<uint8_t> bytes = make_cdna4_padded_group_flat_code_object();
  ASSERT_FALSE(bytes.empty());
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_flat_check_trap = true;
  options.flat_provenance_mode = ConSanFlatProvenanceMode::Strict;
  options.report_buffer_address = 0x100000000ull;
  options.report_marker = 0x51c0u;
  options.max_patches = 1;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineFlatLoadCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 3u * sizeof(uint32_t));
  EXPECT_EQ(result.patches.front().scratch_vgpr, 3u);
  EXPECT_EQ(result.patches.front().original_size, 17u * sizeof(uint32_t));
  ASSERT_FALSE(result.elf_bytes.empty());
  AmdGpuCodeObject replacement(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_EQ(replacement.text_sections().size(), 1u);
  const Section *text = replacement.text_sections().front();
  const auto words = std::span<const uint32_t>(reinterpret_cast<const uint32_t *>(text->data()),
                                               text->size() / sizeof(uint32_t));
  const auto vcc_save = flat_check_trap_vcc_save_sgpr(result.patches.front());
  ASSERT_TRUE(vcc_save);
  const auto save_vcc =
      instrumentation::build_s_mov_b64(*vcc_save, kRdna4VccLo, ROCJITSU_CODE_ARCH_CDNA4);
  const auto restore_vcc =
      instrumentation::build_s_mov_b64(kRdna4VccLo, *vcc_save, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(save_vcc);
  ASSERT_TRUE(restore_vcc);
  EXPECT_NE(std::ranges::find(words, *save_vcc), words.end());
  EXPECT_NE(std::ranges::find(words, *restore_vcc), words.end());
  EXPECT_NE(std::ranges::find(words, 0xbf8c0070u), words.end()); // vmcnt(0), lgkmcnt(0).
  EXPECT_NE(std::ranges::find(words, *build_cdna4_v_cmp_ne_u32_vcc(vector_source_vgpr(2), 3,
                                                                   ROCJITSU_CODE_ARCH_CDNA4)),
            words.end());
  const auto report_store = build_cdna4_flat_store_b32(
      /*vaddr=*/4, /*vsrc=*/6, /*byte_offset=*/0, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(report_store);
  EXPECT_NE(std::ranges::find(words, (*report_store)[0]), words.end());
}

TEST(ConSan, Cdna4SuperColliderComparesGroupFlatShortValuesAsU16) {
  constexpr std::array<std::array<uint32_t, 2>, 2> kAccesses = {{
      {0xDC480000u, 0x02000000u}, // flat_load_ushort v2, v[0:1]
      {0xDC680000u, 0x00000200u}, // flat_store_short v[0:1], v2
  }};
  for (const auto &access : kAccesses) {
    std::vector<uint32_t> text_words = {
        0xbe8001ebu, // s_mov_b64 s[0:1], SRC_SHARED_BASE
        build_v_mov_b32_e32(/*vdst=*/0, /*scalar s0=*/0, ROCJITSU_CODE_ARCH_CDNA4),
        build_v_mov_b32_e32(/*vdst=*/1, /*scalar s1=*/1, ROCJITSU_CODE_ARCH_CDNA4),
        access[0],
        access[1],
    };
    text_words.resize(1200, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
    text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);
    const std::vector<uint8_t> bytes =
        make_cdna4_lds_code_object(text_words, "gfx950_flat_short_supercollider");
    ConSanOptions options;
    options.flavor = ConSanFlavor::SuperCollider;
    options.probe_flat_check_trap = true;
    options.flat_provenance_mode = ConSanFlatProvenanceMode::Strict;
    options.report_buffer_address = 0x100000000ull;
    options.max_patches = 1;

    const ConSanResult result = try_patch_consan(bytes, options);

    ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
    ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
    EXPECT_TRUE(result.final_validation_passed);
    ASSERT_EQ(result.patches.size(), 1u);
    AmdGpuCodeObject replacement(result.elf_bytes.data(), result.elf_bytes.size());
    const Section *text = replacement.text_sections().front();
    const auto words = std::span<const uint32_t>(reinterpret_cast<const uint32_t *>(text->data()),
                                                 text->size() / sizeof(uint32_t));
    const auto compare =
        build_cdna4_v_cmp_ne_u16_vcc(vector_source_vgpr(2), 3, ROCJITSU_CODE_ARCH_CDNA4);
    ASSERT_TRUE(compare);
    EXPECT_NE(std::ranges::find(words, *compare), words.end());
  }
}

TEST(ConSan, Cdna4SuperColliderFarGroupFlatFallsBackToDeadScalarWindow) {
  constexpr size_t kLargeTextWords = 33000u;
  constexpr uint16_t kCallTargetSgpr = 2u;
  constexpr uint16_t kReturnSgpr = 30u;
  constexpr uint16_t kLiteralOperand = 255u;
  constexpr uint16_t kInlineInt0 = 128u;
  constexpr uint32_t kFunctionDelta = 24u;
  const auto load = build_cdna4_flat_load_b32(
      /*vaddr=*/0, /*vdst=*/2, /*byte_offset=*/0, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(load);
  const std::array<uint32_t, 7> kernel_words = {
      build_s_getpc_b64(kCallTargetSgpr, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_add_u32(kCallTargetSgpr, kCallTargetSgpr, kLiteralOperand, ROCJITSU_CODE_ARCH_CDNA4),
      kFunctionDelta,
      build_s_addc_u32(kCallTargetSgpr + 1u, kCallTargetSgpr + 1u, kInlineInt0,
                       ROCJITSU_CODE_ARCH_CDNA4),
      build_s_swappc_b64(kReturnSgpr, kCallTargetSgpr, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_mov_b32(99, 99, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  std::vector<uint32_t> function_words = {
      0xbe8001ebu, // s_mov_b64 s[0:1], SRC_SHARED_BASE
      build_v_mov_b32_e32(/*vdst=*/0, /*scalar s0=*/0, ROCJITSU_CODE_ARCH_CDNA4),
      build_v_mov_b32_e32(/*vdst=*/1, /*scalar s1=*/1, ROCJITSU_CODE_ARCH_CDNA4),
      (*load)[0],
      (*load)[1],
  };
  // s99 makes the preferred fresh window begin at s100.  Only s[100:101]
  // remains there, which cannot also hold the two required single-register
  // saves.  The site nevertheless has a liveness-proven dead low window.
  function_words.resize(kLargeTextWords - 1u, build_s_mov_b32(99, 99, ROCJITSU_CODE_ARCH_CDNA4));
  function_words.push_back(build_s_setpc_b64(kReturnSgpr, ROCJITSU_CODE_ARCH_CDNA4));
  std::vector<uint8_t> bytes = make_rdna4_code_object_with_local_function(
      kernel_words, function_words, {}, /*vgpr_granulated=*/0u);
  mutate_elf_header(bytes,
                    [](Elf64_Ehdr &header) { header.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX950; });
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_flat_check_trap = true;
  options.flat_provenance_mode = ConSanFlatProvenanceMode::Strict;
  options.max_patches = 1;
  // Keep this test specific to the scalar-indirect fallback. The first five
  // filler words remain available for the displaced entry island; the rest models
  // text already owned by an earlier composition phase and cannot donate a
  // direct branch reservoir.
  options.preapplied_reserved_ranges.push_back(
      {.text_offset = sizeof(uint32_t) + kFunctionDelta + 10u * sizeof(uint32_t),
       .size = static_cast<uint32_t>((function_words.size() - 11u) * sizeof(uint32_t))});

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  const auto body = std::ranges::find(result.patches, ConSanPatchKind::LocalCaveFlatLoadCheckTrap,
                                      &ConSanPatchInfo::kind);
  ASSERT_NE(body, result.patches.end());
  ASSERT_EQ(body->owner_descriptor_file_offsets.size(), 1u);
  ASSERT_TRUE(body->indirect_saved_vcc_sgpr);
  ASSERT_TRUE(body->indirect_pc_sgpr);
  ASSERT_TRUE(body->indirect_saved_scc_sgpr);
  EXPECT_LT(*body->indirect_pc_sgpr, 100u);
  const auto ranges_overlap = [](uint16_t lhs_base, uint16_t lhs_width, uint16_t rhs_base,
                                 uint16_t rhs_width) {
    return static_cast<uint32_t>(lhs_base) < static_cast<uint32_t>(rhs_base) + rhs_width &&
           static_cast<uint32_t>(rhs_base) < static_cast<uint32_t>(lhs_base) + lhs_width;
  };
  EXPECT_FALSE(ranges_overlap(*body->indirect_saved_vcc_sgpr, 2u, *body->indirect_pc_sgpr, 2u));
  EXPECT_FALSE(
      ranges_overlap(*body->indirect_saved_scc_sgpr, 1u, *body->indirect_saved_vcc_sgpr, 2u));
  EXPECT_FALSE(ranges_overlap(*body->indirect_saved_scc_sgpr, 1u, *body->indirect_pc_sgpr, 2u));
  EXPECT_GE(body->indirect_required_sgpr_count,
            static_cast<uint16_t>(*body->indirect_saved_vcc_sgpr + 2u));
  EXPECT_GE(body->indirect_required_sgpr_count,
            static_cast<uint16_t>(*body->indirect_pc_sgpr + 2u));
  EXPECT_GE(body->indirect_required_sgpr_count,
            static_cast<uint16_t>(*body->indirect_saved_scc_sgpr + 1u));
  EXPECT_GT(body->trampoline_offset, body->anchor_offset);
}

TEST(ConSanMoi, Cdna4RecordAndInlineEmitStronglyClassifiedGroupFlatAccess) {
  const std::vector<uint8_t> bytes = make_cdna4_padded_group_flat_code_object();
  ASSERT_FALSE(bytes.empty());
  for (ConSanMoiEngine engine : {ConSanMoiEngine::RecordReplay, ConSanMoiEngine::InlineShadow}) {
    SCOPED_TRACE(static_cast<uint32_t>(engine));
    ConSanOptions options = moi_options(engine);
    options.flat_provenance_mode = ConSanFlatProvenanceMode::Strict;
    options.scratch_vgpr = 8;
    options.moi_owner_vgpr = 24;
    options.moi_epoch_vgpr = 25;
    if (engine == ConSanMoiEngine::InlineShadow) {
      options.moi_workgroup_key_vgpr = 26;
      options.moi_init_owner_epoch = true;
    }
    options.moi_report_buffer_address = 0x100000000ull;
    options.moi_report_buffer_size = engine == ConSanMoiEngine::InlineShadow
                                         ? kInlineShadowFullLdsReportBufferSize
                                         : consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

    const ConSanResult result = try_patch_consan(bytes, options);

    ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
    ASSERT_EQ(result.moi_candidates.size(), 1u);
    EXPECT_EQ(result.moi_candidates.front().source, ConSanMoiCandidateSource::FlatGroup);
    EXPECT_EQ(result.moi_candidates.front().size, 2u * sizeof(uint32_t));
    EXPECT_EQ(result.moi_candidates.front().mnemonic, "flat_load_dword");
    EXPECT_EQ(result.moi_candidates.front().raw_segment, 0u);
    ASSERT_TRUE(result.moi_candidates.front().addr_vgpr);
    EXPECT_EQ(*result.moi_candidates.front().addr_vgpr, 0u);
    EXPECT_EQ(result.moi_candidates.front().raw_ioffset, 0);
    ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
    EXPECT_TRUE(result.final_validation_passed);
    if (engine == ConSanMoiEngine::InlineShadow) {
      EXPECT_TRUE(result.resolved_moi_workgroup_key_vgpr);
      EXPECT_EQ(std::ranges::count(result.patches,
                                   ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue,
                                   &ConSanPatchInfo::kind),
                1);
    }
  }
}

TEST(ConSanMoi, Cdna4RecordReplayEmitsGroupFlatShortAccesses) {
  constexpr std::array<uint32_t, 2> kLoadUshort = {
      0xDC480000u, // flat_load_ushort v2, v[0:1]
      0x02000000u,
  };
  constexpr std::array<uint32_t, 2> kStoreShort = {
      0xDC680000u, // flat_store_short v[0:1], v2
      0x00000200u,
  };
  std::vector<uint32_t> text_words = {
      0xbe8001ebu, // s_mov_b64 s[0:1], SRC_SHARED_BASE
      build_v_mov_b32_e32(/*vdst=*/0, /*scalar s0=*/0, ROCJITSU_CODE_ARCH_CDNA4),
      build_v_mov_b32_e32(/*vdst=*/1, /*scalar s1=*/1, ROCJITSU_CODE_ARCH_CDNA4),
      kStoreShort[0],
      kStoreShort[1],
      kLoadUshort[0],
      kLoadUshort[1],
  };
  text_words.resize(1200, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4);
  const std::vector<uint8_t> bytes =
      make_cdna4_lds_code_object(text_words, "gfx950_flat_short_emission");
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.flat_provenance_mode = ConSanFlatProvenanceMode::Strict;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 24;
  options.moi_epoch_vgpr = 25;
  options.max_patches = 2;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(2, 0, 0, 0);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  ASSERT_EQ(result.moi_candidates.size(), 2u);
  EXPECT_EQ(result.moi_candidates[0].mnemonic, "flat_store_short");
  EXPECT_EQ(result.moi_candidates[0].width_bits, 16u);
  EXPECT_EQ(result.moi_candidates[1].mnemonic, "flat_load_ushort");
  EXPECT_EQ(result.moi_candidates[1].width_bits, 16u);
  EXPECT_EQ(std::ranges::count(result.site_dispositions, ConSanSiteDisposition::Supported,
                               &ConSanSiteDispositionRecord::disposition),
            2);
  EXPECT_EQ(
      std::ranges::count_if(result.patches,
                            [](const ConSanPatchInfo &patch) {
                              return patch.kind == ConSanPatchKind::InlineMoiAccessRecordStore ||
                                     patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore;
                            }),
      2)
      << "patches=" << testing::PrintToString(result.patches)
      << " warnings=" << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSanMoi, Gfx1250RecordReplayEmitsGroupFlatShortAccesses) {
  constexpr std::array<uint32_t, 3> kLoadU16 = {
      0xEC048000u, // flat_load_u16 v2, v[0:1]
      0x00000002u,
      0x00000000u,
  };
  constexpr std::array<uint32_t, 3> kStoreB16 = {
      0xEC064000u, // flat_store_b16 v[0:1], v2
      0x01000000u,
      0x00000000u,
  };
  std::vector<uint32_t> text_words = {
      0xbe8001ebu, // s_mov_b64 s[0:1], SRC_SHARED_BASE
      build_v_mov_b32_e32(/*vdst=*/0, /*scalar s0=*/0, ROCJITSU_CODE_ARCH_GFX1250),
      build_v_mov_b32_e32(/*vdst=*/1, /*scalar s1=*/1, ROCJITSU_CODE_ARCH_GFX1250),
      kStoreB16[0],
      kStoreB16[1],
      kStoreB16[2],
      kLoadU16[0],
      kLoadU16[1],
      kLoadU16[2],
  };
  text_words.resize(1200, build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250));
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250);
  const std::vector<uint8_t> bytes =
      make_gfx1250_code_object(text_words, "gfx1250_flat_short_emission");
  ConSanOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.flat_provenance_mode = ConSanFlatProvenanceMode::Strict;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 24;
  options.moi_epoch_vgpr = 25;
  options.max_patches = 2;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(2, 0, 0, 0);

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  ASSERT_EQ(result.moi_candidates.size(), 2u);
  EXPECT_EQ(result.moi_candidates[0].mnemonic, "flat_store_b16");
  EXPECT_EQ(result.moi_candidates[0].width_bits, 16u);
  EXPECT_EQ(result.moi_candidates[1].mnemonic, "flat_load_u16");
  EXPECT_EQ(result.moi_candidates[1].width_bits, 16u);
  EXPECT_EQ(std::ranges::count(result.site_dispositions, ConSanSiteDisposition::Supported,
                               &ConSanSiteDispositionRecord::disposition),
            2);
  EXPECT_EQ(
      std::ranges::count_if(result.patches,
                            [](const ConSanPatchInfo &patch) {
                              return patch.kind == ConSanPatchKind::InlineMoiAccessRecordStore ||
                                     patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore;
                            }),
      2)
      << "patches=" << testing::PrintToString(result.patches)
      << " warnings=" << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSan, Gfx1250SuperColliderChecksGroupFlatShortValues) {
  constexpr std::array<std::array<uint32_t, 3>, 2> kAccesses = {{
      {0xEC048000u, 0x00000002u, 0x00000000u}, // flat_load_u16 v2, v[0:1]
      {0xEC064000u, 0x01000000u, 0x00000000u}, // flat_store_b16 v[0:1], v2
  }};
  for (const auto &access : kAccesses) {
    std::vector<uint32_t> text_words = {
        0xbe8001ebu, // s_mov_b64 s[0:1], SRC_SHARED_BASE
        build_v_mov_b32_e32(/*vdst=*/0, /*scalar s0=*/0, ROCJITSU_CODE_ARCH_GFX1250),
        build_v_mov_b32_e32(/*vdst=*/1, /*scalar s1=*/1, ROCJITSU_CODE_ARCH_GFX1250),
        access[0],
        access[1],
        access[2],
    };
    text_words.resize(1200, build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250));
    text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250);
    const std::vector<uint8_t> bytes =
        make_gfx1250_code_object(text_words, "gfx1250_flat_short_supercollider");
    ConSanOptions options;
    options.flavor = ConSanFlavor::SuperCollider;
    options.probe_flat_check_trap = true;
    options.flat_provenance_mode = ConSanFlatProvenanceMode::Strict;
    options.report_buffer_address = 0x100000000ull;
    options.max_patches = 1;

    const ConSanResult result = try_patch_consan(bytes, options);

    ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
    ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
    EXPECT_TRUE(result.final_validation_passed);
    ASSERT_EQ(result.patches.size(), 1u);
  }
}

TEST(ConSan, InventoriesCdna4FlatAtomicAddressShape) {
  const auto atomic = build_cdna4_flat_atomic_add_u32(
      /*vaddr=*/2, /*vsrc=*/4, /*vdst=*/5, /*return_old_value=*/true,
      /*scope=*/2, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(atomic);
  const std::array<uint32_t, 3> text_words = {
      (*atomic)[0],
      (*atomic)[1],
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  const std::vector<uint8_t> bytes = make_cdna4_lds_code_object(text_words, "flat_atomic_probe");
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.kernels.size(), 1u);
  const ConSanKernelInfo &kernel = result.kernels.front();
  EXPECT_TRUE(kernel.decoded);
  EXPECT_EQ(kernel.stats.instruction_count, 2u);
  EXPECT_EQ(kernel.stats.flat_atomic_count, 1u);
  ASSERT_EQ(kernel.atomic_sites.size(), 1u);
  const ConSanAtomicSite &site = kernel.atomic_sites.front();
  EXPECT_EQ(site.mnemonic, "flat_atomic_add");
  EXPECT_EQ(site.size, 8u);
  EXPECT_EQ(site.width_bits, 32u);
  ASSERT_TRUE(site.addr_vgpr && site.data_vgpr && site.dst_vgpr);
  EXPECT_EQ(*site.addr_vgpr, 2u);
  EXPECT_EQ(*site.data_vgpr, 4u);
  EXPECT_EQ(*site.dst_vgpr, 5u);
  ASSERT_TRUE(site.raw_saddr && site.raw_vaddr && site.raw_vsrc && site.raw_vdst &&
              site.raw_ioffset && site.raw_scope && site.returns_old_value);
  EXPECT_EQ(*site.raw_saddr, 0u);
  EXPECT_EQ(*site.raw_vaddr, 2u);
  EXPECT_EQ(*site.raw_vsrc, 4u);
  EXPECT_EQ(*site.raw_vdst, 5u);
  EXPECT_EQ(*site.raw_ioffset, 0);
  EXPECT_EQ(*site.raw_scope, 2u);
  EXPECT_TRUE(*site.returns_old_value);

  const ConSanMoiAtomicAddressPlan plan = plan_consan_moi_atomic_address(
      site, /*scratch_vgpr=*/8, /*scratch_vgpr_count=*/24, ConSanRegisterAllocationSource::Explicit,
      ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(plan.supported());
  EXPECT_EQ(plan.kind, ConSanMoiAtomicAddressKind::FlatGuestPair);
  const auto materialization = build_consan_moi_atomic_address_materialization(
      plan, /*vcc_save_sgpr=*/80, /*scc_save_sgpr=*/82, ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(materialization);
  EXPECT_TRUE(materialization->empty());
}

TEST(ConSan, AssociatesCdna4CompilerAtomicAcquireReleaseShape) {
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
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4));
  const std::vector<uint8_t> bytes =
      make_cdna4_lds_code_object(text_words, "atomic_acquire_release");
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.sync_events.size(), 3u);
  EXPECT_EQ(result.sync_events[0].mnemonic, "buffer_wbl2");
  EXPECT_EQ(result.sync_events[1].mnemonic, "flat_atomic_add");
  EXPECT_EQ(result.sync_events[2].mnemonic, "buffer_inv");
  ASSERT_EQ(result.sync_sequences.size(), 1u);
  const ConSanSyncSequence &sequence = result.sync_sequences.front();
  EXPECT_EQ(sequence.kind, ConSanSyncSequenceKind::Atomic);
  EXPECT_EQ(sequence.operation, ConSanSyncOperation::AtomicRmw);
  EXPECT_EQ(sequence.memory_role, ConSanSyncMemoryRole::AcquireRelease);
  EXPECT_EQ(sequence.memory_role_confidence, ConSanSemanticConfidence::Conservative);
  EXPECT_EQ(sequence.confidence, ConSanSemanticConfidence::Conservative);
  EXPECT_EQ(sequence.rmw_outcome, ConSanSyncRmwOutcome::ReturnsOldValue);
  ASSERT_EQ(sequence.member_event_identities.size(), 3u);
  EXPECT_EQ(sequence.begin_text_offset, 0u);
  EXPECT_EQ(sequence.end_text_offset, 32u);
}

TEST(ConSan, InventoriesRdna4GlobalAtomicScopeAndReturnBits) {
  const std::vector<uint8_t> bytes = make_rdna4_global_atomic_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.kernels.size(), 1u);

  const ConSanKernelInfo &kernel = result.kernels.front();
  EXPECT_TRUE(kernel.decoded);
  EXPECT_EQ(kernel.code_size, 16u);
  EXPECT_EQ(kernel.stats.instruction_count, 2u);
  EXPECT_EQ(kernel.stats.global_memory_count, 1u);
  EXPECT_EQ(kernel.stats.lds_atomic_count, 0u);
  EXPECT_TRUE(kernel.lds_sites.empty());
  EXPECT_TRUE(kernel.fence_sites.empty());
  ASSERT_EQ(kernel.atomic_sites.size(), 1u);

  const ConSanAtomicSite &atomic = kernel.atomic_sites.front();
  EXPECT_EQ(atomic.address_space_hint, ConSanAtomicAddressSpaceHint::Global);
  EXPECT_EQ(atomic.mnemonic, "global_atomic_add_f32");
  EXPECT_EQ(atomic.text_offset, 0u);
  EXPECT_EQ(atomic.file_offset, 0x100u);
  EXPECT_EQ(atomic.size, 12u);
  EXPECT_EQ(atomic.width_bits, 32u);
  ASSERT_TRUE(atomic.dst_vgpr);
  ASSERT_TRUE(atomic.addr_vgpr);
  ASSERT_TRUE(atomic.data_vgpr);
  ASSERT_TRUE(atomic.saddr_sgpr);
  EXPECT_EQ(*atomic.dst_vgpr, 0u);
  EXPECT_EQ(*atomic.addr_vgpr, 2u);
  EXPECT_EQ(*atomic.data_vgpr, 1u);
  EXPECT_EQ(*atomic.saddr_sgpr, 4u);
  ASSERT_TRUE(atomic.raw_saddr);
  ASSERT_TRUE(atomic.raw_vaddr);
  ASSERT_TRUE(atomic.raw_vsrc);
  ASSERT_TRUE(atomic.raw_vdst);
  ASSERT_TRUE(atomic.raw_ioffset);
  ASSERT_TRUE(atomic.raw_scope);
  ASSERT_TRUE(atomic.raw_th);
  ASSERT_TRUE(atomic.returns_old_value);
  EXPECT_EQ(*atomic.raw_saddr, 4u);
  EXPECT_EQ(*atomic.raw_vaddr, 2u);
  EXPECT_EQ(*atomic.raw_vsrc, 1u);
  EXPECT_EQ(*atomic.raw_vdst, 0u);
  EXPECT_EQ(*atomic.raw_ioffset, 0);
  EXPECT_EQ(*atomic.raw_scope, 2u);
  EXPECT_EQ(*atomic.raw_th, 1u);
  EXPECT_TRUE(*atomic.returns_old_value);

  ASSERT_EQ(result.sync_events.size(), 1u);
  const ConSanSyncEvent &event = result.sync_events.front();
  EXPECT_EQ(event.kind, ConSanSyncEventKind::Atomic);
  EXPECT_EQ(event.operation, ConSanSyncOperation::AtomicRmw);
  EXPECT_EQ(event.address_source, ConSanSyncAddressSource::GlobalScalarVector);
  EXPECT_EQ(event.memory_role, ConSanSyncMemoryRole::Unknown);
  EXPECT_EQ(event.rmw_outcome, ConSanSyncRmwOutcome::ReturnsOldValue);
  EXPECT_EQ(event.confidence, ConSanSemanticConfidence::Conservative);
  ASSERT_TRUE(event.static_byte_offset);
  EXPECT_EQ(*event.static_byte_offset, 0);
  ASSERT_TRUE(event.raw_scope);
  EXPECT_EQ(*event.raw_scope, 2u);
  EXPECT_NE(event.identity.find("|kernel=lds_probe|event=atomic|"), std::string::npos);
}

TEST(ConSan, SyncInventoryRetainsUnsupportedFlatAtomicWithoutProvenance) {
  const std::vector<uint8_t> bytes = make_rdna4_flat_atomic_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.kernels.front().atomic_sites.size(), 1u);
  EXPECT_EQ(result.kernels.front().atomic_sites.front().address_space_hint,
            ConSanAtomicAddressSpaceHint::FlatUnknown);
  ASSERT_EQ(result.sync_events.size(), 1u);
  const ConSanSyncEvent &event = result.sync_events.front();
  EXPECT_EQ(event.address_source, ConSanSyncAddressSource::FlatVector);
  EXPECT_EQ(event.confidence, ConSanSemanticConfidence::Unsupported);
  EXPECT_EQ(event.memory_role_confidence, ConSanSemanticConfidence::Unsupported);
  EXPECT_NE(event.confidence_reason.find("no usable provenance"), std::string::npos);
}

TEST(ConSan, SyncSequencesAssociatePinnedGlobalAtomicCachePattern) {
  const auto wait_store = build_s_wait_storecnt0(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(wait_store);
  const std::vector<uint32_t> text_words = {
      0xEE0B0000u, 0x00000000u, 0x00000000u, // global_wb
      *wait_store, 0xEE158004u, 0x00980000u,
      0x00000002u, // global_atomic_add_f32 v0, v2, v1, s[4:5]
      *wait_store, 0xEE0AC000u, 0x00000000u, 0x00000000u, // global_inv
      0xBFB00000u,                                        // s_endpgm
  };
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const auto result = try_patch_consan(make_rdna4_lds_code_object(text_words), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.sync_events.size(), 3u);
  ASSERT_EQ(result.sync_sequences.size(), 1u);
  const ConSanSyncSequence &sequence = result.sync_sequences.front();
  EXPECT_EQ(sequence.kind, ConSanSyncSequenceKind::Atomic);
  EXPECT_EQ(sequence.operation, ConSanSyncOperation::AtomicRmw);
  EXPECT_EQ(sequence.memory_role, ConSanSyncMemoryRole::AcquireRelease);
  EXPECT_EQ(sequence.address_source, ConSanSyncAddressSource::GlobalScalarVector);
  EXPECT_EQ(sequence.rmw_outcome, ConSanSyncRmwOutcome::ReturnsOldValue);
  EXPECT_EQ(sequence.confidence, ConSanSemanticConfidence::Conservative);
  EXPECT_EQ(sequence.memory_role_confidence, ConSanSemanticConfidence::Conservative);
  ASSERT_EQ(sequence.member_event_identities.size(), 3u);
  EXPECT_EQ(sequence.member_event_identities[0], result.sync_events[0].identity);
  EXPECT_EQ(sequence.member_event_identities[1], result.sync_events[1].identity);
  EXPECT_EQ(sequence.member_event_identities[2], result.sync_events[2].identity);
  ASSERT_TRUE(sequence.raw_scope);
  EXPECT_EQ(*sequence.raw_scope, 2u);

  const std::array<uint32_t, 7> release_words = {
      0xEE0B0000u, 0x00000000u, 0x00000000u, // global_wb
      0xEE158004u, 0x00980000u,
      0x00000002u, // global_atomic_add_f32 v0, v2, v1, s[4:5]
      0xBFB00000u, // s_endpgm
  };
  const auto release = try_patch_consan(make_rdna4_lds_code_object(release_words), options);
  ASSERT_TRUE(release.errors.empty()) << (release.errors.empty() ? "" : release.errors.front());
  ASSERT_EQ(release.sync_sequences.size(), 1u);
  EXPECT_EQ(release.sync_sequences.front().memory_role, ConSanSyncMemoryRole::Release);

  const std::array<uint32_t, 7> acquire_words = {
      0xEE158004u, 0x00980000u,
      0x00000002u,                           // global_atomic_add_f32 v0, v2, v1, s[4:5]
      0xEE0AC000u, 0x00000000u, 0x00000000u, // global_inv
      0xBFB00000u,                           // s_endpgm
  };
  const auto acquire = try_patch_consan(make_rdna4_lds_code_object(acquire_words), options);
  ASSERT_TRUE(acquire.errors.empty()) << (acquire.errors.empty() ? "" : acquire.errors.front());
  ASSERT_EQ(acquire.sync_sequences.size(), 1u);
  EXPECT_EQ(acquire.sync_sequences.front().memory_role, ConSanSyncMemoryRole::Acquire);
}

TEST(ConSan, SyncSequencesAssociateRetainedBoundedAtomicAcquireShapes) {
  constexpr std::array<uint32_t, 3> rmw = {0xEE0D400Cu, 0x01980002u, 0x00000002u};
  constexpr std::array<uint32_t, 3> exchange = {0xEE0CC006u, 0x00180004u, 0x00000005u};
  constexpr std::array<uint32_t, 3> failed_cas = {0xEE0D0006u, 0x00180000u, 0x00000001u};
  constexpr std::array<uint32_t, 8> rmw_bookkeeping = {
      0xBF88FF9Eu, // s_wait_alu
      0x8C7E007Eu, // s_or_b32 exec restore
      0xBF118008u, // s_cmp_lg_u64
      0xBFC00000u, // s_wait_loadcnt 0
      0x7E000500u, // v_readfirstlane_b32
      0x980080C1u, // s_cselect_b32
      0xBF028016u, // s_cmp_gt_i32
      0xBFC90000u, // s_wait_storecnt_dscnt 0
  };
  constexpr std::array<uint32_t, 2> short_bookkeeping = {
      0x980180C1u, // s_cselect_b32
      0xBFC00000u, // s_wait_loadcnt 0
  };
  const std::array<std::vector<uint8_t>, 4> fixtures = {
      make_rdna4_bounded_atomic_acquire_code_object(rmw, rmw_bookkeeping, "rmw_acquire"),
      make_rdna4_bounded_atomic_acquire_code_object(exchange, short_bookkeeping,
                                                    "exchange_acquire"),
      make_rdna4_successful_cas_self_loop_acquire_code_object(),
      make_rdna4_bounded_atomic_acquire_code_object(failed_cas, short_bookkeeping,
                                                    "failed_cas_acquire"),
  };
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  for (size_t index = 0; index < fixtures.size(); ++index) {
    SCOPED_TRACE(index);
    const ConSanResult result = try_patch_consan(fixtures[index], options);
    ASSERT_TRUE(consan_patch_succeeded(result));
    const auto sequence = std::ranges::find_if(result.sync_sequences, [](const auto &item) {
      return item.kind == ConSanSyncSequenceKind::Atomic &&
             item.memory_role == ConSanSyncMemoryRole::Acquire;
    });
    ASSERT_NE(sequence, result.sync_sequences.end());
    EXPECT_EQ(sequence->memory_role_confidence, ConSanSemanticConfidence::Conservative);
    EXPECT_EQ(sequence->member_event_identities.size(), 2u);
    EXPECT_NE(sequence->identity.find("|acquire-cache="), std::string::npos);
  }
}

TEST(ConSan, SyncSequencesAssociateBoundedAtomicAcquireAtFallthroughJoin) {
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const ConSanResult result =
      try_patch_consan(make_rdna4_atomic_acquire_fallthrough_join_code_object(), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  const auto sequence = std::ranges::find_if(result.sync_sequences, [](const auto &item) {
    return item.kind == ConSanSyncSequenceKind::Atomic &&
           item.memory_role == ConSanSyncMemoryRole::Acquire;
  });
  ASSERT_NE(sequence, result.sync_sequences.end());
  EXPECT_EQ(sequence->operation, ConSanSyncOperation::AtomicRmw);
  EXPECT_EQ(sequence->memory_role_confidence, ConSanSemanticConfidence::Conservative);
  EXPECT_EQ(sequence->member_event_identities.size(), 2u);
  EXPECT_NE(sequence->identity.find("|acquire-cache="), std::string::npos);
}

TEST(ConSan, BoundedAtomicAcquireAssociationRejectsUnprovenShapes) {
  constexpr std::array<uint32_t, 3> atomic = {0xEE0D400Cu, 0x01980002u, 0x00000002u};
  constexpr std::array<uint32_t, 3> load = {0xEE050006u, 0x00080001u, 0x00000000u};
  const auto acquire_count = [](const std::vector<uint8_t> &bytes) {
    ConSanOptions options;
    options.flavor = ConSanFlavor::SuperCollider;
    const ConSanResult result = try_patch_consan(bytes, options);
    return std::ranges::count_if(result.sync_sequences, [](const auto &item) {
      return item.kind == ConSanSyncSequenceKind::Atomic &&
             item.memory_role == ConSanSyncMemoryRole::Acquire;
    });
  };
  const auto fixture = [&](std::span<const uint32_t> middle,
                           std::string_view name = "rejected_atomic_acquire") {
    return make_rdna4_bounded_atomic_acquire_code_object(atomic, middle, name);
  };
  const std::array<uint32_t, 1> missing_wait = {0xBF800000u}; // s_nop
  std::array<uint32_t, 17> overlong{};
  overlong.fill(0xBF800000u);
  overlong.front() = 0xBFC00000u;
  const std::array<uint32_t, 4> intervening_memory = {0xBFC00000u, load[0], load[1], load[2]};
  const std::array<uint32_t, 2> intervening_barrier = {0xBFC00000u, 0xBF940000u};
  const std::array<uint32_t, 2> scalar_clause = {0xBFC00000u, 0xBF850001u};
  const std::array<uint32_t, 2> forward_branch = {0xBFC00000u, 0xBFA00000u};
  const std::array<std::vector<uint8_t>, 6> rejected = {
      fixture(missing_wait),        fixture(overlong),      fixture(intervening_memory),
      fixture(intervening_barrier), fixture(scalar_clause), fixture(forward_branch),
  };
  for (size_t index = 0; index < rejected.size(); ++index) {
    SCOPED_TRACE(index);
    EXPECT_EQ(acquire_count(rejected[index]), 0u);
  }
}

TEST(ConSan, SyncSequencesAssociateExactReleaseWaitWithNoReturnAtomic) {
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const ConSanResult result =
      try_patch_consan(make_rdna4_release_wait_no_return_bitwise_code_object(), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.sync_events.size(), 1u);
  ASSERT_EQ(result.sync_sequences.size(), 1u);
  const ConSanSyncSequence &sequence = result.sync_sequences.front();
  EXPECT_EQ(sequence.kind, ConSanSyncSequenceKind::Atomic);
  EXPECT_EQ(sequence.operation, ConSanSyncOperation::AtomicRmw);
  EXPECT_EQ(sequence.memory_role, ConSanSyncMemoryRole::Release);
  EXPECT_EQ(sequence.memory_role_confidence, ConSanSemanticConfidence::Conservative);
  EXPECT_EQ(sequence.rmw_outcome, ConSanSyncRmwOutcome::NoReturn);
  ASSERT_TRUE(sequence.release_wait_text_offset);
  EXPECT_EQ(*sequence.release_wait_text_offset, 0u);
  EXPECT_EQ(sequence.begin_text_offset, 0u);
  EXPECT_EQ(sequence.end_text_offset, 16u);
  ASSERT_EQ(sequence.member_event_identities.size(), 1u);
  EXPECT_EQ(sequence.member_event_identities.front(), result.sync_events.front().identity);
  EXPECT_NE(sequence.identity.find("|release-wait=pc=0x"), std::string::npos);

  const ConSanResult nonzero =
      try_patch_consan(make_rdna4_release_wait_no_return_bitwise_code_object(0xbfc90001u), options);
  ASSERT_TRUE(nonzero.errors.empty()) << testing::PrintToString(nonzero.errors);
  ASSERT_EQ(nonzero.sync_sequences.size(), 1u);
  EXPECT_EQ(nonzero.sync_sequences.front().memory_role, ConSanSyncMemoryRole::Unknown);
  EXPECT_FALSE(nonzero.sync_sequences.front().release_wait_text_offset);
}

TEST(ConSan, SyncSequencesUpgradeAcquireWithExactReleaseWaitToAcquireRelease) {
  const auto wait_store = build_s_wait_storecnt0(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(wait_store);
  const std::array<uint32_t, 9> text_words = {
      *wait_store, 0xBFC80000u, // s_wait_loadcnt_dscnt 0
      0xEE0D400Cu, 0x01980002u,
      0x00000002u, // global_atomic_add_u32 v12, v2, v3, s[4:5], return old
      0xBFC80000u, // s_wait_loadcnt_dscnt 0
      0xEE0AC000u, 0x00000000u, 0x00000000u, // global_inv
  };
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const ConSanResult result = try_patch_consan(make_rdna4_lds_code_object(text_words), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.sync_sequences.size(), 1u);
  const ConSanSyncSequence &sequence = result.sync_sequences.front();
  EXPECT_EQ(sequence.kind, ConSanSyncSequenceKind::Atomic);
  EXPECT_EQ(sequence.memory_role, ConSanSyncMemoryRole::AcquireRelease);
  EXPECT_EQ(sequence.memory_role_confidence, ConSanSemanticConfidence::Conservative);
  ASSERT_TRUE(sequence.release_wait_text_offset);
  EXPECT_EQ(*sequence.release_wait_text_offset, 0u);
  EXPECT_EQ(sequence.begin_text_offset, 0u);
  EXPECT_NE(sequence.identity.find("|release-wait=pc=0x"), std::string::npos);
  EXPECT_NE(sequence.identity.find("|acquire-cache="), std::string::npos);
}

TEST(ConSan, MoiFenceSelectionCarriesUniqueAtomicCommunicationEvent) {
  const auto wait_store = build_s_wait_storecnt0(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(wait_store);
  const std::vector<uint32_t> text_words = {
      0xEE0B0000u, 0x00000000u, 0x00000000u, // global_wb
      *wait_store, 0xEE158004u, 0x00980000u,
      0x00000002u, // global_atomic_add_f32 v0, v2, v1, s[4:5]
      *wait_store, 0xEE0AC000u, 0x00000000u, 0x00000000u, // global_inv
      0xBFB00000u,                                        // s_endpgm
  };
  ConSanOptions options = moi_options();

  const ConSanResult result = try_patch_consan(make_rdna4_lds_code_object(text_words), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.sync_events.size(), 3u);
  ASSERT_EQ(result.sync_sequences.size(), 1u);
  ASSERT_EQ(result.moi_fence_candidates.size(), 2u);
  const std::string &communication_identity = result.sync_events[1].identity;
  const ConSanMoiFenceCandidate &release = result.moi_fence_candidates[0];
  const ConSanMoiFenceCandidate &acquire = result.moi_fence_candidates[1];
  EXPECT_TRUE(release.eligible) << release.rejection_reason;
  EXPECT_TRUE(acquire.eligible) << acquire.rejection_reason;
  EXPECT_EQ(release.memory_role, ConSanSyncMemoryRole::Release);
  EXPECT_EQ(acquire.memory_role, ConSanSyncMemoryRole::Acquire);
  EXPECT_EQ(release.sequence_identity, result.sync_sequences.front().identity);
  EXPECT_EQ(acquire.sequence_identity, result.sync_sequences.front().identity);
  EXPECT_EQ(release.communication_event_identity, communication_identity);
  EXPECT_EQ(acquire.communication_event_identity, communication_identity);
  EXPECT_EQ(release.communication_address_source, ConSanSyncAddressSource::GlobalScalarVector);
  EXPECT_EQ(acquire.communication_address_source, ConSanSyncAddressSource::GlobalScalarVector);
  ASSERT_TRUE(release.raw_scope);
  ASSERT_TRUE(acquire.raw_scope);
  EXPECT_EQ(*release.raw_scope, 2u);
  EXPECT_EQ(*acquire.raw_scope, 2u);
  EXPECT_FALSE(release.identity.empty());
  EXPECT_FALSE(acquire.identity.empty());
}

TEST(ConSan, MoiFenceSelectionRejectsUnassociatedCacheOperations) {
  const auto wait_store = build_s_wait_storecnt0(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(wait_store);
  const std::array<uint32_t, 11> text_words = {
      0xEE0B0000u, 0x00000000u, 0x00000000u, // global_wb
      *wait_store,
      0xBE804EC1u,                                        // s_barrier_signal -1
      0xBF94FFFFu,                                        // s_barrier_wait -1
      *wait_store, 0xEE0AC000u, 0x00000000u, 0x00000000u, // global_inv
      0xBFB00000u,                                        // s_endpgm
  };
  ConSanOptions options = moi_options();

  const ConSanResult result = try_patch_consan(make_rdna4_lds_code_object(text_words), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.moi_fence_candidates.size(), 2u);
  for (const ConSanMoiFenceCandidate &candidate : result.moi_fence_candidates) {
    EXPECT_FALSE(candidate.eligible);
    EXPECT_EQ(candidate.rejection_reason,
              "fence-not-associated-with-addressed-communication-sequence");
    EXPECT_TRUE(candidate.communication_event_identity.empty());
    EXPECT_FALSE(candidate.raw_scope);
  }
}

TEST(ConSan, SyncSequencesRejectNonWaitInsideAtomicCachePattern) {
  const std::vector<uint32_t> text_words = {
      0xEE0B0000u,
      0x00000000u,
      0x00000000u, // global_wb
      build_v_mov_b32_e32(/*vdst=*/7, vector_source_vgpr(7), ROCJITSU_CODE_ARCH_RDNA4),
      0xEE158004u,
      0x00980000u,
      0x00000002u, // global_atomic_add_f32 v0, v2, v1, s[4:5]
      0xBFB00000u, // s_endpgm
  };
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const auto result = try_patch_consan(make_rdna4_lds_code_object(text_words), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.sync_events.size(), 2u);
  ASSERT_EQ(result.sync_sequences.size(), 2u);
  EXPECT_EQ(result.sync_sequences[0].kind, ConSanSyncSequenceKind::Fence);
  EXPECT_EQ(result.sync_sequences[1].kind, ConSanSyncSequenceKind::Atomic);
  EXPECT_EQ(result.sync_sequences[1].memory_role, ConSanSyncMemoryRole::Unknown);
  ASSERT_EQ(result.moi_fence_candidates.size(), 1u);
  EXPECT_FALSE(result.moi_fence_candidates.front().eligible);
  EXPECT_EQ(result.moi_fence_candidates.front().rejection_reason,
            "fence-not-associated-with-addressed-communication-sequence");
  EXPECT_TRUE(result.moi_fence_candidates.front().communication_event_identity.empty());
}

TEST(ConSan, SyncSequencesDoNotPairBarrierAcrossAtomic) {
  const std::array<uint32_t, 6> text_words = {
      0xBE804EC1u, // s_barrier_signal -1
      0xEE158004u, 0x00980000u,
      0x00000002u, // global_atomic_add_f32 v0, v2, v1, s[4:5]
      0xBF94FFFFu, // s_barrier_wait -1
      0xBFB00000u, // s_endpgm
  };
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult result = try_patch_consan(make_rdna4_lds_code_object(text_words), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.sync_sequences.size(), 3u);
  EXPECT_EQ(result.sync_sequences[0].operation, ConSanSyncOperation::BarrierSignal);
  EXPECT_EQ(result.sync_sequences[0].confidence, ConSanSemanticConfidence::Ambiguous);
  EXPECT_EQ(result.sync_sequences[1].operation, ConSanSyncOperation::AtomicRmw);
  EXPECT_EQ(result.sync_sequences[1].memory_role, ConSanSyncMemoryRole::Unknown);
  EXPECT_EQ(result.sync_sequences[2].operation, ConSanSyncOperation::BarrierWait);
  EXPECT_EQ(result.sync_sequences[2].confidence, ConSanSemanticConfidence::Ambiguous);
}

TEST(ConSan, SyncSequencesDoNotAssociateCacheOperationAcrossBarrier) {
  const auto wait_store = build_s_wait_storecnt0(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(wait_store);
  const std::array<uint32_t, 14> text_words = {
      0xEE0B0000u, 0x00000000u, 0x00000000u, // global_wb
      *wait_store,
      0xBE804EC1u, // s_barrier_signal -1
      0xBF94FFFFu, // s_barrier_wait -1
      *wait_store, 0xEE158004u, 0x00980000u,
      0x00000002u,                           // global_atomic_add_f32 v0, v2, v1, s[4:5]
      0xEE0AC000u, 0x00000000u, 0x00000000u, // global_inv
      0xBFB00000u,                           // s_endpgm
  };
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult result = try_patch_consan(make_rdna4_lds_code_object(text_words), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.sync_sequences.size(), 3u);
  EXPECT_EQ(result.sync_sequences[0].kind, ConSanSyncSequenceKind::Fence);
  EXPECT_EQ(result.sync_sequences[0].memory_role, ConSanSyncMemoryRole::Unknown);
  EXPECT_EQ(result.sync_sequences[1].operation, ConSanSyncOperation::BarrierFull);
  EXPECT_EQ(result.sync_sequences[1].memory_role, ConSanSyncMemoryRole::AcquireRelease);
  EXPECT_EQ(result.sync_sequences[2].operation, ConSanSyncOperation::AtomicRmw);
  // The global_wb remains separated by the barrier, while the exact wait
  // immediately before the atomic independently proves its release half.
  EXPECT_EQ(result.sync_sequences[2].memory_role, ConSanSyncMemoryRole::AcquireRelease);
  EXPECT_NE(result.sync_sequences[2].identity.find("|release-wait=pc=0x"), std::string::npos);
  ASSERT_EQ(result.sync_sequences[2].member_event_identities.size(), 2u);
}

TEST(ConSan, SyncSequencesRetainAtomicCacheAssociationBeforeBarrier) {
  const auto wait_store = build_s_wait_storecnt0(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(wait_store);
  const std::array<uint32_t, 14> text_words = {
      0xEE0B0000u, 0x00000000u, 0x00000000u, // global_wb
      *wait_store, 0xEE158004u, 0x00980000u,
      0x00000002u, // global_atomic_add_f32 v0, v2, v1, s[4:5]
      *wait_store, 0xEE0AC000u, 0x00000000u, 0x00000000u, // global_inv
      0xBE804EC1u,                                        // s_barrier_signal -1
      0xBF94FFFFu,                                        // s_barrier_wait -1
      0xBFB00000u,                                        // s_endpgm
  };
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult result = try_patch_consan(make_rdna4_lds_code_object(text_words), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.sync_sequences.size(), 2u);
  EXPECT_EQ(result.sync_sequences[0].operation, ConSanSyncOperation::AtomicRmw);
  EXPECT_EQ(result.sync_sequences[0].memory_role, ConSanSyncMemoryRole::AcquireRelease);
  ASSERT_EQ(result.sync_sequences[0].member_event_identities.size(), 3u);
  EXPECT_EQ(result.sync_sequences[1].operation, ConSanSyncOperation::BarrierFull);
  EXPECT_EQ(result.sync_sequences[1].memory_role, ConSanSyncMemoryRole::AcquireRelease);
  ASSERT_EQ(result.sync_sequences[1].member_event_identities.size(), 2u);
}

TEST(ConSan, SyncSequencesKeepCacheOperationsSeparateAroundBarrier) {
  const auto wait_store = build_s_wait_storecnt0(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(wait_store);
  const std::array<uint32_t, 11> text_words = {
      0xEE0B0000u, 0x00000000u, 0x00000000u, // global_wb
      *wait_store,
      0xBE804EC1u,                                        // s_barrier_signal -1
      0xBF94FFFFu,                                        // s_barrier_wait -1
      *wait_store, 0xEE0AC000u, 0x00000000u, 0x00000000u, // global_inv
      0xBFB00000u,                                        // s_endpgm
  };
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult result = try_patch_consan(make_rdna4_lds_code_object(text_words), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.sync_sequences.size(), 3u);
  EXPECT_EQ(result.sync_sequences[0].kind, ConSanSyncSequenceKind::Fence);
  EXPECT_EQ(result.sync_sequences[0].memory_role, ConSanSyncMemoryRole::Unknown);
  EXPECT_EQ(result.sync_sequences[1].operation, ConSanSyncOperation::BarrierFull);
  EXPECT_EQ(result.sync_sequences[1].memory_role, ConSanSyncMemoryRole::AcquireRelease);
  EXPECT_EQ(result.sync_sequences[2].kind, ConSanSyncSequenceKind::Fence);
  EXPECT_EQ(result.sync_sequences[2].memory_role, ConSanSyncMemoryRole::Unknown);
}

TEST(ConSan, SyncInventoryMarksMaybeGroupFlatAtomicAmbiguous) {
  const auto atomic = build_flat_atomic_add_u32_vaddr_vsrc_vdst(
      /*vaddr=*/0, /*vsrc=*/2, /*vdst=*/3, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(atomic);
  const std::array<uint32_t, 9> text_words = {
      0xBE8001EBu,               // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u,  0x00000080u, // v_mov_b32_e64 v0, 0
      0xD5810001u,  0x00000001u, // v_mov_b32_e64 v1, s1
      (*atomic)[0], (*atomic)[1], (*atomic)[2],
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.kernels.front().atomic_sites.size(), 1u);
  EXPECT_EQ(result.kernels.front().atomic_sites.front().address_space_hint,
            ConSanAtomicAddressSpaceHint::FlatMaybeGroup);
  ASSERT_EQ(result.sync_events.size(), 1u);
  const ConSanSyncEvent &event = result.sync_events.front();
  EXPECT_EQ(event.address_source, ConSanSyncAddressSource::FlatVector);
  EXPECT_EQ(event.confidence, ConSanSemanticConfidence::Ambiguous);
  EXPECT_NE(event.confidence_reason.find("not statically distinguishable"), std::string::npos);
}

TEST(ConSan, Gfx1250AtomicInventoryPreservesAddressAndOrderingFields) {
  const auto atomic = build_gfx1250_flat_atomic_add_u32(
      /*vaddr=*/2, /*vsrc=*/4, /*vdst=*/2, /*return_old_value=*/true, /*scope=*/2,
      ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(atomic);
  EXPECT_EQ(*atomic, (std::array<uint32_t, 3>{0xEC0D407Cu, 0x02180002u, 0x00000002u}));
  const std::array<uint32_t, 4> text_words = {
      (*atomic)[0], (*atomic)[1], (*atomic)[2],
      0xBFB00000u, // s_endpgm
  };
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const ConSanResult result = try_patch_consan(make_gfx1250_code_object(text_words), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.kernels.front().atomic_sites.size(), 1u);
  const ConSanAtomicSite &site = result.kernels.front().atomic_sites.front();
  EXPECT_EQ(site.mnemonic, "flat_atomic_add_u32");
  EXPECT_EQ(site.size, 3u * sizeof(uint32_t));
  EXPECT_EQ(site.width_bits, 32u);
  EXPECT_EQ(site.addr_vgpr, 2u);
  EXPECT_EQ(site.data_vgpr, 4u);
  EXPECT_EQ(site.dst_vgpr, 2u);
  EXPECT_EQ(site.raw_saddr, static_cast<uint32_t>(gfx1250::OPR_SREG_NULL));
  EXPECT_EQ(site.raw_vaddr, 2u);
  EXPECT_EQ(site.raw_vsrc, 4u);
  EXPECT_EQ(site.raw_vdst, 2u);
  EXPECT_EQ(site.raw_ioffset, 0);
  EXPECT_EQ(site.raw_scope, 2u);
  EXPECT_EQ(site.raw_th, 1u);
  EXPECT_EQ(site.returns_old_value, true);
}

TEST(ConSan, SyncSequencesPreserveBasicBlockBoundaryBetweenAdjacentEvents) {
  const std::array<uint32_t, 4> text_words = {
      0xBE804EC1u,                                 // s_barrier_signal -1
      build_s_branch(0, ROCJITSU_CODE_ARCH_RDNA4), // immediately following instruction
      0xBF94FFFFu,                                 // s_barrier_wait -1
      0xBFB00000u,                                 // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.sync_events.size(), 2u);
  ASSERT_EQ(result.sync_sequences.size(), 2u);
  ASSERT_TRUE(result.sync_sequences[0].basic_block_index);
  ASSERT_TRUE(result.sync_sequences[1].basic_block_index);
  EXPECT_NE(result.sync_sequences[0].basic_block_index, result.sync_sequences[1].basic_block_index);
  EXPECT_EQ(result.sync_sequences[0].member_event_identities.size(), 1u);
  EXPECT_EQ(result.sync_sequences[1].member_event_identities.size(), 1u);
  EXPECT_EQ(result.sync_sequences[0].confidence, ConSanSemanticConfidence::Ambiguous);
  EXPECT_EQ(result.sync_sequences[1].confidence, ConSanSemanticConfidence::Ambiguous);
}

TEST(ConSan, SyncSequencesAssociateOnlyImmediateSameBlockBarrierPair) {
  const std::array<uint32_t, 3> paired_words = {
      0xBE804EC1u, // s_barrier_signal -1
      0xBF94FFFFu, // s_barrier_wait -1
      0xBFB00000u, // s_endpgm
  };
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  const auto paired = try_patch_consan(make_rdna4_lds_code_object(paired_words), options);

  ASSERT_TRUE(paired.errors.empty()) << (paired.errors.empty() ? "" : paired.errors.front());
  ASSERT_EQ(paired.sync_events.size(), 2u);
  ASSERT_EQ(paired.sync_sequences.size(), 1u);
  const ConSanSyncSequence &barrier = paired.sync_sequences.front();
  EXPECT_EQ(barrier.kind, ConSanSyncSequenceKind::Barrier);
  EXPECT_EQ(barrier.operation, ConSanSyncOperation::BarrierFull);
  EXPECT_EQ(barrier.memory_role, ConSanSyncMemoryRole::AcquireRelease);
  EXPECT_EQ(barrier.confidence, ConSanSemanticConfidence::Conservative);
  ASSERT_TRUE(barrier.barrier_id);
  EXPECT_EQ(*barrier.barrier_id, -1);
  EXPECT_EQ(barrier.barrier_operand_source, ConSanBarrierSite::OperandSource::Immediate);
  EXPECT_EQ(barrier.barrier_scope, ConSanBarrierSite::Scope::Workgroup);
  ASSERT_EQ(barrier.member_event_identities.size(), 2u);
  EXPECT_EQ(barrier.member_event_identities[0], paired.sync_events[0].identity);
  EXPECT_EQ(barrier.member_event_identities[1], paired.sync_events[1].identity);
  ASSERT_EQ(paired.fault_sites.size(), 2u);
  ASSERT_TRUE(paired.fault_sites[0].sync_sequence_identity);
  ASSERT_TRUE(paired.fault_sites[1].sync_sequence_identity);
  EXPECT_EQ(*paired.fault_sites[0].sync_sequence_identity, barrier.identity);
  EXPECT_EQ(*paired.fault_sites[1].sync_sequence_identity, barrier.identity);
  EXPECT_NE(paired.fault_sites[0].decoded_operands.find("barrier_id=-1"), std::string::npos);
  EXPECT_NE(paired.fault_sites[0].decoded_operands.find("operand_source=immediate"),
            std::string::npos);
  EXPECT_NE(paired.fault_sites[0].decoded_operands.find("scope=workgroup"), std::string::npos);
  const auto repeated = try_patch_consan(make_rdna4_lds_code_object(paired_words), options);
  ASSERT_EQ(repeated.sync_sequences.size(), 1u);
  EXPECT_EQ(repeated.sync_sequences.front().identity, barrier.identity);

  const std::array<uint32_t, 4> intervened_words = {
      0xBE804EC1u, // s_barrier_signal -1
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
      0xBF94FFFFu, // s_barrier_wait -1
      0xBFB00000u, // s_endpgm
  };
  const auto intervened = try_patch_consan(make_rdna4_lds_code_object(intervened_words), options);
  ASSERT_TRUE(intervened.errors.empty())
      << (intervened.errors.empty() ? "" : intervened.errors.front());
  ASSERT_EQ(intervened.sync_sequences.size(), 2u);
  EXPECT_EQ(intervened.sync_sequences[0].operation, ConSanSyncOperation::BarrierSignal);
  EXPECT_EQ(intervened.sync_sequences[1].operation, ConSanSyncOperation::BarrierWait);
  EXPECT_EQ(intervened.sync_sequences[0].confidence, ConSanSemanticConfidence::Ambiguous);
  EXPECT_EQ(intervened.sync_sequences[1].confidence, ConSanSemanticConfidence::Ambiguous);

  const std::array<uint32_t, 3> reversed_words = {
      0xBF94FFFFu, // s_barrier_wait -1
      0xBE804EC1u, // s_barrier_signal -1
      0xBFB00000u, // s_endpgm
  };
  const auto reversed = try_patch_consan(make_rdna4_lds_code_object(reversed_words), options);
  ASSERT_TRUE(reversed.errors.empty()) << (reversed.errors.empty() ? "" : reversed.errors.front());
  ASSERT_EQ(reversed.sync_sequences.size(), 2u);
  EXPECT_EQ(reversed.sync_sequences[0].operation, ConSanSyncOperation::BarrierWait);
  EXPECT_EQ(reversed.sync_sequences[1].operation, ConSanSyncOperation::BarrierSignal);
  EXPECT_EQ(reversed.sync_sequences[0].confidence, ConSanSemanticConfidence::Ambiguous);
  EXPECT_EQ(reversed.sync_sequences[1].confidence, ConSanSemanticConfidence::Ambiguous);
}

TEST(ConSan, SyncSequencesAssociateRetainedNonadjacentBarrierPairConservatively) {
  const std::array<uint32_t, 8> retained_words = {
      0xBE804EC1u,              // s_barrier_signal -1
      0x4A160503u,              // v_add_nc_u32_e32
      0xA98EFF00u, 0x00000090u, // s_add_nc_u64
      0xBEA10080u,              // s_mov_b32 s33, 0
      0xBE9F007Eu,              // s_mov_b32 s31, exec_lo
      0xBF94FFFFu,              // s_barrier_wait -1
      0xBFB00000u,              // s_endpgm
  };
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult retained =
      try_patch_consan(make_rdna4_lds_code_object(retained_words), options);

  ASSERT_TRUE(retained.errors.empty()) << testing::PrintToString(retained.errors);
  ASSERT_EQ(retained.sync_sequences.size(), 1u);
  const ConSanSyncSequence &pair = retained.sync_sequences.front();
  EXPECT_EQ(pair.operation, ConSanSyncOperation::BarrierFull);
  EXPECT_EQ(pair.memory_role, ConSanSyncMemoryRole::AcquireRelease);
  EXPECT_EQ(pair.confidence, ConSanSemanticConfidence::Conservative);
  EXPECT_EQ(pair.member_event_identities.size(), 2u);
  EXPECT_EQ(pair.begin_text_offset, 0u);
  EXPECT_EQ(pair.end_text_offset, 7u * sizeof(uint32_t));
  EXPECT_NE(pair.confidence_reason.find("bounded same-block"), std::string::npos);

  const auto full_pair_count = [&](std::span<const uint32_t> words) {
    const ConSanResult result = try_patch_consan(make_rdna4_lds_code_object(words), options);
    return std::ranges::count(result.sync_sequences, ConSanSyncOperation::BarrierFull,
                              &ConSanSyncSequence::operation);
  };
  const std::array<uint32_t, 4> intervening_barrier = {
      0xBE804EC1u, // s_barrier_signal -1
      0xBE804EC3u, // s_barrier_signal -3
      0xBF94FFFFu, // s_barrier_wait -1
      0xBFB00000u,
  };
  const std::array<uint32_t, 4> intervening_control = {
      0xBE804EC1u, build_s_branch(0, ROCJITSU_CODE_ARCH_RDNA4), 0xBF94FFFFu, 0xBFB00000u};
  const std::array<uint32_t, 4> mismatched_id = {0xBE804EC1u, 0x4A160503u, 0xBF94FFFEu,
                                                 0xBFB00000u};
  const std::array<uint32_t, 5> intervening_memory = {0xBE804EC1u, 0xD8340000u,
                                                      0x00000000u, // ds_store_b32
                                                      0xBF94FFFFu, 0xBFB00000u};
  const std::array<uint32_t, 4> intervening_call = {
      0xBE804EC1u, pack_sopk(/*s_call_b64=*/0x14, /*sdst=*/30, /*simm16=*/0), 0xBF94FFFFu,
      0xBFB00000u};
  const std::array<uint32_t, 8> excessive_bookkeeping = {
      0xBE804EC1u, 0xBEA10080u, 0xBEA10080u,
      0xBEA10080u, 0xBEA10080u, 0xBEA10080u, // Fifth s_mov_b32 exceeds the instruction bound.
      0xBF94FFFFu, 0xBFB00000u};
  const std::array<uint32_t, 4> ambiguous_same_id_signal = {0xBE804EC1u, 0xBE804EC1u, 0xBF94FFFFu,
                                                            0xBFB00000u};
  const std::array<uint32_t, 4> mismatched_scope = {
      0xBE804EC3u, // cluster-scoped s_barrier_signal -3
      0x4A160503u,
      0xBF94FFFFu, // workgroup-scoped s_barrier_wait -1
      0xBFB00000u};
  const std::array<uint32_t, 4> broad_move_family = {
      0xBE804EC1u, 0xBE8001EBu, // s_mov_b64 is not in the retained whitelist.
      0xBF94FFFFu, 0xBFB00000u};
  EXPECT_EQ(full_pair_count(intervening_barrier), 0u);
  EXPECT_EQ(full_pair_count(intervening_control), 0u);
  EXPECT_EQ(full_pair_count(mismatched_id), 0u);
  EXPECT_EQ(full_pair_count(intervening_memory), 0u);
  EXPECT_EQ(full_pair_count(intervening_call), 0u);
  EXPECT_EQ(full_pair_count(excessive_bookkeeping), 0u);
  EXPECT_EQ(full_pair_count(ambiguous_same_id_signal), 0u);
  EXPECT_EQ(full_pair_count(mismatched_scope), 0u);
  EXPECT_EQ(full_pair_count(broad_move_family), 0u);
}

TEST(ConSan, AllProfilesAbortOnlyStaticallyUnmatchedImmediateBarrierWait) {
  const std::array<uint32_t, 2> unmatched_words = {
      0xBF94FFFFu, // s_barrier_wait -1 without a signal
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> unmatched =
      make_rdna4_lds_code_object(unmatched_words, "unmatched_wait_abort");
  const std::array<ConSanMoiEngine, 3> engines = {
      ConSanMoiEngine::RecordReplay,
      ConSanMoiEngine::InlineShadow,
      ConSanMoiEngine::Sampled,
  };
  for (size_t profile = 0; profile != 4; ++profile) {
    ConSanOptions options;
    options.flavor = profile == 0 ? ConSanFlavor::SuperCollider : ConSanFlavor::Moi;
    if (profile != 0)
      options.moi_engine = engines[profile - 1];
    options.abort_unmatched_barrier_wait = true;
    const ConSanResult result = try_patch_consan(unmatched, options);
    ASSERT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid)
        << profile << testing::PrintToString(result.errors)
        << testing::PrintToString(result.warnings);
    ASSERT_EQ(result.patches.size(), 1u) << profile;
    const ConSanPatchInfo &patch = result.patches.front();
    EXPECT_EQ(patch.kind, ConSanPatchKind::InlineMalformedBarrierAbort);
    EXPECT_EQ(patch.phase, ConSanPatchPhase::Instrumentation);
    EXPECT_EQ(patch.anchor_offset, 0u);
    EXPECT_EQ(patch.original_size, sizeof(uint32_t));
    AmdGpuCodeObject replacement(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(replacement.is_valid());
    uint32_t replacement_word = 0;
    std::memcpy(&replacement_word, replacement.text_sections().front()->data(),
                sizeof(replacement_word));
    EXPECT_EQ(replacement_word, build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4));
    EXPECT_TRUE(validate_consan_modified_elf(unmatched, result).empty());
  }

  const std::array<uint32_t, 5> matched_words = {
      0xBE804EC1u, // s_barrier_signal -1
      0xBF800000u, // s_nop 0
      0xBF800000u, // s_nop 0
      0xBF94FFFFu, // s_barrier_wait -1
      0xBFB00000u, // s_endpgm
  };
  ConSanOptions matched_options;
  matched_options.flavor = ConSanFlavor::SuperCollider;
  matched_options.abort_unmatched_barrier_wait = true;
  const ConSanResult matched =
      try_patch_consan(make_rdna4_lds_code_object(matched_words, "matched_wait"), matched_options);
  EXPECT_EQ(matched.outcome, ConSanTransformOutcome::Unchanged);
  EXPECT_TRUE(matched.patches.empty());

  const std::array<uint32_t, 2> signal_only_words = {
      0xBE804EC1u, // unmatched signal is completing and must not be rewritten
      0xBFB00000u,
  };
  const ConSanResult signal_only = try_patch_consan(
      make_rdna4_lds_code_object(signal_only_words, "unmatched_signal"), matched_options);
  EXPECT_EQ(signal_only.outcome, ConSanTransformOutcome::Unchanged);
  EXPECT_TRUE(signal_only.patches.empty());
}

TEST(ConSan, SyncSequencesRejectAdjacentBarrierPairWithMismatchedIds) {
  const std::array<uint32_t, 3> text_words = {
      0xBE804EC1u, // s_barrier_signal -1
      0xBF94FFFEu, // s_barrier_wait -2
      0xBFB00000u, // s_endpgm
  };
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult result = try_patch_consan(make_rdna4_lds_code_object(text_words), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.sync_events.size(), 2u);
  ASSERT_EQ(result.sync_sequences.size(), 2u);
  ASSERT_TRUE(result.sync_events[0].barrier_id);
  ASSERT_TRUE(result.sync_events[1].barrier_id);
  EXPECT_EQ(*result.sync_events[0].barrier_id, -1);
  EXPECT_EQ(*result.sync_events[1].barrier_id, -2);
  EXPECT_EQ(result.sync_events[0].barrier_scope, ConSanBarrierSite::Scope::Workgroup);
  EXPECT_EQ(result.sync_events[1].barrier_scope, ConSanBarrierSite::Scope::Workgroup);
  EXPECT_EQ(result.sync_sequences[0].operation, ConSanSyncOperation::BarrierSignal);
  EXPECT_EQ(result.sync_sequences[1].operation, ConSanSyncOperation::BarrierWait);
  EXPECT_EQ(result.sync_sequences[0].confidence, ConSanSemanticConfidence::Ambiguous);
  EXPECT_EQ(result.sync_sequences[1].confidence, ConSanSemanticConfidence::Ambiguous);
}

TEST(ConSan, SyncInventoryDecodesClusterAndNamedWorkgroupBarrierScopes) {
  const std::array<uint32_t, 5> text_words = {
      0xBE804EC3u, // s_barrier_signal -3
      0xBF94FFFDu, // s_barrier_wait -3
      0xBE804E81u, // s_barrier_signal 1
      0xBF940001u, // s_barrier_wait 1
      0xBFB00000u, // s_endpgm
  };
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult result = try_patch_consan(make_rdna4_lds_code_object(text_words), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.sync_sequences.size(), 2u);
  ASSERT_TRUE(result.sync_sequences[0].barrier_id);
  EXPECT_EQ(*result.sync_sequences[0].barrier_id, -3);
  EXPECT_EQ(result.sync_sequences[0].barrier_scope, ConSanBarrierSite::Scope::Cluster);
  EXPECT_EQ(result.sync_sequences[0].operation, ConSanSyncOperation::BarrierFull);
  ASSERT_TRUE(result.sync_sequences[1].barrier_id);
  EXPECT_EQ(*result.sync_sequences[1].barrier_id, 1);
  EXPECT_EQ(result.sync_sequences[1].barrier_scope, ConSanBarrierSite::Scope::Workgroup);
  EXPECT_EQ(result.sync_sequences[1].operation, ConSanSyncOperation::BarrierFull);
}

TEST(ConSan, SyncSequencesAssociateClusterBarrierAcrossConditionalTriangle) {
  const auto bypass_signal = build_s_cbranch_scc1(/*offset_dwords=*/1, ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(bypass_signal);
  const std::array<uint32_t, 6> text_words = {
      *bypass_signal, // Guard either bypasses or falls through the signal arm.
      0xBE804EC3u,    // s_barrier_signal -3
      0x3600009Fu,    // v_and_b32_e32 v0, 31, v0
      0xBF048475u,    // s_cmp_lt_i32 ttmp9, 4
      0xBF94FFFDu,    // s_barrier_wait -3
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250),
  };
  const ConSanResult result =
      try_patch_consan(make_gfx1250_code_object(text_words, "cluster_triangle"), moi_options());

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  ASSERT_EQ(result.sync_sequences.size(), 1u);
  const ConSanSyncSequence &pair = result.sync_sequences.front();
  EXPECT_EQ(pair.operation, ConSanSyncOperation::BarrierFull);
  EXPECT_EQ(pair.barrier_scope, ConSanBarrierSite::Scope::Cluster);
  EXPECT_EQ(pair.memory_role, ConSanSyncMemoryRole::AcquireRelease);
  EXPECT_EQ(pair.confidence, ConSanSemanticConfidence::Conservative);
  EXPECT_EQ(pair.member_event_identities.size(), 2u);
  EXPECT_NE(pair.confidence_reason.find("conditional CFG triangle"), std::string::npos);
  ASSERT_EQ(pair.execution_owners.size(), 1u);
}

TEST(ConSan, SyncSequencesRejectClusterTriangleWithNontrivialSignalArm) {
  const auto bypass_signal = build_s_cbranch_scc1(/*offset_dwords=*/2, ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(bypass_signal);
  const std::array<uint32_t, 7> text_words = {
      *bypass_signal,
      0xBF800000u, // s_nop 0 makes the signal arm more than the exact arrive lowering.
      0xBE804EC3u, // s_barrier_signal -3
      0x3600009Fu, // v_and_b32_e32 v0, 31, v0
      0xBF048475u, // s_cmp_lt_i32 ttmp9, 4
      0xBF94FFFDu, // s_barrier_wait -3
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250),
  };
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult result = try_patch_consan(
      make_gfx1250_code_object(text_words, "nontrivial_cluster_triangle"), options);

  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);
  ASSERT_EQ(result.sync_sequences.size(), 2u);
  EXPECT_EQ(result.sync_sequences[0].operation, ConSanSyncOperation::BarrierSignal);
  EXPECT_EQ(result.sync_sequences[1].operation, ConSanSyncOperation::BarrierWait);
  EXPECT_EQ(result.sync_sequences[0].confidence, ConSanSemanticConfidence::Ambiguous);
  EXPECT_EQ(result.sync_sequences[1].confidence, ConSanSemanticConfidence::Ambiguous);
}

TEST(ConSan, SyncSequencesRejectDynamicM0BarrierSignal) {
  const std::array<uint32_t, 3> text_words = {
      0xBE804E7Du, // s_barrier_signal m0
      0xBF94FFFFu, // s_barrier_wait -1
      0xBFB00000u, // s_endpgm
  };
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult result = try_patch_consan(make_rdna4_lds_code_object(text_words), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.sync_events.size(), 2u);
  ASSERT_EQ(result.sync_sequences.size(), 2u);
  EXPECT_FALSE(result.sync_events[0].barrier_id);
  EXPECT_EQ(result.sync_events[0].barrier_operand_source,
            ConSanBarrierSite::OperandSource::DynamicM0);
  EXPECT_EQ(result.sync_events[0].barrier_scope, ConSanBarrierSite::Scope::Unknown);
  EXPECT_EQ(result.sync_sequences[0].operation, ConSanSyncOperation::BarrierSignal);
  EXPECT_EQ(result.sync_sequences[0].confidence, ConSanSemanticConfidence::Unsupported);
  EXPECT_EQ(result.sync_sequences[1].operation, ConSanSyncOperation::BarrierWait);
  EXPECT_EQ(result.sync_sequences[1].confidence, ConSanSemanticConfidence::Ambiguous);
  ASSERT_FALSE(result.fault_sites.empty());
  EXPECT_NE(result.fault_sites[0].decoded_operands.find("operand_source=dynamic-m0"),
            std::string::npos);
}

TEST(ConSan, SyncInventoryClassifiesGfx1250BarrierLifecycleWithoutOrderingClaims) {
  const std::array<uint32_t, 8> text_words = {
      0xBE80517Du, // s_barrier_init m0
      0xBE80527Du, // s_barrier_join m0
      0xBE804EC1u, // s_barrier_signal -1
      0xBF94FFFFu, // s_barrier_wait -1
      0xBF950000u, // s_barrier_leave
      0xBE80577Du, // s_wakeup_barrier m0
      0xBE84507Du, // s_get_barrier_state s4, m0
      0xBFB00000u, // s_endpgm
  };
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult result = try_patch_consan(make_gfx1250_code_object(text_words), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.kernels.size(), 1u);
  const ConSanKernelInfo &kernel = result.kernels.front();
  ASSERT_EQ(kernel.barrier_sites.size(), 7u);
  EXPECT_EQ(kernel.barrier_sites[0].operation, ConSanBarrierSite::Operation::Init);
  EXPECT_EQ(kernel.barrier_sites[1].operation, ConSanBarrierSite::Operation::Join);
  EXPECT_EQ(kernel.barrier_sites[2].operation, ConSanBarrierSite::Operation::Signal);
  EXPECT_EQ(kernel.barrier_sites[3].operation, ConSanBarrierSite::Operation::Wait);
  EXPECT_EQ(kernel.barrier_sites[4].operation, ConSanBarrierSite::Operation::Leave);
  EXPECT_EQ(kernel.barrier_sites[5].operation, ConSanBarrierSite::Operation::Wakeup);
  EXPECT_EQ(kernel.barrier_sites[6].operation, ConSanBarrierSite::Operation::StateQuery);
  for (const size_t index : {0u, 1u, 5u, 6u}) {
    EXPECT_EQ(kernel.barrier_sites[index].operand_source,
              ConSanBarrierSite::OperandSource::DynamicM0);
    EXPECT_FALSE(kernel.barrier_sites[index].barrier_id);
  }
  EXPECT_EQ(kernel.barrier_sites[2].operand_source, ConSanBarrierSite::OperandSource::Immediate);
  ASSERT_TRUE(kernel.barrier_sites[2].barrier_id);
  EXPECT_EQ(*kernel.barrier_sites[2].barrier_id, -1);
  EXPECT_EQ(kernel.barrier_sites[4].operand_source, ConSanBarrierSite::OperandSource::Unknown);

  ASSERT_EQ(result.sync_events.size(), 7u);
  const std::array<ConSanSyncOperation, 7> expected_operations = {
      ConSanSyncOperation::BarrierInit,       ConSanSyncOperation::BarrierJoin,
      ConSanSyncOperation::BarrierSignal,     ConSanSyncOperation::BarrierWait,
      ConSanSyncOperation::BarrierLeave,      ConSanSyncOperation::BarrierWakeup,
      ConSanSyncOperation::BarrierStateQuery,
  };
  for (size_t i = 0; i < expected_operations.size(); ++i)
    EXPECT_EQ(result.sync_events[i].operation, expected_operations[i]);

  for (const size_t index : {0u, 1u, 4u, 5u, 6u}) {
    const ConSanSyncEvent &event = result.sync_events[index];
    EXPECT_EQ(event.memory_role, ConSanSyncMemoryRole::Unknown);
    EXPECT_EQ(event.memory_role_confidence, ConSanSemanticConfidence::Unsupported);
    EXPECT_EQ(event.confidence, ConSanSemanticConfidence::Unsupported);
    EXPECT_NE(event.confidence_reason.find("participant semantics are unavailable"),
              std::string::npos);
  }
  EXPECT_EQ(result.sync_events[2].memory_role, ConSanSyncMemoryRole::Release);
  EXPECT_EQ(result.sync_events[3].memory_role, ConSanSyncMemoryRole::Acquire);

  ASSERT_EQ(result.sync_sequences.size(), 6u);
  EXPECT_EQ(result.sync_sequences[0].operation, ConSanSyncOperation::BarrierInit);
  EXPECT_EQ(result.sync_sequences[1].operation, ConSanSyncOperation::BarrierJoin);
  EXPECT_EQ(result.sync_sequences[2].operation, ConSanSyncOperation::BarrierFull);
  EXPECT_EQ(result.sync_sequences[3].operation, ConSanSyncOperation::BarrierLeave);
  EXPECT_EQ(result.sync_sequences[4].operation, ConSanSyncOperation::BarrierWakeup);
  EXPECT_EQ(result.sync_sequences[5].operation, ConSanSyncOperation::BarrierStateQuery);
  EXPECT_EQ(result.sync_sequences[2].memory_role, ConSanSyncMemoryRole::AcquireRelease);
}

TEST(ConSan, SyncInventoryPreservesGfx1250BarrierOperandEncodingForms) {
  constexpr uint64_t kLiteral64 = 0x9ABCDEF012345678ull;
  const std::array<uint32_t, 9> text_words = {
      0xBE805181u, // s_barrier_init 1
      0xBE80517Du, // s_barrier_init m0
      0xBE8051FFu,
      0xFFFFFFFDu, // s_barrier_init literal32(-3)
      0xBE8052FEu,
      static_cast<uint32_t>(kLiteral64),
      static_cast<uint32_t>(kLiteral64 >> 32u), // s_barrier_join literal64
      0xBF951234u,                              // s_barrier_leave raw simm16 0x1234
      0xBFB00000u,                              // s_endpgm
  };
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult result = try_patch_consan(make_gfx1250_code_object(text_words), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.kernels.size(), 1u);
  const auto &sites = result.kernels.front().barrier_sites;
  ASSERT_EQ(sites.size(), 5u);

  EXPECT_EQ(sites[0].size, 4u);
  EXPECT_EQ(sites[0].operand_source, ConSanBarrierSite::OperandSource::Immediate);
  EXPECT_EQ(sites[0].raw_operand_selector, 129u);
  EXPECT_EQ(sites[0].barrier_id, 1);

  EXPECT_EQ(sites[1].size, 4u);
  EXPECT_EQ(sites[1].operand_source, ConSanBarrierSite::OperandSource::DynamicM0);
  EXPECT_EQ(sites[1].raw_operand_selector, 125u);
  EXPECT_FALSE(sites[1].barrier_id);

  EXPECT_EQ(sites[2].size, 8u);
  EXPECT_EQ(sites[2].operand_source, ConSanBarrierSite::OperandSource::Literal32);
  EXPECT_EQ(sites[2].raw_operand_selector, 255u);
  EXPECT_EQ(sites[2].literal_width_bits, 32u);
  EXPECT_EQ(sites[2].literal_value, 0xFFFFFFFDu);
  EXPECT_EQ(sites[2].barrier_id, -3);
  EXPECT_EQ(sites[2].scope, ConSanBarrierSite::Scope::Cluster);

  EXPECT_EQ(sites[3].size, 12u);
  EXPECT_EQ(sites[3].operand_source, ConSanBarrierSite::OperandSource::Literal64);
  EXPECT_EQ(sites[3].raw_operand_selector, 254u);
  EXPECT_EQ(sites[3].literal_width_bits, 64u);
  EXPECT_EQ(sites[3].literal_value, kLiteral64);
  EXPECT_FALSE(sites[3].barrier_id);
  EXPECT_EQ(sites[3].scope, ConSanBarrierSite::Scope::Unknown);

  EXPECT_EQ(sites[4].size, 4u);
  EXPECT_EQ(sites[4].operation, ConSanBarrierSite::Operation::Leave);
  EXPECT_EQ(sites[4].raw_simm16, 0x1234u);
  EXPECT_FALSE(sites[4].barrier_id);

  ASSERT_EQ(result.sync_events.size(), sites.size());
  EXPECT_EQ(result.sync_events[2].barrier_operand_source,
            ConSanBarrierSite::OperandSource::Literal32);
  EXPECT_EQ(result.sync_events[2].barrier_raw_operand_selector, 255u);
  EXPECT_EQ(result.sync_events[2].barrier_literal_width_bits, 32u);
  EXPECT_EQ(result.sync_events[2].barrier_literal_value, 0xFFFFFFFDu);
  EXPECT_EQ(result.sync_events[3].barrier_operand_source,
            ConSanBarrierSite::OperandSource::Literal64);
  EXPECT_EQ(result.sync_events[3].barrier_literal_value, kLiteral64);
  EXPECT_EQ(result.sync_events[4].barrier_raw_simm16, 0x1234u);
  for (const ConSanSyncEvent &event : result.sync_events) {
    EXPECT_EQ(event.confidence, ConSanSemanticConfidence::Unsupported);
    EXPECT_FALSE(event.participant_count);
    EXPECT_FALSE(event.participant_mask);
  }
}

TEST(ConSan, SyncInventoryAdmitsStaticBarrierLifecycleGroupViaJoinAssociation) {
  const std::array<uint32_t, 6> text_words = {
      0xBE805181u, // s_barrier_init 1
      0xBE805281u, // s_barrier_join 1
      0xBE804E81u, // s_barrier_signal 1
      0xBF940001u, // s_barrier_wait 1
      0xBF950000u, // s_barrier_leave (fixed-zero encoding)
      0xBFB00000u, // s_endpgm
  };
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.fault_dry_run = true;
  const ConSanResult result = try_patch_consan(make_gfx1250_code_object(text_words), options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.barrier_lifecycle_groups.size(), 1u);
  const ConSanBarrierLifecycleGroup &group = result.barrier_lifecycle_groups.front();
  EXPECT_TRUE(group.admissible);
  EXPECT_EQ(group.confidence, ConSanSemanticConfidence::Conservative);
  EXPECT_EQ(group.barrier_id, 1);
  EXPECT_EQ(group.barrier_scope, ConSanBarrierSite::Scope::Workgroup);
  ASSERT_TRUE(group.basic_block_index);
  EXPECT_EQ(group.begin_text_offset, 0u);
  EXPECT_EQ(group.end_text_offset, 20u);
  EXPECT_EQ(group.member_event_identities.size(), 5u);
  EXPECT_TRUE(group.rejection_reason.empty());
}

TEST(ConSan, SyncInventoryRejectsDynamicMismatchedAndCrossBlockLifecycles) {
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.fault_dry_run = true;

  const std::array<uint32_t, 6> dynamic_words = {
      0xBE80517Du, // s_barrier_init m0
      0xBE805281u, // s_barrier_join 1
      0xBE804E81u, // s_barrier_signal 1
      0xBF940001u, // s_barrier_wait 1
      0xBF950000u, // s_barrier_leave
      0xBFB00000u,
  };
  const ConSanResult dynamic = try_patch_consan(make_gfx1250_code_object(dynamic_words), options);
  ASSERT_EQ(dynamic.barrier_lifecycle_groups.size(), 1u);
  EXPECT_NE(dynamic.barrier_lifecycle_groups[0].rejection_reason.find("no proven static ID"),
            std::string::npos);

  const std::array<uint32_t, 6> mismatched_words = {
      0xBE805181u, // s_barrier_init 1
      0xBE805282u, // s_barrier_join 2
      0xBE804E81u, // s_barrier_signal 1
      0xBF940001u, // s_barrier_wait 1
      0xBF950000u, // s_barrier_leave
      0xBFB00000u,
  };
  const ConSanResult mismatched =
      try_patch_consan(make_gfx1250_code_object(mismatched_words), options);
  ASSERT_EQ(mismatched.barrier_lifecycle_groups.size(), 1u);
  EXPECT_NE(
      mismatched.barrier_lifecycle_groups[0].rejection_reason.find("do not have one matching"),
      std::string::npos);

  const std::array<uint32_t, 7> cross_block_words = {
      0xBE805181u, // s_barrier_init 1
      build_s_branch(0, ROCJITSU_CODE_ARCH_GFX1250),
      0xBE805281u, // s_barrier_join 1
      0xBE804E81u, // s_barrier_signal 1
      0xBF940001u, // s_barrier_wait 1
      0xBF950001u, // s_barrier_leave
      0xBFB00000u,
  };
  const ConSanResult cross_block =
      try_patch_consan(make_gfx1250_code_object(cross_block_words), options);
  ASSERT_EQ(cross_block.barrier_lifecycle_groups.size(), 1u);
  EXPECT_NE(cross_block.barrier_lifecycle_groups[0].rejection_reason.find("crosses a block"),
            std::string::npos);
}

TEST(ConSan, SyncInventoryRejectsLifecycleWithoutJoinOrFixedZeroLeave) {
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.fault_dry_run = true;

  const std::array<uint32_t, 5> no_join_words = {
      0xBE805181u, // s_barrier_init 1
      0xBE804E81u, // s_barrier_signal 1
      0xBF940001u, // s_barrier_wait 1
      0xBF950000u, // s_barrier_leave
      0xBFB00000u,
  };
  const ConSanResult no_join = try_patch_consan(make_gfx1250_code_object(no_join_words), options);
  ASSERT_EQ(no_join.barrier_lifecycle_groups.size(), 1u);
  EXPECT_FALSE(no_join.barrier_lifecycle_groups[0].admissible);
  EXPECT_NE(no_join.barrier_lifecycle_groups[0].rejection_reason.find("no preceding matching"),
            std::string::npos);

  const std::array<uint32_t, 6> nonzero_leave_words = {
      0xBE805181u, // s_barrier_init 1
      0xBE805281u, // s_barrier_join 1
      0xBE804E81u, // s_barrier_signal 1
      0xBF940001u, // s_barrier_wait 1
      0xBF950001u, // invalid non-zero fixed simm16
      0xBFB00000u,
  };
  const ConSanResult nonzero_leave =
      try_patch_consan(make_gfx1250_code_object(nonzero_leave_words), options);
  ASSERT_EQ(nonzero_leave.barrier_lifecycle_groups.size(), 1u);
  EXPECT_FALSE(nonzero_leave.barrier_lifecycle_groups[0].admissible);
  EXPECT_NE(nonzero_leave.barrier_lifecycle_groups[0].rejection_reason.find("fixed-zero"),
            std::string::npos);
}

TEST(ConSan, FinalValidationExhaustivelyProvesExactBarrierLifecycleRewrite) {
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
  const auto barrier = std::ranges::find(inventory.sync_sequences, ConSanSyncOperation::BarrierFull,
                                         &ConSanSyncSequence::operation);
  ASSERT_NE(barrier, inventory.sync_sequences.end());
  ConSanOptions options = inventory_options;
  options.fault_dry_run = false;
  options.fault_mutate_barrier_id_scope = true;
  options.fault_barrier_sequence_identity = barrier->identity;
  options.fault_barrier_target_id = 16;
  options.fault_require_exactly_one = true;
  const ConSanResult valid = try_patch_consan(bytes, options);
  ASSERT_EQ(valid.outcome, ConSanTransformOutcome::ModifiedValid)
      << (valid.errors.empty() ? "" : valid.errors.front());
  ASSERT_EQ(valid.patches.size(), 5u);
  ASSERT_EQ(valid.text_sections.size(), 1u);
  AmdGpuCodeObject original(bytes.data(), bytes.size());
  ASSERT_EQ(original.text_sections().size(), 1u);
  const auto original_text = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(original.text_sections().front()->data()),
      original.text_sections().front()->size());
  const uint64_t text_file_offset = valid.text_sections.front().file_offset;

  const auto expect_invalid = [&](const ConSanResult &corrupted) {
    EXPECT_FALSE(validate_consan_modified_elf(bytes, corrupted).empty());
  };
  for (size_t i = 0; i < valid.patches.size(); ++i) {
    const ConSanPatchInfo &patch = valid.patches[i];
    ASSERT_EQ(patch.kind, ConSanPatchKind::InlineBarrierIdScopeRewrite);

    // Every init/join/signal/wait member is required and must name the same
    // decoded target ID and scope.
    ConSanResult wrong_target = valid;
    const uint64_t operand_offset =
        patch.original_size == 2u * sizeof(uint32_t) ? sizeof(uint32_t) : 0u;
    uint32_t word = 0;
    std::memcpy(&word,
                wrong_target.elf_bytes.data() + text_file_offset + patch.anchor_offset +
                    operand_offset,
                sizeof(word));
    if (patch.original_size == 2u * sizeof(uint32_t)) {
      word = 15u;
    } else if (i + 1u == valid.patches.size()) {
      word = (word & 0xffff0000u) | 15u;
    } else {
      word = (word & ~0xffu) | (128u + 15u);
    }
    std::memcpy(wrong_target.elf_bytes.data() + text_file_offset + patch.anchor_offset +
                    operand_offset,
                &word, sizeof(word));
    expect_invalid(wrong_target);

    // Restoring bytes while omitting the corresponding record defeats generic
    // byte accounting alone; pristine group derivation must still reject it.
    ConSanResult omitted = valid;
    std::memcpy(omitted.elf_bytes.data() + text_file_offset + patch.anchor_offset,
                original_text.data() + patch.anchor_offset, patch.original_size);
    omitted.patches.erase(omitted.patches.begin() + static_cast<ptrdiff_t>(i));
    expect_invalid(omitted);

    // No opcode or non-operand bit may change, and the decoded instruction
    // must retain its original size and encoding class.
    ConSanResult wrong_shape = valid;
    uint32_t first_word = 0;
    std::memcpy(&first_word, wrong_shape.elf_bytes.data() + text_file_offset + patch.anchor_offset,
                sizeof(first_word));
    first_word ^= 1u << 24u;
    std::memcpy(wrong_shape.elf_bytes.data() + text_file_offset + patch.anchor_offset, &first_word,
                sizeof(first_word));
    expect_invalid(wrong_shape);
  }

  ConSanResult wrong_encoding_class = valid;
  uint32_t literal_init_word = 0;
  std::memcpy(&literal_init_word, wrong_encoding_class.elf_bytes.data() + text_file_offset,
              sizeof(literal_init_word));
  literal_init_word = (literal_init_word & ~0xffu) | (128u + 16u);
  std::memcpy(wrong_encoding_class.elf_bytes.data() + text_file_offset, &literal_init_word,
              sizeof(literal_init_word));
  expect_invalid(wrong_encoding_class);

  ConSanResult wrong_member_size = valid;
  wrong_member_size.patches.front().original_size = sizeof(uint32_t);
  expect_invalid(wrong_member_size);

  // A consistent cluster target would pass simple equality checks, but it is
  // invalid metadata for a named-barrier lifecycle.
  ConSanResult wrong_scope = valid;
  for (size_t i = 0; i < wrong_scope.patches.size(); ++i) {
    const ConSanPatchInfo &patch = wrong_scope.patches[i];
    const uint64_t operand_offset =
        patch.original_size == 2u * sizeof(uint32_t) ? sizeof(uint32_t) : 0u;
    uint32_t word = 0;
    std::memcpy(&word,
                wrong_scope.elf_bytes.data() + text_file_offset + patch.anchor_offset +
                    operand_offset,
                sizeof(word));
    if (patch.original_size == 2u * sizeof(uint32_t)) {
      word = 0xFFFFFFFDu;
    } else if (i + 1u == wrong_scope.patches.size()) {
      word = (word & 0xffff0000u) | 0xFFFDu;
    } else {
      word = (word & ~0xffu) | 195u;
    }
    std::memcpy(wrong_scope.elf_bytes.data() + text_file_offset + patch.anchor_offset +
                    operand_offset,
                &word, sizeof(word));
  }
  const std::vector<std::string> scope_errors = validate_consan_modified_elf(bytes, wrong_scope);
  EXPECT_TRUE(std::ranges::any_of(scope_errors, [](const std::string &error) {
    return error.find("invalid lifecycle target ID/scope metadata") != std::string::npos;
  }));

  // Account the leave range under a fake non-mutation record so that semantic
  // validation, not merely unaccounted-byte detection, proves it remains the
  // pristine fixed-zero instruction.
  ConSanResult changed_leave = valid;
  const uint64_t leave_offset = 8u * sizeof(uint32_t);
  uint32_t nonzero_leave = 0xBF950001u;
  std::memcpy(changed_leave.elf_bytes.data() + text_file_offset + leave_offset, &nonzero_leave,
              sizeof(nonzero_leave));
  ConSanPatchInfo fake_leave_accounting;
  fake_leave_accounting.phase = ConSanPatchPhase::Instrumentation;
  fake_leave_accounting.kind = ConSanPatchKind::InlineNopRewrite;
  fake_leave_accounting.anchor_offset = leave_offset;
  fake_leave_accounting.trampoline_offset = leave_offset;
  fake_leave_accounting.original_size = sizeof(uint32_t);
  changed_leave.patches.push_back(fake_leave_accounting);
  const std::vector<std::string> leave_errors = validate_consan_modified_elf(bytes, changed_leave);
  EXPECT_TRUE(std::ranges::any_of(leave_errors, [](const std::string &error) {
    return error.find("fixed-zero lifecycle leave") != std::string::npos;
  }));

  ConSanResult unaccounted = valid;
  uint32_t changed_endpgm = build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250);
  std::memcpy(unaccounted.elf_bytes.data() + text_file_offset + 9u * sizeof(uint32_t),
              &changed_endpgm, sizeof(changed_endpgm));
  const std::vector<std::string> accounting_errors =
      validate_consan_modified_elf(bytes, unaccounted);
  EXPECT_TRUE(std::ranges::any_of(accounting_errors, [](const std::string &error) {
    return error.find("unaccounted executable byte change") != std::string::npos;
  }));
}

TEST(ConSan, SyncSequencesInventoryUnmatchedBarrierComponentsWithoutPairing) {
  const std::array<uint32_t, 2> signal_words = {
      0xBE804EC1u, // s_barrier_signal -1
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 2> wait_words = {
      0xBF94FFFFu, // s_barrier_wait -1
      0xBFB00000u, // s_endpgm
  };
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult signal = try_patch_consan(make_rdna4_lds_code_object(signal_words), options);
  const ConSanResult wait = try_patch_consan(make_rdna4_lds_code_object(wait_words), options);

  ASSERT_EQ(signal.sync_sequences.size(), 1u);
  EXPECT_EQ(signal.sync_sequences.front().operation, ConSanSyncOperation::BarrierSignal);
  EXPECT_EQ(signal.sync_sequences.front().confidence, ConSanSemanticConfidence::Ambiguous);
  ASSERT_EQ(signal.sync_sequences.front().member_event_identities.size(), 1u);
  ASSERT_EQ(wait.sync_sequences.size(), 1u);
  EXPECT_EQ(wait.sync_sequences.front().operation, ConSanSyncOperation::BarrierWait);
  EXPECT_EQ(wait.sync_sequences.front().confidence, ConSanSemanticConfidence::Ambiguous);
  ASSERT_EQ(wait.sync_sequences.front().member_event_identities.size(), 1u);
}

TEST(ConSan, CommunicationAddressRecipeSupportsAtomicReleaseAddress) {
  ConSanOptions options = moi_options();
  options.fault_dry_run = true;
  const ConSanResult result =
      try_patch_consan(make_rdna4_ordered_global_atomic_release_acquire_code_object(), options);

  const auto recipe = std::ranges::find_if(
      result.communication_address_recipes, [](const ConSanCommunicationAddressRecipe &candidate) {
        return candidate.kind == ConSanCommunicationAddressKind::Atomic && candidate.supported();
      });
  ASSERT_NE(recipe, result.communication_address_recipes.end());
  EXPECT_EQ(recipe->address_source, ConSanSyncAddressSource::GlobalScalarVector);
  EXPECT_EQ(recipe->address_vgpr, 2u);
  EXPECT_EQ(recipe->address_vgpr_count, 1u);
  ASSERT_TRUE(recipe->address_sgpr);
  EXPECT_EQ(*recipe->address_sgpr, 4u);
  EXPECT_EQ(recipe->address_sgpr_count, 2u);
  EXPECT_EQ(recipe->static_byte_offset, 0);
  ASSERT_TRUE(recipe->scratch_vgpr);
  EXPECT_EQ(recipe->scratch_vgpr_count, 2u);
}

TEST(ConSan, FinalValidationRederivesStructuredExecDiamondProof) {
  const auto save_exec =
      build_s_and_saveexec_b64(/*sdst=*/30, /*ssrc0=*/106, ROCJITSU_CODE_ARCH_RDNA4);
  const auto restore_exec = build_s_mov_b64(/*sdst=*/126, /*ssrc0=*/30, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(save_exec);
  ASSERT_TRUE(restore_exec);
  const std::array<uint32_t, 12> text_words = {
      *save_exec,
      pack_sopp(/*s_cbranch_execz=*/37, /*simm16=*/3),
      build_v_mov_b32_e32(/*vdst=*/1, vector_source_vgpr(1), ROCJITSU_CODE_ARCH_RDNA4),
      0xD8340000u,
      0x00000000u, // optional-arm ds_store_b32 after a leading instruction
      *restore_exec,
      0xBE804EC1u,
      0xBF94FFFFu,
      0xBFB00000u,
      0xBF800000u,
      0xBF800000u,
      0xBF800000u,
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "validate_structured_divergent_move");
  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  const ConSanResult inventory = try_patch_consan(bytes, inventory_options);
  const auto destination = std::ranges::find(inventory.barrier_move_destinations, 12u,
                                             &ConSanBarrierMoveDestination::text_offset);
  ASSERT_NE(destination, inventory.barrier_move_destinations.end());

  ConSanOptions options = inventory_options;
  options.fault_move_barrier = true;
  options.fault_allow_destructive_divergent_barrier_move = true;
  options.fault_site_identity = inventory.fault_sites.front().identity;
  options.fault_barrier_move_direction = ConSanBarrierMoveDirection::Earlier;
  options.fault_barrier_destination_identity = destination->identity;
  const ConSanResult valid = try_patch_consan(bytes, options);
  ASSERT_EQ(valid.outcome, ConSanTransformOutcome::ModifiedValid);

  ConSanResult stale_inventory = valid;
  ASSERT_FALSE(stale_inventory.kernels.empty());
  stale_inventory.kernels.front().entry_text_offset += sizeof(uint32_t);
  EXPECT_TRUE(validate_consan_modified_elf(bytes, stale_inventory).empty());

  ConSanResult corrupted = valid;
  ASSERT_TRUE(corrupted.patches.back().structured_guard_offset);
  ++*corrupted.patches.back().structured_guard_offset;
  const auto errors = validate_consan_modified_elf(bytes, corrupted);
  EXPECT_TRUE(std::ranges::any_of(errors, [](const std::string &error) {
    return error.find("could not rederive its structured CFG contract") != std::string::npos;
  }));

  corrupted = valid;
  ASSERT_TRUE(corrupted.patches.back().structured_destination_block_index);
  --*corrupted.patches.back().structured_destination_block_index;
  const auto block_errors = validate_consan_modified_elf(bytes, corrupted);
  EXPECT_TRUE(std::ranges::any_of(block_errors, [](const std::string &error) {
    return error.find("block metadata that does not contain") != std::string::npos;
  }));

  corrupted = valid;
  ConSanPatchInfo &target = corrupted.patches.back();
  target.barrier_move_cfg_contract = ConSanBarrierMoveCfgContract::SameBlock;
  target.structured_guard_block_index.reset();
  target.structured_destination_block_index.reset();
  target.structured_source_block_index.reset();
  target.structured_guard_offset.reset();
  target.structured_destination_offset.reset();
  target.structured_source_offset.reset();
  const auto missing_errors = validate_consan_modified_elf(bytes, corrupted);
  EXPECT_TRUE(std::ranges::any_of(missing_errors, [](const std::string &error) {
    return error.find("cross-block move without a structured CFG contract and metadata") !=
           std::string::npos;
  }));
}

TEST(ConSan, FinalValidationRejectsMissingMovedBarrierTarget) {
  const std::array<uint32_t, 7> text_words = {
      0xBE804EC1u, // s_barrier_signal -1
      0xBF94FFFFu, // s_barrier_wait -1
      0xD8340000u,
      0x00000000u, // ds_store_b32
      build_s_nop(42, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_nop(43, ROCJITSU_CODE_ARCH_RDNA4),
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.fault_move_barrier = true;
  const ConSanResult valid = try_patch_consan(bytes, options);
  ASSERT_EQ(valid.outcome, ConSanTransformOutcome::ModifiedValid);
  ASSERT_EQ(valid.patches.size(), 4u);

  ConSanResult corrupted = valid;
  const size_t target_file_offset =
      valid.text_sections.front().file_offset + valid.patches[2].anchor_offset;
  const uint32_t marker = build_s_nop(42, ROCJITSU_CODE_ARCH_RDNA4);
  std::memcpy(corrupted.elf_bytes.data() + target_file_offset, &marker, sizeof(marker));
  const std::vector<std::string> errors = validate_consan_modified_elf(bytes, corrupted);

  EXPECT_TRUE(std::ranges::any_of(errors, [](const std::string &error) {
    return error.find("mutation proof did not place a barrier at the relocation target") !=
           std::string::npos;
  }));
}

TEST(ConSan, MarksSupportedLdsKernelAsPreflightCandidate) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  ASSERT_EQ(result.kernels.size(), 1u);

  const ConSanKernelInfo &kernel = result.kernels.front();
  EXPECT_TRUE(kernel.decoded);
  EXPECT_EQ(kernel.code_size, 24u);
  EXPECT_EQ(kernel.stats.instruction_count, 4u);
  EXPECT_EQ(kernel.stats.lds_read_count, 1u);
  EXPECT_EQ(kernel.stats.lds_write_count, 1u);
  EXPECT_EQ(kernel.stats.lds_atomic_count, 0u);
  EXPECT_EQ(kernel.stats.fence_like_count, 0u);
  ASSERT_EQ(kernel.lds_sites.size(), 2u);
  EXPECT_EQ(kernel.lds_sites[0].kind, ConSanLdsAccessKind::Write);
  EXPECT_TRUE(kernel.lds_sites[0].supported_mvp);
  EXPECT_EQ(kernel.lds_sites[0].text_offset, 0u);
  EXPECT_EQ(kernel.lds_sites[0].width_bits, 32u);
  EXPECT_EQ(kernel.lds_sites[1].kind, ConSanLdsAccessKind::Read);
  EXPECT_TRUE(kernel.lds_sites[1].supported_mvp);
  EXPECT_EQ(kernel.lds_sites[1].text_offset, 8u);
  EXPECT_EQ(kernel.lds_sites[1].width_bits, 32u);
  EXPECT_EQ(kernel.preflight_action, ConSanPreflightAction::Candidate);
  EXPECT_GE(kernel.preflight_reasons.size(), 2u);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
}

TEST(ConSan, FailClosedAdmitsOrdinaryLdsAlongsideExcludedAtomic) {
  const std::vector<uint8_t> bytes = make_rdna4_unsupported_lds_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.fail_closed = true;

  const auto result = try_patch_consan(bytes, options);

  EXPECT_TRUE(result.errors.empty());
  EXPECT_NE(result.outcome, ConSanTransformOutcome::Unsupported);
  ASSERT_EQ(result.kernels.size(), 1u);
  const ConSanKernelInfo &kernel = result.kernels.front();
  EXPECT_EQ(kernel.preflight_action, ConSanPreflightAction::Candidate);
  EXPECT_TRUE(std::ranges::any_of(kernel.preflight_reasons, [](const std::string &reason) {
    return reason == "atomic LDS accesses excluded: 1";
  }));
}

} // namespace
} // namespace rocjitsu
