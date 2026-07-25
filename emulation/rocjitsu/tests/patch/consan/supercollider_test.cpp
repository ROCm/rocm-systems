// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"

#include "rocjitsu/code/patch/instrumentation_builder.h"
#include "rocjitsu/code/patch/spill_manager.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/opcodes.h"

namespace rocjitsu {
namespace {

TEST(ConSan, FlatTrapProofRewritesLikelyGroupLocalFunctionSite) {
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
  options.probe_flat_trap = true;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineFlatTrapRewrite);
  EXPECT_EQ(result.patches.front().anchor_offset, 24u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 36u);
  EXPECT_EQ(result.patches.front().original_size, 12u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());
  EXPECT_NE(result.elf_bytes, bytes);

  const auto rewritten_words = patched_words_at_file_offset<3>(result, 0x118);
  EXPECT_EQ(rewritten_words[0], 0xBF900000u); // s_trap 0
  EXPECT_EQ(rewritten_words[1], 0xBF800000u); // s_nop 0
  EXPECT_EQ(rewritten_words[2], 0xBF800000u); // s_nop 0
}

TEST(ConSan, FlatCheckTrapProofUsesReachableAppendedCave) {
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
  options.probe_flat_check_trap = true;
  options.delay_nops = 1;
  options.scratch_vgpr = 5;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::LocalCaveFlatLoadCheckTrap);
  EXPECT_EQ(result.patches.front().original_size, 12u);
  EXPECT_GT(result.patches.front().trampoline_offset, result.patches.front().anchor_offset);
  EXPECT_GT(result.patches.front().trampoline_size, 0u);
  EXPECT_GT(result.elf_bytes.size(), bytes.size());
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSan, FlatCheckTrapProofUsesIndirectIslandForFarAppendedCave) {
  constexpr size_t kLargeTextWords = 33000u;
  std::vector<uint32_t> text_words = {
      0xBE8001EBu,                           // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u, 0x00000000u,              // v_mov_b32_e64 v0, s0
      0xD5810001u, 0x00000001u,              // v_mov_b32_e64 v1, s1
      0xEC05007Cu, 0x00000002u, 0x00000000u, // flat_load_b32 v2, v[0:1]
      0xBFB00000u,                           // s_endpgm
  };
  for (size_t i = 0; i < 7u; ++i)
    text_words.push_back(build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words.resize(kLargeTextWords - 1u, build_s_mov_b32(100, 100, ROCJITSU_CODE_ARCH_RDNA4));
  text_words.push_back(0xBFB00000u); // s_endpgm

  const uint64_t original_text_size = text_words.size() * sizeof(uint32_t);
  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  // Leave the seven NOPs outside executable kernel text as a proven island.
  mutate_elf_symbol(bytes, 1, [](Elf64_Sym &symbol) { symbol.st_size = 9u * sizeof(uint32_t); });
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_flat_check_trap = true;
  options.delay_nops = 1;
  options.scratch_vgpr = 5;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified);
  const auto island = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineScIndirectBranchIsland, &ConSanPatchInfo::kind);
  const auto body = std::ranges::find(result.patches, ConSanPatchKind::LocalCaveFlatLoadCheckTrap,
                                      &ConSanPatchInfo::kind);
  ASSERT_NE(island, result.patches.end());
  ASSERT_NE(body, result.patches.end());
  EXPECT_EQ(island->anchor_offset, 5u * sizeof(uint32_t));
  EXPECT_EQ(island->trampoline_offset, 9u * sizeof(uint32_t));
  EXPECT_EQ(island->trampoline_size, 7u * sizeof(uint32_t));
  EXPECT_EQ(body->trampoline_offset, original_text_size);
  EXPECT_GT(body->indirect_required_sgpr_count, 0u);
  ASSERT_EQ(body->owner_descriptor_file_offsets.size(), 1u);
  EXPECT_TRUE(body->indirect_return_offset.has_value());
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSan, FlatCheckTrapProofRelocatesPrefixForFarAppendedCave) {
  constexpr size_t kLargeTextWords = 33000u;
  std::vector<uint32_t> text_words = {
      0xBE8001EBu,                           // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u, 0x00000000u,              // v_mov_b32_e64 v0, s0
      0xD5810001u, 0x00000001u,              // v_mov_b32_e64 v1, s1
      0xEC05007Cu, 0x00000002u, 0x00000000u, // flat_load_b32 v2, v[0:1]
      0xBE860086u,                           // s_mov_b32 s6, s6
      0xBE870087u,                           // s_mov_b32 s7, s7
      0xBE880088u,                           // s_mov_b32 s8, s8
      0xBE890089u,                           // s_mov_b32 s9, s9
  };
  text_words.resize(kLargeTextWords - 1u, build_s_mov_b32(10, 10, ROCJITSU_CODE_ARCH_RDNA4));
  text_words.push_back(0xBFB00000u); // s_endpgm

  const uint64_t original_text_size = text_words.size() * sizeof(uint32_t);
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_flat_check_trap = true;
  options.delay_nops = 1;
  options.scratch_vgpr = 5;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineScIndirectBranchIsland,
                               &ConSanPatchInfo::kind),
            0u);
  const auto body = std::ranges::find(result.patches, ConSanPatchKind::LocalCaveFlatLoadCheckTrap,
                                      &ConSanPatchInfo::kind);
  ASSERT_NE(body, result.patches.end());
  EXPECT_EQ(body->anchor_offset, 5u * sizeof(uint32_t));
  EXPECT_EQ(body->original_size, 7u * sizeof(uint32_t));
  EXPECT_EQ(body->trampoline_offset, original_text_size);
  EXPECT_TRUE(body->indirect_return_offset.has_value());
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSan, FlatCheckTrapProofUsesReachableUncoveredNopCave) {
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
  const std::array<uint32_t, 14> tail_words = {
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words, tail_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_flat_check_trap = true;
  options.delay_nops = 1;
  options.scratch_vgpr = 5;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::LocalCaveFlatLoadCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 24u);
  EXPECT_GT(result.patches.front().trampoline_offset, 40u);
  EXPECT_EQ(result.patches.front().original_size, 12u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 5u);
  ASSERT_GT(result.elf_bytes.size(), bytes.size());
  EXPECT_NE(result.elf_bytes, bytes);

  std::array<uint32_t, 3> anchor_words{};
  std::memcpy(anchor_words.data(), result.elf_bytes.data() + 0x118,
              anchor_words.size() * sizeof(uint32_t));
  EXPECT_EQ(anchor_words[1], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(anchor_words[2], build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));

  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSan, FlatB128CheckTrapSharesOneMismatchActionInLocalCave) {
  const std::array<uint32_t, 1> kernel_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 9> function_words = {
      0xBE8001EBu,                           // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u, 0x00000000u,              // v_mov_b32_e64 v0, s0
      0xD5810001u, 0x00000001u,              // v_mov_b32_e64 v1, s1
      0xEC05C07Cu, 0x00000002u, 0x00000000u, // flat_load_b128 v[2:5], v[0:1]
      0xBFB00000u,                           // s_endpgm
  };
  std::array<uint32_t, 24> tail_words{};
  tail_words.fill(build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words, tail_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_flat_check_trap = true;
  options.delay_nops = 1;
  options.scratch_vgpr = 10;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::LocalCaveFlatLoadCheckTrap);
  ASSERT_EQ(result.patches.front().trampoline_size % sizeof(uint32_t), 0u);
  const std::vector<uint32_t> cave_words =
      patched_words_at_file_offset(result, 0x100 + result.patches.front().trampoline_offset,
                                   result.patches.front().trampoline_size);
  EXPECT_EQ(std::count(cave_words.begin(), cave_words.end(), 0xBF900000u), 1);
  for (int16_t distance : {7, 5, 3, 1}) {
    const auto branch = build_s_cbranch_vccnz(distance, ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_TRUE(branch);
    EXPECT_NE(std::find(cave_words.begin(), cave_words.end(), *branch), cave_words.end());
  }
}

TEST(ConSan, FlatCheckTrapProofDoesNotClobberLiveThroughVgpr) {
  const std::array<uint32_t, 1> kernel_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 10> function_words = {
      0xBE8001EBu,                           // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u, 0x00000000u,              // v_mov_b32_e64 v0, s0
      0xD5810001u, 0x00000001u,              // v_mov_b32_e64 v1, s1
      0xEC05007Cu, 0x00000002u, 0x00000000u, // flat_load_b32 v2, v[0:1]
      0x7E080303u,                           // v_mov_b32_e32 v4, v3
      0xBFB00000u,                           // s_endpgm
  };
  const std::array<uint32_t, 12> tail_words = {
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words, tail_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_flat_check_trap = true;
  options.delay_nops = 1;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 4u);
  EXPECT_NE(*result.patches.front().scratch_vgpr, 3u);
}

TEST(ConSan, FlatLoadCheckTrapProofRewritesPaddedLocalFunctionSite) {
  const std::array<uint32_t, 1> kernel_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 19> function_words = {
      0xBE8001EBu,                           // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u, 0x00000000u,              // v_mov_b32_e64 v0, s0
      0xD5810001u, 0x00000001u,              // v_mov_b32_e64 v1, s1
      0xEC05007Cu, 0x00000002u, 0x00000000u, // flat_load_b32 v2, v[0:1]
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_flat_check_trap = true;
  options.delay_nops = 1;
  options.scratch_vgpr = 5;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineFlatLoadCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 24u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 36u);
  EXPECT_EQ(result.patches.front().original_size, 52u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 5u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());
  EXPECT_NE(result.elf_bytes, bytes);

  ASSERT_GE(result.patches.front().required_sgpr_count, 2u);
  const uint16_t vcc_save_sgpr =
      static_cast<uint16_t>(result.patches.front().required_sgpr_count - 2u);
  const std::array<uint32_t, 13> expected_words = {
      *instrumentation::build_s_mov_b64(vcc_save_sgpr, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0xEC05007Cu,
      0x00000002u,
      0x00000000u, // original flat_load_b32 v2, v[0:1]
      0xBF800000u, // delay
      0xEC05007Cu,
      0x00000005u,
      0x00000000u, // duplicate flat_load_b32 v5, v[0:1]
      0xBFC60000u, // s_wait_dscnt 0
      0x7C9A0B02u, // v_cmp_ne_u32_e32 v2, v5
      0xBFA30001u, // s_cbranch_vccz +1
      0xBF900000u, // s_trap 0
      *instrumentation::build_s_mov_b64(kRdna4VccLo, vcc_save_sgpr, ROCJITSU_CODE_ARCH_RDNA4),
  };
  const auto rewritten_words = patched_words_at_file_offset<expected_words.size()>(result, 0x118);
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(ConSan, CombinedCheckTrapFallsBackToFlatWhenNoNativeLdsPatchApplies) {
  const std::array<uint32_t, 1> kernel_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 19> function_words = {
      0xBE8001EBu,                           // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u, 0x00000000u,              // v_mov_b32_e64 v0, s0
      0xD5810001u, 0x00000001u,              // v_mov_b32_e64 v1, s1
      0xEC05007Cu, 0x00000002u, 0x00000000u, // flat_load_b32 v2, v[0:1]
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.probe_flat_check_trap = true;
  options.delay_nops = 1;
  options.scratch_vgpr = 5;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineFlatLoadCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 24u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 36u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 5u);
}

TEST(ConSan, CombinedCheckTrapCanPatchNativeLdsAndFlatInSameCodeObject) {
  const std::array<uint32_t, 13> kernel_words = {
      0xD8340000u,
      0x00000102u, // ds_store_b32 v2, v1
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 19> function_words = {
      0xBE8001EBu,                           // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u, 0x00000000u,              // v_mov_b32_e64 v0, s0
      0xD5810001u, 0x00000001u,              // v_mov_b32_e64 v1, s1
      0xEC05007Cu, 0x00000002u, 0x00000000u, // flat_load_b32 v2, v[0:1]
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.probe_flat_check_trap = true;
  options.delay_nops = 1;
  options.scratch_vgpr = 3;
  options.max_patches = 2;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 2u);
  EXPECT_EQ(result.patches[0].kind, ConSanPatchKind::InlineLdsStoreCheckTrap);
  EXPECT_EQ(result.patches[0].anchor_offset, 0u);
  EXPECT_EQ(result.patches[0].trampoline_offset, 8u);
  EXPECT_EQ(result.patches[0].original_size, 44u);
  EXPECT_EQ(result.patches[0].trampoline_size, 0u);
  ASSERT_TRUE(result.patches[0].scratch_vgpr);
  EXPECT_EQ(*result.patches[0].scratch_vgpr, 3u);
  EXPECT_EQ(result.patches[1].kind, ConSanPatchKind::InlineFlatLoadCheckTrap);
  EXPECT_EQ(result.patches[1].anchor_offset, 72u);
  EXPECT_EQ(result.patches[1].trampoline_offset, 84u);
  EXPECT_EQ(result.patches[1].original_size, 52u);
  EXPECT_EQ(result.patches[1].trampoline_size, 0u);
  ASSERT_TRUE(result.patches[1].scratch_vgpr);
  EXPECT_EQ(*result.patches[1].scratch_vgpr, 3u);
}

TEST(ConSan, CombinedCheckTrapIgnoresMetadataOnlyIslandAnchor) {
  const std::array<uint32_t, 10> kernel_words = {
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xBFB00000u, // s_endpgm
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
  };
  constexpr size_t kLargeFunctionWords = 33000u;
  std::vector<uint32_t> function_words(kLargeFunctionWords,
                                       build_s_mov_b32(100, 100, ROCJITSU_CODE_ARCH_RDNA4));
  constexpr size_t kFlatWord = kLargeFunctionWords - 14u;
  function_words[kFlatWord - 5u] = 0xBE8001EBu; // s_mov_b64 s[0:1], src_shared_base
  function_words[kFlatWord - 4u] = 0xD5810000u;
  function_words[kFlatWord - 3u] = 0x00000000u; // v_mov_b32_e64 v0, s0
  function_words[kFlatWord - 2u] = 0xD5810001u;
  function_words[kFlatWord - 1u] = 0x00000001u; // v_mov_b32_e64 v1, s1
  function_words[kFlatWord] = 0xEC05007Cu;
  function_words[kFlatWord + 1u] = 0x00000002u;
  function_words[kFlatWord + 2u] = 0x00000000u; // flat_load_b32 v2, v[0:1]
  for (size_t i = kFlatWord + 3u; i + 1u < function_words.size(); ++i)
    function_words[i] = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
  function_words.back() = 0xBFB00000u; // s_endpgm

  std::vector<uint8_t> bytes = make_rdna4_code_object_with_local_function(
      kernel_words, function_words, {}, kRdna4Wave64AllVgprsGranulated,
      /*function_is_kernel=*/true);
  // Leave seven linker-padding NOPs outside the first executable kernel so
  // its far native LDS body has a proven indirect entry island.
  mutate_elf_symbol(bytes, 1, [](Elf64_Sym &symbol) { symbol.st_size = 3u * sizeof(uint32_t); });
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.probe_flat_check_trap = true;
  options.delay_nops = 1;
  options.scratch_vgpr = 5;
  options.max_patches = 2;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::LocalCaveLdsLoadCheckTrap,
                               &ConSanPatchInfo::kind),
            1u)
      << testing::PrintToString(result.warnings);
  const auto island = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineScIndirectBranchIsland, &ConSanPatchInfo::kind);
  ASSERT_NE(island, result.patches.end());
  EXPECT_EQ(island->original_size, 0u);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::InlineFlatLoadCheckTrap,
                               &ConSanPatchInfo::kind),
            1u);
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSan, FlatCheckTrapAllSupportedPolicyIgnoresNominalPatchLimit) {
  const std::array<uint32_t, 1> kernel_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 32> function_words = {
      0xBE8001EBu, // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u,
      0x00000000u, // v_mov_b32_e64 v0, s0
      0xD5810001u,
      0x00000001u, // v_mov_b32_e64 v1, s1
      0xEC05007Cu, 0x00000002u,
      0x00000000u, // flat_load_b32 v2, v[0:1]
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xEC05007Cu, 0x00000006u,
      0x00000000u, // flat_load_b32
                   // v6, v[0:1]
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_flat_check_trap = true;
  options.delay_nops = 1;
  options.scratch_vgpr = 5;
  options.max_patches = 1;
  options.max_patches_is_expert_limit = false;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 2u);
  EXPECT_EQ(result.patches[0].kind, ConSanPatchKind::InlineFlatLoadCheckTrap);
  EXPECT_EQ(result.patches[0].anchor_offset, 24u);
  EXPECT_EQ(result.patches[0].trampoline_offset, 36u);
  EXPECT_EQ(result.patches[0].original_size, 52u);
  ASSERT_TRUE(result.patches[0].scratch_vgpr);
  EXPECT_EQ(*result.patches[0].scratch_vgpr, 5u);
  EXPECT_EQ(result.patches[1].kind, ConSanPatchKind::InlineFlatLoadCheckTrap);
  EXPECT_EQ(result.patches[1].anchor_offset, 76u);
  EXPECT_EQ(result.patches[1].trampoline_offset, 88u);
  EXPECT_EQ(result.patches[1].original_size, 52u);
  ASSERT_TRUE(result.patches[1].scratch_vgpr);
  EXPECT_EQ(*result.patches[1].scratch_vgpr, 5u);
}

TEST(ConSan, FlatStoreCheckTrapProofRewritesPaddedLocalFunctionSite) {
  const std::array<uint32_t, 1> kernel_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 19> function_words = {
      0xBE8001EBu,                           // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u, 0x00000000u,              // v_mov_b32_e64 v0, s0
      0xD5810001u, 0x00000001u,              // v_mov_b32_e64 v1, s1
      0xEC06807Cu, 0x01000000u, 0x00000000u, // flat_store_b32 v[0:1], v2
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_flat_check_trap = true;
  options.delay_nops = 1;
  options.scratch_vgpr = 5;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineFlatStoreCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 24u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 36u);
  EXPECT_EQ(result.patches.front().original_size, 52u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 5u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());
  EXPECT_NE(result.elf_bytes, bytes);

  ASSERT_GE(result.patches.front().required_sgpr_count, 2u);
  const uint16_t vcc_save_sgpr =
      static_cast<uint16_t>(result.patches.front().required_sgpr_count - 2u);
  const std::array<uint32_t, 13> expected_words = {
      *instrumentation::build_s_mov_b64(vcc_save_sgpr, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0xEC06807Cu,
      0x01000000u,
      0x00000000u, // original flat_store_b32 v[0:1], v2
      0xBF800000u, // delay
      0xEC05007Cu,
      0x00000005u,
      0x00000000u, // readback flat_load_b32 v5, v[0:1]
      0xBFC60000u, // s_wait_dscnt 0
      0x7C9A0B02u, // v_cmp_ne_u32_e32 v2, v5
      0xBFA30001u, // s_cbranch_vccz +1
      0xBF900000u, // s_trap 0
      *instrumentation::build_s_mov_b64(kRdna4VccLo, vcc_save_sgpr, ROCJITSU_CODE_ARCH_RDNA4),
  };
  const auto rewritten_words = patched_words_at_file_offset<expected_words.size()>(result, 0x118);
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(ConSan, FlatStoreB16CheckTrapProofEncodesRdna4Readback) {
  const std::array<uint32_t, 1> kernel_words = {
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  constexpr auto store =
      rdna4::build_vflat(rdna4::kFlatStoreB16Vflat, {.saddr = 124, .vsrc = 2, .vaddr = 0});
  std::vector<uint32_t> function_words = {
      0xBE8001EBu,              // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u, 0x00000000u, // v_mov_b32_e64 v0, s0
      0xD5810001u, 0x00000001u, // v_mov_b32_e64 v1, s1
      store[0],    store[1],    store[2],
  };
  function_words.resize(32, build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  function_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4));
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_flat_check_trap = true;
  options.delay_nops = 1;
  options.scratch_vgpr = 5;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineFlatStoreCheckTrap);
  EXPECT_TRUE(result.final_validation_passed);
  ASSERT_GE(result.patches.front().required_sgpr_count, 2u);
  const uint16_t vcc_save_sgpr =
      static_cast<uint16_t>(result.patches.front().required_sgpr_count - 2u);
  const auto rewritten_words = patched_words_at_file_offset<13>(result, 0x118);
  EXPECT_EQ(rewritten_words[0], *instrumentation::build_s_mov_b64(vcc_save_sgpr, kRdna4VccLo,
                                                                  ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(rewritten_words[1], store[0]);
  EXPECT_EQ(rewritten_words[2], store[1]);
  EXPECT_EQ(rewritten_words[3], store[2]);
  constexpr auto readback =
      rdna4::build_vflat(rdna4::kFlatLoadU16Vflat, {.saddr = 124, .vdst = 5, .vaddr = 0});
  EXPECT_EQ(rewritten_words[5], readback[0]);
  EXPECT_EQ(rewritten_words[6], readback[1]);
  EXPECT_EQ(rewritten_words[7], readback[2]);
  EXPECT_EQ(rewritten_words[12], *instrumentation::build_s_mov_b64(kRdna4VccLo, vcc_save_sgpr,
                                                                   ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(ConSan, FlatStoreCheckTrapProofRewritesGfx1250VflatStore) {
  const std::array<uint32_t, 1> kernel_words = {
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250),
  };
  constexpr auto store =
      gfx1250::build_vflat(gfx1250::kFlatStoreB32Vflat, {.saddr = 124, .vsrc = 2, .vaddr = 0});
  std::vector<uint32_t> function_words = {
      0xBE8001EBu, // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u,
      0x00000000u, // v_mov_b32_e64 v0, s0
      0xD5810001u,
      0x00000001u, // v_mov_b32_e64 v1, s1
      store[0],    store[1], store[2],
  };
  function_words.resize(34, build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250));
  function_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250));
  const std::vector<uint8_t> bytes =
      make_gfx1250_code_object_with_local_function(kernel_words, function_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_flat_check_trap = true;
  options.delay_nops = 1;
  options.scratch_vgpr = 5;
  options.report_buffer_address = 0x1234567887654321ull;
  options.report_marker = 0xABCDEF01u;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineFlatStoreCheckTrap);
  EXPECT_TRUE(result.final_validation_passed);
  ASSERT_GE(result.patches.front().required_sgpr_count, 2u);
  const uint16_t vcc_save_sgpr =
      static_cast<uint16_t>(result.patches.front().required_sgpr_count - 2u);
  const auto rewritten_words = patched_words_at_file_offset<8>(result, 0x118);
  EXPECT_EQ(rewritten_words[0], *instrumentation::build_s_mov_b64(vcc_save_sgpr, kRdna4VccLo,
                                                                  ROCJITSU_CODE_ARCH_GFX1250));
  EXPECT_EQ(rewritten_words[1], store[0]);
  EXPECT_EQ(rewritten_words[2], store[1]);
  EXPECT_EQ(rewritten_words[3], store[2]);
  constexpr auto readback =
      gfx1250::build_vflat(gfx1250::kFlatLoadB32Vflat, {.saddr = 124, .vdst = 5, .vaddr = 0});
  EXPECT_EQ(rewritten_words[5], readback[0]);
  EXPECT_EQ(rewritten_words[6], readback[1]);
  EXPECT_EQ(rewritten_words[7], readback[2]);
}

TEST(ConSan, FlatStoreCheckTrapProofRuntimeGatesGfx1250Wave64MaybeGroupReadback) {
  constexpr auto store =
      gfx1250::build_vflat(gfx1250::kFlatStoreD16HiB8Vflat, {.saddr = 124, .vsrc = 2, .vaddr = 0});
  std::vector<uint32_t> text_words = {
      0xBE8001EBu,              // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u, 0x00000080u, // v_mov_b32_e64 v0, 0
      0xD5810001u, 0x00000001u, // v_mov_b32_e64 v1, s1
      store[0],    store[1],    store[2],
  };
  text_words.resize(40u, build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250));
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250));
  const std::vector<uint8_t> bytes = make_gfx1250_code_object(
      text_words, "gfx1250_maybe_group_store", kRdna4Wave64AllVgprsGranulated, /*wave32=*/false);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_flat_check_trap = true;
  options.delay_nops = 1;
  options.scratch_vgpr = 5;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.kernels.front().flat_sites.size(), 1u);
  EXPECT_EQ(result.kernels.front().flat_sites.front().address_space_hint,
            ConSanFlatAddressSpaceHint::MaybeGroup);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineFlatStoreCheckTrap);
  EXPECT_TRUE(result.final_validation_passed);

  constexpr uint16_t kScalarInlineMinusOne = 0xC1;
  const auto group_aperture_high =
      build_v_cmp_eq_u32_e32_vcc(kScalarInlineMinusOne, /*vsrc1=*/1, ROCJITSU_CODE_ARCH_GFX1250);
  const auto skip_non_group =
      build_s_cbranch_vccz(/*offset_dwords=*/10, ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(group_aperture_high);
  ASSERT_TRUE(skip_non_group);
  ASSERT_GE(result.patches.front().required_sgpr_count, 2u);
  const uint16_t vcc_save_sgpr =
      static_cast<uint16_t>(result.patches.front().required_sgpr_count - 2u);
  const auto rewritten_words = patched_words_at_file_offset<18>(result, 0x114);
  EXPECT_EQ(rewritten_words[0], *instrumentation::build_s_mov_b64(vcc_save_sgpr, kRdna4VccLo,
                                                                  ROCJITSU_CODE_ARCH_GFX1250));
  EXPECT_EQ(rewritten_words[1], store[0]);
  EXPECT_EQ(rewritten_words[2], store[1]);
  EXPECT_EQ(rewritten_words[3], store[2]);
  EXPECT_EQ(rewritten_words[5], *group_aperture_high);
  EXPECT_EQ(rewritten_words[6], *skip_non_group);
  constexpr auto readback =
      gfx1250::build_vflat(gfx1250::kFlatLoadU8Vflat, {.saddr = 124, .vdst = 5, .vaddr = 0});
  EXPECT_EQ(rewritten_words[7], readback[0]);
  EXPECT_EQ(rewritten_words[8], readback[1]);
  EXPECT_EQ(rewritten_words[9], readback[2]);
  const auto select_high = instrumentation::build_v_lshrrev_b32(
      /*vdst=*/6, scalar_positive_inline_u32(16u), /*vsrc1=*/2, ROCJITSU_CODE_ARCH_GFX1250);
  const auto mask_byte = instrumentation::build_v_and_b32_literal(
      /*vdst=*/6, 0xffu, /*vsrc1=*/6, ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(select_high);
  ASSERT_TRUE(mask_byte);
  EXPECT_EQ(rewritten_words[11], *select_high);
  ASSERT_EQ(mask_byte->size(), 2u);
  EXPECT_EQ(rewritten_words[12], (*mask_byte)[0]);
  EXPECT_EQ(rewritten_words[13], (*mask_byte)[1]);
  EXPECT_EQ(rewritten_words[17], *instrumentation::build_s_mov_b64(kRdna4VccLo, vcc_save_sgpr,
                                                                   ROCJITSU_CODE_ARCH_GFX1250));
}

TEST(ConSan, Gfx1250FlatStoreCheckTrapSpillsLiveVccSavePairThroughVgprsInBothWaveModes) {
  constexpr auto store =
      gfx1250::build_vflat(gfx1250::kFlatStoreB32Vflat, {.saddr = 124, .vsrc = 2, .vaddr = 0});
  std::vector<uint32_t> text_words = {
      0xBE8001EBu, // s_mov_b64 s[0:1], src_shared_base
      build_v_mov_b32_e32(/*vdst=*/0, /*scalar s0=*/0, ROCJITSU_CODE_ARCH_GFX1250),
      build_v_mov_b32_e32(/*vdst=*/1, /*scalar s1=*/1, ROCJITSU_CODE_ARCH_GFX1250),
      store[0],
      store[1],
      store[2],
  };
  for (uint16_t sgpr = 0; sgpr < REGISTER_SET_ALLOCATABLE_SGPRS; ++sgpr)
    text_words.push_back(build_s_mov_b32(sgpr, sgpr, ROCJITSU_CODE_ARCH_GFX1250));
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250));
  for (const bool wave32 : {true, false}) {
    SCOPED_TRACE(wave32 ? "wave32" : "wave64");
    std::vector<uint8_t> bytes = make_gfx1250_code_object(
        text_words, "gfx1250_flat_scalar_vcc_spill", kRdna4Wave64AllVgprsGranulated, wave32);
    mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
      AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                      kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 13u);
    });
    ConSanOptions options;
    options.flavor = ConSanFlavor::SuperCollider;
    options.probe_flat_check_trap = true;
    options.flat_provenance_mode = ConSanFlatProvenanceMode::Strict;
    options.delay_mode = ConSanDelayMode::SleepVar;
    options.delay_nops = 1;
    options.delay_var_ssrc = 0u;
    options.scratch_vgpr = 3u;
    options.max_patches = 1u;

    const ConSanResult result = try_patch_consan(bytes, options);

    ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
    ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
    ASSERT_TRUE(result.final_validation_passed);
    const auto patch = std::ranges::find(
        result.patches, ConSanPatchKind::LocalCaveFlatStoreCheckTrap, &ConSanPatchInfo::kind);
    ASSERT_NE(patch, result.patches.end());
    ASSERT_TRUE(patch->scratch_vgpr);
    EXPECT_EQ(*patch->scratch_vgpr, 3u);
    ASSERT_TRUE(patch->scalar_vcc_spill_vgpr);
    EXPECT_EQ(*patch->scalar_vcc_spill_vgpr, 4u);
    EXPECT_EQ(patch->scalar_vcc_spill_vgpr_count, 1u);
    ASSERT_GE(patch->required_sgpr_count, 2u);
    EXPECT_LE(patch->required_sgpr_count, REGISTER_SET_ALLOCATABLE_SGPRS);
    const uint16_t vcc_save_sgpr = static_cast<uint16_t>(patch->required_sgpr_count - 2u);
    EXPECT_FALSE(vcc_save_sgpr == 0u || vcc_save_sgpr == 1u);

    AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(patched.is_valid());
    ASSERT_EQ(patched.text_sections().size(), 1u);
    const Section *text = patched.text_sections().front();
    std::vector<uint32_t> body(patch->trampoline_size / sizeof(uint32_t));
    std::memcpy(body.data(), text->data() + patch->trampoline_offset, patch->trampoline_size);
    ASSERT_GE(body.size(), 9u);
    constexpr uint16_t kScalarSpillVgpr = 4u;
    const auto save_lo = instrumentation::build_v_writelane_b32(
        kScalarSpillVgpr, vcc_save_sgpr, /*lane=*/0, ROCJITSU_CODE_ARCH_GFX1250);
    const auto save_hi = instrumentation::build_v_writelane_b32(
        kScalarSpillVgpr, vcc_save_sgpr + 1u, /*lane=*/1, ROCJITSU_CODE_ARCH_GFX1250);
    const auto restore_lo = instrumentation::build_v_readlane_b32(
        vcc_save_sgpr, kScalarSpillVgpr, /*lane=*/0, ROCJITSU_CODE_ARCH_GFX1250);
    const auto restore_hi = instrumentation::build_v_readlane_b32(
        vcc_save_sgpr + 1u, kScalarSpillVgpr, /*lane=*/1, ROCJITSU_CODE_ARCH_GFX1250);
    ASSERT_TRUE(save_lo);
    ASSERT_TRUE(save_hi);
    ASSERT_TRUE(restore_lo);
    ASSERT_TRUE(restore_hi);
    EXPECT_TRUE(std::ranges::equal(std::span(body).first<2>(), *save_lo));
    EXPECT_TRUE(std::ranges::equal(std::span(body).subspan(2u, 2u), *save_hi));
    EXPECT_TRUE(std::ranges::equal(std::span(body).subspan(body.size() - 5u, 2u), *restore_lo));
    EXPECT_TRUE(std::ranges::equal(std::span(body).subspan(body.size() - 3u, 2u), *restore_hi));
  }
}

[[nodiscard]] std::vector<uint32_t>
make_gfx1250_full_pressure_flat_store_words(bool append_endpgm = true) {
  constexpr auto store =
      gfx1250::build_vflat(gfx1250::kFlatStoreB32Vflat, {.saddr = 124, .vsrc = 2, .vaddr = 0});
  std::vector<uint32_t> words = {
      0xBE8001EBu, // s_mov_b64 s[0:1], src_shared_base
      build_v_mov_b32_e32(/*vdst=*/0, /*scalar s0=*/0, ROCJITSU_CODE_ARCH_GFX1250),
      build_v_mov_b32_e32(/*vdst=*/1, /*scalar s1=*/1, ROCJITSU_CODE_ARCH_GFX1250),
      store[0],
      store[1],
      store[2],
  };
  for (uint16_t sgpr = 0; sgpr < REGISTER_SET_ALLOCATABLE_SGPRS; ++sgpr)
    words.push_back(build_s_mov_b32(sgpr, sgpr, ROCJITSU_CODE_ARCH_GFX1250));
  for (uint16_t vgpr = 0; vgpr < REGISTER_SET_MAX_VGPRS; ++vgpr) {
    words.push_back(
        build_v_mov_b32_e32(vgpr, vector_source_vgpr(vgpr), ROCJITSU_CODE_ARCH_GFX1250));
  }
  if (append_endpgm)
    words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250));
  return words;
}

TEST(ConSan, Gfx1250FlatStoreCheckTrapSpillsSimultaneouslyLiveRegisterFilesInBothWaveModes) {
  const std::vector<uint32_t> text_words = make_gfx1250_full_pressure_flat_store_words();

  for (const bool wave32 : {true, false}) {
    SCOPED_TRACE(wave32 ? "wave32" : "wave64");
    std::vector<uint8_t> bytes = make_gfx1250_code_object(
        text_words, "gfx1250_flat_full_register_spill", kRdna4Wave64AllVgprsGranulated, wave32);
    mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
      AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                      kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 13u);
    });
    AmdGpuCodeObject original(bytes.data(), bytes.size());
    ASSERT_TRUE(original.is_valid());
    ASSERT_EQ(original.text_sections().size(), 1u);
    const uint64_t original_text_size = original.text_sections().front()->size();
    ConSanOptions options;
    options.flavor = ConSanFlavor::SuperCollider;
    options.probe_flat_check_trap = true;
    options.flat_provenance_mode = ConSanFlatProvenanceMode::Strict;
    options.delay_mode = ConSanDelayMode::SleepVar;
    options.delay_nops = 1;
    options.delay_var_ssrc = 0u;
    options.max_patches = 1u;

    const ConSanResult result = try_patch_consan(bytes, options);

    ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
    ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
    ASSERT_TRUE(result.final_validation_passed);
    const auto patch = std::ranges::find(
        result.patches, ConSanPatchKind::LocalCaveFlatStoreCheckTrap, &ConSanPatchInfo::kind);
    ASSERT_NE(patch, result.patches.end());
    EXPECT_EQ(patch->scratch_vgpr, 3u);
    EXPECT_EQ(patch->scalar_vcc_spill_sgpr, 2u);
    EXPECT_EQ(patch->scalar_vcc_spill_vgpr, 4u);
    EXPECT_EQ(patch->scalar_vcc_spill_vgpr_count, 2u);
    EXPECT_EQ(patch->spilled_vgpr_count, 3u);
    EXPECT_EQ(patch->required_private_segment_size, 12u);
    EXPECT_EQ(patch->trampoline_offset, original_text_size);
    EXPECT_GT(result.elf_bytes.size(), bytes.size());

    SpillManager expected_manager(/*original_private_bytes=*/0,
                                  *address_free_scratch_private_limit(ROCJITSU_CODE_ARCH_GFX1250));
    const auto expected_spill = build_vgpr_spill_sequence(
        expected_manager, /*vgpr_base=*/3, /*vgpr_count=*/3, ROCJITSU_CODE_ARCH_GFX1250);
    ASSERT_TRUE(expected_spill);
    AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(patched.is_valid());
    ASSERT_EQ(patched.kernels().size(), 1u);
    ASSERT_EQ(patched.text_sections().size(), 1u);
    const Section *text = patched.text_sections().front();
    std::vector<uint32_t> body(patch->trampoline_size / sizeof(uint32_t));
    std::memcpy(body.data(), text->data() + patch->trampoline_offset, patch->trampoline_size);
    ASSERT_GT(body.size(),
              1u + expected_spill->save_words.size() + expected_spill->restore_words.size());
    const auto skip_empty_wave = instrumentation::build_s_cbranch_execz(
        static_cast<int16_t>(body.size() - 2u), ROCJITSU_CODE_ARCH_GFX1250);
    ASSERT_TRUE(skip_empty_wave);
    EXPECT_EQ(body.front(), *skip_empty_wave);
    EXPECT_TRUE(std::ranges::equal(std::span(body).subspan(1u, expected_spill->save_words.size()),
                                   expected_spill->save_words));
    EXPECT_TRUE(std::ranges::equal(
        std::span(body).subspan(body.size() - 1u - expected_spill->restore_words.size(),
                                expected_spill->restore_words.size()),
        expected_spill->restore_words));

    KD descriptor{};
    std::memcpy(&descriptor,
                result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
                sizeof(descriptor));
    EXPECT_EQ(descriptor.private_segment_fixed_size, 12u);
    EXPECT_EQ(
        AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT),
        1u);
  }
}

TEST(ConSan, Gfx1250FullRegisterFlatDynamicStackFallbackIsAccounted) {
  constexpr std::string_view kKernelName = "gfx1250_flat_full_register_dynamic_stack";
  const std::vector<uint32_t> text_words = make_gfx1250_full_pressure_flat_store_words();
  std::vector<uint8_t> bytes = make_gfx1250_code_object(text_words, kKernelName);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 13u);
  });
  append_kernel_metadata_note(bytes, kKernelName, /*uses_dynamic_stack=*/true,
                              /*sgpr_count=*/REGISTER_SET_ALLOCATABLE_SGPRS);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_flat_check_trap = true;
  options.flat_provenance_mode = ConSanFlatProvenanceMode::Strict;
  options.delay_mode = ConSanDelayMode::SleepVar;
  options.delay_nops = 1;
  options.delay_var_ssrc = 0u;
  options.max_patches = 1u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.patches.empty());
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("dynamic_stack_spill_rejections=1") != std::string::npos;
  })) << testing::PrintToString(result.warnings);
}

TEST(ConSan, Gfx1250FullRegisterFlatPrivateCapacityFailureIsAccounted) {
  const std::vector<uint32_t> text_words = make_gfx1250_full_pressure_flat_store_words();
  std::vector<uint8_t> bytes =
      make_gfx1250_code_object(text_words, "gfx1250_flat_full_register_private_limit");
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 13u);
    descriptor.private_segment_fixed_size = kMaxAddressFreeScratchPrivateBytes;
  });
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_flat_check_trap = true;
  options.flat_provenance_mode = ConSanFlatProvenanceMode::Strict;
  options.delay_mode = ConSanDelayMode::SleepVar;
  options.delay_nops = 1;
  options.delay_var_ssrc = 0u;
  options.max_patches = 1u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.patches.empty());
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("private_spill_encoding_failures=1") != std::string::npos;
  })) << testing::PrintToString(result.warnings);
}

TEST(ConSan, Gfx1250FlatLoadCheckTrapSpillsLiveVccSavePairForFullAndHighHalfLoads) {
  struct LoadCase {
    const char *name;
    std::array<uint32_t, 3> words;
    uint16_t scalar_spill_vgpr;
  };
  constexpr std::array cases = {
      LoadCase{
          "b32",
          gfx1250::build_vflat(gfx1250::kFlatLoadB32Vflat, {.saddr = 124, .vdst = 2, .vaddr = 0}),
          4u},
      LoadCase{"d16_hi_b16",
               gfx1250::build_vflat(gfx1250::kFlatLoadD16HiB16Vflat,
                                    {.saddr = 124, .vdst = 2, .vaddr = 0}),
               5u},
  };
  for (const auto &load_case : cases) {
    SCOPED_TRACE(load_case.name);
    std::vector<uint32_t> text_words = {
        0xBE8001EBu, // s_mov_b64 s[0:1], src_shared_base
        build_v_mov_b32_e32(/*vdst=*/0, /*scalar s0=*/0, ROCJITSU_CODE_ARCH_GFX1250),
        build_v_mov_b32_e32(/*vdst=*/1, /*scalar s1=*/1, ROCJITSU_CODE_ARCH_GFX1250),
    };
    text_words.insert(text_words.end(), load_case.words.begin(), load_case.words.end());
    for (uint16_t sgpr = 0; sgpr < REGISTER_SET_ALLOCATABLE_SGPRS; ++sgpr)
      text_words.push_back(build_s_mov_b32(sgpr, sgpr, ROCJITSU_CODE_ARCH_GFX1250));
    text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250));

    std::vector<uint8_t> bytes =
        make_gfx1250_code_object(text_words, "gfx1250_flat_load_scalar_vcc_spill");
    mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
      AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                      kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 13u);
    });
    ConSanOptions options;
    options.flavor = ConSanFlavor::SuperCollider;
    options.probe_flat_check_trap = true;
    options.flat_provenance_mode = ConSanFlatProvenanceMode::Strict;
    options.delay_mode = ConSanDelayMode::SleepVar;
    options.delay_nops = 1;
    options.delay_var_ssrc = 0u;
    options.scratch_vgpr = 3u;
    options.max_patches = 1u;

    const ConSanResult result = try_patch_consan(bytes, options);

    ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
    ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
    ASSERT_TRUE(result.final_validation_passed);
    const auto patch = std::ranges::find(
        result.patches, ConSanPatchKind::LocalCaveFlatLoadCheckTrap, &ConSanPatchInfo::kind);
    ASSERT_NE(patch, result.patches.end());
    ASSERT_TRUE(patch->scalar_vcc_spill_vgpr);
    EXPECT_EQ(*patch->scalar_vcc_spill_vgpr, load_case.scalar_spill_vgpr);
    EXPECT_EQ(patch->scalar_vcc_spill_vgpr_count, 1u);
    const uint16_t vcc_save_sgpr = static_cast<uint16_t>(patch->required_sgpr_count - 2u);
    AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(patched.is_valid());
    const Section *text = patched.text_sections().front();
    std::vector<uint32_t> body(patch->trampoline_size / sizeof(uint32_t));
    std::memcpy(body.data(), text->data() + patch->trampoline_offset, patch->trampoline_size);
    ASSERT_GE(body.size(), 9u);
    const auto save_lo = instrumentation::build_v_writelane_b32(
        load_case.scalar_spill_vgpr, vcc_save_sgpr, /*lane=*/0, ROCJITSU_CODE_ARCH_GFX1250);
    const auto save_hi = instrumentation::build_v_writelane_b32(
        load_case.scalar_spill_vgpr, vcc_save_sgpr + 1u, /*lane=*/1, ROCJITSU_CODE_ARCH_GFX1250);
    const auto restore_lo = instrumentation::build_v_readlane_b32(
        vcc_save_sgpr, load_case.scalar_spill_vgpr, /*lane=*/0, ROCJITSU_CODE_ARCH_GFX1250);
    const auto restore_hi = instrumentation::build_v_readlane_b32(
        vcc_save_sgpr + 1u, load_case.scalar_spill_vgpr, /*lane=*/1, ROCJITSU_CODE_ARCH_GFX1250);
    ASSERT_TRUE(save_lo);
    ASSERT_TRUE(save_hi);
    ASSERT_TRUE(restore_lo);
    ASSERT_TRUE(restore_hi);
    EXPECT_TRUE(std::ranges::equal(std::span(body).first<2>(), *save_lo));
    EXPECT_TRUE(std::ranges::equal(std::span(body).subspan(2u, 2u), *save_hi));
    EXPECT_TRUE(std::ranges::equal(std::span(body).subspan(body.size() - 5u, 2u), *restore_lo));
    EXPECT_TRUE(std::ranges::equal(std::span(body).subspan(body.size() - 3u, 2u), *restore_hi));
  }
}

TEST(ConSan, Gfx950FlatCheckTrapFailsClosedWithoutDeadVccSavePair) {
  const auto store = build_cdna4_flat_store_b32(/*vaddr=*/0, /*vsrc=*/2, /*byte_offset=*/0,
                                                ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(store);
  std::vector<uint32_t> function_words = {
      0xBE8001EBu, // s_mov_b64 s[0:1], src_shared_base
      build_v_mov_b32_e32(/*vdst=*/0, /*scalar s0=*/0, ROCJITSU_CODE_ARCH_CDNA4),
      build_v_mov_b32_e32(/*vdst=*/1, /*scalar s1=*/1, ROCJITSU_CODE_ARCH_CDNA4),
  };
  function_words.insert(function_words.end(), store->begin(), store->end());
  for (uint16_t sgpr = 0; sgpr < REGISTER_SET_ALLOCATABLE_SGPRS; ++sgpr)
    function_words.push_back(build_s_mov_b32(sgpr, sgpr, ROCJITSU_CODE_ARCH_CDNA4));
  function_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4));
  const std::array<uint32_t, 1> kernel_words = {
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  const std::vector<uint8_t> bytes =
      make_cdna4_code_object_with_local_function(kernel_words, function_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_flat_check_trap = true;
  options.flat_provenance_mode = ConSanFlatProvenanceMode::Strict;
  options.delay_nops = 1u;
  options.scratch_vgpr = 3u;
  options.max_patches = 1u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.patches.empty());
}

TEST(ConSan, Gfx1250SharedFlatVccSpillUsesAllOwnersCommonSgprAllocation) {
  constexpr auto store =
      gfx1250::build_vflat(gfx1250::kFlatStoreB32Vflat, {.saddr = 124, .vsrc = 2, .vaddr = 0});
  std::vector<uint32_t> helper_words = {
      0xBE8001EBu, // s_mov_b64 s[0:1], src_shared_base
      build_v_mov_b32_e32(/*vdst=*/0, /*scalar s0=*/0, ROCJITSU_CODE_ARCH_GFX1250),
      build_v_mov_b32_e32(/*vdst=*/1, /*scalar s1=*/1, ROCJITSU_CODE_ARCH_GFX1250),
      store[0],
      store[1],
      store[2],
  };
  for (uint16_t sgpr = 0; sgpr < REGISTER_SET_ALLOCATABLE_SGPRS; ++sgpr)
    helper_words.push_back(build_s_mov_b32(sgpr, sgpr, ROCJITSU_CODE_ARCH_GFX1250));

  TwoKernelSharedFixtureOptions fixture;
  std::vector<uint8_t> bytes =
      make_two_kernel_shared_helper_code_object(fixture, ROCJITSU_CODE_ARCH_RDNA4, helper_words);
  mutate_elf_header(bytes,
                    [](Elf64_Ehdr &header) { header.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX1250; });
  AmdGpuCodeObject original(bytes.data(), bytes.size());
  ASSERT_TRUE(original.is_valid());
  ASSERT_EQ(original.kernels().size(), 3u);
  const auto first_owner =
      std::ranges::find(original.kernels(), "shared_owner_0", &AmdGpuKernelInfo::name);
  const auto second_owner =
      std::ranges::find(original.kernels(), "shared_owner_1", &AmdGpuKernelInfo::name);
  ASSERT_NE(first_owner, original.kernels().end());
  ASSERT_NE(second_owner, original.kernels().end());
  ASSERT_LE(first_owner->descriptor_file_offset + sizeof(KD), bytes.size());
  ASSERT_LE(second_owner->descriptor_file_offset + sizeof(KD), bytes.size());
  const auto set_sgpr_granulation = [&](uint64_t descriptor_offset, uint32_t granulated) {
    KD descriptor{};
    std::memcpy(&descriptor, bytes.data() + descriptor_offset, sizeof(descriptor));
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, granulated);
    std::memcpy(bytes.data() + descriptor_offset, &descriptor, sizeof(descriptor));
  };
  set_sgpr_granulation(first_owner->descriptor_file_offset, 3u);
  set_sgpr_granulation(second_owner->descriptor_file_offset, 0u);

  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_flat_check_trap = true;
  options.flat_provenance_mode = ConSanFlatProvenanceMode::Strict;
  options.delay_mode = ConSanDelayMode::SleepVar;
  options.delay_nops = 1;
  options.delay_var_ssrc = 0u;
  options.scratch_vgpr = 3u;
  options.max_patches = 1u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.final_validation_passed);
  const auto patch = std::ranges::find(result.patches, ConSanPatchKind::LocalCaveFlatStoreCheckTrap,
                                       &ConSanPatchInfo::kind);
  ASSERT_NE(patch, result.patches.end());
  ASSERT_TRUE(patch->scalar_vcc_spill_vgpr);
  ASSERT_EQ(patch->owner_descriptor_file_offsets.size(), 2u);
  EXPECT_EQ(patch->owner_descriptor_file_offsets[0], first_owner->descriptor_file_offset);
  EXPECT_EQ(patch->owner_descriptor_file_offsets[1], second_owner->descriptor_file_offset);
  // The second owner has only one eight-SGPR granule. The shared body must
  // borrow a pair that is already allocated by both owners rather than grow
  // one caller around a site-local choice.
  EXPECT_LE(patch->required_sgpr_count, 8u);
  EXPECT_GT(patch->required_sgpr_count, 2u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  for (const auto &[owner_name, expected_granulated] :
       {std::pair{"shared_owner_0", 3u}, std::pair{"shared_owner_1", 0u}}) {
    const auto owner = std::ranges::find(patched.kernels(), owner_name, &AmdGpuKernelInfo::name);
    ASSERT_NE(owner, patched.kernels().end());
    ASSERT_LE(owner->descriptor_file_offset + sizeof(KD), result.elf_bytes.size());
    KD descriptor{};
    std::memcpy(&descriptor, result.elf_bytes.data() + owner->descriptor_file_offset,
                sizeof(descriptor));
    EXPECT_EQ(AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc1,
                              kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT),
              expected_granulated);
  }
}

TEST(ConSan, Gfx1250SharedFlatRegisterSpillUsesOneLayoutForEveryOwner) {
  const std::vector<uint32_t> helper_words =
      make_gfx1250_full_pressure_flat_store_words(/*append_endpgm=*/false);

  TwoKernelSharedFixtureOptions fixture;
  fixture.first_private_bytes = 16u;
  fixture.second_private_bytes = 32u;
  std::vector<uint8_t> bytes =
      make_two_kernel_shared_helper_code_object(fixture, ROCJITSU_CODE_ARCH_RDNA4, helper_words);
  mutate_elf_header(bytes,
                    [](Elf64_Ehdr &header) { header.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX1250; });
  AmdGpuCodeObject original(bytes.data(), bytes.size());
  ASSERT_TRUE(original.is_valid());
  const auto first_owner =
      std::ranges::find(original.kernels(), "shared_owner_0", &AmdGpuKernelInfo::name);
  const auto second_owner =
      std::ranges::find(original.kernels(), "shared_owner_1", &AmdGpuKernelInfo::name);
  ASSERT_NE(first_owner, original.kernels().end());
  ASSERT_NE(second_owner, original.kernels().end());
  for (const uint64_t descriptor_offset :
       {first_owner->descriptor_file_offset, second_owner->descriptor_file_offset}) {
    KD descriptor{};
    std::memcpy(&descriptor, bytes.data() + descriptor_offset, sizeof(descriptor));
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 13u);
    std::memcpy(bytes.data() + descriptor_offset, &descriptor, sizeof(descriptor));
  }

  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_flat_check_trap = true;
  options.flat_provenance_mode = ConSanFlatProvenanceMode::Strict;
  options.delay_mode = ConSanDelayMode::SleepVar;
  options.delay_nops = 1;
  options.delay_var_ssrc = 0u;
  options.max_patches = 1u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.final_validation_passed);
  const auto patch = std::ranges::find(result.patches, ConSanPatchKind::LocalCaveFlatStoreCheckTrap,
                                       &ConSanPatchInfo::kind);
  ASSERT_NE(patch, result.patches.end());
  EXPECT_EQ(patch->scratch_vgpr, 3u);
  EXPECT_EQ(patch->scalar_vcc_spill_vgpr, 4u);
  EXPECT_EQ(patch->scalar_vcc_spill_vgpr_count, 2u);
  EXPECT_EQ(patch->spilled_vgpr_count, 3u);
  EXPECT_EQ(patch->required_private_segment_size, 44u);
  ASSERT_EQ(patch->owner_descriptor_file_offsets.size(), 2u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  for (std::string_view owner_name : {"shared_owner_0", "shared_owner_1"}) {
    const auto owner = std::ranges::find(patched.kernels(), owner_name, &AmdGpuKernelInfo::name);
    ASSERT_NE(owner, patched.kernels().end());
    KD descriptor{};
    std::memcpy(&descriptor, result.elf_bytes.data() + owner->descriptor_file_offset,
                sizeof(descriptor));
    EXPECT_EQ(descriptor.private_segment_fixed_size, 44u);
    EXPECT_EQ(
        AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT),
        1u);
  }
}

TEST(ConSan, FlatStoreCheckTrapProofCanUseSleepDelay) {
  const std::array<uint32_t, 1> kernel_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 19> function_words = {
      0xBE8001EBu,                           // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u, 0x00000000u,              // v_mov_b32_e64 v0, s0
      0xD5810001u, 0x00000001u,              // v_mov_b32_e64 v1, s1
      0xEC06807Cu, 0x01000000u, 0x00000000u, // flat_store_b32 v[0:1], v2
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_flat_check_trap = true;
  options.delay_mode = ConSanDelayMode::Sleep;
  options.delay_nops = 9;
  options.scratch_vgpr = 5;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineFlatStoreCheckTrap);
  EXPECT_EQ(result.patches.front().original_size, 52u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  ASSERT_GE(result.patches.front().required_sgpr_count, 2u);
  const uint16_t vcc_save_sgpr =
      static_cast<uint16_t>(result.patches.front().required_sgpr_count - 2u);
  const std::array<uint32_t, 13> expected_words = {
      *instrumentation::build_s_mov_b64(vcc_save_sgpr, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0xEC06807Cu,
      0x01000000u,
      0x00000000u, // original flat_store_b32 v[0:1], v2
      build_s_sleep(9, ROCJITSU_CODE_ARCH_RDNA4),
      0xEC05007Cu,
      0x00000005u,
      0x00000000u, // readback flat_load_b32 v5, v[0:1]
      0xBFC60000u, // s_wait_dscnt 0
      0x7C9A0B02u, // v_cmp_ne_u32_e32 v2, v5
      0xBFA30001u, // s_cbranch_vccz +1
      0xBF900000u, // s_trap 0
      *instrumentation::build_s_mov_b64(kRdna4VccLo, vcc_save_sgpr, ROCJITSU_CODE_ARCH_RDNA4),
  };
  const auto rewritten_words = patched_words_at_file_offset<expected_words.size()>(result, 0x118);
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(ConSan, FlatStoreCheckTrapProofCanUseSleepVarDelay) {
  const std::array<uint32_t, 1> kernel_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 19> function_words = {
      0xBE8001EBu,                           // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u, 0x00000000u,              // v_mov_b32_e64 v0, s0
      0xD5810001u, 0x00000001u,              // v_mov_b32_e64 v1, s1
      0xEC06807Cu, 0x01000000u, 0x00000000u, // flat_store_b32 v[0:1], v2
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_flat_check_trap = true;
  options.delay_mode = ConSanDelayMode::SleepVar;
  options.delay_nops = 1;
  options.delay_var_ssrc = 0u;
  options.scratch_vgpr = 5;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineFlatStoreCheckTrap);
  EXPECT_EQ(result.patches.front().original_size, 52u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  ASSERT_GE(result.patches.front().required_sgpr_count, 2u);
  const uint16_t vcc_save_sgpr =
      static_cast<uint16_t>(result.patches.front().required_sgpr_count - 2u);
  EXPECT_FALSE(vcc_save_sgpr == 0u || vcc_save_sgpr == 1u);
  const std::array<uint32_t, 13> expected_words = {
      *instrumentation::build_s_mov_b64(vcc_save_sgpr, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0xEC06807Cu,
      0x01000000u,
      0x00000000u, // original flat_store_b32 v[0:1], v2
      build_s_sleep_var(0u, ROCJITSU_CODE_ARCH_RDNA4),
      0xEC05007Cu,
      0x00000005u,
      0x00000000u, // readback flat_load_b32 v5, v[0:1]
      0xBFC60000u, // s_wait_dscnt 0
      0x7C9A0B02u, // v_cmp_ne_u32_e32 v2, v5
      0xBFA30001u, // s_cbranch_vccz +1
      0xBF900000u, // s_trap 0
      *instrumentation::build_s_mov_b64(kRdna4VccLo, vcc_save_sgpr, ROCJITSU_CODE_ARCH_RDNA4),
  };
  const auto rewritten_words = patched_words_at_file_offset<expected_words.size()>(result, 0x118);
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(ConSan, ProbeNopModeEmitsPatchedElfForCandidate) {
  const std::array<uint32_t, 7> text_words = {
      0xD8340000u, 0x00000000u, // ds_store_b32
      0xD8D80000u, 0x00000000u, // ds_load_b32
      0xBFC60000u,              // s_wait_dscnt
      0x06040F06u,              // v_add_f32_e32 v2, v6, v7
      0xBFB00000u,              // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_nop = true;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.text_sections.size(), 1u);
  EXPECT_EQ(result.kernels.front().preflight_action, ConSanPreflightAction::Candidate);
  EXPECT_TRUE(result.modified);
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_FALSE(result.elf_bytes.empty());
  EXPECT_NE(result.elf_bytes, bytes);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::TrampolineNop);
  EXPECT_EQ(result.patches.front().anchor_offset, 20u);
  EXPECT_EQ(result.patches.front().original_size, 4u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  EXPECT_GT(patched.text_sections().front()->size(), result.text_sections.front().size);
}

TEST(ConSan, ProbeNopModeRewritesExistingNopInPlace) {
  const std::array<uint32_t, 4> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32
      0xBF800000u, // s_nop 0
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_nop = true;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineNopRewrite);
  EXPECT_EQ(result.patches.front().anchor_offset, 8u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 8u);
  EXPECT_EQ(result.patches.front().original_size, 4u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());
  EXPECT_NE(result.elf_bytes, bytes);

  uint32_t rewritten_word = 0;
  std::memcpy(&rewritten_word, result.elf_bytes.data() + 0x100 + 8, sizeof(rewritten_word));
  EXPECT_EQ(rewritten_word, 0xBF800001u);
}

TEST(ConSan, ProbeTrampolineNopModeSkipsExistingNopRewrite) {
  const std::array<uint32_t, 5> text_words = {
      0xBF800000u, // s_nop 0
      0xD8340000u,
      0x00000000u, // ds_store_b32
      0x06040F06u, // v_add_f32_e32 v2, v6, v7
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_trampoline_nop = true;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::TrampolineNop);
  EXPECT_EQ(result.patches.front().anchor_offset, 12u);
  EXPECT_EQ(result.patches.front().original_size, 4u);
  ASSERT_GT(result.elf_bytes.size(), bytes.size());

  uint32_t original_nop = 0;
  std::memcpy(&original_nop, result.elf_bytes.data() + 0x100, sizeof(original_nop));
  EXPECT_EQ(original_nop, 0xBF800000u);
}

TEST(ConSan, ProbeTrampolineNopModeSkipsSClauseRun) {
  const std::array<uint32_t, 10> text_words = {
      0xD8340000u, 0x00000000u, // ds_store_b32
      0xBF850001u,              // s_clause 1
      0xF4002200u, 0xF8000020u, // s_load_b64 s[8:9], s[0:1], 0x20
      0xF4006000u, 0xF8000000u, // s_load_b256 s[0:7], s[0:1], 0x0
      0xBFC70000u,              // s_wait_kmcnt 0
      0x06040F06u,              // v_add_f32_e32 v2, v6, v7
      0xBFB00000u,              // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_trampoline_nop = true;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::TrampolineNop);
  EXPECT_EQ(result.patches.front().anchor_offset, 32u);
  EXPECT_EQ(result.patches.front().original_size, 4u);
  ASSERT_GT(result.elf_bytes.size(), bytes.size());

  const auto prefix_words = patched_words_at_file_offset<8>(result, 0x100);
  EXPECT_EQ(prefix_words[0], 0xD8340000u);
  EXPECT_EQ(prefix_words[1], 0x00000000u);
  EXPECT_EQ(prefix_words[2], 0xBF850001u);
  EXPECT_EQ(prefix_words[3], 0xF4002200u);
  EXPECT_EQ(prefix_words[4], 0xF8000020u);
  EXPECT_EQ(prefix_words[5], 0xF4006000u);
  EXPECT_EQ(prefix_words[6], 0xF8000000u);
  EXPECT_EQ(prefix_words[7], 0xBFC70000u);
}

TEST(ConSan, ProbeTrampolineNopModeSkipsRocclrRuntimeHelpers) {
  const std::array<uint32_t, 4> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32
      0x06040F06u, // v_add_f32_e32 v2, v6, v7
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "__amd_rocclr_fillBufferAligned");
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_trampoline_nop = true;

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

TEST(ConSan, ProbeTrampolineNopModeSkipsCodeObjectWithoutCandidateSites) {
  const std::vector<uint8_t> bytes = make_rdna4_ds_atomic_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_trampoline_nop = true;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
  EXPECT_TRUE(result.patches.empty());

  bool saw_skip_warning = false;
  for (const std::string &warning : result.warnings)
    saw_skip_warning |= warning.find("without supported DBI candidate") != std::string::npos;
  EXPECT_TRUE(saw_skip_warning);
}

TEST(ConSan, ProbeNopModePrefersVectorAluAnchorOverMemoryAnchor) {
  const std::array<uint32_t, 4> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32
      0x06040F06u, // v_add_f32_e32 v2, v6, v7
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_nop = true;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::TrampolineNop);
  EXPECT_EQ(result.patches.front().anchor_offset, 8u);
  EXPECT_EQ(result.patches.front().original_size, 4u);
}

TEST(ConSan, ProbeEndpgmModeRewritesVectorAluInPlace) {
  const std::array<uint32_t, 4> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32
      0x06040F06u, // v_add_f32_e32 v2, v6, v7
      0xBF800000u, // s_nop 0
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_nop = true;
  options.probe_endpgm = true;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineEndpgmRewrite);
  EXPECT_EQ(result.patches.front().anchor_offset, 8u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 8u);
  EXPECT_EQ(result.patches.front().original_size, 4u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());
  EXPECT_NE(result.elf_bytes, bytes);

  uint32_t rewritten_word = 0;
  std::memcpy(&rewritten_word, result.elf_bytes.data() + 0x100 + 8, sizeof(rewritten_word));
  EXPECT_EQ(rewritten_word, 0xBFB00000u);
}

TEST(ConSan, ProbeLdsEndpgmModeRewritesFirstSupportedReadInPlace) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_endpgm = true;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineLdsEndpgmRewrite);
  EXPECT_EQ(result.patches.front().anchor_offset, 8u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 8u);
  EXPECT_EQ(result.patches.front().original_size, 8u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());
  EXPECT_NE(result.elf_bytes, bytes);

  uint32_t rewritten_word = 0;
  std::memcpy(&rewritten_word, result.elf_bytes.data() + 0x100 + 8, sizeof(rewritten_word));
  EXPECT_EQ(rewritten_word, 0xBFB00000u);
}

TEST(ConSan, ProbeLdsCheckTrapModeRewritesPaddedLoadInPlace) {
  const std::array<uint32_t, 13> text_words = {
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3;
  options.delay_nops = 2;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineLdsLoadCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 8u);
  EXPECT_EQ(result.patches.front().original_size, 48u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 3u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  const std::array<uint32_t, 13> expected_words = {
      0xD8D80000u,
      0x01000002u, // original ds_load_b32 v1, v2
      0xBF800000u,
      0xBF800000u, // delay
      0xD8D80000u,
      0x03000002u, // duplicate ds_load_b32 v3, v2
      0xBFC60000u, // s_wait_dscnt 0
      build_s_mov_b32(0, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0x7C9A0701u, // v_cmp_ne_u32_e32 vcc_lo, v1, v3
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      build_s_mov_b32(kRdna4VccLo, 0, ROCJITSU_CODE_ARCH_RDNA4),
      0xBFB00000u, // original s_endpgm after padding
  };
  const auto rewritten_words = patched_words_at_file_offset<expected_words.size()>(result, 0x100);
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(ConSan, ProbeLdsCheckTrapModeRewritesGfx1250VdsLoadInPlace) {
  constexpr auto load = gfx1250::build_vds(gfx1250::kDsLoadB32Vds, {.addr = 2, .vdst = 1});
  const std::array<uint32_t, 13> text_words = {
      load[0],
      load[1],
      build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250),
      build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250),
      build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250),
      build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250),
      build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250),
      build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250),
      build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250),
      build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250),
      build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250),
      build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250),
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250),
  };
  const std::vector<uint8_t> bytes = make_gfx1250_code_object(text_words, "gfx1250_vds_load");
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3;
  options.delay_nops = 2;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineLdsLoadCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 8u);
  EXPECT_EQ(result.patches.front().original_size, 48u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 3u);
  EXPECT_TRUE(result.final_validation_passed);

  const auto rewritten_words = patched_words_at_file_offset<6>(result, 0x100);
  EXPECT_EQ(rewritten_words[0], load[0]);
  EXPECT_EQ(rewritten_words[1], load[1]);
  const auto duplicate = gfx1250::build_vds(gfx1250::kDsLoadB32Vds, {.addr = 2, .vdst = 3});
  EXPECT_EQ(rewritten_words[4], duplicate[0]);
  EXPECT_EQ(rewritten_words[5], duplicate[1]);
}

TEST(ConSan, ProbeLdsCheckTrapModePreservesGfx1250GuestVgprBankMode) {
  constexpr auto load = gfx1250::build_vds(gfx1250::kDsLoadB32Vds, {.addr = 2, .vdst = 1});
  constexpr uint32_t kSelectLowVgprBank = 0xBF860100u;
  constexpr uint32_t kSelectGuestVgprBank = 0xBF860001u;
  std::vector<uint32_t> text_words = {kSelectGuestVgprBank, load[0], load[1]};
  text_words.resize(32u, build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250));
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250);
  const std::vector<uint8_t> bytes =
      make_gfx1250_code_object(text_words, "gfx1250_vds_banked_load");
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified);
  const auto patch = std::ranges::find(result.patches, ConSanPatchKind::InlineLdsLoadCheckTrap,
                                       &ConSanPatchInfo::kind);
  ASSERT_NE(patch, result.patches.end());
  EXPECT_EQ(patch->anchor_offset, sizeof(uint32_t));
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const auto patched_text = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(patched.text_sections().front()->data()),
      patched.text_sections().front()->size());
  const auto word_at = [&](uint64_t offset) {
    uint32_t word = 0;
    std::memcpy(&word, patched_text.data() + offset, sizeof(word));
    return word;
  };
  EXPECT_EQ(word_at(patch->anchor_offset), load[0]);
  EXPECT_EQ(word_at(patch->anchor_offset + sizeof(uint32_t)), load[1]);
  EXPECT_EQ(word_at(patch->anchor_offset + 2u * sizeof(uint32_t)), kSelectLowVgprBank);
  EXPECT_EQ(word_at(patch->anchor_offset + patch->original_size - sizeof(uint32_t)),
            kSelectGuestVgprBank);
}

TEST(ConSan, ProbeLdsCheckTrapModePreservesGfx1250LowBankAddressForHighBankLoad) {
  constexpr auto load = gfx1250::build_vds(gfx1250::kDsLoadB128Vds, {.addr = 2, .vdst = 1});
  constexpr uint32_t kSelectLowVgprBank = 0xBF864000u;
  constexpr uint32_t kSelectGuestVgprBank = 0xBF860040u;
  std::vector<uint32_t> text_words = {kSelectGuestVgprBank, load[0], load[1]};
  text_words.resize(48u, build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250));
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250);
  const std::vector<uint8_t> bytes =
      make_gfx1250_code_object(text_words, "gfx1250_vds_banked_overlapping_address");
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 8;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified);
  const auto patch = std::ranges::find(result.patches, ConSanPatchKind::InlineLdsLoadCheckTrap,
                                       &ConSanPatchInfo::kind);
  ASSERT_NE(patch, result.patches.end());
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const auto patched_text = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(patched.text_sections().front()->data()),
      patched.text_sections().front()->size());
  const auto word_at = [&](uint64_t offset) {
    uint32_t word = 0;
    std::memcpy(&word, patched_text.data() + offset, sizeof(word));
    return word;
  };
  const auto save_address =
      build_v_mov_b32_e32(12, vector_source_vgpr(2), ROCJITSU_CODE_ARCH_GFX1250);
  EXPECT_EQ(word_at(patch->anchor_offset), kSelectLowVgprBank);
  EXPECT_EQ(word_at(patch->anchor_offset + sizeof(uint32_t)), save_address);
  EXPECT_EQ(word_at(patch->anchor_offset + 2u * sizeof(uint32_t)), kSelectGuestVgprBank);
  EXPECT_EQ(word_at(patch->anchor_offset + 3u * sizeof(uint32_t)), load[0]);
  EXPECT_EQ(word_at(patch->anchor_offset + 4u * sizeof(uint32_t)), load[1]);
  EXPECT_EQ(word_at(patch->anchor_offset + 5u * sizeof(uint32_t)), kSelectLowVgprBank);
  EXPECT_EQ(word_at(patch->anchor_offset + patch->original_size - sizeof(uint32_t)),
            kSelectGuestVgprBank);
}

TEST(ConSan, ProbeLdsCheckTrapModeComparesGfx1250LoadAcrossVgprBankBoundary) {
  constexpr auto load = gfx1250::build_vds(gfx1250::kDsLoadB128Vds, {.addr = 2, .vdst = 254});
  std::vector<uint32_t> text_words = {load[0], load[1]};
  text_words.resize(48u, build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250));
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250);
  const std::vector<uint8_t> bytes =
      make_gfx1250_code_object(text_words, "gfx1250_vds_cross_bank_load");
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto patch = std::ranges::find(result.patches, ConSanPatchKind::InlineLdsLoadCheckTrap,
                                       &ConSanPatchInfo::kind);
  ASSERT_NE(patch, result.patches.end());
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const auto patched_text = std::span<const uint32_t>(
      reinterpret_cast<const uint32_t *>(patched.text_sections().front()->data()),
      patched.text_sections().front()->size() / sizeof(uint32_t));
  EXPECT_EQ(std::ranges::count(patched_text, 0xBF860001u), 2u);
  EXPECT_EQ(std::ranges::count(patched_text, 0xBF860100u), 2u);
}

TEST(ConSan, ProbeLdsCheckTrapModeRewritesGfx1250TransposeLoadWithSymbolPadding) {
  constexpr auto load = gfx1250::build_vds(gfx1250::kDsLoadTr16B128Vds, {.addr = 2, .vdst = 4});
  std::vector<uint32_t> text_words = {load[0], load[1]};
  text_words.insert(text_words.end(), 48u, build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250));
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250));
  text_words.insert(text_words.end(), 3u, 0u); // Kernel-symbol alignment padding.
  const std::vector<uint8_t> bytes =
      make_gfx1250_code_object(text_words, "gfx1250_vds_transpose_load");
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 20;
  options.delay_nops = 2;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineLdsLoadCheckTrap);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_EQ(result.kernels.front().stats.decode_error_count, 0u);
}

TEST(ConSan, ProbeLdsCheckTrapModeRewritesGfx1250U16VdsLoadInPlace) {
  constexpr auto load = gfx1250::build_vds(gfx1250::kDsLoadU16Vds, {.addr = 2, .vdst = 1});
  std::vector<uint32_t> text_words = {load[0], load[1]};
  text_words.insert(text_words.end(), 10u, build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250));
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250));
  const std::vector<uint8_t> bytes = make_gfx1250_code_object(text_words, "gfx1250_vds_load_u16");
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3;
  options.delay_nops = 2;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineLdsLoadCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 8u);
  EXPECT_EQ(result.patches.front().original_size, 48u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 3u);
  EXPECT_TRUE(result.final_validation_passed);

  const auto rewritten_words = patched_words_at_file_offset<6>(result, 0x100);
  EXPECT_EQ(rewritten_words[0], load[0]);
  EXPECT_EQ(rewritten_words[1], load[1]);
  const auto duplicate = gfx1250::build_vds(gfx1250::kDsLoadU16Vds, {.addr = 2, .vdst = 3});
  EXPECT_EQ(rewritten_words[4], duplicate[0]);
  EXPECT_EQ(rewritten_words[5], duplicate[1]);
}

TEST(ConSan, ProbeLdsCheckTrapModeRewritesGfx1250I16VdsLoadInPlace) {
  constexpr auto load = gfx1250::build_vds(gfx1250::kDsLoadI16Vds, {.addr = 2, .vdst = 1});
  std::vector<uint32_t> text_words = {load[0], load[1]};
  text_words.insert(text_words.end(), 10u, build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250));
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250));
  const std::vector<uint8_t> bytes = make_gfx1250_code_object(text_words, "gfx1250_vds_load_i16");
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3;
  options.delay_nops = 2;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineLdsLoadCheckTrap);
  EXPECT_TRUE(result.final_validation_passed);

  const auto rewritten_words = patched_words_at_file_offset<6>(result, 0x100);
  const auto duplicate = gfx1250::build_vds(gfx1250::kDsLoadI16Vds, {.addr = 2, .vdst = 3});
  EXPECT_EQ(rewritten_words[4], duplicate[0]);
  EXPECT_EQ(rewritten_words[5], duplicate[1]);
}

TEST(ConSan, ProbeLdsCheckTrapModeRewritesGfx1250U8VdsLoadInPlace) {
  constexpr auto load = gfx1250::build_vds(gfx1250::kDsLoadU8Vds, {.addr = 2, .vdst = 1});
  std::vector<uint32_t> text_words = {load[0], load[1]};
  text_words.insert(text_words.end(), 10u, build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250));
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250));
  const std::vector<uint8_t> bytes = make_gfx1250_code_object(text_words, "gfx1250_vds_load_u8");
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3;
  options.delay_nops = 2;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineLdsLoadCheckTrap);
  EXPECT_TRUE(result.final_validation_passed);

  const auto rewritten_words = patched_words_at_file_offset<6>(result, 0x100);
  EXPECT_EQ(rewritten_words[0], load[0]);
  EXPECT_EQ(rewritten_words[1], load[1]);
  const auto duplicate = gfx1250::build_vds(gfx1250::kDsLoadU8Vds, {.addr = 2, .vdst = 3});
  EXPECT_EQ(rewritten_words[4], duplicate[0]);
  EXPECT_EQ(rewritten_words[5], duplicate[1]);
}

TEST(ConSan, ProbeLdsCheckTrapModeRewritesGfx1250I8VdsLoadInPlace) {
  constexpr auto load = gfx1250::build_vds(gfx1250::kDsLoadI8Vds, {.addr = 2, .vdst = 1});
  std::vector<uint32_t> text_words = {load[0], load[1]};
  text_words.insert(text_words.end(), 10u, build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250));
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250));
  const std::vector<uint8_t> bytes = make_gfx1250_code_object(text_words, "gfx1250_vds_load_i8");
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3;
  options.delay_nops = 2;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineLdsLoadCheckTrap);
  EXPECT_TRUE(result.final_validation_passed);
  const auto rewritten_words = patched_words_at_file_offset<6>(result, 0x100);
  constexpr auto duplicate = gfx1250::build_vds(gfx1250::kDsLoadI8Vds, {.addr = 2, .vdst = 3});
  EXPECT_EQ(rewritten_words[4], duplicate[0]);
  EXPECT_EQ(rewritten_words[5], duplicate[1]);
}

TEST(ConSan, ProbeLdsCheckTrapModeRewritesGfx1250B96VdsLoadInPlace) {
  constexpr auto load = gfx1250::build_vds(gfx1250::kDsLoadB96Vds, {.addr = 10, .vdst = 1});
  std::vector<uint32_t> text_words = {load[0], load[1]};
  text_words.insert(text_words.end(), 32u, build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250));
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250));
  const std::vector<uint8_t> bytes = make_gfx1250_code_object(text_words, "gfx1250_vds_load_b96");
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 4;
  options.delay_nops = 2;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineLdsLoadCheckTrap);
  ASSERT_EQ(result.kernels.size(), 1u);
  ASSERT_EQ(result.kernels.front().lds_sites.size(), 1u);
  EXPECT_TRUE(result.kernels.front().lds_sites.front().supported_mvp);
  EXPECT_TRUE(result.final_validation_passed);
  const auto rewritten_words = patched_words_at_file_offset<6>(result, 0x100);
  constexpr auto duplicate = gfx1250::build_vds(gfx1250::kDsLoadB96Vds, {.addr = 10, .vdst = 4});
  EXPECT_EQ(rewritten_words[4], duplicate[0]);
  EXPECT_EQ(rewritten_words[5], duplicate[1]);
}

TEST(ConSan, ProbeLdsCheckTrapModeSpillsLiveGfx1250ScratchWindow) {
  constexpr auto load = gfx1250::build_vds(gfx1250::kDsLoadB32Vds, {.addr = 2, .vdst = 1});
  std::vector<uint32_t> text_words = {load[0], load[1]};
  text_words.insert(text_words.end(), 24u, build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250));
  for (uint16_t vgpr = 0; vgpr < 256; ++vgpr) {
    text_words.push_back(
        build_v_mov_b32_e32(vgpr, vector_source_vgpr(vgpr), ROCJITSU_CODE_ARCH_GFX1250));
  }
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250));
  const std::vector<uint8_t> bytes =
      make_gfx1250_code_object(text_words, "gfx1250_vds_spill", /*vgpr_granulated=*/15u);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.delay_nops = 1;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &candidate) {
    return candidate.spilled_vgpr_count != 0;
  });
  ASSERT_NE(patch, result.patches.end());
  EXPECT_EQ(patch->spilled_vgpr_count, 1u);
  EXPECT_EQ(patch->required_private_segment_size, 4u);
  ASSERT_TRUE(patch->scratch_vgpr);
  EXPECT_EQ(*patch->scratch_vgpr, 0u);
  EXPECT_TRUE(result.final_validation_passed);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.kernels().size(), 1u);
  ASSERT_EQ(patch->owner_descriptor_file_offsets.size(), 1u);
  EXPECT_EQ(patch->owner_descriptor_file_offsets.front(),
            patched.kernels().front().descriptor_file_offset);
  KD descriptor{};
  std::memcpy(&descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  EXPECT_EQ(descriptor.private_segment_fixed_size, 4u);
  EXPECT_EQ(
      AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT),
      1u);
}

TEST(ConSan, ProbeLdsCheckTrapModeRewritesGfx1250VdsStoreInPlace) {
  constexpr auto store = gfx1250::build_vds(gfx1250::kDsStoreB32Vds, {.addr = 2, .data0 = 1});
  const std::array<uint32_t, 13> text_words = {
      store[0],
      store[1],
      build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250),
      build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250),
      build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250),
      build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250),
      build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250),
      build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250),
      build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250),
      build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250),
      build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250),
      build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250),
      build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250),
  };
  const std::vector<uint8_t> bytes = make_gfx1250_code_object(text_words, "gfx1250_vds_store");
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3;
  options.delay_nops = 2;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineLdsStoreCheckTrap);
  EXPECT_TRUE(result.final_validation_passed);

  const auto rewritten_words = patched_words_at_file_offset<6>(result, 0x100);
  EXPECT_EQ(rewritten_words[0], store[0]);
  EXPECT_EQ(rewritten_words[1], store[1]);
  constexpr auto readback = gfx1250::build_vds(gfx1250::kDsLoadB32Vds, {.addr = 2, .vdst = 3});
  EXPECT_EQ(rewritten_words[4], readback[0]);
  EXPECT_EQ(rewritten_words[5], readback[1]);
}

TEST(ConSan, ProbeLdsCheckTrapModeReadsBackGfx1250B96VdsStore) {
  constexpr auto store = gfx1250::build_vds(gfx1250::kDsStoreB96Vds, {.addr = 10, .data0 = 1});
  std::vector<uint32_t> text_words = {store[0], store[1]};
  text_words.insert(text_words.end(), 32u, build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250));
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250));
  const std::vector<uint8_t> bytes = make_gfx1250_code_object(text_words, "gfx1250_vds_store_b96");
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 4;
  options.delay_nops = 2;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineLdsStoreCheckTrap);
  EXPECT_TRUE(result.final_validation_passed);

  const auto rewritten_words = patched_words_at_file_offset<6>(result, 0x100);
  constexpr auto readback = gfx1250::build_vds(gfx1250::kDsLoadB96Vds, {.addr = 10, .vdst = 4});
  EXPECT_EQ(rewritten_words[4], readback[0]);
  EXPECT_EQ(rewritten_words[5], readback[1]);
}

TEST(ConSan, ProbeLdsCheckTrapModeMasksGfx1250B8VdsStoreBeforeComparingReadback) {
  constexpr auto store = gfx1250::build_vds(gfx1250::kDsStoreB8Vds, {.addr = 2, .data0 = 1});
  std::vector<uint32_t> text_words = {store[0], store[1]};
  text_words.insert(text_words.end(), 12u, build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250));
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250));
  const std::vector<uint8_t> bytes = make_gfx1250_code_object(text_words, "gfx1250_vds_store_b8");
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3;
  options.delay_nops = 2;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineLdsStoreCheckTrap);
  EXPECT_TRUE(result.final_validation_passed);

  const auto rewritten_words = patched_words_at_file_offset<10>(result, 0x100);
  EXPECT_EQ(rewritten_words[0], store[0]);
  EXPECT_EQ(rewritten_words[1], store[1]);
  constexpr auto readback = gfx1250::build_vds(gfx1250::kDsLoadU8Vds, {.addr = 2, .vdst = 3});
  EXPECT_EQ(rewritten_words[4], readback[0]);
  EXPECT_EQ(rewritten_words[5], readback[1]);
  const auto mask = build_v_and_b32_e32_literal(4, 0xffu, 1, ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(mask);
  ASSERT_EQ(mask->size(), 2u);
  EXPECT_EQ(rewritten_words[7], (*mask)[0]);
  EXPECT_EQ(rewritten_words[8], (*mask)[1]);
}

TEST(ConSan, ProbeLdsCheckTrapModeSelectsGfx1250HighByteStoreValue) {
  constexpr auto store = gfx1250::build_vds(gfx1250::kDsStoreB8D16HiVds, {.addr = 2, .data0 = 1});
  std::vector<uint32_t> text_words = {store[0], store[1]};
  text_words.insert(text_words.end(), 14u, build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250));
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250));
  const std::vector<uint8_t> bytes =
      make_gfx1250_code_object(text_words, "gfx1250_vds_store_b8_d16_hi");
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3;
  options.delay_nops = 2;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineLdsStoreCheckTrap);
  EXPECT_TRUE(result.final_validation_passed);
  const auto rewritten_words = patched_words_at_file_offset<11>(result, 0x100);
  constexpr auto readback = gfx1250::build_vds(gfx1250::kDsLoadU8Vds, {.addr = 2, .vdst = 3});
  EXPECT_EQ(rewritten_words[4], readback[0]);
  EXPECT_EQ(rewritten_words[5], readback[1]);
  const auto shift =
      build_v_lshrrev_b32_e32(4, scalar_positive_inline_u32(16u), 1, ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(shift);
  EXPECT_EQ(rewritten_words[7], *shift);
  const auto mask = build_v_and_b32_e32_literal(4, 0xffu, 4, ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(mask);
  EXPECT_EQ(rewritten_words[8], (*mask)[0]);
  EXPECT_EQ(rewritten_words[9], (*mask)[1]);
}

TEST(ConSan, ProbeLdsCheckTrapModeMasksGfx1250B16StoreValues) {
  const auto check = [](uint16_t store_op, bool high_half, std::string_view kernel_name) {
    const auto store = gfx1250::build_vds(store_op, {.addr = 2, .data0 = 1});
    std::vector<uint32_t> text_words = {store[0], store[1]};
    text_words.insert(text_words.end(), 14u, build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250));
    text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250));
    const std::vector<uint8_t> bytes = make_gfx1250_code_object(text_words, kernel_name);
    ConSanOptions options;
    options.flavor = ConSanFlavor::SuperCollider;
    options.probe_lds_check_trap = true;
    options.scratch_vgpr = 3;
    options.delay_nops = 2;

    const ConSanResult result = try_patch_consan(bytes, options);

    EXPECT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
    EXPECT_TRUE(result.modified) << testing::PrintToString(result.warnings);
    EXPECT_TRUE(result.final_validation_passed);
    ASSERT_EQ(result.kernels.size(), 1u);
    ASSERT_EQ(result.kernels.front().lds_sites.size(), 1u);
    EXPECT_TRUE(result.kernels.front().lds_sites.front().supported_mvp);
    if (!result.modified)
      return;
    const auto rewritten_words = patched_words_at_file_offset<11>(result, 0x100);
    const auto readback = gfx1250::build_vds(gfx1250::kDsLoadU16Vds, {.addr = 2, .vdst = 3});
    EXPECT_EQ(rewritten_words[4], readback[0]);
    EXPECT_EQ(rewritten_words[5], readback[1]);
    size_t mask_index = 7u;
    if (high_half) {
      const auto shift = build_v_lshrrev_b32_e32(4, scalar_positive_inline_u32(16u), 1,
                                                 ROCJITSU_CODE_ARCH_GFX1250);
      ASSERT_TRUE(shift);
      EXPECT_EQ(rewritten_words[7], *shift);
      mask_index = 8u;
    }
    const auto mask =
        build_v_and_b32_e32_literal(4, 0xffffu, high_half ? 4u : 1u, ROCJITSU_CODE_ARCH_GFX1250);
    ASSERT_TRUE(mask);
    EXPECT_EQ(rewritten_words[mask_index], (*mask)[0]);
    EXPECT_EQ(rewritten_words[mask_index + 1u], (*mask)[1]);
  };

  check(gfx1250::kDsStoreB16Vds, false, "gfx1250_vds_store_b16");
  check(gfx1250::kDsStoreB16D16HiVds, true, "gfx1250_vds_store_b16_d16_hi");
}

TEST(ConSan, ProbeLdsCheckTrapModeReadsBackGfx1250Stride64Stores) {
  const auto check = [](uint16_t store_op, uint16_t load_op, uint8_t data1,
                        std::string_view kernel_name) {
    const auto store =
        gfx1250::build_vds(store_op, {.offset1 = 8, .addr = 12, .data0 = 2, .data1 = data1});
    std::vector<uint32_t> text_words = {store[0], store[1]};
    text_words.insert(text_words.end(), 28u, build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250));
    text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250));
    const std::vector<uint8_t> bytes = make_gfx1250_code_object(text_words, kernel_name);
    ConSanOptions options;
    options.flavor = ConSanFlavor::SuperCollider;
    options.probe_lds_check_trap = true;
    options.scratch_vgpr = 8;
    options.delay_nops = 2;

    const ConSanResult result = try_patch_consan(bytes, options);

    EXPECT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
    EXPECT_TRUE(result.modified) << testing::PrintToString(result.warnings);
    EXPECT_TRUE(result.final_validation_passed);
    if (!result.modified)
      return;
    const auto rewritten_words = patched_words_at_file_offset<6>(result, 0x100);
    const auto readback = gfx1250::build_vds(load_op, {.offset1 = 8, .addr = 12, .vdst = 8});
    EXPECT_EQ(rewritten_words[4], readback[0]);
    EXPECT_EQ(rewritten_words[5], readback[1]);
  };

  check(gfx1250::kDsStore2addrStride64B32Vds, gfx1250::kDsLoad2addrStride64B32Vds, 7,
        "gfx1250_vds_store_2addr_stride64_b32");
  check(gfx1250::kDsStore2addrStride64B64Vds, gfx1250::kDsLoad2addrStride64B64Vds, 4,
        "gfx1250_vds_store_2addr_stride64_b64");
}

TEST(ConSan, ProbeLdsCheckTrapModeSpillsGfx1250B8VdsStoreScratchWindow) {
  constexpr auto store = gfx1250::build_vds(gfx1250::kDsStoreB8Vds, {.addr = 2, .data0 = 1});
  std::vector<uint32_t> text_words = {store[0], store[1]};
  text_words.insert(text_words.end(), 28u, build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250));
  for (uint16_t vgpr = 0; vgpr < 256; ++vgpr) {
    text_words.push_back(
        build_v_mov_b32_e32(vgpr, vector_source_vgpr(vgpr), ROCJITSU_CODE_ARCH_GFX1250));
  }
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250));
  const std::vector<uint8_t> bytes =
      make_gfx1250_code_object(text_words, "gfx1250_vds_store_b8_spill", /*vgpr_granulated=*/15u);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.delay_nops = 1;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  const auto patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &candidate) {
    return candidate.spilled_vgpr_count != 0;
  });
  ASSERT_NE(patch, result.patches.end());
  EXPECT_EQ(patch->spilled_vgpr_count, 2u);
  EXPECT_EQ(patch->required_private_segment_size, 8u);
  EXPECT_TRUE(result.final_validation_passed);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.kernels().size(), 1u);
  KD descriptor{};
  std::memcpy(&descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  EXPECT_EQ(descriptor.private_segment_fixed_size, 8u);
  EXPECT_EQ(
      AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT),
      1u);
}

TEST(ConSan, ProbeLdsCheckTrapModeRewritesCdna4ReadInPlace) {
  const std::array<uint32_t, 13> text_words = {
      0xD86C0004u,
      0x04000002u, // ds_read_b32 v4, v2 offset:4
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  const std::vector<uint8_t> bytes = make_cdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 6;
  options.delay_nops = 2;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineLdsLoadCheckTrap);
  EXPECT_TRUE(result.final_validation_passed);
  const auto rewritten_words = patched_words_at_file_offset<6>(result, 0x100);
  EXPECT_EQ(rewritten_words[0], 0xD86C0004u);
  EXPECT_EQ(rewritten_words[1], 0x04000002u);
  EXPECT_EQ(rewritten_words[4], 0xD86C0004u);
  EXPECT_EQ(rewritten_words[5], 0x06000002u);
  const auto all_rewritten_words = patched_words_at_file_offset<13>(result, 0x100);
  bool preserves_wave64_vcc = false;
  for (uint16_t save_sgpr = 0; save_sgpr < 106; save_sgpr += 2) {
    const auto save = build_cdna4_s_mov_b64(save_sgpr, 106, ROCJITSU_CODE_ARCH_CDNA4);
    const auto restore = build_cdna4_s_mov_b64(106, save_sgpr, ROCJITSU_CODE_ARCH_CDNA4);
    if (save && restore &&
        std::ranges::find(all_rewritten_words, *save) != all_rewritten_words.end() &&
        std::ranges::find(all_rewritten_words, *restore) != all_rewritten_words.end()) {
      preserves_wave64_vcc = true;
      break;
    }
  }
  EXPECT_TRUE(preserves_wave64_vcc);
}

TEST(ConSan, ProbeLdsCheckTrapModeComparesCdna4AccvgprB128Reads) {
  struct AccReadCase {
    uint16_t acc_base;
    std::array<uint32_t, 2> load;
  };
  constexpr std::array<AccReadCase, 2> cases = {{
      {0u, {0xDBFE0000u, 0x000000FFu}},  // ds_read_b128 a[0:3], v255
      {76u, {0xDBFE3C00u, 0x4C0000FEu}}, // ds_read_b128 a[76:79], v254 offset:15360
  }};

  for (const AccReadCase &test_case : cases) {
    SCOPED_TRACE(test_case.acc_base);
    std::vector<uint32_t> text_words(test_case.load.begin(), test_case.load.end());
    text_words.insert(text_words.end(), 64u, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
    text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4));
    const std::vector<uint8_t> bytes = make_cdna4_lds_code_object(text_words);
    ConSanOptions options;
    options.flavor = ConSanFlavor::SuperCollider;
    options.probe_lds_check_trap = true;
    options.scratch_vgpr = 20u;
    options.delay_nops = 1u;

    const ConSanResult result = try_patch_consan(bytes, options);

    ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
    ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
    ASSERT_TRUE(result.final_validation_passed);
    ASSERT_EQ(result.patches.size(), 1u);
    ASSERT_EQ(result.kernels.size(), 1u);
    ASSERT_EQ(result.kernels.front().lds_sites.size(), 1u);
    EXPECT_FALSE(result.kernels.front().lds_sites.front().dst_vgpr);
    EXPECT_EQ(result.kernels.front().lds_sites.front().dst_accvgpr, test_case.acc_base);

    AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
    ASSERT_TRUE(patched.is_valid());
    ASSERT_EQ(patched.text_sections().size(), 1u);
    const Section *text = patched.text_sections().front();
    const std::vector<uint32_t> body =
        patched_words_at_file_offset(result, text->sectionOffset(), text->size());

    const std::array<uint32_t, 2> duplicate = {
        test_case.load[0] & ~(uint32_t{1} << 25u),
        (test_case.load[1] & 0x00FFFFFFu) | (20u << 24u),
    };
    EXPECT_NE(std::search(body.begin(), body.end(), duplicate.begin(), duplicate.end()),
              body.end());
    for (uint16_t lane = 0; lane < 4u; ++lane) {
      constexpr uint16_t compare_vgpr = 24u;
      const auto acc_read =
          cdna4::build_vop3p(cdna4::kVAccvgprReadVop3p,
                             {.vdst = static_cast<uint8_t>(compare_vgpr),
                              .op_sel_hi_2 = 1u,
                              .src0 = static_cast<uint16_t>(256u + test_case.acc_base + lane),
                              .op_sel_hi = 3u});
      const std::array<uint32_t, 2> expected_acc_read = {
          0xD3D84018u,
          static_cast<uint32_t>(0x18000100u | (test_case.acc_base + lane)),
      };
      EXPECT_EQ(acc_read, expected_acc_read);
      EXPECT_NE(std::search(body.begin(), body.end(), acc_read.begin(), acc_read.end()),
                body.end());
    }
  }
}

TEST(ConSan, ProbeLdsCheckTrapModeSpillsCdna4AccvgprB128ScratchWithoutMovingBoundary) {
  constexpr std::array<uint32_t, 2> load = {
      0xDBFE3C00u,
      0x4C0000FEu, // ds_read_b128 a[76:79], v254 offset:15360
  };
  std::vector<uint32_t> text_words(load.begin(), load.end());
  text_words.insert(text_words.end(), 256u, build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  for (uint16_t vgpr = 0; vgpr < 256u; ++vgpr) {
    text_words.push_back(
        build_v_mov_b32_e32(vgpr, vector_source_vgpr(vgpr), ROCJITSU_CODE_ARCH_CDNA4));
  }
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4));
  std::vector<uint8_t> bytes = make_cdna4_lds_code_object(text_words, "cdna4_accvgpr_b128_spill",
                                                          kRdna4Wave64AllVgprsGranulated);
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 63u);
  });
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.delay_nops = 1u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.final_validation_passed);
  const auto patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &candidate) {
    return candidate.spilled_vgpr_count != 0u;
  });
  ASSERT_NE(patch, result.patches.end());
  EXPECT_EQ(patch->spilled_vgpr_count, 5u);
  EXPECT_EQ(patch->required_private_segment_size, 32u);
  EXPECT_EQ(patch->scratch_vgpr, 0u);

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
}

TEST(ConSan, ProbeLdsCheckTrapModeAlignsCdna4B32AutoReportTuple) {
  const std::array<uint32_t, 3> text_words = {
      0xD86C0004u,
      0x04000002u, // ds_read_b32 v4, v2 offset:4
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  const std::vector<uint8_t> bytes = make_cdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 6;
  options.report_buffer_address = 0x1234567887654321ull;
  options.report_marker = 0xABCDEF01u;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::LocalCaveLdsLoadCheckTrap);
  EXPECT_TRUE(result.final_validation_passed);
}

TEST(ConSan, ProbeLdsCheckTrapModeRewritesCdna4U16ReadInPlace) {
  const std::array<uint32_t, 13> text_words = {
      0xD8780004u,
      0x04000002u, // ds_read_u16 v4, v2 offset:4
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  const std::vector<uint8_t> bytes = make_cdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 6;
  options.delay_nops = 2;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineLdsLoadCheckTrap);
  EXPECT_TRUE(result.final_validation_passed);
  const auto rewritten_words = patched_words_at_file_offset<6>(result, 0x100);
  EXPECT_EQ(rewritten_words[0], 0xD8780004u);
  EXPECT_EQ(rewritten_words[1], 0x04000002u);
  EXPECT_EQ(rewritten_words[4], 0xD8780004u);
  EXPECT_EQ(rewritten_words[5], 0x06000002u);
}

TEST(ConSan, ProbeLdsCheckTrapModeRewritesCdna4TransposeRead) {
  const std::array<uint32_t, 3> text_words = {
      0xD9C60800u,
      0x0E000001u, // ds_read_b64_tr_b16 v[14:15], v1 offset:2048
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  const std::vector<uint8_t> bytes = make_cdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 20;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.patches.size(), 1u);
  const ConSanPatchInfo &patch = result.patches.front();
  EXPECT_EQ(patch.kind, ConSanPatchKind::LocalCaveLdsLoadCheckTrap);
  EXPECT_TRUE(result.final_validation_passed);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const Section *text = patched.text_sections().front();
  const auto body =
      patched_words_at_file_offset<4>(result, text->sectionOffset() + patch.trampoline_offset);
  EXPECT_EQ(body[0], text_words[0]);
  EXPECT_EQ(body[1], text_words[1]);
  EXPECT_EQ(body[2], text_words[0]);
  EXPECT_EQ(body[3], 0x14000001u);
}

TEST(ConSan, ProbeLdsCheckTrapModeRewritesCdna4Read2B64) {
  const std::array<uint32_t, 3> text_words = {
      0xD8EE0400u,
      0x0800000Bu, // ds_read2_b64 v[8:11], v11 offset1:4
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  const std::vector<uint8_t> bytes = make_cdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 20;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.patches.size(), 1u);
  const ConSanPatchInfo &patch = result.patches.front();
  EXPECT_EQ(patch.kind, ConSanPatchKind::LocalCaveLdsLoadCheckTrap);
  EXPECT_TRUE(result.final_validation_passed);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const Section *text = patched.text_sections().front();
  const auto body =
      patched_words_at_file_offset<5>(result, text->sectionOffset() + patch.trampoline_offset);
  EXPECT_EQ(body[1], text_words[0]);
  EXPECT_EQ(body[2], text_words[1]);
  EXPECT_EQ(body[3], text_words[0]);
  EXPECT_EQ(body[4], 0x14000018u); // duplicate v[20:23], preserved address v24
}

TEST(ConSan, ProbeLdsCheckTrapModeReadsBackCdna4Write2st64) {
  const std::array<uint32_t, 3> text_words = {
      0xD89E0400u,
      0x0006080Bu, // ds_write2st64_b64 v11, v[8:9], v[6:7] offset1:4
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  const std::vector<uint8_t> bytes = make_cdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 20;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.patches.size(), 1u);
  const ConSanPatchInfo &patch = result.patches.front();
  EXPECT_EQ(patch.kind, ConSanPatchKind::LocalCaveLdsStoreCheckTrap);
  EXPECT_TRUE(result.final_validation_passed);
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const Section *text = patched.text_sections().front();
  const auto body =
      patched_words_at_file_offset<4>(result, text->sectionOffset() + patch.trampoline_offset);
  EXPECT_EQ(body[0], text_words[0]);
  EXPECT_EQ(body[1], text_words[1]);
  EXPECT_EQ(body[2], 0xD8F00400u); // ds_read2st64_b64, retained offsets
  EXPECT_EQ(body[3], 0x1400000Bu); // duplicate v[20:23], address v11
}

TEST(ConSan, ProbeLdsCheckTrapModeRewritesCdna4WriteInPlace) {
  const std::array<uint32_t, 13> text_words = {
      0xD81A0004u,
      0x00000302u, // ds_write_b32 v2, v3 offset:4
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4),
  };
  const std::vector<uint8_t> bytes = make_cdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 6;
  options.delay_nops = 2;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineLdsStoreCheckTrap);
  EXPECT_TRUE(result.final_validation_passed);
  const auto rewritten_words = patched_words_at_file_offset<6>(result, 0x100);
  EXPECT_EQ(rewritten_words[0], 0xD81A0004u);
  EXPECT_EQ(rewritten_words[1], 0x00000302u);
  EXPECT_EQ(rewritten_words[4], 0xD86C0004u);
  EXPECT_EQ(rewritten_words[5], 0x06000002u);
}

TEST(ConSan, ProbeLdsCheckTrapModeHonorsExactKernelFilter) {
  const std::array<uint32_t, 13> text_words = {
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words, "selected_lds_probe");
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3;
  options.delay_nops = 2;
  options.test_kernel_name_filter = "selected_lds_probe";

  const auto selected = try_patch_consan(bytes, options);
  ASSERT_TRUE(selected.errors.empty());
  EXPECT_TRUE(selected.modified);
  ASSERT_EQ(selected.patches.size(), 1u);
  EXPECT_EQ(selected.patches.front().kind, ConSanPatchKind::InlineLdsLoadCheckTrap);

  options.test_kernel_name_filter = "different_kernel";
  const auto excluded = try_patch_consan(bytes, options);
  ASSERT_TRUE(excluded.errors.empty());
  EXPECT_FALSE(excluded.modified);
  EXPECT_TRUE(excluded.patches.empty());
}

TEST(ConSan, ProbeLdsCheckTrapAllSupportedPolicyIgnoresNominalPatchLimit) {
  const std::array<uint32_t, 23> text_words = {
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xD8D80000u,
      0x04000005u, // ds_load_b32 v4, v5
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.delay_nops = 1;
  options.max_patches = 1;
  options.max_patches_is_expert_limit = false;
  options.scratch_vgpr = 3;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 2u);
  EXPECT_EQ(result.patches[0].kind, ConSanPatchKind::InlineLdsLoadCheckTrap);
  EXPECT_EQ(result.patches[0].anchor_offset, 0u);
  EXPECT_EQ(result.patches[0].trampoline_offset, 8u);
  EXPECT_EQ(result.patches[0].original_size, 44u);
  EXPECT_EQ(result.patches[1].kind, ConSanPatchKind::InlineLdsLoadCheckTrap);
  EXPECT_EQ(result.patches[1].anchor_offset, 44u);
  EXPECT_EQ(result.patches[1].trampoline_offset, 52u);
  EXPECT_EQ(result.patches[1].original_size, 44u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  const std::array<uint32_t, 22> expected_words = {
      0xD8D80000u,
      0x01000002u, // original ds_load_b32 v1, v2
      0xBF800000u, // delay
      0xD8D80000u,
      0x03000002u, // duplicate ds_load_b32 v3, v2
      0xBFC60000u, // s_wait_dscnt 0
      build_s_mov_b32(0, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0x7C9A0701u, // v_cmp_ne_u32_e32 vcc_lo, v1, v3
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      build_s_mov_b32(kRdna4VccLo, 0, ROCJITSU_CODE_ARCH_RDNA4),
      0xD8D80000u,
      0x04000005u, // original ds_load_b32 v4, v5
      0xBF800000u, // delay
      0xD8D80000u,
      0x03000005u, // duplicate ds_load_b32 v3, v5
      0xBFC60000u, // s_wait_dscnt 0
      build_s_mov_b32(0, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0x7C9A0704u, // v_cmp_ne_u32_e32 vcc_lo, v4, v3
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      build_s_mov_b32(kRdna4VccLo, 0, ROCJITSU_CODE_ARCH_RDNA4),
  };
  const auto rewritten_words = patched_words_at_file_offset<expected_words.size()>(result, 0x100);
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(ConSan, ProbeLdsCheckTrapModeRewritesPaddedU16D16LoadInPlace) {
  const std::array<uint32_t, 14> text_words = {
      0xDA980000u,
      0x01000002u, // ds_load_u16_d16 v1, v2
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.delay_nops = 1;
  options.scratch_vgpr = 3;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineLdsLoadCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 8u);
  EXPECT_EQ(result.patches.front().original_size, 52u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  const std::array<uint32_t, 14> expected_words = {
      0xDA980000u,
      0x01000002u, // original ds_load_u16_d16 v1, v2
      0xBFC60000u, // s_wait_dscnt 0 for original d16 load
      build_v_mov_b32_e32(3, vector_source_vgpr(1), ROCJITSU_CODE_ARCH_RDNA4),
      0xBF800000u, // delay
      0xDA980000u,
      0x03000002u, // duplicate ds_load_u16_d16 v3, v2
      0xBFC60000u, // s_wait_dscnt 0
      build_s_mov_b32(0, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0x7C9A0701u, // v_cmp_ne_u32_e32 vcc_lo, v1, v3
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      build_s_mov_b32(kRdna4VccLo, 0, ROCJITSU_CODE_ARCH_RDNA4),
      0xBFB00000u, // original s_endpgm after padding
  };
  const auto rewritten_words = patched_words_at_file_offset<expected_words.size()>(result, 0x100);
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(ConSan, ProbeLdsCheckTrapModeRewritesPaddedU16D16HiLoadInPlace) {
  const std::array<uint32_t, 14> text_words = {
      0xDA9C0000u,
      0x01000002u, // ds_load_u16_d16_hi v1, v2
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.delay_nops = 1;
  options.scratch_vgpr = 3;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineLdsLoadCheckTrap);
  EXPECT_EQ(result.patches.front().original_size, 52u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  const std::array<uint32_t, 14> expected_words = {
      0xDA9C0000u,
      0x01000002u, // original ds_load_u16_d16_hi v1, v2
      0xBFC60000u, // s_wait_dscnt 0 for original d16 load
      build_v_mov_b32_e32(3, vector_source_vgpr(1), ROCJITSU_CODE_ARCH_RDNA4),
      0xBF800000u, // delay
      0xDA9C0000u,
      0x03000002u, // duplicate ds_load_u16_d16_hi v3, v2
      0xBFC60000u, // s_wait_dscnt 0
      build_s_mov_b32(0, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0x7C9A0701u, // v_cmp_ne_u32_e32 vcc_lo, v1, v3
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      build_s_mov_b32(kRdna4VccLo, 0, ROCJITSU_CODE_ARCH_RDNA4),
      0xBFB00000u, // original s_endpgm after padding
  };
  const auto rewritten_words = patched_words_at_file_offset<expected_words.size()>(result, 0x100);
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(ConSan, ProbeLdsCheckTrapModeCanUseSleepDelay) {
  const std::array<uint32_t, 12> text_words = {
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3;
  options.delay_mode = ConSanDelayMode::Sleep;
  options.delay_nops = 7;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineLdsLoadCheckTrap);
  EXPECT_EQ(result.patches.front().original_size, 44u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  const std::array<uint32_t, 12> expected_words = {
      0xD8D80000u,
      0x01000002u, // original ds_load_b32 v1, v2
      build_s_sleep(7, ROCJITSU_CODE_ARCH_RDNA4),
      0xD8D80000u,
      0x03000002u, // duplicate ds_load_b32 v3, v2
      0xBFC60000u, // s_wait_dscnt 0
      build_s_mov_b32(0, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0x7C9A0701u, // v_cmp_ne_u32_e32 vcc_lo, v1, v3
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      build_s_mov_b32(kRdna4VccLo, 0, ROCJITSU_CODE_ARCH_RDNA4),
      0xBFB00000u, // original s_endpgm after padding
  };
  const auto rewritten_words = patched_words_at_file_offset<expected_words.size()>(result, 0x100);
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(ConSan, ProbeLdsCheckTrapModeCanUseSleepVarDelay) {
  const std::array<uint32_t, 12> text_words = {
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3;
  options.delay_mode = ConSanDelayMode::SleepVar;
  options.delay_nops = 1;
  options.delay_var_ssrc = kRdna4VccLo;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineLdsLoadCheckTrap);
  EXPECT_EQ(result.patches.front().original_size, 44u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  const std::array<uint32_t, 12> expected_words = {
      0xD8D80000u,
      0x01000002u, // original ds_load_b32 v1, v2
      build_s_sleep_var(kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0xD8D80000u,
      0x03000002u, // duplicate ds_load_b32 v3, v2
      0xBFC60000u, // s_wait_dscnt 0
      build_s_mov_b32(0, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0x7C9A0701u, // v_cmp_ne_u32_e32 vcc_lo, v1, v3
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      build_s_mov_b32(kRdna4VccLo, 0, ROCJITSU_CODE_ARCH_RDNA4),
      0xBFB00000u, // original s_endpgm after padding
  };
  const auto rewritten_words = patched_words_at_file_offset<expected_words.size()>(result, 0x100);
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(ConSan, ProbeLdsCheckTrapModeRejectsOversizedSleepDelay) {
  const std::array<uint32_t, 4> text_words = {
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.delay_mode = ConSanDelayMode::Sleep;
  options.delay_nops = 65536;

  const auto result = try_patch_consan(bytes, options);

  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
  ASSERT_FALSE(result.errors.empty());
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::Invalid);
  EXPECT_NE(result.errors.front().find("16-bit s_sleep"), std::string::npos);
}

TEST(ConSan, ProbeLdsCheckTrapModeRewritesPaddedStoreInPlace) {
  const std::array<uint32_t, 13> text_words = {
      0xD8340000u,
      0x00000102u, // ds_store_b32 v2, v1
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3;
  options.delay_nops = 2;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineLdsStoreCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 8u);
  EXPECT_EQ(result.patches.front().original_size, 48u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 3u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  const std::array<uint32_t, 13> expected_words = {
      0xD8340000u,
      0x00000102u, // original ds_store_b32 v2, v1
      0xBF800000u,
      0xBF800000u, // delay
      0xD8D80000u,
      0x03000002u, // readback ds_load_b32 v3, v2
      0xBFC60000u, // s_wait_dscnt 0
      build_s_mov_b32(0, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0x7C9A0701u, // v_cmp_ne_u32_e32 vcc_lo, v1, v3
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      build_s_mov_b32(kRdna4VccLo, 0, ROCJITSU_CODE_ARCH_RDNA4),
      0xBFB00000u, // original s_endpgm after padding
  };
  const auto rewritten_words = patched_words_at_file_offset<expected_words.size()>(result, 0x100);
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(ConSan, ProbeLdsCheckTrapModeCanReportMismatchToMarkerBuffer) {
  const std::array<uint32_t, 24> text_words = {
      0xD8340000u,
      0x00000102u, // ds_store_b32 v2, v1
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3;
  options.report_buffer_address = 0x1234567887654321ull;
  options.report_marker = 0xABCDEF01u;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineLdsStoreCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 8u);
  EXPECT_EQ(result.patches.front().original_size, 84u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 3u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  const auto mov_report_lo = build_v_mov_b32_e64_literal(4, 0x87654321u, ROCJITSU_CODE_ARCH_RDNA4);
  const auto mov_report_hi = build_v_mov_b32_e64_literal(5, 0x12345678u, ROCJITSU_CODE_ARCH_RDNA4);
  const auto mov_marker = build_v_mov_b32_e64_literal(6, 0xABCDEF01u, ROCJITSU_CODE_ARCH_RDNA4);
  const auto store_marker = build_flat_store_b32_vaddr_vsrc(4, 6, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(mov_report_lo);
  ASSERT_TRUE(mov_report_hi);
  ASSERT_TRUE(mov_marker);
  ASSERT_TRUE(store_marker);

  const std::array<uint32_t, 24> expected_words = {
      0xD8340000u,
      0x00000102u, // original ds_store_b32 v2, v1
      0xD8D80000u,
      0x03000002u, // readback ds_load_b32 v3, v2
      0xBFC60000u, // s_wait_dscnt 0
      build_s_mov_b32(0, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0x7C9A0701u, // v_cmp_ne_u32_e32 vcc_lo, v1, v3
      0xBFA3000Cu, // s_cbranch_vccz +12, skipping marker store when equal
      (*mov_report_lo)[0],
      (*mov_report_lo)[1],
      (*mov_report_lo)[2],
      (*mov_report_hi)[0],
      (*mov_report_hi)[1],
      (*mov_report_hi)[2],
      (*mov_marker)[0],
      (*mov_marker)[1],
      (*mov_marker)[2],
      (*store_marker)[0],
      (*store_marker)[1],
      (*store_marker)[2],
      build_s_mov_b32(kRdna4VccLo, 0, ROCJITSU_CODE_ARCH_RDNA4),
      0xBF800000u,
      0xBF800000u,
      0xBFB00000u, // original s_endpgm after padding
  };
  const auto rewritten_words = patched_words_at_file_offset<expected_words.size()>(result, 0x100);
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(ConSan, ProbeLdsCheckTrapModeRewritesPaddedB64LoadInPlace) {
  const std::array<uint32_t, 16> text_words = {
      0xD9D80000u,
      0x01000009u, // ds_load_b64 v[1:2], v9
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 5;
  options.delay_nops = 2;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineLdsLoadCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 8u);
  EXPECT_EQ(result.patches.front().original_size, 60u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 5u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  const std::array<uint32_t, 16> expected_words = {
      0xD9D80000u,
      0x01000009u, // original ds_load_b64 v[1:2], v9
      0xBF800000u,
      0xBF800000u, // delay
      0xD9D80000u,
      0x05000009u, // duplicate ds_load_b64 v[5:6], v9
      0xBFC60000u, // s_wait_dscnt 0
      build_s_mov_b32(0, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0x7C9A0B01u, // v_cmp_ne_u32_e32 vcc_lo, v1, v5
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      0x7C9A0D02u, // v_cmp_ne_u32_e32 vcc_lo, v2, v6
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      build_s_mov_b32(kRdna4VccLo, 0, ROCJITSU_CODE_ARCH_RDNA4),
      0xBFB00000u, // original s_endpgm after padding
  };
  const auto rewritten_words = patched_words_at_file_offset<expected_words.size()>(result, 0x100);
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(ConSan, ProbeLdsCheckTrapModeRewritesPaddedB128StoreInPlace) {
  const std::array<uint32_t, 21> text_words = {
      0xDB7C0000u,
      0x00000109u, // ds_store_b128 v9, v[1:4]
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 5;
  options.delay_nops = 1;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineLdsStoreCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 8u);
  EXPECT_EQ(result.patches.front().original_size, 80u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 5u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  const std::array<uint32_t, 21> expected_words = {
      0xDB7C0000u,
      0x00000109u, // original ds_store_b128 v9, v[1:4]
      0xBF800000u, // delay
      0xDBFC0000u,
      0x05000009u, // readback ds_load_b128 v[5:8], v9
      0xBFC60000u, // s_wait_dscnt 0
      build_s_mov_b32(0, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0x7C9A0B01u, // v_cmp_ne_u32_e32 vcc_lo, v1, v5
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      0x7C9A0D02u, // v_cmp_ne_u32_e32 vcc_lo, v2, v6
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      0x7C9A0F03u, // v_cmp_ne_u32_e32 vcc_lo, v3, v7
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      0x7C9A1104u, // v_cmp_ne_u32_e32 vcc_lo, v4, v8
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      build_s_mov_b32(kRdna4VccLo, 0, ROCJITSU_CODE_ARCH_RDNA4),
      0xBFB00000u, // original s_endpgm after padding
  };
  const auto rewritten_words = patched_words_at_file_offset<expected_words.size()>(result, 0x100);
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(ConSan, ProbeLdsCheckTrapModeAutoScratchUsesLiveness) {
  const std::array<uint32_t, 14> text_words = {
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0x06080603u, // v_add_f32_e32 v4, v3, v3
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.delay_nops = 2;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineLdsLoadCheckTrap);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 5u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  const std::array<uint32_t, 14> expected_words = {
      0xD8D80000u,
      0x01000002u, // original ds_load_b32 v1, v2
      0xBF800000u,
      0xBF800000u, // delay
      0xD8D80000u,
      0x05000002u, // duplicate ds_load_b32 v5, v2
      0xBFC60000u, // s_wait_dscnt 0
      build_s_mov_b32(4, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0x7C9A0B01u, // v_cmp_ne_u32_e32 vcc_lo, v1, v5
      0xBFA30001u, // s_cbranch_vccz +1, skipping trap when equal
      0xBF900000u, // s_trap 0
      build_s_mov_b32(kRdna4VccLo, 4, ROCJITSU_CODE_ARCH_RDNA4),
      0x06080603u, // original v_add_f32_e32 after padding
      0xBFB00000u, // original s_endpgm
  };
  const auto rewritten_words = patched_words_at_file_offset<expected_words.size()>(result, 0x100);
  EXPECT_EQ(rewritten_words, expected_words);
}

TEST(ConSan, ProbeLdsCheckTrapModeCanGrowDescriptorForAutoScratch) {
  const std::array<uint32_t, 14> text_words = {
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xD8340000u,
      0x00000302u, // ds_store_b32 v2, v3
      0xBFB00000u, // s_endpgm
  };
  const uint32_t four_vgprs_granulated = 0;
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "lds_probe", four_vgprs_granulated);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.delay_nops = 1;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineLdsLoadCheckTrap);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 4u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  const uint64_t descriptor_offset = 0x100 + text_words.size() * sizeof(uint32_t);
  KD descriptor{};
  std::memcpy(&descriptor, result.elf_bytes.data() + descriptor_offset, sizeof(descriptor));
  const uint32_t granulated = AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc1,
                                              kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
  EXPECT_EQ(granulated, 1u);
}

TEST(ConSan, ProbeLdsCheckTrapModeCanGrowDescriptorForB64ScratchHeadroom) {
  const std::array<uint32_t, 4> text_words = {
      0xD9D80000u,
      0x01000009u, // ds_load_b64 v[1:2], v9
      build_v_mov_b32_e32(13, vector_source_vgpr(13), ROCJITSU_CODE_ARCH_RDNA4),
      0xBFB00000u, // s_endpgm
  };
  const uint32_t sixteen_vgprs_granulated = 3;
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "lds_probe", sixteen_vgprs_granulated);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.delay_nops = 1;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::LocalCaveLdsLoadCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().trampoline_offset, text_words.size() * sizeof(uint32_t));
  EXPECT_EQ(result.patches.front().original_size, 8u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 14u);
  EXPECT_GT(result.elf_bytes.size(), bytes.size());

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.kernels().size(), 1u);
  KD descriptor{};
  std::memcpy(&descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  const uint32_t granulated = AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc1,
                                              kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
  EXPECT_EQ(granulated, 4u);
}

TEST(ConSan, ProbeLdsCheckTrapModePreservesOverwrittenTwoAddressLoadAddress) {
  const std::array<uint32_t, 3> text_words = {
      0xD9DC0100u,
      0x00000000u, // ds_load_2addr_b64 v[0:3], v0 offset1:1
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "lds_probe", /*granulated_vgpr_count=*/3);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 8;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  ASSERT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  const ConSanPatchInfo &patch = result.patches.front();
  EXPECT_EQ(patch.kind, ConSanPatchKind::LocalCaveLdsLoadCheckTrap);
  EXPECT_EQ(patch.anchor_offset, 0u);
  EXPECT_EQ(patch.original_size, 8u);
  EXPECT_EQ(patch.trampoline_size, 84u);

  std::array<uint32_t, 5> body_prefix{};
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const Section *text = patched.text_sections().front();
  std::memcpy(body_prefix.data(),
              result.elf_bytes.data() + text->sectionOffset() + patch.trampoline_offset,
              sizeof(body_prefix));
  EXPECT_EQ(body_prefix[0],
            build_v_mov_b32_e32(12, vector_source_vgpr(0), ROCJITSU_CODE_ARCH_RDNA4));
  EXPECT_EQ(body_prefix[1], text_words[0]);
  EXPECT_EQ(body_prefix[2], text_words[1]);
  EXPECT_EQ(body_prefix[3], text_words[0]);
  EXPECT_EQ(body_prefix[4], 0x0800000Cu); // duplicate result v8, preserved address v12
}

TEST(ConSan, ProbeLdsCheckTrapModeReadsBackTwoAddressStore) {
  const std::array<uint32_t, 3> text_words = {
      0xD8382200u,
      0x00010005u, // ds_store_2addr_b32 v5, v0, v1 offset1:34
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "lds_probe", /*granulated_vgpr_count=*/3);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 8;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  ASSERT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  const ConSanPatchInfo &patch = result.patches.front();
  EXPECT_EQ(patch.kind, ConSanPatchKind::LocalCaveLdsStoreCheckTrap);
  EXPECT_EQ(patch.anchor_offset, 0u);
  EXPECT_EQ(patch.original_size, 8u);
  EXPECT_EQ(patch.trampoline_size, 56u);

  std::array<uint32_t, 4> body_prefix{};
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const Section *text = patched.text_sections().front();
  std::memcpy(body_prefix.data(),
              result.elf_bytes.data() + text->sectionOffset() + patch.trampoline_offset,
              sizeof(body_prefix));
  EXPECT_EQ(body_prefix[0], text_words[0]);
  EXPECT_EQ(body_prefix[1], text_words[1]);
  EXPECT_EQ(body_prefix[2], 0xD8DC2200u); // ds_load_2addr_b32, retained offsets
  EXPECT_EQ(body_prefix[3], 0x08000005u); // duplicate result v[8:9], address v5
}

TEST(ConSan, ProbeLdsCheckTrapModePrefersDescriptorCoveredCandidate) {
  const std::array<uint32_t, 15> text_words = {
      0xD9D80000u,
      0x01000009u, // ds_load_b64 v[1:2], v9; would need descriptor growth
      build_v_mov_b32_e32(13, vector_source_vgpr(13), ROCJITSU_CODE_ARCH_RDNA4),
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2; can use descriptor-covered v14
      0xBF800000u,
      0xBF800000u,
      0xBF800000u,
      0xBF800000u,
      0xBF800000u,
      0xBF800000u,
      0xBF800000u,
      0xBF800000u,
      0xBF800000u,
      0xBFB00000u, // s_endpgm
  };
  const uint32_t sixteen_vgprs_granulated = 3;
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "lds_probe", sixteen_vgprs_granulated);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.delay_nops = 1;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineLdsLoadCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 3u * sizeof(uint32_t));
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 14u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());

  const uint64_t descriptor_offset = 0x100 + text_words.size() * sizeof(uint32_t);
  KD descriptor{};
  std::memcpy(&descriptor, result.elf_bytes.data() + descriptor_offset, sizeof(descriptor));
  const uint32_t granulated = AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc1,
                                              kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
  EXPECT_EQ(granulated, sixteen_vgprs_granulated);
}

TEST(ConSan, ProbeLdsCheckTrapModeRejectsLocalCaveOwnedByAnotherFunction) {
  const std::array<uint32_t, 3> kernel_words = {
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 1> function_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 12> tail_words = {
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words, tail_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3;
  options.delay_nops = 1;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::LocalCaveLdsLoadCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 64u);
  EXPECT_EQ(result.patches.front().original_size, 8u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 3u);
  ASSERT_GT(result.elf_bytes.size(), bytes.size());
  EXPECT_NE(result.elf_bytes, bytes);

  const std::array<uint32_t, 2> expected_anchor = {
      build_s_branch(15, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
  };
  std::array<uint32_t, expected_anchor.size()> anchor_words{};
  std::memcpy(anchor_words.data(), result.elf_bytes.data() + 0x100,
              anchor_words.size() * sizeof(uint32_t));
  EXPECT_EQ(anchor_words, expected_anchor);

  const std::array<uint32_t, 12> expected_cave = {
      0xD8D80000u,
      0x01000002u, // original ds_load_b32 v1, v2
      0xBF800000u, // delay
      0xD8D80000u,
      0x03000002u, // duplicate ds_load_b32 v3, v2
      0xBFC60000u, // s_wait_dscnt 0
      build_s_mov_b32(0, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0x7C9A0701u, // v_cmp_ne_u32_e32 vcc_lo, v1, v3
      0xBFA30001u, // s_cbranch_vccz +1
      0xBF900000u, // s_trap 0
      build_s_mov_b32(kRdna4VccLo, 0, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_branch(-26, ROCJITSU_CODE_ARCH_RDNA4),
  };
  const auto cave_words = patched_words_at_file_offset<expected_cave.size()>(
      result, 0x100 + result.patches.front().trampoline_offset);
  EXPECT_EQ(cave_words, expected_cave);
}

TEST(ConSan, ProbeLdsCheckTrapModeDoesNotDecodeNonSymbolTextPadding) {
  const std::array<uint32_t, 3> kernel_words = {
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 1> function_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 13> tail_words = {
      0xFFFFFFFFu, // non-instruction data/alignment outside every function symbol
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words, tail_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3;
  options.delay_nops = 1;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::LocalCaveLdsLoadCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 68u);
}

TEST(ConSan, ProbeLdsCheckTrapModeLeavesAdjacentAtomicAndBarrierUntouched) {
  const std::array<uint32_t, 7> kernel_words = {
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xD8000000u,
      0x00000302u, // ds_add_u32 v2, v3
      0xBE804EC1u, // s_barrier_signal -1
      0xBF94FFFFu, // s_barrier_wait -1
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 1> function_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 12> tail_words = {
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words, tail_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 4;
  options.delay_nops = 1;
  // Match an ordinary runtime clean transform: fault discovery is a separate
  // validation phase and barrier-move destinations were not requested.
  options.collect_barrier_move_destinations = false;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::LocalCaveLdsLoadCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().original_size, 8u);
  EXPECT_TRUE(result.fault_sites.empty());
  EXPECT_TRUE(result.sync_events.empty());
  EXPECT_TRUE(result.sync_sequences.empty());
  constexpr uint64_t adjacent_file_offset = 0x108u;
  constexpr uint64_t adjacent_size = 5u * sizeof(uint32_t);
  EXPECT_EQ(0, std::memcmp(result.elf_bytes.data() + adjacent_file_offset,
                           bytes.data() + adjacent_file_offset, adjacent_size));
}

TEST(ConSan, ProbeLdsCheckTrapModeSelectsMultipleLocalCavesOwnedByKernel) {
  const std::array<uint32_t, 1> kernel_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 5> function_words = {
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xD8D80000u,
      0x04000005u, // ds_load_b32 v4, v5
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 25> tail_words = {
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBFB00000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
  };
  const std::vector<uint8_t> bytes = make_rdna4_code_object_with_local_function(
      kernel_words, function_words, tail_words, kRdna4Wave64AllVgprsGranulated,
      /*function_is_kernel=*/true);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.max_patches = 2;
  options.scratch_vgpr = 6;
  options.delay_nops = 1;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 2u);
  EXPECT_EQ(result.patches[0].kind, ConSanPatchKind::LocalCaveLdsLoadCheckTrap);
  EXPECT_EQ(result.patches[0].anchor_offset, 4u);
  EXPECT_EQ(result.patches[0].trampoline_offset, 24u);
  EXPECT_EQ(result.patches[0].original_size, 8u);
  EXPECT_EQ(result.patches[1].kind, ConSanPatchKind::LocalCaveLdsLoadCheckTrap);
  EXPECT_EQ(result.patches[1].anchor_offset, 12u);
  EXPECT_EQ(result.patches[1].trampoline_offset, 76u);
  EXPECT_EQ(result.patches[1].original_size, 8u);
}

TEST(ConSan, ProbeLdsCheckTrapModeRejectsCrossKernelLocalCave) {
  const std::array<uint32_t, 3> first_kernel_words = {
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 1> second_kernel_words = {
      0xBFB00000u, // s_endpgm
  };
  std::array<uint32_t, 12> second_kernel_padding{};
  second_kernel_padding.fill(build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  const std::vector<uint8_t> bytes = make_rdna4_code_object_with_local_function(
      first_kernel_words, second_kernel_words, second_kernel_padding,
      kRdna4Wave64AllVgprsGranulated, /*function_is_kernel=*/true);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3;
  options.delay_nops = 1;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  // The padding at 16 belongs to the second kernel. The first kernel must use
  // a fresh appended cave rather than branch into another kernel's text.
  EXPECT_EQ(result.patches.front().trampoline_offset, 64u);
  EXPECT_GT(result.elf_bytes.size(), bytes.size());
}

TEST(ConSan, ProbeLdsCheckTrapModeUsesReachableUncoveredNopCaveFor2addrB64Load) {
  const std::array<uint32_t, 3> kernel_words = {
      0xD9DC0000u,
      0x01000009u, // ds_load_2addr_b64 v[1:4], v9
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 1> function_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 21> tail_words = {
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words, tail_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 5;
  options.delay_nops = 1;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::LocalCaveLdsLoadCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 100u);
  EXPECT_EQ(result.patches.front().original_size, 8u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 5u);
  ASSERT_GT(result.elf_bytes.size(), bytes.size());
  EXPECT_NE(result.elf_bytes, bytes);

  const std::array<uint32_t, 2> expected_anchor = {
      build_s_branch(24, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
  };
  std::array<uint32_t, expected_anchor.size()> anchor_words{};
  std::memcpy(anchor_words.data(), result.elf_bytes.data() + 0x100,
              anchor_words.size() * sizeof(uint32_t));
  EXPECT_EQ(anchor_words, expected_anchor);

  const std::array<uint32_t, 21> expected_cave = {
      0xD9DC0000u,
      0x01000009u, // original ds_load_2addr_b64 v[1:4], v9
      0xBF800000u, // delay
      0xD9DC0000u,
      0x05000009u, // duplicate ds_load_2addr_b64 v[5:8], v9
      0xBFC60000u, // s_wait_dscnt 0
      build_s_mov_b32(0, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0x7C9A0B01u, // v_cmp_ne_u32_e32 vcc_lo, v1, v5
      0xBFA30001u, // s_cbranch_vccz +1
      0xBF900000u, // s_trap 0
      0x7C9A0D02u, // v_cmp_ne_u32_e32 vcc_lo, v2, v6
      0xBFA30001u, // s_cbranch_vccz +1
      0xBF900000u, // s_trap 0
      0x7C9A0F03u, // v_cmp_ne_u32_e32 vcc_lo, v3, v7
      0xBFA30001u, // s_cbranch_vccz +1
      0xBF900000u, // s_trap 0
      0x7C9A1104u, // v_cmp_ne_u32_e32 vcc_lo, v4, v8
      0xBFA30001u, // s_cbranch_vccz +1
      0xBF900000u, // s_trap 0
      build_s_mov_b32(kRdna4VccLo, 0, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_branch(-44, ROCJITSU_CODE_ARCH_RDNA4),
  };
  const auto cave_words = patched_words_at_file_offset<expected_cave.size()>(
      result, 0x100 + result.patches.front().trampoline_offset);
  EXPECT_EQ(cave_words, expected_cave);
}

TEST(ConSan, ProbeLdsCheckTrapModeUsesReachableUncoveredNopCaveForB128Store) {
  const std::array<uint32_t, 3> kernel_words = {
      0xDB7C0000u,
      0x00000109u, // ds_store_b128 v9, v[1:4]
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 1> function_words = {
      0xBFB00000u, // s_endpgm
  };
  const std::array<uint32_t, 21> tail_words = {
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words, tail_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 5;
  options.delay_nops = 1;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::LocalCaveLdsStoreCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 100u);
  EXPECT_EQ(result.patches.front().original_size, 8u);
  ASSERT_TRUE(result.patches.front().scratch_vgpr);
  EXPECT_EQ(*result.patches.front().scratch_vgpr, 5u);
  ASSERT_GT(result.elf_bytes.size(), bytes.size());
  EXPECT_NE(result.elf_bytes, bytes);

  const std::array<uint32_t, 2> expected_anchor = {
      build_s_branch(24, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
  };
  std::array<uint32_t, expected_anchor.size()> anchor_words{};
  std::memcpy(anchor_words.data(), result.elf_bytes.data() + 0x100,
              anchor_words.size() * sizeof(uint32_t));
  EXPECT_EQ(anchor_words, expected_anchor);

  const std::array<uint32_t, 21> expected_cave = {
      0xDB7C0000u,
      0x00000109u, // original ds_store_b128 v9, v[1:4]
      0xBF800000u, // delay
      0xDBFC0000u,
      0x05000009u, // readback ds_load_b128 v[5:8], v9
      0xBFC60000u, // s_wait_dscnt 0
      build_s_mov_b32(0, kRdna4VccLo, ROCJITSU_CODE_ARCH_RDNA4),
      0x7C9A0B01u, // v_cmp_ne_u32_e32 vcc_lo, v1, v5
      0xBFA30001u, // s_cbranch_vccz +1
      0xBF900000u, // s_trap 0
      0x7C9A0D02u, // v_cmp_ne_u32_e32 vcc_lo, v2, v6
      0xBFA30001u, // s_cbranch_vccz +1
      0xBF900000u, // s_trap 0
      0x7C9A0F03u, // v_cmp_ne_u32_e32 vcc_lo, v3, v7
      0xBFA30001u, // s_cbranch_vccz +1
      0xBF900000u, // s_trap 0
      0x7C9A1104u, // v_cmp_ne_u32_e32 vcc_lo, v4, v8
      0xBFA30001u, // s_cbranch_vccz +1
      0xBF900000u, // s_trap 0
      build_s_mov_b32(kRdna4VccLo, 0, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_branch(-44, ROCJITSU_CODE_ARCH_RDNA4),
  };
  const auto cave_words = patched_words_at_file_offset<expected_cave.size()>(
      result, 0x100 + result.patches.front().trampoline_offset);
  EXPECT_EQ(cave_words, expected_cave);
}

TEST(ConSan, ProbeLdsCheckTrapModeUsesAppendedTextCaveWhenNoLocalCaveFits) {
  const std::array<uint32_t, 3> text_words = {
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::LocalCaveLdsLoadCheckTrap);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 12u);
  EXPECT_EQ(result.patches.front().original_size, 8u);
  EXPECT_GT(result.elf_bytes.size(), bytes.size());

  const std::array<uint32_t, 2> expected_anchor = {
      build_s_branch(2, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
  };
  std::array<uint32_t, expected_anchor.size()> anchor_words{};
  std::memcpy(anchor_words.data(), result.elf_bytes.data() + 0x100,
              anchor_words.size() * sizeof(uint32_t));
  EXPECT_EQ(anchor_words, expected_anchor);
}

TEST(ConSan, ProbeLdsCheckTrapModeUsesIndirectIslandForLargeAppendedTextCave) {
  constexpr size_t kLargeTextWords = 33000u;
  std::vector<uint32_t> text_words;
  text_words.reserve(kLargeTextWords);
  text_words.push_back(0xD8D80000u);
  text_words.push_back(0x01000002u); // ds_load_b32 v1, v2
  text_words.push_back(0xBFB00000u); // s_endpgm
  for (size_t i = 0; i < 7u; ++i)
    text_words.push_back(build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words.resize(kLargeTextWords - 1u, build_s_mov_b32(100, 100, ROCJITSU_CODE_ARCH_RDNA4));
  text_words.push_back(0xBFB00000u); // s_endpgm

  const uint64_t original_text_size = text_words.size() * sizeof(uint32_t);
  ASSERT_FALSE(compute_sopp_branch_simm16(0u, original_text_size));
  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  // The seven NOPs are linker-style padding owned by this kernel, not
  // executable kernel text, so they are eligible as a local entry island.
  mutate_elf_symbol(bytes, 1, [](Elf64_Sym &symbol) { symbol.st_size = 3u * sizeof(uint32_t); });
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.modified);
  const auto island = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineScIndirectBranchIsland, &ConSanPatchInfo::kind);
  const auto body = std::ranges::find(result.patches, ConSanPatchKind::LocalCaveLdsLoadCheckTrap,
                                      &ConSanPatchInfo::kind);
  ASSERT_NE(island, result.patches.end());
  ASSERT_NE(body, result.patches.end());
  EXPECT_EQ(island->anchor_offset, 0u);
  EXPECT_EQ(island->trampoline_offset, 3u * sizeof(uint32_t));
  EXPECT_EQ(island->trampoline_size, 7u * sizeof(uint32_t));
  EXPECT_EQ(body->anchor_offset, 0u);
  EXPECT_EQ(body->trampoline_offset, original_text_size);
  EXPECT_GT(body->indirect_required_sgpr_count, 0u);
  ASSERT_EQ(body->owner_descriptor_file_offsets.size(), 1u);
  ASSERT_TRUE(body->indirect_pc_sgpr.has_value());
  ASSERT_TRUE(body->indirect_saved_scc_sgpr.has_value());
  ASSERT_TRUE(body->indirect_saved_vcc_sgpr.has_value());
  ASSERT_TRUE(body->indirect_return_offset.has_value());
  EXPECT_NE(*body->indirect_saved_scc_sgpr, *body->indirect_saved_vcc_sgpr);
  EXPECT_TRUE(*body->indirect_saved_scc_sgpr < *body->indirect_pc_sgpr ||
              *body->indirect_saved_scc_sgpr > *body->indirect_pc_sgpr + 1u);
  EXPECT_FALSE(compute_sopp_branch_simm16(body->trampoline_offset + body->trampoline_size -
                                              7u * sizeof(uint32_t),
                                          body->anchor_offset + body->original_size));

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  EXPECT_GT(patched.text_sections().front()->size(), original_text_size);
  ASSERT_EQ(patched.kernels().size(), 1u);
  KD descriptor{};
  std::memcpy(&descriptor,
              result.elf_bytes.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  const uint32_t sgpr_granulated = AMDHSA_BITS_GET(
      descriptor.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT);
  EXPECT_GE((sgpr_granulated + 1u) * 8u, body->indirect_required_sgpr_count);
  EXPECT_TRUE(result.final_validation_passed);

  ConSanResult corrupted = result;
  const uint64_t text_file_offset = patched.text_sections().front()->sectionOffset();
  const uint32_t nop = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
  std::memcpy(corrupted.elf_bytes.data() + text_file_offset + *body->indirect_return_offset +
                  5u * sizeof(uint32_t),
              &nop, sizeof(nop));
  const auto corrupted_errors = validate_consan_modified_elf(bytes, corrupted);
  EXPECT_TRUE(std::ranges::any_of(corrupted_errors, [](const std::string &error) {
    return error.find("indirect-island proof found corrupted SCC preservation") !=
           std::string::npos;
  }));
}

TEST(ConSan, ProbeLdsCheckTrapModePartitionsLongLocalCaveIntoEntryIslands) {
  constexpr size_t kLargeTextWords = 33000u;
  std::vector<uint32_t> text_words = {
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xD8D80000u,
      0x04000005u, // ds_load_b32 v4, v5
      0xBFB00000u, // s_endpgm
  };
  for (size_t i = 0; i < 21u; ++i)
    text_words.push_back(build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4));
  text_words.resize(kLargeTextWords - 1u, build_s_mov_b32(100, 100, ROCJITSU_CODE_ARCH_RDNA4));
  text_words.push_back(0xBFB00000u); // s_endpgm

  const uint64_t original_text_size = text_words.size() * sizeof(uint32_t);
  ASSERT_FALSE(compute_sopp_branch_simm16(0u, original_text_size));
  std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  mutate_elf_symbol(bytes, 1, [](Elf64_Sym &symbol) { symbol.st_size = 5u * sizeof(uint32_t); });
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.max_patches = 2;
  options.scratch_vgpr = 6;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.modified);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::LocalCaveLdsLoadCheckTrap,
                               &ConSanPatchInfo::kind),
            2u);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineScIndirectBranchIsland,
                               &ConSanPatchInfo::kind),
            2u);
  std::vector<uint64_t> island_offsets;
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.kind == ConSanPatchKind::TrampolineScIndirectBranchIsland)
      island_offsets.push_back(patch.trampoline_offset);
  }
  std::ranges::sort(island_offsets);
  EXPECT_EQ(island_offsets, (std::vector<uint64_t>{5u * sizeof(uint32_t), 12u * sizeof(uint32_t)}));
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
}

TEST(ConSan, Gfx1250DenseCheckTrapUsesExplicitKeysAtScalarLimit) {
  constexpr size_t kSiteCount = 1025u;
  constexpr size_t kTextWords = 66000u;
  constexpr auto load = gfx1250::build_vds(gfx1250::kDsLoadB32Vds, {.addr = 2, .vdst = 1});
  std::vector<uint32_t> text_words(kTextWords, build_s_mov_b32(97, 97, ROCJITSU_CODE_ARCH_GFX1250));
  for (size_t site = 0; site < kSiteCount; ++site) {
    text_words[2u * site] = load[0];
    text_words[2u * site + 1u] = load[1];
  }
  for (uint16_t sgpr = 0; sgpr < 98u; ++sgpr) {
    text_words[2u * kSiteCount + sgpr] = build_s_mov_b32(sgpr, sgpr, ROCJITSU_CODE_ARCH_GFX1250);
  }
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250);

  const std::vector<uint8_t> bytes =
      make_gfx1250_code_object(text_words, "gfx1250_dense_explicit_key");
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::LocalCaveLdsLoadCheckTrap,
                               &ConSanPatchInfo::kind),
            kSiteCount);
  const auto explicit_dispatcher = std::ranges::find_if(result.patches, [](const auto &patch) {
    return patch.kind == ConSanPatchKind::TrampolineScDenseCallDispatcher &&
           patch.sc_dense_explicit_key_sgpr.has_value();
  });
  ASSERT_NE(explicit_dispatcher, result.patches.end());
  EXPECT_FALSE(explicit_dispatcher->sc_dense_call_return_sgpr.has_value());
}

TEST(ConSan, Gfx1250CheckTrapSpillsLiveVccSaveScalarThroughVgpr) {
  constexpr auto load = gfx1250::build_vds(gfx1250::kDsLoadB32Vds, {.addr = 2, .vdst = 1});
  std::vector<uint32_t> text_words = {load[0], load[1]};
  for (uint16_t sgpr = 0; sgpr < REGISTER_SET_ALLOCATABLE_SGPRS; ++sgpr)
    text_words.push_back(build_s_mov_b32(sgpr, sgpr, ROCJITSU_CODE_ARCH_GFX1250));
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250));

  const std::vector<uint8_t> bytes =
      make_gfx1250_code_object(text_words, "gfx1250_scalar_vcc_spill");
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.delay_mode = ConSanDelayMode::SleepVar;
  options.delay_nops = 1u;
  options.delay_var_ssrc = 0u;
  options.scratch_vgpr = 3;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.final_validation_passed);
  ASSERT_EQ(result.patches.size(), 1u);
  const ConSanPatchInfo &patch = result.patches.front();
  EXPECT_EQ(patch.kind, ConSanPatchKind::LocalCaveLdsLoadCheckTrap);
  EXPECT_EQ(patch.scratch_vgpr, 3u);
  EXPECT_FALSE(patch.indirect_pc_sgpr.has_value());
  ASSERT_TRUE(patch.scalar_vcc_spill_sgpr);
  ASSERT_TRUE(patch.scalar_vcc_spill_vgpr);
  EXPECT_NE(*patch.scalar_vcc_spill_sgpr, options.delay_var_ssrc);
  EXPECT_EQ(patch.required_sgpr_count, static_cast<uint16_t>(*patch.scalar_vcc_spill_sgpr + 1u));

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const auto scalar_save = instrumentation::build_v_writelane_b32(
      *patch.scalar_vcc_spill_vgpr, *patch.scalar_vcc_spill_sgpr, 0u, ROCJITSU_CODE_ARCH_GFX1250);
  const auto scalar_restore = instrumentation::build_v_readlane_b32(
      *patch.scalar_vcc_spill_sgpr, *patch.scalar_vcc_spill_vgpr, 0u, ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(scalar_save);
  ASSERT_TRUE(scalar_restore);
  const auto text = patched.text_sections().front();
  std::vector<uint32_t> body(patch.trampoline_size / sizeof(uint32_t));
  std::memcpy(body.data(), text->data() + patch.trampoline_offset, patch.trampoline_size);
  EXPECT_TRUE(contains_subsequence(body, *scalar_save));
  EXPECT_TRUE(contains_subsequence(body, *scalar_restore));
}

TEST(ConSan, Gfx1250CheckTrapBorrowsAndPreservesS0WhenRuntimeDelayIsDisabled) {
  constexpr auto load = gfx1250::build_vds(gfx1250::kDsLoadB32Vds, {.addr = 2, .vdst = 1});
  std::vector<uint32_t> text_words = {load[0], load[1]};
  for (uint16_t sgpr = 0; sgpr < REGISTER_SET_ALLOCATABLE_SGPRS; ++sgpr)
    text_words.push_back(build_s_mov_b32(sgpr, sgpr, ROCJITSU_CODE_ARCH_GFX1250));
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250));

  const std::vector<uint8_t> bytes =
      make_gfx1250_code_object(text_words, "gfx1250_scalar_vcc_spill_s0");
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.final_validation_passed);
  ASSERT_EQ(result.patches.size(), 1u);
  const ConSanPatchInfo &patch = result.patches.front();
  EXPECT_EQ(patch.scalar_vcc_spill_sgpr, 0u);
  ASSERT_TRUE(patch.scalar_vcc_spill_vgpr);
  EXPECT_EQ(patch.required_sgpr_count, 1u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const auto scalar_save = instrumentation::build_v_writelane_b32(
      *patch.scalar_vcc_spill_vgpr, 0u, /*lane=*/0u, ROCJITSU_CODE_ARCH_GFX1250);
  const auto scalar_restore = instrumentation::build_v_readlane_b32(
      0u, *patch.scalar_vcc_spill_vgpr, /*lane=*/0u, ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(scalar_save && scalar_restore);
  const std::vector<uint32_t> body =
      text_words_at_offset(patched, patch.trampoline_offset, patch.trampoline_size);
  EXPECT_TRUE(contains_subsequence(body, *scalar_save));
  EXPECT_TRUE(contains_subsequence(body, *scalar_restore));
}

TEST(ConSan, Gfx1250SharedLdsVccSpillUsesAllOwnersCommonSgprAllocation) {
  constexpr auto load = gfx1250::build_vds(gfx1250::kDsLoadB32Vds, {.addr = 2, .vdst = 1});
  std::vector<uint32_t> helper_words = {load[0], load[1]};
  for (uint16_t sgpr = 0; sgpr < REGISTER_SET_ALLOCATABLE_SGPRS; ++sgpr)
    helper_words.push_back(build_s_mov_b32(sgpr, sgpr, ROCJITSU_CODE_ARCH_GFX1250));

  TwoKernelSharedFixtureOptions fixture;
  std::vector<uint8_t> bytes =
      make_two_kernel_shared_helper_code_object(fixture, ROCJITSU_CODE_ARCH_RDNA4, helper_words);
  mutate_elf_header(bytes,
                    [](Elf64_Ehdr &header) { header.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX1250; });
  AmdGpuCodeObject original(bytes.data(), bytes.size());
  ASSERT_TRUE(original.is_valid());
  const auto first_owner =
      std::ranges::find(original.kernels(), "shared_owner_0", &AmdGpuKernelInfo::name);
  const auto second_owner =
      std::ranges::find(original.kernels(), "shared_owner_1", &AmdGpuKernelInfo::name);
  ASSERT_NE(first_owner, original.kernels().end());
  ASSERT_NE(second_owner, original.kernels().end());
  const auto set_sgpr_granulation = [&](uint64_t descriptor_offset, uint32_t granulated) {
    KD descriptor{};
    std::memcpy(&descriptor, bytes.data() + descriptor_offset, sizeof(descriptor));
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, granulated);
    std::memcpy(bytes.data() + descriptor_offset, &descriptor, sizeof(descriptor));
  };
  set_sgpr_granulation(first_owner->descriptor_file_offset, 3u);
  set_sgpr_granulation(second_owner->descriptor_file_offset, 0u);

  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.delay_mode = ConSanDelayMode::SleepVar;
  options.delay_nops = 1u;
  options.delay_var_ssrc = 0u;
  options.scratch_vgpr = 3u;
  options.max_patches = 1u;
  options.test_kernel_name_filter = "shared_owner_0";

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.final_validation_passed);
  const auto patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &info) {
    return info.kind == ConSanPatchKind::InlineLdsLoadCheckTrap ||
           info.kind == ConSanPatchKind::LocalCaveLdsLoadCheckTrap;
  });
  ASSERT_NE(patch, result.patches.end());
  ASSERT_TRUE(patch->scalar_vcc_spill_sgpr);
  ASSERT_TRUE(patch->scalar_vcc_spill_vgpr);
  EXPECT_NE(*patch->scalar_vcc_spill_sgpr, options.delay_var_ssrc);
  ASSERT_EQ(patch->owner_descriptor_file_offsets.size(), 2u);
  EXPECT_EQ(patch->owner_descriptor_file_offsets[0], first_owner->descriptor_file_offset);
  EXPECT_EQ(patch->owner_descriptor_file_offsets[1], second_owner->descriptor_file_offset);
  // The second owner has only one eight-SGPR granule. The shared body must
  // borrow a scalar already allocated by both owners and preserve it in the
  // emitted fixed-lane round trip.
  EXPECT_LE(patch->required_sgpr_count, 8u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const auto scalar_save = instrumentation::build_v_writelane_b32(
      *patch->scalar_vcc_spill_vgpr, *patch->scalar_vcc_spill_sgpr, /*lane=*/0u,
      ROCJITSU_CODE_ARCH_GFX1250);
  const auto scalar_restore = instrumentation::build_v_readlane_b32(
      *patch->scalar_vcc_spill_sgpr, *patch->scalar_vcc_spill_vgpr, /*lane=*/0u,
      ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_TRUE(scalar_save && scalar_restore);
  const std::vector<uint32_t> body =
      text_words_at_offset(patched, patch->trampoline_offset, patch->trampoline_size);
  EXPECT_TRUE(contains_subsequence(body, *scalar_save));
  EXPECT_TRUE(contains_subsequence(body, *scalar_restore));
  for (const auto &[owner_name, expected_granulated] :
       {std::pair{"shared_owner_0", 3u}, std::pair{"shared_owner_1", 0u}}) {
    const auto owner = std::ranges::find(patched.kernels(), owner_name, &AmdGpuKernelInfo::name);
    ASSERT_NE(owner, patched.kernels().end());
    KD descriptor{};
    std::memcpy(&descriptor, result.elf_bytes.data() + owner->descriptor_file_offset,
                sizeof(descriptor));
    EXPECT_EQ(AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc1,
                              kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT),
              expected_granulated);
  }
}

TEST(ConSan, Gfx1250SharedLdsDeadVccSaveStaysWithinCommonSgprAllocation) {
  constexpr auto load = gfx1250::build_vds(gfx1250::kDsLoadB32Vds, {.addr = 2, .vdst = 1});
  const std::array<uint32_t, 3> helper_words = {
      load[0],
      load[1],
      build_s_mov_b32(32u, 32u, ROCJITSU_CODE_ARCH_GFX1250),
  };

  TwoKernelSharedFixtureOptions fixture;
  std::vector<uint8_t> bytes =
      make_two_kernel_shared_helper_code_object(fixture, ROCJITSU_CODE_ARCH_RDNA4, helper_words);
  mutate_elf_header(bytes,
                    [](Elf64_Ehdr &header) { header.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX1250; });
  AmdGpuCodeObject original(bytes.data(), bytes.size());
  ASSERT_TRUE(original.is_valid());
  const auto first_owner =
      std::ranges::find(original.kernels(), "shared_owner_0", &AmdGpuKernelInfo::name);
  const auto second_owner =
      std::ranges::find(original.kernels(), "shared_owner_1", &AmdGpuKernelInfo::name);
  ASSERT_NE(first_owner, original.kernels().end());
  ASSERT_NE(second_owner, original.kernels().end());
  const auto set_sgpr_granulation = [&](uint64_t descriptor_offset, uint32_t granulated) {
    KD descriptor{};
    std::memcpy(&descriptor, bytes.data() + descriptor_offset, sizeof(descriptor));
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, granulated);
    std::memcpy(bytes.data() + descriptor_offset, &descriptor, sizeof(descriptor));
  };
  set_sgpr_granulation(first_owner->descriptor_file_offset, 3u);
  set_sgpr_granulation(second_owner->descriptor_file_offset, 0u);

  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3u;
  options.max_patches = 1u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.final_validation_passed);
  const auto patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &info) {
    return info.kind == ConSanPatchKind::InlineLdsLoadCheckTrap ||
           info.kind == ConSanPatchKind::LocalCaveLdsLoadCheckTrap;
  });
  ASSERT_NE(patch, result.patches.end());
  EXPECT_FALSE(patch->scalar_vcc_spill_sgpr);
  EXPECT_EQ(patch->required_sgpr_count, 0u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> body =
      text_words_at_offset(patched, patch->trampoline_offset, patch->trampoline_size);
  EXPECT_TRUE(contains_subsequence(
      body, std::array{build_s_mov_b32(0u, kRdna4VccLo, ROCJITSU_CODE_ARCH_GFX1250)}));
  EXPECT_TRUE(contains_subsequence(
      body, std::array{build_s_mov_b32(kRdna4VccLo, 0u, ROCJITSU_CODE_ARCH_GFX1250)}));
  for (const auto &[owner_name, expected_granulated] :
       {std::pair{"shared_owner_0", 3u}, std::pair{"shared_owner_1", 0u}}) {
    const auto owner = std::ranges::find(patched.kernels(), owner_name, &AmdGpuKernelInfo::name);
    ASSERT_NE(owner, patched.kernels().end());
    KD descriptor{};
    std::memcpy(&descriptor, result.elf_bytes.data() + owner->descriptor_file_offset,
                sizeof(descriptor));
    EXPECT_EQ(AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc1,
                              kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT),
              expected_granulated);
  }
}

TEST(ConSan, Gfx1250SharedLdsAutoScratchUsesAllOwnersAndGrowsEveryDescriptor) {
  constexpr auto load = gfx1250::build_vds(gfx1250::kDsLoadB32Vds, {.addr = 2, .vdst = 1});
  const std::array<uint32_t, 2> helper_words = {load[0], load[1]};

  TwoKernelSharedFixtureOptions fixture;
  fixture.first_vgpr_granulated = 0u;
  fixture.second_vgpr_granulated = 0u;
  fixture.first_continuation_live_vgprs = {0u};
  fixture.second_continuation_live_vgprs = {0u, 3u, 4u, 5u, 6u, 7u};
  std::vector<uint8_t> bytes =
      make_two_kernel_shared_helper_code_object(fixture, ROCJITSU_CODE_ARCH_RDNA4, helper_words);
  mutate_elf_header(bytes,
                    [](Elf64_Ehdr &header) { header.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX1250; });
  AmdGpuCodeObject original(bytes.data(), bytes.size());
  ASSERT_TRUE(original.is_valid());
  const auto first_owner =
      std::ranges::find(original.kernels(), "shared_owner_0", &AmdGpuKernelInfo::name);
  const auto second_owner =
      std::ranges::find(original.kernels(), "shared_owner_1", &AmdGpuKernelInfo::name);
  ASSERT_NE(first_owner, original.kernels().end());
  ASSERT_NE(second_owner, original.kernels().end());

  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.max_patches = 1u;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.final_validation_passed);
  const auto patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &info) {
    return info.kind == ConSanPatchKind::InlineLdsLoadCheckTrap ||
           info.kind == ConSanPatchKind::LocalCaveLdsLoadCheckTrap;
  });
  ASSERT_NE(patch, result.patches.end());
  ASSERT_TRUE(patch->scratch_vgpr);
  EXPECT_EQ(*patch->scratch_vgpr, 8u);
  ASSERT_EQ(patch->owner_descriptor_file_offsets.size(), 2u);
  EXPECT_EQ(patch->owner_descriptor_file_offsets[0], first_owner->descriptor_file_offset);
  EXPECT_EQ(patch->owner_descriptor_file_offsets[1], second_owner->descriptor_file_offset);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  for (std::string_view owner_name : {"shared_owner_0", "shared_owner_1"}) {
    const auto owner = std::ranges::find(patched.kernels(), owner_name, &AmdGpuKernelInfo::name);
    ASSERT_NE(owner, patched.kernels().end());
    KD descriptor{};
    std::memcpy(&descriptor, result.elf_bytes.data() + owner->descriptor_file_offset,
                sizeof(descriptor));
    EXPECT_EQ(AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc1,
                              kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT),
              1u);
  }
}

TEST(ConSan, Gfx1250CheckTrapRoutesSpillBackedFarBodyWithoutScalarPcPair) {
  constexpr size_t kTextWords = 33000u;
  constexpr size_t kRelaySiteWord = 16000u;
  constexpr auto load = gfx1250::build_vds(gfx1250::kDsLoadB32Vds, {.addr = 2, .vdst = 1});
  std::vector<uint32_t> text_words(kTextWords,
                                   build_s_mov_b32(100, 100, ROCJITSU_CODE_ARCH_GFX1250));
  text_words[0] = load[0];
  text_words[1] = load[1];
  for (uint16_t sgpr = 0; sgpr < REGISTER_SET_ALLOCATABLE_SGPRS; ++sgpr)
    text_words[2u + sgpr] = build_s_mov_b32(sgpr, sgpr, ROCJITSU_CODE_ARCH_GFX1250);
  text_words[kRelaySiteWord] = load[0];
  text_words[kRelaySiteWord + 1u] = load[1];
  text_words[kRelaySiteWord + 2u] = load[0];
  text_words[kRelaySiteWord + 3u] = load[1];
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250);

  const std::vector<uint8_t> bytes =
      make_gfx1250_code_object(text_words, "gfx1250_branch_only_scalar_spill");
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3;
  options.max_patches = 3;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.final_validation_passed);
  const auto branch_only =
      std::ranges::find(result.patches, true, &ConSanPatchInfo::sc_branch_only_continuation);
  ASSERT_NE(branch_only, result.patches.end()) << testing::PrintToString(result.warnings);
  EXPECT_EQ(branch_only->anchor_offset, 0u);
  EXPECT_FALSE(branch_only->indirect_pc_sgpr.has_value());
  ASSERT_FALSE(branch_only->sc_branch_only_entry_relay_offsets.empty());
  ASSERT_FALSE(branch_only->sc_branch_only_return_relay_offsets.empty());

  ConSanResult corrupted = result;
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const uint64_t text_file_offset = patched.text_sections().front()->sectionOffset();
  const uint32_t nop = build_s_nop(0, ROCJITSU_CODE_ARCH_GFX1250);
  std::memcpy(corrupted.elf_bytes.data() + text_file_offset +
                  branch_only->sc_branch_only_return_relay_offsets.front(),
              &nop, sizeof(nop));
  const auto corrupted_errors = validate_consan_modified_elf(bytes, corrupted);
  EXPECT_TRUE(std::ranges::any_of(corrupted_errors, [](const std::string &error) {
    return error.find("branch-only continuation proof found a stale, shared, or corrupted route") !=
           std::string::npos;
  }));
}

TEST(ConSan, Gfx1250CheckTrapRoutesSpillBackedFarBodyThroughRelayReservoir) {
  constexpr size_t kTextWords = 33010u;
  constexpr auto load = gfx1250::build_vds(gfx1250::kDsLoadB32Vds, {.addr = 2, .vdst = 1});
  std::vector<uint32_t> text_words(kTextWords,
                                   build_s_mov_b32(100, 100, ROCJITSU_CODE_ARCH_GFX1250));
  text_words[0] = load[0];
  text_words[1] = load[1];
  for (uint16_t sgpr = 0; sgpr < REGISTER_SET_ALLOCATABLE_SGPRS; ++sgpr)
    text_words[2u + sgpr] = build_s_mov_b32(sgpr, sgpr, ROCJITSU_CODE_ARCH_GFX1250);
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250);

  const std::vector<uint8_t> bytes =
      make_gfx1250_code_object(text_words, "gfx1250_branch_only_relay_reservoir");
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3;
  options.max_patches = 1;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.final_validation_passed);
  const auto branch_only =
      std::ranges::find(result.patches, true, &ConSanPatchInfo::sc_branch_only_continuation);
  ASSERT_NE(branch_only, result.patches.end()) << testing::PrintToString(result.warnings);
  EXPECT_EQ(branch_only->anchor_offset, 0u);
  EXPECT_FALSE(branch_only->sc_branch_only_entry_relay_offsets.empty());
  EXPECT_FALSE(branch_only->sc_branch_only_return_relay_offsets.empty());
  const auto reservoir = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineScRelayReservoir, &ConSanPatchInfo::kind);
  ASSERT_NE(reservoir, result.patches.end());
}

TEST(ConSan, ProbeLdsCheckTrapModeRoutesThroughRelocatedAnchorSecondWord) {
  constexpr uint64_t kMaximumForwardHop = 4u + 32767u * sizeof(uint32_t);
  constexpr uint64_t kSecondAnchorOffset = kMaximumForwardHop - sizeof(uint32_t);
  constexpr size_t kTextWords = (2u * kMaximumForwardHop) / sizeof(uint32_t) - 64u;
  std::vector<uint32_t> text_words(kTextWords, build_s_mov_b32(100, 100, ROCJITSU_CODE_ARCH_RDNA4));
  text_words[0] = 0xD8D80000u;
  text_words[1] = 0x01000002u; // ds_load_b32 v1, v2
  const size_t second_anchor_word = kSecondAnchorOffset / sizeof(uint32_t);
  text_words[second_anchor_word] = 0xD8D80000u;
  text_words[second_anchor_word + 1u] = 0x04000005u; // ds_load_b32 v4, v5
  text_words.back() = 0xBFB00000u;                   // s_endpgm

  const uint64_t original_text_size = text_words.size() * sizeof(uint32_t);
  ASSERT_FALSE(compute_sopp_branch_simm16(0u, original_text_size));
  ASSERT_TRUE(compute_sopp_branch_simm16(kSecondAnchorOffset, original_text_size));
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.max_patches = 2;
  options.scratch_vgpr = 6;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.modified);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::LocalCaveLdsLoadCheckTrap,
                               &ConSanPatchInfo::kind),
            2u);
  const auto island = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineScIndirectBranchIsland, &ConSanPatchInfo::kind);
  ASSERT_NE(island, result.patches.end());
  EXPECT_EQ(island->anchor_offset, 0u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  uint32_t relay_word = 0;
  const uint64_t relay_offset = kSecondAnchorOffset + sizeof(uint32_t);
  std::memcpy(&relay_word, patched.text_sections().front()->data() + relay_offset,
              sizeof(relay_word));
  const auto relay_branch = compute_sopp_branch_simm16(relay_offset, island->trampoline_offset);
  ASSERT_TRUE(relay_branch);
  EXPECT_EQ(relay_word, build_s_branch(*relay_branch, ROCJITSU_CODE_ARCH_RDNA4));
}

TEST(ConSan, ProbeLdsCheckTrapModeUsesVariableRelayReservoirAtMaximumCardinality) {
  constexpr size_t kTextWords = 33010u;
  constexpr uint64_t kReservoirOffset = 65536u;
  constexpr size_t kReservoirWords = 16u;
  constexpr size_t kReservoirEntryWords = 8u;
  constexpr size_t kReservoirTailWords = 1u;
  constexpr size_t kReservoirAppendedOverheadWords = 9u;
  const uint32_t inadmissible_filler = 0xBF870001u; // s_delay_alu instid0(VALU_DEP_1)
  std::vector<uint32_t> text_words(kTextWords, inadmissible_filler);
  for (size_t site = 0; site < 3u; ++site) {
    text_words[2u * site] = 0xD8D80000u;
    text_words[2u * site + 1u] = 0x01000002u; // ds_load_b32 v1, v2
  }
  std::array<uint32_t, kReservoirWords> reservoir_original{};
  for (size_t index = 0; index < reservoir_original.size(); ++index) {
    reservoir_original[index] =
        build_s_mov_b32(100, static_cast<uint16_t>(index), ROCJITSU_CODE_ARCH_RDNA4);
  }
  std::ranges::copy(reservoir_original,
                    text_words.begin() +
                        static_cast<ptrdiff_t>(kReservoirOffset / sizeof(uint32_t)));
  text_words.back() = 0xBFB00000u; // s_endpgm

  const uint64_t original_text_size = text_words.size() * sizeof(uint32_t);
  ASSERT_FALSE(compute_sopp_branch_simm16(0u, original_text_size));
  ASSERT_TRUE(compute_sopp_branch_simm16(0u, kReservoirOffset));
  ASSERT_TRUE(compute_sopp_branch_simm16(kReservoirOffset, original_text_size));
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "lds_probe", kRdna4Wave64AllVgprsGranulated,
                                 /*wave32=*/true);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.max_patches = 3;
  options.scratch_vgpr = 6;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.modified);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::LocalCaveLdsLoadCheckTrap,
                               &ConSanPatchInfo::kind),
            3u);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineScIndirectBranchIsland,
                               &ConSanPatchInfo::kind),
            3u);
  const auto reservoir = std::ranges::find(
      result.patches, ConSanPatchKind::TrampolineScRelayReservoir, &ConSanPatchInfo::kind);
  ASSERT_NE(reservoir, result.patches.end());
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineScRelayReservoir,
                               &ConSanPatchInfo::kind),
            1u);
  EXPECT_EQ(reservoir->anchor_offset, kReservoirOffset);
  EXPECT_EQ(reservoir->original_size, kReservoirWords * sizeof(uint32_t));
  EXPECT_EQ(reservoir->trampoline_size,
            (kReservoirWords + kReservoirAppendedOverheadWords) * sizeof(uint32_t));
  ASSERT_TRUE(reservoir->indirect_saved_vcc_sgpr.has_value());
  ASSERT_TRUE(reservoir->indirect_return_saved_vcc_sgpr.has_value());

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  const auto patched_text = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(patched.text_sections().front()->data()),
      patched.text_sections().front()->size());
  EXPECT_EQ(std::memcmp(reservoir_original.data(),
                        patched_text.data() + reservoir->trampoline_offset + sizeof(uint32_t),
                        reservoir->original_size),
            0);
  size_t used_payload_words = 0;
  for (uint64_t offset = reservoir->anchor_offset + kReservoirEntryWords * sizeof(uint32_t);
       offset <
       reservoir->anchor_offset + reservoir->original_size - kReservoirTailWords * sizeof(uint32_t);
       offset += sizeof(uint32_t)) {
    uint32_t word = 0;
    std::memcpy(&word, patched_text.data() + offset, sizeof(word));
    used_payload_words += word != build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
  }
  EXPECT_GE(used_payload_words, 3u);

  ConSanResult corrupted_body = result;
  const uint64_t text_file_offset = patched.text_sections().front()->sectionOffset();
  corrupted_body
      .elf_bytes[text_file_offset + reservoir->trampoline_offset + 2u * sizeof(uint32_t)] ^= 1u;
  const auto body_errors = validate_consan_modified_elf(bytes, corrupted_body);
  EXPECT_TRUE(std::ranges::any_of(body_errors, [](const std::string &error) {
    return error.find("relay reservoir proof found a corrupted displaced sequence") !=
           std::string::npos;
  }));

  ConSanResult unused = result;
  for (uint64_t offset = reservoir->anchor_offset + kReservoirEntryWords * sizeof(uint32_t);
       offset <
       reservoir->anchor_offset + reservoir->original_size - kReservoirTailWords * sizeof(uint32_t);
       offset += sizeof(uint32_t)) {
    const uint32_t nop = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
    std::memcpy(unused.elf_bytes.data() + text_file_offset + offset, &nop, sizeof(nop));
  }
  const auto unused_errors = validate_consan_modified_elf(bytes, unused);
  EXPECT_TRUE(std::ranges::any_of(unused_errors, [](const std::string &error) {
    return error.find("relay reservoir proof found an unused reservoir") != std::string::npos;
  }));
}

TEST(ConSan, ProbeLdsCheckTrapModePreplansIslandsBeforeAppendedCursorDriftsOutOfRange) {
  constexpr size_t kSiteCount = 160u;
  std::vector<uint32_t> text_words;
  text_words.reserve(2u * kSiteCount + 1u);
  for (size_t i = 0; i < kSiteCount; ++i) {
    text_words.push_back(0xD8D80000u);
    text_words.push_back(0x01000002u); // ds_load_b32 v1, v2
  }
  text_words.push_back(0xBFB00000u); // s_endpgm
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.max_patches = kSiteCount;
  options.scratch_vgpr = 3;
  options.delay_nops = 200;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.modified);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::LocalCaveLdsLoadCheckTrap,
                               &ConSanPatchInfo::kind),
            kSiteCount);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineScIndirectBranchIsland,
                               &ConSanPatchInfo::kind),
            kSiteCount);

  const auto last_body = std::find_if(
      result.patches.rbegin(), result.patches.rend(), [](const ConSanPatchInfo &patch) {
        return patch.kind == ConSanPatchKind::LocalCaveLdsLoadCheckTrap;
      });
  ASSERT_NE(last_body, result.patches.rend());
  EXPECT_FALSE(compute_sopp_branch_simm16(last_body->trampoline_offset, last_body->anchor_offset));
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
}

TEST(ConSan, Gfx1250CheckTrapPreplansBranchFallbackBeforeReachableBodiesDrift) {
  constexpr size_t kSiteCount = 40u;
  constexpr size_t kTextWords = 32000u;
  constexpr auto load = gfx1250::build_vds(gfx1250::kDsLoadB32Vds, {.addr = 2, .vdst = 1});
  std::vector<uint32_t> text_words(kTextWords,
                                   build_s_mov_b32(100, 100, ROCJITSU_CODE_ARCH_GFX1250));
  for (size_t i = 0; i < kSiteCount; ++i) {
    const size_t word = i * 780u;
    text_words[word] = load[0];
    text_words[word + 1u] = load[1];
  }
  // Leave one scalar word available for saving VCC, but not the three
  // additional words required by an indirect PC/SCC continuation.
  for (uint16_t sgpr = 1; sgpr < REGISTER_SET_ALLOCATABLE_SGPRS; ++sgpr) {
    const size_t word = kTextWords - REGISTER_SET_ALLOCATABLE_SGPRS + sgpr - 1u;
    text_words[word] = build_s_mov_b32(sgpr, sgpr, ROCJITSU_CODE_ARCH_GFX1250);
  }
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_GFX1250);

  const std::vector<uint8_t> bytes =
      make_gfx1250_code_object(text_words, "gfx1250_branch_fallback_preplan");
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.max_patches = kSiteCount;
  options.scratch_vgpr = 3;
  options.delay_nops = 200;

  const ConSanResult result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.warnings.empty()) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(result.modified);
  EXPECT_TRUE(result.final_validation_passed);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::LocalCaveLdsLoadCheckTrap,
                               &ConSanPatchInfo::kind),
            kSiteCount);
  EXPECT_NE(std::ranges::find(result.patches, true, &ConSanPatchInfo::sc_branch_only_continuation),
            result.patches.end());
}

TEST(ConSan, ProbeLdsCheckTrapModeReservesMultipleAppendedTextCaves) {
  const std::array<uint32_t, 5> text_words = {
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xD8D80000u,
      0x04000005u, // ds_load_b32 v4, v5
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.max_patches = 2;
  options.scratch_vgpr = 6;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  ASSERT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 2u);
  EXPECT_EQ(result.patches[0].kind, ConSanPatchKind::LocalCaveLdsLoadCheckTrap);
  EXPECT_EQ(result.patches[0].anchor_offset, 0u);
  EXPECT_EQ(result.patches[0].trampoline_offset, text_words.size() * sizeof(uint32_t));
  EXPECT_EQ(result.patches[1].kind, ConSanPatchKind::LocalCaveLdsLoadCheckTrap);
  EXPECT_EQ(result.patches[1].anchor_offset, 8u);
  EXPECT_EQ(result.patches[1].trampoline_offset,
            result.patches[0].trampoline_offset + result.patches[0].trampoline_size);
  EXPECT_GT(result.elf_bytes.size(), bytes.size());

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  EXPECT_TRUE(patched.is_valid());
}

TEST(ConSan, ProbeLdsCheckTrapModeComposesInlineAndAppendedTextCaves) {
  const std::array<uint32_t, 13> text_words = {
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u, 0xBF800000u,
      0xBF800000u, 0xBF800000u, 0xBF800000u, 0xD8D80000u,
      0x04000005u, // ds_load_b32 v4, v5
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.max_patches = 2;
  options.scratch_vgpr = 6;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  ASSERT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 2u);
  EXPECT_EQ(result.patches[0].kind, ConSanPatchKind::InlineLdsLoadCheckTrap);
  EXPECT_EQ(result.patches[0].anchor_offset, 0u);
  EXPECT_EQ(result.patches[1].kind, ConSanPatchKind::LocalCaveLdsLoadCheckTrap);
  EXPECT_EQ(result.patches[1].anchor_offset, 10u * sizeof(uint32_t));
  EXPECT_EQ(result.patches[1].trampoline_offset, text_words.size() * sizeof(uint32_t));
  EXPECT_GT(result.elf_bytes.size(), bytes.size());

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  EXPECT_TRUE(patched.is_valid());
}

TEST(ConSan, ProbeLdsCheckTrapModeReportsExcessiveDelay) {
  const std::array<uint32_t, 3> text_words = {
      0xD8D80000u,
      0x01000002u, // ds_load_b32 v1, v2
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_check_trap = true;
  options.scratch_vgpr = 3;
  options.delay_nops = 300;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_FALSE(result.warnings.empty());
  EXPECT_NE(result.warnings.back().find("requested delay needs too much padding"),
            std::string::npos);
  EXPECT_FALSE(result.modified);
  EXPECT_TRUE(result.elf_bytes.empty());
}

TEST(ConSan, ProbeLdsEndpgmModeCanRewriteCandidateWithExcludedAtomic) {
  const std::vector<uint8_t> bytes = make_rdna4_unsupported_lds_code_object();
  ConSanOptions options;
  options.flavor = ConSanFlavor::SuperCollider;
  options.probe_lds_endpgm = true;

  const auto result = try_patch_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.kernels.size(), 1u);
  EXPECT_EQ(result.kernels.front().preflight_action, ConSanPreflightAction::Candidate);
  EXPECT_TRUE(result.modified);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::InlineLdsEndpgmRewrite);
  EXPECT_EQ(result.patches.front().anchor_offset, 8u);
  EXPECT_EQ(result.patches.front().trampoline_offset, 8u);
  EXPECT_EQ(result.patches.front().original_size, 8u);
  ASSERT_EQ(result.elf_bytes.size(), bytes.size());
  EXPECT_NE(result.elf_bytes, bytes);

  uint32_t rewritten_word = 0;
  std::memcpy(&rewritten_word, result.elf_bytes.data() + 0x100 + 8, sizeof(rewritten_word));
  EXPECT_EQ(rewritten_word, 0xBFB00000u);
}

} // namespace
} // namespace rocjitsu
