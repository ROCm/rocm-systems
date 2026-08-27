// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"

namespace rocjitsu {
namespace {

struct MoiEngineConformanceCase {
  ConSanMoiEngine engine;
  ConSanPatchKind access_patch_kind;
  uint32_t access_body_words;
  uint32_t expected_island_bytes;
  const char *name;
};

class MoiEngineConformanceTest : public testing::TestWithParam<MoiEngineConformanceCase> {};

uint64_t report_buffer_bytes(const MoiEngineConformanceCase &test_case, uint32_t access_count) {
  switch (test_case.engine) {
  case ConSanMoiEngine::RecordReplay:
    return consan_moi_report_buffer_min_bytes(access_count, 0, 0, 0);
  case ConSanMoiEngine::Sampled:
    return direct_sampled_report_bytes(access_count);
  case ConSanMoiEngine::InlineShadow:
    return kInlineShadowFullLdsReportBufferSize;
  }
  ADD_FAILURE() << "unknown MOI engine";
  return 0;
}

ConSanOptions conformance_options(const MoiEngineConformanceCase &test_case,
                                  uint32_t access_count) {
  ConSanOptions options = moi_options(test_case.engine);
  options.scratch_vgpr = 8;
  options.moi_exec_save_sgpr = test_case.engine == ConSanMoiEngine::InlineShadow ? 60u : 80u;
  options.moi_owner_vgpr = 40;
  options.moi_epoch_vgpr = 41;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = report_buffer_bytes(test_case, access_count);
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.max_patches = access_count;
  return options;
}

TEST_P(MoiEngineConformanceTest, UsesBranchIslandsForManyLargeAccessBodies) {
  constexpr uint32_t kAccessCount = 9;
  const MoiEngineConformanceCase &test_case = GetParam();
  std::vector<uint32_t> text_words;
  for (uint32_t i = 0; i < kAccessCount; ++i) {
    text_words.push_back(0xD8340000u); // ds_store_b32 v0, v0
    text_words.push_back(0x00000000u);
  }
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4));

  const auto result = test_lower_consan(make_rdna4_lds_code_object(text_words),
                                        conformance_options(test_case, kAccessCount));

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified()) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid)
      << testing::PrintToString(result.errors);
  EXPECT_EQ(std::ranges::count(result.patches, test_case.access_patch_kind, &ConSanPatchInfo::kind),
            kAccessCount);
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiIndirectBranchIsland,
                               &ConSanPatchInfo::kind),
            kAccessCount);
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.kind != ConSanPatchKind::TrampolineMoiIndirectBranchIsland)
      continue;
    EXPECT_EQ(patch.trampoline_size, test_case.expected_island_bytes);
    EXPECT_TRUE(compute_sopp_branch_simm16(patch.anchor_offset, patch.trampoline_offset));
  }
}

TEST_P(MoiEngineConformanceTest, InstrumentsGfx1100NativeLdsAccess) {
  constexpr auto store = rdna3::build_ds(rdna3::kDsStoreB32Ds, {.addr = 0, .data0 = 1});
  const std::array<uint32_t, 3> text_words = {store[0], store[1],
                                              build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA3)};
  const MoiEngineConformanceCase &test_case = GetParam();
  for (bool wave32 : std::array{false, true}) {
    SCOPED_TRACE(wave32 ? "wave32" : "wave64");
    const ConSanTransformArtifacts result =
        test_lower_consan(make_rdna3_lds_code_object(text_words, "gfx1100_native_lds",
                                                     kRdna4Wave64AllVgprsGranulated, wave32,
                                                     /*uses_dynamic_stack=*/false,
                                                     /*workgroup_id_dimension_mask=*/7u),
                          conformance_options(test_case, /*access_count=*/1u));

    ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
    ASSERT_TRUE(result.modified()) << testing::PrintToString(result.warnings);
    ASSERT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid)
        << testing::PrintToString(result.errors);
    EXPECT_EQ(result.program_inventory.target(), ROCJITSU_CODE_TARGET_GFX1100);
    EXPECT_EQ(result.program_inventory.arch(), ROCJITSU_CODE_ARCH_RDNA3);
    EXPECT_EQ(
        std::ranges::count(result.patches, test_case.access_patch_kind, &ConSanPatchInfo::kind),
        1u);
  }
}

TEST_P(MoiEngineConformanceTest, InstrumentsGfx1100SingletonWorkgroupBarrier) {
  constexpr auto store = rdna3::build_ds(rdna3::kDsStoreB32Ds, {.addr = 0, .data0 = 1});
  const std::array<uint32_t, 4> text_words = {store[0], store[1],
                                              *build_rdna3_s_barrier(ROCJITSU_CODE_ARCH_RDNA3),
                                              build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA3)};
  const MoiEngineConformanceCase &test_case = GetParam();
  ConSanOptions options = conformance_options(test_case, /*access_count=*/1u);
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.moi_track_barriers = true;
  options.max_patches = 2u;
  const ConSanPatchKind expected_kind = test_case.engine == ConSanMoiEngine::InlineShadow
                                            ? ConSanPatchKind::TrampolineMoiInlineEpochBarrier
                                        : test_case.engine == ConSanMoiEngine::Sampled
                                            ? ConSanPatchKind::TrampolineMoiSampledSyncMetadata
                                            : ConSanPatchKind::TrampolineMoiBarrierRecord;
  for (bool wave32 : std::array{false, true}) {
    SCOPED_TRACE(wave32 ? "wave32" : "wave64");
    const ConSanTransformArtifacts result =
        test_lower_consan(make_rdna3_lds_code_object(text_words, "gfx1100_workgroup_barrier",
                                                     kRdna4Wave64AllVgprsGranulated, wave32,
                                                     /*uses_dynamic_stack=*/false,
                                                     /*workgroup_id_dimension_mask=*/7u),
                          options);

    ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
    ASSERT_TRUE(result.modified()) << testing::PrintToString(result.warnings);
    ASSERT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid)
        << testing::PrintToString(result.errors);
    EXPECT_EQ(result.program_inventory.target(), ROCJITSU_CODE_TARGET_GFX1100);
    EXPECT_EQ(result.program_inventory.arch(), ROCJITSU_CODE_ARCH_RDNA3);
    EXPECT_EQ(std::ranges::count(result.patches, expected_kind, &ConSanPatchInfo::kind), 1u)
        << testing::PrintToString(result.warnings) << testing::PrintToString(result.patches);
  }
}

TEST_P(MoiEngineConformanceTest, MaterializesGfx12ScalarVectorGroupFlatAddress) {
  const MoiEngineConformanceCase &test_case = GetParam();
  for (const rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    SCOPED_TRACE(arch == ROCJITSU_CODE_ARCH_RDNA4 ? "gfx1201" : "gfx1250");
    std::array<uint32_t, 5> text_words{};
    text_words[0] = 0xBE8E01EBu; // s_mov_b64 s[14:15], src_shared_base
    if (arch == ROCJITSU_CODE_ARCH_RDNA4) {
      constexpr auto load = rdna4::build_vflat(
          rdna4::kFlatLoadB32Vflat, {.saddr = 14u, .vdst = 2u, .vaddr = 2u, .ioffset = 12u});
      std::ranges::copy(load, text_words.begin() + 1u);
    } else {
      constexpr auto load = cdna5::build_vflat(
          cdna5::kFlatLoadB32Vflat,
          {.saddr = 14u, .vdst = 2u, .scale_offset = 1u, .vaddr = 2u, .ioffset = 12u});
      std::ranges::copy(load, text_words.begin() + 1u);
    }
    text_words.back() = build_s_endpgm(arch);
    const std::vector<uint8_t> bytes =
        arch == ROCJITSU_CODE_ARCH_RDNA4
            ? make_rdna4_lds_code_object(text_words, "gfx1201_scalar_vector_group_flat")
            : make_gfx1250_code_object(text_words, "gfx1250_scalar_vector_group_flat");

    const ConSanTransformArtifacts result =
        test_lower_consan(bytes, conformance_options(test_case, /*access_count=*/1u));

    ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
    ASSERT_TRUE(result.modified()) << testing::PrintToString(result.warnings);
    ASSERT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid)
        << testing::PrintToString(result.errors);
    ASSERT_EQ(test_admitted_accesses(result).size(), 1u);
    const ConSanAccessInventorySite candidate = test_admitted_accesses(result).front();
    EXPECT_EQ(candidate.origin, ConSanAccessOrigin::Flat);
    EXPECT_EQ(candidate.flat_address_space_hint, ConSanFlatAddressSpaceHint::Group);
    EXPECT_EQ(candidate.operands.raw_saddr, 14u);
    EXPECT_EQ(candidate.operands.raw_ioffset, 12);
    ASSERT_TRUE(candidate.operands.raw_scale_offset);
    EXPECT_EQ(*candidate.operands.raw_scale_offset, arch == ROCJITSU_CODE_ARCH_CDNA5);
    EXPECT_EQ(
        std::ranges::count(result.patches, test_case.access_patch_kind, &ConSanPatchInfo::kind), 1u)
        << testing::PrintToString(result.warnings);

    AmdGpuCodeObject patched(result.replacement.data(), result.replacement.size());
    ASSERT_TRUE(patched.is_valid());
    ASSERT_EQ(patched.text_sections().size(), 1u);
    std::vector<uint32_t> patched_words(patched.text_sections().front()->size() / sizeof(uint32_t));
    std::memcpy(patched_words.data(), patched.text_sections().front()->data(),
                patched.text_sections().front()->size());
    bool found_materialization = false;
    for (uint16_t result_vgpr = 8u; result_vgpr <= 64u && !found_materialization; ++result_vgpr) {
      const uint16_t offset_vgpr = static_cast<uint16_t>(result_vgpr + 2u);
      for (const uint16_t input_vgpr : {uint16_t{2u}, result_vgpr}) {
        std::vector<uint32_t> expected;
        if (arch == ROCJITSU_CODE_ARCH_CDNA5) {
          const auto scale = instrumentation::build_v_mul_lo_u32_literal(
              offset_vgpr, offset_vgpr, sizeof(uint32_t), input_vgpr, arch);
          ASSERT_TRUE(scale);
          expected.insert(expected.end(), scale->begin(), scale->end());
        } else {
          expected.push_back(
              build_v_mov_b32_e32(offset_vgpr, vector_source_vgpr(input_vgpr), arch));
        }
        expected.push_back(build_v_mov_b32_e32(result_vgpr, /*s14=*/14u, arch));
        expected.push_back(
            build_v_mov_b32_e32(static_cast<uint16_t>(result_vgpr + 1u), /*s15=*/15u, arch));
        const auto add_vector =
            instrumentation::build_v_add_u64_vgpr_offset(result_vgpr, offset_vgpr, arch);
        const auto add_immediate =
            instrumentation::build_v_add_u64_signed_i24(result_vgpr, 12, arch);
        ASSERT_TRUE(add_vector && add_immediate);
        expected.insert(expected.end(), add_vector->begin(), add_vector->end());
        expected.insert(expected.end(), add_immediate->begin(), add_immediate->end());
        if (contains_subsequence(patched_words, expected)) {
          found_materialization = true;
          break;
        }
      }
    }
    EXPECT_TRUE(found_materialization)
        << "missing scalar-base + vector-offset + immediate VFLAT materialization";
  }
}

TEST_P(MoiEngineConformanceTest, RelocatesStraightLinePrefixWhenNoEntryIslandIsReachable) {
  const MoiEngineConformanceCase &test_case = GetParam();
  const std::array<uint32_t, 2> kernel_words = {
      pack_sopk(/*s_call_b64=*/0x14, /*sdst=*/30, /*simm16=*/1),
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  std::vector<uint32_t> function_words = {
      0xBE8001EBu, // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u,
      0x00000000u, // v_mov_b32_e64 v0, s0
      0xD5810001u,
      0x00000001u, // v_mov_b32_e64 v1, s1
      0xEC05007Cu, 0x00000002u,
      0x00000000u, // flat_load_b32 v2, v[0:1]
  };
  const uint32_t displaced_scalar_count = test_case.access_body_words - 3;
  for (uint32_t i = 0; i < displaced_scalar_count; ++i) {
    const uint16_t sgpr = static_cast<uint16_t>(20 + i);
    function_words.push_back(build_s_mov_b32(sgpr, sgpr, ROCJITSU_CODE_ARCH_RDNA4));
  }
  function_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4));
  const uint16_t tail_sgpr = static_cast<uint16_t>(20 + displaced_scalar_count);
  std::vector<uint32_t> tail_words(40000u,
                                   build_s_mov_b32(tail_sgpr, tail_sgpr, ROCJITSU_CODE_ARCH_RDNA4));
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words, tail_words);

  const auto result = test_lower_consan(bytes, conformance_options(test_case, 1));

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified()) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid)
      << testing::PrintToString(result.errors);
  const auto patch =
      std::ranges::find(result.patches, test_case.access_patch_kind, &ConSanPatchInfo::kind);
  ASSERT_NE(patch, result.patches.end());
  EXPECT_EQ(patch->anchor_offset, 28u);
  EXPECT_EQ(patch->original_size, test_case.access_body_words * sizeof(uint32_t));
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiIndirectBranchIsland,
                               &ConSanPatchInfo::kind),
            0);
}

TEST_P(MoiEngineConformanceTest, InstrumentsEveryAdjacentAccessInsideRelocationRange) {
  const MoiEngineConformanceCase &test_case = GetParam();
  const std::array<uint32_t, 2> kernel_words = {
      pack_sopk(/*s_call_b64=*/0x14, /*sdst=*/30, /*simm16=*/1),
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  std::vector<uint32_t> function_words = {
      0xBE8001EBu, // s_mov_b64 s[0:1], src_shared_base
      0xD5810000u,
      0x00000000u, // v_mov_b32_e64 v0, s0
      0xD5810001u,
      0x00000001u,                           // v_mov_b32_e64 v1, s1
      0xEC05007Cu, 0x00000002u, 0x00000000u, // flat_load_b32 v2, v[0:1]
      0xEC05007Cu, 0x00000003u, 0x00000000u, // flat_load_b32 v3, v[0:1]
  };
  for (uint32_t i = 0; i < test_case.access_body_words; ++i)
    function_words.push_back(build_s_mov_b32(
        static_cast<uint16_t>(20 + i), static_cast<uint16_t>(20 + i), ROCJITSU_CODE_ARCH_RDNA4));
  function_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4));
  std::vector<uint32_t> tail_words(40000u, build_s_mov_b32(40, 40, ROCJITSU_CODE_ARCH_RDNA4));
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words, tail_words);

  const auto result = test_lower_consan(bytes, conformance_options(test_case, 2));

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid)
      << testing::PrintToString(result.errors);
  // Relocation must not silently consume another admitted source site: every
  // adjacent access retains an independently identifiable instrumentation patch.
  EXPECT_EQ(std::ranges::count(result.patches, test_case.access_patch_kind, &ConSanPatchInfo::kind),
            2u);
  for (uint64_t anchor_offset : {28u, 40u}) {
    EXPECT_TRUE(std::ranges::any_of(result.patches, [&](const ConSanPatchInfo &patch) {
      return patch.kind == test_case.access_patch_kind && patch.anchor_offset == anchor_offset;
    }));
  }
}

TEST_P(MoiEngineConformanceTest, Gfx1250RoutesSparseAccessesWithStrandedAppendedEntries) {
  constexpr uint32_t kAccessCount = 9;
  constexpr size_t kTextWords = 40000u;
  const uint32_t filler = build_s_mov_b32(/*sdst=*/0, /*ssrc0=*/0, ROCJITSU_CODE_ARCH_CDNA5);
  std::vector<uint32_t> text_words(kTextWords, filler);
  text_words[18000u] = build_s_branch(/*simm16=*/8, ROCJITSU_CODE_ARCH_CDNA5);
  constexpr auto store = cdna5::build_vds(cdna5::kDsStoreB32Vds, {.addr = 0, .data0 = 1});
  constexpr std::array<size_t, kAccessCount> kStoreOffsets = {
      32u, 4000u, 8000u, 12000u, 16000u, 20000u, 24000u, 28000u, 32000u,
  };
  for (size_t offset : kStoreOffsets) {
    text_words[offset] = store[0];
    text_words[offset + 1u] = store[1];
  }
  text_words.back() = build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5);

  const MoiEngineConformanceCase &test_case = GetParam();
  ConSanOptions options = conformance_options(test_case, kAccessCount);
  if (test_case.engine == ConSanMoiEngine::InlineShadow) {
    options.scratch_vgpr = 82;
    options.moi_owner_vgpr = 80;
    options.moi_epoch_vgpr = 81;
    options.moi_exec_save_sgpr = 60;
  }
  const ConSanTransformArtifacts result = test_lower_consan(
      make_gfx1250_code_object(text_words, "gfx1250_sparse_stranded_accesses"), options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified()) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid)
      << testing::PrintToString(result.errors);
  EXPECT_EQ(std::ranges::count(result.patches, test_case.access_patch_kind, &ConSanPatchInfo::kind),
            kAccessCount)
      << testing::PrintToString(result.warnings);
  for (size_t store_offset : kStoreOffsets) {
    const uint64_t anchor_offset = store_offset * sizeof(uint32_t);
    EXPECT_TRUE(std::ranges::any_of(result.patches,
                                    [&](const ConSanPatchInfo &patch) {
                                      return patch.kind == test_case.access_patch_kind &&
                                             patch.anchor_offset == anchor_offset;
                                    }))
        << "missing access patch at byte offset " << anchor_offset;
  }
  EXPECT_EQ(std::ranges::count(result.patches, ConSanPatchKind::TrampolineMoiIndirectBranchIsland,
                               &ConSanPatchInfo::kind),
            2u); // One relocatable host plus one appended return-PC dispatcher.
}

INSTANTIATE_TEST_SUITE_P(
    AllEngines, MoiEngineConformanceTest,
    testing::Values(MoiEngineConformanceCase{ConSanMoiEngine::RecordReplay,
                                             ConSanPatchKind::TrampolineMoiAccessRecordStore, 7, 40,
                                             "RecordReplay"},
                    MoiEngineConformanceCase{ConSanMoiEngine::Sampled,
                                             ConSanPatchKind::TrampolineMoiSampledWatchpointStore,
                                             7, 28, "Sampled"},
                    MoiEngineConformanceCase{ConSanMoiEngine::InlineShadow,
                                             ConSanPatchKind::TrampolineMoiExactShadowStore, 8, 32,
                                             "InlineShadow"}),
    [](const testing::TestParamInfo<MoiEngineConformanceCase> &info) { return info.param.name; });

} // namespace
} // namespace rocjitsu
