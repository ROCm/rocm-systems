// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"
#include "rocjitsu/code/patch/consan/consan_cfg.h"
#include "rocjitsu/code/patch/consan/consan_moi_internal.h"
#include "rocjitsu/code/patch/consan/consan_physical_site_alias.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"
#include "rocjitsu/code/patch/spill_manager.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <limits>
#include <set>
#include <string_view>

namespace rocjitsu {
namespace {

TEST(ConSanMoi, ResidentWaveOwnerTargetOperationLowersEverySupportedProfile) {
  constexpr uint16_t destination_sgpr = 20u;
  for (const ConSanTargetProfile &target : kConSanTargetProfiles) {
    SCOPED_TRACE(rj_code_target_name(target.target));
    const auto hwreg = build_hwreg_imm(target.resident_wave_identity.hwreg_id,
                                       target.resident_wave_identity.bit_offset,
                                       target.resident_wave_identity.bit_width);
    const auto read =
        hwreg ? instrumentation::build_s_getreg_b32(destination_sgpr, *hwreg, target.arch)
              : std::nullopt;
    const auto delay = instrumentation::build_salu_dependency_delay(target.arch);
    const auto wait = instrumentation::build_salu_to_valu_dependency_wait(target.arch);
    ASSERT_TRUE(read);
    ASSERT_TRUE(delay);
    ASSERT_TRUE(wait);

    for (bool one_based : {false, true}) {
      SCOPED_TRACE(one_based ? "one-based" : "zero-based");
      const consan_detail::MoiResidentWaveOwnerRequest request{
          .destination_sgpr = destination_sgpr,
          .one_based = one_based,
      };
      EXPECT_EQ(request, (consan_detail::MoiResidentWaveOwnerRequest{
                             .destination_sgpr = destination_sgpr,
                             .one_based = one_based,
                         }));
      std::vector<uint32_t> words;
      ASSERT_TRUE(consan_detail::append_moi_resident_wave_owner(words, request, target));
      std::vector<uint32_t> expected = {*read, *delay};
      if (one_based) {
        expected.push_back(build_s_add_u32(destination_sgpr, destination_sgpr,
                                           scalar_positive_inline_u32(1), target.arch));
        expected.push_back(*delay);
      }
      expected.push_back(*wait);
      EXPECT_EQ(words, expected);
    }
  }
}

TEST(ConSanMoi, ResidentWaveOwnerTargetOperationRejectsWithoutPartialOutput) {
  ASSERT_FALSE(kConSanTargetProfiles.empty());
  const ConSanTargetProfile &target = kConSanTargetProfiles.front();
  const std::vector<uint32_t> prefix = {0x12345678u};

  std::vector<uint32_t> invalid_destination_words = prefix;
  EXPECT_FALSE(consan_detail::append_moi_resident_wave_owner(
      invalid_destination_words,
      {.destination_sgpr = std::numeric_limits<uint16_t>::max(), .one_based = true}, target));
  EXPECT_EQ(invalid_destination_words, prefix);

  ConSanTargetProfile invalid_target = target;
  invalid_target.resident_wave_identity.bit_width = 0u;
  std::vector<uint32_t> invalid_target_words = prefix;
  EXPECT_FALSE(consan_detail::append_moi_resident_wave_owner(
      invalid_target_words, {.destination_sgpr = 20u, .one_based = true}, invalid_target));
  EXPECT_EQ(invalid_target_words, prefix);
}

TEST(ConSanMoi, SpecialStatePreservationTargetOperationCoversEveryProfile) {
  const consan_detail::MoiSpecialStateSgprs registers{
      .vcc_save_sgpr = 20u,
      .scc_save_sgpr = 24u,
  };
  EXPECT_EQ(registers, (consan_detail::MoiSpecialStateSgprs{
                           .vcc_save_sgpr = 20u,
                           .scc_save_sgpr = 24u,
                       }));
  for (const ConSanTargetProfile &target : kConSanTargetProfiles) {
    SCOPED_TRACE(rj_code_target_name(target.target));
    const auto save_scc =
        instrumentation::build_s_cselect_b32(registers.scc_save_sgpr, scalar_positive_inline_u32(1),
                                             scalar_positive_inline_u32(0), target.arch);
    const auto save_vcc = instrumentation::build_s_mov_b64(
        registers.vcc_save_sgpr, scalar_operand_vcc_lo(target.arch), target.arch);
    const auto restore_vcc = instrumentation::build_s_mov_b64(scalar_operand_vcc_lo(target.arch),
                                                              registers.vcc_save_sgpr, target.arch);
    const auto restore_scc = instrumentation::build_s_cmp_lg_u32(
        registers.scc_save_sgpr, scalar_positive_inline_u32(0), target.arch);
    ASSERT_TRUE(save_scc);
    ASSERT_TRUE(save_vcc);
    ASSERT_TRUE(restore_vcc);
    ASSERT_TRUE(restore_scc);

    std::vector<uint32_t> words;
    ASSERT_TRUE(consan_detail::append_save_moi_special_state(words, registers, target));
    EXPECT_EQ(words, (std::vector<uint32_t>{*save_scc, *save_vcc}));
    words.clear();
    ASSERT_TRUE(consan_detail::append_restore_moi_special_state(words, registers, target));
    EXPECT_EQ(words, (std::vector<uint32_t>{*restore_vcc, *restore_scc}));
  }
}

TEST(ConSanMoi, SpecialStatePreservationRejectsWithoutPartialOutput) {
  ASSERT_FALSE(kConSanTargetProfiles.empty());
  const ConSanTargetProfile &target = kConSanTargetProfiles.front();
  const std::vector<uint32_t> prefix = {0x12345678u};

  std::vector<uint32_t> save_words = prefix;
  EXPECT_FALSE(consan_detail::append_save_moi_special_state(
      save_words, {.vcc_save_sgpr = std::numeric_limits<uint16_t>::max(), .scc_save_sgpr = 24u},
      target));
  EXPECT_EQ(save_words, prefix);

  std::vector<uint32_t> restore_words = prefix;
  EXPECT_FALSE(consan_detail::append_restore_moi_special_state(
      restore_words, {.vcc_save_sgpr = 20u, .scc_save_sgpr = std::numeric_limits<uint16_t>::max()},
      target));
  EXPECT_EQ(restore_words, prefix);
}

TEST(ConSanMoi, EncodedRouteSccRestoreIsConfinedToQualifiedTargetProfiles) {
  constexpr consan_detail::MoiEncodedSccRestoreRequest request{.encoded_sgpr = 20u};
  EXPECT_EQ(request,
            (consan_detail::MoiEncodedSccRestoreRequest{.encoded_sgpr = request.encoded_sgpr}));
  for (const ConSanTargetProfile &target : kConSanTargetProfiles) {
    SCOPED_TRACE(rj_code_target_name(target.target));
    std::vector<uint32_t> words;
    const bool gfx9_cdna = target.encoding_family == ConSanEncodingFamily::Gfx9Cdna3 ||
                           target.encoding_family == ConSanEncodingFamily::Gfx9Cdna4;
    EXPECT_EQ(consan_detail::append_restore_moi_scc_from_route_key(words, request, target),
              gfx9_cdna);
    if (!gfx9_cdna) {
      EXPECT_TRUE(words.empty());
      continue;
    }
    const uint16_t opcode = target.encoding_family == ConSanEncodingFamily::Gfx9Cdna3
                                ? cdna3::kSBitcmp1B32Sopc
                                : cdna4::kSBitcmp1B32Sopc;
    const auto normalize =
        instrumentation::build_s_cselect_b32(request.encoded_sgpr, scalar_positive_inline_u32(1),
                                             scalar_positive_inline_u32(0), target.arch);
    ASSERT_TRUE(normalize);
    EXPECT_EQ(words, (std::vector<uint32_t>{
                         build_sopc_encoding(target.arch, opcode, request.encoded_sgpr,
                                             scalar_positive_inline_u32(0)),
                         *normalize,
                     }));
  }
}

TEST(ConSanMoi, EncodedRouteSccRestoreRejectsInvalidRegisterTransactionally) {
  const ConSanTargetProfile *target = consan_target_profile(ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_NE(target, nullptr);
  const std::vector<uint32_t> prefix = {0x12345678u};
  std::vector<uint32_t> words = prefix;
  EXPECT_FALSE(consan_detail::append_restore_moi_scc_from_route_key(
      words, {.encoded_sgpr = std::numeric_limits<uint16_t>::max()}, *target));
  EXPECT_EQ(words, prefix);
}

TEST(ConSanMoi, DeviceCacheRefreshUsesOnlyQualifiedTargetSequences) {
  for (const ConSanTargetProfile &target : kConSanTargetProfiles) {
    SCOPED_TRACE(rj_code_target_name(target.target));
    std::vector<uint32_t> words;
    ASSERT_TRUE(consan_detail::append_moi_device_cache_refresh(words, target));
    switch (target.encoding_family) {
    case ConSanEncodingFamily::Gfx9Cdna3: {
      const auto expected = build_cdna3_buffer_inv_sc1(target.arch);
      ASSERT_TRUE(expected);
      EXPECT_EQ(words, std::vector<uint32_t>(expected->begin(), expected->end()));
      break;
    }
    case ConSanEncodingFamily::Gfx9Cdna4: {
      const auto expected = build_cdna4_buffer_inv_sc1(target.arch);
      ASSERT_TRUE(expected);
      EXPECT_EQ(words, std::vector<uint32_t>(expected->begin(), expected->end()));
      break;
    }
    case ConSanEncodingFamily::Gfx11:
    case ConSanEncodingFamily::Gfx12:
      EXPECT_TRUE(words.empty());
      break;
    }
  }
}

TEST(ConSanMoi, GlobalAtomicCompletionUsesEachTargetsCounterModel) {
  for (const ConSanTargetProfile &target : kConSanTargetProfiles) {
    SCOPED_TRACE(rj_code_target_name(target.target));
    const auto load = instrumentation::build_s_wait_global_load0(target.arch);
    ASSERT_TRUE(load);
    std::vector<uint32_t> expected = {*load};
    if (target.encoding_family != ConSanEncodingFamily::Gfx9Cdna3 &&
        target.encoding_family != ConSanEncodingFamily::Gfx9Cdna4) {
      const auto store = instrumentation::build_s_wait_global_store0(target.arch);
      ASSERT_TRUE(store);
      expected.push_back(*store);
    }
    std::vector<uint32_t> words;
    EXPECT_TRUE(consan_detail::append_moi_global_atomic_completion(words, target));
    EXPECT_EQ(words, expected);
  }
}

TEST(ConSanMoi, GlobalAtomicCompletionRejectsInvalidTargetTransactionally) {
  ConSanTargetProfile target = kConSanTargetProfiles.front();
  target.arch = ROCJITSU_CODE_ARCH_INVALID;
  const std::vector<uint32_t> prefix = {0x12345678u};
  std::vector<uint32_t> words = prefix;
  EXPECT_FALSE(consan_detail::append_moi_global_atomic_completion(words, target));
  EXPECT_EQ(words, prefix);
}

TEST(ConSanMoi, AtomicCounterIncrementLowersEveryTargetProfile) {
  constexpr consan_detail::MoiAtomicCounterIncrementRequest kRequest{
      .counter_address = 0x1234567800000040ull,
      .result_vgpr = 42u,
      .address_vgpr = 40u,
  };
  for (const ConSanTargetProfile &target : kConSanTargetProfiles) {
    SCOPED_TRACE(rj_code_target_name(target.target));
    const auto address_lo = instrumentation::build_v_mov_b32_literal(
        kRequest.address_vgpr, static_cast<uint32_t>(kRequest.counter_address), target.arch);
    const auto address_hi = instrumentation::build_v_mov_b32_literal(
        static_cast<uint16_t>(kRequest.address_vgpr + 1u),
        static_cast<uint32_t>(kRequest.counter_address >> 32u), target.arch);
    const auto one =
        instrumentation::build_v_mov_b32_literal(kRequest.result_vgpr, 1u, target.arch);
    const auto atomic = instrumentation::build_flat_atomic_add_u32(
        kRequest.address_vgpr, kRequest.result_vgpr, kRequest.result_vgpr,
        /*return_old_value=*/true, /*scope=*/2u, target.arch);
    ASSERT_TRUE(address_lo && address_hi && one && atomic);
    std::vector<uint32_t> expected;
    expected.insert(expected.end(), address_lo->begin(), address_lo->end());
    expected.insert(expected.end(), address_hi->begin(), address_hi->end());
    expected.insert(expected.end(), one->begin(), one->end());
    expected.insert(expected.end(), atomic->begin(), atomic->end());
    ASSERT_TRUE(consan_detail::append_moi_global_atomic_completion(expected, target));

    std::vector<uint32_t> words;
    EXPECT_TRUE(consan_detail::append_moi_atomic_counter_increment(words, kRequest, target));
    EXPECT_EQ(words, expected);
  }
}

TEST(ConSanMoi, AtomicCounterIncrementRejectsInvalidRegistersTransactionally) {
  const ConSanTargetProfile *target = consan_target_profile(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(target, nullptr);
  const auto rejects = [&](uint16_t result_vgpr, uint16_t address_vgpr,
                           const ConSanTargetProfile *request_target = nullptr) {
    const std::vector<uint32_t> prefix = {0x12345678u};
    std::vector<uint32_t> words = prefix;
    EXPECT_FALSE(consan_detail::append_moi_atomic_counter_increment(
        words,
        {
            .counter_address = 0x1234567800000040ull,
            .result_vgpr = result_vgpr,
            .address_vgpr = address_vgpr,
        },
        request_target != nullptr ? *request_target : *target));
    EXPECT_EQ(words, prefix);
  };
  rejects(/*result_vgpr=*/40u, /*address_vgpr=*/40u);
  rejects(/*result_vgpr=*/41u, /*address_vgpr=*/40u);
  rejects(/*result_vgpr=*/256u, /*address_vgpr=*/40u);
  rejects(/*result_vgpr=*/42u, /*address_vgpr=*/255u);
  ConSanTargetProfile invalid_target = *target;
  invalid_target.arch = ROCJITSU_CODE_ARCH_INVALID;
  rejects(/*result_vgpr=*/42u, /*address_vgpr=*/40u, &invalid_target);
}

TEST(ConSanMoi, WorkgroupShadowClearPlanCoversEveryTargetProfile) {
  for (const ConSanTargetProfile &target : kConSanTargetProfiles) {
    SCOPED_TRACE(rj_code_target_name(target.target));
    EXPECT_EQ(consan_detail::moi_workgroup_shadow_initialization_lanes(target, std::nullopt), 32u);
    EXPECT_EQ(consan_detail::moi_workgroup_shadow_initialization_lanes(
                  target, std::array<uint32_t, 3>{16u, 1u, 1u}),
              32u);
    EXPECT_EQ(consan_detail::moi_workgroup_shadow_initialization_lanes(
                  target, std::array<uint32_t, 3>{128u, 1u, 1u}),
              target.workgroup_shadow_clear.maximum_lanes);

    const auto plan = consan_detail::plan_moi_workgroup_shadow_clear(
        target, /*initialization_size=*/16u, target.workgroup_shadow_clear.maximum_lanes,
        /*has_quad_zero_tuple=*/true);
    ASSERT_TRUE(plan);
    switch (target.workgroup_shadow_clear.encoding) {
    case ConSanWorkgroupShadowClearEncoding::SplitB32Pair:
      EXPECT_EQ(plan->store_form, consan_detail::MoiWorkgroupShadowClearStoreForm::SplitB32Pair);
      EXPECT_EQ(plan->zero_vgpr_count, 2u);
      break;
    case ConSanWorkgroupShadowClearEncoding::PackedB64:
      EXPECT_EQ(plan->store_form, consan_detail::MoiWorkgroupShadowClearStoreForm::PackedB64);
      EXPECT_EQ(plan->zero_vgpr_count, 2u);
      break;
    case ConSanWorkgroupShadowClearEncoding::PackedB128:
      EXPECT_EQ(plan->store_form, consan_detail::MoiWorkgroupShadowClearStoreForm::PackedB128);
      EXPECT_EQ(plan->zero_vgpr_count, 4u);
      break;
    }
    EXPECT_EQ(plan->initialization_lanes, target.workgroup_shadow_clear.maximum_lanes);
    EXPECT_EQ(consan_detail::moi_workgroup_shadow_preferred_zero_vgpr_count(target),
              target.workgroup_shadow_clear.encoding ==
                      ConSanWorkgroupShadowClearEncoding::PackedB128
                  ? 4u
                  : 2u);
  }
}

TEST(ConSanMoi, WorkgroupShadowClearPlanFallsBackAndRejectsMalformedLayouts) {
  const ConSanTargetProfile *target = consan_target_profile(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(target, nullptr);
  const auto missing_quad = consan_detail::plan_moi_workgroup_shadow_clear(
      *target, /*initialization_size=*/16u, /*initialization_lanes=*/64u,
      /*has_quad_zero_tuple=*/false);
  ASSERT_TRUE(missing_quad);
  EXPECT_EQ(missing_quad->store_form, consan_detail::MoiWorkgroupShadowClearStoreForm::PackedB64);
  EXPECT_EQ(missing_quad->zero_vgpr_count, 2u);
  const auto unaligned_range = consan_detail::plan_moi_workgroup_shadow_clear(
      *target, /*initialization_size=*/24u, /*initialization_lanes=*/64u,
      /*has_quad_zero_tuple=*/true);
  ASSERT_TRUE(unaligned_range);
  EXPECT_EQ(unaligned_range->store_form,
            consan_detail::MoiWorkgroupShadowClearStoreForm::PackedB64);

  EXPECT_FALSE(consan_detail::plan_moi_workgroup_shadow_clear(*target, /*initialization_size=*/0u,
                                                              /*initialization_lanes=*/64u, true));
  EXPECT_FALSE(consan_detail::plan_moi_workgroup_shadow_clear(*target, /*initialization_size=*/12u,
                                                              /*initialization_lanes=*/64u, true));
  EXPECT_FALSE(consan_detail::plan_moi_workgroup_shadow_clear(*target, /*initialization_size=*/16u,
                                                              /*initialization_lanes=*/0u, true));
  EXPECT_FALSE(consan_detail::plan_moi_workgroup_shadow_clear(*target, /*initialization_size=*/16u,
                                                              /*initialization_lanes=*/65u, true));

  ConSanTargetProfile malformed = *target;
  malformed.workgroup_shadow_clear.encoding = static_cast<ConSanWorkgroupShadowClearEncoding>(255u);
  EXPECT_FALSE(consan_detail::plan_moi_workgroup_shadow_clear(
      malformed, /*initialization_size=*/16u, /*initialization_lanes=*/64u, true));
}

TEST(ConSanMoi, WorkgroupKeyRegisterPlanRequiresOneUnambiguousExecutionPlan) {
  using consan_detail::MoiWorkgroupKeyRegisterPlan;
  const MoiWorkgroupKeyRegisterPlan missing_exec{};
  const MoiWorkgroupKeyRegisterPlan derived{
      .exec_save_sgpr = 40u,
      .cached_key_vgpr = std::nullopt,
      .cached_key_sgpr = std::nullopt,
  };
  const MoiWorkgroupKeyRegisterPlan vector_cached{
      .exec_save_sgpr = 40u,
      .cached_key_vgpr = 20u,
      .cached_key_sgpr = std::nullopt,
  };
  const MoiWorkgroupKeyRegisterPlan scalar_cached{
      .exec_save_sgpr = 40u,
      .cached_key_vgpr = std::nullopt,
      .cached_key_sgpr = 60u,
  };
  const MoiWorkgroupKeyRegisterPlan ambiguous{
      .exec_save_sgpr = 40u,
      .cached_key_vgpr = 20u,
      .cached_key_sgpr = 60u,
  };
  EXPECT_FALSE(missing_exec.is_well_formed());
  EXPECT_TRUE(derived.is_well_formed());
  EXPECT_TRUE(vector_cached.is_well_formed());
  EXPECT_TRUE(scalar_cached.is_well_formed());
  EXPECT_FALSE(ambiguous.is_well_formed());
}

[[nodiscard]] constexpr bool sgpr_ranges_overlap(uint16_t base, uint16_t width, uint16_t other_base,
                                                 uint16_t other_width) {
  return base < static_cast<uint32_t>(other_base) + other_width &&
         other_base < static_cast<uint32_t>(base) + width;
}

TEST(ConSanMoi, SampledAtomicPhysicalAliasRejectsEverySemanticMismatch) {
  using Semantics = consan_detail::SampledAtomicSemantics;
  struct Candidate {
    uint64_t file_offset = 24u;
    std::string container_name;
    Semantics semantics;
  };
  const Semantics baseline{
      .role = ConSanMoiSampledSyncRole::RmwRelease,
      .scope = ConSanMoiSampledSyncScope::Agent,
      .outcome = ConSanMoiSampledSyncOutcome::RmwNoReturn,
      .byte_count = 4u,
      .descriptor = 0x123u,
      .cas_failure_descriptor = std::nullopt,
  };
  std::array<Semantics, 6> mismatches;
  mismatches.fill(baseline);
  mismatches[0].role = ConSanMoiSampledSyncRole::RmwAcquire;
  mismatches[1].scope = ConSanMoiSampledSyncScope::System;
  mismatches[2].outcome = ConSanMoiSampledSyncOutcome::RmwReturnsOld;
  mismatches[3].byte_count = 8u;
  mismatches[4].descriptor = 0x456u;
  mismatches[5].cas_failure_descriptor = 0x789u;

  for (size_t mismatch_index = 0; mismatch_index < mismatches.size(); ++mismatch_index) {
    SCOPED_TRACE(mismatch_index);
    std::vector<Candidate> candidates = {
        {.container_name = "first", .semantics = baseline},
        {.container_name = "conflict", .semantics = mismatches[mismatch_index]},
    };
    std::vector<std::string> errors;

    EXPECT_FALSE(consan_detail::canonicalize_physical_site_aliases(
        candidates, errors, "ConSan MOI sampled atomic site",
        [](const Candidate &candidate) { return candidate.file_offset; },
        [](const Candidate &candidate) -> std::string_view { return candidate.container_name; },
        [](const Candidate &lhs, const Candidate &rhs) { return lhs.semantics == rhs.semantics; }));

    ASSERT_EQ(errors.size(), 1u);
    EXPECT_EQ(errors.front(),
              "ConSan MOI sampled atomic site at file offset 24 was decoded inconsistently "
              "through aliases 'first' and 'conflict'");
  }
}

TEST(ConSanMoi, WorkgroupSourceRequiresExactlyOneOperandKind) {
  const ConSanMoiWorkgroupSource absent;
  EXPECT_FALSE(absent.has_value());
  EXPECT_TRUE(absent.is_well_formed());
  EXPECT_FALSE(absent.operand());

  const ConSanMoiWorkgroupSource scalar = ConSanMoiWorkgroupSource::scalar(17u);
  EXPECT_TRUE(scalar.has_value());
  EXPECT_TRUE(scalar.is_well_formed());
  EXPECT_EQ(scalar.operand(), 17u);

  const ConSanMoiWorkgroupSource vector = ConSanMoiWorkgroupSource::vector(23u);
  EXPECT_TRUE(vector.has_value());
  EXPECT_TRUE(vector.is_well_formed());
  EXPECT_EQ(vector.operand(), vector_source_vgpr(23u));

  ConSanMoiWorkgroupSource ambiguous = scalar;
  ambiguous.vector_src = 23u;
  EXPECT_FALSE(ambiguous.has_value());
  EXPECT_FALSE(ambiguous.is_well_formed());
  EXPECT_FALSE(ambiguous.operand());
}

TEST(ConSanMoi, FullWorkgroupPayloadRequirementUsesMoiPatchSemantics) {
  ConSanTransformArtifacts result;
  result.observation_plan.engine = ConSanCapabilityEngine::RecordReplay;
  ConSanPatchInfo patch;
  patch.owner_descriptor_file_offsets.push_back(64u);
  patch.kind = ConSanPatchKind::InlineMoiAccessRecordStore;

  EXPECT_TRUE(consan_detail::patch_requires_full_workgroup_id_payload(
      result.observation_plan.engine, ROCJITSU_CODE_ARCH_RDNA4, patch));
  EXPECT_TRUE(consan_detail::patch_requires_full_workgroup_id_payload(
      result.observation_plan.engine, ROCJITSU_CODE_ARCH_CDNA5, patch));
  EXPECT_FALSE(consan_detail::patch_requires_full_workgroup_id_payload(
      result.observation_plan.engine, ROCJITSU_CODE_ARCH_CDNA4, patch));
  EXPECT_FALSE(consan_detail::patch_requires_full_workgroup_id_payload(
      result.observation_plan.engine, ROCJITSU_CODE_ARCH_INVALID, patch));

  patch.kind = ConSanPatchKind::InlineMalformedBarrierAbort;
  EXPECT_FALSE(consan_detail::patch_requires_full_workgroup_id_payload(
      result.observation_plan.engine, ROCJITSU_CODE_ARCH_RDNA4, patch));
  patch.kind = ConSanPatchKind::TrampolineMoiIndirectBranchIsland;
  EXPECT_TRUE(consan_detail::patch_requires_full_workgroup_id_payload(
      result.observation_plan.engine, ROCJITSU_CODE_ARCH_RDNA4, patch));

  patch.kind = ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue;
  EXPECT_FALSE(consan_detail::patch_requires_full_workgroup_id_payload(
      result.observation_plan.engine, ROCJITSU_CODE_ARCH_CDNA4, patch));
  patch.persistent_sgpr_state.record_replay_workgroup = {.x = 20u, .y = 21u, .z = 22u};
  EXPECT_TRUE(consan_detail::patch_requires_full_workgroup_id_payload(
      result.observation_plan.engine, ROCJITSU_CODE_ARCH_CDNA4, patch));
  patch.persistent_sgpr_state = {};
  patch.persistent_record_replay_workgroup_vgprs = {.x = 20u, .y = 21u, .z = 22u};
  EXPECT_TRUE(consan_detail::patch_requires_full_workgroup_id_payload(
      result.observation_plan.engine, ROCJITSU_CODE_ARCH_CDNA4, patch));
  patch.persistent_record_replay_workgroup_vgprs = {};
  patch.persistent_record_replay_workgroup_private_offsets = {.x = 0u, .y = 4u, .z = 8u};
  EXPECT_TRUE(consan_detail::patch_requires_full_workgroup_id_payload(
      result.observation_plan.engine, ROCJITSU_CODE_ARCH_CDNA3, patch));

  result.observation_plan.engine = ConSanCapabilityEngine::SuperCollider;
  EXPECT_FALSE(consan_detail::patch_requires_full_workgroup_id_payload(
      result.observation_plan.engine, ROCJITSU_CODE_ARCH_RDNA4, patch));
  result.observation_plan.engine = ConSanCapabilityEngine::RecordReplay;
  patch.owner_descriptor_file_offsets.clear();
  EXPECT_FALSE(consan_detail::patch_requires_full_workgroup_id_payload(
      result.observation_plan.engine, ROCJITSU_CODE_ARCH_RDNA4, patch));
}

TEST(ConSanMoi, DynamicRecordAddressExecutesExactStrideOnEveryTarget) {
  constexpr uint16_t kAddressVgpr = 40u;
  constexpr uint16_t kSlotVgpr = 42u;
  constexpr std::array<uint16_t, 3> kUnrelatedVgprs = {43u, 44u, 45u};
  constexpr std::array<uint32_t, 3> kUnrelatedValues = {0x12345678u, 0x90abcdefu, 0x55aa55aau};
  // Real record strides plus 64, the power-of-two case where no accumulation
  // add is needed.
  constexpr std::array<uint32_t, 5> kStrideBytes = {
      sizeof(ConSanMoiBarrierRecord), sizeof(ConSanMoiFenceRecord),      64u,
      sizeof(ConSanMoiAccessRecord),  sizeof(ConSanMoiDiagnosticRecord),
  };
  constexpr std::array<uint32_t, 4> kSlots = {0u, 1u, 3u, 1000u};
  constexpr std::array<uint64_t, 3> kFieldAddresses = {
      0x1234567800000010ull,
      0x00000000ffffff00ull,
      0x12345678fffffff0ull,
  };
  size_t case_index = 0;
  for (const ConSanTargetProfile &target : kConSanTargetProfiles) {
    const rj_code_arch_t arch = target.arch;
    for (uint32_t stride_bytes : kStrideBytes) {
      for (uint32_t slot : kSlots) {
        for (uint64_t field_address : kFieldAddresses) {
          SCOPED_TRACE("arch=" + std::to_string(static_cast<uint32_t>(arch)) +
                       " stride=" + std::to_string(stride_bytes) + " slot=" + std::to_string(slot) +
                       " base=" + std::to_string(field_address));
          std::vector<uint32_t> words;
          ASSERT_TRUE(
              consan_detail::append_dynamic_record_address(words,
                                                           {
                                                               .field_address = field_address,
                                                               .stride_bytes = stride_bytes,
                                                               .address_vgpr = kAddressVgpr,
                                                               .slot_vgpr = kSlotVgpr,
                                                           },
                                                           target));

          const std::string component =
              "consan_dynamic_record_address_" + std::to_string(case_index++);
          amdgpu::GpuMemory memory(component + "_memory");
          amdgpu::L2Cache cache(component + "_cache");
          cache.set_backing_memory(&memory);
          amdgpu::ComputeUnitCore::Config config{};
          config.arch = arch;
          config.num_wf_slots = 1;
          config.sgprs_per_wf = 128;
          config.vgprs_per_wf = 256;
          config.lds_size_kb = 64;
          auto compute_unit = amdgpu::ComputeUnitCore::create(component, config, &memory, &cache);
          ASSERT_NE(compute_unit, nullptr);
          amdgpu::Wavefront *wave =
              compute_unit->dispatch_wf(0, 0, config.sgprs_per_wf, config.vgprs_per_wf);
          ASSERT_NE(wave, nullptr);
          for (size_t index = 0; index < words.size(); ++index)
            memory.write32(index * sizeof(uint32_t), words[index]);
          wave->pc = 0u;
          wave->set_exec(1u);
          const uint32_t vgpr_base = wave->vgpr_alloc().base;
          compute_unit->write_vgpr(vgpr_base + kSlotVgpr, 0u, slot);
          for (size_t index = 0; index < kUnrelatedVgprs.size(); ++index)
            compute_unit->write_vgpr(vgpr_base + kUnrelatedVgprs[index], 0u,
                                     kUnrelatedValues[index]);

          size_t steps = 0;
          while (wave->pc < words.size() * sizeof(uint32_t)) {
            ASSERT_LT(steps, words.size());
            ++steps;
            compute_unit->step();
          }

          const uint64_t expected = field_address + static_cast<uint64_t>(slot) * stride_bytes;
          const uint64_t actual =
              (static_cast<uint64_t>(compute_unit->read_vgpr(vgpr_base + kAddressVgpr + 1u, 0u))
               << 32u) |
              compute_unit->read_vgpr(vgpr_base + kAddressVgpr, 0u);
          EXPECT_EQ(actual, expected);
          EXPECT_EQ(compute_unit->read_vgpr(vgpr_base + kSlotVgpr, 0u), slot);
          for (size_t index = 0; index < kUnrelatedVgprs.size(); ++index) {
            EXPECT_EQ(compute_unit->read_vgpr(vgpr_base + kUnrelatedVgprs[index], 0u),
                      kUnrelatedValues[index]);
          }
          if (!wave->is_halted())
            wave->halt();
        }
      }
    }
  }
}

TEST(ConSanMoi, DynamicRecordAddressRejectsInvalidRegistersWithoutPartialOutput) {
  constexpr uint64_t kFieldAddress = 0x1234567800000010ull;
  constexpr uint16_t kAddressVgpr = 40u;
  constexpr uint16_t kSlotVgpr = 42u;
  constexpr uint32_t kStrideBytes = sizeof(ConSanMoiAccessRecord);
  const ConSanTargetProfile *target = consan_target_profile(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(target, nullptr);
  const auto rejects_without_appending = [&](uint32_t stride_bytes, uint16_t address_vgpr,
                                             uint16_t slot_vgpr,
                                             const ConSanTargetProfile &request_target) {
    std::vector<uint32_t> words = {0x12345678u};
    EXPECT_FALSE(consan_detail::append_dynamic_record_address(words,
                                                              {
                                                                  .field_address = kFieldAddress,
                                                                  .stride_bytes = stride_bytes,
                                                                  .address_vgpr = address_vgpr,
                                                                  .slot_vgpr = slot_vgpr,
                                                              },
                                                              request_target));
    EXPECT_EQ(words, std::vector<uint32_t>{0x12345678u});
  };

  rejects_without_appending(0u, kAddressVgpr, kSlotVgpr, *target);
  rejects_without_appending(kStrideBytes, kAddressVgpr, kAddressVgpr, *target);
  rejects_without_appending(kStrideBytes, kAddressVgpr, static_cast<uint16_t>(kAddressVgpr + 1u),
                            *target);
  rejects_without_appending(kStrideBytes, 255u, kSlotVgpr, *target);
  rejects_without_appending(kStrideBytes, kAddressVgpr, 256u, *target);
  ConSanTargetProfile invalid_target = *target;
  invalid_target.arch = ROCJITSU_CODE_ARCH_INVALID;
  rejects_without_appending(kStrideBytes, kAddressVgpr, kSlotVgpr, invalid_target);
}

TEST(ConSanMoi, SpilledVgprReloadSelectsFixedAndDynamicTargetEncodings) {
  constexpr uint16_t kSpillBase = 8u;
  constexpr uint16_t kSource = 9u;
  constexpr uint16_t kDestination = 40u;
  constexpr uint16_t kFrameBaseSgpr = 33u;
  constexpr uint32_t kSlotOffset = 20u;
  constexpr std::array kArchitectures = {
      ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4,
      ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5,
  };

  for (rj_code_arch_t arch : kArchitectures) {
    SCOPED_TRACE(arch);
    VgprSpillSequence fixed{
        .vgpr_base = kSpillBase,
        .vgpr_count = 2u,
        .slot_offsets = {16u, kSlotOffset},
        .save_words = {},
        .restore_words = {},
        .total_private_bytes = 24u,
        .uses_dynamic_stack_frame = false,
        .dynamic_frame_base_sgpr = 0u,
        .dynamic_frame_bytes = 0u,
    };
    std::vector<uint32_t> fixed_words;
    EXPECT_EQ(consan_detail::append_reload_moi_spilled_vgpr(fixed_words, fixed, kDestination,
                                                            kSource, arch),
              consan_detail::MoiSpilledVgprReloadResult::Appended);
    const auto fixed_load =
        instrumentation::build_private_load_b32(kDestination, kSlotOffset, arch);
    const auto wait = instrumentation::build_s_wait_private_load0(arch);
    ASSERT_TRUE(fixed_load && wait);
    std::vector<uint32_t> expected_fixed(fixed_load->begin(), fixed_load->end());
    expected_fixed.push_back(*wait);
    EXPECT_EQ(fixed_words, expected_fixed);

    VgprSpillSequence dynamic = fixed;
    dynamic.uses_dynamic_stack_frame = true;
    dynamic.dynamic_frame_base_sgpr = kFrameBaseSgpr;
    std::vector<uint32_t> dynamic_words;
    EXPECT_EQ(consan_detail::append_reload_moi_spilled_vgpr(dynamic_words, dynamic, kDestination,
                                                            kSource, arch),
              consan_detail::MoiSpilledVgprReloadResult::Appended);
    std::vector<uint32_t> expected_dynamic;
    if (arch == ROCJITSU_CODE_ARCH_RDNA3) {
      const auto load =
          build_rdna3_scratch_load_b32_saddr(kDestination, kFrameBaseSgpr, kSlotOffset, arch);
      ASSERT_TRUE(load);
      expected_dynamic.assign(load->begin(), load->end());
    } else if (arch == ROCJITSU_CODE_ARCH_CDNA3) {
      const auto load =
          build_cdna3_scratch_load_b32_saddr(kDestination, kFrameBaseSgpr, kSlotOffset, arch);
      ASSERT_TRUE(load);
      expected_dynamic.assign(load->begin(), load->end());
    } else if (arch == ROCJITSU_CODE_ARCH_CDNA4) {
      const auto load =
          build_cdna4_scratch_load_b32_saddr(kDestination, kFrameBaseSgpr, kSlotOffset, arch);
      ASSERT_TRUE(load);
      expected_dynamic.assign(load->begin(), load->end());
    } else {
      const auto load =
          build_scratch_load_b32_saddr(kDestination, kFrameBaseSgpr, kSlotOffset, arch);
      ASSERT_TRUE(load);
      expected_dynamic.assign(load->begin(), load->end());
    }
    expected_dynamic.push_back(*wait);
    EXPECT_EQ(dynamic_words, expected_dynamic);
  }
}

TEST(ConSanMoi, SpilledVgprReloadClassifiesRejectedRequestsWithoutAppending) {
  using Result = consan_detail::MoiSpilledVgprReloadResult;
  const auto rejects = [](const VgprSpillSequence &spill, uint16_t destination, uint16_t source,
                          rj_code_arch_t arch, Result expected) {
    std::vector<uint32_t> words = {0x12345678u};
    EXPECT_EQ(
        consan_detail::append_reload_moi_spilled_vgpr(words, spill, destination, source, arch),
        expected);
    EXPECT_EQ(words, std::vector<uint32_t>{0x12345678u});
  };

  const VgprSpillSequence complete{
      .vgpr_base = 8u,
      .vgpr_count = 2u,
      .slot_offsets = {16u, 20u},
      .save_words = {},
      .restore_words = {},
      .total_private_bytes = 24u,
      .uses_dynamic_stack_frame = false,
      .dynamic_frame_base_sgpr = 0u,
      .dynamic_frame_bytes = 0u,
  };
  rejects(complete, /*destination=*/40u, /*source=*/7u, ROCJITSU_CODE_ARCH_CDNA4,
          Result::SourceOutsideWindow);
  VgprSpillSequence incomplete = complete;
  incomplete.slot_offsets.pop_back();
  rejects(incomplete, /*destination=*/40u, /*source=*/9u, ROCJITSU_CODE_ARCH_CDNA4,
          Result::IncompleteSlotMetadata);
  rejects(complete, /*destination=*/256u, /*source=*/9u, ROCJITSU_CODE_ARCH_CDNA4,
          Result::UnsupportedEncoding);
  rejects(complete, /*destination=*/40u, /*source=*/9u, ROCJITSU_CODE_ARCH_INVALID,
          Result::UnsupportedEncoding);

  constexpr std::array kUnsupportedArchitectures = {
      ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2,   ROCJITSU_CODE_ARCH_RDNA1,
      ROCJITSU_CODE_ARCH_RDNA2, ROCJITSU_CODE_ARCH_RDNA3_5,
  };
  for (rj_code_arch_t arch : kUnsupportedArchitectures) {
    SCOPED_TRACE(arch);
    rejects(complete, /*destination=*/40u, /*source=*/9u, arch, Result::UnsupportedEncoding);
    VgprSpillSequence dynamic = complete;
    dynamic.uses_dynamic_stack_frame = true;
    dynamic.dynamic_frame_base_sgpr = 33u;
    rejects(dynamic, /*destination=*/40u, /*source=*/9u, arch, Result::UnsupportedEncoding);
  }
}

TEST(ConSanMoi, SpilledVgprReloadResultNamesAreDistinctAndComplete) {
  using Result = consan_detail::MoiSpilledVgprReloadResult;
  constexpr std::array kResults = {
      Result::Appended,
      Result::SourceOutsideWindow,
      Result::IncompleteSlotMetadata,
      Result::UnsupportedEncoding,
  };
  std::set<std::string_view> names;
  for (Result result : kResults) {
    const std::string_view name = consan_detail::moi_spilled_vgpr_reload_result_name(result);
    EXPECT_NE(name, "unknown");
    EXPECT_TRUE(names.insert(name).second) << name;
  }
}

TEST(ConSanMoi, ResourcePlanAlternativeNamesAreExhaustiveAndDistinct) {
  EXPECT_EQ(ConSanResourcePlanAlternative{}.kind,
            ConSanResourcePlanAlternativeKind::GuestOperandOverlapSpill);
  EXPECT_STREQ(consan_resource_plan_alternative_kind_name(
                   ConSanResourcePlanAlternativeKind::GuestOperandOverlapSpill),
               "guest_operand_overlap_spill");
  EXPECT_STREQ(consan_resource_plan_alternative_kind_name(
                   ConSanResourcePlanAlternativeKind::SpillBackedOperandRecovery),
               "spill_backed_operand_recovery");
  EXPECT_STREQ(
      consan_resource_plan_alternative_outcome_name(ConSanResourcePlanAlternativeOutcome::Selected),
      "selected");
  EXPECT_STREQ(
      consan_resource_plan_alternative_outcome_name(ConSanResourcePlanAlternativeOutcome::Rejected),
      "rejected");
  EXPECT_STREQ(consan_resource_plan_alternative_outcome_name(
                   ConSanResourcePlanAlternativeOutcome::Superseded),
               "superseded");
  EXPECT_STREQ(consan_resource_plan_alternative_outcome_name(
                   ConSanResourcePlanAlternativeOutcome::Contributed),
               "contributed");
  EXPECT_STREQ(
      consan_resource_plan_alternative_outcome_name(ConSanResourcePlanAlternativeOutcome::Vetoed),
      "vetoed");
}

TEST(ConSanMoi, ResourcePlanAlternativeOutcomeTracksFinalPlanVeto) {
  ConSanCandidateResourcePlan plan;
  plan.source = ConSanRegisterAllocationSource::SpillRequired;
  plan.reason = ConSanRegisterPlanReason::None;
  plan.scratch_vgpr_count = 16u;
  const ConSanResourcePlanAlternative alternative{
      .kind = ConSanResourcePlanAlternativeKind::SpillBackedOperandRecovery,
      .source = ConSanRegisterAllocationSource::SpillRequired,
      .reason = ConSanRegisterPlanReason::None,
      .scratch_vgpr_count = 16u,
      .outcome = ConSanResourcePlanAlternativeOutcome::Selected,
  };
  EXPECT_EQ(consan_resource_plan_alternative_outcome(plan, alternative),
            ConSanResourcePlanAlternativeOutcome::Selected);

  plan.source = ConSanRegisterAllocationSource::Unsupported;
  plan.reason = ConSanRegisterPlanReason::ForbiddenOverlap;
  plan.scratch_vgpr.reset();
  EXPECT_EQ(consan_resource_plan_alternative_outcome(plan, alternative),
            ConSanResourcePlanAlternativeOutcome::Vetoed);

  ConSanResourcePlanAlternative rejected = alternative;
  rejected.source = ConSanRegisterAllocationSource::Unsupported;
  rejected.reason = ConSanRegisterPlanReason::NoLegalWindow;
  rejected.outcome = ConSanResourcePlanAlternativeOutcome::Rejected;
  EXPECT_EQ(consan_resource_plan_alternative_outcome(plan, rejected),
            ConSanResourcePlanAlternativeOutcome::Rejected);
}

TEST(ConSanMoi, PrivateWorkgroupSourceAppliesPackedCoordinateExtraction) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  constexpr uint16_t kValueVgpr = 12u;
  constexpr uint32_t kPrivateOffset = 36u;
  for (const auto &[mask_low, extract_high] : {std::pair{true, false}, std::pair{false, true}}) {
    SCOPED_TRACE("mask_low=" + std::to_string(mask_low) +
                 " shift_right=" + std::to_string(extract_high));
    ConSanMoiWorkgroupSource source = ConSanMoiWorkgroupSource::private_state(kPrivateOffset);
    source.mask_low_16 = mask_low;
    source.shift_right_16 = extract_high;
    std::vector<uint32_t> words;

    ASSERT_TRUE(consan_detail::append_workgroup_source_value(words, source, kValueVgpr, kArch));

    const auto load = instrumentation::build_private_load_b32(kValueVgpr, kPrivateOffset, kArch);
    const auto shift_left = instrumentation::build_v_lshlrev_b32(
        kValueVgpr, scalar_positive_inline_u32(16), kValueVgpr, kArch);
    const auto shift_right = instrumentation::build_v_lshrrev_b32(
        kValueVgpr, scalar_positive_inline_u32(16), kValueVgpr, kArch);
    ASSERT_TRUE(load && shift_left && shift_right);
    EXPECT_TRUE(contains_subsequence(words, *load));
    if (mask_low) {
      EXPECT_NE(std::ranges::find(words, *shift_left), words.end());
      EXPECT_NE(std::ranges::find(words, *shift_right), words.end());
    } else {
      EXPECT_EQ(std::ranges::find(words, *shift_left), words.end());
      EXPECT_NE(std::ranges::find(words, *shift_right), words.end());
    }
  }
}

TEST(ConSanMoi, ScalarPersistentTemporaryValidationIsNoopWhenDisabled) {
  MoiOptions disabled;
  std::vector<std::string> errors;
  EXPECT_TRUE(consan_detail::validate_scalar_state_temporaries(disabled, "test consumer", errors));
  EXPECT_TRUE(errors.empty());
}

TEST(ConSanMoi, ScalarPersistentTemporaryValidationFailsClosed) {
  for (uint32_t present_mask = 0u; present_mask < 3u; ++present_mask) {
    SCOPED_TRACE(present_mask);
    MoiOptions options;
    options.moi_persistent_sgprs.owner = 40u;
    options.moi_persistent_sgprs.epoch = 41u;
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

  MoiOptions valid;
  valid.moi_persistent_sgprs.owner = 40u;
  valid.moi_persistent_sgprs.epoch = 41u;
  valid.moi_owner_vgpr = 6u;
  valid.moi_epoch_vgpr = 7u;
  std::vector<std::string> errors;
  EXPECT_TRUE(consan_detail::validate_scalar_state_temporaries(valid, "test consumer", errors));
  EXPECT_TRUE(errors.empty());
}

TEST(ConSanMoi, ScalarOwnerContextResolutionFailsClosedAndComputesTailFloor) {
  using Summary = consan_detail::ScalarOwnerContextSummary;
  struct TailFloorCase {
    uint16_t current;
    uint16_t max_referenced;
    bool indirect;
    bool coverage_complete;
    uint32_t expected;
  };
  constexpr std::array kTailFloorCases = {
      TailFloorCase{80u, 72u, false, true, 72u},  TailFloorCase{40u, 48u, false, true, 48u},
      TailFloorCase{80u, 72u, true, true, 80u},   TailFloorCase{40u, 48u, true, true, 48u},
      TailFloorCase{80u, 72u, false, false, 80u}, TailFloorCase{40u, 48u, false, false, 48u},
  };
  constexpr std::array<uint64_t, 1> kSingleOwner = {0x10u};
  for (const TailFloorCase &test_case : kTailFloorCases) {
    SCOPED_TRACE(testing::PrintToString(test_case.expected));
    const std::array context = {
        Summary{.descriptor_file_offset = 0x10u,
                .current_sgpr_count = test_case.current,
                .max_referenced_sgpr_count = test_case.max_referenced,
                .has_indirect_sgpr_access = test_case.indirect,
                .sgpr_reference_coverage_complete = test_case.coverage_complete,
                .descriptor_valid = true},
    };
    const auto resolved = consan_detail::resolve_scalar_owner_contexts(true, context, kSingleOwner);
    ASSERT_TRUE(resolved);
    EXPECT_EQ(resolved->tail_floor, test_case.expected);
  }

  const std::array contexts = {
      Summary{.descriptor_file_offset = 0x10u,
              .current_sgpr_count = 40u,
              .max_referenced_sgpr_count = 48u,
              .has_indirect_sgpr_access = false,
              .sgpr_reference_coverage_complete = true,
              .descriptor_valid = true},
      Summary{.descriptor_file_offset = 0x20u,
              .current_sgpr_count = 80u,
              .max_referenced_sgpr_count = 72u,
              .has_indirect_sgpr_access = true,
              .sgpr_reference_coverage_complete = true,
              .descriptor_valid = true},
  };
  constexpr std::array<uint64_t, 2> kOwners = {0x20u, 0x10u};

  const auto resolved = consan_detail::resolve_scalar_owner_contexts(true, contexts, kOwners);

  ASSERT_TRUE(resolved);
  EXPECT_EQ(resolved->context_indices, (std::vector<size_t>{1u, 0u}));
  EXPECT_EQ(resolved->tail_floor, 80u);

  std::array direct_only_contexts = contexts;
  direct_only_contexts[1].has_indirect_sgpr_access = false;
  // A closed direct-only owner may reclaim allocated but unreferenced SGPRs.
  const auto direct_only =
      consan_detail::resolve_scalar_owner_contexts(true, direct_only_contexts, kOwners);
  ASSERT_TRUE(direct_only);
  EXPECT_EQ(direct_only->tail_floor, 72u);

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

TEST(ConSanMoi, ScalarOwnerWindowQualificationUsesEveryTailAndPhysicalVcc) {
  using Range = consan_detail::ScalarOwnerSgprRange;
  using Summary = consan_detail::ScalarOwnerContextSummary;
  const std::array direct_owners = {
      Summary{.descriptor_file_offset = 0x10u,
              .current_sgpr_count = 32u,
              .max_referenced_sgpr_count = 12u,
              .sgpr_reference_coverage_complete = true,
              .descriptor_valid = true},
      Summary{.descriptor_file_offset = 0x20u,
              .current_sgpr_count = 48u,
              .max_referenced_sgpr_count = 12u,
              .sgpr_reference_coverage_complete = true,
              .descriptor_valid = true},
  };
  const std::array misses_both_vcc{Range{12u, 2u}};
  const std::array dispatch_hits_second_vcc{Range{42u, 2u}};
  const std::array hits_second_vcc{Range{36u, 30u}};

  EXPECT_FALSE(consan_detail::scalar_owner_contexts_conflict_with_physical_vcc(direct_owners,
                                                                               misses_both_vcc));
  EXPECT_TRUE(consan_detail::scalar_owner_contexts_conflict_with_physical_vcc(
      direct_owners, dispatch_hits_second_vcc));
  EXPECT_TRUE(consan_detail::scalar_owner_contexts_conflict_with_physical_vcc(direct_owners,
                                                                              hits_second_vcc));
  EXPECT_TRUE(consan_detail::scalar_owner_contexts_admit_reserved_window(
      direct_owners, 12u, 2u, /*protect_physical_vcc=*/true));

  std::array indirect_owners = direct_owners;
  indirect_owners[1].has_indirect_sgpr_access = true;
  EXPECT_FALSE(consan_detail::scalar_owner_contexts_admit_reserved_window(
      indirect_owners, 12u, 2u, /*protect_physical_vcc=*/false));
  EXPECT_TRUE(consan_detail::scalar_owner_contexts_admit_reserved_window(
      indirect_owners, 48u, 2u, /*protect_physical_vcc=*/false));

  std::array invalid_owners = direct_owners;
  invalid_owners[1].descriptor_valid = false;
  EXPECT_TRUE(consan_detail::scalar_owner_contexts_conflict_with_physical_vcc(invalid_owners,
                                                                              misses_both_vcc));
  EXPECT_FALSE(consan_detail::scalar_owner_contexts_admit_reserved_window(
      invalid_owners, 48u, 2u, /*protect_physical_vcc=*/false));
  EXPECT_TRUE(consan_detail::scalar_owner_contexts_conflict_with_physical_vcc(
      std::span<const Summary>{}, misses_both_vcc));
  EXPECT_FALSE(consan_detail::scalar_owner_contexts_admit_reserved_window(
      std::span<const Summary>{}, 48u, 2u, /*protect_physical_vcc=*/false));
}

TEST(ConSanMoi, Cdna4HeterogeneousOwnersKeepUsableComponentAcrossMoiEngines) {
  constexpr uint64_t kHighPressureEntry = 320u * sizeof(uint32_t);
  for (ConSanMoiEngine engine :
       {ConSanMoiEngine::RecordReplay, ConSanMoiEngine::Sampled, ConSanMoiEngine::InlineShadow}) {
    for (uint16_t live_sgpr_count : {96u, 98u}) {
      SCOPED_TRACE(testing::PrintToString(engine) +
                   " live_sgprs=" + std::to_string(live_sgpr_count));
      std::vector<uint8_t> bytes =
          make_cdna4_disconnected_scalar_pressure_code_object(live_sgpr_count);
      ASSERT_FALSE(bytes.empty());
      for (std::string_view kernel_name : {"lds_probe", "lds_helper"}) {
        mutate_kernel_descriptor(bytes, kernel_name, [](KD &descriptor) {
          AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2,
                          kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X, 1u);
          AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2,
                          kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y, 1u);
          AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2,
                          kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z, 1u);
        });
      }
      MoiOptions options = moi_options(engine);
      options.moi_track_barriers = false;
      options.moi_track_atomics = false;
      options.moi_runtime_sample_stride = 2u;
      options.moi_report_buffer_address = 0x123456780000ull;
      options.moi_report_buffer_size = engine == ConSanMoiEngine::InlineShadow
                                           ? kInlineShadowFullLdsReportBufferSize
                                           : 64u * 1024u * 1024u;

      const ConSanTransformArtifacts result = test_lower_consan(bytes, options);

      ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
      ASSERT_TRUE(result.modified()) << testing::PrintToString(result.warnings);
      EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
      ASSERT_TRUE(test_moi_dispatch_id_sgpr(result));
      ASSERT_TRUE(test_moi_exec_save_sgpr(result));
      EXPECT_EQ(*test_moi_dispatch_id_sgpr(result), 96u);
      EXPECT_LT(*test_moi_exec_save_sgpr(result), 96u);

      const ConSanIntentCoverageEntry *low = consan_access_coverage_at(result, 0u);
      const ConSanIntentCoverageEntry *high = consan_access_coverage_at(result, kHighPressureEntry);
      ASSERT_NE(low, nullptr);
      ASSERT_NE(high, nullptr);
      EXPECT_EQ(low->lowering, ConSanLoweringOutcomeKind::Instrumented);
      EXPECT_EQ(high->lowering, ConSanLoweringOutcomeKind::ResourceRejected);
      const auto high_plan = std::ranges::find(result.resource_plans, kHighPressureEntry,
                                               &ConSanCandidateResourcePlan::text_offset);
      ASSERT_NE(high_plan, result.resource_plans.end());
      EXPECT_EQ(high_plan->reason, ConSanRegisterPlanReason::ForbiddenOverlap);
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
  constexpr auto store = cdna5::build_vds(cdna5::kDsStoreB32Vds, {.addr = 0, .data0 = 1});
  const std::array<uint32_t, 3> text_words = {store[0], store[1],
                                              build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5)};
  const std::vector<uint8_t> bytes =
      make_gfx1250_code_object(text_words, "lds_probe", /*vgpr_granulated=*/4);
  MoiOptions options = moi_options(ConSanMoiEngine::RecordReplay);

  const ConSanTransformArtifacts result = test_lower_consan(bytes, options);

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
  MoiOptions options = moi_options(ConSanMoiEngine::RecordReplay);

  const ConSanTransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().current_vgpr_count, 32u);
}

TEST(ConSanMoi, Gfx1250TwoAddressLoadUsesNormalizedRangesAndSafeScratch) {
  constexpr auto load = cdna5::build_vds(cdna5::kDsLoad2addrStride64B32Vds,
                                         {.offset0 = 3, .offset1 = 5, .addr = 0, .vdst = 1});
  const std::array<uint32_t, 3> text_words = {load[0], load[1],
                                              build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5)};
  const std::vector<uint8_t> bytes = make_gfx1250_code_object(text_words, "two_address_load");
  MoiOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(2, 0, 0, 0);

  const ConSanTransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified()) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.program_inventory.access_sites().size(), 1u);
  ASSERT_EQ(test_admitted_accesses(result).size(), 1u);
  const ConSanAccessInventorySite &site = result.program_inventory.access_sites().front();
  ConSanMoiCandidate candidate;
  static_cast<ConSanAccessInventorySite &>(candidate) = test_admitted_accesses(result).front();
  ASSERT_EQ(site.ranges.size(), 2u);
  EXPECT_EQ(candidate.ranges, site.ranges);
  for (size_t range_index = 0; range_index < site.ranges.size(); ++range_index) {
    SCOPED_TRACE(range_index);
    ASSERT_TRUE(site.ranges[range_index].static_byte_offset);
    EXPECT_EQ(candidate.lowering_offset(candidate.ranges[range_index]),
              *site.ranges[range_index].static_byte_offset);
  }
  EXPECT_EQ(candidate.lowering_offset(candidate.ranges[0]), 3u * 256u);
  EXPECT_EQ(candidate.lowering_offset(candidate.ranges[1]), 5u * 256u);

  const ConSanTargetProfile *gfx1250 = consan_target_profile(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(gfx1250, nullptr);
  EXPECT_TRUE(
      consan_detail::moi_guest_access_relocation_requires_adjusted_address(candidate, *gfx1250));
  std::vector<std::string> relocation_errors;
  const auto relocated =
      consan_detail::build_moi_relocated_guest_access_words({.image = bytes,
                                                             .candidate = &candidate,
                                                             .target = gfx1250,
                                                             .replay_address_vgpr = 20u,
                                                             .adjusted_address_vgpr = std::nullopt},
                                                            relocation_errors);
  ASSERT_TRUE(relocated) << testing::PrintToString(relocation_errors);
  const auto first = cdna5::build_vds(cdna5::kDsLoadB32Vds,
                                      {.offset0 = 0u, .offset1 = 3u, .addr = 20u, .vdst = 1u});
  const auto second = cdna5::build_vds(cdna5::kDsLoadB32Vds,
                                       {.offset0 = 0u, .offset1 = 5u, .addr = 20u, .vdst = 2u});
  const std::vector<uint32_t> expected_relocation = {first[0], first[1], second[0], second[1]};
  EXPECT_EQ(*relocated, expected_relocation);

  for (const ConSanTargetProfile &target : kConSanTargetProfiles) {
    if (target.requires_split_two_address_lds_relocation)
      continue;
    SCOPED_TRACE(rj_code_target_name(target.target));
    EXPECT_FALSE(
        consan_detail::moi_guest_access_relocation_requires_adjusted_address(candidate, target));
    relocation_errors.clear();
    const auto copied = consan_detail::build_moi_relocated_guest_access_words(
        {.image = bytes,
         .candidate = &candidate,
         .target = &target,
         .replay_address_vgpr = 20u,
         .adjusted_address_vgpr = std::nullopt},
        relocation_errors);
    ASSERT_TRUE(copied) << testing::PrintToString(relocation_errors);
    EXPECT_EQ(*copied, (std::vector<uint32_t>{load[0], load[1]}));
  }
  ASSERT_EQ(result.resource_plans.size(), 1u);
  ASSERT_TRUE(result.resource_plans.front().scratch_vgpr);
  EXPECT_GE(*result.resource_plans.front().scratch_vgpr, 3u);
  ASSERT_EQ(non_entry_prologue_patch_count(result), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::TrampolineMoiAccessRecordStore);
}

TEST(ConSanMoi, GuestAccessRelocationRejectsIncompleteTypedRequest) {
  std::vector<std::string> errors;
  EXPECT_FALSE(
      consan_detail::build_moi_relocated_guest_access_words({.image = {},
                                                             .candidate = nullptr,
                                                             .target = nullptr,
                                                             .replay_address_vgpr = 0u,
                                                             .adjusted_address_vgpr = std::nullopt},
                                                            errors));
  ASSERT_EQ(errors.size(), 1u);
  EXPECT_EQ(errors.front(), "ConSan MOI guest relocation requires a candidate and target profile");
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
      MoiOptions options = moi_options(engine);
      options.moi_track_atomics = false;
      options.moi_track_barriers = false;
      options.moi_runtime_sample_stride = engine == ConSanMoiEngine::Sampled ? 2u : 1u;
      options.moi_report_buffer_address = 0x123456780000ull;
      options.moi_report_buffer_size = engine == ConSanMoiEngine::InlineShadow
                                           ? kInlineShadowFullLdsReportBufferSize
                                           : 64u * 1024u * 1024u;

      const ConSanTransformArtifacts result = test_lower_consan(bytes, options);

      ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
      ASSERT_TRUE(result.modified()) << testing::PrintToString(result.warnings);
      ASSERT_EQ(test_admitted_accesses(result).size(), 1u);
      EXPECT_EQ(test_admitted_accesses(result).front().mnemonic, mnemonic);
      EXPECT_EQ(test_admitted_accesses(result).front().decoded_width_bits, 96u);
      const ConSanPatchKind expected_patch_kind =
          engine == ConSanMoiEngine::RecordReplay ? ConSanPatchKind::TrampolineMoiAccessRecordStore
          : engine == ConSanMoiEngine::InlineShadow
              ? ConSanPatchKind::TrampolineMoiExactShadowStore
              : ConSanPatchKind::TrampolineMoiSampledWatchpointStore;
      const auto access_patch =
          std::ranges::find(result.patches, expected_patch_kind, &ConSanPatchInfo::kind);
      ASSERT_NE(access_patch, result.patches.end());
      ASSERT_TRUE(access_patch->scratch_vgpr);
      AmdGpuCodeObject patched(result.replacement.data(), result.replacement.size());
      ASSERT_TRUE(patched.is_valid());
      const std::vector<uint32_t> body = text_words_at_offset(
          patched, access_patch->trampoline_offset, access_patch->trampoline_size);
      if (engine == ConSanMoiEngine::RecordReplay) {
        EXPECT_TRUE(contains_subsequence(
            body, make_expected_literal_offset_store_words(
                      offsetof(ConSanMoiAccessRecord, lds_byte_count), 3u * sizeof(uint32_t),
                      *access_patch->scratch_vgpr,
                      static_cast<uint16_t>(*access_patch->scratch_vgpr + 2u), arch)));
      } else if (engine == ConSanMoiEngine::InlineShadow) {
        const uint16_t loop_counter_vgpr = consan_detail::inline_shadow_loop_counter_vgpr(
            *access_patch->scratch_vgpr, test_moi_exec_save_sgpr(result).has_value(),
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
        // Three dwords starting at an arbitrary byte offset span at most four cells.
        const auto four_cell_bound = instrumentation::build_v_cmp_gt_u32_vcc(
            scalar_positive_inline_u32(4u), loop_counter_vgpr, arch);
        ASSERT_TRUE(four_cell_bound);
        EXPECT_NE(std::ranges::find(body, *four_cell_bound), body.end());
        if (load_clobbers_address) {
          const uint16_t saved_address_vgpr = static_cast<uint16_t>(
              loop_counter_vgpr + consan_detail::inline_shadow_loop_scratch_count(
                                      test_admitted_accesses(result).front().decoded_width_bits,
                                      consan_moi_exact_shadow::granule_bytes));
          EXPECT_LT(saved_address_vgpr, reserved_end);
          EXPECT_NE(std::ranges::find(body, build_v_mov_b32_e32(saved_address_vgpr,
                                                                vector_source_vgpr(0u), arch)),
                    body.end());
        }
      } else {
        const uint16_t high_vgpr = static_cast<uint16_t>(*access_patch->scratch_vgpr + 3u);
        const uint16_t tmp_vgpr = static_cast<uint16_t>(*access_patch->scratch_vgpr + 4u);
        const uint32_t encoded_twelve_byte_count =
            encode_consan_moi_sampled_byte_count(12u)
            << (consan_moi_sampled_watchpoint::count_shift - 32u);
        const auto byte_count_literal =
            instrumentation::build_v_mov_b32_literal(tmp_vgpr, encoded_twelve_byte_count, arch);
        const auto add_byte_count = instrumentation::build_v_add_u32(
            high_vgpr, vector_source_vgpr(high_vgpr), tmp_vgpr, arch);
        ASSERT_TRUE(byte_count_literal && add_byte_count);
        std::vector<uint32_t> expected = *byte_count_literal;
        expected.insert(expected.end(), add_byte_count->begin(), add_byte_count->end());
        EXPECT_TRUE(contains_subsequence(body, expected));
      }
      EXPECT_EQ(consan_access_lowering_count(result, ConSanLoweringOutcomeKind::Instrumented), 1u);
    }
  }
}

TEST(ConSanMoi, Gfx1250MoiEnginesAdmitNativeB96Accesses) {
  constexpr auto store =
      cdna5::build_vds(cdna5::kDsStoreB96Vds, {.offset0 = 12, .addr = 0, .data0 = 1});
  constexpr auto load =
      cdna5::build_vds(cdna5::kDsLoadB96Vds, {.offset0 = 12, .addr = 0, .vdst = 4});
  constexpr auto aliasing_load =
      cdna5::build_vds(cdna5::kDsLoadB96Vds, {.offset0 = 12, .addr = 0, .vdst = 0});
  constexpr std::array<NativeB96Access, 3> accesses = {
      NativeB96Access{store, "ds_store_b96", false},
      NativeB96Access{load, "ds_load_b96", false},
      NativeB96Access{aliasing_load, "ds_load_b96", true},
  };

  expect_moi_engines_admit_native_b96_accesses(
      ROCJITSU_CODE_ARCH_CDNA5, accesses, [](const auto &text_words) {
        return make_gfx1250_code_object(text_words, "native_b96_access");
      });
}

TEST(ConSanMoi, Gfx1250RelaxedLdsAtomicIsAccessButNotSynchronization) {
  constexpr auto atomic = cdna5::build_vds(
      cdna5::kDsCmpstoreRtnB32Vds, {.offset0 = 12, .addr = 0, .data0 = 1, .data1 = 2, .vdst = 3});
  const std::array<uint32_t, 3> text_words = {atomic[0], atomic[1],
                                              build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5)};
  const std::vector<uint8_t> bytes = make_gfx1250_code_object(text_words, "relaxed_lds_atomic");
  MoiOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_track_atomics = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 0, 1, 1);

  const ConSanTransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified()) << "warnings=" << testing::PrintToString(result.warnings)
                                 << " plans=" << testing::PrintToString(result.resource_plans);
  ASSERT_EQ(test_admitted_accesses(result).size(), 1u);
  EXPECT_EQ(test_admitted_accesses(result).front().kind, ConSanLdsAccessKind::Atomic);
  EXPECT_EQ(test_admitted_accesses(result).front().mnemonic, "ds_cmpstore_rtn_b32");
  EXPECT_EQ(consan_access_lowering_count(result, ConSanLoweringOutcomeKind::Instrumented), 1u);
  ASSERT_EQ(result.observation_plan.atomic_site_decisions.size(), 1u);
  EXPECT_EQ(result.observation_plan.atomic_site_decisions.front().kind,
            ConSanSiteDecisionKind::NotApplicable);
  EXPECT_EQ(result.observation_plan.atomic_site_decisions.front().reason,
            ConSanAtomicPolicyReason::UnqualifiedSyncSequence);
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
  MoiOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_track_barriers = false;
  options.moi_track_atomics = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(5, 0, 0, 0, 0, 1, 1);

  const ConSanTransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified()) << testing::PrintToString(result.warnings);
  ASSERT_EQ(test_admitted_accesses(result).size(), 5u);
  EXPECT_TRUE(std::ranges::all_of(test_admitted_accesses(result), [](const auto &candidate) {
    return candidate.kind == ConSanLdsAccessKind::Atomic;
  }));
  EXPECT_EQ(consan_access_decision_count(result, ConSanSiteDecisionKind::Admitted), 5u);
  EXPECT_GT(consan_access_lowering_count(result, ConSanLoweringOutcomeKind::Instrumented), 0u);
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
    MoiOptions options = moi_options(ConSanMoiEngine::RecordReplay);
    options.moi_track_atomics = true;

    const ConSanTransformArtifacts result = test_lower_consan(bytes, options);

    ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
    ASSERT_EQ(result.program_inventory.sync().moi_fence_candidates.size(), 1u);
    EXPECT_FALSE(result.program_inventory.sync().moi_fence_candidates.front().eligible());
    ASSERT_EQ(result.observation_plan.fence_site_decisions.size(), 1u);
    EXPECT_EQ(result.observation_plan.fence_site_decisions.front().kind,
              ConSanSiteDecisionKind::NotApplicable);
    EXPECT_EQ(result.observation_plan.fence_site_decisions.front().reason,
              ConSanFencePolicyReason::AssociationUnavailable);
  }
}

TEST(ConSanMoi, Cdna4UnassociatedFenceIsNotApplicable) {
  const auto fence = build_cdna4_s_dcache_inv_vol(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(fence);
  std::vector<uint32_t> text_words(fence->begin(), fence->end());
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4));
  const std::vector<uint8_t> bytes = make_cdna4_lds_code_object(text_words, "unassociated_fence");
  MoiOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_track_atomics = true;

  const ConSanTransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_EQ(result.program_inventory.sync().moi_fence_candidates.size(), 1u);
  EXPECT_FALSE(result.program_inventory.sync().moi_fence_candidates.front().eligible());
  ASSERT_EQ(result.observation_plan.fence_site_decisions.size(), 1u);
  EXPECT_EQ(result.observation_plan.fence_site_decisions.front().kind,
            ConSanSiteDecisionKind::NotApplicable);
  EXPECT_EQ(result.observation_plan.fence_site_decisions.front().reason,
            ConSanFencePolicyReason::AssociationUnavailable);
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
  MoiOptions options = moi_options();

  const auto result = test_lower_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.program_inventory.kernels().size(), 1u);
  ASSERT_TRUE(result.program_inventory.kernels().front().uses_dynamic_stack.has_value());
  EXPECT_TRUE(*result.program_inventory.kernels().front().uses_dynamic_stack);
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
  MoiOptions options = moi_options();

  const auto result = test_lower_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.program_inventory.kernels().size(), 1u);
  ASSERT_TRUE(result.program_inventory.kernels().front().uses_dynamic_stack.has_value());
  EXPECT_TRUE(*result.program_inventory.kernels().front().uses_dynamic_stack);
  ASSERT_TRUE(result.program_inventory.kernels().front().sgpr_count.has_value());
  EXPECT_EQ(*result.program_inventory.kernels().front().sgpr_count, 83u);
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

  const ConSanTransformArtifacts result = test_lower_consan(bytes, moi_options());

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.program_inventory.kernels().size(), 1u);
  EXPECT_TRUE(result.program_inventory.kernels().front().has_dynamic_lds);
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
  MoiOptions options = moi_options();

  const auto result = test_lower_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.program_inventory.functions().size(), 1u);
  ASSERT_EQ(result.program_inventory.access_sites().size(), 1u);
  ASSERT_EQ(test_admitted_accesses(result).size(), 1u);
  const ConSanAccessInventorySite candidate = test_admitted_accesses(result).front();
  EXPECT_EQ(candidate.origin, ConSanAccessOrigin::Flat);
  EXPECT_EQ(candidate.kind, ConSanLdsAccessKind::Read);
  EXPECT_EQ(candidate.flat_address_space_hint, ConSanFlatAddressSpaceHint::Group);
  EXPECT_EQ(candidate.container.kind, ConSanProgramContainerKind::Function);
  EXPECT_EQ(candidate.container.name, "lds_helper");
  EXPECT_EQ(candidate.mnemonic, "flat_load_b32");
  EXPECT_EQ(candidate.physical_id.original_text_offset, 24u);
  EXPECT_EQ(candidate.file_offset, 0x118u);
  EXPECT_EQ(candidate.instruction_size, 3u * sizeof(uint32_t));
  EXPECT_EQ(candidate.decoded_width_bits, 32u);
  ASSERT_TRUE(candidate.operands.destination_vgpr);
  EXPECT_EQ(*candidate.operands.destination_vgpr, 2u);
  ASSERT_TRUE(candidate.operands.address_vgpr);
  EXPECT_EQ(*candidate.operands.address_vgpr, 0u);
  ASSERT_TRUE(candidate.operands.raw_vaddr);
  EXPECT_EQ(*candidate.operands.raw_vaddr, 0u);
  ASSERT_TRUE(candidate.operands.raw_vdst);
  EXPECT_EQ(*candidate.operands.raw_vdst, 2u);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_TRUE(result.resource_plans.front().owner_descriptor_file_offsets.empty());
  EXPECT_EQ(result.resource_plans.front().source, ConSanRegisterAllocationSource::Unsupported);
  EXPECT_EQ(result.resource_plans.front().reason, ConSanRegisterPlanReason::MissingOwner);
  EXPECT_EQ(test_resource_plan_summary(result).unsupported_plans, 1u);
}

TEST(ConSanMoi, LoweringOffsetsPreserveNativeRangesAndDoNotReapplyFlatImmediate) {
  ConSanMoiCandidate candidate;
  ConSanAccessRange range{.id = {}, .static_byte_offset = -8, .byte_width = 4u};
  candidate.origin = ConSanAccessOrigin::Flat;
  EXPECT_EQ(candidate.lowering_offset(range), 0u);

  candidate.origin = ConSanAccessOrigin::NativeLds;
  range.static_byte_offset = 3u * 256u;
  EXPECT_EQ(candidate.lowering_offset(range), 3u * 256u);
}

TEST(ConSanMoi, SharedHelperPlanUsesCommonDeadWindowAcrossTwoOwners) {
  const std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object();
  MoiOptions options = moi_options();

  const auto result = test_lower_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.program_inventory.kernels().size(), 3u);
  ASSERT_EQ(result.program_inventory.functions().size(), 1u);
  ASSERT_EQ(test_admitted_accesses(result).size(), 1u);
  EXPECT_EQ(test_admitted_accesses(result).front().container.kind,
            ConSanProgramContainerKind::Function);
  EXPECT_EQ(test_admitted_accesses(result).front().container.name, "shared_lds_helper");
  ASSERT_EQ(result.resource_plans.size(), 1u);
  const ConSanCandidateResourcePlan &plan = result.resource_plans.front();
  ASSERT_EQ(plan.owner_descriptor_file_offsets.size(), 2u);
  EXPECT_EQ(plan.source, ConSanRegisterAllocationSource::LivenessDead);
  EXPECT_EQ(plan.reason, ConSanRegisterPlanReason::None);
  EXPECT_EQ(plan.scratch_vgpr, 1);
  EXPECT_EQ(plan.scratch_vgpr_count, 6u);
}

std::vector<uint8_t> make_cdna4_shared_scalar_owner_code_object(bool indirect_sgpr_access) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  const auto access =
      build_cdna4_ds_store_b32(/*vaddr=*/10, /*vdata=*/11, /*byte_offset=*/0, kArch);
  if (!access)
    return {};
  std::vector<uint32_t> helper(access->begin(), access->end());
  if (indirect_sgpr_access)
    helper.push_back(0xBE802A02u); // s_movrels_b32 s0, s2

  TwoKernelSharedFixtureOptions fixture;
  fixture.first_vgpr_granulated = 3u;
  fixture.second_vgpr_granulated = 3u;
  fixture.entry_nop_words = 1u;
  std::vector<uint8_t> bytes = make_two_kernel_shared_helper_code_object(fixture, kArch, helper);
  mutate_kernel_descriptor(bytes, "shared_owner_0", [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 4u);
  });
  mutate_kernel_descriptor(bytes, "shared_owner_1", [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z,
                    1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 9u);
  });
  constexpr std::array<std::string_view, 1> kAdditionalOwners = {"shared_owner_1"};
  append_kernel_metadata_note(bytes, "shared_owner_0", /*uses_dynamic_stack=*/true,
                              /*sgpr_count=*/0u, std::nullopt, std::nullopt,
                              /*has_dynamic_lds=*/false, kAdditionalOwners);
  return bytes;
}

TEST(ConSanMoi, Cdna4DirectScalarStateReusesUnreferencedSharedOwnerAllocation) {
  const std::vector<uint8_t> bytes =
      make_cdna4_shared_scalar_owner_code_object(/*indirect_sgpr_access=*/false);
  ASSERT_FALSE(bytes.empty());

  MoiOptions options = moi_options(ConSanMoiEngine::Sampled);
  options.test_force_vgpr_spill = true;
  options.moi_runtime_sample_stride = 2u;
  options.moi_track_barriers = false;
  options.moi_track_atomics = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = direct_sampled_report_bytes(2);
  options.max_patches = 1u;

  const ConSanTransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified()) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(test_moi_persistent_sgpr_state(result).owner);
  ASSERT_FALSE(result.resource_plans.empty());
  EXPECT_LT(*test_moi_persistent_sgpr_state(result).owner, 80u);
  EXPECT_TRUE(std::ranges::all_of(result.resource_plans, [](const auto &plan) {
    return !plan.has_indirect_sgpr_access && plan.sgpr_reference_coverage_complete &&
           plan.scalar_tail_floor < 80u;
  }));
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
}

TEST(ConSanMoi, Cdna4ScalarStateClearsEverySharedOwnerAllocation) {
  const std::vector<uint8_t> bytes =
      make_cdna4_shared_scalar_owner_code_object(/*indirect_sgpr_access=*/true);
  ASSERT_FALSE(bytes.empty());

  // The per-owner context path is shared by the Sampled/Inline dynamic-stack
  // fallback and Record/Replay's scalar tail. InlineShadow's scalar entry and
  // emission path is covered by
  // Cdna4InlineScalarPersistencePlansEntryScratchForEveryComponent in
  // moi_inline_shadow_test.cpp.
  for (ConSanMoiEngine engine : {ConSanMoiEngine::Sampled, ConSanMoiEngine::RecordReplay}) {
    SCOPED_TRACE(testing::PrintToString(engine));
    MoiOptions options = moi_options(engine);
    options.test_force_vgpr_spill = true;
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

    const ConSanTransformArtifacts result = test_lower_consan(bytes, options);

    ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
    ASSERT_TRUE(test_moi_dispatch_id_sgpr(result));
    ASSERT_TRUE(test_moi_exec_save_sgpr(result));
    EXPECT_GE(*test_moi_dispatch_id_sgpr(result), 80u);
    EXPECT_GE(*test_moi_exec_save_sgpr(result), 80u);
    for (const ConSanMoiTransientSgprAssignment &assignment :
         test_moi_transient_sgpr_assignments(result)) {
      EXPECT_GE(assignment.exec_save_sgpr, 80u);
      if (assignment.dispatch_id_sgpr) {
        EXPECT_GE(*assignment.dispatch_id_sgpr, 80u);
      }
    }
    if (engine == ConSanMoiEngine::Sampled) {
      // Above the 80-SGPR scalar-relative tail, CDNA4 has too few physical-VCC
      // safe holes for dispatch state, the nine-register transient window, and
      // persistent owner/epoch state. Failing closed is the only safe result.
      EXPECT_FALSE(result.modified());
      EXPECT_EQ(result.outcome, ConSanTransformOutcome::Unsupported);
      EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
        return warning.find("cannot place persistent scalar state") != std::string::npos;
      })) << testing::PrintToString(result.warnings);
      continue;
    }
    ASSERT_TRUE(result.modified()) << testing::PrintToString(result.warnings);
    ASSERT_TRUE(test_moi_persistent_sgpr_state(result).owner)
        << testing::PrintToString(result.warnings);
    ASSERT_TRUE(test_moi_persistent_sgpr_state(result).epoch);
    EXPECT_GE(*test_moi_persistent_sgpr_state(result).owner, 80u);
    EXPECT_EQ(*test_moi_persistent_sgpr_state(result).epoch,
              *test_moi_persistent_sgpr_state(result).owner + 1u);
    if (engine == ConSanMoiEngine::RecordReplay) {
      EXPECT_FALSE(test_moi_persistent_sgpr_state(result).workgroup_key);
      ASSERT_TRUE(test_moi_persistent_sgpr_state(result).record_replay_workgroup.complete());
      EXPECT_EQ(*test_moi_persistent_sgpr_state(result).record_replay_workgroup.x,
                *test_moi_persistent_sgpr_state(result).epoch + 1u);
      EXPECT_EQ(*test_moi_persistent_sgpr_state(result).record_replay_workgroup.y,
                *test_moi_persistent_sgpr_state(result).record_replay_workgroup.x + 1u);
      EXPECT_EQ(*test_moi_persistent_sgpr_state(result).record_replay_workgroup.z,
                *test_moi_persistent_sgpr_state(result).record_replay_workgroup.y + 1u);
    }
    const auto access_patch =
        std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
          return patch.phase == ConSanPatchPhase::Instrumentation &&
                 patch.owner_descriptor_file_offsets.size() == 2u;
        });
    ASSERT_NE(access_patch, result.patches.end());
    ASSERT_FALSE(result.resource_plans.empty());
    EXPECT_TRUE(
        std::ranges::all_of(result.resource_plans, [](const ConSanCandidateResourcePlan &plan) {
          return plan.has_indirect_sgpr_access && plan.sgpr_reference_coverage_complete &&
                 plan.scalar_tail_floor >= 80u;
        }));
    EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  }
}

TEST(ConSanMoi, Cdna4SharedInlineExecSaveAvoidsEveryOwnerPhysicalVcc) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  const auto access =
      build_cdna4_ds_store_b32(/*vaddr=*/10u, /*vdata=*/11u, /*byte_offset=*/0u, kArch);
  ASSERT_TRUE(access);
  TwoKernelSharedFixtureOptions fixture;
  fixture.first_vgpr_granulated = 7u;
  fixture.second_vgpr_granulated = 7u;
  fixture.entry_nop_words = 8u;
  std::vector<uint8_t> bytes = make_two_kernel_shared_helper_code_object(fixture, kArch, *access);
  ASSERT_FALSE(bytes.empty());
  mutate_kernel_descriptor(bytes, "shared_owner_0", [](KD &descriptor) {
    // 32 decoded SGPRs place physical VCC at s26:s27.
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 3u);
    // The fixture has no accumulator operands; keep its boundary empty at the
    // end of the 64-register unified allocation.
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 15u);
  });
  mutate_kernel_descriptor(bytes, "shared_owner_1", [](KD &descriptor) {
    // 48 decoded SGPRs place physical VCC at s42:s43.
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                    kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 5u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc3, kd::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 15u);
  });

  MoiOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.scratch_vgpr = 12u;
  options.moi_owner_vgpr = 40u;
  options.moi_epoch_vgpr = 41u;
  options.moi_track_barriers = false;
  options.moi_track_atomics = false;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  options.max_patches = 1u;

  const ConSanTransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_NE(result.resource_plans.front().source, ConSanRegisterAllocationSource::Unsupported);
  EXPECT_EQ(result.resource_plans.front().owner_descriptor_file_offsets.size(), 2u);
  ASSERT_TRUE(result.modified()) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(test_moi_dispatch_id_sgpr(result));
  ASSERT_TRUE(test_moi_exec_save_sgpr(result));
  // The highest original VCC ends at s43, so the wide window must start at or
  // above s44.
  EXPECT_GE(*test_moi_exec_save_sgpr(result), 44u);
  constexpr std::array<uint16_t, 2> kOriginalVccBases = {26u, 42u};
  for (uint16_t vcc_base : kOriginalVccBases) {
    EXPECT_FALSE(sgpr_ranges_overlap(*test_moi_dispatch_id_sgpr(result), 2u, vcc_base, 2u));
    EXPECT_FALSE(sgpr_ranges_overlap(*test_moi_exec_save_sgpr(result),
                                     kConSanMoiInlineExecSaveSgprCount, vcc_base, 2u));
  }
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
}

TEST(ConSanMoi, SharedHelperAtomicUsesCommonOwnerResourcePlan) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.helper_has_ordered_atomic = true;
  const std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object(fixture);
  ASSERT_FALSE(bytes.empty());
  MoiOptions options = moi_options();
  options.moi_track_atomics = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 0, 1, 1);

  const auto result = test_lower_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.program_inventory.functions().size(), 1u);
  ASSERT_EQ(result.program_inventory.functions().front().atomic_sites.size(), 1u);
  ASSERT_EQ(result.resource_plans.size(), 2u);
  const auto plan_it = std::ranges::find_if(result.resource_plans, [](const auto &item) {
    return item.site_kind == ConSanResourceSiteKind::Atomic;
  });
  ASSERT_NE(plan_it, result.resource_plans.end());
  const ConSanCandidateResourcePlan &plan = *plan_it;
  EXPECT_EQ(plan.site_kind, ConSanResourceSiteKind::Atomic);
  EXPECT_EQ(plan.source, ConSanRegisterAllocationSource::LivenessDead);
  ASSERT_EQ(plan.owner_descriptor_file_offsets.size(), 2u);
  ASSERT_TRUE(result.modified()) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(test_moi_owner_vgpr(result));
  EXPECT_TRUE(test_moi_epoch_vgpr(result));
  EXPECT_TRUE(test_moi_record_replay_workgroup_vgprs(result).complete());
  ASSERT_TRUE(test_moi_exec_save_sgpr(result));
  const auto atomic_patch = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineMoiAtomicRecord;
  });
  ASSERT_NE(atomic_patch, result.patches.end());
  EXPECT_EQ(atomic_patch->owner_descriptor_file_offsets, plan.owner_descriptor_file_offsets);
  EXPECT_EQ(std::count_if(result.patches.begin(), result.patches.end(),
                          [](const ConSanPatchInfo &patch) {
                            return patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue;
                          }),
            2);
}

TEST(ConSanMoi, SharedHelperAtomicSpillUsesOneLayoutForEveryOwner) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.helper_has_ordered_atomic = true;
  fixture.first_private_bytes = 0;
  fixture.second_private_bytes = 20;
  const std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object(fixture);
  ASSERT_FALSE(bytes.empty());
  MoiOptions options = moi_options();
  options.moi_track_atomics = true;
  options.test_force_vgpr_spill = true;
  options.moi_owner_vgpr = 10;
  options.moi_epoch_vgpr = 11;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 0, 1, 1);

  const auto result = test_lower_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified()) << testing::PrintToString(result.warnings);
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
  const auto fence_it = std::ranges::find(result.patches, ConSanPatchKind::TrampolineMoiFenceRecord,
                                          &ConSanPatchInfo::kind);
  ASSERT_NE(fence_it, result.patches.end());
  EXPECT_EQ(fence_it->owner_descriptor_file_offsets, plan.owner_descriptor_file_offsets);
  uint32_t shared_private_size = 0u;
  for (const ConSanPatchInfo &item : result.patches) {
    if (item.owner_descriptor_file_offsets == plan.owner_descriptor_file_offsets) {
      shared_private_size = std::max(shared_private_size, item.required_private_segment_size);
    }
  }
  ASSERT_GT(shared_private_size, 0u);

  AmdGpuCodeObject patched(result.replacement.data(), result.replacement.size());
  ASSERT_TRUE(patched.is_valid());
  for (const AmdGpuKernelInfo &kernel : patched.kernels()) {
    KD descriptor{};
    std::memcpy(&descriptor, result.replacement.data() + kernel.descriptor_file_offset,
                sizeof(descriptor));
    if (kernel.name == "shared_owner_0" || kernel.name == "shared_owner_1") {
      EXPECT_EQ(descriptor.private_segment_fixed_size, shared_private_size);
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

  MoiOptions options = moi_options();
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);
  const auto result = test_lower_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified());
  ASSERT_EQ(non_entry_prologue_patch_count(result), 1u);
  const ConSanPatchInfo &patch = result.patches.front();
  EXPECT_EQ(patch.kind, ConSanPatchKind::TrampolineMoiAccessRecordStore);
  EXPECT_EQ(patch.anchor_offset, 20u);
  ASSERT_EQ(patch.owner_descriptor_file_offsets.size(), 2u);
  EXPECT_EQ(patch.owner_descriptor_file_offsets,
            result.resource_plans.front().owner_descriptor_file_offsets);

  AmdGpuCodeObject patched(result.replacement.data(), result.replacement.size());
  ASSERT_TRUE(patched.is_valid());
  const auto patched_unrelated =
      std::ranges::find_if(patched.kernels(), [](const AmdGpuKernelInfo &kernel) {
        return kernel.name == "unrelated_kernel";
      });
  ASSERT_NE(patched_unrelated, patched.kernels().end());
  KD patched_unrelated_descriptor{};
  std::memcpy(&patched_unrelated_descriptor,
              result.replacement.data() + patched_unrelated->descriptor_file_offset,
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
  MoiOptions options = moi_options();
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const auto result = test_lower_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified());
  ASSERT_EQ(result.resource_plans.size(), 1u);
  const ConSanCandidateResourcePlan &plan = result.resource_plans.front();
  EXPECT_EQ(plan.source, ConSanRegisterAllocationSource::DescriptorGrowth);
  EXPECT_EQ(plan.scratch_vgpr, 4);
  EXPECT_EQ(plan.required_vgpr_count, 10u);
  ASSERT_EQ(plan.owner_descriptor_file_offsets.size(), 2u);

  AmdGpuCodeObject patched(result.replacement.data(), result.replacement.size());
  ASSERT_TRUE(patched.is_valid());
  for (const AmdGpuKernelInfo &kernel : patched.kernels()) {
    KD descriptor{};
    std::memcpy(&descriptor, result.replacement.data() + kernel.descriptor_file_offset,
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
  MoiOptions options = moi_options();
  options.test_force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const auto result = test_lower_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified());
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().source, ConSanRegisterAllocationSource::SpillRequired);
  EXPECT_EQ(result.resource_plans.front().original_private_segment_size, 20u);
  ASSERT_EQ(non_entry_prologue_patch_count(result), 1u);
  const ConSanPatchInfo &patch = result.patches.front();
  EXPECT_EQ(patch.spilled_vgpr_count, 6u);
  EXPECT_EQ(patch.required_private_segment_size, 56u);
  ASSERT_EQ(patch.owner_descriptor_file_offsets.size(), 2u);

  AmdGpuCodeObject patched(result.replacement.data(), result.replacement.size());
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
    std::memcpy(&descriptor, result.replacement.data() + kernel.descriptor_file_offset,
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
  MoiOptions options = moi_options();
  options.test_force_vgpr_spill = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const auto result = test_lower_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified());
  ASSERT_EQ(result.resource_plans.size(), 1u);
  const ConSanCandidateResourcePlan &plan = result.resource_plans.front();
  EXPECT_EQ(plan.source, ConSanRegisterAllocationSource::SpillRequired);
  EXPECT_EQ(plan.reason, ConSanRegisterPlanReason::None);
  EXPECT_EQ(plan.owner_descriptor_file_offsets.size(), 2u);
  ASSERT_EQ(non_entry_prologue_patch_count(result), 1u);
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
  MoiOptions options = moi_options();
  options.test_force_vgpr_spill = true;
  options.test_kernel_name_filter = "shared_lds_helper";
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const auto result = test_lower_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  ASSERT_EQ(non_entry_prologue_patch_count(result), 1u);
  EXPECT_EQ(result.patches.front().required_private_segment_size, 56u);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().text_offset, result.patches.front().anchor_offset);
  EXPECT_LT(result.resource_plans.front().required_vgpr_count, 256u);

  AmdGpuCodeObject patched(result.replacement.data(), result.replacement.size());
  ASSERT_TRUE(patched.is_valid());
  for (const AmdGpuKernelInfo &kernel : patched.kernels()) {
    KD descriptor{};
    std::memcpy(&descriptor, result.replacement.data() + kernel.descriptor_file_offset,
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
  MoiOptions options = moi_options();
  options.scratch_vgpr = 1;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0);

  const auto result = test_lower_consan(bytes, options);

  EXPECT_TRUE(result.errors.empty());
  EXPECT_FALSE(result.modified());
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().source, ConSanRegisterAllocationSource::Unsupported);
  EXPECT_EQ(result.resource_plans.front().reason, ConSanRegisterPlanReason::ExplicitLive);
  EXPECT_EQ(result.resource_plans.front().owner_descriptor_file_offsets.size(), 2u);
}

TEST(ConSanMoi, SharedPrivateOwnerSupportsMixedWaveSizesWithResidentWaveIdentity) {
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
  MoiOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.test_force_private_epoch = true;
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const auto result = test_lower_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified()) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().owner_descriptor_file_offsets.size(), 2u);
  EXPECT_NE(result.resource_plans.front().source, ConSanRegisterAllocationSource::Unsupported);
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
}

TEST(ConSanMoi, InventorySkipsUnknownFlatSites) {
  const std::vector<uint8_t> bytes = make_rdna4_flat_memory_code_object();
  MoiOptions options = moi_options();

  const auto result = test_lower_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.program_inventory.kernels().size(), 1u);
  ASSERT_EQ(result.program_inventory.access_sites().size(), 2u);
  EXPECT_TRUE(test_admitted_accesses(result).empty());

  EXPECT_EQ(std::ranges::count(result.observation_plan.site_decisions,
                               ConSanAccessPolicyReason::FlatProvenancePolicyExcluded,
                               &ConSanSiteDecision::reason),
            2u);
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

    MoiOptions options = moi_options(ConSanMoiEngine::InlineShadow);
    options.moi_report_buffer_address = 0x100000000ull;
    options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
    const ConSanTransformArtifacts result = test_lower_consan(bytes, options);

    ASSERT_TRUE(result.errors.empty())
        << "mask=" << mask << " " << (result.errors.empty() ? "" : result.errors.front());
    ASSERT_TRUE(result.modified()) << "mask=" << mask;
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

    AmdGpuCodeObject patched(result.replacement.data(), result.replacement.size());
    ASSERT_TRUE(patched.is_valid());
    KD descriptor{};
    std::memcpy(&descriptor,
                result.replacement.data() + patched.kernels().front().descriptor_file_offset,
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

  MoiOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  const ConSanTransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified());
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

  AmdGpuCodeObject patched(result.replacement.data(), result.replacement.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);
  KD descriptor{};
  std::memcpy(&descriptor,
              result.replacement.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  EXPECT_EQ(AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT),
            16u);
  EXPECT_EQ(AMDHSA_BITS_GET(descriptor.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_LENGTH), 1u);
  EXPECT_EQ(AMDHSA_BITS_GET(descriptor.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_OFFSET), 3u);

  const auto verify_entry = [&](uint64_t entry_offset) {
    const char *text = patched.text_sections().front()->data();
    uint64_t cursor = entry_offset;
    ASSERT_TRUE(prologue->entry_scalar_backup_vgpr);
    ASSERT_TRUE(prologue->entry_scalar_backup_sgpr_base);
    EXPECT_EQ(prologue->entry_scalar_backup_sgpr_count, 1u);
    const auto owner_backup = instrumentation::build_v_writelane_b32(
        *prologue->entry_scalar_backup_vgpr, *prologue->entry_scalar_backup_sgpr_base,
        /*lane=*/0u, ROCJITSU_CODE_ARCH_RDNA4);
    ASSERT_TRUE(owner_backup);
    std::array<uint32_t, 2> encoded_owner_backup{};
    std::memcpy(encoded_owner_backup.data(), text + cursor, sizeof(encoded_owner_backup));
    EXPECT_EQ(encoded_owner_backup, *owner_backup);
    cursor += sizeof(encoded_owner_backup);
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
  EXPECT_EQ(prologue->dispatch_id_primary_prologue_offset.has_value(),
            prologue->dispatch_id_secondary_prologue_offset.has_value());
  verify_entry(prologue->dispatch_id_primary_prologue_offset.value_or(prologue->trampoline_offset));
  verify_entry(
      prologue->dispatch_id_secondary_prologue_offset.value_or(prologue->trampoline_offset + 256u));
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
  MoiOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  const ConSanTransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified());
  const auto prologue = std::ranges::find_if(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.dispatch_id_capture_sgpr.has_value();
  });
  ASSERT_NE(prologue, result.patches.end());
  EXPECT_EQ(prologue->dispatch_id_source_sgpr, 2u);
  EXPECT_EQ(prologue->dispatch_id_original_user_sgpr_count, 4u);
  EXPECT_EQ(prologue->dispatch_id_expanded_user_sgpr_count, 4u);
  EXPECT_FALSE(prologue->dispatch_id_preload_inserted);

  AmdGpuCodeObject patched(result.replacement.data(), result.replacement.size());
  ASSERT_TRUE(patched.is_valid());
  KD descriptor{};
  std::memcpy(&descriptor,
              result.replacement.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  EXPECT_EQ(AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT),
            4u);
  const std::vector<uint32_t> prologue_words =
      text_words_at_offset(patched, prologue->trampoline_offset, prologue->trampoline_size);
  ASSERT_TRUE(test_moi_exec_save_sgpr(result));
  const auto hwreg = build_hwreg_imm(/*reg_id=*/23, /*offset=*/0, /*size_bits=*/10);
  ASSERT_TRUE(hwreg);
  const auto owner_init =
      build_s_getreg_b32(*test_moi_exec_save_sgpr(result), *hwreg, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(owner_init);
  EXPECT_NE(std::ranges::find(prologue_words, *owner_init), prologue_words.end());
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
  MoiOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  const ConSanTransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified()) << testing::PrintToString(result.warnings);
  std::vector<const ConSanPatchInfo *> prologues;
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.dispatch_id_capture_sgpr)
      prologues.push_back(&patch);
  }
  ASSERT_EQ(prologues.size(), 2u);
  ASSERT_TRUE(test_moi_dispatch_id_sgpr(result));
  EXPECT_EQ(prologues[0]->dispatch_id_capture_sgpr, test_moi_dispatch_id_sgpr(result));
  EXPECT_EQ(prologues[1]->dispatch_id_capture_sgpr, test_moi_dispatch_id_sgpr(result));
  std::array<uint16_t, 2> sources = {prologues[0]->dispatch_id_source_sgpr,
                                     prologues[1]->dispatch_id_source_sgpr};
  std::ranges::sort(sources);
  EXPECT_EQ(sources, (std::array<uint16_t, 2>{2u, 6u}));

  AmdGpuCodeObject original(bytes.data(), bytes.size());
  AmdGpuCodeObject patched(result.replacement.data(), result.replacement.size());
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
              result.replacement.data() + patched_unrelated->descriptor_file_offset, sizeof(KD));
  EXPECT_EQ(patched_descriptor.compute_pgm_rsrc1, original_descriptor.compute_pgm_rsrc1);
  EXPECT_EQ(patched_descriptor.compute_pgm_rsrc2, original_descriptor.compute_pgm_rsrc2);
  EXPECT_EQ(patched_descriptor.kernel_code_properties, original_descriptor.kernel_code_properties);
  EXPECT_EQ(patched_descriptor.kernarg_preload, original_descriptor.kernarg_preload);
  EXPECT_EQ(patched_descriptor.private_segment_fixed_size,
            original_descriptor.private_segment_fixed_size);

  std::vector<uint8_t> rejected_bytes = make_fixture();
  mutate_kernel_descriptor(rejected_bytes, "shared_owner_1", [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                    kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR, 1u);
    AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                    kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 1u);
    AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                    kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE, 1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 15u);
    AMDHSA_BITS_SET(descriptor.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_LENGTH, 1u);
  });
  const ConSanTransformArtifacts rejected = test_lower_consan(rejected_bytes, options);
  EXPECT_EQ(rejected.outcome, ConSanTransformOutcome::Unsupported);
  EXPECT_FALSE(rejected.modified());
  EXPECT_TRUE(rejected.replacement.empty());
  EXPECT_TRUE(rejected.patches.empty());
}

TEST(ConSanMoi, DispatchPreloadUnsupportedLayoutsRollbackTransactionally) {
  const auto run = [](const auto &mutator, std::optional<uint16_t> explicit_pair = std::nullopt) {
    std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
    mutate_first_kernel_descriptor(bytes, mutator);
    MoiOptions options = moi_options(ConSanMoiEngine::InlineShadow);
    options.moi_dispatch_id_sgpr = explicit_pair;
    options.moi_report_buffer_address = 0x100000000ull;
    options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
    return test_lower_consan(bytes, options);
  };

  const ConSanTransformArtifacts user_limit = run([](KD &descriptor) {
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
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 15u);
    AMDHSA_BITS_SET(descriptor.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_LENGTH, 1u);
  });
  EXPECT_EQ(user_limit.outcome, ConSanTransformOutcome::Unsupported);
  EXPECT_FALSE(user_limit.modified());
  EXPECT_TRUE(user_limit.replacement.empty());
  EXPECT_TRUE(user_limit.patches.empty());

  const ConSanTransformArtifacts malformed_prefix = run([](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                    kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER, 1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 2u);
  });
  EXPECT_EQ(malformed_prefix.outcome, ConSanTransformOutcome::Unsupported);
  EXPECT_FALSE(malformed_prefix.modified());
  EXPECT_TRUE(malformed_prefix.replacement.empty());
  EXPECT_TRUE(malformed_prefix.patches.empty());

  const ConSanTransformArtifacts invalid_pair = run([](KD &) {}, 105u);
  EXPECT_EQ(invalid_pair.outcome, ConSanTransformOutcome::Unsupported);
  EXPECT_FALSE(invalid_pair.modified());
  EXPECT_TRUE(invalid_pair.replacement.empty());
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
  MoiOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_report_buffer_address = 0x100000000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;
  const ConSanTransformArtifacts valid = test_lower_consan(bytes, options);
  ASSERT_TRUE(valid.errors.empty()) << (valid.errors.empty() ? "" : valid.errors.front());
  ASSERT_TRUE(valid.modified());
  EXPECT_TRUE(validate_consan_modified_elf(bytes, valid).empty());

  ConSanTransformArtifacts descriptor_corruption = valid;
  AmdGpuCodeObject descriptor_object(descriptor_corruption.replacement.data(),
                                     descriptor_corruption.replacement.size());
  ASSERT_TRUE(descriptor_object.is_valid());
  const uint64_t descriptor_offset = descriptor_object.kernels().front().descriptor_file_offset;
  KD descriptor{};
  std::memcpy(&descriptor, descriptor_corruption.replacement.data() + descriptor_offset,
              sizeof(descriptor));
  AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                  kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID, 0u);
  std::memcpy(descriptor_corruption.replacement.data() + descriptor_offset, &descriptor,
              sizeof(descriptor));
  EXPECT_FALSE(validate_consan_modified_elf(bytes, descriptor_corruption).empty());

  ConSanTransformArtifacts capture_corruption = valid;
  const auto prologue =
      std::ranges::find_if(capture_corruption.patches, [](const ConSanPatchInfo &patch) {
        return patch.dispatch_id_capture_sgpr.has_value();
      });
  ASSERT_NE(prologue, capture_corruption.patches.end());
  const size_t capture_file_offset =
      capture_corruption.program_inventory.text_sections().front().file_offset +
      prologue->trampoline_offset;
  const uint32_t wrong_capture = build_s_mov_b32(
      *prologue->dispatch_id_capture_sgpr,
      static_cast<uint16_t>(prologue->dispatch_id_source_sgpr + 1u), ROCJITSU_CODE_ARCH_RDNA4);
  std::memcpy(capture_corruption.replacement.data() + capture_file_offset, &wrong_capture,
              sizeof(wrong_capture));
  EXPECT_FALSE(validate_consan_modified_elf(bytes, capture_corruption).empty());
}

TEST(ConSanMoi, WarnsWhenReportBufferIsSmallerThanHeader) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  MoiOptions options = moi_options();
  options.moi_report_buffer_address = 0x1000;
  options.moi_report_buffer_size = sizeof(ConSanMoiReportHeader) - 1;

  const auto result = test_lower_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  bool saw_small_buffer_warning = false;
  for (const std::string &warning : result.warnings)
    saw_small_buffer_warning |=
        warning.find("smaller than the report ABI header") != std::string::npos;
  EXPECT_TRUE(saw_small_buffer_warning);
}

TEST(ConSanMoi, RejectsReportBufferLargerThanDynamicRecordOffsetWindow) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  MoiOptions options = moi_options();
  options.moi_report_buffer_address = 0x1000;
  options.moi_report_buffer_size = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1u;

  const auto result = test_lower_consan(bytes, options);

  EXPECT_FALSE(consan_patch_succeeded(result));
  EXPECT_TRUE(std::ranges::any_of(result.errors, [](const std::string &error) {
    return error.find("32-bit dynamic record-offset window") != std::string::npos;
  }));
}

TEST(ConSanMoi, InventorySkipsUnsupportedNativeLdsSites) {
  const std::vector<uint8_t> bytes = make_rdna4_unsupported_lds_code_object();
  MoiOptions options = moi_options(ConSanMoiEngine::RecordReplay);

  const auto result = test_lower_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(test_admitted_accesses(result).size(), 2u);
  EXPECT_EQ(test_admitted_accesses(result)[0].mnemonic, "ds_store_b32");
  EXPECT_EQ(test_admitted_accesses(result)[1].mnemonic, "ds_load_b32");
  EXPECT_EQ(std::ranges::count(result.observation_plan.site_decisions,
                               ConSanAccessPolicyReason::OperationKindExcluded,
                               &ConSanSiteDecision::reason),
            1u);
}

TEST(ConSanMoi, LoweringCandidatesAreExactlyTheAdmittedAccessIntents) {
  const std::vector<uint8_t> bytes = make_rdna4_unsupported_lds_code_object();
  for (ConSanMoiEngine engine :
       {ConSanMoiEngine::RecordReplay, ConSanMoiEngine::Sampled, ConSanMoiEngine::InlineShadow}) {
    SCOPED_TRACE(consan_moi_engine_name(engine));
    const ConSanTransformArtifacts result = test_lower_consan(bytes, moi_options(engine));
    ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);

    std::set<uint64_t> intended_offsets;
    for (const ConSanProbeIntent &intent : result.observation_plan.probe_intents) {
      if (intent.kind == ConSanProbeIntentKind::AccessRecord ||
          intent.kind == ConSanProbeIntentKind::SampledAccess ||
          intent.kind == ConSanProbeIntentKind::ExactShadowAccess) {
        intended_offsets.insert(intent.physical_site.original_text_offset);
      }
    }
    std::set<uint64_t> candidate_offsets;
    for (const ConSanAccessInventorySite &candidate : test_admitted_accesses(result)) {
      candidate_offsets.insert(candidate.physical_id.original_text_offset);
      const auto site = std::ranges::find_if(
          result.program_inventory.access_sites(), [&](const ConSanAccessInventorySite &access) {
            return access.physical_id.original_text_offset ==
                       candidate.physical_id.original_text_offset &&
                   access.container.name == candidate.container.name;
          });
      ASSERT_NE(site, result.program_inventory.access_sites().end());
      EXPECT_EQ(static_cast<const ConSanAccessInventorySite &>(candidate), *site);
    }

    EXPECT_EQ(candidate_offsets, intended_offsets);
    EXPECT_EQ(test_admitted_accesses(result).size(), intended_offsets.size());
  }
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

TEST(ConSanMoi, Gfx1100MoiEnginesAdmitNativeB96Accesses) {
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
      ROCJITSU_CODE_ARCH_RDNA3, accesses, [](const auto &text_words) {
        return make_rdna3_lds_code_object(text_words, "gfx1100_native_b96_access",
                                          kRdna4Wave64AllVgprsGranulated, /*wave32=*/true);
      });
}

TEST(ConSanMoi, InventoryUsesSemanticArchNotDisplayTarget) {
  constexpr std::array<uint32_t, 3> text_words = {0xDB78000Cu,
                                                  0x00000100u, // ds_store_b96 v0, v[1:3] offset:12
                                                  build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4)};
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words, "typed_arch_inventory");
  MoiOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_track_atomics = false;
  options.moi_track_barriers = false;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = 64u * 1024u * 1024u;
  ConSanTransformArtifacts result = test_lower_consan(bytes, options);
  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_EQ(result.program_inventory.target(), ROCJITSU_CODE_TARGET_GFX1201);
  ASSERT_EQ(result.program_inventory.arch(), ROCJITSU_CODE_ARCH_RDNA4);

  ProgramInventoryBuilder display_target_revision(result.program_inventory);
  display_target_revision.set_code_object_facts(
      result.program_inventory.kernel_metadata_trustworthy(),
      result.program_inventory.malformed_kernel_metadata_note_count(),
      result.program_inventory.arch(), ROCJITSU_CODE_TARGET_GFX942);
  result.program_inventory = display_target_revision.view();
  EXPECT_EQ(plan_test_moi_evidence_inventory(result, options).access_range_count, 1u);
}

TEST(ConSanMoi, NativeB96CapabilityMatchesArchitectureBoundary) {
  for (std::string_view mnemonic : {"ds_load_b96", "ds_store_b96"}) {
    SCOPED_TRACE(mnemonic);
    EXPECT_TRUE(consan_moi_supports_native_lds_mnemonic(mnemonic, ROCJITSU_CODE_ARCH_RDNA4));
    EXPECT_TRUE(consan_moi_supports_native_lds_mnemonic(mnemonic, ROCJITSU_CODE_ARCH_CDNA5));
    EXPECT_TRUE(consan_moi_supports_native_lds_mnemonic(mnemonic, ROCJITSU_CODE_ARCH_RDNA3));
    EXPECT_TRUE(consan_moi_supports_native_lds_mnemonic(mnemonic, ROCJITSU_CODE_ARCH_RDNA3_5));
    EXPECT_FALSE(consan_moi_supports_native_lds_mnemonic(mnemonic, ROCJITSU_CODE_ARCH_CDNA3));
    EXPECT_FALSE(consan_moi_supports_native_lds_mnemonic(mnemonic, ROCJITSU_CODE_ARCH_CDNA4));
  }
  for (std::string_view mnemonic : {"ds_read_b96", "ds_write_b96"}) {
    SCOPED_TRACE(mnemonic);
    EXPECT_TRUE(consan_moi_supports_native_lds_mnemonic(mnemonic, ROCJITSU_CODE_ARCH_CDNA3));
    EXPECT_TRUE(consan_moi_supports_native_lds_mnemonic(mnemonic, ROCJITSU_CODE_ARCH_CDNA4));
    EXPECT_FALSE(consan_moi_supports_native_lds_mnemonic(mnemonic, ROCJITSU_CODE_ARCH_RDNA3));
    EXPECT_FALSE(consan_moi_supports_native_lds_mnemonic(mnemonic, ROCJITSU_CODE_ARCH_RDNA3_5));
    EXPECT_FALSE(consan_moi_supports_native_lds_mnemonic(mnemonic, ROCJITSU_CODE_ARCH_RDNA4));
    EXPECT_FALSE(consan_moi_supports_native_lds_mnemonic(mnemonic, ROCJITSU_CODE_ARCH_CDNA5));
  }
}

TEST(ConSanMoi, SharedAccessShapeContractOwnsTwoRangeAndFlatVocabulary) {
  using consan_detail::is_supported_moi_flat_access_mnemonic;
  using consan_detail::native_lds_two_address_form;

  struct ExpectedForm {
    std::string_view mnemonic;
    ConSanLdsAccessKind kind;
    uint32_t width_bits;
    uint32_t scale_bytes;
  };
  constexpr std::array expected_forms = {
      ExpectedForm{"ds_load_2addr_b32", ConSanLdsAccessKind::Read, 32u, 4u},
      ExpectedForm{"ds_store_2addr_b32", ConSanLdsAccessKind::Write, 32u, 4u},
      ExpectedForm{"ds_read2_b32", ConSanLdsAccessKind::Read, 32u, 4u},
      ExpectedForm{"ds_write2_b32", ConSanLdsAccessKind::Write, 32u, 4u},
      ExpectedForm{"ds_load_2addr_b64", ConSanLdsAccessKind::Read, 64u, 8u},
      ExpectedForm{"ds_store_2addr_b64", ConSanLdsAccessKind::Write, 64u, 8u},
      ExpectedForm{"ds_read2_b64", ConSanLdsAccessKind::Read, 64u, 8u},
      ExpectedForm{"ds_write2_b64", ConSanLdsAccessKind::Write, 64u, 8u},
      ExpectedForm{"ds_load_2addr_stride64_b32", ConSanLdsAccessKind::Read, 32u, 256u},
      ExpectedForm{"ds_store_2addr_stride64_b32", ConSanLdsAccessKind::Write, 32u, 256u},
      ExpectedForm{"ds_read2st64_b32", ConSanLdsAccessKind::Read, 32u, 256u},
      ExpectedForm{"ds_write2st64_b32", ConSanLdsAccessKind::Write, 32u, 256u},
      ExpectedForm{"ds_load_2addr_stride64_b64", ConSanLdsAccessKind::Read, 64u, 512u},
      ExpectedForm{"ds_store_2addr_stride64_b64", ConSanLdsAccessKind::Write, 64u, 512u},
      ExpectedForm{"ds_read2st64_b64", ConSanLdsAccessKind::Read, 64u, 512u},
      ExpectedForm{"ds_write2st64_b64", ConSanLdsAccessKind::Write, 64u, 512u},
  };
  for (const auto &expected : expected_forms) {
    SCOPED_TRACE(expected.mnemonic);
    const auto actual = native_lds_two_address_form(expected.mnemonic);
    ASSERT_TRUE(actual);
    EXPECT_EQ(actual->kind, expected.kind);
    EXPECT_EQ(actual->element_width_bits, expected.width_bits);
    EXPECT_EQ(actual->offset_scale_bytes, expected.scale_bytes);
  }
  EXPECT_FALSE(native_lds_two_address_form("ds_load_b32"));

  EXPECT_TRUE(is_supported_moi_flat_access_mnemonic("flat_load_b128"));
  EXPECT_TRUE(is_supported_moi_flat_access_mnemonic("flat_store_short"));
  EXPECT_FALSE(is_supported_moi_flat_access_mnemonic("flat_load_dwordx3"));
  EXPECT_FALSE(is_supported_moi_flat_access_mnemonic("global_load_dword"));
}

TEST(ConSanMoi, CdnaMoiEnginesAdmitNativeB96Accesses) {
  constexpr auto cdna3_store = cdna3::build_ds(cdna3::kDsWriteB96Ds, {.addr = 0, .data0 = 1});
  constexpr auto cdna3_load = cdna3::build_ds(cdna3::kDsReadB96Ds, {.addr = 0, .vdst = 4});
  constexpr auto cdna3_aliasing_load = cdna3::build_ds(cdna3::kDsReadB96Ds, {.addr = 0, .vdst = 0});
  constexpr auto cdna4_store = cdna4::build_ds(cdna4::kDsWriteB96Ds, {.addr = 0, .data0 = 1});
  constexpr auto cdna4_load = cdna4::build_ds(cdna4::kDsReadB96Ds, {.addr = 0, .vdst = 4});
  constexpr auto cdna4_aliasing_load = cdna4::build_ds(cdna4::kDsReadB96Ds, {.addr = 0, .vdst = 0});
  constexpr std::array cdna3_accesses = {
      NativeB96Access{cdna3_store, "ds_write_b96", false},
      NativeB96Access{cdna3_load, "ds_read_b96", false},
      NativeB96Access{cdna3_aliasing_load, "ds_read_b96", true},
  };
  constexpr std::array cdna4_accesses = {
      NativeB96Access{cdna4_store, "ds_write_b96", false},
      NativeB96Access{cdna4_load, "ds_read_b96", false},
      NativeB96Access{cdna4_aliasing_load, "ds_read_b96", true},
  };

  expect_moi_engines_admit_native_b96_accesses(
      ROCJITSU_CODE_ARCH_CDNA3, cdna3_accesses, [](const auto &text_words) {
        return make_cdna3_lds_code_object(text_words, "cdna3_native_b96_access");
      });
  expect_moi_engines_admit_native_b96_accesses(
      ROCJITSU_CODE_ARCH_CDNA4, cdna4_accesses, [](const auto &text_words) {
        return make_cdna4_lds_code_object(text_words, "cdna4_native_b96_access");
      });
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
    MoiOptions options = moi_options();
    options.moi_engine = engine;
    const ConSanTransformArtifacts result = test_lower_consan(bytes, options);

    ASSERT_TRUE(consan_patch_succeeded(result));
    EXPECT_TRUE(test_admitted_accesses(result).empty());
    ASSERT_EQ(result.observation_plan.site_decisions.size(), 1u);
    const ConSanSiteDecision &decision = result.observation_plan.site_decisions.front();
    EXPECT_EQ(decision.kind, ConSanSiteDecisionKind::Unsupported);
    EXPECT_EQ(decision.reason, ConSanAccessPolicyReason::MissingAddressOperand);
    EXPECT_TRUE(decision.intent_ids.empty());
  }
}

TEST(ConSanMoi, MixedAccessLedgerRetainsSupportedAndUnsupportedFinalCodeSites) {
  const std::array<uint32_t, 5> text_words = {
      0xD8340000u, 0x00000102u, // ds_store_b32 v2, v1
      0xDAC40000u, 0x00000000u, // ds_load_addtid_b32
      0xBFB00000u,              // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  MoiOptions options = moi_options();

  const ConSanTransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.observation_plan.site_decisions.size(), 2u);
  EXPECT_EQ(result.observation_plan.site_decisions[0].kind, ConSanSiteDecisionKind::Admitted);
  EXPECT_EQ(result.observation_plan.site_decisions[0].reason, ConSanAccessPolicyReason::None);
  EXPECT_EQ(result.observation_plan.site_decisions[1].kind, ConSanSiteDecisionKind::Unsupported);
  EXPECT_EQ(result.observation_plan.site_decisions[1].reason,
            ConSanAccessPolicyReason::MissingAddressOperand);
  ASSERT_EQ(result.coverage_ledger.intent_entries().size(), 1u);
  EXPECT_EQ(result.coverage_ledger.intent_entries().front().lowering,
            ConSanLoweringOutcomeKind::PlacementRejected);
}

TEST(ConSanMoi, AutoReportInventoryCountsAdmittedLogicalRangesBeforeAllocation) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  for (ConSanMoiEngine engine :
       {ConSanMoiEngine::RecordReplay, ConSanMoiEngine::Sampled, ConSanMoiEngine::InlineShadow}) {
    SCOPED_TRACE(consan_moi_engine_name(engine));
    MoiOptions options = moi_options();
    options.moi_engine = engine;
    options.max_patches = 1u << 20u;
    options.moi_runtime_sample_stride = engine == ConSanMoiEngine::Sampled ? 256u : 1u;
    const ConSanTransformArtifacts result = test_lower_consan(bytes, options);

    ASSERT_TRUE(consan_patch_succeeded(result));
    const ConSanMoiAutoReportInventory inventory =
        plan_test_moi_evidence_inventory(result, options);
    EXPECT_EQ(inventory.engine, engine);
    EXPECT_EQ(inventory.access_range_count, 2u);
    EXPECT_GE(inventory.diagnostic_count, 2u);
    if (engine == ConSanMoiEngine::Sampled) {
      EXPECT_EQ(inventory.sampled_range_bank_count, 16u);
      EXPECT_EQ(inventory.sampled_watchpoint_count, 16u);
    } else if (engine == ConSanMoiEngine::InlineShadow) {
      EXPECT_TRUE(inventory.inline_diagnostic_count_adaptive);
      EXPECT_EQ(inventory.inline_lds_bytes, kConSanMoiInlineShadowConservativeExactShadowEntries *
                                                consan_moi_exact_shadow::granule_bytes);
      const uint64_t dispatch_banks =
          consan_moi_inline_exact_dispatch_bank_count_for_lds(inventory.inline_lds_bytes);
      EXPECT_EQ(inventory.diagnostic_count, inventory.access_range_count * dispatch_banks *
                                                kConSanMoiInlineShadowDiagnosticHeadroomPerAccess);
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

TEST(ConSanMoi, InlineAutoReportBudgetsDiagnosticsAcrossDispatchBanks) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  MoiOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.max_patches = 1u << 20u;

  const ConSanTransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  const ConSanMoiAutoReportInventory inventory = plan_test_moi_evidence_inventory(result, options);
  const uint64_t dispatch_banks =
      consan_moi_inline_exact_dispatch_bank_count_for_lds(inventory.inline_lds_bytes);
  ASSERT_GT(dispatch_banks, 1u);
  ASSERT_EQ(inventory.access_range_count, 2u);
  EXPECT_EQ(inventory.diagnostic_count, inventory.access_range_count * dispatch_banks *
                                            kConSanMoiInlineShadowDiagnosticHeadroomPerAccess);

  const ConSanMoiAutoReportPlan plan = plan_consan_moi_auto_report(inventory);
  ASSERT_TRUE(plan.complete());
  EXPECT_EQ(plan.layout.diagnostic_capacity, inventory.diagnostic_count);
  EXPECT_EQ(plan.layout.inline_exact_dispatch_bank_count, dispatch_banks);
}

TEST(ConSanMoi, Gfx1250AutoReportUsesRuntimeApertureForDescriptorOpaqueLds) {
  constexpr uint32_t kRuntimeLdsBytes = 96u * 1024u;
  constexpr auto store = cdna5::build_vds(cdna5::kDsStoreB32Vds, {.addr = 0, .data0 = 1});
  const std::array<uint32_t, 3> text_words = {store[0], store[1],
                                              build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5)};
  const std::vector<uint8_t> bytes =
      make_gfx1250_code_object(text_words, "opaque_lds", /*vgpr_granulated=*/4);
  MoiOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.max_workgroup_lds_bytes = kRuntimeLdsBytes;

  const ConSanTransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  const ConSanMoiAutoReportInventory inventory = plan_test_moi_evidence_inventory(result, options);
  EXPECT_EQ(inventory.inline_lds_bytes, kRuntimeLdsBytes);
  const ConSanMoiAutoReportPlan plan = plan_consan_moi_auto_report(inventory);
  ASSERT_TRUE(plan.complete());
  EXPECT_EQ(plan.layout.inline_exact_dispatch_bank_count,
            consan_moi_inline_exact_dispatch_bank_count_for_lds(inventory.inline_lds_bytes));
  EXPECT_EQ(plan.layout.exact_shadow_entry_capacity,
            kRuntimeLdsBytes * plan.layout.inline_exact_dispatch_bank_count);
}

TEST(ConSanMoi, Gfx1250AutoReportCoversFullApertureForDynamicLds) {
  constexpr auto store = cdna5::build_vds(cdna5::kDsStoreB32Vds, {.addr = 0, .data0 = 1});
  const std::array<uint32_t, 3> text_words = {store[0], store[1],
                                              build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5)};
  constexpr std::string_view kernel_name = "dynamic_lds_auto_report";
  std::vector<uint8_t> bytes = make_gfx1250_code_object(text_words, kernel_name);
  mutate_first_kernel_descriptor(bytes,
                                 [](KD &descriptor) { descriptor.group_segment_fixed_size = 4u; });
  append_kernel_metadata_note(bytes, kernel_name, /*uses_dynamic_stack=*/false,
                              /*sgpr_count=*/0u, std::nullopt, std::nullopt,
                              /*has_dynamic_lds=*/true);
  MoiOptions options = moi_options(ConSanMoiEngine::InlineShadow);

  const ConSanTransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.program_inventory.kernels().size(), 1u);
  EXPECT_TRUE(result.program_inventory.kernels().front().has_dynamic_lds);
  const ConSanMoiAutoReportInventory inventory = plan_test_moi_evidence_inventory(result, options);
  EXPECT_EQ(inventory.inline_lds_bytes,
            consan_moi_max_workgroup_lds_bytes(ROCJITSU_CODE_ARCH_CDNA5));
  const ConSanMoiAutoReportPlan plan = plan_consan_moi_auto_report(inventory);
  ASSERT_TRUE(plan.complete());
  EXPECT_EQ(plan.layout.inline_exact_dispatch_bank_count,
            consan_moi_inline_exact_dispatch_bank_count_for_lds(inventory.inline_lds_bytes));
  EXPECT_EQ(plan.layout.exact_shadow_entry_capacity,
            consan_moi_max_workgroup_lds_bytes(ROCJITSU_CODE_ARCH_CDNA5) *
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
  MoiOptions options = moi_options(ConSanMoiEngine::InlineShadow);

  const ConSanTransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(test_admitted_accesses(result).size(), 1u);
  ASSERT_EQ(test_admitted_accesses(result).front().origin, ConSanAccessOrigin::Flat);
  ASSERT_EQ(test_admitted_accesses(result).front().flat_address_space_hint,
            ConSanFlatAddressSpaceHint::Group);
  const ConSanMoiAutoReportInventory inventory = plan_test_moi_evidence_inventory(result, options);
  EXPECT_EQ(inventory.access_range_count, 1u);
  EXPECT_EQ(inventory.inline_lds_bytes, kConSanMoiInlineShadowConservativeExactShadowEntries *
                                            consan_moi_exact_shadow::granule_bytes);
  const ConSanMoiAutoReportPlan plan = plan_consan_moi_auto_report(inventory);
  ASSERT_TRUE(plan.complete());
  EXPECT_EQ(plan.layout.inline_exact_dispatch_bank_count,
            consan_moi_inline_exact_dispatch_bank_count_for_lds(inventory.inline_lds_bytes));
  EXPECT_EQ(plan.layout.exact_shadow_entry_capacity,
            kConSanMoiInlineShadowConservativeExactShadowEntries *
                consan_moi_exact_shadow::granule_bytes *
                plan.layout.inline_exact_dispatch_bank_count);
}

TEST(ConSanMoi, OwnerEpochPrologueRedirectsKernelDescriptorEntry) {
  constexpr uint64_t kExpectedPrologueOffset = 256;
  const std::array<uint32_t, 2> text_words = {
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };

  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  MoiOptions options = moi_options();
  options.moi_init_owner_epoch = true;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;

  const auto result = test_lower_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified());
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);
  EXPECT_EQ(result.patches.front().trampoline_offset, kExpectedPrologueOffset);
  EXPECT_EQ(result.patches.front().original_size, 0u);
  EXPECT_EQ(result.patches.front().trampoline_size, 3u * sizeof(uint32_t));

  AmdGpuCodeObject patched(result.replacement.data(), result.replacement.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.kernels().size(), 1u);
  ASSERT_EQ(patched.text_sections().size(), 1u);
  EXPECT_EQ(patched.text_sections().front()->size(),
            kExpectedPrologueOffset + 3u * sizeof(uint32_t));

  KD descriptor{};
  std::memcpy(&descriptor,
              result.replacement.data() + patched.kernels().front().descriptor_file_offset,
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
  MoiOptions options = moi_options();
  options.moi_init_owner_epoch = true;
  options.moi_owner_vgpr = 10;
  options.moi_epoch_vgpr = 11;

  const ConSanTransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified()) << testing::PrintToString(result.warnings);
  EXPECT_EQ(result.outcome, ConSanTransformOutcome::ModifiedValid);
  EXPECT_EQ(result.program_inventory.target(), ROCJITSU_CODE_TARGET_GFX950);
  EXPECT_EQ(result.program_inventory.arch(), ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue);
  EXPECT_EQ(result.patches.front().anchor_offset, 0u);

  AmdGpuCodeObject patched(result.replacement.data(), result.replacement.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.kernels().size(), 1u);
  KD descriptor{};
  std::memcpy(&descriptor,
              result.replacement.data() + patched.kernels().front().descriptor_file_offset,
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
  MoiOptions options = moi_options();
  options.moi_init_owner_epoch = true;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;
  options.moi_exec_save_sgpr = 30;

  const auto result = test_lower_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified());
  ASSERT_EQ(result.patches.size(), 1u);
  const ConSanPatchInfo &patch = result.patches.front();
  EXPECT_EQ(patch.kind, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue);
  EXPECT_EQ(patch.trampoline_size, 8u * sizeof(uint32_t));

  AmdGpuCodeObject patched(result.replacement.data(), result.replacement.size());
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
  MoiOptions options = moi_options();
  options.moi_init_owner_epoch = true;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;

  const auto result = test_lower_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified());

  AmdGpuCodeObject patched(result.replacement.data(), result.replacement.size());
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
  MoiOptions options = moi_options();
  options.moi_init_owner_epoch = true;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;

  const auto result = test_lower_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified());

  AmdGpuCodeObject patched(result.replacement.data(), result.replacement.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.kernels().size(), 1u);
  KD descriptor{};
  std::memcpy(&descriptor,
              result.replacement.data() + patched.kernels().front().descriptor_file_offset,
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
  MoiOptions options = moi_options();
  options.moi_init_owner_epoch = true;
  options.moi_owner_source = ConSanMoiOwnerSource::HwId;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;

  const auto result = test_lower_consan(bytes, options);

  EXPECT_FALSE(result.errors.empty());
  EXPECT_FALSE(result.modified());
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
  MoiOptions options = moi_options();
  options.moi_init_owner_epoch = true;
  options.moi_owner_source = ConSanMoiOwnerSource::HwId;
  options.moi_owner_sgpr = 20;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;

  const auto result = test_lower_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  EXPECT_TRUE(result.modified());
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue);
  EXPECT_EQ(result.patches.front().trampoline_size, 6u * sizeof(uint32_t));

  AmdGpuCodeObject patched(result.replacement.data(), result.replacement.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.kernels().size(), 1u);
  ASSERT_EQ(patched.text_sections().size(), 1u);

  KD descriptor{};
  std::memcpy(&descriptor,
              result.replacement.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  const uint32_t vgpr_granulated = AMDHSA_BITS_GET(
      descriptor.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
  EXPECT_EQ(vgpr_granulated, 3u);
  const uint32_t sgpr_granulated = AMDHSA_BITS_GET(
      descriptor.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT);
  // RDNA's zero field already denotes the complete fixed per-wave SGPR pool.
  EXPECT_EQ(sgpr_granulated, 0u);

  const auto hwreg = build_hwreg_imm(/*reg_id=*/23, /*offset=*/0, /*size_bits=*/10);
  ASSERT_TRUE(hwreg);
  const auto get_hw_id = build_s_getreg_b32(/*sdst=*/20, *hwreg, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(get_hw_id);
  std::vector<uint32_t> expected_prologue_words;
  expected_prologue_words.push_back(*get_hw_id);
  expected_prologue_words.push_back(build_s_delay_alu(kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4));
  expected_prologue_words.push_back(
      *instrumentation::build_salu_to_valu_dependency_wait(ROCJITSU_CODE_ARCH_RDNA4));
  expected_prologue_words.push_back(build_v_mov_b32_e32(11, /*src0=*/20, ROCJITSU_CODE_ARCH_RDNA4));
  expected_prologue_words.push_back(
      build_v_mov_b32_e32(12, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4));
  const auto branch =
      compute_sopp_branch_simm16(kExpectedPrologueOffset + 5u * sizeof(uint32_t), 0);
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

TEST(ConSanMoi, Gfx1100OwnerEpochPrologueUsesHwId1ResidentWaveIdentity) {
  constexpr uint64_t kExpectedPrologueOffset = 256;
  const std::array<uint32_t, 2> text_words = {
      build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA3),
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA3),
  };

  const std::vector<uint8_t> bytes =
      make_rdna3_lds_code_object(text_words, "gfx1100_owner_prologue", /*vgpr_granulated=*/0);
  MoiOptions options = moi_options();
  options.moi_init_owner_epoch = true;
  options.moi_owner_source = ConSanMoiOwnerSource::HwId;
  options.moi_owner_sgpr = 20;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;

  const auto result = test_lower_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified()) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.patches.size(), 1u);
  EXPECT_EQ(result.patches.front().kind, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue);

  AmdGpuCodeObject patched(result.replacement.data(), result.replacement.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.text_sections().size(), 1u);

  const auto hwreg = build_hwreg_imm(/*reg_id=*/23, /*offset=*/0, /*size_bits=*/10);
  ASSERT_TRUE(hwreg);
  const auto get_hw_id =
      instrumentation::build_s_getreg_b32(/*sdst=*/20, *hwreg, ROCJITSU_CODE_ARCH_RDNA3);
  ASSERT_TRUE(get_hw_id);
  const std::array<uint32_t, 6> expected_prologue_words = {
      *get_hw_id,
      build_s_delay_alu(kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA3),
      build_s_delay_alu(kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA3),
      build_v_mov_b32_e32(11, /*src0=*/20, ROCJITSU_CODE_ARCH_RDNA3),
      build_v_mov_b32_e32(12, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA3),
      build_s_branch(
          *compute_sopp_branch_simm16(kExpectedPrologueOffset + 5u * sizeof(uint32_t), 0),
          ROCJITSU_CODE_ARCH_RDNA3),
  };

  const auto text_word_count = patched.text_sections().front()->size() / sizeof(uint32_t);
  std::vector<uint32_t> actual_words(text_word_count);
  std::memcpy(actual_words.data(), patched.text_sections().front()->data(),
              actual_words.size() * sizeof(uint32_t));
  const size_t prologue_word_offset = kExpectedPrologueOffset / sizeof(uint32_t);
  ASSERT_GE(actual_words.size(), prologue_word_offset + expected_prologue_words.size());
  EXPECT_TRUE(std::equal(expected_prologue_words.begin(), expected_prologue_words.end(),
                         actual_words.begin() + static_cast<ptrdiff_t>(prologue_word_offset)));
}

TEST(ConSanMoi, InlineShadowHwIdOwnerPrologueRemapsReservedZero) {
  const std::array<uint32_t, 3> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32 v0, v0
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };

  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "lds_probe", /*vgpr_granulated=*/0);
  MoiOptions options = moi_options(ConSanMoiEngine::InlineShadow);
  options.moi_init_owner_epoch = true;
  options.moi_owner_source = ConSanMoiOwnerSource::HwId;
  options.moi_owner_sgpr = 20;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = kInlineShadowFullLdsReportBufferSize;

  const auto result = test_lower_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified()) << testing::PrintToString(result.warnings);
  const auto prologue = std::ranges::find(
      result.patches, ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue, &ConSanPatchInfo::kind);
  ASSERT_NE(prologue, result.patches.end());
  ASSERT_TRUE(options.moi_owner_sgpr);
  ASSERT_TRUE(test_moi_owner_vgpr(result));

  AmdGpuCodeObject patched(result.replacement.data(), result.replacement.size());
  ASSERT_TRUE(patched.is_valid());
  const auto hwreg = build_hwreg_imm(/*reg_id=*/23, /*offset=*/0, /*size_bits=*/10);
  ASSERT_TRUE(hwreg);
  const uint16_t owner_sgpr = *options.moi_owner_sgpr;
  const auto get_hw_id = build_s_getreg_b32(owner_sgpr, *hwreg, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(get_hw_id);
  const std::array<uint32_t, 7> expected_prefix = {
      *get_hw_id,
      build_s_delay_alu(kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4),
      build_s_add_u32(owner_sgpr, owner_sgpr, scalar_positive_inline_u32(1),
                      ROCJITSU_CODE_ARCH_RDNA4),
      build_s_delay_alu(kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4),
      *instrumentation::build_salu_to_valu_dependency_wait(ROCJITSU_CODE_ARCH_RDNA4),
      build_v_mov_b32_e32(*test_moi_owner_vgpr(result), owner_sgpr, ROCJITSU_CODE_ARCH_RDNA4),
      build_v_mov_b32_e32(12, scalar_positive_inline_u32(0), ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint32_t> actual_words =
      text_words_at_offset(patched, prologue->trampoline_offset, prologue->trampoline_size);
  EXPECT_TRUE(contains_subsequence(actual_words, expected_prefix));
}

TEST(ConSanMoi, AutomaticPersistentProloguesOnlyTargetEmittedProbeOwners) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.unrelated_has_lds = true;
  const std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object(fixture);

  MoiOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_init_owner_epoch = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(8, 0, 0, 0, 1);
  options.max_patches = 1;

  const ConSanTransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified()) << testing::PrintToString(result.warnings);
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

  MoiOptions options = moi_options(ConSanMoiEngine::RecordReplay);
  options.moi_init_owner_epoch = true;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(2, 0, 0, 0, 1);
  options.max_patches = 1;

  const ConSanTransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_TRUE(result.modified()) << testing::PrintToString(result.warnings);
  EXPECT_FALSE(test_moi_owner_vgpr(result));
  EXPECT_FALSE(test_moi_epoch_vgpr(result));
  ASSERT_TRUE(test_moi_persistent_sgpr_state(result).owner);
  ASSERT_TRUE(test_moi_persistent_sgpr_state(result).epoch);
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
  MoiOptions options = moi_options();
  options.moi_track_atomics = true;
  options.scratch_vgpr = 8;
  options.moi_owner_vgpr = 11;
  options.moi_epoch_vgpr = 12;
  options.moi_report_buffer_address = 0x123456780000ull;
  options.moi_report_buffer_size = consan_moi_report_buffer_min_bytes(1, 0, 0, 0, 0, 1, 1);

  const auto result = test_lower_consan(bytes, options);

  ASSERT_TRUE(consan_patch_succeeded(result));
  ASSERT_EQ(result.program_inventory.sync().sync_sequences.size(), 1u);
  EXPECT_EQ(result.program_inventory.sync().sync_sequences.front().memory_role,
            ConSanSyncMemoryRole::Unknown);
  EXPECT_EQ(result.program_inventory.sync().sync_sequences.front().memory_role_confidence,
            ConSanSemanticConfidence::Unsupported);
  EXPECT_FALSE(result.modified());
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
  const auto maybe_high = instrumentation::build_s_cselect_b32(
      /*sdst=*/1u, /*ssrc0=*/1u, /*ssrc1=*/8u, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(maybe_high);
  const std::array<uint32_t, 10> text_words = {
      0xBE8001EBu,                           // s_mov_b64 s[0:1], src_shared_base
      *maybe_high,                           // s_cselect_b32 s1, s1, s8
      0xD5810000u, 0x00000080u,              // v_mov_b32_e64 v0, 0
      0xD5810001u, 0x00000001u,              // v_mov_b32_e64 v1, s1
      0xEC05007Cu, 0x00000002u, 0x00000000u, // flat_load_b32 v2, v[0:1]
      0xBFB00000u,                           // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);

  MoiOptions likely_options;
  likely_options.flavor = ConSanFlavor::Moi;
  const auto likely_result = test_lower_consan(bytes, likely_options);
  ASSERT_TRUE(likely_result.errors.empty());
  ASSERT_EQ(test_admitted_accesses(likely_result).size(), 1u);
  EXPECT_EQ(test_admitted_accesses(likely_result).front().origin, ConSanAccessOrigin::Flat);
  EXPECT_EQ(test_admitted_accesses(likely_result).front().flat_address_space_hint,
            ConSanFlatAddressSpaceHint::MaybeGroup);

  MoiOptions strict_options = likely_options;
  strict_options.flat_provenance_mode = ConSanFlatProvenanceMode::Strict;
  const auto strict_result = test_lower_consan(bytes, strict_options);
  ASSERT_TRUE(strict_result.errors.empty());
  EXPECT_TRUE(test_admitted_accesses(strict_result).empty());
  ASSERT_EQ(strict_result.observation_plan.site_decisions.size(), 1u);
  EXPECT_EQ(strict_result.observation_plan.site_decisions.front().reason,
            ConSanAccessPolicyReason::FlatProvenancePolicyExcluded);
}

TEST(ConSanMoi, CfgBuildInputsCanonicalizeInventoryAndComposedCodeRanges) {
  constexpr std::array<uint32_t, 1> kernel_words = {
      0xBFB00000u, // s_endpgm
  };
  constexpr std::array<uint32_t, 2> function_words = {
      0xBF800000u, // s_nop 0
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words);
  const ConSanTransformArtifacts inventory =
      test_lower_consan(bytes, moi_options(ConSanMoiEngine::RecordReplay));
  ASSERT_TRUE(inventory.errors.empty());
  ASSERT_FALSE(inventory.program_inventory.kernels().empty());
  ASSERT_FALSE(inventory.program_inventory.functions().empty());
  const AmdGpuCodeObject code_object(bytes.data(), bytes.size());
  ASSERT_TRUE(code_object.is_valid());

  const uint64_t duplicate_kernel_entry =
      inventory.program_inventory.kernels().front().entry_text_offset;
  const uint64_t duplicate_function_entry =
      inventory.program_inventory.functions().front().entry_text_offset;
  constexpr uint64_t kComposedEntry = 0x100000u;
  constexpr uint64_t kComposedContinuation = kComposedEntry + 16u;
  const std::array preapplied = {
      ConSanPreappliedCodeRange{.text_offset = duplicate_kernel_entry,
                                .size = 0u,
                                .kernel_name = "kernel",
                                .continuation_text_offset = duplicate_function_entry},
      ConSanPreappliedCodeRange{.text_offset = kComposedEntry,
                                .size = 12u,
                                .kernel_name = "kernel",
                                .continuation_text_offset = kComposedContinuation},
  };

  const consan_detail::ConSanCfgBuildInputs cfg =
      consan_detail::build_consan_cfg_inputs(code_object, inventory.program_inventory.kernels(),
                                             inventory.program_inventory.functions(), preapplied);

  EXPECT_TRUE(std::ranges::is_sorted(cfg.leaders));
  EXPECT_TRUE(std::ranges::is_sorted(cfg.kernel_entries));
  EXPECT_EQ(std::ranges::count(cfg.leaders, duplicate_kernel_entry), 1u);
  EXPECT_EQ(std::ranges::count(cfg.leaders, duplicate_function_entry), 1u);
  EXPECT_NE(std::ranges::find(cfg.leaders, kComposedEntry), cfg.leaders.end());
  EXPECT_NE(std::ranges::find(cfg.leaders, kComposedContinuation), cfg.leaders.end());
  for (const ConSanKernelInfo &kernel : inventory.program_inventory.kernels()) {
    if (!kernel.has_text_range)
      continue;
    EXPECT_NE(std::ranges::find(cfg.kernel_entries, kernel.entry_text_offset),
              cfg.kernel_entries.end());
  }
  EXPECT_EQ(
      std::ranges::count(cfg.code_ranges, kComposedEntry, &BasicBlock::CodeRange::start_offset),
      1u);
  EXPECT_EQ(std::ranges::count(cfg.code_ranges, duplicate_kernel_entry,
                               &BasicBlock::CodeRange::start_offset),
            std::ranges::count(code_object.functions(), duplicate_kernel_entry,
                               &AmdGpuFunctionInfo::entry_text_offset));
  const auto composed =
      std::ranges::find(cfg.code_ranges, kComposedEntry, &BasicBlock::CodeRange::start_offset);
  ASSERT_NE(composed, cfg.code_ranges.end());
  EXPECT_EQ(composed->size, 12u);
}

TEST(ConSanMoi, OccupiedTextRangesCanonicalizeAndUseHalfOpenOverlap) {
  const consan_detail::MoiOccupiedTextRanges occupied({
      {40u, 50u},
      {10u, 20u},
      {20u, 25u},
      {22u, 30u},
      {70u, 70u},
      {90u, 80u},
  });

  ASSERT_EQ(occupied.ranges().size(), 2u);
  EXPECT_EQ(occupied.ranges()[0], std::make_pair(10u, 30u));
  EXPECT_EQ(occupied.ranges()[1], std::make_pair(40u, 50u));
  EXPECT_FALSE(occupied.overlaps(0u, 10u));
  EXPECT_TRUE(occupied.overlaps(9u, 11u));
  EXPECT_TRUE(occupied.overlaps(29u, 40u));
  EXPECT_FALSE(occupied.overlaps(30u, 40u));
  EXPECT_FALSE(occupied.overlaps(50u, 60u));
  EXPECT_FALSE(occupied.overlaps(20u, 20u));
  EXPECT_FALSE(occupied.overlaps(30u, 20u));
}

TEST(ConSanMoi, DenseRouteSitesPartitionByTypedOwnerCapacityAndBranchReach) {
  using consan_detail::MoiDenseRouteOwner;
  using consan_detail::MoiDenseRouteSite;
  const MoiDenseRouteOwner kernel_owner{ConSanProgramContainerKind::Kernel, "same_name"};
  const MoiDenseRouteOwner function_owner{ConSanProgramContainerKind::Function, "same_name"};
  const std::array sites = {
      MoiDenseRouteSite{kernel_owner, 100u, 10u},    MoiDenseRouteSite{function_owner, 30u, 20u},
      MoiDenseRouteSite{kernel_owner, 300000u, 30u}, MoiDenseRouteSite{kernel_owner, 20u, 40u},
      MoiDenseRouteSite{kernel_owner, 40u, 50u},
  };

  const auto capacity_two = consan_detail::partition_moi_dense_route_sites(sites, 2u);
  ASSERT_EQ(capacity_two.size(), 4u);
  EXPECT_EQ(capacity_two[0].owner, kernel_owner);
  EXPECT_EQ(capacity_two[0].first_anchor, 20u);
  EXPECT_EQ(capacity_two[0].source_indices, (std::vector<size_t>{40u, 50u}));
  EXPECT_EQ(capacity_two[1].source_indices, (std::vector<size_t>{10u}));
  EXPECT_EQ(capacity_two[2].first_anchor, 300000u);
  EXPECT_EQ(capacity_two[2].source_indices, (std::vector<size_t>{30u}));
  EXPECT_EQ(capacity_two[3].owner, function_owner);
  EXPECT_EQ(capacity_two[3].source_indices, (std::vector<size_t>{20u}));

  const auto unlimited = consan_detail::partition_moi_dense_route_sites(sites, 0u);
  ASSERT_EQ(unlimited.size(), 3u);
  EXPECT_EQ(unlimited[0].source_indices, (std::vector<size_t>{40u, 50u, 10u}));
  EXPECT_EQ(unlimited[1].source_indices, (std::vector<size_t>{30u}));
  EXPECT_EQ(unlimited[2].source_indices, (std::vector<size_t>{20u}));
  EXPECT_TRUE(consan_detail::partition_moi_dense_route_sites({}, 2u).empty());
}

TEST(ConSanMoi, DenseCandidatePartitionSeparatesOwnerCapacityAndBranchReach) {
  const auto make_candidate = [](ConSanProgramContainerKind kind, std::string name,
                                 uint64_t anchor) {
    ConSanMoiCandidate candidate;
    candidate.container.kind = kind;
    candidate.container.name = std::move(name);
    candidate.physical_id.original_text_offset = anchor;
    candidate.instruction_size = sizeof(uint32_t);
    return candidate;
  };
  std::array candidates = {
      make_candidate(ConSanProgramContainerKind::Kernel, "owner", 100u),
      make_candidate(ConSanProgramContainerKind::Function, "owner", 30u),
      make_candidate(ConSanProgramContainerKind::Kernel, "owner", 300000u),
      make_candidate(ConSanProgramContainerKind::Kernel, "owner", 20u),
      make_candidate(ConSanProgramContainerKind::Kernel, "owner", 40u),
  };
  const std::array<const ConSanMoiCandidate *, candidates.size()> input = {
      &candidates[0], &candidates[1], &candidates[2], &candidates[3], &candidates[4],
  };

  const auto capacity_two = consan_detail::partition_moi_dense_candidates(input, 2u);
  ASSERT_EQ(capacity_two.groups.size(), 4u);
  EXPECT_EQ(capacity_two.groups[0].owner.kind, ConSanProgramContainerKind::Kernel);
  EXPECT_EQ(capacity_two.groups[0].candidates,
            (std::vector<const ConSanMoiCandidate *>{&candidates[3], &candidates[4]}));
  EXPECT_EQ(capacity_two.groups[0].first_candidate_index, 3u);
  EXPECT_EQ(capacity_two.groups[0].anchors(), (std::vector<uint64_t>{20u, 40u}));
  EXPECT_TRUE(capacity_two.groups[0].has_overlapping_entries(24u));
  EXPECT_FALSE(capacity_two.groups[0].has_overlapping_entries(20u));
  EXPECT_TRUE(capacity_two.groups[0].every_site_has_size(sizeof(uint32_t)));
  EXPECT_FALSE(capacity_two.groups[0].every_site_has_size(2u * sizeof(uint32_t)));
  ASSERT_EQ(capacity_two.groups[1].candidates.size(), 1u);
  EXPECT_EQ(capacity_two.groups[1].candidates.front(), &candidates[0]);
  EXPECT_EQ(capacity_two.groups[1].first_candidate_index, 0u);
  ASSERT_EQ(capacity_two.groups[2].candidates.size(), 1u);
  EXPECT_EQ(capacity_two.groups[2].candidates.front(), &candidates[2]);
  EXPECT_EQ(capacity_two.groups[2].first_candidate_index, 2u);
  EXPECT_EQ(capacity_two.groups[3].owner.kind, ConSanProgramContainerKind::Function);
  ASSERT_EQ(capacity_two.groups[3].candidates.size(), 1u);
  EXPECT_EQ(capacity_two.groups[3].candidates.front(), &candidates[1]);
  EXPECT_EQ(capacity_two.index_by_candidate.at(&candidates[4]), 4u);

  const consan_detail::MoiDenseRouteOwner kernel_owner{ConSanProgramContainerKind::Kernel, "owner"};
  ASSERT_EQ(capacity_two.candidates_by_owner.at(kernel_owner).size(), 4u);
  EXPECT_EQ(capacity_two.candidates_by_owner.at(kernel_owner).front(), &candidates[3]);

  const auto unlimited = consan_detail::partition_moi_dense_candidates(input, 0u);
  ASSERT_EQ(unlimited.groups.size(), 3u);
  EXPECT_EQ(unlimited.groups[0].candidates, (std::vector<const ConSanMoiCandidate *>{
                                                &candidates[3], &candidates[4], &candidates[0]}));
  ASSERT_EQ(unlimited.groups[1].candidates.size(), 1u);
  EXPECT_EQ(unlimited.groups[1].candidates.front(), &candidates[2]);
}

} // namespace
} // namespace rocjitsu
