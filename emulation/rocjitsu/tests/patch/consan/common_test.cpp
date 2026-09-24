// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "consan_test_support.h"
#include "rocjitsu/code/patch/consan/consan_access_apply.h"
#include "rocjitsu/code/patch/consan/consan_access_emission.h"
#include "rocjitsu/code/patch/consan/consan_atomic_emission.h"
#include "rocjitsu/code/patch/consan/consan_cfg.h"
#include "rocjitsu/code/patch/consan/consan_device_primitives.h"
#include "rocjitsu/code/patch/consan/consan_internal.h"
#include "rocjitsu/code/patch/consan/consan_physical_site_alias.h"
#include "rocjitsu/code/patch/consan/consan_report_emission.h"
#include "rocjitsu/code/patch/consan/targets/consan_target_profiles.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"
#include "rocjitsu/code/patch/spill_manager.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <limits>
#include <set>
#include <string_view>

namespace rocjitsu::consan {
namespace {

TEST(ConSan, StagedEmissionKeepsOnlyFirstFailureStageAndRollsBackTransaction) {
  std::vector<uint32_t> words{0xfeedfaceu};
  std::vector<std::string> errors;
  {
    InstructionSequence sequence(words);
    detail::StagedEmission emission(sequence, errors, "test transaction");
    emission.stage("claim");
    sequence.append(0x12345678u);
    emission(false, "leaf failure");
    emission.stage("publication");
    emission(false, "later failure");
    EXPECT_FALSE(emission.finish(ROCJITSU_CODE_ARCH_RDNA4));
  }

  EXPECT_EQ(words, std::vector<uint32_t>{0xfeedfaceu});
  EXPECT_EQ(errors,
            (std::vector<std::string>{"leaf failure", "test transaction failed during claim"}));
}

TEST(ConSan, StagedEmissionCommitsWithoutDiagnostic) {
  std::vector<uint32_t> words;
  std::vector<std::string> errors;
  {
    InstructionSequence sequence(words);
    detail::StagedEmission emission(sequence, errors, "test transaction");
    emission.stage("publication");
    sequence.append(0x12345678u);
    EXPECT_TRUE(emission.finish(ROCJITSU_CODE_ARCH_RDNA4));
  }

  EXPECT_EQ(words, std::vector<uint32_t>{0x12345678u});
  EXPECT_TRUE(errors.empty());
}

TEST(ConSan, EmissionRequirementOwnsAppendFailureAndFirstDiagnostic) {
  std::vector<uint32_t> words{0xfeedfaceu};
  std::vector<std::string> errors;
  {
    InstructionSequence sequence(words);
    detail::EmissionRequirement emission(sequence, errors);
    emission.append("first append", 0x12345678u, std::optional<uint32_t>{std::nullopt});
    emission.append("later append", 0x87654321u);
    EXPECT_FALSE(sequence.finish());
  }

  EXPECT_EQ(words, std::vector<uint32_t>{0xfeedfaceu});
  EXPECT_EQ(errors, (std::vector<std::string>{"first append"}));
}

TEST(ConSan, DispatchIdSourcePlanningAuthorizesLiteralsAtTheModeBoundary) {
  OperatingPoint point;
  BoundRuntimeResources resources;
  resources.report_dispatch_id = 0x1234567887654321ull;

  const auto bound = detail::bound_dispatch_id_sources({point, resources});
  ASSERT_TRUE(bound.is_well_formed());
  EXPECT_EQ(bound.literal, resources.report_dispatch_id);

  const auto target_literal =
      detail::target_dispatch_id_sources({point, resources}, ROCJITSU_CODE_ARCH_RDNA4);
  EXPECT_TRUE(target_literal.is_well_formed());
  EXPECT_EQ(target_literal.literal, bound.literal);

  const auto target_without_literal =
      detail::target_dispatch_id_sources({point, resources}, ROCJITSU_CODE_ARCH_CDNA4);
  EXPECT_FALSE(target_without_literal.is_well_formed());

  point.dispatch_sgpr.set(40u);
  const auto scalar =
      detail::target_dispatch_id_sources({point, resources}, ROCJITSU_CODE_ARCH_CDNA4);
  EXPECT_EQ(scalar.sgpr, 40u);
  EXPECT_FALSE(scalar.literal);

  auto malformed = scalar;
  malformed.literal = 1u;
  EXPECT_FALSE(malformed.is_well_formed());
}

TEST(ConSan, AppendedBodyVgprBankTransitionsConsumeTargetCapabilities) {
  EXPECT_TRUE(detail::appended_body_vgpr_bank_mode_is_valid(ROCJITSU_CODE_ARCH_CDNA5, 4u));
  EXPECT_FALSE(detail::appended_body_vgpr_bank_mode_is_valid(ROCJITSU_CODE_ARCH_CDNA4, 4u));
  EXPECT_EQ(detail::appended_body_vgpr_bank_transition_word_count(ROCJITSU_CODE_ARCH_CDNA5,
                                                                  std::nullopt, false, false),
            0u);
  EXPECT_EQ(detail::appended_body_vgpr_bank_transition_word_count(ROCJITSU_CODE_ARCH_CDNA5, 4u,
                                                                  false, false),
            2u);
  EXPECT_EQ(detail::appended_body_vgpr_bank_transition_word_count(ROCJITSU_CODE_ARCH_CDNA5, 4u,
                                                                  false, true),
            4u);
  EXPECT_EQ(detail::appended_body_vgpr_bank_transition_word_count(ROCJITSU_CODE_ARCH_CDNA5, 4u,
                                                                  true, false),
            4u);
  EXPECT_EQ(detail::appended_body_vgpr_bank_transition_word_count(ROCJITSU_CODE_ARCH_CDNA5, 4u,
                                                                  true, true),
            6u);
  EXPECT_EQ(detail::appended_body_vgpr_bank_transition_word_count(ROCJITSU_CODE_ARCH_CDNA4, 4u,
                                                                  true, true),
            0u);

  EXPECT_TRUE(detail::embedded_guest_vgpr_bank_plan_is_valid(false, false, false, 1u));
  EXPECT_TRUE(detail::embedded_guest_vgpr_bank_plan_is_valid(true, true, true, 0u));
  EXPECT_FALSE(detail::embedded_guest_vgpr_bank_plan_is_valid(true, false, true, 0u));
  EXPECT_FALSE(detail::embedded_guest_vgpr_bank_plan_is_valid(true, true, false, 0u));
  EXPECT_FALSE(detail::embedded_guest_vgpr_bank_plan_is_valid(true, true, true, 1u));
}

TEST(ConSan, ResidentWaveOwnerTargetOperationLowersEverySupportedProfile) {
  constexpr uint16_t destination_sgpr = 20u;
  for (const TargetProfile &target : kTargetProfiles) {
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

    const detail::ResidentWaveOwnerRequest request{
        .destination_sgpr = destination_sgpr,

    };
    EXPECT_EQ(request, (detail::ResidentWaveOwnerRequest{
                           .destination_sgpr = destination_sgpr,

                       }));
    std::vector<uint32_t> words;
    ASSERT_TRUE(detail::append_resident_wave_owner(words, request, target));
    std::vector<uint32_t> expected = {*read, *delay};

    expected.push_back(*wait);
    EXPECT_EQ(words, expected);
  }
}

TEST(ConSan, ResidentWaveOwnerTargetOperationRejectsWithoutPartialOutput) {
  ASSERT_FALSE(kTargetProfiles.empty());
  const TargetProfile &target = kTargetProfiles.front();
  const std::vector<uint32_t> prefix = {0x12345678u};

  std::vector<uint32_t> invalid_destination_words = prefix;
  EXPECT_FALSE(detail::append_resident_wave_owner(
      invalid_destination_words, {.destination_sgpr = std::numeric_limits<uint16_t>::max()},
      target));
  EXPECT_EQ(invalid_destination_words, prefix);

  TargetProfile invalid_target = target;
  invalid_target.resident_wave_identity.bit_width = 0u;
  std::vector<uint32_t> invalid_target_words = prefix;
  EXPECT_FALSE(detail::append_resident_wave_owner(invalid_target_words, {.destination_sgpr = 20u},
                                                  invalid_target));
  EXPECT_EQ(invalid_target_words, prefix);
}

TEST(ConSan, SpecialStatePreservationTargetOperationCoversEveryProfile) {
  const detail::SpecialStateSgprs registers{
      .vcc_save_sgpr = 20u,
      .scc_save_sgpr = 24u,
  };
  EXPECT_EQ(registers, (detail::SpecialStateSgprs{
                           .vcc_save_sgpr = 20u,
                           .scc_save_sgpr = 24u,
                       }));
  for (const TargetProfile &target : kTargetProfiles) {
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
    ASSERT_TRUE(detail::append_save_special_state(words, registers, target));
    EXPECT_EQ(words, (std::vector<uint32_t>{*save_scc, *save_vcc}));
    words.clear();
    ASSERT_TRUE(detail::append_restore_special_state(words, registers, target));
    EXPECT_EQ(words, (std::vector<uint32_t>{*restore_vcc, *restore_scc}));
  }
}

TEST(ConSan, SpecialStatePreservationRejectsWithoutPartialOutput) {
  ASSERT_FALSE(kTargetProfiles.empty());
  const TargetProfile &target = kTargetProfiles.front();
  const std::vector<uint32_t> prefix = {0x12345678u};

  std::vector<uint32_t> save_words = prefix;
  EXPECT_FALSE(detail::append_save_special_state(
      save_words, {.vcc_save_sgpr = std::numeric_limits<uint16_t>::max(), .scc_save_sgpr = 24u},
      target));
  EXPECT_EQ(save_words, prefix);

  std::vector<uint32_t> restore_words = prefix;
  EXPECT_FALSE(detail::append_restore_special_state(
      restore_words, {.vcc_save_sgpr = 20u, .scc_save_sgpr = std::numeric_limits<uint16_t>::max()},
      target));
  EXPECT_EQ(restore_words, prefix);
}

TEST(ConSan, EncodedRouteSccRestoreIsConfinedToQualifiedTargetProfiles) {
  constexpr detail::EncodedSccRestoreRequest request{.encoded_sgpr = 20u};
  EXPECT_EQ(request, (detail::EncodedSccRestoreRequest{.encoded_sgpr = request.encoded_sgpr}));
  for (const TargetProfile &target : kTargetProfiles) {
    SCOPED_TRACE(rj_code_target_name(target.target));
    std::vector<uint32_t> words;
    const bool legacy_cdna = arch_is_cdna3_or_cdna4(target.arch);
    EXPECT_EQ(detail::append_restore_scc_from_route_key(words, request, target), legacy_cdna);
    if (!legacy_cdna) {
      EXPECT_TRUE(words.empty());
      continue;
    }
    const uint16_t opcode =
        target.arch == ROCJITSU_CODE_ARCH_CDNA3 ? cdna3::kSBitcmp1B32Sopc : cdna4::kSBitcmp1B32Sopc;
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

TEST(ConSan, EncodedRouteSccRestoreCanPreserveTheRouteKey) {
  constexpr detail::EncodedSccRestoreRequest request{
      .encoded_sgpr = 20u,
      .normalize_encoded_sgpr = false,
  };
  EXPECT_EQ(request, (detail::EncodedSccRestoreRequest{
                         .encoded_sgpr = request.encoded_sgpr,
                         .normalize_encoded_sgpr = request.normalize_encoded_sgpr,
                     }));
  for (const TargetProfile &target : kTargetProfiles) {
    SCOPED_TRACE(rj_code_target_name(target.target));
    std::vector<uint32_t> words;
    const bool legacy_cdna = arch_is_cdna3_or_cdna4(target.arch);
    EXPECT_EQ(detail::append_restore_scc_from_route_key(words, request, target), legacy_cdna);
    if (!legacy_cdna) {
      EXPECT_TRUE(words.empty());
      continue;
    }
    const uint16_t opcode =
        target.arch == ROCJITSU_CODE_ARCH_CDNA3 ? cdna3::kSBitcmp1B32Sopc : cdna4::kSBitcmp1B32Sopc;
    EXPECT_EQ(words,
              (std::vector<uint32_t>{build_sopc_encoding(target.arch, opcode, request.encoded_sgpr,
                                                         scalar_positive_inline_u32(0))}));
  }
}

TEST(ConSan, EncodedRouteSccRestoreRejectsInvalidRegisterTransactionally) {
  const TargetProfile *target = target_profile(ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_NE(target, nullptr);
  const std::vector<uint32_t> prefix = {0x12345678u};
  std::vector<uint32_t> words = prefix;
  EXPECT_FALSE(detail::append_restore_scc_from_route_key(
      words, {.encoded_sgpr = std::numeric_limits<uint16_t>::max()}, *target));
  EXPECT_EQ(words, prefix);
}

TEST(ConSan, DeviceCacheRefreshUsesOnlyQualifiedTargetSequences) {
  for (const TargetProfile &target : kTargetProfiles) {
    SCOPED_TRACE(rj_code_target_name(target.target));
    std::vector<uint32_t> words;
    ASSERT_TRUE(detail::append_device_cache_refresh(words, target));
    switch (target.arch) {
    case ROCJITSU_CODE_ARCH_CDNA3: {
      const auto expected = build_cdna3_buffer_inv_sc1(target.arch);
      ASSERT_TRUE(expected);
      EXPECT_EQ(words, std::vector<uint32_t>(expected->begin(), expected->end()));
      break;
    }
    case ROCJITSU_CODE_ARCH_CDNA4: {
      const auto expected = build_cdna4_buffer_inv_sc1(target.arch);
      ASSERT_TRUE(expected);
      EXPECT_EQ(words, std::vector<uint32_t>(expected->begin(), expected->end()));
      break;
    }
    case ROCJITSU_CODE_ARCH_RDNA3:
    case ROCJITSU_CODE_ARCH_RDNA4:
    case ROCJITSU_CODE_ARCH_CDNA5:
      EXPECT_TRUE(words.empty());
      break;
    default:
      FAIL() << "unexpected ConSan target architecture";
      break;
    }
  }
}

TEST(ConSan, GlobalAtomicCompletionUsesEachTargetsMinimalCounterSequence) {
  for (const TargetProfile &target : kTargetProfiles) {
    SCOPED_TRACE(rj_code_target_name(target.target));
    const auto load = instrumentation::build_s_wait_global_load0(target.arch);
    const auto store = instrumentation::build_s_wait_global_store0(target.arch);
    ASSERT_TRUE(load && store);
    std::vector<uint32_t> expected = {*load};
    if (*store != *load)
      expected.push_back(*store);
    std::vector<uint32_t> words;
    EXPECT_TRUE(detail::append_global_atomic_completion(words, target));
    EXPECT_EQ(words, expected);
  }
}

TEST(ConSan, GlobalAtomicCompletionRejectsInvalidTargetTransactionally) {
  TargetProfile target = kTargetProfiles.front();
  target.arch = ROCJITSU_CODE_ARCH_INVALID;
  const std::vector<uint32_t> prefix = {0x12345678u};
  std::vector<uint32_t> words = prefix;
  EXPECT_FALSE(detail::append_global_atomic_completion(words, target));
  EXPECT_EQ(words, prefix);
}

TEST(ConSan, AtomicCounterIncrementLowersEveryTargetProfile) {
  constexpr detail::AtomicCounterIncrementRequest kRequest{
      .counter_address = 0x1234567800000040ull,
      .result_vgpr = 42u,
      .address_vgpr = 40u,
  };
  for (const TargetProfile &target : kTargetProfiles) {
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
    ASSERT_TRUE(detail::append_global_atomic_completion(expected, target));

    std::vector<uint32_t> words;
    EXPECT_TRUE(detail::append_atomic_counter_increment(words, kRequest, target));
    EXPECT_EQ(words, expected);
  }
}

TEST(ConSan, AtomicCounterIncrementRejectsInvalidRegistersTransactionally) {
  const TargetProfile *target = target_profile(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(target, nullptr);
  const auto rejects = [&](uint16_t result_vgpr, uint16_t address_vgpr,
                           const TargetProfile *request_target = nullptr) {
    const std::vector<uint32_t> prefix = {0x12345678u};
    std::vector<uint32_t> words = prefix;
    EXPECT_FALSE(detail::append_atomic_counter_increment(
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
  TargetProfile invalid_target = *target;
  invalid_target.arch = ROCJITSU_CODE_ARCH_INVALID;
  rejects(/*result_vgpr=*/42u, /*address_vgpr=*/40u, &invalid_target);
}

TEST(ConSan, WorkitemOwnerDerivationPlanDefinesLiveAndEntryCapturedSources) {
  using detail::WorkitemOwnerDerivationPlan;
  using detail::WorkitemOwnerDerivationRequest;
  const WorkitemOwnerDerivationPlan live{
      .entry_workitem_x_private_offset = std::nullopt,
      .wave_size_shift = 5u,
  };
  const WorkitemOwnerDerivationPlan captured{
      .entry_workitem_x_private_offset = 64u,
      .wave_size_shift = 6u,
  };
  const WorkitemOwnerDerivationPlan invalid_shift{
      .entry_workitem_x_private_offset = std::nullopt,
      .wave_size_shift = 32u,
  };
  EXPECT_TRUE(live.is_well_formed());
  EXPECT_TRUE(captured.is_well_formed());
  EXPECT_FALSE(invalid_shift.is_well_formed());
  EXPECT_TRUE((WorkitemOwnerDerivationRequest{.plan = live, .result_vgpr = 255u}).is_well_formed());
  EXPECT_FALSE(
      (WorkitemOwnerDerivationRequest{.plan = live, .result_vgpr = 256u}).is_well_formed());
}

TEST(ConSan, AtomicEvidenceSitePlanRequiresOneCompletePolicyToLoweringJoin) {
  detail::AtomicEvidenceSitePlan plan;
  plan.event = {0u};
  plan.sequence = {0u};
  plan.source_site = {0u};
  plan.address_capture_intent = {.value = 3u};
  plan.evidence_intent = {.value = 4u};
  plan.lowering_form.kind = AtomicLoweringFormKind::FlatVectorAddress;
  EXPECT_TRUE(plan.is_well_formed());

  detail::AtomicEvidenceSitePlan missing_source_site = plan;
  missing_source_site.source_site = {};
  EXPECT_FALSE(missing_source_site.is_well_formed());
  detail::AtomicEvidenceSitePlan missing_event = plan;
  missing_event.event = {};
  EXPECT_FALSE(missing_event.is_well_formed());
  detail::AtomicEvidenceSitePlan missing_sequence = plan;
  missing_sequence.sequence = {};
  EXPECT_FALSE(missing_sequence.is_well_formed());
  detail::AtomicEvidenceSitePlan aliased_intents = plan;
  aliased_intents.evidence_intent = aliased_intents.address_capture_intent;
  EXPECT_FALSE(aliased_intents.is_well_formed());
  detail::AtomicEvidenceSitePlan missing_lowering_form = plan;
  missing_lowering_form.lowering_form.kind = AtomicLoweringFormKind::Count;
  EXPECT_FALSE(missing_lowering_form.is_well_formed());
}

TEST(ConSan, BarrierEvidenceSitePlanRequiresOneCompletingGraphEvent) {
  detail::BarrierEvidenceSitePlan plan;
  plan.event = {0u};
  plan.sequence = {0u};
  plan.source_site = {0u};
  plan.evidence_intent = {.value = 9u};
  EXPECT_TRUE(plan.is_well_formed());

  detail::BarrierEvidenceSitePlan missing_source_site = plan;
  missing_source_site.source_site = {};
  EXPECT_FALSE(missing_source_site.is_well_formed());
  detail::BarrierEvidenceSitePlan missing_event = plan;
  missing_event.event = {};
  EXPECT_FALSE(missing_event.is_well_formed());
  detail::BarrierEvidenceSitePlan missing_sequence = plan;
  missing_sequence.sequence = {};
  EXPECT_FALSE(missing_sequence.is_well_formed());
}

TEST(ConSan, WorkitemOwnerDerivationLowersBothSourcesOnEveryTargetProfile) {
  using detail::WorkitemOwnerDerivationPlan;
  using detail::WorkitemOwnerDerivationRequest;
  constexpr uint16_t kResultVgpr = 42u;
  constexpr uint32_t kPrivateOffset = 64u;
  for (const TargetProfile &target : kTargetProfiles) {
    SCOPED_TRACE(rj_code_target_name(target.target));
    for (const std::optional<uint32_t> private_offset :
         {std::optional<uint32_t>{}, std::optional<uint32_t>{kPrivateOffset}}) {
      const WorkitemOwnerDerivationRequest request{
          .plan =
              WorkitemOwnerDerivationPlan{
                  .entry_workitem_x_private_offset = private_offset,
                  .wave_size_shift = 6u,
              },
          .result_vgpr = kResultVgpr,
      };
      std::vector<uint32_t> expected;
      if (private_offset) {
        const auto load =
            instrumentation::build_private_load_b32(kResultVgpr, *private_offset, target.arch);
        const auto wait = instrumentation::build_s_wait_private_load0(target.arch);
        ASSERT_TRUE(load && wait);
        expected.insert(expected.end(), load->begin(), load->end());
        expected.push_back(*wait);
      }
      const auto shift = instrumentation::build_v_lshrrev_b32(
          kResultVgpr, scalar_positive_inline_u32(6u), private_offset ? kResultVgpr : uint16_t{0u},
          target.arch);
      ASSERT_TRUE(shift);
      expected.push_back(*shift);

      std::vector<uint32_t> words;
      EXPECT_TRUE(detail::append_workitem_owner_derivation(words, request, target));
      EXPECT_EQ(words, expected);
    }
  }
}

TEST(ConSan, WorkitemOwnerDerivationRejectsInvalidRequestsTransactionally) {
  const TargetProfile *target = target_profile(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(target, nullptr);
  const std::vector<uint32_t> prefix = {0x12345678u};
  const auto rejects = [&](const detail::WorkitemOwnerDerivationRequest &request,
                           const TargetProfile &request_target) {
    std::vector<uint32_t> words = prefix;
    EXPECT_FALSE(detail::append_workitem_owner_derivation(words, request, request_target));
    EXPECT_EQ(words, prefix);
  };
  rejects({.plan = {.entry_workitem_x_private_offset = std::nullopt, .wave_size_shift = 32u},
           .result_vgpr = 42u},
          *target);
  rejects({.plan = {.entry_workitem_x_private_offset = std::nullopt, .wave_size_shift = 6u},
           .result_vgpr = 256u},
          *target);
  TargetProfile invalid_target = *target;
  invalid_target.arch = ROCJITSU_CODE_ARCH_INVALID;
  rejects({.plan = {.entry_workitem_x_private_offset = std::nullopt, .wave_size_shift = 6u},
           .result_vgpr = 42u},
          invalid_target);
}

TEST(ConSan, DispatchCaptureSelectsOnePersistentRepresentation) {
  const DispatchIdCapture absent{};
  const DispatchIdCapture scalar = DispatchIdCapture::in_sgprs(40u);
  const DispatchIdCapture vector = DispatchIdCapture::in_vgprs(20u);
  EXPECT_FALSE(absent.present());
  EXPECT_TRUE(scalar.present());
  EXPECT_EQ(scalar.sgpr(), 40u);
  EXPECT_FALSE(scalar.vgpr());
  EXPECT_TRUE(vector.present());
  EXPECT_EQ(vector.vgpr(), 20u);
  EXPECT_FALSE(vector.sgpr());
}

TEST(ConSan, DispatchPrologueEffectCombinesPreloadAndCaptureSgprRequirements) {
  DispatchIdPrologueEffect effect{
      .preload = {.support = DispatchIdPreloadSupport::SupportedInsert,
                  .identity_salt_sgpr = std::nullopt,
                  .required_sgpr_count = 96u},
      .capture = DispatchIdCapture::in_sgprs(40u),
  };
  EXPECT_EQ(effect.required_sgpr_count(), 96u);

  effect.capture = DispatchIdCapture::in_sgprs(100u);
  EXPECT_EQ(effect.required_sgpr_count(), 102u);

  effect.capture = DispatchIdCapture::in_vgprs(100u);
  EXPECT_EQ(effect.required_sgpr_count(), 96u);
}

TEST(ConSan, PrivateStateLayoutOwnsRangesAndDistinctAllocationBoundaries) {
  const PrivateStateLayout valid{
      .epoch_offset = 0u,
      .owner_offset = 4u,

      .dispatch_id_offset = 12u,
      .exact_workgroup_offsets = PersistentWorkgroupPrivateOffsets{20u, 24u, 28u, 32u},
      .persistent_state_end = 36u,
      .ephemeral_base = 48u,
  };
  EXPECT_TRUE(valid.is_well_formed());

  PrivateStateLayout malformed = valid;
  malformed.owner_offset = valid.epoch_offset;
  EXPECT_FALSE(malformed.is_well_formed());
  malformed = valid;
  malformed.dispatch_id_offset = 30u;
  EXPECT_FALSE(malformed.is_well_formed());
  malformed = valid;
  malformed.persistent_state_end -= sizeof(uint32_t);
  EXPECT_FALSE(malformed.is_well_formed());
  malformed = valid;
  malformed.ephemeral_base = malformed.persistent_state_end - sizeof(uint32_t);
  EXPECT_FALSE(malformed.is_well_formed());
}

TEST(ConSan, VgprStateEffectOwnsPairLifetimeAndDistinctRanges) {
  const VgprStateEffect valid{
      .owner_epoch = {10u, 11u},

      .exact_workgroup = PersistentWorkgroupRegisters{20u, 21u, 22u, 23u},
      .owner_epoch_lifetime = OwnerEpochVgprLifetime::OwnerLocalPersistent,
  };
  EXPECT_TRUE(valid.is_well_formed());
  EXPECT_EQ(valid.required_vgpr_count(), 24u);

  VgprStateEffect code_object_persistent = valid;
  code_object_persistent.owner_epoch_lifetime = OwnerEpochVgprLifetime::CodeObjectPersistent;
  EXPECT_TRUE(code_object_persistent.is_well_formed());

  VgprStateEffect transient = valid;
  transient.owner_epoch_lifetime = OwnerEpochVgprLifetime::Transient;
  EXPECT_TRUE(transient.is_well_formed());

  VgprStateEffect malformed = valid;
  malformed.exact_workgroup = PersistentWorkgroupRegisters{valid.owner_epoch.epoch, 21u, 22u, 23u};
  EXPECT_FALSE(malformed.is_well_formed());
  malformed = valid;
  malformed.owner_epoch_lifetime = static_cast<OwnerEpochVgprLifetime>(99u);
  EXPECT_FALSE(malformed.is_well_formed());
}

TEST(ConSan, PrivateEpochProloguePlanTypesScratchAndDispatchRequirements) {
  detail::PrivateEpochPrologueEmissionPlan plan;
  plan.scratch_vgpr = 255u;
  plan.private_state_layout = PrivateStateLayout{
      .epoch_offset = 0u,
      .owner_offset = std::nullopt,

      .dispatch_id_offset = std::nullopt,
      .exact_workgroup_offsets = {},
      .persistent_state_end = 4u,
      .ephemeral_base = 16u,
  };
  plan.spill.vgpr_base = plan.scratch_vgpr;
  plan.spill.vgpr_count = 1u;
  EXPECT_EQ(plan.required_scratch_vgpr_count(), 1u);
  EXPECT_TRUE(plan.is_well_formed());

  // A dispatch capture needs two temporaries, including at the register-file boundary.
  plan.scratch_vgpr = 254u;
  plan.spill.vgpr_base = plan.scratch_vgpr;
  plan.spill.vgpr_count = 2u;
  plan.private_state_layout.dispatch_id_offset = 96u;
  plan.private_state_layout.persistent_state_end = 104u;
  plan.private_state_layout.ephemeral_base = 112u;
  EXPECT_FALSE(plan.is_well_formed());
  plan.dispatch_plan = DispatchIdPreloadPlan{};
  plan.dispatch_capture = DispatchIdCapture::in_vgprs(plan.scratch_vgpr);
  EXPECT_EQ(plan.required_scratch_vgpr_count(), 2u);
  EXPECT_TRUE(plan.is_well_formed());
  plan.dispatch_capture = DispatchIdCapture::in_sgprs(40u);
  EXPECT_FALSE(plan.is_well_formed());
}

TEST(ConSan, OwnerEpochProloguePlanValidatesResolvedOwnerAndEntrySources) {
  detail::OwnerEpochPrologueEmissionPlan plan;
  plan.vgpr_state.owner_epoch = {20u, 21u};
  plan.owner_shift_bits = 6u;
  plan.owner_source = OwnerSource::WorkitemId;
  EXPECT_TRUE(plan.is_well_formed());

  plan.vgpr_state.owner_epoch.epoch = plan.vgpr_state.owner_epoch.owner;
  EXPECT_FALSE(plan.is_well_formed());
  plan.vgpr_state.owner_epoch.epoch = 21u;
  plan.owner_shift_bits = 32u;
  EXPECT_FALSE(plan.is_well_formed());
  plan.owner_shift_bits = 6u;

  plan.owner_source = OwnerSource::HwId;
  EXPECT_FALSE(plan.is_well_formed());
  plan.owner_sgpr = 40u;
  EXPECT_TRUE(plan.is_well_formed());

  plan.workgroup_sources = WorkgroupSources{};
  plan.workgroup_sources->x.scalar_src = 17u;
  plan.workgroup_sources->x.vector_src = 23u;
  EXPECT_FALSE(plan.is_well_formed());
}

TEST(ConSan, EntryScalarBackupValidatesCarrierAndScalarWindow) {
  constexpr uint16_t kVgprLimit = 256u;
  constexpr uint16_t kSgprLimit = 106u;
  EXPECT_TRUE((EntryScalarBackup{.vgpr = 255u, .sgpr_base = 42u, .sgpr_count = 64u})
                  .is_well_formed(kVgprLimit, kSgprLimit));
  EXPECT_FALSE((EntryScalarBackup{.vgpr = 256u, .sgpr_base = 42u, .sgpr_count = 1u})
                   .is_well_formed(kVgprLimit, kSgprLimit));
  EXPECT_FALSE((EntryScalarBackup{.vgpr = 20u, .sgpr_base = 42u, .sgpr_count = 0u})
                   .is_well_formed(kVgprLimit, kSgprLimit));
  EXPECT_FALSE((EntryScalarBackup{.vgpr = 20u, .sgpr_base = 0u, .sgpr_count = 65u})
                   .is_well_formed(kVgprLimit, kSgprLimit));
  EXPECT_FALSE((EntryScalarBackup{.vgpr = 20u, .sgpr_base = 43u, .sgpr_count = 64u})
                   .is_well_formed(kVgprLimit, kSgprLimit));
}

[[nodiscard]] constexpr bool sgpr_ranges_overlap(uint16_t base, uint16_t width, uint16_t other_base,
                                                 uint16_t other_width) {
  return base < static_cast<uint32_t>(other_base) + other_width &&
         other_base < static_cast<uint32_t>(base) + width;
}

TEST(ConSan, AtomicPhysicalAliasRejectsEverySemanticMismatch) {
  using Semantics = detail::AtomicSemantics;
  struct Candidate {
    uint64_t file_offset = 24u;
    std::string container_name;
    Semantics semantics;
  };
  const Semantics baseline{
      .role = SyncRole::RmwRelease,
      .scope = SyncScope::Agent,
      .outcome = SyncOutcome::RmwNoReturn,
      .byte_count = 4u,
      .descriptor = 0x123u,
      .cas_failure_descriptor = std::nullopt,
  };
  std::array<Semantics, 6> mismatches;
  mismatches.fill(baseline);
  mismatches[0].role = SyncRole::RmwAcquire;
  mismatches[1].scope = SyncScope::System;
  mismatches[2].outcome = SyncOutcome::RmwReturnsOld;
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

    EXPECT_FALSE(detail::canonicalize_physical_site_aliases(
        candidates, errors, "ConSan atomic site",
        [](const Candidate &candidate) { return candidate.file_offset; },
        [](const Candidate &candidate) -> std::string_view { return candidate.container_name; },
        [](const Candidate &lhs, const Candidate &rhs) { return lhs.semantics == rhs.semantics; }));

    ASSERT_EQ(errors.size(), 1u);
    EXPECT_EQ(errors.front(), "ConSan atomic site at file offset 24 was decoded inconsistently "
                              "through aliases 'first' and 'conflict'");
  }
}

TEST(ConSan, WorkgroupSourceRequiresExactlyOneOperandKind) {
  const WorkgroupSource absent;
  EXPECT_FALSE(absent.has_value());
  EXPECT_TRUE(absent.is_well_formed());
  EXPECT_FALSE(absent.operand());

  const WorkgroupSource scalar = WorkgroupSource::scalar(17u);
  EXPECT_TRUE(scalar.has_value());
  EXPECT_TRUE(scalar.is_well_formed());
  EXPECT_EQ(scalar.operand(), 17u);

  const WorkgroupSource vector = WorkgroupSource::vector(23u);
  EXPECT_TRUE(vector.has_value());
  EXPECT_TRUE(vector.is_well_formed());
  EXPECT_EQ(vector.operand(), vector_source_vgpr(23u));

  WorkgroupSource ambiguous = scalar;
  ambiguous.vector_src = 23u;
  EXPECT_FALSE(ambiguous.has_value());
  EXPECT_FALSE(ambiguous.is_well_formed());
  EXPECT_FALSE(ambiguous.operand());

  WorkgroupSource invalid_extraction = scalar;
  invalid_extraction.right_shift = 17u;
  invalid_extraction.low_bit_count = 16u;
  EXPECT_FALSE(invalid_extraction.is_well_formed());

  WorkgroupSource transformed_absent;
  transformed_absent.low_bit_count = 12u;
  EXPECT_FALSE(transformed_absent.is_well_formed());
}

TEST(ConSan, WorkgroupSourcesRejectAnyAmbiguousCoordinate) {
  WorkgroupSources sources{
      .x = WorkgroupSource::scalar(17u),
      .y = WorkgroupSource::vector(23u),
      .z = {},
      .cluster_workgroup_id = WorkgroupSource::private_state(64u),
      .cdna_full_payload_base = std::nullopt,
      .cdna_guest_payload_base = std::nullopt,
      .cdna_guest_payload_mask = 0u,
  };
  EXPECT_TRUE(sources.is_well_formed());
  const WorkgroupSources copy = sources;
  EXPECT_EQ(copy, sources);
  sources.y.scalar_src = 19u;
  EXPECT_FALSE(sources.is_well_formed());
}

TEST(ConSan, FullWorkgroupPayloadRequirementUsesPatchSemantics) {
  constexpr Mode mode = Mode::Default;
  PatchInfo patch;
  patch.owner_descriptor_file_offsets.push_back(64u);
  patch.kind = PatchKind::InlineWatchpointStore;

  EXPECT_TRUE(
      detail::patch_requires_full_workgroup_id_payload(mode, ROCJITSU_CODE_ARCH_RDNA4, patch));
  EXPECT_TRUE(
      detail::patch_requires_full_workgroup_id_payload(mode, ROCJITSU_CODE_ARCH_CDNA5, patch));
  EXPECT_FALSE(
      detail::patch_requires_full_workgroup_id_payload(mode, ROCJITSU_CODE_ARCH_CDNA4, patch));
  EXPECT_FALSE(
      detail::patch_requires_full_workgroup_id_payload(mode, ROCJITSU_CODE_ARCH_INVALID, patch));

  patch.kind = PatchKind::InlineMalformedBarrierAbort;
  EXPECT_FALSE(
      detail::patch_requires_full_workgroup_id_payload(mode, ROCJITSU_CODE_ARCH_RDNA4, patch));

  patch.kind = PatchKind::KernelEntryOwnerEpochPrologue;
  EXPECT_FALSE(
      detail::patch_requires_full_workgroup_id_payload(mode, ROCJITSU_CODE_ARCH_CDNA4, patch));
  patch.persistent_sgpr_state.exact_workgroup = PersistentWorkgroupRegisters{20u, 21u, 22u};
  EXPECT_TRUE(
      detail::patch_requires_full_workgroup_id_payload(mode, ROCJITSU_CODE_ARCH_CDNA4, patch));
  patch.persistent_sgpr_state = {};
  patch.vgpr_state = VgprStateEffect{
      .owner_epoch = {18u, 19u},

      .exact_workgroup = PersistentWorkgroupRegisters{20u, 21u, 22u},
      .owner_epoch_lifetime = OwnerEpochVgprLifetime::CodeObjectPersistent,
  };
  EXPECT_TRUE(
      detail::patch_requires_full_workgroup_id_payload(mode, ROCJITSU_CODE_ARCH_CDNA4, patch));
  patch.vgpr_state.reset();
  patch.private_state_layout = PrivateStateLayout{
      .epoch_offset = 12u,
      .owner_offset = std::nullopt,

      .dispatch_id_offset = std::nullopt,
      .exact_workgroup_offsets = PersistentWorkgroupPrivateOffsets{0u, 4u, 8u},
      .persistent_state_end = 16u,
      .ephemeral_base = 16u,
  };
  EXPECT_TRUE(
      detail::patch_requires_full_workgroup_id_payload(mode, ROCJITSU_CODE_ARCH_CDNA3, patch));

  EXPECT_FALSE(detail::patch_requires_full_workgroup_id_payload(Mode::SuperCollider,
                                                                ROCJITSU_CODE_ARCH_RDNA4, patch));
  patch.owner_descriptor_file_offsets.clear();
  EXPECT_FALSE(
      detail::patch_requires_full_workgroup_id_payload(mode, ROCJITSU_CODE_ARCH_RDNA4, patch));
}

TEST(ConSan, WindowBanksDoNotSystematicallyAliasWorkgroupsAndWaveOwners) {
  // Each emulated lane represents one wave from a two-workgroup histogram:
  // 16 waves per workgroup, with 256 retention banks. The old XOR tuple hash
  // deterministically collapsed all 32 identities onto just 16 banks.
  for (const TargetProfile &target : kTargetProfiles) {
    for (uint32_t dispatch : {0u, 1u, 0x87654321u, 0x95511559u}) {
      SCOPED_TRACE(testing::Message() << rj_code_target_name(target.target) << '/' << dispatch);
      std::vector<uint32_t> words;
      detail::ReportDispatchIdSource dispatch_source;
      dispatch_source.sgpr = 20u;
      WorkgroupSources sources;
      sources.x = WorkgroupSource::vector(43u);
      ASSERT_TRUE(detail::append_window_bank_index(words, dispatch_source, sources, 256u, 40u, 41u,
                                                   42u, target.arch));
      const std::string component = "consan_window_bank_hash";
      amdgpu::GpuMemory memory(component + "_memory");
      amdgpu::L2Cache cache(component + "_cache");
      cache.set_backing_memory(&memory);
      amdgpu::ComputeUnitCore::Config config{};
      config.arch = target.arch;
      config.num_wf_slots = 1;
      config.sgprs_per_wf = 128;
      config.vgprs_per_wf = 256;
      config.lds_size_kb = 64;
      auto compute_unit = amdgpu::ComputeUnitCore::create(component, config, &memory, &cache);
      ASSERT_NE(compute_unit, nullptr);
      amdgpu::Wavefront *wave = compute_unit->dispatch_wf(0, 0, 128, 256);
      ASSERT_NE(wave, nullptr);
      for (size_t index = 0; index < words.size(); ++index)
        memory.write32(index * sizeof(uint32_t), words[index]);
      wave->pc = 0u;
      wave->set_exec(0xffffffffu);
      const uint32_t vgpr_base = wave->vgpr_alloc().base;
      compute_unit->write_sgpr(wave->sgpr_alloc().base + 20u, dispatch);
      compute_unit->write_sgpr(wave->sgpr_alloc().base + 21u, 0u);
      for (uint32_t lane = 0; lane < 32u; ++lane) {
        compute_unit->write_vgpr(vgpr_base + 42u, lane, lane % 16u);
        compute_unit->write_vgpr(vgpr_base + 43u, lane, lane / 16u);
      }
      size_t steps = 0;
      while (wave->pc < words.size() * sizeof(uint32_t)) {
        ASSERT_LT(steps++, words.size());
        compute_unit->step();
      }
      std::set<uint32_t> banks;
      for (uint32_t lane = 0; lane < 32u; ++lane) {
        const uint32_t bank = compute_unit->read_vgpr(vgpr_base + 40u, lane);
        EXPECT_LT(bank, 256u);
        banks.insert(bank);
        EXPECT_EQ(compute_unit->read_vgpr(vgpr_base + 42u, lane), lane % 16u);
        EXPECT_EQ(compute_unit->read_vgpr(vgpr_base + 43u, lane), lane / 16u);
      }
      EXPECT_EQ(banks.size(), 32u);
      if (!wave->is_halted())
        wave->halt();
    }
  }
}

TEST(ConSan, IndexedReportAddressExecutesExactStrideOnEveryTarget) {
  constexpr uint16_t kAddressVgpr = 40u;
  constexpr uint16_t kSlotVgpr = 42u;
  constexpr std::array<uint16_t, 3> kUnrelatedVgprs = {43u, 44u, 45u};
  constexpr std::array<uint32_t, 3> kUnrelatedValues = {0x12345678u, 0x90abcdefu, 0x55aa55aau};
  // Exercise current ConSan table strides and a power-of-two stride.
  constexpr std::array<uint32_t, 4> kStrideBytes = {
      sizeof(CausalWindow),
      sizeof(SyncMetadataPacked),
      sizeof(PendingAcquireSlot),
      64u,
  };
  constexpr std::array<uint32_t, 4> kSlots = {0u, 1u, 3u, 1000u};
  constexpr std::array<uint64_t, 3> kFieldAddresses = {
      0x1234567800000010ull,
      0x00000000ffffff00ull,
      0x12345678fffffff0ull,
  };
  size_t case_index = 0;
  for (const TargetProfile &target : kTargetProfiles) {
    const rj_code_arch_t arch = target.arch;
    for (uint32_t stride_bytes : kStrideBytes) {
      for (uint32_t slot : kSlots) {
        for (uint64_t field_address : kFieldAddresses) {
          SCOPED_TRACE("arch=" + std::to_string(static_cast<uint32_t>(arch)) +
                       " stride=" + std::to_string(stride_bytes) + " slot=" + std::to_string(slot) +
                       " base=" + std::to_string(field_address));
          std::vector<uint32_t> words;
          ASSERT_TRUE(detail::append_indexed_address(words,
                                                     {
                                                         .table_address = field_address,
                                                         .stride_bytes = stride_bytes,
                                                         .address_vgpr = kAddressVgpr,
                                                         .index_vgpr = kSlotVgpr,
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

TEST(ConSan, IndexedReportAddressRejectsInvalidRegistersWithoutPartialOutput) {
  constexpr uint64_t kFieldAddress = 0x1234567800000010ull;
  constexpr uint16_t kAddressVgpr = 40u;
  constexpr uint16_t kSlotVgpr = 42u;
  constexpr uint32_t kStrideBytes = sizeof(CausalWindow);
  const TargetProfile *target = target_profile(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(target, nullptr);
  const auto rejects_without_appending = [&](uint32_t stride_bytes, uint16_t address_vgpr,
                                             uint16_t slot_vgpr,
                                             const TargetProfile &request_target) {
    std::vector<uint32_t> words = {0x12345678u};
    EXPECT_FALSE(detail::append_indexed_address(words,
                                                {
                                                    .table_address = kFieldAddress,
                                                    .stride_bytes = stride_bytes,
                                                    .address_vgpr = address_vgpr,
                                                    .index_vgpr = slot_vgpr,
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
  TargetProfile invalid_target = *target;
  invalid_target.arch = ROCJITSU_CODE_ARCH_INVALID;
  rejects_without_appending(kStrideBytes, kAddressVgpr, kSlotVgpr, invalid_target);
}

TEST(ConSan, SpilledVgprReloadSelectsFixedAndDynamicTargetEncodings) {
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
    EXPECT_EQ(detail::append_reload_spilled_vgpr(fixed_words, fixed, kDestination, kSource, arch),
              detail::SpilledVgprReloadResult::Appended);
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
    EXPECT_EQ(
        detail::append_reload_spilled_vgpr(dynamic_words, dynamic, kDestination, kSource, arch),
        detail::SpilledVgprReloadResult::Appended);
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

TEST(ConSan, SpilledVgprReloadClassifiesRejectedRequestsWithoutAppending) {
  using Result = detail::SpilledVgprReloadResult;
  const auto rejects = [](const VgprSpillSequence &spill, uint16_t destination, uint16_t source,
                          rj_code_arch_t arch, Result expected) {
    std::vector<uint32_t> words = {0x12345678u};
    EXPECT_EQ(detail::append_reload_spilled_vgpr(words, spill, destination, source, arch),
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

TEST(ConSan, SpilledVgprReloadResultNamesAreDistinctAndComplete) {
  using Result = detail::SpilledVgprReloadResult;
  constexpr std::array kResults = {
      Result::Appended,
      Result::SourceOutsideWindow,
      Result::IncompleteSlotMetadata,
      Result::UnsupportedEncoding,
  };
  std::set<std::string_view> names;
  for (Result result : kResults) {
    const std::string_view name = detail::spilled_vgpr_reload_result_name(result);
    EXPECT_NE(name, "unknown");
    EXPECT_TRUE(names.insert(name).second) << name;
  }
}

TEST(ConSan, ResourcePlanAlternativeOutcomeTracksFinalPlanVeto) {
  CandidateResourcePlan plan;
  plan.source = RegisterAllocationSource::SpillRequired;
  plan.reason = RegisterPlanReason::None;
  plan.scratch_vgpr_count = 16u;
  const ResourcePlanAlternative alternative{
      .kind = ResourcePlanAlternativeKind::SpillBackedOperandRecovery,
      .source = RegisterAllocationSource::SpillRequired,
      .reason = RegisterPlanReason::None,
      .scratch_vgpr_count = 16u,
      .outcome = ResourcePlanAlternativeOutcome::Selected,
  };
  EXPECT_EQ(resource_plan_alternative_outcome(plan, alternative),
            ResourcePlanAlternativeOutcome::Selected);

  plan.source = RegisterAllocationSource::Unsupported;
  plan.reason = RegisterPlanReason::ForbiddenOverlap;
  plan.scratch_vgpr.reset();
  EXPECT_EQ(resource_plan_alternative_outcome(plan, alternative),
            ResourcePlanAlternativeOutcome::Vetoed);

  ResourcePlanAlternative rejected = alternative;
  rejected.source = RegisterAllocationSource::Unsupported;
  rejected.reason = RegisterPlanReason::NoLegalWindow;
  rejected.outcome = ResourcePlanAlternativeOutcome::Rejected;
  EXPECT_EQ(resource_plan_alternative_outcome(plan, rejected),
            ResourcePlanAlternativeOutcome::Rejected);
}

TEST(ConSan, PrivateWorkgroupSourceAppliesPackedCoordinateExtraction) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_RDNA4;
  constexpr uint16_t kValueVgpr = 12u;
  constexpr uint32_t kPrivateOffset = 36u;
  for (const auto &[mask_low, extract_high] : {std::pair{true, false}, std::pair{false, true}}) {
    SCOPED_TRACE("mask_low=" + std::to_string(mask_low) +
                 " shift_right=" + std::to_string(extract_high));
    WorkgroupSource source = WorkgroupSource::private_state(kPrivateOffset);
    source.low_bit_count = mask_low ? 16u : 0u;
    source.right_shift = extract_high ? 16u : 0u;
    std::vector<uint32_t> words;

    ASSERT_TRUE(detail::append_workgroup_source_value(words, source, kValueVgpr, kArch));

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

TEST(ConSan, ScalarPersistentTemporaryValidationIsNoopWhenDisabled) {
  TestOptions disabled{};
  std::vector<std::string> errors;
  EXPECT_TRUE(detail::validate_scalar_state_temporaries(
      disabled.persistent_sgprs, owner_epoch_vgpr_sources(disabled.owner_epoch_vgprs),
      "test consumer", errors));
  EXPECT_TRUE(errors.empty());
}

TEST(ConSan, ScalarPersistentTemporaryValidationFailsClosed) {
  for (uint32_t present_mask = 0u; present_mask < 3u; ++present_mask) {
    SCOPED_TRACE(present_mask);
    TestOptions options;
    options.persistent_sgprs.set_owner_epoch(40u, 41u);
    const OwnerEpochVgprSources owner_epoch_vgprs{
        .owner = present_mask & 1u ? std::optional<uint16_t>{6u} : std::nullopt,
        .epoch = present_mask & 2u ? std::optional<uint16_t>{7u} : std::nullopt,
    };
    std::vector<std::string> errors;

    EXPECT_FALSE(detail::validate_scalar_state_temporaries(
        options.persistent_sgprs, owner_epoch_vgprs, "test consumer", errors));
    ASSERT_EQ(errors.size(), 1u);
    EXPECT_NE(errors.front().find("test consumer has no scalar-state VGPR temporaries"),
              std::string::npos);
  }

  TestOptions valid;
  valid.persistent_sgprs.set_owner_epoch(40u, 41u);
  valid.set_owner_epoch_vgprs(6u, 7u);
  std::vector<std::string> errors;
  EXPECT_TRUE(detail::validate_scalar_state_temporaries(
      valid.persistent_sgprs, owner_epoch_vgpr_sources(valid.owner_epoch_vgprs), "test consumer",
      errors));
  EXPECT_TRUE(errors.empty());
}

TEST(ConSan, ScalarOwnerContextResolutionFailsClosedAndComputesTailFloor) {
  using Summary = detail::ScalarOwnerContextSummary;
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
    const auto resolved = detail::resolve_scalar_owner_contexts(true, context, kSingleOwner);
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

  const auto resolved = detail::resolve_scalar_owner_contexts(true, contexts, kOwners);

  ASSERT_TRUE(resolved);
  EXPECT_EQ(resolved->context_indices, (std::vector<size_t>{1u, 0u}));
  EXPECT_EQ(resolved->tail_floor, 80u);

  std::array direct_only_contexts = contexts;
  direct_only_contexts[1].has_indirect_sgpr_access = false;
  // A closed direct-only owner may reclaim allocated but unreferenced SGPRs.
  const auto direct_only =
      detail::resolve_scalar_owner_contexts(true, direct_only_contexts, kOwners);
  ASSERT_TRUE(direct_only);
  EXPECT_EQ(direct_only->tail_floor, 72u);

  EXPECT_FALSE(detail::resolve_scalar_owner_contexts(false, contexts, kOwners));
  EXPECT_FALSE(detail::resolve_scalar_owner_contexts(true, contexts, std::span<const uint64_t>{}));
  constexpr std::array<uint64_t, 1> kMissingOwner = {0x30u};
  EXPECT_FALSE(detail::resolve_scalar_owner_contexts(true, contexts, kMissingOwner));

  std::array invalid_contexts = contexts;
  invalid_contexts[1].descriptor_valid = false;
  EXPECT_FALSE(detail::resolve_scalar_owner_contexts(true, invalid_contexts, kOwners));
  invalid_contexts[1].descriptor_valid = true;
  invalid_contexts[1].descriptor_file_offset = std::nullopt;
  EXPECT_FALSE(detail::resolve_scalar_owner_contexts(true, invalid_contexts, kOwners));

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
  const auto skipped_invalid =
      detail::resolve_scalar_owner_contexts(true, contexts_with_unrelated_invalid, kValidOwner);
  ASSERT_TRUE(skipped_invalid);
  EXPECT_EQ(skipped_invalid->context_indices, (std::vector<size_t>{1u}));
  EXPECT_EQ(skipped_invalid->tail_floor, 48u);
}

TEST(ConSan, ScalarOwnerWindowQualificationUsesEveryTailAndPhysicalVcc) {
  using Range = detail::ScalarOwnerSgprRange;
  using Summary = detail::ScalarOwnerContextSummary;
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

  EXPECT_FALSE(
      detail::scalar_owner_contexts_conflict_with_physical_vcc(direct_owners, misses_both_vcc));
  EXPECT_TRUE(detail::scalar_owner_contexts_conflict_with_physical_vcc(direct_owners,
                                                                       dispatch_hits_second_vcc));
  EXPECT_TRUE(
      detail::scalar_owner_contexts_conflict_with_physical_vcc(direct_owners, hits_second_vcc));
  const std::array streamk_transient_state = {Range{4u, 30u}, Range{34u, 2u}, Range{36u, 1u},
                                              Range{76u, 1u}};
  const std::array streamk_owner = {
      Summary{.descriptor_file_offset = 0x30u,
              .current_sgpr_count = 80u,
              .max_referenced_sgpr_count = 74u,
              .sgpr_reference_coverage_complete = true,
              .descriptor_valid = true},
  };
  // A previously qualified dispatch pair at s80:s81 grows the allocation to
  // 88 SGPRs and moves physical VCC to s82:s83.  The later transient search
  // must use that same final allocation when qualifying its s76 latch.
  EXPECT_TRUE(detail::scalar_owner_contexts_conflict_with_physical_vcc(streamk_owner,
                                                                       streamk_transient_state));
  EXPECT_FALSE(detail::scalar_owner_contexts_conflict_with_physical_vcc(
      streamk_owner, streamk_transient_state, /*required_sgpr_count_floor=*/82u));
  EXPECT_TRUE(detail::scalar_owner_contexts_admit_reserved_window(direct_owners, 12u, 2u,
                                                                  /*protect_physical_vcc=*/true));

  std::array indirect_owners = direct_owners;
  indirect_owners[1].has_indirect_sgpr_access = true;
  EXPECT_FALSE(detail::scalar_owner_contexts_admit_reserved_window(indirect_owners, 12u, 2u,
                                                                   /*protect_physical_vcc=*/false));
  EXPECT_TRUE(detail::scalar_owner_contexts_admit_reserved_window(indirect_owners, 48u, 2u,
                                                                  /*protect_physical_vcc=*/false));

  std::array invalid_owners = direct_owners;
  invalid_owners[1].descriptor_valid = false;
  EXPECT_TRUE(
      detail::scalar_owner_contexts_conflict_with_physical_vcc(invalid_owners, misses_both_vcc));
  EXPECT_FALSE(detail::scalar_owner_contexts_admit_reserved_window(invalid_owners, 48u, 2u,
                                                                   /*protect_physical_vcc=*/false));
  EXPECT_TRUE(detail::scalar_owner_contexts_conflict_with_physical_vcc(std::span<const Summary>{},
                                                                       misses_both_vcc));
  EXPECT_FALSE(detail::scalar_owner_contexts_admit_reserved_window(
      std::span<const Summary>{}, 48u, 2u, /*protect_physical_vcc=*/false));
}

TEST(ConSan, Cdna4HeterogeneousOwnersKeepUsableComponentWith) {
  constexpr uint64_t kHighPressureEntry = 320u * sizeof(uint32_t);
  for (uint16_t live_sgpr_count : {96u, 98u}) {
    SCOPED_TRACE("live_sgprs=" + std::to_string(live_sgpr_count));
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
    TestOptions options = test_options();
    options.track_barriers = false;
    options.track_atomics = false;
    options.runtime_sample_stride = 2u;
    options.report_buffer_address = 0x123456780000ull;
    options.report_buffer_size = 64u * 1024u * 1024u;

    const TransformArtifacts result = test_lower_consan(bytes, options);

    ASSERT_TRUE(patch_succeeded(result)) << testing::PrintToString(result.errors);
    ASSERT_TRUE(result.modified()) << testing::PrintToString(result.warnings);
    EXPECT_EQ(result.outcome, TransformOutcome::ModifiedValid);
    ASSERT_TRUE(test_dispatch_id_sgpr(result));
    ASSERT_TRUE(test_exec_save_sgpr(result));
    EXPECT_EQ(*test_dispatch_id_sgpr(result), 96u);
    EXPECT_LT(*test_exec_save_sgpr(result), 96u);

    const IntentCoverageEntry *low = access_coverage_at(result, 0u);
    const IntentCoverageEntry *high = access_coverage_at(result, kHighPressureEntry);
    ASSERT_NE(low, nullptr);
    ASSERT_NE(high, nullptr);
    EXPECT_EQ(low->lowering, LoweringOutcomeKind::Instrumented);
    EXPECT_EQ(high->lowering, LoweringOutcomeKind::ResourceRejected);
    const auto high_plan = std::ranges::find(result.resource_plans, kHighPressureEntry,
                                             &CandidateResourcePlan::text_offset);
    ASSERT_NE(high_plan, result.resource_plans.end());
    EXPECT_EQ(high_plan->reason, RegisterPlanReason::ForbiddenOverlap);
    EXPECT_TRUE(std::ranges::any_of(result.patches, [](const PatchInfo &patch) {
      return patch.phase == PatchPhase::Instrumentation && patch.anchor_offset == 0u;
    }));
    EXPECT_TRUE(std::ranges::none_of(result.patches, [=](const PatchInfo &patch) {
      return patch.phase == PatchPhase::Instrumentation &&
             patch.anchor_offset == kHighPressureEntry;
    }));
    if (live_sgpr_count == 98u) {
      EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
        return warning.find("reserved a high dispatch-ID SGPR pair") != std::string::npos;
      }));
    }
  }
}

TEST(ConSan, Gfx1250Wave32DescriptorUsesSixteenVgprGranules) {
  constexpr auto store = cdna5::build_vds(cdna5::kDsStoreB32Vds, {.addr = 0, .data0 = 1});
  const std::array<uint32_t, 3> text_words = {store[0], store[1],
                                              build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5)};
  const std::vector<uint8_t> bytes =
      make_gfx1250_code_object(text_words, "lds_probe", /*vgpr_granulated=*/4);
  TestOptions options = test_options();

  const TransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_TRUE(patch_succeeded(result));
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().current_vgpr_count, 80u);
}

TEST(ConSan, Cdna3Wave64DescriptorUsesEightVgprGranules) {
  constexpr auto store = cdna3::build_ds(cdna3::kDsWriteB32Ds, {.addr = 0, .data0 = 1});
  const std::array<uint32_t, 3> text_words = {store[0], store[1],
                                              build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA3)};
  const std::vector<uint8_t> bytes =
      make_cdna3_lds_code_object(text_words, "lds_probe", /*vgpr_granulated=*/3);
  TestOptions options = test_options();

  const TransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_TRUE(patch_succeeded(result));
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().current_vgpr_count, 32u);
}

TEST(ConSan, Gfx1250TwoAddressLoadUsesNormalizedRangesAndSafeScratch) {
  constexpr auto load = cdna5::build_vds(cdna5::kDsLoad2addrStride64B32Vds,
                                         {.offset0 = 3, .offset1 = 5, .addr = 0, .vdst = 1});
  const std::array<uint32_t, 3> text_words = {load[0], load[1],
                                              build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5)};
  const std::vector<uint8_t> bytes = make_gfx1250_code_object(text_words, "two_address_load");
  TestOptions options = test_options();
  options.report_buffer_address = 0x123456780000ull;
  options.report_buffer_size = direct_report_bytes(8);

  const TransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_TRUE(patch_succeeded(result));
  ASSERT_TRUE(result.modified()) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.program_inventory.access_sites().size(), 1u);
  ASSERT_EQ(test_admitted_accesses(result).size(), 1u);
  const ProgramSite &site = result.program_inventory.access_sites().front();
  Candidate candidate(site);
  ASSERT_EQ(site.ranges.size(), 2u);
  EXPECT_EQ(candidate.site().ranges, site.ranges);
  for (size_t range_index = 0; range_index < site.ranges.size(); ++range_index) {
    SCOPED_TRACE(range_index);
    ASSERT_TRUE(site.ranges[range_index].static_byte_offset);
    EXPECT_EQ(candidate.lowering_offset(candidate.site().ranges[range_index]),
              *site.ranges[range_index].static_byte_offset);
  }
  EXPECT_EQ(candidate.lowering_offset(candidate.site().ranges[0]), 3u * 256u);
  EXPECT_EQ(candidate.lowering_offset(candidate.site().ranges[1]), 5u * 256u);

  const TargetProfile *gfx1250 = target_profile(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(gfx1250, nullptr);
  EXPECT_TRUE(detail::guest_access_relocation_requires_adjusted_address(candidate, *gfx1250));
  std::vector<std::string> relocation_errors;
  const auto relocated =
      detail::build_relocated_guest_access_words({.image = bytes,
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

  for (const TargetProfile &target : kTargetProfiles) {
    if (target.requires_split_two_address_lds_relocation)
      continue;
    SCOPED_TRACE(rj_code_target_name(target.target));
    EXPECT_FALSE(detail::guest_access_relocation_requires_adjusted_address(candidate, target));
    relocation_errors.clear();
    const auto copied =
        detail::build_relocated_guest_access_words({.image = bytes,
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
  EXPECT_EQ(*result.resource_plans.front().scratch_vgpr % 2u, 0u)
      << "CDNA5 instrumentation uses the first scratch pair for FLAT addresses";
  ASSERT_EQ(non_entry_prologue_patch_count(result), 1u);
  EXPECT_EQ(result.patches.front().kind, PatchKind::TrampolineWatchpointStore);
}

TEST(ConSan, Gfx1250SelfClobberingTwoAddressLoadRetainsCombinedGuestSemantics) {
  constexpr auto load = cdna5::build_vds(
      cdna5::kDsLoad2addrB64Vds, {.offset0 = 28u, .offset1 = 30u, .addr = 30u, .vdst = 30u});
  const std::array<uint32_t, 3> text_words = {load[0], load[1],
                                              build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5)};
  const std::vector<uint8_t> bytes =
      make_gfx1250_code_object(text_words, "self_clobbering_two_address_load");
  TestOptions options = test_options();
  options.scratch_vgpr = 44u;
  options.exec_save_sgpr = 80u;
  options.set_owner_epoch_vgprs(60u, 61u);
  options.report_buffer_address = 0x123456780000ull;
  options.report_buffer_size = direct_report_bytes(2u);
  options.track_barriers = false;
  options.track_atomics = false;

  const TransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_TRUE(patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified()) << testing::PrintToString(result.warnings);
  const auto access =
      std::ranges::find(result.patches, PatchKind::TrampolineWatchpointStore, &PatchInfo::kind);
  ASSERT_NE(access, result.patches.end());
  ASSERT_TRUE(access->relocated_guest_instruction_offset);
  AmdGpuCodeObject patched(result.replacement.data(), result.replacement.size());
  ASSERT_TRUE(patched.is_valid());
  EXPECT_EQ(text_words_at_offset(patched, *access->relocated_guest_instruction_offset,
                                 load.size() * sizeof(uint32_t)),
            (std::vector<uint32_t>{load[0], load[1]}));
}

TEST(ConSan, GuestAccessRelocationRejectsIncompleteTypedRequest) {
  std::vector<std::string> errors;
  EXPECT_FALSE(detail::build_relocated_guest_access_words({.image = {},
                                                           .candidate = nullptr,
                                                           .target = nullptr,
                                                           .replay_address_vgpr = 0u,
                                                           .adjusted_address_vgpr = std::nullopt},
                                                          errors));
  ASSERT_EQ(errors.size(), 1u);
  EXPECT_EQ(errors.front(), "ConSan guest relocation requires a candidate and target profile");
}

struct NativeB96Access {
  std::array<uint32_t, 2> words;
  std::string_view mnemonic;
};

template <size_t AccessCount, typename CodeObjectFactory>
void expect_admits_native_b96_accesses(rj_code_arch_t arch,
                                       const std::array<NativeB96Access, AccessCount> &accesses,
                                       CodeObjectFactory code_object_factory) {
  for (const auto &[access, mnemonic] : accesses) {
    SCOPED_TRACE(std::string("default") + " " + std::string(mnemonic));
    const std::array<uint32_t, 3> text_words = {access[0], access[1], build_s_endpgm(arch)};
    const std::vector<uint8_t> bytes = code_object_factory(text_words);
    TestOptions options = test_options();
    options.track_atomics = false;
    options.track_barriers = false;
    options.runtime_sample_stride = 2u;
    options.report_buffer_address = 0x123456780000ull;
    options.report_buffer_size = 64u * 1024u * 1024u;

    const TransformArtifacts result = test_lower_consan(bytes, options);

    ASSERT_TRUE(patch_succeeded(result)) << testing::PrintToString(result.errors);
    ASSERT_TRUE(result.modified()) << testing::PrintToString(result.warnings);
    ASSERT_EQ(test_admitted_accesses(result).size(), 1u);
    EXPECT_EQ(test_admitted_accesses(result).front().mnemonic_view(), mnemonic);
    EXPECT_EQ(test_admitted_accesses(result).front().decoded_width_bits, 96u);
    const PatchKind expected_patch_kind = PatchKind::TrampolineWatchpointStore;
    const auto access_patch =
        std::ranges::find(result.patches, expected_patch_kind, &PatchInfo::kind);
    ASSERT_NE(access_patch, result.patches.end());
    ASSERT_TRUE(access_patch->scratch_vgpr);
    AmdGpuCodeObject patched(result.replacement.data(), result.replacement.size());
    ASSERT_TRUE(patched.is_valid());
    const std::vector<uint32_t> body = text_words_at_offset(
        patched, access_patch->trampoline_offset, access_patch->trampoline_size);
    {
      const uint16_t high_vgpr = static_cast<uint16_t>(*access_patch->scratch_vgpr + 3u);
      const uint16_t tmp_vgpr = static_cast<uint16_t>(*access_patch->scratch_vgpr + 4u);
      const uint32_t encoded_twelve_byte_count = encode_byte_count(12u)
                                                 << (watchpoint::count_shift - 32u);
      const auto byte_count_literal =
          instrumentation::build_v_mov_b32_literal(tmp_vgpr, encoded_twelve_byte_count, arch);
      const auto add_byte_count = instrumentation::build_v_add_u32(
          high_vgpr, vector_source_vgpr(high_vgpr), tmp_vgpr, arch);
      ASSERT_TRUE(byte_count_literal && add_byte_count);
      std::vector<uint32_t> expected = *byte_count_literal;
      expected.insert(expected.end(), add_byte_count->begin(), add_byte_count->end());
      EXPECT_TRUE(contains_subsequence(body, expected));
    }
    EXPECT_EQ(access_lowering_count(result, LoweringOutcomeKind::Instrumented), 1u);
  }
}

TEST(ConSan, Gfx1250AdmitsNativeB96Accesses) {
  constexpr auto store =
      cdna5::build_vds(cdna5::kDsStoreB96Vds, {.offset0 = 12, .addr = 0, .data0 = 1});
  constexpr auto load =
      cdna5::build_vds(cdna5::kDsLoadB96Vds, {.offset0 = 12, .addr = 0, .vdst = 4});
  constexpr auto aliasing_load =
      cdna5::build_vds(cdna5::kDsLoadB96Vds, {.offset0 = 12, .addr = 0, .vdst = 0});
  constexpr std::array<NativeB96Access, 3> accesses = {
      NativeB96Access{store, "ds_store_b96"},
      NativeB96Access{load, "ds_load_b96"},
      NativeB96Access{aliasing_load, "ds_load_b96"},
  };

  expect_admits_native_b96_accesses(ROCJITSU_CODE_ARCH_CDNA5, accesses, [](const auto &text_words) {
    return make_gfx1250_code_object(text_words, "native_b96_access");
  });
}

TEST(ConSan, Gfx1250RelaxedLdsAtomicIsAccessButNotSynchronization) {
  constexpr auto atomic = cdna5::build_vds(
      cdna5::kDsCmpstoreRtnB32Vds, {.offset0 = 12, .addr = 0, .data0 = 1, .data1 = 2, .vdst = 3});
  const std::array<uint32_t, 3> text_words = {atomic[0], atomic[1],
                                              build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA5)};
  const std::vector<uint8_t> bytes = make_gfx1250_code_object(text_words, "relaxed_lds_atomic");
  TestOptions options = test_options();
  options.track_atomics = true;
  options.report_buffer_address = 0x123456780000ull;
  options.report_buffer_size = direct_report_bytes(8);

  const TransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_TRUE(patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified()) << "warnings=" << testing::PrintToString(result.warnings)
                                 << " plans=" << testing::PrintToString(result.resource_plans);
  ASSERT_EQ(test_admitted_accesses(result).size(), 1u);
  EXPECT_EQ(test_admitted_accesses(result).front().kind, LdsAccessKind::Atomic);
  EXPECT_EQ(test_admitted_accesses(result).front().mnemonic_view(), "ds_cmpstore_rtn_b32");
  ASSERT_EQ(result.program_inventory.access_sites().size(), 1u);
  EXPECT_NE(result.program_inventory.access_sites().front().get_if<AtomicSite>(), nullptr);
  EXPECT_EQ(access_lowering_count(result, LoweringOutcomeKind::Instrumented), 1u);
  ASSERT_EQ(result.observation_plan().atomic_site_decisions.size(), 1u);
  EXPECT_EQ(result.observation_plan().atomic_site_decisions.front().kind,
            SiteDecisionKind::NotApplicable);
  EXPECT_EQ(result.observation_plan().atomic_site_decisions.front().reason,
            AtomicPolicyReason::MissingDirectionalAccessWindow);
}

TEST(ConSan, Cdna4HistogramLdsAtomicsAreAccessesButNotSynchronization) {
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
  TestOptions options = test_options();
  options.track_barriers = false;
  options.track_atomics = true;
  options.report_buffer_address = 0x123456780000ull;
  options.report_buffer_size = direct_report_bytes(8);

  const TransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_TRUE(patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified()) << testing::PrintToString(result.warnings);
  ASSERT_EQ(test_admitted_accesses(result).size(), 5u);
  EXPECT_TRUE(std::ranges::all_of(test_admitted_accesses(result), [](const auto &candidate) {
    return candidate.kind == LdsAccessKind::Atomic;
  }));
  EXPECT_EQ(access_decision_count(result, SiteDecisionKind::Admitted), 5u);
  EXPECT_GT(access_lowering_count(result, LoweringOutcomeKind::Instrumented), 0u);
  EXPECT_TRUE(std::ranges::none_of(result.patches, [](const PatchInfo &patch) {
    return patch.kind == PatchKind::TrampolineSyncMetadata;
  }));

  EXPECT_FALSE(shadow_kind_conflicts(ShadowAccessKind::Atomic, ShadowAccessKind::Atomic));
  EXPECT_TRUE(shadow_kind_conflicts(ShadowAccessKind::Atomic, ShadowAccessKind::Read));
  EXPECT_TRUE(shadow_kind_conflicts(ShadowAccessKind::Atomic, ShadowAccessKind::Write));
}

TEST(ConSan, Rdna4HistogramLdsAtomicsAreAccessesButNotSynchronization) {
  constexpr auto add_u32 =
      rdna4::build_vds(rdna4::kDsAddU32Vds, {.offset0 = 4, .addr = 3, .data0 = 7});
  constexpr auto add_u64 =
      rdna4::build_vds(rdna4::kDsAddU64Vds, {.offset0 = 8, .addr = 5, .data0 = 8});
  constexpr auto add_f32 =
      rdna4::build_vds(rdna4::kDsAddF32Vds, {.offset0 = 12, .addr = 7, .data0 = 3});
  constexpr auto cmpst =
      rdna4::build_vds(rdna4::kDsCmpstoreRtnB32Vds,
                       {.offset0 = 20, .addr = 12, .data0 = 11, .data1 = 13, .vdst = 13});
  const std::array<uint32_t, 9> text_words = {
      add_u32[0], add_u32[1], add_u64[0],
      add_u64[1], add_f32[0], add_f32[1],
      cmpst[0],   cmpst[1],   build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "rdna4_histogram_lds_atomics");
  TestOptions options = test_options();
  options.max_patches = 4;
  options.track_barriers = false;
  options.track_atomics = true;
  options.report_buffer_address = 0x123456780000ull;
  options.report_buffer_size = direct_report_bytes(8);

  const TransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_TRUE(patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified()) << testing::PrintToString(result.warnings);
  ASSERT_EQ(test_admitted_accesses(result).size(), 4u);
  EXPECT_TRUE(std::ranges::all_of(test_admitted_accesses(result), [](const auto &candidate) {
    return candidate.kind == LdsAccessKind::Atomic;
  }));
  EXPECT_EQ(access_decision_count(result, SiteDecisionKind::Admitted), 4u);
  EXPECT_EQ(access_lowering_count(result, LoweringOutcomeKind::Instrumented), 4u);
  EXPECT_TRUE(std::ranges::none_of(result.patches, [](const PatchInfo &patch) {
    return patch.kind == PatchKind::TrampolineSyncMetadata;
  }));

  EXPECT_FALSE(shadow_kind_conflicts(ShadowAccessKind::Atomic, ShadowAccessKind::Atomic));
  EXPECT_TRUE(shadow_kind_conflicts(ShadowAccessKind::Atomic, ShadowAccessKind::Read));
  EXPECT_TRUE(shadow_kind_conflicts(ShadowAccessKind::Atomic, ShadowAccessKind::Write));
}

TEST(ConSan, UnassociatedFenceIsNotApplicableOnEverySupportedTarget) {
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
    TestOptions options = test_options();
    options.track_atomics = true;

    const TransformArtifacts result = test_lower_consan(bytes, options);

    ASSERT_TRUE(patch_succeeded(result)) << testing::PrintToString(result.errors);
    ASSERT_EQ(result.program_inventory.sync().fence_candidates.size(), 1u);
    EXPECT_FALSE(result.program_inventory.sync().fence_candidates.front().eligible());
    ASSERT_EQ(result.observation_plan().fence_site_decisions.size(), 1u);
    EXPECT_EQ(result.observation_plan().fence_site_decisions.front().kind,
              SiteDecisionKind::NotApplicable);
    EXPECT_EQ(result.observation_plan().fence_site_decisions.front().reason,
              FencePolicyReason::AssociationUnavailable);
  }
}

TEST(ConSan, Cdna4UnassociatedFenceIsNotApplicable) {
  const auto fence = build_cdna4_s_dcache_inv_vol(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_TRUE(fence);
  std::vector<uint32_t> text_words(fence->begin(), fence->end());
  text_words.push_back(build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4));
  const std::vector<uint8_t> bytes = make_cdna4_lds_code_object(text_words, "unassociated_fence");
  TestOptions options = test_options();
  options.track_atomics = true;

  const TransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_TRUE(patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_EQ(result.program_inventory.sync().fence_candidates.size(), 1u);
  EXPECT_FALSE(result.program_inventory.sync().fence_candidates.front().eligible());
  ASSERT_EQ(result.observation_plan().fence_site_decisions.size(), 1u);
  EXPECT_EQ(result.observation_plan().fence_site_decisions.front().kind,
            SiteDecisionKind::NotApplicable);
  EXPECT_EQ(result.observation_plan().fence_site_decisions.front().reason,
            FencePolicyReason::AssociationUnavailable);
}

TEST(ConSan, InventoriesDynamicStackMarker) {
  const std::array<uint32_t, 3> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32 v0, v0
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "dynamic_stack_kernel", kRdna4Wave64AllVgprsGranulated,
                                 /*wave32=*/false, /*uses_dynamic_stack=*/true);
  TestOptions options = test_options();

  const auto result = test_lower_consan(bytes, options);

  ASSERT_TRUE(patch_succeeded(result));
  ASSERT_EQ(result.program_inventory.kernels().size(), 1u);
  ASSERT_TRUE(result.program_inventory.kernels().front().uses_dynamic_stack.has_value());
  EXPECT_TRUE(*result.program_inventory.kernels().front().uses_dynamic_stack);
}

TEST(ConSan, DynamicStackMetadataOverridesZeroValuedMarker) {
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
  TestOptions options = test_options();

  const auto result = test_lower_consan(bytes, options);

  ASSERT_TRUE(patch_succeeded(result));
  ASSERT_EQ(result.program_inventory.kernels().size(), 1u);
  ASSERT_TRUE(result.program_inventory.kernels().front().uses_dynamic_stack.has_value());
  EXPECT_TRUE(*result.program_inventory.kernels().front().uses_dynamic_stack);
  ASSERT_TRUE(result.program_inventory.kernels().front().sgpr_count.has_value());
  EXPECT_EQ(*result.program_inventory.kernels().front().sgpr_count, 83u);
  ASSERT_FALSE(result.resource_plans.empty());
  for (const CandidateResourcePlan &plan : result.resource_plans)
    EXPECT_EQ(plan.max_referenced_sgpr_count, 83u);
}

TEST(ConSan, InventoriesHiddenDynamicLdsArgument) {
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

  const TransformArtifacts result = test_lower_consan(bytes, test_options());

  ASSERT_TRUE(patch_succeeded(result));
  ASSERT_EQ(result.program_inventory.kernels().size(), 1u);
  EXPECT_TRUE(result.program_inventory.kernels().front().has_dynamic_lds);
}

TEST(ConSan, InventoryIncludesLikelyGroupFlatSitesFromLocalFunctions) {
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
  TestOptions options = test_options();

  const auto result = test_lower_consan(bytes, options);

  ASSERT_TRUE(patch_succeeded(result));
  ASSERT_EQ(result.program_inventory.functions().size(), 1u);
  ASSERT_EQ(result.program_inventory.access_sites().size(), 1u);
  ASSERT_EQ(test_admitted_accesses(result).size(), 1u);
  const ProgramSite candidate = test_admitted_accesses(result).front();
  EXPECT_EQ(candidate.origin, AccessOrigin::Flat);
  EXPECT_EQ(candidate.kind, LdsAccessKind::Read);
  EXPECT_EQ(candidate.flat_address_space_hint, FlatAddressSpaceHint::Group);
  ASSERT_NE(test_program_container(result, candidate), nullptr);
  EXPECT_EQ(test_program_container(result, candidate)->kind, ProgramContainerKind::Function);
  EXPECT_EQ(test_program_container_name(result, candidate), "lds_helper");
  EXPECT_EQ(candidate.mnemonic_view(), "flat_load_b32");
  EXPECT_EQ(candidate.physical_id.original_text_offset, 24u);
  EXPECT_EQ(candidate.decoded_file_offset(), 0x118u);
  EXPECT_EQ(candidate.size(), 3u * sizeof(uint32_t));
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
  EXPECT_TRUE(result.resource_plans.front().owner_kernel_ids.empty());
  EXPECT_EQ(result.resource_plans.front().source, RegisterAllocationSource::Unsupported);
  EXPECT_EQ(result.resource_plans.front().reason, RegisterPlanReason::MissingOwner);
  EXPECT_EQ(test_resource_plan_summary(result).unsupported_plans, 1u);
}

TEST(ConSan, LoweringOffsetsPreserveNativeRangesAndDoNotReapplyFlatImmediate) {
  ProgramSite candidate_site;
  AccessRange range{.id = {}, .static_byte_offset = -8, .byte_width = 4u};
  candidate_site.origin = AccessOrigin::Flat;
  candidate_site.lowering.form.emplace();
  candidate_site.lowering.form->kind = AccessLoweringFormKind::FlatVectorAddress;
  Candidate candidate(candidate_site);
  EXPECT_EQ(candidate.lowering_offset(range), 0u);

  candidate_site.origin = AccessOrigin::NativeLds;
  candidate_site.lowering.form->kind = AccessLoweringFormKind::NativeSingleRange;
  range.static_byte_offset = 3u * 256u;
  EXPECT_EQ(candidate.lowering_offset(range), 3u * 256u);
}

TEST(ConSan, SharedHelperPlanUsesCommonDeadWindowAcrossTwoOwners) {
  const std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object();
  TestOptions options = test_options();

  const auto result = test_lower_consan(bytes, options);

  ASSERT_TRUE(patch_succeeded(result));
  ASSERT_EQ(result.program_inventory.kernels().size(), 3u);
  ASSERT_EQ(result.program_inventory.functions().size(), 1u);
  ASSERT_EQ(test_admitted_accesses(result).size(), 1u);
  const ProgramSite candidate = test_admitted_accesses(result).front();
  ASSERT_NE(test_program_container(result, candidate), nullptr);
  EXPECT_EQ(test_program_container(result, candidate)->kind, ProgramContainerKind::Function);
  EXPECT_EQ(test_program_container_name(result, candidate), "shared_lds_helper");
  ASSERT_EQ(result.resource_plans.size(), 1u);
  const CandidateResourcePlan &plan = result.resource_plans.front();
  ASSERT_EQ(plan.owner_kernel_ids.size(), 2u);
  EXPECT_EQ(plan.source, RegisterAllocationSource::LivenessDead);
  EXPECT_EQ(plan.reason, RegisterPlanReason::None);
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

TEST(ConSan, Cdna4DirectScalarStateReusesUnreferencedSharedOwnerAllocation) {
  const std::vector<uint8_t> bytes =
      make_cdna4_shared_scalar_owner_code_object(/*indirect_sgpr_access=*/false);
  ASSERT_FALSE(bytes.empty());

  TestOptions options = test_options();
  options.test_force_vgpr_spill = true;
  options.runtime_sample_stride = 2u;
  options.track_barriers = false;
  options.track_atomics = true;
  options.report_buffer_address = 0x123456780000ull;
  options.report_buffer_size = direct_report_bytes(2);
  options.max_patches = 1u;

  const TransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_TRUE(patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(result.modified()) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(test_persistent_sgpr_state(result).owner());
  ASSERT_FALSE(result.resource_plans.empty());
  EXPECT_LT(*test_persistent_sgpr_state(result).owner(), 80u);
  EXPECT_EQ(result.outcome, TransformOutcome::ModifiedValid);
}

TEST(ConSan, Cdna4ScalarStateClearsEverySharedOwnerAllocation) {
  const std::vector<uint8_t> bytes =
      make_cdna4_shared_scalar_owner_code_object(/*indirect_sgpr_access=*/true);
  ASSERT_FALSE(bytes.empty());

  TestOptions options = test_options();
  options.test_force_vgpr_spill = true;
  options.runtime_sample_stride = 2u;
  options.track_barriers = false;
  options.track_atomics = true;
  options.init_owner_epoch = false;
  options.report_buffer_address = 0x123456780000ull;
  options.report_buffer_size = direct_report_bytes(2);
  options.max_patches = 1u;

  const TransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_TRUE(patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_TRUE(test_dispatch_id_sgpr(result));
  ASSERT_TRUE(test_exec_save_sgpr(result));
  EXPECT_GE(*test_dispatch_id_sgpr(result), 80u);
  EXPECT_GE(*test_exec_save_sgpr(result), 80u);
  for (const TransientSgprAssignment &assignment : test_transient_sgpr_assignments(result)) {
    EXPECT_GE(assignment.exec_save_sgpr, 80u);
    if (assignment.dispatch_id_sgpr) {
      EXPECT_GE(*assignment.dispatch_id_sgpr, 80u);
    }
  }
  // The automatic layout exactly packs dispatch state, persistent identity,
  // and the transient window into s80:s96 while moving physical VCC above
  // them. ConSan must retain that legal boundary solution.
  ASSERT_TRUE(result.modified()) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(test_persistent_sgpr_state(result).owner())
      << testing::PrintToString(result.warnings);
  ASSERT_TRUE(test_persistent_sgpr_state(result).epoch());
  EXPECT_GE(*test_persistent_sgpr_state(result).owner(), 80u);
  EXPECT_EQ(*test_persistent_sgpr_state(result).epoch(),
            *test_persistent_sgpr_state(result).owner() + 1u);

  const auto access_patch = std::ranges::find_if(result.patches, [](const PatchInfo &patch) {
    return patch.phase == PatchPhase::Instrumentation &&
           patch.owner_descriptor_file_offsets.size() == 2u;
  });
  ASSERT_NE(access_patch, result.patches.end());
  ASSERT_FALSE(result.resource_plans.empty());
  EXPECT_EQ(result.outcome, TransformOutcome::ModifiedValid);
}

TEST(ConSan, Cdna4SharedExecSaveAvoidsEveryOwnerPhysicalVcc) {
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

  TestOptions options = test_options();
  options.scratch_vgpr = 12u;
  options.set_owner_epoch_vgprs(40u, 41u);
  options.track_barriers = false;
  options.track_atomics = false;
  options.report_buffer_address = 0x123456780000ull;
  options.report_buffer_size = direct_report_bytes(8);
  options.max_patches = 1u;

  const TransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_TRUE(patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_NE(result.resource_plans.front().source, RegisterAllocationSource::Unsupported);
  EXPECT_EQ(result.resource_plans.front().owner_kernel_ids.size(), 2u);
  ASSERT_TRUE(result.modified()) << testing::PrintToString(result.warnings);
  ASSERT_TRUE(test_dispatch_id_sgpr(result));
  ASSERT_TRUE(test_exec_save_sgpr(result));
  // The seven-SGPR ConSan window must avoid both owners' physical VCC pairs.
  constexpr std::array<uint16_t, 2> kOriginalVccBases = {26u, 42u};
  for (uint16_t vcc_base : kOriginalVccBases) {
    EXPECT_FALSE(sgpr_ranges_overlap(*test_dispatch_id_sgpr(result), 2u, vcc_base, 2u));
    EXPECT_FALSE(sgpr_ranges_overlap(*test_exec_save_sgpr(result), 7u, vcc_base, 2u));
  }
  EXPECT_EQ(result.outcome, TransformOutcome::ModifiedValid);
}

TEST(ConSan, SharedHelperAtomicUsesCommonOwnerResourcePlan) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.helper_has_ordered_atomic = true;
  fixture.helper_atomic_has_lds_producer = true;
  const std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object(fixture);
  ASSERT_FALSE(bytes.empty());
  TestOptions options = test_options();
  options.track_atomics = true;
  options.max_patches = 8u;
  options.report_buffer_address = 0x123456780000ull;
  options.report_buffer_size = direct_report_bytes(8);

  const auto result = test_lower_consan(bytes, options);

  ASSERT_TRUE(patch_succeeded(result));
  ASSERT_EQ(result.program_inventory.functions().size(), 1u);
  ASSERT_EQ(test_decoded_sites<AtomicSite>(result.program_inventory,
                                           result.program_inventory.functions().front())
                .size(),
            1u);
  ASSERT_EQ(result.resource_plans.size(), 2u);
  const auto plan_it = std::ranges::find_if(result.resource_plans, [](const auto &item) {
    return item.site_kind == ResourceSiteKind::Atomic;
  });
  ASSERT_NE(plan_it, result.resource_plans.end());
  const CandidateResourcePlan &plan = *plan_it;
  EXPECT_EQ(plan.site_kind, ResourceSiteKind::Atomic);
  EXPECT_EQ(plan.source, RegisterAllocationSource::LivenessDead);
  ASSERT_EQ(plan.owner_kernel_ids.size(), 2u);
  ASSERT_TRUE(result.modified()) << testing::PrintToString(result.warnings);
  EXPECT_TRUE(test_owner_vgpr(result));
  EXPECT_TRUE(test_epoch_vgpr(result));
  EXPECT_TRUE(test_exact_workgroup_vgprs(result).complete());
  ASSERT_TRUE(test_exec_save_sgpr(result));
  const auto atomic_patch = std::ranges::find_if(result.patches, [](const PatchInfo &patch) {
    return patch.kind == PatchKind::TrampolineSyncMetadata;
  });
  ASSERT_NE(atomic_patch, result.patches.end());
  const auto plan_owners = result.program_inventory.kernel_descriptors(plan.owner_kernel_ids);
  ASSERT_TRUE(plan_owners);
  EXPECT_EQ(atomic_patch->owner_descriptor_file_offsets, *plan_owners);
  EXPECT_EQ(std::count_if(result.patches.begin(), result.patches.end(),
                          [](const PatchInfo &patch) {
                            return patch.kind == PatchKind::KernelEntryOwnerEpochPrologue;
                          }),
            2);
}

TEST(ConSan, SharedHelperAtomicSpillUsesOneLayoutForEveryOwner) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.helper_has_ordered_atomic = true;
  fixture.helper_atomic_has_lds_producer = true;
  fixture.first_private_bytes = 0;
  fixture.second_private_bytes = 20;
  const std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object(fixture);
  ASSERT_FALSE(bytes.empty());
  TestOptions options = test_options();
  options.track_atomics = true;
  options.max_patches = 8u;
  options.test_force_vgpr_spill = true;
  options.set_owner_epoch_vgprs(10, 11);
  options.report_buffer_address = 0x123456780000ull;
  options.report_buffer_size = direct_report_bytes(8);

  const auto result = test_lower_consan(bytes, options);

  ASSERT_TRUE(patch_succeeded(result));
  ASSERT_TRUE(result.modified()) << testing::PrintToString(result.warnings);
  ASSERT_EQ(result.resource_plans.size(), 2u);
  const auto plan_it = std::ranges::find_if(result.resource_plans, [](const auto &item) {
    return item.site_kind == ResourceSiteKind::Atomic;
  });
  ASSERT_NE(plan_it, result.resource_plans.end());
  const CandidateResourcePlan &plan = *plan_it;
  EXPECT_EQ(plan.site_kind, ResourceSiteKind::Atomic);
  EXPECT_EQ(plan.source, RegisterAllocationSource::SpillRequired);
  EXPECT_EQ(plan.original_private_segment_size, 20u);
  ASSERT_EQ(plan.owner_kernel_ids.size(), 2u);
  const auto patch_it = std::ranges::find_if(result.patches, [](const PatchInfo &patch) {
    return patch.kind == PatchKind::TrampolineSyncMetadata;
  });
  ASSERT_NE(patch_it, result.patches.end());
  const PatchInfo &patch = *patch_it;
  const auto plan_owners = result.program_inventory.kernel_descriptors(plan.owner_kernel_ids);
  ASSERT_TRUE(plan_owners);
  EXPECT_EQ(patch.kind, PatchKind::TrampolineSyncMetadata);
  EXPECT_EQ(patch.spilled_vgpr_count, 11u);
  EXPECT_EQ(patch.required_private_segment_size, 76u);
  EXPECT_EQ(patch.owner_descriptor_file_offsets, *plan_owners);
  uint32_t shared_private_size = 0u;
  for (const PatchInfo &item : result.patches) {
    if (item.owner_descriptor_file_offsets == *plan_owners) {
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

TEST(ConSan, SharedHelperPatchNamesEveryOwnerAndLeavesUnrelatedDescriptorUnchanged) {
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

  TestOptions options = test_options();
  options.report_buffer_address = 0x123456780000ull;
  options.report_buffer_size = direct_report_bytes(8);
  const auto result = test_lower_consan(bytes, options);

  ASSERT_TRUE(patch_succeeded(result));
  ASSERT_TRUE(result.modified());
  ASSERT_EQ(non_entry_prologue_patch_count(result), 2u);
  const PatchInfo &patch = result.patches.front();
  EXPECT_EQ(patch.kind, PatchKind::TrampolineWatchpointStore);
  EXPECT_EQ(patch.anchor_offset, 20u);
  ASSERT_EQ(patch.owner_descriptor_file_offsets.size(), 2u);
  const auto plan_owners =
      result.program_inventory.kernel_descriptors(result.resource_plans.front().owner_kernel_ids);
  ASSERT_TRUE(plan_owners);
  EXPECT_EQ(patch.owner_descriptor_file_offsets, *plan_owners);

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

TEST(ConSan, SharedHelperPlanGrowsEveryOwnerForOneFreshWindow) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.first_vgpr_granulated = 0;
  fixture.second_vgpr_granulated = 0;
  fixture.helper_keeps_v1_v3_live = true;
  const std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object(fixture);
  TestOptions options = test_options();
  options.report_buffer_address = 0x123456780000ull;
  options.report_buffer_size = direct_report_bytes(8);

  const auto result = test_lower_consan(bytes, options);

  ASSERT_TRUE(patch_succeeded(result));
  ASSERT_TRUE(result.modified());
  ASSERT_EQ(result.resource_plans.size(), 1u);
  const CandidateResourcePlan &plan = result.resource_plans.front();
  EXPECT_EQ(plan.source, RegisterAllocationSource::DescriptorGrowth);
  EXPECT_EQ(plan.scratch_vgpr, 4);
  EXPECT_EQ(plan.required_vgpr_count, 12u);
  ASSERT_EQ(plan.owner_kernel_ids.size(), 2u);

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

TEST(ConSan, SharedHelperSpillUsesOneLayoutAndGrowsEveryOwner) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.first_private_bytes = 0;
  fixture.second_private_bytes = 20;
  const std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object(fixture);
  TestOptions options = test_options();
  options.test_force_vgpr_spill = true;
  options.report_buffer_address = 0x123456780000ull;
  options.report_buffer_size = direct_report_bytes(8);

  const auto result = test_lower_consan(bytes, options);

  ASSERT_TRUE(patch_succeeded(result));
  ASSERT_TRUE(result.modified());
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().source, RegisterAllocationSource::SpillRequired);
  EXPECT_EQ(result.resource_plans.front().original_private_segment_size, 20u);
  ASSERT_EQ(non_entry_prologue_patch_count(result), 2u);
  const PatchInfo &patch = result.patches.front();
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

TEST(ConSan, IndirectSharedHelperSpillUsesEveryRecoveredOwner) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.use_indirect_calls = true;
  const std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object(fixture);
  TestOptions options = test_options();
  options.test_force_vgpr_spill = true;
  options.report_buffer_address = 0x123456780000ull;
  options.report_buffer_size = direct_report_bytes(8);

  const auto result = test_lower_consan(bytes, options);

  ASSERT_TRUE(patch_succeeded(result));
  EXPECT_TRUE(result.modified());
  ASSERT_EQ(result.resource_plans.size(), 1u);
  const CandidateResourcePlan &plan = result.resource_plans.front();
  EXPECT_EQ(plan.source, RegisterAllocationSource::SpillRequired);
  EXPECT_EQ(plan.reason, RegisterPlanReason::None);
  EXPECT_EQ(plan.owner_kernel_ids.size(), 2u);
  ASSERT_EQ(non_entry_prologue_patch_count(result), 2u);
  const auto plan_owners = result.program_inventory.kernel_descriptors(plan.owner_kernel_ids);
  ASSERT_TRUE(plan_owners);
  EXPECT_EQ(result.patches.front().owner_descriptor_file_offsets, *plan_owners);
}

TEST(ConSan, ScopedSpillPlanningExcludesUnselectedFullVgprCandidate) {
  TwoKernelSharedFixtureOptions fixture;
  // Keep the selected owners large enough for ConSan's temporary window.
  // The unrelated full-VGPR candidate must remain excluded from scoped planning.
  fixture.first_vgpr_granulated = 1;
  fixture.second_vgpr_granulated = 1;
  fixture.first_private_bytes = 20;
  fixture.unrelated_has_lds = true;
  const std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object(fixture);
  TestOptions options = test_options();
  options.test_force_vgpr_spill = true;
  options.test_kernel_name_filter = "shared_lds_helper";
  options.report_buffer_address = 0x123456780000ull;
  options.report_buffer_size = direct_report_bytes(8);

  const auto result = test_lower_consan(bytes, options);

  ASSERT_TRUE(patch_succeeded(result));
  EXPECT_EQ(result.outcome, TransformOutcome::ModifiedValid);
  ASSERT_EQ(non_entry_prologue_patch_count(result), 2u);
  const uint32_t selected_private_size = result.patches.front().required_private_segment_size;
  EXPECT_GT(selected_private_size, fixture.first_private_bytes);
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
      EXPECT_EQ(descriptor.private_segment_fixed_size, selected_private_size);
    } else if (kernel.name == "unrelated_kernel") {
      EXPECT_EQ(descriptor.private_segment_fixed_size, 0u);
    }
  }
}

TEST(ConSan, SharedHelperRejectsAssignmentLiveInAnyOwnerScope) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.first_continuation_uses_v1 = true;
  const std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object(fixture);
  TestOptions options = test_options();
  options.scratch_vgpr = 1;
  options.report_buffer_address = 0x123456780000ull;
  options.report_buffer_size = direct_report_bytes(8);

  const auto result = test_lower_consan(bytes, options);

  EXPECT_TRUE(result.errors.empty());
  EXPECT_FALSE(result.modified());
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().source, RegisterAllocationSource::Unsupported);
  EXPECT_EQ(result.resource_plans.front().reason, RegisterPlanReason::ExplicitLive);
  EXPECT_EQ(result.resource_plans.front().owner_kernel_ids.size(), 2u);
}

TEST(ConSan, SharedPrivateOwnerRejectsIncompatibleWaveSizes) {
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
  TestOptions options = test_options();
  options.test_force_private_epoch = true;
  options.report_buffer_address = 0x100000000ull;
  options.report_buffer_size = direct_report_bytes(8);

  const auto result = test_lower_consan(bytes, options);

  ASSERT_TRUE(patch_succeeded(result));
  EXPECT_FALSE(result.modified());
  EXPECT_TRUE(result.patches.empty());
  EXPECT_TRUE(std::ranges::any_of(result.warnings, [](const std::string &warning) {
    return warning.find("incompatible descriptor ABI inputs: owner shift") != std::string::npos;
  }));
  ASSERT_EQ(result.resource_plans.size(), 1u);
  EXPECT_EQ(result.resource_plans.front().owner_kernel_ids.size(), 2u);
  EXPECT_NE(result.resource_plans.front().source, RegisterAllocationSource::Unsupported);
  EXPECT_EQ(result.outcome, TransformOutcome::Unchanged);
}

TEST(ConSan, InventorySkipsUnknownFlatSites) {
  const std::vector<uint8_t> bytes = make_rdna4_flat_memory_code_object();
  TestOptions options = test_options();

  const auto result = test_lower_consan(bytes, options);

  ASSERT_TRUE(patch_succeeded(result));
  ASSERT_EQ(result.program_inventory.kernels().size(), 1u);
  ASSERT_EQ(result.program_inventory.access_sites().size(), 2u);
  EXPECT_TRUE(test_admitted_accesses(result).empty());

  EXPECT_EQ(std::ranges::count(result.observation_plan().site_decisions,
                               AccessPolicyReason::FlatProvenancePolicyExcluded,
                               &SiteDecision::reason),
            2u);
}

TEST(ConSan, DispatchIdPreloadPlanPreservesShiftedGuestSgprs) {
  EXPECT_EQ(amdhsa_dispatch_id_prefix_sgpr_count(
                /*private_segment_buffer=*/true, /*dispatch_ptr=*/true,
                /*queue_ptr=*/true, /*kernarg_segment_ptr=*/true),
            10u);
  const auto plan = plan_dispatch_id_preload(
      /*original_user_sgpr_count=*/12, /*system_sgpr_count=*/3,
      /*dispatch_id_prefix_sgpr_count=*/10, /*dispatch_id_already_enabled=*/false);
  ASSERT_TRUE(plan.supported());
  EXPECT_TRUE(plan.descriptor_change_required());
  EXPECT_EQ(plan.support, DispatchIdPreloadSupport::SupportedInsert);
  EXPECT_EQ(plan.dispatch_id_sgpr, 10u);
  EXPECT_EQ(plan.expanded_user_sgpr_count, 14u);
  EXPECT_EQ(plan.first_shifted_guest_sgpr, 10u);
  EXPECT_EQ(plan.shifted_guest_sgpr_count, 5u);
  EXPECT_EQ(plan.required_sgpr_count, 17u);

  std::array<uint32_t, 17> expanded{};
  for (uint16_t guest_sgpr = 0; guest_sgpr < 15; ++guest_sgpr) {
    const auto source = dispatch_id_restore_source(plan, guest_sgpr);
    expanded[source.value_or(guest_sgpr)] = 0x1000u + guest_sgpr;
  }
  expanded[plan.dispatch_id_sgpr] = 0xD15A7C01u;
  expanded[plan.dispatch_id_sgpr + 1u] = 0xD15A7C02u;
  const std::array<uint32_t, 2> captured_dispatch = {expanded[plan.dispatch_id_sgpr],
                                                     expanded[plan.dispatch_id_sgpr + 1u]};
  for (uint16_t destination = plan.first_shifted_guest_sgpr;
       destination < plan.first_shifted_guest_sgpr + plan.shifted_guest_sgpr_count; ++destination) {
    const auto source = dispatch_id_restore_source(plan, destination);
    ASSERT_TRUE(source.has_value());
    expanded[destination] = expanded[*source];
  }
  EXPECT_EQ(captured_dispatch[0], 0xD15A7C01u);
  EXPECT_EQ(captured_dispatch[1], 0xD15A7C02u);
  for (uint16_t guest_sgpr = 0; guest_sgpr < 15; ++guest_sgpr)
    EXPECT_EQ(expanded[guest_sgpr], 0x1000u + guest_sgpr);
}

TEST(ConSan, DispatchIdPreloadPlanRejectsTruncationAndInvalidLayouts) {
  const auto existing = plan_dispatch_id_preload(
      /*original_user_sgpr_count=*/14, /*system_sgpr_count=*/4,
      /*dispatch_id_prefix_sgpr_count=*/8, /*dispatch_id_already_enabled=*/true);
  ASSERT_TRUE(existing.supported());
  EXPECT_EQ(existing.support, DispatchIdPreloadSupport::SupportedAlreadyEnabled);
  EXPECT_FALSE(existing.descriptor_change_required());
  EXPECT_FALSE(dispatch_id_restore_source(existing, 8).has_value());

  EXPECT_EQ(plan_dispatch_id_preload(15, 1, 10, false).support,
            DispatchIdPreloadSupport::UserSgprInitializationLimit);
  EXPECT_EQ(plan_dispatch_id_preload(8, 1, 9, false).support,
            DispatchIdPreloadSupport::InvalidDispatchPosition);
  EXPECT_EQ(plan_dispatch_id_preload(14, 91, 10, false).support,
            DispatchIdPreloadSupport::SgprAllocationLimit);
  EXPECT_EQ(plan_dispatch_id_preload(17, 0, 10, true).support,
            DispatchIdPreloadSupport::UserSgprInitializationLimit);
}

TEST(ConSan, DispatchIdPreloadPlanSupportsGfx1250UserSgprLimit) {
  const auto plan = plan_dispatch_id_preload(
      /*original_user_sgpr_count=*/29, /*system_sgpr_count=*/1,
      /*dispatch_id_prefix_sgpr_count=*/2, /*dispatch_id_already_enabled=*/false,
      /*sgpr_limit=*/106, /*user_sgpr_initialization_limit=*/32);
  ASSERT_TRUE(plan.supported());
  EXPECT_EQ(plan.support, DispatchIdPreloadSupport::SupportedInsert);
  EXPECT_EQ(plan.expanded_user_sgpr_count, 31u);
  EXPECT_EQ(plan.shifted_guest_sgpr_count, 28u);
  EXPECT_EQ(plan.required_sgpr_count, 32u);
}

TEST(ConSan, DispatchPreloadDescriptorPermutationsUseExactAmdhsaPrefix) {
  for (uint32_t mask = 0; mask < 16u; ++mask) {
    std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
    const uint16_t prefix =
        amdhsa_dispatch_id_prefix_sgpr_count(mask & 1u, mask & 2u, mask & 4u, mask & 8u);
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

    TestOptions options = test_options();
    options.report_buffer_address = 0x100000000ull;
    options.report_buffer_size = direct_report_bytes(8);
    const TransformArtifacts result = test_lower_consan(bytes, options);

    ASSERT_TRUE(result.errors.empty())
        << "mask=" << mask << " " << (result.errors.empty() ? "" : result.errors.front());
    ASSERT_TRUE(result.modified()) << "mask=" << mask;
    const auto prologue = std::ranges::find_if(result.patches, [](const PatchInfo &patch) {
      return patch.kind == PatchKind::KernelEntryOwnerEpochPrologue;
    });
    ASSERT_NE(prologue, result.patches.end());
    ASSERT_TRUE(prologue->dispatch_id_prologue);
    ASSERT_TRUE(prologue->dispatch_id_prologue->capture.sgpr());
    const uint16_t queue_insertion = (mask & 4u) != 0u ? 0u : 2u;
    EXPECT_EQ(prologue->dispatch_id_prologue->preload.dispatch_id_sgpr, prefix + queue_insertion);
    EXPECT_EQ(prologue->dispatch_id_prologue->preload.original_user_sgpr_count, prefix + 2u);
    EXPECT_EQ(prologue->dispatch_id_prologue->preload.expanded_user_sgpr_count,
              prefix + 4u + queue_insertion);
    EXPECT_EQ(prologue->dispatch_id_prologue->preload.system_sgpr_count, 2u);
    EXPECT_TRUE(prologue->dispatch_id_prologue->preload.descriptor_change_required());

    AmdGpuCodeObject patched(result.replacement.data(), result.replacement.size());
    ASSERT_TRUE(patched.is_valid());
    KD descriptor{};
    std::memcpy(&descriptor,
                result.replacement.data() + patched.kernels().front().descriptor_file_offset,
                sizeof(descriptor));
    EXPECT_EQ(AMDHSA_BITS_GET(descriptor.kernel_code_properties,
                              kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID),
              1u);
    EXPECT_EQ(AMDHSA_BITS_GET(descriptor.kernel_code_properties,
                              kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR),
              1u);
    EXPECT_EQ(AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT),
              prefix + 4u + queue_insertion);
  }
}

TEST(ConSan, DispatchPrologueCapturesBeforeAscendingRestoreAtBothKernargEntries) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  std::vector<uint32_t> text_words(80u, build_s_nop(0, kArch));
  const auto store = build_cdna4_ds_store_b32(/*vaddr=*/0u, /*vdata=*/0u,
                                              /*byte_offset=*/0u, kArch);
  ASSERT_TRUE(store);
  std::ranges::copy(*store, text_words.begin());
  text_words.back() = build_s_endpgm(kArch);
  std::vector<uint8_t> bytes = make_cdna4_lds_code_object(text_words);
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

  TestOptions options = test_options();
  options.report_buffer_address = 0x100000000ull;
  options.report_buffer_size = direct_report_bytes(8);
  const TransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_TRUE(patch_succeeded(result));
  ASSERT_TRUE(result.modified());
  std::vector<const PatchInfo *> prologues;
  for (const PatchInfo &patch : result.patches) {
    if (patch.kind == PatchKind::KernelEntryOwnerEpochPrologue)
      prologues.push_back(&patch);
  }
  ASSERT_EQ(prologues.size(), 1u);
  const PatchInfo &prologue = *prologues.front();
  ASSERT_TRUE(prologue.dispatch_id_prologue);
  ASSERT_TRUE(prologue.dispatch_id_prologue->capture.sgpr());
  EXPECT_EQ(prologue.dispatch_id_prologue->preload.dispatch_id_sgpr, 10u);
  EXPECT_EQ(prologue.dispatch_id_prologue->preload.original_user_sgpr_count, 14u);
  EXPECT_EQ(prologue.dispatch_id_prologue->preload.expanded_user_sgpr_count, 16u);
  EXPECT_EQ(prologue.dispatch_id_prologue->preload.system_sgpr_count, 4u);
  ASSERT_TRUE(result.text_relocation);

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
    EXPECT_FALSE(prologue.entry_scalar_backup);
    const auto dependency_delay = instrumentation::build_salu_dependency_delay(kArch);
    ASSERT_TRUE(dependency_delay);
    const auto expect_write = [&](uint32_t expected) {
      uint32_t word = 0;
      std::memcpy(&word, text + cursor, sizeof(word));
      EXPECT_EQ(word, expected);
      cursor += sizeof(word);
      std::memcpy(&word, text + cursor, sizeof(word));
      EXPECT_EQ(word, *dependency_delay);
      cursor += sizeof(word);
    };
    // ConSan preserves the shifted launch tuple before restoring guest SGPRs.
    const auto workgroup = test_exact_workgroup_vgprs(result);
    ASSERT_TRUE(workgroup.complete());
    for (size_t coordinate = 0; coordinate < 3u; ++coordinate) {
      uint32_t word = 0;
      std::memcpy(&word, text + cursor, sizeof(word));
      EXPECT_EQ(word,
                build_v_mov_b32_e32(*workgroup.values()[coordinate], 16u + coordinate, kArch));
      cursor += sizeof(word);
    }
    const uint16_t persistent = *prologue.dispatch_id_prologue->capture.sgpr();
    expect_write(build_s_mov_b32(persistent, 10u, kArch));
    expect_write(build_s_mov_b32(static_cast<uint16_t>(persistent + 1u), 11u, kArch));
    ASSERT_TRUE(prologue.dispatch_id_prologue->preload.identity_salt_sgpr);
    const auto mix = instrumentation::build_s_xor_b64(
        persistent, persistent, *prologue.dispatch_id_prologue->preload.identity_salt_sgpr, kArch);
    ASSERT_TRUE(mix);
    expect_write(*mix);
    expect_write(build_s_add_u32(persistent, persistent, scalar_positive_inline_u32(1), kArch));
    expect_write(build_s_addc_u32(static_cast<uint16_t>(persistent + 1u),
                                  static_cast<uint16_t>(persistent + 1u),
                                  scalar_positive_inline_u32(0), kArch));
    const auto &preload = prologue.dispatch_id_prologue->preload;
    for (uint16_t i = 0u; i < preload.guest_restore_count; ++i)
      expect_write(build_s_mov_b32(preload.guest_restore_destinations[i],
                                   preload.guest_restore_sources[i], kArch));
  };
  ASSERT_TRUE(prologue.dispatch_id_primary_prologue_offset);
  ASSERT_TRUE(prologue.dispatch_id_secondary_prologue_offset);
  verify_entry(*prologue.dispatch_id_primary_prologue_offset);
  verify_entry(*prologue.dispatch_id_secondary_prologue_offset);
}

TEST(ConSan, DispatchAlreadyEnabledInsertsQueueAndRestoresGuestAbi) {
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
  TestOptions options = test_options();
  options.report_buffer_address = 0x100000000ull;
  options.report_buffer_size = direct_report_bytes(8);
  const TransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_TRUE(patch_succeeded(result));
  ASSERT_TRUE(result.modified());
  const auto prologue = std::ranges::find_if(result.patches, [](const PatchInfo &patch) {
    return patch.dispatch_id_prologue && patch.dispatch_id_prologue->capture.sgpr();
  });
  ASSERT_NE(prologue, result.patches.end());
  EXPECT_EQ(prologue->dispatch_id_prologue->preload.dispatch_id_sgpr, 4u);
  EXPECT_EQ(prologue->dispatch_id_prologue->preload.original_user_sgpr_count, 4u);
  EXPECT_EQ(prologue->dispatch_id_prologue->preload.expanded_user_sgpr_count, 6u);
  EXPECT_TRUE(prologue->dispatch_id_prologue->preload.descriptor_change_required());
  EXPECT_FALSE(prologue->dispatch_id_prologue->preload.queue_ptr_was_enabled);
  EXPECT_TRUE(prologue->dispatch_id_prologue->preload.dispatch_id_was_enabled);
  EXPECT_EQ(prologue->dispatch_id_prologue->preload.identity_salt_sgpr, 0u);
  EXPECT_EQ(prologue->dispatch_id_prologue->preload.guest_restore_count, 4u);
  for (uint16_t index = 0u; index < 4u; ++index) {
    EXPECT_EQ(prologue->dispatch_id_prologue->preload.guest_restore_destinations[index], index);
    EXPECT_EQ(prologue->dispatch_id_prologue->preload.guest_restore_sources[index], index + 2u);
  }

  AmdGpuCodeObject patched(result.replacement.data(), result.replacement.size());
  ASSERT_TRUE(patched.is_valid());
  KD descriptor{};
  std::memcpy(&descriptor,
              result.replacement.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  EXPECT_EQ(AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT),
            6u);
  EXPECT_EQ(AMDHSA_BITS_GET(descriptor.kernel_code_properties,
                            kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR),
            1u);
  const std::vector<uint32_t> prologue_words =
      text_words_at_offset(patched, prologue->trampoline_offset, prologue->trampoline_size);
  ASSERT_TRUE(prologue->vgpr_state);
  const auto &owner_epoch = prologue->vgpr_state->owner_epoch;
  const auto owner_init = build_v_lshrrev_b32_e32(owner_epoch.owner, scalar_positive_inline_u32(6),
                                                  0, ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(owner_init);
  EXPECT_NE(std::ranges::find(prologue_words, *owner_init), prologue_words.end());
}

TEST(ConSan, AlreadyEnabledQueueAndDispatchPreloadsReuseExactAbiPositions) {
  std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                    kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR, 1u);
    AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                    kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 1u);
    AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                    kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID, 1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 6u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X,
                    1u);
  });
  TestOptions options = test_options();
  options.report_buffer_address = 0x100000000ull;
  options.report_buffer_size = direct_report_bytes(8);
  const TransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_TRUE(patch_succeeded(result));
  ASSERT_TRUE(result.modified());
  const auto prologue = std::ranges::find_if(result.patches, [](const PatchInfo &patch) {
    return patch.dispatch_id_prologue && patch.dispatch_id_prologue->capture.sgpr();
  });
  ASSERT_NE(prologue, result.patches.end());
  const auto &preload = prologue->dispatch_id_prologue->preload;
  EXPECT_EQ(preload.support, DispatchIdPreloadSupport::SupportedAlreadyEnabled);
  EXPECT_EQ(preload.identity_salt_sgpr, 0u);
  EXPECT_EQ(preload.dispatch_id_sgpr, 4u);
  EXPECT_EQ(preload.original_user_sgpr_count, 6u);
  EXPECT_EQ(preload.expanded_user_sgpr_count, 6u);
  EXPECT_EQ(preload.guest_restore_count, 0u);
  EXPECT_FALSE(preload.descriptor_change_required());
}

TEST(ConSan, SharedHelperDispatchCaptureUsesPerKernelLayoutsAndOnePersistentPair) {
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
  TestOptions options = test_options();
  options.report_buffer_address = 0x100000000ull;
  options.report_buffer_size = direct_report_bytes(8);
  const TransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_TRUE(patch_succeeded(result));
  ASSERT_TRUE(result.modified()) << testing::PrintToString(result.warnings);
  std::vector<const PatchInfo *> prologues;
  for (const PatchInfo &patch : result.patches) {
    if (patch.dispatch_id_prologue && patch.dispatch_id_prologue->capture.sgpr())
      prologues.push_back(&patch);
  }
  ASSERT_EQ(prologues.size(), 2u);
  ASSERT_TRUE(test_dispatch_id_sgpr(result));
  EXPECT_EQ(prologues[0]->dispatch_id_prologue->capture.sgpr(), test_dispatch_id_sgpr(result));
  EXPECT_EQ(prologues[1]->dispatch_id_prologue->capture.sgpr(), test_dispatch_id_sgpr(result));
  std::array<uint16_t, 2> sources = {prologues[0]->dispatch_id_prologue->preload.dispatch_id_sgpr,
                                     prologues[1]->dispatch_id_prologue->preload.dispatch_id_sgpr};
  std::ranges::sort(sources);
  EXPECT_EQ(sources, (std::array<uint16_t, 2>{4u, 6u}));

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
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 16u);
    AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                    kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 0u);
    AMDHSA_BITS_SET(descriptor.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_LENGTH, 5u);
  });
  const TransformArtifacts rejected = test_lower_consan(rejected_bytes, options);
  EXPECT_EQ(rejected.outcome, TransformOutcome::Unsupported);
  EXPECT_FALSE(rejected.modified());
  EXPECT_TRUE(rejected.replacement.empty());
  EXPECT_TRUE(rejected.patches.empty());
}

TEST(ConSan, DispatchPreloadUnsupportedLayoutsRollbackTransactionally) {
  const auto run = [](const auto &mutator, std::optional<uint16_t> explicit_pair = std::nullopt) {
    std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
    mutate_first_kernel_descriptor(bytes, mutator);
    TestOptions options = test_options();
    options.dispatch_sgpr.set(explicit_pair);
    options.report_buffer_address = 0x100000000ull;
    options.report_buffer_size = direct_report_bytes(8);
    return test_lower_consan(bytes, options);
  };

  const TransformArtifacts user_limit = run([](KD &descriptor) {
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
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 16u);
    AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                    kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 0u);
    AMDHSA_BITS_SET(descriptor.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_LENGTH, 4u);
  });
  EXPECT_EQ(user_limit.outcome, TransformOutcome::Unsupported);
  EXPECT_FALSE(user_limit.modified());
  EXPECT_TRUE(user_limit.replacement.empty());
  EXPECT_TRUE(user_limit.patches.empty());

  const TransformArtifacts malformed_prefix = run([](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                    kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER, 1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 2u);
  });
  EXPECT_EQ(malformed_prefix.outcome, TransformOutcome::Unsupported);
  EXPECT_FALSE(malformed_prefix.modified());
  EXPECT_TRUE(malformed_prefix.replacement.empty());
  EXPECT_TRUE(malformed_prefix.patches.empty());

  const TransformArtifacts invalid_pair = run([](KD &) {}, 105u);
  EXPECT_EQ(invalid_pair.outcome, TransformOutcome::Unsupported);
  EXPECT_FALSE(invalid_pair.modified());
  EXPECT_TRUE(invalid_pair.replacement.empty());
  EXPECT_TRUE(invalid_pair.patches.empty());
}

TEST(ConSan, FinalValidationPinsDispatchDescriptorAndCaptureSequence) {
  std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  mutate_first_kernel_descriptor(bytes, [](KD &descriptor) {
    AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                    kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 1u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 2u);
    AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X,
                    1u);
  });
  TestOptions options = test_options();
  options.report_buffer_address = 0x100000000ull;
  options.report_buffer_size = direct_report_bytes(8);
  const TransformArtifacts valid = test_lower_consan(bytes, options);
  ASSERT_TRUE(valid.errors.empty()) << (valid.errors.empty() ? "" : valid.errors.front());
  ASSERT_TRUE(valid.modified());
  EXPECT_TRUE(validate_modified_elf(bytes, valid).empty());

  TransformArtifacts descriptor_corruption = valid;
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
  EXPECT_FALSE(validate_modified_elf(bytes, descriptor_corruption).empty());

  TransformArtifacts queue_descriptor_corruption = valid;
  std::memcpy(&descriptor, queue_descriptor_corruption.replacement.data() + descriptor_offset,
              sizeof(descriptor));
  AMDHSA_BITS_SET(descriptor.kernel_code_properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR,
                  0u);
  std::memcpy(queue_descriptor_corruption.replacement.data() + descriptor_offset, &descriptor,
              sizeof(descriptor));
  EXPECT_FALSE(validate_modified_elf(bytes, queue_descriptor_corruption).empty());

  TransformArtifacts capture_corruption = valid;
  const auto prologue =
      std::ranges::find_if(capture_corruption.patches, [](const PatchInfo &patch) {
        return patch.dispatch_id_prologue && patch.dispatch_id_prologue->capture.sgpr();
      });
  ASSERT_NE(prologue, capture_corruption.patches.end());
  const size_t capture_file_offset =
      capture_corruption.program_inventory.text_sections().front().file_offset +
      prologue->trampoline_offset;
  const uint32_t wrong_capture = build_s_mov_b32(
      *prologue->dispatch_id_prologue->capture.sgpr(),
      static_cast<uint16_t>(prologue->dispatch_id_prologue->preload.dispatch_id_sgpr + 1u),
      ROCJITSU_CODE_ARCH_RDNA4);
  std::memcpy(capture_corruption.replacement.data() + capture_file_offset, &wrong_capture,
              sizeof(wrong_capture));
  EXPECT_FALSE(validate_modified_elf(bytes, capture_corruption).empty());

  TransformArtifacts mix_corruption = valid;
  const auto &effect = *prologue->dispatch_id_prologue;
  ASSERT_TRUE(effect.preload.identity_salt_sgpr);
  const auto expected_mix = instrumentation::build_s_xor_b64(
      *effect.capture.sgpr(), *effect.capture.sgpr(), *effect.preload.identity_salt_sgpr,
      ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(expected_mix);
  AmdGpuCodeObject mix_object(mix_corruption.replacement.data(), mix_corruption.replacement.size());
  ASSERT_TRUE(mix_object.is_valid());
  std::vector<uint32_t> mix_words =
      text_words_at_offset(mix_object, prologue->trampoline_offset, prologue->trampoline_size);
  const auto mix = std::ranges::find(mix_words, *expected_mix);
  ASSERT_NE(mix, mix_words.end());
  const size_t mix_word_index = static_cast<size_t>(mix - mix_words.begin());
  const size_t mix_file_offset = mix_object.text_sections().front()->sectionOffset() +
                                 prologue->trampoline_offset + mix_word_index * sizeof(uint32_t);
  const uint32_t missing_mix = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
  std::memcpy(mix_corruption.replacement.data() + mix_file_offset, &missing_mix,
              sizeof(missing_mix));
  EXPECT_FALSE(validate_modified_elf(bytes, mix_corruption).empty());

  TransformArtifacts restore_metadata_corruption = valid;
  auto corrupted_prologue =
      std::ranges::find_if(restore_metadata_corruption.patches, [](const PatchInfo &patch) {
        return patch.dispatch_id_prologue.has_value();
      });
  ASSERT_NE(corrupted_prologue, restore_metadata_corruption.patches.end());
  auto &corrupted_preload = corrupted_prologue->dispatch_id_prologue->preload;
  ASSERT_GT(corrupted_preload.guest_restore_count, 0u);
  // The explicit source map is exclusively a user-preload repair. The old
  // combined user+system bound incorrectly admitted this first system SGPR.
  corrupted_preload.guest_restore_destinations[0] = corrupted_preload.original_user_sgpr_count;
  const std::vector<std::string> restore_errors =
      validate_modified_elf(bytes, restore_metadata_corruption);
  EXPECT_TRUE(std::ranges::any_of(restore_errors, [](const std::string &error) {
    return error.find("dispatch-ID restore bounds") != std::string::npos;
  })) << testing::PrintToString(restore_errors);
}

TEST(ConSan, WarnsWhenReportBufferIsSmallerThanHeader) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  TestOptions options = test_options();
  options.report_buffer_address = 0x1000;
  options.report_buffer_size = sizeof(ReportHeader) - 1;

  const auto result = test_lower_consan(bytes, options);

  ASSERT_TRUE(patch_succeeded(result));
  bool saw_small_buffer_warning = false;
  for (const std::string &warning : result.warnings)
    saw_small_buffer_warning |=
        warning.find("smaller than the report ABI header") != std::string::npos;
  EXPECT_TRUE(saw_small_buffer_warning);
}

TEST(ConSan, RejectsReportBufferLargerThanDynamicRecordOffsetWindow) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  TestOptions options = test_options();
  options.report_buffer_address = 0x1000;
  options.report_buffer_size = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1u;

  const auto result = test_lower_consan(bytes, options);

  EXPECT_FALSE(patch_succeeded(result));
  EXPECT_TRUE(std::ranges::any_of(result.errors, [](const std::string &error) {
    return error.find("32-bit dynamic record-offset window") != std::string::npos;
  }));
}

TEST(ConSan, InventorySkipsUnsupportedNativeLdsSites) {
  const std::vector<uint8_t> bytes = make_rdna4_unsupported_lds_code_object();
  TestOptions options = test_options();

  const auto result = test_lower_consan(bytes, options);

  ASSERT_TRUE(patch_succeeded(result));
  ASSERT_EQ(test_admitted_accesses(result).size(), 2u);
  EXPECT_EQ(test_admitted_accesses(result)[0].mnemonic_view(), "ds_store_b32");
  EXPECT_EQ(test_admitted_accesses(result)[1].mnemonic_view(), "ds_load_b32");
  EXPECT_EQ(std::ranges::count(result.observation_plan().site_decisions,
                               AccessPolicyReason::OperationKindExcluded, &SiteDecision::reason),
            1u);
}

TEST(ConSan, LoweringCandidatesAreExactlyTheAdmittedAccessIntents) {
  const std::vector<uint8_t> bytes = make_rdna4_unsupported_lds_code_object();
  const TransformArtifacts result = test_lower_consan(bytes, test_options());
  ASSERT_TRUE(result.errors.empty()) << testing::PrintToString(result.errors);

  std::set<uint64_t> intended_offsets;
  for (const ProbeIntent &intent : result.observation_plan().probe_intents) {
    if (intent.kind == ProbeIntentKind::Access) {
      intended_offsets.insert(intent.physical_site.original_text_offset);
    }
  }
  std::set<uint64_t> candidate_offsets;
  for (const ProgramSite &candidate : test_admitted_accesses(result)) {
    candidate_offsets.insert(candidate.physical_id.original_text_offset);
    const auto site = std::ranges::find_if(
        result.program_inventory.access_sites(), [&](const ProgramSite &access) {
          return access.physical_id.original_text_offset ==
                     candidate.physical_id.original_text_offset &&
                 access.container == candidate.container;
        });
    ASSERT_NE(site, result.program_inventory.access_sites().end());
    EXPECT_EQ(static_cast<const ProgramSite &>(candidate), *site);
  }

  EXPECT_EQ(candidate_offsets, intended_offsets);
  EXPECT_EQ(test_admitted_accesses(result).size(), intended_offsets.size());
}

TEST(ConSan, Gfx1201AdmitsNativeB96Accesses) {
  constexpr auto store =
      rdna4::build_vds(rdna4::kDsStoreB96Vds, {.offset0 = 12, .addr = 0, .data0 = 1});
  constexpr auto load =
      rdna4::build_vds(rdna4::kDsLoadB96Vds, {.offset0 = 12, .addr = 0, .vdst = 4});
  constexpr auto aliasing_load =
      rdna4::build_vds(rdna4::kDsLoadB96Vds, {.offset0 = 12, .addr = 0, .vdst = 0});
  constexpr std::array<NativeB96Access, 3> accesses = {
      NativeB96Access{store, "ds_store_b96"},
      NativeB96Access{load, "ds_load_b96"},
      NativeB96Access{aliasing_load, "ds_load_b96"},
  };

  expect_admits_native_b96_accesses(ROCJITSU_CODE_ARCH_RDNA4, accesses, [](const auto &text_words) {
    return make_rdna4_lds_code_object(text_words, "native_b96_access");
  });
}

TEST(ConSan, Gfx1100AdmitsNativeB96Accesses) {
  constexpr auto store =
      rdna4::build_vds(rdna4::kDsStoreB96Vds, {.offset0 = 12, .addr = 0, .data0 = 1});
  constexpr auto load =
      rdna4::build_vds(rdna4::kDsLoadB96Vds, {.offset0 = 12, .addr = 0, .vdst = 4});
  constexpr auto aliasing_load =
      rdna4::build_vds(rdna4::kDsLoadB96Vds, {.offset0 = 12, .addr = 0, .vdst = 0});
  constexpr std::array<NativeB96Access, 3> accesses = {
      NativeB96Access{store, "ds_store_b96"},
      NativeB96Access{load, "ds_load_b96"},
      NativeB96Access{aliasing_load, "ds_load_b96"},
  };

  expect_admits_native_b96_accesses(ROCJITSU_CODE_ARCH_RDNA3, accesses, [](const auto &text_words) {
    return make_rdna3_lds_code_object(text_words, "gfx1100_native_b96_access",
                                      kRdna4Wave64AllVgprsGranulated, /*wave32=*/true);
  });
}

TEST(ConSan, InventoryUsesSemanticArchNotDisplayTarget) {
  constexpr std::array<uint32_t, 3> text_words = {0xDB78000Cu,
                                                  0x00000100u, // ds_store_b96 v0, v[1:3] offset:12
                                                  build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4)};
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words, "typed_arch_inventory");
  TestOptions options = test_options();
  options.track_atomics = false;
  options.track_barriers = false;
  options.report_buffer_address = 0x123456780000ull;
  options.report_buffer_size = 64u * 1024u * 1024u;
  TransformArtifacts result = test_lower_consan(bytes, options);
  ASSERT_TRUE(patch_succeeded(result)) << testing::PrintToString(result.errors);
  ASSERT_EQ(result.program_inventory.target(), ROCJITSU_CODE_TARGET_GFX1201);
  ASSERT_EQ(result.program_inventory.arch(), ROCJITSU_CODE_ARCH_RDNA4);

  ProgramInventoryBuilder display_target_revision(result.program_inventory);
  display_target_revision.set_code_object_facts(
      result.program_inventory.kernel_metadata_trustworthy(),
      result.program_inventory.malformed_kernel_metadata_note_count(),
      result.program_inventory.arch(), ROCJITSU_CODE_TARGET_GFX942);
  result.program_inventory = display_target_revision.view();
  EXPECT_EQ(plan_test_evidence_inventory(result, options).access_range_count, 1u);
}

TEST(ConSan, SharedAccessDecoderOwnsTwoRangeGeometryWithoutAdmission) {
  using detail::decode_native_lds_two_range_shape;

  struct ExpectedForm {
    std::string_view mnemonic;
    LdsAccessKind kind;
    uint32_t width_bits;
    uint32_t scale_bytes;
  };
  constexpr std::array expected_forms = {
      ExpectedForm{"ds_load_2addr_b32", LdsAccessKind::Read, 32u, 4u},
      ExpectedForm{"ds_store_2addr_b32", LdsAccessKind::Write, 32u, 4u},
      ExpectedForm{"ds_read2_b32", LdsAccessKind::Read, 32u, 4u},
      ExpectedForm{"ds_write2_b32", LdsAccessKind::Write, 32u, 4u},
      ExpectedForm{"ds_load_2addr_b64", LdsAccessKind::Read, 64u, 8u},
      ExpectedForm{"ds_store_2addr_b64", LdsAccessKind::Write, 64u, 8u},
      ExpectedForm{"ds_read2_b64", LdsAccessKind::Read, 64u, 8u},
      ExpectedForm{"ds_write2_b64", LdsAccessKind::Write, 64u, 8u},
      ExpectedForm{"ds_load_2addr_stride64_b32", LdsAccessKind::Read, 32u, 256u},
      ExpectedForm{"ds_store_2addr_stride64_b32", LdsAccessKind::Write, 32u, 256u},
      ExpectedForm{"ds_read2st64_b32", LdsAccessKind::Read, 32u, 256u},
      ExpectedForm{"ds_write2st64_b32", LdsAccessKind::Write, 32u, 256u},
      ExpectedForm{"ds_load_2addr_stride64_b64", LdsAccessKind::Read, 64u, 512u},
      ExpectedForm{"ds_store_2addr_stride64_b64", LdsAccessKind::Write, 64u, 512u},
      ExpectedForm{"ds_read2st64_b64", LdsAccessKind::Read, 64u, 512u},
      ExpectedForm{"ds_write2st64_b64", LdsAccessKind::Write, 64u, 512u},
  };
  for (const auto &expected : expected_forms) {
    SCOPED_TRACE(expected.mnemonic);
    const auto actual = decode_native_lds_two_range_shape(expected.mnemonic);
    ASSERT_TRUE(actual);
    EXPECT_EQ(actual->kind, expected.kind);
    EXPECT_EQ(actual->element_width_bits, expected.width_bits);
    EXPECT_EQ(actual->offset_scale_bytes, expected.scale_bytes);
  }
  EXPECT_FALSE(decode_native_lds_two_range_shape("ds_load_b32"));
}

TEST(ConSan, CdnaAdmitsNativeB96Accesses) {
  constexpr auto cdna3_store = cdna3::build_ds(cdna3::kDsWriteB96Ds, {.addr = 0, .data0 = 1});
  constexpr auto cdna3_load = cdna3::build_ds(cdna3::kDsReadB96Ds, {.addr = 0, .vdst = 4});
  constexpr auto cdna3_aliasing_load = cdna3::build_ds(cdna3::kDsReadB96Ds, {.addr = 0, .vdst = 0});
  constexpr auto cdna4_store = cdna4::build_ds(cdna4::kDsWriteB96Ds, {.addr = 0, .data0 = 1});
  constexpr auto cdna4_load = cdna4::build_ds(cdna4::kDsReadB96Ds, {.addr = 0, .vdst = 4});
  constexpr auto cdna4_aliasing_load = cdna4::build_ds(cdna4::kDsReadB96Ds, {.addr = 0, .vdst = 0});
  constexpr std::array cdna3_accesses = {
      NativeB96Access{cdna3_store, "ds_write_b96"},
      NativeB96Access{cdna3_load, "ds_read_b96"},
      NativeB96Access{cdna3_aliasing_load, "ds_read_b96"},
  };
  constexpr std::array cdna4_accesses = {
      NativeB96Access{cdna4_store, "ds_write_b96"},
      NativeB96Access{cdna4_load, "ds_read_b96"},
      NativeB96Access{cdna4_aliasing_load, "ds_read_b96"},
  };

  expect_admits_native_b96_accesses(
      ROCJITSU_CODE_ARCH_CDNA3, cdna3_accesses, [](const auto &text_words) {
        return make_cdna3_lds_code_object(text_words, "cdna3_native_b96_access");
      });
  expect_admits_native_b96_accesses(
      ROCJITSU_CODE_ARCH_CDNA4, cdna4_accesses, [](const auto &text_words) {
        return make_cdna4_lds_code_object(text_words, "cdna4_native_b96_access");
      });
}

TEST(ConSan, UnsupportedOnlyAccessRemainsApplicableInPreFilterLedger) {
  const std::array<uint32_t, 3> text_words = {
      0xDAC40000u,
      0x00000000u, // ds_load_addtid_b32 (implicit address, unsupported by ConSan)
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);

  TestOptions options = test_options();
  const TransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_TRUE(patch_succeeded(result));
  EXPECT_TRUE(test_admitted_accesses(result).empty());
  ASSERT_EQ(result.observation_plan().site_decisions.size(), 1u);
  const SiteDecision &decision = result.observation_plan().site_decisions.front();
  EXPECT_EQ(decision.kind, SiteDecisionKind::Unsupported);
  EXPECT_EQ(decision.reason, AccessPolicyReason::MissingAddressOperand);
  EXPECT_TRUE(result.observation_plan().probe_intents.empty());
}

TEST(ConSan, MixedAccessLedgerRetainsSupportedAndUnsupportedFinalCodeSites) {
  const std::array<uint32_t, 5> text_words = {
      0xD8340000u, 0x00000102u, // ds_store_b32 v2, v1
      0xDAC40000u, 0x00000000u, // ds_load_addtid_b32
      0xBFB00000u,              // s_endpgm
  };
  const std::vector<uint8_t> bytes = make_rdna4_lds_code_object(text_words);
  TestOptions options = test_options();

  const TransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_TRUE(patch_succeeded(result));
  ASSERT_EQ(result.observation_plan().site_decisions.size(), 2u);
  EXPECT_EQ(result.observation_plan().site_decisions[0].kind, SiteDecisionKind::Admitted);
  EXPECT_EQ(result.observation_plan().site_decisions[0].reason, AccessPolicyReason::None);
  EXPECT_EQ(result.observation_plan().site_decisions[1].kind, SiteDecisionKind::Unsupported);
  EXPECT_EQ(result.observation_plan().site_decisions[1].reason,
            AccessPolicyReason::MissingAddressOperand);
  ASSERT_EQ(result.coverage_ledger.intent_entries().size(), 1u);
  EXPECT_EQ(result.coverage_ledger.intent_entries().front().lowering,
            LoweringOutcomeKind::PlacementRejected);
}

TEST(ConSan, AutoReportInventoryCountsAdmittedLogicalRangesBeforeAllocation) {
  const std::vector<uint8_t> bytes = make_rdna4_supported_lds_code_object();
  TestOptions options = test_options();
  options.max_patches = 1u << 20u;
  options.runtime_sample_stride = 256u;
  const TransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_TRUE(patch_succeeded(result));
  const AutoReportInventory inventory = plan_test_evidence_inventory(result, options);
  EXPECT_EQ(inventory.access_range_count, 2u);
  EXPECT_EQ(inventory.range_bank_count, 16u);
  EXPECT_EQ(inventory.watchpoint_count, 16u);
  EXPECT_TRUE(plan_auto_report(inventory).complete());
}

TEST(ConSan, NoAccessConsumerLeavesDescriptorsAndCodeUnmodified) {
  for (rj_code_arch_t arch :
       {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA4}) {
    SCOPED_TRACE(arch);
    const std::array<uint32_t, 2> words = {build_s_nop(0, arch), build_s_endpgm(arch)};
    const auto bytes = arch == ROCJITSU_CODE_ARCH_RDNA3   ? make_rdna3_lds_code_object(words)
                       : arch == ROCJITSU_CODE_ARCH_RDNA4 ? make_rdna4_lds_code_object(words)
                                                          : make_cdna4_lds_code_object(words);
    TestOptions options = test_options();
    options.init_owner_epoch = true;
    options.set_owner_epoch_vgprs(11u, 12u);
    options.report_buffer_address = 0x100000000ull;
    options.report_buffer_size = direct_report_bytes(8);
    const auto result = test_lower_consan(bytes, options);
    ASSERT_TRUE(patch_succeeded(result));
    EXPECT_FALSE(result.modified());
    EXPECT_TRUE(result.patches.empty());
    EXPECT_TRUE(result.replacement.empty());
  }
}

TEST(ConSan, AutomaticPersistentProloguesOnlyTargetEmittedProbeOwners) {
  TwoKernelSharedFixtureOptions fixture;
  fixture.unrelated_has_lds = true;
  const std::vector<uint8_t> bytes = make_rdna4_two_kernel_shared_helper_code_object(fixture);

  TestOptions options = test_options();
  options.init_owner_epoch = true;
  options.report_buffer_address = 0x123456780000ull;
  options.report_buffer_size = direct_report_bytes(8);
  options.max_patches = 1;

  const TransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_TRUE(patch_succeeded(result));
  ASSERT_TRUE(result.modified()) << testing::PrintToString(result.warnings);
  const auto access_patch = std::ranges::find_if(result.patches, [](const PatchInfo &patch) {
    return patch.kind == PatchKind::InlineWatchpointStore ||
           patch.kind == PatchKind::TrampolineWatchpointStore;
  });
  ASSERT_NE(access_patch, result.patches.end());
  ASSERT_FALSE(access_patch->owner_descriptor_file_offsets.empty());

  const auto owns_access_patch = [&](uint64_t descriptor_offset) {
    return std::ranges::find(access_patch->owner_descriptor_file_offsets, descriptor_offset) !=
           access_patch->owner_descriptor_file_offsets.end();
  };
  size_t prologue_count = 0;
  for (const PatchInfo &patch : result.patches) {
    if (patch.kind != PatchKind::KernelEntryOwnerEpochPrologue)
      continue;
    ++prologue_count;
    ASSERT_EQ(patch.owner_descriptor_file_offsets.size(), 1u);
    EXPECT_TRUE(owns_access_patch(patch.owner_descriptor_file_offsets.front()));
  }
  EXPECT_EQ(prologue_count, access_patch->owner_descriptor_file_offsets.size());

  const auto omitted_planned_owner =
      std::ranges::find_if(result.resource_plans, [&](const CandidateResourcePlan &plan) {
        return std::ranges::any_of(plan.owner_kernel_ids, [&](ProgramContainerId owner) {
          const ProgramContainer *kernel = result.program_inventory.container(owner);
          return kernel != nullptr && !owns_access_patch(kernel->descriptor_file_offset);
        });
      });
  ASSERT_NE(omitted_planned_owner, result.resource_plans.end());
  const auto omitted_owner_descriptors =
      result.program_inventory.kernel_descriptors(omitted_planned_owner->owner_kernel_ids);
  ASSERT_TRUE(omitted_owner_descriptors);
  for (uint64_t owner : *omitted_owner_descriptors) {
    if (owns_access_patch(owner))
      continue;
    EXPECT_FALSE(std::ranges::any_of(result.patches, [&](const PatchInfo &patch) {
      return patch.kind == PatchKind::KernelEntryOwnerEpochPrologue &&
             std::ranges::find(patch.owner_descriptor_file_offsets, owner) !=
                 patch.owner_descriptor_file_offsets.end();
    }));
  }
}

TEST(ConSan, AutomaticStatePreservesGuestVgprAllocation) {
  const std::array<uint32_t, 4> text_words = {
      0xD8340000u,
      0x00000000u, // ds_store_b32 v0, v0
      build_v_mov_b32_e32(/*vdst=*/62, vector_source_vgpr(62), ROCJITSU_CODE_ARCH_RDNA4),
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  constexpr uint32_t kWave64Vgpr64Granulated = 15;
  const std::vector<uint8_t> bytes =
      make_rdna4_lds_code_object(text_words, "vgpr_pressure", kWave64Vgpr64Granulated);

  TestOptions options = test_options();
  options.init_owner_epoch = true;
  options.report_buffer_address = 0x123456780000ull;
  options.report_buffer_size = direct_report_bytes(8);
  options.max_patches = 1;

  const TransformArtifacts result = test_lower_consan(bytes, options);

  ASSERT_TRUE(patch_succeeded(result));
  ASSERT_TRUE(result.modified()) << testing::PrintToString(result.warnings);
  EXPECT_FALSE(test_owner_vgpr(result));
  EXPECT_FALSE(test_epoch_vgpr(result));
  EXPECT_EQ(std::ranges::count(result.patches, PatchKind::KernelEntryPrivateEpochPrologue,
                               &PatchInfo::kind),
            1);
  EXPECT_TRUE(std::ranges::any_of(result.patches, [](const PatchInfo &patch) {
    return patch.kind == PatchKind::InlineWatchpointStore ||
           patch.kind == PatchKind::TrampolineWatchpointStore;
  }));
  EXPECT_TRUE(std::ranges::all_of(result.resource_plans, [](const auto &plan) {
    return plan.required_vgpr_count <= plan.current_vgpr_count;
  }));
  AmdGpuCodeObject patched(result.replacement.data(), result.replacement.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_EQ(patched.kernels().size(), 1u);
  KD descriptor{};
  std::memcpy(&descriptor,
              result.replacement.data() + patched.kernels().front().descriptor_file_offset,
              sizeof(descriptor));
  EXPECT_EQ(AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc1,
                            kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT),
            kWave64Vgpr64Granulated);
}

TEST(ConSan, HardwareOwnerDoesNotUseWorkitemPrivateState) {
  const std::array<uint32_t, 4> words = {
      0xD8340000u,
      0x00000000u,
      build_v_mov_b32_e32(62, vector_source_vgpr(62), ROCJITSU_CODE_ARCH_RDNA4),
      build_s_endpgm(ROCJITSU_CODE_ARCH_RDNA4),
  };
  const auto bytes = make_rdna4_lds_code_object(words, "hardware_owner", 15);
  TestOptions options = test_options();
  options.owner_source = OwnerSource::HwId;
  options.init_owner_epoch = true;
  options.report_buffer_address = 0x123456780000ull;
  options.report_buffer_size = direct_report_bytes(8);
  options.max_patches = 1;
  const auto result = test_lower_consan(bytes, options);
  ASSERT_TRUE(patch_succeeded(result));
  ASSERT_TRUE(result.modified()) << testing::PrintToString(result.warnings);
  EXPECT_FALSE(result.operating_point.automatic_private_epoch);
  EXPECT_EQ(std::ranges::count(result.patches, PatchKind::KernelEntryOwnerEpochPrologue,
                               &PatchInfo::kind),
            1);
  EXPECT_EQ(std::ranges::count(result.patches, PatchKind::KernelEntryPrivateEpochPrologue,
                               &PatchInfo::kind),
            0);
}

TEST(ConSan, AtomicConsumersRejectUnqualifiedStandaloneMemoryRole) {
  const std::vector<uint8_t> bytes = make_rdna4_flat_atomic_code_object();
  TestOptions options = test_options();
  options.track_atomics = true;
  options.scratch_vgpr = 8;
  options.set_owner_epoch_vgprs(11, 12);
  options.report_buffer_address = 0x123456780000ull;
  options.report_buffer_size = direct_report_bytes(8);

  const auto result = test_lower_consan(bytes, options);

  ASSERT_TRUE(patch_succeeded(result));
  ASSERT_EQ(result.program_inventory.sync().sync_sequences.size(), 1u);
  EXPECT_EQ(result.program_inventory.sync().sync_sequences.front().memory_role,
            SyncMemoryRole::Unknown);
  EXPECT_EQ(result.program_inventory.sync().sync_sequences.front().memory_role_confidence,
            SemanticConfidence::Unsupported);
  EXPECT_FALSE(result.modified());
  EXPECT_TRUE(result.patches.empty());
  EXPECT_TRUE(result.resource_plans.empty());
}

TEST(ConSan, LdsCellRangesRoundUnalignedBytesToFourByteGranules) {
  constexpr LdsCellRange byte_0_to_3 = lds_cell_range_for_bytes(0, 4);
  constexpr LdsCellRange byte_3_to_4 = lds_cell_range_for_bytes(3, 2);
  constexpr LdsCellRange byte_8_to_11 = lds_cell_range_for_bytes(8, 4);
  constexpr LdsCellRange adjacent = lds_cell_range_for_bytes(12, 4);

  EXPECT_EQ(byte_0_to_3.start_cell, 0u);
  EXPECT_EQ(byte_0_to_3.cell_count, 1u);
  EXPECT_EQ(byte_3_to_4.start_cell, 0u);
  EXPECT_EQ(byte_3_to_4.cell_count, 2u);
  EXPECT_EQ(byte_8_to_11.start_cell, 2u);
  EXPECT_EQ(byte_8_to_11.cell_count, 1u);
  EXPECT_TRUE(cell_ranges_overlap(byte_3_to_4, byte_0_to_3));
  EXPECT_FALSE(cell_ranges_overlap(byte_8_to_11, adjacent));
}

TEST(ConSan, StrictFlatProvenanceExcludesMaybeGroupCandidates) {
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

  TestOptions likely_options;
  likely_options.mode = Mode::Default;
  const auto likely_result = test_lower_consan(bytes, likely_options);
  ASSERT_TRUE(likely_result.errors.empty());
  ASSERT_EQ(test_admitted_accesses(likely_result).size(), 1u);
  EXPECT_EQ(test_admitted_accesses(likely_result).front().origin, AccessOrigin::Flat);
  EXPECT_EQ(test_admitted_accesses(likely_result).front().flat_address_space_hint,
            FlatAddressSpaceHint::MaybeGroup);

  TestOptions strict_options = likely_options;
  strict_options.flat_provenance_mode = FlatProvenanceMode::Strict;
  const auto strict_result = test_lower_consan(bytes, strict_options);
  ASSERT_TRUE(strict_result.errors.empty());
  EXPECT_TRUE(test_admitted_accesses(strict_result).empty());
  ASSERT_EQ(strict_result.observation_plan().site_decisions.size(), 1u);
  EXPECT_EQ(strict_result.observation_plan().site_decisions.front().reason,
            AccessPolicyReason::FlatProvenancePolicyExcluded);
}

TEST(ConSan, CfgBuildInputsCanonicalizeInventoryAndComposedCodeRanges) {
  constexpr std::array<uint32_t, 1> kernel_words = {
      0xBFB00000u, // s_endpgm
  };
  constexpr std::array<uint32_t, 2> function_words = {
      0xBF800000u, // s_nop 0
      0xBFB00000u, // s_endpgm
  };
  const std::vector<uint8_t> bytes =
      make_rdna4_code_object_with_local_function(kernel_words, function_words);
  const TransformArtifacts inventory = test_lower_consan(bytes, test_options());
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
      PreappliedCodeRange{.text_offset = duplicate_kernel_entry,
                          .size = 0u,
                          .kernel_name = "kernel",
                          .continuation_text_offset = duplicate_function_entry},
      PreappliedCodeRange{.text_offset = kComposedEntry,
                          .size = 12u,
                          .kernel_name = "kernel",
                          .continuation_text_offset = kComposedContinuation},
  };

  const detail::CfgBuildInputs cfg =
      detail::build_cfg_inputs(code_object, inventory.program_inventory.containers(), preapplied);

  EXPECT_TRUE(std::ranges::is_sorted(cfg.leaders));
  EXPECT_TRUE(std::ranges::is_sorted(cfg.kernel_entries));
  EXPECT_EQ(std::ranges::count(cfg.leaders, duplicate_kernel_entry), 1u);
  EXPECT_EQ(std::ranges::count(cfg.leaders, duplicate_function_entry), 1u);
  EXPECT_NE(std::ranges::find(cfg.leaders, kComposedEntry), cfg.leaders.end());
  EXPECT_NE(std::ranges::find(cfg.leaders, kComposedContinuation), cfg.leaders.end());
  for (const ProgramContainer &kernel : inventory.program_inventory.kernels()) {
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

} // namespace
} // namespace rocjitsu::consan
