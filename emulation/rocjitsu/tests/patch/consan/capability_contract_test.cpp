// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_capability_contract.h"

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>

namespace rocjitsu {
namespace {

/// Independent expected values for one target-wide architectural contract.
///
/// This test-only record deliberately does not reuse a production profile row:
/// changing the production table must require an explicit review of every
/// expected architectural fact below.
struct ExpectedTargetProfile {
  rj_code_target_id_t target;
  rj_code_arch_t arch;
  ConSanArchitectureFamily architecture_family;
  ConSanEncodingFamily encoding_family;
  ConSanAccumulatorModel accumulator_model;
  ConSanDispatchIdentitySource dispatch_identity;
  ConSanWorkgroupIdentitySource workgroup_identity;
  ConSanWaitCounterFamily wait_counter_family;
  ConSanDirectCallForm direct_call_form;
  ConSanResidentWaveIdentityEncoding resident_wave_identity;
  bool supports_wave32;
  bool supports_wave64;
  uint8_t exec_register_width_bits;
  uint8_t global_address_width_bits;
  uint8_t lds_address_width_bits;
  uint8_t vgpr_allocation_granularity_wave32;
  uint8_t vgpr_allocation_granularity_wave64;
  uint8_t sgpr_allocation_granularity;
  uint8_t accumulator_offset_granularity;
  uint16_t ordinary_sgpr_limit;
  uint16_t reserved_ordinary_sgpr_base;
  uint16_t reserved_ordinary_sgpr_count;
  uint16_t user_sgpr_initialization_limit;
  uint32_t address_free_private_limit_bytes;
  uint32_t private_allocation_granularity_bytes;
  uint32_t max_group_segment_bytes;
  int32_t direct_branch_min_displacement_bytes;
  int32_t direct_branch_max_displacement_bytes;
  bool supports_kernarg_preload_overflow_recovery;
  bool has_cluster_facilities;
  bool has_selectable_vgpr_bank;
  bool requires_even_vgpr_tuples;
  bool requires_split_two_address_lds_relocation;
  uint16_t semantic_form_mask;
};

constexpr std::array<ExpectedTargetProfile, 5> kExpectedTargetProfiles = {{
    {
        .target = ROCJITSU_CODE_TARGET_GFX942,
        .arch = ROCJITSU_CODE_ARCH_CDNA3,
        .architecture_family = ConSanArchitectureFamily::Cdna,
        .encoding_family = ConSanEncodingFamily::Gfx9Cdna3,
        .accumulator_model = ConSanAccumulatorModel::DescriptorPartitioned,
        .dispatch_identity = ConSanDispatchIdentitySource::PreloadedSgprPair,
        .workgroup_identity = ConSanWorkgroupIdentitySource::DescriptorSystemSgprs,
        .wait_counter_family = ConSanWaitCounterFamily::Gfx9,
        .direct_call_form = ConSanDirectCallForm::SCallB64,
        .resident_wave_identity = {.hwreg_id = 4, .bit_offset = 0, .bit_width = 6},
        .supports_wave32 = false,
        .supports_wave64 = true,
        .exec_register_width_bits = 64,
        .global_address_width_bits = 64,
        .lds_address_width_bits = 32,
        .vgpr_allocation_granularity_wave32 = 0,
        .vgpr_allocation_granularity_wave64 = 8,
        .sgpr_allocation_granularity = 8,
        .accumulator_offset_granularity = 4,
        .ordinary_sgpr_limit = 102,
        .reserved_ordinary_sgpr_base = 0,
        .reserved_ordinary_sgpr_count = 0,
        .user_sgpr_initialization_limit = 16,
        .address_free_private_limit_bytes = 0x1000u,
        .private_allocation_granularity_bytes = 16,
        .max_group_segment_bytes = 64u * 1024u,
        .direct_branch_min_displacement_bytes = -131068,
        .direct_branch_max_displacement_bytes = 131072,
        .supports_kernarg_preload_overflow_recovery = true,
        .has_cluster_facilities = false,
        .has_selectable_vgpr_bank = false,
        .requires_even_vgpr_tuples = true,
        .requires_split_two_address_lds_relocation = false,
        .semantic_form_mask = kConSanCdnaSemanticFormMask,
    },
    {
        .target = ROCJITSU_CODE_TARGET_GFX950,
        .arch = ROCJITSU_CODE_ARCH_CDNA4,
        .architecture_family = ConSanArchitectureFamily::Cdna,
        .encoding_family = ConSanEncodingFamily::Gfx9Cdna4,
        .accumulator_model = ConSanAccumulatorModel::DescriptorPartitioned,
        .dispatch_identity = ConSanDispatchIdentitySource::PreloadedSgprPair,
        .workgroup_identity = ConSanWorkgroupIdentitySource::DescriptorSystemSgprs,
        .wait_counter_family = ConSanWaitCounterFamily::Gfx9,
        .direct_call_form = ConSanDirectCallForm::SCallB64,
        .resident_wave_identity = {.hwreg_id = 4, .bit_offset = 0, .bit_width = 6},
        .supports_wave32 = false,
        .supports_wave64 = true,
        .exec_register_width_bits = 64,
        .global_address_width_bits = 64,
        .lds_address_width_bits = 32,
        .vgpr_allocation_granularity_wave32 = 0,
        .vgpr_allocation_granularity_wave64 = 8,
        .sgpr_allocation_granularity = 8,
        .accumulator_offset_granularity = 4,
        .ordinary_sgpr_limit = 102,
        .reserved_ordinary_sgpr_base = 0,
        .reserved_ordinary_sgpr_count = 0,
        .user_sgpr_initialization_limit = 16,
        .address_free_private_limit_bytes = 0x1000u,
        .private_allocation_granularity_bytes = 16,
        .max_group_segment_bytes = 64u * 1024u,
        .direct_branch_min_displacement_bytes = -131068,
        .direct_branch_max_displacement_bytes = 131072,
        .supports_kernarg_preload_overflow_recovery = true,
        .has_cluster_facilities = false,
        .has_selectable_vgpr_bank = false,
        .requires_even_vgpr_tuples = true,
        .requires_split_two_address_lds_relocation = false,
        .semantic_form_mask = kConSanCdnaSemanticFormMask,
    },
    {
        .target = ROCJITSU_CODE_TARGET_GFX1100,
        .arch = ROCJITSU_CODE_ARCH_RDNA3,
        .architecture_family = ConSanArchitectureFamily::Rdna,
        .encoding_family = ConSanEncodingFamily::Gfx11,
        .accumulator_model = ConSanAccumulatorModel::None,
        .dispatch_identity = ConSanDispatchIdentitySource::CodeObjectLiteral,
        .workgroup_identity = ConSanWorkgroupIdentitySource::DescriptorSystemSgprs,
        .wait_counter_family = ConSanWaitCounterFamily::Gfx11,
        .direct_call_form = ConSanDirectCallForm::SCallB64,
        .resident_wave_identity = {.hwreg_id = 23, .bit_offset = 0, .bit_width = 10},
        .supports_wave32 = true,
        .supports_wave64 = true,
        .exec_register_width_bits = 64,
        .global_address_width_bits = 64,
        .lds_address_width_bits = 32,
        .vgpr_allocation_granularity_wave32 = 8,
        .vgpr_allocation_granularity_wave64 = 4,
        .sgpr_allocation_granularity = 8,
        .accumulator_offset_granularity = 0,
        .ordinary_sgpr_limit = 106,
        .reserved_ordinary_sgpr_base = 0,
        .reserved_ordinary_sgpr_count = 0,
        .user_sgpr_initialization_limit = 16,
        .address_free_private_limit_bytes = 0x1000u,
        .private_allocation_granularity_bytes = 16,
        .max_group_segment_bytes = 64u * 1024u,
        .direct_branch_min_displacement_bytes = -131068,
        .direct_branch_max_displacement_bytes = 131072,
        .supports_kernarg_preload_overflow_recovery = true,
        .has_cluster_facilities = false,
        .has_selectable_vgpr_bank = false,
        .requires_even_vgpr_tuples = false,
        .requires_split_two_address_lds_relocation = false,
        .semantic_form_mask = kConSanCommonSemanticFormMask,
    },
    {
        .target = ROCJITSU_CODE_TARGET_GFX1201,
        .arch = ROCJITSU_CODE_ARCH_RDNA4,
        .architecture_family = ConSanArchitectureFamily::Rdna,
        .encoding_family = ConSanEncodingFamily::Gfx12,
        .accumulator_model = ConSanAccumulatorModel::None,
        .dispatch_identity = ConSanDispatchIdentitySource::CodeObjectLiteral,
        .workgroup_identity = ConSanWorkgroupIdentitySource::CommandProcessorTtmps,
        .wait_counter_family = ConSanWaitCounterFamily::Gfx12,
        .direct_call_form = ConSanDirectCallForm::SCallB64,
        .resident_wave_identity = {.hwreg_id = 23, .bit_offset = 0, .bit_width = 10},
        .supports_wave32 = true,
        .supports_wave64 = true,
        .exec_register_width_bits = 64,
        .global_address_width_bits = 64,
        .lds_address_width_bits = 32,
        .vgpr_allocation_granularity_wave32 = 8,
        .vgpr_allocation_granularity_wave64 = 4,
        .sgpr_allocation_granularity = 8,
        .accumulator_offset_granularity = 0,
        .ordinary_sgpr_limit = 106,
        .reserved_ordinary_sgpr_base = 0,
        .reserved_ordinary_sgpr_count = 0,
        .user_sgpr_initialization_limit = 16,
        .address_free_private_limit_bytes = 0x800000u,
        .private_allocation_granularity_bytes = 1,
        .max_group_segment_bytes = 64u * 1024u,
        .direct_branch_min_displacement_bytes = -131068,
        .direct_branch_max_displacement_bytes = 131072,
        .supports_kernarg_preload_overflow_recovery = false,
        .has_cluster_facilities = false,
        .has_selectable_vgpr_bank = false,
        .requires_even_vgpr_tuples = false,
        .requires_split_two_address_lds_relocation = false,
        .semantic_form_mask = kConSanCommonSemanticFormMask,
    },
    {
        .target = ROCJITSU_CODE_TARGET_GFX1250,
        .arch = ROCJITSU_CODE_ARCH_CDNA5,
        .architecture_family = ConSanArchitectureFamily::Cdna,
        .encoding_family = ConSanEncodingFamily::Gfx12,
        .accumulator_model = ConSanAccumulatorModel::SelectableVgprBank,
        .dispatch_identity = ConSanDispatchIdentitySource::CodeObjectLiteral,
        .workgroup_identity = ConSanWorkgroupIdentitySource::CommandProcessorTtmps,
        .wait_counter_family = ConSanWaitCounterFamily::Gfx12,
        .direct_call_form = ConSanDirectCallForm::SCallI64,
        .resident_wave_identity = {.hwreg_id = 23, .bit_offset = 0, .bit_width = 10},
        .supports_wave32 = true,
        .supports_wave64 = true,
        .exec_register_width_bits = 64,
        .global_address_width_bits = 64,
        .lds_address_width_bits = 32,
        .vgpr_allocation_granularity_wave32 = 16,
        .vgpr_allocation_granularity_wave64 = 8,
        .sgpr_allocation_granularity = 8,
        .accumulator_offset_granularity = 0,
        .ordinary_sgpr_limit = 106,
        .reserved_ordinary_sgpr_base = 102,
        .reserved_ordinary_sgpr_count = 4,
        .user_sgpr_initialization_limit = 32,
        .address_free_private_limit_bytes = 0x800000u,
        .private_allocation_granularity_bytes = 1,
        .max_group_segment_bytes = static_cast<uint32_t>(ROCJITSU_GFX1250_LDS_SIZE_KB) * 1024u,
        .direct_branch_min_displacement_bytes = -131068,
        .direct_branch_max_displacement_bytes = 131072,
        .supports_kernarg_preload_overflow_recovery = false,
        .has_cluster_facilities = true,
        .has_selectable_vgpr_bank = true,
        .requires_even_vgpr_tuples = true,
        .requires_split_two_address_lds_relocation = true,
        .semantic_form_mask = kConSanGfx1250SemanticFormMask,
    },
}};

void expect_profile_matches(const ConSanTargetProfile &actual,
                            const ExpectedTargetProfile &expected) {
  EXPECT_EQ(actual.target, expected.target);
  EXPECT_EQ(actual.arch, expected.arch);
  EXPECT_EQ(actual.architecture_family, expected.architecture_family);
  EXPECT_EQ(actual.encoding_family, expected.encoding_family);
  EXPECT_EQ(actual.accumulator_model, expected.accumulator_model);
  EXPECT_EQ(actual.dispatch_identity, expected.dispatch_identity);
  EXPECT_EQ(actual.workgroup_identity, expected.workgroup_identity);
  EXPECT_EQ(actual.wait_counter_family, expected.wait_counter_family);
  EXPECT_EQ(actual.direct_call_form, expected.direct_call_form);
  EXPECT_EQ(actual.resident_wave_identity, expected.resident_wave_identity);
  EXPECT_EQ(actual.supports_wave32, expected.supports_wave32);
  EXPECT_EQ(actual.supports_wave64, expected.supports_wave64);
  EXPECT_EQ(actual.exec_register_width_bits, expected.exec_register_width_bits);
  EXPECT_EQ(actual.global_address_width_bits, expected.global_address_width_bits);
  EXPECT_EQ(actual.lds_address_width_bits, expected.lds_address_width_bits);
  EXPECT_EQ(actual.vgpr_allocation_granularity_wave32, expected.vgpr_allocation_granularity_wave32);
  EXPECT_EQ(actual.vgpr_allocation_granularity_wave64, expected.vgpr_allocation_granularity_wave64);
  EXPECT_EQ(actual.sgpr_allocation_granularity, expected.sgpr_allocation_granularity);
  EXPECT_EQ(actual.accumulator_offset_granularity, expected.accumulator_offset_granularity);
  EXPECT_EQ(actual.ordinary_sgpr_limit, expected.ordinary_sgpr_limit);
  EXPECT_EQ(actual.reserved_ordinary_sgpr_base, expected.reserved_ordinary_sgpr_base);
  EXPECT_EQ(actual.reserved_ordinary_sgpr_count, expected.reserved_ordinary_sgpr_count);
  EXPECT_EQ(actual.user_sgpr_initialization_limit, expected.user_sgpr_initialization_limit);
  EXPECT_EQ(actual.address_free_private_limit_bytes, expected.address_free_private_limit_bytes);
  EXPECT_EQ(actual.private_allocation_granularity_bytes,
            expected.private_allocation_granularity_bytes);
  EXPECT_EQ(actual.max_group_segment_bytes, expected.max_group_segment_bytes);
  EXPECT_EQ(actual.direct_branch_min_displacement_bytes,
            expected.direct_branch_min_displacement_bytes);
  EXPECT_EQ(actual.direct_branch_max_displacement_bytes,
            expected.direct_branch_max_displacement_bytes);
  EXPECT_EQ(actual.supports_kernarg_preload_overflow_recovery,
            expected.supports_kernarg_preload_overflow_recovery);
  EXPECT_EQ(actual.has_cluster_facilities, expected.has_cluster_facilities);
  EXPECT_EQ(actual.has_selectable_vgpr_bank, expected.has_selectable_vgpr_bank);
  EXPECT_EQ(actual.requires_even_vgpr_tuples, expected.requires_even_vgpr_tuples);
  EXPECT_EQ(actual.requires_split_two_address_lds_relocation,
            expected.requires_split_two_address_lds_relocation);
  EXPECT_EQ(actual.semantic_form_mask, expected.semantic_form_mask);
}

TEST(ConSanCapabilityContract, TargetProfileRowsDeclareEveryArchitecturalFact) {
  ASSERT_EQ(kConSanTargetProfiles.size(), kExpectedTargetProfiles.size());
  EXPECT_TRUE(consan_target_profiles_are_valid());
  for (size_t index = 0; index < kExpectedTargetProfiles.size(); ++index) {
    SCOPED_TRACE(rj_code_target_name(kExpectedTargetProfiles[index].target));
    expect_profile_matches(kConSanTargetProfiles[index], kExpectedTargetProfiles[index]);
  }
}

TEST(ConSanCapabilityContract, TargetProfileValidatorRejectsEveryMalformedInvariant) {
  constexpr std::array<ConSanTargetProfile, 0> empty_profiles = {};
  EXPECT_FALSE(consan_target_profiles_are_valid(empty_profiles));

  const auto expect_invalid = [](std::string_view reason, auto mutate) {
    SCOPED_TRACE(reason);
    auto profiles = kConSanTargetProfiles;
    mutate(profiles);
    EXPECT_FALSE(consan_target_profiles_are_valid(profiles));
  };

  expect_invalid("invalid target",
                 [](auto &profiles) { profiles[0].target = ROCJITSU_CODE_TARGET_INVALID; });
  expect_invalid("invalid architecture",
                 [](auto &profiles) { profiles[0].arch = ROCJITSU_CODE_ARCH_INVALID; });
  expect_invalid("wave64 unsupported", [](auto &profiles) { profiles[0].supports_wave64 = false; });
  expect_invalid("EXEC width", [](auto &profiles) { profiles[0].exec_register_width_bits = 32u; });
  expect_invalid("global address width",
                 [](auto &profiles) { profiles[0].global_address_width_bits = 32u; });
  expect_invalid("LDS address width",
                 [](auto &profiles) { profiles[0].lds_address_width_bits = 64u; });
  expect_invalid("missing wave64 allocation granularity",
                 [](auto &profiles) { profiles[0].vgpr_allocation_granularity_wave64 = 0u; });
  expect_invalid("wave32 support without allocation granularity",
                 [](auto &profiles) { profiles[0].supports_wave32 = true; });
  expect_invalid("wave32 allocation granularity without support",
                 [](auto &profiles) { profiles[2].supports_wave32 = false; });
  expect_invalid("missing SGPR allocation granularity",
                 [](auto &profiles) { profiles[0].sgpr_allocation_granularity = 0u; });
  expect_invalid("missing ordinary SGPR limit",
                 [](auto &profiles) { profiles[0].ordinary_sgpr_limit = 0u; });
  expect_invalid("missing user SGPR initialization limit",
                 [](auto &profiles) { profiles[0].user_sgpr_initialization_limit = 0u; });
  expect_invalid("missing private limit",
                 [](auto &profiles) { profiles[0].address_free_private_limit_bytes = 0u; });
  expect_invalid("missing private granularity",
                 [](auto &profiles) { profiles[0].private_allocation_granularity_bytes = 0u; });
  expect_invalid("missing group segment limit",
                 [](auto &profiles) { profiles[0].max_group_segment_bytes = 0u; });
  expect_invalid("nonnegative minimum branch displacement",
                 [](auto &profiles) { profiles[0].direct_branch_min_displacement_bytes = 0; });
  expect_invalid("nonpositive maximum branch displacement",
                 [](auto &profiles) { profiles[0].direct_branch_max_displacement_bytes = 0; });
  expect_invalid("resident-wave HWREG ID outside encoding",
                 [](auto &profiles) { profiles[0].resident_wave_identity.hwreg_id = 64u; });
  expect_invalid("resident-wave HWREG offset outside encoding",
                 [](auto &profiles) { profiles[0].resident_wave_identity.bit_offset = 32u; });
  expect_invalid("empty resident-wave identity",
                 [](auto &profiles) { profiles[0].resident_wave_identity.bit_width = 0u; });
  expect_invalid("oversized resident-wave identity",
                 [](auto &profiles) { profiles[0].resident_wave_identity.bit_width = 33u; });
  expect_invalid("resident-wave identity extends past its HWREG", [](auto &profiles) {
    profiles[0].resident_wave_identity.bit_offset = 31u;
    profiles[0].resident_wave_identity.bit_width = 2u;
  });
  expect_invalid("two-address relocation split on a non-gfx12 encoding", [](auto &profiles) {
    profiles[0].requires_split_two_address_lds_relocation = true;
  });
  expect_invalid("semantic form bit outside the enum", [](auto &profiles) {
    profiles[0].semantic_form_mask = static_cast<uint16_t>(
        profiles[0].semantic_form_mask | (1u << static_cast<uint8_t>(ConSanCapabilityForm::Count)));
  });
  expect_invalid("empty semantic form mask",
                 [](auto &profiles) { profiles[0].semantic_form_mask = 0u; });
  expect_invalid("selectable bank without selectable-bank accumulator model",
                 [](auto &profiles) { profiles[0].has_selectable_vgpr_bank = true; });
  expect_invalid("selectable-bank accumulator model without selectable bank",
                 [](auto &profiles) { profiles[4].has_selectable_vgpr_bank = false; });
  expect_invalid("cluster facility without cluster semantic form", [](auto &profiles) {
    profiles[4].semantic_form_mask =
        static_cast<uint16_t>(profiles[4].semantic_form_mask &
                              ~consan_capability_form_bit(ConSanCapabilityForm::ClusterBarrier));
  });
  expect_invalid("reserved SGPR base without count",
                 [](auto &profiles) { profiles[0].reserved_ordinary_sgpr_base = 1u; });
  expect_invalid("reserved SGPR count without base",
                 [](auto &profiles) { profiles[0].reserved_ordinary_sgpr_count = 1u; });
  expect_invalid("reserved SGPR range beyond ordinary limit",
                 [](auto &profiles) { profiles[4].reserved_ordinary_sgpr_count = 5u; });
  expect_invalid("duplicate target",
                 [](auto &profiles) { profiles[1].target = profiles[0].target; });
  expect_invalid("duplicate architecture",
                 [](auto &profiles) { profiles[1].arch = profiles[0].arch; });
}

TEST(ConSanCapabilityContract, TargetProfileLookupIsTotalUniqueAndRejectsUnsupportedValues) {
  for (size_t index = 0; index < kExpectedTargetProfiles.size(); ++index) {
    const ExpectedTargetProfile &expected = kExpectedTargetProfiles[index];
    const ConSanTargetProfile &profile = kConSanTargetProfiles[index];
    SCOPED_TRACE(rj_code_target_name(expected.target));
    EXPECT_EQ(consan_target_profile(expected.target), &profile);
    EXPECT_EQ(consan_target_profile(expected.arch), &profile);
    EXPECT_EQ(consan_arch_for_target(expected.target), expected.arch);
    EXPECT_TRUE(consan_is_capability_target(expected.target));
    EXPECT_TRUE(consan_is_capability_arch(expected.arch));
    EXPECT_EQ(consan_arch_is_cdna(expected.arch),
              expected.architecture_family == ConSanArchitectureFamily::Cdna);
    EXPECT_EQ(consan_arch_is_rdna(expected.arch),
              expected.architecture_family == ConSanArchitectureFamily::Rdna);
    EXPECT_EQ(consan_arch_supports_kernarg_preload_overflow_recovery(expected.arch),
              expected.supports_kernarg_preload_overflow_recovery);
    for (size_t other = index + 1; other < kConSanTargetProfiles.size(); ++other) {
      EXPECT_NE(profile.target, kConSanTargetProfiles[other].target);
      EXPECT_NE(profile.arch, kConSanTargetProfiles[other].arch);
    }
  }

  constexpr std::array unsupported_targets = {
      ROCJITSU_CODE_TARGET_INVALID,
      ROCJITSU_CODE_TARGET_GFX90A,
      ROCJITSU_CODE_TARGET_GFX1200,
  };
  for (rj_code_target_id_t target : unsupported_targets) {
    EXPECT_EQ(consan_target_profile(target), nullptr);
    EXPECT_EQ(consan_arch_for_target(target), ROCJITSU_CODE_ARCH_INVALID);
    EXPECT_FALSE(consan_is_capability_target(target));
  }
  constexpr std::array unsupported_arches = {
      ROCJITSU_CODE_ARCH_INVALID,
      ROCJITSU_CODE_ARCH_CDNA2,
      ROCJITSU_CODE_ARCH_RDNA3_5,
  };
  for (rj_code_arch_t arch : unsupported_arches) {
    EXPECT_EQ(consan_target_profile(arch), nullptr);
    EXPECT_FALSE(consan_is_capability_arch(arch));
    EXPECT_FALSE(consan_arch_is_cdna(arch));
    EXPECT_FALSE(consan_arch_supports_kernarg_preload_overflow_recovery(arch));
  }
}

TEST(ConSanCapabilityContract, KernelTargetProfileValidatesWaveSelectionAndAllocation) {
  for (size_t index = 0; index < kExpectedTargetProfiles.size(); ++index) {
    const ExpectedTargetProfile &expected = kExpectedTargetProfiles[index];
    const ConSanTargetProfile &profile = kConSanTargetProfiles[index];
    SCOPED_TRACE(rj_code_target_name(expected.target));

    EXPECT_EQ(consan_profile_supports_wave_size(profile, 32u), expected.supports_wave32);
    EXPECT_TRUE(consan_profile_supports_wave_size(profile, 64u));
    EXPECT_FALSE(consan_profile_supports_wave_size(profile, 0u));
    EXPECT_FALSE(consan_profile_supports_wave_size(profile, 16u));
    EXPECT_FALSE(consan_profile_supports_wave_size(profile, 128u));
    EXPECT_EQ(consan_profile_vgpr_allocation_granularity(profile, 32u),
              expected.vgpr_allocation_granularity_wave32);
    EXPECT_EQ(consan_profile_vgpr_allocation_granularity(profile, 64u),
              expected.vgpr_allocation_granularity_wave64);
    EXPECT_EQ(consan_profile_vgpr_allocation_granularity(profile, 16u), 0u);

    const std::optional<ConSanKernelTargetProfile> wave64 =
        consan_kernel_target_profile(profile, 64u);
    ASSERT_TRUE(wave64);
    EXPECT_EQ(wave64->target, &profile);
    EXPECT_EQ(wave64->wave_size, 64u);
    EXPECT_EQ(wave64->active_exec_mask_width_bits, 64u);
    EXPECT_EQ(wave64->vgpr_allocation_granularity, expected.vgpr_allocation_granularity_wave64);

    const std::optional<ConSanKernelTargetProfile> wave32 =
        consan_kernel_target_profile(profile, 32u);
    EXPECT_EQ(wave32.has_value(), expected.supports_wave32);
    if (wave32) {
      EXPECT_EQ(wave32->target, &profile);
      EXPECT_EQ(wave32->wave_size, 32u);
      EXPECT_EQ(wave32->active_exec_mask_width_bits, 32u);
      EXPECT_EQ(wave32->vgpr_allocation_granularity, expected.vgpr_allocation_granularity_wave32);
    }
    EXPECT_FALSE(consan_kernel_target_profile(profile, 0u));
    EXPECT_FALSE(consan_kernel_target_profile(profile, 16u));
    EXPECT_FALSE(consan_kernel_target_profile(profile, 128u));
  }
}

TEST(ConSanCapabilityContract, ReservedSgprOverlapUsesHalfOpenRanges) {
  const ConSanTargetProfile *gfx1250 = consan_target_profile(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(gfx1250, nullptr);
  ASSERT_EQ(gfx1250->reserved_ordinary_sgpr_base, 102u);
  ASSERT_EQ(gfx1250->reserved_ordinary_sgpr_count, 4u);

  EXPECT_FALSE(consan_profile_reserved_sgpr_range_overlaps(*gfx1250, 0u, 0u));
  EXPECT_FALSE(consan_profile_reserved_sgpr_range_overlaps(*gfx1250, 0u, 102u));
  EXPECT_TRUE(consan_profile_reserved_sgpr_range_overlaps(*gfx1250, 0u, 103u));
  EXPECT_FALSE(consan_profile_reserved_sgpr_range_overlaps(*gfx1250, 101u, 1u));
  EXPECT_TRUE(consan_profile_reserved_sgpr_range_overlaps(*gfx1250, 101u, 2u));
  EXPECT_TRUE(consan_profile_reserved_sgpr_range_overlaps(*gfx1250, 102u, 1u));
  EXPECT_TRUE(consan_profile_reserved_sgpr_range_overlaps(*gfx1250, 105u, 1u));
  EXPECT_TRUE(consan_profile_reserved_sgpr_range_overlaps(*gfx1250, 105u, 2u));
  EXPECT_FALSE(consan_profile_reserved_sgpr_range_overlaps(*gfx1250, 106u, 1u));
  EXPECT_FALSE(consan_profile_reserved_sgpr_range_overlaps(
      *gfx1250, std::numeric_limits<uint16_t>::max(), std::numeric_limits<uint16_t>::max()));

  const ConSanTargetProfile *gfx950 = consan_target_profile(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(gfx950, nullptr);
  EXPECT_FALSE(consan_profile_reserved_sgpr_range_overlaps(*gfx950, 0u, 106u));
  EXPECT_FALSE(consan_profile_reserved_sgpr_range_overlaps(*gfx950, 102u, 4u));
}

TEST(ConSanCapabilityContract, PrivateSizeNormalizationCoversGranularityLimitAndOverflow) {
  const ConSanTargetProfile *gfx950 = consan_target_profile(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(gfx950, nullptr);
  EXPECT_EQ(consan_profile_normalize_private_size(*gfx950, 0u), 0u);
  EXPECT_EQ(consan_profile_normalize_private_size(*gfx950, 1u), 16u);
  EXPECT_EQ(consan_profile_normalize_private_size(*gfx950, 15u), 16u);
  EXPECT_EQ(consan_profile_normalize_private_size(*gfx950, 16u), 16u);
  EXPECT_EQ(consan_profile_normalize_private_size(*gfx950, 17u), 32u);
  EXPECT_EQ(consan_profile_normalize_private_size(*gfx950, 0xfffu), 0x1000u);
  EXPECT_EQ(consan_profile_normalize_private_size(*gfx950, 0x1000u), 0x1000u);
  EXPECT_FALSE(consan_profile_normalize_private_size(*gfx950, 0x1001u));
  EXPECT_FALSE(
      consan_profile_normalize_private_size(*gfx950, std::numeric_limits<uint32_t>::max()));

  const ConSanTargetProfile *gfx1201 = consan_target_profile(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(gfx1201, nullptr);
  EXPECT_EQ(consan_profile_normalize_private_size(*gfx1201, 0u), 0u);
  EXPECT_EQ(consan_profile_normalize_private_size(*gfx1201, 1u), 1u);
  EXPECT_EQ(consan_profile_normalize_private_size(*gfx1201, 0x800000u), 0x800000u);
  EXPECT_FALSE(consan_profile_normalize_private_size(*gfx1201, 0x800001u));

  for (const ExpectedTargetProfile &expected : kExpectedTargetProfiles) {
    SCOPED_TRACE(rj_code_target_name(expected.target));
    EXPECT_EQ(consan_address_free_private_limit(expected.arch),
              expected.address_free_private_limit_bytes);
    EXPECT_EQ(consan_normalize_address_free_private_size(expected.arch, 1u),
              expected.private_allocation_granularity_bytes);
    EXPECT_FALSE(consan_normalize_address_free_private_size(
        expected.arch, expected.address_free_private_limit_bytes + 1u));
  }
  EXPECT_FALSE(consan_address_free_private_limit(ROCJITSU_CODE_ARCH_CDNA2));
  EXPECT_FALSE(consan_normalize_address_free_private_size(ROCJITSU_CODE_ARCH_CDNA2, 1u));
}

TEST(ConSanCapabilityContract, DerivedArchitecturePredicatesProjectOnlyTheirTypedFacts) {
  for (const ExpectedTargetProfile &expected : kExpectedTargetProfiles) {
    SCOPED_TRACE(rj_code_target_name(expected.target));
    EXPECT_EQ(consan_uses_gfx9_cdna_encoding(expected.arch),
              expected.encoding_family == ConSanEncodingFamily::Gfx9Cdna3 ||
                  expected.encoding_family == ConSanEncodingFamily::Gfx9Cdna4);
    EXPECT_EQ(consan_uses_gfx11_encoding(expected.arch),
              expected.encoding_family == ConSanEncodingFamily::Gfx11);
    EXPECT_EQ(consan_uses_gfx12_encoding(expected.arch),
              expected.encoding_family == ConSanEncodingFamily::Gfx12);
    EXPECT_EQ(consan_uses_gfx12_cdna_execution(expected.arch),
              expected.encoding_family == ConSanEncodingFamily::Gfx12 &&
                  expected.architecture_family == ConSanArchitectureFamily::Cdna);
    EXPECT_EQ(consan_uses_gfx12_rdna_execution(expected.arch),
              expected.encoding_family == ConSanEncodingFamily::Gfx12 &&
                  expected.architecture_family == ConSanArchitectureFamily::Rdna);
    EXPECT_EQ(consan_uses_gfx11_or_gfx12_encoding(expected.arch),
              expected.encoding_family == ConSanEncodingFamily::Gfx11 ||
                  expected.encoding_family == ConSanEncodingFamily::Gfx12);
    EXPECT_EQ(consan_arch_has_s_call_i64(expected.arch),
              expected.direct_call_form == ConSanDirectCallForm::SCallI64);
    EXPECT_EQ(consan_arch_has_s_call_b64(expected.arch),
              expected.direct_call_form == ConSanDirectCallForm::SCallB64);
    EXPECT_EQ(consan_arch_has_cluster_facilities(expected.arch), expected.has_cluster_facilities);
    EXPECT_EQ(consan_arch_has_selectable_vgpr_bank(expected.arch),
              expected.has_selectable_vgpr_bank);
    EXPECT_EQ(consan_arch_has_descriptor_partitioned_accumulators(expected.arch),
              expected.accumulator_model == ConSanAccumulatorModel::DescriptorPartitioned);
    EXPECT_EQ(consan_arch_uses_literal_dispatch_identity(expected.arch),
              expected.dispatch_identity == ConSanDispatchIdentitySource::CodeObjectLiteral);
  }

  constexpr rj_code_arch_t unsupported = ROCJITSU_CODE_ARCH_CDNA2;
  EXPECT_FALSE(consan_uses_gfx9_cdna_encoding(unsupported));
  EXPECT_FALSE(consan_uses_gfx11_encoding(unsupported));
  EXPECT_FALSE(consan_uses_gfx12_encoding(unsupported));
  EXPECT_FALSE(consan_uses_gfx12_cdna_execution(unsupported));
  EXPECT_FALSE(consan_uses_gfx12_rdna_execution(unsupported));
  EXPECT_FALSE(consan_uses_gfx11_or_gfx12_encoding(unsupported));
  EXPECT_FALSE(consan_arch_is_cdna(unsupported));
  EXPECT_FALSE(consan_arch_is_rdna(unsupported));
  EXPECT_FALSE(consan_arch_has_s_call_i64(unsupported));
  EXPECT_FALSE(consan_arch_has_s_call_b64(unsupported));
  EXPECT_FALSE(consan_arch_has_cluster_facilities(unsupported));
  EXPECT_FALSE(consan_arch_has_selectable_vgpr_bank(unsupported));
  EXPECT_FALSE(consan_arch_has_descriptor_partitioned_accumulators(unsupported));
  EXPECT_FALSE(consan_arch_uses_literal_dispatch_identity(unsupported));
}

TEST(ConSanCapabilityContract, ProfileResourceAndCallFactsAgreeWithSharedBuilders) {
  constexpr uint64_t branch_pc = 200000u;
  for (const ExpectedTargetProfile &expected : kExpectedTargetProfiles) {
    SCOPED_TRACE(rj_code_target_name(expected.target));
    EXPECT_EQ(address_free_scratch_private_limit(expected.arch),
              expected.address_free_private_limit_bytes);
    EXPECT_EQ(normalize_address_free_scratch_private_size(expected.arch, 1u),
              expected.private_allocation_granularity_bytes);
    EXPECT_EQ(instrumentation::build_s_call_i64(0u, 0, expected.arch).has_value(),
              expected.direct_call_form == ConSanDirectCallForm::SCallI64);

    const int64_t minimum_target =
        static_cast<int64_t>(branch_pc) + expected.direct_branch_min_displacement_bytes;
    const int64_t maximum_target =
        static_cast<int64_t>(branch_pc) + expected.direct_branch_max_displacement_bytes;
    ASSERT_GE(minimum_target, 0);
    EXPECT_TRUE(compute_sopp_branch_simm16(branch_pc, static_cast<uint64_t>(minimum_target)));
    EXPECT_FALSE(compute_sopp_branch_simm16(branch_pc, static_cast<uint64_t>(minimum_target - 4)));
    EXPECT_TRUE(compute_sopp_branch_simm16(branch_pc, static_cast<uint64_t>(maximum_target)));
    EXPECT_FALSE(compute_sopp_branch_simm16(branch_pc, static_cast<uint64_t>(maximum_target + 4)));
  }
}

TEST(ConSanCapabilityContract, EnumIterationTablesAreCompleteAndRejectMalformedTables) {
  constexpr std::array expected_engines = {
      ConSanCapabilityEngine::SuperCollider,
      ConSanCapabilityEngine::RecordReplay,
      ConSanCapabilityEngine::Sampled,
      ConSanCapabilityEngine::InlineShadow,
  };
  constexpr std::array expected_domains = {
      ConSanCapabilityDomain::Access,
      ConSanCapabilityDomain::Barrier,
      ConSanCapabilityDomain::Atomic,
      ConSanCapabilityDomain::Fence,
  };
  constexpr std::array expected_forms = {
      ConSanCapabilityForm::NativeLdsAccess,        ConSanCapabilityForm::GroupFlatAccess,
      ConSanCapabilityForm::WorkgroupBarrier,       ConSanCapabilityForm::ClusterBarrier,
      ConSanCapabilityForm::OrderedFlatAtomic,      ConSanCapabilityForm::OrderedVglobalAtomic,
      ConSanCapabilityForm::OrderedLdsAtomic,       ConSanCapabilityForm::RelaxedLdsAtomicAccess,
      ConSanCapabilityForm::AddressedOrdinaryFence,
  };
  EXPECT_EQ(kConSanCapabilityEngines, expected_engines);
  EXPECT_EQ(kConSanCapabilityDomains, expected_domains);
  EXPECT_EQ(kConSanCapabilityForms, expected_forms);
  EXPECT_TRUE(consan_capability_enum_is_complete(kConSanCapabilityEngines));
  EXPECT_TRUE(consan_capability_enum_is_complete(kConSanCapabilityDomains));
  EXPECT_TRUE(consan_capability_enum_is_complete(kConSanCapabilityForms));

  constexpr std::array duplicate_engines = {
      ConSanCapabilityEngine::SuperCollider,
      ConSanCapabilityEngine::RecordReplay,
      ConSanCapabilityEngine::Sampled,
      ConSanCapabilityEngine::Sampled,
  };
  constexpr std::array short_engines = {
      ConSanCapabilityEngine::SuperCollider,
      ConSanCapabilityEngine::RecordReplay,
      ConSanCapabilityEngine::Sampled,
  };
  constexpr std::array out_of_range_engines = {
      ConSanCapabilityEngine::SuperCollider,
      ConSanCapabilityEngine::RecordReplay,
      ConSanCapabilityEngine::Sampled,
      static_cast<ConSanCapabilityEngine>(255),
  };
  EXPECT_FALSE(consan_capability_enum_is_complete(duplicate_engines));
  EXPECT_FALSE(consan_capability_enum_is_complete(short_engines));
  EXPECT_FALSE(consan_capability_enum_is_complete(out_of_range_engines));
}

TEST(ConSanCapabilityContract, CapabilityFormBitsAreUniqueBoundedAndComposeDeclaredMasks) {
  uint16_t seen = 0u;
  for (size_t index = 0; index < kConSanCapabilityForms.size(); ++index) {
    const ConSanCapabilityForm form = kConSanCapabilityForms[index];
    const uint16_t bit = consan_capability_form_bit(form);
    EXPECT_EQ(bit, static_cast<uint16_t>(1u << index));
    EXPECT_EQ(seen & bit, 0u);
    seen = static_cast<uint16_t>(seen | bit);
  }
  EXPECT_EQ(seen,
            static_cast<uint16_t>((1u << static_cast<uint8_t>(ConSanCapabilityForm::Count)) - 1u));
  EXPECT_EQ(consan_capability_form_bit(ConSanCapabilityForm::Count), 0u);
  EXPECT_EQ(consan_capability_form_bit(static_cast<ConSanCapabilityForm>(255)), 0u);

  const uint16_t expected_common =
      consan_capability_form_bit(ConSanCapabilityForm::NativeLdsAccess) |
      consan_capability_form_bit(ConSanCapabilityForm::GroupFlatAccess) |
      consan_capability_form_bit(ConSanCapabilityForm::WorkgroupBarrier) |
      consan_capability_form_bit(ConSanCapabilityForm::OrderedFlatAtomic) |
      consan_capability_form_bit(ConSanCapabilityForm::OrderedVglobalAtomic) |
      consan_capability_form_bit(ConSanCapabilityForm::AddressedOrdinaryFence);
  EXPECT_EQ(kConSanCommonSemanticFormMask, expected_common);
  EXPECT_EQ(kConSanCdnaSemanticFormMask,
            static_cast<uint16_t>(
                expected_common |
                consan_capability_form_bit(ConSanCapabilityForm::RelaxedLdsAtomicAccess)));
  EXPECT_EQ(
      kConSanGfx1250SemanticFormMask,
      static_cast<uint16_t>(kConSanCdnaSemanticFormMask |
                            consan_capability_form_bit(ConSanCapabilityForm::ClusterBarrier) |
                            consan_capability_form_bit(ConSanCapabilityForm::OrderedLdsAtomic)));
}

TEST(ConSanCapabilityContract, CapabilityDomainsMapEveryFormAndRejectSentinels) {
  constexpr std::array expected_domains = {
      ConSanCapabilityDomain::Access,  ConSanCapabilityDomain::Access,
      ConSanCapabilityDomain::Barrier, ConSanCapabilityDomain::Barrier,
      ConSanCapabilityDomain::Atomic,  ConSanCapabilityDomain::Atomic,
      ConSanCapabilityDomain::Atomic,  ConSanCapabilityDomain::Atomic,
      ConSanCapabilityDomain::Fence,
  };
  static_assert(expected_domains.size() == kConSanCapabilityForms.size());
  for (size_t index = 0; index < kConSanCapabilityForms.size(); ++index) {
    SCOPED_TRACE(index);
    EXPECT_EQ(consan_capability_domain(kConSanCapabilityForms[index]), expected_domains[index]);
  }
  EXPECT_EQ(consan_capability_domain(ConSanCapabilityForm::Count), ConSanCapabilityDomain::Count);
  EXPECT_EQ(consan_capability_domain(static_cast<ConSanCapabilityForm>(255)),
            ConSanCapabilityDomain::Count);
}

TEST(ConSanCapabilityContract, CapabilityNamesCoverEveryValueAndRejectSentinels) {
  constexpr std::array<std::string_view, 4> engine_names = {
      "SuperCollider",
      "Record/Replay",
      "Sampled",
      "Inline Shadow",
  };
  for (size_t index = 0; index < kConSanCapabilityEngines.size(); ++index) {
    SCOPED_TRACE(index);
    EXPECT_EQ(consan_capability_engine_name(kConSanCapabilityEngines[index]), engine_names[index]);
  }
  EXPECT_EQ(consan_capability_engine_name(ConSanCapabilityEngine::Count), "unknown");
  EXPECT_EQ(consan_capability_engine_name(static_cast<ConSanCapabilityEngine>(255)), "unknown");

  constexpr std::array<std::string_view, 9> form_names = {
      "native LDS",  "group FLAT",      "workgroup",
      "cluster",     "ordered FLAT",    "ordered VGLOBAL",
      "ordered LDS", "relaxed LDS RMW", "addressed ordinary",
  };
  for (size_t index = 0; index < kConSanCapabilityForms.size(); ++index) {
    SCOPED_TRACE(index);
    EXPECT_EQ(consan_capability_form_name(kConSanCapabilityForms[index]), form_names[index]);
  }
  EXPECT_EQ(consan_capability_form_name(ConSanCapabilityForm::Count), "unknown");
  EXPECT_EQ(consan_capability_form_name(static_cast<ConSanCapabilityForm>(255)), "unknown");

  constexpr std::array dispositions = {
      ConSanCapabilityDisposition::OutOfContract, ConSanCapabilityDisposition::NotApplicable,
      ConSanCapabilityDisposition::Supported,     ConSanCapabilityDisposition::MutationOnly,
      ConSanCapabilityDisposition::AccessOnly,    ConSanCapabilityDisposition::AssociatedOnly,
  };
  constexpr std::array<std::string_view, 6> disposition_names = {
      "out of contract", "not applicable", "supported",
      "mutation only",   "access only",    "associated only",
  };
  for (size_t index = 0; index < dispositions.size(); ++index) {
    SCOPED_TRACE(index);
    EXPECT_EQ(consan_capability_disposition_name(dispositions[index]), disposition_names[index]);
  }
  EXPECT_EQ(consan_capability_disposition_name(static_cast<ConSanCapabilityDisposition>(255)),
            "unknown");
}

TEST(ConSanCapabilityContract, CapabilityFormAvailabilityMatchesEveryTargetMask) {
  for (const ExpectedTargetProfile &expected : kExpectedTargetProfiles) {
    SCOPED_TRACE(rj_code_target_name(expected.target));
    for (ConSanCapabilityForm form : kConSanCapabilityForms) {
      const uint16_t bit = consan_capability_form_bit(form);
      EXPECT_EQ(consan_arch_supports_capability_form(expected.arch, form),
                (expected.semantic_form_mask & bit) != 0u);
    }
    EXPECT_FALSE(consan_arch_supports_capability_form(expected.arch, ConSanCapabilityForm::Count));
    EXPECT_FALSE(consan_arch_supports_capability_form(expected.arch,
                                                      static_cast<ConSanCapabilityForm>(255)));
  }
  for (ConSanCapabilityForm form : kConSanCapabilityForms)
    EXPECT_FALSE(consan_arch_supports_capability_form(ROCJITSU_CODE_ARCH_CDNA2, form));
}

TEST(ConSanCapabilityContract, CapabilityDispositionExhaustsTargetEngineAndFormContract) {
  /// Expected engine-specific dispositions after target-form availability has
  /// admitted a form. Target-specific `NotApplicable` is applied separately so
  /// this table remains an independent statement of engine semantics.
  struct ExpectedFormDisposition {
    ConSanCapabilityForm form;
    std::array<ConSanCapabilityDisposition, 4> by_engine;
  };
  constexpr auto supported = ConSanCapabilityDisposition::Supported;
  constexpr auto mutation = ConSanCapabilityDisposition::MutationOnly;
  constexpr auto not_applicable = ConSanCapabilityDisposition::NotApplicable;
  constexpr auto access_only = ConSanCapabilityDisposition::AccessOnly;
  constexpr auto associated_only = ConSanCapabilityDisposition::AssociatedOnly;
  constexpr std::array expected_forms = {
      ExpectedFormDisposition{ConSanCapabilityForm::NativeLdsAccess,
                              {supported, supported, supported, supported}},
      ExpectedFormDisposition{ConSanCapabilityForm::GroupFlatAccess,
                              {supported, supported, supported, supported}},
      ExpectedFormDisposition{ConSanCapabilityForm::WorkgroupBarrier,
                              {mutation, supported, supported, supported}},
      ExpectedFormDisposition{ConSanCapabilityForm::ClusterBarrier,
                              {mutation, supported, supported, supported}},
      ExpectedFormDisposition{ConSanCapabilityForm::OrderedFlatAtomic,
                              {mutation, supported, supported, supported}},
      ExpectedFormDisposition{ConSanCapabilityForm::OrderedVglobalAtomic,
                              {mutation, supported, supported, supported}},
      ExpectedFormDisposition{ConSanCapabilityForm::OrderedLdsAtomic,
                              {mutation, supported, supported, supported}},
      ExpectedFormDisposition{ConSanCapabilityForm::RelaxedLdsAtomicAccess,
                              {not_applicable, access_only, access_only, access_only}},
      ExpectedFormDisposition{ConSanCapabilityForm::AddressedOrdinaryFence,
                              {mutation, supported, associated_only, associated_only}},
  };
  static_assert(expected_forms.size() == kConSanCapabilityForms.size());

  for (const ExpectedTargetProfile &target : kExpectedTargetProfiles) {
    SCOPED_TRACE(rj_code_target_name(target.target));
    for (const ExpectedFormDisposition &form : expected_forms) {
      const bool available =
          (target.semantic_form_mask & consan_capability_form_bit(form.form)) != 0;
      for (size_t engine_index = 0; engine_index < kConSanCapabilityEngines.size();
           ++engine_index) {
        SCOPED_TRACE(consan_capability_engine_name(kConSanCapabilityEngines[engine_index]));
        SCOPED_TRACE(consan_capability_form_name(form.form));
        const ConSanCapabilityDisposition expected =
            available ? form.by_engine[engine_index] : not_applicable;
        EXPECT_EQ(consan_capability_disposition(target.target,
                                                kConSanCapabilityEngines[engine_index], form.form),
                  expected);
      }
    }
  }
}

TEST(ConSanCapabilityContract, CapabilityDispositionFailsClosedForInvalidTypedInputs) {
  constexpr rj_code_target_id_t valid_target = ROCJITSU_CODE_TARGET_GFX950;
  constexpr ConSanCapabilityEngine valid_engine = ConSanCapabilityEngine::RecordReplay;
  constexpr ConSanCapabilityForm valid_form = ConSanCapabilityForm::NativeLdsAccess;
  EXPECT_EQ(consan_capability_disposition(ROCJITSU_CODE_TARGET_INVALID, valid_engine, valid_form),
            ConSanCapabilityDisposition::OutOfContract);
  EXPECT_EQ(consan_capability_disposition(ROCJITSU_CODE_TARGET_GFX90A, valid_engine, valid_form),
            ConSanCapabilityDisposition::OutOfContract);
  EXPECT_EQ(consan_capability_disposition(valid_target, ConSanCapabilityEngine::Count, valid_form),
            ConSanCapabilityDisposition::OutOfContract);
  EXPECT_EQ(consan_capability_disposition(valid_target, static_cast<ConSanCapabilityEngine>(255),
                                          valid_form),
            ConSanCapabilityDisposition::OutOfContract);
  EXPECT_EQ(consan_capability_disposition(valid_target, valid_engine, ConSanCapabilityForm::Count),
            ConSanCapabilityDisposition::OutOfContract);
  EXPECT_EQ(consan_capability_disposition(valid_target, valid_engine,
                                          static_cast<ConSanCapabilityForm>(255)),
            ConSanCapabilityDisposition::OutOfContract);
}

} // namespace
} // namespace rocjitsu
