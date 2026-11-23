// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/fault_injection/lds_wrong_address.h"
#include "target_coverage_fixture.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace {

std::vector<test::TargetLdsCase> owner_cases() {
  std::vector<test::TargetLdsCase> cases;
  for (rj_code_target_id_t target : {ROCJITSU_CODE_TARGET_GFX1100, ROCJITSU_CODE_TARGET_GFX1251}) {
    for (test::TargetLdsCase fixture : test::target_lds_cases(target)) {
      if (fixture.form.form != "ds_store_b32")
        continue;
      // Assembled for both targets: ds_store_b32 v2, v3 offset:260.
      // v3 also fits the smallest Wave64 descriptor used below.
      fixture.form.words = {0xd8340104u, 0x00000302u};
      fixture.expected = {0xd8340104u, 0x00000306u};
      cases.push_back(fixture);
    }
  }
  return cases;
}

TEST(LdsTargetCoverage, CapabilitiesMatchSevenTargetsAndTenWaveProfiles) {
  using TargetWave = std::pair<rj_code_target_id_t, uint32_t>;
  const std::map<TargetWave, size_t> expected = {
      {{ROCJITSU_CODE_TARGET_GFX1100, 32}, 30}, {{ROCJITSU_CODE_TARGET_GFX1100, 64}, 30},
      {{ROCJITSU_CODE_TARGET_GFX1200, 32}, 30}, {{ROCJITSU_CODE_TARGET_GFX1200, 64}, 30},
      {{ROCJITSU_CODE_TARGET_GFX1201, 32}, 30}, {{ROCJITSU_CODE_TARGET_GFX1201, 64}, 30},
      {{ROCJITSU_CODE_TARGET_GFX942, 64}, 30},  {{ROCJITSU_CODE_TARGET_GFX950, 64}, 31},
      {{ROCJITSU_CODE_TARGET_GFX1250, 32}, 24}, {{ROCJITSU_CODE_TARGET_GFX1251, 32}, 34},
  };
  std::map<TargetWave, size_t> actual;
  std::set<std::tuple<rj_code_target_id_t, uint32_t, std::string_view>> identities;
  const std::vector<LdsCapability> capabilities = lds_wrong_address_capabilities();
  EXPECT_EQ(capabilities.size(), 299u);
  for (const LdsCapability &capability : capabilities) {
    ++actual[{capability.target, capability.wave_size}];
    EXPECT_TRUE(
        identities.emplace(capability.target, capability.wave_size, capability.form).second);
  }
  EXPECT_EQ(actual, expected);
}

class LdsTargetOwners : public testing::TestWithParam<test::TargetLdsCase> {};

TEST_P(LdsTargetOwners, DirectSharedSiteRetainsBothOwnersAndExactRebinding) {
  const test::TargetLdsCase &fixture = GetParam();
  const std::vector<uint8_t> image = test::make_target_shared_lds_elf(fixture);
  const FaultMutationContext source(image);
  FaultMutationError error;
  std::optional<LdsSelection> first;
  std::vector<uint8_t> first_output;
  for (std::string_view owner : {"test_kernel0", "test_kernel1"}) {
    SCOPED_TRACE(owner);
    const FailureOr<LdsInventory> inventory = discover_lds_wrong_address(source, owner, error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    ASSERT_EQ(inventory.value().sites().size(), 1u);
    const LdsSite &site = inventory.value().sites().front();
    EXPECT_EQ(site.offset, 512u);
    EXPECT_EQ(site.address_bank, 0u);
    EXPECT_EQ(site.form, fixture.form.form);
    ASSERT_EQ(site.affected_owners.size(), 2u);
    EXPECT_EQ(site.affected_owners[0].entry, "test_kernel0");
    EXPECT_EQ(site.affected_owners[1].entry, "test_kernel1");
    const FailureOr<LdsSelection> selected = select_lds_wrong_address(
        inventory.value(),
        {.exact = first, .ordinal = first ? std::numeric_limits<size_t>::max() : 0}, error);
    ASSERT_TRUE(selected.succeeded()) << error.message;
    ASSERT_NE(selected.value().owner(), nullptr);
    EXPECT_EQ(selected.value().owner()->entry, owner);
    const FailureOr<LdsResult> result = apply_lds_wrong_address(source, selected.value(), 6, error);
    ASSERT_TRUE(result.succeeded()) << error.message;
    EXPECT_EQ(result.value().logical_mutation_count, 1u);
    EXPECT_TRUE(test::target_lds_output_matches(source, fixture, 512, 6, result.value().image));
    if (first) {
      EXPECT_EQ(result.value().image, first_output);
    } else {
      first = selected.value();
      first_output = result.value().image;
    }
  }
  EXPECT_TRUE(std::ranges::equal(source.image(), image));
}

TEST_P(LdsTargetOwners, EveryOwnerBoundsTheReplacementAllocation) {
  const test::TargetLdsCase &fixture = GetParam();
  const std::vector<uint8_t> image = test::make_target_shared_lds_elf(fixture, {3, 0});
  const FaultMutationContext source(image);
  const uint32_t bound = test::target_lds_allocation(fixture, 0);
  for (std::string_view owner : {"test_kernel0", "test_kernel1"}) {
    FaultMutationError error;
    const FailureOr<LdsInventory> inventory = discover_lds_wrong_address(source, owner, error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    ASSERT_EQ(inventory.value().sites().size(), 1u);
    ASSERT_EQ(inventory.value().sites().front().affected_owners.size(), 2u);
    const FailureOr<LdsSelection> selected = select_lds_wrong_address(inventory.value(), {}, error);
    ASSERT_TRUE(selected.succeeded()) << error.message;
    const FailureOr<LdsResult> last =
        apply_lds_wrong_address(source, selected.value(), bound - 1, error);
    ASSERT_TRUE(last.succeeded()) << error.message;
    EXPECT_TRUE(
        test::target_lds_output_matches(source, fixture, 512, bound - 1, last.value().image));
    EXPECT_TRUE(apply_lds_wrong_address(source, selected.value(), bound, error).failed());
    EXPECT_EQ(error.code, FaultMutationErrorCode::InvalidParameters);
  }
  EXPECT_TRUE(std::ranges::equal(source.image(), image));
}

TEST(LdsTargetCoverage, Rdna3MixedWaveOwnersUseEachDescriptorGranule) {
  const test::TargetLdsCase fixture = owner_cases().front();
  const std::vector<uint8_t> image =
      test::make_target_shared_lds_elf(fixture, {0, 0}, {}, {}, {32, 64});
  const FaultMutationContext source(image);
  for (std::string_view owner : {"test_kernel0", "test_kernel1"}) {
    FaultMutationError error;
    const FailureOr<LdsInventory> inventory = discover_lds_wrong_address(source, owner, error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    ASSERT_EQ(inventory.value().sites().size(), 1u);
    const FailureOr<LdsSelection> selected = select_lds_wrong_address(inventory.value(), {}, error);
    ASSERT_TRUE(selected.succeeded()) << error.message;
    const FailureOr<LdsResult> result = apply_lds_wrong_address(source, selected.value(), 3, error);
    ASSERT_TRUE(result.succeeded()) << error.message;
    EXPECT_TRUE(test::target_lds_output_matches(source, fixture, 512, 3, result.value().image));
    EXPECT_TRUE(apply_lds_wrong_address(source, selected.value(), 4, error).failed());
    EXPECT_EQ(error.code, FaultMutationErrorCode::InvalidParameters);
  }
}

TEST(LdsTargetCoverage, Cdna5SharedBanksMustAgreeAndFitEveryDescriptor) {
  const test::TargetLdsCase fixture = owner_cases().back();
  const std::array<uint32_t, 1> bank_one = {
      cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 1})[0]};
  const std::array<uint32_t, 1> bank_zero = {
      cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 0})[0]};
  // Ordinary gfx1251 MODE slice write to Src0 from an unknown SGPR.
  const std::array<uint32_t, 1> unknown = {cdna5::build_sopk(
      cdna5::kSSetregB32Sopk, {.simm16 = 1u | (14u << 6) | (1u << 11), .sdst = 0})[0]};
  for (std::span<const uint32_t> second :
       {std::span<const uint32_t>(bank_one), std::span<const uint32_t>(bank_zero),
        std::span<const uint32_t>(unknown)}) {
    const std::vector<uint8_t> image =
        test::make_target_shared_lds_elf(fixture, {31, 16}, bank_one, second);
    const FaultMutationContext source(image);
    for (std::string_view owner : {"test_kernel0", "test_kernel1"}) {
      FaultMutationError error;
      const FailureOr<LdsInventory> inventory = discover_lds_wrong_address(source, owner, error);
      ASSERT_TRUE(inventory.succeeded()) << error.message;
      if (second.data() != bank_one.data()) {
        EXPECT_TRUE(inventory.value().sites().empty());
        continue;
      }
      ASSERT_EQ(inventory.value().sites().size(), 1u);
      EXPECT_EQ(inventory.value().sites().front().address_bank, 1u);
      const FailureOr<LdsSelection> selected =
          select_lds_wrong_address(inventory.value(), {}, error);
      ASSERT_TRUE(selected.succeeded()) << error.message;
      const FailureOr<LdsResult> result =
          apply_lds_wrong_address(source, selected.value(), 6, error);
      ASSERT_TRUE(result.succeeded()) << error.message;
      EXPECT_TRUE(test::target_lds_output_matches(source, fixture, 512, 6, result.value().image));
      const FailureOr<LdsResult> last =
          apply_lds_wrong_address(source, selected.value(), 15, error);
      ASSERT_TRUE(last.succeeded()) << error.message;
      EXPECT_TRUE(test::target_lds_output_matches(source, fixture, 512, 15, last.value().image));
      EXPECT_TRUE(apply_lds_wrong_address(source, selected.value(), 16, error).failed());
      EXPECT_EQ(error.code, FaultMutationErrorCode::InvalidParameters);
    }
    EXPECT_TRUE(std::ranges::equal(source.image(), image));
  }
}

TEST(LdsTargetCoverage, Gfx1251OtherOwnerModeSliceMustAgree) {
  const test::TargetLdsCase fixture = owner_cases().back();
  const std::array<uint32_t, 1> bank_zero = {
      cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 0})[0]};
  const std::array<uint32_t, 2> bank_two = {
      cdna5::build_sopk(cdna5::kSSetregImm32B32Sopk, {.simm16 = 1u | (14u << 6) | (1u << 11)})[0],
      2};
  const std::vector<uint8_t> image =
      test::make_target_shared_lds_elf(fixture, {31, 32}, bank_zero, bank_two);
  const FaultMutationContext source(image);
  for (std::string_view owner : {"test_kernel0", "test_kernel1"}) {
    SCOPED_TRACE(owner);
    FaultMutationError error;
    const FailureOr<LdsInventory> inventory = discover_lds_wrong_address(source, owner, error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    EXPECT_TRUE(inventory.value().sites().empty());
  }
  EXPECT_TRUE(std::ranges::equal(source.image(), image));
}

TEST(LdsTargetCoverage, Gfx1251OtherOwnerModeBitZeroPreservesBank) {
  const test::TargetLdsCase fixture = owner_cases().back();
  const std::array<uint32_t, 1> bank_one = {
      cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 1})[0]};
  const std::array<uint32_t, 3> second = {
      bank_one[0], cdna5::build_sopk(cdna5::kSSetregImm32B32Sopk, {.simm16 = 1})[0], 1};
  const std::vector<uint8_t> image =
      test::make_target_shared_lds_elf(fixture, {31, 16}, bank_one, second);
  const FaultMutationContext source(image);
  for (std::string_view owner : {"test_kernel0", "test_kernel1"}) {
    SCOPED_TRACE(owner);
    FaultMutationError error;
    const FailureOr<LdsInventory> inventory = discover_lds_wrong_address(source, owner, error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    ASSERT_EQ(inventory.value().sites().size(), 1u);
    EXPECT_EQ(inventory.value().sites().front().address_bank, 1u);
    ASSERT_EQ(inventory.value().sites().front().affected_owners.size(), 2u);
    const FailureOr<LdsSelection> selected = select_lds_wrong_address(inventory.value(), {}, error);
    ASSERT_TRUE(selected.succeeded()) << error.message;
    const FailureOr<LdsResult> result =
        apply_lds_wrong_address(source, selected.value(), 15, error);
    ASSERT_TRUE(result.succeeded()) << error.message;
    EXPECT_TRUE(test::target_lds_output_matches(source, fixture, 512, 15, result.value().image));
    EXPECT_TRUE(apply_lds_wrong_address(source, selected.value(), 16, error).failed());
  }
  EXPECT_TRUE(std::ranges::equal(source.image(), image));
}

TEST(LdsTargetCoverage, Cdna5DynamicVgprOwnersAreNotListed) {
  for (rj_code_target_id_t target : {ROCJITSU_CODE_TARGET_GFX1250, ROCJITSU_CODE_TARGET_GFX1251}) {
    SCOPED_TRACE(static_cast<int>(target));
    test::TargetLdsCase fixture = owner_cases().back();
    fixture.form.target = target;
    fixture.form.machine = target == ROCJITSU_CODE_TARGET_GFX1250 ? EF_AMDGPU_MACH_AMDGCN_GFX1250
                                                                  : EF_AMDGPU_MACH_AMDGCN_GFX1251;
    std::vector<uint8_t> image =
        test::make_target_lds_elf(fixture, test::kLdsDynamicVgprPrefix, {}, 2);
    ASSERT_NO_FATAL_FAILURE(test::enable_target_lds_dynamic_vgprs(image));
    const FaultMutationContext source(image);
    FaultMutationError error;
    const FailureOr<LdsInventory> inventory =
        discover_lds_wrong_address(source, "test_kernel", error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    EXPECT_TRUE(inventory.value().sites().empty());
    EXPECT_TRUE(select_lds_wrong_address(inventory.value(), {}, error).failed());
    EXPECT_TRUE(std::ranges::equal(source.image(), image));
  }
}

TEST(LdsTargetCoverage, Cdna5SharedDynamicVgprOwnerIsNotListed) {
  const test::TargetLdsCase fixture = owner_cases().back();
  for (size_t dynamic_owner : {0u, 1u}) {
    SCOPED_TRACE(dynamic_owner);
    std::vector<uint8_t> image = test::make_target_shared_lds_elf(fixture, {2, 2});
    ASSERT_NO_FATAL_FAILURE(test::enable_target_lds_dynamic_vgprs(image, dynamic_owner));
    const FaultMutationContext source(image);
    for (std::string_view owner : {"test_kernel0", "test_kernel1"}) {
      SCOPED_TRACE(owner);
      FaultMutationError error;
      const FailureOr<LdsInventory> inventory = discover_lds_wrong_address(source, owner, error);
      ASSERT_TRUE(inventory.succeeded()) << error.message;
      EXPECT_TRUE(inventory.value().sites().empty());
      EXPECT_TRUE(select_lds_wrong_address(inventory.value(), {}, error).failed());
    }
    EXPECT_TRUE(std::ranges::equal(source.image(), image));
  }
}

TEST(LdsTargetCoverage, Gfx1251UnrelatedDynamicOwnerDoesNotExcludeStaticSite) {
  const test::TargetLdsCase fixture = owner_cases().back();
  std::vector<uint8_t> image = test::make_target_shared_lds_elf(fixture, {2, 2});
  ASSERT_NO_FATAL_FAILURE(test::enable_target_lds_dynamic_vgprs(image, 1));
  const AmdGpuCodeObject original(image.data(), image.size());
  const Section &text = *original.text_sections().front();
  const uint32_t end = build_s_endpgm(IsaFamily::Cdna5).value();
  // Owner 1 exits without calling the helper, so it no longer owns the store.
  std::memcpy(image.data() + text.sectionOffset() + 256, &end, sizeof(end));
  const FaultMutationContext source(image);
  FaultMutationError error;
  const FailureOr<LdsInventory> inventory =
      discover_lds_wrong_address(source, "test_kernel0", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().sites().size(), 1u);
  ASSERT_EQ(inventory.value().sites().front().affected_owners.size(), 1u);
  EXPECT_EQ(inventory.value().sites().front().affected_owners.front().entry, "test_kernel0");
  const FailureOr<LdsSelection> selected = select_lds_wrong_address(inventory.value(), {}, error);
  ASSERT_TRUE(selected.succeeded()) << error.message;
  const FailureOr<LdsResult> result = apply_lds_wrong_address(source, selected.value(), 40, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  EXPECT_TRUE(test::target_lds_output_matches(source, fixture, 512, 40, result.value().image));
  EXPECT_TRUE(std::ranges::equal(source.image(), image));
}

TEST(LdsTargetCoverage, Cdna5LegacyDescriptorWaveBitAlwaysUsesWave32Allocation) {
  for (rj_code_target_id_t target : {ROCJITSU_CODE_TARGET_GFX1250, ROCJITSU_CODE_TARGET_GFX1251}) {
    test::TargetLdsCase fixture = owner_cases().back();
    fixture.form.target = target;
    fixture.form.machine = target == ROCJITSU_CODE_TARGET_GFX1250 ? EF_AMDGPU_MACH_AMDGCN_GFX1250
                                                                  : EF_AMDGPU_MACH_AMDGCN_GFX1251;
    for (bool legacy_wave_bit : {false, true}) {
      SCOPED_TRACE(static_cast<int>(target));
      SCOPED_TRACE(legacy_wave_bit);
      std::vector<uint8_t> image = test::make_target_lds_elf(fixture);
      const AmdGpuCodeObject source_object(image.data(), image.size());
      const Section &text = *source_object.text_sections().front();
      const std::vector<KernelDescriptorInfo> kernels = scan_kernel_descriptors(
          image, text.sectionOffset(), text.size(), text.sectionHeaderIndex());
      ASSERT_EQ(kernels.size(), 1u);
      amdhsa::kernel_descriptor_t descriptor = kernels.front().descriptor;
      RJ_AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                         amdhsa::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32, legacy_wave_bit);
      std::memcpy(image.data() + kernels.front().descriptor_file_offset, &descriptor,
                  sizeof(descriptor));
      EXPECT_EQ(kernel_wavefront_size(IsaFamily::Cdna5, descriptor), 32u);
      const FaultMutationContext source(image);
      FaultMutationError error;
      const FailureOr<LdsInventory> inventory =
          discover_lds_wrong_address(source, "test_kernel", error);
      ASSERT_TRUE(inventory.succeeded()) << error.message;
      ASSERT_EQ(inventory.value().sites().size(), 1u);
      const FailureOr<LdsSelection> selected =
          select_lds_wrong_address(inventory.value(), {}, error);
      ASSERT_TRUE(selected.succeeded()) << error.message;
      const FailureOr<LdsResult> result =
          apply_lds_wrong_address(source, selected.value(), 63, error);
      ASSERT_TRUE(result.succeeded()) << error.message;
      EXPECT_TRUE(test::target_lds_output_matches(source, fixture, 0, 63, result.value().image));
      EXPECT_TRUE(apply_lds_wrong_address(source, selected.value(), 64, error).failed());
    }
  }
}

INSTANTIATE_TEST_SUITE_P(NewTargetWaves, LdsTargetOwners, testing::ValuesIn(owner_cases()),
                         [](const testing::TestParamInfo<test::TargetLdsCase> &info) {
                           return "T" + std::to_string(static_cast<int>(info.param.form.target)) +
                                  "Wave" + std::to_string(info.param.wave);
                         });

} // namespace
} // namespace rocjitsu
