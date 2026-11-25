// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/fault_injection/lds_wrong_address.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/opcodes.h"
#include "target_coverage_fixture.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace rocjitsu {
namespace {

std::vector<uint8_t> truncated_gfx1100_elf(const test::TargetLdsCase &fixture) {
  std::vector<uint8_t> image =
      test::make_amdgpu_kernel_elf({fixture.form.words[0]}, 0, 3, fixture.form.machine, 3, 3, false,
                                   false, false, false, fixture.wave == 32);
  image[EI_DATA] = ELFDATA2LSB;
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  return image;
}

class Gfx1100Lds : public ::testing::TestWithParam<test::TargetLdsCase> {};

TEST_P(Gfx1100Lds, SourceOrderExactAndOrdinalPreserveIndependentGuestAndWholeImage) {
  const test::TargetLdsCase &fixture = GetParam();
  constexpr std::array<uint32_t, 1> prefix = {0xbf800001u};
  const std::array<uint32_t, 3> suffix = {0xbf800003u, fixture.form.words[0],
                                          fixture.form.words[1]};
  const std::vector<uint8_t> image = test::make_target_lds_elf(fixture, prefix, suffix);
  const FaultMutationContext source(image);
  FaultMutationError error;
  const FailureOr<LdsInventory> inventory =
      discover_lds_wrong_address(source, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().sites().size(), 2u);
  const uint32_t allocation = test::target_lds_allocation(fixture);
  EXPECT_EQ(allocation, fixture.wave == 32 ? 32u : 16u);
  for (size_t ordinal = 0; ordinal < 2; ++ordinal) {
    SCOPED_TRACE(ordinal);
    const uint64_t offset = ordinal == 0 ? 4u : 16u;
    const LdsSite &site = inventory.value().sites()[ordinal];
    EXPECT_EQ(site.offset, offset);
    EXPECT_EQ(site.form, fixture.form.form);
    EXPECT_EQ(site.address_vgpr, 2u);
    EXPECT_EQ(site.address_bank, 0u);
    ASSERT_EQ(site.affected_owners.size(), 1u);
    EXPECT_EQ(site.affected_owners[0].entry, "test_kernel");
    const FailureOr<LdsSelection> selected = select_lds_wrong_address(
        inventory.value(), {.exact = std::nullopt, .ordinal = ordinal}, error);
    ASSERT_TRUE(selected.succeeded()) << error.message;
    const FailureOr<LdsSelection> exact = select_lds_wrong_address(
        inventory.value(),
        {.exact = selected.value(), .ordinal = std::numeric_limits<size_t>::max()}, error);
    ASSERT_TRUE(exact.succeeded()) << error.message;
    for (uint32_t address : {6u, allocation - 1}) {
      const FailureOr<LdsResult> result =
          apply_lds_wrong_address(source, exact.value(), address, error);
      ASSERT_TRUE(result.succeeded()) << error.message;
      EXPECT_EQ(result.value().logical_mutation_count, 1u);
      EXPECT_EQ(result.value().selection.site().offset, offset);
      EXPECT_TRUE(
          test::target_lds_output_matches(source, fixture, offset, address, result.value().image));
      const FailureOr<LdsResult> repeated =
          apply_lds_wrong_address(source, selected.value(), address, error);
      ASSERT_TRUE(repeated.succeeded()) << error.message;
      EXPECT_EQ(repeated.value().image, result.value().image);
    }
  }
  EXPECT_TRUE(std::ranges::equal(source.image(), image));
}

TEST_P(Gfx1100Lds, InvalidParametersAndForeignOrExpiredSelectionsDoNotChangeSnapshot) {
  const test::TargetLdsCase &fixture = GetParam();
  std::vector<uint8_t> input = test::make_target_lds_elf(fixture);
  const std::vector<uint8_t> pristine = input;
  const FaultMutationContext source(input), foreign(input);
  const Section &text = *source.source().text_sections().front();
  input[text.sectionOffset() + 4] = 9;
  FaultMutationError error;
  const FailureOr<LdsInventory> inventory =
      discover_lds_wrong_address(source, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().sites().size(), 1u);
  EXPECT_EQ(inventory.value().sites()[0].address_vgpr, 2u);
  const FailureOr<LdsSelection> selected = select_lds_wrong_address(inventory.value(), {}, error);
  ASSERT_TRUE(selected.succeeded()) << error.message;
  const uint32_t allocation = test::target_lds_allocation(fixture);
  for (const std::optional<uint32_t> address : std::array<std::optional<uint32_t>, 6>{
           std::nullopt, 2u, allocation, 255u, 256u, UINT32_MAX}) {
    EXPECT_TRUE(apply_lds_wrong_address(source, selected.value(), address, error).failed());
    EXPECT_EQ(error.code, FaultMutationErrorCode::InvalidParameters);
  }
  EXPECT_TRUE(apply_lds_wrong_address(foreign, selected.value(), 6, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  EXPECT_TRUE(
      select_lds_wrong_address(inventory.value(), {.exact = LdsSelection{}, .ordinal = 0}, error)
          .failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  EXPECT_TRUE(
      select_lds_wrong_address(inventory.value(), {.exact = std::nullopt, .ordinal = 1}, error)
          .failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::OrdinalOutOfRange);
  const FailureOr<LdsInventory> foreign_inventory =
      discover_lds_wrong_address(foreign, "test_kernel", error);
  ASSERT_TRUE(foreign_inventory.succeeded()) << error.message;
  EXPECT_TRUE(select_lds_wrong_address(foreign_inventory.value(),
                                       {.exact = selected.value(), .ordinal = 0}, error)
                  .failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  std::optional<LdsInventory> expired_inventory;
  std::optional<LdsSelection> expired_selection;
  {
    const FaultMutationContext temporary(pristine);
    const FailureOr<LdsInventory> found =
        discover_lds_wrong_address(temporary, "test_kernel", error);
    ASSERT_TRUE(found.succeeded()) << error.message;
    expired_inventory = found.value();
    const FailureOr<LdsSelection> one = select_lds_wrong_address(*expired_inventory, {}, error);
    ASSERT_TRUE(one.succeeded()) << error.message;
    expired_selection = one.value();
  }
  EXPECT_TRUE(select_lds_wrong_address(*expired_inventory, {}, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  EXPECT_TRUE(apply_lds_wrong_address(source, *expired_selection, 6, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  const FailureOr<LdsResult> result = apply_lds_wrong_address(source, selected.value(), 6, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  EXPECT_TRUE(test::target_lds_output_matches(source, fixture, 0, 6, result.value().image));
  EXPECT_TRUE(std::ranges::equal(source.image(), pristine));
}

TEST_P(Gfx1100Lds, GdsBitSeventeenIsExcludedAndReservedBitSixteenIsPreserved) {
  const test::TargetLdsCase &fixture = GetParam();
  FaultMutationError error;
  test::TargetLdsCase gds = fixture;
  gds.form.words[0] |= 0x00020000u;
  gds.expected[0] |= 0x00020000u;
  const FaultMutationContext excluded(test::make_target_lds_elf(gds));
  const FailureOr<LdsInventory> gds_inventory =
      discover_lds_wrong_address(excluded, "test_kernel", error);
  ASSERT_TRUE(gds_inventory.succeeded()) << error.message;
  EXPECT_TRUE(gds_inventory.value().sites().empty());
  EXPECT_TRUE(select_lds_wrong_address(gds_inventory.value(), {}, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::NoCandidates);
  test::TargetLdsCase reserved = fixture;
  reserved.form.words[0] |= 0x00010000u;
  reserved.expected[0] |= 0x00010000u;
  const FaultMutationContext source(test::make_target_lds_elf(reserved));
  const FailureOr<LdsInventory> inventory =
      discover_lds_wrong_address(source, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().sites().size(), 1u);
  const FailureOr<LdsSelection> selected = select_lds_wrong_address(inventory.value(), {}, error);
  ASSERT_TRUE(selected.succeeded()) << error.message;
  const FailureOr<LdsResult> result = apply_lds_wrong_address(source, selected.value(), 6, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  EXPECT_TRUE(test::target_lds_output_matches(source, reserved, 0, 6, result.value().image));
}

TEST_P(Gfx1100Lds, IncompleteSourceExtentFailsBeforeMutation) {
  const std::vector<uint8_t> image = truncated_gfx1100_elf(GetParam());
  const FaultMutationContext source(image);
  FaultMutationError error;
  EXPECT_TRUE(discover_lds_wrong_address(source, "test_kernel", error).failed());
  EXPECT_NE(error.code, FaultMutationErrorCode::None);
  EXPECT_TRUE(std::ranges::equal(source.image(), image));
}

TEST_P(Gfx1100Lds, IndependentOracleRejectsGuestContinuationUnselectedAndMetadataCorruption) {
  const test::TargetLdsCase &fixture = GetParam();
  const FaultMutationContext source(test::make_target_lds_elf(fixture));
  FaultMutationError error;
  const FailureOr<LdsInventory> inventory =
      discover_lds_wrong_address(source, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const FailureOr<LdsSelection> selection = select_lds_wrong_address(inventory.value(), {}, error);
  ASSERT_TRUE(selection.succeeded()) << error.message;
  const FailureOr<LdsResult> result = apply_lds_wrong_address(source, selection.value(), 6, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  ASSERT_TRUE(test::target_lds_output_matches(source, fixture, 0, 6, result.value().image));
  const AmdGpuCodeObject output(result.value().image.data(), result.value().image.size());
  ASSERT_TRUE(output.is_valid());
  ASSERT_EQ(output.text_sections().size(), 1u);
  const Section &text = *output.text_sections().front();
  const std::optional<uint64_t> guest = test::lds_emitted_guest(text, 0, fixture.form.target);
  ASSERT_TRUE(guest);
  for (uint64_t offset : {*guest, *guest + 5, *guest + 7, *guest + 8, uint64_t{8}}) {
    std::vector<uint8_t> corrupted = result.value().image;
    ASSERT_LT(text.sectionOffset() + offset, corrupted.size());
    corrupted[text.sectionOffset() + offset] ^= 1;
    EXPECT_FALSE(test::target_lds_output_matches(source, fixture, 0, 6, corrupted));
  }
  const std::vector<KernelDescriptorInfo> descriptors = scan_kernel_descriptors(
      result.value().image, text.sectionOffset(), text.size(), text.sectionHeaderIndex());
  ASSERT_EQ(descriptors.size(), 1u);
  std::vector<uint8_t> corrupted = result.value().image;
  const uint64_t private_size_offset =
      descriptors[0].descriptor_file_offset +
      offsetof(amdhsa::kernel_descriptor_t, private_segment_fixed_size);
  ASSERT_LT(private_size_offset, corrupted.size());
  corrupted[private_size_offset] ^= 1;
  EXPECT_FALSE(test::target_lds_output_matches(source, fixture, 0, 6, corrupted));
  corrupted = result.value().image;
  corrupted[0] ^= 1;
  EXPECT_FALSE(test::target_lds_output_matches(source, fixture, 0, 6, corrupted));
}

TEST(Gfx1100LdsControls, NopAddtidAndAtomicFormsNeverBecomeAddressFaultSites) {
  constexpr std::array<uint32_t, 8> prefix = {
      0xd8500000u, 0u,          // ds_nop
      0xdac00000u, 0x00000800u, // ds_store_addtid_b32 v8
      0xdac40000u, 0x0c000000u, // ds_load_addtid_b32 v12
      0xd8000004u, 0x00000802u, // ds_add_u32 v2, v8 offset:4
  };
  std::unique_ptr<Decoder> decoder =
      Decoder::create(default_isa_target_registry(), ROCJITSU_CODE_TARGET_GFX1100);
  ASSERT_TRUE(decoder);
  constexpr std::array<std::string_view, 4> forms = {"ds_nop", "ds_store_addtid_b32",
                                                     "ds_load_addtid_b32", "ds_add_u32"};
  for (size_t index = 0; index < forms.size(); ++index) {
    const DecodeResult decoded = decoder->decode_window(std::span(prefix).subspan(index * 2, 2));
    ASSERT_TRUE(decoded.succeeded());
    EXPECT_EQ(decoded.value()->mnemonic(), forms[index]);
  }
  const std::vector<test::TargetLdsCase> cases =
      test::target_lds_cases(ROCJITSU_CODE_TARGET_GFX1100);
  for (uint32_t wave : {32u, 64u}) {
    const std::vector<test::TargetLdsCase>::const_iterator fixture =
        std::ranges::find(cases, wave, &test::TargetLdsCase::wave);
    ASSERT_NE(fixture, cases.end());
    const FaultMutationContext source(test::make_target_lds_elf(*fixture, prefix));
    FaultMutationError error;
    const FailureOr<LdsInventory> inventory =
        discover_lds_wrong_address(source, "test_kernel", error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    ASSERT_EQ(inventory.value().sites().size(), 1u);
    EXPECT_EQ(inventory.value().sites()[0].offset, sizeof(prefix));
    const FailureOr<LdsSelection> selection =
        select_lds_wrong_address(inventory.value(), {}, error);
    ASSERT_TRUE(selection.succeeded()) << error.message;
    const FailureOr<LdsResult> result =
        apply_lds_wrong_address(source, selection.value(), 6, error);
    ASSERT_TRUE(result.succeeded()) << error.message;
    EXPECT_TRUE(
        test::target_lds_output_matches(source, *fixture, sizeof(prefix), 6, result.value().image));
  }
}

TEST(Gfx1100LdsControls, MissingConcreteTransposeFormsFailDiscoveryWithoutChangingInput) {
  const std::vector<test::TargetLdsCase> cases =
      test::target_lds_cases(ROCJITSU_CODE_TARGET_GFX1100);
  for (uint32_t wave : {32u, 64u}) {
    const std::vector<test::TargetLdsCase>::const_iterator fixture =
        std::ranges::find(cases, wave, &test::TargetLdsCase::wave);
    ASSERT_NE(fixture, cases.end());
    for (uint32_t opcode_word : {0xdb8c0004u, 0xdbe80004u, 0xdbec0004u, 0xdbf00004u, 0xdbf40004u}) {
      test::TargetLdsCase unsupported = *fixture;
      unsupported.form.words = {opcode_word, 0x0c000002u};
      const std::vector<uint8_t> image = test::make_target_lds_elf(unsupported);
      const FaultMutationContext source(image);
      FaultMutationError error;
      EXPECT_TRUE(discover_lds_wrong_address(source, "test_kernel", error).failed());
      EXPECT_NE(error.code, FaultMutationErrorCode::None);
      EXPECT_TRUE(std::ranges::equal(source.image(), image));
    }
  }
}

TEST(Gfx1100LdsControls, RetainedPcAndTrampolineRangeRejectionsLeaveSeedReusable) {
  const std::vector<test::TargetLdsCase> cases =
      test::target_lds_cases(ROCJITSU_CODE_TARGET_GFX1100);
  constexpr std::array<uint32_t, 1> get_pc = rdna3::build_sop1(rdna3::kSGetPcB64Sop1, {.sdst = 0});
  const std::vector<uint32_t> distant_suffix(32769, 0xbf800000u);
  for (uint32_t wave : {32u, 64u}) {
    const std::vector<test::TargetLdsCase>::const_iterator fixture =
        std::ranges::find(cases, wave, &test::TargetLdsCase::wave);
    ASSERT_NE(fixture, cases.end());
    for (bool pc_relative : {true, false}) {
      const std::vector<uint8_t> image =
          pc_relative ? test::make_target_lds_elf(*fixture, get_pc)
                      : test::make_target_lds_elf(*fixture, {}, distant_suffix);
      const FaultMutationContext source(image);
      FaultMutationError error;
      const FailureOr<LdsInventory> inventory =
          discover_lds_wrong_address(source, "test_kernel", error);
      ASSERT_TRUE(inventory.succeeded()) << error.message;
      ASSERT_EQ(inventory.value().sites().size(), 1u);
      const FailureOr<LdsSelection> selection =
          select_lds_wrong_address(inventory.value(), {}, error);
      ASSERT_TRUE(selection.succeeded()) << error.message;
      for (int attempt = 0; attempt < 2; ++attempt) {
        EXPECT_TRUE(apply_lds_wrong_address(source, selection.value(), 6, error).failed());
        EXPECT_NE(error.code, FaultMutationErrorCode::None);
        if (pc_relative) {
          EXPECT_NE(error.message.find("retained PC-relative"), std::string::npos);
        }
        EXPECT_TRUE(std::ranges::equal(source.image(), image));
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(ConcreteFormsAndWaves, Gfx1100Lds,
                         ::testing::ValuesIn(test::target_lds_cases(ROCJITSU_CODE_TARGET_GFX1100)),
                         [](const ::testing::TestParamInfo<test::TargetLdsCase> &info) {
                           return std::string(info.param.form.form) + "Wave" +
                                  std::to_string(info.param.wave);
                         });

} // namespace
} // namespace rocjitsu
