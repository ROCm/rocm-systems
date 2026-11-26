// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cdna_lds_fixture.h"
#include "rocjitsu/code/patch/fault_injection/lds_wrong_address.h"

#include <gtest/gtest.h>

namespace rocjitsu {
namespace {
class CdnaLds : public ::testing::TestWithParam<test::CdnaLdsTarget> {};

TEST_P(CdnaLds, BothFormsAndAccumulatorPreserveCompleteOutput) {
  for (bool acc : {false, true}) {
    SCOPED_TRACE(acc);
    const std::vector<uint8_t> image = test::make_cdna_lds_elf(GetParam(), acc);
    const FaultMutationContext source(image);
    FaultMutationError error;
    const FailureOr<LdsInventory> inventory =
        discover_lds_wrong_address(source, "test_kernel", error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    ASSERT_EQ(inventory.value().sites().size(), 2u);
    for (size_t ordinal = 0; ordinal < 2; ++ordinal) {
      EXPECT_EQ(inventory.value().sites()[ordinal].form,
                ordinal == 0 ? "ds_write_b32" : "ds_write_b128");
      const FailureOr<LdsSelection> selected = select_lds_wrong_address(
          inventory.value(), {.exact = std::nullopt, .ordinal = ordinal}, error);
      ASSERT_TRUE(selected.succeeded()) << error.message;
      EXPECT_EQ(selected.value().site().offset, ordinal * 8);
      const FailureOr<LdsSelection> exact = select_lds_wrong_address(
          inventory.value(),
          {.exact = selected.value(), .ordinal = std::numeric_limits<size_t>::max()}, error);
      ASSERT_TRUE(exact.succeeded()) << error.message;
      const FailureOr<LdsResult> mutant = apply_lds_wrong_address(source, exact.value(), 10, error);
      ASSERT_TRUE(mutant.succeeded()) << error.message;
      EXPECT_TRUE(
          test::cdna_lds_output_matches(source, GetParam(), ordinal * 8, 10, mutant.value().image));
      const FailureOr<LdsResult> repeated =
          apply_lds_wrong_address(source, selected.value(), 10, error);
      ASSERT_TRUE(repeated.succeeded()) << error.message;
      EXPECT_EQ(mutant.value().image, repeated.value().image);
    }
    EXPECT_TRUE(std::ranges::equal(source.image(), image));
  }
}

TEST_P(CdnaLds, InvalidRegistersForeignAndExpiredSelections) {
  const std::vector<uint8_t> image = test::make_cdna_lds_elf(GetParam(), false, 1, 3);
  const FaultMutationContext source(image), foreign(image);
  FaultMutationError error;
  const FailureOr<LdsInventory> inventory =
      discover_lds_wrong_address(source, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const FailureOr<LdsSelection> selected = select_lds_wrong_address(inventory.value(), {}, error);
  ASSERT_TRUE(selected.succeeded()) << error.message;
  for (std::optional<uint32_t> value :
       std::array<std::optional<uint32_t>, 4>{std::nullopt, 2u, 16u, 256u}) {
    EXPECT_TRUE(apply_lds_wrong_address(source, selected.value(), value, error).failed());
  }
  EXPECT_TRUE(apply_lds_wrong_address(foreign, selected.value(), 10, error).failed());
  EXPECT_TRUE(
      select_lds_wrong_address(inventory.value(), {.exact = LdsSelection{}, .ordinal = 0}, error)
          .failed());
  EXPECT_TRUE(
      select_lds_wrong_address(inventory.value(), {.exact = std::nullopt, .ordinal = 2}, error)
          .failed());
  const FailureOr<LdsResult> boundary =
      apply_lds_wrong_address(source, selected.value(), 15, error);
  ASSERT_TRUE(boundary.succeeded()) << error.message;
  EXPECT_TRUE(test::cdna_lds_output_matches(source, GetParam(), 0, 15, boundary.value().image));
  std::optional<LdsInventory> expired_inventory;
  std::optional<LdsSelection> expired_selection;
  {
    const FaultMutationContext temporary(image);
    const FailureOr<LdsInventory> found =
        discover_lds_wrong_address(temporary, "test_kernel", error);
    ASSERT_TRUE(found.succeeded());
    expired_inventory = found.value();
    const FailureOr<LdsSelection> selection =
        select_lds_wrong_address(*expired_inventory, {}, error);
    ASSERT_TRUE(selection.succeeded());
    expired_selection = selection.value();
  }
  EXPECT_TRUE(select_lds_wrong_address(*expired_inventory, {}, error).failed());
  EXPECT_TRUE(apply_lds_wrong_address(source, *expired_selection, 10, error).failed());
  EXPECT_TRUE(std::ranges::equal(source.image(), image));
}

TEST_P(CdnaLds, GdsAndInvalidWaveAreNotAdmitted) {
  FaultMutationError error;
  const FaultMutationContext gds(test::make_cdna_lds_elf(GetParam(), false, 3, 3, true));
  const FailureOr<LdsInventory> inventory = discover_lds_wrong_address(gds, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().sites().size(), 1u);
  EXPECT_EQ(inventory.value().sites()[0].offset, 8u);
  const FaultMutationContext wave32(test::make_cdna_lds_elf(GetParam(), false, 3, 3, false, true));
  EXPECT_TRUE(discover_lds_wrong_address(wave32, "test_kernel", error).failed());
}

TEST_P(CdnaLds, ReplacementMustStayBelowAccumulatorWindow) {
  // Total allocation is 16, but ACCUM_OFFSET=0 starts the accumulator window at v4.
  // The fixture's data operands use AGPRs, leaving only v0..v3 as ordinary VGPRs.
  const std::vector<uint8_t> image = test::make_cdna_lds_elf(GetParam(), true, 1, 0);
  const FaultMutationContext source(image);
  FaultMutationError error;
  const FailureOr<LdsInventory> inventory =
      discover_lds_wrong_address(source, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const FailureOr<LdsSelection> selection = select_lds_wrong_address(inventory.value(), {}, error);
  ASSERT_TRUE(selection.succeeded()) << error.message;
  const FailureOr<LdsResult> last_ordinary =
      apply_lds_wrong_address(source, selection.value(), 3, error);
  ASSERT_TRUE(last_ordinary.succeeded()) << error.message;
  EXPECT_TRUE(test::cdna_lds_output_matches(source, GetParam(), 0, 3, last_ordinary.value().image));
  EXPECT_TRUE(apply_lds_wrong_address(source, selection.value(), 4, error).failed());
  EXPECT_TRUE(std::ranges::equal(source.image(), image));
}

TEST_P(CdnaLds, CatalogRegisterAndAllocationBoundary) {
  // Wave64 CDNA descriptor count 6 provides 56 addressable VGPRs.
  const std::vector<uint8_t> image = test::make_cdna_lds_elf(GetParam(), false, 6, 13);
  const FaultMutationContext source(image);
  FaultMutationError error;
  const FailureOr<LdsInventory> inventory =
      discover_lds_wrong_address(source, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const FailureOr<LdsSelection> selected =
      select_lds_wrong_address(inventory.value(), {.exact = std::nullopt, .ordinal = 1}, error);
  ASSERT_TRUE(selected.succeeded()) << error.message;
  for (uint32_t replacement : {54u, 55u}) {
    const FailureOr<LdsResult> result =
        apply_lds_wrong_address(source, selected.value(), replacement, error);
    ASSERT_TRUE(result.succeeded()) << error.message;
    EXPECT_TRUE(
        test::cdna_lds_output_matches(source, GetParam(), 8, replacement, result.value().image));
  }
  EXPECT_TRUE(apply_lds_wrong_address(source, selected.value(), 56, error).failed());
  EXPECT_TRUE(std::ranges::equal(source.image(), image));
}

TEST_P(CdnaLds, ReadIsAdmittedAndTruncatedExtentIsRejected) {
  const test::CdnaLdsTarget &row = GetParam();
  // llvm-mc gfx942/gfx950: ds_read_b32 v0,v2 offset:4.
  std::vector<uint8_t> image = test::make_amdgpu_kernel_elf(
      {0xd86c0004u, 0x00000002u, 0xd9be0010u, 0x00000401u, 0xbf810000u}, 0, 3, row.elf_machine, 3,
      3);
  image[EI_DATA] = ELFDATA2LSB;
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  const FaultMutationContext source(image);
  FaultMutationError error;
  const FailureOr<LdsInventory> inventory =
      discover_lds_wrong_address(source, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().sites().size(), 2u);
  EXPECT_EQ(inventory.value().sites()[0].offset, 0u);
  EXPECT_EQ(inventory.value().sites()[0].form, "ds_read_b32");
  EXPECT_EQ(inventory.value().sites()[1].offset, 8u);
  EXPECT_EQ(inventory.value().sites()[1].form, "ds_write_b128");
  const test::LdsFormCase read_case = {.target = row.target,
                                       .machine = row.elf_machine,
                                       .arch = row.arch,
                                       .form = "ds_read_b32",
                                       .words = {0xd86c0004u, 0x00000002u}};
  const FailureOr<LdsSelection> selection = select_lds_wrong_address(inventory.value(), {}, error);
  ASSERT_TRUE(selection.succeeded()) << error.message;
  const FailureOr<LdsResult> mutant = apply_lds_wrong_address(source, selection.value(), 10, error);
  ASSERT_TRUE(mutant.succeeded()) << error.message;
  EXPECT_TRUE(test::lds_form_output_matches(source, read_case, 0, 10, mutant.value().image));
  image = test::make_amdgpu_kernel_elf({0xd81a0004u}, 0, 3, row.elf_machine, 3, 3);
  image[EI_DATA] = ELFDATA2LSB;
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  const FaultMutationContext truncated(image);
  EXPECT_TRUE(discover_lds_wrong_address(truncated, "test_kernel", error).failed());
  EXPECT_TRUE(std::ranges::equal(truncated.image(), image));
}

TEST_P(CdnaLds, FailedEmissionPreservesSource) {
  const test::CdnaLdsTarget &row = GetParam();
  std::vector<uint32_t> words = {0xd81a0004u, 0x00000302u};
  words.insert(words.end(), 32769, 0xbf800000u);
  words.push_back(0xbf810000u);
  std::vector<uint8_t> image = test::make_amdgpu_kernel_elf(words, 0, 3, row.elf_machine, 3, 3);
  image[EI_DATA] = ELFDATA2LSB;
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  const FaultMutationContext source(image);
  FaultMutationError error;
  const FailureOr<LdsInventory> inventory =
      discover_lds_wrong_address(source, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const FailureOr<LdsSelection> selected = select_lds_wrong_address(inventory.value(), {}, error);
  ASSERT_TRUE(selected.succeeded()) << error.message;
  EXPECT_TRUE(apply_lds_wrong_address(source, selected.value(), 10, error).failed());
  EXPECT_TRUE(std::ranges::equal(source.image(), image));
}

INSTANTIATE_TEST_SUITE_P(Targets, CdnaLds, ::testing::ValuesIn(test::kCdnaLdsTargets),
                         [](const ::testing::TestParamInfo<test::CdnaLdsTarget> &info) {
                           return info.param.name;
                         });
} // namespace
} // namespace rocjitsu
