// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "b96_fixture.h"
#include "rocjitsu/code/patch/fault_injection/lds_wrong_address.h"

#include <gtest/gtest.h>

namespace rocjitsu {
namespace {
class B96Lds : public ::testing::TestWithParam<test::B96Target> {};

TEST_P(B96Lds, LoadStoreAndDestinationAliasHaveIndependentExactMutants) {
  for (bool wave32 : {false, true}) {
    if (wave32 && GetParam().arch != IsaFamily::Rdna4)
      continue;
    SCOPED_TRACE(wave32);
    const test::B96Target &target = GetParam();
    const std::vector<uint8_t> image = test::make_b96_elf(target, 3, 3, false, false, {}, wave32);
    const FaultMutationContext source(image);
    FaultMutationError error;
    const FailureOr<LdsInventory> inventory =
        discover_lds_wrong_address(source, "test_kernel", error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    ASSERT_EQ(inventory.value().sites().size(), 3u);
    for (size_t ordinal = 0; ordinal < 3; ++ordinal) {
      const FailureOr<LdsSelection> selected = select_lds_wrong_address(
          inventory.value(), {.exact = std::nullopt, .ordinal = ordinal}, error);
      ASSERT_TRUE(selected.succeeded()) << error.message;
      EXPECT_EQ(selected.value().site().offset, ordinal * 8);
      const FailureOr<LdsSelection> exact = select_lds_wrong_address(
          inventory.value(),
          {.exact = selected.value(), .ordinal = std::numeric_limits<size_t>::max()}, error);
      ASSERT_TRUE(exact.succeeded()) << error.message;
      const FailureOr<LdsResult> mutant = apply_lds_wrong_address(source, exact.value(), 6, error);
      ASSERT_TRUE(mutant.succeeded()) << error.message;
      EXPECT_EQ(mutant.value().logical_mutation_count, 1u);
      EXPECT_TRUE(test::b96_output_matches(source, target, ordinal * 8, 6, mutant.value().image));
      const uint32_t bound =
          target.target == ROCJITSU_CODE_TARGET_GFX1250 ? 64u : (wave32 ? 32u : 16u);
      EXPECT_TRUE(apply_lds_wrong_address(source, exact.value(), bound, error).failed());
      const FailureOr<LdsResult> last_allocated =
          apply_lds_wrong_address(source, exact.value(), bound - 1, error);
      ASSERT_TRUE(last_allocated.succeeded()) << error.message;
      EXPECT_TRUE(test::b96_output_matches(source, target, ordinal * 8, bound - 1,
                                           last_allocated.value().image));
      const FailureOr<LdsResult> repeated =
          apply_lds_wrong_address(source, selected.value(), 6, error);
      ASSERT_TRUE(repeated.succeeded());
      EXPECT_EQ(mutant.value().image, repeated.value().image);
      std::vector<uint8_t> corrupted = mutant.value().image;
      const AmdGpuCodeObject changed(corrupted.data(), corrupted.size());
      ASSERT_TRUE(changed.is_valid());
      const std::optional<uint64_t> guest =
          test::lds_emitted_guest(*changed.text_sections().front(), ordinal * 8, target.target);
      ASSERT_TRUE(guest);
      corrupted[changed.text_sections().front()->sectionOffset() + *guest + 5] ^= 1;
      EXPECT_FALSE(test::b96_output_matches(source, target, ordinal * 8, 6, corrupted));
    }
    EXPECT_TRUE(std::ranges::equal(source.image(), image));
  }
}

TEST_P(B96Lds, RejectsInvalidRegisterAndForeignOrExpiredSource) {
  const test::B96Target &target = GetParam();
  const std::vector<uint8_t> image = test::make_b96_elf(target);
  const FaultMutationContext source(image), foreign(image);
  FaultMutationError error;
  const FailureOr<LdsInventory> inventory =
      discover_lds_wrong_address(source, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const FailureOr<LdsSelection> selected = select_lds_wrong_address(inventory.value(), {}, error);
  ASSERT_TRUE(selected.succeeded()) << error.message;
  const uint32_t bound = target.target == ROCJITSU_CODE_TARGET_GFX1250 ? 64u : 16u;
  for (std::optional<uint32_t> value :
       std::array<std::optional<uint32_t>, 4>{std::nullopt, 2u, bound, 256u})
    EXPECT_TRUE(apply_lds_wrong_address(source, selected.value(), value, error).failed());
  EXPECT_TRUE(apply_lds_wrong_address(foreign, selected.value(), 6, error).failed());
  EXPECT_TRUE(
      select_lds_wrong_address(inventory.value(), {.exact = LdsSelection{}, .ordinal = 0}, error)
          .failed());
  EXPECT_TRUE(
      select_lds_wrong_address(inventory.value(), {.exact = std::nullopt, .ordinal = 3}, error)
          .failed());
  std::optional<LdsInventory> expired_inventory;
  std::optional<LdsSelection> expired_selection;
  {
    const FaultMutationContext temporary(image);
    const FailureOr<LdsInventory> found =
        discover_lds_wrong_address(temporary, "test_kernel", error);
    ASSERT_TRUE(found.succeeded());
    expired_inventory = found.value();
    const FailureOr<LdsSelection> one = select_lds_wrong_address(*expired_inventory, {}, error);
    ASSERT_TRUE(one.succeeded());
    expired_selection = one.value();
  }
  EXPECT_TRUE(select_lds_wrong_address(*expired_inventory, {}, error).failed());
  EXPECT_TRUE(apply_lds_wrong_address(source, *expired_selection, 6, error).failed());
  EXPECT_TRUE(std::ranges::equal(source.image(), image));
}

TEST_P(B96Lds, IgnoresUnsupportedFormAndRejectsTruncatedExtent) {
  const test::B96Target &target = GetParam();
  // DS_NOP shares the encoding family but has no address operand.
  const uint32_t nop = target.cdna ? 0xd8280000u : 0xd8500000u;
  const uint32_t load = target.cdna ? 0xd9fc0004u : 0xdbf80004u;
  std::vector<uint8_t> image = test::make_amdgpu_kernel_elf(
      {nop, 0u, load, 0x04000002u, target.cdna ? 0xbf810000u : 0xbfb00000u}, 0, 3, target.machine,
      3, 3);
  image[EI_DATA] = ELFDATA2LSB;
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  const FaultMutationContext source(image);
  FaultMutationError error;
  const FailureOr<LdsInventory> inventory =
      discover_lds_wrong_address(source, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().sites().size(), 1u);
  EXPECT_EQ(inventory.value().sites()[0].offset, 8u);
  image = test::make_amdgpu_kernel_elf({load}, 0, 3, target.machine, 3, 3);
  image[EI_DATA] = ELFDATA2LSB;
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  const FaultMutationContext truncated(image);
  EXPECT_TRUE(discover_lds_wrong_address(truncated, "test_kernel", error).failed());
  EXPECT_TRUE(std::ranges::equal(truncated.image(), image));
}

TEST_P(B96Lds, EmissionFailureRollsBack) {
  const test::B96Target &target = GetParam();
  const uint32_t load = target.cdna ? 0xd9fc0004u : 0xdbf80004u;
  std::vector<uint32_t> words = {load, 0x04000002u};
  words.insert(words.end(), 32769, 0xbf800000u);
  words.push_back(target.cdna ? 0xbf810000u : 0xbfb00000u);
  std::vector<uint8_t> image = test::make_amdgpu_kernel_elf(words, 0, 3, target.machine, 3, 3);
  image[EI_DATA] = ELFDATA2LSB;
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  const FaultMutationContext source(image);
  FaultMutationError error;
  const FailureOr<LdsInventory> inventory =
      discover_lds_wrong_address(source, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const FailureOr<LdsSelection> selected = select_lds_wrong_address(inventory.value(), {}, error);
  ASSERT_TRUE(selected.succeeded()) << error.message;
  EXPECT_TRUE(apply_lds_wrong_address(source, selected.value(), 6, error).failed());
  EXPECT_TRUE(std::ranges::equal(source.image(), image));
}

TEST(B96LdsControls, Wave32DescriptorAdmissionMatchesTarget) {
  for (const test::B96Target &target : test::kB96Targets) {
    if (target.target == ROCJITSU_CODE_TARGET_GFX1250)
      continue;
    SCOPED_TRACE(target.name);
    const FaultMutationContext source(test::make_b96_elf(target, 3, 3, false, false, {}, true));
    FaultMutationError error;
    const FailureOr<LdsInventory> inventory =
        discover_lds_wrong_address(source, "test_kernel", error);
    if (target.cdna) {
      EXPECT_TRUE(inventory.failed());
      EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm);
    } else {
      ASSERT_TRUE(inventory.succeeded()) << error.message;
      ASSERT_EQ(inventory.value().sites().size(), 3u);
      const FailureOr<LdsSelection> selection =
          select_lds_wrong_address(inventory.value(), {}, error);
      ASSERT_TRUE(selection.succeeded());
      const FailureOr<LdsResult> mutated =
          apply_lds_wrong_address(source, selection.value(), 6, error);
      ASSERT_TRUE(mutated.succeeded()) << error.message;
      EXPECT_TRUE(test::b96_output_matches(source, target, 0, 6, mutated.value().image));
    }
  }
}

TEST(B96LdsControls, CdnaAccumulatorAndGdsAndOrdinaryBound) {
  for (const test::B96Target &target : test::kB96Targets) {
    if (!target.cdna)
      continue;
    SCOPED_TRACE(target.name);
    const std::vector<uint8_t> image = test::make_b96_elf(target, 1, 0, false, true);
    const FaultMutationContext source(image);
    FaultMutationError error;
    const FailureOr<LdsInventory> inventory =
        discover_lds_wrong_address(source, "test_kernel", error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    ASSERT_EQ(inventory.value().sites().size(), 3u);
    for (size_t ordinal = 0; ordinal < 3; ++ordinal) {
      const FailureOr<LdsSelection> selected = select_lds_wrong_address(
          inventory.value(), {.exact = std::nullopt, .ordinal = ordinal}, error);
      ASSERT_TRUE(selected.succeeded());
      EXPECT_TRUE(apply_lds_wrong_address(source, selected.value(), 4, error).failed());
      const FailureOr<LdsResult> mutant =
          apply_lds_wrong_address(source, selected.value(), 3, error);
      ASSERT_TRUE(mutant.succeeded()) << error.message;
      EXPECT_TRUE(test::b96_output_matches(source, target, ordinal * 8, 3, mutant.value().image));
    }
    const FaultMutationContext gds(test::make_b96_elf(target, 1, 0, true, true));
    const FailureOr<LdsInventory> excluded = discover_lds_wrong_address(gds, "test_kernel", error);
    ASSERT_TRUE(excluded.succeeded()) << error.message;
    EXPECT_TRUE(excluded.value().sites().empty());
  }
}

TEST(B96LdsControls, Gfx1250RequiresKnownAddressBankAndAllocation) {
  const test::B96Target &target = test::kB96Targets[4];
  constexpr std::array<uint32_t, 1> set_one =
      cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 1});
  const std::vector<uint8_t> image = test::make_b96_elf(target, 16, 0, false, false, set_one);
  const FaultMutationContext source(image);
  FaultMutationError error;
  const FailureOr<LdsInventory> inventory =
      discover_lds_wrong_address(source, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().sites().size(), 3u);
  for (size_t ordinal = 0; ordinal < 3; ++ordinal) {
    EXPECT_EQ(inventory.value().sites()[ordinal].address_bank, 1u);
    const FailureOr<LdsSelection> selected = select_lds_wrong_address(
        inventory.value(), {.exact = std::nullopt, .ordinal = ordinal}, error);
    ASSERT_TRUE(selected.succeeded());
    EXPECT_TRUE(apply_lds_wrong_address(source, selected.value(), 16, error).failed());
    const FailureOr<LdsResult> mutant =
        apply_lds_wrong_address(source, selected.value(), 15, error);
    ASSERT_TRUE(mutant.succeeded()) << error.message;
    EXPECT_TRUE(
        test::b96_output_matches(source, target, 4 + ordinal * 8, 15, mutant.value().image));
  }
  const FaultMutationContext out_of_allocation(
      test::make_b96_elf(target, 15, 0, false, false, set_one));
  const FailureOr<LdsInventory> absent =
      discover_lds_wrong_address(out_of_allocation, "test_kernel", error);
  ASSERT_TRUE(absent.succeeded()) << error.message;
  EXPECT_TRUE(absent.value().sites().empty());
  constexpr std::array<uint32_t, 1> branch_else =
      cdna5::build_sopp(cdna5::kSCbranchScc0Sopp, {.simm16 = 2});
  constexpr std::array<uint32_t, 1> branch_join =
      cdna5::build_sopp(cdna5::kSBranchSopp, {.simm16 = 1});
  constexpr std::array<uint32_t, 1> set_zero = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp);
  const std::array<uint32_t, 4> ambiguous = {branch_else[0], set_one[0], branch_join[0],
                                             set_zero[0]};
  const FaultMutationContext unknown(test::make_b96_elf(target, 16, 0, false, false, ambiguous));
  const FailureOr<LdsInventory> unknown_sites =
      discover_lds_wrong_address(unknown, "test_kernel", error);
  ASSERT_TRUE(unknown_sites.succeeded()) << error.message;
  EXPECT_TRUE(unknown_sites.value().sites().empty());
}
INSTANTIATE_TEST_SUITE_P(Targets, B96Lds, ::testing::ValuesIn(test::kB96Targets),
                         [](const ::testing::TestParamInfo<test::B96Target> &info) {
                           return info.param.name;
                         });
} // namespace
} // namespace rocjitsu
