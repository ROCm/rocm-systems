// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "rocjitsu/code/patch/fault_injection/lds_wrong_address.h"
#include "scalar_two_fixture.h"

#include <gtest/gtest.h>

namespace rocjitsu {
namespace {
class ScalarTwoLds : public ::testing::TestWithParam<test::LdsFormCase> {};
TEST_P(ScalarTwoLds, ExactAndOrdinalPreserveEveryNonAddressField) {
  for (bool wave32 : {false, true}) {
    if (wave32 && GetParam().arch != IsaFamily::Rdna4)
      continue;
    SCOPED_TRACE(wave32);
    const test::LdsFormCase &fixture = GetParam();
    SCOPED_TRACE(fixture.form);
    SCOPED_TRACE(static_cast<int>(fixture.target));
    const std::vector<uint8_t> image = test::make_scalar_two_elf(fixture, false, wave32);
    const FaultMutationContext source(image), foreign(image);
    FaultMutationError error;
    const FailureOr<LdsInventory> inventory =
        discover_lds_wrong_address(source, "test_kernel", error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    ASSERT_EQ(inventory.value().sites().size(), 1u);
    EXPECT_EQ(inventory.value().sites()[0].form, fixture.form);
    const FailureOr<LdsSelection> selected = select_lds_wrong_address(inventory.value(), {}, error);
    ASSERT_TRUE(selected.succeeded()) << error.message;
    const FailureOr<LdsSelection> exact = select_lds_wrong_address(
        inventory.value(),
        {.exact = selected.value(), .ordinal = std::numeric_limits<size_t>::max()}, error);
    ASSERT_TRUE(exact.succeeded()) << error.message;
    const uint32_t bound =
        fixture.target == ROCJITSU_CODE_TARGET_GFX1250 ? 64u : (wave32 ? 32u : 16u);
    for (std::optional<uint32_t> value :
         std::array<std::optional<uint32_t>, 4>{std::nullopt, 2u, bound, 256u})
      EXPECT_TRUE(apply_lds_wrong_address(source, exact.value(), value, error).failed());
    EXPECT_TRUE(apply_lds_wrong_address(foreign, exact.value(), 6, error).failed());
    const FailureOr<LdsResult> mutant = apply_lds_wrong_address(source, exact.value(), 6, error);
    ASSERT_TRUE(mutant.succeeded()) << error.message;
    EXPECT_EQ(mutant.value().logical_mutation_count, 1u);
    EXPECT_TRUE(test::lds_form_output_matches(source, fixture, 0, 6, mutant.value().image));
    const FailureOr<LdsResult> last_allocated =
        apply_lds_wrong_address(source, exact.value(), bound - 1, error);
    ASSERT_TRUE(last_allocated.succeeded()) << error.message;
    EXPECT_TRUE(
        test::lds_form_output_matches(source, fixture, 0, bound - 1, last_allocated.value().image));
    const FailureOr<LdsResult> repeated =
        apply_lds_wrong_address(source, selected.value(), 6, error);
    ASSERT_TRUE(repeated.succeeded());
    EXPECT_EQ(mutant.value().image, repeated.value().image);
    std::vector<uint8_t> corrupted = mutant.value().image;
    const AmdGpuCodeObject changed(corrupted.data(), corrupted.size());
    ASSERT_TRUE(changed.is_valid());
    const std::optional<uint64_t> guest =
        test::lds_emitted_guest(*changed.text_sections().front(), 0, fixture.target);
    ASSERT_TRUE(guest);
    corrupted[changed.text_sections().front()->sectionOffset() + *guest + 5] ^= 1;
    EXPECT_FALSE(test::lds_form_output_matches(source, fixture, 0, 6, corrupted));
    EXPECT_TRUE(std::ranges::equal(source.image(), image));
    if (fixture.target == ROCJITSU_CODE_TARGET_GFX942 ||
        fixture.target == ROCJITSU_CODE_TARGET_GFX950) {
      test::LdsFormCase gds = fixture;
      gds.words[0] |= 0x00010000u;
      const FaultMutationContext source(test::make_scalar_two_elf(gds));
      const FailureOr<LdsInventory> inventory =
          discover_lds_wrong_address(source, "test_kernel", error);
      ASSERT_TRUE(inventory.succeeded()) << error.message;
      EXPECT_TRUE(inventory.value().sites().empty());
    }
    if (fixture.target == ROCJITSU_CODE_TARGET_GFX1250) {
      const FaultMutationContext source(test::make_scalar_two_elf(fixture, true));
      const FailureOr<LdsInventory> inventory =
          discover_lds_wrong_address(source, "test_kernel", error);
      ASSERT_TRUE(inventory.succeeded()) << error.message;
      ASSERT_EQ(inventory.value().sites().size(), 1u);
      EXPECT_EQ(inventory.value().sites()[0].address_bank, 1u);
      const FailureOr<LdsSelection> selected =
          select_lds_wrong_address(inventory.value(), {}, error);
      ASSERT_TRUE(selected.succeeded());
      EXPECT_TRUE(apply_lds_wrong_address(source, selected.value(), 16, error).failed());
      const FailureOr<LdsResult> mutant =
          apply_lds_wrong_address(source, selected.value(), 15, error);
      ASSERT_TRUE(mutant.succeeded()) << error.message;
      EXPECT_TRUE(test::lds_form_output_matches(source, fixture, 4, 15, mutant.value().image));
    }
  }
}
TEST(ScalarTwoLdsControls, CdnaAccumulatorTuplesPreserveEveryNonAddressField) {
  for (rj_code_target_id_t target : {ROCJITSU_CODE_TARGET_GFX942, ROCJITSU_CODE_TARGET_GFX950}) {
    SCOPED_TRACE(static_cast<int>(target));
    const decltype(test::kScalarTwoCases)::const_iterator found =
        std::ranges::find_if(test::kScalarTwoCases, [&](const test::LdsFormCase &fixture) {
          return fixture.target == target && fixture.form == "ds_write2_b64";
        });
    ASSERT_NE(found, test::kScalarTwoCases.end());
    test::LdsFormCase fixture = *found;
    // llvm-mc: ds_write2_b64 v2, a[4:5], a[6:7] offset0:1 offset1:3.
    fixture.words = {0xda9c0301u, 0x00060402u};
    const FaultMutationContext source(test::make_scalar_two_elf(fixture));
    FaultMutationError error;
    const FailureOr<LdsInventory> inventory =
        discover_lds_wrong_address(source, "test_kernel", error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    ASSERT_EQ(inventory.value().sites().size(), 1u);
    EXPECT_EQ(inventory.value().sites()[0].form, fixture.form);
    const FailureOr<LdsSelection> selected = select_lds_wrong_address(inventory.value(), {}, error);
    ASSERT_TRUE(selected.succeeded()) << error.message;
    const FailureOr<LdsResult> mutant = apply_lds_wrong_address(source, selected.value(), 6, error);
    ASSERT_TRUE(mutant.succeeded()) << error.message;
    EXPECT_TRUE(test::lds_form_output_matches(source, fixture, 0, 6, mutant.value().image));
    std::vector<uint8_t> corrupted = mutant.value().image;
    const AmdGpuCodeObject changed(corrupted.data(), corrupted.size());
    ASSERT_TRUE(changed.is_valid());
    const std::optional<uint64_t> guest =
        test::lds_emitted_guest(*changed.text_sections().front(), 0, fixture.target);
    ASSERT_TRUE(guest);
    corrupted[changed.text_sections().front()->sectionOffset() + *guest + 3] ^= 2; // Clear ACC.
    EXPECT_FALSE(test::lds_form_output_matches(source, fixture, 0, 6, corrupted));
  }
}
TEST(ScalarTwoLdsControls, UnsupportedAtomicNopAndSplitDoNotBecomeSites) {
  for (const rj_code_target_id_t target :
       {ROCJITSU_CODE_TARGET_GFX1200, ROCJITSU_CODE_TARGET_GFX1201, ROCJITSU_CODE_TARGET_GFX942,
        ROCJITSU_CODE_TARGET_GFX950, ROCJITSU_CODE_TARGET_GFX1250}) {
    SCOPED_TRACE(static_cast<int>(target));
    const decltype(test::kScalarTwoCases)::const_iterator fixture =
        std::ranges::find(test::kScalarTwoCases, target, &test::LdsFormCase::target);
    ASSERT_NE(fixture, test::kScalarTwoCases.end());
    const bool cdna =
        target == ROCJITSU_CODE_TARGET_GFX942 || target == ROCJITSU_CODE_TARGET_GFX950;
    std::vector<uint32_t> words = {cdna ? 0xd8280000u : 0xd8500000u, 0u, 0xd8000004u, 0x00000402u};
    if (target == ROCJITSU_CODE_TARGET_GFX1250)
      words.insert(words.end(), {0xd8e00301u, 0x04000002u});
    const uint64_t offset = words.size() * 4;
    words.insert(words.end(), fixture->words.begin(), fixture->words.end());
    words.push_back(cdna ? 0xbf810000u : 0xbfb00000u);
    std::vector<uint8_t> image = test::make_amdgpu_kernel_elf(words, 0, 3, fixture->machine, 3, 3);
    image[EI_DATA] = ELFDATA2LSB;
    image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
    const FaultMutationContext source(image);
    FaultMutationError error;
    const FailureOr<LdsInventory> inventory =
        discover_lds_wrong_address(source, "test_kernel", error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    ASSERT_EQ(inventory.value().sites().size(), 1u);
    EXPECT_EQ(inventory.value().sites()[0].offset, offset);
    EXPECT_EQ(inventory.value().sites()[0].form, fixture->form);
  }
}

TEST(ScalarTwoLdsControls, ExpiredSourceTruncationAndFailedEmission) {
  for (const rj_code_target_id_t target :
       {ROCJITSU_CODE_TARGET_GFX1200, ROCJITSU_CODE_TARGET_GFX1201, ROCJITSU_CODE_TARGET_GFX942,
        ROCJITSU_CODE_TARGET_GFX950, ROCJITSU_CODE_TARGET_GFX1250}) {
    SCOPED_TRACE(static_cast<int>(target));
    const decltype(test::kScalarTwoCases)::const_iterator fixture =
        std::ranges::find(test::kScalarTwoCases, target, &test::LdsFormCase::target);
    ASSERT_NE(fixture, test::kScalarTwoCases.end());
    FaultMutationError error;
    std::optional<LdsInventory> inventory;
    std::optional<LdsSelection> selection;
    {
      const FaultMutationContext temporary(test::make_scalar_two_elf(*fixture));
      const FailureOr<LdsInventory> found =
          discover_lds_wrong_address(temporary, "test_kernel", error);
      ASSERT_TRUE(found.succeeded()) << error.message;
      inventory = found.value();
      const FailureOr<LdsSelection> selected = select_lds_wrong_address(*inventory, {}, error);
      ASSERT_TRUE(selected.succeeded()) << error.message;
      selection = selected.value();
    }
    EXPECT_TRUE(select_lds_wrong_address(*inventory, {}, error).failed());
    const FaultMutationContext fresh(test::make_scalar_two_elf(*fixture));
    EXPECT_TRUE(apply_lds_wrong_address(fresh, *selection, 6, error).failed());
    std::vector<uint8_t> truncated =
        test::make_amdgpu_kernel_elf({fixture->words[0]}, 0, 3, fixture->machine, 3, 3);
    truncated[EI_DATA] = ELFDATA2LSB;
    truncated[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
    const FaultMutationContext broken(truncated);
    EXPECT_TRUE(discover_lds_wrong_address(broken, "test_kernel", error).failed());
    EXPECT_TRUE(std::ranges::equal(broken.image(), truncated));
    std::vector<uint32_t> words(fixture->words.begin(), fixture->words.end());
    words.insert(words.end(), 32769, 0xbf800000u);
    const bool cdna =
        target == ROCJITSU_CODE_TARGET_GFX942 || target == ROCJITSU_CODE_TARGET_GFX950;
    words.push_back(cdna ? 0xbf810000u : 0xbfb00000u);
    std::vector<uint8_t> image = test::make_amdgpu_kernel_elf(words, 0, 3, fixture->machine, 3, 3);
    image[EI_DATA] = ELFDATA2LSB;
    image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
    const FaultMutationContext distant(image);
    const FailureOr<LdsInventory> distant_inventory =
        discover_lds_wrong_address(distant, "test_kernel", error);
    ASSERT_TRUE(distant_inventory.succeeded()) << error.message;
    const FailureOr<LdsSelection> selected =
        select_lds_wrong_address(distant_inventory.value(), {}, error);
    ASSERT_TRUE(selected.succeeded()) << error.message;
    EXPECT_TRUE(apply_lds_wrong_address(distant, selected.value(), 6, error).failed());
    EXPECT_TRUE(std::ranges::equal(distant.image(), image));
  }
}

TEST(ScalarTwoLdsControls, Gfx1250AllSplitFormsStayExcluded) {
  struct SplitCase {
    std::string_view form;
    std::array<uint32_t, 2> words;
  };
  // llvm-mc gfx1250 witnesses; split relocation remains outside this FI recipe.
  constexpr std::array<SplitCase, 8> cases = {{
      {.form = "ds_load_2addr_b32", .words = {0xd8dc0301u, 0x04000002u}},
      {.form = "ds_store_2addr_b32", .words = {0xd8380301u, 0x00050402u}},
      {.form = "ds_load_2addr_stride64_b32", .words = {0xd8e00301u, 0x04000002u}},
      {.form = "ds_store_2addr_stride64_b32", .words = {0xd83c0301u, 0x00050402u}},
      {.form = "ds_load_2addr_b64", .words = {0xd9dc0301u, 0x04000002u}},
      {.form = "ds_store_2addr_b64", .words = {0xd9380301u, 0x00060402u}},
      {.form = "ds_load_2addr_stride64_b64", .words = {0xd9e00301u, 0x04000002u}},
      {.form = "ds_store_2addr_stride64_b64", .words = {0xd93c0301u, 0x00060402u}},
  }};
  const decltype(test::kScalarTwoCases)::const_iterator fixture = std::ranges::find(
      test::kScalarTwoCases, ROCJITSU_CODE_TARGET_GFX1250, &test::LdsFormCase::target);
  ASSERT_NE(fixture, test::kScalarTwoCases.end());
  std::unique_ptr<Decoder> decoder =
      Decoder::create(default_isa_target_registry(), fixture->target);
  ASSERT_TRUE(decoder);
  for (const SplitCase &split : cases) {
    SCOPED_TRACE(split.form);
    const DecodeResult decoded = decoder->decode_window(split.words);
    ASSERT_TRUE(decoded.succeeded());
    ASSERT_EQ(decoded.value()->mnemonic(), split.form);
    const FaultMutationContext source(test::make_cdna5_lds_elf(
        {split.words[0], split.words[1], fixture->words[0], fixture->words[1]}, 3));
    FaultMutationError error;
    const FailureOr<LdsInventory> inventory =
        discover_lds_wrong_address(source, "test_kernel", error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    ASSERT_EQ(inventory.value().sites().size(), 1u);
    EXPECT_EQ(inventory.value().sites()[0].offset, 8u);
    EXPECT_EQ(inventory.value().sites()[0].form, fixture->form);
  }
}

TEST(ScalarTwoLdsControls, Gfx1250UnknownAddressBankIsNotAdmitted) {
  const decltype(test::kScalarTwoCases)::const_iterator fixture = std::ranges::find(
      test::kScalarTwoCases, ROCJITSU_CODE_TARGET_GFX1250, &test::LdsFormCase::target);
  ASSERT_NE(fixture, test::kScalarTwoCases.end());
  constexpr std::array<uint32_t, 1> dynamic = cdna5::build_sopk(
      cdna5::kSSetregB32Sopk, {.simm16 = 1u | (14u << 6) | (1u << 11), .sdst = 0});
  const FaultMutationContext source(
      test::make_cdna5_lds_elf({dynamic[0], fixture->words[0], fixture->words[1]}, 63));
  FaultMutationError error;
  const FailureOr<LdsInventory> inventory =
      discover_lds_wrong_address(source, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  EXPECT_TRUE(inventory.value().sites().empty());
}

TEST(ScalarTwoLdsControls, MixedProfilesKeepSourceOrderAndRewriterSemantics) {
  for (rj_code_target_id_t target :
       {ROCJITSU_CODE_TARGET_GFX1200, ROCJITSU_CODE_TARGET_GFX1201, ROCJITSU_CODE_TARGET_GFX942,
        ROCJITSU_CODE_TARGET_GFX950, ROCJITSU_CODE_TARGET_GFX1250}) {
    SCOPED_TRACE(static_cast<int>(target));
    const decltype(test::kScalarTwoCases)::const_iterator scalar =
        std::ranges::find(test::kScalarTwoCases, target, &test::LdsFormCase::target);
    ASSERT_NE(scalar, test::kScalarTwoCases.end());
    const bool cdna =
        target == ROCJITSU_CODE_TARGET_GFX942 || target == ROCJITSU_CODE_TARGET_GFX950;
    const test::LdsFormCase b96 = {.target = target,
                                   .machine = scalar->machine,
                                   .arch = scalar->arch,
                                   .form = cdna ? "ds_read_b96" : "ds_load_b96",
                                   .words = {cdna ? 0xd9fc0004u : 0xdbf80004u, 0x04000002u}};
    const test::LdsFormCase basic = {.target = target,
                                     .machine = scalar->machine,
                                     .arch = scalar->arch,
                                     .form = cdna ? "ds_write_b32" : "ds_store_b32",
                                     .words = {cdna ? 0xd81a0004u : 0xd8340004u, 0x00000402u}};
    // Source order deliberately differs from the profile table's order.
    const std::array<test::LdsFormCase, 3> fixtures = {*scalar, b96, basic};
    for (bool wave32 : {false, true}) {
      if (wave32 && scalar->arch != IsaFamily::Rdna4)
        continue;
      SCOPED_TRACE(wave32);
      std::vector<uint32_t> words;
      for (const test::LdsFormCase &fixture : fixtures)
        words.insert(words.end(), fixture.words.begin(), fixture.words.end());
      words.push_back(cdna ? 0xbf810000u : 0xbfb00000u);
      std::vector<uint8_t> image = test::make_amdgpu_kernel_elf(words, 0, 3, scalar->machine, 3, 3,
                                                                false, false, false, false, wave32);
      image[EI_DATA] = ELFDATA2LSB;
      image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
      const FaultMutationContext source(image);
      FaultMutationError error;
      const FailureOr<LdsInventory> inventory =
          discover_lds_wrong_address(source, "test_kernel", error);
      ASSERT_TRUE(inventory.succeeded()) << error.message;
      ASSERT_EQ(inventory.value().sites().size(), fixtures.size());
      for (size_t ordinal = 0; ordinal < fixtures.size(); ++ordinal) {
        SCOPED_TRACE(ordinal);
        const test::LdsFormCase &fixture = fixtures[ordinal];
        EXPECT_EQ(inventory.value().sites()[ordinal].offset, ordinal * 8);
        EXPECT_EQ(inventory.value().sites()[ordinal].form, fixture.form);
        const FailureOr<LdsSelection> selected = select_lds_wrong_address(
            inventory.value(), {.exact = std::nullopt, .ordinal = ordinal}, error);
        ASSERT_TRUE(selected.succeeded()) << error.message;
        const FailureOr<LdsResult> result =
            apply_lds_wrong_address(source, selected.value(), 6, error);
        ASSERT_TRUE(result.succeeded()) << error.message;
        EXPECT_EQ(result.value().logical_mutation_count, 1u);
        EXPECT_TRUE(
            test::lds_form_output_matches(source, fixture, ordinal * 8, 6, result.value().image));
      }
      EXPECT_TRUE(std::ranges::equal(source.image(), image));
    }
  }
}
INSTANTIATE_TEST_SUITE_P(ConcreteForms, ScalarTwoLds, ::testing::ValuesIn(test::kScalarTwoCases));
} // namespace
} // namespace rocjitsu
