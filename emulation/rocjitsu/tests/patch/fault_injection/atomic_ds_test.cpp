// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "atomic_ds_fixture.h"
#include "rocjitsu/code/patch/fault_injection/atomic_wrong_address.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace rocjitsu {
namespace {

class FaultAtomicDs : public ::testing::TestWithParam<test::AtomicTargetCase> {};

TEST_P(FaultAtomicDs, PreservesReturningCasDataOffset1AndModifiers) {
  const test::AtomicTargetCase &fixture = GetParam();
  const std::vector<AtomicCapability> capabilities = atomic_wrong_address_capabilities();
  EXPECT_TRUE(std::ranges::any_of(capabilities, [&](const AtomicCapability &capability) {
    return capability.target == fixture.target && capability.wave_size == fixture.wave &&
           capability.form == "ds_atomic";
  }));
  const size_t modifier_states = fixture.arch == IsaFamily::Cdna3 ||
                                         fixture.arch == IsaFamily::Cdna4 ||
                                         fixture.arch == IsaFamily::Rdna3
                                     ? 2
                                     : 1;
  for (test::AtomicDsForm form :
       {test::AtomicDsForm::Add, test::AtomicDsForm::AddReturning, test::AtomicDsForm::CompareStore,
        test::AtomicDsForm::CompareStoreReturning, test::AtomicDsForm::ExchangeReturning}) {
    SCOPED_TRACE(static_cast<int>(form));
    for (size_t modifiers = 0; modifiers < modifier_states; ++modifiers) {
      SCOPED_TRACE(modifiers);
      const std::vector<uint8_t> image =
          test::make_atomic_ds_elf(fixture, form, 4, 5, modifiers != 0);
      const FaultMutationContext source(image);
      FaultMutationError error;
      const FailureOr<AtomicInventory> inventory =
          discover_atomic_wrong_address(source, "test_kernel", error);
      ASSERT_TRUE(inventory.succeeded()) << error.message;
      ASSERT_EQ(inventory.value().sites().size(), 1u);
      const AtomicSite &site = inventory.value().sites().front();
      EXPECT_EQ(site.offset, 0u);
      EXPECT_EQ(site.instruction_size, 8u);
      EXPECT_EQ(site.form, test::atomic_ds_form(fixture, form));
      EXPECT_EQ(site.byte_offset, 4);
      EXPECT_TRUE(site.address_supported);
      const FailureOr<AtomicSelection> selected =
          select_atomic_wrong_address(inventory.value(), {}, error);
      ASSERT_TRUE(selected.succeeded()) << error.message;
      const FailureOr<AtomicResult> mutant =
          apply_atomic_wrong_address(source, selected.value(), 8, error);
      ASSERT_TRUE(mutant.succeeded()) << error.message;
      EXPECT_EQ(mutant.value().logical_mutation_count, 1u);
      EXPECT_TRUE(test::atomic_ds_output_matches(source, 0, 8, mutant.value().image));
      const AmdGpuCodeObject candidate(mutant.value().image.data(), mutant.value().image.size());
      ASSERT_TRUE(candidate.is_valid());
      const Section &text = *candidate.text_sections().front();
      const std::optional<uint64_t> guest = test::atomic_emitted_guest(text, 0, fixture.target);
      ASSERT_TRUE(guest);
      ASSERT_LE(*guest + 8u, text.size());
      std::array<uint32_t, 2> changed = {};
      std::memcpy(changed.data(), text.data() + *guest, sizeof(changed));
      const std::array<uint32_t, 2> original =
          test::atomic_ds_words(fixture, form, 4, 5, modifiers != 0);
      EXPECT_EQ(changed[0] & ~0xffu, original[0] & ~0xffu);
      EXPECT_EQ(changed[0] & 0xffu, 12u);
      EXPECT_EQ((changed[0] >> 8) & 0xffu, 5u);
      EXPECT_EQ(changed[1], original[1]);
      EXPECT_EQ(image, std::vector<uint8_t>(source.image().begin(), source.image().end()));
    }
  }
}

TEST_P(FaultAtomicDs, ChecksOffset0OverflowAndNaturalAlignmentWithoutCarryingOffset1) {
  const test::AtomicTargetCase &fixture = GetParam();
  const std::vector<uint8_t> image = test::make_atomic_ds_elf(fixture);
  const FaultMutationContext source(image);
  FaultMutationError error;
  const FailureOr<AtomicInventory> inventory =
      discover_atomic_wrong_address(source, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const FailureOr<AtomicSelection> selected =
      select_atomic_wrong_address(inventory.value(), {}, error);
  ASSERT_TRUE(selected.succeeded()) << error.message;
  for (std::optional<uint32_t> delta :
       {std::optional<uint32_t>{}, std::optional<uint32_t>{0}, std::optional<uint32_t>{2},
        std::optional<uint32_t>{256}, std::optional<uint32_t>{0x800000},
        std::optional<uint32_t>{UINT32_MAX}}) {
    EXPECT_TRUE(apply_atomic_wrong_address(source, selected.value(), delta, error).failed());
    EXPECT_EQ(error.code, FaultMutationErrorCode::InvalidParameters);
    EXPECT_TRUE(std::ranges::equal(source.image(), image));
  }
  const FailureOr<AtomicResult> maximum =
      apply_atomic_wrong_address(source, selected.value(), 252, error);
  ASSERT_TRUE(maximum.succeeded()) << error.message;
  EXPECT_TRUE(test::atomic_ds_output_matches(source, 0, 252, maximum.value().image));
  for (uint8_t offset : {uint8_t{2}, uint8_t{252}, uint8_t{255}}) {
    SCOPED_TRACE(static_cast<int>(offset));
    const std::vector<uint8_t> invalid =
        test::make_atomic_ds_elf(fixture, test::AtomicDsForm::Add, offset, 5);
    const FaultMutationContext bounded(invalid);
    const FailureOr<AtomicInventory> found =
        discover_atomic_wrong_address(bounded, "test_kernel", error);
    ASSERT_TRUE(found.succeeded()) << error.message;
    const FailureOr<AtomicSelection> selection =
        select_atomic_wrong_address(found.value(), {}, error);
    ASSERT_TRUE(selection.succeeded()) << error.message;
    EXPECT_TRUE(apply_atomic_wrong_address(bounded, selection.value(), 4, error).failed());
    EXPECT_EQ(error.code, FaultMutationErrorCode::InvalidParameters);
    EXPECT_TRUE(std::ranges::equal(bounded.image(), invalid));
  }
}

TEST_P(FaultAtomicDs, UnsupportedWidthRemainsInStableAtomicOrdinalUnion) {
  const test::AtomicTargetCase &fixture = GetParam();
  const std::array<uint32_t, 2> wide = test::atomic_ds_words(fixture, test::AtomicDsForm::WideAdd);
  const std::array<uint32_t, 2> supported =
      test::atomic_ds_words(fixture, test::AtomicDsForm::CompareStoreReturning);
  const FaultMutationContext source(
      test::make_atomic_ds_sequence_elf(fixture, {wide[0], wide[1], supported[0], supported[1]}));
  FaultMutationError error;
  const FailureOr<AtomicInventory> inventory =
      discover_atomic_wrong_address(source, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().sites().size(), 2u);
  EXPECT_EQ(inventory.value().sites()[0].offset, 0u);
  EXPECT_EQ(inventory.value().sites()[0].form, "ds_add_u64");
  EXPECT_FALSE(inventory.value().sites()[0].address_supported);
  EXPECT_EQ(inventory.value().sites()[1].offset, 8u);
  EXPECT_TRUE(inventory.value().sites()[1].address_supported);
  const FailureOr<AtomicSelection> rejected =
      select_atomic_wrong_address(inventory.value(), {}, error);
  ASSERT_TRUE(rejected.succeeded()) << error.message;
  EXPECT_TRUE(apply_atomic_wrong_address(source, rejected.value(), 4, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm);
  const FailureOr<AtomicSelection> selected =
      select_atomic_wrong_address(inventory.value(), {.exact = std::nullopt, .ordinal = 1}, error);
  ASSERT_TRUE(selected.succeeded()) << error.message;
  const FailureOr<AtomicResult> mutant =
      apply_atomic_wrong_address(source, selected.value(), 4, error);
  ASSERT_TRUE(mutant.succeeded()) << error.message;
  EXPECT_TRUE(test::atomic_ds_output_matches(source, 8, 4, mutant.value().image));
}

TEST_P(FaultAtomicDs, ExactOwnerBoundSelectionAndIndependentMutantsUsePristineBytes) {
  const test::AtomicTargetCase &fixture = GetParam();
  const std::vector<uint8_t> image =
      test::make_atomic_ds_elf(fixture, test::AtomicDsForm::AddReturning, 4);
  const FaultMutationContext source(image), foreign(image);
  FaultMutationError error;
  const FailureOr<AtomicInventory> inventory =
      discover_atomic_wrong_address(source, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const FailureOr<AtomicSelection> selected =
      select_atomic_wrong_address(inventory.value(), {}, error);
  ASSERT_TRUE(selected.succeeded()) << error.message;
  const FailureOr<AtomicSelection> exact = select_atomic_wrong_address(
      inventory.value(), {.exact = selected.value(), .ordinal = std::numeric_limits<size_t>::max()},
      error);
  ASSERT_TRUE(exact.succeeded()) << error.message;
  for (uint32_t delta : {4u, 8u}) {
    const FailureOr<AtomicResult> mutant =
        apply_atomic_wrong_address(source, exact.value(), delta, error);
    ASSERT_TRUE(mutant.succeeded()) << error.message;
    EXPECT_TRUE(test::atomic_ds_output_matches(source, 0, delta, mutant.value().image));
  }
  EXPECT_TRUE(apply_atomic_wrong_address(foreign, exact.value(), 4, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  EXPECT_TRUE(select_atomic_wrong_address(inventory.value(),
                                          {.exact = AtomicSelection{}, .ordinal = 0}, error)
                  .failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  EXPECT_TRUE(std::ranges::equal(source.image(), image));
}

TEST_P(FaultAtomicDs, IndependentOracleRejectsOffset1OperandsModifiersWaitAndContinuationDrift) {
  const test::AtomicTargetCase &fixture = GetParam();
  const FaultMutationContext source(
      test::make_atomic_ds_elf(fixture, test::AtomicDsForm::CompareStoreReturning, 4, 5));
  FaultMutationError error;
  const FailureOr<AtomicInventory> inventory =
      discover_atomic_wrong_address(source, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const FailureOr<AtomicSelection> selected =
      select_atomic_wrong_address(inventory.value(), {}, error);
  ASSERT_TRUE(selected.succeeded()) << error.message;
  const FailureOr<AtomicResult> mutant =
      apply_atomic_wrong_address(source, selected.value(), 4, error);
  ASSERT_TRUE(mutant.succeeded()) << error.message;
  ASSERT_TRUE(test::atomic_ds_output_matches(source, 0, 4, mutant.value().image));
  const AmdGpuCodeObject candidate(mutant.value().image.data(), mutant.value().image.size());
  ASSERT_TRUE(candidate.is_valid());
  const Section &text = *candidate.text_sections().front();
  const std::optional<uint64_t> guest = test::atomic_emitted_guest(text, 0, fixture.target);
  ASSERT_TRUE(guest);
  ASSERT_LE(*guest + 12u, text.size());
  struct Corruption {
    size_t word;
    uint32_t mask;
  };
  constexpr std::array<Corruption, 8> corruptions = {{
      {.word = 0, .mask = 4u},
      {.word = 0, .mask = 1u << 8},
      {.word = 0, .mask = 1u << 17},
      {.word = 0, .mask = 1u << 18},
      {.word = 1, .mask = 1u},
      {.word = 1, .mask = 1u << 8},
      {.word = 1, .mask = 1u << 16},
      {.word = 1, .mask = 1u << 24},
  }};
  for (const Corruption &corruption : corruptions) {
    SCOPED_TRACE(corruption.word);
    SCOPED_TRACE(corruption.mask);
    std::vector<uint8_t> corrupted = mutant.value().image;
    const uint64_t location = text.sectionOffset() + *guest + corruption.word * sizeof(uint32_t);
    uint32_t word = 0;
    std::memcpy(&word, corrupted.data() + location, sizeof(word));
    word ^= corruption.mask;
    std::memcpy(corrupted.data() + location, &word, sizeof(word));
    EXPECT_FALSE(test::atomic_ds_output_matches(source, 0, 4, corrupted));
  }
  for (uint64_t location : {text.sectionOffset() + 8, text.sectionOffset() + *guest + 8}) {
    std::vector<uint8_t> corrupted = mutant.value().image;
    corrupted[location] ^= 1u;
    EXPECT_FALSE(test::atomic_ds_output_matches(source, 0, 4, corrupted));
  }
}

TEST(FaultAtomicDsPolicy, PreservesOriginalTwoAddressOffset0UnitsAndStrideExclusion) {
  for (const test::AtomicTargetCase &fixture : test::kAtomicDsTargetCases) {
    SCOPED_TRACE(fixture.name);
    for (test::AtomicDsForm form :
         {test::AtomicDsForm::Exchange2, test::AtomicDsForm::Exchange2Stride64}) {
      const FaultMutationContext source(test::make_atomic_ds_elf(fixture, form, 4, 5));
      FaultMutationError error;
      const FailureOr<AtomicInventory> inventory =
          discover_atomic_wrong_address(source, "test_kernel", error);
      ASSERT_TRUE(inventory.succeeded()) << error.message;
      ASSERT_EQ(inventory.value().sites().size(), 1u);
      EXPECT_EQ(inventory.value().sites().front().form, test::atomic_ds_form(fixture, form));
      const bool supported = form == test::AtomicDsForm::Exchange2;
      EXPECT_EQ(inventory.value().sites().front().address_supported, supported);
      const FailureOr<AtomicSelection> selected =
          select_atomic_wrong_address(inventory.value(), {}, error);
      ASSERT_TRUE(selected.succeeded()) << error.message;
      const FailureOr<AtomicResult> mutant =
          apply_atomic_wrong_address(source, selected.value(), 4, error);
      if (supported) {
        ASSERT_TRUE(mutant.succeeded()) << error.message;
        EXPECT_TRUE(test::atomic_ds_output_matches(source, 0, 4, mutant.value().image));
      } else {
        EXPECT_TRUE(mutant.failed());
        EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm);
      }
    }
  }
}

TEST(FaultAtomicDsPolicy, NativeCdnaExchangesKeepMixedUnsupportedOrdinalsAndFullGuests) {
  for (const test::AtomicTargetCase &fixture : test::kNativeCdnaExchangeCases) {
    SCOPED_TRACE(fixture.name);
    for (bool modifiers : {false, true}) {
      SCOPED_TRACE(modifiers);
      const std::vector<uint8_t> image =
          test::make_mixed_native_cdna_exchange_elf(fixture, modifiers);
      const FaultMutationContext source(image);
      FaultMutationError error;
      const FailureOr<AtomicInventory> inventory =
          discover_atomic_wrong_address(source, "test_kernel", error);
      ASSERT_TRUE(inventory.succeeded()) << error.message;
      ASSERT_EQ(inventory.value().sites().size(), 5u);
      constexpr std::array<std::string_view, 5> names = {
          "ds_wrxchg_rtn_b64", "ds_wrxchg2st64_rtn_b32", "ds_wrxchg_rtn_b32", "ds_wrxchg2_rtn_b32",
          "ds_add_rtn_u32"};
      for (size_t ordinal = 0; ordinal < names.size(); ++ordinal) {
        SCOPED_TRACE(ordinal);
        const AtomicSite &site = inventory.value().sites()[ordinal];
        EXPECT_EQ(site.offset, ordinal * 8);
        EXPECT_EQ(site.form, names[ordinal]);
        EXPECT_EQ(site.byte_offset, ordinal == 0 ? 8 : 4);
        EXPECT_EQ(site.address_supported, ordinal >= 2);
        const FailureOr<AtomicSelection> selected = select_atomic_wrong_address(
            inventory.value(), {.exact = std::nullopt, .ordinal = ordinal}, error);
        ASSERT_TRUE(selected.succeeded()) << error.message;
        const FailureOr<AtomicSelection> exact = select_atomic_wrong_address(
            inventory.value(), {.exact = selected.value(), .ordinal = SIZE_MAX}, error);
        ASSERT_TRUE(exact.succeeded()) << error.message;
        const FailureOr<AtomicResult> mutant =
            apply_atomic_wrong_address(source, exact.value(), 8, error);
        if (ordinal < 2) {
          EXPECT_TRUE(mutant.failed());
          EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm);
        } else {
          ASSERT_TRUE(mutant.succeeded()) << error.message;
          EXPECT_EQ(mutant.value().logical_mutation_count, 1u);
          EXPECT_TRUE(test::atomic_ds_output_matches(source, ordinal * 8, 8, mutant.value().image));
        }
      }
      EXPECT_TRUE(select_atomic_wrong_address(inventory.value(),
                                              {.exact = AtomicSelection{}, .ordinal = 2}, error)
                      .failed());
      EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
      EXPECT_TRUE(std::ranges::equal(source.image(), image));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Targets, FaultAtomicDs, ::testing::ValuesIn(test::kAtomicDsTargetCases),
                         [](const ::testing::TestParamInfo<test::AtomicTargetCase> &info) {
                           return info.param.name;
                         });

} // namespace
} // namespace rocjitsu
