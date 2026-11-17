// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "ordinary_wrong_address_oracle.h"
#include "rocjitsu/code/patch/fault_injection/ordinary_wrong_address.h"

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

class FaultOrdinaryWrongAddress : public ::testing::TestWithParam<test::OrdinaryTargetCase> {};

TEST_P(FaultOrdinaryWrongAddress, ChangesOnlySignedByteOffsetAndPreservesCompleteAcquireProof) {
  const test::OrdinaryTargetCase &fixture = GetParam();
  for (bool zero_sized : {false, true}) {
    SCOPED_TRACE(zero_sized);
    for (uint32_t scope : {2u, 3u}) {
      SCOPED_TRACE(scope);
      for (int32_t offset : {-4, 0, 4}) {
        SCOPED_TRACE(offset);
        for (bool modifiers : {false, true}) {
          SCOPED_TRACE(modifiers);
          std::vector<uint8_t> image =
              test::make_modern_ordinary_address_elf(fixture, scope, offset, modifiers);
          if (zero_sized) {
            ASSERT_TRUE(test::zero_ordinary_function_size(image, 0));
          }
          const FaultMutationContext source(image);
          FaultMutationError error;
          const FailureOr<OrdinaryAcquireInventory> found =
              discover_ordinary_acquires(source, "test_kernel", error);
          ASSERT_TRUE(found.succeeded()) << error.message;
          ASSERT_EQ(found.value().sites().size(), 1u);
          const OrdinaryAcquireSite &site = found.value().sites().front();
          EXPECT_EQ(site.profile, fixture.arch == IsaFamily::Cdna5
                                      ? OrdinaryAcquireProfile::Cdna5GlobalB32
                                      : OrdinaryAcquireProfile::Rdna4GlobalB32);
          EXPECT_EQ(site.form, "global_load_b32");
          EXPECT_EQ(site.instruction_size, 12u);
          EXPECT_EQ(site.byte_offset, offset);
          EXPECT_EQ(site.raw_scope, scope);
          ASSERT_EQ(site.wait_members.size(), 1u);
          EXPECT_EQ(site.wait_members.front().offset, 12u);
          ASSERT_EQ(site.cache_members.size(), 1u);
          EXPECT_EQ(site.cache_members.front().offset, 20u);
          const FailureOr<OrdinaryAcquireSelection> selected =
              select_ordinary_acquire(found.value(), {}, error);
          ASSERT_TRUE(selected.succeeded()) << error.message;
          const FailureOr<OrdinaryWrongAddressResult> mutant =
              apply_ordinary_wrong_address(source, selected.value(), 8, error);
          ASSERT_TRUE(mutant.succeeded()) << error.message;
          EXPECT_EQ(mutant.value().logical_mutation_count, 1u);
          EXPECT_EQ(mutant.value().selection.site(), site);
          EXPECT_TRUE(
              test::ordinary_wrong_address_output_matches(source, 0, 8, mutant.value().image));
          const AmdGpuCodeObject candidate(mutant.value().image.data(),
                                           mutant.value().image.size());
          ASSERT_TRUE(candidate.is_valid());
          const Section &text = *candidate.text_sections().front();
          const std::optional<uint64_t> guest =
              test::ordinary_emitted_guest(text, 0, fixture.target);
          ASSERT_TRUE(guest);
          ASSERT_LE(*guest + 12u, text.size());
          std::array<uint32_t, 3> changed = {};
          std::memcpy(changed.data(), text.data() + *guest, sizeof(changed));
          const std::array<uint32_t, 3> original =
              test::modern_ordinary_address_words(fixture, scope, offset, modifiers);
          EXPECT_EQ(changed[0], original[0]);
          EXPECT_EQ(changed[1], original[1]);
          EXPECT_EQ(changed[2] & 0xffu, original[2] & 0xffu);
          EXPECT_EQ(changed[2] >> 8, static_cast<uint32_t>(offset + 8));
          EXPECT_EQ((changed[1] >> 16) & 1u,
                    static_cast<uint32_t>(modifiers && fixture.arch == IsaFamily::Cdna5));
          EXPECT_EQ((changed[1] >> 18) & 3u, scope);
          EXPECT_EQ((changed[1] >> 20) & 7u, modifiers ? 6u : 0u);
          const std::array<uint32_t, 1> expected_wait = {test::kOrdinaryWaitLoadZero};
          EXPECT_EQ(std::memcmp(text.data() + 12, expected_wait.data(), sizeof(expected_wait)), 0);
          EXPECT_EQ(std::memcmp(text.data() + 20, test::kOrdinaryAcquireCache.data(), 12), 0);
          EXPECT_TRUE(std::ranges::equal(source.image(), image));
        }
      }
    }
  }
}

TEST_P(FaultOrdinaryWrongAddress, RejectsInvalidDeltaAndChecksSignedSumAlignmentAndOverflow) {
  const test::OrdinaryTargetCase &fixture = GetParam();
  const std::vector<uint8_t> image = test::make_ordinary_elf(fixture);
  const FaultMutationContext source(image);
  FaultMutationError error;
  const FailureOr<OrdinaryAcquireInventory> found =
      discover_ordinary_acquires(source, "test_kernel", error);
  ASSERT_TRUE(found.succeeded()) << error.message;
  const FailureOr<OrdinaryAcquireSelection> selected =
      select_ordinary_acquire(found.value(), {}, error);
  ASSERT_TRUE(selected.succeeded()) << error.message;
  for (std::optional<uint32_t> delta :
       {std::optional<uint32_t>{}, std::optional<uint32_t>{0}, std::optional<uint32_t>{2},
        std::optional<uint32_t>{0x800000}, std::optional<uint32_t>{UINT32_MAX}}) {
    EXPECT_TRUE(apply_ordinary_wrong_address(source, selected.value(), delta, error).failed());
    EXPECT_EQ(error.code, FaultMutationErrorCode::InvalidParameters);
    EXPECT_TRUE(std::ranges::equal(source.image(), image));
  }
  const FailureOr<OrdinaryWrongAddressResult> maximum =
      apply_ordinary_wrong_address(source, selected.value(), 0x7ffffc, error);
  ASSERT_TRUE(maximum.succeeded()) << error.message;
  EXPECT_TRUE(
      test::ordinary_wrong_address_output_matches(source, 0, 0x7ffffc, maximum.value().image));
  for (int32_t original : {-0x800000, -8, 0x7ffff8}) {
    SCOPED_TRACE(original);
    const FaultMutationContext bounded(test::make_ordinary_elf(fixture, 2, original));
    const FailureOr<OrdinaryAcquireInventory> inventory =
        discover_ordinary_acquires(bounded, "test_kernel", error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    const FailureOr<OrdinaryAcquireSelection> selection =
        select_ordinary_acquire(inventory.value(), {}, error);
    ASSERT_TRUE(selection.succeeded()) << error.message;
    const FailureOr<OrdinaryWrongAddressResult> mutant =
        apply_ordinary_wrong_address(bounded, selection.value(), 4, error);
    ASSERT_TRUE(mutant.succeeded()) << error.message;
    EXPECT_TRUE(test::ordinary_wrong_address_output_matches(bounded, 0, 4, mutant.value().image));
  }
  for (int32_t original : {-2, 2, 0x7ffffc}) {
    SCOPED_TRACE(original);
    const std::vector<uint8_t> bounded_image = test::make_ordinary_elf(fixture, 2, original);
    const FaultMutationContext bounded(bounded_image);
    const FailureOr<OrdinaryAcquireInventory> inventory =
        discover_ordinary_acquires(bounded, "test_kernel", error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    const FailureOr<OrdinaryAcquireSelection> selection =
        select_ordinary_acquire(inventory.value(), {}, error);
    ASSERT_TRUE(selection.succeeded()) << error.message;
    EXPECT_TRUE(apply_ordinary_wrong_address(bounded, selection.value(), 4, error).failed());
    EXPECT_EQ(error.code, FaultMutationErrorCode::InvalidParameters);
    EXPECT_TRUE(std::ranges::equal(bounded.image(), bounded_image));
  }
}

TEST_P(FaultOrdinaryWrongAddress, ExactSelectionUsesQualifiedOrdinalAndPristineIndependentMutants) {
  const test::OrdinaryTargetCase &fixture = GetParam();
  std::vector<uint32_t> words = test::ordinary_acquire_words(0);
  for (int32_t offset : {4, -4}) {
    const std::vector<uint32_t> acquire = test::ordinary_acquire_words(2, offset);
    words.insert(words.end(), acquire.begin(), acquire.end());
  }
  const std::vector<uint8_t> image = test::make_ordinary_words_elf(fixture, std::move(words));
  const FaultMutationContext source(image);
  FaultMutationError error;
  const FailureOr<OrdinaryAcquireInventory> found =
      discover_ordinary_acquires(source, "test_kernel", error);
  ASSERT_TRUE(found.succeeded()) << error.message;
  ASSERT_EQ(found.value().sites().size(), 2u);
  EXPECT_EQ(found.value().sites()[0].offset, 28u);
  EXPECT_EQ(found.value().sites()[1].offset, 56u);
  const FailureOr<OrdinaryAcquireSelection> selected =
      select_ordinary_acquire(found.value(), {.exact = std::nullopt, .ordinal = 1}, error);
  ASSERT_TRUE(selected.succeeded()) << error.message;
  const FailureOr<OrdinaryAcquireSelection> exact = select_ordinary_acquire(
      found.value(), {.exact = selected.value(), .ordinal = std::numeric_limits<size_t>::max()},
      error);
  ASSERT_TRUE(exact.succeeded()) << error.message;
  std::vector<uint8_t> first;
  for (uint32_t delta : {4u, 8u}) {
    const FailureOr<OrdinaryWrongAddressResult> mutant =
        apply_ordinary_wrong_address(source, exact.value(), delta, error);
    ASSERT_TRUE(mutant.succeeded()) << error.message;
    EXPECT_TRUE(
        test::ordinary_wrong_address_output_matches(source, 56, delta, mutant.value().image));
    if (first.empty())
      first = mutant.value().image;
    else
      EXPECT_NE(first, mutant.value().image);
  }
  EXPECT_TRUE(select_ordinary_acquire(found.value(),
                                      {.exact = OrdinaryAcquireSelection{}, .ordinal = 0}, error)
                  .failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  EXPECT_TRUE(std::ranges::equal(source.image(), image));
}

TEST_P(FaultOrdinaryWrongAddress, RejectsForeignDefaultAndExpiredSourceHandles) {
  const test::OrdinaryTargetCase &fixture = GetParam();
  const std::vector<uint8_t> image = test::make_ordinary_elf(fixture);
  const FaultMutationContext source(image), foreign(image);
  FaultMutationError error;
  const FailureOr<OrdinaryAcquireInventory> found =
      discover_ordinary_acquires(source, "test_kernel", error);
  ASSERT_TRUE(found.succeeded()) << error.message;
  const FailureOr<OrdinaryAcquireSelection> selected =
      select_ordinary_acquire(found.value(), {}, error);
  ASSERT_TRUE(selected.succeeded()) << error.message;
  EXPECT_TRUE(apply_ordinary_wrong_address(foreign, selected.value(), 4, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  EXPECT_TRUE(apply_ordinary_wrong_address(source, OrdinaryAcquireSelection{}, 4, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  std::optional<OrdinaryAcquireInventory> expired_inventory;
  std::optional<OrdinaryAcquireSelection> expired_selection;
  {
    const FaultMutationContext transient(image);
    const FailureOr<OrdinaryAcquireInventory> inventory =
        discover_ordinary_acquires(transient, "test_kernel", error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    const FailureOr<OrdinaryAcquireSelection> selection =
        select_ordinary_acquire(inventory.value(), {}, error);
    ASSERT_TRUE(selection.succeeded()) << error.message;
    expired_inventory = inventory.value();
    expired_selection = selection.value();
  }
  EXPECT_EQ(expired_selection->site().form, "global_load_b32");
  ASSERT_EQ(expired_selection->site().cache_members.size(), 1u);
  EXPECT_EQ(expired_selection->site().cache_members.front().form, "global_inv");
  EXPECT_TRUE(select_ordinary_acquire(*expired_inventory, {}, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  EXPECT_TRUE(apply_ordinary_wrong_address(source, *expired_selection, 4, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  EXPECT_TRUE(std::ranges::equal(source.image(), image));
  EXPECT_TRUE(std::ranges::equal(foreign.image(), image));
}

TEST_P(FaultOrdinaryWrongAddress, SharedHelperRetainsCompleteOwnersAndRebindsExactOwner) {
  const test::OrdinaryTargetCase &fixture = GetParam();
  for (bool zero_sized : {false, true}) {
    SCOPED_TRACE(zero_sized);
    std::vector<uint8_t> image = test::make_shared_ordinary_elf(fixture);
    if (zero_sized) {
      ASSERT_TRUE(test::zero_ordinary_function_size(image, 512));
    }
    const FaultMutationContext source(image);
    FaultMutationError error;
    const FailureOr<OrdinaryAcquireInventory> first =
        discover_ordinary_acquires(source, "test_kernel0", error);
    ASSERT_TRUE(first.succeeded()) << error.message;
    const FailureOr<OrdinaryAcquireInventory> second =
        discover_ordinary_acquires(source, "test_kernel1", error);
    ASSERT_TRUE(second.succeeded()) << error.message;
    ASSERT_EQ(first.value().sites().size(), 1u);
    ASSERT_EQ(second.value().sites().size(), 1u);
    const OrdinaryAcquireSite &site = first.value().sites().front();
    EXPECT_EQ(site.offset, 512u);
    EXPECT_EQ(site, second.value().sites().front());
    ASSERT_EQ(site.affected_owners.size(), 2u);
    EXPECT_EQ(site.affected_owners[0].entry, "test_kernel0");
    EXPECT_EQ(site.affected_owners[1].entry, "test_kernel1");
    const FailureOr<OrdinaryAcquireSelection> selected =
        select_ordinary_acquire(first.value(), {}, error);
    ASSERT_TRUE(selected.succeeded()) << error.message;
    const FailureOr<OrdinaryAcquireSelection> rebound = select_ordinary_acquire(
        second.value(), {.exact = selected.value(), .ordinal = SIZE_MAX}, error);
    ASSERT_TRUE(rebound.succeeded()) << error.message;
    ASSERT_NE(rebound.value().owner(), nullptr);
    EXPECT_EQ(rebound.value().owner()->entry, "test_kernel1");
    for (const OrdinaryAcquireSelection &selection : {selected.value(), rebound.value()}) {
      const FailureOr<OrdinaryWrongAddressResult> mutant =
          apply_ordinary_wrong_address(source, selection, 4, error);
      ASSERT_TRUE(mutant.succeeded()) << error.message;
      EXPECT_EQ(mutant.value().selection.site().affected_owners, site.affected_owners);
      EXPECT_TRUE(
          test::ordinary_wrong_address_output_matches(source, 512, 4, mutant.value().image));
    }
    EXPECT_TRUE(std::ranges::equal(source.image(), image));
  }
}

TEST_P(FaultOrdinaryWrongAddress, PreservesBoundedFallthroughAndSelfLoopProofMembers) {
  const test::OrdinaryTargetCase &fixture = GetParam();
  for (bool polling : {false, true}) {
    SCOPED_TRACE(polling);
    const std::vector<uint8_t> image = polling ? test::make_ordinary_poll_elf(fixture)
                                               : test::make_ordinary_fallthrough_elf(fixture);
    const FaultMutationContext source(image);
    FaultMutationError error;
    const FailureOr<OrdinaryAcquireInventory> found =
        discover_ordinary_acquires(source, "test_kernel", error);
    ASSERT_TRUE(found.succeeded()) << error.message;
    ASSERT_EQ(found.value().sites().size(), 1u);
    const OrdinaryAcquireSite &site = found.value().sites().front();
    EXPECT_EQ(site.path, polling ? OrdinaryAcquirePath::ExactSelfLoopExit
                                 : OrdinaryAcquirePath::ExactFallthroughJoin);
    const FailureOr<OrdinaryAcquireSelection> selected =
        select_ordinary_acquire(found.value(), {}, error);
    ASSERT_TRUE(selected.succeeded()) << error.message;
    const FailureOr<OrdinaryWrongAddressResult> mutant =
        apply_ordinary_wrong_address(source, selected.value(), 4, error);
    ASSERT_TRUE(mutant.succeeded()) << error.message;
    EXPECT_TRUE(
        test::ordinary_wrong_address_output_matches(source, site.offset, 4, mutant.value().image));
    EXPECT_TRUE(std::ranges::equal(source.image(), image));
  }
}

TEST_P(FaultOrdinaryWrongAddress, IndependentOracleRejectsOperandModifierAndAcquireMemberDrift) {
  const test::OrdinaryTargetCase &fixture = GetParam();
  const FaultMutationContext source(test::make_modern_ordinary_address_elf(fixture, 3, 4, true));
  FaultMutationError error;
  const FailureOr<OrdinaryAcquireInventory> found =
      discover_ordinary_acquires(source, "test_kernel", error);
  ASSERT_TRUE(found.succeeded()) << error.message;
  const FailureOr<OrdinaryAcquireSelection> selected =
      select_ordinary_acquire(found.value(), {}, error);
  ASSERT_TRUE(selected.succeeded()) << error.message;
  const FailureOr<OrdinaryWrongAddressResult> mutant =
      apply_ordinary_wrong_address(source, selected.value(), 4, error);
  ASSERT_TRUE(mutant.succeeded()) << error.message;
  ASSERT_TRUE(test::ordinary_wrong_address_output_matches(source, 0, 4, mutant.value().image));
  const AmdGpuCodeObject candidate(mutant.value().image.data(), mutant.value().image.size());
  ASSERT_TRUE(candidate.is_valid());
  const Section &text = *candidate.text_sections().front();
  const std::optional<uint64_t> guest = test::ordinary_emitted_guest(text, 0, fixture.target);
  ASSERT_TRUE(guest);
  ASSERT_LE(*guest + 16u, text.size());
  struct Corruption {
    size_t word;
    uint32_t mask;
  };
  constexpr std::array<Corruption, 9> corruptions = {{
      {.word = 0, .mask = 1u},
      {.word = 0, .mask = 1u << 7},
      {.word = 1, .mask = 1u},
      {.word = 1, .mask = 1u << 16},
      {.word = 1, .mask = 1u << 17},
      {.word = 1, .mask = 1u << 18},
      {.word = 1, .mask = 1u << 20},
      {.word = 2, .mask = 1u},
      {.word = 2, .mask = 1u << 10},
  }};
  for (const Corruption &corruption : corruptions) {
    std::vector<uint8_t> corrupted = mutant.value().image;
    const uint64_t location = text.sectionOffset() + *guest + corruption.word * sizeof(uint32_t);
    uint32_t word = 0;
    std::memcpy(&word, corrupted.data() + location, sizeof(word));
    word ^= corruption.mask;
    std::memcpy(corrupted.data() + location, &word, sizeof(word));
    EXPECT_FALSE(test::ordinary_wrong_address_output_matches(source, 0, 4, corrupted));
  }
  for (uint64_t location :
       {text.sectionOffset(), text.sectionOffset() + 12, text.sectionOffset() + 16,
        text.sectionOffset() + 20, text.sectionOffset() + *guest + 12}) {
    std::vector<uint8_t> corrupted = mutant.value().image;
    corrupted[location] ^= 1u;
    EXPECT_FALSE(test::ordinary_wrong_address_output_matches(source, 0, 4, corrupted));
  }
  std::vector<uint8_t> corrupted_header = mutant.value().image;
  corrupted_header[EI_ABIVERSION] ^= 1u;
  EXPECT_FALSE(test::ordinary_wrong_address_output_matches(source, 0, 4, corrupted_header));
}

TEST_P(FaultOrdinaryWrongAddress, EmissionFailureLeavesPristineSourceForRepeatedAttempts) {
  const test::OrdinaryTargetCase &fixture = GetParam();
  std::vector<uint32_t> words = test::ordinary_acquire_words();
  words.insert(words.end(), 33000, build_s_nop(0, fixture.arch).value());
  const std::vector<uint8_t> image = test::make_ordinary_words_elf(fixture, std::move(words));
  const FaultMutationContext source(image);
  FaultMutationError error;
  const FailureOr<OrdinaryAcquireInventory> found =
      discover_ordinary_acquires(source, "test_kernel", error);
  ASSERT_TRUE(found.succeeded()) << error.message;
  const FailureOr<OrdinaryAcquireSelection> selected =
      select_ordinary_acquire(found.value(), {}, error);
  ASSERT_TRUE(selected.succeeded()) << error.message;
  for (int attempt = 0; attempt < 2; ++attempt) {
    SCOPED_TRACE(attempt);
    EXPECT_TRUE(apply_ordinary_wrong_address(source, selected.value(), 4, error).failed());
    EXPECT_EQ(error.code, FaultMutationErrorCode::ValidationFailure);
    EXPECT_TRUE(std::ranges::equal(source.image(), image));
  }
  const FaultMutationContext independent(test::make_ordinary_elf(fixture));
  const FailureOr<OrdinaryAcquireInventory> small =
      discover_ordinary_acquires(independent, "test_kernel", error);
  ASSERT_TRUE(small.succeeded()) << error.message;
  const FailureOr<OrdinaryAcquireSelection> small_selection =
      select_ordinary_acquire(small.value(), {}, error);
  ASSERT_TRUE(small_selection.succeeded()) << error.message;
  const FailureOr<OrdinaryWrongAddressResult> mutant =
      apply_ordinary_wrong_address(independent, small_selection.value(), 4, error);
  ASSERT_TRUE(mutant.succeeded()) << error.message;
  EXPECT_TRUE(test::ordinary_wrong_address_output_matches(independent, 0, 4, mutant.value().image));
}

TEST(FaultOrdinaryWrongAddressCapabilities, AllModernProfilesAdvertiseMatchingMutationRows) {
  const std::vector<OrdinaryWrongAddressCapability> capabilities =
      ordinary_wrong_address_capabilities();
  ASSERT_EQ(capabilities.size(), 6u);
  for (const test::OrdinaryTargetCase &fixture : test::kOrdinaryTargetCases)
    EXPECT_EQ(std::ranges::count_if(capabilities,
                                    [&](const OrdinaryWrongAddressCapability &capability) {
                                      return capability.target == fixture.target &&
                                             capability.wave_size == fixture.wave &&
                                             capability.profile ==
                                                 (fixture.arch == IsaFamily::Cdna5
                                                      ? OrdinaryAcquireProfile::Cdna5GlobalB32
                                                      : OrdinaryAcquireProfile::Rdna4GlobalB32);
                                    }),
              1);
  for (const test::OrdinaryTargetCase &fixture : test::kOrdinaryTargetCases) {
    if (fixture.arch != IsaFamily::Rdna4)
      continue;
    const FaultMutationContext source(test::make_ordinary_elf(fixture));
    FaultMutationError error;
    const FailureOr<OrdinaryAcquireInventory> found =
        discover_ordinary_acquires(source, "test_kernel", error);
    ASSERT_TRUE(found.succeeded()) << error.message;
    ASSERT_EQ(found.value().sites().size(), 1u);
    const FailureOr<OrdinaryAcquireSelection> selected =
        select_ordinary_acquire(found.value(), {}, error);
    ASSERT_TRUE(selected.succeeded()) << error.message;
    const FailureOr<OrdinaryWrongAddressResult> mutant =
        apply_ordinary_wrong_address(source, selected.value(), 4, error);
    ASSERT_TRUE(mutant.succeeded()) << error.message;
    EXPECT_TRUE(test::ordinary_wrong_address_output_matches(source, 0, 4, mutant.value().image));
  }
}

class FaultRdna4OrdinaryParity : public ::testing::TestWithParam<test::OrdinaryTargetCase> {};

TEST_P(FaultRdna4OrdinaryParity, OriginalQualifiedVariantsKeepMutationAndExactIdentity) {
  const test::OrdinaryTargetCase &fixture = GetParam();
  for (const test::OrdinaryAddressParityCase &parity :
       test::rdna4_ordinary_address_parity_cases(fixture)) {
    SCOPED_TRACE(parity.name);
    ASSERT_FALSE(parity.image.empty());
    const FaultMutationContext source(parity.image);
    FaultMutationError error;
    const FailureOr<OrdinaryAcquireInventory> inventory =
        discover_ordinary_acquires(source, parity.kernel, error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    ASSERT_EQ(inventory.value().sites().size(), 1u);
    const FailureOr<OrdinaryAcquireSelection> selected =
        select_ordinary_acquire(inventory.value(), {}, error);
    ASSERT_TRUE(selected.succeeded()) << error.message;
    EXPECT_EQ(selected.value().site().offset, parity.offset);
    const FailureOr<OrdinaryAcquireSelection> exact = select_ordinary_acquire(
        inventory.value(), {.exact = selected.value(), .ordinal = SIZE_MAX}, error);
    ASSERT_TRUE(exact.succeeded()) << error.message;
    const FailureOr<OrdinaryWrongAddressResult> mutant =
        apply_ordinary_wrong_address(source, exact.value(), 4, error);
    ASSERT_TRUE(mutant.succeeded()) << error.message;
    EXPECT_TRUE(test::ordinary_wrong_address_output_matches(source, parity.offset, 4,
                                                            mutant.value().image));
    EXPECT_TRUE(std::ranges::equal(source.image(), parity.image));
  }
}

INSTANTIATE_TEST_SUITE_P(Targets, FaultOrdinaryWrongAddress,
                         ::testing::ValuesIn(test::kOrdinaryTargetCases),
                         [](const ::testing::TestParamInfo<test::OrdinaryTargetCase> &info) {
                           return info.param.name;
                         });

INSTANTIATE_TEST_SUITE_P(Targets, FaultRdna4OrdinaryParity,
                         ::testing::ValuesIn(test::kRdna4OrdinaryAddressCases),
                         [](const ::testing::TestParamInfo<test::OrdinaryTargetCase> &info) {
                           return info.param.name;
                         });

} // namespace
} // namespace rocjitsu
