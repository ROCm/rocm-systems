// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "legacy_ordinary_fixture.h"
#include "legacy_ordinary_oracle.h"
#include "rocjitsu/code/patch/fault_injection/ordinary_wrong_address.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <vector>

namespace rocjitsu {
namespace {

class FaultCdna34OrdinaryAddress : public ::testing::TestWithParam<test::OrdinaryTargetCase> {};

TEST_P(FaultCdna34OrdinaryAddress, FullEightByteGuestPreservesOperandsFlagsWaitAndCache) {
  const test::OrdinaryTargetCase &fixture = GetParam();
  EXPECT_TRUE(std::ranges::any_of(
      ordinary_wrong_address_capabilities(), [&](const OrdinaryWrongAddressCapability &row) {
        return row.target == fixture.target && row.wave_size == fixture.wave &&
               row.profile == OrdinaryAcquireProfile::Cdna34GlobalDword;
      }));
  for (bool modifiers : {false, true}) {
    const std::vector<uint8_t> image = test::make_cdna34_ordinary_elf(fixture, -4, modifiers);
    const FaultMutationContext context(image);
    FaultMutationError error;
    const FailureOr<OrdinaryAcquireInventory> found =
        discover_ordinary_acquires(context, "test_kernel", error);
    ASSERT_TRUE(found.succeeded()) << error.message;
    ASSERT_EQ(found.value().sites().size(), 1u);
    const OrdinaryAcquireSite &site = found.value().sites().front();
    EXPECT_EQ(site.form, "global_load_dword");
    EXPECT_EQ(site.instruction_size, 8u);
    EXPECT_EQ(site.byte_offset, -4);
    EXPECT_FALSE(site.raw_scope);
    EXPECT_EQ(site.profile, OrdinaryAcquireProfile::Cdna34GlobalDword);
    ASSERT_EQ(site.cache_members.size(), 1u);
    EXPECT_EQ(site.cache_members.front().form, "buffer_inv");
    EXPECT_EQ(site.cache_members.front().offset, 12u);
    ASSERT_EQ(site.wait_members.size(), 1u);
    EXPECT_EQ(site.wait_members.front().offset, 8u);
    EXPECT_EQ(site.wait_members.front().form, "s_waitcnt");
    const FailureOr<OrdinaryAcquireSelection> selected =
        select_ordinary_acquire(found.value(), {}, error);
    ASSERT_TRUE(selected.succeeded()) << error.message;
    const FailureOr<OrdinaryAcquireSelection> exact = select_ordinary_acquire(
        found.value(), {.exact = selected.value(), .ordinal = SIZE_MAX}, error);
    ASSERT_TRUE(exact.succeeded()) << error.message;
    const FailureOr<OrdinaryWrongAddressResult> mutant =
        apply_ordinary_wrong_address(context, exact.value(), 8, error);
    ASSERT_TRUE(mutant.succeeded()) << error.message;
    EXPECT_EQ(mutant.value().logical_mutation_count, 1u);
    EXPECT_TRUE(test::legacy_ordinary_output_matches(context, 0, 8, mutant.value().image,
                                                     "global_load_dword"));
    const FailureOr<OrdinaryWrongAddressResult> repeated =
        apply_ordinary_wrong_address(context, selected.value(), 8, error);
    ASSERT_TRUE(repeated.succeeded()) << error.message;
    EXPECT_EQ(repeated.value().image, mutant.value().image);
    EXPECT_TRUE(std::ranges::equal(context.image(), image));
  }
}

TEST_P(FaultCdna34OrdinaryAddress, NativeLiteralSourceAndGuestKeepScratchControlClear) {
  const test::OrdinaryTargetCase &fixture = GetParam();
  for (bool modifiers : {false, true}) {
    SCOPED_TRACE(modifiers);
    const std::array<uint32_t, 2> expected_source =
        modifiers ? std::array<uint32_t, 2>{0xde539ffcu, 0x07040503u}
                  : std::array<uint32_t, 2>{0xde509ffcu, 0x027f0000u};
    const std::array<uint32_t, 2> expected_guest =
        modifiers ? std::array<uint32_t, 2>{0xde538004u, 0x07040503u}
                  : std::array<uint32_t, 2>{0xde508004u, 0x027f0000u};
    const std::vector<uint8_t> image = test::make_cdna34_ordinary_elf(fixture, -4, modifiers);
    const FaultMutationContext context(image);
    ASSERT_EQ(context.source().text_sections().size(), 1u);
    const Section &source_text = *context.source().text_sections().front();
    ASSERT_GE(source_text.size(), sizeof(expected_source));
    std::array<uint32_t, 2> source_words = {};
    std::memcpy(source_words.data(), source_text.data(), sizeof(source_words));
    EXPECT_EQ(source_words, expected_source);
    FaultMutationError error;
    const FailureOr<OrdinaryAcquireInventory> found =
        discover_ordinary_acquires(context, "test_kernel", error);
    ASSERT_TRUE(found.succeeded()) << error.message;
    const FailureOr<OrdinaryAcquireSelection> selected =
        select_ordinary_acquire(found.value(), {}, error);
    ASSERT_TRUE(selected.succeeded()) << error.message;
    const FailureOr<OrdinaryWrongAddressResult> mutant =
        apply_ordinary_wrong_address(context, selected.value(), 8, error);
    ASSERT_TRUE(mutant.succeeded()) << error.message;
    const FaultMutationContext output(mutant.value().image);
    ASSERT_EQ(output.source().text_sections().size(), 1u);
    const Section &text = *output.source().text_sections().front();
    const std::optional<uint64_t> guest = test::ordinary_emitted_guest(text, 0, fixture.target);
    ASSERT_TRUE(guest);
    ASSERT_LE(*guest, text.size());
    ASSERT_GE(text.size() - *guest, sizeof(expected_guest));
    std::array<uint32_t, 2> guest_words = {};
    std::memcpy(guest_words.data(), text.data() + *guest, sizeof(guest_words));
    EXPECT_EQ(guest_words, expected_guest);
    EXPECT_TRUE(test::legacy_ordinary_output_matches(context, 0, 8, mutant.value().image,
                                                     "global_load_dword"));
    EXPECT_TRUE(std::ranges::equal(context.image(), image));
  }
}

TEST_P(FaultCdna34OrdinaryAddress, NativeLoadScopeMustCoverAgentRegardlessOfSc0AndNt) {
  const test::OrdinaryTargetCase &fixture = GetParam();
  for (uint32_t modifiers : {0u, 0x00010000u, 0x00020000u, 0x00030000u}) {
    SCOPED_TRACE(modifiers);
    for (bool agent_scope : {false, true}) {
      SCOPED_TRACE(agent_scope);
      std::vector<uint32_t> words = test::cdna34_ordinary_words(-4, true);
      words[0] = 0xdc509ffcu | modifiers | (agent_scope ? 0x02000000u : 0u);
      const std::vector<uint8_t> image = test::make_ordinary_words_elf(fixture, words);
      const FaultMutationContext context(image);
      FaultMutationError error;
      const FailureOr<OrdinaryAcquireInventory> found =
          discover_ordinary_acquires(context, "test_kernel", error);
      ASSERT_TRUE(found.succeeded()) << error.message;
      if (agent_scope) {
        ASSERT_EQ(found.value().sites().size(), 1u);
        EXPECT_FALSE(found.value().sites().front().raw_scope);
        const FailureOr<OrdinaryAcquireSelection> selected =
            select_ordinary_acquire(found.value(), {}, error);
        ASSERT_TRUE(selected.succeeded()) << error.message;
        const FailureOr<OrdinaryWrongAddressResult> mutant =
            apply_ordinary_wrong_address(context, selected.value(), 8, error);
        ASSERT_TRUE(mutant.succeeded()) << error.message;
        EXPECT_TRUE(test::legacy_ordinary_output_matches(context, 0, 8, mutant.value().image,
                                                         "global_load_dword"));
      } else {
        EXPECT_TRUE(found.value().sites().empty());
        EXPECT_TRUE(select_ordinary_acquire(found.value(), {}, error).failed());
        EXPECT_EQ(error.code, FaultMutationErrorCode::NoCandidates);
      }
      EXPECT_TRUE(std::ranges::equal(context.image(), image));
    }
  }
  for (uint32_t modifiers : {0u, 0x00010000u, 0x00020000u, 0x00030000u}) {
    SCOPED_TRACE(modifiers);
    std::vector<uint32_t> low_scope = test::cdna34_ordinary_words(-4, true);
    low_scope[0] = 0xdc509ffcu | modifiers;
    std::vector<uint32_t> words = low_scope;
    low_scope[0] |= 0x02000000u;
    words.insert(words.end(), low_scope.begin(), low_scope.end());
    const std::vector<uint8_t> image = test::make_ordinary_words_elf(fixture, words);
    const FaultMutationContext context(image);
    FaultMutationError error;
    const FailureOr<OrdinaryAcquireInventory> found =
        discover_ordinary_acquires(context, "test_kernel", error);
    ASSERT_TRUE(found.succeeded()) << error.message;
    ASSERT_EQ(found.value().sites().size(), 1u);
    const FailureOr<OrdinaryAcquireSelection> selected =
        select_ordinary_acquire(found.value(), {}, error);
    ASSERT_TRUE(selected.succeeded()) << error.message;
    EXPECT_EQ(selected.value().site().offset, 20u);
    const FailureOr<OrdinaryAcquireSelection> exact = select_ordinary_acquire(
        found.value(), {.exact = selected.value(), .ordinal = SIZE_MAX}, error);
    ASSERT_TRUE(exact.succeeded()) << error.message;
    EXPECT_EQ(exact.value().site().offset, 20u);
    const FailureOr<OrdinaryWrongAddressResult> mutant =
        apply_ordinary_wrong_address(context, exact.value(), 8, error);
    ASSERT_TRUE(mutant.succeeded()) << error.message;
    EXPECT_TRUE(test::legacy_ordinary_output_matches(context, 20, 8, mutant.value().image,
                                                     "global_load_dword"));
    EXPECT_TRUE(std::ranges::equal(context.image(), image));
  }
}

TEST_P(FaultCdna34OrdinaryAddress, NativeCacheFlagsMustEstablishAnEffectiveAgentAcquire) {
  const test::OrdinaryTargetCase &fixture = GetParam();
  for (uint32_t cache_word : {0xe0a40000u, 0xe0a44000u, 0xe0a48000u, 0xe0a4c000u}) {
    SCOPED_TRACE(cache_word);
    std::vector<uint32_t> words = test::cdna34_ordinary_words(-4, true);
    words[3] = cache_word;
    const std::vector<uint8_t> image = test::make_ordinary_words_elf(fixture, words);
    const FaultMutationContext context(image);
    ASSERT_TRUE(context.source().is_valid());
    const std::array<uint32_t, 2> cache = {cache_word, 0};
    std::unique_ptr<Decoder> decoder =
        Decoder::create(default_isa_target_registry(), fixture.target);
    ASSERT_NE(decoder, nullptr);
    const DecodeResult decoded = decoder->decode_window(cache, 12);
    ASSERT_TRUE(decoded.succeeded());
    EXPECT_EQ(decoded.value()->mnemonic(), "buffer_inv");
    EXPECT_EQ(decoded.value()->size(), 8);
    FaultMutationError error;
    const FailureOr<OrdinaryAcquireInventory> found =
        discover_ordinary_acquires(context, "test_kernel", error);
    ASSERT_TRUE(found.succeeded()) << error.message;
    if (cache_word == 0xe0a40000u || cache_word == 0xe0a44000u) {
      EXPECT_TRUE(found.value().sites().empty());
      EXPECT_TRUE(select_ordinary_acquire(found.value(), {}, error).failed());
      EXPECT_EQ(error.code, FaultMutationErrorCode::NoCandidates);
      EXPECT_TRUE(apply_ordinary_wrong_address(context, {}, 8, error).failed());
      EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
    } else {
      ASSERT_EQ(found.value().sites().size(), 1u);
      const FailureOr<OrdinaryAcquireSelection> selected =
          select_ordinary_acquire(found.value(), {}, error);
      ASSERT_TRUE(selected.succeeded()) << error.message;
      const FailureOr<OrdinaryWrongAddressResult> mutant =
          apply_ordinary_wrong_address(context, selected.value(), 8, error);
      ASSERT_TRUE(mutant.succeeded()) << error.message;
      EXPECT_TRUE(test::legacy_ordinary_output_matches(context, 0, 8, mutant.value().image,
                                                       "global_load_dword"));
    }
    EXPECT_TRUE(std::ranges::equal(context.image(), image));
  }
  std::vector<uint32_t> sequences;
  for (uint32_t cache_word : {0xe0a40000u, 0xe0a44000u, 0xe0a48000u, 0xe0a4c000u}) {
    std::vector<uint32_t> sequence = test::cdna34_ordinary_words(-4, true);
    sequence[3] = cache_word;
    sequences.insert(sequences.end(), sequence.begin(), sequence.end());
  }
  const std::vector<uint8_t> image = test::make_ordinary_words_elf(fixture, sequences);
  const FaultMutationContext context(image);
  FaultMutationError error;
  const FailureOr<OrdinaryAcquireInventory> found =
      discover_ordinary_acquires(context, "test_kernel", error);
  ASSERT_TRUE(found.succeeded()) << error.message;
  ASSERT_EQ(found.value().sites().size(), 2u);
  for (size_t ordinal : {0u, 1u}) {
    const uint64_t expected_offset = 40u + ordinal * 20u;
    const FailureOr<OrdinaryAcquireSelection> selected =
        select_ordinary_acquire(found.value(), {.exact = std::nullopt, .ordinal = ordinal}, error);
    ASSERT_TRUE(selected.succeeded()) << error.message;
    EXPECT_EQ(selected.value().site().offset, expected_offset);
    const FailureOr<OrdinaryAcquireSelection> exact = select_ordinary_acquire(
        found.value(), {.exact = selected.value(), .ordinal = SIZE_MAX}, error);
    ASSERT_TRUE(exact.succeeded()) << error.message;
    const FailureOr<OrdinaryWrongAddressResult> mutant =
        apply_ordinary_wrong_address(context, exact.value(), 8, error);
    ASSERT_TRUE(mutant.succeeded()) << error.message;
    EXPECT_TRUE(test::legacy_ordinary_output_matches(context, expected_offset, 8,
                                                     mutant.value().image, "global_load_dword"));
  }
  EXPECT_TRUE(std::ranges::equal(context.image(), image));
}

TEST_P(FaultCdna34OrdinaryAddress, ParameterAlignmentAndSigned13OverflowAreCheckedWithoutClamping) {
  for (int32_t offset : {-4096, 4092, 2}) {
    const std::vector<uint8_t> image = test::make_cdna34_ordinary_elf(GetParam(), offset);
    const FaultMutationContext context(image);
    FaultMutationError error;
    const FailureOr<OrdinaryAcquireInventory> found =
        discover_ordinary_acquires(context, "test_kernel", error);
    ASSERT_TRUE(found.succeeded()) << error.message;
    const FailureOr<OrdinaryAcquireSelection> selected =
        select_ordinary_acquire(found.value(), {}, error);
    ASSERT_TRUE(selected.succeeded()) << error.message;
    const FailureOr<OrdinaryWrongAddressResult> result =
        apply_ordinary_wrong_address(context, selected.value(), 4, error);
    if (offset == -4096) {
      ASSERT_TRUE(result.succeeded()) << error.message;
      const FailureOr<OrdinaryWrongAddressResult> full_span =
          apply_ordinary_wrong_address(context, selected.value(), 8188, error);
      ASSERT_TRUE(full_span.succeeded()) << error.message;
      EXPECT_TRUE(test::legacy_ordinary_output_matches(context, 0, 8188, full_span.value().image,
                                                       "global_load_dword"));
      EXPECT_TRUE(test::legacy_ordinary_output_matches(context, 0, 4, result.value().image,
                                                       "global_load_dword"));
    } else {
      EXPECT_TRUE(result.failed());
      EXPECT_EQ(error.code, FaultMutationErrorCode::InvalidParameters);
    }
    for (const std::optional<uint32_t> delta :
         {std::optional<uint32_t>{}, std::optional<uint32_t>{0}, std::optional<uint32_t>{2},
          std::optional<uint32_t>{8192}, std::optional<uint32_t>{UINT32_MAX}}) {
      EXPECT_TRUE(apply_ordinary_wrong_address(context, selected.value(), delta, error).failed());
      EXPECT_EQ(error.code, FaultMutationErrorCode::InvalidParameters);
    }
    EXPECT_TRUE(std::ranges::equal(context.image(), image));
  }
}

TEST_P(FaultCdna34OrdinaryAddress, LegacyWaitMustActuallyDrainVmcntAndInterveningEffectsReject) {
  const test::OrdinaryTargetCase &fixture = GetParam();
  const std::array<uint32_t, 2> load = test::cdna34_ordinary_load();
  const std::array<uint32_t, 2> cache = test::kCdna34OrdinaryCache;
  const std::vector<std::vector<uint32_t>> rejected = {
      {load[0], load[1], cache[0], cache[1]},
      {load[0], load[1], test::kCdna34OrdinaryWait},
      {load[0], load[1], test::kCdna34OrdinaryWait | 1u, cache[0], cache[1]},
      {load[0], load[1], 0xbf8cffffu, cache[0], cache[1]},
      {load[0], load[1], test::kCdna34OrdinaryWait, 0xde708000u, 0x00000100u, cache[0], cache[1]},
      {load[0], load[1], test::kCdna34OrdinaryWait, 0xbf8a0000u, cache[0], cache[1]},
      {load[0], load[1], test::kCdna34OrdinaryWait, build_s_branch(0, fixture.arch).value(),
       cache[0], cache[1]},
      {0xde708000u, 0x00000100u, test::kCdna34OrdinaryWait, cache[0], cache[1]},
      {load[0] & ~(3u << 14), load[1], test::kCdna34OrdinaryWait, cache[0], cache[1]},
  };
  for (size_t index = 0; index < rejected.size(); ++index) {
    SCOPED_TRACE(index);
    const FaultMutationContext context(test::make_ordinary_words_elf(fixture, rejected[index]));
    FaultMutationError error;
    const FailureOr<OrdinaryAcquireInventory> found =
        discover_ordinary_acquires(context, "test_kernel", error);
    ASSERT_TRUE(found.succeeded()) << error.message;
    EXPECT_TRUE(found.value().sites().empty());
  }
}

TEST_P(FaultCdna34OrdinaryAddress, QualifiedOrdinalsAndIndependentMutantsRemainSourceBased) {
  std::vector<uint32_t> words;
  for (size_t ordinal : {0u, 1u, 2u}) {
    std::vector<uint32_t> sequence = test::cdna34_ordinary_words(ordinal == 2 ? -4 : 4);
    if (ordinal == 0)
      sequence[2] |= 1u;
    words.insert(words.end(), sequence.begin(), sequence.end());
  }
  const FaultMutationContext context(test::make_ordinary_words_elf(GetParam(), std::move(words)));
  FaultMutationError error;
  const FailureOr<OrdinaryAcquireInventory> found =
      discover_ordinary_acquires(context, "test_kernel", error);
  ASSERT_TRUE(found.succeeded()) << error.message;
  ASSERT_EQ(found.value().sites().size(), 2u);
  for (size_t ordinal : {0u, 1u}) {
    const FailureOr<OrdinaryAcquireSelection> selected =
        select_ordinary_acquire(found.value(), {.exact = std::nullopt, .ordinal = ordinal}, error);
    ASSERT_TRUE(selected.succeeded()) << error.message;
    EXPECT_EQ(selected.value().site().offset, 20u * (ordinal + 1));
    const FailureOr<OrdinaryWrongAddressResult> mutant =
        apply_ordinary_wrong_address(context, selected.value(), 8, error);
    ASSERT_TRUE(mutant.succeeded()) << error.message;
    EXPECT_TRUE(test::legacy_ordinary_output_matches(context, 20u * (ordinal + 1), 8,
                                                     mutant.value().image, "global_load_dword"));
  }
  EXPECT_TRUE(select_ordinary_acquire(found.value(),
                                      {.exact = OrdinaryAcquireSelection{}, .ordinal = 0}, error)
                  .failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  EXPECT_TRUE(select_ordinary_acquire(found.value(), {.exact = std::nullopt, .ordinal = 2}, error)
                  .failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::OrdinalOutOfRange);
}

TEST_P(FaultCdna34OrdinaryAddress, BoundedFallthroughAndPollingPathsKeepTheAcquireProfile) {
  const test::OrdinaryTargetCase &fixture = GetParam();
  const std::array<uint32_t, 2> load = test::cdna34_ordinary_load();
  const std::array<uint32_t, 2> cache = test::kCdna34OrdinaryCache;
  const std::array<std::vector<uint32_t>, 2> paths = {{
      {test::ordinary_vccz_branch(3, fixture.arch), load[0], load[1], test::kCdna34OrdinaryWait,
       cache[0], cache[1]},
      {load[0], load[1], test::kCdna34OrdinaryWait, test::ordinary_vccz_branch(-4, fixture.arch),
       cache[0], cache[1]},
  }};
  for (size_t path : {0u, 1u}) {
    const FaultMutationContext context(test::make_ordinary_words_elf(fixture, paths[path]));
    FaultMutationError error;
    const FailureOr<OrdinaryAcquireInventory> found =
        discover_ordinary_acquires(context, "test_kernel", error);
    ASSERT_TRUE(found.succeeded()) << error.message;
    ASSERT_EQ(found.value().sites().size(), 1u);
    EXPECT_EQ(found.value().sites().front().path, path == 0
                                                      ? OrdinaryAcquirePath::ExactFallthroughJoin
                                                      : OrdinaryAcquirePath::ExactSelfLoopExit);
    const FailureOr<OrdinaryAcquireSelection> selected =
        select_ordinary_acquire(found.value(), {}, error);
    ASSERT_TRUE(selected.succeeded()) << error.message;
    const FailureOr<OrdinaryWrongAddressResult> mutant =
        apply_ordinary_wrong_address(context, selected.value(), 4, error);
    ASSERT_TRUE(mutant.succeeded()) << error.message;
    EXPECT_TRUE(test::legacy_ordinary_output_matches(context, path == 0 ? 4 : 0, 4,
                                                     mutant.value().image, "global_load_dword"));
  }
}

TEST_P(FaultCdna34OrdinaryAddress,
       RealSharedContainerNeedsEqualCompleteOwnersAndExactRequestedOwner) {
  const test::OrdinaryTargetCase &fixture = GetParam();
  const std::vector<uint32_t> body = test::cdna34_ordinary_words();
  const FaultMutationContext context(test::make_legacy_ordinary_shared_elf(fixture, body, 12));
  FaultMutationError error;
  const FailureOr<OrdinaryAcquireInventory> first =
      discover_ordinary_acquires(context, "test_kernel0", error);
  ASSERT_TRUE(first.succeeded()) << error.message;
  const FailureOr<OrdinaryAcquireInventory> second =
      discover_ordinary_acquires(context, "test_kernel1", error);
  ASSERT_TRUE(second.succeeded()) << error.message;
  ASSERT_EQ(first.value().sites().size(), 1u);
  ASSERT_EQ(second.value().sites().size(), 1u);
  const OrdinaryAcquireSite &site = first.value().sites().front();
  EXPECT_EQ(site.container_name, "function@512");
  EXPECT_EQ(site.container_entry, 512u);
  EXPECT_EQ(site.container_size, 24u);
  ASSERT_EQ(site.affected_owners.size(), 2u);
  const FailureOr<OrdinaryAcquireSelection> selected =
      select_ordinary_acquire(first.value(), {}, error);
  ASSERT_TRUE(selected.succeeded()) << error.message;
  const FailureOr<OrdinaryAcquireSelection> shared = select_ordinary_acquire(
      second.value(), {.exact = selected.value(), .ordinal = SIZE_MAX}, error);
  ASSERT_TRUE(shared.succeeded()) << error.message;
  ASSERT_NE(shared.value().owner(), nullptr);
  EXPECT_EQ(shared.value().owner()->entry, "test_kernel1");
  const FailureOr<OrdinaryWrongAddressResult> mutant =
      apply_ordinary_wrong_address(context, shared.value(), 4, error);
  ASSERT_TRUE(mutant.succeeded()) << error.message;
  EXPECT_TRUE(test::legacy_ordinary_output_matches(context, 512, 4, mutant.value().image,
                                                   "global_load_dword"));
  for (bool mismatch : {false, true}) {
    const FaultMutationContext isolated(
        test::make_legacy_ordinary_shared_elf(fixture, body, 12, mismatch, mismatch));
    const FailureOr<OrdinaryAcquireInventory> found =
        discover_ordinary_acquires(isolated, mismatch ? "test_kernel0" : "test_kernel1", error);
    ASSERT_TRUE(found.succeeded()) << error.message;
    EXPECT_TRUE(found.value().sites().empty());
    EXPECT_TRUE(
        select_ordinary_acquire(found.value(), {.exact = selected.value(), .ordinal = 0}, error)
            .failed());
  }
  // The launch helper reports64 for every CDNA descriptor; reject the raw
  // unsupported Wave32 property for both the selected and affected owners.
  std::vector<uint8_t> selected_wave32 = test::make_cdna34_ordinary_elf(fixture);
  ASSERT_TRUE(test::mark_legacy_ordinary_wave32(selected_wave32, "test_kernel"));
  const FaultMutationContext invalid_selected(selected_wave32);
  EXPECT_TRUE(discover_ordinary_acquires(invalid_selected, "test_kernel", error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm);
  EXPECT_TRUE(std::ranges::equal(invalid_selected.image(), selected_wave32));
  for (bool shared_sibling : {false, true}) {
    std::vector<uint8_t> sibling_wave32 =
        test::make_legacy_ordinary_shared_elf(fixture, body, 12, shared_sibling);
    ASSERT_TRUE(test::mark_legacy_ordinary_wave32(sibling_wave32, "test_kernel1"));
    const FaultMutationContext invalid_sibling(sibling_wave32);
    const FailureOr<OrdinaryAcquireInventory> sibling_inventory =
        discover_ordinary_acquires(invalid_sibling, "test_kernel0", error);
    if (shared_sibling) {
      EXPECT_TRUE(sibling_inventory.failed());
      EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm);
    } else {
      ASSERT_TRUE(sibling_inventory.succeeded()) << error.message;
      ASSERT_EQ(sibling_inventory.value().sites().size(), 1u);
    }
    EXPECT_TRUE(discover_ordinary_acquires(invalid_sibling, "test_kernel1", error).failed());
    EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm);
    EXPECT_TRUE(std::ranges::equal(invalid_sibling.image(), sibling_wave32));
  }
  const FaultMutationContext split(
      test::make_legacy_ordinary_split_container_elf(fixture, body, 12));
  const FailureOr<OrdinaryAcquireInventory> split_found =
      discover_ordinary_acquires(split, "test_kernel", error);
  ASSERT_TRUE(split_found.succeeded()) << error.message;
  EXPECT_TRUE(split_found.value().sites().empty());
}

TEST_P(FaultCdna34OrdinaryAddress, CrossContextExpiredHandlesAndInputMutationCannotChangeSource) {
  std::vector<uint8_t> image = test::make_cdna34_ordinary_elf(GetParam());
  const std::vector<uint8_t> pristine = image;
  const FaultMutationContext context(image), other(image);
  image[0] = 0;
  FaultMutationError error;
  const FailureOr<OrdinaryAcquireInventory> found =
      discover_ordinary_acquires(context, "test_kernel", error);
  ASSERT_TRUE(found.succeeded()) << error.message;
  const FailureOr<OrdinaryAcquireSelection> selected =
      select_ordinary_acquire(found.value(), {}, error);
  ASSERT_TRUE(selected.succeeded()) << error.message;
  EXPECT_TRUE(apply_ordinary_wrong_address(other, selected.value(), 4, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  std::optional<OrdinaryAcquireSelection> expired;
  {
    const FaultMutationContext temporary(pristine);
    const FailureOr<OrdinaryAcquireInventory> inventory =
        discover_ordinary_acquires(temporary, "test_kernel", error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    const FailureOr<OrdinaryAcquireSelection> handle =
        select_ordinary_acquire(inventory.value(), {}, error);
    ASSERT_TRUE(handle.succeeded()) << error.message;
    expired = handle.value();
  }
  EXPECT_EQ(expired->site().form, "global_load_dword");
  EXPECT_EQ(expired->site().cache_members.front().form, "buffer_inv");
  EXPECT_TRUE(apply_ordinary_wrong_address(context, *expired, 4, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  EXPECT_TRUE(std::ranges::equal(context.image(), pristine));
}

TEST_P(FaultCdna34OrdinaryAddress,
       IndependentOracleRejectsOffsetOperandWaitCacheAndContinuationCorruption) {
  const FaultMutationContext context(test::make_cdna34_ordinary_elf(GetParam()));
  FaultMutationError error;
  const FailureOr<OrdinaryAcquireInventory> found =
      discover_ordinary_acquires(context, "test_kernel", error);
  ASSERT_TRUE(found.succeeded()) << error.message;
  const FailureOr<OrdinaryAcquireSelection> selected =
      select_ordinary_acquire(found.value(), {}, error);
  ASSERT_TRUE(selected.succeeded()) << error.message;
  const FailureOr<OrdinaryWrongAddressResult> mutant =
      apply_ordinary_wrong_address(context, selected.value(), 4, error);
  ASSERT_TRUE(mutant.succeeded()) << error.message;
  const AmdGpuCodeObject output(mutant.value().image.data(), mutant.value().image.size());
  const Section &text = *output.text_sections().front();
  const std::optional<uint64_t> guest = test::ordinary_emitted_guest(text, 0, GetParam().target);
  ASSERT_TRUE(guest);
  for (uint64_t byte : {*guest, *guest + 4, uint64_t{8}, uint64_t{12}, *guest + 8}) {
    std::vector<uint8_t> corrupt = mutant.value().image;
    corrupt[text.sectionOffset() + byte] ^= 1;
    EXPECT_FALSE(test::legacy_ordinary_output_matches(context, 0, 4, corrupt, "global_load_dword"));
  }
}

INSTANTIATE_TEST_SUITE_P(Targets, FaultCdna34OrdinaryAddress,
                         ::testing::ValuesIn(test::kCdna34OrdinaryCases),
                         [](const ::testing::TestParamInfo<test::OrdinaryTargetCase> &info) {
                           return info.param.name;
                         });

} // namespace
} // namespace rocjitsu
