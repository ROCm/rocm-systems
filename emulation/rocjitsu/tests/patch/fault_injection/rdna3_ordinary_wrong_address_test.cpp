// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "legacy_ordinary_oracle.h"
#include "rdna3_ordinary_fixture.h"
#include "rocjitsu/code/analysis/instruction_boundaries.h"
#include "rocjitsu/code/patch/fault_injection/ordinary_wrong_address.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/target_registry.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace rocjitsu {
namespace {

class FaultRdna3OrdinaryAddress : public ::testing::TestWithParam<test::OrdinaryTargetCase> {};

TEST_P(FaultRdna3OrdinaryAddress, FullEightByteGuestPreservesOperandsFlagsWaitAndCache) {
  const test::OrdinaryTargetCase &fixture = GetParam();
  EXPECT_TRUE(std::ranges::any_of(
      ordinary_wrong_address_capabilities(), [&](const OrdinaryWrongAddressCapability &row) {
        return row.target == fixture.target && row.wave_size == fixture.wave &&
               row.profile == OrdinaryAcquireProfile::Rdna3GlobalB32Pair;
      }));
  for (bool modifiers : {false, true}) {
    const std::vector<uint8_t> image = test::make_rdna3_ordinary_elf(fixture, -4, modifiers);
    const FaultMutationContext context(image);
    FaultMutationError error;
    const FailureOr<OrdinaryAcquireInventory> found =
        discover_ordinary_acquires(context, "test_kernel", error);
    ASSERT_TRUE(found.succeeded()) << error.message;
    ASSERT_EQ(found.value().sites().size(), 1u);
    const OrdinaryAcquireSite &site = found.value().sites().front();
    EXPECT_EQ(site.form, "global_load_b32");
    EXPECT_EQ(site.instruction_size, 8u);
    EXPECT_EQ(site.byte_offset, -4);
    EXPECT_FALSE(site.raw_scope);
    EXPECT_EQ(site.profile, OrdinaryAcquireProfile::Rdna3GlobalB32Pair);
    ASSERT_EQ(site.cache_members.size(), 2u);
    EXPECT_EQ(site.cache_members[1].form, "buffer_gl0_inv");
    EXPECT_EQ(site.cache_members[1].offset, 20u);
    EXPECT_EQ(site.cache_members.front().form, "buffer_gl1_inv");
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
                                                     "global_load_b32"));
    const FailureOr<OrdinaryWrongAddressResult> repeated =
        apply_ordinary_wrong_address(context, selected.value(), 8, error);
    ASSERT_TRUE(repeated.succeeded()) << error.message;
    EXPECT_EQ(repeated.value().image, mutant.value().image);
    EXPECT_TRUE(std::ranges::equal(context.image(), image));
  }
}

TEST_P(FaultRdna3OrdinaryAddress, ParameterAlignmentAndSigned13OverflowAreCheckedWithoutClamping) {
  for (int32_t offset : {-4096, 4092, 2}) {
    const std::vector<uint8_t> image = test::make_rdna3_ordinary_elf(GetParam(), offset);
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
                                                       "global_load_b32"));
      EXPECT_TRUE(test::legacy_ordinary_output_matches(context, 0, 4, result.value().image,
                                                       "global_load_b32"));
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

TEST_P(FaultRdna3OrdinaryAddress, ExactPairRejectsMissingReversedCrossedOrInterveningMembers) {
  const test::OrdinaryTargetCase &fixture = GetParam();
  const std::array<uint32_t, 2> load = test::rdna3_ordinary_load();
  const std::array<uint32_t, 4> cache = test::kRdna3OrdinaryCache;
  const auto make = [&](std::vector<uint32_t> suffix, uint32_t wait = test::kRdna3OrdinaryWait) {
    std::vector<uint32_t> words{load[0], load[1], wait};
    words.insert(words.end(), suffix.begin(), suffix.end());
    return words;
  };
  std::vector<uint32_t> malformed = test::rdna3_ordinary_words();
  malformed[0] |= 1u << 25;
  const std::vector<std::vector<uint32_t>> rejected = {
      {load[0], load[1], cache[0], cache[1], cache[2], cache[3]},
      {load[0], load[1], test::kRdna3OrdinaryWait},
      make({cache[0], cache[1]}),
      make({cache[2], cache[3]}),
      make({cache[2], cache[3], cache[0], cache[1]}),
      make({cache[0], cache[1], cache[0], cache[1], cache[2], cache[3]}),
      make({cache[0], cache[1], build_s_nop(0, fixture.arch).value(), cache[2], cache[3]}),
      make({cache[0], cache[1], build_s_branch(0, fixture.arch).value(), cache[2], cache[3]}),
      make({cache[0], cache[1], 0xbfbd0000u, cache[2], cache[3]}),
      make({0xdc6a0000u, 0x00000100u, cache[0], cache[1], cache[2], cache[3]}),
      make({cache[0], cache[1], cache[2], cache[3]}, test::kRdna3OrdinaryWait | (1u << 10)),
      make({cache[0], cache[1], cache[2], cache[3]}, 0xbf89ffffu),
      {0xdc6a0000u, 0x00000100u, test::kRdna3OrdinaryWait, cache[0], cache[1], cache[2], cache[3]},
      {load[0] & ~(3u << 16), load[1], test::kRdna3OrdinaryWait, cache[0], cache[1], cache[2],
       cache[3]},
      malformed,
  };
  for (size_t index = 0; index < rejected.size(); ++index) {
    SCOPED_TRACE(index);
    const FaultMutationContext context(test::make_ordinary_words_elf(fixture, rejected[index]));
    FaultMutationError error;
    const FailureOr<OrdinaryAcquireInventory> found =
        discover_ordinary_acquires(context, "test_kernel", error);
    if (index == rejected.size() - 1 && found.failed())
      continue; // malformed padding may fail BI admission before FI query.
    ASSERT_TRUE(found.succeeded()) << error.message;
    EXPECT_TRUE(found.value().sites().empty());
  }
}

TEST_P(FaultRdna3OrdinaryAddress, QualifiedOrdinalsAndIndependentMutantsRemainSourceBased) {
  std::vector<uint32_t> words;
  for (size_t ordinal : {0u, 1u, 2u}) {
    std::vector<uint32_t> sequence = test::rdna3_ordinary_words(ordinal == 2 ? -4 : 4);
    if (ordinal == 0)
      sequence[2] |= 1u << 10;
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
    EXPECT_EQ(selected.value().site().offset, 28u * (ordinal + 1));
    const FailureOr<OrdinaryWrongAddressResult> mutant =
        apply_ordinary_wrong_address(context, selected.value(), 8, error);
    ASSERT_TRUE(mutant.succeeded()) << error.message;
    EXPECT_TRUE(test::legacy_ordinary_output_matches(context, 28u * (ordinal + 1), 8,
                                                     mutant.value().image, "global_load_b32"));
  }
  EXPECT_TRUE(select_ordinary_acquire(found.value(),
                                      {.exact = OrdinaryAcquireSelection{}, .ordinal = 0}, error)
                  .failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  EXPECT_TRUE(select_ordinary_acquire(found.value(), {.exact = std::nullopt, .ordinal = 2}, error)
                  .failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::OrdinalOutOfRange);
}

TEST_P(FaultRdna3OrdinaryAddress, BoundedFallthroughAndPollingPathsKeepTheAcquireProfile) {
  const test::OrdinaryTargetCase &fixture = GetParam();
  const std::array<uint32_t, 2> load = test::rdna3_ordinary_load();
  const std::array<uint32_t, 4> cache = test::kRdna3OrdinaryCache;
  const std::array<std::vector<uint32_t>, 2> paths = {{
      {test::ordinary_vccz_branch(3, fixture.arch), load[0], load[1], test::kRdna3OrdinaryWait,
       cache[0], cache[1], cache[2], cache[3]},
      {load[0], load[1], test::kRdna3OrdinaryWait, test::ordinary_vccz_branch(-4, fixture.arch),
       cache[0], cache[1], cache[2], cache[3]},
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
                                                     mutant.value().image, "global_load_b32"));
  }
}

TEST_P(FaultRdna3OrdinaryAddress,
       RealSharedContainerNeedsEqualCompleteOwnersAndExactRequestedOwner) {
  const test::OrdinaryTargetCase &fixture = GetParam();
  const std::vector<uint32_t> body = test::rdna3_ordinary_words();
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
  EXPECT_EQ(site.container_size, 32u);
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
                                                   "global_load_b32"));
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
  const FaultMutationContext split(
      test::make_legacy_ordinary_split_container_elf(fixture, body, 12));
  const FailureOr<OrdinaryAcquireInventory> split_found =
      discover_ordinary_acquires(split, "test_kernel", error);
  ASSERT_TRUE(split_found.succeeded()) << error.message;
  EXPECT_TRUE(split_found.value().sites().empty());
}

TEST_P(FaultRdna3OrdinaryAddress, CrossContextExpiredHandlesAndInputMutationCannotChangeSource) {
  std::vector<uint8_t> image = test::make_rdna3_ordinary_elf(GetParam());
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
  EXPECT_EQ(expired->site().form, "global_load_b32");
  EXPECT_EQ(expired->site().cache_members.front().form, "buffer_gl1_inv");
  EXPECT_TRUE(apply_ordinary_wrong_address(context, *expired, 4, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  EXPECT_TRUE(std::ranges::equal(context.image(), pristine));
}

TEST_P(FaultRdna3OrdinaryAddress,
       IndependentOracleRejectsOffsetOperandWaitCacheAndContinuationCorruption) {
  const FaultMutationContext context(test::make_rdna3_ordinary_elf(GetParam()));
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
  for (uint64_t byte : {*guest, *guest + 4, uint64_t{8}, uint64_t{12}, uint64_t{20}, *guest + 8}) {
    std::vector<uint8_t> corrupt = mutant.value().image;
    corrupt[text.sectionOffset() + byte] ^= 1;
    EXPECT_FALSE(test::legacy_ordinary_output_matches(context, 0, 4, corrupt, "global_load_b32"));
  }
}

TEST_P(FaultRdna3OrdinaryAddress, WaitOnlyPairGapCanExtendPastTheFirstCacheBound) {
  for (size_t pair_waits : {1u, 17u}) {
    const FaultMutationContext context(
        test::make_rdna3_ordinary_elf(GetParam(), 4, false, pair_waits));
    FaultMutationError error;
    const FailureOr<OrdinaryAcquireInventory> found =
        discover_ordinary_acquires(context, "test_kernel", error);
    ASSERT_TRUE(found.succeeded()) << error.message;
    ASSERT_EQ(found.value().sites().size(), 1u);
    const OrdinaryAcquireSite &site = found.value().sites().front();
    ASSERT_EQ(site.cache_members.size(), 2u);
    EXPECT_EQ(site.cache_members[0].offset, 12u);
    EXPECT_EQ(site.cache_members[1].offset, 20 + pair_waits * 4);
    EXPECT_EQ(site.wait_members.size(), pair_waits + 1);
    const FailureOr<OrdinaryAcquireSelection> selected =
        select_ordinary_acquire(found.value(), {}, error);
    ASSERT_TRUE(selected.succeeded()) << error.message;
    const FailureOr<OrdinaryWrongAddressResult> result =
        apply_ordinary_wrong_address(context, selected.value(), 4, error);
    ASSERT_TRUE(result.succeeded()) << error.message;
    EXPECT_TRUE(test::legacy_ordinary_output_matches(context, 0, 4, result.value().image,
                                                     "global_load_b32"));
  }
}

TEST_P(FaultRdna3OrdinaryAddress, OriginalSingleFenceMetadataPrecedenceSuppressesPairFallback) {
  for (bool after_pair : {false, true}) {
    std::vector<uint32_t> words = test::rdna3_ordinary_words();
    const std::vector<uint32_t>::iterator where = after_pair ? words.end() : words.begin() + 3;
    words.insert(where, {0xf4840000u, 0}); // actual RDNA3 s_dcache_inv
    const FaultMutationContext context(test::make_ordinary_words_elf(GetParam(), std::move(words)));
    FaultMutationError error;
    const FailureOr<OrdinaryAcquireInventory> found =
        discover_ordinary_acquires(context, "test_kernel", error);
    ASSERT_TRUE(found.succeeded()) << error.message;
    EXPECT_TRUE(found.value().sites().empty());
  }
  // Conservative Single metadata suppresses pair fallback even when a scalar
  // clause prevents that Single from being an exact edit/path member.
  std::vector<uint32_t> clause_words = test::rdna3_ordinary_words();
  clause_words.insert(clause_words.end(), {0xbf850000u, 0xf4840000u, 0});
  const std::vector<uint8_t> clause_image = test::make_ordinary_words_elf(GetParam(), clause_words);
  const FaultMutationContext clause_context(clause_image);
  const Section &clause_text = *clause_context.source().text_sections().front();
  std::array<uint32_t, 3> native_clause = {};
  std::memcpy(native_clause.data(), clause_text.data() + 28, sizeof(native_clause));
  EXPECT_EQ(native_clause, (std::array<uint32_t, 3>{0xbf850000u, 0xf4840000u, 0}));
  std::unique_ptr<Decoder> clause_decoder =
      Decoder::create(default_isa_target_registry(), GetParam().target);
  ASSERT_NE(clause_decoder, nullptr);
  const DecodeResult decoded_clause =
      clause_decoder->decode_window(std::span(clause_words).subspan(7), 28);
  const DecodeResult decoded_single =
      clause_decoder->decode_window(std::span(clause_words).subspan(8), 32);
  ASSERT_TRUE(decoded_clause.succeeded());
  ASSERT_TRUE(decoded_single.succeeded());
  EXPECT_EQ(decoded_clause.value()->mnemonic(), "s_clause");
  EXPECT_EQ(decoded_clause.value()->size(), 4);
  EXPECT_EQ(decoded_single.value()->mnemonic(), "s_dcache_inv");
  EXPECT_EQ(decoded_single.value()->size(), 8);
  InstructionBoundaries clause_boundaries;
  clause_boundaries.record(*decoded_clause.value(), 0);
  clause_boundaries.record(*decoded_single.value(), 0);
  const InstructionBoundaries::Entry *single_boundary = clause_boundaries.find(32);
  ASSERT_NE(single_boundary, nullptr);
  ASSERT_TRUE(single_boundary->clause_member);
  FaultMutationError clause_error;
  const FailureOr<OrdinaryAcquireInventory> clause_inventory =
      discover_ordinary_acquires(clause_context, "test_kernel", clause_error);
  ASSERT_TRUE(clause_inventory.succeeded()) << clause_error.message;
  EXPECT_TRUE(clause_inventory.value().sites().empty());
  EXPECT_TRUE(select_ordinary_acquire(clause_inventory.value(), {}, clause_error).failed());
  EXPECT_EQ(clause_error.code, FaultMutationErrorCode::NoCandidates);
  EXPECT_TRUE(
      apply_ordinary_wrong_address(clause_context, OrdinaryAcquireSelection{}, 8, clause_error)
          .failed());
  EXPECT_EQ(clause_error.code, FaultMutationErrorCode::BadIdentity);
  EXPECT_TRUE(std::ranges::equal(clause_context.image(), clause_image));

  // A second complete pair after the first cannot be another bounded match:
  // the first pair itself blocks the path to the later prefix.
  std::vector<uint32_t> words = test::rdna3_ordinary_words();
  words.insert(words.end(), test::kRdna3OrdinaryCache.begin(), test::kRdna3OrdinaryCache.end());
  const FaultMutationContext context(test::make_ordinary_words_elf(GetParam(), std::move(words)));
  FaultMutationError error;
  const FailureOr<OrdinaryAcquireInventory> found =
      discover_ordinary_acquires(context, "test_kernel", error);
  ASSERT_TRUE(found.succeeded()) << error.message;
  ASSERT_EQ(found.value().sites().size(), 1u);
  EXPECT_EQ(found.value().sites().front().cache_members[0].offset, 12u);
}

TEST_P(FaultRdna3OrdinaryAddress, PairCompletionNeedsTheSameOwnersContainerAndBlockAsItsPrefix) {
  const std::vector<uint32_t> body = test::rdna3_ordinary_words();
  const FaultMutationContext owners(
      test::make_legacy_ordinary_shared_elf(GetParam(), body, 20, true, true));
  FaultMutationError error;
  const FailureOr<OrdinaryAcquireInventory> owner_found =
      discover_ordinary_acquires(owners, "test_kernel0", error);
  ASSERT_TRUE(owner_found.succeeded()) << error.message;
  EXPECT_TRUE(owner_found.value().sites().empty());
  const FaultMutationContext split(
      test::make_legacy_ordinary_split_container_elf(GetParam(), body, 20));
  const FailureOr<OrdinaryAcquireInventory> split_found =
      discover_ordinary_acquires(split, "test_kernel", error);
  ASSERT_TRUE(split_found.succeeded()) << error.message;
  EXPECT_TRUE(split_found.value().sites().empty());
  // The inactive branch targets the tail, splitting only that member into a
  // new block. A fallthrough load-to-prefix proof alone cannot admit the pair.
  std::vector<uint32_t> split_block{test::ordinary_vccz_branch(5, GetParam().arch)};
  split_block.insert(split_block.end(), body.begin(), body.end());
  const FaultMutationContext branch(
      test::make_ordinary_words_elf(GetParam(), std::move(split_block)));
  const FailureOr<OrdinaryAcquireInventory> branch_found =
      discover_ordinary_acquires(branch, "test_kernel", error);
  ASSERT_TRUE(branch_found.succeeded()) << error.message;
  EXPECT_TRUE(branch_found.value().sites().empty());
}

TEST_P(FaultRdna3OrdinaryAddress, NativeGlobalFixtureRetainsValidIgnoredDataWithoutScratchEnable) {
  for (bool modifiers : {false, true}) {
    const std::array<uint32_t, 2> words = test::rdna3_ordinary_load(-4, modifiers);
    // Native GLOBAL never sets scratch-only SVE. VDATA five is valid and ignored by a load.
    const std::array<uint32_t, 2> original = {modifiers ? 0xdc52fffcu : 0xdc525ffcu,
                                              modifiers ? 0x07040503u : 0x027c0000u};
    ASSERT_EQ(words, original);
    const FaultMutationContext context(test::make_rdna3_ordinary_elf(GetParam(), -4, modifiers));
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
    EXPECT_TRUE(test::legacy_ordinary_output_matches(context, 0, 8, mutant.value().image,
                                                     "global_load_b32"));
    const AmdGpuCodeObject output(mutant.value().image.data(), mutant.value().image.size());
    ASSERT_TRUE(output.is_valid());
    const Section &text = *output.text_sections().front();
    const std::optional<uint64_t> guest = test::ordinary_emitted_guest(text, 0, GetParam().target);
    ASSERT_TRUE(guest);
    ASSERT_LE(*guest + sizeof(words), text.size());
    std::array<uint32_t, 2> actual = {};
    std::memcpy(actual.data(), text.data() + *guest, sizeof(actual));
    const std::array<uint32_t, 2> replacement = {modifiers ? 0xdc52e004u : 0xdc524004u,
                                                 original[1]};
    EXPECT_EQ(actual, replacement);
  }
}

TEST_P(FaultRdna3OrdinaryAddress, GlcGatesAgentAcquireWithoutRestrictingSlcDlc) {
  // Native assembler source/replacement words for all independent SLC/DLC combinations.
  constexpr std::array<std::array<uint32_t, 2>, 4> offsets = {{
      {0xdc525ffcu, 0xdc524004u},
      {0xdc527ffcu, 0xdc526004u},
      {0xdc52dffcu, 0xdc52c004u},
      {0xdc52fffcu, 0xdc52e004u},
  }};
  for (const std::array<uint32_t, 2> &golden : offsets) {
    SCOPED_TRACE(golden[0]);
    for (uint32_t data : {0u, 5u}) {
      SCOPED_TRACE(data);
      for (bool glc : {false, true}) {
        SCOPED_TRACE(glc);
        std::vector<uint32_t> words = test::rdna3_ordinary_words(-4);
        words[0] = glc ? golden[0] : golden[0] & ~(1u << 14);
        words[1] = 0x07040003u | (data << 8);
        const std::vector<uint8_t> image = test::make_ordinary_words_elf(GetParam(), words);
        const FaultMutationContext context(image);
        std::unique_ptr<Decoder> decoder =
            Decoder::create(default_isa_target_registry(), GetParam().target);
        ASSERT_NE(decoder, nullptr);
        const DecodeResult decoded = decoder->decode_window(std::span(words).first(2), 0);
        ASSERT_TRUE(decoded.succeeded());
        EXPECT_EQ(decoded.value()->mnemonic(), "global_load_b32");
        FaultMutationError error;
        const FailureOr<OrdinaryAcquireInventory> found =
            discover_ordinary_acquires(context, "test_kernel", error);
        ASSERT_TRUE(found.succeeded()) << error.message;
        if (!glc) {
          EXPECT_TRUE(found.value().sites().empty());
          EXPECT_TRUE(select_ordinary_acquire(found.value(), {}, error).failed());
          EXPECT_EQ(error.code, FaultMutationErrorCode::NoCandidates);
        } else {
          ASSERT_EQ(found.value().sites().size(), 1u);
          EXPECT_FALSE(found.value().sites().front().raw_scope);
          const FailureOr<OrdinaryAcquireSelection> selected =
              select_ordinary_acquire(found.value(), {}, error);
          ASSERT_TRUE(selected.succeeded()) << error.message;
          const FailureOr<OrdinaryWrongAddressResult> mutant =
              apply_ordinary_wrong_address(context, selected.value(), 8, error);
          ASSERT_TRUE(mutant.succeeded()) << error.message;
          EXPECT_TRUE(test::legacy_ordinary_output_matches(context, 0, 8, mutant.value().image,
                                                           "global_load_b32"));
          const AmdGpuCodeObject output(mutant.value().image.data(), mutant.value().image.size());
          ASSERT_TRUE(output.is_valid());
          const Section &text = *output.text_sections().front();
          const std::optional<uint64_t> guest =
              test::ordinary_emitted_guest(text, 0, GetParam().target);
          ASSERT_TRUE(guest);
          ASSERT_LE(*guest + 8, text.size());
          std::array<uint32_t, 2> actual = {};
          std::memcpy(actual.data(), text.data() + *guest, sizeof(actual));
          const std::array<uint32_t, 2> replacement = {golden[1], words[1]};
          EXPECT_EQ(actual, replacement);
        }
        EXPECT_TRUE(std::ranges::equal(context.image(), image));
      }
    }
  }
}

TEST_P(FaultRdna3OrdinaryAddress, CuLoadBeforeAgentAcquireDoesNotEnterQualifiedOrdinals) {
  std::vector<uint32_t> words = test::rdna3_ordinary_words();
  words[0] &= ~(1u << 14);
  const std::vector<uint32_t> agent = test::rdna3_ordinary_words();
  words.insert(words.end(), agent.begin(), agent.end());
  const std::vector<uint8_t> image = test::make_ordinary_words_elf(GetParam(), words);
  const FaultMutationContext context(image);
  FaultMutationError error;
  const FailureOr<OrdinaryAcquireInventory> found =
      discover_ordinary_acquires(context, "test_kernel", error);
  ASSERT_TRUE(found.succeeded()) << error.message;
  ASSERT_EQ(found.value().sites().size(), 1u);
  EXPECT_EQ(found.value().sites().front().offset, 28u);
  const FailureOr<OrdinaryAcquireSelection> selected =
      select_ordinary_acquire(found.value(), {}, error);
  ASSERT_TRUE(selected.succeeded()) << error.message;
  const FailureOr<OrdinaryWrongAddressResult> mutant =
      apply_ordinary_wrong_address(context, selected.value(), 8, error);
  ASSERT_TRUE(mutant.succeeded()) << error.message;
  EXPECT_TRUE(test::legacy_ordinary_output_matches(context, 28, 8, mutant.value().image,
                                                   "global_load_b32"));
  EXPECT_TRUE(select_ordinary_acquire(found.value(), {.exact = std::nullopt, .ordinal = 1}, error)
                  .failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::OrdinalOutOfRange);
  EXPECT_TRUE(std::ranges::equal(context.image(), image));
}

INSTANTIATE_TEST_SUITE_P(Targets, FaultRdna3OrdinaryAddress,
                         ::testing::ValuesIn(test::kRdna3OrdinaryCases),
                         [](const ::testing::TestParamInfo<test::OrdinaryTargetCase> &info) {
                           return info.param.name;
                         });

} // namespace
} // namespace rocjitsu
