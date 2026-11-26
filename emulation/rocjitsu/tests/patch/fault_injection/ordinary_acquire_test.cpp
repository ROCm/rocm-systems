// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "ordinary_fixture.h"
#include "rocjitsu/code/kernel_scope.h"
#include "rocjitsu/code/patch/fault_injection/ordinary_acquire.h"
#include "rocjitsu/code/relocation_function_table.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/target_registry.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace {

class FaultOrdinaryAcquire : public ::testing::TestWithParam<test::OrdinaryTargetCase> {};

// Compare literal source bytes and decode them independently of the acquire
// qualifier. An inventory offset/mnemonic must identify a real instruction.
bool source_instruction_matches(const FaultMutationContext &context, uint64_t offset,
                                std::span<const uint32_t> expected, std::string_view form) {
  if (context.source().text_sections().size() != 1)
    return false;
  const Section &text = *context.source().text_sections().front();
  if (offset > text.size() || expected.size_bytes() > text.size() - offset)
    return false;
  std::vector<uint32_t> actual(expected.size());
  std::memcpy(actual.data(), reinterpret_cast<const uint8_t *>(text.data()) + offset,
              expected.size_bytes());
  if (!std::ranges::equal(actual, expected))
    return false;
  std::unique_ptr<Decoder> decoder =
      Decoder::create(default_isa_target_registry(), context.source().target_id());
  if (!decoder)
    return false;
  const DecodeResult decoded = decoder->decode_window(actual, offset);
  if (decoded.failed() || decoded.value()->mnemonic() != form ||
      decoded.value()->size() != static_cast<int>(expected.size_bytes()))
    return false;
  if (form == "global_load_b32")
    return decoded.value()->src_operand(0) && decoded.value()->dst_operand(0) &&
           decoded.value()->src_operand(0)->decoded_vgpr() &&
           decoded.value()->dst_operand(0)->decoded_vgpr() &&
           decoded.value()->src_operand(0)->encoding_value() ==
               static_cast<int>(expected[2] & 0xffu) &&
           decoded.value()->dst_operand(0)->encoding_value() ==
               static_cast<int>(expected[1] & 0xffu);
  return true;
}

bool has_function_range(const AmdGpuCodeObject &source, uint64_t entry, uint64_t size) {
  return std::ranges::any_of(discover_text_function_symbol_ranges(source),
                             [&](const TextFunctionSymbolRange &range) {
                               return range.start_offset == entry && range.size == size;
                             });
}

TEST_P(FaultOrdinaryAcquire, SameBlockKeepsExactLoadWaitCacheAndDifferentRawScopes) {
  const test::OrdinaryTargetCase &fixture = GetParam();
  for (uint32_t scope : {2u, 3u}) {
    SCOPED_TRACE(scope);
    const std::vector<uint8_t> image = test::make_ordinary_elf(fixture, scope, -4, true);
    ASSERT_FALSE(image.empty());
    const FaultMutationContext context(image);
    ASSERT_EQ(context.source().target_id(), fixture.target);
    ASSERT_TRUE(has_function_range(context.source(), 0, 36));
    FaultMutationError error;
    const FailureOr<OrdinaryAcquireInventory> found =
        discover_ordinary_acquires(context, "test_kernel", error);
    ASSERT_TRUE(found.succeeded()) << error.message;
    ASSERT_EQ(found.value().sites().size(), 1u);
    const OrdinaryAcquireSite &site = found.value().sites().front();
    EXPECT_EQ(site.offset, 0u);
    EXPECT_EQ(site.instruction_size, 12u);
    EXPECT_EQ(site.form, "global_load_b32");
    EXPECT_EQ(site.profile, fixture.arch == IsaFamily::Cdna5
                                ? OrdinaryAcquireProfile::Cdna5GlobalB32
                                : OrdinaryAcquireProfile::Rdna4GlobalB32);
    EXPECT_EQ(site.byte_offset, -4);
    EXPECT_EQ(site.raw_scope, scope);
    EXPECT_EQ(site.path, OrdinaryAcquirePath::SameBlock);
    EXPECT_EQ(site.container_entry, 0u);
    EXPECT_EQ(site.container_size, 36u);
    EXPECT_EQ(site.container_name, "test_kernel");
    EXPECT_FALSE(site.polling_loop_header);
    ASSERT_EQ(site.cache_members.size(), 1u);
    EXPECT_EQ(site.cache_members.front().offset, 20u);
    EXPECT_EQ(site.cache_members.front().instruction_size, 12u);
    EXPECT_EQ(site.cache_members.front().form, "global_inv");
    ASSERT_EQ(site.wait_members.size(), 1u);
    EXPECT_EQ(site.wait_members.front().offset, 12u);
    EXPECT_EQ(site.wait_members.front().instruction_size, 4u);
    EXPECT_EQ(site.wait_members.front().form, "s_wait_loadcnt");
    ASSERT_EQ(site.affected_owners.size(), 1u);
    EXPECT_EQ(site.affected_owners.front().entry, "test_kernel");
    EXPECT_TRUE(source_instruction_matches(context, 0, test::ordinary_load_words(scope, -4),
                                           "global_load_b32"));
    EXPECT_TRUE(source_instruction_matches(context, 20, test::kOrdinaryAcquireCache, "global_inv"));
    const std::array<uint32_t, 1> wait = {test::kOrdinaryWaitLoadZero};
    EXPECT_TRUE(source_instruction_matches(context, 12, wait, "s_wait_loadcnt"));
    const FailureOr<OrdinaryAcquireSelection> selected =
        select_ordinary_acquire(found.value(), {}, error);
    ASSERT_TRUE(selected.succeeded()) << error.message;
    ASSERT_NE(selected.value().owner(), nullptr);
    EXPECT_EQ(selected.value().owner()->entry, "test_kernel");
    const FailureOr<OrdinaryAcquireSelection> exact = select_ordinary_acquire(
        found.value(), {.exact = selected.value(), .ordinal = SIZE_MAX}, error);
    ASSERT_TRUE(exact.succeeded()) << error.message;
    EXPECT_EQ(exact.value().site(), site);
    EXPECT_TRUE(std::ranges::equal(context.image(), image));
  }
}

TEST_P(FaultOrdinaryAcquire, OriginalMalformedAndUnqualifiedShapesNeverBecomeRawLoadSites) {
  const test::OrdinaryTargetCase &fixture = GetParam();
  const std::array<uint32_t, 3> load = test::ordinary_load_words();
  const std::array<uint32_t, 3> store = {0xee068004u, (2u << 18) | (7u << 23), 10u};
  const auto shape = [&](std::span<const uint32_t> access, std::span<const uint32_t> middle,
                         bool cache = true) {
    std::vector<uint32_t> words(access.begin(), access.end());
    words.insert(words.end(), middle.begin(), middle.end());
    if (cache)
      words.insert(words.end(), test::kOrdinaryAcquireCache.begin(),
                   test::kOrdinaryAcquireCache.end());
    return words;
  };
  const std::array<uint32_t, 1> wait = {test::kOrdinaryWaitLoadZero};
  const std::array<uint32_t, 4> intervening_store = {test::kOrdinaryWaitLoadZero, store[0],
                                                     store[1], store[2]};
  const std::array<uint32_t, 2> barrier = {test::kOrdinaryWaitLoadZero, 0xbf940000u};
  const std::array<uint32_t, 2> explicit_branch = {test::kOrdinaryWaitLoadZero, 0xbfa00000u};
  std::array<uint32_t, 3> malformed = load;
  malformed[0] |= 1u << 8;
  std::vector<std::vector<uint32_t>> rejected = {
      shape(load, {}),
      shape(load, wait, false),
      shape(load, intervening_store),
      shape(load, barrier),
      shape(load, explicit_branch),
      shape(test::ordinary_load_words(0), wait),
      shape(test::ordinary_load_words(1), wait),
      shape(malformed, wait),
      shape(store, wait),
  };
  // Exercise every reserved-padding region in the common modern raw form.
  for (const auto &[word_index, bit] :
       std::array<std::pair<size_t, uint32_t>, 3>{{{0, 22}, {1, 8}, {1, 31}}}) {
    std::array<uint32_t, 3> padded = load;
    padded[word_index] |= 1u << bit;
    rejected.push_back(shape(padded, wait));
  }
  for (size_t index = 0; index < rejected.size(); ++index) {
    SCOPED_TRACE(index);
    const std::vector<uint8_t> image = test::make_ordinary_words_elf(fixture, rejected[index]);
    const FaultMutationContext context(image);
    FaultMutationError error;
    const FailureOr<OrdinaryAcquireInventory> found =
        discover_ordinary_acquires(context, "test_kernel", error);
    // A malformed encoding can fail structural source admission before the
    // qualifier runs; either result must expose no qualified acquire load.
    if (found.succeeded()) {
      EXPECT_TRUE(found.value().sites().empty());
    }
    EXPECT_TRUE(std::ranges::equal(context.image(), image));
  }
}

TEST_P(FaultOrdinaryAcquire, LaterCacheDoesNotMakeTheFirstExactAssociationAmbiguous) {
  const FaultMutationContext context(test::make_ordinary_elf(GetParam(), 2, 0, false, true));
  FaultMutationError error;
  const FailureOr<OrdinaryAcquireInventory> found =
      discover_ordinary_acquires(context, "test_kernel", error);
  ASSERT_TRUE(found.succeeded()) << error.message;
  ASSERT_EQ(found.value().sites().size(), 1u);
  ASSERT_EQ(found.value().sites().front().cache_members.size(), 1u);
  EXPECT_EQ(found.value().sites().front().cache_members.front().offset, 16u);
  EXPECT_TRUE(source_instruction_matches(context, 28, test::kOrdinaryAcquireCache, "global_inv"));
}

TEST_P(FaultOrdinaryAcquire, CacheAdmissionOrderAndAmbiguousContainersKeepOriginalBoundaries) {
  const test::OrdinaryTargetCase &fixture = GetParam();
  const std::array<uint32_t, 3> load = test::ordinary_load_words();
  {
    SCOPED_TRACE("cache padding remains accepted");
    // The prototype checks padding on the load, while cache admission uses the
    // decoded mnemonic. Preserve that distinction as a host-query contract.
    const std::array<uint32_t, 3> padded_cache = {
        test::kOrdinaryAcquireCache[0] | (1u << 8) | (1u << 22),
        test::kOrdinaryAcquireCache[1] | (1u << 8) | (1u << 31), test::kOrdinaryAcquireCache[2]};
    std::vector<uint32_t> words(load.begin(), load.end());
    words.push_back(test::kOrdinaryWaitLoadZero);
    words.insert(words.end(), padded_cache.begin(), padded_cache.end());
    const std::vector<uint8_t> image = test::make_ordinary_words_elf(fixture, std::move(words));
    const FaultMutationContext context(image);
    ASSERT_TRUE(source_instruction_matches(context, 0, load, "global_load_b32"));
    ASSERT_TRUE(source_instruction_matches(context, 16, padded_cache, "global_inv"));
    FaultMutationError error;
    const FailureOr<OrdinaryAcquireInventory> found =
        discover_ordinary_acquires(context, "test_kernel", error);
    ASSERT_TRUE(found.succeeded()) << error.message;
    ASSERT_EQ(found.value().sites().size(), 1u);
    const OrdinaryAcquireSite &site = found.value().sites().front();
    EXPECT_EQ(site.offset, 0u);
    ASSERT_EQ(site.cache_members.size(), 1u);
    EXPECT_EQ(site.cache_members.front().offset, 16u);
    EXPECT_EQ(site.cache_members.front().form, "global_inv");
    EXPECT_TRUE(std::ranges::equal(context.image(), image));
  }
  {
    SCOPED_TRACE("first single-cache metadata suppresses a later global_inv");
    // s_dcache_inv is a valid acquire association in the prototype, but its
    // ordinary mutation flag is false. A later global_inv cannot rescue it.
    constexpr std::array<uint32_t, 2> scalar_cache = {0xf4042000u, 0u};
    std::vector<uint32_t> words(load.begin(), load.end());
    words.push_back(test::kOrdinaryWaitLoadZero);
    words.insert(words.end(), scalar_cache.begin(), scalar_cache.end());
    words.push_back(test::kOrdinaryWaitLoadZero);
    words.insert(words.end(), test::kOrdinaryAcquireCache.begin(),
                 test::kOrdinaryAcquireCache.end());
    const std::vector<uint8_t> image = test::make_ordinary_words_elf(fixture, std::move(words));
    const FaultMutationContext context(image);
    ASSERT_TRUE(source_instruction_matches(context, 0, load, "global_load_b32"));
    ASSERT_TRUE(source_instruction_matches(context, 16, scalar_cache, "s_dcache_inv"));
    ASSERT_TRUE(source_instruction_matches(context, 28, test::kOrdinaryAcquireCache, "global_inv"));
    FaultMutationError error;
    const FailureOr<OrdinaryAcquireInventory> found =
        discover_ordinary_acquires(context, "test_kernel", error);
    ASSERT_TRUE(found.succeeded()) << error.message;
    EXPECT_TRUE(found.value().sites().empty());
    EXPECT_TRUE(select_ordinary_acquire(found.value(), {}, error).failed());
    EXPECT_EQ(error.code, FaultMutationErrorCode::NoCandidates);
    EXPECT_TRUE(std::ranges::equal(context.image(), image));
  }
  {
    SCOPED_TRACE("cache before load is not an acquire association");
    std::vector<uint32_t> words(test::kOrdinaryAcquireCache.begin(),
                                test::kOrdinaryAcquireCache.end());
    words.insert(words.end(), load.begin(), load.end());
    words.push_back(test::kOrdinaryWaitLoadZero);
    const std::vector<uint8_t> image = test::make_ordinary_words_elf(fixture, std::move(words));
    const FaultMutationContext context(image);
    ASSERT_TRUE(source_instruction_matches(context, 0, test::kOrdinaryAcquireCache, "global_inv"));
    ASSERT_TRUE(source_instruction_matches(context, 12, load, "global_load_b32"));
    FaultMutationError error;
    const FailureOr<OrdinaryAcquireInventory> found =
        discover_ordinary_acquires(context, "test_kernel", error);
    ASSERT_TRUE(found.succeeded()) << error.message;
    EXPECT_TRUE(found.value().sites().empty());
    EXPECT_TRUE(std::ranges::equal(context.image(), image));
  }
  for (uint64_t alias_size : {4u, 8u, 28u}) {
    SCOPED_TRACE("overlapping explicit function extents are ambiguous");
    SCOPED_TRACE(alias_size);
    std::vector<uint8_t> image = test::make_ordinary_elf(fixture);
    ASSERT_TRUE(test::add_ordinary_function_symbol(image, 0, alias_size, "overlapping_alias"));
    const FaultMutationContext context(image);
    ASSERT_TRUE(has_function_range(context.source(), 0, 32));
    ASSERT_TRUE(has_function_range(context.source(), 0, alias_size));
    ASSERT_TRUE(source_instruction_matches(context, 0, load, "global_load_b32"));
    ASSERT_TRUE(source_instruction_matches(context, 16, test::kOrdinaryAcquireCache, "global_inv"));
    FaultMutationError error;
    const FailureOr<OrdinaryAcquireInventory> found =
        discover_ordinary_acquires(context, "test_kernel", error);
    ASSERT_TRUE(found.succeeded()) << error.message;
    EXPECT_TRUE(found.value().sites().empty());
    EXPECT_TRUE(std::ranges::equal(context.image(), image));
  }
  {
    SCOPED_TRACE("identical-bound aliases retain the actual container");
    std::vector<uint8_t> image = test::make_ordinary_elf(fixture);
    ASSERT_TRUE(test::add_ordinary_function_symbol(image, 0, 32, "same_extent_alias"));
    const FaultMutationContext context(image);
    FaultMutationError error;
    const FailureOr<OrdinaryAcquireInventory> found =
        discover_ordinary_acquires(context, "test_kernel", error);
    ASSERT_TRUE(found.succeeded()) << error.message;
    ASSERT_EQ(found.value().sites().size(), 1u);
    EXPECT_EQ(found.value().sites().front().container_entry, 0u);
    EXPECT_EQ(found.value().sites().front().container_size, 32u);
    EXPECT_EQ(found.value().sites().front().container_name, "test_kernel");
    EXPECT_TRUE(std::ranges::equal(context.image(), image));
  }
}

TEST_P(FaultOrdinaryAcquire, ExactFallthroughCanJoinAnAdditionalCachePredecessor) {
  const FaultMutationContext context(test::make_ordinary_fallthrough_elf(GetParam()));
  FaultMutationError error;
  const FailureOr<OrdinaryAcquireInventory> found =
      discover_ordinary_acquires(context, "test_kernel", error);
  ASSERT_TRUE(found.succeeded()) << error.message;
  ASSERT_EQ(found.value().sites().size(), 1u);
  const OrdinaryAcquireSite &site = found.value().sites().front();
  EXPECT_EQ(site.offset, 4u);
  EXPECT_EQ(site.path, OrdinaryAcquirePath::ExactFallthroughJoin);
  ASSERT_EQ(site.cache_members.size(), 1u);
  EXPECT_EQ(site.cache_members.front().offset, 24u);
  EXPECT_FALSE(site.polling_loop_header);
}

TEST_P(FaultOrdinaryAcquire, OriginalPollingLoopAndBoundedAddressBookkeepingAreRetained) {
  for (uint32_t prefix_words : {0u, 32u}) {
    SCOPED_TRACE(prefix_words);
    const FaultMutationContext context(test::make_ordinary_poll_elf(GetParam(), prefix_words));
    FaultMutationError error;
    const FailureOr<OrdinaryAcquireInventory> found =
        discover_ordinary_acquires(context, "test_kernel", error);
    ASSERT_TRUE(found.succeeded()) << error.message;
    ASSERT_EQ(found.value().sites().size(), 1u);
    const OrdinaryAcquireSite &site = found.value().sites().front();
    EXPECT_EQ(site.offset, prefix_words * 4);
    EXPECT_EQ(site.path, OrdinaryAcquirePath::ExactSelfLoopExit);
    ASSERT_TRUE(site.polling_loop_header);
    EXPECT_EQ(*site.polling_loop_header, 0u);
    ASSERT_EQ(site.cache_members.size(), 1u);
    EXPECT_EQ(site.cache_members.front().offset, prefix_words * 4 + 40);
    ASSERT_EQ(site.wait_members.size(), 1u);
    EXPECT_EQ(site.wait_members.front().offset, prefix_words * 4 + 12);
  }
  for (const std::vector<uint8_t> &image : {test::make_ordinary_poll_elf(GetParam(), 33),
                                            test::make_ordinary_poll_elf(GetParam(), 0, true)}) {
    const FaultMutationContext context(image);
    FaultMutationError error;
    const FailureOr<OrdinaryAcquireInventory> found =
        discover_ordinary_acquires(context, "test_kernel", error);
    ASSERT_TRUE(found.succeeded()) << error.message;
    EXPECT_TRUE(found.value().sites().empty());
  }
  // A wait in the polling-loop prefix does not witness the required wait
  // after the load. Replace the post-load wait with a nop independently.
  std::vector<uint8_t> prefix_wait = test::make_ordinary_poll_elf(GetParam(), 1);
  const AmdGpuCodeObject layout(prefix_wait.data(), prefix_wait.size());
  ASSERT_TRUE(layout.is_valid());
  const size_t text_file = layout.text_sections().front()->sectionOffset();
  const uint32_t wait = test::kOrdinaryWaitLoadZero;
  const uint32_t nop = build_s_nop(0, GetParam().arch).value();
  std::memcpy(prefix_wait.data() + text_file, &wait, sizeof(wait));
  std::memcpy(prefix_wait.data() + text_file + 16, &nop, sizeof(nop));
  const FaultMutationContext prefix_context(prefix_wait);
  FaultMutationError prefix_error;
  const FailureOr<OrdinaryAcquireInventory> prefix_found =
      discover_ordinary_acquires(prefix_context, "test_kernel", prefix_error);
  ASSERT_TRUE(prefix_found.succeeded()) << prefix_error.message;
  EXPECT_TRUE(prefix_found.value().sites().empty());
}

TEST_P(FaultOrdinaryAcquire, WaitAndGapBoundsMatchTheOriginalQualifier) {
  const test::OrdinaryTargetCase &fixture = GetParam();
  for (size_t nops : {15u, 16u}) {
    const std::array<uint32_t, 3> load = test::ordinary_load_words();
    std::vector<uint32_t> words(load.begin(), load.end());
    words.push_back(test::kOrdinaryWaitLoadZero);
    words.insert(words.end(), nops, build_s_nop(0, fixture.arch).value());
    words.insert(words.end(), test::kOrdinaryAcquireCache.begin(),
                 test::kOrdinaryAcquireCache.end());
    const FaultMutationContext context(test::make_ordinary_words_elf(fixture, std::move(words)));
    FaultMutationError error;
    const FailureOr<OrdinaryAcquireInventory> found =
        discover_ordinary_acquires(context, "test_kernel", error);
    ASSERT_TRUE(found.succeeded()) << error.message;
    EXPECT_EQ(found.value().sites().size(), nops == 15 ? 1u : 0u);
  }
  // The original proof accepts any decoded s_wait* as its wait witness. Keep
  // that query contract separate from a claim about runtime counter draining.
  std::vector<uint32_t> words = test::ordinary_acquire_words();
  words[3] = 0xbf88ff9eu; // s_wait_alu
  const FaultMutationContext context(test::make_ordinary_words_elf(fixture, std::move(words)));
  FaultMutationError error;
  const FailureOr<OrdinaryAcquireInventory> found =
      discover_ordinary_acquires(context, "test_kernel", error);
  ASSERT_TRUE(found.succeeded()) << error.message;
  ASSERT_EQ(found.value().sites().size(), 1u);
  ASSERT_EQ(found.value().sites().front().wait_members.size(), 1u);
  EXPECT_TRUE(found.value().sites().front().wait_members.front().form.starts_with("s_wait"));
}

TEST_P(FaultOrdinaryAcquire, OrdinalsCountOnlyQualifiedLoadsAndExactNeverFallsBack) {
  const test::OrdinaryTargetCase &fixture = GetParam();
  std::vector<uint32_t> words;
  for (uint32_t scope : {0u, 2u, 3u}) {
    const std::vector<uint32_t> sequence = test::ordinary_acquire_words(scope);
    words.insert(words.end(), sequence.begin(), sequence.end());
  }
  const FaultMutationContext context(test::make_ordinary_words_elf(fixture, std::move(words)));
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
    const FailureOr<OrdinaryAcquireSelection> exact = select_ordinary_acquire(
        found.value(), {.exact = selected.value(), .ordinal = SIZE_MAX}, error);
    ASSERT_TRUE(exact.succeeded()) << error.message;
    EXPECT_EQ(exact.value().site(), selected.value().site());
  }
  EXPECT_TRUE(select_ordinary_acquire(found.value(), {.exact = std::nullopt, .ordinal = 2}, error)
                  .failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::OrdinalOutOfRange);
  EXPECT_TRUE(select_ordinary_acquire(found.value(),
                                      {.exact = OrdinaryAcquireSelection{}, .ordinal = 0}, error)
                  .failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  EXPECT_TRUE(discover_ordinary_acquires(context, "test_", error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
}

TEST_P(FaultOrdinaryAcquire, GenuineSharedHelperContainerKeepsBothOwnersAndExactIdentity) {
  const std::vector<uint8_t> image = test::make_shared_ordinary_elf(GetParam());
  ASSERT_FALSE(image.empty());
  const FaultMutationContext context(image);
  ASSERT_TRUE(has_function_range(context.source(), 512, 32));
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
  EXPECT_EQ(site.offset, 512u);
  EXPECT_EQ(site.container_entry, 512u);
  EXPECT_EQ(site.container_size, 32u);
  EXPECT_EQ(site.container_name, "function@512");
  ASSERT_EQ(site.cache_members.size(), 1u);
  EXPECT_EQ(site.cache_members.front().offset, 528u);
  ASSERT_EQ(site.affected_owners.size(), 2u);
  EXPECT_EQ(site.affected_owners[0].entry, "test_kernel0");
  EXPECT_EQ(site.affected_owners[1].entry, "test_kernel1");
  EXPECT_EQ(site, second.value().sites().front());
  EXPECT_TRUE(
      source_instruction_matches(context, 512, test::ordinary_load_words(), "global_load_b32"));
  EXPECT_TRUE(source_instruction_matches(context, 528, test::kOrdinaryAcquireCache, "global_inv"));
  const FailureOr<OrdinaryAcquireSelection> selected =
      select_ordinary_acquire(first.value(), {}, error);
  ASSERT_TRUE(selected.succeeded()) << error.message;
  const FailureOr<OrdinaryAcquireSelection> other = select_ordinary_acquire(
      second.value(), {.exact = selected.value(), .ordinal = SIZE_MAX}, error);
  ASSERT_TRUE(other.succeeded()) << error.message;
  ASSERT_NE(other.value().owner(), nullptr);
  EXPECT_EQ(other.value().owner()->entry, "test_kernel1");
  EXPECT_TRUE(std::ranges::equal(context.image(), image));
}

TEST_P(FaultOrdinaryAcquire, ZeroSizedKernelAndSharedHelperKeepOriginalAcquireQualification) {
  const test::OrdinaryTargetCase &fixture = GetParam();
  for (bool shared : {false, true}) {
    SCOPED_TRACE(shared);
    std::vector<uint8_t> image =
        shared ? test::make_shared_ordinary_elf(fixture) : test::make_ordinary_elf(fixture);
    const uint64_t entry = shared ? 512 : 0;
    ASSERT_TRUE(test::zero_ordinary_function_size(image, entry));
    const FaultMutationContext context(image);
    EXPECT_FALSE(has_function_range(context.source(), entry, 32));
    const std::vector<TextFunctionSymbolExtent> extents =
        discover_text_function_symbol_extents(context.source());
    const auto extent =
        std::ranges::find_if(extents, [&](const TextFunctionSymbolExtent &candidate) {
          return candidate.range.start_offset == entry && candidate.range.size == 32;
        });
    ASSERT_NE(extent, extents.end());
    EXPECT_TRUE(extent->inferred_from_zero);
    FaultMutationError error;
    const FailureOr<OrdinaryAcquireInventory> found =
        discover_ordinary_acquires(context, shared ? "test_kernel0" : "test_kernel", error);
    ASSERT_TRUE(found.succeeded()) << error.message;
    ASSERT_EQ(found.value().sites().size(), 1u);
    const OrdinaryAcquireSite &site = found.value().sites().front();
    EXPECT_EQ(site.offset, entry);
    EXPECT_EQ(site.container_entry, entry);
    EXPECT_EQ(site.container_size, 32u);
    EXPECT_EQ(site.container_name, shared ? "function@512" : "test_kernel");
    ASSERT_EQ(site.cache_members.size(), 1u);
    EXPECT_EQ(site.cache_members.front().offset, entry + 16);
    EXPECT_EQ(site.affected_owners.size(), shared ? 2u : 1u);
    if (shared) {
      const FailureOr<OrdinaryAcquireInventory> other =
          discover_ordinary_acquires(context, "test_kernel1", error);
      ASSERT_TRUE(other.succeeded()) << error.message;
      ASSERT_EQ(other.value().sites().size(), 1u);
      EXPECT_EQ(other.value().sites().front(), site);
      const FailureOr<OrdinaryAcquireSelection> selected =
          select_ordinary_acquire(found.value(), {}, error);
      ASSERT_TRUE(selected.succeeded()) << error.message;
      EXPECT_TRUE(select_ordinary_acquire(other.value(),
                                          {.exact = selected.value(), .ordinal = SIZE_MAX}, error)
                      .succeeded())
          << error.message;
    }
    EXPECT_TRUE(std::ranges::equal(context.image(), image));
  }
}

TEST_P(FaultOrdinaryAcquire, UnrelatedOwnerAndEndpointOwnerMismatchCannotQualifySharedLoad) {
  const FaultMutationContext context(test::make_shared_ordinary_elf(GetParam(), false));
  FaultMutationError error;
  const FailureOr<OrdinaryAcquireInventory> owner =
      discover_ordinary_acquires(context, "test_kernel0", error);
  ASSERT_TRUE(owner.succeeded()) << error.message;
  ASSERT_EQ(owner.value().sites().size(), 1u);
  const FailureOr<OrdinaryAcquireInventory> unrelated =
      discover_ordinary_acquires(context, "test_kernel1", error);
  ASSERT_TRUE(unrelated.succeeded()) << error.message;
  EXPECT_TRUE(unrelated.value().sites().empty());
  EXPECT_TRUE(select_ordinary_acquire(unrelated.value(), {}, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::NoCandidates);
  const FailureOr<OrdinaryAcquireSelection> selected =
      select_ordinary_acquire(owner.value(), {}, error);
  ASSERT_TRUE(selected.succeeded()) << error.message;
  EXPECT_TRUE(
      select_ordinary_acquire(unrelated.value(), {.exact = selected.value(), .ordinal = 0}, error)
          .failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);

  const FaultMutationContext mismatch(test::make_shared_ordinary_elf(GetParam(), true, true));
  const FailureOr<OrdinaryAcquireInventory> mismatched =
      discover_ordinary_acquires(mismatch, "test_kernel0", error);
  ASSERT_TRUE(mismatched.succeeded()) << error.message;
  EXPECT_TRUE(mismatched.value().sites().empty());
}

TEST(FaultOrdinaryAcquireIdentity, SnapshotForeignContextAndExpiredMetadataAreIndependent) {
  std::vector<uint8_t> input = test::make_ordinary_elf(test::kOrdinaryTargetCases.front());
  const std::vector<uint8_t> pristine = input;
  const FaultMutationContext context(input), foreign(input);
  input[0] = 0;
  FaultMutationError error;
  const FailureOr<OrdinaryAcquireInventory> found =
      discover_ordinary_acquires(context, "test_kernel", error);
  const FailureOr<OrdinaryAcquireInventory> foreign_found =
      discover_ordinary_acquires(foreign, "test_kernel", error);
  ASSERT_TRUE(found.succeeded()) << error.message;
  ASSERT_TRUE(foreign_found.succeeded()) << error.message;
  const FailureOr<OrdinaryAcquireSelection> selected =
      select_ordinary_acquire(found.value(), {}, error);
  ASSERT_TRUE(selected.succeeded()) << error.message;
  EXPECT_TRUE(select_ordinary_acquire(foreign_found.value(),
                                      {.exact = selected.value(), .ordinal = 0}, error)
                  .failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  EXPECT_TRUE(std::ranges::equal(context.image(), pristine));

  std::optional<OrdinaryAcquireInventory> expired_inventory;
  std::optional<OrdinaryAcquireSelection> expired_selection;
  {
    const FaultMutationContext temporary(
        test::make_shared_ordinary_elf(test::kOrdinaryTargetCases.front()));
    const FailureOr<OrdinaryAcquireInventory> inventory =
        discover_ordinary_acquires(temporary, "test_kernel0", error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    const FailureOr<OrdinaryAcquireSelection> exact =
        select_ordinary_acquire(inventory.value(), {}, error);
    ASSERT_TRUE(exact.succeeded()) << error.message;
    expired_inventory = inventory.value();
    expired_selection = exact.value();
  }
  EXPECT_EQ(expired_inventory->sites().front().form, "global_load_b32");
  EXPECT_EQ(expired_selection->site().container_name, "function@512");
  EXPECT_EQ(expired_selection->site().cache_members.front().form, "global_inv");
  EXPECT_TRUE(select_ordinary_acquire(*expired_inventory, {}, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
}

TEST(FaultOrdinaryAcquireIdentity,
     TargetsWithoutAnAcquireProfileAndDefaultInventoryFailExplicitly) {
  std::vector<uint8_t> image = test::make_amdgpu_kernel_elf(
      {build_s_endpgm(IsaFamily::Cdna4).value()}, 0, 3, EF_AMDGPU_MACH_AMDGCN_GFX950);
  image[EI_DATA] = ELFDATA2LSB;
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  const FaultMutationContext context(image);
  FaultMutationError error;
  EXPECT_TRUE(discover_ordinary_acquires(context, "test_kernel", error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedTarget);
  EXPECT_TRUE(select_ordinary_acquire({}, {}, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
}

TEST(FaultOrdinaryAcquireIdentity, CrossingARealFunctionBoundaryCannotUseKernelOwnerAsContainer) {
  const test::OrdinaryTargetCase &fixture = test::kOrdinaryTargetCases.front();
  // Mirror the positive fallthrough fixture: the branch targets the cache,
  // making the second FUNC's entry an actual reachable CFG block leader.
  std::vector<uint32_t> words{test::ordinary_vccz_branch(5, fixture.arch)};
  const std::array<uint32_t, 3> load = test::ordinary_load_words();
  words.insert(words.end(), load.begin(), load.end());
  words.push_back(test::kOrdinaryWaitLoadZero);
  words.push_back(test::kOrdinaryDelay);
  words.insert(words.end(), test::kOrdinaryAcquireCache.begin(), test::kOrdinaryAcquireCache.end());
  words.push_back(build_s_endpgm(fixture.arch).value());
  std::vector<uint8_t> image = test::make_amdgpu_kernel_elf(
      words, 0, 3, fixture.machine, 3, 0, false, false, false, false, fixture.wave == 32);
  image[EI_DATA] = ELFDATA2LSB;
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  ASSERT_TRUE(test::add_ordinary_function_symbol(image, 0, 24, "test_kernel"));
  ASSERT_TRUE(test::add_ordinary_function_symbol(image, 24, 16, "cache_helper"));

  // Establish complete source ownership independently of acquire qualification.
  // Both exact members belong to one kernel despite their distinct containers.
  const std::shared_ptr<const AmdGpuCodeObject> ownership_source =
      std::make_shared<AmdGpuCodeObject>(image.data(), image.size());
  std::string ownership_error;
  const FailureOr<KernelOwnership> ownership =
      KernelOwnership::build(ownership_source, fixture.target, ownership_error);
  ASSERT_TRUE(ownership.succeeded()) << ownership_error;
  const std::optional<std::span<const KernelExecutionOwner>> load_owners =
      ownership.value().owners_at(4);
  const std::optional<std::span<const KernelExecutionOwner>> cache_owners =
      ownership.value().owners_at(24);
  ASSERT_TRUE(load_owners);
  ASSERT_TRUE(cache_owners);
  ASSERT_EQ(load_owners->size(), 1u);
  ASSERT_EQ(cache_owners->size(), 1u);
  EXPECT_EQ(load_owners->front(), cache_owners->front());

  const FaultMutationContext context(image);
  ASSERT_TRUE(has_function_range(context.source(), 0, 24));
  ASSERT_TRUE(has_function_range(context.source(), 24, 16));
  EXPECT_TRUE(source_instruction_matches(context, 4, load, "global_load_b32"));
  EXPECT_TRUE(source_instruction_matches(context, 24, test::kOrdinaryAcquireCache, "global_inv"));
  FaultMutationError error;
  const FailureOr<OrdinaryAcquireInventory> found =
      discover_ordinary_acquires(context, "test_kernel", error);
  ASSERT_TRUE(found.succeeded()) << error.message;
  EXPECT_TRUE(found.value().sites().empty());
}

INSTANTIATE_TEST_SUITE_P(Targets, FaultOrdinaryAcquire,
                         ::testing::ValuesIn(test::kOrdinaryTargetCases),
                         [](const ::testing::TestParamInfo<test::OrdinaryTargetCase> &info) {
                           return info.param.name;
                         });

} // namespace
} // namespace rocjitsu
