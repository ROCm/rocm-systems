// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "atomic_fixture.h"
#include "rocjitsu/code/patch/fault_injection/atomic_wrong_address.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <vector>

namespace rocjitsu {
namespace {

std::array<uint32_t, 3> buffer_words(const test::AtomicTargetCase &fixture, int32_t offset,
                                     uint8_t th = 1) {
  const uint32_t encoded = static_cast<uint32_t>(offset) & 0xffffffu;
  if (fixture.arch == IsaFamily::Cdna5)
    return cdna5::build_vbuffer(cdna5::kBufferAtomicAddU32Vbuffer, {.soffset = 7,
                                                                    .vdata = 6,
                                                                    .rsrc = 8,
                                                                    .scope = 2,
                                                                    .th = th,
                                                                    .offen = 1,
                                                                    .idxen = 1,
                                                                    .vaddr = 2,
                                                                    .ioffset = encoded});
  return rdna4::build_vbuffer(rdna4::kBufferAtomicAddU32Vbuffer, {.soffset = 7,
                                                                  .vdata = 6,
                                                                  .rsrc = 8,
                                                                  .scope = 2,
                                                                  .th = th,
                                                                  .offen = 1,
                                                                  .idxen = 1,
                                                                  .vaddr = 2,
                                                                  .ioffset = encoded});
}

std::vector<uint8_t> buffer_image(const test::AtomicTargetCase &fixture, int32_t offset,
                                  uint8_t th = 1) {
  const std::array<uint32_t, 3> words = buffer_words(fixture, offset, th);
  return test::make_atomic_sequence_elf(fixture, {words[0], words[1], words[2]});
}

class FaultAtomicBuffer : public ::testing::TestWithParam<test::AtomicTargetCase> {};

TEST_P(FaultAtomicBuffer, PreservesReturningDataResourceAddressModifiersAndWait) {
  const test::AtomicTargetCase &fixture = GetParam();
  const std::vector<AtomicCapability> capabilities = atomic_wrong_address_capabilities();
  EXPECT_TRUE(std::ranges::any_of(capabilities, [&](const AtomicCapability &capability) {
    return capability.target == fixture.target && capability.wave_size == fixture.wave &&
           capability.form == "buffer_atomic";
  }));
  for (int32_t offset : {0, 4}) {
    SCOPED_TRACE(offset);
    for (uint8_t th : {uint8_t{0}, uint8_t{1}}) {
      SCOPED_TRACE(static_cast<int>(th));
      const std::vector<uint8_t> image = buffer_image(fixture, offset, th);
      const FaultMutationContext source(image);
      FaultMutationError error;
      const FailureOr<AtomicInventory> inventory =
          discover_atomic_wrong_address(source, "test_kernel", error);
      ASSERT_TRUE(inventory.succeeded()) << error.message;
      ASSERT_EQ(inventory.value().sites().size(), 1u);
      const AtomicSite &site = inventory.value().sites().front();
      EXPECT_EQ(site.offset, 0u);
      EXPECT_EQ(site.instruction_size, 12u);
      EXPECT_EQ(site.form, "buffer_atomic_add_u32");
      ASSERT_TRUE(site.byte_offset);
      EXPECT_EQ(*site.byte_offset, offset);
      EXPECT_TRUE(site.address_supported);
      const FailureOr<AtomicSelection> selected =
          select_atomic_wrong_address(inventory.value(), {}, error);
      ASSERT_TRUE(selected.succeeded()) << error.message;
      const FailureOr<AtomicResult> mutant =
          apply_atomic_wrong_address(source, selected.value(), 8, error);
      ASSERT_TRUE(mutant.succeeded()) << error.message;
      EXPECT_EQ(mutant.value().logical_mutation_count, 1u);
      EXPECT_TRUE(test::atomic_output_matches(source, 0, 8, mutant.value().image));
      const AmdGpuCodeObject candidate(mutant.value().image.data(), mutant.value().image.size());
      ASSERT_TRUE(candidate.is_valid());
      const Section &text = *candidate.text_sections().front();
      const std::optional<uint64_t> guest = test::atomic_emitted_guest(text, 0, fixture.target);
      ASSERT_TRUE(guest);
      ASSERT_LE(*guest + 12u, text.size());
      std::array<uint32_t, 3> changed = {};
      std::memcpy(changed.data(), text.data() + *guest, sizeof(changed));
      const std::array<uint32_t, 3> original = buffer_words(fixture, offset, th);
      EXPECT_EQ(changed[0], original[0]);
      EXPECT_EQ(changed[1], original[1]);
      EXPECT_EQ(changed[2] & 0xffu, 2u);
      EXPECT_EQ(changed[2] >> 8, static_cast<uint32_t>(offset + 8));
      EXPECT_EQ(changed[0] & 0x7fu, 7u);
      EXPECT_EQ(changed[1] & 0xffu, 6u);
      EXPECT_EQ((changed[1] >> 9) & 0x1ffu, 8u);
      EXPECT_EQ((changed[1] >> 18) & 3u, 2u);
      EXPECT_EQ((changed[1] >> 20) & 7u, th);
      EXPECT_EQ((changed[1] >> 30) & 3u, 3u);
      const FailureOr<AtomicResult> independent =
          apply_atomic_wrong_address(source, selected.value(), 4, error);
      ASSERT_TRUE(independent.succeeded()) << error.message;
      EXPECT_TRUE(test::atomic_output_matches(source, 0, 4, independent.value().image));
      EXPECT_NE(independent.value().image, mutant.value().image);
      EXPECT_TRUE(std::ranges::equal(source.image(), image));
    }
  }
}

TEST_P(FaultAtomicBuffer, NegativeSourceRemainsSelectableButCannotMutate) {
  const test::AtomicTargetCase &fixture = GetParam();
  for (int32_t offset : {-4, -8}) {
    SCOPED_TRACE(offset);
    const std::vector<uint8_t> image = buffer_image(fixture, offset);
    const FaultMutationContext source(image);
    FaultMutationError error;
    const FailureOr<AtomicInventory> inventory =
        discover_atomic_wrong_address(source, "test_kernel", error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    ASSERT_EQ(inventory.value().sites().size(), 1u);
    EXPECT_EQ(inventory.value().sites().front().byte_offset, offset);
    EXPECT_FALSE(inventory.value().sites().front().address_supported);
    const FailureOr<AtomicSelection> selected =
        select_atomic_wrong_address(inventory.value(), {}, error);
    ASSERT_TRUE(selected.succeeded()) << error.message;
    // A positive replacement cannot make an illegal VBUFFER source eligible.
    for (uint32_t delta : {4u, 8u}) {
      SCOPED_TRACE(delta);
      EXPECT_TRUE(apply_atomic_wrong_address(source, selected.value(), delta, error).failed());
      EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm);
      EXPECT_TRUE(std::ranges::equal(source.image(), image));
    }
  }
}

TEST_P(FaultAtomicBuffer, DeltaAndSignedOffsetBoundsRejectWithoutMutation) {
  const test::AtomicTargetCase &fixture = GetParam();
  const std::vector<uint8_t> image = buffer_image(fixture, 0);
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
        std::optional<uint32_t>{0x800000}, std::optional<uint32_t>{UINT32_MAX}}) {
    EXPECT_TRUE(apply_atomic_wrong_address(source, selected.value(), delta, error).failed());
    EXPECT_EQ(error.code, FaultMutationErrorCode::InvalidParameters);
    EXPECT_TRUE(std::ranges::equal(source.image(), image));
  }
  const FailureOr<AtomicResult> largest_delta =
      apply_atomic_wrong_address(source, selected.value(), 0x7ffffc, error);
  ASSERT_TRUE(largest_delta.succeeded()) << error.message;
  EXPECT_TRUE(test::atomic_output_matches(source, 0, 0x7ffffc, largest_delta.value().image));
  for (int32_t offset : {0x7ffffc, 0x7fffff}) {
    SCOPED_TRACE(offset);
    const std::vector<uint8_t> original = buffer_image(fixture, offset);
    const FaultMutationContext bounded(original);
    const FailureOr<AtomicInventory> found =
        discover_atomic_wrong_address(bounded, "test_kernel", error);
    ASSERT_TRUE(found.succeeded()) << error.message;
    const FailureOr<AtomicSelection> selection =
        select_atomic_wrong_address(found.value(), {}, error);
    ASSERT_TRUE(selection.succeeded()) << error.message;
    EXPECT_TRUE(apply_atomic_wrong_address(bounded, selection.value(), 4, error).failed());
    EXPECT_EQ(error.code, FaultMutationErrorCode::InvalidParameters);
    EXPECT_TRUE(std::ranges::equal(bounded.image(), original));
  }
}

TEST_P(FaultAtomicBuffer, IndependentOracleRejectsOffsetAndUnselectedFieldCorruption) {
  const test::AtomicTargetCase &fixture = GetParam();
  const FaultMutationContext source(buffer_image(fixture, 4));
  FaultMutationError error;
  const FailureOr<AtomicInventory> inventory =
      discover_atomic_wrong_address(source, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const FailureOr<AtomicSelection> selected =
      select_atomic_wrong_address(inventory.value(), {}, error);
  ASSERT_TRUE(selected.succeeded()) << error.message;
  const FailureOr<AtomicResult> mutant =
      apply_atomic_wrong_address(source, selected.value(), 8, error);
  ASSERT_TRUE(mutant.succeeded()) << error.message;
  ASSERT_TRUE(test::atomic_output_matches(source, 0, 8, mutant.value().image));
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
  constexpr std::array<Corruption, 10> corruptions = {{
      {.word = 0, .mask = 1u},
      {.word = 0, .mask = 1u << 14},
      {.word = 1, .mask = 1u},
      {.word = 1, .mask = 1u << 9},
      {.word = 1, .mask = 1u << 18},
      {.word = 1, .mask = 1u << 20},
      {.word = 1, .mask = 1u << 30},
      {.word = 1, .mask = 1u << 31},
      {.word = 2, .mask = 1u},
      {.word = 2, .mask = 1u << 8},
  }};
  for (const Corruption &corruption : corruptions) {
    SCOPED_TRACE(corruption.word);
    SCOPED_TRACE(corruption.mask);
    std::vector<uint8_t> corrupted = mutant.value().image;
    const uint64_t location = text.sectionOffset() + *guest + corruption.word * sizeof(uint32_t);
    ASSERT_LE(location + sizeof(uint32_t), corrupted.size());
    uint32_t word = 0;
    std::memcpy(&word, corrupted.data() + location, sizeof(word));
    word ^= corruption.mask;
    std::memcpy(corrupted.data() + location, &word, sizeof(word));
    EXPECT_FALSE(test::atomic_output_matches(source, 0, 8, corrupted));
  }
  std::vector<uint8_t> changed_wait = mutant.value().image;
  changed_wait[text.sectionOffset() + 12u] ^= 1u;
  EXPECT_FALSE(test::atomic_output_matches(source, 0, 8, changed_wait));
}

INSTANTIATE_TEST_SUITE_P(Targets, FaultAtomicBuffer, ::testing::ValuesIn(test::kAtomicTargetCases),
                         [](const ::testing::TestParamInfo<test::AtomicTargetCase> &info) {
                           return info.param.name;
                         });

} // namespace
} // namespace rocjitsu
