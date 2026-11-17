// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "atomic_legacy_fixture.h"
#include "rocjitsu/code/patch/fault_injection/atomic_legacy_fields.h"
#include "rocjitsu/code/patch/fault_injection/atomic_wrong_address.h"

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
#include <string_view>
#include <vector>

namespace rocjitsu {
namespace {

using test::LegacyAtomicForm;
using test::LegacyAtomicTargetCase;

FailureOr<AtomicSelection> select_legacy(const FaultMutationContext &source,
                                         FaultMutationError &error, size_t ordinal = 0) {
  const FailureOr<AtomicInventory> inventory =
      discover_atomic_wrong_address(source, "test_kernel", error);
  if (inventory.failed())
    return Result::failure();
  return select_atomic_wrong_address(inventory.value(), {.exact = std::nullopt, .ordinal = ordinal},
                                     error);
}

class FaultLegacyAtomic : public ::testing::TestWithParam<LegacyAtomicTargetCase> {};

TEST_P(FaultLegacyAtomic, FlatGlobalBufferAddCasAndReturnModesPreserveCompleteGuests) {
  const LegacyAtomicTargetCase &target = GetParam();
  const std::vector<AtomicCapability> capabilities = atomic_wrong_address_capabilities();
  for (std::string_view category : {"flat_atomic", "global_atomic", "buffer_atomic"}) {
    EXPECT_TRUE(std::ranges::any_of(capabilities, [&](const AtomicCapability &capability) {
      return capability.target == target.target && capability.wave_size == target.wave &&
             capability.form == category;
    }));
  }
  for (LegacyAtomicForm form : test::kLegacyAtomicForms) {
    SCOPED_TRACE(test::legacy_atomic_mnemonic(target, form));
    for (bool returning : {false, true}) {
      SCOPED_TRACE(returning);
      const int32_t offset = test::legacy_atomic_signed(target, form) ? -4 : 4;
      const std::vector<uint8_t> image =
          test::make_legacy_atomic_elf(target, form, offset, returning);
      const FaultMutationContext source(image);
      FaultMutationError error;
      const FailureOr<AtomicInventory> inventory =
          discover_atomic_wrong_address(source, "test_kernel", error);
      ASSERT_TRUE(inventory.succeeded()) << error.message;
      ASSERT_EQ(inventory.value().sites().size(), 1u);
      const AtomicSite &site = inventory.value().sites().front();
      EXPECT_EQ(site.offset, 0u);
      EXPECT_EQ(site.instruction_size, 8u);
      EXPECT_EQ(site.form, test::legacy_atomic_mnemonic(target, form));
      EXPECT_EQ(site.byte_offset, offset);
      EXPECT_TRUE(site.address_supported);
      ASSERT_EQ(site.affected_owners.size(), 1u);
      EXPECT_EQ(site.affected_owners.front().entry, "test_kernel");
      const FailureOr<AtomicSelection> ordinal =
          select_atomic_wrong_address(inventory.value(), {}, error);
      ASSERT_TRUE(ordinal.succeeded()) << error.message;
      const FailureOr<AtomicSelection> exact = select_atomic_wrong_address(
          inventory.value(), {.exact = ordinal.value(), .ordinal = SIZE_MAX}, error);
      ASSERT_TRUE(exact.succeeded()) << error.message;
      const FailureOr<AtomicResult> mutant =
          apply_atomic_wrong_address(source, exact.value(), 8, error);
      ASSERT_TRUE(mutant.succeeded()) << error.message;
      EXPECT_EQ(mutant.value().logical_mutation_count, 1u);
      EXPECT_EQ(mutant.value().selection.site(), site);
      EXPECT_TRUE(test::legacy_atomic_output_matches(source, 0, 8, mutant.value().image));
      const AmdGpuCodeObject output(mutant.value().image.data(), mutant.value().image.size());
      ASSERT_TRUE(output.is_valid());
      const Section &text = *output.text_sections().front();
      const std::optional<uint64_t> guest = test::legacy_atomic_guest(text, 0, target.target);
      ASSERT_TRUE(guest);
      ASSERT_LE(*guest + 8, text.size());
      std::array<uint32_t, 2> actual = {};
      std::memcpy(actual.data(), text.data() + *guest, sizeof(actual));
      const std::array<uint32_t, 2> literal_expected =
          test::legacy_atomic_words(target, form, offset + 8, returning);
      EXPECT_EQ(actual, literal_expected);
      const FailureOr<AtomicResult> independent =
          apply_atomic_wrong_address(source, ordinal.value(), 4, error);
      ASSERT_TRUE(independent.succeeded()) << error.message;
      EXPECT_TRUE(test::legacy_atomic_output_matches(source, 0, 4, independent.value().image));
      EXPECT_NE(independent.value().image, mutant.value().image);
      EXPECT_TRUE(std::ranges::equal(source.image(), image));
    }
  }
}

TEST_P(FaultLegacyAtomic, SignednessDeltaBoundsAndUnalignedSourcesUseEachNativeField) {
  const LegacyAtomicTargetCase &target = GetParam();
  for (LegacyAtomicForm form : test::kLegacyAtomicForms) {
    SCOPED_TRACE(test::legacy_atomic_mnemonic(target, form));
    const std::vector<uint8_t> image = test::make_legacy_atomic_elf(target, form, 0);
    const FaultMutationContext source(image);
    FaultMutationError error;
    const FailureOr<AtomicSelection> selected = select_legacy(source, error);
    ASSERT_TRUE(selected.succeeded()) << error.message;
    for (std::optional<uint32_t> delta :
         {std::optional<uint32_t>{}, std::optional<uint32_t>{0}, std::optional<uint32_t>{2},
          std::optional<uint32_t>{0x800000}, std::optional<uint32_t>{UINT32_MAX}}) {
      EXPECT_TRUE(apply_atomic_wrong_address(source, selected.value(), delta, error).failed());
      EXPECT_EQ(error.code, FaultMutationErrorCode::InvalidParameters);
      EXPECT_TRUE(std::ranges::equal(source.image(), image));
    }
    for (uint32_t delta : {4096u, 0x7ffffcu}) {
      EXPECT_TRUE(apply_atomic_wrong_address(source, selected.value(), delta, error).failed());
      EXPECT_EQ(error.code, FaultMutationErrorCode::InvalidParameters);
    }
    const FailureOr<AtomicResult> largest =
        apply_atomic_wrong_address(source, selected.value(), 4092, error);
    ASSERT_TRUE(largest.succeeded()) << error.message;
    EXPECT_TRUE(test::legacy_atomic_output_matches(source, 0, 4092, largest.value().image));
    const std::array<int32_t, 4> offsets = {2, 4091,
                                            test::legacy_atomic_signed(target, form) ? -4096 : 0,
                                            test::legacy_atomic_signed(target, form) ? -8 : 4};
    for (int32_t offset : offsets) {
      SCOPED_TRACE(offset);
      const std::vector<uint8_t> original = test::make_legacy_atomic_elf(target, form, offset);
      const FaultMutationContext valid(original);
      const FailureOr<AtomicSelection> selection = select_legacy(valid, error);
      ASSERT_TRUE(selection.succeeded()) << error.message;
      EXPECT_EQ(selection.value().site().byte_offset, offset);
      const FailureOr<AtomicResult> mutant =
          apply_atomic_wrong_address(valid, selection.value(), 4, error);
      ASSERT_TRUE(mutant.succeeded()) << error.message;
      EXPECT_TRUE(test::legacy_atomic_output_matches(valid, 0, 4, mutant.value().image));
      EXPECT_TRUE(std::ranges::equal(valid.image(), original));
    }
    for (int32_t offset : {4092, 4095}) {
      const std::vector<uint8_t> original = test::make_legacy_atomic_elf(target, form, offset);
      const FaultMutationContext bounded(original);
      const FailureOr<AtomicSelection> selection = select_legacy(bounded, error);
      ASSERT_TRUE(selection.succeeded()) << error.message;
      EXPECT_TRUE(apply_atomic_wrong_address(bounded, selection.value(), 4, error).failed());
      EXPECT_EQ(error.code, FaultMutationErrorCode::InvalidParameters);
      EXPECT_TRUE(std::ranges::equal(bounded.image(), original));
    }
    const FailureOr<AtomicResult> after_failures =
        apply_atomic_wrong_address(source, selected.value(), 4, error);
    ASSERT_TRUE(after_failures.succeeded()) << error.message;
    EXPECT_TRUE(test::legacy_atomic_output_matches(source, 0, 4, after_failures.value().image));
    EXPECT_TRUE(std::ranges::equal(source.image(), image));
  }
}

TEST_P(FaultLegacyAtomic, UnsupportedWidthDoesNotRenumberSupportedLegacyForms) {
  const LegacyAtomicTargetCase &target = GetParam();
  const std::array<uint32_t, 2> wide = test::legacy_wide_ds_atomic_words(target);
  std::unique_ptr<Decoder> decoder = Decoder::create(default_isa_target_registry(), target.target);
  ASSERT_NE(decoder, nullptr);
  const DecodeResult decoded = decoder->decode_window(wide, 0);
  ASSERT_TRUE(decoded.succeeded());
  EXPECT_EQ(decoded.value()->size(), 8);
  EXPECT_EQ(decoded.value()->mnemonic(), "ds_add_u64");
  const std::vector<uint8_t> image = test::make_mixed_legacy_atomic_elf(target);
  const FaultMutationContext source(image);
  FaultMutationError error;
  const FailureOr<AtomicInventory> inventory =
      discover_atomic_wrong_address(source, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().sites().size(), 4u);
  EXPECT_EQ(inventory.value().sites().front().form, "ds_add_u64");
  for (size_t ordinal = 0; ordinal < 4; ++ordinal) {
    const AtomicSite &site = inventory.value().sites()[ordinal];
    EXPECT_EQ(site.offset, ordinal * 8);
    EXPECT_EQ(site.address_supported, ordinal != 0);
    const FailureOr<AtomicSelection> selected = select_atomic_wrong_address(
        inventory.value(), {.exact = std::nullopt, .ordinal = ordinal}, error);
    ASSERT_TRUE(selected.succeeded()) << error.message;
    const FailureOr<AtomicResult> mutant =
        apply_atomic_wrong_address(source, selected.value(), 8, error);
    if (ordinal == 0) {
      EXPECT_TRUE(mutant.failed());
      EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm);
    } else {
      ASSERT_TRUE(mutant.succeeded()) << error.message;
      EXPECT_TRUE(test::legacy_atomic_output_matches(source, ordinal * 8, 8, mutant.value().image));
    }
  }
  EXPECT_TRUE(
      select_atomic_wrong_address(inventory.value(), {.exact = std::nullopt, .ordinal = 4}, error)
          .failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::OrdinalOutOfRange);
  EXPECT_TRUE(std::ranges::equal(source.image(), image));
}

TEST_P(FaultLegacyAtomic, InvalidScratchSegmentRefusesInventoryBeforeLaterSupportedSiteSelection) {
  const LegacyAtomicTargetCase &target = GetParam();
  const std::vector<uint8_t> image = test::make_invalid_scratch_legacy_atomic_elf(target);
  const FaultMutationContext source(image);
  ASSERT_TRUE(source.source().is_valid());
  const Section &text = *source.source().text_sections().front();
  std::array<uint32_t, 2> scratch = {};
  std::memcpy(scratch.data(), text.data(), sizeof(scratch));
  std::unique_ptr<Decoder> decoder = Decoder::create(default_isa_target_registry(), target.target);
  ASSERT_NE(decoder, nullptr);
  EXPECT_TRUE(decoder->decode_window(scratch, 0).failed());
  const std::array<uint32_t, 2> later =
      test::legacy_atomic_words(target, LegacyAtomicForm::BufferCas);
  const DecodeResult supported = decoder->decode_window(later, 8);
  ASSERT_TRUE(supported.succeeded());
  EXPECT_TRUE(supported.value()->mnemonic().starts_with("buffer_atomic"));
  FaultMutationError error;
  // Discovery validates the entire source before an exact or ordinal can be selected.
  EXPECT_TRUE(discover_atomic_wrong_address(source, "test_kernel", error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm);
  EXPECT_TRUE(std::ranges::equal(source.image(), image));
}

TEST_P(FaultLegacyAtomic, ExactIdentityAndCompleteSharedOwnersRemainBoundToSource) {
  const LegacyAtomicTargetCase &target = GetParam();
  const std::vector<uint8_t> image = test::make_shared_legacy_atomic_elf(target);
  const FaultMutationContext source(image), foreign(image);
  FaultMutationError error;
  const FailureOr<AtomicInventory> first =
      discover_atomic_wrong_address(source, "test_kernel0", error);
  const FailureOr<AtomicInventory> second =
      discover_atomic_wrong_address(source, "test_kernel1", error);
  ASSERT_TRUE(first.succeeded()) << error.message;
  ASSERT_TRUE(second.succeeded()) << error.message;
  ASSERT_EQ(first.value().sites().size(), 1u);
  EXPECT_EQ(first.value().sites().front().offset, 512u);
  ASSERT_EQ(first.value().sites().front().affected_owners.size(), 2u);
  EXPECT_EQ(first.value().sites().front().affected_owners,
            second.value().sites().front().affected_owners);
  const FailureOr<AtomicSelection> exact = select_atomic_wrong_address(first.value(), {}, error);
  ASSERT_TRUE(exact.succeeded()) << error.message;
  const FailureOr<AtomicSelection> rebound = select_atomic_wrong_address(
      second.value(), {.exact = exact.value(), .ordinal = SIZE_MAX}, error);
  ASSERT_TRUE(rebound.succeeded()) << error.message;
  ASSERT_NE(rebound.value().owner(), nullptr);
  EXPECT_EQ(rebound.value().owner()->entry, "test_kernel1");
  const FailureOr<AtomicResult> mutant =
      apply_atomic_wrong_address(source, rebound.value(), 8, error);
  ASSERT_TRUE(mutant.succeeded()) << error.message;
  EXPECT_TRUE(test::legacy_atomic_output_matches(source, 512, 8, mutant.value().image));
  EXPECT_TRUE(apply_atomic_wrong_address(foreign, exact.value(), 8, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  EXPECT_TRUE(
      select_atomic_wrong_address(first.value(), {.exact = AtomicSelection{}, .ordinal = 0}, error)
          .failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  const FaultMutationContext isolated(test::make_shared_legacy_atomic_elf(target, false));
  const FailureOr<AtomicInventory> unrelated =
      discover_atomic_wrong_address(isolated, "test_kernel1", error);
  ASSERT_TRUE(unrelated.succeeded()) << error.message;
  EXPECT_TRUE(unrelated.value().sites().empty());
  std::optional<AtomicInventory> expired_inventory;
  std::optional<AtomicSelection> expired_selection;
  {
    const FaultMutationContext temporary(image);
    const FailureOr<AtomicInventory> inventory =
        discover_atomic_wrong_address(temporary, "test_kernel0", error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    const FailureOr<AtomicSelection> selected =
        select_atomic_wrong_address(inventory.value(), {}, error);
    ASSERT_TRUE(selected.succeeded()) << error.message;
    expired_inventory = inventory.value();
    expired_selection = selected.value();
  }
  EXPECT_EQ(expired_selection->site().form,
            test::legacy_atomic_mnemonic(target, LegacyAtomicForm::GlobalCas));
  EXPECT_TRUE(select_atomic_wrong_address(*expired_inventory, {}, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  EXPECT_TRUE(apply_atomic_wrong_address(source, *expired_selection, 8, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  EXPECT_TRUE(std::ranges::equal(source.image(), image));
}

TEST_P(FaultLegacyAtomic, FormSegmentAndExtentMustMatchBeforeLegacyFieldAccess) {
  const LegacyAtomicTargetCase &target = GetParam();
  using fault_mutation_detail::read_legacy_atomic_byte_offset;
  using fault_mutation_detail::replace_legacy_atomic_byte_offset;
  const std::array<uint32_t, 2> global =
      test::legacy_atomic_words(target, LegacyAtomicForm::GlobalAdd, -4);
  const std::string mnemonic = test::legacy_atomic_mnemonic(target, LegacyAtomicForm::GlobalAdd);
  EXPECT_EQ(read_legacy_atomic_byte_offset(target.target, mnemonic, global), -4);
  for (std::string_view mismatch : {"flat_atomic_add", "buffer_atomic_add", "global_load_dword"}) {
    EXPECT_FALSE(read_legacy_atomic_byte_offset(target.target, mismatch, global));
    EXPECT_FALSE(replace_legacy_atomic_byte_offset(target.target, mismatch, global, 4));
  }
  const uint32_t shift = target.cdna ? 14 : 16;
  for (uint32_t segment : {1u, 3u}) {
    std::array<uint32_t, 2> words = global;
    words[0] = (words[0] & ~(3u << shift)) | (segment << shift);
    EXPECT_FALSE(read_legacy_atomic_byte_offset(target.target, mnemonic, words));
    EXPECT_FALSE(replace_legacy_atomic_byte_offset(target.target, mnemonic, words, 4));
  }
  const std::array<uint32_t, 3> extra = {global[0], global[1], 0x12345678u};
  for (std::span<const uint32_t> extent :
       {std::span<const uint32_t>{}, std::span<const uint32_t>(global).first(1),
        std::span<const uint32_t>(extra)}) {
    EXPECT_FALSE(read_legacy_atomic_byte_offset(target.target, mnemonic, extent));
    EXPECT_FALSE(replace_legacy_atomic_byte_offset(target.target, mnemonic, extent, 4));
  }
  for (rj_code_target_id_t unsupported :
       {ROCJITSU_CODE_TARGET_INVALID, ROCJITSU_CODE_TARGET_GFX90A, ROCJITSU_CODE_TARGET_GFX1250,
        ROCJITSU_CODE_TARGET_GFX1201}) {
    EXPECT_FALSE(read_legacy_atomic_byte_offset(unsupported, mnemonic, global));
    EXPECT_FALSE(replace_legacy_atomic_byte_offset(unsupported, mnemonic, global, 4));
  }
  EXPECT_FALSE(replace_legacy_atomic_byte_offset(target.target, mnemonic, global, -4097));
  EXPECT_FALSE(replace_legacy_atomic_byte_offset(target.target, mnemonic, global, 4096));
}

TEST_P(FaultLegacyAtomic, IndependentOracleRejectsOffsetOperandCacheWaitAndMappingCorruption) {
  const LegacyAtomicTargetCase &target = GetParam();
  const FaultMutationContext source(
      test::make_legacy_atomic_elf(target, LegacyAtomicForm::GlobalCas, -4));
  FaultMutationError error;
  const FailureOr<AtomicSelection> selection = select_legacy(source, error);
  ASSERT_TRUE(selection.succeeded()) << error.message;
  const FailureOr<AtomicResult> mutant =
      apply_atomic_wrong_address(source, selection.value(), 8, error);
  ASSERT_TRUE(mutant.succeeded()) << error.message;
  ASSERT_TRUE(test::legacy_atomic_output_matches(source, 0, 8, mutant.value().image));
  const AmdGpuCodeObject output(mutant.value().image.data(), mutant.value().image.size());
  ASSERT_TRUE(output.is_valid());
  const Section &text = *output.text_sections().front();
  const std::optional<uint64_t> guest = test::legacy_atomic_guest(text, 0, target.target);
  ASSERT_TRUE(guest);
  ASSERT_LE(*guest + 12, text.size());
  struct Corruption {
    uint64_t byte;
    uint8_t mask;
  };
  const std::array<Corruption, 9> corruptions = {{
      {.byte = *guest, .mask = 1u},
      {.byte = *guest + 1, .mask = target.cdna ? uint8_t{0x20} : uint8_t{0x80}},
      {.byte = *guest + 2, .mask = 4u},
      {.byte = *guest + 4, .mask = 1u},
      {.byte = *guest + 5, .mask = 1u},
      {.byte = *guest + 7, .mask = 1u},
      {.byte = *guest + 8, .mask = 1u},
      {.byte = 0, .mask = 1u},
      {.byte = 8, .mask = 1u},
  }};
  for (const Corruption &corruption : corruptions) {
    std::vector<uint8_t> changed = mutant.value().image;
    changed[text.sectionOffset() + corruption.byte] ^= corruption.mask;
    EXPECT_FALSE(test::legacy_atomic_output_matches(source, 0, 8, changed));
  }
}

INSTANTIATE_TEST_SUITE_P(Targets, FaultLegacyAtomic,
                         ::testing::ValuesIn(test::kLegacyAtomicTargetCases),
                         [](const ::testing::TestParamInfo<LegacyAtomicTargetCase> &info) {
                           return info.param.name;
                         });

} // namespace
} // namespace rocjitsu
