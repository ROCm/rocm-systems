// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "atomic_cdna_exchange_fixture.h"
#include "atomic_fixture.h"
#include "atomic_native_inventory_fixture.h"
#include "rocjitsu/code/patch/fault_injection/atomic_wrong_address.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu {
namespace {

class FaultAtomicWrongAddress : public ::testing::TestWithParam<test::AtomicTargetCase> {};

TEST(FaultAtomicWrongAddressLifetime, LegacyMnemonicMetadataSurvivesContextDestruction) {
  std::optional<AtomicInventory> inventory;
  std::optional<AtomicSelection> selection;
  std::optional<AtomicSite> site;
  FaultMutationError error;
  {
    std::vector<uint8_t> image = test::make_amdgpu_kernel_elf(
        {0xdd098000u, 0x002e0100u, build_s_endpgm(IsaFamily::Cdna4).value()}, 0, 3,
        EF_AMDGPU_MACH_AMDGCN_GFX950);
    image[EI_DATA] = ELFDATA2LSB;
    image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
    const FaultMutationContext context(image);
    const FailureOr<AtomicInventory> found =
        discover_atomic_wrong_address(context, "test_kernel", error);
    ASSERT_TRUE(found.succeeded()) << error.message;
    ASSERT_EQ(found.value().sites().size(), 1u);
    EXPECT_EQ(found.value().sites().front().form, "global_atomic_add");
    const FailureOr<AtomicSelection> selected =
        select_atomic_wrong_address(found.value(), {}, error);
    ASSERT_TRUE(selected.succeeded()) << error.message;
    inventory = found.value();
    selection = selected.value();
    site = selected.value().site();
  }
  EXPECT_EQ(inventory->sites().front().form, "global_atomic_add");
  EXPECT_EQ(selection->site().form, "global_atomic_add");
  EXPECT_EQ(site->form, "global_atomic_add");
  EXPECT_TRUE(select_atomic_wrong_address(*inventory, {}, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
}

TEST_P(FaultAtomicWrongAddress, FlatGlobalReturningAndCasPreserveGuestFields) {
  const test::AtomicTargetCase &fixture = GetParam();
  const std::vector<AtomicCapability> capabilities = atomic_wrong_address_capabilities();
  for (test::AtomicTestForm form :
       {test::AtomicTestForm::FlatAdd, test::AtomicTestForm::GlobalAdd,
        test::AtomicTestForm::FlatCas, test::AtomicTestForm::GlobalCas}) {
    SCOPED_TRACE(static_cast<int>(form));
    const bool global =
        form == test::AtomicTestForm::GlobalAdd || form == test::AtomicTestForm::GlobalCas;
    EXPECT_TRUE(std::ranges::any_of(capabilities, [&](const AtomicCapability &capability) {
      return capability.target == fixture.target && capability.wave_size == fixture.wave &&
             capability.form == (global ? "global_atomic" : "flat_atomic");
    }));
    const size_t modifier_states = fixture.arch == IsaFamily::Cdna5 ? 2 : 1;
    for (size_t scale = 0; scale < modifier_states; ++scale) {
      SCOPED_TRACE(scale);
      const std::vector<uint8_t> image = test::make_atomic_elf(fixture, form, -4, scale != 0);
      const FaultMutationContext source(image);
      FaultMutationError error;
      const FailureOr<AtomicInventory> inventory =
          discover_atomic_wrong_address(source, "test_kernel", error);
      ASSERT_TRUE(inventory.succeeded()) << error.message;
      ASSERT_EQ(inventory.value().sites().size(), 1u);
      const AtomicSite &site = inventory.value().sites().front();
      EXPECT_EQ(site.offset, 0u);
      EXPECT_EQ(site.instruction_size, 12u);
      ASSERT_TRUE(site.byte_offset);
      EXPECT_EQ(*site.byte_offset, -4);
      EXPECT_TRUE(site.address_supported);
      ASSERT_EQ(site.affected_owners.size(), 1u);
      EXPECT_EQ(site.affected_owners.front().entry, "test_kernel");
      const FailureOr<AtomicSelection> ordinal =
          select_atomic_wrong_address(inventory.value(), {}, error);
      ASSERT_TRUE(ordinal.succeeded()) << error.message;
      EXPECT_EQ(ordinal.value().kind(), FaultMutationKind::AtomicWrongAddress);
      const FailureOr<AtomicSelection> exact = select_atomic_wrong_address(
          inventory.value(),
          {.exact = ordinal.value(), .ordinal = std::numeric_limits<size_t>::max()}, error);
      ASSERT_TRUE(exact.succeeded()) << error.message;
      const FailureOr<AtomicResult> mutant =
          apply_atomic_wrong_address(source, exact.value(), 8, error);
      ASSERT_TRUE(mutant.succeeded()) << error.message;
      EXPECT_EQ(mutant.value().logical_mutation_count, 1u);
      EXPECT_EQ(mutant.value().selection.site().offset, 0u);
      EXPECT_TRUE(test::atomic_output_matches(source, 0, 8, mutant.value().image));
      const FailureOr<AtomicResult> by_ordinal =
          apply_atomic_wrong_address(source, ordinal.value(), 8, error);
      ASSERT_TRUE(by_ordinal.succeeded()) << error.message;
      EXPECT_EQ(by_ordinal.value().image, mutant.value().image);
      EXPECT_TRUE(std::ranges::equal(source.image(), image));
    }
  }
}

TEST_P(FaultAtomicWrongAddress, DeltaAlignmentBoundsAndOverflowRejectWithoutMutation) {
  const test::AtomicTargetCase &fixture = GetParam();
  const std::vector<uint8_t> image = test::make_atomic_elf(fixture);
  const FaultMutationContext source(image);
  FaultMutationError error;
  const FailureOr<AtomicInventory> inventory =
      discover_atomic_wrong_address(source, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const FailureOr<AtomicSelection> selection =
      select_atomic_wrong_address(inventory.value(), {}, error);
  ASSERT_TRUE(selection.succeeded()) << error.message;
  for (std::optional<uint32_t> delta :
       {std::optional<uint32_t>{}, std::optional<uint32_t>{0}, std::optional<uint32_t>{2},
        std::optional<uint32_t>{0x800000}, std::optional<uint32_t>{UINT32_MAX}}) {
    EXPECT_TRUE(apply_atomic_wrong_address(source, selection.value(), delta, error).failed());
    EXPECT_EQ(error.code, FaultMutationErrorCode::InvalidParameters);
    EXPECT_TRUE(std::ranges::equal(source.image(), image));
  }
  const FailureOr<AtomicResult> valid =
      apply_atomic_wrong_address(source, selection.value(), 4, error);
  ASSERT_TRUE(valid.succeeded()) << error.message;
  EXPECT_TRUE(test::atomic_output_matches(source, 0, 4, valid.value().image));

  for (int32_t offset : {0x7ffffc, 0x7fffff}) {
    SCOPED_TRACE(offset);
    const std::vector<uint8_t> invalid =
        test::make_atomic_elf(fixture, test::AtomicTestForm::FlatAdd, offset);
    const FaultMutationContext bounded(invalid);
    const FailureOr<AtomicInventory> found =
        discover_atomic_wrong_address(bounded, "test_kernel", error);
    ASSERT_TRUE(found.succeeded()) << error.message;
    const FailureOr<AtomicSelection> selected =
        select_atomic_wrong_address(found.value(), {}, error);
    ASSERT_TRUE(selected.succeeded()) << error.message;
    EXPECT_TRUE(apply_atomic_wrong_address(bounded, selected.value(), 4, error).failed());
    EXPECT_EQ(error.code, FaultMutationErrorCode::InvalidParameters);
    EXPECT_TRUE(std::ranges::equal(bounded.image(), invalid));
  }
  // FlatLike requires aligned deltas, not an aligned encoded starting offset.
  const FaultMutationContext unaligned(
      test::make_atomic_elf(fixture, test::AtomicTestForm::FlatAdd, 2));
  const FailureOr<AtomicInventory> unaligned_inventory =
      discover_atomic_wrong_address(unaligned, "test_kernel", error);
  ASSERT_TRUE(unaligned_inventory.succeeded()) << error.message;
  const FailureOr<AtomicSelection> unaligned_selection =
      select_atomic_wrong_address(unaligned_inventory.value(), {}, error);
  ASSERT_TRUE(unaligned_selection.succeeded()) << error.message;
  const FailureOr<AtomicResult> shifted =
      apply_atomic_wrong_address(unaligned, unaligned_selection.value(), 4, error);
  ASSERT_TRUE(shifted.succeeded()) << error.message;
  EXPECT_TRUE(test::atomic_output_matches(unaligned, 0, 4, shifted.value().image));
}

TEST_P(FaultAtomicWrongAddress, UnsupportedBufferAndDsRemainInAtomicOrdinalUnion) {
  const test::AtomicTargetCase &fixture = GetParam();
  const FaultMutationContext source(test::make_mixed_atomic_elf(fixture));
  FaultMutationError error;
  const FailureOr<AtomicInventory> inventory =
      discover_atomic_wrong_address(source, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().sites().size(), 3u);
  EXPECT_EQ(inventory.value().sites()[0].offset, 0u);
  EXPECT_EQ(inventory.value().sites()[0].instruction_size, 12u);
  EXPECT_EQ(inventory.value().sites()[0].form, "buffer_atomic_add_u32");
  EXPECT_EQ(inventory.value().sites()[1].offset, 12u);
  EXPECT_EQ(inventory.value().sites()[1].instruction_size, 8u);
  EXPECT_EQ(inventory.value().sites()[1].form, "ds_add_rtn_u32");
  EXPECT_EQ(inventory.value().sites()[2].offset, 20u);
  EXPECT_EQ(inventory.value().sites()[2].form, "flat_atomic_add_u32");
  for (size_t ordinal = 0; ordinal < 3; ++ordinal) {
    const FailureOr<AtomicSelection> selected = select_atomic_wrong_address(
        inventory.value(), {.exact = std::nullopt, .ordinal = ordinal}, error);
    ASSERT_TRUE(selected.succeeded()) << error.message;
    EXPECT_EQ(selected.value().site().address_supported, ordinal == 2);
    const FailureOr<AtomicResult> mutant =
        apply_atomic_wrong_address(source, selected.value(), 8, error);
    if (ordinal < 2) {
      EXPECT_TRUE(mutant.failed());
      EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm);
    } else {
      ASSERT_TRUE(mutant.succeeded()) << error.message;
      EXPECT_TRUE(test::atomic_output_matches(source, 20, 8, mutant.value().image));
    }
  }
  EXPECT_TRUE(
      select_atomic_wrong_address(inventory.value(), {.exact = std::nullopt, .ordinal = 3}, error)
          .failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::OrdinalOutOfRange);
}

TEST_P(FaultAtomicWrongAddress, InvalidExactForeignContextAndExpiredHandlesFailClosed) {
  const std::vector<uint8_t> image = test::make_atomic_elf(GetParam());
  const FaultMutationContext source(image), foreign(image);
  FaultMutationError error;
  const FailureOr<AtomicInventory> inventory =
      discover_atomic_wrong_address(source, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const FailureOr<AtomicSelection> selected =
      select_atomic_wrong_address(inventory.value(), {}, error);
  ASSERT_TRUE(selected.succeeded()) << error.message;
  EXPECT_TRUE(select_atomic_wrong_address(inventory.value(),
                                          {.exact = AtomicSelection{}, .ordinal = 0}, error)
                  .failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  EXPECT_TRUE(apply_atomic_wrong_address(foreign, selected.value(), 4, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  const FailureOr<AtomicInventory> foreign_inventory =
      discover_atomic_wrong_address(foreign, "test_kernel", error);
  ASSERT_TRUE(foreign_inventory.succeeded());
  EXPECT_TRUE(select_atomic_wrong_address(foreign_inventory.value(),
                                          {.exact = selected.value(), .ordinal = 0}, error)
                  .failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  std::optional<AtomicInventory> expired_inventory;
  std::optional<AtomicSelection> expired_selection;
  {
    const FaultMutationContext temporary(image);
    const FailureOr<AtomicInventory> found =
        discover_atomic_wrong_address(temporary, "test_kernel", error);
    ASSERT_TRUE(found.succeeded());
    const FailureOr<AtomicSelection> exact = select_atomic_wrong_address(found.value(), {}, error);
    ASSERT_TRUE(exact.succeeded());
    expired_inventory = found.value();
    expired_selection = exact.value();
  }
  EXPECT_TRUE(select_atomic_wrong_address(*expired_inventory, {}, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  EXPECT_TRUE(apply_atomic_wrong_address(source, *expired_selection, 4, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  EXPECT_TRUE(discover_atomic_wrong_address(source, "test_", error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
}

TEST_P(FaultAtomicWrongAddress, OracleRejectsOffsetDataWaitAndContinuationCorruption) {
  const FaultMutationContext source(test::make_atomic_elf(GetParam()));
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
  const AmdGpuCodeObject output(mutant.value().image.data(), mutant.value().image.size());
  ASSERT_TRUE(output.is_valid());
  const Section &text = *output.text_sections().front();
  const std::optional<uint64_t> guest = test::atomic_emitted_guest(text, 0, GetParam().target);
  ASSERT_TRUE(guest);
  for (uint64_t text_byte : {*guest + 9, *guest + 4, uint64_t{12}, *guest + 12}) {
    SCOPED_TRACE(text_byte);
    std::vector<uint8_t> corrupt = mutant.value().image;
    corrupt[text.sectionOffset() + text_byte] ^= 1;
    EXPECT_FALSE(test::atomic_output_matches(source, 0, 8, corrupt));
  }
}

TEST(FaultAtomicWrongAddressLifetime, InputSnapshotAndIndependentMutantsStayPristine) {
  const test::AtomicTargetCase &fixture = test::kAtomicTargetCases.front();
  const std::array<uint32_t, 3> first = test::atomic_test_words(fixture);
  const std::array<uint32_t, 3> second =
      test::atomic_test_words(fixture, test::AtomicTestForm::GlobalCas);
  std::vector<uint8_t> input = test::make_atomic_sequence_elf(
      fixture, {first[0], first[1], first[2], second[0], second[1], second[2]});
  const std::vector<uint8_t> pristine = input;
  const FaultMutationContext source(input);
  input[0] = 0;
  EXPECT_TRUE(std::ranges::equal(source.image(), pristine));
  FaultMutationError error;
  const FailureOr<AtomicInventory> inventory =
      discover_atomic_wrong_address(source, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().sites().size(), 2u);
  for (size_t ordinal = 0; ordinal < 2; ++ordinal) {
    const FailureOr<AtomicSelection> selected = select_atomic_wrong_address(
        inventory.value(), {.exact = std::nullopt, .ordinal = ordinal}, error);
    ASSERT_TRUE(selected.succeeded());
    const FailureOr<AtomicResult> mutant =
        apply_atomic_wrong_address(source, selected.value(), 8, error);
    ASSERT_TRUE(mutant.succeeded()) << error.message;
    EXPECT_TRUE(test::atomic_output_matches(source, ordinal * 12, 8, mutant.value().image));
    const FailureOr<AtomicResult> repeated =
        apply_atomic_wrong_address(source, selected.value(), 8, error);
    ASSERT_TRUE(repeated.succeeded());
    EXPECT_EQ(repeated.value().image, mutant.value().image);
  }
  EXPECT_TRUE(std::ranges::equal(source.image(), pristine));
}

TEST(FaultAtomicWrongAddressOwners, DirectMutationAndRecoveredSelectionKeepExactIdentity) {
  const test::AtomicTargetCase &fixture = test::kAtomicTargetCases[3];
  for (bool indirect : {false, true}) {
    SCOPED_TRACE(indirect);
    const std::vector<uint8_t> image = test::make_shared_atomic_elf(fixture, indirect);
    const FaultMutationContext source(image);
    FaultMutationError error;
    const FailureOr<AtomicInventory> first =
        discover_atomic_wrong_address(source, "test_kernel0", error);
    const FailureOr<AtomicInventory> second =
        discover_atomic_wrong_address(source, "test_kernel1", error);
    ASSERT_TRUE(first.succeeded()) << error.message;
    ASSERT_TRUE(second.succeeded()) << error.message;
    ASSERT_EQ(first.value().sites().size(), 1u);
    ASSERT_EQ(second.value().sites().size(), 1u);
    EXPECT_EQ(first.value().sites().front().offset, 512u);
    ASSERT_EQ(first.value().sites().front().affected_owners.size(), 2u);
    EXPECT_EQ(first.value().sites().front().affected_owners,
              second.value().sites().front().affected_owners);
    const FailureOr<AtomicSelection> selected =
        select_atomic_wrong_address(first.value(), {}, error);
    ASSERT_TRUE(selected.succeeded());
    const FailureOr<AtomicSelection> other_owner = select_atomic_wrong_address(
        second.value(), {.exact = selected.value(), .ordinal = SIZE_MAX}, error);
    ASSERT_TRUE(other_owner.succeeded()) << error.message;
    ASSERT_NE(other_owner.value().owner(), nullptr);
    EXPECT_EQ(other_owner.value().owner()->entry, "test_kernel1");
    const FailureOr<AtomicResult> mutant =
        apply_atomic_wrong_address(source, other_owner.value(), 8, error);
    if (indirect) {
      // The original recovered-call positive proves inventory and selection.
      // Retained getpc-derived addresses still obey the shared writer's guard.
      EXPECT_TRUE(mutant.failed());
      EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm);
      EXPECT_NE(error.message.find("replacement growth cannot preserve retained PC-relative"),
                std::string::npos);
    } else {
      ASSERT_TRUE(mutant.succeeded()) << error.message;
      EXPECT_TRUE(test::atomic_output_matches(source, 512, 8, mutant.value().image));
    }
    EXPECT_TRUE(std::ranges::equal(source.image(), image));
  }

  const FaultMutationContext isolated(test::make_shared_atomic_elf(fixture, false, false));
  FaultMutationError error;
  const FailureOr<AtomicInventory> owner =
      discover_atomic_wrong_address(isolated, "test_kernel0", error);
  const FailureOr<AtomicInventory> unrelated =
      discover_atomic_wrong_address(isolated, "test_kernel1", error);
  ASSERT_TRUE(owner.succeeded()) << error.message;
  ASSERT_TRUE(unrelated.succeeded()) << error.message;
  EXPECT_TRUE(unrelated.value().sites().empty());
  const FailureOr<AtomicSelection> selected = select_atomic_wrong_address(owner.value(), {}, error);
  ASSERT_TRUE(selected.succeeded());
  EXPECT_TRUE(select_atomic_wrong_address(unrelated.value(), {}, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::NoCandidates);
  EXPECT_TRUE(select_atomic_wrong_address(unrelated.value(),
                                          {.exact = selected.value(), .ordinal = 0}, error)
                  .failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
}

class FaultNativeCdnaAtomicInventory : public ::testing::TestWithParam<test::AtomicTargetCase> {};

TEST_P(FaultNativeCdnaAtomicInventory,
       NativeExchangesKeepExactAndSourceOrderBeforeAddressAdmission) {
  const test::AtomicTargetCase &target = GetParam();
  const std::vector<uint8_t> image = test::make_native_cdna_exchange_inventory_elf(target);
  const FaultMutationContext source(image);
  std::unique_ptr<Decoder> decoder = Decoder::create(default_isa_target_registry(), target.target);
  ASSERT_NE(decoder, nullptr);
  for (test::NativeCdnaExchangeForm form : test::kNativeCdnaExchangeInventoryForms) {
    const std::array<uint32_t, 2> words = test::native_cdna_exchange_words(target, form);
    const DecodeResult decoded = decoder->decode_window(words, 0);
    ASSERT_TRUE(decoded.succeeded());
    EXPECT_EQ(decoded.value()->mnemonic(), test::native_cdna_exchange_mnemonic(form));
    EXPECT_EQ(decoded.value()->size(), 8);
  }
  FaultMutationError error;
  const FailureOr<AtomicInventory> inventory =
      discover_atomic_wrong_address(source, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().sites().size(), test::kNativeCdnaExchangeInventoryForms.size());
  for (size_t ordinal = 0; ordinal < test::kNativeCdnaExchangeInventoryForms.size(); ++ordinal) {
    const test::NativeCdnaExchangeForm form = test::kNativeCdnaExchangeInventoryForms[ordinal];
    SCOPED_TRACE(test::native_cdna_exchange_mnemonic(form));
    const AtomicSite &site = inventory.value().sites()[ordinal];
    EXPECT_EQ(site.offset, ordinal * 8);
    EXPECT_EQ(site.instruction_size, 8u);
    EXPECT_EQ(site.form, test::native_cdna_exchange_mnemonic(form));
    const FailureOr<AtomicSelection> selected = select_atomic_wrong_address(
        inventory.value(), {.exact = std::nullopt, .ordinal = ordinal}, error);
    ASSERT_TRUE(selected.succeeded()) << error.message;
    const FailureOr<AtomicSelection> exact = select_atomic_wrong_address(
        inventory.value(), {.exact = selected.value(), .ordinal = SIZE_MAX}, error);
    ASSERT_TRUE(exact.succeeded()) << error.message;
    EXPECT_EQ(exact.value().site(), site);
    // These controls remain valid after later slices admit 32-bit DS address recipes.
    if (form != test::NativeCdnaExchangeForm::ExchangeB32 &&
        form != test::NativeCdnaExchangeForm::Exchange2B32) {
      EXPECT_FALSE(site.address_supported);
      EXPECT_TRUE(apply_atomic_wrong_address(source, exact.value(), 8, error).failed());
      EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm);
    }
  }
  const FailureOr<AtomicSelection> first =
      select_atomic_wrong_address(inventory.value(), {}, error);
  ASSERT_TRUE(first.succeeded()) << error.message;
  EXPECT_EQ(first.value().site().offset, 0u);
  EXPECT_TRUE(apply_atomic_wrong_address(source, first.value(), 8, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm);
  EXPECT_TRUE(select_atomic_wrong_address(inventory.value(),
                                          {.exact = AtomicSelection{}, .ordinal = 6}, error)
                  .failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  EXPECT_TRUE(
      select_atomic_wrong_address(inventory.value(), {.exact = std::nullopt, .ordinal = 7}, error)
          .failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::OrdinalOutOfRange);
  EXPECT_TRUE(std::ranges::equal(source.image(), image));
}

INSTANTIATE_TEST_SUITE_P(Targets, FaultNativeCdnaAtomicInventory,
                         ::testing::ValuesIn(test::kNativeCdnaExchangeCases),
                         [](const ::testing::TestParamInfo<test::AtomicTargetCase> &info) {
                           return info.param.name;
                         });

class FaultNativeAtomicInventory : public ::testing::TestWithParam<test::AtomicTargetCase> {};

TEST_P(FaultNativeAtomicInventory, SemanticAtomicsAndLegacyPrefixesPrecedeAddressAdmission) {
  const test::AtomicTargetCase &target = GetParam();
  const std::vector<test::NativeAtomicInventorySample> samples =
      test::native_atomic_inventory_samples(target);
  const std::vector<uint8_t> image = test::make_native_atomic_inventory_elf(target);
  const FaultMutationContext source(image);
  std::unique_ptr<Decoder> decoder = Decoder::create(default_isa_target_registry(), target.target);
  ASSERT_NE(decoder, nullptr);
  uint64_t flat_offset = 0;
  for (const test::NativeAtomicInventorySample &sample : samples) {
    SCOPED_TRACE(sample.form);
    const DecodeResult decoded = decoder->decode_window(sample.words, flat_offset);
    ASSERT_TRUE(decoded.succeeded());
    EXPECT_EQ(decoded.value()->mnemonic(), sample.form);
    EXPECT_EQ(decoded.value()->size(), sample.words.size() * sizeof(uint32_t));
    EXPECT_EQ(decoded.value()->amdgpu_memory_kind(), sample.memory_kind);
    flat_offset += sample.words.size() * sizeof(uint32_t);
  }
  flat_offset += sizeof(uint32_t); // The non-atomic nop is absent from the inventory.
  const bool has_flat = test::native_atomic_inventory_has_flat(target);
  FaultMutationError error;
  const FailureOr<AtomicInventory> inventory =
      discover_atomic_wrong_address(source, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().sites().size(), samples.size() + static_cast<size_t>(has_flat));
  uint64_t offset = 0;
  for (size_t ordinal = 0; ordinal < samples.size(); ++ordinal) {
    SCOPED_TRACE(samples[ordinal].form);
    const AtomicSite &site = inventory.value().sites()[ordinal];
    EXPECT_EQ(site.offset, offset);
    EXPECT_EQ(site.instruction_size, samples[ordinal].words.size() * sizeof(uint32_t));
    EXPECT_EQ(site.form, samples[ordinal].form);
    EXPECT_FALSE(site.address_supported);
    const FailureOr<AtomicSelection> selected = select_atomic_wrong_address(
        inventory.value(), {.exact = std::nullopt, .ordinal = ordinal}, error);
    ASSERT_TRUE(selected.succeeded()) << error.message;
    const FailureOr<AtomicSelection> exact = select_atomic_wrong_address(
        inventory.value(), {.exact = selected.value(), .ordinal = SIZE_MAX}, error);
    ASSERT_TRUE(exact.succeeded()) << error.message;
    EXPECT_EQ(exact.value().site(), site);
    EXPECT_TRUE(apply_atomic_wrong_address(source, exact.value(), 8, error).failed());
    EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm);
    offset += samples[ordinal].words.size() * sizeof(uint32_t);
  }
  if (has_flat) {
    const FailureOr<AtomicSelection> selected = select_atomic_wrong_address(
        inventory.value(), {.exact = std::nullopt, .ordinal = samples.size()}, error);
    ASSERT_TRUE(selected.succeeded()) << error.message;
    EXPECT_EQ(selected.value().site().offset, flat_offset);
    EXPECT_TRUE(selected.value().site().address_supported);
    const FailureOr<AtomicResult> mutant =
        apply_atomic_wrong_address(source, selected.value(), 8, error);
    ASSERT_TRUE(mutant.succeeded()) << error.message;
    EXPECT_TRUE(test::atomic_output_matches(source, flat_offset, 8, mutant.value().image));
  }
  EXPECT_TRUE(select_atomic_wrong_address(inventory.value(),
                                          {.exact = AtomicSelection{}, .ordinal = 0}, error)
                  .failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  EXPECT_TRUE(select_atomic_wrong_address(
                  inventory.value(),
                  {.exact = std::nullopt, .ordinal = inventory.value().sites().size()}, error)
                  .failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::OrdinalOutOfRange);
  EXPECT_TRUE(std::ranges::equal(source.image(), image));
}

INSTANTIATE_TEST_SUITE_P(Targets, FaultNativeAtomicInventory,
                         ::testing::ValuesIn(test::kNativeAtomicInventoryCases),
                         [](const ::testing::TestParamInfo<test::AtomicTargetCase> &info) {
                           return info.param.name;
                         });

INSTANTIATE_TEST_SUITE_P(Targets, FaultAtomicWrongAddress,
                         ::testing::ValuesIn(test::kAtomicTargetCases),
                         [](const ::testing::TestParamInfo<test::AtomicTargetCase> &info) {
                           return info.param.name;
                         });

} // namespace
} // namespace rocjitsu
