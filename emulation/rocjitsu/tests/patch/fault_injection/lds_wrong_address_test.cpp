// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "lds_fixture.h"
#include "rocjitsu/code/patch/fault_injection/drop_barrier.h"
#include "rocjitsu/code/patch/fault_injection/lds_wrong_address.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <vector>

namespace rocjitsu {
namespace {

TEST(FaultLdsWrongAddress, Rdna4Wave32BasicFormsPreserveGuestAndAllocation) {
  const std::vector<LdsCapability> capabilities = lds_wrong_address_capabilities();
  ASSERT_EQ(capabilities.size(), 14u);
  for (uint32_t machine : {EF_AMDGPU_MACH_AMDGCN_GFX1200, EF_AMDGPU_MACH_AMDGCN_GFX1201}) {
    SCOPED_TRACE(machine);
    const std::vector<uint8_t> image = test::make_lds_load_elf(machine, true, 0);
    const FaultMutationContext source(image), foreign(image);
    FaultMutationError error;
    const FailureOr<LdsInventory> inventory =
        discover_lds_wrong_address(source, "test_kernel", error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    ASSERT_EQ(inventory.value().sites().size(), 2u);
    for (size_t ordinal = 0; ordinal < inventory.value().sites().size(); ++ordinal) {
      SCOPED_TRACE(ordinal);
      const LdsSite &site = inventory.value().sites()[ordinal];
      EXPECT_TRUE(std::ranges::any_of(capabilities, [&](const LdsCapability &capability) {
        return capability.target == source.source().target_id() && capability.wave_size == 32 &&
               capability.form == site.form;
      }));
      const FailureOr<LdsSelection> selected = select_lds_wrong_address(
          inventory.value(), {.exact = std::nullopt, .ordinal = ordinal}, error);
      ASSERT_TRUE(selected.succeeded()) << error.message;
      const FailureOr<LdsSelection> exact = select_lds_wrong_address(
          inventory.value(),
          {.exact = selected.value(), .ordinal = std::numeric_limits<size_t>::max()}, error);
      ASSERT_TRUE(exact.succeeded()) << error.message;
      for (std::optional<uint32_t> replacement :
           {std::optional<uint32_t>{}, std::optional<uint32_t>{site.address_vgpr},
            std::optional<uint32_t>{8}, std::optional<uint32_t>{256}}) {
        EXPECT_TRUE(apply_lds_wrong_address(source, exact.value(), replacement, error).failed());
        EXPECT_EQ(error.code, FaultMutationErrorCode::InvalidParameters);
      }
      EXPECT_TRUE(apply_lds_wrong_address(foreign, exact.value(), 7, error).failed());
      EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
      const FailureOr<LdsResult> mutant = apply_lds_wrong_address(source, exact.value(), 7, error);
      ASSERT_TRUE(mutant.succeeded()) << error.message;
      EXPECT_TRUE(
          test::lds_output_matches(source, site.offset, 7, mutant.value().image, ordinal == 0));
      const FailureOr<LdsResult> by_ordinal =
          apply_lds_wrong_address(source, selected.value(), 7, error);
      ASSERT_TRUE(by_ordinal.succeeded()) << error.message;
      EXPECT_EQ(mutant.value().image, by_ordinal.value().image);
      EXPECT_TRUE(std::ranges::equal(source.image(), image));
    }
  }
}

TEST(FaultLdsWrongAddress, Rdna4LoadSelectionMutationAndControls) {
  for (uint32_t target : {EF_AMDGPU_MACH_AMDGCN_GFX1200, EF_AMDGPU_MACH_AMDGCN_GFX1201}) {
    SCOPED_TRACE(target);
    const std::vector<uint8_t> image = test::make_lds_load_elf(target);
    const FaultMutationContext context(image), foreign(image);
    FaultMutationError error;
    const FailureOr<LdsInventory> inventory =
        discover_lds_wrong_address(context, "test_kernel", error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    ASSERT_EQ(inventory.value().sites().size(), 2u);
    EXPECT_EQ(inventory.value().sites()[0].offset, 0u);
    EXPECT_EQ(inventory.value().sites()[1].offset, 8u);
    EXPECT_EQ(inventory.value().sites()[0].form, "ds_load_b32");
    EXPECT_EQ(inventory.value().sites()[1].form, "ds_store_b32");
    const FailureOr<LdsSelection> selection =
        select_lds_wrong_address(inventory.value(), {}, error);
    ASSERT_TRUE(selection.succeeded()) << error.message;
    const FailureOr<LdsSelection> exact = select_lds_wrong_address(
        inventory.value(),
        {.exact = selection.value(), .ordinal = std::numeric_limits<size_t>::max()}, error);
    ASSERT_TRUE(exact.succeeded()) << error.message;
    EXPECT_TRUE(apply_lds_wrong_address(context, exact.value(), std::nullopt, error).failed());
    EXPECT_TRUE(apply_lds_wrong_address(context, exact.value(), 2, error).failed());
    EXPECT_TRUE(apply_lds_wrong_address(context, exact.value(), 8, error).failed());
    EXPECT_TRUE(apply_lds_wrong_address(context, exact.value(), 256, error).failed());
    EXPECT_TRUE(apply_lds_wrong_address(foreign, exact.value(), 6, error).failed());
    EXPECT_TRUE(
        select_lds_wrong_address(inventory.value(), {.exact = LdsSelection{}, .ordinal = 0}, error)
            .failed());
    const FailureOr<LdsResult> mutant = apply_lds_wrong_address(context, exact.value(), 6, error);
    ASSERT_TRUE(mutant.succeeded()) << error.message;
    EXPECT_TRUE(test::lds_output_matches(context, 0, 6, mutant.value().image, true));
    std::vector<uint8_t> corrupt = mutant.value().image;
    const AmdGpuCodeObject output(corrupt.data(), corrupt.size());
    ASSERT_TRUE(output.is_valid());
    const std::optional<uint64_t> guest =
        test::lds_emitted_guest(*output.text_sections().front(), 0, context.source().target_id());
    ASSERT_TRUE(guest);
    corrupt[output.text_sections().front()->sectionOffset() + *guest + 7] ^= 1;
    EXPECT_FALSE(test::lds_output_matches(context, 0, 6, corrupt, true));
    const FailureOr<LdsResult> repeated = apply_lds_wrong_address(context, exact.value(), 6, error);
    ASSERT_TRUE(repeated.succeeded());
    EXPECT_EQ(mutant.value().image, repeated.value().image);
    const FailureOr<LdsSelection> store =
        select_lds_wrong_address(inventory.value(), {.exact = std::nullopt, .ordinal = 1}, error);
    ASSERT_TRUE(store.succeeded());
    const FailureOr<LdsResult> store_mutant =
        apply_lds_wrong_address(context, store.value(), 6, error);
    ASSERT_TRUE(store_mutant.succeeded()) << error.message;
    EXPECT_TRUE(test::lds_output_matches(context, 8, 6, store_mutant.value().image));
    EXPECT_TRUE(std::ranges::equal(context.image(), image));
    const FaultMutationContext wave32(test::make_lds_load_elf(target, true));
    const FailureOr<LdsInventory> wave32_inventory =
        discover_lds_wrong_address(wave32, "test_kernel", error);
    ASSERT_TRUE(wave32_inventory.succeeded()) << error.message;
    EXPECT_EQ(wave32_inventory.value().sites().size(), 2u);
    const FaultMutationContext allocation(test::make_lds_load_elf(target, false, 0));
    const FailureOr<LdsInventory> bounded =
        discover_lds_wrong_address(allocation, "test_kernel", error);
    ASSERT_TRUE(bounded.succeeded()) << error.message;
    const FailureOr<LdsSelection> bounded_selection =
        select_lds_wrong_address(bounded.value(), {}, error);
    ASSERT_TRUE(bounded_selection.succeeded());
    EXPECT_TRUE(apply_lds_wrong_address(allocation, bounded_selection.value(), 4, error).failed());
    EXPECT_TRUE(
        apply_lds_wrong_address(allocation, bounded_selection.value(), 3, error).succeeded());
    std::optional<LdsSelection> expired;
    {
      const FaultMutationContext temporary(image);
      const FailureOr<LdsInventory> found =
          discover_lds_wrong_address(temporary, "test_kernel", error);
      ASSERT_TRUE(found.succeeded());
      const FailureOr<LdsSelection> selected = select_lds_wrong_address(found.value(), {}, error);
      ASSERT_TRUE(selected.succeeded());
      expired = selected.value();
    }
    EXPECT_TRUE(apply_lds_wrong_address(context, *expired, 6, error).failed());

    const FaultMutationContext dual(test::make_lds_elf(1, false, target, true));
    const FailureOr<LdsInventory> dual_inventory =
        discover_lds_wrong_address(dual, "test_kernel", error);
    ASSERT_TRUE(dual_inventory.succeeded());
    ASSERT_EQ(dual_inventory.value().sites().size(), 1u);
    EXPECT_EQ(dual_inventory.value().sites().front().offset, 8u);

    std::vector<uint8_t> truncated = test::make_amdgpu_kernel_elf({0xd8d80004u}, 0, 3, target, 1);
    truncated[EI_DATA] = ELFDATA2LSB;
    truncated[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
    const FaultMutationContext incomplete(truncated);
    EXPECT_TRUE(discover_lds_wrong_address(incomplete, "test_kernel", error).failed());
    EXPECT_TRUE(std::ranges::equal(incomplete.image(), truncated));

    std::vector<uint32_t> long_words = {0xd8d80004u, 0x05000002u};
    long_words.insert(long_words.end(), 32769, 0xbf800000u);
    long_words.push_back(0xbfb00000u);
    std::vector<uint8_t> long_image = test::make_amdgpu_kernel_elf(long_words, 0, 3, target, 1);
    long_image[EI_DATA] = ELFDATA2LSB;
    long_image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
    const FaultMutationContext long_context(long_image);
    const FailureOr<LdsInventory> long_inventory =
        discover_lds_wrong_address(long_context, "test_kernel", error);
    ASSERT_TRUE(long_inventory.succeeded()) << error.message;
    const FailureOr<LdsSelection> long_selection =
        select_lds_wrong_address(long_inventory.value(), {}, error);
    ASSERT_TRUE(long_selection.succeeded());
    EXPECT_TRUE(apply_lds_wrong_address(long_context, long_selection.value(), 6, error).failed());
    EXPECT_TRUE(std::ranges::equal(long_context.image(), long_image));
  }
}

TEST(FaultLdsWrongAddress, Gfx1250StoreAndLoadAreAvailable) {
  const std::array<uint32_t, 2> store =
      cdna5::build_vds(cdna5::kDsStoreB32Vds, {.addr = 2, .data0 = 3});
  const std::array<uint32_t, 2> load =
      cdna5::build_vds(cdna5::kDsLoadB32Vds, {.addr = 2, .vdst = 5});
  const FaultMutationContext context(
      test::make_cdna5_lds_elf({store[0], store[1], load[0], load[1]}));
  FaultMutationError error;
  const FailureOr<LdsInventory> inventory =
      discover_lds_wrong_address(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().sites().size(), 2u);
  EXPECT_EQ(inventory.value().sites()[0].form, "ds_store_b32");
  EXPECT_EQ(inventory.value().sites()[1].form, "ds_load_b32");
}

TEST(FaultLdsWrongAddress, Gfx1250MutatesBothFormsAndChecksAddressBankAllocation) {
  constexpr std::array<uint32_t, 1> set_bank =
      cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 1});
  constexpr std::array<uint32_t, 2> store =
      cdna5::build_vds(cdna5::kDsStoreB32Vds, {.offset0 = 4, .addr = 2, .data0 = 3});
  constexpr std::array<uint32_t, 2> load =
      cdna5::build_vds(cdna5::kDsLoadB32Vds, {.offset0 = 8, .addr = 2, .vdst = 5});
  const std::vector<uint8_t> image =
      test::make_cdna5_lds_elf({set_bank[0], store[0], store[1], load[0], load[1]}, 16);
  const FaultMutationContext context(image);
  FaultMutationError error;
  const FailureOr<LdsInventory> inventory =
      discover_lds_wrong_address(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().sites().size(), 2u);
  for (size_t ordinal = 0; ordinal < 2; ++ordinal) {
    EXPECT_EQ(inventory.value().sites()[ordinal].address_bank, 1u);
    EXPECT_EQ(inventory.value().sites()[ordinal].form,
              ordinal == 0 ? "ds_store_b32" : "ds_load_b32");
    const FailureOr<LdsSelection> selection = select_lds_wrong_address(
        inventory.value(), {.exact = std::nullopt, .ordinal = ordinal}, error);
    ASSERT_TRUE(selection.succeeded()) << error.message;
    EXPECT_TRUE(apply_lds_wrong_address(context, selection.value(), std::nullopt, error).failed());
    EXPECT_TRUE(apply_lds_wrong_address(context, selection.value(), 2, error).failed());
    EXPECT_TRUE(apply_lds_wrong_address(context, selection.value(), 16, error).failed());
    EXPECT_TRUE(apply_lds_wrong_address(context, selection.value(), 256, error).failed());
    const FailureOr<LdsResult> result =
        apply_lds_wrong_address(context, selection.value(), 15, error);
    ASSERT_TRUE(result.succeeded()) << error.message;
    EXPECT_TRUE(test::lds_output_matches(context, ordinal == 0 ? 4 : 12, 15, result.value().image));
    std::vector<uint8_t> corrupted = result.value().image;
    const AmdGpuCodeObject mutant(corrupted.data(), corrupted.size());
    const std::optional<uint64_t> guest = test::lds_emitted_guest(
        *mutant.text_sections().front(), ordinal == 0 ? 4 : 12, ROCJITSU_CODE_TARGET_GFX1250);
    ASSERT_TRUE(guest);
    corrupted[mutant.text_sections().front()->sectionOffset() + *guest + 5] ^= 1;
    EXPECT_FALSE(test::lds_output_matches(context, ordinal == 0 ? 4 : 12, 15, corrupted));
  }
  EXPECT_TRUE(std::ranges::equal(context.image(), image));
  const FaultMutationContext unallocated(
      test::make_cdna5_lds_elf({set_bank[0], store[0], store[1]}, 15));
  const FailureOr<LdsInventory> unallocated_inventory =
      discover_lds_wrong_address(unallocated, "test_kernel", error);
  ASSERT_TRUE(unallocated_inventory.succeeded()) << error.message;
  EXPECT_TRUE(unallocated_inventory.value().sites().empty());
}

TEST(FaultLdsWrongAddress, Gfx1250JoinMustProveAddressSourceBank) {
  constexpr std::array<uint32_t, 1> branch_else =
      cdna5::build_sopp(cdna5::kSCbranchScc0Sopp, {.simm16 = 2});
  constexpr std::array<uint32_t, 1> set_one =
      cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 1});
  constexpr std::array<uint32_t, 1> branch_join =
      cdna5::build_sopp(cdna5::kSBranchSopp, {.simm16 = 1});
  constexpr std::array<uint32_t, 1> set_zero = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp);
  constexpr std::array<uint32_t, 2> access =
      cdna5::build_vds(cdna5::kDsStoreB32Vds, {.addr = 2, .data0 = 3});
  for (bool agree : {false, true}) {
    const FaultMutationContext context(
        test::make_cdna5_lds_elf({branch_else[0], set_one[0], branch_join[0],
                                  agree ? set_one[0] : set_zero[0], access[0], access[1]},
                                 16));
    FaultMutationError error;
    const FailureOr<LdsInventory> inventory =
        discover_lds_wrong_address(context, "test_kernel", error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    EXPECT_EQ(inventory.value().sites().size(), agree ? 1u : 0u);
    if (agree) {
      EXPECT_EQ(inventory.value().sites()[0].address_bank, 1u);
    }
  }
}

TEST(FaultLdsWrongAddress, Gfx1250RejectsTwoAddressAndKeepsBankZeroLoad) {
  constexpr std::array<uint32_t, 2> dual =
      cdna5::build_vds(cdna5::kDsLoad2addrStride64B32Vds, {.addr = 2});
  constexpr std::array<uint32_t, 2> load =
      cdna5::build_vds(cdna5::kDsLoadB32Vds, {.addr = 2, .vdst = 5});
  const FaultMutationContext context(
      test::make_cdna5_lds_elf({dual[0], dual[1], load[0], load[1]}));
  FaultMutationError error;
  const FailureOr<LdsInventory> inventory =
      discover_lds_wrong_address(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().sites().size(), 1u);
  EXPECT_EQ(inventory.value().sites()[0].offset, 8u);
  EXPECT_EQ(inventory.value().sites()[0].address_bank, 0u);
  const FailureOr<LdsSelection> selected = select_lds_wrong_address(inventory.value(), {}, error);
  ASSERT_TRUE(selected.succeeded()) << error.message;
  const FailureOr<LdsResult> result = apply_lds_wrong_address(context, selected.value(), 6, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  EXPECT_TRUE(test::lds_output_matches(context, 8, 6, result.value().image));
}

TEST(FaultLdsWrongAddress, Gfx1250DynamicBankAndAdjacentFixup) {
  constexpr std::array<uint32_t, 1> dynamic = cdna5::build_sopk(
      cdna5::kSSetregB32Sopk, {.simm16 = 1u | (14u << 6) | (1u << 11), .sdst = 0});
  constexpr std::array<uint32_t, 1> immediate =
      cdna5::build_sopk(cdna5::kSSetregImm32B32Sopk, {.simm16 = 1});
  constexpr std::array<uint32_t, 1> dropped =
      cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 3});
  constexpr std::array<uint32_t, 2> access =
      cdna5::build_vds(cdna5::kDsLoadB32Vds, {.addr = 2, .vdst = 5});
  const FaultMutationContext unknown(
      test::make_cdna5_lds_elf({dynamic[0], access[0], access[1]}, 63));
  FaultMutationError error;
  const FailureOr<LdsInventory> unknown_sites =
      discover_lds_wrong_address(unknown, "test_kernel", error);
  ASSERT_TRUE(unknown_sites.succeeded()) << error.message;
  EXPECT_TRUE(unknown_sites.value().sites().empty());
  // gfx1250 uses literal bits 19:12 to clobber banks; the adjacent set is dropped.
  const FaultMutationContext hazard(
      test::make_cdna5_lds_elf({immediate[0], 0x00004000u, dropped[0], access[0], access[1]}, 16));
  const FailureOr<LdsInventory> hazard_sites =
      discover_lds_wrong_address(hazard, "test_kernel", error);
  ASSERT_TRUE(hazard_sites.succeeded()) << error.message;
  ASSERT_EQ(hazard_sites.value().sites().size(), 1u);
  EXPECT_EQ(hazard_sites.value().sites()[0].address_bank, 1u);
}

TEST(FaultLdsWrongAddress, Gfx1250StaleSelectionAndFailedEmissionLeaveSourceUnchanged) {
  constexpr std::array<uint32_t, 2> access =
      cdna5::build_vds(cdna5::kDsStoreB32Vds, {.addr = 2, .data0 = 3});
  std::optional<LdsInventory> inventory;
  std::optional<LdsSelection> selection;
  {
    const FaultMutationContext context(test::make_cdna5_lds_elf({access[0], access[1]}));
    FaultMutationError error;
    const FailureOr<LdsInventory> found = discover_lds_wrong_address(context, "test_kernel", error);
    ASSERT_TRUE(found.succeeded()) << error.message;
    inventory = found.value();
    const FailureOr<LdsSelection> selected = select_lds_wrong_address(*inventory, {}, error);
    ASSERT_TRUE(selected.succeeded()) << error.message;
    selection = selected.value();
  }
  FaultMutationError error;
  const FaultMutationContext fresh(test::make_cdna5_lds_elf({access[0], access[1]}));
  EXPECT_TRUE(select_lds_wrong_address(*inventory, {}, error).failed());
  EXPECT_TRUE(apply_lds_wrong_address(fresh, *selection, 6, error).failed());
  std::vector<uint32_t> words = {access[0], access[1]};
  words.insert(words.end(), 32769, 0xbf800000u);
  const std::vector<uint8_t> image = test::make_cdna5_lds_elf(words);
  const FaultMutationContext distant(image);
  const FailureOr<LdsInventory> found = discover_lds_wrong_address(distant, "test_kernel", error);
  ASSERT_TRUE(found.succeeded()) << error.message;
  const FailureOr<LdsSelection> selected = select_lds_wrong_address(found.value(), {}, error);
  ASSERT_TRUE(selected.succeeded()) << error.message;
  EXPECT_TRUE(apply_lds_wrong_address(distant, selected.value(), 6, error).failed());
  EXPECT_TRUE(std::ranges::equal(distant.image(), image));
}

TEST(FaultLdsWrongAddress, Gfx1200Wave64StoreSelectionAndRegisterControls) {
  const std::vector<uint8_t> image = test::make_lds_elf(1, false, EF_AMDGPU_MACH_AMDGCN_GFX1200);
  const FaultMutationContext context(image);
  const FaultMutationContext foreign(image);
  ASSERT_EQ(context.source().target_id(), ROCJITSU_CODE_TARGET_GFX1200);
  FaultMutationError error;
  const FailureOr<LdsInventory> inventory =
      discover_lds_wrong_address(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().sites().size(), 2u);
  const FailureOr<LdsSelection> first = select_lds_wrong_address(inventory.value(), {}, error);
  const FailureOr<LdsSelection> second =
      select_lds_wrong_address(inventory.value(), {.exact = std::nullopt, .ordinal = 1}, error);
  ASSERT_TRUE(first.succeeded()) << error.message;
  ASSERT_TRUE(second.succeeded()) << error.message;
  const FailureOr<LdsSelection> exact = select_lds_wrong_address(
      inventory.value(), {.exact = second.value(), .ordinal = std::numeric_limits<size_t>::max()},
      error);
  ASSERT_TRUE(exact.succeeded()) << error.message;
  EXPECT_EQ(exact.value().site().offset, 8u);
  EXPECT_TRUE(apply_lds_wrong_address(context, first.value(), std::nullopt, error).failed());
  EXPECT_TRUE(apply_lds_wrong_address(context, first.value(), 2, error).failed());
  EXPECT_TRUE(apply_lds_wrong_address(context, first.value(), 256, error).failed());
  EXPECT_TRUE(apply_lds_wrong_address(context, first.value(), 8, error).failed());
  EXPECT_TRUE(apply_lds_wrong_address(foreign, first.value(), 6, error).failed());
  EXPECT_TRUE(
      select_lds_wrong_address(inventory.value(), {.exact = LdsSelection{}, .ordinal = 0}, error)
          .failed());
  const FailureOr<LdsResult> original = apply_lds_wrong_address(context, first.value(), 6, error);
  ASSERT_TRUE(original.succeeded()) << error.message;
  EXPECT_TRUE(test::lds_output_matches(context, 0, 6, original.value().image));
  std::vector<uint8_t> corrupted = original.value().image;
  const AmdGpuCodeObject mutant(corrupted.data(), corrupted.size());
  ASSERT_TRUE(mutant.is_valid());
  const std::optional<uint64_t> guest =
      test::lds_emitted_guest(*mutant.text_sections().front(), 0, ROCJITSU_CODE_TARGET_GFX1200);
  ASSERT_TRUE(guest);
  const uint64_t corrupted_offset = mutant.text_sections().front()->sectionOffset() + *guest + 5;
  ASSERT_LT(corrupted_offset, corrupted.size());
  corrupted[corrupted_offset] ^= 1;
  EXPECT_FALSE(test::lds_output_matches(context, 0, 6, corrupted));
  const FailureOr<LdsResult> other = apply_lds_wrong_address(context, exact.value(), 7, error);
  ASSERT_TRUE(other.succeeded()) << error.message;
  EXPECT_TRUE(test::lds_output_matches(context, 8, 7, other.value().image));
  const FailureOr<LdsResult> repeated = apply_lds_wrong_address(context, first.value(), 6, error);
  ASSERT_TRUE(repeated.succeeded()) << error.message;
  EXPECT_EQ(original.value().image, repeated.value().image);
  EXPECT_TRUE(std::ranges::equal(context.image(), image));
}

TEST(FaultLdsWrongAddress, Gfx1200RejectsUnsupportedShapesAndRollsBack) {
  FaultMutationError error;
  const FaultMutationContext wave32(test::make_lds_elf(1, true, EF_AMDGPU_MACH_AMDGCN_GFX1200));
  const FailureOr<LdsInventory> wave32_inventory =
      discover_lds_wrong_address(wave32, "test_kernel", error);
  ASSERT_TRUE(wave32_inventory.succeeded()) << error.message;
  EXPECT_EQ(wave32_inventory.value().sites().size(), 2u);
  const FaultMutationContext dual(
      test::make_lds_elf(1, false, EF_AMDGPU_MACH_AMDGCN_GFX1200, true));
  const FailureOr<LdsInventory> dual_inventory =
      discover_lds_wrong_address(dual, "test_kernel", error);
  ASSERT_TRUE(dual_inventory.succeeded()) << error.message;
  ASSERT_EQ(dual_inventory.value().sites().size(), 1u);
  EXPECT_EQ(dual_inventory.value().sites().front().offset, 8u);

  std::optional<LdsInventory> expired_inventory;
  std::optional<LdsSelection> expired_selection;
  {
    const FaultMutationContext temporary(
        test::make_lds_elf(1, false, EF_AMDGPU_MACH_AMDGCN_GFX1200));
    const FailureOr<LdsInventory> found =
        discover_lds_wrong_address(temporary, "test_kernel", error);
    ASSERT_TRUE(found.succeeded()) << error.message;
    expired_inventory = found.value();
    const FailureOr<LdsSelection> selected =
        select_lds_wrong_address(*expired_inventory, {}, error);
    ASSERT_TRUE(selected.succeeded()) << error.message;
    expired_selection = selected.value();
  }
  EXPECT_TRUE(select_lds_wrong_address(*expired_inventory, {}, error).failed());
  const FaultMutationContext fresh(test::make_lds_elf(1, false, EF_AMDGPU_MACH_AMDGCN_GFX1200));
  EXPECT_TRUE(apply_lds_wrong_address(fresh, *expired_selection, 6, error).failed());

  const std::array<uint32_t, 2> access =
      rdna4::build_vds(rdna4::kDsStoreB32Vds, {.offset0 = 4, .addr = 2, .data0 = 3});
  std::vector<uint8_t> truncated =
      test::make_amdgpu_kernel_elf({access[0]}, 0, 3, EF_AMDGPU_MACH_AMDGCN_GFX1200, 1);
  truncated[EI_DATA] = ELFDATA2LSB;
  truncated[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  const FaultMutationContext incomplete(truncated);
  EXPECT_TRUE(discover_lds_wrong_address(incomplete, "test_kernel", error).failed());
  EXPECT_TRUE(std::ranges::equal(incomplete.image(), truncated));

  std::vector<uint32_t> words = {access[0], access[1]};
  words.insert(words.end(), 32769, 0xbf800000u);
  words.push_back(0xbfb00000u);
  std::vector<uint8_t> image =
      test::make_amdgpu_kernel_elf(words, 0, 3, EF_AMDGPU_MACH_AMDGCN_GFX1200, 1);
  image[EI_DATA] = ELFDATA2LSB;
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  const FaultMutationContext context(image);
  const FailureOr<LdsInventory> inventory =
      discover_lds_wrong_address(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const FailureOr<LdsSelection> selection = select_lds_wrong_address(inventory.value(), {}, error);
  ASSERT_TRUE(selection.succeeded()) << error.message;
  EXPECT_TRUE(apply_lds_wrong_address(context, selection.value(), 6, error).failed());
  EXPECT_TRUE(std::ranges::equal(context.image(), image));
}

TEST(FaultLdsWrongAddress, DropRejectedSourceStillMutatesInEitherOrder) {
  const std::array<uint32_t, 2> access =
      rdna4::build_vds(rdna4::kDsStoreB32Vds, {.offset0 = 4, .addr = 2, .data0 = 3});
  // An unmatched builtin signal fails Drop qualification but does not disqualify LDS.
  const std::vector<uint32_t> words = {0xbe804ec1u, access[0], access[1], 0xbfb00000u};
  std::vector<uint8_t> image =
      test::make_amdgpu_kernel_elf(words, 0, 3, EF_AMDGPU_MACH_AMDGCN_GFX1201, 1);
  image[EI_DATA] = ELFDATA2LSB;
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;

  for (bool drop_first : {false, true}) {
    SCOPED_TRACE(drop_first);
    const FaultMutationContext context(image);
    FaultMutationError error;
    if (drop_first) {
      EXPECT_TRUE(discover_drop_barriers(context, "test_kernel", error).failed());
      EXPECT_EQ(error.code, FaultMutationErrorCode::IncompleteSelection);
    }
    const FailureOr<LdsInventory> inventory =
        discover_lds_wrong_address(context, "test_kernel", error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    ASSERT_EQ(inventory.value().sites().size(), 1u);
    EXPECT_EQ(inventory.value().sites().front().offset, 4u);
    EXPECT_EQ(inventory.value().sites().front().address_vgpr, 2u);
    const FailureOr<LdsSelection> selection =
        select_lds_wrong_address(inventory.value(), {}, error);
    ASSERT_TRUE(selection.succeeded()) << error.message;
    const FailureOr<LdsResult> mutant =
        apply_lds_wrong_address(context, selection.value(), 6, error);
    ASSERT_TRUE(mutant.succeeded()) << error.message;
    EXPECT_TRUE(test::lds_output_matches(context, 4, 6, mutant.value().image));

    EXPECT_TRUE(discover_drop_barriers(context, "test_kernel", error).failed());
    EXPECT_EQ(error.code, FaultMutationErrorCode::IncompleteSelection);
    const FailureOr<LdsResult> repeated =
        apply_lds_wrong_address(context, selection.value(), 6, error);
    ASSERT_TRUE(repeated.succeeded()) << error.message;
    EXPECT_EQ(repeated.value().image, mutant.value().image);
    EXPECT_TRUE(std::ranges::equal(context.image(), image));
  }
}

TEST(FaultLdsWrongAddress, ExactAndOrdinalProduceIndependentMutants) {
  const std::vector<uint8_t> image = test::make_lds_elf();
  const FaultMutationContext context(image);
  FaultMutationError error;
  const FailureOr<LdsInventory> inventory =
      discover_lds_wrong_address(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().sites().size(), 2u);
  EXPECT_EQ(inventory.value().sites()[0].offset, 0u);
  EXPECT_EQ(inventory.value().sites()[0].address_vgpr, 2u);
  EXPECT_EQ(inventory.value().sites()[1].offset, 8u);
  const FailureOr<LdsSelection> first = select_lds_wrong_address(inventory.value(), {}, error);
  const FailureOr<LdsSelection> second =
      select_lds_wrong_address(inventory.value(), {.exact = std::nullopt, .ordinal = 1}, error);
  ASSERT_TRUE(first.succeeded()) << error.message;
  ASSERT_TRUE(second.succeeded()) << error.message;
  const FailureOr<LdsSelection> exact = select_lds_wrong_address(
      inventory.value(), {.exact = second.value(), .ordinal = std::numeric_limits<size_t>::max()},
      error);
  ASSERT_TRUE(exact.succeeded()) << error.message;
  EXPECT_EQ(exact.value().site().offset, 8u);
  const FailureOr<LdsResult> first_result =
      apply_lds_wrong_address(context, first.value(), 6, error);
  ASSERT_TRUE(first_result.succeeded()) << error.message;
  EXPECT_EQ(first_result.value().logical_mutation_count, 1u);
  EXPECT_TRUE(test::lds_output_matches(context, 0, 6, first_result.value().image));
  const FailureOr<LdsResult> second_result =
      apply_lds_wrong_address(context, exact.value(), 6, error);
  ASSERT_TRUE(second_result.succeeded()) << error.message;
  EXPECT_TRUE(test::lds_output_matches(context, 8, 6, second_result.value().image));
  const FailureOr<LdsResult> repeated = apply_lds_wrong_address(context, first.value(), 6, error);
  ASSERT_TRUE(repeated.succeeded()) << error.message;
  EXPECT_EQ(first_result.value().image, repeated.value().image);
  EXPECT_TRUE(std::ranges::equal(context.image(), image));

  std::vector<uint8_t> corrupted = first_result.value().image;
  const AmdGpuCodeObject mutant(corrupted.data(), corrupted.size());
  ASSERT_TRUE(mutant.is_valid());
  const std::optional<uint64_t> guest = test::lds_emitted_guest(*mutant.text_sections().front(), 0);
  ASSERT_TRUE(guest);
  const uint64_t file_offset = mutant.text_sections().front()->sectionOffset() + *guest + 5;
  ASSERT_LT(file_offset, corrupted.size());
  corrupted[file_offset] ^= 1;
  EXPECT_FALSE(test::lds_output_matches(context, 0, 6, corrupted));
}

TEST(FaultLdsWrongAddress, RejectsInvalidRegistersAndForeignSelection) {
  const FaultMutationContext context(test::make_lds_elf());
  const FaultMutationContext foreign(test::make_lds_elf());
  FaultMutationError error;
  const FailureOr<LdsInventory> inventory =
      discover_lds_wrong_address(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const FailureOr<LdsSelection> selection = select_lds_wrong_address(inventory.value(), {}, error);
  ASSERT_TRUE(selection.succeeded()) << error.message;
  EXPECT_TRUE(apply_lds_wrong_address(context, selection.value(), std::nullopt, error).failed());
  EXPECT_TRUE(apply_lds_wrong_address(context, selection.value(), 2, error).failed());
  EXPECT_TRUE(apply_lds_wrong_address(context, selection.value(), 256, error).failed());
  // Wave64 granulated count 1 allocates v0..v7.
  EXPECT_TRUE(apply_lds_wrong_address(context, selection.value(), 8, error).failed());
  EXPECT_TRUE(apply_lds_wrong_address(foreign, selection.value(), 6, error).failed());
  EXPECT_TRUE(
      select_lds_wrong_address(inventory.value(), {.exact = std::nullopt, .ordinal = 2}, error)
          .failed());
  EXPECT_TRUE(
      select_lds_wrong_address(inventory.value(), {.exact = LdsSelection{}, .ordinal = 0}, error)
          .failed());
  const FailureOr<LdsResult> boundary =
      apply_lds_wrong_address(context, selection.value(), 7, error);
  ASSERT_TRUE(boundary.succeeded()) << error.message;
  EXPECT_TRUE(test::lds_output_matches(context, 0, 7, boundary.value().image));
}

TEST(FaultLdsWrongAddress, TargetsWavesAndTwoAddressFormsStayBounded) {
  FaultMutationError error;
  const FaultMutationContext wave32(test::make_lds_elf(1, true));
  const FailureOr<LdsInventory> wave32_inventory =
      discover_lds_wrong_address(wave32, "test_kernel", error);
  ASSERT_TRUE(wave32_inventory.succeeded()) << error.message;
  EXPECT_EQ(wave32_inventory.value().sites().size(), 2u);
  const FaultMutationContext other(test::make_lds_elf(1, false, EF_AMDGPU_MACH_AMDGCN_GFX1100));
  EXPECT_TRUE(discover_lds_wrong_address(other, "test_kernel", error).failed());
  const FaultMutationContext dual(
      test::make_lds_elf(1, false, EF_AMDGPU_MACH_AMDGCN_GFX1201, true));
  const FailureOr<LdsInventory> inventory = discover_lds_wrong_address(dual, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().sites().size(), 1u);
  EXPECT_EQ(inventory.value().sites().front().offset, 8u);
}

TEST(FaultLdsWrongAddress, ExpiredSourceCannotBeSelectedOrApplied) {
  std::optional<LdsInventory> inventory;
  std::optional<LdsSelection> selection;
  {
    const FaultMutationContext context(test::make_lds_elf());
    FaultMutationError error;
    const FailureOr<LdsInventory> found = discover_lds_wrong_address(context, "test_kernel", error);
    ASSERT_TRUE(found.succeeded()) << error.message;
    inventory = found.value();
    const FailureOr<LdsSelection> selected = select_lds_wrong_address(*inventory, {}, error);
    ASSERT_TRUE(selected.succeeded()) << error.message;
    selection = selected.value();
  }
  FaultMutationError error;
  EXPECT_TRUE(select_lds_wrong_address(*inventory, {}, error).failed());
  const FaultMutationContext fresh(test::make_lds_elf());
  EXPECT_TRUE(apply_lds_wrong_address(fresh, *selection, 6, error).failed());
}

TEST(FaultLdsWrongAddress, RejectsTruncatedInstructionExtent) {
  const std::array<uint32_t, 2> access =
      rdna4::build_vds(rdna4::kDsStoreB32Vds, {.offset0 = 4, .addr = 2, .data0 = 3});
  std::vector<uint8_t> image =
      test::make_amdgpu_kernel_elf({access[0]}, 0, 3, EF_AMDGPU_MACH_AMDGCN_GFX1201, 1);
  image[EI_DATA] = ELFDATA2LSB;
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  const FaultMutationContext context(image);
  FaultMutationError error;
  EXPECT_TRUE(discover_lds_wrong_address(context, "test_kernel", error).failed());
  EXPECT_TRUE(std::ranges::equal(context.image(), image));
}

TEST(FaultLdsWrongAddress, DoesNotAdvertiseClauseMember) {
  const std::array<uint32_t, 2> access =
      rdna4::build_vds(rdna4::kDsStoreB32Vds, {.offset0 = 4, .addr = 2, .data0 = 3});
  // Structural clause control; this does not claim a GPU execution result.
  const std::vector<uint32_t> words = {0xbf850000u, access[0], access[1],
                                       access[0],   access[1], 0xbfb00000u};
  std::vector<uint8_t> image =
      test::make_amdgpu_kernel_elf(words, 0, 3, EF_AMDGPU_MACH_AMDGCN_GFX1201, 1);
  image[EI_DATA] = ELFDATA2LSB;
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  const FaultMutationContext context(image);
  FaultMutationError error;
  const FailureOr<LdsInventory> inventory =
      discover_lds_wrong_address(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().sites().size(), 1u);
  EXPECT_EQ(inventory.value().sites().front().offset, 12u);
}

TEST(FaultLdsWrongAddress, EmissionFailureDoesNotChangeSource) {
  const std::array<uint32_t, 2> access =
      rdna4::build_vds(rdna4::kDsStoreB32Vds, {.offset0 = 4, .addr = 2, .data0 = 3});
  // The replacement cave is beyond a signed 16-bit dword branch from the site.
  std::vector<uint32_t> words = {access[0], access[1]};
  words.insert(words.end(), 32769, 0xbf800000u);
  words.push_back(0xbfb00000u);
  std::vector<uint8_t> image =
      test::make_amdgpu_kernel_elf(words, 0, 3, EF_AMDGPU_MACH_AMDGCN_GFX1201, 1);
  image[EI_DATA] = ELFDATA2LSB;
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  const FaultMutationContext context(image);
  FaultMutationError error;
  const FailureOr<LdsInventory> inventory =
      discover_lds_wrong_address(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const FailureOr<LdsSelection> selection = select_lds_wrong_address(inventory.value(), {}, error);
  ASSERT_TRUE(selection.succeeded()) << error.message;
  EXPECT_TRUE(apply_lds_wrong_address(context, selection.value(), 6, error).failed());
  EXPECT_TRUE(std::ranges::equal(context.image(), image));
}

} // namespace
} // namespace rocjitsu
