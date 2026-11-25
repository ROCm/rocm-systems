// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "../../gfx1251_lds_fixture.h"
#include "rocjitsu/code/patch/fault_injection/lds_wrong_address.h"
#include "target_coverage_fixture.h"

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

constexpr uint32_t kSetBankOne = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 1})[0];
constexpr uint32_t kSetBankTwo = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 2})[0];
constexpr uint32_t kSetBankThree = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 3})[0];

class Gfx1251Lds : public testing::TestWithParam<test::TargetLdsCase> {};

TEST_P(Gfx1251Lds, ExactAndOrdinalMutationsPreserveEveryFormAndProvenBank) {
  const test::TargetLdsCase &row = GetParam();
  const std::vector<LdsCapability> capabilities = lds_wrong_address_capabilities();
  EXPECT_EQ(std::ranges::count_if(capabilities,
                                  [&](const LdsCapability &capability) {
                                    return capability.target == row.form.target &&
                                           capability.wave_size == row.wave &&
                                           capability.form == row.form.form;
                                  }),
            1);
  EXPECT_EQ(std::ranges::count_if(capabilities,
                                  [&](const LdsCapability &capability) {
                                    return capability.target == row.form.target &&
                                           capability.wave_size == 64;
                                  }),
            0);
  for (uint32_t bank : {0u, 1u}) {
    SCOPED_TRACE(bank);
    const std::vector<uint32_t> prefix =
        bank == 0 ? std::vector<uint32_t>{} : std::vector<uint32_t>{kSetBankOne};
    const std::array<uint32_t, 3> suffix = {0xbf800007u, row.form.words[0], row.form.words[1]};
    const std::vector<uint8_t> image =
        test::make_target_lds_elf(row, prefix, suffix, bank == 0 ? 3 : 16);
    const FaultMutationContext source(image);
    FaultMutationError error;
    const FailureOr<LdsInventory> inventory =
        discover_lds_wrong_address(source, "test_kernel", error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    ASSERT_EQ(inventory.value().sites().size(), 2u);
    for (size_t ordinal = 0; ordinal < 2; ++ordinal) {
      const uint64_t offset = prefix.size() * sizeof(uint32_t) + ordinal * 12;
      const LdsSite &site = inventory.value().sites()[ordinal];
      EXPECT_EQ(site.offset, offset);
      EXPECT_EQ(site.form, row.form.form);
      EXPECT_EQ(site.address_vgpr, 2u);
      EXPECT_EQ(site.address_bank, bank);
      ASSERT_EQ(site.affected_owners.size(), 1u);
      EXPECT_EQ(site.affected_owners.front().entry, "test_kernel");
      const FailureOr<LdsSelection> selected = select_lds_wrong_address(
          inventory.value(), {.exact = std::nullopt, .ordinal = ordinal}, error);
      ASSERT_TRUE(selected.succeeded()) << error.message;
      const FailureOr<LdsSelection> exact = select_lds_wrong_address(
          inventory.value(),
          {.exact = selected.value(), .ordinal = std::numeric_limits<size_t>::max()}, error);
      ASSERT_TRUE(exact.succeeded()) << error.message;
      const FailureOr<LdsResult> mutant = apply_lds_wrong_address(source, exact.value(), 6, error);
      ASSERT_TRUE(mutant.succeeded()) << error.message;
      EXPECT_EQ(mutant.value().logical_mutation_count, 1u);
      EXPECT_TRUE(test::target_lds_output_matches(source, row, offset, 6, mutant.value().image));
      const FailureOr<LdsResult> repeated =
          apply_lds_wrong_address(source, selected.value(), 6, error);
      ASSERT_TRUE(repeated.succeeded()) << error.message;
      EXPECT_EQ(repeated.value().image, mutant.value().image);
    }
    EXPECT_TRUE(std::ranges::equal(source.image(), image));
  }
}

TEST_P(Gfx1251Lds, RegisterAndIdentityControlsRespectEachBanksOrdinaryAllocation) {
  const test::TargetLdsCase &row = GetParam();
  for (uint32_t bank : {0u, 1u}) {
    SCOPED_TRACE(bank);
    const std::vector<uint32_t> prefix =
        bank == 0 ? std::vector<uint32_t>{} : std::vector<uint32_t>{kSetBankOne};
    const uint32_t granulated = bank == 0 ? 3 : 16;
    const uint32_t bound = test::target_lds_allocation(row, granulated) - bank * 256;
    const std::vector<uint8_t> image = test::make_target_lds_elf(row, prefix, {}, granulated);
    const FaultMutationContext source(image), foreign(image);
    FaultMutationError error;
    const FailureOr<LdsInventory> inventory =
        discover_lds_wrong_address(source, "test_kernel", error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    ASSERT_EQ(inventory.value().sites().size(), 1u);
    const FailureOr<LdsSelection> selected = select_lds_wrong_address(inventory.value(), {}, error);
    ASSERT_TRUE(selected.succeeded()) << error.message;
    for (std::optional<uint32_t> reg :
         {std::optional<uint32_t>{}, std::optional<uint32_t>{2}, std::optional<uint32_t>{bound},
          std::optional<uint32_t>{256}, std::optional<uint32_t>{UINT32_MAX}}) {
      EXPECT_TRUE(apply_lds_wrong_address(source, selected.value(), reg, error).failed());
      EXPECT_EQ(error.code, FaultMutationErrorCode::InvalidParameters);
    }
    const FailureOr<LdsResult> last =
        apply_lds_wrong_address(source, selected.value(), bound - 1, error);
    ASSERT_TRUE(last.succeeded()) << error.message;
    EXPECT_TRUE(test::target_lds_output_matches(source, row, prefix.size() * 4, bound - 1,
                                                last.value().image));
    EXPECT_TRUE(apply_lds_wrong_address(foreign, selected.value(), 6, error).failed());
    EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
    EXPECT_TRUE(
        select_lds_wrong_address(inventory.value(), {.exact = std::nullopt, .ordinal = 1}, error)
            .failed());
    EXPECT_EQ(error.code, FaultMutationErrorCode::OrdinalOutOfRange);
    EXPECT_TRUE(
        select_lds_wrong_address(inventory.value(), {.exact = LdsSelection{}, .ordinal = 0}, error)
            .failed());
    EXPECT_TRUE(std::ranges::equal(source.image(), image));
  }
  const std::array<uint32_t, 1> bank_one = {kSetBankOne};
  const FaultMutationContext unallocated(test::make_target_lds_elf(row, bank_one, {}, 15));
  FaultMutationError error;
  const FailureOr<LdsInventory> none =
      discover_lds_wrong_address(unallocated, "test_kernel", error);
  ASSERT_TRUE(none.succeeded()) << error.message;
  EXPECT_TRUE(none.value().sites().empty());

  const std::array<uint32_t, 4> disagreeing = {
      cdna5::build_sopp(cdna5::kSCbranchScc0Sopp, {.simm16 = 2})[0], kSetBankOne,
      cdna5::build_sopp(cdna5::kSBranchSopp, {.simm16 = 1})[0], kSetBankTwo};
  const FaultMutationContext ambiguous(test::make_target_lds_elf(row, disagreeing, {}, 63));
  const FailureOr<LdsInventory> unknown =
      discover_lds_wrong_address(ambiguous, "test_kernel", error);
  ASSERT_TRUE(unknown.succeeded()) << error.message;
  EXPECT_TRUE(unknown.value().sites().empty());
}

TEST(Gfx1251Lds, ConcreteModeWritesAndAdjacencyDifferFromGfx1250) {
  const test::TargetLdsCase gfx1251 = test::target_lds_cases(ROCJITSU_CODE_TARGET_GFX1251).front();
  const uint32_t set_src0 =
      cdna5::build_sopk(cdna5::kSSetregImm32B32Sopk, {.simm16 = 1u | (14u << 6) | (1u << 11)})[0];
  const uint32_t dynamic_src0 = cdna5::build_sopk(
      cdna5::kSSetregB32Sopk, {.simm16 = 1u | (14u << 6) | (1u << 11), .sdst = 0})[0];
  const uint32_t dynamic_disjoint =
      cdna5::build_sopk(cdna5::kSSetregB32Sopk, {.simm16 = 1u | (25u << 6), .sdst = 0})[0];
  const uint32_t immediate_disjoint =
      cdna5::build_sopk(cdna5::kSSetregImm32B32Sopk, {.simm16 = 1u | (25u << 6)})[0];
  const uint32_t immediate_bit_zero =
      cdna5::build_sopk(cdna5::kSSetregImm32B32Sopk, {.simm16 = 1})[0];
  struct Case {
    const char *name;
    std::vector<uint32_t> prefix;
    std::optional<uint32_t> gfx1250_bank;
    std::optional<uint32_t> gfx1251_bank;
  };
  const std::array<Case, 5> cases = {{
      {.name = "right-justified Src0 slice",
       .prefix = {set_src0, 2},
       .gfx1250_bank = 0,
       .gfx1251_bank = 2},
      {.name = "dynamic disjoint MODE slice",
       .prefix = {kSetBankOne, dynamic_disjoint},
       .gfx1250_bank = std::nullopt,
       .gfx1251_bank = 1},
      {.name = "immediate disjoint MODE slice",
       .prefix = {kSetBankOne, immediate_disjoint, 0},
       .gfx1250_bank = 0,
       .gfx1251_bank = 1},
      {.name = "adjacent set executes only without fixup",
       .prefix = {immediate_bit_zero, 0x4000u, kSetBankThree},
       .gfx1250_bank = 1,
       .gfx1251_bank = 3},
      {.name = "dynamic address-bank slice",
       .prefix = {dynamic_src0},
       .gfx1250_bank = std::nullopt,
       .gfx1251_bank = std::nullopt},
  }};
  for (const Case &item : cases) {
    SCOPED_TRACE(item.name);
    for (bool newer : {false, true}) {
      SCOPED_TRACE(newer);
      test::TargetLdsCase row = gfx1251;
      if (!newer) {
        row.form.target = ROCJITSU_CODE_TARGET_GFX1250;
        row.form.machine = EF_AMDGPU_MACH_AMDGCN_GFX1250;
      }
      const std::optional<uint32_t> expected = newer ? item.gfx1251_bank : item.gfx1250_bank;
      const FaultMutationContext source(test::make_target_lds_elf(row, item.prefix, {}, 63));
      FaultMutationError error;
      const FailureOr<LdsInventory> inventory =
          discover_lds_wrong_address(source, "test_kernel", error);
      ASSERT_TRUE(inventory.succeeded()) << error.message;
      ASSERT_EQ(inventory.value().sites().size(), expected ? 1u : 0u);
      if (!expected)
        continue;
      EXPECT_EQ(inventory.value().sites().front().address_bank, *expected);
      const FailureOr<LdsSelection> selected =
          select_lds_wrong_address(inventory.value(), {}, error);
      ASSERT_TRUE(selected.succeeded()) << error.message;
      const FailureOr<LdsResult> mutant =
          apply_lds_wrong_address(source, selected.value(), 6, error);
      ASSERT_TRUE(mutant.succeeded()) << error.message;
      EXPECT_TRUE(test::target_lds_output_matches(source, row, item.prefix.size() * 4, 6,
                                                  mutant.value().image));
    }
  }
}

TEST(Gfx1251Lds, UnsupportedOperationsAndFeatureInvalidWordsAreNotAdmitted) {
  const test::TargetLdsCase row = test::target_lds_cases(ROCJITSU_CODE_TARGET_GFX1251).front();
  std::unique_ptr<Decoder> decoder =
      Decoder::create(default_isa_target_registry(), row.form.target);
  ASSERT_NE(decoder, nullptr);
  const std::array<std::array<uint32_t, 2>, 5> excluded = {{
      test::kGfx1251ImplicitAddressLdsFixtures[0].source,
      test::kGfx1251ImplicitAddressLdsFixtures[1].source,
      {0xd8500000u, 0},           // ds_nop
      {0xd8000000u, 0x00000302u}, // ds_add_u32
      {0xdac80000u, 0x0c000302u}, // ds_permute_b32
  }};
  for (const std::array<uint32_t, 2> &words : excluded) {
    const DecodeResult decoded = decoder->decode_window(words);
    ASSERT_TRUE(decoded.succeeded());
    SCOPED_TRACE(decoded.value()->mnemonic());
    const FaultMutationContext source(test::make_target_lds_elf(row, words));
    FaultMutationError error;
    const FailureOr<LdsInventory> inventory =
        discover_lds_wrong_address(source, "test_kernel", error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    ASSERT_EQ(inventory.value().sites().size(), 1u);
    EXPECT_EQ(inventory.value().sites().front().offset, 8u);
  }
  // Public LLVM gfx1251 packed U64 witness. The shared CDNA5 family must not
  // make this unselected instruction legal for gfx1250.
  const std::array<uint32_t, 2> packed = {0xcc4c4004u, 0x1a021908u};
  const std::vector<uint8_t> image = test::make_target_lds_elf(row, packed);
  const FaultMutationContext valid(image);
  FaultMutationError error;
  const FailureOr<LdsInventory> inventory = discover_lds_wrong_address(valid, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().sites().size(), 1u);
  const FailureOr<LdsSelection> selected = select_lds_wrong_address(inventory.value(), {}, error);
  ASSERT_TRUE(selected.succeeded()) << error.message;
  const FailureOr<LdsResult> mutant = apply_lds_wrong_address(valid, selected.value(), 6, error);
  ASSERT_TRUE(mutant.succeeded()) << error.message;
  EXPECT_TRUE(test::target_lds_output_matches(valid, row, 8, 6, mutant.value().image));
  test::TargetLdsCase wrong = row;
  wrong.form.target = ROCJITSU_CODE_TARGET_GFX1250;
  wrong.form.machine = EF_AMDGPU_MACH_AMDGCN_GFX1250;
  const FaultMutationContext invalid(test::make_target_lds_elf(wrong, packed));
  EXPECT_TRUE(discover_lds_wrong_address(invalid, "test_kernel", error).failed());
  EXPECT_NE(error.message.find("invalid source instruction"), std::string::npos);
}

TEST(Gfx1251Lds, ExpiredSelectionsTruncatedInstructionsAndEmissionFailureLeaveSourceIntact) {
  const test::TargetLdsCase row = test::target_lds_cases(ROCJITSU_CODE_TARGET_GFX1251).front();
  const std::vector<uint8_t> image = test::make_target_lds_elf(row);
  std::optional<LdsInventory> expired_inventory;
  std::optional<LdsSelection> expired_selection;
  FaultMutationError error;
  {
    const FaultMutationContext source(image);
    const FailureOr<LdsInventory> inventory =
        discover_lds_wrong_address(source, "test_kernel", error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    expired_inventory = inventory.value();
    const FailureOr<LdsSelection> selected = select_lds_wrong_address(inventory.value(), {}, error);
    ASSERT_TRUE(selected.succeeded()) << error.message;
    expired_selection = selected.value();
  }
  const FaultMutationContext fresh(image);
  EXPECT_TRUE(select_lds_wrong_address(*expired_inventory, {}, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  EXPECT_TRUE(apply_lds_wrong_address(fresh, *expired_selection, 6, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  EXPECT_TRUE(std::ranges::equal(fresh.image(), image));

  std::vector<uint8_t> truncated =
      test::make_amdgpu_kernel_elf({row.form.words[0]}, 0, 3, row.form.machine, 3);
  truncated[EI_DATA] = ELFDATA2LSB;
  truncated[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  const FaultMutationContext incomplete(truncated);
  for (int attempt = 0; attempt < 2; ++attempt) {
    EXPECT_TRUE(discover_lds_wrong_address(incomplete, "test_kernel", error).failed());
    EXPECT_TRUE(std::ranges::equal(incomplete.image(), truncated));
  }
  const std::vector<uint32_t> distant(32769, 0xbf800000u);
  const std::vector<uint8_t> long_image = test::make_target_lds_elf(row, {}, distant);
  const FaultMutationContext source(long_image);
  const FailureOr<LdsInventory> inventory =
      discover_lds_wrong_address(source, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const FailureOr<LdsSelection> selected = select_lds_wrong_address(inventory.value(), {}, error);
  ASSERT_TRUE(selected.succeeded()) << error.message;
  for (int attempt = 0; attempt < 2; ++attempt) {
    EXPECT_TRUE(apply_lds_wrong_address(source, selected.value(), 6, error).failed());
    EXPECT_TRUE(std::ranges::equal(source.image(), long_image));
  }
}

TEST(Gfx1251Lds, WholeObjectOracleRejectsGuestContinuationDescriptorAndUnselectedCorruption) {
  const test::TargetLdsCase row = test::target_lds_cases(ROCJITSU_CODE_TARGET_GFX1251).front();
  const FaultMutationContext source(test::make_target_lds_elf(row));
  FaultMutationError error;
  const FailureOr<LdsInventory> inventory =
      discover_lds_wrong_address(source, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const FailureOr<LdsSelection> selected = select_lds_wrong_address(inventory.value(), {}, error);
  ASSERT_TRUE(selected.succeeded()) << error.message;
  const FailureOr<LdsResult> result = apply_lds_wrong_address(source, selected.value(), 6, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  const AmdGpuCodeObject emitted(result.value().image.data(), result.value().image.size());
  const Section &text = *emitted.text_sections().front();
  const std::optional<uint64_t> guest = test::lds_emitted_guest(text, 0, row.form.target);
  ASSERT_TRUE(guest);
  ASSERT_TRUE(text.sectionHeaderIndex());
  const std::vector<KernelDescriptorInfo> kernels = scan_kernel_descriptors(
      result.value().image, text.sectionOffset(), text.size(), text.sectionHeaderIndex());
  ASSERT_EQ(kernels.size(), 1u);
  const uint64_t descriptor = kernels.front().descriptor_file_offset;
  Elf64_Ehdr header = {};
  std::memcpy(&header, result.value().image.data(), sizeof(header));
  const uint64_t mapping = header.e_shoff + *text.sectionHeaderIndex() * sizeof(Elf64_Shdr) +
                           offsetof(Elf64_Shdr, sh_addr);
  for (uint64_t byte : {text.sectionOffset() + *guest + 5, text.sectionOffset() + *guest + 8,
                        text.sectionOffset() + 8, descriptor, mapping,
                        static_cast<uint64_t>(result.value().image.size() - 1)}) {
    SCOPED_TRACE(byte);
    std::vector<uint8_t> corrupted = result.value().image;
    ASSERT_LT(byte, corrupted.size());
    corrupted[byte] ^= 1;
    EXPECT_FALSE(test::target_lds_output_matches(source, row, 0, 6, corrupted));
  }
}

INSTANTIATE_TEST_SUITE_P(All34Forms, Gfx1251Lds,
                         testing::ValuesIn(test::target_lds_cases(ROCJITSU_CODE_TARGET_GFX1251)),
                         [](const testing::TestParamInfo<test::TargetLdsCase> &info) {
                           return std::string(info.param.form.form);
                         });

} // namespace
} // namespace rocjitsu
