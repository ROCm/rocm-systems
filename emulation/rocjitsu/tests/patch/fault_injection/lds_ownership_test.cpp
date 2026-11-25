// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "lds_owner_fixture.h"
#include "rocjitsu/code/kernel_scope.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/fault_injection/drop_barrier.h"
#include "rocjitsu/code/patch/fault_injection/lds_wrong_address.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

namespace rocjitsu {
namespace {

TEST(FaultLdsWrongAddress, DirectSharedHelperUsesExistingReplacement) {
  const std::vector<uint8_t> image = test::make_shared_lds_elf(false);
  const FaultMutationContext source(image);
  FaultMutationError error;
  const FailureOr<LdsInventory> inventory =
      discover_lds_wrong_address(source, "test_kernel0", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().sites().size(), 1u);
  const FailureOr<LdsSelection> selection = select_lds_wrong_address(inventory.value(), {}, error);
  ASSERT_TRUE(selection.succeeded()) << error.message;
  const FailureOr<LdsResult> mutant = apply_lds_wrong_address(source, selection.value(), 1, error);
  ASSERT_TRUE(mutant.succeeded()) << error.message;
  EXPECT_TRUE(test::lds_output_matches(source, 512, 1, mutant.value().image));
  EXPECT_TRUE(std::ranges::equal(source.image(), image));
}

TEST(FaultLdsWrongAddress, EmptySelectionOwnerCanBeQueriedSafely) {
  EXPECT_EXIT(
      {
        const LdsSelection empty;
        std::_Exit(empty.owner() == nullptr ? 0 : 1);
      },
      ::testing::ExitedWithCode(0), "");
  EXPECT_EQ(LdsSelection{}.owner(), nullptr);
}

TEST(FaultLdsWrongAddress, WrongKernelRetainsDirectSourceLookupDiagnostics) {
  for (bool dynamic : {false, true}) {
    std::vector<uint8_t> image = test::make_lds_elf();
    if (dynamic) {
      Elf64_Ehdr header = {};
      std::memcpy(&header, image.data(), sizeof(header));
      Elf64_Phdr rodata = {};
      const size_t rodata_header = header.e_phoff + sizeof(Elf64_Phdr);
      std::memcpy(&rodata, image.data() + rodata_header, sizeof(rodata));
      const uint64_t place = rodata.p_vaddr + rodata.p_filesz;
      rodata.p_filesz += sizeof(uint32_t);
      rodata.p_memsz += sizeof(uint32_t);
      std::memcpy(image.data() + rodata_header, &rodata, sizeof(rodata));
      std::vector<Elf64_Shdr> sections(header.e_shnum);
      std::memcpy(sections.data(), image.data() + header.e_shoff,
                  sections.size() * sizeof(Elf64_Shdr));
      const Elf64_Rela relocation = {
          .r_offset = place, .r_info = (uint64_t{1} << 32) | R_AMDGPU_ABS32_LO, .r_addend = 0};
      const Elf64_Shdr section = {.sh_name = 0,
                                  .sh_type = SHT_RELA,
                                  .sh_flags = 0,
                                  .sh_addr = 0,
                                  .sh_offset = image.size(),
                                  .sh_size = sizeof(relocation),
                                  .sh_link = 3,
                                  .sh_info = 0,
                                  .sh_addralign = 0,
                                  .sh_entsize = sizeof(relocation)};
      image.resize(image.size() + sizeof(relocation));
      std::memcpy(image.data() + section.sh_offset, &relocation, sizeof(relocation));
      sections.push_back(section);
      header.e_shoff = image.size();
      header.e_shnum = static_cast<uint16_t>(sections.size());
      image.resize(image.size() + sections.size() * sizeof(Elf64_Shdr));
      std::memcpy(image.data() + header.e_shoff, sections.data(),
                  sections.size() * sizeof(Elf64_Shdr));
      std::memcpy(image.data(), &header, sizeof(header));
    }
    const FaultMutationContext source(image);
    FaultMutationError error;
    const FailureOr<LdsInventory> valid = discover_lds_wrong_address(source, "test_kernel", error);
    ASSERT_TRUE(valid.succeeded()) << error.message;
    EXPECT_TRUE(discover_lds_wrong_address(source, "missing", error).failed());
    EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
    EXPECT_NE(error.message.find("test_kernel"), std::string::npos);
  }
}

TEST(FaultLdsWrongAddress, DirectSharedHelperIsDiscoverableFromEachOwner) {
  constexpr IsaFamily arch = IsaFamily::Rdna4;
  const std::vector<uint32_t> words = {build_s_call_b64(30, 3, arch).value(),
                                       build_s_endpgm(arch).value(),
                                       build_s_call_b64(30, 1, arch).value(),
                                       build_s_endpgm(arch).value(),
                                       0xd8340004u,
                                       0x00000302u,
                                       build_s_setpc_b64(30, arch).value()};
  std::vector<uint8_t> image =
      test::make_amdgpu_two_kernel_elf(words, 0, 3, EF_AMDGPU_MACH_AMDGCN_GFX1201);
  image[EI_DATA] = ELFDATA2LSB;
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  const size_t descriptor_offset =
      0x100 + words.size() * sizeof(uint32_t) + sizeof(amdhsa::kernel_descriptor_t);
  amdhsa::kernel_descriptor_t sibling = {};
  std::memcpy(&sibling, image.data() + descriptor_offset, sizeof(sibling));
  sibling.kernel_code_entry_byte_offset += 8;
  std::memcpy(image.data() + descriptor_offset, &sibling, sizeof(sibling));
  const FaultMutationContext source(image);
  for (std::string_view owner : {"test_kernel0", "test_kernel1"}) {
    SCOPED_TRACE(owner);
    FaultMutationError error;
    const FailureOr<LdsInventory> inventory = discover_lds_wrong_address(source, owner, error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    ASSERT_EQ(inventory.value().sites().size(), 1u);
    EXPECT_EQ(inventory.value().sites()[0].offset, 16u);
    ASSERT_EQ(inventory.value().sites()[0].affected_owners.size(), 2u);
    EXPECT_EQ(inventory.value().sites()[0].affected_owners[0].entry, "test_kernel0");
    EXPECT_EQ(inventory.value().sites()[0].affected_owners[1].entry, "test_kernel1");
    const FailureOr<LdsSelection> selection =
        select_lds_wrong_address(inventory.value(), {}, error);
    ASSERT_TRUE(selection.succeeded()) << error.message;
    const FailureOr<LdsResult> mutant =
        apply_lds_wrong_address(source, selection.value(), 1, error);
    ASSERT_TRUE(mutant.succeeded()) << error.message;
    EXPECT_TRUE(test::lds_owner_output_matches(source, 16, 1, mutant.value().image));
  }
}

TEST(FaultLdsWrongAddress, RecoveredSharedInventoryPreservesGrowthRejection) {
  const std::vector<uint8_t> image = test::make_shared_lds_elf(true);
  const FaultMutationContext source(image);
  for (std::string_view owner : {"test_kernel0", "test_kernel1"}) {
    SCOPED_TRACE(owner);
    FaultMutationError error;
    const FailureOr<LdsInventory> inventory = discover_lds_wrong_address(source, owner, error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    ASSERT_EQ(inventory.value().sites().size(), 1u);
    ASSERT_EQ(inventory.value().sites()[0].affected_owners.size(), 2u);
    const FailureOr<LdsSelection> selection =
        select_lds_wrong_address(inventory.value(), {}, error);
    ASSERT_TRUE(selection.succeeded()) << error.message;
    const FailureOr<LdsResult> mutant =
        apply_lds_wrong_address(source, selection.value(), 1, error);
    EXPECT_TRUE(mutant.failed());
    EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm);
    EXPECT_NE(error.message.find("replacement growth cannot preserve retained PC-relative"),
              std::string::npos);
    EXPECT_TRUE(std::ranges::equal(source.image(), image));
  }
}

TEST(FaultLdsWrongAddress, CdnaSharedHelpersRetainEveryOwner) {
  for (uint32_t machine : {EF_AMDGPU_MACH_AMDGCN_GFX942, EF_AMDGPU_MACH_AMDGCN_GFX950}) {
    for (bool indirect : {false, true}) {
      SCOPED_TRACE(machine);
      SCOPED_TRACE(indirect);
      const std::vector<uint8_t> image = test::make_shared_lds_elf(indirect, 0, true, machine);
      const FaultMutationContext source(image);
      for (std::string_view owner : {"test_kernel0", "test_kernel1"}) {
        SCOPED_TRACE(owner);
        FaultMutationError error;
        const FailureOr<LdsInventory> inventory = discover_lds_wrong_address(source, owner, error);
        ASSERT_TRUE(inventory.succeeded()) << error.message;
        ASSERT_EQ(inventory.value().sites().size(), 1u);
        EXPECT_EQ(inventory.value().sites()[0].offset, 512u);
        ASSERT_EQ(inventory.value().sites()[0].affected_owners.size(), 2u);
        const FailureOr<LdsSelection> selection =
            select_lds_wrong_address(inventory.value(), {}, error);
        ASSERT_TRUE(selection.succeeded()) << error.message;
        const FailureOr<LdsResult> mutant =
            apply_lds_wrong_address(source, selection.value(), 1, error);
        if (indirect) {
          EXPECT_TRUE(mutant.failed());
          EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm);
          EXPECT_NE(error.message.find("retained PC-relative"), std::string::npos);
          continue;
        }
        ASSERT_TRUE(mutant.succeeded()) << error.message;
        EXPECT_TRUE(test::lds_owner_output_matches(source, 512, 1, mutant.value().image));
      }
      EXPECT_TRUE(std::ranges::equal(source.image(), image));
    }
  }
}

TEST(FaultLdsWrongAddress, Gfx1200SharedHelpersRetainEveryOwner) {
  for (bool indirect : {false, true}) {
    SCOPED_TRACE(indirect);
    const std::vector<uint8_t> image = test::make_shared_lds_elf(
        indirect, 0, false, EF_AMDGPU_MACH_AMDGCN_GFX950, EF_AMDGPU_MACH_AMDGCN_GFX1200);
    const FaultMutationContext source(image);
    for (std::string_view owner : {"test_kernel0", "test_kernel1"}) {
      SCOPED_TRACE(owner);
      FaultMutationError error;
      const FailureOr<LdsInventory> inventory = discover_lds_wrong_address(source, owner, error);
      ASSERT_TRUE(inventory.succeeded()) << error.message;
      ASSERT_EQ(inventory.value().sites().size(), 1u);
      EXPECT_EQ(inventory.value().sites()[0].offset, 512u);
      ASSERT_EQ(inventory.value().sites()[0].affected_owners.size(), 2u);
      const FailureOr<LdsSelection> selection =
          select_lds_wrong_address(inventory.value(), {}, error);
      ASSERT_TRUE(selection.succeeded()) << error.message;
      const FailureOr<LdsResult> mutant =
          apply_lds_wrong_address(source, selection.value(), 1, error);
      if (indirect) {
        EXPECT_TRUE(mutant.failed());
        EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm);
        EXPECT_NE(error.message.find("retained PC-relative"), std::string::npos);
        continue;
      }
      ASSERT_TRUE(mutant.succeeded()) << error.message;
      EXPECT_TRUE(test::lds_owner_output_matches(source, 512, 1, mutant.value().image));
    }
    EXPECT_TRUE(std::ranges::equal(source.image(), image));
  }
}

TEST(FaultLdsWrongAddress, SharedHelperChecksAllOwnersAndKeepsDropPolicy) {
  std::vector<uint8_t> image = test::make_shared_lds_elf(false);
  constexpr size_t first_descriptor = 0x100 + 131 * sizeof(uint32_t);
  amdhsa::kernel_descriptor_t first = {};
  std::memcpy(&first, image.data() + first_descriptor, sizeof(first));
  RJ_AMDHSA_BITS_SET(first.compute_pgm_rsrc1,
                     amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT, 1);
  std::memcpy(image.data() + first_descriptor, &first, sizeof(first));
  const FaultMutationContext source(image), foreign(image);
  FaultMutationError error;
  EXPECT_TRUE(discover_drop_barriers(source, "test_kernel0", error).failed());
  const FailureOr<LdsInventory> a = discover_lds_wrong_address(source, "test_kernel0", error);
  ASSERT_TRUE(a.succeeded()) << error.message;
  ASSERT_EQ(a.value().sites().size(), 1u);
  EXPECT_EQ(a.value().sites()[0].affected_owners.size(), 2u);
  EXPECT_TRUE(discover_lds_wrong_address(source, "unrelated_kernel", error).failed());
  const FailureOr<LdsSelection> selection = select_lds_wrong_address(a.value(), {}, error);
  ASSERT_TRUE(selection.succeeded());
  EXPECT_TRUE(apply_lds_wrong_address(source, selection.value(), std::nullopt, error).failed());
  EXPECT_TRUE(apply_lds_wrong_address(source, selection.value(), 2, error).failed());
  EXPECT_TRUE(apply_lds_wrong_address(source, selection.value(), 4, error).failed());
  EXPECT_TRUE(apply_lds_wrong_address(source, selection.value(), 256, error).failed());
  EXPECT_TRUE(apply_lds_wrong_address(foreign, selection.value(), 1, error).failed());
  const FailureOr<LdsResult> valid = apply_lds_wrong_address(source, selection.value(), 1, error);
  ASSERT_TRUE(valid.succeeded()) << error.message;
  EXPECT_TRUE(test::lds_owner_output_matches(source, 512, 1, valid.value().image));
  EXPECT_TRUE(discover_drop_barriers(source, "test_kernel0", error).failed());
  EXPECT_TRUE(std::ranges::equal(source.image(), image));
}

TEST(FaultLdsWrongAddress, SharedHelperRejectsInvalidFlagsAndMalformedSibling) {
  for (bool flags : {false, true}) {
    std::vector<uint8_t> image = test::make_shared_lds_elf(false);
    if (flags) {
      Elf64_Ehdr header = {};
      std::memcpy(&header, image.data(), sizeof(header));
      header.e_flags |= 0x80000000u;
      std::memcpy(image.data(), &header, sizeof(header));
    } else {
      constexpr size_t offset =
          0x100 + 131 * sizeof(uint32_t) + sizeof(amdhsa::kernel_descriptor_t);
      amdhsa::kernel_descriptor_t sibling = {};
      std::memcpy(&sibling, image.data() + offset, sizeof(sibling));
      sibling.kernel_code_entry_byte_offset = std::numeric_limits<int64_t>::max();
      std::memcpy(image.data() + offset, &sibling, sizeof(sibling));
    }
    const FaultMutationContext source(image);
    FaultMutationError error;
    EXPECT_TRUE(discover_lds_wrong_address(source, "test_kernel0", error).failed());
  }
}

TEST(FaultLdsWrongAddress, PrivateAndSharedSitesUseTheirOwnAffectedOwnerAllocations) {
  std::vector<uint8_t> image = test::make_shared_lds_elf(false);
  // A owns the private store at0 and calls the shared helper at512. B only
  // calls that helper. A has eight VGPRs; B has four.
  constexpr IsaFamily arch = IsaFamily::Rdna4;
  const std::array<uint32_t, 4> entry = {0xd8340004u, 0x00000302u,
                                         build_s_call_b64(30, 125, arch).value(),
                                         build_s_endpgm(arch).value()};
  std::memcpy(image.data() + 0x100, entry.data(), sizeof(entry));
  constexpr size_t first_descriptor = 0x100 + 131 * sizeof(uint32_t);
  amdhsa::kernel_descriptor_t descriptor = {};
  std::memcpy(&descriptor, image.data() + first_descriptor, sizeof(descriptor));
  RJ_AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                     amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT, 1);
  std::memcpy(image.data() + first_descriptor, &descriptor, sizeof(descriptor));
  const FaultMutationContext source(image);
  FaultMutationError error;
  const FailureOr<LdsInventory> inventory =
      discover_lds_wrong_address(source, "test_kernel0", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().sites().size(), 2u);
  EXPECT_EQ(inventory.value().sites()[0].offset, 0u);
  EXPECT_EQ(inventory.value().sites()[0].affected_owners.size(), 1u);
  EXPECT_EQ(inventory.value().sites()[1].offset, 512u);
  EXPECT_EQ(inventory.value().sites()[1].affected_owners.size(), 2u);
  const FailureOr<LdsSelection> private_site =
      select_lds_wrong_address(inventory.value(), {}, error);
  ASSERT_TRUE(private_site.succeeded());
  const FailureOr<LdsSelection> shared_site =
      select_lds_wrong_address(inventory.value(), {.exact = std::nullopt, .ordinal = 1}, error);
  ASSERT_TRUE(shared_site.succeeded());
  const FailureOr<LdsResult> private_result =
      apply_lds_wrong_address(source, private_site.value(), 4, error);
  ASSERT_TRUE(private_result.succeeded()) << error.message;
  EXPECT_TRUE(test::lds_owner_output_matches(source, 0, 4, private_result.value().image));
  EXPECT_TRUE(apply_lds_wrong_address(source, shared_site.value(), 4, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::InvalidParameters);
  const FailureOr<LdsResult> shared_result =
      apply_lds_wrong_address(source, shared_site.value(), 3, error);
  ASSERT_TRUE(shared_result.succeeded()) << error.message;
  EXPECT_TRUE(test::lds_owner_output_matches(source, 512, 3, shared_result.value().image));
  EXPECT_TRUE(std::ranges::equal(source.image(), image));
}

TEST(FaultLdsWrongAddress, SharedRdnaMixedWavesUseEachOwnersAllocation) {
  std::vector<uint8_t> image = test::make_shared_lds_elf(false);
  constexpr size_t offset = 0x100 + 131 * sizeof(uint32_t) + sizeof(amdhsa::kernel_descriptor_t);
  amdhsa::kernel_descriptor_t sibling = {};
  std::memcpy(&sibling, image.data() + offset, sizeof(sibling));
  RJ_AMDHSA_BITS_SET(sibling.kernel_code_properties,
                     amdhsa::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32, 1);
  std::memcpy(image.data() + offset, &sibling, sizeof(sibling));
  const FaultMutationContext source(image);
  FaultMutationError error;
  for (std::string_view entry : {"test_kernel0", "test_kernel1"}) {
    const FailureOr<LdsInventory> inventory = discover_lds_wrong_address(source, entry, error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    ASSERT_EQ(inventory.value().sites().size(), 1u);
    EXPECT_EQ(inventory.value().sites()[0].affected_owners.size(), 2u);
    const FailureOr<LdsSelection> selection =
        select_lds_wrong_address(inventory.value(), {}, error);
    ASSERT_TRUE(selection.succeeded());
    EXPECT_TRUE(apply_lds_wrong_address(source, selection.value(), 4, error).failed());
    const FailureOr<LdsResult> mutant =
        apply_lds_wrong_address(source, selection.value(), 3, error);
    ASSERT_TRUE(mutant.succeeded()) << error.message;
    EXPECT_TRUE(test::lds_owner_output_matches(source, 512, 3, mutant.value().image));
  }
}

TEST(FaultLdsWrongAddress, SharedSelectionRejectsActualUnrelatedKernel) {
  const std::vector<uint8_t> image = test::make_shared_lds_with_unrelated_owner();
  const FaultMutationContext source(image);
  FaultMutationError error;
  const FailureOr<LdsInventory> owner = discover_lds_wrong_address(source, "test_kernel0", error);
  ASSERT_TRUE(owner.succeeded()) << error.message;
  ASSERT_EQ(owner.value().sites().size(), 1u);
  ASSERT_EQ(owner.value().sites()[0].affected_owners.size(), 2u);
  const FailureOr<LdsSelection> selection = select_lds_wrong_address(owner.value(), {}, error);
  ASSERT_TRUE(selection.succeeded());
  const FailureOr<LdsInventory> sibling = discover_lds_wrong_address(source, "test_kernel1", error);
  ASSERT_TRUE(sibling.succeeded()) << error.message;
  const FailureOr<LdsSelection> rebound = select_lds_wrong_address(
      sibling.value(), {.exact = selection.value(), .ordinal = std::numeric_limits<size_t>::max()},
      error);
  ASSERT_TRUE(rebound.succeeded()) << error.message;
  ASSERT_NE(rebound.value().owner(), nullptr);
  EXPECT_EQ(rebound.value().owner()->entry, "test_kernel1");
  const FailureOr<LdsSelection> reversed =
      select_lds_wrong_address(owner.value(), {.exact = rebound.value()}, error);
  ASSERT_TRUE(reversed.succeeded()) << error.message;
  ASSERT_NE(reversed.value().owner(), nullptr);
  EXPECT_EQ(reversed.value().owner()->entry, "test_kernel0");
  const FailureOr<LdsInventory> unrelated = discover_lds_wrong_address(source, "unrelated", error);
  ASSERT_TRUE(unrelated.succeeded()) << error.message;
  EXPECT_TRUE(unrelated.value().sites().empty());
  EXPECT_TRUE(
      select_lds_wrong_address(unrelated.value(), {.exact = selection.value()}, error).failed());
  EXPECT_TRUE(select_lds_wrong_address(unrelated.value(), {}, error).failed());
  const FailureOr<LdsResult> valid = apply_lds_wrong_address(source, selection.value(), 1, error);
  ASSERT_TRUE(valid.succeeded()) << error.message;
  EXPECT_TRUE(test::lds_owner_output_matches(source, 512, 1, valid.value().image));
  const FailureOr<LdsResult> rebound_result =
      apply_lds_wrong_address(source, rebound.value(), 1, error);
  ASSERT_TRUE(rebound_result.succeeded()) << error.message;
  EXPECT_EQ(valid.value().image, rebound_result.value().image);
}

TEST(FaultLdsWrongAddress, SharedSelectionCannotOutliveItsSourceContext) {
  const std::vector<uint8_t> image = test::make_shared_lds_elf(false);
  std::unique_ptr<FaultMutationContext> source = std::make_unique<FaultMutationContext>(image);
  FaultMutationError error;
  const FailureOr<LdsInventory> inventory =
      discover_lds_wrong_address(*source, "test_kernel0", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const FailureOr<LdsSelection> selection = select_lds_wrong_address(inventory.value(), {}, error);
  ASSERT_TRUE(selection.succeeded());
  source.reset();
  EXPECT_TRUE(select_lds_wrong_address(inventory.value(), {}, error).failed());
  const FaultMutationContext foreign(image);
  EXPECT_TRUE(apply_lds_wrong_address(foreign, selection.value(), 1, error).failed());
}

TEST(FaultLdsWrongAddress, SharedPathRejectsPartialLoadAliasAndDynamicRelocations) {
  for (bool alias : {false, true}) {
    std::vector<uint8_t> image = test::make_shared_lds_elf(false);
    Elf64_Ehdr header = {};
    std::memcpy(&header, image.data(), sizeof(header));
    if (alias) {
      const Elf64_Phdr segment = {.p_type = PT_LOAD,
                                  .p_flags = 4,
                                  .p_offset = 0x104,
                                  .p_vaddr = 0x900000,
                                  .p_paddr = 0x900000,
                                  .p_filesz = 4,
                                  .p_memsz = 4,
                                  .p_align = 1};
      const size_t offset = header.e_phoff + header.e_phnum * header.e_phentsize;
      ASSERT_LE(offset + sizeof(segment), 0x100u);
      std::memcpy(image.data() + offset, &segment, sizeof(segment));
      ++header.e_phnum;
    } else {
      std::vector<Elf64_Shdr> sections(header.e_shnum);
      std::memcpy(sections.data(), image.data() + header.e_shoff,
                  sections.size() * sizeof(Elf64_Shdr));
      const Elf64_Shdr relocation = {.sh_name = 0,
                                     .sh_type = SHT_RELA,
                                     .sh_flags = 0,
                                     .sh_addr = 0,
                                     .sh_offset = 0,
                                     .sh_size = 0,
                                     .sh_link = 0,
                                     .sh_info = 0,
                                     .sh_addralign = 0,
                                     .sh_entsize = sizeof(Elf64_Rela)};
      sections.push_back(relocation);
      header.e_shoff = image.size();
      header.e_shnum = static_cast<uint16_t>(sections.size());
      image.resize(image.size() + sections.size() * sizeof(Elf64_Shdr));
      std::memcpy(image.data() + header.e_shoff, sections.data(),
                  sections.size() * sizeof(Elf64_Shdr));
    }
    std::memcpy(image.data(), &header, sizeof(header));
    const FaultMutationContext source(image);
    FaultMutationError error;
    EXPECT_TRUE(discover_lds_wrong_address(source, "test_kernel0", error).failed());
  }
}

TEST(FaultLdsWrongAddress, SharedHelperRangeAndReferenceFailuresLeavePristineSource) {
  const std::vector<uint8_t> image = test::make_shared_lds_elf(false, 32770);
  const FaultMutationContext source(image);
  FaultMutationError error;
  const FailureOr<LdsInventory> inventory =
      discover_lds_wrong_address(source, "test_kernel0", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const FailureOr<LdsSelection> selection = select_lds_wrong_address(inventory.value(), {}, error);
  ASSERT_TRUE(selection.succeeded());
  const FailureOr<LdsResult> long_result =
      apply_lds_wrong_address(source, selection.value(), 1, error);
  EXPECT_TRUE(long_result.failed());
  EXPECT_TRUE(std::ranges::equal(source.image(), image));

  std::vector<uint8_t> bad_image = test::make_shared_lds_elf(false);
  const uint32_t interior_branch = build_s_branch(96, IsaFamily::Rdna4).value();
  std::memcpy(bad_image.data() + 0x100 + 128, &interior_branch, sizeof(interior_branch));
  const FaultMutationContext bad_source(bad_image);
  const FailureOr<LdsInventory> bad_inventory =
      discover_lds_wrong_address(bad_source, "test_kernel0", error);
  ASSERT_TRUE(bad_inventory.succeeded()) << error.message;
  const FailureOr<LdsSelection> bad_selection =
      select_lds_wrong_address(bad_inventory.value(), {}, error);
  ASSERT_TRUE(bad_selection.succeeded());
  for (int attempt = 0; attempt < 2; ++attempt) {
    EXPECT_TRUE(apply_lds_wrong_address(bad_source, bad_selection.value(), 1, error).failed());
    EXPECT_TRUE(std::ranges::equal(bad_source.image(), bad_image));
  }
}

TEST(FaultLdsWrongAddress, CdnaPreloadEntryIsIncludedInLdsInventory) {
  const uint32_t nop = build_s_nop(0, IsaFamily::Cdna4).value();
  std::vector<uint32_t> words(64, nop);
  words[0] = build_s_endpgm(IsaFamily::Cdna4).value();
  words.insert(words.end(), {0xd81a0004u, 0x00000302u, build_s_endpgm(IsaFamily::Cdna4).value()});
  std::vector<uint8_t> image =
      test::make_amdgpu_kernel_elf(words, 0, 3, EF_AMDGPU_MACH_AMDGCN_GFX950, 3, 3);
  image[EI_DATA] = ELFDATA2LSB;
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  const size_t offset = 0x100 + words.size() * sizeof(uint32_t);
  amdhsa::kernel_descriptor_t descriptor = {};
  std::memcpy(&descriptor, image.data() + offset, sizeof(descriptor));
  RJ_AMDHSA_BITS_SET(descriptor.kernarg_preload, amdhsa::KERNARG_PRELOAD_SPEC_LENGTH, 1);
  std::memcpy(image.data() + offset, &descriptor, sizeof(descriptor));
  const FaultMutationContext source(image);
  FaultMutationError error;
  const FailureOr<LdsInventory> inventory =
      discover_lds_wrong_address(source, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().sites().size(), 1u);
  EXPECT_EQ(inventory.value().sites()[0].offset, 256u);
  ASSERT_EQ(inventory.value().sites()[0].affected_owners.size(), 1u);
  const FailureOr<LdsSelection> selection = select_lds_wrong_address(inventory.value(), {}, error);
  ASSERT_TRUE(selection.succeeded());
  const FailureOr<LdsResult> mutant = apply_lds_wrong_address(source, selection.value(), 6, error);
  ASSERT_TRUE(mutant.succeeded()) << error.message;
  EXPECT_TRUE(test::lds_owner_output_matches(source, 256, 6, mutant.value().image));
  EXPECT_TRUE(discover_drop_barriers(source, "test_kernel", error).failed());
}

TEST(FaultLdsWrongAddress, PreloadFallbackRejectsMalformedSiblingAliasesAndRelocation) {
  for (int defect : {0, 1, 2, 3, 4}) {
    std::vector<uint8_t> image = test::make_shared_lds_elf(false, 0, true);
    constexpr size_t descriptor0 = 0x100 + 131 * sizeof(uint32_t);
    constexpr size_t descriptor1 = descriptor0 + sizeof(amdhsa::kernel_descriptor_t);
    const uint32_t end = build_s_endpgm(IsaFamily::Cdna4).value();
    std::memcpy(image.data() + 0x100 + 128, &end, sizeof(end));
    amdhsa::kernel_descriptor_t first = {}, sibling = {};
    std::memcpy(&first, image.data() + descriptor0, sizeof(first));
    std::memcpy(&sibling, image.data() + descriptor1, sizeof(sibling));
    RJ_AMDHSA_BITS_SET(first.kernarg_preload, amdhsa::KERNARG_PRELOAD_SPEC_LENGTH, 1);
    sibling.kernel_code_entry_byte_offset -= 128;
    if (defect == 1)
      sibling.kernel_code_entry_byte_offset = std::numeric_limits<int64_t>::max();
    std::memcpy(image.data() + descriptor0, &first, sizeof(first));
    std::memcpy(image.data() + descriptor1, &sibling, sizeof(sibling));
    Elf64_Ehdr header = {};
    std::memcpy(&header, image.data(), sizeof(header));
    if (defect == 2 || defect == 3) {
      const uint64_t alias_address = defect == 2 ? 0x900000 : 0x1104;
      const Elf64_Phdr alias = {.p_type = PT_LOAD,
                                .p_flags = 4,
                                .p_offset = defect == 2 ? 0x104 : image.size(),
                                .p_vaddr = alias_address,
                                .p_paddr = alias_address,
                                .p_filesz = 4,
                                .p_memsz = 4,
                                .p_align = 1};
      if (defect == 3)
        image.resize(image.size() + 4);
      const size_t offset = header.e_phoff + header.e_phnum * header.e_phentsize;
      std::memcpy(image.data() + offset, &alias, sizeof(alias));
      ++header.e_phnum;
    } else if (defect == 4) {
      Elf64_Phdr rodata = {};
      std::memcpy(&rodata, image.data() + header.e_phoff + sizeof(Elf64_Phdr), sizeof(rodata));
      const Elf64_Rela relocation = {.r_offset = rodata.p_vaddr,
                                     .r_info = (uint64_t{1} << 32) | R_AMDGPU_ABS32_LO,
                                     .r_addend = 0};
      std::vector<Elf64_Shdr> sections(header.e_shnum);
      std::memcpy(sections.data(), image.data() + header.e_shoff,
                  sections.size() * sizeof(Elf64_Shdr));
      const Elf64_Shdr section = {.sh_name = 0,
                                  .sh_type = SHT_RELA,
                                  .sh_flags = 0,
                                  .sh_addr = 0,
                                  .sh_offset = image.size(),
                                  .sh_size = sizeof(relocation),
                                  .sh_link = 3,
                                  .sh_info = 0,
                                  .sh_addralign = 0,
                                  .sh_entsize = sizeof(relocation)};
      image.resize(image.size() + sizeof(relocation));
      std::memcpy(image.data() + section.sh_offset, &relocation, sizeof(relocation));
      sections.push_back(section);
      header.e_shoff = image.size();
      header.e_shnum = static_cast<uint16_t>(sections.size());
      image.resize(image.size() + sections.size() * sizeof(Elf64_Shdr));
      std::memcpy(image.data() + header.e_shoff, sections.data(),
                  sections.size() * sizeof(Elf64_Shdr));
    }
    std::memcpy(image.data(), &header, sizeof(header));
    const FaultMutationContext source(image);
    FaultMutationError error;
    const FailureOr<LdsInventory> inventory =
        discover_lds_wrong_address(source, "test_kernel0", error);
    if (defect == 0) {
      ASSERT_TRUE(inventory.succeeded()) << error.message;
      ASSERT_EQ(inventory.value().sites().size(), 1u);
    } else {
      EXPECT_TRUE(inventory.failed()) << defect;
    }
  }
}

TEST(FaultLdsWrongAddress, FailedPreloadStructuralFactsAreNeverPublished) {
  constexpr IsaFamily arch = IsaFamily::Cdna4;
  std::vector<uint32_t> words(64, build_s_nop(0, arch).value());
  words[0] = build_s_endpgm(arch).value();
  words[32] = 0xffffffffu; // Invalid, unreachable from either hardware root.
  words.insert(words.end(), {0xd81a0004u, 0x00000302u, build_s_endpgm(arch).value()});
  std::vector<uint8_t> image =
      test::make_amdgpu_kernel_elf(words, 0, 3, EF_AMDGPU_MACH_AMDGCN_GFX950, 3, 3);
  image[EI_DATA] = ELFDATA2LSB;
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  const size_t offset = 0x100 + words.size() * sizeof(uint32_t);
  amdhsa::kernel_descriptor_t descriptor = {};
  std::memcpy(&descriptor, image.data() + offset, sizeof(descriptor));
  RJ_AMDHSA_BITS_SET(descriptor.kernarg_preload, amdhsa::KERNARG_PRELOAD_SPEC_LENGTH, 1);
  std::memcpy(image.data() + offset, &descriptor, sizeof(descriptor));
  const std::shared_ptr<const AmdGpuCodeObject> owner_source =
      std::make_shared<const AmdGpuCodeObject>(image.data(), image.size());
  std::unique_ptr<Decoder> decoder =
      Decoder::create(default_isa_target_registry(), ROCJITSU_CODE_TARGET_GFX950);
  ASSERT_TRUE(decoder);
  ASSERT_TRUE(decoder->decode_window(std::span(words).subspan(32), 128).failed());
  std::string detail;
  EXPECT_TRUE(KernelOwnership::build(owner_source, ROCJITSU_CODE_TARGET_GFX950, detail).succeeded())
      << detail;
  const FaultMutationContext source(image);
  FaultMutationError error;
  for (int attempt = 0; attempt < 2; ++attempt) {
    EXPECT_TRUE(discover_lds_wrong_address(source, "test_kernel", error).failed());
    EXPECT_NE(error.message.find("invalid source instruction"), std::string::npos);
    EXPECT_TRUE(std::ranges::equal(source.image(), image));
  }
}

TEST(FaultLdsWrongAddress, SharedGfx1250RequiresEveryOwnerToProveSameBank) {
  constexpr IsaFamily arch = IsaFamily::Cdna5;
  const uint32_t nop = build_s_nop(0, arch).value();
  const uint32_t end = build_s_endpgm(arch).value();
  const std::array<uint32_t, 1> set_one = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 1});
  const std::array<uint32_t, 1> set_zero = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp);
  const std::array<uint32_t, 1> dynamic = cdna5::build_sopk(
      cdna5::kSSetregB32Sopk, {.simm16 = 1u | (14u << 6) | (1u << 11), .sdst = 0});
  const std::array<uint32_t, 2> access =
      cdna5::build_vds(cdna5::kDsStoreB32Vds, {.offset0 = 4, .addr = 2, .data0 = 3});
  for (int mode : {0, 1, 2}) {
    const bool agree = mode == 1;
    std::vector<uint32_t> words(128, nop);
    words[0] = set_one[0];
    words[1] = build_s_call_b64(30, 126, arch).value();
    words[2] = end;
    words[64] = agree ? set_one[0] : (mode == 2 ? dynamic[0] : set_zero[0]);
    words[65] = build_s_call_b64(30, 62, arch).value();
    words[66] = end;
    words.insert(words.end(), {access[0], access[1], set_zero[0], access[0], access[1],
                               build_s_setpc_b64(30, arch).value()});
    std::vector<uint8_t> image =
        test::make_amdgpu_two_kernel_elf(words, 0, 3, EF_AMDGPU_MACH_AMDGCN_GFX1250);
    image[EI_DATA] = ELFDATA2LSB;
    image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
    const size_t first_descriptor = 0x100 + words.size() * sizeof(uint32_t);
    for (size_t owner = 0; owner < 2; ++owner) {
      const size_t offset = first_descriptor + owner * sizeof(amdhsa::kernel_descriptor_t);
      amdhsa::kernel_descriptor_t descriptor = {};
      std::memcpy(&descriptor, image.data() + offset, sizeof(descriptor));
      descriptor.kernel_code_entry_byte_offset += owner * 256;
      RJ_AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                         amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT, 16);
      std::memcpy(image.data() + offset, &descriptor, sizeof(descriptor));
    }
    const FaultMutationContext source(image);
    FaultMutationError error;
    for (std::string_view owner : {"test_kernel0", "test_kernel1"}) {
      const FailureOr<LdsInventory> inventory = discover_lds_wrong_address(source, owner, error);
      ASSERT_TRUE(inventory.succeeded()) << error.message;
      // The second site occurs after the helper resets the address bank, so
      // it remains valid even when the incoming owners disagree or are unknown.
      ASSERT_EQ(inventory.value().sites().size(), agree ? 2u : 1u);
      if (!agree) {
        EXPECT_EQ(inventory.value().sites()[0].offset, 524u);
        EXPECT_EQ(inventory.value().sites()[0].address_bank, 0u);
        const FailureOr<LdsSelection> reset =
            select_lds_wrong_address(inventory.value(), {}, error);
        ASSERT_TRUE(reset.succeeded()) << error.message;
        const FailureOr<LdsResult> reset_mutant =
            apply_lds_wrong_address(source, reset.value(), 1, error);
        ASSERT_TRUE(reset_mutant.succeeded()) << error.message;
        EXPECT_TRUE(test::lds_owner_output_matches(source, 524, 1, reset_mutant.value().image));
        continue;
      }
      ASSERT_EQ(inventory.value().sites()[0].affected_owners.size(), 2u);
      EXPECT_EQ(inventory.value().sites()[0].address_bank, 1u);
      EXPECT_EQ(inventory.value().sites()[1].address_bank, 0u);
      const FailureOr<LdsSelection> selection =
          select_lds_wrong_address(inventory.value(), {}, error);
      ASSERT_TRUE(selection.succeeded());
      const FailureOr<LdsResult> mutant =
          apply_lds_wrong_address(source, selection.value(), 1, error);
      ASSERT_TRUE(mutant.succeeded()) << error.message;
      EXPECT_TRUE(test::lds_owner_output_matches(source, 512, 1, mutant.value().image));
      const FailureOr<LdsResult> last_allocated =
          apply_lds_wrong_address(source, selection.value(), 15, error);
      ASSERT_TRUE(last_allocated.succeeded()) << error.message;
      EXPECT_TRUE(test::lds_owner_output_matches(source, 512, 15, last_allocated.value().image));
      EXPECT_TRUE(apply_lds_wrong_address(source, selection.value(), 16, error).failed());
      const FailureOr<LdsSelection> second =
          select_lds_wrong_address(inventory.value(), {.exact = std::nullopt, .ordinal = 1}, error);
      ASSERT_TRUE(second.succeeded());
      const FailureOr<LdsResult> second_mutant =
          apply_lds_wrong_address(source, second.value(), 1, error);
      ASSERT_TRUE(second_mutant.succeeded()) << error.message;
      EXPECT_TRUE(test::lds_owner_output_matches(source, 524, 1, second_mutant.value().image));
    }
  }
}

TEST(FaultLdsWrongAddress, SharedTrampolineOracleRejectsNonAddressAndMappingCorruption) {
  const std::vector<uint8_t> image = test::make_shared_lds_elf(false);
  const FaultMutationContext source(image);
  FaultMutationError error;
  const FailureOr<LdsInventory> inventory =
      discover_lds_wrong_address(source, "test_kernel0", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const FailureOr<LdsSelection> selection = select_lds_wrong_address(inventory.value(), {}, error);
  ASSERT_TRUE(selection.succeeded()) << error.message;
  const FailureOr<LdsResult> mutant = apply_lds_wrong_address(source, selection.value(), 1, error);
  ASSERT_TRUE(mutant.succeeded()) << error.message;
  EXPECT_TRUE(test::lds_owner_output_matches(source, 512, 1, mutant.value().image));
  const Section &text = *source.source().text_sections().front();
  ASSERT_TRUE(text.sectionHeaderIndex());
  const AmdGpuCodeObject emitted(mutant.value().image.data(), mutant.value().image.size());
  const Section &emitted_text = *emitted.text_sections().front();
  Elf64_Ehdr header = {};
  std::memcpy(&header, mutant.value().image.data(), sizeof(header));
  const uint64_t mapping = header.e_shoff +
                           *emitted_text.sectionHeaderIndex() * sizeof(Elf64_Shdr) +
                           offsetof(Elf64_Shdr, sh_addr);
  const std::vector<KernelDescriptorInfo> kernels =
      scan_kernel_descriptors(mutant.value().image, emitted_text.sectionOffset(),
                              emitted_text.size(), emitted_text.sectionHeaderIndex());
  ASSERT_EQ(kernels.size(), 2u);
  const uint64_t descriptor = kernels.front().descriptor_file_offset;
  const std::optional<uint64_t> guest =
      test::lds_emitted_guest(emitted_text, 512, source.source().target_id());
  ASSERT_TRUE(guest);
  for (uint64_t offset : std::array<uint64_t, 4>{emitted_text.sectionOffset() + *guest + 5,
                                                 text.sectionOffset(), descriptor, mapping}) {
    SCOPED_TRACE(offset);
    std::vector<uint8_t> corrupted = mutant.value().image;
    ASSERT_LT(offset, corrupted.size());
    corrupted[offset] ^= 1;
    EXPECT_FALSE(test::lds_owner_output_matches(source, 512, 1, corrupted));
  }
  EXPECT_TRUE(std::ranges::equal(source.image(), image));
}

} // namespace
} // namespace rocjitsu
