/*
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
*/

// Exercise the production PAL code-object reader without a GPU.

#include <libelf.h>

#include "amd_hsa_code.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace {

using amd::hsa::code::AmdHsaCode;

constexpr uint32_t XnackSettings[] = {
    ELF::EF_AMDGPU_FEATURE_XNACK_UNSUPPORTED_V4,
    ELF::EF_AMDGPU_FEATURE_XNACK_ANY_V4,
    ELF::EF_AMDGPU_FEATURE_XNACK_OFF_V4,
    ELF::EF_AMDGPU_FEATURE_XNACK_ON_V4,
};

void CheckIsa(uint8_t abi, uint32_t flags, const char* target, uint16_t type = ET_DYN,
              unsigned expected_generic_version = 0) {
  SCOPED_TRACE(::testing::Message() << "ABI=" << unsigned(abi) << " flags=" << flags
                                   << " type=" << type << " target=" << target);
  std::unique_ptr<amd::elf::Image> image(amd::elf::NewElf64Image());
  ASSERT_TRUE(image->initNew(ELF::EM_AMDGPU, type, ELF::ELFOSABI_AMDGPU_HSA, abi, flags));
  image->symtab();
  ASSERT_TRUE(image->Freeze());
  std::vector<char> bytes(image->size());
  ASSERT_TRUE(image->copyToBuffer(bytes.data(), bytes.size()));

  AmdHsaCode code;
  ASSERT_TRUE(code.InitAsBuffer(bytes.data(), bytes.size()));
  std::string name;
  unsigned generic_version = 99;
  ASSERT_TRUE(code.GetIsa(name, &generic_version));
  EXPECT_EQ(std::string("amdgcn-amd-amdhsa--") + target, name);
  EXPECT_EQ(expected_generic_version, generic_version);
}

class CodeObjectIsaTest : public ::testing::TestWithParam<uint8_t> {};

TEST_P(CodeObjectIsaTest, SelectableXnackPreservesModifiers) {
  const char* targets[] = {"gfx900", "gfx900", "gfx900:xnack-", "gfx900:xnack+"};
  for (unsigned i = 0; i < 4; ++i) {
    CheckIsa(GetParam(), ELF::EF_AMDGPU_MACH_AMDGCN_GFX900 | XnackSettings[i], targets[i]);
  }
}

TEST_P(CodeObjectIsaTest, UnsupportedXnackHasNoModifier) {
  const struct {
    uint32_t mach;
    const char* target;
  } targets[] = {
      {ELF::EF_AMDGPU_MACH_AMDGCN_GFX1030, "gfx1030"},
      {ELF::EF_AMDGPU_MACH_AMDGCN_GFX1100, "gfx1100"},
      {ELF::EF_AMDGPU_MACH_AMDGCN_GFX1200, "gfx1200"},
      {ELF::EF_AMDGPU_MACH_AMDGCN_GFX12_GENERIC, "gfx12-generic"},
  };
  for (const auto& target : targets) {
    for (uint32_t xnack : XnackSettings) {
      for (uint16_t type : {ET_REL, ET_DYN}) {
        CheckIsa(GetParam(), target.mach | xnack, target.target, type);
      }
    }
  }
}

TEST_P(CodeObjectIsaTest, SramEccModifiersArePreserved) {
  const uint32_t flags = ELF::EF_AMDGPU_MACH_AMDGCN_GFX906 | ELF::EF_AMDGPU_FEATURE_XNACK_ON_V4;
  CheckIsa(GetParam(), flags, "gfx906:xnack+");
  CheckIsa(GetParam(), flags | ELF::EF_AMDGPU_FEATURE_SRAMECC_OFF_V4,
           "gfx906:sramecc-:xnack+");
  CheckIsa(GetParam(), flags | ELF::EF_AMDGPU_FEATURE_SRAMECC_ON_V4,
           "gfx906:sramecc+:xnack+");
}

TEST_P(CodeObjectIsaTest, GenericVersionIsPreserved) {
  const uint32_t flags = ELF::EF_AMDGPU_MACH_AMDGCN_GFX12_GENERIC |
                        ELF::EF_AMDGPU_FEATURE_XNACK_ON_V4 |
                        (2u << ELF::EF_AMDGPU_GENERIC_VERSION_OFFSET);
  CheckIsa(GetParam(), flags, "gfx12-generic", ET_DYN,
           GetParam() == ELF::ELFABIVERSION_AMDGPU_HSA_V6 ? 2 : 0);
}

INSTANTIATE_TEST_SUITE_P(CodeObjectVersions, CodeObjectIsaTest,
                         ::testing::Values(ELF::ELFABIVERSION_AMDGPU_HSA_V4,
                                           ELF::ELFABIVERSION_AMDGPU_HSA_V5,
                                           ELF::ELFABIVERSION_AMDGPU_HSA_V6));

TEST(CodeObjectIsaLegacyTest, Version3XnackIsUnchanged) {
  CheckIsa(ELF::ELFABIVERSION_AMDGPU_HSA_V3, ELF::EF_AMDGPU_MACH_AMDGCN_GFX900,
           "gfx900:xnack-");
  CheckIsa(ELF::ELFABIVERSION_AMDGPU_HSA_V3,
           ELF::EF_AMDGPU_MACH_AMDGCN_GFX900 | ELF::EF_AMDGPU_FEATURE_XNACK_V3,
           "gfx900:xnack+");
}

TEST(CodeObjectIsaLegacyTest, Version2XnackIsUnchanged) {
  for (bool xnack : {false, true}) {
    AmdHsaCode original;
    ASSERT_TRUE(original.InitNew(xnack));
    original.AddNoteCodeObjectVersion(2, 1);
    original.AddNoteHsail(1, 0, HSA_PROFILE_FULL, HSA_MACHINE_MODEL_LARGE,
                          HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT);
    original.AddNoteIsa("AMD", "AMDGPU", 9, 0, 0);
    ASSERT_TRUE(original.Freeze());
    std::vector<char> bytes(original.ElfSize());
    ASSERT_TRUE(original.WriteToBuffer(bytes.data()));

    AmdHsaCode code;
    ASSERT_TRUE(code.InitAsBuffer(bytes.data(), bytes.size()));
    std::string name;
    ASSERT_TRUE(code.GetIsa(name));
    EXPECT_EQ(xnack ? "amdgcn-amd-amdhsa--gfx900:xnack+" : "amdgcn-amd-amdhsa--gfx900:xnack-",
              name);
  }
}

}  // namespace
