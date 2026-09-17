/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>

#include <fstream>
#include <map>
#include <vector>

#include "core/inc/amd_aie_elf.h"

namespace {

std::vector<std::uint8_t> ReadFile(const char* path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return {};
  const auto size = static_cast<std::size_t>(f.tellg());
  f.seekg(0);
  std::vector<std::uint8_t> bytes(size);
  f.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
  return bytes;
}

constexpr const char* kElf = "kernel_full_elf_vsadd/aie.elf";

TEST(AieElfParse, ParsesVectorScalarAdd) {
  const auto image = ReadFile(kElf);
  if (image.empty()) GTEST_SKIP() << "full-ELF artifact not built";

  std::map<std::string, rocr::AMD::aie_elf::Kernel> kernels;
  std::string error;
  ASSERT_EQ(rocr::AMD::aie_elf::Parse(image.data(), image.size(), &kernels, &error),
            HSA_STATUS_SUCCESS) << error;
  ASSERT_FALSE(kernels.empty());

  const auto& k = kernels.begin()->second;
  EXPECT_EQ(k.ctrl_code.size(), 316u);   // .ctrltext.0, measured on this branch
  EXPECT_EQ(k.pdi.size(), 2992u);        // .pdi.1
  EXPECT_TRUE(k.has_pdi_patch);
  EXPECT_NE(k.pdi_patch_offset, 0u);
  EXPECT_GT(k.num_args(), 0u);
}

TEST(AieElfParse, RejectsGarbage) {
  const std::vector<std::uint8_t> garbage(512, 0xA5);
  std::map<std::string, rocr::AMD::aie_elf::Kernel> kernels;
  std::string error;
  EXPECT_NE(rocr::AMD::aie_elf::Parse(garbage.data(), garbage.size(), &kernels, &error),
            HSA_STATUS_SUCCESS);
  EXPECT_FALSE(error.empty());
}

TEST(AieElfParse, RejectsTruncated) {
  auto image = ReadFile(kElf);
  if (image.empty()) GTEST_SKIP() << "full-ELF artifact not built";
  image.resize(image.size() / 2);

  std::map<std::string, rocr::AMD::aie_elf::Kernel> kernels;
  std::string error;
  EXPECT_NE(rocr::AMD::aie_elf::Parse(image.data(), image.size(), &kernels, &error),
            HSA_STATUS_SUCCESS);
}

TEST(AieElfPatch, ShimDma48IsAdditive) {
  // Two applications must not equal one: this is why §4 copies from pristine.
  std::uint32_t once[3] = {0, 0x1000, 0};
  std::uint32_t twice[3] = {0, 0x1000, 0};
  rocr::AMD::aie_elf::PatchShimDma48(once, 0x2000);
  rocr::AMD::aie_elf::PatchShimDma48(twice, 0x2000);
  rocr::AMD::aie_elf::PatchShimDma48(twice, 0x2000);
  EXPECT_NE(once[1], twice[1]);
}

}  // namespace
