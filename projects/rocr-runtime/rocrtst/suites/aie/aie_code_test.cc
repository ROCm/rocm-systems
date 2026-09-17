/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Exercises AMD::AieCode::Create/Parse (core/inc/amd_aie_code.hpp) directly, the way
// elf_parse_test.cc exercises aie_elf::Parse: the parser sources are compiled directly into this
// binary because core/inc is not installed alongside the runtime, and amd::elf::Image (which
// AieCode depends on) is built with hidden visibility inside libhsa-runtime64.so, so it cannot be
// linked against from outside that shared library.

#include <gtest/gtest.h>

#include <fstream>
#include <vector>

#include "core/inc/amd_elf_image.hpp"
#include "core/inc/amd_aie_code.hpp"

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

// Same hsaco dispatch.cc's HsacoKernelObjectIsPublished loads; built by the same CMake rule.
constexpr const char* kHsaco = "vsadd.hsaco";

TEST(AieCodeParse, ParsesVsaddHsaco) {
  const auto hsaco = ReadFile(kHsaco);
  if (hsaco.empty()) GTEST_SKIP() << "hsaco was not built: " << kHsaco;

  auto code = rocr::AMD::AieCode::Create(hsaco.data(), hsaco.size());
  ASSERT_NE(code, nullptr);

  const auto names = code->GetKernelNames();
  ASSERT_EQ(names.size(), 1u);

  const auto* info = code->GetKernel(names.front());
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->kind, rocr::AMD::AieKernelKind::PdiInsts);
  EXPECT_GT(info->insts_size, 0u);
  EXPECT_GT(info->pdi_size, 0u);
}

TEST(AieCodeParse, RejectsNullAndEmpty) {
  EXPECT_EQ(rocr::AMD::AieCode::Create(nullptr, 0), nullptr);
  const std::vector<std::uint8_t> empty;
  EXPECT_EQ(rocr::AMD::AieCode::Create(empty.data(), empty.size()), nullptr);
}

TEST(AieCodeParse, RejectsGarbage) {
  const std::vector<std::uint8_t> garbage(512, 0xA5);
  EXPECT_EQ(rocr::AMD::AieCode::Create(garbage.data(), garbage.size()), nullptr);
}

TEST(AieCodeParse, RejectsTruncated) {
  auto hsaco = ReadFile(kHsaco);
  if (hsaco.empty()) GTEST_SKIP() << "hsaco was not built: " << kHsaco;
  hsaco.resize(hsaco.size() / 2);

  EXPECT_EQ(rocr::AMD::AieCode::Create(hsaco.data(), hsaco.size()), nullptr);
}

}  // namespace
