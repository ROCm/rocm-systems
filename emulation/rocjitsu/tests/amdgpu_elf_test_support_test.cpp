// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "amdgpu_elf_test_support.h"
#include "dbi_test_util.h"

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string_view>
#include <vector>

namespace rocjitsu {
namespace {

TEST(AmdGpuElfFixture, DbiNamesPreserveOffsetsAndBytesAcrossAppends) {
  for (bool reserve : {false, true}) {
    SCOPED_TRACE(reserve);
    std::vector<uint8_t> names = {0};
    if (reserve)
      names.reserve(64);
    EXPECT_EQ(test::add_elf_name(names, "test_kernel0.kd"), 1u);
    EXPECT_EQ(test::add_elf_name(names, std::string_view{}), 17u);
    constexpr char embedded_null[] = {'x', '\0', 'y', 'z'};
    EXPECT_EQ(test::add_elf_name(names, std::string_view(embedded_null, 3)), 18u);
    const std::vector<uint8_t> expected = {0,   't', 'e', 's', 't', '_', 'k', 'e', 'r', 'n', 'e',
                                           'l', '0', '.', 'k', 'd', 0,   0,   'x', 0,   'y', 0};
    EXPECT_EQ(names, expected);
  }
}

TEST(AmdGpuElfFixture, ReportsInvalidCallerExtentAndTargetWithoutThrowing) {
  const std::vector<uint32_t> text(4, 0);
  std::vector<uint8_t> image;
  EXPECT_NONFATAL_FAILURE(
      image = test_support::make_minimal_amdgpu_elf_with_descriptor_after_text({}, 1),
      "text function extent exceeds .text fixture");
  EXPECT_GE(image.size(), sizeof(Elf64_Ehdr));
  EXPECT_NONFATAL_FAILURE(
      image = test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(text, 3, 2),
      "text function extent exceeds .text fixture");
  EXPECT_GE(image.size(), sizeof(Elf64_Ehdr));
  EXPECT_NONFATAL_FAILURE(image = test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
                              text, 1, std::numeric_limits<size_t>::max()),
                          "text function extent exceeds .text fixture");
  EXPECT_GE(image.size(), sizeof(Elf64_Ehdr));
  EXPECT_NONFATAL_FAILURE(image = test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(
                              text, std::nullopt, 0, 4),
                          "function pointer target exceeds .text fixture");
  EXPECT_GE(image.size(), sizeof(Elf64_Ehdr));
  EXPECT_FALSE(
      test_support::make_minimal_amdgpu_elf_with_descriptor_after_text(text, 2, 2, 3).empty());
}

} // namespace
} // namespace rocjitsu
