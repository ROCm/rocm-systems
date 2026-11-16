// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "amdgpu_elf_test_support.h"

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <limits>
#include <vector>

namespace rocjitsu {
namespace {

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
