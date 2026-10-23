// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "code_object_fuzz_core.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/executable.h"
#include "scoped_temp.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string_view>
#include <vector>

namespace rocjitsu::code_object_fuzz {
namespace {

TEST(CodeObjectFuzzTest, SeedsReachStandaloneAndBundledCodeObjects) {
  RunStats total;
  for (const auto &seed : seed_corpus())
    total += run_input(seed);

  EXPECT_EQ(total.inputs, 4u);
  EXPECT_GE(total.code_objects_valid, 3u);
  EXPECT_GE(total.executables_valid, 2u);
  EXPECT_GE(total.bundled_code_objects, 2u);
  EXPECT_GE(total.text_sections, 3u);
  EXPECT_GE(total.descriptors, 3u);
  EXPECT_GE(total.load_segments, 6u);
}

TEST(CodeObjectFuzzTest, ArbitraryBytesAreRejectedWithoutThrowing) {
  const std::array inputs = {
      std::vector<uint8_t>{},
      std::vector<uint8_t>{0x7f, 'E', 'L', 'F'},
      std::vector<uint8_t>(128, 0xff),
      std::vector<uint8_t>(512, 0),
  };
  for (const auto &input : inputs) {
    SCOPED_TRACE(input.size());
    const RunStats stats = run_input(input);
    EXPECT_EQ(stats.inputs, 1u);
  }
}

TEST(CodeObjectFuzzTest, HostElfWithoutSectionHeadersIsRejected) {
  const std::vector<uint8_t> image = seed_corpus()[2];
  const Executable executable(image);
  EXPECT_FALSE(executable.is_valid());
}

TEST(CodeObjectFuzzTest, DeviceElfWithoutSectionHeadersIsRejected) {
  std::vector<uint8_t> image = seed_corpus()[0];
  Elf64_Ehdr header{};
  std::memcpy(&header, image.data(), sizeof(header));
  header.e_shnum = 0;
  std::memcpy(image.data(), &header, sizeof(header));
  const AmdGpuCodeObject object(image.data(), image.size(), "", "");
  const Executable executable(image);
  EXPECT_FALSE(object.is_valid());
  EXPECT_FALSE(executable.is_valid());
  EXPECT_EQ(run_input(image).inputs, 1u);
}

TEST(CodeObjectFuzzTest, GeneratorIsDeterministicAndBounded) {
  constexpr std::array<uint64_t, 5> seeds = {0, 1, 17, 97, std::numeric_limits<uint64_t>::max()};
  for (uint64_t seed : seeds) {
    SCOPED_TRACE(seed);
    const auto first = generate_input(seed);
    EXPECT_EQ(first, generate_input(seed));
    EXPECT_LE(first.size(), kMaxInputSize);
  }
}

TEST(CodeObjectFuzzTest, InMemoryExecutableCopiesTheSameBundleAsPathLoading) {
  std::vector<uint8_t> image = seed_corpus()[1];
  test::ScopedTempFile file("rj-code-object-fuzz-bundle-");
  file.write(std::string_view(reinterpret_cast<const char *>(image.data()), image.size()));

  const Executable path_executable(file.path());
  const Executable memory_executable(image);
  ASSERT_TRUE(path_executable.is_valid());
  ASSERT_TRUE(memory_executable.is_valid());
  const rj_code_target_id_t target = ROCJITSU_CODE_TARGET_GFX1250;
  ASSERT_EQ(path_executable.num_code_objects(target), 1u);
  ASSERT_EQ(memory_executable.num_code_objects(target), 1u);
  ASSERT_NE(path_executable.code_object(target, 0), nullptr);
  ASSERT_NE(memory_executable.code_object(target, 0), nullptr);
  EXPECT_EQ(path_executable.code_object(target, 0)->image_size(),
            memory_executable.code_object(target, 0)->image_size());

  std::ranges::fill(image, 0);
  EXPECT_EQ(memory_executable.num_code_objects(target), 1u);
  EXPECT_TRUE(memory_executable.code_object(target, 0)->is_valid());
}

TEST(CodeObjectFuzzTest, LoadOracleChecksTheEntireImageRange) {
  const std::array<uint8_t, 16> image{};
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(image.data());

  EXPECT_TRUE(load_segment_in_bounds(image, begin, image.size()));
  EXPECT_TRUE(load_segment_in_bounds(image, begin + image.size(), 0));
  EXPECT_FALSE(load_segment_in_bounds(image, begin - 1, 1));
  EXPECT_FALSE(load_segment_in_bounds(image, begin + image.size(), 1));
  EXPECT_FALSE(load_segment_in_bounds(image, std::numeric_limits<std::uintptr_t>::max(), 16));
}

TEST(CodeObjectFuzzTest, UnknownBundledTargetStillExercisesCodeObject) {
  std::vector<uint8_t> image = seed_corpus()[1];
  Elf64_Ehdr host_ehdr{};
  std::memcpy(&host_ehdr, image.data(), sizeof(host_ehdr));
  std::vector<Elf64_Shdr> shdrs(host_ehdr.e_shnum);
  std::memcpy(shdrs.data(), image.data() + host_ehdr.e_shoff, shdrs.size() * sizeof(Elf64_Shdr));
  ASSERT_GT(shdrs.size(), 1u);
  const uint64_t fatbin_offset = shdrs[1].sh_offset;

  uint64_t object_offset = 0;
  std::memcpy(&object_offset, image.data() + fatbin_offset + 32, sizeof(object_offset));
  Elf64_Ehdr device_ehdr{};
  std::memcpy(&device_ehdr, image.data() + fatbin_offset + object_offset, sizeof(device_ehdr));
  device_ehdr.e_flags = 0;
  std::memcpy(image.data() + fatbin_offset + object_offset, &device_ehdr, sizeof(device_ehdr));

  constexpr std::string_view known_target = "gfx1250";
  const std::vector<uint8_t>::iterator target =
      std::search(image.begin(), image.end(), known_target.begin(), known_target.end());
  ASSERT_NE(target, image.end());
  std::copy_n("gfx9999", known_target.size(), target);

  const RunStats stats = run_input(image);
  EXPECT_EQ(stats.executables_valid, 1u);
  EXPECT_EQ(stats.bundled_code_objects, 1u);
  EXPECT_EQ(stats.code_objects_valid, 1u);
}

} // namespace
} // namespace rocjitsu::code_object_fuzz
