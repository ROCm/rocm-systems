// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/executable.h"
#include "rocjitsu/code/file_io.h"
#include "rocjitsu/code/rj_code.h"
#include "scoped_temp.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <system_error>
#include <vector>

namespace rocjitsu {
namespace {

class FileLoadingBundleCountTest : public ::testing::TestWithParam<uint64_t> {};

TEST_P(FileLoadingBundleCountTest, RejectsImpossibleCountWithoutThrowing) {
  constexpr char names[] = "\0.shstrtab\0.hip_fatbin";
  constexpr char magic[] = "__CLANG_OFFLOAD_BUNDLE__";
  const uint64_t count = GetParam();
  Elf64_Ehdr header{};
  std::memcpy(header.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  header.e_ident[EI_CLASS] = ELFCLASS64;
  header.e_ident[EI_DATA] = 1; // Little-endian.
  header.e_shoff = sizeof(header);
  header.e_shentsize = sizeof(Elf64_Shdr);
  header.e_shnum = 3;
  header.e_shstrndx = 1;

  std::array<Elf64_Shdr, 3> sections{};
  sections[1].sh_name = 1;
  sections[1].sh_type = SHT_STRTAB;
  sections[1].sh_offset = sizeof(header) + sizeof(sections);
  sections[1].sh_size = sizeof(names);
  sections[2].sh_name = 11;
  sections[2].sh_type = SHT_PROGBITS;
  sections[2].sh_offset = sections[1].sh_offset + sizeof(names);
  sections[2].sh_size = sizeof(magic) - 1 + sizeof(count);

  std::string image(reinterpret_cast<const char *>(&header), sizeof(header));
  image.append(reinterpret_cast<const char *>(sections.data()), sizeof(sections));
  image.append(names, sizeof(names));
  image.append(magic, sizeof(magic) - 1);
  image.append(reinterpret_cast<const char *>(&count), sizeof(count));
  const test::ScopedTempFile file("rocjitsu_malformed_fatbin_");
  file.write(image);

  EXPECT_FALSE(Executable(file.path()).is_valid());
  char sentinel;
  auto *const original = reinterpret_cast<rj_code_executable_t *>(&sentinel);
  rj_code_executable_t *executable = original;
  EXPECT_EQ(rj_code_executable_create(file.path().c_str(), &executable),
            ROCJITSU_STATUS_INVALID_CODE_OBJECT);
  EXPECT_EQ(executable, original);
}

INSTANTIATE_TEST_SUITE_P(HugeCounts, FileLoadingBundleCountTest,
                         ::testing::Values(std::numeric_limits<uint64_t>::max(),
                                           uint64_t{1} << 30));

TEST(FileLoadingTest, RejectsDirectoryWithoutThrowing) {
  const test::ScopedTempDirectory directory("rocjitsu_file_loading_");
  EXPECT_TRUE(detail::read_file_bytes(directory.path()).failed());
  EXPECT_FALSE(Executable(directory.path()).is_valid());
  EXPECT_FALSE(AmdGpuCodeObject(directory.path()).is_valid());

  rj_code_executable_t *executable = nullptr;
  EXPECT_EQ(rj_code_executable_create(directory.path().c_str(), &executable),
            ROCJITSU_STATUS_INVALID_CODE_OBJECT);
  EXPECT_EQ(executable, nullptr);
  rj_code_executable_destroy(executable);
}

TEST(FileLoadingTest, RejectsMissingFileWithoutThrowing) {
  const test::ScopedTempDirectory directory("rocjitsu_file_loading_");
  const std::string path = directory.path() + "/missing";
  EXPECT_TRUE(detail::read_file_bytes(path).failed());
  EXPECT_FALSE(Executable(path).is_valid());
  EXPECT_FALSE(AmdGpuCodeObject(path).is_valid());

  rj_code_executable_t *executable = nullptr;
  EXPECT_EQ(rj_code_executable_create(path.c_str(), &executable),
            ROCJITSU_STATUS_INVALID_CODE_OBJECT);
  EXPECT_EQ(executable, nullptr);
  rj_code_executable_destroy(executable);
}

TEST(FileLoadingTest, ReadsEmptyFileButRejectsItAsCode) {
  const test::ScopedTempFile file("rocjitsu_file_loading_");
  util::FailureOr<std::vector<char>> image = detail::read_file_bytes(file.path());
  ASSERT_TRUE(image.succeeded());
  EXPECT_TRUE(image.value().empty());
  EXPECT_FALSE(Executable(file.path()).is_valid());
  EXPECT_FALSE(AmdGpuCodeObject(file.path()).is_valid());

  rj_code_executable_t *executable = nullptr;
  EXPECT_EQ(rj_code_executable_create(file.path().c_str(), &executable),
            ROCJITSU_STATUS_INVALID_CODE_OBJECT);
  EXPECT_EQ(executable, nullptr);
  rj_code_executable_destroy(executable);
}

TEST(FileLoadingTest, RejectsUnreadableRegularFileWithoutThrowing) {
  const test::ScopedTempFile file("rocjitsu_file_loading_");
  file.write("not empty");
  std::error_code error;
  std::filesystem::permissions(file.path(), std::filesystem::perms::none, error);
  ASSERT_FALSE(error) << error.message();
  ASSERT_TRUE(std::filesystem::is_regular_file(file.path(), error));
  ASSERT_FALSE(error) << error.message();

  {
    std::ifstream probe(file.path(), std::ios::binary);
    if (probe.is_open())
      GTEST_SKIP() << "The process can still read a file with mode 000";
  }

  EXPECT_TRUE(detail::read_file_bytes(file.path()).failed());
  EXPECT_FALSE(Executable(file.path()).is_valid());
  EXPECT_FALSE(AmdGpuCodeObject(file.path()).is_valid());

  rj_code_executable_t *executable = nullptr;
  EXPECT_EQ(rj_code_executable_create(file.path().c_str(), &executable),
            ROCJITSU_STATUS_INVALID_CODE_OBJECT);
  EXPECT_EQ(executable, nullptr);
  rj_code_executable_destroy(executable);
}

} // namespace
} // namespace rocjitsu
