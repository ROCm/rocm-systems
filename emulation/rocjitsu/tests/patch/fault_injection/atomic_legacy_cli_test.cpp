// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "atomic_legacy_fixture.h"
#include "cli_test_support.h"
#include "rocjitsu/code/file_io.h"
#include "rocjitsu/code/patch/fault_injection/atomic_wrong_address.h"
#include "scoped_temp.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace {

class FaultLegacyAtomicCli : public ::testing::TestWithParam<test::LegacyAtomicTargetCase> {
protected:
  test::ScopedTempDirectory temporary_;
  std::filesystem::path input_, output_;
  std::string cli_;

  void SetUp() override {
    ASSERT_TRUE(test::mutation_cli_path(cli_));
    ASSERT_TRUE(temporary_.initialize("legacy-atomic-cli-"));
    input_ = std::filesystem::path(temporary_.path()) / "source.hsaco";
    output_ = std::filesystem::path(temporary_.path()) / "mutant.hsaco";
  }

  static void write(const std::filesystem::path &path, std::span<const uint8_t> image) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char *>(image.data()),
               static_cast<std::streamsize>(image.size()));
    file.close();
    ASSERT_FALSE(file.fail());
  }

  static std::vector<uint8_t> read(const std::filesystem::path &path) {
    const FailureOr<std::vector<char>> bytes = detail::read_file_bytes(path.string());
    EXPECT_TRUE(bytes.succeeded()) << path;
    return bytes.succeeded() ? std::vector<uint8_t>(bytes.value().begin(), bytes.value().end())
                             : std::vector<uint8_t>{};
  }

  int run(std::vector<std::string> arguments) const {
    arguments.insert(arguments.begin(), cli_);
    std::vector<char *> pointers;
    for (std::string &argument : arguments)
      pointers.push_back(argument.data());
    pointers.push_back(nullptr);
    const pid_t child = fork();
    if (child == 0) {
      execv(pointers.front(), pointers.data());
      _exit(127);
    }
    if (child < 0)
      return -1;
    int status = 0;
    if (waitpid(child, &status, 0) != child || !WIFEXITED(status))
      return -1;
    return WEXITSTATUS(status);
  }

  std::vector<std::string> args(std::vector<std::string> extra = {},
                                std::string_view entry = "test_kernel",
                                std::string delta = "8") const {
    std::vector<std::string> command = {"--input",          input_.string(),  "--kernel",
                                        std::string(entry), "--kind",         "AtomicWrongAddress",
                                        "--output",         output_.string(), "--address-delta",
                                        std::move(delta)};
    command.insert(command.end(), extra.begin(), extra.end());
    return command;
  }
};

TEST_P(FaultLegacyAtomicCli, NativeFormsExactOrdinalAndLibraryAgree) {
  const test::LegacyAtomicTargetCase &target = GetParam();
  for (test::LegacyAtomicForm form : test::kLegacyAtomicForms) {
    SCOPED_TRACE(test::legacy_atomic_mnemonic(target, form));
    const int32_t offset = test::legacy_atomic_signed(target, form) ? -4 : 4;
    const std::vector<uint8_t> image = test::make_legacy_atomic_elf(target, form, offset);
    ASSERT_NO_FATAL_FAILURE(write(input_, image));
    const FaultMutationContext source(image);
    FaultMutationError error;
    const FailureOr<AtomicInventory> inventory =
        discover_atomic_wrong_address(source, "test_kernel", error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    ASSERT_EQ(inventory.value().sites().size(), 1u);
    EXPECT_EQ(inventory.value().sites().front().instruction_size, 8u);
    const FailureOr<AtomicSelection> selection =
        select_atomic_wrong_address(inventory.value(), {}, error);
    ASSERT_TRUE(selection.succeeded()) << error.message;
    const FailureOr<AtomicResult> expected =
        apply_atomic_wrong_address(source, selection.value(), 8, error);
    ASSERT_TRUE(expected.succeeded()) << error.message;
    for (const std::vector<std::string> &selector :
         {std::vector<std::string>{"--ordinal", "0"},
          std::vector<std::string>{"--exact", "0", "--ordinal", "999"}}) {
      ASSERT_EQ(run(args(selector)), 0);
      const std::vector<uint8_t> actual = read(output_);
      EXPECT_EQ(actual, expected.value().image);
      EXPECT_TRUE(test::legacy_atomic_output_matches(source, 0, 8, actual));
      EXPECT_EQ(read(input_), image);
      ASSERT_TRUE(std::filesystem::remove(output_));
    }
  }
}

TEST_P(FaultLegacyAtomicCli, UnsupportedWidthKeepsOrdinalAndFailuresLeaveNoOutput) {
  const test::LegacyAtomicTargetCase &target = GetParam();
  const std::vector<uint8_t> image = test::make_mixed_legacy_atomic_elf(target);
  ASSERT_NO_FATAL_FAILURE(write(input_, image));
  const FaultMutationContext source(image);
  FaultMutationError error;
  const FailureOr<AtomicInventory> inventory =
      discover_atomic_wrong_address(source, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().sites().size(), 4u);
  EXPECT_EQ(inventory.value().sites().front().form, "ds_add_u64");
  EXPECT_FALSE(inventory.value().sites().front().address_supported);
  EXPECT_NE(run(args({"--ordinal", "0"})), 0);
  EXPECT_FALSE(std::filesystem::exists(output_));
  for (uint32_t ordinal : {1u, 2u, 3u}) {
    const FailureOr<AtomicSelection> selected = select_atomic_wrong_address(
        inventory.value(), {.exact = std::nullopt, .ordinal = ordinal}, error);
    ASSERT_TRUE(selected.succeeded()) << error.message;
    const FailureOr<AtomicResult> expected =
        apply_atomic_wrong_address(source, selected.value(), 8, error);
    ASSERT_TRUE(expected.succeeded()) << error.message;
    ASSERT_EQ(run(args({"--ordinal", std::to_string(ordinal)})), 0);
    const std::vector<uint8_t> actual = read(output_);
    EXPECT_EQ(actual, expected.value().image);
    EXPECT_TRUE(test::legacy_atomic_output_matches(source, ordinal * 8, 8, actual));
    ASSERT_TRUE(std::filesystem::remove(output_));
  }
  for (const std::vector<std::string> &selector :
       {std::vector<std::string>{"--ordinal", "4"},
        std::vector<std::string>{"--exact", "4", "--ordinal", "1"},
        std::vector<std::string>{"--target", "gfx1250"}}) {
    EXPECT_NE(run(args(selector)), 0);
    EXPECT_FALSE(std::filesystem::exists(output_));
  }
  EXPECT_EQ(read(input_), image);
  for (test::LegacyAtomicForm form :
       {test::LegacyAtomicForm::FlatAdd, test::LegacyAtomicForm::GlobalAdd,
        test::LegacyAtomicForm::BufferAdd}) {
    const std::vector<uint8_t> bounded = test::make_legacy_atomic_elf(target, form, 4092);
    ASSERT_NO_FATAL_FAILURE(write(input_, bounded));
    EXPECT_NE(run(args({}, "test_kernel", "4")), 0);
    EXPECT_FALSE(std::filesystem::exists(output_));
    EXPECT_EQ(read(input_), bounded);
    const std::vector<uint8_t> valid = test::make_legacy_atomic_elf(target, form, 0);
    ASSERT_NO_FATAL_FAILURE(write(input_, valid));
    for (const char *delta : {"0", "2", "4096", "-4", "4294967296"}) {
      EXPECT_NE(run(args({}, "test_kernel", delta)), 0);
      EXPECT_FALSE(std::filesystem::exists(output_));
      EXPECT_EQ(read(input_), valid);
    }
  }
  const std::vector<uint8_t> valid = test::make_legacy_atomic_elf(target);
  ASSERT_NO_FATAL_FAILURE(write(input_, valid));
  const std::vector<uint8_t> sentinel = {'k', 'e', 'e', 'p'};
  ASSERT_NO_FATAL_FAILURE(write(output_, sentinel));
  EXPECT_NE(run(args()), 0);
  EXPECT_EQ(read(output_), sentinel);
  EXPECT_EQ(read(input_), valid);
}

TEST_P(FaultLegacyAtomicCli, InvalidScratchSegmentRefusesListAndLaterSupportedSiteSelections) {
  const test::LegacyAtomicTargetCase &target = GetParam();
  const std::vector<uint8_t> image = test::make_invalid_scratch_legacy_atomic_elf(target);
  ASSERT_NO_FATAL_FAILURE(write(input_, image));
  EXPECT_NE(run({"--input", input_.string(), "--kernel", "test_kernel", "--kind",
                 "AtomicWrongAddress", "--list"}),
            0);
  EXPECT_FALSE(std::filesystem::exists(output_));
  EXPECT_EQ(read(input_), image);
  for (const std::vector<std::string> &selector :
       {std::vector<std::string>{"--ordinal", "0"}, std::vector<std::string>{"--ordinal", "1"},
        std::vector<std::string>{"--exact", "8", "--ordinal", "999"}}) {
    EXPECT_NE(run(args(selector)), 0);
    EXPECT_FALSE(std::filesystem::exists(output_));
    EXPECT_EQ(read(input_), image);
  }
}

TEST_P(FaultLegacyAtomicCli, SharedHelperExactBindsEveryActualOwner) {
  const test::LegacyAtomicTargetCase &target = GetParam();
  const std::vector<uint8_t> image = test::make_shared_legacy_atomic_elf(target);
  ASSERT_NO_FATAL_FAILURE(write(input_, image));
  const FaultMutationContext source(image);
  for (std::string_view entry : {"test_kernel0", "test_kernel1"}) {
    ASSERT_EQ(run(args({"--exact", "512", "--ordinal", "999"}, entry)), 0);
    EXPECT_TRUE(test::legacy_atomic_output_matches(source, 512, 8, read(output_)));
    EXPECT_EQ(read(input_), image);
    ASSERT_TRUE(std::filesystem::remove(output_));
  }
  const std::vector<uint8_t> isolated = test::make_shared_legacy_atomic_elf(target, false);
  ASSERT_NO_FATAL_FAILURE(write(input_, isolated));
  EXPECT_NE(run(args({"--exact", "512"}, "test_kernel1")), 0);
  EXPECT_FALSE(std::filesystem::exists(output_));
  EXPECT_EQ(read(input_), isolated);
}

TEST_P(FaultLegacyAtomicCli, FlatPaddingPreservesUnsignedDeltaAndOverflowLeavesNoOutput) {
  const test::LegacyAtomicTargetCase &target = GetParam();
  for (test::LegacyAtomicForm form :
       {test::LegacyAtomicForm::FlatAdd, test::LegacyAtomicForm::FlatCas}) {
    for (int32_t offset : {4, 4092}) {
      SCOPED_TRACE(offset);
      std::array<uint32_t, 2> words = test::legacy_atomic_words(target, form, offset);
      words[0] |= 0x1000u;
      const std::vector<uint8_t> image =
          test::make_legacy_atomic_sequence_elf(target, {words[0], words[1]});
      ASSERT_NO_FATAL_FAILURE(write(input_, image));
      const FaultMutationContext source(image);
      for (const std::vector<std::string> &selector :
           {std::vector<std::string>{"--ordinal", "0"},
            std::vector<std::string>{"--exact", "0", "--ordinal", "999"}}) {
        if (offset == 4092) {
          EXPECT_NE(run(args(selector)), 0);
          EXPECT_FALSE(std::filesystem::exists(output_));
        } else {
          ASSERT_EQ(run(args(selector)), 0);
          EXPECT_TRUE(test::legacy_atomic_output_matches(source, 0, 8, read(output_)));
          ASSERT_TRUE(std::filesystem::remove(output_));
        }
        EXPECT_EQ(read(input_), image);
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Targets, FaultLegacyAtomicCli,
                         ::testing::ValuesIn(test::kLegacyAtomicTargetCases),
                         [](const ::testing::TestParamInfo<test::LegacyAtomicTargetCase> &info) {
                           return info.param.name;
                         });

} // namespace
} // namespace rocjitsu
