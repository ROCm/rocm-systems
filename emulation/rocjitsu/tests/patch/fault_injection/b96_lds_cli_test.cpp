// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "b96_fixture.h"
#include "cli_test_support.h"
#include "rocjitsu/code/file_io.h"
#include "rocjitsu/code/patch/fault_injection/lds_wrong_address.h"
#include "scoped_temp.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace rocjitsu {
namespace {

class B96LdsCli : public ::testing::TestWithParam<test::B96Target> {
protected:
  test::ScopedTempDirectory temporary_;
  std::string executable_;
  std::filesystem::path input_, output_;

  void SetUp() override {
    ASSERT_TRUE(test::mutation_cli_path(executable_));
    ASSERT_TRUE(temporary_.initialize("lds-cli-"));
    input_ = std::filesystem::path(temporary_.path()) / "source.hsaco";
    output_ = std::filesystem::path(temporary_.path()) / "mutant.hsaco";
  }

  void write_source(const std::vector<uint8_t> &bytes) {
    std::ofstream file(input_, std::ios::binary);
    file.write(reinterpret_cast<const char *>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
    file.close();
    ASSERT_FALSE(file.fail());
  }
  int run(std::vector<std::string> args) {
    args.insert(args.begin(), executable_);
    std::vector<char *> argv;
    for (std::string &arg : args) {
      argv.push_back(arg.data());
    }
    argv.push_back(nullptr);
    const pid_t pid = fork();
    if (pid == 0) {
      execv(argv[0], argv.data());
      _exit(127);
    }
    if (pid < 0) {
      return -1;
    }
    int status = 0;
    if (waitpid(pid, &status, 0) != pid || !WIFEXITED(status)) {
      return -1;
    }
    return WEXITSTATUS(status);
  }
  std::vector<std::string> args() const {
    return {"--input", input_.string(), "--kernel", "test_kernel", "--kind", "LdsWrongAddress"};
  }
  std::vector<uint8_t> read() const {
    const FailureOr<std::vector<char>> data = detail::read_file_bytes(output_.string());
    EXPECT_TRUE(data.succeeded());
    return data.succeeded() ? std::vector<uint8_t>(data.value().begin(), data.value().end())
                            : std::vector<uint8_t>{};
  }
};

TEST_P(B96LdsCli, ExactOrdinalAndLibraryAgree) {
  for (bool wave32 : {false, true}) {
    if (wave32 && GetParam().arch != IsaFamily::Rdna4) {
      continue;
    }
    SCOPED_TRACE(wave32);
    const std::vector<uint8_t> image =
        test::make_b96_elf(GetParam(), 3, 3, false, false, {}, wave32);
    write_source(image);
    const FaultMutationContext context(image);
    FaultMutationError error;
    const FailureOr<LdsInventory> inventory =
        discover_lds_wrong_address(context, "test_kernel", error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    const std::string target_name(default_isa_target_registry().find_target(GetParam().target)->id);
    std::vector<std::string> command = args();
    command.insert(command.end(), {"--target", target_name, "--list"});
    ASSERT_EQ(run(command), 0);
    for (size_t ordinal = 0; ordinal < 3; ++ordinal) {
      const FailureOr<LdsSelection> selection = select_lds_wrong_address(
          inventory.value(), {.exact = std::nullopt, .ordinal = ordinal}, error);
      ASSERT_TRUE(selection.succeeded()) << error.message;
      const FailureOr<LdsResult> result =
          apply_lds_wrong_address(context, selection.value(), 6, error);
      ASSERT_TRUE(result.succeeded()) << error.message;
      command = args();
      command.insert(command.end(), {"--target", target_name, "--output", output_.string(),
                                     "--ordinal", std::to_string(ordinal), "--address-vgpr", "6"});
      ASSERT_EQ(run(command), 0);
      EXPECT_EQ(read(), result.value().image);
      EXPECT_TRUE(test::b96_output_matches(context, GetParam(), ordinal * 8, 6, read()));
      ASSERT_TRUE(std::filesystem::remove(output_));
      command = args();
      command.insert(command.end(),
                     {"--output", output_.string(), "--exact", std::to_string(ordinal * 8),
                      "--ordinal", "999", "--address-vgpr", "6"});
      ASSERT_EQ(run(command), 0);
      EXPECT_EQ(read(), result.value().image);
      ASSERT_TRUE(std::filesystem::remove(output_));
    }
    const uint32_t bound =
        GetParam().target == ROCJITSU_CODE_TARGET_GFX1250 ? 64u : (wave32 ? 32u : 16u);
    const FailureOr<LdsSelection> first = select_lds_wrong_address(inventory.value(), {}, error);
    ASSERT_TRUE(first.succeeded()) << error.message;
    const FailureOr<LdsResult> boundary =
        apply_lds_wrong_address(context, first.value(), bound - 1, error);
    ASSERT_TRUE(boundary.succeeded()) << error.message;
    command = args();
    command.insert(command.end(),
                   {"--output", output_.string(), "--address-vgpr", std::to_string(bound - 1)});
    ASSERT_EQ(run(command), 0);
    EXPECT_EQ(read(), boundary.value().image);
    EXPECT_TRUE(test::b96_output_matches(context, GetParam(), 0, bound - 1, read()));
    ASSERT_TRUE(std::filesystem::remove(output_));
    command = args();
    command.insert(command.end(),
                   {"--output", output_.string(), "--address-vgpr", std::to_string(bound)});
    EXPECT_NE(run(command), 0);
    EXPECT_FALSE(std::filesystem::exists(output_));
    if (GetParam().target == ROCJITSU_CODE_TARGET_GFX1250) {
      constexpr std::array<uint32_t, 1> set_bank =
          cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 1});
      const std::vector<uint8_t> banked_image =
          test::make_b96_elf(GetParam(), 16, 0, false, false, set_bank);
      write_source(banked_image);
      const FaultMutationContext banked(banked_image);
      const FailureOr<LdsInventory> sites =
          discover_lds_wrong_address(banked, "test_kernel", error);
      ASSERT_TRUE(sites.succeeded()) << error.message;
      ASSERT_EQ(sites.value().sites().size(), 3u);
      EXPECT_EQ(sites.value().sites()[0].address_bank, 1u);
      const FailureOr<LdsSelection> selected = select_lds_wrong_address(sites.value(), {}, error);
      ASSERT_TRUE(selected.succeeded());
      const FailureOr<LdsResult> expected =
          apply_lds_wrong_address(banked, selected.value(), 15, error);
      ASSERT_TRUE(expected.succeeded()) << error.message;
      command = args();
      command.insert(command.end(), {"--target", "gfx1250", "--exact", "4", "--output",
                                     output_.string(), "--address-vgpr", "15"});
      ASSERT_EQ(run(command), 0);
      EXPECT_EQ(read(), expected.value().image);
      EXPECT_TRUE(test::b96_output_matches(banked, GetParam(), 4, 15, read()));
      ASSERT_TRUE(std::filesystem::remove(output_));
      command = args();
      command.insert(command.end(),
                     {"--exact", "4", "--output", output_.string(), "--address-vgpr", "16"});
      EXPECT_NE(run(command), 0);
      EXPECT_FALSE(std::filesystem::exists(output_));
    }
  }
}

TEST_P(B96LdsCli, InvalidRequestsDoNotPublishOutput) {
  write_source(test::make_b96_elf(GetParam()));
  std::vector<std::string> wrong_target = args();
  wrong_target.insert(wrong_target.end(),
                      {"--target", "gfx1251", "--output", output_.string(), "--address-vgpr", "6"});
  EXPECT_NE(run(wrong_target), 0);
  EXPECT_FALSE(std::filesystem::exists(output_));
  for (const std::vector<std::string> &extra :
       {std::vector<std::string>{"--exact", "4", "--ordinal", "0", "--address-vgpr", "6"},
        std::vector<std::string>{}, std::vector<std::string>{"--address-vgpr", "2"},
        std::vector<std::string>{"--address-vgpr", "64"},
        std::vector<std::string>{"--address-vgpr", "256"}}) {
    std::vector<std::string> command = args();
    command.insert(command.end(), {"--output", output_.string()});
    command.insert(command.end(), extra.begin(), extra.end());
    EXPECT_NE(run(command), 0);
    EXPECT_FALSE(std::filesystem::exists(output_));
  }
}

INSTANTIATE_TEST_SUITE_P(Targets, B96LdsCli, ::testing::ValuesIn(test::kB96Targets),
                         [](const ::testing::TestParamInfo<test::B96Target> &info) {
                           return info.param.name;
                         });

} // namespace
} // namespace rocjitsu
