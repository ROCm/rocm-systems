// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cdna_lds_fixture.h"
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

class CdnaLdsCli : public ::testing::TestWithParam<test::CdnaLdsTarget> {
protected:
  test::ScopedTempDirectory temporary_;
  std::filesystem::path input_, output_;
  std::string cli_;

  void SetUp() override {
    ASSERT_TRUE(test::mutation_cli_path(cli_));
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
    args.insert(args.begin(), cli_);
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

TEST_P(CdnaLdsCli, ExactOrdinalAndLibraryAgree) {
  const std::vector<uint8_t> image = test::make_cdna_lds_elf(GetParam());
  write_source(image);
  const FaultMutationContext context(image);
  FaultMutationError error;
  const FailureOr<LdsInventory> inventory =
      discover_lds_wrong_address(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  std::vector<std::string> command = args();
  command.insert(command.end(), {"--list"});
  ASSERT_EQ(run(command), 0);
  for (size_t ordinal = 0; ordinal < 2; ++ordinal) {
    const FailureOr<LdsSelection> selection = select_lds_wrong_address(
        inventory.value(), {.exact = std::nullopt, .ordinal = ordinal}, error);
    ASSERT_TRUE(selection.succeeded()) << error.message;
    const FailureOr<LdsResult> result =
        apply_lds_wrong_address(context, selection.value(), 6, error);
    ASSERT_TRUE(result.succeeded()) << error.message;
    command = args();
    command.insert(command.end(), {"--output", output_.string(), "--ordinal",
                                   std::to_string(ordinal), "--address-vgpr", "6"});
    ASSERT_EQ(run(command), 0);
    EXPECT_EQ(read(), result.value().image);
    EXPECT_TRUE(test::cdna_lds_output_matches(context, GetParam(), ordinal * 8, 6, read()));
    ASSERT_TRUE(std::filesystem::remove(output_));
    command = args();
    command.insert(command.end(),
                   {"--output", output_.string(), "--exact", std::to_string(ordinal * 8),
                    "--ordinal", "999", "--address-vgpr", "6"});
    ASSERT_EQ(run(command), 0);
    EXPECT_EQ(read(), result.value().image);
    ASSERT_TRUE(std::filesystem::remove(output_));
  }
}

TEST_P(CdnaLdsCli, InvalidRequestsDoNotPublishOutput) {
  write_source(test::make_cdna_lds_elf(GetParam()));
  for (const std::vector<std::string> &extra :
       {std::vector<std::string>{"--exact", "4", "--ordinal", "0", "--address-vgpr", "6"},
        std::vector<std::string>{}, std::vector<std::string>{"--address-vgpr", "2"},
        std::vector<std::string>{"--address-vgpr", "32"},
        std::vector<std::string>{"--address-vgpr", "256"}}) {
    std::vector<std::string> command = args();
    command.insert(command.end(), {"--output", output_.string()});
    command.insert(command.end(), extra.begin(), extra.end());
    EXPECT_NE(run(command), 0);
    EXPECT_FALSE(std::filesystem::exists(output_));
  }
}

INSTANTIATE_TEST_SUITE_P(Targets, CdnaLdsCli, ::testing::ValuesIn(test::kCdnaLdsTargets),
                         [](const ::testing::TestParamInfo<test::CdnaLdsTarget> &info) {
                           return info.param.name;
                         });

} // namespace
} // namespace rocjitsu
