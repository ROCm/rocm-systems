// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cli_test_support.h"
#include "rocjitsu/code/file_io.h"
#include "rocjitsu/code/patch/fault_injection/lds_wrong_address.h"
#include "scoped_temp.h"
#include "target_coverage_fixture.h"

#include <gtest/gtest.h>

#include <array>
#include <cerrno>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

namespace rocjitsu {
namespace {

class Gfx1251LdsCliBase : public testing::Test {
protected:
  test::ScopedTempDirectory temporary_;
  std::string executable_;
  std::filesystem::path input_;
  std::filesystem::path output_;
  std::filesystem::path log_;

  void SetUp() override {
    ASSERT_TRUE(test::mutation_cli_path(executable_));
    ASSERT_TRUE(temporary_.initialize("gfx1251-lds-cli-"));
    input_ = std::filesystem::path(temporary_.path()) / "source.hsaco";
    output_ = std::filesystem::path(temporary_.path()) / "mutant.hsaco";
    log_ = std::filesystem::path(temporary_.path()) / "cli.txt";
  }

  void write_source(const std::vector<uint8_t> &bytes) const {
    std::ofstream file(input_, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char *>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
    file.close();
    ASSERT_FALSE(file.fail());
  }

  int run(std::vector<std::string> args, std::optional<rlim_t> file_limit = std::nullopt) const {
    args.insert(args.begin(), executable_);
    std::vector<char *> argv;
    for (std::string &arg : args)
      argv.push_back(arg.data());
    argv.push_back(nullptr);
    const int log_fd = open(log_.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (log_fd < 0)
      return -1;
    const pid_t pid = fork();
    if (pid == 0) {
      if (dup2(log_fd, STDOUT_FILENO) < 0 || dup2(log_fd, STDERR_FILENO) < 0)
        _exit(126);
      close(log_fd);
      if (file_limit) {
        const rlimit core = {.rlim_cur = 0, .rlim_max = 0};
        const rlimit limit = {.rlim_cur = *file_limit, .rlim_max = *file_limit};
        if (setrlimit(RLIMIT_CORE, &core) != 0 || setrlimit(RLIMIT_FSIZE, &limit) != 0)
          _exit(126);
      }
      execv(argv[0], argv.data());
      _exit(127);
    }
    close(log_fd);
    if (pid < 0)
      return -1;
    int status = 0;
    pid_t waited = 0;
    do {
      waited = waitpid(pid, &status, 0);
    } while (waited < 0 && errno == EINTR);
    return waited == pid && WIFEXITED(status) ? WEXITSTATUS(status) : -1;
  }

  std::vector<std::string> args() const {
    return {"--input", input_.string(), "--kernel", "test_kernel", "--kind", "LdsWrongAddress"};
  }

  std::vector<uint8_t> read(const std::filesystem::path &path) const {
    const FailureOr<std::vector<char>> data = detail::read_file_bytes(path.string());
    EXPECT_TRUE(data.succeeded());
    return data.succeeded() ? std::vector<uint8_t>(data.value().begin(), data.value().end())
                            : std::vector<uint8_t>{};
  }

  std::string read_log() const {
    const std::vector<uint8_t> bytes = read(log_);
    return std::string(bytes.begin(), bytes.end());
  }

  std::vector<std::string> mutation_args() const {
    std::vector<std::string> command = args();
    command.insert(command.end(), {"--output", output_.string(), "--address-vgpr", "6"});
    return command;
  }
};

class Gfx1251LdsCli : public Gfx1251LdsCliBase,
                      public testing::WithParamInterface<test::TargetLdsCase> {};

TEST_P(Gfx1251LdsCli, ListExactOrdinalAndWholeObjectAgreeWithLibraryForEachBank) {
  const test::TargetLdsCase &row = GetParam();
  for (uint32_t bank : {0u, 1u}) {
    SCOPED_TRACE(bank);
    const std::vector<uint32_t> prefix =
        bank == 0
            ? std::vector<uint32_t>{}
            : std::vector<uint32_t>{cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 1})[0]};
    const std::array<uint32_t, 3> suffix = {0xbf800007u, row.form.words[0], row.form.words[1]};
    const uint32_t granulated = bank == 0 ? 3 : 16;
    const std::vector<uint8_t> image = test::make_target_lds_elf(row, prefix, suffix, granulated);
    write_source(image);
    const FaultMutationContext source(image);
    FaultMutationError error;
    const FailureOr<LdsInventory> inventory =
        discover_lds_wrong_address(source, "test_kernel", error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    ASSERT_EQ(inventory.value().sites().size(), 2u);
    std::vector<std::string> command = args();
    command.insert(command.end(), {"--target", "gfx1251", "--list"});
    ASSERT_EQ(run(command), 0) << read_log();
    std::string expected_list;
    for (size_t ordinal = 0; ordinal < 2; ++ordinal) {
      const LdsSite &site = inventory.value().sites()[ordinal];
      ASSERT_EQ(site.affected_owners.size(), 1u);
      const FaultMutationOwner &owner = site.affected_owners.front();
      expected_list +=
          "ordinal=" + std::to_string(ordinal) + " exact=" + std::to_string(site.offset) +
          " address_vgpr=2 address_bank=" + std::to_string(bank) +
          " target=gfx1251 form=" + std::string(row.form.form) +
          " affected_owners=1 owner=" + owner.entry + ':' + std::to_string(owner.entry_offset) +
          ':' + std::to_string(owner.descriptor_file_offset) + '\n';
    }
    EXPECT_EQ(read_log(), expected_list);
    EXPECT_FALSE(std::filesystem::exists(output_));
    for (size_t ordinal = 0; ordinal < 2; ++ordinal) {
      const FailureOr<LdsSelection> selected = select_lds_wrong_address(
          inventory.value(), {.exact = std::nullopt, .ordinal = ordinal}, error);
      ASSERT_TRUE(selected.succeeded()) << error.message;
      const FailureOr<LdsResult> mutant =
          apply_lds_wrong_address(source, selected.value(), 6, error);
      ASSERT_TRUE(mutant.succeeded()) << error.message;
      for (bool exact : {false, true}) {
        SCOPED_TRACE(exact);
        command = mutation_args();
        if (exact)
          command.insert(command.end(), {"--exact", std::to_string(selected.value().site().offset),
                                         "--ordinal", "999"});
        else
          command.insert(command.end(),
                         {"--target", "gfx1251", "--ordinal", std::to_string(ordinal)});
        ASSERT_EQ(run(command), 0) << read_log();
        const std::vector<uint8_t> output = read(output_);
        EXPECT_EQ(output, mutant.value().image);
        EXPECT_TRUE(test::target_lds_output_matches(source, row, selected.value().site().offset, 6,
                                                    output));
        EXPECT_NE(read_log().find("kind=LdsWrongAddress target=gfx1251"), std::string::npos);
        EXPECT_NE(read_log().find("logical_mutations=1"), std::string::npos);
        ASSERT_TRUE(std::filesystem::remove(output_));
      }
    }
    const uint32_t last = test::target_lds_allocation(row, granulated) - bank * 256 - 1;
    command = args();
    command.insert(command.end(),
                   {"--output", output_.string(), "--address-vgpr", std::to_string(last)});
    ASSERT_EQ(run(command), 0) << read_log();
    EXPECT_TRUE(
        test::target_lds_output_matches(source, row, prefix.size() * 4, last, read(output_)));
    ASSERT_TRUE(std::filesystem::remove(output_));
    EXPECT_EQ(read(input_), image);
  }
}

TEST_P(Gfx1251LdsCli, InvalidSelectorsRegistersAndIdentityDoNotPublishOutput) {
  const test::TargetLdsCase &row = GetParam();
  for (uint32_t bank : {0u, 1u}) {
    const std::vector<uint32_t> prefix =
        bank == 0
            ? std::vector<uint32_t>{}
            : std::vector<uint32_t>{cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 1})[0]};
    const uint32_t granulated = bank == 0 ? 3 : 16;
    const uint32_t bound = test::target_lds_allocation(row, granulated) - bank * 256;
    const std::vector<uint8_t> image = test::make_target_lds_elf(row, prefix, {}, granulated);
    write_source(image);
    const std::vector<std::vector<std::string>> invalid = {
        {},
        {"--address-vgpr", "2"},
        {"--address-vgpr", std::to_string(bound)},
        {"--address-vgpr", "256"},
        {"--address-vgpr", "4294967295"},
        {"--exact", std::to_string(prefix.size() * 4 + 4), "--ordinal", "0", "--address-vgpr", "6"},
        {"--ordinal", "1", "--address-vgpr", "6"},
        {"--target", "gfx1250", "--address-vgpr", "6"},
    };
    for (const std::vector<std::string> &extra : invalid) {
      std::vector<std::string> command = args();
      command.insert(command.end(), {"--output", output_.string()});
      command.insert(command.end(), extra.begin(), extra.end());
      const int status = run(command);
      EXPECT_GT(status, 0) << read_log();
      EXPECT_NE(status, 126);
      EXPECT_NE(status, 127);
      EXPECT_FALSE(std::filesystem::exists(output_));
    }
    EXPECT_EQ(read(input_), image);
  }
}

TEST_F(Gfx1251LdsCliBase, DynamicVgprOwnersCannotListOrPublishMutants) {
  const test::TargetLdsCase fixture = test::target_lds_cases(ROCJITSU_CODE_TARGET_GFX1251).front();
  for (bool shared : {false, true}) {
    SCOPED_TRACE(shared);
    std::vector<uint8_t> image =
        shared ? test::make_target_shared_lds_elf(fixture, {2, 2}, {}, test::kLdsDynamicVgprPrefix)
               : test::make_target_lds_elf(fixture, test::kLdsDynamicVgprPrefix, {}, 2);
    ASSERT_NO_FATAL_FAILURE(test::enable_target_lds_dynamic_vgprs(image, shared ? 1 : 0));
    write_source(image);
    const std::vector<std::string> common = {"--input",  input_.string(),
                                             "--kernel", shared ? "test_kernel0" : "test_kernel",
                                             "--kind",   "LdsWrongAddress"};
    std::vector<std::string> command = common;
    command.push_back("--list");
    ASSERT_EQ(run(command), 0) << read_log();
    EXPECT_TRUE(read_log().empty()) << read_log();
    for (bool exact : {false, true}) {
      command = common;
      command.insert(command.end(), {"--output", output_.string(), "--address-vgpr", "40"});
      if (exact)
        command.insert(command.end(), {"--exact", shared ? "512" : "12", "--ordinal", "0"});
      else
        command.insert(command.end(), {"--ordinal", "0"});
      EXPECT_GT(run(command), 0) << read_log();
      EXPECT_EQ(read_log().find("logical_mutations="), std::string::npos);
      EXPECT_FALSE(std::filesystem::exists(output_));
      if (std::filesystem::exists(output_)) {
        ASSERT_TRUE(std::filesystem::remove(output_));
      }
    }
    EXPECT_EQ(read(input_), image);
  }
}

TEST_F(Gfx1251LdsCliBase, OutputFailuresAndEmissionRollbackLeaveNoPartialFile) {
  const test::TargetLdsCase row = test::target_lds_cases(ROCJITSU_CODE_TARGET_GFX1251).front();
  const std::vector<uint8_t> image = test::make_target_lds_elf(row);
  write_source(image);
  {
    std::ofstream existing(output_, std::ios::binary);
    existing << "preserve existing output";
    existing.close();
    ASSERT_FALSE(existing.fail());
  }
  const std::vector<uint8_t> sentinel = read(output_);
  EXPECT_EQ(run(mutation_args()), 3) << read_log();
  EXPECT_EQ(read(output_), sentinel);
  ASSERT_TRUE(std::filesystem::remove(output_));
  std::vector<std::string> bad_directory = args();
  const std::filesystem::path missing =
      std::filesystem::path(temporary_.path()) / "missing" / "out";
  bad_directory.insert(bad_directory.end(), {"--output", missing.string(), "--address-vgpr", "6"});
  EXPECT_EQ(run(bad_directory), 3) << read_log();
  EXPECT_FALSE(std::filesystem::exists(missing));
  // This child can create the file but cannot write the complete ELF.
  EXPECT_EQ(run(mutation_args(), 128), 3) << read_log();
  EXPECT_FALSE(std::filesystem::exists(output_));
  EXPECT_EQ(read(input_), image);

  const std::vector<uint32_t> distant(32769, 0xbf800000u);
  const std::vector<uint8_t> too_distant = test::make_target_lds_elf(row, {}, distant);
  write_source(too_distant);
  EXPECT_GT(run(mutation_args()), 0) << read_log();
  EXPECT_FALSE(std::filesystem::exists(output_));
  EXPECT_EQ(read(input_), too_distant);
}

TEST_F(Gfx1251LdsCliBase, UnsupportedFormsAndConcreteFeatureMismatchCannotUseExactFallback) {
  const test::TargetLdsCase row = test::target_lds_cases(ROCJITSU_CODE_TARGET_GFX1251).front();
  const std::array<std::array<uint32_t, 2>, 5> excluded = {{
      test::kGfx1251ImplicitAddressLdsFixtures[0].source,
      test::kGfx1251ImplicitAddressLdsFixtures[1].source,
      {0xd8500000u, 0},
      {0xd8000000u, 0x00000302u},
      {0xdac80000u, 0x0c000302u},
  }};
  for (const std::array<uint32_t, 2> &prefix : excluded) {
    write_source(test::make_target_lds_elf(row, prefix));
    std::vector<std::string> command = mutation_args();
    command.insert(command.end(), {"--exact", "0", "--ordinal", "0"});
    EXPECT_GT(run(command), 0) << read_log();
    EXPECT_FALSE(std::filesystem::exists(output_));
    command = mutation_args();
    command.insert(command.end(), {"--exact", "8", "--ordinal", "999"});
    ASSERT_EQ(run(command), 0) << read_log();
    ASSERT_TRUE(std::filesystem::remove(output_));
  }
  const std::array<uint32_t, 2> packed = {0xcc4c4004u, 0x1a021908u};
  const std::vector<uint8_t> image = test::make_target_lds_elf(row, packed);
  write_source(image);
  ASSERT_EQ(run(mutation_args()), 0) << read_log();
  const FaultMutationContext source(image);
  EXPECT_TRUE(test::target_lds_output_matches(source, row, 8, 6, read(output_)));
  ASSERT_TRUE(std::filesystem::remove(output_));
  test::TargetLdsCase wrong = row;
  wrong.form.target = ROCJITSU_CODE_TARGET_GFX1250;
  wrong.form.machine = EF_AMDGPU_MACH_AMDGCN_GFX1250;
  write_source(test::make_target_lds_elf(wrong, packed));
  EXPECT_GT(run(mutation_args()), 0) << read_log();
  EXPECT_FALSE(std::filesystem::exists(output_));

  std::vector<uint8_t> truncated =
      test::make_amdgpu_kernel_elf({row.form.words[0]}, 0, 3, row.form.machine, 3);
  truncated[EI_DATA] = ELFDATA2LSB;
  truncated[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  write_source(truncated);
  EXPECT_GT(run(mutation_args()), 0) << read_log();
  EXPECT_FALSE(std::filesystem::exists(output_));
  ASSERT_TRUE(std::filesystem::remove(input_));
  EXPECT_GT(run(mutation_args()), 0) << read_log();
  EXPECT_FALSE(std::filesystem::exists(output_));
}

INSTANTIATE_TEST_SUITE_P(All34Forms, Gfx1251LdsCli,
                         testing::ValuesIn(test::target_lds_cases(ROCJITSU_CODE_TARGET_GFX1251)),
                         [](const testing::TestParamInfo<test::TargetLdsCase> &info) {
                           return std::string(info.param.form.form);
                         });

} // namespace
} // namespace rocjitsu
