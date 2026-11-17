// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cli_test_support.h"
#include "rocjitsu/code/file_io.h"
#include "rocjitsu/code/patch/fault_injection/lds_wrong_address.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/opcodes.h"
#include "scoped_temp.h"
#include "target_coverage_fixture.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace rocjitsu {
namespace {

class Gfx1100LdsCliBase : public ::testing::Test {
protected:
  test::ScopedTempDirectory temporary_;
  std::string executable_;
  std::filesystem::path input_, output_, stdout_path_, stderr_path_;
  std::string stdout_, stderr_;

  void SetUp() override {
    ASSERT_TRUE(test::mutation_cli_path(executable_));
    ASSERT_TRUE(temporary_.initialize("gfx1100-lds-cli-"));
    const std::filesystem::path directory(temporary_.path());
    input_ = directory / "source.hsaco";
    output_ = directory / "mutant.hsaco";
    stdout_path_ = directory / "stdout.txt";
    stderr_path_ = directory / "stderr.txt";
  }

  ::testing::AssertionResult write_source(std::span<const uint8_t> bytes) const {
    std::ofstream file(input_, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char *>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
    file.close();
    return file ? ::testing::AssertionSuccess()
                : ::testing::AssertionFailure() << "Cannot write fixture " << input_;
  }

  std::vector<uint8_t> read_bytes(const std::filesystem::path &path) const {
    const FailureOr<std::vector<char>> data = detail::read_file_bytes(path.string());
    EXPECT_TRUE(data.succeeded()) << path;
    return data.succeeded() ? std::vector<uint8_t>(data.value().begin(), data.value().end())
                            : std::vector<uint8_t>{};
  }

  int run(std::vector<std::string> arguments, bool limit_file_size = false) {
    stdout_.clear();
    stderr_.clear();
    arguments.insert(arguments.begin(), executable_);
    std::vector<char *> argv;
    for (std::string &argument : arguments)
      argv.push_back(argument.data());
    argv.push_back(nullptr);
    const pid_t child = fork();
    if (child == 0) {
      const int output_fd = open(stdout_path_.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
      const int error_fd = open(stderr_path_.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
      if (output_fd < 0 || error_fd < 0 || dup2(output_fd, STDOUT_FILENO) < 0 ||
          dup2(error_fd, STDERR_FILENO) < 0)
        _exit(126);
      close(output_fd);
      close(error_fd);
      if (limit_file_size) {
        // Exercise a newly created output whose buffered write/close fails;
        // the limit and signal disposition affect only this child process.
        const rlimit limit = {.rlim_cur = 64, .rlim_max = 64};
        if (std::signal(SIGXFSZ, SIG_IGN) == SIG_ERR || setrlimit(RLIMIT_FSIZE, &limit) != 0)
          _exit(126);
      }
      execv(argv[0], argv.data());
      _exit(127);
    }
    if (child < 0)
      return -1;
    int status = 0;
    pid_t waited = 0;
    do {
      waited = waitpid(child, &status, 0);
    } while (waited < 0 && errno == EINTR);
    const std::vector<uint8_t> output = read_bytes(stdout_path_);
    const std::vector<uint8_t> errors = read_bytes(stderr_path_);
    stdout_.assign(output.begin(), output.end());
    stderr_.assign(errors.begin(), errors.end());
    return waited == child && WIFEXITED(status) ? WEXITSTATUS(status) : -1;
  }

  std::vector<std::string> args() const {
    return {"--input", input_.string(), "--kernel", "test_kernel", "--kind", "LdsWrongAddress"};
  }

  std::vector<std::string> mutation_args(const std::filesystem::path &destination) const {
    std::vector<std::string> command = args();
    command.insert(command.end(), {"--output", destination.string(), "--address-vgpr", "6"});
    return command;
  }
};

class Gfx1100LdsCli : public Gfx1100LdsCliBase,
                      public ::testing::WithParamInterface<test::TargetLdsCase> {};
class Gfx1100LdsCliControls : public Gfx1100LdsCliBase {};

TEST_P(Gfx1100LdsCli, ListOrdinalAndExactMatchLibraryAndIndependentWholeImageOracle) {
  const test::TargetLdsCase &fixture = GetParam();
  constexpr std::array<uint32_t, 1> prefix = {0xbf800001u};
  const std::array<uint32_t, 3> suffix = {0xbf800003u, fixture.form.words[0],
                                          fixture.form.words[1]};
  const std::vector<uint8_t> image = test::make_target_lds_elf(fixture, prefix, suffix);
  ASSERT_TRUE(write_source(image));
  const FaultMutationContext source(image);
  FaultMutationError error;
  const FailureOr<LdsInventory> inventory =
      discover_lds_wrong_address(source, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().sites().size(), 2u);
  std::ostringstream expected_list;
  for (size_t ordinal = 0; ordinal < inventory.value().sites().size(); ++ordinal) {
    const LdsSite &site = inventory.value().sites()[ordinal];
    EXPECT_EQ(site.offset, ordinal == 0 ? 4u : 16u);
    EXPECT_EQ(site.form, fixture.form.form);
    EXPECT_EQ(site.address_vgpr, 2u);
    EXPECT_EQ(site.address_bank, 0u);
    ASSERT_EQ(site.affected_owners.size(), 1u);
    expected_list << "ordinal=" << ordinal << " exact=" << site.offset
                  << " address_vgpr=2 address_bank=0 target=gfx1100 form=" << fixture.form.form
                  << " affected_owners=1";
    for (const FaultMutationOwner &owner : site.affected_owners)
      expected_list << " owner=" << owner.entry << ':' << owner.entry_offset << ':'
                    << owner.descriptor_file_offset;
    expected_list << '\n';
  }
  std::vector<std::string> command = args();
  command.insert(command.end(), {"--target", "gfx1100", "--list"});
  ASSERT_EQ(run(command), 0) << stderr_;
  EXPECT_EQ(stdout_, expected_list.str());
  EXPECT_FALSE(std::filesystem::exists(output_));
  const uint32_t allocation = test::target_lds_allocation(fixture);
  for (size_t ordinal = 0; ordinal < 2; ++ordinal) {
    const uint64_t offset = ordinal == 0 ? 4u : 16u;
    const FailureOr<LdsSelection> selection = select_lds_wrong_address(
        inventory.value(), {.exact = std::nullopt, .ordinal = ordinal}, error);
    ASSERT_TRUE(selection.succeeded()) << error.message;
    for (uint32_t address : {6u, allocation - 1}) {
      const FailureOr<LdsResult> expected =
          apply_lds_wrong_address(source, selection.value(), address, error);
      ASSERT_TRUE(expected.succeeded()) << error.message;
      for (bool exact : {false, true}) {
        command = args();
        command.insert(command.end(),
                       {"--output", output_.string(), "--address-vgpr", std::to_string(address)});
        if (exact)
          command.insert(command.end(), {"--exact", std::to_string(offset), "--ordinal", "999"});
        else
          command.insert(command.end(),
                         {"--target", "gfx1100", "--ordinal", std::to_string(ordinal)});
        ASSERT_EQ(run(command), 0) << stderr_;
        const std::vector<uint8_t> emitted = read_bytes(output_);
        EXPECT_EQ(emitted, expected.value().image);
        EXPECT_TRUE(test::target_lds_output_matches(source, fixture, offset, address, emitted));
        EXPECT_NE(stdout_.find("kind=LdsWrongAddress target=gfx1100 exact=" +
                               std::to_string(offset) + " address_vgpr=" + std::to_string(address)),
                  std::string::npos);
        EXPECT_NE(stdout_.find("logical_mutations=1"), std::string::npos);
        EXPECT_NE(stdout_.find("affected_owners=1"), std::string::npos);
        ASSERT_TRUE(std::filesystem::remove(output_));
      }
    }
  }
  EXPECT_EQ(read_bytes(input_), image);
  EXPECT_TRUE(std::ranges::equal(source.image(), image));
}

TEST_P(Gfx1100LdsCli, InvalidRequestsNeverPublishMutantsOrFallbackFromInvalidExact) {
  const test::TargetLdsCase &fixture = GetParam();
  const std::vector<uint8_t> image = test::make_target_lds_elf(fixture);
  ASSERT_TRUE(write_source(image));
  const uint32_t allocation = test::target_lds_allocation(fixture);
  for (const std::vector<std::string> &extra : {
           std::vector<std::string>{},
           std::vector<std::string>{"--address-vgpr", "2"},
           std::vector<std::string>{"--address-vgpr", std::to_string(allocation)},
           std::vector<std::string>{"--address-vgpr", "255"},
           std::vector<std::string>{"--address-vgpr", "256"},
           std::vector<std::string>{"--address-vgpr", "4294967295"},
           std::vector<std::string>{"--address-vgpr", "-1"},
           std::vector<std::string>{"--address-vgpr", "6", "--ordinal", "1"},
           std::vector<std::string>{"--address-vgpr", "6", "--exact", "4", "--ordinal", "0"},
           std::vector<std::string>{"--address-vgpr", "6", "--target", "gfx1201"},
       }) {
    std::vector<std::string> command = args();
    command.insert(command.end(), {"--output", output_.string()});
    command.insert(command.end(), extra.begin(), extra.end());
    EXPECT_NE(run(command), 0);
    EXPECT_FALSE(std::filesystem::exists(output_));
    EXPECT_EQ(stdout_.find("logical_mutations="), std::string::npos);
    EXPECT_FALSE(stderr_.empty());
    EXPECT_EQ(read_bytes(input_), image);
  }
}

TEST_P(Gfx1100LdsCli, GdsBitSeventeenIsExcludedWhileReservedBitSixteenRemainsLossless) {
  const test::TargetLdsCase &fixture = GetParam();
  for (uint32_t bits : {0x00020000u, 0x00030000u}) {
    test::TargetLdsCase gds = fixture;
    gds.form.words[0] |= bits;
    const std::vector<uint8_t> image = test::make_target_lds_elf(gds);
    ASSERT_TRUE(write_source(image));
    std::vector<std::string> command = args();
    command.push_back("--list");
    ASSERT_EQ(run(command), 0) << stderr_;
    EXPECT_TRUE(stdout_.empty());
    EXPECT_NE(run(mutation_args(output_)), 0);
    EXPECT_FALSE(std::filesystem::exists(output_));
    EXPECT_EQ(stdout_.find("logical_mutations="), std::string::npos);
    EXPECT_EQ(read_bytes(input_), image);
  }
  test::TargetLdsCase reserved = fixture;
  reserved.form.words[0] |= 0x00010000u;
  reserved.expected[0] |= 0x00010000u;
  const std::vector<uint8_t> image = test::make_target_lds_elf(reserved);
  ASSERT_TRUE(write_source(image));
  const FaultMutationContext source(image);
  FaultMutationError error;
  const FailureOr<LdsInventory> inventory =
      discover_lds_wrong_address(source, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().sites().size(), 1u);
  const FailureOr<LdsSelection> selection = select_lds_wrong_address(inventory.value(), {}, error);
  ASSERT_TRUE(selection.succeeded()) << error.message;
  const FailureOr<LdsResult> expected =
      apply_lds_wrong_address(source, selection.value(), 6, error);
  ASSERT_TRUE(expected.succeeded()) << error.message;
  std::vector<std::string> command = mutation_args(output_);
  command.insert(command.end(), {"--exact", "0", "--ordinal", "999"});
  ASSERT_EQ(run(command), 0) << stderr_;
  const std::vector<uint8_t> emitted = read_bytes(output_);
  EXPECT_EQ(emitted, expected.value().image);
  EXPECT_TRUE(test::target_lds_output_matches(source, reserved, 0, 6, emitted));
  EXPECT_EQ(read_bytes(input_), image);
}

TEST_F(Gfx1100LdsCliControls, ExistingAliasesAndFailedWritesPreserveInputsAndRemovePartialOutput) {
  const std::vector<test::TargetLdsCase> cases =
      test::target_lds_cases(ROCJITSU_CODE_TARGET_GFX1100);
  for (uint32_t wave : {32u, 64u}) {
    SCOPED_TRACE(wave);
    const std::vector<test::TargetLdsCase>::const_iterator fixture =
        std::ranges::find(cases, wave, &test::TargetLdsCase::wave);
    ASSERT_NE(fixture, cases.end());
    const std::vector<uint8_t> image = test::make_target_lds_elf(*fixture);
    ASSERT_TRUE(write_source(image));
    const std::vector<uint8_t> sentinel = {'k', 'e', 'e', 'p', '\n'};
    {
      std::ofstream file(output_, std::ios::binary);
      file.write(reinterpret_cast<const char *>(sentinel.data()),
                 static_cast<std::streamsize>(sentinel.size()));
      file.close();
      ASSERT_TRUE(file);
    }
    EXPECT_EQ(run(mutation_args(output_)), 3) << stderr_;
    EXPECT_EQ(read_bytes(output_), sentinel);
    EXPECT_EQ(stdout_.find("logical_mutations="), std::string::npos);
    ASSERT_TRUE(std::filesystem::remove(output_));
    EXPECT_EQ(run(mutation_args(input_)), 3) << stderr_;
    EXPECT_EQ(read_bytes(input_), image);
    std::filesystem::create_symlink(input_, output_);
    EXPECT_EQ(run(mutation_args(output_)), 3) << stderr_;
    EXPECT_TRUE(std::filesystem::is_symlink(output_));
    EXPECT_EQ(read_bytes(input_), image);
    ASSERT_TRUE(std::filesystem::remove(output_));
    const std::filesystem::path missing_parent =
        std::filesystem::path(temporary_.path()) / "missing" / "mutant.hsaco";
    EXPECT_EQ(run(mutation_args(missing_parent)), 3) << stderr_;
    EXPECT_FALSE(std::filesystem::exists(missing_parent));
    ASSERT_TRUE(std::filesystem::create_directory(output_));
    EXPECT_EQ(run(mutation_args(output_)), 3) << stderr_;
    EXPECT_TRUE(std::filesystem::is_directory(output_));
    ASSERT_TRUE(std::filesystem::remove(output_));
    EXPECT_EQ(run(mutation_args(output_), true), 3) << stderr_;
    EXPECT_FALSE(std::filesystem::exists(output_));
    EXPECT_EQ(stdout_.find("logical_mutations="), std::string::npos);
    EXPECT_EQ(read_bytes(input_), image);
  }
}

TEST_F(Gfx1100LdsCliControls, NonAddressProfilesAndMissingConcreteFormsCannotPublishMutants) {
  constexpr std::array<uint32_t, 8> prefix = {0xd8500000u, 0u,          0xdac00000u, 0x00000800u,
                                              0xdac40000u, 0x0c000000u, 0xd8000004u, 0x00000802u};
  const std::vector<test::TargetLdsCase> cases =
      test::target_lds_cases(ROCJITSU_CODE_TARGET_GFX1100);
  for (uint32_t wave : {32u, 64u}) {
    const std::vector<test::TargetLdsCase>::const_iterator fixture =
        std::ranges::find(cases, wave, &test::TargetLdsCase::wave);
    ASSERT_NE(fixture, cases.end());
    const std::vector<uint8_t> image = test::make_target_lds_elf(*fixture, prefix);
    ASSERT_TRUE(write_source(image));
    std::vector<std::string> command = args();
    command.push_back("--list");
    ASSERT_EQ(run(command), 0) << stderr_;
    EXPECT_NE(stdout_.find("ordinal=0 exact=32 address_vgpr=2"), std::string::npos);
    EXPECT_EQ(std::ranges::count(stdout_, '\n'), 1);
    for (uint64_t offset : {0u, 8u, 16u, 24u}) {
      command = mutation_args(output_);
      command.insert(command.end(), {"--exact", std::to_string(offset), "--ordinal", "0"});
      EXPECT_NE(run(command), 0);
      EXPECT_FALSE(std::filesystem::exists(output_));
    }
    for (uint32_t opcode_word : {0xdb8c0004u, 0xdbe80004u, 0xdbec0004u, 0xdbf00004u, 0xdbf40004u}) {
      test::TargetLdsCase unsupported = *fixture;
      unsupported.form.words = {opcode_word, 0x0c000002u};
      const std::vector<uint8_t> invalid = test::make_target_lds_elf(unsupported);
      ASSERT_TRUE(write_source(invalid));
      EXPECT_NE(run(mutation_args(output_)), 0);
      EXPECT_FALSE(std::filesystem::exists(output_));
      EXPECT_EQ(stdout_.find("logical_mutations="), std::string::npos);
      EXPECT_EQ(read_bytes(input_), invalid);
    }
    std::vector<uint8_t> truncated =
        test::make_amdgpu_kernel_elf({fixture->form.words[0]}, 0, 3, fixture->form.machine, 3, 3,
                                     false, false, false, false, fixture->wave == 32);
    truncated[EI_DATA] = ELFDATA2LSB;
    truncated[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
    ASSERT_TRUE(write_source(truncated));
    EXPECT_NE(run(mutation_args(output_)), 0);
    EXPECT_FALSE(std::filesystem::exists(output_));
    EXPECT_EQ(stdout_.find("logical_mutations="), std::string::npos);
    EXPECT_EQ(read_bytes(input_), truncated);
  }
}

TEST_F(Gfx1100LdsCliControls, PcRelativeAndRangeEmissionFailuresDoNotPublishPartialOutput) {
  const std::vector<test::TargetLdsCase> cases =
      test::target_lds_cases(ROCJITSU_CODE_TARGET_GFX1100);
  constexpr std::array<uint32_t, 1> get_pc = rdna3::build_sop1(rdna3::kSGetPcB64Sop1, {.sdst = 0});
  const std::vector<uint32_t> distant_suffix(32769, 0xbf800000u);
  for (uint32_t wave : {32u, 64u}) {
    const std::vector<test::TargetLdsCase>::const_iterator fixture =
        std::ranges::find(cases, wave, &test::TargetLdsCase::wave);
    ASSERT_NE(fixture, cases.end());
    for (bool pc_relative : {true, false}) {
      const std::vector<uint8_t> image =
          pc_relative ? test::make_target_lds_elf(*fixture, get_pc)
                      : test::make_target_lds_elf(*fixture, {}, distant_suffix);
      ASSERT_TRUE(write_source(image));
      std::vector<std::string> command = args();
      command.push_back("--list");
      ASSERT_EQ(run(command), 0) << stderr_;
      EXPECT_NE(stdout_.find("ordinal=0 exact="), std::string::npos);
      EXPECT_NE(run(mutation_args(output_)), 0);
      EXPECT_FALSE(std::filesystem::exists(output_));
      EXPECT_EQ(stdout_.find("logical_mutations="), std::string::npos);
      if (pc_relative) {
        EXPECT_NE(stderr_.find("retained PC-relative"), std::string::npos);
      }
      EXPECT_EQ(read_bytes(input_), image);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(ConcreteFormsAndWaves, Gfx1100LdsCli,
                         ::testing::ValuesIn(test::target_lds_cases(ROCJITSU_CODE_TARGET_GFX1100)),
                         [](const ::testing::TestParamInfo<test::TargetLdsCase> &info) {
                           return std::string(info.param.form.form) + "Wave" +
                                  std::to_string(info.param.wave);
                         });

} // namespace
} // namespace rocjitsu
