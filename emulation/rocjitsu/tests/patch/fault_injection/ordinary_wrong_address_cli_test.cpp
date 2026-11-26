// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cli_test_support.h"
#include "ordinary_wrong_address_oracle.h"
#include "rocjitsu/code/file_io.h"
#include "rocjitsu/code/patch/fault_injection/ordinary_wrong_address.h"
#include "scoped_temp.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <sys/wait.h>
#include <system_error>
#include <unistd.h>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace {

class FaultOrdinaryCliSupport : public ::testing::Test {
protected:
  test::ScopedTempDirectory temporary_;
  std::filesystem::path root_, input_, output_;
  std::string cli_, stdout_, stderr_;

  void SetUp() override {
    ASSERT_TRUE(test::mutation_cli_path(cli_));
    ASSERT_TRUE(temporary_.initialize("ordinary-address-cli-"));
    root_ = temporary_.path();
    input_ = root_ / "source.hsaco";
    output_ = root_ / "mutant.hsaco";
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

  int run(std::vector<std::string> arguments) {
    arguments.insert(arguments.begin(), cli_);
    std::vector<char *> pointers;
    for (std::string &argument : arguments)
      pointers.push_back(argument.data());
    pointers.push_back(nullptr);
    const int out = open((root_ / "stdout").c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    const int err = open((root_ / "stderr").c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (out < 0 || err < 0) {
      if (out >= 0)
        close(out);
      if (err >= 0)
        close(err);
      return -1;
    }
    const pid_t child = fork();
    if (child == 0) {
      if (dup2(out, STDOUT_FILENO) < 0 || dup2(err, STDERR_FILENO) < 0)
        _exit(126);
      close(out);
      close(err);
      execv(pointers.front(), pointers.data());
      _exit(127);
    }
    close(out);
    close(err);
    if (child < 0)
      return -1;
    int status = 0;
    if (waitpid(child, &status, 0) != child || !WIFEXITED(status))
      return -1;
    const std::vector<uint8_t> out_bytes = read(root_ / "stdout");
    const std::vector<uint8_t> err_bytes = read(root_ / "stderr");
    stdout_.assign(out_bytes.begin(), out_bytes.end());
    stderr_.assign(err_bytes.begin(), err_bytes.end());
    return WEXITSTATUS(status);
  }

  std::vector<std::string> args(std::string_view kernel = "test_kernel") const {
    return {"--input",           input_.string(), "--kernel",
            std::string(kernel), "--kind",        "OrdinaryWrongAddress"};
  }

  std::vector<std::string> mutation_args(std::vector<std::string> selector = {},
                                         std::string_view kernel = "test_kernel") const {
    std::vector<std::string> command = args(kernel);
    command.insert(command.end(), {"--output", output_.string(), "--address-delta", "8"});
    command.insert(command.end(), selector.begin(), selector.end());
    return command;
  }

  void expect_no_partial_output() const {
    EXPECT_FALSE(std::filesystem::exists(output_));
    for (const std::filesystem::directory_entry &item :
         std::filesystem::directory_iterator(root_)) {
      const std::string name = item.path().filename().string();
      EXPECT_TRUE(name == "source.hsaco" || name == "stdout" || name == "stderr") << name;
    }
  }
};

class FaultOrdinaryWrongAddressCli
    : public FaultOrdinaryCliSupport,
      public ::testing::WithParamInterface<test::OrdinaryTargetCase> {};

TEST_P(FaultOrdinaryWrongAddressCli,
       ThreeBoundedPathsExactOrdinalLibraryAndIndependentOracleAgree) {
  const test::OrdinaryTargetCase &fixture = GetParam();
  for (bool zero_sized : {false, true}) {
    SCOPED_TRACE(zero_sized);
    const IsaTargetBinding *description = default_isa_target_registry().find_target(fixture.target);
    ASSERT_NE(description, nullptr);
    for (OrdinaryAcquirePath path :
         {OrdinaryAcquirePath::SameBlock, OrdinaryAcquirePath::ExactFallthroughJoin,
          OrdinaryAcquirePath::ExactSelfLoopExit}) {
      SCOPED_TRACE(static_cast<int>(path));
      std::vector<uint8_t> image =
          path == OrdinaryAcquirePath::SameBlock
              ? test::make_modern_ordinary_address_elf(fixture, 3, -4, true)
          : path == OrdinaryAcquirePath::ExactFallthroughJoin
              ? test::make_ordinary_fallthrough_elf(fixture)
              : test::make_ordinary_poll_elf(fixture);
      if (zero_sized) {
        ASSERT_TRUE(test::zero_ordinary_function_size(image, 0));
      }
      ASSERT_NO_FATAL_FAILURE(write(input_, image));
      const FaultMutationContext source(image);
      FaultMutationError error;
      const FailureOr<OrdinaryAcquireInventory> inventory =
          discover_ordinary_acquires(source, "test_kernel", error);
      ASSERT_TRUE(inventory.succeeded()) << error.message;
      ASSERT_EQ(inventory.value().sites().size(), 1u);
      const OrdinaryAcquireSite &site = inventory.value().sites().front();
      EXPECT_EQ(site.path, path);
      const FailureOr<OrdinaryAcquireSelection> selected =
          select_ordinary_acquire(inventory.value(), {}, error);
      ASSERT_TRUE(selected.succeeded()) << error.message;
      const FailureOr<OrdinaryWrongAddressResult> expected =
          apply_ordinary_wrong_address(source, selected.value(), 8, error);
      ASSERT_TRUE(expected.succeeded()) << error.message;
      std::vector<std::string> list = args();
      list.push_back("--list");
      ASSERT_EQ(run(list), 0) << stderr_;
      EXPECT_NE(stdout_.find("ordinal=0 exact=" + std::to_string(site.offset)), std::string::npos);
      EXPECT_NE(stdout_.find("form=global_load_b32"), std::string::npos);
      EXPECT_NE(stdout_.find("affected_owners=1"), std::string::npos);
      for (const std::vector<std::string> &selector :
           {std::vector<std::string>{"--ordinal", "0"},
            std::vector<std::string>{"--exact", std::to_string(site.offset), "--ordinal", "999"}}) {
        std::vector<std::string> command = mutation_args(selector);
        command.insert(command.end(), {"--target", std::string(description->id)});
        ASSERT_EQ(run(command), 0) << stderr_;
        const std::vector<uint8_t> actual = read(output_);
        EXPECT_EQ(actual, expected.value().image);
        EXPECT_TRUE(test::ordinary_wrong_address_output_matches(source, site.offset, 8, actual));
        EXPECT_NE(stdout_.find("kind=OrdinaryWrongAddress"), std::string::npos);
        EXPECT_NE(stdout_.find("exact=" + std::to_string(site.offset)), std::string::npos);
        EXPECT_NE(stdout_.find("address_delta=8"), std::string::npos);
        EXPECT_EQ(read(input_), image);
        ASSERT_TRUE(std::filesystem::remove(output_));
      }
    }
  }
}

TEST_P(FaultOrdinaryWrongAddressCli, QualifiedOrdinalIgnoresRawLoadAndInvalidExactNeverFallsBack) {
  const test::OrdinaryTargetCase &fixture = GetParam();
  std::vector<uint32_t> words = test::ordinary_acquire_words(0);
  for (int32_t offset : {4, -4}) {
    const std::vector<uint32_t> acquire = test::ordinary_acquire_words(2, offset);
    words.insert(words.end(), acquire.begin(), acquire.end());
  }
  const std::vector<uint8_t> image = test::make_ordinary_words_elf(fixture, std::move(words));
  ASSERT_NO_FATAL_FAILURE(write(input_, image));
  const FaultMutationContext source(image);
  FaultMutationError error;
  const FailureOr<OrdinaryAcquireInventory> inventory =
      discover_ordinary_acquires(source, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().sites().size(), 2u);
  const FailureOr<OrdinaryAcquireSelection> selected =
      select_ordinary_acquire(inventory.value(), {.exact = std::nullopt, .ordinal = 1}, error);
  ASSERT_TRUE(selected.succeeded()) << error.message;
  EXPECT_EQ(selected.value().site().offset, 56u);
  const FailureOr<OrdinaryWrongAddressResult> expected =
      apply_ordinary_wrong_address(source, selected.value(), 8, error);
  ASSERT_TRUE(expected.succeeded()) << error.message;
  std::vector<std::string> list = args();
  list.push_back("--list");
  ASSERT_EQ(run(list), 0) << stderr_;
  EXPECT_NE(stdout_.find("ordinal=0 exact=28"), std::string::npos);
  EXPECT_NE(stdout_.find("ordinal=1 exact=56"), std::string::npos);
  EXPECT_EQ(stdout_.find("exact=0 "), std::string::npos);
  EXPECT_EQ(stdout_.find("ordinal=2 "), std::string::npos);
  for (const std::vector<std::string> &selector :
       {std::vector<std::string>{"--ordinal", "1"},
        std::vector<std::string>{"--exact", "0x38", "--ordinal", "999"}}) {
    ASSERT_EQ(run(mutation_args(selector)), 0) << stderr_;
    const std::vector<uint8_t> actual = read(output_);
    EXPECT_EQ(actual, expected.value().image);
    EXPECT_TRUE(test::ordinary_wrong_address_output_matches(source, 56, 8, actual));
    ASSERT_TRUE(std::filesystem::remove(output_));
  }
  for (const std::vector<std::string> &selector :
       {std::vector<std::string>{"--exact", "0", "--ordinal", "0"},
        std::vector<std::string>{"--exact", "4", "--ordinal", "1"},
        std::vector<std::string>{"--ordinal", "2"}}) {
    EXPECT_NE(run(mutation_args(selector)), 0);
    EXPECT_FALSE(stderr_.empty());
    ASSERT_NO_FATAL_FAILURE(expect_no_partial_output());
    EXPECT_EQ(read(input_), image);
  }
}

TEST_P(FaultOrdinaryWrongAddressCli, InvalidRequestsAndUnalignedOrOverflowingSourcesDoNotPublish) {
  const test::OrdinaryTargetCase &fixture = GetParam();
  const std::vector<uint8_t> image = test::make_ordinary_elf(fixture);
  ASSERT_NO_FATAL_FAILURE(write(input_, image));
  for (const std::vector<std::string> &extra :
       {std::vector<std::string>{}, std::vector<std::string>{"--address-delta", "0"},
        std::vector<std::string>{"--address-delta", "2"},
        std::vector<std::string>{"--address-delta", "-4"},
        std::vector<std::string>{"--address-delta", "8388608"},
        std::vector<std::string>{"--address-delta", "4294967296"},
        std::vector<std::string>{"--address-delta", "8", "--address-delta", "4"},
        std::vector<std::string>{"--address-delta", "8", "--address-vgpr", "6"},
        std::vector<std::string>{"--address-delta", "8", "--target", "gfx942"},
        std::vector<std::string>{"--address-delta", "8", "--target", "not-a-target"},
        std::vector<std::string>{"--address-delta", "8", "--ordinal", "-1"}}) {
    std::vector<std::string> command = args();
    command.insert(command.end(), {"--output", output_.string()});
    command.insert(command.end(), extra.begin(), extra.end());
    EXPECT_NE(run(command), 0);
    EXPECT_FALSE(stderr_.empty());
    ASSERT_NO_FATAL_FAILURE(expect_no_partial_output());
    EXPECT_EQ(read(input_), image);
  }
  for (const char *kind : {"DropBarrier", "LdsWrongAddress", "AtomicWrongAddress", "UnknownKind"}) {
    const std::vector<std::string> command = {
        "--input", input_.string(), "--kernel",       "test_kernel",     "--kind",
        kind,      "--output",      output_.string(), "--address-delta", "8"};
    EXPECT_NE(run(command), 0);
    ASSERT_NO_FATAL_FAILURE(expect_no_partial_output());
    EXPECT_EQ(read(input_), image);
  }
  std::vector<std::string> invalid_list = args();
  invalid_list.insert(invalid_list.end(), {"--list", "--address-delta", "8"});
  EXPECT_NE(run(invalid_list), 0);
  ASSERT_NO_FATAL_FAILURE(expect_no_partial_output());
  for (int32_t offset : {-2, 2, 0x7ffffc}) {
    SCOPED_TRACE(offset);
    const std::vector<uint8_t> invalid = test::make_ordinary_elf(fixture, 2, offset);
    ASSERT_NO_FATAL_FAILURE(write(input_, invalid));
    EXPECT_NE(run(mutation_args()), 0);
    EXPECT_NE(stderr_.find("InvalidParameters"), std::string::npos);
    ASSERT_NO_FATAL_FAILURE(expect_no_partial_output());
    EXPECT_EQ(read(input_), invalid);
  }
}

TEST_P(FaultOrdinaryWrongAddressCli, ExistingOutputsAndInputAliasesRetainTheirOriginalContents) {
  const test::OrdinaryTargetCase &fixture = GetParam();
  const std::vector<uint8_t> image = test::make_ordinary_elf(fixture);
  ASSERT_NO_FATAL_FAILURE(write(input_, image));
  constexpr std::array<uint8_t, 4> sentinel = {5, 6, 7, 8};
  ASSERT_NO_FATAL_FAILURE(write(output_, sentinel));
  EXPECT_NE(run(mutation_args()), 0);
  EXPECT_EQ(read(output_), std::vector<uint8_t>(sentinel.begin(), sentinel.end()));
  EXPECT_EQ(read(input_), image);
  ASSERT_TRUE(std::filesystem::remove(output_));
  std::vector<std::string> inplace = args();
  inplace.insert(inplace.end(), {"--output", input_.string(), "--address-delta", "8"});
  EXPECT_NE(run(inplace), 0);
  EXPECT_EQ(read(input_), image);
  std::error_code error;
  std::filesystem::create_symlink(input_, output_, error);
  ASSERT_FALSE(error) << error.message();
  EXPECT_NE(run(mutation_args()), 0);
  EXPECT_TRUE(std::filesystem::is_symlink(output_));
  EXPECT_EQ(read(input_), image);
  ASSERT_TRUE(std::filesystem::remove(output_));
  std::filesystem::create_hard_link(input_, output_, error);
  ASSERT_FALSE(error) << error.message();
  EXPECT_NE(run(mutation_args()), 0);
  EXPECT_EQ(read(output_), image);
  EXPECT_EQ(read(input_), image);
}

TEST_P(FaultOrdinaryWrongAddressCli, SharedHelperListsAndMutatesAllRealOwnersWithoutOwnerFallback) {
  const test::OrdinaryTargetCase &fixture = GetParam();
  for (bool zero_sized : {false, true}) {
    SCOPED_TRACE(zero_sized);
    std::vector<uint8_t> image = test::make_shared_ordinary_elf(fixture);
    if (zero_sized) {
      ASSERT_TRUE(test::zero_ordinary_function_size(image, 512));
    }
    ASSERT_NO_FATAL_FAILURE(write(input_, image));
    const FaultMutationContext source(image);
    for (const char *owner : {"test_kernel0", "test_kernel1"}) {
      FaultMutationError error;
      const FailureOr<OrdinaryAcquireInventory> inventory =
          discover_ordinary_acquires(source, owner, error);
      ASSERT_TRUE(inventory.succeeded()) << error.message;
      const FailureOr<OrdinaryAcquireSelection> selected =
          select_ordinary_acquire(inventory.value(), {}, error);
      ASSERT_TRUE(selected.succeeded()) << error.message;
      EXPECT_EQ(selected.value().site().offset, 512u);
      ASSERT_EQ(selected.value().site().affected_owners.size(), 2u);
      const FailureOr<OrdinaryWrongAddressResult> expected =
          apply_ordinary_wrong_address(source, selected.value(), 8, error);
      ASSERT_TRUE(expected.succeeded()) << error.message;
      std::vector<std::string> list = args(owner);
      list.push_back("--list");
      ASSERT_EQ(run(list), 0) << stderr_;
      EXPECT_NE(stdout_.find("ordinal=0 exact=512"), std::string::npos);
      EXPECT_NE(stdout_.find("affected_owners=2"), std::string::npos);
      EXPECT_NE(stdout_.find("test_kernel0"), std::string::npos);
      EXPECT_NE(stdout_.find("test_kernel1"), std::string::npos);
      ASSERT_EQ(run(mutation_args({"--exact", "512", "--ordinal", "999"}, owner)), 0) << stderr_;
      const std::vector<uint8_t> actual = read(output_);
      EXPECT_EQ(actual, expected.value().image);
      EXPECT_TRUE(test::ordinary_wrong_address_output_matches(source, 512, 8, actual));
      EXPECT_NE(stdout_.find("affected_owners=2"), std::string::npos);
      EXPECT_EQ(read(input_), image);
      ASSERT_TRUE(std::filesystem::remove(output_));
    }
    EXPECT_NE(run(mutation_args({"--exact", "512", "--ordinal", "0"}, "test_kernel")), 0);
    ASSERT_NO_FATAL_FAILURE(expect_no_partial_output());
    const std::vector<uint8_t> unshared = test::make_shared_ordinary_elf(fixture, false);
    ASSERT_NO_FATAL_FAILURE(write(input_, unshared));
    EXPECT_NE(run(mutation_args({"--exact", "512", "--ordinal", "0"}, "test_kernel1")), 0);
    ASSERT_NO_FATAL_FAILURE(expect_no_partial_output());
    EXPECT_EQ(read(input_), unshared);
  }
}

TEST_P(FaultOrdinaryWrongAddressCli, EmissionFailureDoesNotPublishPartialOutput) {
  const test::OrdinaryTargetCase &fixture = GetParam();
  std::vector<uint32_t> words = test::ordinary_acquire_words();
  words.insert(words.end(), 33000, build_s_nop(0, fixture.arch).value());
  const std::vector<uint8_t> image = test::make_ordinary_words_elf(fixture, std::move(words));
  ASSERT_NO_FATAL_FAILURE(write(input_, image));
  EXPECT_NE(run(mutation_args()), 0);
  EXPECT_NE(stderr_.find("ValidationFailure"), std::string::npos);
  ASSERT_NO_FATAL_FAILURE(expect_no_partial_output());
  EXPECT_EQ(read(input_), image);
}

TEST_F(FaultOrdinaryCliSupport,
       Rdna4AcquireMutationRetainsQualifiedProofAndRefusesMalformedShapes) {
  for (const test::OrdinaryTargetCase &fixture : test::kOrdinaryTargetCases) {
    if (fixture.arch != IsaFamily::Rdna4)
      continue;
    SCOPED_TRACE(fixture.name);
    const std::vector<uint8_t> image = test::make_ordinary_elf(fixture);
    ASSERT_NO_FATAL_FAILURE(write(input_, image));
    const FaultMutationContext source(image);
    FaultMutationError error;
    const FailureOr<OrdinaryAcquireInventory> inventory =
        discover_ordinary_acquires(source, "test_kernel", error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    ASSERT_EQ(inventory.value().sites().size(), 1u);
    const FailureOr<OrdinaryAcquireSelection> selected =
        select_ordinary_acquire(inventory.value(), {}, error);
    ASSERT_TRUE(selected.succeeded()) << error.message;
    const FailureOr<OrdinaryWrongAddressResult> expected =
        apply_ordinary_wrong_address(source, selected.value(), 8, error);
    ASSERT_TRUE(expected.succeeded()) << error.message;
    std::vector<std::string> list = args();
    list.push_back("--list");
    ASSERT_EQ(run(list), 0) << stderr_;
    EXPECT_NE(stdout_.find("ordinal=0 exact=0"), std::string::npos);
    EXPECT_NE(stdout_.find("form=global_load_b32"), std::string::npos);
    ASSERT_EQ(run(mutation_args({"--exact", "0", "--ordinal", "999"})), 0) << stderr_;
    const std::vector<uint8_t> actual = read(output_);
    EXPECT_EQ(actual, expected.value().image);
    EXPECT_TRUE(test::ordinary_wrong_address_output_matches(source, 0, 8, actual));
    ASSERT_TRUE(std::filesystem::remove(output_));
    EXPECT_EQ(read(input_), image);
    std::vector<std::vector<uint32_t>> malformed = {test::ordinary_acquire_words(1)};
    std::vector<uint32_t> missing_wait = test::ordinary_acquire_words();
    missing_wait.erase(missing_wait.begin() + 3);
    malformed.push_back(std::move(missing_wait));
    std::vector<uint32_t> padding = test::ordinary_acquire_words();
    padding[0] |= 1u << 8;
    malformed.push_back(std::move(padding));
    std::vector<uint32_t> store = test::ordinary_acquire_words();
    store.insert(store.begin() + 4, {0xee068004u, (2u << 18) | (7u << 23), 10u});
    malformed.push_back(std::move(store));
    std::vector<uint32_t> barrier = test::ordinary_acquire_words();
    barrier.insert(barrier.begin() + 4, 0xbf940000u);
    malformed.push_back(std::move(barrier));
    for (const std::vector<uint32_t> &words : malformed) {
      const std::vector<uint8_t> invalid = test::make_ordinary_words_elf(fixture, words);
      ASSERT_NO_FATAL_FAILURE(write(input_, invalid));
      const FaultMutationContext invalid_source(invalid);
      const FailureOr<OrdinaryAcquireInventory> rejected =
          discover_ordinary_acquires(invalid_source, "test_kernel", error);
      if (rejected.succeeded()) {
        EXPECT_TRUE(rejected.value().sites().empty());
      }
      EXPECT_NE(run(mutation_args({"--exact", "0", "--ordinal", "0"})), 0);
      ASSERT_NO_FATAL_FAILURE(expect_no_partial_output());
      EXPECT_EQ(read(input_), invalid);
    }
  }
}

TEST_F(FaultOrdinaryCliSupport, HelpAndCapabilitiesMatchEnabledOrdinaryProfiles) {
  ASSERT_EQ(run({"--help"}), 0) << stderr_;
  EXPECT_NE(stdout_.find("OrdinaryWrongAddress"), std::string::npos);
  EXPECT_NE(stdout_.find("--address-delta"), std::string::npos);
  ASSERT_EQ(run({"--capabilities"}), 0) << stderr_;
  size_t lines = 0;
  for (size_t position = 0;
       (position = stdout_.find("kind=OrdinaryWrongAddress ", position)) != std::string::npos;
       ++position)
    ++lines;
  EXPECT_EQ(lines, 8u);
  for (const test::OrdinaryTargetCase &fixture : test::kOrdinaryTargetCases) {
    const IsaTargetBinding *description = default_isa_target_registry().find_target(fixture.target);
    ASSERT_NE(description, nullptr);
    const std::string row =
        "kind=OrdinaryWrongAddress target=" + std::string(description->id) +
        " wave=" + std::to_string(fixture.wave) +
        " profile=" + (fixture.arch == IsaFamily::Cdna5 ? "Cdna5GlobalB32" : "Rdna4GlobalB32");
    EXPECT_NE(stdout_.find(row), std::string::npos);
  }
}

class FaultRdna4OrdinaryParityCli : public FaultOrdinaryCliSupport,
                                    public ::testing::WithParamInterface<test::OrdinaryTargetCase> {
};

TEST_P(FaultRdna4OrdinaryParityCli, OriginalQualifiedVariantsAgreeWithLibraryAndIndependentOracle) {
  const test::OrdinaryTargetCase &fixture = GetParam();
  for (const test::OrdinaryAddressParityCase &parity :
       test::rdna4_ordinary_address_parity_cases(fixture)) {
    SCOPED_TRACE(parity.name);
    ASSERT_FALSE(parity.image.empty());
    ASSERT_NO_FATAL_FAILURE(write(input_, parity.image));
    const FaultMutationContext source(parity.image);
    FaultMutationError error;
    const FailureOr<OrdinaryAcquireInventory> inventory =
        discover_ordinary_acquires(source, parity.kernel, error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    ASSERT_EQ(inventory.value().sites().size(), 1u);
    const FailureOr<OrdinaryAcquireSelection> selected =
        select_ordinary_acquire(inventory.value(), {}, error);
    ASSERT_TRUE(selected.succeeded()) << error.message;
    const FailureOr<OrdinaryWrongAddressResult> expected =
        apply_ordinary_wrong_address(source, selected.value(), 8, error);
    ASSERT_TRUE(expected.succeeded()) << error.message;
    std::vector<std::string> list = args(parity.kernel);
    list.push_back("--list");
    ASSERT_EQ(run(list), 0) << stderr_;
    EXPECT_NE(stdout_.find("ordinal=0 exact=" + std::to_string(parity.offset)), std::string::npos);
    for (const std::vector<std::string> &selector :
         {std::vector<std::string>{"--ordinal", "0"},
          std::vector<std::string>{"--exact", std::to_string(parity.offset), "--ordinal", "999"}}) {
      ASSERT_EQ(run(mutation_args(selector, parity.kernel)), 0) << stderr_;
      const std::vector<uint8_t> actual = read(output_);
      EXPECT_EQ(actual, expected.value().image);
      EXPECT_TRUE(test::ordinary_wrong_address_output_matches(source, parity.offset, 8, actual));
      EXPECT_EQ(read(input_), parity.image);
      ASSERT_TRUE(std::filesystem::remove(output_));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Targets, FaultOrdinaryWrongAddressCli,
                         ::testing::ValuesIn(test::kOrdinaryTargetCases),
                         [](const ::testing::TestParamInfo<test::OrdinaryTargetCase> &info) {
                           return info.param.name;
                         });

INSTANTIATE_TEST_SUITE_P(Targets, FaultRdna4OrdinaryParityCli,
                         ::testing::ValuesIn(test::kRdna4OrdinaryAddressCases),
                         [](const ::testing::TestParamInfo<test::OrdinaryTargetCase> &info) {
                           return info.param.name;
                         });

} // namespace
} // namespace rocjitsu
