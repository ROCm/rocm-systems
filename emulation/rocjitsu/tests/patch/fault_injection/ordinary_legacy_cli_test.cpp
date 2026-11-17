// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cli_test_support.h"
#include "legacy_ordinary_fixture.h"
#include "legacy_ordinary_oracle.h"
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
#include <span>
#include <string>
#include <string_view>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace {

class FaultLegacyOrdinaryCliSupport : public ::testing::Test {
protected:
  test::ScopedTempDirectory temporary_;
  std::filesystem::path root_, input_, output_;
  std::string cli_, stdout_, stderr_;

  void SetUp() override {
    ASSERT_TRUE(test::mutation_cli_path(cli_));
    ASSERT_TRUE(temporary_.initialize("legacy-ordinary-cli-"));
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

inline constexpr std::array<test::OrdinaryTargetCase, 2> kLegacyOrdinaryCliCases =
    test::kCdna34OrdinaryCases;

std::vector<uint32_t> legacy_cli_words(const test::OrdinaryTargetCase &, int32_t offset = 4,
                                       bool modifiers = false) {
  return test::cdna34_ordinary_words(offset, modifiers);
}

std::string_view legacy_cli_form(const test::OrdinaryTargetCase &) { return "global_load_dword"; }

std::string_view legacy_cli_profile(const test::OrdinaryTargetCase &) {
  return "Cdna34GlobalDword";
}

uint32_t legacy_cli_nonzero_wait(const test::OrdinaryTargetCase &) {
  return test::kCdna34OrdinaryWait | 1u;
}

uint32_t legacy_cli_store(const test::OrdinaryTargetCase &) { return 0xde708000u; }
uint32_t legacy_cli_barrier(const test::OrdinaryTargetCase &) { return 0xbf8a0000u; }

class FaultLegacyOrdinaryCli : public FaultLegacyOrdinaryCliSupport,
                               public ::testing::WithParamInterface<test::OrdinaryTargetCase> {};

TEST_P(FaultLegacyOrdinaryCli, ExactOrdinalLibraryAndFullGuestOracleMatchEachConcreteProfile) {
  const test::OrdinaryTargetCase &fixture = GetParam();
  const std::vector<uint8_t> image =
      test::make_ordinary_words_elf(fixture, legacy_cli_words(fixture, -4, true));
  ASSERT_NO_FATAL_FAILURE(write(input_, image));
  const IsaTargetBinding *description = default_isa_target_registry().find_target(fixture.target);
  ASSERT_NE(description, nullptr);
  const FaultMutationContext context(image);
  FaultMutationError error;
  const FailureOr<OrdinaryAcquireInventory> found =
      discover_ordinary_acquires(context, "test_kernel", error);
  ASSERT_TRUE(found.succeeded()) << error.message;
  ASSERT_EQ(found.value().sites().size(), 1u);
  EXPECT_EQ(found.value().sites().front().instruction_size, 8u);
  const FailureOr<OrdinaryAcquireSelection> selected =
      select_ordinary_acquire(found.value(), {}, error);
  ASSERT_TRUE(selected.succeeded()) << error.message;
  const FailureOr<OrdinaryWrongAddressResult> expected =
      apply_ordinary_wrong_address(context, selected.value(), 8, error);
  ASSERT_TRUE(expected.succeeded()) << error.message;
  ASSERT_EQ(run({"--capabilities"}), 0) << stderr_;
  const std::string row = "kind=OrdinaryWrongAddress target=" + std::string(description->id) +
                          " wave=" + std::to_string(fixture.wave) +
                          " profile=" + std::string(legacy_cli_profile(fixture));
  EXPECT_NE(stdout_.find(row), std::string::npos);
  std::vector<std::string> list = args();
  list.push_back("--list");
  ASSERT_EQ(run(list), 0) << stderr_;
  EXPECT_NE(stdout_.find("ordinal=0 exact=0"), std::string::npos);
  EXPECT_NE(stdout_.find("form=" + std::string(legacy_cli_form(fixture))), std::string::npos);
  for (const std::vector<std::string> &selector :
       {std::vector<std::string>{"--ordinal", "0"},
        std::vector<std::string>{"--exact", "0x0", "--ordinal", "999"}}) {
    std::vector<std::string> command = mutation_args(selector);
    command.insert(command.end(), {"--target", std::string(description->id)});
    ASSERT_EQ(run(command), 0) << stderr_;
    const std::vector<uint8_t> actual = read(output_);
    EXPECT_EQ(actual, expected.value().image);
    EXPECT_TRUE(
        test::legacy_ordinary_output_matches(context, 0, 8, actual, legacy_cli_form(fixture)));
    EXPECT_EQ(read(input_), image);
    ASSERT_TRUE(std::filesystem::remove(output_));
  }
}

class FaultCdna34OrdinaryCacheCli : public FaultLegacyOrdinaryCliSupport,
                                    public ::testing::WithParamInterface<test::OrdinaryTargetCase> {
};

TEST_P(FaultCdna34OrdinaryCacheCli, NativeLoadScopeMustCoverAgentRegardlessOfSc0AndNt) {
  const test::OrdinaryTargetCase &fixture = GetParam();
  for (uint32_t modifiers : {0u, 0x00010000u, 0x00020000u, 0x00030000u}) {
    SCOPED_TRACE(modifiers);
    for (bool agent_scope : {false, true}) {
      SCOPED_TRACE(agent_scope);
      std::vector<uint32_t> words = test::cdna34_ordinary_words(-4, true);
      words[0] = 0xdc509ffcu | modifiers | (agent_scope ? 0x02000000u : 0u);
      const std::vector<uint8_t> image = test::make_ordinary_words_elf(fixture, words);
      ASSERT_NO_FATAL_FAILURE(write(input_, image));
      std::vector<std::string> list = args();
      list.push_back("--list");
      ASSERT_EQ(run(list), 0) << stderr_;
      if (agent_scope) {
        EXPECT_NE(stdout_.find("ordinal=0 exact=0"), std::string::npos);
        ASSERT_EQ(run(mutation_args()), 0) << stderr_;
        const FaultMutationContext context(image);
        EXPECT_TRUE(test::legacy_ordinary_output_matches(context, 0, 8, read(output_),
                                                         "global_load_dword"));
        ASSERT_TRUE(std::filesystem::remove(output_));
      } else {
        EXPECT_EQ(stdout_.find("ordinal="), std::string::npos);
        EXPECT_NE(run(mutation_args()), 0);
        ASSERT_NO_FATAL_FAILURE(expect_no_partial_output());
        if (std::filesystem::exists(output_)) {
          ASSERT_TRUE(std::filesystem::remove(output_));
        }
      }
      EXPECT_EQ(read(input_), image);
    }
  }
  for (uint32_t modifiers : {0u, 0x00010000u, 0x00020000u, 0x00030000u}) {
    SCOPED_TRACE(modifiers);
    std::vector<uint32_t> sequence = test::cdna34_ordinary_words(-4, true);
    sequence[0] = 0xdc509ffcu | modifiers;
    std::vector<uint32_t> words = sequence;
    sequence[0] |= 0x02000000u;
    words.insert(words.end(), sequence.begin(), sequence.end());
    const std::vector<uint8_t> image = test::make_ordinary_words_elf(fixture, words);
    ASSERT_NO_FATAL_FAILURE(write(input_, image));
    const FaultMutationContext context(image);
    FaultMutationError error;
    const FailureOr<OrdinaryAcquireInventory> found =
        discover_ordinary_acquires(context, "test_kernel", error);
    ASSERT_TRUE(found.succeeded()) << error.message;
    ASSERT_EQ(found.value().sites().size(), 1u);
    const FailureOr<OrdinaryAcquireSelection> selected =
        select_ordinary_acquire(found.value(), {}, error);
    ASSERT_TRUE(selected.succeeded()) << error.message;
    EXPECT_EQ(selected.value().site().offset, 20u);
    const FailureOr<OrdinaryWrongAddressResult> expected =
        apply_ordinary_wrong_address(context, selected.value(), 8, error);
    ASSERT_TRUE(expected.succeeded()) << error.message;
    std::vector<std::string> list = args();
    list.push_back("--list");
    ASSERT_EQ(run(list), 0) << stderr_;
    EXPECT_NE(stdout_.find("ordinal=0 exact=20"), std::string::npos);
    EXPECT_EQ(stdout_.find("ordinal=1"), std::string::npos);
    for (const std::vector<std::string> &selector :
         {std::vector<std::string>{"--ordinal", "0"},
          std::vector<std::string>{"--exact", "20", "--ordinal", "999"}}) {
      ASSERT_EQ(run(mutation_args(selector)), 0) << stderr_;
      EXPECT_EQ(read(output_), expected.value().image);
      EXPECT_TRUE(
          test::legacy_ordinary_output_matches(context, 20, 8, read(output_), "global_load_dword"));
      ASSERT_TRUE(std::filesystem::remove(output_));
    }
    EXPECT_NE(run(mutation_args({"--exact", "0"})), 0);
    ASSERT_NO_FATAL_FAILURE(expect_no_partial_output());
    EXPECT_EQ(read(input_), image);
  }
}

TEST_P(FaultCdna34OrdinaryCacheCli, NativeCacheFlagsGateQualifiedListingAndMutation) {
  const test::OrdinaryTargetCase &fixture = GetParam();
  for (uint32_t cache_word : {0xe0a40000u, 0xe0a44000u, 0xe0a48000u, 0xe0a4c000u}) {
    SCOPED_TRACE(cache_word);
    std::vector<uint32_t> words = test::cdna34_ordinary_words(-4, true);
    words[3] = cache_word;
    const std::vector<uint8_t> image = test::make_ordinary_words_elf(fixture, words);
    ASSERT_NO_FATAL_FAILURE(write(input_, image));
    std::vector<std::string> list = args();
    list.push_back("--list");
    ASSERT_EQ(run(list), 0) << stderr_;
    const bool effective = cache_word == 0xe0a48000u || cache_word == 0xe0a4c000u;
    if (effective) {
      EXPECT_NE(stdout_.find("ordinal=0 exact=0"), std::string::npos);
    } else {
      EXPECT_EQ(stdout_.find("ordinal="), std::string::npos);
    }
    const FaultMutationContext context(image);
    for (const std::vector<std::string> &selector :
         {std::vector<std::string>{"--ordinal", "0"},
          std::vector<std::string>{"--exact", "0", "--ordinal", "999"}}) {
      if (effective) {
        ASSERT_EQ(run(mutation_args(selector)), 0) << stderr_;
        const std::vector<uint8_t> actual = read(output_);
        EXPECT_TRUE(
            test::legacy_ordinary_output_matches(context, 0, 8, actual, "global_load_dword"));
        ASSERT_TRUE(std::filesystem::remove(output_));
      } else {
        EXPECT_NE(run(mutation_args(selector)), 0);
        ASSERT_NO_FATAL_FAILURE(expect_no_partial_output());
      }
      EXPECT_EQ(read(input_), image);
      if (std::filesystem::exists(output_)) {
        ASSERT_TRUE(std::filesystem::remove(output_));
      }
    }
  }
}

TEST_P(FaultLegacyOrdinaryCli, QualifiedOrdinalsAndInvalidExactKeepTheOriginalSelectionContract) {
  const test::OrdinaryTargetCase &fixture = GetParam();
  std::vector<uint32_t> words;
  const size_t sequence_bytes = legacy_cli_words(fixture).size() * 4;
  for (size_t ordinal : {0u, 1u, 2u}) {
    std::vector<uint32_t> sequence = legacy_cli_words(fixture);
    if (ordinal == 0)
      sequence[2] = legacy_cli_nonzero_wait(fixture);
    words.insert(words.end(), sequence.begin(), sequence.end());
  }
  const std::vector<uint8_t> image = test::make_ordinary_words_elf(fixture, std::move(words));
  ASSERT_NO_FATAL_FAILURE(write(input_, image));
  const FaultMutationContext context(image);
  FaultMutationError error;
  const FailureOr<OrdinaryAcquireInventory> found =
      discover_ordinary_acquires(context, "test_kernel", error);
  ASSERT_TRUE(found.succeeded()) << error.message;
  ASSERT_EQ(found.value().sites().size(), 2u);
  const FailureOr<OrdinaryAcquireSelection> selected =
      select_ordinary_acquire(found.value(), {.exact = std::nullopt, .ordinal = 1}, error);
  ASSERT_TRUE(selected.succeeded()) << error.message;
  EXPECT_EQ(selected.value().site().offset, sequence_bytes * 2);
  const FailureOr<OrdinaryWrongAddressResult> expected =
      apply_ordinary_wrong_address(context, selected.value(), 8, error);
  ASSERT_TRUE(expected.succeeded()) << error.message;
  for (const std::vector<std::string> &selector :
       {std::vector<std::string>{"--ordinal", "1"},
        std::vector<std::string>{"--exact", std::to_string(sequence_bytes * 2), "--ordinal",
                                 "999"}}) {
    ASSERT_EQ(run(mutation_args(selector)), 0) << stderr_;
    const std::vector<uint8_t> actual = read(output_);
    EXPECT_EQ(actual, expected.value().image);
    EXPECT_TRUE(test::legacy_ordinary_output_matches(context, sequence_bytes * 2, 8, actual,
                                                     legacy_cli_form(fixture)));
    ASSERT_TRUE(std::filesystem::remove(output_));
  }
  for (const std::vector<std::string> &selector :
       {std::vector<std::string>{"--exact", "0", "--ordinal", "0"},
        std::vector<std::string>{"--exact", "4", "--ordinal", "1"},
        std::vector<std::string>{"--ordinal", "2"}}) {
    EXPECT_NE(run(mutation_args(selector)), 0);
    ASSERT_NO_FATAL_FAILURE(expect_no_partial_output());
    EXPECT_EQ(read(input_), image);
  }
  std::vector<std::string> wrong_target = mutation_args();
  wrong_target.insert(wrong_target.end(), {"--target", "gfx1250"});
  EXPECT_NE(run(wrong_target), 0);
  ASSERT_NO_FATAL_FAILURE(expect_no_partial_output());
}

TEST_P(FaultLegacyOrdinaryCli, IncompleteAcquireProfilesRefuseWithoutPublishingAnyOutput) {
  const test::OrdinaryTargetCase &fixture = GetParam();
  const std::vector<uint32_t> valid = legacy_cli_words(fixture);
  std::vector<std::vector<uint32_t>> invalid;
  std::vector<uint32_t> no_wait = valid;
  no_wait.erase(no_wait.begin() + 2);
  invalid.push_back(no_wait);
  invalid.push_back({valid[0], valid[1], valid[2]});
  std::vector<uint32_t> wrong_wait = valid;
  wrong_wait[2] = legacy_cli_nonzero_wait(fixture);
  invalid.push_back(wrong_wait);
  std::vector<uint32_t> store = valid;
  store[0] = legacy_cli_store(fixture);
  invalid.push_back(store);
  std::vector<uint32_t> barrier = valid;
  barrier.insert(barrier.begin() + 3, legacy_cli_barrier(fixture));
  invalid.push_back(barrier);
  if (fixture.arch != IsaFamily::Rdna3) {
    std::vector<uint8_t> invalid_wave = test::make_ordinary_words_elf(fixture, valid);
    ASSERT_TRUE(test::mark_legacy_ordinary_wave32(invalid_wave, "test_kernel"));
    ASSERT_NO_FATAL_FAILURE(write(input_, invalid_wave));
    EXPECT_NE(run(mutation_args()), 0);
    ASSERT_NO_FATAL_FAILURE(expect_no_partial_output());
    EXPECT_EQ(read(input_), invalid_wave);
  }
  for (const std::vector<uint32_t> &words : invalid) {
    const std::vector<uint8_t> image = test::make_ordinary_words_elf(fixture, words);
    ASSERT_NO_FATAL_FAILURE(write(input_, image));
    EXPECT_NE(run(mutation_args()), 0);
    EXPECT_FALSE(stderr_.empty());
    ASSERT_NO_FATAL_FAILURE(expect_no_partial_output());
    EXPECT_EQ(read(input_), image);
  }
}

TEST_P(FaultLegacyOrdinaryCli, InvalidDeltaOverflowAndUnalignedSumDoNotPublishPartialFiles) {
  const test::OrdinaryTargetCase &fixture = GetParam();
  const std::vector<uint8_t> image =
      test::make_ordinary_words_elf(fixture, legacy_cli_words(fixture));
  ASSERT_NO_FATAL_FAILURE(write(input_, image));
  for (std::string_view delta : {"0", "2", "-4", "4294967295"}) {
    std::vector<std::string> command = args();
    command.insert(command.end(),
                   {"--output", output_.string(), "--address-delta", std::string(delta)});
    EXPECT_NE(run(command), 0);
    ASSERT_NO_FATAL_FAILURE(expect_no_partial_output());
    EXPECT_EQ(read(input_), image);
  }
  for (int32_t offset : {4092, 2}) {
    const std::vector<uint8_t> invalid =
        test::make_ordinary_words_elf(fixture, legacy_cli_words(fixture, offset));
    ASSERT_NO_FATAL_FAILURE(write(input_, invalid));
    EXPECT_NE(run(mutation_args()), 0);
    ASSERT_NO_FATAL_FAILURE(expect_no_partial_output());
    EXPECT_EQ(read(input_), invalid);
  }
}

TEST_P(FaultLegacyOrdinaryCli, RealSharedOwnerRoundtripAndUnrelatedOwnerRefusalAreConcrete) {
  const test::OrdinaryTargetCase &fixture = GetParam();
  const std::vector<uint32_t> body = legacy_cli_words(fixture);
  const std::vector<uint8_t> image = test::make_legacy_ordinary_shared_elf(fixture, body, 12);
  ASSERT_NO_FATAL_FAILURE(write(input_, image));
  const FaultMutationContext context(image);
  FaultMutationError error;
  const FailureOr<OrdinaryAcquireInventory> found =
      discover_ordinary_acquires(context, "test_kernel1", error);
  ASSERT_TRUE(found.succeeded()) << error.message;
  const FailureOr<OrdinaryAcquireSelection> selected =
      select_ordinary_acquire(found.value(), {}, error);
  ASSERT_TRUE(selected.succeeded()) << error.message;
  EXPECT_EQ(selected.value().site().affected_owners.size(), 2u);
  const FailureOr<OrdinaryWrongAddressResult> expected =
      apply_ordinary_wrong_address(context, selected.value(), 8, error);
  ASSERT_TRUE(expected.succeeded()) << error.message;
  ASSERT_EQ(run(mutation_args({"--exact", "512"}, "test_kernel1")), 0) << stderr_;
  const std::vector<uint8_t> actual = read(output_);
  EXPECT_EQ(actual, expected.value().image);
  EXPECT_TRUE(
      test::legacy_ordinary_output_matches(context, 512, 8, actual, legacy_cli_form(fixture)));
  ASSERT_TRUE(std::filesystem::remove(output_));
  EXPECT_EQ(read(input_), image);
  const std::vector<uint8_t> unrelated =
      test::make_legacy_ordinary_shared_elf(fixture, body, 12, false);
  ASSERT_NO_FATAL_FAILURE(write(input_, unrelated));
  EXPECT_NE(run(mutation_args({"--exact", "512"}, "test_kernel1")), 0);
  ASSERT_NO_FATAL_FAILURE(expect_no_partial_output());
  EXPECT_EQ(read(input_), unrelated);
  if (fixture.arch != IsaFamily::Rdna3) {
    for (bool shared_sibling : {false, true}) {
      std::vector<uint8_t> sibling_wave32 =
          test::make_legacy_ordinary_shared_elf(fixture, body, 12, shared_sibling);
      ASSERT_TRUE(test::mark_legacy_ordinary_wave32(sibling_wave32, "test_kernel1"));
      ASSERT_NO_FATAL_FAILURE(write(input_, sibling_wave32));
      if (shared_sibling) {
        EXPECT_NE(run(mutation_args({"--exact", "512"}, "test_kernel0")), 0);
        ASSERT_NO_FATAL_FAILURE(expect_no_partial_output());
      } else {
        const FaultMutationContext valid_owner(sibling_wave32);
        const FailureOr<OrdinaryAcquireInventory> inventory =
            discover_ordinary_acquires(valid_owner, "test_kernel0", error);
        ASSERT_TRUE(inventory.succeeded()) << error.message;
        const FailureOr<OrdinaryAcquireSelection> selected_owner =
            select_ordinary_acquire(inventory.value(), {}, error);
        ASSERT_TRUE(selected_owner.succeeded()) << error.message;
        const FailureOr<OrdinaryWrongAddressResult> allowed =
            apply_ordinary_wrong_address(valid_owner, selected_owner.value(), 8, error);
        ASSERT_TRUE(allowed.succeeded()) << error.message;
        ASSERT_EQ(run(mutation_args({"--exact", "512"}, "test_kernel0")), 0) << stderr_;
        const std::vector<uint8_t> actual_owner = read(output_);
        EXPECT_EQ(actual_owner, allowed.value().image);
        EXPECT_TRUE(test::legacy_ordinary_output_matches(valid_owner, 512, 8, actual_owner,
                                                         legacy_cli_form(fixture)));
        ASSERT_TRUE(std::filesystem::remove(output_));
      }
      EXPECT_EQ(read(input_), sibling_wave32);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Targets, FaultCdna34OrdinaryCacheCli,
                         ::testing::ValuesIn(test::kCdna34OrdinaryCases),
                         [](const ::testing::TestParamInfo<test::OrdinaryTargetCase> &info) {
                           return info.param.name;
                         });

INSTANTIATE_TEST_SUITE_P(Targets, FaultLegacyOrdinaryCli,
                         ::testing::ValuesIn(kLegacyOrdinaryCliCases),
                         [](const ::testing::TestParamInfo<test::OrdinaryTargetCase> &info) {
                           return info.param.name;
                         });

} // namespace
} // namespace rocjitsu
