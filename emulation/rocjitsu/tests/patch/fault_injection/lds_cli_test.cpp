// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cli_test_support.h"
#include "lds_fixture.h"
#include "remaining_subword_fixture.h"
#include "rocjitsu/code/file_io.h"
#include "rocjitsu/code/patch/fault_injection/lds_wrong_address.h"
#include "scoped_temp.h"
#include "subword_fixture.h"
#include "two_address_fixture.h"

#include <gtest/gtest.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace rocjitsu {
namespace {

class FaultLdsCli : public ::testing::Test {
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
    for (std::string &arg : args)
      argv.push_back(arg.data());
    argv.push_back(nullptr);
    const pid_t pid = fork();
    if (pid == 0) {
      execv(argv[0], argv.data());
      _exit(127);
    }
    if (pid < 0)
      return -1;
    int status = 0;
    if (waitpid(pid, &status, 0) != pid || !WIFEXITED(status))
      return -1;
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

TEST_F(FaultLdsCli, SharedParserAcceptsEqualsSelectorAndRejectsTruncation) {
  const std::vector<uint8_t> image = test::make_lds_elf();
  write_source(image);
  const FaultMutationContext context(image);
  ASSERT_EQ(run({"--input=" + input_.string(), "--kernel=test_kernel", "--kind=LdsWrongAddress",
                 "--output=" + output_.string(), "--ordinal=0", "--address-vgpr=6"}),
            0);
  EXPECT_TRUE(test::lds_output_matches(context, 0, 6, read()));
  ASSERT_TRUE(std::filesystem::remove(output_));
  for (const std::vector<std::string> &extra :
       {std::vector<std::string>{"--address-vgpr=6", "--address-vgpr", "7"},
        std::vector<std::string>{"--address-vgpr="}, std::vector<std::string>{"--address-vgpr=-1"},
        std::vector<std::string>{"--address-vgpr=4294967302"}}) {
    // 2^32 + 6 would become an allocated selector if narrowed before validation.
    std::vector<std::string> command = args();
    command.insert(command.end(), {"--output", output_.string()});
    command.insert(command.end(), extra.begin(), extra.end());
    EXPECT_EQ(run(command), 2);
    EXPECT_FALSE(std::filesystem::exists(output_));
  }
}

TEST_F(FaultLdsCli, Rdna4Wave32ExactOrdinalAndLibraryAgree) {
  for (uint32_t machine : {EF_AMDGPU_MACH_AMDGCN_GFX1200, EF_AMDGPU_MACH_AMDGCN_GFX1201}) {
    SCOPED_TRACE(machine);
    const std::vector<uint8_t> image = test::make_lds_load_elf(machine, true, 0);
    write_source(image);
    const FaultMutationContext source(image);
    FaultMutationError error;
    const FailureOr<LdsInventory> inventory =
        discover_lds_wrong_address(source, "test_kernel", error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    ASSERT_EQ(inventory.value().sites().size(), 2u);
    std::vector<std::string> list = args();
    list.push_back("--list");
    ASSERT_EQ(run(list), 0);
    for (size_t ordinal = 0; ordinal < 2; ++ordinal) {
      const FailureOr<LdsSelection> selection = select_lds_wrong_address(
          inventory.value(), {.exact = std::nullopt, .ordinal = ordinal}, error);
      ASSERT_TRUE(selection.succeeded());
      const FailureOr<LdsResult> result =
          apply_lds_wrong_address(source, selection.value(), 7, error);
      ASSERT_TRUE(result.succeeded()) << error.message;
      for (const std::vector<std::string> &selector :
           {std::vector<std::string>{"--ordinal", std::to_string(ordinal)},
            std::vector<std::string>{"--exact", std::to_string(ordinal * 8), "--ordinal", "999"}}) {
        std::vector<std::string> command = args();
        command.insert(command.end(), {"--output", output_.string(), "--address-vgpr", "7"});
        command.insert(command.end(), selector.begin(), selector.end());
        ASSERT_EQ(run(command), 0);
        EXPECT_EQ(read(), result.value().image);
        EXPECT_TRUE(test::lds_output_matches(source, ordinal * 8, 7, read(), ordinal == 0));
        ASSERT_TRUE(std::filesystem::remove(output_));
      }
    }
    for (const std::vector<std::string> &extra :
         {std::vector<std::string>{}, std::vector<std::string>{"--address-vgpr", "2"},
          std::vector<std::string>{"--address-vgpr", "8"},
          std::vector<std::string>{"--address-vgpr", "256"}}) {
      std::vector<std::string> command = args();
      command.insert(command.end(), {"--output", output_.string()});
      command.insert(command.end(), extra.begin(), extra.end());
      EXPECT_NE(run(command), 0);
      EXPECT_FALSE(std::filesystem::exists(output_));
    }
  }
}

TEST_F(FaultLdsCli, ExactOrdinalAndLibraryAgree) {
  const std::vector<uint8_t> image = test::make_lds_elf();
  write_source(image);
  const FaultMutationContext context(image);
  FaultMutationError error;
  const FailureOr<LdsInventory> inventory =
      discover_lds_wrong_address(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const FailureOr<LdsSelection> selection =
      select_lds_wrong_address(inventory.value(), {.exact = std::nullopt, .ordinal = 1}, error);
  ASSERT_TRUE(selection.succeeded()) << error.message;
  const FailureOr<LdsResult> result = apply_lds_wrong_address(context, selection.value(), 6, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  std::vector<std::string> command = args();
  command.push_back("--list");
  ASSERT_EQ(run(command), 0);
  command = args();
  command.insert(command.end(), {"--list"});
  ASSERT_EQ(run(command), 0);
  command = args();
  command.insert(command.end(),
                 {"--output", output_.string(), "--ordinal", "1", "--address-vgpr", "6"});
  ASSERT_EQ(run(command), 0);
  EXPECT_EQ(read(), result.value().image);
  EXPECT_TRUE(test::lds_output_matches(context, 8, 6, read()));
  ASSERT_TRUE(std::filesystem::remove(output_));
  command = args();
  command.insert(command.end(), {"--output", output_.string(), "--exact", "8", "--ordinal", "999",
                                 "--address-vgpr", "6"});
  ASSERT_EQ(run(command), 0);
  EXPECT_EQ(read(), result.value().image);
}

TEST_F(FaultLdsCli, Rdna4LoadExactOrdinalAndLibraryAgree) {
  for (uint32_t target : {EF_AMDGPU_MACH_AMDGCN_GFX1200, EF_AMDGPU_MACH_AMDGCN_GFX1201}) {
    SCOPED_TRACE(target);
    const std::vector<uint8_t> image = test::make_lds_load_elf(target);
    write_source(image);
    const FaultMutationContext context(image);
    FaultMutationError error;
    const FailureOr<LdsInventory> inventory =
        discover_lds_wrong_address(context, "test_kernel", error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    const FailureOr<LdsSelection> selection =
        select_lds_wrong_address(inventory.value(), {}, error);
    ASSERT_TRUE(selection.succeeded()) << error.message;
    EXPECT_EQ(selection.value().site().offset, 0u);
    const FailureOr<LdsResult> result =
        apply_lds_wrong_address(context, selection.value(), 6, error);
    ASSERT_TRUE(result.succeeded()) << error.message;
    std::vector<std::string> command = args();
    command.insert(command.end(),
                   {"--output", output_.string(), "--ordinal", "0", "--address-vgpr", "6"});
    ASSERT_EQ(run(command), 0);
    EXPECT_EQ(read(), result.value().image);
    EXPECT_TRUE(test::lds_output_matches(context, 0, 6, read(), true));
    ASSERT_TRUE(std::filesystem::remove(output_));
    command = args();
    command.insert(command.end(), {"--output", output_.string(), "--exact", "0", "--ordinal", "999",
                                   "--address-vgpr", "6"});
    ASSERT_EQ(run(command), 0);
    EXPECT_EQ(read(), result.value().image);
    ASSERT_TRUE(std::filesystem::remove(output_));
    for (const std::vector<std::string> &extra :
         {std::vector<std::string>{"--exact", "4", "--ordinal", "0", "--address-vgpr", "6"},
          std::vector<std::string>{}, std::vector<std::string>{"--address-vgpr", "2"},
          std::vector<std::string>{"--address-vgpr", "8"},
          std::vector<std::string>{"--address-vgpr", "256"}}) {
      command = args();
      command.insert(command.end(), {"--output", output_.string()});
      command.insert(command.end(), extra.begin(), extra.end());
      EXPECT_NE(run(command), 0);
      EXPECT_FALSE(std::filesystem::exists(output_));
    }
  }
}

TEST_F(FaultLdsCli, Gfx1200Wave64ExactOrdinalAndLibraryAgree) {
  const std::vector<uint8_t> image = test::make_lds_elf(1, false, EF_AMDGPU_MACH_AMDGCN_GFX1200);
  write_source(image);
  std::vector<std::string> command = args();
  command.insert(command.end(), {"--target", "gfx1200", "--list"});
  ASSERT_EQ(run(command), 0);
  const FaultMutationContext context(image);
  FaultMutationError error;
  const FailureOr<LdsInventory> inventory =
      discover_lds_wrong_address(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const FailureOr<LdsSelection> selection =
      select_lds_wrong_address(inventory.value(), {.exact = std::nullopt, .ordinal = 1}, error);
  ASSERT_TRUE(selection.succeeded()) << error.message;
  const FailureOr<LdsResult> result = apply_lds_wrong_address(context, selection.value(), 6, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  command = args();
  command.insert(command.end(), {"--target", "gfx1200", "--output", output_.string(), "--ordinal",
                                 "1", "--address-vgpr", "6"});
  ASSERT_EQ(run(command), 0);
  EXPECT_EQ(read(), result.value().image);
  EXPECT_TRUE(test::lds_output_matches(context, 8, 6, read()));
  ASSERT_TRUE(std::filesystem::remove(output_));
  command = args();
  command.insert(command.end(), {"--target", "gfx1200", "--output", output_.string(), "--exact",
                                 "8", "--ordinal", "999", "--address-vgpr", "6"});
  ASSERT_EQ(run(command), 0);
  EXPECT_EQ(read(), result.value().image);
}

TEST_F(FaultLdsCli, Gfx1250LibraryAndCliSelectSameBankedLoad) {
  constexpr std::array<uint32_t, 1> set_bank =
      cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 1});
  constexpr std::array<uint32_t, 2> store =
      cdna5::build_vds(cdna5::kDsStoreB32Vds, {.addr = 2, .data0 = 3});
  constexpr std::array<uint32_t, 2> load =
      cdna5::build_vds(cdna5::kDsLoadB32Vds, {.addr = 2, .vdst = 5});
  const std::vector<uint8_t> image =
      test::make_cdna5_lds_elf({set_bank[0], store[0], store[1], load[0], load[1]}, 16);
  write_source(image);
  const FaultMutationContext context(image);
  FaultMutationError error;
  const FailureOr<LdsInventory> inventory =
      discover_lds_wrong_address(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const FailureOr<LdsSelection> selection =
      select_lds_wrong_address(inventory.value(), {.exact = std::nullopt, .ordinal = 1}, error);
  ASSERT_TRUE(selection.succeeded()) << error.message;
  const FailureOr<LdsResult> result =
      apply_lds_wrong_address(context, selection.value(), 15, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  for (const std::vector<std::string> &selector :
       {std::vector<std::string>{"--ordinal", "1"},
        std::vector<std::string>{"--exact", "12", "--ordinal", "999"}}) {
    std::vector<std::string> command = args();
    command.insert(command.end(), selector.begin(), selector.end());
    command.insert(command.end(), {"--output", output_.string(), "--address-vgpr", "15"});
    ASSERT_EQ(run(command), 0);
    EXPECT_EQ(read(), result.value().image);
    EXPECT_TRUE(test::lds_output_matches(context, 12, 15, read()));
    ASSERT_TRUE(std::filesystem::remove(output_));
  }
  std::vector<std::string> invalid = args();
  invalid.insert(invalid.end(), {"--output", output_.string(), "--address-vgpr", "16"});
  EXPECT_NE(run(invalid), 0);
  EXPECT_FALSE(std::filesystem::exists(output_));
}

TEST_F(FaultLdsCli, InvalidRequestsDoNotPublishOutput) {
  write_source(test::make_lds_elf());
  for (const std::vector<std::string> &extra :
       {std::vector<std::string>{"--exact", "4", "--ordinal", "0", "--address-vgpr", "6"},
        std::vector<std::string>{}, std::vector<std::string>{"--address-vgpr", "2"},
        std::vector<std::string>{"--address-vgpr", "8"},
        std::vector<std::string>{"--address-vgpr", "256"}}) {
    std::vector<std::string> command = args();
    command.insert(command.end(), {"--output", output_.string()});
    command.insert(command.end(), extra.begin(), extra.end());
    EXPECT_NE(run(command), 0);
    EXPECT_FALSE(std::filesystem::exists(output_));
  }
}

TEST_F(FaultLdsCli, Gfx1200InvalidRequestsDoNotPublishOutput) {
  write_source(test::make_lds_elf(1, false, EF_AMDGPU_MACH_AMDGCN_GFX1200));
  for (const std::vector<std::string> &extra :
       {std::vector<std::string>{"--exact", "4", "--ordinal", "0", "--address-vgpr", "6"},
        std::vector<std::string>{}, std::vector<std::string>{"--address-vgpr", "2"},
        std::vector<std::string>{"--address-vgpr", "8"},
        std::vector<std::string>{"--address-vgpr", "256"}}) {
    std::vector<std::string> command = args();
    command.insert(command.end(), {"--output", output_.string()});
    command.insert(command.end(), extra.begin(), extra.end());
    EXPECT_NE(run(command), 0);
    EXPECT_FALSE(std::filesystem::exists(output_));
  }
}

TEST_F(FaultLdsCli, TwoAddressExactOrdinalAndLibraryAgree) {
  for (const test::TwoAddressTarget &target : test::kTwoAddressTargets) {
    for (bool wave32 : {false, true}) {
      if (target.cdna && wave32)
        continue;
      SCOPED_TRACE(static_cast<int>(target.target));
      SCOPED_TRACE(wave32);
      const uint32_t allocation = wave32 ? 32 : 16;
      const std::vector<uint8_t> image = test::make_two_address_elf(target, wave32);
      write_source(image);
      const FaultMutationContext context(image);
      FaultMutationError error;
      const FailureOr<LdsInventory> inventory =
          discover_lds_wrong_address(context, "test_kernel", error);
      ASSERT_TRUE(inventory.succeeded()) << error.message;
      ASSERT_EQ(inventory.value().sites().size(), 2u);
      std::vector<std::string> command = args();
      command.push_back("--list");
      ASSERT_EQ(run(command), 0);
      for (size_t ordinal = 0; ordinal < 2; ++ordinal) {
        SCOPED_TRACE(ordinal);
        const FailureOr<LdsSelection> selection = select_lds_wrong_address(
            inventory.value(), {.exact = std::nullopt, .ordinal = ordinal}, error);
        ASSERT_TRUE(selection.succeeded());
        EXPECT_EQ(selection.value().site().offset, ordinal * 8);
        for (uint32_t replacement : {6u, allocation - 1}) {
          const FailureOr<LdsResult> result =
              apply_lds_wrong_address(context, selection.value(), replacement, error);
          ASSERT_TRUE(result.succeeded()) << error.message;
          for (const std::vector<std::string> &selector :
               {std::vector<std::string>{"--ordinal", std::to_string(ordinal)},
                std::vector<std::string>{"--exact", std::to_string(ordinal * 8), "--ordinal",
                                         "999"}}) {
            command = args();
            command.insert(command.end(), {"--output", output_.string(), "--address-vgpr",
                                           std::to_string(replacement)});
            command.insert(command.end(), selector.begin(), selector.end());
            ASSERT_EQ(run(command), 0);
            const std::vector<uint8_t> emitted = read();
            EXPECT_EQ(emitted, result.value().image);
            EXPECT_TRUE(test::two_address_output_matches(context, target, ordinal * 8, replacement,
                                                         emitted));
            ASSERT_TRUE(std::filesystem::remove(output_));
          }
        }
      }
      for (const std::vector<std::string> &extra :
           {std::vector<std::string>{"--exact", "4", "--ordinal", "0", "--address-vgpr", "6"},
            std::vector<std::string>{}, std::vector<std::string>{"--address-vgpr", "2"},
            std::vector<std::string>{"--address-vgpr", std::to_string(allocation)},
            std::vector<std::string>{"--address-vgpr", "256"}}) {
        command = args();
        command.insert(command.end(), {"--output", output_.string()});
        command.insert(command.end(), extra.begin(), extra.end());
        EXPECT_NE(run(command), 0);
        EXPECT_FALSE(std::filesystem::exists(output_));
      }
    }
  }
}

struct SubwordCliCase {
  test::LdsFormCase fixture;
  bool wave32;
};

std::vector<SubwordCliCase> subword_cli_cases() {
  std::vector<SubwordCliCase> cases;
  for (std::span<const test::LdsFormCase> fixtures :
       {std::span<const test::LdsFormCase>(test::kSubwordCases),
        std::span<const test::LdsFormCase>(test::kRemainingSubwordCases)}) {
    for (const test::LdsFormCase &fixture : fixtures) {
      cases.push_back({.fixture = fixture, .wave32 = false});
      if (fixture.arch == IsaFamily::Rdna4)
        cases.push_back({.fixture = fixture, .wave32 = true});
    }
  }
  return cases;
}

class FaultLdsCliSubword : public FaultLdsCli,
                           public ::testing::WithParamInterface<SubwordCliCase> {};

TEST_P(FaultLdsCliSubword, ExactOrdinalAndLibraryAgree) {
  const test::LdsFormCase &fixture = GetParam().fixture;
  const bool wave32 = GetParam().wave32;
  SCOPED_TRACE(fixture.form);
  SCOPED_TRACE(static_cast<int>(fixture.target));
  SCOPED_TRACE(wave32);
  const uint32_t boundary =
      fixture.target == ROCJITSU_CODE_TARGET_GFX1250 ? 64u : (wave32 ? 32u : 16u);
  const std::vector<uint8_t> image = test::make_lds_form_elf(fixture, false, wave32);
  write_source(image);
  const FaultMutationContext context(image);
  FaultMutationError error;
  const FailureOr<LdsInventory> inventory =
      discover_lds_wrong_address(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().sites().size(), 1u);
  EXPECT_EQ(inventory.value().sites()[0].form, fixture.form);
  const FailureOr<LdsSelection> selection =
      select_lds_wrong_address(inventory.value(), {.exact = std::nullopt, .ordinal = 0}, error);
  ASSERT_TRUE(selection.succeeded()) << error.message;
  const FailureOr<LdsSelection> exact = select_lds_wrong_address(
      inventory.value(),
      {.exact = selection.value(), .ordinal = std::numeric_limits<size_t>::max()}, error);
  ASSERT_TRUE(exact.succeeded()) << error.message;
  for (uint32_t replacement : {6u, boundary - 1}) {
    const FailureOr<LdsResult> result =
        apply_lds_wrong_address(context, exact.value(), replacement, error);
    ASSERT_TRUE(result.succeeded()) << error.message;
    const FailureOr<LdsResult> by_ordinal =
        apply_lds_wrong_address(context, selection.value(), replacement, error);
    ASSERT_TRUE(by_ordinal.succeeded()) << error.message;
    EXPECT_EQ(result.value().image, by_ordinal.value().image);
    for (const std::vector<std::string> &selector :
         {std::vector<std::string>{"--ordinal", "0"},
          std::vector<std::string>{"--exact", "0", "--ordinal", "999"}}) {
      std::vector<std::string> command = args();
      command.insert(command.end(),
                     {"--output", output_.string(), "--address-vgpr", std::to_string(replacement)});
      command.insert(command.end(), selector.begin(), selector.end());
      ASSERT_EQ(run(command), 0);
      const std::vector<uint8_t> emitted = read();
      EXPECT_EQ(emitted, result.value().image);
      EXPECT_TRUE(test::lds_form_output_matches(context, fixture, 0, replacement, emitted));
      ASSERT_TRUE(std::filesystem::remove(output_));
    }
  }
  for (const std::vector<std::string> &extra :
       {std::vector<std::string>{"--exact", "4", "--ordinal", "0", "--address-vgpr", "6"},
        std::vector<std::string>{}, std::vector<std::string>{"--address-vgpr", "2"},
        std::vector<std::string>{"--address-vgpr", std::to_string(boundary)},
        std::vector<std::string>{"--address-vgpr", "256"}}) {
    std::vector<std::string> command = args();
    command.insert(command.end(), {"--output", output_.string()});
    command.insert(command.end(), extra.begin(), extra.end());
    EXPECT_NE(run(command), 0);
    EXPECT_FALSE(std::filesystem::exists(output_));
  }
  EXPECT_TRUE(std::ranges::equal(context.image(), image));
}

INSTANTIATE_TEST_SUITE_P(TargetFormWaves, FaultLdsCliSubword,
                         ::testing::ValuesIn(subword_cli_cases()),
                         [](const ::testing::TestParamInfo<SubwordCliCase> &info) {
                           const uint32_t wave = info.param.fixture.arch == IsaFamily::Cdna5
                                                     ? 32u
                                                     : (info.param.wave32 ? 32u : 64u);
                           return "T" +
                                  std::to_string(static_cast<int>(info.param.fixture.target)) +
                                  "_" + std::string(info.param.fixture.form) + "_Wave" +
                                  std::to_string(wave);
                         });

} // namespace
} // namespace rocjitsu
