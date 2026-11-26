// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "atomic_cdna_exchange_fixture.h"
#include "atomic_ds_fixture.h"
#include "atomic_fixture.h"
#include "atomic_native_inventory_fixture.h"
#include "cli_test_support.h"
#include "rocjitsu/code/file_io.h"
#include "rocjitsu/code/patch/fault_injection/atomic_wrong_address.h"
#include "rocjitsu/isa/target_registry.h"
#include "scoped_temp.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
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
#include <vector>

namespace rocjitsu {
namespace {

class FaultAtomicCli : public ::testing::Test {
protected:
  test::ScopedTempDirectory temporary_;
  std::filesystem::path root_, input_, output_;
  std::string cli_, stdout_, stderr_;

  void SetUp() override {
    ASSERT_TRUE(test::mutation_cli_path(cli_));
    ASSERT_TRUE(temporary_.initialize("atomic-cli-"));
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
            std::string(kernel), "--kind",        "AtomicWrongAddress"};
  }

  std::vector<std::string> mutation_args(std::vector<std::string> selector = {},
                                         std::string_view kernel = "test_kernel") const {
    std::vector<std::string> command = args(kernel);
    command.insert(command.end(), {"--output", output_.string(), "--address-delta", "8"});
    command.insert(command.end(), selector.begin(), selector.end());
    return command;
  }
};

TEST_F(FaultAtomicCli, ModernTargetWavesExactOrdinalAndLibraryAgree) {
  for (const test::AtomicTargetCase &target : test::kAtomicTargetCases) {
    SCOPED_TRACE(target.name);
    SCOPED_TRACE(target.wave);
    const IsaTargetBinding *description = default_isa_target_registry().find_target(target.target);
    ASSERT_NE(description, nullptr);
    for (test::AtomicTestForm form :
         {test::AtomicTestForm::FlatAdd, test::AtomicTestForm::GlobalCas}) {
      SCOPED_TRACE(static_cast<int>(form));
      const std::vector<uint8_t> image = test::make_atomic_elf(target, form, -4);
      ASSERT_NO_FATAL_FAILURE(write(input_, image));
      const FaultMutationContext context(image);
      FaultMutationError error;
      const FailureOr<AtomicInventory> inventory =
          discover_atomic_wrong_address(context, "test_kernel", error);
      ASSERT_TRUE(inventory.succeeded()) << error.message;
      ASSERT_EQ(inventory.value().sites().size(), 1u);
      const FailureOr<AtomicSelection> selected =
          select_atomic_wrong_address(inventory.value(), {}, error);
      ASSERT_TRUE(selected.succeeded()) << error.message;
      const FailureOr<AtomicResult> expected =
          apply_atomic_wrong_address(context, selected.value(), 8, error);
      ASSERT_TRUE(expected.succeeded()) << error.message;

      std::vector<std::string> list = args();
      list.push_back("--list");
      ASSERT_EQ(run(list), 0) << stderr_;
      EXPECT_NE(stdout_.find("ordinal=0 exact=0"), std::string::npos);
      EXPECT_NE(stdout_.find("address_supported=1"), std::string::npos);
      for (const std::vector<std::string> &selector :
           {std::vector<std::string>{"--ordinal", "0"},
            std::vector<std::string>{"--exact", "0x0", "--ordinal", "999"}}) {
        std::vector<std::string> command = mutation_args(selector);
        command.insert(command.end(), {"--target", std::string(description->id)});
        ASSERT_EQ(run(command), 0) << stderr_;
        const std::vector<uint8_t> actual = read(output_);
        EXPECT_EQ(actual, expected.value().image);
        EXPECT_TRUE(test::atomic_output_matches(context, 0, 8, actual));
        EXPECT_EQ(read(input_), image);
        ASSERT_TRUE(std::filesystem::remove(output_));
      }
    }
  }
}

TEST_F(FaultAtomicCli, ModernBufferTargetsExactOrdinalAndLibraryAgree) {
  for (const test::AtomicTargetCase &target : test::kAtomicTargetCases) {
    SCOPED_TRACE(target.name);
    const std::vector<uint8_t> image = test::make_mixed_atomic_elf(target);
    ASSERT_NO_FATAL_FAILURE(write(input_, image));
    const FaultMutationContext context(image);
    FaultMutationError error;
    const FailureOr<AtomicInventory> inventory =
        discover_atomic_wrong_address(context, "test_kernel", error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    ASSERT_EQ(inventory.value().sites().size(), 3u);
    const FailureOr<AtomicSelection> selected =
        select_atomic_wrong_address(inventory.value(), {}, error);
    ASSERT_TRUE(selected.succeeded()) << error.message;
    EXPECT_EQ(selected.value().site().form, "buffer_atomic_add_u32");
    const FailureOr<AtomicResult> expected =
        apply_atomic_wrong_address(context, selected.value(), 8, error);
    ASSERT_TRUE(expected.succeeded()) << error.message;
    for (const std::vector<std::string> &selector :
         {std::vector<std::string>{"--ordinal", "0"},
          std::vector<std::string>{"--exact", "0x0", "--ordinal", "999"}}) {
      ASSERT_EQ(run(mutation_args(selector)), 0) << stderr_;
      const std::vector<uint8_t> actual = read(output_);
      EXPECT_EQ(actual, expected.value().image);
      EXPECT_TRUE(test::atomic_output_matches(context, 0, 8, actual));
      EXPECT_EQ(read(input_), image);
      ASSERT_TRUE(std::filesystem::remove(output_));
    }
  }
}

TEST_F(FaultAtomicCli, NativeCdnaDsExchangesMatchExactOrdinalLibraryAndFullGuestOracle) {
  for (const test::AtomicTargetCase &target : test::kNativeCdnaExchangeCases) {
    SCOPED_TRACE(target.name);
    for (bool modifiers : {false, true}) {
      SCOPED_TRACE(modifiers);
      const std::vector<uint8_t> image =
          test::make_mixed_native_cdna_exchange_elf(target, modifiers);
      ASSERT_NO_FATAL_FAILURE(write(input_, image));
      const FaultMutationContext source(image);
      FaultMutationError error;
      const FailureOr<AtomicInventory> inventory =
          discover_atomic_wrong_address(source, "test_kernel", error);
      ASSERT_TRUE(inventory.succeeded()) << error.message;
      ASSERT_EQ(inventory.value().sites().size(), 5u);
      std::vector<std::string> list = args();
      list.push_back("--list");
      ASSERT_EQ(run(list), 0) << stderr_;
      EXPECT_NE(stdout_.find("ordinal=0 exact=0 size=8 byte_offset=8 address_supported=0"),
                std::string::npos);
      EXPECT_NE(stdout_.find("ordinal=1 exact=8 size=8 byte_offset=4 address_supported=0"),
                std::string::npos);
      EXPECT_NE(stdout_.find("ordinal=2 exact=16 size=8 byte_offset=4 address_supported=1"),
                std::string::npos);
      for (size_t ordinal = 0; ordinal < 5; ++ordinal) {
        SCOPED_TRACE(ordinal);
        const FailureOr<AtomicSelection> selected = select_atomic_wrong_address(
            inventory.value(), {.exact = std::nullopt, .ordinal = ordinal}, error);
        ASSERT_TRUE(selected.succeeded()) << error.message;
        if (ordinal < 2) {
          for (const std::vector<std::string> &selector :
               {std::vector<std::string>{"--ordinal", std::to_string(ordinal)},
                std::vector<std::string>{"--exact", std::to_string(ordinal * 8), "--ordinal",
                                         "999"}}) {
            EXPECT_NE(run(mutation_args(selector)), 0);
            EXPECT_NE(stderr_.find("UnsupportedForm"), std::string::npos) << stderr_;
            EXPECT_FALSE(std::filesystem::exists(output_));
          }
          continue;
        }
        const FailureOr<AtomicResult> expected =
            apply_atomic_wrong_address(source, selected.value(), 8, error);
        ASSERT_TRUE(expected.succeeded()) << error.message;
        for (const std::vector<std::string> &selector :
             {std::vector<std::string>{"--ordinal", std::to_string(ordinal)},
              std::vector<std::string>{"--exact", std::to_string(ordinal * 8), "--ordinal",
                                       "999"}}) {
          ASSERT_EQ(run(mutation_args(selector)), 0) << stderr_;
          const std::vector<uint8_t> actual = read(output_);
          EXPECT_EQ(actual, expected.value().image);
          EXPECT_TRUE(test::atomic_ds_output_matches(source, ordinal * 8, 8, actual));
          EXPECT_EQ(read(input_), image);
          ASSERT_TRUE(std::filesystem::remove(output_));
        }
      }
      EXPECT_NE(run(mutation_args({"--exact", "4", "--ordinal", "2"})), 0);
      EXPECT_NE(stderr_.find("InvalidParameters"), std::string::npos);
      EXPECT_FALSE(std::filesystem::exists(output_));
      EXPECT_EQ(read(input_), image);
    }
  }
}

TEST_F(FaultAtomicCli, NegativeBufferSourceKeepsOrdinalAndNeverPublishes) {
  for (const test::AtomicTargetCase &target : test::kAtomicTargetCases) {
    SCOPED_TRACE(target.name);
    std::vector<uint8_t> image = test::make_mixed_atomic_elf(target);
    const FaultMutationContext original(image);
    const Section &text = *original.source().text_sections().front();
    const uint64_t immediate_word = text.sectionOffset() + 2 * sizeof(uint32_t);
    uint32_t word = 0;
    std::memcpy(&word, image.data() + immediate_word, sizeof(word));
    word = (word & 0xffu) | (0xfffffcu << 8);
    std::memcpy(image.data() + immediate_word, &word, sizeof(word));
    ASSERT_NO_FATAL_FAILURE(write(input_, image));
    const FaultMutationContext source(image);
    FaultMutationError error;
    const FailureOr<AtomicInventory> inventory =
        discover_atomic_wrong_address(source, "test_kernel", error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    ASSERT_EQ(inventory.value().sites().size(), 3u);
    EXPECT_EQ(inventory.value().sites()[0].byte_offset, -4);
    EXPECT_FALSE(inventory.value().sites()[0].address_supported);
    std::vector<std::string> list = args();
    list.push_back("--list");
    ASSERT_EQ(run(list), 0) << stderr_;
    EXPECT_NE(stdout_.find("ordinal=0 exact=0 size=12 byte_offset=-4 address_supported=0"),
              std::string::npos);
    EXPECT_NE(stdout_.find("ordinal=2 exact=20"), std::string::npos);
    for (const std::vector<std::string> &selector :
         {std::vector<std::string>{"--ordinal", "0"},
          std::vector<std::string>{"--exact", "0", "--ordinal", "2"}}) {
      EXPECT_NE(run(mutation_args(selector)), 0);
      EXPECT_NE(stderr_.find("UnsupportedForm"), std::string::npos);
      EXPECT_FALSE(std::filesystem::exists(output_));
      EXPECT_EQ(read(input_), image);
    }
    const FailureOr<AtomicSelection> later = select_atomic_wrong_address(
        inventory.value(), {.exact = std::nullopt, .ordinal = 2}, error);
    ASSERT_TRUE(later.succeeded()) << error.message;
    const FailureOr<AtomicResult> expected =
        apply_atomic_wrong_address(source, later.value(), 8, error);
    ASSERT_TRUE(expected.succeeded()) << error.message;
    ASSERT_EQ(run(mutation_args({"--ordinal", "2"})), 0) << stderr_;
    EXPECT_EQ(read(output_), expected.value().image);
    EXPECT_TRUE(test::atomic_output_matches(source, 20, 8, read(output_)));
    EXPECT_EQ(read(input_), image);
    ASSERT_TRUE(std::filesystem::remove(output_));
    EXPECT_NE(run(mutation_args({"--exact", "4", "--ordinal", "2"})), 0);
    EXPECT_FALSE(std::filesystem::exists(output_));
  }
}

TEST_F(FaultAtomicCli, SevenTargetDsExactOrdinalAndLibraryAgree) {
  for (const test::AtomicTargetCase &target : test::kAtomicDsTargetCases) {
    SCOPED_TRACE(target.name);
    const std::vector<uint8_t> image =
        test::make_atomic_ds_elf(target, test::AtomicDsForm::AddReturning, 4, 2, true);
    ASSERT_NO_FATAL_FAILURE(write(input_, image));
    const FaultMutationContext context(image);
    FaultMutationError error;
    const FailureOr<AtomicInventory> inventory =
        discover_atomic_wrong_address(context, "test_kernel", error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    ASSERT_EQ(inventory.value().sites().size(), 1u);
    const FailureOr<AtomicSelection> selected =
        select_atomic_wrong_address(inventory.value(), {}, error);
    ASSERT_TRUE(selected.succeeded()) << error.message;
    const FailureOr<AtomicResult> expected =
        apply_atomic_wrong_address(context, selected.value(), 8, error);
    ASSERT_TRUE(expected.succeeded()) << error.message;
    for (const std::vector<std::string> &selector :
         {std::vector<std::string>{"--ordinal", "0"},
          std::vector<std::string>{"--exact", "0x0", "--ordinal", "999"}}) {
      ASSERT_EQ(run(mutation_args(selector)), 0) << stderr_;
      const std::vector<uint8_t> actual = read(output_);
      EXPECT_EQ(actual, expected.value().image);
      EXPECT_TRUE(test::atomic_ds_output_matches(context, 0, 8, actual));
      EXPECT_EQ(read(input_), image);
      ASSERT_TRUE(std::filesystem::remove(output_));
    }
    const std::vector<uint8_t> wide = test::make_atomic_ds_elf(target, test::AtomicDsForm::WideAdd);
    ASSERT_NO_FATAL_FAILURE(write(input_, wide));
    for (const std::vector<std::string> &selector :
         {std::vector<std::string>{"--ordinal", "0"},
          std::vector<std::string>{"--exact", "0x0", "--ordinal", "999"}}) {
      EXPECT_NE(run(mutation_args(selector)), 0);
      EXPECT_NE(stderr_.find("UnsupportedForm"), std::string::npos);
      EXPECT_FALSE(std::filesystem::exists(output_));
      EXPECT_EQ(read(input_), wide);
    }
  }
}

TEST_F(FaultAtomicCli, DsAdmissionKeepsTheMixedAtomicSourceOrder) {
  const test::AtomicTargetCase &target = test::kAtomicTargetCases.front();
  const std::vector<uint8_t> image = test::make_mixed_atomic_elf(target);
  ASSERT_NO_FATAL_FAILURE(write(input_, image));
  const FaultMutationContext context(image);
  FaultMutationError error;
  const FailureOr<AtomicInventory> inventory =
      discover_atomic_wrong_address(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().sites().size(), 3u);
  EXPECT_EQ(inventory.value().sites()[0].offset, 0u);
  EXPECT_EQ(inventory.value().sites()[1].offset, 12u);
  EXPECT_EQ(inventory.value().sites()[2].offset, 20u);
  std::vector<std::string> list = args();
  list.push_back("--list");
  ASSERT_EQ(run(list), 0) << stderr_;
  const size_t buffer = stdout_.find("ordinal=0 exact=0");
  const size_t ds = stdout_.find("ordinal=1 exact=12");
  const size_t flat = stdout_.find("ordinal=2 exact=20");
  ASSERT_NE(buffer, std::string::npos);
  ASSERT_NE(ds, std::string::npos);
  ASSERT_NE(flat, std::string::npos);
  EXPECT_LT(buffer, ds);
  EXPECT_LT(ds, flat);
  for (size_t ordinal : {size_t{1}, size_t{2}}) {
    const FailureOr<AtomicSelection> selected = select_atomic_wrong_address(
        inventory.value(), {.exact = std::nullopt, .ordinal = ordinal}, error);
    ASSERT_TRUE(selected.succeeded()) << error.message;
    const FailureOr<AtomicResult> expected =
        apply_atomic_wrong_address(context, selected.value(), 8, error);
    ASSERT_TRUE(expected.succeeded()) << error.message;
    for (const std::vector<std::string> &selector :
         {std::vector<std::string>{"--ordinal", std::to_string(ordinal)},
          std::vector<std::string>{"--exact", std::to_string(selected.value().site().offset),
                                   "--ordinal", "999"}}) {
      ASSERT_EQ(run(mutation_args(selector)), 0) << stderr_;
      const std::vector<uint8_t> actual = read(output_);
      EXPECT_EQ(actual, expected.value().image);
      EXPECT_TRUE(ordinal == 1 ? test::atomic_ds_output_matches(context, 12, 8, actual)
                               : test::atomic_output_matches(context, 20, 8, actual));
      EXPECT_EQ(read(input_), image);
      ASSERT_TRUE(std::filesystem::remove(output_));
    }
  }
  EXPECT_NE(run(mutation_args({"--exact", "4", "--ordinal", "2"})), 0);
  EXPECT_FALSE(std::filesystem::exists(output_));
  EXPECT_EQ(read(input_), image);
}

TEST_F(FaultAtomicCli, InvalidRequestsDoNotPublishOrChangeSource) {
  const test::AtomicTargetCase &target = test::kAtomicTargetCases.front();
  const std::vector<uint8_t> image = test::make_atomic_elf(target);
  ASSERT_NO_FATAL_FAILURE(write(input_, image));
  for (const std::vector<std::string> &extra :
       {std::vector<std::string>{}, std::vector<std::string>{"--address-delta", "0"},
        std::vector<std::string>{"--address-delta", "2"},
        std::vector<std::string>{"--address-delta", "8388608"},
        std::vector<std::string>{"--address-delta", "4294967296"},
        std::vector<std::string>{"--address-delta", "-4"},
        std::vector<std::string>{"--address-delta", "8", "--address-delta", "4"},
        std::vector<std::string>{"--address-delta", "8", "--exact", "4", "--ordinal", "0"},
        std::vector<std::string>{"--address-delta", "8", "--ordinal", "999"},
        std::vector<std::string>{"--address-delta", "8", "--target", "gfx942"},
        std::vector<std::string>{"--address-delta", "8", "--address-vgpr", "6"}}) {
    std::vector<std::string> command = args();
    command.insert(command.end(), {"--output", output_.string()});
    command.insert(command.end(), extra.begin(), extra.end());
    EXPECT_NE(run(command), 0);
    EXPECT_FALSE(std::filesystem::exists(output_));
    EXPECT_EQ(read(input_), image);
  }
  for (const char *kind : {"DropBarrier", "LdsWrongAddress"}) {
    std::vector<std::string> command = {
        "--input", input_.string(), "--kernel",       "test_kernel",     "--kind",
        kind,      "--output",      output_.string(), "--address-delta", "8"};
    EXPECT_NE(run(command), 0);
    EXPECT_FALSE(std::filesystem::exists(output_));
  }
  std::vector<std::string> invalid_list = args();
  invalid_list.insert(invalid_list.end(), {"--list", "--address-delta", "8"});
  EXPECT_NE(run(invalid_list), 0);
  EXPECT_FALSE(std::filesystem::exists(output_));
  EXPECT_EQ(read(input_), image);
  const std::vector<uint8_t> near_limit =
      test::make_atomic_elf(target, test::AtomicTestForm::FlatAdd, 0x7ffffc);
  ASSERT_NO_FATAL_FAILURE(write(input_, near_limit));
  EXPECT_NE(run(mutation_args()), 0);
  EXPECT_FALSE(std::filesystem::exists(output_));
  EXPECT_EQ(read(input_), near_limit);
}

TEST_F(FaultAtomicCli, ExistingOutputAndInputAliasesRemainUnchanged) {
  const std::vector<uint8_t> image = test::make_atomic_elf(test::kAtomicTargetCases.front());
  ASSERT_NO_FATAL_FAILURE(write(input_, image));
  constexpr std::array<uint8_t, 4> sentinel = {1, 2, 3, 4};
  ASSERT_NO_FATAL_FAILURE(write(output_, sentinel));
  EXPECT_NE(run(mutation_args()), 0);
  EXPECT_EQ(read(output_), std::vector<uint8_t>(sentinel.begin(), sentinel.end()));
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
}

TEST_F(FaultAtomicCli, SharedExactSiteRemainsBoundToRequestedOwner) {
  const test::AtomicTargetCase &target = test::kAtomicTargetCases.front();
  const std::vector<uint8_t> image = test::make_shared_atomic_elf(target);
  ASSERT_NO_FATAL_FAILURE(write(input_, image));
  const FaultMutationContext context(image);
  for (const char *owner : {"test_kernel0", "test_kernel1"}) {
    ASSERT_EQ(run(mutation_args({"--exact", "512", "--ordinal", "999"}, owner)), 0) << stderr_;
    EXPECT_TRUE(test::atomic_output_matches(context, 512, 8, read(output_)));
    EXPECT_NE(stdout_.find("affected_owners=2"), std::string::npos);
    ASSERT_TRUE(std::filesystem::remove(output_));
  }
  EXPECT_NE(run(mutation_args({"--exact", "512"}, "test_kernel")), 0);
  EXPECT_FALSE(std::filesystem::exists(output_));
  const std::vector<uint8_t> unshared = test::make_shared_atomic_elf(target, false, false);
  ASSERT_NO_FATAL_FAILURE(write(input_, unshared));
  EXPECT_NE(run(mutation_args({"--exact", "512", "--ordinal", "0"}, "test_kernel1")), 0);
  EXPECT_FALSE(std::filesystem::exists(output_));
  EXPECT_EQ(read(input_), unshared);
}

TEST_F(FaultAtomicCli, NativeCdnaExchangeInventoryKeepsUnsupportedExactAndOrdinalSites) {
  for (const test::AtomicTargetCase &target : test::kNativeCdnaExchangeCases) {
    SCOPED_TRACE(target.name);
    const std::vector<uint8_t> image = test::make_native_cdna_exchange_inventory_elf(target);
    ASSERT_NO_FATAL_FAILURE(write(input_, image));
    std::vector<std::string> list = args();
    list.push_back("--list");
    ASSERT_EQ(run(list), 0) << stderr_;
    for (size_t ordinal = 0; ordinal < test::kNativeCdnaExchangeInventoryForms.size(); ++ordinal) {
      const std::string identity =
          "ordinal=" + std::to_string(ordinal) + " exact=" + std::to_string(ordinal * 8) + " ";
      const size_t position = stdout_.find(identity);
      ASSERT_NE(position, std::string::npos) << stdout_;
      const size_t end = stdout_.find('\n', position);
      ASSERT_NE(end, std::string::npos);
      const std::string_view line(stdout_.data() + position, end - position);
      EXPECT_NE(line.find(test::native_cdna_exchange_mnemonic(
                    test::kNativeCdnaExchangeInventoryForms[ordinal])),
                std::string::npos);
    }
    for (size_t ordinal : {0u, 3u, 4u, 5u, 6u}) {
      for (const std::vector<std::string> &selector :
           {std::vector<std::string>{"--ordinal", std::to_string(ordinal)},
            std::vector<std::string>{"--exact", std::to_string(ordinal * 8), "--ordinal", "999"}}) {
        EXPECT_NE(run(mutation_args(selector)), 0);
        EXPECT_NE(stderr_.find("UnsupportedForm"), std::string::npos) << stderr_;
        EXPECT_FALSE(std::filesystem::exists(output_));
        EXPECT_EQ(read(input_), image);
      }
    }
    EXPECT_NE(run(mutation_args({"--exact", "4", "--ordinal", "6"})), 0);
    EXPECT_NE(stderr_.find("InvalidParameters"), std::string::npos);
    EXPECT_FALSE(std::filesystem::exists(output_));
    EXPECT_NE(run(mutation_args({"--ordinal", "7"})), 0);
    EXPECT_NE(stderr_.find("OrdinalOutOfRange"), std::string::npos);
    EXPECT_FALSE(std::filesystem::exists(output_));
    EXPECT_EQ(read(input_), image);
  }
}

TEST_F(FaultAtomicCli, NativeSemanticInventoryKeepsUnsupportedOrdinalsBeforeFlat) {
  for (const test::AtomicTargetCase &target : test::kNativeAtomicInventoryCases) {
    SCOPED_TRACE(target.name);
    const std::vector<test::NativeAtomicInventorySample> samples =
        test::native_atomic_inventory_samples(target);
    const std::vector<uint8_t> image = test::make_native_atomic_inventory_elf(target);
    ASSERT_NO_FATAL_FAILURE(write(input_, image));
    std::vector<std::string> list = args();
    list.push_back("--list");
    ASSERT_EQ(run(list), 0) << stderr_;
    uint64_t offset = 0;
    for (size_t ordinal = 0; ordinal < samples.size(); ++ordinal) {
      const std::string identity =
          "ordinal=" + std::to_string(ordinal) + " exact=" + std::to_string(offset) + " ";
      const size_t position = stdout_.find(identity);
      ASSERT_NE(position, std::string::npos) << stdout_;
      const size_t end = stdout_.find('\n', position);
      ASSERT_NE(end, std::string::npos);
      const std::string_view line(stdout_.data() + position, end - position);
      EXPECT_NE(line.find(samples[ordinal].form), std::string::npos);
      EXPECT_NE(line.find("address_supported=0"), std::string::npos);
      offset += samples[ordinal].words.size() * sizeof(uint32_t);
    }
    for (const std::vector<std::string> &selector :
         {std::vector<std::string>{"--ordinal", "0"},
          std::vector<std::string>{"--exact", "0", "--ordinal", "999"}}) {
      EXPECT_NE(run(mutation_args(selector)), 0);
      EXPECT_NE(stderr_.find("UnsupportedForm"), std::string::npos) << stderr_;
      EXPECT_FALSE(std::filesystem::exists(output_));
      EXPECT_EQ(read(input_), image);
    }
    if (!test::native_atomic_inventory_has_flat(target))
      continue;
    const uint64_t flat_offset = offset + sizeof(uint32_t);
    const FaultMutationContext source(image);
    FaultMutationError error;
    const FailureOr<AtomicInventory> inventory =
        discover_atomic_wrong_address(source, "test_kernel", error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    const FailureOr<AtomicSelection> selected = select_atomic_wrong_address(
        inventory.value(), {.exact = std::nullopt, .ordinal = samples.size()}, error);
    ASSERT_TRUE(selected.succeeded()) << error.message;
    const FailureOr<AtomicResult> expected =
        apply_atomic_wrong_address(source, selected.value(), 8, error);
    ASSERT_TRUE(expected.succeeded()) << error.message;
    for (const std::vector<std::string> &selector :
         {std::vector<std::string>{"--ordinal", std::to_string(samples.size())},
          std::vector<std::string>{"--exact", std::to_string(flat_offset), "--ordinal", "999"}}) {
      ASSERT_EQ(run(mutation_args(selector)), 0) << stderr_;
      const std::vector<uint8_t> actual = read(output_);
      EXPECT_EQ(actual, expected.value().image);
      EXPECT_TRUE(test::atomic_output_matches(source, flat_offset, 8, actual));
      EXPECT_EQ(read(input_), image);
      ASSERT_TRUE(std::filesystem::remove(output_));
    }
  }
}

TEST_F(FaultAtomicCli, HelpAndCapabilitiesMatchEnabledProfiles) {
  ASSERT_EQ(run({"--help"}), 0) << stderr_;
  EXPECT_NE(stdout_.find("AtomicWrongAddress"), std::string::npos);
  EXPECT_NE(stdout_.find("--address-delta"), std::string::npos);
  ASSERT_EQ(run({"--capabilities"}), 0) << stderr_;
  size_t atomic_lines = 0;
  for (size_t position = 0;
       (position = stdout_.find("kind=AtomicWrongAddress ", position)) != std::string::npos;
       ++position)
    ++atomic_lines;
  EXPECT_EQ(atomic_lines, 28u);
  for (const test::AtomicTargetCase &target : test::kAtomicTargetCases) {
    const IsaTargetBinding *description = default_isa_target_registry().find_target(target.target);
    ASSERT_NE(description, nullptr);
    const std::string prefix = "kind=AtomicWrongAddress target=" + std::string(description->id) +
                               " wave=" + std::to_string(target.wave) + " form=";
    EXPECT_NE(stdout_.find(prefix + "flat_atomic"), std::string::npos);
    EXPECT_NE(stdout_.find(prefix + "global_atomic"), std::string::npos);
    EXPECT_NE(stdout_.find(prefix + "buffer_atomic"), std::string::npos);
    EXPECT_NE(stdout_.find(prefix + "ds_atomic"), std::string::npos);
  }
}

} // namespace
} // namespace rocjitsu
