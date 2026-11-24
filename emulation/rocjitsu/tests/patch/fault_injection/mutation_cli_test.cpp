// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "barrier_fixture.h"
#include "barrier_oracle.h"
#include "cdna_lds_fixture.h"
#include "lds_fixture.h"
#include "remaining_subword_fixture.h"
#include "rocjitsu/code/file_io.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/fault_injection/drop_barrier.h"
#include "rocjitsu/code/patch/fault_injection/lds_wrong_address.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "scoped_temp.h"
#include "subword_fixture.h"
#include "two_address_fixture.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <format>
#include <fstream>
#include <ios>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <sys/resource.h>
#include <sys/wait.h>
#include <system_error>
#include <unistd.h>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace {
using test::checked_encoding;

class FaultMutationCli : public ::testing::Test {
protected:
  test::ScopedTempDirectory directory_;
  std::filesystem::path root_, input_, output_;
  std::string cli_, stdout_, stderr_;

  void SetUp() override {
    std::error_code error;
    const std::filesystem::path executable = std::filesystem::read_symlink("/proc/self/exe", error);
    ASSERT_FALSE(error) << error.message();
    // Relocated tests must use their package even while the build tree exists.
    cli_ = executable == RJ_CODE_MUTATE_TEST_PATH
               ? RJ_CODE_MUTATE_PATH
               : (executable.parent_path() / RJ_INSTALLED_CODE_MUTATE_PATH)
                     .lexically_normal()
                     .string();
    ASSERT_TRUE(std::filesystem::is_regular_file(cli_));
    ASSERT_TRUE(directory_.initialize("fault-cli-"));
    root_ = directory_.path();
    input_ = root_ / "original.hsaco";
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
    const util::FailureOr<std::vector<char>> bytes = detail::read_file_bytes(path.string());
    EXPECT_TRUE(bytes.succeeded()) << path;
    return bytes.succeeded() ? std::vector<uint8_t>(bytes.value().begin(), bytes.value().end())
                             : std::vector<uint8_t>{};
  }
  enum class StreamFailure { None, FullDevice, ClosedPipe };
  struct InvokeOptions {
    bool limit_write = false;
    StreamFailure stdout_failure = StreamFailure::None;
    StreamFailure stderr_failure = StreamFailure::None;
  };
  static int open_capture(const std::filesystem::path &path, StreamFailure failure) {
    if (failure == StreamFailure::ClosedPipe) {
      int descriptors[2] = {};
      if (pipe(descriptors) != 0) {
        return -1;
      }
      close(descriptors[0]);
      return descriptors[1];
    }
    return failure == StreamFailure::FullDevice
               ? open("/dev/full", O_WRONLY)
               : open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
  }
  int invoke(std::vector<std::string> arguments, InvokeOptions options) {
    arguments.insert(arguments.begin(), cli_);
    std::vector<char *> pointers;
    for (std::string &argument : arguments) {
      pointers.push_back(argument.data());
    }
    pointers.push_back(nullptr);
    const int out = open_capture(root_ / "stdout", options.stdout_failure);
    const int err = open_capture(root_ / "stderr", options.stderr_failure);
    EXPECT_GE(out, 0);
    EXPECT_GE(err, 0);
    const pid_t child = fork();
    if (child == 0) {
      if (chdir(root_.c_str()) != 0 || dup2(out, STDOUT_FILENO) < 0 ||
          dup2(err, STDERR_FILENO) < 0) {
        _exit(126);
      }
      close(out);
      close(err);
      // Exercise the executable's signal policy, independent of the test runner.
      if (std::signal(SIGPIPE, SIG_DFL) == SIG_ERR || std::signal(SIGXFSZ, SIG_DFL) == SIG_ERR) {
        _exit(126);
      }
      if (options.limit_write) {
        const rlimit core_limit = {.rlim_cur = 0, .rlim_max = 0};
        const rlimit limit = {.rlim_cur = 512, .rlim_max = 512};
        if (setrlimit(RLIMIT_CORE, &core_limit) != 0 || setrlimit(RLIMIT_FSIZE, &limit) != 0) {
          _exit(126);
        }
      }
      execv(pointers.front(), pointers.data());
      _exit(127);
    }
    close(out);
    close(err);
    if (child < 0) {
      return -1;
    }
    int status = 0;
    if (waitpid(child, &status, 0) != child) {
      return -1;
    }
    const std::vector<uint8_t> out_bytes = options.stdout_failure == StreamFailure::None
                                               ? read(root_ / "stdout")
                                               : std::vector<uint8_t>{};
    const std::vector<uint8_t> err_bytes = options.stderr_failure == StreamFailure::None
                                               ? read(root_ / "stderr")
                                               : std::vector<uint8_t>{};
    stdout_.assign(out_bytes.begin(), out_bytes.end());
    stderr_.assign(err_bytes.begin(), err_bytes.end());
    if (WIFEXITED(status)) {
      return WEXITSTATUS(status);
    }
    return WIFSIGNALED(status) ? 128 + WTERMSIG(status) : -1;
  }
  int invoke(std::vector<std::string> arguments) { return invoke(std::move(arguments), {}); }
  std::vector<std::string> args(std::vector<std::string> extra = {}) {
    std::vector<std::string> result{"--input",     input_.string(), "--kernel",
                                    "test_kernel", "--output",      output_.string()};
    result.insert(result.end(), extra.begin(), extra.end());
    return result;
  }
  static void expect_complete_output(std::span<const uint8_t> image,
                                     const std::vector<uint8_t> &candidate,
                                     std::string_view entry = "test_kernel") {
    const FaultMutationContext context(image);
    FaultMutationError error;
    const FailureOr<FaultBarrierInventory> inventory =
        discover_drop_barriers(context, entry, error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    const FailureOr<DropBarrierSelection> selection =
        select_drop_barrier(inventory.value(), {}, error);
    ASSERT_TRUE(selection.succeeded()) << error.message;
    const std::optional<std::vector<uint8_t>> expected =
        test::expected_drop_image(context, selection.value());
    ASSERT_TRUE(expected);
    ASSERT_EQ(candidate.size(), expected->size());
    EXPECT_EQ(candidate, *expected);
    EXPECT_TRUE(test::matches_drop_output(context, selection.value(), candidate, error).succeeded())
        << error.message;
  }
  void expect_rejected(std::vector<std::string> arguments, int status = 2) {
    EXPECT_EQ(invoke(std::move(arguments)), status) << stderr_;
    EXPECT_FALSE(stderr_.empty());
    EXPECT_TRUE(stdout_.empty());
    EXPECT_FALSE(std::filesystem::exists(output_));
  }
};

class FaultMutationCliFamily : public FaultMutationCli,
                               public ::testing::WithParamInterface<test::DropTargetRow> {};

TEST_P(FaultMutationCliFamily, ListsSelectsAndEmitsExactLibraryBytes) {
  const test::DropTargetRow &row = GetParam();
  const std::vector<uint8_t> image =
      test::make_barrier_elf_for_target(row, test::make_barrier_words(row));
  write(input_, image);
  const FaultMutationContext context(image);
  FaultMutationError error;
  const FailureOr<FaultBarrierInventory> inventory =
      discover_drop_barriers(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().barriers().size(), 2u);
  ASSERT_EQ(invoke({"--input", input_.string(), "--kernel", "test_kernel", "--list"}), 0)
      << stderr_;
  for (size_t ordinal = 0; ordinal < 2; ++ordinal) {
    const StaticBarrier &barrier = inventory.value().barriers()[ordinal];
    EXPECT_EQ(barrier.members.size(), row.split ? 2u : 1u);
    std::string members;
    for (const StaticBarrierMember &member : barrier.members) {
      if (!members.empty()) {
        members += ',';
      }
      members += std::to_string(member.offset) + ":4";
    }
    EXPECT_NE(stdout_.find("ordinal=" + std::to_string(ordinal) +
                           " exact=" + std::to_string(barrier.members.front().offset)),
              std::string::npos);
    EXPECT_NE(stdout_.find("members=" + members + "\n"), std::string::npos);
  }
  const std::string target(default_isa_target_registry().find_target(row.target)->id);
  for (const std::string mode : {"default", "ordinal", "exact"}) {
    SCOPED_TRACE(mode);
    const size_t ordinal = mode == "default" ? 0 : 1;
    const StaticBarrier &barrier = inventory.value().barriers()[ordinal];
    std::vector<std::string> options{"--kind", "DropBarrier", "--target", target};
    if (mode == "ordinal") {
      options.insert(options.end(), {"--ordinal", "1"});
    }
    if (mode == "exact") {
      options.insert(options.end(),
                     {"--exact", std::format("{:#x}", barrier.members.front().offset), "--ordinal",
                      "18446744073709551615"});
    }
    ASSERT_EQ(invoke(args(options)), 0) << stderr_;
    const std::vector<uint8_t> mutant = read(output_);
    const std::string expected_stdout = std::format(
        "kind=DropBarrier target={} exact={} logical_mutations=1 members={} output_bytes={}\n",
        target, barrier.members.front().offset, barrier.members.size(), mutant.size());
    EXPECT_EQ(stdout_, expected_stdout);
    const FailureOr<DropBarrierSelection> selection =
        select_drop_barrier(inventory.value(), {.exact = std::nullopt, .ordinal = ordinal}, error);
    ASSERT_TRUE(selection.succeeded());
    const FailureOr<DropBarrierResult> library =
        apply_drop_barrier(context, selection.value(), error);
    ASSERT_TRUE(library.succeeded()) << error.message;
    std::string expected_stderr;
    for (const std::string &assumption : library.value().assumptions) {
      expected_stderr += "rj_code_mutate: assumption: " + assumption + '\n';
    }
    EXPECT_EQ(stderr_, expected_stderr);
    EXPECT_EQ(mutant, library.value().image);
    const std::optional<std::vector<uint8_t>> expected =
        test::expected_drop_image(context, selection.value());
    ASSERT_TRUE(expected);
    ASSERT_EQ(mutant.size(), expected->size());
    EXPECT_EQ(mutant, *expected);
    EXPECT_TRUE(test::matches_drop_output(context, selection.value(), mutant, error).succeeded())
        << error.message;
    const FaultMutationContext fresh(mutant);
    EXPECT_EQ(fresh.source().target_id(), row.target);
    const FailureOr<FaultBarrierInventory> remaining =
        discover_drop_barriers(fresh, "test_kernel", error);
    ASSERT_TRUE(remaining.succeeded()) << error.message;
    ASSERT_EQ(remaining.value().barriers().size(), 1u);
    EXPECT_TRUE(test::same_barrier_members(remaining.value().barriers()[0],
                                           inventory.value().barriers()[1 - ordinal]));
    std::filesystem::remove(output_);
    ASSERT_EQ(invoke(args(options)), 0) << stderr_;
    EXPECT_EQ(stdout_, expected_stdout);
    EXPECT_EQ(read(output_), mutant);
    std::filesystem::remove(output_);
  }
  EXPECT_EQ(read(input_), image);
}

INSTANTIATE_TEST_SUITE_P(TargetWaves, FaultMutationCliFamily, ::testing::ValuesIn(test::kDropRows),
                         [](const ::testing::TestParamInfo<test::DropTargetRow> &info) {
                           return info.param.name;
                         });

TEST_F(FaultMutationCli, ListsLoadAndStoreForms) {
  write(input_, test::make_lds_load_elf(EF_AMDGPU_MACH_AMDGCN_GFX1200));
  ASSERT_EQ(invoke({"--input", input_.string(), "--kernel", "test_kernel", "--kind",
                    "LdsWrongAddress", "--list"}),
            0);
  EXPECT_NE(stdout_.find(
                "ordinal=0 exact=0 address_vgpr=2 address_bank=0 target=gfx1200 form=ds_load_b32"),
            std::string::npos);
  EXPECT_NE(stdout_.find(
                "ordinal=1 exact=8 address_vgpr=1 address_bank=0 target=gfx1200 form=ds_store_b32"),
            std::string::npos);
}

TEST_F(FaultMutationCli, LdsMutationSummaryMatchesSelectedLibraryResultAndEmittedImage) {
  const std::vector<uint8_t> image =
      test::make_lds_load_elf(EF_AMDGPU_MACH_AMDGCN_GFX1200, true, 0);
  write(input_, image);
  const FaultMutationContext source(image);
  FaultMutationError error;
  const FailureOr<LdsInventory> inventory =
      discover_lds_wrong_address(source, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const FailureOr<LdsSelection> selection =
      select_lds_wrong_address(inventory.value(), {.exact = std::nullopt, .ordinal = 1}, error);
  ASSERT_TRUE(selection.succeeded()) << error.message;
  const FailureOr<LdsResult> result = apply_lds_wrong_address(source, selection.value(), 7, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  ASSERT_EQ(result.value().logical_mutation_count, 1u);
  ASSERT_EQ(invoke({"--input", input_.string(), "--output", output_.string(), "--kernel",
                    "test_kernel", "--kind", "LdsWrongAddress", "--exact",
                    std::to_string(selection.value().site().offset), "--ordinal", "999",
                    "--address-vgpr", "7"}),
            0);
  const std::vector<uint8_t> emitted = read(output_);
  ASSERT_EQ(emitted, result.value().image);
  const AmdGpuCodeObject mutant(emitted.data(), emitted.size());
  ASSERT_TRUE(mutant.is_valid());
  EXPECT_EQ(mutant.target_id(), source.source().target_id());
  EXPECT_TRUE(test::lds_output_matches(source, selection.value().site().offset, 7, emitted));
  EXPECT_EQ(stdout_, "kind=LdsWrongAddress target=gfx1200 exact=" +
                         std::to_string(selection.value().site().offset) +
                         " address_vgpr=7 logical_mutations=" +
                         std::to_string(result.value().logical_mutation_count) +
                         " output_bytes=" + std::to_string(emitted.size()) + "\n");
  EXPECT_TRUE(stderr_.empty());
}

TEST_F(FaultMutationCli, ListsCdnaAndBankedForms) {
  for (const test::CdnaLdsTarget &target : test::kCdnaLdsTargets) {
    write(input_, test::make_cdna_lds_elf(target));
    ASSERT_EQ(invoke({"--input", input_.string(), "--kernel", "test_kernel", "--kind",
                      "LdsWrongAddress", "--list"}),
              0);
    EXPECT_NE(stdout_.find("form=ds_write_b32"), std::string::npos);
    EXPECT_NE(stdout_.find("form=ds_write_b128"), std::string::npos);
  }
  constexpr std::array<uint32_t, 1> set_bank =
      cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 1});
  constexpr std::array<uint32_t, 2> load =
      cdna5::build_vds(cdna5::kDsLoadB32Vds, {.addr = 2, .vdst = 5});
  write(input_, test::make_cdna5_lds_elf({set_bank[0], load[0], load[1]}, 16));
  ASSERT_EQ(invoke({"--input", input_.string(), "--kernel", "test_kernel", "--kind",
                    "LdsWrongAddress", "--list"}),
            0);
  EXPECT_NE(stdout_.find("address_bank=1 target=gfx1250 form=ds_load_b32"), std::string::npos);
}

TEST_F(FaultMutationCli, HelpCapabilitiesAndArguments) {
  EXPECT_EQ(invoke({"--help"}), 0);
  EXPECT_NE(stdout_.find("Exact takes precedence"), std::string::npos);
  EXPECT_NE(stdout_.find("LdsWrongAddress admits gfx1200/gfx1201 wave32/wave64 scalar"),
            std::string::npos);
  EXPECT_NE(stdout_.find("allocated --address-vgpr"), std::string::npos);
  EXPECT_NE(stdout_.find("Selected subword and transpose forms are also supported."),
            std::string::npos);
  EXPECT_NE(stdout_.find("the first member of a barrier, or the LDS instruction"),
            std::string::npos);
  EXPECT_EQ(invoke({"--capabilities"}), 0);
  const std::string capabilities = stdout_;
  EXPECT_EQ(std::count(capabilities.begin(), capabilities.end(), '\n'), 215);
  EXPECT_NE(capabilities.find("kind=LdsWrongAddress target=gfx1250 wave=32 "
                              "form=ds_store_b32 admission=direct-single-owner\n"),
            std::string::npos);
  EXPECT_NE(capabilities.find("kind=LdsWrongAddress target=gfx1250 wave=32 "
                              "form=ds_load_b32 admission=direct-single-owner\n"),
            std::string::npos);
  constexpr std::string_view lds1200 = "kind=LdsWrongAddress target=gfx1200 wave=64 "
                                       "form=ds_store_b32 admission=direct-single-owner\n";
  constexpr std::string_view lds1201 = "kind=LdsWrongAddress target=gfx1201 wave=64 "
                                       "form=ds_store_b32 admission=direct-single-owner\n";
  ASSERT_NE(capabilities.find(lds1200), std::string::npos);
  ASSERT_NE(capabilities.find(lds1201), std::string::npos);
  for (const std::string target : {"gfx1200", "gfx1201"}) {
    EXPECT_NE(capabilities.find("kind=LdsWrongAddress target=" + target +
                                " wave=64 form=ds_load_b32 admission=direct-single-owner\n"),
              std::string::npos);
  }
  for (const std::string target : {"gfx1200", "gfx1201"}) {
    for (const std::string form : {"ds_store_b32", "ds_load_b32"}) {
      EXPECT_NE(capabilities.find("kind=LdsWrongAddress target=" + target +
                                  " wave=32 form=" + form + " admission=direct-single-owner\n"),
                std::string::npos);
    }
  }
  for (const std::string target : {"gfx942", "gfx950"}) {
    for (const std::string form : {"ds_write_b32", "ds_write_b128"}) {
      EXPECT_NE(capabilities.find("kind=LdsWrongAddress target=" + target +
                                  " wave=64 form=" + form + " admission=direct-single-owner\n"),
                std::string::npos);
    }
  }
  const std::vector<LdsCapability> lds_capabilities = lds_wrong_address_capabilities();
  ASSERT_EQ(lds_capabilities.size(), 205u);
  for (const LdsCapability &capability : lds_capabilities) {
    const IsaTargetBinding *target = default_isa_target_registry().find_target(capability.target);
    ASSERT_NE(target, nullptr);
    const std::string line = "kind=LdsWrongAddress target=" + std::string(target->id) +
                             " wave=" + std::to_string(capability.wave_size) +
                             " form=" + std::string(capability.form) +
                             " admission=direct-single-owner\n";
    const size_t position = capabilities.find(line);
    ASSERT_NE(position, std::string::npos) << line;
    EXPECT_EQ(capabilities.find(line, position + line.size()), std::string::npos) << line;
  }
  for (const FaultMutationCapability &capability : fault_mutation_capabilities()) {
    const std::string target(default_isa_target_registry().find_target(capability.target)->id);
    EXPECT_NE(capabilities.find("kind=DropBarrier target=" + target +
                                " wave=" + std::to_string(capability.wave_size)),
              std::string::npos);
  }
  write(input_, test::make_barrier_elf_for_target(test::kDropRows[4],
                                                  test::make_barrier_words(test::kDropRows[4])));
  for (const std::vector<std::string> &options :
       std::vector<std::vector<std::string>>{{"--kind", "MoveBarrierPair"},
                                             {"--ordinal", "-1"},
                                             {"--ordinal", "1junk"},
                                             {"--ordinal", "18446744073709551616"},
                                             {"--exact", "0x"},
                                             {"--exact", "no"},
                                             {"--kind"},
                                             {"--unknown"},
                                             {"--ordinal", "0", "--ordinal", "1"},
                                             {"--target", "gfx1251"},
                                             {"--target", "bogus"},
                                             {"--list"}}) {
    expect_rejected(args(options));
  }
  expect_rejected({});
  expect_rejected({"--input", input_.string(), "--output", output_.string()});
}

TEST_F(FaultMutationCli, EqualsSyntaxPreservesSelectionAndSummary) {
  const test::DropTargetRow &row = test::kDropRows[4];
  const std::vector<uint8_t> image =
      test::make_barrier_elf_for_target(row, test::make_barrier_words(row));
  write(input_, image);
  const FaultMutationContext context(image);
  FaultMutationError error;
  const FailureOr<FaultBarrierInventory> inventory =
      discover_drop_barriers(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().barriers().size(), 2u);
  const StaticBarrier &barrier = inventory.value().barriers()[1];
  ASSERT_EQ(barrier.members.size(), 2u);
  const std::string target(default_isa_target_registry().find_target(row.target)->id);
  ASSERT_EQ(invoke({"--input", input_.string(), "--kernel", "test_kernel", "--list"}), 0)
      << stderr_;
  const std::string listing = stdout_;
  ASSERT_EQ(invoke({"--input=" + input_.string(), "--kernel=test_kernel", "--kind=DropBarrier",
                    "--target=" + target, "--list=true"}),
            0)
      << stderr_;
  EXPECT_EQ(stdout_, listing);
  EXPECT_TRUE(stderr_.empty());
  EXPECT_FALSE(std::filesystem::exists(output_));
  const FailureOr<DropBarrierSelection> selection =
      select_drop_barrier(inventory.value(), {.exact = std::nullopt, .ordinal = 1}, error);
  ASSERT_TRUE(selection.succeeded()) << error.message;
  const FailureOr<DropBarrierResult> library =
      apply_drop_barrier(context, selection.value(), error);
  ASSERT_TRUE(library.succeeded()) << error.message;
  for (const std::string &exact : {std::to_string(barrier.members.front().offset),
                                   std::format("{:#x}", barrier.members.front().offset)}) {
    SCOPED_TRACE(exact);
    ASSERT_EQ(invoke({"--input=" + input_.string(), "--kernel=test_kernel", "--kind=DropBarrier",
                      "--target=" + target, "--output=" + output_.string(), "--exact=" + exact,
                      "--ordinal=0"}),
              0)
        << stderr_;
    const std::vector<uint8_t> mutant = read(output_);
    EXPECT_EQ(mutant, library.value().image);
    EXPECT_EQ(stdout_,
              std::format("kind=DropBarrier target={} exact={} logical_mutations=1 members={} "
                          "output_bytes={}\n",
                          target, barrier.members.front().offset, barrier.members.size(),
                          mutant.size()));
    EXPECT_TRUE(test::matches_drop_output(context, selection.value(), mutant, error).succeeded())
        << error.message;
    EXPECT_EQ(read(input_), image);
    std::filesystem::remove(output_);
  }
}

TEST_F(FaultMutationCli, SharedParserRejectsAmbiguousAndMalformedArguments) {
  const test::DropTargetRow &row = test::kDropRows[4];
  const std::vector<uint8_t> image =
      test::make_barrier_elf_for_target(row, test::make_barrier_words(row));
  write(input_, image);
  const std::vector<std::vector<std::string>> rejected_arguments = {
      args({"--ordinal=0", "--ordinal", "1"}),
      args({"--kernel=test_kernel"}),
      {"--input", input_.string(), "--kernel", "test_kernel", "--list", "--list=true"},
      {"--help", "--help=true"},
      args({"--exact"}),
      {"--input", input_.string(), "--kernel", "test_kernel", "--output", "--list"},
      args({"--target="}),
      args({"--ordinal="}),
      args({"--exact="}),
      args({"--unknown=value"}),
      args({"unexpected"}),
      args({"--", "--unknown"}),
      args({"--help"}),
      args({"--capabilities"}),
      {"--help", "--capabilities"},
      {"--capabilities", "--help"},
      {"--help", "--unknown"},
      {"--help", "unexpected"},
      args({"--help=false"}),
      args({"--capabilities=0"}),
      args({"--list=false"}),
      {"--input", input_.string(), "--kernel", "test_kernel", "--list", "--ordinal=0"},
      {"--input", input_.string(), "--kernel", "test_kernel", "--list", "--exact=0"},
      args({"--ordinal=18446744073709551616"}),
      args({"--exact=0x10000000000000000"}),
      args({"--exact=-0"}),
      args({"--ordinal=1junk"})};
  for (const std::vector<std::string> &arguments : rejected_arguments) {
    SCOPED_TRACE(::testing::PrintToString(arguments));
    expect_rejected(arguments);
    EXPECT_FALSE(std::filesystem::exists(root_ / "--list"));
    EXPECT_EQ(read(input_), image);
  }
  for (const std::string exact : {"18446744073709551615", "0xffffffffffffffff"}) {
    SCOPED_TRACE(exact);
    expect_rejected(args({"--exact=" + exact, "--ordinal=0"}));
    EXPECT_NE(stderr_.find("selection failed: code=InvalidParameters offset=18446744073709551615 "),
              std::string::npos)
        << stderr_;
    EXPECT_EQ(read(input_), image);
  }
}

TEST_F(FaultMutationCli, GeneratedHelpListsRegisteredFlagsAndValuePlaceholders) {
  ASSERT_EQ(invoke({"--help"}), 0) << stderr_;
  const std::string help = stdout_;
  for (const std::string_view entry :
       {"--input FILE\n", "--output NEW-FILE\n", "--kernel ENTRY\n", "--kind KIND\n",
        "--target GFX-NAME\n", "--exact TEXT-OFFSET\n", "--ordinal N\n", "--list\n",
        "--capabilities\n", "--help\n"}) {
    SCOPED_TRACE(entry);
    EXPECT_NE(help.find(entry), std::string::npos) << help;
  }
  EXPECT_TRUE(stderr_.empty());
  EXPECT_FALSE(std::filesystem::exists(output_));
  ASSERT_EQ(invoke({"--help=true"}), 0) << stderr_;
  EXPECT_EQ(stdout_, help);
  EXPECT_TRUE(stderr_.empty());
  EXPECT_FALSE(std::filesystem::exists(output_));
}

TEST_F(FaultMutationCli, SelectionAndMalformedInputsFailWithoutPublication) {
  const test::DropTargetRow &row = test::kDropRows[4];
  const std::vector<uint8_t> image =
      test::make_barrier_elf_for_target(row, test::make_barrier_words(row));
  write(input_, image);
  expect_rejected(args({"--ordinal", "2"}));
  EXPECT_NE(stderr_.find("code=OrdinalOutOfRange"), std::string::npos) << stderr_;
  for (const uint64_t offset : {0, 5, 8, 12, 16, 9999}) {
    SCOPED_TRACE(offset);
    expect_rejected(args({"--exact", std::to_string(offset), "--ordinal", "0"}));
    EXPECT_NE(stderr_.find("no complete barrier starts at .text offset " + std::to_string(offset) +
                           " for this entry"),
              std::string::npos)
        << stderr_;
    EXPECT_NE(stderr_.find("code=InvalidParameters"), std::string::npos) << stderr_;
    EXPECT_NE(stderr_.find("offset=" + std::to_string(offset) + ' '), std::string::npos) << stderr_;
    EXPECT_EQ(stderr_.find("BadIdentity"), std::string::npos) << stderr_;
  }
  expect_rejected({"--input", input_.string(), "--kernel", "test", "--output", output_.string()});
  for (const std::string name : {"missing.hsaco", "."}) {
    expect_rejected({"--input", (root_ / name).string(), "--kernel", "test_kernel", "--output",
                     output_.string()});
  }
  for (size_t size : {size_t{0}, size_t{10}, image.size() - 1}) {
    write(input_, std::span(image).first(size));
    expect_rejected(args());
  }
  // Empty inventories and a corrupt final physical member must not become half-drops.
  for (const std::vector<uint32_t> &words : std::vector<std::vector<uint32_t>>{
           {checked_encoding(build_s_endpgm(row.arch))},
           {test::kSignal, test::kMov, 0xffffffffu, checked_encoding(build_s_endpgm(row.arch))},
           {test::kSignal, test::kMov, checked_encoding(build_s_endpgm(row.arch))}}) {
    write(input_, test::make_barrier_elf_for_target(row, words));
    expect_rejected(args());
  }
  for (int field : {EI_CLASS, EI_DATA, EI_OSABI}) {
    std::vector<uint8_t> malformed = image;
    malformed[static_cast<size_t>(field)] = 0;
    write(input_, malformed);
    expect_rejected(args());
  }
  for (uint16_t type : {uint16_t{ET_REL}, uint16_t{2}}) {
    std::vector<uint8_t> malformed = image;
    std::memcpy(malformed.data() + offsetof(Elf64_Ehdr, e_type), &type, sizeof(type));
    write(input_, malformed);
    expect_rejected(args());
  }
  for (const std::string magic : {"__CLANG_OFFLOAD_BUNDLE__", "!<arch>\n"}) {
    write(input_, std::span(reinterpret_cast<const uint8_t *>(magic.data()), magic.size()));
    expect_rejected(args());
  }
  write(input_, image);
  std::filesystem::resize_file(input_, 256 * 1024 * 1024 + 1);
  expect_rejected(args());
  EXPECT_NE(stderr_.find("bounded input"), std::string::npos);
}

TEST_F(FaultMutationCli, UnknownKernelListsDescriptorEntries) {
  const test::DropTargetRow &row = test::kDropRows[7];
  for (const bool siblings : {false, true}) {
    SCOPED_TRACE(siblings);
    const std::vector<uint8_t> image =
        test::make_barrier_elf_for_target(row, test::make_barrier_words(row), siblings);
    write(input_, image);
    for (const bool list : {false, true}) {
      SCOPED_TRACE(list);
      std::vector<std::string> arguments = {"--input", input_.string(), "--kernel",
                                            "missing_kernel"};
      if (list) {
        arguments.push_back("--list");
      } else {
        arguments.insert(arguments.end(), {"--output", output_.string()});
      }
      expect_rejected(std::move(arguments));
      EXPECT_NE(stderr_.find("discovery failed: code=BadIdentity"), std::string::npos);
      EXPECT_NE(stderr_.find("descriptor entries:"), std::string::npos);
      if (siblings) {
        EXPECT_NE(stderr_.find("test_kernel0"), std::string::npos);
        EXPECT_NE(stderr_.find("test_kernel1"), std::string::npos);
      } else {
        EXPECT_NE(stderr_.find("test_kernel"), std::string::npos);
      }
      EXPECT_EQ(stderr_.find("offset="), std::string::npos);
      EXPECT_EQ(read(input_), image);
    }
  }
}

TEST_F(FaultMutationCli, PreservesTypedDiscoveryAndSelectionErrors) {
  const test::DropTargetRow &row = test::kDropRows[4];
  std::vector<uint8_t> image =
      test::make_barrier_elf_for_target(row, test::make_barrier_words(row));
  write(input_, image);
  expect_rejected(
      {"--input", input_.string(), "--kernel", "missing", "--output", output_.string()});
  EXPECT_NE(stderr_.find("discovery failed: code=BadIdentity"), std::string::npos) << stderr_;
  EXPECT_EQ(stderr_.find("offset="), std::string::npos) << stderr_;
  image[EI_DATA] = 0;
  write(input_, image);
  expect_rejected(args());
  EXPECT_NE(stderr_.find("discovery failed: code=UnsupportedForm"), std::string::npos) << stderr_;
  image[EI_DATA] = ELFDATA2LSB;
  const uint32_t unsupported = EF_AMDGPU_MACH_AMDGCN_GFX90A;
  std::memcpy(image.data() + offsetof(Elf64_Ehdr, e_flags), &unsupported, sizeof(unsupported));
  write(input_, image);
  expect_rejected(args());
  EXPECT_NE(stderr_.find("discovery failed: code=UnsupportedTarget"), std::string::npos) << stderr_;
  write(input_,
        test::make_barrier_elf_for_target(row, {checked_encoding(build_s_endpgm(row.arch))}));
  expect_rejected(args());
  EXPECT_NE(stderr_.find("selection failed: code=NoCandidates"), std::string::npos) << stderr_;
  EXPECT_EQ(stderr_.find("offset="), std::string::npos) << stderr_;
}

TEST_F(FaultMutationCli, EmissionFailureDoesNotPublishOutput) {
  const std::vector<uint8_t> image = test::make_out_of_range_second_member_elf();
  write(input_, image);
  ASSERT_EQ(invoke({"--input", input_.string(), "--kernel", "test_kernel", "--list"}), 0)
      << stderr_;
  EXPECT_NE(stdout_.find("ordinal=0 exact=0 "), std::string::npos) << stdout_;
  expect_rejected(args({"--exact", "0"}));
  EXPECT_NE(stderr_.find("mutation failed: code=ValidationFailure"), std::string::npos) << stderr_;
  EXPECT_NE(stderr_.find("return branch exceeds s_branch simm16"), std::string::npos) << stderr_;
  EXPECT_EQ(read(input_), image);
}

TEST_F(FaultMutationCli, FailedStdoutReturnsOutputIoError) {
  const test::DropTargetRow &row = test::kDropRows[4];
  const std::vector<uint8_t> image =
      test::make_barrier_elf_for_target(row, test::make_barrier_words(row));
  write(input_, image);
  for (const std::vector<std::string> &arguments : std::vector<std::vector<std::string>>{
           {"--help"},
           {"--capabilities"},
           {"--input", input_.string(), "--kernel", "test_kernel", "--list"},
           args()}) {
    SCOPED_TRACE(::testing::PrintToString(arguments));
    EXPECT_EQ(invoke(arguments, {.stdout_failure = StreamFailure::FullDevice}), 3) << stderr_;
    EXPECT_NE(stderr_.find("cannot write/flush stdout"), std::string::npos) << stderr_;
  }
  // The complete file survives a later failure to report the mutation summary.
  const FaultMutationContext mutant(read(output_));
  FaultMutationError error;
  const FailureOr<FaultBarrierInventory> remaining =
      discover_drop_barriers(mutant, "test_kernel", error);
  ASSERT_TRUE(remaining.succeeded()) << error.message;
  EXPECT_EQ(remaining.value().barriers().size(), 1u);
  EXPECT_EQ(read(input_), image);
}

TEST_F(FaultMutationCli, FailedAssumptionOutputDoesNotReportSuccess) {
  const test::DropTargetRow &row = test::kDropRows[4];
  const std::vector<uint8_t> image =
      test::make_barrier_elf_for_target(row, test::make_barrier_words(row));
  write(input_, image);
  EXPECT_EQ(invoke(args(), {.stderr_failure = StreamFailure::FullDevice}), 3);
  EXPECT_TRUE(stdout_.empty());
  expect_complete_output(image, read(output_));
  EXPECT_EQ(read(input_), image);
}

TEST_F(FaultMutationCli, ClosedOutputPipesReturnOutputIoError) {
  const test::DropTargetRow &row = test::kDropRows[4];
  const std::vector<uint8_t> image =
      test::make_barrier_elf_for_target(row, test::make_barrier_words(row));
  write(input_, image);
  for (const std::vector<std::string> &arguments : std::vector<std::vector<std::string>>{
           {"--help"},
           {"--capabilities"},
           {"--input", input_.string(), "--kernel", "test_kernel", "--list"},
           args()}) {
    SCOPED_TRACE(::testing::PrintToString(arguments));
    EXPECT_EQ(invoke(arguments, {.stdout_failure = StreamFailure::ClosedPipe}), 3) << stderr_;
    EXPECT_NE(stderr_.find("cannot write/flush stdout"), std::string::npos) << stderr_;
  }
  // A summary failure preserves the same complete mutant as an assumption failure.
  const std::vector<uint8_t> mutant = read(output_);
  expect_complete_output(image, mutant);
  std::filesystem::remove(output_);
  EXPECT_EQ(invoke(args(), {.stderr_failure = StreamFailure::ClosedPipe}), 3);
  EXPECT_TRUE(stdout_.empty());
  EXPECT_EQ(read(output_), mutant);
  EXPECT_EQ(read(input_), image);
}

TEST_F(FaultMutationCli, RejectsIllegalWavesAndForms) {
  for (test::DropTargetRow row : {test::kDropRows[6], test::kDropRows[7]}) {
    row.wave_size = row.wave_size == 32 ? 64 : 32;
    write(input_, test::make_barrier_elf_for_target(row, test::make_barrier_words(row)));
    expect_rejected(args());
  }
  const test::DropTargetRow &row = test::kDropRows[4];
  // Named signal/wait cannot be admitted by the CLI as builtin complete Drop.
  write(input_, test::make_barrier_elf_for_target(
                    row, {0xbe804e80u, 0xbf940000u, checked_encoding(build_s_endpgm(row.arch))}));
  expect_rejected(args());
}

TEST_F(FaultMutationCli, RejectsTrapBarriersBeforeSelectingUserBarrier) {
  for (const test::DropTargetRow &row : test::kDropRows) {
    if (!row.split) {
      continue;
    }
    SCOPED_TRACE(row.name);
    for (const int32_t id : {-2, -4}) {
      SCOPED_TRACE(id);
      const uint32_t signal = 0xbe804e00u | static_cast<uint32_t>(192 - id);
      const uint32_t wait = 0xbf940000u | static_cast<uint16_t>(id);
      const std::vector<uint8_t> image =
          test::make_barrier_elf_for_target(row, {signal, wait, test::kSignal, test::kWait,
                                                  checked_encoding(build_s_endpgm(row.arch))});
      const std::string_view diagnostic =
          row.arch == IsaFamily::Rdna4 && id == -4 ? "invalid scalar source selector" : "trap-only";
      write(input_, image);
      expect_rejected({"--input", input_.string(), "--kernel", "test_kernel", "--list"});
      EXPECT_NE(stderr_.find(diagnostic), std::string::npos) << stderr_;
      expect_rejected(args());
      EXPECT_NE(stderr_.find(diagnostic), std::string::npos) << stderr_;
      EXPECT_EQ(read(input_), image);
    }
  }
}

TEST_F(FaultMutationCli, EmitsAdmittedClusterBarrierAtExactOffsetZero) {
  for (const test::DropTargetRow &row : {test::kDropRows[4], test::kDropRows[5]}) {
    SCOPED_TRACE(row.name);
    const int32_t id = -3;
    const uint32_t signal = 0xbe804e00u | static_cast<uint32_t>(192 - id);
    const uint32_t wait = 0xbf940000u | static_cast<uint16_t>(id);
    const std::vector<uint8_t> image = test::make_barrier_elf_for_target(
        row, {signal, wait, checked_encoding(build_s_endpgm(row.arch))});
    write(input_, image);
    ASSERT_EQ(invoke({"--input", input_.string(), "--kernel", "test_kernel", "--list"}), 0)
        << stderr_;
    EXPECT_NE(stdout_.find("exact=0"), std::string::npos);
    EXPECT_NE(stdout_.find("id=-3 scope=cluster members=0:4,4:4"), std::string::npos);
    ASSERT_EQ(invoke(args({"--exact", "0", "--ordinal", "18446744073709551615"})), 0) << stderr_;
    EXPECT_NE(stderr_.find("Same-block membership does not prove barrier convergence."),
              std::string::npos)
        << stderr_;
    EXPECT_NE(stderr_.find("Execution and detector qualification have not been performed."),
              std::string::npos)
        << stderr_;
    for (const std::string_view required_fact :
         {"Cluster barrier -3 assumes a clustered dispatch with multiple workgroups per cluster",
          "it is inert otherwise", "exactly one signal per workgroup",
          "after workgroup synchronization", "every wave waiting",
          "does not verify workgroup size or this protocol"}) {
      EXPECT_NE(stderr_.find(required_fact), std::string::npos) << required_fact << '\n' << stderr_;
    }
    expect_complete_output(image, read(output_));
    EXPECT_EQ(read(input_), image);
    std::filesystem::remove(output_);
  }
}

TEST_F(FaultMutationCli, SharedAbiAndFixedCdna5WaveAdmission) {
  test::DropTargetRow row = test::kDropRows[4];
  // kernel_wavefront_size() fixes CDNA5 at Wave32 irrespective of the RDNA bit.
  row.wave_size = 64;
  std::vector<uint8_t> image =
      test::make_barrier_elf_for_target(row, test::make_barrier_words(row));
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V4;
  write(input_, image);
  ASSERT_EQ(invoke(args()), 0) << stderr_;
  const std::vector<uint8_t> mutant = read(output_);
  expect_complete_output(image, mutant);
  ASSERT_GT(mutant.size(), static_cast<size_t>(EI_ABIVERSION));
  EXPECT_EQ(mutant[EI_ABIVERSION], ELFABIVERSION_AMDGPU_HSA_V4);
  EXPECT_EQ(read(input_), image);
}

TEST_F(FaultMutationCli, ExactSiblingEntryPreservesUnselectedKernel) {
  const test::DropTargetRow &row = test::kDropRows[4];
  const std::vector<uint8_t> image =
      test::make_barrier_elf_for_target(row, test::make_barrier_words(row), true);
  write(input_, image);
  expect_rejected({"--input", input_.string(), "--kernel", "test_kernel1", "--exact", "4",
                   "--output", output_.string()});
  ASSERT_EQ(invoke({"--input", input_.string(), "--kernel", "test_kernel1", "--output",
                    output_.string()}),
            0)
      << stderr_;
  const FaultMutationContext context(image);
  FaultMutationError error;
  const FailureOr<FaultBarrierInventory> inventory =
      discover_drop_barriers(context, "test_kernel1", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const FailureOr<DropBarrierSelection> selected =
      select_drop_barrier(inventory.value(), {}, error);
  ASSERT_TRUE(selected.succeeded());
  const FailureOr<DropBarrierResult> mutant = apply_drop_barrier(context, selected.value(), error);
  ASSERT_TRUE(mutant.succeeded());
  EXPECT_EQ(read(output_), mutant.value().image);
  expect_complete_output(image, read(output_), "test_kernel1");
  const FaultMutationContext fresh(read(output_));
  const FailureOr<FaultBarrierInventory> sibling =
      discover_drop_barriers(fresh, "test_kernel0", error);
  ASSERT_TRUE(sibling.succeeded()) << error.message;
  EXPECT_EQ(sibling.value().barriers().size(), 2u);
  EXPECT_EQ(read(input_), image);
}

TEST_F(FaultMutationCli, RejectsAmbiguousEntryAndUnsupportedConcreteTarget) {
  const test::DropTargetRow &row = test::kDropRows[4];
  std::vector<uint8_t> image =
      test::make_barrier_elf_for_target(row, test::make_barrier_words(row), true);
  const AmdGpuCodeObject source(image.data(), image.size());
  const CodeObjectPatcher patcher(source);
  for (const Elf64_Shdr &section : patcher.section_headers()) {
    if (section.sh_type != SHT_SYMTAB) {
      continue;
    }
    Elf64_Sym first = {}, second = {};
    std::memcpy(&first, image.data() + section.sh_offset + sizeof(Elf64_Sym), sizeof(first));
    std::memcpy(&second, image.data() + section.sh_offset + 2 * sizeof(Elf64_Sym), sizeof(second));
    second.st_name = first.st_name;
    std::memcpy(image.data() + section.sh_offset + 2 * sizeof(Elf64_Sym), &second, sizeof(second));
  }
  write(input_, image);
  expect_rejected(
      {"--input", input_.string(), "--kernel", "test_kernel0", "--output", output_.string()});
  EXPECT_EQ(read(input_), image);
  image = test::make_barrier_elf_for_target(row, test::make_barrier_words(row));
  // A model provider is linked for gfx90a, but its family has no Drop recipe.
  const uint32_t unsupported = EF_AMDGPU_MACH_AMDGCN_GFX90A;
  std::memcpy(image.data() + offsetof(Elf64_Ehdr, e_flags), &unsupported, sizeof(unsupported));
  write(input_, image);
  expect_rejected(args());
  EXPECT_EQ(read(input_), image);
}

TEST_F(FaultMutationCli, FileFailuresPreserveInputAndExistingDestination) {
  const test::DropTargetRow &row = test::kDropRows[4];
  const std::vector<uint8_t> image =
      test::make_barrier_elf_for_target(row, test::make_barrier_words(row));
  write(input_, image);
  for (const std::filesystem::path &destination : {input_, root_ / "absent" / "output", root_}) {
    EXPECT_EQ(invoke({"--input", input_.string(), "--kernel", "test_kernel", "--output",
                      destination.string()}),
              3)
        << stderr_;
    EXPECT_EQ(read(input_), image);
  }
  write(output_, image);
  EXPECT_EQ(invoke(args()), 3);
  EXPECT_EQ(read(output_), image);
  std::filesystem::remove(output_);
  for (bool hardlink : {false, true}) {
    if (hardlink) {
      std::filesystem::create_hard_link(input_, output_);
    } else {
      std::filesystem::create_symlink(input_, output_);
    }
    EXPECT_EQ(invoke(args()), 3);
    EXPECT_EQ(read(input_), image);
    EXPECT_EQ(read(output_), image);
    std::filesystem::remove(output_);
  }
  std::filesystem::create_symlink(root_ / "missing", output_);
  EXPECT_EQ(invoke(args()), 3);
  EXPECT_TRUE(std::filesystem::is_symlink(output_));
  EXPECT_FALSE(std::filesystem::exists(root_ / "missing"));
  std::filesystem::remove(output_);
  EXPECT_EQ(invoke(args(), {.limit_write = true}), 3) << stderr_;
  EXPECT_NE(stderr_.find("write/close"), std::string::npos);
  EXPECT_FALSE(std::filesystem::exists(output_));
  EXPECT_EQ(read(input_), image);
}

TEST_F(FaultMutationCli, ListsTwoAddressForms) {
  for (const test::TwoAddressTarget &target : test::kTwoAddressTargets) {
    for (bool wave32 : {false, true}) {
      if (target.cdna && wave32) {
        continue;
      }
      SCOPED_TRACE(static_cast<int>(target.target));
      SCOPED_TRACE(wave32);
      write(input_, test::make_two_address_elf(target, wave32));
      ASSERT_EQ(invoke({"--input", input_.string(), "--kernel", "test_kernel", "--kind",
                        "LdsWrongAddress", "--list"}),
                0);
      EXPECT_NE(stdout_.find(target.cdna ? "form=ds_read2_b64" : "form=ds_load_2addr_b64"),
                std::string::npos);
      EXPECT_NE(stdout_.find(target.cdna ? "form=ds_write2st64_b64" : "form=ds_store_2addr_b32"),
                std::string::npos);
    }
  }
}

TEST_F(FaultMutationCli, ListsSubwordAndTransposeForms) {
  for (const test::LdsFormCase &fixture : test::kSubwordCases) {
    for (bool wave32 : {false, true}) {
      if (fixture.arch != IsaFamily::Rdna4 && wave32) {
        continue;
      }
      SCOPED_TRACE(wave32);
      SCOPED_TRACE(fixture.form);
      write(input_, test::make_lds_form_elf(fixture, false, wave32));
      ASSERT_EQ(invoke({"--input", input_.string(), "--kernel", "test_kernel", "--kind",
                        "LdsWrongAddress", "--list"}),
                0);
      EXPECT_NE(stdout_.find("form=" + std::string(fixture.form)), std::string::npos);
    }
  }
}

TEST_F(FaultMutationCli, ListsRemainingSubwordForms) {
  for (const test::LdsFormCase &fixture : test::kRemainingSubwordCases) {
    for (bool wave32 : {false, true}) {
      if (fixture.arch != IsaFamily::Rdna4 && wave32) {
        continue;
      }
      SCOPED_TRACE(wave32);
      SCOPED_TRACE(fixture.form);
      SCOPED_TRACE(static_cast<int>(fixture.target));
      write(input_, test::make_lds_form_elf(fixture, false, wave32));
      ASSERT_EQ(invoke({"--input", input_.string(), "--kernel", "test_kernel", "--kind",
                        "LdsWrongAddress", "--list"}),
                0);
      EXPECT_NE(stdout_.find("form=" + std::string(fixture.form)), std::string::npos);
    }
  }
}

TEST_F(FaultMutationCli, Rdna4Wave32ListsSourceSites) {
  for (const std::pair<uint32_t, const char *> &row :
       std::array<std::pair<uint32_t, const char *>, 2>{
           {{EF_AMDGPU_MACH_AMDGCN_GFX1200, "gfx1200"},
            {EF_AMDGPU_MACH_AMDGCN_GFX1201, "gfx1201"}}}) {
    const uint32_t machine = row.first;
    const char *target = row.second;
    SCOPED_TRACE(target);
    write(input_, test::make_lds_load_elf(machine, true, 0));
    ASSERT_EQ(invoke({"--input", input_.string(), "--kernel", "test_kernel", "--kind",
                      "LdsWrongAddress", "--list"}),
              0)
        << stderr_;
    EXPECT_NE(stdout_.find("ordinal=0 exact=0 address_vgpr=2 address_bank=0 target=" +
                           std::string(target) + " form=ds_load_b32\n"),
              std::string::npos);
    EXPECT_NE(stdout_.find("ordinal=1 exact=8 address_vgpr=1 address_bank=0 target=" +
                           std::string(target) + " form=ds_store_b32\n"),
              std::string::npos);
  }
}

} // namespace
} // namespace rocjitsu
