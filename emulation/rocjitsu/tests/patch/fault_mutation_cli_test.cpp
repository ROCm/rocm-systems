// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "fault_barrier_fixture.h"
#include "rocjitsu/code/file_io.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/fault_injection/drop_barrier.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "scoped_temp.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <csignal>
#include <fcntl.h>
#include <format>
#include <fstream>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

namespace rocjitsu {
namespace {

class FaultMutationCli : public ::testing::Test {
protected:
  test::ScopedTempDirectory directory_{"fault-cli-"};
  std::filesystem::path root_ = directory_.path();
  std::filesystem::path input_ = root_ / "original.hsaco";
  std::filesystem::path output_ = root_ / "mutant.hsaco";
  std::string stdout_, stderr_;

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
      if (pipe(descriptors) != 0)
        return -1;
      close(descriptors[0]);
      return descriptors[1];
    }
    return failure == StreamFailure::FullDevice
               ? open("/dev/full", O_WRONLY)
               : open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
  }
  int invoke(std::vector<std::string> arguments, InvokeOptions options) {
    arguments.insert(arguments.begin(), RJ_CODE_MUTATE_PATH);
    std::vector<char *> pointers;
    for (std::string &argument : arguments)
      pointers.push_back(argument.data());
    pointers.push_back(nullptr);
    const int out = open_capture(root_ / "stdout", options.stdout_failure);
    const int err = open_capture(root_ / "stderr", options.stderr_failure);
    EXPECT_GE(out, 0);
    EXPECT_GE(err, 0);
    const pid_t child = fork();
    if (child == 0) {
      if (dup2(out, STDOUT_FILENO) < 0 || dup2(err, STDERR_FILENO) < 0)
        _exit(126);
      close(out);
      close(err);
      // Exercise the executable's signal policy, independent of the test runner.
      if (std::signal(SIGPIPE, SIG_DFL) == SIG_ERR || std::signal(SIGXFSZ, SIG_DFL) == SIG_ERR)
        _exit(126);
      if (options.limit_write) {
        const rlimit core_limit = {.rlim_cur = 0, .rlim_max = 0};
        const rlimit limit = {.rlim_cur = 512, .rlim_max = 512};
        if (setrlimit(RLIMIT_CORE, &core_limit) != 0 || setrlimit(RLIMIT_FSIZE, &limit) != 0)
          _exit(126);
      }
      execv(pointers.front(), pointers.data());
      _exit(127);
    }
    close(out);
    close(err);
    if (child < 0)
      return -1;
    int status = 0;
    if (waitpid(child, &status, 0) != child)
      return -1;
    const std::vector<uint8_t> out_bytes = options.stdout_failure == StreamFailure::None
                                               ? read(root_ / "stdout")
                                               : std::vector<uint8_t>{};
    const std::vector<uint8_t> err_bytes = options.stderr_failure == StreamFailure::None
                                               ? read(root_ / "stderr")
                                               : std::vector<uint8_t>{};
    stdout_.assign(out_bytes.begin(), out_bytes.end());
    stderr_.assign(err_bytes.begin(), err_bytes.end());
    if (WIFEXITED(status))
      return WEXITSTATUS(status);
    return WIFSIGNALED(status) ? 128 + WTERMSIG(status) : -1;
  }
  int invoke(std::vector<std::string> arguments) { return invoke(std::move(arguments), {}); }
  std::vector<std::string> args(std::vector<std::string> extra = {}) {
    std::vector<std::string> result{"--input",     input_.string(), "--kernel",
                                    "test_kernel", "--output",      output_.string()};
    result.insert(result.end(), extra.begin(), extra.end());
    return result;
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
      if (!members.empty())
        members += ',';
      members += std::to_string(member.offset) + ":4";
    }
    EXPECT_NE(stdout_.find("ordinal=" + std::to_string(ordinal) +
                           " exact=" + std::to_string(barrier.members.front().offset)),
              std::string::npos);
    EXPECT_NE(stdout_.find("members=" + members + "\n"), std::string::npos);
  }
  const std::string target(
      default_isa_target_registry().find_gpu_target(row.target)->code_object_id);
  for (const std::string mode : {"default", "ordinal", "exact"}) {
    SCOPED_TRACE(mode);
    const size_t ordinal = mode == "default" ? 0 : 1;
    const StaticBarrier &barrier = inventory.value().barriers()[ordinal];
    std::vector<std::string> options{"--kind", "DropBarrier", "--target", target};
    if (mode == "ordinal")
      options.insert(options.end(), {"--ordinal", "1"});
    if (mode == "exact")
      options.insert(options.end(),
                     {"--exact", std::format("{:#x}", barrier.members.front().offset), "--ordinal",
                      "18446744073709551615"});
    ASSERT_EQ(invoke(args(options)), 0) << stderr_;
    const std::vector<uint8_t> mutant = read(output_);
    ASSERT_EQ(mutant.size(), image.size());
    const FailureOr<DropBarrierSelection> selection =
        select_drop_barrier(inventory.value(), {.exact = std::nullopt, .ordinal = ordinal}, error);
    ASSERT_TRUE(selection.succeeded());
    const FailureOr<DropBarrierResult> library =
        apply_drop_barrier(context, selection.value(), error);
    ASSERT_TRUE(library.succeeded()) << error.message;
    std::string expected_stderr;
    for (const std::string &assumption : library.value().assumptions)
      expected_stderr += "rj_code_mutate: assumption: " + assumption + '\n';
    EXPECT_EQ(stderr_, expected_stderr);
    EXPECT_EQ(mutant, library.value().edit_result.image);
    // Independent full-image oracle: only the selected complete members become
    // concrete-target s_nop 0 instructions. Headers, descriptors and waits stay exact.
    std::vector<uint8_t> expected = image;
    const uint32_t nop = build_s_nop(0, row.arch);
    std::unique_ptr<Decoder> decoder = Decoder::create(default_isa_target_registry(), row.target);
    for (const StaticBarrierMember &member : barrier.members) {
      const size_t file_offset = inventory.value().section().file_offset + member.offset;
      std::memcpy(expected.data() + file_offset, &nop, 4);
      uint32_t word = 0;
      std::memcpy(&word, mutant.data() + file_offset, 4);
      const DecodeResult decoded = decoder->decode_window(std::span(&word, 1), member.offset);
      ASSERT_TRUE(decoded.succeeded());
      EXPECT_EQ(decoded.value()->mnemonic(), "s_nop");
      EXPECT_EQ(decoded.value()->size(), 4u);
      EXPECT_EQ(decoded.value()->src_operand(0)->encoding_value(), 0u);
    }
    EXPECT_EQ(mutant, expected);
    const FaultMutationContext fresh(mutant);
    EXPECT_EQ(fresh.source().target_id(), row.target);
    const FailureOr<FaultBarrierInventory> remaining =
        discover_drop_barriers(fresh, "test_kernel", error);
    ASSERT_TRUE(remaining.succeeded()) << error.message;
    ASSERT_EQ(remaining.value().barriers().size(), 1u);
    EXPECT_EQ(remaining.value().barriers()[0], inventory.value().barriers()[1 - ordinal]);
    std::filesystem::remove(output_);
    ASSERT_EQ(invoke(args(options)), 0) << stderr_;
    EXPECT_EQ(read(output_), mutant);
    std::filesystem::remove(output_);
  }
  EXPECT_EQ(read(input_), image);
}

INSTANTIATE_TEST_SUITE_P(TargetWaves, FaultMutationCliFamily, ::testing::ValuesIn(test::kDropRows),
                         [](const ::testing::TestParamInfo<test::DropTargetRow> &info) {
                           return info.param.name;
                         });

TEST_F(FaultMutationCli, HelpCapabilitiesAndArguments) {
  EXPECT_EQ(invoke({"--help"}), 0);
  EXPECT_NE(stdout_.find("Exact takes precedence"), std::string::npos);
  EXPECT_EQ(invoke({"--capabilities"}), 0);
  const std::string capabilities = stdout_;
  EXPECT_EQ(std::count(capabilities.begin(), capabilities.end(), '\n'), 10);
  EXPECT_EQ(capabilities.find("LdsWrongAddress"), std::string::npos);
  for (const FaultMutationCapability &capability : fault_mutation_capabilities()) {
    const std::string target(
        default_isa_target_registry().find_gpu_target(capability.target)->code_object_id);
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
                                             {"--list"}})
    expect_rejected(args(options));
  expect_rejected({});
  expect_rejected({"--input", input_.string(), "--output", output_.string()});
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
  for (const std::string name : {"missing.hsaco", "."})
    expect_rejected({"--input", (root_ / name).string(), "--kernel", "test_kernel", "--output",
                     output_.string()});
  for (size_t size : {size_t{0}, size_t{10}, image.size() - 1}) {
    write(input_, std::span(image).first(size));
    expect_rejected(args());
  }
  // Empty inventories and a corrupt final physical member must not become half-drops.
  for (const std::vector<uint32_t> &words :
       std::vector<std::vector<uint32_t>>{{0xbfb00000u},
                                          {0xbe804ec1u, 0xbe800081u, 0xffffffffu, 0xbfb00000u},
                                          {0xbe804ec1u, 0xbe800081u, 0xbfb00000u}}) {
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

TEST_F(FaultMutationCli, PreservesTypedDiscoveryAndSelectionErrors) {
  const test::DropTargetRow &row = test::kDropRows[4];
  std::vector<uint8_t> image =
      test::make_barrier_elf_for_target(row, test::make_barrier_words(row));
  write(input_, image);
  expect_rejected(
      {"--input", input_.string(), "--kernel", "missing", "--output", output_.string()});
  EXPECT_NE(stderr_.find("discovery failed: code=BadIdentity"), std::string::npos) << stderr_;
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
  write(input_, test::make_barrier_elf_for_target(row, {build_s_endpgm(row.arch)}));
  expect_rejected(args());
  EXPECT_NE(stderr_.find("selection failed: code=NoCandidates"), std::string::npos) << stderr_;
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
  EXPECT_EQ(read(output_).size(), image.size());
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
  ASSERT_EQ(mutant.size(), image.size());
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
  write(input_, test::make_barrier_elf_for_target(row, {0xbe804e80u, 0xbf940000u, 0xbfb00000u}));
  expect_rejected(args());
}

TEST_F(FaultMutationCli, RejectsTrapBarriersBeforeSelectingUserBarrier) {
  for (const test::DropTargetRow &row : test::kDropRows) {
    if (!row.split)
      continue;
    SCOPED_TRACE(row.name);
    for (const int32_t id : {-2, -4}) {
      SCOPED_TRACE(id);
      const uint32_t signal = 0xbe804e00u | static_cast<uint32_t>(192 - id);
      const uint32_t wait = 0xbf940000u | static_cast<uint16_t>(id);
      const std::vector<uint8_t> image = test::make_barrier_elf_for_target(
          row, {signal, wait, 0xbe804ec1u, 0xbf94ffffu, build_s_endpgm(row.arch)});
      const std::string_view diagnostic = row.arch == ROCJITSU_CODE_ARCH_RDNA4 && id == -4
                                              ? "invalid scalar source selector"
                                              : "trap-only";
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
    const std::vector<uint8_t> image = test::make_barrier_elf_for_target(
        row, {0xbe804ec3u, 0xbf94fffdu, build_s_endpgm(row.arch)});
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
    EXPECT_NE(
        stderr_.find(
            "Cluster barrier -3 only synchronizes a clustered dispatch; it is inert otherwise."),
        std::string::npos)
        << stderr_;
    const AmdGpuCodeObject source(image.data(), image.size());
    ASSERT_TRUE(source.is_valid());
    ASSERT_EQ(source.text_sections().size(), 1u);
    const size_t text_offset = source.text_sections().front()->sectionOffset();
    std::vector<uint8_t> expected = image;
    const uint32_t nop = build_s_nop(0, row.arch);
    std::memcpy(expected.data() + text_offset, &nop, 4);
    std::memcpy(expected.data() + text_offset + 4, &nop, 4);
    EXPECT_EQ(read(output_), expected);
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
  ASSERT_EQ(mutant.size(), image.size());
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
  EXPECT_EQ(read(output_), mutant.value().edit_result.image);
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
    if (section.sh_type != SHT_SYMTAB)
      continue;
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
    if (hardlink)
      std::filesystem::create_hard_link(input_, output_);
    else
      std::filesystem::create_symlink(input_, output_);
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

} // namespace
} // namespace rocjitsu
