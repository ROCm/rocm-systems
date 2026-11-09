// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "multiplexer_fuzz_core.h"

#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#ifdef __AFL_HAVE_MANUAL_CONTROL
#include <unistd.h>

__AFL_FUZZ_INIT();
#endif

namespace {
using namespace rocjitsu::multiplexer_fuzz;

void usage(std::ostream &stream) {
  stream << "Usage: rj_fuzz --input FILE\n"
            "       rj_fuzz --emit-seeds DIRECTORY\n"
            "       rj_fuzz --afl\n";
}

util::FailureOr<std::vector<uint8_t>> read_input(const std::filesystem::path &path,
                                                 util::DiagnosticEmitter diagnostic) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream || stream.tellg() <= 0 ||
      stream.tellg() > static_cast<std::streamoff>(kMaxInputSize)) {
    return diagnostic.emit() << "input has an invalid size";
  }
  std::vector<uint8_t> input(static_cast<std::size_t>(stream.tellg()));
  stream.seekg(0);
  if (!stream.read(reinterpret_cast<char *>(input.data()), input.size()))
    return diagnostic.emit() << "cannot read input";
  return input;
}

util::Result write_input(const std::filesystem::path &path, std::span<const uint8_t> input,
                         util::DiagnosticEmitter diagnostic) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.write(reinterpret_cast<const char *>(input.data()), input.size());
  stream.close();
  if (!stream)
    return diagnostic.emit() << "cannot write fuzz seed";
  return util::Result::success();
}

std::string_view target_name(FuzzTarget target);

util::Result emit_seeds(const std::filesystem::path &directory,
                        util::DiagnosticEmitter diagnostic) {
  std::error_code error;
  const bool exists = std::filesystem::exists(directory, error);
  if (error)
    return diagnostic.emit() << "cannot inspect seed destination: " << error.message();
  if (exists) {
    const bool empty = std::filesystem::is_empty(directory, error);
    if (error || !empty)
      return diagnostic.emit() << "seed destination must be an empty directory";
    const bool is_directory = std::filesystem::is_directory(directory, error);
    if (error || !is_directory)
      return diagnostic.emit() << "seed destination must be an empty directory";
  }
  std::filesystem::create_directories(directory, error);
  if (error)
    return diagnostic.emit() << "cannot create seed directory: " << error.message();
  const util::FailureOr<std::vector<std::vector<uint8_t>>> maybe_seeds = seed_corpus(diagnostic);
  if (maybe_seeds.failed())
    return util::Result::failure();
  const auto &seeds = maybe_seeds.value();
  std::map<FuzzTarget, std::size_t> emitted_counts;
  for (const std::vector<uint8_t> &seed : seeds) {
    assert(!seed.empty() && "multiplexer fuzz seed registry is inconsistent");
    const std::optional<FuzzTarget> target = selected_target(seed.front());
    assert(target && "multiplexer fuzz seed registry is inconsistent");
    std::string filename(target_name(*target));
    std::size_t &count = emitted_counts[*target];
    if (count != 0)
      filename += "-" + std::to_string(count);
    ++count;
    if (write_input(directory / (filename + ".bin"), seed, diagnostic).failed())
      return util::Result::failure();
  }
  std::cout << "wrote " << seeds.size() << " multiplexer fuzz seeds to " << directory << '\n';
  return util::Result::success();
}

std::string_view target_name(FuzzTarget target) {
  for (const TargetDescriptor &descriptor : targets()) {
    if (descriptor.id == target)
      return descriptor.name;
  }
  assert(false && "multiplexer fuzz target registry is inconsistent");
  std::abort();
}

} // namespace

int run_afl() {
#ifdef __AFL_HAVE_MANUAL_CONTROL
  __AFL_INIT();
  unsigned char *buffer = __AFL_FUZZ_TESTCASE_BUF;
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-statement-expression-from-macro-expansion"
#endif
  const std::size_t size = __AFL_FUZZ_TESTCASE_LEN;
#ifdef __clang__
#pragma clang diagnostic pop
#endif
  // Keep each broad-dispatch input in a fresh forkserver child. The focused
  // targets retain persistent loops for throughput; process isolation here
  // prevents one target's lazy initialization and helper threads from making
  // another target's coverage depend on dispatch order.
  if (size > 0 && size <= kMaxInputSize &&
      run_input(std::span<const uint8_t>(buffer, size)).failed())
    std::abort();
  return 0;
#else
  std::cerr << "rj_fuzz requires an AFL++ compiler wrapper for --afl\n";
  return 2;
#endif
}

int main(int argc, char **argv) {
  bool afl = false;
  std::string input;
  std::string seed_directory;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--afl")
      afl = true;
    else if (argument == "--input" && index + 1 < argc)
      input = argv[++index];
    else if (argument == "--emit-seeds" && index + 1 < argc)
      seed_directory = argv[++index];
    else if (argument == "--help") {
      usage(std::cout);
      return 0;
    } else {
      usage(std::cerr);
      return 2;
    }
  }

  const int modes = static_cast<int>(afl) + static_cast<int>(!input.empty()) +
                    static_cast<int>(!seed_directory.empty());
  if (modes != 1) {
    usage(std::cerr);
    return 2;
  }
  // Let simulator failures and sanitizer findings reach AFL++ directly.
  if (afl)
    return run_afl();

  util::StringDiagnostic diagnostic;
  const auto fail = [&] {
    std::cerr << "rj_fuzz: " << diagnostic.message() << '\n';
    return 1;
  };
  if (!seed_directory.empty()) {
    if (emit_seeds(seed_directory, diagnostic.emitter()).failed())
      return fail();
    return 0;
  }
  const util::FailureOr<std::vector<uint8_t>> bytes = read_input(input, diagnostic.emitter());
  if (bytes.failed())
    return fail();
  const util::FailureOr<std::optional<DispatchResult>> maybe_result =
      run_input(bytes.value(), diagnostic.emitter());
  if (maybe_result.failed())
    return fail();
  const std::optional<DispatchResult> &result = maybe_result.value();
  if (!result) {
    std::cout << "{\"target\":\"unassigned\",\"work_units\":0}\n";
    return 0;
  }
  std::cout << "{\"target\":\"" << target_name(result->target)
            << "\",\"work_units\":" << result->work_units << "}\n";
  return 0;
}
