// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "aggregate_fuzz_core.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifdef __AFL_HAVE_MANUAL_CONTROL
#include <unistd.h>

__AFL_FUZZ_INIT();
#endif

namespace {
using namespace rocjitsu::aggregate_fuzz;

void usage(std::ostream &stream) {
  stream << "Usage: rj_fuzz --input FILE\n"
            "       rj_fuzz --emit-seeds DIRECTORY\n"
            "       rj_fuzz --afl\n";
}

std::vector<uint8_t> read_input(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream || stream.tellg() <= 0 ||
      stream.tellg() > static_cast<std::streamoff>(kMaxInputSize)) {
    throw std::invalid_argument("input has an invalid size");
  }
  std::vector<uint8_t> input(static_cast<std::size_t>(stream.tellg()));
  stream.seekg(0);
  if (!stream.read(reinterpret_cast<char *>(input.data()), input.size()))
    throw std::runtime_error("cannot read input");
  return input;
}

void write_input(const std::filesystem::path &path, std::span<const uint8_t> input) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream.write(reinterpret_cast<const char *>(input.data()), input.size()))
    throw std::runtime_error("cannot write fuzz seed");
}

std::string_view target_name(FuzzTarget target);

void emit_seeds(const std::filesystem::path &directory) {
  if (std::filesystem::exists(directory) &&
      (!std::filesystem::is_directory(directory) ||
       std::filesystem::directory_iterator(directory) != std::filesystem::directory_iterator())) {
    throw std::invalid_argument("seed destination must be an empty directory");
  }
  std::filesystem::create_directories(directory);
  const std::vector<std::vector<uint8_t>> seeds = seed_corpus();
  std::map<FuzzTarget, std::size_t> emitted_counts;
  for (const std::vector<uint8_t> &seed : seeds) {
    if (seed.empty())
      throw std::logic_error("aggregate fuzz seed registry is inconsistent");
    const std::optional<FuzzTarget> target = selected_target(seed.front());
    if (!target)
      throw std::logic_error("aggregate fuzz seed registry is inconsistent");
    std::string filename(target_name(*target));
    std::size_t &count = emitted_counts[*target];
    if (count != 0)
      filename += "-" + std::to_string(count);
    ++count;
    write_input(directory / (filename + ".bin"), seed);
  }
  std::cout << "wrote " << seeds.size() << " aggregate fuzz seeds to " << directory << '\n';
}

std::string_view target_name(FuzzTarget target) {
  for (const TargetDescriptor &descriptor : targets()) {
    if (descriptor.id == target)
      return descriptor.name;
  }
  throw std::logic_error("aggregate fuzz target registry is inconsistent");
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
  if (size > 0 && size <= kMaxInputSize)
    (void)run_input(std::span<const uint8_t>(buffer, size));
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

  try {
    if (!seed_directory.empty()) {
      emit_seeds(seed_directory);
      return 0;
    }
    const std::optional<DispatchResult> result = run_input(read_input(input));
    if (!result) {
      std::cout << "{\"target\":\"unassigned\",\"work_units\":0}\n";
      return 0;
    }
    std::cout << "{\"target\":\"" << target_name(result->target)
              << "\",\"work_units\":" << result->work_units << "}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "rj_fuzz: " << error.what() << '\n';
    return 1;
  }
}
