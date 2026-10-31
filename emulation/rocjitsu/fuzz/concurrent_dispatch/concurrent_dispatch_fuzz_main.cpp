// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "concurrent_dispatch_fuzz_core.h"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifdef __AFL_HAVE_MANUAL_CONTROL
__AFL_FUZZ_INIT();
#endif

namespace {
using namespace rocjitsu::concurrent_dispatch_fuzz;

uint64_t parse_uint64(std::string_view text) {
  uint64_t value = 0;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
    throw std::invalid_argument("expected an unsigned integer");
  return value;
}

std::vector<uint8_t> read_input(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream || stream.tellg() <= 0 || stream.tellg() > static_cast<std::streamoff>(kMaxInputSize))
    throw std::invalid_argument("input has an invalid size");
  std::vector<uint8_t> input(static_cast<std::size_t>(stream.tellg()));
  stream.seekg(0);
  if (!stream.read(reinterpret_cast<char *>(input.data()), input.size()))
    throw std::runtime_error("cannot read input");
  return input;
}

void write_input(const std::filesystem::path &path, std::span<const uint8_t> input) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream.write(reinterpret_cast<const char *>(input.data()), input.size()))
    throw std::runtime_error("cannot write input: " + path.string());
}

void emit_seeds(const std::filesystem::path &directory) {
  if (std::filesystem::exists(directory) &&
      (!std::filesystem::is_directory(directory) ||
       std::filesystem::directory_iterator(directory) != std::filesystem::directory_iterator()))
    throw std::invalid_argument("seed destination must be an empty directory");
  std::filesystem::create_directories(directory);
  const auto seeds = seed_corpus();
  for (std::size_t index = 0; index < seeds.size(); ++index)
    write_input(directory / std::format("seed-{:04}.bin", index), seeds[index]);
  std::cout << "wrote " << seeds.size() << " concurrent-dispatch seeds to " << directory << '\n';
}

void print_stats(const RunStats &stats) {
  std::cout << "{\"inputs\":" << stats.inputs << ",\"targets\":" << stats.targets
            << ",\"modes\":" << stats.modes << ",\"completed_modes\":" << stats.completed_modes
            << ",\"budget_limited_modes\":" << stats.budget_limited_modes
            << ",\"terminal_faulted_modes\":" << stats.terminal_faulted_modes
            << ",\"instructions\":" << stats.instructions << ",\"packets\":" << stats.packets
            << ",\"workgroups\":" << stats.workgroups << ",\"workitems\":" << stats.workitems
            << ",\"kernel_bytes\":" << stats.kernel_bytes
            << ",\"completed_inputs\":" << stats.completed_inputs
            << ",\"incomplete_inputs\":" << stats.incomplete_inputs
            << ",\"budget_limited_inputs\":" << stats.budget_limited_inputs
            << ",\"multi_xcd_inputs\":" << stats.multi_xcd_inputs
            << ",\"cache_primed_inputs\":" << stats.cache_primed_inputs
            << ",\"fanout_inputs\":" << stats.fanout_inputs
            << ",\"differential_inputs\":" << stats.differential_inputs
            << ",\"pooled_dispatch_threads\":" << stats.pooled_dispatch_threads << "}\n";
}

void accumulate_stats(RunStats &total, const RunStats &stats) {
  total.inputs += stats.inputs;
  total.targets += stats.targets;
  total.modes += stats.modes;
  total.completed_modes += stats.completed_modes;
  total.budget_limited_modes += stats.budget_limited_modes;
  total.terminal_faulted_modes += stats.terminal_faulted_modes;
  total.instructions += stats.instructions;
  total.packets += stats.packets;
  total.workgroups += stats.workgroups;
  total.workitems += stats.workitems;
  total.kernel_bytes += stats.kernel_bytes;
  total.completed_inputs += stats.completed_inputs;
  total.incomplete_inputs += stats.incomplete_inputs;
  total.budget_limited_inputs += stats.budget_limited_inputs;
  total.multi_xcd_inputs += stats.multi_xcd_inputs;
  total.cache_primed_inputs += stats.cache_primed_inputs;
  total.fanout_inputs += stats.fanout_inputs;
  total.differential_inputs += stats.differential_inputs;
  total.pooled_dispatch_threads =
      std::max(total.pooled_dispatch_threads, stats.pooled_dispatch_threads);
}

void usage(std::ostream &stream) {
  stream << "Usage: rj_concurrent_dispatch_fuzz --input FILE\n"
            "       rj_concurrent_dispatch_fuzz --random COUNT [--seed N] --output DIRECTORY\n"
            "       rj_concurrent_dispatch_fuzz --emit-seeds DIRECTORY\n"
            "       rj_concurrent_dispatch_fuzz --afl\n";
}

int run_afl() {
#ifdef __AFL_HAVE_MANUAL_CONTROL
  __AFL_INIT();
  unsigned char *buffer = __AFL_FUZZ_TESTCASE_BUF;
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-statement-expression-from-macro-expansion"
#endif
  while (__AFL_LOOP(1000)) {
    const std::size_t size = __AFL_FUZZ_TESTCASE_LEN;
    if (size > 0 && size <= kMaxInputSize)
      (void)run_input(std::span<const uint8_t>(buffer, size));
  }
#ifdef __clang__
#pragma clang diagnostic pop
#endif
  return 0;
#else
  std::cerr << "rj_concurrent_dispatch_fuzz requires an AFL++ compiler wrapper for --afl\n";
  return 2;
#endif
}

} // namespace

int main(int argc, char **argv) {
  bool fuzz_loop_running = false;
  try {
    std::string input_path;
    std::string output_path;
    std::string seed_path;
    uint64_t cases = 0;
    uint64_t first_seed = 0;
    bool afl = false;
    for (int index = 1; index < argc; ++index) {
      const std::string_view argument(argv[index]);
      if (argument == "--help") {
        usage(std::cout);
        return 0;
      } else if (argument == "--afl") {
        afl = true;
      } else if (argument == "--input" && index + 1 < argc) {
        input_path = argv[++index];
      } else if (argument == "--output" && index + 1 < argc) {
        output_path = argv[++index];
      } else if (argument == "--emit-seeds" && index + 1 < argc) {
        seed_path = argv[++index];
      } else if (argument == "--random" && index + 1 < argc) {
        cases = parse_uint64(argv[++index]);
      } else if (argument == "--seed" && index + 1 < argc) {
        first_seed = parse_uint64(argv[++index]);
      } else {
        throw std::invalid_argument("unknown or incomplete argument");
      }
    }

    const unsigned modes = !input_path.empty() + (cases != 0) + !seed_path.empty() + afl;
    if (modes != 1)
      throw std::invalid_argument("choose exactly one execution mode");
    if (cases != 0 && output_path.empty())
      throw std::invalid_argument("random mode requires --output DIRECTORY");
    if (cases == 0 && (!output_path.empty() || first_seed != 0))
      throw std::invalid_argument("--output and --seed require random mode");

    if (afl) {
      fuzz_loop_running = true;
      return run_afl();
    }
    if (!seed_path.empty()) {
      emit_seeds(seed_path);
      return 0;
    }
    if (!input_path.empty()) {
      const std::vector<uint8_t> input = read_input(input_path);
      const RunStats stats = run_input(input);
      print_stats(stats);
      return 0;
    }

    std::filesystem::create_directories(output_path);
    RunStats total;
    for (uint64_t index = 0; index < cases; ++index) {
      const uint64_t seed = first_seed + index;
      const std::vector<uint8_t> input = random_input(seed);
      try {
        const RunStats stats = run_input(input);
        accumulate_stats(total, stats);
      } catch (...) {
        write_input(std::filesystem::path(output_path) / std::format("failure-seed-{}.bin", seed),
                    input);
        throw;
      }
    }
    print_stats(total);
    return 0;
  } catch (const std::invalid_argument &error) {
    if (fuzz_loop_running)
      std::terminate();
    std::cerr << "rj_concurrent_dispatch_fuzz: " << error.what() << '\n';
    usage(std::cerr);
    return 2;
  } catch (const std::exception &error) {
    if (fuzz_loop_running)
      std::terminate();
    std::cerr << "rj_concurrent_dispatch_fuzz: " << error.what() << '\n';
    return 1;
  }
}
