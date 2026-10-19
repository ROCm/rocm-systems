// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "execution_fuzz_core.h"

#include "rocjitsu/vm/amdgpu/partitioning.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#ifdef __AFL_HAVE_MANUAL_CONTROL
__AFL_FUZZ_INIT();
#endif

namespace {
using namespace rocjitsu::execution_fuzz;

uint64_t parse_uint64(std::string_view text) {
  uint64_t result = 0;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
    throw std::invalid_argument("expected an unsigned integer");
  return result;
}

void write_input(const std::filesystem::path &path, std::span<const uint8_t> data) {
  std::ofstream stream(path, std::ios::binary);
  if (!stream.write(reinterpret_cast<const char *>(data.data()), data.size()))
    throw std::runtime_error("cannot write input: " + path.string());
}

void write_seed(const std::filesystem::path &path, uint64_t seed) {
  std::ofstream stream(path);
  if (!(stream << seed << '\n'))
    throw std::runtime_error("cannot write seed: " + path.string());
}

void emit_seeds(const std::filesystem::path &directory, uint64_t first_seed,
                std::optional<FuzzTarget> target) {
  if (std::filesystem::exists(directory) &&
      (!std::filesystem::is_directory(directory) ||
       std::filesystem::directory_iterator(directory) != std::filesystem::directory_iterator()))
    throw std::invalid_argument("seed destination must be an empty directory");
  std::filesystem::create_directories(directory);
  constexpr uint64_t kSeedCount = 128;
  constexpr std::array<size_t, 4> kSeedSizes = {8, 32, 256, kMaxInputSize};
  for (uint64_t index = 0; index != kSeedCount; ++index) {
    const uint64_t seed = first_seed + index;
    write_input(directory / std::format("seed-{:04}.bin", index),
                random_input(seed, kSeedSizes[index % kSeedSizes.size()], target));
  }
  std::cout << "wrote " << kSeedCount << " execution seeds to " << directory << '\n';
}

void accumulate_stats(RunStats &total, const RunStats &result) {
  total.executed += result.executed;
  total.async_executed += result.async_executed;
  total.async_capable_inputs += result.async_capable_inputs;
  total.async_enabled_inputs += result.async_enabled_inputs;
  total.concurrent_inputs += result.concurrent_inputs;
  total.issuer_executions += result.issuer_executions;
  total.waves_halted += result.waves_halted;
  total.rounds += result.rounds;
  total.budget_exhausted += result.budget_exhausted;
  total.inputs += result.inputs;
  total.inputs_executed += result.inputs_executed;
  total.inputs_halted += result.inputs_halted;
  total.inputs_stopped += result.inputs_stopped;
}

void usage(std::ostream &stream) {
  stream << "Usage: rj_execution_fuzz --input FILE [--target auto|NAME]\n"
            "       rj_execution_fuzz --random COUNT [--seed N] [--threads auto|N]\n"
            "           [--target auto|NAME] [--issuers auto|N] [--async auto|on|off]\n"
            "           [--async-helpers N] --output DIRECTORY\n"
            "       rj_execution_fuzz --emit-seeds DIRECTORY [--seed N] [--target auto|NAME]\n"
            "       rj_execution_fuzz --afl [--target auto|NAME]\n"
            "           [--issuers auto|N] [--async auto|on|off] [--async-helpers N]\n";
}

int run_afl(std::optional<FuzzTarget> target,
            const std::shared_ptr<rocjitsu::amdgpu::matrix_coexecution::ExecutionResources>
                &async_resources,
            AsyncMode async_mode, std::optional<unsigned> issuers) {
#ifdef __AFL_HAVE_MANUAL_CONTROL
  __AFL_INIT();
  unsigned char *buffer = __AFL_FUZZ_TESTCASE_BUF;
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-statement-expression-from-macro-expansion"
#endif
  while (__AFL_LOOP(1000)) {
    const size_t size = __AFL_FUZZ_TESTCASE_LEN;
    if (size > 0 && size <= kMaxInputSize)
      (void)run_input(std::span<const uint8_t>(buffer, size), target, async_resources, async_mode,
                      issuers);
  }
#ifdef __clang__
#pragma clang diagnostic pop
#endif
  return 0;
#else
  (void)target;
  (void)async_resources;
  (void)async_mode;
  (void)issuers;
  std::cerr << "rj_execution_fuzz requires an AFL++ compiler wrapper for --afl\n";
  return 2;
#endif
}
} // namespace

int main(int argc, char **argv) {
  bool fuzz_loop_running = false;
  try {
    std::string input;
    std::string output;
    std::string seed_directory;
    uint64_t cases = 0;
    uint64_t seed = 0;
    uint64_t async_helpers = 1;
    bool afl = false;
    std::optional<FuzzTarget> target;
    std::optional<uint64_t> threads;
    std::optional<unsigned> issuers;
    AsyncMode async_mode = AsyncMode::Auto;
    for (int i = 1; i < argc; ++i) {
      const std::string_view arg(argv[i]);
      if (arg == "--help") {
        usage(std::cout);
        return 0;
      } else if (arg == "--afl")
        afl = true;
      else if (arg == "--input" && i + 1 < argc)
        input = argv[++i];
      else if (arg == "--output" && i + 1 < argc)
        output = argv[++i];
      else if (arg == "--emit-seeds" && i + 1 < argc)
        seed_directory = argv[++i];
      else if (arg == "--random" && i + 1 < argc)
        cases = parse_uint64(argv[++i]);
      else if (arg == "--seed" && i + 1 < argc)
        seed = parse_uint64(argv[++i]);
      else if (arg == "--threads" && i + 1 < argc) {
        const std::string_view value(argv[++i]);
        if (value == "auto")
          threads.reset();
        else
          threads = parse_uint64(value);
      } else if (arg == "--issuers" && i + 1 < argc) {
        const std::string_view value(argv[++i]);
        if (value == "auto") {
          issuers.reset();
        } else {
          const uint64_t count = parse_uint64(value);
          if (count == 0 || count > 8)
            throw std::invalid_argument("issuers must be auto or 1..8");
          issuers = static_cast<unsigned>(count);
        }
      } else if (arg == "--async" && i + 1 < argc) {
        const std::string_view value(argv[++i]);
        if (value == "auto")
          async_mode = AsyncMode::Auto;
        else if (value == "off")
          async_mode = AsyncMode::Disabled;
        else if (value == "on")
          async_mode = AsyncMode::Enabled;
        else
          throw std::invalid_argument("async mode must be auto, on, or off");
      } else if (arg == "--async-helpers" && i + 1 < argc)
        async_helpers = parse_uint64(argv[++i]);
      else if (arg == "--target" && i + 1 < argc) {
        const std::string_view name(argv[++i]);
        if (name == "auto") {
          target.reset();
        } else {
          target = parse_target(name);
          if (!target)
            throw std::invalid_argument("unknown execution fuzz target");
        }
      } else
        throw std::invalid_argument("unknown or incomplete argument");
    }
    if ((threads && *threads == 0) || async_helpers > 128)
      throw std::invalid_argument("threads must be positive and async helpers must be 0..128");
    if (async_mode == AsyncMode::Enabled && async_helpers == 0)
      throw std::invalid_argument("--async on requires at least one async helper");
    if (!seed_directory.empty()) {
      if (afl || !input.empty() || cases != 0 || !output.empty() || threads || issuers ||
          async_mode != AsyncMode::Auto || async_helpers != 1)
        throw std::invalid_argument("seed emission cannot be combined with execution modes");
      emit_seeds(seed_directory, seed, target);
      return 0;
    }
    const auto async_resources =
        async_mode != AsyncMode::Disabled && async_helpers
            ? make_async_execution_resources(static_cast<unsigned>(async_helpers))
            : nullptr;
    if (afl) {
      if (!input.empty() || cases != 0 || !output.empty() || (threads && *threads != 1))
        throw std::invalid_argument("AFL mode cannot be combined with file or random mode");
      fuzz_loop_running = true;
      return run_afl(target, async_resources, async_mode, issuers);
    }
    if ((input.empty() == (cases == 0)) || (cases && output.empty()))
      throw std::invalid_argument("choose --input FILE or --random COUNT --output DIRECTORY");
    if (!input.empty() && threads && *threads != 1)
      throw std::invalid_argument("threaded execution requires random mode");

    RunStats total;
    uint64_t reported_threads = 1;
    if (!input.empty()) {
      std::ifstream stream(input, std::ios::binary | std::ios::ate);
      if (!stream || stream.tellg() < 1 ||
          stream.tellg() > static_cast<std::streamoff>(kMaxInputSize))
        throw std::invalid_argument("input has an invalid size");
      std::vector<uint8_t> bytes(static_cast<size_t>(stream.tellg()));
      stream.seekg(0);
      if (!stream.read(reinterpret_cast<char *>(bytes.data()), bytes.size()))
        throw std::runtime_error("cannot read input");
      total = run_input(bytes, target, async_resources, async_mode, issuers);
    } else {
      std::filesystem::create_directories(output);
      std::filesystem::remove(std::filesystem::path(output) / "failure.bin");
      std::filesystem::remove(std::filesystem::path(output) / "failure.seed");
      const uint64_t automatic_threads =
          std::min<uint64_t>(8, std::max(1u, rocjitsu::amdgpu::available_host_threads()));
      reported_threads = std::min(threads.value_or(automatic_threads), cases);
      std::vector<RunStats> thread_stats(reported_threads);
      struct Failure {
        uint64_t seed = 0;
        std::string message;
      };
      std::vector<std::optional<Failure>> failures(reported_threads);
      std::atomic<bool> stop{false};
      std::vector<std::thread> workers;
      workers.reserve(reported_threads);
      if (reported_threads != 1)
        for (uint64_t worker = 0; worker != reported_threads; ++worker)
          std::filesystem::create_directories(std::filesystem::path(output) /
                                              std::format("thread-{}", worker));
      for (uint64_t worker = 0; worker != reported_threads; ++worker) {
        workers.emplace_back([&, worker] {
          const auto work = reported_threads == 1
                                ? std::filesystem::path(output)
                                : std::filesystem::path(output) / std::format("thread-{}", worker);
          const auto replay = work / "current.bin";
          const auto replay_seed = work / "current.seed";
          for (uint64_t i = worker; i < cases && !stop.load(std::memory_order_relaxed);
               i += reported_threads) {
            const uint64_t current_seed = seed + i;
            try {
              const auto bytes = random_input(current_seed, kMaxInputSize, target);
              write_input(replay, bytes);
              write_seed(replay_seed, current_seed);
              const RunStats result =
                  run_input(bytes, target, async_resources, async_mode, issuers);
              accumulate_stats(thread_stats[worker], result);
            } catch (const std::exception &error) {
              failures[worker] = Failure{current_seed, error.what()};
              stop.store(true, std::memory_order_relaxed);
            } catch (...) {
              failures[worker] = Failure{current_seed, "unknown host exception"};
              stop.store(true, std::memory_order_relaxed);
            }
          }
        });
      }
      for (auto &worker : workers)
        worker.join();
      for (uint64_t worker = 0; worker != reported_threads; ++worker) {
        const auto &failure = failures[worker];
        if (!failure)
          continue;
        const auto work = reported_threads == 1
                              ? std::filesystem::path(output)
                              : std::filesystem::path(output) / std::format("thread-{}", worker);
        std::filesystem::copy_file(work / "current.bin",
                                   std::filesystem::path(output) / "failure.bin",
                                   std::filesystem::copy_options::overwrite_existing);
        write_seed(std::filesystem::path(output) / "failure.seed", failure->seed);
        std::cerr << "seed=" << failure->seed << " replay=" << output << ": " << failure->message
                  << '\n';
        return 1;
      }
      for (uint64_t worker = 0; worker != reported_threads; ++worker) {
        accumulate_stats(total, thread_stats[worker]);
        const auto work = reported_threads == 1
                              ? std::filesystem::path(output)
                              : std::filesystem::path(output) / std::format("thread-{}", worker);
        std::filesystem::remove(work / "current.bin");
        std::filesystem::remove(work / "current.seed");
        if (reported_threads != 1)
          std::filesystem::remove(work);
      }
    }

    const std::string_view reported_target =
        target ? target_name(*target) : std::string_view("auto");
    const std::string_view reported_async = async_mode == AsyncMode::Auto      ? "auto"
                                            : async_mode == AsyncMode::Enabled ? "on"
                                                                               : "off";
    std::cout << "{\"target\":\"" << reported_target << "\",\"executed\":" << total.executed
              << ",\"async_executed\":" << total.async_executed
              << ",\"async_capable_inputs\":" << total.async_capable_inputs
              << ",\"async_enabled_inputs\":" << total.async_enabled_inputs
              << ",\"concurrent_inputs\":" << total.concurrent_inputs
              << ",\"issuer_executions\":" << total.issuer_executions
              << ",\"waves_halted\":" << total.waves_halted << ",\"rounds\":" << total.rounds
              << ",\"budget_exhausted\":" << total.budget_exhausted
              << ",\"inputs\":" << total.inputs << ",\"inputs_executed\":" << total.inputs_executed
              << ",\"inputs_halted\":" << total.inputs_halted
              << ",\"inputs_stopped\":" << total.inputs_stopped
              << ",\"threads\":" << reported_threads << ",\"issuers\":\""
              << (issuers ? std::to_string(*issuers) : "auto") << "\"" << ",\"async\":\""
              << reported_async << "\""
              << ",\"async_helpers\":" << (async_resources ? async_helpers : 0) << "}\n";
    return 0;
  } catch (const std::invalid_argument &error) {
    if (fuzz_loop_running)
      std::terminate();
    std::cerr << error.what() << '\n';
    usage(std::cerr);
    return 2;
  } catch (const std::exception &error) {
    if (fuzz_loop_running)
      std::terminate();
    std::cerr << error.what() << '\n';
    return 1;
  }
}
