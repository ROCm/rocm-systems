// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "execution_fuzz_core.h"

#include "rocjitsu/vm/amdgpu/partitioning.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#ifdef __AFL_HAVE_MANUAL_CONTROL
__AFL_FUZZ_INIT();
#endif

namespace {
using namespace rocjitsu::execution_fuzz;

constexpr uint64_t kSeedCount = 128;

enum class SeedEmissionStatus { Succeeded, InvalidInput, Failed };

util::Result parse_uint64(std::string_view text, uint64_t &result,
                          util::DiagnosticEmitter diagnostic) {
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
    return diagnostic.emit() << "expected an unsigned integer";
  return util::Result::success();
}

bool seed_range_overflows(uint64_t seed, uint64_t count) {
  return count != 0 && count - 1 > std::numeric_limits<uint64_t>::max() - seed;
}

util::Result write_input(const std::filesystem::path &path, std::span<const uint8_t> data,
                         util::DiagnosticEmitter diagnostic) {
  std::ofstream stream(path, std::ios::binary);
  stream.write(reinterpret_cast<const char *>(data.data()), data.size());
  stream.close();
  if (!stream)
    return diagnostic.emit() << "cannot write input: " << path.string();
  return util::Result::success();
}

util::Result write_seed(const std::filesystem::path &path, uint64_t seed,
                        util::DiagnosticEmitter diagnostic) {
  const auto temporary = path.string() + ".tmp";
  std::ofstream stream(temporary);
  stream << seed << '\n';
  stream.close();
  if (!stream)
    return diagnostic.emit() << "cannot write seed: " << path.string();
  std::error_code error;
  std::filesystem::rename(temporary, path, error);
  if (error)
    return diagnostic.emit() << "cannot publish seed: " << path << ": " << error.message();
  return util::Result::success();
}

util::Result create_directory(const std::filesystem::path &path,
                              util::DiagnosticEmitter diagnostic) {
  std::error_code error;
  std::filesystem::create_directories(path, error);
  if (error)
    return diagnostic.emit() << "cannot create directory: " << path << ": " << error.message();
  return util::Result::success();
}

util::Result remove_file(const std::filesystem::path &path, util::DiagnosticEmitter diagnostic) {
  std::error_code error;
  std::filesystem::remove(path, error);
  if (error)
    return diagnostic.emit() << "cannot remove artifact: " << path << ": " << error.message();
  return util::Result::success();
}

SeedEmissionStatus emit_seeds(const std::filesystem::path &directory, uint64_t first_seed,
                              std::optional<FuzzTarget> target,
                              util::DiagnosticEmitter diagnostic) {
  const auto reject = [&](std::string_view message, SeedEmissionStatus status) {
    (void)(diagnostic.emit() << message).emit();
    return status;
  };
  std::error_code error;
  const bool exists = std::filesystem::exists(directory, error);
  if (error)
    return reject("cannot inspect seed destination: " + error.message(),
                  SeedEmissionStatus::Failed);
  if (exists) {
    const bool empty = std::filesystem::is_empty(directory, error);
    if (error)
      return reject("cannot inspect seed destination: " + error.message(),
                    SeedEmissionStatus::Failed);
    if (!empty)
      return reject("seed destination must be an empty directory",
                    SeedEmissionStatus::InvalidInput);
    const bool is_directory = std::filesystem::is_directory(directory, error);
    if (error)
      return reject("cannot inspect seed destination: " + error.message(),
                    SeedEmissionStatus::Failed);
    if (!is_directory)
      return reject("seed destination must be an empty directory",
                    SeedEmissionStatus::InvalidInput);
  }
  if (create_directory(directory, diagnostic).failed())
    return SeedEmissionStatus::Failed;
  if (seed_range_overflows(first_seed, kSeedCount))
    return reject("seed range exceeds uint64", SeedEmissionStatus::InvalidInput);
  constexpr std::array<size_t, 4> kSeedSizes = {8, 32, kGeneratedKernelSeedSize, kMaxInputSize};
  const size_t kernel_seeds = target ? 1 : static_cast<size_t>(FuzzTarget::Count);
  for (uint64_t index = 0; index != kSeedCount; ++index) {
    const uint64_t seed = first_seed + index;
    const util::FailureOr<std::vector<uint8_t>> input =
        index < kernel_seeds
            ? generated_kernel_input(seed, kGeneratedKernelSeedSize,
                                     target.value_or(static_cast<FuzzTarget>(index)), diagnostic)
            : random_input(seed, kSeedSizes[index % kSeedSizes.size()], target, diagnostic);
    if (input.failed() ||
        write_input(directory / std::format("seed-{:04}.bin", index), input.value(), diagnostic)
            .failed())
      return SeedEmissionStatus::Failed;
  }
  std::cout << "wrote " << kSeedCount << " execution seeds to " << directory << '\n';
  return SeedEmissionStatus::Succeeded;
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
    if (size > 0 && size <= kMaxInputSize &&
        run_input(std::span<const uint8_t>(buffer, size), target, async_resources, async_mode,
                  issuers)
            .failed())
      std::abort();
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
  util::StringDiagnostic diagnostic;
  const auto fail = [&](std::string_view message) {
    std::cerr << "rj_execution_fuzz: " << message << '\n';
    return 1;
  };
  const auto invalid = [&](std::string_view message) {
    std::cerr << "rj_execution_fuzz: " << message << '\n';
    usage(std::cerr);
    return 2;
  };
  std::string input;
  std::string output;
  std::string seed_directory;
  uint64_t cases = 0;
  uint64_t seed = 0;
  uint64_t async_helpers = 1;
  bool afl = false;
  std::optional<FuzzTarget> target;
  uint64_t threads = 0;
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
    else if (arg == "--random" && i + 1 < argc) {
      if (parse_uint64(argv[++i], cases, diagnostic.emitter()).failed())
        return invalid(diagnostic.message());
    } else if (arg == "--seed" && i + 1 < argc) {
      if (parse_uint64(argv[++i], seed, diagnostic.emitter()).failed())
        return invalid(diagnostic.message());
    } else if (arg == "--threads" && i + 1 < argc) {
      const std::string_view value(argv[++i]);
      if (value == "auto") {
        threads = 0;
      } else {
        if (parse_uint64(value, threads, diagnostic.emitter()).failed())
          return invalid(diagnostic.message());
        if (threads == 0)
          return invalid("threads must be positive");
      }
    } else if (arg == "--issuers" && i + 1 < argc) {
      const std::string_view value(argv[++i]);
      if (value == "auto") {
        issuers.reset();
      } else {
        uint64_t count = 0;
        if (parse_uint64(value, count, diagnostic.emitter()).failed())
          return invalid(diagnostic.message());
        if (count == 0 || count > 8)
          return invalid("issuers must be auto or 1..8");
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
        return invalid("async mode must be auto, on, or off");
    } else if (arg == "--async-helpers" && i + 1 < argc) {
      if (parse_uint64(argv[++i], async_helpers, diagnostic.emitter()).failed())
        return invalid(diagnostic.message());
    } else if (arg == "--target" && i + 1 < argc) {
      const std::string_view name(argv[++i]);
      if (name == "auto") {
        target.reset();
      } else {
        target = parse_target(name);
        if (!target)
          return invalid("unknown execution fuzz target");
      }
    } else
      return invalid("unknown or incomplete argument");
  }
  if (async_helpers > 128)
    return invalid("async helpers must be 0..128");
  if (async_mode == AsyncMode::Enabled && async_helpers == 0)
    return invalid("--async on requires at least one async helper");
  if (!seed_directory.empty()) {
    if (afl || !input.empty() || cases != 0 || !output.empty() || threads || issuers ||
        async_mode != AsyncMode::Auto || async_helpers != 1)
      return invalid("seed emission cannot be combined with execution modes");
    const SeedEmissionStatus status =
        emit_seeds(seed_directory, seed, target, diagnostic.emitter());
    if (status == SeedEmissionStatus::InvalidInput)
      return invalid(diagnostic.message());
    if (status == SeedEmissionStatus::Failed)
      return fail(diagnostic.message());
    return 0;
  }
  std::shared_ptr<rocjitsu::amdgpu::matrix_coexecution::ExecutionResources> async_resources;
  if (async_mode != AsyncMode::Disabled && async_helpers) {
    util::FailureOr<std::shared_ptr<rocjitsu::amdgpu::matrix_coexecution::ExecutionResources>>
        resources = make_async_execution_resources(static_cast<unsigned>(async_helpers),
                                                   diagnostic.emitter());
    if (resources.failed())
      return fail(diagnostic.message());
    async_resources = std::move(resources).value();
  }
  if (afl) {
    if (!input.empty() || cases != 0 || !output.empty() || (threads && threads != 1))
      return invalid("AFL mode cannot be combined with file or random mode");
    return run_afl(target, async_resources, async_mode, issuers);
  }
  if ((input.empty() == (cases == 0)) || (cases && output.empty()))
    return invalid("choose --input FILE or --random COUNT --output DIRECTORY");
  if (!input.empty() && threads && threads != 1)
    return invalid("threaded execution requires random mode");
  if (seed_range_overflows(seed, cases))
    return invalid("seed range exceeds uint64");

  RunStats total;
  uint64_t reported_threads = 1;
  if (!input.empty()) {
    std::ifstream stream(input, std::ios::binary | std::ios::ate);
    if (!stream || stream.tellg() < 1 ||
        stream.tellg() > static_cast<std::streamoff>(kMaxInputSize))
      return invalid("input has an invalid size");
    std::vector<uint8_t> bytes(static_cast<size_t>(stream.tellg()));
    stream.seekg(0);
    if (!stream.read(reinterpret_cast<char *>(bytes.data()), bytes.size()))
      return fail("cannot read input");
    const util::FailureOr<RunStats> result =
        run_input(bytes, target, async_resources, async_mode, issuers, diagnostic.emitter());
    if (result.failed())
      return fail(diagnostic.message());
    total = result.value();
  } else {
    if (create_directory(output, diagnostic.emitter()).failed())
      return fail(diagnostic.message());
    const uint64_t automatic_threads =
        std::min<uint64_t>(8, std::max(1u, rocjitsu::amdgpu::available_host_threads()));
    reported_threads = std::min((threads ? threads : automatic_threads), cases);
    std::vector<RunStats> thread_stats(reported_threads);
    struct Failure {
      uint64_t seed = 0;
      std::string message;
      bool replay_ready = false;
    };
    std::vector<std::optional<Failure>> failures(reported_threads);
    std::atomic<bool> stop{false};
    std::vector<std::jthread> workers;
    workers.reserve(reported_threads);
    if (reported_threads != 1)
      for (uint64_t worker = 0; worker != reported_threads; ++worker)
        if (create_directory(std::filesystem::path(output) / std::format("thread-{}", worker),
                             diagnostic.emitter())
                .failed())
          return fail(diagnostic.message());
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
          util::StringDiagnostic failure;
          bool replay_ready = false;
          const auto execute = [&]() -> util::Result {
            const util::FailureOr<std::vector<uint8_t>> maybe_bytes =
                random_input(current_seed, kMaxInputSize, target, failure.emitter());
            if (maybe_bytes.failed())
              return util::Result::failure();
            const std::vector<uint8_t> &bytes = maybe_bytes.value();
            // A seed marker publishes a complete input. Invalidate it before replacing bytes.
            if (remove_file(replay_seed, failure.emitter()).failed() ||
                write_input(replay, bytes, failure.emitter()).failed() ||
                write_seed(replay_seed, current_seed, failure.emitter()).failed())
              return util::Result::failure();
            replay_ready = true;
            const util::FailureOr<RunStats> result =
                run_input(bytes, target, async_resources, async_mode, issuers, failure.emitter());
            if (result.failed())
              return util::Result::failure();
            accumulate_stats(thread_stats[worker], result.value());
            return util::Result::success();
          };
          if (execute().failed()) {
            failures[worker] = Failure{current_seed, failure.message(), replay_ready};
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
      if (failure->replay_ready) {
        std::error_code error;
        std::filesystem::copy_file(work / "current.bin",
                                   std::filesystem::path(output) / "failure.bin",
                                   std::filesystem::copy_options::overwrite_existing, error);
        if (error)
          return fail("cannot preserve failure input: " + error.message());
        if (write_seed(std::filesystem::path(output) / "failure.seed", failure->seed,
                       diagnostic.emitter())
                .failed())
          return fail(diagnostic.message());
      }
      std::cerr << "seed=" << failure->seed << " replay=" << output << ": " << failure->message
                << '\n';
      return 1;
    }
    for (uint64_t worker = 0; worker != reported_threads; ++worker) {
      accumulate_stats(total, thread_stats[worker]);
      const auto work = reported_threads == 1
                            ? std::filesystem::path(output)
                            : std::filesystem::path(output) / std::format("thread-{}", worker);
      if (remove_file(work / "current.bin", diagnostic.emitter()).failed() ||
          remove_file(work / "current.seed", diagnostic.emitter()).failed() ||
          (reported_threads != 1 && remove_file(work, diagnostic.emitter()).failed()))
        return fail(diagnostic.message());
    }
  }

  const std::string_view reported_target = target ? target_name(*target) : std::string_view("auto");
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
            << ",\"budget_exhausted\":" << total.budget_exhausted << ",\"inputs\":" << total.inputs
            << ",\"inputs_executed\":" << total.inputs_executed
            << ",\"inputs_halted\":" << total.inputs_halted
            << ",\"inputs_stopped\":" << total.inputs_stopped << ",\"threads\":" << reported_threads
            << ",\"issuers\":\"" << (issuers ? std::to_string(*issuers) : "auto") << "\""
            << ",\"async\":\"" << reported_async << "\""
            << ",\"async_helpers\":" << (async_resources ? async_helpers : 0) << "}\n";
  return 0;
}
