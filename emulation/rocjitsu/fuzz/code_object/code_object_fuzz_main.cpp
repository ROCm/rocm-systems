// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "code_object_fuzz_core.h"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#ifdef __AFL_HAVE_MANUAL_CONTROL
#include <unistd.h>

__AFL_FUZZ_INIT();
#endif

namespace {
using namespace rocjitsu::code_object_fuzz;

util::Result parse_uint64(std::string_view text, uint64_t &value,
                          util::DiagnosticEmitter diagnostic) {
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size())
    return diagnostic.emit() << "expected an unsigned integer";
  return util::Result::success();
}

util::Result publish_file(const std::filesystem::path &temporary, const std::filesystem::path &path,
                          util::DiagnosticEmitter diagnostic) {
  std::error_code error;
  std::filesystem::rename(temporary, path, error);
  if (error)
    return diagnostic.emit() << "cannot publish fuzz artifact: " << path << ": " << error.message();
  return util::Result::success();
}

util::Result write_input(const std::filesystem::path &path, std::span<const uint8_t> bytes,
                         util::DiagnosticEmitter diagnostic) {
  const auto temporary = path.string() + ".tmp";
  {
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    stream.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    stream.close();
    if (!stream)
      return diagnostic.emit() << "cannot write fuzz input: " << path.string();
  }
  return publish_file(temporary, path, diagnostic);
}

util::Result write_seed(const std::filesystem::path &path, uint64_t seed,
                        util::DiagnosticEmitter diagnostic) {
  const auto temporary = path.string() + ".tmp";
  {
    std::ofstream stream(temporary, std::ios::trunc);
    stream << seed << '\n';
    stream.close();
    if (!stream)
      return diagnostic.emit() << "cannot write fuzz seed: " << path.string();
  }
  return publish_file(temporary, path, diagnostic);
}

util::Result update_active_seed(std::ofstream &stream, uint64_t seed,
                                util::DiagnosticEmitter diagnostic) {
  const std::string record = std::format("{:020}\n", seed);
  stream.seekp(0);
  stream.write(record.data(), record.size());
  stream.flush();
  if (!stream)
    return diagnostic.emit() << "cannot update active fuzz seed";
  return util::Result::success();
}

util::FailureOr<std::vector<uint8_t>> read_input(const std::filesystem::path &path,
                                                 util::DiagnosticEmitter diagnostic) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream)
    return diagnostic.emit() << "input has an invalid size";
  const std::streamoff size = stream.tellg();
  if (size < 0 || size > static_cast<std::streamoff>(kMaxInputSize))
    return diagnostic.emit() << "input has an invalid size";
  std::vector<uint8_t> bytes(static_cast<std::size_t>(size));
  stream.seekg(0);
  if (!stream ||
      (!bytes.empty() && !stream.read(reinterpret_cast<char *>(bytes.data()), bytes.size())))
    return diagnostic.emit() << "cannot read input";
  return bytes;
}

util::Result create_directory(const std::filesystem::path &directory,
                              util::DiagnosticEmitter diagnostic) {
  std::error_code error;
  std::filesystem::create_directories(directory, error);
  if (error)
    return diagnostic.emit() << "cannot create fuzz directory: " << directory << ": "
                             << error.message();
  return util::Result::success();
}

util::Result remove_path(const std::filesystem::path &path, util::DiagnosticEmitter diagnostic) {
  std::error_code error;
  std::filesystem::remove(path, error);
  if (error)
    return diagnostic.emit() << "cannot remove fuzz path: " << path << ": " << error.message();
  return util::Result::success();
}

util::Result emit_seeds(const std::filesystem::path &directory,
                        util::DiagnosticEmitter diagnostic) {
  std::error_code error;
  const bool exists = std::filesystem::exists(directory, error);
  if (error)
    return diagnostic.emit() << "cannot inspect seed destination: " << error.message();
  if (exists) {
    const bool is_directory = std::filesystem::is_directory(directory, error);
    if (error)
      return diagnostic.emit() << "cannot inspect seed destination: " << error.message();
    if (!is_directory)
      return diagnostic.emit() << "seed destination must be an empty directory";
    const bool empty = std::filesystem::is_empty(directory, error);
    if (error)
      return diagnostic.emit() << "cannot inspect seed destination: " << error.message();
    if (!empty)
      return diagnostic.emit() << "seed destination must be an empty directory";
  }
  if (create_directory(directory, diagnostic).failed())
    return util::Result::failure();
  const auto seeds = seed_corpus();
  for (std::size_t index = 0; index != seeds.size(); ++index) {
    if (write_input(directory / std::format("seed-{:04}.bin", index), seeds[index], diagnostic)
            .failed())
      return util::Result::failure();
  }
  std::cout << "wrote " << seeds.size() << " code-object seeds to " << directory << '\n';
  return util::Result::success();
}

void print_stats(const RunStats &stats, uint64_t threads) {
  std::cout << "{\"inputs\":" << stats.inputs
            << ",\"code_objects_valid\":" << stats.code_objects_valid
            << ",\"executables_valid\":" << stats.executables_valid
            << ",\"bundled_code_objects\":" << stats.bundled_code_objects
            << ",\"sections\":" << stats.sections << ",\"text_sections\":" << stats.text_sections
            << ",\"descriptors\":" << stats.descriptors
            << ",\"load_segments\":" << stats.load_segments << ",\"threads\":" << threads << "}\n";
}

void usage(std::ostream &stream) {
  stream << "Usage: rj_code_object_fuzz --input FILE\n"
            "       rj_code_object_fuzz --random COUNT [--seed N] [--threads auto|N]\n"
            "           --output DIRECTORY\n"
            "       rj_code_object_fuzz --emit-seeds DIRECTORY\n"
            "       rj_code_object_fuzz --afl\n";
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
  while (__AFL_LOOP(1000)) {
    const std::size_t size = __AFL_FUZZ_TESTCASE_LEN;
    if (size <= kMaxInputSize)
      (void)run_input(std::span<const uint8_t>(buffer, size));
  }
#ifdef __clang__
#pragma clang diagnostic pop
#endif
  return 0;
#else
  std::cerr << "rj_code_object_fuzz requires an AFL++ compiler wrapper for --afl\n";
  return 2;
#endif
}

int main(int argc, char **argv) {
  std::string input;
  std::string output;
  std::string seed_directory;
  uint64_t cases = 0;
  uint64_t first_seed = 0;
  uint64_t requested_threads = 0;
  bool afl = false;

  util::StringDiagnostic diagnostic;
  const auto fail = [&](std::string_view message) {
    std::cerr << "rj_code_object_fuzz: " << message << '\n';
    usage(std::cerr);
    return 2;
  };
  const auto report_failure = [&] { return fail(diagnostic.message()); };

  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--help") {
      usage(std::cout);
      return 0;
    } else if (argument == "--afl") {
      afl = true;
    } else if (argument == "--input" && index + 1 < argc) {
      input = argv[++index];
    } else if (argument == "--random" && index + 1 < argc) {
      if (parse_uint64(argv[++index], cases, diagnostic.emitter()).failed())
        return report_failure();
    } else if (argument == "--seed" && index + 1 < argc) {
      if (parse_uint64(argv[++index], first_seed, diagnostic.emitter()).failed())
        return report_failure();
    } else if (argument == "--threads" && index + 1 < argc) {
      const std::string_view value(argv[++index]);
      if (value == "auto") {
        requested_threads = 0;
      } else {
        if (parse_uint64(value, requested_threads, diagnostic.emitter()).failed())
          return report_failure();
        if (requested_threads == 0)
          return fail("threads must be positive");
      }
    } else if (argument == "--output" && index + 1 < argc) {
      output = argv[++index];
    } else if (argument == "--emit-seeds" && index + 1 < argc) {
      seed_directory = argv[++index];
    } else {
      return fail("unknown or incomplete argument");
    }
  }

  if (!seed_directory.empty()) {
    if (afl || !input.empty() || cases != 0 || !output.empty() || requested_threads)
      return fail("seed emission cannot be combined with fuzz execution");
  }
  if (afl) {
    if (!input.empty() || cases != 0 || !output.empty() || requested_threads)
      return fail("AFL mode cannot be combined with other execution modes");
  }
  if (seed_directory.empty() && !afl &&
      ((input.empty() == (cases == 0)) || (cases != 0 && output.empty())))
    return fail("choose --input FILE or --random COUNT --output DIRECTORY");
  if (!input.empty() && requested_threads && requested_threads != 1)
    return fail("threaded execution requires random mode");
  // Loader invariant failures terminate the AFL child by signal so AFL++
  // retains the input.
  if (afl)
    return run_afl();

  if (!seed_directory.empty()) {
    if (emit_seeds(seed_directory, diagnostic.emitter()).failed())
      return report_failure();
    return 0;
  }
  RunStats total;
  uint64_t threads = 1;
  if (!input.empty()) {
    const util::FailureOr<std::vector<uint8_t>> bytes = read_input(input, diagnostic.emitter());
    if (bytes.failed())
      return report_failure();
    const util::FailureOr<RunStats> result = run_input(bytes.value(), diagnostic.emitter());
    if (result.failed())
      return report_failure();
    total = result.value();
  } else {
    if (create_directory(output, diagnostic.emitter()).failed())
      return report_failure();
    const uint64_t automatic_threads = std::max(1u, std::thread::hardware_concurrency());
    threads = std::min(cases, (requested_threads ? requested_threads : automatic_threads));
    std::vector<RunStats> thread_stats(threads);
    struct Failure {
      uint64_t seed;
      std::string message;
    };
    std::vector<std::optional<Failure>> failures(threads);
    std::atomic<bool> stop{false};
    std::vector<std::filesystem::path> worker_directories;
    worker_directories.reserve(threads);
    for (uint64_t worker = 0; worker != threads; ++worker) {
      const auto directory = threads == 1
                                 ? std::filesystem::path(output)
                                 : std::filesystem::path(output) / std::format("thread-{}", worker);
      if (create_directory(directory, diagnostic.emitter()).failed() ||
          remove_path(directory / "current.bin", diagnostic.emitter()).failed() ||
          remove_path(directory / "current.seed", diagnostic.emitter()).failed())
        return report_failure();
      worker_directories.push_back(directory);
    }

    std::vector<std::jthread> workers;
    workers.reserve(threads);
    for (uint64_t worker = 0; worker != threads; ++worker) {
      const std::filesystem::path &directory = worker_directories[worker];
      workers.emplace_back([&, worker, directory](std::stop_token token) {
        const auto current_seed = directory / "current.seed";
        std::ofstream replay_seed(current_seed, std::ios::trunc);
        if (!replay_seed) {
          failures[worker] = Failure{first_seed + worker, "cannot create active fuzz seed"};
          stop.store(true, std::memory_order_relaxed);
          return;
        }
        for (uint64_t index = worker;
             index < cases && !token.stop_requested() && !stop.load(std::memory_order_relaxed);
             index += threads) {
          const uint64_t seed = first_seed + index;
          util::StringDiagnostic worker_diagnostic;
          if (update_active_seed(replay_seed, seed, worker_diagnostic.emitter()).failed()) {
            failures[worker] = Failure{seed, worker_diagnostic.message()};
            stop.store(true, std::memory_order_relaxed);
            break;
          }
          const auto bytes = generate_input(seed);
          const util::FailureOr<RunStats> result = run_input(bytes, worker_diagnostic.emitter());
          if (result.failed()) {
            failures[worker] = Failure{seed, worker_diagnostic.message()};
            stop.store(true, std::memory_order_relaxed);
            break;
          }
          thread_stats[worker] += result.value();
        }
      });
    }
    for (std::jthread &worker : workers)
      worker.join();

    for (uint64_t worker = 0; worker != threads; ++worker) {
      if (!failures[worker])
        continue;
      if (write_input(std::filesystem::path(output) / "failure.bin",
                      generate_input(failures[worker]->seed), diagnostic.emitter())
              .failed() ||
          write_seed(std::filesystem::path(output) / "failure.seed", failures[worker]->seed,
                     diagnostic.emitter())
              .failed())
        return report_failure();
      std::cerr << "seed=" << failures[worker]->seed << " replay=" << output << ": "
                << failures[worker]->message << '\n';
      return 1;
    }
    for (uint64_t worker = 0; worker != threads; ++worker) {
      total += thread_stats[worker];
      const std::filesystem::path &directory = worker_directories[worker];
      if (remove_path(directory / "current.seed", diagnostic.emitter()).failed())
        return report_failure();
      if (threads != 1 && remove_path(directory, diagnostic.emitter()).failed())
        return report_failure();
    }
  }
  print_stats(total, threads);
  return 0;
}
