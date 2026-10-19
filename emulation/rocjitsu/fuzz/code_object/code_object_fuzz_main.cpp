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
#include <stdexcept>
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

uint64_t parse_uint64(std::string_view text) {
  uint64_t value = 0;
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size())
    throw std::invalid_argument("expected an unsigned integer");
  return value;
}

void write_input(const std::filesystem::path &path, std::span<const uint8_t> bytes) {
  const auto temporary = path.string() + ".tmp";
  {
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    if (!stream.write(reinterpret_cast<const char *>(bytes.data()), bytes.size()))
      throw std::runtime_error("cannot write fuzz input: " + path.string());
  }
  std::filesystem::rename(temporary, path);
}

void write_seed(const std::filesystem::path &path, uint64_t seed) {
  const auto temporary = path.string() + ".tmp";
  {
    std::ofstream stream(temporary, std::ios::trunc);
    if (!(stream << seed << '\n'))
      throw std::runtime_error("cannot write fuzz seed: " + path.string());
  }
  std::filesystem::rename(temporary, path);
}

void update_active_seed(std::ofstream &stream, uint64_t seed) {
  const std::string record = std::format("{:020}\n", seed);
  stream.seekp(0);
  stream.write(record.data(), record.size());
  stream.flush();
  if (!stream)
    throw std::runtime_error("cannot update active fuzz seed");
}

std::vector<uint8_t> read_input(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream || stream.tellg() < 0 || stream.tellg() > static_cast<std::streamoff>(kMaxInputSize))
    throw std::invalid_argument("input has an invalid size");
  std::vector<uint8_t> bytes(static_cast<std::size_t>(stream.tellg()));
  stream.seekg(0);
  if (!bytes.empty() && !stream.read(reinterpret_cast<char *>(bytes.data()), bytes.size()))
    throw std::runtime_error("cannot read input");
  return bytes;
}

void emit_seeds(const std::filesystem::path &directory) {
  if (std::filesystem::exists(directory) &&
      (!std::filesystem::is_directory(directory) ||
       std::filesystem::directory_iterator(directory) != std::filesystem::directory_iterator()))
    throw std::invalid_argument("seed destination must be an empty directory");
  std::filesystem::create_directories(directory);
  const auto seeds = seed_corpus();
  for (std::size_t index = 0; index != seeds.size(); ++index)
    write_input(directory / std::format("seed-{:04}.bin", index), seeds[index]);
  std::cout << "wrote " << seeds.size() << " code-object seeds to " << directory << '\n';
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
  std::optional<uint64_t> requested_threads;
  bool afl = false;

  try {
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
        cases = parse_uint64(argv[++index]);
      } else if (argument == "--seed" && index + 1 < argc) {
        first_seed = parse_uint64(argv[++index]);
      } else if (argument == "--threads" && index + 1 < argc) {
        const std::string_view value(argv[++index]);
        if (value == "auto")
          requested_threads.reset();
        else
          requested_threads = parse_uint64(value);
      } else if (argument == "--output" && index + 1 < argc) {
        output = argv[++index];
      } else if (argument == "--emit-seeds" && index + 1 < argc) {
        seed_directory = argv[++index];
      } else {
        throw std::invalid_argument("unknown or incomplete argument");
      }
    }

    if (requested_threads && *requested_threads == 0)
      throw std::invalid_argument("threads must be positive");
    if (!seed_directory.empty()) {
      if (afl || !input.empty() || cases != 0 || !output.empty() || requested_threads)
        throw std::invalid_argument("seed emission cannot be combined with fuzz execution");
    }
    if (afl) {
      if (!input.empty() || cases != 0 || !output.empty() || requested_threads)
        throw std::invalid_argument("AFL mode cannot be combined with other execution modes");
    }
    if (seed_directory.empty() && !afl &&
        ((input.empty() == (cases == 0)) || (cases != 0 && output.empty())))
      throw std::invalid_argument("choose --input FILE or --random COUNT --output DIRECTORY");
    if (!input.empty() && requested_threads && *requested_threads != 1)
      throw std::invalid_argument("threaded execution requires random mode");
  } catch (const std::exception &error) {
    std::cerr << "rj_code_object_fuzz: " << error.what() << '\n';
    usage(std::cerr);
    return 2;
  }

  // Unexpected loader exceptions must terminate the AFL child by signal so
  // AFL++ retains the input. Other modes keep friendly diagnostics.
  if (afl)
    return run_afl();

  try {
    if (!seed_directory.empty()) {
      emit_seeds(seed_directory);
      return 0;
    }
    RunStats total;
    uint64_t threads = 1;
    if (!input.empty()) {
      total = run_input(read_input(input));
    } else {
      std::filesystem::create_directories(output);
      std::filesystem::remove(std::filesystem::path(output) / "failure.bin");
      std::filesystem::remove(std::filesystem::path(output) / "failure.seed");
      const uint64_t automatic_threads = std::max(1u, std::thread::hardware_concurrency());
      threads = std::min(cases, requested_threads.value_or(automatic_threads));
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
        const auto directory =
            threads == 1 ? std::filesystem::path(output)
                         : std::filesystem::path(output) / std::format("thread-{}", worker);
        std::filesystem::create_directories(directory);
        std::filesystem::remove(directory / "current.bin");
        std::filesystem::remove(directory / "current.seed");
        worker_directories.push_back(directory);
      }

      std::vector<std::jthread> workers;
      workers.reserve(threads);
      for (uint64_t worker = 0; worker != threads; ++worker) {
        const std::filesystem::path &directory = worker_directories[worker];
        workers.emplace_back([&, worker, directory] {
          const auto current_seed = directory / "current.seed";
          std::ofstream replay_seed(current_seed, std::ios::trunc);
          if (!replay_seed) {
            failures[worker] = Failure{first_seed + worker, "cannot create active fuzz seed"};
            stop.store(true, std::memory_order_relaxed);
            return;
          }
          for (uint64_t index = worker; index < cases && !stop.load(std::memory_order_relaxed);
               index += threads) {
            const uint64_t seed = first_seed + index;
            try {
              update_active_seed(replay_seed, seed);
              const auto bytes = generate_input(seed);
              thread_stats[worker] += run_input(bytes);
            } catch (const std::exception &error) {
              failures[worker] = Failure{seed, error.what()};
              stop.store(true, std::memory_order_relaxed);
            } catch (...) {
              failures[worker] = Failure{seed, "unknown host exception"};
              stop.store(true, std::memory_order_relaxed);
            }
          }
        });
      }
      for (std::jthread &worker : workers)
        worker.join();

      for (uint64_t worker = 0; worker != threads; ++worker) {
        if (!failures[worker])
          continue;
        write_input(std::filesystem::path(output) / "failure.bin",
                    generate_input(failures[worker]->seed));
        write_seed(std::filesystem::path(output) / "failure.seed", failures[worker]->seed);
        std::cerr << "seed=" << failures[worker]->seed << " replay=" << output << ": "
                  << failures[worker]->message << '\n';
        return 1;
      }
      for (uint64_t worker = 0; worker != threads; ++worker) {
        total += thread_stats[worker];
        const std::filesystem::path &directory = worker_directories[worker];
        std::filesystem::remove(directory / "current.seed");
        if (threads != 1)
          std::filesystem::remove(directory);
      }
    }
    print_stats(total, threads);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "rj_code_object_fuzz: " << error.what() << '\n';
    usage(std::cerr);
    return 2;
  }
}
