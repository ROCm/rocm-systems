// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "concurrent_dispatch_fuzz_core.h"

#include "util/diagnostic.h"

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <unistd.h>
#include <vector>

#ifdef __AFL_HAVE_MANUAL_CONTROL
#include <unistd.h>

__AFL_FUZZ_INIT();
#endif

namespace {
using namespace rocjitsu::concurrent_dispatch_fuzz;

enum class CliStatus { Succeeded, InvalidInput, Failed };

util::Result parse_uint64(std::string_view text, uint64_t &value,
                          util::DiagnosticEmitter diagnostic) {
  const std::from_chars_result parsed =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) {
    return diagnostic.emit() << "expected an unsigned integer";
  }
  return util::Result::success();
}

util::Result parse_path(std::string_view value, std::string &path,
                        util::DiagnosticEmitter diagnostic) {
  if (value.empty()) {
    return diagnostic.emit() << "expected a nonempty path";
  }
  path = value;
  return util::Result::success();
}

CliStatus read_input(const std::filesystem::path &path, std::vector<uint8_t> &input,
                     util::DiagnosticEmitter diagnostic) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream || stream.tellg() <= 0 ||
      stream.tellg() > static_cast<std::streamoff>(kMaxInputSize)) {
    (void)(diagnostic.emit() << "input has an invalid size").emit();
    return CliStatus::InvalidInput;
  }
  input.resize(static_cast<std::size_t>(stream.tellg()));
  stream.seekg(0);
  if (!stream.read(reinterpret_cast<char *>(input.data()), input.size())) {
    (void)(diagnostic.emit() << "cannot read input").emit();
    return CliStatus::Failed;
  }
  return CliStatus::Succeeded;
}

util::Result write_input(const std::filesystem::path &path, std::span<const uint8_t> input,
                         util::DiagnosticEmitter diagnostic) {
  const std::string filename = path.string();
  std::string temporary_filename = (path.parent_path() / ".rj-input-XXXXXX").string();
  const int temporary_fd = mkstemp(temporary_filename.data());
  if (temporary_fd < 0) {
    return diagnostic.emit() << "cannot stage input: " << filename << ": "
                             << std::error_code(errno, std::generic_category()).message();
  }
  const std::filesystem::path temporary(temporary_filename);
  std::FILE *stream = fdopen(temporary_fd, "wb");
  if (!stream) {
    const int error = errno;
    close(temporary_fd);
    unlink(temporary_filename.c_str());
    return diagnostic.emit() << "cannot stage input: " << filename << ": "
                             << std::error_code(error, std::generic_category()).message();
  }

  const bool written = std::fwrite(input.data(), 1, input.size(), stream) == input.size();
  const int close_result = std::fclose(stream);
  std::error_code publish_error;
  // A hard link publishes the complete file without replacing an existing
  // replay.
  if (written && close_result == 0) {
    std::filesystem::create_hard_link(temporary, path, publish_error);
  }
  std::error_code cleanup_error;
  std::filesystem::remove(temporary, cleanup_error);
  if (cleanup_error) {
    return diagnostic.emit() << "cannot remove staged input: " << temporary.string() << ": "
                             << cleanup_error.message();
  }
  if (!written || close_result != 0) {
    return diagnostic.emit() << "cannot write input: " << filename;
  }
  if (publish_error) {
    return diagnostic.emit() << "cannot create input: " << filename << ": "
                             << publish_error.message();
  }
  return util::Result::success();
}

util::Result create_directory(const std::filesystem::path &directory,
                              util::DiagnosticEmitter diagnostic) {
  std::error_code error;
  std::filesystem::create_directories(directory, error);
  if (error) {
    return diagnostic.emit() << "cannot create directory: " << directory << ": " << error.message();
  }
  return util::Result::success();
}

CliStatus emit_seeds(const std::filesystem::path &directory, util::DiagnosticEmitter diagnostic) {
  const auto reject = [&](std::string_view message, CliStatus status) {
    (void)(diagnostic.emit() << message).emit();
    return status;
  };
  std::error_code error;
  const bool exists = std::filesystem::exists(directory, error);
  if (error) {
    return reject("cannot inspect seed destination: " + error.message(), CliStatus::Failed);
  }
  if (exists) {
    const bool is_directory = std::filesystem::is_directory(directory, error);
    if (error) {
      return reject("cannot inspect seed destination: " + error.message(), CliStatus::Failed);
    }
    if (!is_directory) {
      return reject("seed destination must be an empty directory", CliStatus::InvalidInput);
    }
    const bool empty = std::filesystem::is_empty(directory, error);
    if (error) {
      return reject("cannot inspect seed destination: " + error.message(), CliStatus::Failed);
    }
    if (!empty) {
      return reject("seed destination must be an empty directory", CliStatus::InvalidInput);
    }
  }
  if (create_directory(directory, diagnostic).failed()) {
    return CliStatus::Failed;
  }
  const util::FailureOr<std::vector<std::vector<uint8_t>>> generated = seed_corpus(diagnostic);
  if (generated.failed()) {
    return CliStatus::Failed;
  }
  const std::vector<std::vector<uint8_t>> &seeds = generated.value();
  for (std::size_t index = 0; index < seeds.size(); ++index) {
    if (write_input(directory / std::format("seed-{:04}.bin", index), seeds[index], diagnostic)
            .failed()) {
      return CliStatus::Failed;
    }
  }
  std::cout << "wrote " << seeds.size() << " concurrent-dispatch seeds to " << directory << '\n';
  return CliStatus::Succeeded;
}

void print_stats(const RunStats &stats) {
  std::cout << "{\"inputs\":" << stats.inputs << ",\"targets\":" << stats.targets
            << ",\"modes\":" << stats.modes << ",\"completed_modes\":" << stats.completed_modes
            << ",\"budget_limited_modes\":" << stats.budget_limited_modes
            << ",\"primary_stalled_modes\":" << stats.primary_stalled_modes
            << ",\"fatal_exit_modes\":" << stats.fatal_exit_modes
            << ",\"terminal_faulted_modes\":" << stats.terminal_faulted_modes
            << ",\"published_completions\":" << stats.published_completions
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
  total.primary_stalled_modes += stats.primary_stalled_modes;
  total.fatal_exit_modes += stats.fatal_exit_modes;
  total.terminal_faulted_modes += stats.terminal_faulted_modes;
  total.published_completions += stats.published_completions;
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
            "       rj_concurrent_dispatch_fuzz --random COUNT [--seed N] "
            "--output DIRECTORY\n"
            "       rj_concurrent_dispatch_fuzz --emit-seeds DIRECTORY\n"
            "       rj_concurrent_dispatch_fuzz --afl\n";
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
    if (size > 0 && size <= kMaxInputSize) {
      if (run_input(std::span<const uint8_t>(buffer, size)).failed()) {
        std::abort();
      }
    }
  }
#ifdef __clang__
#pragma clang diagnostic pop
#endif
  return 0;
#else
  std::cerr << "rj_concurrent_dispatch_fuzz requires an AFL++ compiler wrapper "
               "for --afl\n";
  return 2;
#endif
}

int main(int argc, char **argv) {
  {
    util::StringDiagnostic diagnostic;
    const auto invalid = [&](std::string_view message) {
      std::cerr << "rj_concurrent_dispatch_fuzz: " << message << '\n';
      usage(std::cerr);
      return 2;
    };
    const auto fail = [&](std::string_view message) {
      std::cerr << "rj_concurrent_dispatch_fuzz: " << message << '\n';
      return 1;
    };
    const auto report = [&](CliStatus status) {
      return status == CliStatus::InvalidInput ? invalid(diagnostic.message())
                                               : fail(diagnostic.message());
    };
    std::string input_path;
    std::string output_path;
    std::string seed_path;
    std::optional<uint64_t> cases;
    std::optional<uint64_t> first_seed;
    bool afl = false;
    for (int index = 1; index < argc; ++index) {
      const std::string_view argument(argv[index]);
      if (argument == "--help") {
        usage(std::cout);
        return 0;
      } else if (argument == "--afl") {
        afl = true;
      } else if (argument == "--input" && index + 1 < argc) {
        if (parse_path(argv[++index], input_path, diagnostic.emitter()).failed()) {
          return invalid(diagnostic.message());
        }
      } else if (argument == "--output" && index + 1 < argc) {
        if (parse_path(argv[++index], output_path, diagnostic.emitter()).failed()) {
          return invalid(diagnostic.message());
        }
      } else if (argument == "--emit-seeds" && index + 1 < argc) {
        if (parse_path(argv[++index], seed_path, diagnostic.emitter()).failed()) {
          return invalid(diagnostic.message());
        }
      } else if (argument == "--random" && index + 1 < argc) {
        uint64_t value = 0;
        if (parse_uint64(argv[++index], value, diagnostic.emitter()).failed()) {
          return invalid(diagnostic.message());
        }
        cases = value;
      } else if (argument == "--seed" && index + 1 < argc) {
        uint64_t value = 0;
        if (parse_uint64(argv[++index], value, diagnostic.emitter()).failed()) {
          return invalid(diagnostic.message());
        }
        first_seed = value;
      } else {
        return invalid("unknown or incomplete argument");
      }
    }

    const unsigned modes = !input_path.empty() + cases.has_value() + !seed_path.empty() + afl;
    if (modes != 1) {
      return invalid("choose exactly one execution mode");
    }
    if (cases && *cases == 0) {
      return invalid("--random requires a positive count");
    }
    if (cases && output_path.empty()) {
      return invalid("random mode requires --output DIRECTORY");
    }
    if (!cases && (!output_path.empty() || first_seed)) {
      return invalid("--output and --seed require random mode");
    }
    const uint64_t seed_start = first_seed.value_or(0);
    if (cases && *cases - 1 > std::numeric_limits<uint64_t>::max() - seed_start) {
      return invalid("seed range exceeds uint64");
    }

    if (afl) {
      return run_afl();
    }
    if (!seed_path.empty()) {
      const CliStatus status = emit_seeds(seed_path, diagnostic.emitter());
      return status == CliStatus::Succeeded ? 0 : report(status);
    }
    if (!input_path.empty()) {
      std::vector<uint8_t> input;
      const CliStatus status = read_input(input_path, input, diagnostic.emitter());
      if (status != CliStatus::Succeeded) {
        return report(status);
      }
      const util::FailureOr<RunStats> stats = run_input(input, diagnostic.emitter());
      if (stats.failed()) {
        return fail(diagnostic.message());
      }
      print_stats(stats.value());
      return 0;
    }

    if (create_directory(output_path, diagnostic.emitter()).failed()) {
      return fail(diagnostic.message());
    }
    RunStats total;
    for (uint64_t index = 0; index < *cases; ++index) {
      const uint64_t seed = seed_start + index;
      const util::FailureOr<std::vector<uint8_t>> generated =
          random_input(seed, diagnostic.emitter());
      if (generated.failed()) {
        return fail(diagnostic.message());
      }
      const std::vector<uint8_t> &input = generated.value();
      const std::filesystem::path replay =
          std::filesystem::path(output_path) / std::format("failure-seed-{}.bin", seed);
      if (write_input(replay, input, diagnostic.emitter()).failed()) {
        return fail(diagnostic.message());
      }
      const util::FailureOr<RunStats> stats = run_input(input, diagnostic.emitter());
      if (stats.failed()) {
        return fail(diagnostic.message());
      }
      accumulate_stats(total, stats.value());
      std::error_code error;
      std::filesystem::remove(replay, error);
      if (error) {
        return fail("cannot remove artifact: " + replay.string() + ": " + error.message());
      }
    }
    print_stats(total);
    return 0;
  }
}
