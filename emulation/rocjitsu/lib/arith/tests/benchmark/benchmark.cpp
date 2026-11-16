// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Benchmark command-line parsing, workload ordering, and CSV output.

#include "benchmark/csv.h"
#include "benchmark/ordering.h"
#include "benchmark/workloads.h"

#include "rocjitsu/cli.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string_view>
#include <vector>

int main(int argc, char **argv) {
  namespace bench = rocjitsu::arith::test;
  namespace cli = rocjitsu::cli;
  cli::Command command(argv[0]);
  command.set_positional_usage("");
  auto duration = std::make_shared<cli::StringFlag>(
      cli::FlagInfo{.name = "--min-ms",
                    .aliases = {},
                    .description = "Minimum milliseconds per repetition (default 1)"},
      "1");
  auto repetitions = std::make_shared<cli::Int64Flag>(
      cli::FlagInfo{.name = "--repetitions",
                    .aliases = {},
                    .description = "Repetitions per workload (default 5)"},
      5);
  auto help = std::make_shared<cli::BoolFlag>(cli::FlagInfo{.name = "--help",
                                                            .aliases = {},
                                                            .description = "Show usage",
                                                            .value_mode = cli::ValueMode::Switch});
  util::StringDiagnostic diagnostic;
  if (command.add_flags("Benchmark options", {duration, repetitions, help}, diagnostic.emitter())
          .failed()) {
    std::fprintf(stderr, "%s\n", diagnostic.message().c_str());
    return 2;
  }
  std::vector<std::string_view> arguments(argv + 1, argv + argc);
  auto parsed = cli::parse(command, arguments, diagnostic.emitter());
  if (parsed.failed() || !parsed.value().positional.empty()) {
    std::fprintf(stderr, "%s\n%s", diagnostic.message().c_str(), cli::format_help(command).c_str());
    return 2;
  }
  if (help->value()) {
    std::fputs(cli::format_help(command).c_str(), stdout);
    return 0;
  }
  bench::Options options;
  const char *text = duration->value().c_str();
  char *end;
  errno = 0;
  options.min_ms = std::strtod(text, &end);
  if (!*text || end == text || *end || errno || !std::isfinite(options.min_ms) ||
      options.min_ms < 0 || repetitions->value() <= 0 ||
      repetitions->value() > std::numeric_limits<int>::max()) {
    std::fprintf(stderr, "Expected finite nonnegative --min-ms and positive int --repetitions\n");
    return 2;
  }
  options.repetitions = int(repetitions->value());
  std::vector<bench::Benchmark> cases;
  bench::add_wmma(cases);
  bench::add_dot(cases);
  bench::add_fp(cases);
  bench::add_math(cases);
  std::stable_sort(cases.begin(), cases.end(), bench::benchmark_less);
  if (bench::run(cases, options, stderr).failed()) {
    return 1;
  }
  bench::print_csv(stdout, cases);
  if (std::fflush(stdout) != 0 || std::ferror(stdout)) {
    std::fprintf(stderr, "Failed to write benchmark CSV\n");
    return 1;
  }
  return 0;
}
