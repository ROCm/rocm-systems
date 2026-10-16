// SPDX-License-Identifier: MIT

// Benchmark command-line parsing, workload ordering, and CSV output.

#include "benchmark/csv.h"
#include "benchmark/ordering.h"
#include "benchmark/workloads.h"

#include <algorithm>

#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <system_error>

namespace {

// Parse a strictly positive decimal integer; return false for invalid or trailing text.
bool parse_positive_integer(const char *text, int &value) {
  const char *end = text + std::strlen(text);
  auto parsed = std::from_chars(text, end, value);
  return parsed.ec == std::errc{} && parsed.ptr == end && value > 0;
}

// Parse a finite nonnegative duration; return false for invalid or trailing text.
bool parse_duration(const char *text, double &value) {
  char *end;
  errno = 0;
  value = std::strtod(text, &end);
  return *text && end != text && !*end && !errno && std::isfinite(value) && value >= 0;
}

// Parse benchmark option/value pairs; reject unknown, repeated, or incomplete options.
bool parse_options(int argc, char **argv, rocjitsu::arith::bench::Options &options) {
  std::set<std::string> seen;
  for (int arg = 1; arg < argc; arg += 2) {
    std::string name = argv[arg];
    if (arg + 1 == argc || !seen.insert(name).second) {
      return false;
    }
    const char *value = argv[arg + 1];
    if (name == "--min-ms") {
      if (!parse_duration(value, options.min_ms)) {
        return false;
      }
    } else if (name == "--repetitions") {
      if (!parse_positive_integer(value, options.repetitions)) {
        return false;
      }
    } else {
      return false;
    }
  }
  return true;
}

// Print successful measured rows as CSV, with speedups relative to matching scalar rows.
void print_csv(const std::vector<rocjitsu::arith::bench::Benchmark> &cases) {
  // Only variant descriptions can contain CSV delimiters; other metadata uses fixed tokens.
  std::puts(
      "Category,Instruction,Wave,Semantics,CPU path,Variant,Non-finite inputs,ns/wave,Speedup");
  std::map<std::string, double> scalar;
  for (const auto &b : cases) {
    if (b.cpu == "scalar" && !b.samples.empty()) {
      scalar[b.instruction + '/' + b.semantics + '/' + b.non_finite_inputs] = b.median;
    }
  }
  for (const auto &b : cases) {
    if (b.samples.empty()) {
      continue;
    }
    std::printf("%s,%s,%d,%s,%s,%s,%s,", b.category.c_str(), b.instruction.c_str(), b.wave,
                b.semantics.c_str(), b.cpu.c_str(),
                rocjitsu::arith::bench::variant_field(b.variants).c_str(),
                b.non_finite_inputs.c_str());
    std::printf("%.1f,", b.median);
    auto baseline = scalar.find(b.instruction + '/' + b.semantics + '/' + b.non_finite_inputs);
    if (b.cpu != "scalar" && baseline != scalar.end()) {
      std::printf("%.3f", baseline->second / b.median);
    }
    std::putchar('\n');
  }
}

} // namespace

// Run the requested benchmark repetitions and emit CSV; report invalid options or failures on
// stderr.
int main(int argc, char **argv) {
  rocjitsu::arith::bench::Options options;
  if (!parse_options(argc, argv, options)) {
    std::fprintf(stderr,
                 "Usage: %s [--min-ms N] [--repetitions N]\n"
                 "Defaults: 1 ms per repetition, 5 repetitions.\n",
                 argv[0]);
    return 2;
  }
  try {
    std::vector<rocjitsu::arith::bench::Benchmark> cases;
    rocjitsu::arith::bench::add_wmma(cases);
    rocjitsu::arith::bench::add_dot(cases);
    rocjitsu::arith::bench::add_fp(cases);
    rocjitsu::arith::bench::add_math(cases);
    std::stable_sort(cases.begin(), cases.end(), rocjitsu::arith::bench::benchmark_less);
    rocjitsu::arith::bench::run(cases, options, stderr);
    print_csv(cases);
  } catch (const std::exception &error) {
    std::fprintf(stderr, "Benchmark failed: %s\n", error.what());
    return 1;
  }
}
