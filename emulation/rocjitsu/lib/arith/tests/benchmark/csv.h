// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// CSV escaping and assembly of workload variant labels.

#pragma once

#include "benchmark/sampling.h"

#include <cstdio>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace rocjitsu::arith::test {

// Join variant descriptions into one CSV field, quoting embedded CSV delimiters.
inline std::string variant_field(const std::vector<std::string> &variants) {
  std::string value;
  for (int i = 0; i < int(variants.size()); ++i) {
    if (i) {
      value += ", ";
    }
    value += variants[i];
  }
  if (value.find_first_of(",\"\r\n") == std::string::npos) {
    return value;
  }
  std::string quoted = "\"";
  for (char c : value) {
    if (c == '"') {
      quoted += '"';
    }
    quoted += c;
  }
  return quoted + '"';
}

// Print successful measured rows as CSV, with speedups relative to matching scalar rows.
inline void print_csv(FILE *output, const std::vector<Benchmark> &cases) {
  // Only variant descriptions can contain CSV delimiters; other metadata uses fixed tokens.
  std::fprintf(output, "Category,Instruction,Wave,Semantics,Requested CPU level,Variant,Non-finite "
                       "inputs,ns/wave,Speedup\n");
  using Key = std::tuple<std::string, std::string, std::string, std::vector<std::string>>;
  const auto key = [](const Benchmark &b) {
    return Key{b.instruction, b.semantics, b.non_finite_inputs, b.variants};
  };
  std::map<Key, double> scalar;
  for (const auto &b : cases) {
    if (b.cpu == "scalar" && !b.samples.empty()) {
      scalar[key(b)] = b.median;
    }
  }
  for (const auto &b : cases) {
    if (b.samples.empty()) {
      continue;
    }
    std::fprintf(output, "%s,%s,%d,%s,%s,%s,%s,", b.category.c_str(), b.instruction.c_str(), b.wave,
                 b.semantics.c_str(), b.cpu.c_str(),
                 rocjitsu::arith::test::variant_field(b.variants).c_str(),
                 b.non_finite_inputs.c_str());
    std::fprintf(output, "%.1f,", b.median);
    auto baseline = scalar.find(key(b));
    if (b.cpu != "scalar" && baseline != scalar.end()) {
      std::fprintf(output, "%.3f", baseline->second / b.median);
    }
    std::fputc('\n', output);
  }
}

} // namespace rocjitsu::arith::test
