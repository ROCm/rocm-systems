// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Interleaved fixed-repetition sampling, medians, and terminal progress.

#pragma once

#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace rocjitsu::arith::bench {

struct Options {
  double min_ms = 1;
  int repetitions = 5;
};

// One row owns both its metadata and its repetition latencies.
// sample performs one repetition, not one instruction call. An empty callback
// denotes an unsupported configuration and never participates in scheduling.
struct Benchmark {
  std::string category, instruction, semantics, cpu, non_finite_inputs;
  int wave = 32;
  std::vector<std::string> variants;
  int api_status = 0;
  std::function<double(double)> sample;
  std::vector<double> samples;
  double median = 0;
};

// Run a fixed number of interleaved repetitions and compute median latencies,
// optionally logging progress. Options require repetitions > 0.
void run(std::vector<Benchmark> &cases, const Options &options, FILE *progress = nullptr);

} // namespace rocjitsu::arith::bench
