// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Interleaved fixed-repetition sampling, medians, and terminal progress.

#include "benchmark/sampling.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

namespace rocjitsu::arith::bench {

namespace {

// Carriage return and padding suffice: no terminal escape sequences.
class TerminalProgress {
public:
  // Bind progress to stream, enabling line overwrites only when it is a terminal.
  explicit TerminalProgress(FILE *stream) : stream_(stream) {
    if (stream_) {
#if defined(_WIN32)
      terminal_ = _isatty(_fileno(stream_));
#else
      terminal_ = isatty(fileno(stream_));
#endif
    }
  }

  // Terminate the progress line when terminal output was enabled.
  ~TerminalProgress() {
    if (terminal_) {
      std::fputc('\n', stream_);
      std::fflush(stream_);
    }
  }

  // Return whether progress can overwrite the last line of the supplied stream.
  bool enabled() const { return terminal_; }

  // Report completed work and elapsed/remaining time when the stream is a terminal.
  void update(double completed, double total) {
    if (!terminal_) {
      return;
    }
    auto now = Clock::now();
    bool finished = completed == total;
    if (completed > 0 && !finished && now - last_ < std::chrono::milliseconds(250)) {
      return;
    }
    last_ = now;
    double elapsed = std::chrono::duration<double>(now - start_).count();
    double percent = total > 0 ? 100 * completed / total : 100;
    char line[128];
    if (completed == 0 && total > 0) {
      std::snprintf(line, sizeof(line), "%5.1f%% | elapsed %.1fs | remaining --", percent, elapsed);
    } else {
      std::snprintf(line, sizeof(line), "%5.1f%% | elapsed %.1fs | remaining %.1fs", percent,
                    elapsed, completed > 0 ? elapsed * (total - completed) / completed : 0);
    }
    // Pad to the longest prior line so shrinking estimates leave no stale text.
    width_ = std::max(width_, int(std::strlen(line)));
    std::fprintf(stream_, "\r%-*s", width_, line);
    std::fflush(stream_);
  }

private:
  using Clock = std::chrono::steady_clock;
  FILE *stream_;
  bool terminal_ = false;
  int width_ = 0;
  Clock::time_point start_ = Clock::now(), last_ = start_;
};

} // namespace

// Replace stored samples with interleaved repetitions and compute their medians. Report progress
// when requested; reject nonpositive or nonfinite measurements.
void run(std::vector<Benchmark> &cases, const Options &options, FILE *progress) {
  int measured = 0;
  for (auto &b : cases) {
    b.samples.clear();
    b.median = 0;
    if (b.sample) {
      ++measured;
    }
  }
  TerminalProgress terminal(progress);
  double completed = 0;
  double total = double(measured) * options.repetitions;
  terminal.update(completed, total);
  if (progress && !terminal.enabled()) {
    std::fprintf(progress, "%d benchmarks; %d repetitions each\n", measured, options.repetitions);
  }
  // Interleave workloads so each one's repetitions are spread across the run.
  for (int repetition = 0; repetition < options.repetitions; ++repetition) {
    if (progress && !terminal.enabled()) {
      std::fprintf(progress, "  Repetition %d/%d\n", repetition + 1, options.repetitions);
      std::fflush(progress);
    }
    for (auto &b : cases) {
      if (!b.sample) {
        continue;
      }
      double latency = b.sample(options.min_ms);
      if (!std::isfinite(latency) || latency <= 0) {
        throw std::runtime_error("Invalid measured latency: " + b.instruction);
      }
      b.samples.push_back(latency);
      terminal.update(++completed, total);
    }
  }
  for (auto &b : cases) {
    if (b.samples.empty()) {
      continue;
    }
    auto sorted = b.samples;
    std::sort(sorted.begin(), sorted.end());
    // Use the middle sample, or the midpoint of the middle pair, without reordering the stored
    // samples.
    int lower = (int(sorted.size()) - 1) / 2;
    int upper = int(sorted.size()) / 2;
    b.median = sorted[lower] + (sorted[upper] - sorted[lower]) * 0.5;
  }
}

} // namespace rocjitsu::arith::bench
