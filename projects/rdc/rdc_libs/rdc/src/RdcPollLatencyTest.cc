/*
Copyright (c) 2026 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include "rdc_lib/impl/RdcPollLatencyTest.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <thread>
#include <vector>

#include "rdc_lib/rdc_common.h"

namespace amd {
namespace rdc {

namespace {

constexpr uint32_t kMaxSamples = 1000;
constexpr uint32_t kMaxIntervalMs = 60000;
// samples x interval_ms: a run takes at most about an hour.
constexpr uint64_t kMaxDurationMs = 3600000;

// Nearest-rank percentile of a sorted, non-empty list.
uint64_t percentile(const std::vector<uint64_t>& sorted, uint32_t percent) {
  size_t rank = (static_cast<size_t>(percent) * sorted.size() + 99) / 100;
  rank = std::min(std::max<size_t>(rank, 1), sorted.size());
  return sorted[rank - 1];
}

std::string ms(uint64_t us) {
  char buf[32];
  snprintf(buf, sizeof(buf), "%.2f ms", static_cast<double>(us) / 1000.0);
  return buf;
}

std::string count_of(uint64_t n, const char* noun) {
  return std::to_string(n) + " " + noun + (n == 1 ? "" : "s");
}

// Accepts decimal digits only, so "-1" and "1e3" are rejected.
bool parse_number(const std::string& text, uint32_t min, uint32_t max, uint32_t* out) {
  if (text.empty() || text.size() > 9) return false;
  for (char c : text) {
    if (c < '0' || c > '9') return false;
  }
  const unsigned long value = std::stoul(text);
  if (value < min || value > max) return false;
  *out = static_cast<uint32_t>(value);
  return true;
}

}  // namespace

PollClock real_poll_clock() {
  return {[] { return std::chrono::steady_clock::now(); },
          [](std::chrono::steady_clock::time_point t) { std::this_thread::sleep_until(t); }};
}

rdc_status_t parse_poll_latency_config(const char* config, size_t config_size,
                                       PollLatencyConfig* cfg, std::string* error) {
  if (cfg == nullptr || error == nullptr) return RDC_ST_BAD_PARAMETER;
  if (config == nullptr || config_size == 0) return RDC_ST_OK;

  std::string text(config, strnlen(config, config_size));
  std::replace(text.begin(), text.end(), ';', ' ');
  std::replace(text.begin(), text.end(), ',', ' ');
  std::istringstream settings(text);
  std::string setting;
  PollLatencyConfig parsed = *cfg;
  while (settings >> setting) {
    const size_t eq = setting.find('=');
    const std::string key = setting.substr(0, eq);
    const std::string value = (eq == std::string::npos) ? "" : setting.substr(eq + 1);
    if (key == "samples") {
      if (!parse_number(value, 1, kMaxSamples, &parsed.samples)) {
        *error = "Poll latency check: samples must be a number from 1 to " +
                 std::to_string(kMaxSamples) + ".";
        return RDC_ST_BAD_PARAMETER;
      }
    } else if (key == "interval_ms") {
      if (!parse_number(value, 0, kMaxIntervalMs, &parsed.interval_ms)) {
        *error = "Poll latency check: interval_ms must be a number from 0 to " +
                 std::to_string(kMaxIntervalMs) + ".";
        return RDC_ST_BAD_PARAMETER;
      }
    } else {
      *error =
          "Poll latency check: unknown setting \"" + setting + "\". Use samples=N;interval_ms=M.";
      return RDC_ST_BAD_PARAMETER;
    }
  }
  if (static_cast<uint64_t>(parsed.samples) * parsed.interval_ms > kMaxDurationMs) {
    *error = "Poll latency check: samples x interval_ms must be at most " +
             std::to_string(kMaxDurationMs) + " (one hour).";
    return RDC_ST_BAD_PARAMETER;
  }
  *cfg = parsed;
  return RDC_ST_OK;
}

rdc_status_t run_poll_latency_check(const uint32_t* gpu_index, uint32_t gpu_count,
                                    uint32_t field_count, const PollLatencyConfig& cfg,
                                    const PollReadFn& read, const PollClock& clock,
                                    rdc_diag_test_result_t* result) {
  if (result == nullptr || !read || !clock.now || !clock.wait_until ||
      (gpu_count > 0 && gpu_index == nullptr)) {
    return RDC_ST_BAD_PARAMETER;
  }

  result->test_case = RDC_DIAG_POLL_LATENCY;
  result->per_gpu_result_count = 0;
  result->details.code = 0;
  result->details.msg[0] = '\0';
  result->info[0] = '\0';

  if (gpu_count == 0) {
    result->status = RDC_DIAG_RESULT_FAIL;
    strncpy_with_null(result->info, "The group has no GPUs to poll.", MAX_DIAG_MSG_LENGTH);
    return RDC_ST_OK;
  }
  gpu_count = std::min<uint32_t>(gpu_count, RDC_MAX_NUM_DEVICES);
  const uint32_t samples = std::max<uint32_t>(cfg.samples, 1);

  // The first read of a GPU can include one-time setup, which a running rdcd
  // has already done, so it is not timed.
  for (uint32_t g = 0; g < gpu_count; g++) {
    uint64_t unused_us = 0;
    PollValueCounts unused;
    read(gpu_index[g], &unused_us, &unused);
  }

  std::vector<std::vector<uint64_t>> gpu_us(gpu_count);
  std::vector<PollValueCounts> gpu_counts(gpu_count);
  std::vector<uint64_t> poll_us;
  poll_us.reserve(samples);
  const auto interval = std::chrono::milliseconds(cfg.interval_ms);
  const auto start = clock.now();
  for (uint32_t s = 0; s < samples; s++) {
    if (s > 0) clock.wait_until(start + s * interval);
    uint64_t total_us = 0;
    for (uint32_t g = 0; g < gpu_count; g++) {
      uint64_t us = 0;
      PollValueCounts counts;
      read(gpu_index[g], &us, &counts);
      // A value the read did not return, because the read failed or skipped the
      // field, counts as failed.
      const uint64_t returned =
          static_cast<uint64_t>(counts.ok) + counts.not_supported + counts.failed;
      if (returned < field_count) counts.failed += field_count - static_cast<uint32_t>(returned);
      gpu_us[g].push_back(us);
      gpu_counts[g].ok += counts.ok;
      gpu_counts[g].not_supported += counts.not_supported;
      gpu_counts[g].failed += counts.failed;
      total_us += us;
    }
    poll_us.push_back(total_us);
  }

  PollValueCounts total;
  for (uint32_t g = 0; g < gpu_count; g++) {
    std::sort(gpu_us[g].begin(), gpu_us[g].end());
    const std::string msg = "median " + ms(percentile(gpu_us[g], 50)) + ", max " +
                            ms(gpu_us[g].back()) + " per read; " +
                            count_of(gpu_counts[g].ok, "value") + " read, " +
                            std::to_string(gpu_counts[g].failed) + " failed, " +
                            std::to_string(gpu_counts[g].not_supported) + " not supported";
    rdc_diag_per_gpu_result_t& per_gpu = result->gpu_results[g];
    per_gpu.gpu_index = gpu_index[g];
    per_gpu.gpu_result.code = gpu_counts[g].failed;
    strncpy_with_null(per_gpu.gpu_result.msg, msg.c_str(), MAX_DIAG_MSG_LENGTH);
    total.ok += gpu_counts[g].ok;
    total.not_supported += gpu_counts[g].not_supported;
    total.failed += gpu_counts[g].failed;
  }
  result->per_gpu_result_count = gpu_count;

  std::sort(poll_us.begin(), poll_us.end());
  const uint64_t max_us = poll_us.back();
  std::string info = "Read " + count_of(field_count, "field") + " from " +
                     count_of(gpu_count, "GPU") + " " + count_of(samples, "time") + ", " +
                     (cfg.interval_ms > 0 ? std::to_string(cfg.interval_ms) + " ms apart"
                                          : std::string("back to back")) +
                     ": median " + ms(percentile(poll_us, 50)) + ", p95 " +
                     ms(percentile(poll_us, 95)) + ", max " + ms(max_us) + " per poll. " +
                     count_of(total.ok, "value") + " read, " + std::to_string(total.failed) +
                     " failed, " + std::to_string(total.not_supported) + " not supported.";

  result->status = RDC_DIAG_RESULT_PASS;
  if (total.ok == 0) {
    result->status = RDC_DIAG_RESULT_FAIL;
    info += " No value could be read.";
  } else if (total.failed > 0) {
    result->status = RDC_DIAG_RESULT_WARN;
    info += " Some values could not be read.";
  }
  if (cfg.interval_ms > 0 && max_us > static_cast<uint64_t>(cfg.interval_ms) * 1000) {
    if (result->status == RDC_DIAG_RESULT_PASS) result->status = RDC_DIAG_RESULT_WARN;
    info += " The slowest poll took longer than the " + std::to_string(cfg.interval_ms) +
            " ms interval.";
  }
  strncpy_with_null(result->info, info.c_str(), MAX_DIAG_MSG_LENGTH);
  return RDC_ST_OK;
}

}  // namespace rdc
}  // namespace amd
