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

// The poll latency check of rdci diag, run with a fake GPU read and a fake
// clock: its timing statistics, PASS/WARN/FAIL rules, messages and settings.
// RdcPollLatencyTest.cc is compiled into this binary, so it needs no GPU.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "rdc/rdc.h"
#include "rdc_lib/impl/RdcPollLatencyTest.h"

namespace {

using amd::rdc::parse_poll_latency_config;
using amd::rdc::PollClock;
using amd::rdc::PollLatencyConfig;
using amd::rdc::PollReadFn;
using amd::rdc::PollValueCounts;
using amd::rdc::run_poll_latency_check;
using Clock = std::chrono::steady_clock;

constexpr uint32_t kFields = 9;

// Records what the check does, in order, without sleeping.
struct FakeClock {
  Clock::time_point start = Clock::time_point(std::chrono::seconds(1000));
  std::vector<Clock::time_point> waits;
  std::vector<std::string>* events = nullptr;

  PollClock clock() {
    return {[this] {
              if (events != nullptr) events->push_back("now");
              return start;
            },
            [this](Clock::time_point t) {
              if (events != nullptr) events->push_back("wait");
              waits.push_back(t);
            }};
  }
};

// A read that takes us_per_read[gpu] and returns every field.
PollReadFn read_all_ok(std::vector<uint64_t> us_per_read) {
  return [us_per_read](uint32_t gpu, uint64_t* elapsed_us, PollValueCounts* counts) {
    *elapsed_us = us_per_read.at(gpu);
    counts->ok += kFields;
    return RDC_ST_OK;
  };
}

std::unique_ptr<rdc_diag_test_result_t> new_result() {
  return std::make_unique<rdc_diag_test_result_t>();
}

PollLatencyConfig make_cfg(uint32_t samples, uint32_t interval_ms) {
  PollLatencyConfig cfg;
  cfg.samples = samples;
  cfg.interval_ms = interval_ms;
  return cfg;
}

// --- Settings ---------------------------------------------------------------

TEST(PollLatencyConfigTest, EmptyConfigKeepsTheDefaults) {
  PollLatencyConfig cfg;
  std::string error;
  EXPECT_EQ(parse_poll_latency_config(nullptr, 0, &cfg, &error), RDC_ST_OK);
  EXPECT_EQ(parse_poll_latency_config("", 0, &cfg, &error), RDC_ST_OK);
  EXPECT_EQ(parse_poll_latency_config("  ", 2, &cfg, &error), RDC_ST_OK);
  EXPECT_EQ(cfg.samples, 20u);
  EXPECT_EQ(cfg.interval_ms, 100u);
  EXPECT_TRUE(error.empty());
}

TEST(PollLatencyConfigTest, ReadsSamplesAndInterval) {
  const struct {
    const char* config;
    uint32_t samples;
    uint32_t interval_ms;
  } cases[] = {
      {"samples=5;interval_ms=0", 5, 0},
      {"interval_ms=250,samples=3", 3, 250},
      {" samples=1000 interval_ms=3600 ", 1000, 3600},
      {"samples=60;interval_ms=60000", 60, 60000},
      {"samples=1", 1, 100},
      {"interval_ms=007", 20, 7},
  };
  for (const auto& c : cases) {
    PollLatencyConfig cfg;
    std::string error;
    EXPECT_EQ(parse_poll_latency_config(c.config, strlen(c.config), &cfg, &error), RDC_ST_OK)
        << c.config << ": " << error;
    EXPECT_EQ(cfg.samples, c.samples) << c.config;
    EXPECT_EQ(cfg.interval_ms, c.interval_ms) << c.config;
  }
}

TEST(PollLatencyConfigTest, RejectsBadSettingsAndKeepsTheConfig) {
  const char* bad[] = {"samples=0",
                       "samples=1001",
                       "samples=-1",
                       "samples=",
                       "samples",
                       "samples=5x",
                       "samples=1e3",
                       "interval_ms=60001",
                       "interval_ms=-5",
                       "foo=1",
                       "samples=5;bar",
                       "=5",
                       "samples=99999999999",
                       "samples=1000;interval_ms=3601",
                       "samples=61,interval_ms=60000"};
  for (const char* config : bad) {
    PollLatencyConfig cfg = make_cfg(7, 9);
    std::string error;
    EXPECT_EQ(parse_poll_latency_config(config, strlen(config), &cfg, &error), RDC_ST_BAD_PARAMETER)
        << config;
    EXPECT_FALSE(error.empty()) << config;
    EXPECT_EQ(cfg.samples, 7u) << config;
    EXPECT_EQ(cfg.interval_ms, 9u) << config;
  }
}

TEST(PollLatencyConfigTest, ReadsOnlyConfigSizeBytes) {
  // Not NUL-terminated: the parser must stop at config_size.
  const char config[] = {'s', 'a', 'm', 'p', 'l', 'e', 's', '=', '5'};
  PollLatencyConfig cfg;
  std::string error;
  EXPECT_EQ(parse_poll_latency_config(config, sizeof(config), &cfg, &error), RDC_ST_OK);
  EXPECT_EQ(cfg.samples, 5u);

  const char* longer = "samples=6;interval_ms=1";
  cfg = PollLatencyConfig();
  EXPECT_EQ(parse_poll_latency_config(longer, 9, &cfg, &error), RDC_ST_OK);
  EXPECT_EQ(cfg.samples, 6u);
  EXPECT_EQ(cfg.interval_ms, 100u);
}

TEST(PollLatencyConfigTest, RejectsNullOutputs) {
  PollLatencyConfig cfg;
  std::string error;
  EXPECT_EQ(parse_poll_latency_config("samples=5", 9, nullptr, &error), RDC_ST_BAD_PARAMETER);
  EXPECT_EQ(parse_poll_latency_config("samples=5", 9, &cfg, nullptr), RDC_ST_BAD_PARAMETER);
}

// --- The check ----------------------------------------------------------------

TEST(PollLatencyCheckTest, PassReportsPollAndReadTimes) {
  const uint32_t gpus[] = {0, 1};
  FakeClock fake;
  auto result = new_result();
  ASSERT_EQ(run_poll_latency_check(gpus, 2, kFields, make_cfg(4, 100), read_all_ok({1000, 2000}),
                                   fake.clock(), result.get()),
            RDC_ST_OK);

  EXPECT_EQ(result->status, RDC_DIAG_RESULT_PASS);
  EXPECT_EQ(result->test_case, RDC_DIAG_POLL_LATENCY);
  EXPECT_STREQ(result->info,
               "Read 9 fields from 2 GPUs 4 times, 100 ms apart: median 3.00 ms, p95 3.00 ms, "
               "max 3.00 ms per poll. 72 values read, 0 failed, 0 not supported.");
  ASSERT_EQ(result->per_gpu_result_count, 2u);
  EXPECT_EQ(result->gpu_results[0].gpu_index, 0u);
  EXPECT_STREQ(result->gpu_results[0].gpu_result.msg,
               "median 1.00 ms, max 1.00 ms per read; 36 values read, 0 failed, 0 not supported");
  EXPECT_EQ(result->gpu_results[0].gpu_result.code, 0u);
  EXPECT_EQ(result->gpu_results[1].gpu_index, 1u);
  EXPECT_STREQ(result->gpu_results[1].gpu_result.msg,
               "median 2.00 ms, max 2.00 ms per read; 36 values read, 0 failed, 0 not supported");
}

TEST(PollLatencyCheckTest, ReportsGpuIndexesOfTheGroup) {
  const uint32_t gpus[] = {5, 3};
  FakeClock fake;
  auto result = new_result();
  std::vector<uint32_t> read_gpus;
  PollReadFn read = [&](uint32_t gpu, uint64_t* elapsed_us, PollValueCounts* counts) {
    read_gpus.push_back(gpu);
    *elapsed_us = 100;
    counts->ok += kFields;
    return RDC_ST_OK;
  };
  ASSERT_EQ(
      run_poll_latency_check(gpus, 2, kFields, make_cfg(1, 100), read, fake.clock(), result.get()),
      RDC_ST_OK);
  EXPECT_EQ(read_gpus, (std::vector<uint32_t>{5, 3, 5, 3}));
  ASSERT_EQ(result->per_gpu_result_count, 2u);
  EXPECT_EQ(result->gpu_results[0].gpu_index, 5u);
  EXPECT_EQ(result->gpu_results[1].gpu_index, 3u);
}

TEST(PollLatencyCheckTest, MedianP95AndMaxAreNearestRank) {
  // Poll s takes s + 1 ms. The untimed first read takes 999 ms and must not count.
  const uint32_t gpus[] = {0};
  FakeClock fake;
  auto result = new_result();
  uint32_t calls = 0;
  PollReadFn read = [&](uint32_t, uint64_t* elapsed_us, PollValueCounts* counts) {
    *elapsed_us = (calls == 0) ? 999000 : calls * 1000;
    calls++;
    counts->ok += kFields;
    return RDC_ST_OK;
  };
  ASSERT_EQ(
      run_poll_latency_check(gpus, 1, kFields, make_cfg(20, 100), read, fake.clock(), result.get()),
      RDC_ST_OK);
  EXPECT_EQ(calls, 21u);
  EXPECT_EQ(result->status, RDC_DIAG_RESULT_PASS);
  EXPECT_NE(strstr(result->info, "median 10.00 ms, p95 19.00 ms, max 20.00 ms per poll."), nullptr)
      << result->info;
  EXPECT_STREQ(
      result->gpu_results[0].gpu_result.msg,
      "median 10.00 ms, max 20.00 ms per read; 180 values read, 0 failed, 0 not supported");
}

TEST(PollLatencyCheckTest, ReadsEachGpuUntimedFirstThenPollsAtAFixedRate) {
  const uint32_t gpus[] = {0, 1};
  std::vector<std::string> events;
  FakeClock fake;
  fake.events = &events;
  auto result = new_result();
  PollReadFn read = [&](uint32_t gpu, uint64_t* elapsed_us, PollValueCounts* counts) {
    events.push_back("read " + std::to_string(gpu));
    *elapsed_us = 500;
    counts->ok += kFields;
    return RDC_ST_OK;
  };
  ASSERT_EQ(
      run_poll_latency_check(gpus, 2, kFields, make_cfg(3, 250), read, fake.clock(), result.get()),
      RDC_ST_OK);
  EXPECT_EQ(events, (std::vector<std::string>{"read 0", "read 1", "now", "read 0", "read 1", "wait",
                                              "read 0", "read 1", "wait", "read 0", "read 1"}));
  // Polls start at fixed times, so a slow poll does not shift the next ones.
  EXPECT_EQ(fake.waits,
            (std::vector<Clock::time_point>{fake.start + std::chrono::milliseconds(250),
                                            fake.start + std::chrono::milliseconds(500)}));
}

TEST(PollLatencyCheckTest, BackToBackWhenTheIntervalIsZero) {
  const uint32_t gpus[] = {0};
  FakeClock fake;
  auto result = new_result();
  ASSERT_EQ(run_poll_latency_check(gpus, 1, kFields, make_cfg(3, 0), read_all_ok({500000}),
                                   fake.clock(), result.get()),
            RDC_ST_OK);
  // A 500 ms poll is no warning without an interval to keep.
  EXPECT_EQ(result->status, RDC_DIAG_RESULT_PASS);
  EXPECT_NE(strstr(result->info, "3 times, back to back: median 500.00 ms"), nullptr)
      << result->info;
}

TEST(PollLatencyCheckTest, WarnsWhenAPollIsSlowerThanTheInterval) {
  const uint32_t gpus[] = {0, 1};
  FakeClock fake;
  auto result = new_result();
  // Each read is within 10 ms, but a poll of both GPUs takes 12 ms.
  ASSERT_EQ(run_poll_latency_check(gpus, 2, kFields, make_cfg(5, 10), read_all_ok({6000, 6000}),
                                   fake.clock(), result.get()),
            RDC_ST_OK);
  EXPECT_EQ(result->status, RDC_DIAG_RESULT_WARN);
  EXPECT_STREQ(result->info,
               "Read 9 fields from 2 GPUs 5 times, 10 ms apart: median 12.00 ms, p95 12.00 ms, "
               "max 12.00 ms per poll. 90 values read, 0 failed, 0 not supported. The slowest "
               "poll took longer than the 10 ms interval.");
}

TEST(PollLatencyCheckTest, APollAsLongAsTheIntervalPasses) {
  const uint32_t gpus[] = {0};
  FakeClock fake;
  auto result = new_result();
  ASSERT_EQ(run_poll_latency_check(gpus, 1, kFields, make_cfg(5, 10), read_all_ok({10000}),
                                   fake.clock(), result.get()),
            RDC_ST_OK);
  EXPECT_EQ(result->status, RDC_DIAG_RESULT_PASS) << result->info;
}

TEST(PollLatencyCheckTest, WarnsWhenValuesCannotBeRead) {
  const uint32_t gpus[] = {0, 1};
  FakeClock fake;
  auto result = new_result();
  uint32_t calls = 0;
  PollReadFn read = [&](uint32_t gpu, uint64_t* elapsed_us, PollValueCounts* counts) {
    *elapsed_us = 1000;
    calls++;
    // GPU 1 fails one value in one timed poll.
    if (gpu == 1 && calls == 6) {
      counts->ok += kFields - 1;
      counts->failed += 1;
    } else {
      counts->ok += kFields;
    }
    return RDC_ST_OK;
  };
  ASSERT_EQ(
      run_poll_latency_check(gpus, 2, kFields, make_cfg(3, 100), read, fake.clock(), result.get()),
      RDC_ST_OK);
  EXPECT_EQ(result->status, RDC_DIAG_RESULT_WARN);
  EXPECT_NE(strstr(result->info,
                   "53 values read, 1 failed, 0 not supported. Some values could "
                   "not be read."),
            nullptr)
      << result->info;
  EXPECT_EQ(result->gpu_results[0].gpu_result.code, 0u);
  EXPECT_EQ(result->gpu_results[1].gpu_result.code, 1u);
  EXPECT_NE(strstr(result->gpu_results[1].gpu_result.msg, "26 values read, 1 failed"), nullptr)
      << result->gpu_results[1].gpu_result.msg;
}

TEST(PollLatencyCheckTest, FailedValuesInTheUntimedReadDoNotCount) {
  const uint32_t gpus[] = {0};
  FakeClock fake;
  auto result = new_result();
  uint32_t calls = 0;
  PollReadFn read = [&](uint32_t, uint64_t* elapsed_us, PollValueCounts* counts) {
    *elapsed_us = 1000;
    if (calls++ == 0) {
      counts->failed += kFields;
      return RDC_ST_SMI_ERROR;
    }
    counts->ok += kFields;
    return RDC_ST_OK;
  };
  ASSERT_EQ(
      run_poll_latency_check(gpus, 1, kFields, make_cfg(2, 100), read, fake.clock(), result.get()),
      RDC_ST_OK);
  EXPECT_EQ(result->status, RDC_DIAG_RESULT_PASS) << result->info;
}

TEST(PollLatencyCheckTest, UnsupportedValuesDoNotWarn) {
  const uint32_t gpus[] = {0};
  FakeClock fake;
  auto result = new_result();
  PollReadFn read = [](uint32_t, uint64_t* elapsed_us, PollValueCounts* counts) {
    *elapsed_us = 1000;
    counts->ok += kFields - 2;
    counts->not_supported += 2;
    return RDC_ST_OK;
  };
  ASSERT_EQ(
      run_poll_latency_check(gpus, 1, kFields, make_cfg(4, 100), read, fake.clock(), result.get()),
      RDC_ST_OK);
  EXPECT_EQ(result->status, RDC_DIAG_RESULT_PASS);
  EXPECT_NE(strstr(result->info, "28 values read, 0 failed, 8 not supported."), nullptr)
      << result->info;
}

TEST(PollLatencyCheckTest, FailsWhenNoValueCanBeRead) {
  const uint32_t gpus[] = {0, 1};
  FakeClock fake;
  auto result = new_result();
  // The read fails before it returns any value.
  PollReadFn read = [](uint32_t, uint64_t* elapsed_us, PollValueCounts*) {
    *elapsed_us = 50;
    return RDC_ST_SMI_ERROR;
  };
  ASSERT_EQ(
      run_poll_latency_check(gpus, 2, kFields, make_cfg(2, 100), read, fake.clock(), result.get()),
      RDC_ST_OK);
  EXPECT_EQ(result->status, RDC_DIAG_RESULT_FAIL);
  EXPECT_NE(strstr(result->info,
                   "0 values read, 36 failed, 0 not supported. No value could be "
                   "read."),
            nullptr)
      << result->info;
  EXPECT_EQ(result->gpu_results[1].gpu_result.code, 18u);
}

TEST(PollLatencyCheckTest, AFailedReadCountsTheValuesItDidNotReturnAsFailed) {
  const uint32_t gpus[] = {0};
  FakeClock fake;
  auto result = new_result();
  PollReadFn read = [](uint32_t, uint64_t* elapsed_us, PollValueCounts* counts) {
    *elapsed_us = 50;
    counts->ok += 3;
    counts->not_supported += 1;
    return RDC_ST_SMI_ERROR;
  };
  ASSERT_EQ(
      run_poll_latency_check(gpus, 1, kFields, make_cfg(2, 100), read, fake.clock(), result.get()),
      RDC_ST_OK);
  EXPECT_EQ(result->status, RDC_DIAG_RESULT_WARN);
  EXPECT_NE(strstr(result->info, "6 values read, 10 failed, 2 not supported."), nullptr)
      << result->info;
}

TEST(PollLatencyCheckTest, ValuesASuccessfulReadSkipsCountAsFailed) {
  const uint32_t gpus[] = {0};
  FakeClock fake;
  auto result = new_result();
  PollReadFn read = [](uint32_t, uint64_t* elapsed_us, PollValueCounts* counts) {
    *elapsed_us = 50;
    counts->ok += kFields - 2;
    return RDC_ST_OK;
  };
  ASSERT_EQ(
      run_poll_latency_check(gpus, 1, kFields, make_cfg(2, 100), read, fake.clock(), result.get()),
      RDC_ST_OK);
  EXPECT_EQ(result->status, RDC_DIAG_RESULT_WARN);
  EXPECT_NE(strstr(result->info, "14 values read, 4 failed, 0 not supported."), nullptr)
      << result->info;
}

TEST(PollLatencyCheckTest, FailsForAGroupWithoutGpus) {
  FakeClock fake;
  auto result = new_result();
  bool read_called = false;
  PollReadFn read = [&](uint32_t, uint64_t*, PollValueCounts*) {
    read_called = true;
    return RDC_ST_OK;
  };
  ASSERT_EQ(run_poll_latency_check(nullptr, 0, kFields, make_cfg(2, 100), read, fake.clock(),
                                   result.get()),
            RDC_ST_OK);
  EXPECT_EQ(result->status, RDC_DIAG_RESULT_FAIL);
  EXPECT_EQ(result->test_case, RDC_DIAG_POLL_LATENCY);
  EXPECT_STREQ(result->info, "The group has no GPUs to poll.");
  EXPECT_EQ(result->per_gpu_result_count, 0u);
  EXPECT_FALSE(read_called);
}

TEST(PollLatencyCheckTest, ReportsAtMostMaxDevicesGpus) {
  std::vector<uint32_t> gpus(RDC_MAX_NUM_DEVICES + 10);
  for (uint32_t i = 0; i < gpus.size(); i++) gpus[i] = i;
  FakeClock fake;
  auto result = new_result();
  uint32_t highest = 0;
  PollReadFn read = [&](uint32_t gpu, uint64_t* elapsed_us, PollValueCounts* counts) {
    highest = std::max(highest, gpu);
    *elapsed_us = 10;
    counts->ok += kFields;
    return RDC_ST_OK;
  };
  ASSERT_EQ(run_poll_latency_check(gpus.data(), static_cast<uint32_t>(gpus.size()), kFields,
                                   make_cfg(1, 100), read, fake.clock(), result.get()),
            RDC_ST_OK);
  EXPECT_EQ(result->per_gpu_result_count, static_cast<uint32_t>(RDC_MAX_NUM_DEVICES));
  EXPECT_EQ(highest, RDC_MAX_NUM_DEVICES - 1u);
  EXPECT_NE(strstr(result->info, "from 128 GPUs"), nullptr) << result->info;
}

TEST(PollLatencyCheckTest, OverwritesAnUninitializedResult) {
  const uint32_t gpus[] = {0};
  FakeClock fake;
  auto result = new_result();
  memset(result.get(), 0xAB, sizeof(*result));
  ASSERT_EQ(run_poll_latency_check(gpus, 1, kFields, make_cfg(1, 100), read_all_ok({100}),
                                   fake.clock(), result.get()),
            RDC_ST_OK);
  EXPECT_EQ(result->status, RDC_DIAG_RESULT_PASS);
  EXPECT_EQ(result->test_case, RDC_DIAG_POLL_LATENCY);
  EXPECT_EQ(result->per_gpu_result_count, 1u);
  EXPECT_EQ(result->details.code, 0u);
  EXPECT_STREQ(result->details.msg, "");
  EXPECT_LT(strnlen(result->info, MAX_DIAG_MSG_LENGTH), static_cast<size_t>(MAX_DIAG_MSG_LENGTH));
  EXPECT_LT(strnlen(result->gpu_results[0].gpu_result.msg, MAX_DIAG_MSG_LENGTH),
            static_cast<size_t>(MAX_DIAG_MSG_LENGTH));
}

TEST(PollLatencyCheckTest, UsesSingularNouns) {
  const uint32_t gpus[] = {0};
  FakeClock fake;
  auto result = new_result();
  PollReadFn read = [](uint32_t, uint64_t* elapsed_us, PollValueCounts* counts) {
    *elapsed_us = 1234;
    counts->ok += 1;
    return RDC_ST_OK;
  };
  ASSERT_EQ(run_poll_latency_check(gpus, 1, 1, make_cfg(1, 100), read, fake.clock(), result.get()),
            RDC_ST_OK);
  EXPECT_STREQ(result->info,
               "Read 1 field from 1 GPU 1 time, 100 ms apart: median 1.23 ms, p95 1.23 ms, "
               "max 1.23 ms per poll. 1 value read, 0 failed, 0 not supported.");
}

TEST(PollLatencyCheckTest, RejectsMissingArguments) {
  const uint32_t gpus[] = {0};
  FakeClock fake;
  auto result = new_result();
  const PollReadFn read = read_all_ok({100});
  EXPECT_EQ(run_poll_latency_check(gpus, 1, kFields, make_cfg(1, 100), read, fake.clock(), nullptr),
            RDC_ST_BAD_PARAMETER);
  EXPECT_EQ(run_poll_latency_check(nullptr, 1, kFields, make_cfg(1, 100), read, fake.clock(),
                                   result.get()),
            RDC_ST_BAD_PARAMETER);
  EXPECT_EQ(run_poll_latency_check(gpus, 1, kFields, make_cfg(1, 100), PollReadFn(), fake.clock(),
                                   result.get()),
            RDC_ST_BAD_PARAMETER);
  EXPECT_EQ(
      run_poll_latency_check(gpus, 1, kFields, make_cfg(1, 100), read, PollClock(), result.get()),
      RDC_ST_BAD_PARAMETER);
}

}  // namespace
