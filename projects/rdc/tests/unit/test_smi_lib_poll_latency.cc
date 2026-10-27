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

// The poll latency check as RdcSmiLib runs it: which fields it reads, how it
// counts the values, the CPUs it skips, and that it never reads fields while
// rdcd's collector reads them. A fake metric fetcher stands in for amd-smi, so
// the tests need no GPU; amd_smi is linked only for calls they never make.

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "rdc/rdc.h"
#include "rdc_lib/RdcMetricFetcher.h"
#include "rdc_lib/impl/RdcSmiLib.h"

namespace {

using amd::rdc::RdcSmiLib;

const std::vector<rdc_field_t> kExpectedFields = {
    RDC_FI_GPU_CLOCK,         RDC_FI_MEMORY_TEMP,       RDC_FI_GPU_TEMP,
    RDC_FI_GPU_JUNCTION_TEMP, RDC_FI_POWER_USAGE,       RDC_FI_GPU_UTIL,
    RDC_FI_GPU_MEMORY_USAGE,  RDC_FI_ECC_CORRECT_TOTAL, RDC_FI_ECC_UNCORRECT_TOTAL};

class FakeFetcher : public amd::rdc::RdcMetricFetcher {
 public:
  // What fetch_smi_field() does for a field: fill in a value with `status`,
  // or return `ret` without touching the value, like its early returns do.
  struct Behavior {
    bool fill = true;
    int status = AMDSMI_STATUS_SUCCESS;
    rdc_status_t ret = RDC_ST_OK;
  };

  std::map<rdc_field_t, Behavior> behavior;
  std::chrono::microseconds delay{0};
  std::atomic<bool> overlapped{false};

  std::vector<std::pair<uint32_t, rdc_field_t>> calls() {
    std::lock_guard<std::mutex> guard(mutex_);
    return calls_;
  }

  rdc_status_t acquire_smi_handle(RdcFieldKey) override { return RDC_ST_OK; }
  rdc_status_t delete_smi_handle(RdcFieldKey) override { return RDC_ST_OK; }
  rdc_status_t bulk_fetch_smi_fields(rdc_gpu_field_t*, uint32_t,
                                     std::vector<rdc_gpu_field_value_t>&) override {
    return RDC_ST_NOT_SUPPORTED;
  }

  rdc_status_t fetch_smi_field(uint32_t gpu_index, rdc_field_t field_id,
                               rdc_field_value* value) override {
    if (in_flight_.fetch_add(1) != 0) overlapped = true;
    if (delay.count() > 0) std::this_thread::sleep_for(delay);
    {
      std::lock_guard<std::mutex> guard(mutex_);
      calls_.emplace_back(gpu_index, field_id);
    }
    Behavior b;
    auto it = behavior.find(field_id);
    if (it != behavior.end()) b = it->second;
    if (b.fill) {
      value->ts = 1;
      value->field_id = field_id;
      value->status = b.status;
      value->type = INTEGER;
      value->value.l_int = 42;
    }
    in_flight_.fetch_sub(1);
    return b.ret;
  }

 private:
  std::atomic<int> in_flight_{0};
  std::mutex mutex_;
  std::vector<std::pair<uint32_t, rdc_field_t>> calls_;
};

struct Lib {
  std::shared_ptr<FakeFetcher> fetcher = std::make_shared<FakeFetcher>();
  RdcSmiLib lib{fetcher};
  std::unique_ptr<rdc_diag_test_result_t> result = std::make_unique<rdc_diag_test_result_t>();

  rdc_status_t run_on(std::vector<uint32_t> gpus, const char* config) {
    const uint32_t count = static_cast<uint32_t>(gpus.size());
    gpus.resize(RDC_MAX_NUM_DEVICES);  // the API takes an array of RDC_MAX_NUM_DEVICES
    return lib.rdc_test_case_run(RDC_DIAG_POLL_LATENCY, gpus.data(), count, config, strlen(config),
                                 result.get(), nullptr);
  }
};

uint32_t cpu_entity(uint32_t index) {
  rdc_entity_info_t info = {};
  info.device_index = index;
  info.instance_index = 0;
  info.entity_role = RDC_DEVICE_ROLE_PHYSICAL;
  info.device_type = RDC_DEVICE_TYPE_CPU;
  return rdc_get_entity_index_from_info(info);
}

TEST(SmiLibPollLatencyTest, IsListedAsATestCase) {
  Lib l;
  rdc_diag_test_cases_t cases[MAX_TEST_CASES];
  uint32_t count = 0;
  ASSERT_EQ(l.lib.rdc_diag_test_cases_query(cases, &count), RDC_ST_OK);
  EXPECT_NE(std::find(cases, cases + count, RDC_DIAG_POLL_LATENCY), cases + count);
}

TEST(SmiLibPollLatencyTest, ReadsTheMonitoringFieldsOfEachGpu) {
  Lib l;
  ASSERT_EQ(l.run_on({0, 2}, "samples=2;interval_ms=0"), RDC_ST_OK);
  EXPECT_EQ(l.result->status, RDC_DIAG_RESULT_PASS) << l.result->info;
  EXPECT_EQ(l.result->test_case, RDC_DIAG_POLL_LATENCY);
  EXPECT_NE(strstr(l.result->info, "Read 9 fields from 2 GPUs 2 times, back to back"), nullptr)
      << l.result->info;
  EXPECT_NE(strstr(l.result->info, "36 values read, 0 failed, 0 not supported."), nullptr)
      << l.result->info;
  ASSERT_EQ(l.result->per_gpu_result_count, 2u);
  EXPECT_EQ(l.result->gpu_results[0].gpu_index, 0u);
  EXPECT_EQ(l.result->gpu_results[1].gpu_index, 2u);

  // One untimed read and two polls of each GPU, every field in order.
  const auto calls = l.fetcher->calls();
  ASSERT_EQ(calls.size(), 2u * 3u * kExpectedFields.size());
  for (size_t read = 0; read < 6; read++) {
    const uint32_t gpu = (read % 2 == 0) ? 0 : 2;
    for (size_t f = 0; f < kExpectedFields.size(); f++) {
      const auto& call = calls[read * kExpectedFields.size() + f];
      EXPECT_EQ(call.first, gpu) << "read " << read << " field " << f;
      EXPECT_EQ(call.second, kExpectedFields[f]) << "read " << read << " field " << f;
    }
  }
}

TEST(SmiLibPollLatencyTest, CountsValuesByStatus) {
  Lib l;
  l.fetcher->behavior[RDC_FI_MEMORY_TEMP].status = AMDSMI_STATUS_NOT_SUPPORTED;
  l.fetcher->behavior[RDC_FI_GPU_TEMP].status = AMDSMI_STATUS_NOT_YET_IMPLEMENTED;
  l.fetcher->behavior[RDC_FI_POWER_USAGE].status = AMDSMI_STATUS_INVAL;
  ASSERT_EQ(l.run_on({0}, "samples=2;interval_ms=0"), RDC_ST_OK);
  EXPECT_EQ(l.result->status, RDC_DIAG_RESULT_WARN);
  EXPECT_NE(strstr(l.result->info, "12 values read, 2 failed, 4 not supported."), nullptr)
      << l.result->info;
  EXPECT_EQ(l.result->gpu_results[0].gpu_result.code, 2u);
}

TEST(SmiLibPollLatencyTest, CountsValuesTheFetcherDidNotFillIn) {
  Lib l;
  // fetch_smi_field() returns early, before it fills in the value.
  l.fetcher->behavior[RDC_FI_GPU_CLOCK] = {false, AMDSMI_STATUS_SUCCESS, RDC_ST_NOT_SUPPORTED};
  l.fetcher->behavior[RDC_FI_GPU_UTIL] = {false, AMDSMI_STATUS_SUCCESS, RDC_ST_UNKNOWN_ERROR};
  ASSERT_EQ(l.run_on({0}, "samples=2;interval_ms=0"), RDC_ST_OK);
  EXPECT_EQ(l.result->status, RDC_DIAG_RESULT_WARN);
  EXPECT_NE(strstr(l.result->info, "14 values read, 2 failed, 2 not supported."), nullptr)
      << l.result->info;
}

TEST(SmiLibPollLatencyTest, ReportsValuesTheFetcherDidNotFillInToTheCollector) {
  Lib l;
  // Fill the stack with a successful read first, so a value the next read
  // leaves untouched would show up as a stale success.
  rdc_gpu_field_t fields[] = {{3, RDC_FI_GPU_CLOCK}, {3, RDC_FI_GPU_UTIL}};
  auto ignore = [](rdc_gpu_field_value_t*, uint32_t, void*) { return RDC_ST_OK; };
  ASSERT_EQ(l.lib.rdc_telemetry_fields_value_get(fields, 2, ignore, nullptr), RDC_ST_OK);

  l.fetcher->behavior[RDC_FI_GPU_CLOCK] = {false, AMDSMI_STATUS_SUCCESS, RDC_ST_NOT_SUPPORTED};
  l.fetcher->behavior[RDC_FI_GPU_UTIL] = {false, AMDSMI_STATUS_SUCCESS, RDC_ST_UNKNOWN_ERROR};
  std::vector<rdc_gpu_field_value_t> seen;
  auto collect = [](rdc_gpu_field_value_t* values, uint32_t n, void* user_data) {
    auto* out = static_cast<std::vector<rdc_gpu_field_value_t>*>(user_data);
    out->insert(out->end(), values, values + n);
    return RDC_ST_OK;
  };
  ASSERT_EQ(l.lib.rdc_telemetry_fields_value_get(fields, 2, collect, &seen), RDC_ST_OK);
  ASSERT_EQ(seen.size(), 2u);
  EXPECT_EQ(seen[0].gpu_index, 3u);
  EXPECT_EQ(seen[0].field_value.field_id, RDC_FI_GPU_CLOCK);
  EXPECT_EQ(seen[0].field_value.status, static_cast<int>(AMDSMI_STATUS_NOT_SUPPORTED));
  EXPECT_EQ(seen[1].gpu_index, 3u);
  EXPECT_EQ(seen[1].field_value.field_id, RDC_FI_GPU_UTIL);
  EXPECT_EQ(seen[1].field_value.status, static_cast<int>(AMDSMI_STATUS_UNKNOWN_ERROR));
}

TEST(SmiLibPollLatencyTest, SkipsTheCpusOfTheGroup) {
  Lib l;
  const uint32_t cpu = cpu_entity(0);
  ASSERT_EQ(l.run_on({cpu, 1}, "samples=1;interval_ms=0"), RDC_ST_OK);
  EXPECT_EQ(l.result->status, RDC_DIAG_RESULT_PASS) << l.result->info;
  ASSERT_EQ(l.result->per_gpu_result_count, 1u);
  EXPECT_EQ(l.result->gpu_results[0].gpu_index, 1u);
  for (const auto& call : l.fetcher->calls()) {
    EXPECT_NE(call.first, cpu);
  }
}

TEST(SmiLibPollLatencyTest, FailsForAGroupOfCpus) {
  Lib l;
  ASSERT_EQ(l.run_on({cpu_entity(0), cpu_entity(1)}, ""), RDC_ST_OK);
  EXPECT_EQ(l.result->status, RDC_DIAG_RESULT_FAIL);
  EXPECT_STREQ(l.result->info, "The group has no GPUs to poll.");
  EXPECT_TRUE(l.fetcher->calls().empty());
}

TEST(SmiLibPollLatencyTest, RejectsABadConfig) {
  Lib l;
  EXPECT_EQ(l.run_on({0}, "samples=0"), RDC_ST_BAD_PARAMETER);
  EXPECT_EQ(l.result->status, RDC_DIAG_RESULT_FAIL);
  EXPECT_EQ(l.result->test_case, RDC_DIAG_POLL_LATENCY);
  EXPECT_EQ(l.result->per_gpu_result_count, 0u);
  EXPECT_STREQ(l.result->info, "Poll latency check: samples must be a number from 1 to 1000.");
  EXPECT_TRUE(l.fetcher->calls().empty());
}

TEST(SmiLibPollLatencyTest, DoesNotReadWhileTheCollectorReads) {
  Lib l;
  l.fetcher->delay = std::chrono::microseconds(200);
  std::atomic<bool> stop{false};
  std::thread collector([&] {
    std::vector<rdc_gpu_field_t> fields;
    for (rdc_field_t f : kExpectedFields) fields.push_back({0, f});
    auto ignore = [](rdc_gpu_field_value_t*, uint32_t, void*) { return RDC_ST_OK; };
    while (!stop) {
      l.lib.rdc_telemetry_fields_value_get(fields.data(), static_cast<uint32_t>(fields.size()),
                                           ignore, nullptr);
      // rdcd's collector also pauses between updates; std::mutex is not fair.
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  });
  const rdc_status_t status = l.run_on({0}, "samples=10;interval_ms=0");
  stop = true;
  collector.join();
  ASSERT_EQ(status, RDC_ST_OK);
  EXPECT_EQ(l.result->status, RDC_DIAG_RESULT_PASS) << l.result->info;
  EXPECT_FALSE(l.fetcher->overlapped);
}

}  // namespace
