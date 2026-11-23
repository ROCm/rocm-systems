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
#include "rdc_tests/functional/rdc_poll_latency.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "rdc/rdc.h"

namespace {

// The fields that GPU monitoring polls most: clock, temperatures, power,
// utilization, VRAM usage and ECC error counts.
const rdc_field_t kFields[] = {
    RDC_FI_GPU_CLOCK,         RDC_FI_MEMORY_TEMP,       RDC_FI_GPU_TEMP,
    RDC_FI_GPU_JUNCTION_TEMP, RDC_FI_POWER_USAGE,       RDC_FI_GPU_UTIL,
    RDC_FI_GPU_MEMORY_USAGE,  RDC_FI_ECC_CORRECT_TOTAL, RDC_FI_ECC_UNCORRECT_TOTAL};
constexpr uint32_t kFieldCount = sizeof(kFields) / sizeof(kFields[0]);
constexpr uint32_t kPolls = 20;
constexpr std::chrono::milliseconds kInterval(100);

using Clock = std::chrono::steady_clock;

uint64_t microseconds_since(Clock::time_point start) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count());
}

// Nearest-rank percentile of a sorted, non-empty list: the smallest value with
// at least `percent`% of the values at or below it.
uint64_t percentile(const std::vector<uint64_t>& sorted, size_t percent) {
  size_t rank = (percent * sorted.size() + 99) / 100;
  rank = std::min(std::max<size_t>(rank, 1), sorted.size());
  return sorted[rank - 1];
}

// "12.34 ms", or "56 us" below a millisecond.
std::string format_time(uint64_t us) {
  char buf[32];
  if (us < 1000) {
    snprintf(buf, sizeof(buf), "%u us", static_cast<unsigned>(us));
  } else {
    snprintf(buf, sizeof(buf), "%.2f ms", static_cast<double>(us) / 1000.0);
  }
  return buf;
}

// "median 1.23 ms, p95 2.34 ms, max 3.45 ms". The same numbers become test
// properties, so they also appear in --gtest_output files.
std::string summarize(std::vector<uint64_t> us, const std::string& name) {
  std::sort(us.begin(), us.end());
  const uint64_t median = percentile(us, 50);
  const uint64_t p95 = percentile(us, 95);
  const uint64_t max = us.back();
  testing::Test::RecordProperty(name + "_median_us", std::to_string(median));
  testing::Test::RecordProperty(name + "_p95_us", std::to_string(p95));
  testing::Test::RecordProperty(name + "_max_us", std::to_string(max));
  return "median " + format_time(median) + ", p95 " + format_time(p95) + ", max " +
         format_time(max);
}

}  // namespace

TestRdcPollLatency::TestRdcPollLatency() : TestBase() {
  set_title("\tRDC Poll Latency Test");
  set_description(
      "\tReports how long RDC takes to read telemetry from all GPUs: 9 fields per "
      "GPU, 20 polls, 100 ms apart. Embedded mode times the collection from the "
      "GPUs and the reads of the latest values; standalone mode times the reads "
      "from rdcd. The test fails only when no value can be read.");
}

TestRdcPollLatency::~TestRdcPollLatency(void) {}

void TestRdcPollLatency::SetUp(void) {
  TestBase::SetUp();
  rdc_status_t result = AllocateRDCChannel();
  ASSERT_EQ(result, RDC_ST_OK);
}

void TestRdcPollLatency::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestRdcPollLatency::DisplayResults(void) const {
  TestBase::DisplayResults();
  std::cout << report_;
}

void TestRdcPollLatency::Close() {
  TestBase::Close();
  // Release what Run() created here, so that a failed assertion in Run() does
  // not leave, for example, a 100 ms watch polling a running rdcd.
  if (watching_) {
    EXPECT_EQ(rdc_field_unwatch(rdc_handle, group_id_, field_group_id_), RDC_ST_OK);
  }
  if (group_created_) {
    EXPECT_EQ(rdc_group_gpu_destroy(rdc_handle, group_id_), RDC_ST_OK);
  }
  if (field_group_created_) {
    EXPECT_EQ(rdc_group_field_destroy(rdc_handle, field_group_id_), RDC_ST_OK);
  }
  if (started_) {
    if (standalone_) {
      IF_VERB(STANDARD) { std::cout << "\t**Disconnecting from host....\n" << std::endl; }
      EXPECT_EQ(rdc_disconnect(rdc_handle), RDC_ST_OK);
    } else {
      IF_VERB(STANDARD) { std::cout << "\t**Stopping Embedded RDC Engine....\n" << std::endl; }
      EXPECT_EQ(rdc_stop_embedded(rdc_handle), RDC_ST_OK);
    }
  }
  EXPECT_EQ(rdc_shutdown(), RDC_ST_OK);
}

void TestRdcPollLatency::Run(void) {
  TestBase::Run();
  rdc_status_t result;
  if (standalone_) {
    // The rdcd that -s and -p name, else the one on this host.
    const std::string address =
        (monitor_server_ip().empty() ? std::string("localhost") : monitor_server_ip()) + ":" +
        (monitor_server_port().empty() ? std::string("50051") : monitor_server_port());
    IF_VERB(STANDARD) { std::cout << "\t**Connecting to " << address << "....\n" << std::endl; }
    result = rdc_connect(address.c_str(), &rdc_handle, nullptr, nullptr, nullptr);
    ASSERT_EQ(result, RDC_ST_OK);
  } else {
    // In manual mode the fields are read only when this test calls
    // rdc_field_update_all(), so the test can time the collection itself.
    IF_VERB(STANDARD) { std::cout << "\t**Starting embedded RDC engine....\n" << std::endl; }
    result = rdc_start_embedded(RDC_OPERATION_MODE_MANUAL, &rdc_handle);
    ASSERT_EQ(result, RDC_ST_OK);
  }
  started_ = true;

  rdc_group_info_t group_info;
  result = rdc_group_gpu_create(rdc_handle, RDC_GROUP_DEFAULT, "GRP_POLL_LATENCY", &group_id_);
  ASSERT_EQ(result, RDC_ST_OK);
  group_created_ = true;
  result = rdc_group_gpu_get_info(rdc_handle, group_id_, &group_info);
  ASSERT_EQ(result, RDC_ST_OK);
  ASSERT_GT(group_info.count, 0u);

  rdc_field_t fields[kFieldCount];
  std::copy(kFields, kFields + kFieldCount, fields);
  result =
      rdc_group_field_create(rdc_handle, kFieldCount, fields, "FLD_POLL_LATENCY", &field_group_id_);
  ASSERT_EQ(result, RDC_ST_OK);
  field_group_created_ = true;
  // rdcd reads the fields every 100 ms. In manual mode they are due at every
  // rdc_field_update_all() call.
  const uint64_t update_freq_us = standalone_ ? 100000 : 1000;
  result = rdc_field_watch(rdc_handle, group_id_, field_group_id_, update_freq_us, 60, 10);
  ASSERT_EQ(result, RDC_ST_OK);
  watching_ = true;

  // The first update includes one-time setup, so it is not timed.
  result = rdc_field_update_all(rdc_handle, 1);
  ASSERT_EQ(result, RDC_ST_OK);

  std::vector<uint64_t> collect_us;
  std::vector<uint64_t> read_us;
  uint64_t values_read = 0;
  const uint64_t values_total = uint64_t{kPolls} * group_info.count * kFieldCount;
  const Clock::time_point start = Clock::now();
  for (uint32_t poll = 0; poll < kPolls; poll++) {
    std::this_thread::sleep_until(start + poll * kInterval);
    if (!standalone_) {
      const Clock::time_point collect_start = Clock::now();
      result = rdc_field_update_all(rdc_handle, 1);
      collect_us.push_back(microseconds_since(collect_start));
      EXPECT_EQ(result, RDC_ST_OK);
    }
    const Clock::time_point read_start = Clock::now();
    for (uint32_t i = 0; i < group_info.count; i++) {
      for (rdc_field_t field : kFields) {
        rdc_field_value value;
        if (rdc_field_get_latest_value(rdc_handle, group_info.entity_ids[i], field, &value) ==
            RDC_ST_OK) {
          values_read++;
        }
      }
    }
    read_us.push_back(microseconds_since(read_start));
  }

  const std::string mode = standalone_ ? "standalone" : "embedded";
  testing::Test::RecordProperty("mode", mode);
  testing::Test::RecordProperty("gpus", std::to_string(group_info.count));
  testing::Test::RecordProperty("values_read", std::to_string(values_read));
  testing::Test::RecordProperty("values_total", std::to_string(values_total));
  report_ = "\tRead " + std::to_string(kFieldCount) + " fields from " +
            std::to_string(group_info.count) + (group_info.count == 1 ? " GPU " : " GPUs ") +
            std::to_string(kPolls) + " times, 100 ms apart, in " + mode + " mode:\n";
  if (!standalone_) {
    report_ +=
        "\t  Collection from the GPUs (rdc_field_update_all): " + summarize(collect_us, "collect") +
        " per poll\n";
  }
  report_ += "\t  Reading the latest values (" + std::to_string(group_info.count * kFieldCount) +
             " rdc_field_get_latest_value calls): " + summarize(read_us, "read") + " per poll\n";
  report_ += "\t  " + std::to_string(values_read) + " of " + std::to_string(values_total) +
             " values read\n";

  EXPECT_GT(values_read, 0u) << "No value could be read";
}
