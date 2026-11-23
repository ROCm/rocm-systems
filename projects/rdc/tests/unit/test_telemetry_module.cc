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

// How RdcTelemetryModule hands watched fields to the telemetry libraries, with a stub
// library in place of amd-smi, rocprofiler and the others.

#include <gtest/gtest.h>
#include <stdio.h>
#include <unistd.h>

#include <list>
#include <memory>
#include <set>
#include <vector>

#include "rdc/rdc.h"
#include "rdc_lib/RdcTelemetry.h"
#include "rdc_lib/impl/RdcTelemetryModule.h"

namespace {

using amd::rdc::RdcTelemetry;
using amd::rdc::RdcTelemetryModule;
using amd::rdc::RdcTelemetryPtr;

constexpr int kTooManyFields = 1;  // exit code: a call got more than MAX_NUM_FIELDS fields
constexpr int kWrongFields = 2;    // exit code: the library did not get each field once, in order

// A library that supports RDC_FI_GPU_CLOCK and records the fields it is asked to fetch.
class StubLibrary : public RdcTelemetry {
 public:
  rdc_status_t rdc_telemetry_fields_query(uint32_t field_ids[MAX_NUM_FIELDS],
                                          uint32_t* field_count) override {
    field_ids[0] = RDC_FI_GPU_CLOCK;
    *field_count = 1;
    return RDC_ST_OK;
  }

  rdc_status_t rdc_telemetry_fields_value_get(rdc_gpu_field_t* fields, uint32_t fields_count,
                                              rdc_field_value_f, void*) override {
    // Libraries hold the fields of one call in arrays of MAX_NUM_FIELDS.
    if (fields_count > MAX_NUM_FIELDS) {
      fprintf(stderr, "one call got %u fields\n", fields_count);
      _exit(kTooManyFields);
    }
    for (uint32_t i = 0; i < fields_count; i++) {
      fetched.push_back(fields[i].gpu_index);
    }
    return RDC_ST_OK;
  }

  rdc_status_t rdc_telemetry_fields_watch(rdc_gpu_field_t*, uint32_t) override { return RDC_ST_OK; }

  rdc_status_t rdc_telemetry_fields_unwatch(rdc_gpu_field_t*, uint32_t) override {
    return RDC_ST_OK;
  }

  std::vector<uint32_t> fetched;  // gpu_index of each field, in the order fetched
};

// One callback: the values, which the module owns only for the call.
struct Report {
  bool values_were_null;
  std::vector<rdc_gpu_field_value_t> values;
};

rdc_status_t record(rdc_gpu_field_value_t* values, uint32_t count, void* user_data) {
  Report report{values == nullptr, {}};
  if (values != nullptr) {
    report.values.assign(values, values + count);
  }
  static_cast<std::vector<Report>*>(user_data)->push_back(report);
  return RDC_ST_OK;
}

// `count` fields; field i is on GPU index i, so each one can be told apart.
std::vector<rdc_gpu_field_t> watch(uint32_t count, rdc_field_t field) {
  std::vector<rdc_gpu_field_t> fields(count);
  for (uint32_t i = 0; i < count; i++) {
    fields[i] = {i, field};
  }
  return fields;
}

// Two watches of 64 GPUs and 128 fields that are due in the same update: 16,384 fields for
// one library. RDC copied them into an array of MAX_NUM_FIELDS (8,192) on the stack; that
// can crash the test, hence a death test.
TEST(RdcTelemetryModuleTest, FetchesMoreFieldsThanOneLibraryCallTakes) {
  EXPECT_EXIT(
      {
        auto stub = std::make_shared<StubLibrary>();
        RdcTelemetryModule module(std::list<RdcTelemetryPtr>{stub});
        std::vector<rdc_gpu_field_t> fields = watch(2 * MAX_NUM_FIELDS + 1, RDC_FI_GPU_CLOCK);
        std::vector<Report> reports;
        module.rdc_telemetry_fields_value_get(fields.data(), fields.size(), record, &reports);
        for (uint32_t i = 0; i < fields.size(); i++) {
          if (i >= stub->fetched.size() || stub->fetched[i] != i) {
            _exit(kWrongFields);
          }
        }
        _exit(stub->fetched.size() == fields.size() ? 0 : kWrongFields);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST(RdcTelemetryModuleTest, ReportsUnsupportedFieldsOnlyWhenThereAreSome) {
  auto stub = std::make_shared<StubLibrary>();
  RdcTelemetryModule module(std::list<RdcTelemetryPtr>{stub});
  std::vector<Report> reports;

  std::vector<rdc_gpu_field_t> supported = watch(2, RDC_FI_GPU_CLOCK);
  module.rdc_telemetry_fields_value_get(supported.data(), supported.size(), record, &reports);
  EXPECT_EQ(stub->fetched.size(), 2u);
  EXPECT_TRUE(reports.empty()) << "called back for no unsupported fields, with "
                               << (reports[0].values_were_null ? "a null pointer" : "values");

  std::vector<rdc_gpu_field_t> mixed = {{0, RDC_FI_GPU_CLOCK}, {3, RDC_FI_MEM_CLOCK}};
  module.rdc_telemetry_fields_value_get(mixed.data(), mixed.size(), record, &reports);
  ASSERT_EQ(reports.size(), 1u);
  ASSERT_EQ(reports[0].values.size(), 1u);
  EXPECT_EQ(reports[0].values[0].gpu_index, 3u);
  EXPECT_EQ(reports[0].values[0].field_value.field_id, RDC_FI_MEM_CLOCK);
  EXPECT_EQ(reports[0].values[0].field_value.status, RDC_ST_NOT_SUPPORTED);
}

// A library that supports `count` field IDs from `first` on and returns `status` from watch,
// unwatch and value_get, counting those calls.
class ScriptedLibrary : public RdcTelemetry {
 public:
  ScriptedLibrary(uint32_t first, uint32_t count, rdc_status_t status)
      : first_(first), count_(count), status_(status) {}

  rdc_status_t rdc_telemetry_fields_query(uint32_t field_ids[MAX_NUM_FIELDS],
                                          uint32_t* field_count) override {
    for (uint32_t i = 0; i < count_; i++) {
      field_ids[i] = first_ + i;
    }
    *field_count = count_;
    return RDC_ST_OK;
  }

  rdc_status_t rdc_telemetry_fields_value_get(rdc_gpu_field_t*, uint32_t, rdc_field_value_f,
                                              void*) override {
    value_gets++;
    return status_;
  }

  rdc_status_t rdc_telemetry_fields_watch(rdc_gpu_field_t*, uint32_t) override {
    watches++;
    return status_;
  }

  rdc_status_t rdc_telemetry_fields_unwatch(rdc_gpu_field_t*, uint32_t) override {
    unwatches++;
    return status_;
  }

  int value_gets = 0;
  int watches = 0;
  int unwatches = 0;

 private:
  const uint32_t first_;
  const uint32_t count_;
  const rdc_status_t status_;
};

// Two libraries of 5,000 fields each: more than the MAX_NUM_FIELDS (8,192) that the caller's
// array holds. This array has room for twice that, so an overrun lands in its guard words
// instead of the heap.
TEST(RdcTelemetryModuleTest, QueriesAtMostMaxNumFields) {
  auto first = std::make_shared<ScriptedLibrary>(1, 5000, RDC_ST_OK);
  auto second = std::make_shared<ScriptedLibrary>(6000, 5000, RDC_ST_OK);
  RdcTelemetryModule module(std::list<RdcTelemetryPtr>{first, second});
  constexpr uint32_t kGuard = 0xA5A5A5A5;
  std::vector<uint32_t> field_ids(2 * MAX_NUM_FIELDS, kGuard);
  uint32_t count = 0;
  EXPECT_EQ(module.rdc_telemetry_fields_query(field_ids.data(), &count), RDC_ST_MAX_LIMIT);
  EXPECT_EQ(count, static_cast<uint32_t>(MAX_NUM_FIELDS));
  for (size_t i = MAX_NUM_FIELDS; i < field_ids.size(); i++) {
    ASSERT_EQ(field_ids[i], kGuard) << "a field ID was written at index " << i;
  }
}

TEST(RdcTelemetryModuleTest, QueriesEveryLibrarysFields) {
  auto first = std::make_shared<ScriptedLibrary>(4000, 3, RDC_ST_OK);
  auto second = std::make_shared<ScriptedLibrary>(5000, 2, RDC_ST_OK);
  RdcTelemetryModule module(std::list<RdcTelemetryPtr>{first, second});
  std::vector<uint32_t> field_ids(MAX_NUM_FIELDS);
  uint32_t count = 0;
  ASSERT_EQ(module.rdc_telemetry_fields_query(field_ids.data(), &count), RDC_ST_OK);
  ASSERT_LE(count, static_cast<uint32_t>(MAX_NUM_FIELDS));
  std::vector<uint32_t> got(field_ids.begin(), field_ids.begin() + count);
  EXPECT_EQ(got, (std::vector<uint32_t>{4000, 4001, 4002, 5000, 5001}));
}

// A library's failure comes back to the caller, and the other library still gets each call.
TEST(RdcTelemetryModuleTest, ReturnsALibrarysFailure) {
  auto failing = std::make_shared<ScriptedLibrary>(4000, 1, RDC_ST_NOT_SUPPORTED);
  auto working = std::make_shared<ScriptedLibrary>(5000, 1, RDC_ST_OK);
  RdcTelemetryModule module(std::list<RdcTelemetryPtr>{failing, working});
  std::vector<rdc_gpu_field_t> fields = {{0, static_cast<rdc_field_t>(4000)},
                                         {0, static_cast<rdc_field_t>(5000)}};
  std::vector<Report> reports;

  EXPECT_EQ(module.rdc_telemetry_fields_watch(fields.data(), fields.size()), RDC_ST_NOT_SUPPORTED);
  EXPECT_EQ(module.rdc_telemetry_fields_unwatch(fields.data(), fields.size()),
            RDC_ST_NOT_SUPPORTED);
  EXPECT_EQ(module.rdc_telemetry_fields_value_get(fields.data(), fields.size(), record, &reports),
            RDC_ST_NOT_SUPPORTED);
  EXPECT_EQ(failing->watches, 1);
  EXPECT_EQ(working->watches, 1);
  EXPECT_EQ(working->unwatches, 1);
  EXPECT_EQ(working->value_gets, 1);
}

// When several libraries fail, each one still gets every call.
TEST(RdcTelemetryModuleTest, CallsEveryLibraryWhenSeveralFail) {
  auto first = std::make_shared<ScriptedLibrary>(4000, 1, RDC_ST_NOT_SUPPORTED);
  auto second = std::make_shared<ScriptedLibrary>(5000, 1, RDC_ST_UNKNOWN_ERROR);
  RdcTelemetryModule module(std::list<RdcTelemetryPtr>{first, second});
  std::vector<rdc_gpu_field_t> fields = {{0, static_cast<rdc_field_t>(4000)},
                                         {0, static_cast<rdc_field_t>(5000)}};
  std::vector<Report> reports;
  const std::set<rdc_status_t> failures = {RDC_ST_NOT_SUPPORTED, RDC_ST_UNKNOWN_ERROR};

  EXPECT_EQ(failures.count(module.rdc_telemetry_fields_watch(fields.data(), fields.size())), 1u);
  EXPECT_EQ(failures.count(module.rdc_telemetry_fields_unwatch(fields.data(), fields.size())), 1u);
  EXPECT_EQ(failures.count(module.rdc_telemetry_fields_value_get(fields.data(), fields.size(),
                                                                 record, &reports)),
            1u);
  for (const auto& library : {first, second}) {
    EXPECT_EQ(library->watches, 1);
    EXPECT_EQ(library->unwatches, 1);
    EXPECT_EQ(library->value_gets, 1);
  }
}

}  // namespace
