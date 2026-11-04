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

// RdcTelemetryModule::rdc_telemetry_fields_value_get() reports the fields no
// module supports through the callback. With every field supported it used to call
// the callback with &unsupport_fields[0] of an empty vector, which aborts under
// libstdc++ assertions and otherwise hands the callback a null pointer.
//
// The test builds RdcTelemetryModule.cc itself with a fake telemetry module, so it
// needs no GPU; librdc_bootstrap provides the logger.

#include <gtest/gtest.h>

#include <list>
#include <memory>
#include <vector>

#include "rdc_lib/impl/RdcTelemetryModule.h"

namespace {

constexpr rdc_field_t kSupportedField = RDC_FI_GPU_TEMP;
constexpr rdc_field_t kUnsupportedField = RDC_FI_GPU_MEMORY_USAGE;

class FakeTelemetry : public amd::rdc::RdcTelemetry {
 public:
  rdc_status_t rdc_telemetry_fields_query(uint32_t field_ids[MAX_NUM_FIELDS],
                                          uint32_t* field_count) override {
    field_ids[0] = kSupportedField;
    *field_count = 1;
    return RDC_ST_OK;
  }

  rdc_status_t rdc_telemetry_fields_value_get(rdc_gpu_field_t* fields, uint32_t fields_count,
                                              rdc_field_value_f callback,
                                              void* user_data) override {
    std::vector<rdc_gpu_field_value_t> values(fields_count);
    for (uint32_t i = 0; i < fields_count; i++) {
      values[i].gpu_index = fields[i].gpu_index;
      values[i].field_value.field_id = fields[i].field_id;
      values[i].field_value.status = RDC_ST_OK;
    }
    return callback(values.data(), fields_count, user_data);
  }

  rdc_status_t rdc_telemetry_fields_watch(rdc_gpu_field_t*, uint32_t) override { return RDC_ST_OK; }
  rdc_status_t rdc_telemetry_fields_unwatch(rdc_gpu_field_t*, uint32_t) override {
    return RDC_ST_OK;
  }
};

struct Received {
  std::vector<rdc_gpu_field_value_t> values;
  int calls = 0;
};

rdc_status_t collect(rdc_gpu_field_value_t* values, uint32_t count, void* user_data) {
  if (values == nullptr || count == 0) {
    return RDC_ST_BAD_PARAMETER;
  }
  auto* received = static_cast<Received*>(user_data);
  received->calls++;
  received->values.insert(received->values.end(), values, values + count);
  return RDC_ST_OK;
}

class TelemetryModuleTest : public ::testing::Test {
 protected:
  amd::rdc::RdcTelemetryModule module_{
      std::list<amd::rdc::RdcTelemetryPtr>{std::make_shared<FakeTelemetry>()}};
};

TEST_F(TelemetryModuleTest, AllFieldsSupportedMakesNoCallbackForUnsupported) {
  rdc_gpu_field_t fields[] = {{0, kSupportedField}, {1, kSupportedField}};
  Received received;
  EXPECT_EQ(module_.rdc_telemetry_fields_value_get(fields, 2, collect, &received), RDC_ST_OK);
  EXPECT_EQ(received.calls, 1);
  ASSERT_EQ(received.values.size(), 2u);
  for (const auto& value : received.values) {
    EXPECT_EQ(value.field_value.status, RDC_ST_OK);
  }
}

TEST_F(TelemetryModuleTest, UnsupportedFieldsAreReported) {
  rdc_gpu_field_t fields[] = {{0, kSupportedField}, {0, kUnsupportedField}};
  Received received;
  EXPECT_EQ(module_.rdc_telemetry_fields_value_get(fields, 2, collect, &received), RDC_ST_OK);
  EXPECT_EQ(received.calls, 2);
  ASSERT_EQ(received.values.size(), 2u);
  EXPECT_EQ(received.values[0].field_value.field_id, kSupportedField);
  EXPECT_EQ(received.values[0].field_value.status, RDC_ST_OK);
  EXPECT_EQ(received.values[1].field_value.field_id, kUnsupportedField);
  EXPECT_EQ(received.values[1].field_value.status, RDC_ST_NOT_SUPPORTED);
}

TEST_F(TelemetryModuleTest, OnlyUnsupportedFieldsAreReported) {
  rdc_gpu_field_t fields[] = {{0, kUnsupportedField}};
  Received received;
  EXPECT_EQ(module_.rdc_telemetry_fields_value_get(fields, 1, collect, &received), RDC_ST_OK);
  EXPECT_EQ(received.calls, 1);
  ASSERT_EQ(received.values.size(), 1u);
  EXPECT_EQ(received.values[0].field_value.status, RDC_ST_NOT_SUPPORTED);
}

}  // namespace
