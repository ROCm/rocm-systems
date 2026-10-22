// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * The helper behind amdsmi_get_nic_hwmon_sensors: how many records reach the caller's array, which
 * status says so, and where each record field lands.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "amd_smi/impl/amd_smi_nic_stats.h"
#include "nic_unit_fixture.h"

namespace {

using amd::smi::copy_nic_sensors;

constexpr auto kUnsetValue = int64_t{-777};
constexpr auto kFirstValue = int64_t{1000};

// Stands in for the internal record type; the helper only reads these fields.
struct TestRecord {
  uint32_t category;
  uint32_t attribute;
  uint32_t unit;
  uint32_t channel;
  uint32_t instance;
  std::string label;
  int64_t value;
};

auto make_records(uint32_t count) -> std::vector<TestRecord> {
  auto records = std::vector<TestRecord>{};
  for (auto i = uint32_t{0}; i < count; ++i) {
    records.push_back({AMDSMI_METRIC_CATEGORY_TEMPERATURE, AMDSMI_NIC_SENSOR_NAME_INPUT,
                       AMDSMI_METRIC_UNIT_MILLIDEGREE_CELSIUS, i, 0, "sensor_" + std::to_string(i),
                       kFirstValue + i});
  }
  return records;
}

auto make_out(uint32_t capacity) -> std::vector<amdsmi_nic_sensor_t> {
  auto out = std::vector<amdsmi_nic_sensor_t>(capacity);
  for (auto& sensor : out) {
    sensor = {};
    sensor.value = kUnsetValue;
  }
  return out;
}

TEST_F(NicUnit, NicSensorsCopyReportsSuccessForAnExactFit) {
  const auto records = make_records(3);
  auto out = make_out(3);
  auto num_sensors = uint32_t{3};

  EXPECT_EQ(copy_nic_sensors(records, &num_sensors, out.data()), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(num_sensors, 3U);
  EXPECT_EQ(out[0].value, 1000);
  EXPECT_EQ(out[2].value, 1002);
  EXPECT_STREQ(out[2].label, "sensor_2");
}

TEST_F(NicUnit, NicSensorsCopyIntoAShortArrayReportsInsufficientSize) {
  const auto records = make_records(5);
  auto out = make_out(4);
  auto num_sensors = uint32_t{2};

  EXPECT_EQ(copy_nic_sensors(records, &num_sensors, out.data()), AMDSMI_STATUS_INSUFFICIENT_SIZE);
  EXPECT_EQ(num_sensors, 5U);
  EXPECT_EQ(out[0].value, 1000);
  EXPECT_EQ(out[1].value, 1001);
  EXPECT_EQ(out[2].value, kUnsetValue);
  EXPECT_EQ(out[3].value, kUnsetValue);
}

TEST_F(NicUnit, NicSensorsCopyOfNoRecordsSucceedsWithACountOfZero) {
  const auto records = make_records(0);
  auto out = make_out(2);
  auto num_sensors = uint32_t{2};

  EXPECT_EQ(copy_nic_sensors(records, &num_sensors, out.data()), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(num_sensors, 0U);
  EXPECT_EQ(out[0].value, kUnsetValue);
}

TEST_F(NicUnit, NicSensorsCopyIntoALargeArrayLeavesTheRestUntouched) {
  const auto records = make_records(2);
  auto out = make_out(6);
  auto num_sensors = uint32_t{6};

  EXPECT_EQ(copy_nic_sensors(records, &num_sensors, out.data()), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(num_sensors, 2U);
  EXPECT_EQ(out[1].value, 1001);
  EXPECT_EQ(out[2].value, kUnsetValue);
  EXPECT_EQ(out[5].value, kUnsetValue);
}

TEST_F(NicUnit, NicSensorsCopyWithoutAnArrayOnlyReportsTheCount) {
  const auto records = make_records(4);
  auto num_sensors = uint32_t{0};

  EXPECT_EQ(copy_nic_sensors(records, &num_sensors, nullptr), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(num_sensors, 4U);
}

TEST_F(NicUnit, NicSensorsCopyMapsEveryField) {
  const auto records =
      std::vector<TestRecord>{{AMDSMI_METRIC_CATEGORY_POWER, AMDSMI_NIC_SENSOR_NAME_CRIT,
                               AMDSMI_METRIC_UNIT_MICROWATT, 7, 3, "total power", 35340000}};
  auto out = make_out(1);
  auto num_sensors = uint32_t{1};

  ASSERT_EQ(copy_nic_sensors(records, &num_sensors, out.data()), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(out[0].category, AMDSMI_METRIC_CATEGORY_POWER);
  EXPECT_EQ(out[0].name, AMDSMI_NIC_SENSOR_NAME_CRIT);
  EXPECT_EQ(out[0].unit, AMDSMI_METRIC_UNIT_MICROWATT);
  EXPECT_EQ(out[0].channel, 7U);
  EXPECT_EQ(out[0].instance, 3U);
  EXPECT_STREQ(out[0].label, "total power");
  EXPECT_EQ(out[0].value, 35340000);
}

TEST_F(NicUnit, NicSensorsCopyTruncatesALongLabelWithATerminator) {
  auto records = make_records(1);
  records[0].label = std::string(sizeof(amdsmi_nic_sensor_t::label) + 20, 'x');
  auto out = make_out(1);
  auto num_sensors = uint32_t{1};

  ASSERT_EQ(copy_nic_sensors(records, &num_sensors, out.data()), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(std::string(out[0].label).size(), sizeof(out[0].label) - 1);
}

}  // namespace
