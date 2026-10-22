// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "telemetry_read.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "functional/nic/nic_handles.h"
#include "test_common.h"

TestNicTelemetryRead::TestNicTelemetryRead() : TestBase() {
  set_title("AMDSMI NIC Telemetry Read Test");
  set_description(
      "This test verifies the public NIC telemetry / firmware bridge: "
      "amdsmi_get_nic_telemetry() and amdsmi_get_nic_fw_info() return sane, "
      "bounded values and reject null output pointers. It references "
      "amdsmi_nic_telemetry_t by field so a field dropped from the getter's "
      "copy bridge surfaces here.");
}

TestNicTelemetryRead::~TestNicTelemetryRead(void) {}

// NIC telemetry needs the NIC backend, not the default GPU init.
void TestNicTelemetryRead::SetUp(void) { TestBase::SetUp(AMDSMI_INIT_AMD_NICS); }

void TestNicTelemetryRead::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestNicTelemetryRead::DisplayResults(void) const { TestBase::DisplayResults(); }

void TestNicTelemetryRead::Close() { TestBase::Close(); }

void TestNicTelemetryRead::Run(void) {
  TestBase::Run();
  if (setup_failed_) {
    IF_VERB(STANDARD) { std::cout << "** SetUp Failed for this test. Skipping.**\n"; }
    return;
  }

  std::vector<amdsmi_processor_handle> nics = nic_handles();
  if (nics.empty()) {
    IF_VERB(STANDARD) { std::cout << "\t**No AMD NIC present. Skipping.**\n"; }
    return;
  }
  amdsmi_processor_handle nic = nics[0];

  // ── amdsmi_get_nic_telemetry: succeeds and returns sane values ──────────
  {
    amdsmi_nic_telemetry_t telem;
    ASSERT_EQ(amdsmi_get_nic_telemetry(nic, &telem), AMDSMI_STATUS_SUCCESS);

    // Health state is one of the defined enum values.
    EXPECT_LE(telem.health.state, AMDSMI_NIC_HEALTH_UNSUPPORTED);

    // reporter is NUL-terminated within its fixed buffer.
    EXPECT_LT(strnlen(telem.health.reporter, sizeof(telem.health.reporter)),
              sizeof(telem.health.reporter));

    // Each temperature is either a plausible reading or the unsupported reserved value.
    for (uint16_t t : {telem.temperature.asic_temp_c, telem.temperature.transceiver_temp_c,
                       telem.temperature.board_temp_c}) {
      EXPECT_TRUE((t == UINT16_MAX) || (t <= 150)) << "implausible temperature: " << t;
    }

    // Bind every remaining public field by name so a field dropped or renamed
    // from the copy bridge fails to compile here (companion to the sizeof guard).
    EXPECT_TRUE((telem.port_split.splittable == UINT8_MAX) || (telem.port_split.splittable <= 1))
        << "implausible splittable: " << static_cast<unsigned>(telem.port_split.splittable);
    const uint8_t split_count = telem.port_split.split_count;
    const uint32_t error_count = telem.health.error_count;  // any value is valid
    (void)split_count;
    (void)error_count;
  }

  // ── amdsmi_get_nic_fw_info: succeeds and is bounded ─────────────────────
  {
    amdsmi_nic_fw_info_t fw;
    const amdsmi_status_t fw_status = amdsmi_get_nic_fw_info(nic, &fw);
    if (fw_status == AMDSMI_STATUS_NOT_SUPPORTED) {
      // Firmware versions come over devlink; not every NIC or build exposes them.
      IF_VERB(STANDARD) { std::cout << "\t**NIC firmware info not supported. Skipping.**\n"; }
    } else {
      ASSERT_EQ(fw_status, AMDSMI_STATUS_SUCCESS);
      EXPECT_LE(fw.num_fw, static_cast<uint32_t>(AMDSMI_MAX_NIC_FW));
    }
  }

  // ── amdsmi_get_nic_hwmon_sensors: count call, fetch call, short buffer ──
  {
    uint32_t count = 0;
    ASSERT_EQ(amdsmi_get_nic_hwmon_sensors(nic, &count, nullptr), AMDSMI_STATUS_SUCCESS);
    if (count == 0) {
      // A NIC without an hwmon node (for example a UALoE endpoint) reports no sensors.
      IF_VERB(STANDARD) { std::cout << "\t**NIC exposes no hwmon sensors.**\n"; }
    } else {
      std::vector<amdsmi_nic_sensor_t> sensors(count);
      uint32_t fetched = count;
      ASSERT_EQ(amdsmi_get_nic_hwmon_sensors(nic, &fetched, sensors.data()), AMDSMI_STATUS_SUCCESS);
      EXPECT_EQ(fetched, count);

      for (const amdsmi_nic_sensor_t& sensor : sensors) {
        EXPECT_TRUE((sensor.category == AMDSMI_METRIC_CATEGORY_TEMPERATURE) ||
                    (sensor.category == AMDSMI_METRIC_CATEGORY_POWER) ||
                    (sensor.category == AMDSMI_METRIC_CATEGORY_VOLTAGE))
            << "unexpected category " << sensor.category;
        EXPECT_NE(sensor.name, AMDSMI_NIC_SENSOR_NAME_UNKNOWN);
        EXPECT_LT(strnlen(sensor.label, sizeof(sensor.label)), sizeof(sensor.label));
        const bool is_temperature_input = (sensor.category == AMDSMI_METRIC_CATEGORY_TEMPERATURE) &&
                                          (sensor.name == AMDSMI_NIC_SENSOR_NAME_INPUT);
        if (is_temperature_input) {
          // milli-degrees Celsius: a plausible silicon reading, not a value in whole degrees
          EXPECT_GT(sensor.value, -50000) << sensor.label;
          EXPECT_LT(sensor.value, 150000) << sensor.label;
        }
      }

      // The telemetry temperatures come from the same hwmon labels, so they must agree with the
      // labelled sensors. Skipped for a NIC that labels no die or board sensor.
      amdsmi_nic_telemetry_t telem;
      ASSERT_EQ(amdsmi_get_nic_telemetry(nic, &telem), AMDSMI_STATUS_SUCCESS);
      constexpr int kMaxDriftC = 3;  // the two calls read the sensor at different moments
      auto expect_matches_labelled_input = [&](uint16_t telemetry_temp_c, const char* word) {
        if (telemetry_temp_c == UINT16_MAX) {
          return;
        }
        for (const amdsmi_nic_sensor_t& sensor : sensors) {
          std::string label(sensor.label);
          std::transform(label.begin(), label.end(), label.begin(),
                         [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
          const bool is_labelled_input = (sensor.category == AMDSMI_METRIC_CATEGORY_TEMPERATURE) &&
                                         (sensor.name == AMDSMI_NIC_SENSOR_NAME_INPUT) &&
                                         (label.find(word) != std::string::npos);
          if (is_labelled_input) {
            const int sensor_temp_c = static_cast<int>(sensor.value / 1000);
            EXPECT_LE(std::abs(static_cast<int>(telemetry_temp_c) - sensor_temp_c), kMaxDriftC)
                << word << " telemetry " << telemetry_temp_c << " C vs sensor '" << sensor.label
                << "' " << sensor_temp_c << " C";
            return;
          }
        }
      };
      expect_matches_labelled_input(telem.temperature.asic_temp_c, "die");
      expect_matches_labelled_input(telem.temperature.board_temp_c, "board");

      if (count > 1) {
        amdsmi_nic_sensor_t one;
        uint32_t too_small = 1;
        EXPECT_EQ(amdsmi_get_nic_hwmon_sensors(nic, &too_small, &one),
                  AMDSMI_STATUS_INSUFFICIENT_SIZE);
        EXPECT_EQ(too_small, count);
      }
    }
  }

  // ── null output pointers are rejected ───────────────────────────────────
  {
    EXPECT_EQ(amdsmi_get_nic_telemetry(nic, nullptr), AMDSMI_STATUS_INVAL);
    EXPECT_EQ(amdsmi_get_nic_fw_info(nic, nullptr), AMDSMI_STATUS_INVAL);
    EXPECT_EQ(amdsmi_get_nic_hwmon_sensors(nic, nullptr, nullptr), AMDSMI_STATUS_INVAL);
  }
}
