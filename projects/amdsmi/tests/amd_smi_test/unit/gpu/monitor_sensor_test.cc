// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Unit tests for Monitor::getTempSensorIndex()/getVoltSensorIndex(), the sensor
// lookup behind amdsmi_get_temp_metric() and amdsmi_get_gpu_volt_metric(). A
// freshly constructed Monitor has empty label maps, which is the same state a
// device that exposes no such sensor leaves them in. No GPU required.
//
// These resolved the sensor with std::map::at, which threw std::out_of_range
// before the caller could test the result for the INVALID sentinel, making the
// NOT_SUPPORTED path unreachable and surfacing as "Exception caught: map::at".

#include <gtest/gtest.h>

#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_monitor.h"

namespace {

amd::smi::Monitor MakeUnpopulatedMonitor() {
  return amd::smi::Monitor("/nonexistent/hwmon", nullptr);
}

TEST(GpuUnit, TempSensorIndexReportsInvalidWhenUnmapped) {
  amd::smi::Monitor mon = MakeUnpopulatedMonitor();
  for (auto type : {RSMI_TEMP_TYPE_EDGE, RSMI_TEMP_TYPE_JUNCTION, RSMI_TEMP_TYPE_MEMORY}) {
    EXPECT_EQ(mon.getTempSensorIndex(type), static_cast<uint32_t>(RSMI_TEMP_TYPE_INVALID))
        << "sensor type " << type;
  }
}

TEST(GpuUnit, VoltSensorIndexReportsInvalidWhenUnmapped) {
  amd::smi::Monitor mon = MakeUnpopulatedMonitor();
  EXPECT_EQ(mon.getVoltSensorIndex(RSMI_VOLT_TYPE_VDDGFX),
            static_cast<uint32_t>(RSMI_VOLT_TYPE_INVALID));
}

// A type outside the enum reaches the same lookup; it must report the sentinel
// rather than escape as an exception across the C ABI.
TEST(GpuUnit, SensorIndexReportsInvalidForOutOfRangeType) {
  amd::smi::Monitor mon = MakeUnpopulatedMonitor();
  EXPECT_EQ(mon.getTempSensorIndex(static_cast<rsmi_temperature_type_t>(0xDEAD)),
            static_cast<uint32_t>(RSMI_TEMP_TYPE_INVALID));
  EXPECT_EQ(mon.getVoltSensorIndex(static_cast<rsmi_voltage_type_t>(0xDEAD)),
            static_cast<uint32_t>(RSMI_VOLT_TYPE_INVALID));
}

}  // namespace
