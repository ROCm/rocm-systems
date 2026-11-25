// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AILITOOLS-299: hwmon temperature and voltage lookups that used to throw std::out_of_range.
// TRY/CATCH turned each throw into AMDSMI_STATUS_INTERNAL_EXCEPTION, or, during discovery, failed
// the whole device:
//   * amdsmi_get_temp_metric() with a type that is not a key in Monitor's temp map. amdsmi pads
//     AMDSMI_TEMPERATURE_TYPE_BASEBOARD_LAST to 249 while rsmi aliases it to 222, and the value is
//     forwarded unchanged.
//   * A metric value outside amdsmi_temperature_metric_t / amdsmi_voltage_metric_t.
//   * A hwmon label the library does not recognize (temp<N>_label, in<N>_label).
//   * A labelled temp<N> whose index falls between the seeded ranges (8..99).
//
// A fake hwmon node under a fake sysfs tree stands in for the GPU, so this runs with no hardware.
// It mirrors real boards: temp1..temp3 carry labels and only in0 does.

#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_drm.h"
#include "amd_smi/impl/amd_smi_gpu_device.h"
#include "rocm_smi/rocm_smi_common.h"
#include "rocm_smi/rocm_smi_device.h"
#include "rocm_smi/rocm_smi_main.h"
#include "rocm_smi/rocm_smi_monitor.h"
#include "test_common.h"
#include "test_fixture_utils.h"

namespace {

constexpr int64_t kEdgeMilliC = 45000;
constexpr int64_t kVddgfxMilliV = 800;

// One public metric: its name for the log, the hwmon file it reads and the value written there.
struct MetricFile {
  const char* name;
  const char* file;
  int64_t value;
};

// A fake GPU with an hwmon node, registered with RocmSMI and AMDSmiSystem so the public API can
// reach it. `extra_files` are added to the hwmon directory, e.g. an unrecognized label.
class FakeHwmonGpu {
 public:
  explicit FakeHwmonGpu(const std::map<std::string, std::string>& extra_files = {})
      : device_(tree_.root() + "/card0"),
        gpu_device_(device_.index(), "card0", amdsmi_bdf_t{}, drm_),
        registration_(&gpu_device_) {
    const std::string hwmon_rel = "card0/device/hwmon/hwmon0/";
    tree_.WriteFile(hwmon_rel + "name", "amdgpu\n");
    tree_.WriteFile(hwmon_rel + "temp1_label", "edge\n");
    tree_.WriteFile(hwmon_rel + "temp1_input", std::to_string(kEdgeMilliC) + "\n");
    tree_.WriteFile(hwmon_rel + "temp2_label", "junction\n");
    tree_.WriteFile(hwmon_rel + "temp3_label", "mem\n");
    tree_.WriteFile(hwmon_rel + "in0_label", "vddgfx\n");
    tree_.WriteFile(hwmon_rel + "in0_input", std::to_string(kVddgfxMilliV) + "\n");
    for (const auto& [name, contents] : extra_files) {
      tree_.WriteFile(hwmon_rel + name, contents);
    }

    // The same three steps device discovery runs; none of them may throw.
    auto monitor = std::make_shared<amd::smi::Monitor>(tree_.root() + "/" + hwmon_rel, nullptr);
    EXPECT_NO_THROW({
      monitor->setTempSensorLabelMap();
      monitor->setVoltSensorLabelMap();
      SupportedFuncMap supported_funcs;
      monitor->fillSupportedFuncs(&supported_funcs);
    });
    amd::smi::RocmSMI::getInstance().devices()[device_.index()]->set_monitor(monitor);
  }

  amdsmi_processor_handle handle() const { return registration_.handle(); }

 private:
  ScopedAmdsmiInit init_;
  FakeSysfsTree tree_;
  ScopedRocmSmiDevice device_;
  amd::smi::AMDSmiDrm drm_;
  amd::smi::AMDSmiGPUDevice gpu_device_;
  amd::smi::testing::ScopedProcessorRegistration registration_;
};

// Each helper calls the API, prints it the way the other amdsmi tests do (a "### func(desc)"
// header, then the status against `expected`) and returns the status for the caller to assert on.
amdsmi_status_t TempMetric(const FakeHwmonGpu& gpu, uint32_t type, uint32_t metric, int64_t* value,
                           amdsmi_status_t expected, std::string_view desc,
                           const char* file = __builtin_FILE(), uint32_t line = __builtin_LINE()) {
  DISPLAY_AMDSMI_API("amdsmi_get_temp_metric", desc, IsVerbose());
  const amdsmi_status_t ret =
      amdsmi_get_temp_metric(gpu.handle(), static_cast<amdsmi_temperature_type_t>(type),
                             static_cast<amdsmi_temperature_metric_t>(metric), value);
  DISPLAY_AMDSMI_STATUS(IsVerbose(), file, line, ret, expected);
  return ret;
}

amdsmi_status_t VoltMetric(const FakeHwmonGpu& gpu, uint32_t type, uint32_t metric, int64_t* value,
                           amdsmi_status_t expected, std::string_view desc,
                           const char* file = __builtin_FILE(), uint32_t line = __builtin_LINE()) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_volt_metric", desc, IsVerbose());
  const amdsmi_status_t ret =
      amdsmi_get_gpu_volt_metric(gpu.handle(), static_cast<amdsmi_voltage_type_t>(type),
                                 static_cast<amdsmi_voltage_metric_t>(metric), value);
  DISPLAY_AMDSMI_STATUS(IsVerbose(), file, line, ret, expected);
  return ret;
}

// After discovery met something odd, the sensors that are fine must still read.
void ExpectGoodSensorsStillRead(const FakeHwmonGpu& gpu) {
  int64_t temperature = 0;
  ASSERT_EQ(TempMetric(gpu, AMDSMI_TEMPERATURE_TYPE_EDGE, AMDSMI_TEMP_CURRENT, &temperature,
                       AMDSMI_STATUS_SUCCESS, "type=EDGE, metric=CURRENT"),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(temperature, kEdgeMilliC / 1000);

  int64_t voltage = 0;
  ASSERT_EQ(VoltMetric(gpu, AMDSMI_VOLT_TYPE_VDDGFX, AMDSMI_VOLT_CURRENT, &voltage,
                       AMDSMI_STATUS_SUCCESS, "type=VDDGFX, metric=CURRENT"),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(voltage, kVddgfxMilliV);
}

}  // namespace

// The padded sentinel is not a map key; it reads as unsupported, not as an exception.
TEST(GpuUnit, HWMON_SENSOR_LOOKUP__TempMetricPaddedBaseboardLastIsNotSupported) {
  FakeHwmonGpu gpu;

  int64_t temperature = 0;
  EXPECT_EQ(
      TempMetric(gpu, AMDSMI_TEMPERATURE_TYPE_BASEBOARD_LAST, AMDSMI_TEMP_CURRENT, &temperature,
                 AMDSMI_STATUS_NOT_SUPPORTED, "type=BASEBOARD_LAST, metric=CURRENT"),
      AMDSMI_STATUS_NOT_SUPPORTED);
}

// Control: a real baseboard type is a key (seeded, no hwmon label) and was already NOT_SUPPORTED.
TEST(GpuUnit, HWMON_SENSOR_LOOKUP__TempMetricSeededButUnlabeledTypeIsNotSupported) {
  FakeHwmonGpu gpu;

  int64_t temperature = 0;
  EXPECT_EQ(
      TempMetric(gpu, AMDSMI_TEMPERATURE_TYPE_BASEBOARD_IBC, AMDSMI_TEMP_CURRENT, &temperature,
                 AMDSMI_STATUS_NOT_SUPPORTED, "type=BASEBOARD_IBC, metric=CURRENT"),
      AMDSMI_STATUS_NOT_SUPPORTED);
}

// Control: a labelled sensor still reads hwmon, in degrees C.
TEST(GpuUnit, HWMON_SENSOR_LOOKUP__TempMetricLabeledEdgeReadsHwmon) {
  FakeHwmonGpu gpu;

  ExpectGoodSensorsStillRead(gpu);
}

// No temperature type, including every group sentinel and values outside the enum, may surface
// as an internal exception.
TEST(GpuUnit, HWMON_SENSOR_LOOKUP__TempMetricNeverReportsInternalException) {
  FakeHwmonGpu gpu;

  DISPLAY_AMDSMI_API("amdsmi_get_temp_metric",
                     "every type 0..AMDSMI_TEMPERATURE_TYPE__MAX + 1, metric=CURRENT", IsVerbose());
  for (uint32_t type = 0; type <= static_cast<uint32_t>(AMDSMI_TEMPERATURE_TYPE__MAX) + 1; ++type) {
    SCOPED_TRACE("temperature type " + std::to_string(type));
    int64_t temperature = 0;
    EXPECT_NE(amdsmi_get_temp_metric(gpu.handle(), static_cast<amdsmi_temperature_type_t>(type),
                                     AMDSMI_TEMP_CURRENT, &temperature),
              AMDSMI_STATUS_INTERNAL_EXCEPTION);
  }
  for (const uint32_t type : {1000U, 0xFFFFFFFFU}) {
    int64_t temperature = 0;
    EXPECT_NE(TempMetric(gpu, type, AMDSMI_TEMP_CURRENT, &temperature, AMDSMI_STATUS_NOT_SUPPORTED,
                         "type=" + std::to_string(type) + ", metric=CURRENT"),
              AMDSMI_STATUS_INTERNAL_EXCEPTION);
  }
}

// The first value past the public enum is a bad argument. It used to reach a monitor-file lookup
// with no valid entry and throw. Larger values are not tried: loading an out-of-range enum is
// undefined behavior that UBSan reports. There is no voltage counterpart, since
// AMDSMI_VOLT_LAST + 1 is 8, outside that enum's 3-bit range.
TEST(GpuUnit, HWMON_SENSOR_LOOKUP__TempMetricPastLastMetricIsInvalidArgs) {
  FakeHwmonGpu gpu;

  const uint32_t metric = static_cast<uint32_t>(AMDSMI_TEMP_LAST) + 1;
  int64_t temperature = 0;
  EXPECT_EQ(TempMetric(gpu, AMDSMI_TEMPERATURE_TYPE_EDGE, metric, &temperature, AMDSMI_STATUS_INVAL,
                       "type=EDGE, metric=" + std::to_string(metric)),
            AMDSMI_STATUS_INVAL);
}

// amdgpu exposes the ASIC shutdown limit as temp#_emergency, so SHUTDOWN reads the same file.
TEST(GpuUnit, HWMON_SENSOR_LOOKUP__TempMetricShutdownReadsEmergency) {
  FakeHwmonGpu gpu(std::map<std::string, std::string>{{"temp1_emergency", "105000\n"}});

  int64_t shutdown = 0;
  EXPECT_EQ(TempMetric(gpu, AMDSMI_TEMPERATURE_TYPE_EDGE, AMDSMI_TEMP_SHUTDOWN, &shutdown,
                       AMDSMI_STATUS_SUCCESS, "type=EDGE, metric=SHUTDOWN"),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(shutdown, 105);

  int64_t emergency = 0;
  EXPECT_EQ(TempMetric(gpu, AMDSMI_TEMPERATURE_TYPE_EDGE, AMDSMI_TEMP_EMERGENCY, &emergency,
                       AMDSMI_STATUS_SUCCESS, "type=EDGE, metric=EMERGENCY"),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(emergency, shutdown);
}

// Every public temperature metric reads its own hwmon file. A metric added to the enum without a
// file here fails the lookup. SHUTDOWN shares the emergency file.
TEST(GpuUnit, HWMON_SENSOR_LOOKUP__TempMetricEveryPublicMetricReadsItsFile) {
  const std::map<uint32_t, MetricFile> kMetricFiles = {
      {AMDSMI_TEMP_CURRENT, {"CURRENT", "temp1_input", kEdgeMilliC}},
      {AMDSMI_TEMP_MAX, {"MAX", "temp1_max", 90000}},
      {AMDSMI_TEMP_MIN, {"MIN", "temp1_min", 10000}},
      {AMDSMI_TEMP_MAX_HYST, {"MAX_HYST", "temp1_max_hyst", 85000}},
      {AMDSMI_TEMP_MIN_HYST, {"MIN_HYST", "temp1_min_hyst", 15000}},
      {AMDSMI_TEMP_CRITICAL, {"CRITICAL", "temp1_crit", 100000}},
      {AMDSMI_TEMP_CRITICAL_HYST, {"CRITICAL_HYST", "temp1_crit_hyst", 95000}},
      {AMDSMI_TEMP_EMERGENCY, {"EMERGENCY", "temp1_emergency", 105000}},
      {AMDSMI_TEMP_EMERGENCY_HYST, {"EMERGENCY_HYST", "temp1_emergency_hyst", 102000}},
      {AMDSMI_TEMP_CRIT_MIN, {"CRIT_MIN", "temp1_lcrit", 5000}},
      {AMDSMI_TEMP_CRIT_MIN_HYST, {"CRIT_MIN_HYST", "temp1_lcrit_hyst", 6000}},
      {AMDSMI_TEMP_OFFSET, {"OFFSET", "temp1_offset", 2000}},
      {AMDSMI_TEMP_LOWEST, {"LOWEST", "temp1_lowest", 20000}},
      {AMDSMI_TEMP_HIGHEST, {"HIGHEST", "temp1_highest", 70000}},
      {AMDSMI_TEMP_SHUTDOWN, {"SHUTDOWN", "temp1_emergency", 105000}},
  };
  std::map<std::string, std::string> files;
  for (const auto& [metric, metric_file] : kMetricFiles) {
    files[metric_file.file] = std::to_string(metric_file.value) + "\n";
  }
  FakeHwmonGpu gpu(files);

  for (uint32_t metric = AMDSMI_TEMP_FIRST; metric <= AMDSMI_TEMP_LAST; ++metric) {
    const auto expected = kMetricFiles.find(metric);
    ASSERT_NE(expected, kMetricFiles.end()) << "temperature metric " << metric << " has no file";
    SCOPED_TRACE(std::string("temperature metric ") + expected->second.name);
    int64_t temperature = 0;
    EXPECT_EQ(
        TempMetric(gpu, AMDSMI_TEMPERATURE_TYPE_EDGE, metric, &temperature, AMDSMI_STATUS_SUCCESS,
                   std::string("type=EDGE, metric=") + expected->second.name),
        AMDSMI_STATUS_SUCCESS);
    EXPECT_EQ(temperature, expected->second.value / 1000);
  }
}

// Control: VDDBOARD has no in<N>_label, so it is not a key in the voltage map and was already
// NOT_SUPPORTED.
TEST(GpuUnit, HWMON_SENSOR_LOOKUP__VoltMetricUnlabeledTypeIsNotSupported) {
  FakeHwmonGpu gpu;

  int64_t voltage = 0;
  EXPECT_EQ(VoltMetric(gpu, AMDSMI_VOLT_TYPE_VDDBOARD, AMDSMI_VOLT_CURRENT, &voltage,
                       AMDSMI_STATUS_NOT_SUPPORTED, "type=VDDBOARD, metric=CURRENT"),
            AMDSMI_STATUS_NOT_SUPPORTED);
}

// Every public voltage metric reads its own hwmon file, so none can fall to the rsmi default
// branch for an unknown metric. A metric added to the enum without a file here fails the lookup.
TEST(GpuUnit, HWMON_SENSOR_LOOKUP__VoltMetricEveryPublicMetricReadsItsFile) {
  const std::map<uint32_t, MetricFile> kMetricFiles = {
      {AMDSMI_VOLT_CURRENT, {"CURRENT", "in0_input", kVddgfxMilliV}},
      {AMDSMI_VOLT_MAX, {"MAX", "in0_max", 900}},
      {AMDSMI_VOLT_MIN_CRIT, {"MIN_CRIT", "in0_lcrit", 600}},
      {AMDSMI_VOLT_MIN, {"MIN", "in0_min", 700}},
      {AMDSMI_VOLT_MAX_CRIT, {"MAX_CRIT", "in0_crit", 1000}},
      {AMDSMI_VOLT_AVERAGE, {"AVERAGE", "in0_average", 810}},
      {AMDSMI_VOLT_LOWEST, {"LOWEST", "in0_lowest", 650}},
      {AMDSMI_VOLT_HIGHEST, {"HIGHEST", "in0_highest", 950}},
  };
  std::map<std::string, std::string> files;
  for (const auto& [metric, metric_file] : kMetricFiles) {
    files[metric_file.file] = std::to_string(metric_file.value) + "\n";
  }
  FakeHwmonGpu gpu(files);

  for (uint32_t metric = AMDSMI_VOLT_FIRST; metric <= AMDSMI_VOLT_LAST; ++metric) {
    const auto expected = kMetricFiles.find(metric);
    ASSERT_NE(expected, kMetricFiles.end()) << "voltage metric " << metric << " has no file";
    SCOPED_TRACE(std::string("voltage metric ") + expected->second.name);
    int64_t voltage = 0;
    EXPECT_EQ(VoltMetric(gpu, AMDSMI_VOLT_TYPE_VDDGFX, metric, &voltage, AMDSMI_STATUS_SUCCESS,
                         std::string("type=VDDGFX, metric=") + expected->second.name),
              AMDSMI_STATUS_SUCCESS);
    EXPECT_EQ(voltage, expected->second.value);
  }
}

// A temp label the library does not know is ignored; it used to throw out of discovery.
TEST(GpuUnit, HWMON_SENSOR_LOOKUP__UnrecognizedTempLabelDoesNotBreakDiscovery) {
  FakeHwmonGpu gpu({{"temp4_label", "hbm\n"}, {"temp4_input", "50000\n"}});

  ExpectGoodSensorsStillRead(gpu);
}

TEST(GpuUnit, HWMON_SENSOR_LOOKUP__UnrecognizedVoltLabelDoesNotBreakDiscovery) {
  FakeHwmonGpu gpu({{"in2_label", "vddnb\n"}, {"in2_input", "1\n"}});

  ExpectGoodSensorsStillRead(gpu);
}

// temp8 sits between the seeded ranges (GENERAL_LAST is 7, GPUBOARD_NODE_FIRST is 100), so it
// was never added to the index-to-type map and the lookup during discovery threw.
TEST(GpuUnit, HWMON_SENSOR_LOOKUP__LabeledTempOutsideSeededRangesDoesNotBreakDiscovery) {
  FakeHwmonGpu gpu({{"temp8_label", "edge\n"}, {"temp8_input", "50000\n"}});

  ExpectGoodSensorsStillRead(gpu);
}
