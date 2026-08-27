// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "volt_read.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <iostream>
#include <string>

#include "amd_smi/amdsmi.h"
#include "test_common.h"

TestVoltRead::TestVoltRead() : TestBase() {
  set_title("AMDSMI Volt Read Test");
  set_description(
      "The Voltage Read tests verifies that the voltage "
      "monitors can be read properly.");
}

TestVoltRead::~TestVoltRead(void) {}

void TestVoltRead::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestVoltRead::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestVoltRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestVoltRead::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

void TestVoltRead::Run(void) {
  amdsmi_status_t err;
  int64_t val_i64;

  TestBase::Run();
  PRINT_VERBOSITY();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  amdsmi_voltage_type_t type = AMDSMI_VOLT_TYPE_VDDGFX;

  for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
    PrintDeviceHeader(processor_handles_[i]);

    auto print_volt_metric = [&](amdsmi_voltage_metric_t met, std::string label) {
      DISPLAY_AMDSMI_API("amdsmi_get_gpu_volt_metric(label: " + label + ")",
                         "gpu=" + std::to_string(i), VERB(STANDARD));
      err = amdsmi_get_gpu_volt_metric(processor_handles_[i], type, met, &val_i64);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);

      if (err != AMDSMI_STATUS_SUCCESS) {
        if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
          ASSERT_EQ(err, AMDSMI_STATUS_NOT_SUPPORTED);
          IF_VERB(STANDARD) {
            std::cout << "\t**" << label << ": "
                      << "Not supported on this machine" << std::endl;
          }

          // Verify api support checking functionality is working
          DISPLAY_AMDSMI_API("amdsmi_get_gpu_volt_metric(nullptr check)",
                             "gpu=" + std::to_string(i) + ", label: " + label, VERB(STANDARD));
          err = amdsmi_get_gpu_volt_metric(processor_handles_[i], type, met, nullptr);
          DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err,
                                AMDSMI_STATUS_NOT_SUPPORTED);
          ASSERT_EQ(err, AMDSMI_STATUS_NOT_SUPPORTED);
          return;
        } else {
          CHK_ERR_ASRT(err)
        }
      }
      // Verify api support checking functionality is working
      DISPLAY_AMDSMI_API("amdsmi_get_gpu_volt_metric(nullptr check)",
                         "gpu=" + std::to_string(i) + ", label: " + label, VERB(STANDARD));
      err = amdsmi_get_gpu_volt_metric(processor_handles_[i], type, met, nullptr);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
      ASSERT_EQ(err, AMDSMI_STATUS_INVAL);

      IF_VERB(STANDARD) { std::cout << "\t**" << label << ": " << val_i64 << "mV" << std::endl; }
    };
    for (uint32_t i = AMDSMI_VOLT_TYPE_FIRST; i <= AMDSMI_VOLT_TYPE_LAST; ++i) {
      IF_VERB(STANDARD) {
        std::cout << "\t** **********"
                  << GetVoltSensorNameStr(static_cast<amdsmi_voltage_type_t>(i))
                  << " Voltage **********" << std::endl;
      }
      print_volt_metric(AMDSMI_VOLT_CURRENT, "Current Voltage");
      print_volt_metric(AMDSMI_VOLT_MAX, "Voltage max value");
      print_volt_metric(AMDSMI_VOLT_MIN, "Voltage min value");
      print_volt_metric(AMDSMI_VOLT_MAX_CRIT, "Voltage critical max value");
      print_volt_metric(AMDSMI_VOLT_MIN_CRIT, "Voltage critical min value");
      print_volt_metric(AMDSMI_VOLT_AVERAGE, "Voltage critical max value");
      print_volt_metric(AMDSMI_VOLT_LOWEST, "Historical minimum temperature");
      print_volt_metric(AMDSMI_VOLT_HIGHEST, "Historical maximum temperature");
    }
  }
}
