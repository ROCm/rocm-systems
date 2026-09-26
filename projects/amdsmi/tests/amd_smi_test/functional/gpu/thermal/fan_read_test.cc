// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "fan_read.h"

#include <gtest/gtest.h>
#include <sys/stat.h>

#include <cstdint>
#include <iostream>

#include "amd_smi/amdsmi.h"
#include "test_common.h"

TestFanRead::TestFanRead() : TestBase() {
  set_title("AMDSMI Fan Read Test");
  set_description(
      "The Fan Read tests verifies that the fan monitors can be "
      "read properly.");
}

TestFanRead::~TestFanRead(void) {}

void TestFanRead::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestFanRead::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestFanRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestFanRead::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

void TestFanRead::Run(void) {
  uint64_t val_ui64;
  amdsmi_status_t err;
  int64_t val_i64;

  TestBase::Run();
  PRINT_VERBOSITY();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }
  for (uint32_t x = 0; x < num_iterations(); ++x) {
    for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
      PrintDeviceHeader(processor_handles_[i]);

      DISPLAY_AMDSMI_API("amdsmi_get_gpu_fan_speed", "gpu=" + std::to_string(i), VERB(STANDARD));
      err = amdsmi_get_gpu_fan_speed(processor_handles_[i], 0, &val_i64);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);
      if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
        ASSERT_EQ(err, AMDSMI_STATUS_NOT_SUPPORTED);
        // Verify api support checking functionality is working
        DISPLAY_AMDSMI_API("amdsmi_get_gpu_fan_speed", "gpu=" + std::to_string(i), VERB(STANDARD));
        err = amdsmi_get_gpu_fan_speed(processor_handles_[i], 0, nullptr);
        DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_NOT_SUPPORTED);
        ASSERT_EQ(err, AMDSMI_STATUS_NOT_SUPPORTED);
        continue;
      } else {
        CHK_ERR_ASRT(err)
        IF_VERB(STANDARD) { std::cout << "\t**Current Fan Speed: " << val_i64 << std::endl; }
      }

      // Verify api support checking functionality is working
      DISPLAY_AMDSMI_API("amdsmi_get_gpu_fan_speed", "gpu=" + std::to_string(i), VERB(STANDARD));
      err = amdsmi_get_gpu_fan_speed(processor_handles_[i], 0, nullptr);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
      ASSERT_EQ(err, AMDSMI_STATUS_INVAL);

      DISPLAY_AMDSMI_API("amdsmi_get_gpu_fan_speed_max", "gpu=" + std::to_string(i),
                         VERB(STANDARD));
      err = amdsmi_get_gpu_fan_speed_max(processor_handles_[i], 0, &val_ui64);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) {
        std::cout << "\t**Current fan RPMs: ";
        std::cout << static_cast<float>(val_i64) / static_cast<float>(val_ui64) * 100;
        std::cout << "% (" << val_i64 << "/" << val_ui64 << ")" << std::endl;
      }
      // Verify api support checking functionality is working
      DISPLAY_AMDSMI_API("amdsmi_get_gpu_fan_speed_max", "gpu=" + std::to_string(i),
                         VERB(STANDARD));
      err = amdsmi_get_gpu_fan_speed_max(processor_handles_[i], 0, nullptr);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
      ASSERT_EQ(err, AMDSMI_STATUS_INVAL);

      DISPLAY_AMDSMI_API("amdsmi_get_gpu_fan_rpms", "gpu=" + std::to_string(i), VERB(STANDARD));
      err = amdsmi_get_gpu_fan_rpms(processor_handles_[i], 0, &val_i64);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);
      struct stat dxg_st{};
      bool on_wsl = (stat("/dev/dxg", &dxg_st) == 0);
      if (err == AMDSMI_STATUS_NOT_SUPPORTED && on_wsl) {
        // fan_rpms dispatches to WSLGPUBackend::get_fan_rpms, a hard NOT_SUPPORTED
        // stub (no RPM telemetry over WDDM); fan_speed/fan_speed_max are backed
        // by real rocdxg_smi metrics.
        DISPLAY_AMDSMI_API("amdsmi_get_gpu_fan_rpms", "gpu=" + std::to_string(i), VERB(STANDARD));
        err = amdsmi_get_gpu_fan_rpms(processor_handles_[i], 0, nullptr);
        DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_NOT_SUPPORTED);
        ASSERT_EQ(err, AMDSMI_STATUS_NOT_SUPPORTED);
      } else {
        CHK_ERR_ASRT(err)
        IF_VERB(STANDARD) {
          std::cout << "\t**Current fan RPMs: ";
          std::cout << val_i64 << std::endl;
        }
        // Verify api support checking functionality is working
        DISPLAY_AMDSMI_API("amdsmi_get_gpu_fan_rpms", "gpu=" + std::to_string(i), VERB(STANDARD));
        err = amdsmi_get_gpu_fan_rpms(processor_handles_[i], 0, nullptr);
        DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
        ASSERT_EQ(err, AMDSMI_STATUS_INVAL);
      }
    }
  }
}
