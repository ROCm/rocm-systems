// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "fan_read_write.h"

#include <gtest/gtest.h>
#include <unistd.h>

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "amd_smi/amdsmi.h"
#include "rocm_smi/rocm_smi_utils.h"
#include "test_common.h"

namespace {

// amdsmi_set_gpu_fan_speed validates against the gpu_od OD_RANGE on GPUs that
// expose it. That range is a different unit space than pwm1/pwm1_max, so it
// cannot be derived from amdsmi_get_gpu_fan_speed_max().
bool GetGpuOdFanSetRange(amdsmi_processor_handle handle, uint64_t* od_min, uint64_t* od_max) {
  amdsmi_bdf_t bdf;
  if (amdsmi_get_gpu_device_bdf(handle, &bdf) != AMDSMI_STATUS_SUCCESS) {
    return false;
  }

  std::ostringstream path;
  path << "/sys/bus/pci/devices/" << std::hex << std::setfill('0') << std::setw(4)
       << static_cast<uint64_t>(bdf.bdf.domain_number) << ":" << std::setw(2)
       << static_cast<uint64_t>(bdf.bdf.bus_number) << ":" << std::setw(2)
       << static_cast<uint64_t>(bdf.bdf.device_number) << "."
       << static_cast<uint64_t>(bdf.bdf.function_number) << "/gpu_od/fan_ctrl/fan_minimum_pwm";

  return amd::smi::ParseGpuOdFanRange(path.str(), od_min, od_max) == 0;
}

}  // namespace

TestFanReadWrite::TestFanReadWrite() : TestBase() {
  set_title("AMDSMI Fan Read/Write Test");
  set_description(
      "The Fan Read tests verifies that the fan monitors can be "
      "read and controlled properly.");
}

TestFanReadWrite::~TestFanReadWrite(void) {}

void TestFanReadWrite::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestFanReadWrite::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestFanReadWrite::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestFanReadWrite::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

void TestFanReadWrite::Run(void) {
  amdsmi_status_t ret;
  int64_t orig_speed;
  int64_t new_speed;
  int64_t cur_speed;
  uint64_t max_speed;

  TestBase::Run();
  PRINT_VERBOSITY();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t dv_ind = 0; dv_ind < num_monitor_devs(); ++dv_ind) {
    PrintDeviceHeader(processor_handles_[dv_ind]);

    bool can_read_speed = false;

    // Try read original fan speed
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_fan_speed", "gpu=" + std::to_string(dv_ind), VERB(STANDARD));
    ret = amdsmi_get_gpu_fan_speed(processor_handles_[dv_ind], 0, &orig_speed);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
      continue;
    }
    // Fan speed read may not be supported on some GPUs
    if (ret != AMDSMI_STATUS_SUCCESS) {
      IF_VERB(STANDARD) {
        std::cout << "Fan speed read unavailable. "
                  << "Testing set/reset only." << std::endl;
      }
      orig_speed = 0;
    } else {
      can_read_speed = true;
    }
    IF_VERB(STANDARD) { std::cout << "Original fan speed: " << orig_speed << std::endl; }

    // Verify max fan speed returns a sensible value for both interfaces
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_fan_speed_max", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    ret = amdsmi_get_gpu_fan_speed_max(processor_handles_[dv_ind], 0, &max_speed);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    CHK_ERR_ASRT(ret)
    IF_VERB(STANDARD) { std::cout << "Max fan speed: " << max_speed << std::endl; }
    // max_speed always reports the hwmon pwm1_max ceiling, in the same unit
    // space as amdsmi_get_gpu_fan_speed()
    ASSERT_GT(max_speed, static_cast<uint64_t>(0));
    ASSERT_LE(max_speed, static_cast<uint64_t>(AMDSMI_MAX_FAN_SPEED));

    // The set path validates against the gpu_od OD_RANGE where available, which
    // max_speed does not describe.
    uint64_t od_min = 0;
    uint64_t od_max = 0;
    const bool has_gpu_od = GetGpuOdFanSetRange(processor_handles_[dv_ind], &od_min, &od_max);
    const int64_t set_min = has_gpu_od ? static_cast<int64_t>(od_min) : 0;
    const int64_t set_max =
        has_gpu_od ? static_cast<int64_t>(od_max) : static_cast<int64_t>(max_speed);
    IF_VERB(STANDARD) {
      std::cout << "Settable fan speed range: " << set_min << "-" << set_max << " ("
                << (has_gpu_od ? "gpu_od OD_RANGE" : "legacy hwmon") << ")" << std::endl;
    }
    if (set_max <= set_min) {
      std::cout << "***No usable fan speed range on this GPU. Skipping." << std::endl;
      continue;
    }

    if (!has_gpu_od && can_read_speed && orig_speed > 0) {
      // Fans are spinning — use a speed slightly above current for the test
      new_speed = static_cast<int64_t>(1.1F * static_cast<float>(orig_speed));

      if (new_speed > set_max) {
        std::cout << "***System fan speed value is close to max. Will not adjust upward."
                  << std::endl;
        continue;
      }
    } else {
      // orig_speed is not comparable to the settable range on gpu_od GPUs, so
      // drive the midpoint of the range the set path actually accepts
      new_speed = set_min + (set_max - set_min) / 2;
    }

    IF_VERB(STANDARD) { std::cout << "Setting fan speed to " << new_speed << std::endl; }

    DISPLAY_AMDSMI_API("amdsmi_set_gpu_fan_speed", "gpu=" + std::to_string(dv_ind), VERB(STANDARD));
    ret = amdsmi_set_gpu_fan_speed(processor_handles_[dv_ind], 0, static_cast<uint64_t>(new_speed));
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NO_PERM);

    if (ret == AMDSMI_STATUS_NO_PERM || ret == AMDSMI_STATUS_NOT_SUPPORTED) {
      std::cout << "\t**Set fan speed: Not supported or requires root/sudo. Skipping..."
                << std::endl;
      continue;
    }
    CHK_ERR_ASRT(ret)

    sleep(4);

    // Read back fan speed
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_fan_speed", "gpu=" + std::to_string(dv_ind), VERB(STANDARD));
    ret = amdsmi_get_gpu_fan_speed(processor_handles_[dv_ind], 0, &cur_speed);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    if (ret == AMDSMI_STATUS_SUCCESS) {
      IF_VERB(STANDARD) { std::cout << "New fan speed: " << cur_speed << std::endl; }

      // Readback is only comparable to new_speed in the hwmon unit space
      if (!has_gpu_od && can_read_speed && orig_speed > 0) {
        IF_VERB(STANDARD) {
          if (!((cur_speed > static_cast<int64_t>(0.80 * static_cast<double>(new_speed)) &&
                 cur_speed < static_cast<int64_t>(1.25 * static_cast<double>(new_speed))) ||
                (cur_speed > static_cast<int64_t>(0.80 * static_cast<double>(max_speed))))) {
            std::cout << "WARNING: Fan speed is not within the expected range!" << std::endl;
          }
        }
      }
    } else {
      IF_VERB(STANDARD) { std::cout << "Fan speed readback unavailable on this GPU." << std::endl; }
    }

    IF_VERB(STANDARD) { std::cout << "Resetting fan control to auto..." << std::endl; }

    DISPLAY_AMDSMI_API("amdsmi_reset_gpu_fan", "gpu=" + std::to_string(dv_ind), VERB(STANDARD));
    ret = amdsmi_reset_gpu_fan(processor_handles_[dv_ind], 0);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    CHK_ERR_ASRT(ret)

    sleep(3);

    DISPLAY_AMDSMI_API("amdsmi_get_gpu_fan_speed", "gpu=" + std::to_string(dv_ind), VERB(STANDARD));
    ret = amdsmi_get_gpu_fan_speed(processor_handles_[dv_ind], 0, &cur_speed);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    if (ret == AMDSMI_STATUS_SUCCESS) {
      IF_VERB(STANDARD) { std::cout << "End fan speed: " << cur_speed << std::endl; }
    } else {
      IF_VERB(STANDARD) {
        std::cout << "End fan speed readback unavailable on this GPU." << std::endl;
      }
    }
  }
}
