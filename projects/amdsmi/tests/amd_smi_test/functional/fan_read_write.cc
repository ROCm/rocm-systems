/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include "fan_read_write.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <iostream>

#include "../test_common.h"
#include "amd_smi/amdsmi.h"

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
    // Max speed must be > 0 and either 255 (legacy hwmon) or <= 100 (gpu_od OD_RANGE)
    ASSERT_GT(max_speed, static_cast<uint64_t>(0));
    ASSERT_LE(max_speed, static_cast<uint64_t>(AMDSMI_MAX_FAN_SPEED));

    if (can_read_speed && orig_speed > 0) {
      // Fans are spinning — use a speed slightly above current for the test
      new_speed = static_cast<int64_t>(1.1F * static_cast<float>(orig_speed));

      if (new_speed > static_cast<int64_t>(max_speed)) {
        std::cout << "***System fan speed value is close to max. Will not adjust upward."
                  << std::endl;
        continue;
      }
    } else {
      // Fans are idle or read is unavailable — use a safe mid-range value
      // that works for both legacy hwmon (0-255) and gpu_od (typically 20-100)
      new_speed = max_speed / 2;
    }

    IF_VERB(STANDARD) { std::cout << "Setting fan speed to " << new_speed << std::endl; }

    DISPLAY_AMDSMI_API("amdsmi_set_gpu_fan_speed", "gpu=" + std::to_string(dv_ind), VERB(STANDARD));
    ret = amdsmi_set_gpu_fan_speed(processor_handles_[dv_ind], 0, new_speed);
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

      // Only verify readback range when fans were originally spinning
      if (can_read_speed && orig_speed > 0) {
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
