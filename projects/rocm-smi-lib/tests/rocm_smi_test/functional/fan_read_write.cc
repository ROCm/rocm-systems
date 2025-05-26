/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2019, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */

#include <stdint.h>
#include <stddef.h>

#include <iostream>

#include "gtest/gtest.h"
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi_test/functional/fan_read_write.h"
#include "rocm_smi_test/test_common.h"

TestFanReadWrite::TestFanReadWrite() : TestBase() {
  set_title("RSMI Fan Read/Write Test");
  set_description("The Fan Read tests verifies that the fan monitors can be "
                  "read and controlled properly.");
}

TestFanReadWrite::~TestFanReadWrite(void) {
}

void TestFanReadWrite::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestFanReadWrite::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestFanReadWrite::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestFanReadWrite::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // rsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}


void TestFanReadWrite::Run(void) {
  rsmi_status_t ret;
  int64_t orig_speed;
  int64_t new_speed;
  int64_t cur_speed;
  uint64_t max_speed;

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t dv_ind = 0; dv_ind < num_monitor_devs(); ++dv_ind) {
    PrintDeviceHeader(dv_ind);

    ret = rsmi_dev_fan_speed_get(dv_ind, 0, &orig_speed);
    if (ret == RSMI_STATUS_NOT_SUPPORTED) {
       IF_VERB(STANDARD) {
          std::cout << "\t**" <<  ": " <<
                             "Not supported on this machine" << std::endl;
        }
        return;
    } else {
        CHK_ERR_ASRT(ret)
    }
    IF_VERB(STANDARD) {
      std::cout << "Original fan speed: " << orig_speed << std::endl;
    }

    if (orig_speed == 0) {
      std::cout << "***System fan speed value is 0. Skip fan test." <<
                                                                    std::endl;
      return;
    }

    ret = rsmi_dev_fan_speed_max_get(dv_ind, 0, &max_speed);
    CHK_ERR_ASRT(ret)

    new_speed = 1.1 * orig_speed;

    if (new_speed > static_cast<int64_t>(max_speed)) {
      std::cout <<
      "***System fan speed value is close to max. Will not adjust upward." <<
                                                                     std::endl;
      continue;
    }

    IF_VERB(STANDARD) {
      std::cout << "Setting fan speed to " << new_speed << std::endl;
    }

    ret = rsmi_dev_fan_speed_set(dv_ind, 0, new_speed);
    if (ret == RSMI_STATUS_NOT_SUPPORTED) {
      std::cout << "***System fan set is not supported." << std::endl;
      continue;
    }
    CHK_ERR_ASRT(ret)

    sleep(4);

    ret = rsmi_dev_fan_speed_get(dv_ind, 0, &cur_speed);
    CHK_ERR_ASRT(ret)

    IF_VERB(STANDARD) {
      std::cout << "New fan speed: " << cur_speed << std::endl;
    }

    // EXPECT_TRUE((cur_speed > 0.95 * new_speed &&
    //                cur_speed < 1.1 * new_speed) ||
    //                    cur_speed > 0.95 * RSMI_MAX_FAN_SPEED);
    IF_VERB(STANDARD) {
      if (!((cur_speed > 0.95 * new_speed && cur_speed < 1.1 * new_speed) ||
                                (cur_speed > 0.95 * RSMI_MAX_FAN_SPEED))) {
        std::cout << "WARNING: Fan speed is not within the expected range!" <<
                                                                      std::endl;
      }
    }

    IF_VERB(STANDARD) {
      std::cout << "Resetting fan control to auto..." << std::endl;
    }

    ret = rsmi_dev_fan_reset(dv_ind, 0);
    CHK_ERR_ASRT(ret)

    sleep(3);

    ret = rsmi_dev_fan_speed_get(dv_ind, 0, &cur_speed);
    CHK_ERR_ASRT(ret)

    IF_VERB(STANDARD) {
      std::cout << "End fan speed: " << cur_speed << std::endl;
    }
  }
}
