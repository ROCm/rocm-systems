/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2025, Advanced Micro Devices, Inc.
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
#include <bitset>
#include <string>
#include <algorithm>

#include "gtest/gtest.h"
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi_test/functional/power_cap_read_write.h"
#include "rocm_smi/rocm_smi_utils.h"
#include "rocm_smi_test/test_common.h"


TestPowerCapReadWrite::TestPowerCapReadWrite() : TestBase() {
  set_title("RSMI Power Cap Read/Write Test");
  set_description("The Power Cap tests verify that the power profile "
                             "settings can be read and written properly.");
}

TestPowerCapReadWrite::~TestPowerCapReadWrite(void) {
}

void TestPowerCapReadWrite::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestPowerCapReadWrite::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestPowerCapReadWrite::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestPowerCapReadWrite::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // rsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

void TestPowerCapReadWrite::Run(void) {
  rsmi_status_t ret;
  uint64_t orig, min, max, new_cap;
  clock_t start, end;
  double cpu_time_used;

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t dv_ind = 0; dv_ind < num_monitor_devs(); ++dv_ind) {
    PrintDeviceHeader(dv_ind);

    // Verify api support checking functionality is working
    ret = rsmi_dev_power_cap_range_get(dv_ind, 0, nullptr, nullptr);
    ASSERT_EQ(ret, RSMI_STATUS_INVALID_ARGS);

    // Verify api support checking functionality is working
    ret = rsmi_dev_power_cap_get(dv_ind, 0, nullptr);
    ASSERT_EQ(ret, RSMI_STATUS_INVALID_ARGS);

    ret = rsmi_dev_power_cap_range_get(dv_ind, 0, &max, &min);
    if (ret == RSMI_STATUS_NOT_SUPPORTED) {
      std::cout << "\t**rsmi_dev_power_cap_range_get(): Not supported on this machine" << std::endl;
      ASSERT_EQ(ret, RSMI_STATUS_NOT_SUPPORTED);
      continue;
    }

    ret = rsmi_dev_power_cap_get(dv_ind, 0, &orig);
    CHK_ERR_ASRT(ret)

    // Check if power cap is within the range
    // skip the test otherwise
    if (orig < min || orig > max) {
      std::cout << "Power cap is not within the range. Skipping test for " << dv_ind << std::endl;
      continue;
    }

    if (amd::smi::is_vm_guest()) {
      std::cout << "VM guest is not supported for power cap test. Skipping test for "
                << dv_ind << std::endl;
      continue;
    }

    new_cap = (max + min)/2;

    IF_VERB(STANDARD) {
      std::cout << "Original Power Cap: " << orig << " uW" << std::endl;
      std::cout << "Power Cap Range: " << max << " uW to " << min <<
                                                             " uW" << std::endl;
      std::cout << "Setting new cap to " << new_cap << "..." << std::endl;
    }
    start = clock();
    ret = rsmi_dev_power_cap_set(dv_ind, 0, new_cap);
    end = clock();
    cpu_time_used = (static_cast<double>(end - start)) * 1000000UL / CLOCKS_PER_SEC;

    CHK_ERR_ASRT(ret)

    ret = rsmi_dev_power_cap_get(dv_ind, 0, &new_cap);
    CHK_ERR_ASRT(ret)

    // TODO(cfreehil) add some kind of assertion to verify new_cap is correct
    //       (or within a range)
    IF_VERB(STANDARD) {
      std::cout << "Time spent: " << cpu_time_used << " uS" << std::endl;
      std::cout << "New Power Cap: " << new_cap << " uW" << std::endl;
      std::cout << "Resetting cap to " << orig << "..." << std::endl;
    }

    ret = rsmi_dev_power_cap_set(dv_ind, 0, orig);
    CHK_ERR_ASRT(ret)

    ret = rsmi_dev_power_cap_get(dv_ind, 0, &new_cap);
    CHK_ERR_ASRT(ret)

    IF_VERB(STANDARD) {
      std::cout << "Current Power Cap: " << new_cap << " uW" << std::endl;
    }
  }
}
