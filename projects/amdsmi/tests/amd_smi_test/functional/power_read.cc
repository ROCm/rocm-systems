/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2019-2024, Advanced Micro Devices, Inc.
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
#include <string>

#include <gtest/gtest.h>
#include "amd_smi/amdsmi.h"
#include "power_read.h"
#include "../test_common.h"

TestPowerRead::TestPowerRead() : TestBase() {
  set_title("AMDSMI Power Read Test");
  set_description("The Power Read tests verifies that "
                                "power related values can be read properly.");
}

TestPowerRead::~TestPowerRead(void) {
}

void TestPowerRead::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestPowerRead::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestPowerRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestPowerRead::Close() {
  // This will close handles opened within amdsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}


void TestPowerRead::Run(void) {
  amdsmi_status_t err;
  uint64_t val_ui64, val2_ui64;
  amdsmi_power_type_t type = AMDSMI_INVALID_POWER;

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t x = 0; x < num_iterations(); ++x) {
    for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
      PrintDeviceHeader(processor_handles_[i]);

      amdsmi_power_cap_info_t info;
      err = amdsmi_get_power_cap_info(processor_handles_[i], 0, &info);
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) {
        std::cout << "\t**Current Power Cap: " << info.power_cap << "uW" <<std::endl;
      }

      IF_VERB(STANDARD) {
        std::cout << "\t**Default Power Cap: " << info.default_power_cap << "uW" <<std::endl;
        std::cout << "\t**Power Cap Range: " << info.min_power_cap << " to " <<
                                                 info.max_power_cap << " uW" << std::endl;
      }
      // TODO: Add current_socket_power tests
    }
  }
}
