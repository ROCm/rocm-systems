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
#include <map>

#include "gtest/gtest.h"
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi_test/functional/perf_level_read_write.h"
#include "rocm_smi_test/test_common.h"


TestPerfLevelReadWrite::TestPerfLevelReadWrite() : TestBase() {
  set_title("RSMI Performance Level Read/Write Test");
  set_description("The Performance Level tests verify that the performance "
                       "level settings can be read and controlled properly.");
}

TestPerfLevelReadWrite::~TestPerfLevelReadWrite(void) {
}

void TestPerfLevelReadWrite::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestPerfLevelReadWrite::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestPerfLevelReadWrite::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestPerfLevelReadWrite::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // rsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}


void TestPerfLevelReadWrite::Run(void) {
  rsmi_status_t ret;
  rsmi_dev_perf_level_t pfl, orig_pfl;

  TestBase::Run();
  if (setup_failed_) {
    IF_VERB(STANDARD) {
      std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    }
    return;
  }

  for (uint32_t dv_ind = 0; dv_ind < num_monitor_devs(); ++dv_ind) {
    PrintDeviceHeader(dv_ind);

    ret = rsmi_dev_perf_level_get(dv_ind, &orig_pfl);
    CHK_ERR_ASRT(ret)

    IF_VERB(STANDARD) {
      std::cout << "\t**Original Perf Level:" <<
                                       GetPerfLevelStr(orig_pfl) << std::endl;
    }

    uint32_t pfl_i = static_cast<uint32_t>(RSMI_DEV_PERF_LEVEL_FIRST);
    for (; pfl_i <=  static_cast<uint32_t>(RSMI_DEV_PERF_LEVEL_LAST); pfl_i++) {
      if (pfl_i == static_cast<uint32_t>(orig_pfl)) {
        continue;
      }

      IF_VERB(STANDARD) {
        std::cout << "Set Performance Level to " <<
            GetPerfLevelStr(static_cast<rsmi_dev_perf_level_t>(pfl_i)) <<
                                                            " ..." << std::endl;
      }
      ret = rsmi_dev_perf_level_set(dv_ind,
                                     static_cast<rsmi_dev_perf_level_t>(pfl_i));
      if (ret == RSMI_STATUS_NOT_SUPPORTED) {
          std::cout << "\t**" << GetPerfLevelStr(static_cast<rsmi_dev_perf_level_t>(pfl_i)) 
                  << " returned RSMI_STATUS_NOT_SUPPORTED"  << std::endl;
      } else {
          CHK_ERR_ASRT(ret)
          ret = rsmi_dev_perf_level_get(dv_ind, &pfl);
          CHK_ERR_ASRT(ret)
          IF_VERB(STANDARD) {
              std::cout << "\t**New Perf Level:" << GetPerfLevelStr(pfl) <<
                                                                    std::endl;
        }
      }
    }
    IF_VERB(STANDARD) {
      std::cout << "Reset Perf level to " << GetPerfLevelStr(orig_pfl) <<
                                                            " ..." << std::endl;
    }
    ret = rsmi_dev_perf_level_set(dv_ind, orig_pfl);
    CHK_ERR_ASRT(ret)
    ret = rsmi_dev_perf_level_get(dv_ind, &pfl);
    CHK_ERR_ASRT(ret)

    IF_VERB(STANDARD) {
      std::cout << "\t**New Perf Level:" << GetPerfLevelStr(pfl) <<
                                                                      std::endl;
    }
  }
}
