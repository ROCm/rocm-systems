/*
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
#include <string>
#include <map>

#include "gtest/gtest.h"
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi_test/functional/mem_util_read.h"
#include "rocm_smi_test/test_common.h"

TestMemUtilRead::TestMemUtilRead() : TestBase() {
  set_title("Memory Utilization Read Test");
  set_description("The Memory Utilization Read tests verifies that "
           "memory busy percent, size and amount used can be read properly.");
}

TestMemUtilRead::~TestMemUtilRead(void) {
}

void TestMemUtilRead::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestMemUtilRead::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestMemUtilRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestMemUtilRead::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // rsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

static const std::map<rsmi_memory_type_t, const char *>
   kDevMemoryTypeNameMap = {
    {RSMI_MEM_TYPE_VRAM, "VRAM memory"},
    {RSMI_MEM_TYPE_VIS_VRAM, "Visible VRAM memory"},
    {RSMI_MEM_TYPE_GTT, "GTT memory"},
};

void TestMemUtilRead::Run(void) {
  rsmi_status_t err;
  uint64_t total;
  uint64_t usage;

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  auto err_chk = [&](const char *str) {
    if (err != RSMI_STATUS_SUCCESS) {
      if (err == RSMI_STATUS_FILE_ERROR) {
        IF_VERB(STANDARD) {
          std::cout << "\t** " << str << ": Not supported on this machine"
                                                                << std::endl;
        }
      } else {
        CHK_ERR_ASRT(err)
      }
    }
  };

  for (uint32_t x = 0; x < num_iterations(); ++x) {
    for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
      PrintDeviceHeader(i);

#if 0
      err = rsmi_dev_memory_busy_percent_get(i, &mem_busy_percent);
      err_chk("rsmi_dev_memory_busy_percent_get()");
      if (err != RSMI_STATUS_SUCCESS) {
        return;
      }
      IF_VERB(STANDARD) {
        std::cout << "\t**" << "GPU Memory Busy %: " << mem_busy_percent <<
                                                                      std::endl;
      }
#endif
      for (uint32_t mem_type = RSMI_MEM_TYPE_FIRST;
                                   mem_type <= RSMI_MEM_TYPE_LAST; ++mem_type) {
        err = rsmi_dev_memory_total_get(i,
                             static_cast<rsmi_memory_type_t>(mem_type), &total);
        err_chk("rsmi_dev_memory_total_get()");
        if (err != RSMI_STATUS_SUCCESS) {
          return;
        }

        err = rsmi_dev_memory_usage_get(i,
                             static_cast<rsmi_memory_type_t>(mem_type), &usage);
        err_chk("rsmi_dev_memory_usage_get()");
        if (err != RSMI_STATUS_SUCCESS) {
          return;
        }

        IF_VERB(STANDARD) {
          std::cout << "\t**" <<
           kDevMemoryTypeNameMap.at(static_cast<rsmi_memory_type_t>(mem_type))
            << " Calculated Utilization: " <<
              (static_cast<float>(usage)*100)/total << "% ("<< usage <<
                                              "/" << total << ")" << std::endl;
        }
      }
    }
  }
}
