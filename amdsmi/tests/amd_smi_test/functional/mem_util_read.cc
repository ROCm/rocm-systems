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

#include <stdint.h>
#include <stddef.h>
#include <gtest/gtest.h>

#include <iostream>
#include <string>
#include <map>

#include "amd_smi/amdsmi.h"
#include "mem_util_read.h"
#include "../test_common.h"
#include "amd_smi/impl/amd_smi_utils.h"

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
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

static const std::map<amdsmi_memory_type_t, const char *>
   kDevMemoryTypeNameMap = {
    {AMDSMI_MEM_TYPE_VRAM, "VRAM memory"},
    {AMDSMI_MEM_TYPE_VIS_VRAM, "Visible VRAM memory"},
    {AMDSMI_MEM_TYPE_GTT, "GTT memory"},
};

void TestMemUtilRead::Run(void) {
  amdsmi_status_t err;
  uint64_t total;
  uint64_t usage;

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  auto err_chk = [&](const char *str) {
    IF_VERB(STANDARD) {
      std::cout << "\t** " << str << std::endl;
    }
    if (err != AMDSMI_STATUS_SUCCESS) {
      if (err == AMDSMI_STATUS_FILE_ERROR ||
          err == AMDSMI_STATUS_NOT_SUPPORTED) {
        ASSERT_TRUE(err == AMDSMI_STATUS_NOT_SUPPORTED
                    || err == AMDSMI_STATUS_FILE_ERROR);
      } else {
        CHK_ERR_ASRT(err)
      }
    }
  };

  for (uint32_t x = 0; x < num_iterations(); ++x) {
    for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
      PrintDeviceHeader(processor_handles_[i]);

      for (uint32_t mem_type = AMDSMI_MEM_TYPE_FIRST;
                                   mem_type <= AMDSMI_MEM_TYPE_LAST; ++mem_type) {
        err = amdsmi_get_gpu_memory_total(processor_handles_[i],
                             static_cast<amdsmi_memory_type_t>(mem_type), &total);
        smi_amdgpu_get_status_string(err, false);
        std::string mem_type_str =
          kDevMemoryTypeNameMap.at(static_cast<amdsmi_memory_type_t>(mem_type));
        std::string input_str =
          "amdsmi_get_gpu_memory_total(" + mem_type_str + "): "
          + smi_amdgpu_get_status_string(err, false);
        err_chk(input_str.c_str());
        if (err != AMDSMI_STATUS_SUCCESS) {
          continue;
        }

        err = amdsmi_get_gpu_memory_usage(processor_handles_[i],
                             static_cast<amdsmi_memory_type_t>(mem_type), &usage);
        input_str =
          "amdsmi_get_gpu_memory_usage(" + mem_type_str + "): "
          + smi_amdgpu_get_status_string(err, false);
        err_chk(input_str.c_str());
        if (err != AMDSMI_STATUS_SUCCESS) {
          continue;
        }

        IF_VERB(STANDARD) {
          std::cout << "\t**" <<
           kDevMemoryTypeNameMap.at(static_cast<amdsmi_memory_type_t>(mem_type))
            << " Calculated Utilization: " <<
              (static_cast<float>(usage)*100)/static_cast<float>(total) << "% (" << usage <<
                                              "/" << total << ")" << std::endl;
        }
      }
    }
  }
}
