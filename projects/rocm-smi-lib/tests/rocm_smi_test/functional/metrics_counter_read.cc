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
#include <string>
#include <map>

#include "gtest/gtest.h"
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi_test/functional/metrics_counter_read.h"
#include "rocm_smi_test/test_common.h"


TestMetricsCounterRead::TestMetricsCounterRead() : TestBase() {
  set_title("RSMI GPU Metrics Counter Read Test");
  set_description("The GPU Metrics Counter tests verifies that "
                  "the gpu metrics counter info can be read properly.");
}

TestMetricsCounterRead::~TestMetricsCounterRead(void) {
}

void TestMetricsCounterRead::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestMetricsCounterRead::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestMetricsCounterRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestMetricsCounterRead::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // rsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}


void TestMetricsCounterRead::Run(void) {
  rsmi_status_t err;

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
    PrintDeviceHeader(i);

    IF_VERB(STANDARD) {
        std::cout << "\t**GPU METRICS ENERGY COUNTER:\n";
    }

    uint64_t power;
    uint64_t timestamp;
    float counter_resolution;
    err = rsmi_dev_energy_count_get(i, &power, &counter_resolution, &timestamp);
    if (err != RSMI_STATUS_SUCCESS) {
      if (err == RSMI_STATUS_NOT_SUPPORTED) {
        IF_VERB(STANDARD) {
          std::cout << "\t**" <<
          "Not supported on this machine" << std::endl;
          return;
        }
      }
    } else {
      CHK_ERR_ASRT(err);
      IF_VERB(STANDARD) {
          std::cout << std::dec << "power counter="
          << power << '\n';
          std::cout << "power in uJ="
          << (double)(power * counter_resolution) << '\n';
          std::cout << std::dec << "timestamp="
          << timestamp << '\n';
      }
    }

    // Verify api support checking functionality is working
    err = rsmi_dev_energy_count_get(i, nullptr, nullptr, nullptr);
    ASSERT_EQ(err, RSMI_STATUS_INVALID_ARGS);

    // Coarse Grain counters
    rsmi_utilization_counter_t utilization_counters[2];
    utilization_counters[0].type = RSMI_COARSE_GRAIN_GFX_ACTIVITY;
    utilization_counters[1].type = RSMI_COARSE_GRAIN_MEM_ACTIVITY;
    err = rsmi_utilization_count_get(i, utilization_counters,
                    2, &timestamp);
    if (err != RSMI_STATUS_SUCCESS) {
      if (err == RSMI_STATUS_NOT_SUPPORTED) {
        IF_VERB(STANDARD) {
          std::cout << "\t**" <<
          "Not supported on this machine" << std::endl;
          return;
        }
      }
    } else {
      CHK_ERR_ASRT(err);
      IF_VERB(STANDARD) {
          std::cout << std::dec << "gfx_activity="
          << utilization_counters[0].value << '\n';
          std::cout << std::dec << "mem_activity="
          << utilization_counters[1].value << '\n';
          std::cout << std::dec << "timestamp="
          << timestamp << '\n';
      }
    }

    // Verify api support checking functionality is working
    err = rsmi_utilization_count_get(i, nullptr,
                    1 , nullptr);
    ASSERT_EQ(err, RSMI_STATUS_INVALID_ARGS);
  }  // end for
}
