/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2020, Advanced Micro Devices, Inc.
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
#include "rocm_smi_test/functional/gpu_metrics_read.h"
#include "rocm_smi_test/test_common.h"


TestGpuMetricsRead::TestGpuMetricsRead() : TestBase() {
  set_title("RSMI GPU Metrics Read Test");
  set_description("The GPU Metrics tests verifies that "
                  "the gpu metrics info can be read properly.");
}

TestGpuMetricsRead::~TestGpuMetricsRead(void) {
}

void TestGpuMetricsRead::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestGpuMetricsRead::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestGpuMetricsRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestGpuMetricsRead::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // rsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}


void TestGpuMetricsRead::Run(void) {
  rsmi_status_t err;

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
    PrintDeviceHeader(i);

    IF_VERB(STANDARD) {
        std::cout << "\t**GPU METRICS:\n";
    }
    rsmi_gpu_metrics_t smu;
    err = rsmi_dev_gpu_metrics_info_get(i, &smu);
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
          std::cout << std::dec << "system_clock_counter="
          << smu.system_clock_counter << '\n';
          std::cout << std::dec << "temperature_edge="
          << smu.temperature_edge << '\n';
          std::cout << std::dec << "temperature_hotspot="
          << smu.temperature_hotspot << '\n';
          std::cout << std::dec << "temperature_mem="
          << smu.temperature_mem << '\n';
          std::cout << std::dec << "temperature_vrgfx="
          << smu.temperature_vrgfx << '\n';
          std::cout << std::dec << "temperature_vrsoc="
          << smu.temperature_vrsoc << '\n';
          std::cout << std::dec << "temperature_vrmem="
          << smu.temperature_vrmem << '\n';
          std::cout << std::dec << "average_gfx_activity="
          << smu.average_gfx_activity << '\n';
          std::cout << std::dec << "average_umc_activity="
          << smu.average_umc_activity << '\n';
          std::cout << std::dec << "average_mm_activity="
          << smu.average_mm_activity << '\n';
          std::cout << std::dec << "average_socket_power="
          << smu.average_socket_power << '\n';
          std::cout << std::dec << "energy_accumulator="
          << smu.energy_accumulator << '\n';
          std::cout << std::dec << "average_gfxclk_frequency="
          << smu.average_gfxclk_frequency << '\n';
          std::cout << std::dec << "average_gfxclk_frequency="
          << smu.average_gfxclk_frequency << '\n';
          std::cout << std::dec << "average_uclk_frequency="
          << smu.average_uclk_frequency << '\n';
          std::cout << std::dec << "average_vclk0_frequency="
          << smu.average_vclk0_frequency << '\n';
          std::cout << std::dec << "average_dclk0_frequency="
          << smu.average_dclk0_frequency << '\n';
          std::cout << std::dec << "average_vclk1_frequency="
          << smu.average_vclk1_frequency << '\n';
          std::cout << std::dec << "average_dclk1_frequency="
          << smu.average_dclk1_frequency << '\n';
          std::cout << std::dec << "current_gfxclk="
          << smu.current_gfxclk << '\n';
          std::cout << std::dec << "current_socclk="
          << smu.current_socclk << '\n';
          std::cout << std::dec << "current_uclk="
          << smu.current_uclk << '\n';
          std::cout << std::dec << "current_vclk0="
          << smu.current_vclk0 << '\n';
          std::cout << std::dec << "current_dclk0="
          << smu.current_dclk0 << '\n';
          std::cout << std::dec << "current_vclk1="
          << smu.current_vclk1 << '\n';
          std::cout << std::dec << "current_dclk1="
          << smu.current_dclk1 << '\n';
          std::cout << std::dec << "throttle_status="
          << smu.throttle_status << '\n';
          std::cout << std::dec << "current_fan_speed="
          << smu.current_fan_speed << '\n';
          std::cout << "pcie_link_width="
          << std::to_string(smu.pcie_link_width) << '\n';
          std::cout << "pcie_link_width="
          << std::to_string(smu.pcie_link_speed) << '\n';
          std::cout << "gfx_activity_acc="
          << std::dec << smu.gfx_activity_acc << '\n';
          std::cout << "mem_actvity_acc="
          << std::dec << smu.mem_actvity_acc  << '\n';

          for (int i = 0; i < RSMI_NUM_HBM_INSTANCES; ++i) {
            std::cout << "temperature_hbm[" << i << "]=" << std::dec <<
                                               smu.temperature_hbm[i] << '\n';
          }
      }
    }

    // Verify api support checking functionality is working
    err = rsmi_dev_gpu_metrics_info_get(i, nullptr);
    ASSERT_EQ(err, RSMI_STATUS_INVALID_ARGS);
  }
}
