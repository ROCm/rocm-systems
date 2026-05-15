/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2019-2025, Advanced Micro Devices, Inc.
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

#include "gtest/gtest.h"
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi_test/functional/volt_freq_curv_read.h"
#include "rocm_smi_test/test_common.h"
#include "rocm_smi/rocm_smi_utils.h"

TestVoltCurvRead::TestVoltCurvRead() : TestBase() {
  set_title("RSMI Voltage-Frequency Curve Read Test");
  set_description("The Voltage-Frequency Read tests verifies that the voltage"
                         " frequency curve information can be read properly.");
}

TestVoltCurvRead::~TestVoltCurvRead(void) {
}

void TestVoltCurvRead::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestVoltCurvRead::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestVoltCurvRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestVoltCurvRead::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // rsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

void TestVoltCurvRead::Run(void) {
  rsmi_status_t err, ret;
  rsmi_od_volt_freq_data_t odv{};
  rsmi_dev_perf_level_t pfl;

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
    PrintDeviceHeader(i);

    std::cout << "\n\t**Resetting performance determinism to auto\n";
    err = rsmi_dev_perf_level_set(i, RSMI_DEV_PERF_LEVEL_AUTO);
    IF_VERB(STANDARD) {
      std::cout << "\t**rsmi_dev_perf_level_set(i, RSMI_DEV_PERF_LEVEL_AUTO): "
                << amd::smi::getRSMIStatusString(err, false)
                << "\n";
    }

    if (err != rsmi_status_t::RSMI_STATUS_NOT_SUPPORTED) {
      ASSERT_EQ(err, rsmi_status_t::RSMI_STATUS_SUCCESS);
      ret = rsmi_dev_perf_level_get(i, &pfl);
      IF_VERB(STANDARD) {
        std::cout << "\t**rsmi_dev_perf_level_get(i, &pfl): "
                  << amd::smi::getRSMIStatusString(ret, false) << "\n";
      }
      ASSERT_EQ(err, rsmi_status_t::RSMI_STATUS_SUCCESS);
    }
    else {
      IF_VERB(STANDARD) {
        std::cout << "\t**rsmi_dev_perf_level_get: Not supported on this "
                    "machine" << std::endl;
      }
    }

    // Verify api support checking functionality is working
    err = rsmi_dev_od_volt_info_get(i, nullptr);
    IF_VERB(STANDARD) {
      std::cout << "\t**rsmi_dev_od_volt_info_get(i, nullptr): "
                << amd::smi::getRSMIStatusString(err, false) << "\n";
    }

    if (err != rsmi_status_t::RSMI_STATUS_NOT_SUPPORTED) {
      ASSERT_EQ(err, rsmi_status_t::RSMI_STATUS_INVALID_ARGS);
      err = rsmi_dev_od_volt_info_get(i, &odv);
      IF_VERB(STANDARD) {
          std::cout << "\t**rsmi_dev_od_volt_info_get(i, &odv): "
                    << amd::smi::getRSMIStatusString(err, false) << "\n"
                    << amd::smi::print_rsmi_od_volt_freq_data_t(&odv)
                    << "\t**odv.num_regions = " << std::dec
                    << odv.num_regions << "\n";
      }
      if (err == rsmi_status_t::RSMI_STATUS_SUCCESS) {
        std::cout << "\t**Frequency-voltage curve data:" << "\n";
        std::cout << amd::smi::print_rsmi_od_volt_freq_data_t(&odv);

        rsmi_freq_volt_region_t *regions{};
        uint32_t num_regions;
        regions = new rsmi_freq_volt_region_t[odv.num_regions];
        ASSERT_NE(regions, nullptr);

        num_regions = odv.num_regions;
        err = rsmi_dev_od_volt_curve_regions_get(i, &num_regions, regions);
        IF_VERB(STANDARD) {
          std::cout << "\t**rsmi_dev_od_volt_curve_regions_get("
                    << "i, &num_regions, regions): "
                    << amd::smi::getRSMIStatusString(err, false) << "\n"
                    << "\t**Number of regions: " << std::dec << num_regions
                    << "\n";
        }
        ASSERT_TRUE(err == RSMI_STATUS_SUCCESS
                    || err == RSMI_STATUS_NOT_SUPPORTED
                    || err == RSMI_STATUS_UNEXPECTED_DATA
                    || err == RSMI_STATUS_UNEXPECTED_SIZE
                    || err == RSMI_STATUS_INVALID_ARGS);
        if (err != RSMI_STATUS_SUCCESS) {
          IF_VERB(STANDARD) {
            std::cout << "\t**rsmi_dev_od_volt_curve_regions_get: "
                        "Not supported on this machine" << std::endl;
          }
          continue;
        }
        ASSERT_EQ(err, rsmi_status_t::RSMI_STATUS_SUCCESS);
        ASSERT_EQ(num_regions, odv.num_regions);

        std::cout << "\t**Frequency-voltage curve regions:" << std::endl;
        std::cout << amd::smi::print_rsmi_od_volt_freq_regions(num_regions,
                                                              regions);

        delete []regions;
      }
    }
    else {
      IF_VERB(STANDARD) {
        std::cout << "\t**rsmi_dev_od_volt_info_get: Not supported on this "
                    "machine" << std::endl;
      }
    }
  }
}
