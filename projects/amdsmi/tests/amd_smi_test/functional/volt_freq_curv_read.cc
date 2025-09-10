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

#include <cstdint>

#include <iostream>
#include <string>

#include <gtest/gtest.h>
#include "amd_smi/amdsmi.h"
#include "volt_freq_curv_read.h"

TestVoltCurvRead::TestVoltCurvRead() : TestBase() {
  set_title("AMDSMI Voltage-Frequency Curve Read Test");
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
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

static void pt_rng_Mhz(std::string title, amdsmi_range_t *r) {
  assert(r != nullptr);

  std::cout << title << std::endl;
  std::cout << "\t\t** " << r->lower_bound/1000000 << " to " <<
                                r->upper_bound/1000000 << " MHz" << std::endl;
}

static void pt_rng_mV(std::string title, amdsmi_range_t *r) {
  assert(r != nullptr);

  std::cout << title << std::endl;
  std::cout << "\t\t** " << r->lower_bound << " to " << r->upper_bound <<
                                                           " mV" << std::endl;
}

static void print_pnt(amdsmi_od_vddc_point_t *pt) {
  std::cout << "\t\t** Frequency: " << pt->frequency/1000000 << "MHz" <<
                                                                    std::endl;
  std::cout << "\t\t** Voltage: " << pt->voltage << "mV" << std::endl;
}
static void pt_vddc_curve(amdsmi_od_volt_curve_t *c) {
  assert(c != nullptr);

  for (uint32_t i = 0; i < AMDSMI_NUM_VOLTAGE_CURVE_POINTS; ++i) {
    print_pnt(&c->vc_points[i]);
  }
}

static void print_amdsmi_od_volt_freq_data_t(amdsmi_od_volt_freq_data_t *odv) {
  assert(odv != nullptr);

  std::cout.setf(std::ios::dec, std::ios::basefield);
  pt_rng_Mhz("\t\tCurrent SCLK frequency range:", &odv->curr_sclk_range);
  pt_rng_Mhz("\t\tCurrent MCLK frequency range:", &odv->curr_mclk_range);
  pt_rng_Mhz("\t\tMin/Max Possible SCLK frequency range:",
                                                      &odv->sclk_freq_limits);
  pt_rng_Mhz("\t\tMin/Max Possible MCLK frequency range:",
                                                      &odv->mclk_freq_limits);

  std::cout << "\t\tCurrent Freq/Volt. curve:" << std::endl;
  pt_vddc_curve(&odv->curve);

  std::cout << "\tNumber of Freq./Volt. regions: " <<
                                                odv->num_regions << std::endl;
}

static void print_odv_region(amdsmi_freq_volt_region_t *region) {
  pt_rng_Mhz("\t\tFrequency range:", &region->freq_range);
  pt_rng_mV("\t\tVoltage range:", &region->volt_range);
}

static void print_amdsmi_od_volt_freq_regions(uint32_t num_regions,
                                             amdsmi_freq_volt_region_t *regions) {
  for (uint32_t i = 0; i < num_regions; ++i) {
    std::cout << "\tRegion " << i << ":" << std::endl;
    print_odv_region(&regions[i]);
  }
}

void TestVoltCurvRead::Run(void) {
  amdsmi_status_t err;
  amdsmi_od_volt_freq_data_t odv{};

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
    PrintDeviceHeader(processor_handles_[i]);

    err =  amdsmi_get_gpu_od_volt_info(processor_handles_[i], &odv);
    if (err == AMDSMI_STATUS_NOT_SUPPORTED
          || err == AMDSMI_STATUS_NOT_YET_IMPLEMENTED) {
      //TODO add perf_level tests
      IF_VERB(STANDARD) {
        std::cout <<
            "\t** amdsmi_get_gpu_od_volt_info: Not supported on this machine"
                                                               << std::endl;
      }
      // Verify api support checking functionality is working
      if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
        err =  amdsmi_get_gpu_od_volt_info(processor_handles_[i], nullptr);
        ASSERT_EQ(err, AMDSMI_STATUS_NOT_SUPPORTED);
      }
    } else {
      // Verify api support checking functionality is working
      err =  amdsmi_get_gpu_od_volt_info(processor_handles_[i], nullptr);
      ASSERT_EQ(err, AMDSMI_STATUS_INVAL);
    }

    if (err == AMDSMI_STATUS_SUCCESS) {
      std::cout << "\t**Frequency-voltage curve data:" << std::endl;
      print_amdsmi_od_volt_freq_data_t(&odv);

      amdsmi_freq_volt_region_t *regions{};
      uint32_t num_regions;
      regions = new amdsmi_freq_volt_region_t[odv.num_regions];
      ASSERT_NE(regions, nullptr);

      num_regions = odv.num_regions;
      err =  amdsmi_get_gpu_od_volt_curve_regions(processor_handles_[i],
                                                  &num_regions, regions);

      IF_VERB(STANDARD) {
        std::cout << "\t**amdsmi_get_gpu_od_volt_curve_regions("
                  << "processor_handles_[i], &num_regions, regions): "
                  << err << "\n"
                  << "\t**Number of regions: " << std::dec << num_regions
                  << "\n";
      }
      ASSERT_TRUE(err == AMDSMI_STATUS_SUCCESS
                  || err == AMDSMI_STATUS_NOT_SUPPORTED
                  || err == AMDSMI_STATUS_UNEXPECTED_DATA
                  || err == AMDSMI_STATUS_UNEXPECTED_SIZE
                  || err == AMDSMI_STATUS_INVAL);
      if (err != AMDSMI_STATUS_SUCCESS) {
        IF_VERB(STANDARD) {
          std::cout << "\t**amdsmi_get_gpu_od_volt_curve_regions: "
                       "Not supported on this machine" << std::endl;
        }
        continue;
      }

      ASSERT_EQ(err, AMDSMI_STATUS_SUCCESS);
      ASSERT_EQ(num_regions, odv.num_regions);

      std::cout << "\t**Frequency-voltage curve regions:" << std::endl;
      print_amdsmi_od_volt_freq_regions(num_regions, regions);

      delete []regions;
    }
  }
}
