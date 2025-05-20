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

#include <cstdint>
#include <iostream>
#include <string>

#include "gtest/gtest.h"
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi_test/functional/frequencies_read.h"
#include "rocm_smi_test/test_common.h"

TestFrequenciesRead::TestFrequenciesRead() : TestBase() {
  set_title("RSMI Frequencies Read Test");
  set_description("The Frequency Read tests verifies that the "
              "available and current frequency levels can be read properly.");
}

TestFrequenciesRead::~TestFrequenciesRead(void) {
}

void TestFrequenciesRead::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestFrequenciesRead::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestFrequenciesRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestFrequenciesRead::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // rsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}


static void print_frequencies(rsmi_frequencies_t *f, uint32_t *l = nullptr) {
  assert(f != nullptr);
  for (uint32_t clk_i = 0; clk_i < f->num_supported; ++clk_i) {
    std::string clk_i_str;
    if (f->has_deep_sleep) {
      clk_i_str = (clk_i == 0) ? "S" : std::to_string(clk_i-1);
    } else {
      clk_i_str = std::to_string(clk_i);
    }
    std::cout << "\t**  " <<
      std::setw(2) << std::right << clk_i_str << ": " <<
      std::setw(11) << std::right << f->frequency[clk_i];
    if (l != nullptr) {
      std::cout << "T/s; x" << l[clk_i];
    } else {
      std::cout << "Hz";
    }

    if (clk_i == f->current) {
      std::cout << " *";
    }
    std::cout << std::endl;
  }
}

void TestFrequenciesRead::Run(void) {
  rsmi_status_t err;
  rsmi_frequencies_t f;
  rsmi_pcie_bandwidth_t b;

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t x = 0; x < num_iterations(); ++x) {
    for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
      auto freq_output = [&](rsmi_clk_type_t t, const char *name) {
        err = rsmi_dev_gpu_clk_freq_get(i, t, &f);
        if (err == RSMI_STATUS_NOT_SUPPORTED) {
          std::cout << "\t**Get " << name <<
                               ": Not supported on this machine" << std::endl;
          // Verify api support checking functionality is working
          err = rsmi_dev_gpu_clk_freq_get(i, t, nullptr);
          ASSERT_EQ(err, RSMI_STATUS_NOT_SUPPORTED);
          return;
        }

        // special driver issue, shouldn't normally occur
        if (err == RSMI_STATUS_UNEXPECTED_DATA) {
          std::cerr << "WARN: Clock file [" << FreqEnumToStr(t) << "] exists on device [" << i << "] but empty!" << std::endl;
          std::cerr << "      Likely a driver issue!" << std::endl;
          return;
        }

        CHK_ERR_ASRT(err)
        IF_VERB(STANDARD) {
          std::cout << "\t**Supported " << name << " clock frequencies: ";
          std::cout << f.num_supported << std::endl;
          print_frequencies(&f);
          // Verify api support checking functionality is working
          err = rsmi_dev_gpu_clk_freq_get(i, t, nullptr);
          ASSERT_EQ(err, RSMI_STATUS_INVALID_ARGS);
        }
      };

      PrintDeviceHeader(i);

      freq_output(RSMI_CLK_TYPE_MEM, "Supported GPU Memory");
      freq_output(RSMI_CLK_TYPE_SYS, "Supported GPU");
      freq_output(RSMI_CLK_TYPE_DF, "Data Fabric Clock");
      freq_output(RSMI_CLK_TYPE_DCEF, "Display Controller Engine Clock");
      freq_output(RSMI_CLK_TYPE_SOC, "SOC Clock");

      err = rsmi_dev_pci_bandwidth_get(i, &b);
      if (err == RSMI_STATUS_NOT_SUPPORTED) {
        std::cout << "\t**Get PCIE Bandwidth: Not supported on this machine"
                                                              << std::endl;
        // Verify api support checking functionality is working
        err = rsmi_dev_pci_bandwidth_get(i, nullptr);
        ASSERT_EQ(err, RSMI_STATUS_NOT_SUPPORTED);
      } else {
        CHK_ERR_ASRT(err)
        IF_VERB(STANDARD) {
          std::cout << "\t**Supported PCIe bandwidths: ";
          std::cout << b.transfer_rate.num_supported << std::endl;
          print_frequencies(&b.transfer_rate, b.lanes);
          // Verify api support checking functionality is working
          err = rsmi_dev_pci_bandwidth_get(i, nullptr);
          if (err != rsmi_status_t::RSMI_STATUS_NOT_SUPPORTED) {
            ASSERT_EQ(err, RSMI_STATUS_INVALID_ARGS);
          }
          else {
            auto status_string("");
            rsmi_status_string(err, &status_string);
            std::cout << "\t\t** rsmi_dev_pci_bandwidth_get(): " << status_string << "\n";
          }
        }
      }
    }
  }
}
