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
  for (uint32_t j = 0; j < f->num_supported; ++j) {
    std::cout << "\t**  " << j << ": " << f->frequency[j];
    if (l != nullptr) {
      std::cout << "T/s; x" << l[j];
    } else {
      std::cout << "Hz";
    }

    if (j == f->current) {
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


  for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
    auto freq_output = [&](rsmi_clk_type_t t, const char *name) {
      err = rsmi_dev_gpu_clk_freq_get(i, t, &f);
      if (err == RSMI_STATUS_NOT_SUPPORTED || err == RSMI_STATUS_FILE_ERROR) {
        std::cout << "\t**Get " << name << ": Not supported on this machine"
                                                                   << std::endl;
      } else {
          CHK_ERR_ASRT(err)
          IF_VERB(STANDARD) {
            std::cout << "\t**Supported " << name << " clock frequencies: ";
            std::cout << f.num_supported << std::endl;
            print_frequencies(&f);
          }
      }
    };

    PrintDeviceHeader(i);

    freq_output(RSMI_CLK_TYPE_MEM, "Supported GPU Memory");
    freq_output(RSMI_CLK_TYPE_SYS, "Supported GPU");
    freq_output(RSMI_CLK_TYPE_DF, "Data Fabric Clock");
    freq_output(RSMI_CLK_TYPE_DCEF, "Display Controller Engine Clock");
    freq_output(RSMI_CLK_TYPE_SOC, "SOC Clock");

    err = rsmi_dev_pci_bandwidth_get(i, &b);
    if (err == RSMI_STATUS_NOT_SUPPORTED || err == RSMI_STATUS_FILE_ERROR) {
      std::cout << "\t**Get PCIE Bandwidth: Not supported on this machine"
                                                            << std::endl;
    } else {
        CHK_ERR_ASRT(err)
        IF_VERB(STANDARD) {
          std::cout << "\t**Supported PCIe bandwidths: ";
          std::cout << b.transfer_rate.num_supported << std::endl;
          print_frequencies(&b.transfer_rate, b.lanes);
        }
    }
  }
}
