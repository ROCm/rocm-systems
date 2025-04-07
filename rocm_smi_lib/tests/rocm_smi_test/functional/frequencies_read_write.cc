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
#include <bitset>
#include <string>
#include <algorithm>

#include "gtest/gtest.h"
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi_test/functional/frequencies_read_write.h"
#include "rocm_smi_test/test_common.h"


TestFrequenciesReadWrite::TestFrequenciesReadWrite() : TestBase() {
  set_title("RSMI Frequencies Read/Write Test");
  set_description("The Frequencies tests verify that the frequency "
                       "settings can be read and controlled properly.");
}

TestFrequenciesReadWrite::~TestFrequenciesReadWrite(void) {
}

void TestFrequenciesReadWrite::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestFrequenciesReadWrite::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestFrequenciesReadWrite::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestFrequenciesReadWrite::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // rsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}


void TestFrequenciesReadWrite::Run(void) {
  rsmi_status_t ret;
  rsmi_frequencies_t f;
  uint32_t freq_bitmask;
  rsmi_clk_type rsmi_clk;

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t dv_ind = 0; dv_ind < num_monitor_devs(); ++dv_ind) {
    PrintDeviceHeader(dv_ind);

    for (uint32_t clk = RSMI_CLK_TYPE_FIRST; clk <= RSMI_CLK_TYPE_LAST; ++clk) {
      rsmi_clk = (rsmi_clk_type)clk;

      auto freq_read = [&]() -> bool {
        ret = rsmi_dev_gpu_clk_freq_get(dv_ind, rsmi_clk, &f);
        if (ret == RSMI_STATUS_NOT_SUPPORTED) {
          std::cout << "\t**Set " << FreqEnumToStr(rsmi_clk) <<
                               ": Not supported on this machine" << std::endl;
          return false;
        }

        // special driver issue, shouldn't normally occur
        if (ret == RSMI_STATUS_UNEXPECTED_DATA) {
          std::cerr << "WARN: Clock file [" << FreqEnumToStr(rsmi_clk) << "] exists on device [" << dv_ind << "] but empty!" << std::endl;
          std::cerr << "      Likely a driver issue!" << std::endl;
        }

        // CHK_ERR_ASRT(ret)
        IF_VERB(STANDARD) {
          std::cout << "Initial frequency for clock " <<
              FreqEnumToStr(rsmi_clk) << " is " << f.current << std::endl;
        }
        return true;
      };

      auto freq_write = [&]() {
        // Set clocks to something other than the usual default of the lowest
        // frequency.
        freq_bitmask = 0b01100;  // Try the 3rd and 4th clocks

        std::string freq_bm_str =
              std::bitset<RSMI_MAX_NUM_FREQUENCIES>(freq_bitmask).to_string();

        freq_bm_str.erase(0, std::min(freq_bm_str.find_first_not_of('0'),
                                                       freq_bm_str.size()-1));

        IF_VERB(STANDARD) {
        std::cout << "Setting frequency mask for " <<
            FreqEnumToStr(rsmi_clk) << " to 0b" << freq_bm_str << " ..." <<
                                                                    std::endl;
        }
        ret = rsmi_dev_gpu_clk_freq_set(dv_ind, rsmi_clk, freq_bitmask);
        // Certain ASICs does not allow to set particular clocks. If set function for a clock returns
        // permission error despite root access, manually set ret value to success and return
        //
        // Sometimes setting clock frequencies is completely not supported
        if ((ret == RSMI_STATUS_PERMISSION && geteuid() == 0) ||
            (ret == RSMI_STATUS_NOT_SUPPORTED)) {
          std::cout << "\t**Set " << FreqEnumToStr(rsmi_clk) <<
                              ": Not supported on this machine. Skipping..." << std::endl;
          ret = RSMI_STATUS_SUCCESS;
          return;
        }

        CHK_ERR_ASRT(ret)
        ret = rsmi_dev_gpu_clk_freq_get(dv_ind, rsmi_clk, &f);
        if (ret != RSMI_STATUS_SUCCESS) {
          return;
        }

        IF_VERB(STANDARD) {
          std::cout << "Frequency is now index " << f.current << std::endl;
          std::cout << "Resetting mask to all frequencies." << std::endl;
        }
        ret = rsmi_dev_gpu_clk_freq_set(dv_ind, rsmi_clk, 0xFFFFFFFF);
        if (ret == RSMI_STATUS_NOT_SUPPORTED) {
          std::cout << "\t**Set " << FreqEnumToStr(rsmi_clk)
                           << ": Not supported on this machine. Skipping..." << std::endl;
          ret = RSMI_STATUS_SUCCESS;
          return;
        }
        if (ret != RSMI_STATUS_SUCCESS) {
          return;
        }

        ret = rsmi_dev_perf_level_set(dv_ind, RSMI_DEV_PERF_LEVEL_AUTO);
        if (ret == RSMI_STATUS_NOT_SUPPORTED) {
          std::cout << "\t**Setting performance level is not supported on this machine. Skipping..." << std::endl;
          ret = RSMI_STATUS_SUCCESS;
          return;
        }
      };

      if (freq_read()) {
        CHK_ERR_ASRT(ret)
      } else {
        continue;
      }
      freq_write();
      CHK_ERR_ASRT(ret)
    }
  }
}
