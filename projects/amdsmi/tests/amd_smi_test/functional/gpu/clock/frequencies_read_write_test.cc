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
#include "frequencies_read_write.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <bitset>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_utils.h"
#include "test_common.h"

TestFrequenciesReadWrite::TestFrequenciesReadWrite() : TestBase() {
  set_title("AMDSMI Frequencies Read/Write Test");
  set_description(
      "The Frequencies tests verify that the frequency "
      "settings can be read and controlled properly.");
}

TestFrequenciesReadWrite::~TestFrequenciesReadWrite(void) {}

void TestFrequenciesReadWrite::SetUp(void) {
  TestBase::SetUp();
  // Capture the per-device performance level so Close() can restore it and
  // avoid leaking a forced 'manual' perf state to later tests.
  SavePerfLevels();
  return;
}

void TestFrequenciesReadWrite::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestFrequenciesReadWrite::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestFrequenciesReadWrite::Close() {
  // Restore the performance level captured in SetUp() before amdsmi is shut
  // down. Runs even if Run() aborted on a failed assertion, so this test cannot
  // leak a forced 'manual' state to subsequent tests.
  RestorePerfLevels();
  // This will close handles opened within rsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

// Number of writable DPM labels. The deep-sleep pseudo-level ('S') is listed in
// the table but has no settable numeric label, so it is excluded.
static uint32_t settable_level_count(const amdsmi_frequencies_t& f) {
  return (f.has_deep_sleep && f.num_supported > 0) ? f.num_supported - 1 : f.num_supported;
}

// The subset of a requested set-mask that can actually take effect: the bits
// within the settable label range [0, settable_level_count). amd-smi (and the
// driver) drop bits above that, so they never apply.
static uint64_t effective_level_mask(uint64_t requested_mask, const amdsmi_frequencies_t& f) {
  uint32_t settable = settable_level_count(f);
  uint64_t in_range = (settable >= 64) ? ~0ULL : ((1ULL << settable) - 1);
  return requested_mask & in_range;
}

void TestFrequenciesReadWrite::Run(void) {
  amdsmi_status_t ret;
  amdsmi_frequencies_t f;
  uint64_t freq_bitmask;
  amdsmi_clk_type_t amdsmi_clk;
  const std::map<amdsmi_clk_type_t, std::string> clk_type_map = {
      {AMDSMI_CLK_TYPE_SYS, "SYS"},     {AMDSMI_CLK_TYPE_GFX, "GFX"},
      {AMDSMI_CLK_TYPE_DF, "DF"},       {AMDSMI_CLK_TYPE_DCEF, "DCEF"},
      {AMDSMI_CLK_TYPE_SOC, "SOC"},     {AMDSMI_CLK_TYPE_MEM, "MEM"},
      {AMDSMI_CLK_TYPE_PCIE, "PCIE"},   {AMDSMI_CLK_TYPE_VCLK0, "VCLK0"},
      {AMDSMI_CLK_TYPE_VCLK1, "VCLK1"}, {AMDSMI_CLK_TYPE_DCLK0, "DCLK0"},
      {AMDSMI_CLK_TYPE_DCLK1, "DCLK1"},
  };

  TestBase::Run();
  PRINT_VERBOSITY();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t dv_ind = 0; dv_ind < num_monitor_devs(); ++dv_ind) {
    PrintDeviceHeader(processor_handles_[dv_ind]);

    // Optional debug filter: set AMDSMI_TEST_CLK to a clock name (e.g. "SYS",
    // "GFX", "DF", "SOC", "MEM", "VCLK0", ...) to restrict this test to a
    // single clock type. Useful for isolating which clock a set error comes
    // from by running the test once per clock. Unset => exercise all clocks.
    const char* only_clk_env = std::getenv("AMDSMI_TEST_CLK");

    for (uint32_t clk = AMDSMI_CLK_TYPE_FIRST; clk <= AMDSMI_CLK_TYPE__MAX; ++clk) {
      amdsmi_clk = (amdsmi_clk_type_t)clk;

      if (only_clk_env != nullptr) {
        auto name_it = clk_type_map.find(amdsmi_clk);
        std::string clk_name = (name_it != clk_type_map.end()) ? name_it->second : "";
        if (clk_name != only_clk_env) {
          continue;  // skip clocks that don't match AMDSMI_TEST_CLK
        }
      }

      auto freq_read = [&]() -> bool {
        // Skip AMDSMI_CLK_TYPE_PCIE, which does not supported in rocm-smi.
        if (auto it = clk_type_map.find(amdsmi_clk); it != clk_type_map.end()) {
          if (amdsmi_clk == AMDSMI_CLK_TYPE_PCIE) {
            return false;  // Quietly skip PCIE clock
                           // Cannot read/write to PCIE clock in driver
          }
          std::cout << "amdsmi_get_clk_freq(" << it->second << ", f)" << std::endl;
        }

        DISPLAY_AMDSMI_API("amdsmi_get_clk_freq(" + std::string(FreqEnumToStr(amdsmi_clk)) + ")",
                           "gpu=" + std::to_string(dv_ind), VERB(STANDARD));
        ret = amdsmi_get_clk_freq(processor_handles_[dv_ind], amdsmi_clk, &f);
        DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
        if (auto it = clk_type_map.find(amdsmi_clk); it != clk_type_map.end()) {
          std::cout << ": " << smi_amdgpu_get_status_string(ret, false) << std::endl;
        }

        IF_VERB(STANDARD) {
          if (ret == AMDSMI_STATUS_SUCCESS) {
            std::cout << "\t**" << FreqEnumToStr(amdsmi_clk) << ":\n"
                      << "\t\t**Is in deep sleep: " << (f.has_deep_sleep ? "Yes" : "No") << "\n"
                      << "\t\t**Number of supported frequencies: " << f.num_supported << "\n"
                      << "\t\t**Current frequency index: " << f.current << "\n"
                      << "\t\t**Supported frequencies: ";
            for (uint32_t i = 0; i < f.num_supported; ++i) {
              std::cout << f.frequency[i] << " Hz";
              if (i < f.num_supported - 1) {
                std::cout << ", ";
              } else {
                std::cout << std::endl;
              }
            }
          }
        }

        if (ret == AMDSMI_STATUS_NOT_SUPPORTED || ret == AMDSMI_STATUS_NOT_YET_IMPLEMENTED) {
          std::cout << "\t**Set " << FreqEnumToStr(amdsmi_clk) << ": Not supported on this machine"
                    << std::endl;
          return false;
        }

        // special driver issue, shouldn't normally occur
        if (ret == AMDSMI_STATUS_UNEXPECTED_DATA) {
          std::cerr << "WARN: Clock file [" << FreqEnumToStr(amdsmi_clk) << "] exists on device ["
                    << dv_ind << "] but empty!" << std::endl;
          std::cerr << "      Likely a driver issue!" << std::endl;
        }

        // CHK_ERR_ASRT(ret)
        IF_VERB(STANDARD) {
          std::cout << "Initial frequency for clock " << FreqEnumToStr(amdsmi_clk) << " is "
                    << f.current << std::endl;
        }
        return true;
      };

      auto freq_write = [&]() {
        // Set clocks to something other than the usual default of the lowest
        // frequency.
        // Skip AMDSMI_CLK_TYPE_PCIE, which does not supported in rocm-smi.
        if (amdsmi_clk == AMDSMI_CLK_TYPE_PCIE) return;

        freq_bitmask = 0b01100;  // Try the 3rd and 4th DPM levels

        std::string freq_bm_str = std::bitset<AMDSMI_MAX_NUM_FREQUENCIES>(freq_bitmask).to_string();

        freq_bm_str.erase(0, std::min(freq_bm_str.find_first_not_of('0'), freq_bm_str.size() - 1));

        IF_VERB(STANDARD) {
          std::cout << "Setting frequency mask for " << FreqEnumToStr(amdsmi_clk) << " to 0b"
                    << freq_bm_str << " ..." << std::endl;
        }
        std::string set_api = "amdsmi_set_clk_freq(" + std::string(FreqEnumToStr(amdsmi_clk)) +
                              ", 0b" + freq_bm_str + ")";
        DISPLAY_AMDSMI_API(set_api, "gpu=" + std::to_string(dv_ind), VERB(STANDARD));
        ret = amdsmi_set_clk_freq(processor_handles_[dv_ind], amdsmi_clk, freq_bitmask);
        DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS,
                              AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NO_PERM,
                              AMDSMI_STATUS_INVAL);
        // Certain ASICs does not allow to set particular clocks. If set function for a clock
        // returns permission error despite root access, manually set ret value to success and
        // return
        //
        // Sometimes setting clock frequencies is completely not supported
        if ((ret == AMDSMI_STATUS_NO_PERM && geteuid() == 0) ||
            (ret == AMDSMI_STATUS_NOT_SUPPORTED)) {
          std::cout << "\t**Set " << FreqEnumToStr(amdsmi_clk)
                    << ": Not supported on this machine. Skipping..." << std::endl;
          ret = AMDSMI_STATUS_SUCCESS;
          return;
        }

        // Lenient set contract: amd-smi applies the in-range subset of the
        // requested mask and drops labels above the max settable level (like the
        // driver), so a set only fails with INVAL when the mask selects no
        // settable level at all.
        if (ret == AMDSMI_STATUS_INVAL) {
          std::cout << "\t**[Warn] AMDSMI_STATUS_INVAL - Could not set frequency mask for "
                    << FreqEnumToStr(amdsmi_clk) << " to 0b" << freq_bm_str << " on device ["
                    << dv_ind << "]!" << std::endl;
          ASSERT_EQ(effective_level_mask(freq_bitmask, f), 0ULL);
        } else {
          CHK_ERR_ASRT(ret)
        }

        DISPLAY_AMDSMI_API("amdsmi_get_clk_freq", "gpu=" + std::to_string(dv_ind), VERB(STANDARD));
        ret = amdsmi_get_clk_freq(processor_handles_[dv_ind], amdsmi_clk, &f);
        DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
        if (ret != AMDSMI_STATUS_SUCCESS) {
          return;
        }
        IF_VERB(STANDARD) {
          std::cout << "Frequency is now index " << f.current << std::endl;
          std::cout << "Resetting mask to all frequencies." << std::endl;
        }

        freq_bitmask = 0xFFFFFFFF;
        set_api = "amdsmi_set_clk_freq(" + std::string(FreqEnumToStr(amdsmi_clk)) + ", 0xFFFFFFFF)";
        DISPLAY_AMDSMI_API(set_api, "gpu=" + std::to_string(dv_ind), VERB(STANDARD));
        ret = amdsmi_set_clk_freq(processor_handles_[dv_ind], amdsmi_clk, freq_bitmask);
        DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS,
                              AMDSMI_STATUS_INVAL);
        if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
          ret = AMDSMI_STATUS_SUCCESS;
          return;
        }
        if (ret == AMDSMI_STATUS_INVAL) {
          // Only tolerated for a clock with no settable level (e.g. deep-sleep
          // only); otherwise the "enable all" reset must succeed.
          ASSERT_EQ(effective_level_mask(freq_bitmask, f), 0ULL);
          return;
        }
        CHK_ERR_ASRT(ret)

        DISPLAY_AMDSMI_API("amdsmi_set_gpu_perf_level", "gpu=" + std::to_string(dv_ind),
                           VERB(STANDARD));
        ret = amdsmi_set_gpu_perf_level(processor_handles_[dv_ind], AMDSMI_DEV_PERF_LEVEL_AUTO);
        DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
        if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
          ret = AMDSMI_STATUS_SUCCESS;
          return;
        }
      };

      if (freq_read()) {
        CHK_ERR_ASRT(ret)
      } else {
        continue;
      }
      freq_write();
      if (ret != AMDSMI_STATUS_INVAL) {
        CHK_ERR_ASRT(ret)
      }
    }
  }
}
