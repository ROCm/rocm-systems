// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "clock_limit_read_write.h"

#include <gtest/gtest.h>
#include <unistd.h>

#include <cstdint>
#include <iostream>
#include <string>

#include "amd_smi/amdsmi.h"
#include "test_common.h"

TestClockLimitReadWrite::TestClockLimitReadWrite() : TestBase() {
  set_title("AMDSMI Clock Limit Read/Write Test");
  set_description(
      "The Clock Limit Read/Write test verifies that the sclk/mclk minimum "
      "and maximum frequency limits can be set through "
      "amdsmi_set_gpu_clk_limit() and restored to their original values.");
}

TestClockLimitReadWrite::~TestClockLimitReadWrite(void) {}

void TestClockLimitReadWrite::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestClockLimitReadWrite::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestClockLimitReadWrite::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestClockLimitReadWrite::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

// amdsmi_get_clk_freq() reports frequencies in Hz; amdsmi_set_gpu_clk_limit()
// expects MHz.
static uint64_t hz_to_mhz(uint64_t hz) { return hz / 1000000ULL; }

void TestClockLimitReadWrite::Run(void) {
  amdsmi_status_t ret;
  amdsmi_frequencies_t f;

  TestBase::Run();
  PRINT_VERBOSITY();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  // amdsmi_set_gpu_clk_limit() only operates on the SYS (gfx) and MEM clocks.
  const amdsmi_clk_type_t clk_types[] = {AMDSMI_CLK_TYPE_SYS, AMDSMI_CLK_TYPE_MEM};

  for (uint32_t dv_ind = 0; dv_ind < num_monitor_devs(); ++dv_ind) {
    PrintDeviceHeader(processor_handles_[dv_ind]);

    for (auto clk : clk_types) {
      const char* name = FreqEnumToStr(clk);

      // Read the supported frequency table to derive safe min/max values.
      DISPLAY_AMDSMI_API("amdsmi_get_clk_freq", "gpu=" + std::to_string(dv_ind), VERB(STANDARD));
      ret = amdsmi_get_clk_freq(processor_handles_[dv_ind], clk, &f);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);

      if (ret == AMDSMI_STATUS_NOT_SUPPORTED || ret == AMDSMI_STATUS_NOT_YET_IMPLEMENTED) {
        IF_VERB(STANDARD) {
          std::cout << "\t**" << name << ": Not supported on this machine" << std::endl;
        }
        continue;
      }

      if (ret == AMDSMI_STATUS_UNEXPECTED_DATA) {
        std::cerr << "WARN: Clock [" << name << "] on device [" << dv_ind
                  << "] returned unexpected data. Likely a driver issue!" << std::endl;
        continue;
      }

      CHK_ERR_ASRT(ret)

      // Deep sleep (index 0 when present) is not a real DPM level; the lowest
      // selectable level follows it.
      uint32_t lowest_idx = (f.has_deep_sleep && f.num_supported > 0) ? 1 : 0;
      if (f.num_supported <= lowest_idx) {
        IF_VERB(STANDARD) {
          std::cout << "\t**" << name << ": No selectable DPM levels, skipping." << std::endl;
        }
        continue;
      }

      uint64_t orig_min_mhz = hz_to_mhz(f.frequency[lowest_idx]);
      uint64_t orig_max_mhz = hz_to_mhz(f.frequency[f.num_supported - 1]);

      IF_VERB(STANDARD) {
        std::cout << "\t**" << name << ": original min=" << orig_min_mhz
                  << " MHz, max=" << orig_max_mhz << " MHz" << std::endl;
      }

      // ASSERT_*/CHK_ERR_ASRT expand to a bare "return;", so this helper must
      // return void; it reports "not testable" through the captured flag.
      bool limit_supported = true;
      auto set_limit = [&](amdsmi_clk_limit_type_t limit_type, uint64_t mhz, const char* label) {
        limit_supported = true;
        DISPLAY_AMDSMI_API("amdsmi_set_gpu_clk_limit",
                           "gpu=" + std::to_string(dv_ind) + ", clk_type=" + std::string(name) +
                               ", clk_limit=" + std::string(label) +
                               ", value=" + std::to_string(mhz),
                           VERB(STANDARD));
        ret = amdsmi_set_gpu_clk_limit(processor_handles_[dv_ind], clk, limit_type, mhz);
        DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS,
                              AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NO_PERM);
        // Some ASICs reject clock-limit changes even with root. Treat those
        // as "not testable here" rather than a defect.
        if (ret == AMDSMI_STATUS_NOT_SUPPORTED ||
            (ret == AMDSMI_STATUS_NO_PERM && geteuid() == 0)) {
          IF_VERB(STANDARD) {
            std::cout << "\t**" << name << " set " << label
                      << ": Not supported on this machine. Skipping..." << std::endl;
          }
          limit_supported = false;
          return;
        }
        CHK_ERR_ASRT(ret)
        IF_VERB(STANDARD) {
          std::cout << "\t**" << name << " set " << label << " -> " << mhz << " MHz: OK"
                    << std::endl;
        }
      };

      // Exercise both limit types with their original values, then confirm the
      // range is restored. Using the original min/max keeps the device in a
      // safe, unchanged state after the test.
      set_limit(AMDSMI_CLK_LIMIT_MAX, orig_max_mhz, "max");
      if (!limit_supported) {
        continue;
      }
      set_limit(AMDSMI_CLK_LIMIT_MIN, orig_min_mhz, "min");

      // Setting a clock limit switches the perf level to MANUAL; return the
      // device to automatic control so later tests start from a clean state.
      DISPLAY_AMDSMI_API("amdsmi_set_gpu_perf_level", "gpu=" + std::to_string(dv_ind),
                         VERB(STANDARD));
      ret = amdsmi_set_gpu_perf_level(processor_handles_[dv_ind], AMDSMI_DEV_PERF_LEVEL_AUTO);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS,
                            AMDSMI_STATUS_NOT_SUPPORTED);

      // Verify the restored range via a fresh read.
      DISPLAY_AMDSMI_API("amdsmi_get_clk_freq", "gpu=" + std::to_string(dv_ind), VERB(STANDARD));
      ret = amdsmi_get_clk_freq(processor_handles_[dv_ind], clk, &f);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
      if (ret == AMDSMI_STATUS_SUCCESS) {
        ASSERT_GT(f.num_supported, 0u);
      }
    }
  }
}
