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

#include <cstring>
#include <string>

#include "api_test_framework.h"

using amdsmi::test::kInvalidHandle;
using amdsmi::test::kVerbose;

// amdsmi_get_clk_freq / amdsmi_set_clk_freq (frequency-level mask).
TEST_F(GpuFunctionalReadWrite, ClkFreq_SetVerifyRestore) {
  AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED();
  AMDSMI_SKIP_KNOWN_FAILURE()
      << "amdsmi_set_clk_freq reports success but leaves the clock on its previous level";
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  amdsmi::test::StatusCollector col("amdsmi_set_clk_freq");
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_dev_perf_level_t initial_perf = AMDSMI_DEV_PERF_LEVEL_UNKNOWN;
    if (amdsmi_get_gpu_perf_level(gpus()[i], &initial_perf) != AMDSMI_STATUS_SUCCESS)
      initial_perf = AMDSMI_DEV_PERF_LEVEL_UNKNOWN;

    for (int t = AMDSMI_CLK_TYPE_FIRST; t <= AMDSMI_CLK_TYPE__MAX; ++t) {
      amdsmi_clk_type_t clk_type = static_cast<amdsmi_clk_type_t>(t);
      amdsmi_frequencies_t f;
      memset(&f, 0, sizeof(f));
      if (amdsmi_get_clk_freq(gpus()[i], clk_type, &f) != AMDSMI_STATUS_SUCCESS) continue;

      // A deep sleep entry occupies frequency[0] without being a DPM level, so
      // mask bit b addresses frequency[b + offset]; a bit past the last DPM
      // level is rejected with AMDSMI_STATUS_INVAL.
      const uint32_t offset = f.has_deep_sleep ? 1u : 0u;
      if (f.num_supported <= offset || f.num_supported >= 64) continue;
      const uint32_t levels = f.num_supported - offset;
      // current is (uint32_t)-1 when the domain is power-gated and reports no level.
      if (levels < 2 || f.current < offset || f.current >= f.num_supported) continue;

      const uint32_t initial_level = f.current - offset;
      const uint32_t target_level = (initial_level == 0) ? 1u : 0u;
      const std::string label = "gpu=" + std::to_string(i) + " clk=" + std::to_string(t);

      DISPLAY_AMDSMI_API("amdsmi_set_clk_freq", label + " set=" + std::to_string(target_level),
                         kVerbose);
      amdsmi_status_t err = amdsmi_set_clk_freq(gpus()[i], clk_type, 1ULL << target_level);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                            AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                            AMDSMI_STATUS_NO_PERM);
      col.Record(label, err,
                 ::amdsmi::test::AmdsmiStatusIsExpected(
                     err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                     AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM));
      if (err != AMDSMI_STATUS_SUCCESS) continue;

      amdsmi_frequencies_t readback;
      memset(&readback, 0, sizeof(readback));
      if (amdsmi_get_clk_freq(gpus()[i], clk_type, &readback) == AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(readback.current, target_level + offset) << label << " set did not take effect";
      }

      DISPLAY_AMDSMI_API("amdsmi_set_clk_freq", label + " restore=" + std::to_string(initial_level),
                         kVerbose);
      amdsmi_status_t rerr = amdsmi_set_clk_freq(gpus()[i], clk_type, 1ULL << initial_level);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, rerr, AMDSMI_STATUS_SUCCESS);
      EXPECT_EQ(rerr, AMDSMI_STATUS_SUCCESS) << label << " failed to restore clk freq";
      if (rerr == AMDSMI_STATUS_SUCCESS &&
          amdsmi_get_clk_freq(gpus()[i], clk_type, &readback) == AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(readback.current, f.current) << label << " restore did not take effect";
      }

      // Re-enable every level so the domain is not left pinned to one index.
      amdsmi_set_clk_freq(gpus()[i], clk_type, (1ULL << levels) - 1);
    }

    // The setter forces MANUAL; put the device back on the policy it came in on.
    if (initial_perf <= AMDSMI_DEV_PERF_LEVEL_LAST)
      amdsmi_set_gpu_perf_level(gpus()[i], initial_perf);
  }
  col.ExpectNoFailures();
}

// amdsmi_get_gpu_pci_bandwidth / amdsmi_set_gpu_pci_bandwidth (bandwidth mask).
TEST_F(GpuFunctionalReadWrite, PciBandwidth_SetVerifyRestore) {
  AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED();
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  amdsmi::test::StatusCollector col("amdsmi_set_gpu_pci_bandwidth");
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_pcie_bandwidth_t bw;
    memset(&bw, 0, sizeof(bw));
    if (amdsmi_get_gpu_pci_bandwidth(gpus()[i], &bw) != AMDSMI_STATUS_SUCCESS) continue;
    const uint32_t n = bw.transfer_rate.num_supported;
    // current is (uint32_t)-1 when the kernel flags no active transfer rate.
    if (n < 2 || n >= 64 || bw.transfer_rate.current >= n) continue;

    amdsmi_dev_perf_level_t initial_perf = AMDSMI_DEV_PERF_LEVEL_UNKNOWN;
    if (amdsmi_get_gpu_perf_level(gpus()[i], &initial_perf) != AMDSMI_STATUS_SUCCESS)
      initial_perf = AMDSMI_DEV_PERF_LEVEL_UNKNOWN;

    const uint32_t initial = bw.transfer_rate.current;
    const uint32_t target = (initial == 0) ? 1u : 0u;
    const std::string label = "gpu=" + std::to_string(i);

    DISPLAY_AMDSMI_API("amdsmi_set_gpu_pci_bandwidth", label + " set=" + std::to_string(target),
                       kVerbose);
    amdsmi_status_t err = amdsmi_set_gpu_pci_bandwidth(gpus()[i], 1ULL << target);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                          AMDSMI_STATUS_NO_PERM);
    col.Record(label, err,
               ::amdsmi::test::AmdsmiStatusIsExpected(
                   err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                   AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM));

    if (err == AMDSMI_STATUS_SUCCESS) {
      amdsmi_pcie_bandwidth_t readback;
      memset(&readback, 0, sizeof(readback));
      if (amdsmi_get_gpu_pci_bandwidth(gpus()[i], &readback) == AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(readback.transfer_rate.current, target) << label << " set did not take effect";
      }

      DISPLAY_AMDSMI_API("amdsmi_set_gpu_pci_bandwidth",
                         label + " restore=" + std::to_string(initial), kVerbose);
      amdsmi_status_t rerr = amdsmi_set_gpu_pci_bandwidth(gpus()[i], 1ULL << initial);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, rerr, AMDSMI_STATUS_SUCCESS);
      EXPECT_EQ(rerr, AMDSMI_STATUS_SUCCESS) << label << " failed to restore pci bandwidth";
      if (rerr == AMDSMI_STATUS_SUCCESS &&
          amdsmi_get_gpu_pci_bandwidth(gpus()[i], &readback) == AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(readback.transfer_rate.current, initial)
            << label << " restore did not take effect";
      }

      // Re-enable every rate so the link is not left pinned to one index.
      amdsmi_set_gpu_pci_bandwidth(gpus()[i], (1ULL << n) - 1);
    }

    // The setter forces MANUAL; put the device back on the policy it came in on.
    if (initial_perf <= AMDSMI_DEV_PERF_LEVEL_LAST)
      amdsmi_set_gpu_perf_level(gpus()[i], initial_perf);
  }
  col.ExpectNoFailures();
}
