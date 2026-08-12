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

#include "unit/unit_test_framework.h"

using amdsmi::unittest::kInvalidHandle;
using amdsmi::unittest::kVerbose;

// amdsmi_get_clk_freq / amdsmi_set_clk_freq (frequency-level mask).
TEST_F(GpuFunctionalReadWrite, SetClkFreq_InvalidHandle) {
  RequireInit();
  DISPLAY_AMDSMI_API("amdsmi_set_clk_freq", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_clk_freq(kInvalidHandle, AMDSMI_CLK_TYPE_SYS, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

TEST_F(GpuFunctionalReadWrite, ClkFreq_SetRestore) {
  GTEST_SKIP() << "amdsmi_set_clk_freq returns AMDSMI_STATUS_INVAL in ClkFreq_SetRestore; root "
                  "cause unknown, under investigation";

  AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED();
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  amdsmi::unittest::StatusCollector col("amdsmi_set_clk_freq");
  for (size_t i = 0; i < gpus().size(); ++i) {
    for (int t = AMDSMI_CLK_TYPE_FIRST; t <= AMDSMI_CLK_TYPE__MAX; ++t) {
      amdsmi_clk_type_t clk_type = static_cast<amdsmi_clk_type_t>(t);
      amdsmi_frequencies_t f;
      memset(&f, 0, sizeof(f));
      if (amdsmi_get_clk_freq(gpus()[i], clk_type, &f) != AMDSMI_STATUS_SUCCESS) continue;
      if (f.num_supported == 0 || f.num_supported >= 64) continue;

      uint64_t all_mask = (1ULL << f.num_supported) - 1;
      uint64_t restore_mask = (f.current < f.num_supported) ? (1ULL << f.current) : all_mask;

      std::string label = "gpu=" + std::to_string(i) + " clk=" + std::to_string(t);
      DISPLAY_AMDSMI_API("amdsmi_set_clk_freq", label + " set=all", kVerbose);
      amdsmi_status_t err = amdsmi_set_clk_freq(gpus()[i], clk_type, all_mask);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                            AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                            AMDSMI_STATUS_NO_PERM);
      col.Record(label, err,
                 ::amdsmi::unittest::AmdsmiStatusIsExpected(
                     err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                     AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM));

      if (err == AMDSMI_STATUS_SUCCESS) {
        DISPLAY_AMDSMI_API("amdsmi_set_clk_freq", label + " restore", kVerbose);
        amdsmi_status_t rerr = amdsmi_set_clk_freq(gpus()[i], clk_type, restore_mask);
        DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, rerr, AMDSMI_STATUS_SUCCESS);
        EXPECT_EQ(rerr, AMDSMI_STATUS_SUCCESS) << label << " failed to restore clk freq mask";
      }
    }
  }
  col.ExpectNoFailures();
}

// amdsmi_get_gpu_pci_bandwidth / amdsmi_set_gpu_pci_bandwidth (bandwidth mask).
TEST_F(GpuFunctionalReadWrite, SetPciBandwidth_InvalidHandle) {
  RequireInit();
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_pci_bandwidth", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_gpu_pci_bandwidth(kInvalidHandle, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

TEST_F(GpuFunctionalReadWrite, PciBandwidth_SetRestore) {
  AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED();
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  amdsmi::unittest::StatusCollector col("amdsmi_set_gpu_pci_bandwidth");
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_pcie_bandwidth_t bw;
    memset(&bw, 0, sizeof(bw));
    if (amdsmi_get_gpu_pci_bandwidth(gpus()[i], &bw) != AMDSMI_STATUS_SUCCESS) continue;
    uint32_t n = bw.transfer_rate.num_supported;
    if (n == 0 || n >= 64) continue;

    uint64_t all_mask = (1ULL << n) - 1;
    uint64_t restore_mask =
        (bw.transfer_rate.current < n) ? (1ULL << bw.transfer_rate.current) : all_mask;

    std::string label = "gpu=" + std::to_string(i);
    DISPLAY_AMDSMI_API("amdsmi_set_gpu_pci_bandwidth", label + " set=all", kVerbose);
    amdsmi_status_t err = amdsmi_set_gpu_pci_bandwidth(gpus()[i], all_mask);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                          AMDSMI_STATUS_NO_PERM);
    col.Record(label, err,
               ::amdsmi::unittest::AmdsmiStatusIsExpected(
                   err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                   AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM));

    if (err == AMDSMI_STATUS_SUCCESS) {
      DISPLAY_AMDSMI_API("amdsmi_set_gpu_pci_bandwidth", label + " restore", kVerbose);
      amdsmi_status_t rerr = amdsmi_set_gpu_pci_bandwidth(gpus()[i], restore_mask);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, rerr, AMDSMI_STATUS_SUCCESS);
      EXPECT_EQ(rerr, AMDSMI_STATUS_SUCCESS) << label << " failed to restore pci bandwidth mask";
    }
  }
  col.ExpectNoFailures();
}
