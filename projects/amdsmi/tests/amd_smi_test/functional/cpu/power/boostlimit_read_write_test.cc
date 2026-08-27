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

using amdsmi::test::AmdsmiStatusIsExpected;
using amdsmi::test::kInvalidHandle;
using amdsmi::test::kVerbose;

// amdsmi_get_cpu_core_boostlimit / amdsmi_set_cpu_core_boostlimit.
// amdsmi_get_cpu_core_floor_freq_limit / amdsmi_set_cpu_core_floor_freq_limit.
// amdsmi_get_cpu_floor_freq_limit / amdsmi_set_cpu_floor_freq_limit.
// MSR floor-frequency limit setters (no getter).
// ---- amdsmi_set_cpu_core_boostlimit (per core) ----
TEST_F(CpuFunctionalReadWrite, SetCoreBoostlimit_InvalidHandle) {
  RequireInit();
  DISPLAY_AMDSMI_API("amdsmi_set_cpu_core_boostlimit", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_cpu_core_boostlimit(kInvalidHandle, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

TEST_F(CpuFunctionalReadWrite, CoreBoostlimit_SetVerifyRestore) {
  AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED();
  if (cpu_cores().empty()) GTEST_SKIP() << "No CPU cores";
  amdsmi::test::StatusCollector col("amdsmi_set_cpu_core_boostlimit");
  for (size_t i = 0; i < cpu_cores().size(); ++i) {
    uint32_t initial = 0;
    if (amdsmi_get_cpu_core_boostlimit(cpu_cores()[i], &initial) != AMDSMI_STATUS_SUCCESS) continue;

    // Nudge by a small delta within a plausible range.
    uint32_t target = (initial > 100) ? (initial - 100) : (initial + 100);
    DISPLAY_AMDSMI_API("amdsmi_set_cpu_core_boostlimit",
                       "core=" + std::to_string(i) + " set=" + std::to_string(target), kVerbose);
    amdsmi_status_t err = amdsmi_set_cpu_core_boostlimit(cpu_cores()[i], target);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                          AMDSMI_STATUS_NO_PERM, AMDSMI_STATUS_NO_HSMP_MSG_SUP);
    col.Record("core=" + std::to_string(i), err,
               AmdsmiStatusIsExpected(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                                      AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM,
                                      AMDSMI_STATUS_NO_HSMP_MSG_SUP));

    if (err == AMDSMI_STATUS_SUCCESS) {
      uint32_t readback = 0;
      if (amdsmi_get_cpu_core_boostlimit(cpu_cores()[i], &readback) == AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(readback, target) << "core=" << i << " set did not take effect";
      }
      amdsmi_status_t rerr = amdsmi_set_cpu_core_boostlimit(cpu_cores()[i], initial);
      EXPECT_EQ(rerr, AMDSMI_STATUS_SUCCESS) << "core=" << i << " failed to restore boostlimit";
      if (rerr == AMDSMI_STATUS_SUCCESS &&
          amdsmi_get_cpu_core_boostlimit(cpu_cores()[i], &readback) == AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(readback, initial) << "core=" << i << " restore did not take effect";
      }
    }
  }
  col.ExpectNoFailures();
}

// ---- floor-frequency limit setters ----
TEST_F(CpuFunctionalReadWrite, SetCoreFloorFreqLimit_InvalidHandle) {
  RequireInit();
  DISPLAY_AMDSMI_API("amdsmi_set_cpu_core_floor_freq_limit", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_cpu_core_floor_freq_limit(kInvalidHandle, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

TEST_F(CpuFunctionalReadWrite, SetFloorFreqLimit_InvalidHandle) {
  RequireInit();
  DISPLAY_AMDSMI_API("amdsmi_set_cpu_floor_freq_limit", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_cpu_floor_freq_limit(kInvalidHandle, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

TEST_F(CpuFunctionalReadWrite, SetMsrFloorFreqLimit_InvalidHandle) {
  RequireInit();
  DISPLAY_AMDSMI_API("amdsmi_set_cpu_msr_floor_freq_limit", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_cpu_msr_floor_freq_limit(kInvalidHandle, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

TEST_F(CpuFunctionalReadWrite, SetCoreMsrFloorFreqLimit_InvalidHandle) {
  RequireInit();
  DISPLAY_AMDSMI_API("amdsmi_set_cpu_core_msr_floor_freq_limit", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_cpu_core_msr_floor_freq_limit(kInvalidHandle, 0);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

TEST_F(CpuFunctionalReadWrite, FloorFreqLimit_SetVerifyRestore) {
  AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED();
  if (cpu_cores().empty() && cpus().empty()) GTEST_SKIP() << "No CPU processors or cores";
  // A floor of 0 clears the limit, so alternating with the socket minimum gives a
  // target that always differs from the current setting and stays in range.
  uint32_t fmax = 0;
  uint32_t fmin = 0;
  if (amdsmi_get_cpu_freq_range(&fmax, &fmin) != AMDSMI_STATUS_SUCCESS || fmin == 0)
    GTEST_SKIP() << "No socket frequency range to pick a floor from";

  amdsmi::test::StatusCollector col("amdsmi_set_cpu_floor_freq_limits");
  for (size_t i = 0; i < cpu_cores().size(); ++i) {
    uint32_t initial = 0;
    if (amdsmi_get_cpu_core_floor_freq_limit(cpu_cores()[i], &initial) != AMDSMI_STATUS_SUCCESS)
      continue;
    uint32_t target = (initial == 0) ? fmin : 0;
    if (target == initial) continue;

    DISPLAY_AMDSMI_API("amdsmi_set_cpu_core_floor_freq_limit",
                       "core=" + std::to_string(i) + " set=" + std::to_string(target), kVerbose);
    amdsmi_status_t err = amdsmi_set_cpu_core_floor_freq_limit(cpu_cores()[i], target);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NO_PERM,
                          AMDSMI_STATUS_NO_HSMP_MSG_SUP, AMDSMI_STATUS_INVAL);
    col.Record("core_floor core=" + std::to_string(i), err,
               AmdsmiStatusIsExpected(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                                      AMDSMI_STATUS_NO_PERM, AMDSMI_STATUS_NO_HSMP_MSG_SUP,
                                      AMDSMI_STATUS_INVAL));

    if (err == AMDSMI_STATUS_SUCCESS) {
      uint32_t readback = 0;
      if (amdsmi_get_cpu_core_floor_freq_limit(cpu_cores()[i], &readback) ==
          AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(readback, target) << "core=" << i << " set did not take effect";
      }
      amdsmi_status_t rerr = amdsmi_set_cpu_core_floor_freq_limit(cpu_cores()[i], initial);
      EXPECT_EQ(rerr, AMDSMI_STATUS_SUCCESS) << "core=" << i << " failed to restore core floor";
      if (rerr == AMDSMI_STATUS_SUCCESS &&
          amdsmi_get_cpu_core_floor_freq_limit(cpu_cores()[i], &readback) ==
              AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(readback, initial) << "core=" << i << " restore did not take effect";
      }
    }
  }

  for (size_t i = 0; i < cpus().size(); ++i) {
    uint32_t initial = 0;
    if (amdsmi_get_cpu_floor_freq_limit(cpus()[i], &initial) != AMDSMI_STATUS_SUCCESS) continue;
    uint32_t target = (initial == 0) ? fmin : 0;
    if (target == initial) continue;

    DISPLAY_AMDSMI_API("amdsmi_set_cpu_floor_freq_limit",
                       "cpu=" + std::to_string(i) + " set=" + std::to_string(target), kVerbose);
    amdsmi_status_t err = amdsmi_set_cpu_floor_freq_limit(cpus()[i], target);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NO_PERM,
                          AMDSMI_STATUS_NO_HSMP_MSG_SUP, AMDSMI_STATUS_INVAL);
    col.Record("floor cpu=" + std::to_string(i), err,
               AmdsmiStatusIsExpected(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                                      AMDSMI_STATUS_NO_PERM, AMDSMI_STATUS_NO_HSMP_MSG_SUP,
                                      AMDSMI_STATUS_INVAL));

    if (err == AMDSMI_STATUS_SUCCESS) {
      uint32_t readback = 0;
      if (amdsmi_get_cpu_floor_freq_limit(cpus()[i], &readback) == AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(readback, target) << "cpu=" << i << " set did not take effect";
      }
      amdsmi_status_t rerr = amdsmi_set_cpu_floor_freq_limit(cpus()[i], initial);
      EXPECT_EQ(rerr, AMDSMI_STATUS_SUCCESS) << "cpu=" << i << " failed to restore socket floor";
      if (rerr == AMDSMI_STATUS_SUCCESS &&
          amdsmi_get_cpu_floor_freq_limit(cpus()[i], &readback) == AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(readback, initial) << "cpu=" << i << " restore did not take effect";
      }
    }
  }
  col.ExpectNoFailures();
}

// Both MSR floor setters are write-only -- AMD SMI exposes no MSR floor getter,
// so there is nothing to read back and compare against the value written.
// Write-only: the *_eff_floor_* getters report the effective floor, not the MSR
// value written, so there is nothing to read back and verify against the input.
TEST_F(CpuFunctionalReadWrite, MsrFloorFreqLimit_Set) {
  AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED();
  if (cpu_cores().empty() && cpus().empty()) GTEST_SKIP() << "No CPU processors or cores";
  amdsmi::test::StatusCollector col("amdsmi_set_cpu_msr_floor_freq_limits");
  for (size_t i = 0; i < cpu_cores().size(); ++i) {
    DISPLAY_AMDSMI_API("amdsmi_set_cpu_core_msr_floor_freq_limit", "core=" + std::to_string(i),
                       kVerbose);
    amdsmi_status_t err = amdsmi_set_cpu_core_msr_floor_freq_limit(cpu_cores()[i], 0);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NO_PERM,
                          AMDSMI_STATUS_NO_HSMP_MSG_SUP, AMDSMI_STATUS_INVAL);
    col.Record("core_msr_floor core=" + std::to_string(i), err,
               AmdsmiStatusIsExpected(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                                      AMDSMI_STATUS_NO_PERM, AMDSMI_STATUS_NO_HSMP_MSG_SUP,
                                      AMDSMI_STATUS_INVAL));
  }
  for (size_t i = 0; i < cpus().size(); ++i) {
    DISPLAY_AMDSMI_API("amdsmi_set_cpu_msr_floor_freq_limit", "cpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_set_cpu_msr_floor_freq_limit(cpus()[i], 0);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NO_PERM,
                          AMDSMI_STATUS_NO_HSMP_MSG_SUP, AMDSMI_STATUS_INVAL);
    col.Record("msr_floor cpu=" + std::to_string(i), err,
               AmdsmiStatusIsExpected(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                                      AMDSMI_STATUS_NO_PERM, AMDSMI_STATUS_NO_HSMP_MSG_SUP,
                                      AMDSMI_STATUS_INVAL));
  }
  col.ExpectNoFailures();
}
