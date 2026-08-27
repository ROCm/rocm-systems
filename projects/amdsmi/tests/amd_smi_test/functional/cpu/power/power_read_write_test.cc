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
#include <vector>

#include "api_test_framework.h"

using amdsmi::test::AmdsmiStatusIsExpected;
using amdsmi::test::kInvalidHandle;
using amdsmi::test::kVerbose;

// amdsmi_get_cpu_socket_power_cap / amdsmi_set_cpu_socket_power_cap.
// amdsmi_get_cpu_pwr_efficiency_mode / amdsmi_set_cpu_pwr_efficiency_mode.
// amdsmi_get_cpu_sdps_limit / amdsmi_set_cpu_sdps_limit.
// amdsmi_set_cpu_socket_boostlimit (no socket getter; reads back per core).
TEST_F(CpuFunctionalReadWrite, SocketPowerCap_SetVerifyRestore) {
  AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED();
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  amdsmi::test::StatusCollector col("amdsmi_set_cpu_socket_power_cap");
  for (size_t i = 0; i < cpus().size(); ++i) {
    uint32_t initial = 0;
    if (amdsmi_get_cpu_socket_power_cap(cpus()[i], &initial) != AMDSMI_STATUS_SUCCESS) continue;
    uint32_t cap_max = 0;
    if (amdsmi_get_cpu_socket_power_cap_max(cpus()[i], &cap_max) != AMDSMI_STATUS_SUCCESS) continue;

    // Pick a target strictly inside (0, cap_max] and different from initial.
    uint32_t target = (cap_max > 1) ? (cap_max / 2) : cap_max;
    if (target == initial) target = (initial > 0) ? (initial - 1) : (initial + 1);

    DISPLAY_AMDSMI_API("amdsmi_set_cpu_socket_power_cap",
                       "cpu=" + std::to_string(i) + " set=" + std::to_string(target), kVerbose);
    amdsmi_status_t err = amdsmi_set_cpu_socket_power_cap(cpus()[i], target);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                          AMDSMI_STATUS_NO_PERM, AMDSMI_STATUS_NO_HSMP_MSG_SUP);
    col.Record("cpu=" + std::to_string(i), err,
               AmdsmiStatusIsExpected(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                                      AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM,
                                      AMDSMI_STATUS_NO_HSMP_MSG_SUP));

    if (err == AMDSMI_STATUS_SUCCESS) {
      uint32_t readback = 0;
      if (amdsmi_get_cpu_socket_power_cap(cpus()[i], &readback) == AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(readback, target) << "cpu=" << i << " set did not take effect";
      }
      amdsmi_status_t rerr = amdsmi_set_cpu_socket_power_cap(cpus()[i], initial);
      EXPECT_EQ(rerr, AMDSMI_STATUS_SUCCESS) << "cpu=" << i << " failed to restore power cap";
      if (rerr == AMDSMI_STATUS_SUCCESS &&
          amdsmi_get_cpu_socket_power_cap(cpus()[i], &readback) == AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(readback, initial) << "cpu=" << i << " restore did not take effect";
      }
    }
  }
  col.ExpectNoFailures();
}

TEST_F(CpuFunctionalReadWrite, PwrEfficiencyMode_SetVerifyRestore) {
  AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED();
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  amdsmi::test::StatusCollector col("amdsmi_set_cpu_pwr_efficiency_mode");
  for (size_t i = 0; i < cpus().size(); ++i) {
    uint32_t initial = 0;
    uint32_t util = 0;
    uint32_t ppt = 0;
    if (amdsmi_get_cpu_pwr_efficiency_mode(cpus()[i], &initial, &util, &ppt) !=
        AMDSMI_STATUS_SUCCESS)
      continue;

    // Modes are small enumerators; pick a different one from current.
    uint8_t target = (initial == 0) ? 1 : 0;
    DISPLAY_AMDSMI_API("amdsmi_set_cpu_pwr_efficiency_mode",
                       "cpu=" + std::to_string(i) + " set=" + std::to_string(target), kVerbose);
    amdsmi_status_t err = amdsmi_set_cpu_pwr_efficiency_mode(cpus()[i], target, &util, &ppt);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                          AMDSMI_STATUS_NO_PERM, AMDSMI_STATUS_NO_HSMP_MSG_SUP);
    col.Record("cpu=" + std::to_string(i), err,
               AmdsmiStatusIsExpected(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                                      AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM,
                                      AMDSMI_STATUS_NO_HSMP_MSG_SUP));

    if (err == AMDSMI_STATUS_SUCCESS) {
      uint32_t readback = 0;
      if (amdsmi_get_cpu_pwr_efficiency_mode(cpus()[i], &readback, &util, &ppt) ==
          AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(readback, static_cast<uint32_t>(target))
            << "cpu=" << i << " set did not take effect";
      }
      amdsmi_status_t rerr =
          amdsmi_set_cpu_pwr_efficiency_mode(cpus()[i], static_cast<uint8_t>(initial), &util, &ppt);
      EXPECT_EQ(rerr, AMDSMI_STATUS_SUCCESS) << "cpu=" << i << " failed to restore efficiency mode";
      if (rerr == AMDSMI_STATUS_SUCCESS &&
          amdsmi_get_cpu_pwr_efficiency_mode(cpus()[i], &readback, &util, &ppt) ==
              AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(readback, initial) << "cpu=" << i << " restore did not take effect";
      }
    }
  }
  col.ExpectNoFailures();
}

TEST_F(CpuFunctionalReadWrite, SdpsLimit_SetVerifyRestore) {
  AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED();
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  amdsmi::test::StatusCollector col("amdsmi_set_cpu_sdps_limit");
  for (size_t i = 0; i < cpus().size(); ++i) {
    uint32_t initial = 0;
    if (amdsmi_get_cpu_sdps_limit(cpus()[i], &initial) != AMDSMI_STATUS_SUCCESS) continue;

    uint32_t target = (initial > 0) ? (initial - 1) : (initial + 1);
    DISPLAY_AMDSMI_API("amdsmi_set_cpu_sdps_limit",
                       "cpu=" + std::to_string(i) + " set=" + std::to_string(target), kVerbose);
    amdsmi_status_t err = amdsmi_set_cpu_sdps_limit(cpus()[i], target);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                          AMDSMI_STATUS_NO_PERM, AMDSMI_STATUS_NO_HSMP_MSG_SUP);
    col.Record("cpu=" + std::to_string(i), err,
               AmdsmiStatusIsExpected(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                                      AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM,
                                      AMDSMI_STATUS_NO_HSMP_MSG_SUP));

    if (err == AMDSMI_STATUS_SUCCESS) {
      uint32_t readback = 0;
      if (amdsmi_get_cpu_sdps_limit(cpus()[i], &readback) == AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(readback, target) << "cpu=" << i << " set did not take effect";
      }
      amdsmi_status_t rerr = amdsmi_set_cpu_sdps_limit(cpus()[i], initial);
      EXPECT_EQ(rerr, AMDSMI_STATUS_SUCCESS) << "cpu=" << i << " failed to restore sdps limit";
      if (rerr == AMDSMI_STATUS_SUCCESS &&
          amdsmi_get_cpu_sdps_limit(cpus()[i], &readback) == AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(readback, initial) << "cpu=" << i << " restore did not take effect";
      }
    }
  }
  col.ExpectNoFailures();
}

// There is no socket-level boostlimit getter; the per-core getter reports the
// limit a socket write enforces, and attributing the enumerated cores to a
// socket is only unambiguous when the host has exactly one.
TEST_F(CpuFunctionalReadWrite, SocketBoostlimit_SetVerifyRestore) {
  AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED();
  if (cpus().empty()) GTEST_SKIP() << "No CPU processors";
  if (cpus().size() != 1 || cpu_cores().empty())
    GTEST_SKIP() << "socket boostlimit readback needs a single socket and its cores";

  std::vector<uint32_t> initial(cpu_cores().size(), 0);
  for (size_t c = 0; c < cpu_cores().size(); ++c) {
    if (amdsmi_get_cpu_core_boostlimit(cpu_cores()[c], &initial[c]) != AMDSMI_STATUS_SUCCESS)
      GTEST_SKIP() << "core boostlimit unreadable, so the socket write cannot be verified";
  }
  uint32_t target = (initial[0] > 100) ? (initial[0] - 100) : (initial[0] + 100);

  amdsmi::test::StatusCollector col("amdsmi_set_cpu_socket_boostlimit");
  DISPLAY_AMDSMI_API("amdsmi_set_cpu_socket_boostlimit", "cpu=0 set=" + std::to_string(target),
                     kVerbose);
  amdsmi_status_t err = amdsmi_set_cpu_socket_boostlimit(cpus()[0], target);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                        AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NO_PERM,
                        AMDSMI_STATUS_NO_HSMP_MSG_SUP, AMDSMI_STATUS_INVAL);
  col.Record("cpu=0", err,
             AmdsmiStatusIsExpected(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                                    AMDSMI_STATUS_NO_PERM, AMDSMI_STATUS_NO_HSMP_MSG_SUP,
                                    AMDSMI_STATUS_INVAL));

  if (err == AMDSMI_STATUS_SUCCESS) {
    for (size_t c = 0; c < cpu_cores().size(); ++c) {
      uint32_t readback = 0;
      if (amdsmi_get_cpu_core_boostlimit(cpu_cores()[c], &readback) == AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(readback, target) << "core=" << c << " socket set did not take effect";
      }
    }
    for (size_t c = 0; c < cpu_cores().size(); ++c) {
      amdsmi_status_t rerr = amdsmi_set_cpu_core_boostlimit(cpu_cores()[c], initial[c]);
      EXPECT_EQ(rerr, AMDSMI_STATUS_SUCCESS) << "core=" << c << " failed to restore boostlimit";
      uint32_t readback = 0;
      if (rerr == AMDSMI_STATUS_SUCCESS &&
          amdsmi_get_cpu_core_boostlimit(cpu_cores()[c], &readback) == AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(readback, initial[c]) << "core=" << c << " restore did not take effect";
      }
    }
  }
  col.ExpectNoFailures();
}
