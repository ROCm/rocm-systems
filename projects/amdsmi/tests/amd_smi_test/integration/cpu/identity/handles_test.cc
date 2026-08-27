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
#include <cstring>
#include <string>
#include <vector>

#include "api_test_framework.h"

using amdsmi::test::kInvalidHandle;
using amdsmi::test::kVerbose;

// Enumeration, identity and socket-count style CPU/ESMI APIs. Several of these
// take no processor handle and the current implementation dereferences the
// output pointer whenever the underlying HSMP call succeeds, so a null-output
// negative test is unsafe on CPU hardware and is intentionally omitted.

// ---- amdsmi_get_cpu_handles ----
TEST_F(CpuIntegration, GetCpuHandles_NullCount) {
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_handles", "count=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_handles(nullptr, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(CpuIntegration, GetCpuHandles_CountThenFill) {
  if (!cpu_supported()) GTEST_SKIP() << "No CPU processors";
  uint32_t count = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_handles", "query count", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_handles(&count, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                        AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
  AMDSMI_EXPECT_STATUS(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                       AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
  if (err != AMDSMI_STATUS_SUCCESS || count == 0) return;
  std::vector<amdsmi_processor_handle> handles(count, nullptr);
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_handles", "fill count=" + std::to_string(count), kVerbose);
  err = amdsmi_get_cpu_handles(&count, handles.data());
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                        AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
  AMDSMI_EXPECT_STATUS(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                       AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
}

// ---- amdsmi_get_cpucore_handles ----
TEST_F(CpuIntegration, GetCpuCoreHandles_NullCount) {
  DISPLAY_AMDSMI_API("amdsmi_get_cpucore_handles", "count=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpucore_handles(nullptr, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(CpuIntegration, GetCpuCoreHandles_CountThenFill) {
  if (!cpu_supported()) GTEST_SKIP() << "No CPU processors";
  uint32_t count = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_cpucore_handles", "query count", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpucore_handles(&count, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                        AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
  AMDSMI_EXPECT_STATUS(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                       AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
  if (err != AMDSMI_STATUS_SUCCESS || count == 0) return;
  std::vector<amdsmi_processor_handle> handles(count, nullptr);
  DISPLAY_AMDSMI_API("amdsmi_get_cpucore_handles", "fill count=" + std::to_string(count), kVerbose);
  err = amdsmi_get_cpucore_handles(&count, handles.data());
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                        AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
  AMDSMI_EXPECT_STATUS(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                       AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
}

// ---- amdsmi_get_cpu_freq_range (output guarded) ----
TEST_F(CpuIntegration, GetCpuFreqRange_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_freq_range", "fmax/fmin=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_freq_range(nullptr, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(CpuIntegration, GetCpuFreqRange_Valid) {
  if (!cpu_supported()) GTEST_SKIP() << "No CPU processors";
  uint32_t fmax = 0, fmin = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_freq_range", "query", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_freq_range(&fmax, &fmin);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                        AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
  AMDSMI_EXPECT_STATUS(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                       AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
}

// ---- no-handle identity APIs (impl lacks null-output guard; valid only) ----
TEST_F(CpuIntegration, GetCpuSocketCount_Valid) {
  if (!cpu_supported()) GTEST_SKIP() << "No CPU processors";
  uint32_t count = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_socket_count", "query", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_socket_count(&count);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                        AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
  AMDSMI_EXPECT_STATUS(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                       AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
}
TEST_F(CpuIntegration, GetThreadsPerCore_Valid) {
  if (!cpu_supported()) GTEST_SKIP() << "No CPU processors";
  uint32_t tpc = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_threads_per_core", "query", kVerbose);
  amdsmi_status_t err = amdsmi_get_threads_per_core(&tpc);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                        AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
  AMDSMI_EXPECT_STATUS(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                       AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
}
TEST_F(CpuIntegration, GetCpuFamily_Valid) {
  if (!cpu_supported()) GTEST_SKIP() << "No CPU processors";
  uint32_t family = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_family", "query", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_family(&family);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                        AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
  AMDSMI_EXPECT_STATUS(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                       AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
}
TEST_F(CpuIntegration, GetCpuModel_Valid) {
  if (!cpu_supported()) GTEST_SKIP() << "No CPU processors";
  uint32_t model = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_model", "query", kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_model(&model);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                        AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
  AMDSMI_EXPECT_STATUS(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                       AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
}
TEST_F(CpuIntegration, GetCpuCoresPerSocket_Valid) {
  if (!cpu_supported()) GTEST_SKIP() << "No CPU processors";
  uint32_t sock_count = 0;
  amdsmi_status_t sc = amdsmi_get_cpu_socket_count(&sock_count);
  if (sc != AMDSMI_STATUS_SUCCESS) sock_count = 1;
  amdsmi_sock_info_t info;
  memset(&info, 0, sizeof(info));
  DISPLAY_AMDSMI_API("amdsmi_get_cpu_cores_per_socket", "sock_count=" + std::to_string(sock_count),
                     kVerbose);
  amdsmi_status_t err = amdsmi_get_cpu_cores_per_socket(sock_count, &info);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                        AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
  AMDSMI_EXPECT_STATUS(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                       AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
}

// ---- amdsmi_get_esmi_err_msg (string lookup) ----
TEST_F(CpuIntegration, GetEsmiErrMsg_Valid) {
  if (!cpu_supported()) GTEST_SKIP() << "No CPU processors";
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_esmi_err_msg");
  static constexpr amdsmi_status_t kStatuses[] = {
      AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_INVAL, AMDSMI_STATUS_NO_HSMP_SUP,
      AMDSMI_STATUS_NO_HSMP_MSG_SUP, AMDSMI_STATUS_HSMP_TIMEOUT};
  for (auto st : kStatuses) {
    const char* msg = nullptr;
    DISPLAY_AMDSMI_API("amdsmi_get_esmi_err_msg", "status=" + std::to_string(st), kVerbose);
    amdsmi_status_t err = amdsmi_get_esmi_err_msg(st, &msg);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("status=" + std::to_string(st), err,
                      ::amdsmi::test::AmdsmiStatusIsExpected(err, AMDSMI_STATUS_SUCCESS,
                                                             AMDSMI_STATUS_NOT_SUPPORTED,
                                                             AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }
  amdsmi_col.ExpectNoFailures();
}
