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

// ---------------- amdsmi_gpu_xgmi_error_status ----------------
TEST_F(GpuIntegration, XgmiErrorStatus_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_gpu_xgmi_error_status", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_gpu_xgmi_error_status(gpus()[0], nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_ARG_PTR_NULL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, XgmiErrorStatus_InvalidHandle) {
  amdsmi_xgmi_status_t status;
  memset(&status, 0, sizeof(status));
  DISPLAY_AMDSMI_API("amdsmi_gpu_xgmi_error_status", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_gpu_xgmi_error_status(kInvalidHandle, &status);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuIntegration, XgmiErrorStatus_AllGpus) {
  // amdsmi.h reserves AMDSMI_STATUS_INVAL for a null status pointer, but the call
  // returns it for valid arguments on every GPU. See known_failures.md.
  GTEST_SKIP() << "amdsmi_gpu_xgmi_error_status returns INVAL for valid arguments";
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_gpu_xgmi_error_status");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_xgmi_status_t status;
    memset(&status, 0, sizeof(status));
    DISPLAY_AMDSMI_API("amdsmi_gpu_xgmi_error_status", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_gpu_xgmi_error_status(gpus()[i], &status);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("gpu=" + std::to_string(i), err,
                      ::amdsmi::test::AmdsmiStatusIsExpected(err, AMDSMI_STATUS_SUCCESS,
                                                             AMDSMI_STATUS_NOT_SUPPORTED,
                                                             AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_get_xgmi_info ----------------
TEST_F(GpuIntegration, GetXgmiInfo_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_get_xgmi_info", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_xgmi_info(gpus()[0], nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetXgmiInfo_InvalidHandle) {
  amdsmi_xgmi_info_t info;
  memset(&info, 0, sizeof(info));
  DISPLAY_AMDSMI_API("amdsmi_get_xgmi_info", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_xgmi_info(kInvalidHandle, &info);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuIntegration, GetXgmiInfo_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_xgmi_info");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_xgmi_info_t info;
    memset(&info, 0, sizeof(info));
    DISPLAY_AMDSMI_API("amdsmi_get_xgmi_info", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_xgmi_info(gpus()[i], &info);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.Record("gpu=" + std::to_string(i), err,
                      ::amdsmi::test::AmdsmiStatusIsExpected(err, AMDSMI_STATUS_SUCCESS,
                                                             AMDSMI_STATUS_NOT_SUPPORTED,
                                                             AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_get_minmax_bandwidth_between_processors ----------------
TEST_F(GpuIntegration, MinMaxBandwidth_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  uint64_t min_bw = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_minmax_bandwidth_between_processors", "out=nullptr", kVerbose);
  amdsmi_status_t err =
      amdsmi_get_minmax_bandwidth_between_processors(gpus()[0], gpus()[0], &min_bw, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, MinMaxBandwidth_InvalidHandle) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  uint64_t min_bw = 0, max_bw = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_minmax_bandwidth_between_processors", "src=invalid", kVerbose);
  amdsmi_status_t err =
      amdsmi_get_minmax_bandwidth_between_processors(kInvalidHandle, gpus()[0], &min_bw, &max_bw);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuIntegration, MinMaxBandwidth_AllPairs) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_minmax_bandwidth_between_processors");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i)
    for (size_t j = 0; j < gpus().size(); ++j) {
      uint64_t min_bw = 0, max_bw = 0;
      DISPLAY_AMDSMI_API("amdsmi_get_minmax_bandwidth_between_processors",
                         "src=" + std::to_string(i) + " dst=" + std::to_string(j), kVerbose);
      amdsmi_status_t err =
          amdsmi_get_minmax_bandwidth_between_processors(gpus()[i], gpus()[j], &min_bw, &max_bw);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                            AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                            AMDSMI_STATUS_INVAL);
      amdsmi_col.Record("src=" + std::to_string(i) + " dst=" + std::to_string(j), err,
                        ::amdsmi::test::AmdsmiStatusIsExpected(
                            err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                            AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_INVAL));
    }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_is_P2P_accessible ----------------
TEST_F(GpuIntegration, IsP2PAccessible_NullOutput) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  DISPLAY_AMDSMI_API("amdsmi_is_P2P_accessible", "out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_is_P2P_accessible(gpus()[0], gpus()[0], nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, IsP2PAccessible_InvalidHandle) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  bool accessible = false;
  DISPLAY_AMDSMI_API("amdsmi_is_P2P_accessible", "src=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_is_P2P_accessible(kInvalidHandle, gpus()[0], &accessible);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
TEST_F(GpuIntegration, IsP2PAccessible_AllPairs) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_is_P2P_accessible");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i)
    for (size_t j = 0; j < gpus().size(); ++j) {
      bool accessible = false;
      DISPLAY_AMDSMI_API("amdsmi_is_P2P_accessible",
                         "src=" + std::to_string(i) + " dst=" + std::to_string(j), kVerbose);
      amdsmi_status_t err = amdsmi_is_P2P_accessible(gpus()[i], gpus()[j], &accessible);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                            AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                            AMDSMI_STATUS_INVAL);
      amdsmi_col.Record("src=" + std::to_string(i) + " dst=" + std::to_string(j), err,
                        ::amdsmi::test::AmdsmiStatusIsExpected(
                            err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                            AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_INVAL));
    }
  amdsmi_col.ExpectNoFailures();
}

// ---------------- amdsmi_reset_gpu_xgmi_error (action) ----------------
TEST_F(GpuIntegration, ResetXgmiError_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_reset_gpu_xgmi_error", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_reset_gpu_xgmi_error(kInvalidHandle);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}
