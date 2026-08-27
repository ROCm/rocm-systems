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

using amdsmi::test::kInvalidHandle;
using amdsmi::test::kVerbose;

namespace {
constexpr amdsmi_socket_handle kInvalidSocket = nullptr;

constexpr amdsmi_processor_type_t kProcessorTypes[] = {
    AMDSMI_PROCESSOR_TYPE_UNKNOWN,     AMDSMI_PROCESSOR_TYPE_AMD_GPU,
    AMDSMI_PROCESSOR_TYPE_AMD_CPU,     AMDSMI_PROCESSOR_TYPE_NON_AMD_GPU,
    AMDSMI_PROCESSOR_TYPE_NON_AMD_CPU, AMDSMI_PROCESSOR_TYPE_AMD_CPU_CORE,
    AMDSMI_PROCESSOR_TYPE_AMD_APU,     AMDSMI_PROCESSOR_TYPE_AMD_NIC,
    AMDSMI_PROCESSOR_TYPE_BRCM_NIC,    AMDSMI_PROCESSOR_TYPE_BRCM_SWITCH};
}  // namespace

// ============================ socket handles ============================

// ---- amdsmi_get_socket_handles : invalid params first ----
TEST_F(SystemIntegration, GetSocketHandles_NullCount) {
  DISPLAY_AMDSMI_API("amdsmi_get_socket_handles", "socket_count=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_socket_handles(nullptr, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(SystemIntegration, GetSocketHandles_CountThenBuffer) {
  uint32_t socket_count = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_socket_handles", "query count", kVerbose);
  amdsmi_status_t err = amdsmi_get_socket_handles(&socket_count, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                        AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
  AMDSMI_EXPECT_STATUS(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                       AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
  if (err != AMDSMI_STATUS_SUCCESS || socket_count == 0) return;
  std::vector<amdsmi_socket_handle> handles(socket_count);
  DISPLAY_AMDSMI_API("amdsmi_get_socket_handles", "count=" + std::to_string(socket_count),
                     kVerbose);
  err = amdsmi_get_socket_handles(&socket_count, handles.data());
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                        AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
  AMDSMI_EXPECT_STATUS(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                       AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
}

// ---- amdsmi_get_socket_info : invalid params first ----

TEST_F(SystemIntegration, GetSocketInfo_NullOutput) {
  if (sockets().empty()) GTEST_SKIP() << "No sockets";
  DISPLAY_AMDSMI_API("amdsmi_get_socket_info", "name=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_socket_info(sockets()[0], 128, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(SystemIntegration, GetSocketInfo_InvalidHandle) {
  char name[128];
  memset(name, 0, sizeof(name));
  DISPLAY_AMDSMI_API("amdsmi_get_socket_info", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_socket_info(kInvalidSocket, sizeof(name), name);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(SystemIntegration, GetSocketInfo_AllSockets) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_socket_info");
  if (sockets().empty()) GTEST_SKIP() << "No sockets";
  for (size_t i = 0; i < sockets().size(); ++i) {
    char name[128];
    memset(name, 0, sizeof(name));
    DISPLAY_AMDSMI_API("amdsmi_get_socket_info", "socket=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_socket_info(sockets()[i], sizeof(name), name);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("socket=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_get_processor_handles : invalid params first ----

TEST_F(SystemIntegration, GetProcessorHandles_NullCount) {
  if (sockets().empty()) GTEST_SKIP() << "No sockets";
  DISPLAY_AMDSMI_API("amdsmi_get_processor_handles", "processor_count=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_processor_handles(sockets()[0], nullptr, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(SystemIntegration, GetProcessorHandles_AllSockets) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_processor_handles");
  if (sockets().empty()) GTEST_SKIP() << "No sockets";
  for (size_t i = 0; i < sockets().size(); ++i) {
    uint32_t processor_count = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_processor_handles",
                       "socket=" + std::to_string(i) + " query count", kVerbose);
    amdsmi_status_t err = amdsmi_get_processor_handles(sockets()[i], &processor_count, nullptr);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("socket=" + std::to_string(i) + " query count", err);
    if (err != AMDSMI_STATUS_SUCCESS || processor_count == 0) continue;
    std::vector<amdsmi_processor_handle> handles(processor_count);
    DISPLAY_AMDSMI_API("amdsmi_get_processor_handles",
                       "socket=" + std::to_string(i) + " count=" + std::to_string(processor_count),
                       kVerbose);
    err = amdsmi_get_processor_handles(sockets()[i], &processor_count, handles.data());
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive(
        "socket=" + std::to_string(i) + " count=" + std::to_string(processor_count), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_get_node_handle : invalid params first ----

TEST_F(SystemIntegration, GetNodeHandle_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_node_handle", "node_handle=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_node_handle(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(SystemIntegration, GetNodeHandle_InvalidHandle) {
  amdsmi_node_handle node_handle = nullptr;
  DISPLAY_AMDSMI_API("amdsmi_get_node_handle", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_node_handle(kInvalidHandle, &node_handle);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(SystemIntegration, GetNodeHandle_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_node_handle");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_node_handle node_handle = nullptr;
    DISPLAY_AMDSMI_API("amdsmi_get_node_handle", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_node_handle(gpus()[i], &node_handle);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_get_processor_type : invalid params first ----

TEST_F(SystemIntegration, GetProcessorType_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_processor_type", "processor_type=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_processor_type(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(SystemIntegration, GetProcessorType_InvalidHandle) {
  amdsmi_processor_type_t type = AMDSMI_PROCESSOR_TYPE_UNKNOWN;
  DISPLAY_AMDSMI_API("amdsmi_get_processor_type", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_processor_type(kInvalidHandle, &type);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(SystemIntegration, GetProcessorType_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_processor_type");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_processor_type_t type = AMDSMI_PROCESSOR_TYPE_UNKNOWN;
    DISPLAY_AMDSMI_API("amdsmi_get_processor_type", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_processor_type(gpus()[i], &type);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_get_processor_info : invalid params first ----

TEST_F(SystemIntegration, GetProcessorInfo_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_processor_info", "name=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_processor_info(any_gpu(), 128, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(SystemIntegration, GetProcessorInfo_InvalidHandle) {
  char name[128];
  memset(name, 0, sizeof(name));
  DISPLAY_AMDSMI_API("amdsmi_get_processor_info", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_processor_info(kInvalidHandle, sizeof(name), name);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(SystemIntegration, GetProcessorInfo_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_processor_info");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    char name[128];
    memset(name, 0, sizeof(name));
    DISPLAY_AMDSMI_API("amdsmi_get_processor_info", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_processor_info(gpus()[i], sizeof(name), name);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_get_processor_count_from_handles : invalid params first ----

TEST_F(SystemIntegration, GetProcessorCountFromHandles_NullCount) {
  std::vector<amdsmi_processor_handle> handles(gpus());
  DISPLAY_AMDSMI_API("amdsmi_get_processor_count_from_handles", "processor_count=nullptr",
                     kVerbose);
  amdsmi_status_t err =
      amdsmi_get_processor_count_from_handles(handles.data(), nullptr, nullptr, nullptr, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(SystemIntegration, GetProcessorCountFromHandles_Valid) {
  std::vector<amdsmi_processor_handle> handles(gpus());
  uint32_t processor_count = static_cast<uint32_t>(handles.size());
  uint32_t nr_cpusockets = 0;
  uint32_t nr_cpucores = 0;
  uint32_t nr_gpus = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_processor_count_from_handles",
                     "count=" + std::to_string(processor_count), kVerbose);
  amdsmi_status_t err = amdsmi_get_processor_count_from_handles(
      handles.data(), &processor_count, &nr_cpusockets, &nr_cpucores, &nr_gpus);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                        AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
  AMDSMI_EXPECT_STATUS(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                       AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
}

// ---- amdsmi_get_processor_handles_by_type : invalid params first ----

TEST_F(SystemIntegration, GetProcessorHandlesByType_NullCount) {
  if (sockets().empty()) GTEST_SKIP() << "No sockets";
  DISPLAY_AMDSMI_API("amdsmi_get_processor_handles_by_type", "processor_count=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_processor_handles_by_type(
      sockets()[0], AMDSMI_PROCESSOR_TYPE_AMD_GPU, nullptr, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(SystemIntegration, GetProcessorHandlesByType_AllSocketsAllTypes) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_processor_handles_by_type");
  if (sockets().empty()) GTEST_SKIP() << "No sockets";
  for (size_t i = 0; i < sockets().size(); ++i) {
    for (auto ptype : kProcessorTypes) {
      uint32_t processor_count = 0;
      DISPLAY_AMDSMI_API("amdsmi_get_processor_handles_by_type",
                         "socket=" + std::to_string(i) + " type=" + std::to_string(ptype),
                         kVerbose);
      amdsmi_status_t err =
          amdsmi_get_processor_handles_by_type(sockets()[i], ptype, nullptr, &processor_count);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                            AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
      amdsmi_col.RecordPositive("socket=" + std::to_string(i) + " type=" + std::to_string(ptype),
                                err);
      if (err != AMDSMI_STATUS_SUCCESS || processor_count == 0) continue;
      std::vector<amdsmi_processor_handle> handles(processor_count);
      DISPLAY_AMDSMI_API("amdsmi_get_processor_handles_by_type",
                         "socket=" + std::to_string(i) + " type=" + std::to_string(ptype) +
                             " count=" + std::to_string(processor_count),
                         kVerbose);
      err = amdsmi_get_processor_handles_by_type(sockets()[i], ptype, handles.data(),
                                                 &processor_count);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                            AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
      amdsmi_col.RecordPositive("socket=" + std::to_string(i) + " type=" + std::to_string(ptype) +
                                    " count=" + std::to_string(processor_count),
                                err);
    }
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_get_processor_handle_from_bdf : invalid params first ----

TEST_F(SystemIntegration, GetProcessorHandleFromBdf_NullOutput) {
  amdsmi_bdf_t bdf;
  memset(&bdf, 0, sizeof(bdf));
  DISPLAY_AMDSMI_API("amdsmi_get_processor_handle_from_bdf", "processor_handle=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_processor_handle_from_bdf(bdf, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(SystemIntegration, GetProcessorHandleFromBdf_ZeroBdf) {
  GTEST_SKIP() << "amdsmi_get_processor_handle_from_bdf returns AMDSMI_STATUS_API_FAILED for zero "
                  "BDF; should return NOT_FOUND or INVAL, library validation bug";

  amdsmi_bdf_t bdf;
  memset(&bdf, 0, sizeof(bdf));
  amdsmi_processor_handle handle = nullptr;
  DISPLAY_AMDSMI_API("amdsmi_get_processor_handle_from_bdf", "bdf=0", kVerbose);
  amdsmi_status_t err = amdsmi_get_processor_handle_from_bdf(bdf, &handle);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                        AMDSMI_STATUS_NOT_FOUND, AMDSMI_STATUS_INVAL, AMDSMI_STATUS_NOT_SUPPORTED,
                        AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
  AMDSMI_EXPECT_STATUS(err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_FOUND, AMDSMI_STATUS_INVAL,
                       AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
}

TEST_F(SystemIntegration, LibVersion_NullOutput) {
  RequireInit();
  DISPLAY_AMDSMI_API("amdsmi_get_lib_version", "version=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_lib_version(nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}

TEST_F(SystemIntegration, LibVersion_Stable) {
  RequireInit();
  amdsmi_version_t a, b;
  memset(&a, 0, sizeof(a));
  memset(&b, 0, sizeof(b));
  DISPLAY_AMDSMI_API("amdsmi_get_lib_version", "read x2", kVerbose);
  amdsmi_status_t e1 = amdsmi_get_lib_version(&a);
  amdsmi_status_t e2 = amdsmi_get_lib_version(&b);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, e1, AMDSMI_STATUS_SUCCESS,
                        AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
  AMDSMI_EXPECT_STATUS(e1, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                       AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
  if (e1 == AMDSMI_STATUS_SUCCESS && e2 == AMDSMI_STATUS_SUCCESS) {
    EXPECT_EQ(a.major, b.major);
    EXPECT_EQ(a.minor, b.minor);
    EXPECT_EQ(a.release, b.release);
  }
}

TEST_F(SystemIntegration, SocketHandles_Stable) {
  RequireInit();
  uint32_t c1 = 0, c2 = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_socket_handles", "count x2", kVerbose);
  amdsmi_status_t e1 = amdsmi_get_socket_handles(&c1, nullptr);
  amdsmi_status_t e2 = amdsmi_get_socket_handles(&c2, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, e1, AMDSMI_STATUS_SUCCESS,
                        AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
  AMDSMI_EXPECT_STATUS(e1, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                       AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
  if (e1 == AMDSMI_STATUS_SUCCESS && e2 == AMDSMI_STATUS_SUCCESS) {
    EXPECT_EQ(c1, c2) << "socket count not stable";
  }
}

TEST_F(SystemIntegration, ProcessorType_Stable) {
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  amdsmi::test::StatusCollector col("amdsmi_get_processor_type");
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_processor_type_t t1 = AMDSMI_PROCESSOR_TYPE_UNKNOWN;
    amdsmi_processor_type_t t2 = AMDSMI_PROCESSOR_TYPE_UNKNOWN;
    DISPLAY_AMDSMI_API("amdsmi_get_processor_type", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t e1 = amdsmi_get_processor_type(gpus()[i], &t1);
    amdsmi_status_t e2 = amdsmi_get_processor_type(gpus()[i], &t2);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, e1, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    col.Record("gpu=" + std::to_string(i), e1,
               ::amdsmi::test::AmdsmiStatusIsExpected(e1, AMDSMI_STATUS_SUCCESS,
                                                      AMDSMI_STATUS_NOT_SUPPORTED,
                                                      AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
    if (e1 == AMDSMI_STATUS_SUCCESS && e2 == AMDSMI_STATUS_SUCCESS) {
      EXPECT_EQ(t1, t2) << "gpu=" << i << " processor type not stable";
    }
  }
  col.ExpectNoFailures();
}
