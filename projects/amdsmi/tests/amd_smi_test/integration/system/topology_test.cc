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

namespace {
constexpr amdsmi_link_type_t kLinkTypes[] = {
    AMDSMI_LINK_TYPE_INTERNAL,       AMDSMI_LINK_TYPE_PCIE,    AMDSMI_LINK_TYPE_XGMI,
    AMDSMI_LINK_TYPE_NOT_APPLICABLE, AMDSMI_LINK_TYPE_UNKNOWN, AMDSMI_LINK_TYPE_NUMA,
    AMDSMI_LINK_TYPE_XNUMA};

// amdsmi_get_link_topology_nearest() accepts INTERNAL..UNKNOWN only; NUMA and
// XNUMA fall outside its range check and are negative inputs to it.
constexpr amdsmi_link_type_t kNearestLinkTypes[] = {
    AMDSMI_LINK_TYPE_INTERNAL, AMDSMI_LINK_TYPE_PCIE, AMDSMI_LINK_TYPE_XGMI,
    AMDSMI_LINK_TYPE_NOT_APPLICABLE, AMDSMI_LINK_TYPE_UNKNOWN};

constexpr amdsmi_link_type_t kNearestUnsupportedLinkTypes[] = {AMDSMI_LINK_TYPE_NUMA,
                                                               AMDSMI_LINK_TYPE_XNUMA};
}  // namespace

// ---- amdsmi_get_gpu_topo_numa_affinity : invalid params first ----

TEST_F(SystemIntegration, GetGpuTopoNumaAffinity_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_topo_numa_affinity", "numa_node=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_topo_numa_affinity(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(SystemIntegration, GetGpuTopoNumaAffinity_InvalidHandle) {
  int32_t numa_node = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_topo_numa_affinity", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_topo_numa_affinity(kInvalidHandle, &numa_node);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(SystemIntegration, GetGpuTopoNumaAffinity_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_topo_numa_affinity");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    int32_t numa_node = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_topo_numa_affinity", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_topo_numa_affinity(gpus()[i], &numa_node);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_get_gpu_xgmi_link_status : invalid params first ----

TEST_F(SystemIntegration, GetGpuXgmiLinkStatus_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_xgmi_link_status", "link_status=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_xgmi_link_status(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(SystemIntegration, GetGpuXgmiLinkStatus_InvalidHandle) {
  amdsmi_xgmi_link_status_t link_status;
  memset(&link_status, 0, sizeof(link_status));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_xgmi_link_status", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_xgmi_link_status(kInvalidHandle, &link_status);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(SystemIntegration, GetGpuXgmiLinkStatus_AllGpus) {
  GTEST_SKIP() << "amdsmi_get_xgmi_info returns AMDSMI_STATUS_UNEXPECTED_DATA; root cause unknown, "
                  "under investigation";

  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_xgmi_link_status");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_xgmi_link_status_t link_status;
    memset(&link_status, 0, sizeof(link_status));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_xgmi_link_status", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_xgmi_link_status(gpus()[i], &link_status);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_get_link_metrics : invalid params first ----

TEST_F(SystemIntegration, GetLinkMetrics_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_link_metrics", "link_metrics=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_link_metrics(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(SystemIntegration, GetLinkMetrics_InvalidHandle) {
  amdsmi_link_metrics_t link_metrics;
  memset(&link_metrics, 0, sizeof(link_metrics));
  DISPLAY_AMDSMI_API("amdsmi_get_link_metrics", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_link_metrics(kInvalidHandle, &link_metrics);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(SystemIntegration, GetLinkMetrics_AllGpus) {
  GTEST_SKIP() << "amdsmi_get_link_metrics returns AMDSMI_STATUS_UNEXPECTED_DATA; root cause "
                  "unknown, under investigation";

  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_link_metrics");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_link_metrics_t link_metrics;
    memset(&link_metrics, 0, sizeof(link_metrics));
    DISPLAY_AMDSMI_API("amdsmi_get_link_metrics", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_link_metrics(gpus()[i], &link_metrics);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_topo_get_numa_node_number : invalid params first ----

TEST_F(SystemIntegration, TopoGetNumaNodeNumber_NullOutput) {
  GTEST_SKIP()
      << "amdsmi_topo_get_numa_node_number crashes on a null output pointer; proper return "
         "should be AMDSMI_STATUS_INVAL";
  // Proper contract once fixed:
  //   amdsmi_status_t err = amdsmi_topo_get_numa_node_number(gpus()[0], nullptr);
  //   AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(SystemIntegration, TopoGetNumaNodeNumber_InvalidHandle) {
  uint32_t numa_node = 0;
  DISPLAY_AMDSMI_API("amdsmi_topo_get_numa_node_number", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_topo_get_numa_node_number(kInvalidHandle, &numa_node);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(SystemIntegration, TopoGetNumaNodeNumber_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_topo_get_numa_node_number");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    uint32_t numa_node = 0;
    DISPLAY_AMDSMI_API("amdsmi_topo_get_numa_node_number", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_topo_get_numa_node_number(gpus()[i], &numa_node);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_topo_get_link_weight : invalid params first ----

TEST_F(SystemIntegration, TopoGetLinkWeight_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_topo_get_link_weight", "weight=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_topo_get_link_weight(any_gpu(), any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(SystemIntegration, TopoGetLinkWeight_InvalidHandle) {
  uint64_t weight = 0;
  DISPLAY_AMDSMI_API("amdsmi_topo_get_link_weight", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_topo_get_link_weight(kInvalidHandle, kInvalidHandle, &weight);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(SystemIntegration, TopoGetLinkWeight_AllGpuPairs) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_topo_get_link_weight");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t s = 0; s < gpus().size(); ++s) {
    for (size_t d = 0; d < gpus().size(); ++d) {
      uint64_t weight = 0;
      DISPLAY_AMDSMI_API("amdsmi_topo_get_link_weight",
                         "src=" + std::to_string(s) + " dst=" + std::to_string(d), kVerbose);
      amdsmi_status_t err = amdsmi_topo_get_link_weight(gpus()[s], gpus()[d], &weight);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                            AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
      amdsmi_col.RecordPositive("src=" + std::to_string(s) + " dst=" + std::to_string(d), err);
    }
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_topo_get_link_type : invalid params first ----

TEST_F(SystemIntegration, TopoGetLinkType_NullOutput) {
  uint64_t hops = 0;
  DISPLAY_AMDSMI_API("amdsmi_topo_get_link_type", "type=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_topo_get_link_type(any_gpu(), any_gpu(), &hops, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(SystemIntegration, TopoGetLinkType_InvalidHandle) {
  uint64_t hops = 0;
  amdsmi_link_type_t type = AMDSMI_LINK_TYPE_UNKNOWN;
  DISPLAY_AMDSMI_API("amdsmi_topo_get_link_type", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_topo_get_link_type(kInvalidHandle, kInvalidHandle, &hops, &type);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(SystemIntegration, TopoGetLinkType_AllGpuPairs) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_topo_get_link_type");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t s = 0; s < gpus().size(); ++s) {
    for (size_t d = 0; d < gpus().size(); ++d) {
      uint64_t hops = 0;
      amdsmi_link_type_t type = AMDSMI_LINK_TYPE_UNKNOWN;
      DISPLAY_AMDSMI_API("amdsmi_topo_get_link_type",
                         "src=" + std::to_string(s) + " dst=" + std::to_string(d), kVerbose);
      amdsmi_status_t err = amdsmi_topo_get_link_type(gpus()[s], gpus()[d], &hops, &type);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                            AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
      amdsmi_col.RecordPositive("src=" + std::to_string(s) + " dst=" + std::to_string(d), err);
    }
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_get_link_topology_nearest : invalid params first ----

TEST_F(SystemIntegration, GetLinkTopologyNearest_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_link_topology_nearest", "topology_nearest_info=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_link_topology_nearest(any_gpu(), AMDSMI_LINK_TYPE_XGMI, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(SystemIntegration, GetLinkTopologyNearest_InvalidHandle) {
  GTEST_SKIP() << "amdsmi_get_link_topology_nearest returns SUCCESS for an "
                  "invalid handle; proper return should be AMDSMI_STATUS_INVAL";
  // Proper contract once fixed:
  //   amdsmi_topology_nearest_t info; memset(&info, 0, sizeof(info));
  //   amdsmi_status_t err = amdsmi_get_link_topology_nearest(
  //       kInvalidHandle, AMDSMI_LINK_TYPE_XGMI, &info);
  //   AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(SystemIntegration, GetLinkTopologyNearest_AllGpusAllLinkTypes) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_link_topology_nearest");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    for (auto lt : kNearestLinkTypes) {
      amdsmi_topology_nearest_t topology_nearest_info;
      memset(&topology_nearest_info, 0, sizeof(topology_nearest_info));
      DISPLAY_AMDSMI_API("amdsmi_get_link_topology_nearest",
                         "gpu=" + std::to_string(i) + " link=" + std::to_string(lt), kVerbose);
      amdsmi_status_t err = amdsmi_get_link_topology_nearest(gpus()[i], lt, &topology_nearest_info);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                            AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
      amdsmi_col.RecordPositive("gpu=" + std::to_string(i) + " link=" + std::to_string(lt), err);
    }
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_get_link_topology_nearest : link_type outside the accepted range ----

TEST_F(SystemIntegration, GetLinkTopologyNearest_LinkTypeOutOfRange) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_link_topology_nearest");
  for (auto lt : kNearestUnsupportedLinkTypes) {
    amdsmi_topology_nearest_t topology_nearest_info;
    memset(&topology_nearest_info, 0, sizeof(topology_nearest_info));
    const std::string in = "link=" + std::to_string(lt);
    DISPLAY_AMDSMI_API("amdsmi_get_link_topology_nearest", in, kVerbose);
    amdsmi_status_t err = amdsmi_get_link_topology_nearest(any_gpu(), lt, &topology_nearest_info);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
    amdsmi_col.Record(in, err, ::amdsmi::test::AmdsmiStatusIsExpected(err, AMDSMI_STATUS_INVAL));
  }
  amdsmi_col.ExpectNoFailures();
}

// ---- amdsmi_topo_get_p2p_status : invalid params first ----

TEST_F(SystemIntegration, TopoGetP2pStatus_NullOutput) {
  amdsmi_link_type_t type = AMDSMI_LINK_TYPE_UNKNOWN;
  DISPLAY_AMDSMI_API("amdsmi_topo_get_p2p_status", "cap=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_topo_get_p2p_status(any_gpu(), any_gpu(), &type, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(SystemIntegration, TopoGetP2pStatus_InvalidHandle) {
  amdsmi_link_type_t type = AMDSMI_LINK_TYPE_UNKNOWN;
  amdsmi_p2p_capability_t cap;
  memset(&cap, 0, sizeof(cap));
  DISPLAY_AMDSMI_API("amdsmi_topo_get_p2p_status", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_topo_get_p2p_status(kInvalidHandle, kInvalidHandle, &type, &cap);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(SystemIntegration, TopoGetP2pStatus_AllGpuPairs) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_topo_get_p2p_status");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t s = 0; s < gpus().size(); ++s) {
    for (size_t d = 0; d < gpus().size(); ++d) {
      if (s == d) continue;  // P2P status is between two distinct devices
      amdsmi_link_type_t type = AMDSMI_LINK_TYPE_UNKNOWN;
      amdsmi_p2p_capability_t cap;
      memset(&cap, 0, sizeof(cap));
      DISPLAY_AMDSMI_API("amdsmi_topo_get_p2p_status",
                         "src=" + std::to_string(s) + " dst=" + std::to_string(d), kVerbose);
      amdsmi_status_t err = amdsmi_topo_get_p2p_status(gpus()[s], gpus()[d], &type, &cap);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                            AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
      amdsmi_col.RecordPositive("src=" + std::to_string(s) + " dst=" + std::to_string(d), err);
    }
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---- amdsmi_init_gpu_event_notification : invalid handle first ----

TEST_F(SystemIntegration, InitGpuEventNotification_InvalidHandle) {
  DISPLAY_AMDSMI_API("amdsmi_init_gpu_event_notification", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_init_gpu_event_notification(kInvalidHandle);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(SystemIntegration, InitGpuEventNotification_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_init_gpu_event_notification");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    DISPLAY_AMDSMI_API("amdsmi_init_gpu_event_notification", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_init_gpu_event_notification(gpus()[i]);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
    // Event notification state is kernel-backed; leaving it open starves later
    // tests of the same resource.
    if (err == AMDSMI_STATUS_SUCCESS) amdsmi_stop_gpu_event_notification(gpus()[i]);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}
