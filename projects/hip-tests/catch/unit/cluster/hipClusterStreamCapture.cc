/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Cluster dimensions of kernel launches captured into a graph by stream capture.

#include <hip_test_common.hh>
#include <hip/hip_cooperative_groups.h>

#include <vector>

#include "ClusterHelper.hpp"

namespace cg = cooperative_groups;

namespace {

constexpr int kBlocks = 4;
constexpr int kThreads = 64;

// Every block records the cluster shape it runs in.
__global__ void RecordClusterDims(dim3* out) {
  if (threadIdx.x == 0) out[blockIdx.x] = cg::this_cluster().dim_blocks();
}

__global__ void CLUSTER_DIMS(2, 1, 1) RecordClusterDimsCD(dim3* out) {
  if (threadIdx.x == 0) out[blockIdx.x] = cg::this_cluster().dim_blocks();
}

// The cluster dimensions stored on the graph's only kernel node.
dim3 KernelNodeClusterDims(hipGraph_t graph) {
  size_t num_nodes = 0;
  HIP_CHECK(hipGraphGetNodes(graph, nullptr, &num_nodes));
  std::vector<hipGraphNode_t> nodes(num_nodes);
  HIP_CHECK(hipGraphGetNodes(graph, nodes.data(), &num_nodes));
  hipGraphNode_t kernel_node = nullptr;
  for (auto node : nodes) {
    hipGraphNodeType type;
    HIP_CHECK(hipGraphNodeGetType(node, &type));
    if (type == hipGraphNodeTypeKernel) {
      REQUIRE(kernel_node == nullptr);
      kernel_node = node;
    }
  }
  REQUIRE(kernel_node != nullptr);
  hipKernelNodeAttrValue value{};
  HIP_CHECK(
      hipGraphKernelNodeGetAttribute(kernel_node, hipLaunchAttributeClusterDimension, &value));
  return dim3(value.clusterDim.x, value.clusterDim.y, value.clusterDim.z);
}

// Instantiates and launches the graph, then checks the cluster shape every block ran in.
void ReplayAndCheck(hipGraph_t graph, hipStream_t stream, dim3* d_out, dim3 expected) {
  HIP_CHECK(hipMemset(d_out, 0, kBlocks * sizeof(dim3)));
  HIP_CHECK(hipDeviceSynchronize());
  hipGraphExec_t exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  std::vector<dim3> out(kBlocks);
  HIP_CHECK(hipMemcpy(out.data(), d_out, kBlocks * sizeof(dim3), hipMemcpyDeviceToHost));
  for (int b = 0; b < kBlocks; ++b) {
    INFO("block " << b);
    REQUIRE(out[b].x == expected.x);
    REQUIRE(out[b].y == expected.y);
    REQUIRE(out[b].z == expected.z);
  }
  HIP_CHECK(hipGraphExecDestroy(exec));
}

}  // namespace

/**
 * @addtogroup cluster
 * @{
 * @ingroup ClusterTest
 */

/**
 * Test Description
 * ------------------------
 *  - A hipLaunchKernelEx launch with hipLaunchAttributeClusterDimension = {2, 1, 1} is captured
 *    into a graph. The captured kernel node must carry the cluster dimensions, and every block of
 *    the replayed graph must run in a 2x1x1 cluster.
 *  - Without the attribute, the captured node must have no cluster ({1, 1, 1}).
 *
 * Test source
 * ------------------------
 *  - catch/unit/cluster/hipClusterStreamCapture.cc
 *
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */
HIP_TEST_CASE(Unit_hipClusterLaunch_StreamCapture_LaunchAttribute) {
  if (!CheckTargetSupport()) {
    HIP_SKIP_TEST("Cluster launch is not supported on this device");
    return;
  }
  dim3* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_out, kBlocks * sizeof(dim3)));
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

  hipLaunchAttribute attr{};
  attr.id = hipLaunchAttributeClusterDimension;
  attr.val.clusterDim.x = 2;
  attr.val.clusterDim.y = 1;
  attr.val.clusterDim.z = 1;
  hipLaunchConfig_t config{};
  config.gridDim = dim3(kBlocks);
  config.blockDim = dim3(kThreads);
  config.stream = stream;
  const bool with_attribute = GENERATE(true, false);
  INFO("cluster attribute " << (with_attribute ? "{2, 1, 1}" : "absent"));
  config.attrs = with_attribute ? &attr : nullptr;
  config.numAttrs = with_attribute ? 1 : 0;
  const dim3 expected = with_attribute ? dim3(2, 1, 1) : dim3(1, 1, 1);

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipLaunchKernelEx(&config, RecordClusterDims, d_out));
  HIP_CHECK(hipStreamEndCapture(stream, &graph));

  const dim3 node_dims = KernelNodeClusterDims(graph);
  REQUIRE(node_dims.x == expected.x);
  REQUIRE(node_dims.y == expected.y);
  REQUIRE(node_dims.z == expected.z);
  ReplayAndCheck(graph, stream, d_out, expected);

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(d_out));
}

/**
 * Test Description
 * ------------------------
 *  - A kernel declared with __cluster_dims__(2, 1, 1) is captured from launches that give no
 *    cluster dimensions (hipLaunchKernelGGL, and hipLaunchKernelEx without attributes). The
 *    captured kernel node must take the kernel's own cluster dimensions, and the replayed blocks
 *    must run in 2x1x1 clusters. A launch attribute that matches the kernel's dimensions must
 *    also be accepted.
 *
 * Test source
 * ------------------------
 *  - catch/unit/cluster/hipClusterStreamCapture.cc
 *
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */
HIP_TEST_CASE(Unit_hipClusterLaunch_StreamCapture_CompilerDirective) {
  if (!CheckTargetSupport()) {
    HIP_SKIP_TEST("Cluster launch is not supported on this device");
    return;
  }
  dim3* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_out, kBlocks * sizeof(dim3)));
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

  hipLaunchAttribute attr{};
  attr.id = hipLaunchAttributeClusterDimension;
  attr.val.clusterDim.x = 2;
  attr.val.clusterDim.y = 1;
  attr.val.clusterDim.z = 1;
  hipLaunchConfig_t config{};
  config.gridDim = dim3(kBlocks);
  config.blockDim = dim3(kThreads);
  config.stream = stream;

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  SECTION("hipLaunchKernelGGL") {
    hipLaunchKernelGGL(RecordClusterDimsCD, dim3(kBlocks), dim3(kThreads), 0, stream, d_out);
    HIP_CHECK(hipGetLastError());
  }
  SECTION("hipLaunchKernelEx without attributes") {
    HIP_CHECK(hipLaunchKernelEx(&config, RecordClusterDimsCD, d_out));
  }
  SECTION("hipLaunchKernelEx with a matching cluster attribute") {
    config.attrs = &attr;
    config.numAttrs = 1;
    HIP_CHECK(hipLaunchKernelEx(&config, RecordClusterDimsCD, d_out));
  }
  HIP_CHECK(hipStreamEndCapture(stream, &graph));

  const dim3 node_dims = KernelNodeClusterDims(graph);
  REQUIRE(node_dims.x == 2);
  REQUIRE(node_dims.y == 1);
  REQUIRE(node_dims.z == 1);
  ReplayAndCheck(graph, stream, d_out, dim3(2, 1, 1));

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(d_out));
}

/**
 * End doxygen group ClusterTest.
 * @}
 */
