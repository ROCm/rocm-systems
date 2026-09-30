/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * Tests for hipGraphLaunch when the launch stream's device differs from the
 * device on which the graph was instantiated (cross-device launch). This
 * exercises segmented replay on capture-device streams and the dependency
 * bridging required to preserve ordering with the foreign launch stream.
 */

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>
#include <utils.hh>

// ---------------------------------------------------------------------------
// Helper: set up a cross-device launch context.
// launch_stream - created on launch_dev (caller-owned)
// ---------------------------------------------------------------------------
static void setupCrossDeviceStream(int inst_dev, int launch_dev, hipStream_t& launch_stream) {
  HIP_CHECK(hipSetDevice(launch_dev));
  HIP_CHECK(hipStreamCreate(&launch_stream));
  HIP_CHECK(hipSetDevice(inst_dev));
}

struct HostCheckContext {
  int* value = nullptr;
  int expected = 0;
  int observed = 0;
};

static void recordHostValue(void* user_data) {
  auto* context = reinterpret_cast<HostCheckContext*>(user_data);
  context->observed = *context->value;
}

// ---------------------------------------------------------------------------

/**
 * Test Description
 * ------------------------
 *  - Cross-stream dependency ordering: a host node on the launch-device stream depends on work
 *    dispatched to the instantiate-device stream. The host node observes a D2H result that is only
 *    valid if the kernel and memcpy dependencies completed first.
 * ------------------------
 *  - catch/unit/graph/hipGraphCrossDeviceLaunch.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 *  - Multi-device
 */
HIP_TEST_CASE(Unit_hipGraphCrossDeviceLaunch_CrossStreamDependencyOrdering) {
  int nGpus = 0;
  HIP_CHECK(hipGetDeviceCount(&nGpus));
  if (nGpus < 2) HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);

  hipStream_t launch_stream;
  setupCrossDeviceStream(1, 0, launch_stream);

  HIP_CHECK(hipSetDevice(1));
  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  int* d_value = nullptr;
  HIP_CHECK(hipMalloc(&d_value, sizeof(int)));
  int h_value = 0;

  constexpr int expected_value = 1234;
  hipGraphNode_t set_node = nullptr;
  hipKernelNodeParams set_params{};
  int set_value = expected_value;
  size_t count = 1;
  void* set_args[] = {&d_value, &set_value, &count};
  set_params.func = reinterpret_cast<void*>(VectorSet<int>);
  set_params.gridDim = dim3(1);
  set_params.blockDim = dim3(1);
  set_params.kernelParams = reinterpret_cast<void**>(set_args);
  HIP_CHECK(hipGraphAddKernelNode(&set_node, graph, nullptr, 0, &set_params));

  hipGraphNode_t memcpy_node = nullptr;
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_node, graph, &set_node, 1, &h_value, d_value,
                                    sizeof(int), hipMemcpyDeviceToHost));

  HostCheckContext context;
  context.value = &h_value;
  context.expected = expected_value;
  hipGraphNode_t host_node = nullptr;
  hipHostNodeParams host_params = {0, 0};
  host_params.fn = recordHostValue;
  host_params.userData = &context;

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipGraphAddHostNode(&host_node, graph, &memcpy_node, 1, &host_params));

  HIP_CHECK(hipSetDevice(1));
  hipGraphExec_t graph_exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipGraphLaunch(graph_exec, launch_stream));
  HIP_CHECK(hipStreamSynchronize(launch_stream));

  REQUIRE(context.observed == context.expected);

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(d_value));
  HIP_CHECK(hipStreamDestroy(launch_stream));
}

/**
 * Test Description
 * ------------------------
 *  - Explicit-node graph (hipGraphAdd*): verifies cross-device launch works
 *    for graphs built via hipGraphAdd* rather than stream capture. These nodes
 *    have stream_id_==-1, which previously caused an assertion failure in
 *    RunNodes() when max_streams > 1; the cross-device branch must handle them.
 * ------------------------
 *  - catch/unit/graph/hipGraphCrossDeviceLaunch.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 *  - Multi-device
 */
HIP_TEST_CASE(Unit_hipGraphCrossDeviceLaunch_ExplicitNodes) {
  int nGpus = 0;
  HIP_CHECK(hipGetDeviceCount(&nGpus));
  if (nGpus < 2) HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);

  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;
  constexpr auto threadsPerBlock = 256;

  hipStream_t launch_stream;
  setupCrossDeviceStream(1, 0, launch_stream);

  HIP_CHECK(hipSetDevice(1));
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t memcpy_A, memcpy_B, kernel_node, memcpy_C;
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_A, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_B, graph, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));

  size_t NElem = N;
  void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
  hipKernelNodeParams kNodeParams{};
  kNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kNodeParams.gridDim = dim3(blocks);
  kNodeParams.blockDim = dim3(threadsPerBlock);
  kNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  HIP_CHECK(hipGraphAddKernelNode(&kernel_node, graph, nullptr, 0, &kNodeParams));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_C, graph, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));

  HIP_CHECK(hipGraphAddDependencies(graph, &memcpy_A, &kernel_node, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpy_B, &kernel_node, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &kernel_node, &memcpy_C, 1));

  HIP_CHECK(hipSetDevice(1));
  hipGraphExec_t graph_exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipGraphLaunch(graph_exec, launch_stream));
  HIP_CHECK(hipStreamSynchronize(launch_stream));

  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(launch_stream));
}

