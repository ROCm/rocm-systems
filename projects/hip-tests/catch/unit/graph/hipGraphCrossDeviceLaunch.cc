/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * Tests for hipGraphLaunch when the launch stream's device differs from the
 * device on which the graph was instantiated (cross-device launch). This
 * exercises the Path B branch in GraphExec::Run, including dependency waits
 * between work submitted to instantiate-device streams and launch-device streams.
 */

#include <cstdint>

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>
#include <utils.hh>

// ---------------------------------------------------------------------------
// Helper: set up a cross-device launch context.
// cap_stream  - created on inst_dev, used for capture (caller-owned)
// launch_stream - created on launch_dev (caller-owned)
// ---------------------------------------------------------------------------
static void setupCrossDeviceStreams(int inst_dev, int launch_dev, hipStream_t& cap_stream,
                                    hipStream_t& launch_stream) {
  HIP_CHECK(hipSetDevice(launch_dev));
  HIP_CHECK(hipStreamCreate(&launch_stream));
  HIP_CHECK(hipSetDevice(inst_dev));
  HIP_CHECK(hipStreamCreate(&cap_stream));
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

static void setHostValue(void* user_data) {
  auto* context = reinterpret_cast<HostCheckContext*>(user_data);
  *context->value = context->expected;
}

// ---------------------------------------------------------------------------

/**
 * Test Description
 * ------------------------
 *  - Cross-stream dependency ordering: a host node on the launch-device stream depends on work
 *    dispatched to the instantiate-device stream. The host node observes a D2H result that is only
 *    valid if the delay, kernel, and memcpy dependencies completed first.
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

  hipStream_t cap_stream, launch_stream;
  setupCrossDeviceStreams(1, 0, cap_stream, launch_stream);

  int clockrate = 0;
  HIP_CHECK(hipDeviceGetAttribute(&clockrate, hipDeviceAttributeMemoryClockRate, 1));
  constexpr uint32_t delay_ms = 200;
  uint32_t delay_arg = delay_ms;
  uint32_t ticks_per_ms = static_cast<uint32_t>(clockrate);

  HIP_CHECK(hipSetDevice(1));
  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  int* d_value = nullptr;
  HIP_CHECK(hipMalloc(&d_value, sizeof(int)));
  int h_value = 0;

  hipGraphNode_t delay_node = nullptr;
  hipKernelNodeParams kernel_params{};
  void* kernel_args[] = {&delay_arg, &ticks_per_ms};
  kernel_params.func = reinterpret_cast<void*>(Delay);
  kernel_params.gridDim = dim3(1);
  kernel_params.blockDim = dim3(1);
  kernel_params.kernelParams = reinterpret_cast<void**>(kernel_args);
  HIP_CHECK(hipGraphAddKernelNode(&delay_node, graph, nullptr, 0, &kernel_params));

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
  HIP_CHECK(hipGraphAddKernelNode(&set_node, graph, &delay_node, 1, &set_params));

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
  HIP_CHECK(hipStreamDestroy(cap_stream));
  HIP_CHECK(hipStreamDestroy(launch_stream));
}

/**
 * Test Description
 * ------------------------
 *  - Launch-stream start ordering: work queued on the launch stream before hipGraphLaunch must
 *    complete before root nodes on the instantiate device begin. A delay kernel and host callback
 *    on the launch stream update a host value; the graph root H2D must observe the updated value.
 * ------------------------
 *  - catch/unit/graph/hipGraphCrossDeviceLaunch.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 *  - Multi-device
 */
HIP_TEST_CASE(Unit_hipGraphCrossDeviceLaunch_LaunchStreamStartOrdering) {
  int nGpus = 0;
  HIP_CHECK(hipGetDeviceCount(&nGpus));
  if (nGpus < 2) HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);

  hipStream_t cap_stream, launch_stream;
  setupCrossDeviceStreams(1, 0, cap_stream, launch_stream);

  int clockrate = 0;
  HIP_CHECK(hipDeviceGetAttribute(&clockrate, hipDeviceAttributeMemoryClockRate, 0));
  constexpr uint32_t delay_ms = 200;
  uint32_t delay_arg = delay_ms;
  uint32_t ticks_per_ms = static_cast<uint32_t>(clockrate);

  constexpr int expected_value = 42;
  int* h_value = nullptr;
  int* h_check = nullptr;
  HIP_CHECK(hipHostMalloc(&h_value, sizeof(int)));
  HIP_CHECK(hipHostMalloc(&h_check, sizeof(int)));
  *h_value = 0;
  *h_check = 0;

  HostCheckContext set_context;
  set_context.value = h_value;
  set_context.expected = expected_value;

  HostCheckContext check_context;
  check_context.value = h_check;
  check_context.expected = expected_value;

  void* delay_args[] = {&delay_arg, &ticks_per_ms};

  HIP_CHECK(hipSetDevice(1));
  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  int* d_value = nullptr;
  HIP_CHECK(hipMalloc(&d_value, sizeof(int)));

  hipGraphNode_t h2d_node = nullptr;
  HIP_CHECK(hipGraphAddMemcpyNode1D(&h2d_node, graph, nullptr, 0, d_value, h_value, sizeof(int),
                                    hipMemcpyHostToDevice));

  hipGraphNode_t d2h_node = nullptr;
  HIP_CHECK(hipGraphAddMemcpyNode1D(&d2h_node, graph, &h2d_node, 1, h_check, d_value, sizeof(int),
                                    hipMemcpyDeviceToHost));

  hipGraphNode_t host_node = nullptr;
  hipHostNodeParams host_params = {0, 0};
  host_params.fn = recordHostValue;
  host_params.userData = &check_context;
  HIP_CHECK(hipGraphAddHostNode(&host_node, graph, &d2h_node, 1, &host_params));

  hipGraphExec_t graph_exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

  HIP_CHECK(hipSetDevice(0));
  *h_value = 0;
  *h_check = 0;
  HIP_CHECK(hipLaunchKernel(reinterpret_cast<void*>(Delay), dim3(1), dim3(1), delay_args, 0,
                            launch_stream));
  HIP_CHECK(hipLaunchHostFunc(launch_stream, setHostValue, &set_context));
  HIP_CHECK(hipGraphLaunch(graph_exec, launch_stream));
  HIP_CHECK(hipStreamSynchronize(launch_stream));

  REQUIRE(check_context.observed == check_context.expected);

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(d_value));
  HIP_CHECK(hipHostFree(h_value));
  HIP_CHECK(hipHostFree(h_check));
  HIP_CHECK(hipStreamDestroy(cap_stream));
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

  hipStream_t cap_stream, launch_stream;
  setupCrossDeviceStreams(1, 0, cap_stream, launch_stream);

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
  HIP_CHECK(hipStreamDestroy(cap_stream));
  HIP_CHECK(hipStreamDestroy(launch_stream));
}

