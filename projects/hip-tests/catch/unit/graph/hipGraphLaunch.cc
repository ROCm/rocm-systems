/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <vector>

/**
 * @addtogroup hipGraphLaunch hipGraphLaunch
 * @{
 * @ingroup GraphTest
 * `hipGraphLaunch(hipGraphExec_t graphExec, hipStream_t stream)` -
 * Launches an executable graph in a stream
 */

// Simple index-writing kernel: out[i] = i, guarded by bounds check.
__global__ void WriteIndexKernel(int* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = idx;
  }
}

static constexpr int kWriteIndexN = 256;
static constexpr int kWriteIndexBlockDim = 64;

/* Create a deterministic executable graph containing a single kernel node that writes
   out[i] = i for i in [0, kWriteIndexN) to verify repeat runs have no change in output. */
static void CreateKernelNodeExecutableGraph(hipGraphExec_t* graph_exec, int** d_out) {
  HIP_CHECK(hipMalloc(d_out, kWriteIndexN * sizeof(int)));

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipKernelNodeParams node_params{};
  node_params.func = reinterpret_cast<void*>(WriteIndexKernel);
  node_params.gridDim = dim3(kWriteIndexN / kWriteIndexBlockDim, 1, 1);
  node_params.blockDim = dim3(kWriteIndexBlockDim, 1, 1);
  node_params.sharedMemBytes = 0;
  int n_val = kWriteIndexN;

  void* kernel_params[] = {d_out, &n_val};
  node_params.kernelParams = reinterpret_cast<void**>(kernel_params);

  hipGraphNode_t kernel_node;
  HIP_CHECK(hipGraphAddKernelNode(&kernel_node, graph, nullptr, 0, &node_params));

  hipGraphNode_t node_error;
  HIP_CHECK(hipGraphInstantiate(graph_exec, graph, &node_error, nullptr, 0));
  HIP_CHECK(hipGraphDestroy(graph));
}

static void HostFunctionSetToZero(void* arg) {
  int* test_number = (int*)arg;
  (*test_number) = 0;
}

static void HostFunctionAddOne(void* arg) {
  int* test_number = (int*)arg;
  (*test_number) += 1;
}

/* create an executable graph that will set an integer pointed to by 'number' to one*/
static void CreateTestExecutableGraph(hipGraphExec_t* graph_exec, int* number) {
  hipGraph_t graph;
  hipGraphNode_t node_error;

  hipGraphNode_t node_set_zero;
  hipHostNodeParams params_set_to_zero = {HostFunctionSetToZero, number};

  hipGraphNode_t node_add_one;
  hipHostNodeParams params_set_add_one = {HostFunctionAddOne, number};

  HIP_CHECK(hipGraphCreate(&graph, 0));

  HIP_CHECK(hipGraphAddHostNode(&node_set_zero, graph, nullptr, 0, &params_set_to_zero));
  HIP_CHECK(hipGraphAddHostNode(&node_add_one, graph, &node_set_zero, 1, &params_set_add_one));

  HIP_CHECK(hipGraphInstantiate(graph_exec, graph, &node_error, nullptr, 0));
  HIP_CHECK(hipGraphDestroy(graph));
}

static void HipGraphLaunch_Positive_Simple(hipStream_t stream) {
  int number = 5;

  hipGraphExec_t graph_exec;
  CreateTestExecutableGraph(&graph_exec, &number);

  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  REQUIRE(number == 1);

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
}


/**
 * Test Description
 * ------------------------
 *    - Basic positive test for hipGraphLaunch
 *        -# stream as a created stream
 *        -# with stream as hipStreamPerThread
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphLaunch.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipGraphLaunch_Positive) {
  SECTION("stream as a created stream") {
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    HipGraphLaunch_Positive_Simple(stream);
    HIP_CHECK(hipStreamDestroy(stream));
  }

  SECTION("with stream as hipStreamPerThread") {
    HipGraphLaunch_Positive_Simple(hipStreamPerThread);
  }
}

/**
 * Test Description
 * ------------------------
 *    - Negative parameter test for hipGraphLaunch
 *        -# graphExec is nullptr and stream is a created stream
 *        -# graphExec is nullptr and stream is hipStreamPerThread
 *        -# graphExec is an empty object
 *        -# graphExec is destroyed before calling hipGraphLaunch
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphLaunch.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipGraphLaunch_Negative_Parameters) {
  SECTION("graphExec is nullptr and stream is a created stream") {
    hipStream_t stream;
    hipError_t ret;
    HIP_CHECK(hipStreamCreate(&stream));
    ret = hipGraphLaunch(nullptr, stream);
    HIP_CHECK(hipStreamDestroy(stream));
    REQUIRE(ret == hipErrorInvalidValue);
  }

  SECTION("graphExec is nullptr and stream is hipStreamPerThread") {
    HIP_CHECK_ERROR(hipGraphLaunch(nullptr, hipStreamPerThread), hipErrorInvalidValue);
  }

  SECTION("graphExec is an empty object") {
    hipGraphExec_t graph_exec{};
    HIP_CHECK_ERROR(hipGraphLaunch(graph_exec, hipStreamPerThread), hipErrorInvalidValue);
  }

  SECTION("graphExec is destroyed") {
    int number = 5;
    hipGraphExec_t graph_exec;
    CreateTestExecutableGraph(&graph_exec, &number);
    HIP_CHECK(hipGraphLaunch(graph_exec, hipStreamPerThread));
    HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
    REQUIRE(number == 1);
    HIP_CHECK(hipGraphExecDestroy(graph_exec));
    HIP_CHECK_ERROR(hipGraphLaunch(graph_exec, hipStreamPerThread), hipErrorInvalidValue);
  }
}

/**
 * Test Description
 * ------------------------
 *    -   : Test repeat calls to hipGraphLaunch check invariance between runs
 *        - Instantiate a graph whose nodes reset state then compute a result.
 *        - Dirty the tracked variable before each of N launches so that each
 *          replay must re-execute the full reset+compute sequence.
 *        - Validate correct output after each launch
 *        -# stream as a created stream
 *        -# with stream as hipStreamPerThread
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphLaunch.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipGraphLaunch_Positive_RepeatedLaunchIdempotency) {
  constexpr int kNumLaunches = 5;

  auto RunReplayTest = [&](hipStream_t stream) {
    int* d_out = nullptr;
    hipGraphExec_t graph_exec;
    CreateKernelNodeExecutableGraph(&graph_exec, &d_out);

    std::vector<int> h_out(kWriteIndexN);

    for (int i = 0; i < kNumLaunches; ++i) {
      // Fill buffer with sentinal value
      HIP_CHECK(hipMemset(d_out, 0xFF, kWriteIndexN * sizeof(int)));

      HIP_CHECK(hipGraphLaunch(graph_exec, stream));
      HIP_CHECK(hipStreamSynchronize(stream));

      HIP_CHECK(hipMemcpy(h_out.data(), d_out, kWriteIndexN * sizeof(int), hipMemcpyDeviceToHost));

      for (int j = 0; j < kWriteIndexN; ++j) {
        REQUIRE(h_out[j] == j);
      }
    }

    HIP_CHECK(hipGraphExecDestroy(graph_exec));
    HIP_CHECK(hipFree(d_out));
  };

  SECTION("stream as a created stream") {
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    RunReplayTest(stream);
    HIP_CHECK(hipStreamDestroy(stream));
  }

  SECTION("with stream as hipStreamPerThread") { RunReplayTest(hipStreamPerThread); }
}

/**
 * End doxygen group GraphTest.
 * @}
 */
