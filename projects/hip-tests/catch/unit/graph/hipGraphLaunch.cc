/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>

/**
 * @addtogroup hipGraphLaunch hipGraphLaunch
 * @{
 * @ingroup GraphTest
 * `hipGraphLaunch(hipGraphExec_t graphExec, hipStream_t stream)` -
 * Launches an executable graph in a stream
 */

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
TEST_CASE("Unit_hipGraphLaunch_Positive") {
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
TEST_CASE("Unit_hipGraphLaunch_Negative_Parameters") {
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

// Each independent root writes its own location. Atomic accesses make an early
// consumer observable without introducing a concurrent non-atomic data race.
static __global__ void GraphCompletionWrite(int* values, int slot, int token) {
  volatile unsigned int delay = static_cast<unsigned int>(slot + 1);
  // Slot 11 is a tail omitted by the old same-level stream reduction. Keep it
  // eight times longer than all other roots so unrelated waited tails cannot
  // hide a missing final join by finishing at roughly the same time.
  const int delay_iterations = slot == 11 ? 262144 : 32768;
  for (int i = 0; i < delay_iterations; ++i) {
    delay = delay * 1664525u + 1013904223u;
  }
  atomicExch(values + slot, token);
}

static __global__ void GraphCompletionSnapshot(int* values, int* snapshot, int count) {
  const int slot = static_cast<int>(threadIdx.x);
  if (slot < count) snapshot[slot] = atomicAdd(values + slot, 0);
}

/**
 * Test Description
 * ------------------------
 *    - Work submitted after hipGraphLaunch on the launch stream sees every root.
 *    - Independent roots exercise final completion without a graph sink node.
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphLaunch.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphLaunch_IndependentRootsCompletion") {
  // One root is the single-stream control. On release/rocm-rel-7.2 defaults,
  // 14/15 one-node segments use segmented scheduling; 16 uses classic fallback.
  const int roots = GENERATE(1, 14, 15, 16);
  int* values = nullptr;
  int* snapshot = nullptr;
  HIP_CHECK(hipMalloc(&values, roots * sizeof(int)));
  HIP_CHECK(hipMalloc(&snapshot, roots * sizeof(int)));
  hipStream_t stream;
  HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  hipGraph_t graph;
  hipGraphExec_t executable;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  const int token = 123;
  for (int slot = 0; slot < roots; ++slot) {
    int argument_slot = slot;
    int argument_token = token;
    void* args[] = {&values, &argument_slot, &argument_token};
    hipKernelNodeParams params{};
    params.func = reinterpret_cast<void*>(GraphCompletionWrite);
    params.gridDim = dim3(1);
    params.blockDim = dim3(1);
    params.kernelParams = args;
    hipGraphNode_t node;
    HIP_CHECK(hipGraphAddKernelNode(&node, graph, nullptr, 0, &params));
  }
  HIP_CHECK(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  bool complete = true;
  for (int iteration = 0; iteration < 8; ++iteration) {
    HIP_CHECK(hipMemsetAsync(values, 0, roots * sizeof(int), stream));
    HIP_CHECK(hipMemsetAsync(snapshot, 0, roots * sizeof(int), stream));
    // Finish initialization before launch: this test does not rely on the
    // separate initial-fork ordering contract of internally created streams.
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipGraphLaunch(executable, stream));
    // No device sync, graph sink, or cross-stream event before this consumer.
    hipLaunchKernelGGL(GraphCompletionSnapshot, dim3(1), dim3(32), 0, stream,
                       values, snapshot, roots);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(stream));
    int observed[16] = {};
    HIP_CHECK(hipMemcpy(observed, snapshot, roots * sizeof(int), hipMemcpyDeviceToHost));
    // Retire even work missed by a broken join before reinitialization/free.
    // This cannot repair the snapshot already stored by the consumer.
    HIP_CHECK(hipDeviceSynchronize());
    for (int slot = 0; slot < roots; ++slot) complete &= observed[slot] == token;
  }
  HIP_CHECK(hipGraphExecDestroy(executable));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(snapshot));
  HIP_CHECK(hipFree(values));
  INFO("independent roots: " << roots);
  REQUIRE(complete);
}

/**
 * End doxygen group GraphTest.
 * @}
 */
