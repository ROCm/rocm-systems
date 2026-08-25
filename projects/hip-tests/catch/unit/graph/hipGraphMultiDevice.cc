/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <functional>

#include <hip_test_helper.hh>

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

#include <algorithm>
#include <vector>

#include "graph_memset_node_test_common.hh"
#include "graph_tests_common.hh"

/**
 * @addtogroup hipGraphLaunch
 * @{
 * @ingroup GraphTest
 * `hipGraphLaunch(hipGraphExec_t graphExec, hipStream_t stream)` -
 * Launches an executable graph on the multi device
 */

/**
 * Test Description
 * ------------------------
 *  - Launches the single branch graph on multi device and verify the result
 * ------------------------
 *  - catch/unit/graph//hipGraphMultiDevice.cc
 * Test requirements
 * ------------------------
 *  - Multi-device
 *  - HIP_VERSION >= 7.2
 */
static void check_output(int* inp, int* out, size_t size) {
  for (size_t i = 0; i < size; i++) {
    REQUIRE(out[i] == ((inp[i] * inp[i]) * (inp[i] * inp[i])));
  }
}

static void init_input(int* a, size_t size) {
  unsigned int seed = time(nullptr);
  for (size_t i = 0; i < size; i++) {
    a[i] = (HipTest::RAND_R(&seed) & 0xFF);
  }
}

namespace {

constexpr size_t kN = 4096;
constexpr size_t kNbytes = kN * sizeof(int);
constexpr auto kThreadsPerBlock = 256;
constexpr unsigned kBlocks =
    static_cast<unsigned>((kN + kThreadsPerBlock - 1) / kThreadsPerBlock);

struct ParallelGraph {
  hipGraph_t graph = nullptr;
  int* d0 = nullptr;
  int* d1 = nullptr;
  std::vector<int> input0 = std::vector<int>(kN);
  std::vector<int> input1 = std::vector<int>(kN);
  std::vector<int> output0 = std::vector<int>(kN, -1);
  std::vector<int> output1 = std::vector<int>(kN, -1);
};

void init_input(int* data, size_t n, int salt) {
  for (size_t i = 0; i < n; ++i) {
    data[i] = static_cast<int>((i + static_cast<size_t>(salt)) & 0xFF);
  }
}

void check_squared(const int* input, const int* output, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    const int expected = input[i] * input[i];
    REQUIRE(output[i] == expected);
  }
}

__global__ void writePeerPatternKernel(int* peer, int base, size_t count) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    peer[index] = base + static_cast<int>(index);
  }
}

void build_parallel_graph(ParallelGraph& graph) {
  size_t n_elem = kN;

  init_input(graph.input0.data(), kN, 0);
  init_input(graph.input1.data(), kN, 3);
  std::fill(graph.output0.begin(), graph.output0.end(), -1);
  std::fill(graph.output1.begin(), graph.output1.end(), -1);

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipGraphCreate(&graph.graph, 0));
  HIP_CHECK(hipMalloc(&graph.d0, kNbytes));
  hipGraphNode_t h2d0 = nullptr;
  HIP_CHECK(hipGraphAddMemcpyNode1D(&h2d0, graph.graph, nullptr, 0, graph.d0,
                                    graph.input0.data(), kNbytes, hipMemcpyHostToDevice));

  hipGraphNode_t kernel0 = nullptr;
  hipKernelNodeParams kernel0_params{};
  void* kernel0_args[] = {&graph.d0, &graph.d0, &n_elem};
  kernel0_params.func = reinterpret_cast<void*>(HipTest::vector_square<int>);
  kernel0_params.gridDim = dim3(kBlocks);
  kernel0_params.blockDim = dim3(kThreadsPerBlock);
  kernel0_params.kernelParams = reinterpret_cast<void**>(kernel0_args);
  HIP_CHECK(hipGraphAddKernelNode(&kernel0, graph.graph, &h2d0, 1, &kernel0_params));

  hipGraphNode_t d2h0 = nullptr;
  HIP_CHECK(hipGraphAddMemcpyNode1D(&d2h0, graph.graph, &kernel0, 1, graph.output0.data(),
                                    graph.d0, kNbytes, hipMemcpyDeviceToHost));

  HIP_CHECK(hipSetDevice(1));
  HIP_CHECK(hipMalloc(&graph.d1, kNbytes));
  hipGraphNode_t h2d1 = nullptr;
  HIP_CHECK(hipGraphAddMemcpyNode1D(&h2d1, graph.graph, nullptr, 0, graph.d1,
                                    graph.input1.data(), kNbytes, hipMemcpyHostToDevice));

  hipGraphNode_t kernel1 = nullptr;
  hipKernelNodeParams kernel1_params{};
  void* kernel1_args[] = {&graph.d1, &graph.d1, &n_elem};
  kernel1_params.func = reinterpret_cast<void*>(HipTest::vector_square<int>);
  kernel1_params.gridDim = dim3(kBlocks);
  kernel1_params.blockDim = dim3(kThreadsPerBlock);
  kernel1_params.kernelParams = reinterpret_cast<void**>(kernel1_args);
  HIP_CHECK(hipGraphAddKernelNode(&kernel1, graph.graph, &h2d1, 1, &kernel1_params));

  hipGraphNode_t d2h1 = nullptr;
  HIP_CHECK(hipGraphAddMemcpyNode1D(&d2h1, graph.graph, &kernel1, 1, graph.output1.data(),
                                    graph.d1, kNbytes, hipMemcpyDeviceToHost));
}

void destroy_parallel_graph(ParallelGraph& graph) {
  if (graph.graph != nullptr) {
    HIP_CHECK(hipGraphDestroy(graph.graph));
  }
  if (graph.d0 != nullptr) {
    HIP_CHECK(hipFree(graph.d0));
  }
  if (graph.d1 != nullptr) {
    HIP_CHECK(hipFree(graph.d1));
  }
}

void launch_graph(hipGraphExec_t graph_exec, int launch_device = 0) {
  hipStream_t launch_stream = nullptr;
  HIP_CHECK(hipSetDevice(launch_device));
  HIP_CHECK(hipStreamCreate(&launch_stream));
  HIP_CHECK(hipGraphLaunch(graph_exec, launch_stream));
  HIP_CHECK(hipStreamSynchronize(launch_stream));
  HIP_CHECK(hipStreamDestroy(launch_stream));
}

}  // namespace


HIP_TEST_CASE(Unit_hipGraphMultiDevice) {
  int nGpus = 0;
  HIP_CHECK(hipGetDeviceCount(&nGpus));
  if (nGpus < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }
  int can_access_peer = 0;
  HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, 1, 0));
  if (!can_access_peer) {
    HIP_SKIP_TEST(HipTest::SkipReason::kPeerAccessUnavailable);
  }
  HIP_CHECK(hipSetDevice(1));
  HIP_CHECK(hipDeviceEnablePeerAccess(0, 0));

  hipStream_t streamdev1, streamdev2;
  hipEvent_t eventdev1, eventdev2;
  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;

  constexpr size_t buffer_size = (1024 * 1024);
  constexpr auto blocksPerCU = 6;
  constexpr int block_size = 512;

  int *ibuf_h, *buf_d1, *buf_d2, *outbuf_h;
  ibuf_h = new int[buffer_size];
  outbuf_h = new int[buffer_size];
  REQUIRE(ibuf_h != nullptr);

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipStreamCreate(&streamdev1));
  HIP_CHECK(hipMalloc(&buf_d1, buffer_size * sizeof(int)));
  HIP_CHECK(hipEventCreate(&eventdev1));

  HIP_CHECK(hipSetDevice(1));
  HIP_CHECK(hipStreamCreate(&streamdev2));
  HIP_CHECK(hipMalloc(&buf_d2, buffer_size * sizeof(int)));
  HIP_CHECK(hipEventCreate(&eventdev2));

  HIP_CHECK(hipSetDevice(0));
  init_input(ibuf_h, buffer_size);
  unsigned grid_size = HipTest::setNumBlocks(blocksPerCU, block_size, buffer_size);

  HIP_CHECK(hipStreamBeginCapture(streamdev1, hipStreamCaptureModeGlobal));

  HIP_CHECK(
      hipMemcpyAsync(buf_d1, ibuf_h, sizeof(int) * buffer_size, hipMemcpyHostToDevice, streamdev1));
  HipTest::vector_square<int>
      <<<grid_size, block_size, 0, streamdev1>>>(buf_d1, buf_d1, buffer_size);
  HIP_CHECK(hipEventRecord(eventdev1, streamdev1));
  HIP_CHECK(hipStreamWaitEvent(streamdev2, eventdev1, 0));

  HIP_CHECK(hipSetDevice(1));
  HIP_CHECK(
      hipMemcpyAsync(buf_d2, buf_d1, sizeof(int) * buffer_size, hipMemcpyDeviceToDevice, streamdev2));
  HipTest::vector_square<int>
      <<<grid_size, block_size, 0, streamdev2>>>(buf_d2, buf_d2, buffer_size);
  HIP_CHECK(hipEventRecord(eventdev2, streamdev2));
  HIP_CHECK(hipStreamWaitEvent(streamdev1, eventdev2, 0));

  HIP_CHECK(hipStreamEndCapture(streamdev1, &graph));

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graph_exec, streamdev1));
  HIP_CHECK(hipStreamSynchronize(streamdev1));

  HIP_CHECK(hipSetDevice(1));
  HIP_CHECK(hipMemcpy(outbuf_h, buf_d2, sizeof(int) * buffer_size, hipMemcpyDeviceToHost));
  check_output(ibuf_h, outbuf_h, buffer_size);

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));

  delete[] ibuf_h;
  delete[] outbuf_h;
  HIP_CHECK(hipFree(buf_d1));
  HIP_CHECK(hipFree(buf_d2));
  HIP_CHECK(hipStreamDestroy(streamdev1));
  HIP_CHECK(hipStreamDestroy(streamdev2));
  HIP_CHECK(hipEventDestroy(eventdev1));
  HIP_CHECK(hipEventDestroy(eventdev2));
}

/**
 * Test Description
 * ------------------------
 *  - Instantiates a true multi-device GraphExec once, then launches sequentially from GPU0,
 *    GPU1, and GPU2 when available. Host inputs change between launches so stale results or
 *    incomplete launch-stream bridging cannot pass.
 * ------------------------
 *  - catch/unit/graph/hipGraphMultiDevice.cc
 * Test requirements
 * ------------------------
 *  - Multi-device (>= 2 GPUs)
 */
HIP_TEST_CASE(Unit_hipGraphRealMultiDevice_RepeatLaunch) {
  CHECK_MULTIGPU

  ParallelGraph graph{};
  build_parallel_graph(graph);

  hipGraphExec_t graph_exec = nullptr;
  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph.graph, nullptr, nullptr, 0));

  int n_gpus = 0;
  HIP_CHECK(hipGetDeviceCount(&n_gpus));
  std::vector<int> launch_devices = {0, 1, 0};
  if (n_gpus >= 3) {
    launch_devices.push_back(2);
  }

  for (size_t launch = 0; launch < launch_devices.size(); ++launch) {
    init_input(graph.input0.data(), kN, 2 + static_cast<int>(launch) * 7);
    init_input(graph.input1.data(), kN, 5 + static_cast<int>(launch) * 8);
    std::fill(graph.output0.begin(), graph.output0.end(), -1);
    std::fill(graph.output1.begin(), graph.output1.end(), -1);

    launch_graph(graph_exec, launch_devices[launch]);
    check_squared(graph.input0.data(), graph.output0.data(), kN);
    check_squared(graph.input1.data(), graph.output1.data(), kN);
  }

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  destroy_parallel_graph(graph);
}

/**
 * Test Description
 * ------------------------
 *  - Builds a peer-enabled graph where a GPU0 kernel writes peer-visible GPU1 memory and a
 *    dependent GPU1 kernel reads it directly without an intervening memcpy node. Verifies both
 *    execution ordering and memory visibility.
 * ------------------------
 *  - catch/unit/graph/hipGraphMultiDevice.cc
 * Test requirements
 * ------------------------
 *  - Multi-device (>= 2 GPUs)
 *  - Peer access from GPU0 to GPU1
 */
HIP_TEST_CASE(Unit_hipGraphRealMultiDevice_DirectPeerMemoryDependency) {
  CHECK_MULTIGPU
  CHECK_P2P_SUPPORT

  HIP_CHECK(hipSetDevice(0));
  const hipError_t peer_status = hipDeviceEnablePeerAccess(1, 0);
  REQUIRE((peer_status == hipSuccess || peer_status == hipErrorPeerAccessAlreadyEnabled));

  int peer_base = 17;
  size_t n_elem = kN;
  std::vector<int> output(kN, -1);

  int* d_peer = nullptr;
  int* d_output = nullptr;
  HIP_CHECK(hipSetDevice(1));
  HIP_CHECK(hipMalloc(&d_peer, kNbytes));
  HIP_CHECK(hipMalloc(&d_output, kNbytes));

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t producer = nullptr;
  hipKernelNodeParams producer_params{};
  void* producer_args[] = {&d_peer, &peer_base, &n_elem};
  producer_params.func = reinterpret_cast<void*>(writePeerPatternKernel);
  producer_params.gridDim = dim3(kBlocks);
  producer_params.blockDim = dim3(kThreadsPerBlock);
  producer_params.kernelParams = reinterpret_cast<void**>(producer_args);
  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipGraphAddKernelNode(&producer, graph, nullptr, 0, &producer_params));

  hipGraphNode_t consumer = nullptr;
  hipKernelNodeParams consumer_params{};
  void* consumer_args[] = {&d_peer, &d_output, &n_elem};
  consumer_params.func = reinterpret_cast<void*>(HipTest::vector_square<int>);
  consumer_params.gridDim = dim3(kBlocks);
  consumer_params.blockDim = dim3(kThreadsPerBlock);
  consumer_params.kernelParams = reinterpret_cast<void**>(consumer_args);
  HIP_CHECK(hipSetDevice(1));
  HIP_CHECK(hipGraphAddKernelNode(&consumer, graph, &producer, 1, &consumer_params));

  hipGraphNode_t d2h = nullptr;
  HIP_CHECK(hipGraphAddMemcpyNode1D(&d2h, graph, &consumer, 1, output.data(), d_output,
                                    kNbytes, hipMemcpyDeviceToHost));

  hipGraphExec_t graph_exec = nullptr;
  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  launch_graph(graph_exec);

  for (size_t i = 0; i < kN; ++i) {
    const int expected_input = peer_base + static_cast<int>(i);
    REQUIRE(output[i] == expected_input * expected_input);
  }

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(d_peer));
  HIP_CHECK(hipFree(d_output));
  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipDeviceDisablePeerAccess(1));
}

/**
 * End doxygen group GraphMultiDevice.
 * @}
 */
