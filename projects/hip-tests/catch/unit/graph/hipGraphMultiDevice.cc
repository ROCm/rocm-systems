/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <functional>
#include <vector>

#include <hip_test_helper.hh>

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

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
 *  - Verifies that a 1D memcpy graph node between host memory and device memory
 *    runs on the device that owns the device buffer, not on the device that was
 *    current when hipHostMalloc was called or when the node was added.
 *    Section 1: D2H copy from device 1 buffer into host memory allocated while
 *    device 0 was current. Section 2: H2D copy from host memory allocated while
 *    device 0 was current into a device 1 buffer; a kernel node on device 1
 *    overwrites the second half so both halves can be verified independently.
 * ------------------------
 *  - catch/unit/graph/hipGraphMultiDevice.cc
 * Test requirements
 * ------------------------
 *  - Multi-device (>= 2 GPUs)
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipGraphAddMemcpyNode1D_HostMemoryFromOtherDevice) {
  int n_gpus = 0;
  HIP_CHECK(hipGetDeviceCount(&n_gpus));
  if (n_gpus < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }

  constexpr int kN = 1024;
  constexpr int kFillValue = 42;
  constexpr int kValueA = 7;
  constexpr int kValueB = 99;
  constexpr int block_size = 256;

  SECTION("D2H into host memory allocated on another device") {
    // Allocate host buffer while device 0 is current.
    HIP_CHECK(hipSetDevice(0));
    int* host_buf = nullptr;
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&host_buf), kN * sizeof(int), 0));
    memset(host_buf, 0, kN * sizeof(int));

    // Set up device 1 buffer and stream.
    HIP_CHECK(hipSetDevice(1));
    int* buf_d1 = nullptr;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&buf_d1), kN * sizeof(int)));

    hipStream_t stream1 = nullptr;
    HIP_CHECK(hipStreamCreate(&stream1));

    // Build the graph: kernel fills buf_d1, then D2H memcpy to host_buf.
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipGraphCreate(&graph, 0));

    // Kernel node: fill buf_d1 with kFillValue on device 1.
    hipGraphNode_t kernel_node = nullptr;
    {
      unsigned grid_size = (kN + block_size - 1) / block_size;
      int64_t n_elems = kN;
      int fill_val = kFillValue;
      void* kernel_args[] = {reinterpret_cast<void*>(&buf_d1),
                             reinterpret_cast<void*>(&fill_val),
                             reinterpret_cast<void*>(&n_elems)};
      hipKernelNodeParams kp{};
      kp.func = reinterpret_cast<void*>(HipTest::memsetReverse<int>);
      kp.gridDim = dim3(grid_size, 1, 1);
      kp.blockDim = dim3(block_size, 1, 1);
      kp.sharedMemBytes = 0;
      kp.kernelParams = kernel_args;
      kp.extra = nullptr;
      HIP_CHECK(hipGraphAddKernelNode(&kernel_node, graph, nullptr, 0, &kp));
    }

    // Switch to device 0 to add the memcpy node; it must still run on device 1
    // (the device that owns buf_d1).
    HIP_CHECK(hipSetDevice(0));
    hipGraphNode_t memcpy_node = nullptr;
    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_node, graph, &kernel_node, 1, host_buf, buf_d1,
                                      kN * sizeof(int), hipMemcpyDeviceToHost));

    // Instantiate and run on device 1's stream.
    hipGraphExec_t graph_exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graph_exec, stream1));
    HIP_CHECK(hipStreamSynchronize(stream1));

    for (int i = 0; i < kN; ++i) {
      REQUIRE(host_buf[i] == kFillValue);
    }

    HIP_CHECK(hipGraphExecDestroy(graph_exec));
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipStreamDestroy(stream1));
    HIP_CHECK(hipFree(buf_d1));
    HIP_CHECK(hipHostFree(host_buf));
  }

  SECTION("H2D from host memory allocated on another device") {
    // Allocate and fill host buffer while device 0 is current.
    HIP_CHECK(hipSetDevice(0));
    int* host_buf = nullptr;
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&host_buf), kN * sizeof(int), 0));
    for (int i = 0; i < kN; ++i) {
      host_buf[i] = kValueA;
    }

    // Set up device 1 resources; device 1 stays current for the rest of setup.
    HIP_CHECK(hipSetDevice(1));
    int* buf_d1 = nullptr;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&buf_d1), kN * sizeof(int)));

    hipStream_t stream1 = nullptr;
    HIP_CHECK(hipStreamCreate(&stream1));

    hipGraph_t graph = nullptr;
    HIP_CHECK(hipGraphCreate(&graph, 0));

    // Memcpy node: H2D host_buf -> buf_d1[0..kN-1].
    hipGraphNode_t memcpy_node = nullptr;
    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_node, graph, nullptr, 0, buf_d1, host_buf,
                                      kN * sizeof(int), hipMemcpyHostToDevice));

    // Kernel node (depends on memcpy): overwrite the second half of buf_d1 with kValueB.
    hipGraphNode_t kernel_node = nullptr;
    {
      int half = kN / 2;
      unsigned grid_size = (half + block_size - 1) / block_size;
      int* buf_d1_second_half = buf_d1 + half;
      int fill_val = kValueB;
      int64_t n_elems = half;
      void* kernel_args[] = {reinterpret_cast<void*>(&buf_d1_second_half),
                             reinterpret_cast<void*>(&fill_val),
                             reinterpret_cast<void*>(&n_elems)};
      hipKernelNodeParams kp{};
      kp.func = reinterpret_cast<void*>(HipTest::memsetReverse<int>);
      kp.gridDim = dim3(grid_size, 1, 1);
      kp.blockDim = dim3(block_size, 1, 1);
      kp.sharedMemBytes = 0;
      kp.kernelParams = kernel_args;
      kp.extra = nullptr;
      HIP_CHECK(hipGraphAddKernelNode(&kernel_node, graph, &memcpy_node, 1, &kp));
    }

    hipGraphExec_t graph_exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graph_exec, stream1));
    HIP_CHECK(hipStreamSynchronize(stream1));

    // Copy result back and verify: first half kValueA, second half kValueB.
    std::vector<int> result(kN, 0);
    HIP_CHECK(hipMemcpy(result.data(), buf_d1, kN * sizeof(int), hipMemcpyDeviceToHost));
    for (int i = 0; i < kN / 2; ++i) {
      REQUIRE(result[i] == kValueA);
    }
    for (int i = kN / 2; i < kN; ++i) {
      REQUIRE(result[i] == kValueB);
    }

    HIP_CHECK(hipGraphExecDestroy(graph_exec));
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipStreamDestroy(stream1));
    HIP_CHECK(hipFree(buf_d1));
    HIP_CHECK(hipHostFree(host_buf));
  }
}

/**
 * End doxygen group GraphMultiDevice.
 * @}
 */
