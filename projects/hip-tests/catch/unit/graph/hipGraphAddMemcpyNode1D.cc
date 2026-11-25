/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <functional>

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <memcpy1d_tests_common.hh>

#include "graph_tests_common.hh"

/**
 * @addtogroup hipGraphAddMemcpyNode1D hipGraphAddMemcpyNode1D
 * @{
 * @ingroup GraphTest
 * `hipGraphAddMemcpyNode1D(hipGraphNode_t *pGraphNode, hipGraph_t graph, const hipGraphNode_t
 * *pDependencies, size_t numDependencies, void *dst, const void *src, size_t count, hipMemcpyKind
 * kind)` - Creates a 1D memcpy node and adds it to a graph
 */

/**
 * Test Description
 * ------------------------
 *    - Verify basic API behavior. A Memcpy1D node is created with parameters set according to the
 * test run, after which the graph is run and the memcpy results are verified.
 * The test is run for all possible memcpy directions, with both the corresponding memcpy
 * kind and hipMemcpyDefault, as well as half page and full page allocation sizes.
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphAddMemcpyNode1D.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipGraphAddMemcpyNode1D_Positive_Basic) {
  constexpr auto f = [](void* dst, void* src, size_t count, hipMemcpyKind direction) {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipGraphCreate(&graph, 0));
    hipGraphNode_t node = nullptr;
    HIP_CHECK(hipGraphAddMemcpyNode1D(&node, graph, nullptr, 0, dst, src, count, direction));
    hipGraphExec_t graph_exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graph_exec, hipStreamPerThread));
    HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));

    HIP_CHECK(hipGraphExecDestroy(graph_exec));
    HIP_CHECK(hipGraphDestroy(graph));

    return hipSuccess;
  };

#if HT_NVIDIA
  MemcpyWithDirectionCommonTests<false>(f);
#else
  using namespace std::placeholders;

  SECTION("Device to host") {
    MemcpyDeviceToHostShell<false>(std::bind(f, _1, _2, _3, hipMemcpyDeviceToHost));
  }

  SECTION("Device to host with default kind") {
    MemcpyDeviceToHostShell<false>(std::bind(f, _1, _2, _3, hipMemcpyDefault));
  }

  SECTION("Host to device") {
    MemcpyHostToDeviceShell<false>(std::bind(f, _1, _2, _3, hipMemcpyHostToDevice));
  }

  SECTION("Host to device with default kind") {
    MemcpyHostToDeviceShell<false>(std::bind(f, _1, _2, _3, hipMemcpyDefault));
  }

// Disabled on AMD due to defect - EXSWHTEC-209
#if 0
  SECTION("Host to host") {
    MemcpyHostToHostShell<false>(std::bind(f, _1, _2, _3, hipMemcpyHostToHost));
  }

  SECTION("Host to host with default kind") {
    MemcpyHostToHostShell<false>(std::bind(f, _1, _2, _3, hipMemcpyDefault));
  }
#endif

  SECTION("Device to device") {
    SECTION("Peer access enabled") {
      MemcpyDeviceToDeviceShell<false, true>(std::bind(f, _1, _2, _3, hipMemcpyDeviceToDevice));
    }
    SECTION("Peer access disabled") {
      MemcpyDeviceToDeviceShell<false, false>(std::bind(f, _1, _2, _3, hipMemcpyDeviceToDevice));
    }
  }

  SECTION("Device to device with default kind") {
    SECTION("Peer access enabled") {
      MemcpyDeviceToDeviceShell<false, true>(std::bind(f, _1, _2, _3, hipMemcpyDefault));
    }
    SECTION("Peer access disabled") {
      MemcpyDeviceToDeviceShell<false, false>(std::bind(f, _1, _2, _3, hipMemcpyDefault));
    }
  }
#endif
}

/**
 * Test Description
 * ------------------------
 *    - Verify API behaviour with invalid arguments:
 *        -# node is nullptr
 *        -# graph is nullptr
 *        -# pDependencies is nullptr when numDependencies is not zero
 *        -# A node in pDependencies originates from a different graph
 *        -# numDependencies is invalid
 *        -# A node is duplicated in pDependencies
 *        -# dst is nullptr
 *        -# src is nullptr
 *        -# kind is an invalid enum value
 *        -# count is zero
 *        -# count is larger than dst allocation size
 *        -# count is larger than src allocation size
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphAddMemcpyNode1D.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipGraphAddMemcpyNode1D_Negative_Parameters) {
  using namespace std::placeholders;
  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t node = nullptr;
  int src[2] = {}, dst[2] = {};

  GraphAddNodeCommonNegativeTests(
      std::bind(hipGraphAddMemcpyNode1D, _1, _2, _3, _4, dst, src, sizeof(dst), hipMemcpyDefault),
      graph);

  MemcpyWithDirectionCommonNegativeTests(
      std::bind(hipGraphAddMemcpyNode1D, &node, graph, nullptr, 0, _1, _2, _3, _4), dst, src,
      sizeof(dst), hipMemcpyDefault);

// Disabled on AMD due to defect - EXSWHTEC-211
#if HT_NVIDIA
  SECTION("count == 0") {
    HIP_CHECK_ERROR(
        hipGraphAddMemcpyNode1D(&node, graph, nullptr, 0, dst, src, 0, hipMemcpyDefault),
        hipErrorInvalidValue);
  }
#endif

  SECTION("count larger than dst allocation size") {
    LinearAllocGuard<int> dev_dst(LinearAllocs::hipMalloc, sizeof(int));
    HIP_CHECK_ERROR(hipGraphAddMemcpyNode1D(&node, graph, nullptr, 0, dev_dst.ptr(), src,
                                            sizeof(src), hipMemcpyDefault),
                    hipErrorInvalidValue);
  }

  SECTION("count larger than src allocation size") {
    LinearAllocGuard<int> dev_src(LinearAllocs::hipMalloc, sizeof(int));
    HIP_CHECK_ERROR(hipGraphAddMemcpyNode1D(&node, graph, nullptr, 0, dst, dev_src.ptr(),
                                            sizeof(dst), hipMemcpyDefault),
                    hipErrorInvalidValue);
  }

  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * End doxygen group GraphTest.
 * @}
 */

namespace {
__global__ void ScaleAddKernel(int* out, const int* in, int add, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    out[i] = in[i] * 2 + add;
  }
}
}  // namespace

/**
 * Test Description
 * ------------------------
 *  - A chain kernel -> D2H copy -> H2D copy -> kernel -> D2H copy, with copies large enough to
 *    take SDMA, replayed back to back. Each copy must see the previous kernel's writes and each
 *    kernel the previous copy's, with no markers forced between the nodes.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphAddMemcpyNode1D.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipGraphAddMemcpyNode1D_Positive_KernelCopyChainOrdering) {
  constexpr size_t kCount = (8u << 20) / sizeof(int);
  constexpr size_t kBytes = kCount * sizeof(int);
  constexpr int kLaunches = 8;

  LinearAllocGuard<int> dev_in(LinearAllocs::hipMalloc, kBytes);
  LinearAllocGuard<int> dev_mid(LinearAllocs::hipMalloc, kBytes);
  LinearAllocGuard<int> dev_out(LinearAllocs::hipMalloc, kBytes);
  LinearAllocGuard<int> host_mid(LinearAllocs::hipHostMalloc, kBytes);
  LinearAllocGuard<int> host_out(LinearAllocs::hipHostMalloc, kBytes);
  for (size_t i = 0; i < kCount; i++) host_mid.host_ptr()[i] = -1;
  std::vector<int> init(kCount);
  for (size_t i = 0; i < kCount; i++) init[i] = static_cast<int>(i % 1000);
  HIP_CHECK(hipMemcpy(dev_in.ptr(), init.data(), kBytes, hipMemcpyHostToDevice));

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t k1, d2h1, h2d, k2, d2h2;
  int* in_ptr = dev_in.ptr();
  int* mid_ptr = dev_mid.ptr();
  int* out_ptr = dev_out.ptr();
  size_t count = kCount;
  int add1 = 1, add2 = 3;
  void* args1[] = {&mid_ptr, &in_ptr, &add1, &count};
  void* args2[] = {&out_ptr, &in_ptr, &add2, &count};
  hipKernelNodeParams kp{};
  kp.func = reinterpret_cast<void*>(ScaleAddKernel);
  kp.gridDim = dim3(256);
  kp.blockDim = dim3(256);
  kp.kernelParams = args1;
  HIP_CHECK(hipGraphAddKernelNode(&k1, graph, nullptr, 0, &kp));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&d2h1, graph, &k1, 1, host_mid.host_ptr(), mid_ptr, kBytes,
                                    hipMemcpyDeviceToHost));
  // Round trip through the host into dev_in, which the second kernel reads
  HIP_CHECK(hipGraphAddMemcpyNode1D(&h2d, graph, &d2h1, 1, in_ptr, host_mid.host_ptr(), kBytes,
                                    hipMemcpyHostToDevice));
  kp.kernelParams = args2;
  HIP_CHECK(hipGraphAddKernelNode(&k2, graph, &h2d, 1, &kp));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&d2h2, graph, &k2, 1, host_out.host_ptr(), out_ptr, kBytes,
                                    hipMemcpyDeviceToHost));

  hipGraphExec_t exec;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  StreamGuard stream_guard(Streams::created);
  for (int l = 0; l < kLaunches; l++) {
    HIP_CHECK(hipGraphLaunch(exec, stream_guard.stream()));
  }
  HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));

  // Each launch: mid = 2*in + 1, in = mid, out = 2*in + 3. Track the values on the host.
  std::vector<long long> v(init.begin(), init.end());
  for (int l = 0; l < kLaunches; l++) {
    for (auto& x : v) x = static_cast<int>(x * 2 + 1);
  }
  for (size_t i = 0; i < kCount; i += 997) {
    const int expected_mid = static_cast<int>(v[i]);
    const int expected_out = static_cast<int>(static_cast<int>(v[i]) * 2 + 3);
    INFO("index " << i);
    REQUIRE(host_mid.host_ptr()[i] == expected_mid);
    REQUIRE(host_out.host_ptr()[i] == expected_out);
  }

  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
}
