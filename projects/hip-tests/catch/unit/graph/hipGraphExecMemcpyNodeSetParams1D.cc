/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <functional>

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <memcpy1d_tests_common.hh>

#include "graph_tests_common.hh"

/**
 * @addtogroup hipGraphExecMemcpyNodeSetParams1D hipGraphExecMemcpyNodeSetParams1D
 * @{
 * @ingroup GraphTest
 * `hipGraphExecMemcpyNodeSetParams1D(hipGraphExec_t hGraphExec, hipGraphNode_t node, void *dst,
 * const void *src, size_t count, hipMemcpyKind kind)` - Sets the parameters for a memcpy node in
 * the given graphExec to perform a 1-dimensional copy
 */

/**
 * Test Description
 * ------------------------
 *    - Verify that node parameters get updated correctly by creating a node with valid but
 * incorrect parameters, and the setting them to the correct values in the executable graph. The
 * executable graph is run and the results of the memcpy verified. The test is run for all possible
 * memcpy directions, with both the corresponding memcpy kind and hipMemcpyDefault, as well as half
 * page and full page allocation sizes. Test source
 * ------------------------
 *    - unit/graph/hipGraphExecMemcpyNodeSetParams1D.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphExecMemcpyNodeSetParams1D_Positive_Basic") {
  constexpr auto f = [](void* dst, void* src, size_t count, hipMemcpyKind direction) {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipGraphCreate(&graph, 0));
    hipGraphNode_t node = nullptr;
    const auto offset_src = reinterpret_cast<uint8_t*>(src) + 1;
    const auto offset_dst = reinterpret_cast<uint8_t*>(dst) + 1;
    HIP_CHECK(hipGraphAddMemcpyNode1D(&node, graph, nullptr, 0, offset_dst, offset_src, count - 1,
                                      direction));
    hipGraphExec_t graph_exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphExecMemcpyNodeSetParams1D(graph_exec, node, dst, src, count, direction));
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

  SECTION("Host to device") {
    MemcpyHostToDeviceShell<false>(std::bind(f, _1, _2, _3, hipMemcpyHostToDevice));
  }

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

// Disabled on AMD due to defect - EXSWHTEC-209
#if 0
  SECTION("Host to host") {
    MemcpyHostToHostShell<false>(std::bind(f, _1, _2, _3, hipMemcpyHostToHost));
  }

  SECTION("Host to host with default kind") {
    MemcpyHostToHostShell<false>(std::bind(f, _1, _2, _3, hipMemcpyDefault));
  }
#endif

// Disabled on AMD due to defect - EXSWHTEC-210
#if 0
  SECTION("Device to host with default kind") {
    MemcpyDeviceToHostShell<false>(std::bind(f, _1, _2, _3, hipMemcpyDefault));
  }

  SECTION("Host to device with default kind") {
    MemcpyHostToDeviceShell<false>(std::bind(f, _1, _2, _3, hipMemcpyDefault));
  }
#endif

#endif
}

/**
 * Test Description
 * ------------------------
 *    - Verify API behaviour with invalid arguments:
 *        -# pGraphExec is nullptr
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
TEST_CASE("Unit_hipGraphExecMemcpyNodeSetParams1D_Negative_Parameters") {
  using namespace std::placeholders;
  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  int src[2] = {}, dst[2] = {};

  hipGraphNode_t node = nullptr;
  HIP_CHECK(
      hipGraphAddMemcpyNode1D(&node, graph, nullptr, 0, dst, src, sizeof(dst), hipMemcpyDefault));

  hipGraphExec_t graph_exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

  SECTION("pGraphExec == nullptr") {
    HIP_CHECK_ERROR(
        hipGraphExecMemcpyNodeSetParams1D(nullptr, node, dst, src, sizeof(dst), hipMemcpyDefault),
        hipErrorInvalidValue);
  }

  SECTION("node == nullptr") {
    HIP_CHECK_ERROR(hipGraphExecMemcpyNodeSetParams1D(graph_exec, nullptr, dst, src, sizeof(dst),
                                                      hipMemcpyDefault),
                    hipErrorInvalidValue);
  }

  MemcpyWithDirectionCommonNegativeTests(
      std::bind(hipGraphExecMemcpyNodeSetParams1D, graph_exec, node, _1, _2, _3, _4), dst, src,
      sizeof(dst), hipMemcpyDefault);

  SECTION("count == 0") {
    HIP_CHECK_ERROR(
        hipGraphExecMemcpyNodeSetParams1D(graph_exec, node, dst, src, 0, hipMemcpyDefault),
        hipErrorInvalidValue);
  }

  SECTION("count larger than dst allocation size") {
    LinearAllocGuard<int> dev_dst(LinearAllocs::hipMalloc, sizeof(int));
    HIP_CHECK_ERROR(hipGraphExecMemcpyNodeSetParams1D(graph_exec, node, dev_dst.ptr(), src,
                                                      sizeof(src), hipMemcpyDefault),
                    hipErrorInvalidValue);
  }

  SECTION("count larger than src allocation size") {
    LinearAllocGuard<int> dev_src(LinearAllocs::hipMalloc, sizeof(int));
    HIP_CHECK_ERROR(hipGraphExecMemcpyNodeSetParams1D(graph_exec, node, dst, dev_src.ptr(),
                                                      sizeof(dst), hipMemcpyDefault),
                    hipErrorInvalidValue);
  }

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    - Verify that memcpy direction cannot be altered in an executable graph. The test is run for
 * all memcpy directions with appropriate memory allocations.
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphExecMemcpyNodeSetParams1D.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphExecMemcpyNodeSetParams1D_Negative_Changing_Memcpy_Direction") {
  int *host1, *host2, *dev1, *dev2;
  HIP_CHECK(hipHostMalloc(&host1, sizeof(int)));
  HIP_CHECK(hipHostMalloc(&host2, sizeof(int)));
  HIP_CHECK(hipMalloc(&dev1, sizeof(int)));
  HIP_CHECK(hipMalloc(&dev2, sizeof(int)));

  const auto [dir, src, dst] = GENERATE_REF(std::make_tuple(hipMemcpyHostToHost, host1, host2),
                                            std::make_tuple(hipMemcpyHostToDevice, host1, dev1),
                                            std::make_tuple(hipMemcpyDeviceToHost, dev1, host1),
                                            std::make_tuple(hipMemcpyDeviceToDevice, dev1, dev2));

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t node = nullptr;
  HIP_CHECK(hipGraphAddMemcpyNode1D(&node, graph, nullptr, 0, dst, src, sizeof(int), dir));

  hipGraphExec_t graph_exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

  const auto set_dir = GENERATE(hipMemcpyHostToHost, hipMemcpyHostToDevice, hipMemcpyDeviceToHost,
                                hipMemcpyDeviceToDevice, hipMemcpyDefault);
  if (dir == set_dir) {
    HIP_CHECK(hipGraphExecDestroy(graph_exec));
    HIP_CHECK(hipGraphDestroy(graph));
    return;
  }

  HIP_CHECK_ERROR(
      hipGraphExecMemcpyNodeSetParams1D(graph_exec, node, dst, src, sizeof(int), set_dir),
      hipErrorInvalidValue);

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));

  HIP_CHECK(hipHostFree(host1));
  HIP_CHECK(hipHostFree(host2));
  HIP_CHECK(hipFree(dev1));
  HIP_CHECK(hipFree(dev2));
}

/**
 * End doxygen group GraphTest.
 * @}
 */
