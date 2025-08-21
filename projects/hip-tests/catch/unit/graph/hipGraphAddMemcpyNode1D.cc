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
TEST_CASE("Unit_hipGraphAddMemcpyNode1D_Positive_Basic") {
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
TEST_CASE("Unit_hipGraphAddMemcpyNode1D_Negative_Parameters") {
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
