/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <functional>

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <hip_test_checkers.hh>
#include <memcpy3d_tests_common.hh>

#include "numeric"
#include "graph_tests_common.hh"

#define XSIZE 32

/**
 * @addtogroup hipDrvGraphAddMemcpyNode hipDrvGraphAddMemcpyNode
 * @{
 * @ingroup GraphTest
 * `hipDrvGraphAddMemcpyNode(hipGraphNode_t *pGraphNode, hipGraph_t graph, const
 * hipGraphNode_t *pDependencies, size_t numDependencies, const HIP_MEMCPY3D* copyParams, hipCtx_t
 ctx)`
 - Creates a memcpy node and adds it to a graph
 */

// APIs hipDrvGraphMemcpyNodeGetParams, hipDrvGraphMemcpyNodeSetParams are yet to be implemented in
// HIP runtime.
#if 0
/**
 * Test Description
 * ------------------------
 *  - Negative Scenario - for API hipDrvGraphAddMemcpyNode
 * 1) Pass memcpyNode as nullptr
 * 2) Pass memcpyNode as empty structure
 * 3) Pass graph as nullptr
 * 4) Pass myparams as nullptr
 * 5) Pass context as nullptr
 * 6) When numDependencies is max & pDependencies is not valid ptr
 * 7) When pDependencies is nullptr, but numDependencies is non-zero
 * 8) API expects atleast one memcpy Source pointer to be set
 * 9) API expects atleast one memcpy Destination pointer to be set
 * 10) Passing different element size for srcArray and dstArray
 * Test source
 * ------------------------
 *  - unit/graph/hipDrvGraphAddMemcpyNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.1
 */
#if HT_AMD
TEST_CASE("Unit_hipDrvGraphAddMemcpyNode_Negative") {
  CHECK_IMAGE_SUPPORT

  constexpr size_t size = 1024;
  size_t numW = size * sizeof(int);
  // Host Vectors
  std::vector<int> A_h(numW);
  std::vector<int> B_h(numW);
  hipCtx_t context;
  // Initialization
  std::iota(A_h.begin(), A_h.end(), 0);
  std::fill_n(B_h.begin(), size, 0);

  hipError_t ret;
  hipGraph_t graph;
  hipStream_t streamForGraph;
  hipGraphNode_t memcpyNode;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipCtxCreate(&context, 0, 0));

  HIP_MEMCPY3D myparams{};
  memset(&myparams, 0x0, sizeof(HIP_MEMCPY3D));
  myparams.srcXInBytes = 0;
  myparams.srcY = 0;
  myparams.srcZ = 0;
  myparams.dstXInBytes = 0;
  myparams.dstY = 0;
  myparams.dstZ = 0;
  myparams.WidthInBytes = numW;
  myparams.Height = 1;
  myparams.Depth = 1;
  myparams.srcMemoryType = hipMemoryTypeHost;
  myparams.dstMemoryType = hipMemoryTypeHost;
  myparams.srcHost = A_h.data();
  myparams.srcPitch = numW;
  myparams.srcHeight = 1;
  myparams.dstHost = B_h.data();
  myparams.dstPitch = numW;
  myparams.dstHeight = 1;

  SECTION("Pass memcpyNode as nullptr") {
    ret = hipDrvGraphAddMemcpyNode(nullptr, graph, nullptr,
                                   0, &myparams, context);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass memcpyNode as empty structure") {
    hipGraphNode_t memcpyNode_t = {};
    ret = hipDrvGraphAddMemcpyNode(&memcpyNode_t, graph, nullptr,
                                   0, &myparams, context);
    REQUIRE(hipSuccess == ret);
  }
  SECTION("Pass graph as nullptr") {
    ret = hipDrvGraphAddMemcpyNode(&memcpyNode, nullptr, nullptr,
                                   0, &myparams, context);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass myparams as nullptr") {
    ret = hipDrvGraphAddMemcpyNode(&memcpyNode, graph, nullptr,
                                   0, nullptr, context);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass context as nullptr") {
    ret = hipDrvGraphAddMemcpyNode(&memcpyNode, graph, nullptr,
                                   0, &myparams, nullptr);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("When numDependencies is max & pDependencies is not valid ptr") {
    ret = hipDrvGraphAddMemcpyNode(&memcpyNode, graph, nullptr,
                                   INT_MAX, &myparams, context);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("When pDependencies is nullptr, but numDependencies is non-zero") {
    ret = hipDrvGraphAddMemcpyNode(&memcpyNode, graph, nullptr,
                                   2, &myparams, context);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("API expects atleast one memcpy src pointer to be set") {
    memset(&myparams, 0x0, sizeof(HIP_MEMCPY3D));
    myparams.srcXInBytes = 0;
    myparams.srcY = 0;
    myparams.srcZ = 0;
    myparams.dstXInBytes = 0;
    myparams.dstY = 0;
    myparams.dstZ = 0;
    myparams.WidthInBytes = numW;
    myparams.Height = 1;
    myparams.Depth = 1;
    myparams.srcMemoryType = hipMemoryTypeHost;
    myparams.dstMemoryType = hipMemoryTypeHost;
    myparams.srcPitch = numW;
    myparams.srcHeight = 1;
    myparams.dstHost = B_h.data();
    myparams.dstPitch = numW;
    myparams.dstHeight = 1;

    ret = hipDrvGraphAddMemcpyNode(&memcpyNode, graph, nullptr,
                                   0, &myparams, context);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("API expects atleast one memcpy dst pointer to be set") {
    memset(&myparams, 0x0, sizeof(HIP_MEMCPY3D));
    myparams.srcXInBytes = 0;
    myparams.srcY = 0;
    myparams.srcZ = 0;
    myparams.dstXInBytes = 0;
    myparams.dstY = 0;
    myparams.dstZ = 0;
    myparams.WidthInBytes = numW;
    myparams.Height = 1;
    myparams.Depth = 1;
    myparams.srcMemoryType = hipMemoryTypeHost;
    myparams.dstMemoryType = hipMemoryTypeHost;
    myparams.srcHost = A_h.data();
    myparams.srcPitch = numW;
    myparams.srcHeight = 1;
    myparams.dstPitch = numW;
    myparams.dstHeight = 1;

    ret = hipDrvGraphAddMemcpyNode(&memcpyNode, graph, nullptr,
                                   0, &myparams, context);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass source memory type as hipMemoryTypeHost and set srcDevice") {
    memset(&myparams, 0x0, sizeof(HIP_MEMCPY3D));
    myparams.srcXInBytes = 0;
    myparams.srcY = 0;
    myparams.srcZ = 0;
    myparams.dstXInBytes = 0;
    myparams.dstY = 0;
    myparams.dstZ = 0;
    myparams.WidthInBytes = numW;
    myparams.Height = 1;
    myparams.Depth = 1;
    myparams.srcMemoryType = hipMemoryTypeHost;
    myparams.dstMemoryType = hipMemoryTypeHost;
    myparams.srcDevice = hipDeviceptr_t(A_h.data());
    myparams.dstHost = B_h.data();
    myparams.srcPitch = numW;
    myparams.srcHeight = 1;
    myparams.dstPitch = numW;
    myparams.dstHeight = 1;

    ret = hipDrvGraphAddMemcpyNode(&memcpyNode, graph, nullptr,
                                   0, &myparams, context);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass dest memory type as hipMemoryTypeHost and set dstDevice") {
    memset(&myparams, 0x0, sizeof(HIP_MEMCPY3D));
    myparams.srcXInBytes = 0;
    myparams.srcY = 0;
    myparams.srcZ = 0;
    myparams.dstXInBytes = 0;
    myparams.dstY = 0;
    myparams.dstZ = 0;
    myparams.WidthInBytes = numW;
    myparams.Height = 1;
    myparams.Depth = 1;
    myparams.srcMemoryType = hipMemoryTypeHost;
    myparams.dstMemoryType = hipMemoryTypeHost;
    myparams.srcHost = A_h.data();
    myparams.dstDevice = hipDeviceptr_t(B_h.data());
    myparams.srcPitch = numW;
    myparams.srcHeight = 1;
    myparams.dstPitch = numW;
    myparams.dstHeight = 1;

    ret = hipDrvGraphAddMemcpyNode(&memcpyNode, graph, nullptr,
                                   0, &myparams, context);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Passing different element size for HIP_MEMCPY3D::srcArray"
                  "and HIP_MEMCPY3D::dstArray") {
    memset(&myparams, 0x0, sizeof(HIP_MEMCPY3D));
    myparams.srcXInBytes = 0;
    myparams.srcY = 0;
    myparams.srcZ = 0;
    myparams.dstXInBytes = 0;
    myparams.dstY = 0;
    myparams.dstZ = 0;
    myparams.WidthInBytes = numW;
    myparams.Height = 1;
    myparams.Depth = 1;
    myparams.srcMemoryType = hipMemoryTypeHost;
    myparams.srcHost = A_h.data();
    myparams.srcDevice = hipDeviceptr_t(A_h.data());
    myparams.srcPitch = numW * sizeof(int);
    myparams.srcHeight = 1;
    myparams.dstMemoryType = hipMemoryTypeArray;
    hipChannelFormatDesc channelDesc = hipCreateChannelDesc(sizeof(int) * 8,
                                       0, 0, 0, hipChannelFormatKindSigned);
    hipArray *devArray1, *devArray2;
    HIP_CHECK(hipMalloc3DArray(&devArray1, &channelDesc,
                               make_hipExtent(10, 10, 10), hipArrayDefault));
    HIP_CHECK(hipMalloc3DArray(&devArray2, &channelDesc,
                               make_hipExtent(11, 11, 11), hipArrayDefault));
    myparams.srcArray = devArray1;
    myparams.dstArray = devArray2;
    ret = hipDrvGraphAddMemcpyNode(&memcpyNode, graph, nullptr,
                                   0, &myparams, context);
    REQUIRE(hipErrorInvalidValue == ret);
    HIP_CHECK(hipFreeArray(devArray1));
    HIP_CHECK(hipFreeArray(devArray2));
  }
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

/*
 * Create two host pointers, copy the data between them by the api
 * hipDrvGraphAddMemcpyNode with data transfer kind hipMemcpyHostToHost.
 * Validate the output.
 */
static void hipDrvGraphAddMemcpyNode_test(int deviceid = 0) {
  HIP_CHECK(hipSetDevice(deviceid));

  constexpr size_t size = 1024;
  size_t numW = size * sizeof(int);
  // Host Vectors
  std::vector<int> A_h(numW);
  std::vector<int> B_h(numW);
  hipCtx_t context;
  // Initialization
  std::iota(A_h.begin(), A_h.end(), 0);
  std::fill_n(B_h.begin(), size, 0);

  hipGraph_t graph;
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;
  hipGraphNode_t memcpyH2H;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipCtxCreate(&context, 0, deviceid));

  HIP_MEMCPY3D myparams{};
  memset(&myparams, 0x0, sizeof(HIP_MEMCPY3D));
  myparams.srcXInBytes = 0;
  myparams.srcY = 0;
  myparams.srcZ = 0;
  myparams.dstXInBytes = 0;
  myparams.dstY = 0;
  myparams.dstZ = 0;
  myparams.WidthInBytes = numW;
  myparams.Height = 1;
  myparams.Depth = 1;
  myparams.srcMemoryType = hipMemoryTypeHost;
  myparams.dstMemoryType = hipMemoryTypeHost;
  myparams.srcHost = A_h.data();
  myparams.srcPitch = numW;
  myparams.srcHeight = 1;
  myparams.dstHost = B_h.data();
  myparams.dstPitch = numW;
  myparams.dstHeight = 1;
  // Host to Host
  HIP_CHECK(hipDrvGraphAddMemcpyNode(&memcpyH2H, graph, nullptr,
                                     0, &myparams, context));
  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  // Validation
  REQUIRE(memcmp(A_h.data(), B_h.data(), numW) == 0);
}

/**
 * Test Description
 * ------------------------
 *  - Functional Scenario -
 * 1) Create two host pointers, copy the data between them by the api
 *    hipDrvGraphAddMemcpyNode with data transfer kind hipMemcpyHostToHost.
 *    Validate the output.
 * Test source
 * ------------------------
 *  - unit/graph/hipDrvGraphAddMemcpyNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.1
 */

TEST_CASE("Unit_hipDrvGraphAddMemcpyNode_test") {
  CHECK_IMAGE_SUPPORT

  hipDrvGraphAddMemcpyNode_test();
}

/**
 * Test Description
 * ------------------------
 *  - Functional Scenario - for Multiple device case
 * 1) Create two host pointers, copy the data between them by the api
 *    hipDrvGraphAddMemcpyNode with data transfer kind hipMemcpyHostToHost.
 *    Validate the output.
 * Test source
 * ------------------------
 *  - unit/graph/hipDrvGraphAddMemcpyNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.1
 */

TEST_CASE("Unit_hipDrvGraphAddMemcpyNode_MulitDevice") {
  CHECK_IMAGE_SUPPORT

  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices > 1) {
    for (int device=0; device < numDevices; device++) {
      hipDrvGraphAddMemcpyNode_test(device);
    }
  } else {
    SUCCEED("skipped the testcase as no of devices is less than 2");
  }
}
#endif

/**
 * Test Description
 * ------------------------
 *    - Verify basic API behavior. A Memcpy node is created with parameters set according to the
 * test run, after which the graph is run and the memcpy results are verified.
 * The test is run for all possible memcpy directions, with both the corresponding memcpy
 * kind and hipMemcpyDefault, as well as half page and full page allocation sizes.
 * Test source
 * ------------------------
 *    - unit/graph/hipDrvGraphAddMemcpyNode.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */

TEST_CASE("Unit_hipDrvGraphAddMemcpyNode_Positive_Basic") {
  using namespace std::placeholders;

  constexpr bool async = false;
  HIP_CHECK(hipInit(0));
  hipDevice_t device;
  hipCtx_t context;
  HIP_CHECK(hipDeviceGet(&device, 0));
  HIP_CHECK(hipCtxCreate(&context, 0, device));

  SECTION("Device to host") {
    Memcpy3DDeviceToHostShell<async>(
        std::bind(DrvMemcpy3DGraphWrapper<>, _1, _2, _3, _4, _5, _6, context, _7));
  }

  SECTION("Host to device") {
    Memcpy3DHostToDeviceShell<async>(
        std::bind(DrvMemcpy3DGraphWrapper<>, _1, _2, _3, _4, _5, _6, context, _7));
  }

  SECTION("Host to host") {
    Memcpy3DHostToHostShell<async>(
        std::bind(DrvMemcpy3DGraphWrapper<>, _1, _2, _3, _4, _5, _6, context, _7));
  }

  SECTION("Device to device") {
    SECTION("Peer access enabled") {
      Memcpy3DDeviceToDeviceShell<async, true>(
          std::bind(DrvMemcpy3DGraphWrapper<>, _1, _2, _3, _4, _5, _6, context, _7));
    }
    SECTION("Peer access disabled") {
      Memcpy3DDeviceToDeviceShell<async, false>(
          std::bind(DrvMemcpy3DGraphWrapper<>, _1, _2, _3, _4, _5, _6, context, _7));
    }
  }

  HIP_CHECK(hipCtxPopCurrent(&context));
  HIP_CHECK(hipCtxDestroy(context));
}

TEST_CASE("Unit_hipDrvGraphAddMemcpyNode_Positive_Array") {
  CHECK_IMAGE_SUPPORT

  using namespace std::placeholders;

  constexpr bool async = false;
  HIP_CHECK(hipInit(0));
  hipDevice_t device;
  hipCtx_t context;
  HIP_CHECK(hipDeviceGet(&device, 0));
  HIP_CHECK(hipCtxCreate(&context, 0, device));

  SECTION("Array from/to Host") {
    DrvMemcpy3DArrayHostShell<async>(
        std::bind(DrvMemcpy3DGraphWrapper<>, _1, _2, _3, _4, _5, _6, context, _7));
  }
  SECTION("Array from/to Device") {
    DrvMemcpy3DArrayDeviceShell<async>(
        std::bind(DrvMemcpy3DGraphWrapper<>, _1, _2, _3, _4, _5, _6, context, _7));
  }

  HIP_CHECK(hipCtxPopCurrent(&context));
  HIP_CHECK(hipCtxDestroy(context));
}
#endif  // if 0

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
 *        -# dstPitch < width
 *        -# srcPitch < width
 *        -# dstPitch > max pitch
 *        -# srcPitch > max pitch
 *        -# WidthInBytes + dstXInBytes > dstPitch
 *        -# WidthInBytes + srcXInBytes > srcPitch
 *        -# dstY out of bounds
 *        -# srcY out of bounds
 *        -# dstZ out of bounds
 *        -# srcZ out of bounds
 * Test source
 * ------------------------
 *    - unit/graph/hipDrvGraphAddMemcpyNode.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipDrvGraphAddMemcpyNode_Negative_Parameters") {
  using namespace std::placeholders;

  HIP_CHECK(hipInit(0));
  hipDevice_t device;
  hipCtx_t context;
  HIP_CHECK(hipDeviceGet(&device, 0));
  HIP_CHECK(hipCtxCreate(&context, 0, device));

  constexpr hipExtent extent{128 * sizeof(int), 128, 8};

  constexpr auto NegativeTests = [](hipPitchedPtr dst_ptr, hipPos dst_pos, hipPitchedPtr src_ptr,
                                    hipPos src_pos, hipExtent extent, hipMemcpyKind kind,
                                    hipCtx_t context) {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipGraphCreate(&graph, 0));
    hipGraphNode_t node = nullptr;

    auto params = GetDrvMemcpy3DParms(dst_ptr, dst_pos, src_ptr, src_pos, extent, kind);
    GraphAddNodeCommonNegativeTests(
        std::bind(hipDrvGraphAddMemcpyNode, _1, _2, _3, _4, &params, context), graph);

    SECTION("dst_ptr.ptr == nullptr") {
      hipPitchedPtr invalid_ptr = dst_ptr;
      invalid_ptr.ptr = nullptr;
      auto params = GetDrvMemcpy3DParms(invalid_ptr, dst_pos, src_ptr, src_pos, extent, kind);
      HIP_CHECK_ERROR(hipDrvGraphAddMemcpyNode(&node, graph, nullptr, 0, &params, context),
                      hipErrorInvalidValue);
    }

    SECTION("src_ptr.ptr == nullptr") {
      hipPitchedPtr invalid_ptr = src_ptr;
      invalid_ptr.ptr = nullptr;
      auto params = GetDrvMemcpy3DParms(dst_ptr, dst_pos, invalid_ptr, src_pos, extent, kind);
      HIP_CHECK_ERROR(hipDrvGraphAddMemcpyNode(&node, graph, nullptr, 0, &params, context),
                      hipErrorInvalidValue);
    }

    SECTION("dstPitch < width") {
      hipPitchedPtr invalid_ptr = dst_ptr;
      invalid_ptr.pitch = extent.width - 1;
      auto params = GetDrvMemcpy3DParms(invalid_ptr, dst_pos, src_ptr, src_pos, extent, kind);
      HIP_CHECK_ERROR(hipDrvGraphAddMemcpyNode(&node, graph, nullptr, 0, &params, context),
                      hipErrorInvalidValue);
    }

    SECTION("srcPitch < width") {
      hipPitchedPtr invalid_ptr = src_ptr;
      invalid_ptr.pitch = extent.width - 1;
      auto params = GetDrvMemcpy3DParms(dst_ptr, dst_pos, invalid_ptr, src_pos, extent, kind);
      HIP_CHECK_ERROR(hipDrvGraphAddMemcpyNode(&node, graph, nullptr, 0, &params, context),
                      hipErrorInvalidValue);
    }

    SECTION("dstPitch > max pitch") {
      int attr = 0;
      HIP_CHECK(hipDeviceGetAttribute(&attr, hipDeviceAttributeMaxPitch, 0));
      hipPitchedPtr invalid_ptr = dst_ptr;
      invalid_ptr.pitch = attr;
      auto params = GetDrvMemcpy3DParms(invalid_ptr, dst_pos, src_ptr, src_pos, extent, kind);
      HIP_CHECK_ERROR(hipDrvGraphAddMemcpyNode(&node, graph, nullptr, 0, &params, context),
                      hipErrorInvalidValue);
    }

    SECTION("srcPitch > max pitch") {
      int attr = 0;
      HIP_CHECK(hipDeviceGetAttribute(&attr, hipDeviceAttributeMaxPitch, 0));
      hipPitchedPtr invalid_ptr = src_ptr;
      invalid_ptr.pitch = attr;
      auto params = GetDrvMemcpy3DParms(dst_ptr, dst_pos, invalid_ptr, src_pos, extent, kind);
      HIP_CHECK_ERROR(hipDrvGraphAddMemcpyNode(&node, graph, nullptr, 0, &params, context),
                      hipErrorInvalidValue);
    }

    SECTION("WidthInBytes + dstXInBytes > dstPitch") {
      hipPos invalid_pos = dst_pos;
      invalid_pos.x = dst_ptr.pitch - extent.width + 1;
      auto params = GetDrvMemcpy3DParms(dst_ptr, invalid_pos, src_ptr, src_pos, extent, kind);
      HIP_CHECK_ERROR(hipDrvGraphAddMemcpyNode(&node, graph, nullptr, 0, &params, context),
                      hipErrorInvalidValue);
    }

    SECTION("WidthInBytes + srcXInBytes > srcPitch") {
      hipPos invalid_pos = src_pos;
      invalid_pos.x = src_ptr.pitch - extent.width + 1;
      auto params = GetDrvMemcpy3DParms(dst_ptr, dst_pos, src_ptr, invalid_pos, extent, kind);
      HIP_CHECK_ERROR(hipDrvGraphAddMemcpyNode(&node, graph, nullptr, 0, &params, context),
                      hipErrorInvalidValue);
    }

    SECTION("dstY out of bounds") {
      hipPos invalid_pos = dst_pos;
      invalid_pos.y = 1;
      auto params = GetDrvMemcpy3DParms(dst_ptr, invalid_pos, src_ptr, src_pos, extent, kind);
      HIP_CHECK_ERROR(hipDrvGraphAddMemcpyNode(&node, graph, nullptr, 0, &params, context),
                      hipErrorInvalidValue);
    }

    SECTION("srcY out of bounds") {
      hipPos invalid_pos = src_pos;
      invalid_pos.y = 1;
      auto params = GetDrvMemcpy3DParms(dst_ptr, dst_pos, src_ptr, invalid_pos, extent, kind);
      HIP_CHECK_ERROR(hipDrvGraphAddMemcpyNode(&node, graph, nullptr, 0, &params, context),
                      hipErrorInvalidValue);
    }

    SECTION("dstZ out of bounds") {
      hipPos invalid_pos = dst_pos;
      invalid_pos.z = 1;
      auto params = GetDrvMemcpy3DParms(dst_ptr, invalid_pos, src_ptr, src_pos, extent, kind);
      HIP_CHECK_ERROR(hipDrvGraphAddMemcpyNode(&node, graph, nullptr, 0, &params, context),
                      hipErrorInvalidValue);
    }

    SECTION("srcZ out of bounds") {
      hipPos invalid_pos = src_pos;
      invalid_pos.z = 1;
      auto params = GetDrvMemcpy3DParms(dst_ptr, dst_pos, src_ptr, invalid_pos, extent, kind);
      HIP_CHECK_ERROR(hipDrvGraphAddMemcpyNode(&node, graph, nullptr, 0, &params, context),
                      hipErrorInvalidValue);
    }

    HIP_CHECK(hipGraphDestroy(graph));
  };

  SECTION("Host to Device") {
    LinearAllocGuard3D<int> device_alloc(extent);
    LinearAllocGuard<int> host_alloc(
        LinearAllocs::hipHostMalloc,
        device_alloc.pitch() * device_alloc.height() * device_alloc.depth());
    NegativeTests(device_alloc.pitched_ptr(), make_hipPos(0, 0, 0),
                  make_hipPitchedPtr(host_alloc.ptr(), device_alloc.pitch(), device_alloc.width(),
                                     device_alloc.height()),
                  make_hipPos(0, 0, 0), extent, hipMemcpyHostToDevice, context);
  }

  SECTION("Device to Host") {
    LinearAllocGuard3D<int> device_alloc(extent);
    LinearAllocGuard<int> host_alloc(
        LinearAllocs::hipHostMalloc,
        device_alloc.pitch() * device_alloc.height() * device_alloc.depth());
    NegativeTests(make_hipPitchedPtr(host_alloc.ptr(), device_alloc.pitch(), device_alloc.width(),
                                     device_alloc.height()),
                  make_hipPos(0, 0, 0), device_alloc.pitched_ptr(), make_hipPos(0, 0, 0), extent,
                  hipMemcpyDeviceToHost, context);
  }

  SECTION("Host to Host") {
    LinearAllocGuard<int> src_alloc(LinearAllocs::hipHostMalloc,
                                    extent.width * extent.height * extent.depth);
    LinearAllocGuard<int> dst_alloc(LinearAllocs::hipHostMalloc,
                                    extent.width * extent.height * extent.depth);
    NegativeTests(make_hipPitchedPtr(dst_alloc.ptr(), extent.width, extent.width, extent.height),
                  make_hipPos(0, 0, 0),
                  make_hipPitchedPtr(src_alloc.ptr(), extent.width, extent.width, extent.height),
                  make_hipPos(0, 0, 0), extent, hipMemcpyHostToHost, context);
  }

  SECTION("Device to Device") {
    LinearAllocGuard3D<int> src_alloc(extent);
    LinearAllocGuard3D<int> dst_alloc(extent);
    NegativeTests(dst_alloc.pitched_ptr(), make_hipPos(0, 0, 0), src_alloc.pitched_ptr(),
                  make_hipPos(0, 0, 0), extent, hipMemcpyDeviceToDevice, context);
  }

  HIP_CHECK(hipCtxPopCurrent(&context));
  HIP_CHECK(hipCtxDestroy(context));
}

/**
 * End doxygen group GraphTest.
 * @}
 */
