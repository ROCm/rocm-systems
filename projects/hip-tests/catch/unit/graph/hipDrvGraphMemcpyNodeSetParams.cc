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
#include <memcpy3d_tests_common.hh>

/**
 * @addtogroup hipDrvGraphMemcpyNodeSetParams hipDrvGraphMemcpyNodeSetParams
 * @{
 * @ingroup GraphTest
 * `hipDrvGraphMemcpyNodeSetParams(hipGraphNode_t hNode, const HIP_MEMCPY3D* nodeParams)` - Sets a
 * memcpy node's parameters
 */

/**
 * Test Description
 * ------------------------
 *    - Verify that node parameters get updated correctly by creating a node with valid but
 * incorrect parameters, and then setting them to the correct values after which the graph is
 * executed and the results of the memcpy verified.
 * The test is run for all possible memcpy directions, with both the corresponding memcpy
 * kind and hipMemcpyDefault, as well as half page and full page allocation sizes.
 * Test source
 * ------------------------
 *    - unit/graph/hipDrvGraphMemcpyNodeSetParams.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipDrvGraphMemcpyNodeSetParams_Positive_Basic") {
  using namespace std::placeholders;

  constexpr bool async = false;
  HIP_CHECK(hipInit(0));
  hipDevice_t device;
  hipCtx_t context;
  HIP_CHECK(hipDeviceGet(&device, 0));
  HIP_CHECK(hipCtxCreate(&context, 0, device));

  SECTION("Device to host") {
    Memcpy3DDeviceToHostShell<async>(
        std::bind(DrvMemcpy3DGraphWrapper<true>, _1, _2, _3, _4, _5, _6, context, _7));
  }

  SECTION("Host to device") {
    Memcpy3DHostToDeviceShell<async>(
        std::bind(DrvMemcpy3DGraphWrapper<true>, _1, _2, _3, _4, _5, _6, context, _7));
  }

  SECTION("Host to host") {
    Memcpy3DHostToHostShell<async>(
        std::bind(DrvMemcpy3DGraphWrapper<true>, _1, _2, _3, _4, _5, _6, context, _7));
  }

  SECTION("Device to device") {
    SECTION("Peer access enabled") {
      Memcpy3DDeviceToDeviceShell<async, true>(
          std::bind(DrvMemcpy3DGraphWrapper<true>, _1, _2, _3, _4, _5, _6, context, _7));
    }
    SECTION("Peer access disabled") {
      Memcpy3DDeviceToDeviceShell<async, false>(
          std::bind(DrvMemcpy3DGraphWrapper<true>, _1, _2, _3, _4, _5, _6, context, _7));
    }
    /*
     * Setting back the old context, since in the Memcpy3DDeviceToDeviceShell
     * function the current context is getting modified with hipSetDevice calls
     */
    HIP_CHECK(hipCtxSetCurrent(context));
  }

  HIP_CHECK(hipCtxPopCurrent(&context));
  HIP_CHECK(hipCtxDestroy(context));
}

TEST_CASE("Unit_hipDrvGraphMemcpyNodeSetParams_Positive_Array") {
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
        std::bind(DrvMemcpy3DGraphWrapper<true>, _1, _2, _3, _4, _5, _6, context, _7));
  }
  SECTION("Array from/to Device") {
    DrvMemcpy3DArrayDeviceShell<async>(
        std::bind(DrvMemcpy3DGraphWrapper<true>, _1, _2, _3, _4, _5, _6, context, _7));
  }

  HIP_CHECK(hipCtxPopCurrent(&context));
  HIP_CHECK(hipCtxDestroy(context));
}


/**
 * Test Description
 * ------------------------
 *    - Verify API behaviour with invalid arguments:
 *        -# node is nullptr
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
 *    - unit/graph/hipDrvGraphMemcpyNodeSetParams.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipDrvGraphMemcpyNodeSetParams_Negative_Parameters") {
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
    HIP_CHECK(hipDrvGraphAddMemcpyNode(&node, graph, nullptr, 0, &params, context));

    SECTION("node == nullptr") {
      HIP_CHECK_ERROR(hipDrvGraphMemcpyNodeSetParams(nullptr, &params), hipErrorInvalidValue);
    }

    SECTION("dst_ptr.ptr == nullptr") {
      hipPitchedPtr invalid_ptr = dst_ptr;
      invalid_ptr.ptr = nullptr;
      auto invalid_params =
          GetDrvMemcpy3DParms(invalid_ptr, dst_pos, src_ptr, src_pos, extent, kind);
      HIP_CHECK_ERROR(hipDrvGraphMemcpyNodeSetParams(node, &invalid_params), hipErrorInvalidValue);
    }

    SECTION("src_ptr.ptr == nullptr") {
      hipPitchedPtr invalid_ptr = src_ptr;
      invalid_ptr.ptr = nullptr;
      auto invalid_params =
          GetDrvMemcpy3DParms(dst_ptr, dst_pos, invalid_ptr, src_pos, extent, kind);
      HIP_CHECK_ERROR(hipDrvGraphMemcpyNodeSetParams(node, &invalid_params), hipErrorInvalidValue);
    }

    SECTION("dstPitch < width") {
      hipPitchedPtr invalid_ptr = dst_ptr;
      invalid_ptr.pitch = extent.width - 1;
      auto invalid_params =
          GetDrvMemcpy3DParms(invalid_ptr, dst_pos, src_ptr, src_pos, extent, kind);
      HIP_CHECK_ERROR(hipDrvGraphMemcpyNodeSetParams(node, &invalid_params), hipErrorInvalidValue);
    }

    SECTION("srcPitch < width") {
      hipPitchedPtr invalid_ptr = src_ptr;
      invalid_ptr.pitch = extent.width - 1;
      auto invalid_params =
          GetDrvMemcpy3DParms(dst_ptr, dst_pos, invalid_ptr, src_pos, extent, kind);
      HIP_CHECK_ERROR(hipDrvGraphMemcpyNodeSetParams(node, &invalid_params), hipErrorInvalidValue);
    }

    SECTION("dstPitch > max pitch") {
      int attr = 0;
      HIP_CHECK(hipDeviceGetAttribute(&attr, hipDeviceAttributeMaxPitch, 0));
      hipPitchedPtr invalid_ptr = dst_ptr;
      invalid_ptr.pitch = attr;
      auto invalid_params =
          GetDrvMemcpy3DParms(invalid_ptr, dst_pos, src_ptr, src_pos, extent, kind);
      HIP_CHECK_ERROR(hipDrvGraphMemcpyNodeSetParams(node, &invalid_params), hipErrorInvalidValue);
    }

    SECTION("srcPitch > max pitch") {
      int attr = 0;
      HIP_CHECK(hipDeviceGetAttribute(&attr, hipDeviceAttributeMaxPitch, 0));
      hipPitchedPtr invalid_ptr = src_ptr;
      invalid_ptr.pitch = attr;
      auto invalid_params =
          GetDrvMemcpy3DParms(dst_ptr, dst_pos, invalid_ptr, src_pos, extent, kind);
      HIP_CHECK_ERROR(hipDrvGraphMemcpyNodeSetParams(node, &invalid_params), hipErrorInvalidValue);
    }

    SECTION("WidthInBytes + dstXInBytes > dstPitch") {
      hipPos invalid_pos = dst_pos;
      invalid_pos.x = dst_ptr.pitch - extent.width + 1;
      auto invalid_params =
          GetDrvMemcpy3DParms(dst_ptr, invalid_pos, src_ptr, src_pos, extent, kind);
      HIP_CHECK_ERROR(hipDrvGraphMemcpyNodeSetParams(node, &invalid_params), hipErrorInvalidValue);
    }

    SECTION("WidthInBytes + srcXInBytes > srcPitch") {
      hipPos invalid_pos = src_pos;
      invalid_pos.x = src_ptr.pitch - extent.width + 1;
      auto invalid_params =
          GetDrvMemcpy3DParms(dst_ptr, dst_pos, src_ptr, invalid_pos, extent, kind);
      HIP_CHECK_ERROR(hipDrvGraphMemcpyNodeSetParams(node, &invalid_params), hipErrorInvalidValue);
    }

    SECTION("dstY out of bounds") {
      hipPos invalid_pos = dst_pos;
      invalid_pos.y = 1;
      auto invalid_params =
          GetDrvMemcpy3DParms(dst_ptr, invalid_pos, src_ptr, src_pos, extent, kind);
      HIP_CHECK_ERROR(hipDrvGraphMemcpyNodeSetParams(node, &invalid_params), hipErrorInvalidValue);
    }

    SECTION("srcY out of bounds") {
      hipPos invalid_pos = src_pos;
      invalid_pos.y = 1;
      auto invalid_params =
          GetDrvMemcpy3DParms(dst_ptr, dst_pos, src_ptr, invalid_pos, extent, kind);
      HIP_CHECK_ERROR(hipDrvGraphMemcpyNodeSetParams(node, &invalid_params), hipErrorInvalidValue);
    }

    SECTION("dstZ out of bounds") {
      hipPos invalid_pos = dst_pos;
      invalid_pos.z = 1;
      auto invalid_params =
          GetDrvMemcpy3DParms(dst_ptr, invalid_pos, src_ptr, src_pos, extent, kind);
      HIP_CHECK_ERROR(hipDrvGraphMemcpyNodeSetParams(node, &invalid_params), hipErrorInvalidValue);
    }

    SECTION("srcZ out of bounds") {
      hipPos invalid_pos = src_pos;
      invalid_pos.z = 1;
      auto invalid_params =
          GetDrvMemcpy3DParms(dst_ptr, dst_pos, src_ptr, invalid_pos, extent, kind);
      HIP_CHECK_ERROR(hipDrvGraphMemcpyNodeSetParams(node, &invalid_params), hipErrorInvalidValue);
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
