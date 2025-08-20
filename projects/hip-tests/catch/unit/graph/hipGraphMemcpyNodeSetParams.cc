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
#include <memcpy3d_tests_common.hh>

#include "graph_tests_common.hh"

/**
 * @addtogroup hipGraphMemcpyNodeSetParams hipGraphMemcpyNodeSetParams
 * @{
 * @ingroup GraphTest
 * `hipGraphMemcpyNodeSetParams (hipGraphNode_t node, const hipMemcpy3DParms *pNodeParams)` - Sets a
 * memcpy node's parameters
 */

/**
 * Test Description
 * ------------------------
 *    - Verify that node parameters get updated correctly by creating a node with valid but
 * incorrect parameters, and the setting them to the correct values after which the graph is
 * executed and the results of the memcpy verified.
 * The test is run for all possible memcpy directions, with both the corresponding memcpy
 * kind and hipMemcpyDefault, as well as half page and full page allocation sizes.
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphMemcpyNodeSetParams.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphMemcpyNodeSetParams_Positive_Basic") {
  CHECK_IMAGE_SUPPORT

  constexpr bool async = false;

  SECTION("Device to host") {
    Memcpy3DDeviceToHostShell<async>(Memcpy3DWrapper<async, true, true>);
  }

  SECTION("Device to host with default kind") {
    Memcpy3DDeviceToHostShell<async>(Memcpy3DWrapper<async, true, true>);
  }

  SECTION("Host to device") {
    Memcpy3DHostToDeviceShell<async>(Memcpy3DWrapper<async, true, true>);
  }

  SECTION("Host to device with default kind") {
    Memcpy3DHostToDeviceShell<async>(Memcpy3DWrapper<async, true, true>);
  }

  SECTION("Host to host") { Memcpy3DHostToHostShell<async>(Memcpy3DWrapper<async, true, true>); }

  SECTION("Host to host with default kind") {
    Memcpy3DHostToHostShell<async>(Memcpy3DWrapper<async, true, true>);
  }

  SECTION("Device to device") {
    SECTION("Peer access enabled") {
      Memcpy3DDeviceToDeviceShell<async, true>(Memcpy3DWrapper<async, true, true>);
    }
    SECTION("Peer access disabled") {
      Memcpy3DDeviceToDeviceShell<async, false>(Memcpy3DWrapper<async, true, true>);
    }
  }

  SECTION("Device to device with default kind") {
    SECTION("Peer access enabled") {
      Memcpy3DDeviceToDeviceShell<async, true>(Memcpy3DWrapper<async, true, true>);
    }
    SECTION("Peer access disabled") {
      Memcpy3DDeviceToDeviceShell<async, false>(Memcpy3DWrapper<async, true, true>);
    }
  }

  SECTION("Array from/to Host") {
    Memcpy3DArrayHostShell<async>(Memcpy3DWrapper<async, true, true>);
  }

#if HT_NVIDIA  // Disabled on AMD due to defect - EXSWHTEC-220
  SECTION("Array from/to Device") {
    Memcpy3DArrayDeviceShell<async>(Memcpy3DWrapper<async, true, true>);
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
 *    - unit/graph/hipGraphAddMemcpyNode.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphMemcpyNodeSetParams_Negative_Parameters") {
  CHECK_IMAGE_SUPPORT

  using namespace std::placeholders;

  constexpr hipExtent extent{128 * sizeof(int), 128, 8};

  constexpr auto NegativeTests = [](hipPitchedPtr dst_ptr, hipPos dst_pos, hipPitchedPtr src_ptr,
                                    hipPos src_pos, hipExtent extent, hipMemcpyKind kind) {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipGraphCreate(&graph, 0));

    hipGraphNode_t node = nullptr;
    auto params = GetMemcpy3DParms(dst_ptr, dst_pos, src_ptr, src_pos, extent, kind);
    HIP_CHECK(hipGraphAddMemcpyNode(&node, graph, nullptr, 0, &params));

    SECTION("node == nullptr") {
      params = GetMemcpy3DParms(dst_ptr, dst_pos, src_ptr, src_pos, extent, kind);
      HIP_CHECK_ERROR(hipGraphMemcpyNodeSetParams(nullptr, &params), hipErrorInvalidValue);
    }

    SECTION("dst_ptr.ptr == nullptr") {
      hipPitchedPtr invalid_ptr = dst_ptr;
      invalid_ptr.ptr = nullptr;
      params = GetMemcpy3DParms(invalid_ptr, dst_pos, src_ptr, src_pos, extent, kind);
      HIP_CHECK_ERROR(hipGraphMemcpyNodeSetParams(node, &params), hipErrorInvalidValue);
    }

    SECTION("src_ptr.ptr == nullptr") {
      hipPitchedPtr invalid_ptr = src_ptr;
      invalid_ptr.ptr = nullptr;
      params = GetMemcpy3DParms(dst_ptr, dst_pos, invalid_ptr, src_pos, extent, kind);
      HIP_CHECK_ERROR(hipGraphMemcpyNodeSetParams(node, &params), hipErrorInvalidValue);
    }

    SECTION("dst_ptr.pitch < width") {
      hipPitchedPtr invalid_ptr = dst_ptr;
      invalid_ptr.pitch = extent.width - 1;
      params = GetMemcpy3DParms(invalid_ptr, dst_pos, src_ptr, src_pos, extent, kind);
      HIP_CHECK_ERROR(hipGraphMemcpyNodeSetParams(node, &params), hipErrorInvalidPitchValue);
    }

    SECTION("src_ptr.pitch < width") {
      hipPitchedPtr invalid_ptr = src_ptr;
      invalid_ptr.pitch = extent.width - 1;
      params = GetMemcpy3DParms(dst_ptr, dst_pos, invalid_ptr, src_pos, extent, kind);
      HIP_CHECK_ERROR(hipGraphMemcpyNodeSetParams(node, &params), hipErrorInvalidPitchValue);
    }

    SECTION("dst_ptr.pitch > max pitch") {
      int attr = 0;
      HIP_CHECK(hipDeviceGetAttribute(&attr, hipDeviceAttributeMaxPitch, 0));
      hipPitchedPtr invalid_ptr = dst_ptr;
      invalid_ptr.pitch = attr;
      params = GetMemcpy3DParms(invalid_ptr, dst_pos, src_ptr, src_pos, extent, kind);
      HIP_CHECK_ERROR(hipGraphMemcpyNodeSetParams(node, &params), hipErrorInvalidValue);
    }

    SECTION("src_ptr.pitch > max pitch") {
      int attr = 0;
      HIP_CHECK(hipDeviceGetAttribute(&attr, hipDeviceAttributeMaxPitch, 0));
      hipPitchedPtr invalid_ptr = src_ptr;
      invalid_ptr.pitch = attr;
      params = GetMemcpy3DParms(dst_ptr, dst_pos, invalid_ptr, src_pos, extent, kind);
      HIP_CHECK_ERROR(hipGraphMemcpyNodeSetParams(node, &params), hipErrorInvalidValue);
    }

    SECTION("extent.width + dst_pos.x > dst_ptr.pitch") {
      hipPos invalid_pos = dst_pos;
      invalid_pos.x = dst_ptr.pitch - extent.width + 1;
      params = GetMemcpy3DParms(dst_ptr, invalid_pos, src_ptr, src_pos, extent, kind);
      HIP_CHECK_ERROR(hipGraphMemcpyNodeSetParams(node, &params), hipErrorInvalidValue);
    }

    SECTION("extent.width + src_pos.x > src_ptr.pitch") {
      hipPos invalid_pos = src_pos;
      invalid_pos.x = src_ptr.pitch - extent.width + 1;
      params = GetMemcpy3DParms(dst_ptr, dst_pos, src_ptr, invalid_pos, extent, kind);
      HIP_CHECK_ERROR(hipGraphMemcpyNodeSetParams(node, &params), hipErrorInvalidValue);
    }

    SECTION("dst_pos.y out of bounds") {
      hipPos invalid_pos = dst_pos;
      invalid_pos.y = 1;
      params = GetMemcpy3DParms(dst_ptr, invalid_pos, src_ptr, src_pos, extent, kind);
      HIP_CHECK_ERROR(hipGraphMemcpyNodeSetParams(node, &params), hipErrorInvalidValue);
    }

    SECTION("src_pos.y out of bounds") {
      hipPos invalid_pos = src_pos;
      invalid_pos.y = 1;
      params = GetMemcpy3DParms(dst_ptr, dst_pos, src_ptr, invalid_pos, extent, kind);
      HIP_CHECK_ERROR(hipGraphMemcpyNodeSetParams(node, &params), hipErrorInvalidValue);
    }

    SECTION("dst_pos.z out of bounds") {
      hipPos invalid_pos = dst_pos;
      invalid_pos.z = 1;
      params = GetMemcpy3DParms(dst_ptr, invalid_pos, src_ptr, src_pos, extent, kind);
      HIP_CHECK_ERROR(hipGraphMemcpyNodeSetParams(node, &params), hipErrorInvalidValue);
    }

    SECTION("src_pos.z out of bounds") {
      hipPos invalid_pos = src_pos;
      invalid_pos.z = 1;
      params = GetMemcpy3DParms(dst_ptr, dst_pos, src_ptr, invalid_pos, extent, kind);
      HIP_CHECK_ERROR(hipGraphMemcpyNodeSetParams(node, &params), hipErrorInvalidValue);
    }

    SECTION("Invalid MemcpyKind") {
      params = GetMemcpy3DParms(dst_ptr, dst_pos, src_ptr, src_pos, extent,
                                static_cast<hipMemcpyKind>(-1));
      HIP_CHECK_ERROR(hipGraphMemcpyNodeSetParams(node, &params), hipErrorInvalidMemcpyDirection);
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
                  make_hipPos(0, 0, 0), extent, hipMemcpyHostToDevice);
  }

  SECTION("Device to Host") {
    LinearAllocGuard3D<int> device_alloc(extent);
    LinearAllocGuard<int> host_alloc(
        LinearAllocs::hipHostMalloc,
        device_alloc.pitch() * device_alloc.height() * device_alloc.depth());
    NegativeTests(make_hipPitchedPtr(host_alloc.ptr(), device_alloc.pitch(), device_alloc.width(),
                                     device_alloc.height()),
                  make_hipPos(0, 0, 0), device_alloc.pitched_ptr(), make_hipPos(0, 0, 0), extent,
                  hipMemcpyDeviceToHost);
  }

  SECTION("Host to Host") {
    LinearAllocGuard<int> src_alloc(LinearAllocs::hipHostMalloc,
                                    extent.width * extent.height * extent.depth);
    LinearAllocGuard<int> dst_alloc(LinearAllocs::hipHostMalloc,
                                    extent.width * extent.height * extent.depth);
    NegativeTests(make_hipPitchedPtr(dst_alloc.ptr(), extent.width, extent.width, extent.height),
                  make_hipPos(0, 0, 0),
                  make_hipPitchedPtr(src_alloc.ptr(), extent.width, extent.width, extent.height),
                  make_hipPos(0, 0, 0), extent, hipMemcpyHostToHost);
  }

  SECTION("Device to Device") {
    LinearAllocGuard3D<int> src_alloc(extent);
    LinearAllocGuard3D<int> dst_alloc(extent);
    NegativeTests(dst_alloc.pitched_ptr(), make_hipPos(0, 0, 0), src_alloc.pitched_ptr(),
                  make_hipPos(0, 0, 0), extent, hipMemcpyDeviceToDevice);
  }
}

/**
 * End doxygen group GraphTest.
 * @}
 */
