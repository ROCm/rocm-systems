/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
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
HIP_TEST_CASE(Unit_hipGraphMemcpyNodeSetParams_Positive_Basic) {
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
 *        -# pNodeParams is nullptr
 *        -# node is uninitialized
 *        -# dst is nullptr
 *        -# src is nullptr
 *        -# dst pitch is less than width
 *        -# src pitch is less than width
 *        -# dst pitch exceeds max pitch
 *        -# src pitch exceeds max pitch
 *        -# extent width + dst position exceeds dst pitch
 *        -# extent width + src position exceeds src pitch
 *        -# dst position y is out of bounds
 *        -# src position y is out of bounds
 *        -# dst position z is out of bounds
 *        -# src position z is out of bounds
 *        -# kind is an invalid enum value
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphAddMemcpyNode.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipGraphMemcpyNodeSetParams_Negative_Parameters) {
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

    SECTION("pNodeParams == nullptr") {
      HIP_CHECK_ERROR(hipGraphMemcpyNodeSetParams(node, nullptr), hipErrorInvalidValue);
    }

    SECTION("Uninitialized node") {
      hipGraphNode_t node_uninit{};
      params = GetMemcpy3DParms(dst_ptr, dst_pos, src_ptr, src_pos, extent, kind);
      HIP_CHECK_ERROR(hipGraphMemcpyNodeSetParams(node_uninit, &params), hipErrorInvalidValue);
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
 * Test Description
 * ------------------------
 *  - Verify CUDA single-memcpy-node-type semantics across HIP's node variants: a linear 3D
 *    description updates a 1D node (and round-trips through the 3D getter), a 1D update applies
 *    to a 3D node, and a non-linear 3D description is rejected on a 1D node.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphMemcpyNodeSetParams.cc
 */
HIP_TEST_CASE(Unit_hipGraphMemcpyNodeSetParams_CrossKind) {
  constexpr size_t kCount = 32;
  constexpr size_t kBytes = kCount * sizeof(int);
  int* src = nullptr;
  int* dst = nullptr;
  int* src2 = nullptr;
  int* dst2 = nullptr;
  HIP_CHECK(hipMalloc(&src, kBytes));
  HIP_CHECK(hipMalloc(&dst, kBytes));
  HIP_CHECK(hipMalloc(&src2, kBytes));
  HIP_CHECK(hipMalloc(&dst2, kBytes));
  std::vector<int> host(kCount);
  for (size_t i = 0; i < kCount; ++i) host[i] = static_cast<int>(i) * 3 + 1;
  HIP_CHECK(hipMemcpy(src2, host.data(), kBytes, hipMemcpyHostToDevice));

  auto linear3D = [&](void* d, const void* s) {
    hipMemcpy3DParms p{};
    p.srcPtr = make_hipPitchedPtr(const_cast<void*>(s), kBytes, kCount, 1);
    p.dstPtr = make_hipPitchedPtr(d, kBytes, kCount, 1);
    p.srcPos = make_hipPos(0, 0, 0);
    p.dstPos = make_hipPos(0, 0, 0);
    p.extent = make_hipExtent(kBytes, 1, 1);
    p.kind = hipMemcpyDeviceToDevice;
    return p;
  };
  auto launch_and_check = [&](hipGraph_t graph, int* expected_dst) {
    hipGraphExec_t exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(exec, hipStreamPerThread));
    HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
    std::vector<int> actual(kCount);
    HIP_CHECK(hipMemcpy(actual.data(), expected_dst, kBytes, hipMemcpyDeviceToHost));
    REQUIRE(actual == host);
    HIP_CHECK(hipGraphExecDestroy(exec));
  };

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  SECTION("linear 3D params update a 1D node and round-trip") {
    hipGraphNode_t node1D = nullptr;
    HIP_CHECK(hipGraphAddMemcpyNode1D(&node1D, graph, nullptr, 0, dst, src, kBytes,
                                      hipMemcpyDeviceToDevice));
    hipMemcpy3DParms params = linear3D(dst2, src2);
    HIP_CHECK(hipGraphMemcpyNodeSetParams(node1D, &params));
    hipMemcpy3DParms returned{};
    HIP_CHECK(hipGraphMemcpyNodeGetParams(node1D, &returned));
    REQUIRE(returned.srcPtr.ptr == src2);
    REQUIRE(returned.dstPtr.ptr == dst2);
    REQUIRE(returned.extent.width == kBytes);
    REQUIRE(returned.kind == hipMemcpyDeviceToDevice);
    launch_and_check(graph, dst2);
  }
  SECTION("non-linear 3D params are rejected on a 1D node") {
    hipGraphNode_t node1D = nullptr;
    HIP_CHECK(hipGraphAddMemcpyNode1D(&node1D, graph, nullptr, 0, dst, src, kBytes,
                                      hipMemcpyDeviceToDevice));
    hipMemcpy3DParms params = linear3D(dst2, src2);
    params.extent = make_hipExtent(kBytes / 2, 2, 1);
    HIP_CHECK_ERROR(hipGraphMemcpyNodeSetParams(node1D, &params), hipErrorInvalidValue);
  }
  SECTION("1D setter applies to a 3D node") {
    hipGraphNode_t node3D = nullptr;
    hipMemcpy3DParms params = linear3D(dst, src);
    HIP_CHECK(hipGraphAddMemcpyNode(&node3D, graph, nullptr, 0, &params));
    HIP_CHECK(hipGraphMemcpyNodeSetParams1D(node3D, dst2, src2, kBytes, hipMemcpyDeviceToDevice));
    hipMemcpy3DParms returned{};
    HIP_CHECK(hipGraphMemcpyNodeGetParams(node3D, &returned));
    REQUIRE(returned.srcPtr.ptr == src2);
    REQUIRE(returned.dstPtr.ptr == dst2);
    launch_and_check(graph, dst2);
  }

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(dst2));
  HIP_CHECK(hipFree(src2));
  HIP_CHECK(hipFree(dst));
  HIP_CHECK(hipFree(src));
}

/**
 * End doxygen group GraphTest.
 * @}
 */
