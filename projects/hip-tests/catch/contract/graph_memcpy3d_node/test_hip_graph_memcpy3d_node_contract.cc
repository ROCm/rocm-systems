/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <array>
#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

namespace {
constexpr size_t kWidth = 7;
constexpr size_t kHeight = 5;
constexpr size_t kDepth = 3;
constexpr size_t kElems = kWidth * kHeight * kDepth;

std::array<uint8_t, kElems> MakePattern(uint8_t seed) {
  std::array<uint8_t, kElems> pattern{};
  for (size_t i = 0; i < pattern.size(); ++i) {
    pattern[i] = static_cast<uint8_t>(seed + i);
  }
  return pattern;
}

hipPitchedPtr HostPitchedPtr(void* ptr) {
  return make_hipPitchedPtr(ptr, kWidth, kWidth, kHeight);
}

hipExtent ByteExtent() { return make_hipExtent(kWidth, kHeight, kDepth); }

bool TryMalloc3D(hipPitchedPtr* device_ptr) {
  const hipError_t status = hipMalloc3D(device_ptr, ByteExtent());
  if (status == hipSuccess) {
    return true;
  }
  if (status == hipErrorOutOfMemory || status == hipErrorNotSupported) {
    return false;
  }
  HIP_CHECK(status);
  return false;
}

// Builds a host-to-device 3D copy parameter block from host_src into device.
hipMemcpy3DParms MakeH2DParams(const void* host_src, hipPitchedPtr device) {
  hipMemcpy3DParms params{};
  params.srcPtr = HostPitchedPtr(const_cast<void*>(host_src));
  params.dstPtr = device;
  params.extent = ByteExtent();
  params.kind = hipMemcpyHostToDevice;
  return params;
}

// Copies the full extent back from device to host through an ordinary 3D copy
// so a graph node's effect can be observed.
void ReadBack(std::array<uint8_t, kElems>* host, hipPitchedPtr device) {
  hipMemcpy3DParms d2h{};
  d2h.srcPtr = device;
  d2h.dstPtr = HostPitchedPtr(host->data());
  d2h.extent = ByteExtent();
  d2h.kind = hipMemcpyDeviceToHost;
  HIP_CHECK(hipMemcpy3D(&d2h));
}

void LaunchGraph(hipGraph_t graph) {
  hipGraphExec_t exec = nullptr;
  hipStream_t stream = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphLaunch(exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(exec));
}
}  // namespace

HIP_TEST_CASE(Contract_GraphMemcpy3DNode_AddNode_LaunchesCopyThroughGraph) {
  hipPitchedPtr device{};
  if (!TryMalloc3D(&device)) {
    HIP_SKIP_TEST("hipMalloc3D is not supported by this device/runtime path.");
  }

  const auto src = MakePattern(0x40);
  std::array<uint8_t, kElems> dst{};
  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;

  HIP_CHECK(hipGraphCreate(&graph, 0));

  // A 3D memcpy node added to the graph must, when the graph is launched,
  // deliver the full extent to the device buffer exactly like a direct
  // hipMemcpy3D would.
  hipMemcpy3DParms params = MakeH2DParams(src.data(), device);
  HIP_CHECK(hipGraphAddMemcpyNode(&node, graph, nullptr, 0, &params));

  LaunchGraph(graph);
  ReadBack(&dst, device);
  REQUIRE(dst == src);

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(device.ptr));
}

HIP_TEST_CASE(Contract_GraphMemcpy3DNode_GetParams_ReflectsAddedNode) {
  hipPitchedPtr device{};
  if (!TryMalloc3D(&device)) {
    HIP_SKIP_TEST("hipMalloc3D is not supported by this device/runtime path.");
  }

  const auto src = MakePattern(0x50);
  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipMemcpy3DParms params = MakeH2DParams(src.data(), device);
  HIP_CHECK(hipGraphAddMemcpyNode(&node, graph, nullptr, 0, &params));

  // The node getter reports the copy kind, extent, and endpoints the node was
  // created with.
  hipMemcpy3DParms retrieved{};
  HIP_CHECK(hipGraphMemcpyNodeGetParams(node, &retrieved));
  REQUIRE(retrieved.kind == hipMemcpyHostToDevice);
  REQUIRE(retrieved.extent.width == kWidth);
  REQUIRE(retrieved.extent.height == kHeight);
  REQUIRE(retrieved.extent.depth == kDepth);
  REQUIRE(retrieved.srcPtr.ptr == src.data());
  REQUIRE(retrieved.dstPtr.ptr == device.ptr);

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(device.ptr));
}

HIP_TEST_CASE(Contract_GraphMemcpy3DNode_SetParams_RetargetsSourceBeforeInstantiate) {
  hipPitchedPtr device{};
  if (!TryMalloc3D(&device)) {
    HIP_SKIP_TEST("hipMalloc3D is not supported by this device/runtime path.");
  }

  const auto first = MakePattern(0x11);
  const auto second = MakePattern(0x88);
  std::array<uint8_t, kElems> dst{};
  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;

  HIP_CHECK(hipGraphCreate(&graph, 0));

  // Create the node reading from the first host buffer, then re-point it at the
  // second buffer before instantiation. The launched graph must copy the second
  // buffer's contents.
  hipMemcpy3DParms initial = MakeH2DParams(first.data(), device);
  HIP_CHECK(hipGraphAddMemcpyNode(&node, graph, nullptr, 0, &initial));

  hipMemcpy3DParms updated = MakeH2DParams(second.data(), device);
  HIP_CHECK(hipGraphMemcpyNodeSetParams(node, &updated));

  LaunchGraph(graph);
  ReadBack(&dst, device);
  REQUIRE(dst == second);

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(device.ptr));
}

HIP_TEST_CASE(Contract_GraphMemcpy3DNode_ExecSetParams_RetargetsSourceAfterInstantiate) {
  hipPitchedPtr device{};
  if (!TryMalloc3D(&device)) {
    HIP_SKIP_TEST("hipMalloc3D is not supported by this device/runtime path.");
  }

  const auto first = MakePattern(0x22);
  const auto second = MakePattern(0x99);
  std::array<uint8_t, kElems> dst{};
  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipMemcpy3DParms initial = MakeH2DParams(first.data(), device);
  HIP_CHECK(hipGraphAddMemcpyNode(&node, graph, nullptr, 0, &initial));

  hipGraphExec_t exec = nullptr;
  hipStream_t stream = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  // Re-point the instantiated node at the second buffer through the executable
  // setter. The next launch must copy the second buffer without re-instantiate.
  hipMemcpy3DParms updated = MakeH2DParams(second.data(), device);
  HIP_CHECK(hipGraphExecMemcpyNodeSetParams(exec, node, &updated));

  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphLaunch(exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  ReadBack(&dst, device);
  REQUIRE(dst == second);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(device.ptr));
}
