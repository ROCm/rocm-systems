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
#include <contract_cleanup.hh>

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

// @asserts: hipGraphAddMemcpyNode - a 3D memcpy node delivers the full extent to the device buffer when the graph is launched
HIP_TEST_CASE(Contract_GraphMemcpy3DNode_AddNode_LaunchesCopyThroughGraph) {
  hipPitchedPtr device{};
  if (!TryMalloc3D(&device)) {
    HIP_SKIP_TEST("hipMalloc3D is not supported by this device/runtime path.");
  }

  hip::contract::ContractCleanup cleanup;
  cleanup.Add([p0 = device.ptr] { (void)hipFree(p0); });
  const auto src = MakePattern(0x40);
  std::array<uint8_t, kElems> dst{};
  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });

  // A 3D memcpy node added to the graph must, when the graph is launched,
  // deliver the full extent to the device buffer exactly like a direct
  // hipMemcpy3D would.
  hipMemcpy3DParms params = MakeH2DParams(src.data(), device);
  HIP_CHECK(hipGraphAddMemcpyNode(&node, graph, nullptr, 0, &params));

  LaunchGraph(graph);
  ReadBack(&dst, device);
  REQUIRE(dst == src);
}

// @asserts: hipGraphMemcpyNodeGetParams - getter reports the copy kind, extent, and endpoints the 3D memcpy node was created with
HIP_TEST_CASE(Contract_GraphMemcpy3DNode_GetParams_ReflectsAddedNode) {
  hipPitchedPtr device{};
  if (!TryMalloc3D(&device)) {
    HIP_SKIP_TEST("hipMalloc3D is not supported by this device/runtime path.");
  }

  hip::contract::ContractCleanup cleanup;
  cleanup.Add([p0 = device.ptr] { (void)hipFree(p0); });
  const auto src = MakePattern(0x50);
  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });
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
}

// @asserts: hipGraphMemcpyNodeSetParams - re-pointing the 3D memcpy node's source before instantiation makes the launched graph copy the new buffer
HIP_TEST_CASE(Contract_GraphMemcpy3DNode_SetParams_RetargetsSourceBeforeInstantiate) {
  hipPitchedPtr device{};
  if (!TryMalloc3D(&device)) {
    HIP_SKIP_TEST("hipMalloc3D is not supported by this device/runtime path.");
  }

  hip::contract::ContractCleanup cleanup;
  cleanup.Add([p0 = device.ptr] { (void)hipFree(p0); });
  const auto first = MakePattern(0x11);
  const auto second = MakePattern(0x88);
  std::array<uint8_t, kElems> dst{};
  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });

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
}

// @asserts: hipGraphExecMemcpyNodeSetParams - exec-time retarget of a 3D memcpy node to a different allocation is accepted on AMD but rejected with hipErrorInvalidValue on NVIDIA
HIP_TEST_CASE(Contract_GraphMemcpy3DNode_ExecSetParams_RetargetsSourceAfterInstantiate) {
  hipPitchedPtr device{};
  if (!TryMalloc3D(&device)) {
    HIP_SKIP_TEST("hipMalloc3D is not supported by this device/runtime path.");
  }

  hip::contract::ContractCleanup cleanup;
  cleanup.Add([p0 = device.ptr] { (void)hipFree(p0); });
  const auto first = MakePattern(0x22);
  const auto second = MakePattern(0x99);
  std::array<uint8_t, kElems> dst{};
  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });
  hipMemcpy3DParms initial = MakeH2DParams(first.data(), device);
  HIP_CHECK(hipGraphAddMemcpyNode(&node, graph, nullptr, 0, &initial));

  hipGraphExec_t exec = nullptr;
  hipStream_t stream = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  cleanup.Add([exec] { (void)hipGraphExecDestroy(exec); });

  // Re-point the instantiated node at the second buffer through the executable
  // setter.
  hipMemcpy3DParms updated = MakeH2DParams(second.data(), device);
  const hipError_t update_status = hipGraphExecMemcpyNodeSetParams(exec, node, &updated);

  // BACKEND-DIFF: the executable memcpy-node setter diverges on whether an
  // instantiated node's memory operands may be re-pointed to a different
  // allocation. AMD accepts it; NVIDIA rejects it with hipErrorInvalidValue (see
  // the #else branch). Behavioral delta, reconcilable if CUDA relaxes it.
#if HT_AMD
  // On AMD the executable setter accepts re-pointing the copy at a different host
  // source allocation after instantiation, and the next launch copies the second
  // buffer without a re-instantiate.
  HIP_CHECK(update_status);

  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipGraphLaunch(exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  ReadBack(&dst, device);
  REQUIRE(dst == second);
#else
  // On NVIDIA hipGraphExecMemcpyNodeSetParams maps to
  // cudaGraphExecMemcpyNodeSetParams, which does NOT allow changing the memory
  // operands of an instantiated node to a different allocation: re-pointing the
  // source to a separate host buffer is rejected with hipErrorInvalidValue
  // (probe-confirmed: an identical-params update and the non-exec
  // hipGraphMemcpyNodeSetParams both succeed; only the exec-time retarget to a
  // different allocation is refused). Assert the documented rejection, then prove
  // the executable is still usable by launching the originally instantiated copy
  // (the first buffer) so the node/exec remain valid. If CUDA relaxes this
  // restriction, this branch is where the expectation changes to match AMD.
  REQUIRE(update_status == hipErrorInvalidValue);
  (void)hipGetLastError();

  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipGraphLaunch(exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  ReadBack(&dst, device);
  REQUIRE(dst == first);
#endif
}
