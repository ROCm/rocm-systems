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

// The driver-style graph node entry points (hipDrvGraphAddMemcpyNode,
// hipDrvGraphAddMemsetNode, and their get/set/exec-set params) take a driver
// HIP_MEMCPY3D / hipMemsetParams plus a hipCtx_t and are AMD-side in this tree,
// so the whole domain is gated like the other driver-context contracts.
#if HT_AMD

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

hipExtent ByteExtent() { return make_hipExtent(kWidth, kHeight, kDepth); }

hipPitchedPtr HostPitchedPtr(void* ptr) {
  return make_hipPitchedPtr(ptr, kWidth, kWidth, kHeight);
}

// The current context is the one the runtime established for the default
// device. The driver graph node APIs require an explicit hipCtx_t; a null
// context would be an invalid argument rather than an implicit default.
hipCtx_t CurrentContext() {
  hipCtx_t ctx = nullptr;
  HIP_CHECK(hipCtxGetCurrent(&ctx));
  REQUIRE(ctx != nullptr);
  return ctx;
}

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

HIP_MEMCPY3D HostToDeviceCopy(hipPitchedPtr dst, const void* src) {
  HIP_MEMCPY3D copy{};
  copy.srcMemoryType = hipMemoryTypeHost;
  copy.srcHost = src;
  copy.srcPitch = kWidth;
  copy.srcHeight = kHeight;
  copy.dstMemoryType = hipMemoryTypeDevice;
  copy.dstDevice = reinterpret_cast<hipDeviceptr_t>(dst.ptr);
  copy.dstPitch = dst.pitch;
  copy.dstHeight = dst.ysize;
  copy.WidthInBytes = kWidth;
  copy.Height = kHeight;
  copy.Depth = kDepth;
  return copy;
}

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

HIP_TEST_CASE(Contract_DriverGraphNode_AddMemcpyNode_LaunchesCopyThroughGraph) {
  hip::contract::ContractCleanup cleanup;
  hipPitchedPtr device{};
  if (!TryMalloc3D(&device)) {
    HIP_SKIP_TEST("hipMalloc3D is not supported by this device/runtime path.");
  }
  cleanup.Add([&] { (void)hipFree(device.ptr); });

  const auto src = MakePattern(0x40);
  std::array<uint8_t, kElems> dst{};
  const hipCtx_t ctx = CurrentContext();
  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([&] { (void)hipGraphDestroy(graph); });

  // A driver-style 3D memcpy node bound to the current context must, when the
  // graph is launched, deliver the full extent to the device buffer exactly
  // like a direct hipDrvMemcpy3D would.
  HIP_MEMCPY3D copy = HostToDeviceCopy(device, src.data());
  HIP_CHECK(hipDrvGraphAddMemcpyNode(&node, graph, nullptr, 0, &copy, ctx));

  LaunchGraph(graph);
  ReadBack(&dst, device);
  REQUIRE(dst == src);
}

HIP_TEST_CASE(Contract_DriverGraphNode_MemcpyNodeGetParams_RoundTripsExtent) {
  hip::contract::ContractCleanup cleanup;
  hipPitchedPtr device{};
  if (!TryMalloc3D(&device)) {
    HIP_SKIP_TEST("hipMalloc3D is not supported by this device/runtime path.");
  }
  cleanup.Add([&] { (void)hipFree(device.ptr); });

  const auto src = MakePattern(0x50);
  const hipCtx_t ctx = CurrentContext();
  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([&] { (void)hipGraphDestroy(graph); });
  HIP_MEMCPY3D copy = HostToDeviceCopy(device, src.data());
  HIP_CHECK(hipDrvGraphAddMemcpyNode(&node, graph, nullptr, 0, &copy, ctx));

  // The driver-node getter reports the extent and endpoints the node was
  // created with.
  HIP_MEMCPY3D retrieved{};
  HIP_CHECK(hipDrvGraphMemcpyNodeGetParams(node, &retrieved));
  REQUIRE(retrieved.WidthInBytes == kWidth);
  REQUIRE(retrieved.Height == kHeight);
  REQUIRE(retrieved.Depth == kDepth);
  REQUIRE(retrieved.srcHost == src.data());
  REQUIRE(retrieved.dstDevice == reinterpret_cast<hipDeviceptr_t>(device.ptr));

  // The graph-node setter accepts the same parameters back.
  HIP_CHECK(hipDrvGraphMemcpyNodeSetParams(node, &copy));
}

HIP_TEST_CASE(Contract_DriverGraphNode_ExecMemcpyNodeSetParams_RetargetsSourceAfterInstantiate) {
  hip::contract::ContractCleanup cleanup;
  hipPitchedPtr device{};
  if (!TryMalloc3D(&device)) {
    HIP_SKIP_TEST("hipMalloc3D is not supported by this device/runtime path.");
  }
  cleanup.Add([&] { (void)hipFree(device.ptr); });

  const auto first = MakePattern(0x22);
  const auto second = MakePattern(0x99);
  std::array<uint8_t, kElems> dst{};
  const hipCtx_t ctx = CurrentContext();
  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([&] { (void)hipGraphDestroy(graph); });
  HIP_MEMCPY3D initial = HostToDeviceCopy(device, first.data());
  HIP_CHECK(hipDrvGraphAddMemcpyNode(&node, graph, nullptr, 0, &initial, ctx));

  hipGraphExec_t exec = nullptr;
  hipStream_t stream = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  cleanup.Add([&] { (void)hipGraphExecDestroy(exec); });

  // Re-point the instantiated driver node at the second host buffer through the
  // executable setter. The next launch must copy the second buffer without a
  // re-instantiation.
  HIP_MEMCPY3D updated = HostToDeviceCopy(device, second.data());
  HIP_CHECK(hipDrvGraphExecMemcpyNodeSetParams(exec, node, &updated, ctx));

  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([&] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipGraphLaunch(exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  ReadBack(&dst, device);
  REQUIRE(dst == second);
}

HIP_TEST_CASE(Contract_DriverGraphNode_AddMemsetNode_LaunchesExpectedValue) {
  hip::contract::ContractCleanup cleanup;
  hipPitchedPtr device{};
  if (!TryMalloc3D(&device)) {
    HIP_SKIP_TEST("hipMalloc3D is not supported by this device/runtime path.");
  }
  cleanup.Add([&] { (void)hipFree(device.ptr); });

  std::array<uint8_t, kElems> dst{};
  const hipCtx_t ctx = CurrentContext();
  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([&] { (void)hipGraphDestroy(graph); });

  // A driver-style memset node writes one row of the pitched allocation. The
  // memset covers width*height elements of a single-byte element, matched to the
  // pitched buffer's pitch so the first row is fully written.
  hipMemsetParams params{};
  params.dst = device.ptr;
  params.value = 0x5A;
  params.elementSize = sizeof(uint8_t);
  params.width = kWidth;
  params.height = 1;
  params.pitch = device.pitch;
  HIP_CHECK(hipDrvGraphAddMemsetNode(&node, graph, nullptr, 0, &params, ctx));

  LaunchGraph(graph);

  // Read back the first row and confirm the requested byte landed.
  ReadBack(&dst, device);
  REQUIRE(dst[0] == 0x5A);
  REQUIRE(dst[kWidth - 1] == 0x5A);
}

HIP_TEST_CASE(Contract_DriverGraphNode_ExecMemsetNodeSetParams_UpdatesValueAfterInstantiate) {
  hip::contract::ContractCleanup cleanup;
  hipPitchedPtr device{};
  if (!TryMalloc3D(&device)) {
    HIP_SKIP_TEST("hipMalloc3D is not supported by this device/runtime path.");
  }
  cleanup.Add([&] { (void)hipFree(device.ptr); });

  std::array<uint8_t, kElems> dst{};
  const hipCtx_t ctx = CurrentContext();
  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([&] { (void)hipGraphDestroy(graph); });

  hipMemsetParams params{};
  params.dst = device.ptr;
  params.value = 0x11;
  params.elementSize = sizeof(uint8_t);
  params.width = kWidth;
  params.height = 1;
  params.pitch = device.pitch;
  HIP_CHECK(hipDrvGraphAddMemsetNode(&node, graph, nullptr, 0, &params, ctx));

  hipGraphExec_t exec = nullptr;
  hipStream_t stream = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  cleanup.Add([&] { (void)hipGraphExecDestroy(exec); });

  // Re-parameterize the instantiated driver memset node with a new byte value
  // through the executable setter. The next launch must write the updated value.
  hipMemsetParams updated = params;
  updated.value = 0x22;
  HIP_CHECK(hipDrvGraphExecMemsetNodeSetParams(exec, node, &updated, ctx));

  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([&] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipGraphLaunch(exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  ReadBack(&dst, device);
  REQUIRE(dst[0] == 0x22);
}

#endif  // HT_AMD
