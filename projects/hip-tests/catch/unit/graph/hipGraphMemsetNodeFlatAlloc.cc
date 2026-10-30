/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup GraphTest GraphTest
 * @{
 * Coverage for 2D/3D memset graph nodes whose destination is a "flat"
 * allocation (plain hipMalloc, hipHostMalloc, hipMallocManaged) rather than
 * a pitched/3D allocation. These cases regressed because internal validation
 * consulted per-allocation 2D extent metadata that is only populated by
 * hipMallocPitch / hipMalloc3D.
 *
 * The tests below also pin down hipGraphExecUpdate's contract for memset
 * nodes (CUDA: "For 2d memsets, only address and assigned value may be
 * updated"), and the equivalent contract reached through
 * hipGraphExecNodeSetParams.
 */

#include <hip_test_common.hh>
#include <resource_guards.hh>
#include <utils.hh>

#include <algorithm>
#include <limits>
#include <vector>

namespace {

enum class VmmFillMode { CaptureBytes, CaptureTyped, Graph };
enum class InvalidGeometry { WidthOverflow, AreaOverflow, Misaligned, OutsideSpan };

// Allocates `pitch * height` bytes via plain hipMalloc, with `pitch` chosen
// independently of any device alignment requirement -- so the 2D extent
// metadata that hipMallocPitch would have recorded is left at zero.
struct FlatPitched {
  void* ptr = nullptr;
  size_t pitch = 0;
  size_t width_bytes = 0;
  size_t height = 0;

  FlatPitched(size_t width_bytes_, size_t height_, size_t align = 512)
      : width_bytes(width_bytes_), height(height_) {
    pitch = ((width_bytes + align - 1) / align) * align;
    HIP_CHECK(hipMalloc(&ptr, pitch * height));
  }
  ~FlatPitched() { static_cast<void>(hipFree(ptr)); }
  FlatPitched(const FlatPitched&) = delete;
  FlatPitched(FlatPitched&&) = delete;
};

void Verify2DMemsetResult(void* dst, size_t pitch, size_t width_bytes, size_t height,
                          unsigned char expected) {
  std::vector<unsigned char> host(width_bytes * height);
  HIP_CHECK(hipMemcpy2D(host.data(), width_bytes, dst, pitch, width_bytes, height,
                        hipMemcpyDeviceToHost));
  ArrayFindIfNot(host.data(), expected, width_bytes * height);
}

hipMemsetParams Make2DParams(void* dst, size_t pitch, size_t width, size_t height,
                             unsigned int element_size, unsigned int value) {
  hipMemsetParams p = {};
  p.dst = dst;
  p.pitch = pitch;
  p.elementSize = element_size;
  p.width = width;
  p.height = height;
  p.value = value;
  return p;
}

}  // namespace

/**
 * Test Description
 * ------------------------
 *  - hipGraphMemsetNodeSetParams (template-edit, isExec=false path) accepts a
 *    2D memset whose destination is plain hipMalloc memory. Previously the
 *    implementation rejected this with hipErrorInvalidValue because it
 *    consulted per-allocation extent metadata that hipMalloc never sets.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphMemsetNodeFlatAlloc.cc
 */
HIP_TEST_CASE(Unit_hipGraphMemsetNodeSetParams_FlatAlloc_2D_Positive) {
  constexpr size_t width_bytes = 456;
  constexpr size_t height = 123;
  FlatPitched alloc(width_bytes, height);

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // Seed node with placeholder params (1D so AddMemsetNode trivially accepts).
  auto initial = Make2DParams(alloc.ptr, alloc.pitch, 1, 1, 1, 0);
  hipGraphNode_t node = nullptr;
  HIP_CHECK(hipGraphAddMemsetNode(&node, graph, nullptr, 0, &initial));

  // Now reshape to a real 2D memset on the same flat allocation.
  auto reshaped = Make2DParams(alloc.ptr, alloc.pitch, width_bytes, height, 1, 0xAB);
  HIP_CHECK(hipGraphMemsetNodeSetParams(node, &reshaped));

  hipGraphExec_t exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(exec, hipStreamPerThread));
  HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));

  Verify2DMemsetResult(alloc.ptr, alloc.pitch, width_bytes, height, 0xAB);

  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *  - When the destination IS a hipMallocPitch allocation, the original
 *    declared-extent enforcement must remain: requesting a 2D memset whose
 *    width/height exceeds the declared extent is rejected. Regression guard
 *    against the FlatAlloc fix above accidentally weakening pitched-alloc
 *    validation.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphMemsetNodeFlatAlloc.cc
 */
HIP_TEST_CASE(Unit_hipGraphMemsetNodeSetParams_PitchedAlloc_OverExtent_Negative) {
  constexpr size_t width = 64;
  constexpr size_t height = 16;
  LinearAllocGuard2D<unsigned char> alloc(width, height);  // hipMallocPitch

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  auto initial = Make2DParams(alloc.ptr(), alloc.pitch(), width, height, 1, 0);
  hipGraphNode_t node = nullptr;
  HIP_CHECK(hipGraphAddMemsetNode(&node, graph, nullptr, 0, &initial));

  auto over = Make2DParams(alloc.ptr(), alloc.pitch(), width * 2, height * 2, 1, 0);
  HIP_CHECK_ERROR(hipGraphMemsetNodeSetParams(node, &over), hipErrorInvalidValue);

  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *  - Reproduces the original test1.cpp scenario: build two stream-captured
 *    graphs that both invoke hipMemset2DAsync on the same plain hipMalloc
 *    pointer with identical parameters, instantiate the first, then update
 *    the executable graph using the second. Should succeed.
 *  - Exercises hipGraphExecUpdate's polymorphic SetParams(GraphNode*) path
 *    on a memset node whose destination is a flat allocation.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphMemsetNodeFlatAlloc.cc
 */
HIP_TEST_CASE(Unit_hipGraphExecUpdate_FlatAlloc_2DMemset_Idempotent) {
  constexpr size_t width_bytes = 456;
  constexpr size_t height = 123;
  FlatPitched alloc(width_bytes, height);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  auto capture_graph = [&](unsigned char value) {
    hipGraph_t g = nullptr;
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
    HIP_CHECK(hipMemset2DAsync(alloc.ptr, alloc.pitch, value, width_bytes, height, stream));
    HIP_CHECK(hipStreamEndCapture(stream, &g));
    return g;
  };

  hipGraph_t gA = capture_graph(0);
  hipGraph_t gB = capture_graph(0);

  hipGraphExec_t exec = nullptr;
  HIP_CHECK(hipGraphInstantiateWithFlags(&exec, gA, 0));

  hipGraphNode_t err_node = nullptr;
  hipGraphExecUpdateResult result = hipGraphExecUpdateError;
  HIP_CHECK(hipGraphExecUpdate(exec, gB, &err_node, &result));
  REQUIRE(result == hipGraphExecUpdateSuccess);

  HIP_CHECK(hipGraphLaunch(exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  Verify2DMemsetResult(alloc.ptr, alloc.pitch, width_bytes, height, 0);

  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(gB));
  HIP_CHECK(hipGraphDestroy(gA));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 *  - hipGraphExecUpdate must reject any 2D memset shape change. Per CUDA's
 *    documented contract for cudaGraphExecUpdate: "For 2d memsets, only
 *    address and assigned value may be updated."
 *  - Covers width, height, and elementSize changes individually.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphMemsetNodeFlatAlloc.cc
 */
HIP_TEST_CASE(Unit_hipGraphExecUpdate_2DMemset_ShapeChange_Negative) {
  constexpr size_t width_bytes = 128;
  constexpr size_t height = 8;
  FlatPitched alloc(width_bytes, height);

  hipGraph_t gA = nullptr;
  HIP_CHECK(hipGraphCreate(&gA, 0));
  auto base = Make2DParams(alloc.ptr, alloc.pitch, width_bytes, height, 1, 0);
  hipGraphNode_t nodeA = nullptr;
  HIP_CHECK(hipGraphAddMemsetNode(&nodeA, gA, nullptr, 0, &base));

  hipGraphExec_t exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, gA, nullptr, nullptr, 0));

  auto try_update = [&](const hipMemsetParams& changed) {
    hipGraph_t gB = nullptr;
    HIP_CHECK(hipGraphCreate(&gB, 0));
    hipGraphNode_t nodeB = nullptr;
    HIP_CHECK(hipGraphAddMemsetNode(&nodeB, gB, nullptr, 0, &changed));
    hipGraphNode_t err_node = nullptr;
    hipGraphExecUpdateResult result = hipGraphExecUpdateSuccess;
    hipError_t status = hipGraphExecUpdate(exec, gB, &err_node, &result);
    REQUIRE(status == hipErrorGraphExecUpdateFailure);
    REQUIRE(result == hipGraphExecUpdateErrorParametersChanged);
    HIP_CHECK(hipGraphDestroy(gB));
  };

  SECTION("width changes") {
    try_update(Make2DParams(alloc.ptr, alloc.pitch, width_bytes / 2, height, 1, 0));
  }
  SECTION("height changes") {
    try_update(Make2DParams(alloc.ptr, alloc.pitch, width_bytes, height / 2, 1, 0));
  }
  SECTION("byte-width changes via elementSize") {
    // HIP's strict check compares `width * elementSize` as a single byte count, so an
    // elementSize change that *preserves* total bytes is allowed (more lenient than CUDA's
    // "only address and value may change" contract). Verify rejection only when the byte count
    // actually grows.
    try_update(Make2DParams(alloc.ptr, alloc.pitch, width_bytes, height, 2, 0));
  }

  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(gA));
}

/**
 * Test Description
 * ------------------------
 *  - hipGraphExecUpdate must accept changes to value and dst on a 2D memset
 *    node, even though shape is fixed (per CUDA spec: address and value are
 *    the two updatable fields).
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphMemsetNodeFlatAlloc.cc
 */
HIP_TEST_CASE(Unit_hipGraphExecUpdate_2DMemset_ValueAndDst_Positive) {
  constexpr size_t width_bytes = 128;
  constexpr size_t height = 8;
  FlatPitched allocA(width_bytes, height);
  FlatPitched allocB(width_bytes, height);

  hipGraph_t gA = nullptr;
  HIP_CHECK(hipGraphCreate(&gA, 0));
  auto pA = Make2DParams(allocA.ptr, allocA.pitch, width_bytes, height, 1, 0x11);
  hipGraphNode_t nodeA = nullptr;
  HIP_CHECK(hipGraphAddMemsetNode(&nodeA, gA, nullptr, 0, &pA));

  hipGraphExec_t exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, gA, nullptr, nullptr, 0));

  hipGraph_t gB = nullptr;
  HIP_CHECK(hipGraphCreate(&gB, 0));
  // dst -> allocB, value -> 0x77, but shape unchanged.
  auto pB = Make2DParams(allocB.ptr, allocB.pitch, width_bytes, height, 1, 0x77);
  hipGraphNode_t nodeB = nullptr;
  HIP_CHECK(hipGraphAddMemsetNode(&nodeB, gB, nullptr, 0, &pB));

  hipGraphNode_t err_node = nullptr;
  hipGraphExecUpdateResult result = hipGraphExecUpdateError;
  HIP_CHECK(hipGraphExecUpdate(exec, gB, &err_node, &result));
  REQUIRE(result == hipGraphExecUpdateSuccess);

  HIP_CHECK(hipGraphLaunch(exec, hipStreamPerThread));
  HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
  Verify2DMemsetResult(allocB.ptr, allocB.pitch, width_bytes, height, 0x77);

  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(gB));
  HIP_CHECK(hipGraphDestroy(gA));
}

/**
 * Test Description
 * ------------------------
 *  - hipGraphExecUpdate recurses into child-graph nodes; the child-graph
 *    branch must enter the same exec-strict path as top-level memset nodes.
 *    Builds a host graph containing a child graph whose body is a 2D memset
 *    on plain hipMalloc memory, then updates the exec graph from a second
 *    identically-shaped graph.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphMemsetNodeFlatAlloc.cc
 */
HIP_TEST_CASE(Unit_hipGraphExecUpdate_FlatAlloc_2DMemset_ChildGraph) {
  constexpr size_t width_bytes = 200;
  constexpr size_t height = 10;
  FlatPitched alloc(width_bytes, height);

  auto build_outer = [&](unsigned char value, hipGraph_t* outer_out) {
    hipGraph_t inner = nullptr;
    HIP_CHECK(hipGraphCreate(&inner, 0));
    auto p = Make2DParams(alloc.ptr, alloc.pitch, width_bytes, height, 1, value);
    hipGraphNode_t inner_node = nullptr;
    HIP_CHECK(hipGraphAddMemsetNode(&inner_node, inner, nullptr, 0, &p));

    hipGraph_t outer = nullptr;
    HIP_CHECK(hipGraphCreate(&outer, 0));
    hipGraphNode_t child = nullptr;
    HIP_CHECK(hipGraphAddChildGraphNode(&child, outer, nullptr, 0, inner));

    HIP_CHECK(hipGraphDestroy(inner));
    *outer_out = outer;
  };

  hipGraph_t gA = nullptr;
  hipGraph_t gB = nullptr;
  build_outer(0x33, &gA);
  build_outer(0x33, &gB);

  hipGraphExec_t exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, gA, nullptr, nullptr, 0));

  hipGraphNode_t err_node = nullptr;
  hipGraphExecUpdateResult result = hipGraphExecUpdateError;
  HIP_CHECK(hipGraphExecUpdate(exec, gB, &err_node, &result));
  REQUIRE(result == hipGraphExecUpdateSuccess);

  HIP_CHECK(hipGraphLaunch(exec, hipStreamPerThread));
  HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
  Verify2DMemsetResult(alloc.ptr, alloc.pitch, width_bytes, height, 0x33);

  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(gB));
  HIP_CHECK(hipGraphDestroy(gA));
}

/**
 * Test Description
 * ------------------------
 *  - hipGraphExecMemsetNodeSetParams on a 2D memset whose destination is a
 *    plain hipMalloc allocation: changing only `value` (preserving shape)
 *    must succeed; changing shape (width/height) must fail per the same
 *    CUDA contract that governs hipGraphExecUpdate.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphMemsetNodeFlatAlloc.cc
 */
HIP_TEST_CASE(Unit_hipGraphExecMemsetNodeSetParams_FlatAlloc_2D) {
  constexpr size_t width_bytes = 128;
  constexpr size_t height = 8;
  FlatPitched alloc(width_bytes, height);

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  auto base = Make2DParams(alloc.ptr, alloc.pitch, width_bytes, height, 1, 0);
  hipGraphNode_t node = nullptr;
  HIP_CHECK(hipGraphAddMemsetNode(&node, graph, nullptr, 0, &base));

  hipGraphExec_t exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  SECTION("value-only change accepted") {
    auto only_value = Make2DParams(alloc.ptr, alloc.pitch, width_bytes, height, 1, 0x5A);
    HIP_CHECK(hipGraphExecMemsetNodeSetParams(exec, node, &only_value));
    HIP_CHECK(hipGraphLaunch(exec, hipStreamPerThread));
    HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
    Verify2DMemsetResult(alloc.ptr, alloc.pitch, width_bytes, height, 0x5A);
  }

  SECTION("width change rejected") {
    auto width_change = Make2DParams(alloc.ptr, alloc.pitch, width_bytes / 2, height, 1, 0);
    HIP_CHECK_ERROR(hipGraphExecMemsetNodeSetParams(exec, node, &width_change),
                    hipErrorInvalidValue);
  }

  SECTION("height change rejected") {
    auto height_change = Make2DParams(alloc.ptr, alloc.pitch, width_bytes, height / 2, 1, 0);
    HIP_CHECK_ERROR(hipGraphExecMemsetNodeSetParams(exec, node, &height_change),
                    hipErrorInvalidValue);
  }

  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *  - Verify that a 3D memset update uses the new allocation while preserving the operation extent.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphMemsetNodeFlatAlloc.cc
 */
HIP_TEST_CASE(Unit_hipGraphExecUpdate_3DMemset_AllocationExtent) {
  const hipExtent allocation_extent_a = make_hipExtent(64, 4, 2);
  const hipExtent allocation_extent_b = make_hipExtent(128, 8, 2);
  const hipExtent operation_extent = make_hipExtent(16, 2, 2);
  LinearAllocGuard3D<unsigned char> allocation_a(allocation_extent_a);
  LinearAllocGuard3D<unsigned char> allocation_b(allocation_extent_b);
  HIP_CHECK(hipMemset3D(allocation_b.pitched_ptr(), 0, allocation_extent_b));

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  auto capture_graph = [&](hipPitchedPtr allocation, int value) {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    HIP_CHECK(hipMemset3DAsync(allocation, value, operation_extent, stream));
    HIP_CHECK(hipStreamEndCapture(stream, &graph));
    return graph;
  };

  hipGraph_t graph_a = capture_graph(allocation_a.pitched_ptr(), 0x11);
  hipGraph_t graph_b = capture_graph(allocation_b.pitched_ptr(), 0x5A);
  hipGraphExec_t exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, graph_a, nullptr, nullptr, 0));
  hipGraphNode_t error_node = nullptr;
  hipGraphExecUpdateResult result = hipGraphExecUpdateError;
  HIP_CHECK(hipGraphExecUpdate(exec, graph_b, &error_node, &result));
  REQUIRE(result == hipGraphExecUpdateSuccess);
  HIP_CHECK(hipGraphLaunch(exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  auto allocation = allocation_b.pitched_ptr();
  auto* base = static_cast<unsigned char*>(allocation.ptr);
  unsigned char first_slice = 0;
  unsigned char second_slice = 0;
  HIP_CHECK(hipMemcpy(&first_slice, base, 1, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&second_slice, base + allocation.pitch * allocation.ysize, 1,
                      hipMemcpyDeviceToHost));
  REQUIRE(first_slice == 0x5A);
  REQUIRE(second_slice == 0x5A);
  unsigned char outside_width = 0xFF, outside_height = 0xFF;
  HIP_CHECK(hipMemcpy(&outside_width, base + operation_extent.width, 1, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&outside_height, base + allocation.pitch * operation_extent.height, 1,
                      hipMemcpyDeviceToHost));
  REQUIRE(outside_width == 0);
  REQUIRE(outside_height == 0);

  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph_b));
  HIP_CHECK(hipGraphDestroy(graph_a));
  HIP_CHECK(hipStreamDestroy(stream));
}

#if HT_AMD
/**
 * Test Description
 * ------------------------
 *  - hipGraphExecNodeSetParams (the generic node-params variant) on a memset
 *    node must follow the same exec-strict contract as
 *    hipGraphExecMemsetNodeSetParams. Previously this entry point hard-coded
 *    isExec=false, allowing shape changes that violate CUDA's contract.
 *  - AMD-only: hipGraphNodeParams / hipGraphExecNodeSetParams are not
 *    implemented for the NVIDIA backend in this tree (per CMakeLists).
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphMemsetNodeFlatAlloc.cc
 */
HIP_TEST_CASE(Unit_hipGraphExecNodeSetParams_Memset_FlatAlloc_2D) {
  constexpr size_t width_bytes = 128;
  constexpr size_t height = 8;
  FlatPitched alloc(width_bytes, height);

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNodeParams initial = {};
  initial.type = hipGraphNodeTypeMemset;
  initial.memset = Make2DParams(alloc.ptr, alloc.pitch, width_bytes, height, 1, 0);
  hipGraphNode_t node = nullptr;
  HIP_CHECK(hipGraphAddNode(&node, graph, nullptr, 0, &initial));

  hipGraphExec_t exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  SECTION("value-only change accepted") {
    hipGraphNodeParams only_value = {};
    only_value.type = hipGraphNodeTypeMemset;
    only_value.memset = Make2DParams(alloc.ptr, alloc.pitch, width_bytes, height, 1, 0x9C);
    HIP_CHECK(hipGraphExecNodeSetParams(exec, node, &only_value));
    HIP_CHECK(hipGraphLaunch(exec, hipStreamPerThread));
    HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
    Verify2DMemsetResult(alloc.ptr, alloc.pitch, width_bytes, height, 0x9C);
  }

  SECTION("shape change rejected") {
    hipGraphNodeParams shape_change = {};
    shape_change.type = hipGraphNodeTypeMemset;
    shape_change.memset = Make2DParams(alloc.ptr, alloc.pitch, width_bytes, height / 2, 1, 0);
    HIP_CHECK_ERROR(hipGraphExecNodeSetParams(exec, node, &shape_change), hipErrorInvalidValue);
  }

  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
}
#endif  // HT_AMD


/**
 * Test Description
 * ------------------------
 *  - A single-row graph node ignores pitch and retains the 1D fill alignment behavior.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphMemsetNodeFlatAlloc.cc
 */
HIP_TEST_CASE(Unit_hipGraphMemsetNode_SingleRowIgnoresPitch) {
  LinearAllocGuard<uint8_t> device(LinearAllocs::hipMalloc, 512);
  StreamGuard stream(Streams::withFlags, hipStreamNonBlocking);
  std::vector<uint8_t> expected(512, 0xa5);
  std::vector<uint8_t> output(512);
  HIP_CHECK(hipMemcpy(device.ptr(), expected.data(), expected.size(), hipMemcpyHostToDevice));
  hipMemsetParams params{};
  params.dst = device.ptr() + 65;
  params.pitch = 2;
  params.value = 0x31313131;
  params.elementSize = 4;
  params.width = 3;
  params.height = 1;
  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;
  hipGraphExec_t executable = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphAddMemsetNode(&node, graph, nullptr, 0, &params));
  HIP_CHECK(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(executable, stream.stream()));
  HIP_CHECK(hipStreamSynchronize(stream.stream()));
  HIP_CHECK(hipGraphExecDestroy(executable));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipMemcpy(output.data(), device.ptr(), output.size(), hipMemcpyDeviceToHost));
  std::fill_n(expected.begin() + 65, 12, 0x31);
  REQUIRE(output == expected);
}

/**
 * Test Description
 * ------------------------
 *  - Captured and explicit graph fills span multiple adjacent VMM physical mappings.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphMemsetNodeFlatAlloc.cc
 */
HIP_TEST_CASE(Unit_hipGraphMemsetNode_SpansVmmMappings) {
  const auto mode =
      GENERATE(VmmFillMode::CaptureBytes, VmmFillMode::CaptureTyped, VmmFillMode::Graph);
  int device = 0;
  int supported = 0;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipDeviceGetAttribute(&supported, hipDeviceAttributeVirtualMemoryManagementSupported,
                                  device));
  if (!supported) HIP_SKIP_TEST("Virtual memory management is unavailable");
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;
  size_t granularity = 0;
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  void* ptr = nullptr;
  hipMemGenericAllocationHandle_t handles[2]{};
  HIP_CHECK(hipMemAddressReserve(&ptr, 2 * granularity, 0, nullptr, 0));
  for (size_t chunk = 0; chunk < 2; ++chunk) {
    HIP_CHECK(hipMemCreate(&handles[chunk], granularity, &prop, 0));
    HIP_CHECK(hipMemMap(static_cast<uint8_t*>(ptr) + chunk * granularity, granularity, 0,
                        handles[chunk], 0));
  }
  hipMemAccessDesc access{};
  access.location = prop.location;
  access.flags = hipMemAccessFlagsProtReadWrite;
  HIP_CHECK(hipMemSetAccess(ptr, 2 * granularity, &access, 1));
  StreamGuard stream(Streams::withFlags, hipStreamNonBlocking);
  std::vector<uint8_t> output(2 * granularity, 0);
  HIP_CHECK(hipMemcpy(ptr, output.data(), output.size(), hipMemcpyHostToDevice));
  hipMemsetParams params{};
  params.dst = ptr;
  params.value = 0x5a5a5a5a;
  params.elementSize = 4;
  params.width = output.size() / 4;
  params.height = 1;
  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;
  if (mode == VmmFillMode::Graph) {
    HIP_CHECK(hipGraphCreate(&graph, 0));
    HIP_CHECK(hipGraphAddMemsetNode(&node, graph, nullptr, 0, &params));
    HIP_CHECK(hipGraphMemsetNodeSetParams(node, &params));
  } else {
    HIP_CHECK(hipStreamBeginCapture(stream.stream(), hipStreamCaptureModeGlobal));
    if (mode == VmmFillMode::CaptureBytes) {
      HIP_CHECK(hipMemsetAsync(ptr, 0x5a, output.size(), stream.stream()));
    } else {
      HIP_CHECK(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(ptr), 0x5a5a5a5a,
                                  output.size() / 4, stream.stream()));
    }
    HIP_CHECK(hipStreamEndCapture(stream.stream(), &graph));
  }
  hipGraphExec_t executable = nullptr;
  HIP_CHECK(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  if (mode == VmmFillMode::Graph) {
    HIP_CHECK(hipGraphExecMemsetNodeSetParams(executable, node, &params));
  }
  HIP_CHECK(hipGraphLaunch(executable, stream.stream()));
  HIP_CHECK(hipStreamSynchronize(stream.stream()));
  HIP_CHECK(hipGraphExecDestroy(executable));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipMemcpy(output.data(), ptr, output.size(), hipMemcpyDeviceToHost));
  for (size_t chunk = 0; chunk < 2; ++chunk) {
    HIP_CHECK(hipMemUnmap(static_cast<uint8_t*>(ptr) + chunk * granularity, granularity));
    HIP_CHECK(hipMemRelease(handles[chunk]));
  }
  HIP_CHECK(hipMemAddressFree(ptr, output.size()));
  REQUIRE(std::all_of(output.begin(), output.end(), [](uint8_t value) { return value == 0x5a; }));
}

/**
 * Test Description
 * ------------------------
 *  - Reject overflowing, misaligned and out-of-allocation graph fill geometry.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphMemsetNodeFlatAlloc.cc
 */
HIP_TEST_CASE(Unit_hipGraphMemsetNodeFlatAlloc_InvalidGeometry) {
  constexpr size_t max_size = std::numeric_limits<size_t>::max();
  LinearAllocGuard<uint8_t> device(LinearAllocs::hipMalloc, 2048);
  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;
  hipGraphExec_t executable = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipMemsetParams params{};
  params.dst = device.ptr();
  params.pitch = 512;
  params.value = 7;
  params.elementSize = 4;
  params.width = 17;
  params.height = 2;
  HIP_CHECK(hipGraphAddMemsetNode(&node, graph, nullptr, 0, &params));
  HIP_CHECK(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  const auto valid_params = params;
  const auto invalid = GENERATE(InvalidGeometry::WidthOverflow, InvalidGeometry::AreaOverflow,
                                InvalidGeometry::Misaligned, InvalidGeometry::OutsideSpan);
  {
    params = valid_params;
    if (invalid == InvalidGeometry::WidthOverflow) params.width = max_size / 4 + 1;
    if (invalid == InvalidGeometry::AreaOverflow) {
      params.width = max_size / 8 + 1;
      params.height = 4;
    }
    if (invalid == InvalidGeometry::Misaligned) params.dst = device.ptr() + 1;
    if (invalid == InvalidGeometry::OutsideSpan) params.dst = device.ptr() + 1536;
    hipGraphNode_t rejected = nullptr;
    HIP_CHECK_ERROR(hipGraphAddMemsetNode(&rejected, graph, nullptr, 0, &params),
                    hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipGraphMemsetNodeSetParams(node, &params), hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipGraphExecMemsetNodeSetParams(executable, node, &params),
                    hipErrorInvalidValue);
  }
  HIP_CHECK(hipGraphExecDestroy(executable));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *  - Update a captured fill to a different slice pitch and validate the incoming span.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphMemsetNodeFlatAlloc.cc
 */
HIP_TEST_CASE(Unit_hipGraphExecUpdate_MemsetSliceHeight) {
  constexpr size_t pitch = 256;
  constexpr size_t width = 64;
  constexpr size_t height = 4;
  constexpr size_t depth = 3;
  const size_t new_height = GENERATE(4, 8);
  const size_t old_height = new_height == 4 ? 8 : 4;
  const auto span = [](size_t slice_height) {
    return (depth - 1) * pitch * slice_height + (height - 1) * pitch + width;
  };
  LinearAllocGuard<uint8_t> original(LinearAllocs::hipMalloc, span(old_height));
  LinearAllocGuard<uint8_t> replacement(LinearAllocs::hipMalloc, span(new_height));
  StreamGuard stream(Streams::withFlags, hipStreamNonBlocking);
  std::vector<uint8_t> expected(span(new_height), 0xa5);
  std::vector<uint8_t> output(expected.size());
  HIP_CHECK(hipMemcpy(replacement.ptr(), expected.data(), expected.size(), hipMemcpyHostToDevice));
  const auto capture = [&](void* ptr, size_t slice_height) {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(stream.stream(), hipStreamCaptureModeGlobal));
    HIP_CHECK(hipMemset3DAsync(make_hipPitchedPtr(ptr, pitch, width, slice_height), 0x22,
                               make_hipExtent(width, height, depth), stream.stream()));
    HIP_CHECK(hipStreamEndCapture(stream.stream(), &graph));
    return graph;
  };
  hipGraph_t first = capture(original.ptr(), old_height);
  hipGraph_t second = capture(replacement.ptr(), new_height);
  hipGraphExec_t executable = nullptr;
  HIP_CHECK(hipGraphInstantiate(&executable, first, nullptr, nullptr, 0));
  hipGraphNode_t error_node = nullptr;
  hipGraphExecUpdateResult update_result{};
  HIP_CHECK(hipGraphExecUpdate(executable, second, &error_node, &update_result));
  HIP_CHECK(hipGraphLaunch(executable, stream.stream()));
  HIP_CHECK(hipStreamSynchronize(stream.stream()));
  HIP_CHECK(hipGraphExecDestroy(executable));
  HIP_CHECK(hipGraphDestroy(first));
  HIP_CHECK(hipGraphDestroy(second));
  HIP_CHECK(hipMemcpy(output.data(), replacement.ptr(), output.size(), hipMemcpyDeviceToHost));
  for (size_t z = 0; z < depth; ++z) {
    for (size_t y = 0; y < height; ++y) {
      std::fill_n(expected.begin() + z * pitch * new_height + y * pitch, width, 0x22);
    }
  }
  REQUIRE(update_result == hipGraphExecUpdateSuccess);
  REQUIRE(output == expected);
}

/**
 * End doxygen group GraphTest.
 * @}
 */
