/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipStreamBeginRecaptureToGraph hipStreamBeginRecaptureToGraph
 * @{
 * @ingroup GraphTest
 * `hipError_t hipStreamBeginRecaptureToGraph(hipStream_t stream, hipStreamCaptureMode mode,
 *                                            hipGraph_t graph,
 *                                            hipGraphRecaptureCallbackData* callbackData);` -
 *  Begins a stream capture that matches every captured operation, in creation order, against
 *  the existing nodes of a graph and updates them in place.
 */
#include <hip_test_common.hh>
#include <resource_guards.hh>

#include <atomic>
#include <chrono>
#include <functional>
#include <thread>
#include <vector>

// Available on cuda 13.3 and above
#if !HT_NVIDIA || CUDA_VERSION >= CUDA_13030

namespace {
constexpr size_t kInts = 4;
constexpr size_t kBytes = kInts * sizeof(int);
constexpr size_t kAllocBytes = 1024;
constexpr hipStreamCaptureMode kMode = hipStreamCaptureModeThreadLocal;

__global__ void WriteValue(int* output, int value) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *output = value;
  }
}

__global__ void AddValue(int* output, int value) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *output += value;
  }
}

__global__ void CopyValue(int* dst, const int* src) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *dst = *src;
  }
}

__device__ int g_symbol[kInts];

void HostNoop(void*) {}
void OtherHostNoop(void*) {}

std::atomic<int> g_userObjectsDestroyed{0};
void DestroyUserObject(void*) { g_userObjectsDestroyed.fetch_add(1); }

struct Resources {
  StreamGuard stream{Streams::created};
  StreamGuard side{Streams::created};
  EventsGuard events{4};
  LinearAllocGuard<int> output{LinearAllocs::hipMalloc, kBytes};
  LinearAllocGuard<int> scratch{LinearAllocs::hipMalloc, kBytes};
  LinearAllocGuard<int> host{LinearAllocs::hipHostMalloc, kBytes};

  Resources() {
    const int scratchValues[kInts] = {7, 8, 9, 10};
    HIP_CHECK(hipMemcpy(scratch.ptr(), scratchValues, kBytes, hipMemcpyHostToDevice));
    for (size_t i = 0; i < kInts; ++i) {
      host.host_ptr()[i] = 21 + static_cast<int>(i);
    }
  }

  hipStream_t s() const { return stream.stream(); }
  hipStream_t sideStream() const { return side.stream(); }
  int* out() const { return output.ptr(); }
  hipEvent_t fork() { return events[0]; }
  hipEvent_t join() { return events[1]; }
  hipEvent_t eventA() { return events[2]; }
  hipEvent_t eventB() { return events[3]; }
};

struct CallbackState {
  int calls = 0;
  int rejectCall = 0;  // 1-based callback invocation to reject; 0 rejects none.
  bool rejectAll = false;
  bool typesMatch = true;
  std::vector<hipGraphNode_t> nodes;
  std::vector<hipGraphRecaptureStatus> statuses;
  std::vector<int> originalValues;   // Second argument of WriteValue / AddValue, else -1.
  std::vector<int> recaptureValues;  // Second argument of WriteValue / AddValue, else -1.
  std::function<hipError_t()> onFirstCall;  // Graph edit run inside the first callback.
  hipError_t onFirstCallResult = hipSuccess;
};

int KernelValue(const hipGraphNodeParams* params) {
  if (params->type != hipGraphNodeTypeKernel || params->kernel.kernelParams == nullptr) {
    return -1;
  }
  if (params->kernel.func != reinterpret_cast<void*>(WriteValue) &&
      params->kernel.func != reinterpret_cast<void*>(AddValue)) {
    return -1;
  }
  return *static_cast<int*>(params->kernel.kernelParams[1]);
}

// Runs inside the capturing HIP call, so it records only and does not use Catch macros.
hipError_t RecordCallback(void* data, hipGraphNode_t node, const hipGraphNodeParams* original,
                          const hipGraphNodeParams* recaptured, hipGraphRecaptureStatus status) {
  auto* state = static_cast<CallbackState*>(data);
  ++state->calls;
  state->nodes.push_back(node);
  state->statuses.push_back(status);
  state->typesMatch = state->typesMatch && original != nullptr && recaptured != nullptr &&
                      original->type == recaptured->type;
  state->originalValues.push_back(original != nullptr ? KernelValue(original) : -1);
  state->recaptureValues.push_back(recaptured != nullptr ? KernelValue(recaptured) : -1);
  if (state->calls == 1 && state->onFirstCall) {
    state->onFirstCallResult = state->onFirstCall();
  }
  const bool reject = state->rejectAll || state->calls == state->rejectCall;
  return reject ? hipErrorInvalidValue : hipSuccess;
}

// A failed recapture call leaves its error in the last-error state; clear it first so the
// result belongs to this launch only.
hipError_t LaunchWrite(hipStream_t stream, int* output, int value, dim3 grid = dim3(1),
                       dim3 block = dim3(1), size_t sharedBytes = 0) {
  (void)hipGetLastError();
  hipLaunchKernelGGL(WriteValue, grid, block, sharedBytes, stream, output, value);
  return hipGetLastError();
}

hipError_t LaunchAdd(hipStream_t stream, int* output, int value) {
  (void)hipGetLastError();
  hipLaunchKernelGGL(AddValue, dim3(1), dim3(1), 0, stream, output, value);
  return hipGetLastError();
}

hipError_t LaunchCopy(hipStream_t stream, int* dst, const int* src) {
  (void)hipGetLastError();
  hipLaunchKernelGGL(CopyValue, dim3(1), dim3(1), 0, stream, dst, src);
  return hipGetLastError();
}

hipError_t LaunchWriteEx(hipStream_t stream, int* output, int value, hipLaunchAttribute* attrs,
                         unsigned int numAttrs, dim3 grid = dim3(1)) {
  hipLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = dim3(1);
  config.stream = stream;
  config.attrs = attrs;
  config.numAttrs = numAttrs;
  void* args[] = {&output, &value};
  return hipLaunchKernelExC(&config, reinterpret_cast<const void*>(WriteValue), args);
}

hipError_t Memcpy3D(hipStream_t stream, int* dst, int* src, size_t widthBytes) {
  hipMemcpy3DParms params{};
  params.srcPtr = make_hipPitchedPtr(src, kBytes, widthBytes, 1);
  params.dstPtr = make_hipPitchedPtr(dst, kBytes, widthBytes, 1);
  params.extent = make_hipExtent(widthBytes, 1, 1);
  params.kind = hipMemcpyDeviceToDevice;
  return hipMemcpy3DAsync(&params, stream);
}

hipError_t EnqueueChain(hipStream_t stream, int* output, int a, int b) {
  const hipError_t result = LaunchWrite(stream, output, a);
  return result == hipSuccess ? LaunchWrite(stream, output + 1, b) : result;
}

// A writes output[0] on the origin stream, B writes output[1] on the side stream. Returns the
// first error; B's own result is also stored in bResult when it is non-null.
hipError_t EnqueueForked(Resources& r, int a, int b, bool sideFirst,
                         hipError_t* bResult = nullptr) {
  HIP_CHECK(hipEventRecord(r.fork(), r.s()));
  HIP_CHECK(hipStreamWaitEvent(r.sideStream(), r.fork(), 0));
  hipError_t resultB = hipSuccess;
  hipError_t resultA = hipSuccess;
  if (sideFirst) resultB = LaunchWrite(r.sideStream(), r.out() + 1, b);
  resultA = LaunchWrite(r.s(), r.out(), a);
  if (!sideFirst) resultB = LaunchWrite(r.sideStream(), r.out() + 1, b);
  HIP_CHECK(hipEventRecord(r.join(), r.sideStream()));
  HIP_CHECK(hipStreamWaitEvent(r.s(), r.join(), 0));
  if (bResult != nullptr) *bResult = resultB;
  return resultA != hipSuccess ? resultA : resultB;
}

template <typename Enqueue> hipGraph_t Capture(hipStream_t stream, Enqueue enqueue) {
  HIP_CHECK(hipStreamBeginCapture(stream, kMode));
  const hipError_t result = enqueue();
  hipGraph_t graph = nullptr;
  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  HIP_CHECK(result);
  REQUIRE(graph != nullptr);
  return graph;
}

struct EndResult {
  hipError_t error = hipSuccess;
  hipGraph_t graph = nullptr;
};

// Recaptures into graph and always ends the capture. The stream must leave capture either way.
template <typename Enqueue>
EndResult Recapture(hipStream_t stream, hipGraph_t graph, hipGraphRecaptureCallbackData* callback,
                    Enqueue enqueue, hipStreamCaptureMode mode = kMode) {
  HIP_CHECK(hipStreamBeginRecaptureToGraph(stream, mode, graph, callback));
  enqueue();
  EndResult end;
  end.error = hipStreamEndCapture(stream, &end.graph);
  hipStreamCaptureStatus status = hipStreamCaptureStatusActive;
  HIP_CHECK(hipStreamIsCapturing(stream, &status));
  REQUIRE(status == hipStreamCaptureStatusNone);
  return end;
}

void RequireEndSuccess(const EndResult& end, hipGraph_t graph) {
  HIP_CHECK(end.error);
  REQUIRE(end.graph == graph);
}

// A failed end destroys a top-level input graph, so callers must not touch it again.
void RequireEndFailure(const EndResult& end, hipError_t error = hipErrorGraphRecaptureFailure) {
  REQUIRE(end.error == error);
  REQUIRE(end.graph == nullptr);
}

std::vector<int> LaunchExecAndRead(hipGraphExec_t exec, hipStream_t stream, int* output,
                                   size_t count) {
  HIP_CHECK(hipMemsetAsync(output, 0, count * sizeof(int), stream));
  HIP_CHECK(hipGraphLaunch(exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  std::vector<int> values(count);
  HIP_CHECK(hipMemcpy(values.data(), output, count * sizeof(int), hipMemcpyDeviceToHost));
  return values;
}

std::vector<int> LaunchAndRead(hipGraph_t graph, hipStream_t stream, int* output,
                               size_t count) {
  hipGraphExec_t exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  std::vector<int> values = LaunchExecAndRead(exec, stream, output, count);
  HIP_CHECK(hipGraphExecDestroy(exec));
  return values;
}

std::vector<hipGraphNode_t> GetNodes(hipGraph_t graph) {
  size_t count = 0;
  HIP_CHECK(hipGraphGetNodes(graph, nullptr, &count));
  std::vector<hipGraphNode_t> nodes(count);
  if (count != 0) {
    HIP_CHECK(hipGraphGetNodes(graph, nodes.data(), &count));
  }
  return nodes;
}

// Captures original, recaptures changed with an accepting callback, and requires exactly one
// callback with the given status. Status 1 must predict that hipGraphExecUpdate of an exec
// instantiated before the recapture fails, and status 0 that it succeeds. When expected is
// non-empty the graph is launched and output[0..expected.size()) is compared with it.
template <typename Original, typename Changed>
void RequireChangeStatus(Resources& r, Original original, Changed changed,
                         hipGraphRecaptureStatus status, const std::vector<int>& expected = {}) {
  hipGraph_t graph = Capture(r.s(), original);
  const size_t nodeCount = GetNodes(graph).size();
  hipGraphExec_t exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  CallbackState state;
  hipGraphRecaptureCallbackData callback{RecordCallback, &state};
  const EndResult end = Recapture(r.s(), graph, &callback, [&] { HIP_CHECK(changed()); });
  RequireEndSuccess(end, graph);
  REQUIRE(state.calls == 1);
  REQUIRE(state.statuses[0] == status);
  REQUIRE(state.typesMatch);
  REQUIRE(GetNodes(graph).size() == nodeCount);
  hipGraphNode_t errorNode = nullptr;
  hipGraphExecUpdateResult updateResult = hipGraphExecUpdateError;
  const hipError_t updateError = hipGraphExecUpdate(exec, graph, &errorNode, &updateResult);
  REQUIRE((updateError == hipSuccess) == (status == hipGraphRecaptureEligibleForUpdate));
  HIP_CHECK(hipGraphExecDestroy(exec));
  if (!expected.empty()) {
    REQUIRE(LaunchAndRead(graph, r.s(), r.out(), expected.size()) == expected);
  }
  HIP_CHECK(hipGraphDestroy(graph));
}

hipError_t EnqueueAllocChain(hipStream_t stream, int* output, size_t bytes, int value,
                             void** allocation) {
  hipError_t result = hipMallocAsync(allocation, bytes, stream);
  if (result != hipSuccess) return result;
  int* allocated = static_cast<int*>(*allocation);
  result = LaunchWrite(stream, allocated, value);
  if (result != hipSuccess) return result;
  result = LaunchCopy(stream, output, allocated);
  if (result != hipSuccess) return result;
  return hipFreeAsync(*allocation, stream);
}

bool WaitForUserObjectsDestroyed(int count) {
  for (int i = 0; i < 100 && g_userObjectsDestroyed.load() < count; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return g_userObjectsDestroyed.load() == count;
}

hipGraphNode_t AddKernelNode(hipGraph_t graph, const hipGraphNode_t* deps, size_t numDeps,
                             int* output, int value) {
  void* args[] = {&output, &value};
  hipKernelNodeParams params{};
  params.func = reinterpret_cast<void*>(WriteValue);
  params.gridDim = dim3(1);
  params.blockDim = dim3(1);
  params.kernelParams = args;
  hipGraphNode_t node = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&node, graph, deps, numDeps, &params));
  return node;
}

// A parent graph with one child graph node whose embedded graph holds one kernel per value.
struct ChildGraph {
  hipGraph_t parent = nullptr;
  hipGraphNode_t childNode = nullptr;
  hipGraph_t embedded = nullptr;

  ChildGraph(int* output, const std::vector<int>& values) {
    hipGraph_t child = nullptr;
    HIP_CHECK(hipGraphCreate(&child, 0));
    hipGraphNode_t previous = nullptr;
    for (size_t i = 0; i < values.size(); ++i) {
      previous = AddKernelNode(child, previous != nullptr ? &previous : nullptr,
                               previous != nullptr ? 1 : 0, output + i, values[i]);
    }
    HIP_CHECK(hipGraphCreate(&parent, 0));
    HIP_CHECK(hipGraphAddChildGraphNode(&childNode, parent, nullptr, 0, child));
    HIP_CHECK(hipGraphDestroy(child));
    HIP_CHECK(hipGraphChildGraphNodeGetGraph(childNode, &embedded));
  }
  ChildGraph(const ChildGraph&) = delete;
  ChildGraph& operator=(const ChildGraph&) = delete;
  ~ChildGraph() { static_cast<void>(hipGraphDestroy(parent)); }
};
}  // namespace

/**
 * Test Description
 * ------------------------
 *  - Recapturing one kernel with a new argument updates the node in place: the callback runs
 *    once with status 0 and the original node handle, end returns the input graph, the node
 *    handle is kept, an exec instantiated before is a snapshot, and hipGraphExecUpdate then
 *    applies the change.
 * Test source
 * ------------------------
 *  - unit/graph/hipStreamBeginRecaptureToGraph.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipStreamBeginRecaptureToGraph_InPlaceUpdate) {
  Resources r;
  hipGraph_t graph = Capture(r.s(), [&] { return LaunchWrite(r.s(), r.out(), 1); });
  const std::vector<hipGraphNode_t> before = GetNodes(graph);
  REQUIRE(before.size() == 1);
  hipGraphExec_t exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  CallbackState state;
  hipGraphRecaptureCallbackData callback{RecordCallback, &state};
  const EndResult end =
      Recapture(r.s(), graph, &callback, [&] { HIP_CHECK(LaunchWrite(r.s(), r.out(), 2)); });
  RequireEndSuccess(end, graph);
  REQUIRE(state.calls == 1);
  REQUIRE(state.statuses[0] == hipGraphRecaptureEligibleForUpdate);
  REQUIRE(state.nodes[0] == before[0]);
  REQUIRE(state.originalValues[0] == 1);
  REQUIRE(state.recaptureValues[0] == 2);
  REQUIRE(GetNodes(graph) == before);

  REQUIRE(LaunchExecAndRead(exec, r.s(), r.out(), 1) == std::vector<int>{1});
  hipGraphNode_t errorNode = nullptr;
  hipGraphExecUpdateResult updateResult = hipGraphExecUpdateError;
  HIP_CHECK(hipGraphExecUpdate(exec, graph, &errorNode, &updateResult));
  REQUIRE(updateResult == hipGraphExecUpdateSuccess);
  REQUIRE(LaunchExecAndRead(exec, r.s(), r.out(), 1) == std::vector<int>{2});
  HIP_CHECK(hipGraphExecDestroy(exec));

  REQUIRE(LaunchAndRead(graph, r.s(), r.out(), 1) == std::vector<int>{2});
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *  - Callback contract: an identical recapture makes no calls; without a callback, or with a
 *    null callbackFunc, changes are applied; a rejected single node fails its call with 918,
 *    keeps its old parameters and end succeeds; rejecting the first of two nodes terminates
 *    the recapture so both calls fail and end fails with a null graph.
 *  - Graph edits from the callback, matching CUDA 13.3: a rejected add (918) or destroy
 *    (hipErrorIllegalState) applies the update, then the capturing call returns that error and
 *    the recapture continues; a parameter edit succeeds and the next node is compared against it.
 * Test source
 * ------------------------
 *  - unit/graph/hipStreamBeginRecaptureToGraph.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipStreamBeginRecaptureToGraph_Callback) {
  Resources r;
  const auto oneKernel = [&] { return LaunchWrite(r.s(), r.out(), 1); };

  SECTION("Identical recapture makes no callback") {
    hipGraph_t graph = Capture(r.s(), oneKernel);
    CallbackState state;
    hipGraphRecaptureCallbackData callback{RecordCallback, &state};
    const EndResult end =
        Recapture(r.s(), graph, &callback, [&] { HIP_CHECK(LaunchWrite(r.s(), r.out(), 1)); });
    RequireEndSuccess(end, graph);
    REQUIRE(state.calls == 0);
    REQUIRE(LaunchAndRead(graph, r.s(), r.out(), 1) == std::vector<int>{1});
    HIP_CHECK(hipGraphDestroy(graph));
  }

  SECTION("No callback data applies the change") {
    hipGraph_t graph = Capture(r.s(), oneKernel);
    const EndResult end =
        Recapture(r.s(), graph, nullptr, [&] { HIP_CHECK(LaunchWrite(r.s(), r.out(), 4)); });
    RequireEndSuccess(end, graph);
    REQUIRE(LaunchAndRead(graph, r.s(), r.out(), 1) == std::vector<int>{4});
    HIP_CHECK(hipGraphDestroy(graph));
  }

  SECTION("Null callbackFunc applies the change") {
    hipGraph_t graph = Capture(r.s(), oneKernel);
    hipGraphRecaptureCallbackData callback{nullptr, nullptr};
    const EndResult end =
        Recapture(r.s(), graph, &callback, [&] { HIP_CHECK(LaunchWrite(r.s(), r.out(), 4)); });
    RequireEndSuccess(end, graph);
    REQUIRE(LaunchAndRead(graph, r.s(), r.out(), 1) == std::vector<int>{4});
    HIP_CHECK(hipGraphDestroy(graph));
  }

  SECTION("Rejected single node keeps its parameters") {
    hipGraph_t graph = Capture(r.s(), oneKernel);
    CallbackState state;
    state.rejectAll = true;
    hipGraphRecaptureCallbackData callback{RecordCallback, &state};
    const EndResult end = Recapture(r.s(), graph, &callback, [&] {
      REQUIRE(LaunchWrite(r.s(), r.out(), 8) == hipErrorGraphRecaptureFailure);
    });
    RequireEndSuccess(end, graph);
    REQUIRE(state.calls == 1);
    REQUIRE(LaunchAndRead(graph, r.s(), r.out(), 1) == std::vector<int>{1});
    HIP_CHECK(hipGraphDestroy(graph));
  }

  SECTION("Rejecting the first of two nodes terminates the recapture") {
    hipGraph_t graph = Capture(r.s(), [&] { return EnqueueChain(r.s(), r.out(), 1, 2); });
    CallbackState state;
    state.rejectCall = 1;
    hipGraphRecaptureCallbackData callback{RecordCallback, &state};
    const EndResult end = Recapture(r.s(), graph, &callback, [&] {
      REQUIRE(LaunchWrite(r.s(), r.out(), 11) == hipErrorGraphRecaptureFailure);
      REQUIRE(LaunchWrite(r.s(), r.out() + 1, 22) == hipErrorGraphRecaptureFailure);
    });
    RequireEndFailure(end);
    REQUIRE(state.calls == 1);
  }

  SECTION("Node added from the callback fails the capturing call after its update") {
    hipGraph_t graph = Capture(r.s(), [&] { return EnqueueChain(r.s(), r.out(), 1, 2); });
    CallbackState state;
    state.onFirstCall = [&] {
      hipGraphNode_t empty = nullptr;
      return hipGraphAddEmptyNode(&empty, graph, nullptr, 0);
    };
    hipGraphRecaptureCallbackData callback{RecordCallback, &state};
    const EndResult end = Recapture(r.s(), graph, &callback, [&] {
      REQUIRE(LaunchWrite(r.s(), r.out(), 3) == hipErrorGraphRecaptureFailure);
      HIP_CHECK(LaunchWrite(r.s(), r.out() + 1, 2));
    });
    REQUIRE(state.onFirstCallResult == hipErrorGraphRecaptureFailure);
    RequireEndSuccess(end, graph);
    REQUIRE(state.calls == 1);
    REQUIRE(GetNodes(graph).size() == 2);
    REQUIRE(LaunchAndRead(graph, r.s(), r.out(), 2) == std::vector<int>{3, 2});
    HIP_CHECK(hipGraphDestroy(graph));
  }

  SECTION("Node destroyed from the callback fails the capturing call after its update") {
    hipGraph_t graph = Capture(r.s(), [&] { return EnqueueChain(r.s(), r.out(), 1, 2); });
    const std::vector<hipGraphNode_t> nodes = GetNodes(graph);
    REQUIRE(nodes.size() == 2);
    CallbackState state;
    state.onFirstCall = [&] { return hipGraphDestroyNode(nodes[1]); };
    hipGraphRecaptureCallbackData callback{RecordCallback, &state};
    const EndResult end = Recapture(r.s(), graph, &callback, [&] {
      REQUIRE(LaunchWrite(r.s(), r.out(), 3) == hipErrorIllegalState);
      HIP_CHECK(LaunchWrite(r.s(), r.out() + 1, 2));
    });
    REQUIRE(state.onFirstCallResult == hipErrorIllegalState);
    RequireEndSuccess(end, graph);
    REQUIRE(state.calls == 1);
    REQUIRE(GetNodes(graph) == nodes);
    REQUIRE(LaunchAndRead(graph, r.s(), r.out(), 2) == std::vector<int>{3, 2});
    HIP_CHECK(hipGraphDestroy(graph));
  }

  SECTION("Parameter edit from the callback is what the next node is compared with") {
    hipGraph_t graph = Capture(r.s(), [&] { return EnqueueChain(r.s(), r.out(), 1, 2); });
    const std::vector<hipGraphNode_t> nodes = GetNodes(graph);
    REQUIRE(nodes.size() == 2);
    int value = 9;
    int* target = r.out() + 1;
    void* args[] = {&target, &value};
    hipKernelNodeParams params{};
    params.func = reinterpret_cast<void*>(WriteValue);
    params.gridDim = dim3(1);
    params.blockDim = dim3(1);
    params.kernelParams = args;
    CallbackState state;
    state.onFirstCall = [&] { return hipGraphKernelNodeSetParams(nodes[1], &params); };
    hipGraphRecaptureCallbackData callback{RecordCallback, &state};
    const EndResult end = Recapture(r.s(), graph, &callback, [&] {
      HIP_CHECK(LaunchWrite(r.s(), r.out(), 3));
      HIP_CHECK(LaunchWrite(r.s(), r.out() + 1, 2));
    });
    HIP_CHECK(state.onFirstCallResult);
    RequireEndSuccess(end, graph);
    REQUIRE(state.calls == 2);
    REQUIRE(state.originalValues == std::vector<int>{1, 9});
    REQUIRE(state.recaptureValues == std::vector<int>{3, 2});
    REQUIRE(LaunchAndRead(graph, r.s(), r.out(), 2) == std::vector<int>{3, 2});
    HIP_CHECK(hipGraphDestroy(graph));
  }
}

/**
 * Test Description
 * ------------------------
 *  - The status passed to the callback for each kind of parameter change, and that the change is
 *    applied. Status 1 is checked against hipGraphExecUpdate itself.
 * Test source
 * ------------------------
 *  - unit/graph/hipStreamBeginRecaptureToGraph.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipStreamBeginRecaptureToGraph_ParameterStatus) {
  Resources r;
  hipStream_t s = r.s();
  int* out = r.out();
  int* scratch = r.scratch.ptr();
  int* host = r.host.host_ptr();
  constexpr auto kEligible = hipGraphRecaptureEligibleForUpdate;
  constexpr auto kIneligible = hipGraphRecaptureIneligibleForUpdate;
  const auto oneKernel = [&] { return LaunchWrite(s, out, 1); };
  const auto oneMemcpy = [&] {
    return hipMemcpyAsync(out, scratch, sizeof(int), hipMemcpyDefault, s);
  };

  SECTION("Kernel argument") {
    RequireChangeStatus(r, oneKernel, [&] { return LaunchWrite(s, out, 2); }, kEligible, {2});
  }
  SECTION("Kernel function") {
    RequireChangeStatus(r, oneKernel, [&] { return LaunchAdd(s, out, 1); }, kEligible, {1});
  }
  SECTION("Kernel grid") {
    RequireChangeStatus(r, oneKernel, [&] { return LaunchWrite(s, out, 1, dim3(2)); }, kEligible,
                        {1});
  }
  SECTION("Kernel block") {
    RequireChangeStatus(r, oneKernel, [&] { return LaunchWrite(s, out, 1, dim3(1), dim3(32)); },
                        kEligible, {1});
  }
  SECTION("Kernel dynamic shared memory") {
    RequireChangeStatus(
        r, oneKernel, [&] { return LaunchWrite(s, out, 1, dim3(1), dim3(1), 256); }, kEligible,
        {1});
  }
  SECTION("Memset value") {
    RequireChangeStatus(
        r, [&] { return hipMemsetAsync(out, 0, sizeof(int), s); },
        [&] { return hipMemsetAsync(out, 1, sizeof(int), s); }, kEligible, {0x01010101});
  }
  SECTION("Memset 1D size") {
    RequireChangeStatus(
        r, [&] { return hipMemsetAsync(out, 1, sizeof(int), s); },
        [&] { return hipMemsetAsync(out, 1, 2 * sizeof(int), s); }, kEligible,
        {0x01010101, 0x01010101});
  }
  SECTION("Memcpy size") {
    RequireChangeStatus(
        r, oneMemcpy, [&] { return hipMemcpyAsync(out, scratch, 2 * sizeof(int), hipMemcpyDefault, s); },
        kEligible, {7, 8});
  }
  SECTION("Memcpy source moved from device to pinned host") {
    RequireChangeStatus(
        r, oneMemcpy, [&] { return hipMemcpyAsync(out, host, sizeof(int), hipMemcpyDefault, s); },
        kEligible, {21});
  }
  SECTION("Memcpy destination moved from device to pinned host") {
    RequireChangeStatus(
        r, oneMemcpy,
        [&] { return hipMemcpyAsync(host, scratch, sizeof(int), hipMemcpyDefault, s); },
        kEligible);
  }
  SECTION("Memcpy kind Default to DeviceToDevice") {
    RequireChangeStatus(
        r, oneMemcpy,
        [&] { return hipMemcpyAsync(out, scratch, sizeof(int), hipMemcpyDeviceToDevice, s); },
        kIneligible);
  }
  SECTION("Memcpy 3D width") {
    RequireChangeStatus(
        r, [&] { return Memcpy3D(s, out, scratch, 16); }, [&] { return Memcpy3D(s, out, scratch, 8); },
        kEligible, {7, 8, 0, 0});
  }
  SECTION("Memset 2D height 2 to 3 rows") {
    RequireChangeStatus(
        r, [&] { return hipMemset2DAsync(out, sizeof(int), 1, sizeof(int), 2, s); },
        [&] { return hipMemset2DAsync(out, sizeof(int), 1, sizeof(int), 3, s); }, kIneligible,
        {0x01010101, 0x01010101, 0x01010101, 0});
  }
  SECTION("Memset 1D to 2D") {
    RequireChangeStatus(
        r, [&] { return hipMemsetAsync(out, 0, sizeof(int), s); },
        [&] { return hipMemset2DAsync(out, 8, 0, sizeof(int), 2, s); }, kIneligible);
  }
  SECTION("Memcpy to symbol offset") {
    RequireChangeStatus(
        r,
        [&] {
          return hipMemcpyToSymbolAsync(HIP_SYMBOL(g_symbol), scratch, sizeof(int), 0,
                                        hipMemcpyDeviceToDevice, s);
        },
        [&] {
          return hipMemcpyToSymbolAsync(HIP_SYMBOL(g_symbol), scratch, sizeof(int), sizeof(int),
                                        hipMemcpyDeviceToDevice, s);
        },
        kEligible);
  }
  SECTION("Memcpy to symbol kind change") {
    RequireChangeStatus(
        r,
        [&] {
          return hipMemcpyToSymbolAsync(HIP_SYMBOL(g_symbol), scratch, sizeof(int), 0,
                                        hipMemcpyDeviceToDevice, s);
        },
        [&] {
          return hipMemcpyToSymbolAsync(HIP_SYMBOL(g_symbol), host, sizeof(int), 0,
                                        hipMemcpyHostToDevice, s);
        },
        kIneligible);
  }
  SECTION("Memcpy from symbol destination") {
    RequireChangeStatus(
        r,
        [&] {
          return hipMemcpyFromSymbolAsync(out, HIP_SYMBOL(g_symbol), sizeof(int), 0,
                                          hipMemcpyDeviceToDevice, s);
        },
        [&] {
          return hipMemcpyFromSymbolAsync(out + 1, HIP_SYMBOL(g_symbol), sizeof(int), 0,
                                          hipMemcpyDeviceToDevice, s);
        },
        kEligible);
  }
  SECTION("Host node user data") {
    int a = 0, b = 0;
    RequireChangeStatus(
        r, [&] { return hipLaunchHostFunc(s, HostNoop, &a); },
        [&] { return hipLaunchHostFunc(s, HostNoop, &b); }, kEligible);
  }
  SECTION("Host node function") {
    int a = 0;
    RequireChangeStatus(
        r, [&] { return hipLaunchHostFunc(s, HostNoop, &a); },
        [&] { return hipLaunchHostFunc(s, OtherHostNoop, &a); }, kEligible);
  }
  SECTION("Event record node event") {
    RequireChangeStatus(
        r, [&] { return hipEventRecordWithFlags(r.eventA(), s, hipEventRecordExternal); },
        [&] { return hipEventRecordWithFlags(r.eventB(), s, hipEventRecordExternal); },
        kEligible);
  }
  SECTION("Event wait node event") {
    RequireChangeStatus(
        r, [&] { return hipStreamWaitEvent(s, r.eventA(), hipEventWaitExternal); },
        [&] { return hipStreamWaitEvent(s, r.eventB(), hipEventWaitExternal); }, kEligible);
  }
}

/**
 * Test Description
 * ------------------------
 *  - A status-1 change (memcpy kind) is accepted and applied. hipGraphExecUpdate on an exec
 *    instantiated before the recapture then fails with ParametersChanged, and a fresh
 *    instantiate of the updated graph works.
 * Test source
 * ------------------------
 *  - unit/graph/hipStreamBeginRecaptureToGraph.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipStreamBeginRecaptureToGraph_IneligibleChangeThenExecUpdate) {
  Resources r;
  int* scratch = r.scratch.ptr();
  hipGraph_t graph = Capture(r.s(), [&] {
    return hipMemcpyAsync(r.out(), scratch, sizeof(int), hipMemcpyDefault, r.s());
  });
  hipGraphExec_t exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  CallbackState state;
  hipGraphRecaptureCallbackData callback{RecordCallback, &state};
  const EndResult end = Recapture(r.s(), graph, &callback, [&] {
    HIP_CHECK(hipMemcpyAsync(r.out(), scratch, sizeof(int), hipMemcpyDeviceToDevice, r.s()));
  });
  RequireEndSuccess(end, graph);
  REQUIRE(state.calls == 1);
  REQUIRE(state.statuses[0] == hipGraphRecaptureIneligibleForUpdate);

  hipGraphNode_t errorNode = nullptr;
  hipGraphExecUpdateResult updateResult = hipGraphExecUpdateSuccess;
  REQUIRE(hipGraphExecUpdate(exec, graph, &errorNode, &updateResult) ==
          hipErrorGraphExecUpdateFailure);
  REQUIRE(updateResult == hipGraphExecUpdateErrorParametersChanged);
  HIP_CHECK(hipGraphExecDestroy(exec));

  REQUIRE(LaunchAndRead(graph, r.s(), r.out(), 1) == std::vector<int>{7});
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *  - Positional matching: soft failures (type mismatch, dependency mismatch, extra node) fail
 *    only that call and a later matching call continues; missing nodes and topology changes
 *    make end fail; two independent nodes recaptured in swapped order are compared crosswise.
 * Test source
 * ------------------------
 *  - unit/graph/hipStreamBeginRecaptureToGraph.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipStreamBeginRecaptureToGraph_Matching) {
  Resources r;
  hipStream_t s = r.s();
  int* out = r.out();
  const auto chain = [&] { return EnqueueChain(s, out, 1, 2); };
  const auto forked = [&] { return EnqueueForked(r, 10, 20, false); };
  CallbackState state;
  hipGraphRecaptureCallbackData callback{RecordCallback, &state};

  SECTION("Type mismatch on the only node") {
    hipGraph_t graph = Capture(s, [&] { return LaunchWrite(s, out, 1); });
    const EndResult end = Recapture(s, graph, &callback, [&] {
      REQUIRE(hipMemsetAsync(out, 0, sizeof(int), s) == hipErrorGraphRecaptureFailure);
    });
    RequireEndFailure(end);
    REQUIRE(state.calls == 0);
  }

  SECTION("Type mismatch then retried") {
    hipGraph_t graph = Capture(s, chain);
    const EndResult end = Recapture(s, graph, &callback, [&] {
      HIP_CHECK(LaunchWrite(s, out, 11));
      REQUIRE(hipMemsetAsync(out + 1, 0, sizeof(int), s) == hipErrorGraphRecaptureFailure);
      HIP_CHECK(LaunchWrite(s, out + 1, 22));
    });
    RequireEndSuccess(end, graph);
    REQUIRE(state.calls == 2);
    REQUIRE(LaunchAndRead(graph, s, out, 2) == std::vector<int>{11, 22});
    HIP_CHECK(hipGraphDestroy(graph));
  }

  SECTION("Extra node") {
    hipGraph_t graph = Capture(s, [&] { return LaunchWrite(s, out, 1); });
    const EndResult end = Recapture(s, graph, nullptr, [&] {
      HIP_CHECK(LaunchWrite(s, out, 5));
      REQUIRE(LaunchWrite(s, out, 6) == hipErrorGraphRecaptureFailure);
    });
    RequireEndSuccess(end, graph);
    REQUIRE(GetNodes(graph).size() == 1);
    REQUIRE(LaunchAndRead(graph, s, out, 1) == std::vector<int>{5});
    HIP_CHECK(hipGraphDestroy(graph));
  }

  SECTION("Missing node") {
    hipGraph_t graph = Capture(s, chain);
    const EndResult end =
        Recapture(s, graph, nullptr, [&] { HIP_CHECK(LaunchWrite(s, out, 3)); });
    RequireEndFailure(end);
  }

  SECTION("Dependency mismatch then retried with the right dependency") {
    hipGraph_t graph = Capture(s, chain);
    const EndResult end = Recapture(s, graph, &callback, [&] {
      hipError_t resultB = hipSuccess;
      HIP_CHECK(hipEventRecord(r.fork(), s));
      HIP_CHECK(hipStreamWaitEvent(r.sideStream(), r.fork(), 0));
      HIP_CHECK(LaunchWrite(s, out, 11));
      resultB = LaunchWrite(r.sideStream(), out + 1, 22);
      HIP_CHECK(hipEventRecord(r.join(), r.sideStream()));
      HIP_CHECK(hipStreamWaitEvent(s, r.join(), 0));
      REQUIRE(resultB == hipErrorGraphRecaptureFailure);
      HIP_CHECK(LaunchWrite(s, out + 1, 22));
    });
    RequireEndSuccess(end, graph);
    REQUIRE(LaunchAndRead(graph, s, out, 2) == std::vector<int>{11, 22});
    HIP_CHECK(hipGraphDestroy(graph));
  }

  SECTION("Independent nodes recaptured in swapped order") {
    hipGraph_t graph = Capture(s, forked);
    const EndResult end =
        Recapture(s, graph, &callback, [&] { HIP_CHECK(EnqueueForked(r, 10, 20, true)); });
    RequireEndSuccess(end, graph);
    REQUIRE(state.calls == 2);
    REQUIRE(state.originalValues == std::vector<int>{10, 20});
    REQUIRE(state.recaptureValues == std::vector<int>{20, 10});
    REQUIRE(LaunchAndRead(graph, s, out, 2) == std::vector<int>{10, 20});
    HIP_CHECK(hipGraphDestroy(graph));
  }

  SECTION("Dependency removed") {
    hipGraph_t graph = Capture(s, [&] { return EnqueueChain(s, out, 10, 20); });
    const EndResult end = Recapture(s, graph, &callback, [&] {
      hipError_t resultB = hipSuccess;
      REQUIRE(EnqueueForked(r, 10, 20, false, &resultB) == hipErrorGraphRecaptureFailure);
      REQUIRE(resultB == hipErrorGraphRecaptureFailure);
    });
    RequireEndFailure(end);
  }

  SECTION("Dependency added") {
    hipGraph_t graph = Capture(s, forked);
    const EndResult end = Recapture(s, graph, &callback, [&] {
      HIP_CHECK(LaunchWrite(s, out, 10));
      REQUIRE(LaunchWrite(s, out + 1, 20) == hipErrorGraphRecaptureFailure);
    });
    RequireEndFailure(end);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Allocation and free nodes: a matching hipMallocAsync returns the original address, the
 *    allocation can be read by a kernel, and a size change, a swapped free order or a
 *    free of another pointer are status 2 and fail with 918. A null free is a no-op.
 * Test source
 * ------------------------
 *  - unit/graph/hipStreamBeginRecaptureToGraph.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipStreamBeginRecaptureToGraph_AllocFree) {
  Resources r;
  hipStream_t s = r.s();
  int* out = r.out();
  void* original = nullptr;
  void* recaptured = nullptr;
  CallbackState state;
  hipGraphRecaptureCallbackData callback{RecordCallback, &state};

  SECTION("Same address, copied out by a kernel") {
    hipGraph_t graph =
        Capture(s, [&] { return EnqueueAllocChain(s, out, kAllocBytes, 1, &original); });
    const EndResult end = Recapture(s, graph, &callback, [&] {
      HIP_CHECK(EnqueueAllocChain(s, out, kAllocBytes, 2, &recaptured));
    });
    RequireEndSuccess(end, graph);
    REQUIRE(recaptured == original);
    REQUIRE(state.calls == 1);
    REQUIRE(LaunchAndRead(graph, s, out, 1) == std::vector<int>{2});
    HIP_CHECK(hipGraphDestroy(graph));
  }

  SECTION("Identical recapture makes no callback") {
    hipGraph_t graph =
        Capture(s, [&] { return EnqueueAllocChain(s, out, kAllocBytes, 3, &original); });
    const EndResult end = Recapture(s, graph, &callback, [&] {
      HIP_CHECK(EnqueueAllocChain(s, out, kAllocBytes, 3, &recaptured));
    });
    RequireEndSuccess(end, graph);
    REQUIRE(state.calls == 0);
    REQUIRE(LaunchAndRead(graph, s, out, 1) == std::vector<int>{3});
    HIP_CHECK(hipGraphDestroy(graph));
  }

  SECTION("Size change is status 2") {
    hipGraph_t graph =
        Capture(s, [&] { return EnqueueAllocChain(s, out, kAllocBytes, 1, &original); });
    const EndResult end = Recapture(s, graph, &callback, [&] {
      REQUIRE(hipMallocAsync(&recaptured, 2 * kAllocBytes, s) == hipErrorGraphRecaptureFailure);
      REQUIRE(recaptured == nullptr);
    });
    RequireEndFailure(end);
    REQUIRE(state.calls == 1);
    REQUIRE(state.statuses[0] == hipGraphRecaptureError);
  }

  SECTION("Size change without a callback") {
    hipGraph_t graph =
        Capture(s, [&] { return EnqueueAllocChain(s, out, kAllocBytes, 1, &original); });
    const EndResult end = Recapture(s, graph, nullptr, [&] {
      REQUIRE(hipMallocAsync(&recaptured, 2 * kAllocBytes, s) == hipErrorGraphRecaptureFailure);
    });
    RequireEndFailure(end);
  }

  SECTION("Free order swapped is status 2") {
    void* first = nullptr;
    void* second = nullptr;
    hipGraph_t graph = Capture(s, [&] {
      HIP_CHECK(hipMallocAsync(&first, kAllocBytes, s));
      HIP_CHECK(hipMallocAsync(&second, kAllocBytes, s));
      HIP_CHECK(hipFreeAsync(first, s));
      return hipFreeAsync(second, s);
    });
    const EndResult end = Recapture(s, graph, &callback, [&] {
      void* firstAgain = nullptr;
      void* secondAgain = nullptr;
      HIP_CHECK(hipMallocAsync(&firstAgain, kAllocBytes, s));
      HIP_CHECK(hipMallocAsync(&secondAgain, kAllocBytes, s));
      REQUIRE(firstAgain == first);
      REQUIRE(secondAgain == second);
      REQUIRE(hipFreeAsync(secondAgain, s) == hipErrorGraphRecaptureFailure);
      REQUIRE(hipFreeAsync(firstAgain, s) == hipErrorGraphRecaptureFailure);
    });
    RequireEndFailure(end);
    REQUIRE(state.calls == 1);
    REQUIRE(state.statuses[0] == hipGraphRecaptureError);
  }

  SECTION("Free of a hipMalloc pointer is status 2 and is not validated") {
    hipGraph_t graph =
        Capture(s, [&] { return EnqueueAllocChain(s, out, kAllocBytes, 1, &original); });
    const EndResult end = Recapture(s, graph, &callback, [&] {
      HIP_CHECK(hipMallocAsync(&recaptured, kAllocBytes, s));
      HIP_CHECK(LaunchWrite(s, static_cast<int*>(recaptured), 1));
      HIP_CHECK(LaunchCopy(s, out, static_cast<int*>(recaptured)));
      REQUIRE(hipFreeAsync(r.scratch.ptr(), s) == hipErrorGraphRecaptureFailure);
      REQUIRE(hipFreeAsync(recaptured, s) == hipErrorGraphRecaptureFailure);
    });
    RequireEndFailure(end);
    REQUIRE(state.calls == 1);
    REQUIRE(state.statuses[0] == hipGraphRecaptureError);
  }

  SECTION("Null free is a no-op") {
    hipGraph_t graph =
        Capture(s, [&] { return EnqueueAllocChain(s, out, kAllocBytes, 1, &original); });
    const EndResult end = Recapture(s, graph, &callback, [&] {
      HIP_CHECK(hipMallocAsync(&recaptured, kAllocBytes, s));
      HIP_CHECK(LaunchWrite(s, static_cast<int*>(recaptured), 4));
      HIP_CHECK(LaunchCopy(s, out, static_cast<int*>(recaptured)));
      HIP_CHECK(hipFreeAsync(nullptr, s));
      HIP_CHECK(hipFreeAsync(recaptured, s));
    });
    RequireEndSuccess(end, graph);
    REQUIRE(state.calls == 1);
    REQUIRE(LaunchAndRead(graph, s, out, 1) == std::vector<int>{4});
    HIP_CHECK(hipGraphDestroy(graph));
  }
}

/**
 * Test Description
 * ------------------------
 *  - Launch attributes are not parameters: priority (taken from the recapturing stream) and
 *    the cooperative launch mode are applied to the original node without a callback. The node
 *    then reads back like a fresh capture of the same launch. Because HIP's hipGraphExecUpdate
 *    rejects a cooperative-mode change, an older exec can no longer be updated (design Q2).
 * Test source
 * ------------------------
 *  - unit/graph/hipStreamBeginRecaptureToGraph.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipStreamBeginRecaptureToGraph_LaunchAttributes) {
  Resources r;
  CallbackState state;
  hipGraphRecaptureCallbackData callback{RecordCallback, &state};

  SECTION("Priority follows the recapturing stream") {
    int least = 0, greatest = 0;
    HIP_CHECK(hipDeviceGetStreamPriorityRange(&least, &greatest));
    if (least == greatest) {
      HIP_SKIP_TEST("Device does not support stream priority");
    }
    StreamGuard high(Streams::withPriority, hipStreamDefault, greatest);
    hipGraph_t graph = Capture(r.s(), [&] { return LaunchWrite(r.s(), r.out(), 1); });
    const hipGraphNode_t node = GetNodes(graph).at(0);
    hipKernelNodeAttrValue value{};
    HIP_CHECK(hipGraphKernelNodeGetAttribute(node, hipLaunchAttributePriority, &value));
    REQUIRE(value.priority != greatest);

    const EndResult end = Recapture(high.stream(), graph, &callback, [&] {
      HIP_CHECK(LaunchWrite(high.stream(), r.out(), 1));
    });
    RequireEndSuccess(end, graph);
    REQUIRE(state.calls == 0);
    HIP_CHECK(hipGraphKernelNodeGetAttribute(node, hipLaunchAttributePriority, &value));
    REQUIRE(value.priority == greatest);
    REQUIRE(LaunchAndRead(graph, r.s(), r.out(), 1) == std::vector<int>{1});
    HIP_CHECK(hipGraphDestroy(graph));
  }

  SECTION("Cooperative attribute") {
    int cooperative = 0;
    HIP_CHECK(hipDeviceGetAttribute(&cooperative, hipDeviceAttributeCooperativeLaunch, 0));
    if (cooperative == 0) {
      HIP_SKIP_TEST("Device does not support cooperative launch");
    }
    hipLaunchAttribute attr{};
    attr.id = hipLaunchAttributeCooperative;
    attr.val.cooperative = 1;
    hipGraph_t reference =
        Capture(r.s(), [&] { return LaunchWriteEx(r.s(), r.out(), 1, &attr, 1); });
    hipKernelNodeAttrValue expected{};
    HIP_CHECK(hipGraphKernelNodeGetAttribute(GetNodes(reference).at(0),
                                             hipLaunchAttributeCooperative, &expected));
    HIP_CHECK(hipGraphDestroy(reference));

    hipGraph_t graph =
        Capture(r.s(), [&] { return LaunchWriteEx(r.s(), r.out(), 1, nullptr, 0); });
    const hipGraphNode_t node = GetNodes(graph).at(0);
    hipGraphExec_t exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    const EndResult end = Recapture(r.s(), graph, &callback, [&] {
      HIP_CHECK(LaunchWriteEx(r.s(), r.out(), 1, &attr, 1));
    });
    RequireEndSuccess(end, graph);
    REQUIRE(state.calls == 0);
    hipKernelNodeAttrValue value{};
    HIP_CHECK(hipGraphKernelNodeGetAttribute(node, hipLaunchAttributeCooperative, &value));
    REQUIRE(value.cooperative == expected.cooperative);
    // The cooperative launch mode was applied, and hipGraphExecUpdate rejects that change.
    hipGraphNode_t errorNode = nullptr;
    hipGraphExecUpdateResult updateResult = hipGraphExecUpdateSuccess;
    REQUIRE(hipGraphExecUpdate(exec, graph, &errorNode, &updateResult) ==
            hipErrorGraphExecUpdateFailure);
    HIP_CHECK(hipGraphExecDestroy(exec));
    REQUIRE(LaunchAndRead(graph, r.s(), r.out(), 1) == std::vector<int>{1});
    HIP_CHECK(hipGraphDestroy(graph));
  }
}

/**
 * Test Description
 * ------------------------
 *  - Graph sources: a manually built graph, a clone (recaptured independently of its source),
 *    and repeated recaptures that each compare against the parameters left by the previous one.
 * Test source
 * ------------------------
 *  - unit/graph/hipStreamBeginRecaptureToGraph.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipStreamBeginRecaptureToGraph_GraphSources) {
  Resources r;
  CallbackState state;
  hipGraphRecaptureCallbackData callback{RecordCallback, &state};

  SECTION("Manual graph") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipGraphCreate(&graph, 0));
    const hipGraphNode_t node = AddKernelNode(graph, nullptr, 0, r.out(), 1);
    const EndResult end = Recapture(r.s(), graph, &callback,
                                    [&] { HIP_CHECK(LaunchWrite(r.s(), r.out(), 2)); });
    RequireEndSuccess(end, graph);
    REQUIRE(state.calls == 1);
    REQUIRE(state.nodes[0] == node);
    REQUIRE(LaunchAndRead(graph, r.s(), r.out(), 1) == std::vector<int>{2});
    HIP_CHECK(hipGraphDestroy(graph));
  }

  SECTION("Clone") {
    hipGraph_t graph = Capture(r.s(), [&] { return EnqueueChain(r.s(), r.out(), 1, 2); });
    hipGraph_t clone = nullptr;
    HIP_CHECK(hipGraphClone(&clone, graph));
    const EndResult end = Recapture(r.s(), clone, &callback,
                                    [&] { HIP_CHECK(EnqueueChain(r.s(), r.out(), 11, 22)); });
    RequireEndSuccess(end, clone);
    REQUIRE(state.calls == 2);
    REQUIRE(LaunchAndRead(clone, r.s(), r.out(), 2) == std::vector<int>{11, 22});
    REQUIRE(LaunchAndRead(graph, r.s(), r.out(), 2) == std::vector<int>{1, 2});
    HIP_CHECK(hipGraphDestroy(clone));
    HIP_CHECK(hipGraphDestroy(graph));
  }

  SECTION("Repeated recapture") {
    hipGraph_t graph = Capture(r.s(), [&] { return LaunchWrite(r.s(), r.out(), 1); });
    CallbackState reject;
    reject.rejectAll = true;
    hipGraphRecaptureCallbackData rejectCallback{RecordCallback, &reject};
    struct Step {
      int value;
      hipGraphRecaptureCallbackData* callback;
      int expected;
    };
    const Step steps[] = {
        {2, &callback, 2}, {3, &callback, 3}, {4, &rejectCallback, 3}, {5, &callback, 5}};
    for (const Step& step : steps) {
      const EndResult end = Recapture(r.s(), graph, step.callback,
                                      [&] { (void)LaunchWrite(r.s(), r.out(), step.value); });
      RequireEndSuccess(end, graph);
      REQUIRE(LaunchAndRead(graph, r.s(), r.out(), 1) == std::vector<int>{step.expected});
    }
    REQUIRE(state.calls == 3);
    REQUIRE(reject.calls == 1);
    HIP_CHECK(hipGraphDestroy(graph));
  }
}

/**
 * Test Description
 * ------------------------
 *  - Recapture into the graph embedded in a child graph node. Success updates the embedded
 *    graph and the parent picks up the change.
 * Test source
 * ------------------------
 *  - unit/graph/hipStreamBeginRecaptureToGraph.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipStreamBeginRecaptureToGraph_EmbeddedGraph) {
  Resources r;
  CallbackState state;
  hipGraphRecaptureCallbackData callback{RecordCallback, &state};

  SECTION("Success updates the parent") {
    ChildGraph child(r.out(), {1});
    const EndResult end = Recapture(r.s(), child.embedded, &callback,
                                    [&] { HIP_CHECK(LaunchWrite(r.s(), r.out(), 2)); });
    RequireEndSuccess(end, child.embedded);
    REQUIRE(state.calls == 1);
    REQUIRE(LaunchAndRead(child.parent, r.s(), r.out(), 1) == std::vector<int>{2});
  }

}

/**
 * Test Description
 * ------------------------
 *  - Begin releases the user objects the graph owns. A user object retained on the graph
 *    during the recapture is kept until the graph is destroyed.
 * Test source
 * ------------------------
 *  - unit/graph/hipStreamBeginRecaptureToGraph.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipStreamBeginRecaptureToGraph_UserObjects) {
  Resources r;
  hipGraph_t graph = Capture(r.s(), [&] { return LaunchWrite(r.s(), r.out(), 1); });
  g_userObjectsDestroyed = 0;
  hipUserObject_t before = nullptr;
  HIP_CHECK(hipUserObjectCreate(&before, nullptr, DestroyUserObject, 1,
                                hipUserObjectNoDestructorSync));
  HIP_CHECK(hipGraphRetainUserObject(graph, before, 1, hipGraphUserObjectMove));

  HIP_CHECK(hipStreamBeginRecaptureToGraph(r.s(), kMode, graph, nullptr));
  REQUIRE(WaitForUserObjectsDestroyed(1));
  hipUserObject_t during = nullptr;
  HIP_CHECK(hipUserObjectCreate(&during, nullptr, DestroyUserObject, 1,
                                hipUserObjectNoDestructorSync));
  HIP_CHECK(hipGraphRetainUserObject(graph, during, 1, hipGraphUserObjectMove));
  HIP_CHECK(LaunchWrite(r.s(), r.out(), 1));
  hipGraph_t returned = nullptr;
  HIP_CHECK(hipStreamEndCapture(r.s(), &returned));
  REQUIRE(returned == graph);
  REQUIRE(g_userObjectsDestroyed.load() == 1);

  HIP_CHECK(hipGraphDestroy(graph));
  REQUIRE(WaitForUserObjectsDestroyed(2));
}

/**
 * Test Description
 * ------------------------
 *  - Graphs that cannot be recaptured: a manual empty node, a creation order that is not a
 *    dependency order. Adding a node directly during recapture fails with 918 and adds
 *    nothing; the recapture continues and end returns the graph unchanged.
 *  - Other direct edits: dependency edits succeed and the next node no longer matches;
 *    hipGraphDestroyNode returns hipErrorIllegalState and the node stays; a parameter edit
 *    succeeds and recapture compares against the new values.
 * Test source
 * ------------------------
 *  - unit/graph/hipStreamBeginRecaptureToGraph.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipStreamBeginRecaptureToGraph_Unrecapturable) {
  Resources r;
  hipStream_t s = r.s();
  int* out = r.out();

  SECTION("Manual empty node is never matched") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipGraphCreate(&graph, 0));
    hipGraphNode_t a = AddKernelNode(graph, nullptr, 0, out, 1);
    hipGraphNode_t empty = nullptr;
    HIP_CHECK(hipGraphAddEmptyNode(&empty, graph, &a, 1));
    const EndResult end = Recapture(s, graph, nullptr, [&] { HIP_CHECK(LaunchWrite(s, out, 1)); });
    RequireEndFailure(end);
  }

  SECTION("Creation order differs from dependency order") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipGraphCreate(&graph, 0));
    hipGraphNode_t b = AddKernelNode(graph, nullptr, 0, out + 1, 2);
    hipGraphNode_t a = AddKernelNode(graph, nullptr, 0, out, 1);
    HIP_CHECK(hipGraphAddDependencies(graph, &a, &b, 1));
    const EndResult end =
        Recapture(s, graph, nullptr, [&] { (void)EnqueueChain(s, out, 1, 2); });
    RequireEndFailure(end);
  }

  // Each add runs after the first original node matched. It fails with 918 and adds nothing,
  // and the recapture continues, so the second original node still matches.
  const auto requireAddRejected = [&](auto add) {
    hipGraph_t graph = Capture(s, [&] { return EnqueueChain(s, out, 1, 2); });
    const std::vector<hipGraphNode_t> nodes = GetNodes(graph);
    REQUIRE(nodes.size() == 2);
    const EndResult end = Recapture(s, graph, nullptr, [&] {
      HIP_CHECK(LaunchWrite(s, out, 1));
      REQUIRE(add(graph, nodes) == hipErrorGraphRecaptureFailure);
      HIP_CHECK(LaunchWrite(s, out + 1, 2));
    });
    RequireEndSuccess(end, graph);
    REQUIRE(GetNodes(graph) == nodes);
    REQUIRE(LaunchAndRead(graph, s, out, 2) == std::vector<int>{1, 2});
    HIP_CHECK(hipGraphDestroy(graph));
  };
  using Nodes = std::vector<hipGraphNode_t>;

  SECTION("hipGraphAddEmptyNode") {
    requireAddRejected([&](hipGraph_t g, const Nodes&) {
      hipGraphNode_t node = nullptr;
      return hipGraphAddEmptyNode(&node, g, nullptr, 0);
    });
  }
  SECTION("hipGraphAddMemsetNode") {
    requireAddRejected([&](hipGraph_t g, const Nodes&) {
      hipMemsetParams params{};
      params.dst = out;
      params.elementSize = sizeof(int);
      params.width = 1;
      params.height = 1;
      hipGraphNode_t node = nullptr;
      return hipGraphAddMemsetNode(&node, g, nullptr, 0, &params);
    });
  }
  SECTION("hipGraphAddMemcpyNode1D") {
    requireAddRejected([&](hipGraph_t g, const Nodes&) {
      hipGraphNode_t node = nullptr;
      return hipGraphAddMemcpyNode1D(&node, g, nullptr, 0, out, r.scratch.ptr(), sizeof(int),
                                     hipMemcpyDeviceToDevice);
    });
  }
  SECTION("hipGraphAddMemcpyNode") {
    requireAddRejected([&](hipGraph_t g, const Nodes&) {
      hipMemcpy3DParms params{};
      params.srcPtr = make_hipPitchedPtr(r.scratch.ptr(), kBytes, sizeof(int), 1);
      params.dstPtr = make_hipPitchedPtr(out, kBytes, sizeof(int), 1);
      params.extent = make_hipExtent(sizeof(int), 1, 1);
      params.kind = hipMemcpyDeviceToDevice;
      hipGraphNode_t node = nullptr;
      return hipGraphAddMemcpyNode(&node, g, nullptr, 0, &params);
    });
  }
  SECTION("hipGraphAddHostNode") {
    requireAddRejected([&](hipGraph_t g, const Nodes&) {
      hipHostNodeParams params{HostNoop, nullptr};
      hipGraphNode_t node = nullptr;
      return hipGraphAddHostNode(&node, g, nullptr, 0, &params);
    });
  }
  SECTION("hipGraphAddChildGraphNode") {
    hipGraph_t child = nullptr;
    HIP_CHECK(hipGraphCreate(&child, 0));
    requireAddRejected([&](hipGraph_t g, const Nodes&) {
      hipGraphNode_t node = nullptr;
      return hipGraphAddChildGraphNode(&node, g, nullptr, 0, child);
    });
    HIP_CHECK(hipGraphDestroy(child));
  }
  SECTION("hipGraphAddEventRecordNode") {
    requireAddRejected([&](hipGraph_t g, const Nodes&) {
      hipGraphNode_t node = nullptr;
      return hipGraphAddEventRecordNode(&node, g, nullptr, 0, r.eventA());
    });
  }
  SECTION("hipGraphAddEventWaitNode") {
    requireAddRejected([&](hipGraph_t g, const Nodes&) {
      hipGraphNode_t node = nullptr;
      return hipGraphAddEventWaitNode(&node, g, nullptr, 0, r.eventA());
    });
  }
  SECTION("hipGraphAddMemAllocNode") {
    requireAddRejected([&](hipGraph_t g, const Nodes&) {
      hipMemAllocNodeParams params{};
      params.poolProps.allocType = hipMemAllocationTypePinned;
      params.poolProps.location.type = hipMemLocationTypeDevice;
      params.poolProps.location.id = 0;
      params.bytesize = kAllocBytes;
      hipGraphNode_t node = nullptr;
      return hipGraphAddMemAllocNode(&node, g, nullptr, 0, &params);
    });
  }
  SECTION("hipGraphAddMemFreeNode") {
    // The free must name a live graph allocation, or validation fails before the guard runs.
    hipGraph_t other = nullptr;
    HIP_CHECK(hipGraphCreate(&other, 0));
    hipMemAllocNodeParams allocParams{};
    allocParams.poolProps.allocType = hipMemAllocationTypePinned;
    allocParams.poolProps.location.type = hipMemLocationTypeDevice;
    allocParams.poolProps.location.id = 0;
    allocParams.bytesize = kAllocBytes;
    hipGraphNode_t allocNode = nullptr;
    HIP_CHECK(hipGraphAddMemAllocNode(&allocNode, other, nullptr, 0, &allocParams));
    requireAddRejected([&](hipGraph_t g, const Nodes&) {
      hipGraphNode_t node = nullptr;
      return hipGraphAddMemFreeNode(&node, g, nullptr, 0, allocParams.dptr);
    });
    // The rejected free added nothing, so the allocation can still be freed where it belongs.
    hipGraphNode_t freeNode = nullptr;
    HIP_CHECK(hipGraphAddMemFreeNode(&freeNode, other, &allocNode, 1, allocParams.dptr));
    HIP_CHECK(hipGraphDestroy(other));
  }
  SECTION("hipGraphAddNode") {
    requireAddRejected([&](hipGraph_t g, const Nodes&) {
      int value = 3;
      void* args[] = {&out, &value};
      hipGraphNodeParams params{};
      params.type = hipGraphNodeTypeKernel;
      params.kernel.func = reinterpret_cast<void*>(WriteValue);
      params.kernel.gridDim = dim3(1);
      params.kernel.blockDim = dim3(1);
      params.kernel.kernelParams = args;
      hipGraphNode_t node = nullptr;
      return hipGraphAddNode(&node, g, nullptr, 0, &params);
    });
  }
  SECTION("hipGraphAddDependencies") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipGraphCreate(&graph, 0));
    hipGraphNode_t a = AddKernelNode(graph, nullptr, 0, out, 1);
    hipGraphNode_t b = AddKernelNode(graph, nullptr, 0, out + 1, 2);
    const EndResult end = Recapture(s, graph, nullptr, [&] {
      HIP_CHECK(hipEventRecord(r.fork(), s));
      HIP_CHECK(hipStreamWaitEvent(r.sideStream(), r.fork(), 0));
      HIP_CHECK(LaunchWrite(s, out, 1));
      // The edge is added, so b now depends on a and the side-stream launch no longer matches.
      HIP_CHECK(hipGraphAddDependencies(graph, &a, &b, 1));
      REQUIRE(LaunchWrite(r.sideStream(), out + 1, 2) == hipErrorGraphRecaptureFailure);
      HIP_CHECK(hipEventRecord(r.join(), r.sideStream()));
      HIP_CHECK(hipStreamWaitEvent(s, r.join(), 0));
    });
    RequireEndFailure(end);
  }
  SECTION("hipGraphRemoveDependencies") {
    hipGraph_t graph = Capture(s, [&] { return EnqueueChain(s, out, 1, 2); });
    const std::vector<hipGraphNode_t> nodes = GetNodes(graph);
    REQUIRE(nodes.size() == 2);
    const EndResult end = Recapture(s, graph, nullptr, [&] {
      HIP_CHECK(LaunchWrite(s, out, 1));
      // The edge is removed, so the second launch, which depends on the first, no longer matches.
      HIP_CHECK(hipGraphRemoveDependencies(graph, &nodes[0], &nodes[1], 1));
      REQUIRE(LaunchWrite(s, out + 1, 2) == hipErrorGraphRecaptureFailure);
    });
    RequireEndFailure(end);
  }
  SECTION("hipGraphDestroyNode") {
    hipGraph_t graph = Capture(s, [&] { return EnqueueChain(s, out, 1, 2); });
    const std::vector<hipGraphNode_t> nodes = GetNodes(graph);
    REQUIRE(nodes.size() == 2);
    const EndResult end = Recapture(s, graph, nullptr, [&] {
      HIP_CHECK(LaunchWrite(s, out, 1));
      REQUIRE(hipGraphDestroyNode(nodes[1]) == hipErrorIllegalState);
      HIP_CHECK(LaunchWrite(s, out + 1, 2));
    });
    RequireEndSuccess(end, graph);
    REQUIRE(GetNodes(graph) == nodes);
    REQUIRE(LaunchAndRead(graph, s, out, 2) == std::vector<int>{1, 2});
    HIP_CHECK(hipGraphDestroy(graph));
  }
  SECTION("hipGraphKernelNodeSetParams") {
    hipGraph_t graph = Capture(s, [&] { return EnqueueChain(s, out, 1, 2); });
    const std::vector<hipGraphNode_t> nodes = GetNodes(graph);
    REQUIRE(nodes.size() == 2);
    const EndResult end = Recapture(s, graph, nullptr, [&] {
      HIP_CHECK(LaunchWrite(s, out, 1));
      int value = 9;
      int* target = out + 1;
      void* args[] = {&target, &value};
      hipKernelNodeParams params{};
      params.func = reinterpret_cast<void*>(WriteValue);
      params.gridDim = dim3(1);
      params.blockDim = dim3(1);
      params.kernelParams = args;
      HIP_CHECK(hipGraphKernelNodeSetParams(nodes[1], &params));
      // The recaptured value 2 differs from the edited 9; with no callback it is applied.
      HIP_CHECK(LaunchWrite(s, out + 1, 2));
    });
    RequireEndSuccess(end, graph);
    REQUIRE(LaunchAndRead(graph, s, out, 2) == std::vector<int>{1, 2});
    HIP_CHECK(hipGraphDestroy(graph));
  }
}

/**
 * Test Description
 * ------------------------
 *  - hipStreamGetCaptureInfo_v2 during recapture reports the input graph and, as dependencies,
 *    the original node handles of the matched nodes.
 * Test source
 * ------------------------
 *  - unit/graph/hipStreamBeginRecaptureToGraph.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipStreamBeginRecaptureToGraph_CaptureInfo) {
  Resources r;
  hipGraph_t graph = Capture(r.s(), [&] { return EnqueueChain(r.s(), r.out(), 1, 2); });
  const std::vector<hipGraphNode_t> nodes = GetNodes(graph);
  REQUIRE(nodes.size() == 2);

  const auto requireInfo = [&](std::vector<hipGraphNode_t> expected) {
    hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
    unsigned long long id = 0;
    hipGraph_t captureGraph = nullptr;
    const hipGraphNode_t* deps = nullptr;
    size_t count = 0;
    HIP_CHECK(hipStreamGetCaptureInfo_v2(r.s(), &status, &id, &captureGraph, &deps, &count));
    REQUIRE(status == hipStreamCaptureStatusActive);
    REQUIRE(captureGraph == graph);
    REQUIRE(std::vector<hipGraphNode_t>(deps, deps + count) == expected);
  };

  HIP_CHECK(hipStreamBeginRecaptureToGraph(r.s(), kMode, graph, nullptr));
  requireInfo({});
  HIP_CHECK(LaunchWrite(r.s(), r.out(), 11));
  requireInfo({nodes[0]});
  HIP_CHECK(LaunchWrite(r.s(), r.out() + 1, 22));
  requireInfo({nodes[1]});
  hipGraph_t returned = nullptr;
  HIP_CHECK(hipStreamEndCapture(r.s(), &returned));
  REQUIRE(returned == graph);
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *  - Capture rules are the same as for normal capture: unsafe calls under each mode, sync
 *    invalidation, an unjoined side stream (allowed during recapture), and the per-thread
 *    stream.
 * Test source
 * ------------------------
 *  - unit/graph/hipStreamBeginRecaptureToGraph.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipStreamBeginRecaptureToGraph_CaptureRules) {
  Resources r;
  hipStream_t s = r.s();
  int* out = r.out();
  hipGraph_t graph = Capture(s, [&] { return LaunchWrite(s, out, 1); });
  void* allocation = nullptr;

  const auto unsafeCall = [&](hipStreamCaptureMode mode, bool otherThread) {
    hipError_t result = hipSuccess;
    const EndResult end = Recapture(
        s, graph, nullptr,
        [&] {
          HIP_CHECK(LaunchWrite(s, out, 1));
          if (otherThread) {
            std::thread([&] { result = hipMalloc(&allocation, 16); }).join();
          } else {
            result = hipMalloc(&allocation, 16);
          }
        },
        mode);
    if (allocation != nullptr) {
      HIP_CHECK(hipFree(allocation));
    }
    return std::make_pair(result, end);
  };

  SECTION("Global mode, hipMalloc on the capturing thread invalidates") {
    const auto [result, end] = unsafeCall(hipStreamCaptureModeGlobal, false);
    REQUIRE(result == hipErrorStreamCaptureUnsupported);
    RequireEndFailure(end, hipErrorStreamCaptureInvalidated);
    graph = nullptr;
  }
  SECTION("Thread-local mode, hipMalloc on the capturing thread invalidates") {
    const auto [result, end] = unsafeCall(hipStreamCaptureModeThreadLocal, false);
    REQUIRE(result == hipErrorStreamCaptureUnsupported);
    RequireEndFailure(end, hipErrorStreamCaptureInvalidated);
    graph = nullptr;
  }
  SECTION("Thread-local mode, hipMalloc on another thread is allowed") {
    const auto [result, end] = unsafeCall(hipStreamCaptureModeThreadLocal, true);
    HIP_CHECK(result);
    RequireEndSuccess(end, graph);
  }
  SECTION("Relaxed mode, hipMalloc on the capturing thread is allowed") {
    const auto [result, end] = unsafeCall(hipStreamCaptureModeRelaxed, false);
    HIP_CHECK(result);
    RequireEndSuccess(end, graph);
  }

  SECTION("Stream synchronize invalidates") {
    const EndResult end = Recapture(s, graph, nullptr, [&] {
      HIP_CHECK(LaunchWrite(s, out, 1));
      REQUIRE(hipStreamSynchronize(s) == hipErrorStreamCaptureUnsupported);
    });
    RequireEndFailure(end, hipErrorStreamCaptureInvalidated);
    graph = nullptr;
  }

  SECTION("Unjoined side stream is allowed") {
    HIP_CHECK(hipGraphDestroy(graph));
    graph = Capture(s, [&] { return EnqueueForked(r, 10, 20, false); });
    const EndResult end = Recapture(s, graph, nullptr, [&] {
      HIP_CHECK(hipEventRecord(r.fork(), s));
      HIP_CHECK(hipStreamWaitEvent(r.sideStream(), r.fork(), 0));
      HIP_CHECK(LaunchWrite(s, out, 10));
      HIP_CHECK(LaunchWrite(r.sideStream(), out + 1, 20));
    });
    RequireEndSuccess(end, graph);
    hipStreamCaptureStatus status = hipStreamCaptureStatusActive;
    HIP_CHECK(hipStreamIsCapturing(r.sideStream(), &status));
    REQUIRE(status == hipStreamCaptureStatusNone);
  }

  SECTION("Per-thread stream") {
    HIP_CHECK(hipGraphDestroy(graph));
    graph = Capture(hipStreamPerThread, [&] { return LaunchWrite(hipStreamPerThread, out, 1); });
    const EndResult end = Recapture(hipStreamPerThread, graph, nullptr, [&] {
      HIP_CHECK(LaunchWrite(hipStreamPerThread, out, 2));
    });
    RequireEndSuccess(end, graph);
    REQUIRE(LaunchAndRead(graph, s, out, 1) == std::vector<int>{2});
  }

  if (graph != nullptr) {
    HIP_CHECK(hipGraphDestroy(graph));
  }
}

/**
 * Test Description
 * ------------------------
 *  - Begin validation: null graph and invalid mode, null and legacy streams, a stream already
 *    capturing, and the empty graph rules.
 * Test source
 * ------------------------
 *  - unit/graph/hipStreamBeginRecaptureToGraph.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipStreamBeginRecaptureToGraph_Validation) {
  Resources r;
  hipStream_t s = r.s();
  int* out = r.out();
  hipGraph_t graph = Capture(s, [&] { return LaunchWrite(s, out, 1); });

  SECTION("Invalid arguments") {
    REQUIRE(hipStreamBeginRecaptureToGraph(s, kMode, nullptr, nullptr) == hipErrorInvalidValue);
    REQUIRE(hipStreamBeginRecaptureToGraph(s, static_cast<hipStreamCaptureMode>(99), graph,
                                           nullptr) == hipErrorInvalidValue);
    REQUIRE(hipStreamBeginRecaptureToGraph(nullptr, kMode, graph, nullptr) ==
            hipErrorStreamCaptureUnsupported);
    REQUIRE(hipStreamBeginRecaptureToGraph(hipStreamLegacy, kMode, graph, nullptr) ==
            hipErrorStreamCaptureUnsupported);
    hipStreamCaptureStatus status = hipStreamCaptureStatusActive;
    HIP_CHECK(hipStreamIsCapturing(s, &status));
    REQUIRE(status == hipStreamCaptureStatusNone);
  }

  SECTION("Stream already capturing") {
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    REQUIRE(hipStreamBeginRecaptureToGraph(s, kMode, graph, nullptr) == hipErrorIllegalState);
    hipGraph_t other = nullptr;
    HIP_CHECK(hipStreamEndCapture(s, &other));
    HIP_CHECK(hipGraphDestroy(other));
  }

  SECTION("Empty graph") {
    hipGraph_t empty = nullptr;
    HIP_CHECK(hipGraphCreate(&empty, 0));
    RequireEndSuccess(Recapture(s, empty, nullptr, [] {}), empty);
    const EndResult end = Recapture(s, empty, nullptr, [&] {
      REQUIRE(LaunchWrite(s, out, 1) == hipErrorGraphRecaptureFailure);
    });
    RequireEndSuccess(end, empty);
    REQUIRE(GetNodes(empty).empty());
    HIP_CHECK(hipGraphDestroy(empty));
  }

  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *  - HIP-specific node counts (design section 4.10): a zero-size memset creates no node, so
 *    changing its size to non-zero adds a node.
 * Test source
 * ------------------------
 *  - unit/graph/hipStreamBeginRecaptureToGraph.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipStreamBeginRecaptureToGraph_HipSpecificNodeCounts) {
  Resources r;
  hipStream_t s = r.s();
  int* out = r.out();

  SECTION("Zero-size memset creates no node") {
    hipGraph_t graph = Capture(s, [&] {
      HIP_CHECK(hipMemsetAsync(out, 1, 0, s));
      return LaunchWrite(s, out + 1, 1);
    });
    REQUIRE(GetNodes(graph).size() == 1);
    const EndResult end = Recapture(s, graph, nullptr, [&] {
      HIP_CHECK(hipMemsetAsync(out, 1, 0, s));
      HIP_CHECK(LaunchWrite(s, out + 1, 2));
    });
    RequireEndSuccess(end, graph);
    REQUIRE(LaunchAndRead(graph, s, out, 2) == std::vector<int>{0, 2});
    HIP_CHECK(hipGraphDestroy(graph));
  }

  SECTION("Zero-size memset made non-zero adds a node") {
    hipGraph_t graph = Capture(s, [&] {
      HIP_CHECK(hipMemsetAsync(out, 1, 0, s));
      return LaunchWrite(s, out + 1, 1);
    });
    const EndResult end = Recapture(s, graph, nullptr, [&] {
      REQUIRE(hipMemsetAsync(out, 1, sizeof(int), s) == hipErrorGraphRecaptureFailure);
      HIP_CHECK(LaunchWrite(s, out + 1, 1));
    });
    RequireEndSuccess(end, graph);
    REQUIRE(LaunchAndRead(graph, s, out, 2) == std::vector<int>{0, 1});
    HIP_CHECK(hipGraphDestroy(graph));
  }
}

/**
 * Test Description
 * ------------------------
 *  - Forked streams recaptured from two threads. With the threads ordered externally the
 *    recapture is deterministic; run concurrently, both independent nodes still match (in
 *    either order) and the launch result is the same.
 * Test source
 * ------------------------
 *  - unit/graph/hipStreamBeginRecaptureToGraph.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipStreamBeginRecaptureToGraph_ForkedThreads) {
  Resources r;
  hipStream_t s = r.s();
  int* out = r.out();
  hipGraph_t graph = Capture(s, [&] { return EnqueueForked(r, 10, 20, false); });
  const bool concurrent = GENERATE(false, true);

  HIP_CHECK(hipStreamBeginRecaptureToGraph(s, hipStreamCaptureModeRelaxed, graph, nullptr));
  HIP_CHECK(hipEventRecord(r.fork(), s));
  HIP_CHECK(hipStreamWaitEvent(r.sideStream(), r.fork(), 0));
  hipError_t resultA = hipErrorUnknown;
  hipError_t resultB = hipErrorUnknown;
  {
    std::thread a([&] { resultA = LaunchWrite(s, out, 11); });
    if (!concurrent) a.join();
    std::thread b([&] { resultB = LaunchWrite(r.sideStream(), out + 1, 22); });
    if (concurrent) a.join();
    b.join();
  }
  HIP_CHECK(hipEventRecord(r.join(), r.sideStream()));
  HIP_CHECK(hipStreamWaitEvent(s, r.join(), 0));
  hipGraph_t returned = nullptr;
  HIP_CHECK(hipStreamEndCapture(s, &returned));
  HIP_CHECK(resultA);
  HIP_CHECK(resultB);
  REQUIRE(returned == graph);
  REQUIRE(LaunchAndRead(graph, s, out, 2) == std::vector<int>{11, 22});
  HIP_CHECK(hipGraphDestroy(graph));
}

#endif

/**
 * End doxygen group GraphTest.
 * @}
 */
