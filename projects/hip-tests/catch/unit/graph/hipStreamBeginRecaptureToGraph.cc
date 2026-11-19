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
#include <thread>
#include <vector>

// Available on cuda 13.3 and above
#if !HT_NVIDIA || CUDA_VERSION >= CUDA_13030

namespace {
constexpr size_t kCount = 4;
constexpr size_t kBytes = kCount * sizeof(int);
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

__device__ int g_symbol[kCount];

void HostNoop(void*) {}
void OtherHostNoop(void*) {}

std::atomic<int> g_userObjectsDestroyed{0};
void DestroyUserObject(void*) { g_userObjectsDestroyed.fetch_add(1); }

// A failed recapture leaves that error in the thread's last-error slot. Clear it so the
// returned status belongs to this launch.
// hipLaunchKernel reports its status twice: as the return value and as the thread's last error.
// Return the status and clear the last error, so a failure that the test expects does not leak
// into the next call or the test's leftover-error check.
hipError_t LaunchAndClear(const void* function, dim3 grid, dim3 block, void** args,
                          size_t sharedBytes, hipStream_t stream) {
  (void)hipGetLastError();
  const hipError_t result = hipLaunchKernel(function, grid, block, args, sharedBytes, stream);
  (void)hipGetLastError();
  return result;
}

hipError_t LaunchWrite(hipStream_t stream, int* output, int value, dim3 grid = dim3(1),
                       dim3 block = dim3(1), size_t sharedBytes = 0) {
  void* args[] = {&output, &value};
  return LaunchAndClear(reinterpret_cast<const void*>(WriteValue), grid, block, args, sharedBytes,
                        stream);
}

hipError_t LaunchAdd(hipStream_t stream, int* output, int value) {
  void* args[] = {&output, &value};
  return LaunchAndClear(reinterpret_cast<const void*>(AddValue), dim3(1), dim3(1), args, 0, stream);
}

hipError_t LaunchCopy(hipStream_t stream, int* dst, const int* src) {
  void* args[] = {&dst, &src};
  return LaunchAndClear(reinterpret_cast<const void*>(CopyValue), dim3(1), dim3(1), args, 0,
                        stream);
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
  const hipError_t first = LaunchWrite(stream, output, a);
  return first == hipSuccess ? LaunchWrite(stream, output + 1, b) : first;
}

// A writes output[0] on the origin stream, B writes output[1] on the side stream.
hipError_t EnqueueForked(hipStream_t stream, hipStream_t side, hipEvent_t fork, hipEvent_t join,
                         int* output, int a, int b, bool sideFirst, hipError_t* bResult = nullptr) {
  HIP_CHECK(hipEventRecord(fork, stream));
  HIP_CHECK(hipStreamWaitEvent(side, fork, 0));
  hipError_t resultB = hipSuccess;
  hipError_t resultA = hipSuccess;
  if (sideFirst) resultB = LaunchWrite(side, output + 1, b);
  resultA = LaunchWrite(stream, output, a);
  if (!sideFirst) resultB = LaunchWrite(side, output + 1, b);
  HIP_CHECK(hipEventRecord(join, side));
  HIP_CHECK(hipStreamWaitEvent(stream, join, 0));
  if (bResult != nullptr) *bResult = resultB;
  return resultA != hipSuccess ? resultA : resultB;
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

hipGraphNode_t AddWriteNode(hipGraph_t graph, const hipGraphNode_t* deps, size_t numDeps,
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

std::vector<hipGraphNode_t> GetNodes(hipGraph_t graph) {
  size_t count = 0;
  HIP_CHECK(hipGraphGetNodes(graph, nullptr, &count));
  std::vector<hipGraphNode_t> nodes(count);
  if (count != 0) {
    HIP_CHECK(hipGraphGetNodes(graph, nodes.data(), &count));
  }
  return nodes;
}

// End capture and require the stream to have left capture, whichever way the end went.
hipError_t EndRecapture(hipStream_t stream, hipGraph_t* graph) {
  const hipError_t error = hipStreamEndCapture(stream, graph);
  hipStreamCaptureStatus status = hipStreamCaptureStatusActive;
  HIP_CHECK(hipStreamIsCapturing(stream, &status));
  REQUIRE(status == hipStreamCaptureStatusNone);
  return error;
}

void RequireExecLaunch(hipGraphExec_t exec, hipStream_t stream, int* output,
                       std::initializer_list<int> expected) {
  HIP_CHECK(hipMemsetAsync(output, 0, expected.size() * sizeof(int), stream));
  HIP_CHECK(hipGraphLaunch(exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  std::vector<int> actual(expected.size());
  HIP_CHECK(hipMemcpy(actual.data(), output, actual.size() * sizeof(int), hipMemcpyDeviceToHost));
  REQUIRE(actual == std::vector<int>(expected));
}

void RequireGraphLaunch(hipGraph_t graph, hipStream_t stream, int* output,
                        std::initializer_list<int> expected) {
  hipGraphExec_t exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  RequireExecLaunch(exec, stream, output, expected);
  HIP_CHECK(hipGraphExecDestroy(exec));
}

void FillScratch(int* scratch, int* host) {
  const int scratchValues[kCount] = {7, 8, 9, 10};
  HIP_CHECK(hipMemcpy(scratch, scratchValues, sizeof(scratchValues), hipMemcpyHostToDevice));
  if (host != nullptr) {
    for (int i = 0; i < static_cast<int>(kCount); ++i) {
      host[i] = 21 + i;
    }
  }
}

bool WaitForUserObjectsDestroyed(int count) {
  for (int i = 0; i < 100 && g_userObjectsDestroyed.load() < count; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return g_userObjectsDestroyed.load() == count;
}

// User data for hipGraphRecaptureCallback_t. The callback runs inside the capturing call.
struct RecaptureLog {
  int calls = 0;
  bool typesMatch = true;
  std::vector<hipGraphNode_t> nodes;
  std::vector<hipGraphRecaptureStatus> statuses;
  std::vector<int> originalValues;
  std::vector<int> recaptureValues;
  hipError_t editResult = hipSuccess;
  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;
  int* paramOutput = nullptr;
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

void NoteRecapture(RecaptureLog* log, hipGraphNode_t node, const hipGraphNodeParams* original,
                   const hipGraphNodeParams* recaptured, hipGraphRecaptureStatus status) {
  ++log->calls;
  log->nodes.push_back(node);
  log->statuses.push_back(status);
  log->typesMatch = log->typesMatch && original != nullptr && recaptured != nullptr &&
                    original->type == recaptured->type;
  log->originalValues.push_back(original != nullptr ? KernelValue(original) : -1);
  log->recaptureValues.push_back(recaptured != nullptr ? KernelValue(recaptured) : -1);
}

hipError_t AcceptRecapture(void* data, hipGraphNode_t node, const hipGraphNodeParams* original,
                           const hipGraphNodeParams* recaptured, hipGraphRecaptureStatus status) {
  NoteRecapture(static_cast<RecaptureLog*>(data), node, original, recaptured, status);
  return hipSuccess;
}

hipError_t RejectRecapture(void* data, hipGraphNode_t node, const hipGraphNodeParams* original,
                           const hipGraphNodeParams* recaptured, hipGraphRecaptureStatus status) {
  NoteRecapture(static_cast<RecaptureLog*>(data), node, original, recaptured, status);
  return hipErrorInvalidValue;
}

hipError_t RejectFirstRecapture(void* data, hipGraphNode_t node, const hipGraphNodeParams* original,
                                const hipGraphNodeParams* recaptured,
                                hipGraphRecaptureStatus status) {
  auto* log = static_cast<RecaptureLog*>(data);
  NoteRecapture(log, node, original, recaptured, status);
  return log->calls == 1 ? hipErrorInvalidValue : hipSuccess;
}

hipError_t AddEmptyDuringRecapture(void* data, hipGraphNode_t node,
                                   const hipGraphNodeParams* original,
                                   const hipGraphNodeParams* recaptured,
                                   hipGraphRecaptureStatus status) {
  auto* log = static_cast<RecaptureLog*>(data);
  NoteRecapture(log, node, original, recaptured, status);
  if (log->calls == 1) {
    hipGraphNode_t empty = nullptr;
    log->editResult = hipGraphAddEmptyNode(&empty, log->graph, nullptr, 0);
  }
  return hipSuccess;
}

hipError_t DestroyNodeDuringRecapture(void* data, hipGraphNode_t node,
                                      const hipGraphNodeParams* original,
                                      const hipGraphNodeParams* recaptured,
                                      hipGraphRecaptureStatus status) {
  auto* log = static_cast<RecaptureLog*>(data);
  NoteRecapture(log, node, original, recaptured, status);
  if (log->calls == 1) {
    log->editResult = hipGraphDestroyNode(log->node);
  }
  return hipSuccess;
}

hipError_t SetParamsDuringRecapture(void* data, hipGraphNode_t node,
                                    const hipGraphNodeParams* original,
                                    const hipGraphNodeParams* recaptured,
                                    hipGraphRecaptureStatus status) {
  auto* log = static_cast<RecaptureLog*>(data);
  NoteRecapture(log, node, original, recaptured, status);
  if (log->calls == 1) {
    int value = 9;
    int* target = log->paramOutput;
    void* args[] = {&target, &value};
    hipKernelNodeParams params{};
    params.func = reinterpret_cast<void*>(WriteValue);
    params.gridDim = dim3(1);
    params.blockDim = dim3(1);
    params.kernelParams = args;
    log->editResult = hipGraphKernelNodeSetParams(log->node, &params);
  }
  return hipSuccess;
}

// The original graph has one node. Recapture already ended successfully. Status 0 must be
// applicable with hipGraphExecUpdate. Status 1 is applied to the graph, but an exec instantiated
// before the recapture rejects it with ParametersChanged.
void RequireRecaptureStatus(hipStream_t stream, int* output, hipGraph_t graph, hipGraphExec_t exec,
                            const RecaptureLog& log, hipGraphRecaptureStatus status,
                            std::initializer_list<int> expected) {
  REQUIRE(log.calls == 1);
  REQUIRE(log.statuses.at(0) == status);
  REQUIRE(log.typesMatch);
  REQUIRE(GetNodes(graph).size() == 1);
  hipGraphNode_t errorNode = nullptr;
  hipGraphExecUpdateResult updateResult = hipGraphExecUpdateError;
  const hipError_t updateError = hipGraphExecUpdate(exec, graph, &errorNode, &updateResult);
  if (status == hipGraphRecaptureEligibleForUpdate) {
    REQUIRE(updateError == hipSuccess);
    REQUIRE(updateResult == hipGraphExecUpdateSuccess);
  } else {
    REQUIRE(updateError == hipErrorGraphExecUpdateFailure);
    REQUIRE(updateResult == hipGraphExecUpdateErrorParametersChanged);
  }
  HIP_CHECK(hipGraphExecDestroy(exec));
  if (expected.size() != 0) {
    RequireGraphLaunch(graph, stream, output, expected);
  }
  HIP_CHECK(hipGraphDestroy(graph));
}
}  // namespace

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
  LinearAllocGuard<int> output(LinearAllocs::hipMalloc, kBytes);
  StreamGuard stream{Streams::created};
  hipStream_t s = stream.stream();
  int* out = output.ptr();

  SECTION("Identical recapture makes no callback") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(LaunchWrite(s, out, 1));
    HIP_CHECK(hipStreamEndCapture(s, &graph));

    RecaptureLog log;
    hipGraphRecaptureCallbackData callback{AcceptRecapture, &log};
    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, &callback));
    HIP_CHECK(LaunchWrite(s, out, 1));
    hipGraph_t returned = nullptr;
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == graph);
    REQUIRE(log.calls == 0);
    RequireGraphLaunch(graph, s, out, {1});
    HIP_CHECK(hipGraphDestroy(graph));
  }

  SECTION("No callback data applies the change") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(LaunchWrite(s, out, 1));
    HIP_CHECK(hipStreamEndCapture(s, &graph));

    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, nullptr));
    HIP_CHECK(LaunchWrite(s, out, 4));
    hipGraph_t returned = nullptr;
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == graph);
    RequireGraphLaunch(graph, s, out, {4});
    HIP_CHECK(hipGraphDestroy(graph));
  }

  SECTION("Null callbackFunc applies the change") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(LaunchWrite(s, out, 1));
    HIP_CHECK(hipStreamEndCapture(s, &graph));

    hipGraphRecaptureCallbackData callback{nullptr, nullptr};
    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, &callback));
    HIP_CHECK(LaunchWrite(s, out, 4));
    hipGraph_t returned = nullptr;
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == graph);
    RequireGraphLaunch(graph, s, out, {4});
    HIP_CHECK(hipGraphDestroy(graph));
  }

  SECTION("Rejected single node keeps its parameters") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(LaunchWrite(s, out, 1));
    HIP_CHECK(hipStreamEndCapture(s, &graph));

    RecaptureLog log;
    hipGraphRecaptureCallbackData callback{RejectRecapture, &log};
    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, &callback));
    HIP_CHECK_ERROR(LaunchWrite(s, out, 8), hipErrorGraphRecaptureFailure);
    hipGraph_t returned = nullptr;
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == graph);
    REQUIRE(log.calls == 1);
    RequireGraphLaunch(graph, s, out, {1});
    HIP_CHECK(hipGraphDestroy(graph));
  }

  SECTION("Rejecting the first of two nodes terminates the recapture") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(EnqueueChain(s, out, 1, 2));
    HIP_CHECK(hipStreamEndCapture(s, &graph));

    RecaptureLog log;
    hipGraphRecaptureCallbackData callback{RejectFirstRecapture, &log};
    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, &callback));
    HIP_CHECK_ERROR(LaunchWrite(s, out, 11), hipErrorGraphRecaptureFailure);
    HIP_CHECK_ERROR(LaunchWrite(s, out + 1, 22), hipErrorGraphRecaptureFailure);
    hipGraph_t returned = nullptr;
    HIP_CHECK_ERROR(EndRecapture(s, &returned), hipErrorGraphRecaptureFailure);
    REQUIRE(returned == nullptr);
    REQUIRE(log.calls == 1);
  }

  SECTION("Node added from the callback fails the capturing call after its update") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(EnqueueChain(s, out, 1, 2));
    HIP_CHECK(hipStreamEndCapture(s, &graph));

    RecaptureLog log;
    log.graph = graph;
    hipGraphRecaptureCallbackData callback{AddEmptyDuringRecapture, &log};
    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, &callback));
    HIP_CHECK_ERROR(LaunchWrite(s, out, 3), hipErrorGraphRecaptureFailure);
    HIP_CHECK(LaunchWrite(s, out + 1, 2));
    hipGraph_t returned = nullptr;
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == graph);
    REQUIRE(log.editResult == hipErrorGraphRecaptureFailure);
    REQUIRE(log.calls == 1);
    REQUIRE(GetNodes(graph).size() == 2);
    RequireGraphLaunch(graph, s, out, {3, 2});
    HIP_CHECK(hipGraphDestroy(graph));
  }

  SECTION("Node destroyed from the callback fails the capturing call after its update") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(EnqueueChain(s, out, 1, 2));
    HIP_CHECK(hipStreamEndCapture(s, &graph));
    const std::vector<hipGraphNode_t> nodes = GetNodes(graph);
    REQUIRE(nodes.size() == 2);

    RecaptureLog log;
    log.node = nodes[1];
    hipGraphRecaptureCallbackData callback{DestroyNodeDuringRecapture, &log};
    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, &callback));
    HIP_CHECK_ERROR(LaunchWrite(s, out, 3), hipErrorIllegalState);
    HIP_CHECK(LaunchWrite(s, out + 1, 2));
    hipGraph_t returned = nullptr;
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == graph);
    REQUIRE(log.editResult == hipErrorIllegalState);
    REQUIRE(log.calls == 1);
    REQUIRE(GetNodes(graph) == nodes);
    RequireGraphLaunch(graph, s, out, {3, 2});
    HIP_CHECK(hipGraphDestroy(graph));
  }

  SECTION("Parameter edit from the callback is what the next node is compared with") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(EnqueueChain(s, out, 1, 2));
    HIP_CHECK(hipStreamEndCapture(s, &graph));
    const std::vector<hipGraphNode_t> nodes = GetNodes(graph);
    REQUIRE(nodes.size() == 2);

    RecaptureLog log;
    log.node = nodes[1];
    log.paramOutput = out + 1;
    hipGraphRecaptureCallbackData callback{SetParamsDuringRecapture, &log};
    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, &callback));
    HIP_CHECK(LaunchWrite(s, out, 3));
    HIP_CHECK(LaunchWrite(s, out + 1, 2));
    hipGraph_t returned = nullptr;
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == graph);
    HIP_CHECK(log.editResult);
    REQUIRE(log.calls == 2);
    REQUIRE(log.originalValues == std::vector<int>{1, 9});
    REQUIRE(log.recaptureValues == std::vector<int>{3, 2});
    RequireGraphLaunch(graph, s, out, {3, 2});
    HIP_CHECK(hipGraphDestroy(graph));
  }
}

/**
 * Test Description
 * ------------------------
 *  - One kernel argument change updates the original node and leaves an already instantiated
 *    exec unchanged until hipGraphExecUpdate. Each other parameter change is reported to the
 *    callback and applied. Status 0 can be applied with hipGraphExecUpdate; status 1 cannot.
 *    Priority and the cooperative launch mode are attributes: they are applied with no callback.
 * Test source
 * ------------------------
 *  - unit/graph/hipStreamBeginRecaptureToGraph.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipStreamBeginRecaptureToGraph_ParameterStatus) {
  LinearAllocGuard<int> output(LinearAllocs::hipMalloc, kBytes);
  LinearAllocGuard<int> scratch(LinearAllocs::hipMalloc, kBytes);
  LinearAllocGuard<int> host(LinearAllocs::hipHostMalloc, kBytes);
  FillScratch(scratch.ptr(), host.ptr());
  StreamGuard stream{Streams::created};
  EventsGuard events{2};
  hipStream_t s = stream.stream();
  int* out = output.ptr();
  int* device = scratch.ptr();
  int* pinned = host.ptr();
  constexpr auto kEligible = hipGraphRecaptureEligibleForUpdate;
  constexpr auto kIneligible = hipGraphRecaptureIneligibleForUpdate;

  SECTION("In-place kernel argument") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(LaunchWrite(s, out, 1));
    HIP_CHECK(hipStreamEndCapture(s, &graph));
    const std::vector<hipGraphNode_t> before = GetNodes(graph);
    REQUIRE(before.size() == 1);

    hipGraphExec_t exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    RecaptureLog log;
    hipGraphRecaptureCallbackData callback{AcceptRecapture, &log};
    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, &callback));
    HIP_CHECK(LaunchWrite(s, out, 2));
    hipGraph_t returned = nullptr;
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == graph);
    REQUIRE(log.calls == 1);
    REQUIRE(log.statuses.at(0) == kEligible);
    REQUIRE(log.nodes.at(0) == before[0]);
    REQUIRE(log.originalValues.at(0) == 1);
    REQUIRE(log.recaptureValues.at(0) == 2);
    REQUIRE(GetNodes(graph) == before);

    RequireExecLaunch(exec, s, out, {1});
    hipGraphNode_t errorNode = nullptr;
    hipGraphExecUpdateResult updateResult = hipGraphExecUpdateError;
    HIP_CHECK(hipGraphExecUpdate(exec, graph, &errorNode, &updateResult));
    REQUIRE(updateResult == hipGraphExecUpdateSuccess);
    RequireExecLaunch(exec, s, out, {2});
    HIP_CHECK(hipGraphExecDestroy(exec));
    RequireGraphLaunch(graph, s, out, {2});
    HIP_CHECK(hipGraphDestroy(graph));
  }

  SECTION("Kernel function") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(LaunchWrite(s, out, 1));
    HIP_CHECK(hipStreamEndCapture(s, &graph));
    hipGraphExec_t exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    RecaptureLog log;
    hipGraphRecaptureCallbackData callback{AcceptRecapture, &log};
    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, &callback));
    HIP_CHECK(LaunchAdd(s, out, 1));
    hipGraph_t returned = nullptr;
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == graph);
    RequireRecaptureStatus(s, out, graph, exec, log, kEligible, {1});
  }

  SECTION("Kernel grid") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(LaunchWrite(s, out, 1));
    HIP_CHECK(hipStreamEndCapture(s, &graph));
    hipGraphExec_t exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    RecaptureLog log;
    hipGraphRecaptureCallbackData callback{AcceptRecapture, &log};
    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, &callback));
    HIP_CHECK(LaunchWrite(s, out, 1, dim3(2)));
    hipGraph_t returned = nullptr;
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == graph);
    RequireRecaptureStatus(s, out, graph, exec, log, kEligible, {1});
  }

  SECTION("Kernel dynamic shared memory") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(LaunchWrite(s, out, 1));
    HIP_CHECK(hipStreamEndCapture(s, &graph));
    hipGraphExec_t exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    RecaptureLog log;
    hipGraphRecaptureCallbackData callback{AcceptRecapture, &log};
    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, &callback));
    HIP_CHECK(LaunchWrite(s, out, 1, dim3(1), dim3(1), 256));
    hipGraph_t returned = nullptr;
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == graph);
    RequireRecaptureStatus(s, out, graph, exec, log, kEligible, {1});
  }

  SECTION("Memset value") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(hipMemsetAsync(out, 0, sizeof(int), s));
    HIP_CHECK(hipStreamEndCapture(s, &graph));
    hipGraphExec_t exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    RecaptureLog log;
    hipGraphRecaptureCallbackData callback{AcceptRecapture, &log};
    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, &callback));
    HIP_CHECK(hipMemsetAsync(out, 1, sizeof(int), s));
    hipGraph_t returned = nullptr;
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == graph);
    RequireRecaptureStatus(s, out, graph, exec, log, kEligible, {0x01010101});
  }

  SECTION("Memset 1D size") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(hipMemsetAsync(out, 1, sizeof(int), s));
    HIP_CHECK(hipStreamEndCapture(s, &graph));
    hipGraphExec_t exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    RecaptureLog log;
    hipGraphRecaptureCallbackData callback{AcceptRecapture, &log};
    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, &callback));
    HIP_CHECK(hipMemsetAsync(out, 1, 2 * sizeof(int), s));
    hipGraph_t returned = nullptr;
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == graph);
    RequireRecaptureStatus(s, out, graph, exec, log, kEligible, {0x01010101, 0x01010101});
  }

  SECTION("Memcpy size") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(hipMemcpyAsync(out, device, sizeof(int), hipMemcpyDefault, s));
    HIP_CHECK(hipStreamEndCapture(s, &graph));
    hipGraphExec_t exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    RecaptureLog log;
    hipGraphRecaptureCallbackData callback{AcceptRecapture, &log};
    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, &callback));
    HIP_CHECK(hipMemcpyAsync(out, device, 2 * sizeof(int), hipMemcpyDefault, s));
    hipGraph_t returned = nullptr;
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == graph);
    RequireRecaptureStatus(s, out, graph, exec, log, kEligible, {7, 8});
  }

  SECTION("Memcpy source moved from device to pinned host") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(hipMemcpyAsync(out, device, sizeof(int), hipMemcpyDefault, s));
    HIP_CHECK(hipStreamEndCapture(s, &graph));
    hipGraphExec_t exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    RecaptureLog log;
    hipGraphRecaptureCallbackData callback{AcceptRecapture, &log};
    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, &callback));
    HIP_CHECK(hipMemcpyAsync(out, pinned, sizeof(int), hipMemcpyDefault, s));
    hipGraph_t returned = nullptr;
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == graph);
    RequireRecaptureStatus(s, out, graph, exec, log, kEligible, {21});
  }

  SECTION("Memcpy destination moved from device to pinned host") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(hipMemcpyAsync(out, device, sizeof(int), hipMemcpyDefault, s));
    HIP_CHECK(hipStreamEndCapture(s, &graph));
    hipGraphExec_t exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    RecaptureLog log;
    hipGraphRecaptureCallbackData callback{AcceptRecapture, &log};
    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, &callback));
    HIP_CHECK(hipMemcpyAsync(pinned, device, sizeof(int), hipMemcpyDefault, s));
    hipGraph_t returned = nullptr;
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == graph);
    RequireRecaptureStatus(s, out, graph, exec, log, kEligible, {});
  }

  SECTION("Memcpy kind Default to DeviceToDevice") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(hipMemcpyAsync(out, device, sizeof(int), hipMemcpyDefault, s));
    HIP_CHECK(hipStreamEndCapture(s, &graph));
    hipGraphExec_t exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    RecaptureLog log;
    hipGraphRecaptureCallbackData callback{AcceptRecapture, &log};
    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, &callback));
    HIP_CHECK(hipMemcpyAsync(out, device, sizeof(int), hipMemcpyDeviceToDevice, s));
    hipGraph_t returned = nullptr;
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == graph);
    RequireRecaptureStatus(s, out, graph, exec, log, kIneligible, {7});
  }

  SECTION("Memcpy 3D width") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(Memcpy3D(s, out, device, 16));
    HIP_CHECK(hipStreamEndCapture(s, &graph));
    hipGraphExec_t exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    RecaptureLog log;
    hipGraphRecaptureCallbackData callback{AcceptRecapture, &log};
    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, &callback));
    HIP_CHECK(Memcpy3D(s, out, device, 8));
    hipGraph_t returned = nullptr;
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == graph);
    RequireRecaptureStatus(s, out, graph, exec, log, kEligible, {7, 8, 0, 0});
  }

  SECTION("Memset 2D height 2 to 3 rows") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(hipMemset2DAsync(out, sizeof(int), 1, sizeof(int), 2, s));
    HIP_CHECK(hipStreamEndCapture(s, &graph));
    hipGraphExec_t exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    RecaptureLog log;
    hipGraphRecaptureCallbackData callback{AcceptRecapture, &log};
    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, &callback));
    HIP_CHECK(hipMemset2DAsync(out, sizeof(int), 1, sizeof(int), 3, s));
    hipGraph_t returned = nullptr;
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == graph);
    RequireRecaptureStatus(s, out, graph, exec, log, kIneligible, {0x01010101, 0x01010101, 0x01010101, 0});
  }

  SECTION("Memset 1D to 2D") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(hipMemsetAsync(out, 0, sizeof(int), s));
    HIP_CHECK(hipStreamEndCapture(s, &graph));
    hipGraphExec_t exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    RecaptureLog log;
    hipGraphRecaptureCallbackData callback{AcceptRecapture, &log};
    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, &callback));
    HIP_CHECK(hipMemset2DAsync(out, 8, 0, sizeof(int), 2, s));
    hipGraph_t returned = nullptr;
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == graph);
    RequireRecaptureStatus(s, out, graph, exec, log, kIneligible, {});
  }

  SECTION("Memcpy to symbol offset") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(hipMemcpyToSymbolAsync(HIP_SYMBOL(g_symbol), device, sizeof(int), 0,
                                    hipMemcpyDeviceToDevice, s));
    HIP_CHECK(hipStreamEndCapture(s, &graph));
    hipGraphExec_t exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    RecaptureLog log;
    hipGraphRecaptureCallbackData callback{AcceptRecapture, &log};
    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, &callback));
    HIP_CHECK(hipMemcpyToSymbolAsync(HIP_SYMBOL(g_symbol), device, sizeof(int), sizeof(int),
                                    hipMemcpyDeviceToDevice, s));
    hipGraph_t returned = nullptr;
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == graph);
    RequireRecaptureStatus(s, out, graph, exec, log, kEligible, {});
  }

  SECTION("Memcpy to symbol kind change") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(hipMemcpyToSymbolAsync(HIP_SYMBOL(g_symbol), device, sizeof(int), 0,
                                    hipMemcpyDeviceToDevice, s));
    HIP_CHECK(hipStreamEndCapture(s, &graph));
    hipGraphExec_t exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    RecaptureLog log;
    hipGraphRecaptureCallbackData callback{AcceptRecapture, &log};
    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, &callback));
    HIP_CHECK(hipMemcpyToSymbolAsync(HIP_SYMBOL(g_symbol), pinned, sizeof(int), 0,
                                    hipMemcpyHostToDevice, s));
    hipGraph_t returned = nullptr;
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == graph);
    RequireRecaptureStatus(s, out, graph, exec, log, kIneligible, {});
  }

  SECTION("Host node function") {
    int a = 0;
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(hipLaunchHostFunc(s, HostNoop, &a));
    HIP_CHECK(hipStreamEndCapture(s, &graph));
    hipGraphExec_t exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    RecaptureLog log;
    hipGraphRecaptureCallbackData callback{AcceptRecapture, &log};
    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, &callback));
    HIP_CHECK(hipLaunchHostFunc(s, OtherHostNoop, &a));
    hipGraph_t returned = nullptr;
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == graph);
    RequireRecaptureStatus(s, out, graph, exec, log, kEligible, {});
  }

  SECTION("Event record and wait") {
    hipGraph_t record = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(hipEventRecordWithFlags(events[0], s, hipEventRecordExternal));
    HIP_CHECK(hipStreamEndCapture(s, &record));
    hipGraphExec_t recordExec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&recordExec, record, nullptr, nullptr, 0));
    RecaptureLog recordLog;
    hipGraphRecaptureCallbackData recordCallback{AcceptRecapture, &recordLog};
    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, record, &recordCallback));
    HIP_CHECK(hipEventRecordWithFlags(events[1], s, hipEventRecordExternal));
    hipGraph_t returned = nullptr;
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == record);
    RequireRecaptureStatus(s, out, record, recordExec, recordLog, kEligible, {});

    hipGraph_t wait = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(hipStreamWaitEvent(s, events[0], hipEventWaitExternal));
    HIP_CHECK(hipStreamEndCapture(s, &wait));
    hipGraphExec_t waitExec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&waitExec, wait, nullptr, nullptr, 0));
    RecaptureLog waitLog;
    hipGraphRecaptureCallbackData waitCallback{AcceptRecapture, &waitLog};
    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, wait, &waitCallback));
    HIP_CHECK(hipStreamWaitEvent(s, events[1], hipEventWaitExternal));
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == wait);
    RequireRecaptureStatus(s, out, wait, waitExec, waitLog, kEligible, {});
  }

  SECTION("Priority follows the recapturing stream") {
    int least = 0, greatest = 0;
    HIP_CHECK(hipDeviceGetStreamPriorityRange(&least, &greatest));
    if (least == greatest) {
      HIP_SKIP_TEST("Device does not support stream priority");
    }
    StreamGuard high(Streams::withPriority, hipStreamDefault, greatest);
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(LaunchWrite(s, out, 1));
    HIP_CHECK(hipStreamEndCapture(s, &graph));
    const hipGraphNode_t node = GetNodes(graph).at(0);
    hipKernelNodeAttrValue value{};
    HIP_CHECK(hipGraphKernelNodeGetAttribute(node, hipLaunchAttributePriority, &value));
    REQUIRE(value.priority != greatest);

    RecaptureLog log;
    hipGraphRecaptureCallbackData callback{AcceptRecapture, &log};
    HIP_CHECK(hipStreamBeginRecaptureToGraph(high.stream(), kMode, graph, &callback));
    HIP_CHECK(LaunchWrite(high.stream(), out, 1));
    hipGraph_t returned = nullptr;
    HIP_CHECK(EndRecapture(high.stream(), &returned));
    REQUIRE(returned == graph);
    REQUIRE(log.calls == 0);
    HIP_CHECK(hipGraphKernelNodeGetAttribute(node, hipLaunchAttributePriority, &value));
    REQUIRE(value.priority == greatest);
    RequireGraphLaunch(graph, s, out, {1});
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

    hipGraph_t reference = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(LaunchWriteEx(s, out, 1, &attr, 1));
    HIP_CHECK(hipStreamEndCapture(s, &reference));
    hipKernelNodeAttrValue expected{};
    HIP_CHECK(hipGraphKernelNodeGetAttribute(GetNodes(reference).at(0), hipLaunchAttributeCooperative,
                                             &expected));
    HIP_CHECK(hipGraphDestroy(reference));

    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(LaunchWriteEx(s, out, 1, nullptr, 0));
    HIP_CHECK(hipStreamEndCapture(s, &graph));
    const hipGraphNode_t node = GetNodes(graph).at(0);
    hipGraphExec_t exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    RecaptureLog log;
    hipGraphRecaptureCallbackData callback{AcceptRecapture, &log};
    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, &callback));
    HIP_CHECK(LaunchWriteEx(s, out, 1, &attr, 1));
    hipGraph_t returned = nullptr;
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == graph);
    REQUIRE(log.calls == 0);
    hipKernelNodeAttrValue value{};
    HIP_CHECK(hipGraphKernelNodeGetAttribute(node, hipLaunchAttributeCooperative, &value));
    REQUIRE(value.cooperative == expected.cooperative);
    // The cooperative launch mode was applied, and hipGraphExecUpdate rejects that change.
    hipGraphNode_t errorNode = nullptr;
    hipGraphExecUpdateResult updateResult = hipGraphExecUpdateSuccess;
    HIP_CHECK_ERROR(hipGraphExecUpdate(exec, graph, &errorNode, &updateResult),
                    hipErrorGraphExecUpdateFailure);
    HIP_CHECK(hipGraphExecDestroy(exec));
    RequireGraphLaunch(graph, s, out, {1});
    HIP_CHECK(hipGraphDestroy(graph));
  }
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
  LinearAllocGuard<int> output(LinearAllocs::hipMalloc, kBytes);
  StreamGuard stream{Streams::created};
  StreamGuard side{Streams::created};
  EventsGuard events{2};
  hipStream_t s = stream.stream();
  int* out = output.ptr();

  SECTION("Type mismatch on the only node") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(LaunchWrite(s, out, 1));
    HIP_CHECK(hipStreamEndCapture(s, &graph));

    RecaptureLog log;
    hipGraphRecaptureCallbackData callback{AcceptRecapture, &log};
    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, &callback));
    HIP_CHECK_ERROR(hipMemsetAsync(out, 0, sizeof(int), s), hipErrorGraphRecaptureFailure);
    hipGraph_t returned = nullptr;
    HIP_CHECK_ERROR(EndRecapture(s, &returned), hipErrorGraphRecaptureFailure);
    REQUIRE(returned == nullptr);
    REQUIRE(log.calls == 0);
  }

  SECTION("Type mismatch then retried") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(EnqueueChain(s, out, 1, 2));
    HIP_CHECK(hipStreamEndCapture(s, &graph));

    RecaptureLog log;
    hipGraphRecaptureCallbackData callback{AcceptRecapture, &log};
    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, &callback));
    HIP_CHECK(LaunchWrite(s, out, 11));
    HIP_CHECK_ERROR(hipMemsetAsync(out + 1, 0, sizeof(int), s), hipErrorGraphRecaptureFailure);
    HIP_CHECK(LaunchWrite(s, out + 1, 22));
    hipGraph_t returned = nullptr;
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == graph);
    REQUIRE(log.calls == 2);
    RequireGraphLaunch(graph, s, out, {11, 22});
    HIP_CHECK(hipGraphDestroy(graph));
  }

  SECTION("Extra node") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(LaunchWrite(s, out, 1));
    HIP_CHECK(hipStreamEndCapture(s, &graph));

    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, nullptr));
    HIP_CHECK(LaunchWrite(s, out, 5));
    HIP_CHECK_ERROR(LaunchWrite(s, out, 6), hipErrorGraphRecaptureFailure);
    hipGraph_t returned = nullptr;
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == graph);
    REQUIRE(GetNodes(graph).size() == 1);
    RequireGraphLaunch(graph, s, out, {5});
    HIP_CHECK(hipGraphDestroy(graph));
  }

  SECTION("Missing node") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(EnqueueChain(s, out, 1, 2));
    HIP_CHECK(hipStreamEndCapture(s, &graph));

    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, nullptr));
    HIP_CHECK(LaunchWrite(s, out, 3));
    hipGraph_t returned = nullptr;
    HIP_CHECK_ERROR(EndRecapture(s, &returned), hipErrorGraphRecaptureFailure);
    REQUIRE(returned == nullptr);
  }

  SECTION("Dependency mismatch then retried with the right dependency") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(EnqueueChain(s, out, 1, 2));
    HIP_CHECK(hipStreamEndCapture(s, &graph));

    RecaptureLog log;
    hipGraphRecaptureCallbackData callback{AcceptRecapture, &log};
    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, &callback));
    HIP_CHECK(hipEventRecord(events[0], s));
    HIP_CHECK(hipStreamWaitEvent(side.stream(), events[0], 0));
    HIP_CHECK(LaunchWrite(s, out, 11));
    HIP_CHECK_ERROR(LaunchWrite(side.stream(), out + 1, 22), hipErrorGraphRecaptureFailure);
    HIP_CHECK(hipEventRecord(events[1], side.stream()));
    HIP_CHECK(hipStreamWaitEvent(s, events[1], 0));
    HIP_CHECK(LaunchWrite(s, out + 1, 22));
    hipGraph_t returned = nullptr;
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == graph);
    RequireGraphLaunch(graph, s, out, {11, 22});
    HIP_CHECK(hipGraphDestroy(graph));
  }

  SECTION("Independent nodes recaptured in swapped order") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(EnqueueForked(s, side.stream(), events[0], events[1], out, 10, 20, false));
    HIP_CHECK(hipStreamEndCapture(s, &graph));

    RecaptureLog log;
    hipGraphRecaptureCallbackData callback{AcceptRecapture, &log};
    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, &callback));
    HIP_CHECK(EnqueueForked(s, side.stream(), events[0], events[1], out, 10, 20, true));
    hipGraph_t returned = nullptr;
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == graph);
    REQUIRE(log.calls == 2);
    REQUIRE(log.originalValues == std::vector<int>{10, 20});
    REQUIRE(log.recaptureValues == std::vector<int>{20, 10});
    RequireGraphLaunch(graph, s, out, {10, 20});
    HIP_CHECK(hipGraphDestroy(graph));
  }

  SECTION("Dependency removed") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(EnqueueChain(s, out, 10, 20));
    HIP_CHECK(hipStreamEndCapture(s, &graph));

    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, nullptr));
    hipError_t resultB = hipSuccess;
    HIP_CHECK_ERROR(EnqueueForked(s, side.stream(), events[0], events[1], out, 10, 20, false, &resultB),
                    hipErrorGraphRecaptureFailure);
    REQUIRE(resultB == hipErrorGraphRecaptureFailure);
    hipGraph_t returned = nullptr;
    HIP_CHECK_ERROR(EndRecapture(s, &returned), hipErrorGraphRecaptureFailure);
    REQUIRE(returned == nullptr);
  }

  SECTION("Dependency added") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(EnqueueForked(s, side.stream(), events[0], events[1], out, 10, 20, false));
    HIP_CHECK(hipStreamEndCapture(s, &graph));

    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, nullptr));
    HIP_CHECK(LaunchWrite(s, out, 10));
    HIP_CHECK_ERROR(LaunchWrite(s, out + 1, 20), hipErrorGraphRecaptureFailure);
    hipGraph_t returned = nullptr;
    HIP_CHECK_ERROR(EndRecapture(s, &returned), hipErrorGraphRecaptureFailure);
    REQUIRE(returned == nullptr);
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
  LinearAllocGuard<int> output(LinearAllocs::hipMalloc, kBytes);
  LinearAllocGuard<int> scratch(LinearAllocs::hipMalloc, kBytes);
  StreamGuard stream{Streams::created};
  hipStream_t s = stream.stream();
  int* out = output.ptr();

  SECTION("Same address, copied out by a kernel") {
    void* original = nullptr;
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(EnqueueAllocChain(s, out, kAllocBytes, 1, &original));
    HIP_CHECK(hipStreamEndCapture(s, &graph));

    void* recaptured = nullptr;
    RecaptureLog log;
    hipGraphRecaptureCallbackData callback{AcceptRecapture, &log};
    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, &callback));
    HIP_CHECK(EnqueueAllocChain(s, out, kAllocBytes, 2, &recaptured));
    hipGraph_t returned = nullptr;
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == graph);
    REQUIRE(recaptured == original);
    REQUIRE(log.calls == 1);
    RequireGraphLaunch(graph, s, out, {2});
    HIP_CHECK(hipGraphDestroy(graph));
  }

  SECTION("Size change is status 2") {
    void* original = nullptr;
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(EnqueueAllocChain(s, out, kAllocBytes, 1, &original));
    HIP_CHECK(hipStreamEndCapture(s, &graph));

    void* recaptured = nullptr;
    RecaptureLog log;
    hipGraphRecaptureCallbackData callback{AcceptRecapture, &log};
    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, &callback));
    HIP_CHECK_ERROR(hipMallocAsync(&recaptured, 2 * kAllocBytes, s), hipErrorGraphRecaptureFailure);
    REQUIRE(recaptured == nullptr);
    hipGraph_t returned = nullptr;
    HIP_CHECK_ERROR(EndRecapture(s, &returned), hipErrorGraphRecaptureFailure);
    REQUIRE(returned == nullptr);
    REQUIRE(log.calls == 1);
    REQUIRE(log.statuses.at(0) == hipGraphRecaptureError);
  }

  SECTION("Free order swapped is status 2") {
    void* first = nullptr;
    void* second = nullptr;
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(hipMallocAsync(&first, kAllocBytes, s));
    HIP_CHECK(hipMallocAsync(&second, kAllocBytes, s));
    HIP_CHECK(hipFreeAsync(first, s));
    HIP_CHECK(hipFreeAsync(second, s));
    HIP_CHECK(hipStreamEndCapture(s, &graph));

    RecaptureLog log;
    hipGraphRecaptureCallbackData callback{AcceptRecapture, &log};
    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, &callback));
    void* firstAgain = nullptr;
    void* secondAgain = nullptr;
    HIP_CHECK(hipMallocAsync(&firstAgain, kAllocBytes, s));
    HIP_CHECK(hipMallocAsync(&secondAgain, kAllocBytes, s));
    REQUIRE(firstAgain == first);
    REQUIRE(secondAgain == second);
    HIP_CHECK_ERROR(hipFreeAsync(secondAgain, s), hipErrorGraphRecaptureFailure);
    HIP_CHECK_ERROR(hipFreeAsync(firstAgain, s), hipErrorGraphRecaptureFailure);
    hipGraph_t returned = nullptr;
    HIP_CHECK_ERROR(EndRecapture(s, &returned), hipErrorGraphRecaptureFailure);
    REQUIRE(returned == nullptr);
    REQUIRE(log.calls == 1);
    REQUIRE(log.statuses.at(0) == hipGraphRecaptureError);
  }

  SECTION("Free of a hipMalloc pointer is status 2 and is not validated") {
    void* original = nullptr;
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(EnqueueAllocChain(s, out, kAllocBytes, 1, &original));
    HIP_CHECK(hipStreamEndCapture(s, &graph));

    void* recaptured = nullptr;
    RecaptureLog log;
    hipGraphRecaptureCallbackData callback{AcceptRecapture, &log};
    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, &callback));
    HIP_CHECK(hipMallocAsync(&recaptured, kAllocBytes, s));
    HIP_CHECK(LaunchWrite(s, static_cast<int*>(recaptured), 1));
    HIP_CHECK(LaunchCopy(s, out, static_cast<int*>(recaptured)));
    HIP_CHECK_ERROR(hipFreeAsync(scratch.ptr(), s), hipErrorGraphRecaptureFailure);
    HIP_CHECK_ERROR(hipFreeAsync(recaptured, s), hipErrorGraphRecaptureFailure);
    hipGraph_t returned = nullptr;
    HIP_CHECK_ERROR(EndRecapture(s, &returned), hipErrorGraphRecaptureFailure);
    REQUIRE(returned == nullptr);
    REQUIRE(log.calls == 1);
    REQUIRE(log.statuses.at(0) == hipGraphRecaptureError);
  }

  SECTION("Null free is a no-op") {
    void* original = nullptr;
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(EnqueueAllocChain(s, out, kAllocBytes, 1, &original));
    HIP_CHECK(hipStreamEndCapture(s, &graph));

    void* recaptured = nullptr;
    RecaptureLog log;
    hipGraphRecaptureCallbackData callback{AcceptRecapture, &log};
    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, &callback));
    HIP_CHECK(hipMallocAsync(&recaptured, kAllocBytes, s));
    HIP_CHECK(LaunchWrite(s, static_cast<int*>(recaptured), 4));
    HIP_CHECK(LaunchCopy(s, out, static_cast<int*>(recaptured)));
    HIP_CHECK(hipFreeAsync(nullptr, s));
    HIP_CHECK(hipFreeAsync(recaptured, s));
    hipGraph_t returned = nullptr;
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == graph);
    REQUIRE(log.calls == 1);
    RequireGraphLaunch(graph, s, out, {4});
    HIP_CHECK(hipGraphDestroy(graph));
  }
}

/**
 * Test Description
 * ------------------------
 *  - Graph sources: a manually built graph, a clone (recaptured independently of its source),
 *    and repeated recaptures that each compare against the parameters left by the previous one.
 *    Recapture of an embedded child graph updates the parent. Begin releases user objects the
 *    graph already owns.
 * Test source
 * ------------------------
 *  - unit/graph/hipStreamBeginRecaptureToGraph.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipStreamBeginRecaptureToGraph_GraphSources) {
  LinearAllocGuard<int> output(LinearAllocs::hipMalloc, kBytes);
  StreamGuard stream{Streams::created};
  hipStream_t s = stream.stream();
  int* out = output.ptr();

  SECTION("Manual graph") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipGraphCreate(&graph, 0));
    const hipGraphNode_t node = AddWriteNode(graph, nullptr, 0, out, 1);

    RecaptureLog log;
    hipGraphRecaptureCallbackData callback{AcceptRecapture, &log};
    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, &callback));
    HIP_CHECK(LaunchWrite(s, out, 2));
    hipGraph_t returned = nullptr;
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == graph);
    REQUIRE(log.calls == 1);
    REQUIRE(log.nodes.at(0) == node);
    RequireGraphLaunch(graph, s, out, {2});
    HIP_CHECK(hipGraphDestroy(graph));
  }

  SECTION("Clone") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(EnqueueChain(s, out, 1, 2));
    HIP_CHECK(hipStreamEndCapture(s, &graph));
    hipGraph_t clone = nullptr;
    HIP_CHECK(hipGraphClone(&clone, graph));

    RecaptureLog log;
    hipGraphRecaptureCallbackData callback{AcceptRecapture, &log};
    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, clone, &callback));
    HIP_CHECK(EnqueueChain(s, out, 11, 22));
    hipGraph_t returned = nullptr;
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == clone);
    REQUIRE(log.calls == 2);
    RequireGraphLaunch(clone, s, out, {11, 22});
    RequireGraphLaunch(graph, s, out, {1, 2});
    HIP_CHECK(hipGraphDestroy(clone));
    HIP_CHECK(hipGraphDestroy(graph));
  }

  SECTION("Repeated recapture") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(LaunchWrite(s, out, 1));
    HIP_CHECK(hipStreamEndCapture(s, &graph));

    RecaptureLog accepted;
    RecaptureLog rejected;
    hipGraphRecaptureCallbackData accept{AcceptRecapture, &accepted};
    hipGraphRecaptureCallbackData reject{RejectRecapture, &rejected};

    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, &accept));
    HIP_CHECK(LaunchWrite(s, out, 2));
    hipGraph_t returned = nullptr;
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == graph);
    RequireGraphLaunch(graph, s, out, {2});

    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, &accept));
    HIP_CHECK(LaunchWrite(s, out, 3));
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == graph);
    RequireGraphLaunch(graph, s, out, {3});

    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, &reject));
    HIP_CHECK_ERROR(LaunchWrite(s, out, 4), hipErrorGraphRecaptureFailure);
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == graph);
    RequireGraphLaunch(graph, s, out, {3});

    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, &accept));
    HIP_CHECK(LaunchWrite(s, out, 5));
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == graph);
    RequireGraphLaunch(graph, s, out, {5});

    REQUIRE(accepted.calls == 3);
    REQUIRE(rejected.calls == 1);
    HIP_CHECK(hipGraphDestroy(graph));
  }

  SECTION("Embedded child graph") {
    hipGraph_t child = nullptr;
    HIP_CHECK(hipGraphCreate(&child, 0));
    AddWriteNode(child, nullptr, 0, out, 1);
    hipGraph_t parent = nullptr;
    hipGraphNode_t childNode = nullptr;
    HIP_CHECK(hipGraphCreate(&parent, 0));
    HIP_CHECK(hipGraphAddChildGraphNode(&childNode, parent, nullptr, 0, child));
    HIP_CHECK(hipGraphDestroy(child));
    hipGraph_t embedded = nullptr;
    HIP_CHECK(hipGraphChildGraphNodeGetGraph(childNode, &embedded));

    RecaptureLog log;
    hipGraphRecaptureCallbackData callback{AcceptRecapture, &log};
    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, embedded, &callback));
    HIP_CHECK(LaunchWrite(s, out, 2));
    hipGraph_t returned = nullptr;
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == embedded);
    REQUIRE(log.calls == 1);
    RequireGraphLaunch(parent, s, out, {2});
    HIP_CHECK(hipGraphDestroy(parent));
  }

  SECTION("User objects are released at begin") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(LaunchWrite(s, out, 1));
    HIP_CHECK(hipStreamEndCapture(s, &graph));

    g_userObjectsDestroyed = 0;
    hipUserObject_t before = nullptr;
    HIP_CHECK(hipUserObjectCreate(&before, nullptr, DestroyUserObject, 1,
                                  hipUserObjectNoDestructorSync));
    HIP_CHECK(hipGraphRetainUserObject(graph, before, 1, hipGraphUserObjectMove));

    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, nullptr));
    REQUIRE(WaitForUserObjectsDestroyed(1));
    hipUserObject_t during = nullptr;
    HIP_CHECK(hipUserObjectCreate(&during, nullptr, DestroyUserObject, 1,
                                  hipUserObjectNoDestructorSync));
    HIP_CHECK(hipGraphRetainUserObject(graph, during, 1, hipGraphUserObjectMove));
    HIP_CHECK(LaunchWrite(s, out, 1));
    hipGraph_t returned = nullptr;
    HIP_CHECK(hipStreamEndCapture(s, &returned));
    REQUIRE(returned == graph);
    REQUIRE(g_userObjectsDestroyed.load() == 1);

    HIP_CHECK(hipGraphDestroy(graph));
    REQUIRE(WaitForUserObjectsDestroyed(2));
  }
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
  LinearAllocGuard<int> output(LinearAllocs::hipMalloc, kBytes);
  StreamGuard stream{Streams::created};
  StreamGuard side{Streams::created};
  EventsGuard events{2};
  hipStream_t s = stream.stream();
  int* out = output.ptr();

  SECTION("Manual empty node is never matched") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipGraphCreate(&graph, 0));
    hipGraphNode_t kernel = AddWriteNode(graph, nullptr, 0, out, 1);
    hipGraphNode_t empty = nullptr;
    HIP_CHECK(hipGraphAddEmptyNode(&empty, graph, &kernel, 1));

    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, nullptr));
    HIP_CHECK(LaunchWrite(s, out, 1));
    hipGraph_t returned = nullptr;
    HIP_CHECK_ERROR(EndRecapture(s, &returned), hipErrorGraphRecaptureFailure);
    REQUIRE(returned == nullptr);
  }

  SECTION("Creation order differs from dependency order") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipGraphCreate(&graph, 0));
    hipGraphNode_t b = AddWriteNode(graph, nullptr, 0, out + 1, 2);
    hipGraphNode_t a = AddWriteNode(graph, nullptr, 0, out, 1);
    HIP_CHECK(hipGraphAddDependencies(graph, &a, &b, 1));

    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, nullptr));
    (void)EnqueueChain(s, out, 1, 2);
    hipGraph_t returned = nullptr;
    HIP_CHECK_ERROR(EndRecapture(s, &returned), hipErrorGraphRecaptureFailure);
    REQUIRE(returned == nullptr);
  }

  SECTION("hipGraphAddNode") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(EnqueueChain(s, out, 1, 2));
    HIP_CHECK(hipStreamEndCapture(s, &graph));
    const std::vector<hipGraphNode_t> nodes = GetNodes(graph);
    REQUIRE(nodes.size() == 2);

    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, nullptr));
    HIP_CHECK(LaunchWrite(s, out, 1));
    int value = 3;
    void* args[] = {&out, &value};
    hipGraphNodeParams params{};
    params.type = hipGraphNodeTypeKernel;
    params.kernel.func = reinterpret_cast<void*>(WriteValue);
    params.kernel.gridDim = dim3(1);
    params.kernel.blockDim = dim3(1);
    params.kernel.kernelParams = args;
    hipGraphNode_t added = nullptr;
    HIP_CHECK_ERROR(hipGraphAddNode(&added, graph, nullptr, 0, &params),
                    hipErrorGraphRecaptureFailure);
    HIP_CHECK(LaunchWrite(s, out + 1, 2));
    hipGraph_t returned = nullptr;
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == graph);
    REQUIRE(GetNodes(graph) == nodes);
    RequireGraphLaunch(graph, s, out, {1, 2});
    HIP_CHECK(hipGraphDestroy(graph));
  }

  SECTION("hipGraphAddDependencies") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipGraphCreate(&graph, 0));
    hipGraphNode_t a = AddWriteNode(graph, nullptr, 0, out, 1);
    hipGraphNode_t b = AddWriteNode(graph, nullptr, 0, out + 1, 2);

    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, nullptr));
    HIP_CHECK(hipEventRecord(events[0], s));
    HIP_CHECK(hipStreamWaitEvent(side.stream(), events[0], 0));
    HIP_CHECK(LaunchWrite(s, out, 1));
    // The edge is added, so b now depends on a and the side-stream launch no longer matches.
    HIP_CHECK(hipGraphAddDependencies(graph, &a, &b, 1));
    HIP_CHECK_ERROR(LaunchWrite(side.stream(), out + 1, 2), hipErrorGraphRecaptureFailure);
    HIP_CHECK(hipEventRecord(events[1], side.stream()));
    HIP_CHECK(hipStreamWaitEvent(s, events[1], 0));
    hipGraph_t returned = nullptr;
    HIP_CHECK_ERROR(EndRecapture(s, &returned), hipErrorGraphRecaptureFailure);
    REQUIRE(returned == nullptr);
  }

  SECTION("hipGraphRemoveDependencies") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(EnqueueChain(s, out, 1, 2));
    HIP_CHECK(hipStreamEndCapture(s, &graph));
    const std::vector<hipGraphNode_t> nodes = GetNodes(graph);
    REQUIRE(nodes.size() == 2);

    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, nullptr));
    HIP_CHECK(LaunchWrite(s, out, 1));
    // The edge is removed, so the second launch, which depends on the first, no longer matches.
    HIP_CHECK(hipGraphRemoveDependencies(graph, &nodes[0], &nodes[1], 1));
    HIP_CHECK_ERROR(LaunchWrite(s, out + 1, 2), hipErrorGraphRecaptureFailure);
    hipGraph_t returned = nullptr;
    HIP_CHECK_ERROR(EndRecapture(s, &returned), hipErrorGraphRecaptureFailure);
    REQUIRE(returned == nullptr);
  }

  SECTION("hipGraphDestroyNode") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(EnqueueChain(s, out, 1, 2));
    HIP_CHECK(hipStreamEndCapture(s, &graph));
    const std::vector<hipGraphNode_t> nodes = GetNodes(graph);
    REQUIRE(nodes.size() == 2);

    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, nullptr));
    HIP_CHECK(LaunchWrite(s, out, 1));
    HIP_CHECK_ERROR(hipGraphDestroyNode(nodes[1]), hipErrorIllegalState);
    HIP_CHECK(LaunchWrite(s, out + 1, 2));
    hipGraph_t returned = nullptr;
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == graph);
    REQUIRE(GetNodes(graph) == nodes);
    RequireGraphLaunch(graph, s, out, {1, 2});
    HIP_CHECK(hipGraphDestroy(graph));
  }

  SECTION("hipGraphKernelNodeSetParams") {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(EnqueueChain(s, out, 1, 2));
    HIP_CHECK(hipStreamEndCapture(s, &graph));
    const std::vector<hipGraphNode_t> nodes = GetNodes(graph);
    REQUIRE(nodes.size() == 2);

    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, nullptr));
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
    hipGraph_t returned = nullptr;
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == graph);
    RequireGraphLaunch(graph, s, out, {1, 2});
    HIP_CHECK(hipGraphDestroy(graph));
  }
}

/**
 * Test Description
 * ------------------------
 *  - Capture rules shared with normal capture: hipMalloc in global mode invalidates, as does
 *    synchronizing the capturing stream. An unjoined side stream is allowed, and the
 *    per-thread stream can be recaptured. Capture info reports the input graph and the
 *    original node handles. Begin rejects a null graph, an invalid mode, a null or legacy
 *    stream, and a stream that is already capturing. An empty graph accepts an empty
 *    recapture and rejects an added node. A zero-size memset creates no node. Forked streams
 *    recaptured from two threads match whether the threads are ordered or concurrent.
 * Test source
 * ------------------------
 *  - unit/graph/hipStreamBeginRecaptureToGraph.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipStreamBeginRecaptureToGraph_CaptureRules) {
  LinearAllocGuard<int> output(LinearAllocs::hipMalloc, kBytes);
  StreamGuard stream{Streams::created};
  StreamGuard side{Streams::created};
  EventsGuard events{2};
  hipStream_t s = stream.stream();
  int* out = output.ptr();

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipStreamBeginCapture(s, kMode));
  HIP_CHECK(LaunchWrite(s, out, 1));
  HIP_CHECK(hipStreamEndCapture(s, &graph));

  SECTION("Global mode, hipMalloc on the capturing thread invalidates") {
    void* allocation = nullptr;
    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, hipStreamCaptureModeGlobal, graph, nullptr));
    HIP_CHECK(LaunchWrite(s, out, 1));
    HIP_CHECK_ERROR(hipMalloc(&allocation, 16), hipErrorStreamCaptureUnsupported);
    hipGraph_t returned = nullptr;
    HIP_CHECK_ERROR(EndRecapture(s, &returned), hipErrorStreamCaptureInvalidated);
    REQUIRE(returned == nullptr);
    graph = nullptr;
  }

  SECTION("Stream synchronize invalidates") {
    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, nullptr));
    HIP_CHECK(LaunchWrite(s, out, 1));
    HIP_CHECK_ERROR(hipStreamSynchronize(s), hipErrorStreamCaptureUnsupported);
    hipGraph_t returned = nullptr;
    HIP_CHECK_ERROR(EndRecapture(s, &returned), hipErrorStreamCaptureInvalidated);
    REQUIRE(returned == nullptr);
    graph = nullptr;
  }

  SECTION("Unjoined side stream is allowed") {
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(EnqueueForked(s, side.stream(), events[0], events[1], out, 10, 20, false));
    HIP_CHECK(hipStreamEndCapture(s, &graph));

    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, nullptr));
    HIP_CHECK(hipEventRecord(events[0], s));
    HIP_CHECK(hipStreamWaitEvent(side.stream(), events[0], 0));
    HIP_CHECK(LaunchWrite(s, out, 10));
    HIP_CHECK(LaunchWrite(side.stream(), out + 1, 20));
    hipGraph_t returned = nullptr;
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == graph);
    hipStreamCaptureStatus status = hipStreamCaptureStatusActive;
    HIP_CHECK(hipStreamIsCapturing(side.stream(), &status));
    REQUIRE(status == hipStreamCaptureStatusNone);
  }

  SECTION("Per-thread stream") {
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipStreamBeginCapture(hipStreamPerThread, kMode));
    HIP_CHECK(LaunchWrite(hipStreamPerThread, out, 1));
    HIP_CHECK(hipStreamEndCapture(hipStreamPerThread, &graph));

    HIP_CHECK(hipStreamBeginRecaptureToGraph(hipStreamPerThread, kMode, graph, nullptr));
    HIP_CHECK(LaunchWrite(hipStreamPerThread, out, 2));
    hipGraph_t returned = nullptr;
    HIP_CHECK(EndRecapture(hipStreamPerThread, &returned));
    REQUIRE(returned == graph);
    RequireGraphLaunch(graph, s, out, {2});
  }

  SECTION("Capture info reports the original nodes") {
    HIP_CHECK(hipGraphDestroy(graph));
    graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(EnqueueChain(s, out, 1, 2));
    HIP_CHECK(hipStreamEndCapture(s, &graph));
    const std::vector<hipGraphNode_t> nodes = GetNodes(graph);
    REQUIRE(nodes.size() == 2);

    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, nullptr));
    hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
    unsigned long long id = 0;
    hipGraph_t captureGraph = nullptr;
    const hipGraphNode_t* deps = nullptr;
    size_t count = 0;
    HIP_CHECK(hipStreamGetCaptureInfo_v2(s, &status, &id, &captureGraph, &deps, &count));
    REQUIRE(status == hipStreamCaptureStatusActive);
    REQUIRE(captureGraph == graph);
    REQUIRE(count == 0);

    HIP_CHECK(LaunchWrite(s, out, 11));
    HIP_CHECK(hipStreamGetCaptureInfo_v2(s, &status, &id, &captureGraph, &deps, &count));
    REQUIRE(status == hipStreamCaptureStatusActive);
    REQUIRE(captureGraph == graph);
    REQUIRE(count == 1);
    REQUIRE(deps[0] == nodes[0]);

    HIP_CHECK(LaunchWrite(s, out + 1, 22));
    HIP_CHECK(hipStreamGetCaptureInfo_v2(s, &status, &id, &captureGraph, &deps, &count));
    REQUIRE(status == hipStreamCaptureStatusActive);
    REQUIRE(captureGraph == graph);
    REQUIRE(count == 1);
    REQUIRE(deps[0] == nodes[1]);

    hipGraph_t returned = nullptr;
    HIP_CHECK(hipStreamEndCapture(s, &returned));
    REQUIRE(returned == graph);
  }

  SECTION("Invalid arguments") {
    HIP_CHECK_ERROR(hipStreamBeginRecaptureToGraph(s, kMode, nullptr, nullptr), hipErrorInvalidValue);
    HIP_CHECK_ERROR(
        hipStreamBeginRecaptureToGraph(s, static_cast<hipStreamCaptureMode>(99), graph, nullptr),
        hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipStreamBeginRecaptureToGraph(nullptr, kMode, graph, nullptr),
                    hipErrorStreamCaptureUnsupported);
    HIP_CHECK_ERROR(hipStreamBeginRecaptureToGraph(hipStreamLegacy, kMode, graph, nullptr),
                    hipErrorStreamCaptureUnsupported);
    hipStreamCaptureStatus status = hipStreamCaptureStatusActive;
    HIP_CHECK(hipStreamIsCapturing(s, &status));
    REQUIRE(status == hipStreamCaptureStatusNone);
  }

  SECTION("Stream already capturing") {
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK_ERROR(hipStreamBeginRecaptureToGraph(s, kMode, graph, nullptr), hipErrorIllegalState);
    hipGraph_t other = nullptr;
    HIP_CHECK(hipStreamEndCapture(s, &other));
    HIP_CHECK(hipGraphDestroy(other));
  }

  SECTION("Empty graph") {
    hipGraph_t empty = nullptr;
    HIP_CHECK(hipGraphCreate(&empty, 0));
    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, empty, nullptr));
    hipGraph_t returned = nullptr;
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == empty);

    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, empty, nullptr));
    HIP_CHECK_ERROR(LaunchWrite(s, out, 1), hipErrorGraphRecaptureFailure);
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == empty);
    REQUIRE(GetNodes(empty).empty());
    HIP_CHECK(hipGraphDestroy(empty));
  }

  SECTION("Zero-size memset creates no node") {
    HIP_CHECK(hipGraphDestroy(graph));
    graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(hipMemsetAsync(out, 1, 0, s));
    HIP_CHECK(LaunchWrite(s, out + 1, 1));
    HIP_CHECK(hipStreamEndCapture(s, &graph));
    REQUIRE(GetNodes(graph).size() == 1);

    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, nullptr));
    HIP_CHECK(hipMemsetAsync(out, 1, 0, s));
    HIP_CHECK(LaunchWrite(s, out + 1, 2));
    hipGraph_t returned = nullptr;
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == graph);
    RequireGraphLaunch(graph, s, out, {0, 2});
  }

  SECTION("Zero-size memset made non-zero adds a node") {
    HIP_CHECK(hipGraphDestroy(graph));
    graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(hipMemsetAsync(out, 1, 0, s));
    HIP_CHECK(LaunchWrite(s, out + 1, 1));
    HIP_CHECK(hipStreamEndCapture(s, &graph));

    HIP_CHECK(hipStreamBeginRecaptureToGraph(s, kMode, graph, nullptr));
    HIP_CHECK_ERROR(hipMemsetAsync(out, 1, sizeof(int), s), hipErrorGraphRecaptureFailure);
    HIP_CHECK(LaunchWrite(s, out + 1, 1));
    hipGraph_t returned = nullptr;
    HIP_CHECK(EndRecapture(s, &returned));
    REQUIRE(returned == graph);
    RequireGraphLaunch(graph, s, out, {0, 1});
  }

  SECTION("Forked streams from two threads") {
    HIP_CHECK(hipGraphDestroy(graph));
    graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(s, kMode));
    HIP_CHECK(EnqueueForked(s, side.stream(), events[0], events[1], out, 10, 20, false));
    HIP_CHECK(hipStreamEndCapture(s, &graph));

    for (const bool concurrent : {false, true}) {
      HIP_CHECK(hipStreamBeginRecaptureToGraph(s, hipStreamCaptureModeRelaxed, graph, nullptr));
      HIP_CHECK(hipEventRecord(events[0], s));
      HIP_CHECK(hipStreamWaitEvent(side.stream(), events[0], 0));
      hipError_t resultA = hipErrorUnknown;
      hipError_t resultB = hipErrorUnknown;
      {
        std::thread a([&] { resultA = LaunchWrite(s, out, 11); });
        if (!concurrent) a.join();
        std::thread b([&] { resultB = LaunchWrite(side.stream(), out + 1, 22); });
        if (concurrent) a.join();
        b.join();
      }
      HIP_CHECK(hipEventRecord(events[1], side.stream()));
      HIP_CHECK(hipStreamWaitEvent(s, events[1], 0));
      hipGraph_t returned = nullptr;
      HIP_CHECK(hipStreamEndCapture(s, &returned));
      HIP_CHECK(resultA);
      HIP_CHECK(resultB);
      REQUIRE(returned == graph);
      RequireGraphLaunch(graph, s, out, {11, 22});
    }
  }

  if (graph != nullptr) {
    HIP_CHECK(hipGraphDestroy(graph));
  }
}

#endif

/**
 * End doxygen group GraphTest.
 * @}
 */
