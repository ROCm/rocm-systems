/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>

#include <hip_test_aqllog_trace_region.hh>

#include "hipGraphInstantiateNullStreamFence_common.hh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>

namespace {

void check(hipError_t status, const char* expression, int line) {
  if (status == hipSuccess) {
    return;
  }
  std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, line, expression,
               hipGetErrorString(status));
  std::exit(EXIT_FAILURE);
}

#define CHECK(expression) check((expression), #expression, __LINE__)

constexpr size_t kCopyBytes = 4096;

__global__ void writeValue(float* out, float value) {
  if (threadIdx.x == 0) {
    out[0] = value;
  }
}

void launchWriteKernel(hipStream_t stream, float* buffer) {
  hipLaunchKernelGGL(writeValue, dim3(1), dim3(1), 0, stream, buffer, 1.0f);
  CHECK(hipGetLastError());
}

// Complete and clean the command before probing so the scenarios differ only
// by whether graph instantiation occurs.
void primeNullStream(float* buffer) {
  launchWriteKernel(nullptr, buffer);

  hipEvent_t event = nullptr;
  CHECK(hipEventCreateWithFlags(&event, hipEventDefault));
  CHECK(hipEventRecord(event, nullptr));

  hipError_t status = hipErrorNotReady;
  while (status == hipErrorNotReady) {
    status = hipEventQuery(event);
  }
  CHECK(status);
  // The polled hipErrorNotReady results leave the sticky error set.
  static_cast<void>(hipGetLastError());
  CHECK(hipEventDestroy(event));
}

void instantiateKernelGraph(float* buffer, hipGraph_t* graph, hipGraphExec_t* exec) {
  CHECK(hipGraphCreate(graph, 0));

  float value = 2.0f;
  void* args[] = {&buffer, &value};
  hipKernelNodeParams params{};
  params.func = reinterpret_cast<void*>(writeValue);
  params.gridDim = dim3(1);
  params.blockDim = dim3(1);
  params.sharedMemBytes = 0;
  params.kernelParams = args;
  params.extra = nullptr;

  hipGraphNode_t node = nullptr;
  CHECK(hipGraphAddKernelNode(&node, *graph, nullptr, 0, &params));
  CHECK(hipGraphInstantiate(exec, *graph, nullptr, nullptr, 0));
}

void runFenceScenario(bool instantiate_graph) {
  float* buffer = nullptr;
  CHECK(hipMalloc(&buffer, sizeof(float)));
  primeNullStream(buffer);

  hipGraph_t graph = nullptr;
  hipGraphExec_t exec = nullptr;
  if (instantiate_graph) {
    instantiateKernelGraph(buffer, &graph, &exec);
  }

  // The graph is not launched because only packet construction is under test.
  hipStream_t stream = nullptr;
  CHECK(hipStreamCreate(&stream));
  hip::aqllog::BeginRegion(graph_fence_test::kFenceRegion);
  launchWriteKernel(stream, buffer);
  launchWriteKernel(stream, buffer);
  CHECK(hipStreamSynchronize(stream));
  hip::aqllog::EndRegion(graph_fence_test::kFenceRegion);
  CHECK(hipStreamDestroy(stream));

  if (instantiate_graph) {
    CHECK(hipGraphExecDestroy(exec));
    CHECK(hipGraphDestroy(graph));
  }
  CHECK(hipFree(buffer));
}

void runCapturedSystemScopeScenario() {
  // A non-coherent pinned source makes the captured blit request system scope.
  void* host = nullptr;
  void* device = nullptr;
  CHECK(hipHostMalloc(&host, kCopyBytes, hipHostMallocNonCoherent));
  CHECK(hipMalloc(&device, kCopyBytes));
  std::memset(host, 1, kCopyBytes);

  hipGraph_t graph = nullptr;
  CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t node = nullptr;
  CHECK(hipGraphAddMemcpyNode1D(&node, graph, nullptr, 0, device, host, kCopyBytes,
                                hipMemcpyHostToDevice));
  hipGraphExec_t exec = nullptr;
  CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  hipStream_t stream = nullptr;
  CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  hip::aqllog::BeginRegion(graph_fence_test::kCapturedSystemScopeRegion);
  CHECK(hipGraphLaunch(exec, stream));
  CHECK(hipStreamSynchronize(stream));
  hip::aqllog::EndRegion(graph_fence_test::kCapturedSystemScopeRegion);

  CHECK(hipStreamDestroy(stream));
  CHECK(hipGraphExecDestroy(exec));
  CHECK(hipGraphDestroy(graph));
  CHECK(hipFree(device));
  CHECK(hipHostFree(host));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <scenario>\n", argv[0]);
    return EXIT_FAILURE;
  }

  const std::string_view scenario{argv[1]};
  if (scenario == graph_fence_test::kScenarioGraph) {
    runFenceScenario(true);
  } else if (scenario == graph_fence_test::kScenarioNoGraph) {
    runFenceScenario(false);
  } else if (scenario == graph_fence_test::kScenarioCapturedSystemScope) {
    runCapturedSystemScopeScenario();
  } else {
    std::fprintf(stderr, "unknown graph-fence scenario: %.*s\n", static_cast<int>(scenario.size()),
                 scenario.data());
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
