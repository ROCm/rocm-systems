/*
 * Standalone test for graph profiling timestamp correctness.
 * Build: hipcc --offload-arch=gfx942 -o graphTimingTest hipGraphProfilingTimestamps_standalone.cpp
 * Run:   ./graphTimingTest
 */

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t err = (call);                                                   \
    if (err != hipSuccess) {                                                   \
      fprintf(stderr, "HIP error %d (%s) at %s:%d\n", err,                    \
              hipGetErrorString(err), __FILE__, __LINE__);                      \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static constexpr int kN = 1024;

__global__ void busyKernel(float* out, const float* in, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float val = in[idx];
    for (int i = 0; i < 50; ++i) {
      val = val * 1.001f + 0.001f;
    }
    out[idx] = val;
  }
}

// Test 1: Verify elapsed time from a graph launch with many kernels
// Events are recorded OUTSIDE the graph (around the launch) to measure wall time.
static bool testGraphTimingWithEvents(int numKernels) {
  printf("  [GraphTimingWithEvents] numKernels=%d ... ", numKernels);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  float* d_in;
  float* d_out;
  HIP_CHECK(hipMalloc(&d_in, kN * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out, kN * sizeof(float)));

  std::vector<float> h_in(kN);
  std::iota(h_in.begin(), h_in.end(), 1.0f);
  HIP_CHECK(hipMemcpy(d_in, h_in.data(), kN * sizeof(float), hipMemcpyHostToDevice));

  hipEvent_t evStart, evEnd;
  HIP_CHECK(hipEventCreate(&evStart));
  HIP_CHECK(hipEventCreate(&evEnd));

  hipGraph_t graph;
  hipGraphExec_t graphExec;

  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  for (int i = 0; i < numKernels; ++i) {
    busyKernel<<<dim3((kN + 255) / 256), dim3(256), 0, stream>>>(d_out, d_in, kN);
  }
  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  bool pass = true;
  constexpr int kIterations = 10;
  for (int iter = 0; iter < kIterations; ++iter) {
    HIP_CHECK(hipEventRecord(evStart, stream));
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipEventRecord(evEnd, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    float elapsed = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed, evStart, evEnd));

    if (elapsed <= 0.0f || elapsed >= 60000.0f) {
      printf("FAIL (iter %d: elapsed=%.4f ms)\n", iter, elapsed);
      pass = false;
      break;
    }
  }
  if (pass) printf("PASS\n");

  HIP_CHECK(hipEventDestroy(evStart));
  HIP_CHECK(hipEventDestroy(evEnd));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipStreamDestroy(stream));
  return pass;
}

// Test 2: Verify per-kernel timing using hipGraphAddEventRecordNode
// Events are embedded as graph nodes between kernels.
static bool testPerKernelTiming(int numKernels) {
  printf("  [PerKernelTiming] numKernels=%d ... ", numKernels);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  float* d_in;
  float* d_out;
  HIP_CHECK(hipMalloc(&d_in, kN * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out, kN * sizeof(float)));

  std::vector<float> h_in(kN);
  std::iota(h_in.begin(), h_in.end(), 1.0f);
  HIP_CHECK(hipMemcpy(d_in, h_in.data(), kN * sizeof(float), hipMemcpyHostToDevice));

  int numEvents = numKernels + 1;
  std::vector<hipEvent_t> events(numEvents);
  for (int i = 0; i < numEvents; ++i) {
    HIP_CHECK(hipEventCreate(&events[i]));
  }

  // Build graph manually with interleaved event record nodes and kernel nodes
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipKernelNodeParams kNodeParams = {};
  kNodeParams.func = reinterpret_cast<void*>(busyKernel);
  kNodeParams.gridDim = dim3((kN + 255) / 256);
  kNodeParams.blockDim = dim3(256);
  kNodeParams.sharedMemBytes = 0;
  int n = kN;
  void* kArgs[] = {&d_out, &d_in, &n};
  kNodeParams.kernelParams = kArgs;
  kNodeParams.extra = nullptr;

  // Chain: eventRec[0] -> kernel[0] -> eventRec[1] -> kernel[1] -> ... -> eventRec[N]
  std::vector<hipGraphNode_t> eventNodes(numEvents);
  std::vector<hipGraphNode_t> kernelNodes(numKernels);

  // First event node (no dependencies)
  HIP_CHECK(hipGraphAddEventRecordNode(&eventNodes[0], graph, nullptr, 0, events[0]));

  for (int i = 0; i < numKernels; ++i) {
    // Kernel depends on previous event
    HIP_CHECK(hipGraphAddKernelNode(&kernelNodes[i], graph, &eventNodes[i], 1, &kNodeParams));
    // Event after kernel
    HIP_CHECK(hipGraphAddEventRecordNode(&eventNodes[i + 1], graph, &kernelNodes[i], 1,
                                          events[i + 1]));
  }

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  bool pass = true;
  int negativeCount = 0;
  for (int i = 0; i < numKernels; ++i) {
    float elapsed = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed, events[i], events[i + 1]));
    if (elapsed < 0.0f) {
      printf("FAIL (kernel %d: elapsed=%.4f ms < 0)\n", i, elapsed);
      negativeCount++;
    } else if (elapsed >= 10000.0f) {
      printf("FAIL (kernel %d: elapsed=%.4f ms >= 10s)\n", i, elapsed);
      pass = false;
      break;
    }
  }
  if (negativeCount > 0) pass = false;

  if (pass) {
    float overallElapsed = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&overallElapsed, events[0], events[numKernels]));
    if (overallElapsed <= 0.0f || overallElapsed >= 60000.0f) {
      printf("FAIL (overall=%.4f ms)\n", overallElapsed);
      pass = false;
    }
  }
  if (pass) printf("PASS\n");

  for (int i = 0; i < numEvents; ++i) {
    HIP_CHECK(hipEventDestroy(events[i]));
  }
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipStreamDestroy(stream));
  return pass;
}

// Test 3: Verify timing consistency across repeated graph launches
static bool testTimingConsistency(int numKernels) {
  printf("  [TimingConsistency] numKernels=%d ... ", numKernels);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  float* d_in;
  float* d_out;
  HIP_CHECK(hipMalloc(&d_in, kN * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out, kN * sizeof(float)));

  std::vector<float> h_in(kN);
  std::iota(h_in.begin(), h_in.end(), 1.0f);
  HIP_CHECK(hipMemcpy(d_in, h_in.data(), kN * sizeof(float), hipMemcpyHostToDevice));

  hipEvent_t evStart, evEnd;
  HIP_CHECK(hipEventCreate(&evStart));
  HIP_CHECK(hipEventCreate(&evEnd));

  hipGraph_t graph;
  hipGraphExec_t graphExec;

  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  for (int i = 0; i < numKernels; ++i) {
    busyKernel<<<dim3((kN + 255) / 256), dim3(256), 0, stream>>>(d_out, d_in, kN);
  }
  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  // Warmup
  HIP_CHECK(hipEventRecord(evStart, stream));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipEventRecord(evEnd, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  constexpr int kRuns = 20;
  std::vector<float> timings(kRuns);
  bool pass = true;
  for (int i = 0; i < kRuns; ++i) {
    HIP_CHECK(hipEventRecord(evStart, stream));
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
    HIP_CHECK(hipEventRecord(evEnd, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    float elapsed = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed, evStart, evEnd));
    timings[i] = elapsed;
    if (elapsed <= 0.0f) {
      printf("FAIL (run %d: elapsed=%.4f ms <= 0)\n", i, elapsed);
      pass = false;
      break;
    }
  }

  if (pass) {
    std::sort(timings.begin(), timings.end());
    float median = timings[kRuns / 2];
    for (int i = 0; i < kRuns; ++i) {
      if (timings[i] > median * 100.0f || timings[i] < median * 0.01f) {
        printf("FAIL (run %d: %.4f ms, median=%.4f ms, ratio=%.2fx)\n",
               i, timings[i], median, timings[i] / median);
        pass = false;
        break;
      }
    }
    if (pass) printf("PASS (median=%.4f ms)\n", median);
  }

  HIP_CHECK(hipEventDestroy(evStart));
  HIP_CHECK(hipEventDestroy(evEnd));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipStreamDestroy(stream));
  return pass;
}

int main() {
  printf("=== Graph Profiling Timestamp Tests ===\n\n");

  int failures = 0;

  printf("Test 1: Graph timing with events (small, no signal recycling)\n");
  if (!testGraphTimingWithEvents(8)) failures++;

  printf("Test 2: Graph timing with events (large, forces signal recycling)\n");
  if (!testGraphTimingWithEvents(320)) failures++;

  printf("Test 3: Per-kernel timing (small)\n");
  if (!testPerKernelTiming(8)) failures++;

  printf("Test 4: Per-kernel timing (large, forces signal recycling)\n");
  if (!testPerKernelTiming(128)) failures++;

  printf("Test 5: Timing consistency (small)\n");
  if (!testTimingConsistency(8)) failures++;

  printf("Test 6: Timing consistency (large, forces signal recycling)\n");
  if (!testTimingConsistency(320)) failures++;

  printf("\n=== Results: %d/%d passed ===\n", 6 - failures, 6);
  return failures > 0 ? 1 : 0;
}
