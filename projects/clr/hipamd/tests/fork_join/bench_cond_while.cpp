// Perf check for a GPU-side conditional WHILE loop (AIRUNTIME-1379 scenario).
//
//   WHILE (counter < limit):
//       body:  counter++ ; setConditional(counter < limit)
//
// The body is deliberately lightweight so per-iteration control-flow overhead
// dominates -- this is exactly the case the GPU-resident scheduler targets
// (no host/PCIe round-trip per iteration).
//
// Same graph is launched many times back-to-back; we report average wall
// latency per launch and the derived per-iteration cost. On-GPU time is also
// captured via HIP events.
//
// Run once with the default (GPU-resident command-buffer) path and once with
//   DEBUG_HIP_GRAPH_CLASSIC_PATH=1
// to get the host-driven baseline for comparison.
#include "common.h"
#include <chrono>

__global__ void kBody(hipGraphConditionalHandle h, int* counter, const int* limit) {
  *counter += 1;
  hipGraphSetConditional(h, (*counter < *limit) ? 1 : 0);
}

static double now_us() {
  using namespace std::chrono;
  return duration<double, std::micro>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
  const int limit  = (argc > 1) ? atoi(argv[1]) : 1000;  // iterations per launch
  const int iters  = (argc > 2) ? atoi(argv[2]) : 200;   // launches to average
  const int warmup = 50;

  int *counter, *limitD;
  CHECK(hipMalloc(&counter, sizeof(int)));
  CHECK(hipMalloc(&limitD, sizeof(int)));
  CHECK(hipMemcpy(limitD, &limit, sizeof(int), hipMemcpyHostToDevice));

  hipGraph_t graph;
  CHECK(hipGraphCreate(&graph, 0));

  hipGraphConditionalHandle whileHandle;
  CHECK(hipGraphConditionalHandleCreate(&whileHandle, graph, 1, 0));

  hipGraph_t body;
  CHECK(hipGraphCreate(&body, 0));
  void* bArgs[] = {&whileHandle, &counter, &limitD};
  addKernel(body, (void*)kBody, dim3(1), dim3(1), bArgs, nullptr, 0);

  hipGraphNode_t whileNode;
  CHECK(hipGraphAddConditionalNode(&whileNode, graph, nullptr, 0, whileHandle,
                                   hipGraphCondTypeWhile, 1, &body, 0));

  hipGraphExec_t exec;
  CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  hipStream_t stream;
  CHECK(hipStreamCreate(&stream));
  hipEvent_t evStart, evStop;
  CHECK(hipEventCreate(&evStart));
  CHECK(hipEventCreate(&evStop));

  // Correctness + warmup.
  for (int i = 0; i < warmup; ++i) {
    CHECK(hipMemset(counter, 0, sizeof(int)));
    CHECK(hipGraphLaunch(exec, stream));
    CHECK(hipStreamSynchronize(stream));
  }
  {
    int hc = -1;
    CHECK(hipMemcpy(&hc, counter, sizeof(int), hipMemcpyDeviceToHost));
    if (hc != limit) {
      fprintf(stderr, "MISMATCH: counter=%d exp=%d\n", hc, limit);
      return 1;
    }
  }

  // Timed: per-launch wall latency, split host-enqueue vs GPU-wait.
  double host_sum = 0.0, wait_sum = 0.0;
  double t0 = now_us();
  for (int i = 0; i < iters; ++i) {
    CHECK(hipMemset(counter, 0, sizeof(int)));
    double a = now_us();
    CHECK(hipGraphLaunch(exec, stream));
    double b = now_us();
    CHECK(hipStreamSynchronize(stream));
    double c = now_us();
    host_sum += (b - a);
    wait_sum += (c - b);
  }
  double wall = (now_us() - t0) / iters;
  double host = host_sum / iters;
  double wait = wait_sum / iters;

  // GPU-only time for the loop via events.
  float gpu_ms_sum = 0.0f;
  for (int i = 0; i < iters; ++i) {
    CHECK(hipMemset(counter, 0, sizeof(int)));
    CHECK(hipEventRecord(evStart, stream));
    CHECK(hipGraphLaunch(exec, stream));
    CHECK(hipEventRecord(evStop, stream));
    CHECK(hipEventSynchronize(evStop));
    float ms = 0.0f;
    CHECK(hipEventElapsedTime(&ms, evStart, evStop));
    gpu_ms_sum += ms;
  }
  double gpu_us = (gpu_ms_sum / iters) * 1000.0;

  printf("cond-while: limit(iters)=%d  launches=%d\n", limit, iters);
  printf("  wall  us/launch : %9.2f   (%.4f us/iter)\n", wall, wall / limit);
  printf("    host-enqueue  : %9.2f\n", host);
  printf("    GPU-wait      : %9.2f\n", wait);
  printf("  GPU   us/launch : %9.2f   (%.4f us/iter)\n", gpu_us, gpu_us / limit);

  CHECK(hipEventDestroy(evStart));
  CHECK(hipEventDestroy(evStop));
  CHECK(hipStreamDestroy(stream));
  CHECK(hipGraphExecDestroy(exec));
  CHECK(hipGraphDestroy(body));
  CHECK(hipGraphDestroy(graph));
  return 0;
}
