// Structure + perf probe for a conditional WHILE whose body is WIDE:
//   WHILE (counter < limit):
//       body:  W independent kWork kernels  ->  kTick (counter++, set cond)
//
// Every other cond benchmark here has a single-packet body, so the block
// completion signal is always folded into that one packet (inline_signal) and
// the trailing BARRIER_AND fallback is never exercised. A wide body is the
// first case that can produce a MULTI-packet block, which is what decides:
//   - folded signal vs a trailing BARRIER_AND packet, and
//   - whether the W dispatches serialize (barrier bit) or overlap.
//
// Run with AMD_LOG_LEVEL=4 and grep "Built multi-block" to see how the body
// was split into blocks/packets.
#include "common.h"
#include <chrono>
#include <vector>

__global__ void kWork(float* buf) { buf[threadIdx.x] += 1.0f; }

__global__ void kTick(hipGraphConditionalHandle h, int* counter, const int* limit) {
  *counter += 1;
  hipGraphSetConditional(h, (*counter < *limit) ? 1 : 0);
}

static double now_us() {
  using namespace std::chrono;
  return duration<double, std::micro>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
  const int width  = (argc > 1) ? atoi(argv[1]) : 10;   // parallel kernels in body
  const int limit  = (argc > 2) ? atoi(argv[2]) : 1000; // iterations per launch
  const int iters  = (argc > 3) ? atoi(argv[3]) : 100;  // launches to average
  const int warmup = 20;

  int *counter, *limitD;
  CHECK(hipMalloc(&counter, sizeof(int)));
  CHECK(hipMalloc(&limitD, sizeof(int)));
  CHECK(hipMemcpy(limitD, &limit, sizeof(int), hipMemcpyHostToDevice));

  // Separate buffer per parallel kernel: no overlap, so the runtime's memory
  // dependency tracking has no reason to force the syncing (barrier) header.
  std::vector<float*> bufs(width);
  for (int i = 0; i < width; ++i) {
    CHECK(hipMalloc(&bufs[i], 64 * sizeof(float)));
    CHECK(hipMemset(bufs[i], 0, 64 * sizeof(float)));
  }

  hipGraph_t graph;
  CHECK(hipGraphCreate(&graph, 0));

  hipGraphConditionalHandle whileHandle;
  CHECK(hipGraphConditionalHandleCreate(&whileHandle, graph, 1, 0));

  hipGraph_t body;
  CHECK(hipGraphCreate(&body, 0));

  // Arg arrays must outlive graph construction.
  std::vector<std::vector<void*>> work_args(width);
  std::vector<hipGraphNode_t> par(width);
  for (int i = 0; i < width; ++i) {
    work_args[i] = {&bufs[i]};
    par[i] = addKernel(body, (void*)kWork, dim3(1), dim3(64), work_args[i].data(), nullptr, 0);
  }
  void* tArgs[] = {&whileHandle, &counter, &limitD};
  addKernel(body, (void*)kTick, dim3(1), dim3(1), tArgs, par.data(), par.size());

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

  float gpu_ms_sum = 0.0f;
  double t0 = now_us();
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
  double wall = (now_us() - t0) / iters;
  double gpu_us = (gpu_ms_sum / iters) * 1000.0;

  printf("cond-wide: width=%d limit(iters)=%d launches=%d\n", width, limit, iters);
  printf("  wall  us/launch : %9.2f\n", wall);
  printf("  GPU   us/launch : %9.2f   (%.4f us/iter)\n", gpu_us, gpu_us / limit);
  printf("  per-iter/kernel : %9.4f us\n", gpu_us / limit / (width + 1));

  CHECK(hipEventDestroy(evStart));
  CHECK(hipEventDestroy(evStop));
  CHECK(hipStreamDestroy(stream));
  CHECK(hipGraphExecDestroy(exec));
  CHECK(hipGraphDestroy(body));
  CHECK(hipGraphDestroy(graph));
  for (int i = 0; i < width; ++i) CHECK(hipFree(bufs[i]));
  CHECK(hipFree(counter));
  CHECK(hipFree(limitD));
  return 0;
}
