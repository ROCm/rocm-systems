// Perf check for a plain (no control-flow) kernel graph launched repeatedly.
//
// Captures NKERNEL back-to-back kernels into a graph (single linear segment),
// instantiates once, then launches many times. This is the command-buffer
// scenario: each launch is a SINGLE scheduler-kernel dispatch that copies all
// baked kernel packets into the internal queue device-side (no host re-enqueue
// of N packets per launch).
//
// Compare:
//   default                          -> GPU-resident command buffer, internal
//                                       queue in device memory (HIP_GRAPH_INTERNALQ_DEVMEM=1)
//   HIP_GRAPH_INTERNALQ_DEVMEM=0     -> internal queue in system memory
//   DEBUG_HIP_GRAPH_CLASSIC_PATH=1   -> host-driven baseline
#include "common.h"
#include <chrono>

__global__ void kSimple(float* out, const float* in, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) out[idx] = 5.34f * in[idx];
}

static double now_us() {
  using namespace std::chrono;
  return duration<double, std::micro>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
  const int nkernel = (argc > 1) ? atoi(argv[1]) : 50;   // kernels per graph
  const int iters   = (argc > 2) ? atoi(argv[2]) : 200;  // launches to average
  const int warmup  = 50;
  const int N = 1024 * 1024;

  float *in_d, *out_d;
  CHECK(hipMalloc(&in_d, N * sizeof(float)));
  CHECK(hipMalloc(&out_d, N * sizeof(float)));
  CHECK(hipMemset(in_d, 0, N * sizeof(float)));

  hipStream_t stream;
  CHECK(hipStreamCreate(&stream));

  dim3 grid(N / 512, 1, 1), block(512, 1, 1);

  // Capture NKERNEL linear kernels into a graph.
  hipGraph_t graph;
  CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  for (int k = 0; k < nkernel; ++k) {
    kSimple<<<grid, block, 0, stream>>>(out_d, in_d, N);
  }
  CHECK(hipStreamEndCapture(stream, &graph));

  hipGraphExec_t exec;
  CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  hipEvent_t evStart, evStop;
  CHECK(hipEventCreate(&evStart));
  CHECK(hipEventCreate(&evStop));

  for (int i = 0; i < warmup; ++i) {
    CHECK(hipGraphLaunch(exec, stream));
    CHECK(hipStreamSynchronize(stream));
  }

  double host_sum = 0.0, wait_sum = 0.0;
  double t0 = now_us();
  for (int i = 0; i < iters; ++i) {
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

  float gpu_ms_sum = 0.0f;
  for (int i = 0; i < iters; ++i) {
    CHECK(hipEventRecord(evStart, stream));
    CHECK(hipGraphLaunch(exec, stream));
    CHECK(hipEventRecord(evStop, stream));
    CHECK(hipEventSynchronize(evStop));
    float ms = 0.0f;
    CHECK(hipEventElapsedTime(&ms, evStart, evStop));
    gpu_ms_sum += ms;
  }
  double gpu_us = (gpu_ms_sum / iters) * 1000.0;

  printf("simple-graph: kernels=%d  launches=%d\n", nkernel, iters);
  printf("  wall  us/launch : %9.2f   (%.4f us/kernel)\n", wall, wall / nkernel);
  printf("    host-enqueue  : %9.2f\n", host);
  printf("    GPU-wait      : %9.2f\n", wait);
  printf("  GPU   us/launch : %9.2f   (%.4f us/kernel)\n", gpu_us, gpu_us / nkernel);

  CHECK(hipEventDestroy(evStart));
  CHECK(hipEventDestroy(evStop));
  CHECK(hipStreamDestroy(stream));
  CHECK(hipGraphExecDestroy(exec));
  CHECK(hipGraphDestroy(graph));
  CHECK(hipFree(in_d));
  CHECK(hipFree(out_d));
  return 0;
}
