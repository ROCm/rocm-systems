// True host-driven WHILE baseline: the host owns the loop and pays a full
// host<->device round trip every iteration (launch body kernel, copy the
// condition back to host, test it, decide whether to relaunch).
//
// This is the classic pattern that the GPU-resident conditional WHILE is meant
// to replace, so it is the honest baseline for "how much does GPU-residency
// save per iteration".
#include "common.h"
#include <chrono>

__global__ void kBodyHost(int* counter) {
  *counter += 1;
}

static double now_us() {
  using namespace std::chrono;
  return duration<double, std::micro>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
  const int limit  = (argc > 1) ? atoi(argv[1]) : 1000;  // iterations per launch
  const int iters  = (argc > 2) ? atoi(argv[2]) : 100;   // launches to average
  const int warmup = 20;

  int *counter;
  CHECK(hipMalloc(&counter, sizeof(int)));

  hipStream_t stream;
  CHECK(hipStreamCreate(&stream));

  // One "launch" == run the whole WHILE loop to completion, host-driven.
  auto run_once = [&]() {
    CHECK(hipMemsetAsync(counter, 0, sizeof(int), stream));
    int hc = 0;
    // Host owns the condition: keep launching the body until counter reaches limit.
    while (hc < limit) {
      hipLaunchKernelGGL(kBodyHost, dim3(1), dim3(1), 0, stream, counter);
      // Round trip: pull the updated condition back to the host each iteration.
      CHECK(hipMemcpyAsync(&hc, counter, sizeof(int), hipMemcpyDeviceToHost, stream));
      CHECK(hipStreamSynchronize(stream));
    }
    return hc;
  };

  for (int i = 0; i < warmup; ++i) {
    int hc = run_once();
    if (hc != limit) { fprintf(stderr, "MISMATCH: counter=%d exp=%d\n", hc, limit); return 1; }
  }

  double t0 = now_us();
  for (int i = 0; i < iters; ++i) run_once();
  double wall = (now_us() - t0) / iters;

  printf("host-while (true round trip): limit(iters)=%d  launches=%d\n", limit, iters);
  printf("  wall  us/launch : %9.2f   (%.4f us/iter)\n", wall, wall / limit);

  CHECK(hipStreamDestroy(stream));
  return 0;
}
