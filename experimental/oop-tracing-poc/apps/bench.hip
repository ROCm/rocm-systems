// Local POC overhead benchmark.  bench <segment> [go_file]
//   calls      10 x 1,000,000 hipGetDevice calls                         -> ns per call (CPU)
//   launches   10 x 20,000 empty launches + sync, CPU-timed (GPU-bound)  -> ns per launch
//   enqueue    10 x (1000 batches x 64 empty launches), only the enqueue
//              loop timed, sync between batches (queue never fills)     -> ns per launch (CPU)
//   gpu_short  10 x 20,000 empty kernels back to back, HIP-event timed   -> ns per kernel (GPU)
//   gpu_long   10 x 2,000 fixed-work kernels back to back, event timed   -> ns per kernel (GPU)
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <unistd.h>

__global__ void empty() {}

__global__ void spin_fixed(float* out, int rounds) {
  float v = threadIdx.x;
  for (int r = 0; r < rounds; ++r) v = v * 1.000001f + 0.5f;
  if (v == -1.0f) out[threadIdx.x] = v;  // never true; keeps the loop
}

using clk = std::chrono::steady_clock;
static double ns_since(clk::time_point t0, clk::time_point t1) {
  return std::chrono::duration<double, std::nano>(t1 - t0).count();
}

int main(int argc, char** argv) {
  const char* seg = argc > 1 ? argv[1] : "calls";
  const char* go_file = (argc > 2 && argv[2][0] != 0) ? argv[2] : nullptr;
  (void)hipSetDevice(0);
  (void)hipFree(nullptr);
  if (go_file != nullptr) {
    std::printf("pid %d waiting\n", getpid());
    std::fflush(stdout);
    while (access(go_file, F_OK) != 0) std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  hipStream_t s;
  (void)hipStreamCreate(&s);
  float* out = nullptr;
  (void)hipMalloc(reinterpret_cast<void**>(&out), 1024 * sizeof(float));
  hipEvent_t e0, e1;
  (void)hipEventCreate(&e0);
  (void)hipEventCreate(&e1);
  hipLaunchKernelGGL(empty, dim3(1), dim3(64), 0, s);  // warm up: queue, code objects
  hipLaunchKernelGGL(spin_fixed, dim3(1), dim3(256), 0, s, out, 10);
  (void)hipStreamSynchronize(s);
  int dev = 0;
  for (int i = 0; i < 10000; ++i) (void)hipGetDevice(&dev);
  if (const char* settle = std::getenv("BENCH_SETTLE_MS")) {  // let a selector finish setup
    std::this_thread::sleep_for(std::chrono::milliseconds(std::atoi(settle)));
    hipLaunchKernelGGL(empty, dim3(1), dim3(64), 0, s);
    (void)hipStreamSynchronize(s);
  }

  const int kRounds = 2000;   // spin_fixed work per kernel
  for (int k = 0; k < 10; ++k) {
    double per = 0;
    double wall = 0;
    if (std::strcmp(seg, "calls") == 0) {
      const long n = 1000000;
      const auto t0 = clk::now();
      for (long i = 0; i < n; ++i) (void)hipGetDevice(&dev);
      per = ns_since(t0, clk::now()) / n;
    } else if (std::strcmp(seg, "launches") == 0) {
      const long n = 20000;
      const auto t0 = clk::now();
      for (long i = 0; i < n; ++i) hipLaunchKernelGGL(empty, dim3(1), dim3(64), 0, s);
      (void)hipStreamSynchronize(s);
      per = ns_since(t0, clk::now()) / n;
    } else if (std::strcmp(seg, "enqueue") == 0) {
      const int batches = 1000, batch = 64;
      double total = 0;
      for (int b = 0; b < batches; ++b) {
        const auto t0 = clk::now();
        for (int i = 0; i < batch; ++i) hipLaunchKernelGGL(empty, dim3(1), dim3(64), 0, s);
        total += ns_since(t0, clk::now());
        (void)hipStreamSynchronize(s);  // untimed: keeps the queue from filling
      }
      per = total / (batches * batch);
    } else if (std::strcmp(seg, "gpu_short") == 0 || std::strcmp(seg, "gpu_long") == 0) {
      const bool lng = seg[4] == 'l';
      const int n = lng ? 2000 : 20000;
      (void)hipEventRecord(e0, s);
      for (int i = 0; i < n; ++i) {
        if (lng) hipLaunchKernelGGL(spin_fixed, dim3(1), dim3(256), 0, s, out, kRounds);
        else hipLaunchKernelGGL(empty, dim3(1), dim3(64), 0, s);
      }
      (void)hipEventRecord(e1, s);
      (void)hipEventSynchronize(e1);
      float ms = 0;
      (void)hipEventElapsedTime(&ms, e0, e1);
      per = ms * 1e6 / n;
    } else if (std::strcmp(seg, "multi") == 0) {
      // multi <streams> <rounds>: each stream runs 2000 back-to-back spin_fixed kernels; all
      // streams start together after e0 and are joined into e1 on stream 0 (GPU-event timed).
      static int ns = argc > 3 ? std::atoi(argv[3]) : 1;
      static int rounds = argc > 4 ? std::atoi(argv[4]) : 2000;
      static hipStream_t ms[8];
      static hipEvent_t done[8];
      static bool init = false;
      if (!init) {
        ms[0] = s;
        for (int i = 1; i < ns; ++i) (void)hipStreamCreate(&ms[i]);
        for (int i = 0; i < ns; ++i) (void)hipEventCreateWithFlags(&done[i], hipEventDisableTiming);
        for (int i = 0; i < ns; ++i) {  // warm each queue
          hipLaunchKernelGGL(spin_fixed, dim3(1), dim3(256), 0, ms[i], out, 10);
        }
        (void)hipDeviceSynchronize();
        init = true;
      }
      const int n = 2000;
      const auto w0 = clk::now();
      (void)hipEventRecord(e0, ms[0]);
      for (int i = 1; i < ns; ++i) (void)hipStreamWaitEvent(ms[i], e0, 0);
      for (int k2 = 0; k2 < n; ++k2)
        for (int i = 0; i < ns; ++i)
          hipLaunchKernelGGL(spin_fixed, dim3(1), dim3(256), 0, ms[i], out, rounds);
      for (int i = 1; i < ns; ++i) {
        (void)hipEventRecord(done[i], ms[i]);
        (void)hipStreamWaitEvent(ms[0], done[i], 0);
      }
      (void)hipEventRecord(e1, ms[0]);
      (void)hipEventSynchronize(e1);
      float ms_elapsed = 0;
      (void)hipEventElapsedTime(&ms_elapsed, e0, e1);
      per = ms_elapsed * 1e6 / n;  // ns of GPU time per kernel per stream
      wall = ns_since(w0, clk::now()) / n;  // CPU wall time per kernel per stream, incl. sync
    } else {
      std::fprintf(stderr, "unknown segment %s\n", seg);
      return 2;
    }
    if (std::strcmp(seg, "multi") == 0) std::printf("multi,%s,%s,%.2f,%.2f\n", argc > 3 ? argv[3] : "1", argc > 4 ? argv[4] : "2000", per, wall);
    else std::printf("%s,%.2f\n", seg, per);
  }
  (void)hipEventDestroy(e0);
  (void)hipEventDestroy(e1);
  (void)hipFree(out);
  (void)hipStreamDestroy(s);
  return 0;
}
