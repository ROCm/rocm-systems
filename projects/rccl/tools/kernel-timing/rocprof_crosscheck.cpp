// Cross-check dispatch-attached event timings against rocprof's kernel trace,
// and measure what rocprof itself costs. Run bare, then under rocprofv3.

#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

#define HC(x)                                                                            \
  do {                                                                                   \
    hipError_t _e = (x);                                                                 \
    if (_e != hipSuccess) printf("FAIL %s -> %s\n", #x, hipGetErrorString(_e));           \
  } while (0)

__global__ void spin(uint64_t ticks) {
  uint64_t t0 = __builtin_amdgcn_s_memrealtime();
  while (__builtin_amdgcn_s_memrealtime() - t0 < ticks) {
  }
}

int main() {
  const int N = 300;
  hipDeviceProp_t p;
  HC(hipGetDeviceProperties(&p, 0));
  int khz = 0;
  HC(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, 0));
  uint64_t ticks = (uint64_t)(50.0 * khz / 1000.0);
  size_t sz = sizeof(ticks);
  void* extra[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &ticks, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz,
                   HIP_LAUNCH_PARAM_END};

  hipFunction_t fn;
  HC(hipGetFuncBySymbol(&fn, (const void*)spin));
  hipStream_t s;
  HC(hipStreamCreate(&s));
  std::vector<hipEvent_t> x0(N), x1(N);
  for (int i = 0; i < N; ++i) {
    HC(hipEventCreate(&x0[i]));
    HC(hipEventCreate(&x1[i]));
  }

  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < N; ++i)
    HC(hipExtModuleLaunchKernel(fn, 8 * 256, 1, 1, 256, 1, 1, 0, s, nullptr, extra, x0[i], x1[i],
                                0));
  HC(hipStreamSynchronize(s));
  auto t1 = std::chrono::steady_clock::now();

  std::vector<double> ex;
  for (int i = 0; i < N; ++i) {
    float ms = 0;
    HC(hipEventElapsedTime(&ms, x0[i], x1[i]));
    ex.push_back(ms * 1000.0);
  }
  std::sort(ex.begin(), ex.end());
  printf("device %s, %d launches of a 50 us kernel\n", p.gcnArchName, N);
  printf("attached-event duration: median %.2f us  min %.2f  max %.2f\n", ex[N / 2], ex.front(),
         ex.back());
  printf("host wall time per launch: %.3f us\n",
         std::chrono::duration<double, std::micro>(t1 - t0).count() / N);
  return 0;
}
