// Local POC test app: two streams, two kernels, copies. Waits for a go-file after HIP init so an
// out-of-process collector can attach before the first queue is created.
//   app [iterations] [go_file]
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <thread>
#include <unistd.h>
#include <vector>

#define CHECK(x)                                                                                   \
  do {                                                                                             \
    hipError_t e_ = (x);                                                                           \
    if (e_ != hipSuccess) {                                                                        \
      std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_));      \
      std::exit(1);                                                                                \
    }                                                                                              \
  } while (0)

__global__ void saxpy(float a, const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}

__global__ void spin(float* y, int n, int rounds) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = y[i];
    for (int r = 0; r < rounds; ++r) v = v * 1.0001f + 0.5f;
    y[i] = v;
  }
}

int main(int argc, char** argv) {
  const int iters = argc > 1 ? std::atoi(argv[1]) : 20;
  const char* go_file = argc > 2 ? argv[2] : nullptr;
  const int n = 1 << 22;

  CHECK(hipSetDevice(0));
  CHECK(hipFree(nullptr));  // initialize the runtime; no queue yet
  if (go_file != nullptr) {
    std::printf("pid %d waiting for %s\n", getpid(), go_file);
    std::fflush(stdout);
    while (access(go_file, F_OK) != 0) std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  std::vector<float> h(n, 1.0f);
  float* x = nullptr;
  float* y = nullptr;
  CHECK(hipMalloc(reinterpret_cast<void**>(&x), n * sizeof(float)));
  CHECK(hipMalloc(reinterpret_cast<void**>(&y), n * sizeof(float)));
  CHECK(hipMemcpy(x, h.data(), n * sizeof(float), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(y, h.data(), n * sizeof(float), hipMemcpyHostToDevice));

  hipStream_t s[2];
  CHECK(hipStreamCreate(&s[0]));
  CHECK(hipStreamCreate(&s[1]));
  const dim3 block(256);
  const dim3 grid((n + 255) / 256);
  for (int it = 0; it < iters; ++it) {
    hipLaunchKernelGGL(saxpy, grid, block, 0, s[0], 2.0f, x, y, n);
    hipLaunchKernelGGL(spin, grid, block, 0, s[1], y, n, 200);
    CHECK(hipMemcpyAsync(h.data(), y, 4096, hipMemcpyDeviceToHost, s[0]));
    if (it % 5 == 4) CHECK(hipDeviceSynchronize());
  }
  CHECK(hipStreamSynchronize(s[0]));
  CHECK(hipStreamSynchronize(s[1]));
  CHECK(hipMemcpy(h.data(), y, n * sizeof(float), hipMemcpyDeviceToHost));
  CHECK(hipStreamDestroy(s[0]));
  CHECK(hipStreamDestroy(s[1]));
  CHECK(hipFree(x));
  CHECK(hipFree(y));
  std::printf("done %d iterations, y[0]=%f\n", iters, h[0]);
  return 0;
}
