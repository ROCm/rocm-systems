// Host harness for the rewriter: loads lds_conflict from a code object file
// with hipModuleLoad, runs it, and writes the output buffer to a file so a
// rewritten code object can be checked bit-for-bit against the original.
//
// Usage: lds_module <code_object> <out.bin> [blocks iters_a iters_b]
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(x)                                                              \
  do {                                                                        \
    hipError_t e = (x);                                                       \
    if (e != hipSuccess) {                                                    \
      std::printf("%s failed: %s\n", #x, hipGetErrorString(e));               \
      return 1;                                                               \
    }                                                                         \
  } while (0)

constexpr int kThreads = 256;

int main(int argc, char **argv) {
  if (argc < 3) {
    std::printf("usage: %s <code_object> <out.bin> [blocks iters_a iters_b]\n", argv[0]);
    return 2;
  }
  const int blocks = argc > 3 ? std::atoi(argv[3]) : 64;
  const int iters_a = argc > 4 ? std::atoi(argv[4]) : 64;
  const int iters_b = argc > 5 ? std::atoi(argv[5]) : 64;

  hipModule_t module;
  hipFunction_t kernel;
  CHECK(hipModuleLoad(&module, argv[1]));
  CHECK(hipModuleGetFunction(&kernel, module, "_Z12lds_conflictPfii"));

  float *out = nullptr;
  CHECK(hipMalloc(&out, sizeof(float) * blocks * kThreads));
  struct {
    float *out;
    int iters_a;
    int iters_b;
  } args{out, iters_a, iters_b};
  size_t size = sizeof(args);
  void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                    &size, HIP_LAUNCH_PARAM_END};
  CHECK(hipModuleLaunchKernel(kernel, blocks, 1, 1, kThreads, 1, 1, 0, nullptr, nullptr,
                              config));
  CHECK(hipDeviceSynchronize());

  std::vector<float> h(blocks * kThreads);
  CHECK(hipMemcpy(h.data(), out, sizeof(float) * h.size(), hipMemcpyDeviceToHost));
  FILE *f = std::fopen(argv[2], "wb");
  std::fwrite(h.data(), sizeof(float), h.size(), f);
  std::fclose(f);
  std::printf("blocks=%d out[0]=%f out[last]=%f\n", blocks, h.front(), h.back());
  CHECK(hipFree(out));
  CHECK(hipModuleUnload(module));
  return 0;
}
