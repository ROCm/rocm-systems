#include <hip/hip_runtime_api.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <vector>

namespace {

constexpr uint32_t kWorkItems = 64;
constexpr uint32_t kThreadsPerBlock = 64;

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t err__ = (expr);                                                 \
    if (err__ != hipSuccess) {                                                 \
      fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__,          \
              hipGetErrorString(err__));                                       \
      exit(2);                                                                 \
    }                                                                          \
  } while (0)

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <entry_liveness_kernels.hsaco>\n", argv[0]);
    return 1;
  }

  hipModule_t module = nullptr;
  hipFunction_t kernel = nullptr;
  HIP_CHECK(hipModuleLoad(&module, argv[1]));
  HIP_CHECK(hipModuleGetFunction(&kernel, module, "entry_liveness_kernel"));

  uint32_t *device_output = nullptr;
  HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&device_output),
                      sizeof(uint32_t) * kWorkItems));
  HIP_CHECK(hipMemset(device_output, 0, sizeof(uint32_t) * kWorkItems));

  void *args[] = {&device_output};
  HIP_CHECK(hipModuleLaunchKernel(kernel, 1, 1, 1, kThreadsPerBlock, 1, 1, 0,
                                  nullptr, args, nullptr));
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<uint32_t> output(kWorkItems, 0);
  HIP_CHECK(hipMemcpy(output.data(), device_output,
                      sizeof(uint32_t) * output.size(), hipMemcpyDeviceToHost));

  uint32_t checksum = 0;
  for (uint32_t value : output)
    checksum ^= value;
  printf("entry_liveness checksum=%u\n", checksum);

  HIP_CHECK(hipFree(device_output));
  HIP_CHECK(hipModuleUnload(module));
  return 0;
}
