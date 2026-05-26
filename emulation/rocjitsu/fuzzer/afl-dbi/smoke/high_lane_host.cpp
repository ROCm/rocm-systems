#include <hip/hip_runtime_api.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <vector>

namespace {

constexpr uint32_t kInputBytes = 64;
constexpr uint32_t kWorkItems = 64;
constexpr uint32_t kOutputWords = 160;

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t err__ = (expr);                                                 \
    if (err__ != hipSuccess) {                                                 \
      fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__,          \
              hipGetErrorString(err__));                                       \
      exit(2);                                                                 \
    }                                                                          \
  } while (0)

std::vector<uint8_t> read_seed(const char *path) {
  FILE *file = fopen(path, "rb");
  if (file == nullptr) {
    perror(path);
    exit(1);
  }

  std::vector<uint8_t> bytes(kInputBytes, 0);
  size_t n = fread(bytes.data(), 1, bytes.size(), file);
  fclose(file);
  const uint8_t fill = n != 0 ? bytes[0] : 0;
  for (size_t i = n; i < bytes.size(); ++i) {
    bytes[i] = fill;
  }
  return bytes;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s <high_lane_kernels.hsaco> <input>\n", argv[0]);
    return 1;
  }

  const std::vector<uint8_t> input = read_seed(argv[2]);
  hipModule_t module = nullptr;
  hipFunction_t kernel = nullptr;
  HIP_CHECK(hipModuleLoad(&module, argv[1]));
  HIP_CHECK(hipModuleGetFunction(&kernel, module, "high_lane_kernel"));

  uint8_t *device_input = nullptr;
  uint32_t *device_output = nullptr;
  HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&device_input), input.size()));
  HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&device_output), sizeof(uint32_t) * kOutputWords));
  HIP_CHECK(hipMemcpy(device_input, input.data(), input.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(device_output, 0, sizeof(uint32_t) * kOutputWords));

  void *args[] = {&device_input, &device_output};
  HIP_CHECK(
      hipModuleLaunchKernel(kernel, 1, 1, 1, kWorkItems, 1, 1, 0, nullptr, args, nullptr));
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<uint32_t> output(kOutputWords, 0);
  HIP_CHECK(hipMemcpy(output.data(), device_output, sizeof(uint32_t) * output.size(),
                      hipMemcpyDeviceToHost));
  uint32_t result = 0;
  for (uint32_t value : output) {
    result ^= value + 0x9e3779b9u + (result << 6) + (result >> 2);
  }
  printf("selector=%u result=%u\n", static_cast<unsigned>(input[0] & 1u), result);

  HIP_CHECK(hipFree(device_output));
  HIP_CHECK(hipFree(device_input));
  HIP_CHECK(hipModuleUnload(module));
  return 0;
}
