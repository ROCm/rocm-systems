#include <hip/hip_runtime_api.h>

#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

namespace {

constexpr uint32_t kInputBytes = 256;
constexpr uint32_t kWorkItems = 256;
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

std::vector<uint8_t> read_seed(const char *path) {
  FILE *file = fopen(path, "rb");
  if (file == nullptr) {
    perror(path);
    exit(1);
  }

  std::vector<uint8_t> bytes(kInputBytes, 0);
  size_t n = fread(bytes.data(), 1, bytes.size(), file);
  fclose(file);
  for (size_t i = n; i < bytes.size(); ++i) {
    bytes[i] = static_cast<uint8_t>(i * 131u + 17u);
  }
  return bytes;
}

void synchronize_after_launches() {
  const char *mode = getenv("ROCFUZZ_BRANCHY_SYNC");
  if (mode != nullptr && strcmp(mode, "none") == 0)
    return;
  if (mode != nullptr && strcmp(mode, "stream") == 0) {
    HIP_CHECK(hipStreamSynchronize(nullptr));
    return;
  }
  if (mode != nullptr && strcmp(mode, "event") == 0) {
    hipEvent_t event = nullptr;
    HIP_CHECK(hipEventCreate(&event));
    HIP_CHECK(hipEventRecord(event, nullptr));
    HIP_CHECK(hipEventSynchronize(event));
    HIP_CHECK(hipEventDestroy(event));
    return;
  }
  HIP_CHECK(hipDeviceSynchronize());
}

bool parse_expected_result(const char *text, uint32_t *result) {
  if (text == nullptr || *text == '\0')
    return false;
  errno = 0;
  char *end = nullptr;
  unsigned long value = strtoul(text, &end, 0);
  if (errno != 0 || end == text || *end != '\0' || value > UINT32_MAX)
    return false;
  *result = static_cast<uint32_t>(value);
  return true;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s <branchy_kernels.hsaco> <input>\n", argv[0]);
    return 1;
  }

  const std::vector<uint8_t> input = read_seed(argv[2]);
  uint32_t selector = 0;
  for (uint32_t i = 0; i < 8; ++i) {
    selector |= static_cast<uint32_t>(input[i] & 1u) << i;
  }

  hipModule_t module = nullptr;
  hipFunction_t kernel_a = nullptr;
  hipFunction_t kernel_b = nullptr;
  HIP_CHECK(hipModuleLoad(&module, argv[1]));
  HIP_CHECK(hipModuleGetFunction(&kernel_a, module, "branchy_kernel_a"));
  HIP_CHECK(hipModuleGetFunction(&kernel_b, module, "branchy_kernel_b"));

  uint8_t *device_input = nullptr;
  uint32_t *device_scratch = nullptr;
  uint32_t *device_result = nullptr;
  HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&device_input), input.size()));
  HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&device_scratch),
                      sizeof(uint32_t) * kWorkItems));
  HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&device_result),
                      sizeof(uint32_t) * kWorkItems));

  HIP_CHECK(hipMemcpy(device_input, input.data(), input.size(),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(device_scratch, 0, sizeof(uint32_t) * kWorkItems));
  HIP_CHECK(hipMemset(device_result, 0, sizeof(uint32_t) * kWorkItems));

  uint32_t n = kWorkItems;

  void *args_a[] = {&device_input, &device_scratch, &n};
  HIP_CHECK(hipModuleLaunchKernel(kernel_a, kWorkItems / kThreadsPerBlock, 1, 1,
                                  kThreadsPerBlock, 1, 1, 0, nullptr, args_a,
                                  nullptr));

  void *args_b[] = {&device_scratch, &device_result, &n, &selector};
  HIP_CHECK(hipModuleLaunchKernel(kernel_b, kWorkItems / kThreadsPerBlock, 1, 1,
                                  kThreadsPerBlock, 1, 1, 0, nullptr, args_b,
                                  nullptr));
  if (getenv("ROCFUZZ_REPEAT_BRANCHY_KERNEL_B") != nullptr) {
    HIP_CHECK(hipModuleLaunchKernel(kernel_b, kWorkItems / kThreadsPerBlock, 1, 1,
                                    kThreadsPerBlock, 1, 1, 0, nullptr, args_b,
                                    nullptr));
  }
  if (getenv("ROCFUZZ_BRANCHY_IGNORE_SYNC_AND_EXIT") != nullptr) {
    (void)hipDeviceSynchronize();
    return 0;
  }
  synchronize_after_launches();

  std::vector<uint32_t> output(kWorkItems, 0);
  HIP_CHECK(hipMemcpy(output.data(), device_result,
                      sizeof(uint32_t) * output.size(), hipMemcpyDeviceToHost));
  uint32_t result = 0;
  for (uint32_t value : output) {
    result += value;
  }
  printf("selector=%u result=%u\n", selector, result);
  bool result_mismatch = false;
  if (const char *expected_text = getenv("ROCFUZZ_BRANCHY_EXPECT_RESULT")) {
    uint32_t expected = 0;
    if (!parse_expected_result(expected_text, &expected)) {
      fprintf(stderr, "invalid ROCFUZZ_BRANCHY_EXPECT_RESULT=%s\n",
              expected_text);
      result_mismatch = true;
    } else if (result != expected) {
      fprintf(stderr, "expected branchy result %u, got %u\n", expected,
              result);
      result_mismatch = true;
    }
  }

  HIP_CHECK(hipFree(device_result));
  HIP_CHECK(hipFree(device_scratch));
  HIP_CHECK(hipFree(device_input));
  if (getenv("ROCFUZZ_SKIP_MODULE_UNLOAD") == nullptr)
    HIP_CHECK(hipModuleUnload(module));
  return result_mismatch ? 3 : 0;
}
