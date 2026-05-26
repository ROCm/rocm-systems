#include <hip/hip_runtime_api.h>

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <vector>

namespace {

constexpr uint32_t kInputBytes = 256;
constexpr uint32_t kWorkItems = 256;
constexpr uint32_t kThreadsPerBlock = 64;

using PersistentHook = int (*)();

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

uint32_t selector_from_input(const std::vector<uint8_t> &input) {
  uint32_t selector = 0;
  for (uint32_t i = 0; i < 8; ++i) {
    selector |= static_cast<uint32_t>(input[i] & 1u) << i;
  }
  return selector;
}

PersistentHook load_hook(const char *name) {
  return reinterpret_cast<PersistentHook>(dlsym(RTLD_DEFAULT, name));
}

void call_hook(PersistentHook hook, const char *name) {
  if (hook == nullptr)
    return;
  int rc = hook();
  if (rc != 0) {
    fprintf(stderr, "%s failed with rc=%d\n", name, rc);
    exit(3);
  }
}

uint32_t run_iteration(uint32_t iteration, const std::vector<uint8_t> &input,
                       hipFunction_t kernel_a, hipFunction_t kernel_b,
                       uint8_t *device_input, uint32_t *device_scratch,
                       uint32_t *device_result, PersistentHook begin,
                       PersistentHook end) {
  uint32_t selector = selector_from_input(input);
  call_hook(begin, "rocjitsu_afl_persistent_begin");

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
  if (end != nullptr) {
    call_hook(end, "rocjitsu_afl_persistent_end");
  } else {
    HIP_CHECK(hipDeviceSynchronize());
  }

  std::vector<uint32_t> output(kWorkItems, 0);
  HIP_CHECK(hipMemcpy(output.data(), device_result,
                      sizeof(uint32_t) * output.size(), hipMemcpyDeviceToHost));
  uint32_t result = 0;
  for (uint32_t value : output) {
    result += value;
  }
  printf("iteration=%u selector=%u result=%u\n", iteration, selector, result);
  return result;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 4) {
    fprintf(stderr, "usage: %s <branchy_kernels.hsaco> <input-a> <input-b>\n", argv[0]);
    return 1;
  }

  PersistentHook begin = load_hook("rocjitsu_afl_persistent_begin");
  PersistentHook end = load_hook("rocjitsu_afl_persistent_end");
  if (getenv("ROCJITSU_AFL_REQUIRE_PERSISTENT_HOOKS") != nullptr &&
      (begin == nullptr || end == nullptr)) {
    fprintf(stderr, "rocjitsu AFL persistent hooks are not available\n");
    return 4;
  }

  const std::vector<uint8_t> input_a = read_seed(argv[2]);
  const std::vector<uint8_t> input_b = read_seed(argv[3]);

  hipModule_t module = nullptr;
  hipFunction_t kernel_a = nullptr;
  hipFunction_t kernel_b = nullptr;
  HIP_CHECK(hipModuleLoad(&module, argv[1]));
  HIP_CHECK(hipModuleGetFunction(&kernel_a, module, "branchy_kernel_a"));
  HIP_CHECK(hipModuleGetFunction(&kernel_b, module, "branchy_kernel_b"));

  uint8_t *device_input = nullptr;
  uint32_t *device_scratch = nullptr;
  uint32_t *device_result = nullptr;
  HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&device_input), kInputBytes));
  HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&device_scratch),
                      sizeof(uint32_t) * kWorkItems));
  HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&device_result),
                      sizeof(uint32_t) * kWorkItems));

  const uint32_t result_a =
      run_iteration(0, input_a, kernel_a, kernel_b, device_input, device_scratch,
                    device_result, begin, end);
  const uint32_t result_b =
      run_iteration(1, input_b, kernel_a, kernel_b, device_input, device_scratch,
                    device_result, begin, end);
  if (result_a == result_b) {
    fprintf(stderr, "persistent smoke inputs should exercise different outputs\n");
    return 5;
  }

  HIP_CHECK(hipFree(device_result));
  HIP_CHECK(hipFree(device_scratch));
  HIP_CHECK(hipFree(device_input));
  HIP_CHECK(hipModuleUnload(module));
  return 0;
}
