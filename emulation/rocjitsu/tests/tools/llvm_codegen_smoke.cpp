// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime_api.h>

#include <array>
#include <cstdio>
#include <string_view>

#define HIP_CHECK(call)                                                                            \
  do {                                                                                             \
    const hipError_t status = (call);                                                              \
    if (status != hipSuccess) {                                                                    \
      std::fprintf(stderr, "%s: %s (%s)\n", #call, hipGetErrorName(status),                        \
                   hipGetErrorString(status));                                                     \
      return 1;                                                                                    \
    }                                                                                              \
  } while (false)

int main(int argc, char **argv) {
  if (argc != 3) {
    std::fprintf(stderr, "Usage: %s execute|reject CODE_OBJECT\n", argv[0]);
    return 2;
  }
  const std::string_view mode = argv[1];
  if (mode != "execute" && mode != "reject")
    return 2;

  HIP_CHECK(hipInit(0));
  HIP_CHECK(hipSetDevice(0));
  hipModule_t module = nullptr;
  const hipError_t load_status = hipModuleLoad(&module, argv[2]);
  if (mode == "reject") {
    if (load_status != hipErrorNoBinaryForGpu) {
      std::fprintf(stderr, "Expected incompatible feature modes to be rejected, got %s\n",
                   hipGetErrorName(load_status));
      return 1;
    }
    HIP_CHECK(hipDeviceReset());
    return 0;
  }
  HIP_CHECK(load_status);
  hipFunction_t kernel = nullptr;
  HIP_CHECK(hipModuleGetFunction(&kernel, module, "smoke"));
  constexpr size_t count = 128;
  std::array<unsigned, count> input{}, output{};
  for (size_t i = 0; i < count; ++i)
    input[i] = static_cast<unsigned>(i * 3);
  void *device_input = nullptr;
  void *device_output = nullptr;
  HIP_CHECK(hipMalloc(&device_input, sizeof(input)));
  HIP_CHECK(hipMalloc(&device_output, sizeof(output)));
  HIP_CHECK(hipMemcpy(device_input, input.data(), sizeof(input), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(device_output, 0, sizeof(output)));
  void *arguments[] = {&device_input, &device_output};
  HIP_CHECK(hipModuleLaunchKernel(kernel, 2, 1, 1, 64, 1, 1, 0, nullptr, arguments, nullptr));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(output.data(), device_output, sizeof(output), hipMemcpyDeviceToHost));
  for (size_t i = 0; i < count; ++i) {
    if (output[i] != input[i] + 7) {
      std::fprintf(stderr, "Element %zu: got %u, expected %u\n", i, output[i], input[i] + 7);
      return 1;
    }
  }
  HIP_CHECK(hipFree(device_input));
  HIP_CHECK(hipFree(device_output));
  HIP_CHECK(hipModuleUnload(module));
  HIP_CHECK(hipDeviceReset());
  std::puts("128 kernel results matched");
  return 0;
}
