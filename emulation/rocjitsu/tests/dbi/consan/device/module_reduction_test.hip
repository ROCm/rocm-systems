// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Exercises ConSan on a caller-supplied code object loaded through the HIP
/// module API. This is deliberately not a statically linked HIP kernel.

#include "consan_device_test_support.h"

#include <vector>

#ifndef RJ_CONSAN_MODULE_REDUCTION_HSACO
#error "RJ_CONSAN_MODULE_REDUCTION_HSACO must name the module fixture"
#endif

namespace {

void run_module_reduction(const char *kernel_name, bool check_reduction) {
  constexpr uint32_t kBlockSize = 256u;
  constexpr uint32_t kBlockCount = 16u;
  constexpr uint32_t kElementCount = kBlockSize * kBlockCount;

  hipModule_t module = nullptr;
  HIP_ASSERT(hipModuleLoad(&module, RJ_CONSAN_MODULE_REDUCTION_HSACO));
  hipFunction_t function = nullptr;
  HIP_ASSERT(hipModuleGetFunction(&function, module, kernel_name));

  std::vector<uint32_t> input(kElementCount, 1u);
  std::vector<uint32_t> output(kBlockCount, 0u);
  uint32_t *device_input = nullptr;
  uint32_t *device_output = nullptr;
  HIP_ASSERT(hipMalloc(&device_input, input.size() * sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&device_output, output.size() * sizeof(uint32_t)));
  HIP_ASSERT(hipMemcpy(device_input, input.data(), input.size() * sizeof(uint32_t),
                       hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemset(device_output, 0, output.size() * sizeof(uint32_t)));

  struct {
    const uint32_t *input;
    uint32_t *output;
    uint32_t count;
  } arguments{device_input, device_output, kElementCount};
  size_t argument_size = sizeof(arguments);
  void *configuration[] = {
      HIP_LAUNCH_PARAM_BUFFER_POINTER,
      &arguments,
      HIP_LAUNCH_PARAM_BUFFER_SIZE,
      &argument_size,
      HIP_LAUNCH_PARAM_END,
  };
  HIP_ASSERT(hipModuleLaunchKernel(function, kBlockCount, 1u, 1u, kBlockSize, 1u, 1u, 0u, nullptr,
                                   nullptr, configuration));
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipMemcpy(output.data(), device_output, output.size() * sizeof(uint32_t),
                       hipMemcpyDeviceToHost));

  if (check_reduction) {
    for (uint32_t block = 0; block < kBlockCount; ++block)
      EXPECT_EQ(output[block], kBlockSize) << "block=" << block;
  }

  HIP_ASSERT(hipFree(device_input));
  HIP_ASSERT(hipFree(device_output));
  HIP_ASSERT(hipModuleUnload(module));
}

TEST(ConSanDeviceModuleReductionTest, Correct) {
  run_module_reduction("module_reduction_correct", true);
}

TEST(ConSanDeviceModuleReductionTest, Incorrect) {
  run_module_reduction("module_reduction_incorrect", false);
}

} // namespace
