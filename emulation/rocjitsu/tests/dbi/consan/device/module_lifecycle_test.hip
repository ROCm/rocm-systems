// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Exercises a behavioral ConSan pair through HIP module load, unload/reload,
/// simultaneous independent code objects, and launches on separate streams.
/// The kernels are deliberately absent from this executable's static image.

#include "consan_device_test_support.h"

#ifndef RJ_CONSAN_MODULE_LIFECYCLE_HSACO_A
#error "RJ_CONSAN_MODULE_LIFECYCLE_HSACO_A must name the first module fixture"
#endif
#ifndef RJ_CONSAN_MODULE_LIFECYCLE_HSACO_B
#error "RJ_CONSAN_MODULE_LIFECYCLE_HSACO_B must name the second module fixture"
#endif

namespace {

struct ModuleSpec {
  const char *path;
  const char *correct_kernel;
  const char *incorrect_kernel;
  uint32_t tag;
};

constexpr ModuleSpec kModuleA{RJ_CONSAN_MODULE_LIFECYCLE_HSACO_A, "module_lifecycle_correct_a",
                              "module_lifecycle_incorrect_a", 0xa1000000u};
constexpr ModuleSpec kModuleB{RJ_CONSAN_MODULE_LIFECYCLE_HSACO_B, "module_lifecycle_correct_b",
                              "module_lifecycle_incorrect_b", 0xb2000000u};

void launch_loaded_module(hipModule_t module, const ModuleSpec &spec, bool synchronize,
                          hipStream_t stream, uint32_t *device_output, uint32_t marker) {
  hipFunction_t function = nullptr;
  HIP_ASSERT(hipModuleGetFunction(&function, module,
                                  synchronize ? spec.correct_kernel : spec.incorrect_kernel));
  struct {
    uint32_t *output;
    uint32_t marker;
  } arguments{device_output, marker};
  size_t argument_size = sizeof(arguments);
  void *configuration[] = {
      HIP_LAUNCH_PARAM_BUFFER_POINTER,
      &arguments,
      HIP_LAUNCH_PARAM_BUFFER_SIZE,
      &argument_size,
      HIP_LAUNCH_PARAM_END,
  };
  const uint32_t thread_count = 2u * consan_device_test::active_wave_size();
  HIP_ASSERT(hipModuleLaunchKernel(function, 1u, 1u, 1u, thread_count, 1u, 1u, 0u, stream, nullptr,
                                   configuration));
}

void check_output(uint32_t *device_output, uint32_t expected) {
  uint32_t output = 0u;
  HIP_ASSERT(hipMemcpy(&output, device_output, sizeof(output), hipMemcpyDeviceToHost));
  EXPECT_EQ(output, expected);
}

void run_unload_reload(const ModuleSpec &spec, bool synchronize) {
  constexpr uint32_t kMarker = 0x005a17e5u;
  for (uint32_t generation = 0u; generation < 2u; ++generation) {
    hipModule_t module = nullptr;
    HIP_ASSERT(hipModuleLoad(&module, spec.path));
    uint32_t *device_output = nullptr;
    HIP_ASSERT(hipMalloc(&device_output, sizeof(uint32_t)));
    HIP_ASSERT(hipMemset(device_output, 0, sizeof(uint32_t)));
    launch_loaded_module(module, spec, synchronize, nullptr, device_output, kMarker + generation);
    HIP_ASSERT(hipDeviceSynchronize());
    if (synchronize)
      check_output(device_output, (kMarker + generation) ^ spec.tag);
    HIP_ASSERT(hipFree(device_output));
    HIP_ASSERT(hipModuleUnload(module));
  }
}

void run_simultaneous_modules(bool synchronize) {
  constexpr uint32_t kMarkerA = 0x00123456u;
  constexpr uint32_t kMarkerB = 0x00654321u;
  hipModule_t module_a = nullptr;
  hipModule_t module_b = nullptr;
  HIP_ASSERT(hipModuleLoad(&module_a, kModuleA.path));
  HIP_ASSERT(hipModuleLoad(&module_b, kModuleB.path));
  hipStream_t stream_a = nullptr;
  hipStream_t stream_b = nullptr;
  HIP_ASSERT(hipStreamCreate(&stream_a));
  HIP_ASSERT(hipStreamCreate(&stream_b));
  uint32_t *output_a = nullptr;
  uint32_t *output_b = nullptr;
  HIP_ASSERT(hipMalloc(&output_a, sizeof(uint32_t)));
  HIP_ASSERT(hipMalloc(&output_b, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(output_a, 0, sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(output_b, 0, sizeof(uint32_t)));

  launch_loaded_module(module_a, kModuleA, synchronize, stream_a, output_a, kMarkerA);
  launch_loaded_module(module_b, kModuleB, synchronize, stream_b, output_b, kMarkerB);
  HIP_ASSERT(hipStreamSynchronize(stream_a));
  HIP_ASSERT(hipStreamSynchronize(stream_b));
  if (synchronize) {
    check_output(output_a, kMarkerA ^ kModuleA.tag);
    check_output(output_b, kMarkerB ^ kModuleB.tag);
  }

  HIP_ASSERT(hipFree(output_a));
  HIP_ASSERT(hipFree(output_b));
  HIP_ASSERT(hipStreamDestroy(stream_a));
  HIP_ASSERT(hipStreamDestroy(stream_b));
  HIP_ASSERT(hipModuleUnload(module_a));
  HIP_ASSERT(hipModuleUnload(module_b));
}

void run_module_lifecycle(bool synchronize) {
  run_unload_reload(kModuleA, synchronize);
  run_simultaneous_modules(synchronize);
}

TEST(ConSanDeviceModuleLifecycleTest, Correct) { run_module_lifecycle(true); }

TEST(ConSanDeviceModuleLifecycleTest, Incorrect) { run_module_lifecycle(false); }

} // namespace
