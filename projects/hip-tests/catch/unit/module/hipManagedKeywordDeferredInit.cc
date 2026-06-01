/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

constexpr auto kCodeFile = "managed_kernel.code";

// Verifies that a __managed__ variable loaded at runtime via hipModuleLoad is
// initialized during the call to hipMemcpy
HIP_TEST_CASE(Unit_hipModuleLaunchKernel_ManagedVar_Memcpy) {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  for (int i = 0; i < numDevices; ++i) {
    CHECK_MANAGED_MEMORY_SUPPORT_ON_DEVICE(i);
  }

  for (int i = 0; i < numDevices; ++i) {
    HIP_CHECK(hipSetDevice(i));
    CTX_CREATE();

    hipModule_t module;
    HIP_CHECK(hipModuleLoad(&module, kCodeFile));

    hipDeviceptr_t dptr;
    size_t bytes = 0;
    HIP_CHECK(hipModuleGetGlobal(&dptr, &bytes, module, "x"));
    REQUIRE(bytes == sizeof(int));

    constexpr int kSeed = 42;
    HIP_CHECK(hipMemcpy(reinterpret_cast<void*>(dptr), &kSeed, sizeof(int), hipMemcpyHostToDevice));

    hipFunction_t func;
    HIP_CHECK(hipModuleGetFunction(&func, module, "GPU_func"));
    HIP_CHECK(hipModuleLaunchKernel(func, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr,
                                    nullptr));
    HIP_CHECK(hipDeviceSynchronize());

    int result = 0;
    HIP_CHECK(hipMemcpy(&result, reinterpret_cast<void*>(dptr), sizeof(int), hipMemcpyDeviceToHost));
    REQUIRE(result == kSeed + 1);

    HIP_CHECK(hipModuleUnload(module));
    CTX_DESTROY();
  }
}
