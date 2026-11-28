/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#include <algorithm>
#include <vector>

/**
 * @addtogroup hipModuleGetFunction hipModuleGetFunction
 * @{
 * @ingroup ModuleTest
 */

/**
 * Test Description
 * ------------------------
 *  - hipLaunchKernel and hipFuncSetAttribute accept a hipFunction_t from hipModuleGetFunction
 *    in place of a host function: launch addKernel from addKernel.code that way and check its
 *    result. Once the module is unloaded the handle is no longer a kernel, so both must fail
 *    with hipErrorInvalidDeviceFunction instead of using the freed kernel.
 * Test source
 * ------------------------
 *  - unit/module/hipModuleFunctionHandle.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.14
 */
HIP_TEST_CASE(Unit_hipModuleGetFunction_LaunchHandleWithHipLaunchKernel) {
  constexpr int kSize = 64;
  hipModule_t module;
  hipFunction_t function;
  HIP_CHECK(hipModuleLoad(&module, "addKernel.code"));
  HIP_CHECK(hipModuleGetFunction(&function, module, "addKernel"));

  int* data = nullptr;
  HIP_CHECK(hipMalloc(&data, kSize * sizeof(int)));
  HIP_CHECK(hipMemset(data, 0, kSize * sizeof(int)));
  int size = kSize;
  void* args[] = {&data, &size};
  HIP_CHECK(hipLaunchKernel(reinterpret_cast<const void*>(function), dim3(1), dim3(kSize), args, 0,
                            nullptr));
  std::vector<int> host(kSize);
  HIP_CHECK(hipMemcpy(host.data(), data, kSize * sizeof(int), hipMemcpyDeviceToHost));
  REQUIRE(std::all_of(host.begin(), host.end(), [](int v) { return v == 2; }));

  HIP_CHECK(hipFuncSetAttribute(reinterpret_cast<const void*>(function),
                                hipFuncAttributeMaxDynamicSharedMemorySize, 0));

  HIP_CHECK(hipModuleUnload(module));
  HIP_CHECK_ERROR(hipLaunchKernel(reinterpret_cast<const void*>(function), dim3(1), dim3(kSize),
                                  args, 0, nullptr),
                  hipErrorInvalidDeviceFunction);
  HIP_CHECK_ERROR(hipFuncSetAttribute(reinterpret_cast<const void*>(function),
                                      hipFuncAttributeMaxDynamicSharedMemorySize, 0),
                  hipErrorInvalidDeviceFunction);
  HIP_CHECK(hipFree(data));
}

/**
 * End doxygen group ModuleTest.
 * @}
 */
