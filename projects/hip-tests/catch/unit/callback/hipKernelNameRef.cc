/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

/**
 * @addtogroup hipKernelNameRef hipKernelNameRef
 * @{
 * @ingroup CallbackTest
 * `hipKernelNameRef(const hipFunction_t f)` -
 * returns the name of passed function object
 */

/**
 * Test Description
 * ------------------------
 *  - Loads the simple kernel function from the matching module
 *  - Checks that the valid name is returned for the loaded kernel function
 * Test source
 * ------------------------
 *  - unit/callback/hipKernelNameRef.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 *  - Platform specific (AMD)
 */
HIP_TEST_CASE(Unit_hipKernelNameRef_Positive_Basic) {
  hipModule_t kernel_module{nullptr};
  hipFunction_t kernel_function{nullptr};

  HIP_CHECK(hipModuleLoad(&kernel_module, "SimpleKernel.code"));
  HIP_CHECK(hipModuleGetFunction(&kernel_function, kernel_module, "simple_kernel"));
  REQUIRE(hipKernelNameRef(kernel_function) != nullptr);
  HIP_CHECK(hipModuleUnload(kernel_module));
}

/**
 * Test Description
 * ------------------------
 *  - Checks that the API returns nullptr if the passed function is not loaded
 * Test source
 * ------------------------
 *  - unit/callback/hipKernelNameRef.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 *  - Platform specific (AMD)
 */
HIP_TEST_CASE(Unit_hipKernelNameRef_Negative_Parameters) {
  hipFunction_t kernel_function{nullptr};
  REQUIRE(hipKernelNameRef(kernel_function) == nullptr);
}

/**
 * End doxygen group CallbackTest.
 * @}
 */
