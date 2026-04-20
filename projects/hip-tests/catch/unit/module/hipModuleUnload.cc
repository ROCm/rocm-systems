/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <hip/hip_runtime_api.h>

HIP_TEST_CASE(Unit_hipModuleUnload_Negative_Module_Is_Nullptr) {
  HIP_CHECK(hipFree(nullptr));

  HIP_CHECK_ERROR(hipModuleUnload(nullptr), hipErrorInvalidResourceHandle);
}

HIP_TEST_CASE(Unit_hipModuleUnload_Negative_Double_Unload) {
  HIP_CHECK(hipFree(nullptr));

  hipModule_t module = nullptr;
  HIP_CHECK(hipModuleLoad(&module, "empty_module.code"));
  HIP_CHECK(hipModuleUnload(module));
#if HT_AMD
  HIP_CHECK_ERROR(hipModuleUnload(module), hipErrorNotFound);
#else
  HIP_CHECK_ERROR(hipModuleUnload(module), hipErrorInvalidResourceHandle);
#endif
}
/**
 * @addtogroup hipModuleUnload
 * @{
 * @ingroup ModuleTest
 * `hipError_t hipModuleUnload(hipModule_t module)` -
 * Frees the module
 */

/**
 * Test Description
 * ------------------------
 * - Test case to verify the module release.
 * Test source
 * ------------------------
 * - catch/unit/module/hipModuleUnload.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.6
 */
HIP_TEST_CASE(Unit_hipModuleUnload_basic) {
  CTX_CREATE();
  constexpr auto fileName = "vcpy_kernel.code";
  hipModule_t module;
  HIP_CHECK(hipModuleLoad(&module, fileName));
  REQUIRE(module != nullptr);
  HIP_CHECK(hipModuleUnload(module));
  CTX_DESTROY();
}
