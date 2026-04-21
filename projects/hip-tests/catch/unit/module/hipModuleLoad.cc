/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>

HIP_TEST_CASE(Unit_hipModuleLoad_Positive_Basic) {
  HIP_CHECK(hipFree(nullptr));
  hipModule_t module = nullptr;
  HIP_CHECK(hipModuleLoad(&module, "empty_module.code"));
  REQUIRE(module != nullptr);
  HIP_CHECK(hipModuleUnload(module));
}

HIP_TEST_CASE(Unit_hipModuleLoad_Negative_Parameters) {
  HIP_CHECK(hipFree(nullptr));
  hipModule_t module;

  SECTION("module == nullptr") {
    HIP_CHECK_ERROR(hipModuleLoad(nullptr, "empty_module.code"), hipErrorInvalidValue);
  }

  SECTION("fname == nullptr") {
    HIP_CHECK_ERROR(hipModuleLoad(&module, nullptr), hipErrorInvalidValue);
  }

  SECTION("fname == empty string") {
    HIP_CHECK_ERROR(hipModuleLoad(&module, ""), hipErrorInvalidValue);
  }

  SECTION("fname == non existent file") {
    HIP_CHECK_ERROR(hipModuleLoad(&module, "non existent file"), hipErrorFileNotFound);
  }
}

HIP_TEST_CASE(Unit_hipModuleLoad_Negative_Load_From_A_File_That_Is_Not_A_Module) {
  HIP_CHECK(hipFree(nullptr));
  hipModule_t module;

  HIP_CHECK_ERROR(hipModuleLoad(&module, "not_a_module.txt"), hipErrorInvalidImage);
  HIP_CHECK_ERROR(hipModuleLoad(&module, "empty_file.txt"), hipErrorInvalidImage);
}
