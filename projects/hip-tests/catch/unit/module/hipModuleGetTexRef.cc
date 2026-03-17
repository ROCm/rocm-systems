/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hip_module_common.hh"

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>

#if defined(__HIP_PLATFORM_AMD__) || CUDA_VERSION < CUDA_12000

static hipModule_t GetModule() {
  HIP_CHECK(hipFree(nullptr));
  static const auto mg = ModuleGuard::LoadModule("get_tex_ref_module.code");
  return mg.module();
}

TEST_CASE(Unit_hipModuleGetTexRef_Positive_Basic) {
  CHECK_IMAGE_SUPPORT
  hipTexRef tex_ref = nullptr;
  HIP_CHECK(hipModuleGetTexRef(&tex_ref, GetModule(), "tex"));
  REQUIRE(tex_ref != nullptr);
}

TEST_CASE(Unit_hipModuleGetTexRef_Negative_Parameters) {
  CHECK_IMAGE_SUPPORT
  hipModule_t module = GetModule();
  hipTexRef tex_ref = nullptr;

  SECTION("texRef == nullptr") {
    HIP_CHECK_ERROR(hipModuleGetTexRef(nullptr, module, "tex"), hipErrorInvalidValue);
  }

  SECTION("name == nullptr") {
    HIP_CHECK_ERROR(hipModuleGetTexRef(&tex_ref, module, nullptr), hipErrorInvalidValue);
  }

  SECTION("name == non existent texture") {
    HIP_CHECK_ERROR(hipModuleGetTexRef(&tex_ref, module, "non_existent_texture"), hipErrorNotFound);
  }
}

TEST_CASE(Unit_hipModuleGetTexRef_Negative_Hmod_Is_Nullptr) {
  CHECK_IMAGE_SUPPORT
  hipTexRef tex_ref = nullptr;

  CTX_CREATE();
  HIP_CHECK_ERROR(hipModuleGetTexRef(&tex_ref, nullptr, "tex"), hipErrorInvalidResourceHandle);
  CTX_DESTROY();
}

TEST_CASE(Unit_hipModuleGetTexRef_Negative_Name_Is_Empty_String) {
  CHECK_IMAGE_SUPPORT
  hipModule_t module = GetModule();
  hipTexRef tex_ref = nullptr;

  HIP_CHECK_ERROR(hipModuleGetTexRef(&tex_ref, module, ""), hipErrorInvalidValue);
}

#endif
