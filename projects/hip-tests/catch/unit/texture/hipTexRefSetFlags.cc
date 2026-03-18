/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma clang diagnostic ignored "-Wunused-parameter"
#include <hip_test_common.hh>
#include <hip/texture_types.h>

#if defined(__HIP_PLATFORM_AMD__) || CUDA_VERSION < CUDA_12000

TEST_CASE(Unit_hipTexRefSetFlags_Negative_Parameters) {
  CHECK_IMAGE_SUPPORT

  hipCtx_t ctx;
  hipDevice_t device;

  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipCtxCreate(&ctx, 0, device));

  hipTexRef tex_ref = nullptr;
  hipModule_t module = nullptr;
  HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
  HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));

  SECTION("Null texture") {
#if HT_AMD
    HIP_CHECK_ERROR(hipTexRefSetFlags(nullptr, HIP_TRSF_READ_AS_INTEGER), hipErrorInvalidValue);
#else
    HIP_CHECK_ERROR(hipTexRefSetFlags(nullptr, HIP_TRSF_READ_AS_INTEGER),
                    hipErrorInvalidResourceHandle);
#endif
  }

  HIP_CHECK(hipModuleUnload(module));
  HIP_CHECK(hipCtxDestroy(ctx));
}

TEST_CASE(Unit_hipTexRefSetFlags_Positive) {
  CHECK_IMAGE_SUPPORT

  hipCtx_t ctx;
  hipDevice_t device;

  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipCtxCreate(&ctx, 0, device));

  hipTexRef tex_ref = nullptr;
  hipModule_t module = nullptr;
  HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
  HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));

  unsigned int flags =
      GENERATE(HIP_TRSF_READ_AS_INTEGER, HIP_TRSF_NORMALIZED_COORDINATES, HIP_TRSF_SRGB);
  HIP_CHECK(hipTexRefSetFlags(tex_ref, flags));

  HIP_CHECK(hipModuleUnload(module));
  HIP_CHECK(hipCtxDestroy(ctx));
}

#endif
