/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma clang diagnostic ignored "-Wunused-parameter"
#include <hip_test_common.hh>
#include <hip/texture_types.h>

#if defined(__HIP_PLATFORM_AMD__) || CUDA_VERSION < CUDA_12000

HIP_TEST_CASE(Unit_hipTexRefSetMaxAnisotropy_Negative_Parameters) {
  CHECK_IMAGE_SUPPORT

  hipCtx_t ctx;
  hipDevice_t device;

  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipCtxCreate(&ctx, 0, device));

  hipTexRef tex_ref = nullptr;
  hipModule_t module = nullptr;
  HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
  HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));

  int max_anisotropy = 4;

  SECTION("Null texture") {
#if HT_AMD
    HIP_CHECK_ERROR(hipTexRefSetMaxAnisotropy(nullptr, max_anisotropy), hipErrorInvalidValue);
#else
    HIP_CHECK_ERROR(hipTexRefSetMaxAnisotropy(nullptr, max_anisotropy),
                    hipErrorInvalidResourceHandle);
#endif
  }

  HIP_CHECK(hipModuleUnload(module));
  HIP_CHECK(hipCtxDestroy(ctx));
}

HIP_TEST_CASE(Unit_hipTexRefSetMaxAnisotropy_Positive) {
  CHECK_IMAGE_SUPPORT

  hipCtx_t ctx;
  hipDevice_t device;

  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipCtxCreate(&ctx, 0, device));

  hipTexRef tex_ref = nullptr;
  hipModule_t module = nullptr;
  HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
  HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));

  unsigned int max_anisotropy = GENERATE(1, 2, 4, 16);
  HIP_CHECK(hipTexRefSetMaxAnisotropy(tex_ref, max_anisotropy));

  HIP_CHECK(hipModuleUnload(module));
  HIP_CHECK(hipCtxDestroy(ctx));
}

#endif
