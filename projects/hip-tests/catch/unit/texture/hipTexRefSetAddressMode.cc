/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma clang diagnostic ignored "-Wunused-parameter"
#include <hip_test_common.hh>

#if defined(__HIP_PLATFORM_AMD__) || CUDA_VERSION < CUDA_12000

HIP_TEST_CASE(Unit_hipTexRefSetAddressMode_Negative_Parameters) {
  CHECK_IMAGE_SUPPORT

  hipCtx_t ctx;
  hipDevice_t device;

  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipCtxCreate(&ctx, 0, device));

  hipTexRef tex_ref = nullptr;
  hipModule_t module = nullptr;
  HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
  HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));

#if HT_AMD
  hipTextureAddressMode am = hipAddressModeWrap;
#else
  HIPaddress_mode am = HIP_TR_ADDRESS_MODE_WRAP;
#endif

  SECTION("Invalid texture") {
#if HT_AMD
    HIP_CHECK_ERROR(hipTexRefSetAddressMode(nullptr, 0, am), hipErrorInvalidValue);
#else
    hipTexRef tex_ref = nullptr;
    HIP_CHECK_ERROR(hipTexRefSetAddressMode(tex_ref, 0, am), hipErrorInvalidResourceHandle);
#endif
  }

  SECTION("Invalid dimension") {
    HIP_CHECK_ERROR(hipTexRefSetAddressMode(tex_ref, -1, am), hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipTexRefSetAddressMode(tex_ref, 3, am), hipErrorInvalidValue);
  }

  HIP_CHECK(hipModuleUnload(module));
  HIP_CHECK(hipCtxDestroy(ctx));
}

HIP_TEST_CASE(Unit_hipTexRefSetAddressMode_Positive) {
  CHECK_IMAGE_SUPPORT

  hipCtx_t ctx;
  hipDevice_t device;

  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipCtxCreate(&ctx, 0, device));

  hipTexRef tex_ref = nullptr;
  hipModule_t module = nullptr;
  HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
  HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));

  auto dim = GENERATE(0, 1, 2);
#if HT_AMD
  auto am =
      GENERATE(hipAddressModeWrap, hipAddressModeClamp, hipAddressModeMirror, hipAddressModeBorder);
#else
  auto am = GENERATE(HIP_TR_ADDRESS_MODE_WRAP, HIP_TR_ADDRESS_MODE_CLAMP,
                     HIP_TR_ADDRESS_MODE_MIRROR, HIP_TR_ADDRESS_MODE_BORDER);
#endif

  HIP_CHECK(hipTexRefSetAddressMode(tex_ref, dim, am));

  HIP_CHECK(hipModuleUnload(module));
  HIP_CHECK(hipCtxDestroy(ctx));
}

#endif
