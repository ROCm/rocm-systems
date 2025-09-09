/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma clang diagnostic ignored "-Wunused-parameter"
#include <hip_test_common.hh>
#include <hip/texture_types.h>

#if defined(__HIP_PLATFORM_AMD__) || CUDA_VERSION < CUDA_12000

TEST_CASE("Unit_hipTexRefGetAddressMode_Negative_Parameters") {
  CHECK_IMAGE_SUPPORT

  hipCtx_t ctx;
  hipDevice_t device;

  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipCtxCreate(&ctx, 0, device));

  hipTexRef tex_ref = nullptr;
  hipModule_t module = nullptr;
  HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
  HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));

  int dim = 0;

#if HT_AMD
  hipTextureAddressMode am;
#else
  HIPaddress_mode am;
#endif

  SECTION("Null texture") {
#if HT_AMD
    HIP_CHECK_ERROR(hipTexRefGetAddressMode(&am, nullptr, dim), hipErrorInvalidValue);
#else
    HIP_CHECK_ERROR(hipTexRefGetAddressMode(&am, nullptr, dim), hipErrorInvalidResourceHandle);
#endif
  }

  SECTION("Null address mode") {
    HIP_CHECK_ERROR(hipTexRefGetAddressMode(nullptr, tex_ref, dim), hipErrorInvalidValue);
  }

  SECTION("Invalid dimension") {
    dim = GENERATE(-1, 3);
    HIP_CHECK_ERROR(hipTexRefGetAddressMode(&am, tex_ref, dim), hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipTexRefGetAddressMode(&am, tex_ref, dim), hipErrorInvalidValue);
  }

  HIP_CHECK(hipModuleUnload(module));
  HIP_CHECK(hipCtxDestroy(ctx));
}

TEST_CASE("Unit_hipTexRefGetAddressMode_Positive") {
  CHECK_IMAGE_SUPPORT

  hipCtx_t ctx;
  hipDevice_t device;

  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipCtxCreate(&ctx, 0, device));

  hipTexRef tex_ref = nullptr;
  hipModule_t module = nullptr;
  HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
  HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));

  int dim = 0;
#if HT_AMD
  hipTextureAddressMode am =
      GENERATE(hipAddressModeWrap, hipAddressModeClamp, hipAddressModeMirror, hipAddressModeBorder);
  hipTextureAddressMode out_am;
#else
  HIPaddress_mode am = GENERATE(HIP_TR_ADDRESS_MODE_WRAP, HIP_TR_ADDRESS_MODE_CLAMP);
  HIPaddress_mode out_am;
#endif

  HIP_CHECK(hipTexRefSetAddressMode(tex_ref, dim, am));
  HIP_CHECK(hipTexRefGetAddressMode(&out_am, tex_ref, dim));
  REQUIRE(out_am == am);

  HIP_CHECK(hipModuleUnload(module));
  HIP_CHECK(hipCtxDestroy(ctx));
}

#endif
