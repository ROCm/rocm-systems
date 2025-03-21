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

#if defined(__HIP_PLATFORM_AMD__) || CUDA_VERSION < CUDA_12000

TEST_CASE("Unit_hipTexRefSetAddressMode_Negative_Parameters") {
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

TEST_CASE("Unit_hipTexRefSetAddressMode_Positive") {
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
