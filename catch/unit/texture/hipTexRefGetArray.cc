/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>

#if defined(__HIP_PLATFORM_AMD__) || CUDA_VERSION < CUDA_12000

texture<float, 1, hipReadModeElementType> tex;

TEST_CASE("Unit_hipTexRefGetArray_Positive") {
  CHECK_IMAGE_SUPPORT
  hipArray_t array_set = nullptr;
  hipArray_t array_get = nullptr;
  hipModule_t module = nullptr;
  hipTexRef tex_ref = nullptr;
  HIP_ARRAY_DESCRIPTOR array_desc;

  array_desc.Format = HIP_AD_FORMAT_FLOAT;
  array_desc.NumChannels = 1;
  array_desc.Width = 16;
  array_desc.Height = 16;

  HIP_CHECK(hipFree(nullptr));
  HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
  HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));
  HIP_CHECK(hipArrayCreate(&array_set, &array_desc));
  HIP_CHECK(hipTexRefSetArray(tex_ref, array_set, HIP_TRSA_OVERRIDE_FORMAT));
  HIP_CHECK(hipTexRefGetArray(&array_get, tex_ref));
  REQUIRE(array_get == array_set);
  HIP_CHECK(hipArrayDestroy(array_set));
  HIP_CHECK(hipModuleUnload(module));
}

TEST_CASE("Unit_hipTexRefGetArray_Negative") {
  CHECK_IMAGE_SUPPORT
  hipArray_t array_set = nullptr;
  hipArray_t array_get = nullptr;
  hipModule_t module = nullptr;
  hipTexRef tex_ref = nullptr;
  HIP_ARRAY_DESCRIPTOR array_desc;

  array_desc.Format = HIP_AD_FORMAT_FLOAT;
  array_desc.NumChannels = 1;
  array_desc.Width = 16;
  array_desc.Height = 16;

  HIP_CHECK(hipFree(nullptr));
  HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
  HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));
  HIP_CHECK(hipArrayCreate(&array_set, &array_desc));
  HIP_CHECK(hipTexRefSetArray(tex_ref, array_set, HIP_TRSA_OVERRIDE_FORMAT));

// Cuda crashes with SIGSEGV
#if HT_AMD
  SECTION("array is null") {
    HIP_CHECK_ERROR(hipTexRefGetArray(nullptr, tex_ref), hipErrorInvalidValue);
  }
#endif

  SECTION("texture reference is null") {
#if HT_AMD
    HIP_CHECK_ERROR(hipTexRefGetArray(&array_get, nullptr), hipErrorInvalidValue);
#else
    HIP_CHECK_ERROR(hipTexRefGetArray(&array_get, nullptr), hipErrorInvalidResourceHandle);
#endif
  }

  HIP_CHECK(hipArrayDestroy(array_set));
  HIP_CHECK(hipModuleUnload(module));
}

#endif
