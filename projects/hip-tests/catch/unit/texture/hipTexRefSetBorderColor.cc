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

// It will be added for HIP in ROCm7.0
#if HT_NVIDIA
TEST_CASE("Unit_hipTexRefSetBorderColor_Positive") {
  float set_border_color[3] = {1, 2, 3};
  float get_border_color[3] = {0, 0, 0};
  hipModule_t module = nullptr;
  hipTexRef tex_ref = nullptr;

  HIP_CHECK(hipFree(nullptr));
  HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
  HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));

  HIP_CHECK(hipTexRefSetBorderColor(tex_ref, set_border_color));
  HIP_CHECK(hipTexRefGetBorderColor(get_border_color, tex_ref));

  for (int i = 0; i < 3; i++) {
    REQUIRE(set_border_color[i] == get_border_color[i]);
  }
  HIP_CHECK(hipModuleUnload(module));
}
#endif

TEST_CASE("Unit_hipTexRefSetBorderColor_Negative") {
  CHECK_IMAGE_SUPPORT
  float border_color[3] = {0, 0, 0};
  hipModule_t module = nullptr;
  hipTexRef tex_ref = nullptr;

  HIP_CHECK(hipFree(nullptr));
  HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
  HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));

  SECTION("border_color is null") {
    HIP_CHECK_ERROR(hipTexRefSetBorderColor(tex_ref, nullptr), hipErrorInvalidValue);
  }

  SECTION("texture reference is null") {
#if HT_AMD
    HIP_CHECK_ERROR(hipTexRefSetBorderColor(nullptr, border_color), hipErrorInvalidValue);
#else
    HIP_CHECK_ERROR(hipTexRefSetBorderColor(nullptr, border_color), hipErrorInvalidResourceHandle);
#endif
  }

  HIP_CHECK(hipModuleUnload(module));
}

#endif
