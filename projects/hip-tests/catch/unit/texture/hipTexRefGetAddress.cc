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

#include <hip_test_common.hh>

#if defined(__HIP_PLATFORM_AMD__) || CUDA_VERSION < CUDA_12000

texture<float, 1, hipReadModeElementType> tex;

TEST_CASE("Unit_hipTexRefGetAddress_Positive") {
  CHECK_IMAGE_SUPPORT
  hipDeviceptr_t device_ptr;
  hipModule_t module = nullptr;
  hipTexRef tex_ref = nullptr;
  float* tex_buffer = nullptr;
  size_t offset = 0, tex_size = sizeof(float);

  HIP_CHECK(hipFree(nullptr));
  HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
  HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));
  HIP_CHECK(hipMalloc(&tex_buffer, sizeof(float)));
  HIP_CHECK(hipTexRefSetAddress(&offset, tex_ref, reinterpret_cast<hipDeviceptr_t>(tex_buffer),
                                tex_size));
  HIP_CHECK(hipTexRefGetAddress(&device_ptr, tex_ref));

  REQUIRE(reinterpret_cast<void*>(device_ptr) != nullptr);
  HIP_CHECK(hipModuleUnload(module));
  HIP_CHECK(hipFree(tex_buffer));
}

TEST_CASE("Unit_hipTexRefGetAddress_Negative") {
  CHECK_IMAGE_SUPPORT
  hipDeviceptr_t device_ptr;
  hipModule_t module = nullptr;
  hipTexRef tex_ref = nullptr;
  float* tex_buffer = nullptr;
  size_t offset = 0, tex_size = sizeof(float);

  HIP_CHECK(hipFree(nullptr));
  HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
  HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));
  HIP_CHECK(hipMalloc(&tex_buffer, sizeof(float)));
  HIP_CHECK(hipTexRefSetAddress(&offset, tex_ref, reinterpret_cast<hipDeviceptr_t>(tex_buffer),
                                tex_size));

  SECTION("texture reference is null") {
#if HT_AMD
    HIP_CHECK_ERROR(hipTexRefGetAddress(&device_ptr, nullptr), hipErrorInvalidValue);
#else
    HIP_CHECK_ERROR(hipTexRefGetAddress(&device_ptr, nullptr), hipErrorInvalidResourceHandle);
#endif
  }

  HIP_CHECK(hipModuleUnload(module));
  HIP_CHECK(hipFree(tex_buffer));
}

TEST_CASE("Unit_hipTexRefGetAddress_AdressNotSet") {
  CHECK_IMAGE_SUPPORT
  hipDeviceptr_t device_ptr;
  hipModule_t module = nullptr;
  hipTexRef tex_ref = nullptr;
  float* tex_buffer = nullptr;

  HIP_CHECK(hipFree(nullptr));
  HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
  HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));
  HIP_CHECK(hipMalloc(&tex_buffer, sizeof(float)));
  HIP_CHECK_ERROR(hipTexRefGetAddress(&device_ptr, tex_ref), hipErrorInvalidValue);

  HIP_CHECK(hipModuleUnload(module));
  HIP_CHECK(hipFree(tex_buffer));
}

#endif
