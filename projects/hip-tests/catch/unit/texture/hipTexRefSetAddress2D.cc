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

TEST_CASE("Unit_hipTexRefSetAddress2D_Negative_Parameters") {
  CHECK_IMAGE_SUPPORT

  constexpr int width = 256;
  constexpr int height = 256;

  hipCtx_t ctx;
  hipDevice_t device;

  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipCtxCreate(&ctx, 0, device));

  hipTexRef tex_ref = nullptr;
  hipModule_t module = nullptr;
  HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
  HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));

  int size = width * height * sizeof(float);
  float* h_data = new float[size];
  for (int i = 0; i < width * height; ++i) {
    h_data[i] = static_cast<float>(i);
  }

  hipDeviceptr_t d_data;
  size_t dest_pitch;
  HIP_CHECK(hipMemAllocPitch(&d_data, &dest_pitch, width * sizeof(float), height, sizeof(float)));
  HIP_CHECK(hipMemcpy2D((void*)d_data, dest_pitch, h_data, width * sizeof(float),
                        width * sizeof(float), height, hipMemcpyHostToDevice));

  HIP_ARRAY_DESCRIPTOR array_desc;
  array_desc.Format = HIP_AD_FORMAT_FLOAT;
  array_desc.Height = height;
  array_desc.Width = width;
  array_desc.NumChannels = 1;

  SECTION("Null texture") {
#if HT_AMD
    HIP_CHECK_ERROR(hipTexRefSetAddress2D(nullptr, &array_desc, d_data, dest_pitch),
                    hipErrorInvalidValue);
#else
    HIP_CHECK_ERROR(hipTexRefSetAddress2D(nullptr, &array_desc, d_data, dest_pitch),
                    hipErrorInvalidResourceHandle);
#endif
  }

  SECTION("Null array descriptor") {
    HIP_CHECK_ERROR(hipTexRefSetAddress2D(tex_ref, nullptr, d_data, dest_pitch),
                    hipErrorInvalidValue);
  }

  free(h_data);
  HIP_CHECK(hipFree((void*)d_data));
  HIP_CHECK(hipModuleUnload(module));
  HIP_CHECK(hipCtxDestroy(ctx));
}

TEST_CASE("Unit_hipTexRefSetAddress2D_Positive") {
  CHECK_IMAGE_SUPPORT

  constexpr int width = 256;
  constexpr int height = 256;

  hipCtx_t ctx;
  hipDevice_t device;

  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipCtxCreate(&ctx, 0, device));

  hipTexRef tex_ref = nullptr;
  hipModule_t module = nullptr;
  HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
  HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));

  int size = width * height * sizeof(float);
  float* h_data = new float[size];
  for (int i = 0; i < width * height; ++i) {
    h_data[i] = static_cast<float>(i);
  }

  hipDeviceptr_t d_data;
  size_t dest_pitch;
  HIP_CHECK(hipMemAllocPitch(&d_data, &dest_pitch, width * sizeof(float), height, sizeof(float)));
  HIP_CHECK(hipMemcpy2D((void*)d_data, dest_pitch, h_data, width * sizeof(float),
                        width * sizeof(float), height, hipMemcpyHostToDevice));

  HIP_ARRAY_DESCRIPTOR array_desc;
  array_desc.Format = HIP_AD_FORMAT_FLOAT;
  array_desc.Height = height;
  array_desc.Width = width;
  array_desc.NumChannels = 1;
  HIP_CHECK(hipTexRefSetAddress2D(tex_ref, &array_desc, d_data, dest_pitch));

  free(h_data);
  HIP_CHECK(hipFree((void*)d_data));
  HIP_CHECK(hipModuleUnload(module));
  HIP_CHECK(hipCtxDestroy(ctx));
}

#endif
