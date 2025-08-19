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

texture<float, hipTextureType1D, hipReadModeElementType> tex_1D;
texture<float, hipTextureType2D, hipReadModeElementType> tex_2D;
texture<float, hipTextureType3D, hipReadModeElementType> tex_3D;

TEST_CASE("Unit_hipUnbindTexture_Null_Texture") {
#if HT_AMD
  HIP_CHECK_ERROR(hipUnbindTexture(nullptr), hipErrorInvalidValue);
#else
  struct texture<float, 2, hipReadModeElementType>* null_tex = nullptr;
  HIP_CHECK_ERROR(hipUnbindTexture(null_tex), hipErrorInvalidTexture);
#endif
}

TEST_CASE("Unit_hipUnbindTexture_Positive_1D") {
  CHECK_IMAGE_SUPPORT

  constexpr int size = 1024;
  float* tex_buffer;
  float array[size];
  size_t offset = 0;

  for (int i = 0; i < size; i++) {
    array[i] = i + 1;
  }

  HIP_CHECK(hipMalloc(&tex_buffer, size * sizeof(float)));
  HIP_CHECK(hipMemcpy(tex_buffer, array, size * sizeof(float), hipMemcpyHostToDevice));

  hipChannelFormatDesc desc = hipCreateChannelDesc<float>();
  tex_1D.addressMode[0] = hipAddressModeClamp;
  tex_1D.filterMode = hipFilterModePoint;
  tex_1D.normalized = 0;
  HIP_CHECK(hipBindTexture(&offset, tex_1D, reinterpret_cast<void*>(tex_buffer), desc,
                           size * sizeof(float)));

  HIP_CHECK(hipUnbindTexture(&tex_1D));
  HIP_CHECK(hipFree(tex_buffer));
}

TEST_CASE("Unit_hipUnbindTexture_Positive_2D") {
  CHECK_IMAGE_SUPPORT

  constexpr int width = 1024;
  constexpr int height = 1024;
  float* dev_ptr;
  size_t dev_pitch, tex_offset;

  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&dev_ptr), &dev_pitch, width * sizeof(float),
                           height));

  hipChannelFormatDesc desc = hipCreateChannelDesc<float>();
  tex_2D.addressMode[0] = hipAddressModeWrap;
  tex_2D.addressMode[1] = hipAddressModeWrap;
  tex_2D.filterMode = hipFilterModePoint;
  tex_2D.normalized = 0;
  HIP_CHECK(hipBindTexture2D(&tex_offset, &tex_2D, dev_ptr, &desc, width, height, dev_pitch));

  HIP_CHECK(hipUnbindTexture(tex_2D));
  HIP_CHECK(hipFree(dev_ptr));
}

TEST_CASE("Unit_hipUnbindTexture_Positive_3D") {
  CHECK_IMAGE_SUPPORT

  constexpr unsigned int width = 256;
  constexpr unsigned int height = 256;
  constexpr unsigned int depth = 256;
  constexpr unsigned int size = width * height * depth * sizeof(int);
  unsigned int i, j, k;

  int* data = reinterpret_cast<int*>(malloc(size));
  REQUIRE(data != nullptr);
  for (i = 0; i < depth; i++) {
    for (j = 0; j < height; j++) {
      for (k = 0; k < width; k++) {
        data[i * width * height + j * width + k] = i * width * height + j * width + k;
      }
    }
  }

  hipChannelFormatDesc desc = hipCreateChannelDesc<float>();

  hipArray_t array;
  HIP_CHECK(hipMalloc3DArray(&array, &desc, make_hipExtent(width, height, depth), hipArrayDefault));
  hipMemcpy3DParms copy_params{};
  copy_params.srcPos = make_hipPos(0, 0, 0);
  copy_params.dstPos = make_hipPos(0, 0, 0);
  copy_params.srcPtr = make_hipPitchedPtr(data, width * sizeof(int), width, height);
  copy_params.dstArray = array;
  copy_params.extent = make_hipExtent(width, height, depth);
  copy_params.kind = hipMemcpyHostToDevice;

  HIP_CHECK(hipMemcpy3D(&copy_params));

  tex_3D.addressMode[0] = hipAddressModeWrap;
  tex_3D.addressMode[1] = hipAddressModeWrap;
  tex_3D.filterMode = hipFilterModePoint;
  tex_3D.normalized = 0;

  HIP_CHECK(hipBindTextureToArray(tex_3D, array, desc));

  HIP_CHECK(hipUnbindTexture(tex_3D));
  HIP_CHECK(hipFreeArray(array));
  free(data);
}

#endif
