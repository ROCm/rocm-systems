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

texture<int, hipTextureType1D, hipReadModeElementType> tex_1D;
texture<int, hipTextureType2D, hipReadModeElementType> tex_2D;
texture<int, hipTextureType3D, hipReadModeElementType> tex_3D;

__global__ void validate_texture1D(int* array, int width) {
#if !__HIP_NO_IMAGE_SUPPORT
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  while (tid < width) {
    array[tid] = tex1D(tex_1D, tid);
    tid += blockDim.x * gridDim.x;
  }
#endif
}

__global__ void validate_texture2D(int* array, int width, int height) {
#if !__HIP_NO_IMAGE_SUPPORT
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  while (tid < height * width) {
    int x = tid % width;
    int y = tid / width;
    array[tid] = tex2D(tex_2D, x, y);
    tid += blockDim.x * gridDim.x;
  }
#endif
}

__global__ void validate_texture3D(int* array, int width, int height, int depth) {
#if !__HIP_NO_IMAGE_SUPPORT
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  while (tid < height * width * depth) {
    int x = tid / (height * width);
    int y = (tid / width) % height;
    int z = tid % width;
    array[tid] = tex3D(tex_3D, z, y, x);
    tid += blockDim.x * gridDim.x;
  }
#endif
}

bool compare_array_elements(int* array1, int* array2, int width, int height = 1, int depth = 1) {
  for (int i = 0; i < depth; i++) {
    for (int j = 0; j < height; j++) {
      for (int k = 0; k < width; k++) {
        if (array1[i * width * height + j * width + k] !=
            array2[i * width * height + j * width + k]) {
          return false;
        }
      }
    }
  }

  return true;
}

TEST_CASE("Unit_hipBindTextureToArray_Negative_Parameters") {
  CHECK_IMAGE_SUPPORT

  constexpr unsigned int width = 16;
  constexpr unsigned int height = 16;
  constexpr unsigned int size = width * height * sizeof(int);
  unsigned int i, j;

  int* data = reinterpret_cast<int*>(malloc(size));
  REQUIRE(data != nullptr);
  memset(data, 0, size);
  for (i = 0; i < height; i++) {
    for (j = 0; j < width; j++) {
      data[i * width + j] = i * width + j;
    }
  }

  hipChannelFormatDesc channelDesc = hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindSigned);

  hipArray_t hipArray;
  HIP_CHECK(hipMallocArray(&hipArray, &channelDesc, width, height));
  HIP_CHECK(hipMemcpyToArray(hipArray, 0, 0, data, size, hipMemcpyHostToDevice));

  tex_2D.addressMode[0] = hipAddressModeWrap;
  tex_2D.addressMode[1] = hipAddressModeWrap;
  tex_2D.filterMode = hipFilterModePoint;
  tex_2D.normalized = 0;

  SECTION("texref == nullptr") {
#if HT_AMD
    HIP_CHECK_ERROR(hipBindTextureToArray(nullptr, hipArray, &channelDesc), hipErrorInvalidValue);
#else
    struct texture<int, 2, hipReadModeElementType>* tex_null = nullptr;
    HIP_CHECK_ERROR(hipBindTextureToArray(tex_null, hipArray, &channelDesc),
                    hipErrorInvalidTexture);
#endif
  }

  SECTION("array == nullptr") {
#if HT_AMD
    HIP_CHECK_ERROR(hipBindTextureToArray(tex_2D, nullptr, channelDesc), hipErrorInvalidValue);
#else
    HIP_CHECK_ERROR(hipBindTextureToArray(tex_2D, nullptr, channelDesc),
                    hipErrorInvalidResourceHandle);
#endif
  }

#if HT_AMD
  SECTION("channelDesc == nullptr") {
    const hipChannelFormatDesc* desc = nullptr;
    HIP_CHECK_ERROR(hipBindTextureToArray(&tex_2D, hipArray, desc), hipErrorInvalidValue);
  }
#endif

  SECTION("invalid channel desc") {
    hipChannelFormatDesc invalid_desc{-1, -1, -1, -1, hipChannelFormatKindSigned};
#if HT_AMD
    HIP_CHECK_ERROR(hipBindTextureToArray(tex_2D, hipArray, invalid_desc), hipErrorInvalidValue);
#else
    HIP_CHECK_ERROR(hipBindTextureToArray(tex_2D, hipArray, invalid_desc),
                    hipErrorInvalidChannelDescriptor);
#endif
  }

  HIP_CHECK(hipFreeArray(hipArray));
  free(data);
}

TEST_CASE("Unit_hipBindTextureToArray_Positive_1D") {
  CHECK_IMAGE_SUPPORT

  constexpr unsigned int width = 1024;
  constexpr unsigned int height = 1;
  constexpr unsigned int size = width * height * sizeof(int);
  unsigned int i;

  int* data = reinterpret_cast<int*>(malloc(size));
  REQUIRE(data != nullptr);
  memset(data, 0, size);
  for (i = 0; i < width; i++) {
    data[i] = i + 1;
  }

  hipChannelFormatDesc desc = hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindSigned);

  hipArray_t array;
  HIP_CHECK(hipMallocArray(&array, &desc, width, height));
  HIP_CHECK(hipMemcpyToArray(array, 0, 0, data, size, hipMemcpyHostToDevice));

  tex_1D.addressMode[0] = hipAddressModeWrap;
  tex_1D.filterMode = hipFilterModePoint;
  tex_1D.normalized = 0;

  HIP_CHECK(hipBindTextureToArray(tex_1D, array, desc));

  int* d_array;
  HIP_CHECK(hipMalloc(&d_array, size));
  validate_texture1D<<<width / 32, 32>>>(d_array, width);

  int* out_array = reinterpret_cast<int*>(malloc(size));
  REQUIRE(out_array != nullptr);
  HIP_CHECK(hipMemcpy(out_array, d_array, size, hipMemcpyDeviceToHost));

  bool valid = compare_array_elements(data, out_array, width);
  REQUIRE(valid == true);

  HIP_CHECK(hipFree(d_array));
  HIP_CHECK(hipUnbindTexture(tex_1D));
  HIP_CHECK(hipFreeArray(array));
  free(out_array);
  free(data);
}

TEST_CASE("Unit_hipBindTextureToArray_Positive_2D") {
  CHECK_IMAGE_SUPPORT

  constexpr unsigned int width = 256;
  constexpr unsigned int height = 256;
  constexpr unsigned int size = width * height * sizeof(int);
  unsigned int i, j;

  int* data = reinterpret_cast<int*>(malloc(size));
  REQUIRE(data != nullptr);
  for (i = 0; i < height; i++) {
    for (j = 0; j < width; j++) {
      data[i * width + j] = i * width + j;
    }
  }

  hipChannelFormatDesc desc = hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindSigned);

  hipArray_t array;
  HIP_CHECK(hipMallocArray(&array, &desc, width, height));
  HIP_CHECK(hipMemcpyToArray(array, 0, 0, data, size, hipMemcpyHostToDevice));

  tex_2D.addressMode[0] = hipAddressModeWrap;
  tex_2D.addressMode[1] = hipAddressModeWrap;
  tex_2D.filterMode = hipFilterModePoint;
  tex_2D.normalized = 0;

  HIP_CHECK(hipBindTextureToArray(tex_2D, array, desc));

  int* d_array;
  HIP_CHECK(hipMalloc(&d_array, size));
  int thread_num = 32;
  int block_num = width * height / thread_num;
  validate_texture2D<<<block_num, thread_num>>>(d_array, width, height);

  int* out_array = reinterpret_cast<int*>(malloc(size));
  REQUIRE(out_array != nullptr);
  HIP_CHECK(hipMemcpy(out_array, d_array, size, hipMemcpyDeviceToHost));

  bool valid = compare_array_elements(data, out_array, width, height);
  REQUIRE(valid == true);

  HIP_CHECK(hipFree(d_array));
  HIP_CHECK(hipUnbindTexture(tex_2D));
  HIP_CHECK(hipFreeArray(array));
  free(out_array);
  free(data);
}

TEST_CASE("Unit_hipBindTextureToArray_Positive_3D") {
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

  hipChannelFormatDesc desc = hipCreateChannelDesc<int>();

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

  int* d_array;
  HIP_CHECK(hipMalloc(&d_array, size));
  int thread_num = 32;
  int block_num = width * height * depth / thread_num;
  thread_num = 2;
  block_num = 2;


  validate_texture3D<<<block_num, thread_num>>>(d_array, width, height, depth);

  int* out_array = reinterpret_cast<int*>(malloc(size));
  REQUIRE(out_array != nullptr);
  HIP_CHECK(hipMemcpy(out_array, d_array, size, hipMemcpyDeviceToHost));

  bool valid = compare_array_elements(data, out_array, width, height, depth);
  REQUIRE(valid == true);

  HIP_CHECK(hipFree(d_array));
  HIP_CHECK(hipUnbindTexture(tex_3D));
  HIP_CHECK(hipFreeArray(array));
  free(out_array);
  free(data);
}

#endif
