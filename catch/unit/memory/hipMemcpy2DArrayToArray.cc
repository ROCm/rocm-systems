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
#include <hip/hip_runtime_api.h>

bool compare_arrays(int* arr1, int* arr2, int width, int height) {
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      if (arr1[i * width + j] != arr2[i * width + j]) {
        return false;
      }
    }
  }

  return true;
}

TEST_CASE("Unit_hipMemcpy2DArrayToArray_Negative") {
  CHECK_IMAGE_SUPPORT

  constexpr int width = 256;
  constexpr int height = 256;

  constexpr int size = width * height * sizeof(int);
  int h_array[width][height];
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      h_array[i][j] = i + j + 1;
    }
  }

  hipArray_t d_src_arr;
  hipArray_t d_dst_arr;
  hipChannelFormatDesc desc = hipCreateChannelDesc<int>();
  HIP_CHECK(hipMallocArray(&d_src_arr, &desc, width, height));
  HIP_CHECK(hipMallocArray(&d_dst_arr, &desc, width, height));

  HIP_CHECK(hipMemcpyToArray(d_src_arr, 0, 0, h_array, size, hipMemcpyHostToDevice));

  SECTION("Source array is nullptr") {
    HIP_CHECK_ERROR(hipMemcpy2DArrayToArray(d_dst_arr, 0, 0, nullptr, 0, 0, width * sizeof(int),
                                            height, hipMemcpyDeviceToDevice),
                    hipErrorInvalidResourceHandle);
  }

  SECTION("Destination array is nullptr") {
#if HT_AMD
    HIP_CHECK_ERROR(hipMemcpy2DArrayToArray(nullptr, 0, 0, d_src_arr, 0, 0, width * sizeof(int),
                                            height, hipMemcpyDeviceToDevice),
                    hipErrorInvalidResourceHandle);
#else
    HIP_CHECK_ERROR(hipMemcpy2DArrayToArray(nullptr, 0, 0, d_src_arr, 0, 0, width * sizeof(int),
                                            height, hipMemcpyDeviceToDevice),
                    hipErrorInvalidValue);
#endif
  }

  SECTION("Invalid width offset") {
    // width_offset + width > arr->width
    int width_offset = 1;
    HIP_CHECK_ERROR(hipMemcpy2DArrayToArray(d_dst_arr, width_offset, 0, d_src_arr, 0, 0,
                                            width * sizeof(int), height, hipMemcpyDeviceToDevice),
                    hipErrorInvalidValue);

    HIP_CHECK_ERROR(hipMemcpy2DArrayToArray(d_dst_arr, 0, 0, d_src_arr, width_offset, 0,
                                            width * sizeof(int), height, hipMemcpyDeviceToDevice),
                    hipErrorInvalidValue);
  }

  SECTION("Invalid height offset") {
    // height_offset + height > arr->height
    int height_offset = 1;
    HIP_CHECK_ERROR(hipMemcpy2DArrayToArray(d_dst_arr, 0, height_offset, d_src_arr, 0, 0,
                                            width * sizeof(int), height, hipMemcpyDeviceToDevice),
                    hipErrorInvalidValue);

    HIP_CHECK_ERROR(hipMemcpy2DArrayToArray(d_dst_arr, 0, 0, d_src_arr, 0, height_offset,
                                            width * sizeof(int), height, hipMemcpyDeviceToDevice),
                    hipErrorInvalidValue);
  }

  HIP_CHECK(hipFreeArray(d_src_arr));
  HIP_CHECK(hipFreeArray(d_dst_arr));
}

TEST_CASE("Unit_hipMemcpy2DArrayToArray_Positive") {
  CHECK_IMAGE_SUPPORT

  constexpr int width = 4;
  constexpr int height = 4;

  constexpr int size = width * height * sizeof(int);
  int h_array[width][height];
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      h_array[i][j] = i + j + 1;
    }
  }

  hipArray_t d_src_arr;
  hipArray_t d_dst_arr;
  hipChannelFormatDesc desc = hipCreateChannelDesc<int>();
  HIP_CHECK(hipMallocArray(&d_src_arr, &desc, width, height));
  HIP_CHECK(hipMallocArray(&d_dst_arr, &desc, width, height));

  HIP_CHECK(hipMemcpyToArray(d_src_arr, 0, 0, h_array, size, hipMemcpyHostToDevice));

  HIP_CHECK(hipMemcpy2DArrayToArray(d_dst_arr, 0, 0, d_src_arr, 0, 0, width * sizeof(int), height,
                                    hipMemcpyDeviceToDevice));

  int* out_arr = reinterpret_cast<int*>(malloc(size));
  REQUIRE(out_arr != nullptr);
  HIP_CHECK(hipMemcpyFromArray(out_arr, d_dst_arr, 0, 0, size, hipMemcpyDeviceToHost));

  bool valid = compare_arrays((int*)h_array, out_arr, width, height);
  REQUIRE(valid == true);

  free(out_arr);
  HIP_CHECK(hipFreeArray(d_src_arr));
  HIP_CHECK(hipFreeArray(d_dst_arr));
}
