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

#include <numeric>

#define N 32

TEMPLATE_TEST_CASE("Unit_hipMemcpyDtoA_Basic", "", char, int, float) {
  CHECK_IMAGE_SUPPORT

  hipArray_t dst_array = nullptr;
  hipDeviceptr_t src_device;
  std::vector<TestType> src_host(N);
  std::vector<TestType> dst_host(N);
  size_t copy_size = N * sizeof(TestType);
  size_t offset = GENERATE(0, N * sizeof(TestType) / 2);

  std::iota(src_host.begin(), src_host.end(), 0);

  auto channel_descriptor = hipCreateChannelDesc<TestType>();
  HIP_CHECK(hipMallocArray(&dst_array, &channel_descriptor, copy_size));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&src_device), copy_size));

  HIP_CHECK(hipMemcpyHtoD(src_device, src_host.data(), copy_size));
  HIP_CHECK(hipMemcpyDtoA(dst_array, offset, src_device, copy_size - offset));
  HIP_CHECK(hipMemcpyAtoH(dst_host.data(), dst_array, offset, copy_size - offset));

  for (int index = 0; index < offset / sizeof(TestType); index++) {
    if (src_host[index] != dst_host[index]) {
      REQUIRE(false);
    }
  }

  HIP_CHECK(hipFreeArray(dst_array));
  HIP_CHECK(hipFree(reinterpret_cast<void*>(src_device)));
}

TEST_CASE("Unit_hipMemcpyDtoA_Negative") {
  CHECK_IMAGE_SUPPORT

  hipArray_t dst_array = nullptr;
  hipDeviceptr_t src_device;
  size_t copy_size = N * sizeof(int);

  auto channel_descriptor = hipCreateChannelDesc<int>();
  HIP_CHECK(hipMallocArray(&dst_array, &channel_descriptor, copy_size));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&src_device), copy_size));

  SECTION("dst array is null") {
    HIP_CHECK_ERROR(hipMemcpyDtoA(nullptr, 0, src_device, copy_size), hipErrorInvalidValue);
    HIP_CHECK(hipFree(reinterpret_cast<void*>(src_device)));
  }

  SECTION("src device ptr is invalid") {
    HIP_CHECK(hipFree(reinterpret_cast<void*>(src_device)));
    HIP_CHECK_ERROR(hipMemcpyDtoA(dst_array, 0, src_device, copy_size), hipErrorInvalidValue);
  }

  HIP_CHECK(hipFreeArray(dst_array));
}

/**
 * Test Description
 * ------------------------
 * - Test case to validate basic functionality of hipMemcpyDtoA,
 *   Step 1 : Take two host arrays srcHost, dstHost(fill with value)
 *   Step 2 : Fill srcHost with valid value and dstHost with 0
 *   Step 3 : Take srcDevice and copy data from srcHost to srcDevice
 *   Step 4 : Take the array and copy data from srcDevice to array
 *   Step 5 : Copy data from array to dstHost
 *   Step 6 : Validate dstHost, it should contain valid value
 * Test source
 * ------------------------
 *    - catch/unit/memory/hipMemcpyDtoA.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMemcpyDtoA_BasicPositive") {
  CHECK_IMAGE_SUPPORT

  const size_t width = 1024;
  const size_t height = 1;
  const int size = width * height;
  const int sizeBytes = size * sizeof(int);
  const int value = 200;

  std::vector<int> srcHost(size), dstHost(size);
  for (int i = 0; i < size; i++) {
    srcHost[i] = value;
    dstHost[i] = 0;
  }

  int* srcDevice = nullptr;
  HIP_CHECK(hipMalloc(&srcDevice, sizeBytes));
  REQUIRE(srcDevice != nullptr);
  HIP_CHECK(hipMemcpy(srcDevice, srcHost.data(), sizeBytes, hipMemcpyHostToDevice));

  hipChannelFormatDesc desc = hipCreateChannelDesc<int>();
  unsigned int flags = hipArrayDefault;
  hipArray_t array = nullptr;
  HIP_CHECK(hipMallocArray(&array, &desc, width, height, flags));
  REQUIRE(array != nullptr);

  HIP_CHECK(hipMemcpyDtoA(array, 0, reinterpret_cast<hipDeviceptr_t>(srcDevice), sizeBytes));

  HIP_CHECK(hipMemcpyAtoH(dstHost.data(), array, 0, sizeBytes));
  for (int i = 0; i < size; i++) {
    REQUIRE(dstHost[i] == value);
  }

  HIP_CHECK(hipFree(srcDevice));
  HIP_CHECK(hipFreeArray(array));
}
