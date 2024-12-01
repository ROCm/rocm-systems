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
    HIP_CHECK_ERROR(hipMemcpyDtoA(dst_array, 0, src_device, copy_size),
                    hipErrorInvalidValue);
  }

  HIP_CHECK(hipFreeArray(dst_array));
}
