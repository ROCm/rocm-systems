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

TEMPLATE_TEST_CASE("Unit_hipMemcpyHtoAAsync_Basic", "", char, int, float) {
  CHECK_IMAGE_SUPPORT

  hipArray_t dst_array = nullptr;
  std::vector<TestType> host_src(N);
  std::vector<TestType> host_dst(N);
  size_t copy_size = N * sizeof(TestType);
  size_t offset = GENERATE(0, N * sizeof(TestType) / 2);

  std::iota(host_src.begin(), host_src.end(), 0);

  auto channel_descriptor = hipCreateChannelDesc<TestType>();
  HIP_CHECK(hipMallocArray(&dst_array, &channel_descriptor, copy_size));

  HIP_CHECK(hipMemcpyHtoAAsync(dst_array, offset, host_src.data(), copy_size - offset, nullptr));

  HIP_CHECK(hipStreamSynchronize(nullptr));

  HIP_CHECK(hipMemcpyAtoH(host_dst.data(), dst_array, offset, copy_size - offset));

  for (int i = 0; i < offset / sizeof(TestType); i++) {
    if (host_src[i] != host_dst[i]) {
      REQUIRE(false);
    }
  }

  HIP_CHECK(hipFreeArray(dst_array));
}

TEST_CASE("Unit_hipMemcpyHtoAAsync_Negative") {
  CHECK_IMAGE_SUPPORT

  hipArray_t dst_array = nullptr;
  std::vector<int> host_src(N);
  size_t copy_size = N * sizeof(int);

  std::iota(host_src.begin(), host_src.end(), 0);

  auto channel_descriptor = hipCreateChannelDesc<int>();
  HIP_CHECK(hipMallocArray(&dst_array, &channel_descriptor, copy_size));

  SECTION("dst array is null") {
    HIP_CHECK_ERROR(hipMemcpyHtoAAsync(nullptr, 0, host_src.data(), copy_size, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("src is nullptr") {
    HIP_CHECK_ERROR(hipMemcpyHtoAAsync(dst_array, 0, nullptr, copy_size, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("copy size is negative") {
    HIP_CHECK_ERROR(hipMemcpyHtoAAsync(dst_array, 0, host_src.data(), -1, nullptr),
                    hipErrorInvalidValue);
  }

  HIP_CHECK(hipFreeArray(dst_array));
}
