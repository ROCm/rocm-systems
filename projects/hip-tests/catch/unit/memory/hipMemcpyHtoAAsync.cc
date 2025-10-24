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
#include <hip_test_checkers.hh>
#include <hip_test_defgroups.hh>
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
/**
* @addtogroup hipMemcpyHtoAAsync hipMemcpyHtoAAsync
* @{
* @ingroup MemoryTest
* `hipError_t hipMemcpyHtoAAsync(hipArray_t dstArray, size_t dstOffset,
*                                const void* srcHost,
                                 size_t ByteCount, hipStream_t stream)` -
* Copies from host memory to a 1D array.
*/

/**
 * Test Description
 * ------------------------
 *  - This testcase is to verify the basic functionality of hipMemcpyHtoAAsync API
 *  with different streams.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpyHtoAAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMemcpyHtoAAsync_BasicTstsWithDiffStreams") {
#if HT_NVIDIA
  HipTest::HIP_SKIP_TEST("API currently unsupported on nvidia, skipping...");
  return;
#else
  CHECK_IMAGE_SUPPORT
  HIP_CHECK(hipSetDevice(0));
  int row, col;
  row = 1;
  col = GENERATE(3, 4, 100);
  int* A_h = reinterpret_cast<int*>(malloc(sizeof(int) * row * col));
  int* B_h = reinterpret_cast<int*>(malloc(sizeof(int) * row * col));
  for (int i = 0; i < (row * col); i++) {
    B_h[i] = i;
  }
  hipArray_t A_a;
  hipChannelFormatDesc desc = hipCreateChannelDesc<int>();
  HIP_CHECK(hipMallocArray(&A_a, &desc, col, row, hipArrayDefault));
  SECTION("With Default Stream") {
    HIP_CHECK(hipMemcpyHtoAAsync(A_a, 0, B_h, sizeof(int) * col * row, 0));
    HIP_CHECK(hipMemcpyAtoHAsync(A_h, A_a, 0, sizeof(int) * col * row, 0));
    HIP_CHECK(hipStreamSynchronize(0));
  }
  SECTION("With User Stream") {
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipMemcpyHtoAAsync(A_a, 0, B_h, sizeof(int) * col * row, stream));
    HIP_CHECK(hipMemcpyAtoHAsync(A_h, A_a, 0, sizeof(int) * col * row, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipStreamDestroy(stream));
  }
  SECTION("With Stream per thread") {
    HIP_CHECK(hipMemcpyHtoAAsync(A_a, 0, B_h, sizeof(int) * col * row, hipStreamPerThread));
    HIP_CHECK(hipMemcpyAtoHAsync(A_h, A_a, 0, sizeof(int) * col * row, hipStreamPerThread));
    HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
  }
  SECTION("With Legacy Stream") {
    HIP_CHECK(hipMemcpyHtoAAsync(A_a, 0, B_h, sizeof(int) * col * row, hipStreamLegacy));
    HIP_CHECK(hipMemcpyAtoHAsync(A_h, A_a, 0, sizeof(int) * col * row, hipStreamLegacy));
    HIP_CHECK(hipStreamSynchronize(hipStreamLegacy));
  }

  for (int i = 0; i < (row * col); i++) {
    REQUIRE(A_h[i] == B_h[i]);
  }
  HIP_CHECK(hipFreeArray(A_a));
  free(A_h);
  free(B_h);
#endif
}
/**
 * Test Description
 * ------------------------
 *  - This testcase is to verify the basic functionality of hipMemcpyHtoAAsync API
 *  with different streams on multiple devices.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpyHtoAAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipMemcpyHtoAAsync_MultiDevice") {
#if HT_NVIDIA
  HipTest::HIP_SKIP_TEST("API currently unsupported on nvidia, skipping...");
  return;
#else
  CHECK_IMAGE_SUPPORT
  int devCount = 0;
  HIP_CHECK(hipGetDeviceCount(&devCount));
  for (int i = 0; i < devCount; i++) {
    HIP_CHECK(hipSetDevice(i));
    int row, col;
    row = 1;
    col = GENERATE(3, 4, 100);
    int* A_h = reinterpret_cast<int*>(malloc(sizeof(int) * row * col));
    int* B_h = reinterpret_cast<int*>(malloc(sizeof(int) * row * col));
    for (int i = 0; i < (row * col); i++) {
      B_h[i] = i;
    }
    hipArray_t A_a;
    hipChannelFormatDesc desc = hipCreateChannelDesc<int>();
    HIP_CHECK(hipMallocArray(&A_a, &desc, col, row, hipArrayDefault));
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipMemcpyHtoAAsync(A_a, 0, B_h, sizeof(int) * col * row, stream));
    HIP_CHECK(hipMemcpyAtoHAsync(A_h, A_a, 0, sizeof(int) * col * row, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    for (int i = 0; i < (row * col); i++) {
      REQUIRE(A_h[i] == B_h[i]);
    }
    HIP_CHECK(hipFreeArray(A_a));
    HIP_CHECK(hipStreamDestroy(stream));
    free(A_h);
    free(B_h);
  }
#endif
}

TEST_CASE("UnitHipMemcpyHtoAAsync_Capture") {
  CHECK_IMAGE_SUPPORT

  auto host_src = std::make_unique<std::vector<int>>(N);
  auto host_dst = std::make_unique<std::vector<int>>(N);
  constexpr size_t kCopySize = N * sizeof(int);
  size_t offset = GENERATE(0, N * sizeof(int) / 2);

  std::iota(host_src->begin(), host_src->end(), 0);

  auto channel_desc = hipCreateChannelDesc<int>();
  hipArray_t dst_array = nullptr;
  HIP_CHECK(hipMallocArray(&dst_array, &channel_desc, kCopySize));

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  GENERATE_CAPTURE();
  BEGIN_CAPTURE(stream);
  HIP_CHECK(hipMemcpyHtoAAsync(dst_array, offset, host_src->data(), kCopySize - offset, nullptr));
  END_CAPTURE(stream);

  HIP_CHECK(hipStreamSynchronize(nullptr));

  HIP_CHECK(hipMemcpyAtoH(host_dst->data(), dst_array, offset, kCopySize - offset));

  for (size_t i = 0; i < offset / sizeof(int); ++i) {
    REQUIRE((*host_src)[i] == (*host_dst)[i]);
  }

  HIP_CHECK(hipFreeArray(dst_array));
  HIP_CHECK(hipStreamDestroy(stream));
}
/**
 * End doxygen group MemoryTest.
 * @}
 */
