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

/*
hipFreeArray API test scenarios
1. Call hipFreeArray on valid HIP arrays of different types and sizes
2. Negative Scenarios
3. Double free the same HIP array
4. Multithreaded scenario
*/

#include <hip_test_common.hh>
#include <hip_array_common.hh>


TEMPLATE_TEST_CASE("Unit_hipFreeArray_DifferentSizes", "", uchar2, char, ushort, short, short4,
                   uint, int, int4, float, float4) {
  CHECK_IMAGE_SUPPORT

  size_t width = GENERATE(32, 512, 1024);
  size_t height = GENERATE(0, 32, 512, 1024);
  hipArray_t arrayPtr{};
  hipExtent extent{};
  extent.width = width;
  extent.height = height;
  hipChannelFormatDesc desc = hipCreateChannelDesc<TestType>();

  HIP_CHECK(hipMallocArray(&arrayPtr, &desc, extent.width, extent.height, hipArrayDefault));

  HIP_CHECK(hipFreeArray(arrayPtr));
}

TEST_CASE("Unit_hipFreeArray_NegativeArray") {
#if HT_NVIDIA
  HIP_CHECK(hipFreeArray(nullptr));
#else
  HIP_CHECK_ERROR(hipFreeArray(nullptr), hipErrorInvalidValue);
#endif
}

TEST_CASE("Unit_hipFreeArray_DoubleFree") {
#if HT_NVIDIA
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-120");
  return;
#endif

  CHECK_IMAGE_SUPPORT

  size_t width = GENERATE(32, 512, 1024);
  size_t height = GENERATE(0, 32, 512, 1024);
  hipArray_t arrayPtr{};
  hipExtent extent{};
  extent.width = width;
  extent.height = height;
  hipChannelFormatDesc desc = hipCreateChannelDesc<char>();

  HIP_CHECK(hipMallocArray(&arrayPtr, &desc, extent.width, extent.height, hipArrayDefault));

  HIP_CHECK(hipFreeArray(arrayPtr));
  HIP_CHECK_ERROR(hipFreeArray(arrayPtr), hipErrorContextIsDestroyed);
}

/**
 * Test Description
 * ------------------------
 *  - Test to verify hipFreeArray in scenario where multiple threads concurrently allocate
 *    arrays of different types and size and then call hipFreeArray for each array
 */

TEMPLATE_TEST_CASE("Unit_hipFreeArray_MultiThreaded", "", char, int, float2, float4) {
  CHECK_IMAGE_SUPPORT

  constexpr size_t arr_size = 1024;
  std::vector<hipArray_t> arr_ptrs(arr_size);

  size_t width = GENERATE(32, 512, 1024);
  size_t height = GENERATE(0, 32, 512, 1024);
  hipExtent extent{};
  extent.width = width;
  extent.height = height;
  hipChannelFormatDesc desc = hipCreateChannelDesc<TestType>();

  std::vector<std::thread> threads;

  for (auto arr : arr_ptrs) {
    HIP_CHECK(hipMallocArray(&arr, &desc, extent.width, extent.height, hipArrayDefault));

    threads.emplace_back([arr] {
      HIP_CHECK_THREAD(hipFreeArray(arr));
      HIP_CHECK_THREAD(hipStreamQuery(nullptr));
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  HIP_CHECK_THREAD_FINALIZE();
}
