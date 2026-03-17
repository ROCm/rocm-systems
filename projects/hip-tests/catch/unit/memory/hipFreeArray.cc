/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
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


HIP_TEMPLATE_TEST_CASE(Unit_hipFreeArray_DifferentSizes, uchar2, char, ushort, short, short4,
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

HIP_TEST_CASE(Unit_hipFreeArray_NegativeArray) {
#if HT_NVIDIA
  HIP_CHECK(hipFreeArray(nullptr));
#else
  HIP_CHECK_ERROR(hipFreeArray(nullptr), hipErrorInvalidValue);
#endif
}

HIP_TEST_CASE(Unit_hipFreeArray_DoubleFree) {
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

HIP_TEMPLATE_TEST_CASE(Unit_hipFreeArray_MultiThreaded, char, int, float2, float4) {
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
