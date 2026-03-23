/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>

/**
 * @addtogroup hipMemcpyAtoHAsync hipMemcpyAtoHAsync
 * @{
 * @ingroup MemoryTest
 * `hipError_t hipMemcpyAtoHAsync(void* dstHost, hipArray_t srcArray,
 *                                size_t srcOffset, size_t ByteCount,
 *                                hipStream_t stream)` -
 * Copies from one 1D array to host memory.
 */

/**
 * Test Description
 * ------------------------
 *  - This testcase initially copies data from host to 1D array and then performs
 *  hipMemcpyAtoHAsync api call and verifies with initial host values.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpyAtoHAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipMemcpyAtoHAsync_Basic) {
#if HT_NVIDIA
  HipTest::HIP_SKIP_TEST("API currently unsupported on nvidia, skipping...");
  return;
#else
  HIP_CHECK(hipSetDevice(0));
  CHECK_IMAGE_SUPPORT
  int row, col;
  row = 1;
  col = GENERATE(3, 4, 100);
  int* A_h = reinterpret_cast<int*>(malloc(sizeof(int) * row * col));
  int* B_h = reinterpret_cast<int*>(malloc(sizeof(int) * row * col));
  for (int i = 0; i < (row * col); i++) {
    A_h[i] = i;
  }
  hipArray_t A_a;
  hipChannelFormatDesc desc = hipCreateChannelDesc<int>();
  HIP_CHECK(hipMallocArray(&A_a, &desc, col, row, hipArrayDefault));
  HIP_CHECK(hipMemcpy2DToArray(A_a, 0, 0, A_h, col * sizeof(int), col * sizeof(int), row,
                               hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpyAtoHAsync(B_h, A_a, 0, sizeof(int) * col * row, 0));
  HIP_CHECK(hipStreamSynchronize(0));
  for (int i = 0; i < (row * col); i++) {
    REQUIRE(A_h[i] == B_h[i]);
  }
  HIP_CHECK(hipFreeArray(A_a));
  free(A_h);
  free(B_h);
#endif
}

HIP_TEST_CASE(Unit_hipMemcpyAtoHAsync_Capture) {
  CHECK_IMAGE_SUPPORT

  constexpr int kRows = 1;
  constexpr int kCols = 1;
  auto host_data = std::make_unique<int[]>(kRows * kCols);

  hipArray_t device_array = nullptr;
  hipChannelFormatDesc channel_desc = hipCreateChannelDesc<int>();
  HIP_CHECK(hipMallocArray(&device_array, &channel_desc, kCols, kRows, hipArrayDefault));

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  GENERATE_CAPTURE();
  BEGIN_CAPTURE(stream);
  HIP_CHECK(
      hipMemcpyAtoHAsync(host_data.get(), device_array, 0, sizeof(int) * kCols * kRows, stream));
  END_CAPTURE(stream);

  HIP_CHECK(hipFreeArray(device_array));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * End doxygen group MemoryTest.
 * @}
 */
