/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>
#include <hip_test_defgroups.hh>
/**
 * @addtogroup hipMemsetD2D32 hipMemsetD2D32
 * @{
 * @ingroup MemoryTest
 * `hipError_t hipMemsetD2D32(hipDeviceptr_t dst, size_t dstPitch, unsigned int value, size_t width,
                          size_t height)` -
 * Fills 2D memory range of 'width' 32-bit values synchronously to thespecified int value.
 * Height specifies numbers of rows to set and dstPitch speicifies the number of bytes between each
 * row.
 */
/**
 * Test Description
 * ------------------------
 * - Checks that allocated buffers have the expected value
 * after setting it to a known constant.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemsetD2D32.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE(Unit_hipMemsetD2D32_BasicFunctional) {
  constexpr int memsetval = 15;
  constexpr size_t numH = 256;
  constexpr size_t numW = 256;
  size_t pitch_A;
  size_t width = numW * sizeof(int);
  size_t sizeElements = numW * numH;
  hipDeviceptr_t A_d;

  std::vector<int>A_h(sizeElements, 1);
  HIP_CHECK(hipMemAllocPitch(&A_d, &pitch_A, width, numH, sizeof(int)));

  HIP_CHECK(hipMemsetD2D32(A_d, pitch_A, memsetval, width, numH));
  HIP_CHECK(hipMemcpy2D(A_h.data(), width, reinterpret_cast<void *>(A_d), pitch_A, width, numH, hipMemcpyDeviceToHost));

  for (size_t i = 0; i < sizeElements; i++) {
    INFO("Memset2D mismatch at index:" << i << " computed:" << A_h[i]
                                       << " memsetval:" << memsetval);
    REQUIRE(A_h[i] == memsetval);
  }
  HIP_CHECK(hipFree(reinterpret_cast<void *>(A_d)));
}
/**
 * Test Description
 * ------------------------
 * - Uneven width and Hight 2D Memory.
 * - Checks that allocated buffers have the expected value
 * after setting it to a known constant.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemsetD2D32.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE(Unit_hipMemsetD2D32_UnEvenRowsCols) {
  constexpr int memsetval = 5;
  hipDeviceptr_t A_d;
  int rows, cols;
  rows = GENERATE(3, 4, 100);
  cols = GENERATE(3, 4, 100);
  size_t devPitch;

  size_t size = rows * cols;
  std::vector<int>B_h(size, 1);
  HIP_CHECK(hipMemAllocPitch(&A_d, &devPitch, sizeof(int) * cols, rows,
                             sizeof(int)));

  HIP_CHECK(hipMemsetD2D32(A_d, devPitch, memsetval, sizeof(int) * cols, rows));
  HIP_CHECK(hipMemcpy2D(B_h.data(), sizeof(int) * cols, reinterpret_cast<void *>(A_d), devPitch, sizeof(int) * cols, rows,
                        hipMemcpyDeviceToHost));

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      INFO("Memset2D mismatch at index:" << i << " computed:" << B_h[i * cols + j]
                                              << " memsetval:" << memsetval);
      REQUIRE(B_h[i * cols + j] == memsetval);
    }
  }
  HIP_CHECK(hipFree(reinterpret_cast<void*>(A_d)));
}
/**
 * Test Description
 * ------------------------
 * - Checks function behaviour when provided invalid arguments.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemsetD2D32.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 7.1
 */
TEST_CASE(Unit_hipMemsetD2D32_NegTsts) {
  hipDeviceptr_t A_d;
  constexpr size_t numH = 256;
  constexpr size_t numW = 256;
  size_t width = numW * sizeof(int);
  size_t devPitch;
  constexpr int memsetval = static_cast<int>(0x26);
  HIP_CHECK(hipMemAllocPitch(&A_d, &devPitch, width, numH, sizeof(int)));
  SECTION("nullptr destination") {
    HIP_CHECK_ERROR(hipMemsetD2D32(NULL, devPitch, memsetval, numW, numH), hipErrorInvalidValue);
  }
  SECTION("OutOfBound destination") {
    void* outOfBoundsDst{reinterpret_cast<int*>(A_d) + devPitch * numH + 1};
    HIP_CHECK_ERROR(hipMemsetD2D32(reinterpret_cast<hipDeviceptr_t>(outOfBoundsDst), devPitch, memsetval, numW, numH),
                    hipErrorInvalidValue);
  }
  SECTION("Dst pointer points to Source Memory") {
    hipDeviceptr_t B_d;
    std::unique_ptr<int[]> hostPtr;
    hostPtr.reset(new int[numH * width]);
    B_d = reinterpret_cast<hipDeviceptr_t>(hostPtr.get());
    HIP_CHECK_ERROR(hipMemsetD2D32(B_d, devPitch, memsetval, numW, numH), hipErrorInvalidValue);
  }
  SECTION("Invalid Pitch") {
    size_t inValidPitch = 1;
    HIP_CHECK_ERROR(hipMemsetD2D32(A_d, inValidPitch, memsetval, numW, numH), hipErrorInvalidValue);
  }
  SECTION("Negative Values of Hight, Width") {
    HIP_CHECK_ERROR(hipMemsetD2D32(A_d, devPitch, memsetval, numW, -10), hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipMemsetD2D32(A_d, devPitch, memsetval, -10, numH), hipErrorInvalidValue);
  }
  HIP_CHECK(hipFree(reinterpret_cast<void *>(A_d)));
}
/**
 * End doxygen group MemoryTest.
 * @}
 */
