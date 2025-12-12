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
 * @addtogroup hipMemsetD2D16 hipMemsetD2D16
 * @{
 * @ingroup MemoryTest
 * `hipError_t hipMemsetD2D16(hipDeviceptr_t dst, size_t dstPitch, unsigned short value,
 *                            size_t width, size_t height);` -
 * Fills 2D memory range of 'width' 16-bit values synchronously to the specified short value.
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
 * - catch/unit/memory/hipMemsetD2D16.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Unit_hipMemsetD2D16_BasicFunctional) {
  constexpr uint16_t memsetval = static_cast<uint16_t>(0xDEADBEEF);
  constexpr size_t numH = 256;
  constexpr size_t numW = 256;
  size_t pitch_A;
  size_t width = numW * sizeof(uint16_t);
  size_t sizeElements = numW * numH;

  hipDeviceptr_t A_d;
  HIP_CHECK(hipMemAllocPitch(&A_d, &pitch_A, width, numH,
                             2 * sizeof(uint16_t)));
  std::vector<uint16_t>A_h(sizeElements, 1);

  HIP_CHECK(hipMemsetD2D16(A_d, pitch_A, memsetval, width, numH));
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
 * - catch/unit/memory/hipMemsetD2D16.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Unit_hipMemsetD2D16_UnEvenRowsCols) {
  hipDeviceptr_t A_d;
  constexpr uint16_t memsetVal = 5;
  int rows, cols;
  rows = GENERATE(3, 4, 100);
  cols = GENERATE(5, 6, 100);
  size_t devPitch;

  size_t size = rows * cols;
  std::vector<uint16_t>B_h(size, 1);

  HIP_CHECK(hipMemAllocPitch(&A_d, &devPitch, sizeof(uint16_t) * cols,
                             rows, 2 * sizeof(uint16_t)));

  HIP_CHECK(hipMemsetD2D16(A_d, devPitch, memsetVal, sizeof(uint16_t) * cols, rows));
  HIP_CHECK(hipMemcpy2D(B_h.data(), sizeof(uint16_t) * cols, reinterpret_cast<void*>(A_d), devPitch, sizeof(uint16_t) * cols, rows,
                        hipMemcpyDeviceToHost));

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      INFO("Memset2D mismatch at index:" << i << " computed:" << B_h[i * cols + j]
                                              << " memsetval:" << memsetVal);
      REQUIRE(B_h[i * cols + j] == memsetVal);
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
 * - catch/unit/memory/hipMemsetD2D16.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Unit_hipMemsetD2D16_NegTsts) {
  hipDeviceptr_t A_d;
  constexpr size_t numH = 256;
  constexpr size_t numW = 256;
  size_t width = numW * sizeof(uint16_t);
  size_t devPitch;
  constexpr uint16_t memsetval = static_cast<uint16_t>(0x26);
  HIP_CHECK(hipMemAllocPitch(&A_d, &devPitch, width, numH,
                             2 * sizeof(uint16_t)));
  SECTION("nullptr destination") {
    HIP_CHECK_ERROR(hipMemsetD2D16(NULL, devPitch, memsetval, numW, numH), hipErrorInvalidValue);
  }
  SECTION("Dst pointer points to Source Memory") {
    hipDeviceptr_t B_d;
    std::unique_ptr<uint16_t[]> hostPtr;
    hostPtr.reset(new uint16_t[numH * width]);
    B_d = reinterpret_cast<hipDeviceptr_t>(hostPtr.get());
    HIP_CHECK_ERROR(hipMemsetD2D16(B_d, devPitch, memsetval, numW, numH), hipErrorInvalidValue);
  }
  SECTION("Invalid Pitch") {
    size_t inValidPitch = 1;
    HIP_CHECK_ERROR(hipMemsetD2D16(A_d, inValidPitch, memsetval, numW, numH), hipErrorInvalidValue);
  }
  SECTION("Negative Values of Hight, Width") {
    HIP_CHECK_ERROR(hipMemsetD2D16(A_d, devPitch, memsetval, numW, -10), hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipMemsetD2D16(A_d, devPitch, memsetval, -10, numH), hipErrorInvalidValue);
  }
  HIP_CHECK(hipFree(reinterpret_cast<void*>(A_d)));
}
/**
 * End doxygen group MemoryTest.
 * @}
 */
