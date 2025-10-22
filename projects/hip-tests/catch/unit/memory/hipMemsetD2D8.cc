/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_defgroups.hh>
/**
 * @addtogroup hipMemsetD2D8 hipMemsetD2D8
 * @{
 * @ingroup MemoryTest
 * `hipError_t hipMemsetD2D8(hipDeviceptr_t dst, size_t dstPitch, unsigned char value,
    size_t width, size_t height);` -
 * Fills 2D memory range of 'width' 8-bit values synchronously to the specified char value.
 * Height specifies numbers of rows to set and dstPitch speicifies the number of bytes between each
 * row.
 */
/**
 * Test Description
 * ------------------------
 *  - Checks that allocated buffers have the expected value
 * after setting it to a known constant.
 * Test source
 * ------------------------
 *  - catch/unit/memory/hipMemsetD2D8.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipMemsetD2D8_BasicFunctional") {
  constexpr char memsetval = 'c';
  constexpr size_t numH = 256;
  constexpr size_t numW = 256;
  size_t pitch_A;
  size_t width = numW * sizeof(char);
  size_t sizeElements = numW * numH;
  hipDeviceptr_t A_d;
  HIP_CHECK(
      hipMemAllocPitch(&A_d, &pitch_A, width, numH, 4 * sizeof(char)));
  std::vector<char>A_h(sizeElements, 'a');
  HIP_CHECK(hipMemsetD2D8(A_d, pitch_A, memsetval, width, numH));
  HIP_CHECK(hipMemcpy2D(A_h.data(), width, reinterpret_cast<void*>(A_d), pitch_A, width, numH, hipMemcpyDeviceToHost));

  for (size_t i = 0; i < sizeElements; i++) {
    INFO("Memset2D mismatch at index:" << i << " computed:" << A_h[i]
                                       << " memsetval:" << memsetval);
    REQUIRE(A_h[i] == memsetval);
  }
  HIP_CHECK(hipFree(reinterpret_cast<void*>(A_d)));
}
/**
 * Test Description
 * ------------------------
 * - Uneven width and Hight 2D Memory.
 * - Checks that allocated buffers have the expected value
 * after setting it to a known constant.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemsetD2D8.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipMemsetD2D8_UnEvenRowsCols") {
  hipDeviceptr_t A_d;
  int rows, cols;
  rows = GENERATE(3, 4, 100);
  cols = GENERATE(4, 5, 100);
  size_t devPitch;
  constexpr char memsetval = 'c';
  size_t size = rows * cols;
  std::vector<char>A_h(size, 'a');
  std::vector<char>B_h(size, 'a');
  HIP_CHECK(hipMemAllocPitch(&A_d, &devPitch, sizeof(char) * cols, rows,
                             4 * sizeof(char)));
  HIP_CHECK(hipMemcpy2D(reinterpret_cast<void *>(A_d), devPitch, A_h.data(), sizeof(char) * cols, sizeof(char) * cols, rows,
                        hipMemcpyHostToDevice));

  HIP_CHECK(hipMemsetD2D8(A_d, devPitch, memsetval, sizeof(char) * cols, rows));

  HIP_CHECK(hipMemcpy2D(B_h.data(), sizeof(char) * cols, reinterpret_cast<void *>(A_d), devPitch, sizeof(char) * cols, rows,
                        hipMemcpyDeviceToHost));

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      INFO("Memset2D mismatch at index:" << i << " computed:" << B_h[i * cols + j]
                                         << " memsetval:" << memsetval);
      REQUIRE(B_h[i * cols + j] == memsetval);
    }
  }
  HIP_CHECK(hipFree(reinterpret_cast<void *>(A_d)));
}
/**
 * Test Description
 * ------------------------
 * - Checks function behaviour when provided invalid arguments.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemsetD2D8.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipMemsetD2D8_NegTsts") {
  hipDeviceptr_t A_d;
  constexpr size_t numH = 256;
  constexpr size_t numW = 256;
  size_t width = numW * sizeof(char);
  size_t devPitch;
  constexpr char memsetval = 'c';
  HIP_CHECK(
      hipMemAllocPitch(&A_d, &devPitch, width, numH, 4 * sizeof(char)));
  SECTION("nullptr destination") {
    HIP_CHECK_ERROR(hipMemsetD2D8(NULL, devPitch, memsetval, numW, numH), hipErrorInvalidValue);
  }
  SECTION("OutOfBound destination") {
    void* outOfBoundsDst{reinterpret_cast<char*>(A_d) + devPitch * numH + 1};
    HIP_CHECK_ERROR(hipMemsetD2D8(reinterpret_cast<hipDeviceptr_t>( outOfBoundsDst ), devPitch, memsetval, numW, numH),
                    hipErrorInvalidValue);
  }
  SECTION("Dst pointer points to Source Memory") {
    hipDeviceptr_t B_d;
    std::unique_ptr<char[]> hostPtr;
    hostPtr.reset(new char[numH * width]);
    B_d = reinterpret_cast<hipDeviceptr_t>( hostPtr.get() );
    HIP_CHECK_ERROR(hipMemsetD2D8(B_d, devPitch, memsetval, numW, numH), hipErrorInvalidValue);
  }
  SECTION("Invalid Pitch") {
    size_t inValidPitch = 1;
    HIP_CHECK_ERROR(hipMemsetD2D8(A_d, inValidPitch, memsetval, numW, numH), hipErrorInvalidValue);
  }
  SECTION("Negative Values of Hight, Width") {
    HIP_CHECK_ERROR(hipMemsetD2D8(A_d, devPitch, memsetval, numW, -10), hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipMemsetD2D8(A_d, devPitch, memsetval, -10, numH), hipErrorInvalidValue);
  }
  HIP_CHECK(hipFree(reinterpret_cast<void*>( A_d )));
}
/**
 * End doxygen group MemoryTest.
 * @}
 */
