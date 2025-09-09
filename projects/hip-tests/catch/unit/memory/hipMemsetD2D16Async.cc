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
#include <hip_test_kernels.hh>
#include <hip_test_defgroups.hh>
/**
 * @addtogroup hipMemsetD2D16Async hipMemsetD2D16Async
 * @{
 * @ingroup MemoryTest
 * `hipError_t hipMemsetD2D16Async(hipDeviceptr_t dst, size_t dstPitch, unsigned short value,
                                   size_t width, size_t height, hipStream_t stream __dparm(0))` -
 * Fills 2D memory range of 'width' 16-bit values asynchronously to the specified short
 * value. Height specifies numbers of rows to set and dstPitch speicifies the number of bytes
 * between each row.
 */
/**
 * Test Description
 * ------------------------
 * - Checks that allocated buffers have the expected value
 * after setting it to a known constant.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemsetD2D16Async.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipMemsetD2D16Async_BasicFunctional") {
  constexpr int memsetval = 0x24;
  constexpr size_t numH = 256;
  constexpr size_t numW = 256;
  size_t pitch_A;
  size_t width = numW * sizeof(uint16_t);
  size_t sizeElements = width * numH;
  size_t elements = numW * numH;
  uint16_t *A_d, *A_h;
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipMemAllocPitch(reinterpret_cast<void**>(&A_d), &pitch_A, width, numH,
                             2 * sizeof(uint16_t)));
  A_h = reinterpret_cast<uint16_t*>(malloc(sizeElements));
  REQUIRE(A_h != nullptr);

  for (size_t i = 0; i < elements; i++) {
    A_h[i] = 1;
  }
  HIP_CHECK(hipMemsetD2D16Async(A_d, pitch_A, memsetval, width, numH, stream));
  HIP_CHECK(hipMemcpy2DAsync(A_h, width, A_d, pitch_A, width, numH, hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  for (size_t i = 0; i < elements; i++) {
    if (A_h[i] != memsetval) {
      INFO("Memset2D mismatch at index:" << i << " computed:" << A_h[i]
                                         << " memsetval:" << memsetval);
      REQUIRE(false);
    }
  }
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(A_d));
  free(A_h);
}
/**
 * Test Description
 * ------------------------
 * - Uneven width and Hight 2D Memory.
 * - Checks that allocated buffers have the expected value
 * after setting it to a known constant.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemsetD2D16Async.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipMemsetD2D16Async_UnEvenRowsCols") {
  uint16_t *A_h, *B_h, *A_d;
  int rows, cols;
  rows = GENERATE(3, 4, 100);
  cols = GENERATE(3, 4, 100);
  size_t devPitch;
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  A_h = reinterpret_cast<uint16_t*>(malloc(sizeof(uint16_t) * rows * cols));
  B_h = reinterpret_cast<uint16_t*>(malloc(sizeof(uint16_t) * rows * cols));
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      A_h[i * cols + j] = 1;
    }
  }
  HIP_CHECK(hipMemAllocPitch(reinterpret_cast<void**>(&A_d), &devPitch, sizeof(uint16_t) * cols,
                             rows, 2 * sizeof(uint16_t)));
  HIP_CHECK(hipMemcpy2DAsync(A_d, devPitch, A_h, sizeof(uint16_t) * cols, sizeof(uint16_t) * cols,
                             rows, hipMemcpyHostToDevice, stream));
  HIP_CHECK(hipMemsetD2D16Async(A_d, devPitch, 5, sizeof(uint16_t) * cols, rows, stream));
  HIP_CHECK(hipMemcpy2DAsync(B_h, sizeof(uint16_t) * cols, A_d, devPitch, sizeof(uint16_t) * cols,
                             rows, hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      REQUIRE(B_h[i * cols + j] == 5);
    }
  }
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(A_d));
  free(A_h);
  free(B_h);
}
/**
 * Test Description
 * ------------------------
 * - Checks that the Kernel allocated buffer has the expected value
 * after setting it to a known constant.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemsetD2D16Async.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipMemsetD2D16Async_KernelOperation") {
  constexpr size_t size = 4096;
  constexpr uint16_t memsetval = static_cast<uint16_t>(0x26);
  constexpr unsigned blocksPerCU = 6;
  constexpr unsigned threadsPerBlock = 256;
  uint16_t *C_h, *A_d, *B_d, *C_d;
  constexpr size_t numH = 256;
  constexpr size_t numW = 256;
  size_t devPitchA, devPitchB, devPitchC;
  size_t width = numW * sizeof(uint16_t);
  size_t sizeElements = width * numH;

  C_h = reinterpret_cast<uint16_t*>(malloc(sizeElements));
  for (int i = 0; i < numH; i++) {
    for (int j = 0; j < numW; j++) {
      C_h[i * numH + j] = 0;
    }
  }
  HIP_CHECK(hipMemAllocPitch(reinterpret_cast<void**>(&A_d), &devPitchA, width, numH,
                             2 * sizeof(uint16_t)));
  HIP_CHECK(hipMemAllocPitch(reinterpret_cast<void**>(&B_d), &devPitchB, width, numH,
                             2 * sizeof(uint16_t)));
  HIP_CHECK(hipMemAllocPitch(reinterpret_cast<void**>(&C_d), &devPitchC, width, numH,
                             2 * sizeof(uint16_t)));

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipMemsetD2D16Async(A_d, devPitchA, memsetval, numW, numH, stream));
  HIP_CHECK(hipMemsetD2D16Async(B_d, devPitchB, memsetval, numW, numH, stream));

  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, size);

  hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock), 0, stream, A_d, B_d,
                     C_d, size);
  HIP_CHECK(
      hipMemcpy2DAsync(C_h, width, C_d, devPitchC, width, numH, hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  for (int i = 0; i < numH; i++) {
    for (int j = 0; j < numW; j++) {
      C_h[i * numH + j] = memsetval + memsetval;
    }
  }
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(C_d));
  free(C_h);
}
/**
 * Test Description
 * ------------------------
 * - Checks function behaviour when provided invalid arguments.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemsetD2D16Async.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipMemsetD2D16Async_NegTsts") {
  uint16_t* A_d;
  constexpr size_t numH = 256;
  constexpr size_t numW = 256;
  size_t width = numW * sizeof(uint16_t);
  size_t devPitch;
  constexpr uint16_t memsetval = static_cast<uint16_t>(0x26);
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipMemAllocPitch(reinterpret_cast<void**>(&A_d), &devPitch, width, numH,
                             2 * sizeof(uint16_t)));
  SECTION("nullptr destination") {
    HIP_CHECK_ERROR(hipMemsetD2D16Async(nullptr, devPitch, memsetval, numW, numH, stream),
                    hipErrorInvalidValue);
  }
  SECTION("OutOfBound destination") {
    void* outOfBoundsDst{reinterpret_cast<uint16_t*>(A_d) + devPitch * numH + 1};
    HIP_CHECK_ERROR(hipMemsetD2D16Async(outOfBoundsDst, devPitch, memsetval, numW, numH, stream),
                    hipErrorInvalidValue);
  }
  SECTION("Dst pointer points to Source Memory") {
    uint16_t* B_d;
    std::unique_ptr<uint16_t[]> hostPtr;
    hostPtr.reset(new uint16_t[numH * width]);
    B_d = hostPtr.get();
    HIP_CHECK_ERROR(hipMemsetD2D16Async(B_d, devPitch, memsetval, numW, numH, stream),
                    hipErrorInvalidValue);
  }
  SECTION("Invalid Pitch") {
    size_t inValidPitch = 1;
    HIP_CHECK_ERROR(hipMemsetD2D16Async(A_d, inValidPitch, memsetval, numW, numH, stream),
                    hipErrorInvalidValue);
  }
  SECTION("Negative Values of Hight, Width") {
    HIP_CHECK_ERROR(hipMemsetD2D16Async(A_d, devPitch, memsetval, numW, -10, stream),
                    hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipMemsetD2D16Async(A_d, devPitch, memsetval, -10, numH, stream),
                    hipErrorInvalidValue);
  }
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipStreamDestroy(stream));
}
/**
 * End doxygen group MemoryTest.
 * @}
 */
