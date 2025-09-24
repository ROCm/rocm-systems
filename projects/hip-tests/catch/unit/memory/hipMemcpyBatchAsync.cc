/*
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
/**
 * @addtogroup hipMemcpyBatchAsync hipMemcpyBatchAsync
 * @{
 * @ingroup MemoryTest
 * `hipError_t hipMemcpyBatchAsync(void** dsts, void** srcs, size_t* sizes,
 size_t count, hipMemcpyAttributes* attrs, size_t* attrsIdxs, size_t numAttrs,
                               size_t* failIdx, hipStream_t stream __dparm(0))`
 -
 * Perform Batch of 1D copies.
 */
/**
 * Test Description
 * ------------------------
 * - Test case to verify the 1D batch memory copy.
 * 1. Create Array of device pointers(Src, Dst).
 * 2. Set the MemcpyBatch params. As of now no support for memcpy Attributes.
 * 3. Perform batch memcpy operation from deviceptr to deviceptr.
 * 4. Validate data on host.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
#if HT_AMD
TEST_CASE("Unit_hipMemcpyBatchAsync_D2D_BasicFunctional") {
  const size_t count = 2;
  size_t numAttrs = 0;
  const size_t size = 4096 * sizeof(char);
  hipStream_t stream = NULL;
  HIP_CHECK(hipStreamCreate(&stream));

  // Allocate buffers for pointer-ptr copy
  void *srcPtr[count], *dstPtr[count];
  std::vector<std::array<char, size>> hostPtr(count);
  size_t sizes[2];
  size_t attrsIdxs[1];
  for (int i = 0; i < count; i++) {
    HIP_CHECK(hipMalloc(&srcPtr[i], size));
    HIP_CHECK(hipMalloc(&dstPtr[i], size));
    HIP_CHECK(hipMemset(srcPtr[i], 'b', size)); // Fill with value
    sizes[i] = size;
  }

  attrsIdxs[0] = 0;
  size_t failIdx;

  HIP_CHECK(hipMemcpyBatchAsync(dstPtr, srcPtr, sizes, count, nullptr,
                                attrsIdxs, numAttrs, &failIdx, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  // validation
  for (int i = 0; i < count; i++) {
    HIP_CHECK(
        hipMemcpy(hostPtr[i].data(), dstPtr[i], size, hipMemcpyDeviceToHost));
    for (int j = 0; j < size; j++) {
      INFO("Array FAILURE at Index: " << i << " " << j
                                      << "\nval : " << hostPtr[i][j]);
      REQUIRE(hostPtr[i][j] == 'b');
    }
  }

  // Clean up
  for (int i = 0; i < count; i++) {
    HIP_CHECK(hipFree(srcPtr[i]));
    HIP_CHECK(hipFree(dstPtr[i]));
  }
  HIP_CHECK(hipStreamDestroy(stream));
}
/**
 * Test Description
 * ------------------------
 * - Test case to verify the 1D batch memory copy.
 * 1. Create Array of device pointers(Src, Dst).
 * 2. Set the MemcpyBatch params. As of now no support for memcpy Attributes.
 * 3. Perform batch memcpy operation From Host to Device.
 * 4. Validate data on host.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipMemcpyBatchAsync_H2D_BasicFunctional") {
  const size_t count = 2;
  size_t numAttrs = 0;
  const size_t size = 4096 * sizeof(char);
  hipStream_t stream = NULL;
  HIP_CHECK(hipStreamCreate(&stream));

  // Allocate buffers for pointer-ptr copy
  void *hostSrcPtr[count], *dstPtr[count];
  std::vector<std::array<char, size>> hostPtr(count);
  std::array<char, size> arr;
  arr.fill('b');
  size_t sizes[2];
  size_t attrsIdxs[1];
  for (int i = 0; i < count; i++) {
    hostSrcPtr[i] = arr.data();
    HIP_CHECK(hipMalloc(&dstPtr[i], size));
    sizes[i] = size;
  }
  attrsIdxs[0] = 0;
  size_t failIdx;

  HIP_CHECK(hipMemcpyBatchAsync(dstPtr, hostSrcPtr, sizes, count, nullptr,
                                attrsIdxs, numAttrs, &failIdx, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  // validation
  for (int i = 0; i < count; i++) {
    HIP_CHECK(
        hipMemcpy(hostPtr[i].data(), dstPtr[i], size, hipMemcpyDeviceToHost));
    for (int j = 0; j < size; j++) {
      INFO("Array FAILURE at Index: " << i << " " << j
                                      << "\nval : " << hostPtr[i][j]);
      REQUIRE(hostPtr[i][j] == 'b');
    }
  }

  // Clean up
  for (int i = 0; i < count; i++) {
    HIP_CHECK(hipFree(dstPtr[i]));
  }
  HIP_CHECK(hipStreamDestroy(stream));
}
/**
 * Test Description
 * ------------------------
 * - Test case to verify the 1D batch memory copy.
 * 1. Create Array of device pointers(Src, Dst).
 * 2. Set the MemcpyBatch params. As of now no support for memcpy Attributes.
 * 3. Perform batch memcpy operation From Device to Host.
 * 4. Validate data on host.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipMemcpyBatchAsync_D2H_BasicFunctional") {
  const size_t count = 2;
  size_t numAttrs = 0;
  const size_t size = 4096 * sizeof(char);
  hipStream_t stream = NULL;
  HIP_CHECK(hipStreamCreate(&stream));

  // Allocate buffers for pointer-ptr copy
  char *hostDstPtr[count];
  void *deviceSrcPtr[count];
  std::vector<std::array<char, size>> hostPtr(count);
  std::array<char, size> arr;
  arr.fill('a');
  size_t sizes[2];
  size_t attrsIdxs[1];
  for (int i = 0; i < count; i++) {
    hostDstPtr[i] = arr.data();
    HIP_CHECK(hipMalloc(&deviceSrcPtr[i], size));
    HIP_CHECK(hipMemset(deviceSrcPtr[i], 'b', size)); // Fill with value
    sizes[i] = size;
  }
  attrsIdxs[0] = 0;
  size_t failIdx;

  HIP_CHECK(hipMemcpyBatchAsync(reinterpret_cast<void **>(hostDstPtr),
                                deviceSrcPtr, sizes, count, nullptr, attrsIdxs,
                                numAttrs, &failIdx, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  // validation
  for (int i = 0; i < count; i++) {
    for (int j = 0; j < size; j++) {
      INFO("Array FAILURE at Index: " << i << " " << j
                                      << "\nval : " << hostDstPtr[i][j]);
      REQUIRE(hostDstPtr[i][j] == 'b');
    }
  }
  // Clean up
  for (int i = 0; i < count; i++) {
    HIP_CHECK(hipFree(deviceSrcPtr[i]));
  }
  HIP_CHECK(hipStreamDestroy(stream));
}
/**
 * Test Description
 * ------------------------
 * - Test case to verify the 1D batch memory copy.
 * 1. Create Array of device pointers(Src, Dst).
 * 2. Set the MemcpyBatch params. As of now no support for memcpy Attributes.
 * 3. Perform batch memcpy operation From Host to Host.
 * 4. Validate data on host.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipMemcpyBatchAsync_H2H_BasicFunctional") {
  const size_t count = 2;
  size_t numAttrs = 0;
  const size_t size = 4096 * sizeof(char);
  hipStream_t stream = NULL;
  HIP_CHECK(hipStreamCreate(&stream));

  // Allocate buffers for pointer-ptr copy
  char *hostDstPtr[count], *hostSrcPtr[count];
  std::vector<std::array<char, size>> hostPtr(count);
  std::array<char, size> arr1, arr2;
  arr1.fill('a');
  arr2.fill('b');
  size_t sizes[2];
  size_t attrsIdxs[1];
  for (int i = 0; i < count; i++) {
    hostDstPtr[i] = arr1.data();
    hostSrcPtr[i] = arr2.data();
    sizes[i] = size;
  }
  attrsIdxs[0] = 0;
  size_t failIdx;

  HIP_CHECK(hipMemcpyBatchAsync(reinterpret_cast<void **>(hostDstPtr),
                                reinterpret_cast<void **>(hostSrcPtr), sizes,
                                count, nullptr, attrsIdxs, numAttrs, &failIdx,
                                stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  // validation
  for (int i = 0; i < count; i++) {
    for (int j = 0; j < size; j++) {
      INFO("Array FAILURE at Index: " << i << " " << j
                                      << "\nval : " << hostDstPtr[i][j]);
      REQUIRE(hostDstPtr[i][j] == 'b');
    }
  }
  // Clean up
  HIP_CHECK(hipStreamDestroy(stream));
}
#endif
/**
 * Test Description
 * ------------------------
 * - Test case to verify the negative cases of hipMemcpyBatchAsync.
 * 1. Dst Array as nullptr.
 * 2. Src Array as nullptr.
 * 3. Operations Count as 0.
 * 4. Num of attributes as 0.
 * 5. Sizes Array as nullptr.
 * 6. Attr Array as nullptr.
 * 7. AttrsIdxs Array as nullptr.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipMemcpyBatchAsync_NegativeTsts") {
  const size_t count = 2;
  size_t numAttrs = 0;
  size_t sizes[2];
  size_t attrsIdxs[1];
  const size_t size = 4096 * sizeof(char);
  hipStream_t stream = NULL;
  HIP_CHECK(hipStreamCreate(&stream));
  void *srcPtr[count], *dstPtr[count];
  for (int i = 0; i < count; i++) {
    HIP_CHECK(hipMalloc(&srcPtr[i], size));
    HIP_CHECK(hipMalloc(&dstPtr[i], size));
    sizes[i] = size;
  }

  attrsIdxs[0] = 0;
  size_t failIdx;
  SECTION("Dst Array as nullptr") {
    HIP_CHECK_ERROR(hipMemcpyBatchAsync(nullptr, srcPtr, sizes, count, nullptr,
                                        attrsIdxs, numAttrs, &failIdx, stream),
                    hipErrorInvalidValue);
  }
  SECTION("Src Array as nullptr") {
    HIP_CHECK_ERROR(hipMemcpyBatchAsync(dstPtr, nullptr, sizes, count, nullptr,
                                        attrsIdxs, numAttrs, &failIdx, stream),
                    hipErrorInvalidValue);
  }
  SECTION("Count as zero") {
    HIP_CHECK_ERROR(hipMemcpyBatchAsync(dstPtr, srcPtr, sizes, 0, nullptr,
                                        attrsIdxs, numAttrs, &failIdx, stream),
                    hipErrorInvalidValue);
  }
  SECTION("sizes Array as nullptr") {
    HIP_CHECK_ERROR(hipMemcpyBatchAsync(dstPtr, srcPtr, nullptr, count, nullptr,
                                        attrsIdxs, numAttrs, &failIdx, stream),
                    hipErrorInvalidValue);
  }
#if 0 // Enable these tests when support for memcpy attributes is enabled.
  SECTION("Number of Attributes as zero") {
    HIP_CHECK_ERROR(
        hipMemcpyBatchAsync(dstPtr, srcPtr, sizes, count, attr, attrsIdxs, 0, &failIdx, stream),
        hipErrorInvalidValue);
  }
  SECTION("Attr Array as nullptr") {
    HIP_CHECK_ERROR(hipMemcpyBatchAsync(dstPtr, srcPtr, sizes, count, nullptr, attrsIdxs, numAttrs,
                                        &failIdx, stream),
                    hipErrorInvalidValue);
  }

  SECTION("attrsIdxs Array as nullptr") {
    HIP_CHECK_ERROR(hipMemcpyBatchAsync(dstPtr, srcPtr, sizes, count, attr, nullptr, numAttrs,
                                        &failIdx, stream),
                    hipErrorInvalidValue);
  }
#endif
  // Clean up
  for (int i = 0; i < count; i++) {
    HIP_CHECK(hipFree(srcPtr[i]));
    HIP_CHECK(hipFree(dstPtr[i]));
  }
  HIP_CHECK(hipStreamDestroy(stream));
}
/**
 * End doxygen group MemoryTest.
 * @}
 */
