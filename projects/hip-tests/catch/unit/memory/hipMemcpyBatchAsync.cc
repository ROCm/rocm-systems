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
 * `hipError_t hipMemcpyBatchAsync(void** dsts, void** srcs, size_t* sizes, size_t count,
                               hipMemcpyAttributes* attrs, size_t* attrsIdxs, size_t numAttrs,
                               size_t* failIdx, hipStream_t stream __dparm(0))` -
 * Perform Batch of 1D copies.
 */

#if HT_AMD
TEST_CASE("Unit_hipMemcpyBatchAsync_H2D_BasicFunctional") {
  const size_t count = 2;
  size_t numAttrs = 0;
  const size_t size = 4096 * sizeof(char);
  hipStream_t stream = NULL;
  HIP_CHECK(hipStreamCreate(&stream));

  // Allocate buffers for pointer-ptr copy
  void *hostSrcPtr[count], *dstPtr[count];
  std::vector<std::array<char, size>> hostPtr(count);
  std::array<char, size>arr;
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

  HIP_CHECK(hipMemcpyBatchAsync(dstPtr, hostSrcPtr, sizes, count, nullptr, attrsIdxs, numAttrs, &failIdx,
                                stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  // validation
  for (int i = 0; i < count; i++) {
    HIP_CHECK(hipMemcpy(hostPtr[i].data(), dstPtr[i], size, hipMemcpyDeviceToHost));
    for (int j = 0; j < size; j++) {
      INFO("Array FAILURE at Index: "<< i << " "<< j << "\nval : " << hostPtr[i][j]);
      REQUIRE(hostPtr[i][j] == 'b');
    }
  }

  // Clean up
  for (int i = 0; i < count; i++) {
    HIP_CHECK(hipFree(dstPtr[i]));
  }
  HIP_CHECK(hipStreamDestroy(stream));
}
TEST_CASE("Unit_hipMemcpyBatchAsync_D2H_BasicFunctional") {
  const size_t count = 2;
  size_t numAttrs = 0;
  const size_t size = 4096 * sizeof(char);
  hipStream_t stream = NULL;
  HIP_CHECK(hipStreamCreate(&stream));

  // Allocate buffers for pointer-ptr copy
  char *hostSrcPtr[count];void *dstPtr[count];
  std::vector<std::array<char, size>> hostPtr(count);
  std::array<char, size>arr;
  arr.fill('a');
  size_t sizes[2];
  size_t attrsIdxs[1];
  for (int i = 0; i < count; i++) {
    hostSrcPtr[i] = arr.data();
    HIP_CHECK(hipMalloc(&dstPtr[i], size));
        HIP_CHECK(hipMemset(dstPtr[i], 'b', size));  // Fill with value
    sizes[i] = size;
  }

  attrsIdxs[0] = 0;
  size_t failIdx;

  HIP_CHECK(hipMemcpyBatchAsync(reinterpret_cast<void**>(hostSrcPtr), dstPtr, sizes, count, nullptr, attrsIdxs, numAttrs, &failIdx,
                                stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  // validation
  for (int i = 0; i < count; i++) {
     //std::array<char, size>chk;
     //chk = hostSrcPtr[i];
    //HIP_CHECK(hipMemcpy(hostPtr[i].data(), dstPtr[i], size, hipMemcpyDeviceToHost));
    for (int j = 0; j < size; j++) {
      INFO("Array FAILURE at Index: "<< i << " "<< j << "\nval : " << hostSrcPtr[i][j]);
      REQUIRE(hostSrcPtr[i][j] == 'b');
    }
  }

  // Clean up
  for (int i = 0; i < count; i++) {
    HIP_CHECK(hipFree(dstPtr[i]));
  }
  HIP_CHECK(hipStreamDestroy(stream));

}
TEST_CASE("Unit_hipMemcpyBatchAsync_H2H_BasicFunctional") {
  const size_t count = 2;
  size_t numAttrs = 0;
  const size_t size = 4096 * sizeof(char);
  hipStream_t stream = NULL;
  HIP_CHECK(hipStreamCreate(&stream));

  // Allocate buffers for pointer-ptr copy
  char *hostSrcPtr[count], *dstPtr[count];
  std::vector<std::array<char, size>> hostPtr(count);
  std::array<char, size>arr1, arr2;
  arr1.fill('a');
  arr2.fill('b');
  size_t sizes[2];
  size_t attrsIdxs[1];
  for (int i = 0; i < count; i++) {
    hostSrcPtr[i] = arr1.data();
        dstPtr[i] = arr2.data();
    sizes[i] = size;
  }

  attrsIdxs[0] = 0;
  size_t failIdx;

  HIP_CHECK(hipMemcpyBatchAsync(reinterpret_cast<void**>(hostSrcPtr), reinterpret_cast<void**>(dstPtr), sizes, count, nullptr, attrsIdxs, numAttrs, &failIdx,
                                stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  // validation
  for (int i = 0; i < count; i++) {
     //std::array<char, size>chk;
     //chk = hostSrcPtr[i];
    //HIP_CHECK(hipMemcpy(hostPtr[i].data(), dstPtr[i], size, hipMemcpyDeviceToHost));
    for (int j = 0; j < size; j++) {
      INFO("Array FAILURE at Index: "<< i << " "<< j << "\nval : " << hostSrcPtr[i][j]);
      REQUIRE(hostSrcPtr[i][j] == 'b');
    }
  }

  // Clean up
  /*for (int i = 0; i < count; i++) {
    HIP_CHECK(hipFree(dstPtr[i]));
  }*/
  HIP_CHECK(hipStreamDestroy(stream));
}
#endif

