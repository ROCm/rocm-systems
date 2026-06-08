/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
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
HIP_TEMPLATE_TEST_CASE(Unit_hipMemcpyBatchAsync_D2D_Functional, char, int,
                   float) {
  const size_t count = 2;
  size_t numAttrs = 0;
  const size_t arrSize = 4096;
  const size_t size = 4096 * sizeof(TestType);
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  constexpr auto kfloatval1 = 2.25f;
  constexpr auto kfloatval2 = 0.25f;
  const TestType val1 = std::is_floating_point_v<TestType> ? kfloatval1
                        : std::is_integral_v<TestType>     ? 10
                                                           : 'a';
  const TestType val2 = std::is_floating_point_v<TestType> ? kfloatval2
                        : std::is_integral_v<TestType>     ? 4
                                                           : 'b';

  // Allocate buffers for pointer-ptr copy
  void *srcPtr[count], *dstPtr[count];
  std::vector<std::vector<TestType>> hostPtr1(
      count, std::vector<TestType>(arrSize, val1));
  std::vector<std::vector<TestType>> hostPtr2(
      count, std::vector<TestType>(arrSize, val2));
  size_t sizes[2];
  size_t attrsIdxs[1];
  for (int i = 0; i < count; i++) {
    HIP_CHECK(hipMalloc(&srcPtr[i], size));
    HIP_CHECK(hipMalloc(&dstPtr[i], size));
    HIP_CHECK(
        hipMemcpy(srcPtr[i], hostPtr2[i].data(), size, hipMemcpyHostToDevice));
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
        hipMemcpy(hostPtr1[i].data(), dstPtr[i], size, hipMemcpyDeviceToHost));
    for (int j = 0; j < arrSize; j++) {
      INFO("Array FAILURE at Index: " << i << " " << j
                                      << "\nval : " << hostPtr1[i][j]);
      REQUIRE(hostPtr1[i][j] == val2);
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
HIP_TEMPLATE_TEST_CASE(Unit_hipMemcpyBatchAsync_H2D_Functional, char, int,
                   float) {
  const size_t count = 2;
  size_t numAttrs = 0;
  const size_t arrSize = 4096;
  const size_t size = 4096 * sizeof(TestType);
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  // Allocate buffers for pointer-ptr copy
  void *hostSrcPtr[count], *dstPtr[count];
  constexpr auto kfloatval1 = 2.25f;
  constexpr auto kfloatval2 = 0.25f;
  const TestType val1 = std::is_floating_point_v<TestType> ? kfloatval1
                        : std::is_integral_v<TestType>     ? 10
                                                           : 'a';
  const TestType val2 = std::is_floating_point_v<TestType> ? kfloatval2
                        : std::is_integral_v<TestType>     ? 4
                                                           : 'b';
  std::vector<std::vector<TestType>> hostPtr(
      count, std::vector<TestType>(arrSize, val2));
  std::array<TestType, arrSize> arr;
  arr.fill(val1);
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
    for (int j = 0; j < arrSize; j++) {
      INFO("Array FAILURE at Index: " << i << " " << j
                                      << "\nval : " << hostPtr[i][j]);
      REQUIRE(hostPtr[i][j] == val1);
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
HIP_TEMPLATE_TEST_CASE(Unit_hipMemcpyBatchAsync_D2H_Functional, char, int,
                   float) {
  const size_t count = 2;
  size_t numAttrs = 0;
  const size_t arrSize = 4096;
  const size_t size = 4096 * sizeof(TestType);
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  constexpr auto kfloatval1 = 2.25f;
  constexpr auto kfloatval2 = 0.25f;
  const TestType val1 = std::is_floating_point_v<TestType> ? kfloatval1
                        : std::is_integral_v<TestType>     ? 10
                                                           : 'a';
  const TestType val2 = std::is_floating_point_v<TestType> ? kfloatval2
                        : std::is_integral_v<TestType>     ? 4
                                                           : 'b';
  // Allocate buffers for pointer-ptr copy
  TestType *hostDstPtr[count];
  void *deviceSrcPtr[count];
  std::vector<std::vector<TestType>> hostPtr(
      count, std::vector<TestType>(arrSize, val1));
  std::array<TestType, arrSize> arr;
  arr.fill(val2);
  size_t sizes[2];
  size_t attrsIdxs[1];
  for (int i = 0; i < count; i++) {
    hostDstPtr[i] = arr.data();
    HIP_CHECK(hipMalloc(&deviceSrcPtr[i], size));
    HIP_CHECK(hipMemcpy(deviceSrcPtr[i], hostPtr[i].data(), size,
                        hipMemcpyHostToDevice));
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
    for (int j = 0; j < arrSize; j++) {
      INFO("Array FAILURE at Index: " << i << " " << j
                                      << "\nval : " << hostDstPtr[i][j]);
      REQUIRE(hostDstPtr[i][j] == val1);
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
HIP_TEMPLATE_TEST_CASE(Unit_hipMemcpyBatchAsync_H2H_Functional, char, int,
                   float) {
  const size_t count = 2;
  size_t numAttrs = 0;
  const size_t arrSize = 4096;
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  constexpr auto kfloatval1 = 2.25;
  const TestType val1 = std::is_floating_point_v<TestType> ? kfloatval1
                        : std::is_integral_v<TestType>     ? 10
                                                           : 'a';
  constexpr auto kfloatval2 = 0.25f;
  const TestType val2 = std::is_floating_point_v<TestType> ? kfloatval2
                        : std::is_integral_v<TestType>     ? 4
                                                           : 'b';

  // Allocate buffers for pointer-ptr copy
  TestType *hostDstPtr[count], *hostSrcPtr[count];
  std::array<TestType, arrSize> arr1, arr2;
  arr1.fill(val1);
  arr2.fill(val2);
  size_t sizes[2];
  size_t attrsIdxs[1];
  for (int i = 0; i < count; i++) {
    hostDstPtr[i] = arr1.data();
    hostSrcPtr[i] = arr2.data();
    sizes[i] = arrSize * sizeof(TestType);
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
    for (int j = 0; j < arrSize; j++) {
      INFO("Array FAILURE at Index: " << i << " " << j
                                      << "\nval : " << hostDstPtr[i][j]);
      REQUIRE(hostDstPtr[i][j] == val2);
    }
  }
  // Clean up
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - Verify hipMemcpyBatchAsync with hipMemcpyFlagExtOpSwap exchanges the
 *   contents of two device buffers.
 * 1. Allocate two device buffers and fill with distinct values.
 * 2. Issue hipMemcpyBatchAsync with swap attribute.
 * 3. Read back both buffers and verify values are exchanged.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_swap_cp) {
  constexpr size_t kNumElements = 4096;
  constexpr size_t kSizeBytes = kNumElements * sizeof(int);
  constexpr int kValA = 42;
  constexpr int kValB = 99;

  // Allocate device buffers
  void* d_a = nullptr;
  void* d_b = nullptr;
  HIP_CHECK(hipMalloc(&d_a, kSizeBytes));
  HIP_CHECK(hipMalloc(&d_b, kSizeBytes));

  // Fill with distinct values
  std::vector<int> hostA(kNumElements, kValA);
  std::vector<int> hostB(kNumElements, kValB);
  HIP_CHECK(hipMemcpy(d_a, hostA.data(), kSizeBytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_b, hostB.data(), kSizeBytes, hipMemcpyHostToDevice));

  // Set up batch swap: dst=d_a, src=d_b with swap flag
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  void* dsts[] = {d_a};
  void* srcs[] = {d_b};
  size_t sizes[] = {kSizeBytes};
  size_t attrsIdxs[] = {0};

  hipMemcpyAttributes attr{};
  attr.flags = hipMemcpyFlagExtOpSwap;
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;

  size_t failIdx = 0;
  hipError_t err = hipMemcpyBatchAsync(dsts, srcs, sizes, 1, &attr, attrsIdxs, 1,
                                       &failIdx, stream);
  if (err == hipErrorNotSupported) {
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_SKIP_TEST(HipTest::SkipReason::kSdmaSwapUnsupported);
  }
  HIP_CHECK(err);
  HIP_CHECK(hipStreamSynchronize(stream));

  // Read back and verify swap
  std::vector<int> resultA(kNumElements);
  std::vector<int> resultB(kNumElements);
  HIP_CHECK(hipMemcpy(resultA.data(), d_a, kSizeBytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(resultB.data(), d_b, kSizeBytes, hipMemcpyDeviceToHost));

  for (size_t i = 0; i < kNumElements; i++) {
    REQUIRE(resultA[i] == kValB);
    REQUIRE(resultB[i] == kValA);
  }

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - Verify asymmetric swap: size_a > size_b.
 *   Per HW spec (OSSIP-SDMA72-16), asymmetric swap decomposes into:
 *   1. Swap min(size_a, size_b) bytes between A and B
 *   2. Copy remaining (size_a - size_b) bytes from A to B
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_swap_asymmetric) {
  constexpr size_t kSizeA = 8192;  // 8 KB
  constexpr size_t kSizeB = 4096;  // 4 KB (smaller side)
  constexpr int kValA = 42;
  constexpr int kValB = 99;

  // Allocate two device buffers, both large enough for kSizeA
  void* d_a = nullptr;
  void* d_b = nullptr;
  HIP_CHECK(hipMalloc(&d_a, kSizeA));
  HIP_CHECK(hipMalloc(&d_b, kSizeA));

  // Fill A entirely with kValA, B entirely with kValB
  std::vector<int> hostA(kSizeA / sizeof(int), kValA);
  std::vector<int> hostB(kSizeA / sizeof(int), kValB);
  HIP_CHECK(hipMemcpy(d_a, hostA.data(), kSizeA, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_b, hostB.data(), kSizeA, hipMemcpyHostToDevice));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  void* dsts[] = {d_a};
  void* srcs[] = {d_b};
  size_t sizesA[] = {kSizeA};
  size_t sizesB[] = {kSizeB};
  size_t attrsIdxs[] = {0};

  hipMemcpyAttributes attr{};
  attr.flags = hipMemcpyFlagExtOpSwap;
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;

  size_t failIdx = 0;
  hipError_t err = hipExtMemcpyBatchAsync(dsts, srcs, sizesA, sizesB,
                                          nullptr, nullptr, nullptr,
                                          1, &attr, attrsIdxs, 1,
                                          &failIdx, stream);
  if (err == hipErrorNotSupported) {
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_SKIP_TEST(HipTest::SkipReason::kSdmaSwapUnsupported);
  }
  HIP_CHECK(err);
  HIP_CHECK(hipStreamSynchronize(stream));

  // Read back both buffers
  std::vector<int> resultA(kSizeA / sizeof(int));
  std::vector<int> resultB(kSizeA / sizeof(int));
  HIP_CHECK(hipMemcpy(resultA.data(), d_a, kSizeA, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(resultB.data(), d_b, kSizeA, hipMemcpyDeviceToHost));

  size_t swapElems = kSizeB / sizeof(int);  // Elements in the swapped region
  size_t totalElems = kSizeA / sizeof(int);

  // A[0..swapElems-1] = old B values (swapped)
  for (size_t i = 0; i < swapElems; i++) {
    REQUIRE(resultA[i] == kValB);
  }
  // A[swapElems..end] = unchanged (still kValA)
  for (size_t i = swapElems; i < totalElems; i++) {
    REQUIRE(resultA[i] == kValA);
  }

  // B[0..swapElems-1] = old A values (swapped)
  for (size_t i = 0; i < swapElems; i++) {
    REQUIRE(resultB[i] == kValA);
  }
  // B[swapElems..end] = old A tail values (copied from A)
  for (size_t i = swapElems; i < totalElems; i++) {
    REQUIRE(resultB[i] == kValA);
  }

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - Verify asymmetric swap with multiple attribute ranges.
 *   Two swap pairs under separate attributes with different swapSizesA/B,
 *   testing that rangeIdx (idx - attrsIdxs[attrIdx]) indexes correctly.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_swap_asymmetric_multi_attr) {
  // Pair 0: attr[0], swap 8KB A with 4KB B
  // Pair 1: attr[1], swap 4KB A with 2KB B
  constexpr int kValA0 = 10, kValB0 = 20;
  constexpr int kValA1 = 30, kValB1 = 40;
  constexpr size_t kBufSize = 8192;  // all allocations are 8KB

  void *dA0, *dB0, *dA1, *dB1;
  HIP_CHECK(hipMalloc(&dA0, kBufSize));
  HIP_CHECK(hipMalloc(&dB0, kBufSize));
  HIP_CHECK(hipMalloc(&dA1, kBufSize));
  HIP_CHECK(hipMalloc(&dB1, kBufSize));

  // Fill each buffer with its value
  std::vector<int> hA0(kBufSize / sizeof(int), kValA0);
  std::vector<int> hB0(kBufSize / sizeof(int), kValB0);
  std::vector<int> hA1(kBufSize / sizeof(int), kValA1);
  std::vector<int> hB1(kBufSize / sizeof(int), kValB1);
  HIP_CHECK(hipMemcpy(dA0, hA0.data(), kBufSize, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dB0, hB0.data(), kBufSize, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dA1, hA1.data(), kBufSize, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dB1, hB1.data(), kBufSize, hipMemcpyHostToDevice));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  void* dsts[] = {dA0, dA1};
  void* srcs[] = {dB0, dB1};
  size_t sizesA[] = {8192, 4096};
  size_t sizesB[] = {4096, 2048};

  hipMemcpyAttributes attrs[2] = {};
  attrs[0].flags = hipMemcpyFlagExtOpSwap;
  attrs[0].srcAccessOrder = hipMemcpySrcAccessOrderStream;
  attrs[1].flags = hipMemcpyFlagExtOpSwap;
  attrs[1].srcAccessOrder = hipMemcpySrcAccessOrderStream;

  size_t attrsIdxs[] = {0, 1};
  size_t failIdx = 0;

  hipError_t err = hipExtMemcpyBatchAsync(dsts, srcs, sizesA, sizesB,
                                          nullptr, nullptr, nullptr,
                                          2, attrs, attrsIdxs, 2,
                                          &failIdx, stream);
  if (err == hipErrorNotSupported) {
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(dA0)); HIP_CHECK(hipFree(dB0));
    HIP_CHECK(hipFree(dA1)); HIP_CHECK(hipFree(dB1));
    HIP_SKIP_TEST(HipTest::SkipReason::kSdmaSwapUnsupported);
  }
  HIP_CHECK(err);
  HIP_CHECK(hipStreamSynchronize(stream));

  // Read back all buffers
  std::vector<int> rA0(kBufSize / sizeof(int));
  std::vector<int> rB0(kBufSize / sizeof(int));
  std::vector<int> rA1(kBufSize / sizeof(int));
  std::vector<int> rB1(kBufSize / sizeof(int));
  HIP_CHECK(hipMemcpy(rA0.data(), dA0, kBufSize, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(rB0.data(), dB0, kBufSize, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(rA1.data(), dA1, kBufSize, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(rB1.data(), dB1, kBufSize, hipMemcpyDeviceToHost));

  // Pair 0: swap 4KB (1024 ints), copy 4KB tail from A0 to B0
  size_t swap0 = 4096 / sizeof(int);
  size_t total = kBufSize / sizeof(int);
  for (size_t i = 0; i < swap0; i++) {
    REQUIRE(rA0[i] == kValB0);   // swapped from B0
    REQUIRE(rB0[i] == kValA0);   // swapped from A0
  }
  for (size_t i = swap0; i < total; i++) {
    REQUIRE(rA0[i] == kValA0);   // unchanged
    REQUIRE(rB0[i] == kValA0);   // tail copied from A0
  }

  // Pair 1: swap 2KB (512 ints), copy 2KB tail from A1 to B1
  size_t swap1 = 2048 / sizeof(int);
  for (size_t i = 0; i < swap1; i++) {
    REQUIRE(rA1[i] == kValB1);   // swapped from B1
    REQUIRE(rB1[i] == kValA1);   // swapped from A1
  }
  for (size_t i = swap1; i < 4096 / sizeof(int); i++) {
    REQUIRE(rA1[i] == kValA1);   // unchanged
    REQUIRE(rB1[i] == kValA1);   // tail copied from A1
  }
  // Beyond swapSizesA: untouched by the swap operation
  for (size_t i = 4096 / sizeof(int); i < total; i++) {
    REQUIRE(rA1[i] == kValA1);   // unchanged
    REQUIRE(rB1[i] == kValB1);   // unchanged
  }

  HIP_CHECK(hipFree(dA0)); HIP_CHECK(hipFree(dB0));
  HIP_CHECK(hipFree(dA1)); HIP_CHECK(hipFree(dB1));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - Verify that hipExtMemcpyBatchAsync with sizesB = nullptr performs
 *   a full symmetric swap (same as hipMemcpyBatchAsync).
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_swap_asymmetric_fallback) {
  constexpr size_t kSizeA = 8192;
  constexpr size_t kSizeB = 4096;  // Set but should be ignored
  constexpr int kValA = 42;
  constexpr int kValB = 99;

  void* d_a = nullptr;
  void* d_b = nullptr;
  HIP_CHECK(hipMalloc(&d_a, kSizeA));
  HIP_CHECK(hipMalloc(&d_b, kSizeA));

  std::vector<int> hostA(kSizeA / sizeof(int), kValA);
  std::vector<int> hostB(kSizeA / sizeof(int), kValB);
  HIP_CHECK(hipMemcpy(d_a, hostA.data(), kSizeA, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_b, hostB.data(), kSizeA, hipMemcpyHostToDevice));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  void* dsts[] = {d_a};
  void* srcs[] = {d_b};
  size_t sizesA[] = {kSizeA};
  size_t attrsIdxs[] = {0};

  hipMemcpyAttributes attr{};
  // Use hipExtMemcpyBatchAsync with sizesB = nullptr.
  // Full symmetric swap expected even though sizesA > sizesB would apply.
  attr.flags = hipMemcpyFlagExtOpSwap;
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;

  size_t failIdx = 0;
  hipError_t err = hipExtMemcpyBatchAsync(dsts, srcs, sizesA, nullptr,
                                          nullptr, nullptr, nullptr,
                                          1, &attr, attrsIdxs, 1,
                                          &failIdx, stream);
  if (err == hipErrorNotSupported) {
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_SKIP_TEST(HipTest::SkipReason::kSdmaSwapUnsupported);
  }
  HIP_CHECK(err);
  HIP_CHECK(hipStreamSynchronize(stream));

  std::vector<int> resultA(kSizeA / sizeof(int));
  std::vector<int> resultB(kSizeA / sizeof(int));
  HIP_CHECK(hipMemcpy(resultA.data(), d_a, kSizeA, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(resultB.data(), d_b, kSizeA, hipMemcpyDeviceToHost));

  size_t totalElems = kSizeA / sizeof(int);

  // Full symmetric swap: ALL of A should be kValB, ALL of B should be kValA
  for (size_t i = 0; i < totalElems; i++) {
    REQUIRE(resultA[i] == kValB);
    REQUIRE(resultB[i] == kValA);
  }

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - Verify that per-entry ops can specify swap operation instead
 *   of using hipMemcpyAttributes.flags. The attrs have no swap flag set;
 *   only ops specifies the swap.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipExtMemcpyBatchAsync_entryFlags_swap) {
  constexpr size_t kNumElements = 4096;
  constexpr size_t kSizeBytes = kNumElements * sizeof(int);
  constexpr int kValA = 42;
  constexpr int kValB = 99;

  void* d_a = nullptr;
  void* d_b = nullptr;
  HIP_CHECK(hipMalloc(&d_a, kSizeBytes));
  HIP_CHECK(hipMalloc(&d_b, kSizeBytes));

  std::vector<int> hostA(kNumElements, kValA);
  std::vector<int> hostB(kNumElements, kValB);
  HIP_CHECK(hipMemcpy(d_a, hostA.data(), kSizeBytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_b, hostB.data(), kSizeBytes, hipMemcpyHostToDevice));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  void* dsts[] = {d_a};
  void* srcs[] = {d_b};
  size_t sizesA[] = {kSizeBytes};

  // Attrs have NO swap flag — swap is specified only via ops
  hipMemcpyAttributes attr{};
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;
  size_t attrsIdxs[] = {0};

  hipExtMemcpyOp ops[] = {hipExtMemcpyOpSwap};

  size_t failIdx = 0;
  hipError_t err = hipExtMemcpyBatchAsync(dsts, srcs, sizesA, nullptr,
                                          nullptr, nullptr, ops,
                                          1, &attr, attrsIdxs, 1,
                                          &failIdx, stream);
  if (err == hipErrorNotSupported) {
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_SKIP_TEST(HipTest::SkipReason::kSdmaSwapUnsupported);
  }
  HIP_CHECK(err);
  HIP_CHECK(hipStreamSynchronize(stream));

  std::vector<int> resultA(kNumElements);
  std::vector<int> resultB(kNumElements);
  HIP_CHECK(hipMemcpy(resultA.data(), d_a, kSizeBytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(resultB.data(), d_b, kSizeBytes, hipMemcpyDeviceToHost));

  for (size_t i = 0; i < kNumElements; i++) {
    REQUIRE(resultA[i] == kValB);
    REQUIRE(resultB[i] == kValA);
  }

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
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
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_NegativeTsts) {
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
