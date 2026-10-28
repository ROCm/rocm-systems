/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <resource_guards.hh>

/**
 * @addtogroup hipExtMemcpyBatchAsync hipExtMemcpyBatchAsync
 * @{
 * @ingroup MemoryTest
 * AMD batch-copy extensions: hipExtMemcpyBatchAsync and the hipMemcpyFlagExtOp* flags.
 */

namespace {

constexpr size_t kSmallCopySize = 4 * 1024;
constexpr int kPatternValue = 0x42;

void RequireDeviceFilled(const void* ptr, size_t size, int expected) {
  std::vector<int> result(size / sizeof(int));
  HIP_CHECK(hipMemcpy(result.data(), ptr, size, hipMemcpyDeviceToHost));
  REQUIRE(std::all_of(result.begin(), result.end(), [=](int v) { return v == expected; }));
}

}  // namespace

/**
 * Test Description
 * ------------------------
 * - Verify asymmetric swap: size_a > size_b, with A = srcs[i] (sizes[i]) and
 *   B = dsts[i] (sizesDst[i]). Asymmetric swap decomposes on the CLR side into:
 *   1. Swap min(size_a, size_b) bytes between A and B
 *   2. Copy remaining (size_a - size_b) bytes from A to B
 * Test source
 * ------------------------
 * - catch/unit/memory/hipExtMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipExtMemcpyBatchAsync_Swap_Asymmetric) {
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

  void* dsts[] = {d_b};
  void* srcs[] = {d_a};
  size_t sizes[] = {kSizeA};
  size_t sizesDst[] = {kSizeB};
  size_t attrsIdxs[] = {0};

  hipExtMemcpyAttributes attr{};
  attr.flags = hipMemcpyFlagExtOpSwap;
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;

  hipError_t err = hipExtMemcpyBatchAsync(dsts, srcs, sizes, sizesDst,
                                          nullptr, nullptr,
                                          1, &attr, attrsIdxs, 1,
                                          stream);
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
 *   Two swap entries under separate attributes with different sizes/sizesDst;
 *   each entry is split at its own sizesDst.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipExtMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipExtMemcpyBatchAsync_Swap_Asymmetric_MultiAttr) {
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

  void* dsts[] = {dB0, dB1};
  void* srcs[] = {dA0, dA1};
  size_t sizes[] = {8192, 4096};
  size_t sizesDst[] = {4096, 2048};

  hipExtMemcpyAttributes attrs[2] = {};
  attrs[0].flags = hipMemcpyFlagExtOpSwap;
  attrs[0].srcAccessOrder = hipMemcpySrcAccessOrderStream;
  attrs[1].flags = hipMemcpyFlagExtOpSwap;
  attrs[1].srcAccessOrder = hipMemcpySrcAccessOrderStream;

  size_t attrsIdxs[] = {0, 1};

  hipError_t err = hipExtMemcpyBatchAsync(dsts, srcs, sizes, sizesDst,
                                          nullptr, nullptr,
                                          2, attrs, attrsIdxs, 2,
                                          stream);
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
  // Beyond sizes[1]: untouched
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
 * - Verify that hipExtMemcpyBatchAsync with sizesDst = nullptr performs
 *   a full symmetric swap (same as hipMemcpyBatchAsync).
 * Test source
 * ------------------------
 * - catch/unit/memory/hipExtMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipExtMemcpyBatchAsync_Swap_NullSizesDst) {
  constexpr size_t kSizeA = 8192;
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
  size_t sizes[] = {kSizeA};
  size_t attrsIdxs[] = {0};

  hipExtMemcpyAttributes attr{};
  // sizesDst = nullptr: full symmetric swap.
  attr.flags = hipMemcpyFlagExtOpSwap;
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;

  hipError_t err = hipExtMemcpyBatchAsync(dsts, srcs, sizes, nullptr,
                                          nullptr, nullptr,
                                          1, &attr, attrsIdxs, 1,
                                          stream);
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
 * - Verifies hipExtMemcpyBatchAsync validates hipExtMemcpyAttributes:
 *   1. Unknown flag bits in flags return hipErrorInvalidValue.
 *   2. Swap combined with an indirect flag returns hipErrorInvalidValue.
 *   3. A non-zero reserved field returns hipErrorInvalidValue.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipExtMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipExtMemcpyBatchAsync_AttrValidation_Negative) {
  constexpr size_t kNumElements = 1024;
  constexpr size_t kSizeBytes = kNumElements * sizeof(int);

  void* d_a = nullptr;
  void* d_b = nullptr;
  HIP_CHECK(hipMalloc(&d_a, kSizeBytes));
  HIP_CHECK(hipMalloc(&d_b, kSizeBytes));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  void* dsts[] = {d_a};
  void* srcs[] = {d_b};
  size_t sizes[] = {kSizeBytes};
  size_t attrsIdxs[] = {0};

  hipExtMemcpyAttributes attr{};
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;

  SECTION("Unknown flag bit") {
    // 0x8 is outside the valid hipMemcpyFlags bit set.
    attr.flags = 0x8;
    hipError_t err = hipExtMemcpyBatchAsync(dsts, srcs, sizes, nullptr, nullptr, nullptr,
                                            1, &attr, attrsIdxs, 1, stream);
    REQUIRE(err == hipErrorInvalidValue);
  }

  SECTION("Swap combined with indirect") {
    attr.flags = hipMemcpyFlagExtOpSwap | hipMemcpyFlagExtOpIndirectSrc;
    hipError_t err = hipExtMemcpyBatchAsync(dsts, srcs, sizes, nullptr, nullptr, nullptr,
                                            1, &attr, attrsIdxs, 1, stream);
    REQUIRE(err == hipErrorInvalidValue);
  }

  SECTION("Non-zero reserved field") {
    attr.reserved[0] = 1;
    hipError_t err = hipExtMemcpyBatchAsync(dsts, srcs, sizes, nullptr, nullptr, nullptr,
                                            1, &attr, attrsIdxs, 1, stream);
    REQUIRE(err == hipErrorInvalidValue);
  }

  SECTION("PreferCE combined with PreferComputeEngine") {
    attr.flags = hipMemcpyFlagExtPreferCE | hipMemcpyFlagExtPreferComputeEngine;
    hipError_t err = hipExtMemcpyBatchAsync(dsts, srcs, sizes, nullptr, nullptr, nullptr,
                                            1, &attr, attrsIdxs, 1, stream);
    REQUIRE(err == hipErrorInvalidValue);
  }

  SECTION("PreferComputeEngine combined with swap") {
    attr.flags = hipMemcpyFlagExtPreferComputeEngine | hipMemcpyFlagExtOpSwap;
    hipError_t err = hipExtMemcpyBatchAsync(dsts, srcs, sizes, nullptr, nullptr, nullptr,
                                            1, &attr, attrsIdxs, 1, stream);
    REQUIRE(err == hipErrorInvalidValue);
  }

  SECTION("PreferComputeEngine combined with indirect") {
    attr.flags = hipMemcpyFlagExtPreferComputeEngine | hipMemcpyFlagExtOpIndirectSrc;
    hipError_t err = hipExtMemcpyBatchAsync(dsts, srcs, sizes, nullptr, nullptr, nullptr,
                                            1, &attr, attrsIdxs, 1, stream);
    REQUIRE(err == hipErrorInvalidValue);
  }

  SECTION("PreferLinear combined with PreferBroadcast") {
    attr.flags = hipMemcpyFlagExtPreferLinear | hipMemcpyFlagExtPreferBroadcast;
    hipError_t err = hipExtMemcpyBatchAsync(dsts, srcs, sizes, nullptr, nullptr, nullptr,
                                            1, &attr, attrsIdxs, 1, stream);
    REQUIRE(err == hipErrorInvalidValue);
  }

  SECTION("PreferBroadcast combined with swap") {
    attr.flags = hipMemcpyFlagExtPreferBroadcast | hipMemcpyFlagExtOpSwap;
    hipError_t err = hipExtMemcpyBatchAsync(dsts, srcs, sizes, nullptr, nullptr, nullptr,
                                            1, &attr, attrsIdxs, 1, stream);
    REQUIRE(err == hipErrorInvalidValue);
  }

  SECTION("PreferBroadcast combined with indirect") {
    attr.flags = hipMemcpyFlagExtPreferBroadcast | hipMemcpyFlagExtOpIndirectSrc;
    hipError_t err = hipExtMemcpyBatchAsync(dsts, srcs, sizes, nullptr, nullptr, nullptr,
                                            1, &attr, attrsIdxs, 1, stream);
    REQUIRE(err == hipErrorInvalidValue);
  }

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - hipMemcpyBatchAsync ignores flag bits outside its original set (unknown bits and the
 *   hipExtMemcpyBatchAsync-only hints), so the copy succeeds and the bytes are correct.
 *   The same unknown bit is rejected by hipExtMemcpyBatchAsync.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipExtMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipMemcpyBatchAsync_LegacyFlags_Ignored) {
  constexpr size_t kNumElements = 1024;
  constexpr size_t kSizeBytes = kNumElements * sizeof(int);
  constexpr int kVal = 37;
  const unsigned int flags = GENERATE(
      0x2u, 0x80000000u, static_cast<unsigned int>(hipMemcpyFlagExtPreferComputeEngine),
      static_cast<unsigned int>(hipMemcpyFlagExtPreferCE | hipMemcpyFlagExtPreferComputeEngine),
      static_cast<unsigned int>(hipMemcpyFlagExtPreferLinear | hipMemcpyFlagExtPreferBroadcast));

  void* d_src = nullptr;
  void* d_dst = nullptr;
  HIP_CHECK(hipMalloc(&d_src, kSizeBytes));
  HIP_CHECK(hipMalloc(&d_dst, kSizeBytes));
  std::vector<int> host(kNumElements, kVal);
  HIP_CHECK(hipMemcpy(d_src, host.data(), kSizeBytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_dst, 0, kSizeBytes));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  void* dsts[] = {d_dst};
  void* srcs[] = {d_src};
  size_t sizes[] = {kSizeBytes};
  size_t attrsIdxs[] = {0};
  size_t failIdx = 0;

  hipMemcpyAttributes attr{hipMemcpySrcAccessOrderStream, {}, {}, flags};
  HIP_CHECK(hipMemcpyBatchAsync(dsts, srcs, sizes, 1, &attr, attrsIdxs, 1, &failIdx, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  REQUIRE(failIdx == SIZE_MAX);

  std::vector<int> out(kNumElements);
  HIP_CHECK(hipMemcpy(out.data(), d_dst, kSizeBytes, hipMemcpyDeviceToHost));
  for (size_t i = 0; i < kNumElements; ++i) {
    REQUIRE(out[i] == kVal);
  }

  hipExtMemcpyAttributes extAttr{};
  extAttr.srcAccessOrder = hipMemcpySrcAccessOrderStream;
  extAttr.flags = 0x2;
  HIP_CHECK_ERROR(hipExtMemcpyBatchAsync(dsts, srcs, sizes, nullptr, nullptr, nullptr, 1,
                                         &extAttr, attrsIdxs, 1, stream),
                  hipErrorInvalidValue);

  HIP_CHECK(hipFree(d_src));
  HIP_CHECK(hipFree(d_dst));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - hipMemcpyFlagExtPreferComputeEngine routes a linear device-to-device batch onto the shader
 *   copy path (compute engine) instead of SDMA. The copied bytes must be correct.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipExtMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipExtMemcpyBatchAsync_PreferComputeEngine_D2D) {
  constexpr size_t kNumElements = 4096;
  constexpr size_t kSizeBytes = kNumElements * sizeof(int);
  constexpr int kValA = 21;
  constexpr int kValB = 73;

  void* d_src_a = nullptr;
  void* d_src_b = nullptr;
  void* d_dst_a = nullptr;
  void* d_dst_b = nullptr;
  HIP_CHECK(hipMalloc(&d_src_a, kSizeBytes));
  HIP_CHECK(hipMalloc(&d_src_b, kSizeBytes));
  HIP_CHECK(hipMalloc(&d_dst_a, kSizeBytes));
  HIP_CHECK(hipMalloc(&d_dst_b, kSizeBytes));

  std::vector<int> host_a(kNumElements, kValA);
  std::vector<int> host_b(kNumElements, kValB);
  HIP_CHECK(hipMemcpy(d_src_a, host_a.data(), kSizeBytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_src_b, host_b.data(), kSizeBytes, hipMemcpyHostToDevice));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  void* dsts[] = {d_dst_a, d_dst_b};
  void* srcs[] = {d_src_a, d_src_b};
  size_t sizes[] = {kSizeBytes, kSizeBytes};
  size_t attrsIdxs[] = {0};

  hipExtMemcpyAttributes attr{};
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;
  attr.flags = hipMemcpyFlagExtPreferComputeEngine;

  hipError_t err = hipExtMemcpyBatchAsync(dsts, srcs, sizes, nullptr, nullptr, nullptr,
                                          2, &attr, attrsIdxs, 1, stream);
  HIP_CHECK(err);
  HIP_CHECK(hipStreamSynchronize(stream));

  std::vector<int> out_a(kNumElements);
  std::vector<int> out_b(kNumElements);
  HIP_CHECK(hipMemcpy(out_a.data(), d_dst_a, kSizeBytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(out_b.data(), d_dst_b, kSizeBytes, hipMemcpyDeviceToHost));
  for (size_t i = 0; i < kNumElements; ++i) {
    REQUIRE(out_a[i] == kValA);
    REQUIRE(out_b[i] == kValB);
  }

  HIP_CHECK(hipFree(d_src_a));
  HIP_CHECK(hipFree(d_src_b));
  HIP_CHECK(hipFree(d_dst_a));
  HIP_CHECK(hipFree(d_dst_b));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - hipMemcpyFlagExtPreferLinear and hipMemcpyFlagExtPreferBroadcast are accepted on a
 *   device-to-device batch where every copy shares the same source. The hints must not
 *   change the result: every destination receives the source bytes.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipExtMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipExtMemcpyBatchAsync_PreferLinearBroadcast_D2D) {
  constexpr size_t kNumDsts = 3;
  constexpr size_t kNumElements = 4096;
  constexpr size_t kSizeBytes = kNumElements * sizeof(int);
  constexpr int kVal = 37;

  const unsigned int flag = GENERATE(static_cast<unsigned int>(hipMemcpyFlagExtPreferLinear),
                                     static_cast<unsigned int>(hipMemcpyFlagExtPreferBroadcast));

  void* d_src = nullptr;
  HIP_CHECK(hipMalloc(&d_src, kSizeBytes));
  std::vector<int> host(kNumElements, kVal);
  HIP_CHECK(hipMemcpy(d_src, host.data(), kSizeBytes, hipMemcpyHostToDevice));

  void* dsts[kNumDsts] = {};
  void* srcs[kNumDsts] = {};
  size_t sizes[kNumDsts] = {};
  for (size_t d = 0; d < kNumDsts; ++d) {
    HIP_CHECK(hipMalloc(&dsts[d], kSizeBytes));
    HIP_CHECK(hipMemset(dsts[d], 0, kSizeBytes));
    srcs[d] = d_src;
    sizes[d] = kSizeBytes;
  }

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  size_t attrsIdxs[] = {0};
  hipExtMemcpyAttributes attr{};
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;
  attr.flags = flag;

  HIP_CHECK(hipExtMemcpyBatchAsync(dsts, srcs, sizes, nullptr, nullptr, nullptr, kNumDsts,
                                   &attr, attrsIdxs, 1, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  std::vector<int> out(kNumElements);
  for (size_t d = 0; d < kNumDsts; ++d) {
    HIP_CHECK(hipMemcpy(out.data(), dsts[d], kSizeBytes, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < kNumElements; ++i) {
      REQUIRE(out[i] == kVal);
    }
    HIP_CHECK(hipFree(dsts[d]));
  }

  HIP_CHECK(hipFree(d_src));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - The GPU-side wait/signal parameters are reserved and must be NULL. Passing a
 *   non-NULL waits or signals array returns hipErrorNotSupported. Device-
 *   independent: the check runs before any device/op dispatch.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipExtMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipExtMemcpyBatchAsync_WaitsSignals_Reserved_Negative) {
  constexpr size_t kBytes = 1024;

  void* d_a = nullptr;
  void* d_b = nullptr;
  HIP_CHECK(hipMalloc(&d_a, kBytes));
  HIP_CHECK(hipMalloc(&d_b, kBytes));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  void* dsts[] = {d_a};
  void* srcs[] = {d_b};
  size_t sizes[] = {kBytes};
  size_t attrsIdxs[] = {0};
  hipExtMemcpyAttributes attr{};
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;

  hipExtMemcpyWait waits[1] = {};
  hipExtMemcpySignal signals[1] = {};

  // Non-NULL waits -> reserved -> hipErrorNotSupported.
  hipError_t errW = hipExtMemcpyBatchAsync(dsts, srcs, sizes, nullptr,
                                           waits, nullptr, 1,
                                           &attr, attrsIdxs, 1, stream);
  REQUIRE(errW == hipErrorNotSupported);

  // Non-NULL signals -> reserved -> hipErrorNotSupported.
  hipError_t errS = hipExtMemcpyBatchAsync(dsts, srcs, sizes, nullptr,
                                           nullptr, signals, 1,
                                           &attr, attrsIdxs, 1, stream);
  REQUIRE(errS == hipErrorNotSupported);

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - Stream capture is not supported: calling on a capturing stream (origin or joined) returns
 *   hipErrorStreamCaptureUnsupported and invalidates the capture.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipExtMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipExtMemcpyBatchAsync_StreamCapture_Negative) {
  constexpr size_t kBytes = 1024;

  void* d_a = nullptr;
  void* d_b = nullptr;
  HIP_CHECK(hipMalloc(&d_a, kBytes));
  HIP_CHECK(hipMalloc(&d_b, kBytes));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  void* dsts[] = {d_a};
  void* srcs[] = {d_b};
  size_t sizes[] = {kBytes};

  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed));
  HIP_CHECK_ERROR(hipExtMemcpyBatchAsync(dsts, srcs, sizes, nullptr, nullptr, nullptr, 1,
                                         nullptr, nullptr, 0, stream),
                  hipErrorStreamCaptureUnsupported);

  hipGraph_t graph = nullptr;
  HIP_CHECK_ERROR(hipStreamEndCapture(stream, &graph), hipErrorStreamCaptureInvalidated);
  REQUIRE(graph == nullptr);

  // On a stream that joined another stream's capture, the origin's capture is invalidated too.
  hipStream_t side;
  hipEvent_t fork;
  HIP_CHECK(hipStreamCreate(&side));
  HIP_CHECK(hipEventCreate(&fork));
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed));
  HIP_CHECK(hipEventRecord(fork, stream));
  HIP_CHECK(hipStreamWaitEvent(side, fork, 0));
  HIP_CHECK_ERROR(hipExtMemcpyBatchAsync(dsts, srcs, sizes, nullptr, nullptr, nullptr, 1,
                                         nullptr, nullptr, 0, side),
                  hipErrorStreamCaptureUnsupported);
  HIP_CHECK_ERROR(hipStreamEndCapture(stream, &graph), hipErrorStreamCaptureInvalidated);
  REQUIRE(graph == nullptr);

  HIP_CHECK(hipEventDestroy(fork));
  HIP_CHECK(hipStreamDestroy(side));
  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - For a swap entry, sizesDst[i] must be non-zero and <= sizes[i]. Both a zero
 *   and an over-large sizesDst return hipErrorInvalidValue, ahead of swap support
 *   and transfer-type checks (device and host-to-host entries).
 * Test source
 * ------------------------
 * - catch/unit/memory/hipExtMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipExtMemcpyBatchAsync_SizesDst_Constraint_Negative) {
  constexpr size_t kBytes = 4096;

  void* d_a = nullptr;
  void* d_b = nullptr;
  HIP_CHECK(hipMalloc(&d_a, kBytes));
  HIP_CHECK(hipMalloc(&d_b, kBytes));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  void* dsts[] = {d_a};
  void* srcs[] = {d_b};
  size_t sizes[] = {kBytes};
  size_t attrsIdxs[] = {0};
  hipExtMemcpyAttributes attr{};
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;
  attr.flags = hipMemcpyFlagExtOpSwap;

  // Case 1: sizesDst[0] == 0 (invalid for a swap entry).
  size_t sizesDst_zero[] = {0};
  hipError_t err = hipExtMemcpyBatchAsync(dsts, srcs, sizes, sizesDst_zero,
                                          nullptr, nullptr, 1,
                                          &attr, attrsIdxs, 1, stream);
  REQUIRE(err == hipErrorInvalidValue);

  // Case 2: sizesDst[0] > sizes[0] (invalid for a swap entry).
  size_t sizesDst_big[] = {kBytes * 2};
  err = hipExtMemcpyBatchAsync(dsts, srcs, sizes, sizesDst_big,
                               nullptr, nullptr, 1,
                               &attr, attrsIdxs, 1, stream);
  REQUIRE(err == hipErrorInvalidValue);

  // Case 3: host-to-host swap entry, which would otherwise be hipErrorNotSupported.
  std::vector<char> h_a(kBytes), h_b(kBytes);
  void* h_dsts[] = {h_a.data()};
  void* h_srcs[] = {h_b.data()};
  err = hipExtMemcpyBatchAsync(h_dsts, h_srcs, sizes, sizesDst_zero,
                               nullptr, nullptr, 1,
                               &attr, attrsIdxs, 1, stream);
  REQUIRE(err == hipErrorInvalidValue);

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - Asymmetric swap (sizes > sizesDst) between device and pinned host memory.
 *   The first sizesDst bytes are swapped between the two buffers; the srcs-side tail
 *   (sizes - sizesDst) is copied into dsts. Verifies the exchanged head and copied
 *   tail on both operands.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipExtMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipExtMemcpyBatchAsync_HD_Swap_Asymmetric) {
  constexpr size_t kSizeA = 8192;
  constexpr size_t kSizeB = 4096;
  constexpr int kValA = 42;
  constexpr int kValB = 99;

  int* h_a = nullptr;             // pinned host (A side)
  void* d_b = nullptr;            // device (B side)
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&h_a), kSizeA));
  HIP_CHECK(hipMalloc(&d_b, kSizeA));

  const size_t totalElems = kSizeA / sizeof(int);
  for (size_t i = 0; i < totalElems; i++) h_a[i] = kValA;
  std::vector<int> hostB(totalElems, kValB);
  HIP_CHECK(hipMemcpy(d_b, hostB.data(), kSizeA, hipMemcpyHostToDevice));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  void* dsts[] = {d_b};   // B = device
  void* srcs[] = {h_a};   // A = pinned host
  size_t sizes[] = {kSizeA};
  size_t sizesDst[] = {kSizeB};
  size_t attrsIdxs[] = {0};
  hipExtMemcpyAttributes attr{};
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;
  attr.flags = hipMemcpyFlagExtOpSwap;

  hipError_t err = hipExtMemcpyBatchAsync(dsts, srcs, sizes, sizesDst,
                                          nullptr, nullptr, 1,
                                          &attr, attrsIdxs, 1, stream);
  if (err == hipErrorNotSupported) {
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipHostFree(h_a));
    HIP_CHECK(hipFree(d_b));
    HIP_SKIP_TEST(HipTest::SkipReason::kSdmaSwapUnsupported);
  }
  HIP_CHECK(err);
  HIP_CHECK(hipStreamSynchronize(stream));

  std::vector<int> resultB(totalElems);
  HIP_CHECK(hipMemcpy(resultB.data(), d_b, kSizeA, hipMemcpyDeviceToHost));

  const size_t swapElems = kSizeB / sizeof(int);
  for (size_t i = 0; i < swapElems; i++) {
    REQUIRE(h_a[i] == kValB);      // A head: swapped from B
    REQUIRE(resultB[i] == kValA);  // B head: swapped from A
  }
  for (size_t i = swapElems; i < totalElems; i++) {
    REQUIRE(h_a[i] == kValA);      // A tail: unchanged
    REQUIRE(resultB[i] == kValA);  // B tail: copied from A
  }

  HIP_CHECK(hipHostFree(h_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - A single batch that mixes an asymmetric swap entry with a plain linear-copy entry.
 *   Exercises the asymmetric-swap decomposition (symmetric head swap + linear tail)
 *   alongside an unrelated op in the same submission. The decomposed head-swap and
 *   tail-copy touch disjoint byte ranges (split at sizesDst), so the batch is correct
 *   regardless of intra-batch execution order.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipExtMemcpyBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Unit_hipExtMemcpyBatchAsync_MixedAsymmetricSwapAndLinear) {
  constexpr size_t kSizeA = 8192;
  constexpr size_t kSizeB = 4096;
  constexpr int kValA = 42;
  constexpr int kValB = 99;
  constexpr int kValSrc = 7;

  int* h_a = nullptr;    // pinned host (swap A side)
  void* d_b = nullptr;   // device (swap B side)
  void* d_lin = nullptr; // device (linear-copy dest)
  int* h_lin = nullptr;  // pinned host (linear-copy src)
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&h_a), kSizeA));
  HIP_CHECK(hipMalloc(&d_b, kSizeA));
  HIP_CHECK(hipMalloc(&d_lin, kSizeA));
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&h_lin), kSizeA));

  const size_t totalElems = kSizeA / sizeof(int);
  for (size_t i = 0; i < totalElems; i++) {
    h_a[i] = kValA;
    h_lin[i] = kValSrc;
  }
  std::vector<int> hostB(totalElems, kValB);
  HIP_CHECK(hipMemcpy(d_b, hostB.data(), kSizeA, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_lin, 0, kSizeA));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  void* dsts[] = {d_b, d_lin};
  void* srcs[] = {h_a, h_lin};
  size_t sizes[] = {kSizeA, kSizeA};
  size_t sizesDst[] = {kSizeB, 0};  // entry0 asymmetric; entry1 non-swap => 0
  size_t attrsIdxs[] = {0, 1};
  hipExtMemcpyAttributes attrs[2]{};
  attrs[0].srcAccessOrder = hipMemcpySrcAccessOrderStream;
  attrs[0].flags = hipMemcpyFlagExtOpSwap;
  attrs[1].srcAccessOrder = hipMemcpySrcAccessOrderStream;
  attrs[1].flags = 0;  // plain linear copy

  hipError_t err = hipExtMemcpyBatchAsync(dsts, srcs, sizes, sizesDst,
                                          nullptr, nullptr, 2,
                                          attrs, attrsIdxs, 2, stream);
  if (err == hipErrorNotSupported) {
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipHostFree(h_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_lin));
    HIP_CHECK(hipHostFree(h_lin));
    HIP_SKIP_TEST(HipTest::SkipReason::kSdmaSwapUnsupported);
  }
  HIP_CHECK(err);
  HIP_CHECK(hipStreamSynchronize(stream));

  // Entry 0: asymmetric swap. B receives all of A; A's head receives B, A's tail unchanged.
  std::vector<int> resultB(totalElems);
  HIP_CHECK(hipMemcpy(resultB.data(), d_b, kSizeA, hipMemcpyDeviceToHost));
  const size_t swapElems = kSizeB / sizeof(int);
  for (size_t i = 0; i < swapElems; i++) {
    REQUIRE(h_a[i] == kValB);
    REQUIRE(resultB[i] == kValA);
  }
  for (size_t i = swapElems; i < totalElems; i++) {
    REQUIRE(h_a[i] == kValA);
    REQUIRE(resultB[i] == kValA);
  }
  // Entry 1: plain linear copy is unaffected by the decomposed swap.
  std::vector<int> resultLin(totalElems);
  HIP_CHECK(hipMemcpy(resultLin.data(), d_lin, kSizeA, hipMemcpyDeviceToHost));
  for (size_t i = 0; i < totalElems; i++) REQUIRE(resultLin[i] == kValSrc);

  HIP_CHECK(hipHostFree(h_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_lin));
  HIP_CHECK(hipHostFree(h_lin));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 * - A non-zero sizesDst[i] on a non-swap entry is rejected with hipErrorInvalidValue;
 *   sizesDst is only meaningful for swap entries.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipExtMemcpyBatchAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Unit_hipExtMemcpyBatchAsync_SizesDst_NonSwap_Negative) {
  constexpr size_t kSize = 4096;
  LinearAllocGuard<int> src(LinearAllocs::hipMalloc, kSize);
  LinearAllocGuard<int> dst(LinearAllocs::hipMalloc, kSize);
  StreamGuard stream_guard(Streams::created);

  void* dsts[] = {dst.ptr()};
  void* srcs[] = {src.ptr()};
  size_t sizes[] = {kSize};
  size_t sizesDst[] = {kSize / 2};  // non-zero on a non-swap (default) entry
  size_t attrs_idxs[] = {0};
  hipExtMemcpyAttributes attr{};
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;
  attr.flags = 0;  // no swap

  HIP_CHECK_ERROR(hipExtMemcpyBatchAsync(dsts, srcs, sizes, sizesDst,
                                         nullptr, nullptr, 1,
                                         &attr, attrs_idxs, 1, stream_guard.stream()),
                  hipErrorInvalidValue);
}

/**
 * Test Description
 * ------------------------
 * - Indirect source specified via attrs flags (hipMemcpyFlagExtOpIndirectSrc): srcs[i]
 *   holds a pointer to the real source buffer, read when the copy executes.
 *   Skipped where indirect copy is unsupported.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipExtMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipExtMemcpyBatchAsync_IndirectSrc) {
  constexpr size_t copy_size = kSmallCopySize;

  StreamGuard stream_guard(Streams::created);
  LinearAllocGuard<int> src(LinearAllocs::hipHostMalloc, copy_size);
  LinearAllocGuard<int> dst(LinearAllocs::hipMalloc, copy_size);
  LinearAllocGuard<char> src_slot(LinearAllocs::hipHostMalloc, sizeof(void*));

  const size_t copy_elements = copy_size / sizeof(int);
  std::fill_n(src.host_ptr(), copy_elements, kPatternValue);

  void* src_ptr = src.ptr();
  std::memcpy(src_slot.ptr(), &src_ptr, sizeof(void*));

  std::vector<void*> dst_ptrs{dst.ptr()};
  std::vector<void*> src_ptrs{src_slot.ptr()};
  std::vector<size_t> sizes{copy_size};
  size_t attrs_idx = 0;
  hipExtMemcpyAttributes attr{};
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;
  attr.flags = hipMemcpyFlagExtOpIndirectSrc;

  hipError_t status = hipExtMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(),
                                             nullptr, nullptr, nullptr, 1, &attr, &attrs_idx,
                                             1, stream_guard.stream());
  if (status == hipErrorNotSupported) {
    HIP_SKIP_TEST(HipTest::SkipReason::kSdmaIndirectUnsupported);
  }
  HIP_CHECK(status);
  HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));
  RequireDeviceFilled(dst.ptr(), copy_size, kPatternValue);
}

/**
 * Test Description
 * ------------------------
 * - Indirect destination specified via attrs flags (hipMemcpyFlagExtOpIndirectDst):
 *   dsts[i] holds a pointer to the real destination buffer, read when the copy
 *   executes. Skipped where indirect copy is unsupported.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipExtMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipExtMemcpyBatchAsync_IndirectDst) {
  constexpr size_t copy_size = kSmallCopySize;

  StreamGuard stream_guard(Streams::created);
  LinearAllocGuard<int> src(LinearAllocs::hipMalloc, copy_size);
  LinearAllocGuard<int> dst(LinearAllocs::hipHostMalloc, copy_size);
  LinearAllocGuard<char> dst_slot(LinearAllocs::hipHostMalloc, sizeof(void*));

  const size_t copy_elements = copy_size / sizeof(int);
  std::vector<int> host_pattern(copy_elements, kPatternValue);
  HIP_CHECK(hipMemcpy(src.ptr(), host_pattern.data(), copy_size, hipMemcpyHostToDevice));
  std::fill_n(dst.host_ptr(), copy_elements, 0);

  void* dst_ptr = dst.ptr();
  std::memcpy(dst_slot.ptr(), &dst_ptr, sizeof(void*));

  std::vector<void*> src_ptrs{src.ptr()};
  std::vector<void*> dst_ptrs{dst_slot.ptr()};
  std::vector<size_t> sizes{copy_size};
  size_t attrs_idx = 0;
  hipExtMemcpyAttributes attr{};
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;
  attr.flags = hipMemcpyFlagExtOpIndirectDst;

  hipError_t status = hipExtMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(),
                                             nullptr, nullptr, nullptr, 1, &attr, &attrs_idx,
                                             1, stream_guard.stream());
  if (status == hipErrorNotSupported) {
    HIP_SKIP_TEST(HipTest::SkipReason::kSdmaIndirectUnsupported);
  }
  HIP_CHECK(status);
  HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));
  REQUIRE(std::all_of(dst.host_ptr(), dst.host_ptr() + copy_elements,
                    [](int v) { return v == kPatternValue; }));
}

/**
 * Test Description
 * ------------------------
 * - Both source and destination indirect via attrs flags (hipMemcpyFlagExtOpIndirectSrc |
 *   hipMemcpyFlagExtOpIndirectDst): srcs[i] and dsts[i] each hold a pointer to the
 *   real buffer, read when the copy executes. Skipped where indirect copy is
 *   unsupported.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipExtMemcpyBatchAsync.cc
 */
HIP_TEST_CASE(Unit_hipExtMemcpyBatchAsync_IndirectSrcDst) {
  constexpr size_t copy_size = kSmallCopySize;

  StreamGuard stream_guard(Streams::created);
  LinearAllocGuard<int> src(LinearAllocs::hipHostMalloc, copy_size);  // real src (host)
  LinearAllocGuard<int> dst(LinearAllocs::hipMalloc, copy_size);      // real dst (device)
  LinearAllocGuard<char> src_slot(LinearAllocs::hipHostMalloc, sizeof(void*));  // host slot
  LinearAllocGuard<char> dst_slot(LinearAllocs::hipMalloc, sizeof(void*));      // device slot

  const size_t copy_elements = copy_size / sizeof(int);
  std::fill_n(src.host_ptr(), copy_elements, kPatternValue);

  void* src_ptr = src.ptr();
  void* dst_ptr = dst.ptr();
  std::memcpy(src_slot.ptr(), &src_ptr, sizeof(void*));
  HIP_CHECK(hipMemcpy(dst_slot.ptr(), &dst_ptr, sizeof(void*), hipMemcpyHostToDevice));

  std::vector<void*> src_ptrs{src_slot.ptr()};
  std::vector<void*> dst_ptrs{dst_slot.ptr()};
  std::vector<size_t> sizes{copy_size};
  size_t attrs_idx = 0;
  hipExtMemcpyAttributes attr{};
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;
  attr.flags = hipMemcpyFlagExtOpIndirectSrc | hipMemcpyFlagExtOpIndirectDst;

  hipError_t status = hipExtMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(),
                                             nullptr, nullptr, nullptr, 1, &attr, &attrs_idx,
                                             1, stream_guard.stream());
  if (status == hipErrorNotSupported) {
    HIP_SKIP_TEST(HipTest::SkipReason::kSdmaIndirectUnsupported);
  }
  HIP_CHECK(status);
  HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));
  RequireDeviceFilled(dst.ptr(), copy_size, kPatternValue);
}

/**
 * End doxygen group hipExtMemcpyBatchAsync.
 * @}
 */
