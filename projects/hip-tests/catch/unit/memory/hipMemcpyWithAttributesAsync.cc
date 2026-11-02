/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>
#include <cstdint>
#include <vector>

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <resource_guards.hh>
#include <utils.hh>
#include "memcpyBatchAsync_common.hh"

namespace {
constexpr size_t kCopySize = 1024;
constexpr size_t kWidth = 8;
constexpr size_t kHeight = 4;
constexpr size_t kDepth = 2;

hipMemcpyAttributes MakeAttr(const hipMemcpySrcAccessOrder order) {
  hipMemcpyAttributes attr{};
  attr.srcAccessOrder = order;
  attr.flags = hipMemcpyFlagDefault;
  return attr;
}

std::vector<unsigned char> MakePattern(const size_t size) {
  std::vector<unsigned char> data(size);
  for (size_t i = 0; i < size; ++i) {
    data[i] = static_cast<unsigned char>(i & 0xFFu);
  }
  return data;
}

hipMemcpy3DBatchOp MakeDenseOp(void* source, void* destination, const hipExtent extent) {
  hipMemcpy3DBatchOp op{};
  op.src.type = hipMemcpyOperandTypePointer;
  op.src.op.ptr.ptr = source;
  op.src.op.ptr.layerHeight = kHeight;
  op.dst.type = hipMemcpyOperandTypePointer;
  op.dst.op.ptr.ptr = destination;
  op.dst.op.ptr.layerHeight = kHeight;
  op.extent = extent;
  op.srcAccessOrder = hipMemcpySrcAccessOrderStream;
  op.flags = hipMemcpyFlagDefault;
  return op;
}
}  // namespace

/**
 * @addtogroup hipMemcpyWithAttributesAsync hipMemcpyWithAttributesAsync
 * @{
 * @ingroup MemoryTest
 * `hipError_t hipMemcpyWithAttributesAsync(void* dst, const void* src, size_t size,
 * hipMemcpyAttributes* attr, hipStream_t stream __dparm(0))`
 *
 * This API is a single copy form of hipMemcpyBatchAsync and shares its implementation, so
 * transfer directions, sizes and flag handling are already covered by the batch tests. These
 * cases stay deliberately small: one sanity copy per source access order, and the argument
 * checks specific to this entry point.
 */

/**
 * Test Description
 * ------------------------
 *  - Copies a host buffer to a device buffer once per source access order and verifies the
 *    result. Host to device is used because it is the direction where srcAccessOrder changes
 *    how the runtime stages the source.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpyWithAttributesAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipMemcpyWithAttributesAsync_Positive_Basic) {
  StreamGuard stream_guard(Streams::created);
  const std::vector<unsigned char> pattern = MakePattern(kCopySize);
  const std::vector<unsigned char> zeros(kCopySize, 0);

  std::vector<LinearAllocGuard<unsigned char>> allocations;
  void* src = addBuffer(allocations, pattern, LinearAllocs::hipHostMalloc);
  void* dst = addBuffer(allocations, zeros, LinearAllocs::hipMalloc);

  for (const hipMemcpySrcAccessOrder order :
       {hipMemcpySrcAccessOrderStream, hipMemcpySrcAccessOrderDuringApiCall,
        hipMemcpySrcAccessOrderAny}) {
    INFO("srcAccessOrder = " << static_cast<int>(order));
    HIP_CHECK(hipMemset(dst, 0, kCopySize));

    hipMemcpyAttributes attr = MakeAttr(order);
    HIP_CHECK(hipMemcpyWithAttributesAsync(dst, src, kCopySize, &attr, stream_guard.stream()));
    HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));

    requireBufferEquals(dst, pattern, LinearAllocs::hipMalloc);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Checks the argument validation this entry point performs. A null attribute pointer is
 *    rejected here, unlike hipMemcpyBatchAsync where numAttrs == 0 selects defaults.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpyWithAttributesAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipMemcpyWithAttributesAsync_Negative_Parameters) {
  StreamGuard stream_guard(Streams::created);
  const hipStream_t stream = stream_guard.stream();
  const std::vector<unsigned char> zeros(kCopySize, 0);

  std::vector<LinearAllocGuard<unsigned char>> allocations;
  void* src = addBuffer(allocations, zeros, LinearAllocs::hipMalloc);
  void* dst = addBuffer(allocations, zeros, LinearAllocs::hipMalloc);
  hipMemcpyAttributes attr = MakeAttr(hipMemcpySrcAccessOrderStream);

  HIP_CHECK_ERROR(hipMemcpyWithAttributesAsync(nullptr, src, kCopySize, &attr, stream),
                  hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipMemcpyWithAttributesAsync(dst, nullptr, kCopySize, &attr, stream),
                  hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipMemcpyWithAttributesAsync(dst, src, 0, &attr, stream), hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipMemcpyWithAttributesAsync(dst, src, kCopySize, nullptr, stream),
                  hipErrorInvalidValue);

  hipMemcpyAttributes bad = MakeAttr(static_cast<hipMemcpySrcAccessOrder>(0x7FFFFFFF));
  HIP_CHECK_ERROR(hipMemcpyWithAttributesAsync(dst, src, kCopySize, &bad, stream),
                  hipErrorInvalidValue);
}

/**
 * End doxygen group hipMemcpyWithAttributesAsync.
 * @}
 */

/**
 * @addtogroup hipMemcpy3DWithAttributesAsync hipMemcpy3DWithAttributesAsync
 * @{
 * @ingroup MemoryTest
 * `hipError_t hipMemcpy3DWithAttributesAsync(hipMemcpy3DBatchOp* op, unsigned long long flags,
 * hipStream_t stream __dparm(0))`
 */

/**
 * Test Description
 * ------------------------
 *  - Performs one dense device to device 3D copy and verifies the result.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpyWithAttributesAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipMemcpy3DWithAttributesAsync_Positive_Basic) {
  constexpr size_t element_count = kWidth * kHeight * kDepth;
  constexpr size_t size_bytes = element_count * sizeof(uint32_t);

  std::vector<uint32_t> pattern(element_count);
  for (uint32_t i = 0; i < element_count; ++i) {
    pattern[i] = 0x11000000u + i;
  }

  StreamGuard stream_guard(Streams::created);
  LinearAllocGuard<uint32_t> source(LinearAllocs::hipMalloc, size_bytes);
  LinearAllocGuard<uint32_t> destination(LinearAllocs::hipMalloc, size_bytes);
  HIP_CHECK(hipMemcpy(source.ptr(), pattern.data(), size_bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(destination.ptr(), 0xFF, size_bytes));

  hipMemcpy3DBatchOp op = MakeDenseOp(source.ptr(), destination.ptr(),
                                      make_hipExtent(kWidth * sizeof(uint32_t), kHeight, kDepth));

  HIP_CHECK(hipMemcpy3DWithAttributesAsync(&op, 0, stream_guard.stream()));
  HIP_CHECK(hipStreamSynchronize(stream_guard.stream()));

  std::vector<uint32_t> copied(element_count);
  HIP_CHECK(hipMemcpy(copied.data(), destination.ptr(), size_bytes, hipMemcpyDeviceToHost));
  REQUIRE(copied == pattern);
}

/**
 * Test Description
 * ------------------------
 *  - Checks the argument validation this entry point performs, including the nonzero flags
 *    rejection it shares with hipMemcpy3DBatchAsync.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpyWithAttributesAsync.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipMemcpy3DWithAttributesAsync_Negative_Parameters) {
  constexpr size_t size_bytes = kWidth * kHeight * kDepth * sizeof(uint32_t);

  StreamGuard stream_guard(Streams::created);
  const hipStream_t stream = stream_guard.stream();
  LinearAllocGuard<uint32_t> source(LinearAllocs::hipMalloc, size_bytes);
  LinearAllocGuard<uint32_t> destination(LinearAllocs::hipMalloc, size_bytes);

  const hipExtent extent = make_hipExtent(kWidth * sizeof(uint32_t), kHeight, kDepth);
  hipMemcpy3DBatchOp op = MakeDenseOp(source.ptr(), destination.ptr(), extent);

  HIP_CHECK_ERROR(hipMemcpy3DWithAttributesAsync(nullptr, 0, stream), hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipMemcpy3DWithAttributesAsync(&op, 1, stream), hipErrorInvalidValue);

  op.srcAccessOrder = static_cast<hipMemcpySrcAccessOrder>(0x7FFFFFFF);
  HIP_CHECK_ERROR(hipMemcpy3DWithAttributesAsync(&op, 0, stream), hipErrorInvalidValue);
}

/**
 * End doxygen group hipMemcpy3DWithAttributesAsync.
 * @}
 */
