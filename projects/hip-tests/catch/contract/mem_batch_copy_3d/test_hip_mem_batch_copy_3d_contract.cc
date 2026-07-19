/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <array>
#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <contract_cleanup.hh>

namespace {
// A one-row, one-layer extent keeps the batch 3D copy a simple contiguous byte
// run, so the host-visible result can be compared against the source directly.
constexpr size_t kBytes = 256;

std::array<uint8_t, kBytes> MakePattern(uint8_t seed) {
  std::array<uint8_t, kBytes> pattern{};
  for (size_t i = 0; i < pattern.size(); ++i) {
    pattern[i] = static_cast<uint8_t>(seed + i);
  }
  return pattern;
}

// Fills a pointer operand describing a contiguous kBytes run at ptr located on
// the given memory-location type. rowLength/layerHeight mirror the extent so a
// single row and layer are described, matching the byte-scaled extent below.
hipMemcpy3DOperand PointerOperand(void* ptr, const hipExtent& extent,
                                  hipMemLocationType location) {
  hipMemcpy3DOperand operand{};
  operand.type = hipMemcpyOperandTypePointer;
  operand.op.ptr.ptr = ptr;
  operand.op.ptr.rowLength = extent.width;
  operand.op.ptr.layerHeight = extent.height;
  operand.op.ptr.locHint.type = location;
  operand.op.ptr.locHint.id = 0;
  return operand;
}

hipMemcpy3DBatchOp PointerCopyOp(void* dst, hipMemLocationType dst_location, void* src,
                                 hipMemLocationType src_location, const hipExtent& extent) {
  hipMemcpy3DBatchOp op{};
  op.src = PointerOperand(src, extent, src_location);
  op.dst = PointerOperand(dst, extent, dst_location);
  op.extent = extent;
  op.srcAccessOrder = hipMemcpySrcAccessOrderStream;
  op.flags = hipMemcpyFlagDefault;
  return op;
}
}  // namespace

// @asserts: hipMemcpy3DBatchAsync - a batch of ordered host->device->host pointer copies round-trips bytes after stream sync
HIP_TEST_CASE(Contract_MemBatchCopy3D_HostDeviceHostRoundTrip_IsVisibleAfterSync) {
  hip::contract::ContractCleanup cleanup;
  const auto src = MakePattern(0x42);
  std::array<uint8_t, kBytes> host_out{};

  void* dev_ptr = nullptr;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipMalloc(&dev_ptr, kBytes));
  cleanup.Add([dev_ptr] { (void)hipFree(dev_ptr); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });

  // Two pointer-to-pointer copies issued as one batch: host source to the device
  // buffer, then the device buffer back to a separate host destination. After the
  // stream is synchronized the final host buffer must hold the original bytes,
  // proving the batch honored both copies in order. The extent width is
  // byte-scaled with a single row and layer, so the copy is a contiguous kBytes
  // run (matching the in-tree unit-test setup for this API).
  const hipExtent extent = make_hipExtent(kBytes, 1, 1);
  hipMemcpy3DBatchOp ops[2] = {
      PointerCopyOp(dev_ptr, hipMemLocationTypeDevice, const_cast<uint8_t*>(src.data()),
                    hipMemLocationTypeHost, extent),
      PointerCopyOp(host_out.data(), hipMemLocationTypeHost, dev_ptr, hipMemLocationTypeDevice,
                    extent),
  };

  // failIdx is not asserted on: the NVIDIA backend drops this out-parameter on
  // CUDA 13+ (the wrapper zeroes it and does not forward it to CUDA), so its
  // contents are not portable. flags must be zero (reserved).
  size_t fail_index = 0;
  const hipError_t status = hipMemcpy3DBatchAsync(2, ops, &fail_index, 0, stream);
  if (status == hipErrorNotSupported) {
    HIP_SKIP_TEST("Batch 3D memcpy is not supported by this device/runtime path.");
  }
  HIP_CHECK(status);
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(host_out == src);
}

// @asserts: hipMemcpy3DBatchAsync - a batch with zero operations is rejected with a non-success status
HIP_TEST_CASE(Contract_MemBatchCopy3D_ZeroOps_IsRejected) {
  hip::contract::ContractCleanup cleanup;
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });

  // A batch with zero operations is invalid input and must be rejected rather
  // than silently succeeding. Any pre-existing sticky error is cleared first, and
  // the sticky error left by the rejection is cleared afterward, so neither leaks
  // into later tests. The exact code is backend-specific, so only a non-success
  // status is required (the documented code is hipErrorInvalidValue).
  HIP_CHECK(hipGetLastError());
  hipMemcpy3DBatchOp ops[1] = {};
  size_t fail_index = 0;
  const hipError_t status = hipMemcpy3DBatchAsync(0, ops, &fail_index, 0, stream);
  if (status == hipErrorNotSupported) {
    HIP_SKIP_TEST("Batch 3D memcpy is not supported by this device/runtime path.");
  }
  REQUIRE(status != hipSuccess);
  (void)hipGetLastError();
}

// @asserts: hipMemcpy3DBatchAsync - a positive op count with a null operation list is rejected with a non-success status
HIP_TEST_CASE(Contract_MemBatchCopy3D_NullOpList_IsRejected) {
  hip::contract::ContractCleanup cleanup;
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });

  // A non-zero operation count with a null operation list is invalid input and
  // must be rejected. The op count is positive so the null-list check is reached
  // rather than short-circuited by a zero count.
  HIP_CHECK(hipGetLastError());
  size_t fail_index = 0;
  const hipError_t status = hipMemcpy3DBatchAsync(1, nullptr, &fail_index, 0, stream);
  if (status == hipErrorNotSupported) {
    HIP_SKIP_TEST("Batch 3D memcpy is not supported by this device/runtime path.");
  }
  REQUIRE(status != hipSuccess);
  (void)hipGetLastError();
}

// @asserts: hipMemcpy3DBatchAsync - a non-zero value in the reserved flags parameter is rejected even with a valid op list
HIP_TEST_CASE(Contract_MemBatchCopy3D_NonZeroFlags_IsRejected) {
  hip::contract::ContractCleanup cleanup;
  const auto src = MakePattern(0x42);

  void* dev_ptr = nullptr;
  hipStream_t stream = nullptr;
  HIP_CHECK(hipMalloc(&dev_ptr, kBytes));
  cleanup.Add([dev_ptr] { (void)hipFree(dev_ptr); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });

  // The flags parameter is reserved and must be zero. The op list itself is a
  // valid host-to-device copy (the same operand the positive test issues), so a
  // rejection here isolates the reserved-flag contract rather than conflating it
  // with an invalid operation.
  HIP_CHECK(hipGetLastError());
  const hipExtent extent = make_hipExtent(kBytes, 1, 1);
  hipMemcpy3DBatchOp ops[1] = {
      PointerCopyOp(dev_ptr, hipMemLocationTypeDevice, const_cast<uint8_t*>(src.data()),
                    hipMemLocationTypeHost, extent),
  };
  size_t fail_index = 0;
  const hipError_t status = hipMemcpy3DBatchAsync(1, ops, &fail_index, 0x1, stream);
  if (status == hipErrorNotSupported) {
    HIP_SKIP_TEST("Batch 3D memcpy is not supported by this device/runtime path.");
  }
  REQUIRE(status != hipSuccess);
  (void)hipGetLastError();
}
