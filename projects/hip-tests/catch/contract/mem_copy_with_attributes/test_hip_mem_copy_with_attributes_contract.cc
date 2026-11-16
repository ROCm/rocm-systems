/*
 * Copyright Advanced Micro Devices, Inc.
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
// A small contiguous run keeps the copy a plain byte sequence, so the host-visible
// result can be compared against the source directly.
constexpr size_t kBytes = 256;

std::array<uint8_t, kBytes> MakePattern(uint8_t seed) {
  std::array<uint8_t, kBytes> pattern{};
  for (size_t i = 0; i < pattern.size(); ++i) {
    pattern[i] = static_cast<uint8_t>(seed + i);
  }
  return pattern;
}

hipMemcpyAttributes StreamOrderedAttr() {
  hipMemcpyAttributes attr{};
  attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;
  attr.flags = hipMemcpyFlagDefault;
  return attr;
}

// One row, one layer, so the 3D extent describes the same contiguous run as the 1D case.
hipMemcpy3DOperand PointerOperand(void* ptr, const hipExtent& extent) {
  hipMemcpy3DOperand operand{};
  operand.type = hipMemcpyOperandTypePointer;
  operand.op.ptr.ptr = ptr;
  operand.op.ptr.rowLength = extent.width;
  operand.op.ptr.layerHeight = extent.height;
  return operand;
}

hipMemcpy3DBatchOp PointerCopyOp(void* dst, void* src, const hipExtent& extent) {
  hipMemcpy3DBatchOp op{};
  op.src = PointerOperand(src, extent);
  op.dst = PointerOperand(dst, extent);
  op.extent = extent;
  op.srcAccessOrder = hipMemcpySrcAccessOrderStream;
  op.flags = hipMemcpyFlagDefault;
  return op;
}
}  // namespace

// @asserts: hipMemcpyWithAttributesAsync - a stream-ordered host->device->host copy round-trips bytes after stream sync
HIP_TEST_CASE(Contract_MemCopyWithAttributes_HipMemcpyWithAttributesAsync_HostDeviceHostRoundTrip_IsVisibleAfterSync) {
  hip::contract::ContractCleanup cleanup;
  const auto src = MakePattern(0x42);
  std::array<uint8_t, kBytes> host_out{};

  void* device = nullptr;
  HIP_CHECK(hipMalloc(&device, kBytes));
  cleanup.Add([device] { (void)hipFree(device); });

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });

  hipMemcpyAttributes attr = StreamOrderedAttr();
  HIP_CHECK(hipMemcpyWithAttributesAsync(device, src.data(), kBytes, &attr, stream));
  HIP_CHECK(hipMemcpyWithAttributesAsync(host_out.data(), device, kBytes, &attr, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(host_out == src);
}

// @asserts: hipMemcpyWithAttributesAsync - a null attributes pointer is rejected with hipErrorInvalidValue
HIP_TEST_CASE(Contract_MemCopyWithAttributes_HipMemcpyWithAttributesAsync_NullAttributes_ReturnsInvalidValue) {
  hip::contract::ContractCleanup cleanup;
  const auto src = MakePattern(0x11);

  void* device = nullptr;
  HIP_CHECK(hipMalloc(&device, kBytes));
  cleanup.Add([device] { (void)hipFree(device); });

  HIP_CHECK_ERROR(hipMemcpyWithAttributesAsync(device, src.data(), kBytes, nullptr, nullptr),
                  hipErrorInvalidValue);
  (void)hipGetLastError();
}

// @asserts: hipMemcpy3DWithAttributesAsync - a single stream-ordered pointer copy round-trips bytes after stream sync
HIP_TEST_CASE(Contract_MemCopyWithAttributes_HipMemcpy3DWithAttributesAsync_HostDeviceHostRoundTrip_IsVisibleAfterSync) {
  hip::contract::ContractCleanup cleanup;
  const auto src = MakePattern(0x73);
  std::array<uint8_t, kBytes> host_out{};
  const hipExtent extent = make_hipExtent(kBytes, 1, 1);

  void* device = nullptr;
  HIP_CHECK(hipMalloc(&device, kBytes));
  cleanup.Add([device] { (void)hipFree(device); });

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });

  hipMemcpy3DBatchOp to_device =
      PointerCopyOp(device, const_cast<uint8_t*>(src.data()), extent);
  HIP_CHECK(hipMemcpy3DWithAttributesAsync(&to_device, 0, stream));

  hipMemcpy3DBatchOp to_host = PointerCopyOp(host_out.data(), device, extent);
  HIP_CHECK(hipMemcpy3DWithAttributesAsync(&to_host, 0, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(host_out == src);
}

// @asserts: hipMemcpy3DWithAttributesAsync - a null operation pointer and a reserved nonzero flags value are rejected with hipErrorInvalidValue
HIP_TEST_CASE(Contract_MemCopyWithAttributes_HipMemcpy3DWithAttributesAsync_NullOpAndReservedFlags_ReturnInvalidValue) {
  hip::contract::ContractCleanup cleanup;
  const auto src = MakePattern(0x5a);
  const hipExtent extent = make_hipExtent(kBytes, 1, 1);

  void* device = nullptr;
  HIP_CHECK(hipMalloc(&device, kBytes));
  cleanup.Add([device] { (void)hipFree(device); });

  HIP_CHECK_ERROR(hipMemcpy3DWithAttributesAsync(nullptr, 0, nullptr), hipErrorInvalidValue);

  hipMemcpy3DBatchOp op = PointerCopyOp(device, const_cast<uint8_t*>(src.data()), extent);
  HIP_CHECK_ERROR(hipMemcpy3DWithAttributesAsync(&op, 1, nullptr), hipErrorInvalidValue);
  (void)hipGetLastError();
}
