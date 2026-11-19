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
constexpr size_t kBytes = 256;

int CurrentDevice() {
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  return device;
}

std::array<uint8_t, kBytes> MakePattern(uint8_t seed) {
  std::array<uint8_t, kBytes> pattern{};
  for (size_t i = 0; i < pattern.size(); ++i) {
    pattern[i] = static_cast<uint8_t>(seed + i);
  }
  return pattern;
}

hipMemLocation DeviceLocation() {
  hipMemLocation location{};
  location.type = hipMemLocationTypeDevice;
  location.id = CurrentDevice();
  return location;
}
}  // namespace

// @asserts: hipMemcpyBatchAsync - a two-op batch delivers each source to its matching destination after synchronize (or reports unsupported)
HIP_TEST_CASE(Contract_MemBatchCopy_HipMemcpyBatchAsync_TwoOps_RoundTripBytes) {
  hip::contract::ContractCleanup cleanup;
  const auto src_a = MakePattern(0x10);
  const auto src_b = MakePattern(0x90);
  std::array<uint8_t, kBytes> out_a{};
  std::array<uint8_t, kBytes> out_b{};

  void* dev_src_a = nullptr;
  void* dev_src_b = nullptr;
  void* dev_dst_a = nullptr;
  void* dev_dst_b = nullptr;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipMalloc(&dev_src_a, kBytes));
  cleanup.Add([dev_src_a] { (void)hipFree(dev_src_a); });
  HIP_CHECK(hipMalloc(&dev_src_b, kBytes));
  cleanup.Add([dev_src_b] { (void)hipFree(dev_src_b); });
  HIP_CHECK(hipMalloc(&dev_dst_a, kBytes));
  cleanup.Add([dev_dst_a] { (void)hipFree(dev_dst_a); });
  HIP_CHECK(hipMalloc(&dev_dst_b, kBytes));
  cleanup.Add([dev_dst_b] { (void)hipFree(dev_dst_b); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });

  HIP_CHECK(hipMemcpy(dev_src_a, src_a.data(), kBytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dev_src_b, src_b.data(), kBytes, hipMemcpyHostToDevice));

  // A batch of two independent device-to-device copies must deliver each source
  // to its matching destination after the stream is synchronized, behaving like
  // two ordinary device-to-device copies issued together. A single default
  // attributes entry (stream access ordering, device location hints) is applied
  // to both copies via a shared attribute index. Supplying one attribute rather
  // than none is the portable form: the AMD runtime accepts a null attributes
  // array and applies defaults, but the NVIDIA path (cudaMemcpyBatchAsync)
  // requires at least one attributes entry and rejects numAttrs == 0.
  hipMemcpyAttributes attribute{};
  attribute.srcAccessOrder = hipMemcpySrcAccessOrderStream;
  attribute.srcLocHint = DeviceLocation();
  attribute.dstLocHint = DeviceLocation();

  void* dsts[2] = {dev_dst_a, dev_dst_b};
  void* srcs[2] = {dev_src_a, dev_src_b};
  size_t sizes[2] = {kBytes, kBytes};
  hipMemcpyAttributes attrs[1] = {attribute};
  size_t attr_indices[1] = {0};
  size_t fail_index = 0;
  const hipError_t status =
      hipMemcpyBatchAsync(dsts, srcs, sizes, 2, attrs, attr_indices, 1, &fail_index, stream);
  if (status == hipErrorNotSupported) {
    HIP_SKIP_TEST("Batch memcpy is not supported by this device/runtime path.");
  }
  HIP_CHECK(status);
  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipMemcpy(out_a.data(), dev_dst_a, kBytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(out_b.data(), dev_dst_b, kBytes, hipMemcpyDeviceToHost));
  REQUIRE(out_a == src_a);
  REQUIRE(out_b == src_b);
}

// @asserts: hipMemcpyBatchAsync - a per-copy attribute (access order, location hints) does not change the copied bytes (or reports unsupported)
HIP_TEST_CASE(Contract_MemBatchCopy_HipMemcpyBatchAsync_WithAttributes_RoundTripBytes) {
  hip::contract::ContractCleanup cleanup;
  const auto src = MakePattern(0x24);
  std::array<uint8_t, kBytes> out{};

  void* dev_src = nullptr;
  void* dev_dst = nullptr;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipMalloc(&dev_src, kBytes));
  cleanup.Add([dev_src] { (void)hipFree(dev_src); });
  HIP_CHECK(hipMalloc(&dev_dst, kBytes));
  cleanup.Add([dev_dst] { (void)hipFree(dev_dst); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipMemcpy(dev_src, src.data(), kBytes, hipMemcpyHostToDevice));

  // A per-copy attribute selecting stream access ordering and device location
  // hints must not change the copy result: the destination still receives the
  // source bytes. The attribute index maps the single attribute onto the single
  // copy operation.
  hipMemcpyAttributes attribute{};
  attribute.srcAccessOrder = hipMemcpySrcAccessOrderStream;
  attribute.srcLocHint = DeviceLocation();
  attribute.dstLocHint = DeviceLocation();

  void* dsts[1] = {dev_dst};
  void* srcs[1] = {dev_src};
  size_t sizes[1] = {kBytes};
  hipMemcpyAttributes attrs[1] = {attribute};
  size_t attr_indices[1] = {0};
  size_t fail_index = 0;
  const hipError_t status =
      hipMemcpyBatchAsync(dsts, srcs, sizes, 1, attrs, attr_indices, 1, &fail_index, stream);
  if (status == hipErrorNotSupported) {
    HIP_SKIP_TEST("Batch memcpy is not supported by this device/runtime path.");
  }
  HIP_CHECK(status);
  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipMemcpy(out.data(), dev_dst, kBytes, hipMemcpyDeviceToHost));
  REQUIRE(out == src);
}

// @asserts: hipMemcpyBatchAsync - a batch with a null destination never returns hipSuccess (or reports unsupported)
HIP_TEST_CASE(Contract_MemBatchCopy_HipMemcpyBatchAsync_NullDestination_IsRejected) {
  hip::contract::ContractCleanup cleanup;
  const auto src = MakePattern(0x33);
  void* dev_src = nullptr;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipMalloc(&dev_src, kBytes));
  cleanup.Add([dev_src] { (void)hipFree(dev_src); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipMemcpy(dev_src, src.data(), kBytes, hipMemcpyHostToDevice));

  // A batch containing a null destination must not silently succeed. The exact
  // error code is backend-specific, so only a non-success status is required.
  // The sticky error left by the rejected call is cleared afterward so it does
  // not leak into later tests. A valid default attributes entry is supplied so
  // the rejection is attributable to the null destination rather than a missing
  // attributes array (which the NVIDIA path rejects on its own).
  hipMemcpyAttributes attribute{};
  attribute.srcAccessOrder = hipMemcpySrcAccessOrderStream;
  attribute.srcLocHint = DeviceLocation();
  attribute.dstLocHint = DeviceLocation();

  HIP_CHECK(hipGetLastError());
  void* dsts[1] = {nullptr};
  void* srcs[1] = {dev_src};
  size_t sizes[1] = {kBytes};
  hipMemcpyAttributes attrs[1] = {attribute};
  size_t attr_indices[1] = {0};
  size_t fail_index = 0;
  const hipError_t status =
      hipMemcpyBatchAsync(dsts, srcs, sizes, 1, attrs, attr_indices, 1, &fail_index, stream);
  if (status == hipErrorNotSupported) {
    HIP_SKIP_TEST("Batch memcpy is not supported by this device/runtime path.");
  }
  REQUIRE(status != hipSuccess);
  (void)hipGetLastError();
}

#if HT_AMD
// @asserts: hipExtMemcpyBatchAsync - a two-op batch with all extended parameters null (no asymmetric swap or wait/signal) and default attributes delivers each source to its destination after synchronize (or reports unsupported)
HIP_TEST_CASE(Contract_MemBatchCopy_HipExtMemcpyBatchAsync_TwoOps_RoundTripBytes) {
  hip::contract::ContractCleanup cleanup;
  const auto src_a = MakePattern(0x11);
  const auto src_b = MakePattern(0x91);
  std::array<uint8_t, kBytes> out_a{};
  std::array<uint8_t, kBytes> out_b{};

  void* dev_src_a = nullptr;
  void* dev_src_b = nullptr;
  void* dev_dst_a = nullptr;
  void* dev_dst_b = nullptr;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipMalloc(&dev_src_a, kBytes));
  cleanup.Add([dev_src_a] { (void)hipFree(dev_src_a); });
  HIP_CHECK(hipMalloc(&dev_src_b, kBytes));
  cleanup.Add([dev_src_b] { (void)hipFree(dev_src_b); });
  HIP_CHECK(hipMalloc(&dev_dst_a, kBytes));
  cleanup.Add([dev_dst_a] { (void)hipFree(dev_dst_a); });
  HIP_CHECK(hipMalloc(&dev_dst_b, kBytes));
  cleanup.Add([dev_dst_b] { (void)hipFree(dev_dst_b); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });

  HIP_CHECK(hipMemcpy(dev_src_a, src_a.data(), kBytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dev_src_b, src_b.data(), kBytes, hipMemcpyHostToDevice));

  hipExtMemcpyAttributes attribute{};
  attribute.srcAccessOrder = hipMemcpySrcAccessOrderStream;
  attribute.srcLocHint = DeviceLocation();
  attribute.dstLocHint = DeviceLocation();

  void* dsts[2] = {dev_dst_a, dev_dst_b};
  void* srcs[2] = {dev_src_a, dev_src_b};
  size_t sizes[2] = {kBytes, kBytes};
  hipExtMemcpyAttributes attrs[1] = {attribute};
  size_t attr_indices[1] = {0};
  const hipError_t status =
      hipExtMemcpyBatchAsync(dsts, srcs, sizes, /*sizesDst=*/nullptr, /*waits=*/nullptr,
                             /*signals=*/nullptr, 2, attrs, attr_indices, 1,
                             stream);
  if (status == hipErrorNotSupported) {
    HIP_SKIP_TEST("Extended batch memcpy is not supported by this device/runtime path.");
  }
  HIP_CHECK(status);
  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipMemcpy(out_a.data(), dev_dst_a, kBytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(out_b.data(), dev_dst_b, kBytes, hipMemcpyDeviceToHost));
  REQUIRE(out_a == src_a);
  REQUIRE(out_b == src_b);
}

// @asserts: hipExtMemcpyBatchAsync - zero-initialized wait and signal entries (op None) behave like null arrays: the copy runs and delivers the source bytes (or reports unsupported)
HIP_TEST_CASE(Contract_MemBatchCopy_HipExtMemcpyBatchAsync_NoneWaitSignal_RoundTripBytes) {
  hip::contract::ContractCleanup cleanup;
  const auto src = MakePattern(0x5a);
  std::array<uint8_t, kBytes> out{};

  void* dev_src = nullptr;
  void* dev_dst = nullptr;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipMalloc(&dev_src, kBytes));
  cleanup.Add([dev_src] { (void)hipFree(dev_src); });
  HIP_CHECK(hipMalloc(&dev_dst, kBytes));
  cleanup.Add([dev_dst] { (void)hipFree(dev_dst); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipMemcpy(dev_src, src.data(), kBytes, hipMemcpyHostToDevice));

  // Zero-initialized entries mean "no wait" and "no signal", so the batch is an
  // ordinary copy and needs no copy-engine wait/signal support.
  hipExtMemcpyWait waits[1] = {};
  hipExtMemcpySignal signals[1] = {};
  void* dsts[1] = {dev_dst};
  void* srcs[1] = {dev_src};
  size_t sizes[1] = {kBytes};
  const hipError_t status =
      hipExtMemcpyBatchAsync(dsts, srcs, sizes, /*sizesDst=*/nullptr, waits, signals, 1,
                             /*attrs=*/nullptr, /*attrsIdxs=*/nullptr, 0, stream);
  if (status == hipErrorNotSupported) {
    HIP_SKIP_TEST("Extended batch memcpy is not supported by this device/runtime path.");
  }
  HIP_CHECK(status);
  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipMemcpy(out.data(), dev_dst, kBytes, hipMemcpyDeviceToHost));
  REQUIRE(out == src);
}

// @asserts: hipExtMemcpyBatchAsync - a malformed wait or signal entry (NULL or misaligned address, unknown op, nonzero reserved field, nonzero fields with op None) returns hipErrorInvalidValue on any device
HIP_TEST_CASE(Contract_MemBatchCopy_HipExtMemcpyBatchAsync_MalformedWaitSignal_ReturnsInvalidValue) {
  hip::contract::ContractCleanup cleanup;
  void* dev_src = nullptr;
  void* dev_dst = nullptr;
  uint64_t* dev_flag = nullptr;
  hipStream_t stream = nullptr;

  HIP_CHECK(hipMalloc(&dev_src, kBytes));
  cleanup.Add([dev_src] { (void)hipFree(dev_src); });
  HIP_CHECK(hipMalloc(&dev_dst, kBytes));
  cleanup.Add([dev_dst] { (void)hipFree(dev_dst); });
  HIP_CHECK(hipMalloc(&dev_flag, 2 * sizeof(uint64_t)));
  cleanup.Add([dev_flag] { (void)hipFree(dev_flag); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });

  // Entry shape is validated before device support, so each case is rejected
  // with hipErrorInvalidValue even where waits/signals are unsupported.
  void* misaligned = reinterpret_cast<uint8_t*>(dev_flag) + 4;
  void* dsts[1] = {dev_dst};
  void* srcs[1] = {dev_src};
  size_t sizes[1] = {kBytes};
  const auto submit = [&](hipExtMemcpyWait wait, hipExtMemcpySignal signal) {
    return hipExtMemcpyBatchAsync(dsts, srcs, sizes, /*sizesDst=*/nullptr, &wait, &signal, 1,
                                  /*attrs=*/nullptr, /*attrsIdxs=*/nullptr, 0, stream);
  };

  HIP_CHECK(hipGetLastError());
  {
    hipExtMemcpyWait wait{};
    wait.compareOp = hipExtMemcpyWaitEq;
    REQUIRE(submit(wait, {}) == hipErrorInvalidValue);
    wait.addr = misaligned;
    REQUIRE(submit(wait, {}) == hipErrorInvalidValue);
    wait.addr = dev_flag;
    wait.compareOp = hipExtMemcpyWaitGt + 1;
    REQUIRE(submit(wait, {}) == hipErrorInvalidValue);
    wait.compareOp = hipExtMemcpyWaitEq;
    wait.reserved[0] = 1;
    REQUIRE(submit(wait, {}) == hipErrorInvalidValue);
    hipExtMemcpyWait none{};
    none.addr = dev_flag;
    REQUIRE(submit(none, {}) == hipErrorInvalidValue);
  }
  {
    hipExtMemcpySignal signal{};
    signal.signalOp = hipExtMemcpySignalWrite;
    REQUIRE(submit({}, signal) == hipErrorInvalidValue);
    signal.addr = misaligned;
    REQUIRE(submit({}, signal) == hipErrorInvalidValue);
    signal.addr = dev_flag;
    signal.signalOp = hipExtMemcpySignalSub + 1;
    REQUIRE(submit({}, signal) == hipErrorInvalidValue);
    signal.signalOp = hipExtMemcpySignalWrite;
    signal.reserved0 = 1;
    REQUIRE(submit({}, signal) == hipErrorInvalidValue);
    hipExtMemcpySignal none{};
    none.data = 1;
    REQUIRE(submit({}, none) == hipErrorInvalidValue);
  }
  (void)hipGetLastError();
}
#endif  // HT_AMD
