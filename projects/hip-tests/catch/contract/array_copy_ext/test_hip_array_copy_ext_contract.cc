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
constexpr size_t kWidth = 8;
constexpr size_t kHeight = 4;
constexpr size_t kByteCount = kWidth * kHeight;

std::array<uint8_t, kByteCount> MakePattern(uint8_t seed) {
  std::array<uint8_t, kByteCount> pattern{};
  for (size_t i = 0; i < pattern.size(); ++i) {
    pattern[i] = static_cast<uint8_t>(seed + i);
  }
  return pattern;
}

hipChannelFormatDesc ByteChannelDesc() { return hipCreateChannelDesc<uint8_t>(); }

// hipMemcpyHtoD/HtoDAsync take a const source on AMD but a non-const `void*`
// source on the NVIDIA backend (the shim forwards to cuMemcpyHtoD, whose source
// is non-const). Provide a mutable void* view of the host source so the call
// compiles on both; the copy never writes through it.
void* HtoDSrc(const void* host) { return const_cast<void*>(host); }
}  // namespace

// @asserts: hipMemcpyDtoA - a device-to-array then array-to-host copy chain round-trips the byte pattern
HIP_TEST_CASE(Contract_ArrayCopyExt_MemcpyDtoAThenAtoH_RoundTripsBytes) {
  CHECK_IMAGE_SUPPORT;

  hip::contract::ContractCleanup cleanup;
  const auto src = MakePattern(0x23);
  std::array<uint8_t, kByteCount> dst{};
  void* device_ptr = nullptr;
  hipArray_t array = nullptr;
  const auto desc = ByteChannelDesc();

  HIP_CHECK(hipMalloc(&device_ptr, src.size()));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });
  HIP_CHECK(hipMallocArray(&array, &desc, kByteCount, 1));
  cleanup.Add([array] { (void)hipFreeArray(array); });

  HIP_CHECK(
      hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(device_ptr), HtoDSrc(src.data()), src.size()));
  HIP_CHECK(hipMemcpyDtoA(array, 0, reinterpret_cast<hipDeviceptr_t>(device_ptr), src.size()));
  HIP_CHECK(hipMemcpyAtoH(dst.data(), array, 0, dst.size()));

  REQUIRE(dst == src);
}

// @asserts: hipMemcpyAtoA - an array-to-array copy preserves the bytes it transfers between two HIP arrays
HIP_TEST_CASE(Contract_ArrayCopyExt_MemcpyAtoA_RoundTripsBytes) {
  CHECK_IMAGE_SUPPORT;

  hip::contract::ContractCleanup cleanup;
  const auto src = MakePattern(0x51);
  std::array<uint8_t, kByteCount> dst{};
  hipArray_t src_array = nullptr;
  hipArray_t dst_array = nullptr;
  const auto desc = ByteChannelDesc();

  HIP_CHECK(hipMallocArray(&src_array, &desc, kByteCount, 1));
  cleanup.Add([src_array] { (void)hipFreeArray(src_array); });
  HIP_CHECK(hipMallocArray(&dst_array, &desc, kByteCount, 1));
  cleanup.Add([dst_array] { (void)hipFreeArray(dst_array); });

  // Seed the source array, copy array-to-array, then read the destination back:
  // hipMemcpyAtoA must preserve the bytes it transfers between two HIP arrays.
  HIP_CHECK(hipMemcpyHtoA(src_array, 0, src.data(), src.size()));
  HIP_CHECK(hipMemcpyAtoA(dst_array, 0, src_array, 0, src.size()));
  HIP_CHECK(hipMemcpyAtoH(dst.data(), dst_array, 0, dst.size()));

  REQUIRE(dst == src);
}

// @asserts: hipMemcpy2DArrayToArray - a 2D array-to-array copy round-trips a width-by-height byte pattern
HIP_TEST_CASE(Contract_ArrayCopyExt_Memcpy2DArrayToArray_RoundTripsBytes) {
  CHECK_IMAGE_SUPPORT;

  hip::contract::ContractCleanup cleanup;
  const auto src = MakePattern(0x45);
  std::array<uint8_t, kByteCount> dst{};
  hipArray_t src_array = nullptr;
  hipArray_t dst_array = nullptr;
  const auto desc = ByteChannelDesc();

  HIP_CHECK(hipMallocArray(&src_array, &desc, kWidth, kHeight));
  cleanup.Add([src_array] { (void)hipFreeArray(src_array); });
  HIP_CHECK(hipMallocArray(&dst_array, &desc, kWidth, kHeight));
  cleanup.Add([dst_array] { (void)hipFreeArray(dst_array); });

  HIP_CHECK(hipMemcpy2DToArray(src_array, 0, 0, src.data(), kWidth, kWidth, kHeight,
                               hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy2DArrayToArray(dst_array, 0, 0, src_array, 0, 0, kWidth, kHeight,
                                    hipMemcpyDeviceToDevice));
  HIP_CHECK(hipMemcpy2DFromArray(dst.data(), kWidth, dst_array, 0, 0, kWidth, kHeight,
                                 hipMemcpyDeviceToHost));

  REQUIRE(dst == src);
}

// @asserts: hipMemcpyHtoAAsync - rejects a null host source through the returned status
HIP_TEST_CASE(Contract_ArrayCopyExt_MemcpyHtoAAsync_NullSource_IsRejected) {
  CHECK_IMAGE_SUPPORT;

  hip::contract::ContractCleanup cleanup;
  hipArray_t array = nullptr;
  hipStream_t stream = nullptr;
  const auto desc = ByteChannelDesc();

  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMallocArray(&array, &desc, kByteCount, 1));
  cleanup.Add([array] { (void)hipFreeArray(array); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });

  const hipError_t status = hipMemcpyHtoAAsync(array, 0, nullptr, kByteCount, stream);

  // A null source must be rejected through the returned status. Whether the
  // rejection also latches into the thread-global last-error is backend-specific:
  // the AMD runtime sets it, but the NVIDIA driver path reports the error only
  // through the return value and leaves hipGetLastError() clear. Assert the
  // returned status and clear any latched error.
  REQUIRE(status != hipSuccess);
  (void)hipGetLastError();
}

// @asserts: hipMemcpyDtoA - rejects a null destination array through the returned status
HIP_TEST_CASE(Contract_ArrayCopyExt_MemcpyDtoA_NullArray_IsRejected) {
  CHECK_IMAGE_SUPPORT;

  hip::contract::ContractCleanup cleanup;
  const auto src = MakePattern(0x67);
  void* device_ptr = nullptr;

  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMalloc(&device_ptr, src.size()));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });
  HIP_CHECK(
      hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(device_ptr), HtoDSrc(src.data()), src.size()));

  const hipError_t status = hipMemcpyDtoA(nullptr, 0, reinterpret_cast<hipDeviceptr_t>(device_ptr),
                                          src.size());

  // A null destination array must be rejected through the returned status; the
  // sticky-error latch is backend-specific (set on AMD, left clear on the NVIDIA
  // driver path), so assert the returned status and clear any latched error.
  REQUIRE(status != hipSuccess);
  (void)hipGetLastError();
}

// @asserts: hipMemcpyAtoD - rejects a null source array through the returned status
HIP_TEST_CASE(Contract_ArrayCopyExt_MemcpyAtoD_NullArray_IsRejected) {
  CHECK_IMAGE_SUPPORT;

  hip::contract::ContractCleanup cleanup;
  void* device_ptr = nullptr;

  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMalloc(&device_ptr, kByteCount));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });

  const hipError_t status =
      hipMemcpyAtoD(reinterpret_cast<hipDeviceptr_t>(device_ptr), nullptr, 0, kByteCount);

  // A null source array must be rejected through the returned status; the
  // sticky-error latch is backend-specific (set on AMD, left clear on the NVIDIA
  // driver path), so assert the returned status and clear any latched error.
  REQUIRE(status != hipSuccess);
  (void)hipGetLastError();
}

// @asserts: hipMemcpy2DArrayToArray - rejects an invalid hipMemcpyKind and latches it into the last-error
HIP_TEST_CASE(Contract_ArrayCopyExt_Memcpy2DArrayToArray_InvalidKind_IsRejected) {
  CHECK_IMAGE_SUPPORT;

  hip::contract::ContractCleanup cleanup;
  hipArray_t src_array = nullptr;
  hipArray_t dst_array = nullptr;
  const auto desc = ByteChannelDesc();

  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMallocArray(&src_array, &desc, kWidth, kHeight));
  cleanup.Add([src_array] { (void)hipFreeArray(src_array); });
  HIP_CHECK(hipMallocArray(&dst_array, &desc, kWidth, kHeight));
  cleanup.Add([dst_array] { (void)hipFreeArray(dst_array); });

  const hipError_t status = hipMemcpy2DArrayToArray(
      dst_array, 0, 0, src_array, 0, 0, kWidth, kHeight, static_cast<hipMemcpyKind>(-1));

  REQUIRE(status != hipSuccess);
  HIP_CHECK_ERROR(hipGetLastError(), status);
  HIP_CHECK(hipGetLastError());
}

// BACKEND-DIFF: the host-to-array / array-to-host async round-trip
// (hipMemcpyHtoAAsync + hipMemcpyAtoHAsync) is exercised only on AMD; the
// positive visible round-trip is not asserted on the NVIDIA backend. Gated with
// HT_AMD (equivalent to the prior !HT_NVIDIA while there are two backends) for
// consistency with the rest of the suite's positive backend gates.
#if HT_AMD
// @asserts: hipMemcpyHtoAAsync - an async host-to-array then array-to-host round-trip is visible after stream sync
HIP_TEST_CASE(Contract_ArrayCopyExt_MemcpyHtoAAsyncThenAtoHAsync_VisibleAfterSync) {
  CHECK_IMAGE_SUPPORT;

  hip::contract::ContractCleanup cleanup;
  const auto src = MakePattern(0x89);
  std::array<uint8_t, kByteCount> dst{};
  hipArray_t array = nullptr;
  hipStream_t stream = nullptr;
  const auto desc = ByteChannelDesc();

  HIP_CHECK(hipMallocArray(&array, &desc, kByteCount, 1));
  cleanup.Add([array] { (void)hipFreeArray(array); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });

  HIP_CHECK(hipMemcpyHtoAAsync(array, 0, src.data(), src.size(), stream));
  HIP_CHECK(hipMemcpyAtoHAsync(dst.data(), array, 0, dst.size(), stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(dst == src);
}
#endif  // HT_AMD
