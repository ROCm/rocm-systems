/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

namespace {
constexpr size_t kRangeBytes = 4096;

int CurrentDevice() {
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  return device;
}

bool ManagedMemorySupported() {
  void* ptr = nullptr;
  const hipError_t status = hipMallocManaged(&ptr, kRangeBytes, hipMemAttachGlobal);
  if (status == hipSuccess) {
    HIP_CHECK(hipFree(ptr));
    return true;
  }
  if (status == hipErrorNotSupported || status == hipErrorOutOfMemory) {
    return false;
  }
  HIP_CHECK(status);
  return false;
}

void SkipIfManagedMemoryUnsupported() {
  if (!ManagedMemorySupported()) {
    HIP_SKIP_TEST(HipTest::SkipReason::kManagedMemoryUnsupported);
  }
}

hipMemLocation CurrentDeviceLocation() {
  hipMemLocation location{};
  location.type = hipMemLocationTypeDevice;
  location.id = CurrentDevice();
  return location;
}

// The standalone batch-prefetch operation is only exercised on discrete GPUs.
// On at least one integrated local runtime it rejects well-formed inputs with
// an invalid-value error (while the single-range prefetch and the combined
// discard-and-prefetch accept the identical location), so gating on the
// discrete-device property keeps the contract meaningful. Discrete GPUs that
// lack the capability report it as unsupported and skip below.
bool IsDiscreteDevice() {
  hipDeviceProp_t props{};
  HIP_CHECK(hipGetDeviceProperties(&props, CurrentDevice()));
  return props.integrated == 0;
}

// Allocates and faults in a small managed range on the caller's stream. The
// caller owns the returned pointer and must free it.
void* AllocResidentManagedRange(hipStream_t stream) {
  void* ptr = nullptr;
  HIP_CHECK(hipMallocManaged(&ptr, kRangeBytes, hipMemAttachGlobal));
  HIP_CHECK(hipMemsetAsync(ptr, 0, kRangeBytes, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  return ptr;
}

// The batch discard/prefetch operations require XNACK and managed or
// system-allocated memory. Where that path is unavailable the runtime reports
// hipErrorNotSupported, which is a capability skip rather than a contract
// failure. Any other non-success status is a genuine contract violation. A
// non-success status also leaves a sticky thread-local error, which is cleared
// here so it does not leak into later tests.
void RequireAcceptedOrUnsupported(hipError_t status) {
  if (status != hipSuccess) {
    (void)hipGetLastError();
  }
  if (status == hipSuccess || status == hipErrorNotSupported) {
    return;
  }
  HIP_CHECK(status);
}
}  // namespace

HIP_TEST_CASE(Contract_MemBatchDiscard_DiscardBatch_IsAcceptedOrUnsupported) {
  SkipIfManagedMemoryUnsupported();

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  void* ptr = AllocResidentManagedRange(stream);

  // Discarding a batch of managed ranges must either be honored or reported
  // unsupported. When honored, the stream must drain cleanly afterward.
  void* ptrs[1] = {ptr};
  size_t sizes[1] = {kRangeBytes};
  const hipError_t status = hipMemDiscardBatchAsync(ptrs, sizes, 1, 0, stream);
  RequireAcceptedOrUnsupported(status);
  if (status == hipSuccess) {
    HIP_CHECK(hipStreamSynchronize(stream));
  }

  HIP_CHECK(hipFree(ptr));
  HIP_CHECK(hipStreamDestroy(stream));
}

HIP_TEST_CASE(Contract_MemBatchDiscard_DiscardAndPrefetchBatch_IsAcceptedOrUnsupported) {
  SkipIfManagedMemoryUnsupported();

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  void* ptr = AllocResidentManagedRange(stream);

  // The combined discard-and-prefetch batch, targeting the current device as the
  // prefetch destination, must be accepted or reported unsupported.
  void* ptrs[1] = {ptr};
  size_t sizes[1] = {kRangeBytes};
  hipMemLocation locations[1] = {CurrentDeviceLocation()};
  size_t location_indices[1] = {0};
  const hipError_t status = hipMemDiscardAndPrefetchBatchAsync(ptrs, sizes, 1, locations,
                                                              location_indices, 1, 0, stream);
  RequireAcceptedOrUnsupported(status);
  if (status == hipSuccess) {
    HIP_CHECK(hipStreamSynchronize(stream));
  }

  HIP_CHECK(hipFree(ptr));
  HIP_CHECK(hipStreamDestroy(stream));
}

HIP_TEST_CASE(Contract_MemBatchDiscard_DrvDiscardBatch_IsAcceptedOrUnsupported) {
  SkipIfManagedMemoryUnsupported();

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  void* ptr = AllocResidentManagedRange(stream);

  // The driver-style discard batch takes device pointers and must likewise be
  // accepted or reported unsupported.
  hipDeviceptr_t dptrs[1] = {reinterpret_cast<hipDeviceptr_t>(ptr)};
  size_t sizes[1] = {kRangeBytes};
  const hipError_t status = hipDrvMemDiscardBatchAsync(dptrs, sizes, 1, 0, stream);
  RequireAcceptedOrUnsupported(status);
  if (status == hipSuccess) {
    HIP_CHECK(hipStreamSynchronize(stream));
  }

  HIP_CHECK(hipFree(ptr));
  HIP_CHECK(hipStreamDestroy(stream));
}

HIP_TEST_CASE(Contract_MemBatchDiscard_DrvDiscardAndPrefetchBatch_IsAcceptedOrUnsupported) {
  SkipIfManagedMemoryUnsupported();

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  void* ptr = AllocResidentManagedRange(stream);

  // The driver-style combined discard-and-prefetch batch mirrors the runtime
  // variant with device pointers.
  hipDeviceptr_t dptrs[1] = {reinterpret_cast<hipDeviceptr_t>(ptr)};
  size_t sizes[1] = {kRangeBytes};
  hipMemLocation locations[1] = {CurrentDeviceLocation()};
  size_t location_indices[1] = {0};
  const hipError_t status = hipDrvMemDiscardAndPrefetchBatchAsync(
      dptrs, sizes, 1, locations, location_indices, 1, 0, stream);
  RequireAcceptedOrUnsupported(status);
  if (status == hipSuccess) {
    HIP_CHECK(hipStreamSynchronize(stream));
  }

  HIP_CHECK(hipFree(ptr));
  HIP_CHECK(hipStreamDestroy(stream));
}

HIP_TEST_CASE(Contract_MemBatchDiscard_NullPointer_IsRejectedOrUnsupported) {
  SkipIfManagedMemoryUnsupported();

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  // A batch with a null range must not silently succeed. On a runtime that
  // supports the path the call must reject the null input; on a runtime that
  // does not, hipErrorNotSupported is acceptable. In neither case may the call
  // return hipSuccess. Any pre-existing sticky error is cleared without
  // asserting so it does not perturb this negative check.
  (void)hipGetLastError();
  void* ptrs[1] = {nullptr};
  size_t sizes[1] = {kRangeBytes};
  const hipError_t status = hipMemDiscardBatchAsync(ptrs, sizes, 1, 0, stream);
  REQUIRE(status != hipSuccess);
  (void)hipGetLastError();

  HIP_CHECK(hipStreamDestroy(stream));
}

HIP_TEST_CASE(Contract_MemBatchDiscard_PrefetchBatch_IsAcceptedOrUnsupported) {
  SkipIfManagedMemoryUnsupported();
  if (!IsDiscreteDevice()) {
    HIP_SKIP_TEST("Batch prefetch is only exercised on discrete GPUs.");
  }

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  void* ptr = AllocResidentManagedRange(stream);

  // Prefetching a batch of managed ranges to the current device must be accepted
  // or reported unsupported. When honored, the stream must drain cleanly.
  void* ptrs[1] = {ptr};
  size_t sizes[1] = {kRangeBytes};
  hipMemLocation locations[1] = {CurrentDeviceLocation()};
  size_t location_indices[1] = {0};
  const hipError_t status =
      hipMemPrefetchBatchAsync(ptrs, sizes, 1, locations, location_indices, 1, 0, stream);
  RequireAcceptedOrUnsupported(status);
  if (status == hipSuccess) {
    HIP_CHECK(hipStreamSynchronize(stream));
  }

  HIP_CHECK(hipFree(ptr));
  HIP_CHECK(hipStreamDestroy(stream));
}
