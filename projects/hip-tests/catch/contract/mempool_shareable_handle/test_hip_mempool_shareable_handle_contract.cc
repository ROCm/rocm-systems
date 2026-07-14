/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

namespace {
// Tiny deterministic allocation size for the pointer export/import round trip.
// The shareable-handle contract does not depend on the allocation size, so a
// small request keeps the test cheap and portable.
constexpr size_t kAllocSize = 64;

int CurrentDevice() {
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  return device;
}

// Properties for an IPC-capable pool: a POSIX-file-descriptor handle type is
// required for the shareable-handle export/import APIs. A pool created with
// hipMemHandleTypeNone cannot be exported, so the handle type is the load-bearing
// field of this contract.
hipMemPoolProps PosixFdPoolProps() {
  hipMemPoolProps props{};
  props.allocType = hipMemAllocationTypePinned;
  props.handleTypes = hipMemHandleTypePosixFileDescriptor;
  props.location.type = hipMemLocationTypeDevice;
  props.location.id = CurrentDevice();
  return props;
}

// Creates a POSIX-fd-capable memory pool, or skips the test when the runtime
// path does not support shareable memory-pool handles. Pool creation is the
// first capability gate: a runtime that lacks POSIX-fd mempool handles reports
// hipErrorNotSupported here, which is a capability skip rather than a contract
// failure. Any other failure is surfaced through HIP_CHECK so genuine
// regressions are not masked.
hipMemPool_t CreatePosixFdPoolOrSkip() {
  const auto props = PosixFdPoolProps();
  hipMemPool_t pool = nullptr;
  const hipError_t status = hipMemPoolCreate(&pool, &props);
  if (status == hipErrorNotSupported) {
    HIP_SKIP_TEST("Shareable memory pool handles are not supported by this runtime path.");
  }
  HIP_CHECK(status);
  REQUIRE(pool != nullptr);
  return pool;
}

// Exports a pool to a POSIX file descriptor, or skips when the export path is
// unsupported. Even when pool creation with a POSIX-fd handle type succeeds, the
// export itself may report hipErrorNotSupported on runtime paths where the
// capability is not implemented; that is a capability skip, not a failure. The
// caller passes the address of the fd storage; the export writes the descriptor
// into it.
bool ExportToFdOrSkip(hipMemPool_t pool, int* fd) {
  const hipError_t status =
      hipMemPoolExportToShareableHandle(fd, pool, hipMemHandleTypePosixFileDescriptor, 0);
  if (status == hipErrorNotSupported) {
    return false;
  }
  HIP_CHECK(status);
  return true;
}
}  // namespace

HIP_TEST_CASE(Contract_MemPoolShareableHandle_ExportImportHandle_RoundTrips) {
  hipMemPool_t pool = CreatePosixFdPoolOrSkip();

  int fd = -1;
  if (!ExportToFdOrSkip(pool, &fd)) {
    HIP_CHECK(hipMemPoolDestroy(pool));
    HIP_SKIP_TEST("Shareable memory pool handles are not supported by this runtime path.");
  }

  // A successful POSIX-fd export must yield a valid, non-negative descriptor.
  REQUIRE(fd >= 0);

  // The exported descriptor must round-trip back into a memory pool within the
  // same process. The descriptor is passed by value as the void* shared handle.
  hipMemPool_t imported = nullptr;
  HIP_CHECK(hipMemPoolImportFromShareableHandle(
      &imported, reinterpret_cast<void*>(static_cast<long>(fd)),
      hipMemHandleTypePosixFileDescriptor, 0));
  REQUIRE(imported != nullptr);

  HIP_CHECK(hipMemPoolDestroy(imported));
  HIP_CHECK(hipMemPoolDestroy(pool));
}

HIP_TEST_CASE(Contract_MemPoolShareableHandle_ExportImportPointer_RoundTrips) {
  hipMemPool_t pool = CreatePosixFdPoolOrSkip();

  int fd = -1;
  if (!ExportToFdOrSkip(pool, &fd)) {
    HIP_CHECK(hipMemPoolDestroy(pool));
    HIP_SKIP_TEST("Shareable memory pool handles are not supported by this runtime path.");
  }
  REQUIRE(fd >= 0);

  hipMemPool_t imported = nullptr;
  HIP_CHECK(hipMemPoolImportFromShareableHandle(
      &imported, reinterpret_cast<void*>(static_cast<long>(fd)),
      hipMemHandleTypePosixFileDescriptor, 0));
  REQUIRE(imported != nullptr);

  // Allocations for pointer export must come from the original (exportable) pool
  // via a stream-ordered allocation; imported pools cannot create allocations.
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  void* dev_ptr = nullptr;
  HIP_CHECK(hipMallocFromPoolAsync(&dev_ptr, kAllocSize, pool, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  REQUIRE(dev_ptr != nullptr);

  // Export the specific allocation as opaque pointer-export data, then import it
  // through the imported pool. The imported pointer must be non-null.
  hipMemPoolPtrExportData export_data{};
  HIP_CHECK(hipMemPoolExportPointer(&export_data, dev_ptr));

  void* imported_ptr = nullptr;
  HIP_CHECK(hipMemPoolImportPointer(&imported_ptr, imported, &export_data));
  REQUIRE(imported_ptr != nullptr);

  HIP_CHECK(hipFreeAsync(dev_ptr, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipMemPoolDestroy(imported));
  HIP_CHECK(hipMemPoolDestroy(pool));
}

HIP_TEST_CASE(Contract_MemPoolShareableHandle_NullArgs_IsRejected) {
  hipMemPool_t pool = CreatePosixFdPoolOrSkip();

  // A null output handle is invalid input. The export API documents
  // hipErrorInvalidValue for this, and must reject the call with a non-success
  // status rather than crash or silently succeed. The exact code is not
  // over-fit: the contract only requires a non-success result.
  const hipError_t status =
      hipMemPoolExportToShareableHandle(nullptr, pool, hipMemHandleTypePosixFileDescriptor, 0);
  REQUIRE(status != hipSuccess);

  // Clear the sticky error left by the intentionally rejected call so it does
  // not leak into later tests.
  (void)hipGetLastError();

  HIP_CHECK(hipMemPoolDestroy(pool));
}
