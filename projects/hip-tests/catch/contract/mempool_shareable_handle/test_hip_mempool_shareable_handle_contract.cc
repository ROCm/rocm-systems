/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>

#if !defined(_WIN32)
#include <unistd.h>  // ::close for the exported POSIX file descriptor
#endif

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <contract_cleanup.hh>

namespace {
// Tiny deterministic allocation size for the pointer export/import round trip.
// The shareable-handle contract does not depend on the allocation size, so a
// small request keeps the test cheap and portable.
constexpr size_t kAllocSize = 64;

// Closes an exported POSIX file descriptor. Registered on the cleanup guard so
// the fd is released even if a later assertion throws and unwinds.
void CloseFd(int fd) {
  if (fd >= 0) {
#if !defined(_WIN32)
    (void)::close(fd);
#endif
  }
}

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

// @asserts: hipMemPoolExportToShareableHandle - a POSIX-fd pool export yields a valid fd that re-imports into a non-null pool, or skips when unsupported
HIP_TEST_CASE(Contract_MemPoolShareableHandle_ExportImportHandle_RoundTrips) {
  hip::contract::ContractCleanup cleanup;
  hipMemPool_t pool = CreatePosixFdPoolOrSkip();
  cleanup.Add([pool] { (void)hipMemPoolDestroy(pool); });

  int fd = -1;
  if (!ExportToFdOrSkip(pool, &fd)) {
    HIP_SKIP_TEST("Shareable memory pool handles are not supported by this runtime path.");
  }
  cleanup.Add([fd] { CloseFd(fd); });

  // A successful POSIX-fd export must yield a valid, non-negative descriptor.
  REQUIRE(fd >= 0);

  // The exported descriptor must round-trip back into a memory pool within the
  // same process. The descriptor is passed by value as the void* shared handle.
  hipMemPool_t imported = nullptr;
  HIP_CHECK(hipMemPoolImportFromShareableHandle(
      &imported, reinterpret_cast<void*>(static_cast<long>(fd)),
      hipMemHandleTypePosixFileDescriptor, 0));
  cleanup.Add([imported] { (void)hipMemPoolDestroy(imported); });
  REQUIRE(imported != nullptr);
}

// @asserts: hipMemPoolExportPointer - an allocation exported via hipMemPoolExportPointer imports into the peer pool as a non-null pointer, or skips when unsupported
HIP_TEST_CASE(Contract_MemPoolShareableHandle_ExportImportPointer_RoundTrips) {
  hip::contract::ContractCleanup cleanup;
  hipMemPool_t pool = CreatePosixFdPoolOrSkip();
  cleanup.Add([pool] { (void)hipMemPoolDestroy(pool); });

  int fd = -1;
  if (!ExportToFdOrSkip(pool, &fd)) {
    HIP_SKIP_TEST("Shareable memory pool handles are not supported by this runtime path.");
  }
  cleanup.Add([fd] { CloseFd(fd); });
  REQUIRE(fd >= 0);

  hipMemPool_t imported = nullptr;
  HIP_CHECK(hipMemPoolImportFromShareableHandle(
      &imported, reinterpret_cast<void*>(static_cast<long>(fd)),
      hipMemHandleTypePosixFileDescriptor, 0));
  cleanup.Add([imported] { (void)hipMemPoolDestroy(imported); });
  REQUIRE(imported != nullptr);

  // Allocations for pointer export must come from the original (exportable) pool
  // via a stream-ordered allocation; imported pools cannot create allocations.
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });
  void* dev_ptr = nullptr;
  HIP_CHECK(hipMallocFromPoolAsync(&dev_ptr, kAllocSize, pool, stream));
  // Free-and-drain on teardown: enqueue the async free, then synchronize the
  // stream so the free completes before the pool-destroy actions (registered
  // earlier, so they run after this one) tear down the pools it belongs to.
  cleanup.Add([dev_ptr, stream] {
    (void)hipFreeAsync(dev_ptr, stream);
    (void)hipStreamSynchronize(stream);
  });
  HIP_CHECK(hipStreamSynchronize(stream));
  REQUIRE(dev_ptr != nullptr);

  // Export the specific allocation as opaque pointer-export data, then import it
  // through the imported pool. The imported pointer must be non-null.
  hipMemPoolPtrExportData export_data{};
  HIP_CHECK(hipMemPoolExportPointer(&export_data, dev_ptr));

  void* imported_ptr = nullptr;
  HIP_CHECK(hipMemPoolImportPointer(&imported_ptr, imported, &export_data));
  REQUIRE(imported_ptr != nullptr);
}

// @asserts: hipMemPoolExportToShareableHandle - rejects a null output-handle pointer with a non-success status instead of crashing or succeeding
HIP_TEST_CASE(Contract_MemPoolShareableHandle_NullArgs_IsRejected) {
  hip::contract::ContractCleanup cleanup;
  hipMemPool_t pool = CreatePosixFdPoolOrSkip();
  cleanup.Add([pool] { (void)hipMemPoolDestroy(pool); });

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
}
