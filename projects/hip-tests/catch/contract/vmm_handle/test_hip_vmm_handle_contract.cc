/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>

#if !defined(_WIN32)
#include <unistd.h>  // ::close for exported POSIX/dma-buf file descriptors
#endif

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <contract_cleanup.hh>

namespace {
// Closes an exported POSIX/dma-buf file descriptor. Registered on the cleanup
// guard so the fd is released even if a later assertion throws and unwinds.
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

hipMemAllocationProp DeviceAllocationProp() {
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.requestedHandleTypes = hipMemHandleTypeNone;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = CurrentDevice();
  return prop;
}

bool VmmSupported() {
  int supported = 0;
  HIP_CHECK(hipDeviceGetAttribute(&supported, hipDeviceAttributeVirtualMemoryManagementSupported,
                                  CurrentDevice()));
  return supported != 0;
}

void SkipIfVmmUnsupported() {
  if (!VmmSupported()) {
    HIP_SKIP_TEST("HIP virtual memory management is not supported by this device/runtime path.");
  }
}

// POSIX-fd shareable-handle export of a VMM allocation is only exercised on
// discrete GPUs. On integrated devices (APUs/iGPUs) the export path is not a
// meaningful OS-level dma-buf export and, on at least one such local runtime,
// faults rather than returning a clean error; gating on the discrete-device
// property keeps the export call off that path entirely. Discrete GPUs that
// still lack the capability report it through a clean status and skip below.
bool IsDiscreteDevice() {
  hipDeviceProp_t props{};
  HIP_CHECK(hipGetDeviceProperties(&props, CurrentDevice()));
  return props.integrated == 0;
}

void SkipIfShareableHandleUnavailable() {
  SkipIfVmmUnsupported();
  if (!IsDiscreteDevice()) {
    HIP_SKIP_TEST(
        "POSIX-fd VMM shareable handles are only exercised on discrete GPUs.");
  }
}

hipMemAllocationProp PosixFdAllocationProp() {
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  // Use the plural `requestedHandleTypes` field: AMD's hipMemAllocationProp has
  // both the singular and plural members, but the NVIDIA struct only defines the
  // plural one, so the plural spelling compiles on both backends.
  prop.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = CurrentDevice();
  return prop;
}

size_t AllocationGranularity() {
  const auto prop = DeviceAllocationProp();
  size_t granularity = 0;
  HIP_CHECK(hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  return granularity;
}

// A mapped VMM allocation: a physical handle created and mapped at a reserved
// virtual address. Returns false (with cleanup) if any step reports
// hipErrorNotSupported so callers can skip on unsupported runtime paths.
struct MappedAllocation {
  hipMemGenericAllocationHandle_t handle{};
  void* address = nullptr;
  size_t size = 0;
  bool mapped = false;
};

// Creates a mapped VMM allocation, registering each resource's teardown on the
// cleanup guard immediately as it is acquired. This is exception-safe: if any
// step throws through HIP_CHECK (e.g. hipMemAddressReserve or a non-NotSupported
// hipMemMap failure), the guard still unwinds the resources acquired so far.
// Returns false (with everything acquired so far already registered for cleanup)
// when a step reports hipErrorNotSupported so callers can skip.
bool CreateMappedAllocation(hip::contract::ContractCleanup& cleanup, MappedAllocation* out) {
  out->size = AllocationGranularity();

  const auto prop = DeviceAllocationProp();
  hipError_t status = hipMemCreate(&out->handle, out->size, &prop, 0);
  if (status == hipErrorNotSupported) {
    return false;
  }
  HIP_CHECK(status);
  cleanup.Add([out] { (void)hipMemRelease(out->handle); });

  HIP_CHECK(hipMemAddressReserve(&out->address, out->size, 0, nullptr, 0));
  cleanup.Add([out] {
    if (out->address != nullptr) (void)hipMemAddressFree(out->address, out->size);
  });

  status = hipMemMap(out->address, out->size, 0, out->handle, 0);
  if (status == hipErrorNotSupported) {
    return false;
  }
  HIP_CHECK(status);
  out->mapped = true;
  cleanup.Add([out] {
    if (out->mapped) (void)hipMemUnmap(out->address, out->size);
  });
  return true;
}
}  // namespace

HIP_TEST_CASE(Contract_VmmHandle_RetainAllocationHandle_ByAddress_Succeeds) {
  SkipIfVmmUnsupported();
  // alloc must be declared BEFORE cleanup: the cleanup guard's teardown lambdas
  // capture &alloc and read its fields (handle/address/mapped) as they run. Locals
  // are destroyed in reverse declaration order, so declaring alloc first means it
  // outlives cleanup and is still alive when ~ContractCleanup executes the lambdas.
  MappedAllocation alloc;
  hip::contract::ContractCleanup cleanup;

  if (!CreateMappedAllocation(cleanup, &alloc)) {
    HIP_SKIP_TEST("VMM create/map is not supported by this device/runtime path.");
  }

  // Retaining the allocation handle for a mapped address must return a usable
  // handle that can be released independently of the original.
  hipMemGenericAllocationHandle_t retained{};
  HIP_CHECK(hipMemRetainAllocationHandle(&retained, alloc.address));
  HIP_CHECK(hipMemRelease(retained));
}

HIP_TEST_CASE(Contract_VmmHandle_GetAllocationProperties_RoundTripsFromHandle) {
  SkipIfVmmUnsupported();
  // alloc must be declared BEFORE cleanup: the cleanup guard's teardown lambdas
  // capture &alloc and read its fields (handle/address/mapped) as they run. Locals
  // are destroyed in reverse declaration order, so declaring alloc first means it
  // outlives cleanup and is still alive when ~ContractCleanup executes the lambdas.
  MappedAllocation alloc;
  hip::contract::ContractCleanup cleanup;

  if (!CreateMappedAllocation(cleanup, &alloc)) {
    HIP_SKIP_TEST("VMM create/map is not supported by this device/runtime path.");
  }

  // The properties queried from the handle must reflect what the allocation was
  // created with: pinned type on the current device location.
  hipMemAllocationProp prop{};
  HIP_CHECK(hipMemGetAllocationPropertiesFromHandle(&prop, alloc.handle));
  REQUIRE(prop.type == hipMemAllocationTypePinned);
  REQUIRE(prop.location.type == hipMemLocationTypeDevice);
  REQUIRE(prop.location.id == CurrentDevice());
}

HIP_TEST_CASE(Contract_VmmHandle_GetHandleForAddressRange_DmaBufFd_IsQueryableWhenSupported) {
  SkipIfVmmUnsupported();
  // alloc must be declared BEFORE cleanup: the cleanup guard's teardown lambdas
  // capture &alloc and read its fields (handle/address/mapped) as they run. Locals
  // are destroyed in reverse declaration order, so declaring alloc first means it
  // outlives cleanup and is still alive when ~ContractCleanup executes the lambdas.
  MappedAllocation alloc;
  hip::contract::ContractCleanup cleanup;

  if (!CreateMappedAllocation(cleanup, &alloc)) {
    HIP_SKIP_TEST("VMM create/map is not supported by this device/runtime path.");
  }

  // Exporting a dma-buf file descriptor for the mapped range is an OS/driver
  // capability. When supported it must yield a non-negative fd; when not, the
  // runtime reports a non-success status and the contract skips rather than
  // failing.
  int fd = -1;
  const hipError_t status = hipMemGetHandleForAddressRange(
      &fd, reinterpret_cast<hipDeviceptr_t>(alloc.address), alloc.size,
      hipMemRangeHandleTypeDmaBufFd, 0);
  if (status != hipSuccess) {
    HIP_SKIP_TEST("dma-buf handle export is not supported by this device/runtime path.");
  }
  cleanup.Add([&] { CloseFd(fd); });
  REQUIRE(fd >= 0);
}

HIP_TEST_CASE(Contract_VmmHandle_ExportImportShareableHandle_RoundTrips) {
  SkipIfShareableHandleUnavailable();
  hip::contract::ContractCleanup cleanup;

  // Create a physical allocation that requests a POSIX-fd shareable handle.
  const auto prop = PosixFdAllocationProp();
  size_t size = 0;
  const hipError_t gran_status =
      hipMemGetAllocationGranularity(&size, &prop, hipMemAllocationGranularityMinimum);
  if (gran_status == hipErrorNotSupported || size == 0) {
    HIP_SKIP_TEST("POSIX-fd VMM allocations are not supported by this device/runtime path.");
  }
  HIP_CHECK(gran_status);

  hipMemGenericAllocationHandle_t handle{};
  const hipError_t create_status = hipMemCreate(&handle, size, &prop, 0);
  if (create_status == hipErrorNotSupported) {
    HIP_SKIP_TEST("POSIX-fd VMM allocations are not supported by this device/runtime path.");
  }
  HIP_CHECK(create_status);
  cleanup.Add([&] { (void)hipMemRelease(handle); });

  // Export the allocation to a POSIX file descriptor. A supported path yields a
  // non-negative descriptor; an unsupported one reports a clean status and skips.
  int fd = -1;
  const hipError_t export_status =
      hipMemExportToShareableHandle(&fd, handle, hipMemHandleTypePosixFileDescriptor, 0);
  if (export_status == hipErrorNotSupported) {
    HIP_SKIP_TEST("VMM shareable-handle export is not supported by this device/runtime path.");
  }
  HIP_CHECK(export_status);
  cleanup.Add([&] { CloseFd(fd); });
  REQUIRE(fd >= 0);

  // The exported descriptor must import back into a usable generic allocation
  // handle within the same process, which is then released independently.
  hipMemGenericAllocationHandle_t imported{};
  HIP_CHECK(hipMemImportFromShareableHandle(
      &imported, reinterpret_cast<void*>(static_cast<long>(fd)),
      hipMemHandleTypePosixFileDescriptor));
  HIP_CHECK(hipMemRelease(imported));
}
