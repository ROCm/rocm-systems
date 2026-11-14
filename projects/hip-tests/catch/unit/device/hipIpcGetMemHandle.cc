/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <resource_guards.hh>

/**
 * @addtogroup hipIpcGetMemHandle hipIpcGetMemHandle
 * @{
 * @ingroup DeviceTest
 * `hipIpcGetMemHandle(hipIpcMemHandle_t* handle, void* devPtr)` -
 * Gets an interprocess memory handle for an existing device memory allocation.
 * ________________________
 * Test cases from other modules:
 *  - @ref Unit_hipIpcMemAccess_ParameterValidation
 */

/**
 * Test Description
 * ------------------------
 *  - Check that unique handles are returned in consecutive calls.
 * Test source
 * ------------------------
 *  - unit/device/hipIpcGetMemHandle.cc
 * Test requirements
 * ------------------------
 *  - Host specific (LINUX)
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipIpcGetMemHandle_Positive_Unique_Handles_Separate_Allocations) {
  LinearAllocGuard<char> alloc1(LinearAllocs::hipMalloc, 1024);
  LinearAllocGuard<char> alloc2(LinearAllocs::hipMalloc, 1024);
  hipIpcMemHandle_t handle1, handle2;
  HIP_CHECK(hipIpcGetMemHandle(&handle1, alloc1.ptr()));
  HIP_CHECK(hipIpcGetMemHandle(&handle2, alloc2.ptr()));

  CHECK(memcmp(&handle1, &handle2, sizeof(handle1)) != 0);
}

/**
 * Test Description
 * ------------------------
 *  - Test if previously freed memory will generate an invalid handle:
 *    -# When memory is freed before getting handle
 *      - Expected output: return `hipErrorInvalidValue
 * Test source
 * ------------------------
 *  - unit/device/hipIpcGetMemHandle.cc
 * Test requirements
 * ------------------------
 *  - Host specific (LINUX)
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipIpcGetMemHandle_Negative_Handle_For_Freed_Memory) {
  void* ptr;
  hipIpcMemHandle_t handle;
  HIP_CHECK(hipMalloc(&ptr, 1024));
  HIP_CHECK(hipFree(ptr));
  HIP_CHECK_ERROR(hipIpcGetMemHandle(&handle, ptr), hipErrorInvalidValue);
}

/**
 * Test Description
 * ------------------------
 *  - Test if out of bounds pointer will generate an error:
 *    -# When the memory pointer is too large
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/device/hipIpcGetMemHandle.cc
 * Test requirements
 * ------------------------
 *  - Host specific (LINUX)
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipIpcGetMemHandle_Negative_Out_Of_Bound_Pointer) {
  constexpr size_t n = 1024;
  hipIpcMemHandle_t handle;
  LinearAllocGuard<int> alloc(LinearAllocs::hipMalloc, n * sizeof(int));
  HIP_CHECK_ERROR(hipIpcGetMemHandle(&handle, reinterpret_cast<void*>(alloc.ptr() + n)),
                  hipErrorInvalidValue);
}

/**
 * A legacy IPC handle cannot describe a virtual-memory-management allocation: the handle
 * identifies a whole physical allocation made by the legacy allocator, whereas a VMM mapping is
 * an arbitrary span of reserved address space backed by one or more `hipMemCreate` handles.
 * Communication libraries probe for a VMM allocation and fall back to the legacy path on the
 * strength of this rejection, so the call must fail cleanly rather than hand back a handle that
 * cannot be opened (or crash).
 */
namespace {
bool VmmSupported(int device_ordinal) {
  int value = 0;
  HIP_CHECK(
      hipDeviceGetAttribute(&value, hipDeviceAttributeVirtualMemoryManagementSupported,
                            device_ordinal));
  return value != 0;
}

/* Owns a virtual-memory-management mapping of `size` bytes described by `prop`, granting
   `access_device` read-write access, and tears it back down on destruction. `ptr()` is nullptr
   when this configuration is not supported on the device, in which case nothing was allocated. */
class VmmMappingGuard {
 public:
  VmmMappingGuard(const hipMemAllocationProp& prop, size_t size, int access_device) {
    size_t granularity = 0;
    const hipError_t granularity_error =
        hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum);
    if (granularity_error != hipSuccess || granularity == 0) {
      (void)hipGetLastError();
      return;
    }
    const size_t padded_size = ((size + granularity - 1) / granularity) * granularity;

    if (hipMemCreate(&handle_, padded_size, &prop, 0) != hipSuccess) {
      (void)hipGetLastError();
      return;
    }
    size_ = padded_size;

    HIP_CHECK(hipMemAddressReserve(&ptr_, size_, 0, nullptr, 0));
    HIP_CHECK(hipMemMap(ptr_, size_, 0, handle_, 0));

    hipMemAccessDesc access_desc = {};
    access_desc.location.type = hipMemLocationTypeDevice;
    access_desc.location.id = access_device;
    access_desc.flags = hipMemAccessFlagsProtReadWrite;
    HIP_CHECK(hipMemSetAccess(ptr_, size_, &access_desc, 1));
  }

  VmmMappingGuard(const VmmMappingGuard&) = delete;
  VmmMappingGuard(VmmMappingGuard&&) = delete;
  VmmMappingGuard& operator=(const VmmMappingGuard&) = delete;
  VmmMappingGuard& operator=(VmmMappingGuard&&) = delete;

  ~VmmMappingGuard() {
    // No Catch macros, don't want to possibly throw in the destructor
    if (ptr_ != nullptr) {
      static_cast<void>(hipMemUnmap(ptr_, size_));
      static_cast<void>(hipMemAddressFree(ptr_, size_));
      static_cast<void>(hipMemRelease(handle_));
    }
  }

  void* ptr() const { return ptr_; }

 private:
  hipMemGenericAllocationHandle_t handle_{};
  void* ptr_ = nullptr;
  size_t size_ = 0;
};
}  // namespace

/**
 * Test Description
 * ------------------------
 *  - Checks that a virtual-memory-management mapping cannot be exported as a legacy IPC handle:
 *    -# When the pointer is a device-backed VMM mapping
 *      - Expected output: return `hipErrorInvalidValue` or `hipErrorInvalidDevicePointer`
 *    -# When the pointer is a host-backed VMM mapping
 *      - Expected output: return `hipErrorInvalidValue` or `hipErrorInvalidDevicePointer`
 *  - A later legacy allocation must still export successfully, i.e. the rejection leaves no
 *    sticky error behind.
 * Test source
 * ------------------------
 *  - unit/device/hipIpcGetMemHandle.cc
 * Test requirements
 * ------------------------
 *  - Host specific (LINUX)
 *  - Virtual memory management supported
 *  - HIP_VERSION >= 6.1
 */
HIP_TEST_CASE(Unit_hipIpcGetMemHandle_Negative_Vmm_Allocation) {
  constexpr int kDevice = 0;
  constexpr size_t kSize = 4096;
  HIP_CHECK(hipSetDevice(kDevice));
  if (!VmmSupported(kDevice)) {
    HIP_SKIP_TEST(HipTest::SkipReason::kVmmUnsupported);
  }

  SECTION("Device-backed VMM mapping") {
    hipMemAllocationProp prop{};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = kDevice;

    VmmMappingGuard mapping(prop, kSize, kDevice);
    REQUIRE(mapping.ptr() != nullptr);

    hipIpcMemHandle_t ipc_handle;
    HIP_CHECK_ERRORS(hipIpcGetMemHandle(&ipc_handle, mapping.ptr()), hipErrorInvalidValue,
                     hipErrorInvalidDevicePointer);
  }

  SECTION("Host-backed VMM mapping") {
    hipMemAllocationProp prop{};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeHost;
    prop.location.id = 0;

    VmmMappingGuard mapping(prop, kSize, kDevice);
    if (mapping.ptr() == nullptr) {
      WARN("Skipping section: host-location VMM allocations are not supported on this device.");
      return;
    }

    hipIpcMemHandle_t ipc_handle;
    HIP_CHECK_ERRORS(hipIpcGetMemHandle(&ipc_handle, mapping.ptr()), hipErrorInvalidValue,
                     hipErrorInvalidDevicePointer);
  }

  // The rejection above must not poison the next export.
  LinearAllocGuard<char> legacy_alloc(LinearAllocs::hipMalloc, kSize);
  hipIpcMemHandle_t legacy_handle;
  HIP_CHECK(hipIpcGetMemHandle(&legacy_handle, legacy_alloc.ptr()));
}

/**
 * End doxygen group DeviceTest.
 * @}
 */
