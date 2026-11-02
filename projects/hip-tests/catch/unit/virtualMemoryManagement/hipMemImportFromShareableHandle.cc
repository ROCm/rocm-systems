/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipMemExportToShareableHandle hipMemExportToShareableHandle
 * @{
 * @ingroup VirtualMemoryManagementTest
 * `hipError_t hipMemImportFromShareableHandle(hipMemGenericAllocationHandle_t *handle,
 *                                             void *osHandle,
 *                                             hipMemAllocationHandleType shHandleType)` -
 * 	Imports an allocation from a requested shareable handle type.
 */

#include <hip_test_common.hh>
#include "hip_vmm_common.hh"

#include <chrono>
#include <thread>

#define DATA_SIZE (1 << 13)
#define THREADS_PER_BLOCK 512
typedef int ShareableHandle;

/**
 Kernel to perform Square of input data.
 */
static __global__ void square_kernel(int* Buff) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int temp = Buff[i] * Buff[i];
  Buff[i] = temp;
}

/**
 * Test Description
 * ------------------------
 *    - Basic sanity test.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemImportFromShareableHandle.cc
 * Test requirements
 * ------------------------
 *    - Host specific (LINUX)
 *    - HIP_VERSION >= 6.1
 */
HIP_TEST_CASE(Unit_hipMemImportFromShareableHandle_Positive_Basic) {
  CTX_CREATE();

  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  checkVMMSupported(device);

  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;

  size_t granularity;
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));

  hipMemGenericAllocationHandle_t handle;
  HIP_CHECK(hipMemCreate(&handle, granularity * 2, &prop, 0));
  ShareableHandle shareable_handle;
  HIP_CHECK(hipMemExportToShareableHandle(&shareable_handle, handle,
                                          hipMemHandleTypePosixFileDescriptor, 0));
  hipMemGenericAllocationHandle_t imported_handle;
  HIP_CHECK(hipMemImportFromShareableHandle(&imported_handle,
            reinterpret_cast<void*>(static_cast<uintptr_t>(shareable_handle)),
            hipMemHandleTypePosixFileDescriptor));
  HIP_CHECK(hipMemRelease(handle));
  HIP_CHECK(hipMemRelease(imported_handle));

  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - Imports a host-backed allocation and checks the allocation properties
 *      reported for the imported handle. The import path derives the location
 *      from the shareable handle rather than from the importing context, so a
 *      host-backed allocation must not come back described as device memory.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemImportFromShareableHandle.cc
 * Test requirements
 * ------------------------
 *    - Host specific (LINUX)
 *    - HIP_VERSION >= 6.1
 */
HIP_TEST_CASE(Unit_hipMemImportFromShareableHandle_HostBackedAlloc) {
  CTX_CREATE();

  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  checkVMMSupported(device);
  checkHostAllocDmaBufSupported(device);

  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;
  prop.location.type = hipMemLocationTypeHost;
  prop.location.id = 0;  // Not a device index for generic host memory.

  size_t granularity = 0;
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);

  hipMemGenericAllocationHandle_t allocation_handle;
  HIP_CHECK(hipMemCreate(&allocation_handle, granularity * 2, &prop, 0));

  ShareableHandle shareable_handle;
  HIP_CHECK(hipMemExportToShareableHandle(&shareable_handle, allocation_handle,
                                          hipMemHandleTypePosixFileDescriptor, 0));

  // Import with a different device current where one exists, so location.id has to
  // describe the allocation rather than whichever device happened to be current.
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count > 1) {
    HIP_CHECK(hipSetDevice(device_count - 1));
  }

  hipMemGenericAllocationHandle_t imported_handle;
  HIP_CHECK(hipMemImportFromShareableHandle(&imported_handle,
            reinterpret_cast<void*>(static_cast<uintptr_t>(shareable_handle)),
            hipMemHandleTypePosixFileDescriptor));

  hipMemAllocationProp original_prop = {};
  hipMemAllocationProp imported_prop = {};
  HIP_CHECK(hipMemGetAllocationPropertiesFromHandle(&original_prop, allocation_handle));
  HIP_CHECK(hipMemGetAllocationPropertiesFromHandle(&imported_prop, imported_handle));

  // The exporting handle keeps describing itself as host memory.
  REQUIRE(original_prop.location.type == hipMemLocationTypeHost);

  // The imported handle has to agree with it. This is the property that the
  // import path used to fabricate, reporting every import as device memory.
  REQUIRE(imported_prop.location.type == original_prop.location.type);
  REQUIRE(imported_prop.type == hipMemAllocationTypePinned);
  REQUIRE(imported_prop.requestedHandleTypes == hipMemHandleTypePosixFileDescriptor);

  // Host memory is not device-indexed, so the id must not name a device.
  REQUIRE(imported_prop.location.id == 0);

  REQUIRE(close(shareable_handle) == 0);
  HIP_CHECK(hipMemRelease(imported_handle));
  HIP_CHECK(hipMemRelease(allocation_handle));

  if (device_count > 1) {
    HIP_CHECK(hipSetDevice(0));
  }

  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - Exports an allocation and imports it back into the exporting process, then
 *      releases only the imported handle. An imported handle borrows the backing
 *      store, so dropping it must leave the exporter's allocation and its live
 *      mapping intact. Regression test: the import path once described imported
 *      handles with the owner's teardown, so releasing the import freed memory the
 *      exporting handle still owned and the mapping became a dangling reference.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemImportFromShareableHandle.cc
 * Test requirements
 * ------------------------
 *    - Host specific (LINUX)
 *    - HIP_VERSION >= 6.1
 */
HIP_TEST_CASE(Unit_hipMemImportFromShareableHandle_SelfImportReleaseKeepsAllocation) {
  CTX_CREATE();

  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  checkVMMSupported(device);

  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;

  size_t granularity = 0;
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  const size_t size_mem = granularity * 2;

  hipMemGenericAllocationHandle_t allocation_handle;
  HIP_CHECK(hipMemCreate(&allocation_handle, size_mem, &prop, 0));

  void* ptr = nullptr;
  HIP_CHECK(hipMemAddressReserve(&ptr, size_mem, 0, nullptr, 0));
  HIP_CHECK(hipMemMap(ptr, size_mem, 0, allocation_handle, 0));

  hipMemAccessDesc access_desc = {};
  access_desc.location.type = hipMemLocationTypeDevice;
  access_desc.location.id = device;
  access_desc.flags = hipMemAccessFlagsProtReadWrite;
  HIP_CHECK(hipMemSetAccess(ptr, size_mem, &access_desc, 1));

  constexpr unsigned char kPattern = 0x5a;
  HIP_CHECK(hipMemsetD8(reinterpret_cast<hipDeviceptr_t>(ptr), kPattern, size_mem));
  HIP_CHECK(hipDeviceSynchronize());

  ShareableHandle shareable_handle;
  HIP_CHECK(hipMemExportToShareableHandle(&shareable_handle, allocation_handle,
                                          hipMemHandleTypePosixFileDescriptor, 0));
  hipMemGenericAllocationHandle_t imported_handle;
  HIP_CHECK(hipMemImportFromShareableHandle(&imported_handle,
            reinterpret_cast<void*>(static_cast<uintptr_t>(shareable_handle)),
            hipMemHandleTypePosixFileDescriptor));
  REQUIRE(close(shareable_handle) == 0);

  // The importer goes away first. The exporting handle is still the owner.
  HIP_CHECK(hipMemRelease(imported_handle));

  // The backing store must have survived, with its contents.
  std::vector<unsigned char> readback(size_mem, 0);
  HIP_CHECK(hipMemcpyDtoH(readback.data(), reinterpret_cast<hipDeviceptr_t>(ptr), size_mem));
  REQUIRE(std::all_of(readback.begin(), readback.end(),
                      [](unsigned char value) { return value == kPattern; }));

  // The mapping bookkeeping must also have survived; this used to fail with
  // hipErrorInvalidValue because the sub-buffer pointed at a freed allocation.
  HIP_CHECK(hipMemUnmap(ptr, size_mem));
  HIP_CHECK(hipMemRelease(allocation_handle));
  HIP_CHECK(hipMemAddressFree(ptr, size_mem));

  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - Repeats the export / self-import / release-import cycle while keeping one
 *      block alive across each iteration, so the freed virtual address ranges are
 *      recycled by the next reservation. Releasing an imported handle must not
 *      disturb the still-mapped block, and every block must keep its own contents.
 *      The recycling is what makes the regression deterministic: a backing store
 *      freed by the imported handle is only observed once its bookkeeping is
 *      reused, which a single-shot test can miss.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemImportFromShareableHandle.cc
 * Test requirements
 * ------------------------
 *    - Host specific (LINUX)
 *    - HIP_VERSION >= 6.1
 */
HIP_TEST_CASE(Unit_hipMemImportFromShareableHandle_SelfImportReleaseAcrossRecycledVa) {
  CTX_CREATE();

  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  checkVMMSupported(device);

  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;

  size_t granularity = 0;
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  const size_t size_mem = granularity * 2;

  hipMemAccessDesc access_desc = {};
  access_desc.location.type = hipMemLocationTypeDevice;
  access_desc.location.id = device;
  access_desc.flags = hipMemAccessFlagsProtReadWrite;

  constexpr int kIterations = 16;
  std::vector<unsigned char> readback(size_mem, 0);

  void* prev_ptr = nullptr;
  hipMemGenericAllocationHandle_t prev_allocation_handle = {};
  hipMemGenericAllocationHandle_t prev_imported_handle = {};
  bool prev_live = false;

  for (int iteration = 0; iteration < kIterations; ++iteration) {
    hipMemGenericAllocationHandle_t allocation_handle;
    HIP_CHECK(hipMemCreate(&allocation_handle, size_mem, &prop, 0));

    void* ptr = nullptr;
    HIP_CHECK(hipMemAddressReserve(&ptr, size_mem, 0, nullptr, 0));
    HIP_CHECK(hipMemMap(ptr, size_mem, 0, allocation_handle, 0));
    HIP_CHECK(hipMemSetAccess(ptr, size_mem, &access_desc, 1));

    const unsigned char pattern = static_cast<unsigned char>(iteration);
    HIP_CHECK(hipMemsetD8(reinterpret_cast<hipDeviceptr_t>(ptr), pattern, size_mem));
    HIP_CHECK(hipDeviceSynchronize());

    ShareableHandle shareable_handle;
    HIP_CHECK(hipMemExportToShareableHandle(&shareable_handle, allocation_handle,
                                            hipMemHandleTypePosixFileDescriptor, 0));
    hipMemGenericAllocationHandle_t imported_handle;
    HIP_CHECK(hipMemImportFromShareableHandle(&imported_handle,
              reinterpret_cast<void*>(static_cast<uintptr_t>(shareable_handle)),
              hipMemHandleTypePosixFileDescriptor));
    REQUIRE(close(shareable_handle) == 0);

    // Tear down the previous block, importer first, while this block stays mapped.
    if (prev_live) {
      HIP_CHECK(hipMemRelease(prev_imported_handle));
      HIP_CHECK(hipMemUnmap(prev_ptr, size_mem));
      HIP_CHECK(hipMemRelease(prev_allocation_handle));
      HIP_CHECK(hipMemAddressFree(prev_ptr, size_mem));
    }

    // Releasing the previous imported handle must not have touched this block.
    HIP_CHECK(hipMemcpyDtoH(readback.data(), reinterpret_cast<hipDeviceptr_t>(ptr), size_mem));
    REQUIRE(std::all_of(readback.begin(), readback.end(),
                        [pattern](unsigned char value) { return value == pattern; }));

    prev_ptr = ptr;
    prev_allocation_handle = allocation_handle;
    prev_imported_handle = imported_handle;
    prev_live = true;
  }

  HIP_CHECK(hipMemRelease(prev_imported_handle));
  HIP_CHECK(hipMemUnmap(prev_ptr, size_mem));
  HIP_CHECK(hipMemRelease(prev_allocation_handle));
  HIP_CHECK(hipMemAddressFree(prev_ptr, size_mem));

  CTX_DESTROY();
}

struct ImportedAllocation {
  void* ptr = nullptr;
  hipMemGenericAllocationHandle_t allocation_handle{};
  hipMemGenericAllocationHandle_t imported_handle{};
};

/**
 * Helper function to skip the test if the device does not support hipStreamWaitValue32
 */
static void CheckStreamWaitValueSupported(hipDevice_t device) {
  int wait_value_supported = 0;
  HIP_CHECK(hipDeviceGetAttribute(&wait_value_supported, hipDeviceAttributeCanUseStreamWaitValue,
                                  device));
  if (wait_value_supported == 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kStreamWaitValueUnsupported);
  }
}

/**
 * Helper function to get a copy size that is a multiple of the granularity of devices
 * [0, num_devices) and above ROC_P2P_SDMA_SIZE (1 MiB), below which AMD copies imported memory
 * with a kernel
 */
static size_t GetOverlapCopySize(int num_devices) {
  size_t granularity = 0;
  for (int dev = 0; dev < num_devices; ++dev) {
    hipMemAllocationProp prop{};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = dev;
    size_t device_granularity = 0;
    HIP_CHECK(hipMemGetAllocationGranularity(&device_granularity, &prop,
                                             hipMemAllocationGranularityMinimum));
    granularity = std::max(granularity, device_granularity);
  }
  REQUIRE(granularity > 0);
  return ((size_t(4) << 20) + granularity - 1) / granularity * granularity;
}

/**
 * Helper function to create an allocation on owner_device, export it as a POSIX fd, import it on
 * import_device and map it there with read-write access for device 0. Leaves device 0 current.
 */
static ImportedAllocation CreateImportedAllocation(int owner_device, int import_device,
                                                   size_t size) {
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = owner_device;

  ImportedAllocation allocation;
  HIP_CHECK(hipSetDevice(owner_device));
  HIP_CHECK(hipMemCreate(&allocation.allocation_handle, size, &prop, 0));
  ShareableHandle shareable_handle;
  HIP_CHECK(hipMemExportToShareableHandle(&shareable_handle, allocation.allocation_handle,
                                          hipMemHandleTypePosixFileDescriptor, 0));
  HIP_CHECK(hipSetDevice(import_device));
  HIP_CHECK(hipMemImportFromShareableHandle(
      &allocation.imported_handle,
      reinterpret_cast<void*>(static_cast<uintptr_t>(shareable_handle)),
      hipMemHandleTypePosixFileDescriptor));
  REQUIRE(close(shareable_handle) == 0);
  HIP_CHECK(hipMemAddressReserve(&allocation.ptr, size, 0, nullptr, 0));
  HIP_CHECK(hipMemMap(allocation.ptr, size, 0, allocation.imported_handle, 0));
  hipMemAccessDesc access_desc{};
  access_desc.location.type = hipMemLocationTypeDevice;
  access_desc.location.id = 0;
  access_desc.flags = hipMemAccessFlagsProtReadWrite;
  HIP_CHECK(hipMemSetAccess(allocation.ptr, size, &access_desc, 1));
  HIP_CHECK(hipSetDevice(0));
  return allocation;
}

/**
 * Helper function to unmap and release an allocation from CreateImportedAllocation
 */
static void ReleaseImportedAllocation(const ImportedAllocation& allocation, size_t size) {
  HIP_CHECK(hipMemUnmap(allocation.ptr, size));
  HIP_CHECK(hipMemRelease(allocation.imported_handle));
  HIP_CHECK(hipMemRelease(allocation.allocation_handle));
  HIP_CHECK(hipMemAddressFree(allocation.ptr, size));
}

/**
 * Helper function to check that device-0 copies from the given sources, each into its own buffer
 * on its own stream, do not serialize. Each copy is held back in turn by hipStreamWaitValue32 and
 * every other copy must complete before it is released. Checks forward progress rather than
 * timing, so it is not sensitive to machine load.
 */
static void CheckImportedCopiesOverlap(const std::vector<ImportedAllocation>& sources,
                                       size_t copy_size) {
  const int copy_count = static_cast<int>(sources.size());
  std::vector<void*> dsts(copy_count);
  std::vector<hipStream_t> streams(copy_count);
  for (int i = 0; i < copy_count; ++i) {
    HIP_CHECK(hipMemset(sources[i].ptr, 0x10 + i, copy_size));
    HIP_CHECK(hipMalloc(&dsts[i], copy_size));
    HIP_CHECK(hipStreamCreateWithFlags(&streams[i], hipStreamNonBlocking));
  }

  uint32_t* release_flag = nullptr;
  HIP_CHECK(hipHostMalloc(&release_flag, sizeof(*release_flag), hipHostMallocCoherent));
  volatile uint32_t* host_release_flag = release_flag;
  std::vector<unsigned char> result(copy_size);

  for (int held_back = 0; held_back < copy_count; ++held_back) {
    INFO("held back copy " << held_back);
    *host_release_flag = 0;
    for (void* dst : dsts) {
      HIP_CHECK(hipMemset(dst, 0, copy_size));
    }
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipStreamWaitValue32(streams[held_back], release_flag, 1, hipStreamWaitValueEq,
                                   0xFFFFFFFF));
    for (int i = 0; i < copy_count; ++i) {
      HIP_CHECK(hipMemcpyAsync(dsts[i], sources[i].ptr, copy_size, hipMemcpyDeviceToDevice,
                               streams[i]));
    }

    const auto progress_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    bool other_copies_done = false;
    while (!other_copies_done && std::chrono::steady_clock::now() < progress_deadline) {
      other_copies_done = true;
      for (int i = 0; i < copy_count; ++i) {
        if (i != held_back && hipStreamQuery(streams[i]) != hipSuccess) {
          other_copies_done = false;
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const hipError_t held_back_status = hipStreamQuery(streams[held_back]);
    // Release before asserting so a failure cannot leave the held-back stream waiting forever.
    *host_release_flag = 1;
    HIP_CHECK(hipDeviceSynchronize());

    // The held-back copy must not have run yet, otherwise the check proves nothing.
    REQUIRE(held_back_status == hipErrorNotReady);
    REQUIRE(other_copies_done);
    for (int i = 0; i < copy_count; ++i) {
      const unsigned char expected = 0x10 + i;
      HIP_CHECK(hipMemcpy(result.data(), dsts[i], copy_size, hipMemcpyDeviceToHost));
      REQUIRE(std::all_of(result.begin(), result.end(),
                          [&](unsigned char value) { return value == expected; }));
    }
  }

  HIP_CHECK(hipHostFree(release_flag));
  for (int i = 0; i < copy_count; ++i) {
    HIP_CHECK(hipStreamDestroy(streams[i]));
    HIP_CHECK(hipFree(dsts[i]));
  }
}

/**
 * Test Description
 * ------------------------
 *    - Allocations created on device 0 and imported back on device 0. Device-0 copies from them,
 *      each into its own buffer on its own stream, must not serialize.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemImportFromShareableHandle.cc
 * Test requirements
 * ------------------------
 *    - Host specific (LINUX)
 *    - Stream wait-value support (hipDeviceAttributeCanUseStreamWaitValue)
 *    - HIP_VERSION >= 6.1
 */
HIP_TEST_CASE(Unit_hipMemImportFromShareableHandle_SelfImportCopiesOverlapAcrossStreams) {
  CTX_CREATE();

  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  checkVMMSupported(device);
  CheckStreamWaitValueSupported(device);

  constexpr int kMaxStreams = 4;
  const size_t copy_size = GetOverlapCopySize(1);
  std::vector<ImportedAllocation> sources;
  for (int i = 0; i < kMaxStreams; ++i) {
    sources.push_back(CreateImportedAllocation(0, 0, copy_size));
  }
  CheckImportedCopiesOverlap(sources, copy_size);
  for (const auto& source : sources) {
    ReleaseImportedAllocation(source, copy_size);
  }

  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - One allocation created on each of devices 0 to 3 (those present), all imported on device 0,
 *      the pattern torch symmetric memory uses to map peer buffers. Device-0 copies from them,
 *      each into its own buffer on its own stream, must not serialize.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemImportFromShareableHandle.cc
 * Test requirements
 * ------------------------
 *    - Host specific (LINUX)
 *    - At least 2 devices, with peer access from device 0 to the other devices used
 *    - Stream wait-value support (hipDeviceAttributeCanUseStreamWaitValue)
 *    - HIP_VERSION >= 6.1
 */
HIP_TEST_CASE(Unit_hipMemImportFromShareableHandle_ImportFromPeersCopiesOverlapAcrossStreams) {
  CTX_CREATE();

  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }
  constexpr int kMaxStreams = 4;
  const int num_devices = std::min(device_count, kMaxStreams);
  for (int dev = 0; dev < num_devices; ++dev) {
    hipDevice_t device;
    HIP_CHECK(hipDeviceGet(&device, dev));
    checkVMMSupported(device);
    int can_access_peer = 1;
    if (dev != 0) {
      HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, 0, dev));
    }
    if (can_access_peer == 0) {
      HIP_SKIP_TEST(HipTest::SkipReason::kPeerAccessUnavailable);
    }
  }
  CheckStreamWaitValueSupported(0);

  const size_t copy_size = GetOverlapCopySize(num_devices);
  std::vector<ImportedAllocation> sources;
  for (int dev = 0; dev < num_devices; ++dev) {
    sources.push_back(CreateImportedAllocation(dev, 0, copy_size));
  }
  CheckImportedCopiesOverlap(sources, copy_size);
  for (const auto& source : sources) {
    ReleaseImportedAllocation(source, copy_size);
  }

  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - Allocations created on device 0, each imported on a different one of devices 0 to 3
 *      (those present) and mapped with access for device 0. Device-0 copies from them, each into
 *      its own buffer on its own stream, must not serialize.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemImportFromShareableHandle.cc
 * Test requirements
 * ------------------------
 *    - Host specific (LINUX)
 *    - At least 2 devices
 *    - Stream wait-value support (hipDeviceAttributeCanUseStreamWaitValue)
 *    - HIP_VERSION >= 6.1
 */
HIP_TEST_CASE(Unit_hipMemImportFromShareableHandle_ImportOnPeersCopiesOverlapAcrossStreams) {
  CTX_CREATE();

  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }
  constexpr int kMaxStreams = 4;
  const int num_devices = std::min(device_count, kMaxStreams);
  for (int dev = 0; dev < num_devices; ++dev) {
    hipDevice_t device;
    HIP_CHECK(hipDeviceGet(&device, dev));
    checkVMMSupported(device);
  }
  CheckStreamWaitValueSupported(0);

  const size_t copy_size = GetOverlapCopySize(1);
  std::vector<ImportedAllocation> sources;
  for (int dev = 0; dev < num_devices; ++dev) {
    sources.push_back(CreateImportedAllocation(0, dev, copy_size));
  }
  CheckImportedCopiesOverlap(sources, copy_size);
  for (const auto& source : sources) {
    ReleaseImportedAllocation(source, copy_size);
  }

  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - Negative parameters test.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemImportFromShareableHandle.cc
 * Test requirements
 * ------------------------
 *    - Host specific (LINUX)
 *    - HIP_VERSION >= 6.1
 */
HIP_TEST_CASE(Unit_hipMemImportFromShareableHandle_Negative_Parameters) {
  CTX_CREATE();

  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  checkVMMSupported(device);

  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;

  size_t granularity;
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));

  hipMemGenericAllocationHandle_t handle;
  HIP_CHECK(hipMemCreate(&handle, granularity * 2, &prop, 0));

  void* shareable_handle = nullptr;
  HIP_CHECK(hipMemExportToShareableHandle(&shareable_handle, handle,
                                          hipMemHandleTypePosixFileDescriptor, 0));

  hipMemGenericAllocationHandle_t imported_handle;

  SECTION("handle == nullptr") {
    HIP_CHECK_ERROR(hipMemImportFromShareableHandle(nullptr, shareable_handle,
                                                    hipMemHandleTypePosixFileDescriptor),
                    hipErrorInvalidValue);
  }

  SECTION("shareableHandle == nullptr") {
    HIP_CHECK_ERROR(hipMemImportFromShareableHandle(&imported_handle, nullptr,
                                                    hipMemHandleTypePosixFileDescriptor),
                    hipErrorInvalidValue);
  }

  HIP_CHECK(hipMemRelease(handle));
  CTX_DESTROY();
}

/**
 * Test Description
 * ------------------------
 *    - Multiprocess functionality test. Create Vmm handle in Parent
 * Process and export it to Child Process using Sockets. The Child
 * Process imports this handle via sockets and uses this handle
 * to perform VMM operations.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemImportFromShareableHandle.cc
 * Test requirements
 * ------------------------
 *    - Host specific (LINUX)
 *    - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipMemImportFromShareableHandle_MulProc_ChldUseHdl) {
  constexpr int N = DATA_SIZE;
  size_t buffer_size = N * sizeof(int);
  int fd[2], fdSig[2];
  REQUIRE(pipe(fd) == 0);
  REQUIRE(pipe(fdSig) == 0);

  auto pid = fork();
  REQUIRE(pid >= 0);

  if (pid == 0) {  // child
    REQUIRE(close(fd[1]) == 0);
    REQUIRE(close(fdSig[0]) == 0);
    CTX_CREATE();
    // Wait for parent process to create the socket.
    size_t size_mem = 0;
    REQUIRE(read(fd[0], &size_mem, sizeof(size_t)) >= 0);
    // Open Socket as client
    ipcSocketCom sockObj(false);
    hipShareableHdl shHandle;
    // Signal Parent process that Child is ready to receive msg
    int sig = 0;
    REQUIRE(write(fdSig[1], &sig, sizeof(int)) >= 0);
    // receive message from parent provess
    checkSysCallErrors(sockObj.recvShareableHdl(&shHandle));
    hipMemGenericAllocationHandle_t imported_handle;
    // import the sareable handle
    HIP_CHECK(hipMemImportFromShareableHandle(&imported_handle,
              reinterpret_cast<void*>(static_cast<uintptr_t>(shHandle)),
              hipMemHandleTypePosixFileDescriptor));
    // Allocate virtual address range
    void* ptrA;
    HIP_CHECK(hipMemAddressReserve(&ptrA, size_mem, 0, 0, 0));
    HIP_CHECK(hipMemMap(ptrA, size_mem, 0, imported_handle, 0));
    // Set access
    hipMemAccessDesc accessDesc = {};
    accessDesc.location.type = hipMemLocationTypeDevice;
    accessDesc.location.id = 0;
    accessDesc.flags = hipMemAccessFlagsProtReadWrite;
    // Make the address accessible to GPU 0
    HIP_CHECK(hipMemSetAccess(ptrA, size_mem, &accessDesc, 1));
    std::vector<int> A_h(N), B_h(N), C_h(N);
    // Initialize with data
    for (size_t idx = 0; idx < N; idx++) {
      A_h[idx] = idx;
      C_h[idx] = idx * idx;
    }
    HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(ptrA), A_h.data(), buffer_size));
    // Invoke kernel
    hipLaunchKernelGGL(square_kernel, dim3(N / THREADS_PER_BLOCK), dim3(THREADS_PER_BLOCK), 0, 0,
                       reinterpret_cast<int*>(ptrA));
    HIP_CHECK(hipMemcpyDtoH(B_h.data(), reinterpret_cast<hipDeviceptr_t>(ptrA), buffer_size));
    HIP_CHECK(hipDeviceSynchronize());
    // validate
    REQUIRE(true == std::equal(B_h.begin(), B_h.end(), C_h.data()));

    // free resources
    HIP_CHECK(hipMemUnmap(ptrA, size_mem));
    HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
    CTX_DESTROY();
    checkSysCallErrors(sockObj.closeThisSock());
    REQUIRE(close(fd[0]) == 0);
    REQUIRE(close(fdSig[1]) == 0);
    exit(0);
  } else {  // parent
    REQUIRE(close(fd[0]) == 0);
    REQUIRE(close(fdSig[1]) == 0);
    CTX_CREATE();

    hipDevice_t device;
    HIP_CHECK(hipDeviceGet(&device, 0));
    checkVMMSupported(device);
    // Set property
    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = device;
    // Set Granularity of the VMM memory
    size_t granularity;
    HIP_CHECK(
        hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
    REQUIRE(granularity > 0);
    size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;
    hipMemGenericAllocationHandle_t handle;
    HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));

    hipShareableHdl shareable_handle;
    HIP_CHECK(hipMemExportToShareableHandle(&shareable_handle, handle,
                                            hipMemHandleTypePosixFileDescriptor, 0));
    // Create the socket for communication as Server
    ipcSocketCom sockObj(true);
    // Signal child process that socket is ready
    REQUIRE(write(fd[1], &size_mem, sizeof(size_t)) >= 0);
    // Wait for the child process to receive msg
    int sig = 0;
    REQUIRE(read(fdSig[0], &sig, sizeof(int)) >= 0);
    checkSysCallErrors(sockObj.sendShareableHdl(shareable_handle, pid));
    // Wait for child process to exit.
    int status;
    REQUIRE(wait(&status) >= 0);
    REQUIRE(status == 0);
    // Free all resources
    checkSysCallErrors(sockObj.closeThisSock());
    HIP_CHECK(hipMemRelease(handle));
    CTX_DESTROY();
    REQUIRE(close(fd[1]) == 0);
    REQUIRE(close(fdSig[0]) == 0);
  }
}

/**
 * Test Description
 * ------------------------
 *    - Multiprocess functionality test. Create Vmm handle in Parent
 * Process and export it to Child Process using Sockets. The Child
 * Process imports this handle via sockets. Both Parent and Child Process
 * uses this handle to perform VMM operations.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemImportFromShareableHandle.cc
 * Test requirements
 * ------------------------
 *    - Host specific (LINUX)
 *    - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipMemImportFromShareableHandle_MulProc_ParntChldUseHdl) {
  constexpr int N = DATA_SIZE;
  size_t buffer_size = N * sizeof(int);
  int fd[2], fdSig[2];
  REQUIRE(pipe(fd) == 0);
  REQUIRE(pipe(fdSig) == 0);

  // Create data buffer
  std::vector<int> A_h(N), B_h(N), C_h(N);
  // Initialize with data
  for (size_t idx = 0; idx < N; idx++) {
    A_h[idx] = idx;
    C_h[idx] = idx * idx;
  }

  auto pid = fork();
  REQUIRE(pid >= 0);

  if (pid == 0) {  // child
    REQUIRE(close(fd[1]) == 0);
    REQUIRE(close(fdSig[0]) == 0);
    CTX_CREATE();
    // Wait for parent process to create the socket.
    size_t size_mem = 0;
    REQUIRE(read(fd[0], &size_mem, sizeof(size_t)) >= 0);

    // Open Socket as client
    ipcSocketCom sockObj(false);
    hipShareableHdl shHandle;

    // Signal Parent process that Child is ready to receive msg
    int sig = 0;
    REQUIRE(write(fdSig[1], &sig, sizeof(int)) >= 0);

    // receive message from parent provess
    checkSysCallErrors(sockObj.recvShareableHdl(&shHandle));
    hipMemGenericAllocationHandle_t imported_handle;

    // import the sareable handle
    HIP_CHECK(hipMemImportFromShareableHandle(&imported_handle,
              reinterpret_cast<void*>(static_cast<uintptr_t>(shHandle)),
              hipMemHandleTypePosixFileDescriptor));
    // Allocate virtual address range
    void* ptrA;
    HIP_CHECK(hipMemAddressReserve(&ptrA, size_mem, 0, 0, 0));
    HIP_CHECK(hipMemMap(ptrA, size_mem, 0, imported_handle, 0));
    // Set access
    hipMemAccessDesc accessDesc = {};
    accessDesc.location.type = hipMemLocationTypeDevice;
    accessDesc.location.id = 0;
    accessDesc.flags = hipMemAccessFlagsProtReadWrite;
    // Make the address accessible to GPU 0
    HIP_CHECK(hipMemSetAccess(ptrA, size_mem, &accessDesc, 1));
    HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(ptrA), A_h.data(), buffer_size));
    // Invoke kernel
    hipLaunchKernelGGL(square_kernel, dim3(N / THREADS_PER_BLOCK), dim3(THREADS_PER_BLOCK), 0, 0,
                       reinterpret_cast<int*>(ptrA));
    HIP_CHECK(hipDeviceSynchronize());

    // free resources
    HIP_CHECK(hipMemUnmap(ptrA, size_mem));
    HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
    checkSysCallErrors(sockObj.closeThisSock());
    REQUIRE(close(fd[0]) == 0);
    REQUIRE(close(fdSig[1]) == 0);
    CTX_DESTROY();
    exit(0);
  } else {  // parent
    REQUIRE(close(fd[0]) == 0);
    REQUIRE(close(fdSig[1]) == 0);
    CTX_CREATE();
    hipDevice_t device;
    HIP_CHECK(hipDeviceGet(&device, 0));
    checkVMMSupported(device);
    // Set property
    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = device;
    // Set Granularity of the VMM memory
    size_t granularity;
    HIP_CHECK(
        hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
    REQUIRE(granularity > 0);
    size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;
    hipMemGenericAllocationHandle_t handle;
    HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));

    hipShareableHdl shareable_handle;
    HIP_CHECK(hipMemExportToShareableHandle(&shareable_handle, handle,
                                            hipMemHandleTypePosixFileDescriptor, 0));

    // Allocate virtual address range
    void* ptrA;
    HIP_CHECK(hipMemAddressReserve(&ptrA, size_mem, 0, 0, 0));
    HIP_CHECK(hipMemMap(ptrA, size_mem, 0, handle, 0));
    // Set access
    hipMemAccessDesc accessDesc = {};
    accessDesc.location.type = hipMemLocationTypeDevice;
    accessDesc.location.id = device;
    accessDesc.flags = hipMemAccessFlagsProtReadWrite;
    // Make the address accessible to GPU 0
    HIP_CHECK(hipMemSetAccess(ptrA, size_mem, &accessDesc, 1));

    // Create the socket for communication as Server
    ipcSocketCom sockObj(true);
    // Signal child process that socket is ready
    REQUIRE(write(fd[1], &size_mem, sizeof(size_t)) >= 0);
    // Wait for the child process to receive msg
    int sig = 0;
    REQUIRE(read(fdSig[0], &sig, sizeof(int)) >= 0);
    checkSysCallErrors(sockObj.sendShareableHdl(shareable_handle, pid));
    // Wait for child process to exit.
    int status;
    REQUIRE(wait(&status) >= 0);
    REQUIRE(status == 0);

    // Check results of Vmm data processing in child
    HIP_CHECK(hipMemcpyDtoH(B_h.data(), reinterpret_cast<hipDeviceptr_t>(ptrA), buffer_size));
    // validate
    REQUIRE(true == std::equal(B_h.begin(), B_h.end(), C_h.data()));

    // Free all resources
    HIP_CHECK(hipMemUnmap(ptrA, size_mem));
    HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
    HIP_CHECK(hipMemRelease(handle));
    checkSysCallErrors(sockObj.closeThisSock());
    CTX_DESTROY();
    REQUIRE(close(fd[1]) == 0);
    REQUIRE(close(fdSig[0]) == 0);
  }
}

/**
 * Test Description
 * ------------------------
 *    - Multiprocess functionality test. Create Vmm handle in Parent
 * Process and export it to Grand Child Process using Sockets. The Grand
 * Child Process imports this handle via sockets. The Grand Child Process
 * uses this handle to perform VMM operations.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemImportFromShareableHandle.cc
 * Test requirements
 * ------------------------
 *    - Host specific (LINUX)
 *    - HIP_VERSION >= 6.2
 */
HIP_TEST_CASE(Unit_hipMemImportFromShareableHandle_MulProc_GrndChldUseHdl) {
  constexpr int N = DATA_SIZE;
  size_t buffer_size = N * sizeof(int);
  int fd[2], fdSig[2], fdpid[2];
  REQUIRE(pipe(fd) == 0);
  REQUIRE(pipe(fdSig) == 0);
  REQUIRE(pipe(fdpid) == 0);
  auto pid = fork();
  REQUIRE(pid >= 0);

  if (pid == 0) {  // child
    auto pid2 = fork();
    if (pid2 == 0) {  // grandchild
      REQUIRE(close(fd[1]) == 0);
      REQUIRE(close(fdSig[0]) == 0);
      // Wait for parent process to create the socket.
      size_t size_mem = 0;
      REQUIRE(read(fd[0], &size_mem, sizeof(size_t)) >= 0);
      CTX_CREATE();

      // Open Socket as client
      ipcSocketCom sockObj(false);
      hipShareableHdl shHandle;

      // Signal Parent process that Child is ready to receive msg
      int sig = 0;
      REQUIRE(write(fdSig[1], &sig, sizeof(int)) >= 0);

      // receive message from parent provess
      checkSysCallErrors(sockObj.recvShareableHdl(&shHandle));
      hipMemGenericAllocationHandle_t imported_handle;

      // import the sareable handle
      HIP_CHECK(hipMemImportFromShareableHandle(&imported_handle,
                reinterpret_cast<void*>(static_cast<uintptr_t>(shHandle)),
                hipMemHandleTypePosixFileDescriptor));
      // Allocate virtual address range
      void* ptrA;
      HIP_CHECK(hipMemAddressReserve(&ptrA, size_mem, 0, 0, 0));
      HIP_CHECK(hipMemMap(ptrA, size_mem, 0, imported_handle, 0));
      // Set access
      hipMemAccessDesc accessDesc = {};
      accessDesc.location.type = hipMemLocationTypeDevice;
      accessDesc.location.id = 0;
      accessDesc.flags = hipMemAccessFlagsProtReadWrite;
      // Make the address accessible to GPU 0
      HIP_CHECK(hipMemSetAccess(ptrA, size_mem, &accessDesc, 1));
      std::vector<int> A_h(N), B_h(N), C_h(N);
      // Initialize with data
      for (size_t idx = 0; idx < N; idx++) {
        A_h[idx] = idx;
        C_h[idx] = idx * idx;
      }
      HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(ptrA), A_h.data(), buffer_size));
      // Invoke kernel
      hipLaunchKernelGGL(square_kernel, dim3(N / THREADS_PER_BLOCK), dim3(THREADS_PER_BLOCK), 0, 0,
                         reinterpret_cast<int*>(ptrA));
      HIP_CHECK(hipMemcpyDtoH(B_h.data(), reinterpret_cast<hipDeviceptr_t>(ptrA), buffer_size));
      HIP_CHECK(hipDeviceSynchronize());
      // validate
      REQUIRE(true == std::equal(B_h.begin(), B_h.end(), C_h.data()));

      // free resources
      HIP_CHECK(hipMemUnmap(ptrA, size_mem));
      HIP_CHECK(hipMemAddressFree(ptrA, size_mem));
      CTX_DESTROY();
      checkSysCallErrors(sockObj.closeThisSock());
      REQUIRE(close(fd[0]) == 0);
      REQUIRE(close(fdSig[1]) == 0);
      exit(0);
    } else {
      int status;
      REQUIRE(close(fdpid[0]) == 0);
      REQUIRE(write(fdpid[1], &pid2, sizeof(pid2)) >= 0);
      REQUIRE(wait(&status) >= 0);
      REQUIRE(status == 0);
      REQUIRE(close(fdpid[1]) == 0);
      exit(0);
    }
  } else {  // parent
    REQUIRE(close(fd[0]) == 0);
    REQUIRE(close(fdSig[1]) == 0);
    REQUIRE(close(fdpid[1]) == 0);
    int pid_grChld = 0;
    REQUIRE(read(fdpid[0], &pid_grChld, sizeof(pid_grChld)) >= 0);
    CTX_CREATE();
    hipDevice_t device;
    HIP_CHECK(hipDeviceGet(&device, 0));
    checkVMMSupported(device);
    // Set property
    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = device;
    // Set Granularity of the VMM memory
    size_t granularity;
    HIP_CHECK(
        hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
    REQUIRE(granularity > 0);
    size_t size_mem = ((granularity + buffer_size - 1) / granularity) * granularity;
    hipMemGenericAllocationHandle_t handle;
    HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));

    hipShareableHdl shareable_handle;
    HIP_CHECK(hipMemExportToShareableHandle(&shareable_handle, handle,
                                            hipMemHandleTypePosixFileDescriptor, 0));

    // Create the socket for communication as Server
    ipcSocketCom sockObj(true);
    // Signal child process that socket is ready
    REQUIRE(write(fd[1], &size_mem, sizeof(size_t)) >= 0);
    // Wait for the child process to receive msg
    int sig = 0;
    REQUIRE(read(fdSig[0], &sig, sizeof(int)) >= 0);
    checkSysCallErrors(sockObj.sendShareableHdl(shareable_handle, pid_grChld));
    // Wait for child process to exit.
    int status;
    REQUIRE(wait(&status) >= 0);
    REQUIRE(status == 0);

    // Free all resources
    HIP_CHECK(hipMemRelease(handle));
    CTX_DESTROY();
    checkSysCallErrors(sockObj.closeThisSock());
    REQUIRE(close(fd[1]) == 0);
    REQUIRE(close(fdSig[0]) == 0);
    REQUIRE(close(fdpid[0]) == 0);
  }
}

HIP_TEST_CASE(Unit_hipMemImportFromShareableHandle_Capture) {
  CTX_CREATE();

  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  checkVMMSupported(device);

  hipMemAllocationProp allocation_prop = {};
  allocation_prop.type = hipMemAllocationTypePinned;
  allocation_prop.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;
  allocation_prop.location.type = hipMemLocationTypeDevice;
  allocation_prop.location.id = device;

  size_t granularity = 0;
  HIP_CHECK(hipMemGetAllocationGranularity(&granularity, &allocation_prop,
                                           hipMemAllocationGranularityMinimum));

  hipMemGenericAllocationHandle_t allocation_handle;
  HIP_CHECK(hipMemCreate(&allocation_handle, granularity * 2, &allocation_prop, 0));
  ShareableHandle shareable_handle;
  HIP_CHECK(hipMemExportToShareableHandle(&shareable_handle, allocation_handle,
                                          hipMemHandleTypePosixFileDescriptor, 0));
  hipMemGenericAllocationHandle_t imported_handle;
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  GENERATE_CAPTURE();
  BEGIN_CAPTURE(stream);
  HIP_CHECK(hipMemImportFromShareableHandle(&imported_handle,
            reinterpret_cast<void*>(static_cast<uintptr_t>(shareable_handle)),
            hipMemHandleTypePosixFileDescriptor));
  END_CAPTURE(stream);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipMemRelease(imported_handle));
  HIP_CHECK(hipMemRelease(allocation_handle));
  CTX_DESTROY();
}
/**
 * End doxygen group VirtualMemoryManagementTest.
 * @}
 */
