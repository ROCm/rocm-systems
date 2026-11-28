/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
 * HIP equivalent of KFDSVMRangeTest.SplitVramRestoreBoundaryTest.
 *
 * The KFD test prefetches an SVM range to VRAM, then re-registers a smaller
 * sub-range with different flags to force the driver to split the VRAM range
 * at a boundary. It then verifies that data across the split boundary is still
 * correct once the ranges are restored/rebuilt.
 *
 * With the HIP managed-memory API the equivalent flow is:
 *   1. hipMallocManaged a large range and fill each page with a distinct value.
 *   2. hipMemPrefetchAsync the whole range to the GPU (migrate to VRAM).
 *   3. hipMemAdvise a small sub-range (split boundary) to force the driver to
 *      split the VRAM range at that boundary.
 *   4. Read back the first page and the page at the split boundary using a GPU
 *      kernel and verify the data is intact.
 */

#include <vector>

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <utils.hh>
#include <resource_guards.hh>

static std::vector<int> GetDevicesWithManagedSupport() {
  const auto device_count = HipTest::getDeviceCount();
  std::vector<int> supported_devices;
  supported_devices.reserve(device_count);
  for (int i = 0; i < device_count; ++i) {
    if (DeviceAttributesSupport(i, hipDeviceAttributeManagedMemory,
                                hipDeviceAttributeConcurrentManagedAccess)) {
      supported_devices.push_back(i);
    }
  }
  return supported_devices;
}

// Reads the byte at the start of the range and the byte at the split boundary,
// mirroring the two SDMA copies used by the KFD test.
__global__ void ReadBoundaryKernel(const uint8_t* buf, uint8_t* out, size_t split_size) {
  out[0] = buf[0];
  out[1] = buf[split_size];
}

HIP_TEST_CASE(Unit_hipMemAdvise_SplitVramRestoreBoundary) {
  const auto supported_devices = GetDevicesWithManagedSupport();
  if (supported_devices.empty()) {
    HIP_SKIP_TEST(HipTest::SkipReason::kManagedMemoryUnsupported);
  }

  constexpr size_t kBufSize = 2 * 1024 * 1024;  // 2MB range, same as KFD test.
  constexpr size_t kSplitSize = 64 * 1024;      // 64KB split boundary.
  constexpr size_t kNumPages = kBufSize / kPageSize;

  for (const auto device : supported_devices) {
    HIP_CHECK(hipSetDevice(device));

    LinearAllocGuard<uint8_t> buf(LinearAllocs::hipMallocManaged, kBufSize);

    // Fill each page with a distinct value: page i -> 0x10 + i.
    for (size_t i = 0; i < kNumPages; ++i) {
      memset(buf.ptr() + i * kPageSize, static_cast<int>(0x10 + i), kPageSize);
    }

    // Migrate the whole range to VRAM.
    HIP_CHECK(hipMemPrefetchAsync(buf.ptr(), kBufSize, device));
    HIP_CHECK(hipStreamSynchronize(nullptr));

    // Advise a sub-range to force the driver to split the VRAM range at the
    // kSplitSize boundary. SetReadMostly maps to the read-only sub-range used
    // by the KFD test.
    HIP_CHECK(hipMemAdvise(buf.ptr(), kSplitSize, hipMemAdviseSetReadMostly, device));

    LinearAllocGuard<uint8_t> result(LinearAllocs::hipMallocManaged, kPageSize);
    result.ptr()[0] = 0;
    result.ptr()[1] = 0;

    // GPU reads across the split boundary.
    ReadBoundaryKernel<<<1, 1>>>(buf.ptr(), result.ptr(), kSplitSize);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(nullptr));

    // Page 0 -> 0x10, page at kSplitSize (page 16) -> 0x20.
    const auto split_page = kSplitSize / kPageSize;
    REQUIRE(result.ptr()[0] == static_cast<uint8_t>(0x10));
    REQUIRE(result.ptr()[1] == static_cast<uint8_t>(0x10 + split_page));
  }
}

// Writes a distinct value to every byte of the range so we can confirm the GPU
// is allowed to write into the freshly (re)mapped virtual address range.
__global__ void WriteKernel(uint8_t* buf, size_t size, uint8_t value) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    buf[idx] = value;
  }
}

// Reproduces the read-only -> unmap -> remap -> write flow:
//   1. Reserve a virtual address range and back it with device memory.
//   2. Set the range to read-only (hipMemAccessFlagsProtRead).
//   3. Unmap the range.
//   4. Remap the same virtual address range with a fresh backing allocation.
//   5. Grant read/write access and write to the range.
// Expected: the write succeeds because the remap produces a fresh mapping that
// does not inherit the earlier read-only restriction.
HIP_TEST_CASE(Unit_hipMemVmm_ReadOnlyUnmapRemapWrite) {
  constexpr int kDevice = 0;

  int vmm = 0;
  HIP_CHECK(hipDeviceGetAttribute(&vmm, hipDeviceAttributeVirtualMemoryManagementSupported,
                                  kDevice));
  if (vmm == 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kVmmUnsupported);
  }

  HIP_CHECK(hipSetDevice(kDevice));

  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, kDevice));

  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;

  size_t granularity = 0;
  HIP_CHECK(hipMemGetAllocationGranularity(&granularity, &prop,
                                           hipMemAllocationGranularityRecommended));

  const size_t size = granularity;

  hipMemAccessDesc read_only_desc{};
  read_only_desc.location = prop.location;
  read_only_desc.flags = hipMemAccessFlagsProtRead;

  hipMemAccessDesc read_write_desc{};
  read_write_desc.location = prop.location;
  read_write_desc.flags = hipMemAccessFlagsProtReadWrite;

  // Reserve the virtual address range that will be reused across the remap.
  void* va = nullptr;
  HIP_CHECK(hipMemAddressReserve(&va, size, 0, nullptr, 0));

  // 1 & 2. Back the range and set it to read-only.
  hipMemGenericAllocationHandle_t first_handle{};
  HIP_CHECK(hipMemCreate(&first_handle, size, &prop, 0));
  HIP_CHECK(hipMemMap(va, size, 0, first_handle, 0));
  HIP_CHECK(hipMemSetAccess(va, size, &read_only_desc, 1));

  // 3. Unmap the read-only range and release its backing allocation.
  HIP_CHECK(hipMemUnmap(va, size));
  HIP_CHECK(hipMemRelease(first_handle));

  // 4. Remap the *same* virtual address range with a fresh allocation.
  hipMemGenericAllocationHandle_t second_handle{};
  HIP_CHECK(hipMemCreate(&second_handle, size, &prop, 0));
  HIP_CHECK(hipMemMap(va, size, 0, second_handle, 0));

  // 5. Grant read/write on the fresh mapping and write to the whole range.
  HIP_CHECK(hipMemSetAccess(va, size, &read_write_desc, 1));

  constexpr uint8_t kValue = 0xAB;
  constexpr unsigned kBlock = 256;
  const unsigned blocks = static_cast<unsigned>((size + kBlock - 1) / kBlock);
  WriteKernel<<<blocks, kBlock>>>(static_cast<uint8_t*>(va), size, kValue);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipStreamSynchronize(nullptr));

  // The write on the fresh mapping must have succeeded.
  std::vector<uint8_t> host(size);
  HIP_CHECK(hipMemcpy(host.data(), va, size, hipMemcpyDeviceToHost));
  for (size_t i = 0; i < size; ++i) {
    REQUIRE(host[i] == kValue);
  }

  HIP_CHECK(hipMemUnmap(va, size));
  HIP_CHECK(hipMemRelease(second_handle));
  HIP_CHECK(hipMemAddressFree(va, size));
}
