/*
Copyright (c) 2024-25 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * @addtogroup hipSymmetricMemory hipSymmetricMemory
 * @{
 * @ingroup VirtualMemoryManagementTest
 *
 * Test for symmetric memory pattern using VMM export/import APIs.
 * Mimics PyTorch symmetric memory: allocate on GPU0, export handle,
 * import and map on other GPUs.
 */

#include <hip_test_common.hh>
#include "hip_vmm_common.hh"

#define DATA_SIZE 1024
#define THREADS_PER_BLOCK 256

/**
 * Simple kernel to double each element
 */
static __global__ void doubleDataKernel(int* data, int N) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < N) {
    data[idx] = data[idx] * 2;
  }
}

/**
 * Kernel to add an offset to each element
 */
static __global__ void addOffsetKernel(int* data, int offset, int N) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < N) {
    data[idx] = data[idx] + offset;
  }
}

/**
 * Verify data matches expected pattern: value[i] = i * multiplier + offset
 */
static bool verifyData(void* devicePtr, size_t count, int multiplier, int offset) {
  std::vector<int> hostData(count);
  hipError_t err = hipMemcpy(hostData.data(), devicePtr, count * sizeof(int),
                             hipMemcpyDeviceToHost);
  if (err != hipSuccess) {
    return false;
  }

  for (size_t i = 0; i < count; i++) {
    int expected = static_cast<int>(i) * multiplier + offset;
    if (hostData[i] != expected) {
      fprintf(stderr, "Mismatch at index %zu: expected %d, got %d\n",
              i, expected, hostData[i]);
      return false;
    }
  }
  return true;
}

/**
 * Initialize data with pattern: value[i] = i * multiplier
 */
static void initializeData(void* devicePtr, size_t count, int multiplier) {
  std::vector<int> hostData(count);
  for (size_t i = 0; i < count; i++) {
    hostData[i] = static_cast<int>(i) * multiplier;
  }
  HIP_CHECK(hipMemcpy(devicePtr, hostData.data(), count * sizeof(int),
                      hipMemcpyHostToDevice));
}

/**
 * Test Description
 * ------------------------
 *    - Symmetric memory test using export/import:
 *      1. Reserve virtual address space
 *      2. Allocate physical memory on GPU 0
 *      3. Map on GPU 0
 *      4. Export to shareable handle
 *      5. On other GPUs: import handle and map to their own VA
 * ------------------------
 *    - unit/virtualMemoryManagement/hipSymmetricMemory.cc
 * Test requirements
 * ------------------------
 *    - Multi-GPU system
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipSymmetricMemory_ExportImport") {
  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  if (deviceCount < 2) {
    HipTest::HIP_SKIP_TEST("Need at least 2 GPUs. Skipping..");
    return;
  }

  // Check VMM support on all devices
  for (int i = 0; i < deviceCount; i++) {
    hipDevice_t device;
    HIP_CHECK(hipDeviceGet(&device, i));
    checkVMMSupported(device);
  }

  constexpr size_t dataBytes = DATA_SIZE * sizeof(int);

  // === GPU 0: Create and map physical memory ===
  HIP_CHECK(hipSetDevice(0));

  hipDevice_t device0;
  HIP_CHECK(hipDeviceGet(&device0, 0));

  // Setup allocation properties with shareable handle type
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device0;
  prop.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;

  // Get granularity
  size_t granularity = 0;
  HIP_CHECK(hipMemGetAllocationGranularity(&granularity, &prop,
                                           hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);

  size_t allocSize = ((dataBytes + granularity - 1) / granularity) * granularity;

  // Step 1: Reserve virtual address space on GPU 0
  void* va0 = nullptr;
  HIP_CHECK(hipMemAddressReserve(&va0, allocSize, 0, 0, 0));

  // Step 2: Allocate physical memory on GPU 0
  hipMemGenericAllocationHandle_t memHandle;
  HIP_CHECK(hipMemCreate(&memHandle, allocSize, &prop, 0));

  // Step 3: Map physical memory to VA on GPU 0
  HIP_CHECK(hipMemMap(va0, allocSize, 0, memHandle, 0));

  // Set access for GPU 0
  hipMemAccessDesc accessDesc0{};
  accessDesc0.location.type = hipMemLocationTypeDevice;
  accessDesc0.location.id = device0;
  accessDesc0.flags = hipMemAccessFlagsProtReadWrite;
  HIP_CHECK(hipMemSetAccess(va0, allocSize, &accessDesc0, 1));

  // Write test data from GPU 0
  std::vector<int> hostData(DATA_SIZE);
  for (int i = 0; i < DATA_SIZE; i++) {
    hostData[i] = i * 10;
  }
  HIP_CHECK(hipMemcpy(va0, hostData.data(), dataBytes, hipMemcpyHostToDevice));

  // Step 4: Export to shareable handle
  hipShareableHdl shareableHandle;
  HIP_CHECK(hipMemExportToShareableHandle(&shareableHandle, memHandle,
                                          hipMemHandleTypePosixFileDescriptor, 0));

  // === Other GPUs: Import and map ===
  std::vector<void*> vaOther(deviceCount - 1);
  std::vector<hipMemGenericAllocationHandle_t> importedHandles(deviceCount - 1);

  for (int gpuIdx = 1; gpuIdx < deviceCount; gpuIdx++) {
    HIP_CHECK(hipSetDevice(gpuIdx));

    hipDevice_t deviceN;
    HIP_CHECK(hipDeviceGet(&deviceN, gpuIdx));

    // Step 5a: Import the shareable handle on this GPU
    HIP_CHECK(hipMemImportFromShareableHandle(&importedHandles[gpuIdx - 1],
               reinterpret_cast<void*>(static_cast<uintptr_t>(shareableHandle)),
               hipMemHandleTypePosixFileDescriptor));

    // Step 5b: Reserve VA on this GPU
    HIP_CHECK(hipMemAddressReserve(&vaOther[gpuIdx - 1], allocSize, 0, 0, 0));

    // Step 5c: Map the imported physical memory to this GPU's VA
    HIP_CHECK(hipMemMap(vaOther[gpuIdx - 1], allocSize, 0,
                        importedHandles[gpuIdx - 1], 0));

    // Step 5d: Set access for this GPU
    hipMemAccessDesc accessDescN{};
    accessDescN.location.type = hipMemLocationTypeDevice;
    accessDescN.location.id = deviceN;
    accessDescN.flags = hipMemAccessFlagsProtReadWrite;
    HIP_CHECK(hipMemSetAccess(vaOther[gpuIdx - 1], allocSize, &accessDescN, 1));

    // Verify: Read back data on this GPU
    std::vector<int> verifyData(DATA_SIZE, -1);
    HIP_CHECK(hipMemcpy(verifyData.data(), vaOther[gpuIdx - 1], dataBytes,
                        hipMemcpyDeviceToHost));

    // Check data matches what GPU 0 wrote
    bool match = true;
    for (int i = 0; i < DATA_SIZE; i++) {
      if (verifyData[i] != i * 10) {
        match = false;
        break;
      }
    }
    REQUIRE(match);
  }

  // === Cleanup ===
  for (int gpuIdx = 1; gpuIdx < deviceCount; gpuIdx++) {
    HIP_CHECK(hipSetDevice(gpuIdx));
    HIP_CHECK(hipMemUnmap(vaOther[gpuIdx - 1], allocSize));
    HIP_CHECK(hipMemAddressFree(vaOther[gpuIdx - 1], allocSize));
    HIP_CHECK(hipMemRelease(importedHandles[gpuIdx - 1]));
  }

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipMemUnmap(va0, allocSize));
  HIP_CHECK(hipMemAddressFree(va0, allocSize));
  HIP_CHECK(hipMemRelease(memHandle));
}

/**
 * Test Description
 * ------------------------
 *    - Symmetric memory test with kernel execution:
 *      1. Setup symmetric memory (same as ExportImport test)
 *      2. GPU 0 runs kernel to modify data
 *      3. Other GPUs verify they see the modification
 *      4. Other GPUs run kernels to modify data
 *      5. GPU 0 verifies it sees modifications from other GPUs
 * ------------------------
 *    - unit/virtualMemoryManagement/hipSymmetricMemory.cc
 * Test requirements
 * ------------------------
 *    - Multi-GPU system
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipSymmetricMemory_KernelAccess") {
  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  if (deviceCount < 2) {
    HipTest::HIP_SKIP_TEST("Need at least 2 GPUs. Skipping..");
    return;
  }

  for (int i = 0; i < deviceCount; i++) {
    hipDevice_t device;
    HIP_CHECK(hipDeviceGet(&device, i));
    checkVMMSupported(device);
  }

  constexpr size_t dataBytes = DATA_SIZE * sizeof(int);

  // === GPU 0: Setup ===
  HIP_CHECK(hipSetDevice(0));

  hipDevice_t device0;
  HIP_CHECK(hipDeviceGet(&device0, 0));

  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device0;
  prop.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;

  size_t granularity = 0;
  HIP_CHECK(hipMemGetAllocationGranularity(&granularity, &prop,
                                           hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);

  size_t allocSize = ((dataBytes + granularity - 1) / granularity) * granularity;

  void* va0 = nullptr;
  HIP_CHECK(hipMemAddressReserve(&va0, allocSize, 0, 0, 0));

  hipMemGenericAllocationHandle_t memHandle;
  HIP_CHECK(hipMemCreate(&memHandle, allocSize, &prop, 0));
  HIP_CHECK(hipMemMap(va0, allocSize, 0, memHandle, 0));

  hipMemAccessDesc accessDesc0{};
  accessDesc0.location.type = hipMemLocationTypeDevice;
  accessDesc0.location.id = device0;
  accessDesc0.flags = hipMemAccessFlagsProtReadWrite;
  HIP_CHECK(hipMemSetAccess(va0, allocSize, &accessDesc0, 1));

  // Initialize data: value[i] = i * 10
  initializeData(va0, DATA_SIZE, 10);

  // GPU 0: Run kernel to double data -> value[i] = i * 20
  dim3 blocks((DATA_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
  dim3 threads(THREADS_PER_BLOCK);
  hipLaunchKernelGGL(doubleDataKernel, blocks, threads, 0, 0,
                     static_cast<int*>(va0), DATA_SIZE);
  HIP_CHECK(hipDeviceSynchronize());

  // Verify on GPU 0
  REQUIRE(verifyData(va0, DATA_SIZE, 20, 0));

  // Export handle
  hipShareableHdl shareableHandle;
  HIP_CHECK(hipMemExportToShareableHandle(&shareableHandle, memHandle,
                                          hipMemHandleTypePosixFileDescriptor, 0));

  // === Other GPUs: Import, verify, and modify ===
  std::vector<void*> vaOther(deviceCount - 1);
  std::vector<hipMemGenericAllocationHandle_t> importedHandles(deviceCount - 1);

  for (int gpuIdx = 1; gpuIdx < deviceCount; gpuIdx++) {
    HIP_CHECK(hipSetDevice(gpuIdx));

    hipDevice_t deviceN;
    HIP_CHECK(hipDeviceGet(&deviceN, gpuIdx));

    HIP_CHECK(hipMemImportFromShareableHandle(&importedHandles[gpuIdx - 1],
               reinterpret_cast<void*>(static_cast<uintptr_t>(shareableHandle)),
               hipMemHandleTypePosixFileDescriptor));

    HIP_CHECK(hipMemAddressReserve(&vaOther[gpuIdx - 1], allocSize, 0, 0, 0));
    HIP_CHECK(hipMemMap(vaOther[gpuIdx - 1], allocSize, 0,
                        importedHandles[gpuIdx - 1], 0));

    hipMemAccessDesc accessDescN{};
    accessDescN.location.type = hipMemLocationTypeDevice;
    accessDescN.location.id = deviceN;
    accessDescN.flags = hipMemAccessFlagsProtReadWrite;
    HIP_CHECK(hipMemSetAccess(vaOther[gpuIdx - 1], allocSize, &accessDescN, 1));

    // Verify this GPU sees data from GPU 0's kernel: value[i] = i * 20
    REQUIRE(verifyData(vaOther[gpuIdx - 1], DATA_SIZE, 20, 0));

    // Run kernel to add offset: value[i] = i * 20 + gpuIdx * 100
    hipLaunchKernelGGL(addOffsetKernel, blocks, threads, 0, 0,
                       static_cast<int*>(vaOther[gpuIdx - 1]), gpuIdx * 100, DATA_SIZE);
    HIP_CHECK(hipDeviceSynchronize());
  }

  // GPU 0: Verify it sees modification from last GPU
  HIP_CHECK(hipSetDevice(0));
  int expectedOffset = (deviceCount - 1) * 100;
  REQUIRE(verifyData(va0, DATA_SIZE, 20, expectedOffset));

  // === Cleanup ===
  for (int gpuIdx = 1; gpuIdx < deviceCount; gpuIdx++) {
    HIP_CHECK(hipSetDevice(gpuIdx));
    HIP_CHECK(hipMemUnmap(vaOther[gpuIdx - 1], allocSize));
    HIP_CHECK(hipMemAddressFree(vaOther[gpuIdx - 1], allocSize));
    HIP_CHECK(hipMemRelease(importedHandles[gpuIdx - 1]));
  }

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipMemUnmap(va0, allocSize));
  HIP_CHECK(hipMemAddressFree(va0, allocSize));
  HIP_CHECK(hipMemRelease(memHandle));
}

/**
 * End doxygen group VirtualMemoryManagementTest.
 * @}
 */
