/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#include <hip_test_common.hh>

#define THREADS_PER_BLOCK 512

/**
 * Kernel to perform Square of input data.
 */
static __global__ void squareKernel(int* Buff) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int temp = Buff[i] * Buff[i];
  Buff[i] = temp;
}

/**
 * Helper function to get the granularity of the device
 */
static inline size_t GetGranularity(hipDevice_t device) {
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;
  size_t granularity = 0;
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  assert(granularity > 0);
  return granularity;
}

/**
 * Helper function to create the Physical memory of given size
 */
static inline hipMemGenericAllocationHandle_t GetPhysicalMemory(hipDevice_t device, size_t size_mem) {
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;

  hipMemGenericAllocationHandle_t handle;
  HIP_CHECK(hipMemCreate(&handle, size_mem, &prop, 0));
  return handle;
}

/**
 * Helper function to create a device memory, fills the data and
 * returns a device memory pointer
 */
static inline void* createDeviceMemoryAndFillData(int size) {
  int sizeBytes = size * sizeof(int);
  void* srcDevMem = nullptr;
  HIP_CHECK(hipMalloc(&srcDevMem, sizeBytes));
  REQUIRE(srcDevMem != nullptr);

  int* srcHostMem = nullptr;
  srcHostMem = reinterpret_cast<int*>(malloc(sizeBytes));
  REQUIRE(srcHostMem != nullptr);
  for (int i = 0; i < size; i++) {
    srcHostMem[i] = i;
  }

  HIP_CHECK(hipMemcpy(srcDevMem, srcHostMem, sizeBytes, hipMemcpyHostToDevice));

  free(srcHostMem);
  return srcDevMem;
}

/**
 * Helper function to create a virtual memory, fills the data and
 * returns a device pointer
 */
static inline hipDeviceptr_t createVirtualMemoryAndFillData(int size, int* reservedAddrSize, int device = 0) {
  size_t granularity = GetGranularity(device);
  if (granularity <= 0) {
    std::cout << "Invalid Granularity" << std::endl;
    return 0;
  }

  int* srcHostMem = reinterpret_cast<int*>(malloc(size * sizeof(int)));
  for (int i = 0; i < size; i++) {
    srcHostMem[i] = i;
  }

  size_t size_mem = ((granularity + (size * sizeof(int)) - 1) / granularity) * granularity;
  hipDeviceptr_t ptrA;
  HIP_CHECK(hipMemAddressReserve(reinterpret_cast<void**>(&ptrA), size_mem, granularity, 0, 0));
  REQUIRE(reinterpret_cast<void*>(ptrA) != nullptr);

  hipMemGenericAllocationHandle_t handle = GetPhysicalMemory(device, size_mem);

  HIP_CHECK(hipMemMap(reinterpret_cast<void*>(ptrA), size_mem, 0, handle, 0));

  hipMemAccessDesc accessDesc = {};
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = device;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;
  HIP_CHECK(hipMemSetAccess(reinterpret_cast<void*>(ptrA), size_mem, &accessDesc, 1));

  HIP_CHECK(hipMemcpy(reinterpret_cast<void*>(ptrA), srcHostMem, size * sizeof(int), hipMemcpyHostToDevice));

  free(srcHostMem);
  *reservedAddrSize = size_mem;
  return ptrA;
}

/**
 * Helper function to validate the handle from hipMemGetHandleForAddressRange
 * by extracting the data from the handle
 */
static inline bool validateHandle(int handle, int size, int device = 0) {
  hipMemGenericAllocationHandle_t imported_handle;
  HIP_CHECK(hipMemImportFromShareableHandle(&imported_handle,
            reinterpret_cast<void*>(static_cast<uintptr_t>(handle)),
            hipMemHandleTypePosixFileDescriptor));

  size_t granularity = GetGranularity(device);
  if (granularity <= 0) {
    std::cout << "Invalid Granularity" << std::endl;
    return false;
  }
  int sizeBytes = size * sizeof(int);
  size_t sizeMem = ((granularity + sizeBytes - 1) / granularity) * granularity;

  void* dstDevMem = nullptr;
  HIP_CHECK(hipMemAddressReserve(&dstDevMem, sizeMem, granularity, 0, 0));
  REQUIRE(dstDevMem != nullptr);
  HIP_CHECK(hipMemMap(dstDevMem, sizeMem, 0, imported_handle, 0));

  hipMemAccessDesc accessDesc = {};
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = device;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;
  HIP_CHECK(hipMemSetAccess(dstDevMem, sizeMem, &accessDesc, 1));

  int* dstHostMem = reinterpret_cast<int*>(malloc(sizeBytes));
  HIP_CHECK(hipMemcpy(dstHostMem, dstDevMem, sizeBytes, hipMemcpyDeviceToHost));

  for (int i = 0; i < size; i++) {
    if (dstHostMem[i] != i) {
      std::cout << "Mismatch at " << i << " : " << dstHostMem[i] << std::endl;
      free(dstHostMem);
      return false;
    }
  }

  hipLaunchKernelGGL(squareKernel, dim3(size / THREADS_PER_BLOCK), dim3(THREADS_PER_BLOCK), 0, 0,
                     static_cast<int*>(dstDevMem));
  HIP_CHECK(hipMemcpy(dstHostMem, dstDevMem, sizeBytes, hipMemcpyDeviceToHost));

  for (int i = 0; i < size; i++) {
    if (dstHostMem[i] != (i * i)) {
      std::cout << "Mismatch at " << i << " : " << dstHostMem[i] << std::endl;
      free(dstHostMem);
      return false;
    }
  }

  HIP_CHECK(hipMemUnmap(dstDevMem, sizeMem));
  HIP_CHECK(hipMemAddressFree(dstDevMem, sizeMem));
  free(dstHostMem);
  return true;
}

