/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#ifndef LIBRARY_SRC_MEMORY_HIP_ALLOCATOR_VMM_COMMON_HPP_
#define LIBRARY_SRC_MEMORY_HIP_ALLOCATOR_VMM_COMMON_HPP_

#if HIP_VERSION >= 70000000

#include <hip/hip_runtime_api.h>
#include <hip/hip_version.h>

namespace rocshmem {

/**
 * Common VMM allocation information structure
 */
struct VMMCommonAllocationInfo {
  hipMemGenericAllocationHandle_t handle;
  size_t size;
};

/**
 * Common VMM allocation helper
 *
 * @param ptr Output pointer for allocated memory
 * @param size Size to allocate
 * @param handle_type Type of IPC handle (e.g., hipMemHandleTypePosixFileDescriptor or hipMemHandleTypeFabric)
 * @param alloc_info Output structure with allocation details
 * @return hipError_t Success or error code
 */
inline hipError_t VMMAllocCommon(void** ptr, size_t size, hipMemAllocationHandleType handle_type,
                                 VMMCommonAllocationInfo* alloc_info)
{
  hipError_t err;
  hipMemGenericAllocationHandle_t handle;
  hipMemAllocationProp prop = {};

#if HIP_VERSION < 7020000
  prop.type = hipMemAllocationTypePinned;
#else
  prop.type = hipMemAllocationTypeUncached;
#endif
  prop.location.type = hipMemLocationTypeDevice;

  // Get current device ID
  int device_id;
  err = hipGetDevice(&device_id);
  if (err != hipSuccess) return err;

  prop.location.id = device_id;
  prop.requestedHandleTypes = handle_type;

  // Get allocation granularity
  size_t granularity;
  err = hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum);
  if (err != hipSuccess) return err;

  // Round up size to granularity
  size_t alloc_size = ((size + granularity - 1) / granularity) * granularity;

  // Create memory handle
  err = hipMemCreate(&handle, alloc_size, &prop, 0);
  if (err != hipSuccess) return err;

  // Reserve address space
  void* dev_ptr = nullptr;
  err = hipMemAddressReserve(&dev_ptr, alloc_size, 0, 0, 0);
  if (err != hipSuccess) {
    (void)hipMemRelease(handle);
    return err;
  }

  // Map memory
  err = hipMemMap(dev_ptr, alloc_size, 0, handle, 0);
  if (err != hipSuccess) {
    (void)hipMemAddressFree(dev_ptr, alloc_size);
    (void)hipMemRelease(handle);
    return err;
  }

  // Set access permissions
  hipMemAccessDesc accessDesc[2];
  accessDesc[0].location.type = hipMemLocationTypeDevice;
  accessDesc[0].location.id = device_id;
  accessDesc[0].flags = hipMemAccessFlagsProtReadWrite;

  accessDesc[1].location.type = hipMemLocationTypeHost;
  accessDesc[1].location.id = 0;
  accessDesc[1].flags = hipMemAccessFlagsProtReadWrite;

  err = hipMemSetAccess(dev_ptr, alloc_size, accessDesc, 2);
  if (err != hipSuccess) {
    (void)hipMemUnmap(dev_ptr, alloc_size);
    (void)hipMemAddressFree(dev_ptr, alloc_size);
    (void)hipMemRelease(handle);
    return err;
  }

  // Success - populate allocation info and set output pointer
  alloc_info->handle = handle;
  alloc_info->size = alloc_size;
  *ptr = dev_ptr;

  return hipSuccess;
}

/**
 * Common VMM free helper
 *
 * @param ptr Pointer to free
 * @param alloc_info Allocation information from VMMAllocCommon
 * @return hipError_t Success or error code
 */
inline hipError_t VMMFreeCommon(void* ptr, const VMMCommonAllocationInfo* alloc_info)
{
  if (ptr == nullptr) {
    return hipSuccess;
  }

  hipError_t err;

  // Unmap memory
  err = hipMemUnmap(ptr, alloc_info->size);
  if (err != hipSuccess) {
    return err;
  }

  // Free address space
  err = hipMemAddressFree(ptr, alloc_info->size);
  if (err != hipSuccess) {
    return err;
  }

  // Release handle
  err = hipMemRelease(alloc_info->handle);
  if (err != hipSuccess) {
    return err;
  }

  return hipSuccess;
}

/**
 * Common VMM GetDmabufHandle helper
 *
 * @param dev_ptr Device pointer to export
 * @param size Size of the region to export
 * @param alloc_info Allocation information for the pointer
 * @param dmabuf_fd Output file descriptor for dmabuf
 * @param dmabuf_offset Output offset within the dmabuf
 * @return hipError_t Success or error code
 */
inline hipError_t VMMGetDmabufHandleCommon(void *dev_ptr, size_t size,
                                           const VMMCommonAllocationInfo* alloc_info,
                                           int *dmabuf_fd, uint64_t *dmabuf_offset)
{
  if (dev_ptr == nullptr || dmabuf_fd == nullptr || dmabuf_offset == nullptr) {
    return hipErrorInvalidValue;
  }

  if (alloc_info == nullptr) {
    return hipErrorInvalidValue;
  }

  // Verify size doesn't exceed allocation size
  if (size > alloc_info->size) {
    return hipErrorInvalidValue;
  }

  // Export the VMM handle to a shareable file descriptor (dmabuf)
  int fd;
  hipError_t err = hipMemExportToShareableHandle(&fd, alloc_info->handle,
                                                 hipMemHandleTypePosixFileDescriptor, 0);
  if (err != hipSuccess) {
    return err;
  }

  // For VMM allocations, the offset is always 0 since we're exporting the base allocation
  *dmabuf_fd = fd;
  *dmabuf_offset = 0;

  return hipSuccess;
}

}  // namespace rocshmem

#endif  // HIP_VERSION >= 70000000

#endif  // LIBRARY_SRC_MEMORY_HIP_ALLOCATOR_VMM_COMMON_HPP_
