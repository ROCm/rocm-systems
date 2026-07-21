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

#include "hip_allocator.hpp"
#include "log.hpp"
#include "hip_allocator_vmm_common.hpp"

#if HIP_VERSION >= 70000000

#include <cstring>

namespace rocshmem {

// Static member definitions
std::map<void*, HIPAllocatorVMMFabric::VMMFabricAllocationInfo> HIPAllocatorVMMFabric::allocations_;
std::map<void*, HIPAllocatorVMMFabric::VMMFabricAllocationInfo> HIPAllocatorVMMFabric::imported_allocations_;

hipError_t HIPAllocatorVMMFabric::VMMAlloc(void** ptr, size_t size)
{
  VMMCommonAllocationInfo common_info;
  hipError_t err = VMMAllocCommon(ptr, size, hipMemHandleTypeFabric, &common_info);
  if (err != hipSuccess) {
    return err;
  }

  // Store allocator-specific metadata
  VMMFabricAllocationInfo info;
  info.handle = common_info.handle;
  info.size = common_info.size;
  allocations_[*ptr] = info;

  return hipSuccess;
}

hipError_t HIPAllocatorVMMFabric::VMMFree(void* ptr)
{
  if (ptr == nullptr) {
    return hipSuccess;
  }

  // Find allocation info
  auto it = allocations_.find(ptr);
  if (it == allocations_.end()) {
    return hipErrorInvalidValue;
  }

  VMMFabricAllocationInfo& info = it->second;

  // Convert to common allocation info
  VMMCommonAllocationInfo common_info;
  common_info.handle = info.handle;
  common_info.size = info.size;

  // Use common free function
  hipError_t err = VMMFreeCommon(ptr, &common_info);
  if (err != hipSuccess) {
    return err;
  }

  // Remove from tracking map
  allocations_.erase(it);

  return hipSuccess;
}

HIPAllocatorVMMFabric::HIPAllocatorVMMFabric()
    : HIPAllocator(VMMAlloc, VMMFree)
{
  type_ = AllocatorTypeVMMFabric;

  // Check if the device supports fabric handles
  int device_id;
  hipError_t err = hipGetDevice(&device_id);
  if (err != hipSuccess) {
    fprintf(stderr, "ROCSHMEM_ERROR: Failed to get current device for fabric handle support check: %s\n",
            hipGetErrorString(err));
    abort();
  }

  int fabric_supported = 0;
  err = hipDeviceGetAttribute(&fabric_supported,
                               hipDeviceAttributeHandleTypeFabricSupported,
                               device_id);
  if (err != hipSuccess) {
    fprintf(stderr, "ROCSHMEM_ERROR: Failed to query fabric handle support attribute: %s\n",
            hipGetErrorString(err));
    abort();
  }

  if (!fabric_supported) {
    fprintf(stderr, "ROCSHMEM_ERROR: Fabric handle type is not supported on device %d. "
            "The USE_HEAP_DEVICE_VMM_FABRIC allocator requires a GPU with fabric handle support. "
            "Please use a different memory allocator.\n",
            device_id);
    abort();
  }

  // Cache the VMM allocation granularity for this allocator.
  size_t granularity = VMMQueryGranularity(hipMemHandleTypeFabric);
  mem_granularity_ = (granularity != 0) ? granularity : 1;
}

hipError_t HIPAllocatorVMMFabric::ExportToFabricHandle(
    hipMemGenericAllocationHandle_t generic_handle, size_t size, void *handle)
{
  hipMemFabricHandle_t fabricHandle;
  hipError_t err = hipMemExportToShareableHandle(&fabricHandle, generic_handle,
                                                 hipMemHandleTypeFabric, 0);
  if (err != hipSuccess) {
    return err;
  }

  HIPIpcMemHandleFabric_t* ipc_handle = reinterpret_cast<HIPIpcMemHandleFabric_t*>(handle);
  ipc_handle->fabric_handle = fabricHandle;
  ipc_handle->size = size;
  ipc_handle->offset = 0;

  return hipSuccess;
}

hipError_t HIPAllocatorVMMFabric::GetIpcHandle(void *dev_ptr, void *handle)
{
  if (dev_ptr == nullptr || handle == nullptr) {
    return hipErrorInvalidValue;
  }

  // Find allocation info
  auto it = allocations_.find(dev_ptr);
  if (it == allocations_.end()) {
    return hipErrorInvalidValue;
  }

  return ExportToFabricHandle(it->second.handle, it->second.size, handle);
}

hipError_t HIPAllocatorVMMFabric::OpenIpcHandle(void **dev_ptr, void *handle)
{
  if (dev_ptr == nullptr || handle == nullptr) {
    return hipErrorInvalidValue;
  }

  HIPIpcMemHandleFabric_t* fabric_ipc_handle = reinterpret_cast<HIPIpcMemHandleFabric_t*>(handle);
  hipMemFabricHandle_t fabric_handle = fabric_ipc_handle->fabric_handle;
  size_t size = fabric_ipc_handle->size;
  hipError_t err;

  // Import the fabric handle
  hipMemGenericAllocationHandle_t imported_handle;
  hipMemAllocationHandleType shHandleType = hipMemHandleTypeFabric;

  err = hipMemImportFromShareableHandle(&imported_handle, (void*)&fabric_handle, shHandleType);
  if (err != hipSuccess) {
    return err;
  }

  // Reserve address space
  void *base_addr = nullptr;
  err = hipMemAddressReserve(&base_addr, size, 0, 0, 0);
  if (err != hipSuccess) {
    (void)hipMemRelease(imported_handle);
    return err;
  }

  // Map the imported handle to the reserved address
  err = hipMemMap(base_addr, size, 0, imported_handle, 0);
  if (err != hipSuccess) {
    (void)hipMemAddressFree(base_addr, size);
    (void)hipMemRelease(imported_handle);
    return err;
  }

  // Get current device ID
  int device_id;
  err = hipGetDevice(&device_id);
  if (err != hipSuccess) {
    (void)hipMemUnmap(base_addr, size);
    (void)hipMemAddressFree(base_addr, size);
    (void)hipMemRelease(imported_handle);
    return err;
  }

  // Set access permissions for device
  hipMemAccessDesc accessDesc;
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = device_id;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;

  err = hipMemSetAccess(base_addr, size, &accessDesc, 1);
  if (err != hipSuccess) {
    (void)hipMemUnmap(base_addr, size);
    (void)hipMemAddressFree(base_addr, size);
    (void)hipMemRelease(imported_handle);
    return err;
  }

  // Track the imported allocation
  imported_allocations_[base_addr] = {imported_handle, size, fabric_handle};

  // Set output pointer to the mapped address
  *dev_ptr = base_addr;

  return hipSuccess;
}

hipError_t HIPAllocatorVMMFabric::CloseIpcHandle(void *dev_ptr)
{
  if (dev_ptr == nullptr) {
    return hipSuccess;
  }

  // Find in imported allocations
  auto it = imported_allocations_.find(dev_ptr);
  if (it == imported_allocations_.end()) {
    return hipErrorInvalidValue;
  }

  VMMFabricAllocationInfo& info = it->second;

  // Unmap memory
  hipError_t err = hipMemUnmap(dev_ptr, info.size);
  if (err != hipSuccess) {
    return err;
  }

  // Free address space
  err = hipMemAddressFree(dev_ptr, info.size);
  if (err != hipSuccess) {
    return err;
  }

  // Release handle
  err = hipMemRelease(info.handle);
  if (err != hipSuccess) {
    return err;
  }

  // Remove from tracking map
  imported_allocations_.erase(it);

  return hipSuccess;
}

hipError_t HIPAllocatorVMMFabric::GetIpcHandleFromPtr(void *dev_ptr, size_t length, void *handle)
{
  if (dev_ptr == nullptr || handle == nullptr || length == 0) {
    return hipErrorInvalidValue;
  }

  // Retain the generic allocation handle backing this VMM pointer so we can
  // export it without relying on this allocator's internal tracking.
  hipMemGenericAllocationHandle_t generic_handle;
  hipError_t err = hipMemRetainAllocationHandle(&generic_handle, dev_ptr);
  if (err != hipSuccess) {
    return err;
  }

  // length is already granularity-aligned (validated at registration).
  err = ExportToFabricHandle(generic_handle, length, handle);

  /*
   * Drop our retained reference regardless of the export outcome. On success
   * the exported handle keeps the underlying memory alive for importers; a
   * release failure only leaks a reference, so warn rather than fail.
   */
  hipError_t rel_err = hipMemRelease(generic_handle);
  if (rel_err != hipSuccess) {
    LOG_WARN("hipMemRelease failed in GetIpcHandleFromPtr: %s",
             hipGetErrorString(rel_err));
  }

  return err;
}

size_t HIPAllocatorVMMFabric::GetIpcHandleSize()
{
  return sizeof(HIPIpcMemHandleFabric_t);
}

HIPIpcHandleVec* HIPAllocatorVMMFabric::AllocateIpcHandleVec(int num_elems)
{
  HIPIpcHandleFabricVec* vec = new HIPIpcHandleFabricVec();
  vec->handle.resize(num_elems);
  return vec;
}

hipError_t HIPAllocatorVMMFabric::GetDmabufHandle(void *dev_ptr, size_t size, int *dmabuf_fd, uint64_t *dmabuf_offset)
{
  // Find allocation info
  auto it = allocations_.find(dev_ptr);
  if (it == allocations_.end()) {
    return hipErrorInvalidValue;
  }

  // Convert to common allocation info
  VMMCommonAllocationInfo common_info;
  common_info.handle = it->second.handle;
  common_info.size = it->second.size;

  // Use common dmabuf export function
  return VMMGetDmabufHandleCommon(dev_ptr, size, &common_info, dmabuf_fd, dmabuf_offset);
}

}  // namespace rocshmem

#endif  // HIP_VERSION >= 70000000
