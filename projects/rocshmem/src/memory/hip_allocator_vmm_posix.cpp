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
#include <cerrno>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/prctl.h>
#include <sys/utsname.h>

namespace rocshmem {

// Static member definitions
std::map<void*, HIPAllocatorVMMPosixFd::VMMAllocationInfo> HIPAllocatorVMMPosixFd::allocations_;
std::map<void*, HIPAllocatorVMMPosixFd::VMMAllocationInfo> HIPAllocatorVMMPosixFd::imported_allocations_;

hipError_t HIPAllocatorVMMPosixFd::VMMAlloc(void** ptr, size_t size)
{
  VMMCommonAllocationInfo common_info;
  hipError_t err = VMMAllocCommon(ptr, size, hipMemHandleTypePosixFileDescriptor, &common_info);
  if (err != hipSuccess) {
    return err;
  }

  // Store allocator-specific metadata
  VMMAllocationInfo info;
  info.handle = common_info.handle;
  info.size = common_info.size;
  info.exported_fd = -1;  // Not yet exported
  allocations_[*ptr] = info;

  return hipSuccess;
}

hipError_t HIPAllocatorVMMPosixFd::VMMFree(void* ptr)
{
  if (ptr == nullptr) {
    return hipSuccess;
  }

  // Find allocation info
  auto it = allocations_.find(ptr);
  if (it == allocations_.end()) {
    return hipErrorInvalidValue;
  }

  VMMAllocationInfo& info = it->second;

  // Close exported fd if it was exported for IPC
  if (info.exported_fd != -1) {
    close(info.exported_fd);
  }

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

HIPAllocatorVMMPosixFd::HIPAllocatorVMMPosixFd() : HIPAllocator(VMMAlloc, VMMFree) {
  type = AllocatorTypeVMMPosix;

  // Check Linux kernel version (recommends >= 6.8)
  struct utsname kernel_info;
  if (uname(&kernel_info) == 0) {
    int major = 0, minor = 0;
    if (sscanf(kernel_info.release, "%d.%d", &major, &minor) == 2) {
      if (major < 6 || (major == 6 && minor < 8)) {
        LOG_WARN("Linux kernel version %d.%d may not work correctly with VMM POSIX allocator. "
                 "Kernel version 6.8 or higher is recommended.", major, minor);
      }
    }
  }

  // Allow other processes to trace this process for pidfd_getfd syscall
  // This avoids the need for CAP_SYS_PTRACE capability
  if (prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0) != 0) {
    LOG_WARN("Failed to set PR_SET_PTRACER: %s. "
             "IPC operations may require CAP_SYS_PTRACE capability.", strerror(errno));
  }

  // Check if the device supports VMM
  int device_id;
  hipError_t err = hipGetDevice(&device_id);
  if (err != hipSuccess) {
    LOG_ERROR_ABORT("Failed to get current device for VMM support check: %s",
                    hipGetErrorString(err));
  }

  int vmm_supported = 0;
  err = hipDeviceGetAttribute(&vmm_supported,
                               hipDeviceAttributeVirtualMemoryManagementSupported,
                               device_id);
  if (err != hipSuccess) {
    LOG_ERROR_ABORT("Failed to query VMM support attribute: %s",
                    hipGetErrorString(err));
  }

  if (!vmm_supported) {
    LOG_ERROR_ABORT("Virtual Memory Management (VMM) is not supported on device %d. "
                    "The USE_HEAP_DEVICE_VMM_POSIX allocator requires a GPU with VMM support. "
                    "Please use a different memory allocator.", device_id);
  }
}

hipError_t HIPAllocatorVMMPosixFd::GetIpcHandle(void *dev_ptr, void *handle)
{
  if (dev_ptr == nullptr || handle == nullptr) {
    return hipErrorInvalidValue;
  }

  // Find allocation info
  auto it = allocations_.find(dev_ptr);
  if (it == allocations_.end()) {
    return hipErrorInvalidValue;
  }

  HIPIpcMemHandlePosix_t* posix_handle = reinterpret_cast<HIPIpcMemHandlePosix_t*>(handle);
  hipError_t err;

  // Check if we already exported this allocation
  int fd;
  if (it->second.exported_fd != -1) {
    // Reuse existing fd
    fd = it->second.exported_fd;
  } else {
    // Export the VMM handle to a shareable file descriptor
    err = hipMemExportToShareableHandle(&fd, it->second.handle,
                                        hipMemHandleTypePosixFileDescriptor, 0);
    if (err != hipSuccess) {
      return err;
    }
    // Store the fd so we can close it later
    it->second.exported_fd = fd;
  }

  // Get current process ID and fill handle
  posix_handle->fd = static_cast<uint64_t>(fd);
  posix_handle->pid = static_cast<uint32_t>(getpid());
  posix_handle->size = it->second.size;

  return hipSuccess;
}

hipError_t HIPAllocatorVMMPosixFd::OpenIpcHandle(void **dev_ptr, void *handle)
{
  if (dev_ptr == nullptr || handle == nullptr) {
    return hipErrorInvalidValue;
  }

  HIPIpcMemHandlePosix_t* posix_handle = reinterpret_cast<HIPIpcMemHandlePosix_t*>(handle);
  int fd = static_cast<int>(posix_handle->fd);
  int pid = static_cast<int>(posix_handle->pid);
  size_t size = posix_handle->size;
  hipError_t err;

  // Open pidfd for the remote process
  int pid_fd = static_cast<int>(syscall(__NR_pidfd_open, pid, 0));
  if (pid_fd == -1) {
    int err_code = errno;
    LOG_ERROR("pidfd_open failed for pid %d: %s (errno=%d). "
              "A common reason is lacking CAP_SYS_PTRACE capability. "
              "You can resolve it e.g. with `sudo setcap 'cap_sys_ptrace=ep' <executable>`",
              pid, strerror(err_code), err_code);
    return hipErrorInvalidValue;
  }

  // Get the file descriptor from the remote process
  int open_fd = static_cast<int>(syscall(__NR_pidfd_getfd, pid_fd, fd, 0));
  if (open_fd == -1) {
    int err_code = errno;
    LOG_ERROR("pidfd_getfd failed for pid %d, fd %d: %s (errno=%d). "
              "A common reason is lacking CAP_SYS_PTRACE capability. "
              "You can resolve it e.g. with `sudo setcap 'cap_sys_ptrace=ep' <executable>`",
              pid, fd, strerror(err_code), err_code);
    close(pid_fd);
    return hipErrorInvalidValue;
  }
  close(pid_fd);

  // Import the shareable handle
  hipMemGenericAllocationHandle_t imported_handle;
  hipMemAllocationHandleType shHandleType = hipMemHandleTypePosixFileDescriptor;

#if HIP_VERSION < 7010000
  err = hipMemImportFromShareableHandle(&imported_handle, (void*)&open_fd, shHandleType);
#else
  err = hipMemImportFromShareableHandle(&imported_handle, (void*)(uintptr_t)open_fd, shHandleType);
#endif
  if (err != hipSuccess) {
    close(open_fd);
    return err;
  }

  // Reserve address space
  void *base_addr = nullptr;
  err = hipMemAddressReserve(&base_addr, size, 0, 0, 0);
  if (err != hipSuccess) {
    close(open_fd);
    (void)hipMemRelease(imported_handle);
    return err;
  }

  // Map the imported handle to the reserved address
  err = hipMemMap(base_addr, size, 0, imported_handle, 0);
  if (err != hipSuccess) {
    close(open_fd);
    (void)hipMemAddressFree(base_addr, size);
    (void)hipMemRelease(imported_handle);
    return err;
  }

  // Get current device ID
  int device_id;
  err = hipGetDevice(&device_id);
  if (err != hipSuccess) {
    close(open_fd);
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
    close(open_fd);
    (void)hipMemUnmap(base_addr, size);
    (void)hipMemAddressFree(base_addr, size);
    (void)hipMemRelease(imported_handle);
    return err;
  }

  // Track the imported allocation and keep the fd open
  // The fd will be closed in CloseIpcHandle
  imported_allocations_[base_addr] = {imported_handle, size, open_fd};

  // Set output pointer to the mapped address
  *dev_ptr = base_addr;

  return hipSuccess;
}

hipError_t HIPAllocatorVMMPosixFd::CloseIpcHandle(void *dev_ptr)
{
  if (dev_ptr == nullptr) {
    return hipSuccess;
  }

  // Find the imported allocation info
  auto it = imported_allocations_.find(dev_ptr);
  if (it == imported_allocations_.end()) {
    return hipErrorInvalidValue;
  }

  VMMAllocationInfo& info = it->second;
  hipError_t err;

  // Close the imported fd if it exists
  // This must be done before releasing the handle to properly decrement
  // the reference count on the underlying memory
  if (info.exported_fd != -1) {
    close(info.exported_fd);
  }

  // Unmap the memory
  err = hipMemUnmap(dev_ptr, info.size);
  if (err != hipSuccess) {
    return err;
  }

  // Free the address space
  err = hipMemAddressFree(dev_ptr, info.size);
  if (err != hipSuccess) {
    return err;
  }

  // Release the imported handle
  err = hipMemRelease(info.handle);
  if (err != hipSuccess) {
    return err;
  }

  // Remove from tracking map
  imported_allocations_.erase(it);

  return hipSuccess;
}

size_t HIPAllocatorVMMPosixFd::GetIpcHandleSize()
{
  return sizeof(HIPIpcMemHandlePosix_t);
}

HIPIpcHandleVec* HIPAllocatorVMMPosixFd::AllocateIpcHandleVec(int num_elems)
{
  HIPIpcHandlePosixVec* vec = new HIPIpcHandlePosixVec();
  vec->handle.resize(num_elems);
  return vec;
}

hipError_t HIPAllocatorVMMPosixFd::GetDmabufHandle(void *dev_ptr, size_t size, int *dmabuf_fd, uint64_t *dmabuf_offset)
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
