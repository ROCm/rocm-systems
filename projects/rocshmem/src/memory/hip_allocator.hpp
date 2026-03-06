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

#ifndef LIBRARY_SRC_MEMORY_HIP_ALLOCATOR_HPP_
#define LIBRARY_SRC_MEMORY_HIP_ALLOCATOR_HPP_

/**
 * @file hip_allocator.hpp
 *
 * @brief Contains HIP wrapper class for memory allocator
 */

#include "rocshmem/rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "memory_allocator.hpp"

#include <hip/hip_runtime_api.h>
#include <hip/hip_version.h>

#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <limits>
#include <map>
#include <vector>
#include <unistd.h>
#include <sys/syscall.h>

namespace rocshmem {

enum HIPIpcHandleType {
  HandleTypeLegacy = 0,
  HandleTypePosix,
  HandleTypeFabric,
  HandleTypeLast
};

enum HIPAllocatorType {
  AllocatorTypeCoarsegrained = 0,
  AllocatorTypeFinegrained,
  AllocatorTypeUncached,
  AllocatorTypeVMM,
  AllocatorTypeLast
};

#if HIP_VERSION >= 70000000
struct hipIpcMemHandlePosix_t {
  uint64_t fd;
  uint32_t pid;
  size_t size;
};
#endif

class HIPIpcHandleVec {
public:
  virtual ~HIPIpcHandleVec() = default;

  virtual HIPIpcHandleType GetIpcHandleType() = 0;
  virtual void* GetHandleVecElem(int elem) = 0;
};

class HIPIpcHandleLegacyVec : public HIPIpcHandleVec {
public:
  friend class HIPAllocator;

  HIPIpcHandleType GetIpcHandleType() { return HandleTypeLegacy; }

  void* GetHandleVecElem(int elem)
  {
    return reinterpret_cast<void*> (&this->handle[elem]);
  }

protected:
  std::vector<hipIpcMemHandle_t> handle;

};

#if HIP_VERSION >= 70000000
class HIPIpcHandlePosixVec : public HIPIpcHandleVec {
public:
  friend class HIPAllocatorVMMPosixFd;

  HIPIpcHandleType GetIpcHandleType() { return HandleTypePosix; }

  void* GetHandleVecElem(int elem)
  {
    return reinterpret_cast<void*> (&this->handle[elem]);
  }

protected:
  std::vector<hipIpcMemHandlePosix_t> handle;

};
#endif

class HIPAllocator : public MemoryAllocator {
 public:

  HIPAllocator() : MemoryAllocator(hipMalloc, hipFree) {}

  HIPAllocator(hipError_t (*hip_alloc_fn)(void**, size_t),
               hipError_t (*hip_free_fn)(void*)) :
      MemoryAllocator (hip_alloc_fn, hip_free_fn) {}

  HIPAllocator (hipError_t (*hip_alloc_fn)(void**, size_t, unsigned),
                hipError_t (*hip_free_fn)(void*), unsigned flags) :
    MemoryAllocator (hip_alloc_fn, hip_free_fn, flags) {}

  HIPAllocatorType type = AllocatorTypeCoarsegrained;

  virtual hipError_t GetIpcHandle(void *dev_ptr, void *handle)
  {
    return hipIpcGetMemHandle(reinterpret_cast<hipIpcMemHandle_t *>(handle), dev_ptr);
  }

  virtual hipError_t OpenIpcHandle(void **dev_ptr, void *handle)
  {
    return hipIpcOpenMemHandle(dev_ptr, *(reinterpret_cast<hipIpcMemHandle_t *>(handle)),
                               hipIpcMemLazyEnablePeerAccess);
  }

  virtual hipError_t CloseIpcHandle(void *dev_ptr)
  {
    return hipIpcCloseMemHandle(dev_ptr);
  }

  virtual size_t GetIpcHandleSize()
  {
    return sizeof(hipIpcMemHandle_t);
  }

  virtual HIPIpcHandleVec* AllocateIpcHandleVec(int num_elems)
  {
    HIPIpcHandleLegacyVec* vec = new HIPIpcHandleLegacyVec();
    vec->handle.resize(num_elems);
    return vec;
  }
};

using HIPAllocatorCoarsegrained = HIPAllocator;

class HIPAllocatorFinegrained : public HIPAllocator {
 public:
  HIPAllocatorFinegrained()
      : HIPAllocator(hipExtMallocWithFlags, hipFree,
                     hipDeviceMallocFinegrained) {
    type = AllocatorTypeFinegrained;
  }
};

#if defined HAVE_DEVICE_MALLOC_UNCACHED
class HIPAllocatorUncached : public HIPAllocator {
 public:
  HIPAllocatorUncached()
      : HIPAllocator(hipExtMallocWithFlags, hipFree,
                     hipDeviceMallocUncached) {
    type = AllocatorTypeUncached;
  }
};
#endif

#if HIP_VERSION >= 70000000
class HIPAllocatorVMMPosixFd : public HIPAllocator {
 private:
  struct VMMAllocationInfo {
    hipMemGenericAllocationHandle_t handle;
    size_t size;
  };
  static std::map<void*, VMMAllocationInfo> allocations_;
  static std::map<void*, VMMAllocationInfo> imported_allocations_;

 public:
  static hipError_t VMMAlloc(void** ptr, size_t size)
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
    prop.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;

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

    // Success - store allocation info and set output pointer
    VMMAllocationInfo info;
    info.handle = handle;
    info.size = alloc_size;
    allocations_[dev_ptr] = info;

    *ptr = dev_ptr;
    return hipSuccess;
  }

  static hipError_t VMMFree(void* ptr)
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
    hipError_t err;

    // Unmap memory
    err = hipMemUnmap(ptr, info.size);
    if (err != hipSuccess) {
      return err;
    }

    // Free address space
    err = hipMemAddressFree(ptr, info.size);
    if (err != hipSuccess) {
      return err;
    }

    // Release handle
    err = hipMemRelease(info.handle);
    if (err != hipSuccess) {
      return err;
    }

    // Remove from tracking map
    allocations_.erase(it);

    return hipSuccess;
  }

  HIPAllocatorVMMPosixFd() : HIPAllocator(VMMAlloc, VMMFree) {
    type = AllocatorTypeVMM;

    // Check if the device supports VMM
    int device_id;
    hipError_t err = hipGetDevice(&device_id);
    if (err != hipSuccess) {
      fprintf(stderr, "ROCSHMEM_ERROR: Failed to get current device for VMM support check: %s\n",
              hipGetErrorString(err));
      abort();
    }

    int vmm_supported = 0;
    err = hipDeviceGetAttribute(&vmm_supported,
                                 hipDeviceAttributeVirtualMemoryManagementSupported,
                                 device_id);
    if (err != hipSuccess) {
      fprintf(stderr, "ROCSHMEM_ERROR: Failed to query VMM support attribute: %s\n",
              hipGetErrorString(err));
      abort();
    }

    if (!vmm_supported) {
      fprintf(stderr, "ROCSHMEM_ERROR: Virtual Memory Management (VMM) is not supported on device %d. "
              "The USE_HEAP_DEVICE_VMM_POSIX allocator requires a GPU with VMM support. "
              "Please use a different memory allocator.\n",
              device_id);
      abort();
    }
  }

  hipError_t GetIpcHandle(void *dev_ptr, void *handle)
  {
    if (dev_ptr == nullptr || handle == nullptr) {
      return hipErrorInvalidValue;
    }

    // Find allocation info
    auto it = allocations_.find(dev_ptr);
    if (it == allocations_.end()) {
      return hipErrorInvalidValue;
    }

    hipIpcMemHandlePosix_t* posix_handle = reinterpret_cast<hipIpcMemHandlePosix_t*>(handle);
    hipError_t err;

    // Export the VMM handle to a shareable file descriptor
    int fd;
    err = hipMemExportToShareableHandle(&fd, it->second.handle,
                                         hipMemHandleTypePosixFileDescriptor, 0);
    if (err != hipSuccess) {
      return err;
    }

    // Get current process ID and fill handle
    posix_handle->fd = static_cast<uint64_t>(fd);
    posix_handle->pid = static_cast<uint32_t>(getpid());
    posix_handle->size = it->second.size;

    return hipSuccess;
  }

  hipError_t OpenIpcHandle(void **dev_ptr, void *handle)
  {
    if (dev_ptr == nullptr || handle == nullptr) {
      return hipErrorInvalidValue;
    }

    hipIpcMemHandlePosix_t* posix_handle = reinterpret_cast<hipIpcMemHandlePosix_t*>(handle);
    int fd = static_cast<int>(posix_handle->fd);
    int pid = static_cast<int>(posix_handle->pid);
    size_t size = posix_handle->size;
    hipError_t err;

    // Open pidfd for the remote process
    int pid_fd = static_cast<int>(syscall(__NR_pidfd_open, pid, 0));
    if (pid_fd == -1) {
      int err_code = errno;
      fprintf(stderr, "pidfd_open failed for pid %d: %s (errno=%d)\n",
              pid, strerror(err_code), err_code);
      fprintf(stderr, "A common reason is lacking CAP_SYS_PTRACE capability.\n");
      fprintf(stderr, "You can resolve it e.g. with `sudo setcap 'cap_sys_ptrace=ep' <executable>` \n");
      return hipErrorInvalidValue;
    }

    // Get the file descriptor from the remote process
    int open_fd = static_cast<int>(syscall(__NR_pidfd_getfd, pid_fd, fd, 0));
    if (open_fd == -1) {
      int err_code = errno;
      fprintf(stderr, "pidfd_getfd failed for pid %d, fd %d: %s (errno=%d)\n",
              pid, fd, strerror(err_code), err_code);
      fprintf(stderr, "A common reason is lacking CAP_SYS_PTRACE capability\n");
      fprintf(stderr, "You can resolve it e.g. with `sudo setcap 'cap_sys_ptrace=ep' <executable>` \n");
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

    // Close the file descriptor as it's no longer needed
    close(open_fd);

    // Track the imported allocation
    imported_allocations_[base_addr] = {imported_handle, size};

    // Set output pointer to the mapped address
    *dev_ptr = base_addr;
    return hipSuccess;
  }

  hipError_t CloseIpcHandle(void *dev_ptr)
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

  size_t GetIpcHandleSize()
  {
    return sizeof(hipIpcMemHandlePosix_t);
  }

  HIPIpcHandleVec* AllocateIpcHandleVec(int num_elems)
  {
    HIPIpcHandlePosixVec* vec = new HIPIpcHandlePosixVec();
    vec->handle.resize(num_elems);
    return vec;
  }
};
#endif

class HIPHostAllocator : public MemoryAllocator {
 public:
  HIPHostAllocator()
      : MemoryAllocator(hipHostMalloc, hipFree, hipHostMallocCoherent) {}
};

class PosixAligned64Allocator : public MemoryAllocator {
 public:
  PosixAligned64Allocator() : MemoryAllocator(posix_memalign, std::free, 64) {}
};

using HostAllocator = PosixAligned64Allocator;
}  // namespace rocshmem

#endif  // LIBRARY_SRC_MEMORY_HIP_ALLOCATOR_HPP_
