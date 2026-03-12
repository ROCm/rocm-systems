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
#include <hsa/hsa_ext_amd.h>

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

  virtual hipError_t GetDmabufHandle(void *dev_ptr, size_t size, int *dmabuf_fd, uint64_t *dmabuf_offset)
  {
    if (dev_ptr == nullptr || dmabuf_fd == nullptr || dmabuf_offset == nullptr) {
      return hipErrorInvalidValue;
    }

    // Use HSA API to export dmabuf from device pointer
    uint64_t offset = 0;
    int fd = -1;
    hsa_status_t status = hsa_amd_portable_export_dmabuf(dev_ptr, size, &fd, &offset);

    if (status != HSA_STATUS_SUCCESS) {
      *dmabuf_fd = -1;
      *dmabuf_offset = 0;
      return hipErrorInvalidValue;
    }

    *dmabuf_fd = fd;
    *dmabuf_offset = offset;
    return hipSuccess;
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
    int exported_fd;  // File descriptor exported for IPC, -1 if not exported
  };
  static std::map<void*, VMMAllocationInfo> allocations_;
  static std::map<void*, VMMAllocationInfo> imported_allocations_;

  static hipError_t VMMAlloc(void** ptr, size_t size);
  static hipError_t VMMFree(void* ptr);

 public:
  HIPAllocatorVMMPosixFd();

  hipError_t GetIpcHandle(void *dev_ptr, void *handle) override;
  hipError_t OpenIpcHandle(void **dev_ptr, void *handle) override;
  hipError_t CloseIpcHandle(void *dev_ptr) override;
  size_t GetIpcHandleSize() override;
  HIPIpcHandleVec* AllocateIpcHandleVec(int num_elems) override;
  hipError_t GetDmabufHandle(void *dev_ptr, size_t size, int *dmabuf_fd, uint64_t *dmabuf_offset) override;
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
