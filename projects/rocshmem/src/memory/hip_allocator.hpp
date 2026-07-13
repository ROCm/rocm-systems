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
#include "hip_allocator_vmm_common.hpp"

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

#if HIP_VERSION >= 70000000
struct HIPIpcMemHandlePosix_t {
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
    return reinterpret_cast<void*> (&this->handle.at(elem));
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
    return reinterpret_cast<void*> (&this->handle.at(elem));
  }

protected:
  std::vector<HIPIpcMemHandlePosix_t> handle;

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

  virtual ~HIPAllocator() = default;

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

#if HIP_VERSION >= 70000000
  /**
   * @brief HIP VMM handle type used by this allocator's symmetric heap.
   *
   * Returns hipMemHandleTypeNone for non-VMM allocators. The VMM allocators
   * override this so callers can verify that a user buffer's requested handle
   * type matches the one used by the symmetric heap.
   */
  virtual hipMemAllocationHandleType GetVMMHandleType() const
  {
    return hipMemHandleTypeNone;
  }

  /**
   * @brief Validate that a user buffer can be registered as symmetric.
   *
   * Performs the single-PE checks required before a buffer can be registered
   * as a symmetric region: non-zero and granularity-aligned size, allocated
   * via the HIP VMM APIs, device memory residing on the given device, and a
   * requested handle type matching this allocator's.
   *
   * Non-VMM allocators report hipMemHandleTypeNone and therefore reject all
   * registrations.
   *
   * @param[in] ptr        Pointer to validate
   * @param[in] size       Registered length in bytes
   * @param[in] device_id  This PE's device ordinal
   * @return true if all local checks pass, false otherwise
   */
  bool ValidateVMMRegistration(const void *ptr, size_t size, int device_id) const
  {
    /* Non-VMM allocators cannot back symmetric registrations. */
    hipMemAllocationHandleType handle_type = GetVMMHandleType();
    if (handle_type == hipMemHandleTypeNone) {
      return false;
    }

    /* Non-zero size. */
    if (ptr == nullptr || size == 0) {
      return false;
    }

    /* Granularity-aligned size. */
    size_t granularity = get_granularity();
    if (granularity == 0) {
      granularity = 1;
    }
    if (size % granularity != 0) {
      return false;
    }

    /*
     * Confirm the pointer is HIP VMM memory before doing anything else.
     * hipMemRetainAllocationHandle can fault on non-VMM pointers (e.g.
     * hipMalloc or host memory), whereas hipMemGetAccess fails gracefully with
     * an error, so use it as a safe gate.
     */
    hipMemLocation location = {};
    location.type = hipMemLocationTypeDevice;
    location.id = device_id;
    unsigned long long access_flags = 0;
    if (hipMemGetAccess(&access_flags, &location, const_cast<void *>(ptr)) !=
        hipSuccess) {
      return false;
    }

    /*
     * This PE's device must have read/write access to the buffer; RMA
     * (puts/gets) would otherwise fault. VMM memory can be mapped without
     * hipMemSetAccess having been called for this device, in which case the
     * call above still succeeds but reports no access.
     */
    if (access_flags != hipMemAccessFlagsProtReadWrite) {
      return false;
    }

    /*
     * Now safe to retain the backing VMM handle to inspect its properties.
     */
    hipMemGenericAllocationHandle_t handle;
    if (hipMemRetainAllocationHandle(&handle, const_cast<void *>(ptr)) !=
        hipSuccess) {
      return false;
    }

    hipMemAllocationProp prop = {};
    hipError_t err = hipMemGetAllocationPropertiesFromHandle(&prop, handle);
    (void)hipMemRelease(handle);
    if (err != hipSuccess) {
      return false;
    }

    /* Right device: device memory must live on this PE's device. */
    if (prop.location.id != device_id) {
      return false;
    }

    /* Matching handle type: must match this allocator's symmetric heap. */
    if (prop.requestedHandleTypes != handle_type) {
      return false;
    }

    return true;
  }

  /**
   * @brief Map an existing VMM allocation to a fresh, distinct virtual address.
   *
   * Retains the generic allocation handle backing @p dev_ptr and maps the same
   * physical memory at a newly reserved virtual address. The alias refers to
   * the same data as @p dev_ptr but at a different address owned by rocSHMEM.
   * Used so symmetric registration can return a rocSHMEM-managed address
   * distinct from the user's pointer.
   *
   * @param[in]  dev_ptr VMM allocation to alias
   * @param[in]  length  Length in bytes (granularity-aligned)
   * @param[out] alias   Filled with the newly mapped virtual address
   * @return hipSuccess, or an error (hipErrorNotSupported for non-VMM allocators)
   */
  hipError_t MapLocalAlias(void *dev_ptr, size_t length, void **alias)
  {
    if (GetVMMHandleType() == hipMemHandleTypeNone) {
      return hipErrorNotSupported;
    }
    if (dev_ptr == nullptr || alias == nullptr || length == 0) {
      return hipErrorInvalidValue;
    }

    hipMemGenericAllocationHandle_t handle;
    hipError_t err = hipMemRetainAllocationHandle(&handle, dev_ptr);
    if (err != hipSuccess) {
      return err;
    }

    void *va = nullptr;
    err = hipMemAddressReserve(&va, length, 0, 0, 0);
    if (err != hipSuccess) {
      (void)hipMemRelease(handle);
      return err;
    }

    err = hipMemMap(va, length, 0, handle, 0);
    if (err != hipSuccess) {
      (void)hipMemAddressFree(va, length);
      (void)hipMemRelease(handle);
      return err;
    }

    int device_id = 0;
    err = hipGetDevice(&device_id);
    if (err == hipSuccess) {
      hipMemAccessDesc access_desc[2];
      access_desc[0].location.type = hipMemLocationTypeDevice;
      access_desc[0].location.id = device_id;
      access_desc[0].flags = hipMemAccessFlagsProtReadWrite;
      access_desc[1].location.type = hipMemLocationTypeHost;
      access_desc[1].location.id = 0;
      access_desc[1].flags = hipMemAccessFlagsProtReadWrite;
      err = hipMemSetAccess(va, length, access_desc, 2);
    }
    if (err != hipSuccess) {
      (void)hipMemUnmap(va, length);
      (void)hipMemAddressFree(va, length);
      (void)hipMemRelease(handle);
      return err;
    }

    /* The mapping holds its own reference; drop ours from the retain above. */
    (void)hipMemRelease(handle);

    *alias = va;
    return hipSuccess;
  }

  /**
   * @brief Unmap and free a virtual address alias created by MapLocalAlias().
   *
   * @param[in] alias  Alias virtual address returned by MapLocalAlias()
   * @param[in] length Length in bytes used when the alias was created
   * @return hipSuccess or the first error encountered
   */
  hipError_t UnmapLocalAlias(void *alias, size_t length)
  {
    if (alias == nullptr || length == 0) {
      return hipErrorInvalidValue;
    }
    hipError_t err = hipMemUnmap(alias, length);
    hipError_t free_err = hipMemAddressFree(alias, length);
    return (err != hipSuccess) ? err : free_err;
  }
#endif

  /**
   * @brief Export an IPC handle for an arbitrary (caller-owned) pointer.
   *
   * Unlike GetIpcHandle(), this does not require the pointer to have been
   * produced by this allocator. It is used to register user-supplied
   * symmetric buffers. Only implemented by the VMM allocators.
   *
   * @param[in]  dev_ptr Pointer to the (VMM) allocation to export
   * @param[in]  length  Length of the region in bytes
   * @param[out] handle  Filled with the shareable IPC handle
   */
  virtual hipError_t GetIpcHandleFromPtr(void * /*dev_ptr*/, size_t /*length*/,
                                         void * /*handle*/)
  {
    return hipErrorNotSupported;
  }

  /**
   * @brief Release a handle previously produced by GetIpcHandleFromPtr().
   *
   * For POSIX-fd based handles this closes the exported file descriptor.
   * Default implementation is a no-op.
   *
   * @param[in] handle Handle previously filled by GetIpcHandleFromPtr()
   */
  virtual hipError_t CloseExportedHandle(void * /*handle*/)
  {
    return hipSuccess;
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
    type_ = AllocatorTypeFinegrained;
  }
};

#if defined HAVE_DEVICE_MALLOC_UNCACHED
class HIPAllocatorUncached : public HIPAllocator {
 public:
  HIPAllocatorUncached()
      : HIPAllocator(hipExtMallocWithFlags, hipFree,
                     hipDeviceMallocUncached) {
    type_ = AllocatorTypeUncached;
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

  // Export a generic VMM handle to a POSIX shareable fd and fill the IPC handle
  // struct. The exported fd is returned via out_fd.
  hipError_t ExportToPosixHandle(hipMemGenericAllocationHandle_t generic_handle,
                                 size_t size, void *handle, int *out_fd);

 public:
  HIPAllocatorVMMPosixFd();

  hipError_t GetIpcHandle(void *dev_ptr, void *handle) override;
  hipError_t OpenIpcHandle(void **dev_ptr, void *handle) override;
  hipError_t CloseIpcHandle(void *dev_ptr) override;
  size_t GetIpcHandleSize() override;
  HIPIpcHandleVec* AllocateIpcHandleVec(int num_elems) override;
  hipError_t GetDmabufHandle(void *dev_ptr, size_t size, int *dmabuf_fd, uint64_t *dmabuf_offset) override;
  hipError_t GetIpcHandleFromPtr(void *dev_ptr, size_t length, void *handle) override;
  hipError_t CloseExportedHandle(void *handle) override;

  hipMemAllocationHandleType GetVMMHandleType() const override
  {
    return hipMemHandleTypePosixFileDescriptor;
  }
};

// Forward declarations for fabric handle support (part of future HIP releases)
#ifndef hipMemFabricHandle_t
typedef uint64_t hipMemFabricHandle_t;
#endif

#ifndef hipMemHandleTypeFabric
#define hipMemHandleTypeFabric (hipMemAllocationHandleType)3
#endif

#ifndef hipDeviceAttributeHandleTypeFabricSupported
#define hipDeviceAttributeHandleTypeFabricSupported (hipDeviceAttribute_t)999
#endif

/**
 * Fabric handle structure for IPC
 */
struct HIPIpcMemHandleFabric_t {
  hipMemFabricHandle_t fabric_handle;
  size_t size;
  size_t offset;
};

/**
 * IPC handle vector for fabric handles
 */
class HIPIpcHandleFabricVec : public HIPIpcHandleVec {
public:
  friend class HIPAllocatorVMMFabric;

  HIPIpcHandleType GetIpcHandleType() override { return HandleTypeFabric; }

  void* GetHandleVecElem(int elem) override
  {
    return reinterpret_cast<void*>(&this->handle.at(elem));
  }

protected:
  std::vector<HIPIpcMemHandleFabric_t> handle;
};

/**
 * HIP VMM allocator using fabric handles for IPC
 */
class HIPAllocatorVMMFabric : public HIPAllocator {
 private:
  struct VMMFabricAllocationInfo {
    hipMemGenericAllocationHandle_t handle;
    size_t size;
    uint64_t fabric_id;  // Fabric handle ID, 0 if not exported
  };

  static std::map<void*, VMMFabricAllocationInfo> allocations_;
  static std::map<void*, VMMFabricAllocationInfo> imported_allocations_;

  static hipError_t VMMAlloc(void** ptr, size_t size);
  static hipError_t VMMFree(void* ptr);

  // Export a generic VMM handle to a fabric handle and fill the IPC handle
  // struct.
  hipError_t ExportToFabricHandle(hipMemGenericAllocationHandle_t generic_handle,
                                  size_t size, void *handle);

 public:
  HIPAllocatorVMMFabric();

  hipError_t GetIpcHandle(void *dev_ptr, void *handle) override;
  hipError_t OpenIpcHandle(void **dev_ptr, void *handle) override;
  hipError_t CloseIpcHandle(void *dev_ptr) override;
  size_t GetIpcHandleSize() override;
  HIPIpcHandleVec* AllocateIpcHandleVec(int num_elems) override;
  hipError_t GetDmabufHandle(void *dev_ptr, size_t size, int *dmabuf_fd, uint64_t *dmabuf_offset) override;
  hipError_t GetIpcHandleFromPtr(void *dev_ptr, size_t length, void *handle) override;

  hipMemAllocationHandleType GetVMMHandleType() const override
  {
    return hipMemHandleTypeFabric;
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
