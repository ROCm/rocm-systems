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

#ifndef LIBRARY_SRC_MEMORY_HIP_ALLOCATOR_VMM_FABRIC_HPP_
#define LIBRARY_SRC_MEMORY_HIP_ALLOCATOR_VMM_FABRIC_HPP_

#include "hip_allocator.hpp"

#if HIP_VERSION >= 70000000

#include <map>
#include <cstdint>

namespace rocshmem {

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
    return reinterpret_cast<void*>(&this->handle[elem]);
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

 public:
  HIPAllocatorVMMFabric();

  hipError_t GetIpcHandle(void *dev_ptr, void *handle) override;
  hipError_t OpenIpcHandle(void **dev_ptr, void *handle) override;
  hipError_t CloseIpcHandle(void *dev_ptr) override;
  size_t GetIpcHandleSize() override;
  HIPIpcHandleVec* AllocateIpcHandleVec(int num_elems) override;
  hipError_t GetDmabufHandle(void *dev_ptr, size_t size, int *dmabuf_fd, uint64_t *dmabuf_offset) override;
};

}  // namespace rocshmem

#endif  // HIP_VERSION >= 70000000

#endif  // LIBRARY_SRC_MEMORY_HIP_ALLOCATOR_VMM_FABRIC_HPP_
