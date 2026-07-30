/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "device_buffer.h"

#include "alloc.h"
#include "checks.h"
#include "debug.h"

#include <cuda_runtime.h>
#include <cstdlib>
#include <utility>
#include <iostream>

namespace meta::comms {

// Helper macro for catching HIP errors
#define HIP_CALL(cmd) \
  do { \
    hipError_t error = (cmd); \
    if (error != hipSuccess) { \
      std::cerr << "Encountered HIP error (" << hipGetErrorString(error) << ") at line " << __LINE__ << " in file " \
                << __FILE__ << "\n"; \
    } \
  } while (0)

DeviceBuffer::DeviceBuffer(std::size_t size, bool useVmm, struct ncclMemManager* manager)
  : size_(size), manager_(manager) {
  if (useVmm && ncclCuMemEnable()) {
    ncclResult_t res = ncclCuMemAlloc(&ptr_, &vmmHandle_, ncclCuMemHandleType, size, manager_);
    if (res == ncclSuccess && ptr_ != nullptr) {
      isVmm_ = true;
      return;
    }
    ptr_ = nullptr; // fall back to legacy allocation
  }
#if defined(HIP_UNCACHED_MEMORY)
  HIP_CALL(hipExtMallocWithFlags((void**)&ptr_, size, hipDeviceMallocUncached));
#else
  HIP_CALL(hipExtMallocWithFlags((void**)&ptr_, size, hipDeviceMallocFinegrained));
#endif
}

void DeviceBuffer::freeBuffer() {
  if (!ptr_) {
    return;
  }
  if (isVmm_) {
    (void)ncclCuMemFree(ptr_, manager_);
  } else {
    CUDACHECKIGNORE(cudaFree(ptr_));
  }
  ptr_ = nullptr;
}

DeviceBuffer::~DeviceBuffer() {
  freeBuffer();
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
  : ptr_(other.ptr_), size_(other.size_), isVmm_(other.isVmm_), manager_(other.manager_), vmmHandle_(other.vmmHandle_) {
  other.ptr_ = nullptr;
  other.size_ = 0;
  other.isVmm_ = false;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
  if (this != &other) {
    freeBuffer();
    ptr_ = other.ptr_;
    size_ = other.size_;
    isVmm_ = other.isVmm_;
    manager_ = other.manager_;
    vmmHandle_ = other.vmmHandle_;
    other.ptr_ = nullptr;
    other.size_ = 0;
    other.isVmm_ = false;
  }
  return *this;
}

} // namespace meta::comms
