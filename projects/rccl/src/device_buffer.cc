/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "device_buffer.h"

#include "checks.h"
#include "debug.h"

#include <cuda_runtime.h>
#include <cstdlib>
#include <utility>
#include <iostream>

namespace meta::comms {

// Helper macro for catching HIP errors
#define HIP_CALL(cmd)                                                                   \
    do {                                                                                \
        hipError_t error = (cmd);                                                       \
        if (error != hipSuccess)                                                        \
        {                                                                               \
            std::cerr << "Encountered HIP error (" << hipGetErrorString(error)          \
                      << ") at line " << __LINE__ << " in file " << __FILE__ << "\n";   \
        }                                                                               \
    } while (0)

DeviceBuffer::DeviceBuffer(std::size_t size) : size_(size) {
  
#if defined(HIP_UNCACHED_MEMORY)  
  HIP_CALL(hipExtMallocWithFlags((void**)&ptr_, size, hipDeviceMallocUncached));
#else
  HIP_CALL(hipExtMallocWithFlags((void**)&ptr_, size, hipDeviceMallocFinegrained));
#endif
}

DeviceBuffer::~DeviceBuffer() {
  if (ptr_) {
    CUDACHECKIGNORE(cudaFree(ptr_));
  }
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : ptr_(other.ptr_), size_(other.size_) {
  other.ptr_ = nullptr;
  other.size_ = 0;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
  if (this != &other) {
    if (ptr_) {
      CUDACHECKIGNORE(cudaFree(ptr_));
    }
    ptr_ = other.ptr_;
    size_ = other.size_;
    other.ptr_ = nullptr;
    other.size_ = 0;
  }
  return *this;
}

} // namespace meta::comms
