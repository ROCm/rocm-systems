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

namespace meta::comms {

DeviceBuffer::DeviceBuffer(std::size_t size) : size_(size) {
  cudaError_t err = cudaMalloc(&ptr_, size);
  if (err != cudaSuccess) {
    WARN("DeviceBuffer: cudaMalloc failed (%s)", cudaGetErrorString(err));
    std::abort();
  }
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
  ptr_ = other.ptr_;
  size_ = other.size_;
  other.ptr_ = nullptr;
  other.size_ = 0;
  return *this;
}

} // namespace meta::comms
