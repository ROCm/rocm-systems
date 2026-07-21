/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Minimal DeviceBuffer from Meta torchcomms CudaRAII (device malloc/free).
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include <cuda.h>

#include <cstddef>

struct ncclMemManager;

namespace meta::comms {

class DeviceBuffer {
public:
  // useVmm: try ncclCuMemAlloc (VMM); falls back to hipExtMalloc if VMM is
  // unavailable or the allocation fails. manager: optional ncclMemManager used
  // for VMM allocation tracking.
  explicit DeviceBuffer(std::size_t size, bool useVmm = false, struct ncclMemManager* manager = nullptr);
  ~DeviceBuffer();

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept;
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

  void* get() const {
    return ptr_;
  }
  bool isVmm() const {
    return isVmm_;
  }
  CUmemGenericAllocationHandle vmmHandle() const {
    return vmmHandle_;
  }

private:
  void freeBuffer();

  void* ptr_{nullptr};
  std::size_t size_{0};
  bool isVmm_{false};
  struct ncclMemManager* manager_{nullptr};
  CUmemGenericAllocationHandle vmmHandle_{};
};

} // namespace meta::comms
