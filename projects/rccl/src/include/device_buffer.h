/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Minimal DeviceBuffer from Meta torchcomms CudaRAII (device malloc/free).
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include <cstddef>

namespace meta::comms {

class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t size);
  ~DeviceBuffer();

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept;
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

  void* get() const { return ptr_; }

 private:
  void* ptr_{nullptr};
  std::size_t size_{0};
};

} // namespace meta::comms
