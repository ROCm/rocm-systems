/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "platform/context.hpp"
#include "platform/memory.hpp"

namespace amd {
class ExternalMemory : public InteropObject {
 public:
  enum class HandleType : uint32_t {
    OpaqueFd = 1,
    OpaqueWin32 = 2,
    OpaqueWin32Kmt = 3,
    D3D12Heap = 4,
    D3D12Resource = 5,
    D3D11Resource = 6,
    D3D11ResourceKmt = 7,
    VkImage = 8
  };

  ExternalMemory(amd::Os::FileDesc handle, const void* name, ExternalMemory::HandleType handle_type)
      : handle_(handle), name_(name), handle_type_(handle_type) {}

  virtual ~ExternalMemory() override {}
  ExternalMemory* asExternalMemory() final { return this; }

  amd::Os::FileDesc Handle() const { return handle_; }
  const void* Name() const { return name_; }
  HandleType Type() const { return handle_type_; }

 protected:
  amd::Os::FileDesc handle_;
  const void* name_;
  ExternalMemory::HandleType handle_type_;
};

class ExternalBuffer final : public Buffer, public ExternalMemory {
 protected:
  // Initializes device memory array, which is located after ExternalBuffer object in memory
  void initDeviceMemory() {
    deviceMemories_ =
        reinterpret_cast<DeviceMemory*>(reinterpret_cast<char*>(this) + sizeof(ExternalBuffer));
    memset(deviceMemories_, 0, context_().devices().size() * sizeof(DeviceMemory));
  }

 public:
  //! \p flags are the CL_MEM_* flags the caller passed to clCreateBufferWithProperties. They
  //! have to reach the Buffer base: CL_MEM_FLAGS is answered from it, and the host
  //! read/write/map paths enforce CL_MEM_HOST_NO_ACCESS and friends through getMemFlags().
  ExternalBuffer(Context& amdContext, size_t size_in_bytes, amd::Os::FileDesc handle,
                 ExternalMemory::HandleType handle_type, Memory::Flags flags = 0,
                 const void* name = nullptr)
      : Buffer(amdContext, flags, size_in_bytes), ExternalMemory(handle, name, handle_type) {
    setInteropObj(this);
  }

  virtual ~ExternalBuffer() {}
};
}  // namespace amd
