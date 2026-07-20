/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "device/rocm/rocassertfaultbuffer.hpp"

#include "device/rocm/rocdevice.hpp"

namespace amd::roc {

AssertFaultBuffer::AssertFaultBuffer(Device& device)
    : faultBuffer_(nullptr), faultBuffer_size_(0), gpuDevice_(device) {}

AssertFaultBuffer::~AssertFaultBuffer() {
  if (faultBuffer_ != nullptr) {
    dev().hostFree(faultBuffer_, faultBuffer_size_);
  }
}

bool AssertFaultBuffer::allocate() {
  if (nullptr != faultBuffer_) {
    return true;
  }

  faultBuffer_size_ = kSize;
  faultBuffer_ = reinterpret_cast<address>(
      dev().hostAlloc(faultBuffer_size_, kAlign, Device::MemorySegment::kAtomics, nullptr, false));
  if (faultBuffer_ != nullptr) {
    memset(faultBuffer_, 0, faultBuffer_size_);
  }

  return (nullptr != faultBuffer_);
}

}  // namespace amd::roc
