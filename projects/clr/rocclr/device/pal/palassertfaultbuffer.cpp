/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "device/pal/palassertfaultbuffer.hpp"

#include "device/pal/paldevice.hpp"

namespace amd::pal {

AssertFaultBuffer::AssertFaultBuffer(Device& device)
    : faultBuffer_(nullptr), gpuDevice_(device) {}

AssertFaultBuffer::~AssertFaultBuffer() {
  if (faultBuffer_ != nullptr) {
    dev().svmFree(faultBuffer_);
  }
}

bool AssertFaultBuffer::allocate() {
  if (nullptr != faultBuffer_) {
    return true;
  }

  faultBuffer_ = dev().svmAlloc(dev().context(), kSize, kAlign,
                                CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, nullptr);
  if (faultBuffer_ != nullptr) {
    memset(faultBuffer_, 0, kSize);
  }

  return (nullptr != faultBuffer_);
}

}  // namespace amd::pal
