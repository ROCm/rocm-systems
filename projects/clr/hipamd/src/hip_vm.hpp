/* Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#ifndef HIP_SRC_HIP_VM_H
#define HIP_SRC_HIP_VM_H

#include <hip/hip_runtime.h>
#include "hip_internal.hpp"

#include "platform/object.hpp"

namespace hip {

hipError_t ihipFree(void* ptr);

class GenericAllocation : public amd::RuntimeObject {
  amd::Memory& phys_mem_ref_;        //<! Physical memory object
  size_t size_;                      //<! Allocated size
  hipMemAllocationProp properties_;  //<! Allocation Properties

 public:
  GenericAllocation(amd::Memory& phys_mem_ref, size_t size, const hipMemAllocationProp& prop)
      : phys_mem_ref_(phys_mem_ref), size_(size), properties_(prop) {}
  ~GenericAllocation() {
    amd::Context* amdContext = g_devices[properties_.location.id]->asContext();
    amd::SvmBuffer::free(*amdContext, phys_mem_ref_.getSvmPtr());
  }

  const hipMemAllocationProp& GetProperties() const { return properties_; }
  hipMemGenericAllocationHandle_t asMemGenericAllocationHandle() {
    return reinterpret_cast<hipMemGenericAllocationHandle_t>(this);
  }
  amd::Memory& asAmdMemory() { return phys_mem_ref_; }

  virtual ObjectType objectType() const { return ObjectTypeVMMAlloc; }
};
};  // namespace hip

#endif  // HIP_SRC_HIP_VM_H
