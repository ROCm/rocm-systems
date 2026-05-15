/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

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
