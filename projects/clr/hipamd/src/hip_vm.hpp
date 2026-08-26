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

//! Resolve the g_devices[] index that owns a VMM allocation's bookkeeping.
//! For Device allocations location.id is the device index. For host allocations
//! (Host / HostNuma / HostNumaCurrent) location.id is either ignored or a NUMA
//! node id, so the VA-level operations are anchored on the current device.
int VmmOwnerDeviceIndex(const hipMemAllocationProp& prop);

class GenericAllocation : public amd::RuntimeObject {
  amd::Memory& phys_mem_ref_;        //<! Physical memory object
  size_t size_;                      //<! Allocated size
  hipMemAllocationProp properties_;  //<! Allocation Properties

 public:
  GenericAllocation(amd::Memory& phys_mem_ref, size_t size, const hipMemAllocationProp& prop)
      : phys_mem_ref_(phys_mem_ref), size_(size), properties_(prop) {}
  ~GenericAllocation() {
    // Host-backed allocations (Host / HostNuma / HostNumaCurrent) are allocated on
    // host_context; only Device allocations live on a per-device context indexed by
    // location.id (which is a NUMA node id, not a device index, for host-NUMA).
    amd::Context* amdContext = (properties_.location.type == hipMemLocationTypeDevice)
        ? g_devices[properties_.location.id]->asContext()
        : hip::host_context;
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
