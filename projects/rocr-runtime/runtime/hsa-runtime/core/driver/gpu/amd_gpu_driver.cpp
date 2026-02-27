////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#include "core/inc/amd_gpu_driver.h"

#include "hsakmt/hsakmt.h"

#include "core/inc/runtime.h"

#if defined(__linux__)
static_assert(
    (sizeof(rocr::core::ShareableHandle::handle) >= sizeof(HsaMemoryObjectHandle)) &&
        (alignof(rocr::core::ShareableHandle::handle) >= alignof(HsaMemoryObjectHandle)),
    "ShareableHandle cannot store a HsaMemoryObjectHandle");
#endif

namespace rocr {
namespace AMD {

GpuDriver::GpuDriver(std::string devnode_name)
    : core::Driver(core::DriverType::GPU, std::move(devnode_name)) {}

HSA_QUEUE_PRIORITY GpuDriver::HsaInternalToKfdPriority(
    HSA::hsa_amd_queue_priority_internal_t priority) {
  switch (priority) {
    case HSA::HSA_AMD_QUEUE_PRIORITY_LOW:
      return HSA_QUEUE_PRIORITY_MINIMUM;
    case HSA::HSA_AMD_QUEUE_PRIORITY_NORMAL:
      return HSA_QUEUE_PRIORITY_NORMAL;
    case HSA::HSA_AMD_QUEUE_PRIORITY_HIGH:
      return HSA_QUEUE_PRIORITY_HIGH;
    case HSA::HSA_AMD_QUEUE_PRIORITY_MAXIMUM:
      return HSA_QUEUE_PRIORITY_MAXIMUM;
    default:
      return HSA_QUEUE_PRIORITY_NORMAL;
  }
}

HsaMemoryMapFlags GpuDriver::mem_perm(hsa_access_permission_t perm) {
  switch (perm) {
  case HSA_ACCESS_PERMISSION_RO:
    return HSA_MEMORY_ACCESS_RO;
  case HSA_ACCESS_PERMISSION_WO:
    return HSA_MEMORY_ACCESS_WO;
  case HSA_ACCESS_PERMISSION_RW:
    return HSA_MEMORY_ACCESS_RW;
  case HSA_ACCESS_PERMISSION_NONE:
  default:
    return HSA_MEMORY_ACCESS_NONE;
  }
}

bool GpuDriver::BindXnackMode() {
  // Get users' preference for Xnack mode of ROCm platform.
  HSAint32 mode = core::Runtime::runtime_singleton_->flag().xnack();
  bool config_xnack = (mode != Flag::XNACK_REQUEST::XNACK_UNCHANGED);

  // Indicate to driver users' preference for Xnack mode
  // Call to driver can fail and is a supported feature
  HSAKMT_STATUS status = HSAKMT_STATUS_ERROR;
  if (config_xnack) {
    status = HSAKMT_CALL(hsaKmtSetXNACKMode(mode));
    if (status == HSAKMT_STATUS_SUCCESS) {
      return (mode != Flag::XNACK_DISABLE);
    }
  }

  // Get Xnack mode of devices bound by driver. This could happen
  // when a call to SET Xnack mode fails or user has no particular
  // preference
  status = HSAKMT_CALL(hsaKmtGetXNACKMode(&mode));
  if (status != HSAKMT_STATUS_SUCCESS) {
    debug_print(
        "KFD does not support xnack mode query.\nROCr must assume "
        "xnack is disabled.\n");
    return false;
  }
  return (mode != Flag::XNACK_DISABLE);
}

void *GpuDriver::AllocateKfdMemory(const HsaMemFlags &flags, uint32_t node_id,
                                   size_t size) {
  void *mem = nullptr;
  const HSAKMT_STATUS status = HSAKMT_CALL(hsaKmtAllocMemory(node_id, size, flags, &mem));
  return (status == HSAKMT_STATUS_SUCCESS) ? mem : nullptr;
}

bool GpuDriver::FreeKfdMemory(void *mem, size_t size) {
  if (mem == nullptr || size == 0) {
    debug_print("Invalid free ptr:%p size:%lu\n", mem, size);
    return false;
  }

  if (HSAKMT_CALL(hsaKmtFreeMemory(mem, size)) != HSAKMT_STATUS_SUCCESS) {
    debug_print("Failed to free ptr:%p size:%lu\n", mem, size);
    return false;
  }
  return true;
}

bool GpuDriver::MakeKfdMemoryResident(size_t num_node, const uint32_t *nodes,
                                      const void *mem, size_t size,
                                      uint64_t *alternate_va,
                                      HsaMemMapFlags map_flag) {
  assert(num_node > 0);
  assert(nodes);

  *alternate_va = 0;

  HSAKMT_STATUS kmt_status(HSAKMT_CALL(hsaKmtMapMemoryToGPUNodes(
      const_cast<void *>(mem), size, alternate_va, map_flag, num_node,
      const_cast<uint32_t *>(nodes))));

  return (kmt_status == HSAKMT_STATUS_SUCCESS);
}

void GpuDriver::MakeKfdMemoryUnresident(const void *mem) {
  HSAKMT_CALL(hsaKmtUnmapMemoryToGPU(const_cast<void *>(mem)));
}

} // namespace AMD
} // namespace rocr
