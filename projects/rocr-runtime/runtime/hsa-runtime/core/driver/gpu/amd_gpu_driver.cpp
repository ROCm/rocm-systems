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

HSA_QUEUE_PRIORITY GpuDriver::HsaInternalToDriverPriority(
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

// Static member definition — process-level state (opaque, defined per-platform).
GpuDriver::ProcessState* GpuDriver::process_state_ = nullptr;

} // namespace AMD
} // namespace rocr
