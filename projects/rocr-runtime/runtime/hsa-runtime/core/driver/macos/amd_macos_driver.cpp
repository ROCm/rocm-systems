////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
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

#ifdef __APPLE__

#include "core/inc/amd_macos_driver.h"

#include <utility>

#include "core/inc/memory_region.h"

namespace rocr {
namespace AMD {

MacOsDriver::MacOsDriver(std::string devnode_name)
    : core::Driver(core::DriverType::MACOS_DEXT, std::move(devnode_name)) {}

// Stage 2B scaffold: always reports no device. Stage 1 (libmacgpu) replaces
// this with an IOKit IOServiceMatching / IOServiceOpen probe against the
// ROCmGPU.dext user-client.
hsa_status_t MacOsDriver::DiscoverDriver(std::unique_ptr<core::Driver>& driver) {
  (void)driver;
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::Init()   { return HSA_STATUS_ERROR; }
hsa_status_t MacOsDriver::ShutDown() { return HSA_STATUS_ERROR; }

hsa_status_t MacOsDriver::QueryKernelModeDriver(core::DriverQuery) {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::Open()  { return HSA_STATUS_ERROR; }
hsa_status_t MacOsDriver::Close() { return HSA_STATUS_ERROR; }

hsa_status_t MacOsDriver::GetSystemProperties(HsaSystemProperties& sys_props) const {
  sys_props.NumNodes = 0;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t MacOsDriver::GetNodeProperties(HsaNodeProperties&, uint32_t) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::GetEdgeProperties(std::vector<HsaIoLinkProperties>&,
                                            uint32_t) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::GetMemoryProperties(uint32_t,
                                              std::vector<HsaMemoryProperties>&) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::GetCacheProperties(uint32_t, uint32_t,
                                             std::vector<HsaCacheProperties>&) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::AllocateMemory(const core::MemoryRegion&,
                                         core::MemoryRegion::AllocateFlags,
                                         void**, size_t, uint32_t) {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::FreeMemory(void*, size_t) { return HSA_STATUS_ERROR; }

hsa_status_t MacOsDriver::CreateQueue(uint32_t, HSA_QUEUE_TYPE, uint32_t,
                                      HSA::hsa_amd_queue_priority_internal_t,
                                      uint32_t, void*, uint64_t, HsaEvent*,
                                      HsaQueueResource&) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::DestroyQueue(HSA_QUEUEID) const { return HSA_STATUS_ERROR; }

hsa_status_t MacOsDriver::UpdateQueue(HSA_QUEUEID, uint32_t,
                                      HSA::hsa_amd_queue_priority_internal_t,
                                      void*, uint64_t, HsaEvent*) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::SetQueueCUMask(HSA_QUEUEID, uint32_t, uint32_t*) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::AllocQueueGWS(HSA_QUEUEID, uint32_t, uint32_t*) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::ExportDMABuf(void*, size_t, int*, size_t*) {
  // Darwin has no dma-buf fd passing between processes for GPU memory —
  // IOSurface is the nearest analog but requires a different
  // handshake. Out of scope.
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::ImportDMABuf(int, const core::Agent&,
                                       core::ShareableHandle*, void*) {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::DestroyImportedShareableHandle(core::ShareableHandle*) {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::Map(core::ShareableHandle, void*, size_t, size_t,
                              hsa_access_permission_t) {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::Unmap(core::ShareableHandle, void*, size_t, size_t) {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::CreateShareableHandle(void*, void*, size_t,
                                                const core::Agent&,
                                                core::ShareableHandle*, uint64_t*,
                                                int*, uint64_t*) {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::DestroyShareableHandle(core::ShareableHandle*) {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::SPMAcquire(uint32_t) const { return HSA_STATUS_ERROR; }
hsa_status_t MacOsDriver::SPMRelease(uint32_t) const { return HSA_STATUS_ERROR; }
hsa_status_t MacOsDriver::SPMSetDestBuffer(uint32_t, uint32_t, uint32_t*, uint32_t*,
                                           void*, bool*) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::SetTrapHandler(uint32_t, const void*, uint64_t,
                                         const void*, uint64_t) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::GetDeviceHandle(uint32_t, void**) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::GetClockCounters(uint32_t, HsaClockCounters*) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::GetTileConfig(uint32_t, HsaGpuTileConfig*) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::IsModelEnabled(bool* enable) const {
  if (enable) *enable = false;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t MacOsDriver::GetWallclockFrequency(uint32_t, uint64_t*) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::AllocateScratchMemory(uint32_t, uint64_t, void**) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::AvailableMemory(uint32_t, uint64_t*) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::RegisterMemory(void*, uint64_t, HsaMemFlags) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::DeregisterMemory(void*) const { return HSA_STATUS_ERROR; }

hsa_status_t MacOsDriver::MakeMemoryResident(const void*, size_t, uint64_t*,
                                             const HsaMemMapFlags*,
                                             uint32_t, const uint32_t*) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::MakeMemoryUnresident(const void*) const {
  return HSA_STATUS_ERROR;
}

hsa_status_t MacOsDriver::GetQueueSaveAreaInfo(HSA_QUEUEID, void**, size_t*) const {
  return HSA_STATUS_ERROR;
}

}  // namespace AMD
}  // namespace rocr

#endif  // __APPLE__
