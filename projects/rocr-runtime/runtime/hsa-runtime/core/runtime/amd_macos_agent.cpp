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

#include "core/inc/amd_macos_agent.h"

#include <cstring>
#include <mach/mach_time.h>

namespace rocr {
namespace AMD {

MacGpuAgent::MacGpuAgent(uint32_t node_id, core::DriverType driver_type)
    : GpuAgentInt(node_id, driver_type),
      current_coherency_type_(HSA_AMD_COHERENCY_TYPE_COHERENT),
      rec_sdma_eng_override_(false) {
  // MVP: empty regions + isas. MacGpuAgent will be populated once
  // a MacOsDriver::GetMemoryProperties path for GPU-local memory exists
  // and gfx1201 ISA is registered for Darwin builds.
}

MacGpuAgent::~MacGpuAgent() = default;

void MacGpuAgent::ReleaseResources() {
  // Nothing held yet.
}

hsa_status_t MacGpuAgent::PostToolsInit() { return HSA_STATUS_SUCCESS; }

hsa_status_t MacGpuAgent::VisitRegion(bool /*include_peer*/,
                                      hsa_status_t (*callback)(hsa_region_t, void*),
                                      void* data) const {
  if (!callback) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  for (const auto& region : regions_) {
    hsa_region_t r = core::MemoryRegion::Convert(region.get());
    hsa_status_t s = callback(r, data);
    if (s != HSA_STATUS_SUCCESS) return s;
  }
  return HSA_STATUS_SUCCESS;
}

// Scratch plumbing — no scratch pool on Darwin MVP. Leave the provided
// ScratchInfo struct in its caller-initialized state; AMD::GpuAgent's
// QueueCreate path won't be reached on Darwin until a future commit
// implements queue creation.
void MacGpuAgent::AcquireQueueMainScratch(ScratchInfo&) {}
void MacGpuAgent::AcquireQueueAltScratch(ScratchInfo&) {}
void MacGpuAgent::ReleaseQueueMainScratch(ScratchInfo&) {}
void MacGpuAgent::ReleaseQueueAltScratch(ScratchInfo&) {}

// Timestamp translation: for now just return the input unchanged.
// Proper translation needs host/device clock-counter mapping, which
// depends on libmacgpu exposing a GetClockCounters escape.
uint64_t MacGpuAgent::TranslateTime(uint64_t tick) { return tick; }

void MacGpuAgent::TranslateTime(core::Signal* /*signal*/,
                                hsa_amd_profiling_dispatch_time_t& time) {
  time.start = 0;
  time.end = 0;
}
void MacGpuAgent::TranslateTime(core::Signal* /*signal*/,
                                hsa_amd_profiling_async_copy_time_t& time) {
  time.start = 0;
  time.end = 0;
}

void MacGpuAgent::InvalidateCodeCaches(void* /*ptr*/, size_t /*size*/) {
  // No GPU cache management until MEC ring is up.
}

bool MacGpuAgent::current_coherency_type(hsa_amd_coherency_type_t type) {
  current_coherency_type_ = type;
  return true;
}

void MacGpuAgent::RegisterGangPeer(core::Agent&, unsigned int) {
  // Gang scheduling is a multi-GPU feature we don't support yet.
}
void MacGpuAgent::RegisterRecSdmaEngIdMaskPeer(core::Agent&, uint32_t) {}

// --- core::Agent ---

hsa_status_t MacGpuAgent::IterateRegion(
    hsa_status_t (*callback)(hsa_region_t, void*), void* data) const {
  return VisitRegion(/*include_peer=*/false, callback, data);
}

hsa_status_t MacGpuAgent::IterateSupportedIsas(
    hsa_status_t (*callback)(hsa_isa_t, void*), void* data) const {
  if (!callback) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  for (const auto* isa : supported_isas_) {
    hsa_isa_t ih;
    ih.handle = reinterpret_cast<uint64_t>(isa);
    hsa_status_t s = callback(ih, data);
    if (s != HSA_STATUS_SUCCESS) return s;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t MacGpuAgent::IterateCache(
    hsa_status_t (*/*callback*/)(hsa_cache_t, void*), void* /*data*/) const {
  // No cache topology yet.
  return HSA_STATUS_SUCCESS;
}

hsa_status_t MacGpuAgent::QueueCreate(size_t /*size*/, hsa_queue_type32_t /*queue_type*/,
                                      uint64_t /*flags*/,
                                      core::HsaEventCallback /*event_callback*/,
                                      void* /*data*/, uint32_t /*private_segment_size*/,
                                      uint32_t /*group_segment_size*/,
                                      core::Queue** /*queue*/) {
  // TODO(macos-port): MEC ring + doorbell via libmacgpu MMIO + AllocDMA.
  return HSA_STATUS_ERROR;
}

hsa_status_t MacGpuAgent::GetInfo(hsa_agent_info_t attribute, void* value) const {
  if (!value) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  switch (attribute) {
    case HSA_AGENT_INFO_NAME:
      std::strcpy(static_cast<char*>(value), "MacGpu (macOS eGPU)");
      return HSA_STATUS_SUCCESS;
    case HSA_AGENT_INFO_VENDOR_NAME:
      std::strcpy(static_cast<char*>(value), "AMD");
      return HSA_STATUS_SUCCESS;
    case HSA_AGENT_INFO_DEVICE:
      *static_cast<hsa_device_type_t*>(value) = HSA_DEVICE_TYPE_GPU;
      return HSA_STATUS_SUCCESS;
    case HSA_AGENT_INFO_PROFILE:
      *static_cast<hsa_profile_t*>(value) = HSA_PROFILE_BASE;
      return HSA_STATUS_SUCCESS;
    default:
      // MVP: report INVALID_ARGUMENT for unknown attributes rather than
      // crash. Full enumeration is a follow-up.
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
}

}  // namespace AMD
}  // namespace rocr

#endif  // __APPLE__
