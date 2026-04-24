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

#ifndef HSA_RUNTIME_CORE_INC_AMD_MACOS_AGENT_H_
#define HSA_RUNTIME_CORE_INC_AMD_MACOS_AGENT_H_

#if !defined(__APPLE__)
#error "amd_macos_agent.h is Darwin-only"
#endif

#include <memory>
#include <vector>

#include "core/inc/amd_gpu_agent.h"
#include "core/inc/memory_region.h"

namespace rocr {
namespace AMD {

/// @brief Minimal Darwin GPU agent.
///
/// Inherits the GpuAgentInt interface so ROCR's topology layer can treat
/// MacGpuAgent like any other GPU. Does NOT inherit from AMD::GpuAgent,
/// which pulls in a large KFD-dependent constructor (hsaKmtGetClockCounters,
/// libdrm initialization, scratch pool, blit shaders, trap handler, …)
/// that we can't satisfy from the DEXT-backed libmacgpu path today.
///
/// Most pure virtuals return HSA_STATUS_ERROR. The invariants that ARE
/// implemented:
///   - node_id / device_type / driver() (from Agent base).
///   - regions() / supported_isas() return the vectors populated in the
///     constructor, including an MVP host-backed framebuffer pool used by
///     the code-object loader.
///   - GetInfo returns fixed identity fields from the libmacgpu probe so
///     rocminfo can print a plausible device name.
///   - current_coherency_type getter/setter store a member var.
///
/// What's deliberately stubbed until MacGpuAgent is fleshed out:
///   - QueueCreate — needs MEC ring setup over libmacgpu MMIO + AllocDMA.
///   - DmaCopy family — needs SDMA ring or libmacgpu-mediated blit.
///   - Scratch acquire/release — needs scratch pool over macgpu_alloc_dma.
///   - PcSampling* — needs the profiler backend, not on the bring-up path.
class MacGpuAgent : public GpuAgentInt {
 public:
  MacGpuAgent(uint32_t node_id, core::DriverType driver_type);
  ~MacGpuAgent() override;

  // --- GpuAgentInt ---
  void ReleaseResources() override;
  hsa_status_t PostToolsInit() override;

  hsa_status_t VisitRegion(bool include_peer,
                           hsa_status_t (*callback)(hsa_region_t, void*),
                           void* data) const override;

  void AcquireQueueMainScratch(ScratchInfo& scratch) override;
  void AcquireQueueAltScratch(ScratchInfo& scratch) override;
  void ReleaseQueueMainScratch(ScratchInfo& scratch) override;
  void ReleaseQueueAltScratch(ScratchInfo& scratch) override;

  void TranslateTime(core::Signal* signal,
                     hsa_amd_profiling_dispatch_time_t& time) override;
  void TranslateTime(core::Signal* signal,
                     hsa_amd_profiling_async_copy_time_t& time) override;
  uint64_t TranslateTime(uint64_t tick) override;

  void InvalidateCodeCaches(void* ptr, size_t size) override;

  bool current_coherency_type(hsa_amd_coherency_type_t type) override;
  hsa_amd_coherency_type_t current_coherency_type() const override {
    return current_coherency_type_;
  }

  void RegisterGangPeer(core::Agent& gang_peer, unsigned int bandwidth_factor) override;
  void RegisterRecSdmaEngIdMaskPeer(core::Agent& gang_peer,
                                    uint32_t rec_sdma_eng_id_mask) override;
  void SetRecSdmaEngOverride(bool flag) override { rec_sdma_eng_override_ = flag; }

  hsa_profile_t profile() const override { return HSA_PROFILE_BASE; }
  uint32_t memory_bus_width() const override { return 0; }
  uint32_t memory_max_frequency() const override { return 0; }

  bool AsyncScratchReclaimEnabled() const override { return false; }
  hsa_status_t SetAsyncScratchThresholds(size_t) override {
    return HSA_STATUS_ERROR;
  }

  hsa_status_t PcSamplingIterateConfig(
      hsa_ven_amd_pcs_iterate_configuration_callback_t, void*) override {
    return HSA_STATUS_ERROR;
  }
  hsa_status_t PcSamplingCreate(pcs::PcsRuntime::PcSamplingSession&) override {
    return HSA_STATUS_ERROR;
  }
  hsa_status_t PcSamplingCreateFromId(
      HsaPcSamplingTraceId, pcs::PcsRuntime::PcSamplingSession&) override {
    return HSA_STATUS_ERROR;
  }
  hsa_status_t PcSamplingDestroy(pcs::PcsRuntime::PcSamplingSession&) override {
    return HSA_STATUS_ERROR;
  }
  hsa_status_t PcSamplingStart(pcs::PcsRuntime::PcSamplingSession&) override {
    return HSA_STATUS_ERROR;
  }
  hsa_status_t PcSamplingStop(pcs::PcsRuntime::PcSamplingSession&) override {
    return HSA_STATUS_ERROR;
  }
  hsa_status_t PcSamplingFlush(pcs::PcsRuntime::PcSamplingSession&) override {
    return HSA_STATUS_ERROR;
  }

  // --- core::Agent ---
  hsa_status_t IterateRegion(hsa_status_t (*callback)(hsa_region_t, void*),
                             void* data) const override;
  hsa_status_t IterateSupportedIsas(hsa_status_t (*callback)(hsa_isa_t, void*),
                                    void* data) const override;
  hsa_status_t IterateCache(hsa_status_t (*callback)(hsa_cache_t, void*),
                            void* data) const override;

  hsa_status_t QueueCreate(size_t size, hsa_queue_type32_t queue_type, uint64_t flags,
                           core::HsaEventCallback event_callback, void* data,
                           uint32_t private_segment_size, uint32_t group_segment_size,
                           core::Queue** queue) override;
  hsa_status_t DmaCopy(void* dst, const void* src, size_t size) override;
  hsa_status_t DmaCopyStatus(core::Agent& dst_agent, core::Agent& src_agent,
                             uint32_t* engine_ids_mask) override;
  hsa_status_t DmaPreferredEngine(core::Agent& dst_agent, core::Agent& src_agent,
                                  uint32_t* recommended_ids_mask) override;

  hsa_status_t GetInfo(hsa_agent_info_t attribute, void* value) const override;

  void InitDerivedCuid() override {}

  const std::vector<std::shared_ptr<const core::MemoryRegion>>& regions() const override {
    return regions_;
  }
  const std::vector<const core::Isa*>& supported_isas() const override {
    return supported_isas_;
  }

  /// @brief Public-API-compatible alias for KFD's gpu-id hash.
  /// The runtime::RegisterAgent path builds agents_by_gpuid_ keyed on this;
  /// Darwin doesn't have KFD, so just return node_id.
  uint64_t KfdGpuID() const { return node_id(); }

 private:
  HsaNodeProperties node_props_;
  hsa_amd_coherency_type_t current_coherency_type_;
  bool rec_sdma_eng_override_;
  std::vector<std::shared_ptr<const core::MemoryRegion>> regions_;
  std::vector<const core::Isa*> supported_isas_;
};

}  // namespace AMD
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_AMD_MACOS_AGENT_H_
