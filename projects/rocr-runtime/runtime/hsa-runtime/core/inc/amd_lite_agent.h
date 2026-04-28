////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef HSA_RUNTIME_CORE_INC_AMD_LITE_AGENT_H_
#define HSA_RUNTIME_CORE_INC_AMD_LITE_AGENT_H_

#if !defined(__linux__)
#error "amd_lite_agent.h is currently Linux-only"
#endif

#include <memory>
#include <vector>

#include "core/inc/amd_gpu_agent.h"
#include "core/inc/memory_region.h"

namespace rocr {
namespace AMD {

class LiteGpuAgent : public GpuAgentInt {
 public:
  LiteGpuAgent(uint32_t node_id, core::DriverType driver_type);
  ~LiteGpuAgent() override;

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

  void RegisterGangPeer(core::Agent& gang_peer,
                        unsigned int bandwidth_factor) override;
  void RegisterRecSdmaEngIdMaskPeer(core::Agent& gang_peer,
                                    uint32_t rec_sdma_eng_id_mask) override;
  void SetRecSdmaEngOverride(bool flag) override { rec_sdma_eng_override_ = flag; }

  hsa_profile_t profile() const override { return HSA_PROFILE_BASE; }
  uint32_t memory_bus_width() const override { return 0; }
  uint32_t memory_max_frequency() const override { return 0; }

  bool AsyncScratchReclaimEnabled() const override { return false; }
  hsa_status_t SetAsyncScratchThresholds(size_t) override { return HSA_STATUS_ERROR; }

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

  hsa_status_t IterateRegion(hsa_status_t (*callback)(hsa_region_t, void*),
                             void* data) const override;
  hsa_status_t IterateSupportedIsas(hsa_status_t (*callback)(hsa_isa_t, void*),
                                    void* data) const override;
  hsa_status_t IterateCache(hsa_status_t (*callback)(hsa_cache_t, void*),
                            void* data) const override;

  hsa_status_t QueueCreate(size_t size, hsa_queue_type32_t queue_type,
                           uint64_t flags,
                           core::HsaEventCallback event_callback, void* data,
                           uint32_t private_segment_size,
                           uint32_t group_segment_size,
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

#endif  // HSA_RUNTIME_CORE_INC_AMD_LITE_AGENT_H_
