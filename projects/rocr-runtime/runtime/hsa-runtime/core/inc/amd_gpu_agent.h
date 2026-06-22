////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2025, Advanced Micro Devices, Inc. All rights reserved.
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

// AMD specific HSA backend.

#ifndef HSA_RUNTIME_CORE_INC_AMD_GPU_AGENT_H_
#define HSA_RUNTIME_CORE_INC_AMD_GPU_AGENT_H_

#include <atomic>
#include <vector>
#include <list>
#include <map>

#include "hsakmt/hsakmt.h"

#include "core/inc/agent.h"
#include "core/inc/blit.h"
#include "core/inc/cache.h"
#include "core/inc/driver.h"
#include "core/inc/runtime.h"
#include "core/inc/scratch_cache.h"
#include "core/inc/signal.h"
#include "core/util/lazy_ptr.h"
#include "core/util/locks.h"
#include "core/util/os.h"
#include "core/util/small_heap.h"
#include "pcs/pcs_runtime.h"
#include "core/inc/counted_queue_manager.h"

namespace rocr {
namespace AMD {
class MemoryRegion;

typedef ScratchCache::ScratchInfo ScratchInfo;

// @brief Interface to represent a GPU agent.
class GpuAgentInt : public core::Agent {
 public:
  // @brief Constructor
  // @param [in] node_id Node id.
  // @param [in] driver_type Driver type. Default is KFD.
  GpuAgentInt(uint32_t node_id, core::DriverType driver_type)
      : core::Agent(core::Runtime::runtime_singleton_->AgentDriver(driver_type), node_id,
      core::Agent::DeviceType::kAmdGpuDevice) {}

   // @brief Ensure blits are ready (performance hint).
   virtual void PreloadBlits() {}

   // @brief Initialization hook invoked after tools library has loaded,
   // to allow tools interception of interface functions.
   //
   // @retval HSA_STATUS_SUCCESS if initialization is successful.
   virtual hsa_status_t PostToolsInit() = 0;

   virtual void ReleaseResources() = 0;

   // @brief Carve scratch memory for main from scratch pool.
   //
   // @param [in,out] scratch Structure to be populated with the carved memory
   // information.
   virtual void AcquireQueueMainScratch(ScratchInfo &scratch) = 0;

   // @brief Carve scratch memory for alt from scratch pool.
   //
   // @param [in,out] scratch Structure to be populated with the carved memory
   // information.
   virtual void AcquireQueueAltScratch(ScratchInfo &scratch) = 0;

   // @brief Release scratch memory from main back to scratch pool.
   //
   // @param [in,out] scratch Scratch memory previously acquired with call to
   // ::AcquireQueueMainScratch.
   virtual void ReleaseQueueMainScratch(ScratchInfo &base) = 0;

   // @brief Release scratch memory back from alternate to scratch pool.
   //
   // @param [in,out] scratch Scratch memory  previously acquired with call to
   // ::AcquireQueueAltScratch.
   virtual void ReleaseQueueAltScratch(ScratchInfo &base) = 0;

   // @brief Translate the kernel start and end dispatch timestamp from agent
   // domain to host domain.
   //
   // @param [in] signal Pointer to signal that provides the dispatch timing.
   // @param [out] time Structure to be populated with the host domain value.
   virtual void TranslateTime(core::Signal *signal,
                              hsa_amd_profiling_dispatch_time_t &time) = 0;

   // @brief Translate the async copy start and end timestamp from agent
   // domain to host domain.
   //
   // @param [in] signal Pointer to signal that provides the async copy timing.
   // @param [out] time Structure to be populated with the host domain value.
   virtual void TranslateTime(core::Signal *signal,
                              hsa_amd_profiling_async_copy_time_t &time) = 0;

   // @brief Translate timestamp agent domain to host domain.
   //
   // @param [out] time Timestamp in agent domain.
   virtual uint64_t TranslateTime(uint64_t tick) = 0;

   // @brief Invalidate caches on the agent which may hold code object data.
   virtual void InvalidateCodeCaches(void *ptr, size_t size) = 0;

   // @brief Sets the coherency type of this agent.
   //
   // @param [in] type New coherency type.
   //
   // @retval true The new coherency type is set successfuly.
   virtual bool current_coherency_type(hsa_amd_coherency_type_t type) = 0;

   // @brief Returns the current coherency type of this agent.
   //
   // @retval Coherency type.
   virtual hsa_amd_coherency_type_t current_coherency_type() const = 0;

   virtual void RegisterGangPeer(core::Agent &gang_peer,
                                 unsigned int bandwidth_factor) = 0;

   virtual void RegisterRecSdmaEngIdMaskPeer(core::Agent &gang_peer,
                                             uint32_t rec_sdma_eng_id_mask) = 0;

   virtual void SetRecSdmaEngOverride(bool flag) = 0;

   // @brief Query the agent HSA profile.
   //
   // @retval HSA profile.
   virtual hsa_profile_t profile() const = 0;

   // @brief Query the agent memory bus width in bit.
   //
   // @retval Bus width in bit.
   virtual uint32_t memory_bus_width() const = 0;

   // @brief Query the agent memory maximum frequency in MHz.
   //
   // @retval Bus width in MHz.
   virtual uint32_t memory_max_frequency() const = 0;

   // @brief Whether agent supports asynchronous scratch reclaim. Depends on CP
   // FW
   virtual bool AsyncScratchReclaimEnabled() const = 0;

   // @brief Update the agent's scratch use-once threshold.
   // Only valid when async scratch reclaim is supported
   // @retval HSA_STATUS_SUCCESS if successful
   virtual hsa_status_t SetAsyncScratchThresholds(size_t use_once_limit) = 0;

   // @brief Iterate through supported PC Sampling configurations
   // @retval HSA_STATUS_SUCCESS if successful
   virtual hsa_status_t
   PcSamplingIterateConfig(hsa_ven_amd_pcs_iterate_configuration_callback_t cb,
                           void *cb_data) = 0;

   virtual hsa_status_t
   PcSamplingCreate(pcs::PcsRuntime::PcSamplingSession &session) = 0;

   virtual hsa_status_t
   PcSamplingCreateFromId(HsaPcSamplingTraceId pcsId,
                          pcs::PcsRuntime::PcSamplingSession &session) = 0;

   virtual hsa_status_t
   PcSamplingDestroy(pcs::PcsRuntime::PcSamplingSession &session) = 0;

   virtual hsa_status_t
   PcSamplingStart(pcs::PcsRuntime::PcSamplingSession &session) = 0;

   virtual hsa_status_t
   PcSamplingStop(pcs::PcsRuntime::PcSamplingSession &session) = 0;

   virtual hsa_status_t
   PcSamplingFlush(pcs::PcsRuntime::PcSamplingSession &session) = 0;
};

class GpuAgent : public GpuAgentInt {
 public:
  // @brief GPU agent constructor.
  //
  // @param [in] node Node id. Each CPU in different socket will get distinct
  // id.
  // @param [in] node_props Node property.
  // @param [in] xnack_mode XNACK mode of device.
  // @param [in] index Index of the GPU device.
  // @param [in] driver_type Driver type. Default is KFD.
  GpuAgent(HSAuint32 node, const HsaNodeProperties& node_props, bool xnack_mode, uint32_t index,
           core::DriverType driver_type = core::DriverType::KFD);

  // @brief GPU agent destructor.
  ~GpuAgent();

  // @brief Release allocated resources and disables agent
  void ReleaseResources() override;

  // @brief Ensure blits are ready (performance hint).
  void PreloadBlits() override;

  // @brief Override from core::Agent.
  hsa_status_t PostToolsInit() override;

  uint16_t GetMicrocodeVersion() const;

  uint16_t GetSdmaMicrocodeVersion() const;

  // @brief Assembles SP3 shader source into ISA or AQL code object.
  //
  // @param [in] src_sp3 SP3 shader source text representation.
  // @param [in] func_name Name of the SP3 function to assemble.
  // @param [in] assemble_target ISA or AQL assembly target.
  // @param [out] code_buf Code object buffer.
  // @param [out] code_buf_size Size of code object buffer in bytes.
  enum class AssembleTarget { ISA, AQL };

  void AssembleShader(const char* func_name, AssembleTarget assemble_target, void*& code_buf,
                      size_t& code_buf_size) const;

  // @brief Frees code object created by AssembleShader.
  //
  // @param [in] code_buf Code object buffer.
  // @param [in] code_buf_size Size of code object buffer in bytes.
  void ReleaseShader(void* code_buf, size_t code_buf_size) const;

  // @brief Override from core::Agent.
  hsa_status_t VisitRegion(bool include_peer,
                           hsa_status_t (*callback)(hsa_region_t region,
                                                    void* data),
                           void* data) const override;

  // @brief Override from core::Agent.
  hsa_status_t IterateRegion(hsa_status_t (*callback)(hsa_region_t region,
                                                      void* data),
                             void* data) const override;

  hsa_status_t IterateSupportedIsas(
                    hsa_status_t (*callback)(hsa_isa_t isa, void* data),
                                                  void* data) const override;

  // @brief Override from core::Agent.
  hsa_status_t IterateCache(hsa_status_t (*callback)(hsa_cache_t cache, void* data),
                            void* value) const override;

  // @brief Override from core::Agent.
  hsa_status_t DmaCopy(void* dst, const void* src, size_t size) override;

  // @brief Override from core::Agent.
  hsa_status_t DmaCopy(void* dst, core::Agent& dst_agent, const void* src,
                       core::Agent& src_agent, size_t size,
                       std::vector<core::Signal*>& dep_signals,
                       core::Signal& out_signal) override;

  // @brief Override from core::Agent.
  hsa_status_t DmaCopyOnEngine(void* dst, core::Agent& dst_agent, const void* src,
                       core::Agent& src_agent, size_t size,
                       std::vector<core::Signal*>& dep_signals,
                       core::Signal& out_signal, int engine_offset,
                       bool force_copy_on_sdma) override;

  // @brief Override from core::Agent.
  hsa_status_t DmaCopyStatus(core::Agent& dst_agent, core::Agent& src_agent,
                             uint32_t *engine_ids_mask) override;

  // @brief Override from core::Agent.
  hsa_status_t DmaPreferredEngine(core::Agent& dst_agent, core::Agent& src_agent,
                                  uint32_t* recommended_ids_mask) override;

  // @brief Pick an SDMA engine from a preferred-engine mask using round-robin.
  // Returns a blit object index (1-indexed, suitable for GetBlitObject), or 0
  // if mask is empty. When advance=true (default) the counter is incremented so
  // successive body assignments spread across engines. Pass advance=false to
  // peek at the current position without consuming a slot — used when selecting
  // a coordinator engine so it rotates independently of body assignments.
  inline uint32_t PickSdmaEngine(uint32_t engine_mask, bool advance = true) {
    if (!engine_mask) return 0;
    int count = rocr::os::Popcount(engine_mask);
    if (count == 1) return rocr::os::Ffs(engine_mask);
    uint32_t rr = advance ? sdma_rr_index_.fetch_add(1, std::memory_order_relaxed)
                          : sdma_rr_index_.load(std::memory_order_relaxed);
    uint32_t m = engine_mask;
    for (uint32_t i = 0, pick = rr % count; i < pick; ++i)
      m &= m - 1;
    return rocr::os::Ffs(m);
  }

  // @brief Override from core::Agent.
  hsa_status_t DmaCopyBatch(const hsa_amd_memory_copy_op_t* ops,
                            uint32_t num_ops,
                            std::vector<core::Signal*>& dep_signals) override;

  // @brief Override from core::Agent.
  hsa_status_t DmaCopyRect(const hsa_pitched_ptr_t* dst, const hsa_dim3_t* dst_offset,
                           const hsa_pitched_ptr_t* src, const hsa_dim3_t* src_offset,
                           const hsa_dim3_t* range, hsa_amd_copy_direction_t dir,
                           std::vector<core::Signal*>& dep_signals, core::Signal& out_signal);

  // @brief Override from core::Agent.
  hsa_status_t DmaFill(void* ptr, uint32_t value, size_t count) override;

  // @brief Override from core::Agent.
  hsa_status_t GetInfo(hsa_agent_info_t attribute, void* value) const override;

  // @brief Override from core::Agent.
  hsa_status_t QueueCreate(size_t size, hsa_queue_type32_t queue_type, uint64_t flags,
                           core::HsaEventCallback event_callback, void* data,
                           uint32_t private_segment_size, uint32_t group_segment_size,
                           bool metadata_queue, core::Queue** queue) override;

  // @brief Decrement GWS ref count.
  void GWSRelease();

  // @brief Override from AMD::GpuAgentInt.
  void AcquireQueueMainScratch(ScratchInfo& scratch) override;
  void ReleaseQueueMainScratch(ScratchInfo& scratch) override;

  void AcquireQueueAltScratch(ScratchInfo& scratch) override;
  void ReleaseQueueAltScratch(ScratchInfo& scratch) override;

  // @brief Create a pool of shared queues for multiple user applications within a max limit
  hsa_status_t AcquireCountedQueue(hsa_queue_type_t type,
                                   HSA::hsa_amd_queue_priority_internal_t priority,
                                   void (*callback)(hsa_status_t, hsa_queue_t*, void*),
                                   void* data, uint64_t flags,
                                   hsa_queue_t** out_queue);

  // @brief Release a queue earlier used by application
  hsa_status_t ReleaseCountedQueue(hsa_queue_t* queue);

  // @brief Override from AMD::GpuAgentInt.
  void TranslateTime(core::Signal* signal, hsa_amd_profiling_dispatch_time_t& time) override;

  // @brief Override from AMD::GpuAgentInt.
  void TranslateTime(core::Signal* signal, hsa_amd_profiling_async_copy_time_t& time) override;

  // @brief Override from AMD::GpuAgentInt.
  uint64_t TranslateTime(uint64_t tick) override;

  // @brief Override from AMD::GpuAgentInt.
  void InvalidateCodeCaches(void* ptr, size_t size) override;

  // @brief Override from AMD::GpuAgentInt.
  bool current_coherency_type(hsa_amd_coherency_type_t type) override;

  hsa_amd_coherency_type_t current_coherency_type() const override {
    return current_coherency_type_;
  }

  hsa_status_t Preload(uint64_t flags);

  core::Agent* GetNearestCpuAgent() const override;

  void RegisterGangPeer(core::Agent& gang_peer, unsigned int bandwidth_factor) override;

  void RegisterRecSdmaEngIdMaskPeer(core::Agent& gang_peer, uint32_t rec_sdma_eng_id_mask) override;

  // Getter & setters.

  // @brief Returns Hive ID
  __forceinline uint64_t HiveId() const override { return  properties_.HiveID; }

  // @brief Returns KFD's GPU id which is a hash used internally.
  __forceinline uint64_t KfdGpuID() const { return properties_.KFDGpuID; }

  // @brief Returns node property.
  __forceinline const HsaNodeProperties& properties() const {
    return properties_;
  }

  // @brief set rec_sdma_eng_override_
  __forceinline void SetRecSdmaEngOverride(bool flag) override { rec_sdma_eng_override_ = flag; }

  // @brief Returns number of data caches.
  __forceinline size_t num_cache() const { return cache_props_.size(); }

  // @brief Returns data cache property.
  //
  // @param [in] idx Cache level.
  __forceinline const HsaCacheProperties& cache_prop(int idx) const {
    return cache_props_[idx];
  }

  // @brief Override from core::Agent.
  const std::vector<std::shared_ptr<const core::MemoryRegion>>& regions() const override {
    return regions_;
  }

  const std::vector<const core::Isa *>& supported_isas() const override {
                                                      return supported_isas_;}

  // @brief Override from AMD::GpuAgentInt.
  __forceinline hsa_profile_t profile() const override { return profile_; }

  // @brief Override from AMD::GpuAgentInt.
  __forceinline uint32_t memory_bus_width() const override {
    return memory_bus_width_;
  }

  // @brief Override from AMD::GpuAgentInt.
  __forceinline uint32_t memory_max_frequency() const override {
    return memory_max_frequency_;
  }

  // @brief Order the device is surfaced in hsa_iterate_agents counting only
  // GPU devices.
  __forceinline uint32_t enumeration_index() const { return enum_index_; }

  // @brief returns true if agent uses MES scheduler
  __forceinline const bool isMES() const { return (supported_isas()[0]->GetMajorVersion() >= 11) ? true : false; };

  // @brief returns the libdrm device handle
  __forceinline amdgpu_device_handle libDrmDev() const { return ldrm_dev_; }
  __forceinline HsaAMDGPUDeviceHandle libThunkDev() const { return libthunk_dev_; }

  __forceinline void CheckClockTicks() {
    // If we did not update t1 since agent initialization, force a SyncClock. Otherwise computing
    // the SystemClockCounter to GPUClockCounter ratio in TranslateTime(tick) results to a division
    // by 0.
    if (t0_.GPUClockCounter == t1_.GPUClockCounter) SyncClocks();
  }

  /// @brief Override from AMD::GpuAgentInt.
  __forceinline bool is_xgmi_cpu_gpu() const { return xgmi_cpu_gpu_; }
  /// @brief Is large BAR support enabled for this GPU.
  __forceinline bool LargeBarEnabled() const { return large_bar_enabled_; }

  /// @brief Force a WC flush on PCIe devices by doing a write and then read-back
  __forceinline void PcieWcFlush(void *ptr, size_t size) const {
    if (!xgmi_cpu_gpu_) {
      _mm_sfence();
      *((uint8_t*)ptr + size - 1) = *((uint8_t*)ptr + size - 1);
      _mm_mfence();
      auto readback = *(reinterpret_cast<volatile uint8_t*>(ptr) + size - 1);
      UNUSED(readback);
    }
  }

  const size_t MAX_SCRATCH_APERTURE_PER_XCC = (1ULL << 32); // 4GB
  const size_t MAX_SCRATCH_APERTURE_PER_XCC_GFX12 = (2ULL << 32); // 8GB
  __forceinline size_t MaxScratchDevice() const {
    return properties_.NumXcc *
          (supported_isas()[0]->GetMajorVersion() >= 12 ? MAX_SCRATCH_APERTURE_PER_XCC_GFX12 :
                                                           MAX_SCRATCH_APERTURE_PER_XCC);
  }

  void ReserveScratch();

  // @brief If agent supports it, release scratch memory for all AQL queues on this agent.
  void AsyncReclaimScratchQueues();

  // @brief Returns true if scratch reclaim is enabled
  __forceinline bool AsyncScratchReclaimEnabled() const override {
    const uint32_t GFX94X_MIN_CP_FW_VERSION_REQUIRED = 177;
    const uint32_t GFX95X_MIN_CP_FW_VERSION_REQUIRED = 24;

    if (!core::Runtime::runtime_singleton_->flag().enable_scratch_async_reclaim())
      return false;

    switch (supported_isas()[0]->GetMajorVersion()) {
      case 9:
        switch (supported_isas()[0]->GetMinorVersion()) {
          case 4:
            return (properties_.EngineId.ui32.uCode >= GFX94X_MIN_CP_FW_VERSION_REQUIRED);
          case 5:
            return (properties_.EngineId.ui32.uCode >= GFX95X_MIN_CP_FW_VERSION_REQUIRED);
          default:
            break;
        }
        break;
      case 12:
        return (supported_isas()[0]->GetMinorVersion() >= 5);
      default:
        break;
    }

    return false;
  };

  hsa_status_t SetAsyncScratchThresholds(size_t use_once_limit) override;

  __forceinline size_t ScratchSingleLimitAsyncThreshold() const {
    return scratch_limit_async_threshold_;
  }

  void Trim() override;

  const std::function<void*(size_t size, size_t align, core::MemoryRegion::AllocateFlags flags)>&
  system_allocator() const {
    return system_allocator_;
  }

  const std::function<void(void*)>& system_deallocator() const { return system_deallocator_; }

  const std::function<void*(size_t size, core::MemoryRegion::AllocateFlags flags)>&
  finegrain_allocator() const {
    return finegrain_allocator_;
  }

  const std::function<void(void*)>& finegrain_deallocator() const { return finegrain_deallocator_; }

  /// @brief Allocate coarse grain device memory on this GPU agent.
  const std::function<void*(size_t size, core::MemoryRegion::AllocateFlags flags)>&
  coarsegrain_allocator() const {
    return coarsegrain_allocator_;
  }

  /// @brief Deallocate memory allocated from the coarsegrain_allocator
  /// on this GPU agent.
  const std::function<void(void*)>& coarsegrain_deallocator() const {
    return coarsegrain_deallocator_;
  }

  /// @brief Get scratch memory base address and size for core dump filtering
  void GetScratchAperture(void** base, size_t* size) const {
    *base = scratch_pool_.base();
    *size = scratch_pool_.size();
  }

  /// @brief Get a snapshot of AQL queues for core dump filtering.
  /// Returns a copy to avoid iterator invalidation from concurrent queue destruction.
  std::vector<core::Queue*> GetAqlQueues() const {
    std::lock_guard<std::mutex> lock(aql_queues_lock_);
    return aql_queues_;
  }

  /// @brief Remove a destroyed AQL queue from agent-owned tracking.
  void UnregisterAqlQueue(core::Queue* queue);

 protected:
  // Sizes are in packets.
  const uint32_t minAqlSize_ = 0x40;     // 4KB min
  const uint32_t maxAqlSize_ = 0x20000;  // 8MB max

  // @brief Create an internal queue allowing tools to be notified.
  core::Queue* CreateInterceptibleQueue(bool metadata_prefetch, const uint32_t size = 0) {
    return CreateInterceptibleQueue(core::Queue::DefaultErrorHandler, nullptr, metadata_prefetch, size);
  }

  // @brief Create an internal queue, with a custom error handler, allowing tools to be
  // notified.
  core::Queue* CreateInterceptibleQueue(void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data),
                                        void* data, bool metadata_prefetch, const uint32_t size);

  // @brief Create SDMA blit object.
  //
  // @retval NULL if SDMA blit creation and initialization failed.
  core::Blit* CreateBlitSdma(bool use_xgmi, int rec_eng);

  // @brief Create Kernel blit object using provided compute queue.
  //
  // @retval NULL if Kernel blit creation and initialization failed.
  core::Blit* CreateBlitKernel(core::Queue* queue);

  // @brief Invoke the user provided callback for every region in @p regions.
  //
  // @param [in] regions Array of region object.
  // @param [in] callback User provided callback function.
  // @param [in] data User provided pointer as input for @p callback.
  //
  // @retval ::HSA_STATUS_SUCCESS if the callback function for each traversed
  // region returns ::HSA_STATUS_SUCCESS.
  hsa_status_t VisitRegion(
      const std::vector<std::shared_ptr<const core::MemoryRegion>>& regions,
      hsa_status_t (*callback)(hsa_region_t region, void* data),
      void* data) const;

  // @brief Update ::t1_ tick count.
  void SyncClocks();

  // @brief Binds the second-level trap handler to this node.
  void BindTrapHandler();

  // @brief Override from core::Agent.
  hsa_status_t EnableDmaProfiling(bool enable) override;

  hsa_status_t PcSamplingIterateConfig(hsa_ven_amd_pcs_iterate_configuration_callback_t cb,
                                       void* cb_data) override;
  hsa_status_t PcSamplingCreate(pcs::PcsRuntime::PcSamplingSession& session) override;
  hsa_status_t PcSamplingCreateFromId(HsaPcSamplingTraceId pcsId,
                            pcs::PcsRuntime::PcSamplingSession& session) override;
  hsa_status_t PcSamplingDestroy(pcs::PcsRuntime::PcSamplingSession& session) override;
  hsa_status_t PcSamplingStart(pcs::PcsRuntime::PcSamplingSession& session) override;
  hsa_status_t PcSamplingStop(pcs::PcsRuntime::PcSamplingSession& session) override;
  hsa_status_t PcSamplingFlush(pcs::PcsRuntime::PcSamplingSession& session) override;
  hsa_status_t PcSamplingFlushDeviceBuffers(pcs::PcsRuntime::PcSamplingSession& session);

  // @brief Node properties.
  const HsaNodeProperties properties_;

  // @brief Current coherency type.
  hsa_amd_coherency_type_t current_coherency_type_;

  // @brief Maximum number of queues that can be created.
  uint32_t max_queues_;

  // @brief Object to manage scratch memory.
  SmallHeap scratch_pool_;

  // @brief Current short duration scratch memory size.
  size_t scratch_used_large_;

  struct HsaSignalLess {
    bool operator()(const hsa_signal_t& lhs, const hsa_signal_t& rhs) const {
      return lhs.handle < rhs.handle;
    }
  };

  // @brief Notifications for scratch release.
  std::map<hsa_signal_t, hsa_signal_value_t, HsaSignalLess> scratch_notifiers_;

  // @brief Default scratch size per queue.
  size_t queue_scratch_len_;

  // @brief Default scratch size per work item.
  size_t scratch_per_thread_;

  // @brief Blit interfaces for each data path.
  enum BlitEnum { BlitDevToDev, BlitHostToDev, BlitDevToHost, DefaultBlitCount };

  // Blit objects managed by an instance of GpuAgent
  std::vector<lazy_ptr<core::Blit>> blits_;

  // List of agents connected via xGMI
  std::vector<const core::Agent*> xgmi_peer_list_;

  // Protects xgmi_peer_list_
  std::mutex xgmi_peer_list_lock_;

  // @brief Number of H2D/D2H engines
  // On systems with more than 2 SDMA engines, this is capped at 2.
  size_t num_h2d_d2h_engines_;

  // @brief Number of P2P engines
  // P2P engines are used for P2P copies between two GPUs.
  // On platforms with xGMI, these are the xGMI engines.
  // On platforms with more than 2 SDMA engines, these are the SDMA engines.
  size_t num_p2p_engines_;


  // @brief AQL queues for cache management and blit compute usage.
  enum QueueEnum {
    QueueUtility,     // Cache management and device to {host,device} blit compute
    QueueBlitOnly,    // Host to device blit
    QueuePCSampling,  // Dedicated high priority queue for PC Sampling
    QueueCount
  };

  lazy_ptr<core::Queue> queues_[QueueCount];

  // @brief Mutex to protect the update to coherency type.
  std::mutex coherency_lock_;

  // @brief Mutex to protect access to scratch pool.
  std::mutex scratch_lock_;

  // @brief Mutex to protect access to ::t1_.
  std::mutex t1_lock_;

  // @brief Mutex to protect access to blit objects.
  std::mutex blit_lock_;

  // @brief Mutex to protect sdma gang submissions.
  std::mutex sdma_gang_lock_;

  // @brief GPU tick on initialization.
  HsaClockCounters t0_;

  HsaClockCounters t1_;

  double historical_clock_ratio_;

  // @brief Offset (in GPU ticks) between AQL dispatch timestamp clock and
  // D3DKMTQueryClockCalibration GPU clock.  Non-zero on Windows where the two
  // GPU clock domains share frequency but differ in epoch.  Computed on first
  // TranslateTime call; zero on Linux (clocks match).
  int64_t gpu_clock_offset_;

  // @brief s_memrealtime nominal clock frequency
  uint64_t wallclock_frequency_;

  // @brief Array of GPU cache property.
  std::vector<HsaCacheProperties> cache_props_;

  // @brief Array of HSA cache objects.
  std::vector<std::unique_ptr<core::Cache>> caches_;

  // @brief Array of regions owned by this agent.
  std::vector<std::shared_ptr<const core::MemoryRegion>> regions_;

  // @brief HSA profile.
  hsa_profile_t profile_;

  // @brief Pool of shared queues owned by this agent
  rocr::core::CountedQueuePoolManager queue_pool_;

  // @brief /// Cached derived CUID for this GPU agent (16 bytes, zeroed if unavailable).
  uint8_t derived_cuid_[16] = {};

  void* trap_code_buf_;

  size_t trap_code_buf_size_;

  // @brief Mappings from doorbell index to queue, for trap handler.
  // Correlates with output of s_sendmsg(MSG_GET_DOORBELL) for queue identification.
  amd_queue_v2_t** doorbell_queue_map_;

  // @brief The GPU memory bus width in bit.
  uint32_t memory_bus_width_;

  // @brief The GPU memory maximum frequency in MHz.
  uint32_t memory_max_frequency_;

  // @brief Enumeration index
  uint32_t enum_index_;

  // @brief HDP flush registers
  hsa_amd_hdp_flush_t HDP_flush_ = {nullptr, nullptr};

 private:
  // @brief Query the driver to get the region list owned by this agent.
  void InitRegionList();

  // @brief Reserve memory for scratch pool to be used by AQL queue of this
  // agent.
  void InitScratchPool();

  // @brief Query the driver to get the cache properties.
  void InitCacheList();

  // @brief Create internal queues and blits.
  void InitDma();

  // @brief Setup GWS accessing queue.
  void InitGWS();

  // @brief Set-up memory allocators
  void InitAllocators();

  // @brief Initialize scratch handler thresholds
  void InitAsyncScratchThresholds();

  // @brief Initialize Secondary CUID for GPU device that
  // this agent is running on.
  void InitDerivedCuid() override;

  // @brief Register signal for notification when scratch may become available.
  // @p signal is notified by OR'ing with @p value.
  bool AddScratchNotifier(hsa_signal_t signal, hsa_signal_value_t value) {
    if (signal.handle == 0) return false;
    scratch_notifiers_[signal] = value;
    return true;
  }

  // @brief Deregister scratch notification signals.
  void ClearScratchNotifiers() { scratch_notifiers_.clear(); }

  // @brief Releases scratch back to the driver.
  // caller must hold scratch_lock_.
  void ReleaseScratch(void* base, size_t size, bool large);

  // Broadcast copy: copies op.src to each destination in op.dst_list.
  // Uses HW broadcast for transfers < 1 MB when supported; otherwise falls
  // back to prologue/body/epilogue fan-out across available SDMA engines.
  hsa_status_t DmaCopyBroadcast(
      const hsa_amd_memory_copy_op_t& op,
      std::vector<core::Signal*>& dep_signals);

  // Multi-linear copy: LINEAR op with num_entries > 0, independent copies
  // (different src/dst/size per entry) sharing a single completion signal.
  // Uses prologue/body/epilogue fan-out across available SDMA engines.
  hsa_status_t DmaCopyMulti(
      const hsa_amd_memory_copy_op_t& op,
      std::vector<core::Signal*>& dep_signals);

  // Linear swap: exchanges the contents of src and dst buffers.
  // Only supported on gfx94X / gfx95X.  Uses DmaCopyFanOutOp with
  // HSA_AMD_MEMORY_COPY_OP_LINEAR_SWAP.
  hsa_status_t DmaCopySwap(
      const hsa_amd_memory_copy_op_t& op,
      std::vector<core::Signal*>& dep_signals);

  // Indirect copy: src and/or dst is a pointer-to-pointer slot that the SDMA
  // engine dereferences just before performing the transfer.  Whatever fills
  // the slot (e.g. a kernel or a host-side write) is expected to synchronize
  // with this op through `dep_signals`: `dep_signals[0]` maps to the packet's
  // hardware WAIT field and the remaining entries are emitted as 64-bit poll
  // commands ahead of the copy.  The runtime itself therefore does not need
  // to know how, or by whom, the slot gets populated.
  hsa_status_t DmaCopyIndirect(
      const hsa_amd_memory_copy_op_t& op,
      std::vector<core::Signal*>& dep_signals);

  // Common fan-out implementation shared by DmaCopyBroadcast, DmaCopyMulti,
  // swap and indirect operations.  Submits prologue, per-entry bodies
  // (selected by @p op), and epilogue with one signal.
  // @p op is the hsa_amd_memory_copy_op_type_t from the public API;
  // currently supported values are HSA_AMD_MEMORY_COPY_OP_LINEAR,
  // HSA_AMD_MEMORY_COPY_OP_LINEAR_SWAP and
  // HSA_AMD_MEMORY_COPY_OP_LINEAR_INDIRECT_{SRC,DST,SRCDST}.
  hsa_status_t DmaCopyFanOutOp(
      hsa_amd_memory_copy_op_type_t op,
      core::Signal& out_signal,
      std::vector<core::Signal*>& dep_signals,
      uint16_t num_entries,
      const void* const* src_list,
      void* const* dst_list,
      const hsa_agent_t* dst_agent_list,
      const size_t* size_list);

  // Bind index of peer device that is connected via xGMI links
  lazy_ptr<core::Blit>& GetXgmiBlit(const core::Agent& peer_agent);

  // Bind the Blit object that will drive the copy operation
  // across PCIe links (H2D or D2H) or is within same device D2D
  lazy_ptr<core::Blit>& GetPcieBlit(const core::Agent& dst_agent, const core::Agent& src_agent);

  // Bind the Blit object that will drive the copy operation
  lazy_ptr<core::Blit>& GetBlitObject(const core::Agent& dst_agent, const core::Agent& src_agent,
                                      const size_t size);

  // Bind the Blit object that will drive the copy operation by engine ID
  lazy_ptr<core::Blit>& GetBlitObject(uint32_t engine_id);

  // @brief initialize libdrm handle
  void InitLibDrm();

  void GetInfoMemoryProperties(uint8_t value[8]) const;

  void GetAqlInfoProperties(uint8_t value[8]) const;

  // @brief Alternative aperture base address. Only on KV.
  uintptr_t ape1_base_;

  // @brief Queue with GWS access.
  struct {
    lazy_ptr<core::Queue> queue_;
    int ref_ct_;
    std::mutex lock_;
  } gws_queue_;

  // @brief list of AQL queues owned by this agent. Indexed by queue pointer
  std::vector<core::Queue*> aql_queues_;

  // @brief Protects aql_queues_ from concurrent modification/iteration.
  mutable std::mutex aql_queues_lock_;

  // Sets and Tracks pending SDMA status check or request counts
  void SetCopyRequestRefCount(bool set);
  void SetCopyStatusCheckRefCount(bool set);
  std::atomic<int> pending_copy_req_ref_;
  std::atomic<int> pending_copy_stat_check_ref_;

  // Tracks what SDMA blits have been used since initialization.
  uint32_t sdma_blit_used_mask_;

  // Scratch limit thresholds when async scratch is enabled.
  uint64_t scratch_limit_async_threshold_;

  ScratchCache scratch_cache_;

  /// @brief System memory allocator in the nearest NUMA node.
  std::function<void*(size_t size, size_t align, core::MemoryRegion::AllocateFlags flags)>
      system_allocator_;
  /// @brief System memory deallocator in the nearest NUMA node.
  std::function<void(void*)> system_deallocator_;
  /// @brief Fine-grain allocator on this GPU.
  std::function<void*(size_t size, core::MemoryRegion::AllocateFlags flags)> finegrain_allocator_;
  /// @brief Fine-grain deallocator on this GPU.
  std::function<void(void*)> finegrain_deallocator_;
  /// @brief Coarse-grain allocator on this GPU.
  std::function<void*(size_t size, core::MemoryRegion::AllocateFlags flags)> coarsegrain_allocator_;
  /// @brief Coarse-grain deallocator on this GPU.
  std::function<void(void*)> coarsegrain_deallocator_;

  void* trap_handler_tma_region_;

  /* PC Sampling fields - begin */
  /* 2nd level Trap handler code is based on the offsets within this structure */
  typedef struct {
    uint64_t buf_write_val;
    uint32_t buf_size;
    uint32_t reserved0;
    uint32_t buf_written_val0;
    uint32_t buf_watermark0;
    hsa_signal_t done_sig0;
    uint32_t buf_written_val1;
    uint32_t buf_watermark1;
    hsa_signal_t done_sig1;
    uint8_t reserved1[16];
    /* pc_sample_t buffer0[buf_size]; */
    /* pc_sample_t buffer1[buf_size]; */
  } pcs_sampling_data_t;

  typedef struct {
    /* Sampling data - stored on device for trap handler access */
    pcs_sampling_data_t* device_data;

    /* Sampling host buffer - stored on host */
    uint8_t* host_buffer;
    size_t host_buffer_size;
    uint8_t* host_buffer_wrap_pos;
    uint8_t* host_write_ptr;
    uint8_t* host_read_ptr;
    size_t lost_sample_count;
    std::mutex host_buffer_mutex;

    uint32_t which_buffer;
    uint64_t* old_val;
    uint32_t* cmd_data;
    size_t cmd_data_sz;
    // signal to pass into ExecutePM4() so that we do not need to re-allocate a
    // new signal on each call
    hsa_signal_t exec_pm4_signal;

    // Host-side copies - cannot read these from device_data on non-large BAR systems
    hsa_signal_t done_sig0;
    hsa_signal_t done_sig1;
    uint32_t buf_size;

    os::Thread thread;
    pcs::PcsRuntime::PcSamplingSession* session;
  } pcs_data_t;
  /* PC Sampling fields - end */

  hsa_status_t UpdateTrapHandlerWithPCS(pcs_sampling_data_t* pcs_hosttrap_buffers,
                                        pcs_sampling_data_t* pcs_stochastic_buffers);

  // @brief Thread function to process PC sampling data collected via host-trap
  // or Stochastic sampling.
  void PcSamplingThread(pcs_data_t& pcs_data, const char* thread_name);

  // @brief device handle
  amdgpu_device_handle ldrm_dev_;
  HsaAMDGPUDeviceHandle libthunk_dev_;

  // Check if SDMA engine by ID is free
  bool DmaEngineIsFree(uint32_t engine_id);

  std::map<uint64_t,unsigned int> gang_peers_info_;

  std::map<uint64_t, uint32_t> rec_sdma_eng_id_peers_info_;

  bool uses_rec_sdma_eng_id_mask_;
  bool rec_sdma_eng_override_;

  // Round-robin index for spreading SDMA work across engines (gfx1250+).
  std::atomic<uint32_t> sdma_rr_index_{0};

  // structure for host trap sampling
  pcs_data_t pcs_hosttrap_data_;

  // structure for stochastic sampling
  pcs_data_t pcs_stochastic_data_;

  /// @brief XGMI CPU<->GPU
  bool xgmi_cpu_gpu_;
  /// @brief Is PCIe large BAR enabled.
  bool large_bar_enabled_;

  bool extended_aql_dispatch_supported_;

  /* Workgroup Cluster Parameters */
  bool workgroup_clusters_supported_;
  hsa_amd_dim3_t kern_cluster_max_dim_;
  hsa_amd_dim3_t cluster_max_dim_;

  size_t max_wave_scratch_;

  DISALLOW_COPY_AND_ASSIGN(GpuAgent);
};

}  // namespace amd
}  // namespace rocr

#endif  // header guard
