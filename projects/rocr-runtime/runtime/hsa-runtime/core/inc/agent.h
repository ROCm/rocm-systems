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

// HSA runtime C++ interface file.

#ifndef HSA_RUNTME_CORE_INC_AGENT_H_
#define HSA_RUNTME_CORE_INC_AGENT_H_

#include <assert.h>
#include <vector>

#include "core/inc/checked.h"
#include "core/inc/isa.h"
#include "core/inc/memory_region.h"
#include "core/inc/queue.h"
#include "core/util/locks.h"
#include "core/util/utils.h"

namespace rocr {

// Forward declare AMD::MemoryRegion
namespace AMD {
class MemoryRegion;
}

namespace core {
class Driver;
class Signal;

typedef void (*HsaEventCallback)(hsa_status_t status, hsa_queue_t* source,
                                 void* data);

// Agent is intended to be an pure interface class and may be wrapped or
// replaced by tools libraries. All funtions other than Convert, node_id,
// device_type, and public_handle must be virtual.
class Agent : public Checked<0xF6BC25EB17E6F917> {
  friend class rocr::AMD::MemoryRegion;

 public:
  // @brief Convert agent object into hsa_agent_t.
  //
  // @param [in] agent Pointer to an agent.
  //
  // @retval hsa_agent_t
  static __forceinline hsa_agent_t Convert(Agent* agent) {
    const hsa_agent_t agent_handle = {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(agent))};
    return agent_handle;
  }

  // @brief Convert agent object into const hsa_agent_t.
  //
  // @param [in] agent Pointer to an agent.
  //
  // @retval const hsa_agent_t
  static __forceinline const hsa_agent_t Convert(const Agent* agent) {
    const hsa_agent_t agent_handle = {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(agent))};
    return agent_handle;
  }

  // @brief Convert hsa_agent_t handle into Agent*.
  //
  // @param [in] agent An hsa_agent_t handle.
  //
  // @retval Agent*
  static __forceinline Agent* Convert(hsa_agent_t agent) {
    return reinterpret_cast<Agent*>(agent.handle);
  }

  // Lightweight RTTI for vendor specific implementations.
  enum DeviceType {
    kAmdGpuDevice = 0,
    kAmdCpuDevice = 1,
    kAmdAieDevice = 2,
    kUnknownDevice = 3
  };

  // @brief Agent class contructor.
  //
  // @param [in] type CPU or GPU or other.
  explicit Agent(Driver &driver, uint32_t node_id, DeviceType type)
      : node_id_(node_id), device_type_(uint32_t(type)), driver_(&driver),
        profiling_enabled_(false), enabled_(false) {
    public_handle_ = Convert(this);
  }

  // @brief Agent class destructor.
  virtual ~Agent() {}

  // @brief Submit DMA copy command to move data from src to dst and wait
  // until it is finished.
  //
  // @details The agent must be able to access @p dst and @p src.
  //
  // @param [in] dst Memory address of the destination.
  // @param [in] src Memory address of the source.
  // @param [in] size Copy size in bytes.
  //
  // @retval HSA_STATUS_SUCCESS The memory copy is finished and successful.
  virtual hsa_status_t DmaCopy(void* dst, const void* src, size_t size) {
    return HSA_STATUS_ERROR;
  }

  // @brief Submit DMA copy command to move data from src to dst. This call
  // does not wait until the copy is finished
  //
  // @details The agent must be able to access @p dst and @p src. Memory copy
  // will be performed after all signals in @p dep_signals have value of 0.
  // On memory copy completion, the value of out_signal is decremented.
  //
  // @param [in] dst Memory address of the destination.
  // @param [in] dst_agent Agent that owns the memory pool associated with @p
  // dst.
  // @param [in] src Memory address of the source.
  // @param [in] src_agent Agent that owns the memory pool associated with @p
  // src.
  // @param [in] size Copy size in bytes.
  // @param [in] dep_signals Array of signal dependency.
  // @param [in] out_signal Completion signal.
  //
  // @retval HSA_STATUS_SUCCESS The memory copy is finished and successful.
  virtual hsa_status_t DmaCopy(void* dst, core::Agent& dst_agent,
                               const void* src, core::Agent& src_agent,
                               size_t size,
                               std::vector<core::Signal*>& dep_signals,
                               core::Signal& out_signal) {
    return HSA_STATUS_ERROR;
  }

  // @brief Submit DMA copy command to move data from src to dst on engine_id.
  // This call does not wait until the copy is finished
  //
  // @details All semantics and params are identical to DmaCopy except for engine_id.
  //
  // @param [in] engine_offset Target engine
  // @param [in] force_copy_on_sdma By default, blit kernel copies are used if
  // dst_agent == src_agent.  Setting this true forces the copy over SDMA1.
  //
  //
  // @retval HSA_STATUS_SUCCESS The memory copy is finished and successful.
  virtual hsa_status_t DmaCopyOnEngine(void* dst, core::Agent& dst_agent,
                               const void* src, core::Agent& src_agent,
                               size_t size,
                               std::vector<core::Signal*>& dep_signals,
                               core::Signal& out_signal,
                               int engine_offset,
                               bool force_copy_on_sdma) {
    return HSA_STATUS_ERROR;
  }

  // @brief Return DMA availability status for copy direction.
  //
  // @param [in] dst_agent Destination agent.
  // @param [in] src_agent Source agent.
  // @param [out] engine_ids_mask Mask of engine ids.
  //
  // @retval HSA_STATUS_SUCCESS DMA engines are available
  // @retval HSA_STATUS_ERROR_OUT_OF_RESOURCES DMA engines are not available
  virtual hsa_status_t DmaCopyStatus(core::Agent& dst_agent, core::Agent& src_agent,
                                     uint32_t *engine_ids_mask) {
    return HSA_STATUS_ERROR;
  }

  // @brief Return DMA availability status for copy direction.
  //
  // @param [in] dst_agent Destination agent.
  // @param [in] src_agent Source agent.
  // @param [out] recommended_ids_mask Mask of recommended engine ids.
  //
  // @retval HSA_STATUS_SUCCESS For mask returned
  virtual hsa_status_t DmaPreferredEngine(core::Agent& dst_agent, core::Agent& src_agent,
                                          uint32_t* recommended_ids_mask) {
    return HSA_STATUS_ERROR;
  }

  // @brief Submit DMA command to set the content of a pointer and wait
  // until it is finished.
  //
  // @details The agent must be able to access @p ptr
  //
  // @param [in] ptr Address of the memory to be set.
  // @param [in] value The value/pattern that will be used to set @p ptr.
  // @param [in] count Number of uint32_t element to be set.
  //
  // @retval HSA_STATUS_SUCCESS The memory fill is finished and successful.
  virtual hsa_status_t DmaFill(void* ptr, uint32_t value, size_t count) {
    return HSA_STATUS_ERROR;
  }

  // @brief Invoke the user provided callback for each region accessible by
  // this agent.
  //
  // @param [in] callback User provided callback function.
  // @param [in] data User provided pointer as input for @p callback.
  //
  // @retval ::HSA_STATUS_SUCCESS if the callback function for each traversed
  // region returns ::HSA_STATUS_SUCCESS.
  virtual hsa_status_t IterateRegion(
      hsa_status_t (*callback)(hsa_region_t region, void* data),
      void* data) const = 0;

  // @brief Invoke the user provided callback for each isa supported by
  // this agent.
  //
  // @param [in] callback User provided callback function.
  // @param [in] data User provided pointer as input for @p callback.
  //
  // @retval ::HSA_STATUS_SUCCESS if the callback function for each traversed
  // isa returns ::HSA_STATUS_SUCCESS.
  virtual hsa_status_t IterateSupportedIsas(
      hsa_status_t (*callback)(hsa_isa_t isa, void* data),
      void* data) const = 0;

  // @brief Invoke the callback for each cache useable by this agent.
  virtual hsa_status_t IterateCache(hsa_status_t (*callback)(hsa_cache_t cache, void* data),
                                    void* data) const = 0;

  /// @brief Create queue.
  ///
  /// @param [in] size Number of packets the queue is expected to hold. Must be a
  /// power of 2 greater than 0.
  /// @param [in] queue_type Queue type.
  /// @param [in] flags Flags to specify queue attributes on creation.
  /// @param [in] event_callback Callback invoked for every
  /// asynchronous event related to the newly created queue. May be NULL.The HSA
  /// runtime passes three arguments to the callback : a code identifying the
  /// event that triggered the invocation, a pointer to the queue where the event
  /// originated, and the application data.
  /// @param [in] data Application data that is passed to @p callback.
  /// @param [in] private_segment_size A hint to indicate the maximum expected
  /// private segment usage per work-item, in bytes.
  /// @param [in] group_segment_size A hint to indicate the maximum expected
  /// group segment usage per work-group, in bytes.
  /// @param[out] queue Memory location where the HSA runtime stores a pointer
  /// to the newly created queue.
  ///
  /// @retval HSA_STATUS_SUCCESS The queue has been created successfully.
  virtual hsa_status_t QueueCreate(size_t size, hsa_queue_type32_t queue_type, uint64_t flags,
                                   HsaEventCallback event_callback, void* data,
                                   uint32_t private_segment_size, uint32_t group_segment_size,
                                   Queue** queue) = 0;

  // @brief Query the value of an attribute.
  //
  // @param [in] attribute Attribute to query.
  // @param [out] value Pointer to store the value of the attribute.
  //
  // @param HSA_STATUS_SUCCESS @p value has been filled with the value of the
  // attribute.
  virtual hsa_status_t GetInfo(hsa_agent_info_t attribute,
                               void* value) const = 0;

  // @brief Returns an array of regions owned by the agent.
  virtual const std::vector<const core::MemoryRegion*>& regions() const = 0;

  // @brief Returns the ISA's supported by the agent.
  // @details The returned vector is a list of pointers to the supported ISA,
  // ordered from most specific (and performant) to most generic. For CPU
  // and AIE agents, this list will be empty.
  virtual const std::vector<const core::Isa *>& supported_isas() const = 0;

  virtual uint64_t HiveId() const { return 0; }

  // @brief Returns the device type (CPU/GPU/Others).
  __forceinline uint32_t device_type() const { return device_type_; }

  // @brief Returns hsa_agent_t handle exposed to end user.
  //
  // @details Only matters when tools library need to intercept HSA calls.
  __forceinline hsa_agent_t public_handle() const { return public_handle_; }

  // @brief Returns node id associated with this agent.
  __forceinline uint32_t node_id() const { return node_id_; }

  // @brief Returns the driver associated with this agent.
  __forceinline Driver& driver() const { return *driver_; }

  // @brief Getter for profiling_enabled_.
  __forceinline bool profiling_enabled() const { return profiling_enabled_; }

  // @brief Setter for profiling_enabled_.
  virtual hsa_status_t profiling_enabled(bool enable) {
    const hsa_status_t stat = EnableDmaProfiling(enable);
    if (HSA_STATUS_SUCCESS == stat) {
      profiling_enabled_ = enable;
    }

    return stat;
  }

  __forceinline bool Enabled() const { return enabled_; }

  __forceinline void Enable() { enabled_ = true; }

  __forceinline void Disable() { enabled_ = false; }

  virtual void Trim() {
    for (auto region : regions()) region->Trim();
  }

  virtual void ReleaseResources() { }

protected:
  // Intention here is to have a polymorphic update procedure for public_handle_
  // which is callable on any Agent* but only from some class dervied from
  // Agent*.  do_set_public_handle should remain protected or private in all
  // derived types.
  static __forceinline void set_public_handle(Agent* agent,
                                              hsa_agent_t handle) {
    agent->do_set_public_handle(handle);
  }

  virtual void do_set_public_handle(hsa_agent_t handle) {
    public_handle_ = handle;
  }

  // @brief Enable profiling of the asynchronous DMA copy. The timestamp
  // of each copy request will be stored in the completion signal structure.
  //
  // @param enable True to enable profiling. False to disable profiling.
  //
  // @retval HSA_STATUS_SUCCESS The profiling is enabled and the
  // timing of subsequent async copy will be measured.
  virtual hsa_status_t EnableDmaProfiling(bool enable) {
    return HSA_STATUS_SUCCESS;
  }

  hsa_agent_t public_handle_;
  std::vector<const core::Isa *> supported_isas_;

 private:
  // @brief Node id.
  const uint32_t node_id_;

  const uint32_t device_type_;

  Driver *driver_;

  bool profiling_enabled_;

  bool enabled_;

  // Used by an Agent's MemoryRegions to ensure serial memory operation on the device.
  // Serial memory operations are needed to ensure, among other things, that allocation failures are
  // due to true OOM conditions and per region caching (Trim and Allocate must be serial and
  // exclusive to ensure this).
  KernelMutex agent_memory_lock_;

  // Forbid copying and moving of this object
  DISALLOW_COPY_AND_ASSIGN(Agent);
};
}  // namespace core
}  // namespace rocr

#endif  // header guard
