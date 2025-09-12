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

#ifndef HSA_RUNTIME_CORE_INC_AMD_CPU_AGENT_H_
#define HSA_RUNTIME_CORE_INC_AMD_CPU_AGENT_H_

#include <vector>

#include "core/inc/runtime.h"
#include "core/inc/agent.h"
#include "core/inc/queue.h"
#include "core/inc/cache.h"
#include "core/inc/driver.h"

namespace rocr {
namespace AMD {
// @brief Class to represent a CPU device.
class CpuAgent : public core::Agent {
 public:
  // @brief CpuAgent constructor.
  //
  // @param [in] node Node id. Each CPU in different socket will get distinct
  // id.
  // @param [in] node_props Node property.
  // @param [in] driver_type Driver type. Default is KFD.
  CpuAgent(HSAuint32 node, const HsaNodeProperties& node_props,
           core::DriverType driver_type = core::DriverType::KFD);

  // @brief CpuAgent destructor.
  ~CpuAgent();

  // @brief Invoke the user provided callback for each region accessible by
  // this agent.
  //
  // @param [in] include_peer If true, the callback will be also invoked on each
  // peer memory region accessible by this agent. If false, only invoke the
  // callback on memory region owned by this agent.
  // @param [in] callback User provided callback function.
  // @param [in] data User provided pointer as input for @p callback.
  //
  // @retval ::HSA_STATUS_SUCCESS if the callback function for each traversed
  // region returns ::HSA_STATUS_SUCCESS.
  hsa_status_t VisitRegion(bool include_peer,
                           hsa_status_t (*callback)(hsa_region_t region,
                                                    void* data),
                           void* data) const;

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
  hsa_status_t GetInfo(hsa_agent_info_t attribute, void* value) const override;

  // @brief Override from core::Agent.
  hsa_status_t QueueCreate(size_t size, hsa_queue_type32_t queue_type, uint64_t flags,
                           core::HsaEventCallback event_callback, void* data,
                           uint32_t private_segment_size, uint32_t group_segment_size,
                           core::Queue** queue) override;

  // @brief Override from core::Agent.
  hsa_status_t DmaCopy(void* dst, core::Agent& dst_agent, const void* src, core::Agent& src_agent,
                       size_t size, std::vector<core::Signal*>& dep_signals,
                       core::Signal& out_signal) override;

  // @brief Returns number of data caches.
  __forceinline size_t num_cache() const { return cache_props_.size(); }

  // @brief Returns Hive ID
  __forceinline uint64_t HiveId() const override { return  properties_.HiveID; }

  // @brief Returns data cache property.
  //
  // @param [in] idx Cache level.
  __forceinline const HsaCacheProperties& cache_prop(int idx) const {
    return cache_props_[idx];
  }

  // @brief Override from core::Agent.
  const std::vector<const core::MemoryRegion*>& regions() const override {
    return regions_;
  }

  // @brief Override from core::Agent.
  const std::vector<const core::Isa*>& supported_isas() const override {
    return supported_isas_;
  }
 private:
  // @brief Query the driver to get the region list owned by this agent.
  void InitRegionList();

  // @brief Query the driver to get the cache properties.
  void InitCacheList();

  // @brief Invoke the user provided callback for every region in @p regions.
  //
  // @param [in] regions Array of region object.
  // @param [in] callback User provided callback function.
  // @param [in] data User provided pointer as input for @p callback.
  //
  // @retval ::HSA_STATUS_SUCCESS if the callback function for each traversed
  // region returns ::HSA_STATUS_SUCCESS.
  hsa_status_t VisitRegion(
      const std::vector<const core::MemoryRegion*>& regions,
      hsa_status_t (*callback)(hsa_region_t region, void* data),
      void* data) const;

  // @brief Node property.
  const HsaNodeProperties properties_;

  // @brief Array of data cache property. The array index represents the cache
  // level.
  std::vector<HsaCacheProperties> cache_props_;

  // @brief Array of HSA cache objects.
  std::vector<std::unique_ptr<core::Cache>> caches_;

  // @brief Array of regions owned by this agent.
  std::vector<const core::MemoryRegion*> regions_;

  DISALLOW_COPY_AND_ASSIGN(CpuAgent);
};

}  // namespace amd
}  // namespace rocr

#endif  // header guard
