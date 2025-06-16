// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef SRC_CORE_PM4_FACTORY_H_
#define SRC_CORE_PM4_FACTORY_H_

#include <assert.h>
#include <hsa/hsa_ext_amd.h>
#include <stdint.h>
#include <string.h>

#include <climits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>

#include "aqlprofile-sdk/aql_profile_v2.h"
#include "core/aql_profile.hpp"
#include "core/aql_profile_exception.h"
#include "def/gpu_block_info.h"
#include "pm4/cmd_builder.h"
#include "pm4/pmc_builder.h"
#include "pm4/spm_builder.h"
#include "pm4/sqtt_builder.h"
#include "util/hsa_rsrc_factory.h"

namespace aql_profile {

struct pm4_agent_info {
  std::string agent_gfxip;
  uint32_t cu_num;
  uint32_t se_num;
  uint32_t shader_arrays_per_se;
  uint32_t xcc_num;
};

const AgentInfo* GetAgentInfo(aqlprofile_agent_handle_t agent_id);

aqlprofile_agent_handle_t RegisterAgent(const aqlprofile_agent_info_v1_t* agent_info);

// GPU enumeration
enum gpu_id_t {
  INVAL_GPU_ID,  // invalid GPU id
  GFX9_GPU_ID,   // generic Gfx9 id
  MI100_GPU_ID,  // Mi100 GPU id
  MI200_GPU_ID,  // Mi200 GPU id
  MI300_GPU_ID,  // Mi300 GPU id
  MI350_GPU_ID,  // Mi350 GPU id
  GFX10_GPU_ID,  // generic Gfx10 id
  GFX11_GPU_ID,  // generic Gfx11 id
  GFX12_GPU_ID,  // generic Gfx12 id
};

// Block info map class
class BlockInfoMap {
 public:
  BlockInfoMap(const GpuBlockInfo** table, const uint32_t& size)
      : block_table_(table), block_count_(size / sizeof(uintptr_t)) {}
  BlockInfoMap(const BlockInfoMap& map)
      : block_table_(map.block_table_), block_count_(map.block_count_) {}

  // Get block info for a given block id
  const GpuBlockInfo* Get(const uint32_t& block_id) const {
    return (block_id < block_count_) ? block_table_[block_id] : NULL;
  }

  // Find block by name
  // Return block id or UINT32_MAX if not found
  uint32_t Find(const char* name) const {
    uint32_t index = 0;
    while (index < block_count_) {
      const GpuBlockInfo* entry = block_table_[index];
      if (entry) {
        if (strcmp(name, entry->name) == 0) break;
      }
      ++index;
    }
    return (index == block_count_) ? UINT32_MAX : index;
  }

 private:
  // Block info table
  const GpuBlockInfo** const block_table_;
  // Number of elements in the block info table
  const uint32_t block_count_;
};

// Factory of PM4 builders
class Pm4Factory {
 public:
  typedef std::mutex mutex_t;

  static Pm4Factory* Create(aqlprofile_agent_handle_t agent_info, bool concurrent = false);
  static Pm4Factory* Create(const AgentInfo* agent_info, gpu_id_t gpu_id, bool concurrent);
  // Create factory for a given agent
  static Pm4Factory* Create(const hsa_agent_t agent, const bool concurrent = false);
  // Create factory for a given profile
  static Pm4Factory* Create(const profile_t* profile) {
    // First check and save the mode
    return Create(profile->agent, CheckConcurrent(profile));
  }

  // Return gpu id
  gpu_id_t GetGpuId() const { return gpu_id_; }
  // Is pmc to be profiled concurrently?
  bool IsConcurrent() const { return concurrent_mode_; }
  // Is getting SPM data using driver public API?
  bool SpmKfdMode() const { return spm_kfd_mode_; }

  // Return PM4 command builder
  pm4_builder::CmdBuilder* GetCmdBuilder() const { return cmd_builder_; }
  // Return PMC PM4 packets builder
  pm4_builder::PmcBuilder* GetPmcBuilder() const { return pmc_builder_; }
  // Return SPM PM4 packets builder
  pm4_builder::SpmBuilder* GetSpmBuilder() const { return spm_builder_; }
  // Return SQTT PM4 packets builder
  pm4_builder::SqttBuilder* GetSqttBuilder() const { return sqtt_builder_; }

  // Return Shader Engines number
  uint32_t GetShaderEnginesNumber() const { return agent_info_->se_num; }
  uint32_t GetShaderArraysNumber() const { return agent_info_->shader_arrays_per_se; }
  uint32_t GetComputeUnitNumber() const { return agent_info_->cu_num; }
  // Return SQTT buffer alignment
  uint32_t GetSQTTBufferAlignment() const { return 0x1000; }
  const char* GetGFX() const { return agent_info_->name; }
  virtual bool IsGFX9() const { return false; }
  virtual bool IsGFX10() const { return false; }
  virtual bool IsGFX11() const { return false; }
  virtual bool IsGFX12() const { return false; }
  // Return number of XCC on the GPU
  uint32_t GetXccNumber() const { return agent_info_->xcc_num; }

  const GpuBlockInfo* GetBlockInfo(const aqlprofile_pmc_event_t* event) const {
    const GpuBlockInfo* info = block_map_.Get(event->block_name);
    if (info == NULL) throw std::runtime_error("Bad Block");
    // Checking that the block index is in proper range
    if (event->block_index >= info->instance_count) throw std::runtime_error("Bad Index");
      // Checking that the counter event index is in proper range
#if 0
    if (event->counter_id > info->event_id_max)
      throw event_exception(std::string("Bad event ID, "), *event);
#endif
    return info;
  }

  // Return block info foor a given event
  const GpuBlockInfo* GetBlockInfo(const event_t* event) const {
    const GpuBlockInfo* info = block_map_.Get(event->block_name);
    if (info == NULL) throw event_exception(std::string("Bad block, "), *event);
    // Checking that the block index is in proper range
    if (event->block_index >= info->instance_count)
      throw event_exception(std::string("Bad block index, "), *event);
      // Checking that the counter event index is in proper range
#if 0
    if (event->counter_id > info->event_id_max)
      throw event_exception(std::string("Bad event ID, "), *event);
#endif
    return info;
  }

  // Return block info for a given block id
  const GpuBlockInfo* GetBlockInfo(const uint32_t& block_id) const {
    return block_map_.Get(block_id);
  }

  virtual size_t GetNumEvents(uint32_t block_name) const {
    size_t se_number = GetShaderEnginesNumber() / GetXccNumber();
    size_t block_samples_count = 1;
    auto* block_info = GetBlockInfo(block_name);

    if (block_info->attr & CounterBlockSeAttr)
      block_samples_count *= se_number;
    if (block_info->attr & CounterBlockSaAttr)
      block_samples_count *= 2;
    if (block_info->attr & CounterBlockWgpAttr)
      block_samples_count *= GetNumWGPs();
    if ((block_info->attr & CounterBlockSqAttr) && IsGFX11()) // TODO: Move to CounterBlockWgpAttr
      block_samples_count *= GetNumWGPs();
    return block_samples_count;
  }

  virtual size_t GetBytesNeeded(uint32_t block_name) const {
    return GetNumEvents(block_name) * GetXccNumber() * sizeof(uint64_t);
  }

  // Return block id for a given block name string
  uint32_t FindBlock(const char* name) const { return block_map_.Find(name); }

  /// Workaround for GFX11. PMC Builder overrides this.
  virtual int GetNumWGPs() const {
    if (pmc_builder_) return pmc_builder_->GetNumWGPs();
    return 1;
  };

  virtual int GetAccumLowID() const { throw HSA_STATUS_ERROR_INVALID_ARGUMENT; };
  virtual int GetAccumHiID() const { throw HSA_STATUS_ERROR_INVALID_ARGUMENT; };

 protected:
  explicit Pm4Factory(const BlockInfoMap& map)
      : cmd_builder_(NULL),
        pmc_builder_(NULL),
        spm_builder_(NULL),
        sqtt_builder_(NULL),
        agent_info_(NULL),
        concurrent_mode_(concurrent_create_mode_),
        block_map_(map) {}

  friend class std::default_delete<Pm4Factory>;
  virtual ~Pm4Factory() {
    delete cmd_builder_;
    delete pmc_builder_;
    delete spm_builder_;
    delete sqtt_builder_;
  }

  // PM4 command builder
  pm4_builder::CmdBuilder* cmd_builder_;
  // PMC PM4 packets builder
  pm4_builder::PmcBuilder* pmc_builder_;
  // SPM PM4 packets builder
  pm4_builder::SpmBuilder* spm_builder_;
  // SQTT PM4 packets builder
  pm4_builder::SqttBuilder* sqtt_builder_;
  // agent info
  const AgentInfo* agent_info_;
  gpu_id_t gpu_id_;
  // Concurrent mode
  static bool concurrent_create_mode_;
  static bool spm_kfd_mode_;
  bool concurrent_mode_;

 private:
  // PM4 factory instance map type
  struct instances_fncomp_t {
    bool operator()(const hsa_agent_t& a, const hsa_agent_t& b) const {
      return a.handle < b.handle;
    }
  };
  typedef std::map<hsa_agent_t, std::unique_ptr<Pm4Factory>, instances_fncomp_t> instances_t;

  // Create GFX9 generic factory
  static Pm4Factory* Gfx9Create(const AgentInfo* agent_info);
  // Create GFX10 generic factory
  static Pm4Factory* Gfx10Create(const AgentInfo* agent_info);
  // Create GFX11 generic factory
  static Pm4Factory* Gfx11Create(const AgentInfo* agent_info);
  // Create GFX12 generic factory
  static Pm4Factory* Gfx12Create(const AgentInfo* agent_info);
  // Create MI100 factory
  static Pm4Factory* Mi100Create(const AgentInfo* agent_info);
  // Create MI200 factory
  static Pm4Factory* Mi200Create(const AgentInfo* agent_info);
  // Create MI300 factory
  static Pm4Factory* Mi300Create(const AgentInfo* agent_info);
  // Create MI350 factory
  static Pm4Factory* Mi350Create(const AgentInfo* agent_info);
  // Return GPU id for a given agent
  static gpu_id_t GetGpuId(std::string_view);

  static bool CheckConcurrent(const profile_t* profile);

  // This struct holds all the "global" state for the factory system
  struct Manager {
    // Mutex for inter thread synchronization for the instances create/destroy
    mutex_t mutex_;
    // Factory instances container
    instances_t instances_;
  };

  static Manager& getManager() {
    static Manager manager_instance;
    return manager_instance;
  }

  // Block info container
  const BlockInfoMap block_map_;
};

inline Pm4Factory* Pm4Factory::Create(const AgentInfo* agent_info, gpu_id_t gpu_id,
                                      bool concurrent) {
  auto& manager = getManager();
  const auto ret = manager.instances_.insert({agent_info->dev_id, nullptr});
  instances_t::iterator it = ret.first;

  concurrent_create_mode_ = concurrent;
  static bool spm_kfd = getenv("ROCP_SPM_KFD_MODE") != NULL;
  spm_kfd_mode_ = spm_kfd;

  // Create a factory implementation for the GPU id
  if (ret.second) {
    Pm4Factory *impl = nullptr;
    switch (gpu_id) {
      // Create Gfx9 generic factory
      case GFX9_GPU_ID:
        impl = Gfx9Create(agent_info);
        break;
      // Create Gfx10 generic factory
      case GFX10_GPU_ID:
        impl = Gfx10Create(agent_info);
        break;
      // Create Gfx11 generic factory
      case GFX11_GPU_ID:
        impl = Gfx11Create(agent_info);
        break;
      case GFX12_GPU_ID:
        impl = Gfx12Create(agent_info);
        break;
      // Create MI100 generic factory
      case MI100_GPU_ID:
        impl = Mi100Create(agent_info);
        break;
      case MI200_GPU_ID:
        impl = Mi200Create(agent_info);
        break;
      case MI300_GPU_ID:
        impl = Mi300Create(agent_info);
        break;
      case MI350_GPU_ID:
        impl = Mi350Create(agent_info);
        break;
      default:
        throw aql_profile_exc_val<gpu_id_t>("GPU id error", gpu_id);
    }
    it->second.reset(impl);
  }

  if (it->second == NULL) throw aql_profile_exc_msg("Pm4Factory::Create() failed");
  it->second->gpu_id_ = gpu_id;
  return it->second.get();
}

// Create PM4 factory
inline Pm4Factory* Pm4Factory::Create(const hsa_agent_t agent, bool concurrent) {
  auto& manager = getManager();
  std::lock_guard<mutex_t> lck(manager.mutex_);
  const AgentInfo* agent_info = HsaRsrcFactory::Instance().GetAgentInfo(agent);
  // Get GPU id for a given agent

  hsa_status_t status = HSA_STATUS_ERROR;
  std::vector<char> agent_name{};
  agent_name.resize(64);
  uint32_t device_id = 0;

  // Getting GfxIP name
  status = hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, agent_name.data());
  if (status == HSA_STATUS_SUCCESS) {
    // Getting DeviceId
    hsa_agent_info_t attribute = static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_CHIP_ID);
    status = hsa_agent_get_info(agent, attribute, &device_id);
  }
  if (status != HSA_STATUS_SUCCESS) {
    throw aql_profile_exc_msg("Pm4Factory::Create() bad agent");
  }

  const gpu_id_t gpu_id = GetGpuId(agent_name.data());
  return Pm4Factory::Create(agent_info, gpu_id, concurrent);
}

inline Pm4Factory* Pm4Factory::Create(aqlprofile_agent_handle_t agent_info, bool concurrent) {
  const auto* info = GetAgentInfo(agent_info);
  if (info == NULL) throw aql_profile_exc_val<uint64_t>("Bad agent handle", agent_info.handle);
  const gpu_id_t gpu_id = GetGpuId(info->gfxip);
  return Pm4Factory::Create(info, gpu_id, concurrent);
}

// Check the setting of pmc profiling mode
inline bool Pm4Factory::CheckConcurrent(const profile_t* profile) {
  for (const hsa_ven_amd_aqlprofile_parameter_t* p = profile->parameters;
       p < (profile->parameters + profile->parameter_count); ++p) {
    if (p->parameter_name == HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_K_CONCURRENT) return true;
  }

  return false;
}

// Return GPU id for a given agent
inline gpu_id_t Pm4Factory::GetGpuId(std::string_view gfx_ip) {
  std::vector<std::pair<std::string, gpu_id_t>> gfxip_map = {
      {"gfx908", MI100_GPU_ID}, {"gfx90a", MI200_GPU_ID}, {"gfx900", GFX9_GPU_ID},
      {"gfx902", GFX9_GPU_ID},  {"gfx906", GFX9_GPU_ID},  {"gfx94", MI300_GPU_ID},
      {"gfx95", MI350_GPU_ID},  {"gfx10", GFX10_GPU_ID},  {"gfx11", GFX11_GPU_ID},
      {"gfx12", GFX12_GPU_ID},
  };

  for (const auto& [name, id] : gfxip_map) {
    if (gfx_ip.rfind(name, 0) == 0) {
      return id;
    }
  }

  return INVAL_GPU_ID;
}

}  // namespace aql_profile

#endif  // SRC_CORE_PM4_FACTORY_H_
