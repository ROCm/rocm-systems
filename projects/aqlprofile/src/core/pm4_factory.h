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

#include "hsa_includes.h"
#include <stdint.h>
#include <string.h>

#include <climits>
#include <string>

#include "aqlprofile-sdk/aql_profile_v2.h"
#include "core/aql_profile.hpp"
#include "core/aql_profile_exception.h"
#include "core/hardware_architecture.hpp"
#include "def/gpu_block_info.h"
#include "pm4/cmd_builder.h"
#include "pm4/primitives_provider.hpp"
#include "util/hsa_rsrc_factory.h"

namespace pm4_builder {
class PmcBuilder;
class SpmBuilder;
class SqttBuilder;
}  // namespace pm4_builder

namespace aql_profile {

const AgentInfo* GetAgentInfo(aqlprofile_agent_handle_t agent_id);

aqlprofile_agent_handle_t RegisterAgent(const aqlprofile_agent_info_v1_t* agent_info);

// Factory of PM4 builders — owns the HardwareArchitecture and all PM4 builder objects.
class __attribute__((visibility("default"))) Pm4Factory {
 public:
  // Create factory for a given agent handle
  static Pm4Factory* Create(aqlprofile_agent_handle_t agent_info, bool concurrent = false);
  // Create factory for a given HSA agent
  static Pm4Factory* Create(const hsa_agent_t agent, const bool concurrent = false);
  // Create factory for a given profile
  static Pm4Factory* Create(const profile_t* profile) {
    return Create(profile->agent, CheckConcurrent(profile));
  }
  // Destroy factory (no-op: callers own factory objects)
  static void Destroy() {}

  virtual bool IsConcurrent() const { return concurrent_mode_; }
  virtual bool SpmKfdMode() const { return spm_kfd_mode_; }

  virtual pm4_builder::CmdBuilder* GetCmdBuilder() const { return cmd_builder_; }
  virtual pm4_builder::PmcBuilder* GetPmcBuilder() const { return pmc_builder_; }
  virtual pm4_builder::SpmBuilder* GetSpmBuilder() const { return spm_builder_; }
  virtual pm4_builder::SqttBuilder* GetSqttBuilder() const { return sqtt_builder_; }

  virtual uint32_t GetShaderEnginesNumber() const { return architecture_->GetConfig().se_count; }
  virtual uint32_t GetShaderArraysNumber() const { return architecture_->GetConfig().sa_per_se_count; }
  virtual uint32_t GetComputeUnitNumber() const { return architecture_->GetConfig().cu_count; }
  virtual uint32_t GetSQTTBufferAlignment() const { return 0x1000; }
  virtual const char* GetGFX() const {
    gfx_name_ = architecture_->GetConfig().name;
    return gfx_name_.c_str();
  }
  virtual bool IsGFX9() const { return architecture_->IsGFX9(); }
  virtual bool IsGFX10() const { return architecture_->IsGFX10(); }
  virtual bool IsGFX11() const { return architecture_->IsGFX11(); }
  virtual bool IsGFX12() const { return architecture_->IsGFX12(); }
  virtual uint32_t GetXccNumber() const { return architecture_->GetConfig().xcc_count; }
  virtual uint32_t GetXccPerAid() const { return architecture_->GetConfig().xcc_per_aid; }

  virtual uint32_t GetSpmSampleDelayMax() { return architecture_->GetSpmSampleDelayMax(); }

  virtual bool HasSpmCore1() const { return architecture_->GetConfig().has_spm_core1; }
  virtual bool HasSqttStatus2Register() const { return architecture_->GetConfig().has_sqtt_status2_register; }
  virtual bool HasWptrRelativeAddressing() const { return architecture_->GetConfig().has_wptr_relative_addressing; }
  virtual bool NeedsSqttHeaderPacket() const { return architecture_->GetConfig().needs_sqtt_header_packet; }
  virtual bool SupportsSpm() const { return architecture_->GetConfig().supports_spm; }
  virtual bool SupportsSpmV2() const { return architecture_->GetConfig().supports_spm_v2; }
  virtual bool HasSriovSpmRestriction() const { return architecture_->GetConfig().has_sriov_spm_restriction; }
  virtual uint32_t GetSqttHeaderVersion() const { return architecture_->GetSqttHeaderVersion(); }

  virtual const GpuBlockInfo* GetBlockInfo(const aqlprofile_pmc_event_t* event) const {
    return architecture_->GetBlockInfo(event->block_name);
  }
  virtual const GpuBlockInfo* GetBlockInfo(const event_t* event) const {
    return architecture_->GetBlockInfo(event->block_name);
  }
  virtual const GpuBlockInfo* GetBlockInfo(const uint32_t& block_id) const {
    return architecture_->GetBlockInfo(block_id);
  }

  virtual size_t GetNumEvents(uint32_t block_name) const {
    return architecture_->GetNumEventsForBlock(block_name);
  }
  virtual size_t GetBytesNeeded(uint32_t block_name) const {
    return architecture_->GetBytesNeededForBlock(block_name);
  }

  virtual uint32_t FindBlock(const char* name) const {
    return architecture_->FindBlockByName(name);
  }

  virtual int GetNumWGPs() const { return architecture_->GetNumWGPs(); }
  virtual int GetAccumLowID() const { return architecture_->GetAccumLowID(); }
  virtual int GetAccumHiID() const { return architecture_->GetAccumHiID(); }

  const HardwareArchitecture* GetArchitecture() const { return architecture_; }

 protected:
  // Production constructor — called by Create().
  Pm4Factory(HardwareArchitecture* arch, const AgentInfo* agent_info);
  // No-arg constructor for test mocks only; architecture_ is null.
  explicit Pm4Factory()
      : cmd_builder_(NULL), pmc_builder_(NULL), spm_builder_(NULL), sqtt_builder_(NULL),
        concurrent_mode_(concurrent_create_mode_), architecture_(nullptr), prims_(nullptr) {}

  virtual ~Pm4Factory();

  pm4_builder::CmdBuilder* cmd_builder_;
  pm4_builder::PmcBuilder* pmc_builder_;
  pm4_builder::SpmBuilder* spm_builder_;
  pm4_builder::SqttBuilder* sqtt_builder_;

  static bool concurrent_create_mode_;
  static bool spm_kfd_mode_;
  bool concurrent_mode_;

 private:
  void InitializeBuilders(const AgentInfo* agent_info);
  static bool CheckConcurrent(const profile_t* profile);

  HardwareArchitecture* architecture_;
  pm4_builder::PrimitivesProvider* prims_;
  mutable std::string gfx_name_;
};

// Check the setting of pmc profiling mode
inline bool Pm4Factory::CheckConcurrent(const profile_t* profile) {
  for (const hsa_ven_amd_aqlprofile_parameter_t* p = profile->parameters;
       p < (profile->parameters + profile->parameter_count); ++p) {
    if (p->parameter_name == HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_K_CONCURRENT) return true;
  }
  return false;
}

}  // namespace aql_profile

#endif  // SRC_CORE_PM4_FACTORY_H_
