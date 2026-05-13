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

#include "pm4_factory.h"

#include <shared_mutex>
#include <vector>

#include "core/architecture_init.hpp"
#include "pm4/pmc_builder.h"
#include "pm4/spm_builder.h"
#include "pm4/sqtt_builder.h"
#include "util/reg_offsets.h"

namespace aql_profile {

Pm4Factory::Pm4Factory(HardwareArchitecture* arch, const AgentInfo* agent_info)
    : cmd_builder_(NULL), pmc_builder_(NULL), spm_builder_(NULL), sqtt_builder_(NULL),
      concurrent_mode_(concurrent_create_mode_),
      architecture_(arch), prims_(nullptr) {
  static bool spm_kfd = getenv("ROCP_SPM_KFD_MODE") != NULL;
  spm_kfd_mode_ = spm_kfd;
  InitializeBuilders(agent_info);
}

void Pm4Factory::InitializeBuilders(const AgentInfo* agent_info) {
  cmd_builder_ = architecture_->CreateCmdBuilder(acquire_ip_offset_table(agent_info));
  prims_        = architecture_->CreatePrimitivesProvider();
  const auto& config = architecture_->GetConfig();
  pmc_builder_  = new pm4_builder::GpuPmcBuilder(config, cmd_builder_, prims_, concurrent_mode_);
  spm_builder_  = new pm4_builder::GpuSpmBuilder(cmd_builder_, prims_);
  sqtt_builder_ = new pm4_builder::GpuSqttBuilder(config, cmd_builder_, prims_, agent_info->timestamp_freq);
}

Pm4Factory::~Pm4Factory() {
  delete cmd_builder_;
  delete pmc_builder_;
  delete spm_builder_;
  delete sqtt_builder_;
  delete prims_;
  delete architecture_;
}

namespace {
struct locked_agent_cache {
  std::shared_mutex mutex;
  std::unordered_map<uint64_t, AgentInfo> cache;

  void add(uint64_t& agent_id, const AgentInfo& agent_info) {
    auto lock = std::unique_lock{mutex};
    agent_id = cache.size();
    cache[agent_id] = agent_info;
  }

  const AgentInfo* get(uint64_t agent_id) {
    auto lock = std::shared_lock{mutex};
    auto it = cache.find(agent_id);
    if (it == cache.end()) return nullptr;
    return &it->second;
  }
};

locked_agent_cache& get_cache() {
  static auto* cache = new locked_agent_cache{};
  return *cache;
}
}  // namespace

aqlprofile_agent_handle_t RegisterAgent(const aqlprofile_agent_info_v1_t* agent_info) {
  aqlprofile_agent_handle_t agent_id;
  AgentInfo int_agent_info;
  int_agent_info.cu_num = agent_info->cu_num;
  int_agent_info.se_num = agent_info->se_num;
  int_agent_info.xcc_num = agent_info->xcc_num;
  int_agent_info.shader_arrays_per_se = agent_info->shader_arrays_per_se;
  int_agent_info.domain = agent_info->domain;
  int_agent_info.bdf_id = agent_info->location_id;

  auto len = strlen(agent_info->agent_gfxip);
  memset(int_agent_info.name, 0, sizeof(int_agent_info.name));
  memcpy(int_agent_info.name, agent_info->agent_gfxip,
         (len >= sizeof(int_agent_info.name) ? sizeof(int_agent_info.name) - 1 : len));
  memset(int_agent_info.gfxip, 0, sizeof(int_agent_info.gfxip));
  memcpy(int_agent_info.gfxip, agent_info->agent_gfxip,
         (len >= sizeof(int_agent_info.gfxip) ? sizeof(int_agent_info.gfxip) - 1 : len));

  // TODO: Temporary patch for gfx1250's asymmetric CU design, will remove
  //       after CU mask support is added to agent_info
  // TODO: gfx1250 defines 1WGP = 1CU, different from other RDNA products.
  //       Patch it to be WGP = 2CU to reuse profiler logic
  if (!strncmp(int_agent_info.name, "gfx1250", 7)) {
    int_agent_info.cu_num = agent_info->se_num * agent_info->shader_arrays_per_se * 9 * 2;
    int_agent_info.xcc_per_aid = 4;
  } else if (!strncmp(int_agent_info.name, "gfx94", 5) ||
             !strncmp(int_agent_info.name, "gfx95", 5)) {
    int_agent_info.xcc_per_aid = 2;
  } else {
    int_agent_info.xcc_per_aid = 1;
  }

  get_cache().add(agent_id.handle, int_agent_info);
  return agent_id;
}

const AgentInfo* GetAgentInfo(aqlprofile_agent_handle_t agent_id) {
  return get_cache().get(agent_id.handle);
}

Pm4Factory* Pm4Factory::Create(const hsa_agent_t agent, bool concurrent) {
  const AgentInfo* agent_info = HsaRsrcFactory::Instance().GetAgentInfo(agent);

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

  concurrent_create_mode_ = concurrent;
  HardwareArchitecture* arch = CreateArchitectureForAgent(agent_info);
  if (!arch) throw aql_profile_exc_msg("Pm4Factory::Create() unrecognised GPU");
  return new Pm4Factory(arch, agent_info);
}

Pm4Factory* Pm4Factory::Create(aqlprofile_agent_handle_t agent_info, bool concurrent) {
  const auto* info = GetAgentInfo(agent_info);
  if (info == NULL) throw aql_profile_exc_val<uint64_t>("Bad agent handle", agent_info.handle);

  concurrent_create_mode_ = concurrent;
  HardwareArchitecture* arch = CreateArchitectureForAgent(info);
  if (!arch) throw aql_profile_exc_msg("Pm4Factory::Create() unrecognised GPU");
  return new Pm4Factory(arch, info);
}

}  // namespace aql_profile
