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

#include "core/pm4_factory_adapter.hpp"
#include "core/aql_profile_exception.h"
#include "pm4/pmc_builder.h"
#include "pm4/spm_builder.h"
#include "pm4/sqtt_builder.h"

namespace aql_profile {

Pm4FactoryAdapter::Pm4FactoryAdapter(HardwareArchitecture* architecture,
                                     const AgentInfo* agent_info)
    : Pm4Factory(BlockInfoMap(nullptr, 0)),
      architecture_(architecture),
      prims_(nullptr),
      concurrent_mode_(false),
      spm_kfd_mode_(false) {
  if (!architecture_) {
    throw aql_profile_exc_msg("Null architecture provided to Pm4FactoryAdapter");
  }

  concurrent_mode_ = concurrent_create_mode_;
  static bool spm_kfd = getenv("ROCP_SPM_KFD_MODE") != NULL;
  spm_kfd_mode_ = spm_kfd;

  InitializeBuilders(agent_info);
}

Pm4FactoryAdapter::~Pm4FactoryAdapter() {
  // architecture_ is owned by the caller; builders are deleted by the base class.
  delete prims_;
}

void Pm4FactoryAdapter::InitializeBuilders(const AgentInfo* agent_info) {
  cmd_builder_ = architecture_->CreateCmdBuilder();
  prims_        = architecture_->CreatePrimitivesProvider();
  const auto& config = architecture_->GetConfig();
  pmc_builder_  = new pm4_builder::GpuPmcBuilder(config, cmd_builder_, prims_, concurrent_mode_);
  spm_builder_  = new pm4_builder::GpuSpmBuilder(cmd_builder_, prims_);
  sqtt_builder_ = new pm4_builder::GpuSqttBuilder(config, cmd_builder_, prims_, agent_info->timestamp_freq);
}

gpu_id_t Pm4FactoryAdapter::MapToLegacyGpuId() const {
  if (architecture_->IsMI100()) return MI100_GPU_ID;
  if (architecture_->IsMI200()) return MI200_GPU_ID;
  if (architecture_->IsMI300()) return MI300_GPU_ID;
  if (architecture_->IsMI350()) return MI350_GPU_ID;
  if (architecture_->IsGFX9())  return GFX9_GPU_ID;
  if (architecture_->IsGFX10()) return GFX10_GPU_ID;
  if (architecture_->IsGFX11()) return GFX11_GPU_ID;
  if (architecture_->IsGFX12()) return GFX12_GPU_ID;
  return INVAL_GPU_ID;
}

gpu_id_t Pm4FactoryAdapter::GetGpuId() const {
  return MapToLegacyGpuId();
}

uint32_t Pm4FactoryAdapter::GetShaderEnginesNumber() const {
  return architecture_->GetConfig().se_count;
}

uint32_t Pm4FactoryAdapter::GetShaderArraysNumber() const {
  return architecture_->GetConfig().sa_per_se_count;
}

uint32_t Pm4FactoryAdapter::GetComputeUnitNumber() const {
  return architecture_->GetConfig().cu_count;
}

const char* Pm4FactoryAdapter::GetGFX() const {
  gfx_name_ = architecture_->GetConfig().name;
  return gfx_name_.c_str();
}

uint32_t Pm4FactoryAdapter::GetXccNumber() const {
  return architecture_->GetConfig().xcc_count;
}

uint32_t Pm4FactoryAdapter::GetSpmSampleDelayMax() {
  return architecture_->GetSpmSampleDelayMax();
}

const GpuBlockInfo* Pm4FactoryAdapter::GetBlockInfo(const aqlprofile_pmc_event_t* event) const {
  return architecture_->GetBlockInfo(event->block_name);
}

const GpuBlockInfo* Pm4FactoryAdapter::GetBlockInfo(const event_t* event) const {
  return architecture_->GetBlockInfo(event->block_name);
}

const GpuBlockInfo* Pm4FactoryAdapter::GetBlockInfo(const uint32_t& block_id) const {
  return architecture_->GetBlockInfo(block_id);
}

size_t Pm4FactoryAdapter::GetNumEvents(uint32_t block_name) const {
  return architecture_->GetNumEventsForBlock(block_name);
}

size_t Pm4FactoryAdapter::GetBytesNeeded(uint32_t block_name) const {
  return architecture_->GetBytesNeededForBlock(block_name);
}

uint32_t Pm4FactoryAdapter::FindBlock(const char* name) const {
  return architecture_->FindBlockByName(name);
}

int Pm4FactoryAdapter::GetNumWGPs() const {
  return architecture_->GetNumWGPs();
}

int Pm4FactoryAdapter::GetAccumLowID() const {
  return architecture_->GetAccumLowID();
}

int Pm4FactoryAdapter::GetAccumHiID() const {
  return architecture_->GetAccumHiID();
}

}  // namespace aql_profile
