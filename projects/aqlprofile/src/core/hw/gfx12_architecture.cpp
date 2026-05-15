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

// Generic GFX12 (gfx1200/gfx1201) architecture implementation.
// GFX12_VARIANT intentionally NOT defined here; gfx12_def.h defaults to
// GFX12_VARIANT_1200 so the generic gfx12 block-info symbols are used.

// Include aql_profile_v2.h first so that AQLPROFILE_BLOCK_NAME_* macros and
// assert are available before gfx12_def.h pulls in gfx12_block_info.h.
#include <cassert>
#include "aqlprofile-sdk/aql_profile_v2.h"
#include "def/gpu_block_info.h"
#include "core/hw/gfx12_architecture.hpp"
#include "def/gfx12_def.h"

#include <cstring>

namespace aql_profile {

// ---------------------------------------------------------------------------
// Gfx12Architecture
// ---------------------------------------------------------------------------

Gfx12Architecture::Gfx12Architecture(const AgentInfo* agent_info)
    : block_table_(nullptr), block_count_(0) {
  InitializeConfig(agent_info);
  InitializeBlockTable();
}

Gfx12Architecture::Gfx12Architecture(const AgentInfo* agent_info, DeferBlockTableInit)
    : block_table_(nullptr), block_count_(0) {
  InitializeConfig(agent_info);
  // InitializeBlockTable() intentionally omitted; derived class calls it.
}

void Gfx12Architecture::InitializeConfig(const AgentInfo* agent_info) {
  config_.name          = agent_info->name;
  config_.gfxip         = agent_info->gfxip;
  config_.se_count      = agent_info->se_num;
  config_.sa_per_se_count = agent_info->shader_arrays_per_se;
  config_.cu_count      = agent_info->cu_num;
  config_.xcc_count     = agent_info->xcc_num;
  config_.xcc_per_aid   = agent_info->xcc_per_aid;
  config_.aid_count     = (agent_info->xcc_per_aid > 0)
                              ? agent_info->xcc_num / agent_info->xcc_per_aid
                              : 1;
  config_.wgp_count     = agent_info->cu_num / 2;

  // GFX12 capability flags
  config_.has_spm_core1                = false;
  config_.has_wptr_relative_addressing = false;
  config_.has_sqtt_status2_register    = true;
  config_.needs_sqtt_header_packet     = false;
  config_.has_sriov_spm_restriction    = false;
  config_.has_asymmetric_cu_design     = false;
  config_.supports_spm_v2              = false;
  config_.spm_sample_delay_max         = 0;
  config_.sqtt_header_version          = 0;
}

void Gfx12Architecture::InitializeBlockTable() {
  static const GpuBlockInfo* table[AQLPROFILE_BLOCKS_NUMBER]{};

  // Global blocks
  table[__BLOCK_ID(CHA)]         = &ChaCounterBlockInfo;
  table[__BLOCK_ID(CHC)]         = &ChcCounterBlockInfo;
  table[__BLOCK_ID_HSA(CPC)]     = &CpcCounterBlockInfo;
  table[__BLOCK_ID_HSA(CPF)]     = &CpfCounterBlockInfo;
  table[__BLOCK_ID(CPG)]         = &CpgCounterBlockInfo;
  table[__BLOCK_ID_HSA(RPB)]     = &RpbCounterBlockInfo;
  table[__BLOCK_ID(GC_UTCL2)]    = &GcUtcl2CounterBlockInfo;
  table[__BLOCK_ID(GC_VML2)]     = &GcVml2CounterBlockInfo;
  table[__BLOCK_ID(GC_VML2_SPM)] = &GcVml2SpmCounterBlockInfo;
  table[__BLOCK_ID_HSA(GCEA)]    = &GceaCounterBlockInfo;
  table[__BLOCK_ID_HSA(GCR)]     = &GcrCounterBlockInfo;
  table[__BLOCK_ID_HSA(GL2A)]    = &Gl2aCounterBlockInfo;
  table[__BLOCK_ID_HSA(GL2C)]    = &Gl2cCounterBlockInfo;
  table[__BLOCK_ID_HSA(GRBM)]    = &GrbmCounterBlockInfo;
  table[__BLOCK_ID(RLC)]         = &RlcCounterBlockInfo;
  table[__BLOCK_ID_HSA(SDMA)]    = &SdmaCounterBlockInfo;
  // SE blocks
  table[__BLOCK_ID(GC_UTCL1)]    = &GcUtcl1CounterBlockInfo;
  table[__BLOCK_ID(GCEA_SE)]     = &GceaSeCounterBlockInfo;
  table[__BLOCK_ID(GRBMH)]       = &GrbmhCounterBlockInfo;
  table[__BLOCK_ID_HSA(SPI)]     = &SpiCounterBlockInfo;
  table[__BLOCK_ID(SQG)]         = &SqgCounterBlockInfo;
  // SA blocks
  table[__BLOCK_ID_HSA(GL1A)]    = &Gl1aCounterBlockInfo;
  table[__BLOCK_ID_HSA(GL1C)]    = &Gl1cCounterBlockInfo;
  // WGP blocks
  table[__BLOCK_ID_HSA(SQ)]      = &SqcCounterBlockInfo;
  table[__BLOCK_ID_HSA(TA)]      = &TaCounterBlockInfo;
  table[__BLOCK_ID_HSA(TCP)]     = &TcpCounterBlockInfo;
  table[__BLOCK_ID_HSA(TD)]      = &TdCounterBlockInfo;

  block_table_ = table;
  block_count_ = AQLPROFILE_BLOCKS_NUMBER;
}

// ---------------------------------------------------------------------------
// Gfx1201Architecture
// ---------------------------------------------------------------------------

Gfx1201Architecture::Gfx1201Architecture(const AgentInfo* agent_info)
    : Gfx12Architecture(agent_info, DeferBlockTableInit{}) {
  InitializeBlockTable();
}

void Gfx1201Architecture::InitializeBlockTable() {
  // Build the gfx1200 base table, then patch the four blocks that differ on
  // gfx1201 (higher instance counts for CHC, GCEA, GCEA_SE, GL2C).
  Gfx12Architecture::InitializeBlockTable();
  block_table_[__BLOCK_ID(CHC)]      = &gfx1201::ChcCounterBlockInfo;
  block_table_[__BLOCK_ID_HSA(GCEA)] = &gfx1201::GceaCounterBlockInfo;
  block_table_[__BLOCK_ID(GCEA_SE)]  = &gfx1201::GceaSeCounterBlockInfo;
  block_table_[__BLOCK_ID_HSA(GL2C)] = &gfx1201::Gl2cCounterBlockInfo;
}

const GpuBlockInfo* Gfx12Architecture::GetBlockInfo(uint32_t block_id) const {
  if (block_id >= block_count_) return nullptr;
  return block_table_[block_id];
}

uint32_t Gfx12Architecture::FindBlockByName(const char* name) const {
  for (uint32_t i = 0; i < block_count_; ++i) {
    const GpuBlockInfo* b = block_table_[i];
    if (b && strcmp(b->name, name) == 0) return i;
  }
  return UINT32_MAX;
}

}  // namespace aql_profile

