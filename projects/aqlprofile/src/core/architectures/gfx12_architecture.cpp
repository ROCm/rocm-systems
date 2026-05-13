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
// GFX12_VARIANT intentionally NOT defined here; gfx12_def.h defaults to GFX12_VARIANT_1200.
#include "core/architectures/gfx12_architecture.hpp"
#include "aqlprofile-sdk/aql_profile_v2.h"
#include "def/gfx12_def.h"
#include "pm4/gfx12_cmd_builder.h"
#include "pm4/gfx12_primitives_provider.hpp"

namespace aql_profile {

// ---------------------------------------------------------------------------
// Gfx12Architecture
// ---------------------------------------------------------------------------

Gfx12Architecture::Gfx12Architecture(const AgentInfo* agent_info)
    : block_table_(nullptr), block_count_(0) {
  InitializeConfig(agent_info);
  InitializeRegisterSchema();
  InitializeBlockTable();
}

void Gfx12Architecture::InitializeConfig(const AgentInfo* agent_info) {
  config_.gfxip = agent_info->gfxip;
  config_.name = agent_info->name;
  config_.se_count = agent_info->se_num;
  config_.sa_per_se_count = agent_info->shader_arrays_per_se;
  config_.cu_count = agent_info->cu_num;
  config_.xcc_count = agent_info->xcc_num;
  config_.aid_count = 1;
  config_.wgp_count = agent_info->cu_num / 2;

  config_.supports_pmc = true;
  config_.supports_spm = true;
  config_.supports_sqtt = true;
  config_.supports_concurrent = true;
  config_.has_grbm_perfcounter = true;
  config_.has_aid_aware_counters = false;
  config_.has_spm_core1 = false;
  config_.spm_sq_32bit_mode = false;
  config_.supports_spm_v2 = false;
  config_.has_sqtt_status2_register = true;
  config_.has_wptr_relative_addressing = false;
  config_.needs_sqtt_header_packet = false;
  config_.spm_sample_delay_max = 0;
  config_.sqtt_buffer_alignment = 0x1000;
}

void Gfx12Architecture::InitializeRegisterSchema() {
  // GFX12 register offsets are captured in gfx12_cntx_prim; builders use
  // them directly.  The schema records the key addresses for callers that
  // query them through the architecture interface.
  register_schema_.DefineRegister(RegisterId::GRBM_GFX_INDEX,
                                  gfx12_cntx_prim::GRBM_GFX_INDEX_ADDR.offset);
  register_schema_.DefineRegister(RegisterId::CP_PERFMON_CNTL,
                                  gfx12_cntx_prim::CP_PERFMON_CNTL_ADDR.offset);
  register_schema_.DefineRegister(RegisterId::SQ_PERFCOUNTER_CTRL,
                                  gfx12_cntx_prim::SQ_PERFCOUNTER_CTRL_ADDR.offset);
  register_schema_.DefineRegister(RegisterId::SQTT_BUF_BASE,
                                  gfx12_cntx_prim::SQ_THREAD_TRACE_BUF0_BASE_LO_ADDR.offset);
  register_schema_.DefineRegister(RegisterId::SQTT_BUF_SIZE,
                                  gfx12_cntx_prim::SQ_THREAD_TRACE_BUF0_SIZE_ADDR.offset);
  register_schema_.DefineRegister(RegisterId::SQTT_BUF_STATUS,
                                  gfx12_cntx_prim::SQ_THREAD_TRACE_STATUS_ADDR.offset);
}

void Gfx12Architecture::InitializeBlockTable() {
  static const GpuBlockInfo* table[AQLPROFILE_BLOCKS_NUMBER]{};

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

const GpuBlockInfo* Gfx12Architecture::GetBlockInfo(uint32_t block_id) const {
  if (block_id >= block_count_) return nullptr;
  return block_table_[block_id];
}

uint32_t Gfx12Architecture::FindBlockByName(const char* name) const {
  for (uint32_t i = 0; i < block_count_; ++i) {
    const GpuBlockInfo* block = block_table_[i];
    if (block && strcmp(block->name, name) == 0) return i;
  }
  return UINT32_MAX;
}

uint32_t Gfx12Architecture::GetBlockCount() const {
  return block_count_;
}

pm4_builder::CmdBuilder* Gfx12Architecture::CreateCmdBuilder(const reg_base_offset_table* table) const {
  return new pm4_builder::Gfx12CmdBuilder(table);
}

pm4_builder::PrimitivesProvider* Gfx12Architecture::CreatePrimitivesProvider() const {
  return new pm4_builder::Gfx12PrimitivesProvider();
}

}  // namespace aql_profile
