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

#include "core/architectures/gfx11_architecture.hpp"
#include "core/factory_block_tables.h"
// gfx11_def.h must come before gfx11_block_table.h and gfx11_primitives.h
// to define the required macros.
#include "def/gfx11_def.h"
#include "pm4/gfx11_cmd_builder.h"

namespace aql_profile {

using namespace gfxip::gfx11;

Gfx11Architecture::Gfx11Architecture(const AgentInfo* agent_info)
    : block_table_(nullptr), block_count_(0) {
  InitializeConfig(agent_info);
  InitializeRegisterSchema();
  InitializeBlockTable();
}

void Gfx11Architecture::InitializeConfig(const AgentInfo* agent_info) {
  config_.gfxip = agent_info->gfxip;
  config_.name = agent_info->name;
  config_.se_count = agent_info->se_num;
  config_.sa_per_se_count = agent_info->shader_arrays_per_se;
  config_.cu_count = agent_info->cu_num;
  config_.xcc_count = agent_info->xcc_num;
  config_.aid_count = 1;

  // GFX11 WGP count
  config_.wgp_count = agent_info->cu_num / 2;

  // GFX11 capabilities
  config_.supports_pmc = true;
  config_.supports_spm = false;  // SPM not supported on RDNA 3
  config_.supports_sqtt = true;
  config_.supports_concurrent = true;
  config_.has_grbm_perfcounter = true;
  config_.has_aid_aware_counters = false;
  config_.has_spm_core1 = false;
  config_.spm_sq_32bit_mode = false;
  config_.spm_sample_delay_max = 0;
  config_.sqtt_buffer_alignment = 0x1000;
}

void Gfx11Architecture::InitializeRegisterSchema() {
  // GRBM registers
  register_schema_.DefineRegister(RegisterId::GRBM_GFX_INDEX,
                                  gfx11_cntx_prim::GRBM_GFX_INDEX_ADDR.offset);
  register_schema_.DefineRegister(RegisterId::CP_PERFMON_CNTL,
                                  gfx11_cntx_prim::CP_PERFMON_CNTL_ADDR.offset);

  // SQ registers
  register_schema_.DefineRegister(RegisterId::SQ_PERFCOUNTER_CTRL,
                                  gfx11_cntx_prim::SQ_PERFCOUNTER_CTRL_ADDR.offset);

  // SQTT registers (GFX11 has different register layout)
  register_schema_.DefineRegister(RegisterId::SQTT_BUF_BASE,
                                  gfx11_cntx_prim::SQ_THREAD_TRACE_BUF0_BASE_LO_ADDR.offset);
  register_schema_.DefineRegister(RegisterId::SQTT_BUF_SIZE,
                                  gfx11_cntx_prim::SQ_THREAD_TRACE_BUF0_SIZE_ADDR.offset);
  register_schema_.DefineRegister(RegisterId::SQTT_BUF_STATUS,
                                  gfx11_cntx_prim::SQ_THREAD_TRACE_STATUS_ADDR.offset);
}

void Gfx11Architecture::InitializeBlockTable() {
  block_table_ = GetGfx11BlockTable();
  block_count_ = static_cast<uint32_t>(GetGfx11BlockTableSize());
}

const GpuBlockInfo* Gfx11Architecture::GetBlockInfo(uint32_t block_id) const {
  if (block_id >= block_count_) {
    return nullptr;
  }
  return block_table_[block_id];
}

uint32_t Gfx11Architecture::FindBlockByName(const char* name) const {
  for (uint32_t i = 0; i < block_count_; ++i) {
    const GpuBlockInfo* block = block_table_[i];
    if (block && strcmp(block->name, name) == 0) {
      return i;
    }
  }
  return UINT32_MAX;
}

uint32_t Gfx11Architecture::GetBlockCount() const {
  return block_count_;
}

pm4_builder::CmdBuilder* Gfx11Architecture::CreateCmdBuilder() const {
  return new pm4_builder::Gfx11CmdBuilder(nullptr);
}

uint32_t Gfx11Architecture::GetNumWGPs() const {
  return config_.wgp_count;
}

}  // namespace aql_profile
