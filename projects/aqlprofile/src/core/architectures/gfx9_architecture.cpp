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

#include "core/architectures/gfx9_architecture.hpp"
#include "core/gfx9_factory.h"
#include "def/gfx9_def.h"
#include "pm4/gfx9_cmd_builder.h"
#include "pm4/gfx9_primitives_provider.hpp"

namespace aql_profile {

using namespace gfxip::gfx9;

Gfx9Architecture::Gfx9Architecture(const AgentInfo* agent_info)
    : block_table_(nullptr), block_count_(0) {
  InitializeConfig(agent_info);
  InitializeRegisterSchema();
  InitializeBlockTable();
}

void Gfx9Architecture::InitializeConfig(const AgentInfo* agent_info) {
  config_.gfxip = agent_info->gfxip;
  config_.name = agent_info->name;
  config_.se_count = agent_info->se_num;
  config_.sa_per_se_count = agent_info->shader_arrays_per_se;
  config_.cu_count = agent_info->cu_num;
  config_.xcc_count = agent_info->xcc_num;
  config_.aid_count = 1;  // GFX9 is single-AID

  // GFX9 specific capabilities
  config_.supports_pmc = true;
  config_.supports_spm = true;
  config_.supports_sqtt = true;
  config_.supports_concurrent = true;
  config_.has_grbm_perfcounter = true;
  config_.has_aid_aware_counters = false;
  config_.has_spm_core1 = false;  // Only MI100/MI200
  config_.spm_sq_32bit_mode = true;
  config_.spm_sample_delay_max = 0;  // Base GFX9 - MI variants override
  config_.sqtt_buffer_alignment = 0x1000;

  // WGP count approximation for GFX9 (pre-RDNA)
  config_.wgp_count = 0;  // Not applicable for GFX9, will use CU-based fallback
}

void Gfx9Architecture::InitializeRegisterSchema() {
  // GRBM registers
  register_schema_.DefineRegister(RegisterId::GRBM_GFX_INDEX,
                                  gfx9_cntx_prim::GRBM_GFX_INDEX_ADDR.offset);
  register_schema_.DefineRegister(RegisterId::CP_PERFMON_CNTL,
                                  gfx9_cntx_prim::CP_PERFMON_CNTL_ADDR.offset);

  // SQ (Shader Quad) registers
  register_schema_.DefineRegister(RegisterId::SQ_PERFCOUNTER_CTRL,
                                  gfx9_cntx_prim::SQ_PERFCOUNTER_CTRL_ADDR.offset);

  // SQTT (Thread Trace) registers
  register_schema_.DefineRegister(RegisterId::SQTT_BUF_BASE,
                                  gfx9_cntx_prim::SQ_THREAD_TRACE_BASE_ADDR.offset);
  register_schema_.DefineRegister(RegisterId::SQTT_BUF_SIZE,
                                  gfx9_cntx_prim::SQ_THREAD_TRACE_SIZE_ADDR.offset);
  register_schema_.DefineRegister(RegisterId::SQTT_BUF_STATUS,
                                  gfx9_cntx_prim::SQ_THREAD_TRACE_STATUS_ADDR.offset);

  // Note: SPM registers would be added here if needed
}

void Gfx9Architecture::InitializeBlockTable() {
  // Gfx9Factory::block_table_ is the shared static GFX9 block table.
  block_table_ = Gfx9Factory::block_table_;
  block_count_ = AQLPROFILE_BLOCKS_NUMBER;
}

const GpuBlockInfo* Gfx9Architecture::GetBlockInfo(uint32_t block_id) const {
  if (block_id >= block_count_) {
    return nullptr;
  }
  return block_table_[block_id];
}

uint32_t Gfx9Architecture::FindBlockByName(const char* name) const {
  for (uint32_t i = 0; i < block_count_; ++i) {
    const GpuBlockInfo* block = block_table_[i];
    if (block && strcmp(block->name, name) == 0) {
      return i;
    }
  }
  return UINT32_MAX;
}

uint32_t Gfx9Architecture::GetBlockCount() const {
  return block_count_;
}

pm4_builder::CmdBuilder* Gfx9Architecture::CreateCmdBuilder() const {
  return new pm4_builder::Gfx9CmdBuilder(nullptr);
}

pm4_builder::PrimitivesProvider* Gfx9Architecture::CreatePrimitivesProvider() const {
  return new pm4_builder::Gfx9PrimitivesProvider();
}

}  // namespace aql_profile
