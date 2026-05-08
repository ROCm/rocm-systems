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

#include "core/architectures/mi100_architecture.hpp"
#include "aqlprofile-sdk/aql_profile_v2.h"
#include "def/gfx9_def.h"

#include <mutex>

namespace aql_profile {

using namespace gfxip::gfx9;

Mi100Architecture::Mi100Architecture(const AgentInfo* agent_info) {
  InitializeConfig(agent_info);
  InitializeRegisterSchema();
  InitializeBlockTable();
}

void Mi100Architecture::InitializeConfig(const AgentInfo* agent_info) {
  Gfx9Architecture::InitializeConfig(agent_info);

  config_.name = "MI100";
  config_.has_spm_core1 = true;
  config_.spm_sample_delay_max = 0x34;
}

void Mi100Architecture::InitializeBlockTable() {
  static const GpuBlockInfo* table[AQLPROFILE_BLOCKS_NUMBER]{};
  static std::once_flag init_flag;

  std::call_once(init_flag, []() {
    // Base GFX9 table (statically initialised).
    static const GpuBlockInfo* gfx9_base[AQLPROFILE_BLOCKS_NUMBER] = {
        &CpcCounterBlockInfo,       // CPC
        &CpfCounterBlockInfo,       // CPF
        &GdsCounterBlockInfo,       // GDS
        &GrbmCounterBlockInfo,      // GRBM
        &GrbmSeCounterBlockInfo,    // GRBMSe
        &SpiCounterBlockInfo,       // SPI
        &SqCounterBlockInfo,        // SQ
        &SqCsCounterBlockInfo,      // SQCs
        NULL,                       // SRBM
        &SxCounterBlockInfo,        // SX
        &TaCounterBlockInfo,        // TA
        &TcaCounterBlockInfo,       // TCA
        &TccCounterBlockInfo,       // TCC
        &TcpCounterBlockInfo,       // TCP
        &TdCounterBlockInfo,        // TD
        // MC blocks
        NULL,                       // MC_ARB
        NULL,                       // MC_HUB
        NULL,                       // MC_MCBVM
        NULL,                       // MC_SEQ
        &McVmL2CounterBlockInfo,    // McVmL2
        NULL,                       // MC_XBAR
        &AtcCounterBlockInfo,       // ATC
        &AtcL2CounterBlockInfo,     // ATCL2
        &GceaCounterBlockInfo,      // GCEA
        &RpbCounterBlockInfo,       // RPB
        // System blocks
        NULL,                       // SDMA
        NULL,                       // GL1A
        NULL,                       // GL1C
        NULL,                       // GL2A
        NULL,                       // GL2C
        NULL,                       // GCR
        NULL,                       // GUS
        NULL,                       // UMC
    };

    for (unsigned i = 0; i < AQLPROFILE_BLOCKS_NUMBER; ++i)
      table[i] = gfx9_base[i];

    // Apply MI100-specific overrides (ported from Mi100Factory constructor).
    auto copy = [&](unsigned id) -> GpuBlockInfo* {
      if (id < AQLPROFILE_BLOCKS_NUMBER && gfx9_base[id])
        return new GpuBlockInfo(*gfx9_base[id]);
      return nullptr;
    };

    if (auto* b = copy(SqCounterBlockId))   { b->event_id_max = 303;                                table[SqCounterBlockId]   = b; }
    if (auto* b = copy(TcpCounterBlockId))  { b->event_id_max = 87;                                 table[TcpCounterBlockId]  = b; }
    if (auto* b = copy(TccCounterBlockId))  { b->instance_count = 32; b->event_id_max = 295;        table[TccCounterBlockId]  = b; }
    if (auto* b = copy(TcaCounterBlockId))  { b->instance_count = 32; b->event_id_max = 58;         table[TcaCounterBlockId]  = b; }
    if (auto* b = copy(GceaCounterBlockId)) { b->instance_count = 32; b->event_id_max = 83;         table[GceaCounterBlockId] = b; }
    if (auto* b = copy(SdmaCounterBlockId)) { b->instance_count = gfx9_cntx_prim::SDMA_COUNTER_BLOCK_NUM_INSTANCES; table[SdmaCounterBlockId] = b; }
    if (auto* b = copy(UmcCounterBlockId))  { b->counter_count  = 6;                                table[UmcCounterBlockId]  = b; }
  });

  block_table_ = table;
  block_count_ = AQLPROFILE_BLOCKS_NUMBER;
}

}  // namespace aql_profile
