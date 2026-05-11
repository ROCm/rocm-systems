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

#include "core/architectures/mi300_architecture.hpp"
#include "aqlprofile-sdk/aql_profile_v2.h"
#include "def/gfx9_def.h"
#include "def/gpu_block_info.h"
#include "pm4/pmc_builder.h"

#include <mutex>

namespace aql_profile {

using namespace gfxip::gfx9;

Mi300Architecture::Mi300Architecture(const AgentInfo* agent_info) {
  InitializeConfig(agent_info);
  InitializeRegisterSchema();
  InitializeBlockTable();
}

void Mi300Architecture::InitializeConfig(const AgentInfo* agent_info) {
  Gfx9Architecture::InitializeConfig(agent_info);

  config_.name = "MI300";
  config_.xcc_count = agent_info->xcc_num;
  config_.aid_count = 4;
  config_.has_aid_aware_counters = true;
  config_.has_spm_core1 = false;
  config_.has_sriov_spm_restriction = true;
  config_.supports_spm_v2 = true;
  config_.xcc_per_aid = 2;
  config_.sqtt_header_version = 5;
  config_.spm_sample_delay_max = 0x27;
}

void Mi300Architecture::InitializeBlockTable() {
  static const GpuBlockInfo* table[AQLPROFILE_BLOCKS_NUMBER]{};
  static std::once_flag init_flag;

  std::call_once(init_flag, []() {
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

    // Apply MI300-specific overrides (ported from Mi300Factory constructor).
    auto copy = [&](unsigned id) -> GpuBlockInfo* {
      if (id < AQLPROFILE_BLOCKS_NUMBER && gfx9_base[id])
        return new GpuBlockInfo(*gfx9_base[id]);
      return nullptr;
    };

    if (auto* b = copy(SqCounterBlockId))   { b->event_id_max = 373;                                              table[SqCounterBlockId]   = b; }
    if (auto* b = copy(TcpCounterBlockId))  { b->event_id_max = 84;                                               table[TcpCounterBlockId]  = b; }
    if (auto* b = copy(TccCounterBlockId))  { b->instance_count = 16;  b->event_id_max = 199;                     table[TccCounterBlockId]  = b; }
    if (auto* b = copy(TcaCounterBlockId))  { b->instance_count = 32;  b->event_id_max = 34;                      table[TcaCounterBlockId]  = b; }
    if (auto* b = copy(GceaCounterBlockId)) { b->instance_count = 32;  b->event_id_max = 82;                      table[GceaCounterBlockId] = b; }
    if (auto* b = copy(SdmaCounterBlockId)) { b->instance_count = 4 * pm4_builder::MAX_AID;                       table[SdmaCounterBlockId] = b; }
    if (auto* b = copy(UmcCounterBlockId))  { b->counter_count  = 11;  b->instance_count = 32 * pm4_builder::MAX_AID; table[UmcCounterBlockId]  = b; }
    // RPB and ATC: MI300 has 4 instances each (vs. 1 in base GFX9).
    { auto* b = new GpuBlockInfo(RpbCounterBlockInfo); b->instance_count = 4; table[RpbCounterBlockId] = b; }
    { auto* b = new GpuBlockInfo(AtcCounterBlockInfo); b->instance_count = 4; table[AtcCounterBlockId] = b; }
  });

  block_table_ = table;
  block_count_ = AQLPROFILE_BLOCKS_NUMBER;
}

size_t Mi300Architecture::GetBytesNeededForBlock(uint32_t block_id) const {
  const GpuBlockInfo* block_info = GetBlockInfo(block_id);
  if (!block_info) return 0;

  // AID-aware blocks (UMC, RPB, ATC) are distributed across AIDs, not XCCs.
  // Their instance_count already reflects the total across all AIDs.
  if (block_info->attr & CounterBlockAidAttr) {
    return block_info->instance_count * sizeof(uint64_t);
  }

  return HardwareArchitecture::GetBytesNeededForBlock(block_id);
}

}  // namespace aql_profile
