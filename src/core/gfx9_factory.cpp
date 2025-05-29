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

#include "core/gfx9_factory.h"
#include "def/gfx9_def.h"
#include "pm4/gfx9_cmd_builder.h"
#include "pm4/pmc_builder.h"
#include "pm4/sqtt_builder.h"

namespace aql_profile {

// Gfx factory init
void Gfx9Factory::Init(const AgentInfo* agent_info) {
  Pm4Factory::cmd_builder_ = new pm4_builder::Gfx9CmdBuilder(nullptr);
  if (Pm4Factory::cmd_builder_ == NULL) throw aql_profile_exc_msg("CmdBuilder allocation failed");

  // Mark and set the mode
  if (Pm4Factory::IsConcurrent()) {
    Pm4Factory::pmc_builder_ =
        new pm4_builder::GpuPmcBuilder<pm4_builder::Gfx9CmdBuilder, gfx9_cntx_prim, true>(
            agent_info);
  } else {
    Pm4Factory::pmc_builder_ =
        new pm4_builder::GpuPmcBuilder<pm4_builder::Gfx9CmdBuilder, gfx9_cntx_prim, false>(
            agent_info);
  }
  if (Pm4Factory::pmc_builder_ == NULL) throw aql_profile_exc_msg("PmcBuilder allocation failed");

  Pm4Factory::spm_builder_ =
      new pm4_builder::GpuSpmBuilder<pm4_builder::Gfx9CmdBuilder, gfx9_cntx_prim>(agent_info);
  if (Pm4Factory::spm_builder_ == NULL) throw aql_profile_exc_msg("SpmBuilder allocation failed");

  Pm4Factory::sqtt_builder_ =
      new pm4_builder::GpuSqttBuilder<pm4_builder::Gfx9CmdBuilder, gfx9_cntx_prim>(agent_info);
  if (Pm4Factory::sqtt_builder_ == NULL) throw aql_profile_exc_msg("SqttBuilder allocation failed");

  agent_info_ = agent_info;
}

void Gfx9Factory::Print(const GpuBlockInfo* block_info) {
  std::cout << "Block name: " << block_info->name << std::endl;
  std::cout << "\tInstances: " << block_info->instance_count << std::endl;
  std::cout << "\tMax Events: " << block_info->event_id_max << std::endl;
  std::cout << "\tCounters: " << block_info->counter_count << std::endl;
  auto counters = block_info->instance_count * block_info->counter_count;
  for (int i = 0; i < counters; ++i) {
    auto reg_info = block_info->counter_reg_info[i];
    std::cout << "\t   " << i << ": select_addr = 0x" << std::hex << reg_info.select_addr.offset
              << "(" << reg_info.select_addr.offset * 4 << ")"
              << ", control_addr = 0x" << reg_info.control_addr.offset << "("
              << reg_info.control_addr.offset * 4 << ")"
              << ", counter_addr_lo = 0x" << reg_info.register_addr_lo.offset << "("
              << reg_info.register_addr_lo.offset * 4 << ")"
              << ", counter_addr_hi = 0x" << reg_info.register_addr_hi.offset << "("
              << reg_info.register_addr_hi.offset * 4 << ")" << std::dec << std::endl;
  }
}

// GFX9 block table
const GpuBlockInfo* Gfx9Factory::block_table_[AQLPROFILE_BLOCKS_NUMBER] = {
    &CpcCounterBlockInfo, &CpfCounterBlockInfo, &GdsCounterBlockInfo, &GrbmCounterBlockInfo,
    &GrbmSeCounterBlockInfo, &SpiCounterBlockInfo, &SqCounterBlockInfo, &SqCsCounterBlockInfo,
    NULL /*GFX? SRBM*/, &SxCounterBlockInfo, &TaCounterBlockInfo, &TcaCounterBlockInfo,
    &TccCounterBlockInfo, &TcpCounterBlockInfo, &TdCounterBlockInfo,
    // MC blocks
    NULL /*MC_ARB*/, NULL /*MC_HUB*/, NULL /*MC_MCBVM*/, NULL /*MC_SEQ*/, &McVmL2CounterBlockInfo,
    NULL /*MC_XBAR*/, &AtcCounterBlockInfo, &AtcL2CounterBlockInfo, &GceaCounterBlockInfo,
    &RpbCounterBlockInfo,
    // System blocks
    NULL /*&SdmaCounterBlockInfo*/, NULL /*GL1A*/, NULL /*GL1C*/, NULL /*GL2A*/, NULL /*GL2C*/,
    NULL /*GCR*/, NULL /*GUS*/, NULL /*&UmcCounterBlockInfo*/
};

// Pm4Factory create mathods
Pm4Factory* Pm4Factory::Gfx9Create(const AgentInfo* agent_info) {
  auto p = new Gfx9Factory(agent_info);
  if (p == NULL) throw aql_profile_exc_msg("Gfx9Factory allocation failed");
  return p;
}

}  // namespace aql_profile
