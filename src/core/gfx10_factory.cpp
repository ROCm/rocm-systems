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

#include "core/pm4_factory.h"
#include "def/gfx10_def.h"
#include "pm4/gfx10_cmd_builder.h"
#include "pm4/pmc_builder.h"
#include "pm4/sqtt_builder.h"

namespace aql_profile {

// Gfx10 factory class
class Gfx10Factory : public Pm4Factory {
 public:
  explicit Gfx10Factory(const AgentInfo* agent_info)
      : Pm4Factory(BlockInfoMap(block_table_, sizeof(block_table_))) {
    Init(agent_info);
  }
  Gfx10Factory(const GpuBlockInfo** table, const uint32_t& size, const AgentInfo* agent_info)
      : Pm4Factory(BlockInfoMap(table, size)) {
    Init(agent_info);
  }
  bool IsGFX10() const override { return true; }

  virtual int GetAccumLowID() const override { return 1; };
  virtual int GetAccumHiID() const override { return 1; };

 protected:
  // void ConstructTable(const AgentInfo* agent_info);
  void Init(const AgentInfo* agent_info);
  // void ConstructBuilders(const AgentInfo* agent_info);
  static const GpuBlockInfo* block_table_[HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER];
};

// Gfx builders init
// void Gfx10Factory::ConstructBuilders(const AgentInfo* agent_info) {
void Gfx10Factory::Init(const AgentInfo* agent_info) {
  Pm4Factory::cmd_builder_ = new pm4_builder::Gfx10CmdBuilder(nullptr);
  if (Pm4Factory::cmd_builder_ == NULL) throw aql_profile_exc_msg("CmdBuilder allocation failed");

  // Mark and set the mode
  if (Pm4Factory::IsConcurrent()) {
    Pm4Factory::pmc_builder_ =
        new pm4_builder::GpuPmcBuilder<pm4_builder::Gfx10CmdBuilder, gfx10_cntx_prim, true>(
            agent_info);
  } else {
    Pm4Factory::pmc_builder_ =
        new pm4_builder::GpuPmcBuilder<pm4_builder::Gfx10CmdBuilder, gfx10_cntx_prim, false>(
            agent_info);
  }
  if (Pm4Factory::pmc_builder_ == NULL) throw aql_profile_exc_msg("PmcBuilder allocation failed");

  Pm4Factory::spm_builder_ =
      new pm4_builder::GpuSpmBuilder<pm4_builder::Gfx10CmdBuilder, gfx10_cntx_prim>(agent_info);
  if (Pm4Factory::spm_builder_ == NULL) throw aql_profile_exc_msg("SpmBuilder allocation failed");

  Pm4Factory::sqtt_builder_ =
      new pm4_builder::GpuSqttBuilder<pm4_builder::Gfx10CmdBuilder, gfx10_cntx_prim>(agent_info);
  if (Pm4Factory::sqtt_builder_ == NULL) throw aql_profile_exc_msg("SqttBuilder allocation failed");

  agent_info_ = agent_info;
}

// GFX10 block table
const GpuBlockInfo* Gfx10Factory::block_table_[HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER] = {
    &CpcCounterBlockInfo, &CpfCounterBlockInfo, &GdsCounterBlockInfo, &GrbmCounterBlockInfo,
    NULL /*&GrbmSeCounterBlockInfo*/, &SpiCounterBlockInfo, &SqCounterBlockInfo,
    NULL /*&SqCsCounterBlockInfo*/, NULL /*GFX8 SRBM*/, &SxCounterBlockInfo, &TaCounterBlockInfo,
    NULL /*&TcaCounterBlockInfo*/, NULL /*&TccCounterBlockInfo*/, NULL /*&TcpCounterBlockInfo*/,
    NULL /*&TdCounterBlockInfo*/,
    // MC blocks
    NULL /*MC_ARB*/, NULL /*MC_HUB*/, NULL /*MC_MCBVM*/, NULL /*MC_SEQ*/,
    NULL /*&McVmL2CounterBlockInfo*/, NULL /*MC_XBAR*/, NULL /*&AtcCounterBlockInfo*/,
    NULL /*&AtcL2CounterBlockInfo*/, &GceaCounterBlockInfo, NULL /*&RpbCounterBlockInfo*/,
    // System blocks
    NULL /*&SdmaCounterBlockInfo*/,
    // new navi blocks
    &Gl1aCounterBlockInfo, &Gl1cCounterBlockInfo, &Gl2aCounterBlockInfo, &Gl2cCounterBlockInfo,
    &GcrCounterBlockInfo, &GusCounterBlockInfo};

// Pm4Factory create mathods
Pm4Factory* Pm4Factory::Gfx10Create(const AgentInfo* agent_info) {
  auto p = new Gfx10Factory(agent_info);
  if (p == NULL) throw aql_profile_exc_msg("Gfx10Factory allocation failed");
  return p;
}

}  // namespace aql_profile
