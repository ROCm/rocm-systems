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
#include "def/gfx12_def.h"
#include "pm4/gfx12_cmd_builder.h"
#include "pm4/pmc_builder.h"
#include "pm4/sqtt_builder.h"

namespace aql_profile {

// Gfx12 factory class
class Gfx12Factory : public Pm4Factory {
 public:
  explicit Gfx12Factory(const AgentInfo* agent_info)
      : Pm4Factory(BlockInfoMap(block_table_, sizeof(block_table_))) {
    Init(agent_info);
  }
  Gfx12Factory(const GpuBlockInfo** table, const uint32_t& size, const AgentInfo* agent_info)
      : Pm4Factory(BlockInfoMap(table, size)) {
    Init(agent_info);
  }
  bool IsGFX12() const override { return true; }

 protected:
  void ConstructBuilders(const AgentInfo* agent_info);
  void ConstructTable(const AgentInfo* agent_info);
  void Init(const AgentInfo* agent_info) {
    agent_info_ = agent_info;
    ConstructBuilders(agent_info);
    ConstructTable(agent_info);
  }
  const GpuBlockInfo* block_table_[LastCounterBlockId + 1]{};
};

void Gfx12Factory::ConstructBuilders(const AgentInfo* agent_info) {
  Pm4Factory::cmd_builder_ = new pm4_builder::Gfx12CmdBuilder(nullptr);
  if (Pm4Factory::cmd_builder_ == NULL) throw aql_profile_exc_msg("CmdBuilder allocation failed");

  // Mark and set the mode
  if (Pm4Factory::IsConcurrent()) {
    Pm4Factory::pmc_builder_ =
        new pm4_builder::GpuPmcBuilder<pm4_builder::Gfx12CmdBuilder, gfx12_cntx_prim, true>(
            agent_info);
  } else {
    Pm4Factory::pmc_builder_ =
        new pm4_builder::GpuPmcBuilder<pm4_builder::Gfx12CmdBuilder, gfx12_cntx_prim, false>(
            agent_info);
  }
  if (Pm4Factory::pmc_builder_ == NULL) throw aql_profile_exc_msg("PmcBuilder allocation failed");

  Pm4Factory::spm_builder_ =
      new pm4_builder::GpuSpmBuilder<pm4_builder::Gfx12CmdBuilder, gfx12_cntx_prim>(agent_info);
  if (Pm4Factory::spm_builder_ == NULL) throw aql_profile_exc_msg("SpmBuilder allocation failed");

  Pm4Factory::sqtt_builder_ =
      new pm4_builder::GpuSqttBuilder<pm4_builder::Gfx12CmdBuilder, gfx12_cntx_prim>(agent_info);
  if (Pm4Factory::sqtt_builder_ == NULL) throw aql_profile_exc_msg("SqttBuilder allocation failed");
}

void Gfx12Factory::ConstructTable(const AgentInfo* agent_info) {
  auto agent_name = std::string_view(agent_info->name).substr(0, 7);
  // Global blocks
  block_table_[__BLOCK_ID(CHA)]         = &ChaCounterBlockInfo;
  block_table_[__BLOCK_ID(CHC)]         = &ChcCounterBlockInfo;
  block_table_[__BLOCK_ID_HSA(CPC)]     = &CpcCounterBlockInfo;
  block_table_[__BLOCK_ID_HSA(CPF)]     = &CpfCounterBlockInfo;
  block_table_[__BLOCK_ID(CPG)]         = &CpgCounterBlockInfo;
  block_table_[__BLOCK_ID_HSA(RPB)]         = &RpbCounterBlockInfo;
  block_table_[__BLOCK_ID(GC_UTCL2)]    = &GcUtcl2CounterBlockInfo;
  block_table_[__BLOCK_ID(GC_VML2)]     = &GcVml2CounterBlockInfo;
  block_table_[__BLOCK_ID(GC_VML2_SPM)] = &GcVml2SpmCounterBlockInfo;
  block_table_[__BLOCK_ID_HSA(GCEA)]    = &GceaCounterBlockInfo;
  block_table_[__BLOCK_ID_HSA(GCR)]     = &GcrCounterBlockInfo;
  block_table_[__BLOCK_ID_HSA(GL2A)]    = &Gl2aCounterBlockInfo;
  block_table_[__BLOCK_ID_HSA(GL2C)]    = &Gl2cCounterBlockInfo;
  block_table_[__BLOCK_ID_HSA(GRBM)]    = &GrbmCounterBlockInfo;
  block_table_[__BLOCK_ID(RLC)]         = &RlcCounterBlockInfo;
  block_table_[__BLOCK_ID_HSA(SDMA)]    = &SdmaCounterBlockInfo;
  // SE blocks
  block_table_[__BLOCK_ID(GC_UTCL1)]    = &GcUtcl1CounterBlockInfo;
  block_table_[__BLOCK_ID(GCEA_SE)]     = &GceaSeCounterBlockInfo;
  block_table_[__BLOCK_ID(GRBMH)]       = &GrbmhCounterBlockInfo;
  block_table_[__BLOCK_ID_HSA(SPI)]     = &SpiCounterBlockInfo;
  block_table_[__BLOCK_ID(SQG)]         = &SqgCounterBlockInfo;
  // SA blocks
  block_table_[__BLOCK_ID_HSA(GL1A)]    = &Gl1aCounterBlockInfo;
  block_table_[__BLOCK_ID_HSA(GL1C)]    = &Gl1cCounterBlockInfo;
  // WGP blocks
  block_table_[__BLOCK_ID_HSA(SQ)]      = &SqcCounterBlockInfo;
  block_table_[__BLOCK_ID_HSA(TA)]      = &TaCounterBlockInfo;
  block_table_[__BLOCK_ID_HSA(TCP)]     = &TcpCounterBlockInfo;
  block_table_[__BLOCK_ID_HSA(TD)]      = &TdCounterBlockInfo;

  if (agent_name == "gfx1201") {
    block_table_[__BLOCK_ID(CHC)]       = &gfx1201::ChcCounterBlockInfo;
    block_table_[__BLOCK_ID_HSA(GCEA)]  = &gfx1201::GceaCounterBlockInfo;
    block_table_[__BLOCK_ID(GCEA_SE)]   = &gfx1201::GceaSeCounterBlockInfo;
    block_table_[__BLOCK_ID_HSA(GL2C)]  = &gfx1201::Gl2cCounterBlockInfo;
  }
}

// Pm4Factory create mathods
Pm4Factory* Pm4Factory::Gfx12Create(const AgentInfo* agent_info) {
  auto p = new Gfx12Factory(agent_info);
  if (p == NULL) throw aql_profile_exc_msg("Gfx12Factory allocation failed");
  return p;
}

}  // namespace aql_profile
