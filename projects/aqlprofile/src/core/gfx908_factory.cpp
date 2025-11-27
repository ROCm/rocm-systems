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
#include "def/gfx908_def.h"
#include "pm4/gfx9_cmd_builder.h"
#include "pm4/pmc_builder.h"
#include "pm4/sqtt_builder.h"

namespace aql_profile {

const GpuBlockInfo* Mi100Factory::block_table_[AQLPROFILE_BLOCKS_NUMBER] = {};

static const uint32_t CpgBlockDelayValue[] = {0x32};
static const uint32_t CpcBlockDelayValue[] = {0x30};
static const uint32_t CpfBlockDelayValue[] = {0x30};
static const uint32_t GdsBlockDelayValue[] = {0x34};
static const uint32_t TccBlockDelayValue[] = {
    0x08, 0x0c, 0x0c, 0x0e, 0x14, 0x10, 0x1e, 0x22, 0x0a, 0x0e, 0x0c, 0x10, 0x14, 0x12, 0x22, 0x28,
    0x14, 0x16, 0x18, 0x18, 0x20, 0x1c, 0x28, 0x2e, 0x14, 0x16, 0x18, 0x18, 0x20, 0x1c, 0x2a, 0x30};
static const uint32_t TcaBlockDelayValue[] = {0x18, 0x1c, 0x24, 0x24};

static const uint32_t SxBlockDelayValue[] = {0x00, 0x01, 0x0a, 0x12, 0x00, 0x02, 0x0a, 0x12};
static const uint32_t TaBlockDelayValue[] = {
    0x20, 0x1e, 0x1c, 0x1a, 0x18, 0x16, 0x14, 0x12, 0x10, 0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04, 0x02,
    0x20, 0x1e, 0x1c, 0x1a, 0x18, 0x16, 0x14, 0x12, 0x10, 0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04, 0x02,
    0x20, 0x1e, 0x1c, 0x1a, 0x18, 0x16, 0x14, 0x12, 0x10, 0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04, 0x02,
    0x26, 0x24, 0x22, 0x20, 0x1e, 0x1c, 0x1a, 0x18, 0x16, 0x14, 0x12, 0x10, 0x0e, 0x0c, 0x0a, 0x08,
    0x19, 0x16, 0x14, 0x12, 0x10, 0x0e, 0x0c, 0x0a, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01,
    0x22, 0x20, 0x1e, 0x1c, 0x1a, 0x18, 0x16, 0x14, 0x12, 0x10, 0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04,
    0x22, 0x20, 0x1e, 0x1c, 0x1a, 0x18, 0x16, 0x14, 0x12, 0x10, 0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04,
    0x26, 0x24, 0x22, 0x20, 0x1e, 0x1c, 0x1a, 0x18, 0x16, 0x14, 0x12, 0x10, 0x0e, 0x0c, 0x0a, 0x08};
static const uint32_t SpiBlockDelayValue[] = {0x11, 0x1b, 0x20, 0x28, 0x15, 0x1b, 0x22, 0x2a};
static const uint32_t SqBlockDelayValue[] = {0x12, 0x1c, 0x20, 0x2c, 0x16, 0x1c, 0x24, 0x2c};

void Mi100Factory::InitSpmBlockDelayTable() {
  cu_block_delay_table_size = sizeof(TaBlockDelayValue) / sizeof(TaBlockDelayValue[0]);
  const uint32_t** p;
  // Global Blocks
  p = spm_block_delay_global;
  *p++ = CpgBlockDelayValue;  // CPG = 0
  *p++ = CpcBlockDelayValue;  // CPC = 1
  *p++ = CpfBlockDelayValue;  // CPF = 2
  *p++ = GdsBlockDelayValue;  // GDS = 3
  *p++ = TccBlockDelayValue;  // TCC = 4
  *p++ = TcaBlockDelayValue;  // TCA = 5
  *p++ = NULL;                // IA = 6
  *p++ = NULL;                // TCS = 7
  // SE Blocks
  p = spm_block_delay_se;
  *p++ = NULL;                // CB = 0
  *p++ = NULL;                // DB = 1
  *p++ = NULL;                // PA = 2
  *p++ = SxBlockDelayValue;   // SSX = 3
  *p++ = NULL;                // SC = 4
  *p++ = TaBlockDelayValue;   // TA = 5
  *p++ = TaBlockDelayValue;   // TD = 6  - Same as TA
  *p++ = TaBlockDelayValue;   // TCP = 7 - Same as TA
  *p++ = SpiBlockDelayValue;  // SPI = 8
  *p++ = SqBlockDelayValue;   // SQG = 9
  *p++ = NULL;                // VGT = 10
}

Mi100Factory::Mi100Factory(const AgentInfo* agent_info)
    : Gfx9Factory(block_table_, sizeof(block_table_), agent_info) {
  InitSpmBlockDelayTable();
  for (unsigned i = 0; i < AQLPROFILE_BLOCKS_NUMBER; ++i) {
    const GpuBlockInfo* base_table_ptr = Gfx9Factory::block_table_[i];
    if (base_table_ptr == NULL) continue;
    GpuBlockInfo* block_info = nullptr;
    if (i == HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_RPB)
      block_info = new GpuBlockInfo(RpbCounterBlockInfo);
    else
      block_info = new GpuBlockInfo(*base_table_ptr);
    block_table_[i] = block_info;

    // overwrite block info for any update from gfx9 to mi100
    InitSpmBlockDelay(block_info);
    switch (block_info->id) {
      case SqCounterBlockId:
        block_info->event_id_max = 303;
        break;
      case TcpCounterBlockId:
        block_info->event_id_max = 87;
        assert(agent_info->se_num * block_info->instance_count == cu_block_delay_table_size);
        break;
      case TccCounterBlockId:
        block_info->instance_count = 32;
        block_info->event_id_max = 295;
        break;
      case TcaCounterBlockId:
        block_info->instance_count = 32;
        block_info->event_id_max = 58;
        break;
      case GceaCounterBlockId:
        block_info->instance_count = 32;
        block_info->event_id_max = 83;
        break;
      case SdmaCounterBlockId:
        block_info->instance_count = gfx9_cntx_prim::SDMA_COUNTER_BLOCK_NUM_INSTANCES;
        break;
      case UmcCounterBlockId:
        block_info->counter_count = 6;
        break;
    }
  }
}

Pm4Factory* Pm4Factory::Mi100Create(const AgentInfo* agent_info) {
  auto p = new Mi100Factory(agent_info);
  if (p == NULL) throw aql_profile_exc_msg("Mi100Factory allocation failed");
  return p;
}

}  // namespace aql_profile
