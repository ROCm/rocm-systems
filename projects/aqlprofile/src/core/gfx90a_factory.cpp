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
#include "def/gfx90a_def.h"
#include "pm4/gfx9_cmd_builder.h"
#include "pm4/pmc_builder.h"
#include "pm4/sqtt_builder.h"

namespace aql_profile {

// Mi200 factory class
class Mi200Factory : public Gfx9Factory {
 public:
  explicit Mi200Factory(const AgentInfo* agent_info);

  virtual int GetAccumLowID() const override { return 1; };
  virtual int GetAccumHiID() const override { return 185; };

 protected:
  static const GpuBlockInfo* block_table_[HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER];
};

const GpuBlockInfo* Mi200Factory::block_table_[HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER] = {};

Mi200Factory::Mi200Factory(const AgentInfo* agent_info)
    : Gfx9Factory(block_table_, sizeof(block_table_), agent_info) {
  for (unsigned i = 0; i < HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER; ++i) {
    const GpuBlockInfo* base_table_ptr = Gfx9Factory::block_table_[i];
    if (base_table_ptr == NULL) continue;
    GpuBlockInfo* block_info = nullptr;
    if (i == HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_RPB)
      block_info = new GpuBlockInfo(RpbCounterBlockInfo);
    else
      block_info = new GpuBlockInfo(*base_table_ptr);
    block_table_[i] = block_info;
    // overwrite block info for any update from gfx9 to mi100
    switch (block_info->id) {
      case SqCounterBlockId:
        block_info->event_id_max = 303;
        break;
      case TcpCounterBlockId:
        block_info->event_id_max = 87;
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
        block_info->instance_count = 5;
        // Print(block_info);
        break;
      case UmcCounterBlockId:
        block_info->counter_count = 9;
        break;
    }
  }
}

Pm4Factory* Pm4Factory::Mi200Create(const AgentInfo* agent_info) {
  auto p = new Mi200Factory(agent_info);
  if (p == NULL) throw aql_profile_exc_msg("Mi200Factory allocation failed");
  return p;
}

}  // namespace aql_profile
