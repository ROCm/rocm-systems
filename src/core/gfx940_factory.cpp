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
#include "def/gfx940_def.h"
#include "pm4/gfx9_cmd_builder.h"
#include "pm4/pmc_builder.h"
#include "pm4/sqtt_builder.h"

namespace aql_profile {

class Mi300Factory : public Mi100Factory {
 public:
  explicit Mi300Factory(const AgentInfo* agent_info) : Mi100Factory(agent_info) {
    for (unsigned blockname_id = 0; blockname_id < HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER;
         ++blockname_id) {
      const GpuBlockInfo* base_table_ptr = Gfx9Factory::block_table_[blockname_id];
      if (base_table_ptr == NULL) continue;
      GpuBlockInfo* block_info = nullptr;
      if (blockname_id == HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_RPB)
        block_info = new GpuBlockInfo(RpbCounterBlockInfo);
      else if (blockname_id == HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATC)
        block_info = new GpuBlockInfo(AtcCounterBlockInfo);
      else
        block_info = new GpuBlockInfo(*base_table_ptr);
      block_table_[blockname_id] = block_info;
      // overwrite block info for any update from gfx9 to mi300
      switch (block_info->id) {
        case SqCounterBlockId:
          block_info->event_id_max = 373;
          break;
        case TcpCounterBlockId:
          block_info->event_id_max = 84;
          break;
        case TccCounterBlockId:
          block_info->instance_count = 16;
          block_info->event_id_max = 199;
          break;
        case TcaCounterBlockId:
          block_info->instance_count = 32;
          block_info->event_id_max = 34;
          break;
        case GceaCounterBlockId:
          block_info->instance_count = 32;
          block_info->event_id_max = 82;
          break;
        case SdmaCounterBlockId:
          block_info->instance_count = 4 * pm4_builder::MAX_AID;
          break;
        case UmcCounterBlockId:
          block_info->counter_count = 11;
          block_info->instance_count = 32 * pm4_builder::MAX_AID;
          break;
        case RpbCounterBlockId:
          block_info->instance_count = 4;
          break;
        case AtcCounterBlockId:
          block_info->instance_count = 4;
          break;
      }
    }
  }

  virtual int GetAccumLowID() const override { return 1; };
  virtual int GetAccumHiID() const override { return 184; };
};

Pm4Factory* Pm4Factory::Mi300Create(const AgentInfo* agent_info) {
  auto p = new Mi300Factory(agent_info);
  if (p == NULL) throw aql_profile_exc_msg("Mi300Factory allocation failed");
  return p;
}

class Mi350Factory : public Mi300Factory {
 public:
  // MI350 is a copy of Mi300
  explicit Mi350Factory(const AgentInfo* agent_info) : Mi300Factory(agent_info) {}

  virtual int GetAccumLowID() const override { return 1; };
  virtual int GetAccumHiID() const override { return 200; };
};

Pm4Factory* Pm4Factory::Mi350Create(const AgentInfo* agent_info) {
  auto p = new Mi350Factory(agent_info);
  if (p == NULL) throw aql_profile_exc_msg("Mi350Factory allocation failed");
  return p;
}

}  // namespace aql_profile
