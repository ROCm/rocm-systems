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
#include "core/factory_block_tables.h"
#include "aqlprofile-sdk/aql_profile_v2.h"
#include "def/gpu_block_info.h"

namespace aql_profile {

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
  config_.spm_sample_delay_max = 0x27;
}

void Mi300Architecture::InitializeBlockTable() {
  // Gfx9Factory::block_table_ is the shared static GFX9 block table.
  // Mi300Factory modifies block entries in-place (RPB/ATC instances, SDMA/UMC
  // per-AID instance counts, event_id_max overrides) so this pointer reflects
  // those overrides once the factory has been constructed for this agent.
  block_table_ = GetGfx9BlockTable();
  block_count_ = static_cast<uint32_t>(GetGfx9BlockTableSize());
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
