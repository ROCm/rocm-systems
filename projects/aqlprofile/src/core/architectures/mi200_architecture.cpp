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

#include "core/architectures/mi200_architecture.hpp"
#include "core/gfx9_factory.h"

namespace aql_profile {

Mi200Architecture::Mi200Architecture(const AgentInfo* agent_info) {
  InitializeConfig(agent_info);
  InitializeRegisterSchema();
  InitializeBlockTable();
}

void Mi200Architecture::InitializeConfig(const AgentInfo* agent_info) {
  Gfx9Architecture::InitializeConfig(agent_info);

  config_.name = "MI200";
  config_.has_spm_core1 = true;
  config_.spm_sample_delay_max = 0x3e;
}

void Mi200Architecture::InitializeBlockTable() {
  // Gfx9Factory::block_table_ is the shared static GFX9 block table.
  // Mi200Factory modifies block entries in-place (SDMA instance_count=5,
  // UMC counter_count=9, etc.) so this pointer reflects those overrides
  // once the factory has been constructed for this agent.
  block_table_ = Gfx9Factory::block_table_;
  block_count_ = AQLPROFILE_BLOCKS_NUMBER;
}

}  // namespace aql_profile
