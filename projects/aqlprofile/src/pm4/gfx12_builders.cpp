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

#include "pm4/gfx12_builders.hpp"
// gfx12_def.h must come before gfx12_cmd_builder.h to define packet macros
#include "def/gfx12_def.h"
#include "pm4/gfx12_cmd_builder.h"
#include "pm4/pmc_builder.h"
#include "pm4/spm_builder.h"
#include "pm4/sqtt_builder.h"

namespace pm4_builder_gfx12 {

pm4_builder::CmdBuilder* MakeCmdBuilder() {
  return new pm4_builder::Gfx12CmdBuilder(nullptr);
}

pm4_builder::PmcBuilder* MakePmcBuilder(const AgentInfo* agent_info, bool concurrent) {
  if (concurrent)
    return new pm4_builder::GpuPmcBuilder<pm4_builder::Gfx12CmdBuilder, gfx12_cntx_prim, true>(agent_info);
  return new pm4_builder::GpuPmcBuilder<pm4_builder::Gfx12CmdBuilder, gfx12_cntx_prim, false>(agent_info);
}

pm4_builder::SpmBuilder* MakeSpmBuilder(const AgentInfo* agent_info) {
  return new pm4_builder::GpuSpmBuilder<pm4_builder::Gfx12CmdBuilder, gfx12_cntx_prim>(agent_info);
}

pm4_builder::SqttBuilder* MakeSqttBuilder(const AgentInfo* agent_info) {
  return new pm4_builder::GpuSqttBuilder<pm4_builder::Gfx12CmdBuilder, gfx12_cntx_prim>(agent_info);
}

}  // namespace pm4_builder_gfx12
