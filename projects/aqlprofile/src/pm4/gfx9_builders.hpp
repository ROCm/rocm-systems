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

#ifndef SRC_PM4_GFX9_BUILDERS_HPP_
#define SRC_PM4_GFX9_BUILDERS_HPP_

#include "pm4/cmd_builder.h"
#include "pm4/pmc_builder.h"
#include "pm4/spm_builder.h"
#include "pm4/sqtt_builder.h"

struct AgentInfo;

namespace pm4_builder_gfx9 {

pm4_builder::CmdBuilder* MakeCmdBuilder();
pm4_builder::PmcBuilder* MakePmcBuilder(const AgentInfo* agent_info, bool concurrent);
pm4_builder::SpmBuilder* MakeSpmBuilder(const AgentInfo* agent_info);
pm4_builder::SqttBuilder* MakeSqttBuilder(const AgentInfo* agent_info);

}  // namespace pm4_builder_gfx9

#endif  // SRC_PM4_GFX9_BUILDERS_HPP_
