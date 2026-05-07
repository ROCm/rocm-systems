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

#include "core/architectures/mi350_architecture.hpp"

namespace aql_profile {

Mi350Architecture::Mi350Architecture(const AgentInfo* agent_info)
    : Mi300Architecture(agent_info) {
  // Re-apply MI350-specific config overrides on top of MI300 base.
  InitializeConfig(agent_info);
}

void Mi350Architecture::InitializeConfig(const AgentInfo* agent_info) {
  Mi300Architecture::InitializeConfig(agent_info);

  config_.name = "MI350";
  config_.sqtt_header_version = 6;
  config_.spm_sample_delay_max = 0x33;
}

}  // namespace aql_profile
