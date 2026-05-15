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

// Shared GFX12 factory base class.  Accepts already-constructed architecture
// and primitives provider objects so that each per-variant factory file only
// needs to supply the two concrete types and nothing else.
//
// Do NOT include gfx12_def.h or any register header from this file; the
// including translation unit must arrange for the correct GFX12_VARIANT to be
// active before any transitive include of gfx12_def.h occurs.

#ifndef SRC_CORE_GFX12_FACTORY_BASE_HPP_
#define SRC_CORE_GFX12_FACTORY_BASE_HPP_

#include "core/pm4_factory.h"
#include "pm4/gfx12/gfx12_cmd_builder.h"
#include "pm4/gfx12/gfx12_pmc_builder.hpp"
#include "pm4/gfx12/gfx12_spm_builder.hpp"
#include "pm4/gfx12/gfx12_sqtt_builder.hpp"
#include "pm4/primitives_provider.hpp"
#include "util/reg_offsets.h"

namespace aql_profile {

class Gfx12FactoryBase : public Pm4Factory {
 public:
  Gfx12FactoryBase(HardwareArchitecture* arch,
                   pm4_builder::PrimitivesProvider* prims,
                   const AgentInfo* agent_info)
      : Pm4Factory(arch), prims_(prims) {
    agent_info_  = agent_info;
    cmd_builder_ = new pm4_builder::Gfx12CmdBuilder(acquire_ip_offset_table(agent_info));
    const auto& config = GetArchitecture()->GetConfig();
    pmc_builder_  = new pm4_builder::Gfx12PmcBuilder(config, cmd_builder_, prims_, IsConcurrent());
    spm_builder_  = new pm4_builder::Gfx12SpmBuilder(cmd_builder_, prims_);
    sqtt_builder_ = new pm4_builder::Gfx12SqttBuilder(config, cmd_builder_, prims_,
                                                       agent_info->timestamp_freq);
  }

  bool IsGFX12() const override { return true; }

  ~Gfx12FactoryBase() override { delete prims_; }

 private:
  pm4_builder::PrimitivesProvider* prims_;
};

}  // namespace aql_profile

#endif  // SRC_CORE_GFX12_FACTORY_BASE_HPP_
