////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in the
//    documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef HSA_RUNTIME_CORE_INC_HOTSWAP_GFX_QUERY_HPP_
#define HSA_RUNTIME_CORE_INC_HOTSWAP_GFX_QUERY_HPP_

#include <cstdint>
#include <string>

#include "inc/hsa.h"

namespace rocr {
namespace hotswap {

// Architecture-agnostic description of an agent: its gfx target and ASIC
// revision. has_asic_revision is false when the runtime cannot query the
// revision.
struct AgentGfxRevision {
  std::string gfx_target;
  uint32_t asic_revision = 0;
  bool has_asic_revision = false;
};

std::string GetAgentIsaName(hsa_agent_t agent);
std::string ExtractGfxTarget(const std::string& isa_name);
bool IsGfx12_5Target(const std::string& gfx_target);
AgentGfxRevision GetAgentGfxRevision(hsa_agent_t agent);
void ResetAgentGfxRevisionCache();

// HotSwap rewrites B0 gfx1250 code objects only when loading for gfx1250 A0
// silicon. B0-on-B0 remains the normal ROCR load path with no rewrite.
bool IsHotswapSupportedGfxRevision(const AgentGfxRevision& gfx);

}  // namespace hotswap
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_HOTSWAP_GFX_QUERY_HPP_
