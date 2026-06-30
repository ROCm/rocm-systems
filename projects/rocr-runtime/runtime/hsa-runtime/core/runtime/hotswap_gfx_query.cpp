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

#include "core/inc/hotswap_gfx_query.hpp"

#include <algorithm>
#include <cctype>
#include <mutex>
#include <unordered_map>

#include "core/inc/hsa_internal.h"

namespace rocr {
namespace hotswap {

std::string GetAgentIsaName(hsa_agent_t agent) {
  std::string name;
  HSA::hsa_agent_iterate_isas(
      agent,
      [](hsa_isa_t isa, void* data) -> hsa_status_t {
        uint32_t len = 0;
        if (HSA::hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME_LENGTH, &len) != HSA_STATUS_SUCCESS) {
          return HSA_STATUS_ERROR;
        }

        auto& out = *static_cast<std::string*>(data);
        out.resize(len);
        if (HSA::hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME, out.data()) != HSA_STATUS_SUCCESS) {
          out.clear();
          return HSA_STATUS_ERROR;
        }

        if (!out.empty() && out.back() == '\0') {
          out.pop_back();
        }
        return HSA_STATUS_INFO_BREAK;
      },
      &name);
  return name;
}

std::string ExtractGfxTarget(const std::string& isa_name) {
  auto pos = isa_name.find("gfx");
  if (pos == std::string::npos) {
    return {};
  }

  auto end = std::find_if_not(isa_name.begin() + pos, isa_name.end(),
                              [](unsigned char c) { return std::isalnum(c); });
  return isa_name.substr(pos, end - isa_name.begin() - pos);
}

namespace {

std::mutex g_agent_gfx_revision_cache_mutex;
std::unordered_map<uint64_t, AgentGfxRevision> g_agent_gfx_revision_cache;

}  // namespace

AgentGfxRevision GetAgentGfxRevision(hsa_agent_t agent) {
  {
    std::scoped_lock lock(g_agent_gfx_revision_cache_mutex);
    const auto it = g_agent_gfx_revision_cache.find(agent.handle);
    if (it != g_agent_gfx_revision_cache.end()) {
      return it->second;
    }
  }

  AgentGfxRevision revision;
  revision.gfx_target = ExtractGfxTarget(GetAgentIsaName(agent));

  uint32_t asic_revision = 0;
  if (HSA::hsa_agent_get_info(agent,
                              static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_ASIC_REVISION),
                              &asic_revision) == HSA_STATUS_SUCCESS) {
    revision.asic_revision = asic_revision;
    revision.has_asic_revision = true;
  }

  {
    std::scoped_lock lock(g_agent_gfx_revision_cache_mutex);
    g_agent_gfx_revision_cache[agent.handle] = revision;
  }
  return revision;
}

void ResetAgentGfxRevisionCache() {
  std::scoped_lock lock(g_agent_gfx_revision_cache_mutex);
  g_agent_gfx_revision_cache.clear();
}

bool IsHotswapSupportedGfxRevision(const AgentGfxRevision& gfx) {
  return gfx.has_asic_revision && gfx.gfx_target == "gfx1250" && gfx.asic_revision == 0;
}

}  // namespace hotswap
}  // namespace rocr
