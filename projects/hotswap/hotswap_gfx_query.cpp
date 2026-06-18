//===- hotswap_gfx_query.cpp - Agent gfx-target / ASIC-revision query -----===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap_gfx_query.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <unordered_map>

namespace rocr::hotswap {

std::string get_agent_isa_name(hsa_agent_t agent) {
  std::string name;

  hsa_agent_iterate_isas(agent, [](hsa_isa_t isa, void *data) -> hsa_status_t {
    uint32_t len = 0;
    if (hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME_LENGTH, &len) !=
        HSA_STATUS_SUCCESS)
      return HSA_STATUS_ERROR;

    auto &out = *static_cast<std::string *>(data);
    out.resize(len);

    if (hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME, out.data()) !=
        HSA_STATUS_SUCCESS) {
      out.clear();
      return HSA_STATUS_ERROR;
    }

    // HSA returns null-terminated length; trim it.
    if (!out.empty() && out.back() == '\0')
      out.pop_back();

    return HSA_STATUS_INFO_BREAK; // only need the first ISA
  }, &name);

  return name;
}

std::string extract_gfx_target(const std::string &isa_name) {
  auto pos = isa_name.find("gfx");
  if (pos == std::string::npos)
    return {};
  auto end = std::find_if_not(isa_name.begin() + pos, isa_name.end(),
                              [](unsigned char c) { return std::isalnum(c); });
  return isa_name.substr(pos, end - isa_name.begin() - pos);
}

namespace {
std::mutex g_cache_mutex;
std::unordered_map<uint64_t, AgentGfxRevision> g_cache;
} // namespace

AgentGfxRevision query_agent_gfx_revision(hsa_agent_t agent) {
  {
    std::scoped_lock lock(g_cache_mutex);
    const auto it = g_cache.find(agent.handle);
    if (it != g_cache.end()) {
      return it->second;
    }
  }

  AgentGfxRevision info;
  info.gfx_target = extract_gfx_target(get_agent_isa_name(agent));

  uint32_t asic_revision = 0;
  if (hsa_agent_get_info(
          agent,
          static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_ASIC_REVISION),
          &asic_revision) == HSA_STATUS_SUCCESS) {
    info.asic_revision = asic_revision;
    info.revision_valid = true;
  }

  fprintf(stderr, "hotswap: agent gfx=%s asic_revision=%u (valid=%s)\n",
          info.gfx_target.empty() ? "?" : info.gfx_target.c_str(),
          info.asic_revision, info.revision_valid ? "yes" : "no");

  {
    std::scoped_lock lock(g_cache_mutex);
    g_cache[agent.handle] = info;
  }
  return info;
}

void reset_gfx_revision_cache() {
  std::scoped_lock lock(g_cache_mutex);
  g_cache.clear();
}

bool gate_allows_hotswap(const AgentGfxRevision &gfx) {
  return gfx.revision_valid && gfx.gfx_target == "gfx1250" &&
         gfx.asic_revision == 0; // A0
}

} // namespace rocr::hotswap
