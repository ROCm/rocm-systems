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

#include "pm4_factory.h"

#include <mutex>
#include <shared_mutex>

namespace aql_profile {
namespace {
struct locked_agent_cache {
  std::shared_mutex mutex;
  std::unordered_map<uint64_t, AgentInfo> cache;

  void add(uint64_t& agent_id, const AgentInfo& agent_info) {
    auto lock = std::unique_lock{mutex};
    agent_id = cache.size();
    cache[agent_id] = agent_info;
  }

  const AgentInfo* get(uint64_t agent_id) {
    auto lock = std::shared_lock{mutex};
    auto it = cache.find(agent_id);
    if (it == cache.end()) return nullptr;
    return &it->second;
  }
};

locked_agent_cache& get_cache() {
  static auto* cache = new locked_agent_cache{};
  return *cache;
}
}  // namespace

aqlprofile_agent_handle_t RegisterAgent(const aqlprofile_agent_info_v1_t* agent_info) {
  aqlprofile_agent_handle_t agent_id;
  AgentInfo int_agent_info;
  int_agent_info.cu_num = agent_info->cu_num;
  int_agent_info.se_num = agent_info->se_num;
  int_agent_info.xcc_num = agent_info->xcc_num;
  int_agent_info.shader_arrays_per_se = agent_info->shader_arrays_per_se;
  int_agent_info.domain = agent_info->domain;
  int_agent_info.bdf_id = agent_info->location_id;

  auto len = strlen(agent_info->agent_gfxip);
  memset(int_agent_info.gfxip, 0, sizeof(int_agent_info.gfxip));
  memcpy(int_agent_info.gfxip, agent_info->agent_gfxip,
         (len >= sizeof(int_agent_info.gfxip) ? sizeof(int_agent_info.gfxip) - 1 : len));
  get_cache().add(agent_id.handle, int_agent_info);
  return agent_id;
}

const AgentInfo* GetAgentInfo(aqlprofile_agent_handle_t agent_id) {
  return get_cache().get(agent_id.handle);
}

}  // namespace aql_profile