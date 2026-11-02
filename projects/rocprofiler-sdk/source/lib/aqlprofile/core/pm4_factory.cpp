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

#include "lib/aqlprofile/core/pm4_factory.h"

#include <cstring>
#include <mutex>
#include <shared_mutex>

namespace aql_profile
{
namespace
{
struct locked_agent_cache
{
    std::shared_mutex                       mutex;
    std::unordered_map<uint64_t, AgentInfo> cache;

    void add(uint64_t& agent_id, const AgentInfo& agent_info)
    {
        auto lock       = std::unique_lock{mutex};
        agent_id        = cache.size();
        cache[agent_id] = agent_info;
    }

    const AgentInfo* get(uint64_t agent_id)
    {
        auto lock = std::shared_lock{mutex};
        auto it   = cache.find(agent_id);
        if(it == cache.end()) return nullptr;
        return &it->second;
    }
};

locked_agent_cache&
get_cache()
{
    static auto* cache = new locked_agent_cache{};
    return *cache;
}
}  // namespace

// Helper: populate common AgentInfo fields from agent_gfxip string
static void
populate_agent_name(AgentInfo& info, const char* agent_gfxip)
{
    auto len = strlen(agent_gfxip);
    memset(info.name, 0, sizeof(info.name));
    memcpy(info.name, agent_gfxip, (len >= sizeof(info.name) ? sizeof(info.name) - 1 : len));
    memset(info.gfxip, 0, sizeof(info.gfxip));
    memcpy(info.gfxip, agent_gfxip, (len >= sizeof(info.gfxip) ? sizeof(info.gfxip) - 1 : len));
}

// Helper: populate arch-specific AgentInfo fields (xcc_per_aid, gfx1250 CU
// remap) by name. Must run after populate_agent_name() since it keys off
// info.name.
static void
populate_arch_specific_info(AgentInfo& info)
{
    // gfx1250 has an asymmetric CU design: size for the widest SA (9 WGPs);
    // the narrower SA is described by the agent cu_bitmap (or, without one,
    // by the PMC builder's asymmetric_cu_patch).
    // TODO: gfx1250 defines 1WGP = 1CU, different from other RDNA products.
    //       Patch it to be WGP = 2CU to reuse profiler logic
    if(!strncmp(info.name, "gfx1250", 7))
    {
        info.cu_num      = info.se_num * info.shader_arrays_per_se * 9 * 2;
        info.xcc_per_aid = 4;
    }
    else if(!strncmp(info.name, "gfx94", 5) || !strncmp(info.name, "gfx95", 5))
    {
        info.xcc_per_aid = 2;
    }
    else
    {
        info.xcc_per_aid = 1;
    }
}

aqlprofile_agent_handle_t
RegisterAgent(const aqlprofile_agent_info_v1_t* agent_info)
{
    aqlprofile_agent_handle_t agent_id;
    AgentInfo                 int_agent_info;
    int_agent_info.cu_num               = agent_info->cu_num;
    int_agent_info.se_num               = agent_info->se_num;
    int_agent_info.xcc_num              = agent_info->xcc_num;
    int_agent_info.shader_arrays_per_se = agent_info->shader_arrays_per_se;
    int_agent_info.domain               = agent_info->domain;
    int_agent_info.bdf_id               = agent_info->location_id;
    populate_agent_name(int_agent_info, agent_info->agent_gfxip);
    populate_arch_specific_info(int_agent_info);

    get_cache().add(agent_id.handle, int_agent_info);
    return agent_id;
}

aqlprofile_agent_handle_t
RegisterAgent(const aqlprofile_agent_info_v2_t* agent_info)
{
    aqlprofile_agent_handle_t agent_id;
    AgentInfo                 int_agent_info;
    int_agent_info.cu_num               = agent_info->cu_num;
    int_agent_info.se_num               = agent_info->se_num;
    int_agent_info.xcc_num              = agent_info->xcc_num;
    int_agent_info.shader_arrays_per_se = agent_info->shader_arrays_per_se;
    int_agent_info.domain               = agent_info->domain;
    int_agent_info.bdf_id               = agent_info->location_id;
    populate_agent_name(int_agent_info, agent_info->agent_gfxip);
    populate_arch_specific_info(int_agent_info);

    // Only accept a bitmap that describes exactly this agent's topology; the
    // PMC builder indexes it with the agent's XCC/SE/SA counts.
    const uint32_t xcc_num        = agent_info->xcc_num;
    const bool     bitmap_matches = agent_info->cu_bitmap != nullptr && xcc_num > 0 &&
                                agent_info->cu_bitmap_num_xcc == xcc_num &&
                                agent_info->cu_bitmap_num_se * xcc_num == agent_info->se_num &&
                                agent_info->cu_bitmap_num_sa == agent_info->shader_arrays_per_se &&
                                agent_info->cu_bitmap_num_se > 0 &&
                                agent_info->cu_bitmap_num_sa > 0;
    if(bitmap_matches)
    {
        const size_t count = size_t{agent_info->cu_bitmap_num_xcc} * agent_info->cu_bitmap_num_se *
                             agent_info->cu_bitmap_num_sa;
        int_agent_info.cu_bitmap.assign(agent_info->cu_bitmap, agent_info->cu_bitmap + count);
    }

    get_cache().add(agent_id.handle, int_agent_info);
    return agent_id;
}

const AgentInfo*
GetAgentInfo(aqlprofile_agent_handle_t agent_id)
{
    return get_cache().get(agent_id.handle);
}

}  // namespace aql_profile
