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

#include <mutex>
#include <shared_mutex>

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <cstring>
#include <drm/amdgpu_drm.h>

namespace aql_profile
{
namespace
{
// DRM fallback: query physical CU topology from the kernel driver when the
// caller uses V0/V1 registration (which lacks cu_bitmap).  The preferred path
// is V2 registration with cu_bitmap supplied by the caller.
void
populate_cu_bitmap_from_drm(AgentInfo& agent_info)
{
    for(int minor = 128; minor < 192; ++minor)
    {
        char path[64];
        snprintf(path, sizeof(path), "/dev/dri/renderD%d", minor);
        int fd = open(path, O_RDWR | O_CLOEXEC);
        if(fd < 0) continue;

        struct drm_amdgpu_info_device dev_info = {};
        struct drm_amdgpu_info        request  = {};
        request.return_pointer                 = reinterpret_cast<uintptr_t>(&dev_info);
        request.return_size                    = sizeof(dev_info);
        request.query                          = AMDGPU_INFO_DEV_INFO;

        int ret = ioctl(fd, DRM_IOCTL_AMDGPU_INFO, &request);
        close(fd);
        if(ret != 0) continue;

        // Match DRM device to agent by active CU count and SE/SA topology
        if(dev_info.cu_active_number == agent_info.cu_num &&
           dev_info.num_shader_engines == agent_info.se_num &&
           dev_info.num_shader_arrays_per_engine == agent_info.shader_arrays_per_se)
        {
            agent_info.cu_per_simd_array = dev_info.num_cu_per_sh;
            memcpy(agent_info.cu_bitmap, dev_info.cu_bitmap, sizeof(agent_info.cu_bitmap));
            return;
        }
    }
}

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
    populate_cu_bitmap_from_drm(int_agent_info);
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
    int_agent_info.cu_per_simd_array    = agent_info->cu_per_simd_array;
    memcpy(int_agent_info.cu_bitmap, agent_info->cu_bitmap, sizeof(int_agent_info.cu_bitmap));
    populate_agent_name(int_agent_info, agent_info->agent_gfxip);
    get_cache().add(agent_id.handle, int_agent_info);
    return agent_id;
}

const AgentInfo*
GetAgentInfo(aqlprofile_agent_handle_t agent_id)
{
    return get_cache().get(agent_id.handle);
}

}  // namespace aql_profile
