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
#include <libdrm/amdgpu_drm.h>

namespace aql_profile
{
namespace
{
// DRM fallback: query physical CU topology from the kernel driver when the
// caller uses V0/V1 registration (which lacks cu_bitmap). The preferred path
// is V2 registration with cu_bitmap supplied by the caller (see
// rocprofiler-sdk/agent.cpp::try_register_agent_v2), which bypasses this
// fallback entirely.
//
// The cu_bitmap is only consumed by the SQ-counter WGP iteration path in
// pm4/pmc_builder.h (bIsWGPcounter11), which only runs on GFX11+. On GFX9
// (MI100/200/300) and GFX10 the bitmap is unused, so the 64-entry DRM scan
// here would be pure overhead serializing on the kernel DRM mutex under
// parallel ctest -- skip it. GetGpuId() is the same chip-family resolver
// used elsewhere in aqlprofile (>= GFX10_GPU_ID for sa_number selection,
// >= GFX12_GPU_ID for trace status2, etc.); the gpu_id_t enum is ordered
// so >= GFX11_GPU_ID covers GFX11 / GFX115X / GFX12 / MI450 and excludes
// the GFX9 family (which sits at enum values 1..5) and GFX10 (=6).
//
// Heuristic match: we iterate render nodes /dev/dri/renderD128..renderD191
// and pick the first one whose (cu_active_number, num_shader_engines,
// num_shader_arrays_per_engine) matches this AgentInfo. That is unambiguous
// on single-GPU systems and on multi-GPU systems with distinct SKUs. It is
// AMBIGUOUS on the pathological multi-GPU case of two identical SKUs whose
// only difference is the per-die harvest mask -- there the first matching
// render node wins, and one agent may receive the other GPU's bitmap. The
// worst-case effect is incorrect harvest-aware WGP iteration on one agent;
// there is no memory-safety impact. Prefer V2 registration end-to-end to
// avoid this entirely.
void
populate_cu_bitmap_from_drm(AgentInfo& agent_info)
{
    if(Pm4Factory::GetGpuId(agent_info.name) < GFX11_GPU_ID) return;

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
            // Size from the (smaller, fixed) kernel uAPI side so this cannot
            // over-read dev_info if the internal cu_bitmap layout ever grows.
            static_assert(sizeof(agent_info.cu_bitmap.bits) >= sizeof(dev_info.cu_bitmap),
                          "drm_amdgpu_info_device.cu_bitmap larger than "
                          "aqlprofile_cu_bitmap_t::bits; bump "
                          "AQLPROFILE_DRM_CU_BITMAP_NUM_SE / "
                          "AQLPROFILE_DRM_CU_BITMAP_NUM_SA_PER_SE in aql_profile_v2.h to "
                          "match the kernel uAPI and bump the V2 ABI version");
            memcpy(agent_info.cu_bitmap.bits, dev_info.cu_bitmap, sizeof(dev_info.cu_bitmap));
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

    // TODO: Temporary patch for gfx1250's asymmetric CU design, will remove
    //       after CU mask support is added to agent_info
    // TODO: gfx1250 defines 1WGP = 1CU, different from other RDNA products.
    //       Patch it to be WGP = 2CU to reuse profiler logic
    if(!strncmp(int_agent_info.name, "gfx1250", 7))
    {
        int_agent_info.cu_num      = agent_info->se_num * agent_info->shader_arrays_per_se * 9 * 2;
        int_agent_info.xcc_per_aid = 4;
    }
    else if(!strncmp(int_agent_info.name, "gfx94", 5) || !strncmp(int_agent_info.name, "gfx95", 5))
    {
        int_agent_info.xcc_per_aid = 2;
    }
    else
    {
        int_agent_info.xcc_per_aid = 1;
    }

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
    int_agent_info.cu_bitmap            = agent_info->cu_bitmap;
    populate_agent_name(int_agent_info, agent_info->agent_gfxip);

    // TODO: Temporary patch for gfx1250's asymmetric CU design, will remove
    //       after CU mask support is added to agent_info
    // TODO: gfx1250 defines 1WGP = 1CU, different from other RDNA products.
    //       Patch it to be WGP = 2CU to reuse profiler logic
    if(!strncmp(int_agent_info.name, "gfx1250", 7))
    {
        int_agent_info.cu_num      = agent_info->se_num * agent_info->shader_arrays_per_se * 9 * 2;
        int_agent_info.xcc_per_aid = 4;
    }
    else if(!strncmp(int_agent_info.name, "gfx94", 5) || !strncmp(int_agent_info.name, "gfx95", 5))
    {
        int_agent_info.xcc_per_aid = 2;
    }
    else
    {
        int_agent_info.xcc_per_aid = 1;
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
