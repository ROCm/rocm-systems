// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/rocprofiler-sdk/tests/details/agent.hpp"
#include "lib/common/filesystem.hpp"
#include "lib/common/utility.hpp"

#include <fstream>

#include <grp.h>
#include <hsa/hsa.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>

#include <stdio.h>
#include <sstream>
#include <string>
#include <vector>

#define RET_IF_HSA_INIT_ERR(err)                                                                   \
    {                                                                                              \
        if((err) != HSA_STATUS_SUCCESS)                                                            \
        {                                                                                          \
            CheckInitError();                                                                      \
            RET_IF_HSA_ERR(err);                                                                   \
        }                                                                                          \
    }

#define RET_IF_HSA_ERR(err)                                                                        \
    {                                                                                              \
        if((err) != HSA_STATUS_SUCCESS)                                                            \
        {                                                                                          \
            char  err_val[12];                                                                     \
            char* err_str = nullptr;                                                               \
            if(hsa_status_string(err, (const char**) &err_str) != HSA_STATUS_SUCCESS)              \
            {                                                                                      \
                sprintf(&(err_val[0]), "%#x", (uint32_t) err);                                     \
                err_str = &(err_val[0]);                                                           \
            }                                                                                      \
            printf("hsa api call failure at: %s:%d\n", __FILE__, __LINE__);                        \
            printf("Call returned %s\n", err_str);                                                 \
            return (err);                                                                          \
        }                                                                                          \
    }

namespace rocprofiler
{
namespace test
{
namespace
{
// Acquire system information
hsa_status_t
AcquireSystemInfo(system_info_t* sys_info)
{
    hsa_status_t err;

    // Get Major and Minor version of runtime
    err = hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MAJOR, &sys_info->major);
    RET_IF_HSA_ERR(err);
    err = hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MINOR, &sys_info->minor);
    RET_IF_HSA_ERR(err);

    // Get timestamp frequency
    err = hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &sys_info->timestamp_frequency);
    RET_IF_HSA_ERR(err);

    // Get maximum duration of a signal wait operation
    err = hsa_system_get_info(HSA_SYSTEM_INFO_SIGNAL_MAX_WAIT, &sys_info->max_wait);
    RET_IF_HSA_ERR(err);

    // Get Endianness of the system
    err = hsa_system_get_info(HSA_SYSTEM_INFO_ENDIANNESS, &sys_info->endianness);
    RET_IF_HSA_ERR(err);

    // Get machine model info
    err = hsa_system_get_info(HSA_SYSTEM_INFO_MACHINE_MODEL, &sys_info->machine_model);
    RET_IF_HSA_ERR(err);
    return err;
}

hsa_status_t
AcquireAgentInfoEntry(hsa_agent_t agent, agent_info_t* agent_i)
{
    // store the hsa_agent_t value
    agent_i->hsa_agent = agent;

    hsa_status_t err;
    // Get agent name and vendor
    err = hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, agent_i->name);
    RET_IF_HSA_ERR(err);
    err = hsa_agent_get_info(agent, HSA_AGENT_INFO_VENDOR_NAME, &agent_i->vendor_name);
    RET_IF_HSA_ERR(err);

    // Get device marketing name
    err = hsa_agent_get_info(
        agent, (hsa_agent_info_t) HSA_AMD_AGENT_INFO_PRODUCT_NAME, &agent_i->device_mkt_name);
    RET_IF_HSA_ERR(err);

    // Get agent feature
    err = hsa_agent_get_info(agent, HSA_AGENT_INFO_FEATURE, &agent_i->agent_feature);
    RET_IF_HSA_ERR(err);

    // Get profile supported by the agent
    err = hsa_agent_get_info(agent, HSA_AGENT_INFO_PROFILE, &agent_i->device_counting_service);
    RET_IF_HSA_ERR(err);

    // Get floating-point rounding mode
    err = hsa_agent_get_info(
        agent, HSA_AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE, &agent_i->float_rounding_mode);
    RET_IF_HSA_ERR(err);

    // Get max number of queue
    err = hsa_agent_get_info(agent, HSA_AGENT_INFO_QUEUES_MAX, &agent_i->max_queue);
    RET_IF_HSA_ERR(err);

    // Get queue min size
    err = hsa_agent_get_info(agent, HSA_AGENT_INFO_QUEUE_MIN_SIZE, &agent_i->queue_min_size);
    RET_IF_HSA_ERR(err);

    // Get queue max size
    err = hsa_agent_get_info(agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &agent_i->queue_max_size);
    RET_IF_HSA_ERR(err);

    // Get queue type
    err = hsa_agent_get_info(agent, HSA_AGENT_INFO_QUEUE_TYPE, &agent_i->queue_type);
    RET_IF_HSA_ERR(err);

    // Get agent node
    err = hsa_agent_get_info(agent, HSA_AGENT_INFO_NODE, &agent_i->node);
    RET_IF_HSA_ERR(err);

    // Get device type
    err = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &agent_i->device_type);
    RET_IF_HSA_ERR(err);

    if(HSA_DEVICE_TYPE_GPU == agent_i->device_type)
    {
        err = hsa_agent_get_info(agent, HSA_AGENT_INFO_ISA, &agent_i->agent_isa);
        RET_IF_HSA_ERR(err);
    }

    // Get cache size
    err = hsa_agent_get_info(agent, HSA_AGENT_INFO_CACHE_SIZE, agent_i->cache_size);
    RET_IF_HSA_ERR(err);

    // Get chip id
    err =
        hsa_agent_get_info(agent, (hsa_agent_info_t) HSA_AMD_AGENT_INFO_CHIP_ID, &agent_i->chip_id);
    RET_IF_HSA_ERR(err);

    // Get cacheline size
    err = hsa_agent_get_info(
        agent, (hsa_agent_info_t) HSA_AMD_AGENT_INFO_CACHELINE_SIZE, &agent_i->cacheline_size);
    RET_IF_HSA_ERR(err);

    // Get Max clock frequency
    err = hsa_agent_get_info(
        agent, (hsa_agent_info_t) HSA_AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY, &agent_i->max_clock_freq);
    RET_IF_HSA_ERR(err);

    // Internal Driver node ID
    err = hsa_agent_get_info(
        agent, (hsa_agent_info_t) HSA_AMD_AGENT_INFO_DRIVER_NODE_ID, &agent_i->internal_node_id);
    RET_IF_HSA_ERR(err);

    // Max number of watch points on mem. addr. ranges to generate exeception
    // events
    err = hsa_agent_get_info(agent,
                             (hsa_agent_info_t) HSA_AMD_AGENT_INFO_MAX_ADDRESS_WATCH_POINTS,
                             &agent_i->max_addr_watch_pts);
    RET_IF_HSA_ERR(err);

    // Get Agent BDFID
    err = hsa_agent_get_info(agent, (hsa_agent_info_t) HSA_AMD_AGENT_INFO_BDFID, &agent_i->bdf_id);
    RET_IF_HSA_ERR(err);

    // Get Max Memory Clock
    // Not supported by hsa_agent_get_info
    //  err = hsa_agent_get_info(agent,d
    //              (hsa_agent_info_t)HSA_AMD_AGENT_INFO_MEMORY_MAX_FREQUENCY,
    //                                                      &agent_i->mem_max_freq);
    //  RET_IF_HSA_ERR(err);

    // Get Num SIMDs per CU
    err = hsa_agent_get_info(
        agent, (hsa_agent_info_t) HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU, &agent_i->simds_per_cu);
    RET_IF_HSA_ERR(err);

    // Get Num Shader Engines
    err = hsa_agent_get_info(
        agent, (hsa_agent_info_t) HSA_AMD_AGENT_INFO_NUM_SHADER_ENGINES, &agent_i->shader_engs);
    RET_IF_HSA_ERR(err);

    // Get Num Shader Arrays per Shader engine
    err = hsa_agent_get_info(agent,
                             (hsa_agent_info_t) HSA_AMD_AGENT_INFO_NUM_SHADER_ARRAYS_PER_SE,
                             &agent_i->shader_arrs_per_sh_eng);
    RET_IF_HSA_ERR(err);

    // Get number of Compute Unit
    err = hsa_agent_get_info(
        agent, (hsa_agent_info_t) HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT, &agent_i->compute_unit);
    RET_IF_HSA_ERR(err);

    // family id
    err = hsa_agent_get_info(
        agent, (hsa_agent_info_t) HSA_AMD_AGENT_INFO_ASIC_FAMILY_ID, &agent_i->family_id);
    RET_IF_HSA_ERR(err);

    // ucode version
    err = hsa_agent_get_info(
        agent, (hsa_agent_info_t) HSA_AMD_AGENT_INFO_UCODE_VERSION, &agent_i->ucode_version);
    RET_IF_HSA_ERR(err);

    // sdma ucode version
    err = hsa_agent_get_info(agent,
                             (hsa_agent_info_t) HSA_AMD_AGENT_INFO_SDMA_UCODE_VERSION,
                             &agent_i->sdma_ucode_version);
    RET_IF_HSA_ERR(err);

    // Check if the agent is kernel agent
    if((agent_i->agent_feature & HSA_AGENT_FEATURE_KERNEL_DISPATCH) != 0)
    {
        // Get flaf of fast_f16 operation
        err = hsa_agent_get_info(agent, HSA_AGENT_INFO_FAST_F16_OPERATION, &agent_i->fast_f16);
        RET_IF_HSA_ERR(err);

        // Get wavefront size
        err = hsa_agent_get_info(agent, HSA_AGENT_INFO_WAVEFRONT_SIZE, &agent_i->wavefront_size);
        RET_IF_HSA_ERR(err);

        // Get max total number of work-items in a workgroup
        err = hsa_agent_get_info(
            agent, HSA_AGENT_INFO_WORKGROUP_MAX_SIZE, &agent_i->workgroup_max_size);
        RET_IF_HSA_ERR(err);

        // Get max number of work-items of each dimension of a work-group
        err = hsa_agent_get_info(
            agent, HSA_AGENT_INFO_WORKGROUP_MAX_DIM, &agent_i->workgroup_max_dim);
        RET_IF_HSA_ERR(err);

        // Get max number of a grid per dimension
        err = hsa_agent_get_info(agent, HSA_AGENT_INFO_GRID_MAX_DIM, &agent_i->grid_max_dim);
        RET_IF_HSA_ERR(err);

        // Get max total number of work-items in a grid
        err = hsa_agent_get_info(agent, HSA_AGENT_INFO_GRID_MAX_SIZE, &agent_i->grid_max_size);
        RET_IF_HSA_ERR(err);

        // Get max number of fbarriers per work group
        err = hsa_agent_get_info(
            agent, HSA_AGENT_INFO_FBARRIER_MAX_SIZE, &agent_i->fbarrier_max_size);
        RET_IF_HSA_ERR(err);

        err = hsa_agent_get_info(agent,
                                 (hsa_agent_info_t) HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU,
                                 &agent_i->max_waves_per_cu);
        RET_IF_HSA_ERR(err);
    }
    return err;
}

hsa_status_t
AcquirePoolInfo(hsa_amd_memory_pool_t pool, pool_info_t* pool_i)
{
    hsa_status_t err;

    err = hsa_amd_memory_pool_get_info(
        pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &pool_i->global_flag);
    RET_IF_HSA_ERR(err);

    err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &pool_i->segment);
    RET_IF_HSA_ERR(err);

    // Get the size of the POOL
    err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SIZE, &pool_i->pool_size);
    RET_IF_HSA_ERR(err);

    err = hsa_amd_memory_pool_get_info(
        pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &pool_i->alloc_allowed);
    RET_IF_HSA_ERR(err);

    err = hsa_amd_memory_pool_get_info(
        pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE, &pool_i->alloc_granule);
    RET_IF_HSA_ERR(err);

    err = hsa_amd_memory_pool_get_info(
        pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT, &pool_i->pool_alloc_alignment);
    RET_IF_HSA_ERR(err);

    err = hsa_amd_memory_pool_get_info(
        pool, HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL, &pool_i->pl_access);
    RET_IF_HSA_ERR(err);

    return HSA_STATUS_SUCCESS;
}

hsa_status_t
get_pool_info(hsa_amd_memory_pool_t pool, void* data)
{
    auto* info   = static_cast<rocm_info*>(data);
    auto& pool_i = info->pools.emplace_back();
    auto  err    = AcquirePoolInfo(pool, &pool_i);
    RET_IF_HSA_ERR(err);

    return err;
}

hsa_status_t
AcquireISAInfo(hsa_isa_t isa, isa_info_t* isa_i)
{
    hsa_status_t err;
    uint32_t     name_len;
    err = hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME_LENGTH, &name_len);
    RET_IF_HSA_ERR(err);

    isa_i->name_str = new char[name_len];
    if(isa_i->name_str == nullptr)
    {
        return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    }

    err = hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME, isa_i->name_str);
    RET_IF_HSA_ERR(err);

    err = hsa_isa_get_info_alt(isa, HSA_ISA_INFO_MACHINE_MODELS, isa_i->mach_models);
    RET_IF_HSA_ERR(err);

    err = hsa_isa_get_info_alt(isa, HSA_ISA_INFO_PROFILES, isa_i->profiles);
    RET_IF_HSA_ERR(err);

    err = hsa_isa_get_info_alt(
        isa, HSA_ISA_INFO_DEFAULT_FLOAT_ROUNDING_MODES, isa_i->def_rounding_modes);
    RET_IF_HSA_ERR(err);

    err = hsa_isa_get_info_alt(
        isa, HSA_ISA_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES, isa_i->base_rounding_modes);
    RET_IF_HSA_ERR(err);

    err = hsa_isa_get_info_alt(isa, HSA_ISA_INFO_FAST_F16_OPERATION, &isa_i->fast_f16);
    RET_IF_HSA_ERR(err);

    err = hsa_isa_get_info_alt(isa, HSA_ISA_INFO_WORKGROUP_MAX_DIM, &isa_i->workgroup_max_dim);
    RET_IF_HSA_ERR(err);

    err = hsa_isa_get_info_alt(isa, HSA_ISA_INFO_WORKGROUP_MAX_SIZE, &isa_i->workgroup_max_size);
    RET_IF_HSA_ERR(err);

    err = hsa_isa_get_info_alt(isa, HSA_ISA_INFO_GRID_MAX_DIM, &isa_i->grid_max_dim);
    RET_IF_HSA_ERR(err);

    err = hsa_isa_get_info_alt(isa, HSA_ISA_INFO_GRID_MAX_SIZE, &isa_i->grid_max_size);
    RET_IF_HSA_ERR(err);

    err = hsa_isa_get_info_alt(isa, HSA_ISA_INFO_FBARRIER_MAX_SIZE, &isa_i->fbarrier_max_size);
    RET_IF_HSA_ERR(err);

    return err;
}

hsa_status_t
get_isa_info(hsa_isa_t isa, void* data)
{
    auto*       info  = static_cast<rocm_info*>(data);
    isa_info_t& isa_i = info->isas.emplace_back();

    isa_i.name_str = nullptr;
    RET_IF_HSA_ERR(AcquireISAInfo(isa, &isa_i));

    return HSA_STATUS_SUCCESS;
}

hsa_status_t
AcquireAgentInfo(hsa_agent_t agent, void* data)
{
    auto*         info    = static_cast<rocm_info*>(data);
    agent_info_t& agent_i = info->agents.emplace_back();

    RET_IF_HSA_ERR(AcquireAgentInfoEntry(agent, &agent_i));
    RET_IF_HSA_ERR(hsa_amd_agent_iterate_memory_pools(agent, get_pool_info, data));

    {
        auto err = hsa_agent_iterate_isas(agent, get_isa_info, data);
        if(err != HSA_STATUS_ERROR_INVALID_AGENT) RET_IF_HSA_ERR(err);
    }

    return HSA_STATUS_SUCCESS;
}

void
CheckInitError()
{
    printf("ROCm initialization failed\n");

    // Check kernel module for ROCk is loaded
    FILE* fd = popen("lsmod | grep amdgpu", "r");
    char  buf[16];
    if(fread(buf, 1, sizeof(buf), fd) <= 0)
    {
        printf("ROCk module is NOT loaded, possibly no GPU devices\n");
        return;
    }

    // Check if user belongs to group "video"
    // @note: User who are not members of "video"
    // group cannot access DRM services
    int           status    = -1;
    bool          member    = false;
    char          gr_name[] = "video";
    struct group* grp       = nullptr;
    do
    {
        grp = getgrent();
        if(grp == nullptr)
        {
            break;
        }
        status = memcmp(gr_name, grp->gr_name, sizeof(gr_name));
        if(status == 0)
        {
            member = true;
            break;
        }
    } while(grp != nullptr);

    if(member == false)
    {
        printf("User is not member of \"video\" group\n");
        return;
    }
}
}  // namespace

// Print out all static information known to HSA about the target system.
// Throughout this program, the Acquire-type functions make HSA calls to
// interate through HSA objects and then perform HSA get_info calls to
// acccumulate information about those objects. Corresponding to each
// Acquire-type function is a Display* function which display the
// accumulated data in a formatted way.
int
get_info(rocm_info& info)
{
    RET_IF_HSA_INIT_ERR(hsa_init());

    // This function will call HSA get_info functions to gather information
    // about the system.
    RET_IF_HSA_ERR(AcquireSystemInfo(&info.system));

    RET_IF_HSA_ERR(hsa_iterate_agents(AcquireAgentInfo, &info));

    RET_IF_HSA_ERR(hsa_shut_down());

    return HSA_STATUS_SUCCESS;
}

#undef RET_IF_HSA_ERR
}  // namespace test
}  // namespace rocprofiler
