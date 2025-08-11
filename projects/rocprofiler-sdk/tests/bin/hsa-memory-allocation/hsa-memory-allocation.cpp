// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>

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
            abort();                                                                               \
        }                                                                                          \
    }

// Callback function to get the list of agents
hsa_status_t
get_agents(hsa_agent_t agent, void* data)
{
    hsa_agent_t** agent_list = (hsa_agent_t**) data;
    **agent_list             = agent;
    ++(*agent_list);

    return HSA_STATUS_SUCCESS;
}

// Callback function to get the number of agents
hsa_status_t
get_num_agents(hsa_agent_t agent, void* data)
{
    (void) agent;
    int* num_agents = (int*) data;
    ++(*num_agents);

    return HSA_STATUS_SUCCESS;
}

// Callback function to get the number of regions of an agent
hsa_status_t
callback_get_num_regions(hsa_region_t region, void* data)
{
    (void) region;
    int* num_regions = (int*) data;
    ++(*num_regions);
    return HSA_STATUS_SUCCESS;
}

// Callback function to get the number of memory pools of an agent
hsa_status_t
callback_get_num_pools(hsa_amd_memory_pool_t memory_pool, void* data)
{
    (void) memory_pool;
    int* num_pools = (int*) data;
    ++(*num_pools);
    return HSA_STATUS_SUCCESS;
}

// Callback function to get the list of regions of an agent
hsa_status_t
callback_get_regions(hsa_region_t region, void* data)
{
    hsa_region_t** region_list = (hsa_region_t**) data;
    **region_list              = region;
    ++(*region_list);
    return HSA_STATUS_SUCCESS;
}

// Callback function to get the list of memory pools of an agent
hsa_status_t
callback_get_memory_pools(hsa_amd_memory_pool_t memory_pool, void* data)
{
    hsa_amd_memory_pool_t** pool_list = (hsa_amd_memory_pool_t**) data;
    **pool_list                       = memory_pool;
    ++(*pool_list);
    return HSA_STATUS_SUCCESS;
}

std::vector<hsa_agent_t>
get_agent_list()
{
    size_t       num_agents = 0;
    hsa_status_t status;
    // Get number of agents
    status = hsa_iterate_agents(get_num_agents, &num_agents);
    RET_IF_HSA_ERR(status)
    if(num_agents < 2)
    {
        printf("Not enough HSA agents available\n");
        abort();
    }

    // Create a array of size num_agents to store the agent list
    std::vector<hsa_agent_t> agents(num_agents);

    // Get the agent list
    hsa_agent_t* agent_iter = agents.data();
    status                  = hsa_iterate_agents(get_agents, &agent_iter);
    RET_IF_HSA_ERR(status)

    return agents;
}

hsa_agent_t
get_cpu_agent(std::vector<hsa_agent_t>& agents)
{
    for(hsa_agent_t agent : agents)
    {
        hsa_device_type_t ag_type;
        hsa_status_t      status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &ag_type);
        RET_IF_HSA_ERR(status)

        if(ag_type == HSA_DEVICE_TYPE_CPU)
        {
            return agent;
        }
    }
    std::cerr << "No CPU agents available" << std::endl;
    abort();
}

hsa_agent_t
get_gpu_agent(std::vector<hsa_agent_t>& agents)
{
    for(hsa_agent_t agent : agents)
    {
        hsa_device_type_t ag_type;
        hsa_status_t      status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &ag_type);
        RET_IF_HSA_ERR(status)

        if(ag_type == HSA_DEVICE_TYPE_GPU)
        {
            return agent;
        }
    }
    std::cerr << "No GPU agents available" << std::endl;
    abort();
}

void
call_hsa_memory_allocate(const size_t i, const size_t base_size, hsa_agent_t agent)
{
    // Getting total number of regions for the agent
    int          num_regions = 0;
    hsa_status_t status = hsa_agent_iterate_regions(agent, callback_get_num_regions, &num_regions);
    RET_IF_HSA_ERR(status)
    if(num_regions < 1)
    {
        printf("No HSA regions available\n");
        abort();
    }
    // Allocate memory to hold region list of an agent
    std::vector<hsa_region_t> region_list(num_regions);
    hsa_region_t*             ptr_reg = region_list.data();
    status = hsa_agent_iterate_regions(agent, callback_get_regions, &ptr_reg);
    RET_IF_HSA_ERR(status)
    auto address_vec = std::vector<void*>{};
    address_vec.reserve(i);

    for(size_t j = 0; j < i; ++j)
    {
        void* addr = nullptr;

        status = hsa_memory_allocate(region_list[0], base_size, &addr);
        RET_IF_HSA_ERR(status)
        address_vec.emplace_back(addr);
    }
    for(void* addr : address_vec)
    {
        status = hsa_memory_free(addr);
        RET_IF_HSA_ERR(status)
    }
}

void
call_hsa_memory_pool_allocate(const size_t i, const size_t base_size, hsa_agent_t agent)
{
    // Getting total number of regions for the agent
    int          num_pools = 0;
    hsa_status_t status =
        hsa_amd_agent_iterate_memory_pools(agent, callback_get_num_pools, &num_pools);
    RET_IF_HSA_ERR(status)
    if(num_pools < 1)
    {
        printf("No memory pools available\n");
        abort();
    }
    // Allocate memory to hold region list of an agent
    std::vector<hsa_amd_memory_pool_t> memory_pool_list(num_pools);
    hsa_amd_memory_pool_t*             ptr_memory_pool = memory_pool_list.data();
    status = hsa_amd_agent_iterate_memory_pools(agent, callback_get_memory_pools, &ptr_memory_pool);
    RET_IF_HSA_ERR(status)
    auto address_vec = std::vector<void*>{};
    address_vec.reserve(i);

    for(size_t j = 0; j < i; ++j)
    {
        void*    addr  = nullptr;
        uint32_t flags = 0;

        status = hsa_amd_memory_pool_allocate(memory_pool_list[0], base_size, flags, &addr);
        RET_IF_HSA_ERR(status)
        address_vec.emplace_back(addr);
    }
    for(void* addr : address_vec)
    {
        status = hsa_amd_memory_pool_free(addr);
        RET_IF_HSA_ERR(status)
    }
}

void
call_hsa_vmem_allocate(const size_t i, hsa_agent_t agent)
{
    // Getting total number of regions for the agent
    int          num_pools = 0;
    hsa_status_t status =
        hsa_amd_agent_iterate_memory_pools(agent, callback_get_num_pools, &num_pools);
    RET_IF_HSA_ERR(status)
    if(num_pools < 1)
    {
        printf("No memory pools available\n");
        abort();
    }
    // Allocate memory to hold region list of an agent
    std::vector<hsa_amd_memory_pool_t> memory_pool_list(num_pools);
    hsa_amd_memory_pool_t*             ptr_memory_pool = memory_pool_list.data();
    status = hsa_amd_agent_iterate_memory_pools(agent, callback_get_memory_pools, &ptr_memory_pool);
    RET_IF_HSA_ERR(status)

    // Ensure Virtual Memory API is supported
    bool supp = false;
    status    = hsa_system_get_info(HSA_AMD_SYSTEM_INFO_VIRTUAL_MEM_API_SUPPORTED, (void*) &supp);
    RET_IF_HSA_ERR(status)
    if(!supp)
    {
        std::cerr << "Virtual Memory API not supported" << std::endl;
        abort();
    }

    // Get runtime allocation granule size. Required for vmem_handle_create
    int size;
    status = hsa_amd_memory_pool_get_info(
        memory_pool_list[0], HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE, (void*) &size);
    RET_IF_HSA_ERR(status)
    for(size_t j = 0; j < i; ++j)
    {
        hsa_amd_vmem_alloc_handle_t memory_handle{};

        status = hsa_amd_vmem_handle_create(
            memory_pool_list[0], size, MEMORY_TYPE_NONE, 0, &memory_handle);
        RET_IF_HSA_ERR(status)
        status = hsa_amd_vmem_handle_release(memory_handle);
        RET_IF_HSA_ERR(status)
    }
}

int
main()
{
    hsa_status_t status;
    status = hsa_init();
    RET_IF_HSA_ERR(status)

    std::vector<hsa_agent_t> agents    = get_agent_list();
    hsa_agent_t              cpu_agent = get_cpu_agent(agents);
    hsa_agent_t              gpu_agent = get_gpu_agent(agents);
    call_hsa_memory_allocate(6, 1024, cpu_agent);
    call_hsa_memory_pool_allocate(9, 2048, gpu_agent);
    // Virtual memory API not supported in CI. Will add back if this changes
    // call_hsa_vmem_allocate(3, gpu_agent);

    status = hsa_shut_down();
    RET_IF_HSA_ERR(status)
    return 0;
}
