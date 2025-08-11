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
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "agent.hpp"
#include <cstring>

#define CHECK_HSA(x) if ((x) != HSA_STATUS_SUCCESS) { std::cerr << __FILE__ << " error at " << __LINE__ << std::endl; exit(-1); }

std::vector<std::shared_ptr<AgentInfo>> AgentInfo::gpu_agents{};
hsa_agent_t AgentInfo::cpu_agent{0};

hsa_amd_memory_pool_t AgentInfo::cpu_pool;
hsa_amd_memory_pool_t AgentInfo::kernarg_pool;

void AgentInfo::add_event(aqlprofile_pmc_event_t block, const std::string& counter, int block_cnt, int event_id)
{
    block.event_id = event_id;
    std::vector<aqlprofile_pmc_event_t> cnt{};
    for (int i=0; i<block_cnt; i++)
    {
        block.block_index = i;
        cnt.push_back(block);
    }
    counters[counter] = std::move(cnt);
}

hsa_status_t AgentInfo::get_agent_handle_cb(hsa_agent_t agent, void* userdata)
{
    hsa_device_type_t type;

    CHECK_HSA(hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type));

    if (type == HSA_DEVICE_TYPE_CPU)
    {
        cpu_agent = agent;
        return HSA_STATUS_SUCCESS;
    }

    std::shared_ptr<AgentInfo> info = std::make_shared<AgentInfo>();
    info->hsa_agent = agent;
    CHECK_HSA(hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, info->gfxip.data()));
    CHECK_HSA(hsa_agent_get_info(agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_NUM_XCC), &info->info.xcc_num));
    CHECK_HSA(hsa_agent_get_info(agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_NUM_SHADER_ENGINES), &info->info.se_num));
    CHECK_HSA(hsa_agent_get_info(agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT), &info->info.cu_num));
    CHECK_HSA(hsa_agent_get_info(agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_NUM_SHADER_ARRAYS_PER_SE), &info->info.shader_arrays_per_se));

    info->info.agent_gfxip = info->gfxip.data();
    CHECK_HSA(aqlprofile_register_agent(&info->handle, &info->info));

    aqlprofile_pmc_event_flags_t flags{.raw = 0};
    aqlprofile_pmc_event_t grbm {.block_index = 0, .flags = flags, .block_name = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBM};
    aqlprofile_pmc_event_t sq {.block_index = 0, .flags = flags, .block_name = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ};
    aqlprofile_pmc_event_t ta {.block_index = 0, .flags = flags, .block_name = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA};
    aqlprofile_pmc_event_t tcp {.block_index = 0, .flags = flags, .block_name = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP};
    aqlprofile_pmc_event_t tcc {.block_index = 0, .flags = flags, .block_name = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC};
    aqlprofile_pmc_event_t gl2c {.block_index = 0, .flags = flags, .block_name = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL2C};

    info->add_event(grbm, "GRBM_COUNT", 1, 0);
    info->add_event(grbm, "GRBM_GUI_ACTIVE", 1, 2);
    info->add_event(sq, "SQ_WAVES", 1, 4);
    info->add_event(sq, "SQ_BUSY_CYCLES", 1, 3);
    info->add_event(sq, "SQ_INSTS_VALU", 1, (info->gfxip.find("gfx1")==0) ? 62 : 26);
    
    info->add_event(ta, "TA_BUSY", 16, (info->gfxip.find("gfx94") != 0 || info->gfxip.find("gfx95") != 0) ? 13 : 15);

    if (info->gfxip.find("gfx1") == 0)
    {
        info->add_event(gl2c, "GL2C_REQ", 32, 3);
        info->add_event(gl2c, "GL2C_READ", 32, 6);
        info->add_event(gl2c, "GL2C_WRITE", 32, 7);
    }
    else if (info->gfxip.find("gfx95") == 0)
    {
        info->add_event(sq, "SQ_INSTS_VALU_FLOPS_FP16", 10, 81);
        info->add_event(sq, "SQ_INSTS_VALU_FLOPS_FP32", 10, 82);
        info->add_event(sq, "SQ_INSTS_VALU_FLOPS_FP64", 10, 83);
        info->add_event(sq, "SQ_INSTS_VALU_FLOPS_FP16_TRANS", 10, 84);
        info->add_event(sq, "SQ_INSTS_VALU_FLOPS_FP32_TRANS", 10, 85);
        info->add_event(sq, "SQ_INSTS_VALU_FLOPS_FP64_TRANS", 10, 86);

        info->add_event(tcp, "TCP_READ", 10, 28);
        info->add_event(tcp, "TCP_WRITE", 10, 30);
        info->add_event(tcp, "TCP_CACHE_ACCESS", 10, 58);
        info->add_event(tcp, "TCP_CACHE_MISS_TG0", 10, 59);
        info->add_event(tcp, "TCP_CACHE_MISS_TG1", 10, 60);
        info->add_event(tcp, "TCP_CACHE_MISS_TG2", 10, 61);
        info->add_event(tcp, "TCP_CACHE_MISS_TG3", 10, 62);
        info->add_event(tcp, "TCP_CACHE_MISS", 10, 63);

        info->add_event(tcc, "TCC_EA0_RDREQ", 16, 42);
        info->add_event(tcc, "TCC_EA0_RDREQ_DRAM", 16, 108);
        info->add_event(tcc, "TCC_EA0_WRREQ_DRAM", 16, 109);
        info->add_event(tcc, "TCC_EA0_WRREQ_WRITE_DRAM", 16, 110);
        info->add_event(tcc, "TCC_EA0_WRREQ_ATOMIC_DRAM", 16, 111);
        info->add_event(tcc, "TCC_EA0_RDREQ_DRAM_32B", 16, 112);
        info->add_event(tcc, "TCC_EA0_RDREQ_GMI_32B", 16, 113);
        info->add_event(tcc, "TCC_EA0_RDREQ_IO_32B", 16, 114);
        info->add_event(tcc, "TCC_EA0_WRREQ_WRITE_DRAM_32B", 16, 115);
        info->add_event(tcc, "TCC_EA0_WRREQ_ATOMIC_DRAM_32B", 16, 116);
        info->add_event(tcc, "TCC_EA0_WRREQ_WRITE_GMI_32B", 16, 117);
        info->add_event(tcc, "TCC_EA0_WRREQ_ATOMIC_GMI_32B", 16, 118);
        info->add_event(tcc, "TCC_EA0_WRREQ_WRITE_IO_32B", 16, 119);
        info->add_event(tcc, "TCC_EA0_WRREQ_ATOMIC_IO_32B", 16, 119);
    }
    else if (info->gfxip.find("gfx94") == 0)
    {
        info->add_event(tcc, "TCC_REQ", 16, 3);
        info->add_event(tcc, "TCC_ATOMIC", 16, 14);
        info->add_event(tcc, "TCC_EA0_ATOMIC", 16, 36);

        info->add_event(tcc, "TCC_EA0_WRREQ_CREDIT_STALL", 16, 30);
        info->add_event(tcc, "TCC_EA0_WRREQ_IO_CREDIT_STALL", 16, 31);
        info->add_event(tcc, "TCC_EA0_WRREQ_GMI_CREDIT_STALL", 16, 32);
        info->add_event(tcc, "TCC_EA0_WRREQ_DRAM_CREDIT_STALL", 16, 33);
        info->add_event(tcc, "TCC_EA0_RDREQ", 16, 38);
        info->add_event(tcc, "TCC_EA0_RDREQ_IO_CREDIT_STALL", 16, 41);
        info->add_event(tcc, "TCC_EA0_RDREQ_GMI_CREDIT_STALL", 16, 42);
        info->add_event(tcc, "TCC_EA0_RDREQ_DRAM_CREDIT_STALL", 16, 43);

        info->add_event(tcp, "TCP_READ", 10, 28);
        info->add_event(tcp, "TCP_WRITE", 10, 30);
        info->add_event(tcp, "TCP_CACHE_ACCESS", 10, 60);
        info->add_event(tcp, "TCP_CACHE_MISS_TG0", 10, 61);
        info->add_event(tcp, "TCP_CACHE_MISS_TG1", 10, 62);
        info->add_event(tcp, "TCP_CACHE_MISS_TG2", 10, 63);
        info->add_event(tcp, "TCP_CACHE_MISS_TG3", 10, 64);
    }
    else if (info->gfxip.find("gfx90a") == 0)
    {
        info->add_event(tcp, "TCP_READ", 16, 30);
        info->add_event(tcp, "TCP_WRITE", 16, 32);
    }
    else if (info->gfxip.find("gfx900") == 0)
    {
        info->add_event(tcp, "TCP_READ", 16, 30);
        info->add_event(tcp, "TCP_WRITE", 16, 32);
    }
    else
    {
        assert(false);
    }

    gpu_agents.push_back(info);
    return HSA_STATUS_SUCCESS;
}

hsa_status_t
FindGlobalPool(hsa_amd_memory_pool_t pool, void* data)
{
    hsa_amd_segment_t segment;
    CHECK_HSA(hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment));

    if(HSA_AMD_SEGMENT_GLOBAL != segment) return HSA_STATUS_SUCCESS;

    uint32_t flag;
    CHECK_HSA(hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flag));
    uint32_t karg_st = flag & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT;

    if (karg_st)
        AgentInfo::kernarg_pool = pool;
    else
        AgentInfo::cpu_pool = pool;

    return HSA_STATUS_SUCCESS;
}

void AgentInfo::iterate_agents()
{
    CHECK_HSA(hsa_iterate_agents(get_agent_handle_cb, nullptr));
    CHECK_HSA(hsa_amd_agent_iterate_memory_pools(cpu_agent, FindGlobalPool, nullptr));
}

bool
Queue::Submit(hsa_ext_amd_aql_pm4_packet_t* packet)
{
    const uint64_t write_idx = hsa_queue_add_write_index_relaxed(queue, 1);

    size_t index      = (write_idx % queue->size) * sizeof(hsa_ext_amd_aql_pm4_packet_t);
    auto*  queue_slot = reinterpret_cast<uint32_t*>(size_t(queue->base_address) + index);  // NOLINT

    const auto* slot_data = reinterpret_cast<const uint32_t*>(packet);

    std::memcpy(&queue_slot[1], &slot_data[1], sizeof(hsa_ext_amd_aql_pm4_packet_t) - sizeof(uint32_t));
    auto* header = reinterpret_cast<std::atomic<uint32_t>*>(queue_slot);

    header->store(slot_data[0], std::memory_order_release);
    hsa_signal_store_screlease(queue->doorbell_signal, write_idx);

    int loops = 0;
    while(hsa_queue_load_read_index_relaxed(queue) <= write_idx)
    {
        loops++;
        usleep(1);
        if(loops > 10000)
        {
            std::cerr << "Codeobj packet submission failed!" << std::endl;
            return false;
        }
    }
    return true;
}

Queue::Queue(std::shared_ptr<AgentInfo>& _agent): agent(_agent)
{
    CHECK_HSA(hsa_queue_create(agent->hsa_agent, 64, HSA_QUEUE_TYPE_SINGLE, NULL, NULL, UINT32_MAX, UINT32_MAX, &this->queue));
}

void Queue::flush()
{
    return;
    hsa_barrier_and_packet_t barrier{};
    barrier.header = HSA_PACKET_TYPE_BARRIER_OR | (1<<HSA_PACKET_HEADER_BARRIER);
    barrier.header |= HSA_FENCE_SCOPE_SYSTEM<<HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;
    barrier.header |= HSA_FENCE_SCOPE_SYSTEM<<HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
    Submit((hsa_ext_amd_aql_pm4_packet_t*)&barrier);
}