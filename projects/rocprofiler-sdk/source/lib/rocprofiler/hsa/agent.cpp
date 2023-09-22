// Copyright (c) 2018-2023 Advanced Micro Devices, Inc.
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

#include "agent.hpp"

#include <glog/logging.h>
#include <filesystem>
#include <fstream>

#include "lib/common/utility.hpp"

namespace fs = std::filesystem;

namespace rocprofiler
{
namespace hsa
{
namespace
{
std::unordered_map<long long, long long>
get_gpu_nodes_near_cpu()
{
    std::unordered_map<long long, long long> gpu_numa_nodes_near_cpu;
    long long                                gpu_numa_nodes_start = 0;

    std::string path = "/sys/class/kfd/kfd/topology/nodes";
    for(const auto& entry : fs::directory_iterator(path))
    {
        long long     node_id = std::stoll(entry.path().filename().c_str());
        std::ifstream gpu_id_file;
        std::string   gpu_path = entry.path().c_str();
        gpu_path += "/gpu_id";
        gpu_id_file.open(gpu_path);
        std::string gpu_id_str;
        if(gpu_id_file.is_open())
        {
            gpu_id_file >> gpu_id_str;

            if(!gpu_id_str.empty())
            {
                auto gpu_id = std::stoll(gpu_id_str);
                if(gpu_id > 0 && (gpu_numa_nodes_start > node_id || gpu_numa_nodes_start == 0))
                {
                    gpu_numa_nodes_start = node_id;
                }
            }
        }
        gpu_id_file.close();
    }

    path = "/sys/class/kfd/kfd/topology/nodes";
    for(const auto& entry : fs::directory_iterator(path))
    {
        long long   node_id        = std::stoll(entry.path().filename().c_str());
        std::string numa_node_path = entry.path().c_str();
        long long   agent_id       = std::stoll(entry.path().filename().c_str());
        if(agent_id >= gpu_numa_nodes_start)
        {
            numa_node_path += "/io_links";
            for(const auto& numa_node_entry : fs::directory_iterator(numa_node_path))
            {
                std::string numa_node_entry_properties_path = numa_node_entry.path().c_str();
                numa_node_entry_properties_path += "/properties";
                std::ifstream gpu_properties_file;
                gpu_properties_file.open(numa_node_entry_properties_path);
                std::string gpu_properties_file_line;
                if(gpu_properties_file.is_open())
                {
                    while(gpu_properties_file)
                    {
                        std::getline(gpu_properties_file, gpu_properties_file_line);
                        std::string       delimiter = " ";
                        std::stringstream ss(gpu_properties_file_line);
                        std::string       word;
                        ss >> word;
                        if(word == "node_to")
                        {
                            ss >> word;
                            long long near_cpu_node_id = std::stoll(word);
                            if(near_cpu_node_id < gpu_numa_nodes_start)
                            {
                                gpu_numa_nodes_near_cpu[node_id] = near_cpu_node_id;
                            }
                        }
                    }
                }
                gpu_properties_file.close();
            }
        }
    }
    return gpu_numa_nodes_near_cpu;
}

// This function checks to see if the provided
// pool has the HSA_AMD_SEGMENT_GLOBAL property. If the kern_arg flag is true,
// the function adds an additional requirement that the pool have the
// HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT property. If kern_arg is false,
// pools must NOT have this property.
// Upon finding a pool that meets these conditions, HSA_STATUS_INFO_BREAK is
// returned. HSA_STATUS_SUCCESS is returned if no errors were encountered, but
// no pool was found meeting the requirements. If an error is encountered, we
// return that error.
hsa_status_t
FindGlobalPool(hsa_amd_memory_pool_t pool, void* data, bool kern_arg)
{
    if(!data) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

    auto [api_ptr, pool_ptr] =
        *static_cast<std::pair<const AmdExtTable*, hsa_amd_memory_pool_t*>*>(data);
    hsa_amd_segment_t segment;
    LOG_IF(FATAL,
           api_ptr->hsa_amd_memory_pool_get_info_fn(
               pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment) == HSA_STATUS_ERROR)
        << "Could not get pool segment";
    if(HSA_AMD_SEGMENT_GLOBAL != segment) return HSA_STATUS_SUCCESS;

    uint32_t flag;
    LOG_IF(FATAL,
           api_ptr->hsa_amd_memory_pool_get_info_fn(
               pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flag) == HSA_STATUS_ERROR)
        << "Could not get flag value";
    uint32_t karg_st = flag & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT;
    if((karg_st == 0 && kern_arg) || (karg_st != 0 && !kern_arg))
    {
        return HSA_STATUS_SUCCESS;
    }
    *(pool_ptr) = pool;
    return HSA_STATUS_INFO_BREAK;
}

// This is the call-back function for hsa_amd_agent_iterate_memory_pools() that
// finds a pool with the properties of HSA_AMD_SEGMENT_GLOBAL and that is NOT
// HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT
hsa_status_t
FindStandardPool(hsa_amd_memory_pool_t pool, void* data)
{
    return FindGlobalPool(pool, data, false);
}

// This is the call-back function for hsa_amd_agent_iterate_memory_pools() that
// finds a pool with the properties of HSA_AMD_SEGMENT_GLOBAL and that IS
// HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT
hsa_status_t
FindKernArgPool(hsa_amd_memory_pool_t pool, void* data)
{
    return FindGlobalPool(pool, data, true);
}

void
init_cpu_pool(const AmdExtTable& api, AgentInfo& cpu_agent)
{
    CHECK(!cpu_agent.isGpu());
    auto params = std::make_pair(&api, &cpu_agent.cpu_pool);

    auto status =
        api.hsa_amd_agent_iterate_memory_pools_fn(cpu_agent.getAgent(), FindStandardPool, &params);
    LOG_IF(FATAL, status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK)
        << "Error: Command Buffer Pool is not initialized";

    params.second = &cpu_agent.kernarg_pool;
    status =
        api.hsa_amd_agent_iterate_memory_pools_fn(cpu_agent.getAgent(), FindKernArgPool, &(params));
    LOG_IF(FATAL, status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK)
        << "Error: Output Buffer Pool is not initialized";
}

void
init_gpu_pool(const AmdExtTable& api, AgentInfo& agent_info)
{
    CHECK(agent_info.isGpu());
    auto params = std::make_pair(&api, &agent_info.gpu_pool);
    auto status =
        api.hsa_amd_agent_iterate_memory_pools_fn(agent_info.getAgent(), FindStandardPool, &params);

    LOG_IF(FATAL, status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK)
        << "Error: GPU Pool is not initialized";
}

}  // namespace

const std::vector<AgentInfo>&
all_agents()
{
    static std::shared_ptr<const std::vector<AgentInfo>> agents = AgentInfo::getAgents(
        {.hsa_iterate_agents_fn = hsa_iterate_agents, .hsa_agent_get_info_fn = hsa_agent_get_info},
        {.hsa_amd_memory_pool_get_info_fn       = hsa_amd_memory_pool_get_info,
         .hsa_amd_agent_iterate_memory_pools_fn = hsa_amd_agent_iterate_memory_pools,
         .hsa_amd_memory_pool_allocate_fn       = hsa_amd_memory_pool_allocate,
         .hsa_amd_memory_pool_free_fn           = hsa_amd_memory_pool_free,
         .hsa_amd_agents_allow_access_fn        = hsa_amd_agents_allow_access});
    return *agents;
}

std::shared_ptr<const std::vector<AgentInfo>>
AgentInfo::getAgents(const CoreApiTable& api, const AmdExtTable& ext_api)
{
    std::vector<hsa_agent_t>                agents;
    std::shared_ptr<std::vector<AgentInfo>> agent_info_ptr =
        std::make_shared<std::vector<AgentInfo>>();
    auto& agent_info = *agent_info_ptr;

    api.hsa_iterate_agents_fn(
        [](hsa_agent_t agent, void* data) {
            CHECK_NOTNULL(static_cast<std::vector<hsa_agent_t>*>(data))->emplace_back(agent);
            return HSA_STATUS_SUCCESS;
        },
        &agents);

    auto                                    near_gpu_map = get_gpu_nodes_near_cpu();
    std::unordered_map<int64_t, AgentInfo*> cpu_id_to_agent;

    // Reserve is required to prevent reallocation (which breaks cpu_id_to_agent)
    agent_info.reserve(agents.size());
    for(auto& agent : agents)
    {
        auto& new_agent = agent_info.emplace_back(agent, api);
        if(!new_agent.isGpu())
        {
            uint32_t cpu_numa_node_id;
            LOG_IF(FATAL,
                   api.hsa_agent_get_info_fn(agent, HSA_AGENT_INFO_NODE, &cpu_numa_node_id) !=
                       HSA_STATUS_SUCCESS)
                << "Could not fetch numa info";
            new_agent.setNumaNode(cpu_numa_node_id);
            cpu_id_to_agent[cpu_numa_node_id] = &new_agent;
            init_cpu_pool(ext_api, new_agent);
        }
        else if(new_agent.isGpu())
        {
            uint32_t node_id;
            LOG_IF(FATAL,
                   api.hsa_agent_get_info_fn(
                       agent,
                       static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_DRIVER_NODE_ID),
                       &node_id) != HSA_STATUS_SUCCESS)
                << "Could not fetch driver node id";
            new_agent.setIndex(node_id);
            LOG_IF(FATAL,
                   api.hsa_agent_get_info_fn(agent,
                                             static_cast<hsa_agent_info_t>(HSA_AGENT_INFO_NODE),
                                             &node_id) != HSA_STATUS_SUCCESS)
                << "Could not fetch driver node id";
            new_agent.setNumaNode(node_id);
            init_gpu_pool(ext_api, new_agent);
        }
    }

    // Sperate for loop to allow cpu_id_to_agent to populate (in case CPUs are not always the first
    // NUMA nodes)
    for(auto& agent : agent_info)
    {
        if(agent.isGpu())
        {
            auto* near_gpu = common::get_val(near_gpu_map, agent.getNumaNode());
            LOG_IF(FATAL, !near_gpu) << fmt::format("No CPU Agent near GPU Agent: {} {}", agent);

            auto* id_to_agent = common::get_val(cpu_id_to_agent, *near_gpu);
            LOG_IF(FATAL, !id_to_agent) << fmt::format("Cannot convert id to agent: {}", *near_gpu);
            agent.setNearCpuAgent((*id_to_agent)->getAgent());
            agent.cpu_pool     = (*id_to_agent)->cpu_pool;
            agent.kernarg_pool = (*id_to_agent)->kernarg_pool;
        }
    }
    return agent_info_ptr;
}

AgentInfo::AgentInfo(const hsa_agent_t agent, const ::CoreApiTable& table)
: handle_(agent.handle)
, agent_(agent)
{
    if(table.hsa_agent_get_info_fn(agent, HSA_AGENT_INFO_DEVICE, &type_) != HSA_STATUS_SUCCESS)
    {
        LOG(FATAL) << "hsa_agent_get_info failed";
    }

    table.hsa_agent_get_info_fn(agent, HSA_AGENT_INFO_NAME, name_);

    const int gfxip_label_len = std::min(strlen(name_) - 2, sizeof(gfxip_) - 1);
    memcpy(gfxip_, name_, gfxip_label_len);
    gfxip_[gfxip_label_len] = '\0';

    if(type_ != HSA_DEVICE_TYPE_GPU)
    {
        return;
    }

    table.hsa_agent_get_info_fn(agent, HSA_AGENT_INFO_WAVEFRONT_SIZE, &max_wave_size_);
    table.hsa_agent_get_info_fn(agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &max_queue_size_);

    table.hsa_agent_get_info_fn(
        agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT), &cu_num_);

    table.hsa_agent_get_info_fn(
        agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU), &simds_per_cu_);

    table.hsa_agent_get_info_fn(
        agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_NUM_SHADER_ENGINES), &se_num_);

    if(table.hsa_agent_get_info_fn(agent,
                                   (hsa_agent_info_t) HSA_AMD_AGENT_INFO_NUM_SHADER_ARRAYS_PER_SE,
                                   &shader_arrays_per_se_) != HSA_STATUS_SUCCESS ||
       table.hsa_agent_get_info_fn(agent,
                                   (hsa_agent_info_t) HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU,
                                   &waves_per_cu_) != HSA_STATUS_SUCCESS)
    {
        LOG(FATAL) << "hsa_agent_get_info for gfxip hardware configuration failed";
    }

    compute_units_per_sh_ = cu_num_ / (se_num_ * shader_arrays_per_se_);
    wave_slots_per_simd_  = waves_per_cu_ / simds_per_cu_;

    if(table.hsa_agent_get_info_fn(agent,
                                   (hsa_agent_info_t) HSA_AMD_AGENT_INFO_DOMAIN,
                                   &pci_domain_) != HSA_STATUS_SUCCESS ||
       table.hsa_agent_get_info_fn(agent,
                                   (hsa_agent_info_t) HSA_AMD_AGENT_INFO_BDFID,
                                   &pci_location_id_) != HSA_STATUS_SUCCESS)
    {
        LOG(FATAL) << "hsa_agent_get_info for PCI info failed";
    }
}

uint64_t
AgentInfo::getIndex() const
{
    return index_;
}

hsa_device_type_t
AgentInfo::getType() const
{
    return type_;
}

uint64_t
AgentInfo::getHandle() const
{
    return handle_;
}

const std::string_view
AgentInfo::getName() const
{
    return name_;
}

std::string
AgentInfo::getGfxip() const
{
    return std::string(gfxip_);
}

uint32_t
AgentInfo::getMaxWaveSize() const
{
    return max_wave_size_;
}

uint32_t
AgentInfo::getMaxQueueSize() const
{
    return max_queue_size_;
}

uint32_t
AgentInfo::getCUCount() const
{
    return cu_num_;
}

uint32_t
AgentInfo::getSimdCountPerCU() const
{
    return simds_per_cu_;
}

uint32_t
AgentInfo::getShaderEngineCount() const
{
    return se_num_;
}

uint32_t
AgentInfo::getShaderArraysPerSE() const
{
    return shader_arrays_per_se_;
}

uint32_t
AgentInfo::getMaxWavesPerCU() const
{
    return waves_per_cu_;
}

uint32_t
AgentInfo::getCUCountPerSH() const
{
    return compute_units_per_sh_;
}

uint32_t
AgentInfo::getWaveSlotsPerSimd() const
{
    return wave_slots_per_simd_;
}

uint32_t
AgentInfo::getPCIDomain() const
{
    return pci_domain_;
}

uint32_t
AgentInfo::getPCILocationID() const
{
    return pci_location_id_;
}

uint32_t
AgentInfo::getXccCount() const
{
    return xcc_num_;
}

void
AgentInfo::setIndex(uint64_t index)
{
    index_ = index;
}

void
AgentInfo::setType(hsa_device_type_t type)
{
    type_ = type;
}

void
AgentInfo::setHandle(uint64_t handle)
{
    handle_ = handle;
}

void
AgentInfo::setName(const std::string& name)
{
    constexpr auto name_len = sizeof(name_) / sizeof(char);
    //
    // char* strncpy(char* destination, const char* source, size_t num)
    //
    // If the end of the source string (which is signaled by a null-character) is found before num
    // characters have been copied, destination is padded with zeros until a total of num characters
    // have been written to it
    strncpy(name_, name.c_str(), name_len - 2);
    // ensure always terminated
    name_[name_len - 1] = '\0';
}

void
AgentInfo::setNumaNode(uint32_t numa_node)
{
    numa_node_ = numa_node;
}

uint32_t
AgentInfo::getNumaNode() const
{
    return numa_node_;
}

void
AgentInfo::setNearCpuAgent(hsa_agent_t near_cpu_agent)
{
    near_cpu_agent_ = near_cpu_agent;
}

hsa_agent_t
AgentInfo::getNearCpuAgent()
{
    return near_cpu_agent_;
}
}  // namespace hsa
}  // namespace rocprofiler
