/* Copyright (c) 2022 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */
#ifndef SRC_CORE_HSA_PACKETS_PACKETS_GENERATOR_H_
#define SRC_CORE_HSA_PACKETS_PACKETS_GENERATOR_H_

#include "rocprofiler.h"

#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>
#include <hsa/hsa_ven_amd_loader.h>

#include <map>
#include <string>
#include <vector>
#include <mutex>

#include "src/core/counters/metrics/eval_metrics.h"

namespace Packet {

typedef hsa_ext_amd_aql_pm4_packet_t packet_t;

std::vector<std::pair<rocmtools::profiling_context_t*, hsa_ven_amd_aqlprofile_profile_t*>>
InitializeAqlPackets(hsa_agent_t cpu_agent, hsa_agent_t gpu_agent,
                     std::vector<std::string>& counter_names, bool is_spm = false);
uint8_t* AllocateSysMemory(hsa_agent_t gpu_agent, size_t size, hsa_amd_memory_pool_t* cpu_pool);
void GetCommandBufferMap(std::map<size_t, uint8_t*>);
void GetOutputBufferMap(std::map<size_t, uint8_t*>);
void InitializePools(hsa_agent_t cpu_agent, Agent::AgentInfo* agent_info);
void InitializeGPUPool(hsa_agent_t gpu_agent, Agent::AgentInfo* agent_info);
hsa_ven_amd_aqlprofile_profile_t* InitializeDeviceProfilingAqlPackets(
    hsa_agent_t cpu_agent, hsa_agent_t gpu_agent, hsa_ven_amd_aqlprofile_event_t* events,
    uint32_t event_count, packet_t* start_packet, packet_t* stop_packet, packet_t* read_packet);
hsa_amd_memory_pool_t& GetCommandPool();
hsa_amd_memory_pool_t& GetOutputPool();


hsa_ven_amd_aqlprofile_profile_t* GenerateATTPackets(
    hsa_agent_t cpu_agent, hsa_agent_t gpu_agent,
    std::vector<hsa_ven_amd_aqlprofile_parameter_t>& att_params, packet_t* start_packet,
    packet_t* stop_packet);


uint8_t* AllocateSysMemory(hsa_agent_t gpu_agent, size_t size, hsa_amd_memory_pool_t* cpu_pool);

void get_command_buffer_map(std::map<size_t, uint8_t*>);
void get_outbuffer_map(std::map<size_t, uint8_t*>);
void initialize_pools(hsa_agent_t cpu_agent);
void CheckPacketReqiurements(std::vector<hsa_agent_t>& gpu_agents);

typedef struct {
  hsa_amd_memory_pool_t cpu_mem_pool;
  hsa_amd_memory_pool_t gpu_mem_pool;
} att_memory_pools_t;

att_memory_pools_t* GetAttMemPools(hsa_agent_t gpu_agent);


}  // namespace Packet
#endif  // SRC_CORE_HSA_PACKETS_PACKETS_GENERATOR_H_
