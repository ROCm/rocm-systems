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

#include "packets_generator.h"
#include "src/api/rocprofiler_singleton.h"

#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>
#include <stddef.h>
#include <stdint.h>
#include <numa.h>

#include <algorithm>
#include <atomic>
#include <exception>
#include <iostream>
#include <map>
#include <string>
#include <set>
#include <vector>
#include <utility>

#include "src/core/counters/basic/basic_counter.h"
#include "src/utils/exception.h"
#include "src/utils/logger.h"
#include "src/core/hsa/hsa_common.h"

#include "src/core/counters/metrics/metrics.h"
#include "src/core/hardware/hsa_info.h"


#define ASSERTM(exp, msg) assert(((void)msg, exp))

#define CHECK_HSA_STATUS(msg, status)                                                              \
  do {                                                                                             \
    if ((status) != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) {                       \
      try {                                                                                        \
        const char* emsg = nullptr;                                                                \
        hsa_status_string(status, &emsg);                                                          \
        if (!emsg) emsg = "<Unknown HSA Error>";                                                   \
        std::cerr << msg << std::endl;                                                             \
        std::cerr << emsg << std::endl;                                                            \
      } catch (std::exception & e) {                                                               \
      }                                                                                            \
      abort();                                                                                     \
    }                                                                                              \
  } while (0)

namespace Packet {
static const size_t MEM_PAGE_BYTES = 0x1000;
static const size_t MEM_PAGE_MASK = MEM_PAGE_BYTES - 1;

// This function checks to see if the provided
// pool has the HSA_AMD_SEGMENT_GLOBAL property. If the kern_arg flag is true,
// the function adds an additional requirement that the pool have the
// HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT property. If kern_arg is false,
// pools must NOT have this property.
// Upon finding a pool that meets these conditions, HSA_STATUS_INFO_BREAK is
// returned. HSA_STATUS_SUCCESS is returned if no errors were encountered, but
// no pool was found meeting the requirements. If an error is encountered, we
// return that error.
static hsa_status_t FindGlobalPool(hsa_amd_memory_pool_t pool, void* data, bool kern_arg) {
  [[maybe_unused]] hsa_status_t err;
  hsa_amd_segment_t segment;
  uint32_t flag;
  if (nullptr == data) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  err = rocprofiler::hsa_support::GetAmdExtTable().hsa_amd_memory_pool_get_info_fn(
      pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
  ASSERTM(err != HSA_STATUS_ERROR, "hsa_amd_memory_pool_get_info");
  if (HSA_AMD_SEGMENT_GLOBAL != segment) {
    return HSA_STATUS_SUCCESS;
  }
  err = rocprofiler::hsa_support::GetAmdExtTable().hsa_amd_memory_pool_get_info_fn(
      pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flag);
  ASSERTM(err != HSA_STATUS_ERROR, "hsa_amd_memory_pool_get_info");
  uint32_t karg_st = flag & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT;
  if ((karg_st == 0 && kern_arg) || (karg_st != 0 && !kern_arg)) {
    return HSA_STATUS_SUCCESS;
  }
  *(reinterpret_cast<hsa_amd_memory_pool_t*>(data)) = pool;
  return HSA_STATUS_INFO_BREAK;
}

// This is the call-back function for hsa_amd_agent_iterate_memory_pools() that
// finds a pool with the properties of HSA_AMD_SEGMENT_GLOBAL and that is NOT
// HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT
hsa_status_t FindStandardPool(hsa_amd_memory_pool_t pool, void* data) {
  return FindGlobalPool(pool, data, false);
}

// This is the call-back function for hsa_amd_agent_iterate_memory_pools() that
// finds a pool with the properties of HSA_AMD_SEGMENT_GLOBAL and that IS
// HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT
hsa_status_t FindKernArgPool(hsa_amd_memory_pool_t pool, void* data) {
  return FindGlobalPool(pool, data, true);
}

void InitializePools(hsa_agent_t cpu_agent, Agent::AgentInfo* agent_info) {
  hsa_status_t status =
      rocprofiler::hsa_support::GetAmdExtTable().hsa_amd_agent_iterate_memory_pools_fn(
          cpu_agent, FindStandardPool, &(agent_info->cpu_pool));
  CHECK_HSA_STATUS("Error: Command Buffer Pool is not initialized", status);

  status = rocprofiler::hsa_support::GetAmdExtTable().hsa_amd_agent_iterate_memory_pools_fn(
      cpu_agent, FindKernArgPool, &(agent_info->kernarg_pool));
  CHECK_HSA_STATUS("Error: Output Buffer Pool is not initialized", status);
}

void InitializeGPUPool(hsa_agent_t gpu_agent, Agent::AgentInfo* agent_info) {
  hsa_status_t status =
      hsa_amd_agent_iterate_memory_pools(gpu_agent, FindStandardPool, &(agent_info->gpu_pool));
  CHECK_HSA_STATUS("hsa_amd_agent_iterate_memory_pools(gpu_pool)", status);
}

struct block_des_t {
  uint32_t id;
  uint32_t index;
};

std::map<uint32_t, rocprofiler::MetricsDict*> metricsDict;
static std::atomic<bool> counters_added{false};

void CheckPacketReqiurements(std::vector<hsa_agent_t>& gpu_agents) {
  for (auto& gpu_agent : gpu_agents) {
    // get the instance of MetricsDict
    Agent::AgentInfo& agentInfo = rocprofiler::hsa_support::GetAgentInfo(gpu_agent.handle);
    metricsDict[gpu_agent.handle] = rocprofiler::MetricsDict::Create(&agentInfo);
  }
}

// Initialize the PM4 commands with having the CPU&GPU agents, the counters,
// counters count to output three packets which are start, stop and read
// packets
std::vector<std::pair<rocprofiler::profiling_context_t*, hsa_ven_amd_aqlprofile_profile_t*>>
InitializeAqlPackets(hsa_agent_t cpu_agent, hsa_agent_t gpu_agent,
                     std::vector<std::string>& counter_names, rocprofiler_session_id_t session_id,
                     bool is_spm) {
  hsa_status_t status = HSA_STATUS_SUCCESS;

  if (!counters_added.load(std::memory_order_acquire)) {
    for (auto& name : counter_names) {
      rocprofiler::GetROCProfilerSingleton()
          ->GetSession(session_id)
          ->GetProfiler()
          ->AddCounterName(name);
    }
    counters_added.exchange(true, std::memory_order_release);
  }

  Agent::AgentInfo& agentInfo = rocprofiler::hsa_support::GetAgentInfo(gpu_agent.handle);
  std::map<std::string, rocprofiler::results_t*> results_map;
  std::vector<rocprofiler::event_t> events_list;
  std::vector<rocprofiler::results_t*> results_list;
  std::map<std::pair<uint32_t, uint32_t>, uint64_t> event_to_max_block_count;
  std::map<std::string, std::set<std::string>> metrics_counters;

  if (!rocprofiler::metrics::ExtractMetricEvents(
          counter_names, gpu_agent, metricsDict[gpu_agent.handle], results_map, events_list,
          results_list, event_to_max_block_count, metrics_counters)) {
    abort();
  }

  // TODO: validate needs to be called on each events_list[i]
  // Validating the events array for the specified gpu agent
  bool validate_event_result;
  status =
      hsa_ven_amd_aqlprofile_validate_event(gpu_agent, &events_list[0], &validate_event_result);
  CHECK_HSA_STATUS("Error: Validating Counters", status);
  if (!validate_event_result) {
    std::cerr << "Error: Events are not valid for the current gpu agent" << std::endl;
    abort();
  }

  std::vector<std::pair<rocprofiler::profiling_context_t*, hsa_ven_amd_aqlprofile_profile_t*>>
      profiles = std::vector<
          std::pair<rocprofiler::profiling_context_t*, hsa_ven_amd_aqlprofile_profile_t*>>();

  // do {
  rocprofiler::profiling_context_t* context = new rocprofiler::profiling_context_t();
  context->gpu_agent = gpu_agent;
  auto result = results_list.begin();
  std::map<std::pair<uint32_t, uint32_t>, uint32_t> block_max_events_count;
  std::set<hsa_ven_amd_aqlprofile_block_name_t> block_names_taken;
  for (auto event = events_list.begin(); event != events_list.end();) {
    if (block_max_events_count[std::make_pair<uint32_t, uint32_t>(
            static_cast<uint32_t>(event->block_name), static_cast<uint32_t>(event->block_index))] <
        event_to_max_block_count[std::make_pair<uint32_t, uint32_t>(
            static_cast<uint32_t>(event->block_name), static_cast<uint32_t>(event->block_index))]) {
      context->events_list.push_back(*event);
      context->results_list.emplace_back(*result);
      block_max_events_count[std::make_pair<uint32_t, uint32_t>(
          static_cast<uint32_t>(event->block_name), static_cast<uint32_t>(event->block_index))]++;
      results_list.erase(result);
      events_list.erase(event);
    } else {
      event++;
      result++;
    }
  }

  std::set<std::string> counters_taken;

  std::set<std::string> metrics_counters_taken;

  for (auto result : context->results_list) {
    rocprofiler::Metric* metric;
    if (std::find(counter_names.begin(), counter_names.end(), result->name) !=
        counter_names.end()) {
      // std::cout << "Counter from Result List: " << result->name << std::endl;
      counters_taken.insert(result->name);
      metric = const_cast<rocprofiler::Metric*>(metricsDict[gpu_agent.handle]->Get(result->name));
      if (metric == nullptr) std::cout << result->name << " not found in metricsDict\n";
      context->metrics_list.push_back(metric);
    } else {
      metrics_counters_taken.insert(result->name);
      // std::cout << "Counter Added: " << result->name << std::endl;
    }
  }

  std::set<std::string> metrics_taken;

  for (auto result : results_map) {
    if (counters_taken.find(result.first) == counters_taken.end() &&
        std::find(counter_names.begin(), counter_names.end(), result.first) !=
            counter_names.end()) {
      bool flag = true;
      for (auto result_basic : results_list) {
        if (result_basic->name.compare(result.first)) {
          flag = false;
          break;
        }
      }
      if (flag) metrics_taken.insert(result.first);
    }
  }

  for (auto metric_name : metrics_taken) {
    bool flag = true;
    if (metrics_counters.find(metric_name) == metrics_counters.end()) continue;
    for (auto metric_counter_name : metrics_counters.at(metric_name)) {
      if (metrics_counters_taken.find(metric_counter_name) == metrics_counters_taken.end() &&
          counters_taken.find(metric_counter_name) == counters_taken.end()) {
        flag = false;
        continue;
      }
    }
    if (flag) {
      // std::cout << "Counter from Result Map: " << metric_name << std::endl;
      counters_taken.insert(metric_name);
      rocprofiler::Metric* metric =
          const_cast<rocprofiler::Metric*>(metricsDict[gpu_agent.handle]->Get(metric_name));
      if (metric == nullptr) std::cout << metric_name << " not found in metricsDict\n";
      context->metrics_list.push_back(metric);
    }
  }

  context->results_map = results_map;
  context->metrics_dict = metricsDict[gpu_agent.handle];

  hsa_ven_amd_aqlprofile_parameter_t* params = {};

  packet_t* start_packet = new packet_t();
  packet_t* stop_packet = new packet_t();
  packet_t* read_packet = new packet_t();

  if (context->events_list.size() <= 0) {
    std::cerr << "Error: No events to profile" << std::endl;
    abort();
  }

  // Preparing the profile structure to get the packets
  hsa_ven_amd_aqlprofile_event_type_t profile_type = HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_PMC;
  if (is_spm) profile_type = HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_TRACE;
  hsa_ven_amd_aqlprofile_profile_t* profile =
      new hsa_ven_amd_aqlprofile_profile_t{gpu_agent,
                                           profile_type,
                                           &(context->events_list[0]),
                                           static_cast<uint32_t>(context->events_list.size()),
                                           params,
                                           0,
                                           0,
                                           0};

  size_t ag_list_count = 1;  // rocprofiler::hsa_support::GetCPUAgentList().size();
  hsa_agent_t ag_list[ag_list_count];
  ag_list[0] = gpu_agent;

  // Preparing an Getting the size of the command and output buffers
  status = hsa_ven_amd_aqlprofile_start(profile, NULL);
  // CHECK_HSA_STATUS("Error: Getting Buffers Size", status);

  if (profile->command_buffer.size > 0 && profile->output_buffer.size > 0) {
    status = HSA_STATUS_ERROR;
    size_t size = profile->command_buffer.size;
    size = (size + MEM_PAGE_MASK) & ~MEM_PAGE_MASK;
    if (size <= 0) {
      std::cerr << __FILE__ << ":" << __LINE__ << " "
                << "Error: Command buffer given size is " << size << std::endl;
      abort();
    }
    status = rocprofiler::hsa_support::GetAmdExtTable().hsa_amd_memory_pool_allocate_fn(
        agentInfo.cpu_pool, size, 0, reinterpret_cast<void**>(&(profile->command_buffer.ptr)));
    if (status != HSA_STATUS_SUCCESS) {
      profile->command_buffer.ptr = malloc(size);
      /*numa_alloc_onnode(
          size,
          rocprofiler::hsa_support::GetAgentInfo(agentInfo.getNearCpuAgent().handle).getNumaNode());*/
      if (profile->command_buffer.ptr == NULL) {
        std::cerr << __FILE__ << ":" << __LINE__ << " "
                  << "Error: allocating memory for command buffer using NUMA" << std::endl;
        abort();
      }
    } else {
      // Both the CPU and GPU can access the memory
      status = rocprofiler::hsa_support::GetAmdExtTable().hsa_amd_agents_allow_access_fn(
          ag_list_count, ag_list, NULL, profile->command_buffer.ptr);
      CHECK_HSA_STATUS("Error: Allowing access to Command Buffer", status);
    }

    if (!is_spm) {
      status = HSA_STATUS_ERROR;
      size_t size = profile->output_buffer.size;
      size = (size + MEM_PAGE_MASK) & ~MEM_PAGE_MASK;
      if (size <= 0) {
        std::cerr << __FILE__ << ":" << __LINE__ << " "
                  << "Error: Output buffer given size is " << size << std::endl;
        abort();
      }
      status = rocprofiler::hsa_support::GetAmdExtTable().hsa_amd_memory_pool_allocate_fn(
          agentInfo.kernarg_pool, size, 0, reinterpret_cast<void**>(&profile->output_buffer.ptr));
      if (status != HSA_STATUS_SUCCESS) {
        profile->output_buffer.ptr = malloc(size);
        /*numa_alloc_onnode(
            size,
            rocprofiler::hsa_support::GetAgentInfo(agentInfo.getNearCpuAgent().handle)
                .getNumaNode());*/
        if (profile->output_buffer.ptr == NULL) {
          std::cerr << __FILE__ << ":" << __LINE__ << " "
                    << "Error: allocating memory for output buffer using NUMA" << std::endl;
          abort();
        }
      } else {
        status = rocprofiler::hsa_support::GetAmdExtTable().hsa_amd_agents_allow_access_fn(
            ag_list_count, ag_list, NULL, profile->output_buffer.ptr);
        CHECK_HSA_STATUS("Error: GPU Agent can't have output buffer access", status);
        memset(profile->output_buffer.ptr, 0x0, profile->output_buffer.size);
      }
    } else {
      profile->output_buffer.size = 0;
    }
    status = hsa_ven_amd_aqlprofile_start(profile, start_packet);
    // CHECK_HSA_STATUS("Error: Creating Start Packet\n", status);
    status = hsa_ven_amd_aqlprofile_stop(profile, stop_packet);
    // CHECK_HSA_STATUS("Error: Creating Stop Packet\n", status);
    status = hsa_ven_amd_aqlprofile_read(profile, read_packet);
    // CHECK_HSA_STATUS("Error: Creating Read Packet\n", status);

    context->start_packet = start_packet;
    context->stop_packet = stop_packet;
    context->read_packet = read_packet;

    // add profiles
    profiles.emplace_back(std::make_pair(context, profile));
  }
  // } while (events_list.size() > 0);
  return profiles;
}

// Initialize the PM4 commands with having the CPU&GPU agents, the counters,
// counters count to output three packets which are start, stop and read
// packets
hsa_ven_amd_aqlprofile_profile_t* InitializeDeviceProfilingAqlPackets(
    hsa_agent_t cpu_agent, hsa_agent_t gpu_agent, hsa_ven_amd_aqlprofile_event_t* events,
    uint32_t event_count, packet_t* start_packet, packet_t* stop_packet, packet_t* read_packet) {
  hsa_status_t status = HSA_STATUS_SUCCESS;

  // Validating the events array for the specified gpu agent
  bool result;
  status = hsa_ven_amd_aqlprofile_validate_event(gpu_agent, events, &result);
  CHECK_HSA_STATUS("Error: Events are not valid for the current gpu agent\n", status);

  hsa_ven_amd_aqlprofile_parameter_t* params = {};

  // Preparing the profile structure to get the packets
  hsa_ven_amd_aqlprofile_profile_t* profile = new hsa_ven_amd_aqlprofile_profile_t{
      gpu_agent, HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_PMC, events, event_count, params, 0, 0, 0};

  // Preparing an Getting the size of the command and output buffers
  status = hsa_ven_amd_aqlprofile_start(profile, NULL);

  Agent::AgentInfo& agentInfo = rocprofiler::hsa_support::GetAgentInfo(gpu_agent.handle);
  size_t ag_list_count = 1;
  hsa_agent_t ag_list[ag_list_count];
  ag_list[0] = gpu_agent;

  // Allocating Command Buffer
  status = HSA_STATUS_ERROR;
  size_t size = profile->command_buffer.size;
  profile->command_buffer.ptr = nullptr;
  if (size <= 0) return nullptr;
  size = (size + MEM_PAGE_MASK) & ~MEM_PAGE_MASK;
  status = rocprofiler::hsa_support::GetAmdExtTable().hsa_amd_memory_pool_allocate_fn(
      agentInfo.cpu_pool, size, 0, reinterpret_cast<void**>(&(profile->command_buffer.ptr)));
  // Both the CPU and GPU can access the memory
  if (status == HSA_STATUS_SUCCESS) {
    status = rocprofiler::hsa_support::GetAmdExtTable().hsa_amd_agents_allow_access_fn(
        ag_list_count, ag_list, NULL, profile->command_buffer.ptr);
    CHECK_HSA_STATUS("Error: GPU Agent can't have command buffer access", status);
  } else {
    profile->command_buffer.ptr = numa_alloc_onnode(
        profile->command_buffer.size,
        rocprofiler::hsa_support::GetAgentInfo(agentInfo.getNearCpuAgent().handle).getNumaNode());
    if (profile->command_buffer.ptr != nullptr) {
      status = HSA_STATUS_SUCCESS;
    } else {
      CHECK_HSA_STATUS("Error: Allocating Command Buffer", status);
    }
  }

  // Allocating Output Buffer
  status = HSA_STATUS_ERROR;
  size = profile->output_buffer.size;
  profile->output_buffer.ptr = nullptr;
  size = (size + MEM_PAGE_MASK) & ~MEM_PAGE_MASK;
  status = rocprofiler::hsa_support::GetAmdExtTable().hsa_amd_memory_pool_allocate_fn(
      agentInfo.gpu_pool, size, 0, reinterpret_cast<void**>(&(profile->output_buffer.ptr)));
  CHECK_HSA_STATUS("Error: Can't Allocate Output Buffer", status);
  // Both the CPU and GPU can access the kernel arguments
  if (status == HSA_STATUS_SUCCESS) {
    status = rocprofiler::hsa_support::GetAmdExtTable().hsa_amd_agents_allow_access_fn(
        ag_list_count, ag_list, NULL, profile->output_buffer.ptr);
    CHECK_HSA_STATUS("Error: Can't allow access on the Output Buffer for the GPU", status);
    memset(profile->output_buffer.ptr, 0x0, profile->output_buffer.size);
  }


  // Creating the start/stop/read packets
  status = hsa_ven_amd_aqlprofile_start(profile, start_packet);
  CHECK_HSA_STATUS("Error: Creating Start Packet\n", status);
  status = hsa_ven_amd_aqlprofile_stop(profile, stop_packet);
  CHECK_HSA_STATUS("Error: Creating Stop Packet\n", status);
  status = hsa_ven_amd_aqlprofile_read(profile, read_packet);
  CHECK_HSA_STATUS("Error: Creating Read Packet\n", status);

  if (status == HSA_STATUS_ERROR) return nullptr;
  return profile;
}

// ATT
uint32_t g_output_buffer_size = 0x40000000;  // 1GB
bool g_output_buffer_local = true;

// Allocate system memory accessible by both CPU and GPU
uint8_t* AllocateSysMemory(hsa_agent_t gpu_agent, size_t size, hsa_amd_memory_pool_t* cpu_pool) {
  size_t ag_list_count = 1;  // rocprofiler::hsa_support::GetCPUAgentList().size();
  hsa_agent_t ag_list[ag_list_count];
  ag_list[0] = gpu_agent;
  hsa_status_t status = HSA_STATUS_ERROR;
  uint8_t* buffer = NULL;
  size = (size + MEM_PAGE_MASK) & ~MEM_PAGE_MASK;
  status = rocprofiler::hsa_support::GetAmdExtTable().hsa_amd_memory_pool_allocate_fn(
      *cpu_pool, size, 0, reinterpret_cast<void**>(&buffer));
  // Both the CPU and GPU can access the memory
  if (status == HSA_STATUS_SUCCESS) {
    status = rocprofiler::hsa_support::GetAmdExtTable().hsa_amd_agents_allow_access_fn(
        ag_list_count, ag_list, NULL, buffer);
  }
  uint8_t* ptr = (status == HSA_STATUS_SUCCESS) ? buffer : NULL;
  return ptr;
}

// Allocate memory for use by a kernel of specified size
uint8_t* AllocateLocalMemory(size_t size, hsa_amd_memory_pool_t* gpu_pool) {
  hsa_status_t status = HSA_STATUS_ERROR;
  uint8_t* buffer = NULL;
  size = (size + MEM_PAGE_MASK) & ~MEM_PAGE_MASK;
  status = hsa_amd_memory_pool_allocate(*gpu_pool, size, 0, reinterpret_cast<void**>(&buffer));
  uint8_t* ptr = (status == HSA_STATUS_SUCCESS) ? buffer : NULL;
  return ptr;
}

hsa_status_t Allocate(hsa_agent_t gpu_agent, hsa_ven_amd_aqlprofile_profile_t* profile) {
  Agent::AgentInfo& agentInfo = rocprofiler::hsa_support::GetAgentInfo(gpu_agent.handle);
  profile->command_buffer.ptr =
      AllocateSysMemory(gpu_agent, profile->command_buffer.size, &agentInfo.cpu_pool);
  profile->output_buffer.size = g_output_buffer_size;
  profile->output_buffer.ptr = (g_output_buffer_local)
      ? AllocateLocalMemory(profile->output_buffer.size, &agentInfo.gpu_pool)
      : AllocateSysMemory(gpu_agent, profile->output_buffer.size, &agentInfo.cpu_pool);
  return (profile->command_buffer.ptr && profile->output_buffer.ptr) ? HSA_STATUS_SUCCESS
                                                                     : HSA_STATUS_ERROR;
}

bool AllocateMemoryPools(hsa_agent_t cpu_agent, hsa_agent_t gpu_agent,
                         hsa_amd_memory_pool_t* cpu_pool, hsa_amd_memory_pool_t* gpu_pool) {
  hsa_status_t status = hsa_amd_agent_iterate_memory_pools(cpu_agent, FindStandardPool, cpu_pool);
  CHECK_HSA_STATUS("hsa_amd_agent_iterate_memory_pools(cpu_pool)", status);

  status = hsa_amd_agent_iterate_memory_pools(gpu_agent, FindStandardPool, gpu_pool);
  CHECK_HSA_STATUS("hsa_amd_agent_iterate_memory_pools(gpu_pool)", status);

  return true;
}

// map between gpu agent handle and att_memory_pools_t
typedef std::map<uint64_t, att_memory_pools_t*> att_mem_pools_map_t;

att_mem_pools_map_t* agent_att_mem_pools_map = nullptr;
std::atomic<bool> att_map_init{false};

att_mem_pools_map_t* GetAttMemPoolsMap() {
  if (!att_map_init.load(std::memory_order_relaxed)) {
    agent_att_mem_pools_map = new att_mem_pools_map_t();
    att_map_init.exchange(true, std::memory_order_release);
  }

  return agent_att_mem_pools_map;
}

// Generate start and stop packets for collecting ATT traces
// Also generate and return the profile object which has the PM4
// command buffer and the output buffer for retrieving the traces
hsa_ven_amd_aqlprofile_profile_t* GenerateATTPackets(
    hsa_agent_t cpu_agent, hsa_agent_t gpu_agent,
    std::vector<hsa_ven_amd_aqlprofile_parameter_t>& att_params, packet_t* start_packet,
    packet_t* stop_packet) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion-null"
  // Preparing the profile structure to get the packets
  hsa_ven_amd_aqlprofile_profile_t* profile =
      new hsa_ven_amd_aqlprofile_profile_t{gpu_agent,      HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_TRACE,
                                           nullptr,        0,
                                           &att_params[0], (uint32_t)att_params.size(),
                                           NULL,           NULL};
#pragma GCC diagnostic pop

  // Check the profile buffer sizes
  hsa_status_t status = hsa_ven_amd_aqlprofile_start(profile, NULL);
  CHECK_HSA_STATUS("Error: Getting PM4 Start Packet", status);
  // TODO: create a separate class for memory allocations
  // Maintain pools per device
  // handle allocation and resource cleanup

  // Allocate command and output buffers
  // command buffer -> from CPU memory pool
  // output buffer -> from GPU memory pool
  status = Allocate(gpu_agent, profile);
  CHECK_HSA_STATUS("Error: Att Buffers Allocation", status);

  // Generate start/stop/read profiling packets
  status = hsa_ven_amd_aqlprofile_start(profile, start_packet);
  CHECK_HSA_STATUS("Error: Creating Start PM4 Packet", status);
  status = hsa_ven_amd_aqlprofile_stop(profile, stop_packet);
  CHECK_HSA_STATUS("Error: Creating Stop PM4 Packet", status);
  return profile;
}

}  // namespace Packet
