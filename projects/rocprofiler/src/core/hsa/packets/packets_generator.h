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
#include <shared_mutex>

#include "src/core/counters/metrics/eval_metrics.h"

namespace Packet {

typedef hsa_ext_amd_aql_pm4_packet_t packet_t;


class AQLPacketProfile
{
public:
  AQLPacketProfile(decltype(hsa_amd_memory_pool_free) _free_fn)
  {
    profile = std::make_unique<hsa_ven_amd_aqlprofile_profile_t>();
    context = std::make_unique<rocprofiler::profiling_context_t>();
    this->free_fn = _free_fn;
    valid_profiles.fetch_add(1);
  };

  ~AQLPacketProfile();

  std::unique_ptr<hsa_ven_amd_aqlprofile_profile_t> profile;
  std::unique_ptr<rocprofiler::profiling_context_t> context;

  static std::unique_ptr<AQLPacketProfile> MoveFromCache(hsa_agent_t gpu_agent);
  static void MoveToCache(hsa_agent_t gpu_agent, std::unique_ptr<AQLPacketProfile>&& packet);

  static void WaitForProfileDeletion();

  static std::atomic<int> valid_profiles;
  static std::condition_variable_any delete_cv;
  static std::shared_mutex deleter_mutex;
  static bool IsDeletingBegin;
  decltype(hsa_amd_memory_pool_free)* free_fn;
};

std::unique_ptr<AQLPacketProfile> InitializeAqlPackets(
  hsa_agent_t cpu_agent,
  hsa_agent_t gpu_agent,
  std::vector<std::string>& counter_names,
  rocprofiler_session_id_t session_id,
  bool is_spm = false
);

uint8_t* AllocateSysMemory(hsa_agent_t gpu_agent, size_t size, hsa_amd_memory_pool_t* cpu_pool);
void GetCommandBufferMap(std::map<size_t, uint8_t*>);
void GetOutputBufferMap(std::map<size_t, uint8_t*>);
void InitializePools(hsa_agent_t cpu_agent, rocprofiler::HSAAgentInfo* agent_info);
void InitializeGPUPool(hsa_agent_t gpu_agent, rocprofiler::HSAAgentInfo* agent_info);
hsa_ven_amd_aqlprofile_profile_t* InitializeDeviceProfilingAqlPackets(
    hsa_agent_t cpu_agent, hsa_agent_t gpu_agent, hsa_ven_amd_aqlprofile_event_t* events,
    uint32_t event_count, packet_t* start_packet, packet_t* stop_packet, packet_t* read_packet);
hsa_amd_memory_pool_t& GetCommandPool();
hsa_amd_memory_pool_t& GetOutputPool();


hsa_ven_amd_aqlprofile_profile_t* GenerateATTPackets(
    hsa_agent_t cpu_agent, hsa_agent_t gpu_agent,
    std::vector<hsa_ven_amd_aqlprofile_parameter_t>& att_params, packet_t* start_packet,
    packet_t* stop_packet, size_t att_buffer_size);

hsa_ven_amd_aqlprofile_descriptor_t
GenerateATTMarkerPackets(
  hsa_agent_t gpu_agent,
  packet_t& marker_packet,
  uint32_t data,
  hsa_ven_amd_aqlprofile_att_marker_channel_t channel
);

uint8_t* AllocateSysMemory(hsa_agent_t gpu_agent, size_t size, hsa_amd_memory_pool_t* cpu_pool);

void get_command_buffer_map(std::map<size_t, uint8_t*>);
void get_outbuffer_map(std::map<size_t, uint8_t*>);
void CheckPacketReqiurements();

typedef struct {
  hsa_amd_memory_pool_t cpu_mem_pool;
  hsa_amd_memory_pool_t gpu_mem_pool;
} att_memory_pools_t;

att_memory_pools_t* GetAttMemPools(hsa_agent_t gpu_agent);

void AddVendorSpecificPacket(const packet_t* packet,
                             std::vector<packet_t>* transformed_packets,
                             const hsa_signal_t& packet_completion_signal);

void CreateBarrierPacket(std::vector<packet_t>* transformed_packets,
                         const hsa_signal_t* packet_dependency_signal,
                         const hsa_signal_t* packet_completion_signal);

bool IsDispatchPacket(const hsa_barrier_and_packet_t& packet);

// Returns a list of pointers to dispatch packets.
std::vector<const hsa_kernel_dispatch_packet_s*> ExtractDispatchPackets(
  const void* packets,
  int pkt_count
);

}  // namespace Packet
#endif  // SRC_CORE_HSA_PACKETS_PACKETS_GENERATOR_H_
