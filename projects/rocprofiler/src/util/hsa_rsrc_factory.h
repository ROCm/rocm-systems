/**********************************************************************
Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:

<95>    Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer.
<95>    Redistributions in binary form must reproduce the above copyright notice, this list of
conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#ifndef SRC_UTIL_HSA_RSRC_FACTORY_H_
#define SRC_UTIL_HSA_RSRC_FACTORY_H_

#include <hsa.h>
#include <hsa_ext_finalize.h>
#include <hsa_ven_amd_aqlprofile.h>
#include <hsa_ven_amd_loader.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <iostream>
#include <mutex>
#include <map>
#include <string>
#include <vector>

#define HSA_ARGUMENT_ALIGN_BYTES 16
#define HSA_QUEUE_ALIGN_BYTES 64
#define HSA_PACKET_ALIGN_BYTES 64

#define CHECK_STATUS(msg, status)                                                                  \
  if (status != HSA_STATUS_SUCCESS) {                                                              \
    const char* emsg = 0;                                                                          \
    hsa_status_string(status, &emsg);                                                              \
    printf("%s: %s\n", msg, emsg ? emsg : "<unknown error>");                                      \
    exit(1);                                                                                       \
  }

namespace rocprofiler {
namespace util {
static const unsigned MEM_PAGE_BYTES = 0x1000;
static const unsigned MEM_PAGE_MASK = MEM_PAGE_BYTES - 1;
typedef decltype(hsa_agent_t::handle) hsa_agent_handle_t;

// Encapsulates information about a Hsa Agent such as its
// handle, name, max queue size, max wavefront size, etc.
struct AgentInfo {
  // Handle of Agent
  hsa_agent_t dev_id;

  // Agent type - Cpu = 0, Gpu = 1 or Dsp = 2
  uint32_t dev_type;

  // APU flag
  bool is_apu;

  // Agent system index
  uint32_t dev_index;

  // GFXIP name
  char gfxip[64];

  // Name of Agent whose length is less than 64
  char name[64];

  // Max size of Wavefront size
  uint32_t max_wave_size;

  // Max size of Queue buffer
  uint32_t max_queue_size;

  // Hsail profile supported by agent
  hsa_profile_t profile;

  // Memory region supporting kernel parameters
  hsa_region_t coarse_region;

  // Memory region supporting kernel arguments
  hsa_region_t kernarg_region;

  // The number of compute unit available in the agent.
  uint32_t cu_num;

  // Maximum number of waves possible in a Compute Unit.
  uint32_t waves_per_cu;

  // Number of SIMD's per compute unit CU
  uint32_t simds_per_cu;

  // Number of Shader Engines (SE) in Gpu
  uint32_t se_num;

  // Number of Shader Arrays Per Shader Engines in Gpu
  uint32_t shader_arrays_per_se;
};

class HsaRsrcFactory {
 public:
  typedef std::recursive_mutex mutex_t;

  static HsaRsrcFactory* Create(bool initialize_hsa = true) {
    std::lock_guard<mutex_t> lck(mutex_);
    if (instance_ == NULL) {
      instance_ = new HsaRsrcFactory(initialize_hsa);
    }
    return instance_;
  }

  static HsaRsrcFactory& Instance() {
    if (instance_ == NULL) instance_ = Create(false);
    hsa_status_t status = (instance_ != NULL) ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR;
    CHECK_STATUS("HsaRsrcFactory::Instance() failed", status);
    return *instance_;
  }

  static void Destroy() {
    std::lock_guard<mutex_t> lck(mutex_);
    if (instance_) delete instance_;
    instance_ = NULL;
  }

  // Return system agent info
  const AgentInfo* GetAgentInfo(const hsa_agent_t agent);

  // Get the count of Hsa Gpu Agents available on the platform
  //
  // @return uint32_t Number of Gpu agents on platform
  //
  uint32_t GetCountOfGpuAgents();

  // Get the count of Hsa Cpu Agents available on the platform
  //
  // @return uint32_t Number of Cpu agents on platform
  //
  uint32_t GetCountOfCpuAgents();

  // Get the AgentInfo handle of a Gpu device
  //
  // @param idx Gpu Agent at specified index
  //
  // @param agent_info Output parameter updated with AgentInfo
  //
  // @return bool true if successful, false otherwise
  //
  bool GetGpuAgentInfo(uint32_t idx, const AgentInfo** agent_info);

  // Get the AgentInfo handle of a Cpu device
  //
  // @param idx Cpu Agent at specified index
  //
  // @param agent_info Output parameter updated with AgentInfo
  //
  // @return bool true if successful, false otherwise
  //
  bool GetCpuAgentInfo(uint32_t idx, const AgentInfo** agent_info);

  // Create a Queue object and return its handle. The queue object is expected
  // to support user requested number of Aql dispatch packets.
  //
  // @param agent_info Gpu Agent on which to create a queue object
  //
  // @param num_Pkts Number of packets to be held by queue
  //
  // @param queue Output parameter updated with handle of queue object
  //
  // @return bool true if successful, false otherwise
  //
  bool CreateQueue(const AgentInfo* agent_info, uint32_t num_pkts, hsa_queue_t** queue);

  // Create a Signal object and return its handle.
  //
  // @param value Initial value of signal object
  //
  // @param signal Output parameter updated with handle of signal object
  //
  // @return bool true if successful, false otherwise
  //
  bool CreateSignal(uint32_t value, hsa_signal_t* signal);

  // Allocate memory for use by a kernel of specified size in specified
  // agent's memory region. Currently supports Global segment whose Kernarg
  // flag set.
  //
  // @param agent_info Agent from whose memory region to allocate
  //
  // @param size Size of memory in terms of bytes
  //
  // @return uint8_t* Pointer to buffer, null if allocation fails.
  //
  uint8_t* AllocateLocalMemory(const AgentInfo* agent_info, size_t size);

  // Allocate memory tp pass kernel parameters.
  //
  // @param agent_info Agent from whose memory region to allocate
  //
  // @param size Size of memory in terms of bytes
  //
  // @return uint8_t* Pointer to buffer, null if allocation fails.
  //
  uint8_t* AllocateSysMemory(const AgentInfo* agent_info, size_t size);

  // Memcopy method
  static bool CopyToHost(void* dest_buff, const void* src_buff, uint32_t length);
  static bool Memcpy(hsa_agent_t agent, void* dest_buff, const void* src_buff, uint32_t length);

  // Free method
  static bool FreeMemory(void* ptr);

  // Loads an Assembled Brig file and Finalizes it into Device Isa
  //
  // @param agent_info Gpu device for which to finalize
  //
  // @param brig_path File path of the Assembled Brig file
  //
  // @param kernel_name Name of the kernel to finalize
  //
  // @param code_desc Handle of finalized Code Descriptor that could
  // be used to submit for execution
  //
  // @return true if successful, false otherwise
  //
  bool LoadAndFinalize(const AgentInfo* agent_info, const char* brig_path, const char* kernel_name,
                        hsa_executable_t* hsa_exec, hsa_executable_symbol_t* code_desc);

  // Print the various fields of Hsa Gpu Agents
  bool PrintGpuAgents(const std::string& header);

  // Submit AQL packet to given queue
  static uint64_t Submit(hsa_queue_t* queue, void* packet);

  // Return AqlProfile API table
  typedef hsa_ven_amd_aqlprofile_1_00_pfn_t aqlprofile_pfn_t;
  const aqlprofile_pfn_t* AqlProfileApi() const { return &aqlprofile_api_; }

  // Return Loader API table
  const hsa_ven_amd_loader_1_00_pfn_t* LoaderApi() const { return &loader_api_; }

 private:
  // System agents iterating callback
  static hsa_status_t GetHsaAgentsCallback(hsa_agent_t agent, void* data);

  // Callback function to find and bind kernarg region of an agent
  static hsa_status_t FindMemRegionsCallback(hsa_region_t region, void* data);

  // Load AQL profile HSA extension library directly
  static hsa_status_t LoadAqlProfileLib(aqlprofile_pfn_t* api);

  // Constructor of the class. Will initialize the Hsa Runtime and
  // query the system topology to get the list of Cpu and Gpu devices
  HsaRsrcFactory(bool initialize_hsa);

  // Destructor of the class
  ~HsaRsrcFactory();

  // HSA was initialized
  const bool initialize_hsa_;

  // Add an instance of AgentInfo representing a Hsa Gpu agent
  const AgentInfo* AddAgentInfo(const hsa_agent_t agent);

  static HsaRsrcFactory* instance_;
  static mutex_t mutex_;

  // Used to maintain a list of Hsa Gpu Agent Info
  std::vector<const AgentInfo*> gpu_list_;

  // Used to maintain a list of Hsa Cpu Agent Info
  std::vector<const AgentInfo*> cpu_list_;

  // System agents map
  std::map<hsa_agent_handle_t, const AgentInfo*> agent_map_;

  // AqlProfile API table
  aqlprofile_pfn_t aqlprofile_api_;

  // Loader API table
  hsa_ven_amd_loader_1_00_pfn_t loader_api_;
};

}  // namespace util
}  // namespace rocprofiler

#endif  // SRC_UTIL_HSA_RSRC_FACTORY_H_
