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

#ifndef TEST_UTIL_HSA_RSRC_FACTORY_H_
#define TEST_UTIL_HSA_RSRC_FACTORY_H_

#include <hsa.h>
#include <hsa_ext_finalize.h>
#include <hsa_ven_amd_aqlprofile.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <iostream>
#include <mutex>
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

static const unsigned MEM_PAGE_BYTES = 0x1000;
static const unsigned MEM_PAGE_MASK = MEM_PAGE_BYTES - 1;

// Encapsulates information about a Hsa Agent such as its
// handle, name, max queue size, max wavefront size, etc.
struct AgentInfo {
  // Handle of Agent
  hsa_agent_t dev_id;

  // Agent type - Cpu = 0, Gpu = 1 or Dsp = 2
  uint32_t dev_type;

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
};

class HsaRsrcFactory {
 public:
  typedef std::recursive_mutex mutex_t;

  static HsaRsrcFactory* Create() {
    std::lock_guard<mutex_t> lck(mutex_);
    if (HsaRsrcFactory::instance_ == NULL) {
      HsaRsrcFactory::instance_ = new HsaRsrcFactory();
    }
    return instance_;
  }

  static void Destroy() {
    std::lock_guard<mutex_t> lck(mutex_);
    if (instance_) delete instance_;
    instance_ = NULL;
  }

  static HsaRsrcFactory& Instance() {
    hsa_status_t status = (instance_ != NULL) ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR;
    CHECK_STATUS("HsaRsrcFactory::Instance()", status);
    return *instance_;
  }

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
  bool GetGpuAgentInfo(uint32_t idx, AgentInfo** agent_info);

  // Get the AgentInfo handle of a Cpu device
  //
  // @param idx Cpu Agent at specified index
  //
  // @param agent_info Output parameter updated with AgentInfo
  //
  // @return bool true if successful, false otherwise
  //
  bool GetCpuAgentInfo(uint32_t idx, AgentInfo** agent_info);

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
  bool CreateQueue(AgentInfo* agent_info, uint32_t num_pkts, hsa_queue_t** queue);

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

  // Transfer data method
  bool TransferData(void* dest_buff, void* src_buff, uint32_t length, bool host_to_dev);

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
  // @return code buffer, non NULL if successful, NULL otherwise
  //
  void* LoadAndFinalize(AgentInfo* agent_info, const char* brig_path, const char* kernel_name,
                        hsa_executable_t* hsa_exec, hsa_executable_symbol_t* code_desc);

  // Add an instance of AgentInfo representing a Hsa Gpu agent
  void AddAgentInfo(AgentInfo* agent_info, bool gpu);

  // Print the various fields of Hsa Gpu Agents
  bool PrintGpuAgents(const std::string& header);

  // Submit AQL packet to given queue
  static uint64_t Submit(hsa_queue_t* queue, void* packet);

  // Return AqlProfile API table
  typedef hsa_ven_amd_aqlprofile_1_00_pfn_t aqlprofile_pfn_t;
  const aqlprofile_pfn_t* AqlProfileApi() const { return &aqlprofile_api_; }

 private:
  // Load AQL profile HSA extension library directly
  static hsa_status_t LoadAqlProfileLib(aqlprofile_pfn_t* api);

  // Constructor of the class. Will initialize the Hsa Runtime and
  // query the system topology to get the list of Cpu and Gpu devices
  HsaRsrcFactory();

  // Destructor of the class
  ~HsaRsrcFactory();

  static HsaRsrcFactory* instance_;
  static mutex_t mutex_;

  // Used to maintain a list of Hsa Gpu Agent Info
  std::vector<AgentInfo*> gpu_list_;

  // Used to maintain a list of Hsa Cpu Agent Info
  std::vector<AgentInfo*> cpu_list_;

  // AqlProfile API table
  aqlprofile_pfn_t aqlprofile_api_;
};

#endif  // TEST_UTIL_HSA_RSRC_FACTORY_H_
