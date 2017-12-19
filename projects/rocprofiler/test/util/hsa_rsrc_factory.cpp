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

#include "util/hsa_rsrc_factory.h"

#include <dlfcn.h>
#include <hsa.h>
#include <hsa_ext_finalize.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Callback function to find and bind kernarg region of an agent
static hsa_status_t FindMemRegionsCallback(hsa_region_t region, void* data) {
  hsa_region_global_flag_t flags;
  hsa_region_segment_t segment_id;

  hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment_id);
  if (segment_id != HSA_REGION_SEGMENT_GLOBAL) {
    return HSA_STATUS_SUCCESS;
  }

  AgentInfo* agent_info = (AgentInfo*)data;
  hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
  if (flags & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED) {
    agent_info->coarse_region = region;
  }

  if (flags & HSA_REGION_GLOBAL_FLAG_KERNARG) {
    agent_info->kernarg_region = region;
  }

  return HSA_STATUS_SUCCESS;
}

// Callback function to get the number of agents
static hsa_status_t GetHsaAgentsCallback(hsa_agent_t agent, void* data) {
  // Copy handle of agent and increment number of agents reported
  HsaRsrcFactory* rsrcFactory = reinterpret_cast<HsaRsrcFactory*>(data);

  // Determine if device is a Gpu agent
  hsa_status_t status;
  hsa_device_type_t type;
  status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
  CHECK_STATUS("Error Calling hsa_agent_get_info", status);
  if (type == HSA_DEVICE_TYPE_DSP) {
    return HSA_STATUS_SUCCESS;
  }

  if (type == HSA_DEVICE_TYPE_CPU) {
    AgentInfo* agent_info = reinterpret_cast<AgentInfo*>(malloc(sizeof(AgentInfo)));
    agent_info->dev_id = agent;
    agent_info->dev_type = HSA_DEVICE_TYPE_CPU;
    rsrcFactory->AddAgentInfo(agent_info, false);
    return HSA_STATUS_SUCCESS;
  }

  // Device is a Gpu agent, build an instance of AgentInfo
  AgentInfo* agent_info = reinterpret_cast<AgentInfo*>(malloc(sizeof(AgentInfo)));
  agent_info->dev_id = agent;
  agent_info->dev_type = HSA_DEVICE_TYPE_GPU;
  hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, agent_info->name);
  agent_info->max_wave_size = 0;
  hsa_agent_get_info(agent, HSA_AGENT_INFO_WAVEFRONT_SIZE, &agent_info->max_wave_size);
  agent_info->max_queue_size = 0;
  hsa_agent_get_info(agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &agent_info->max_queue_size);
  agent_info->profile = hsa_profile_t(108);
  hsa_agent_get_info(agent, HSA_AGENT_INFO_PROFILE, &agent_info->profile);

  // Initialize memory regions to zero
  agent_info->kernarg_region.handle = 0;
  agent_info->coarse_region.handle = 0;

  // Find and Bind Memory regions of the Gpu agent
  hsa_agent_iterate_regions(agent, FindMemRegionsCallback, agent_info);

  // Save the instance of AgentInfo
  rsrcFactory->AddAgentInfo(agent_info, true);

  return HSA_STATUS_SUCCESS;
}

// Constructor of the class
HsaRsrcFactory::HsaRsrcFactory() {
  // Initialize the Hsa Runtime
  hsa_status_t status = hsa_init();
  CHECK_STATUS("Error in hsa_init", status);

  // Discover the set of Gpu devices available on the platform
  status = hsa_iterate_agents(GetHsaAgentsCallback, this);
  CHECK_STATUS("Error Calling hsa_iterate_agents", status);

  // Get AqlProfile API table
  aqlprofile_api_ = {0};
  status = hsa_system_get_extension_table(HSA_EXTENSION_AMD_AQLPROFILE, 1, 0, &aqlprofile_api_);
  if (status != HSA_STATUS_SUCCESS) status = LoadAqlProfileLib(&aqlprofile_api_);
  CHECK_STATUS("aqlprofile API table load failed", status);
}

// Destructor of the class
HsaRsrcFactory::~HsaRsrcFactory() {
  hsa_status_t status = hsa_shut_down();
  CHECK_STATUS("Error in hsa_shut_down", status);
}

hsa_status_t HsaRsrcFactory::LoadAqlProfileLib(aqlprofile_pfn_t* api) {
    void* handle = dlopen(kAqlProfileLib, RTLD_NOW);
    if (handle == NULL) {
      fprintf(stderr, "Loading '%s' failed, %s\n", kAqlProfileLib, dlerror());
      return HSA_STATUS_ERROR;
    }
    dlerror(); /* Clear any existing error */

    api->hsa_ven_amd_aqlprofile_error_string =
      (decltype(::hsa_ven_amd_aqlprofile_error_string)*)
        dlsym(handle, "hsa_ven_amd_aqlprofile_error_string");
    api->hsa_ven_amd_aqlprofile_validate_event =
      (decltype(::hsa_ven_amd_aqlprofile_validate_event)*)
        dlsym(handle, "hsa_ven_amd_aqlprofile_validate_event");
    api->hsa_ven_amd_aqlprofile_start =
      (decltype(::hsa_ven_amd_aqlprofile_start)*)
        dlsym(handle, "hsa_ven_amd_aqlprofile_start");
    api->hsa_ven_amd_aqlprofile_stop =
      (decltype(::hsa_ven_amd_aqlprofile_stop)*)
        dlsym(handle, "hsa_ven_amd_aqlprofile_stop");
    api->hsa_ven_amd_aqlprofile_legacy_get_pm4 =
      (decltype(::hsa_ven_amd_aqlprofile_legacy_get_pm4)*)
        dlsym(handle, "hsa_ven_amd_aqlprofile_legacy_get_pm4");
    api->hsa_ven_amd_aqlprofile_get_info =
      (decltype(::hsa_ven_amd_aqlprofile_get_info)*)
        dlsym(handle, "hsa_ven_amd_aqlprofile_get_info");
    api->hsa_ven_amd_aqlprofile_iterate_data =
      (decltype(::hsa_ven_amd_aqlprofile_iterate_data)*)
        dlsym(handle, "hsa_ven_amd_aqlprofile_iterate_data");

  return HSA_STATUS_SUCCESS;
}

// Get the count of Hsa Gpu Agents available on the platform
//
// @return uint32_t Number of Gpu agents on platform
//
uint32_t HsaRsrcFactory::GetCountOfGpuAgents() { return uint32_t(gpu_list_.size()); }

// Get the count of Hsa Cpu Agents available on the platform
//
// @return uint32_t Number of Cpu agents on platform
//
uint32_t HsaRsrcFactory::GetCountOfCpuAgents() { return uint32_t(cpu_list_.size()); }

// Get the AgentInfo handle of a Gpu device
//
// @param idx Gpu Agent at specified index
//
// @param agent_info Output parameter updated with AgentInfo
//
// @return bool true if successful, false otherwise
//
bool HsaRsrcFactory::GetGpuAgentInfo(uint32_t idx, AgentInfo** agent_info) {
  // Determine if request is valid
  uint32_t size = uint32_t(gpu_list_.size());
  if (idx >= size) {
    return false;
  }

  // Copy AgentInfo from specified index
  *agent_info = gpu_list_[idx];
  return true;
}

// Get the AgentInfo handle of a Cpu device
//
// @param idx Cpu Agent at specified index
//
// @param agent_info Output parameter updated with AgentInfo
//
// @return bool true if successful, false otherwise
//
bool HsaRsrcFactory::GetCpuAgentInfo(uint32_t idx, AgentInfo** agent_info) {
  // Determine if request is valid
  uint32_t size = uint32_t(cpu_list_.size());
  if (idx >= size) {
    return false;
  }

  // Copy AgentInfo from specified index
  *agent_info = cpu_list_[idx];
  return true;
}

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
bool HsaRsrcFactory::CreateQueue(AgentInfo* agent_info, uint32_t num_pkts, hsa_queue_t** queue) {
  hsa_status_t status;
  status = hsa_queue_create(agent_info->dev_id, num_pkts, HSA_QUEUE_TYPE_MULTI, NULL, NULL,
                            UINT32_MAX, UINT32_MAX, queue);
  return (status == HSA_STATUS_SUCCESS);
}

// Create a Signal object and return its handle.
//
// @param value Initial value of signal object
//
// @param signal Output parameter updated with handle of signal object
//
// @return bool true if successful, false otherwise
//
bool HsaRsrcFactory::CreateSignal(uint32_t value, hsa_signal_t* signal) {
  hsa_status_t status;
  status = hsa_signal_create(value, 0, NULL, signal);
  return (status == HSA_STATUS_SUCCESS);
}

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
uint8_t* HsaRsrcFactory::AllocateLocalMemory(const AgentInfo* agent_info, size_t size) {
  hsa_status_t status;
  uint8_t* buffer = NULL;
  size = (size + MEM_PAGE_MASK) & ~MEM_PAGE_MASK;

  if (agent_info->coarse_region.handle != 0) {
    // Allocate in local memory if it is available
    status = hsa_memory_allocate(agent_info->coarse_region, size, (void**)&buffer);
    if (status == HSA_STATUS_SUCCESS) {
      status = hsa_memory_assign_agent(buffer, agent_info->dev_id, HSA_ACCESS_PERMISSION_RW);
    }
  } else {
    // Allocate in system memory if local memory is not available
    status = hsa_memory_allocate(agent_info->kernarg_region, size, (void**)&buffer);
  }

  return (status == HSA_STATUS_SUCCESS) ? buffer : NULL;
}

// Allocate memory tp pass kernel parameters.
//
// @param agent_info Agent from whose memory region to allocate
//
// @param size Size of memory in terms of bytes
//
// @return uint8_t* Pointer to buffer, null if allocation fails.
//
uint8_t* HsaRsrcFactory::AllocateSysMemory(const AgentInfo* agent_info, size_t size) {
  hsa_status_t status;
  size = (size + MEM_PAGE_MASK) & ~MEM_PAGE_MASK;

  uint8_t* buffer = NULL;
  status = hsa_memory_allocate(agent_info->kernarg_region, size, (void**)&buffer);
  return (status == HSA_STATUS_SUCCESS) ? buffer : NULL;
}

// Transfer data method
bool HsaRsrcFactory::TransferData(void* dest_buff, void* src_buff, uint32_t length,
                                  bool host_to_dev) {
  hsa_status_t status;
  status = hsa_memory_copy(dest_buff, src_buff, length);
  return (status == HSA_STATUS_SUCCESS);
}

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
// @return bool true if successful, false otherwise
//
bool HsaRsrcFactory::LoadAndFinalize(AgentInfo* agent_info, const char* brig_path,
                                     char* kernel_name, hsa_executable_symbol_t* code_desc) {
  // Finalize the Hsail object into code object
  hsa_status_t status;
  hsa_code_object_t code_object;

  // Build the code object filename
  std::string filename(brig_path);
  std::clog << "Code object filename: " << filename << std::endl;

  // Open the file containing code object
  std::ifstream codeStream(filename.c_str(), std::ios::binary | std::ios::ate);
  if (!codeStream) {
    std::cerr << "Error: failed to load " << filename << std::endl;
    assert(false);
    return false;
  }

  // Allocate memory to read in code object from file
  size_t size = std::string::size_type(codeStream.tellg());
  char* codeBuff = (char*)AllocateSysMemory(agent_info, size);
  if (!codeBuff) {
    std::cerr << "Error: failed to allocate memory for code object." << std::endl;
    assert(false);
    return false;
  }

  // Read the code object into allocated memory
  codeStream.seekg(0, std::ios::beg);
  std::copy(std::istreambuf_iterator<char>(codeStream), std::istreambuf_iterator<char>(), codeBuff);

  // De-Serialize the code object that has been read into memory
  status = hsa_code_object_deserialize(codeBuff, size, NULL, &code_object);
  if (status != HSA_STATUS_SUCCESS) {
    std::cerr << "Failed to deserialize code object" << std::endl;
    return false;
  }

  // Create executable.
  hsa_executable_t hsaExecutable;
  status =
      hsa_executable_create(HSA_PROFILE_FULL, HSA_EXECUTABLE_STATE_UNFROZEN, "", &hsaExecutable);
  CHECK_STATUS("Error in creating executable object", status);

  // Load code object.
  status = hsa_executable_load_code_object(hsaExecutable, agent_info->dev_id, code_object, "");
  CHECK_STATUS("Error in loading executable object", status);

  // Freeze executable.
  status = hsa_executable_freeze(hsaExecutable, "");
  CHECK_STATUS("Error in freezing executable object", status);

  // Get symbol handle.
  hsa_executable_symbol_t kernelSymbol;
  status = hsa_executable_get_symbol(hsaExecutable, NULL, kernel_name, agent_info->dev_id, 0,
                                     &kernelSymbol);
  CHECK_STATUS("Error in looking up kernel symbol", status);

  // Update output parameter
  *code_desc = kernelSymbol;
  return true;
}

// Add an instance of AgentInfo representing a Hsa Gpu agent
void HsaRsrcFactory::AddAgentInfo(AgentInfo* agent_info, bool gpu) {
  // Add input to Gpu list
  if (gpu) {
    gpu_list_.push_back(agent_info);
    return;
  }

  // Add input to Cpu list
  cpu_list_.push_back(agent_info);
}

// Print the various fields of Hsa Gpu Agents
bool HsaRsrcFactory::PrintGpuAgents(const std::string& header) {
  std::clog << header << " :" << std::endl;

  AgentInfo* agent_info;
  int size = uint32_t(gpu_list_.size());
  for (int idx = 0; idx < size; idx++) {
    agent_info = gpu_list_[idx];

    std::clog << "> agent[" << idx << "] :" << std::endl;
    std::clog << ">> Name : " << agent_info->name << std::endl;
    std::clog << ">> Max Wave Size : " << agent_info->max_wave_size << std::endl;
    std::clog << ">> Max Queue Size : " << agent_info->max_queue_size << std::endl;
    std::clog << ">> Kernarg Region Id : " << agent_info->coarse_region.handle << std::endl;
  }
  return true;
}

HsaRsrcFactory* HsaRsrcFactory::instance_ = NULL;
HsaRsrcFactory::mutex_t HsaRsrcFactory::mutex_;
