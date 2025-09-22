#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <cassert>

#include <iostream>
#include <vector>
#include <string>

#include "hsa.h"
#include "hsa_rsrc_factory.hpp"
#include "hsa_ext_finalize.h"
#include "tools/inc/hsa_ext_profiler.h"
#include "HSAILAmdExt.h"

#include "common.hpp"

using namespace std;

// Provide access to command line arguments passed in by user
uint32_t hsa_cmdline_arg_cnt;
char **hsa_cmdline_arg_list;

// Callback function to find and bind kernarg region of an agent
static hsa_status_t find_memregions(hsa_region_t region, void *data) {

  hsa_region_global_flag_t flags;
  hsa_region_segment_t segment_id;

  hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment_id);
  if (segment_id != HSA_REGION_SEGMENT_GLOBAL) {
    return HSA_STATUS_SUCCESS;
  }

  AgentInfo *agent_info = (AgentInfo *)data;
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
static hsa_status_t get_hsa_agents(hsa_agent_t agent, void *data) {

  // Copy handle of agent and increment number of agents reported
  HsaRsrcFactory *rsrcFactory = reinterpret_cast<HsaRsrcFactory *>(data);

  // Determine if device is a Gpu agent
  hsa_status_t status;
  hsa_device_type_t type;
  status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
  if (type == HSA_DEVICE_TYPE_DSP) {
    return HSA_STATUS_SUCCESS;
  }

  if (type == HSA_DEVICE_TYPE_CPU) {
    AgentInfo *agent_info = reinterpret_cast<AgentInfo *>(malloc(sizeof(AgentInfo)));
    agent_info->dev_id = agent;
    agent_info->dev_type = HSA_DEVICE_TYPE_CPU;
    rsrcFactory->AddAgentInfo(agent_info, false);
    return HSA_STATUS_SUCCESS;
  }
  
  // Device is a Gpu agent, build an instance of AgentInfo
  AgentInfo *agent_info = reinterpret_cast<AgentInfo *>(malloc(sizeof(AgentInfo)));
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
  hsa_agent_iterate_regions(agent, find_memregions, agent_info);

  // Save the instance of AgentInfo
  rsrcFactory->AddAgentInfo(agent_info, true);
  return HSA_STATUS_SUCCESS;
}

// Definitions for Static Data members of the class
char* HsaRsrcFactory::brig_path_ = NULL;
uint32_t HsaRsrcFactory::num_cus_;
uint32_t HsaRsrcFactory::num_waves_;
uint32_t HsaRsrcFactory::num_workitems_;
uint32_t HsaRsrcFactory::kernel_loop_count_;
bool HsaRsrcFactory::print_debug_info_ = false;

char* HsaRsrcFactory::num_cus_key_ = "num_cus";
char* HsaRsrcFactory::brig_path_key_ = "brig_path";
char* HsaRsrcFactory::num_waves_key_ = "waves_per_cu";
char* HsaRsrcFactory::num_workitems_key_ = "workitems_per_wave";
char* HsaRsrcFactory::print_debug_key_ = "print_debug";
char* HsaRsrcFactory::kernel_loop_count_key_ = "kernel_loop_count";

// Constructor of the class
HsaRsrcFactory::HsaRsrcFactory( ) {

  // Initialize the Hsa Runtime
  hsa_status_t status = hsa_init();
  assert(status == HSA_STATUS_SUCCESS);

  // Discover the set of Gpu devices available on the platform
  status = hsa_iterate_agents(get_hsa_agents, this);
  check("Error Calling hsa_iterate_agents", status);

  // Process command line arguments
  ProcessCmdline( );
}

// Destructor of the class
HsaRsrcFactory::~HsaRsrcFactory( ) {

}

// Get the count of Hsa Gpu Agents available on the platform
//
// @return uint32_t Number of Gpu agents on platform
//
uint32_t HsaRsrcFactory::GetCountOfGpuAgents( ) {
  return uint32_t(gpu_list_.size());
}

// Get the count of Hsa Cpu Agents available on the platform
//
// @return uint32_t Number of Cpu agents on platform
//
uint32_t HsaRsrcFactory::GetCountOfCpuAgents( ) {
  return uint32_t(cpu_list_.size());
}

// Get the AgentInfo handle of a Gpu device
//
// @param idx Gpu Agent at specified index
//
// @param agent_info Output parameter updated with AgentInfo
//
// @return bool true if successful, false otherwise
//
bool HsaRsrcFactory::GetGpuAgentInfo(uint32_t idx, AgentInfo **agent_info) {

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
bool HsaRsrcFactory::GetCpuAgentInfo(uint32_t idx, AgentInfo **agent_info) {

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
bool HsaRsrcFactory::CreateQueue(AgentInfo *agent_info,
                                 uint32_t num_pkts, hsa_queue_t **queue) {

  hsa_status_t status;

  // Code to create a Profile Queue object
  if (num_pkts == UINT32_MAX) {
    status = hsa_ext_tools_queue_create_profiled(agent_info->dev_id,
                                  512, HSA_QUEUE_TYPE_SINGLE, NULL,
                                  NULL, UINT32_MAX, UINT32_MAX, queue);
    return (status == HSA_STATUS_SUCCESS);
  }

  status = hsa_queue_create(agent_info->dev_id, num_pkts,
                            HSA_QUEUE_TYPE_MULTI, NULL, NULL,
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
bool HsaRsrcFactory::CreateSignal(uint32_t value, hsa_signal_t *signal) {

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
uint8_t* HsaRsrcFactory::AllocateLocalMemory(AgentInfo *agent_info, size_t size) {

  hsa_status_t status;
  uint8_t *buffer = NULL;

  // Allocate in local memory only if it is available
  if (agent_info->coarse_region.handle != 0) {
    std::cout << "Allocating in local memory" << std::endl;
    status = hsa_memory_allocate(agent_info->coarse_region, size, (void **)&buffer);
    if (status == HSA_STATUS_SUCCESS) {
      status = hsa_memory_assign_agent(buffer, agent_info->dev_id, HSA_ACCESS_PERMISSION_RW);
      return (status == HSA_STATUS_SUCCESS) ? buffer : NULL;
    }
    return NULL;
  }

  // Allocate in system memory if local memory is not available
  std::cout << "Allocating in system memory" << std::endl;
  status = hsa_memory_allocate(agent_info->kernarg_region, size, (void **)&buffer);
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
uint8_t* HsaRsrcFactory::AllocateSysMemory(AgentInfo *agent_info, size_t size) {

  hsa_status_t status;
  uint8_t *buffer = NULL;
  status = hsa_memory_allocate(agent_info->kernarg_region, size, (void **)&buffer);
  return (status == HSA_STATUS_SUCCESS) ? buffer : NULL;
}

bool HsaRsrcFactory::TransferData(uint8_t *dest_buff, uint8_t *src_buff,
                                  uint32_t length, bool host_to_dev) {

  hsa_status_t status;
  status = hsa_memory_copy(dest_buff, src_buff, length);
  return (status == HSA_STATUS_SUCCESS);

}

// Fake method for compilation steps only
uint8_t* HsaRsrcFactory::AllocateMemory(AgentInfo *agent_info, size_t size) {

  hsa_status_t status;
  uint8_t *buffer = NULL;
  status = hsa_memory_allocate(agent_info->kernarg_region, size, (void **)&buffer);
  return (status == HSA_STATUS_SUCCESS) ? buffer : NULL;
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
bool HsaRsrcFactory::LoadAndFinalize(AgentInfo *agent_info,
                                     const char *brig_path, char *kernel_name,
                                     hsa_executable_symbol_t *code_desc) {

  hsa_status_t status;
  // Load BRIG, encapsulated in an ELF container, into a BRIG module.
  /*
  status_t build_err;
  hsa_ext_brig_module_t *brig_obj;
  build_err = (status_t)create_brig_module_from_brig_file(brig_path, &brig_obj);
  check_build("Error in creating the brig module from brig file", build_err);

  // Determine the Brig module has the kernel symbol
  hsa_status_t status;
  hsa_ext_brig_code_section_offset32_t kernel_symbol;
  status = hsa_find_symbol_offset(brig_obj, kernel_name, &kernel_symbol);
  check("Error in Finding the Symbol Offset for the Kernel", status);
  */

  amd::hsail::registerExtensions();

  // Copy handle of Brig object
  hsa_ext_module_t brig_module_v3;
  if (!tool.assembleFromFile(brig_path)) {
    std::cout << tool.output();
    return false;
  }
  brig_module_v3 = tool.brigModule();
  
  // Create hsail program.
  hsa_ext_program_t hsailProgram;
  status = hsa_ext_program_create(HSA_MACHINE_MODEL_LARGE,
                                  agent_info->profile,
                                  HSA_DEFAULT_FLOAT_ROUNDING_MODE_ZERO,
                                  NULL, &hsailProgram);
  check("Error in creating program object", status);

  // Add hsail module.
  status = hsa_ext_program_add_module(hsailProgram, brig_module_v3);
  check("Error in adding module to program object", status);

  // Finalize hsail program.
  hsa_isa_t isa = {0};
  status = hsa_agent_get_info(agent_info->dev_id, HSA_AGENT_INFO_ISA, &isa);
  check("Error in getting Id of Isa supported by agent", status);

  hsa_ext_control_directives_t control_directives;
  memset(&control_directives, 0, sizeof(hsa_ext_control_directives_t));

  hsa_code_object_t code_object;
  status = hsa_ext_program_finalize(hsailProgram,
                                           isa,
                                           0,
                                           control_directives,
                                           NULL, //"-g -O0 -dump-isa",
                                           HSA_CODE_OBJECT_TYPE_PROGRAM,
                                           &code_object);
  check("Error in finalizing program object", status);

  //status = hsa_ext_program_destroy(hsailProgram);
  //check("Error in destroying program object", status);

  // Create executable.
  hsa_executable_t hsaExecutable;
  status = hsa_executable_create(agent_info->profile,
                                 HSA_EXECUTABLE_STATE_UNFROZEN,
                                 "", &hsaExecutable);
  check("Error in creating executable object", status);

  // Load code object.
  status = hsa_executable_load_code_object(hsaExecutable, agent_info->dev_id, code_object, "");
  check("Error in loading executable object", status);

  // Freeze executable.
  status = hsa_executable_freeze(hsaExecutable, "");
  check("Error in freezing executable object", status);

  // Get symbol handle.
  hsa_executable_symbol_t kernelSymbol;
  status = hsa_executable_get_symbol(hsaExecutable, NULL,
                             kernel_name, agent_info->dev_id, 0, &kernelSymbol);
  
  // Update output parameter
  *code_desc = kernelSymbol;
  return true;

  /**

  // Create Hsa Program
  hsa_ext_program_handle_t program;
  status = hsa_ext_program_create(&agent_info->dev_id, 1,
                                  HSA_EXT_BRIG_MACHINE_LARGE,
                                  HSA_EXT_BRIG_PROFILE_FULL, &program);
  check("Error in Creating Hsa Program", status);

  // Add the BRIG module to hsa program.
  hsa_ext_brig_module_handle_t brig_handle;
  status = hsa_ext_add_module(program, brig_obj, &brig_handle);
  check("Error in Adding Brig Module to the Program", status);

  // Construct finalization request list.
  hsa_ext_finalization_request_t finalize_request;
  finalize_request.module = brig_handle;
  finalize_request.symbol = kernel_symbol;
  finalize_request.program_call_convention = 0;

  // Finalize the Hsa Program.
  status = hsa_ext_finalize_program(program, agent_info->dev_id,
                                    1, &finalize_request, NULL, NULL, 0, NULL, 0);
  check("Error in Finalizing the Hsa Program", status);

  // Destroy the brig module. The program was successfully created the kernel
  // symbol was found and the program was finalized, so it is no longer needed.
  destroy_brig_module(brig_obj);

  // Get the hsa code descriptor address.
  status = hsa_ext_query_kernel_descriptor_address(program, brig_handle, kernel_symbol, code_desc);
  check("Error Querying the Kernel Descriptor Address", status);

  return true;
  **/
}

// Add an instance of AgentInfo representing a Hsa Gpu agent
void HsaRsrcFactory::AddAgentInfo(AgentInfo *agent_info, bool gpu) {
  
  // Add input to Gpu list
  if (gpu) {
    gpu_list_.push_back(agent_info);
    return;
  }

  // Add input to Cpu list
  cpu_list_.push_back(agent_info);
}

// Print the various fields of Hsa Gpu Agents
bool HsaRsrcFactory::PrintGpuAgents( ) {

  AgentInfo *agent_info;
  int size = uint32_t(gpu_list_.size());
  for (int idx = 0; idx < size; idx++) {
    agent_info = gpu_list_[idx];
    std::cout << std::endl;
    std::cout << "Hsa Gpu Agent Id: " << agent_info->dev_id.handle << std::endl;
    std::cout << "Hsa Gpu Agent Name: " << agent_info->name << std::endl;
    std::cout << "Hsa Gpu Agent Max Wave Size: " << agent_info->max_wave_size << std::endl;
    std::cout << "Hsa Gpu Agent Max Queue Size: " << agent_info->max_queue_size << std::endl;
    std::cout << "Hsa Gpu Agent Kernarg Region Id: " << agent_info->coarse_region.handle << std::endl;
    std::cout << std::endl;
  }
  return true;
}

// Returns the file path where brig files is located. Value is
// available only after an instance has been built.
char* HsaRsrcFactory::GetBrigPath( ) {
  return HsaRsrcFactory::brig_path_;
}

// Returns the number of compute units present on platform
// Value is available only after an instance has been built.
uint32_t HsaRsrcFactory::GetNumOfCUs( ) {
  return HsaRsrcFactory::num_cus_;
}

// Returns the maximum number of waves that can be launched
// per compute unit. The actual number that can be launched
// is affected by resource availability
//
// Value is available only after an instance has been built.
uint32_t HsaRsrcFactory::GetNumOfWavesPerCU( ) {
  return HsaRsrcFactory::num_waves_;
}

// Returns the number of work-items that can execute per wave
// Value is available only after an instance has been built.
uint32_t HsaRsrcFactory::GetNumOfWorkItemsPerWave( ) {
  return HsaRsrcFactory::num_workitems_;
}

// Returns the number of times kernel loop body should execute.
// Value is available only after an instance has been built.
uint32_t HsaRsrcFactory::GetKernelLoopCount() {
  return HsaRsrcFactory::kernel_loop_count_;
}

// Returns boolean flag to indicate if debug info should be printed
// Value is available only after an instance has been built.
uint32_t HsaRsrcFactory::GetPrintDebugInfo() {
  return HsaRsrcFactory::print_debug_info_;
}

// Process command line arguments. The method will capture
// various user command line parameters for tests to use
void HsaRsrcFactory::ProcessCmdline( ) {
 
  // Command line arguments are given
  uint32_t idx;
  uint32_t arg_idx;
  for (idx = 1; idx < hsa_cmdline_arg_cnt; idx += 2) {
    arg_idx = GetArgIndex((char *)hsa_cmdline_arg_list[idx]);
    switch(arg_idx) {
      case 0:
        HsaRsrcFactory::brig_path_ = hsa_cmdline_arg_list[idx + 1];
        break;
      case 1:
        HsaRsrcFactory::num_cus_ = atoi(hsa_cmdline_arg_list[idx + 1]);
        break;
      case 2:
        HsaRsrcFactory::num_waves_ = atoi(hsa_cmdline_arg_list[idx + 1]);
        break;
      case 3:
        HsaRsrcFactory::num_workitems_ = atoi(hsa_cmdline_arg_list[idx + 1]);
        break;
      case 4:
        HsaRsrcFactory::kernel_loop_count_ = atoi(hsa_cmdline_arg_list[idx + 1]);
        break;
      case 5:
        HsaRsrcFactory::print_debug_info_ = true;
        break;
    }
  }

}

uint32_t HsaRsrcFactory::GetArgIndex(char *arg_value ) {

  // Map Brig file path to index zero
  if (!strcmp(HsaRsrcFactory::brig_path_key_, arg_value)) {
      return 0;
  }

  // Map Number of Compute Units to index one
  if (!strcmp(HsaRsrcFactory::num_cus_key_, arg_value)) {
      return 1;
  }

  // Map Number of Waves per CU to index two
  if (!strcmp(HsaRsrcFactory::num_waves_key_, arg_value)) {
      return 2;
  }

  // Map Number of Workitems per Wave to index three
  if (!strcmp(HsaRsrcFactory::num_workitems_key_, arg_value)) {
      return 3;
  }

  // Map Kernel Loop Count to index four
  if (!strcmp(HsaRsrcFactory::kernel_loop_count_key_, arg_value)) {
      return 4;
  }

  // Map print debug info parameter
  if (!strcmp(HsaRsrcFactory::print_debug_key_, arg_value)) {
      return 5;
  }
  
  return 108;

}

void HsaRsrcFactory::PrintHelpMsg( ) {

  std::cout << "Key for passing Brig filepath: " << HsaRsrcFactory::brig_path_key_ << std::endl;
  std::cout << "Key for passing Number of Compute Units: " << HsaRsrcFactory::num_cus_key_ << std::endl;
  std::cout << "Key for passing Number of Waves per CU: " << HsaRsrcFactory::num_waves_key_ << std::endl;
  std::cout << "Key for passing Number of Workitems per Wave: " << HsaRsrcFactory::num_workitems_key_ << std::endl;
  std::cout << "Key for passing Kernel Loop Count: " << HsaRsrcFactory::kernel_loop_count_key_ << std::endl;

}
