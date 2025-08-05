// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.


#ifndef SRC_UTIL_HSA_RSRC_FACTORY_H_
#define SRC_UTIL_HSA_RSRC_FACTORY_H_

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ext_finalize.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#define HSA_ARGUMENT_ALIGN_BYTES 16
#define HSA_QUEUE_ALIGN_BYTES 64
#define HSA_PACKET_ALIGN_BYTES 64

#define CHECK_STATUS(msg, status)                               \
  do {                                                          \
    if ((status) != HSA_STATUS_SUCCESS) {                       \
      const char* emsg = 0;                                     \
      hsa_status_string(status, &emsg);                         \
      printf("%s: %s\n", msg, emsg ? emsg : "<unknown error>"); \
      abort();                                                  \
    }                                                           \
  } while (0)

#define CHECK_ITER_STATUS(msg, status)                          \
  do {                                                          \
    if ((status) != HSA_STATUS_INFO_BREAK) {                    \
      const char* emsg = 0;                                     \
      hsa_status_string(status, &emsg);                         \
      printf("%s: %s\n", msg, emsg ? emsg : "<unknown error>"); \
      abort();                                                  \
    }                                                           \
  } while (0)

static const size_t MEM_PAGE_BYTES = 0x1000;
static const size_t MEM_PAGE_MASK = MEM_PAGE_BYTES - 1;
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

  // CPU/GPU/kern-arg memory pools
  hsa_amd_memory_pool_t cpu_pool;
  hsa_amd_memory_pool_t gpu_pool;
  hsa_amd_memory_pool_t kern_arg_pool;

  // The number of compute unit available in the agent.
  uint32_t cu_num;

  // Maximum number of waves possible in a Compute Unit.
  uint32_t waves_per_cu;

  // Number of SIMD's per compute unit CU
  uint32_t simds_per_cu;

  // Number of Shader Engines (SE) in Gpu
  uint32_t se_num;

  // Number of MI3000 XCC
  uint32_t xcc_num{1};

  // pcie domain
  uint32_t domain{0};

  // BDF ID (Bus/Device/Function) of the device
  uint32_t bdf_id{0};

  // Number of Shader Arrays Per Shader Engines in Gpu
  uint32_t shader_arrays_per_se;
};

// HSA timer class
// Provides current HSA timestampa and system-clock/ns conversion API
class HsaTimer {
 public:
  typedef uint64_t timestamp_t;
  static const timestamp_t TIMESTAMP_MAX = UINT64_MAX;
  typedef long double freq_t;

  HsaTimer() {
    timestamp_t sysclock_hz = 0;
    hsa_status_t status = hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &sysclock_hz);
    CHECK_STATUS("hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY)", status);
    sysclock_factor_ = (freq_t)1000000000 / (freq_t)sysclock_hz;
  }

  // Methids for system-clock/ns conversion
  timestamp_t sysclock_to_ns(const timestamp_t& sysclock) const {
    return timestamp_t((freq_t)sysclock * sysclock_factor_);
  }
  timestamp_t ns_to_sysclock(const timestamp_t& time) const {
    return timestamp_t((freq_t)time / sysclock_factor_);
  }

  // Return timestamp in 'ns'
  timestamp_t timestamp_ns() const {
    timestamp_t sysclock;
    hsa_status_t status = hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP, &sysclock);
    CHECK_STATUS("hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP)", status);
    return sysclock_to_ns(sysclock);
  }

 private:
  // Timestamp frequency factor
  freq_t sysclock_factor_;
};

class HsaRsrcFactory {
 public:
  static const size_t CMD_SLOT_SIZE_B = 0x40;
  typedef std::recursive_mutex mutex_t;
  typedef HsaTimer::timestamp_t timestamp_t;

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
  // @return uint32_t Number of Gpu agents on platform
  uint32_t GetCountOfGpuAgents();

  // Get the count of Hsa Cpu Agents available on the platform
  // @return uint32_t Number of Cpu agents on platform
  uint32_t GetCountOfCpuAgents();

  // Get the AgentInfo handle of a Gpu device
  // @param idx Gpu Agent at specified index
  // @param agent_info Output parameter updated with AgentInfo
  // @return bool true if successful, false otherwise
  bool GetGpuAgentInfo(uint32_t idx, const AgentInfo** agent_info);

  // Get the AgentInfo handle of a Cpu device
  // @param idx Cpu Agent at specified index
  // @param agent_info Output parameter updated with AgentInfo
  // @return bool true if successful, false otherwise
  bool GetCpuAgentInfo(uint32_t idx, const AgentInfo** agent_info);

  // Create a Queue object and return its handle. The queue object is expected
  // to support user requested number of Aql dispatch packets.
  // @param agent_info Gpu Agent on which to create a queue object
  // @param num_Pkts Number of packets to be held by queue
  // @param queue Output parameter updated with handle of queue object
  // @return bool true if successful, false otherwise
  bool CreateQueue(const AgentInfo* agent_info, uint32_t num_pkts, hsa_queue_t** queue);

  // Create a Signal object and return its handle.
  // @param value Initial value of signal object
  // @param signal Output parameter updated with handle of signal object
  // @return bool true if successful, false otherwise
  bool CreateSignal(uint32_t value, hsa_signal_t* signal);

  // Allocate local GPU memory
  // @param agent_info Agent from whose memory region to allocate
  // @param size Size of memory in terms of bytes
  // @return uint8_t* Pointer to buffer, null if allocation fails.
  uint8_t* AllocateLocalMemory(const AgentInfo* agent_info, size_t size);

  // Allocate memory tp pass kernel parameters
  // Memory is alocated accessible for all CPU agents and for GPU given by AgentInfo parameter.
  // @param agent_info Agent from whose memory region to allocate
  // @param size Size of memory in terms of bytes
  // @return uint8_t* Pointer to buffer, null if allocation fails.
  uint8_t* AllocateKernArgMemory(const AgentInfo* agent_info, size_t size);

  // Allocate system memory accessible from both CPU and GPU
  // Memory is alocated accessible to all CPU agents and AgentInfo parameter is ignored.
  // @param agent_info Agent from whose memory region to allocate
  // @param size Size of memory in terms of bytes
  // @return uint8_t* Pointer to buffer, null if allocation fails.
  uint8_t* AllocateSysMemory(const AgentInfo* agent_info, size_t size);

  // Allocate memory for command buffer.
  // @param agent_info Agent from whose memory region to allocate
  // @param size Size of memory in terms of bytes
  // @return uint8_t* Pointer to buffer, null if allocation fails.
  uint8_t* AllocateCmdMemory(const AgentInfo* agent_info, size_t size);

  // Wait signal
  void SignalWait(const hsa_signal_t& signal) const;

  // Wait signal with signal value restore
  void SignalWaitRestore(const hsa_signal_t& signal, const hsa_signal_value_t& signal_value) const;

  // Copy data from GPU to host memory
  bool Memcpy(const hsa_agent_t& agent, void* dst, const void* src, size_t size);
  bool Memcpy(const AgentInfo* agent_info, void* dst, const void* src, size_t size);

  // Memory free method
  static bool FreeMemory(void* ptr);

  // Loads an Assembled Brig file and Finalizes it into Device Isa
  // @param agent_info Gpu device for which to finalize
  // @param brig_path File path of the Assembled Brig file
  // @param kernel_name Name of the kernel to finalize
  // @param code_desc Handle of finalized Code Descriptor that could
  // be used to submit for execution
  // @return true if successful, false otherwise
  bool LoadAndFinalize(const AgentInfo* agent_info, const char* brig_path, const char* kernel_name,
                       hsa_executable_t* hsa_exec, hsa_executable_symbol_t* code_desc);

  // Print the various fields of Hsa Gpu Agents
  bool PrintGpuAgents(const std::string& header);

  // Submit AQL packet to given queue
  static uint64_t Submit(hsa_queue_t* queue, const void* packet);
  static uint64_t Submit(hsa_queue_t* queue, const void* packet, size_t size_bytes);

  // Return AqlProfile API table
  typedef hsa_ven_amd_aqlprofile_pfn_t aqlprofile_pfn_t;
  const aqlprofile_pfn_t* AqlProfileApi() const { return &aqlprofile_api_; }

  // Return Loader API table
  const hsa_ven_amd_loader_1_00_pfn_t* LoaderApi() const { return &loader_api_; }

  // Methods for system-clock/ns conversion and timestamp in 'ns'
  timestamp_t SysclockToNs(const timestamp_t& sysclock) const {
    return timer_->sysclock_to_ns(sysclock);
  }
  timestamp_t NsToSysclock(const timestamp_t& time) const { return timer_->ns_to_sysclock(time); }
  timestamp_t TimestampNs() const { return timer_->timestamp_ns(); }

  timestamp_t GetSysTimeout() const { return timeout_; }
  static timestamp_t GetTimeoutNs() { return timeout_ns_; }
  static void SetTimeoutNs(const timestamp_t& time) {
    std::lock_guard<mutex_t> lck(mutex_);
    timeout_ns_ = time;
    if (instance_ != NULL) instance_->timeout_ = instance_->timer_->ns_to_sysclock(time);
  }

 private:
  // System agents iterating callback
  static hsa_status_t GetHsaAgentsCallback(hsa_agent_t agent, void* data);

  // Callback function to find and bind kernarg region of an agent
  static hsa_status_t FindMemRegionsCallback(hsa_region_t region, void* data);

  // Load AQL profile HSA extension library directly
  static hsa_status_t LoadAqlProfileLib(aqlprofile_pfn_t* api);

  // Constructor of the class. Will initialize the Hsa Runtime and
  // query the system topology to get the list of Cpu and Gpu devices
  explicit HsaRsrcFactory(bool initialize_hsa);

  // Destructor of the class
  ~HsaRsrcFactory();

  // Add an instance of AgentInfo representing a Hsa Gpu agent
  const AgentInfo* AddAgentInfo(const hsa_agent_t agent);

  // To mmap command buffer memory
  static const bool CMD_MEMORY_MMAP = false;

  // HSA was initialized
  const bool initialize_hsa_;

  static HsaRsrcFactory* instance_;
  static mutex_t mutex_;

  // Used to maintain a list of Hsa Gpu Agent Info
  std::vector<const AgentInfo*> gpu_list_;
  std::vector<hsa_agent_t> gpu_agents_;

  // Used to maintain a list of Hsa Cpu Agent Info
  std::vector<const AgentInfo*> cpu_list_;
  std::vector<hsa_agent_t> cpu_agents_;

  // System agents map
  std::map<hsa_agent_handle_t, const AgentInfo*> agent_map_;

  // AqlProfile API table
  aqlprofile_pfn_t aqlprofile_api_;

  // Loader API table
  hsa_ven_amd_loader_1_00_pfn_t loader_api_;

  // System timeout, ns
  static timestamp_t timeout_ns_;
  // System timeout, sysclock
  timestamp_t timeout_;

  // HSA timer
  HsaTimer* timer_;

  // CPU/kern-arg memory pools
  hsa_amd_memory_pool_t* cpu_pool_;
  hsa_amd_memory_pool_t* kern_arg_pool_;
};

#endif  // SRC_UTIL_HSA_RSRC_FACTORY_H_
