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

#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ext_finalize.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <atomic>
#include <iostream>
#include <mutex>
#include <map>
#include <string>
#include <vector>

#define HSA_ARGUMENT_ALIGN_BYTES 16
#define HSA_QUEUE_ALIGN_BYTES 64
#define HSA_PACKET_ALIGN_BYTES 64

#define CHECK_STATUS(msg, status)                                                                  \
  do {                                                                                             \
    if ((status) != HSA_STATUS_SUCCESS) {                                                          \
      const char* emsg = 0;                                                                        \
      hsa_status_string(status, &emsg);                                                            \
      printf("%s: %s\n", msg, emsg ? emsg : "<unknown error>");                                    \
      abort();                                                                                     \
    }                                                                                              \
  } while (0)

#define CHECK_ITER_STATUS(msg, status)                                                             \
  do {                                                                                             \
    if ((status) != HSA_STATUS_INFO_BREAK) {                                                       \
      const char* emsg = 0;                                                                        \
      hsa_status_string(status, &emsg);                                                            \
      printf("%s: %s\n", msg, emsg ? emsg : "<unknown error>");                                    \
      abort();                                                                                     \
    }                                                                                              \
  } while (0)

static const size_t MEM_PAGE_BYTES = 0x1000;
static const size_t MEM_PAGE_MASK = MEM_PAGE_BYTES - 1;
typedef decltype(hsa_agent_t::handle) hsa_agent_handle_t;

struct hsa_pfn_t {
  decltype(::hsa_init)* hsa_init;
  decltype(::hsa_shut_down)* hsa_shut_down;
  decltype(::hsa_agent_get_info)* hsa_agent_get_info;
  decltype(::hsa_iterate_agents)* hsa_iterate_agents;

  decltype(::hsa_queue_create)* hsa_queue_create;
  decltype(::hsa_queue_destroy)* hsa_queue_destroy;
  decltype(::hsa_queue_load_read_index_relaxed)* hsa_queue_load_read_index_relaxed;
  decltype(::hsa_queue_load_write_index_relaxed)* hsa_queue_load_write_index_relaxed;
  decltype(::hsa_queue_add_write_index_scacq_screl)* hsa_queue_add_write_index_scacq_screl;

  decltype(::hsa_signal_create)* hsa_signal_create;
  decltype(::hsa_signal_destroy)* hsa_signal_destroy;
  decltype(::hsa_signal_load_relaxed)* hsa_signal_load_relaxed;
  decltype(::hsa_signal_store_relaxed)* hsa_signal_store_relaxed;
  decltype(::hsa_signal_wait_scacquire)* hsa_signal_wait_scacquire;
  decltype(::hsa_signal_store_screlease)* hsa_signal_store_screlease;

  decltype(::hsa_code_object_reader_create_from_file)* hsa_code_object_reader_create_from_file;
  decltype(::hsa_executable_create_alt)* hsa_executable_create_alt;
  decltype(::hsa_executable_load_agent_code_object)* hsa_executable_load_agent_code_object;
  decltype(::hsa_executable_freeze)* hsa_executable_freeze;
  decltype(::hsa_executable_destroy)* hsa_executable_destroy;
  decltype(::hsa_executable_get_symbol)* hsa_executable_get_symbol;
  decltype(::hsa_executable_symbol_get_info)* hsa_executable_symbol_get_info;
  decltype(::hsa_executable_iterate_symbols)* hsa_executable_iterate_symbols;

  decltype(::hsa_system_get_info)* hsa_system_get_info;
  decltype(::hsa_system_get_major_extension_table)* hsa_system_get_major_extension_table;

  decltype(::hsa_amd_agent_iterate_memory_pools)* hsa_amd_agent_iterate_memory_pools;
  decltype(::hsa_amd_memory_pool_get_info)* hsa_amd_memory_pool_get_info;
  decltype(::hsa_amd_memory_pool_allocate)* hsa_amd_memory_pool_allocate;
  decltype(::hsa_amd_agents_allow_access)* hsa_amd_agents_allow_access;
  decltype(::hsa_amd_memory_async_copy)* hsa_amd_memory_async_copy;

  decltype(::hsa_amd_signal_async_handler)* hsa_amd_signal_async_handler;
  decltype(::hsa_amd_profiling_set_profiler_enabled)* hsa_amd_profiling_set_profiler_enabled;
  decltype(::hsa_amd_profiling_get_async_copy_time)* hsa_amd_profiling_get_async_copy_time;
  decltype(::hsa_amd_profiling_get_dispatch_time)* hsa_amd_profiling_get_dispatch_time;
};

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

  // Number of Shader Arrays Per Shader Engines in Gpu
  uint32_t shader_arrays_per_se;

  static const uint32_t lds_block_size = 128 * 4;
};

// HSA timer class
// Provides current HSA timestampa and system-clock/ns conversion API
class HsaTimer {
 public:
  typedef uint64_t timestamp_t;
  static const timestamp_t TIMESTAMP_MAX = UINT64_MAX;
  typedef long double freq_t;

  enum time_id_t {
    TIME_ID_CLOCK_REALTIME = 0,
    TIME_ID_CLOCK_REALTIME_COARSE = 1,
    TIME_ID_CLOCK_MONOTONIC = 2,
    TIME_ID_CLOCK_MONOTONIC_COARSE = 3,
    TIME_ID_CLOCK_MONOTONIC_RAW = 4,
    TIME_ID_NUMBER
  };

  HsaTimer(const hsa_pfn_t* hsa_api) : hsa_api_(hsa_api) {
    timestamp_t sysclock_hz = 0;
    hsa_status_t status =
        hsa_api_->hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &sysclock_hz);
    CHECK_STATUS("hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY)", status);
    sysclock_factor_ = (freq_t)1000000000 / (freq_t)sysclock_hz;
  }

  // Methods for system-clock/ns conversion
  timestamp_t sysclock_to_ns(const timestamp_t& sysclock) const {
    return timestamp_t((freq_t)sysclock * sysclock_factor_);
  }
  timestamp_t ns_to_sysclock(const timestamp_t& time) const {
    return timestamp_t((freq_t)time / sysclock_factor_);
  }

  // Method for timespec/ns conversion
  static timestamp_t timespec_to_ns(const timespec& time) {
    return ((timestamp_t)time.tv_sec * 1000000000) + time.tv_nsec;
  }

  // Return timestamp in 'ns'
  timestamp_t timestamp_ns() const {
    timestamp_t sysclock;
    hsa_status_t status = hsa_api_->hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP, &sysclock);
    CHECK_STATUS("hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP)", status);
    return sysclock_to_ns(sysclock);
  }

  // Return time in 'ns'
  timestamp_t clocktime_ns(clockid_t clock_id) const {
    timespec time;
    clock_gettime(clock_id, &time);
    return timespec_to_ns(time);
  }

  // Return pair of correlated values of profiling timestamp and time with
  // correlation error for a given time ID and number of iterations
  void correlated_pair_ns(time_id_t time_id, uint32_t iters, timestamp_t* timestamp_v,
                          timestamp_t* time_v, timestamp_t* error_v) {
    clockid_t clock_id = 0;
    switch (time_id) {
      case TIME_ID_CLOCK_REALTIME:
        clock_id = CLOCK_REALTIME;
        break;
      case TIME_ID_CLOCK_REALTIME_COARSE:
        clock_id = CLOCK_REALTIME_COARSE;
        break;
      case TIME_ID_CLOCK_MONOTONIC:
        clock_id = CLOCK_MONOTONIC;
        break;
      case TIME_ID_CLOCK_MONOTONIC_COARSE:
        clock_id = CLOCK_MONOTONIC_COARSE;
        break;
      case TIME_ID_CLOCK_MONOTONIC_RAW:
        clock_id = CLOCK_MONOTONIC_RAW;
        break;
      default:
        CHECK_STATUS("internal error: invalid time_id", HSA_STATUS_ERROR);
    }

    std::vector<timestamp_t> ts_vec(iters);
    std::vector<timespec> tm_vec(iters);
    const uint32_t steps = iters - 1;

    for (uint32_t i = 0; i < iters; ++i) {
      hsa_api_->hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP, &ts_vec[i]);
      clock_gettime(clock_id, &tm_vec[i]);
    }

    const timestamp_t ts_base = sysclock_to_ns(ts_vec.front());
    const timestamp_t tm_base = timespec_to_ns(tm_vec.front());
    const timestamp_t error = (ts_vec.back() - ts_vec.front()) / (2 * steps);

    timestamp_t ts_accum = 0;
    timestamp_t tm_accum = 0;
    for (uint32_t i = 0; i < iters; ++i) {
      ts_accum += (ts_vec[i] - ts_base);
      tm_accum += (timespec_to_ns(tm_vec[i]) - tm_base);
    }

    *timestamp_v = (ts_accum / iters) + ts_base + error;
    *time_v = (tm_accum / iters) + tm_base;
    *error_v = error;
  }

 private:
  // Timestamp frequency factor
  freq_t sysclock_factor_;
  // HSA API table
  const hsa_pfn_t* const hsa_api_;
};

class HsaRsrcFactory {
 public:
  static const size_t CMD_SLOT_SIZE_B = 0x40;
  typedef std::recursive_mutex mutex_t;
  typedef HsaTimer::timestamp_t timestamp_t;

  // Executables loading tracking
  struct symbols_map_data_t {
    const char* name;
    uint64_t refs_count;
  };
  typedef std::map<uint64_t, symbols_map_data_t> symbols_map_t;

  static HsaRsrcFactory* Create(bool initialize_hsa = true) {
    std::lock_guard<mutex_t> lck(mutex_);
    HsaRsrcFactory* obj = instance_.load(std::memory_order_relaxed);
    if (obj == NULL) {
      obj = new HsaRsrcFactory(initialize_hsa);
      instance_.store(obj, std::memory_order_release);
    }
    return obj;
  }

  static HsaRsrcFactory& Instance() {
    HsaRsrcFactory* obj = instance_.load(std::memory_order_acquire);
    if (obj == NULL) obj = Create(false);
    hsa_status_t status = (obj != NULL) ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR;
    CHECK_STATUS("HsaRsrcFactory::Instance() failed", status);
    return *obj;
  }

  static void Destroy() {
    std::lock_guard<mutex_t> lck(mutex_);
    if (instance_) delete instance_.load();
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
  hsa_signal_value_t SignalWait(const hsa_signal_t& signal,
                                const hsa_signal_value_t& signal_value) const;

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

  // Utils for submitting AQL packet to a given queue
  static void* GetReadPointer(hsa_queue_t* queue);
  static uint64_t Submit(hsa_queue_t* queue, const void* packet);
  static uint64_t Submit(hsa_queue_t* queue, const void* packet, size_t size_bytes);

  // Enable executables loading tracking
  static bool IsExecutableTracking() { return executable_tracking_on_; }
  static void EnableExecutableTracking(HsaApiTable* table);

  typedef symbols_map_t::iterator symbols_map_it_t;

  static inline const char* GetKernelNameRef(const uint64_t& addr) {
    if (symbols_map_ == NULL) {
      fprintf(stderr, "HsaRsrcFactory::GetKernelNameRef: kernel addr (0x%lx), error\n", addr);
      abort();
    }

    std::lock_guard<mutex_t> lck(mutex_);

    const auto it = symbols_map_->find(addr);
    if (it == symbols_map_->end()) {
      fprintf(stderr, "HsaRsrcFactory::GetKernelNameRef: kernel addr (0x%lx) is not found\n", addr);
      abort();
    }

    return it->second.name;
  }

  static inline symbols_map_it_t AcquireKernelNameRef(const uint64_t& addr) {
    if (symbols_map_ == NULL) {
      fprintf(stderr, "HsaRsrcFactory::GetKernelNameRef: kernel addr (0x%lx), error\n", addr);
      abort();
    }

    std::lock_guard<mutex_t> lck(mutex_);

    const auto it = symbols_map_->find(addr);
    if (it == symbols_map_->end()) {
      fprintf(stderr, "HsaRsrcFactory::GetKernelNameRef: kernel addr (0x%lx) is not found\n", addr);
      abort();
    }

    std::atomic<uint64_t>* atomic_ptr =
        reinterpret_cast<std::atomic<uint64_t>*>(&(it->second.refs_count));
    atomic_ptr->fetch_add(1, std::memory_order_relaxed);

    return it;
  }

  static inline void ReleaseKernelNameRef(const symbols_map_it_t& it) {
    std::atomic<uint64_t>* atomic_ptr =
        reinterpret_cast<std::atomic<uint64_t>*>(&(it->second.refs_count));
    atomic_ptr->fetch_sub(1, std::memory_order_relaxed);
  }

  static inline void SetKernelNameRef(const uint64_t& addr, const char* name, const int& free) {
    if (symbols_map_ == NULL) {
      std::lock_guard<mutex_t> lck(mutex_);
      if (symbols_map_ == NULL) symbols_map_ = new symbols_map_t;
    }

    auto it = symbols_map_->find(addr);
    if (it != symbols_map_->end()) {
      while (1) {
        while (it->second.refs_count != 0) sched_yield();
        mutex_.lock();
        if (it->second.refs_count == 0) break;
        mutex_.unlock();
      }
    }

    if (it != symbols_map_->end()) {
      delete[] it->second.name;
      if (free == 1) {
        symbols_map_->erase(it);
      } else {
        fprintf(stderr, "HsaRsrcFactory::SetKernelNameRef: to set kernel addr (0x%lx) conflict\n",
                addr);
        abort();
      }
    } else {
      if (free == 0) {
        symbols_map_->insert({addr, symbols_map_data_t{name, 0}});
      } else {
        fprintf(stderr, "HsaRsrcFactory::SetKernelNameRef: to free kernel addr (0x%lx) not found\n",
                addr);
        abort();
      }
    }

    mutex_.unlock();
  }

  // Initialize HSA API table
  void static InitHsaApiTable(HsaApiTable* table);
  static const hsa_pfn_t* HsaApi() { return &hsa_api_; }

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
    if (instance_ != NULL) Instance().timeout_ = Instance().timer_->ns_to_sysclock(time);
  }

  void CorrelateTime(HsaTimer::time_id_t time_id, uint32_t iters) {
    timestamp_t timestamp_v = 0;
    timestamp_t time_v = 0;
    timestamp_t error_v = 0;
    timer_->correlated_pair_ns(time_id, iters, &timestamp_v, &time_v, &error_v);
    time_shift_[time_id] = time_v - timestamp_v;
    time_error_[time_id] = error_v;
  }

  hsa_status_t GetTimeVal(uint32_t time_id, uint64_t time_stamp, uint64_t* time_value) {
    if (time_id >= HsaTimer::TIME_ID_NUMBER) return HSA_STATUS_ERROR;
    *time_value = time_stamp + time_shift_[time_id];
    return HSA_STATUS_SUCCESS;
  }

  hsa_status_t GetTimeErr(uint32_t time_id, uint64_t* err) {
    *err = time_error_[time_id];
    return HSA_STATUS_SUCCESS;
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

  static std::atomic<HsaRsrcFactory*> instance_;
  static mutex_t mutex_;

  // Used to maintain a list of Hsa Gpu Agent Info
  std::vector<const AgentInfo*> gpu_list_;
  std::vector<hsa_agent_t> gpu_agents_;

  // Used to maintain a list of Hsa Cpu Agent Info
  std::vector<const AgentInfo*> cpu_list_;
  std::vector<hsa_agent_t> cpu_agents_;

  // System agents map
  std::map<hsa_agent_handle_t, const AgentInfo*> agent_map_;

  static symbols_map_t* symbols_map_;
  static bool executable_tracking_on_;
  static hsa_status_t hsa_executable_freeze_interceptor(hsa_executable_t executable,
                                                        const char* options);
  static hsa_status_t hsa_executable_destroy_interceptor(hsa_executable_t executable);
  static hsa_status_t executable_symbols_cb(hsa_executable_t exec, hsa_executable_symbol_t symbol,
                                            void* data);

  // HSA runtime API table
  static hsa_pfn_t hsa_api_;

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

  // Time shift array to support time conversion
  timestamp_t time_shift_[HsaTimer::TIME_ID_NUMBER];
  timestamp_t time_error_[HsaTimer::TIME_ID_NUMBER];

  // CPU/kern-arg memory pools
  hsa_amd_memory_pool_t* cpu_pool_;
  hsa_amd_memory_pool_t* kern_arg_pool_;
};


#endif  // TEST_UTIL_HSA_RSRC_FACTORY_H_
