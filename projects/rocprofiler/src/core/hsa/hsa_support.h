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

#ifndef SRC_CORE_HSA_HSA_SUPPORT_H_
#define SRC_CORE_HSA_HSA_SUPPORT_H_

#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>
#include <hsa/hsa_ven_amd_loader.h>

#include <atomic>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <memory>
#include <optional>
#include <shared_mutex>
#include "rocprofiler.h"
#include "src/core/hardware/hsa_info.h"

// HSA EVT data type
typedef struct {
  union {
    struct {
      const void* ptr;            // allocated area ptr
      size_t size;                // allocated area size, zero size means 'free' callback
      hsa_amd_segment_t segment;  // allocated area's memory segment type
      hsa_amd_memory_pool_global_flag_t global_flag;  // allocated area's memory global flag
      int is_code;                                    // equal to 1 if code is allocated
    } allocate;

    struct {
      hsa_device_type_t type;  // type of assigned device
      uint32_t id;             // id of assigned device
      hsa_agent_t agent;       // device HSA agent handle
      const void* ptr;         // ptr the device is assigned to
    } device;

    struct {
      const void* dst;  // memcopy dst ptr
      const void* src;  // memcopy src ptr
      size_t size;      // memcopy size bytes
    } memcopy;

    struct {
      const void* packet;       // submitted to GPU packet
      const char* kernel_name;  // kernel name, NULL if not a kernel dispatch packet
      hsa_queue_t* queue;       // HSA queue the packet was submitted to
      uint32_t device_type;     // type of device the packet is submitted to
      uint32_t device_id;       // id of device the packet is submitted to
    } submit;

    struct {
      uint64_t object;       // kernel symbol object
      const char* name;      // kernel symbol name
      uint32_t name_length;  // kernel symbol name length
      int unload;            // symbol executable destroy
    } ksymbol;

    struct {
      uint32_t storage_type;  // code object storage type
      int storage_file;       // origin file descriptor
      uint64_t memory_base;   // origin memory base
      uint64_t memory_size;   // origin memory size
      uint64_t load_base;     // code object load base
      uint64_t load_size;     // code object load size
      uint64_t load_delta;    // code object load size
      uint32_t uri_length;    // URI string length (not including the terminating
                              // NUL character)
      const char* uri;        // URI string
      hsa_agent_t agent;      // device HSA agent handle
      int unload;             // unload flag
    } codeobj;
  };

} hsa_evt_data_t;

namespace rocprofiler {


class HSAAgentInfo {
 private:
  hsa_agent_t agent_;
  Agent::DeviceInfo device_info_;
  hsa_agent_t near_cpu_agent_;
  hsa_device_type_t type_;


 public:
  HSAAgentInfo(hsa_agent_t agent, hsa_device_type_t type) : agent_(agent), type_(type){};
  uint64_t getHandle() const;
  const Agent::DeviceInfo& GetDeviceInfo() const;
  void SetNearCpuAgent(hsa_agent_t near_cpu_agent);
  void SetDeviceInfo(Agent::DeviceInfo device_info);
  hsa_agent_t GetNearCpuAgent() const;
  hsa_device_type_t GetType() const;
  hsa_amd_memory_pool_t cpu_pool_;
  hsa_amd_memory_pool_t kernarg_pool_;
  hsa_amd_memory_pool_t gpu_pool_;
};


struct queues_deleter {
  queues_deleter(){};
  queues_deleter(queues_deleter&){};
  void operator()(void* queue) const;
};

struct new_signal_timestamp_t {
  hsa_signal_t new_signal;
  std::optional<hsa_amd_profiling_dispatch_time_t> time;
};

class HSASupport_Singleton {
 private:
  HSASupport_Singleton(){};
  ~HSASupport_Singleton() = delete;
  CoreApiTable saved_core_api;
  AmdExtTable saved_amd_ext_api;
  hsa_ven_amd_loader_1_01_pfn_t hsa_loader_api;
  std::mutex info_map_mutex_;
  std::unordered_map<uint64_t, HSAAgentInfo> HSAagent_info_map_;
  std::atomic<bool> ksymbols_flag{true};
  std::atomic<bool> kernel_names_flag{true};
  std::mutex queues_mutex_;
  std::unordered_map<hsa_queue_t*, std::unique_ptr<void, queues_deleter&>> queues;
  void SetCoreApiTable(CoreApiTable& table);
  void SetAmdExtTable(AmdExtTable& table);
  void SetHSALoaderApi();

 public:
  std::vector<hsa_agent_t> gpu_agents;
  HSAAgentInfo& GetHSAAgentInfo(uint64_t agent_handle);
  HSAAgentInfo& GetHSAAgentInfo(Agent::DeviceInfo device_info);
  Agent::DeviceInfo& GetDeviceInfo(HSAAgentInfo* agent_info);
  std::shared_mutex kernel_names_map_lock;
  std::unordered_map<uint64_t, std::string>* kernel_names;
  std::mutex ksymbol_map_lock;
  std::map<uint64_t, std::string>* ksymbols;
  std::mutex signals_timestamps_map_lock;
  std::map<uint64_t, new_signal_timestamp_t> signals_timestamps;
  void SetHSAAgentInfo(hsa_agent_t agent, HSAAgentInfo hsa_agent_info);
  static HSASupport_Singleton& GetInstance();
  CoreApiTable& GetCoreApiTable();
  AmdExtTable& GetAmdExtTable();
  hsa_ven_amd_loader_1_01_pfn_t& GetHSALoaderApi();
  void AddQueue(hsa_queue_t* queue, std::unique_ptr<void, queues_deleter&>);
  void RemoveQueue(hsa_queue_t* queue);
  void HSAInitialize(HsaApiTable* Table);
  void HSAFinalize();
  void InitKsymbols();
  void FinitKsymbols();
  void CreateSignal(uint32_t attribute, hsa_signal_t* signal);
  HSASupport_Singleton(const HSASupport_Singleton&) = delete;
  HSASupport_Singleton& operator=(const HSASupport_Singleton&) = delete;
};

bool hsa_support_IterateCounters(rocprofiler_counters_info_callback_t counters_info_callback);
}  // namespace rocprofiler

#include "src/core/session/tracer/src/roctracer.h"

namespace roctracer::hsa_support {

struct hsa_trace_data_t {
  hsa_api_data_t api_data;
  uint64_t phase_enter_timestamp;
  uint64_t phase_data;

  void (*phase_enter)(hsa_api_id_t operation_id, hsa_trace_data_t* data);
  void (*phase_exit)(hsa_api_id_t operation_id, hsa_trace_data_t* data);
};

void Initialize(HsaApiTable* table);
void Finalize();

const char* GetApiName(uint32_t id);
const char* GetEvtName(uint32_t id);
const char* GetOpsName(uint32_t id);
uint32_t GetApiCode(const char* str);

void RegisterTracerCallback(int (*function)(rocprofiler_tracer_activity_domain_t domain,
                                            uint32_t operation_id, void* data));
}  // namespace roctracer::hsa_support

#endif  // SRC_CORE_HSA_HSA_SUPPORT_H_
