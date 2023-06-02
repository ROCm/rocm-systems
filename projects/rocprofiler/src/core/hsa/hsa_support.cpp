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

#include "hsa_support.h"

#include <hsa/amd_hsa_signal.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ven_amd_loader.h>

#include <mutex>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/hardware/hsa_info.h"
#include "src/core/session/tracer/src/correlation_id.h"
#include "src/core/session/tracer/src/exception.h"
#include "src/core/session/tracer/src/roctracer.h"
#include "src/utils/helper.h"

#include "src/core/hsa/queues/queue.h"
#include "src/api/rocprofiler_singleton.h"

#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;

namespace {

hsa_status_t hsa_executable_iteration_callback(hsa_executable_t executable, hsa_agent_t agent,
                                               hsa_executable_symbol_t symbol, void* args) {
  hsa_symbol_kind_t type;
  rocprofiler::hsa_support::GetCoreApiTable().hsa_executable_symbol_get_info_fn(
      symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &type);
  if (type == HSA_SYMBOL_KIND_KERNEL) {
    uint32_t name_length;
    rocprofiler::hsa_support::GetCoreApiTable().hsa_executable_symbol_get_info_fn(
        symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &name_length);
    // TODO(aelwazir): to be removed if the HSA fixed the issue of corrupted
    // names overflowing the length given
    if (name_length > 1) {
      if (!(*static_cast<bool*>(args))) {
        char name[name_length + 1];
        uint64_t kernel_object;
        rocprofiler::hsa_support::GetCoreApiTable().hsa_executable_symbol_get_info_fn(
            symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME, name);
        rocprofiler::hsa_support::GetCoreApiTable().hsa_executable_symbol_get_info_fn(
            symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_object);
        std::string kernel_name = std::string(name).substr(0, name_length);
        rocprofiler::AddKernelName(kernel_object, kernel_name);
      } else {
        uint64_t kernel_object;
        rocprofiler::hsa_support::GetCoreApiTable().hsa_executable_symbol_get_info_fn(
            symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_object);
        rocprofiler::RemoveKernelName(kernel_object);
      }
    }
  }

  return HSA_STATUS_SUCCESS;
}

std::atomic<int (*)(rocprofiler_tracer_activity_domain_t domain, uint32_t operation_id, void* data)>
    report_activity;

bool IsEnabled(rocprofiler_tracer_activity_domain_t domain, uint32_t operation_id) {
  auto report = report_activity.load(std::memory_order_relaxed);
  return report && report(domain, operation_id, nullptr) == 0;
}

void ReportActivity(rocprofiler_tracer_activity_domain_t domain, uint32_t operation_id,
                    void* data) {
  if (auto report = report_activity.load(std::memory_order_relaxed))
    report(domain, operation_id, data);
}

}  // namespace

#include "hsa_prof_str.inline.h"

namespace roctracer::hsa_support {

namespace {

CoreApiTable saved_core_api{};
AmdExtTable saved_amd_ext_api{};
hsa_ven_amd_loader_1_01_pfn_t hsa_loader_api{};

struct AgentInfo {
  uint32_t id;
  hsa_device_type_t type;
};
std::unordered_map<decltype(hsa_agent_t::handle), AgentInfo> agent_info_map;

class Tracker {
 public:
  enum { ENTRY_INV = 0, ENTRY_INIT = 1, ENTRY_COMPL = 2 };

  enum entry_type_t {
    DFLT_ENTRY_TYPE = 0,
    API_ENTRY_TYPE = 1,
    COPY_ENTRY_TYPE = 2,
    KERNEL_ENTRY_TYPE = 3,
    NUM_ENTRY_TYPE = 4
  };

  struct entry_t {
    std::atomic<uint32_t> valid;
    entry_type_t type;
    uint64_t correlation_id;
    roctracer_timestamp_t begin;  // begin timestamp, ns
    roctracer_timestamp_t end;    // end timestamp, ns
    hsa_agent_t agent;
    uint32_t dev_index;
    hsa_signal_t orig;
    hsa_signal_t signal;
    void (*handler)(const entry_t*);
    union {
      struct {
      } copy;
      struct {
        const char* name;
        hsa_agent_t agent;
        uint32_t tid;
      } kernel;
    };
  };

  // Add tracker entry
  inline static void Enable(entry_type_t type, const hsa_agent_t& agent, const hsa_signal_t& signal,
                            entry_t* entry) {
    hsa_status_t status = HSA_STATUS_ERROR;

    // Creating a new tracker entry
    entry->type = type;
    entry->agent = agent;
    entry->dev_index = 0;  // hsa_rsrc->GetAgentInfo(agent)->dev_index;
    entry->orig = signal;
    entry->valid.store(ENTRY_INIT, std::memory_order_release);

    // Creating a proxy signal
    status = rocprofiler::hsa_support::GetCoreApiTable().hsa_signal_create_fn(1, 0, NULL,
                                                                            &(entry->signal));
    if (status != HSA_STATUS_SUCCESS) rocprofiler::fatal("hsa_signal_create failed");
    status = rocprofiler::hsa_support::GetAmdExtTable().hsa_amd_signal_async_handler_fn(
        entry->signal, HSA_SIGNAL_CONDITION_LT, 1, Handler, entry);
    if (status != HSA_STATUS_SUCCESS) rocprofiler::fatal("hsa_amd_signal_async_handler failed");
  }

  // Delete tracker entry
  inline static void Disable(entry_t* entry) {
    rocprofiler::hsa_support::GetCoreApiTable().hsa_signal_destroy_fn(entry->signal);
    entry->valid.store(ENTRY_INV, std::memory_order_release);
  }

 private:
  // Entry completion
  inline static void Complete(hsa_signal_value_t signal_value, entry_t* entry) {
    static roctracer_timestamp_t sysclock_period = []() {
      uint64_t sysclock_hz = 0;
      hsa_status_t status = rocprofiler::hsa_support::GetCoreApiTable().hsa_system_get_info_fn(
          HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &sysclock_hz);
      if (status != HSA_STATUS_SUCCESS) rocprofiler::fatal("hsa_system_get_info failed");
      return (uint64_t)1000000000 / sysclock_hz;
    }();

    if (entry->type == COPY_ENTRY_TYPE) {
      hsa_amd_profiling_async_copy_time_t async_copy_time{};
      hsa_status_t status =
          rocprofiler::hsa_support::GetAmdExtTable().hsa_amd_profiling_get_async_copy_time_fn(
              entry->signal, &async_copy_time);
      if (status != HSA_STATUS_SUCCESS)
        rocprofiler::fatal("hsa_amd_profiling_get_async_copy_time failed");
      entry->begin = async_copy_time.start * sysclock_period;
      entry->end = async_copy_time.end * sysclock_period;
    } else {
      assert(false && "should not reach here");
    }

    hsa_signal_t orig = entry->orig;
    hsa_signal_t signal = entry->signal;

    // Releasing completed entry
    entry->valid.store(ENTRY_COMPL, std::memory_order_release);

    assert(entry->handler != nullptr);
    entry->handler(entry);

    // Original intercepted signal completion
    if (orig.handle) {
      amd_signal_t* orig_signal_ptr = reinterpret_cast<amd_signal_t*>(orig.handle);
      amd_signal_t* prof_signal_ptr = reinterpret_cast<amd_signal_t*>(signal.handle);
      orig_signal_ptr->start_ts = prof_signal_ptr->start_ts;
      orig_signal_ptr->end_ts = prof_signal_ptr->end_ts;

      [[maybe_unused]] const hsa_signal_value_t new_value =
          rocprofiler::hsa_support::GetCoreApiTable().hsa_signal_load_relaxed_fn(orig) - 1;
      assert(signal_value == new_value && "Tracker::Complete bad signal value");
      rocprofiler::hsa_support::GetCoreApiTable().hsa_signal_store_screlease_fn(orig, signal_value);
    }
    rocprofiler::hsa_support::GetCoreApiTable().hsa_signal_destroy_fn(signal);
    delete entry;
  }

  // Handler for packet completion
  static bool Handler(hsa_signal_value_t signal_value, void* arg) {
    // Acquire entry
    entry_t* entry = reinterpret_cast<entry_t*>(arg);
    while (entry->valid.load(std::memory_order_acquire) != ENTRY_INIT) sched_yield();

    // Complete entry
    Tracker::Complete(signal_value, entry);
    return false;
  }
};

hsa_status_t HSA_API MemoryAllocateIntercept(hsa_region_t region, size_t size, void** ptr) {
  hsa_status_t status =
      rocprofiler::hsa_support::GetCoreApiTable().hsa_memory_allocate_fn(region, size, ptr);
  if (status != HSA_STATUS_SUCCESS) return status;

  if (IsEnabled(ACTIVITY_DOMAIN_HSA_EVT, HSA_EVT_ID_ALLOCATE)) {
    hsa_evt_data_t data{};
    data.allocate.ptr = *ptr;
    data.allocate.size = size;
    if (rocprofiler::hsa_support::GetCoreApiTable().hsa_region_get_info_fn(
            region, HSA_REGION_INFO_SEGMENT, &data.allocate.segment) != HSA_STATUS_SUCCESS ||
        rocprofiler::hsa_support::GetCoreApiTable().hsa_region_get_info_fn(
            region, HSA_REGION_INFO_GLOBAL_FLAGS, &data.allocate.global_flag) != HSA_STATUS_SUCCESS)
      rocprofiler::fatal("hsa_region_get_info failed");

    ReportActivity(ACTIVITY_DOMAIN_HSA_EVT, HSA_EVT_ID_ALLOCATE, &data);
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t MemoryAssignAgentIntercept(void* ptr, hsa_agent_t agent,
                                        hsa_access_permission_t access) {
  hsa_status_t status =
      rocprofiler::hsa_support::GetCoreApiTable().hsa_memory_assign_agent_fn(ptr, agent, access);
  if (status != HSA_STATUS_SUCCESS) return status;

  if (IsEnabled(ACTIVITY_DOMAIN_HSA_EVT, HSA_EVT_ID_DEVICE)) {
    hsa_evt_data_t data{};
    data.device.ptr = ptr;
    if (rocprofiler::hsa_support::GetCoreApiTable().hsa_agent_get_info_fn(
            agent, HSA_AGENT_INFO_DEVICE, &data.device.type) != HSA_STATUS_SUCCESS)
      rocprofiler::fatal("hsa_agent_get_info failed");

    ReportActivity(ACTIVITY_DOMAIN_HSA_EVT, HSA_EVT_ID_DEVICE, &data);
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t MemoryCopyIntercept(void* dst, const void* src, size_t size) {
  hsa_status_t status =
      rocprofiler::hsa_support::GetCoreApiTable().hsa_memory_copy_fn(dst, src, size);
  if (status != HSA_STATUS_SUCCESS) return status;

  if (IsEnabled(ACTIVITY_DOMAIN_HSA_EVT, HSA_EVT_ID_MEMCOPY)) {
    hsa_evt_data_t data{};
    data.memcopy.dst = dst;
    data.memcopy.src = src;
    data.memcopy.size = size;

    ReportActivity(ACTIVITY_DOMAIN_HSA_EVT, HSA_EVT_ID_MEMCOPY, &data);
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t MemoryPoolAllocateIntercept(hsa_amd_memory_pool_t pool, size_t size, uint32_t flags,
                                         void** ptr) {
  hsa_status_t status = rocprofiler::hsa_support::GetAmdExtTable().hsa_amd_memory_pool_allocate_fn(
      pool, size, flags, ptr);
  if (size == 0 || status != HSA_STATUS_SUCCESS) return status;

  if (IsEnabled(ACTIVITY_DOMAIN_HSA_EVT, HSA_EVT_ID_ALLOCATE)) {
    hsa_evt_data_t data{};
    data.allocate.ptr = *ptr;
    data.allocate.size = size;

    if (rocprofiler::hsa_support::GetAmdExtTable().hsa_amd_memory_pool_get_info_fn(
            pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &data.allocate.segment) != HSA_STATUS_SUCCESS ||
        rocprofiler::hsa_support::GetAmdExtTable().hsa_amd_memory_pool_get_info_fn(
            pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &data.allocate.global_flag) !=
            HSA_STATUS_SUCCESS)
      rocprofiler::fatal("hsa_region_get_info failed");

    ReportActivity(ACTIVITY_DOMAIN_HSA_EVT, HSA_EVT_ID_ALLOCATE, &data);
  }

  if (IsEnabled(ACTIVITY_DOMAIN_HSA_EVT, HSA_EVT_ID_DEVICE)) {
    auto callback_data = std::make_pair(pool, ptr);
    auto agent_callback = [](hsa_agent_t agent, void* iterate_agent_callback_data) {
      auto [pool, ptr] = *reinterpret_cast<decltype(callback_data)*>(iterate_agent_callback_data);

      if (hsa_amd_memory_pool_access_t value;
          rocprofiler::hsa_support::GetAmdExtTable().hsa_amd_agent_memory_pool_get_info_fn(
              agent, pool, HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &value) != HSA_STATUS_SUCCESS ||
          value != HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT)
        return HSA_STATUS_SUCCESS;

      auto it = agent_info_map.find(agent.handle);
      if (it == agent_info_map.end()) rocprofiler::fatal("agent was not found in the agent_info map");

      hsa_evt_data_t data{};
      data.device.type = it->second.type;
      data.device.id = it->second.id;
      data.device.agent = agent;
      data.device.ptr = ptr;

      ReportActivity(ACTIVITY_DOMAIN_HSA_EVT, HSA_EVT_ID_DEVICE, &data);
      return HSA_STATUS_SUCCESS;
    };
    rocprofiler::hsa_support::GetCoreApiTable().hsa_iterate_agents_fn(agent_callback, &callback_data);
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t MemoryPoolFreeIntercept(void* ptr) {
  if (IsEnabled(ACTIVITY_DOMAIN_HSA_EVT, HSA_EVT_ID_ALLOCATE)) {
    hsa_evt_data_t data{};
    data.allocate.ptr = ptr;
    data.allocate.size = 0;
    ReportActivity(ACTIVITY_DOMAIN_HSA_EVT, HSA_EVT_ID_ALLOCATE, &data);
  }

  if (ptr)
    return rocprofiler::hsa_support::GetAmdExtTable().hsa_amd_memory_pool_free_fn(ptr);
  else
    return HSA_STATUS_SUCCESS;
}

// Agent allow access callback 'hsa_amd_agents_allow_access'
hsa_status_t AgentsAllowAccessIntercept(uint32_t num_agents, const hsa_agent_t* agents,
                                        const uint32_t* flags, const void* ptr) {
  hsa_status_t status = rocprofiler::hsa_support::GetAmdExtTable().hsa_amd_agents_allow_access_fn(
      num_agents, agents, flags, ptr);
  if (status != HSA_STATUS_SUCCESS) return status;

  if (IsEnabled(ACTIVITY_DOMAIN_HSA_EVT, HSA_EVT_ID_DEVICE)) {
    while (num_agents--) {
      hsa_agent_t agent = *agents++;
      auto it = agent_info_map.find(agent.handle);
      if (it == agent_info_map.end()) rocprofiler::fatal("agent was not found in the agent_info map");

      hsa_evt_data_t data{};
      data.device.type = it->second.type;
      data.device.id = it->second.id;
      data.device.agent = agent;
      data.device.ptr = ptr;

      ReportActivity(ACTIVITY_DOMAIN_HSA_EVT, HSA_EVT_ID_DEVICE, &data);
    }
  }
  return HSA_STATUS_SUCCESS;
}

struct CodeObjectCallbackArg {
  activity_rtapi_callback_t callback_fun;
  void* callback_arg;
  bool unload;
};

hsa_status_t CodeObjectCallback(hsa_executable_t executable,
                                hsa_loaded_code_object_t loaded_code_object, void* arg) {
  hsa_evt_data_t data{};

  if (rocprofiler::hsa_support::GetHSALoaderApi().hsa_ven_amd_loader_loaded_code_object_get_info(
          loaded_code_object, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_TYPE,
          &data.codeobj.storage_type) != HSA_STATUS_SUCCESS)
    rocprofiler::fatal("hsa_ven_amd_loader_loaded_code_object_get_info failed");

  if (data.codeobj.storage_type == HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_FILE) {
    if (rocprofiler::hsa_support::GetHSALoaderApi().hsa_ven_amd_loader_loaded_code_object_get_info(
            loaded_code_object, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_FILE,
            &data.codeobj.storage_file) != HSA_STATUS_SUCCESS ||
        data.codeobj.storage_file == -1)
      rocprofiler::fatal("hsa_ven_amd_loader_loaded_code_object_get_info failed");
    data.codeobj.memory_base = data.codeobj.memory_size = 0;
  } else if (data.codeobj.storage_type == HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_MEMORY) {
    if (rocprofiler::hsa_support::GetHSALoaderApi().hsa_ven_amd_loader_loaded_code_object_get_info(
            loaded_code_object,
            HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_BASE,
            &data.codeobj.memory_base) != HSA_STATUS_SUCCESS ||
        rocprofiler::hsa_support::GetHSALoaderApi().hsa_ven_amd_loader_loaded_code_object_get_info(
            loaded_code_object,
            HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_SIZE,
            &data.codeobj.memory_size) != HSA_STATUS_SUCCESS)
      rocprofiler::fatal("hsa_ven_amd_loader_loaded_code_object_get_info failed");
    data.codeobj.storage_file = -1;
  } else if (data.codeobj.storage_type == HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_NONE) {
    return HSA_STATUS_SUCCESS;  // FIXME: do we really not care about these
                                // code objects?
  } else {
    rocprofiler::fatal("unknown code object storage type: %d", data.codeobj.storage_type);
  }

  if (rocprofiler::hsa_support::GetHSALoaderApi().hsa_ven_amd_loader_loaded_code_object_get_info(
          loaded_code_object, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE,
          &data.codeobj.load_base) != HSA_STATUS_SUCCESS ||
      rocprofiler::hsa_support::GetHSALoaderApi().hsa_ven_amd_loader_loaded_code_object_get_info(
          loaded_code_object, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_SIZE,
          &data.codeobj.load_size) != HSA_STATUS_SUCCESS ||
      rocprofiler::hsa_support::GetHSALoaderApi().hsa_ven_amd_loader_loaded_code_object_get_info(
          loaded_code_object, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_DELTA,
          &data.codeobj.load_delta) != HSA_STATUS_SUCCESS)
    rocprofiler::fatal("hsa_ven_amd_loader_loaded_code_object_get_info failed");

  if (rocprofiler::hsa_support::GetHSALoaderApi().hsa_ven_amd_loader_loaded_code_object_get_info(
          loaded_code_object, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI_LENGTH,
          &data.codeobj.uri_length) != HSA_STATUS_SUCCESS)
    rocprofiler::fatal("hsa_ven_amd_loader_loaded_code_object_get_info failed");

  std::string uri_str(data.codeobj.uri_length, '\0');
  if (rocprofiler::hsa_support::GetHSALoaderApi().hsa_ven_amd_loader_loaded_code_object_get_info(
          loaded_code_object, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI, uri_str.data()) !=
      HSA_STATUS_SUCCESS)
    rocprofiler::fatal("hsa_ven_amd_loader_loaded_code_object_get_info failed");

  if (rocprofiler::hsa_support::GetHSALoaderApi().hsa_ven_amd_loader_loaded_code_object_get_info(
          loaded_code_object, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_AGENT,
          &data.codeobj.agent) != HSA_STATUS_SUCCESS)
    rocprofiler::fatal("hsa_ven_amd_loader_loaded_code_object_get_info failed");

  data.codeobj.uri = uri_str.c_str();
  data.codeobj.unload = *static_cast<bool*>(arg) ? 1 : 0;
  ReportActivity(ACTIVITY_DOMAIN_HSA_EVT, HSA_EVT_ID_CODEOBJ, &data);

  hsa_executable_iterate_agent_symbols(executable, data.codeobj.agent,
                                       hsa_executable_iteration_callback, &(data.codeobj.unload));

  return HSA_STATUS_SUCCESS;
}

hsa_status_t ExecutableFreezeIntercept(hsa_executable_t executable, const char* options) {
  hsa_status_t status =
      rocprofiler::hsa_support::GetCoreApiTable().hsa_executable_freeze_fn(executable, options);
  if (status != HSA_STATUS_SUCCESS) return status;

  // if (IsEnabled(ACTIVITY_DOMAIN_HSA_EVT, HSA_EVT_ID_CODEOBJ)) {
  bool unload = false;
  rocprofiler::hsa_support::GetHSALoaderApi()
      .hsa_ven_amd_loader_executable_iterate_loaded_code_objects(executable, CodeObjectCallback,
                                                                 &unload);
  // }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t ExecutableDestroyIntercept(hsa_executable_t executable) {
  // if (IsEnabled(ACTIVITY_DOMAIN_HSA_EVT, HSA_EVT_ID_CODEOBJ)) {
  bool unload = true;
  rocprofiler::hsa_support::GetHSALoaderApi()
      .hsa_ven_amd_loader_executable_iterate_loaded_code_objects(executable, CodeObjectCallback,
                                                                 &unload);
  // }

  return rocprofiler::hsa_support::GetCoreApiTable().hsa_executable_destroy_fn(executable);
}

std::atomic<bool> profiling_async_copy_enable{false};

hsa_status_t ProfilingAsyncCopyEnableIntercept(bool enable) {
  hsa_status_t status =
      rocprofiler::hsa_support::GetAmdExtTable().hsa_amd_profiling_async_copy_enable_fn(enable);
  if (status == HSA_STATUS_SUCCESS) {
    profiling_async_copy_enable.exchange(enable, std::memory_order_release);
  }
  return status;
}

void MemoryASyncCopyHandler(const Tracker::entry_t* entry) {
  activity_record_t record{};
  record.domain = ACTIVITY_DOMAIN_HSA_OPS;
  record.op = HSA_OP_ID_COPY;
  record.begin_ns = entry->begin;
  record.end_ns = entry->end;
  record.device_id = 0;
  record.correlation_id = entry->correlation_id;
  ReportActivity(ACTIVITY_DOMAIN_HSA_OPS, HSA_OP_ID_COPY, &record);
}

hsa_status_t MemoryASyncCopyIntercept(void* dst, hsa_agent_t dst_agent, const void* src,
                                      hsa_agent_t src_agent, size_t size, uint32_t num_dep_signals,
                                      const hsa_signal_t* dep_signals,
                                      hsa_signal_t completion_signal) {
  bool is_enabled = IsEnabled(ACTIVITY_DOMAIN_HSA_OPS, HSA_OP_ID_COPY);

  // FIXME: what happens if the state changes before returning?
  [[maybe_unused]] hsa_status_t status =
      rocprofiler::hsa_support::GetAmdExtTable().hsa_amd_profiling_async_copy_enable_fn(
          profiling_async_copy_enable.load(std::memory_order_relaxed) || is_enabled);
  assert(status == HSA_STATUS_SUCCESS && "hsa_amd_profiling_async_copy_enable failed");

  if (!is_enabled) {
    return rocprofiler::hsa_support::GetAmdExtTable().hsa_amd_memory_async_copy_fn(
        dst, dst_agent, src, src_agent, size, num_dep_signals, dep_signals, completion_signal);
  }

  Tracker::entry_t* entry = new Tracker::entry_t();
  entry->handler = MemoryASyncCopyHandler;
  entry->correlation_id = CorrelationId();
  Tracker::Enable(Tracker::COPY_ENTRY_TYPE, hsa_agent_t{}, completion_signal, entry);

  status = rocprofiler::hsa_support::GetAmdExtTable().hsa_amd_memory_async_copy_fn(
      dst, dst_agent, src, src_agent, size, num_dep_signals, dep_signals, entry->signal);
  if (status != HSA_STATUS_SUCCESS) Tracker::Disable(entry);

  return status;
}

hsa_status_t MemoryASyncCopyRectIntercept(const hsa_pitched_ptr_t* dst,
                                          const hsa_dim3_t* dst_offset,
                                          const hsa_pitched_ptr_t* src,
                                          const hsa_dim3_t* src_offset, const hsa_dim3_t* range,
                                          hsa_agent_t copy_agent, hsa_amd_copy_direction_t dir,
                                          uint32_t num_dep_signals, const hsa_signal_t* dep_signals,
                                          hsa_signal_t completion_signal) {
  bool is_enabled = IsEnabled(ACTIVITY_DOMAIN_HSA_OPS, HSA_OP_ID_COPY);

  // FIXME: what happens if the state changes before returning?
  [[maybe_unused]] hsa_status_t status =
      rocprofiler::hsa_support::GetAmdExtTable().hsa_amd_profiling_async_copy_enable_fn(
          profiling_async_copy_enable.load(std::memory_order_relaxed) || is_enabled);
  assert(status == HSA_STATUS_SUCCESS && "hsa_amd_profiling_async_copy_enable failed");

  if (!is_enabled) {
    return rocprofiler::hsa_support::GetAmdExtTable().hsa_amd_memory_async_copy_rect_fn(
        dst, dst_offset, src, src_offset, range, copy_agent, dir, num_dep_signals, dep_signals,
        completion_signal);
  }

  Tracker::entry_t* entry = new Tracker::entry_t();
  entry->handler = MemoryASyncCopyHandler;
  entry->correlation_id = CorrelationId();
  Tracker::Enable(Tracker::COPY_ENTRY_TYPE, hsa_agent_t{}, completion_signal, entry);

  status = rocprofiler::hsa_support::GetAmdExtTable().hsa_amd_memory_async_copy_rect_fn(
      dst, dst_offset, src, src_offset, range, copy_agent, dir, num_dep_signals, dep_signals,
      entry->signal);
  if (status != HSA_STATUS_SUCCESS) Tracker::Disable(entry);

  return status;
}

hsa_status_t MemoryASyncCopyOnEngineIntercept(
    void* dst, hsa_agent_t dst_agent, const void* src, hsa_agent_t src_agent, size_t size,
    uint32_t num_dep_signals, const hsa_signal_t* dep_signals, hsa_signal_t completion_signal,
    hsa_amd_sdma_engine_id_t engine_id, bool force_copy_on_sdma) {
  bool is_enabled = IsEnabled(ACTIVITY_DOMAIN_HSA_OPS, HSA_OP_ID_COPY);

  // FIXME: what happens if the state changes before returning?
  [[maybe_unused]] hsa_status_t status = saved_amd_ext_api.hsa_amd_profiling_async_copy_enable_fn(
      profiling_async_copy_enable.load(std::memory_order_relaxed) || is_enabled);
  assert(status == HSA_STATUS_SUCCESS && "hsa_amd_profiling_async_copy_enable failed");

  if (!is_enabled) {
    return saved_amd_ext_api.hsa_amd_memory_async_copy_on_engine_fn(
        dst, dst_agent, src, src_agent, size, num_dep_signals, dep_signals, completion_signal,
        engine_id, force_copy_on_sdma);
  }

  Tracker::entry_t* entry = new Tracker::entry_t();
  entry->handler = MemoryASyncCopyHandler;
  entry->correlation_id = CorrelationId();
  Tracker::Enable(Tracker::COPY_ENTRY_TYPE, hsa_agent_t{}, completion_signal, entry);

  status = saved_amd_ext_api.hsa_amd_memory_async_copy_on_engine_fn(
      dst, dst_agent, src, src_agent, size, num_dep_signals, dep_signals, entry->signal, engine_id,
      force_copy_on_sdma);
  if (status != HSA_STATUS_SUCCESS) Tracker::Disable(entry);

  return status;
}

}  // namespace

rocprofiler_timestamp_t timestamp_ns() {
  // If the HSA intercept is installed, then use the "original"
  // 'hsa_system_get_info' function to avoid reporting calls for internal use
  // of the HSA API by the tracer.
  auto hsa_system_get_info_fn = rocprofiler::hsa_support::GetCoreApiTable().hsa_system_get_info_fn;

  // If the HSA intercept is not installed, use the default
  // 'hsa_system_get_info'.
  if (hsa_system_get_info_fn == nullptr) hsa_system_get_info_fn = hsa_system_get_info;

  uint64_t sysclock;
  if (hsa_status_t status = hsa_system_get_info_fn(HSA_SYSTEM_INFO_TIMESTAMP, &sysclock);
      status == HSA_STATUS_ERROR_NOT_INITIALIZED)
    return rocprofiler_timestamp_t{0};
  else if (status != HSA_STATUS_SUCCESS)
    rocprofiler::fatal("hsa_system_get_info failed");

  static uint64_t sysclock_period = [&]() {
    uint64_t sysclock_hz = 0;
    if (hsa_status_t status =
            hsa_system_get_info_fn(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &sysclock_hz);
        status != HSA_STATUS_SUCCESS)
      rocprofiler::fatal("hsa_system_get_info failed");

    return (uint64_t)1000000000 / sysclock_hz;
  }();

  return rocprofiler_timestamp_t{sysclock * sysclock_period};
}

void Initialize_roctracer(HsaApiTable* table) {
  // Save the HSA core api and amd_ext api.
  saved_core_api = rocprofiler::hsa_support::GetCoreApiTable();
  saved_amd_ext_api = rocprofiler::hsa_support::GetAmdExtTable();
  hsa_loader_api = rocprofiler::hsa_support::GetHSALoaderApi();

  // Enumerate the agents.
  if (rocprofiler::hsa_support::GetCoreApiTable().hsa_iterate_agents_fn(
          [](hsa_agent_t agent, void* data) {
            hsa_support::AgentInfo agent_info;
            if (rocprofiler::hsa_support::GetCoreApiTable().hsa_agent_get_info_fn(
                    agent, HSA_AGENT_INFO_DEVICE, &agent_info.type) != HSA_STATUS_SUCCESS)
              rocprofiler::fatal("hsa_agent_get_info failed");
            switch (agent_info.type) {
              case HSA_DEVICE_TYPE_CPU:
                static int cpu_agent_count = 0;
                agent_info.id = cpu_agent_count++;
                break;
              case HSA_DEVICE_TYPE_GPU: {
                uint32_t driver_node_id;
                if (rocprofiler::hsa_support::GetCoreApiTable().hsa_agent_get_info_fn(
                        agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_DRIVER_NODE_ID),
                        &driver_node_id) != HSA_STATUS_SUCCESS)
                  rocprofiler::fatal("hsa_agent_get_info failed");

                agent_info.id = driver_node_id;
              } break;
              default:
                static int other_agent_count = 0;
                agent_info.id = other_agent_count++;
                break;
            }
            hsa_support::agent_info_map.emplace(agent.handle, agent_info);
            return HSA_STATUS_SUCCESS;
          },
          nullptr) != HSA_STATUS_SUCCESS)
    rocprofiler::fatal("hsa_iterate_agents failed");
}

const char* GetApiName(uint32_t id) { return detail::GetApiName(id); }

const char* GetEvtName(uint32_t id) {
  switch (id) {
    case HSA_EVT_ID_ALLOCATE:
      return "ALLOCATE";
    case HSA_EVT_ID_DEVICE:
      return "DEVICE";
    case HSA_EVT_ID_MEMCOPY:
      return "MEMCOPY";
    case HSA_EVT_ID_SUBMIT:
      return "SUBMIT";
    case HSA_EVT_ID_KSYMBOL:
      return "KSYMBOL";
    case HSA_EVT_ID_CODEOBJ:
      return "CODEOBJ";
    case HSA_EVT_ID_NUMBER:
      break;
  }
  throw ApiError(ROCTRACER_STATUS_ERROR_INVALID_ARGUMENT, "invalid HSA EVT callback id");
}

const char* GetOpsName(uint32_t id) {
  switch (id) {
    case HSA_OP_ID_DISPATCH:
      return "DISPATCH";
    case HSA_OP_ID_COPY:
      return "COPY";
    case HSA_OP_ID_BARRIER:
      return "BARRIER";
    case HSA_OP_ID_RESERVED1:
      return "PCSAMPLE";
  }
  throw ApiError(ROCTRACER_STATUS_ERROR_INVALID_ARGUMENT, "invalid HSA OPS callback id");
}

uint32_t GetApiCode(const char* str) { return detail::GetApiCode(str); }

void RegisterTracerCallback(int (*function)(activity_domain_t domain, uint32_t operation_id,
                                            void* data)) {
  report_activity.store(function, std::memory_order_relaxed);
}

}  // namespace roctracer::hsa_support

namespace rocprofiler {

namespace hsa_support {

hsa_agent_t cpu_agent;
std::map<uint32_t, std::unique_ptr<queue::Queue>> queues;
std::atomic<uint32_t> active_queues{0};

/**
 * @brief This function is a queue create interceptor. It intercepts the queue
 * creation, registers the profiler, and registers a packet write interceptor.
 * It also creates a Queue Interceptor object to store the
 * newly created queue information.
 **/
hsa_status_t QueueCreateInterceptor(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
                                    void (*callback)(hsa_status_t status, hsa_queue_t* source,
                                                     void* data),
                                    void* data, uint32_t private_segment_size,
                                    uint32_t group_segment_size, hsa_queue_t** queue) {
  // TODO(aelwazir): Queue ID
  static std::mutex qc_mutex;
  std::lock_guard<std::mutex> lk(qc_mutex);
  queues.emplace(active_queues.fetch_add(1, std::memory_order_release),
                 std::make_unique<queue::Queue>(cpu_agent, agent, size, type, callback, data,
                                                private_segment_size, group_segment_size, queue));

  return HSA_STATUS_SUCCESS;
}

/**
 * @brief This function is a queue destroy interceptor. It intercepts the queue
 * destroy. It deletes the queue entry from the template storage and calls the
 * hsa_queue_destroy_fn.
 **/

hsa_status_t QueueDestroyInterceptor(hsa_queue_t* hsa_queue) {
  static std::mutex qd_mutex;
  std::lock_guard<std::mutex> lk(qd_mutex);
  ASSERTM(GetCoreApiTable().hsa_queue_destroy_fn(hsa_queue) == HSA_STATUS_SUCCESS,
          "Queue couldn't be destroyed!");
  queues.erase(active_queues.fetch_sub(1, std::memory_order_release));
  return HSA_STATUS_SUCCESS;
}

std::unordered_map<uint32_t, hsa_agent_t> numa_node_to_cpu_agent;
std::unordered_map<long long, long long> gpu_numa_nodes_near_cpu;
std::vector<hsa_agent_t> gpu_agents;

void Initialize(HsaApiTable* table) {
  InitKsymbols();
  // Save the HSA core api and amd_ext api.
  long long gpu_numa_nodes_start = 0;

  SetCoreApiTable(*table->core_);
  SetAmdExtTable(table->amd_ext_);

  // TODO(aelwazir): FIXME, this is a workaround for the issue of allocating buffers on KernArg
  // Pools that are nearest to the GPU which is not NUMA local to the CPU. This should be remove
  // once ROCR provides such API.
  std::string path = "/sys/class/kfd/kfd/topology/nodes";
  for (const auto& entry : fs::directory_iterator(path)) {
    long long node_id = std::stoll(entry.path().filename().c_str());
    std::ifstream gpu_id_file;
    std::string gpu_path = entry.path().c_str();
    gpu_path += "/gpu_id";
    gpu_id_file.open(gpu_path);
    std::string gpu_id_str;
    if (gpu_id_file.is_open()) {
      gpu_id_file >> gpu_id_str;
      long long gpu_id = std::stoll(gpu_id_str);
      if (gpu_id > 0) {
        gpu_numa_nodes_start = (gpu_numa_nodes_start > node_id || gpu_numa_nodes_start == 0)
            ? node_id
            : gpu_numa_nodes_start;
      }
    }
    gpu_id_file.close();
  }
  path = "/sys/class/kfd/kfd/topology/nodes";
  for (const auto& entry : fs::directory_iterator(path)) {
    long long node_id = std::stoll(entry.path().filename().c_str());
    std::string numa_node_path = entry.path().c_str();
    long long agent_id = std::stoll(entry.path().filename().c_str());
    if (agent_id >= gpu_numa_nodes_start) {
      numa_node_path += "/io_links";
      for (const auto& numa_node_entry : fs::directory_iterator(numa_node_path)) {
        std::string numa_node_entry_properties_path = numa_node_entry.path().c_str();
        numa_node_entry_properties_path += "/properties";
        std::ifstream gpu_properties_file;
        gpu_properties_file.open(numa_node_entry_properties_path);
        std::string gpu_properties_file_line;
        if (gpu_properties_file.is_open()) {
          while (gpu_properties_file) {
            std::getline(gpu_properties_file, gpu_properties_file_line);
            std::string delimiter = " ";
            std::stringstream ss(gpu_properties_file_line);
            std::string word;
            ss >> word;
            if (word.compare("node_to") == 0) {
              ss >> word;
              long long near_cpu_node_id = std::stoll(word);
              if (near_cpu_node_id < gpu_numa_nodes_start) {
                gpu_numa_nodes_near_cpu[node_id] = near_cpu_node_id;
              }
            }
          }
        }
        gpu_properties_file.close();
      }
    }
  }

  // Enumerate the agents.
  if (GetCoreApiTable().hsa_iterate_agents_fn(
          [](hsa_agent_t agent, void* data) {
            Agent::AgentInfo agent_info{agent, &GetCoreApiTable()};
            static int cpu_agent_count = 0;
            static int other_agent_count = 0;
            switch (agent_info.getType()) {
              case HSA_DEVICE_TYPE_CPU:
                agent_info.setIndex(cpu_agent_count++);
                cpu_agent = agent;
                rocprofiler::queue::InitializePools(cpu_agent, &agent_info);
                uint32_t cpu_numa_node_id;
                if (GetCoreApiTable().hsa_agent_get_info_fn(
                        agent, HSA_AGENT_INFO_NODE, &cpu_numa_node_id) != HSA_STATUS_SUCCESS)
                  rocprofiler::fatal("hsa_agent_get_info(HSA_AGENT_INFO_NODE) failed");
                agent_info.setNumaNode(cpu_numa_node_id);
                numa_node_to_cpu_agent[cpu_numa_node_id] = agent;
                break;
              case HSA_DEVICE_TYPE_GPU:
                // TODO(FIXME): When multiple ranks are used, each rank's first
                // logical device always has GPU ID 0, regardless of which
                // physical device is selected with CUDA_VISIBLE_DEVICES.
                // Because of this, when merging traces from multiple ranks,
                // GPU IDs from different processes may overlap.
                //
                // The long term solution is to use KFD's gpu_id, which is
                // stable across APIs and processes, but it isn't currently
                // exposed by ROCr.  We could use the agent's
                // HSA_AMD_AGENT_INFO_DRIVER_NODE_ID in the meantime, as even
                // that would be an improvement--it's what legacy roctracer
                // is currently doing as well as the roctracer compatibility
                // code earlier in this file.
                uint32_t driver_node_id;
                if (rocprofiler::hsa_support::GetCoreApiTable().hsa_agent_get_info_fn(
                        agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_DRIVER_NODE_ID),
                        &driver_node_id) != HSA_STATUS_SUCCESS)
                  rocprofiler::fatal("hsa_agent_get_info failed");
                agent_info.setIndex(driver_node_id);
                uint32_t gpu_cpu_numa_node_id;
                if (GetCoreApiTable().hsa_agent_get_info_fn(
                        agent, HSA_AGENT_INFO_NODE, &gpu_cpu_numa_node_id) != HSA_STATUS_SUCCESS)
                  rocprofiler::fatal("hsa_agent_get_info(HSA_AGENT_INFO_NODE) failed");
                agent_info.setNumaNode(gpu_cpu_numa_node_id);
                agent_info.setNearCpuAgent(
                    numa_node_to_cpu_agent[gpu_numa_nodes_near_cpu[gpu_cpu_numa_node_id]]);
                rocprofiler::queue::InitializeGPUPool(agent, &agent_info);
                gpu_agents.push_back(agent);
                break;
              default:
                agent_info.setIndex(other_agent_count++);
                break;
            }
            SetAgentInfo(agent.handle, agent_info);
            return HSA_STATUS_SUCCESS;
          },
          nullptr) != HSA_STATUS_SUCCESS)
    rocprofiler::fatal("hsa_iterate_agents failed");

  for (auto& agent : gpu_agents) {
    GetAgentInfo(agent.handle).cpu_pool =
        GetAgentInfo(GetAgentInfo(agent.handle).getNearCpuAgent().handle).cpu_pool;
    GetAgentInfo(agent.handle).kernarg_pool =
        GetAgentInfo(GetAgentInfo(agent.handle).getNearCpuAgent().handle).kernarg_pool;
  }

  rocprofiler::queue::CheckPacketReqiurements(gpu_agents);

  gpu_agents.clear();
  numa_node_to_cpu_agent.clear();
  gpu_numa_nodes_near_cpu.clear();

  SetHSALoaderApi();

  roctracer::hsa_support::Initialize_roctracer(table);

  // Install the Queue intercept
  table->core_->hsa_queue_create_fn = QueueCreateInterceptor;
  table->core_->hsa_queue_destroy_fn = QueueDestroyInterceptor;

  // Install the HSA_OPS intercept
  table->amd_ext_->hsa_amd_memory_async_copy_fn = roctracer::hsa_support::MemoryASyncCopyIntercept;
  table->amd_ext_->hsa_amd_memory_async_copy_rect_fn =
      roctracer::hsa_support::MemoryASyncCopyRectIntercept;
  table->amd_ext_->hsa_amd_profiling_async_copy_enable_fn =
      roctracer::hsa_support::ProfilingAsyncCopyEnableIntercept;
  table->amd_ext_->hsa_amd_memory_async_copy_on_engine_fn =
      roctracer::hsa_support::MemoryASyncCopyOnEngineIntercept;

  // Install the HSA_EVT intercept
  table->core_->hsa_memory_allocate_fn = roctracer::hsa_support::MemoryAllocateIntercept;
  table->core_->hsa_memory_assign_agent_fn = roctracer::hsa_support::MemoryAssignAgentIntercept;
  table->core_->hsa_memory_copy_fn = roctracer::hsa_support::MemoryCopyIntercept;
  table->amd_ext_->hsa_amd_memory_pool_allocate_fn =
      roctracer::hsa_support::MemoryPoolAllocateIntercept;
  table->amd_ext_->hsa_amd_memory_pool_free_fn = roctracer::hsa_support::MemoryPoolFreeIntercept;
  table->amd_ext_->hsa_amd_agents_allow_access_fn =
      roctracer::hsa_support::AgentsAllowAccessIntercept;
  table->core_->hsa_executable_freeze_fn = roctracer::hsa_support::ExecutableFreezeIntercept;
  table->core_->hsa_executable_destroy_fn = roctracer::hsa_support::ExecutableDestroyIntercept;

  // Install the HSA_API wrappers
  roctracer::hsa_support::detail::InstallCoreApiWrappers(table->core_);
  roctracer::hsa_support::detail::InstallAmdExtWrappers(table->amd_ext_);
  roctracer::hsa_support::detail::InstallImageExtWrappers(table->image_ext_);
}

void Finalize() {
  while (active_queues.load(std::memory_order_relaxed) != 0) {
  }

  // FinitKsymbols();
  ResetMaps();
}

static std::map<uint64_t, rocprofiler::MetricsDict*> metricsDicts;

bool IterateCounters(rocprofiler_counters_info_callback_t counters_info_callback) {
  if (GetCoreApiTable().hsa_iterate_agents_fn(
          [](hsa_agent_t agent, void* data) {
            Agent::AgentInfo agent_info{agent, &GetCoreApiTable()};
            if (agent_info.getType() == HSA_DEVICE_TYPE_GPU) {
              metricsDicts.emplace(agent.handle, rocprofiler::MetricsDict::Create(&agent_info));
            }
            return HSA_STATUS_SUCCESS;
          },
          nullptr) != HSA_STATUS_SUCCESS)
    rocprofiler::fatal("hsa_iterate_agents failed");
  uint32_t gpu_counter = 0;
  for (auto metricsDictAgent : metricsDicts) {
    rocprofiler::MetricsDict* metricsDict = metricsDictAgent.second;
    std::string gpu_name = metricsDict->GetAgentName();
    auto nodes_vec = metricsDict->GetNodes();

    for (auto* node : nodes_vec) {
      const std::string& name = node->opts["name"];
      const std::string& descr = node->opts["descr"];
      const std::string& expr = node->opts["expr"];
      // Getting the block name
      const std::string block_name = node->opts["block"];
      uint32_t block_counters = 0;

      hsa_ven_amd_aqlprofile_id_query_t query;

      if (expr.empty()) {
        // Querying profile
        hsa_ven_amd_aqlprofile_profile_t profile = {};
        profile.agent = hsa_agent_t{metricsDictAgent.first};
        profile.type = HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_PMC;

        // Query block id info
        query = {block_name.c_str(), 0, 0};
        hsa_status_t status =
            hsa_ven_amd_aqlprofile_get_info(&profile, HSA_VEN_AMD_AQLPROFILE_INFO_BLOCK_ID, &query);
        if (status != HSA_STATUS_SUCCESS)
          AQL_EXC_RAISING(HSA_STATUS_ERROR, "get block id info: '" << block_name << "'");

        // Metric object
        const std::string metric_name = (query.instance_count > 1) ? name + "[0]" : name;
        const Metric* metric = metricsDict->Get(metric_name);
        if (metric == NULL) EXC_RAISING(HSA_STATUS_ERROR, "metric '" << name << "' is not found");

        // Process metrics counters
        const counters_vec_t& counters_vec = metric->GetCounters();
        if (counters_vec.size() != 1)
          EXC_RAISING(HSA_STATUS_ERROR, "error: '" << metric->GetName() << "' is not basic");

        // Query block counters number
        profile.events = &(counters_vec[0]->event);
        status = hsa_ven_amd_aqlprofile_get_info(
            &profile, HSA_VEN_AMD_AQLPROFILE_INFO_BLOCK_COUNTERS, &block_counters);
        if (status != HSA_STATUS_SUCCESS) continue;
      }

      const rocprofiler_counter_info_t counter_info =
          rocprofiler_counter_info_t{strdup(name.c_str()),
                                     strdup(descr.c_str()),
                                     expr.empty() ? nullptr : strdup(expr.c_str()),
                                     query.instance_count,
                                     block_name.c_str(),
                                     block_counters};
      counters_info_callback(counter_info, gpu_name.c_str(), gpu_counter);
    }
    gpu_counter++;

    // auto start = metricsDict->Begin();
    // while (start != metricsDict->End()) {
    //   const xml::Expr* expr = start->second->GetExpr();
    //   std::string expr_str;
    //   if (expr) expr_str = expr->GetStr().c_str();
    //   const rocprofiler_counter_info_t counter_info =
    //       rocprofiler_counter_info_t{start->first.c_str(), "", expr ? expr_str.c_str() :
    //       nullptr};
    //   counters_info_callback(counter_info, gpu_name.c_str(), gpu_counter);
    //   start++;
    // }
  }

  return true;
}

}  // namespace hsa_support
}  // namespace rocprofiler
