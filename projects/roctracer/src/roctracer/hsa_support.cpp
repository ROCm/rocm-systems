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

#include "correlation_id.h"
#include "debug.h"
#include "exception.h"
#include "memory_pool.h"
#include "roctracer.h"
#include "roctracer_hsa.h"

#include <atomic>
#include <hsa/hsa.h>
#include <hsa/amd_hsa_signal.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <unordered_map>
#include <optional>
#include <mutex>

namespace {

std::atomic<int (*)(activity_domain_t domain, uint32_t operation_id, void* data)> report_activity;

bool IsEnabled(activity_domain_t domain, uint32_t operation_id) {
  auto report = report_activity.load(std::memory_order_relaxed);
  return report && report(domain, operation_id, nullptr) == 0;
}

void ReportActivity(activity_domain_t domain, uint32_t operation_id, void* data) {
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
    status = saved_core_api.hsa_signal_create_fn(1, 0, NULL, &(entry->signal));
    if (status != HSA_STATUS_SUCCESS) fatal("hsa_signal_create failed");
    status = saved_amd_ext_api.hsa_amd_signal_async_handler_fn(
        entry->signal, HSA_SIGNAL_CONDITION_LT, 1, Handler, entry);
    if (status != HSA_STATUS_SUCCESS) fatal("hsa_amd_signal_async_handler failed");
  }

  // Delete tracker entry
  inline static void Disable(entry_t* entry) {
    saved_core_api.hsa_signal_destroy_fn(entry->signal);
    entry->valid.store(ENTRY_INV, std::memory_order_release);
  }

 private:
  // Entry completion
  inline static void Complete(hsa_signal_value_t signal_value, entry_t* entry) {
    static roctracer_timestamp_t sysclock_period = []() {
      uint64_t sysclock_hz = 0;
      hsa_status_t status =
          saved_core_api.hsa_system_get_info_fn(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &sysclock_hz);
      if (status != HSA_STATUS_SUCCESS) fatal("hsa_system_get_info failed");
      return (uint64_t)1000000000 / sysclock_hz;
    }();

    if (entry->type == COPY_ENTRY_TYPE) {
      hsa_amd_profiling_async_copy_time_t async_copy_time{};
      hsa_status_t status = saved_amd_ext_api.hsa_amd_profiling_get_async_copy_time_fn(
          entry->signal, &async_copy_time);
      if (status != HSA_STATUS_SUCCESS) fatal("hsa_amd_profiling_get_async_copy_time failed");
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
          saved_core_api.hsa_signal_load_relaxed_fn(orig) - 1;
      assert(signal_value == new_value && "Tracker::Complete bad signal value");
      saved_core_api.hsa_signal_store_screlease_fn(orig, signal_value);
    }
    saved_core_api.hsa_signal_destroy_fn(signal);
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
  hsa_status_t status = saved_core_api.hsa_memory_allocate_fn(region, size, ptr);
  if (status != HSA_STATUS_SUCCESS) return status;

  if (IsEnabled(ACTIVITY_DOMAIN_HSA_EVT, HSA_EVT_ID_ALLOCATE)) {
    hsa_evt_data_t data{};
    data.allocate.ptr = *ptr;
    data.allocate.size = size;
    if (saved_core_api.hsa_region_get_info_fn(region, HSA_REGION_INFO_SEGMENT,
                                              &data.allocate.segment) != HSA_STATUS_SUCCESS ||
        saved_core_api.hsa_region_get_info_fn(region, HSA_REGION_INFO_GLOBAL_FLAGS,
                                              &data.allocate.global_flag) != HSA_STATUS_SUCCESS)
      fatal("hsa_region_get_info failed");

    ReportActivity(ACTIVITY_DOMAIN_HSA_EVT, HSA_EVT_ID_ALLOCATE, &data);
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t MemoryAssignAgentIntercept(void* ptr, hsa_agent_t agent,
                                        hsa_access_permission_t access) {
  hsa_status_t status = saved_core_api.hsa_memory_assign_agent_fn(ptr, agent, access);
  if (status != HSA_STATUS_SUCCESS) return status;

  if (IsEnabled(ACTIVITY_DOMAIN_HSA_EVT, HSA_EVT_ID_DEVICE)) {
    hsa_evt_data_t data{};
    data.device.ptr = ptr;
    if (saved_core_api.hsa_agent_get_info_fn(agent, HSA_AGENT_INFO_DEVICE, &data.device.type) !=
        HSA_STATUS_SUCCESS)
      fatal("hsa_agent_get_info failed");

    ReportActivity(ACTIVITY_DOMAIN_HSA_EVT, HSA_EVT_ID_DEVICE, &data);
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t MemoryCopyIntercept(void* dst, const void* src, size_t size) {
  hsa_status_t status = saved_core_api.hsa_memory_copy_fn(dst, src, size);
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
  hsa_status_t status = saved_amd_ext_api.hsa_amd_memory_pool_allocate_fn(pool, size, flags, ptr);
  if (size == 0 || status != HSA_STATUS_SUCCESS) return status;

  if (IsEnabled(ACTIVITY_DOMAIN_HSA_EVT, HSA_EVT_ID_ALLOCATE)) {
    hsa_evt_data_t data{};
    data.allocate.ptr = *ptr;
    data.allocate.size = size;

    if (saved_amd_ext_api.hsa_amd_memory_pool_get_info_fn(
            pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &data.allocate.segment) != HSA_STATUS_SUCCESS ||
        saved_amd_ext_api.hsa_amd_memory_pool_get_info_fn(
            pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &data.allocate.global_flag) !=
            HSA_STATUS_SUCCESS)
      fatal("hsa_region_get_info failed");

    ReportActivity(ACTIVITY_DOMAIN_HSA_EVT, HSA_EVT_ID_ALLOCATE, &data);
  }

  if (IsEnabled(ACTIVITY_DOMAIN_HSA_EVT, HSA_EVT_ID_DEVICE)) {
    auto callback_data = std::make_pair(pool, ptr);
    auto agent_callback = [](hsa_agent_t agent, void* iterate_agent_callback_data) {
      auto [pool, ptr] = *reinterpret_cast<decltype(callback_data)*>(iterate_agent_callback_data);

      if (hsa_amd_memory_pool_access_t value;
          saved_amd_ext_api.hsa_amd_agent_memory_pool_get_info_fn(
              agent, pool, HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &value) != HSA_STATUS_SUCCESS ||
          value != HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT)
        return HSA_STATUS_SUCCESS;

      auto it = agent_info_map.find(agent.handle);
      if (it == agent_info_map.end()) fatal("agent was not found in the agent_info map");

      hsa_evt_data_t data{};
      data.device.type = it->second.type;
      data.device.id = it->second.id;
      data.device.agent = agent;
      data.device.ptr = ptr;

      ReportActivity(ACTIVITY_DOMAIN_HSA_EVT, HSA_EVT_ID_DEVICE, &data);
      return HSA_STATUS_SUCCESS;
    };
    saved_core_api.hsa_iterate_agents_fn(agent_callback, &callback_data);
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

  return saved_amd_ext_api.hsa_amd_memory_pool_free_fn(ptr);
}

// Agent allow access callback 'hsa_amd_agents_allow_access'
hsa_status_t AgentsAllowAccessIntercept(uint32_t num_agents, const hsa_agent_t* agents,
                                        const uint32_t* flags, const void* ptr) {
  hsa_status_t status =
      saved_amd_ext_api.hsa_amd_agents_allow_access_fn(num_agents, agents, flags, ptr);
  if (status != HSA_STATUS_SUCCESS) return status;

  if (IsEnabled(ACTIVITY_DOMAIN_HSA_EVT, HSA_EVT_ID_DEVICE)) {
    while (num_agents--) {
      hsa_agent_t agent = *agents++;
      auto it = agent_info_map.find(agent.handle);
      if (it == agent_info_map.end()) fatal("agent was not found in the agent_info map");

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

  if (hsa_loader_api.hsa_ven_amd_loader_loaded_code_object_get_info(
          loaded_code_object, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_TYPE,
          &data.codeobj.storage_type) != HSA_STATUS_SUCCESS)
    fatal("hsa_ven_amd_loader_loaded_code_object_get_info failed");

  if (data.codeobj.storage_type == HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_FILE) {
    if (hsa_loader_api.hsa_ven_amd_loader_loaded_code_object_get_info(
            loaded_code_object, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_FILE,
            &data.codeobj.storage_file) != HSA_STATUS_SUCCESS ||
        data.codeobj.storage_file == -1)
      fatal("hsa_ven_amd_loader_loaded_code_object_get_info failed");
    data.codeobj.memory_base = data.codeobj.memory_size = 0;
  } else if (data.codeobj.storage_type == HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_MEMORY) {
    if (hsa_loader_api.hsa_ven_amd_loader_loaded_code_object_get_info(
            loaded_code_object,
            HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_BASE,
            &data.codeobj.memory_base) != HSA_STATUS_SUCCESS ||
        hsa_loader_api.hsa_ven_amd_loader_loaded_code_object_get_info(
            loaded_code_object,
            HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_SIZE,
            &data.codeobj.memory_size) != HSA_STATUS_SUCCESS)
      fatal("hsa_ven_amd_loader_loaded_code_object_get_info failed");
    data.codeobj.storage_file = -1;
  } else if (data.codeobj.storage_type == HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_NONE) {
    return HSA_STATUS_SUCCESS;  // FIXME: do we really not care about these code objects?
  } else {
    fatal("unknown code object storage type: %d", data.codeobj.storage_type);
  }

  if (hsa_loader_api.hsa_ven_amd_loader_loaded_code_object_get_info(
          loaded_code_object, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE,
          &data.codeobj.load_base) != HSA_STATUS_SUCCESS ||
      hsa_loader_api.hsa_ven_amd_loader_loaded_code_object_get_info(
          loaded_code_object, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_SIZE,
          &data.codeobj.load_size) != HSA_STATUS_SUCCESS ||
      hsa_loader_api.hsa_ven_amd_loader_loaded_code_object_get_info(
          loaded_code_object, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_DELTA,
          &data.codeobj.load_delta) != HSA_STATUS_SUCCESS)
    fatal("hsa_ven_amd_loader_loaded_code_object_get_info failed");

  if (hsa_loader_api.hsa_ven_amd_loader_loaded_code_object_get_info(
          loaded_code_object, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI_LENGTH,
          &data.codeobj.uri_length) != HSA_STATUS_SUCCESS)
    fatal("hsa_ven_amd_loader_loaded_code_object_get_info failed");

  std::string uri_str(data.codeobj.uri_length, '\0');
  if (hsa_loader_api.hsa_ven_amd_loader_loaded_code_object_get_info(
          loaded_code_object, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI, uri_str.data()) !=
      HSA_STATUS_SUCCESS)
    fatal("hsa_ven_amd_loader_loaded_code_object_get_info failed");

  data.codeobj.uri = uri_str.c_str();
  data.codeobj.unload = *static_cast<bool*>(arg) ? 1 : 0;
  ReportActivity(ACTIVITY_DOMAIN_HSA_EVT, HSA_EVT_ID_CODEOBJ, &data);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t ExecutableFreezeIntercept(hsa_executable_t executable, const char* options) {
  hsa_status_t status = saved_core_api.hsa_executable_freeze_fn(executable, options);
  if (status != HSA_STATUS_SUCCESS) return status;

  if (IsEnabled(ACTIVITY_DOMAIN_HSA_EVT, HSA_EVT_ID_CODEOBJ)) {
    bool unload = false;
    hsa_loader_api.hsa_ven_amd_loader_executable_iterate_loaded_code_objects(
        executable, CodeObjectCallback, &unload);
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t ExecutableDestroyIntercept(hsa_executable_t executable) {
  if (IsEnabled(ACTIVITY_DOMAIN_HSA_EVT, HSA_EVT_ID_CODEOBJ)) {
    bool unload = true;
    hsa_loader_api.hsa_ven_amd_loader_executable_iterate_loaded_code_objects(
        executable, CodeObjectCallback, &unload);
  }

  return saved_core_api.hsa_executable_destroy_fn(executable);
}

std::atomic<bool> profiling_async_copy_enable{false};

hsa_status_t ProfilingAsyncCopyEnableIntercept(bool enable) {
  hsa_status_t status = saved_amd_ext_api.hsa_amd_profiling_async_copy_enable_fn(enable);
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

hsa_status_t MemoryASyncCopyIntercept(void* dst, hsa_agent_t dst_agent, const void* src,
                                      hsa_agent_t src_agent, size_t size, uint32_t num_dep_signals,
                                      const hsa_signal_t* dep_signals,
                                      hsa_signal_t completion_signal) {
  bool is_enabled = IsEnabled(ACTIVITY_DOMAIN_HSA_OPS, HSA_OP_ID_COPY);

  // FIXME: what happens if the state changes before returning?
  [[maybe_unused]] hsa_status_t status = saved_amd_ext_api.hsa_amd_profiling_async_copy_enable_fn(
      profiling_async_copy_enable.load(std::memory_order_relaxed) || is_enabled);
  assert(status == HSA_STATUS_SUCCESS && "hsa_amd_profiling_async_copy_enable failed");

  if (!is_enabled) {
    return saved_amd_ext_api.hsa_amd_memory_async_copy_fn(
        dst, dst_agent, src, src_agent, size, num_dep_signals, dep_signals, completion_signal);
  }

  Tracker::entry_t* entry = new Tracker::entry_t();
  entry->handler = MemoryASyncCopyHandler;
  entry->correlation_id = CorrelationId();
  Tracker::Enable(Tracker::COPY_ENTRY_TYPE, hsa_agent_t{}, completion_signal, entry);

  status = saved_amd_ext_api.hsa_amd_memory_async_copy_fn(
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
  [[maybe_unused]] hsa_status_t status = saved_amd_ext_api.hsa_amd_profiling_async_copy_enable_fn(
      profiling_async_copy_enable.load(std::memory_order_relaxed) || is_enabled);
  assert(status == HSA_STATUS_SUCCESS && "hsa_amd_profiling_async_copy_enable failed");

  if (!is_enabled) {
    return saved_amd_ext_api.hsa_amd_memory_async_copy_rect_fn(
        dst, dst_offset, src, src_offset, range, copy_agent, dir, num_dep_signals, dep_signals,
        completion_signal);
  }

  Tracker::entry_t* entry = new Tracker::entry_t();
  entry->handler = MemoryASyncCopyHandler;
  entry->correlation_id = CorrelationId();
  Tracker::Enable(Tracker::COPY_ENTRY_TYPE, hsa_agent_t{}, completion_signal, entry);

  status = saved_amd_ext_api.hsa_amd_memory_async_copy_rect_fn(
      dst, dst_offset, src, src_offset, range, copy_agent, dir, num_dep_signals, dep_signals,
      entry->signal);
  if (status != HSA_STATUS_SUCCESS) Tracker::Disable(entry);

  return status;
}

}  // namespace

roctracer_timestamp_t timestamp_ns() {
  // If the HSA intercept is installed, then use the "original" 'hsa_system_get_info' function to
  // avoid reporting calls for internal use of the HSA API by the tracer.
  auto hsa_system_get_info_fn = saved_core_api.hsa_system_get_info_fn;

  // If the HSA intercept is not installed, use the default 'hsa_system_get_info'.
  if (hsa_system_get_info_fn == nullptr) hsa_system_get_info_fn = hsa_system_get_info;

  uint64_t sysclock;
  if (hsa_status_t status = hsa_system_get_info_fn(HSA_SYSTEM_INFO_TIMESTAMP, &sysclock);
      status == HSA_STATUS_ERROR_NOT_INITIALIZED)
    return 0;
  else if (status != HSA_STATUS_SUCCESS)
    fatal("hsa_system_get_info failed");

  static uint64_t sysclock_period = [&]() {
    uint64_t sysclock_hz = 0;
    if (hsa_status_t status =
            hsa_system_get_info_fn(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &sysclock_hz);
        status != HSA_STATUS_SUCCESS)
      fatal("hsa_system_get_info failed");

    return (uint64_t)1000000000 / sysclock_hz;
  }();

  return sysclock * sysclock_period;
}

void Initialize(HsaApiTable* table) {
  // Save the HSA core api and amd_ext api.
  saved_core_api = *table->core_;
  saved_amd_ext_api = *table->amd_ext_;

  // Enumerate the agents.
  if (hsa_support::saved_core_api.hsa_iterate_agents_fn(
          [](hsa_agent_t agent, void* data) {
            hsa_support::AgentInfo agent_info;
            if (hsa_support::saved_core_api.hsa_agent_get_info_fn(
                    agent, HSA_AGENT_INFO_DEVICE, &agent_info.type) != HSA_STATUS_SUCCESS)
              fatal("hsa_agent_get_info failed");
            switch (agent_info.type) {
              case HSA_DEVICE_TYPE_CPU:
                static int cpu_agent_count = 0;
                agent_info.id = cpu_agent_count++;
                break;
              case HSA_DEVICE_TYPE_GPU: {
                uint32_t driver_node_id;
                if (hsa_support::saved_core_api.hsa_agent_get_info_fn(
                        agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_DRIVER_NODE_ID),
                        &driver_node_id) != HSA_STATUS_SUCCESS)
                  fatal("hsa_agent_get_info failed");

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
    fatal("hsa_iterate_agents failed");

  // Install the code object intercept.
  hsa_status_t status = table->core_->hsa_system_get_major_extension_table_fn(
      HSA_EXTENSION_AMD_LOADER, 1, sizeof(hsa_ven_amd_loader_1_01_pfn_t), &hsa_loader_api);
  if (status != HSA_STATUS_SUCCESS) fatal("hsa_system_get_major_extension_table failed");

  // Install the HSA_OPS intercept
  table->amd_ext_->hsa_amd_memory_async_copy_fn = MemoryASyncCopyIntercept;
  table->amd_ext_->hsa_amd_memory_async_copy_rect_fn = MemoryASyncCopyRectIntercept;
  table->amd_ext_->hsa_amd_memory_async_copy_on_engine_fn = MemoryASyncCopyOnEngineIntercept;
  table->amd_ext_->hsa_amd_profiling_async_copy_enable_fn = ProfilingAsyncCopyEnableIntercept;

  // Install the HSA_EVT intercept
  table->core_->hsa_memory_allocate_fn = MemoryAllocateIntercept;
  table->core_->hsa_memory_assign_agent_fn = MemoryAssignAgentIntercept;
  table->core_->hsa_memory_copy_fn = MemoryCopyIntercept;
  table->amd_ext_->hsa_amd_memory_pool_allocate_fn = MemoryPoolAllocateIntercept;
  table->amd_ext_->hsa_amd_memory_pool_free_fn = MemoryPoolFreeIntercept;
  table->amd_ext_->hsa_amd_agents_allow_access_fn = AgentsAllowAccessIntercept;
  table->core_->hsa_executable_freeze_fn = ExecutableFreezeIntercept;
  table->core_->hsa_executable_destroy_fn = ExecutableDestroyIntercept;

  // Install the HSA_API wrappers
  detail::InstallCoreApiWrappers(table->core_);
  detail::InstallAmdExtWrappers(table->amd_ext_);
  detail::InstallImageExtWrappers(table->image_ext_);
}

void Finalize() {
  if (hsa_status_t status =
          saved_amd_ext_api.hsa_amd_profiling_async_copy_enable_fn(profiling_async_copy_enable.load(std::memory_order_relaxed));
      status != HSA_STATUS_SUCCESS)
    assert(!"hsa_amd_profiling_async_copy_enable failed");

  memset(&saved_core_api, '\0', sizeof(saved_core_api));
  memset(&saved_amd_ext_api, '\0', sizeof(saved_amd_ext_api));
  memset(&hsa_loader_api, '\0', sizeof(hsa_loader_api));
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
