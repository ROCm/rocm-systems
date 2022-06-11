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
#include "exception.h"
#include "loader.h"
#include "memory_pool.h"
#include "roctracer.h"
#include "roctracer_hsa.h"
#include "tracker.h"
#include "util/callback_table.h"
#include "util/logger.h"

#include <hsa/hsa.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <unordered_map>
#include <optional>
#include <mutex>

#include "hsa_prof_str.inline.h"

namespace roctracer::hsa_support {

namespace {

util::CallbackTable<ACTIVITY_DOMAIN_HSA_EVT, HSA_EVT_ID_NUMBER> hsa_evt_cb_table;

CoreApiTable saved_core_api{};
AmdExtTable saved_amd_ext_api{};
hsa_ven_amd_loader_1_01_pfn_t hsa_loader_api{};

// async copy activity callback
std::mutex init_mutex;
bool async_copy_callback_enabled = false;
MemoryPool* async_copy_callback_memory_pool = nullptr;

struct AgentInfo {
  int index;
  hsa_device_type_t type;
};
std::unordered_map<decltype(hsa_agent_t::handle), AgentInfo> agent_info_map;

hsa_status_t HSA_API MemoryAllocateIntercept(hsa_region_t region, size_t size, void** ptr) {
  hsa_status_t status = saved_core_api.hsa_memory_allocate_fn(region, size, ptr);
  if (status != HSA_STATUS_SUCCESS) return status;

  if (auto [callback_fun, callback_arg] = hsa_evt_cb_table.Get(HSA_EVT_ID_ALLOCATE); callback_fun) {
    hsa_evt_data_t data{};
    data.allocate.ptr = *ptr;
    data.allocate.size = size;
    if (saved_core_api.hsa_region_get_info_fn(region, HSA_REGION_INFO_SEGMENT,
                                              &data.allocate.segment) != HSA_STATUS_SUCCESS ||
        saved_core_api.hsa_region_get_info_fn(region, HSA_REGION_INFO_GLOBAL_FLAGS,
                                              &data.allocate.global_flag) != HSA_STATUS_SUCCESS)
      FATAL_LOGGING("hsa_region_get_info failed");

    callback_fun(ACTIVITY_DOMAIN_HSA_EVT, HSA_EVT_ID_ALLOCATE, &data, callback_arg);
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t MemoryAssignAgentIntercept(void* ptr, hsa_agent_t agent,
                                        hsa_access_permission_t access) {
  hsa_status_t status = saved_core_api.hsa_memory_assign_agent_fn(ptr, agent, access);
  if (status != HSA_STATUS_SUCCESS) return status;

  if (auto [callback_fun, callback_arg] = hsa_evt_cb_table.Get(HSA_EVT_ID_DEVICE); callback_fun) {
    hsa_evt_data_t data{};
    data.device.ptr = ptr;
    if (saved_core_api.hsa_agent_get_info_fn(agent, HSA_AGENT_INFO_DEVICE, &data.device.type) !=
        HSA_STATUS_SUCCESS)
      FATAL_LOGGING("hsa_agent_get_info failed");

    callback_fun(ACTIVITY_DOMAIN_HSA_EVT, HSA_EVT_ID_DEVICE, &data, callback_arg);
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t MemoryCopyIntercept(void* dst, const void* src, size_t size) {
  hsa_status_t status = saved_core_api.hsa_memory_copy_fn(dst, src, size);
  if (status != HSA_STATUS_SUCCESS) return status;

  if (auto [callback_fun, callback_arg] = hsa_evt_cb_table.Get(HSA_EVT_ID_MEMCOPY); callback_fun) {
    hsa_evt_data_t data{};
    data.memcopy.dst = dst;
    data.memcopy.src = src;
    data.memcopy.size = size;

    callback_fun(ACTIVITY_DOMAIN_HSA_EVT, HSA_EVT_ID_MEMCOPY, &data, callback_arg);
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t MemoryPoolAllocateIntercept(hsa_amd_memory_pool_t pool, size_t size, uint32_t flags,
                                         void** ptr) {
  hsa_status_t status = saved_amd_ext_api.hsa_amd_memory_pool_allocate_fn(pool, size, flags, ptr);
  if (size == 0 || status != HSA_STATUS_SUCCESS) return status;

  if (auto [callback_fun, callback_arg] = hsa_evt_cb_table.Get(HSA_EVT_ID_ALLOCATE); callback_fun) {
    hsa_evt_data_t data{};
    data.allocate.ptr = *ptr;
    data.allocate.size = size;

    if (saved_amd_ext_api.hsa_amd_memory_pool_get_info_fn(
            pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &data.allocate.segment) != HSA_STATUS_SUCCESS ||
        saved_amd_ext_api.hsa_amd_memory_pool_get_info_fn(
            pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &data.allocate.global_flag) !=
            HSA_STATUS_SUCCESS)
      FATAL_LOGGING("hsa_region_get_info failed");

    callback_fun(ACTIVITY_DOMAIN_HSA_EVT, HSA_EVT_ID_ALLOCATE, &data, callback_arg);

    if (std::tie(callback_fun, callback_arg) = hsa_evt_cb_table.Get(HSA_EVT_ID_DEVICE);
        !callback_fun)
      return HSA_STATUS_SUCCESS;

    // FIXME: Why is this only reported if HSA_EVT_ID_ALLOCATE is also set?
    auto callback_data = std::make_tuple(callback_fun, callback_arg, pool, ptr);
    auto agent_callback = [](hsa_agent_t agent, void* iterate_agent_callback_data) {
      auto [callback_fun, callback_arg, pool, ptr] =
          *reinterpret_cast<decltype(callback_data)*>(iterate_agent_callback_data);

      if (hsa_amd_memory_pool_access_t value;
          saved_amd_ext_api.hsa_amd_agent_memory_pool_get_info_fn(
              agent, pool, HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &value) != HSA_STATUS_SUCCESS ||
          value != HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT)
        return HSA_STATUS_SUCCESS;

      auto it = agent_info_map.find(agent.handle);
      if (it == agent_info_map.end()) FATAL_LOGGING("agent was not found in the agent_info map");

      hsa_evt_data_t data{};
      data.device.type = it->second.type;
      data.device.id = it->second.index;
      data.device.agent = agent;
      data.device.ptr = ptr;

      callback_fun(ACTIVITY_DOMAIN_HSA_EVT, HSA_EVT_ID_DEVICE, &data, callback_arg);
      return HSA_STATUS_SUCCESS;
    };
    saved_core_api.hsa_iterate_agents_fn(agent_callback, &callback_data);
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t MemoryPoolFreeIntercept(void* ptr) {
  if (auto [callback_fun, callback_arg] = hsa_evt_cb_table.Get(HSA_EVT_ID_ALLOCATE); callback_fun) {
    hsa_evt_data_t data{};
    data.allocate.ptr = ptr;
    data.allocate.size = 0;
    callback_fun(ACTIVITY_DOMAIN_HSA_EVT, HSA_EVT_ID_ALLOCATE, &data, callback_arg);
  }

  return saved_amd_ext_api.hsa_amd_memory_pool_free_fn(ptr);
}

// Agent allow access callback 'hsa_amd_agents_allow_access'
hsa_status_t AgentsAllowAccessIntercept(uint32_t num_agents, const hsa_agent_t* agents,
                                        const uint32_t* flags, const void* ptr) {
  hsa_status_t status =
      saved_amd_ext_api.hsa_amd_agents_allow_access_fn(num_agents, agents, flags, ptr);
  if (status != HSA_STATUS_SUCCESS) return status;

  if (auto [callback_fun, callback_arg] = hsa_evt_cb_table.Get(HSA_EVT_ID_DEVICE); callback_fun) {
    while (num_agents--) {
      hsa_agent_t agent = *agents++;
      auto it = agent_info_map.find(agent.handle);
      if (it == agent_info_map.end()) FATAL_LOGGING("agent was not found in the agent_info map");

      hsa_evt_data_t data{};
      data.device.type = it->second.type;
      data.device.id = it->second.index;
      data.device.agent = agent;
      data.device.ptr = ptr;

      callback_fun(ACTIVITY_DOMAIN_HSA_EVT, HSA_EVT_ID_DEVICE, &data, callback_arg);
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
  auto* code_object_callback_arg = static_cast<CodeObjectCallbackArg*>(arg);
  hsa_evt_data_t data{};

  if (hsa_loader_api.hsa_ven_amd_loader_loaded_code_object_get_info(
          loaded_code_object, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_TYPE,
          &data.codeobj.storage_type) != HSA_STATUS_SUCCESS)
    FATAL_LOGGING("hsa_ven_amd_loader_loaded_code_object_get_info failed");

  if (data.codeobj.storage_type == HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_FILE) {
    if (hsa_loader_api.hsa_ven_amd_loader_loaded_code_object_get_info(
            loaded_code_object, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_FILE,
            &data.codeobj.storage_file) != HSA_STATUS_SUCCESS ||
        data.codeobj.storage_file == -1)
      FATAL_LOGGING("hsa_ven_amd_loader_loaded_code_object_get_info failed");
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
      FATAL_LOGGING("hsa_ven_amd_loader_loaded_code_object_get_info failed");
    data.codeobj.storage_file = -1;
  } else if (data.codeobj.storage_type == HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_NONE) {
    return HSA_STATUS_SUCCESS;  // FIXME: do we really not care about these code objects?
  } else {
    FATAL_LOGGING("Unknown code object storage type: " << data.codeobj.storage_type);
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
    FATAL_LOGGING("hsa_ven_amd_loader_loaded_code_object_get_info failed");

  if (hsa_loader_api.hsa_ven_amd_loader_loaded_code_object_get_info(
          loaded_code_object, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI_LENGTH,
          &data.codeobj.uri_length) != HSA_STATUS_SUCCESS)
    FATAL_LOGGING("hsa_ven_amd_loader_loaded_code_object_get_info failed");

  std::string uri_str(data.codeobj.uri_length, '\0');
  if (hsa_loader_api.hsa_ven_amd_loader_loaded_code_object_get_info(
          loaded_code_object, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI, uri_str.data()) !=
      HSA_STATUS_SUCCESS)
    FATAL_LOGGING("hsa_ven_amd_loader_loaded_code_object_get_info failed");

  data.codeobj.uri = uri_str.c_str();
  data.codeobj.unload = code_object_callback_arg->unload ? 1 : 0;
  code_object_callback_arg->callback_fun(ACTIVITY_DOMAIN_HSA_EVT, HSA_EVT_ID_CODEOBJ, &data,
                                         code_object_callback_arg->callback_arg);

  return HSA_STATUS_SUCCESS;
}

hsa_status_t ExecutableFreezeIntercept(hsa_executable_t executable, const char* options) {
  hsa_status_t status = saved_core_api.hsa_executable_freeze_fn(executable, options);
  if (status != HSA_STATUS_SUCCESS) return status;

  if (auto [callback_fun, callback_arg] = hsa_evt_cb_table.Get(HSA_EVT_ID_CODEOBJ); callback_fun) {
    CodeObjectCallbackArg arg = {callback_fun, callback_arg, false};
    hsa_loader_api.hsa_ven_amd_loader_executable_iterate_loaded_code_objects(
        executable, CodeObjectCallback, &arg);
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t ExecutableDestroyIntercept(hsa_executable_t executable) {
  if (auto [callback_fun, callback_arg] = hsa_evt_cb_table.Get(HSA_EVT_ID_CODEOBJ); callback_fun) {
    CodeObjectCallbackArg arg = {callback_fun, callback_arg, true};
    hsa_loader_api.hsa_ven_amd_loader_executable_iterate_loaded_code_objects(
        executable, CodeObjectCallback, &arg);
  }

  return saved_core_api.hsa_executable_destroy_fn(executable);
}

void MemoryASyncCopyHandler(const Tracker::entry_t* entry) {
  activity_record_t record{};
  record.domain = ACTIVITY_DOMAIN_HSA_OPS;
  record.op = HSA_OP_ID_COPY;
  record.begin_ns = entry->begin;
  record.end_ns = entry->end;
  record.device_id = 0;
  record.correlation_id = entry->correlation_id;
  entry->pool->Write(record);
}

hsa_status_t MemoryASyncCopyIntercept(void* dst, hsa_agent_t dst_agent, const void* src,
                                      hsa_agent_t src_agent, size_t size, uint32_t num_dep_signals,
                                      const hsa_signal_t* dep_signals,
                                      hsa_signal_t completion_signal) {
  if (!async_copy_callback_enabled) {
    return saved_amd_ext_api.hsa_amd_memory_async_copy_fn(
        dst, dst_agent, src, src_agent, size, num_dep_signals, dep_signals, completion_signal);
  }

  Tracker::entry_t* entry = new Tracker::entry_t();
  entry->handler = MemoryASyncCopyHandler;
  entry->pool = async_copy_callback_memory_pool;
  entry->correlation_id = CorrelationId();
  Tracker::Enable(Tracker::COPY_ENTRY_TYPE, hsa_agent_t{}, completion_signal, entry);

  hsa_status_t status = saved_amd_ext_api.hsa_amd_memory_async_copy_fn(
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
  if (!async_copy_callback_enabled) {
    return saved_amd_ext_api.hsa_amd_memory_async_copy_rect_fn(
        dst, dst_offset, src, src_offset, range, copy_agent, dir, num_dep_signals, dep_signals,
        completion_signal);
  }

  Tracker::entry_t* entry = new Tracker::entry_t();
  entry->handler = MemoryASyncCopyHandler;
  entry->pool = async_copy_callback_memory_pool;
  entry->correlation_id = CorrelationId();
  Tracker::Enable(Tracker::COPY_ENTRY_TYPE, hsa_agent_t{}, completion_signal, entry);

  hsa_status_t status = saved_amd_ext_api.hsa_amd_memory_async_copy_rect_fn(
      dst, dst_offset, src, src_offset, range, copy_agent, dir, num_dep_signals, dep_signals,
      entry->signal);
  if (status != HSA_STATUS_SUCCESS) Tracker::Disable(entry);

  return status;
}

void AsyncActivityCallback(uint32_t op_id, void* record, void* arg) {
  MemoryPool* pool = reinterpret_cast<MemoryPool*>(arg);
  roctracer_record_t* record_ptr = reinterpret_cast<roctracer_record_t*>(record);
  record_ptr->domain = ACTIVITY_DOMAIN_HSA_OPS;
  pool->Write(*record_ptr);
}

}  // namespace

roctracer_timestamp_t timestamp_ns() {
  uint64_t sysclock;

  if (saved_core_api.hsa_system_get_info_fn == nullptr)
    FATAL_LOGGING("HSA intercept is not active");

  if (hsa_status_t status =
          saved_core_api.hsa_system_get_info_fn(HSA_SYSTEM_INFO_TIMESTAMP, &sysclock);
      status == HSA_STATUS_ERROR_NOT_INITIALIZED)
    return 0;
  else if (status != HSA_STATUS_SUCCESS)
    FATAL_LOGGING("hsa_system_get_info failed");

  static uint64_t sysclock_period = []() {
    uint64_t sysclock_hz = 0;
    if (hsa_status_t status = saved_core_api.hsa_system_get_info_fn(
            HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &sysclock_hz);
        status != HSA_STATUS_SUCCESS)
      FATAL_LOGGING("hsa_system_get_info failed");

    return (uint64_t)1000000000 / sysclock_hz;
  }();

  return sysclock * sysclock_period;
}

void Initialize(HsaApiTable* table) {
  std::scoped_lock lock(init_mutex);

  // Save the HSA core api and amd_ext api.
  saved_core_api = *table->core_;
  saved_amd_ext_api = *table->amd_ext_;

  // Enumerate the agents.
  if (hsa_support::saved_core_api.hsa_iterate_agents_fn(
          [](hsa_agent_t agent, void* data) {
            hsa_support::AgentInfo agent_info;
            if (hsa_support::saved_core_api.hsa_agent_get_info_fn(
                    agent, HSA_AGENT_INFO_DEVICE, &agent_info.type) != HSA_STATUS_SUCCESS)
              FATAL_LOGGING("hsa_agent_get_info failed");
            switch (agent_info.type) {
              case HSA_DEVICE_TYPE_CPU:
                static int cpu_agent_count = 0;
                agent_info.index = cpu_agent_count++;
                break;
              case HSA_DEVICE_TYPE_GPU:
                static int gpu_agent_count = 0;
                agent_info.index = gpu_agent_count++;
                break;
              default:
                static int other_agent_count = 0;
                agent_info.index = other_agent_count++;
                break;
            }
            hsa_support::agent_info_map.emplace(agent.handle, agent_info);
            return HSA_STATUS_SUCCESS;
          },
          nullptr) != HSA_STATUS_SUCCESS)
    FATAL_LOGGING("hsa_iterate_agents failed");

  // Install the code object intercept.
  hsa_status_t status = table->core_->hsa_system_get_major_extension_table_fn(
      HSA_EXTENSION_AMD_LOADER, 1, sizeof(hsa_ven_amd_loader_1_01_pfn_t), &hsa_loader_api);
  if (status != HSA_STATUS_SUCCESS) FATAL_LOGGING("hsa_system_get_major_extension_table failed");

  // Install the HSA_OPS intercept
  table->amd_ext_->hsa_amd_memory_async_copy_fn = MemoryASyncCopyIntercept;
  table->amd_ext_->hsa_amd_memory_async_copy_rect_fn = MemoryASyncCopyRectIntercept;

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

  if (async_copy_callback_enabled) {
    [[maybe_unused]] hsa_status_t status =
        saved_amd_ext_api.hsa_amd_profiling_async_copy_enable_fn(true);
    assert(status == HSA_STATUS_SUCCESS && "hsa_amd_profiling_async_copy_enable failed");
  }
}

void Finalize() {
  if (hsa_support::async_copy_callback_enabled) {
    [[maybe_unused]] hsa_status_t status =
        hsa_support::saved_amd_ext_api.hsa_amd_profiling_async_copy_enable_fn(false);
    assert(status == HSA_STATUS_SUCCESS && "hsa_amd_profiling_async_copy_enable failed");
  }
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
  };
  throw ApiError(ROCTRACER_STATUS_ERROR_INVALID_ARGUMENT, "invalid HSA EVT callback id");
}

const char* GetOpsName(uint32_t id) { return RocpLoader::Instance().GetOpName(id); }

uint32_t GetApiCode(const char* str) { return detail::GetApiCode(str); }

void EnableActivity(roctracer_domain_t domain, uint32_t op, roctracer_pool_t* pool) {
  switch (domain) {
    case ACTIVITY_DOMAIN_HSA_OPS:
      if (op == HSA_OP_ID_COPY) {
        std::scoped_lock lock(init_mutex);

        if (saved_amd_ext_api.hsa_amd_profiling_async_copy_enable_fn != nullptr) {
          [[maybe_unused]] hsa_status_t status =
              saved_amd_ext_api.hsa_amd_profiling_async_copy_enable_fn(true);
          assert(status == HSA_STATUS_SUCCESS && "hsa_amd_profiling_async_copy_enable failed");
        }
        async_copy_callback_enabled = true;
        async_copy_callback_memory_pool = reinterpret_cast<MemoryPool*>(pool);
      } else {
        const bool init_phase = (RocpLoader::GetRef() == nullptr);
        if (RocpLoader::GetRef() == nullptr) break;
        if (init_phase) {
          RocpLoader::Instance().InitActivityCallback(
              reinterpret_cast<void*>(AsyncActivityCallback), pool);
        }
        if (!RocpLoader::Instance().EnableActivityCallback(op, true))
          FATAL_LOGGING("HSA::EnableActivityCallback error");
      }
      break;
    case ACTIVITY_DOMAIN_HSA_API:
      // FIXME: Add HSA api activities.
      break;
    case ACTIVITY_DOMAIN_HSA_EVT:
      break;
    default:
      break;
  }
}

void EnableCallback(roctracer_domain_t domain, uint32_t cid, roctracer_rtapi_callback_t callback,
                    void* user_data) {
  switch (domain) {
    case ACTIVITY_DOMAIN_HSA_OPS:
      break;
    case ACTIVITY_DOMAIN_HSA_API:
      if (cid >= HSA_API_ID_NUMBER)
        EXC_RAISING(ROCTRACER_STATUS_ERROR_INVALID_ARGUMENT,
                    "invalid HSA API operation ID(" << cid << ")");

      detail::cb_table.Set(cid, callback, user_data);
      break;
    case ACTIVITY_DOMAIN_HSA_EVT:
      if (cid >= HSA_EVT_ID_NUMBER)
        EXC_RAISING(ROCTRACER_STATUS_ERROR_INVALID_ARGUMENT,
                    "invalid HSA API operation ID(" << cid << ")");

      hsa_evt_cb_table.Set(cid, callback, user_data);
      break;
    default:
      break;
  }
}

void DisableActivity(roctracer_domain_t domain, uint32_t op) {
  switch (domain) {
    case ACTIVITY_DOMAIN_HSA_OPS:
      if (op == HSA_OP_ID_COPY) {
        std::scoped_lock lock(init_mutex);

        async_copy_callback_enabled = false;
        async_copy_callback_memory_pool = nullptr;

        if (saved_amd_ext_api.hsa_amd_profiling_async_copy_enable_fn != nullptr) {
          [[maybe_unused]] hsa_status_t status =
              saved_amd_ext_api.hsa_amd_profiling_async_copy_enable_fn(false);
          assert(status == HSA_STATUS_SUCCESS || status == HSA_STATUS_ERROR_NOT_INITIALIZED ||
                 !"hsa_amd_profiling_async_copy_enable failed");
        }
      } else {
        if (RocpLoader::GetRef() != nullptr &&
            !RocpLoader::Instance().EnableActivityCallback(op, false))
          FATAL_LOGGING("HSA::EnableActivityCallback(false) error, op(" << op << ")");
      }
      break;
    case ACTIVITY_DOMAIN_HSA_API:
      // FIXME: Add HSA api activities.
      break;
    case ACTIVITY_DOMAIN_HSA_EVT:
      break;
    default:
      break;
  }
}

void DisableCallback(roctracer_domain_t domain, uint32_t cid) {
  switch (domain) {
    case ACTIVITY_DOMAIN_HSA_OPS:
      break;
    case ACTIVITY_DOMAIN_HSA_API:
      if (cid >= HSA_API_ID_NUMBER)
        EXC_RAISING(ROCTRACER_STATUS_ERROR_INVALID_ARGUMENT,
                    "invalid HSA API operation ID(" << cid << ")");
      detail::cb_table.Set(cid, nullptr, nullptr);
      break;
    case ACTIVITY_DOMAIN_HSA_EVT:
      if (cid >= HSA_EVT_ID_NUMBER)
        EXC_RAISING(ROCTRACER_STATUS_ERROR_INVALID_ARGUMENT,
                    "invalid HSA EVT operation ID(" << cid << ")");
      hsa_evt_cb_table.Set(cid, nullptr, nullptr);
      break;
    default:
      break;
  }
}

}  // namespace roctracer::hsa_support
