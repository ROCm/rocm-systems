/******************************************************************************
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*******************************************************************************/

#include "rocprofiler.h"

#include <hsa/hsa.h>
#include <string.h>

#include <sstream>
#include <vector>

#include "core/context.h"
#include "core/context_pool.h"
#include "core/gpu_command.h"
#include "core/hsa_queue.h"
#include "core/hsa_interceptor.h"
#include "core/intercept_queue.h"
#include "core/proxy_queue.h"
#include "core/simple_proxy_queue.h"
#include "util/exception.h"
#include "util/hsa_rsrc_factory.h"
#include "util/logger.h"
#include "core/profiling_lock.h"

#define PUBLIC_API __attribute__((visibility("default")))
#define CONSTRUCTOR_API __attribute__((constructor))
#define DESTRUCTOR_API __attribute__((destructor))

#define API_METHOD_PREFIX                                                                          \
  hsa_status_t status = HSA_STATUS_SUCCESS;                                                        \
  try {
#define API_METHOD_SUFFIX                                                                          \
  }                                                                                                \
  catch (std::exception & e) {                                                                     \
    ERR_LOGGING(__FUNCTION__ << "(), " << e.what());                                               \
    status = rocprofiler::GetExcStatus(e);                                                         \
  }                                                                                                \
  return status;

#define ONLOAD_TRACE(str)                                                                          \
  if (getenv("ROCP_ONLOAD_TRACE")) do {                                                            \
      std::cout << "PID(" << GetPid() << "): PROF_LIB::" << __FUNCTION__ << " " << str             \
                << std::endl                                                                       \
                << std::flush;                                                                     \
    } while (0);
#define ONLOAD_TRACE_BEG() ONLOAD_TRACE("begin")
#define ONLOAD_TRACE_END() ONLOAD_TRACE("end")

static inline uint32_t GetPid() { return syscall(__NR_getpid); }

///////////////////////////////////////////////////////////////////////////////////////////////////
// Internal library methods
//
namespace rocprofiler {
hsa_status_t CreateQueuePro(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
                            void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data),
                            void* data, uint32_t private_segment_size, uint32_t group_segment_size,
                            hsa_queue_t** queue);

decltype(::hsa_queue_create)* hsa_queue_create_fn;
decltype(::hsa_queue_destroy)* hsa_queue_destroy_fn;

decltype(::hsa_signal_store_relaxed)* hsa_signal_store_relaxed_fn;
decltype(::hsa_signal_store_relaxed)* hsa_signal_store_screlease_fn;

decltype(::hsa_queue_load_write_index_relaxed)* hsa_queue_load_write_index_relaxed_fn;
decltype(::hsa_queue_store_write_index_relaxed)* hsa_queue_store_write_index_relaxed_fn;
decltype(::hsa_queue_load_read_index_relaxed)* hsa_queue_load_read_index_relaxed_fn;
decltype(::hsa_queue_add_write_index_scacq_screl)* hsa_queue_add_write_index_scacq_screl_fn;

decltype(::hsa_queue_load_write_index_scacquire)* hsa_queue_load_write_index_scacquire_fn;
decltype(::hsa_queue_store_write_index_screlease)* hsa_queue_store_write_index_screlease_fn;
decltype(::hsa_queue_load_read_index_scacquire)* hsa_queue_load_read_index_scacquire_fn;

decltype(::hsa_amd_queue_intercept_create)* hsa_amd_queue_intercept_create_fn;
decltype(::hsa_amd_queue_intercept_register)* hsa_amd_queue_intercept_register_fn;

decltype(::hsa_memory_allocate)* hsa_memory_allocate_fn;
decltype(::hsa_memory_assign_agent)* hsa_memory_assign_agent_fn;
decltype(::hsa_memory_copy)* hsa_memory_copy_fn;
decltype(::hsa_amd_memory_pool_allocate)* hsa_amd_memory_pool_allocate_fn;
decltype(::hsa_amd_memory_pool_free)* hsa_amd_memory_pool_free_fn;
decltype(::hsa_amd_agents_allow_access)* hsa_amd_agents_allow_access_fn;
decltype(::hsa_amd_memory_async_copy)* hsa_amd_memory_async_copy_fn;
decltype(::hsa_amd_memory_async_copy_rect)* hsa_amd_memory_async_copy_rect_fn;
decltype(::hsa_executable_freeze)* hsa_executable_freeze_fn;
decltype(::hsa_executable_destroy)* hsa_executable_destroy_fn;

::HsaApiTable* kHsaApiTable;

void SaveHsaApi(::HsaApiTable* table) {
  util::HsaRsrcFactory::InitHsaApiTable(table);

  kHsaApiTable = table;
  hsa_queue_create_fn = table->core_->hsa_queue_create_fn;
  hsa_queue_destroy_fn = table->core_->hsa_queue_destroy_fn;

  hsa_signal_store_relaxed_fn = table->core_->hsa_signal_store_relaxed_fn;
  hsa_signal_store_screlease_fn = table->core_->hsa_signal_store_screlease_fn;

  hsa_queue_load_write_index_relaxed_fn = table->core_->hsa_queue_load_write_index_relaxed_fn;
  hsa_queue_store_write_index_relaxed_fn = table->core_->hsa_queue_store_write_index_relaxed_fn;
  hsa_queue_load_read_index_relaxed_fn = table->core_->hsa_queue_load_read_index_relaxed_fn;
  hsa_queue_add_write_index_scacq_screl_fn = table->core_->hsa_queue_add_write_index_scacq_screl_fn;

  hsa_queue_load_write_index_scacquire_fn = table->core_->hsa_queue_load_write_index_scacquire_fn;
  hsa_queue_store_write_index_screlease_fn = table->core_->hsa_queue_store_write_index_screlease_fn;
  hsa_queue_load_read_index_scacquire_fn = table->core_->hsa_queue_load_read_index_scacquire_fn;

  hsa_amd_queue_intercept_create_fn = table->amd_ext_->hsa_amd_queue_intercept_create_fn;
  hsa_amd_queue_intercept_register_fn = table->amd_ext_->hsa_amd_queue_intercept_register_fn;
}

void RestoreHsaApi() {
  ::HsaApiTable* table = kHsaApiTable;
  table->core_->hsa_queue_create_fn = hsa_queue_create_fn;
  table->core_->hsa_queue_destroy_fn = hsa_queue_destroy_fn;

  table->core_->hsa_signal_store_relaxed_fn = hsa_signal_store_relaxed_fn;
  table->core_->hsa_signal_store_screlease_fn = hsa_signal_store_screlease_fn;

  table->core_->hsa_queue_load_write_index_relaxed_fn = hsa_queue_load_write_index_relaxed_fn;
  table->core_->hsa_queue_store_write_index_relaxed_fn = hsa_queue_store_write_index_relaxed_fn;
  table->core_->hsa_queue_load_read_index_relaxed_fn = hsa_queue_load_read_index_relaxed_fn;
  table->core_->hsa_queue_add_write_index_scacq_screl_fn = hsa_queue_add_write_index_scacq_screl_fn;

  table->core_->hsa_queue_load_write_index_scacquire_fn = hsa_queue_load_write_index_scacquire_fn;
  table->core_->hsa_queue_store_write_index_screlease_fn = hsa_queue_store_write_index_screlease_fn;
  table->core_->hsa_queue_load_read_index_scacquire_fn = hsa_queue_load_read_index_scacquire_fn;

  table->amd_ext_->hsa_amd_queue_intercept_create_fn = hsa_amd_queue_intercept_create_fn;
  table->amd_ext_->hsa_amd_queue_intercept_register_fn = hsa_amd_queue_intercept_register_fn;
}

void StandaloneIntercept() {
  ::HsaApiTable* table = kHsaApiTable;
  table->core_->hsa_queue_create_fn = rocprofiler::CreateQueuePro;
}

typedef void (*tool_handler_t)();
typedef void (*tool_handler_prop_t)(rocprofiler_settings_t*);
void* tool_handle = NULL;

// Load profiling tool library
// Return true if intercepting mode is enabled
enum {
  DISPATCH_INTERCEPT_MODE = 0x1,
  CODE_OBJ_TRACKING_MODE = 0x2,
  MEMCOPY_INTERCEPT_MODE = 0x4,
  HSA_INTERCEPT_MODE = 0x8,
  INTERCEPT_MODE_DFLT = CODE_OBJ_TRACKING_MODE
};
uint32_t LoadTool() {
  uint32_t intercept_mode = INTERCEPT_MODE_DFLT;
  const char* tool_lib = getenv("ROCP_TOOL_LIB");
  std::ostringstream oss;
  if (tool_lib) oss << "load tool library(" << tool_lib << ")";
  ONLOAD_TRACE(oss.str());

  if (tool_lib) {
    intercept_mode = DISPATCH_INTERCEPT_MODE;

    tool_handle = dlopen(tool_lib, RTLD_NOW);
    if (tool_handle == NULL) {
      fprintf(stderr, "ROCProfiler: can't load tool library \"%s\"\n", tool_lib);
      fprintf(stderr, "%s\n", dlerror());
      abort();
    }
    tool_handler_t handler = reinterpret_cast<tool_handler_t>(dlsym(tool_handle, "OnLoadTool"));
    tool_handler_prop_t handler_prop =
        reinterpret_cast<tool_handler_prop_t>(dlsym(tool_handle, "OnLoadToolProp"));
    if ((handler == NULL) && (handler_prop == NULL)) {
      fprintf(stderr,
              "ROCProfiler: tool library corrupted, OnLoadTool()/OnLoadToolProp() method is "
              "expected\n");
      fprintf(stderr, "%s\n", dlerror());
      abort();
    }
    tool_handler_t on_unload_handler =
        reinterpret_cast<tool_handler_t>(dlsym(tool_handle, "OnUnloadTool"));
    if (on_unload_handler == NULL) {
      fprintf(stderr, "ROCProfiler: tool library corrupted, OnUnloadTool() method is expected\n");
      fprintf(stderr, "%s\n", dlerror());
      abort();
    }

    rocprofiler_settings_t settings{};
    settings.intercept_mode = (intercept_mode != 0) ? 1 : 0;
    settings.trace_size = TraceProfile::GetSize();
    settings.trace_local = TraceProfile::IsLocal() ? 1 : 0;
    settings.timeout = util::HsaRsrcFactory::GetTimeoutNs();
    settings.timestamp_on = InterceptQueue::IsTrackerOn() ? 1 : 0;
    settings.code_obj_tracking = 1;

    if (handler)
      handler();
    else if (handler_prop)
      handler_prop(&settings);

    TraceProfile::SetSize(settings.trace_size);
    TraceProfile::SetLocal(settings.trace_local != 0);
    util::HsaRsrcFactory::SetTimeoutNs(settings.timeout);
    InterceptQueue::TrackerOn(settings.timestamp_on != 0);
    if (settings.intercept_mode != 0) intercept_mode = DISPATCH_INTERCEPT_MODE;
    if (settings.code_obj_tracking) intercept_mode |= CODE_OBJ_TRACKING_MODE;
    if (settings.memcopy_tracking) intercept_mode |= MEMCOPY_INTERCEPT_MODE;
    if (settings.hsa_intercepting) intercept_mode |= HSA_INTERCEPT_MODE;
    if (settings.k_concurrent) {
      Context::k_concurrent_ = settings.k_concurrent;
      InterceptQueue::k_concurrent_ = settings.k_concurrent;
      InterceptQueue::TrackerOn(true);
    }
    if (settings.opt_mode) InterceptQueue::opt_mode_ = true;
  }

  ONLOAD_TRACE("end intercept_mode(" << intercept_mode << ")");
  return intercept_mode;
}

void PmcStopper() {
  rocprofiler::util::HsaRsrcFactory* rsrc = &rocprofiler::util::HsaRsrcFactory::Instance();

  const uint32_t gpu_count = rsrc->GetCountOfGpuAgents();
  for (uint32_t gpu_id = 0; gpu_id < gpu_count; gpu_id++) {
    // Get agent info
    const rocprofiler::util::AgentInfo* agent_info;
    if (rsrc->GetGpuAgentInfo(gpu_id, &agent_info) == false) {
      fprintf(stderr, "Error: GetGpuAgentInfo(%u) \n", gpu_id);
      abort();
    }

    // Create queue
    hsa_queue_t* queue;
    const bool ret = rsrc->CreateQueue(agent_info, 10, &queue);
    if (ret != true) EXC_RAISING(HSA_STATUS_ERROR, "CreateQueue(" << gpu_id << ")");

    // Issue PMC-enable GPU command
    IssueGpuCommand(PMC_DISABLE_GPU_CMD_OP, agent_info, queue);

    rsrc->HsaApi()->hsa_queue_destroy(queue);
  }
}

// Unload profiling tool librray
void UnloadTool() {
  ONLOAD_TRACE("tool handle(" << tool_handle << ")");

  if (Context::k_concurrent_) PmcStopper();

  if (tool_handle) {
    tool_handler_t handler = reinterpret_cast<tool_handler_t>(dlsym(tool_handle, "OnUnloadTool"));
    if (handler == NULL) {
      fprintf(stderr,
              "ROCProfiler error: tool library corrupted, OnUnloadTool() method is expected\n");
      fprintf(stderr, "%s\n", dlerror());
      abort();
    }
    handler();
    dlclose(tool_handle);
  }
  ONLOAD_TRACE_END();
}

CONSTRUCTOR_API void constructor() {
  ONLOAD_TRACE_BEG();
  util::Logger::Create();
  ONLOAD_TRACE_END();
}

DESTRUCTOR_API void destructor() {
  ONLOAD_TRACE_BEG();
  rocprofiler::MetricsDict::Destroy();
  util::HsaRsrcFactory::Destroy();
  util::Logger::Destroy();
  ONLOAD_TRACE_END();
}

const MetricsDict* GetMetrics(const hsa_agent_t& agent) {
  rocprofiler::util::HsaRsrcFactory* hsa_rsrc = &rocprofiler::util::HsaRsrcFactory::Instance();
  const rocprofiler::util::AgentInfo* agent_info = hsa_rsrc->GetAgentInfo(agent);
  if (agent_info == NULL) EXC_RAISING(HSA_STATUS_ERROR, "agent is not found");
  const MetricsDict* metrics = MetricsDict::Create(agent_info);
  if (metrics == NULL) EXC_RAISING(HSA_STATUS_ERROR, "MetricsDict create failed");
  return metrics;
}

hsa_status_t GetExcStatus(const std::exception& e) {
  const util::exception* rocprofiler_exc_ptr = dynamic_cast<const util::exception*>(&e);
  return (rocprofiler_exc_ptr) ? static_cast<hsa_status_t>(rocprofiler_exc_ptr->status())
                               : HSA_STATUS_ERROR;
}

hsa_status_t CreateQueuePro(hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
                            void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data),
                            void* data, uint32_t private_segment_size, uint32_t group_segment_size,
                            hsa_queue_t** queue) {
  // Create HSA queue
  hsa_status_t status = hsa_queue_create_fn(agent, size, type, callback, data, private_segment_size,
                                            group_segment_size, queue);
  if (status != HSA_STATUS_SUCCESS) return status;

  // Issue PMC-enable GPU command
  IssueGpuCommand(PMC_ENABLE_GPU_CMD_OP, agent, *queue);

  return HSA_STATUS_SUCCESS;
}

bool async_copy_handler(hsa_signal_value_t value, void* arg) {
  Tracker::entry_t* entry = reinterpret_cast<Tracker::entry_t*>(arg);
  printf("%lu: async-copy time(%lu,%lu)\n", entry->index, entry->record->begin, entry->record->end);
  return false;
}

hsa_status_t hsa_amd_memory_async_copy_interceptor(void* dst, hsa_agent_t dst_agent,
                                                   const void* src, hsa_agent_t src_agent,
                                                   size_t size, uint32_t num_dep_signals,
                                                   const hsa_signal_t* dep_signals,
                                                   hsa_signal_t completion_signal) {
  Tracker* tracker = &Tracker::Instance();
  Tracker::entry_t* tracker_entry = tracker->Alloc(hsa_agent_t{}, completion_signal);
  hsa_status_t status = hsa_amd_memory_async_copy_fn(
      dst, dst_agent, src, src_agent, size, num_dep_signals, dep_signals, tracker_entry->signal);
  if (status == HSA_STATUS_SUCCESS) {
    tracker->EnableMemcopy(tracker_entry, async_copy_handler,
                           reinterpret_cast<void*>(tracker_entry));
  } else {
    tracker->Delete(tracker_entry);
  }
  return status;
}

hsa_status_t hsa_amd_memory_async_copy_rect_interceptor(
    const hsa_pitched_ptr_t* dst, const hsa_dim3_t* dst_offset, const hsa_pitched_ptr_t* src,
    const hsa_dim3_t* src_offset, const hsa_dim3_t* range, hsa_agent_t copy_agent,
    hsa_amd_copy_direction_t dir, uint32_t num_dep_signals, const hsa_signal_t* dep_signals,
    hsa_signal_t completion_signal) {
  Tracker* tracker = &Tracker::Instance();
  Tracker::entry_t* tracker_entry = tracker->Alloc(hsa_agent_t{}, completion_signal);
  hsa_status_t status =
      hsa_amd_memory_async_copy_rect_fn(dst, dst_offset, src, src_offset, range, copy_agent, dir,
                                        num_dep_signals, dep_signals, tracker_entry->signal);
  if (status == HSA_STATUS_SUCCESS) {
    tracker->EnableMemcopy(tracker_entry, async_copy_handler,
                           reinterpret_cast<void*>(tracker_entry));
  } else {
    tracker->Delete(tracker_entry);
  }
  return status;
}

rocprofiler_properties_t rocprofiler_properties;
uint32_t TraceProfile::output_buffer_size_ = 0x2000000;  // 32M
bool TraceProfile::output_buffer_local_ = true;
std::atomic<Tracker*> Tracker::instance_{};
Tracker::mutex_t Tracker::glob_mutex_;
Tracker::counter_t Tracker::counter_ = 0;
util::Logger::mutex_t util::Logger::mutex_;
std::atomic<util::Logger*> util::Logger::instance_{};
}  // namespace rocprofiler

CONTEXT_INSTANTIATE();

///////////////////////////////////////////////////////////////////////////////////////////////////
// Public library methods
//
extern "C" {

// The HSA_AMD_TOOL_PRIORITY variable must be a constant value type
// initialized by the loader itself, not by code during _init. 'extern const'
// seems do that although that is not a guarantee.
ROCPROFILER_EXPORT extern const uint32_t HSA_AMD_TOOL_PRIORITY = 25;


// HSA-runtime tool on-load method
PUBLIC_API bool OnLoad(HsaApiTable* table, uint64_t runtime_version, uint64_t failed_tool_count,
                       const char* const* failed_tool_names) {
  ONLOAD_TRACE_BEG();
  rocprofiler::SaveHsaApi(table);
  rocprofiler::ProxyQueue::InitFactory();

  // Checking environment to enable intercept mode
  const char* intercept_env = getenv("ROCP_HSA_INTERCEPT");

  int intercept_env_value = 0;
  if (intercept_env != NULL) {
    intercept_env_value = atoi(intercept_env);

    switch (intercept_env_value) {
      case 0:
      case 1:
        // 0: Intercepting disabled
        // 1: Intercepting enabled without timestamping
        rocprofiler::InterceptQueue::TrackerOn(false);
        break;
      case 2:
        // Intercepting enabled with timestamping
        rocprofiler::InterceptQueue::TrackerOn(true);
        break;
      default:
        ERR_LOGGING("Bad ROCP_HSA_INTERCEPT env var value ("
                    << intercept_env << "): "
                    << "valid values are 0 (standalone), 1 (intercepting without timestamp), 2 "
                       "(intercepting with timestamp)");
        return false;
    }
  }

  // always enable excutable tracking
  rocprofiler::util::HsaRsrcFactory::EnableExecutableTracking(table);

  // Loading a tool lib and setting of intercept mode
  const uint32_t intercept_mode_mask = rocprofiler::LoadTool();

  if (intercept_mode_mask & rocprofiler::MEMCOPY_INTERCEPT_MODE) {
    hsa_status_t status = hsa_amd_profiling_async_copy_enable(true);
    if (status != HSA_STATUS_SUCCESS) EXC_ABORT(status, "hsa_amd_profiling_async_copy_enable");
    rocprofiler::hsa_amd_memory_async_copy_fn = table->amd_ext_->hsa_amd_memory_async_copy_fn;
    rocprofiler::hsa_amd_memory_async_copy_rect_fn =
        table->amd_ext_->hsa_amd_memory_async_copy_rect_fn;
    table->amd_ext_->hsa_amd_memory_async_copy_fn =
        rocprofiler::hsa_amd_memory_async_copy_interceptor;
    table->amd_ext_->hsa_amd_memory_async_copy_rect_fn =
        rocprofiler::hsa_amd_memory_async_copy_rect_interceptor;
  }
  if (intercept_mode_mask & rocprofiler::HSA_INTERCEPT_MODE) {
    if (intercept_mode_mask & rocprofiler::MEMCOPY_INTERCEPT_MODE) {
      EXC_ABORT(HSA_STATUS_ERROR, "HSA_INTERCEPT and MEMCOPY_INTERCEPT conflict");
    }
    rocprofiler::HsaInterceptor::Enable(true);
    rocprofiler::HsaInterceptor::HsaIntercept(table);
  }

  // HSA intercepting
  if (intercept_env_value != 0) {
    rocprofiler::ProxyQueue::HsaIntercept(table);
    rocprofiler::InterceptQueue::HsaIntercept(table);
  } else {
    rocprofiler::StandaloneIntercept();
  }

  ONLOAD_TRACE("end intercept_mode(" << std::hex << intercept_env_value << ")"
                                     << " intercept_mode_mask(" << std::hex << intercept_mode_mask
                                     << ")" << std::dec);
  return true;
}

// HSA-runtime tool on-unload method
PUBLIC_API void OnUnload() {
  ONLOAD_TRACE_BEG();
  rocprofiler::UnloadTool();
  rocprofiler::RestoreHsaApi();
  ONLOAD_TRACE_END();
}

// Returns library vesrion
PUBLIC_API uint32_t rocprofiler_version_major() { return ROCPROFILER_VERSION_MAJOR; }
PUBLIC_API uint32_t rocprofiler_version_minor() { return ROCPROFILER_VERSION_MINOR; }

// Returns the last error message
PUBLIC_API hsa_status_t rocprofiler_error_string(const char** str) {
  API_METHOD_PREFIX
  *str = rocprofiler::util::Logger::LastMessage().c_str();
  API_METHOD_SUFFIX
}

// Create new profiling context
PUBLIC_API hsa_status_t rocprofiler_open(hsa_agent_t agent, rocprofiler_feature_t* features,
                                         uint32_t feature_count, rocprofiler_t** handle,
                                         uint32_t mode, rocprofiler_properties_t* properties) {
  API_METHOD_PREFIX
  ProfilingLock::Lock(PROFILER_V1_LOCK);
  rocprofiler::util::HsaRsrcFactory* hsa_rsrc = &rocprofiler::util::HsaRsrcFactory::Instance();
  const rocprofiler::util::AgentInfo* agent_info = hsa_rsrc->GetAgentInfo(agent);
  if (agent_info == NULL) {
    EXC_RAISING(HSA_STATUS_ERROR, "agent is not found");
  }

  rocprofiler::Queue* queue = NULL;
  if (mode != 0) {
    if (mode & ROCPROFILER_MODE_STANDALONE) {
      if (mode & ROCPROFILER_MODE_CREATEQUEUE) {
        if (hsa_rsrc->CreateQueue(agent_info, properties->queue_depth, &(properties->queue)) ==
            false) {
          EXC_RAISING(HSA_STATUS_ERROR, "CreateQueue() failed");
        }
      }
      queue = new rocprofiler::HsaQueue(agent_info, properties->queue);
    } else {
      EXC_RAISING(HSA_STATUS_ERROR, "invalid mode (" << mode << ")");
    }
  }

  rocprofiler::Context** context_ret = reinterpret_cast<rocprofiler::Context**>(handle);
  *context_ret = rocprofiler::Context::Create(agent_info, queue, features, feature_count,
                                              properties->handler, properties->handler_arg);
  API_METHOD_SUFFIX
}

// Delete profiling info
PUBLIC_API hsa_status_t rocprofiler_close(rocprofiler_t* handle) {
  API_METHOD_PREFIX
  rocprofiler::Context* context = reinterpret_cast<rocprofiler::Context*>(handle);
  if (context) rocprofiler::Context::Destroy(context);
  API_METHOD_SUFFIX
}

// Reset context
PUBLIC_API hsa_status_t rocprofiler_reset(rocprofiler_t* handle, uint32_t group_index) {
  API_METHOD_PREFIX
  rocprofiler::Context* context = reinterpret_cast<rocprofiler::Context*>(handle);
  context->Reset(group_index);
  API_METHOD_SUFFIX
}

// Return context agent
PUBLIC_API hsa_status_t rocprofiler_get_agent(rocprofiler_t* handle, hsa_agent_t* agent) {
  API_METHOD_PREFIX
  rocprofiler::Context* context = reinterpret_cast<rocprofiler::Context*>(handle);
  *agent = context->GetAgent();
  API_METHOD_SUFFIX
}

// Get profiling group count
PUBLIC_API hsa_status_t rocprofiler_group_count(const rocprofiler_t* handle,
                                                uint32_t* group_count) {
  API_METHOD_PREFIX
  const rocprofiler::Context* context = reinterpret_cast<const rocprofiler::Context*>(handle);
  *group_count = context->GetGroupCount();
  API_METHOD_SUFFIX
}

// Get profiling group for a given group index
PUBLIC_API hsa_status_t rocprofiler_get_group(rocprofiler_t* handle, uint32_t group_index,
                                              rocprofiler_group_t* group) {
  API_METHOD_PREFIX
  rocprofiler::Context* context = reinterpret_cast<rocprofiler::Context*>(handle);
  *group = context->GetGroupDescr(group_index);
  API_METHOD_SUFFIX
}

// Start profiling
PUBLIC_API hsa_status_t rocprofiler_start(rocprofiler_t* handle, uint32_t group_index) {
  API_METHOD_PREFIX
  rocprofiler::Context* context = reinterpret_cast<rocprofiler::Context*>(handle);
  context->Start(group_index);
  API_METHOD_SUFFIX
}

// Stop profiling
PUBLIC_API hsa_status_t rocprofiler_stop(rocprofiler_t* handle, uint32_t group_index) {
  API_METHOD_PREFIX
  rocprofiler::Context* context = reinterpret_cast<rocprofiler::Context*>(handle);
  context->Stop(group_index);
  API_METHOD_SUFFIX
}

// Read profiling
PUBLIC_API hsa_status_t rocprofiler_read(rocprofiler_t* handle, uint32_t group_index) {
  API_METHOD_PREFIX
  rocprofiler::Context* context = reinterpret_cast<rocprofiler::Context*>(handle);
  context->Read(group_index);
  API_METHOD_SUFFIX
}

// Get profiling data
PUBLIC_API hsa_status_t rocprofiler_get_data(rocprofiler_t* handle, uint32_t group_index) {
  API_METHOD_PREFIX
  rocprofiler::Context* context = reinterpret_cast<rocprofiler::Context*>(handle);
  context->GetData(group_index);
  API_METHOD_SUFFIX
}

// Start profiling
PUBLIC_API hsa_status_t rocprofiler_group_start(rocprofiler_group_t* group) {
  API_METHOD_PREFIX
  rocprofiler_start(group->context, group->index);
  API_METHOD_SUFFIX
}

// Stop profiling
PUBLIC_API hsa_status_t rocprofiler_group_stop(rocprofiler_group_t* group) {
  API_METHOD_PREFIX
  rocprofiler_stop(group->context, group->index);
  API_METHOD_SUFFIX
}

// Read profiling
PUBLIC_API hsa_status_t rocprofiler_group_read(rocprofiler_group_t* group) {
  API_METHOD_PREFIX
  rocprofiler_read(group->context, group->index);
  API_METHOD_SUFFIX
}

// Get profiling data
PUBLIC_API hsa_status_t rocprofiler_group_get_data(rocprofiler_group_t* group) {
  API_METHOD_PREFIX
  rocprofiler::Context* context = reinterpret_cast<rocprofiler::Context*>(group->context);
  context->GetData(group->index);
  API_METHOD_SUFFIX
}

// Get metrics data
PUBLIC_API hsa_status_t rocprofiler_get_metrics(const rocprofiler_t* handle) {
  API_METHOD_PREFIX
  const rocprofiler::Context* context = reinterpret_cast<const rocprofiler::Context*>(handle);
  context->GetMetricsData();
  API_METHOD_SUFFIX
}

// Set/remove queue callbacks
PUBLIC_API hsa_status_t rocprofiler_set_queue_callbacks(rocprofiler_queue_callbacks_t callbacks,
                                                        void* data) {
  API_METHOD_PREFIX
  rocprofiler::InterceptQueue::SetCallbacks(callbacks, data);
  API_METHOD_SUFFIX
}

// Remove queue callbacks
PUBLIC_API hsa_status_t rocprofiler_remove_queue_callbacks() {
  API_METHOD_PREFIX
  rocprofiler::InterceptQueue::RemoveCallbacks();
  API_METHOD_SUFFIX
}

// Start/stop queue callbacks
PUBLIC_API hsa_status_t rocprofiler_start_queue_callbacks() {
  API_METHOD_PREFIX
  rocprofiler::InterceptQueue::Start();
  API_METHOD_SUFFIX
}
PUBLIC_API hsa_status_t rocprofiler_stop_queue_callbacks() {
  API_METHOD_PREFIX
  rocprofiler::InterceptQueue::Stop();
  API_METHOD_SUFFIX
}

// Method for iterating the events output data
PUBLIC_API hsa_status_t rocprofiler_iterate_trace_data(
    rocprofiler_t* handle, hsa_ven_amd_aqlprofile_data_callback_t callback, void* data) {
  API_METHOD_PREFIX
  rocprofiler::Context* context = reinterpret_cast<rocprofiler::Context*>(handle);
  context->IterateTraceData(callback, data);
  API_METHOD_SUFFIX
}

////////////////////////////////////////////////////////////////////////////////
// Open profiling pool
PUBLIC_API hsa_status_t
rocprofiler_pool_open(hsa_agent_t agent,                          // GPU handle
                      rocprofiler_feature_t* features,            // [in] profiling features array
                      uint32_t feature_count,                     // profiling info count
                      rocprofiler_pool_t** pool,                  // [out] context object
                      uint32_t mode,                              // profiling mode mask
                      rocprofiler_pool_properties_t* properties)  // pool properties
{
  API_METHOD_PREFIX
  ProfilingLock::Lock(PROFILER_V1_LOCK);
  rocprofiler::util::HsaRsrcFactory* hsa_rsrc = &rocprofiler::util::HsaRsrcFactory::Instance();
  const rocprofiler::util::AgentInfo* agent_info = hsa_rsrc->GetAgentInfo(agent);
  if (agent_info == NULL) {
    EXC_RAISING(HSA_STATUS_ERROR, "agent is not found");
  }

  rocprofiler::ContextPool* obj = rocprofiler::ContextPool::Create(
      properties->num_entries, properties->payload_bytes, agent_info, features, feature_count,
      properties->handler, properties->handler_arg);
  *pool = reinterpret_cast<rocprofiler_pool_t*>(obj);
  API_METHOD_SUFFIX
}

// Close profiling pool
PUBLIC_API hsa_status_t rocprofiler_pool_close(rocprofiler_pool_t* pool)  // profiling pool handle
{
  API_METHOD_PREFIX
  rocprofiler::ContextPool* obj = reinterpret_cast<rocprofiler::ContextPool*>(pool);
  rocprofiler::ContextPool::Destroy(obj);
  API_METHOD_SUFFIX
}

// Fetch profiling pool entry
PUBLIC_API hsa_status_t
rocprofiler_pool_fetch(rocprofiler_pool_t* pool,         // profiling pool handle
                       rocprofiler_pool_entry_t* entry)  // [out] empty profling pool entry
{
  API_METHOD_PREFIX
  rocprofiler::ContextPool* context_pool = reinterpret_cast<rocprofiler::ContextPool*>(pool);
  context_pool->Fetch(entry);
  API_METHOD_SUFFIX
}

// Fetch profiling pool entry
PUBLIC_API hsa_status_t rocprofiler_pool_flush(rocprofiler_pool_t* pool)  // profiling pool handle
{
  API_METHOD_PREFIX
  rocprofiler::ContextPool* context_pool = reinterpret_cast<rocprofiler::ContextPool*>(pool);
  context_pool->Flush();
  API_METHOD_SUFFIX
}

////////////////////////////////////////////////////////////////////////////////
// Return the info for a given info kind
PUBLIC_API hsa_status_t rocprofiler_get_info(const hsa_agent_t* agent, rocprofiler_info_kind_t kind,
                                             void* data) {
  API_METHOD_PREFIX
  if (agent == NULL) EXC_RAISING(HSA_STATUS_ERROR, "NULL agent");
  uint32_t* result_32bit_ptr = reinterpret_cast<uint32_t*>(data);

  switch (kind) {
    case ROCPROFILER_INFO_KIND_METRIC_COUNT:
      *result_32bit_ptr = rocprofiler::GetMetrics(*agent)->Size();
      break;
    case ROCPROFILER_INFO_KIND_TRACE_COUNT:
      *result_32bit_ptr = 1;
      break;
    default:
      EXC_RAISING(HSA_STATUS_ERROR, "unknown info kind(" << kind << ")");
  }
  API_METHOD_SUFFIX
}

// Iterate over the info for a given info kind, and invoke an application-defined callback on every
// iteration
PUBLIC_API hsa_status_t rocprofiler_iterate_info(
    const hsa_agent_t* agent, rocprofiler_info_kind_t kind,
    hsa_status_t (*callback)(const rocprofiler_info_data_t info, void* data), void* data) {
  API_METHOD_PREFIX
  rocprofiler::util::HsaRsrcFactory* hsa_rsrc = &rocprofiler::util::HsaRsrcFactory::Instance();
  rocprofiler_info_data_t info{};
  info.kind = kind;
  uint32_t agent_idx = hsa_rsrc->GetCountOfCpuAgents();
  uint32_t agent_max = 0;
  const rocprofiler::util::AgentInfo* agent_info = NULL;

  if (agent != NULL) {
    agent_info = hsa_rsrc->GetAgentInfo(*agent);
    agent_idx = agent_info->dev_index;
    agent_max = agent_idx + 1;
  }

  while (hsa_rsrc->GetGpuAgentInfo(agent_idx, &agent_info)) {
    info.agent_index = agent_idx;

    switch (kind) {
      case ROCPROFILER_INFO_KIND_METRIC: {
        const rocprofiler::MetricsDict* dict = rocprofiler::GetMetrics(agent_info->dev_id);
        auto nodes_vec = dict->GetNodes();

        for (auto* node : nodes_vec) {
          const std::string& name = node->opts["name"];
          const std::string& descr = node->opts["descr"];
          const std::string& expr = node->opts["expr"];
          info.metric.name = strdup(name.c_str());
          info.metric.description = strdup(descr.c_str());
          info.metric.expr = expr.empty() ? NULL : strdup(expr.c_str());
          info.metric.instances = 1;

          if (expr.empty()) {
            // Getting the block name
            const std::string block_name = node->opts["block"];

            // Querying profile
            rocprofiler::profile_t profile = {};
            profile.agent = agent_info->dev_id;
            profile.type = HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_PMC;

            // Query block id info
            hsa_ven_amd_aqlprofile_id_query_t query = {block_name.c_str(), 0, 0};
            hsa_status_t status = rocprofiler::util::HsaRsrcFactory::Instance()
                                      .AqlProfileApi()
                                      ->hsa_ven_amd_aqlprofile_get_info(
                                          &profile, HSA_VEN_AMD_AQLPROFILE_INFO_BLOCK_ID, &query);
            if (status != HSA_STATUS_SUCCESS)
              AQL_EXC_RAISING(HSA_STATUS_ERROR, "get block id info: '" << block_name << "'");

            // Metric object
            const std::string metric_name = (query.instance_count > 1) ? name + "[0]" : name;
            const rocprofiler::Metric* metric = dict->Get(metric_name);
            if (metric == NULL)
              EXC_RAISING(HSA_STATUS_ERROR, "metric '" << name << "' is not found");

            // Process metrics counters
            const rocprofiler::counters_vec_t& counters_vec = metric->GetCounters();
            if (counters_vec.size() != 1)
              EXC_RAISING(HSA_STATUS_ERROR, "error: '" << metric->GetName() << "' is not basic");

            // Query block counters number
            uint32_t block_counters;
            profile.events = &(counters_vec[0]->event);
            status = rocprofiler::util::HsaRsrcFactory::Instance()
                         .AqlProfileApi()
                         ->hsa_ven_amd_aqlprofile_get_info(
                             &profile, HSA_VEN_AMD_AQLPROFILE_INFO_BLOCK_COUNTERS, &block_counters);
            if (status != HSA_STATUS_SUCCESS) continue;

            info.metric.instances = query.instance_count;
            info.metric.block_name = block_name.c_str();
            info.metric.block_counters = block_counters;
          }

          status = callback(info, data);
          if (status != HSA_STATUS_SUCCESS) break;
        }
        break;
      }
      case ROCPROFILER_INFO_KIND_TRACE: {
        info.trace.name = strdup("TT");
        info.trace.description = strdup("Thread Trace");
        info.trace.parameter_count = 5;
        status = callback(info, data);
        if (status != HSA_STATUS_SUCCESS) break;
        break;
      }
      default:
        EXC_RAISING(HSA_STATUS_ERROR, "unknown info kind(" << kind << ")");
    }

    ++agent_idx;
    if (agent_idx == agent_max) break;
  }

  if (status == HSA_STATUS_INFO_BREAK) status = HSA_STATUS_SUCCESS;
  if (status != HSA_STATUS_SUCCESS) ERR_LOGGING("iterate_info error, info kind(" << kind << ")");

  API_METHOD_SUFFIX
}

// Iterate over the info for a given info query, and invoke an application-defined callback on every
// iteration
PUBLIC_API hsa_status_t rocprofiler_query_info(
    const hsa_agent_t* agent, rocprofiler_info_query_t query,
    hsa_status_t (*callback)(const rocprofiler_info_data_t info, void* data), void* data) {
  API_METHOD_PREFIX
  EXC_RAISING(HSA_STATUS_ERROR, "Not implemented");
  API_METHOD_SUFFIX
}

// Creates a profiled queue. All dispatches on this queue will be profiled
PUBLIC_API hsa_status_t rocprofiler_queue_create_profiled(
    hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
    void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data), void* data,
    uint32_t private_segment_size, uint32_t group_segment_size, hsa_queue_t** queue) {
  API_METHOD_PREFIX
  status = rocprofiler::InterceptQueue::QueueCreateTracked(
      agent, size, type, callback, data, private_segment_size, group_segment_size, queue);
  API_METHOD_SUFFIX
}

// Return time value for a given time ID and profiling timestamp
PUBLIC_API hsa_status_t rocprofiler_get_time(rocprofiler_time_id_t time_id, uint64_t timestamp,
                                             uint64_t* value_ns, uint64_t* error_ns) {
  API_METHOD_PREFIX
  if (error_ns != NULL) {
    *error_ns = 0;
    status = rocprofiler::util::HsaRsrcFactory::Instance().GetTimeErr(time_id, error_ns);
  }
  if ((status == HSA_STATUS_SUCCESS) && (value_ns != NULL)) {
    *value_ns = 0;
    status = rocprofiler::util::HsaRsrcFactory::Instance().GetTimeVal(time_id, timestamp, value_ns);
  }
  API_METHOD_SUFFIX
}

}  // extern "C"

///////////////////////////////////////////////////////////////////////////////////////////////////
// HSA API callbacks routines
//
bool rocprofiler::HsaInterceptor::enable_ = false;
thread_local bool rocprofiler::HsaInterceptor::recursion_ = false;
;
rocprofiler_hsa_callbacks_t rocprofiler::HsaInterceptor::callbacks_{};
rocprofiler::HsaInterceptor::arg_t rocprofiler::HsaInterceptor::arg_{};
hsa_ven_amd_loader_1_01_pfn_t rocprofiler::HsaInterceptor::LoaderApiTable{};
rocprofiler::HsaInterceptor::mutex_t rocprofiler::HsaInterceptor::mutex_;

// Set HSA callbacks. If a callback is NULL then it is disabled
extern "C" PUBLIC_API hsa_status_t
rocprofiler_set_hsa_callbacks(const rocprofiler_hsa_callbacks_t callbacks, void* arg) {
  API_METHOD_PREFIX
  rocprofiler::HsaInterceptor::SetCallbacks(callbacks, arg);
  rocprofiler::InterceptQueue::SetSubmitCallback(callbacks.submit, arg);
  API_METHOD_SUFFIX
}
