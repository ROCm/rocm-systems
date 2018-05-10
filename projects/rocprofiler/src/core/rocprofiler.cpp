#include "inc/rocprofiler.h"

#include <hsa.h>
#include <hsa_api_trace.h>
#include <string.h>
#include <vector>

#include "core/context.h"
#include "core/hsa_queue.h"
#include "core/intercept_queue.h"
#include "core/proxy_queue.h"
#include "core/simple_proxy_queue.h"
#include "util/exception.h"
#include "util/hsa_rsrc_factory.h"
#include "util/logger.h"

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

///////////////////////////////////////////////////////////////////////////////////////////////////
// Internal library methods
//
namespace rocprofiler {
decltype(hsa_queue_create)* hsa_queue_create_fn;
decltype(hsa_queue_destroy)* hsa_queue_destroy_fn;

decltype(hsa_signal_store_relaxed)* hsa_signal_store_relaxed_fn;
decltype(hsa_signal_store_relaxed)* hsa_signal_store_screlease_fn;

decltype(hsa_queue_load_write_index_relaxed)* hsa_queue_load_write_index_relaxed_fn;
decltype(hsa_queue_store_write_index_relaxed)* hsa_queue_store_write_index_relaxed_fn;
decltype(hsa_queue_load_read_index_relaxed)* hsa_queue_load_read_index_relaxed_fn;

decltype(hsa_queue_load_write_index_scacquire)* hsa_queue_load_write_index_scacquire_fn;
decltype(hsa_queue_store_write_index_screlease)* hsa_queue_store_write_index_screlease_fn;
decltype(hsa_queue_load_read_index_scacquire)* hsa_queue_load_read_index_scacquire_fn;

decltype(hsa_amd_queue_intercept_create)* hsa_amd_queue_intercept_create_fn;
decltype(hsa_amd_queue_intercept_register)* hsa_amd_queue_intercept_register_fn;

::HsaApiTable* kHsaApiTable;

void SaveHsaApi(::HsaApiTable* table) {
  kHsaApiTable = table;
  hsa_queue_create_fn = table->core_->hsa_queue_create_fn;
  hsa_queue_destroy_fn = table->core_->hsa_queue_destroy_fn;

  hsa_signal_store_relaxed_fn = table->core_->hsa_signal_store_relaxed_fn;
  hsa_signal_store_screlease_fn = table->core_->hsa_signal_store_screlease_fn;

  hsa_queue_load_write_index_relaxed_fn = table->core_->hsa_queue_load_write_index_relaxed_fn;
  hsa_queue_store_write_index_relaxed_fn = table->core_->hsa_queue_store_write_index_relaxed_fn;
  hsa_queue_load_read_index_relaxed_fn = table->core_->hsa_queue_load_read_index_relaxed_fn;

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

  table->core_->hsa_queue_load_write_index_scacquire_fn = hsa_queue_load_write_index_scacquire_fn;
  table->core_->hsa_queue_store_write_index_screlease_fn = hsa_queue_store_write_index_screlease_fn;
  table->core_->hsa_queue_load_read_index_scacquire_fn = hsa_queue_load_read_index_scacquire_fn;

  table->amd_ext_->hsa_amd_queue_intercept_create_fn = hsa_amd_queue_intercept_create_fn;
  table->amd_ext_->hsa_amd_queue_intercept_register_fn = hsa_amd_queue_intercept_register_fn;
}

typedef void (*tool_handler_t)();
typedef void (*tool_handler_prop_t)(rocprofiler_settings_t*);
void * tool_handle = NULL;

// Load profiling tool library
// Return true if intercepting mode is enabled
bool LoadTool() {
  bool intercept_mode = false;
  const char* tool_lib = getenv("ROCP_TOOL_LIB");

  if (tool_lib) {
    intercept_mode = true;

    tool_handle = dlopen(tool_lib, RTLD_NOW);
    if (tool_handle == NULL) {
      fprintf(stderr, "ROCProfiler: can't load tool library \"%s\"\n", tool_lib);
      fprintf(stderr, "%s\n", dlerror());
      abort();
    }
    tool_handler_t handler = reinterpret_cast<tool_handler_t>(dlsym(tool_handle, "OnLoadTool"));
    tool_handler_prop_t handler_prop = reinterpret_cast<tool_handler_prop_t>(dlsym(tool_handle, "OnLoadToolProp"));
    if ((handler == NULL) && (handler_prop == NULL)) {
      fprintf(stderr, "ROCProfiler: tool library corrupted, OnLoadTool()/OnLoadToolProp() method is expected\n");
      fprintf(stderr, "%s\n", dlerror());
      abort();
    }
    tool_handler_t on_unload_handler = reinterpret_cast<tool_handler_t>(dlsym(tool_handle, "OnUnloadTool"));
    if (on_unload_handler == NULL) {
      fprintf(stderr, "ROCProfiler: tool library corrupted, OnUnloadTool() method is expected\n");
      fprintf(stderr, "%s\n", dlerror());
      abort();
    }

    rocprofiler_settings_t settings{};
    settings.intercept_mode = (intercept_mode) ? 1 : 0;
    settings.sqtt_size = SqttProfile::GetSize();
    settings.sqtt_local = SqttProfile::IsLocal() ? 1: 0;
    settings.timeout = Context::GetTimeout();
    settings.timestamp_on = InterceptQueue::IsTrackerOn() ? 1 : 0;

    if (handler) handler();
    else if (handler_prop) handler_prop(&settings);

    intercept_mode = (settings.intercept_mode != 0);
    SqttProfile::SetSize(settings.sqtt_size);
    SqttProfile::SetLocal(settings.sqtt_local != 0);
    Context::SetTimeout(settings.timeout);
    InterceptQueue::SetTimeout(settings.timeout);
    InterceptQueue::TrackerOn(settings.timestamp_on != 0);
  }

  return intercept_mode;
}

// Unload profiling tool librray
void UnloadTool() {
  if (tool_handle) {
    tool_handler_t handler = reinterpret_cast<tool_handler_t>(dlsym(tool_handle, "OnUnloadTool"));
    if (handler == NULL) {
      fprintf(stderr, "ROCProfiler error: tool library corrupted, OnUnloadTool() method is expected\n");
      fprintf(stderr, "%s\n", dlerror());
      abort();
    }
    handler();
    dlclose(tool_handle);
  }
}

CONSTRUCTOR_API void constructor() {
  util::Logger::Create();
}

DESTRUCTOR_API void destructor() {
  util::HsaRsrcFactory::Destroy();
  rocprofiler::MetricsDict::Destroy();
  util::Logger::Destroy();
}

const MetricsDict* GetMetrics(const hsa_agent_t& agent) {
  rocprofiler::util::HsaRsrcFactory* hsa_rsrc = &rocprofiler::util::HsaRsrcFactory::Instance();
  const rocprofiler::util::AgentInfo* agent_info = hsa_rsrc->GetAgentInfo(agent);
  if (agent_info == NULL) {
    EXC_RAISING(HSA_STATUS_ERROR, "agent is not found");
  }
  const MetricsDict* metrics = MetricsDict::Create(agent_info);
  if (metrics == NULL) EXC_RAISING(HSA_STATUS_ERROR, "MetricsDict create failed");
  return metrics;
}

hsa_status_t GetExcStatus(const std::exception& e) {
  const util::exception* rocprofiler_exc_ptr = dynamic_cast<const util::exception*>(&e);
  return (rocprofiler_exc_ptr) ? static_cast<hsa_status_t>(rocprofiler_exc_ptr->status())
                               : HSA_STATUS_ERROR;
}

rocprofiler_properties_t rocprofiler_properties;
uint64_t Context::timeout_ = UINT64_MAX;
uint32_t SqttProfile::output_buffer_size_ = 0x2000000;  // 32M
bool SqttProfile::output_buffer_local_ = true;
Tracker::mutex_t Tracker::mutex_;
util::Logger::mutex_t util::Logger::mutex_;
util::Logger* util::Logger::instance_ = NULL;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Public library methods
//
extern "C" {

// HSA-runtime tool on-load method
PUBLIC_API bool OnLoad(HsaApiTable* table, uint64_t runtime_version, uint64_t failed_tool_count,
                       const char* const* failed_tool_names) {
  rocprofiler::SaveHsaApi(table);
  rocprofiler::ProxyQueue::InitFactory();
  bool intercept_mode = false;
  const char* intercept_env = getenv("ROCP_HSA_INTERCEPT");
  if (intercept_env != NULL) {
    if (strncmp(intercept_env, "1", 1) == 0) intercept_mode = true;
  }
  if (rocprofiler::LoadTool()) intercept_mode = true;
  // HSA intercepting
  if (intercept_mode) {
    rocprofiler::ProxyQueue::HsaIntercept(table);
    rocprofiler::InterceptQueue::HsaIntercept(table);
  }
  return true;
}

// HSA-runtime tool on-unload method
PUBLIC_API void OnUnload() {
  rocprofiler::UnloadTool();
  rocprofiler::RestoreHsaApi();
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
                                         uint32_t feature_count, rocprofiler_t** handle, uint32_t mode,
                                         rocprofiler_properties_t* properties) {
  API_METHOD_PREFIX
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

  *handle = new rocprofiler::Context(agent_info, queue, features, feature_count, properties->handler,
                                     properties->handler_arg);
  API_METHOD_SUFFIX
}

// Delete profiling info
PUBLIC_API hsa_status_t rocprofiler_close(rocprofiler_t* handle) {
  API_METHOD_PREFIX
  rocprofiler::Context* context = reinterpret_cast<rocprofiler::Context*>(handle);
  if (context) delete context;
  API_METHOD_SUFFIX
}

// Reset context
PUBLIC_API hsa_status_t rocprofiler_reset(rocprofiler_t* handle, uint32_t group_index) {
  API_METHOD_PREFIX
  rocprofiler::Context* context = reinterpret_cast<rocprofiler::Context*>(handle);
  context->Reset(group_index);
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
  *group = context->GetGroupInfo(group_index);
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
PUBLIC_API hsa_status_t rocprofiler_set_queue_callbacks(rocprofiler_queue_callbacks_t callbacks, void* data) {
  API_METHOD_PREFIX
  rocprofiler::InterceptQueue::SetCallbacks(callbacks.dispatch, callbacks.destroy, data);
  API_METHOD_SUFFIX
}

// Remove queue callbacks
PUBLIC_API hsa_status_t rocprofiler_remove_queue_callbacks() {
  API_METHOD_PREFIX
  rocprofiler::InterceptQueue::SetCallbacks(NULL, NULL, NULL);
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

// Return the info for a given info kind
PUBLIC_API hsa_status_t rocprofiler_get_info(
  const hsa_agent_t *agent,
  rocprofiler_info_kind_t kind,
  void *data)
{
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

// Iterate over the info for a given info kind, and invoke an application-defined callback on every iteration
PUBLIC_API hsa_status_t rocprofiler_iterate_info(
  const hsa_agent_t* agent,
  rocprofiler_info_kind_t kind,
  hsa_status_t (*callback)(const rocprofiler_info_data_t info, void* data),
  void* data)
{
  API_METHOD_PREFIX
  rocprofiler::util::HsaRsrcFactory* hsa_rsrc = &rocprofiler::util::HsaRsrcFactory::Instance();
  rocprofiler_info_data_t info{};
  info.kind = kind;
  uint32_t agent_idx = 0;
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
      case ROCPROFILER_INFO_KIND_METRIC:
      {
        const rocprofiler::MetricsDict* dict = rocprofiler::GetMetrics(agent_info->dev_id);
        auto nodes_vec = dict->GetNodes(agent_info->gfxip);
        auto global_vec = dict->GetNodes("global");
        nodes_vec.insert(nodes_vec.end(), global_vec.begin(), global_vec.end());

        for (auto* node : nodes_vec) {
          const std::string& name = node->opts["name"];
          const std::string& descr = node->opts["descr"];
          const std::string& expr = node->opts["expr"];
          info.metric.name = strdup(name.c_str());
          info.metric.description = strdup(descr.c_str());
          info.metric.expr = expr.empty() ? NULL : strdup(expr.c_str());
          status = callback(info, data);
          if (status != HSA_STATUS_SUCCESS) break;
        }
        break;
      }
      case ROCPROFILER_INFO_KIND_TRACE:
      {
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

// Iterate over the info for a given info query, and invoke an application-defined callback on every iteration
PUBLIC_API hsa_status_t rocprofiler_query_info(
  const hsa_agent_t *agent,
  rocprofiler_info_query_t query,
  hsa_status_t (*callback)(const rocprofiler_info_data_t info, void *data),
  void *data)
{
  API_METHOD_PREFIX
  EXC_RAISING(HSA_STATUS_ERROR, "Not implemented");
  API_METHOD_SUFFIX
}

}  // extern "C"
