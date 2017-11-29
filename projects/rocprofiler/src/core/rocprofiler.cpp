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

namespace rocprofiler {
decltype(hsa_queue_create)* hsa_queue_create_fn;
decltype(hsa_queue_destroy)* hsa_queue_destroy_fn;
decltype(hsa_signal_store_relaxed)* hsa_signal_store_relaxed_fn;
decltype(hsa_queue_load_write_index_relaxed)* hsa_queue_load_write_index_relaxed_fn;
decltype(hsa_queue_store_write_index_relaxed)* hsa_queue_store_write_index_relaxed_fn;
#ifdef ROCP_HSA_PROXY
decltype(hsa_amd_queue_intercept_create)* hsa_amd_queue_intercept_create_fn;
decltype(hsa_amd_queue_intercept_register)* hsa_amd_queue_intercept_register_fn;
#endif

::HsaApiTable* kHsaApiTable;

void SaveHsaApi(::HsaApiTable* table) {
  kHsaApiTable = table;
  hsa_queue_create_fn = table->core_->hsa_queue_create_fn;
  hsa_queue_destroy_fn = table->core_->hsa_queue_destroy_fn;
  hsa_signal_store_relaxed_fn = table->core_->hsa_signal_store_relaxed_fn;
  hsa_queue_load_write_index_relaxed_fn = table->core_->hsa_queue_load_write_index_relaxed_fn;
  hsa_queue_store_write_index_relaxed_fn = table->core_->hsa_queue_store_write_index_relaxed_fn;
#ifdef ROCP_HSA_PROXY
  hsa_amd_queue_intercept_create_fn = table->amd_ext_->hsa_amd_queue_intercept_create_fn;
  hsa_amd_queue_intercept_register_fn = table->amd_ext_->hsa_amd_queue_intercept_register_fn;
#endif
}

void RestoreHsaApi() {
  ::HsaApiTable* table = kHsaApiTable;
  table->core_->hsa_queue_create_fn = hsa_queue_create_fn;
  table->core_->hsa_queue_destroy_fn = hsa_queue_destroy_fn;
  table->core_->hsa_signal_store_relaxed_fn = hsa_signal_store_relaxed_fn;
  table->core_->hsa_queue_load_write_index_relaxed_fn = hsa_queue_load_write_index_relaxed_fn;
  table->core_->hsa_queue_store_write_index_relaxed_fn = hsa_queue_store_write_index_relaxed_fn;
#ifdef ROCP_HSA_PROXY
  table->amd_ext_->hsa_amd_queue_intercept_create_fn = hsa_amd_queue_intercept_create_fn;
  table->amd_ext_->hsa_amd_queue_intercept_register_fn = hsa_amd_queue_intercept_register_fn;
#endif
}

CONSTRUCTOR_API void constructor() {
  util::Logger::Create();
  util::HsaRsrcFactory::Create();
}

DESTRUCTOR_API void destructor() {
  util::HsaRsrcFactory::Destroy();
  util::Logger::Destroy();
}

hsa_status_t GetExcStatus(const std::exception& e) {
  const util::exception* rocprofiler_exc_ptr = dynamic_cast<const util::exception*>(&e);
  return (rocprofiler_exc_ptr) ? static_cast<hsa_status_t>(rocprofiler_exc_ptr->status())
                               : HSA_STATUS_ERROR;
}

util::Logger::mutex_t util::Logger::mutex_;
util::Logger* util::Logger::instance_ = NULL;
}

extern "C" {

// Returns the last error message
PUBLIC_API hsa_status_t rocprofiler_error_string(const char** str) {
  API_METHOD_PREFIX
  *str = rocprofiler::util::Logger::LastMessage().c_str();
  API_METHOD_SUFFIX
}

// Create new profiling context
PUBLIC_API hsa_status_t rocprofiler_open(hsa_agent_t agent, rocprofiler_feature_t* info,
                                         uint32_t info_count, rocprofiler_t** handle, uint32_t mode,
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

  *handle = new rocprofiler::Context(agent_info, queue, info, info_count, properties->handler,
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

// Set kernel dispatch observer
PUBLIC_API hsa_status_t rocprofiler_set_dispatch_callback(rocprofiler_callback_t callback,
                                                          void* data) {
  API_METHOD_PREFIX
  rocprofiler::InterceptQueue::SetDispatchCB(callback, data);
  API_METHOD_SUFFIX
}

// Set kernel dispatch observer
PUBLIC_API hsa_status_t rocprofiler_remove_dispatch_callback() {
  API_METHOD_PREFIX
  rocprofiler::InterceptQueue::UnsetDispatchCB();
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

PUBLIC_API bool OnLoad(HsaApiTable* table, uint64_t runtime_version, uint64_t failed_tool_count,
                       const char* const* failed_tool_names) {
  rocprofiler::SaveHsaApi(table);
  rocprofiler::ProxyQueue::InitFactory();
  rocprofiler::InterceptQueue::SetTool(getenv("ROCP_TOOL_LIB"));
  // HSA intercepting
  if (getenv("ROCP_HSA_INTERCEPT") != NULL) {
    rocprofiler::InterceptQueue::HsaIntercept(table);
    rocprofiler::ProxyQueue::HsaIntercept(table);
  }
  return true;
}

PUBLIC_API void OnUnload() { rocprofiler::RestoreHsaApi(); }

}  // extern "C"
