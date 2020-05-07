/*
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
*/

#include "inc/roctracer.h"
#include "inc/roctracer_hcc.h"
#include "inc/roctracer_hip.h"
#include "inc/roctracer_ext.h"
#include "inc/roctracer_roctx.h"
#define PROF_API_IMPL 1
#include "inc/roctracer_hsa.h"
#include "inc/roctracer_kfd.h"

#include <dirent.h>
#include <pthread.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <atomic>
#include <mutex>
#include <stack>

#include "core/hip_act_cb_tracker.h"
#include "core/journal.h"
#include "core/loader.h"
#include "core/memory_pool.h"
#include "core/trace_buffer.h"
#include "proxy/tracker.h"
#include "ext/hsa_rt_utils.hpp"
#include "util/exception.h"
#include "util/hsa_rsrc_factory.h"
#include "util/logger.h"

#include "proxy/hsa_queue.h"
#include "proxy/intercept_queue.h"
#include "proxy/proxy_queue.h"
#include "proxy/simple_proxy_queue.h"

#define PUBLIC_API __attribute__((visibility("default")))
#define CONSTRUCTOR_API __attribute__((constructor))
#define DESTRUCTOR_API __attribute__((destructor))

#define HIPAPI_CALL(call)                                                                          \
  do {                                                                                             \
    hipError_t err = call;                                                                         \
    if (err != hipSuccess)                                                                         \
      HIP_EXC_RAISING(ROCTRACER_STATUS_HIP_API_ERR, #call " error(" << err << ")");                \
  } while (0)

#define API_METHOD_PREFIX                                                                          \
  roctracer_status_t err = ROCTRACER_STATUS_SUCCESS;                                               \
  try {

#define API_METHOD_SUFFIX                                                                          \
  }                                                                                                \
  catch (std::exception & e) {                                                                     \
    ERR_LOGGING(__FUNCTION__ << "(), " << e.what());                                               \
    err = roctracer::GetExcStatus(e);                                                              \
  }                                                                                                \
  return err;

#define API_METHOD_CATCH(X)                                                                        \
  }                                                                                                \
  catch (std::exception & e) {                                                                     \
    ERR_LOGGING(__FUNCTION__ << "(), " << e.what());                                               \
  }                                                                                                \
  (void)err;                                                                                       \
  return X;

#define ONLOAD_TRACE(str) \
  if (getenv("ROCP_ONLOAD_TRACE")) do { \
    std::cout << "PID(" << GetPid() << "): TRACER_LIB::" << __FUNCTION__ << " " << str << std::endl << std::flush; \
  } while(0);
#define ONLOAD_TRACE_BEG() ONLOAD_TRACE("begin")
#define ONLOAD_TRACE_END() ONLOAD_TRACE("end")


static inline uint32_t GetPid() { return syscall(__NR_getpid); }

///////////////////////////////////////////////////////////////////////////////////////////////////
// Mark callback
//
typedef void (mark_api_callback_t)(uint32_t domain, uint32_t cid, const void* callback_data, void* arg);
mark_api_callback_t* mark_api_callback_ptr = NULL;

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

decltype(hsa_amd_memory_async_copy)* hsa_amd_memory_async_copy_fn;
decltype(hsa_amd_memory_async_copy_rect)* hsa_amd_memory_async_copy_rect_fn;

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
}

namespace roctracer {
decltype(hsa_amd_memory_async_copy)* hsa_amd_memory_async_copy_fn;
decltype(hsa_amd_memory_async_copy_rect)* hsa_amd_memory_async_copy_rect_fn;

typedef decltype(roctracer_enable_op_callback)* roctracer_enable_op_callback_t;
typedef decltype(roctracer_disable_op_callback)* roctracer_disable_op_callback_t;
typedef decltype(roctracer_enable_op_activity_expl)* roctracer_enable_op_activity_t;
typedef decltype(roctracer_disable_op_activity)* roctracer_disable_op_activity_t;

struct cb_journal_data_t {
  roctracer_rtapi_callback_t callback;
  void* user_data;
};
typedef Journal<cb_journal_data_t> CbJournal;
CbJournal* cb_journal;

struct act_journal_data_t {
  roctracer_pool_t* pool;
};
typedef Journal<act_journal_data_t> ActJournal;
ActJournal* act_journal;

template <class T, class F>
struct journal_functor_t {
  typedef typename T::record_t record_t;
  F f_;
  journal_functor_t(F f) : f_(f) {}
  bool fun(const record_t& record) {
    f_((activity_domain_t)record.domain, record.op);
    return true;
  }
};
typedef journal_functor_t<CbJournal, roctracer_enable_op_callback_t> cb_en_functor_t;
typedef journal_functor_t<CbJournal, roctracer_disable_op_callback_t> cb_dis_functor_t;
typedef journal_functor_t<ActJournal, roctracer_enable_op_activity_t> act_en_functor_t;
typedef journal_functor_t<ActJournal, roctracer_disable_op_activity_t> act_dis_functor_t;
template<> bool cb_en_functor_t::fun(const cb_en_functor_t::record_t& record) {
  f_((activity_domain_t)record.domain, record.op, record.data.callback, record.data.user_data);
  return true;
}
template<> bool act_en_functor_t::fun(const act_en_functor_t::record_t& record) {
  f_((activity_domain_t)record.domain, record.op, record.data.pool);
  return true;
}

void hsa_async_copy_handler(::proxy::Tracker::entry_t* entry);
void hsa_kernel_handler(::proxy::Tracker::entry_t* entry);
TraceBuffer<trace_entry_t>::flush_prm_t trace_buffer_prm[] = {
  {COPY_ENTRY_TYPE, hsa_async_copy_handler},
  {KERNEL_ENTRY_TYPE, hsa_kernel_handler}
};
TraceBuffer<trace_entry_t> trace_buffer("HSA GPU", 0x200000, trace_buffer_prm, 2);

namespace hsa_support {
// callbacks table
cb_table_t cb_table;
// asyc copy activity callback
bool async_copy_callback_enabled = false;
activity_async_callback_t async_copy_callback_fun = NULL;
void* async_copy_callback_arg = NULL;
const char* output_prefix = NULL;
// Table of function pointers to HSA Core Runtime
CoreApiTable CoreApiTable_saved{};
// Table of function pointers to AMD extensions
AmdExtTable AmdExtTable_saved{};
// Table of function pointers to HSA Image Extension
ImageExtTable ImageExtTable_saved{};
}  // namespace hsa_support

namespace ext_support {
roctracer_start_cb_t roctracer_start_cb = NULL;
roctracer_stop_cb_t roctracer_stop_cb = NULL;
}  // namespace ext_suppoprt

roctracer_status_t GetExcStatus(const std::exception& e) {
  const util::exception* roctracer_exc_ptr = dynamic_cast<const util::exception*>(&e);
  return (roctracer_exc_ptr) ? static_cast<roctracer_status_t>(roctracer_exc_ptr->status()) : ROCTRACER_STATUS_ERROR;
}

class GlobalCounter {
  public:
  typedef std::mutex mutex_t;
  typedef uint64_t counter_t;
  typedef std::atomic<counter_t> atomic_counter_t;

  static counter_t Increment() { return counter_.fetch_add(1, std::memory_order_relaxed); }

  private:
  static mutex_t mutex_;
  static atomic_counter_t counter_;
};
GlobalCounter::mutex_t GlobalCounter::mutex_;
GlobalCounter::atomic_counter_t GlobalCounter::counter_{1};

// Records storage
struct roctracer_api_data_t {
  union {
    hip_api_data_t hip;
  };
  roctracer_api_data_t() {};
};
struct record_pair_t {
  roctracer_record_t record;
  roctracer_api_data_t data;
  record_pair_t() {};
};
static thread_local std::stack<record_pair_t> record_pair_stack;

// Correlation id storage
static thread_local activity_correlation_id_t correlation_id_tls = 0;
typedef std::map<activity_correlation_id_t, activity_correlation_id_t> correlation_id_map_t;
typedef std::mutex correlation_id_mutex_t;
correlation_id_map_t* correlation_id_map = NULL;
correlation_id_mutex_t correlation_id_mutex;
bool correlation_id_wait = true;

static thread_local std::stack<activity_correlation_id_t> external_id_stack;

static inline void CorrelationIdRegistr(const activity_correlation_id_t& correlation_id) {
  std::lock_guard<correlation_id_mutex_t> lck(correlation_id_mutex);
  if (correlation_id_map == NULL) correlation_id_map = new correlation_id_map_t;
  const auto ret = correlation_id_map->insert({correlation_id, correlation_id_tls});
  if (ret.second == false) EXC_ABORT(ROCTRACER_STATUS_ERROR, "HCC activity id is not unique(" << correlation_id << ")");
}

static inline activity_correlation_id_t CorrelationIdLookup(const activity_correlation_id_t& correlation_id) {
  auto it = correlation_id_map->find(correlation_id);
  if (correlation_id_wait) while (it == correlation_id_map->end()) it = correlation_id_map->find(correlation_id);
  if (it == correlation_id_map->end()) EXC_ABORT(ROCTRACER_STATUS_ERROR, "HCC activity id lookup failed(" << correlation_id << ")");
  return it->second;
}

typedef std::mutex hip_activity_mutex_t;
hip_activity_mutex_t hip_activity_mutex;

hip_act_cb_tracker_t* hip_act_cb_tracker = NULL;

inline uint32_t HipApiActivityEnableCheck(uint32_t op) {
  if (hip_act_cb_tracker == NULL) EXC_ABORT(ROCTRACER_STATUS_ERROR, "hip_act_cb_tracker is NULL");
  const uint32_t mask = hip_act_cb_tracker->enable_check(op, API_CB_MASK);
  const uint32_t ret  = (mask & ACT_CB_MASK);
  return ret;
}

inline uint32_t HipApiActivityDisableCheck(uint32_t op) {
  if (hip_act_cb_tracker == NULL) EXC_ABORT(ROCTRACER_STATUS_ERROR, "hip_act_cb_tracker is NULL");
  const uint32_t mask = hip_act_cb_tracker->disable_check(op, API_CB_MASK);
  const uint32_t ret  = (mask & ACT_CB_MASK);
  return ret;
}

inline uint32_t HipActActivityEnableCheck(uint32_t op) {
  if (hip_act_cb_tracker == NULL) EXC_ABORT(ROCTRACER_STATUS_ERROR, "hip_act_cb_tracker is NULL");
  hip_act_cb_tracker->enable_check(op, ACT_CB_MASK);
  return 0;
}

inline uint32_t HipActActivityDisableCheck(uint32_t op) {
  if (hip_act_cb_tracker == NULL) EXC_ABORT(ROCTRACER_STATUS_ERROR, "hip_act_cb_tracker is NULL");
  const uint32_t mask = hip_act_cb_tracker->disable_check(op, ACT_CB_MASK);
  const uint32_t ret  = (mask & API_CB_MASK);
  return ret;
}

void* HIP_SyncApiDataCallback(
    uint32_t op_id,
    roctracer_record_t* record,
    const void* callback_data,
    void* arg)
{
  const hip_api_data_t* data = reinterpret_cast<const hip_api_data_t*>(callback_data);
  hip_api_data_t* data_ptr = const_cast<hip_api_data_t*>(data);
  MemoryPool* pool = reinterpret_cast<MemoryPool*>(arg);

  int phase = ACTIVITY_API_PHASE_ENTER;
  if (record != NULL) {
    if (data == NULL) EXC_ABORT(ROCTRACER_STATUS_ERROR, "ActivityCallback: data is NULL");
    phase = data->phase;
  } else if (pool != NULL) {
    phase = ACTIVITY_API_PHASE_EXIT;
  }

  if (phase == ACTIVITY_API_PHASE_ENTER) {
    // Allocating a record if NULL passed
    if (record == NULL) {
      if (data != NULL) EXC_ABORT(ROCTRACER_STATUS_ERROR, "ActivityCallback enter: record is NULL");
      record_pair_stack.push({});
      auto& top = record_pair_stack.top();
      data = &(top.data.hip);
      data_ptr = const_cast<hip_api_data_t*>(data);
      data_ptr->phase = phase;
      data_ptr->correlation_id = 0;
    }

    // Correlation ID generating
    uint64_t correlation_id = data->correlation_id;
    if (correlation_id == 0) {
      correlation_id = GlobalCounter::Increment();
      data_ptr->correlation_id = correlation_id;
    }

    // Passing correlatin ID
    correlation_id_tls = correlation_id;

    return data_ptr;
  } else {
    // popping the record entry
    if (!record_pair_stack.empty()) record_pair_stack.pop();

    // Clearing correlatin ID
    correlation_id_tls = 0;

    return NULL;
  }
}

void* HIP_SyncActivityCallback(
    uint32_t op_id,
    roctracer_record_t* record,
    const void* callback_data,
    void* arg)
{
  static hsa_rt_utils::Timer timer;

  const hip_api_data_t* data = reinterpret_cast<const hip_api_data_t*>(callback_data);
  hip_api_data_t* data_ptr = const_cast<hip_api_data_t*>(data);
  MemoryPool* pool = reinterpret_cast<MemoryPool*>(arg);

  int phase = ACTIVITY_API_PHASE_ENTER;
  if (record != NULL) {
    if (data == NULL) EXC_ABORT(ROCTRACER_STATUS_ERROR, "ActivityCallback: data is NULL");
    phase = data->phase;
  } else if (pool != NULL) {
    phase = ACTIVITY_API_PHASE_EXIT; 
  }

  if (phase == ACTIVITY_API_PHASE_ENTER) {
    // Allocating a record if NULL passed
    if (record == NULL) {
      if (data != NULL) EXC_ABORT(ROCTRACER_STATUS_ERROR, "ActivityCallback enter: record is NULL");
      record_pair_stack.push({});
      auto& top = record_pair_stack.top();
      record = &(top.record);
      data = &(top.data.hip);
      data_ptr = const_cast<hip_api_data_t*>(data);
      data_ptr->phase = phase;
      data_ptr->correlation_id = 0;
    }

    // Filing record info
    record->domain = ACTIVITY_DOMAIN_HIP_API;
    record->op = op_id;
    record->begin_ns = timer.timestamp_ns();

    // Correlation ID generating
    uint64_t correlation_id = data->correlation_id;
    if (correlation_id == 0) {
      correlation_id = GlobalCounter::Increment();
      data_ptr->correlation_id = correlation_id;
    }
    record->correlation_id = correlation_id;

    // Passing correlatin ID
    correlation_id_tls = correlation_id;

    return data_ptr; 
  } else {
    if (pool == NULL) EXC_ABORT(ROCTRACER_STATUS_ERROR, "ActivityCallback exit: pool is NULL");

    // Getting record of stacked
    if (record == NULL) {
      if (record_pair_stack.empty())  EXC_ABORT(ROCTRACER_STATUS_ERROR, "ActivityCallback exit: record stack is empty");
      auto& top  = record_pair_stack.top();
      record = &(top.record);
    }

    // Filing record info
    record->end_ns = timer.timestamp_ns();
    record->process_id = syscall(__NR_getpid);
    record->thread_id = syscall(__NR_gettid);

    if (external_id_stack.empty() == false) {
      roctracer_record_t ext_record{};
      ext_record.domain = ACTIVITY_DOMAIN_EXT_API;
      ext_record.op = ACTIVITY_EXT_OP_EXTERN_ID;
      ext_record.correlation_id = record->correlation_id;
      ext_record.external_id = external_id_stack.top();
      pool->Write(ext_record);
    }

    // Writing record to the buffer
    pool->Write(*record);

    // popping the record entry
    if (!record_pair_stack.empty()) record_pair_stack.pop();

    // Clearing correlatin ID
    correlation_id_tls = 0;

    return NULL;
  }
}

void HCC_ActivityIdCallback(activity_correlation_id_t correlation_id) {
  CorrelationIdRegistr(correlation_id);
}

void HCC_AsyncActivityCallback(uint32_t op_id, void* record, void* arg) {
  MemoryPool* pool = reinterpret_cast<MemoryPool*>(arg);
  roctracer_record_t* record_ptr = reinterpret_cast<roctracer_record_t*>(record);
  record_ptr->domain = ACTIVITY_DOMAIN_HCC_OPS;
  record_ptr->correlation_id = CorrelationIdLookup(record_ptr->correlation_id);
  pool->Write(*record_ptr);
}

// Open output file
FILE* open_output_file(const char* prefix, const char* name) {
  FILE* file_handle = NULL;
  if (prefix != NULL) {
    std::ostringstream oss;
    oss << prefix << "/" << GetPid() << "_" << name;
    file_handle = fopen(oss.str().c_str(), "w");
    if (file_handle == NULL) {
      std::ostringstream errmsg;
      errmsg << "ROCTracer: fopen error, file '" << oss.str().c_str() << "'";
      perror(errmsg.str().c_str());
      abort();
    }
  } else file_handle = stdout;
  return file_handle;
}

void close_output_file(FILE* file_handle) {
  if ((file_handle != NULL) && (file_handle != stdout)) fclose(file_handle);
}

FILE* kernel_file_handle = NULL;
void hsa_kernel_handler(::proxy::Tracker::entry_t* entry) {
  static uint64_t index = 0;
  if (index == 0) {
    kernel_file_handle = open_output_file(hsa_support::output_prefix, "results.txt");
  }
  fprintf(kernel_file_handle, "dispatch[%lu], gpu-id(%u), tid(%u), kernel-name(\"%s\"), time(%lu,%lu,%lu,%lu)\n",
    index,
    //::util::HsaRsrcFactory::Instance().GetAgentInfo(entry->agent)->dev_index,
    entry->dev_index,
    entry->kernel.tid,
    entry->kernel.name,
    entry->dispatch,
    entry->begin,
    entry->end,
    entry->complete);
#if 0
  fprintf(file_handle, "dispatch[%u], gpu-id(%u), queue-id(%u), queue-index(%lu), tid(%lu), grd(%u), wgr(%u), lds(%u), scr(%u), vgpr(%u), sgpr(%u), fbar(%u), sig(0x%lx), kernel-name(\"%s\")",
    index,
    HsaRsrcFactory::Instance().GetAgentInfo(entry->agent)->dev_index,
    entry->data.queue_id,
    entry->data.queue_index,
    entry->data.thread_id,
    entry->kernel_properties.grid_size,
    entry->kernel_properties.workgroup_size,
    entry->kernel_properties.lds_size,
    entry->kernel_properties.scratch_size,
    entry->kernel_properties.vgpr_count,
    entry->kernel_properties.sgpr_count,
    entry->kernel_properties.fbarrier_count,
    entry->kernel_properties.signal.handle,
    nik_name.c_str());
  if (record) fprintf(file_handle, ", time(%lu,%lu,%lu,%lu)",
    record->dispatch,
    record->begin,
    record->end,
    record->complete);
  fprintf(file_handle, "\n");
  fflush(file_handle);
#endif
  index++;
}

void hsa_async_copy_handler(::proxy::Tracker::entry_t* entry) {
  activity_record_t record{};
  record.domain = ACTIVITY_DOMAIN_HSA_OPS;   // activity domain id
  record.begin_ns = entry->begin;    // host begin timestamp
  record.end_ns = entry->end;        // host end timestamp
  record.device_id = 0;                      // device id

  hsa_support::async_copy_callback_fun(hsa_support::HSA_OP_ID_async_copy, &record, hsa_support::async_copy_callback_arg);
}

hsa_status_t hsa_amd_memory_async_copy_interceptor(
    void* dst, hsa_agent_t dst_agent, const void* src,
    hsa_agent_t src_agent, size_t size, uint32_t num_dep_signals,
    const hsa_signal_t* dep_signals, hsa_signal_t completion_signal)
{
  hsa_status_t status = HSA_STATUS_SUCCESS;
  if (hsa_support::async_copy_callback_enabled) {
    trace_entry_t* entry = trace_buffer.GetEntry();
    ::proxy::Tracker::Enable(COPY_ENTRY_TYPE, hsa_agent_t{}, completion_signal, entry);
    status = hsa_amd_memory_async_copy_fn(dst, dst_agent, src,
                                          src_agent, size, num_dep_signals,
                                          dep_signals, entry->signal);
    if (status != HSA_STATUS_SUCCESS) ::proxy::Tracker::Disable(entry);
  }
  else
  {
    status = hsa_amd_memory_async_copy_fn(dst, dst_agent, src,
                                          src_agent, size, num_dep_signals,
                                          dep_signals, completion_signal);
  }
  return status;
}

hsa_status_t hsa_amd_memory_async_copy_rect_interceptor(
    const hsa_pitched_ptr_t* dst, const hsa_dim3_t* dst_offset, const hsa_pitched_ptr_t* src,
    const hsa_dim3_t* src_offset, const hsa_dim3_t* range, hsa_agent_t copy_agent,
    hsa_amd_copy_direction_t dir, uint32_t num_dep_signals, const hsa_signal_t* dep_signals,
    hsa_signal_t completion_signal)
{
  hsa_status_t status = HSA_STATUS_SUCCESS;
  if (hsa_support::async_copy_callback_enabled) {
    trace_entry_t* entry = trace_buffer.GetEntry();
    ::proxy::Tracker::Enable(COPY_ENTRY_TYPE, hsa_agent_t{}, completion_signal, entry);
    status = hsa_amd_memory_async_copy_rect_fn(dst, dst_offset, src,
                                               src_offset, range, copy_agent,
                                               dir, num_dep_signals, dep_signals,
                                               entry->signal);
    if (status != HSA_STATUS_SUCCESS) ::proxy::Tracker::Disable(entry);
  }
  else
  {
    status = hsa_amd_memory_async_copy_rect_fn(dst, dst_offset, src,
                                               src_offset, range, copy_agent,
                                               dir, num_dep_signals, dep_signals,
                                               completion_signal);
  }
  return status;
}

void HSA_AsyncActivityCallback(uint32_t op_id, void* record, void* arg) {
  MemoryPool* pool = reinterpret_cast<MemoryPool*>(arg);
  roctracer_record_t* record_ptr = reinterpret_cast<roctracer_record_t*>(record);
  record_ptr->domain = ACTIVITY_DOMAIN_HSA_OPS;
  pool->Write(*record_ptr);
}

// Logger routines and primitives
util::Logger::mutex_t util::Logger::mutex_;
std::atomic<util::Logger*> util::Logger::instance_{};

// Memory pool routines and primitives
MemoryPool* memory_pool = NULL;
typedef std::recursive_mutex memory_pool_mutex_t;
memory_pool_mutex_t memory_pool_mutex;

// Stop sttaus routines and primitives
unsigned stop_status_value = 0;
typedef std::mutex stop_status_mutex_t;
stop_status_mutex_t stop_status_mutex;
unsigned set_stopped(unsigned val) {
  std::lock_guard<stop_status_mutex_t> lock(stop_status_mutex);
  const unsigned ret = (stop_status_value ^ val);
  stop_status_value = val;
  return ret;
}
}  // namespace roctracer

LOADER_INSTANTIATE();
TRACE_BUFFER_INSTANTIATE();

///////////////////////////////////////////////////////////////////////////////////////////////////
// Public library methods
//
extern "C" {

// Returns library vesrion
PUBLIC_API uint32_t roctracer_version_major() { return ROCTRACER_VERSION_MAJOR; }
PUBLIC_API uint32_t roctracer_version_minor() { return ROCTRACER_VERSION_MINOR; }

// Returns the last error
PUBLIC_API const char* roctracer_error_string() {
  return strdup(roctracer::util::Logger::LastMessage().c_str());
}

// Return Op string by given domain and activity/API codes
// NULL returned on the error and the library errno is set
PUBLIC_API const char* roctracer_op_string(
    uint32_t domain,
    uint32_t op,
    uint32_t kind)
{
  API_METHOD_PREFIX
  switch (domain) {
    case ACTIVITY_DOMAIN_HSA_API:
      return roctracer::hsa_support::GetApiName(op);
    case ACTIVITY_DOMAIN_HSA_OPS:
      return roctracer::RocpLoader::Instance().GetOpName(op);
    case ACTIVITY_DOMAIN_HCC_OPS:
      return roctracer::HccLoader::Instance().GetOpName(kind);
    case ACTIVITY_DOMAIN_HIP_API:
      return roctracer::HipLoader::Instance().ApiName(op);
    case ACTIVITY_DOMAIN_KFD_API:
      return roctracer::kfd_support::GetApiName(op);
    default:
      EXC_RAISING(ROCTRACER_STATUS_BAD_DOMAIN, "invalid domain ID(" << domain << ")");
  }
  API_METHOD_CATCH(NULL)
}

// Return Op code and kind by given string
PUBLIC_API roctracer_status_t roctracer_op_code(
    uint32_t domain,
    const char* str,
    uint32_t* op,
    uint32_t* kind)
{
  API_METHOD_PREFIX
  switch (domain) {
    case ACTIVITY_DOMAIN_HSA_API: {
      *op = roctracer::hsa_support::GetApiCode(str);
      if (kind != NULL) *kind = 0;
      break;
    }
    case ACTIVITY_DOMAIN_KFD_API: {
      *op = roctracer::kfd_support::GetApiCode(str);
      if (kind != NULL) *kind = 0;
      break;
    }
    default:
      EXC_RAISING(ROCTRACER_STATUS_BAD_DOMAIN, "limited domain ID(" << domain << ")");
  }
  API_METHOD_SUFFIX
}

static inline uint32_t get_op_num(const uint32_t& domain) {
  switch (domain) {
    case ACTIVITY_DOMAIN_HSA_OPS: return HSA_OP_ID_NUMBER;
    case ACTIVITY_DOMAIN_HSA_API: return HSA_API_ID_NUMBER;
    case ACTIVITY_DOMAIN_HCC_OPS: return HIP_OP_ID_NUMBER;
    case ACTIVITY_DOMAIN_HIP_API: return HIP_API_ID_NUMBER;
    case ACTIVITY_DOMAIN_KFD_API: return KFD_API_ID_NUMBER;
    case ACTIVITY_DOMAIN_EXT_API: return 0;
    case ACTIVITY_DOMAIN_ROCTX: return ROCTX_API_ID_NUMBER;
    default:
      EXC_RAISING(ROCTRACER_STATUS_BAD_DOMAIN, "invalid domain ID(" << domain << ")");
  }
  return 0;
}

// Enable runtime API callbacks
static roctracer_status_t roctracer_enable_callback_fun(
    roctracer_domain_t domain,
    uint32_t op,
    roctracer_rtapi_callback_t callback,
    void* user_data)
{
  switch (domain) {
    case ACTIVITY_DOMAIN_KFD_API: {
      const bool succ = roctracer::KfdLoader::Instance().RegisterApiCallback(op, (void*)callback, user_data);
      if (succ == false) EXC_RAISING(ROCTRACER_STATUS_ERROR, "KFD RegisterApiCallback error(" << op << ") failed");
      break;
    }
    case ACTIVITY_DOMAIN_HSA_OPS: break;
    case ACTIVITY_DOMAIN_HSA_API: {
#if 0
      if (op == HSA_API_ID_DISPATCH) {
        const bool succ = roctracer::RocpLoader::Instance().RegisterApiCallback(op, (void*)callback, user_data);
        if (succ == false) HCC_EXC_RAISING(ROCTRACER_STATUS_HSA_ERR, "HSA::EnableActivityCallback error(" << op << ") failed");
        break;
      }
#endif
      roctracer::hsa_support::cb_table.set(op, callback, user_data);
      break;
    }
    case ACTIVITY_DOMAIN_HCC_OPS: break;
    case ACTIVITY_DOMAIN_HIP_API: {
      if (roctracer::HipLoader::Instance().Enabled() == false) break;
      std::lock_guard<roctracer::hip_activity_mutex_t> lock(roctracer::hip_activity_mutex);

      hipError_t hip_err = roctracer::HipLoader::Instance().RegisterApiCallback(op, (void*)callback, user_data);
      if (hip_err != hipSuccess) HIP_EXC_RAISING(ROCTRACER_STATUS_HIP_API_ERR, "HIP::RegisterApiCallback(" << op << ") error(" << hip_err << ")");

      if (roctracer::HipApiActivityEnableCheck(op) == 0) {
        hip_err = roctracer::HipLoader::Instance().RegisterActivityCallback(op, (void*)roctracer::HIP_SyncApiDataCallback, (void*)1);
        if (hip_err != hipSuccess) HIP_EXC_RAISING(ROCTRACER_STATUS_HIP_API_ERR, "HIPAPI: HIP::RegisterActivityCallback(" << op << ") error(" << hip_err << ")");
      }
      break;
    }
    case ACTIVITY_DOMAIN_ROCTX: {
      if (roctracer::RocTxLoader::Instance().Enabled()) {
        const bool suc = roctracer::RocTxLoader::Instance().RegisterApiCallback(op, (void*)callback, user_data);
        if (suc == false) EXC_RAISING(ROCTRACER_STATUS_ROCTX_ERR, "ROCTX::RegisterApiCallback(" << op << ") failed");
      }
      break;
    }
    default:
      EXC_RAISING(ROCTRACER_STATUS_BAD_DOMAIN, "invalid domain ID(" << domain << ")");
  }
  return ROCTRACER_STATUS_SUCCESS;
}

static void roctracer_enable_callback_impl(
    uint32_t domain,
    uint32_t op,
    roctracer_rtapi_callback_t callback,
    void* user_data)
{
  roctracer::cb_journal->registr({domain, op, {callback, user_data}});
  roctracer_enable_callback_fun((roctracer_domain_t)domain, op, callback, user_data);
}

PUBLIC_API roctracer_status_t roctracer_enable_op_callback(
    roctracer_domain_t domain,
    uint32_t op,
    roctracer_rtapi_callback_t callback,
    void* user_data)
{
  API_METHOD_PREFIX
  roctracer_enable_callback_impl(domain, op, callback, user_data);
  API_METHOD_SUFFIX
}

PUBLIC_API roctracer_status_t roctracer_enable_domain_callback(
    roctracer_domain_t domain,
    roctracer_rtapi_callback_t callback,
    void* user_data)
{
  API_METHOD_PREFIX
  const uint32_t op_num = get_op_num(domain);
  for (uint32_t op = 0; op < op_num; op++) roctracer_enable_callback_impl(domain, op, callback, user_data);
  API_METHOD_SUFFIX
}

PUBLIC_API roctracer_status_t roctracer_enable_callback(
    roctracer_rtapi_callback_t callback,
    void* user_data)
{
  API_METHOD_PREFIX
  for (uint32_t domain = 0; domain < ACTIVITY_DOMAIN_NUMBER; domain++) {
    const uint32_t op_num = get_op_num(domain);
    for (uint32_t op = 0; op < op_num; op++) roctracer_enable_callback_impl(domain, op, callback, user_data);
  }
  API_METHOD_SUFFIX
}

// Disable runtime API callbacks
static roctracer_status_t roctracer_disable_callback_fun(
    roctracer_domain_t domain,
    uint32_t op)
{
  switch (domain) {
    case ACTIVITY_DOMAIN_KFD_API: {
      const bool succ = roctracer::KfdLoader::Instance().RemoveApiCallback(op);
      if (succ == false) EXC_RAISING(ROCTRACER_STATUS_ERROR, "KFD RemoveApiCallback error");
      break;
    }
    case ACTIVITY_DOMAIN_HSA_OPS: break;
    case ACTIVITY_DOMAIN_HSA_API: {
#if 0
      if (op == HSA_API_ID_DISPATCH) {
        const bool succ = roctracer::RocpLoader::Instance().RemoveApiCallback(op);
        if (succ == false) HCC_EXC_RAISING(ROCTRACER_STATUS_HSA_ERR, "HSA::RemoveActivityCallback error(" << op << ") failed");
        break;
      }
#endif
      roctracer::hsa_support::cb_table.set(op, NULL, NULL);
      break;
    }
    case ACTIVITY_DOMAIN_HCC_OPS: break;
    case ACTIVITY_DOMAIN_HIP_API: {
      if (roctracer::HipLoader::Instance().Enabled() == false) break;
      std::lock_guard<roctracer::hip_activity_mutex_t> lock(roctracer::hip_activity_mutex);

      const hipError_t hip_err = roctracer::HipLoader::Instance().RemoveApiCallback(op);
      if (hip_err != hipSuccess) HIP_EXC_RAISING(ROCTRACER_STATUS_HIP_API_ERR, "HIP::RemoveApiCallback(" << op << "), error(" << hip_err << ")");

      if (roctracer::HipApiActivityDisableCheck(op) == 0) {
        const hipError_t hip_err = roctracer::HipLoader::Instance().RemoveActivityCallback(op);
        if (hip_err != hipSuccess) HIP_EXC_RAISING(ROCTRACER_STATUS_HIP_API_ERR, "HIPAPI: HIP::RemoveActivityCallback op(" << op << "), error(" << hip_err << ")");
      }
      break;
    }
    case ACTIVITY_DOMAIN_ROCTX: {
      if (roctracer::RocTxLoader::Instance().Enabled()) {
        const bool suc = roctracer::RocTxLoader::Instance().RemoveApiCallback(op);
        if (suc == false) EXC_RAISING(ROCTRACER_STATUS_ROCTX_ERR, "ROCTX::RemoveApiCallback(" << op << ") failed");
      }
      break;
    }
    default:
      EXC_RAISING(ROCTRACER_STATUS_BAD_DOMAIN, "invalid domain ID(" << domain << ")");
  }
  return ROCTRACER_STATUS_SUCCESS;
}

static void roctracer_disable_callback_impl(
    uint32_t domain,
    uint32_t op)
{
    roctracer::cb_journal->remove({domain, op, {}});
    roctracer_disable_callback_fun((roctracer_domain_t)domain, op);
}

PUBLIC_API roctracer_status_t roctracer_disable_op_callback(
    roctracer_domain_t domain,
    uint32_t op)
{
  API_METHOD_PREFIX
  roctracer_disable_callback_impl(domain, op);
  API_METHOD_SUFFIX
}

PUBLIC_API roctracer_status_t roctracer_disable_domain_callback(
    roctracer_domain_t domain)
{
  API_METHOD_PREFIX
  const uint32_t op_num = get_op_num(domain);
  for (uint32_t op = 0; op < op_num; op++) roctracer_disable_callback_impl(domain, op);
  API_METHOD_SUFFIX
}

PUBLIC_API roctracer_status_t roctracer_disable_callback()
{
  API_METHOD_PREFIX
  for (uint32_t domain = 0; domain < ACTIVITY_DOMAIN_NUMBER; domain++) {
    const uint32_t op_num = get_op_num(domain);
    for (uint32_t op = 0; op < op_num; op++) roctracer_disable_callback_impl(domain, op);
  }
  API_METHOD_SUFFIX
}

// Return default pool and set new one if parameter pool is not NULL.
PUBLIC_API roctracer_pool_t* roctracer_default_pool_expl(roctracer_pool_t* pool) {
  std::lock_guard<roctracer::memory_pool_mutex_t> lock(roctracer::memory_pool_mutex);
  roctracer_pool_t* p = reinterpret_cast<roctracer_pool_t*>(roctracer::memory_pool);
  if (pool != NULL) roctracer::memory_pool = reinterpret_cast<roctracer::MemoryPool*>(pool);
  return p;
}

// Open memory pool
PUBLIC_API roctracer_status_t roctracer_open_pool_expl(
    const roctracer_properties_t* properties,
    roctracer_pool_t** pool)
{
  API_METHOD_PREFIX
  std::lock_guard<roctracer::memory_pool_mutex_t> lock(roctracer::memory_pool_mutex);
  if ((pool == NULL) && (roctracer::memory_pool != NULL)) {
    EXC_RAISING(ROCTRACER_STATUS_ERROR, "default pool already set");
  }
  roctracer::MemoryPool* p = new roctracer::MemoryPool(*properties);
  if (p == NULL) EXC_RAISING(ROCTRACER_STATUS_ERROR, "MemoryPool() error");
  if (pool != NULL) *pool = p;
  else roctracer::memory_pool = p;
  API_METHOD_SUFFIX
}

// Close memory pool
PUBLIC_API roctracer_status_t roctracer_close_pool_expl(roctracer_pool_t* pool) {
  API_METHOD_PREFIX
  std::lock_guard<roctracer::memory_pool_mutex_t> lock(roctracer::memory_pool_mutex);
  roctracer_pool_t* ptr = (pool == NULL) ? roctracer_default_pool() : pool;
  roctracer::MemoryPool* memory_pool = reinterpret_cast<roctracer::MemoryPool*>(ptr);
  delete(memory_pool);
  if (pool == NULL) roctracer::memory_pool = NULL;
  API_METHOD_SUFFIX
}

// Enable activity records logging
static roctracer_status_t roctracer_enable_activity_fun(
    roctracer_domain_t domain,
    uint32_t op,
    roctracer_pool_t* pool)
{
  if (pool == NULL) pool = roctracer_default_pool();
  switch (domain) {
    case ACTIVITY_DOMAIN_HSA_OPS: {
      if (op == HSA_OP_ID_COPY) {
        roctracer::hsa_support::async_copy_callback_enabled = true;
      } else {
        const bool init_phase = (roctracer::RocpLoader::GetRef() == NULL);
        if (roctracer::RocpLoader::GetRef() == NULL) break;
        if (init_phase == true) {
          roctracer::RocpLoader::Instance().InitActivityCallback((void*)roctracer::HSA_AsyncActivityCallback,
                                                                 (void*)pool);
        }
        const bool succ = roctracer::RocpLoader::Instance().EnableActivityCallback(op, true);
        if (succ == false) HCC_EXC_RAISING(ROCTRACER_STATUS_HSA_ERR, "HSA::EnableActivityCallback error");
      }
      break;
    }
    case ACTIVITY_DOMAIN_HSA_API: break;
    case ACTIVITY_DOMAIN_KFD_API: break;
    case ACTIVITY_DOMAIN_HCC_OPS: {
      const bool init_phase = (roctracer::HccLoader::GetRef() == NULL);
      if (roctracer::HccLoader::Instance().Enabled() == false) break;

      if (init_phase == true) {
        if (getenv("ROCP_HCC_CORRID_WAIT") != NULL) {
          roctracer::correlation_id_wait = true;
          fprintf(stdout, "roctracer: HCC correlation ID wait enabled\n"); fflush(stdout);
        }
        if (getenv("ROCP_HCC_CORRID_NOWAIT") != NULL) {
          roctracer::correlation_id_wait = false;
          fprintf(stdout, "roctracer: HCC correlation ID wait disabled\n"); fflush(stdout);
        }
        roctracer::HccLoader::Instance().InitActivityCallback((void*)roctracer::HCC_ActivityIdCallback,
                                                              (void*)roctracer::HCC_AsyncActivityCallback,
                                                              (void*)pool);
      }
      const bool succ = roctracer::HccLoader::Instance().EnableActivityCallback(op, true);
      if (succ == false) HCC_EXC_RAISING(ROCTRACER_STATUS_HCC_OPS_ERR, "HCC::EnableActivityCallback error");
      break;
    }
    case ACTIVITY_DOMAIN_HIP_API: {
      if (roctracer::HipLoader::Instance().Enabled() == false) break;
      std::lock_guard<roctracer::hip_activity_mutex_t> lock(roctracer::hip_activity_mutex);

      if (roctracer::HipActActivityEnableCheck(op) == 0) {
        const hipError_t hip_err = roctracer::HipLoader::Instance().RegisterActivityCallback(op, (void*)roctracer::HIP_SyncActivityCallback, (void*)pool);
        if (hip_err != hipSuccess) HIP_EXC_RAISING(ROCTRACER_STATUS_HIP_API_ERR, "HIP::RegisterActivityCallback(" << op << " error(" << hip_err << ")");
      }
      break;
    }
    case ACTIVITY_DOMAIN_ROCTX: break;
    default:
      EXC_RAISING(ROCTRACER_STATUS_BAD_DOMAIN, "invalid domain ID(" << domain << ")");
  }
  return ROCTRACER_STATUS_SUCCESS;
}

static void roctracer_enable_activity_impl(
    uint32_t domain,
    uint32_t op,
    roctracer_pool_t* pool)
{
    roctracer::act_journal->registr({domain, op, {pool}});
    roctracer_enable_activity_fun((roctracer_domain_t)domain, op, pool);
}

PUBLIC_API roctracer_status_t roctracer_enable_op_activity_expl(
    roctracer_domain_t domain,
    uint32_t op,
    roctracer_pool_t* pool)
{
  API_METHOD_PREFIX
  roctracer_enable_activity_impl(domain, op, pool);
  API_METHOD_SUFFIX
}

PUBLIC_API roctracer_status_t roctracer_enable_domain_activity_expl(
    roctracer_domain_t domain,
    roctracer_pool_t* pool)
{
  API_METHOD_PREFIX
  const uint32_t op_num = get_op_num(domain);
  for (uint32_t op = 0; op < op_num; op++) roctracer_enable_activity_impl(domain, op, pool);
  API_METHOD_SUFFIX
}

PUBLIC_API roctracer_status_t roctracer_enable_activity_expl(
    roctracer_pool_t* pool)
{
  API_METHOD_PREFIX
  for (uint32_t domain = 0; domain < ACTIVITY_DOMAIN_NUMBER; domain++) {
    const uint32_t op_num = get_op_num(domain);
    for (uint32_t op = 0; op < op_num; op++) roctracer_enable_activity_impl(domain, op, pool);
  }
  API_METHOD_SUFFIX
}

// Disable activity records logging
static roctracer_status_t roctracer_disable_activity_fun(
    roctracer_domain_t domain,
    uint32_t op)
{
  switch (domain) {
    case ACTIVITY_DOMAIN_HSA_OPS: {
      if (op == HSA_OP_ID_COPY) {
        roctracer::hsa_support::async_copy_callback_enabled = true;
      } else {
        if (roctracer::RocpLoader::GetRef() == NULL) break;
        const bool succ = roctracer::RocpLoader::Instance().EnableActivityCallback(op, false);
        if (succ == false) HCC_EXC_RAISING(ROCTRACER_STATUS_HSA_ERR, "HSA::EnableActivityCallback(false) error, op(" << op << ")");
      }
      break;
    }
    case ACTIVITY_DOMAIN_HSA_API: break;
    case ACTIVITY_DOMAIN_KFD_API: break;
    case ACTIVITY_DOMAIN_HCC_OPS: {
      if (roctracer::HccLoader::Instance().Enabled() == false) break;

      const bool succ = roctracer::HccLoader::Instance().EnableActivityCallback(op, false);
      if (succ == false) HCC_EXC_RAISING(ROCTRACER_STATUS_HCC_OPS_ERR, "HCC::EnableActivityCallback(NULL) error, op(" << op << ")");
      break;
    }
    case ACTIVITY_DOMAIN_HIP_API: {
      if (roctracer::HipLoader::Instance().Enabled() == false) break;
      std::lock_guard<roctracer::hip_activity_mutex_t> lock(roctracer::hip_activity_mutex);

      if (roctracer::HipActActivityDisableCheck(op) == 0) {
        const hipError_t hip_err = roctracer::HipLoader::Instance().RemoveActivityCallback(op);
        if (hip_err != hipSuccess) HIP_EXC_RAISING(ROCTRACER_STATUS_HIP_API_ERR, "HIP::RemoveActivityCallback op(" << op << "), error(" << hip_err << ")");
      } else {
        const hipError_t hip_err = roctracer::HipLoader::Instance().RegisterActivityCallback(op, (void*)roctracer::HIP_SyncApiDataCallback, (void*)1);
        if (hip_err != hipSuccess) HIP_EXC_RAISING(ROCTRACER_STATUS_HIP_API_ERR, "HIPACT: HIP::RegisterActivityCallback(" << op << ") error(" << hip_err << ")");
      }
      break;
    }
    case ACTIVITY_DOMAIN_ROCTX: break;
    default:
      EXC_RAISING(ROCTRACER_STATUS_BAD_DOMAIN, "invalid domain ID(" << domain << ")");
  }
  return ROCTRACER_STATUS_SUCCESS;
}

static void roctracer_disable_activity_impl(
    uint32_t domain,
    uint32_t op)
{
  roctracer::act_journal->remove({domain, op, {}});
  roctracer_disable_activity_fun((roctracer_domain_t)domain, op);
}

PUBLIC_API roctracer_status_t roctracer_disable_op_activity(
    roctracer_domain_t domain,
    uint32_t op)
{
  API_METHOD_PREFIX
  roctracer_disable_activity_impl(domain, op);
  API_METHOD_SUFFIX
}

PUBLIC_API roctracer_status_t roctracer_disable_domain_activity(
    roctracer_domain_t domain)
{
  API_METHOD_PREFIX
  const uint32_t op_num = get_op_num(domain);
  for (uint32_t op = 0; op < op_num; op++) roctracer_disable_activity_impl(domain, op);
  API_METHOD_SUFFIX
}

PUBLIC_API roctracer_status_t roctracer_disable_activity()
{
  API_METHOD_PREFIX
  for (uint32_t domain = 0; domain < ACTIVITY_DOMAIN_NUMBER; domain++) {
    const uint32_t op_num = get_op_num(domain);
    for (uint32_t op = 0; op < op_num; op++) roctracer_disable_activity_impl(domain, op);
  }
  API_METHOD_SUFFIX
}

// Flush available activity records
PUBLIC_API roctracer_status_t roctracer_flush_activity_expl(roctracer_pool_t* pool) {
  API_METHOD_PREFIX
  if (pool == NULL) pool = roctracer_default_pool();
  roctracer::MemoryPool* memory_pool = reinterpret_cast<roctracer::MemoryPool*>(pool);
  memory_pool->Flush();
  roctracer::TraceBufferBase::FlushAll();
  API_METHOD_SUFFIX
}

// Notifies that the calling thread is entering an external API region.
// Push an external correlation id for the calling thread.
PUBLIC_API roctracer_status_t roctracer_activity_push_external_correlation_id(activity_correlation_id_t id) {
  API_METHOD_PREFIX
  roctracer::external_id_stack.push(id);
  API_METHOD_SUFFIX
}

// Notifies that the calling thread is leaving an external API region.
// Pop an external correlation id for the calling thread.
// 'lastId' returns the last external correlation
PUBLIC_API roctracer_status_t roctracer_activity_pop_external_correlation_id(activity_correlation_id_t* last_id) {
  API_METHOD_PREFIX
  if (last_id != NULL) *last_id = 0;

  if (roctracer::external_id_stack.empty() != true) {
    if (last_id != NULL) *last_id = roctracer::external_id_stack.top();
    roctracer::external_id_stack.pop();
  } else {
#if 0
    EXC_RAISING(ROCTRACER_STATUS_ERROR, "not matching external range pop");
#endif
    return ROCTRACER_STATUS_ERROR;
  }
  API_METHOD_SUFFIX
}

// Mark API
PUBLIC_API void roctracer_mark(const char* str) {
  if (mark_api_callback_ptr) {
    mark_api_callback_ptr(ACTIVITY_DOMAIN_EXT_API, ACTIVITY_EXT_OP_MARK, str, NULL);
    roctracer::GlobalCounter::Increment(); // account for user-defined markers when tracking correlation id
  }
}

// Start API
PUBLIC_API void roctracer_start() {
  if (roctracer::set_stopped(0)) {
    if (roctracer::ext_support::roctracer_start_cb) roctracer::ext_support::roctracer_start_cb();
    roctracer::cb_journal->foreach(roctracer::cb_en_functor_t(roctracer_enable_callback_fun));
    roctracer::act_journal->foreach(roctracer::act_en_functor_t(roctracer_enable_activity_fun));
  }
}

// Stop API
PUBLIC_API void roctracer_stop() {
  if (roctracer::set_stopped(1)) {
    roctracer::cb_journal->foreach(roctracer::cb_dis_functor_t(roctracer_disable_callback_fun));
    roctracer::act_journal->foreach(roctracer::act_dis_functor_t(roctracer_disable_activity_fun));
    if (roctracer::ext_support::roctracer_stop_cb) roctracer::ext_support::roctracer_stop_cb();
  }
}

PUBLIC_API roctracer_status_t roctracer_get_timestamp(uint64_t* timestamp) {
  API_METHOD_PREFIX
  *timestamp = util::HsaRsrcFactory::Instance().TimestampNs();
  API_METHOD_SUFFIX
}

// Set properties
PUBLIC_API roctracer_status_t roctracer_set_properties(
    roctracer_domain_t domain,
    void* properties)
{
  API_METHOD_PREFIX
  switch (domain) {
    case ACTIVITY_DOMAIN_HSA_OPS: {
      // HSA OPS properties
      roctracer::hsa_ops_properties_t* ops_properties = reinterpret_cast<roctracer::hsa_ops_properties_t*>(properties);
      HsaApiTable* table = reinterpret_cast<HsaApiTable*>(ops_properties->table);
      roctracer::hsa_support::async_copy_callback_fun = ops_properties->async_copy_callback_fun;
      roctracer::hsa_support::async_copy_callback_arg = ops_properties->async_copy_callback_arg;
      roctracer::hsa_support::output_prefix = ops_properties->output_prefix;

#if 0
      // HSA dispatches intercepting
      rocprofiler::SaveHsaApi(table);
      rocprofiler::ProxyQueue::InitFactory();
      rocprofiler::ProxyQueue::HsaIntercept(table);
      rocprofiler::InterceptQueue::HsaIntercept(table);
#endif

      // HSA async-copy tracing
      hsa_status_t status = hsa_amd_profiling_async_copy_enable(true);
      if (status != HSA_STATUS_SUCCESS) EXC_ABORT(status, "hsa_amd_profiling_async_copy_enable");
      roctracer::hsa_amd_memory_async_copy_fn = table->amd_ext_->hsa_amd_memory_async_copy_fn;
      roctracer::hsa_amd_memory_async_copy_rect_fn = table->amd_ext_->hsa_amd_memory_async_copy_rect_fn;
      table->amd_ext_->hsa_amd_memory_async_copy_fn = roctracer::hsa_amd_memory_async_copy_interceptor;
      table->amd_ext_->hsa_amd_memory_async_copy_rect_fn = roctracer::hsa_amd_memory_async_copy_rect_interceptor;

      break;
    }
    case ACTIVITY_DOMAIN_KFD_API: {
      roctracer::kfd_support::intercept_KFDApiTable();
      break;
    }
    case ACTIVITY_DOMAIN_HSA_API: {
      // HSA API properties
      HsaApiTable* table = reinterpret_cast<HsaApiTable*>(properties);
      roctracer::hsa_support::intercept_CoreApiTable(table->core_);
      roctracer::hsa_support::intercept_AmdExtTable(table->amd_ext_);
      roctracer::hsa_support::intercept_ImageExtTable(table->image_ext_);
      break;
    }
    case ACTIVITY_DOMAIN_HCC_OPS:
    case ACTIVITY_DOMAIN_HIP_API: {
      mark_api_callback_ptr = reinterpret_cast<mark_api_callback_t*>(properties);
      if (roctracer::hip_act_cb_tracker == NULL) roctracer::hip_act_cb_tracker = new roctracer::hip_act_cb_tracker_t;
      break;
    }
    case ACTIVITY_DOMAIN_EXT_API: {
      roctracer_ext_properties_t* ops_properties = reinterpret_cast<roctracer_ext_properties_t*>(properties);
      roctracer::ext_support::roctracer_start_cb = ops_properties->start_cb;
      roctracer::ext_support::roctracer_stop_cb = ops_properties->stop_cb;
      break;
    }
    default:
      EXC_RAISING(ROCTRACER_STATUS_BAD_DOMAIN, "invalid domain ID(" << domain << ")");
  }
  API_METHOD_SUFFIX
}

static bool is_loaded = false;

PUBLIC_API bool roctracer_load() {
  ONLOAD_TRACE("begin, loaded(" << is_loaded << ")");

  if (is_loaded == true) return true;
  is_loaded = true;

  if (roctracer::cb_journal == NULL) roctracer::cb_journal = new roctracer::CbJournal;
  if (roctracer::act_journal == NULL) roctracer::act_journal = new roctracer::ActJournal;

  ONLOAD_TRACE_END();
  return true;
}

PUBLIC_API void roctracer_unload() {
  ONLOAD_TRACE("begin, loaded(" << is_loaded << ")");

  if (is_loaded == false) return;
  is_loaded = false;

  if (roctracer::cb_journal != NULL) {
    delete roctracer::cb_journal;
    roctracer::cb_journal = NULL;
  }
  if (roctracer::act_journal != NULL) {
    delete roctracer::act_journal;
    roctracer::act_journal = NULL;
  }

  roctracer::trace_buffer.Flush();
  roctracer::close_output_file(roctracer::kernel_file_handle);
  ONLOAD_TRACE_END();
}

CONSTRUCTOR_API void constructor() {
  ONLOAD_TRACE_BEG();
  roctracer::util::Logger::Create();
  roctracer_load();
  ONLOAD_TRACE_END();
}

DESTRUCTOR_API void destructor() {
  ONLOAD_TRACE_BEG();
  roctracer_unload();
  util::HsaRsrcFactory::Destroy();
  roctracer::util::Logger::Destroy();
  ONLOAD_TRACE_END();
}

}  // extern "C"
