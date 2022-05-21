/* Copyright (c) 2018-2022 Advanced Micro Devices, Inc.

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

#include "roctracer.h"
#include "roctracer_hip.h"
#include "roctracer_ext.h"
#include "roctracer_roctx.h"
#include "roctracer_hsa.h"

#include <assert.h>
#include <dirent.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <atomic>
#include <mutex>
#include <stack>
#include <unordered_map>
#include <vector>

#include "journal.h"
#include "loader.h"
#include "memory_pool.h"
#include "tracker.h"
#include "util/exception.h"
#include "util/logger.h"

#define PUBLIC_API __attribute__((visibility("default")))
#define CONSTRUCTOR_API __attribute__((constructor))
#define DESTRUCTOR_API __attribute__((destructor))

#define CHECK_HSA_STATUS(msg, status)                                                              \
  do {                                                                                             \
    if ((status) != HSA_STATUS_SUCCESS) {                                                          \
      const char* status_string = nullptr;                                                         \
      hsa_status_string(status, &status_string);                                                   \
      FATAL_LOGGING(msg << ": " << (status_string ? status_string : "<unknown error>"));           \
    }                                                                                              \
  } while (false)

#define HIPAPI_CALL(call)                                                                          \
  do {                                                                                             \
    hipError_t err = call;                                                                         \
    if (err != hipSuccess) {                                                                       \
      FATAL_LOGGING("HIP error: " #call " error(" << err << ")");                                  \
    }                                                                                              \
  } while (false)

#define API_METHOD_PREFIX                                                                          \
  roctracer_status_t err = ROCTRACER_STATUS_SUCCESS;                                               \
  try {
#define API_METHOD_SUFFIX                                                                          \
  }                                                                                                \
  catch (std::exception & e) {                                                                     \
    ERR_LOGGING(__FUNCTION__ << "(), " << e.what());                                               \
    err = GetExcStatus(e);                                                                         \
  }                                                                                                \
  return err;

#define API_METHOD_CATCH(X)                                                                        \
  }                                                                                                \
  catch (std::exception & e) {                                                                     \
    ERR_LOGGING(__FUNCTION__ << "(), " << e.what());                                               \
  }                                                                                                \
  (void)err;                                                                                       \
  return X;

#define ONLOAD_TRACE(str)                                                                          \
  if (getenv("ROCP_ONLOAD_TRACE")) do {                                                            \
      std::cout << "PID(" << GetPid() << "): TRACER_LIB::" << __FUNCTION__ << " " << str           \
                << std::endl                                                                       \
                << std::flush;                                                                     \
    } while (false);
#define ONLOAD_TRACE_BEG() ONLOAD_TRACE("begin")
#define ONLOAD_TRACE_END() ONLOAD_TRACE("end")

static inline uint32_t GetPid() { return syscall(__NR_getpid); }

///////////////////////////////////////////////////////////////////////////////////////////////////
// Mark callback
//
typedef void(mark_api_callback_t)(uint32_t domain, uint32_t cid, const void* callback_data,
                                  void* arg);
mark_api_callback_t* mark_api_callback_ptr = nullptr;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Internal library methods
//
namespace roctracer {

namespace hsa_support {
decltype(hsa_system_get_info)* hsa_system_get_info_fn = hsa_system_get_info;
decltype(hsa_amd_memory_async_copy)* hsa_amd_memory_async_copy_fn = hsa_amd_memory_async_copy;
decltype(hsa_amd_memory_async_copy_rect)* hsa_amd_memory_async_copy_rect_fn =
    hsa_amd_memory_async_copy_rect;

::HsaApiTable* kHsaApiTable;

void SaveHsaApi(::HsaApiTable* table) {
  kHsaApiTable = table;
  hsa_system_get_info_fn = table->core_->hsa_system_get_info_fn;
  hsa_amd_memory_async_copy_fn = table->amd_ext_->hsa_amd_memory_async_copy_fn;
  hsa_amd_memory_async_copy_rect_fn = table->amd_ext_->hsa_amd_memory_async_copy_rect_fn;
}

void RestoreHsaApi() {
  ::HsaApiTable* table = kHsaApiTable;
  table->core_->hsa_system_get_info_fn = hsa_system_get_info_fn;
  table->amd_ext_->hsa_amd_memory_async_copy_fn = hsa_amd_memory_async_copy_fn;
  table->amd_ext_->hsa_amd_memory_async_copy_rect_fn = hsa_amd_memory_async_copy_rect_fn;
}

// callbacks table
cb_table_t cb_table;
// async copy activity callback
bool async_copy_callback_enabled = false;
MemoryPool* async_copy_callback_memory_pool = nullptr;
// Table of function pointers to HSA Core Runtime
CoreApiTable CoreApiTable_saved{};
// Table of function pointers to AMD extensions
AmdExtTable AmdExtTable_saved{};
// Table of function pointers to HSA Image Extension
ImageExtTable ImageExtTable_saved{};

}  // namespace hsa_support

namespace ext_support {
roctracer_start_cb_t roctracer_start_cb = nullptr;
roctracer_stop_cb_t roctracer_stop_cb = nullptr;
}  // namespace ext_support

namespace util {

uint64_t timestamp_ns() {
  uint64_t sysclock;

  hsa_status_t status = hsa_support::hsa_system_get_info_fn(HSA_SYSTEM_INFO_TIMESTAMP, &sysclock);
  if (status == HSA_STATUS_ERROR_NOT_INITIALIZED) return 0;
  CHECK_HSA_STATUS("hsa_system_get_info()", status);

  static uint64_t sysclock_period = []() {
    uint64_t sysclock_hz = 0;
    hsa_status_t status =
        hsa_support::hsa_system_get_info_fn(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &sysclock_hz);
    CHECK_HSA_STATUS("hsa_system_get_info()", status);
    return (uint64_t)1000000000 / sysclock_hz;
  }();

  return sysclock * sysclock_period;
}

}  // namespace util

struct CallbackJournalData {
  roctracer_rtapi_callback_t callback;
  void* user_data;
};
static Journal<CallbackJournalData> cb_journal;

struct ActivityJournalData {
  roctracer_pool_t* pool;
};
static Journal<ActivityJournalData> act_journal;

roctracer_status_t GetExcStatus(const std::exception& e) {
  const util::exception<roctracer_status_t>* roctracer_exc_ptr =
      dynamic_cast<const util::exception<roctracer_status_t>*>(&e);
  return (roctracer_exc_ptr) ? roctracer_exc_ptr->status() : ROCTRACER_STATUS_ERROR;
}

static auto NextCorrelationId() {
  static std::atomic<uint64_t> counter{1};
  return counter.fetch_add(1, std::memory_order_relaxed);
}

// Records storage
struct RecordDataPair {
  roctracer_record_t record;
  union {
    hip_api_data_t data;
  };
  RecordDataPair() {}
};
static thread_local std::stack<RecordDataPair> record_data_pair_stack;

// Correlation id storage
static thread_local activity_correlation_id_t correlation_id_tls = 0;
static std::map<activity_correlation_id_t, activity_correlation_id_t> correlation_id_map{};
std::mutex correlation_id_mutex;

static thread_local std::stack<activity_correlation_id_t> external_id_stack;

static inline void CorrelationIdRegister(activity_correlation_id_t correlation_id) {
  std::lock_guard lock(correlation_id_mutex);
  [[maybe_unused]] const auto ret = correlation_id_map.insert({correlation_id, correlation_id_tls});
  assert(ret.second && "HIP activity id is not unique");

  DEBUG_TRACE("CorrelationIdRegister id(%lu) id_tls(%lu)\n", correlation_id, correlation_id_tls);
}

static inline activity_correlation_id_t CorrelationIdLookup(
    activity_correlation_id_t correlation_id) {
  std::lock_guard lock(correlation_id_mutex);
  auto it = correlation_id_map.find(correlation_id);
  assert(it != correlation_id_map.end() && "HIP activity id lookup failed");
  const activity_correlation_id_t ret_val = it->second;
  correlation_id_map.erase(it);

  DEBUG_TRACE("CorrelationIdLookup id(%lu) ret(%lu)\n", correlation_id, ret_val);

  return ret_val;
}

std::mutex hip_activity_mutex;

enum { API_CB_MASK = 0x1, ACT_CB_MASK = 0x2 };

class HIPActivityCallbackTracker {
 public:
  uint32_t enable_check(uint32_t op, uint32_t mask) { return data_[op] |= mask; }
  uint32_t disable_check(uint32_t op, uint32_t mask) { return data_[op] &= ~mask; }

 private:
  std::unordered_map<uint32_t, uint32_t> data_;
};

static HIPActivityCallbackTracker hip_act_cb_tracker;

inline uint32_t HipApiActivityEnableCheck(uint32_t op) {
  const uint32_t mask = hip_act_cb_tracker.enable_check(op, API_CB_MASK);
  const uint32_t ret = (mask & ACT_CB_MASK);
  return ret;
}

inline uint32_t HipApiActivityDisableCheck(uint32_t op) {
  const uint32_t mask = hip_act_cb_tracker.disable_check(op, API_CB_MASK);
  const uint32_t ret = (mask & ACT_CB_MASK);
  return ret;
}

inline uint32_t HipActActivityEnableCheck(uint32_t op) {
  hip_act_cb_tracker.enable_check(op, ACT_CB_MASK);
  return 0;
}

inline uint32_t HipActActivityDisableCheck(uint32_t op) {
  const uint32_t mask = hip_act_cb_tracker.disable_check(op, ACT_CB_MASK);
  const uint32_t ret = (mask & API_CB_MASK);
  return ret;
}

void* HIP_SyncApiDataCallback(uint32_t op_id, roctracer_record_t* record, const void* callback_data,
                              void* arg) {
  void* ret = nullptr;
  const hip_api_data_t* data = reinterpret_cast<const hip_api_data_t*>(callback_data);
  hip_api_data_t* data_ptr = const_cast<hip_api_data_t*>(data);
  MemoryPool* pool = reinterpret_cast<MemoryPool*>(arg);

  int phase = ACTIVITY_API_PHASE_ENTER;
  if (record != nullptr) {
    assert(data != nullptr && "ActivityCallback: data is NULL");
    phase = data->phase;
  } else if (pool != nullptr) {
    phase = ACTIVITY_API_PHASE_EXIT;
  }

  if (phase == ACTIVITY_API_PHASE_ENTER) {
    // Allocating a record if nullptr passed
    if (record == nullptr) {
      assert(data == nullptr && "ActivityCallback enter: record is NULL");
      data = &record_data_pair_stack.emplace().data;
      data_ptr = const_cast<hip_api_data_t*>(data);
      data_ptr->phase = phase;
      data_ptr->correlation_id = 0;
    }

    // Correlation ID generating
    uint64_t correlation_id = data->correlation_id;
    if (correlation_id == 0) {
      correlation_id = NextCorrelationId();
      data_ptr->correlation_id = correlation_id;
    }

    // Passing correlation ID
    correlation_id_tls = correlation_id;

    ret = data_ptr;
  } else {
    // popping the record entry
    assert(!record_data_pair_stack.empty() &&
           "HIP_SyncApiDataCallback exit: record stack is empty");
    record_data_pair_stack.pop();

    // Clearing correlation ID
    correlation_id_tls = 0;
  }

  DEBUG_TRACE(
      "HIP_SyncApiDataCallback(\"%s\") phase(%d): op(%u) record(%p) data(%p) pool(%p) depth(%d) "
      "correlation_id(%lu) time_ns(%lu)\n",
      roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, op_id, 0), phase, op_id, record, data, pool,
      (int)(record_data_pair_stack.size()), (data_ptr) ? data_ptr->correlation_id : 0,
      util::timestamp_ns());

  return ret;
}

void* HIP_SyncActivityCallback(uint32_t op_id, roctracer_record_t* record,
                               const void* callback_data, void* arg) {
  const uint64_t timestamp_ns = util::timestamp_ns();
  void* ret = nullptr;
  const hip_api_data_t* data = reinterpret_cast<const hip_api_data_t*>(callback_data);
  hip_api_data_t* data_ptr = const_cast<hip_api_data_t*>(data);
  MemoryPool* pool = reinterpret_cast<MemoryPool*>(arg);

  int phase = ACTIVITY_API_PHASE_ENTER;
  if (record != nullptr) {
    assert(data != nullptr && "ActivityCallback: data is NULL");
    phase = data->phase;
  } else if (pool != nullptr) {
    phase = ACTIVITY_API_PHASE_EXIT;
  }

  if (phase == ACTIVITY_API_PHASE_ENTER) {
    // Allocating a record if nullptr passed
    if (record == nullptr) {
      assert(data == nullptr && "ActivityCallback enter: record is NULL");
      auto& top = record_data_pair_stack.emplace();
      record = &(top.record);
      data = &(top.data);
      data_ptr = const_cast<hip_api_data_t*>(data);
      data_ptr->phase = phase;
      data_ptr->correlation_id = 0;
    }

    // Filing record info
    record->domain = ACTIVITY_DOMAIN_HIP_API;
    record->op = op_id;
    record->begin_ns = timestamp_ns;

    // Correlation ID generating
    uint64_t correlation_id = data->correlation_id;
    if (correlation_id == 0) {
      correlation_id = NextCorrelationId();
      data_ptr->correlation_id = correlation_id;
    }
    record->correlation_id = correlation_id;

    // Passing correlation ID
    correlation_id_tls = correlation_id;

    ret = data_ptr;
  } else {
    assert(pool != nullptr && "ActivityCallback exit: pool is NULL");
    assert(!record_data_pair_stack.empty() && "ActivityCallback exit: record stack is empty");

    // Getting record of stacked
    if (record == nullptr) record = &record_data_pair_stack.top().record;

    // Filing record info
    record->end_ns = timestamp_ns;
    record->process_id = syscall(__NR_getpid);
    record->thread_id = syscall(__NR_gettid);

    if (!external_id_stack.empty()) {
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
    record_data_pair_stack.pop();

    // Clearing correlation ID
    correlation_id_tls = 0;
  }

  DEBUG_TRACE(
      "HIP_SyncActivityCallback(\"%s\") phase(%d): op(%u) record(%p) data(%p) pool(%p) depth(%d) "
      "correlation_id(%lu) beg_ns(%lu) end_ns(%lu)\n",
      roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, op_id, 0), phase, op_id, record, data, pool,
      (int)(record_data_pair_stack.size()), (data_ptr) ? data_ptr->correlation_id : 0,
      timestamp_ns);

  return ret;
}

void HIP_ActivityIdCallback(activity_correlation_id_t correlation_id) {
  CorrelationIdRegister(correlation_id);
}

void HIP_AsyncActivityCallback(uint32_t op_id, void* record, void* arg) {
  MemoryPool* pool = reinterpret_cast<MemoryPool*>(arg);
  roctracer_record_t* record_ptr = reinterpret_cast<roctracer_record_t*>(record);
  record_ptr->domain = ACTIVITY_DOMAIN_HIP_OPS;
  record_ptr->correlation_id = CorrelationIdLookup(record_ptr->correlation_id);
  if (record_ptr->correlation_id == 0) return;
  pool->Write(*record_ptr);

  DEBUG_TRACE(
      "HIP_AsyncActivityCallback(\"%s\"): op(%u) kind(%u) record(%p) pool(%p) correlation_id(%d) "
      "beg_ns(%lu) end_ns(%lu)\n",
      roctracer_op_string(ACTIVITY_DOMAIN_HIP_OPS, record_ptr->op, record_ptr->kind),
      record_ptr->op, record_ptr->kind, record, pool, record_ptr->correlation_id,
      record_ptr->begin_ns, record_ptr->end_ns);
}

namespace hsa_support {

void hsa_async_copy_handler(const Tracker::entry_t* entry) {
  activity_record_t record{};
  record.domain = ACTIVITY_DOMAIN_HSA_OPS;
  record.op = HSA_OP_ID_COPY;
  record.begin_ns = entry->begin;
  record.end_ns = entry->end;
  record.device_id = 0;
  entry->pool->Write(record);
}

hsa_status_t hsa_amd_memory_async_copy_interceptor(void* dst, hsa_agent_t dst_agent,
                                                   const void* src, hsa_agent_t src_agent,
                                                   size_t size, uint32_t num_dep_signals,
                                                   const hsa_signal_t* dep_signals,
                                                   hsa_signal_t completion_signal) {
  hsa_status_t status = HSA_STATUS_SUCCESS;
  if (hsa_support::async_copy_callback_enabled) {
    Tracker::entry_t* entry = new Tracker::entry_t();
    entry->handler = hsa_async_copy_handler;
    entry->pool = hsa_support::async_copy_callback_memory_pool;
    Tracker::Enable(Tracker::COPY_ENTRY_TYPE, hsa_agent_t{}, completion_signal, entry);
    status = hsa_amd_memory_async_copy_fn(dst, dst_agent, src, src_agent, size, num_dep_signals,
                                          dep_signals, entry->signal);
    if (status != HSA_STATUS_SUCCESS) Tracker::Disable(entry);
  } else {
    status = hsa_amd_memory_async_copy_fn(dst, dst_agent, src, src_agent, size, num_dep_signals,
                                          dep_signals, completion_signal);
  }
  return status;
}

hsa_status_t hsa_amd_memory_async_copy_rect_interceptor(
    const hsa_pitched_ptr_t* dst, const hsa_dim3_t* dst_offset, const hsa_pitched_ptr_t* src,
    const hsa_dim3_t* src_offset, const hsa_dim3_t* range, hsa_agent_t copy_agent,
    hsa_amd_copy_direction_t dir, uint32_t num_dep_signals, const hsa_signal_t* dep_signals,
    hsa_signal_t completion_signal) {
  hsa_status_t status = HSA_STATUS_SUCCESS;
  if (hsa_support::async_copy_callback_enabled) {
    Tracker::entry_t* entry = new Tracker::entry_t();
    entry->handler = hsa_async_copy_handler;
    entry->pool = hsa_support::async_copy_callback_memory_pool;
    Tracker::Enable(Tracker::COPY_ENTRY_TYPE, hsa_agent_t{}, completion_signal, entry);
    status = hsa_amd_memory_async_copy_rect_fn(dst, dst_offset, src, src_offset, range, copy_agent,
                                               dir, num_dep_signals, dep_signals, entry->signal);
    if (status != HSA_STATUS_SUCCESS) Tracker::Disable(entry);
  } else {
    status =
        hsa_amd_memory_async_copy_rect_fn(dst, dst_offset, src, src_offset, range, copy_agent, dir,
                                          num_dep_signals, dep_signals, completion_signal);
  }
  return status;
}

}  // namespace hsa_support

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
MemoryPool* default_memory_pool = nullptr;
std::recursive_mutex memory_pool_mutex;

// Stop status routines and primitives
unsigned stop_status_value = 0;
std::mutex stop_status_mutex;
unsigned set_stopped(unsigned val) {
  std::lock_guard lock(stop_status_mutex);
  const unsigned ret = (stop_status_value ^ val);
  stop_status_value = val;
  return ret;
}
}  // namespace roctracer

using namespace roctracer;

LOADER_INSTANTIATE();

///////////////////////////////////////////////////////////////////////////////////////////////////
// Public library methods
//

// Returns library version
PUBLIC_API uint32_t roctracer_version_major() { return ROCTRACER_VERSION_MAJOR; }
PUBLIC_API uint32_t roctracer_version_minor() { return ROCTRACER_VERSION_MINOR; }

// Returns the last error
PUBLIC_API const char* roctracer_error_string() {
  return strdup(util::Logger::LastMessage().c_str());
}

// Return Op string by given domain and activity/API codes
// nullptr returned on the error and the library errno is set
PUBLIC_API const char* roctracer_op_string(uint32_t domain, uint32_t op, uint32_t kind) {
  API_METHOD_PREFIX
  switch (domain) {
    case ACTIVITY_DOMAIN_HSA_API:
      return hsa_support::GetApiName(op);
    case ACTIVITY_DOMAIN_HSA_EVT:
      return RocpLoader::Instance().GetEvtName(op);
    case ACTIVITY_DOMAIN_HSA_OPS:
      return RocpLoader::Instance().GetOpName(op);
    case ACTIVITY_DOMAIN_HIP_OPS:
      return HipLoader::Instance().GetOpName(kind);
    case ACTIVITY_DOMAIN_HIP_API:
      return HipLoader::Instance().ApiName(op);
    case ACTIVITY_DOMAIN_EXT_API:
      return "EXT_API";
    default:
      EXC_RAISING(ROCTRACER_STATUS_ERROR_INVALID_DOMAIN_ID, "invalid domain ID(" << domain << ")");
  }
  API_METHOD_CATCH(nullptr)
}

// Return Op code and kind by given string
PUBLIC_API roctracer_status_t roctracer_op_code(uint32_t domain, const char* str, uint32_t* op,
                                                uint32_t* kind) {
  API_METHOD_PREFIX
  switch (domain) {
    case ACTIVITY_DOMAIN_HSA_API: {
      *op = hsa_support::GetApiCode(str);
      if (*op == HSA_API_ID_NUMBER) {
        EXC_RAISING(ROCTRACER_STATUS_ERROR_INVALID_ARGUMENT,
                    "Invalid API name \"" << str << "\", domain ID(" << domain << ")");
      }
      if (kind != nullptr) *kind = 0;
      break;
    }
    case ACTIVITY_DOMAIN_HIP_API: {
      *op = hipApiIdByName(str);
      if (*op == HIP_API_ID_NONE) {
        EXC_RAISING(ROCTRACER_STATUS_ERROR_INVALID_ARGUMENT,
                    "Invalid API name \"" << str << "\", domain ID(" << domain << ")");
      }
      if (kind != nullptr) *kind = 0;
      break;
    }
    default:
      EXC_RAISING(ROCTRACER_STATUS_ERROR_INVALID_DOMAIN_ID, "limited domain ID(" << domain << ")");
  }
  API_METHOD_SUFFIX
}

static inline uint32_t get_op_begin(uint32_t domain) {
  switch (domain) {
    case ACTIVITY_DOMAIN_HSA_OPS:
      return 0;
    case ACTIVITY_DOMAIN_HSA_API:
      return 0;
    case ACTIVITY_DOMAIN_HSA_EVT:
      return 0;
    case ACTIVITY_DOMAIN_HIP_OPS:
      return 0;
    case ACTIVITY_DOMAIN_HIP_API:
      return HIP_API_ID_FIRST;
    case ACTIVITY_DOMAIN_EXT_API:
      return 0;
    case ACTIVITY_DOMAIN_ROCTX:
      return 0;
    default:
      EXC_RAISING(ROCTRACER_STATUS_ERROR_INVALID_DOMAIN_ID, "invalid domain ID(" << domain << ")");
  }
  return 0;
}

static inline uint32_t get_op_end(uint32_t domain) {
  switch (domain) {
    case ACTIVITY_DOMAIN_HSA_OPS:
      return HSA_OP_ID_NUMBER;
    case ACTIVITY_DOMAIN_HSA_API:
      return HSA_API_ID_NUMBER;
    case ACTIVITY_DOMAIN_HSA_EVT:
      return HSA_EVT_ID_NUMBER;
    case ACTIVITY_DOMAIN_HIP_OPS:
      return HIP_OP_ID_NUMBER;
    case ACTIVITY_DOMAIN_HIP_API:
      return HIP_API_ID_LAST + 1;
    case ACTIVITY_DOMAIN_EXT_API:
      return 0;
    case ACTIVITY_DOMAIN_ROCTX:
      return ROCTX_API_ID_NUMBER;
    default:
      EXC_RAISING(ROCTRACER_STATUS_ERROR_INVALID_DOMAIN_ID, "invalid domain ID(" << domain << ")");
  }
  return 0;
}

// Enable runtime API callbacks
static void roctracer_enable_callback_fun(roctracer_domain_t domain, uint32_t op,
                                          roctracer_rtapi_callback_t callback, void* user_data) {
  switch (domain) {
    case ACTIVITY_DOMAIN_HSA_OPS:
      break;
    case ACTIVITY_DOMAIN_HSA_API: {
#if 0
      if (op == HSA_API_ID_DISPATCH) {
        if (!RocpLoader::Instance().RegisterApiCallback(op, (void*)callback, user_data))
          FATAL_LOGGING("HSA::RegisterApiCallback error(" << op << ") failed");
        break;
      }
#endif
      if (op >= HSA_API_ID_NUMBER)
        EXC_RAISING(ROCTRACER_STATUS_ERROR_INVALID_ARGUMENT,
                    "invalid HSA API operation ID(" << op << ")");

      hsa_support::cb_table.Set(op, callback, user_data);
      break;
    }
    case ACTIVITY_DOMAIN_HSA_EVT: {
      if (!RocpLoader::Instance().RegisterEvtCallback(op, (void*)callback, user_data))
        FATAL_LOGGING("HSA::RegisterEvtCallback error(" << op << ") failed");
      break;
    }
    case ACTIVITY_DOMAIN_HIP_OPS:
      break;
    case ACTIVITY_DOMAIN_HIP_API: {
      if (!HipLoader::Instance().Enabled()) break;
      std::lock_guard lock(hip_activity_mutex);

      hipError_t hip_err =
          HipLoader::Instance().RegisterApiCallback(op, (void*)callback, user_data);
      if (hip_err != hipSuccess)
        FATAL_LOGGING("HIP::RegisterApiCallback(" << op << ") error(" << hip_err << ")");

      if (HipApiActivityEnableCheck(op) == 0) {
        hip_err = HipLoader::Instance().RegisterActivityCallback(op, (void*)HIP_SyncApiDataCallback,
                                                                 (void*)1);
        if (hip_err != hipSuccess)
          FATAL_LOGGING("HIPAPI: HIP::RegisterActivityCallback(" << op << ") error(" << hip_err
                                                                 << ")");
      }
      break;
    }
    case ACTIVITY_DOMAIN_ROCTX: {
      if (RocTxLoader::Instance().Enabled() &&
          !RocTxLoader::Instance().RegisterApiCallback(op, (void*)callback, user_data))
        FATAL_LOGGING("ROCTX::RegisterApiCallback(" << op << ") failed");
      break;
    }
    default:
      EXC_RAISING(ROCTRACER_STATUS_ERROR_INVALID_DOMAIN_ID, "invalid domain ID(" << domain << ")");
  }
}

static void roctracer_enable_callback_impl(roctracer_domain_t domain, uint32_t op,
                                           roctracer_rtapi_callback_t callback, void* user_data) {
  cb_journal.Insert(domain, op, {callback, user_data});
  roctracer_enable_callback_fun(domain, op, callback, user_data);
}

PUBLIC_API roctracer_status_t roctracer_enable_op_callback(roctracer_domain_t domain, uint32_t op,
                                                           roctracer_rtapi_callback_t callback,
                                                           void* user_data) {
  API_METHOD_PREFIX
  roctracer_enable_callback_impl(domain, op, callback, user_data);
  API_METHOD_SUFFIX
}

PUBLIC_API roctracer_status_t roctracer_enable_domain_callback(roctracer_domain_t domain,
                                                               roctracer_rtapi_callback_t callback,
                                                               void* user_data) {
  API_METHOD_PREFIX
  const uint32_t op_end = get_op_end(domain);
  for (uint32_t op = get_op_begin(domain); op < op_end; ++op)
    roctracer_enable_callback_impl(domain, op, callback, user_data);
  API_METHOD_SUFFIX
}

PUBLIC_API roctracer_status_t roctracer_enable_callback(roctracer_rtapi_callback_t callback,
                                                        void* user_data) {
  API_METHOD_PREFIX
  for (uint32_t domain = 0; domain < ACTIVITY_DOMAIN_NUMBER; ++domain) {
    const uint32_t op_end = get_op_end(domain);
    for (uint32_t op = get_op_begin(domain); op < op_end; ++op)
      roctracer_enable_callback_impl((roctracer_domain_t)domain, op, callback, user_data);
  }
  API_METHOD_SUFFIX
}

// Disable runtime API callbacks
static void roctracer_disable_callback_fun(roctracer_domain_t domain, uint32_t op) {
  switch (domain) {
    case ACTIVITY_DOMAIN_HSA_OPS:
      break;
    case ACTIVITY_DOMAIN_HSA_API: {
#if 0
      if (op == HSA_API_ID_DISPATCH && !RocpLoader::Instance().RemoveApiCallback(op))
        FATAL_LOGGING("HSA::RemoveActivityCallback error(" << op << ") failed");
        break;
#endif
      if (op >= HSA_API_ID_NUMBER)
        EXC_RAISING(ROCTRACER_STATUS_ERROR_INVALID_ARGUMENT,
                    "invalid HSA API operation ID(" << op << ")");
      hsa_support::cb_table.Set(op, nullptr, nullptr);
      break;
    }
    case ACTIVITY_DOMAIN_HIP_OPS:
      break;
    case ACTIVITY_DOMAIN_HIP_API: {
      if (!HipLoader::Instance().Enabled()) break;
      std::lock_guard lock(hip_activity_mutex);

      const hipError_t hip_err = HipLoader::Instance().RemoveApiCallback(op);
      if (hip_err != hipSuccess)
        FATAL_LOGGING("HIP::RemoveApiCallback(" << op << "), error(" << hip_err << ")");

      if (HipApiActivityDisableCheck(op) == 0) {
        const hipError_t hip_err = HipLoader::Instance().RemoveActivityCallback(op);
        if (hip_err != hipSuccess)
          FATAL_LOGGING("HIPAPI: HIP::RemoveActivityCallback op(" << op << "), error(" << hip_err
                                                                  << ")");
      }
      break;
    }
    case ACTIVITY_DOMAIN_HSA_EVT: {
      if (!RocpLoader::Instance().RemoveEvtCallback(op))
        FATAL_LOGGING("HSA::RemoveEvtCallback error(" << op << ") failed");
      break;
    }
    case ACTIVITY_DOMAIN_ROCTX: {
      if (RocTxLoader::Instance().Enabled() && !RocTxLoader::Instance().RemoveApiCallback(op))
        FATAL_LOGGING("ROCTX::RemoveApiCallback(" << op << ") failed");
      break;
    }
    default:
      EXC_RAISING(ROCTRACER_STATUS_ERROR_INVALID_DOMAIN_ID, "invalid domain ID(" << domain << ")");
  }
}

static void roctracer_disable_callback_impl(roctracer_domain_t domain, uint32_t op) {
  cb_journal.Remove(domain, op);
  roctracer_disable_callback_fun(domain, op);
}

PUBLIC_API roctracer_status_t roctracer_disable_op_callback(roctracer_domain_t domain,
                                                            uint32_t op) {
  API_METHOD_PREFIX
  roctracer_disable_callback_impl(domain, op);
  API_METHOD_SUFFIX
}

PUBLIC_API roctracer_status_t roctracer_disable_domain_callback(roctracer_domain_t domain) {
  API_METHOD_PREFIX
  const uint32_t op_end = get_op_end(domain);
  for (uint32_t op = get_op_begin(domain); op < op_end; ++op)
    roctracer_disable_callback_impl(domain, op);
  API_METHOD_SUFFIX
}

PUBLIC_API roctracer_status_t roctracer_disable_callback() {
  API_METHOD_PREFIX
  for (uint32_t domain = 0; domain < ACTIVITY_DOMAIN_NUMBER; ++domain) {
    const uint32_t op_end = get_op_end(domain);
    for (uint32_t op = get_op_begin(domain); op < op_end; ++op)
      roctracer_disable_callback_impl((roctracer_domain_t)domain, op);
  }
  API_METHOD_SUFFIX
}

// Return default pool and set new one if parameter pool is not NULL.
PUBLIC_API roctracer_pool_t* roctracer_default_pool_expl(roctracer_pool_t* pool) {
  std::lock_guard lock(memory_pool_mutex);
  roctracer_pool_t* p = reinterpret_cast<roctracer_pool_t*>(default_memory_pool);
  if (pool != nullptr) default_memory_pool = reinterpret_cast<MemoryPool*>(pool);
  return p;
}

PUBLIC_API roctracer_pool_t* roctracer_default_pool() {
  std::lock_guard lock(memory_pool_mutex);
  return reinterpret_cast<roctracer_pool_t*>(default_memory_pool);
}

// Open memory pool
static void roctracer_open_pool_impl(const roctracer_properties_t* properties,
                                     roctracer_pool_t** pool) {
  std::lock_guard lock(memory_pool_mutex);
  if ((pool == nullptr) && (default_memory_pool != nullptr)) {
    EXC_RAISING(ROCTRACER_STATUS_ERROR_DEFAULT_POOL_ALREADY_DEFINED, "default pool already set");
  }
  MemoryPool* p = new MemoryPool(*properties);
  if (p == nullptr) EXC_RAISING(ROCTRACER_STATUS_ERROR_MEMORY_ALLOCATION, "MemoryPool() error");
  if (pool != nullptr)
    *pool = p;
  else
    default_memory_pool = p;
}

PUBLIC_API roctracer_status_t roctracer_open_pool_expl(const roctracer_properties_t* properties,
                                                       roctracer_pool_t** pool) {
  API_METHOD_PREFIX
  roctracer_open_pool_impl(properties, pool);
  API_METHOD_SUFFIX
}

PUBLIC_API roctracer_status_t roctracer_open_pool(const roctracer_properties_t* properties) {
  API_METHOD_PREFIX
  roctracer_open_pool_impl(properties, nullptr);
  API_METHOD_SUFFIX
}

PUBLIC_API roctracer_status_t roctracer_next_record(const activity_record_t* record,
                                                    const activity_record_t** next) {
  API_METHOD_PREFIX
  *next = record + 1;
  API_METHOD_SUFFIX
}

// Enable activity records logging
static void roctracer_enable_activity_fun(roctracer_domain_t domain, uint32_t op,
                                          roctracer_pool_t* pool) {
  assert(pool != nullptr);
  switch (domain) {
    case ACTIVITY_DOMAIN_HSA_OPS: {
      if (op == HSA_OP_ID_COPY) {
        RocpLoader::Instance();
        hsa_support::async_copy_callback_enabled = true;
        hsa_support::async_copy_callback_memory_pool = reinterpret_cast<MemoryPool*>(pool);
      } else {
        const bool init_phase = (RocpLoader::GetRef() == nullptr);
        if (RocpLoader::GetRef() == nullptr) break;
        if (init_phase) {
          RocpLoader::Instance().InitActivityCallback((void*)HSA_AsyncActivityCallback,
                                                      (void*)pool);
        }
        if (!RocpLoader::Instance().EnableActivityCallback(op, true))
          FATAL_LOGGING("HSA::EnableActivityCallback error");
      }
      break;
    }
    case ACTIVITY_DOMAIN_HSA_API:
      break;
    case ACTIVITY_DOMAIN_HSA_EVT:
      RocpLoader::Instance();
      break;
    case ACTIVITY_DOMAIN_HIP_OPS: {
      if (!HipLoader::Instance().Enabled()) break;
      std::lock_guard lock(hip_activity_mutex);

      if (!HipLoader::Instance().InitActivityDone()) {
        HipLoader::Instance().InitActivityCallback((void*)HIP_ActivityIdCallback,
                                                   (void*)HIP_AsyncActivityCallback, (void*)pool);
        HipLoader::Instance().InitActivityDone() = true;
      }
      if (!HipLoader::Instance().EnableActivityCallback(op, true))
        FATAL_LOGGING("HIP::EnableActivityCallback error");
      break;
    }
    case ACTIVITY_DOMAIN_HIP_API: {
      if (!HipLoader::Instance().Enabled()) break;
      std::lock_guard lock(hip_activity_mutex);

      if (HipActActivityEnableCheck(op) == 0) {
        const hipError_t hip_err = HipLoader::Instance().RegisterActivityCallback(
            op, (void*)HIP_SyncActivityCallback, (void*)pool);
        if (hip_err != hipSuccess)
          FATAL_LOGGING("HIP::RegisterActivityCallback(" << op << " error(" << hip_err << ")");
      }
      break;
    }
    case ACTIVITY_DOMAIN_ROCTX:
      break;
    default:
      EXC_RAISING(ROCTRACER_STATUS_ERROR_INVALID_DOMAIN_ID, "invalid domain ID(" << domain << ")");
  }
}

static void roctracer_enable_activity_impl(roctracer_domain_t domain, uint32_t op,
                                           roctracer_pool_t* pool) {
  if (pool == nullptr) pool = default_memory_pool;
  if (pool == nullptr)
    EXC_RAISING(ROCTRACER_STATUS_ERROR_DEFAULT_POOL_UNDEFINED, "no default pool");
  act_journal.Insert(domain, op, {pool});
  roctracer_enable_activity_fun(domain, op, pool);
}

PUBLIC_API roctracer_status_t roctracer_enable_op_activity_expl(roctracer_domain_t domain,
                                                                uint32_t op,
                                                                roctracer_pool_t* pool) {
  API_METHOD_PREFIX
  roctracer_enable_activity_impl(domain, op, pool);
  API_METHOD_SUFFIX
}

PUBLIC_API roctracer_status_t roctracer_enable_op_activity(activity_domain_t domain, uint32_t op) {
  API_METHOD_PREFIX
  roctracer_enable_activity_impl(domain, op, nullptr);
  API_METHOD_SUFFIX
}

static void roctracer_enable_domain_activity_impl(roctracer_domain_t domain,
                                                  roctracer_pool_t* pool) {
  const uint32_t op_end = get_op_end(domain);
  for (uint32_t op = get_op_begin(domain); op < op_end; ++op)
    roctracer_enable_activity_impl(domain, op, pool);
}

PUBLIC_API roctracer_status_t roctracer_enable_domain_activity_expl(roctracer_domain_t domain,
                                                                    roctracer_pool_t* pool) {
  API_METHOD_PREFIX
  roctracer_enable_domain_activity_impl(domain, pool);
  API_METHOD_SUFFIX
}

PUBLIC_API roctracer_status_t roctracer_enable_domain_activity(activity_domain_t domain) {
  API_METHOD_PREFIX
  roctracer_enable_domain_activity_impl(domain, nullptr);
  API_METHOD_SUFFIX
}

static void roctracer_enable_activity_impl(roctracer_pool_t* pool) {
  for (uint32_t domain = 0; domain < ACTIVITY_DOMAIN_NUMBER; ++domain) {
    const uint32_t op_end = get_op_end(domain);
    for (uint32_t op = get_op_begin(domain); op < op_end; ++op)
      roctracer_enable_activity_impl((roctracer_domain_t)domain, op, pool);
  }
}

PUBLIC_API roctracer_status_t roctracer_enable_activity_expl(roctracer_pool_t* pool) {
  API_METHOD_PREFIX
  roctracer_enable_activity_impl(pool);
  API_METHOD_SUFFIX
}

PUBLIC_API roctracer_status_t roctracer_enable_activity() {
  API_METHOD_PREFIX
  roctracer_enable_activity_impl(nullptr);
  API_METHOD_SUFFIX
}

// Disable activity records logging
static void roctracer_disable_activity_fun(roctracer_domain_t domain, uint32_t op) {
  switch (domain) {
    case ACTIVITY_DOMAIN_HSA_OPS: {
      if (op == HSA_OP_ID_COPY) {
        hsa_support::async_copy_callback_enabled = false;
        hsa_support::async_copy_callback_memory_pool = nullptr;
      } else {
        if (RocpLoader::GetRef() == nullptr) break;
        if (!RocpLoader::Instance().EnableActivityCallback(op, false))
          FATAL_LOGGING("HSA::EnableActivityCallback(false) error, op(" << op << ")");
      }
      break;
    }
    case ACTIVITY_DOMAIN_HSA_API:
      break;
    case ACTIVITY_DOMAIN_HSA_EVT:
      break;
    case ACTIVITY_DOMAIN_HIP_OPS: {
      if (HipLoader::Instance().Enabled() &&
          !HipLoader::Instance().EnableActivityCallback(op, false))
        FATAL_LOGGING("HIP::EnableActivityCallback(nullptr) error, op(" << op << ")");
      break;
    }
    case ACTIVITY_DOMAIN_HIP_API: {
      if (!HipLoader::Instance().Enabled()) break;
      std::lock_guard lock(hip_activity_mutex);

      if (HipActActivityDisableCheck(op) == 0) {
        const hipError_t hip_err = HipLoader::Instance().RemoveActivityCallback(op);
        if (hip_err != hipSuccess)
          FATAL_LOGGING("HIP::RemoveActivityCallback op(" << op << "), error(" << hip_err << ")");
      } else {
        const hipError_t hip_err = HipLoader::Instance().RegisterActivityCallback(
            op, (void*)HIP_SyncApiDataCallback, (void*)1);
        if (hip_err != hipSuccess)
          FATAL_LOGGING("HIPACT: HIP::RegisterActivityCallback(" << op << ") error(" << hip_err
                                                                 << ")");
      }
      break;
    }
    case ACTIVITY_DOMAIN_ROCTX:
      break;
    default:
      EXC_RAISING(ROCTRACER_STATUS_ERROR_INVALID_DOMAIN_ID, "invalid domain ID(" << domain << ")");
  }
}

static void roctracer_disable_activity_impl(roctracer_domain_t domain, uint32_t op) {
  act_journal.Remove(domain, op);
  roctracer_disable_activity_fun(domain, op);
}

PUBLIC_API roctracer_status_t roctracer_disable_op_activity(roctracer_domain_t domain,
                                                            uint32_t op) {
  API_METHOD_PREFIX
  roctracer_disable_activity_impl(domain, op);
  API_METHOD_SUFFIX
}

PUBLIC_API roctracer_status_t roctracer_disable_domain_activity(roctracer_domain_t domain) {
  API_METHOD_PREFIX
  const uint32_t op_end = get_op_end(domain);
  for (uint32_t op = get_op_begin(domain); op < op_end; ++op)
    roctracer_disable_activity_impl(domain, op);
  API_METHOD_SUFFIX
}

PUBLIC_API roctracer_status_t roctracer_disable_activity() {
  API_METHOD_PREFIX
  for (uint32_t domain = 0; domain < ACTIVITY_DOMAIN_NUMBER; ++domain) {
    const uint32_t op_end = get_op_end(domain);
    for (uint32_t op = get_op_begin(domain); op < op_end; ++op)
      roctracer_disable_activity_impl((roctracer_domain_t)domain, op);
  }
  API_METHOD_SUFFIX
}

// Close memory pool
static void roctracer_close_pool_impl(roctracer_pool_t* pool) {
  std::lock_guard lock(memory_pool_mutex);
  if (pool == nullptr) pool = reinterpret_cast<roctracer_pool_t*>(default_memory_pool);
  if (pool == nullptr) return;
  MemoryPool* p = reinterpret_cast<MemoryPool*>(pool);
  if (p == default_memory_pool) default_memory_pool = nullptr;

  // Disable any activities that specify the pool being deleted.
  std::vector<std::pair<roctracer_domain_t, uint32_t>> ops;
  act_journal.ForEach(
      [&ops, pool](roctracer_domain_t domain, uint32_t op, const ActivityJournalData& data) {
        if (pool == data.pool) ops.emplace_back(domain, op);
        return true;
      });
  for (auto&& [domain, op] : ops) roctracer_disable_activity_impl(domain, op);

  delete (p);
}

PUBLIC_API roctracer_status_t roctracer_close_pool_expl(roctracer_pool_t* pool) {
  API_METHOD_PREFIX
  roctracer_close_pool_impl(pool);
  API_METHOD_SUFFIX
}

PUBLIC_API roctracer_status_t roctracer_close_pool() {
  API_METHOD_PREFIX
  roctracer_close_pool_impl(NULL);
  API_METHOD_SUFFIX
}

// Flush available activity records
static void roctracer_flush_activity_impl(roctracer_pool_t* pool) {
  if (pool == nullptr) pool = roctracer_default_pool();
  MemoryPool* default_memory_pool = reinterpret_cast<MemoryPool*>(pool);
  if (default_memory_pool != nullptr) default_memory_pool->Flush();
}

PUBLIC_API roctracer_status_t roctracer_flush_activity_expl(roctracer_pool_t* pool) {
  API_METHOD_PREFIX
  roctracer_flush_activity_impl(pool);
  API_METHOD_SUFFIX
}

PUBLIC_API roctracer_status_t roctracer_flush_activity() {
  API_METHOD_PREFIX
  roctracer_flush_activity_impl(nullptr);
  API_METHOD_SUFFIX
}

// Notifies that the calling thread is entering an external API region.
// Push an external correlation id for the calling thread.
PUBLIC_API roctracer_status_t
roctracer_activity_push_external_correlation_id(activity_correlation_id_t id) {
  API_METHOD_PREFIX
  external_id_stack.push(id);
  API_METHOD_SUFFIX
}

// Notifies that the calling thread is leaving an external API region.
// Pop an external correlation id for the calling thread.
// 'lastId' returns the last external correlation
PUBLIC_API roctracer_status_t
roctracer_activity_pop_external_correlation_id(activity_correlation_id_t* last_id) {
  API_METHOD_PREFIX
  if (last_id != nullptr) *last_id = 0;
  if (external_id_stack.empty())
    EXC_RAISING(ROCTRACER_STATUS_ERROR_MISMATCHED_EXTERNAL_CORRELATION_ID,
                "not matching external range pop");
  if (last_id != nullptr) *last_id = external_id_stack.top();
  external_id_stack.pop();
  API_METHOD_SUFFIX
}

// Mark API
extern "C" PUBLIC_API void roctracer_mark(const char* str) {
  if (mark_api_callback_ptr) {
    mark_api_callback_ptr(ACTIVITY_DOMAIN_EXT_API, ACTIVITY_EXT_OP_MARK, str, nullptr);
    NextCorrelationId();  // account for user-defined markers when tracking
                          // correlation id
  }
}

// Start API
PUBLIC_API void roctracer_start() {
  if (set_stopped(0)) {
    if (ext_support::roctracer_start_cb) ext_support::roctracer_start_cb();
    cb_journal.ForEach([](roctracer_domain_t domain, uint32_t op, const CallbackJournalData& data) {
      roctracer_enable_callback_fun(domain, op, data.callback, data.user_data);
      return true;
    });
    act_journal.ForEach(
        [](roctracer_domain_t domain, uint32_t op, const ActivityJournalData& data) {
          roctracer_enable_activity_fun(domain, op, data.pool);
          return true;
        });
  }
}

// Stop API
PUBLIC_API void roctracer_stop() {
  if (set_stopped(1)) {
    // Must disable the activity first as the spawner checks for the activity being NULL
    // to indicate that there is no callback.
    act_journal.ForEach([](roctracer_domain_t domain, uint32_t op, const ActivityJournalData&) {
      roctracer_disable_activity_fun(domain, op);
      return true;
    });
    cb_journal.ForEach([](roctracer_domain_t domain, uint32_t op, const CallbackJournalData&) {
      roctracer_disable_callback_fun(domain, op);
      return true;
    });
    if (ext_support::roctracer_stop_cb) ext_support::roctracer_stop_cb();
  }
}

PUBLIC_API roctracer_status_t roctracer_get_timestamp(uint64_t* timestamp) {
  API_METHOD_PREFIX
  *timestamp = util::timestamp_ns();
  API_METHOD_SUFFIX
}

// Set properties
PUBLIC_API roctracer_status_t roctracer_set_properties(roctracer_domain_t domain,
                                                       void* properties) {
  API_METHOD_PREFIX
  switch (domain) {
    case ACTIVITY_DOMAIN_HSA_OPS: {
      // HSA OPS properties
      hsa_ops_properties_t* ops_properties = reinterpret_cast<hsa_ops_properties_t*>(properties);
      HsaApiTable* table = reinterpret_cast<HsaApiTable*>(ops_properties->table);

      // HSA async-copy tracing
      [[maybe_unused]] hsa_status_t status = hsa_amd_profiling_async_copy_enable(true);
      assert(status == HSA_STATUS_SUCCESS && "hsa_amd_profiling_async_copy_enable failed");
      hsa_support::hsa_amd_memory_async_copy_fn = table->amd_ext_->hsa_amd_memory_async_copy_fn;
      hsa_support::hsa_amd_memory_async_copy_rect_fn =
          table->amd_ext_->hsa_amd_memory_async_copy_rect_fn;
      table->amd_ext_->hsa_amd_memory_async_copy_fn =
          hsa_support::hsa_amd_memory_async_copy_interceptor;
      table->amd_ext_->hsa_amd_memory_async_copy_rect_fn =
          hsa_support::hsa_amd_memory_async_copy_rect_interceptor;

      break;
    }
    case ACTIVITY_DOMAIN_HSA_EVT: {
      break;
    }
    case ACTIVITY_DOMAIN_HSA_API: {
      // HSA API properties
      HsaApiTable* table = reinterpret_cast<HsaApiTable*>(properties);
      hsa_support::intercept_CoreApiTable(table->core_);
      hsa_support::intercept_AmdExtTable(table->amd_ext_);
      hsa_support::intercept_ImageExtTable(table->image_ext_);
      break;
    }
    case ACTIVITY_DOMAIN_HIP_OPS:
    case ACTIVITY_DOMAIN_HIP_API: {
      mark_api_callback_ptr = reinterpret_cast<mark_api_callback_t*>(properties);
      break;
    }
    case ACTIVITY_DOMAIN_EXT_API: {
      roctracer_ext_properties_t* ops_properties =
          reinterpret_cast<roctracer_ext_properties_t*>(properties);
      ext_support::roctracer_start_cb = ops_properties->start_cb;
      ext_support::roctracer_stop_cb = ops_properties->stop_cb;
      break;
    }
    default:
      EXC_RAISING(ROCTRACER_STATUS_ERROR_INVALID_DOMAIN_ID, "invalid domain ID(" << domain << ")");
  }
  API_METHOD_SUFFIX
}

CONSTRUCTOR_API void constructor() {
  ONLOAD_TRACE_BEG();
  util::Logger::Create();
  ONLOAD_TRACE_END();
}

DESTRUCTOR_API void destructor() {
  ONLOAD_TRACE_BEG();
  util::Logger::Destroy();
  ONLOAD_TRACE_END();
}

// HSA-runtime tool on-load method
extern "C" PUBLIC_API bool OnLoad(HsaApiTable* table, uint64_t runtime_version,
                                  uint64_t failed_tool_count,
                                  const char* const* failed_tool_names) {
  hsa_support::SaveHsaApi(table);
  return true;
}

extern "C" PUBLIC_API void OnUnload() {}