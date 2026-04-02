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
#include <hsa/hsa_api_trace.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <atomic>
#include <mutex>
#include <stack>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "correlation_id.h"
#include "debug.h"
#include "exception.h"
#include "hsa_support.h"
#include "loader.h"
#include "logger.h"
#include "memory_pool.h"
#include "registration_table.h"

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

static inline uint32_t GetPid() {
  static auto pid = syscall(__NR_getpid);
  return pid;
}
static inline uint32_t GetTid() {
  static thread_local auto tid = syscall(__NR_gettid);
  return tid;
}

using namespace roctracer;

namespace {

///////////////////////////////////////////////////////////////////////////////////////////////////
// Internal library methods
//

roctracer_start_cb_t roctracer_start_cb = nullptr;
roctracer_stop_cb_t roctracer_stop_cb = nullptr;

roctracer_status_t GetExcStatus(const std::exception& e) {
  const ApiError* roctracer_exc_ptr = dynamic_cast<const ApiError*>(&e);
  return (roctracer_exc_ptr) ? roctracer_exc_ptr->status() : ROCTRACER_STATUS_ERROR;
}

std::mutex registration_mutex;

// Memory pool routines and primitives
std::recursive_mutex memory_pool_mutex;
MemoryPool* default_memory_pool = nullptr;

}  // namespace

///////////////////////////////////////////////////////////////////////////////////////////////////
// Public library methods
//

// Returns library version
ROCTRACER_API uint32_t roctracer_version_major() { return ROCTRACER_VERSION_MAJOR; }
ROCTRACER_API uint32_t roctracer_version_minor() { return ROCTRACER_VERSION_MINOR; }

// Returns the last error
ROCTRACER_API const char* roctracer_error_string() {
  return strdup(util::Logger::Instance().LastMessage().c_str());
}

// Return Op string by given domain and activity/API codes
// nullptr returned on the error and the library errno is set
ROCTRACER_API const char* roctracer_op_string(uint32_t domain, uint32_t op, uint32_t kind) {
  API_METHOD_PREFIX
  switch (domain) {
    case ACTIVITY_DOMAIN_HSA_API:
      return hsa_support::GetApiName(op);
    case ACTIVITY_DOMAIN_HSA_EVT:
      return hsa_support::GetEvtName(op);
    case ACTIVITY_DOMAIN_HSA_OPS:
      return hsa_support::GetOpsName(op);
    case ACTIVITY_DOMAIN_HIP_OPS:
      return HipLoader::Instance().GetOpName(kind);
    case ACTIVITY_DOMAIN_HIP_API:
      return HipLoader::Instance().ApiName(op);
    case ACTIVITY_DOMAIN_EXT_API:
      return "EXT_API";
    default:
      throw roctracer::ApiError(ROCTRACER_STATUS_ERROR_INVALID_DOMAIN_ID, "invalid domain ID");
  }
  API_METHOD_CATCH(nullptr)
}

// Return Op code and kind by given string
ROCTRACER_API roctracer_status_t roctracer_op_code(uint32_t domain, const char* str, uint32_t* op,
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

namespace {

template <activity_domain_t> struct DomainTraits;

template <> struct DomainTraits<ACTIVITY_DOMAIN_HIP_API> {
  using ApiData = hip_api_data_t;
  using OperationId = hip_api_id_t;
  static constexpr size_t kOpIdBegin = HIP_API_ID_FIRST;
  static constexpr size_t kOpIdEnd = HIP_API_ID_LAST + 1;
};

template <> struct DomainTraits<ACTIVITY_DOMAIN_HSA_API> {
  using ApiData = hsa_api_data_t;
  using OperationId = hsa_api_id_t;
  static constexpr size_t kOpIdBegin = 0;
  static constexpr size_t kOpIdEnd = HSA_API_ID_NUMBER;
};

template <> struct DomainTraits<ACTIVITY_DOMAIN_ROCTX> {
  using ApiData = roctx_api_data_t;
  using OperationId = roctx_api_id_t;
  static constexpr size_t kOpIdBegin = 0;
  static constexpr size_t kOpIdEnd = ROCTX_API_ID_NUMBER;
};

template <> struct DomainTraits<ACTIVITY_DOMAIN_HIP_OPS> {
  using OperationId = hip_op_id_t;
  static constexpr size_t kOpIdBegin = 0;
  static constexpr size_t kOpIdEnd = HIP_OP_ID_NUMBER;
};

template <> struct DomainTraits<ACTIVITY_DOMAIN_HSA_OPS> {
  using OperationId = hsa_op_id_t;
  static constexpr size_t kOpIdBegin = 0;
  static constexpr size_t kOpIdEnd = HSA_OP_ID_NUMBER;
};

template <> struct DomainTraits<ACTIVITY_DOMAIN_HSA_EVT> {
  using ApiData = hsa_evt_data_t;
  using OperationId = hsa_evt_id_t;
  static constexpr size_t kOpIdBegin = 0;
  static constexpr size_t kOpIdEnd = HSA_EVT_ID_NUMBER;
};

constexpr uint32_t get_op_begin(activity_domain_t domain) {
  switch (domain) {
    case ACTIVITY_DOMAIN_HSA_OPS:
      return DomainTraits<ACTIVITY_DOMAIN_HSA_OPS>::kOpIdBegin;
    case ACTIVITY_DOMAIN_HSA_API:
      return DomainTraits<ACTIVITY_DOMAIN_HSA_API>::kOpIdBegin;
    case ACTIVITY_DOMAIN_HSA_EVT:
      return DomainTraits<ACTIVITY_DOMAIN_HSA_EVT>::kOpIdBegin;
    case ACTIVITY_DOMAIN_HIP_OPS:
      return DomainTraits<ACTIVITY_DOMAIN_HIP_OPS>::kOpIdBegin;
    case ACTIVITY_DOMAIN_HIP_API:
      return DomainTraits<ACTIVITY_DOMAIN_HIP_API>::kOpIdBegin;
    case ACTIVITY_DOMAIN_ROCTX:
      return DomainTraits<ACTIVITY_DOMAIN_ROCTX>::kOpIdBegin;
    case ACTIVITY_DOMAIN_EXT_API:
      return 0;
    default:
      throw roctracer::ApiError(ROCTRACER_STATUS_ERROR_INVALID_DOMAIN_ID, "invalid domain ID");
  }
}

constexpr uint32_t get_op_end(activity_domain_t domain) {
  switch (domain) {
    case ACTIVITY_DOMAIN_HSA_OPS:
      return DomainTraits<ACTIVITY_DOMAIN_HSA_OPS>::kOpIdEnd;
    case ACTIVITY_DOMAIN_HSA_API:
      return DomainTraits<ACTIVITY_DOMAIN_HSA_API>::kOpIdEnd;
    case ACTIVITY_DOMAIN_HSA_EVT:
      return DomainTraits<ACTIVITY_DOMAIN_HSA_EVT>::kOpIdEnd;
    case ACTIVITY_DOMAIN_HIP_OPS:
      return DomainTraits<ACTIVITY_DOMAIN_HIP_OPS>::kOpIdEnd;
    case ACTIVITY_DOMAIN_HIP_API:
      return DomainTraits<ACTIVITY_DOMAIN_HIP_API>::kOpIdEnd;
    case ACTIVITY_DOMAIN_ROCTX:
      return DomainTraits<ACTIVITY_DOMAIN_ROCTX>::kOpIdEnd;
    case ACTIVITY_DOMAIN_EXT_API:
      return get_op_begin(ACTIVITY_DOMAIN_EXT_API);
    default:
      throw roctracer::ApiError(ROCTRACER_STATUS_ERROR_INVALID_DOMAIN_ID, "invalid domain ID");
  }
}

std::atomic<bool> stopped_status{false};

struct IsStopped {
  bool operator()() const { return stopped_status.load(std::memory_order_relaxed); }
};

struct NeverStopped {
  constexpr bool operator()() { return false; }
};

using UserCallback = std::pair<activity_rtapi_callback_t, void*>;

template <activity_domain_t domain, typename IsStopped>
using CallbackRegistrationTable =
    util::RegistrationTable<UserCallback, DomainTraits<domain>::kOpIdEnd, IsStopped>;

template <activity_domain_t domain, typename IsStopped>
using ActivityRegistrationTable =
    util::RegistrationTable<MemoryPool*, DomainTraits<domain>::kOpIdEnd, IsStopped>;

template <activity_domain_t domain> struct ApiTracer {
  using ApiData = typename DomainTraits<domain>::ApiData;
  using OperationId = typename DomainTraits<domain>::OperationId;

  struct TraceData {
    ApiData api_data;                // API specific data (for example, function arguments).
    uint64_t phase_enter_timestamp;  // timestamp when phase_enter was executed.
    uint64_t phase_data;             // data that can be shared between phase_enter and phase_exit.

    void (*phase_enter)(OperationId operation_id, TraceData* data);
    void (*phase_exit)(OperationId operation_id, TraceData* data);
  };

  static void Exit(OperationId operation_id, TraceData* trace_data) {
    if (auto pool = activity_table.Get(operation_id)) {
      assert(trace_data != nullptr);
      activity_record_t record{};

      record.domain = domain;
      record.op = operation_id;
      record.correlation_id = trace_data->api_data.correlation_id;
      record.begin_ns = trace_data->phase_enter_timestamp;
      record.end_ns = hsa_support::timestamp_ns();
      record.process_id = GetPid();
      record.thread_id = GetTid();

      if (auto external_id = ExternalCorrelationId()) {
        roctracer_record_t ext_record{};
        ext_record.domain = ACTIVITY_DOMAIN_EXT_API;
        ext_record.op = ACTIVITY_EXT_OP_EXTERN_ID;
        ext_record.correlation_id = record.correlation_id;
        ext_record.external_id = *external_id;
        // Write the external correlation id record directly followed by the activity record.
        (*pool)->Write(std::array<roctracer_record_t, 2>{ext_record, record});
      } else {
        // Write record to the buffer.
        (*pool)->Write(record);
      }
    }
    CorrelationIdPop();
  }

  static void Exit_UserCallback(OperationId operation_id, TraceData* trace_data) {
    if (auto user_callback = callback_table.Get(operation_id)) {
      assert(trace_data != nullptr);
      trace_data->api_data.phase = ACTIVITY_API_PHASE_EXIT;
      user_callback->first(domain, operation_id, &trace_data->api_data, user_callback->second);
    }
    Exit(operation_id, trace_data);
  }

  static void Enter_UserCallback(OperationId operation_id, TraceData* trace_data) {
    if (auto user_callback = callback_table.Get(operation_id)) {
      assert(trace_data != nullptr);
      trace_data->api_data.phase = ACTIVITY_API_PHASE_ENTER;
      trace_data->api_data.phase_data = &trace_data->phase_data;
      user_callback->first(domain, operation_id, &trace_data->api_data, user_callback->second);
      trace_data->phase_exit = Exit_UserCallback;
    } else {
      trace_data->phase_exit = Exit;
    }
  }

  static int Enter(OperationId operation_id, TraceData* trace_data) {
    bool callback_enabled = callback_table.Get(operation_id).has_value(),
         activity_enabled = activity_table.Get(operation_id).has_value();
    if (!callback_enabled && !activity_enabled) return -1;

    if (trace_data != nullptr) {
      // Generate a new correlation ID.
      trace_data->api_data.correlation_id = CorrelationIdPush();

      if (activity_enabled) {
        trace_data->phase_enter_timestamp = hsa_support::timestamp_ns();
        trace_data->phase_enter = nullptr;
        trace_data->phase_exit = Exit;
      }
      if (callback_enabled) {
        trace_data->phase_enter = Enter_UserCallback;
        trace_data->phase_exit = [](OperationId, TraceData*) { fatal("should not reach here"); };
      }
    }
    return 0;
  }

  static CallbackRegistrationTable<domain, IsStopped> callback_table;
  static ActivityRegistrationTable<domain, IsStopped> activity_table;
};

template <activity_domain_t domain>
CallbackRegistrationTable<domain, IsStopped> ApiTracer<domain>::callback_table;

template <activity_domain_t domain>
ActivityRegistrationTable<domain, IsStopped> ApiTracer<domain>::activity_table;

using HIP_ApiTracer = ApiTracer<ACTIVITY_DOMAIN_HIP_API>;
using HSA_ApiTracer = ApiTracer<ACTIVITY_DOMAIN_HSA_API>;

CallbackRegistrationTable<ACTIVITY_DOMAIN_ROCTX, NeverStopped> roctx_api_callback_table;
ActivityRegistrationTable<ACTIVITY_DOMAIN_HIP_OPS, IsStopped> hip_ops_activity_table;
ActivityRegistrationTable<ACTIVITY_DOMAIN_HSA_OPS, IsStopped> hsa_ops_activity_table;
CallbackRegistrationTable<ACTIVITY_DOMAIN_HSA_EVT, IsStopped> hsa_evt_callback_table;

int TracerCallback(activity_domain_t domain, uint32_t operation_id, void* data) {
  switch (domain) {
    case ACTIVITY_DOMAIN_HSA_API:
      return HSA_ApiTracer::Enter(static_cast<HSA_ApiTracer::OperationId>(operation_id),
                                  static_cast<HSA_ApiTracer::TraceData*>(data));

    case ACTIVITY_DOMAIN_HIP_API:
      return HIP_ApiTracer::Enter(static_cast<HIP_ApiTracer::OperationId>(operation_id),
                                  static_cast<HIP_ApiTracer::TraceData*>(data));

    case ACTIVITY_DOMAIN_HIP_OPS:
      if (auto pool = hip_ops_activity_table.Get(operation_id)) {
        if (auto record = static_cast<activity_record_t*>(data)) {
          // If the record is for a kernel dispatch, write the kernel name in the pool's data,
          // and make the record point to it. Older HIP runtimes do not provide a kernel
          // name, so record.kernel_name might be null.
          if (operation_id == HIP_OP_ID_DISPATCH && record->kernel_name != nullptr)
            (*pool)->Write(*record, record->kernel_name, strlen(record->kernel_name) + 1,
                           [](auto& record, const void* data) {
                             record.kernel_name = static_cast<const char*>(data);
                           });
          else
            (*pool)->Write(*record);
        }
        return 0;
      }
      break;

    case ACTIVITY_DOMAIN_ROCTX:
      if (auto user_callback = roctx_api_callback_table.Get(operation_id)) {
        if (auto api_data = static_cast<DomainTraits<ACTIVITY_DOMAIN_ROCTX>::ApiData*>(data))
          user_callback->first(ACTIVITY_DOMAIN_ROCTX, operation_id, api_data,
                               user_callback->second);
        return 0;
      }
      break;

    case ACTIVITY_DOMAIN_HSA_OPS:
      if (auto pool = hsa_ops_activity_table.Get(operation_id)) {
        if (auto record = static_cast<activity_record_t*>(data)) (*pool)->Write(*record);
        return 0;
      }
      break;

    case ACTIVITY_DOMAIN_HSA_EVT:
      if (auto user_callback = hsa_evt_callback_table.Get(operation_id)) {
        if (auto api_data = static_cast<DomainTraits<ACTIVITY_DOMAIN_HSA_EVT>::ApiData*>(data))
          user_callback->first(ACTIVITY_DOMAIN_HSA_EVT, operation_id, api_data,
                               user_callback->second);
        return 0;
      }
      break;

    default:
      break;
  }
  return -1;
}

template <typename... Tables> struct RegistrationTableGroup {
 private:
  bool AllEmpty() const {
    return std::apply([](auto&&... tables) { return (tables.IsEmpty() && ...); }, tables_);
  }

 public:
  template <typename Functor1, typename Functor2>
  RegistrationTableGroup(Functor1&& engage_tracer, Functor2&& disengage_tracer, Tables&... tables)
      : engage_tracer_(std::forward<Functor1>(engage_tracer)),
        disengage_tracer_(std::forward<Functor2>(disengage_tracer)),
        tables_(tables...) {}

  template <typename T, typename... Args>
  void Register(T& table, uint32_t operation_id, Args... args) const {
    if (AllEmpty()) engage_tracer_();
    table.Register(operation_id, std::forward<Args>(args)...);
  }

  template <typename T> void Unregister(T& table, uint32_t operation_id) const {
    table.Unregister(operation_id);
    if (AllEmpty()) disengage_tracer_();
  }

 private:
  const std::function<void()> engage_tracer_, disengage_tracer_;
  const std::tuple<const Tables&...> tables_;
};

RegistrationTableGroup HSA_registration_group(
    []() { hsa_support::RegisterTracerCallback(TracerCallback); },
    []() { hsa_support::RegisterTracerCallback(nullptr); }, HSA_ApiTracer::callback_table,
    HSA_ApiTracer::activity_table, hsa_ops_activity_table, hsa_evt_callback_table);

RegistrationTableGroup HIP_registration_group(
    []() { HipLoader::Instance().RegisterTracerCallback(TracerCallback); },
    []() { HipLoader::Instance().RegisterTracerCallback(nullptr); }, HIP_ApiTracer::callback_table,
    HIP_ApiTracer::activity_table, hip_ops_activity_table);

RegistrationTableGroup ROCTX_registration_group(
    []() { RocTxLoader::Instance().RegisterTracerCallback(TracerCallback); },
    []() { RocTxLoader::Instance().RegisterTracerCallback(nullptr); }, roctx_api_callback_table);

}  // namespace

// Enable runtime API callbacks
static void roctracer_enable_callback_impl(roctracer_domain_t domain, uint32_t operation_id,
                                           roctracer_rtapi_callback_t callback, void* user_data) {
  std::lock_guard lock(registration_mutex);

  if (operation_id >= get_op_end(domain) || callback == nullptr)
    throw ApiError(ROCTRACER_STATUS_ERROR_INVALID_ARGUMENT, "invalid argument");

  switch (domain) {
    case ACTIVITY_DOMAIN_HSA_EVT:
      HSA_registration_group.Register(hsa_evt_callback_table, operation_id, callback, user_data);
      break;
    case ACTIVITY_DOMAIN_HSA_API:
      HSA_registration_group.Register(HSA_ApiTracer::callback_table, operation_id, callback,
                                      user_data);
      break;
    case ACTIVITY_DOMAIN_HSA_OPS:
      break;
    case ACTIVITY_DOMAIN_HIP_API:
      if (HipLoader::Instance().IsEnabled())
        HIP_registration_group.Register(HIP_ApiTracer::callback_table, operation_id, callback,
                                        user_data);
      break;
    case ACTIVITY_DOMAIN_HIP_OPS:
      break;
    case ACTIVITY_DOMAIN_ROCTX:
      if (RocTxLoader::Instance().IsEnabled())
        ROCTX_registration_group.Register(roctx_api_callback_table, operation_id, callback,
                                          user_data);
      break;
    default:
      EXC_RAISING(ROCTRACER_STATUS_ERROR_INVALID_DOMAIN_ID, "invalid domain ID(" << domain << ")");
  }
}

ROCTRACER_API roctracer_status_t roctracer_enable_op_callback(roctracer_domain_t domain,
                                                              uint32_t op,
                                                              roctracer_rtapi_callback_t callback,
                                                              void* user_data) {
  API_METHOD_PREFIX
  roctracer_enable_callback_impl(domain, op, callback, user_data);
  API_METHOD_SUFFIX
}

ROCTRACER_API roctracer_status_t roctracer_enable_domain_callback(
    roctracer_domain_t domain, roctracer_rtapi_callback_t callback, void* user_data) {
  API_METHOD_PREFIX
  const uint32_t op_end = get_op_end(domain);
  for (uint32_t op = get_op_begin(domain); op < op_end; ++op)
    roctracer_enable_callback_impl(domain, op, callback, user_data);
  API_METHOD_SUFFIX
}

// Disable runtime API callbacks
static void roctracer_disable_callback_impl(roctracer_domain_t domain, uint32_t operation_id) {
  std::lock_guard lock(registration_mutex);

  if (operation_id >= get_op_end(domain))
    throw ApiError(ROCTRACER_STATUS_ERROR_INVALID_ARGUMENT, "invalid argument");

  switch (domain) {
    case ACTIVITY_DOMAIN_HSA_EVT:
      HSA_registration_group.Unregister(hsa_evt_callback_table, operation_id);
      break;
    case ACTIVITY_DOMAIN_HSA_API:
      HSA_registration_group.Unregister(HSA_ApiTracer::callback_table, operation_id);
      break;
    case ACTIVITY_DOMAIN_HSA_OPS:
      break;
    case ACTIVITY_DOMAIN_HIP_API:
      if (HipLoader::Instance().IsEnabled())
        HIP_registration_group.Unregister(HIP_ApiTracer::callback_table, operation_id);
      break;
    case ACTIVITY_DOMAIN_HIP_OPS:
      break;
    case ACTIVITY_DOMAIN_ROCTX:
      if (RocTxLoader::Instance().IsEnabled())
        ROCTX_registration_group.Unregister(roctx_api_callback_table, operation_id);
      break;
    default:
      EXC_RAISING(ROCTRACER_STATUS_ERROR_INVALID_DOMAIN_ID, "invalid domain ID(" << domain << ")");
  }
}

ROCTRACER_API roctracer_status_t roctracer_disable_op_callback(roctracer_domain_t domain,
                                                               uint32_t op) {
  API_METHOD_PREFIX
  roctracer_disable_callback_impl(domain, op);
  API_METHOD_SUFFIX
}

ROCTRACER_API roctracer_status_t roctracer_disable_domain_callback(roctracer_domain_t domain) {
  API_METHOD_PREFIX
  const uint32_t op_end = get_op_end(domain);
  for (uint32_t op = get_op_begin(domain); op < op_end; ++op)
    roctracer_disable_callback_impl(domain, op);
  API_METHOD_SUFFIX
}

// Return default pool and set new one if parameter pool is not NULL.
ROCTRACER_API roctracer_pool_t* roctracer_default_pool_expl(roctracer_pool_t* pool) {
  std::lock_guard lock(memory_pool_mutex);
  roctracer_pool_t* p = reinterpret_cast<roctracer_pool_t*>(default_memory_pool);
  if (pool != nullptr) default_memory_pool = reinterpret_cast<MemoryPool*>(pool);
  return p;
}

ROCTRACER_API roctracer_pool_t* roctracer_default_pool() {
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

ROCTRACER_API roctracer_status_t roctracer_open_pool_expl(const roctracer_properties_t* properties,
                                                          roctracer_pool_t** pool) {
  API_METHOD_PREFIX
  roctracer_open_pool_impl(properties, pool);
  API_METHOD_SUFFIX
}

ROCTRACER_API roctracer_status_t roctracer_open_pool(const roctracer_properties_t* properties) {
  API_METHOD_PREFIX
  roctracer_open_pool_impl(properties, nullptr);
  API_METHOD_SUFFIX
}

ROCTRACER_API roctracer_status_t roctracer_next_record(const activity_record_t* record,
                                                       const activity_record_t** next) {
  API_METHOD_PREFIX
  *next = record + 1;
  API_METHOD_SUFFIX
}

// Enable activity records logging
static void roctracer_enable_activity_impl(roctracer_domain_t domain, uint32_t op,
                                           roctracer_pool_t* pool) {
  std::lock_guard lock(registration_mutex);

  MemoryPool* memory_pool = reinterpret_cast<MemoryPool*>(pool);
  if (memory_pool == nullptr) memory_pool = default_memory_pool;
  if (memory_pool == nullptr)
    EXC_RAISING(ROCTRACER_STATUS_ERROR_DEFAULT_POOL_UNDEFINED, "no default pool");

  if (op >= get_op_end(domain))
    throw ApiError(ROCTRACER_STATUS_ERROR_INVALID_ARGUMENT, "invalid argument");

  switch (domain) {
    case ACTIVITY_DOMAIN_HSA_EVT:
      break;
    case ACTIVITY_DOMAIN_HSA_API:
      HSA_registration_group.Register(HSA_ApiTracer::activity_table, op, memory_pool);
      break;
    case ACTIVITY_DOMAIN_HSA_OPS:
      HSA_registration_group.Register(hsa_ops_activity_table, op, memory_pool);
      break;
    case ACTIVITY_DOMAIN_HIP_API:
      if (HipLoader::Instance().IsEnabled())
        HIP_registration_group.Register(HIP_ApiTracer::activity_table, op, memory_pool);
      break;
    case ACTIVITY_DOMAIN_HIP_OPS:
      if (HipLoader::Instance().IsEnabled())
        HIP_registration_group.Register(hip_ops_activity_table, op, memory_pool);
      break;
    case ACTIVITY_DOMAIN_ROCTX:
      break;
    default:
      EXC_RAISING(ROCTRACER_STATUS_ERROR_INVALID_DOMAIN_ID, "invalid domain ID(" << domain << ")");
  }
}

ROCTRACER_API roctracer_status_t roctracer_enable_op_activity_expl(roctracer_domain_t domain,
                                                                   uint32_t op,
                                                                   roctracer_pool_t* pool) {
  API_METHOD_PREFIX
  roctracer_enable_activity_impl(domain, op, pool);
  API_METHOD_SUFFIX
}

ROCTRACER_API roctracer_status_t roctracer_enable_op_activity(activity_domain_t domain,
                                                              uint32_t op) {
  API_METHOD_PREFIX
  roctracer_enable_activity_impl(domain, op, nullptr);
  API_METHOD_SUFFIX
}

static void roctracer_enable_domain_activity_impl(roctracer_domain_t domain,
                                                  roctracer_pool_t* pool) {
  const uint32_t op_end = get_op_end(domain);
  for (uint32_t op = get_op_begin(domain); op < op_end; ++op) try {
      roctracer_enable_activity_impl(domain, op, pool);
    } catch (const ApiError& err) {
      if (err.status() != ROCTRACER_STATUS_ERROR_NOT_IMPLEMENTED) throw;
    }
}

ROCTRACER_API roctracer_status_t roctracer_enable_domain_activity_expl(roctracer_domain_t domain,
                                                                       roctracer_pool_t* pool) {
  API_METHOD_PREFIX
  roctracer_enable_domain_activity_impl(domain, pool);
  API_METHOD_SUFFIX
}

ROCTRACER_API roctracer_status_t roctracer_enable_domain_activity(activity_domain_t domain) {
  API_METHOD_PREFIX
  roctracer_enable_domain_activity_impl(domain, nullptr);
  API_METHOD_SUFFIX
}

// Disable activity records logging
static void roctracer_disable_activity_impl(roctracer_domain_t domain, uint32_t op) {
  std::lock_guard lock(registration_mutex);

  if (op >= get_op_end(domain))
    throw ApiError(ROCTRACER_STATUS_ERROR_INVALID_ARGUMENT, "invalid argument");

  switch (domain) {
    case ACTIVITY_DOMAIN_HSA_EVT:
      break;
    case ACTIVITY_DOMAIN_HSA_API:
      HSA_registration_group.Unregister(HSA_ApiTracer::activity_table, op);
      break;
    case ACTIVITY_DOMAIN_HSA_OPS:
      HSA_registration_group.Unregister(hsa_ops_activity_table, op);
      break;
    case ACTIVITY_DOMAIN_HIP_API:
      if (HipLoader::Instance().IsEnabled())
        HIP_registration_group.Unregister(HIP_ApiTracer::activity_table, op);
      break;
    case ACTIVITY_DOMAIN_HIP_OPS:
      if (HipLoader::Instance().IsEnabled())
        HIP_registration_group.Unregister(hip_ops_activity_table, op);
      break;
    case ACTIVITY_DOMAIN_ROCTX:
      break;
    default:
      EXC_RAISING(ROCTRACER_STATUS_ERROR_INVALID_DOMAIN_ID, "invalid domain ID(" << domain << ")");
  }
}

ROCTRACER_API roctracer_status_t roctracer_disable_op_activity(roctracer_domain_t domain,
                                                               uint32_t op) {
  API_METHOD_PREFIX
  roctracer_disable_activity_impl(domain, op);
  API_METHOD_SUFFIX
}

static void roctracer_disable_domain_activity_impl(roctracer_domain_t domain) {
  const uint32_t op_end = get_op_end(domain);
  for (uint32_t op = get_op_begin(domain); op < op_end; ++op) try {
      roctracer_disable_activity_impl(domain, op);
    } catch (const ApiError& err) {
      if (err.status() != ROCTRACER_STATUS_ERROR_NOT_IMPLEMENTED) throw;
    }
}

ROCTRACER_API roctracer_status_t roctracer_disable_domain_activity(roctracer_domain_t domain) {
  API_METHOD_PREFIX
  roctracer_disable_domain_activity_impl(domain);
  API_METHOD_SUFFIX
}

// Close memory pool
static void roctracer_close_pool_impl(roctracer_pool_t* pool) {
  std::lock_guard lock(memory_pool_mutex);
  if (pool == nullptr) pool = reinterpret_cast<roctracer_pool_t*>(default_memory_pool);
  if (pool == nullptr) return;
  MemoryPool* p = reinterpret_cast<MemoryPool*>(pool);
  if (p == default_memory_pool) default_memory_pool = nullptr;

#if 0
  // Disable any activities that specify the pool being deleted.
  std::vector<std::pair<roctracer_domain_t, uint32_t>> ops;
  act_journal.ForEach(
      [&ops, pool](roctracer_domain_t domain, uint32_t op, const ActivityJournalData& data) {
        if (pool == data.pool) ops.emplace_back(domain, op);
        return true;
      });
  for (auto&& [domain, op] : ops) roctracer_disable_activity_impl(domain, op);
#endif

  delete (p);
}

ROCTRACER_API roctracer_status_t roctracer_close_pool_expl(roctracer_pool_t* pool) {
  API_METHOD_PREFIX
  roctracer_close_pool_impl(pool);
  API_METHOD_SUFFIX
}

ROCTRACER_API roctracer_status_t roctracer_close_pool() {
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

ROCTRACER_API roctracer_status_t roctracer_flush_activity_expl(roctracer_pool_t* pool) {
  API_METHOD_PREFIX
  roctracer_flush_activity_impl(pool);
  API_METHOD_SUFFIX
}

ROCTRACER_API roctracer_status_t roctracer_flush_activity() {
  API_METHOD_PREFIX
  roctracer_flush_activity_impl(nullptr);
  API_METHOD_SUFFIX
}

// Notifies that the calling thread is entering an external API region.
// Push an external correlation id for the calling thread.
ROCTRACER_API roctracer_status_t
roctracer_activity_push_external_correlation_id(activity_correlation_id_t id) {
  API_METHOD_PREFIX
  ExternalCorrelationIdPush(id);
  API_METHOD_SUFFIX
}

// Notifies that the calling thread is leaving an external API region.
// Pop an external correlation id for the calling thread, and return it in 'last_id' if not null.
ROCTRACER_API roctracer_status_t
roctracer_activity_pop_external_correlation_id(activity_correlation_id_t* last_id) {
  API_METHOD_PREFIX

  auto external_id = ExternalCorrelationIdPop();
  if (!external_id) {
    if (last_id != nullptr) *last_id = 0;
    EXC_RAISING(ROCTRACER_STATUS_ERROR_MISMATCHED_EXTERNAL_CORRELATION_ID,
                "unbalanced external correlation id pop");
  }

  if (last_id != nullptr) *last_id = *external_id;
  API_METHOD_SUFFIX
}

// Start API
ROCTRACER_API void roctracer_start() {
  if (stopped_status.exchange(false, std::memory_order_relaxed) && roctracer_start_cb)
    roctracer_start_cb();
}

// Stop API
ROCTRACER_API void roctracer_stop() {
  if (!stopped_status.exchange(true, std::memory_order_relaxed) && roctracer_stop_cb)
    roctracer_stop_cb();
}

ROCTRACER_API roctracer_status_t roctracer_get_timestamp(roctracer_timestamp_t* timestamp) {
  API_METHOD_PREFIX
  *timestamp = hsa_support::timestamp_ns();
  API_METHOD_SUFFIX
}

// Set properties
ROCTRACER_API roctracer_status_t roctracer_set_properties(roctracer_domain_t domain,
                                                          void* properties) {
  API_METHOD_PREFIX
  switch (domain) {
    case ACTIVITY_DOMAIN_HSA_OPS:
    case ACTIVITY_DOMAIN_HSA_EVT:
    case ACTIVITY_DOMAIN_HSA_API:
    case ACTIVITY_DOMAIN_HIP_OPS:
    case ACTIVITY_DOMAIN_HIP_API: {
      break;
    }
    case ACTIVITY_DOMAIN_EXT_API: {
      roctracer_ext_properties_t* ops_properties =
          reinterpret_cast<roctracer_ext_properties_t*>(properties);
      roctracer_start_cb = ops_properties->start_cb;
      roctracer_stop_cb = ops_properties->stop_cb;
      break;
    }
    default:
      EXC_RAISING(ROCTRACER_STATUS_ERROR_INVALID_DOMAIN_ID, "invalid domain ID(" << domain << ")");
  }
  API_METHOD_SUFFIX
}

extern "C" {

// The HSA_AMD_TOOL_PRIORITY variable must be a constant value type initialized by the loader
// itself, not by code during _init. 'extern const' seems to do that although that is not a
// guarantee.
ROCTRACER_EXPORT extern const uint32_t HSA_AMD_TOOL_PRIORITY = 50;

// HSA-runtime tool on-load method
ROCTRACER_EXPORT bool OnLoad(HsaApiTable* table, uint64_t runtime_version,
                             uint64_t failed_tool_count, const char* const* failed_tool_names) {
  [](auto&&...) {}(runtime_version, failed_tool_count, failed_tool_names);
  hsa_support::Initialize(table);
  return true;
}

ROCTRACER_EXPORT void OnUnload() { hsa_support::Finalize(); }

}  // extern "C"