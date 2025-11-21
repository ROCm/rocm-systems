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
#include <variant>
#include <vector>

#include "correlation_id.h"
#include "exception.h"
#include "loader.h"
#include "registration_table.h"
#include "src/core/hsa/hsa_support.h"
#include "src/utils/helper.h"
#include "src/api/rocprofiler_singleton.h"

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

session_buffer_id_t session_buffer_id{};

roctracer_start_cb_t roctracer_start_cb = nullptr;
roctracer_stop_cb_t roctracer_stop_cb = nullptr;

std::mutex registration_mutex;

// Memory pool routines and primitives
std::recursive_mutex memory_pool_mutex;

}  // namespace

// Return Op code and kind by given string
void roctracer_op_code(uint32_t domain, const char* str, uint32_t* op, uint32_t* kind) {
  switch (domain) {
    case ACTIVITY_DOMAIN_HSA_API: {
      *op = hsa_support::GetApiCode(str);
      if (*op == HSA_API_ID_NUMBER) {
        throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_INVALID_OPERATION_ID,
                                     "Invalid API name, domain ID");
      }
      if (kind != nullptr) *kind = 0;
      break;
    }
    case ACTIVITY_DOMAIN_HIP_API: {
      *op = hipApiIdByName(str);
      if (*op == HIP_API_ID_NONE) {
        throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_INVALID_OPERATION_ID,
                                     "Invalid API name, domain ID");
      }
      if (kind != nullptr) *kind = 0;
      break;
    }
    default:
      throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_INVALID_OPERATION_ID,
                                   "Invalid API name, domain ID");
  }
}

// Return Op string by given domain and activity/API codes
// nullptr returned on the error and the library errno is set
const char* roctracer_op_string(uint32_t domain, uint32_t op) {
  switch (domain) {
    case ACTIVITY_DOMAIN_HSA_API:
      return hsa_support::GetApiName(op);
    case ACTIVITY_DOMAIN_HSA_EVT:
      return hsa_support::GetEvtName(op);
    case ACTIVITY_DOMAIN_HSA_OPS:
      return hsa_support::GetOpsName(op);
    case ACTIVITY_DOMAIN_HIP_OPS:
      return HipLoader::Instance().GetOpName(op);
    case ACTIVITY_DOMAIN_HIP_API:
      return HipLoader::Instance().ApiName(op);
    case ACTIVITY_DOMAIN_EXT_API:
      return "EXT_API";
    default:
      throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_INVALID_DOMAIN_ID, "Invalid domain ID");
  }
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
      throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_INVALID_DOMAIN_ID, "Invalid domain ID");
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
      throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_INVALID_DOMAIN_ID, "Invalid domain ID");
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
    util::RegistrationTable<roctracer_pool_t*, DomainTraits<domain>::kOpIdEnd, IsStopped>;

template <activity_domain_t domain> struct ApiTracer {
  using ApiData = typename DomainTraits<domain>::ApiData;
  using OperationId = typename DomainTraits<domain>::OperationId;

  struct TraceData {
    ApiData api_data;                // API specific data (for example, function arguments).
    uint64_t phase_enter_timestamp;  // timestamp when phase_enter was executed.
    uint64_t phase_data;             // data that can be shared between phase_enter and
                                     // phase_exit.

    void (*phase_enter)(OperationId operation_id, TraceData* data);
    void (*phase_exit)(OperationId operation_id, TraceData* data);
  };

  static void Exit(OperationId operation_id, TraceData* trace_data) {
  rocprofiler::ROCProfiler_Singleton&  rocprofiler_singleton =
  rocprofiler::ROCProfiler_Singleton::GetInstance();
    if (auto pool = activity_table.Get(operation_id)) {
        if (rocprofiler_singleton.GetSession((*pool)->session_id) &&
            rocprofiler_singleton
                .GetSession((*pool)->session_id)
                ->GetBuffer((*pool)->buffer_id)) {
          if (rocprofiler_singleton
                  .GetSession((*pool)->session_id)
                  ->GetBuffer((*pool)->buffer_id)
                  ->IsValid()) {
            std::lock_guard<std::mutex> lock(rocprofiler_singleton
                                                 .GetSession((*pool)->session_id)
                                                 ->GetBuffer((*pool)->buffer_id)
                                                 ->GetBufferLock());
            assert(trace_data != nullptr);
            rocprofiler_record_tracer_t record{};
            record.header = rocprofiler_record_header_t{
                ROCPROFILER_TRACER_RECORD,
                rocprofiler_record_id_t{rocprofiler_singleton.GetUniqueRecordId()}};
            record.domain = domain;
            record.operation_id = rocprofiler_tracer_operation_id_t{operation_id};
            record.correlation_id =
                rocprofiler_tracer_activity_correlation_id_t{trace_data->api_data.correlation_id};
            record.timestamps = rocprofiler_record_header_timestamp_t{
                rocprofiler_timestamp_t{trace_data->phase_enter_timestamp},
                rocprofiler_singleton.timestamp_ns()};
            record.thread_id = rocprofiler_thread_id_t{GetTid()};
            record.phase = ROCPROFILER_PHASE_NONE;

            if (auto external_id = ExternalCorrelationId()) {
              rocprofiler_record_tracer_t ext_record{};
              ext_record.header = rocprofiler_record_header_t{
                  ROCPROFILER_TRACER_RECORD,
                  rocprofiler_record_id_t{rocprofiler_singleton.GetUniqueRecordId()}};
              ext_record.domain = ACTIVITY_DOMAIN_EXT_API;
              ext_record.operation_id =
                  rocprofiler_tracer_operation_id_t{ACTIVITY_EXT_OP_EXTERN_ID};
              ext_record.correlation_id =
                  rocprofiler_tracer_activity_correlation_id_t{record.correlation_id};
              ext_record.external_id = rocprofiler_tracer_external_id_t{*external_id};
              ext_record.phase = ROCPROFILER_PHASE_NONE;
              // Write the external correlation id record directly followed by the
              // activity record.
              rocprofiler_singleton
                  .GetSession((*pool)->session_id)
                  ->GetBuffer((*pool)->buffer_id)
                  ->AddRecord(std::array<rocprofiler_record_tracer_t, 2>{ext_record, record});
            } else {
              // Write record to the buffer.
              rocprofiler_singleton
                  .GetSession((*pool)->session_id)
                  ->GetBuffer((*pool)->buffer_id)
                  ->AddRecord(record);
            }
          }
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
        trace_data->phase_enter_timestamp =  rocprofiler::ROCProfiler_Singleton::GetInstance().timestamp_ns().value;
        trace_data->phase_enter = nullptr;
        trace_data->phase_exit = Exit;
      }
      if (callback_enabled) {
        trace_data->phase_enter = Enter_UserCallback;
        trace_data->phase_exit = [](OperationId, TraceData*) {
          rocprofiler::fatal("should not reach here");
        };
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
  rocprofiler::ROCProfiler_Singleton&  rocprofiler_singleton = rocprofiler::ROCProfiler_Singleton::GetInstance();
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
          // and make the record point to it. Older HIP runtimes do not provide a kernel name,
          // so record.kernel_name might be null.
            if (rocprofiler_singleton.GetSession((*pool)->session_id) &&
                rocprofiler_singleton.GetSession((*pool)->session_id)
                    ->GetBuffer((*pool)->buffer_id)) {
              std::lock_guard<std::mutex> lock(
                  rocprofiler_singleton.GetSession((*pool)->session_id)
                      ->GetBuffer((*pool)->buffer_id)
                      ->GetBufferLock());

              rocprofiler_record_tracer_t rocprofiler_record{};
              rocprofiler_record.header = rocprofiler_record_header_t{
                  ROCPROFILER_TRACER_RECORD,
                  rocprofiler_record_id_t{rocprofiler_singleton.GetUniqueRecordId()}};
              rocprofiler_record.domain = domain;
              rocprofiler_record.external_id = rocprofiler_tracer_external_id_t{};
              rocprofiler_record.operation_id = rocprofiler_tracer_operation_id_t{record->kind};
              rocprofiler_record.api_data = rocprofiler_tracer_api_data_t{};
              rocprofiler_record.correlation_id =
                  rocprofiler_tracer_activity_correlation_id_t{record->correlation_id};
              rocprofiler_record.timestamps =
                  rocprofiler_record_header_timestamp_t{rocprofiler_timestamp_t{record->begin_ns},
                                                        rocprofiler_timestamp_t{record->end_ns}};
              rocprofiler_record.agent_id = rocprofiler_agent_id_t{(uint64_t)record->device_id};
              rocprofiler_record.queue_id = rocprofiler_queue_id_t{record->queue_id};
              rocprofiler_record.thread_id = rocprofiler_thread_id_t{GetTid()};
              rocprofiler_record.phase = ROCPROFILER_PHASE_NONE;
              if (operation_id == HIP_OP_ID_DISPATCH && record->kernel_name != nullptr) {
                rocprofiler_record.name = record->kernel_name;
                size_t kernel_name_size = (strlen(record->kernel_name) + 1);

                rocprofiler_singleton.GetSession((*pool)->session_id)
                    ->GetBuffer((*pool)->buffer_id)
                    ->AddRecord(rocprofiler_record, rocprofiler_record.name, kernel_name_size,
                                [](auto& rocprofiler_record, const void* data) {
                                  rocprofiler_record.name = static_cast<const char*>(data);
                                });
              } else {
                rocprofiler_singleton.GetSession((*pool)->session_id)
                    ->GetBuffer((*pool)->buffer_id)
                    ->AddRecord(rocprofiler_record);
              }
            }
        }
        return 0;
      }
      break;

    case ACTIVITY_DOMAIN_ROCTX: {
      auto user_callback = roctx_api_callback_table.Get(operation_id);
      if (user_callback) {
        if (user_callback->first) {
          if (auto api_data = static_cast<DomainTraits<ACTIVITY_DOMAIN_ROCTX>::ApiData*>(data))
            user_callback->first(ACTIVITY_DOMAIN_ROCTX, operation_id, api_data,
                                 user_callback->second);
          return 0;
        } else {

          if (
               rocprofiler_singleton.GetSession(
                  reinterpret_cast<session_buffer_id_t*>(user_callback->second)->session_id) &&
               rocprofiler_singleton.GetSession(
                      reinterpret_cast<session_buffer_id_t*>(user_callback->second)->session_id)
                  ->GetBuffer(
                      reinterpret_cast<session_buffer_id_t*>(user_callback->second)->buffer_id)) {
            if (auto api_data = static_cast<DomainTraits<ACTIVITY_DOMAIN_ROCTX>::ApiData*>(data)) {
              std::lock_guard<std::mutex> lock(
                      rocprofiler_singleton.
                      GetSession(
                          reinterpret_cast<session_buffer_id_t*>(user_callback->second)->session_id)
                      ->GetBuffer(
                          reinterpret_cast<session_buffer_id_t*>(user_callback->second)->buffer_id)
                      ->GetBufferLock());
              rocprofiler_tracer_api_data_t tracer_api_data{};
              rocprofiler_record_tracer_t rocprofiler_record{
                  rocprofiler_record_header_t{
                      ROCPROFILER_TRACER_RECORD,
                      rocprofiler_record_id_t{
                          rocprofiler_singleton.GetUniqueRecordId()}},
                  rocprofiler_tracer_external_id_t{api_data ? api_data->args.id : 0},
                  ACTIVITY_DOMAIN_ROCTX,
                  rocprofiler_tracer_operation_id_t{operation_id},
                  tracer_api_data,
                  rocprofiler_tracer_activity_correlation_id_t{0},
                  rocprofiler_record_header_timestamp_t{rocprofiler_singleton.timestamp_ns(),
                                                        rocprofiler_timestamp_t{0}},
                  0,
                  0,
                  GetTid(),
                  ROCPROFILER_PHASE_ENTER,
                  api_data ? (api_data->args.message ? api_data->args.message : nullptr) : nullptr};
              size_t message_size = 0;
              if (api_data && api_data->args.message) {
                message_size = strlen(api_data->args.message) + 1;
              }
              rocprofiler_singleton
                  .GetSession(
                      reinterpret_cast<session_buffer_id_t*>(user_callback->second)->session_id)
                  ->GetBuffer(
                      reinterpret_cast<session_buffer_id_t*>(user_callback->second)->buffer_id)
                  ->AddRecord(rocprofiler_record, rocprofiler_record.name, message_size,
                              [](auto& rocprofiler_record, const void* data) {
                                rocprofiler_record.name = static_cast<const char*>(data);
                              });
            }
          }
        }
      }
      break;
    }

    case ACTIVITY_DOMAIN_HSA_OPS:
      if (auto pool = hsa_ops_activity_table.Get(operation_id)) {
        if (auto record = static_cast<activity_record_t*>(data)) {
          if (rocprofiler_singleton.GetSession((*pool)->session_id) &&
              rocprofiler_singleton.GetSession((*pool)->session_id)
                  ->GetBuffer((*pool)->buffer_id)) {
            std::lock_guard<std::mutex> lock(rocprofiler_singleton
                                                 .GetSession((*pool)->session_id)
                                                 ->GetBuffer((*pool)->buffer_id)
                                                 ->GetBufferLock());
            rocprofiler_record_tracer_t rocprofiler_record{};
            rocprofiler_record.header = rocprofiler_record_header_t{
                ROCPROFILER_TRACER_RECORD,
                rocprofiler_record_id_t{rocprofiler_singleton.GetUniqueRecordId()}};
            rocprofiler_record.domain = domain;
            rocprofiler_record.external_id = rocprofiler_tracer_external_id_t{0};
            rocprofiler_record.operation_id = rocprofiler_tracer_operation_id_t{record->op};
            rocprofiler_record.api_data = rocprofiler_tracer_api_data_t{};
            rocprofiler_record.correlation_id =
                rocprofiler_tracer_activity_correlation_id_t{record->correlation_id};
            rocprofiler_record.timestamps = rocprofiler_record_header_timestamp_t{
                rocprofiler_timestamp_t{record->begin_ns}, rocprofiler_timestamp_t{record->end_ns}};
            rocprofiler_record.agent_id = rocprofiler_agent_id_t{(uint64_t)record->device_id};
            rocprofiler_record.queue_id = rocprofiler_queue_id_t{record->queue_id};
            rocprofiler_record.thread_id = rocprofiler_thread_id_t{GetTid()};
            rocprofiler_record.phase = ROCPROFILER_PHASE_NONE;
            if (record->kernel_name != nullptr && record->op == HSA_OP_ID_DISPATCH) {
              size_t kernel_name_size = strlen(record->kernel_name) + 1;

              rocprofiler_singleton
                  .GetSession((*pool)->session_id)
                  ->GetBuffer((*pool)->buffer_id)
                  ->AddRecord(rocprofiler_record, record->kernel_name, kernel_name_size,
                              [](auto& rocprofiler_record, const void* data) {
                                rocprofiler_record.name = static_cast<const char*>(data);
                              });
            } else {
               rocprofiler_singleton
                  .GetSession((*pool)->session_id)
                  ->GetBuffer((*pool)->buffer_id)
                  ->AddRecord(rocprofiler_record);
            }
          }
        }
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
  }  // namespace
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
static void roctracer_enable_op_callback(activity_domain_t domain, uint32_t operation_id,
                                         roctracer_rtapi_callback_t callback, void* user_data) {
  std::lock_guard lock(registration_mutex);

  if (operation_id >= get_op_end(domain) || callback == nullptr)
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENTS, "Invalid argument");

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
      throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_INVALID_DOMAIN_ID, "Invalid domain ID");
  }
}

void roctracer_enable_domain_callback(activity_domain_t domain, roctracer_rtapi_callback_t callback,
                                      void* user_data) {
  const uint32_t op_end = get_op_end(domain);
  for (uint32_t op = get_op_begin(domain); op < op_end; ++op)
    roctracer_enable_op_callback(domain, op, callback, user_data);
}

// Disable runtime API callbacks
void roctracer_disable_op_callback(activity_domain_t domain, uint32_t operation_id) {
  std::lock_guard lock(registration_mutex);

  if (operation_id >= get_op_end(domain))
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENTS, "Invalid argument");

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
      throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_INVALID_DOMAIN_ID, "Invalid domain ID");
  }
}

void roctracer_disable_domain_callback(activity_domain_t domain) {
  const uint32_t op_end = get_op_end(domain);
  for (uint32_t op = get_op_begin(domain); op < op_end; ++op)
    roctracer_disable_op_callback(domain, op);
}

// Enable activity records logging
void roctracer_enable_op_activity(activity_domain_t domain, uint32_t op,
                                  roctracer_pool_t memory_pool) {
  std::lock_guard lock(registration_mutex);

  if (memory_pool.session_id.handle > 0) {
    session_buffer_id.buffer_id = memory_pool.buffer_id;
    session_buffer_id.session_id = memory_pool.session_id;
  }

  if (op >= get_op_end(domain))
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENTS, "Invalid argument");

  switch (domain) {
    case ACTIVITY_DOMAIN_HSA_EVT:
      break;
    case ACTIVITY_DOMAIN_HSA_API:
      HSA_registration_group.Register(HSA_ApiTracer::activity_table, op, &session_buffer_id);
      break;
    case ACTIVITY_DOMAIN_HSA_OPS:
      HSA_registration_group.Register(hsa_ops_activity_table, op, &session_buffer_id);
      break;
    case ACTIVITY_DOMAIN_HIP_API:
      if (HipLoader::Instance().IsEnabled())
        HIP_registration_group.Register(HIP_ApiTracer::activity_table, op, &session_buffer_id);
      break;
    case ACTIVITY_DOMAIN_HIP_OPS:
      if (HipLoader::Instance().IsEnabled())
        HIP_registration_group.Register(hip_ops_activity_table, op, &session_buffer_id);
      break;
    case ACTIVITY_DOMAIN_ROCTX:
      if (RocTxLoader::Instance().IsEnabled())
        ROCTX_registration_group.Register(roctx_api_callback_table, op, nullptr,
                                          &session_buffer_id);
      break;
    default:
      throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_INVALID_DOMAIN_ID, "Invalid domain ID");
  }
}

void roctracer_enable_domain_activity(activity_domain_t domain, roctracer_pool_t pool) {
  const uint32_t op_end = get_op_end(domain);
  for (uint32_t op = get_op_begin(domain); op < op_end; ++op) {
    try {
      roctracer_enable_op_activity(domain, op, pool);
    } catch (const rocprofiler::Exception& err) {
      if (err.status() != ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED) throw;
    }
  }
}

// Disable activity records logging
void roctracer_disable_activity(activity_domain_t domain, uint32_t op) {
  std::lock_guard lock(registration_mutex);

  if (op >= get_op_end(domain))
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENTS, "Invalid argument");

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
      if (RocTxLoader::Instance().IsEnabled())
        ROCTX_registration_group.Unregister(roctx_api_callback_table, op);
      break;
    default:
      throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_INVALID_DOMAIN_ID, "Invalid domain ID");
  }
}

void roctracer_disable_domain_activity(activity_domain_t domain) {
  const uint32_t op_end = get_op_end(domain);
  for (uint32_t op = get_op_begin(domain); op < op_end; ++op) try {
      roctracer_disable_activity(domain, op);
    } catch (const rocprofiler::Exception& err) {
      if (err.status() != ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED) throw;
    }
}

// Notifies that the calling thread is entering an external API region.
// Push an external correlation id for the calling thread.
void roctracer_activity_push_external_correlation_id(activity_correlation_id_t id) {
  ExternalCorrelationIdPush(id);
}

// Notifies that the calling thread is leaving an external API region.
// Pop an external correlation id for the calling thread, and return it in
// 'last_id' if not null.
void roctracer_activity_pop_external_correlation_id(activity_correlation_id_t* last_id) {
  auto external_id = ExternalCorrelationIdPop();
  if (!external_id) {
    if (last_id != nullptr) *last_id = 0;
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_MISMATCHED_EXTERNAL_CORRELATION_ID,
                                 "Unbalanced external correlation id pop");
  }

  if (last_id != nullptr) *last_id = *external_id;
}

// Start API
void roctracer_start() {
  if (stopped_status.exchange(false, std::memory_order_relaxed) && roctracer_start_cb)
    roctracer_start_cb();
}

// Stop API
void roctracer_stop() {
  if (!stopped_status.exchange(true, std::memory_order_relaxed) && roctracer_stop_cb)
    roctracer_stop_cb();
}

// Set properties
void roctracer_set_properties(activity_domain_t domain, void* properties) {
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
      throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_INVALID_DOMAIN_ID, "Invalid domain ID");
  }
}

static std::string getKernelNameMultiKernelMultiDevice(hipLaunchParams* launchParamsList,
                                                       int numDevices) {
  std::stringstream name_str;
  for (int i = 0; i < numDevices; ++i) {
    if (launchParamsList[i].func != nullptr) {
      name_str << HipLoader::Instance().KernelNameRefByPtr(launchParamsList[i].func) << ":"
               << HipLoader::Instance().GetStreamDeviceId(launchParamsList[i].stream) << ";";
    }
  }
  return name_str.str();
}

template <typename... Ts> struct Overloaded : Ts... { using Ts::operator()...; };
template <class... Ts> Overloaded(Ts...) -> Overloaded<Ts...>;

std::optional<std::string> GetHipKernelName(uint32_t cid, hip_api_data_t* data) {
  std::variant<const void*, hipFunction_t> function;
  switch (cid) {
    case HIP_API_ID_hipExtLaunchMultiKernelMultiDevice: {
      return getKernelNameMultiKernelMultiDevice(
          data->args.hipExtLaunchMultiKernelMultiDevice.launchParamsList,
          data->args.hipExtLaunchMultiKernelMultiDevice.numDevices);
    }
    case HIP_API_ID_hipLaunchCooperativeKernelMultiDevice: {
      return getKernelNameMultiKernelMultiDevice(
          data->args.hipLaunchCooperativeKernelMultiDevice.launchParamsList,
          data->args.hipLaunchCooperativeKernelMultiDevice.numDevices);
    }
    case HIP_API_ID_hipLaunchKernel: {
      function = data->args.hipLaunchKernel.function_address;
      break;
    }
    case HIP_API_ID_hipExtLaunchKernel: {
      function = data->args.hipExtLaunchKernel.function_address;
      break;
    }
    case HIP_API_ID_hipLaunchCooperativeKernel: {
      function = data->args.hipLaunchCooperativeKernel.f;
      break;
    }
    case HIP_API_ID_hipLaunchByPtr: {
      function = data->args.hipLaunchByPtr.hostFunction;
      break;
    }
    case HIP_API_ID_hipGraphAddKernelNode: {
      function = data->args.hipGraphAddKernelNode.pNodeParams->func;
      break;
    }
    case HIP_API_ID_hipGraphExecKernelNodeSetParams: {
      function = data->args.hipGraphExecKernelNodeSetParams.pNodeParams->func;
      break;
    }
    case HIP_API_ID_hipGraphKernelNodeSetParams: {
      function = data->args.hipGraphKernelNodeSetParams.pNodeParams->func;
      break;
    }
    case HIP_API_ID_hipModuleLaunchKernel: {
      function = data->args.hipModuleLaunchKernel.f;
      break;
    }
    case HIP_API_ID_hipExtModuleLaunchKernel: {
      function = data->args.hipExtModuleLaunchKernel.f;
      break;
    }
    case HIP_API_ID_hipHccModuleLaunchKernel: {
      function = data->args.hipHccModuleLaunchKernel.f;
      break;
    }
    default:
      return {};
  }
  return std::visit(
      Overloaded{
          [](const void* func) { return HipLoader::Instance().KernelNameRefByPtr(func); },
          [](hipFunction_t func) { return HipLoader::Instance().KernelNameRef(func); },
      },
      function);
}
