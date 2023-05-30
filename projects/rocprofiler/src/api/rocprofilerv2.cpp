#include <atomic>

#include "src/core/hsa/hsa_support.h"
#include "src/api/rocprofiler_singleton.h"
#include "src/utils/helper.h"
#include "rocprofiler.h"

// TODO(aelwazir): change that to adapt with our own Exception
// What about outside exceptions and callbacks exceptions!!
#define API_METHOD_PREFIX                                                                          \
  rocprofiler_status_t err = ROCPROFILER_STATUS_SUCCESS;                                               \
  try {
#define API_METHOD_SUFFIX                                                                          \
  }                                                                                                \
  catch (rocprofiler::Exception & e) {                                                               \
    std::cout << __FUNCTION__ << "(), " << e.what();                                               \
  }                                                                                                \
  return err;

#define API_INIT_CHECKER                                                                           \
  API_METHOD_PREFIX                                                                                \
  if (!api_started.load(std::memory_order_relaxed))                                                \
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED);

std::atomic<bool> api_started{false};

// Returns library version
ROCPROFILER_API uint32_t rocprofiler_version_major() { return ROCPROFILER_VERSION_MAJOR; }
ROCPROFILER_API uint32_t rocprofiler_version_minor() { return ROCPROFILER_VERSION_MINOR; }

// Return the error string representing the status
ROCPROFILER_API const char* rocprofiler_error_str(rocprofiler_status_t status) {
  switch (status) {
    case ROCPROFILER_STATUS_ERROR_ALREADY_INITIALIZED:
      return "ROCProfiler is already initialized\n";
    case ROCPROFILER_STATUS_ERROR_NOT_INITIALIZED:
      return "ROCProfiler is not initialized or already destroyed\n";
    case ROCPROFILER_STATUS_ERROR_SESSION_MISSING_BUFFER:
      return "Missing Buffer for a session\n";
    case ROCPROFILER_STATUS_ERROR_TIMESTAMP_NOT_APPLICABLE:
      return "Timestamps can't be collected\n";
    case ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND:
      return "Agent is not found with given identifier\n";
    case ROCPROFILER_STATUS_ERROR_AGENT_INFORMATION_MISSING:
      return "Agent information is missing for the given identifier\n";
    case ROCPROFILER_STATUS_ERROR_QUEUE_NOT_FOUND:
      return "Queue is not found for the given identifier\n";
    case ROCPROFILER_STATUS_ERROR_QUEUE_INFORMATION_MISSING:
      return "The requested information about the queue is not found\n";
    case ROCPROFILER_STATUS_ERROR_KERNEL_NOT_FOUND:
      return "Kernel is not found with given identifier\n";
    case ROCPROFILER_STATUS_ERROR_KERNEL_INFORMATION_MISSING:
      return "The requested information about the kernel is not found\n";
    case ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND:
      return "Counter is not found with the given identifier\n";
    case ROCPROFILER_STATUS_ERROR_COUNTER_INFORMATION_MISSING:
      return "The requested Counter information for the given kernel is "
             "missing\n";
    case ROCPROFILER_STATUS_ERROR_TRACER_API_DATA_NOT_FOUND:
      return "The requested Tracing API Data for the given data identifier is "
             "missing\n";
    case ROCPROFILER_STATUS_ERROR_TRACER_API_DATA_INFORMATION_MISSING:
      return "The requested information for the tracing API Data is missing\n";
    case ROCPROFILER_STATUS_ERROR_INCORRECT_DOMAIN:
      return "The given Domain is incorrect\n";
    case ROCPROFILER_STATUS_ERROR_SESSION_NOT_FOUND:
      return "The requested Session given the session identifier is not "
             "found\n";
    case ROCPROFILER_STATUS_ERROR_CORRUPTED_SESSION_BUFFER:
      return "The requested Session Buffer given the session identifier is "
             "corrupted or deleted\n";
    case ROCPROFILER_STATUS_ERROR_RECORD_CORRUPTED:
      return "The requested record given the record identifier is corrupted "
             "or deleted\n";
    case ROCPROFILER_STATUS_ERROR_INCORRECT_REPLAY_MODE:
      return "Incorrect Replay mode\n";
    case ROCPROFILER_STATUS_ERROR_SESSION_MISSING_FILTER:
      return "Missing Filter for a session\n";
    case ROCPROFILER_STATUS_ERROR_INCORRECT_SIZE:
      return "The size given for the buffer is not applicable\n";
    case ROCPROFILER_STATUS_ERROR_INCORRECT_FLUSH_INTERVAL:
      return "Incorrect Flush interval\n";
    case ROCPROFILER_STATUS_ERROR_SESSION_FILTER_DATA_MISMATCH:
      return "The session filter can't accept the given data\n";
    case ROCPROFILER_STATUS_ERROR_FILTER_DATA_CORRUPTED:
      return "The given filter data is corrupted\n";
    case ROCPROFILER_STATUS_ERROR_CORRUPTED_LABEL_DATA:
      return "The given label is corrupted\n";
    case ROCPROFILER_STATUS_ERROR_RANGE_STACK_IS_EMPTY:
      return "There is no label in the labels stack to be popped\n";
    case ROCPROFILER_STATUS_ERROR_PASS_NOT_STARTED:
      return "There is no pass that started\n";
    case ROCPROFILER_STATUS_ERROR_HAS_ACTIVE_SESSION:
      return "There is already Active session, Can't activate two session at "
             "the same time\n";
    case ROCPROFILER_STATUS_ERROR_SESSION_NOT_ACTIVE:
      return "Can't terminate a non active session\n";
    case ROCPROFILER_STATUS_ERROR_FILTER_NOT_FOUND:
      return "The required filter is not found for the given session\n";
    case ROCPROFILER_STATUS_ERROR_BUFFER_NOT_FOUND:
      return "The required buffer is not found for the given session\n";
    case ROCPROFILER_STATUS_ERROR_FILTER_NOT_SUPPORTED:
      return "The required filter is not supported\n";
    default:
      return "Unkown error has occurred\n";
  }
  return "\n";
}

// Initialize the API
ROCPROFILER_API rocprofiler_status_t rocprofiler_initialize() {
  API_METHOD_PREFIX
  if (api_started.load(std::memory_order_relaxed))
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_ALREADY_INITIALIZED);
  rocprofiler::InitROCProfilerSingleton();
  api_started.exchange(true, std::memory_order_release);
  API_METHOD_SUFFIX
}

// Finalize the API
ROCPROFILER_API rocprofiler_status_t rocprofiler_finalize() {
  API_INIT_CHECKER
  rocprofiler::ResetROCProfilerSingleton();
  api_started.exchange(false, std::memory_order_release);
  API_METHOD_SUFFIX
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_get_timestamp(rocprofiler_timestamp_t* timestamp) {
  API_INIT_CHECKER
  *timestamp = rocprofiler::GetCurrentTimestamp();
  if (timestamp->value <= 0)
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_TIMESTAMP_NOT_APPLICABLE);
  API_METHOD_SUFFIX
}

ROCPROFILER_API rocprofiler_status_t
rocprofiler_iterate_counters(rocprofiler_counters_info_callback_t counters_info_callback) {
  API_INIT_CHECKER
  return rocprofiler::IterateCounters(counters_info_callback);
  API_METHOD_SUFFIX
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_query_agent_info_size(rocprofiler_agent_info_kind_t kind,
                                                                 rocprofiler_agent_id_t agent_id,
                                                                 size_t* data_size) {
  API_INIT_CHECKER
  if (!rocprofiler::GetROCProfilerSingleton()->FindAgent(agent_id))
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND);
  *data_size = rocprofiler::GetROCProfilerSingleton()->GetAgentInfoSize(kind, agent_id);
  if (*data_size <= 0) throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_AGENT_INFORMATION_MISSING);
  API_METHOD_SUFFIX
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_query_agent_info(rocprofiler_agent_info_kind_t kind,
                                                            rocprofiler_agent_id_t agent_id,
                                                            const char** data) {
  API_INIT_CHECKER
  if (!rocprofiler::GetROCProfilerSingleton()->FindAgent(agent_id))
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND);
  if (!(*data = rocprofiler::GetROCProfilerSingleton()->GetAgentInfo(kind, agent_id)))
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_AGENT_INFORMATION_MISSING);
  API_METHOD_SUFFIX
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_query_queue_info_size(rocprofiler_queue_info_kind_t kind,
                                                                 rocprofiler_queue_id_t queue_id,
                                                                 size_t* data_size) {
  API_INIT_CHECKER
  if (!rocprofiler::GetROCProfilerSingleton()->FindQueue(queue_id))
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_QUEUE_NOT_FOUND);
  *data_size = rocprofiler::GetROCProfilerSingleton()->GetQueueInfoSize(kind, queue_id);
  if (*data_size <= 0) throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_QUEUE_INFORMATION_MISSING);
  API_METHOD_SUFFIX
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_query_queue_info(rocprofiler_queue_info_kind_t kind,
                                                            rocprofiler_queue_id_t queue_id,
                                                            const char** data) {
  API_INIT_CHECKER
  if (!rocprofiler::GetROCProfilerSingleton()->FindQueue(queue_id))
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_QUEUE_NOT_FOUND);
  if (!(*data = rocprofiler::GetROCProfilerSingleton()->GetQueueInfo(kind, queue_id)))
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_QUEUE_INFORMATION_MISSING);
  API_METHOD_SUFFIX
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_query_kernel_info_size(rocprofiler_kernel_info_kind_t kind,
                                                                  rocprofiler_kernel_id_t kernel_id,
                                                                  size_t* data_size) {
  API_INIT_CHECKER
  // if (!rocprofiler::GetROCProfilerSingleton()->FindKernel(kernel_id))
  //   throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_KERNEL_NOT_FOUND);
  *data_size = rocprofiler::GetROCProfilerSingleton()->GetKernelInfoSize(kind, kernel_id);
  if (*data_size <= 0)
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_KERNEL_INFORMATION_MISSING);
  API_METHOD_SUFFIX
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_query_kernel_info(rocprofiler_kernel_info_kind_t kind,
                                                             rocprofiler_kernel_id_t kernel_id,
                                                             const char** data) {
  API_INIT_CHECKER
  // if (!rocprofiler::GetROCProfilerSingleton()->FindKernel(kernel_id))
  //   throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_KERNEL_NOT_FOUND);
  if (!(*data = rocprofiler::GetROCProfilerSingleton()->GetKernelInfo(kind, kernel_id)))
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_KERNEL_INFORMATION_MISSING);
  API_METHOD_SUFFIX
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_query_counter_info_size(
    rocprofiler_session_id_t session_id, rocprofiler_counter_info_kind_t kind,
    rocprofiler_counter_id_t counter_id, size_t* data_size) {
  API_INIT_CHECKER
  if (!rocprofiler::GetROCProfilerSingleton()->GetSession(session_id)->GetProfiler()->FindCounter(counter_id))
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND);
  *data_size = rocprofiler::GetROCProfilerSingleton()
                   ->GetSession(session_id)
                   ->GetProfiler()
                   ->GetCounterInfoSize(kind, counter_id);
  if (*data_size <= 0)
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_COUNTER_INFORMATION_MISSING);
  API_METHOD_SUFFIX
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_query_counter_info(rocprofiler_session_id_t session_id,
                                                              rocprofiler_counter_info_kind_t kind,
                                                              rocprofiler_counter_id_t counter_id,
                                                              const char** data) {
  API_INIT_CHECKER
  if (!rocprofiler::GetROCProfilerSingleton()->GetSession(session_id)->GetProfiler()->FindCounter(counter_id))
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND);
  if (!(*data = rocprofiler::GetROCProfilerSingleton()
                    ->GetSession(session_id)
                    ->GetProfiler()
                    ->GetCounterInfo(kind, counter_id)))
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_COUNTER_INFORMATION_MISSING);
  API_METHOD_SUFFIX
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_query_roctx_tracer_api_data_info_size(
    rocprofiler_session_id_t session_id, rocprofiler_tracer_roctx_api_data_info_t kind,
    rocprofiler_tracer_api_data_handle_t api_data_id, rocprofiler_tracer_operation_id_t operation_id,
    size_t* data_size) {
  API_INIT_CHECKER
  // TODO(aelwazir): To be implemented
  // if (!rocprofiler::GetROCProfilerSingleton()
  //          ->GetSession(session_id)
  //          ->GetTracer()
  //          ->FindROCTxApiData(api_data_id)) {
  //   if (rocprofiler::GetROCProfilerSingleton()
  //           ->GetSession(session_id)
  //           ->GetTracer()
  //           ->FindHSAApiData(api_data_id) ||
  //       rocprofiler::GetROCProfilerSingleton()
  //           ->GetSession(session_id)
  //           ->GetTracer()
  //           ->FindHIPApiData(api_data_id)) {
  //     throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_INCORRECT_DOMAIN);
  //   } else {
  //     throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_TRACER_API_DATA_NOT_FOUND);
  //   }
  // }
  *data_size = rocprofiler::GetROCProfilerSingleton()
                   ->GetSession(session_id)
                   ->GetTracer()
                   ->GetROCTxApiDataInfoSize(kind, api_data_id, operation_id);
  // if (*data_size <= 0)
  //   throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_TRACER_API_DATA_INFORMATION_MISSING);
  API_METHOD_SUFFIX
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_query_roctx_tracer_api_data_info(
    rocprofiler_session_id_t session_id, rocprofiler_tracer_roctx_api_data_info_t kind,
    rocprofiler_tracer_api_data_handle_t api_data_id, rocprofiler_tracer_operation_id_t operation_id,
    char** data) {
  API_INIT_CHECKER
  // TODO(aelwazir): To be implemented
  // if (!rocprofiler::GetROCProfilerSingleton()
  //          ->GetSession(session_id)
  //          ->GetTracer()
  //          ->FindROCTxApiData(api_data_id)) {
  //   if (rocprofiler::GetROCProfilerSingleton()
  //           ->GetSession(session_id)
  //           ->GetTracer()
  //           ->FindHSAApiData(api_data_id) ||
  //       rocprofiler::GetROCProfilerSingleton()
  //           ->GetSession(session_id)
  //           ->GetTracer()
  //           ->FindHIPApiData(api_data_id)) {
  //     throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_INCORRECT_DOMAIN);
  //   } else {
  //     throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_TRACER_API_DATA_NOT_FOUND);
  //   }
  // }
  if (!(*data = rocprofiler::GetROCProfilerSingleton()
                    ->GetSession(session_id)
                    ->GetTracer()
                    ->GetROCTxApiDataInfo(kind, api_data_id, operation_id)))
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_TRACER_API_DATA_INFORMATION_MISSING);
  API_METHOD_SUFFIX
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_query_hsa_tracer_api_data_info_size(
    rocprofiler_session_id_t session_id, rocprofiler_tracer_hsa_api_data_info_t kind,
    rocprofiler_tracer_api_data_handle_t api_data_id, rocprofiler_tracer_operation_id_t operation_id,
    size_t* data_size) {
  API_INIT_CHECKER
  // TODO(aelwazir): To be implemented
  // if (!rocprofiler::GetROCProfilerSingleton()
  //          ->GetSession(session_id)
  //          ->GetTracer()
  //          ->FindHSAApiData(api_data_id)) {
  //   if (rocprofiler::GetROCProfilerSingleton()
  //           ->GetSession(session_id)
  //           ->GetTracer()
  //           ->FindROCTxApiData(api_data_id) ||
  //       rocprofiler::GetROCProfilerSingleton()
  //           ->GetSession(session_id)
  //           ->GetTracer()
  //           ->FindHIPApiData(api_data_id)) {
  //     throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_INCORRECT_DOMAIN);
  //   } else {
  //     throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_TRACER_API_DATA_NOT_FOUND);
  //   }
  // }
  *data_size = ((kind == ROCPROFILER_HSA_FUNCTION_NAME)
      ? rocprofiler::tracer::GetApiCallFunctionNameSize(ACTIVITY_DOMAIN_HSA_API, operation_id)
      : rocprofiler::GetROCProfilerSingleton()
                   ->GetSession(session_id)
                   ->GetTracer()
                   ->GetHSAApiDataInfoSize(kind, api_data_id, operation_id));
  if (*data_size <= 0)
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_TRACER_API_DATA_INFORMATION_MISSING);
  API_METHOD_SUFFIX
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_query_hsa_tracer_api_data_info(
    rocprofiler_session_id_t session_id, rocprofiler_tracer_hsa_api_data_info_t kind,
    rocprofiler_tracer_api_data_handle_t api_data_id, rocprofiler_tracer_operation_id_t operation_id,
    char** data) {
  API_INIT_CHECKER
  // TODO(aelwazir): To be implemented
  // if (!rocprofiler::GetROCProfilerSingleton()
  //          ->GetSession(session_id)
  //          ->GetTracer()
  //          ->FindHSAApiData(api_data_id)) {
  //   if (rocprofiler::GetROCProfilerSingleton()
  //           ->GetSession(session_id)
  //           ->GetTracer()
  //           ->FindROCTxApiData(api_data_id) ||
  //       rocprofiler::GetROCProfilerSingleton()
  //           ->GetSession(session_id)
  //           ->GetTracer()
  //           ->FindHIPApiData(api_data_id)) {
  //     throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_INCORRECT_DOMAIN);
  //   } else {
  //     throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_TRACER_API_DATA_NOT_FOUND);
  //   }
  // }
  if (!(*data = (kind == ROCPROFILER_HSA_FUNCTION_NAME)
      ? rocprofiler::tracer::GetApiCallFunctionName(ACTIVITY_DOMAIN_HSA_API, operation_id)
      : rocprofiler::GetROCProfilerSingleton()
                    ->GetSession(session_id)
                    ->GetTracer()
                    ->GetHSAApiDataInfo(kind, api_data_id, operation_id)))
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_TRACER_API_DATA_INFORMATION_MISSING);
  API_METHOD_SUFFIX
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_query_hip_tracer_api_data_info_size(
    rocprofiler_session_id_t session_id, rocprofiler_tracer_hip_api_data_info_t kind,
    rocprofiler_tracer_api_data_handle_t api_data_id, rocprofiler_tracer_operation_id_t operation_id,
    size_t* data_size) {
  API_INIT_CHECKER
  // TODO(aelwazir): To be implemented
  // if (!rocprofiler::GetROCProfilerSingleton()
  //          ->GetSession(session_id)
  //          ->GetTracer()
  //          ->FindHIPApiData(api_data_id)) {
  //   if (rocprofiler::GetROCProfilerSingleton()
  //           ->GetSession(session_id)
  //           ->GetTracer()
  //           ->FindHSAApiData(api_data_id) ||
  //       rocprofiler::GetROCProfilerSingleton()
  //           ->GetSession(session_id)
  //           ->GetTracer()
  //           ->FindROCTxApiData(api_data_id)) {
  //     throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_INCORRECT_DOMAIN);
  //   } else {
  //     throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_TRACER_API_DATA_NOT_FOUND);
  //   }
  // }
  *data_size = (kind == ROCPROFILER_HIP_FUNCTION_NAME)
      ? rocprofiler::tracer::GetApiCallFunctionNameSize(ACTIVITY_DOMAIN_HIP_API, operation_id)
      : rocprofiler::GetROCProfilerSingleton()
                   ->GetSession(session_id)
                   ->GetTracer()
                   ->GetHIPApiDataInfoSize(kind, api_data_id, operation_id);
  // if (*data_size <= 0)
  // throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_TRACER_API_DATA_INFORMATION_MISSING);
  API_METHOD_SUFFIX
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_query_hip_tracer_api_data_info(
    rocprofiler_session_id_t session_id, rocprofiler_tracer_hip_api_data_info_t kind,
    rocprofiler_tracer_api_data_handle_t api_data_id, rocprofiler_tracer_operation_id_t operation_id,
    char** data) {
  API_INIT_CHECKER
  // TODO(aelwazir): To be implemented
  // if (!rocprofiler::GetROCProfilerSingleton()
  //          ->GetSession(session_id)
  //          ->GetTracer()
  //          ->FindHIPApiData(api_data_id)) {
  //   if (rocprofiler::GetROCProfilerSingleton()
  //           ->GetSession(session_id)
  //           ->GetTracer()
  //           ->FindHSAApiData(api_data_id) ||
  //       rocprofiler::GetROCProfilerSingleton()
  //           ->GetSession(session_id)
  //           ->GetTracer()
  //           ->FindROCTxApiData(api_data_id)) {
  //     throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_INCORRECT_DOMAIN);
  //   } else {
  //     throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_TRACER_API_DATA_NOT_FOUND);
  //   }
  // }
  // if (!(
  *data = (kind == ROCPROFILER_HIP_FUNCTION_NAME)
      ? rocprofiler::tracer::GetApiCallFunctionName(ACTIVITY_DOMAIN_HIP_API, operation_id)
      : rocprofiler::GetROCProfilerSingleton()
              ->GetSession(session_id)
              ->GetTracer()
              ->GetHIPApiDataInfo(kind, api_data_id, operation_id);
  // ))
  // throw
  // rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_TRACER_API_DATA_INFORMATION_MISSING);
  API_METHOD_SUFFIX
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_flush_data(rocprofiler_session_id_t session_id,
                                                      rocprofiler_buffer_id_t buffer_id) {
  API_INIT_CHECKER
  if (!rocprofiler::GetROCProfilerSingleton()->FindSession(session_id))
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_SESSION_NOT_FOUND);
  if (!rocprofiler::GetROCProfilerSingleton()->GetSession(session_id)->FindBuffer(buffer_id))
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_BUFFER_NOT_FOUND);
  if (!rocprofiler::GetROCProfilerSingleton()->GetSession(session_id)->GetBuffer(buffer_id)->Flush())
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_CORRUPTED_SESSION_BUFFER);
  API_METHOD_SUFFIX
}

#include "src/core/memory/generic_buffer.h"

ROCPROFILER_API rocprofiler_status_t rocprofiler_next_record(const rocprofiler_record_header_t* record,
                                                       const rocprofiler_record_header_t** next,
                                                       rocprofiler_session_id_t session_id,
                                                       rocprofiler_buffer_id_t buffer_id) {
  API_INIT_CHECKER
  if (!rocprofiler::GetROCProfilerSingleton()->FindSession(session_id))
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_SESSION_NOT_FOUND);
  if (!rocprofiler::GetROCProfilerSingleton()->GetSession(session_id)->FindBuffer(buffer_id))
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_BUFFER_NOT_FOUND);
  if (!Memory::GetNextRecord(record, next))
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_RECORD_CORRUPTED);
  API_METHOD_SUFFIX
}

// API to create a session with a given profiling mode and input data
ROCPROFILER_API rocprofiler_status_t rocprofiler_create_session(rocprofiler_replay_mode_t replay_mode,
                                                          rocprofiler_session_id_t* session_id) {
  API_INIT_CHECKER
  *session_id = rocprofiler::GetROCProfilerSingleton()->CreateSession(replay_mode);
  API_METHOD_SUFFIX
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_create_filter(rocprofiler_session_id_t session_id,
                                                         rocprofiler_filter_kind_t filter_kind,
                                                         rocprofiler_filter_data_t filter_data,
                                                         uint64_t data_count,
                                                         rocprofiler_filter_id_t* filter_id,
                                                         rocprofiler_filter_property_t property) {
  API_INIT_CHECKER
  // TODO(aelwazir): CheckFilterData to be implemented
  // int error_code =
  //     rocprofiler::GetROCProfilerSingleton()->CheckFilterData(filter_kind,
  //     filter_data);
  // if (error_code == -1) throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_FILTER_DATA_CORRUPTED);
  // if (error_code == 0)
  //   throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_SESSION_FILTER_DATA_MISMATCH);
  if (!rocprofiler::GetROCProfilerSingleton()->FindSession(session_id))
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_SESSION_NOT_FOUND);
  *filter_id = rocprofiler::GetROCProfilerSingleton()
                   ->GetSession(session_id)
                   ->CreateFilter(filter_kind, filter_data, data_count, property);
  API_METHOD_SUFFIX
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_destroy_filter(rocprofiler_session_id_t session_id,
                                                          rocprofiler_filter_id_t filter_id) {
  API_INIT_CHECKER
  if (!rocprofiler::GetROCProfilerSingleton()->FindSession(session_id))
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_SESSION_NOT_FOUND);
  if (!rocprofiler::GetROCProfilerSingleton()->GetSession(session_id)->FindFilter(filter_id))
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_FILTER_NOT_FOUND);
  rocprofiler::GetROCProfilerSingleton()->GetSession(session_id)->DestroyFilter(filter_id);
  API_METHOD_SUFFIX
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_create_buffer(
    rocprofiler_session_id_t session_id, rocprofiler_buffer_callback_t buffer_callback,
    size_t buffer_size, rocprofiler_buffer_id_t* buffer_id) {
  API_INIT_CHECKER
  if (!rocprofiler::GetROCProfilerSingleton()->FindSession(session_id))
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_SESSION_NOT_FOUND);
  *buffer_id = rocprofiler::GetROCProfilerSingleton()
                   ->GetSession(session_id)
                   ->CreateBuffer(buffer_callback, buffer_size);
  API_METHOD_SUFFIX
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_set_buffer_properties(
    rocprofiler_session_id_t session_id, rocprofiler_buffer_id_t buffer_id,
    rocprofiler_buffer_property_t* buffer_properties, uint32_t buffer_properties_count) {
  API_INIT_CHECKER
  if (!rocprofiler::GetROCProfilerSingleton()->FindSession(session_id))
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_SESSION_NOT_FOUND);
  if (!rocprofiler::GetROCProfilerSingleton()->GetSession(session_id)->FindBuffer(buffer_id))
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_BUFFER_NOT_FOUND);
  rocprofiler::GetROCProfilerSingleton()
      ->GetSession(session_id)
      ->GetBuffer(buffer_id)
      ->SetProperties(buffer_properties, buffer_properties_count);
  API_METHOD_SUFFIX
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_destroy_buffer(rocprofiler_session_id_t session_id,
                                                          rocprofiler_buffer_id_t buffer_id) {
  API_INIT_CHECKER
  if (!rocprofiler::GetROCProfilerSingleton()->FindSession(session_id))
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_SESSION_NOT_FOUND);
  if (!rocprofiler::GetROCProfilerSingleton()->GetSession(session_id)->FindBuffer(buffer_id))
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_BUFFER_NOT_FOUND);
  rocprofiler::GetROCProfilerSingleton()->GetSession(session_id)->DestroyBuffer(buffer_id);
  API_METHOD_SUFFIX
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_set_filter_buffer(rocprofiler_session_id_t session_id,
                                                             rocprofiler_filter_id_t filter_id,
                                                             rocprofiler_buffer_id_t buffer_id) {
  API_INIT_CHECKER
  if (!rocprofiler::GetROCProfilerSingleton()->FindSession(session_id))
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_SESSION_NOT_FOUND);
  if (!rocprofiler::GetROCProfilerSingleton()->GetSession(session_id)->FindBuffer(buffer_id))
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_BUFFER_NOT_FOUND);
  if (!rocprofiler::GetROCProfilerSingleton()->GetSession(session_id)->FindFilter(filter_id))
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_FILTER_NOT_FOUND);
  if (!rocprofiler::GetROCProfilerSingleton()
           ->GetSession(session_id)
           ->CheckFilterBufferSize(filter_id, buffer_id))
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_INCORRECT_SIZE);
  rocprofiler::GetROCProfilerSingleton()->GetSession(session_id)->GetFilter(filter_id)->SetBufferId(buffer_id);
  API_METHOD_SUFFIX
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_set_api_trace_sync_callback(
    rocprofiler_session_id_t session_id, rocprofiler_filter_id_t filter_id,
    rocprofiler_sync_callback_t callback) {
  API_INIT_CHECKER
  if (!rocprofiler::GetROCProfilerSingleton()->FindSession(session_id))
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_SESSION_NOT_FOUND);
  if (!rocprofiler::GetROCProfilerSingleton()->GetSession(session_id)->FindFilter(filter_id))
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_FILTER_NOT_FOUND);
  if (rocprofiler::GetROCProfilerSingleton()->GetSession(session_id)->GetFilter(filter_id)->GetKind() !=
      ROCPROFILER_API_TRACE)
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_FILTER_NOT_SUPPORTED);
  rocprofiler::GetROCProfilerSingleton()->GetSession(session_id)->GetFilter(filter_id)->SetCallback(callback);
  API_METHOD_SUFFIX
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_create_ready_session(
    rocprofiler_replay_mode_t replay_mode, rocprofiler_filter_kind_t filter_kind,
    rocprofiler_filter_data_t filter_data, uint64_t data_count, size_t buffer_size,
    rocprofiler_buffer_callback_t buffer_callback, rocprofiler_session_id_t* session_id,
    rocprofiler_filter_property_t property, rocprofiler_sync_callback_t callback) {
  API_INIT_CHECKER
  // TODO(aelwazir): CheckFilterData to be implemented
  // int error_code =
  //     rocprofiler::GetROCProfilerSingleton()->CheckFilterData(filter_kind,
  //     filter_data);
  // if (error_code == -1) throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_FILTER_DATA_CORRUPTED);
  // if (error_code == 0)
  //   throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_SESSION_FILTER_DATA_MISMATCH);
  *session_id = rocprofiler::GetROCProfilerSingleton()->CreateSession(replay_mode);
  rocprofiler_filter_id_t filter_id =
      rocprofiler::GetROCProfilerSingleton()
          ->GetSession(*session_id)
          ->CreateFilter(filter_kind, filter_data, data_count, property);
  rocprofiler_buffer_id_t buffer_id = rocprofiler::GetROCProfilerSingleton()
                                        ->GetSession(*session_id)
                                        ->CreateBuffer(buffer_callback, buffer_size);
  if (filter_kind == ROCPROFILER_API_TRACE)
    rocprofiler::GetROCProfilerSingleton()
        ->GetSession(*session_id)
        ->GetFilter(filter_id)
        ->SetCallback(callback);
  if (!rocprofiler::GetROCProfilerSingleton()
           ->GetSession(*session_id)
           ->CheckFilterBufferSize(filter_id, buffer_id))
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_INCORRECT_SIZE);
  rocprofiler::GetROCProfilerSingleton()
      ->GetSession(*session_id)
      ->GetFilter(filter_id)
      ->SetBufferId(buffer_id);
  API_METHOD_SUFFIX
}

// API to destroy a session by id
ROCPROFILER_API rocprofiler_status_t rocprofiler_destroy_session(rocprofiler_session_id_t session_id) {
  API_INIT_CHECKER
  if (!rocprofiler::GetROCProfilerSingleton()->FindSession(session_id))
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_SESSION_NOT_FOUND);
  rocprofiler::GetROCProfilerSingleton()->DestroySession(session_id);
  API_METHOD_SUFFIX
}

// API to activate a session by id
ROCPROFILER_API rocprofiler_status_t rocprofiler_start_session(rocprofiler_session_id_t session_id) {
  API_INIT_CHECKER
  if (!rocprofiler::GetROCProfilerSingleton()->FindSession(session_id))
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_SESSION_NOT_FOUND);
  if (!rocprofiler::GetROCProfilerSingleton()->GetSession(session_id)->HasFilter())
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_SESSION_MISSING_FILTER);
  if (!rocprofiler::GetROCProfilerSingleton()->GetSession(session_id)->HasBuffer())
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_SESSION_MISSING_BUFFER);
  if (rocprofiler::GetROCProfilerSingleton()->HasActiveSession())
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_HAS_ACTIVE_SESSION);
  rocprofiler::GetROCProfilerSingleton()->GetSession(session_id)->Start();
  rocprofiler::GetROCProfilerSingleton()->SetCurrentActiveSession(session_id);
  API_METHOD_SUFFIX
}

// API to deactivate a session by id
ROCPROFILER_API rocprofiler_status_t rocprofiler_terminate_session(rocprofiler_session_id_t session_id) {
  API_INIT_CHECKER
  if (!rocprofiler::GetROCProfilerSingleton()->FindSession(session_id))
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_SESSION_NOT_FOUND);
  if (!rocprofiler::GetROCProfilerSingleton()->IsActiveSession(session_id))
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_SESSION_NOT_ACTIVE);
  rocprofiler::GetROCProfilerSingleton()->GetSession(session_id)->Terminate();
  rocprofiler::GetROCProfilerSingleton()->SetCurrentActiveSession(rocprofiler_session_id_t{0});
  API_METHOD_SUFFIX
}


// API to push a custom label for defining a code section
ROCPROFILER_API rocprofiler_status_t rocprofiler_push_range(rocprofiler_session_id_t session_id,
                                                      const char* label) {
  API_INIT_CHECKER
  if (!label) throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_CORRUPTED_LABEL_DATA);
  rocprofiler::GetROCProfilerSingleton()->GetSession(session_id)->PushRangeLabels(label);
  API_METHOD_SUFFIX
}

// API to pop a custom label defined for a code section
ROCPROFILER_API rocprofiler_status_t rocprofiler_pop_range(rocprofiler_session_id_t session_id) {
  API_INIT_CHECKER
  if (!rocprofiler::GetROCProfilerSingleton()->GetSession(session_id)->PopRangeLabels())
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_RANGE_STACK_IS_EMPTY);
  API_METHOD_SUFFIX
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_start_replay_pass(rocprofiler_session_id_t session_id) {
  API_INIT_CHECKER
  if (!rocprofiler::GetROCProfilerSingleton()->FindSession(session_id))
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_SESSION_NOT_FOUND);
  rocprofiler::GetROCProfilerSingleton()->GetSession(session_id)->GetProfiler()->StartReplayPass(session_id);
  API_METHOD_SUFFIX
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_end_replay_pass(rocprofiler_session_id_t session_id) {
  API_INIT_CHECKER
  if (!rocprofiler::GetROCProfilerSingleton()->GetSession(session_id)->GetProfiler()->HasActivePass())
    throw rocprofiler::Exception(ROCPROFILER_STATUS_ERROR_PASS_NOT_STARTED);
  rocprofiler::GetROCProfilerSingleton()->GetSession(session_id)->GetProfiler()->EndReplayPass();
  API_METHOD_SUFFIX
}

ROCPROFILER_API rocprofiler_status_t rocprofiler_device_profiling_session_create(
    const char** counter_names, uint64_t num_counters, rocprofiler_session_id_t* session_id,
    int cpu_index, int gpu_index) {
  API_METHOD_PREFIX
  std::vector<std::string> counters(counter_names, counter_names + num_counters);
  *session_id =
      rocprofiler::GetROCProfilerSingleton()->CreateDeviceProfilingSession(counters, cpu_index, gpu_index);
  API_METHOD_SUFFIX
}

// API to start a device profiling session
ROCPROFILER_API rocprofiler_status_t
rocprofiler_device_profiling_session_start(rocprofiler_session_id_t session_id) {
  API_METHOD_PREFIX
  rocprofiler::GetROCProfilerSingleton()->GetDeviceProfilingSession(session_id)->StartSession();
  API_METHOD_SUFFIX
}

// API to poll a device profiling session
ROCPROFILER_API rocprofiler_status_t rocprofiler_device_profiling_session_poll(
    rocprofiler_session_id_t session_id, rocprofiler_device_profile_metric_t* data) {
  API_METHOD_PREFIX
  rocprofiler::GetROCProfilerSingleton()->GetDeviceProfilingSession(session_id)->PollMetrics(data);
  API_METHOD_SUFFIX
}

// API to stop a device profiling session
ROCPROFILER_API rocprofiler_status_t
rocprofiler_device_profiling_session_stop(rocprofiler_session_id_t session_id) {
  API_METHOD_PREFIX
  rocprofiler::GetROCProfilerSingleton()->GetDeviceProfilingSession(session_id)->StopSession();
  API_METHOD_SUFFIX
}

// API to destroy a device profiling session
ROCPROFILER_API rocprofiler_status_t
rocprofiler_device_profiling_session_destroy(rocprofiler_session_id_t session_id) {
  API_METHOD_PREFIX
  rocprofiler::GetROCProfilerSingleton()->DestroyDeviceProfilingSession(session_id);
  API_METHOD_SUFFIX
}


static bool started{false};

extern "C" {

// TODO(aelwazir): To be enabled if old API is deprecated

// The HSA_AMD_TOOL_PRIORITY variable must be a constant value type
// initialized by the loader itself, not by code during _init. 'extern const'
// seems do that although that is not a guarantee.
ROCPROFILER_EXPORT extern const uint32_t HSA_AMD_TOOL_PRIORITY = 25;

/**
 * @brief Callback function called upon loading the HSA.
 * The function updates the core api table function pointers to point to the
 * interceptor functions in this file.
 */
ROCPROFILER_EXPORT bool OnLoad(HsaApiTable* table, uint64_t runtime_version,
                             uint64_t failed_tool_count, const char* const* failed_tool_names) {
  if (started) rocprofiler::fatal("HSA Tool started already!");
  started = true;
  rocprofiler::hsa_support::Initialize(table);
  return true;
}

/**
 * @brief Callback function upon unloading the HSA.
 */
ROCPROFILER_EXPORT void OnUnload() {
  if (!started) rocprofiler::fatal("HSA Tool hasn't started yet!");
  rocprofiler::hsa_support::Finalize();
  started=false;
}

}  // extern "C"
