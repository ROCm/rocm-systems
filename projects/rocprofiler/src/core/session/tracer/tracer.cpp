#include "tracer.h"

#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <map>
#include <mutex>
#include <utility>

#include "core/session/tracer/src/roctracer.h"
#include "src/api/rocprofiler_singleton.h"
#include "src/utils/helper.h"
#include "src/core/hsa/hsa_support.h"
#include "src/core/memory/generic_buffer.h"

namespace rocprofiler {
namespace tracer {

const char* GetApiCallOperationName(rocprofiler_tracer_activity_domain_t domain,
                                    rocprofiler_tracer_operation_id_t operation_id) {
  return roctracer_op_string(domain, operation_id.id);
}

bool GetApiCallOperationID(rocprofiler_tracer_activity_domain_t domain, const char* name,
                           rocprofiler_tracer_operation_id_t* operation_id) {
  assert(name != nullptr && operation_id != nullptr);
  roctracer_op_code(domain, name, &(operation_id->id), nullptr);
  return true;
}

uint32_t GetPid() {
  static uint32_t pid = syscall(__NR_getpid);
  return pid;
}
uint32_t GetTid() {
  static thread_local uint32_t tid = syscall(__NR_gettid);
  return tid;
}

Tracer::Tracer(rocprofiler_session_id_t session_id, rocprofiler_sync_callback_t callback,
               rocprofiler_buffer_id_t buffer_id,
               std::vector<rocprofiler_tracer_activity_domain_t> domains)
    : domains_(domains), callback_(callback), buffer_id_(buffer_id), session_id_(session_id) {
  assert(!is_active_.load(std::memory_order_acquire) && "Error: The tracer was initialized!");
  std::lock_guard<std::mutex> lock(tracer_lock_);
  callback_data_ = api_callback_data_t{callback, session_id};
  is_active_.exchange(true, std::memory_order_release);
}

void Tracer::StartRoctracer() {
  if (!roctracer_initiated_.load(std::memory_order_acquire)) {
    std::map<rocprofiler_tracer_activity_domain_t, is_filtered_domain_t> domains_filteration_map;
    // TODO(aelwazir): get filter property and parse it here
    for (auto& domain : domains_) {
      domains_filteration_map.emplace(domain, false);
    }
    std::vector<std::string> api_filter_data_vector;
    InitRoctracer(domains_filteration_map, api_filter_data_vector);
    roctracer_initiated_.exchange(true, std::memory_order_release);
  } else {
    roctracer_start();
  }
}

void Tracer::StopRoctracer() {
  if (roctracer_initiated_.load(std::memory_order_acquire)) roctracer_stop();
}

void Tracer::DisableRoctracer() {
  std::lock_guard<std::mutex> lock(tracer_lock_);
  for (auto domain : domains_) {
    switch (domain) {
      case ACTIVITY_DOMAIN_ROCTX: {
        roctracer_disable_domain_callback(ACTIVITY_DOMAIN_ROCTX);
        break;
      }
      case ACTIVITY_DOMAIN_HSA_API: {
        roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HSA_API);
        break;
      }
      case ACTIVITY_DOMAIN_HSA_OPS: {
        roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HSA_OPS);
        break;
      }
      case ACTIVITY_DOMAIN_HIP_API: {
        roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API);
        break;
      }
      case ACTIVITY_DOMAIN_HIP_OPS: {
        roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HIP_OPS);
        break;
      }
      // TODO(aelwazir): Make sure if any other domain is needed by the
      // API(User Usage)
      default: {
        fatal("Error: Provided Domain is not supported!");
      }
    }
  }
}

Tracer::~Tracer() {
  assert(is_active_.load(std::memory_order_acquire) && "Error: The tracer was not initialized!");
  std::lock_guard<std::mutex> lock(tracer_lock_);

  is_active_.exchange(false, std::memory_order_release);
}

std::mutex& Tracer::GetTracerLock() { return tracer_lock_; }

void api_callback(activity_domain_t domain, uint32_t cid, const void* callback_data, void* args) {
  api_callback_data_t* args_data = reinterpret_cast<api_callback_data_t*>(args);
  rocprofiler_tracer_api_data_t api_data{};
  rocprofiler::ROCProfiler_Singleton&  rocprofiler_singleton = rocprofiler::ROCProfiler_Singleton::GetInstance();
  if (args_data  &&
      rocprofiler_singleton.GetSession(args_data->session_id) &&
      rocprofiler_singleton.GetSession(args_data->session_id)->GetTracer()) {
    switch (domain) {
      case ACTIVITY_DOMAIN_ROCTX: {
        const roctx_api_data_t* data = reinterpret_cast<const roctx_api_data_t*>(callback_data);
        api_data.roctx = data;
        args_data->user_sync_callback(
            rocprofiler_record_tracer_t{
                rocprofiler_record_header_t{
                    ROCPROFILER_TRACER_RECORD,
                    rocprofiler_record_id_t{rocprofiler_singleton.GetUniqueRecordId()}},
                rocprofiler_tracer_external_id_t{data ? data->args.id : 0}, ACTIVITY_DOMAIN_ROCTX,
                rocprofiler_tracer_operation_id_t{cid}, api_data,

                rocprofiler_tracer_activity_correlation_id_t{0},
                rocprofiler_record_header_timestamp_t{rocprofiler_singleton.timestamp_ns(),
                                                      rocprofiler_timestamp_t{0}},
                0, 0, GetTid(), ROCPROFILER_PHASE_ENTER},
            args_data->session_id);
        break;
      }
      case ACTIVITY_DOMAIN_HSA_API: {
        hsa_api_data_t* data =
            const_cast<hsa_api_data_t*>(reinterpret_cast<const hsa_api_data_t*>(callback_data));
        api_data.hsa = data;
        if (data->phase == ACTIVITY_API_PHASE_ENTER) {
          args_data->user_sync_callback(
              rocprofiler_record_tracer_t{
                  rocprofiler_record_header_t{
                      ROCPROFILER_TRACER_RECORD,
                      rocprofiler_record_id_t{rocprofiler_singleton.GetUniqueRecordId()}},
                  rocprofiler_tracer_external_id_t{0}, ACTIVITY_DOMAIN_HSA_API,
                  rocprofiler_tracer_operation_id_t{cid}, api_data,
                  rocprofiler_tracer_activity_correlation_id_t{data->correlation_id},
                  rocprofiler_record_header_timestamp_t{rocprofiler_timestamp_t{0},
                                                        rocprofiler_timestamp_t{0}},
                  0, 0, GetTid(), ROCPROFILER_PHASE_ENTER},
              args_data->session_id);
        } else {
          args_data->user_sync_callback(
              rocprofiler_record_tracer_t{
                  rocprofiler_record_header_t{
                      ROCPROFILER_TRACER_RECORD,
                      rocprofiler_record_id_t{rocprofiler_singleton.GetUniqueRecordId()}},
                  rocprofiler_tracer_external_id_t{0}, ACTIVITY_DOMAIN_HSA_API,
                  rocprofiler_tracer_operation_id_t{cid}, api_data,
                  rocprofiler_tracer_activity_correlation_id_t{data->correlation_id},
                  rocprofiler_record_header_timestamp_t{rocprofiler_timestamp_t{0},
                                                        rocprofiler_timestamp_t{0}},
                  0, 0, GetTid(), ROCPROFILER_PHASE_EXIT},
              args_data->session_id);
        }
        break;
      }
      case ACTIVITY_DOMAIN_HIP_API: {
        hip_api_data_t* data =
            const_cast<hip_api_data_t*>(reinterpret_cast<const hip_api_data_t*>(callback_data));
        api_data.hip = data;
        if (data->phase == ACTIVITY_API_PHASE_ENTER) {
          args_data->user_sync_callback(
              rocprofiler_record_tracer_t{
                  rocprofiler_record_header_t{
                      ROCPROFILER_TRACER_RECORD,
                      rocprofiler_record_id_t{rocprofiler_singleton.GetUniqueRecordId()}},
                  rocprofiler_tracer_external_id_t{0}, ACTIVITY_DOMAIN_HIP_API,
                  rocprofiler_tracer_operation_id_t{cid}, api_data,
                  rocprofiler_tracer_activity_correlation_id_t{data->correlation_id},
                  rocprofiler_record_header_timestamp_t{rocprofiler_timestamp_t{0},
                                                        rocprofiler_timestamp_t{0}},
                  0, 0, GetTid(), ROCPROFILER_PHASE_ENTER},
              args_data->session_id);
        } else {
          args_data->user_sync_callback(
              rocprofiler_record_tracer_t{
                  rocprofiler_record_header_t{
                      ROCPROFILER_TRACER_RECORD,
                      rocprofiler_record_id_t{rocprofiler_singleton.GetUniqueRecordId()}},
                  rocprofiler_tracer_external_id_t{0}, ACTIVITY_DOMAIN_HIP_API,
                  rocprofiler_tracer_operation_id_t{cid}, api_data,
                  rocprofiler_tracer_activity_correlation_id_t{data->correlation_id},
                  rocprofiler_record_header_timestamp_t{rocprofiler_timestamp_t{0},
                                                        rocprofiler_timestamp_t{0}},
                  0, 0, GetTid(), ROCPROFILER_PHASE_EXIT},
              args_data->session_id);
        }
        break;
      }
      default:
        warning("Domain(%u) is not supported for Synchronous callbacks!", domain);
    }
  }
}

void Tracer::InitRoctracer(
    const std::map<rocprofiler_tracer_activity_domain_t, is_filtered_domain_t>& domains,
    const std::vector<std::string>& api_filter_data_vector) {
  for (auto domain : domains) {
    switch (domain.first) {
      case ACTIVITY_DOMAIN_ROCTX: {
        assert(!domain.second && "Error: ROCTX API can't be filtered!");
        if (callback_data_.user_sync_callback)
          roctracer_enable_domain_callback(ACTIVITY_DOMAIN_ROCTX, api_callback, &callback_data_);
        else
          roctracer_enable_domain_activity(ACTIVITY_DOMAIN_ROCTX,
                                           session_buffer_id_t{session_id_, buffer_id_});
        break;
      }
      case ACTIVITY_DOMAIN_HSA_API: {
        if (!domain.second) {
          if (callback_data_.user_sync_callback)
            roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HSA_API, api_callback,
                                             &callback_data_);
          else
            roctracer_enable_domain_activity(ACTIVITY_DOMAIN_HSA_API,
                                             session_buffer_id_t{session_id_, buffer_id_});
        } else {
          assert(!api_filter_data_vector.empty() &&
                 "Error: HSA API calls filter data is empty and domain "
                 "filter option was enabled!");
        }
        break;
      }
      case ACTIVITY_DOMAIN_HIP_API: {
        if (!domain.second) {
          if (callback_data_.user_sync_callback)
            roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HIP_API, api_callback,
                                             &callback_data_);
          else
            roctracer_enable_domain_activity(ACTIVITY_DOMAIN_HIP_API,
                                             session_buffer_id_t{session_id_, buffer_id_});
        } else {
          assert(!api_filter_data_vector.empty() &&
                 "Error: HIP API calls filter data is empty and domain "
                 "filter option was enabled!");
        }
        break;
      }
      case ACTIVITY_DOMAIN_HSA_OPS: {
        // assert(!domain.second && "Error: HSA OPS can't be filtered!");
        // Tracer_enable_domain_activity(ACTIVITY_DOMAIN_HSA_OPS, pool);
        // TODO(aelwazir): to be replaced with the above lines after the
        // whole integeration is done, make sure that tracer is responsible
        // of kernel dispatches alongside mem copies, profiler will be only
        // responsible for counter collection
        roctracer_enable_op_activity(ACTIVITY_DOMAIN_HSA_OPS, HSA_OP_ID_COPY,
                                     session_buffer_id_t{session_id_, buffer_id_});
        break;
      }
      case ACTIVITY_DOMAIN_HIP_OPS: {
        assert(!domain.second && "Error: HIP OPS can't be filtered!");
        roctracer_enable_domain_activity(ACTIVITY_DOMAIN_HIP_OPS,
                                         session_buffer_id_t{session_id_, buffer_id_});
        break;
      }
      // TODO(aelwazir): Make sure if any other domain is needed by the
      // API(User Usage)
      default: {
        fatal("Error: Provided Domain is not supported!");
      }
    }
  }
}

}  // namespace tracer
}  // namespace rocprofiler
