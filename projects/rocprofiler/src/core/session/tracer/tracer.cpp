#include "tracer.h"

#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <cassert>
#include <cstddef>
#include <map>
#include <mutex>
#include <utility>

#include "src/api/rocmtool.h"
#include "src/utils/helper.h"
#include "src/core/hsa/hsa_support.h"
#include "src/core/memory/generic_buffer.h"

namespace rocmtools {
namespace tracer {

std::mutex stream_ids_map_lock;
std::map<uint64_t, std::pair<uint64_t, uint64_t>> stream_ids;
std::map<uint64_t, uint64_t> used_stream_ids;
std::atomic<uint64_t> stream_count{1};

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
  assert(!is_active_.load(std::memory_order_release) && "Error: The tracer was initialized!");
  std::lock_guard<std::mutex> lock(tracer_lock_);

  callback_data_ = api_callback_data_t{callback, session_id};

  is_active_.exchange(true, std::memory_order_release);
}

void Tracer::StartRoctracer() {
  if (!roctracer_initiated_.load(std::memory_order_release)) {
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
  if (roctracer_initiated_.load(std::memory_order_release)) roctracer_stop();
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
  assert(is_active_.load(std::memory_order_release) && "Error: The tracer was not initialized!");
  std::lock_guard<std::mutex> lock(tracer_lock_);

  is_active_.exchange(false, std::memory_order_release);
  // tracer_lock_.unlock();
}

std::mutex& Tracer::GetTracerLock() { return tracer_lock_; }

// TODO(aelwazir): To be implemented from here
bool Tracer::FindROCTxApiData(rocprofiler_tracer_api_data_handle_t api_data_handler) {
  // std::lock_guard<std::mutex> lock(tracer_lock_);
  return true;
}
bool Tracer::FindHSAApiData(rocprofiler_tracer_api_data_handle_t api_data_handler) {
  // std::lock_guard<std::mutex> lock(tracer_lock_);
  return true;
}
bool Tracer::FindHIPApiData(rocprofiler_tracer_api_data_handle_t api_data_handler) {
  // std::lock_guard<std::mutex> lock(tracer_lock_);
  return true;
}

size_t Tracer::GetROCTxApiDataInfoSize(rocprofiler_tracer_roctx_api_data_info_t kind,
                                       rocprofiler_tracer_api_data_handle_t api_data_id,
                                       rocprofiler_tracer_operation_id_t operation_id) {
  const roctx_api_data_t* roctx_data =
      reinterpret_cast<const roctx_api_data_t*>(api_data_id.handle);
  switch (kind) {
    case ROCPROFILER_ROCTX_MESSAGE: {
      if (roctx_data && roctx_data->args.message)
        return strlen(reinterpret_cast<const roctx_api_data_t*>(api_data_id.handle)->args.message) +
            1;
      else
        return 0;
    }
    case ROCPROFILER_ROCTX_ID: {
      if (roctx_data && roctx_data->args.id >= 0)
        return std::to_string(roctx_data->args.id).size() + 1;
      else
        return 0;
    }
    default:
      warning("ROCTX API Data Not Supported!");
  }
  return 0;
}
size_t Tracer::GetHSAApiDataInfoSize(rocprofiler_tracer_hsa_api_data_info_t kind,
                                     rocprofiler_tracer_api_data_handle_t api_data_id,
                                     rocprofiler_tracer_operation_id_t operation_id) {
  switch (kind) {
    case ROCPROFILER_HSA_FUNCTION_NAME: {
      return strlen(roctracer_op_string(ACTIVITY_DOMAIN_HSA_API, operation_id.id)) + 1;
    }
    case ROCPROFILER_HSA_ACTIVITY_NAME: {
      return strlen(roctracer_op_string(ACTIVITY_DOMAIN_HSA_OPS, operation_id.id)) + 1;
    }
    case ROCPROFILER_HSA_API_DATA: {
      return api_data_id.size;
    }
    default:
      warning("HSA API Data Not Supported!");
  }
  return 0;
}
size_t Tracer::GetHIPApiDataInfoSize(rocprofiler_tracer_hip_api_data_info_t kind,
                                     rocprofiler_tracer_api_data_handle_t api_data_id,
                                     rocprofiler_tracer_operation_id_t operation_id) {
  switch (kind) {
    case ROCPROFILER_HIP_KERNEL_NAME: {
      hip_api_data_t* hip_data =
          const_cast<hip_api_data_t*>(reinterpret_cast<const hip_api_data_t*>(api_data_id.handle));
      if (api_data_id.handle && hip_data) {
        auto kernel_name = GetHipKernelName(operation_id.id, hip_data);
        if (kernel_name) return kernel_name->size() + 1;
      }
      return 0;
    }
    case ROCPROFILER_HIP_FUNCTION_NAME: {
      return strlen(roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, operation_id.id)) + 1;
    }
    case ROCPROFILER_HIP_ACTIVITY_NAME: {
      return strlen(roctracer_op_string(ACTIVITY_DOMAIN_HIP_OPS, operation_id.id)) + 1;
    }
    case ROCPROFILER_HIP_STREAM_ID: {
      std::lock_guard<std::mutex> lock(stream_ids_map_lock);
      if (!stream_ids.empty() && stream_ids.find(operation_id.id) != stream_ids.end())
        return std::to_string(stream_ids.at(operation_id.id).second).size() + 1;
      else
        return 0;
    }
    case ROCPROFILER_HIP_API_DATA: {
      return api_data_id.size;
    }
    default:
      warning("HIP API Data Not Supported!");
  }
  return 0;
}

char* Tracer::GetROCTxApiDataInfo(rocprofiler_tracer_roctx_api_data_info_t kind,
                                  rocprofiler_tracer_api_data_handle_t api_data_id,
                                  rocprofiler_tracer_operation_id_t operation_id) {
  switch (kind) {
    case ROCPROFILER_ROCTX_MESSAGE: {
      return const_cast<char*>(
          reinterpret_cast<const roctx_api_data_t*>(api_data_id.handle)->args.message);
    }
    case ROCPROFILER_ROCTX_ID: {
      const roctx_api_data_t* roctx_data =
          reinterpret_cast<const roctx_api_data_t*>(api_data_id.handle);
      if (roctx_data && roctx_data->args.id >= 0)
        return strdup(std::to_string(roctx_data->args.id).c_str());
      else
        return nullptr;
    }
    default:
      warning("HSA API Data Not Supported!");
  }
  return nullptr;
}
char* Tracer::GetHSAApiDataInfo(rocprofiler_tracer_hsa_api_data_info_t kind,
                                rocprofiler_tracer_api_data_handle_t api_data_id,
                                rocprofiler_tracer_operation_id_t operation_id) {
  switch (kind) {
    case ROCPROFILER_HSA_FUNCTION_NAME: {
      return const_cast<char*>(roctracer_op_string(ACTIVITY_DOMAIN_HSA_API, operation_id.id));
    }
    case ROCPROFILER_HSA_ACTIVITY_NAME: {
      return const_cast<char*>(roctracer_op_string(ACTIVITY_DOMAIN_HSA_OPS, operation_id.id));
    }
    case ROCPROFILER_HSA_API_DATA: {
      return const_cast<char*>(reinterpret_cast<const char*>(api_data_id.handle));
    }
    default:
      warning("HSA API Data Not Supported!");
  }
  return nullptr;
}
char* Tracer::GetHIPApiDataInfo(rocprofiler_tracer_hip_api_data_info_t kind,
                                rocprofiler_tracer_api_data_handle_t api_data_id,
                                rocprofiler_tracer_operation_id_t operation_id) {
  switch (kind) {
    case ROCPROFILER_HIP_KERNEL_NAME: {
      std::optional<std::string> kernel_name = GetHipKernelName(
          operation_id.id,
          const_cast<hip_api_data_t*>(reinterpret_cast<const hip_api_data_t*>(api_data_id.handle)));

      if (kernel_name && kernel_name->find(" ") == std::string::npos) {
        return strdup(kernel_name->c_str());
      }
      return nullptr;
    }
    case ROCPROFILER_HIP_FUNCTION_NAME: {
      return const_cast<char*>(roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, operation_id.id));
    }
    case ROCPROFILER_HIP_ACTIVITY_NAME: {
      return const_cast<char*>(roctracer_op_string(ACTIVITY_DOMAIN_HIP_OPS, operation_id.id));
    }
    case ROCPROFILER_HIP_STREAM_ID: {
      std::lock_guard<std::mutex> lock(stream_ids_map_lock);
      if (!stream_ids.empty() && stream_ids.find(operation_id.id) != stream_ids.end())
        return strdup(
            const_cast<char*>(std::to_string(stream_ids.at(operation_id.id).second).c_str()));
      else
        return nullptr;
    }
    case ROCPROFILER_HIP_API_DATA: {
      return const_cast<char*>(reinterpret_cast<const char*>(api_data_id.handle));
    }
    default:
      warning("HIP API Data Not Supported!");
  }
  return nullptr;
}

// TODO(aelwazir): Till here

void api_callback(activity_domain_t domain, uint32_t cid, const void* callback_data, void* args) {
  api_callback_data_t* args_data = reinterpret_cast<api_callback_data_t*>(args);
  if (args_data && rocmtools::GetROCMToolObj() &&
      rocmtools::GetROCMToolObj()->GetSession(args_data->session_id) &&
      rocmtools::GetROCMToolObj()->GetSession(args_data->session_id)->GetTracer()) {
    switch (domain) {
      case ACTIVITY_DOMAIN_ROCTX: {
        const roctx_api_data_t* data = reinterpret_cast<const roctx_api_data_t*>(callback_data);
        // if (data->args.message) roctx_labels.emplace(data->args.id, data->args.message);
        args_data->user_sync_callback(
            rocprofiler_record_tracer_t{
                rocprofiler_record_header_t{
                    ROCPROFILER_TRACER_RECORD,
                    rocprofiler_record_id_t{rocmtools::GetROCMToolObj()->GetUniqueRecordId()}},
                rocprofiler_tracer_external_id_t{0}, ACTIVITY_DOMAIN_ROCTX,
                rocprofiler_tracer_operation_id_t{cid},
                rocprofiler_tracer_api_data_handle_t{callback_data, sizeof(*data)},
                rocprofiler_tracer_activity_correlation_id_t{0},
                rocprofiler_record_header_timestamp_t{roctracer::hsa_support::timestamp_ns(),
                                                    rocprofiler_timestamp_t{0}},
                0, 0, GetTid()},
            args_data->session_id);
        break;
      }
      case ACTIVITY_DOMAIN_HSA_API: {
        hsa_api_data_t* data =
            const_cast<hsa_api_data_t*>(reinterpret_cast<const hsa_api_data_t*>(callback_data));
        if (data->phase == ACTIVITY_API_PHASE_ENTER) {
          *(data->phase_data) = roctracer::hsa_support::timestamp_ns().value;
        } else {
          args_data->user_sync_callback(
              rocprofiler_record_tracer_t{
                  rocprofiler_record_header_t{
                      ROCPROFILER_TRACER_RECORD,
                      rocprofiler_record_id_t{rocmtools::GetROCMToolObj()->GetUniqueRecordId()}},
                  rocprofiler_tracer_external_id_t{0}, ACTIVITY_DOMAIN_HSA_API,
                  rocprofiler_tracer_operation_id_t{cid},
                  rocprofiler_tracer_api_data_handle_t{callback_data, sizeof(*data)},
                  rocprofiler_tracer_activity_correlation_id_t{data->correlation_id},
                  rocprofiler_record_header_timestamp_t{rocprofiler_timestamp_t{*(data->phase_data)},
                                                      roctracer::hsa_support::timestamp_ns()},
                  0, 0, GetTid()},
              args_data->session_id);
        }
        break;
      }
      case ACTIVITY_DOMAIN_HIP_API: {
        hip_api_data_t* data =
            const_cast<hip_api_data_t*>(reinterpret_cast<const hip_api_data_t*>(callback_data));
        if (data->phase == ACTIVITY_API_PHASE_ENTER) {
          *(data->phase_data) = roctracer::hsa_support::timestamp_ns().value;
        } else {
          hipApiArgsInit((hip_api_id_t)cid, data);
          std::string hip_api_data_string = hipApiString((hip_api_id_t)cid, data);
          std::string start_str = "stream=";
          int start = hip_api_data_string.find(start_str);
          uint64_t stream_id = 0;
          if (start >= 0) {
            int end = hip_api_data_string.find(",", start);
            std::string stream_id_str = hip_api_data_string.substr(start + start_str.length(), end);
            std::stringstream ss;
            ss << std::hex << stream_id_str;
            ss >> stream_id;
          }
          {
            std::lock_guard<std::mutex> lock(stream_ids_map_lock);
            if (used_stream_ids.find(stream_id) == used_stream_ids.end()) {
              uint64_t stream_generated_id = stream_count.fetch_add(1, std::memory_order_release);
              used_stream_ids.emplace(stream_id, stream_generated_id);
              stream_ids.emplace(data->correlation_id,
                                 std::make_pair(stream_id, stream_generated_id));
            } else {
              stream_ids.emplace(data->correlation_id,
                                 std::make_pair(stream_id, used_stream_ids.at(stream_id)));
            }
          }
          args_data->user_sync_callback(
              rocprofiler_record_tracer_t{
                  rocprofiler_record_header_t{
                      ROCPROFILER_TRACER_RECORD,
                      rocprofiler_record_id_t{rocmtools::GetROCMToolObj()->GetUniqueRecordId()}},
                  rocprofiler_tracer_external_id_t{0}, ACTIVITY_DOMAIN_HIP_API,
                  rocprofiler_tracer_operation_id_t{cid},
                  rocprofiler_tracer_api_data_handle_t{callback_data, sizeof(*data)},
                  rocprofiler_tracer_activity_correlation_id_t{data->correlation_id},
                  rocprofiler_record_header_timestamp_t{rocprofiler_timestamp_t{*(data->phase_data)},
                                                      roctracer::hsa_support::timestamp_ns()},
                  0, 0, GetTid()},
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
        roctracer_enable_domain_callback(ACTIVITY_DOMAIN_ROCTX, api_callback, &callback_data_);
        break;
      }
      case ACTIVITY_DOMAIN_HSA_API: {
        if (!domain.second) {
          roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HSA_API, api_callback, &callback_data_);
        } else {
          assert(!api_filter_data_vector.empty() &&
                 "Error: HSA API calls filter data is empty and domain "
                 "filter option was enabled!");
        }
        break;
      }
      case ACTIVITY_DOMAIN_HIP_API: {
        if (!domain.second) {
          roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HIP_API, api_callback, &callback_data_);
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
}  // namespace rocmtools
