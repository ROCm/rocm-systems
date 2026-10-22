/*
Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.

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
#include "rdc/rdc_api_service.h"

#include <assert.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/impl/call_op_set.h>

#include <algorithm>
#include <chrono>  // NOLINT
#include <csignal>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

#include "rdc.grpc.pb.h"  // NOLINT
#include "rdc.pb.h"
#include "rdc/rdc.h"
#include "rdc/rdc_private.h"
#include "rdc/rdc_server_main.h"
#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/rdc_common.h"

namespace amd {
namespace rdc {

namespace {

// A char array from librdc as a string: up to its first NUL, or all of it.
template <size_t N>
std::string array_string(const char (&chars)[N]) {
  return std::string(chars, strnlen(chars, N));
}

// A count from librdc, at most the length of the array it counts.
template <typename T, size_t N>
uint32_t clamp_count(uint64_t count, const T (&)[N]) {
  return static_cast<uint32_t>(std::min<uint64_t>(count, N));
}

// The status of a request whose job ID does not fit the C API (char[64]) or
// has control characters, which rdcd's log would carry; else RDC_ST_OK.
rdc_status_t check_job_id(const std::string& job_id) {
  return rdc_valid_job_id(job_id.data(), job_id.size()) ? RDC_ST_OK : RDC_ST_BAD_PARAMETER;
}

}  // namespace

std::mutex RdcAPIServiceImpl::policy_mutex_;
std::map<uint32_t, std::vector<std::shared_ptr<RdcAPIServiceImpl::PolicyStream>>>
    RdcAPIServiceImpl::policy_streams_;
size_t RdcAPIServiceImpl::policy_stream_count_ = 0;
bool RdcAPIServiceImpl::policy_shutdown_ = false;

RdcAPIServiceImpl::RdcAPIServiceImpl() : rdc_handle_(nullptr) {}

RdcAPIServiceImpl::DiagnosticSlot::DiagnosticSlot(std::atomic<uint32_t>* runs) : runs_(runs) {
  uint32_t held = runs_->load();
  while (held < kMaxDiagnosticRuns && !runs_->compare_exchange_weak(held, held + 1)) {
  }
  held_ = held < kMaxDiagnosticRuns;
}

RdcAPIServiceImpl::DiagnosticSlot::~DiagnosticSlot() {
  if (held_) {
    runs_->fetch_sub(1);
  }
}

// Ends a diagnostic run that found no free slot, before any GPU work. Logged at debug level:
// any client can repeat it.
template <typename Response>
::grpc::Status RdcAPIServiceImpl::reject_diagnostic(::grpc::ServerWriter<Response>* writer,
                                                    const char* rpc) {
  RDC_LOG(RDC_DEBUG, rpc << " rejected: rdcd runs " << kMaxDiagnosticRuns << " already");
  Response reply;
  reply.set_status(RDC_ST_MAX_LIMIT);
  writer->Write(reply);
  return ::grpc::Status::OK;
}

rdc_status_t RdcAPIServiceImpl::Initialize(uint64_t rdcd_init_flags) {
  rdc_status_t result = rdc_init(rdcd_init_flags);
  if (result != RDC_ST_OK) {
    std::cout << "RDC API initialize fail\n";
    return result;
  }

  result = rdc_start_embedded(RDC_OPERATION_MODE_AUTO, &rdc_handle_);
  if (result != RDC_ST_OK) {
    std::cout << "RDC API start embedded fail\n";
  }

  return result;
}

void RdcAPIServiceImpl::Shutdown() {
  // End every policy stream, and admit no new one.
  std::lock_guard<std::mutex> guard(policy_mutex_);
  policy_shutdown_ = true;
  for (auto& group : policy_streams_) {
    for (auto& stream : group.second) {
      {
        std::lock_guard<std::mutex> lock(stream->mutex);
        stream->shutdown = true;
      }
      stop_policy_stream(stream.get());
    }
  }
}

// Ends a RegisterPolicy stream now, not at its next 1 s wakeup.
void RdcAPIServiceImpl::stop_policy_stream(PolicyStream* stream) {
  std::lock_guard<std::mutex> lock(stream->mutex);
  stream->stop = true;
  stream->cv.notify_all();
}

RdcAPIServiceImpl::~RdcAPIServiceImpl() {
  if (rdc_handle_) {
    rdc_stop_embedded(rdc_handle_);
    rdc_handle_ = nullptr;
  }
  rdc_shutdown();
}

::grpc::Status RdcAPIServiceImpl::GetAllDevices(::grpc::ServerContext* context,
                                                const ::rdc::Empty* request,
                                                ::rdc::GetAllDevicesResponse* reply) {
  (void)(context);
  (void)(request);
  if (!reply) {
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty reply");
  }
  uint32_t gpu_index_list[RDC_MAX_NUM_DEVICES] = {};
  uint32_t count = 0;
  rdc_status_t result = rdc_device_get_all(rdc_handle_, gpu_index_list, &count);
  reply->set_status(result);
  if (result != RDC_ST_OK) {
    return ::grpc::Status::OK;
  }
  count = clamp_count(count, gpu_index_list);
  for (uint32_t i = 0; i < count; i++) {
    reply->add_gpus(gpu_index_list[i]);
  }

  return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::GetDeviceAttributes(
    ::grpc::ServerContext* context, const ::rdc::GetDeviceAttributesRequest* request,
    ::rdc::GetDeviceAttributesResponse* reply) {
  (void)(context);
  if (!reply || !request) {
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
  }
  uint32_t gpu_index = request->gpu_index();
  rdc_device_attributes_t attribute = {};
  rdc_status_t result = rdc_device_get_attributes(rdc_handle_, gpu_index, &attribute);

  ::rdc::DeviceAttributes* attr = reply->mutable_attributes();
  attr->set_device_name(array_string(attribute.device_name));

  reply->set_status(result);

  return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::GetComponentVersion(
    ::grpc::ServerContext* context, const ::rdc::GetComponentVersionRequest* request,
    ::rdc::GetComponentVersionResponse* reply) {
  (void)(context);
  if (!reply) {
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty reply");
  }

  rdc_component_t component = static_cast<rdc_component_t>(request->component_index());
  rdc_component_version_t compv = {};
  rdc_status_t result = rdc_device_get_component_version(rdc_handle_, component, &compv);

  reply->set_version(array_string(compv.version));
  reply->set_status(result);
  return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::CreateGpuGroup(::grpc::ServerContext* context,
                                                 const ::rdc::CreateGpuGroupRequest* request,
                                                 ::rdc::CreateGpuGroupResponse* reply) {
  (void)(context);
  if (!reply || !request) {
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
  }

  const std::string& group_name = request->group_name();
  if (!rdc_valid_name(group_name.data(), group_name.size())) {
    reply->set_status(RDC_ST_BAD_PARAMETER);
    return ::grpc::Status::OK;
  }
  rdc_gpu_group_t group_id = 0;
  rdc_status_t result = rdc_group_gpu_create(
      rdc_handle_, static_cast<rdc_group_type_t>(request->type()), group_name.c_str(), &group_id);
  reply->set_status(result);
  if (result != RDC_ST_OK) {
    return ::grpc::Status::OK;
  }

  reply->set_group_id(group_id);

  return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::AddToGpuGroup(::grpc::ServerContext* context,
                                                const ::rdc::AddToGpuGroupRequest* request,
                                                ::rdc::AddToGpuGroupResponse* reply) {
  (void)(context);
  if (!reply || !request) {
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
  }

  rdc_status_t result = rdc_group_gpu_add(rdc_handle_, request->group_id(), request->gpu_index());
  reply->set_status(result);

  return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::GetGpuGroupInfo(::grpc::ServerContext* context,
                                                  const ::rdc::GetGpuGroupInfoRequest* request,
                                                  ::rdc::GetGpuGroupInfoResponse* reply) {
  (void)(context);
  if (!reply || !request) {
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
  }

  rdc_group_info_t group_info = {};
  rdc_status_t result = rdc_group_gpu_get_info(rdc_handle_, request->group_id(), &group_info);
  reply->set_status(result);
  if (result != RDC_ST_OK) {
    return ::grpc::Status::OK;
  }

  reply->set_group_name(array_string(group_info.group_name));
  const uint32_t count = clamp_count(group_info.count, group_info.entity_ids);
  for (uint32_t i = 0; i < count; i++) {
    reply->add_entity_ids(group_info.entity_ids[i]);
  }

  return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::GetGroupAllIds(::grpc::ServerContext* context,
                                                 const ::rdc::Empty* request,
                                                 ::rdc::GetGroupAllIdsResponse* reply) {
  if (!reply || !request || !context) {
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
  }

  rdc_gpu_group_t group_id_list[RDC_MAX_NUM_GROUPS] = {};
  uint32_t count = 0;
  rdc_status_t result = rdc_group_get_all_ids(rdc_handle_, group_id_list, &count);
  reply->set_status(result);
  if (result != RDC_ST_OK) {
    return ::grpc::Status::OK;
  }

  count = clamp_count(count, group_id_list);
  for (uint32_t i = 0; i < count; i++) {
    reply->add_group_ids(group_id_list[i]);
  }

  return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::GetFieldGroupAllIds(::grpc::ServerContext* context,
                                                      const ::rdc::Empty* request,
                                                      ::rdc::GetFieldGroupAllIdsResponse* reply) {
  if (!reply || !request || !context) {
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
  }

  rdc_field_grp_t field_group_id_list[RDC_MAX_NUM_FIELD_GROUPS] = {};
  uint32_t count = 0;
  rdc_status_t result = rdc_group_field_get_all_ids(rdc_handle_, field_group_id_list, &count);
  reply->set_status(result);
  if (result != RDC_ST_OK) {
    return ::grpc::Status::OK;
  }

  count = clamp_count(count, field_group_id_list);
  for (uint32_t i = 0; i < count; i++) {
    reply->add_field_group_ids(field_group_id_list[i]);
  }

  return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::DestroyGpuGroup(::grpc::ServerContext* context,
                                                  const ::rdc::DestroyGpuGroupRequest* request,
                                                  ::rdc::DestroyGpuGroupResponse* reply) {
  (void)(context);
  if (!reply || !request) {
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
  }

  rdc_status_t result = rdc_group_gpu_destroy(rdc_handle_, request->group_id());
  reply->set_status(result);

  return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::CreateFieldGroup(::grpc::ServerContext* context,
                                                   const ::rdc::CreateFieldGroupRequest* request,
                                                   ::rdc::CreateFieldGroupResponse* reply) {
  (void)(context);
  if (!reply || !request) {
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
  }

  // Reject requests exceeding the fixed-size stack array to prevent overflow.
  const int request_count = request->field_ids_size();
  if (request_count < 0 || request_count > RDC_MAX_FIELD_IDS_PER_FIELD_GROUP) {
    reply->set_status(RDC_ST_BAD_PARAMETER);
    return ::grpc::Status(::grpc::StatusCode::INVALID_ARGUMENT,
                          "field_ids count exceeds RDC_MAX_FIELD_IDS_PER_FIELD_GROUP");
  }

  const std::string& field_group_name = request->field_group_name();
  if (!rdc_valid_name(field_group_name.data(), field_group_name.size())) {
    reply->set_status(RDC_ST_BAD_PARAMETER);
    return ::grpc::Status::OK;
  }

  rdc_field_grp_t field_group_id = 0;
  rdc_field_t field_ids[RDC_MAX_FIELD_IDS_PER_FIELD_GROUP] = {};
  for (int i = 0; i < request_count; i++) {
    field_ids[i] = static_cast<rdc_field_t>(request->field_ids(i));
  }
  rdc_status_t result = rdc_group_field_create(rdc_handle_, request_count, &field_ids[0],
                                               field_group_name.c_str(), &field_group_id);
  reply->set_status(result);
  if (result != RDC_ST_OK) {
    return ::grpc::Status::OK;
  }

  reply->set_field_group_id(field_group_id);

  return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::AddFieldToFieldGroup(
    ::grpc::ServerContext* context, const ::rdc::AddFieldToFieldGroupRequest* request,
    ::rdc::AddFieldToFieldGroupResponse* reply) {
  (void)(context);
  if (!reply || !request) {
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
  }

  rdc_status_t result = rdc_group_field_add_field(rdc_handle_, request->field_group_id(),
                                                  static_cast<rdc_field_t>(request->field_id()));
  reply->set_status(result);

  return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::GetFieldGroupInfo(::grpc::ServerContext* context,
                                                    const ::rdc::GetFieldGroupInfoRequest* request,
                                                    ::rdc::GetFieldGroupInfoResponse* reply) {
  (void)(context);
  if (!reply || !request) {
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
  }

  rdc_field_group_info_t field_info = {};
  rdc_status_t result =
      rdc_group_field_get_info(rdc_handle_, request->field_group_id(), &field_info);
  reply->set_status(result);
  if (result != RDC_ST_OK) {
    return ::grpc::Status::OK;
  }

  reply->set_filed_group_name(array_string(field_info.group_name));
  const uint32_t count = clamp_count(field_info.count, field_info.field_ids);
  for (uint32_t i = 0; i < count; i++) {
    reply->add_field_ids(field_info.field_ids[i]);
  }

  return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::DestroyFieldGroup(::grpc::ServerContext* context,
                                                    const ::rdc::DestroyFieldGroupRequest* request,
                                                    ::rdc::DestroyFieldGroupResponse* reply) {
  (void)(context);
  if (!reply || !request) {
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
  }

  rdc_status_t result = rdc_group_field_destroy(rdc_handle_, request->field_group_id());
  reply->set_status(result);

  return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::WatchFields(::grpc::ServerContext* context,
                                              const ::rdc::WatchFieldsRequest* request,
                                              ::rdc::WatchFieldsResponse* reply) {
  (void)(context);
  if (!reply || !request) {
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
  }

  rdc_status_t result =
      rdc_field_watch(rdc_handle_, request->group_id(), request->field_group_id(),
                      request->update_freq(), request->max_keep_age(), request->max_keep_samples());
  reply->set_status(result);

  return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::GetLatestFieldValue(
    ::grpc::ServerContext* context, const ::rdc::GetLatestFieldValueRequest* request,
    ::rdc::GetLatestFieldValueResponse* reply) {
  (void)(context);
  if (!reply || !request) {
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
  }

  rdc_field_value value = {};
  rdc_status_t result = rdc_field_get_latest_value(
      rdc_handle_, request->gpu_index(), static_cast<rdc_field_t>(request->field_id()), &value);
  reply->set_status(result);
  if (result != RDC_ST_OK) {
    return ::grpc::Status::OK;
  }

  reply->set_field_id(value.field_id);
  reply->set_rdc_status(value.status);
  reply->set_ts(value.ts);
  reply->set_type(static_cast<::rdc::GetLatestFieldValueResponse_FieldType>(value.type));
  if (value.type == INTEGER) {
    reply->set_l_int(value.value.l_int);
  } else if (value.type == DOUBLE) {
    reply->set_dbl(value.value.dbl);
  } else if (value.type == STRING || value.type == BLOB) {
    reply->set_str(array_string(value.value.str));
  }

  return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::GetFieldSince(::grpc::ServerContext* context,
                                                const ::rdc::GetFieldSinceRequest* request,
                                                ::rdc::GetFieldSinceResponse* reply) {
  (void)(context);
  if (!reply || !request) {
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
  }

  rdc_field_value value = {};
  uint64_t next_timestamp = 0;
  rdc_status_t result = rdc_field_get_value_since(
      rdc_handle_, request->gpu_index(), static_cast<rdc_field_t>(request->field_id()),
      request->since_time_stamp(), &next_timestamp, &value);
  reply->set_status(result);
  if (result != RDC_ST_OK) {
    return ::grpc::Status::OK;
  }

  reply->set_next_since_time_stamp(next_timestamp);
  reply->set_field_id(value.field_id);
  reply->set_rdc_status(value.status);
  reply->set_ts(value.ts);
  reply->set_type(static_cast<::rdc::GetFieldSinceResponse_FieldType>(value.type));
  if (value.type == INTEGER) {
    reply->set_l_int(value.value.l_int);
  } else if (value.type == DOUBLE) {
    reply->set_dbl(value.value.dbl);
  } else if (value.type == STRING || value.type == BLOB) {
    // Without its trailing spaces: an all-space string becomes empty, as
    // find_last_not_of() then returns npos and npos + 1 is 0.
    std::string val_str = array_string(value.value.str);
    val_str.erase(val_str.find_last_not_of(' ') + 1);
    reply->set_str(val_str);
  }

  return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::UnWatchFields(::grpc::ServerContext* context,
                                                const ::rdc::UnWatchFieldsRequest* request,
                                                ::rdc::UnWatchFieldsResponse* reply) {
  (void)(context);
  if (!reply || !request) {
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
  }

  rdc_status_t result =
      rdc_field_unwatch(rdc_handle_, request->group_id(), request->field_group_id());
  reply->set_status(result);

  return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::UpdateAllFields(::grpc::ServerContext* context,
                                                  const ::rdc::UpdateAllFieldsRequest* request,
                                                  ::rdc::UpdateAllFieldsResponse* reply) {
  (void)(context);
  if (!reply || !request) {
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
  }

  rdc_status_t result = rdc_field_update_all(rdc_handle_, request->wait_for_update());
  reply->set_status(result);

  return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::StartJobStats(::grpc::ServerContext* context,
                                                const ::rdc::StartJobStatsRequest* request,
                                                ::rdc::StartJobStatsResponse* reply) {
  (void)(context);
  if (!reply || !request) {
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
  }
  rdc_status_t result = check_job_id(request->job_id());
  if (result == RDC_ST_OK) {
    result = rdc_job_start_stats(rdc_handle_, request->group_id(), request->job_id().c_str(),
                                 request->update_freq());
  }
  reply->set_status(result);

  return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::GetJobStats(::grpc::ServerContext* context,
                                              const ::rdc::GetJobStatsRequest* request,
                                              ::rdc::GetJobStatsResponse* reply) {
  (void)(context);
  if (!reply || !request) {
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
  }

  rdc_job_info_t job_info = {};
  rdc_status_t result = check_job_id(request->job_id());
  if (result == RDC_ST_OK) {
    result = rdc_job_get_stats(rdc_handle_, request->job_id().c_str(), &job_info);
  }

  reply->set_status(result);
  if (result != RDC_ST_OK) {
    return ::grpc::Status::OK;
  }

  const uint32_t num_gpus = clamp_count(job_info.num_gpus, job_info.gpus);
  const uint32_t num_processes = clamp_count(job_info.num_processes, job_info.processes);
  reply->set_num_gpus(num_gpus);
  ::rdc::GpuUsageInfo* sinfo = reply->mutable_summary();
  copy_gpu_usage_info(job_info.summary, sinfo);

  for (uint32_t i = 0; i < num_gpus; i++) {
    ::rdc::GpuUsageInfo* ginfo = reply->add_gpus();
    copy_gpu_usage_info(job_info.gpus[i], ginfo);
  }

  reply->set_num_processes(num_processes);
  for (uint32_t i = 0; i < num_processes; i++) {
    ::rdc::RdcProcessStatsInfo* pinfo = reply->add_processes();
    pinfo->set_pid(job_info.processes[i].pid);
    pinfo->set_process_name(array_string(job_info.processes[i].process_name));
    pinfo->set_start_time(job_info.processes[i].start_time);
    pinfo->set_stop_time(job_info.processes[i].stop_time);
  }

  return ::grpc::Status::OK;
}

bool RdcAPIServiceImpl::copy_gpu_usage_info(const rdc_gpu_usage_info_t& src,
                                            ::rdc::GpuUsageInfo* target) {
  if (target == nullptr) {
    return false;
  }

  target->set_gpu_id(src.gpu_id);
  target->set_start_time(src.start_time);
  target->set_end_time(src.end_time);
  target->set_energy_consumed(src.energy_consumed);
  target->set_max_gpu_memory_used(src.max_gpu_memory_used);
  target->set_ecc_correct(src.ecc_correct);
  target->set_ecc_uncorrect(src.ecc_uncorrect);

  ::rdc::JobStatsSummary* stats = target->mutable_power_usage();
  stats->set_max_value(src.power_usage.max_value);
  stats->set_min_value(src.power_usage.min_value);
  stats->set_average(src.power_usage.average);
  stats->set_standard_deviation(src.power_usage.standard_deviation);

  stats = target->mutable_gpu_clock();
  stats->set_max_value(src.gpu_clock.max_value);
  stats->set_min_value(src.gpu_clock.min_value);
  stats->set_average(src.gpu_clock.average);
  stats->set_standard_deviation(src.gpu_clock.standard_deviation);

  stats = target->mutable_gpu_utilization();
  stats->set_max_value(src.gpu_utilization.max_value);
  stats->set_min_value(src.gpu_utilization.min_value);
  stats->set_average(src.gpu_utilization.average);
  stats->set_standard_deviation(src.gpu_utilization.standard_deviation);

  stats = target->mutable_memory_utilization();
  stats->set_max_value(src.memory_utilization.max_value);
  stats->set_min_value(src.memory_utilization.min_value);
  stats->set_average(src.memory_utilization.average);
  stats->set_standard_deviation(src.memory_utilization.standard_deviation);

  stats = target->mutable_pcie_tx();
  stats->set_max_value(src.pcie_tx.max_value);
  stats->set_min_value(src.pcie_tx.min_value);
  stats->set_average(src.pcie_tx.average);
  stats->set_standard_deviation(src.pcie_tx.standard_deviation);

  stats = target->mutable_pcie_rx();
  stats->set_max_value(src.pcie_rx.max_value);
  stats->set_min_value(src.pcie_rx.min_value);
  stats->set_average(src.pcie_rx.average);
  stats->set_standard_deviation(src.pcie_rx.standard_deviation);

  stats = target->mutable_pcie_total();
  stats->set_max_value(src.pcie_total.max_value);
  stats->set_min_value(src.pcie_total.min_value);
  stats->set_average(src.pcie_total.average);
  stats->set_standard_deviation(src.pcie_total.standard_deviation);

  stats = target->mutable_memory_clock();
  stats->set_max_value(src.memory_clock.max_value);
  stats->set_min_value(src.memory_clock.min_value);
  stats->set_average(src.memory_clock.average);
  stats->set_standard_deviation(src.memory_clock.standard_deviation);

  stats = target->mutable_gpu_temperature();
  stats->set_max_value(src.gpu_temperature.max_value);
  stats->set_min_value(src.gpu_temperature.min_value);
  stats->set_average(src.gpu_temperature.average);
  stats->set_standard_deviation(src.gpu_temperature.standard_deviation);

  return true;
}

::grpc::Status RdcAPIServiceImpl::StopJobStats(::grpc::ServerContext* context,
                                               const ::rdc::StopJobStatsRequest* request,
                                               ::rdc::StopJobStatsResponse* reply) {
  (void)(context);
  if (!reply || !request) {
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
  }

  rdc_status_t result = check_job_id(request->job_id());
  if (result == RDC_ST_OK) {
    result = rdc_job_stop_stats(rdc_handle_, request->job_id().c_str());
  }
  reply->set_status(result);

  return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::RemoveJob(::grpc::ServerContext* context,
                                            const ::rdc::RemoveJobRequest* request,
                                            ::rdc::RemoveJobResponse* reply) {
  (void)(context);
  if (!reply || !request) {
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
  }
  rdc_status_t result = check_job_id(request->job_id());
  if (result == RDC_ST_OK) {
    result = rdc_job_remove(rdc_handle_, request->job_id().c_str());
  }
  reply->set_status(result);

  return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::RemoveAllJob(::grpc::ServerContext* context,
                                               const ::rdc::Empty* request,
                                               ::rdc::RemoveAllJobResponse* reply) {
  (void)(context);
  if (!reply || !request) {
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
  }
  rdc_status_t result = rdc_job_remove_all(rdc_handle_);
  reply->set_status(result);

  return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::DiagnosticRun(
    ::grpc::ServerContext* context, const ::rdc::DiagnosticRunRequest* request,
    ::grpc::ServerWriter<::rdc::DiagnosticRunResponse>* writer) {
  (void)(context);
  if (!writer || !request) {
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
  }
  DiagnosticSlot slot(&diagnostic_runs_);  // released when the run returns
  if (!slot.held()) {
    return reject_diagnostic(writer, "DiagnosticRun");
  }

  auto cb_lambda = [](void* w, void* m) -> void {
    if (w == nullptr) {
      RDC_LOG(RDC_ERROR, "BAD WRITER");
      return;
    }

    if (m == nullptr) {
      RDC_LOG(RDC_ERROR, "BAD MESSAGE");
      return;
    }

    auto writer = static_cast<::grpc::ServerWriter<::rdc::DiagnosticRunResponse>*>(w);
    char* message = static_cast<char*>(m);

    ::rdc::DiagnosticRunResponse reply;

    reply.set_log(std::string(message));
    if (!writer->Write(reply)) {
      RDC_LOG(RDC_ERROR, "Failed to write log message");
    }
  };

  rdc_callback_t cb = cb_lambda;
  rdc_diag_callback_t callback = {
      cb,
      writer,
  };

  rdc_diag_response_t diag_response = {};
  rdc_status_t result = rdc_diagnostic_run(
      rdc_handle_, request->group_id(), static_cast<rdc_diag_level_t>(request->level()),
      const_cast<char*>(request->config().c_str()), static_cast<size_t>(request->config().length()),
      &diag_response, &callback);

  ::rdc::DiagnosticRunResponse reply;
  reply.set_status(result);
  if (result != RDC_ST_OK) {
    writer->Write(reply);  // the client returns the status
    return ::grpc::Status::OK;
  }

  ::rdc::DiagnosticResponse* to_response = reply.mutable_response();
  if (diag_response.results_count > MAX_TEST_CASES) {
    reply.set_status(RDC_ST_BAD_PARAMETER);
    writer->Write(reply);
    return ::grpc::Status::OK;
  }
  to_response->set_results_count(diag_response.results_count);

  for (uint32_t i = 0; i < diag_response.results_count; i++) {
    const rdc_diag_test_result_t& test_result = diag_response.diag_info[i];
    ::rdc::DiagnosticTestResult* to_diag_info = to_response->add_diag_info();

    to_diag_info->set_status(test_result.status);

    // details
    auto to_details = to_diag_info->mutable_details();
    to_details->set_code(test_result.details.code);
    to_details->set_msg(array_string(test_result.details.msg));

    to_diag_info->set_test_case(
        static_cast<::rdc::DiagnosticTestResult_DiagnosticTestCase>(test_result.test_case));

    // gpu_results
    const uint32_t gpu_result_count =
        clamp_count(test_result.per_gpu_result_count, test_result.gpu_results);
    to_diag_info->set_per_gpu_result_count(gpu_result_count);
    for (uint32_t j = 0; j < gpu_result_count; j++) {
      auto to_result = to_diag_info->add_gpu_results();
      const rdc_diag_per_gpu_result_t& cur_result = test_result.gpu_results[j];
      to_result->set_gpu_index(cur_result.gpu_index);
      auto to_per_detail = to_result->mutable_gpu_result();
      to_per_detail->set_code(cur_result.gpu_result.code);
      to_per_detail->set_msg(array_string(cur_result.gpu_result.msg));
    }
    to_diag_info->set_info(array_string(test_result.info));
  }

  if (!writer->Write(reply)) {
    return ::grpc::Status::CANCELLED;
  }

  return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::DiagnosticTestCaseRun(
    ::grpc::ServerContext* context, const ::rdc::DiagnosticTestCaseRunRequest* request,
    ::grpc::ServerWriter<::rdc::DiagnosticTestCaseRunResponse>* writer) {
  (void)(context);
  if (!writer || !request) {
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
  }
  DiagnosticSlot slot(&diagnostic_runs_);  // released when the run returns
  if (!slot.held()) {
    return reject_diagnostic(writer, "DiagnosticTestCaseRun");
  }

  auto cb_lambda = [](void* w, void* m) -> void {
    if (w == nullptr) {
      RDC_LOG(RDC_ERROR, "BAD WRITER");
      return;
    }

    if (m == nullptr) {
      RDC_LOG(RDC_ERROR, "BAD MESSAGE");
      return;
    }

    auto writer = static_cast<::grpc::ServerWriter<::rdc::DiagnosticTestCaseRunResponse>*>(w);
    char* message = static_cast<char*>(m);

    ::rdc::DiagnosticTestCaseRunResponse reply;

    reply.set_log(std::string(message));
    if (!writer->Write(reply)) {
      RDC_LOG(RDC_ERROR, "Failed to write log message");
    }
  };

  rdc_callback_t cb = cb_lambda;
  rdc_diag_callback_t callback = {
      cb,
      writer,
  };

  rdc_diag_test_result_t test_result = {};
  rdc_status_t result = rdc_test_case_run(
      rdc_handle_, request->group_id(), static_cast<rdc_diag_test_cases_t>(request->test_case()),
      const_cast<char*>(request->config().c_str()), static_cast<size_t>(request->config().length()),
      &test_result, &callback);

  ::rdc::DiagnosticTestCaseRunResponse reply;
  reply.set_status(result);
  if (result != RDC_ST_OK) {
    writer->Write(reply);  // the client returns the status
    return ::grpc::Status::OK;
  }
  ::rdc::DiagnosticTestResult* to_diag_info = reply.mutable_result();
  to_diag_info->set_status(test_result.status);

  // details
  auto to_details = to_diag_info->mutable_details();
  to_details->set_code(test_result.details.code);
  to_details->set_msg(array_string(test_result.details.msg));

  to_diag_info->set_test_case(
      static_cast<::rdc::DiagnosticTestResult_DiagnosticTestCase>(test_result.test_case));

  // gpu_results
  const uint32_t gpu_result_count =
      clamp_count(test_result.per_gpu_result_count, test_result.gpu_results);
  to_diag_info->set_per_gpu_result_count(gpu_result_count);
  for (uint32_t j = 0; j < gpu_result_count; j++) {
    auto to_result = to_diag_info->add_gpu_results();
    const rdc_diag_per_gpu_result_t& cur_result = test_result.gpu_results[j];
    to_result->set_gpu_index(cur_result.gpu_index);
    auto to_per_detail = to_result->mutable_gpu_result();
    to_per_detail->set_code(cur_result.gpu_result.code);
    to_per_detail->set_msg(array_string(cur_result.gpu_result.msg));
  }
  to_diag_info->set_info(array_string(test_result.info));

  if (!writer->Write(reply)) {
    return ::grpc::Status::CANCELLED;
  }

  return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::GetMixedComponentVersion(
    ::grpc::ServerContext* context, const ::rdc::GetMixedComponentVersionRequest* request,
    ::rdc::GetMixedComponentVersionResponse* reply) {
  (void)(context);
  if (!reply) {
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty reply");
  }

  mixed_component_t component = static_cast<mixed_component_t>(request->component_id());

  if (component == RDCD_COMPONENT) {
    std::string version = RDC_SERVER_VERSION_STRING;
    std::string hash;
    std::string rdcdversion;
#ifdef CURRENT_GIT_HASH
    hash = QUOTE(CURRENT_GIT_HASH);
    rdcdversion = version + "+" + hash;
#else
    hash = "";
    rdcdversion = version;
#endif

    reply->set_version(rdcdversion);
    reply->set_status(RDC_ST_OK);
    return ::grpc::Status::OK;
  } else {
    return ::grpc::Status(::grpc::StatusCode::INVALID_ARGUMENT,
                          "The provided request parameters are invalid");
  }
}

::grpc::Status RdcAPIServiceImpl::SetPolicy(::grpc::ServerContext* context,
                                            const ::rdc::SetPolicyRequest* request,
                                            ::rdc::SetPolicyResponse* reply) {
  (void)(context);
  if (!reply || !request) {
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
  }

  rdc_policy_t policy;
  // construct the policy
  ::rdc::Policy p = request->policy();

  ::rdc::PolicyCondition cond = p.condition();
  policy.condition.type = static_cast<rdc_policy_condition_type_t>(cond.type());
  policy.condition.value = cond.value();
  policy.action = static_cast<rdc_policy_action_t>(p.action());

  // call RDC Policy API
  rdc_status_t result = rdc_policy_set(rdc_handle_, request->group_id(), policy);

  // set status
  reply->set_status(result);

  return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::GetPolicy(::grpc::ServerContext* context,
                                            const ::rdc::GetPolicyRequest* request,
                                            ::rdc::GetPolicyResponse* reply) {
  (void)(context);
  if (!reply || !request) {
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
  }

  uint32_t count = 0;
  rdc_policy_t policies[RDC_MAX_POLICY_SETTINGS] = {};

  // call RDC Policy API
  rdc_status_t result = rdc_policy_get(rdc_handle_, request->group_id(), &count, policies);

  // set status
  reply->set_status(result);
  if (result != RDC_ST_OK) {
    return ::grpc::Status::OK;
  }

  ::rdc::PolicyResponse* to_response = reply->mutable_response();
  count = clamp_count(count, policies);
  to_response->set_count(count);
  for (uint32_t i = 0; i < count; i++) {
    const rdc_policy_t& policy_ref = policies[i];
    ::rdc::Policy* policy = to_response->add_policies();
    policy->set_action(static_cast<::rdc::Policy_Action>(policy_ref.action));

    auto to_conditon = policy->mutable_condition();

    to_conditon->set_type(static_cast<::rdc::PolicyCondition_Type>(policy_ref.condition.type));
    to_conditon->set_value(policy_ref.condition.value);
  }

  return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::DeletePolicy(::grpc::ServerContext* context,
                                               const ::rdc::DeletePolicyRequest* request,
                                               ::rdc::DeletePolicyResponse* reply) {
  (void)(context);
  if (!reply || !request) {
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
  }

  // call RDC Policy API
  rdc_status_t result =
      rdc_policy_delete(rdc_handle_, request->group_id(),
                        static_cast<rdc_policy_condition_type_t>(request->condition_type()));

  // set status
  reply->set_status(result);

  return ::grpc::Status::OK;
}

int RdcAPIServiceImpl::PolicyCallback(rdc_policy_callback_response_t* userData) {
  if (userData == nullptr) {
    std::cerr << "The rdc_policy_callback returns null data\n";
    return 1;
  }

  // Every stream of the group gets the notification; a stream keeps only the latest one.
  std::lock_guard<std::mutex> guard(policy_mutex_);
  auto group = policy_streams_.find(userData->group_id);
  if (group == policy_streams_.end()) {
    return 0;
  }
  for (auto& stream : group->second) {
    std::lock_guard<std::mutex> lock(stream->mutex);
    stream->response = *userData;
    stream->pending = true;
    stream->cv.notify_all();
  }
  return 0;
}

void RdcAPIServiceImpl::end_policy_stream(uint32_t group_id,
                                          const std::shared_ptr<PolicyStream>& stream) {
  std::lock_guard<std::mutex> guard(policy_mutex_);
  policy_stream_count_--;
  // UnRegisterPolicy may have removed the group already, and a new stream may have
  // registered it again since: only this stream's own entry is removed.
  auto group = policy_streams_.find(group_id);
  if (group == policy_streams_.end()) {
    return;
  }
  auto& streams = group->second;
  auto it = std::find(streams.begin(), streams.end(), stream);
  if (it == streams.end()) {
    return;
  }
  streams.erase(it);
  if (streams.empty()) {
    rdc_policy_unregister(rdc_handle_, group_id);
    policy_streams_.erase(group);
  }
}

::grpc::Status RdcAPIServiceImpl::RegisterPolicy(
    ::grpc::ServerContext* context, const ::rdc::RegisterPolicyRequest* request,
    ::grpc::ServerWriter<::rdc::RegisterPolicyResponse>* writer) {
  if (!context || !writer || !request) {
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
  }
  const uint32_t group_id = request->group_id();

  // Admission. A rejected stream ends before any message, with a status the client maps to
  // an RDC status; an admitted one gets the admission key in its initial metadata at once.
  auto stream = std::make_shared<PolicyStream>();
  {
    std::lock_guard<std::mutex> guard(policy_mutex_);
    // Rejections a client causes are logged at debug level: any client can repeat them.
    if (policy_shutdown_) {
      RDC_LOG(RDC_DEBUG, "Policy stream for group " << group_id << " rejected: rdcd is stopping");
      return ::grpc::Status(::grpc::StatusCode::UNAVAILABLE, "rdcd is stopping");
    }
    if (policy_stream_count_ >= kMaxPolicyStreams) {
      RDC_LOG(RDC_DEBUG, "Policy stream for group " << group_id << " rejected: rdcd serves "
                                                    << kMaxPolicyStreams << " already");
      return ::grpc::Status(::grpc::StatusCode::RESOURCE_EXHAUSTED,
                            "rdcd serves the most policy registrations it can");
    }
    auto group = policy_streams_.find(group_id);
    if (group == policy_streams_.end()) {
      rdc_status_t result = rdc_policy_register(rdc_handle_, group_id, PolicyCallback);
      if (result == RDC_ST_GROUP_NOT_FOUND) {
        RDC_LOG(RDC_DEBUG, "Policy stream for group " << group_id << " rejected: no such group");
        return ::grpc::Status(::grpc::StatusCode::NOT_FOUND, "GPU group not found");
      }
      if (result != RDC_ST_OK) {
        RDC_LOG(RDC_ERROR, "Policy stream for group "
                               << group_id << " rejected: " << rdc_status_string(result));
        return ::grpc::Status(::grpc::StatusCode::INTERNAL, rdc_status_string(result));
      }
      group = policy_streams_.emplace(group_id, std::vector<std::shared_ptr<PolicyStream>>()).first;
    }
    group->second.push_back(stream);
    policy_stream_count_++;
  }
  RDC_LOG(RDC_DEBUG, "Policy stream for group " << group_id << " admitted");
  context->AddInitialMetadata(kRdcPolicyAdmissionKey, "1");
  writer->SendInitialMetadata();

  // The stream ends when a write fails, the client cancels or goes away (keepalive),
  // UnRegisterPolicy ends the group's streams, or rdcd stops.
  bool shutdown = false;
  while (true) {
    rdc_policy_callback_response_t response;
    bool send = false;
    {
      std::unique_lock<std::mutex> lock(stream->mutex);
      stream->cv.wait_for(lock, std::chrono::seconds(1),
                          [&stream] { return stream->pending || stream->stop; });
      if (stream->stop) {
        shutdown = stream->shutdown;
        break;
      }
      if (stream->pending) {
        response = stream->response;
        stream->pending = false;
        send = true;
      }
    }
    if (context->IsCancelled()) {
      break;
    }
    // Written without a lock: a slow client must not block RDC's policy thread.
    if (send) {
      ::rdc::RegisterPolicyResponse reply;
      reply.set_status(RDC_ST_OK);
      reply.set_version(response.version);
      reply.set_group_id(response.group_id);
      reply.set_value(response.value);

      ::rdc::PolicyCondition* cond = reply.mutable_condition();
      cond->set_type(static_cast<::rdc::PolicyCondition_Type>(response.condition.type));
      cond->set_value(response.condition.value);
      reply.set_gpu_index(response.gpu_index);
      reply.set_reset_triggered(response.reset_triggered != 0);
      if (!writer->Write(reply)) {
        break;
      }
    }
  }

  end_policy_stream(group_id, stream);
  RDC_LOG(RDC_DEBUG, "Policy stream for group " << group_id << " ended");
  if (shutdown) {
    return ::grpc::Status(::grpc::StatusCode::UNAVAILABLE, "rdcd is stopping");
  }
  return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::UnRegisterPolicy(::grpc::ServerContext* context,
                                                   const ::rdc::UnRegisterPolicyRequest* request,
                                                   ::rdc::UnRegisterPolicyResponse* reply) {
  (void)(context);
  if (!reply || !request) {
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
  }

  // Ends every stream of the group, from any client.
  std::lock_guard<std::mutex> guard(policy_mutex_);
  rdc_status_t result = rdc_policy_unregister(rdc_handle_, request->group_id());
  auto group = policy_streams_.find(request->group_id());
  if (group != policy_streams_.end()) {
    for (auto& stream : group->second) {
      stop_policy_stream(stream.get());
    }
    policy_streams_.erase(group);
  }
  reply->set_status(result);
  return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::SetHealth(::grpc::ServerContext* context,
                                            const ::rdc::SetHealthRequest* request,
                                            ::rdc::SetHealthResponse* reply) {
  (void)(context);
  if (!reply || !request) {
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
  }

  rdc_status_t result = rdc_health_set(rdc_handle_, request->group_id(), request->components());

  reply->set_status(result);

  return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::GetHealth(::grpc::ServerContext* context,
                                            const ::rdc::GetHealthRequest* request,
                                            ::rdc::GetHealthResponse* reply) {
  (void)(context);
  if (!reply || !request) {
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
  }

  unsigned int components = 0;
  rdc_status_t result = rdc_health_get(rdc_handle_, request->group_id(), &components);

  reply->set_status(result);
  if (result != RDC_ST_OK) {
    return ::grpc::Status::OK;
  }

  reply->set_components(components);

  return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::CheckHealth(::grpc::ServerContext* context,
                                              const ::rdc::CheckHealthRequest* request,
                                              ::rdc::CheckHealthResponse* reply) {
  (void)(context);
  if (!reply || !request) {
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
  }

  rdc_health_response_t response = {};
  rdc_status_t result = rdc_health_check(rdc_handle_, request->group_id(), &response);

  reply->set_status(result);
  if (result != RDC_ST_OK) {
    return ::grpc::Status::OK;
  }

  ::rdc::HealthResponse* to_response = reply->mutable_response();
  to_response->set_overall_health(response.overall_health);
  const uint32_t incidents_count = clamp_count(response.incidents_count, response.incidents);
  to_response->set_incidents_count(incidents_count);

  for (uint32_t i = 0; i < incidents_count; i++) {
    const rdc_health_incidents_t& incident = response.incidents[i];
    ::rdc::HealthIncidents* to_incidents = to_response->add_incidents();

    to_incidents->set_gpu_index(incident.gpu_index);
    to_incidents->set_component(incident.component);
    to_incidents->set_health(incident.health);

    // error
    auto to_error = to_incidents->mutable_error();
    to_error->set_code(incident.error.code);
    to_error->set_msg(array_string(incident.error.msg));
  }

  return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::ClearHealth(::grpc::ServerContext* context,
                                              const ::rdc::ClearHealthRequest* request,
                                              ::rdc::ClearHealthResponse* reply) {
  (void)(context);
  if (!reply || !request) {
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
  }

  rdc_status_t result = rdc_health_clear(rdc_handle_, request->group_id());

  reply->set_status(result);

  return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::GetTopology(::grpc::ServerContext* context,
                                              const ::rdc::GetTopologyRequest* request,
                                              ::rdc::GetTopologyResponse* reply) {
  (void)(context);
  if (!reply || !request) {
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
  }
  rdc_device_topology_t topology_results = {};
  // call RDC topology API
  rdc_status_t result =
      rdc_device_topology_get(rdc_handle_, request->gpu_index(), &topology_results);
  reply->set_status(result);
  if (result != RDC_ST_OK) {
    return ::grpc::Status::OK;
  }
  ::rdc::Topology* topology = reply->mutable_toppology();
  const uint32_t num_of_gpus =
      clamp_count(topology_results.num_of_gpus, topology_results.link_infos);
  topology->set_num_of_gpus(num_of_gpus);
  topology->set_numa_node(topology_results.numa_node);
  for (uint32_t i = 0; i < num_of_gpus; ++i) {
    ::rdc::TopologyLinkInfo* linkinfos = topology->add_link_infos();
    linkinfos->set_gpu_index(topology_results.link_infos[i].gpu_index);
    linkinfos->set_weight(topology_results.link_infos[i].weight);
    linkinfos->set_min_bandwidth(topology_results.link_infos[i].min_bandwidth);
    linkinfos->set_max_bandwidth(topology_results.link_infos[i].max_bandwidth);
    linkinfos->set_hops(topology_results.link_infos[i].hops);
    linkinfos->set_link_type(
        static_cast<::rdc::TopologyLinkInfo_LinkType>(topology_results.link_infos[i].link_type));
    linkinfos->set_p2p_accessible(topology_results.link_infos[i].is_p2p_accessible);
  }
  return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::SetConfig(::grpc::ServerContext* context,
                                            const ::rdc::SetConfigRequest* request,
                                            ::rdc::SetConfigResponse* reply) {
  (void)(context);
  rdc_config_setting_t setting;
  ::rdc::rdc_config_setting setting_ref = request->setting();
  setting.type = static_cast<rdc_config_type_t>(setting_ref.type());
  setting.target_value = setting_ref.target_value();

  rdc_status_t status = rdc_config_set(rdc_handle_, request->group_id(), setting);
  reply->set_status(static_cast<::uint32_t>(status));
  return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::GetConfig(::grpc::ServerContext* context,
                                            const ::rdc::GetConfigRequest* request,
                                            ::rdc::GetConfigResponse* reply) {
  (void)(context);
  rdc_config_setting_list_t settings = {};

  rdc_status_t status = rdc_config_get(rdc_handle_, request->group_id(), &settings);

  reply->set_status(status);
  for (uint32_t i = 0; i < settings.total_settings && i < RDC_MAX_CONFIG_SETTINGS; ++i) {
    auto result = reply->add_settings();
    result->set_type(static_cast<::rdc::rdc_config_type>(settings.settings[i].type));
    result->set_target_value(settings.settings[i].target_value);
  }

  return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::ClearConfig(::grpc::ServerContext* context,
                                              const ::rdc::ClearConfigRequest* request,
                                              ::rdc::ClearConfigResponse* reply) {
  (void)(context);
  rdc_status_t status = rdc_config_clear(rdc_handle_, request->group_id());
  reply->set_status(status);
  return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::GetLinkStatus(::grpc::ServerContext* context,
                                                const ::rdc::Empty* request,
                                                ::rdc::GetLinkStatusResponse* reply) {
  (void)(context);
  if (!reply || !request) {
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
  }

  rdc_link_status_t link_status_results = {};
  // call RDC link status API
  rdc_status_t result = rdc_link_status_get(rdc_handle_, &link_status_results);
  reply->set_status(result);
  if (result != RDC_ST_OK) {
    return ::grpc::Status::OK;
  }

  ::rdc::LinkStatus* linkstatus = reply->mutable_linkstatus();
  const uint32_t num_of_gpus =
      clamp_count(std::max<int32_t>(link_status_results.num_of_gpus, 0), link_status_results.gpus);
  linkstatus->set_num_of_gpus(num_of_gpus);
  for (uint32_t i = 0; i < num_of_gpus; ++i) {
    ::rdc::GpuLinkStatus* gpulinkstatus = linkstatus->add_gpus();
    gpulinkstatus->set_gpu_index(link_status_results.gpus[i].gpu_index);
    const uint32_t num_of_links = clamp_count(link_status_results.gpus[i].num_of_links,
                                              link_status_results.gpus[i].link_states);
    gpulinkstatus->set_num_of_links(num_of_links);
    gpulinkstatus->set_link_types(
        static_cast<::rdc::GpuLinkStatus_LinkTypes>(link_status_results.gpus[i].link_types));
    for (uint32_t n = 0; n < num_of_links; n++) {
      gpulinkstatus->add_link_states(
          static_cast<::rdc::GpuLinkStatus_LinkState>(link_status_results.gpus[i].link_states[n]));
    }
  }

  return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::GetNumPartition(::grpc::ServerContext* context,
                                                  const ::rdc::GetNumPartitionRequest* request,
                                                  ::rdc::GetNumPartitionResponse* reply) {
  (void)context;
  if (!request || !reply) {
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty request or reply");
  }

  uint32_t gpu_index = request->gpu_index();
  uint16_t num_partition = 0;
  rdc_status_t result = rdc_get_num_partition(rdc_handle_, gpu_index, &num_partition);
  reply->set_status(result);
  if (result == RDC_ST_OK) {
    reply->set_num_partition(num_partition);
  }
  return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::GetInstanceProfile(
    ::grpc::ServerContext* context, const ::rdc::GetInstanceProfileRequest* request,
    ::rdc::GetInstanceProfileResponse* reply) {
  (void)context;
  if (!request || !reply) {
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty request or reply");
  }

  uint32_t entity_index = request->entity_index();
  uint32_t resource_type = request->resource_type();
  rdc_resource_profile_t profile;
  memset(&profile, 0, sizeof(profile));

  // Call the RDC API that (in embedded mode) uses AMD SMI
  rdc_status_t result =
      rdc_instance_profile_get(rdc_handle_, entity_index,
                               static_cast<rdc_instance_resource_type_t>(resource_type), &profile);
  reply->set_status(result);
  if (result == RDC_ST_OK) {
    reply->set_partition_resource(profile.partition_resource);
    reply->set_num_partitions_share_resource(profile.num_partitions_share_resource);
  }
  return ::grpc::Status::OK;
}

}  // namespace rdc
}  // namespace amd
