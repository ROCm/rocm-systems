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

#include <csignal>
#include <future>
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

std::map<uint32_t, RdcAPIServiceImpl::policy_thread_context*> RdcAPIServiceImpl::policy_threads_;

RdcAPIServiceImpl::RdcAPIServiceImpl() : rdc_handle_(nullptr) {}

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
  // exit policy threads
  for (auto it : policy_threads_) {
    policy_thread_context* ctx = it.second;
    ctx->start = false;
  }
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
  uint32_t gpu_index_list[RDC_MAX_NUM_DEVICES];
  uint32_t count = 0;
  rdc_status_t result = rdc_device_get_all(rdc_handle_, gpu_index_list, &count);
  reply->set_status(result);
  if (result != RDC_ST_OK) {
    return ::grpc::Status::OK;
  }
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
  rdc_device_attributes_t attribute;
  rdc_status_t result = rdc_device_get_attributes(rdc_handle_, gpu_index, &attribute);

  ::rdc::DeviceAttributes* attr = reply->mutable_attributes();
  attr->set_device_name(attribute.device_name);

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
  rdc_component_version_t compv;
  rdc_status_t result = rdc_device_get_component_version(rdc_handle_, component, &compv);

  reply->set_version(compv.version);
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

  rdc_gpu_group_t group_id = 0;
  rdc_status_t result =
      rdc_group_gpu_create(rdc_handle_, static_cast<rdc_group_type_t>(request->type()),
                           request->group_name().c_str(), &group_id);
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

  rdc_group_info_t group_info;
  rdc_status_t result = rdc_group_gpu_get_info(rdc_handle_, request->group_id(), &group_info);
  reply->set_status(result);
  if (result != RDC_ST_OK) {
    return ::grpc::Status::OK;
  }

  reply->set_group_name(group_info.group_name);
  for (uint32_t i = 0; i < group_info.count; i++) {
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

  rdc_gpu_group_t group_id_list[RDC_MAX_NUM_GROUPS];
  uint32_t count = 0;
  rdc_status_t result = rdc_group_get_all_ids(rdc_handle_, group_id_list, &count);
  reply->set_status(result);
  if (result != RDC_ST_OK) {
    return ::grpc::Status::OK;
  }

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

  rdc_field_grp_t field_group_id_list[RDC_MAX_NUM_FIELD_GROUPS];
  uint32_t count = 0;
  rdc_status_t result = rdc_group_field_get_all_ids(rdc_handle_, field_group_id_list, &count);
  reply->set_status(result);
  if (result != RDC_ST_OK) {
    return ::grpc::Status::OK;
  }

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

  rdc_field_grp_t field_group_id;
  rdc_field_t field_ids[RDC_MAX_FIELD_IDS_PER_FIELD_GROUP];
  for (int i = 0; i < request->field_ids_size(); i++) {
    field_ids[i] = static_cast<rdc_field_t>(request->field_ids(i));
  }
  rdc_status_t result =
      rdc_group_field_create(rdc_handle_, request->field_ids_size(), &field_ids[0],
                             request->field_group_name().c_str(), &field_group_id);
  reply->set_status(result);
  if (result != RDC_ST_OK) {
    return ::grpc::Status::OK;
  }

  reply->set_field_group_id(field_group_id);

  return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::GetFieldGroupInfo(::grpc::ServerContext* context,
                                                    const ::rdc::GetFieldGroupInfoRequest* request,
                                                    ::rdc::GetFieldGroupInfoResponse* reply) {
  (void)(context);
  if (!reply || !request) {
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
  }

  rdc_field_group_info_t field_info;
  rdc_status_t result =
      rdc_group_field_get_info(rdc_handle_, request->field_group_id(), &field_info);
  reply->set_status(result);
  if (result != RDC_ST_OK) {
    return ::grpc::Status::OK;
  }

  reply->set_filed_group_name(field_info.group_name);
  for (uint32_t i = 0; i < field_info.count; i++) {
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

  rdc_field_value value;
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
    reply->set_str(value.value.str);
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

  rdc_field_value value;
  uint64_t next_timestamp;
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
    std::string val_str(value.value.str);
    size_t endpos = val_str.find_last_not_of(" ");
    val_str[endpos + 1] = '\0';
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
  rdc_status_t result =
      rdc_job_start_stats(rdc_handle_, request->group_id(),
                          const_cast<char*>(request->job_id().c_str()), request->update_freq());
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

  rdc_job_info_t job_info;
  rdc_status_t result =
      rdc_job_get_stats(rdc_handle_, const_cast<char*>(request->job_id().c_str()), &job_info);

  reply->set_status(result);
  if (result != RDC_ST_OK) {
    return ::grpc::Status::OK;
  }

  reply->set_num_gpus(job_info.num_gpus);
  ::rdc::GpuUsageInfo* sinfo = reply->mutable_summary();
  copy_gpu_usage_info(job_info.summary, sinfo);

  for (uint32_t i = 0; i < job_info.num_gpus; i++) {
    ::rdc::GpuUsageInfo* ginfo = reply->add_gpus();
    copy_gpu_usage_info(job_info.gpus[i], ginfo);
  }

  reply->set_num_processes(job_info.num_processes);
  for (uint32_t i = 0; i < job_info.num_processes; i++) {
    ::rdc::RdcProcessStatsInfo* pinfo = reply->add_processes();
    pinfo->set_pid(job_info.processes[i].pid);
    pinfo->set_process_name(job_info.processes[i].process_name);
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

  rdc_status_t result =
      rdc_job_stop_stats(rdc_handle_, const_cast<char*>(request->job_id().c_str()));
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
  rdc_status_t result = rdc_job_remove(rdc_handle_, const_cast<char*>(request->job_id().c_str()));
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

  rdc_diag_response_t diag_response;
  rdc_status_t result = rdc_diagnostic_run(
      rdc_handle_, request->group_id(), static_cast<rdc_diag_level_t>(request->level()),
      const_cast<char*>(request->config().c_str()), static_cast<size_t>(request->config().length()),
      &diag_response, &callback);

  ::rdc::DiagnosticRunResponse reply;
  reply.set_status(result);
  if (result != RDC_ST_OK) {
    return ::grpc::Status::OK;
  }

  ::rdc::DiagnosticResponse* to_response = reply.mutable_response();
  to_response->set_results_count(diag_response.results_count);

  for (uint32_t i = 0; i < diag_response.results_count; i++) {
    const rdc_diag_test_result_t& test_result = diag_response.diag_info[i];
    ::rdc::DiagnosticTestResult* to_diag_info = to_response->add_diag_info();

    to_diag_info->set_status(test_result.status);

    // details
    auto to_details = to_diag_info->mutable_details();
    to_details->set_code(test_result.details.code);
    to_details->set_msg(test_result.details.msg);

    to_diag_info->set_test_case(
        static_cast<::rdc::DiagnosticTestResult_DiagnosticTestCase>(test_result.test_case));
    to_diag_info->set_per_gpu_result_count(test_result.per_gpu_result_count);

    // gpu_results
    for (uint32_t j = 0; j < test_result.per_gpu_result_count; j++) {
      auto to_result = to_diag_info->add_gpu_results();
      const rdc_diag_per_gpu_result_t& cur_result = test_result.gpu_results[j];
      to_result->set_gpu_index(cur_result.gpu_index);
      auto to_per_detail = to_result->mutable_gpu_result();
      to_per_detail->set_code(cur_result.gpu_result.code);
      to_per_detail->set_msg(cur_result.gpu_result.msg);
    }
    to_diag_info->set_info(test_result.info);
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

  rdc_diag_test_result_t test_result;
  rdc_status_t result = rdc_test_case_run(
      rdc_handle_, request->group_id(), static_cast<rdc_diag_test_cases_t>(request->test_case()),
      const_cast<char*>(request->config().c_str()), static_cast<size_t>(request->config().length()),
      &test_result, &callback);

  ::rdc::DiagnosticTestCaseRunResponse reply;
  reply.set_status(result);
  if (result != RDC_ST_OK) {
    return ::grpc::Status::OK;
  }
  ::rdc::DiagnosticTestResult* to_diag_info = reply.mutable_result();
  to_diag_info->set_status(test_result.status);

  // details
  auto to_details = to_diag_info->mutable_details();
  to_details->set_code(test_result.details.code);
  to_details->set_msg(test_result.details.msg);

  to_diag_info->set_test_case(
      static_cast<::rdc::DiagnosticTestResult_DiagnosticTestCase>(test_result.test_case));
  to_diag_info->set_per_gpu_result_count(test_result.per_gpu_result_count);

  // gpu_results
  for (uint32_t j = 0; j < test_result.per_gpu_result_count; j++) {
    auto to_result = to_diag_info->add_gpu_results();
    const rdc_diag_per_gpu_result_t& cur_result = test_result.gpu_results[j];
    to_result->set_gpu_index(cur_result.gpu_index);
    auto to_per_detail = to_result->mutable_gpu_result();
    to_per_detail->set_code(cur_result.gpu_result.code);
    to_per_detail->set_msg(cur_result.gpu_result.msg);
  }
  to_diag_info->set_info(test_result.info);

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
  // constructure the policy
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
  rdc_policy_t policies[RDC_MAX_POLICY_SETTINGS];

  // call RDC Policy API
  rdc_status_t result = rdc_policy_get(rdc_handle_, request->group_id(), &count, policies);

  // set status
  reply->set_status(result);
  if (result != RDC_ST_OK) {
    return ::grpc::Status::OK;
  }

  ::rdc::PolicyResponse* to_response = reply->mutable_response();
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

  auto it = policy_threads_.find(userData->group_id);
  if (it != policy_threads_.end()) {
    policy_thread_context* ctx = it->second;

    pthread_mutex_lock(&ctx->mutex);
    ctx->response = *userData;
    pthread_cond_signal(&ctx->cond);
    pthread_mutex_unlock(&ctx->mutex);
  }

  return 0;
}

::grpc::Status RdcAPIServiceImpl::RegisterPolicy(
    ::grpc::ServerContext* context, const ::rdc::RegisterPolicyRequest* request,
    ::grpc::ServerWriter<::rdc::RegisterPolicyResponse>* writer) {
  (void)(context);
  if (!writer || !request) {
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
  }
  auto it = policy_threads_.find(request->group_id());
  if (it == policy_threads_.end()) {
    policy_thread_context* data = new policy_thread_context;
    data->mutex = PTHREAD_MUTEX_INITIALIZER;
    data->cond = PTHREAD_COND_INITIALIZER;
    data->start = true;
    policy_threads_.insert(std::make_pair(request->group_id(), data));
  }

  auto updater = std::async(std::launch::async, [this, request, writer]() {
    rdc_status_t result = rdc_policy_register(rdc_handle_, request->group_id(), PolicyCallback);
    if (result == RDC_ST_OK) {
      auto it = policy_threads_.find(request->group_id());
      if (it != policy_threads_.end()) {
        policy_thread_context* ctx = it->second;
        ctx->start = true;
        while (ctx->start) {
          struct timespec ts;
          clock_gettime(CLOCK_REALTIME, &ts);

          ts.tv_sec += 1;

          pthread_mutex_lock(&ctx->mutex);
          int rc = pthread_cond_timedwait(&ctx->cond, &ctx->mutex, &ts);  // wait for the callback
          if (rc == ETIMEDOUT) {
            // timeout;
          } else if (rc == 0) {
            // reply
            ::rdc::RegisterPolicyResponse reply;
            reply.set_status(RDC_ST_OK);
            reply.set_version(ctx->response.version);
            reply.set_group_id(ctx->response.group_id);
            reply.set_value(ctx->response.value);

            ::rdc::PolicyCondition* cond = reply.mutable_condition();
            cond->set_type(static_cast<::rdc::PolicyCondition_Type>(ctx->response.condition.type));
            cond->set_value(ctx->response.condition.value);

            writer->Write(reply);
          }

          pthread_mutex_unlock(&ctx->mutex);
        }
      }
    }
  });

  return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::UnRegisterPolicy(::grpc::ServerContext* context,
                                                   const ::rdc::UnRegisterPolicyRequest* request,
                                                   ::rdc::UnRegisterPolicyResponse* reply) {
  (void)(context);
  if (!reply || !request) {
    return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
  }

  rdc_status_t result = rdc_policy_unregister(rdc_handle_, request->group_id());
  if (result == RDC_ST_OK) {
    auto it = policy_threads_.find(request->group_id());
    if (it != policy_threads_.end()) {
      policy_thread_context* ctx = it->second;
      ctx->start = false;
    }
  }
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

  unsigned int components;
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

  rdc_health_response_t response;
  rdc_status_t result = rdc_health_check(rdc_handle_, request->group_id(), &response);

  reply->set_status(result);
  if (result != RDC_ST_OK) {
    return ::grpc::Status::OK;
  }

  ::rdc::HealthResponse* to_response = reply->mutable_response();
  to_response->set_overall_health(response.overall_health);
  to_response->set_incidents_count(response.incidents_count);

  for (uint32_t i = 0; i < response.incidents_count; i++) {
    const rdc_health_incidents_t& incident = response.incidents[i];
    ::rdc::HealthIncidents* to_incidents = to_response->add_incidents();

    to_incidents->set_gpu_index(incident.gpu_index);
    to_incidents->set_component(incident.component);
    to_incidents->set_health(incident.health);

    // error
    auto to_error = to_incidents->mutable_error();
    to_error->set_code(incident.error.code);
    to_error->set_msg(incident.error.msg);
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
  rdc_device_topology_t topology_results;
  // call RDC topology API
  rdc_status_t result =
      rdc_device_topology_get(rdc_handle_, request->gpu_index(), &topology_results);
  reply->set_status(result);
  if (result != RDC_ST_OK) {
    return ::grpc::Status::OK;
  }
  ::rdc::Topology* topology = reply->mutable_toppology();
  topology->set_num_of_gpus(topology_results.num_of_gpus);
  topology->set_numa_node(topology_results.numa_node);
  for (uint32_t i = 0; i < topology_results.num_of_gpus; ++i) {
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
  rdc_config_setting_list_t settings;

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

  rdc_link_status_t link_status_results;
  // call RDC link status API
  rdc_status_t result = rdc_link_status_get(rdc_handle_, &link_status_results);
  reply->set_status(result);
  if (result != RDC_ST_OK) {
    return ::grpc::Status::OK;
  }

  ::rdc::LinkStatus* linkstatus = reply->mutable_linkstatus();
  linkstatus->set_num_of_gpus(link_status_results.num_of_gpus);
  for (int32_t i = 0; i < link_status_results.num_of_gpus; ++i) {
    ::rdc::GpuLinkStatus* gpulinkstatus = linkstatus->add_gpus();
    gpulinkstatus->set_gpu_index(link_status_results.gpus[i].gpu_index);
    gpulinkstatus->set_num_of_links(link_status_results.gpus[i].num_of_links);
    gpulinkstatus->set_link_types(
        static_cast<::rdc::GpuLinkStatus_LinkTypes>(link_status_results.gpus[i].link_types));
    for (uint32_t n = 0; n < link_status_results.gpus[i].num_of_links; n++) {
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
