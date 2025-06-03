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
#include "rdc_lib/impl/RdcStandaloneHandler.h"

#include <grpcpp/grpcpp.h>

#include <future>

#include "rdc.grpc.pb.h"  // NOLINT
#include "rdc.pb.h"
#include "rdc/rdc.h"
#include "rdc_lib/RdcLogger.h"

amd::rdc::RdcHandler* make_handler(const char* ip_and_port, const char* root_ca,
                                   const char* client_cert, const char* client_key) {
  return new amd::rdc::RdcStandaloneHandler(ip_and_port, root_ca, client_cert, client_key);
}

namespace amd {
namespace rdc {

RdcStandaloneHandler::RdcStandaloneHandler(const char* ip_and_port, const char* root_ca,
                                           const char* client_cert, const char* client_key) {
  std::shared_ptr<grpc::ChannelCredentials> cred(nullptr);
  if (root_ca == nullptr || client_cert == nullptr || client_key == nullptr) {
    cred = grpc::InsecureChannelCredentials();
  } else {
    grpc::SslCredentialsOptions sslOpts{};
    sslOpts.pem_root_certs = root_ca;
    sslOpts.pem_private_key = client_key;
    sslOpts.pem_cert_chain = client_cert;
    cred = grpc::SslCredentials(sslOpts);
  }
  stub_ = ::rdc::RdcAPI::NewStub(grpc::CreateChannel(ip_and_port, cred));
}

rdc_status_t RdcStandaloneHandler::error_handle(::grpc::Status status, uint32_t rdc_status) {
  if (!status.ok()) {
    std::cout << status.error_message() << ". Error code:" << status.error_code() << std::endl;
    return RDC_ST_CLIENT_ERROR;
  }

  return static_cast<rdc_status_t>(rdc_status);
}

// JOB RdcAPI
rdc_status_t RdcStandaloneHandler::rdc_job_start_stats(rdc_gpu_group_t groupId,
                                                       const char job_id[64],
                                                       uint64_t update_freq) {
  ::rdc::StartJobStatsRequest request;
  ::rdc::StartJobStatsResponse reply;
  ::grpc::ClientContext context;

  request.set_group_id(groupId);
  request.set_job_id(job_id);
  request.set_update_freq(update_freq);
  ::grpc::Status status = stub_->StartJobStats(&context, request, &reply);
  rdc_status_t err_status = error_handle(status, reply.status());

  return err_status;
}

bool RdcStandaloneHandler::copy_gpu_usage_info(const ::rdc::GpuUsageInfo& src,
                                               rdc_gpu_usage_info_t* target) {
  if (target == nullptr) {
    return false;
  }

  target->gpu_id = src.gpu_id();
  target->start_time = src.start_time();
  target->end_time = src.end_time();
  target->energy_consumed = src.energy_consumed();
  target->max_gpu_memory_used = src.max_gpu_memory_used();
  target->ecc_correct = src.ecc_correct();
  target->ecc_uncorrect = src.ecc_uncorrect();

  const ::rdc::JobStatsSummary& pstats = src.power_usage();
  target->power_usage.max_value = pstats.max_value();
  target->power_usage.min_value = pstats.min_value();
  target->power_usage.average = pstats.average();
  target->power_usage.standard_deviation = pstats.standard_deviation();

  const ::rdc::JobStatsSummary& cstats = src.gpu_clock();
  target->gpu_clock.max_value = cstats.max_value();
  target->gpu_clock.min_value = cstats.min_value();
  target->gpu_clock.average = cstats.average();
  target->gpu_clock.standard_deviation = cstats.standard_deviation();

  const ::rdc::JobStatsSummary& ustats = src.gpu_utilization();
  target->gpu_utilization.max_value = ustats.max_value();
  target->gpu_utilization.min_value = ustats.min_value();
  target->gpu_utilization.average = ustats.average();
  target->gpu_utilization.standard_deviation = ustats.standard_deviation();

  const ::rdc::JobStatsSummary& mstats = src.memory_utilization();
  target->memory_utilization.max_value = mstats.max_value();
  target->memory_utilization.min_value = mstats.min_value();
  target->memory_utilization.average = mstats.average();
  target->memory_utilization.standard_deviation = mstats.standard_deviation();

  const ::rdc::JobStatsSummary& txstats = src.pcie_tx();
  target->pcie_tx.max_value = txstats.max_value();
  target->pcie_tx.min_value = txstats.min_value();
  target->pcie_tx.average = txstats.average();
  target->pcie_tx.standard_deviation = txstats.standard_deviation();

  const ::rdc::JobStatsSummary& rxstats = src.pcie_rx();
  target->pcie_rx.max_value = rxstats.max_value();
  target->pcie_rx.min_value = rxstats.min_value();
  target->pcie_rx.average = rxstats.average();
  target->pcie_rx.standard_deviation = rxstats.standard_deviation();

  const ::rdc::JobStatsSummary& pcietotalstats = src.pcie_total();
  target->pcie_total.max_value = pcietotalstats.max_value();
  target->pcie_total.min_value = pcietotalstats.min_value();
  target->pcie_total.average = pcietotalstats.average();
  target->pcie_total.standard_deviation = pcietotalstats.standard_deviation();

  const ::rdc::JobStatsSummary& mcstats = src.memory_clock();
  target->memory_clock.max_value = mcstats.max_value();
  target->memory_clock.min_value = mcstats.min_value();
  target->memory_clock.average = mcstats.average();
  target->memory_clock.standard_deviation = mcstats.standard_deviation();

  const ::rdc::JobStatsSummary& gtstats = src.gpu_temperature();
  target->gpu_temperature.max_value = gtstats.max_value();
  target->gpu_temperature.min_value = gtstats.min_value();
  target->gpu_temperature.average = gtstats.average();
  target->gpu_temperature.standard_deviation = gtstats.standard_deviation();

  return true;
}
rdc_status_t RdcStandaloneHandler::rdc_job_get_stats(const char job_id[64],
                                                     rdc_job_info_t* p_job_info) {
  if (!p_job_info) {
    return RDC_ST_BAD_PARAMETER;
  }

  ::rdc::GetJobStatsRequest request;
  ::rdc::GetJobStatsResponse reply;
  ::grpc::ClientContext context;

  request.set_job_id(job_id);
  ::grpc::Status status = stub_->GetJobStats(&context, request, &reply);
  rdc_status_t err_status = error_handle(status, reply.status());
  if (err_status != RDC_ST_OK) return err_status;

  p_job_info->num_gpus = reply.num_gpus();
  copy_gpu_usage_info(reply.summary(), &(p_job_info->summary));
  for (int i = 0; i < reply.gpus_size(); i++) {
    copy_gpu_usage_info(reply.gpus(i), &(p_job_info->gpus[i]));
  }

  p_job_info->num_processes = reply.num_processes();
  for (int i = 0; i < reply.num_processes(); i++) {
    const auto& proc_msg = reply.processes(i);
    p_job_info->processes[i].pid = proc_msg.pid();

    strncpy_with_null(p_job_info->processes[i].process_name, proc_msg.process_name().c_str(),
                      MAX_PROCESS_NAME - 1);

    p_job_info->processes[i].start_time = proc_msg.start_time();
    p_job_info->processes[i].stop_time = proc_msg.stop_time();
  }

  return RDC_ST_OK;
}

rdc_status_t RdcStandaloneHandler::rdc_job_stop_stats(const char job_id[64]) {
  ::rdc::StopJobStatsRequest request;
  ::rdc::StopJobStatsResponse reply;
  ::grpc::ClientContext context;

  request.set_job_id(job_id);
  ::grpc::Status status = stub_->StopJobStats(&context, request, &reply);
  rdc_status_t err_status = error_handle(status, reply.status());

  return err_status;
}

rdc_status_t RdcStandaloneHandler::rdc_job_remove(const char job_id[64]) {
  ::rdc::RemoveJobRequest request;
  ::rdc::RemoveJobResponse reply;
  ::grpc::ClientContext context;

  request.set_job_id(job_id);
  ::grpc::Status status = stub_->RemoveJob(&context, request, &reply);
  rdc_status_t err_status = error_handle(status, reply.status());

  return err_status;
}

rdc_status_t RdcStandaloneHandler::rdc_job_remove_all() {
  ::rdc::Empty request;
  ::rdc::RemoveAllJobResponse reply;
  ::grpc::ClientContext context;

  ::grpc::Status status = stub_->RemoveAllJob(&context, request, &reply);
  rdc_status_t err_status = error_handle(status, reply.status());

  return err_status;
}

// Discovery RdcAPI
rdc_status_t RdcStandaloneHandler::rdc_device_get_all(uint32_t gpu_index_list[RDC_MAX_NUM_DEVICES],
                                                      uint32_t* count) {
  if (!count) {
    return RDC_ST_BAD_PARAMETER;
  }
  ::rdc::Empty request;
  ::rdc::GetAllDevicesResponse reply;
  ::grpc::ClientContext context;

  ::grpc::Status status = stub_->GetAllDevices(&context, request, &reply);
  rdc_status_t err_status = error_handle(status, reply.status());
  if (err_status != RDC_ST_OK) return err_status;

  if (reply.gpus_size() > RDC_MAX_NUM_DEVICES) {
    return RDC_ST_BAD_PARAMETER;
  }

  *count = reply.gpus_size();
  for (uint32_t i = 0; i < *count; i++) {
    gpu_index_list[i] = reply.gpus(i);
  }

  return RDC_ST_OK;
}

rdc_status_t RdcStandaloneHandler::rdc_device_get_attributes(uint32_t gpu_index,
                                                             rdc_device_attributes_t* p_rdc_attr) {
  if (!p_rdc_attr) {
    return RDC_ST_BAD_PARAMETER;
  }
  ::rdc::GetDeviceAttributesRequest request;
  ::rdc::GetDeviceAttributesResponse reply;
  ::grpc::ClientContext context;

  request.set_gpu_index(gpu_index);
  ::grpc::Status status = stub_->GetDeviceAttributes(&context, request, &reply);
  rdc_status_t err_status = error_handle(status, reply.status());
  if (err_status != RDC_ST_OK) return err_status;

  strncpy_with_null(p_rdc_attr->device_name, reply.attributes().device_name().c_str(),
                    RDC_MAX_STR_LENGTH);

  return RDC_ST_OK;
}

rdc_status_t RdcStandaloneHandler::rdc_device_get_component_version(
    rdc_component_t component, rdc_component_version_t* p_rdc_compv) {
  if (!p_rdc_compv) {
    return RDC_ST_BAD_PARAMETER;
  }

  ::rdc::GetComponentVersionRequest request;
  ::rdc::GetComponentVersionResponse reply;
  ::grpc::ClientContext context;

  request.set_component_index(component);
  ::grpc::Status status = stub_->GetComponentVersion(&context, request, &reply);
  rdc_status_t err_status = error_handle(status, reply.status());
  if (err_status != RDC_ST_OK) return err_status;

  strncpy_with_null(p_rdc_compv->version, reply.version().c_str(), RDC_MAX_VERSION_STR_LENGTH);
  return RDC_ST_OK;
}

// Group RdcAPI
rdc_status_t RdcStandaloneHandler::rdc_group_gpu_create(rdc_group_type_t type,
                                                        const char* group_name,
                                                        rdc_gpu_group_t* p_rdc_group_id) {
  if (!group_name || !p_rdc_group_id) {
    return RDC_ST_BAD_PARAMETER;
  }

  ::rdc::CreateGpuGroupRequest request;
  ::rdc::CreateGpuGroupResponse reply;
  ::grpc::ClientContext context;

  request.set_type(static_cast<::rdc::CreateGpuGroupRequest_GpuGroupType>(type));
  request.set_group_name(group_name);
  ::grpc::Status status = stub_->CreateGpuGroup(&context, request, &reply);
  rdc_status_t err_status = error_handle(status, reply.status());
  if (err_status != RDC_ST_OK) return err_status;

  *p_rdc_group_id = reply.group_id();

  return RDC_ST_OK;
}

rdc_status_t RdcStandaloneHandler::rdc_group_gpu_add(rdc_gpu_group_t group_id, uint32_t gpu_index) {
  ::rdc::AddToGpuGroupRequest request;
  ::rdc::AddToGpuGroupResponse reply;
  ::grpc::ClientContext context;

  request.set_group_id(group_id);
  request.set_gpu_index(gpu_index);
  ::grpc::Status status = stub_->AddToGpuGroup(&context, request, &reply);
  rdc_status_t err_status = error_handle(status, reply.status());

  return err_status;
}

rdc_status_t RdcStandaloneHandler::rdc_group_field_create(uint32_t num_field_ids,
                                                          rdc_field_t* field_ids,
                                                          const char* field_group_name,
                                                          rdc_field_grp_t* rdc_field_group_id) {
  if (!field_ids || !field_group_name || !rdc_field_group_id) {
    return RDC_ST_BAD_PARAMETER;
  }

  ::rdc::CreateFieldGroupRequest request;
  ::rdc::CreateFieldGroupResponse reply;
  ::grpc::ClientContext context;

  request.set_field_group_name(field_group_name);
  for (uint32_t i = 0; i < num_field_ids; i++) {
    request.add_field_ids(field_ids[i]);
  }

  ::grpc::Status status = stub_->CreateFieldGroup(&context, request, &reply);
  rdc_status_t err_status = error_handle(status, reply.status());
  if (err_status != RDC_ST_OK) return err_status;
  *rdc_field_group_id = reply.field_group_id();

  return RDC_ST_OK;
}

rdc_status_t RdcStandaloneHandler::rdc_group_field_get_info(
    rdc_field_grp_t rdc_field_group_id, rdc_field_group_info_t* field_group_info) {
  if (!field_group_info) {
    return RDC_ST_BAD_PARAMETER;
  }

  ::rdc::GetFieldGroupInfoRequest request;
  ::rdc::GetFieldGroupInfoResponse reply;
  ::grpc::ClientContext context;

  request.set_field_group_id(rdc_field_group_id);
  ::grpc::Status status = stub_->GetFieldGroupInfo(&context, request, &reply);
  rdc_status_t err_status = error_handle(status, reply.status());
  if (err_status != RDC_ST_OK) return err_status;

  if (reply.field_ids_size() > RDC_MAX_FIELD_IDS_PER_FIELD_GROUP) {
    return RDC_ST_MAX_LIMIT;
  }

  field_group_info->count = reply.field_ids_size();
  strncpy_with_null(field_group_info->group_name, reply.filed_group_name().c_str(),
                    RDC_MAX_STR_LENGTH);
  for (int i = 0; i < reply.field_ids_size(); i++) {
    field_group_info->field_ids[i] = static_cast<rdc_field_t>(reply.field_ids(i));
  }

  return RDC_ST_OK;
}

rdc_status_t RdcStandaloneHandler::rdc_group_gpu_get_info(rdc_gpu_group_t p_rdc_group_id,
                                                          rdc_group_info_t* p_rdc_group_info) {
  if (!p_rdc_group_info) {
    return RDC_ST_BAD_PARAMETER;
  }

  ::rdc::GetGpuGroupInfoRequest request;
  ::rdc::GetGpuGroupInfoResponse reply;
  ::grpc::ClientContext context;

  request.set_group_id(p_rdc_group_id);
  ::grpc::Status status = stub_->GetGpuGroupInfo(&context, request, &reply);
  rdc_status_t err_status = error_handle(status, reply.status());
  if (err_status != RDC_ST_OK) return err_status;

  if (reply.entity_ids_size() > RDC_GROUP_MAX_ENTITIES) {
    return RDC_ST_MAX_LIMIT;
  }

  p_rdc_group_info->count = reply.entity_ids_size();
  strncpy_with_null(p_rdc_group_info->group_name, reply.group_name().c_str(), RDC_MAX_STR_LENGTH);
  for (int i = 0; i < reply.entity_ids_size(); i++) {
    p_rdc_group_info->entity_ids[i] = reply.entity_ids(i);
  }

  return RDC_ST_OK;
}

rdc_status_t RdcStandaloneHandler::rdc_group_get_all_ids(rdc_gpu_group_t group_id_list[],
                                                         uint32_t* count) {
  if (!count) {
    return RDC_ST_BAD_PARAMETER;
  }
  ::rdc::Empty request;
  ::rdc::GetGroupAllIdsResponse reply;
  ::grpc::ClientContext context;

  ::grpc::Status status = stub_->GetGroupAllIds(&context, request, &reply);
  rdc_status_t err_status = error_handle(status, reply.status());
  if (err_status != RDC_ST_OK) return err_status;

  *count = reply.group_ids_size();
  if (*count >= RDC_MAX_NUM_GROUPS) {
    return RDC_ST_MAX_LIMIT;
  }
  for (uint32_t i = 0; i < *count; i++) {
    group_id_list[i] = reply.group_ids(i);
  }

  return RDC_ST_OK;
}

rdc_status_t RdcStandaloneHandler::rdc_group_field_get_all_ids(
    rdc_field_grp_t field_group_id_list[], uint32_t* count) {
  if (!count) {
    return RDC_ST_BAD_PARAMETER;
  }

  ::rdc::Empty request;
  ::rdc::GetFieldGroupAllIdsResponse reply;
  ::grpc::ClientContext context;

  ::grpc::Status status = stub_->GetFieldGroupAllIds(&context, request, &reply);
  rdc_status_t err_status = error_handle(status, reply.status());
  if (err_status != RDC_ST_OK) return err_status;

  *count = reply.field_group_ids_size();
  if (*count >= RDC_MAX_NUM_FIELD_GROUPS) {
    return RDC_ST_MAX_LIMIT;
  }
  for (uint32_t i = 0; i < *count; i++) {
    field_group_id_list[i] = reply.field_group_ids(i);
  }

  return RDC_ST_OK;
}

rdc_status_t RdcStandaloneHandler::rdc_group_gpu_destroy(rdc_gpu_group_t p_rdc_group_id) {
  ::rdc::DestroyGpuGroupRequest request;
  ::rdc::DestroyGpuGroupResponse reply;
  ::grpc::ClientContext context;

  request.set_group_id(p_rdc_group_id);
  ::grpc::Status status = stub_->DestroyGpuGroup(&context, request, &reply);
  return error_handle(status, reply.status());
}

rdc_status_t RdcStandaloneHandler::rdc_group_field_destroy(rdc_field_grp_t rdc_field_group_id) {
  ::rdc::DestroyFieldGroupRequest request;
  ::rdc::DestroyFieldGroupResponse reply;
  ::grpc::ClientContext context;

  request.set_field_group_id(rdc_field_group_id);
  ::grpc::Status status = stub_->DestroyFieldGroup(&context, request, &reply);
  return error_handle(status, reply.status());
}

// Field RdcAPI
rdc_status_t RdcStandaloneHandler::rdc_field_watch(rdc_gpu_group_t group_id,
                                                   rdc_field_grp_t field_group_id,
                                                   uint64_t update_freq, double max_keep_age,
                                                   uint32_t max_keep_samples) {
  ::rdc::WatchFieldsRequest request;
  ::rdc::WatchFieldsResponse reply;
  ::grpc::ClientContext context;

  request.set_group_id(group_id);
  request.set_field_group_id(field_group_id);
  request.set_update_freq(update_freq);
  request.set_max_keep_age(max_keep_age);
  request.set_max_keep_samples(max_keep_samples);
  ::grpc::Status status = stub_->WatchFields(&context, request, &reply);

  return error_handle(status, reply.status());
}

rdc_status_t RdcStandaloneHandler::rdc_field_get_latest_value(uint32_t gpu_index, rdc_field_t field,
                                                              rdc_field_value* value) {
  if (!value) {
    return RDC_ST_BAD_PARAMETER;
  }

  ::rdc::GetLatestFieldValueRequest request;
  ::rdc::GetLatestFieldValueResponse reply;
  ::grpc::ClientContext context;

  request.set_gpu_index(gpu_index);
  request.set_field_id(field);
  ::grpc::Status status = stub_->GetLatestFieldValue(&context, request, &reply);
  rdc_status_t err_status = error_handle(status, reply.status());
  if (err_status != RDC_ST_OK) return err_status;

  value->field_id = static_cast<rdc_field_t>(reply.field_id());
  value->status = reply.rdc_status();
  value->ts = reply.ts();
  value->type = static_cast<rdc_field_type_t>(reply.type());
  if (value->type == INTEGER) {
    value->value.l_int = reply.l_int();
  } else if (value->type == DOUBLE) {
    value->value.dbl = reply.dbl();
  } else if (value->type == STRING || value->type == BLOB) {
    strncpy_with_null(value->value.str, reply.str().c_str(), RDC_MAX_STR_LENGTH);
  }

  return RDC_ST_OK;
}

rdc_status_t RdcStandaloneHandler::rdc_field_get_value_since(uint32_t gpu_index, rdc_field_t field,
                                                             uint64_t since_time_stamp,
                                                             uint64_t* next_since_time_stamp,
                                                             rdc_field_value* value) {
  if (!next_since_time_stamp || !value) {
    return RDC_ST_BAD_PARAMETER;
  }

  ::rdc::GetFieldSinceRequest request;
  ::rdc::GetFieldSinceResponse reply;
  ::grpc::ClientContext context;

  request.set_gpu_index(gpu_index);
  request.set_field_id(field);
  request.set_since_time_stamp(since_time_stamp);
  ::grpc::Status status = stub_->GetFieldSince(&context, request, &reply);
  rdc_status_t err_status = error_handle(status, reply.status());
  if (err_status != RDC_ST_OK) return err_status;

  value->field_id = static_cast<rdc_field_t>(reply.field_id());
  value->status = reply.rdc_status();
  value->ts = reply.ts();
  value->type = static_cast<rdc_field_type_t>(reply.type());
  if (value->type == INTEGER) {
    value->value.l_int = reply.l_int();
  } else if (value->type == DOUBLE) {
    value->value.dbl = reply.dbl();
  } else if (value->type == STRING || value->type == BLOB) {
    strncpy_with_null(value->value.str, reply.str().c_str(), RDC_MAX_STR_LENGTH);
  }
  *next_since_time_stamp = reply.next_since_time_stamp();

  return RDC_ST_OK;
}

rdc_status_t RdcStandaloneHandler::rdc_field_unwatch(rdc_gpu_group_t group_id,
                                                     rdc_field_grp_t field_group_id) {
  ::rdc::UnWatchFieldsRequest request;
  ::rdc::UnWatchFieldsResponse reply;
  ::grpc::ClientContext context;

  request.set_group_id(group_id);
  request.set_field_group_id(field_group_id);
  ::grpc::Status status = stub_->UnWatchFields(&context, request, &reply);

  return error_handle(status, reply.status());
}

// Diagnostic API
rdc_status_t RdcStandaloneHandler::rdc_diagnostic_run(rdc_gpu_group_t group_id,
                                                      rdc_diag_level_t level, const char* config,
                                                      size_t config_size,
                                                      rdc_diag_response_t* response,
                                                      rdc_diag_callback_t* /*callback*/) {
  if (!response) {
    return RDC_ST_BAD_PARAMETER;
  }
  ::rdc::DiagnosticRunRequest request;
  ::rdc::DiagnosticRunResponse reply;
  ::grpc::ClientContext context;

  request.set_group_id(group_id);
  request.set_level(level);
  request.set_config(config);
  request.set_config_size(config_size);

  auto reader = stub_->DiagnosticRun(&context, request);
  // for the duration of the DiagnosticRun (multiple tests) - we're stuck in this loop
  //
  // there are 2 optional reply fields:
  // * log - reports messages back during the diagnostic run
  // * response - delivered when the diagnostic run completes
  while (reader->Read(&reply)) {
    if (reply.has_log()) {
      // TODO: Add different logging levels
      std::cout << "LOG: " << reply.log() << std::endl;
      continue;
    }
    if (reply.has_response()) {
      RDC_LOG(RDC_DEBUG, "HAS RESPONSE!");
      auto res = reply.response();
      response->results_count = res.results_count();

      if (res.diag_info_size() > static_cast<int>(MAX_TEST_CASES)) {
        return RDC_ST_BAD_PARAMETER;
      }
      for (int i = 0; i < res.diag_info_size(); i++) {
        const ::rdc::DiagnosticTestResult& result = res.diag_info(i);
        rdc_diag_test_result_t& to_result = response->diag_info[i];
        to_result.status = static_cast<rdc_diag_result_t>(result.status());

        // Set details
        to_result.details.code = result.details().code();
        strncpy_with_null(to_result.details.msg, result.details().msg().c_str(),
                          MAX_DIAG_MSG_LENGTH);

        to_result.test_case = static_cast<rdc_diag_test_cases_t>(result.test_case());
        to_result.per_gpu_result_count = result.per_gpu_result_count();

        // Set Result details
        if (result.gpu_results_size() > RDC_MAX_NUM_DEVICES) {
          return RDC_ST_BAD_PARAMETER;
        }
        for (int j = 0; j < result.gpu_results_size(); j++) {
          auto per_gpu_result = result.gpu_results(j);
          rdc_diag_per_gpu_result_t& to_per_gpu = to_result.gpu_results[j];
          to_per_gpu.gpu_index = per_gpu_result.gpu_index();
          to_per_gpu.gpu_result.code = per_gpu_result.gpu_result().code();
          strncpy_with_null(to_per_gpu.gpu_result.msg, per_gpu_result.gpu_result().msg().c_str(),
                            MAX_DIAG_MSG_LENGTH);
        }
        strncpy_with_null(to_result.info, result.info().c_str(), MAX_DIAG_MSG_LENGTH);
      }
    }
  }

  auto status = reader->Finish();
  if (status.ok()) {
    RDC_LOG(RDC_DEBUG, "reader status: success!");
  } else {
    RDC_LOG(RDC_ERROR, "reader status: failure!");
  }

  return RDC_ST_OK;
}

rdc_status_t RdcStandaloneHandler::rdc_test_case_run(rdc_gpu_group_t group_id,
                                                     rdc_diag_test_cases_t test_case,
                                                     const char* config, size_t config_size,
                                                     rdc_diag_test_result_t* to_result,
                                                     rdc_diag_callback_t* /*callback*/) {
  if (!to_result) {
    return RDC_ST_BAD_PARAMETER;
  }
  ::rdc::DiagnosticTestCaseRunRequest request;
  ::rdc::DiagnosticTestCaseRunResponse reply;
  ::grpc::ClientContext context;

  request.set_group_id(group_id);
  request.set_config(config);
  request.set_config_size(config_size);
  request.set_test_case(static_cast<::rdc::DiagnosticTestCaseRunRequest_TestCaseType>(test_case));

  auto reader = stub_->DiagnosticTestCaseRun(&context, request);
  while (reader->Read(&reply)) {
    if (!reply.has_result()) {
      RDC_LOG(RDC_ERROR, "NO TEST_RUN RESULT!");
      continue;
    }

    auto result = reply.result();

    to_result->status = static_cast<rdc_diag_result_t>(result.status());

    // Set details
    to_result->details.code = result.details().code();
    strncpy_with_null(to_result->details.msg, result.details().msg().c_str(), MAX_DIAG_MSG_LENGTH);

    to_result->test_case = static_cast<rdc_diag_test_cases_t>(result.test_case());
    to_result->per_gpu_result_count = result.per_gpu_result_count();

    // Set Result details
    if (result.gpu_results_size() > RDC_MAX_NUM_DEVICES) {
      return RDC_ST_BAD_PARAMETER;
    }
    for (int j = 0; j < result.gpu_results_size(); j++) {
      auto per_gpu_result = result.gpu_results(j);
      rdc_diag_per_gpu_result_t& to_per_gpu = to_result->gpu_results[j];
      to_per_gpu.gpu_index = per_gpu_result.gpu_index();
      to_per_gpu.gpu_result.code = per_gpu_result.gpu_result().code();
      strncpy_with_null(to_per_gpu.gpu_result.msg, per_gpu_result.gpu_result().msg().c_str(),
                        MAX_DIAG_MSG_LENGTH);
    }
    strncpy_with_null(to_result->info, result.info().c_str(), MAX_DIAG_MSG_LENGTH);
  }

  auto status = reader->Finish();
  if (status.ok()) {
    RDC_LOG(RDC_DEBUG, "reader status: success!");
  } else {
    RDC_LOG(RDC_ERROR, "reader status: failure!");
  }

  return RDC_ST_OK;
}

// Control RdcAPI
rdc_status_t RdcStandaloneHandler::rdc_field_update_all(uint32_t wait_for_update) {
  ::rdc::UpdateAllFieldsRequest request;
  ::rdc::UpdateAllFieldsResponse reply;
  ::grpc::ClientContext context;

  request.set_wait_for_update(wait_for_update);
  ::grpc::Status status = stub_->UpdateAllFields(&context, request, &reply);

  return error_handle(status, reply.status());
}

// Set one configure
rdc_status_t RdcStandaloneHandler::rdc_config_set(rdc_gpu_group_t group_id,
                                                  rdc_config_setting_t setting) {
  ::rdc::SetConfigRequest request;
  ::rdc::SetConfigResponse reply;
  ::grpc::ClientContext context;

  request.set_group_id(group_id);

  ::rdc::rdc_config_setting* setting_ref = (::rdc::rdc_config_setting*)request.mutable_setting();
  setting_ref->set_type(static_cast<::rdc::rdc_config_type>(setting.type));
  setting_ref->set_target_value(setting.target_value);

  ::grpc::Status status = stub_->SetConfig(&context, request, &reply);
  rdc_status_t err_status = error_handle(status, reply.status());

  return err_status;
}

// Get the setting
rdc_status_t RdcStandaloneHandler::rdc_config_get(rdc_gpu_group_t group_id,
                                                  rdc_config_setting_list_t* settings) {
  int i = 0;
  ::rdc::GetConfigRequest request;
  ::rdc::GetConfigResponse reply;
  ::grpc::ClientContext context;

  request.set_group_id(group_id);
  ::grpc::Status status = stub_->GetConfig(&context, request, &reply);
  rdc_status_t err_status = error_handle(status, reply.status());
  if (err_status != RDC_ST_OK) return err_status;

  auto res = reply.settings();
  if (reply.settings_size() > RDC_MAX_CONFIG_SETTINGS) return RDC_ST_MAX_LIMIT;

  for (i = 0; i < reply.settings_size() && i < RDC_MAX_CONFIG_SETTINGS; ++i) {
    const ::rdc::rdc_config_setting& result = reply.settings(i);
    settings->settings[i].type = static_cast<rdc_config_type_t>(result.type());
    settings->settings[i].target_value = result.target_value();
  }

  settings->total_settings = (reply.settings_size() >= RDC_MAX_CONFIG_SETTINGS)
                                 ? RDC_MAX_CONFIG_SETTINGS
                                 : reply.settings_size();
  err_status = error_handle(status, reply.status());
  return err_status;
}

// Clear the setting
rdc_status_t RdcStandaloneHandler::rdc_config_clear(rdc_gpu_group_t group_id) {
  ::rdc::ClearConfigRequest request;
  ::rdc::ClearConfigResponse reply;
  ::grpc::ClientContext context;

  request.set_group_id(group_id);

  ::grpc::Status status = stub_->ClearConfig(&context, request, &reply);

  rdc_status_t err_status = error_handle(status, reply.status());

  return err_status;
}

// It is only an interface for the client under the GRPC framework and is not used as an RDC API.
rdc_status_t RdcStandaloneHandler::get_mixed_component_version(
    mixed_component_t component, mixed_component_version_t* p_mixed_compv) {
  if (!p_mixed_compv) {
    return RDC_ST_BAD_PARAMETER;
  }

  ::rdc::GetMixedComponentVersionRequest request;
  ::rdc::GetMixedComponentVersionResponse reply;
  ::grpc::ClientContext context;

  request.set_component_id(component);
  ::grpc::Status status = stub_->GetMixedComponentVersion(&context, request, &reply);

  rdc_status_t err_status = error_handle(status, reply.status());
  if (err_status != RDC_ST_OK) return err_status;

  strncpy_with_null(p_mixed_compv->version, reply.version().c_str(), USR_MAX_VERSION_STR_LENGTH);
  return RDC_ST_OK;
}

// Policy RdcAPI
rdc_status_t RdcStandaloneHandler::rdc_policy_set(rdc_gpu_group_t group_id, rdc_policy_t policy) {
  ::rdc::SetPolicyRequest request;
  ::rdc::SetPolicyResponse reply;
  ::grpc::ClientContext context;

  request.set_group_id(group_id);
  auto to_policy = request.mutable_policy();
  to_policy->set_action(static_cast<::rdc::Policy_Action>(policy.action));

  auto to_condition = to_policy->mutable_condition();

  to_condition->set_type(static_cast<::rdc::PolicyCondition_Type>(policy.condition.type));
  to_condition->set_value(policy.condition.value);

  // call gRPC
  ::grpc::Status status = stub_->SetPolicy(&context, request, &reply);

  return error_handle(status, reply.status());
}

rdc_status_t RdcStandaloneHandler::rdc_policy_get(rdc_gpu_group_t group_id, uint32_t* count,
                                                  rdc_policy_t policies[RDC_MAX_POLICY_SETTINGS]) {
  ::rdc::GetPolicyRequest request;
  ::rdc::GetPolicyResponse reply;
  ::grpc::ClientContext context;

  if (count == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }

  request.set_group_id(group_id);

  // call gRPC
  ::grpc::Status status = stub_->GetPolicy(&context, request, &reply);
  rdc_status_t err_status = error_handle(status, reply.status());
  if (err_status != RDC_ST_OK) return err_status;

  auto response = reply.response();
  uint32_t policy_count = response.count();

  for (uint32_t i = 0; i < policy_count; ++i) {
    const ::rdc::Policy& policy = response.policies(i);

    ::rdc::PolicyCondition cond = policy.condition();
    policies[i].condition.type = static_cast<rdc_policy_condition_type_t>(cond.type());
    policies[i].condition.value = cond.value();
    policies[i].action = static_cast<rdc_policy_action_t>(policy.action());
  }

  *count = policy_count;

  return RDC_ST_OK;
}

rdc_status_t RdcStandaloneHandler::rdc_policy_delete(rdc_gpu_group_t group_id,
                                                     rdc_policy_condition_type_t condition_type) {
  ::rdc::DeletePolicyRequest request;
  ::rdc::DeletePolicyResponse reply;
  ::grpc::ClientContext context;

  request.set_group_id(group_id);

  request.set_condition_type(
      static_cast<::rdc::DeletePolicyRequest_PolicyConditionType>(condition_type));

  // call gRPC
  ::grpc::Status status = stub_->DeletePolicy(&context, request, &reply);

  return error_handle(status, reply.status());
}

rdc_status_t RdcStandaloneHandler::rdc_policy_register(rdc_gpu_group_t group_id,
                                                       rdc_policy_register_callback callback) {
  // check if a thread for a group is already registered
  auto it = policy_threads_.find(group_id);
  if (it != policy_threads_.end()) {
    return RDC_ST_CONFLICT;
  }

  // no registered callback, start the thread to read the stream from rdcd
  struct policy_thread_context ctx = {true, nullptr};

  ctx.t = new std::thread([this, group_id, callback]() {
    // call rdcd
    ::rdc::RegisterPolicyRequest request;
    ::rdc::RegisterPolicyResponse reply;
    ::grpc::ClientContext context;

    request.set_group_id(group_id);

    // call to gRPC
    std::unique_ptr<grpc::ClientReader<::rdc::RegisterPolicyResponse>> reader(
        stub_->RegisterPolicy(&context, request));

    bool start = true;
    while (start) {
      auto it = policy_threads_.find(group_id);
      if (it != policy_threads_.end()) {
        if (it->second.start == false) start = false;
      } else {
        start = false;
      }

      if (reader->Read(&reply)) {
        reply.status();
        ::rdc::PolicyCondition cond = reply.condition();

        rdc_policy_callback_response_t response;
        response.version = reply.version();
        response.condition.type = static_cast<rdc_policy_condition_type_t>(cond.type());
        response.condition.value = cond.value();
        response.group_id = reply.group_id();
        response.value = reply.value();

        callback(&response);
      }
    }

    reader->Finish();
  });

  policy_threads_.insert(std::make_pair(group_id, ctx));

  return RDC_ST_OK;
}

rdc_status_t RdcStandaloneHandler::rdc_policy_unregister(rdc_gpu_group_t group_id) {
  ::rdc::UnRegisterPolicyRequest request;
  ::rdc::UnRegisterPolicyResponse reply;
  ::grpc::ClientContext context;

  // stop the assocaticted thread of a group
  auto it = policy_threads_.find(group_id);
  if (it != policy_threads_.end()) {
    struct policy_thread_context& ctx = it->second;
    ctx.start = false;
  }

  // construcut the request
  request.set_group_id(group_id);

  // call gRPC
  ::grpc::Status status = stub_->UnRegisterPolicy(&context, request, &reply);
  return error_handle(status, reply.status());
}

// Health RdcAPI
rdc_status_t RdcStandaloneHandler::rdc_health_set(rdc_gpu_group_t group_id,
                                                  unsigned int components) {
  ::rdc::SetHealthRequest request;
  ::rdc::SetHealthResponse reply;
  ::grpc::ClientContext context;

  request.set_group_id(group_id);
  request.set_components(components);
  ::grpc::Status status = stub_->SetHealth(&context, request, &reply);
  rdc_status_t err_status = error_handle(status, reply.status());

  return err_status;
}

rdc_status_t RdcStandaloneHandler::rdc_health_get(rdc_gpu_group_t group_id,
                                                  unsigned int* components) {
  if (!components) {
    return RDC_ST_BAD_PARAMETER;
  }

  ::rdc::GetHealthRequest request;
  ::rdc::GetHealthResponse reply;
  ::grpc::ClientContext context;

  request.set_group_id(group_id);
  ::grpc::Status status = stub_->GetHealth(&context, request, &reply);
  rdc_status_t err_status = error_handle(status, reply.status());
  if (err_status != RDC_ST_OK) return err_status;

  *components = reply.components();
  return RDC_ST_OK;
}

rdc_status_t RdcStandaloneHandler::rdc_health_check(rdc_gpu_group_t group_id,
                                                    rdc_health_response_t* response) {
  if (!response) {
    return RDC_ST_BAD_PARAMETER;
  }

  ::rdc::CheckHealthRequest request;
  ::rdc::CheckHealthResponse reply;
  ::grpc::ClientContext context;

  request.set_group_id(group_id);
  ::grpc::Status status = stub_->CheckHealth(&context, request, &reply);
  rdc_status_t err_status = error_handle(status, reply.status());
  if (err_status != RDC_ST_OK) return err_status;

  auto res = reply.response();
  response->overall_health = static_cast<rdc_health_result_t>(res.overall_health());
  response->incidents_count = res.incidents_count();

  for (int i = 0; i < res.incidents_size(); i++) {
    const ::rdc::HealthIncidents& result = res.incidents(i);
    rdc_health_incidents_t& to_result = response->incidents[i];

    to_result.gpu_index = result.gpu_index();
    to_result.component = static_cast<rdc_health_system_t>(result.component());
    to_result.health = static_cast<rdc_health_result_t>(result.health());

    // set error
    to_result.error.code = result.error().code();
    strncpy_with_null(to_result.error.msg, result.error().msg().c_str(), MAX_HEALTH_MSG_LENGTH);
  }

  return RDC_ST_OK;
}

rdc_status_t RdcStandaloneHandler::rdc_health_clear(rdc_gpu_group_t group_id) {
  ::rdc::ClearHealthRequest request;
  ::rdc::ClearHealthResponse reply;
  ::grpc::ClientContext context;

  request.set_group_id(group_id);
  ::grpc::Status status = stub_->ClearHealth(&context, request, &reply);
  rdc_status_t err_status = error_handle(status, reply.status());
  if (err_status != RDC_ST_OK) return err_status;

  return RDC_ST_OK;
}

rdc_status_t RdcStandaloneHandler::rdc_device_topology_get(uint32_t gpu_index,
                                                           rdc_device_topology_t* results) {
  ::rdc::GetTopologyRequest request;
  ::rdc::GetTopologyResponse reply;
  ::grpc::ClientContext context;

  request.set_gpu_index(gpu_index);
  ::grpc::Status status = stub_->GetTopology(&context, request, &reply);
  rdc_status_t err_status = error_handle(status, reply.status());
  if (err_status != RDC_ST_OK) return err_status;

  ::rdc::Topology Topology = reply.toppology();
  results->num_of_gpus = Topology.num_of_gpus();
  results->numa_node = Topology.numa_node();

  for (uint32_t i = 0; i < Topology.num_of_gpus(); ++i) {
    ::rdc::TopologyLinkInfo linkinfo = Topology.link_infos(i);
    results->link_infos[i].gpu_index = linkinfo.gpu_index();
    results->link_infos[i].weight = linkinfo.weight();
    results->link_infos[i].min_bandwidth = linkinfo.min_bandwidth();
    results->link_infos[i].max_bandwidth = linkinfo.max_bandwidth();
    results->link_infos[i].hops = linkinfo.hops();
    results->link_infos[i].link_type = static_cast<rdc_topology_link_type_t>(linkinfo.link_type());
    results->link_infos[i].is_p2p_accessible = linkinfo.p2p_accessible();
  }

  return RDC_ST_OK;
}

rdc_status_t RdcStandaloneHandler::rdc_link_status_get(rdc_link_status_t* results) {
  ::rdc::Empty request;
  ::rdc::GetLinkStatusResponse reply;
  ::grpc::ClientContext context;

  ::grpc::Status status = stub_->GetLinkStatus(&context, request, &reply);
  rdc_status_t err_status = error_handle(status, reply.status());
  if (err_status != RDC_ST_OK) return err_status;

  ::rdc::LinkStatus LinkStatus = reply.linkstatus();
  results->num_of_gpus = LinkStatus.num_of_gpus();

  for (uint32_t i = 0; i < LinkStatus.num_of_gpus(); ++i) {
    ::rdc::GpuLinkStatus gpulinkstatus = LinkStatus.gpus(i);
    results->gpus[i].gpu_index = gpulinkstatus.gpu_index();
    results->gpus[i].num_of_links = gpulinkstatus.num_of_links();
    results->gpus[i].link_types = static_cast<rdc_topology_link_type_t>(gpulinkstatus.link_types());
    for (uint32_t n = 0; n < gpulinkstatus.num_of_links(); n++) {
      results->gpus[i].link_states[n] = static_cast<rdc_link_state_t>(gpulinkstatus.link_states(n));
    }
  }

  return RDC_ST_OK;
}

rdc_status_t RdcStandaloneHandler::rdc_get_num_partition(uint32_t index, uint16_t* num_partition) {
  ::rdc::GetNumPartitionRequest request;
  request.set_gpu_index(index);
  ::rdc::GetNumPartitionResponse reply;
  ::grpc::ClientContext context;

  ::grpc::Status status = stub_->GetNumPartition(&context, request, &reply);
  rdc_status_t err_status = error_handle(status, reply.status());
  if (err_status != RDC_ST_OK) {
    return err_status;
  }
  *num_partition = reply.num_partition();
  return RDC_ST_OK;
}

rdc_status_t RdcStandaloneHandler::rdc_instance_profile_get(
    uint32_t entity_index, rdc_instance_resource_type_t resource_type,
    rdc_resource_profile_t* profile) {
  ::rdc::GetInstanceProfileRequest request;
  request.set_entity_index(entity_index);
  request.set_resource_type(static_cast<uint32_t>(resource_type));

  ::rdc::GetInstanceProfileResponse reply;
  ::grpc::ClientContext context;

  ::grpc::Status status = stub_->GetInstanceProfile(&context, request, &reply);
  rdc_status_t err_status = error_handle(status, reply.status());
  if (err_status != RDC_ST_OK) {
    return err_status;
  }

  profile->partition_resource = reply.partition_resource();
  profile->num_partitions_share_resource = reply.num_partitions_share_resource();
  return RDC_ST_OK;
}

}  // namespace rdc
}  // namespace amd
