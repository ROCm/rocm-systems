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

#include <chrono>  // NOLINT
#include <future>
#include <limits>

#include "rdc.grpc.pb.h"  // NOLINT
#include "rdc.pb.h"
#include "rdc/rdc.h"
#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/rdc_common.h"

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

namespace {
// True on the threads that read policy streams, where the callbacks run.
thread_local bool t_in_policy_reader = false;
}  // namespace

RdcStandaloneHandler::~RdcStandaloneHandler() {
  // Every reader the handle owns, also one unregistered from a callback,
  // ends here, before stub_ does. rdc_policy_register() calls that wait for
  // admission give up first and leave without touching the handle again.
  std::unique_lock<std::mutex> lock(policy_streams_mutex_);
  destructing_ = true;
  auto give_up = [](const std::shared_ptr<policy_stream>& stream) {
    {
      std::lock_guard<std::mutex> entry_lock(stream->mutex);
      stream->cancel = true;
    }
    stream->cv.notify_all();
    stop_policy_stream(stream.get());
  };
  for (auto& entry : policy_streams_) {
    give_up(entry.second);
  }
  for (auto& stream : draining_) {
    give_up(stream);
  }
  waiters_cv_.wait(lock, [this] { return in_flight_waiters_ == 0; });

  // Join with the mutex released: a callback may call rdc_policy_unregister(),
  // which returns at once now.
  while (!policy_streams_.empty() || !draining_.empty()) {
    std::vector<std::shared_ptr<policy_stream>> streams;
    streams.swap(draining_);
    for (auto& entry : policy_streams_) {
      streams.push_back(entry.second);
    }
    policy_streams_.clear();
    lock.unlock();
    for (auto& stream : streams) {
      if (!stream->reader.joinable()) {
        continue;
      }
      if (stream->reader.get_id() == std::this_thread::get_id()) {
        // rdc_disconnect() from a callback, which rdc.h does not allow; do not
        // join this thread from itself.
        stream->reader.detach();
      } else {
        stream->reader.join();
      }
    }
    lock.lock();
  }
}

void RdcStandaloneHandler::stop_policy_stream(policy_stream* stream) {
  stream->stop = true;
  stream->context.TryCancel();
}

void RdcStandaloneHandler::reap_draining_locked() {
  for (auto it = draining_.begin(); it != draining_.end();) {
    const std::shared_ptr<policy_stream>& stream = *it;
    bool ended = false;
    {
      std::lock_guard<std::mutex> entry_lock(stream->mutex);
      ended = stream->ended;
    }
    if (!ended || stream->reader.get_id() == std::this_thread::get_id()) {
      ++it;
      continue;
    }
    if (stream->reader.joinable()) {
      stream->reader.join();  // the reader has finished its loop
    }
    {
      std::lock_guard<std::mutex> entry_lock(stream->mutex);
      if (stream->admitted) {
        policy_server_old_ = false;  // rdcd admits streams
      }
    }
    it = draining_.erase(it);
  }
}

rdc_status_t RdcStandaloneHandler::policy_stream_status(const ::grpc::Status& status) {
  switch (status.error_code()) {
    case ::grpc::StatusCode::OK:  // ended normally, for example by an unregister
      return RDC_ST_OK;
    case ::grpc::StatusCode::NOT_FOUND:
      return RDC_ST_GROUP_NOT_FOUND;
    case ::grpc::StatusCode::RESOURCE_EXHAUSTED:
      return RDC_ST_MAX_LIMIT;
    default:  // also rdcd stopping, or rdcd that cannot be reached
      return RDC_ST_CLIENT_ERROR;
  }
}

rdc_status_t RdcStandaloneHandler::error_handle(::grpc::Status status, uint32_t rdc_status) {
  if (!status.ok()) {
    std::cout << status.error_message() << ". Error code:" << status.error_code() << std::endl;
    return RDC_ST_CLIENT_ERROR;
  }

  return static_cast<rdc_status_t>(rdc_status);
}

namespace {

// How long a call waits for rdcd before it fails with RDC_ST_CLIENT_ERROR. A
// diagnostic answers once its tests have run. A test case run, which can run an RVS
// test for as long as its configuration sets, and a policy stream have no deadline.
constexpr std::chrono::seconds kCallTimeout(120);
constexpr std::chrono::minutes kDiagnosticTimeout(30);

void set_timeout(::grpc::ClientContext* context, std::chrono::seconds timeout) {
  context->set_deadline(std::chrono::system_clock::now() + timeout);
}

// Copies a field value from a GetLatestFieldValue or GetFieldSince reply. A value
// of a type the C API does not have is a client error, and leaves *value as it was.
template <typename Reply>
rdc_status_t copy_field_value(const Reply& reply, rdc_field_value* value) {
  const uint32_t type = reply.type();
  if (type != INTEGER && type != DOUBLE && type != STRING && type != BLOB) {
    RDC_LOG(RDC_ERROR, "rdcd replied with a field value of unknown type " << type);
    return RDC_ST_CLIENT_ERROR;
  }
  value->field_id = static_cast<rdc_field_t>(reply.field_id());
  value->status = reply.rdc_status();
  value->ts = reply.ts();
  value->type = static_cast<rdc_field_type_t>(type);
  if (type == INTEGER) {
    value->value.l_int = reply.l_int();
  } else if (type == DOUBLE) {
    value->value.dbl = reply.dbl();
  } else {
    strncpy_with_null(value->value.str, reply.str().c_str(), RDC_MAX_STR_LENGTH);
  }
  return RDC_ST_OK;
}

}  // namespace

// JOB RdcAPI
rdc_status_t RdcStandaloneHandler::rdc_job_start_stats(rdc_gpu_group_t groupId,
                                                       const char job_id[64],
                                                       uint64_t update_freq) {
  ::rdc::StartJobStatsRequest request;
  ::rdc::StartJobStatsResponse reply;
  ::grpc::ClientContext context;
  set_timeout(&context, kCallTimeout);

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
  set_timeout(&context, kCallTimeout);

  request.set_job_id(job_id);
  ::grpc::Status status = stub_->GetJobStats(&context, request, &reply);
  rdc_status_t err_status = error_handle(status, reply.status());
  if (err_status != RDC_ST_OK) return err_status;

  // Count the entries the reply carries, not the counts it states.
  if (reply.gpus_size() > RDC_MAX_NUM_GPUS_PER_JOB) {
    return RDC_ST_BAD_PARAMETER;
  }
  if (reply.processes_size() > RDC_MAX_NUM_PROCESSES_STATUS) {
    return RDC_ST_BAD_PARAMETER;
  }

  // Leave no entries of an earlier call past the ones this reply fills.
  *p_job_info = {};
  p_job_info->num_gpus = reply.gpus_size();
  copy_gpu_usage_info(reply.summary(), &(p_job_info->summary));
  for (int i = 0; i < reply.gpus_size(); i++) {
    copy_gpu_usage_info(reply.gpus(i), &(p_job_info->gpus[i]));
  }

  p_job_info->num_processes = reply.processes_size();
  for (int i = 0; i < reply.processes_size(); i++) {
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
  set_timeout(&context, kCallTimeout);

  request.set_job_id(job_id);
  ::grpc::Status status = stub_->StopJobStats(&context, request, &reply);
  rdc_status_t err_status = error_handle(status, reply.status());

  return err_status;
}

rdc_status_t RdcStandaloneHandler::rdc_job_remove(const char job_id[64]) {
  ::rdc::RemoveJobRequest request;
  ::rdc::RemoveJobResponse reply;
  ::grpc::ClientContext context;
  set_timeout(&context, kCallTimeout);

  request.set_job_id(job_id);
  ::grpc::Status status = stub_->RemoveJob(&context, request, &reply);
  rdc_status_t err_status = error_handle(status, reply.status());

  return err_status;
}

rdc_status_t RdcStandaloneHandler::rdc_job_remove_all() {
  ::rdc::Empty request;
  ::rdc::RemoveAllJobResponse reply;
  ::grpc::ClientContext context;
  set_timeout(&context, kCallTimeout);

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
  set_timeout(&context, kCallTimeout);

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
  set_timeout(&context, kCallTimeout);

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
  set_timeout(&context, kCallTimeout);

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
  set_timeout(&context, kCallTimeout);

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
  set_timeout(&context, kCallTimeout);

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
  set_timeout(&context, kCallTimeout);

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

rdc_status_t RdcStandaloneHandler::rdc_group_field_add_field(rdc_field_grp_t rdc_field_group_id,
                                                             rdc_field_t field_id) {
  ::rdc::AddFieldToFieldGroupRequest request;
  ::rdc::AddFieldToFieldGroupResponse reply;
  ::grpc::ClientContext context;
  set_timeout(&context, kCallTimeout);

  request.set_field_group_id(rdc_field_group_id);
  request.set_field_id(field_id);
  ::grpc::Status status = stub_->AddFieldToFieldGroup(&context, request, &reply);
  rdc_status_t err_status = error_handle(status, reply.status());

  return err_status;
}

rdc_status_t RdcStandaloneHandler::rdc_group_field_get_info(
    rdc_field_grp_t rdc_field_group_id, rdc_field_group_info_t* field_group_info) {
  if (!field_group_info) {
    return RDC_ST_BAD_PARAMETER;
  }

  ::rdc::GetFieldGroupInfoRequest request;
  ::rdc::GetFieldGroupInfoResponse reply;
  ::grpc::ClientContext context;
  set_timeout(&context, kCallTimeout);

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
  set_timeout(&context, kCallTimeout);

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
  set_timeout(&context, kCallTimeout);

  ::grpc::Status status = stub_->GetGroupAllIds(&context, request, &reply);
  rdc_status_t err_status = error_handle(status, reply.status());
  if (err_status != RDC_ST_OK) return err_status;

  // RDC holds up to RDC_MAX_NUM_GROUPS groups.
  if (reply.group_ids_size() > RDC_MAX_NUM_GROUPS) {
    return RDC_ST_MAX_LIMIT;
  }
  *count = reply.group_ids_size();
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
  set_timeout(&context, kCallTimeout);

  ::grpc::Status status = stub_->GetFieldGroupAllIds(&context, request, &reply);
  rdc_status_t err_status = error_handle(status, reply.status());
  if (err_status != RDC_ST_OK) return err_status;

  // RDC holds up to RDC_MAX_NUM_FIELD_GROUPS field groups.
  if (reply.field_group_ids_size() > RDC_MAX_NUM_FIELD_GROUPS) {
    return RDC_ST_MAX_LIMIT;
  }
  *count = reply.field_group_ids_size();
  for (uint32_t i = 0; i < *count; i++) {
    field_group_id_list[i] = reply.field_group_ids(i);
  }

  return RDC_ST_OK;
}

rdc_status_t RdcStandaloneHandler::rdc_group_gpu_destroy(rdc_gpu_group_t p_rdc_group_id) {
  ::rdc::DestroyGpuGroupRequest request;
  ::rdc::DestroyGpuGroupResponse reply;
  ::grpc::ClientContext context;
  set_timeout(&context, kCallTimeout);

  request.set_group_id(p_rdc_group_id);
  ::grpc::Status status = stub_->DestroyGpuGroup(&context, request, &reply);
  return error_handle(status, reply.status());
}

rdc_status_t RdcStandaloneHandler::rdc_group_field_destroy(rdc_field_grp_t rdc_field_group_id) {
  ::rdc::DestroyFieldGroupRequest request;
  ::rdc::DestroyFieldGroupResponse reply;
  ::grpc::ClientContext context;
  set_timeout(&context, kCallTimeout);

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
  set_timeout(&context, kCallTimeout);

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
  set_timeout(&context, kCallTimeout);

  request.set_gpu_index(gpu_index);
  request.set_field_id(field);
  ::grpc::Status status = stub_->GetLatestFieldValue(&context, request, &reply);
  rdc_status_t err_status = error_handle(status, reply.status());
  if (err_status != RDC_ST_OK) return err_status;

  err_status = copy_field_value(reply, value);
  if (err_status != RDC_ST_OK) return err_status;

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
  set_timeout(&context, kCallTimeout);

  request.set_gpu_index(gpu_index);
  request.set_field_id(field);
  request.set_since_time_stamp(since_time_stamp);
  ::grpc::Status status = stub_->GetFieldSince(&context, request, &reply);
  rdc_status_t err_status = error_handle(status, reply.status());
  if (err_status != RDC_ST_OK) return err_status;

  err_status = copy_field_value(reply, value);
  if (err_status != RDC_ST_OK) return err_status;
  *next_since_time_stamp = reply.next_since_time_stamp();

  return RDC_ST_OK;
}

rdc_status_t RdcStandaloneHandler::rdc_field_unwatch(rdc_gpu_group_t group_id,
                                                     rdc_field_grp_t field_group_id) {
  ::rdc::UnWatchFieldsRequest request;
  ::rdc::UnWatchFieldsResponse reply;
  ::grpc::ClientContext context;
  set_timeout(&context, kCallTimeout);

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
  set_timeout(&context, kDiagnosticTimeout);

  request.set_group_id(group_id);
  request.set_level(level);
  if (config != nullptr) {
    request.set_config(config);
  }
  request.set_config_size(config_size);

  auto reader = stub_->DiagnosticRun(&context, request);
  // for the duration of the DiagnosticRun (multiple tests) - we're stuck in this loop
  //
  // there are 2 optional reply fields:
  // * log - reports messages back during the diagnostic run
  // * response - delivered when the diagnostic run completes
  // A reply with an error status instead ends a run that failed or was rejected. (A stream
  // cannot use error_handle(): its status comes in a reply, and the gRPC status at the end.)
  rdc_status_t run_status = RDC_ST_OK;
  bool has_response = false;
  while (reader->Read(&reply)) {
    if (reply.has_log()) {
      // TODO: Add different logging levels
      std::cout << "LOG: " << reply.log() << std::endl;
      continue;
    }
    if (reply.status() != RDC_ST_OK) {
      run_status = static_cast<rdc_status_t>(reply.status());
      continue;
    }
    if (reply.has_response()) {
      RDC_LOG(RDC_DEBUG, "HAS RESPONSE!");
      auto res = reply.response();
      if (res.diag_info_size() > static_cast<int>(MAX_TEST_CASES)) {
        return RDC_ST_BAD_PARAMETER;
      }
      for (const ::rdc::DiagnosticTestResult& result : res.diag_info()) {
        if (result.gpu_results_size() > RDC_MAX_NUM_DEVICES) {
          return RDC_ST_BAD_PARAMETER;
        }
      }
      response->results_count = res.diag_info_size();
      for (int i = 0; i < res.diag_info_size(); i++) {
        const ::rdc::DiagnosticTestResult& result = res.diag_info(i);
        rdc_diag_test_result_t& to_result = response->diag_info[i];
        to_result.status = static_cast<rdc_diag_result_t>(result.status());

        // Set details
        to_result.details.code = result.details().code();
        strncpy_with_null(to_result.details.msg, result.details().msg().c_str(),
                          MAX_DIAG_MSG_LENGTH);

        to_result.test_case = static_cast<rdc_diag_test_cases_t>(result.test_case());
        to_result.per_gpu_result_count = result.gpu_results_size();

        // Set Result details
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
      has_response = true;
    }
  }

  auto status = reader->Finish();
  if (!status.ok()) {
    RDC_LOG(RDC_ERROR, "DiagnosticRun failed: gRPC status " << status.error_code());
    return RDC_ST_CLIENT_ERROR;
  }
  if (run_status == RDC_ST_OK && !has_response) {
    RDC_LOG(RDC_ERROR, "rdcd ended the diagnostic without its results");
    return RDC_ST_CLIENT_ERROR;
  }
  return run_status;
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
  if (config != nullptr) {
    request.set_config(config);
  }
  request.set_config_size(config_size);
  request.set_test_case(static_cast<::rdc::DiagnosticTestCaseRunRequest_TestCaseType>(test_case));

  auto reader = stub_->DiagnosticTestCaseRun(&context, request);
  // A reply with an error status ends a run that failed or was rejected.
  rdc_status_t run_status = RDC_ST_OK;
  bool has_result = false;
  while (reader->Read(&reply)) {
    if (reply.status() != RDC_ST_OK) {
      run_status = static_cast<rdc_status_t>(reply.status());
      continue;
    }
    if (!reply.has_result()) {
      RDC_LOG(RDC_ERROR, "NO TEST_RUN RESULT!");
      continue;
    }

    auto result = reply.result();
    if (result.gpu_results_size() > RDC_MAX_NUM_DEVICES) {
      return RDC_ST_BAD_PARAMETER;
    }

    to_result->status = static_cast<rdc_diag_result_t>(result.status());

    // Set details
    to_result->details.code = result.details().code();
    strncpy_with_null(to_result->details.msg, result.details().msg().c_str(), MAX_DIAG_MSG_LENGTH);

    to_result->test_case = static_cast<rdc_diag_test_cases_t>(result.test_case());
    to_result->per_gpu_result_count = result.gpu_results_size();

    // Set Result details
    for (int j = 0; j < result.gpu_results_size(); j++) {
      auto per_gpu_result = result.gpu_results(j);
      rdc_diag_per_gpu_result_t& to_per_gpu = to_result->gpu_results[j];
      to_per_gpu.gpu_index = per_gpu_result.gpu_index();
      to_per_gpu.gpu_result.code = per_gpu_result.gpu_result().code();
      strncpy_with_null(to_per_gpu.gpu_result.msg, per_gpu_result.gpu_result().msg().c_str(),
                        MAX_DIAG_MSG_LENGTH);
    }
    strncpy_with_null(to_result->info, result.info().c_str(), MAX_DIAG_MSG_LENGTH);
    has_result = true;
  }

  auto status = reader->Finish();
  if (!status.ok()) {
    RDC_LOG(RDC_ERROR, "DiagnosticTestCaseRun failed: gRPC status " << status.error_code());
    return RDC_ST_CLIENT_ERROR;
  }
  if (run_status == RDC_ST_OK && !has_result) {
    RDC_LOG(RDC_ERROR, "rdcd ended the test case run without its result");
    return RDC_ST_CLIENT_ERROR;
  }
  return run_status;
}

// Control RdcAPI
rdc_status_t RdcStandaloneHandler::rdc_field_update_all(uint32_t wait_for_update) {
  ::rdc::UpdateAllFieldsRequest request;
  ::rdc::UpdateAllFieldsResponse reply;
  ::grpc::ClientContext context;
  set_timeout(&context, kCallTimeout);

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
  set_timeout(&context, kCallTimeout);

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
  set_timeout(&context, kCallTimeout);

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
  set_timeout(&context, kCallTimeout);

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
  set_timeout(&context, kCallTimeout);

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
  set_timeout(&context, kCallTimeout);

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
  set_timeout(&context, kCallTimeout);

  if (count == nullptr) {
    return RDC_ST_BAD_PARAMETER;
  }

  request.set_group_id(group_id);

  // call gRPC
  ::grpc::Status status = stub_->GetPolicy(&context, request, &reply);
  rdc_status_t err_status = error_handle(status, reply.status());
  if (err_status != RDC_ST_OK) return err_status;

  const auto& response = reply.response();
  // Count the policies the reply carries, not the count it states.
  if (response.policies_size() > RDC_MAX_POLICY_SETTINGS) {
    return RDC_ST_BAD_PARAMETER;
  }
  const uint32_t policy_count = response.policies_size();

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
  set_timeout(&context, kCallTimeout);

  request.set_group_id(group_id);

  request.set_condition_type(
      static_cast<::rdc::DeletePolicyRequest_PolicyConditionType>(condition_type));

  // call gRPC
  ::grpc::Status status = stub_->DeletePolicy(&context, request, &reply);

  return error_handle(status, reply.status());
}

rdc_status_t RdcStandaloneHandler::rdc_policy_register(rdc_gpu_group_t group_id,
                                                       rdc_policy_register_callback callback) {
  std::unique_lock<std::mutex> lock(policy_streams_mutex_);
  if (destructing_) {
    return RDC_ST_CLIENT_ERROR;
  }
  reap_draining_locked();
  auto existing = policy_streams_.find(group_id);
  if (existing != policy_streams_.end()) {
    std::shared_ptr<policy_stream> old = existing->second;
    bool ended = false;
    {
      std::lock_guard<std::mutex> entry_lock(old->mutex);
      ended = old->ended;
      if (old->admitted) {
        policy_server_old_ = false;  // rdcd admits streams
      }
    }
    if (!ended) {
      return RDC_ST_CONFLICT;
    }
    // The stream ended by itself (rdcd stopped, or rejected it late): its
    // reader has finished, and the group can be registered again.
    policy_streams_.erase(existing);
    if (old->reader.joinable() && old->reader.get_id() != std::this_thread::get_id()) {
      old->reader.join();
    }
  }

  // Read the stream from rdcd on a thread of its own. The thread uses only
  // the stream it is given, never policy_streams_, and touches the handler
  // only to start the call.
  auto stream = std::make_shared<policy_stream>();
  policy_streams_.emplace(group_id, stream);
  try {
    stream->reader = std::thread([this, s = stream, group_id, callback]() {
      t_in_policy_reader = true;
      ::rdc::RegisterPolicyRequest request;
      ::rdc::RegisterPolicyResponse reply;

      request.set_group_id(group_id);

      std::unique_ptr<grpc::ClientReader<::rdc::RegisterPolicyResponse>> reader(
          stub_->RegisterPolicy(&s->context, request));

      // rdcd admits a stream with initial metadata at once. A rejection
      // carries none, and an older rdcd sends it with the first notification.
      reader->WaitForInitialMetadata();
      {
        std::lock_guard<std::mutex> lock(s->mutex);
        const auto& metadata = s->context.GetServerInitialMetadata();
        s->admitted = metadata.find(kRdcPolicyAdmissionKey) != metadata.end();
      }
      s->cv.notify_all();

      // The loop ends when stop_policy_stream() sets stop, or when Read()
      // fails: rdcd ended the stream, the connection is gone or the call was
      // cancelled.
      bool notified = false;
      while (!s->stop && reader->Read(&reply)) {
        if (!notified) {
          notified = true;
          {
            std::lock_guard<std::mutex> lock(s->mutex);
            s->old_server = !s->admitted;
          }
          s->cv.notify_all();
        }
        ::rdc::PolicyCondition cond = reply.condition();

        rdc_policy_callback_response_t response;
        response.version = reply.version();
        response.condition.type = static_cast<rdc_policy_condition_type_t>(cond.type());
        response.condition.value = cond.value();
        response.group_id = reply.group_id();
        response.value = reply.value();
        response.gpu_index = reply.gpu_index();
        response.reset_triggered = reply.reset_triggered();

        callback(&response);
      }

      ::grpc::Status status = reader->Finish();
      {
        std::lock_guard<std::mutex> lock(s->mutex);
        // rdc_policy_register() reports an end it sees; log one that comes later.
        if (s->returned && !s->stop && !status.ok()) {
          RDC_LOG(RDC_ERROR, "The policy registration of group "
                                 << group_id
                                 << " ended: " << rdc_status_string(policy_stream_status(status))
                                 << " (gRPC status " << status.error_code() << ")");
        }
        s->status = status;
        s->ended = true;
      }
      s->cv.notify_all();
    });
  } catch (...) {  // std::system_error or std::bad_alloc
    policy_streams_.erase(group_id);
    return RDC_ST_INSUFF_RESOURCES;
  }

  if (policy_server_old_) {
    // Wait again once a stream of this handle shows that rdcd admits streams.
    for (auto& entry : policy_streams_) {
      std::lock_guard<std::mutex> entry_lock(entry.second->mutex);
      if (entry.second->admitted) {
        policy_server_old_ = false;
        break;
      }
    }
  }

  // Wait up to 5 s for rdcd to admit or reject the stream (not at all for an
  // rdcd that does not admit), holding only the stream's mutex. The copy of
  // the stream keeps it alive even if an unregister erases it meanwhile.
  const auto wait = policy_server_old_ ? std::chrono::seconds(0) : std::chrono::seconds(5);
  in_flight_waiters_++;
  lock.unlock();
  bool admitted = false;
  bool ended = false;
  bool cancel = false;
  ::grpc::Status status;
  {
    std::unique_lock<std::mutex> entry_lock(stream->mutex);
    stream->cv.wait_for(entry_lock, wait, [&stream] {
      return stream->admitted || stream->old_server || stream->ended || stream->cancel;
    });
    admitted = stream->admitted;
    ended = stream->ended;
    cancel = stream->cancel;
    status = stream->status;
    stream->returned = !ended;
  }
  lock.lock();

  rdc_status_t result = RDC_ST_OK;
  if (admitted) {
    policy_server_old_ = false;  // rdcd admits streams
  }
  if (destructing_) {
    result = RDC_ST_CLIENT_ERROR;
  } else if (cancel) {
    result = RDC_ST_OK;  // a concurrent rdc_policy_unregister() gave the stream up
  } else if (ended) {
    // Rejected, or admitted and ended already: the registration is gone.
    result = policy_stream_status(status);
    auto it = policy_streams_.find(group_id);
    if (it != policy_streams_.end() && it->second == stream) {
      policy_streams_.erase(it);
      if (stream->reader.joinable() && stream->reader.get_id() != std::this_thread::get_id()) {
        stream->reader.join();
      }
    }
  } else if (!admitted) {
    // A notification without admission, or no answer within the wait: an
    // rdcd that predates admission. Later registrations do not wait.
    policy_server_old_ = true;
  }
  in_flight_waiters_--;
  waiters_cv_.notify_all();
  return result;
}

rdc_status_t RdcStandaloneHandler::rdc_policy_unregister(rdc_gpu_group_t group_id) {
  ::rdc::UnRegisterPolicyRequest request;
  ::rdc::UnRegisterPolicyResponse reply;
  ::grpc::ClientContext context;
  set_timeout(&context, kCallTimeout);

  // A callback cannot wait for its own reader, or for another reader that
  // may be waiting for it, so from a callback the stream goes to draining_
  // to be joined later.
  const bool from_callback = t_in_policy_reader;
  std::shared_ptr<policy_stream> stream;
  {
    std::lock_guard<std::mutex> guard(policy_streams_mutex_);
    if (destructing_) {
      return RDC_ST_CLIENT_ERROR;
    }
    reap_draining_locked();
    auto it = policy_streams_.find(group_id);
    if (it != policy_streams_.end()) {
      stream = it->second;
      policy_streams_.erase(it);
      {
        std::lock_guard<std::mutex> entry_lock(stream->mutex);
        stream->cancel = true;  // a register waiting for admission returns
        if (stream->admitted) {
          policy_server_old_ = false;  // rdcd admits streams
        }
      }
      stream->cv.notify_all();
      if (from_callback) {
        draining_.push_back(stream);
      }
    }
  }

  // rdcd ends the stream, then the reader is stopped and joined, also when
  // rdcd cannot be reached.
  request.set_group_id(group_id);
  ::grpc::Status status = stub_->UnRegisterPolicy(&context, request, &reply);
  if (stream) {
    stop_policy_stream(stream.get());
    if (!from_callback && stream->reader.joinable()) {
      stream->reader.join();
      // An admission that arrived after the stream was given up still counts.
      std::lock_guard<std::mutex> guard(policy_streams_mutex_);
      std::lock_guard<std::mutex> entry_lock(stream->mutex);
      if (stream->admitted) {
        policy_server_old_ = false;
      }
    }
  }

  return error_handle(status, reply.status());
}

// Health RdcAPI
rdc_status_t RdcStandaloneHandler::rdc_health_set(rdc_gpu_group_t group_id,
                                                  unsigned int components) {
  ::rdc::SetHealthRequest request;
  ::rdc::SetHealthResponse reply;
  ::grpc::ClientContext context;
  set_timeout(&context, kCallTimeout);

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
  set_timeout(&context, kCallTimeout);

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
  set_timeout(&context, kCallTimeout);

  request.set_group_id(group_id);
  ::grpc::Status status = stub_->CheckHealth(&context, request, &reply);
  rdc_status_t err_status = error_handle(status, reply.status());
  if (err_status != RDC_ST_OK) return err_status;

  auto res = reply.response();
  if (res.incidents_size() > HEALTH_MAX_ERROR_ITEMS) {
    return RDC_ST_BAD_PARAMETER;
  }

  response->overall_health = static_cast<rdc_health_result_t>(res.overall_health());
  response->incidents_count = res.incidents_size();

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
  set_timeout(&context, kCallTimeout);

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
  set_timeout(&context, kCallTimeout);

  request.set_gpu_index(gpu_index);
  ::grpc::Status status = stub_->GetTopology(&context, request, &reply);
  rdc_status_t err_status = error_handle(status, reply.status());
  if (err_status != RDC_ST_OK) return err_status;

  ::rdc::Topology Topology = reply.toppology();
  if (Topology.link_infos_size() > RDC_MAX_NUM_DEVICES) {
    return RDC_ST_BAD_PARAMETER;
  }

  results->num_of_gpus = Topology.link_infos_size();
  results->numa_node = Topology.numa_node();

  for (int i = 0; i < Topology.link_infos_size(); ++i) {
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
  set_timeout(&context, kCallTimeout);

  ::grpc::Status status = stub_->GetLinkStatus(&context, request, &reply);
  rdc_status_t err_status = error_handle(status, reply.status());
  if (err_status != RDC_ST_OK) return err_status;

  const ::rdc::LinkStatus& LinkStatus = reply.linkstatus();
  if (LinkStatus.gpus_size() > RDC_MAX_NUM_DEVICES) {
    return RDC_ST_BAD_PARAMETER;
  }
  for (const ::rdc::GpuLinkStatus& gpulinkstatus : LinkStatus.gpus()) {
    if (gpulinkstatus.link_states_size() > RDC_MAX_NUM_OF_LINKS) {
      return RDC_ST_BAD_PARAMETER;
    }
  }

  results->num_of_gpus = LinkStatus.gpus_size();

  for (int i = 0; i < LinkStatus.gpus_size(); ++i) {
    const ::rdc::GpuLinkStatus& gpulinkstatus = LinkStatus.gpus(i);
    results->gpus[i].gpu_index = gpulinkstatus.gpu_index();
    results->gpus[i].num_of_links = gpulinkstatus.link_states_size();
    results->gpus[i].link_types = static_cast<rdc_topology_link_type_t>(gpulinkstatus.link_types());
    for (int n = 0; n < gpulinkstatus.link_states_size(); n++) {
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
  set_timeout(&context, kCallTimeout);

  ::grpc::Status status = stub_->GetNumPartition(&context, request, &reply);
  rdc_status_t err_status = error_handle(status, reply.status());
  if (err_status != RDC_ST_OK) {
    return err_status;
  }
  if (reply.num_partition() > std::numeric_limits<uint16_t>::max()) {
    RDC_LOG(RDC_ERROR, "rdcd replied with " << reply.num_partition() << " partitions");
    return RDC_ST_CLIENT_ERROR;
  }
  *num_partition = static_cast<uint16_t>(reply.num_partition());
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
  set_timeout(&context, kCallTimeout);

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
