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
#include <assert.h>
#include <grpcpp/grpcpp.h>

#include <iostream>
#include <memory>
#include <string>
#include <csignal>

#include "rdc.grpc.pb.h"  // NOLINT
#include "rdc/rdc_api_service.h"
#include "rdc/rdc.h"
#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/rdc_common.h"

namespace amd {
namespace rdc {

RdcAPIServiceImpl::RdcAPIServiceImpl():rdc_handle_(nullptr) {
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
    uint32_t  gpu_index_list[RDC_MAX_NUM_DEVICES];
    uint32_t count = 0;
    rdc_status_t result =  rdc_device_get_all(rdc_handle_,
                  gpu_index_list, &count);
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
      ::grpc::ServerContext* context,
      const ::rdc::GetDeviceAttributesRequest* request,
      ::rdc::GetDeviceAttributesResponse* reply) {
    (void)(context);
    if (!reply || !request) {
      return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
    }
    uint32_t gpu_index = request->gpu_index();
    rdc_device_attributes_t attribute;
    rdc_status_t result = rdc_device_get_attributes(rdc_handle_,
            gpu_index, &attribute);

    ::rdc::DeviceAttributes* attr = reply->mutable_attributes();
    attr->set_device_name(attribute.device_name);

    reply->set_status(result);

    return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::CreateGpuGroup(
                  ::grpc::ServerContext* context,
                  const ::rdc::CreateGpuGroupRequest* request,
                  ::rdc::CreateGpuGroupResponse* reply) {
    (void)(context);
    if (!reply || !request) {
      return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
    }

    rdc_gpu_group_t group_id = 0;
    rdc_status_t result = rdc_group_gpu_create(rdc_handle_,
                static_cast<rdc_group_type_t>(request->type()),
                request->group_name().c_str(), &group_id);
    reply->set_status(result);
    if (result != RDC_ST_OK) {
        return ::grpc::Status::OK;
    }

    reply->set_group_id(group_id);

    return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::AddToGpuGroup(
                  ::grpc::ServerContext* context,
                  const ::rdc::AddToGpuGroupRequest* request,
                  ::rdc::AddToGpuGroupResponse* reply) {
    (void)(context);
    if (!reply || !request) {
      return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
    }

    rdc_status_t result = rdc_group_gpu_add(rdc_handle_,
            request->group_id(), request->gpu_index());
    reply->set_status(result);

    return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::GetGpuGroupInfo(
                  ::grpc::ServerContext* context,
                  const ::rdc::GetGpuGroupInfoRequest* request,
                  ::rdc::GetGpuGroupInfoResponse* reply) {
    (void)(context);
    if (!reply || !request) {
      return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
    }

    rdc_group_info_t group_info;
    rdc_status_t result = rdc_group_gpu_get_info(
                rdc_handle_, request->group_id(), &group_info);
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

::grpc::Status RdcAPIServiceImpl::GetGroupAllIds(
                  ::grpc::ServerContext* context,
                  const ::rdc::Empty* request,
                  ::rdc::GetGroupAllIdsResponse* reply) {
    if (!reply || !request || !context) {
      return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
    }

    rdc_gpu_group_t group_id_list[RDC_MAX_NUM_GROUPS];
    uint32_t count = 0;
    rdc_status_t result = rdc_group_get_all_ids(
                rdc_handle_, group_id_list, &count);
    reply->set_status(result);
    if (result != RDC_ST_OK) {
        return ::grpc::Status::OK;
    }

    for (uint32_t i = 0; i < count; i++) {
         reply->add_group_ids(group_id_list[i]);
    }

    return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::GetFieldGroupAllIds(
                  ::grpc::ServerContext* context,
                  const ::rdc::Empty* request,
                  ::rdc::GetFieldGroupAllIdsResponse* reply) {
    if (!reply || !request || !context) {
      return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
    }

    rdc_field_grp_t field_group_id_list[RDC_MAX_NUM_FIELD_GROUPS];
    uint32_t count = 0;
    rdc_status_t result = rdc_group_field_get_all_ids(
                    rdc_handle_, field_group_id_list, &count);
    reply->set_status(result);
    if (result != RDC_ST_OK) {
        return ::grpc::Status::OK;
    }

    for (uint32_t i = 0; i < count; i++) {
         reply->add_field_group_ids(field_group_id_list[i]);
    }

    return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::DestroyGpuGroup(
                  ::grpc::ServerContext* context,
                  const ::rdc::DestroyGpuGroupRequest* request,
                  ::rdc::DestroyGpuGroupResponse* reply) {
    (void)(context);
    if (!reply || !request) {
      return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
    }

    rdc_status_t result = rdc_group_gpu_destroy(
                rdc_handle_, request->group_id());
    reply->set_status(result);

    return ::grpc::Status::OK;
}


::grpc::Status RdcAPIServiceImpl::CreateFieldGroup(
                  ::grpc::ServerContext* context,
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
    rdc_status_t result = rdc_group_field_create(
            rdc_handle_, request->field_ids_size() , &field_ids[0],
            request->field_group_name().c_str(), &field_group_id);
    reply->set_status(result);
    if (result != RDC_ST_OK) {
        return ::grpc::Status::OK;
    }

    reply->set_field_group_id(field_group_id);

    return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::GetFieldGroupInfo(
                  ::grpc::ServerContext* context,
                  const ::rdc::GetFieldGroupInfoRequest* request,
                  ::rdc::GetFieldGroupInfoResponse* reply) {
    (void)(context);
    if (!reply || !request) {
      return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
    }

    rdc_field_group_info_t field_info;
    rdc_status_t result = rdc_group_field_get_info(
              rdc_handle_, request->field_group_id(), &field_info);
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

::grpc::Status RdcAPIServiceImpl::DestroyFieldGroup(
                  ::grpc::ServerContext* context,
                  const ::rdc::DestroyFieldGroupRequest* request,
                  ::rdc::DestroyFieldGroupResponse* reply) {
    (void)(context);
    if (!reply || !request) {
      return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
    }

    rdc_status_t result = rdc_group_field_destroy(
                  rdc_handle_, request->field_group_id());
    reply->set_status(result);

    return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::WatchFields(
                  ::grpc::ServerContext* context,
                  const ::rdc::WatchFieldsRequest* request,
                  ::rdc::WatchFieldsResponse* reply) {
    (void)(context);
    if (!reply || !request) {
      return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
    }

    rdc_status_t result = rdc_field_watch(
            rdc_handle_, request->group_id(), request->field_group_id(),
            request->update_freq(), request->max_keep_age(),
            request->max_keep_samples());
    reply->set_status(result);

    return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::GetLatestFieldValue(
                  ::grpc::ServerContext* context,
                  const ::rdc::GetLatestFieldValueRequest* request,
                  ::rdc::GetLatestFieldValueResponse* reply) {
    (void)(context);
    if (!reply || !request) {
      return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
    }

    rdc_field_value value;
    rdc_status_t result = rdc_field_get_latest_value(rdc_handle_,
       request->gpu_index(), static_cast<rdc_field_t>(request->field_id()),
                                                                      &value);
    reply->set_status(result);
    if (result != RDC_ST_OK) {
        return ::grpc::Status::OK;
    }

    reply->set_field_id(value.field_id);
    reply->set_rdc_status(value.status);
    reply->set_ts(value.ts);
    reply->set_type(static_cast<::rdc::GetLatestFieldValueResponse_FieldType>
                        (value.type));
    if (value.type == INTEGER) {
        reply->set_l_int(value.value.l_int);
    } else if (value.type == DOUBLE) {
        reply->set_dbl(value.value.dbl);
    } else if (value.type == STRING || value.type == BLOB) {
        reply->set_str(value.value.str);
    }

    return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::GetFieldSince(
                  ::grpc::ServerContext* context,
                  const ::rdc::GetFieldSinceRequest* request,
                  ::rdc::GetFieldSinceResponse* reply) {
    (void)(context);
    if (!reply || !request) {
      return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
    }

    rdc_field_value value;
    uint64_t next_timestamp;
    rdc_status_t result = rdc_field_get_value_since(rdc_handle_,
        request->gpu_index(), static_cast<rdc_field_t>(request->field_id()),
                        request->since_time_stamp(), &next_timestamp, &value);
    reply->set_status(result);
    if (result != RDC_ST_OK) {
        return ::grpc::Status::OK;
    }

    reply->set_next_since_time_stamp(next_timestamp);
    reply->set_field_id(value.field_id);
    reply->set_rdc_status(value.status);
    reply->set_ts(value.ts);
    reply->set_type(static_cast<::rdc::GetFieldSinceResponse_FieldType>
                    (value.type));
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

::grpc::Status RdcAPIServiceImpl::UnWatchFields(
                  ::grpc::ServerContext* context,
                  const ::rdc::UnWatchFieldsRequest* request,
                  ::rdc::UnWatchFieldsResponse* reply) {
    (void)(context);
    if (!reply || !request) {
      return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
    }

    rdc_status_t result = rdc_field_unwatch(
              rdc_handle_, request->group_id(), request->field_group_id());
    reply->set_status(result);

    return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::UpdateAllFields(
                  ::grpc::ServerContext* context,
                  const ::rdc::UpdateAllFieldsRequest* request,
                  ::rdc::UpdateAllFieldsResponse* reply) {
    (void)(context);
    if (!reply || !request) {
      return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
    }

    rdc_status_t result = rdc_field_update_all(
                rdc_handle_, request->wait_for_update());
    reply->set_status(result);

    return ::grpc::Status::OK;
}


::grpc::Status RdcAPIServiceImpl::StartJobStats(
                  ::grpc::ServerContext* context,
                  const ::rdc::StartJobStatsRequest* request,
                  ::rdc::StartJobStatsResponse* reply) {
    (void)(context);
    if (!reply || !request) {
      return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
    }
    rdc_status_t result = rdc_job_start_stats(
                rdc_handle_, request->group_id(),
                const_cast<char*>(request->job_id().c_str()),
                request->update_freq());
    reply->set_status(result);

    return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::GetJobStats(
                  ::grpc::ServerContext* context,
                  const ::rdc::GetJobStatsRequest* request,
                  ::rdc::GetJobStatsResponse* reply) {
    (void)(context);
    if (!reply || !request) {
      return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
    }

    rdc_job_info_t job_info;
    rdc_status_t result = rdc_job_get_stats(
                rdc_handle_,
                const_cast<char*>(request->job_id().c_str()),
                &job_info);

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


::grpc::Status RdcAPIServiceImpl::StopJobStats(
                  ::grpc::ServerContext* context,
                  const ::rdc::StopJobStatsRequest* request,
                  ::rdc::StopJobStatsResponse* reply) {
    (void)(context);
    if (!reply || !request) {
      return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
    }

    rdc_status_t result = rdc_job_stop_stats(
                rdc_handle_,
                const_cast<char*>(request->job_id().c_str()));
    reply->set_status(result);

    return ::grpc::Status::OK;
}


::grpc::Status RdcAPIServiceImpl::RemoveJob(
                  ::grpc::ServerContext* context,
                  const ::rdc::RemoveJobRequest* request,
                  ::rdc::RemoveJobResponse* reply) {
    (void)(context);
    if (!reply || !request) {
      return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
    }
    rdc_status_t result = rdc_job_remove(
                rdc_handle_,  const_cast<char*>(request->job_id().c_str()));
    reply->set_status(result);

    return ::grpc::Status::OK;
}


::grpc::Status RdcAPIServiceImpl::RemoveAllJob(
                  ::grpc::ServerContext* context,
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



}  // namespace rdc
}  // namespace amd


