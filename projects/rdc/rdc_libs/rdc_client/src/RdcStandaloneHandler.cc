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
#include "rdc.grpc.pb.h" // NOLINT

amd::rdc::RdcHandler *make_handler(const char* ip_and_port,
        const char* root_ca, const char* client_cert, const char* client_key) {
    return new amd::rdc::RdcStandaloneHandler(ip_and_port,
        root_ca, client_cert, client_key);
}

namespace amd {
namespace rdc {

RdcStandaloneHandler::RdcStandaloneHandler(const char* ip_and_port,
    const char* root_ca, const char* client_cert, const char* client_key) {
        std::shared_ptr<grpc::ChannelCredentials> cred(nullptr);
        if (root_ca == nullptr || client_cert == nullptr
         || client_key == nullptr) {
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


rdc_status_t RdcStandaloneHandler::error_handle(::grpc::Status status,
        uint32_t rdc_status) {
    if (!status.ok()) {
        std::cout<< status.error_message() <<". Error code:"
            << status.error_code() << std::endl;
        return RDC_ST_CLIENT_ERROR;
    }

    return static_cast<rdc_status_t>(rdc_status);
}

// JOB RdcAPI
rdc_status_t RdcStandaloneHandler::rdc_job_start_stats(rdc_gpu_group_t groupId,
        char job_id[64], uint64_t update_freq) {
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

bool RdcStandaloneHandler::copy_gpu_usage_info(
            const ::rdc::GpuUsageInfo& src,
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

    const ::rdc::JobStatsSummary& cstats = src.gpu_clock();
    target->gpu_clock.max_value = cstats.max_value();
    target->gpu_clock.min_value = cstats.min_value();
    target->gpu_clock.average = cstats.average();

    const ::rdc::JobStatsSummary& ustats = src.gpu_utilization();
    target->gpu_utilization.max_value = ustats.max_value();
    target->gpu_utilization.min_value = ustats.min_value();
    target->gpu_utilization.average = ustats.average();

    const ::rdc::JobStatsSummary& mstats = src.memory_utilization();
    target->memory_utilization.max_value = mstats.max_value();
    target->memory_utilization.min_value = mstats.min_value();
    target->memory_utilization.average = mstats.average();

    const ::rdc::JobStatsSummary& txstats = src.pcie_tx();
    target->pcie_tx.max_value = txstats.max_value();
    target->pcie_tx.min_value = txstats.min_value();
    target->pcie_tx.average = txstats.average();

    const ::rdc::JobStatsSummary& rxstats = src.pcie_rx();
    target->pcie_rx.max_value = rxstats.max_value();
    target->pcie_rx.min_value = rxstats.min_value();
    target->pcie_rx.average = rxstats.average();

    const ::rdc::JobStatsSummary& mcstats = src.memory_clock();
    target->memory_clock.max_value = mcstats.max_value();
    target->memory_clock.min_value = mcstats.min_value();
    target->memory_clock.average = mcstats.average();

    const ::rdc::JobStatsSummary& gtstats = src.gpu_temperature();
    target->gpu_temperature.max_value = gtstats.max_value();
    target->gpu_temperature.min_value = gtstats.min_value();
    target->gpu_temperature.average = gtstats.average();

    return true;
}
rdc_status_t RdcStandaloneHandler::rdc_job_get_stats(char job_id[64],
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

    return RDC_ST_OK;
}

rdc_status_t RdcStandaloneHandler::rdc_job_stop_stats(char job_id[64]) {
    ::rdc::StopJobStatsRequest request;
    ::rdc::StopJobStatsResponse reply;
    ::grpc::ClientContext context;

    request.set_job_id(job_id);
    ::grpc::Status status = stub_->StopJobStats(&context, request, &reply);
    rdc_status_t err_status = error_handle(status, reply.status());

    return err_status;
}

rdc_status_t RdcStandaloneHandler::rdc_job_remove(char job_id[64]) {
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
rdc_status_t RdcStandaloneHandler::rdc_device_get_all(
        uint32_t gpu_index_list[RDC_MAX_NUM_DEVICES], uint32_t* count)  {
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
    for (uint32_t i =0 ; i < *count; i++) {
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
    ::grpc::Status status = stub_->
                GetDeviceAttributes(&context, request, &reply);
    rdc_status_t err_status = error_handle(status, reply.status());
    if (err_status != RDC_ST_OK) return err_status;

    strncpy_with_null(p_rdc_attr->device_name,
        reply.attributes().device_name().c_str(), RDC_MAX_STR_LENGTH);


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

    request.set_type(
        static_cast<::rdc::CreateGpuGroupRequest_GpuGroupType>(type));
    request.set_group_name(group_name);
    ::grpc::Status status = stub_->
        CreateGpuGroup(&context, request, &reply);
    rdc_status_t err_status = error_handle(status, reply.status());
    if (err_status != RDC_ST_OK) return err_status;

    *p_rdc_group_id = reply.group_id();

    return RDC_ST_OK;
}

rdc_status_t RdcStandaloneHandler::rdc_group_gpu_add(rdc_gpu_group_t group_id,
                uint32_t gpu_index) {
    ::rdc::AddToGpuGroupRequest request;
    ::rdc::AddToGpuGroupResponse reply;
    ::grpc::ClientContext context;

    request.set_group_id(group_id);
    request.set_gpu_index(gpu_index);
    ::grpc::Status status = stub_->
        AddToGpuGroup(&context, request, &reply);
    rdc_status_t err_status = error_handle(status, reply.status());

    return err_status;
}

rdc_status_t RdcStandaloneHandler::rdc_group_field_create(
    uint32_t num_field_ids, uint32_t* field_ids,
    const char* field_group_name, rdc_field_grp_t* rdc_field_group_id) {
    if (!field_ids || !field_group_name || !rdc_field_group_id) {
        return RDC_ST_BAD_PARAMETER;
    }

    ::rdc::CreateFieldGroupRequest request;
    ::rdc::CreateFieldGroupResponse reply;
    ::grpc::ClientContext context;

    request.set_field_group_name(field_group_name);
    for (uint32_t i = 0; i < num_field_ids; i++){
         request.add_field_ids(field_ids[i]);
    }

    ::grpc::Status status = stub_->
        CreateFieldGroup(&context, request, &reply);
    rdc_status_t err_status = error_handle(status, reply.status());
    if (err_status != RDC_ST_OK) return err_status;
    *rdc_field_group_id = reply.field_group_id();

    return RDC_ST_OK;
}

rdc_status_t RdcStandaloneHandler::rdc_group_field_get_info(
        rdc_field_grp_t rdc_field_group_id,
        rdc_field_group_info_t* field_group_info) {
    if (!field_group_info) {
         return RDC_ST_BAD_PARAMETER;
    }

    ::rdc::GetFieldGroupInfoRequest request;
    ::rdc::GetFieldGroupInfoResponse reply;
    ::grpc::ClientContext context;

    request.set_field_group_id(rdc_field_group_id);
    ::grpc::Status status = stub_->
        GetFieldGroupInfo(&context, request, &reply);
    rdc_status_t err_status = error_handle(status, reply.status());
    if (err_status != RDC_ST_OK) return err_status;

    if (reply.field_ids_size() > RDC_MAX_FIELD_IDS_PER_FIELD_GROUP) {
        return RDC_ST_MAX_LIMIT;
    }

    field_group_info->count = reply.field_ids_size();
    strncpy_with_null(field_group_info->group_name,
                reply.filed_group_name().c_str(), RDC_MAX_STR_LENGTH);
    for (int i = 0; i < reply.field_ids_size(); i++) {
        field_group_info->field_ids[i] = reply.field_ids(i);
    }

    return RDC_ST_OK;
}

rdc_status_t RdcStandaloneHandler::rdc_group_gpu_get_info(
        rdc_gpu_group_t p_rdc_group_id,
        rdc_group_info_t* p_rdc_group_info) {
    if (!p_rdc_group_info) {
         return RDC_ST_BAD_PARAMETER;
    }

    ::rdc::GetGpuGroupInfoRequest request;
    ::rdc::GetGpuGroupInfoResponse reply;
    ::grpc::ClientContext context;

    request.set_group_id(p_rdc_group_id);
    ::grpc::Status status = stub_->
        GetGpuGroupInfo(&context, request, &reply);
    rdc_status_t err_status = error_handle(status, reply.status());
    if (err_status != RDC_ST_OK) return err_status;

    if (reply.entity_ids_size() > RDC_GROUP_MAX_ENTITIES) {
        return RDC_ST_MAX_LIMIT;
    }

    p_rdc_group_info->count = reply.entity_ids_size();
    strncpy_with_null(p_rdc_group_info->group_name,
                reply.group_name().c_str(), RDC_MAX_STR_LENGTH);
    for (int i = 0; i < reply.entity_ids_size(); i++) {
        p_rdc_group_info->entity_ids[i] = reply.entity_ids(i);
    }

    return RDC_ST_OK;
}

rdc_status_t RdcStandaloneHandler::rdc_group_get_all_ids(
    rdc_gpu_group_t group_id_list[], uint32_t* count) {
    if (!count) {
        return RDC_ST_BAD_PARAMETER;
    }
    ::rdc::Empty request;
    ::rdc::GetGroupAllIdsResponse reply;
    ::grpc::ClientContext context;

    ::grpc::Status status = stub_->
        GetGroupAllIds(&context, request, &reply);
    rdc_status_t err_status = error_handle(status, reply.status());
    if (err_status != RDC_ST_OK) return err_status;

    *count = reply.group_ids_size();
    if (*count >= RDC_MAX_NUM_GROUPS) {
            return RDC_ST_MAX_LIMIT;
    }
    for (uint32_t i =0 ; i < *count; i++) {
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

    ::grpc::Status status = stub_->
        GetFieldGroupAllIds(&context, request, &reply);
    rdc_status_t err_status = error_handle(status, reply.status());
    if (err_status != RDC_ST_OK) return err_status;

    *count = reply.field_group_ids_size();
    if (*count >= RDC_MAX_NUM_FIELD_GROUPS) {
            return RDC_ST_MAX_LIMIT;
    }
    for (uint32_t i =0 ; i < *count; i++) {
        field_group_id_list[i] = reply.field_group_ids(i);
    }

    return RDC_ST_OK;
}

rdc_status_t RdcStandaloneHandler::rdc_group_gpu_destroy(
        rdc_gpu_group_t p_rdc_group_id) {
    ::rdc::DestroyGpuGroupRequest request;
    ::rdc::DestroyGpuGroupResponse reply;
    ::grpc::ClientContext context;

    request.set_group_id(p_rdc_group_id);
    ::grpc::Status status = stub_->
        DestroyGpuGroup(&context, request, &reply);
    return error_handle(status, reply.status());
}

rdc_status_t RdcStandaloneHandler::rdc_group_field_destroy(
        rdc_field_grp_t rdc_field_group_id) {
    ::rdc::DestroyFieldGroupRequest request;
    ::rdc::DestroyFieldGroupResponse reply;
    ::grpc::ClientContext context;

    request.set_field_group_id(rdc_field_group_id);
    ::grpc::Status status = stub_->
        DestroyFieldGroup(&context, request, &reply);
    return error_handle(status, reply.status());
}

// Field RdcAPI
rdc_status_t RdcStandaloneHandler::rdc_field_watch(rdc_gpu_group_t group_id,
        rdc_field_grp_t field_group_id, uint64_t update_freq,
        double max_keep_age, uint32_t max_keep_samples) {
    ::rdc::WatchFieldsRequest request;
    ::rdc::WatchFieldsResponse reply;
    ::grpc::ClientContext context;

    request.set_group_id(group_id);
    request.set_field_group_id(field_group_id);
    request.set_update_freq(update_freq);
    request.set_max_keep_age(max_keep_age);
    request.set_max_keep_samples(max_keep_samples);
    ::grpc::Status status = stub_->
        WatchFields(&context, request, &reply);

    return error_handle(status, reply.status());
}

rdc_status_t RdcStandaloneHandler::rdc_field_get_latest_value(
    uint32_t gpu_index, uint32_t field, rdc_field_value* value) {
    if (!value) {
        return RDC_ST_BAD_PARAMETER;
    }

    ::rdc::GetLatestFieldValueRequest request;
    ::rdc::GetLatestFieldValueResponse reply;
    ::grpc::ClientContext context;

    request.set_gpu_index(gpu_index);
    request.set_field_id(field);
    ::grpc::Status status = stub_->
        GetLatestFieldValue(&context, request, &reply);
    rdc_status_t err_status = error_handle(status, reply.status());
    if (err_status != RDC_ST_OK) return err_status;

    value->field_id = reply.field_id();
    value->status = reply.rdc_status();
    value->ts = reply.ts();
    value->type = static_cast<rdc_field_type_t>(reply.type());
    if (value->type == INTEGER) {
        value->value.l_int = reply.l_int();
    } else if (value->type == DOUBLE) {
        value->value.dbl = reply.dbl();
    } else if (value->type == STRING || value->type == BLOB) {
        strncpy_with_null(value->value.str,
            reply.str().c_str(), RDC_MAX_STR_LENGTH);
    }

    return RDC_ST_OK;
}

rdc_status_t RdcStandaloneHandler::rdc_field_get_value_since(uint32_t gpu_index,
        uint32_t field, uint64_t since_time_stamp,
        uint64_t *next_since_time_stamp, rdc_field_value* value) {
    if (!next_since_time_stamp || !value) {
        return RDC_ST_BAD_PARAMETER;
    }

    ::rdc::GetFieldSinceRequest request;
    ::rdc::GetFieldSinceResponse reply;
    ::grpc::ClientContext context;

    request.set_gpu_index(gpu_index);
    request.set_field_id(field);
    request.set_since_time_stamp(since_time_stamp);
    ::grpc::Status status = stub_->
        GetFieldSince(&context, request, &reply);
    rdc_status_t err_status = error_handle(status, reply.status());
    if (err_status != RDC_ST_OK) return err_status;

    value->field_id = reply.field_id();
    value->status = reply.rdc_status();
    value->ts = reply.ts();
    value->type = static_cast<rdc_field_type_t>(reply.type());
    if (value->type == INTEGER) {
        value->value.l_int = reply.l_int();
    } else if (value->type == DOUBLE) {
        value->value.dbl = reply.dbl();
    } else if (value->type == STRING || value->type == BLOB) {
        strncpy_with_null(value->value.str,
            reply.str().c_str(), RDC_MAX_STR_LENGTH);
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
    ::grpc::Status status = stub_->
        UnWatchFields(&context, request, &reply);

    return error_handle(status, reply.status());
}


// Control RdcAPI
rdc_status_t RdcStandaloneHandler::rdc_field_update_all(
    uint32_t wait_for_update) {
    ::rdc::UpdateAllFieldsRequest request;
    ::rdc::UpdateAllFieldsResponse reply;
    ::grpc::ClientContext context;

    request.set_wait_for_update(wait_for_update);
    ::grpc::Status status = stub_->
        UpdateAllFields(&context, request, &reply);

    return error_handle(status, reply.status());
}



}  // namespace rdc
}  // namespace amd
