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


amd::rdc::RdcHandler *make_handler(const char* ip_and_port) {
    return new amd::rdc::RdcStandaloneHandler(ip_and_port);
}

namespace amd {
namespace rdc {

RdcStandaloneHandler::RdcStandaloneHandler(const char* ip_and_port):
     stub_(::rdc::RdcAPI::NewStub(grpc::CreateChannel(ip_and_port,
     grpc::InsecureChannelCredentials()))) {
}

rdc_status_t RdcStandaloneHandler::error_handle(::grpc::Status status,
        uint32_t rdc_status) {
    if (!status.ok()) {
        std::cout<< status.error_message() <<". Error code:"
            << status.error_code() << std::endl;
        return RDC_ST_CLIENT_ERROR;
    }

    if (rdc_status != RDC_ST_OK) {
        return static_cast<rdc_status_t>(rdc_status);
    }
    return RDC_ST_OK;
}

// JOB RdcAPI
rdc_status_t RdcStandaloneHandler::rdc_job_start_stats(rdc_gpu_group_t groupId,
        char job_id[64], uint64_t update_freq, double  max_keep_age,
        uint32_t  max_keep_samples) {
    // TODO(bill_liu): implement
    (void)(groupId);
    (void)(job_id);
    (void)(update_freq);
    (void)(max_keep_age);
    (void)(max_keep_samples);

    return RDC_ST_OK;
}

rdc_status_t RdcStandaloneHandler::rdc_job_get_stats(char  job_id[64],
           rdc_job_info_t* p_job_info) {
    // TODO(bill_liu): implement
    (void)(job_id);
    (void)(p_job_info);
    return RDC_ST_OK;
}

rdc_status_t RdcStandaloneHandler::rdc_job_stop_stats(char  job_id[64] ) {
    // TODO(bill_liu): implement
    (void)(job_id);
    return RDC_ST_OK;
}


// Discovery RdcAPI
rdc_status_t RdcStandaloneHandler::rdc_get_all_devices(
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

rdc_status_t RdcStandaloneHandler::rdc_get_device_attributes(uint32_t gpu_index,
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
    // TODO(bill_liu): implement
    (void)(type);
    (void)(group_name);
    (void)(p_rdc_group_id);
    return RDC_ST_OK;
}

rdc_status_t RdcStandaloneHandler::rdc_group_gpu_add(rdc_gpu_group_t group_id,
                uint32_t gpu_index) {
    // TODO(bill_liu): implement
    (void)(group_id);
    (void)(gpu_index);
    return RDC_ST_OK;
}

rdc_status_t RdcStandaloneHandler::rdc_group_field_create(
    uint32_t num_field_ids, uint32_t* field_ids,
    const char* field_group_name, rdc_field_grp_t* rdc_field_group_id) {
    // TODO(bill_liu): implement
    (void)(num_field_ids);
    (void)(field_ids);
    (void)(field_group_name);
    (void)(rdc_field_group_id);
    return RDC_ST_OK;
}

rdc_status_t RdcStandaloneHandler::rdc_group_field_get_info(
        rdc_field_grp_t rdc_field_group_id,
        rdc_field_group_info_t* field_group_info) {
    // TODO(bill_liu): implement
    (void)(rdc_field_group_id);
    (void)(field_group_info);
    return RDC_ST_OK;
}

rdc_status_t RdcStandaloneHandler::rdc_group_gpu_get_info(
        rdc_gpu_group_t p_rdc_group_id,
        rdc_group_info_t* p_rdc_group_info) {
    // TODO(bill_liu): implement
    (void)(p_rdc_group_id);
    (void)(p_rdc_group_info);
    return RDC_ST_OK;
}

rdc_status_t RdcStandaloneHandler::rdc_group_gpu_destroy(
        rdc_gpu_group_t p_rdc_group_id) {
    // TODO(bill_liu): implement
    (void)(p_rdc_group_id);
    return RDC_ST_OK;
}

rdc_status_t RdcStandaloneHandler::rdc_group_field_destroy(
        rdc_field_grp_t rdc_field_group_id) {
    // TODO(bill_liu): implement
    (void)(rdc_field_group_id);
    return RDC_ST_OK;
}

// Field RdcAPI
rdc_status_t RdcStandaloneHandler::rdc_watch_fields(rdc_gpu_group_t group_id,
        rdc_field_grp_t field_group_id, uint64_t update_freq,
        double max_keep_age, uint32_t max_keep_samples) {
    // TODO(bill_liu): implement
    (void)(group_id);
    (void)(field_group_id);
    (void)(update_freq);
    (void)(max_keep_age);
    (void)(max_keep_samples);
    return RDC_ST_OK;
}

rdc_status_t RdcStandaloneHandler::rdc_get_latest_value_for_field(
    uint32_t gpu_index, uint32_t field, rdc_field_value* value) {
    // TODO(bill_liu): implement
    if (!value) {
        return RDC_ST_BAD_PARAMETER;
    }
    (void)(gpu_index);
    (void)(field);
    return RDC_ST_NOT_FOUND;
}

rdc_status_t RdcStandaloneHandler::rdc_get_field_value_since(uint32_t gpu_index,
        uint32_t field, uint64_t since_time_stamp,
        uint64_t *next_since_time_stamp, rdc_field_value* value) {
    // TODO(bill_liu): implement
    if (!next_since_time_stamp || !value) {
        return RDC_ST_BAD_PARAMETER;
    }
    (void)(since_time_stamp);
    (void)(gpu_index);
    (void)(field);
    (void)(value);

    return RDC_ST_NOT_FOUND;
}

rdc_status_t RdcStandaloneHandler::rdc_unwatch_fields(rdc_gpu_group_t group_id,
        rdc_field_grp_t field_group_id) {
    // TODO(bill_liu): implement
    (void)(group_id);
    (void)(field_group_id);
    return RDC_ST_OK;
}


// Control RdcAPI
rdc_status_t RdcStandaloneHandler::rdc_update_all_fields(
    uint32_t wait_for_update) {
    // TODO(bill_liu): implement
    (void)(wait_for_update);
    return RDC_ST_OK;
}



}  // namespace rdc
}  // namespace amd
