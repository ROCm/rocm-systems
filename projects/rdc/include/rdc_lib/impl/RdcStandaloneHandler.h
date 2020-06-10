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
#ifndef RDC_LIB_IMPL_RDCSTANDALONEHANDLER_H_
#define RDC_LIB_IMPL_RDCSTANDALONEHANDLER_H_
#include <grpcpp/grpcpp.h>
#include <memory>
#include "rdc.grpc.pb.h"  // NOLINT
#include "rdc_lib/RdcHandler.h"

namespace amd {
namespace rdc {

class RdcStandaloneHandler: public RdcHandler {
 public:
    // Job RdcAPI
    rdc_status_t rdc_job_start_stats(rdc_gpu_group_t groupId,
                        const char job_id[64], uint64_t update_freq) override;
    rdc_status_t rdc_job_get_stats(const char jobId[64],
                rdc_job_info_t* p_job_info) override;
    rdc_status_t rdc_job_stop_stats(const char job_id[64]) override;
    rdc_status_t rdc_job_remove(const char job_id[64]) override;
    rdc_status_t rdc_job_remove_all() override;

    // Discovery RdcAPI
    rdc_status_t rdc_device_get_all(
        uint32_t gpu_index_list[RDC_MAX_NUM_DEVICES], uint32_t* count) override;
    rdc_status_t rdc_device_get_attributes(uint32_t gpu_index,
        rdc_device_attributes_t* p_rdc_attr) override;

    // Group RdcAPI
    rdc_status_t rdc_group_gpu_create(rdc_group_type_t type,
                const char* group_name,
                rdc_gpu_group_t* p_rdc_group_id) override;
    rdc_status_t rdc_group_gpu_add(rdc_gpu_group_t groupId,
                uint32_t gpu_index) override;
    rdc_status_t rdc_group_field_create(uint32_t num_field_ids,
        uint32_t* field_ids, const char* field_group_name,
        rdc_field_grp_t* rdc_field_group_id) override;
    rdc_status_t rdc_group_field_get_info(
        rdc_field_grp_t rdc_field_group_id,
        rdc_field_group_info_t* field_group_info) override;
    rdc_status_t rdc_group_gpu_get_info(rdc_gpu_group_t p_rdc_group_id,
        rdc_group_info_t* p_rdc_group_info) override;
    rdc_status_t rdc_group_get_all_ids(
            rdc_gpu_group_t group_id_list[], uint32_t* count) override;
    rdc_status_t rdc_group_field_get_all_ids(
            rdc_field_grp_t field_group_id_list[], uint32_t* count) override;
    rdc_status_t rdc_group_gpu_destroy(
        rdc_gpu_group_t p_rdc_group_id) override;
    rdc_status_t rdc_group_field_destroy(
        rdc_field_grp_t rdc_field_group_id) override;

    // Field RdcAPI
    rdc_status_t rdc_field_watch(rdc_gpu_group_t group_id,
        rdc_field_grp_t field_group_id, uint64_t update_freq,
        double max_keep_age, uint32_t max_keep_samples) override;
    rdc_status_t rdc_field_get_latest_value(uint32_t gpu_index,
        uint32_t field, rdc_field_value* value) override;
    rdc_status_t rdc_field_get_value_since(uint32_t gpu_index,
        uint32_t field, uint64_t since_time_stamp,
        uint64_t *next_since_time_stamp, rdc_field_value* value) override;
    rdc_status_t rdc_field_unwatch(rdc_gpu_group_t group_id,
        rdc_field_grp_t field_group_id) override;

    // Control RdcAPI
    rdc_status_t rdc_field_update_all(uint32_t wait_for_update) override;

    explicit RdcStandaloneHandler(const char* ip_and_port,
     const char* root_ca, const char* client_cert, const char* client_key);

 private:
    // Helper function to handle the error
    rdc_status_t error_handle(::grpc::Status status, uint32_t rdc_status);

    bool copy_gpu_usage_info(
            const ::rdc::GpuUsageInfo& src,
            rdc_gpu_usage_info_t* target);

    std::unique_ptr<::rdc::RdcAPI::Stub> stub_;
};


}  // namespace rdc
}  // namespace amd

extern "C" {
    amd::rdc::RdcHandler *make_handler(const char* ip_port,
    const char* root_ca, const char* client_cert, const char* client_key);
}

#endif  // RDC_LIB_IMPL_RDCSTANDALONEHANDLER_H_
