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
#include "rdc_lib/impl/RdcEmbeddedHandler.h"
#include <string.h>
#include "rdc_lib/impl/RdcMetricFetcherImpl.h"
#include "rdc_lib/impl/RdcGroupSettingsImpl.h"
#include "rdc_lib/rdc_common.h"
#include "rocm_smi/rocm_smi.h"

namespace {
// call the rsmi_init when load library
// and rsmi_shutdown when unload the library.
class rsmi_initializer {
         rsmi_initializer() { rsmi_init(0);}
         ~rsmi_initializer() { rsmi_shut_down();}
 public:
         static rsmi_initializer& getInstance() {
             static rsmi_initializer instance;
             return instance;
         }
};

static rsmi_initializer& in = rsmi_initializer::getInstance();
}  // namespace

amd::rdc::RdcHandler *make_handler(rdc_operation_mode_t op_mode) {
     return new amd::rdc::RdcEmbeddedHandler(op_mode);
}

namespace amd {
namespace rdc {


RdcEmbeddedHandler::RdcEmbeddedHandler(rdc_operation_mode_t mode) :
    group_settings_(new RdcGroupSettingsImpl())
    , metric_fetcher_(new RdcMetricFetcherImpl()) {
    // TODO(bill_liu): implement the operation mode
    (void)(mode);
}

// JOB API
rdc_status_t RdcEmbeddedHandler::rdc_job_start_stats(rdc_gpu_group_t groupId,
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

rdc_status_t RdcEmbeddedHandler::rdc_job_get_stats(char  job_id[64],
           rdc_job_info_t* p_job_info) {
    // TODO(bill_liu): implement
    (void)(job_id);
    (void)(p_job_info);
    return RDC_ST_OK;
}

rdc_status_t RdcEmbeddedHandler::rdc_job_stop_stats(char  job_id[64] ) {
    // TODO(bill_liu): implement
    (void)(job_id);
    return RDC_ST_OK;
}


// Discovery API
rdc_status_t RdcEmbeddedHandler::rdc_get_all_devices(
        uint32_t gpu_index_list[RDC_MAX_NUM_DEVICES], uint32_t* count)  {
    if (!count) {
        return RDC_ST_BAD_PARAMETER;
    }
    rdc_field_value device_count;
    rdc_status_t status = metric_fetcher_->
        fetch_smi_field(0, RDC_FI_GPU_COUNT, &device_count);
    if (status != RDC_ST_OK) {
        return status;
    }

    // Assign the index to the index list
    *count = device_count.value.l_int;
    for (uint32_t i=0; i < *count; i++) {
        gpu_index_list[i] = i;
    }

    return RDC_ST_OK;
}

rdc_status_t RdcEmbeddedHandler::rdc_get_device_attributes(uint32_t gpu_index,
        rdc_device_attributes_t* p_rdc_attr) {
    if (!p_rdc_attr) {
        return RDC_ST_BAD_PARAMETER;
    }
    rdc_field_value device_name;
    rdc_status_t status = metric_fetcher_->
        fetch_smi_field(gpu_index, RDC_FI_DEV_NAME, &device_name);
    strncpy_with_null(p_rdc_attr->device_name, device_name.value.str,
                RDC_MAX_STR_LENGTH);
    return status;
}


    // Group API
rdc_status_t RdcEmbeddedHandler::rdc_group_gpu_create(rdc_group_type_t type,
                const char* group_name,
                rdc_gpu_group_t* p_rdc_group_id) {
    if (!group_name || !p_rdc_group_id) {
        return RDC_ST_BAD_PARAMETER;
    }
    return group_settings_->
        rdc_group_gpu_create(type, group_name, p_rdc_group_id);
}

rdc_status_t RdcEmbeddedHandler::rdc_group_gpu_add(rdc_gpu_group_t group_id,
                uint32_t gpu_index) {
    return group_settings_->rdc_group_gpu_add(group_id, gpu_index);
}

rdc_status_t RdcEmbeddedHandler::rdc_group_field_create(uint32_t num_field_ids,
        uint32_t* field_ids, const char* field_group_name,
        rdc_field_grp_t* rdc_field_group_id) {
    if (!field_group_name || !rdc_field_group_id || !field_ids) {
        return RDC_ST_BAD_PARAMETER;
    }
    return group_settings_->rdc_group_field_create(
        num_field_ids, field_ids, field_group_name, rdc_field_group_id);
}

rdc_status_t RdcEmbeddedHandler::rdc_group_field_get_info(
        rdc_field_grp_t rdc_field_group_id,
        rdc_field_group_info_t* field_group_info) {
    if (!field_group_info) {
        return RDC_ST_BAD_PARAMETER;
    }
    return group_settings_->rdc_group_field_get_info(
            rdc_field_group_id, field_group_info);
}

rdc_status_t RdcEmbeddedHandler::rdc_group_gpu_get_info(
        rdc_gpu_group_t p_rdc_group_id,
        rdc_group_info_t* p_rdc_group_info) {
    if (!p_rdc_group_info) {
        return RDC_ST_BAD_PARAMETER;
    }

    return group_settings_->rdc_group_gpu_get_info(
            p_rdc_group_id, p_rdc_group_info);
}

rdc_status_t RdcEmbeddedHandler::rdc_group_gpu_destroy(
        rdc_gpu_group_t p_rdc_group_id) {
    return group_settings_->rdc_group_gpu_destroy(p_rdc_group_id);
}

rdc_status_t RdcEmbeddedHandler::rdc_group_field_destroy(
        rdc_field_grp_t rdc_field_group_id) {
    return group_settings_->rdc_group_field_destroy(rdc_field_group_id);
}

// Field API
rdc_status_t RdcEmbeddedHandler::rdc_watch_fields(rdc_gpu_group_t group_id,
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

rdc_status_t RdcEmbeddedHandler::rdc_get_latest_value_for_field(
    uint32_t gpu_index, uint32_t field, rdc_field_value* value) {
    // TODO(bill_liu): implement
    if (!value) {
        return RDC_ST_BAD_PARAMETER;
    }
    (void)(gpu_index);
    (void)(field);
    return RDC_ST_NOT_FOUND;
}

rdc_status_t RdcEmbeddedHandler::rdc_get_field_value_since(uint32_t gpu_index,
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

rdc_status_t RdcEmbeddedHandler::rdc_unwatch_fields(rdc_gpu_group_t group_id,
        rdc_field_grp_t field_group_id) {
    // TODO(bill_liu): implement
    (void)(group_id);
    (void)(field_group_id);
    return RDC_ST_OK;
}


// Control API
rdc_status_t RdcEmbeddedHandler::rdc_update_all_fields(
    uint32_t wait_for_update) {
    // TODO(bill_liu): implement
    (void)(wait_for_update);
    return RDC_ST_OK;
}

}  // namespace rdc
}  // namespace amd
