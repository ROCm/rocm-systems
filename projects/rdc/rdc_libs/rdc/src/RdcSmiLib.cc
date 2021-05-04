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
#include <functional>
#include "rdc_lib/rdc_common.h"
#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/impl/RdcSmiLib.h"



namespace amd {
namespace rdc {

RdcSmiLib::RdcSmiLib(const RdcMetricFetcherPtr& mf): metric_fetcher_(mf) {
}

// Bulk fetch wrapper for the rocm_smi_lib. This will be replaced after
// rocm_smi_lib can support bulk fetch.
rdc_status_t RdcSmiLib::rdc_telemetry_fields_value_get(rdc_gpu_field_t* fields,
            uint32_t fields_count, rdc_field_value_f callback,
            void*  user_data) {
    if (fields == nullptr) {
        return RDC_ST_BAD_PARAMETER;
    }

    RDC_LOG(RDC_DEBUG, "Fetch " << fields_count
            << " fields from rocm_smi_lib.");

    // Bulk fetch fields
    std::vector<rdc_gpu_field_value_t> bulk_results;
    rdc_status_t status = metric_fetcher_->bulk_fetch_smi_fields(
        fields, fields_count, bulk_results);
    RDC_LOG(RDC_DEBUG, "Bulk fetched " << bulk_results.size()
            << " fields from rocm_smi_lib which return " << status);
    if (bulk_results.size() > 0) {
        rdc_status_t status = callback(&bulk_results[0],
                bulk_results.size(), user_data);
        if (status != RDC_ST_OK) {
                return status;
        }
    }

    // Fetch it one by one for left fields
    const int BULK_FIELDS_MAX = 16;
    rdc_gpu_field_value_t values[BULK_FIELDS_MAX];
    uint32_t bulk_count = 0;
    for (uint32_t i = 0; i < fields_count; i++) {
        bool is_fetched = false;
        for (std::size_t j = 0; j < bulk_results.size(); j++) {
            if (bulk_results[j].gpu_index == fields[i].gpu_index &&
                bulk_results[j].field_value.field_id == fields[i].field_id) {
                    is_fetched = true;
                    break;
                }
        }
        if (is_fetched) continue;
        if (bulk_count >= BULK_FIELDS_MAX) {
            rdc_status_t status = callback(values, bulk_count, user_data);
            // When the callback returns errors, stop processing and return.
            if (status != RDC_ST_OK) {
                return status;
            }
            bulk_count = 0;
        }
        values[bulk_count].gpu_index = fields[i].gpu_index;
        metric_fetcher_->fetch_smi_field(
            fields[i].gpu_index,
            static_cast<rdc_field_t>(fields[i].field_id),
            &(values[bulk_count].field_value));
        bulk_count++;
    }
    if (bulk_count != 0) {
        rdc_status_t status = callback(values, bulk_count, user_data);
        if (status != RDC_ST_OK) {
            return status;
        }
        bulk_count = 0;
    }

    return RDC_ST_OK;
}

rdc_status_t RdcSmiLib::rdc_telemetry_fields_watch(rdc_gpu_field_t* fields,
      uint32_t fields_count) {
  rdc_status_t ret;

    if (fields == nullptr) {
        return RDC_ST_BAD_PARAMETER;
    }

    for (uint32_t i = 0; i < fields_count; i++) {
        ret = metric_fetcher_->acquire_rsmi_handle(
            {fields[i].gpu_index, fields[i].field_id});
        if (ret != RDC_ST_OK) {
            RDC_LOG(RDC_ERROR,
                              "Failed to acquire rocm_smi handle for field.");
        }
    }
    RDC_LOG(RDC_DEBUG, "acquire " << fields_count
        << " field handles from rocm_smi_lib");

    return RDC_ST_OK;
}

rdc_status_t RdcSmiLib::rdc_telemetry_fields_unwatch(rdc_gpu_field_t* fields,
      uint32_t fields_count) {
    if (fields == nullptr) {
        return RDC_ST_BAD_PARAMETER;
    }

    for (uint32_t i = 0; i < fields_count; i++) {
        metric_fetcher_->delete_rsmi_handle(
            {fields[i].gpu_index, fields[i].field_id});
    }
    RDC_LOG(RDC_DEBUG, "delete " << fields_count
        << " field handles from rocm_smi_lib");

    return RDC_ST_OK;
}

rdc_status_t RdcSmiLib::rdc_telemetry_fields_query(
     uint32_t field_ids[MAX_NUM_FIELDS],
    uint32_t* field_count) {
    if (field_count == nullptr) {
        return RDC_ST_BAD_PARAMETER;
    }

    // List of fields supported by rocm_smi_lib
    const std::vector<uint32_t> fields{
            RDC_FI_GPU_COUNT, RDC_FI_DEV_NAME, RDC_FI_GPU_CLOCK,
            RDC_FI_MEM_CLOCK, RDC_FI_MEMORY_TEMP, RDC_FI_GPU_TEMP,
            RDC_FI_POWER_USAGE, RDC_FI_PCIE_TX, RDC_FI_PCIE_RX,
            RDC_FI_GPU_UTIL, RDC_FI_GPU_MEMORY_USAGE, RDC_FI_GPU_MEMORY_TOTAL,
            RDC_EVNT_XGMI_0_NOP_TX, RDC_EVNT_XGMI_0_REQ_TX,
            RDC_EVNT_XGMI_0_RESP_TX, RDC_EVNT_XGMI_0_BEATS_TX,
            RDC_EVNT_XGMI_1_NOP_TX, RDC_EVNT_XGMI_1_REQ_TX,
            RDC_EVNT_XGMI_1_RESP_TX, RDC_EVNT_XGMI_1_BEATS_TX,
            RDC_EVNT_XGMI_0_THRPUT, RDC_EVNT_XGMI_1_THRPUT,
            RDC_EVNT_XGMI_2_THRPUT, RDC_EVNT_XGMI_3_THRPUT,
            RDC_EVNT_XGMI_4_THRPUT, RDC_EVNT_XGMI_5_THRPUT,
    };
    std::copy(fields.begin(), fields.end(), field_ids);
    *field_count = fields.size();

    return RDC_ST_OK;
}

rdc_status_t RdcSmiLib::rdc_diag_test_cases_query(
        rdc_diag_test_cases_t test_cases[MAX_TEST_CASES],
        uint32_t* test_case_count) {
    if (test_case_count == nullptr) {
        return RDC_ST_BAD_PARAMETER;
    }

    return RDC_ST_NOT_SUPPORTED;
}

// Run a specific test case
rdc_status_t RdcSmiLib::rdc_test_case_run(
        rdc_diag_test_cases_t test_case,
        uint32_t gpu_index[RDC_MAX_NUM_DEVICES],
        uint32_t gpu_count,
        rdc_diag_test_result_t* result) {
    if (result == nullptr) {
        return RDC_ST_BAD_PARAMETER;
    }
    return RDC_ST_NOT_SUPPORTED;
}

rdc_status_t RdcSmiLib::rdc_diagnostic_run(
        const rdc_group_info_t&,
        rdc_diag_level_t,
        rdc_diag_response_t*) {
    return RDC_ST_NOT_SUPPORTED;
}

rdc_status_t RdcSmiLib::rdc_diag_init(uint64_t) {
    return RDC_ST_OK;
}

rdc_status_t RdcSmiLib::rdc_diag_destroy() {
    return RDC_ST_OK;
}

}  // namespace rdc
}  // namespace amd

