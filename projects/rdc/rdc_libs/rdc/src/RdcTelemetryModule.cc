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
#include "rdc_lib/impl/RdcTelemetryModule.h"
#include <functional>
#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/impl/RdcSmiLib.h"

namespace amd {
namespace rdc {

// Return all supported fields
rdc_status_t RdcTelemetryModule::rdc_telemetry_fields_query(
    uint32_t field_ids[MAX_NUM_FIELDS], uint32_t* field_count) {
    if (field_count == nullptr) {
        return RDC_ST_BAD_PARAMETER;
    }
    auto ite = telemetry_modules_.begin();
    *field_count = 0;
    uint32_t count = 0;
    for (; ite != telemetry_modules_.end(); ite++) {
        rdc_status_t status = (*ite)->rdc_telemetry_fields_query(
                &(field_ids[count]), &count);
        if (status == RDC_ST_OK) {
            *field_count += count;
        }
    }

    return RDC_ST_OK;
}

rdc_status_t RdcTelemetryModule::rdc_telemetry_fields_watch(
    rdc_gpu_field_t* fields, uint32_t fields_count) {
    if (fields == nullptr) {
        return RDC_ST_BAD_PARAMETER;
    }
    auto ite = telemetry_modules_.begin();
    for (; ite != telemetry_modules_.end(); ite++) {
       (*ite)->rdc_telemetry_fields_watch(
           fields, fields_count);
    }
    return RDC_ST_OK;
}


rdc_status_t RdcTelemetryModule::rdc_telemetry_fields_unwatch(
    rdc_gpu_field_t* fields, uint32_t fields_count) {
    if (fields == nullptr) {
        return RDC_ST_BAD_PARAMETER;
    }
    auto ite = telemetry_modules_.begin();
    for (; ite != telemetry_modules_.end(); ite++) {
       (*ite)->rdc_telemetry_fields_unwatch(
           fields, fields_count);
    }
    return RDC_ST_OK;
}

RdcTelemetryModule::RdcTelemetryModule(
    const RdcMetricFetcherPtr& fetcher,
    const RdcRasLibPtr& ras_module) {
    auto smi_telemetry_module = std::make_shared<RdcSmiLib>(fetcher);
    telemetry_modules_.push_back(smi_telemetry_module);
    if (ras_module) {
       telemetry_modules_.push_back(ras_module);
    }

    auto ite = telemetry_modules_.begin();
    for (; ite != telemetry_modules_.end(); ite++) {
       uint32_t field_ids[MAX_NUM_FIELDS];
       uint32_t field_count;
       (*ite)->rdc_telemetry_fields_query(field_ids, &field_count);
       for (uint32_t index = 0; index < field_count; index++) {
           fields_id_module_.insert({field_ids[index], (*ite)});
       }
    }
}

rdc_status_t RdcTelemetryModule::rdc_telemetry_fields_value_get(
    rdc_gpu_field_t* fields, uint32_t fields_count,
    rdc_field_value_f callback, void*  user_data) {
    if (fields == nullptr) {
        return RDC_ST_BAD_PARAMETER;
    }

    // Dispatch the fields to the libraries
    std::map<RdcTelemetryPtr, std::vector<rdc_gpu_field_t>> fields_to_fetch;
    std::vector<rdc_gpu_field_value_t> unsupport_fields;
    for (uint32_t findex = 0; findex < fields_count; findex++) {
        RdcTelemetryPtr module = fields_id_module_[fields[findex].field_id];
        if (module) {
            fields_to_fetch[module].push_back(fields[findex]);
        } else {
            RDC_LOG(RDC_DEBUG, "Unsupported field " <<
                    field_id_string(fields[findex].field_id));
            rdc_gpu_field_value_t value;
            value.gpu_index = fields[findex].gpu_index;
            value.field_value.field_id = fields[findex].field_id;
            value.field_value.status = RDC_ST_NOT_SUPPORTED;
            unsupport_fields.push_back(value);
        }
    }

    auto ite = fields_to_fetch.begin();
    for (; ite != fields_to_fetch.end(); ite ++) {
        rdc_gpu_field_t f[MAX_NUM_FIELDS];
        std::copy(ite->second.begin(), ite->second.end(), f);
        ite->first->rdc_telemetry_fields_value_get(f,
            ite->second.size(), callback, user_data);
    }

    // Notify the caller unsupported fields
    callback(&unsupport_fields[0],  unsupport_fields.size(), user_data);

    return RDC_ST_OK;
}

}  // namespace rdc
}  // namespace amd

