/*
Copyright (c) 2022 - present Advanced Micro Devices, Inc. All rights reserved.

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

#include <sys/time.h>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

#include "rdc/rdc.h"
#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/RdcTelemetryLibInterface.h"
#include "rdc_lib/rdc_common.h"
#include "rdc_modules/rdc_rocp/RdcRocpBase.h"

amd::rdc::RdcRocpBase rocp;

// get supported field ids
// TODO: Query fields with rocprofiler
rdc_status_t rdc_telemetry_fields_query(
    uint32_t field_ids[MAX_NUM_FIELDS],
    uint32_t* field_count) {
    // extract all keys from counter_map
    std::vector<uint32_t> counter_keys;
    counter_keys.reserve(amd::rdc::counter_map_k.size());
    for (auto it : amd::rdc::counter_map_k) {
        counter_keys.push_back(it.first);
    }

    *field_count = counter_keys.size();
    // copy from vector into array
    std::copy(counter_keys.begin(), counter_keys.end(), field_ids);

    return RDC_ST_OK;
}

// Fetch
rdc_status_t rdc_telemetry_fields_value_get(
    rdc_gpu_field_t* fields,
    uint32_t fields_count,
    rdc_field_value_f callback,
    void* user_data) {
    //
    // Bulk fetch fields
    std::vector<rdc_gpu_field_value_t> bulk_results;

    struct timeval tv {};
    gettimeofday(&tv, nullptr);
    const uint64_t curTime =
        static_cast<uint64_t>(tv.tv_sec) * 1000 + tv.tv_usec / 1000;

    // Fetch it one by one for left fields
    const int BULK_FIELDS_MAX = 16;
    rdc_gpu_field_value_t values[BULK_FIELDS_MAX];
    uint32_t bulk_count = 0;
    rdc_status_t status = RDC_ST_UNKNOWN_ERROR;
    double value = 0;
    for (uint32_t i = 0; i < fields_count; i++) {
        if (bulk_count >= BULK_FIELDS_MAX) {
            status = callback(values, bulk_count, user_data);
            // When the callback returns errors, stop processing and return.
            if (status != RDC_ST_OK) {
                return status;
            }
            bulk_count = 0;
        }

        RDC_LOG(RDC_DEBUG, "ID: " << fields[i].field_id);

        status = rocp.rocp_lookup(
            std::make_pair(fields[i].gpu_index, fields[i].field_id), &value);

        // get value
        values[bulk_count].gpu_index = fields[i].gpu_index;
        values[bulk_count].field_value.type = DOUBLE;
        values[bulk_count].field_value.status = status;
        values[bulk_count].field_value.ts = curTime;
        values[bulk_count].field_value.value.dbl = value;
        values[bulk_count].field_value.field_id = fields[i].field_id;
        RDC_LOG(RDC_DEBUG, "VALUE: " << value);
        bulk_count++;
    }
    if (bulk_count != 0) {
        rdc_status_t status = callback(values, bulk_count, user_data);
        if (status != RDC_ST_OK) {
            return status;
        }
        bulk_count = 0;
    }

    return status;
}

rdc_status_t rdc_telemetry_fields_watch(
    rdc_gpu_field_t* fields,
    uint32_t fields_count) {
    rdc_status_t status = RDC_ST_OK;
    for (uint32_t i = 0; i < fields_count; i++) {
        RDC_LOG(RDC_DEBUG, "WATCH: " << fields[i].field_id);
        const rdc_status_t temp_status = rocp.create_session(
            std::make_pair(fields[i].gpu_index, fields[i].field_id));
        if (temp_status != RDC_ST_OK) {
            status = temp_status;
        }
    }
    return status;
}

rdc_status_t rdc_telemetry_fields_unwatch(
    rdc_gpu_field_t* fields,
    uint32_t fields_count) {
    rdc_status_t status = RDC_ST_OK;
    for (uint32_t i = 0; i < fields_count; i++) {
        RDC_LOG(RDC_DEBUG, "UNWATCH: " << fields[i].field_id);
        const rdc_status_t temp_status = rocp.destroy_session(
            std::make_pair(fields[i].gpu_index, fields[i].field_id));
        // return last non-ok status
        if (temp_status != RDC_ST_OK) {
            status = temp_status;
        }
    }
    return status;
}
