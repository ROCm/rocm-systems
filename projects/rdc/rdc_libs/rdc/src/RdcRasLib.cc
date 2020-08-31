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
#include "rdc_lib/impl/RdcRasLib.h"

namespace amd {
namespace rdc {

RdcRasLib::RdcRasLib(const char* lib_name) {
    rdc_status_t status = lib_loader_.load(lib_name);
    if (status != RDC_ST_OK) {
        fields_value_get_ = nullptr;
        fields_query_ = nullptr;
        return;
    }

    status = lib_loader_.load_symbol(&fields_value_get_,
                "rdc_telemetry_fields_value_get");
    if (status != RDC_ST_OK) {
        fields_value_get_ = nullptr;
    }
    status = lib_loader_.load_symbol(&fields_query_,
                "rdc_telemetry_fields_query");
    if (status != RDC_ST_OK) {
        fields_query_ = nullptr;
    }
}

rdc_status_t RdcRasLib::rdc_telemetry_fields_query(
    uint32_t field_ids[MAX_NUM_FIELDS], uint32_t* field_count) {
    if (field_count == nullptr) {
        return RDC_ST_BAD_PARAMETER;
    }
    if (!fields_query_) {
        *field_count = 0;
        return RDC_ST_FAIL_LOAD_MODULE;
    }

    RDC_LOG(RDC_DEBUG, "RAS support " << *field_count << " fields");
    return fields_query_(field_ids, field_count);
}

rdc_status_t RdcRasLib::rdc_telemetry_fields_value_get(
    rdc_gpu_field_t* fields, uint32_t fields_count, rdc_field_value_f callback,
    void*  user_data) {
    if (fields == nullptr) {
        return RDC_ST_BAD_PARAMETER;
    }
    if (!fields_value_get_) {
        return RDC_ST_FAIL_LOAD_MODULE;
    }
    rdc_status_t status = fields_value_get_(fields,
                fields_count, callback, user_data);
    RDC_LOG(RDC_DEBUG, "Bulk fetched " << fields_count << " fields from RAS: "
                    << rdc_status_string(status));
    return status;
}

rdc_status_t RdcRasLib::rdc_telemetry_fields_watch(rdc_gpu_field_t* fields,
      uint32_t fields_count) {
    // TODO(bill_liu): Not Support yet
    if (fields == nullptr) {
        return RDC_ST_BAD_PARAMETER;
    }
    (void)fields;
    (void)fields_count;
    return RDC_ST_NOT_SUPPORTED;
}

rdc_status_t RdcRasLib::rdc_telemetry_fields_unwatch(rdc_gpu_field_t* fields,
      uint32_t fields_count) {
    // TODO(bill_liu): Not Support yet
    if (fields == nullptr) {
        return RDC_ST_BAD_PARAMETER;
    }
    (void)fields;
    (void)fields_count;
    return RDC_ST_NOT_SUPPORTED;
}

}  // namespace rdc
}  // namespace amd

