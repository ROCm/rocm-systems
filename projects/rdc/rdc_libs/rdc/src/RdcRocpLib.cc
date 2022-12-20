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
#include "rdc_lib/impl/RdcRocpLib.h"

#include <cstdint>
#include <functional>

#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/rdc_common.h"

namespace amd {
namespace rdc {

// TODO: Add init and destroy calls support
RdcRocpLib::RdcRocpLib(const char* lib_name)
    : telemetry_fields_query_(nullptr),
      telemetry_fields_value_get_(nullptr),
      telemetry_fields_watch_(nullptr),
      telemetry_fields_unwatch_(nullptr) {
    rdc_status_t status = lib_loader_.load(lib_name);
    if (status != RDC_ST_OK) {
        RDC_LOG(RDC_ERROR, "Rocp related function will not work.");
        return;
    }

    status = lib_loader_.load_symbol(
        &telemetry_fields_query_, "rdc_telemetry_fields_query");
    if (status != RDC_ST_OK) {
        telemetry_fields_query_ = nullptr;
    }

    status = lib_loader_.load_symbol(
        &telemetry_fields_value_get_, "rdc_telemetry_fields_value_get");
    if (status != RDC_ST_OK) {
        telemetry_fields_value_get_ = nullptr;
    }

    status = lib_loader_.load_symbol(
        &telemetry_fields_watch_, "rdc_telemetry_fields_watch");
    if (status != RDC_ST_OK) {
        telemetry_fields_watch_ = nullptr;
    }

    status = lib_loader_.load_symbol(
        &telemetry_fields_unwatch_, "rdc_telemetry_fields_unwatch");
    if (status != RDC_ST_OK) {
        telemetry_fields_unwatch_ = nullptr;
    }
}

RdcRocpLib::~RdcRocpLib() = default;

// get support field ids
rdc_status_t RdcRocpLib::rdc_telemetry_fields_query(
    uint32_t field_ids[MAX_NUM_FIELDS],
    uint32_t* field_count) {
    if (field_count == nullptr) {
        return RDC_ST_BAD_PARAMETER;
    }

    return telemetry_fields_query_(field_ids, field_count);
}

// Fetch
rdc_status_t RdcRocpLib::rdc_telemetry_fields_value_get(
    rdc_gpu_field_t* fields,
    uint32_t fields_count,
    rdc_field_value_f callback,
    void* user_data) {
    if (fields == nullptr) {
        return RDC_ST_BAD_PARAMETER;
    }

    RDC_LOG(RDC_DEBUG, "Fetch " << fields_count << " fields from rocp_lib.");

    return telemetry_fields_value_get_(
        fields, fields_count, callback, user_data);
}

rdc_status_t RdcRocpLib::rdc_telemetry_fields_watch(
    rdc_gpu_field_t* fields,
    uint32_t fields_count) {
    if (fields == nullptr) {
        return RDC_ST_BAD_PARAMETER;
    }
    return telemetry_fields_watch_(fields, fields_count);
}

rdc_status_t RdcRocpLib::rdc_telemetry_fields_unwatch(
    rdc_gpu_field_t* fields,
    uint32_t fields_count) {
    if (fields == nullptr) {
        return RDC_ST_BAD_PARAMETER;
    }
    return telemetry_fields_unwatch_(fields, fields_count);
}

}  // namespace rdc
}  // namespace amd
