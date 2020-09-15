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

RdcRasLib::RdcRasLib(const char* lib_name):
    fields_value_get_(nullptr)
    , fields_query_(nullptr)
    , fields_watch_(nullptr)
    , fields_unwatch_(nullptr)
    , rdc_module_init_(nullptr)
    , rdc_module_destroy_(nullptr) {
    rdc_status_t status = lib_loader_.load(lib_name);
    if (status != RDC_ST_OK) {
        return;
    }

    status = lib_loader_.load_symbol(&rdc_module_init_,
                "rdc_module_init");
    if (status != RDC_ST_OK) {
        rdc_module_init_ = nullptr;
        return;
    }

    if (rdc_module_init_(0) != RDC_ST_OK) {
        RDC_LOG(RDC_ERROR, "Fail to init librdc_ras.so:"
                    << rdc_status_string(status));
         return;
    }


    status = lib_loader_.load_symbol(&rdc_module_destroy_,
                "rdc_module_destroy");
    if (status != RDC_ST_OK) {
        rdc_module_destroy_ = nullptr;
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
    status = lib_loader_.load_symbol(&fields_watch_,
                "rdc_telemetry_fields_watch");
    if (status != RDC_ST_OK) {
        fields_watch_ = nullptr;
    }
    status = lib_loader_.load_symbol(&fields_unwatch_,
                "rdc_telemetry_fields_unwatch");
    if (status != RDC_ST_OK) {
        fields_unwatch_ = nullptr;
    }
}

RdcRasLib::~RdcRasLib() {
    if (rdc_module_destroy_) {
        rdc_module_destroy_();
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

    auto status = fields_query_(field_ids, field_count);
    RDC_LOG(RDC_DEBUG, "RAS support " << *field_count << " fields");
    return status;
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
    if (fields == nullptr) {
        return RDC_ST_BAD_PARAMETER;
    }
    if (!fields_watch_) {
        return RDC_ST_FAIL_LOAD_MODULE;
    }
    rdc_status_t status = fields_watch_(fields, fields_count);
    RDC_LOG(RDC_DEBUG, "Watch " << fields_count << " fields from RAS: "
                    << rdc_status_string(status));
    return status;
}

rdc_status_t RdcRasLib::rdc_telemetry_fields_unwatch(rdc_gpu_field_t* fields,
      uint32_t fields_count) {
    if (fields == nullptr) {
        return RDC_ST_BAD_PARAMETER;
    }
    if (!fields_unwatch_) {
        return RDC_ST_FAIL_LOAD_MODULE;
    }
    rdc_status_t status = fields_unwatch_(fields, fields_count);
    RDC_LOG(RDC_DEBUG, "Unwatch " << fields_count << " fields from RAS: "
                    << rdc_status_string(status));
    return status;
}

}  // namespace rdc
}  // namespace amd

