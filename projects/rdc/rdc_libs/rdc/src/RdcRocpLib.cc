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
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>

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

    status = set_rocmtools_path();
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
    if (telemetry_fields_query_ == nullptr) {
        return RDC_ST_FAIL_LOAD_MODULE;
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
    if (telemetry_fields_value_get_ == nullptr) {
        return RDC_ST_FAIL_LOAD_MODULE;
    }

    return telemetry_fields_value_get_(
        fields, fields_count, callback, user_data);
}

rdc_status_t RdcRocpLib::rdc_telemetry_fields_watch(
    rdc_gpu_field_t* fields,
    uint32_t fields_count) {
    if (fields == nullptr) {
        return RDC_ST_BAD_PARAMETER;
    }
    if (telemetry_fields_watch_ == nullptr) {
        return RDC_ST_FAIL_LOAD_MODULE;
    }

    return telemetry_fields_watch_(fields, fields_count);
}

rdc_status_t RdcRocpLib::rdc_telemetry_fields_unwatch(
    rdc_gpu_field_t* fields,
    uint32_t fields_count) {
    if (fields == nullptr) {
        return RDC_ST_BAD_PARAMETER;
    }
    if (telemetry_fields_unwatch_ == nullptr) {
        return RDC_ST_FAIL_LOAD_MODULE;
    }

    return telemetry_fields_unwatch_(fields, fields_count);
}

std::string RdcRocpLib::get_rocm_path() {
    // set default rocm path in case lookup fails
    std::string rocm_path("/opt/rocm");
    const char* rocm_path_env = getenv("ROCM_PATH");
    if (rocm_path_env != nullptr) {
        rocm_path = rocm_path_env;
    }

    std::ifstream file("/proc/self/maps");

    if (!file.is_open()) {
        return rocm_path;
    }

    std::string line;
    while (getline(file, line)) {
        size_t index_end = line.find("librocmtools.so");
        size_t index_start = index_end;
        if (index_end == std::string::npos) {
            // no library on this line
            continue;
        }
        // walk index backwards until it reaches a space
        while ((index_start > 0) && (line[index_start - 1] != ' ')) {
            index_start--;
        }
        // extract library path, drop library name
        rocm_path = line.substr(index_start, index_end - index_start);
        // appending "../" should result in "/opt/rocm/lib/.." or similar
        rocm_path += "..";
        return rocm_path;
    }

    return rocm_path;
}

rdc_status_t RdcRocpLib::set_rocmtools_path() {
    // librocmtools requires ROCMTOOLS_METRICS_PATH to be set
    std::string rocmtools_metrics_path =
        get_rocm_path() + "/libexec/rocmtools/counters/derived_counters.xml";

    // set rocm prefix
    int result =
        setenv("ROCMTOOLS_METRICS_PATH", rocmtools_metrics_path.c_str(), 0);
    if (result != 0) {
        RDC_LOG(RDC_ERROR, "setenv ROCMTOOLS_METRICS_PATH failed! " << result);
        return RDC_ST_PERM_ERROR;
    }

    // check that env exists
    const char* rocmtools_metrics_env = getenv("ROCMTOOLS_METRICS_PATH");
    if (rocmtools_metrics_env == nullptr) {
        RDC_LOG(RDC_ERROR, "ROCMTOOLS_METRICS_PATH is not set!");
        return RDC_ST_NO_DATA;
    }

    // check that file can be accessed
    std::ifstream test_file(rocmtools_metrics_env);
    if (!test_file.good()) {
        RDC_LOG(
            RDC_ERROR,
            "failed to open ROCMTOOLS_METRICS_PATH: " << rocmtools_metrics_env);
        return RDC_ST_FILE_ERROR;
    }

    return RDC_ST_OK;
}

}  // namespace rdc
}  // namespace amd
