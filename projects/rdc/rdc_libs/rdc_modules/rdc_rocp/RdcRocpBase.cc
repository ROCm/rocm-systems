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

#include <rocmtools.h>

#include <cassert>
#include <chrono>
#include <cstring>
#include <vector>
#include "hsa.h"
#include "rdc/rdc.h"
#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/rdc_common.h"
#include "rdc_modules/rdc_rocp/RdcRocpBase.h"

namespace amd {
namespace rdc {

RdcRocpBase::RdcRocpBase() {
    hsa_status_t err = hsa_init();
    if (err != HSA_STATUS_SUCCESS) {
        const char* errstr = nullptr;
        hsa_status_string(err, &errstr);
        throw std::runtime_error(
            "hsa error code: " + std::to_string(err) + " " + errstr);
    }

    auto status = rocmtools_initialize();
    RDC_LOG(RDC_INFO, "rocmtools_initialize status: " << status);
}

RdcRocpBase::~RdcRocpBase() {
    for (auto& session : sessions) {
        const rdc_status_t status = destroy_session(session.first);
        assert(status == RDC_ST_OK);
    }
    sessions.clear();
    auto status = rocmtools_finalize();
    RDC_LOG(RDC_INFO, "rocmtools_finalize status: " << status);

    hsa_status_t err = hsa_shut_down();
    if (err != HSA_STATUS_SUCCESS) {
        const char* errstr = nullptr;
        hsa_status_string(err, &errstr);
        // cannot throw an error here. print instead
        RDC_LOG(
            RDC_ERROR, "hsa error code: " + std::to_string(err) + " " + errstr);
    }
}

rdc_status_t RdcRocpBase::rocp_lookup(
    pair_gpu_field_t gpu_field,
    double* value) {
    if (sessions.empty()) {
        return RDC_ST_NOT_FOUND;
    }

    if (value == nullptr) {
        return RDC_ST_BAD_PARAMETER;
    }

    rocmtools_device_profile_metric_t counter;
    session_info_t session = sessions.at(gpu_field);
    const rocmtools_status_t status =
        rocmtools_device_profiling_session_poll(session.id, &counter);
    session.stop_time = std::chrono::high_resolution_clock::now();
    if (status != ROCMTOOLS_STATUS_SUCCESS) {
        return Rocp2RdcError(status);
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             session.stop_time - session.start_time)
                             .count();
    // some metrics are derived from others and depend on time passed
    switch (gpu_field.second) {
        case RDC_FI_PROF_GFLOPS_16:
        case RDC_FI_PROF_GFLOPS_32:
        case RDC_FI_PROF_GFLOPS_64:
        case RDC_FI_PROF_MEMR_BW_KBPNS:
        case RDC_FI_PROF_MEMW_BW_KBPNS:
            *value = counter.value.value / elapsed;
            break;
        default:
            *value = counter.value.value;
            break;
    }
    return Rocp2RdcError(status);
}

rdc_status_t RdcRocpBase::create_session(pair_gpu_field_t gpu_field) {
    if (sessions.count(gpu_field) != 0) {
        RDC_LOG(
            RDC_DEBUG, "Session for field (" << gpu_field.second << ") on GPU ["
                                             << gpu_field.first
                                             << "] already exists!");
        return RDC_ST_ALREADY_EXIST;
    }

    session_info_t session = {};

    std::vector<const char*> rocmtools_fields = {
        counter_map_k.at(gpu_field.second)};
    // create session
    rocmtools_status_t status = rocmtools_device_profiling_session_create(
        rocmtools_fields.data(), rocmtools_fields.size(), &session.id, 0,
        gpu_field.first);

    if (status != ROCMTOOLS_STATUS_SUCCESS) {
        return Rocp2RdcError(status);
    }

    // add start time
    session.start_time = std::chrono::high_resolution_clock::now();
    sessions.emplace(gpu_field, session);

    // start session
    status = rocmtools_device_profiling_session_start(session.id);

    return Rocp2RdcError(status);
}

rdc_status_t RdcRocpBase::destroy_session(pair_gpu_field_t gpu_field) {
    if (sessions.empty()) {
        RDC_LOG(RDC_DEBUG, "Cannot destroy empty session...");
        return RDC_ST_OK;
    }

    // no session with field
    if (sessions.count(gpu_field) == 0) {
        RDC_LOG(
            RDC_DEBUG, "Cannot destroy session with field ("
                           << gpu_field.second << ") on GPU ["
                           << gpu_field.first
                           << "] because it doesn't exist...");
        return RDC_ST_OK;
    }

    const rocmtools_session_id_t session_id = sessions.at(gpu_field).id;
    const rocmtools_status_t status =
        rocmtools_device_profiling_session_destroy(session_id);
    if (status == ROCMTOOLS_STATUS_SUCCESS) {
        const auto num_of_destroyed_sessions = sessions.erase(gpu_field);
        RDC_LOG(
            RDC_DEBUG,
            "destroyed (" << num_of_destroyed_sessions << ") sessions");
    }
    return Rocp2RdcError(status);
}

rdc_status_t RdcRocpBase::Rocp2RdcError(rocmtools_status_t rocm_status) {
    switch (rocm_status) {
        case ROCMTOOLS_STATUS_SUCCESS:
            return RDC_ST_OK;
        case ROCMTOOLS_STATUS_ERROR_HAS_ACTIVE_SESSION:
            return RDC_ST_ALREADY_EXIST;
        case ROCMTOOLS_STATUS_ERROR_SESSION_FILTER_DATA_MISMATCH:
        case ROCMTOOLS_STATUS_ERROR_SESSION_MISSING_FILTER:
        case ROCMTOOLS_STATUS_ERROR_SESSION_NOT_FOUND:
            return RDC_ST_BAD_PARAMETER;
        default:
            return RDC_ST_UNKNOWN_ERROR;
    }
}

}  // namespace rdc
}  // namespace amd
