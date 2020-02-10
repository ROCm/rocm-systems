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


RdcEmbeddedHandler::RdcEmbeddedHandler(rdc_operation_mode_t) {
    // TODO(next_step): implement
}

rdc_status_t RdcEmbeddedHandler::rdc_group_gpu_create(rdc_group_type_t type,
        const char* group_name, rdc_gpu_group_t* p_rdc_group_id) {
    // TODO(next_step): implement
    return RDC_ST_OK;
}

rdc_status_t RdcEmbeddedHandler::rdc_group_gpu_add(rdc_gpu_group_t group_id,
        uint32_t gpu_index ) {
    // TODO(next_step): implement
    return RDC_ST_OK;
}


rdc_status_t RdcEmbeddedHandler::rdc_job_start_stats(
          rdc_gpu_group_t groupId, char  job_id[64]) {
    // TODO(next_step): implement
    return RDC_ST_OK;
}

rdc_status_t RdcEmbeddedHandler::rdc_watch_job_fields(rdc_gpu_group_t groupId,
    uint64_t update_freq, double  max_keep_age, uint32_t  max_keep_samples) {
    // TODO(next_step): implement
    return RDC_ST_OK;
}

rdc_status_t RdcEmbeddedHandler::rdc_job_get_stats(char  job_id[64],
           rdc_job_info_t* p_job_info) {
    // TODO(next_step): implement
    LOG_DEBUG("RdcEmbeddedHandler::rdc_job_get_stats:" << job_id);
    return RDC_ST_OK;
}

rdc_status_t RdcEmbeddedHandler::rdc_job_stop_stats(char  job_id[64] ) {
    // TODO(next_step): implement
    return RDC_ST_OK;
}

rdc_status_t RdcEmbeddedHandler::rdc_update_all_fields(
    uint32_t wait_for_update) {
    // TODO(next_step): implement
    LOG_DEBUG("RdcEmbeddedHandler::rdc_update_all_fields");
    return RDC_ST_OK;
}

}  // namespace rdc
}  // namespace amd
