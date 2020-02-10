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
#ifndef RDC_LIB_IMPL_RDCEMBEDDEDHANDLER_H_
#define RDC_LIB_IMPL_RDCEMBEDDEDHANDLER_H_

#include "rdc_lib/RdcHandler.h"


namespace amd {
namespace rdc {

class RdcEmbeddedHandler: public RdcHandler
{
 public:
    rdc_status_t rdc_group_gpu_create(rdc_group_type_t type,
                const char* group_name,
                rdc_gpu_group_t* p_rdc_group_id) override;
    rdc_status_t rdc_group_gpu_add(rdc_gpu_group_t groupId,
                uint32_t gpu_index) override;

    rdc_status_t rdc_job_start_stats(rdc_gpu_group_t groupId,
                char  job_id[64]) override;

    rdc_status_t rdc_watch_job_fields(rdc_gpu_group_t groupId,
            uint64_t update_freq, double  max_keep_age,
            uint32_t  max_keep_samples) override;
    rdc_status_t rdc_job_get_stats(char  jobId[64],
                rdc_job_info_t* p_job_info) override;
    rdc_status_t rdc_job_stop_stats(char  job_id[64]) override;

    rdc_status_t rdc_update_all_fields(uint32_t wait_for_update) override;

    explicit RdcEmbeddedHandler(rdc_operation_mode_t op_mode);
};

}  // namespace rdc
}  // namespace amd

extern "C" {
    amd::rdc::RdcHandler *make_handler(rdc_operation_mode_t op_mode);
}

#endif  // RDC_LIB_IMPL_RDCEMBEDDEDHANDLER_H_
