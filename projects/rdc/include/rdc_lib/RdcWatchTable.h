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
#ifndef RDC_LIB_RDCWATCHTABLE_H_
#define RDC_LIB_RDCWATCHTABLE_H_

#include <memory>
#include <vector>
#include "rdc_lib/rdc_common.h"
#include "rdc/rdc.h"


namespace amd {
namespace rdc {

class RdcWatchTable {
 public:
    virtual rdc_status_t rdc_update_all_fields() = 0;

    virtual rdc_status_t rdc_job_start_stats(rdc_gpu_group_t group_id,
                char  job_id[64]) = 0;
    virtual rdc_status_t rdc_watch_job_fields(rdc_gpu_group_t group_id,
                uint64_t update_freq, double  max_keep_age,
                uint32_t max_keep_samples) = 0;

    virtual rdc_status_t rdc_watch_fields(rdc_gpu_group_t group_id,
                rdc_field_grp_t field_group_id, uint64_t update_freq,
                double  max_keep_age, uint32_t max_keep_samples) = 0;
    virtual rdc_status_t rdc_unwatch_fields(rdc_gpu_group_t group_id,
                rdc_field_grp_t field_group_id) = 0;

    virtual ~RdcWatchTable() {}
};

typedef std::shared_ptr<RdcWatchTable> RdcWatchTablePtr;

}  // namespace rdc
}  // namespace amd

#endif  // RDC_LIB_RDCWATCHTABLE_H_
