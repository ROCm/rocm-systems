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
#ifndef INCLUDE_RDC_LIB_RDCWATCHTABLE_H_
#define INCLUDE_RDC_LIB_RDCWATCHTABLE_H_

#include <memory>
#include <vector>

#include "rdc/rdc.h"
#include "rdc_lib/rdc_common.h"

namespace amd {
namespace rdc {

class RdcWatchTable {
 public:
  virtual rdc_status_t rdc_field_update_all() = 0;
  virtual rdc_status_t rdc_field_listen_notif(uint32_t timeout_ms) = 0;

  virtual rdc_status_t rdc_job_start_stats(rdc_gpu_group_t group_id, const char job_id[64],
                                           uint64_t update_freq,
                                           const rdc_gpu_gauges_t& gpu_gauge) = 0;
  virtual rdc_status_t rdc_job_stop_stats(const char job_id[64],
                                          const rdc_gpu_gauges_t& gpu_gauge) = 0;
  virtual rdc_status_t rdc_job_remove(const char job_id[64]) = 0;
  virtual rdc_status_t rdc_job_remove_all() = 0;

  virtual rdc_status_t rdc_field_watch(rdc_gpu_group_t group_id, rdc_field_grp_t field_group_id,
                                       uint64_t update_freq, double max_keep_age,
                                       uint32_t max_keep_samples) = 0;
  virtual rdc_status_t rdc_field_unwatch(rdc_gpu_group_t group_id,
                                         rdc_field_grp_t field_group_id) = 0;

  virtual rdc_status_t rdc_health_set(rdc_gpu_group_t group_id,
                                      unsigned int components) = 0;
  virtual rdc_status_t rdc_health_get(rdc_gpu_group_t group_id,
                                      unsigned int* components) = 0;
  virtual rdc_status_t rdc_health_check(rdc_gpu_group_t group_id,
                                        rdc_health_response_t *response) = 0;
  virtual rdc_status_t rdc_health_clear(rdc_gpu_group_t group_id) = 0;

  virtual ~RdcWatchTable() {}
};

typedef std::shared_ptr<RdcWatchTable> RdcWatchTablePtr;

}  // namespace rdc
}  // namespace amd

#endif  // INCLUDE_RDC_LIB_RDCWATCHTABLE_H_
