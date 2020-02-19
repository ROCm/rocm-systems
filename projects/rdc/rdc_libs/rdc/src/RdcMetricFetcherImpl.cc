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
#include "rdc_lib/impl/RdcMetricFetcherImpl.h"
#include <sys/time.h>
#include <string.h>
#include <chrono>
#include <vector>
#include "rdc_lib/rdc_common.h"
#include "rocm_smi/rocm_smi.h"

namespace amd {
namespace rdc {

rdc_status_t RdcMetricFetcherImpl::fetch_smi_field(uint32_t gpu_index,
         uint32_t field_id, rdc_field_value* value) {
    if (!value) {
         return RDC_ST_BAD_PARAMETER;
    }
    uint64_t i64 = 0;

    struct timeval  tv;
    gettimeofday(&tv, NULL);
    value->ts = static_cast<uint64_t>(tv.tv_sec) * 1000 + tv.tv_usec / 1000;
    value->field_id = field_id;
    value->status = RSMI_STATUS_NOT_SUPPORTED;

    switch (field_id) {
        case RDC_FI_GPU_MEMORY_USAGE:
             value->status = rsmi_dev_memory_usage_get(gpu_index,
               RSMI_MEM_TYPE_VRAM, &i64);
             value->type = INTEGER;
             if (value->status == RSMI_STATUS_SUCCESS) {
                value->value.l_int = static_cast<int64_t>(i64);
             }
             break;
        case RDC_FI_GPU_MEMORY_TOTAL:
             value->status = rsmi_dev_memory_total_get(gpu_index,
               RSMI_MEM_TYPE_VRAM, &i64);
             value->type = INTEGER;
             if (value->status == RSMI_STATUS_SUCCESS) {
               value->value.l_int = static_cast<int64_t>(i64);
             }
             break;
        case RDC_FI_GPU_COUNT:
             uint32_t num_gpu;
             value->status = rsmi_num_monitor_devices(&num_gpu);
             value->type = INTEGER;
             if (value->status == RSMI_STATUS_SUCCESS) {
                value->value.l_int = static_cast<int64_t>(num_gpu);
             }
             break;
        case RDC_FI_POWER_USAGE:
             value->status = rsmi_dev_power_ave_get(gpu_index,
               RSMI_TEMP_CURRENT, &i64);
             value->type = INTEGER;
             if (value->status == RSMI_STATUS_SUCCESS) {
                 value->value.l_int = static_cast<int64_t>(i64);
             }
             break;
        case RDC_FI_GPU_SM_CLOCK:
             rsmi_frequencies_t f;
             value->status = rsmi_dev_gpu_clk_freq_get(gpu_index,
               RSMI_CLK_TYPE_SYS, &f);
             value->type = INTEGER;
             if (value->status == RSMI_STATUS_SUCCESS) {
                value->value.l_int = f.frequency[f.current];
             }
             break;
        case RDC_FI_GPU_UTIL:
            uint32_t busy_percent;
            value->status = rsmi_dev_busy_percent_get(gpu_index, &busy_percent);
            value->type = INTEGER;
            if (value->status == RSMI_STATUS_SUCCESS) {
               value->value.l_int = static_cast<int64_t>(busy_percent);
            }
            break;
        case RDC_FI_DEV_NAME:
            value->status = rsmi_dev_name_get(gpu_index,
               value->value.str, RDC_MAX_STR_LENGTH);
            value->type = STRING;
            break;
        default:
            break;
    }

    return value->status == RSMI_STATUS_SUCCESS ?  RDC_ST_OK : RDC_ST_MSI_ERROR;
}


}  // namespace rdc
}  // namespace amd
