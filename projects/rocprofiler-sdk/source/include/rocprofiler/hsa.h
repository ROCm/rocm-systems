// MIT License
//
// Copyright (c) 2023 ROCm Developer Tools
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <rocprofiler/hsa/api_args.h>
#include <rocprofiler/hsa/api_id.h>
#include <rocprofiler/hsa/table_api_id.h>

#include <hsa/hsa.h>

#include <stdint.h>

typedef uint32_t                            rocprofiler_trace_record_hsa_operation_kind_t;
typedef struct hsa_kernel_dispatch_packet_s hsa_kernel_dispatch_packet_t;
typedef struct rocprofiler_hsa_trace_data_s rocprofiler_hsa_trace_data_t;
typedef struct rocprofiler_hsa_api_data_s   rocprofiler_hsa_api_data_t;

struct rocprofiler_hsa_api_data_s
{
    uint64_t correlation_id;
    uint32_t phase;
    union
    {
        uint64_t           uint64_t_retval;
        uint32_t           uint32_t_retval;
        hsa_signal_value_t hsa_signal_value_t_retval;
        hsa_status_t       hsa_status_t_retval;
    };
    rocprofiler_hsa_api_args_t args;
    uint64_t*                  phase_data;
};

struct rocprofiler_hsa_trace_data_s
{
    rocprofiler_hsa_api_data_t api_data;
    uint64_t                   phase_enter_timestamp;
    uint64_t                   phase_exit_timestamp;
    uint64_t                   phase_data;

    void (*phase_enter)(rocprofiler_hsa_api_id_t operation_id, rocprofiler_hsa_trace_data_t* data);
    void (*phase_exit)(rocprofiler_hsa_api_id_t operation_id, rocprofiler_hsa_trace_data_t* data);
};
