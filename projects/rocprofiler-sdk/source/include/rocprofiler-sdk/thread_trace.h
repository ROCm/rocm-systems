// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/defines.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hsa.h>

ROCPROFILER_EXTERN_C_INIT

/**
 * @defgroup THREAD_TRACE Thread Trace Service
 * @brief Provides API calls to enable and handle thread trace data
 *
 * @{
 */

typedef void (*rocprofiler_att_data_types_callback_t)(int64_t     data_type_id,
                                                      const char* data_type_name,
                                                      void*       userdata);

rocprofiler_status_t
rocprofiler_att_iterate_data_types(rocprofiler_att_data_types_callback_t callback,
                                   void*                                 userdata) ROCPROFILER_API;

typedef union
{
    struct
    {
        uint32_t enable_async_queue      : 1;
        uint32_t enable_occupancy_mode   : 1;
        uint32_t enable_double_buffering : 1;
        uint32_t disable_att_markers     : 1;
        uint32_t disable_software_header : 1;
    };
    uint32_t raw;
} rocprofiler_att_parameter_flag_t;

typedef struct
{
    rocprofiler_att_parameter_flag_t flags;
    int                              shader_num;
    int*                             shader_ids;
    uint64_t                         buffer_size;
    uint8_t                          target_cu;
    uint8_t                          simd_select;
    uint8_t                          reserved;
    uint8_t                          vmid_mask;
    uint16_t                         perfcounter_mask;
    uint8_t                          perfcounter_ctrl;
    uint8_t                          perfcounter_num;
    const char**                     perfcounter;
} rocprofiler_att_parameters_t;

typedef enum
{
    ROCPROFILER_ATT_CONTROL_NONE           = 0,
    ROCPROFILER_ATT_CONTROL_START          = 1,
    ROCPROFILER_ATT_CONTROL_STOP           = 2,
    ROCPROFILER_ATT_CONTROL_START_AND_STOP = 3
} rocprofiler_att_control_flags_t;

typedef rocprofiler_att_control_flags_t (*rocprofiler_att_dispatch_callback_t)(
    rocprofiler_queue_id_t              queue_id,
    const rocprofiler_agent_t*          agent,
    rocprofiler_correlation_id_t        correlation_id,
    const hsa_kernel_dispatch_packet_t* dispatch_packet,
    uint64_t                            kernel_id,
    void*                               userdata);

typedef void (*rocprofiler_att_shader_data_callback_t)(int64_t     shader_engine_id,
                                                       int64_t     data_type_id,
                                                       const char* data_type_name,
                                                       void*       data,
                                                       size_t      data_size,
                                                       void*       userdata);

rocprofiler_status_t
rocprofiler_configure_thread_trace_service(rocprofiler_context_id_t               context_id,
                                           rocprofiler_att_parameters_t           parameters,
                                           rocprofiler_att_dispatch_callback_t    dispatch_callback,
                                           rocprofiler_att_shader_data_callback_t shader_callback,
                                           void* callback_userdata) ROCPROFILER_API;

/** @} */

ROCPROFILER_EXTERN_C_FINI
