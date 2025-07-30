// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

typedef uint64_t rocprofiler_spm_buffer_id_t;

typedef struct
{
    void*  data;
    size_t size;
    size_t seg_size;
    size_t buffer_num;
} rocprofiler_spm_descriptor_t;

typedef void (*rocprofiler_spm_desc_callback_t)(rocprofiler_agent_id_t       agent,
                                                rocprofiler_spm_descriptor_t desc,
                                                void*                        userdata);

typedef enum
{
    ROCPROFILER_SPM_PARAMETER_SAMPLE_FREQUENCY = 0,  ///< SPM sample frequency for Sclock
    ROCPROFILER_SPM_PARAMETER_BUFFER_SIZE,           ///< SPM Buffer size, in bytes.
    ROCPROFILER_SPM_PARAMETER_TIMEOUT_MS,
    ROCPROFILER_SPM_PARAMETER_LAST
} rocprofiler_spm_parameter_type_t;

typedef struct
{
    rocprofiler_spm_parameter_type_t type;
    uint64_t                         value;
} rocprofiler_spm_parameter_t;

typedef struct
{
    rocprofiler_spm_buffer_id_t buffer_id;
    void*                       data;
    size_t                      data_size;
} rocprofiler_spm_data_record_t;

typedef enum
{
    ROCPROFILER_SPM_RECORD_TYPE_DATA = 0,        ///< "payload" is a rocprofiler_spm_data_record_t.
    ROCPROFILER_SPM_RECORD_TYPE_DATA_LOST,       ///< Data lost. "payload" is a _spm_data_record_t.
    ROCPROFILER_SPM_RECORD_TYPE_SPM_DESC,        ///< "payload" is a rocprofiler_spm_descriptor_t.
    ROCPROFILER_SPM_RECORD_TYPE_AGENT_DATA_END,  ///< No more data is to be received from agent.
    ROCPROFILER_SPM_RECORD_TYPE_DISPATCH_END,    ///< No more data is to be received from dispatch.
    ROCPROFILER_SPM_RECORD_TYPE_LAST
} rocprofiler_spm_record_type_t;

/**
 * @brief Callback to receive SPM data
 *
 * @param [in] agent Which agent generated the spm data
 * @param [in] type one of rocprofiler_spm_record_type_t
 * @param [in] payload as described in rocprofiler_spm_record_type_t
 * @param [in] userdata Passed back to user via dispatch_userdata or _configure_spm_agent_service
 */
typedef void (*rocprofiler_spm_data_callback_t)(rocprofiler_agent_id_t        agent,
                                                rocprofiler_spm_record_type_t type,
                                                void*                         payload,
                                                rocprofiler_user_data_t       userdata);

/**
 * @brief Callback query if dispatch should be profiled
 *
 * @param [in] agent_id Which agent generated the spm data
 * @param [in] queue_id rocprofiler queue_id
 * @param [in] kernel_id kernel_id
 * @param [in] dispatch_id dispatch_id
 * @param [in] config_userdata Passed back from rocprofiler_configure_spm_dispatch_service
 * @param [out] dispatch_userdata To be passed in rocprofiler_spm_data_callback_t
 *
 * @retval 1 dispatch should be profiled
 * @retval 0 dispatch should be not profiled
 */
typedef int (*rocprofiler_spm_dispatch_callback_t)(rocprofiler_agent_id_t    agent_id,
                                                   rocprofiler_queue_id_t    queue_id,
                                                   rocprofiler_kernel_id_t   kernel_id,
                                                   rocprofiler_dispatch_id_t dispatch_id,
                                                   void*                     config_userdata,
                                                   rocprofiler_user_data_t*  dispatch_userdata);
// TODO1: return a enum instead of int
// TODO2: Should we return profile config handles instead of enum?