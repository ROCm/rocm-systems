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

typedef enum
{
    ROCPROFILER_SPM_PARAMETER_SAMPLE_FREQUENCY = 0,  ///< SPM sample frequency for Sclock
    ROCPROFILER_SPM_PARAMETER_BUFFER_SIZE,           ///< SPM Buffer size, in bytes.
    ROCPROFILER_SPM_PARAMETER_TIMEOUT_MS,
    ROCPROFILER_SPM_PARAMETER_LAST
} rocprofiler_spm_parameter_type_t;

typedef enum rocprofiler_spm_record_flags_t
{
    ROCPROFILER_SPM_RECORD_FLAG_DATA_LOST = 0,
    ROCPROFILER_SPM_RECORD_FLAG_END,
    ROCPROFILER_SPM_RECORD_FLAG_DATA,
    ROCPROFILER_SPM_RECORD_FLAG_LAST,
} rocprofiler_spm_record_flags_t;

typedef struct
{
    rocprofiler_agent_id_t agent_id;
    // XCC,  shader, counter id, block instance Id
    rocprofiler_counter_instance_id_t id;

    rocprofiler_timestamp_t timestamp;
    uint64_t                value;
} rocprofiler_spm_counter_record_t;

/**
 * @brief Callback to receive SPM data
 *
 * @param [in] agent Which agent generated the spm data
 * @param [in] records pointer to the array of SPM records
 * @param [in] size_t  size of the record array
 * @param [in] userdata Passed back to user via dispatch_userdata or _configure_spm_agent_service
 */

typedef void (*rocprofiler_spm_data_callback_t)(rocprofiler_spm_counter_record_t* records,
                                                size_t                            record_count,
                                                rocprofiler_spm_record_flags_t    flags,
                                                rocprofiler_user_data_t           userdata);
typedef struct
{
    rocprofiler_spm_parameter_type_t type;
    uint64_t                         value;
} rocprofiler_spm_parameter_t;

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