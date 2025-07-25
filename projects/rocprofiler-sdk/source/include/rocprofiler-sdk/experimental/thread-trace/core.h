// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

/**
 * @brief Types of Thread Trace parameters
 *
 */
typedef enum rocprofiler_thread_trace_parameter_type_t
{
    ROCPROFILER_THREAD_TRACE_PARAMETER_TARGET_CU = 0,       ///< Select the Target CU or WGP
    ROCPROFILER_THREAD_TRACE_PARAMETER_SHADER_ENGINE_MASK,  ///< Bitmask of shader engines.
    ROCPROFILER_THREAD_TRACE_PARAMETER_BUFFER_SIZE,         ///< Size of combined GPU buffer for ATT
    ROCPROFILER_THREAD_TRACE_PARAMETER_SIMD_SELECT,        ///< Bitmask (GFX9) or ID (Navi) of SIMDs
    ROCPROFILER_THREAD_TRACE_PARAMETER_PERFCOUNTERS_CTRL,  ///< Period [1,32] or disable (0) perfmon
    ROCPROFILER_THREAD_TRACE_PARAMETER_PERFCOUNTER,        ///< Perfmon ID and SIMD mask
    ROCPROFILER_THREAD_TRACE_PARAMETER_SERIALIZE_ALL,      ///< Serializes kernels not under thread
                                                           ///< trace
    ROCPROFILER_THREAD_TRACE_PARAMETER_LAST
} rocprofiler_thread_trace_parameter_type_t;

/**
 * @brief Thread Trace parameter specification
 *
 */
typedef struct rocprofiler_thread_trace_parameter_t
{
    rocprofiler_thread_trace_parameter_type_t type;
    union
    {
        uint64_t value;
        struct
        {
            rocprofiler_counter_id_t counter_id;
            uint64_t                 simd_mask : 4;
        };
    };
} rocprofiler_thread_trace_parameter_t;

/**
 * @brief Callback to be triggered every time some ATT data is generated by the device
 * @param [in] agent Identifier for the target agent (@see ::rocprofiler_agent_id_t)
 * @param [in] shader_engine_id ID of shader engine, as enabled by SE_MASK
 * @param [in] data Pointer to the buffer containing the ATT data
 * @param [in] data_size Number of bytes in "data"
 * @param [in] userdata Passed back to user from rocprofiler_thread_trace_dispatch_callback_t()
 */
typedef void (*rocprofiler_thread_trace_shader_data_callback_t)(rocprofiler_agent_id_t agent,
                                                                int64_t shader_engine_id,
                                                                void*   data,
                                                                size_t  data_size,
                                                                rocprofiler_user_data_t userdata);

/** @} */

ROCPROFILER_EXTERN_C_FINI
