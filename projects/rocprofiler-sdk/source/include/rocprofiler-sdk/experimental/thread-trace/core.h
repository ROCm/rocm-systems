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
    ROCPROFILER_THREAD_TRACE_PARAMETER_PERFCOUNTER,        ///< Perfmon ID and SIMD mask. gfx9 only
    ROCPROFILER_THREAD_TRACE_PARAMETER_SERIALIZE_ALL,      ///< Serializes also kernels not under
                                                           ///< thread trace
    ROCPROFILER_THREAD_TRACE_PARAMETER_PERFCOUNTER_EXCLUDE_MASK,  ///< Bitmask of which compute
                                                                  ///< units to exclude from
                                                                  ///< perfcounters. gfx9 only
    ROCPROFILER_THREAD_TRACE_PARAMETER_NO_DETAIL,    ///< Dont collect instruction timing,
                                                     ///< only shader-wide information
    ROCPROFILER_THREAD_TRACE_PARAMETER_NUM_BUFFERS,  ///< Number of CPU staging buffers in the
                                                     ///< producer/consumer pipeline. 0 (default)
                                                     ///< and 1 = single buffer mode (synchronous,
                                                     ///< no async copy). 2 is not allowed. Values
                                                     ///< >= 3 enable the async copy pipeline and
                                                     ///< require a single shader engine.
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
 * @brief Flags for the data received in rocprofiler_thread_trace_shader_data_callback_t
 * The last record takes on the flag ::ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_END.
 * If the CPU or GPU buffer cannot be flushed during the trace, the trace will be interrupted and
 * one (or both) of the buffer full flags will be set. When this happens, the trace will be disabled
 * for that agent until all the shader data is flushed and then re-enabled.
 */
typedef enum rocprofiler_thread_trace_shader_data_flags_t
{
    ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_NONE            = 0,
    ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_END             = 1 << 0,
    ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_GPU_BUFFER_FULL = 1 << 1,
    ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_CPU_BUFFER_FULL = 1 << 2,
    ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_LAST =
        ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_CPU_BUFFER_FULL

    /// @var ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_END
    /// @brief This is the last record for shader engine.
    /// @var ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_GPU_BUFFER_FULL
    /// @brief Trace was interrupted due to GPU buffer full. For multi-buffer mode
    /// (NUM_BUFFERS >= 3), this usually happens when the profiler was unable to retrieve data
    /// from the GPU fast enough. For single buffer (NUM_BUFFERS == 1), this usually happens
    /// when the buffer is not large enough.
    /// @var ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_CPU_BUFFER_FULL
    /// @brief Trace was interrupted due to CPU buffer locked. This usually happens when the user
    /// handling of rocprofiler_thread_trace_shader_data_callback_t takes too long.
} rocprofiler_thread_trace_shader_data_flags_t;

/**
 * @brief Bundle of shader data delivered to
 * rocprofiler_thread_trace_shader_data_callback_t.
 */
typedef struct rocprofiler_thread_trace_shader_data_t
{
    uint64_t size;              ///< Size of this struct.
    void*    data;              ///< Pointer to the buffer containing the ATT data.
    uint64_t data_size;         ///< Number of bytes in data.
    int64_t  shader_engine_id;  ///< ID of shader engine, as enabled by SE_MASK.
    uint64_t chunk_index;       ///< 0-based position of this chunk in the per-trace sequence.
    uint64_t read_offset;       ///< Offset into data where valid unread data begins.

    rocprofiler_agent_id_t                       agent;  ///< Identifier for the target agent.
    rocprofiler_thread_trace_shader_data_flags_t flags;  ///< Flags for this data chunk.
} rocprofiler_thread_trace_shader_data_t;

/**
 * @brief Callback to be triggered every time some ATT data is generated by the device.
 *
 * With NUM_BUFFERS >= 3 the SDK invokes this callback from a pool of worker
 * threads, so callbacks for the same trace can run concurrently and may be
 * delivered out of source order. Each chunk carries a monotonic
 * @p chunk_index that, together with @p shader_engine_id, identifies its
 * position in the original GPU output stream. To reconstruct the trace the
 * caller should buffer chunks per @p shader_engine_id and process them in
 * ascending @p chunk_index order; ::FLAGS_END marks a segment boundary
 * inserted by the SDK on a CPU_BUFFER_FULL / GPU_BUFFER_FULL restart and is
 * itself part of the indexed sequence.
 *
 * @param [in] args Shader data bundle. @p args.chunk_index is the 0-based position of this chunk
 * in the per-trace emission sequence. Index 0 is the buffering-mode header (when present) or the
 * first GPU data chunk otherwise. The index increments by exactly 1 per
 *  callback invocation and never resets within a single
 *  start_thread_trace/stop_thread_trace lifetime, so consumers can detect
 *  drops or reorder out-of-sequence deliveries by sorting on it. In
 *  single-buffer mode (NUM_BUFFERS == 1) callbacks are sequential and
 *  @p args.chunk_index is the underlying ATT iterator index.
 * @param [in] userdata Passed back to user from rocprofiler_thread_trace_dispatch_callback_t()
 */
typedef void (*rocprofiler_thread_trace_shader_data_callback_t)(
    rocprofiler_thread_trace_shader_data_t args,
    rocprofiler_user_data_t                userdata);

/** @} */

ROCPROFILER_EXTERN_C_FINI
