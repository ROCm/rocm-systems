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

typedef enum
{
    ROCPROFILER_ATT_PARAMETER_TARGET_CU = 0,
    ROCPROFILER_ATT_PARAMETER_SHADER_ENGINE_MASK,
    ROCPROFILER_ATT_PARAMETER_BUFFER_SIZE,
    ROCPROFILER_ATT_PARAMETER_SIMD_SELECT,
    ROCPROFILER_ATT_PARAMETER_PERFCOUNTERS_CTRL,
    ROCPROFILER_ATT_PARAMETER_PERFCOUNTER,
    ROCPROFILER_ATT_PARAMETER_OCCUPANCY_MODE_ENABLE,
    ROCPROFILER_ATT_PARAMETER_LAST
} rocprofiler_att_parameter_type_t;

typedef struct
{
    rocprofiler_att_parameter_type_t type;
    uint64_t                         value;
} rocprofiler_att_parameter_t;

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

typedef void (*rocprofiler_att_shader_data_callback_t)(int64_t shader_engine_id,
                                                       void*   data,
                                                       size_t  data_size,
                                                       void*   userdata);

rocprofiler_status_t
rocprofiler_configure_thread_trace_service(rocprofiler_context_id_t               context_id,
                                           rocprofiler_att_parameter_t*           parameters,
                                           size_t                                 num_parameters,
                                           rocprofiler_att_dispatch_callback_t    dispatch_callback,
                                           rocprofiler_att_shader_data_callback_t shader_callback,
                                           void* callback_userdata) ROCPROFILER_API;

/**
 * @brief Callback for rocprofiler to parsed ATT data.
 * The caller must copy a desired instruction on isa_instruction and source_reference,
 * while obeying the max length passed by the caller.
 * If the caller's length is insufficient, then this function writes the minimum sizes to isa_size
 * and source_size and returns ROCPROFILER_STATUS_ERROR_OUT_OF_RESOURCES.
 * If call returns ROCPROFILER_STATUS_SUCCESS, isa_size and source_size are written with bytes used.
 * @param[out] isa_instruction Where to copy the ISA line to.
 * @param[out] isa_memory_size (Auto) The number of bytes to next instruction. 0 for custom ISA.
 * @param[inout] isa_size Size of returned ISA string.
 * @param[in] marker_id The generated ATT marker for given codeobject ID.
 * @param[in] offset The offset from base vaddr for given codeobj ID.
 * If marker_id == 0, this parameter is raw virtual address with no codeobj ID information.
 * @param[in] userdata Arbitrary data pointer to be sent back to the user via callback.
 * @retval ROCPROFILER_STATUS_SUCCESS on success.
 * @retval ROCPROFILER_STATUS_ERROR on generic error.
 * @retval ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT for invalid offset or invalid marker_id.
 * @retval ROCPROFILER_STATUS_ERROR_OUT_OF_RESOURCES for insufficient isa_size or source_size.
 */
typedef rocprofiler_status_t (*rocprofiler_att_parser_isa_callback_t)(char*     isa_instruction,
                                                                      uint64_t* isa_memory_size,
                                                                      uint64_t* isa_size,
                                                                      uint64_t  marker_id,
                                                                      uint64_t  offset,
                                                                      void*     userdata);

/**
 * @brief Callback for the ATT parser to retrieve Shader Engine data.
 * Returns the amount of data filled. If no more data is available, then callback return 0
 * If the space available in the buffer is less than required for parsing the full data,
 * the full data is transfered over multiple calls.
 * When all data has been transfered from current shader_engine_id, the caller has the option to
 *  1) Return -1 on shader_engine ID and parsing terminates
 *  2) Move to the next shader engine.
 * @param[out] shader_engine_id The ID of given shader engine.
 * @param[out] buffer The buffer to fill up with SE data.
 * @param[out] buffer_size The space available in the buffer.
 * @param[in] userdata Arbitrary data pointer to be sent back to the user via callback.
 * @returns Number of bytes remaining in shader engine.
 * @retval 0 if no more SE data is available. Parsing will stop.
 * @retval buffer_size if the buffer does not hold enough data for the current shader engine.
 * @retval 0 > ret > buffer_size for partially filled buffer, and caller moves over to next SE.
 */
typedef uint64_t (*rocprofiler_att_parser_se_data_callback_t)(int*      shader_engine_id,
                                                              uint8_t** buffer,
                                                              uint64_t* buffer_size,
                                                              void*     userdata);

typedef enum
{
    ROCPROFILER_ATT_PARSER_DATA_TYPE_ISA = 0,
    ROCPROFILER_ATT_PARSER_DATA_TYPE_OCCUPANCY,
} rocprofiler_att_parser_data_type_t;

typedef struct
{
    uint64_t marker_id;
    uint64_t offset;
    uint64_t hitcount;
    uint64_t latency;
} rocprofiler_att_data_type_isa_t;

typedef struct
{
    uint64_t marker_id;
    uint64_t offset;
    uint64_t timestamp : 63;
    uint64_t enabled   : 1;
} rocprofiler_att_data_type_occupancy_t;

/**
 * @brief Callback for rocprofiler to return traces back to rocprofiler.
 * @param[in] att_data A datapoint retrieved from thread_trace
 * @param[in] userdata Arbitrary data pointer to be sent back to the user via callback.
 */
typedef void (*rocprofiler_att_parser_trace_callback_t)(rocprofiler_att_parser_data_type_t type,
                                                        void*                              att_data,
                                                        void* userdata);

/**
 * @brief Iterate over all event coordinates for a given agent_t and event_t.
 * @param[in] se_data_callback Callback to return shader engine data from.
 * @param[in] trace_callback Callback where the trace data is returned to.
 * @param[in] isa_callback Callback to return ISA lines.
 * @param[in] userdata Userdata passed back to caller via callback.
 */
rocprofiler_status_t
rocprofiler_att_parse_data(rocprofiler_att_parser_se_data_callback_t se_data_callback,
                           rocprofiler_att_parser_trace_callback_t   trace_callback,
                           rocprofiler_att_parser_isa_callback_t     isa_callback,
                           void*                                     userdata);

/** @} */

ROCPROFILER_EXTERN_C_FINI
