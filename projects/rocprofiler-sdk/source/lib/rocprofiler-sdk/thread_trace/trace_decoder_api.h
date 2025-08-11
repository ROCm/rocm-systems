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

#include <rocprofiler-sdk/experimental/thread-trace/trace_decoder_types.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum
{
    ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS = 0,
    ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR,
    ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES,
    ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT,
    ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_SHADER_DATA,
    ROCPROFILER_THREAD_TRACE_DECODER_STATUS_LAST
} rocprofiler_thread_trace_decoder_status_t;

/**
 * @brief Callback for rocprofiler to return traces back to rocprofiler.
 * @param[in] trace_type_id One of rocprofiler_thread_trace_decoder_record_type_t
 * @param[in] trace_events A pointer to sequence of events, of size trace_size.
 * @param[in] trace_size The number of events in the trace.
 * @param[in] userdata Arbitrary data pointer to be sent back to the user via callback.
 */
typedef rocprofiler_thread_trace_decoder_status_t (*rocprof_trace_decoder_trace_callback_t)(
    rocprofiler_thread_trace_decoder_record_type_t record_type_id,
    void*                                          trace_events,
    uint64_t                                       trace_size,
    void*                                          userdata);

/**
 * @brief Callback for rocprofiler to return ISA to decoder.
 * The caller must copy a desired instruction on isa_instruction and source_reference,
 * while obeying the max length passed by the caller.
 * If the caller's length is insufficient, then this function writes the minimum sizes to isa_size
 * and source_size and returns ::ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES.
 * If call returns _SUCCESS, isa_size and source_size must be written with bytes used.
 * @param[out] instruction Where to copy the ISA line to.
 * @param[out] memory_size (Auto) The number of bytes to next instruction. 0 for custom ISA.
 * @param[inout] size Size of returned ISA string.
 * @param[in] address The code object ID and offset from base vaddr.
 * If marker_id == 0, this parameter is raw virtual address with no codeobj ID information.
 * @param[in] userdata Arbitrary data pointer to be sent back to the user via callback.
 * @retval ::ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS on success.
 * @retval ::ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR on generic error.
 * @retval ::ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT for invalid address.
 * @retval ::ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES for insufficient
 * isa_size.
 */
typedef rocprofiler_thread_trace_decoder_status_t (*rocprof_trace_decoder_isa_callback_t)(
    char*                                 instruction,
    uint64_t*                             memory_size,
    uint64_t*                             size,
    rocprofiler_thread_trace_decoder_pc_t address,
    void*                                 userdata);

/**
 * @brief Callback for the decoder to retrieve Shader Engine data. Return zero to end parsing.
 * @param[out] buffer The buffer to fill up with SE data.
 * @param[out] buffer_size The space available in the buffer.
 * @param[in] userdata Arbitrary data pointer to be sent back to the user via callback.
 * @returns Number of bytes remaining.
 * @retval 0 if no more SE data is available. Parsing will stop.
 * @retval buffer_size if the buffer does not hold enough data.
 * @retval 0 > ret > buffer_size for partially filled buffer, and call ends.
 */
typedef uint64_t (*rocprof_trace_decoder_se_data_callback_t)(uint8_t** buffer,
                                                             uint64_t* buffer_size,
                                                             void*     userdata);

/**
 * @brief Parses thread trace data.
 * @param[in] se_data_callback Callback to return shader engine data from.
 * @param[in] trace_callback Callback where the trace data is returned to.
 * @param[in] isa_callback Callback to return ISA lines.
 * @param[in] userdata Userdata passed back to caller via callback.
 */
rocprofiler_thread_trace_decoder_status_t
rocprof_trace_decoder_parse_data(rocprof_trace_decoder_se_data_callback_t se_data_callback,
                                 rocprof_trace_decoder_trace_callback_t   trace_callback,
                                 rocprof_trace_decoder_isa_callback_t     isa_callback,
                                 void*                                    userdata);

/**
 * @brief Returns the description of a rocprofiler_thread_trace_decoder_info_t record.
 * @param[in] info The decoder info received
 * @retval null terminated string as description of "info".
 */
const char*
rocprof_trace_decoder_get_info_string(rocprofiler_thread_trace_decoder_info_t info);

const char*
rocprof_trace_decoder_get_status_string(rocprofiler_thread_trace_decoder_status_t status);

typedef void (*rocprofiler_thread_trace_decoder_debug_callback_t)(int64_t     time,
                                                                  const char* type,
                                                                  const char* info,
                                                                  void*       userdata);

rocprofiler_thread_trace_decoder_status_t
rocprof_trace_decoder_dump_data(const char*                                       data,
                                size_t                                            data_size,
                                rocprofiler_thread_trace_decoder_debug_callback_t cb,
                                void*                                             userdata);

#ifdef __cplusplus
}
#endif
