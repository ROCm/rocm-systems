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
#include <rocprofiler-sdk/fwd.h>

ROCPROFILER_EXTERN_C_INIT

/**
 * @defgroup THREAD_TRACE Thread Trace Service
 * @brief ROCprof-trace-decoder wrapper. Provides API calls to decode thread trace shader data.
 * @{
 */

/**
 * @brief Handle containing a loaded rocprof-trace-decoder and a decoder state.
 */
typedef struct rocprofiler_thread_trace_decoder_id_t
{
    uint64_t handle;
} rocprofiler_thread_trace_decoder_id_t;

/**
 * @brief Initializes Trace Decoder library with a library search path
 * @param[out] handle Handle to created decoder instance.
 * @param[in] path Path to trace decoder library location (e.g. /opt/rocm/lib).
 * @return ::rocprofiler_status_t
 * @retval ::ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE Library not found
 * @retval ::ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI Library found but version not supported
 * @retval ::ROCPROFILER_STATUS_SUCCESS Handle created
 */
rocprofiler_status_t
rocprofiler_thread_trace_decoder_create(rocprofiler_thread_trace_decoder_id_t* handle,
                                        const char* path) ROCPROFILER_API ROCPROFILER_NONNULL(1, 2);

/**
 * @brief Deletes handle created by ::rocprofiler_thread_trace_decoder_create
 * @param[in] handle Handle to destroy
 */
void
rocprofiler_thread_trace_decoder_destroy(rocprofiler_thread_trace_decoder_id_t handle)
    ROCPROFILER_API;

/**
 * @brief Loads a code object binary to match with Thread Trace.
 * The size, data and load_* are reported by rocprofiler-sdk's code object tracing service.
 * Used for the decoder library to know what code objects to look into when decoding shader data.
 * Not all application code objects are required to be reported here, only the ones containing code
 * executed at the time the shader data was collected by thread_trace services.
 * If a code object not reported here is encountered while decoding shader data, a record of type
 * INFO_STITCH_INCOMPLETE will be generated and instructions will not be reported with a PC address.
 *
 * @param[in] handle Handle to decoder instance.
 * @param[in] load_id Code object load ID.
 * @param[in] load_addr Code object load address.
 * @param[in] load_size Code object load size.
 * @param[in] data Code object binary data.
 * @param[in] size Code object binary data size.
 * @return ::rocprofiler_status_t
 * @retval ::ROCPROFILER_STATUS_ERROR Unable to load code object.
 * @retval ::ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT Invalid handle
 * @retval ::ROCPROFILER_STATUS_SUCCESS Code object loaded
 */
rocprofiler_status_t
rocprofiler_thread_trace_decoder_codeobj_load(rocprofiler_thread_trace_decoder_id_t handle,
                                              uint64_t                              load_id,
                                              uint64_t                              load_addr,
                                              uint64_t                              load_size,
                                              const void*                           data,
                                              uint64_t size) ROCPROFILER_API ROCPROFILER_NONNULL(5);

/**
 * @brief Unloads a code object binary.
 * @param[in] handle Handle to decoder instance.
 * @param[in] load_id Code object load ID to remove.
 * @return ::rocprofiler_status_t
 * @retval ::ROCPROFILER_STATUS_ERROR Code object not loaded.
 * @retval ::ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT Invalid handle
 * @retval ::ROCPROFILER_STATUS_SUCCESS Code object unloaded
 */
rocprofiler_status_t
rocprofiler_thread_trace_decoder_codeobj_unload(rocprofiler_thread_trace_decoder_id_t handle,
                                                uint64_t load_id) ROCPROFILER_API;

/**
 * @brief Callback for rocprof-trace-decoder to return decoder traces back to user.
 * @param[in] record_type_id One of ::rocprofiler_thread_trace_decoder_record_type_t
 * @param[in] trace_events A pointer to sequence of events, of size trace_size.
 * @param[in] trace_size The number of events in the trace.
 * @param[in] userdata Arbitrary data pointer to be sent back to the user via callback.
 */
typedef void (*rocprofiler_thread_trace_decoder_callback_t)(
    rocprofiler_thread_trace_decoder_record_type_t record_type_id,
    void*                                          trace_events,
    uint64_t                                       trace_size,
    void*                                          userdata);

/**
 * @brief Decodes shader data returned by ::rocprofiler_thread_trace_shader_data_callback_t.
 * Use ::rocprofiler_thread_trace_decoder_codeobj_load to add references to loaded code objects
 * during the trace.
 * A ::rocprofiler_thread_trace_decoder_callback_t returns decoded data back to user. The first
 * record is always of type ::ROCPROFILER_THREAD_TRACE_DECODER_RECORD_GFXIP.
 *
 * @param[in] handle Decoder handle
 * @param[in] callback Decoded trace data returned to user.
 * @param[in] data Thread trace binary data.
 * @param[in] size Thread trace binary size.
 * @param[in] userdata Userdata passed back to caller via callback.
 * @return ::rocprofiler_status_t
 * @retval ::ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT invalid argument
 * @retval ::ROCPROFILER_STATUS_ERROR_AGENT_ARCH_NOT_SUPPORTED arch not supported
 * @retval ::ROCPROFILER_STATUS_ERROR generic error
 * @retval ::ROCPROFILER_STATUS_SUCCESS on success
 */
rocprofiler_status_t
rocprofiler_trace_decode(rocprofiler_thread_trace_decoder_id_t       handle,
                         rocprofiler_thread_trace_decoder_callback_t callback,
                         void*                                       data,
                         uint64_t                                    size,
                         void* userdata) ROCPROFILER_API ROCPROFILER_NONNULL(2, 3);

/**
 * @brief Returns the string description of a ::rocprofiler_thread_trace_decoder_info_t record.
 * @param[in] handle Decoder handle
 * @param[in] info The decoder info received
 * @retval null terminated string as description of "info".
 */
const char*
rocprofiler_thread_trace_decoder_info_string(rocprofiler_thread_trace_decoder_id_t   handle,
                                             rocprofiler_thread_trace_decoder_info_t info)
    ROCPROFILER_API;

/** @} */

ROCPROFILER_EXTERN_C_FINI
