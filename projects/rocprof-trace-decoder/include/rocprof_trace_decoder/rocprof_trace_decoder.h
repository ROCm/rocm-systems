// MIT License
//
// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "trace_decoder_instrument.h"
#include "trace_decoder_types.h"

#ifndef ROCPROF_TRACE_DECODER_API
#    if defined(_WIN32) && defined(ROCPROF_TRACE_DECODER_BUILDING_LIBRARY)
#        define ROCPROF_TRACE_DECODER_API __declspec(dllexport)
#    elif defined(__GNUC__) && defined(ROCPROF_TRACE_DECODER_BUILDING_LIBRARY)
#        define ROCPROF_TRACE_DECODER_API __attribute__((visibility("default")))
#    else
#        define ROCPROF_TRACE_DECODER_API
#    endif
#endif

#ifdef __cplusplus
extern "C"
{
#endif

typedef enum
{
    ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS = 0,
    ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR,
    ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES,
    ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT,
    ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_SHADER_DATA,
    ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_NOT_IMPLEMENTED,
    ROCPROFILER_THREAD_TRACE_DECODER_STATUS_LAST
} rocprofiler_thread_trace_decoder_status_t;

/**
 * @brief Callback for rocprofiler to return traces back to rocprofiler.
 * @param[in] trace_type_id One of rocprofiler_thread_trace_decoder_record_type_t
 * @param[in] trace_events A pointer to sequence of events, of size trace_size.
 * @param[in] trace_size The number of events in the trace.
 * @param[in] userdata Arbitrary data pointer to be sent back to the user via callback.
 */
typedef rocprofiler_thread_trace_decoder_status_t (*rocprof_trace_decoder_trace_callback_t
)(rocprofiler_thread_trace_decoder_record_type_t record_type_id, void* trace_events, uint64_t trace_size, void* userdata
);

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
typedef rocprofiler_thread_trace_decoder_status_t (*rocprof_trace_decoder_isa_callback_t
)(char* instruction,
  uint64_t* memory_size,
  uint64_t* size,
  rocprofiler_thread_trace_decoder_pc_t address,
  void* userdata);

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
typedef uint64_t (*rocprof_trace_decoder_se_data_callback_t)(uint8_t** buffer, uint64_t* buffer_size, void* userdata);

/**
 * @brief Returns the description of a rocprofiler_thread_trace_decoder_info_t record.
 * @param[in] info The decoder info received
 * @retval null terminated string as description of "info".
 */
ROCPROF_TRACE_DECODER_API const char* rocprof_trace_decoder_get_info_string(rocprofiler_thread_trace_decoder_info_t info
);

ROCPROF_TRACE_DECODER_API const char* rocprof_trace_decoder_get_status_string(
    rocprofiler_thread_trace_decoder_status_t status
);

/**
 * @brief Parses thread trace data using callbacks (V1 API).
 *
 * Stateless 4-arg parse function with no handle management.
 * Backwards-compatible with rocprofiler-sdk <= 7.13.
 *
 * @param[in] se_data_callback Callback to retrieve shader engine data buffers.
 * @param[in] trace_callback Callback invoked for each decoded record.
 * @param[in] isa_callback Callback for ISA resolution of each instruction.
 * @param[in] userdata Arbitrary data pointer passed back via callbacks.
 * @retval ::ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS on success.
 * @retval ::ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_SHADER_DATA on malformed input.
 * @retval
 * ::ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR on generic error.
 * @deprecated Use the handle-based API
 * (rocprof_trace_decoder_parse) instead.
 */
ROCPROF_TRACE_DECODER_API rocprofiler_thread_trace_decoder_status_t rocprof_trace_decoder_parse_data(
    rocprof_trace_decoder_se_data_callback_t se_data_callback,
    rocprof_trace_decoder_trace_callback_t trace_callback,
    rocprof_trace_decoder_isa_callback_t isa_callback,
    void* userdata
);

/**
 * @defgroup decoder_handle Handle-based Decoder API (V2)
 * @{
 */

/**
 * @brief Opaque handle for a decoder instance with code object tracking.
 */
typedef struct
{
    uint64_t handle;
} rocprof_trace_decoder_handle_t;

/**
 * The decoder supports two mutually exclusive modes for ISA resolution during trace parsing.
 * Only one needs to be configured before calling rocprof_trace_decoder_parse():
 *
 * **Mode 1 — Built-in disassembly (requires COMGR):**
 *   Load code objects via rocprof_trace_decoder_codeobj_load(). The decoder uses COMGR
 *   internally to disassemble instructions. No ISA callback is needed.
 *
 * **Mode 2 — Custom ISA callback:**
 *   Set a callback via rocprof_trace_decoder_set_isa_callback(). The decoder calls it
 *   for every instruction encountered during parsing. No code objects need to be loaded.
 */

/**
 * @brief Creates a decoder handle.
 * @param[out] handle The handle to create.
 * @retval ::ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS on success.
 */
ROCPROF_TRACE_DECODER_API rocprofiler_thread_trace_decoder_status_t
rocprof_trace_decoder_create_handle(rocprof_trace_decoder_handle_t* handle);

/**
 * @brief Destroys a decoder handle and releases all loaded code objects.
 * @param[in] handle The handle to destroy.
 * @retval ::ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS on success.
 * @retval
 * ::ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR on generic error.
 */
ROCPROF_TRACE_DECODER_API rocprofiler_thread_trace_decoder_status_t
rocprof_trace_decoder_destroy_handle(rocprof_trace_decoder_handle_t handle);

/**
 * @brief Loads a code object for the decoder's built-in disassembly (Mode 1).
 *
 * Not required when using a custom ISA callback (Mode 2).
 *
 * @param[in] handle The decoder handle.
 * @param[in] load_id Unique load identifier for the code object.
 * @param[in] load_addr Load address (base virtual address) of the code object.
 * @param[in] load_size Memory size of the loaded code object.
 * @param[in] data Pointer to the code object ELF data.
 * @param[in] data_size Size of the code object data in bytes.
 * @retval ::ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS on success.
 * @retval ::ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_NOT_IMPLEMENTED if COMGR is not available.
 * @retval
 * ::ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR on generic error.
 */
ROCPROF_TRACE_DECODER_API rocprofiler_thread_trace_decoder_status_t rocprof_trace_decoder_codeobj_load(
    rocprof_trace_decoder_handle_t handle,
    uint64_t load_id,
    uint64_t load_addr,
    uint64_t load_size,
    const void* data,
    uint64_t data_size
);

/**
 * @brief Unloads a code object previously loaded with rocprof_trace_decoder_codeobj_load().
 * @param[in] handle The decoder handle.
 * @param[in] load_id The load identifier of the code object to unload.
 * @retval ::ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS on success.
 * @retval ::ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_NOT_IMPLEMENTED if COMGR is not available.
 * @retval
 * ::ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR on generic error.
 */
ROCPROF_TRACE_DECODER_API rocprofiler_thread_trace_decoder_status_t
rocprof_trace_decoder_codeobj_unload(rocprof_trace_decoder_handle_t handle, uint64_t load_id);

/**
 * @brief Sets a custom ISA callback for trace parsing (Mode 2).
 *
 * When set, the decoder uses this callback for all ISA resolution instead of
 * the built-in COMGR disassembly. Code objects do not need to be loaded.
 * Set callback to NULL to revert to built-in disassembly (Mode 1).
 *
 * @param[in] handle The decoder handle.
 * @param[in] callback The ISA callback, or NULL to use built-in disassembly.
 * @param[in] userdata Userdata passed to the callback.
 * @retval ::ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS on
 * success.
 */
ROCPROF_TRACE_DECODER_API rocprofiler_thread_trace_decoder_status_t rocprof_trace_decoder_set_isa_callback(
    rocprof_trace_decoder_handle_t handle, rocprof_trace_decoder_isa_callback_t callback, void* userdata
);

/**
 * @brief Sets a custom SE data callback for trace parsing.
 *
 * When set, the decoder uses this callback to retrieve shader engine data instead
 * of the data/data_size arguments passed to rocprof_trace_decoder_parse().
 * This allows using the handle-based API with the same streaming interface as the
 * deprecated rocprof_trace_decoder_parse_data().
 * Set callback to NULL to revert to using data/data_size arguments.
 *
 * @param[in] handle The decoder handle.
 * @param[in] callback The SE data callback, or NULL to use data/data_size arguments.
 * @param[in] userdata Userdata passed to the callback.
 * @retval ::ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS on
 * success.
 */
ROCPROF_TRACE_DECODER_API rocprofiler_thread_trace_decoder_status_t rocprof_trace_decoder_set_se_data_callback(
    rocprof_trace_decoder_handle_t handle, rocprof_trace_decoder_se_data_callback_t callback, void* userdata
);

/**
 * @brief Parses a buffer of thread trace data (V2 API).
 *
 * Requires either loaded code objects (Mode 1) or a custom ISA callback (Mode 2)
 * to be configured on the handle before calling. See @ref decoder_handle for details.
 *
 * @param[in] handle The decoder handle.
 * @param[in] data Pointer to the shader engine trace data.
 * @param[in] data_size Size of the trace data in bytes.
 * @param[in] trace_callback Callback invoked for each decoded record.
 * @param[in] userdata Userdata passed back to caller via trace_callback.
 * @retval ::ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS on success.
 * @retval ::ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_NOT_IMPLEMENTED if no ISA source is
 * configured (no COMGR and no ISA callback set).
 * @retval ::ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_SHADER_DATA on malformed input.
 * @retval
 * ::ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR on generic error.
 */
ROCPROF_TRACE_DECODER_API rocprofiler_thread_trace_decoder_status_t rocprof_trace_decoder_parse(
    rocprof_trace_decoder_handle_t handle,
    const void* data,
    uint64_t data_size,
    rocprof_trace_decoder_trace_callback_t trace_callback,
    void* userdata
);

/** @} */

/**
 * @brief Performs a fast partial scan of a thread trace buffer.
 *
 * Walks the trace's rare-token cluster only (REG / REG_CS / EVENT / EVENT_CS /
 * REG_CS_PRIV) and emits a single batch of ::rocprof_trace_decoder_event_t
 * records via @p trace_callback with record type
 * ::ROCPROFILER_THREAD_TRACE_DECODER_RECORD_EVENT. Significantly faster than
 * ::rocprof_trace_decoder_parse since waves, instructions, occupancy, perf
 * events and shader data are not decoded.
 *
 * Because this scan does not track time, every emitted event has a zero
 * timestamp; only @c type, @c me_id, @c pipe_id and @c payload are populated
 * (and only for token types whose mapping is known — currently REG_CS /
 * REG_CS_PRIV writes to the COMPUTE_DISPATCH_INITIATOR / COMPUTE_PGM_*
 * registers). No ISA resolution is performed and no decoder handle is
 * required.
 *
 * @param[in] handle Trace header handle.
 * @param[in] chunk_index Chunk index (double buffering)
 * @param[in] data Pointer to the shader engine trace chunk data.
 * @param[in] data_size Size of the trace data in
 * bytes.
 * @param[in] trace_callback Callback invoked once with the batch of decoded
 *   events. Not invoked when no events are produced.
 * @param[in] userdata Userdata passed back to caller via @p trace_callback.
 * @retval ::ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS on success.
 * @retval ::ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT for
 *   null/empty inputs.
 * @retval ::ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_SHADER_DATA
 *   when the header does not describe a
 * recognized trace.
 * @retval ::ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_NOT_IMPLEMENTED for
 *   architectures
 * whose quick scan is not yet wired up.
 */
ROCPROF_TRACE_DECODER_API rocprofiler_thread_trace_decoder_status_t rocprof_trace_decoder_quick_scan(
    rocprof_trace_decoder_handle_t handle,
    uint64_t chunk_index,
    const void* data,
    uint64_t data_size,
    rocprof_trace_decoder_trace_callback_t trace_callback,
    void* userdata
);

/**
 * @brief Retrieves a piece of the trace.
 *
 * @param[in] handle Trace header handle.
 * @param[in] chunk_index Chunk index (double buffering)
 * @param[in] data Pointer to the shader engine trace chunk data.
 * @param[in] data_size Size of the trace data in
 * bytes.
 * @param[in] offset_begin Byte offset where to start the standalone cut.
 * @param[in] offset_end Byte offset where to end the standalone cut.
 * @param[out] data_out Where the cut trace data is written to.
 * @param[inout] size_out Size of @p data_out . If size_out is insufficient, the necessary size_out will be written and
 * the decoder returns OUT_OF_RESOURCES. Recommended offset_end - offset_begin + 4KB
 *
 * @retval ::ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS on success.
 * @retval ::ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT for
 *   invalid chunk and/or handle.
 * @retval ::ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_NOT_IMPLEMENTED for
 *   architectures whose quick scan is
 * not yet wired up.
 * @retval ::ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES if insufficient
 * size_out.
 */
ROCPROF_TRACE_DECODER_API rocprofiler_thread_trace_decoder_status_t rocprof_trace_decoder_build_standalone(
    rocprof_trace_decoder_handle_t handle,
    uint64_t chunk_index,
    const void* data,
    uint64_t data_size,
    uint64_t offset_begin,
    uint64_t offset_end,
    void* data_out,
    uint64_t* size_out
);

#ifdef __cplusplus
}
#endif
