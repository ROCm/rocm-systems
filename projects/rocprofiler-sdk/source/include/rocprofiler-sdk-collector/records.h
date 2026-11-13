// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

/**
 * @file records.h
 * @brief Records of the traced API calls, read after ::rocprofiler_collector_stop.
 *
 * Each HIP API call is delivered as the buffer record rocprofiler-sdk writes for it in
 * process (::rocprofiler_buffer_tracing_hip_api_ext_record_t), so tools format and write
 * it with the same code. Its args union holds the values the runtime's probes recorded:
 * pointer arguments keep the target's addresses (::rocprofiler_iterate_buffer_tracing_record_args
 * prints them, it does not dereference them), and string arguments point to the collector's
 * copies of the recorded text.
 */

#include "rocprofiler-sdk-collector/defines.h"
#include "rocprofiler-sdk-collector/types.h"

// rocprofiler-sdk's hipFile headers use struct timespec without declaring it (an error in C).
#include <time.h>

#include <rocprofiler-sdk/buffer_tracing.h>

#include <stdint.h>

ROCPROFILER_COLLECTOR_EXTERN_C_INIT

/**
 * @defgroup COLLECTOR_RECORDS Records
 * @brief Records of the traced API calls
 *
 * @{
 */

/**
 * @brief Flags of a delivered call.
 */
typedef enum rocprofiler_collector_call_flag_t  // NOLINT(performance-enum-size)
{
    ROCPROFILER_COLLECTOR_CALL_FLAG_NONE = 0,
    /// The exit was not recorded (lost, or the call was running when tracing stopped):
    /// end_timestamp is start_timestamp, and retval is zero.
    ROCPROFILER_COLLECTOR_CALL_FLAG_NO_END = (1 << 0),
    /// Some arguments are zero: the trace has no value for them (a field the runtime's
    /// capture level did not record, or a parameter the probes do not describe).
    ROCPROFILER_COLLECTOR_CALL_FLAG_ARGS_INCOMPLETE = (1 << 1),
} rocprofiler_collector_call_flag_t;

/**
 * @brief What a record does not carry: its process, and its flags.
 */
typedef struct rocprofiler_collector_call_info_t
{
    uint64_t size;   ///< sizeof(rocprofiler_collector_call_info_t)
    uint32_t pid;    ///< process (in the target's PID namespace)
    uint32_t flags;  ///< rocprofiler_collector_call_flag_t bits
    uint8_t  reserved_padding[16];
} rocprofiler_collector_call_info_t;

/**
 * @brief Called for each HIP API call, in order of completion per thread. The record and
 * info are valid during the call; the strings the record points to stay valid until
 * ::rocprofiler_collector_destroy. Do not call collector functions from the callback.
 */
typedef void (*rocprofiler_collector_hip_api_callback_t)(
    const rocprofiler_buffer_tracing_hip_api_ext_record_t* record,
    const rocprofiler_collector_call_info_t*               info,
    void*                                                  user_data);

/**
 * @brief What to read. Zero-initialize it, set @p size, then set the fields needed.
 */
typedef struct rocprofiler_collector_read_config_t
{
    uint64_t size;  ///< sizeof(rocprofiler_collector_read_config_t), for ABI checks

    /// Directory of the runtimes' probe schemas (rocm_hip.json). NULL: the
    /// ROCPROFILER_COLLECTOR_SCHEMA_DIR environment variable, else share/rocm-trace in the
    /// installation of this library.
    const char* schema_dir;

    rocprofiler_collector_hip_api_callback_t hip_api_callback;  ///< may be NULL
    void*                                    user_data;

    uint8_t reserved_padding[32];
} rocprofiler_collector_read_config_t;

/**
 * @brief What a read found.
 */
typedef struct rocprofiler_collector_read_statistics_t
{
    uint64_t size;                   ///< sizeof(rocprofiler_collector_read_statistics_t)
    uint64_t events;                 ///< events in the trace
    uint64_t discarded_events;       ///< events the buffers discarded (gaps)
    uint64_t calls;                  ///< calls delivered
    uint64_t calls_no_end;           ///< of which ROCPROFILER_COLLECTOR_CALL_FLAG_NO_END
    uint64_t calls_args_incomplete;  ///< of which ROCPROFILER_COLLECTOR_CALL_FLAG_ARGS_INCOMPLETE
    uint64_t unmatched_exits;        ///< exits without a recorded enter
    uint64_t ambiguous_exits;        ///< exits not paired: events lost in between
    uint64_t unknown_calls;          ///< calls of APIs rocprofiler-sdk or the schema do not know
    uint32_t clock_stable;           ///< 0: the system was suspended while tracing
    uint8_t  reserved_padding[28];
} rocprofiler_collector_read_statistics_t;

/** @} */

/**
 * @addtogroup COLLECTOR_RECORDS
 * @{
 */

/**
 * @brief Read the trace of a stopped collector and deliver its calls. Timestamps are
 * CLOCK_BOOTTIME, as in rocprofiler-sdk's records. Can be called more than once.
 *
 * @param [in] collector The collector, after ::rocprofiler_collector_stop
 * @param [in] config What to read; see ::rocprofiler_collector_read_config_t
 * @param [out] statistics What the read found (may be NULL)
 * @return ::rocprofiler_collector_status_t
 * @retval ::ROCPROFILER_COLLECTOR_STATUS_ERROR_INVALID_STATE not stopped
 * @retval ::ROCPROFILER_COLLECTOR_STATUS_ERROR_PROBE_SCHEMA rocm_hip.json not found or invalid
 * @retval ::ROCPROFILER_COLLECTOR_STATUS_ERROR_TRACE the trace could not be decoded
 */
rocprofiler_collector_status_t
rocprofiler_collector_read(rocprofiler_collector_t                    collector,
                           const rocprofiler_collector_read_config_t* config,
                           rocprofiler_collector_read_statistics_t*   statistics)
    ROCPROFILER_COLLECTOR_API ROCPROFILER_COLLECTOR_NONNULL(2);

/** @} */

ROCPROFILER_COLLECTOR_EXTERN_C_FINI

#if defined(__cplusplus)
static_assert(sizeof(rocprofiler_collector_call_info_t) == 32,
              "Increment the size of reserved_padding after adding a member");
static_assert(sizeof(rocprofiler_collector_read_config_t) == 64,
              "Increment the size of reserved_padding after adding a member");
static_assert(sizeof(rocprofiler_collector_read_statistics_t) == 104,
              "Increment the size of reserved_padding after adding a member");
#endif
