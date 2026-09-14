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

#include <stdint.h>

/**
 * @brief How rocprofiler-sdk drove the decoder ABI during a decode.
 *
 * Lets a test tell a parse_data decode apart from a handle-based one, and check that the
 * handle was set up and torn down in full.
 */
typedef struct
{
    uint32_t parse_data_calls;
    uint32_t create_handle_calls;
    uint32_t destroy_handle_calls;
    uint32_t set_isa_callback_calls;
    uint32_t set_se_data_callback_calls;
    uint32_t set_analysis_calls;
    uint32_t handle_parse_calls;
    uint64_t analysis_flags;  ///< Flags from the most recent set_analysis.
    uint64_t se_data_bytes;   ///< Bytes drained through the shader engine callback.
} fake_trace_decoder_log_t;

/**
 * @brief Entry points of the handle-based API, in the order the SDK calls them.
 */
typedef enum
{
    FAKE_TRACE_DECODER_STEP_NONE = 0,
    FAKE_TRACE_DECODER_STEP_CREATE_HANDLE,
    FAKE_TRACE_DECODER_STEP_SET_ISA_CALLBACK,
    FAKE_TRACE_DECODER_STEP_SET_SE_DATA_CALLBACK,
    FAKE_TRACE_DECODER_STEP_SET_ANALYSIS,
    FAKE_TRACE_DECODER_STEP_PARSE,
} fake_trace_decoder_step_t;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Returns the call log, which the caller may reset between decodes.
 *
 * Not part of the decoder ABI. The shared library is loaded RTLD_LOCAL by the SDK, so a test
 * reaches this by dlopening the same path, which yields the same mapping.
 */
fake_trace_decoder_log_t*
fake_trace_decoder_log(void);

/**
 * @brief Makes the given step fail, so a test can drive the caller's error handling.
 *
 * Steps before it still succeed. Pass ::FAKE_TRACE_DECODER_STEP_NONE to stop failing.
 */
void
fake_trace_decoder_fail_at(fake_trace_decoder_step_t step);

#ifdef __cplusplus
}
#endif
