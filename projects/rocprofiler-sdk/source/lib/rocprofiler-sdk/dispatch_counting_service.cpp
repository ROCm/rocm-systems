// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/counters/controller.hpp"
#include "lib/rocprofiler-sdk/counters/core.hpp"

#include <rocprofiler-sdk/dispatch_counting_service.h>

extern "C" {
/**
 * @brief Configure buffered dispatch profile Counting Service.
 *        Collects the counters in dispatch packets and stores them
 *        in buffer_id. The buffer may contain packets from more than
 *        one dispatch (denoted by correlation id). Will trigger the
 *        callback based on the parameters setup in buffer_id_t.
 *
 * @param [in] context_id context id
 * @param [in] buffer_id id of the buffer to use for the counting service
 * @param [in] profile profile config to use for dispatch
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t
rocprofiler_configure_buffer_dispatch_counting_service(
    rocprofiler_context_id_t                   context_id,
    rocprofiler_buffer_id_t                    buffer_id,
    rocprofiler_dispatch_counting_service_cb_t callback,
    void*                                      callback_data_args)
{
    return rocprofiler::counters::configure_buffered_dispatch(
        context_id, buffer_id, callback, callback_data_args);
}

/**
 * @brief Configure buffered dispatch profile Counting Service.
 *        Collects the counters in dispatch packets and calls a callback
 *        with the counters collected during that dispatch.
 *
 * @param [in] context_id context id
 * @param [in] dispatch_callback callback to perform when dispatch is enqueued
 * @param [in] dispatch_callback_args callback data for dispatch callback
 * @param [in] record_callback  Record callback for completed profile data
 * @param [in] record_callback_args Callback args for record callback
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t
rocprofiler_configure_callback_dispatch_counting_service(
    rocprofiler_context_id_t                   context_id,
    rocprofiler_dispatch_counting_service_cb_t dispatch_callback,
    void*                                      dispatch_callback_args,
    rocprofiler_dispatch_counting_record_cb_t  record_callback,
    void*                                      record_callback_args)
{
    return rocprofiler::counters::configure_callback_dispatch(context_id,
                                                              dispatch_callback,
                                                              dispatch_callback_args,
                                                              record_callback,
                                                              record_callback_args);
}
}
