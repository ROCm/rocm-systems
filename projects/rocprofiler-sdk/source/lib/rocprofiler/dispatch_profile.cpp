// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocprofiler/rocprofiler.h>

#include "lib/rocprofiler/aql/helpers.hpp"
#include "lib/rocprofiler/counters/core.hpp"
#include "lib/rocprofiler/counters/evaluate_ast.hpp"
#include "lib/rocprofiler/counters/metrics.hpp"
#include "lib/rocprofiler/hsa/agent_cache.hpp"

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
rocprofiler_status_t ROCPROFILER_API
rocprofiler_configure_buffered_dispatch_profile_counting_service(
    rocprofiler_context_id_t                         context_id,
    rocprofiler_buffer_id_t                          buffer_id,
    rocprofiler_profile_counting_dispatch_callback_t callback,
    void*                                            callback_data_args)
{
    return rocprofiler::counters::configure_buffered_dispatch(
               context_id, buffer_id, callback, callback_data_args)
               ? ROCPROFILER_STATUS_SUCCESS
               : ROCPROFILER_STATUS_ERROR;
}
}
