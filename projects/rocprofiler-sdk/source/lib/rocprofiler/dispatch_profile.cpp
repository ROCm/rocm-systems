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
 * @brief Configure Dispatch Profile Counting Service.
 *
 * @param [in] context_id
 * @param [in] agent_id
 * @param [in] buffer_id
 * @param [in] callback
 * @param [in] callback_data_args
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_configure_dispatch_profile_counting_service(
    rocprofiler_context_id_t                         context_id,
    rocprofiler_profile_config_id_t                  profile,
    rocprofiler_profile_counting_dispatch_callback_t callback,
    void*                                            callback_data_args)
{
    return rocprofiler::counters::configure_dispatch(
               context_id, profile.handle, callback, callback_data_args)
               ? ROCPROFILER_STATUS_SUCCESS
               : ROCPROFILER_STATUS_ERROR;
}
}
