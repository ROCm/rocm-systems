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

#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/counters/core.hpp"
#include "lib/rocprofiler-sdk/counters/device_counting.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/deprecated/counting_services.h>
#include <rocprofiler-sdk/device_counting_service.h>
#include <rocprofiler-sdk/fwd.h>

#include <string.h>

namespace
{
constexpr auto rocprofiler_context_none = ROCPROFILER_CONTEXT_NONE;
}

extern "C" {
rocprofiler_status_t
rocprofiler_configure_device_counting_service(rocprofiler_context_id_t                 context_id,
                                              rocprofiler_buffer_id_t                  buffer_id,
                                              rocprofiler_agent_id_t                   agent_id,
                                              rocprofiler_device_counting_service_cb_t cb,
                                              void*                                    user_data)
{
    return rocprofiler::counters::configure_agent_collection(
        context_id, buffer_id, agent_id, cb, user_data);
}

rocprofiler_status_t
rocprofiler_sample_device_counting_service(rocprofiler_context_id_t      context_id,
                                           rocprofiler_user_data_t       user_data,
                                           rocprofiler_counter_flag_t    flags,
                                           rocprofiler_counter_record_t* output_records,
                                           size_t*                       rec_count)
{
    if(context_id == rocprofiler_context_none) return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;

    // if finalized or finalizing, ignore request
    if(rocprofiler::registration::get_fini_status() != 0) return ROCPROFILER_STATUS_ERROR_FINALIZED;

    // capture the active context status now
    const auto* ctx = rocprofiler::context::get_active_context(context_id);

    // do not proceed if context has not been started
    if(!ctx) return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_STARTED;

    if(output_records != nullptr)
    {
        if(!rec_count || (flags & ROCPROFILER_COUNTER_FLAG_ASYNC) != 0)
            return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

        if(*rec_count == 0) return ROCPROFILER_STATUS_ERROR_OUT_OF_RESOURCES;

        auto recs   = std::vector<rocprofiler_counter_record_t>{};
        auto status = rocprofiler::counters::read_agent_ctx(ctx, user_data, flags, &recs);
        if(status == ROCPROFILER_STATUS_SUCCESS)
        {
            if(recs.size() > *rec_count)
            {
                *rec_count = recs.size();
                return ROCPROFILER_STATUS_ERROR_OUT_OF_RESOURCES;
            }
            *rec_count = recs.size();
            std::memcpy(
                output_records, recs.data(), sizeof(rocprofiler_counter_record_t) * recs.size());
        }
        return status;
    }

    return rocprofiler::counters::read_agent_ctx(ctx, user_data, flags, nullptr);
}
}
