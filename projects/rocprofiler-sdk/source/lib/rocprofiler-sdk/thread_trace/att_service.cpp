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

#include <rocprofiler-sdk/rocprofiler.h>

#include "lib/rocprofiler-sdk/aql/helpers.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"

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
rocprofiler_configure_thread_trace_service(rocprofiler_context_id_t               context_id,
                                           rocprofiler_att_parameters_t           parameters,
                                           rocprofiler_att_dispatch_callback_t    dispatch_callback,
                                           rocprofiler_att_shader_data_callback_t shader_callback,
                                           void*                                  callback_userdata)
{
    auto* ctx = rocprofiler::context::get_mutable_registered_context(context_id);
    if(!ctx) return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_STARTED;
    if(ctx->thread_trace) return ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED;
    if(parameters.flags.raw != 0) return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;

    auto thread_tracer = std::make_shared<rocprofiler::thread_trace_parameters>();

    thread_tracer->context_id        = context_id;
    thread_tracer->dispatch_cb_fn    = dispatch_callback;
    thread_tracer->shader_cb_fn      = shader_callback;
    thread_tracer->callback_userdata = callback_userdata;

    thread_tracer->flags       = parameters.flags;
    thread_tracer->buffer_size = parameters.buffer_size;
    thread_tracer->target_cu   = parameters.target_cu;
    thread_tracer->simd_select = parameters.simd_select;
    thread_tracer->vmid_mask   = parameters.vmid_mask;

    thread_tracer->perfcounter_mask = parameters.perfcounter_mask;
    thread_tracer->perfcounter_ctrl = parameters.perfcounter_ctrl;

    for(int i = 0; i < parameters.perfcounter_num; i++)
        thread_tracer->perfcounters.emplace_back(parameters.perfcounter[i]);

    thread_tracer->shader_engine_mask = 0;
    for(int i = 0; i < parameters.shader_num; i++)
        thread_tracer->shader_engine_mask |= 1ul << parameters.shader_ids[i];

    ctx->thread_trace = std::make_shared<rocprofiler::ThreadTracer>(thread_tracer);

    return ROCPROFILER_STATUS_SUCCESS;
}
}
