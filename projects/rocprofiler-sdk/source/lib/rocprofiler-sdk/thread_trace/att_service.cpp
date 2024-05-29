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
#include "lib/rocprofiler-sdk/registration.hpp"

extern "C" {
rocprofiler_status_t ROCPROFILER_API
rocprofiler_configure_thread_trace_service(rocprofiler_context_id_t               context_id,
                                           rocprofiler_att_parameter_t*           parameters,
                                           size_t                                 num_parameters,
                                           rocprofiler_att_dispatch_callback_t    dispatch_callback,
                                           rocprofiler_att_shader_data_callback_t shader_callback,
                                           void*                                  callback_userdata)
{
    if(rocprofiler::registration::get_init_status() > -1)
        return ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED;

    auto* ctx = rocprofiler::context::get_mutable_registered_context(context_id);
    if(!ctx) return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_STARTED;
    if(ctx->thread_trace) return ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED;

    auto param_pack = rocprofiler::thread_trace_parameter_pack{};

    param_pack.context_id        = context_id;
    param_pack.dispatch_cb_fn    = dispatch_callback;
    param_pack.shader_cb_fn      = shader_callback;
    param_pack.callback_userdata = callback_userdata;
    bool bEnableCodeobj          = false;

    for(size_t p = 0; p < num_parameters; p++)
    {
        const rocprofiler_att_parameter_t& param = parameters[p];
        if(param.type > ROCPROFILER_ATT_PARAMETER_LAST)
            return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

        switch(param.type)
        {
            case ROCPROFILER_ATT_PARAMETER_TARGET_CU: param_pack.target_cu = param.value; break;
            case ROCPROFILER_ATT_PARAMETER_SHADER_ENGINE_MASK:
                param_pack.shader_engine_mask = param.value;
                break;
            case ROCPROFILER_ATT_PARAMETER_BUFFER_SIZE: param_pack.buffer_size = param.value; break;
            case ROCPROFILER_ATT_PARAMETER_SIMD_SELECT: param_pack.simd_select = param.value; break;
            case ROCPROFILER_ATT_PARAMETER_CODE_OBJECT_TRACE_ENABLE:
                bEnableCodeobj = param.value != 0;
                break;
            case ROCPROFILER_ATT_PARAMETER_LAST: return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
        }
        // for(int i = 0; i < parameters.perfcounter_num; i++)
        //    thread_tracer->perfcounters.emplace_back(parameters.perfcounter[i]);
    }

    ctx->thread_trace = std::make_shared<rocprofiler::GlobalThreadTracer>(param_pack);

    if(!bEnableCodeobj) return ROCPROFILER_STATUS_SUCCESS;  // Skip TRACING_CODE_OBJECT setup

    auto& client_ctx = ctx->thread_trace->codeobj_client_ctx;

    rocprofiler_status_t status = rocprofiler_create_context(&client_ctx);
    if(status != ROCPROFILER_STATUS_SUCCESS) return status;

    status = rocprofiler_configure_callback_tracing_service(
        client_ctx,
        ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
        nullptr,
        0,
        rocprofiler::GlobalThreadTracer::codeobj_tracing_callback,
        ctx->thread_trace.get());

    return status;
}
}
