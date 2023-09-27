// MIT License
//
// Copyright (c) 2023 ROCm Developer Tools
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

#include "lib/rocprofiler/context/context.hpp"
#include "lib/rocprofiler/context/domain.hpp"
#include "lib/rocprofiler/hsa/hsa.hpp"
#include "lib/rocprofiler/registration.hpp"

#include <glog/logging.h>

#include <atomic>
#include <vector>

#define RETURN_STATUS_ON_FAIL(...)                                                                 \
    if(rocprofiler_status_t _status; (_status = __VA_ARGS__) != ROCPROFILER_STATUS_SUCCESS)        \
    {                                                                                              \
        return _status;                                                                            \
    }

extern "C" {
rocprofiler_status_t
rocprofiler_configure_callback_tracing_service(rocprofiler_context_id_t context_id,
                                               rocprofiler_service_callback_tracing_kind_t kind,
                                               rocprofiler_tracing_operation_t*  operations,
                                               size_t                            operations_count,
                                               rocprofiler_callback_tracing_cb_t callback,
                                               void*                             callback_args)
{
    if(rocprofiler::registration::get_init_status() > -1)
        return ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED;

    if(context_id.handle >= rocprofiler::context::get_registered_contexts().size())
    {
        return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;
    }

    auto& ctx = rocprofiler::context::get_registered_contexts().at(context_id.handle);

    if(!ctx) return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;

    if(!ctx->callback_tracer)
        ctx->callback_tracer = std::make_unique<rocprofiler::context::callback_tracing_service>();

    if(ctx->callback_tracer->callback_data.at(kind).callback)
        return ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED;

    RETURN_STATUS_ON_FAIL(rocprofiler::context::add_domain(ctx->callback_tracer->domains, kind));

    ctx->callback_tracer->callback_data.at(kind) = {callback, callback_args};

    for(size_t i = 0; i < operations_count; ++i)
    {
        RETURN_STATUS_ON_FAIL(rocprofiler::context::add_domain_op(
            ctx->callback_tracer->domains, kind, operations[i]));
    }

    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_iterate_callback_tracing_kind_names(
    rocprofiler_callback_tracing_kind_name_cb_t callback,
    void*                                       data)
{
    // TODO(jrmadsen): need to add for other kinds
    size_t n         = 0;
    bool   premature = false;
    using pair_t     = std::pair<rocprofiler_service_callback_tracing_kind_t, const char*>;
    for(auto [eitr, sitr] : {
            pair_t{ROCPROFILER_SERVICE_CALLBACK_TRACING_HSA_API, "HSA_API"},
            pair_t{ROCPROFILER_SERVICE_CALLBACK_TRACING_HIP_API, "HIP_API"},
            pair_t{ROCPROFILER_SERVICE_CALLBACK_TRACING_MARKER_API, "MARKER_API"},
            pair_t{ROCPROFILER_SERVICE_CALLBACK_TRACING_CODE_OBJECT, "CODE_OBJECT"},
            pair_t{ROCPROFILER_SERVICE_CALLBACK_TRACING_KERNEL_DISPATCH, "KERNEL_DISPATCH"},
        })
    {
        auto _success = callback(eitr, sitr, data);
        if(_success != 0)
        {
            premature = true;
            break;
        }
        ++n;
    }

#if defined(ROCPROFILER_CI)
    if(!premature)
    {
        LOG_ASSERT(n == ROCPROFILER_SERVICE_CALLBACK_TRACING_LAST - 1)
            << " :: new enumeration value added. Update this function";
    }
#else
    (void) n;
    (void) premature;
#endif

    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_iterate_callback_tracing_kind_operation_names(
    rocprofiler_service_callback_tracing_kind_t      kind,
    rocprofiler_callback_tracing_operation_name_cb_t callback,
    void*                                            data)
{
    if(kind == ROCPROFILER_SERVICE_CALLBACK_TRACING_HSA_API)
    {
        auto ops = rocprofiler::hsa::get_ids();
        for(const auto& itr : ops)
        {
            auto _success = callback(kind, itr, rocprofiler::hsa::name_by_id(itr), data);
            if(_success != 0) break;
        }
        return ROCPROFILER_STATUS_SUCCESS;
    }

    return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;
}

rocprofiler_status_t
rocprofiler_iterate_callback_tracing_operation_args(
    rocprofiler_callback_tracing_record_t            record,
    rocprofiler_callback_tracing_operation_args_cb_t callback,
    void*                                            user_data)
{
    if(record.kind == ROCPROFILER_SERVICE_CALLBACK_TRACING_HSA_API)
    {
        rocprofiler::hsa::iterate_args(
            record.operation,
            *static_cast<rocprofiler_hsa_api_callback_tracer_data_t*>(record.payload),
            callback,
            user_data);
        return ROCPROFILER_STATUS_SUCCESS;
    }

    return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;
}
}

#undef RETURN_STATUS_ON_FAIL
