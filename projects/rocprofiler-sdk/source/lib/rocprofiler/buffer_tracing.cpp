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

#include <rocprofiler/fwd.h>
#include <rocprofiler/rocprofiler.h>

#include "lib/rocprofiler/context/context.hpp"
#include "lib/rocprofiler/context/domain.hpp"
#include "lib/rocprofiler/hsa/hsa.hpp"
#include "lib/rocprofiler/registration.hpp"

#include <glog/logging.h>

#include <atomic>
#include <limits>
#include <vector>

#define RETURN_STATUS_ON_FAIL(...)                                                                 \
    if(rocprofiler_status_t _status; (_status = __VA_ARGS__) != ROCPROFILER_STATUS_SUCCESS)        \
    {                                                                                              \
        return _status;                                                                            \
    }

extern "C" {
rocprofiler_status_t
rocprofiler_configure_buffer_tracing_service(rocprofiler_context_id_t                  context_id,
                                             rocprofiler_service_buffer_tracing_kind_t kind,
                                             rocprofiler_tracing_operation_t*          operations,
                                             size_t                  operations_count,
                                             rocprofiler_buffer_id_t buffer_id)
{
    if(rocprofiler::registration::get_init_status() > -1)
        return ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED;

    if(context_id.handle >= rocprofiler::context::get_registered_contexts().size())
    {
        return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;
    }

    auto& ctx = rocprofiler::context::get_registered_contexts().at(context_id.handle);

    if(!ctx) return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;

    constexpr auto invalid_buffer_id =
        rocprofiler_buffer_id_t{std::numeric_limits<uint64_t>::max()};

    if(!ctx->buffered_tracer)
    {
        ctx->buffered_tracer = std::make_unique<rocprofiler::context::buffer_tracing_service>();
        ctx->buffered_tracer->buffer_data.fill(invalid_buffer_id);
    }

    if(ctx->buffered_tracer->buffer_data.at(kind).handle != invalid_buffer_id.handle)
        return ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED;

    RETURN_STATUS_ON_FAIL(rocprofiler::context::add_domain(ctx->buffered_tracer->domains, kind));

    ctx->buffered_tracer->buffer_data.at(kind) = buffer_id;

    for(size_t i = 0; i < operations_count; ++i)
    {
        RETURN_STATUS_ON_FAIL(rocprofiler::context::add_domain_op(
            ctx->buffered_tracer->domains, kind, operations[i]));
    }

    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_iterate_buffer_tracing_kind_names(rocprofiler_buffer_tracing_kind_name_cb_t callback,
                                              void*                                     data)
{
    // TODO(jrmadsen): need to add for other kinds
    size_t n         = 0;
    bool   premature = false;
    using pair_t     = std::pair<rocprofiler_service_buffer_tracing_kind_t, const char*>;
    for(auto [eitr, sitr] : {
            pair_t{ROCPROFILER_SERVICE_BUFFER_TRACING_HSA_API, "HSA_API"},
            pair_t{ROCPROFILER_SERVICE_BUFFER_TRACING_HIP_API, "HIP_API"},
            pair_t{ROCPROFILER_SERVICE_BUFFER_TRACING_MARKER_API, "MARKER_API"},
            pair_t{ROCPROFILER_SERVICE_BUFFER_TRACING_MEMORY_COPY, "MEMORY_COPY"},
            pair_t{ROCPROFILER_SERVICE_BUFFER_TRACING_KERNEL_DISPATCH, "KERNEL_DISPATCH"},
            pair_t{ROCPROFILER_SERVICE_BUFFER_TRACING_PAGE_MIGRATION, "PAGE_MIGRATION"},
            pair_t{ROCPROFILER_SERVICE_BUFFER_TRACING_SCRATCH_MEMORY, "SCRATCH_MEMORY"},
            pair_t{ROCPROFILER_SERVICE_BUFFER_TRACING_EXTERNAL_CORRELATION, "EXTERNAL_CORRELATION"},
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
        LOG_ASSERT(n == ROCPROFILER_SERVICE_BUFFER_TRACING_LAST - 1)
            << " :: new enumeration value added. Update this function";
    }
#else
    (void) n;
    (void) premature;
#endif

    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_iterate_buffer_tracing_kind_operation_names(
    rocprofiler_service_buffer_tracing_kind_t      kind,
    rocprofiler_buffer_tracing_operation_name_cb_t callback,
    void*                                          data)
{
    if(kind == ROCPROFILER_SERVICE_BUFFER_TRACING_HSA_API)
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
}

#undef RETURN_STATUS_ON_FAIL
