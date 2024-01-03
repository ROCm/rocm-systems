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

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/context/domain.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include <glog/logging.h>

#include <atomic>
#include <limits>
#include <vector>

#define RETURN_STATUS_ON_FAIL(...)                                                                 \
    if(rocprofiler_status_t _status; (_status = __VA_ARGS__) != ROCPROFILER_STATUS_SUCCESS)        \
    {                                                                                              \
        return _status;                                                                            \
    }

namespace rocprofiler
{
namespace buffer_tracing
{
namespace
{
#define ROCPROFILER_BUFFER_TRACING_KIND_STRING(CODE)                                               \
    template <>                                                                                    \
    struct buffer_tracing_kind_string<ROCPROFILER_BUFFER_TRACING_##CODE>                           \
    {                                                                                              \
        static constexpr auto value =                                                              \
            std::pair<const char*, size_t>{#CODE, std::string_view{#CODE}.length()};               \
    };

template <size_t Idx>
struct buffer_tracing_kind_string;

ROCPROFILER_BUFFER_TRACING_KIND_STRING(NONE)
ROCPROFILER_BUFFER_TRACING_KIND_STRING(HSA_API)
ROCPROFILER_BUFFER_TRACING_KIND_STRING(HIP_API)
ROCPROFILER_BUFFER_TRACING_KIND_STRING(MARKER_API)
ROCPROFILER_BUFFER_TRACING_KIND_STRING(MEMORY_COPY)
ROCPROFILER_BUFFER_TRACING_KIND_STRING(KERNEL_DISPATCH)
ROCPROFILER_BUFFER_TRACING_KIND_STRING(PAGE_MIGRATION)
ROCPROFILER_BUFFER_TRACING_KIND_STRING(SCRATCH_MEMORY)
ROCPROFILER_BUFFER_TRACING_KIND_STRING(EXTERNAL_CORRELATION)

template <size_t Idx, size_t... Tail>
std::pair<const char*, size_t>
get_kind_name(rocprofiler_buffer_tracing_kind_t kind, std::index_sequence<Idx, Tail...>)
{
    if(kind == Idx) return buffer_tracing_kind_string<Idx>::value;
    // recursion until tail empty
    if constexpr(sizeof...(Tail) > 0) return get_kind_name(kind, std::index_sequence<Tail...>{});
    return {nullptr, 0};
}
}  // namespace
}  // namespace buffer_tracing
}  // namespace rocprofiler

extern "C" {
rocprofiler_status_t
rocprofiler_configure_buffer_tracing_service(rocprofiler_context_id_t          context_id,
                                             rocprofiler_buffer_tracing_kind_t kind,
                                             rocprofiler_tracing_operation_t*  operations,
                                             size_t                            operations_count,
                                             rocprofiler_buffer_id_t           buffer_id)
{
    if(rocprofiler::registration::get_init_status() > -1)
        return ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED;

    auto* ctx = rocprofiler::context::get_mutable_registered_context(context_id);

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
rocprofiler_query_buffer_tracing_kind_name(rocprofiler_buffer_tracing_kind_t kind,
                                           const char**                      name,
                                           uint64_t*                         name_len)
{
    auto&& val = rocprofiler::buffer_tracing::get_kind_name(
        kind, std::make_index_sequence<ROCPROFILER_BUFFER_TRACING_LAST>{});

    if(name) *name = val.first;
    if(name_len) *name_len = val.second;

    return (val.first) ? ROCPROFILER_STATUS_SUCCESS : ROCPROFILER_STATUS_ERROR_KIND_NOT_FOUND;
}

rocprofiler_status_t
rocprofiler_query_buffer_tracing_kind_operation_name(rocprofiler_buffer_tracing_kind_t kind,
                                                     uint32_t                          operation,
                                                     const char**                      name,
                                                     uint64_t*                         name_len)
{
    if(kind < ROCPROFILER_BUFFER_TRACING_NONE || kind >= ROCPROFILER_BUFFER_TRACING_LAST)
        return ROCPROFILER_STATUS_ERROR_KIND_NOT_FOUND;

    if(kind == ROCPROFILER_BUFFER_TRACING_HSA_API)
    {
        const auto* val = rocprofiler::hsa::name_by_id(operation);

        if(!val)
        {
            if(name) *name = nullptr;
            if(name_len) *name_len = 0;

            return ROCPROFILER_STATUS_ERROR_OPERATION_NOT_FOUND;
        }

        if(name) *name = val;
        if(name_len) *name_len = strnlen(val, 4096);

        return ROCPROFILER_STATUS_SUCCESS;
    }

    return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;
}

rocprofiler_status_t
rocprofiler_iterate_buffer_tracing_kinds(rocprofiler_buffer_tracing_kind_cb_t callback, void* data)
{
    for(uint32_t i = 0; i < ROCPROFILER_BUFFER_TRACING_LAST; ++i)
    {
        auto _success = callback(static_cast<rocprofiler_buffer_tracing_kind_t>(i), data);
        if(_success != 0) break;
    }

    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_iterate_buffer_tracing_kind_operations(
    rocprofiler_buffer_tracing_kind_t              kind,
    rocprofiler_buffer_tracing_kind_operation_cb_t callback,
    void*                                          data)
{
    if(kind == ROCPROFILER_BUFFER_TRACING_HSA_API)
    {
        auto ops = rocprofiler::hsa::get_ids();
        for(const auto& itr : ops)
        {
            auto _success = callback(kind, itr, data);
            if(_success != 0) break;
        }
        return ROCPROFILER_STATUS_SUCCESS;
    }

    return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;
}
}

#undef RETURN_STATUS_ON_FAIL
