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

#include "lib/common/logging.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/context/domain.hpp"
#include "lib/rocprofiler-sdk/hip/hip.hpp"
#include "lib/rocprofiler-sdk/hip/stream.hpp"
#include "lib/rocprofiler-sdk/hsa/async_copy.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/hsa/memory_allocation.hpp"
#include "lib/rocprofiler-sdk/hsa/scratch_memory.hpp"
#include "lib/rocprofiler-sdk/kernel_dispatch/kernel_dispatch.hpp"
#include "lib/rocprofiler-sdk/kfd/kfd.hpp"
#include "lib/rocprofiler-sdk/marker/marker.hpp"
#include "lib/rocprofiler-sdk/ompt/ompt.hpp"
#include "lib/rocprofiler-sdk/rccl/rccl.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"
#include "lib/rocprofiler-sdk/rocdecode/rocdecode.hpp"
#include "lib/rocprofiler-sdk/rocjpeg/rocjpeg.hpp"
#include "lib/rocprofiler-sdk/runtime_initialization.hpp"

#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hip/table_id.h>
#include <rocprofiler-sdk/hsa/table_id.h>
#include <rocprofiler-sdk/marker/table_id.h>
#include <rocprofiler-sdk/rccl/table_id.h>
#include <rocprofiler-sdk/rocdecode/table_id.h>
#include <rocprofiler-sdk/rocjpeg/table_id.h>

#include <atomic>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
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
ROCPROFILER_BUFFER_TRACING_KIND_STRING(HSA_CORE_API)
ROCPROFILER_BUFFER_TRACING_KIND_STRING(HSA_AMD_EXT_API)
ROCPROFILER_BUFFER_TRACING_KIND_STRING(HSA_IMAGE_EXT_API)
ROCPROFILER_BUFFER_TRACING_KIND_STRING(HSA_FINALIZE_EXT_API)
ROCPROFILER_BUFFER_TRACING_KIND_STRING(HIP_RUNTIME_API)
ROCPROFILER_BUFFER_TRACING_KIND_STRING(HIP_COMPILER_API)
ROCPROFILER_BUFFER_TRACING_KIND_STRING(MARKER_CORE_API)
ROCPROFILER_BUFFER_TRACING_KIND_STRING(MARKER_CONTROL_API)
ROCPROFILER_BUFFER_TRACING_KIND_STRING(MARKER_NAME_API)
ROCPROFILER_BUFFER_TRACING_KIND_STRING(MEMORY_COPY)
ROCPROFILER_BUFFER_TRACING_KIND_STRING(MEMORY_ALLOCATION)
ROCPROFILER_BUFFER_TRACING_KIND_STRING(KERNEL_DISPATCH)
ROCPROFILER_BUFFER_TRACING_KIND_STRING(SCRATCH_MEMORY)
ROCPROFILER_BUFFER_TRACING_KIND_STRING(CORRELATION_ID_RETIREMENT)
ROCPROFILER_BUFFER_TRACING_KIND_STRING(RCCL_API)
ROCPROFILER_BUFFER_TRACING_KIND_STRING(OMPT)
ROCPROFILER_BUFFER_TRACING_KIND_STRING(RUNTIME_INITIALIZATION)
ROCPROFILER_BUFFER_TRACING_KIND_STRING(ROCDECODE_API)
ROCPROFILER_BUFFER_TRACING_KIND_STRING(ROCJPEG_API)
ROCPROFILER_BUFFER_TRACING_KIND_STRING(HIP_STREAM)
ROCPROFILER_BUFFER_TRACING_KIND_STRING(HIP_RUNTIME_API_EXT)
ROCPROFILER_BUFFER_TRACING_KIND_STRING(HIP_COMPILER_API_EXT)
ROCPROFILER_BUFFER_TRACING_KIND_STRING(ROCDECODE_API_EXT)
ROCPROFILER_BUFFER_TRACING_KIND_STRING(KFD_EVENT_PAGE_MIGRATE)
ROCPROFILER_BUFFER_TRACING_KIND_STRING(KFD_EVENT_PAGE_FAULT)
ROCPROFILER_BUFFER_TRACING_KIND_STRING(KFD_EVENT_QUEUE)
ROCPROFILER_BUFFER_TRACING_KIND_STRING(KFD_EVENT_UNMAP_FROM_GPU)
ROCPROFILER_BUFFER_TRACING_KIND_STRING(KFD_EVENT_DROPPED_EVENTS)
ROCPROFILER_BUFFER_TRACING_KIND_STRING(KFD_PAGE_MIGRATE)
ROCPROFILER_BUFFER_TRACING_KIND_STRING(KFD_PAGE_FAULT)
ROCPROFILER_BUFFER_TRACING_KIND_STRING(KFD_QUEUE)
ROCPROFILER_BUFFER_TRACING_KIND_STRING(MARKER_CORE_RANGE_API)

template <size_t Idx, size_t... Tail>
std::pair<const char*, size_t>
get_kind_name(rocprofiler_buffer_tracing_kind_t kind, std::index_sequence<Idx, Tail...>)
{
    if(kind == Idx) return buffer_tracing_kind_string<Idx>::value;
    // recursion until tail empty
    if constexpr(sizeof...(Tail) > 0) return get_kind_name(kind, std::index_sequence<Tail...>{});
    return {nullptr, 0};
}

auto
get_unsupported()
{
    auto unsupported = std::unordered_set<rocprofiler_buffer_tracing_kind_t>{};

    // #if ROCPROFILER_SDK_USE_SYSTEM_RCCL == 0
    //     // Built against RCCL which does not support API tracing
    //     unsupported.emplace(ROCPROFILER_BUFFER_TRACING_RCCL_API);
    // #endif

    return unsupported;
}
}  // namespace
}  // namespace buffer_tracing
}  // namespace rocprofiler

extern "C" {
rocprofiler_status_t
rocprofiler_configure_buffer_tracing_service(rocprofiler_context_id_t               context_id,
                                             rocprofiler_buffer_tracing_kind_t      kind,
                                             const rocprofiler_tracing_operation_t* operations,
                                             size_t                  operations_count,
                                             rocprofiler_buffer_id_t buffer_id)
{
    static auto unsupported = ::rocprofiler::buffer_tracing::get_unsupported();

    if(rocprofiler::registration::get_init_status() > -1)
        return ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED;

    if(unsupported.count(kind) > 0) return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;

    auto* ctx = rocprofiler::context::get_mutable_registered_context(context_id);

    if(!ctx) return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;

    if(buffer_id.handle == 0) return ROCPROFILER_STATUS_ERROR_BUFFER_NOT_FOUND;

    constexpr auto invalid_buffer_id = rocprofiler_buffer_id_t{0};

    if(!ctx->buffered_tracer)
    {
        ctx->buffered_tracer = std::make_unique<rocprofiler::context::buffer_tracing_service>();
        ctx->buffered_tracer->buffer_data.fill(invalid_buffer_id);
    }

    if(ctx->buffered_tracer->buffer_data.at(kind) != invalid_buffer_id)
        return ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED;

    RETURN_STATUS_ON_FAIL(rocprofiler::context::add_domain(ctx->buffered_tracer->domains, kind));

    ctx->buffered_tracer->buffer_data.at(kind) = buffer_id;

    for(size_t i = 0; i < operations_count; ++i)
    {
        RETURN_STATUS_ON_FAIL(rocprofiler::context::add_domain_op(
            ctx->buffered_tracer->domains, kind, operations[i]));
    }

    {
        static constexpr auto kfd_events =
            std::array{ROCPROFILER_BUFFER_TRACING_KFD_EVENT_PAGE_MIGRATE,
                       ROCPROFILER_BUFFER_TRACING_KFD_EVENT_PAGE_FAULT,
                       ROCPROFILER_BUFFER_TRACING_KFD_EVENT_QUEUE,
                       ROCPROFILER_BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU,
                       ROCPROFILER_BUFFER_TRACING_KFD_PAGE_MIGRATE,
                       ROCPROFILER_BUFFER_TRACING_KFD_PAGE_FAULT,
                       ROCPROFILER_BUFFER_TRACING_KFD_QUEUE};

        if(std::find(kfd_events.begin(), kfd_events.end(), kind) != kfd_events.end())
        {
            RETURN_STATUS_ON_FAIL(rocprofiler::kfd::init());
        }
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
                                                     rocprofiler_tracing_operation_t   operation,
                                                     const char**                      name,
                                                     uint64_t*                         name_len)
{
    if(kind < ROCPROFILER_BUFFER_TRACING_NONE || kind >= ROCPROFILER_BUFFER_TRACING_LAST)
        return ROCPROFILER_STATUS_ERROR_KIND_NOT_FOUND;

    const char* val = nullptr;
    switch(kind)
    {
        case ROCPROFILER_BUFFER_TRACING_NONE:
        case ROCPROFILER_BUFFER_TRACING_LAST:
        {
            return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
        }
        case ROCPROFILER_BUFFER_TRACING_HSA_CORE_API:
        {
            val = rocprofiler::hsa::name_by_id<ROCPROFILER_HSA_TABLE_ID_Core>(operation);
            break;
        }
        case ROCPROFILER_BUFFER_TRACING_HSA_AMD_EXT_API:
        {
            val = rocprofiler::hsa::name_by_id<ROCPROFILER_HSA_TABLE_ID_AmdExt>(operation);
            break;
        }
        case ROCPROFILER_BUFFER_TRACING_HSA_IMAGE_EXT_API:
        {
            val = rocprofiler::hsa::name_by_id<ROCPROFILER_HSA_TABLE_ID_ImageExt>(operation);
            break;
        }
        case ROCPROFILER_BUFFER_TRACING_HSA_FINALIZE_EXT_API:
        {
            val = rocprofiler::hsa::name_by_id<ROCPROFILER_HSA_TABLE_ID_FinalizeExt>(operation);
            break;
        }
        case ROCPROFILER_BUFFER_TRACING_MEMORY_COPY:
        {
            val = rocprofiler::hsa::async_copy::name_by_id(operation);
            break;
        }
        case ROCPROFILER_BUFFER_TRACING_MEMORY_ALLOCATION:
        {
            val = rocprofiler::hsa::memory_allocation::name_by_id(operation);
            break;
        }
        case ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY:
        {
            val = rocprofiler::hsa::scratch_memory::name_by_id(operation);
            break;
        }
        case ROCPROFILER_BUFFER_TRACING_MARKER_CORE_API:
        {
            val = rocprofiler::marker::name_by_id<ROCPROFILER_MARKER_TABLE_ID_RoctxCore>(operation);
            break;
        }
        case ROCPROFILER_BUFFER_TRACING_MARKER_CONTROL_API:
        {
            val = rocprofiler::marker::name_by_id<ROCPROFILER_MARKER_TABLE_ID_RoctxControl>(
                operation);
            break;
        }
        case ROCPROFILER_BUFFER_TRACING_MARKER_NAME_API:
        {
            val = rocprofiler::marker::name_by_id<ROCPROFILER_MARKER_TABLE_ID_RoctxName>(operation);
            break;
        }
        case ROCPROFILER_BUFFER_TRACING_RCCL_API:
        {
            val = rocprofiler::rccl::name_by_id<ROCPROFILER_RCCL_TABLE_ID>(operation);
            break;
        }
        case ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API:
        case ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API_EXT:
        {
            val = rocprofiler::hip::name_by_id<ROCPROFILER_HIP_TABLE_ID_Runtime>(operation);
            break;
        }
        case ROCPROFILER_BUFFER_TRACING_HIP_COMPILER_API:
        case ROCPROFILER_BUFFER_TRACING_HIP_COMPILER_API_EXT:
        {
            val = rocprofiler::hip::name_by_id<ROCPROFILER_HIP_TABLE_ID_Compiler>(operation);
            break;
        }
        case ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH:
        {
            val = rocprofiler::kernel_dispatch::name_by_id(operation);
            break;
        }
        case ROCPROFILER_BUFFER_TRACING_OMPT:
        {
            val = rocprofiler::ompt::name_by_id(operation);
            break;
        }
        case ROCPROFILER_BUFFER_TRACING_RUNTIME_INITIALIZATION:
        {
            val = rocprofiler::runtime_init::name_by_id(operation);
            break;
        }
        case ROCPROFILER_BUFFER_TRACING_CORRELATION_ID_RETIREMENT:
        {
            return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;
        }
        case ROCPROFILER_BUFFER_TRACING_ROCDECODE_API:
        case ROCPROFILER_BUFFER_TRACING_ROCDECODE_API_EXT:
        {
            val =
                rocprofiler::rocdecode::name_by_id<ROCPROFILER_ROCDECODE_TABLE_ID_CORE>(operation);
            break;
        }
        case ROCPROFILER_BUFFER_TRACING_ROCJPEG_API:
        {
            val = rocprofiler::rocjpeg::name_by_id<ROCPROFILER_ROCJPEG_TABLE_ID_CORE>(operation);
            break;
        }
        case ROCPROFILER_BUFFER_TRACING_HIP_STREAM:
        {
            val = rocprofiler::hip::stream::name_by_id(operation);
            break;
        }
        case ROCPROFILER_BUFFER_TRACING_MARKER_CORE_RANGE_API:
        {
            val = rocprofiler::marker::name_by_id<ROCPROFILER_MARKER_TABLE_ID_RoctxCoreRange>(
                operation);
            break;
        }
        case ROCPROFILER_BUFFER_TRACING_KFD_EVENT_PAGE_MIGRATE:
        case ROCPROFILER_BUFFER_TRACING_KFD_EVENT_PAGE_FAULT:
        case ROCPROFILER_BUFFER_TRACING_KFD_EVENT_QUEUE:
        case ROCPROFILER_BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU:
        case ROCPROFILER_BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS:
        case ROCPROFILER_BUFFER_TRACING_KFD_PAGE_MIGRATE:
        case ROCPROFILER_BUFFER_TRACING_KFD_PAGE_FAULT:
        case ROCPROFILER_BUFFER_TRACING_KFD_QUEUE:
        {
            val = rocprofiler::kfd::name_by_id(kind, operation);
            break;
        }
    };

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
    auto ops = std::vector<uint32_t>{};
    switch(kind)
    {
        case ROCPROFILER_BUFFER_TRACING_NONE:
        case ROCPROFILER_BUFFER_TRACING_LAST:
        {
            return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
        }
        case ROCPROFILER_BUFFER_TRACING_HSA_CORE_API:
        {
            ops = rocprofiler::hsa::get_ids<ROCPROFILER_HSA_TABLE_ID_Core>();
            break;
        }
        case ROCPROFILER_BUFFER_TRACING_HSA_AMD_EXT_API:
        {
            ops = rocprofiler::hsa::get_ids<ROCPROFILER_HSA_TABLE_ID_AmdExt>();
            break;
        }
        case ROCPROFILER_BUFFER_TRACING_HSA_IMAGE_EXT_API:
        {
            ops = rocprofiler::hsa::get_ids<ROCPROFILER_HSA_TABLE_ID_ImageExt>();
            break;
        }
        case ROCPROFILER_BUFFER_TRACING_HSA_FINALIZE_EXT_API:
        {
            ops = rocprofiler::hsa::get_ids<ROCPROFILER_HSA_TABLE_ID_FinalizeExt>();
            break;
        }
        case ROCPROFILER_BUFFER_TRACING_MEMORY_COPY:
        {
            ops = rocprofiler::hsa::async_copy::get_ids();
            break;
        }
        case ROCPROFILER_BUFFER_TRACING_MEMORY_ALLOCATION:
        {
            ops = rocprofiler::hsa::memory_allocation::get_ids();
            break;
        }
        case ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY:
        {
            ops = rocprofiler::hsa::scratch_memory::get_ids();
            break;
        }
        case ROCPROFILER_BUFFER_TRACING_MARKER_CORE_API:
        {
            ops = rocprofiler::marker::get_ids<ROCPROFILER_MARKER_TABLE_ID_RoctxCore>();
            break;
        }
        case ROCPROFILER_BUFFER_TRACING_MARKER_CONTROL_API:
        {
            ops = rocprofiler::marker::get_ids<ROCPROFILER_MARKER_TABLE_ID_RoctxControl>();
            break;
        }
        case ROCPROFILER_BUFFER_TRACING_MARKER_NAME_API:
        {
            ops = rocprofiler::marker::get_ids<ROCPROFILER_MARKER_TABLE_ID_RoctxName>();
            break;
        }
        case ROCPROFILER_BUFFER_TRACING_RCCL_API:
        {
            ops = rocprofiler::rccl::get_ids<ROCPROFILER_RCCL_TABLE_ID>();
            break;
        }
        case ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API:
        case ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API_EXT:
        {
            ops = rocprofiler::hip::get_ids<ROCPROFILER_HIP_TABLE_ID_Runtime>();
            break;
        }
        case ROCPROFILER_BUFFER_TRACING_HIP_COMPILER_API:
        case ROCPROFILER_BUFFER_TRACING_HIP_COMPILER_API_EXT:
        {
            ops = rocprofiler::hip::get_ids<ROCPROFILER_HIP_TABLE_ID_Compiler>();
            break;
        }
        case ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH:
        {
            ops = rocprofiler::kernel_dispatch::get_ids();
            break;
        }
        case ROCPROFILER_BUFFER_TRACING_OMPT:
        {
            ops = rocprofiler::ompt::get_ids();
            break;
        }
        case ROCPROFILER_BUFFER_TRACING_RUNTIME_INITIALIZATION:
        {
            ops = rocprofiler::runtime_init::get_ids();
            break;
        }
        case ROCPROFILER_BUFFER_TRACING_CORRELATION_ID_RETIREMENT:
        {
            return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;
        }
        case ROCPROFILER_BUFFER_TRACING_ROCDECODE_API:
        case ROCPROFILER_BUFFER_TRACING_ROCDECODE_API_EXT:
        {
            ops = rocprofiler::rocdecode::get_ids<ROCPROFILER_ROCDECODE_TABLE_ID_CORE>();
            break;
        }
        case ROCPROFILER_BUFFER_TRACING_ROCJPEG_API:
        {
            ops = rocprofiler::rocjpeg::get_ids<ROCPROFILER_ROCJPEG_TABLE_ID_CORE>();
            break;
        }
        case ROCPROFILER_BUFFER_TRACING_HIP_STREAM:
        {
            ops = rocprofiler::hip::stream::get_ids();
            break;
        }
        case ROCPROFILER_BUFFER_TRACING_MARKER_CORE_RANGE_API:
        {
            ops = rocprofiler::marker::get_ids<ROCPROFILER_MARKER_TABLE_ID_RoctxCoreRange>();
            break;
        }
        case ROCPROFILER_BUFFER_TRACING_KFD_EVENT_PAGE_MIGRATE:
        case ROCPROFILER_BUFFER_TRACING_KFD_EVENT_PAGE_FAULT:
        case ROCPROFILER_BUFFER_TRACING_KFD_EVENT_QUEUE:
        case ROCPROFILER_BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU:
        case ROCPROFILER_BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS:
        case ROCPROFILER_BUFFER_TRACING_KFD_PAGE_MIGRATE:
        case ROCPROFILER_BUFFER_TRACING_KFD_PAGE_FAULT:
        case ROCPROFILER_BUFFER_TRACING_KFD_QUEUE:
        {
            ops = rocprofiler::kfd::get_ids(kind);
            break;
        }
    }

    for(const auto& itr : ops)
    {
        auto _success = callback(kind, itr, data);
        if(_success != 0) break;
    }
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_iterate_buffer_tracing_record_args(
    rocprofiler_record_header_t                    record,
    rocprofiler_buffer_tracing_operation_args_cb_t callback,
    void*                                          user_data)
{
    switch(record.kind)
    {
        case ROCPROFILER_BUFFER_TRACING_NONE:
        case ROCPROFILER_BUFFER_TRACING_LAST:
        {
            return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
        }
        case ROCPROFILER_BUFFER_TRACING_HSA_CORE_API:
        case ROCPROFILER_BUFFER_TRACING_HSA_AMD_EXT_API:
        case ROCPROFILER_BUFFER_TRACING_HSA_IMAGE_EXT_API:
        case ROCPROFILER_BUFFER_TRACING_HSA_FINALIZE_EXT_API:
        case ROCPROFILER_BUFFER_TRACING_MARKER_CORE_API:
        case ROCPROFILER_BUFFER_TRACING_MARKER_CONTROL_API:
        case ROCPROFILER_BUFFER_TRACING_MARKER_NAME_API:
        case ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY:
        case ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH:
        case ROCPROFILER_BUFFER_TRACING_MEMORY_COPY:
        case ROCPROFILER_BUFFER_TRACING_RCCL_API:
        {
            return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;
        }
        case ROCPROFILER_BUFFER_TRACING_HIP_COMPILER_API_EXT:
        {
            auto* _payload =
                static_cast<rocprofiler_buffer_tracing_hip_api_ext_record_t*>(record.payload);
            rocprofiler::hip::iterate_args<ROCPROFILER_HIP_TABLE_ID_Compiler>(
                _payload->operation, _payload->args, callback, user_data);
            return ROCPROFILER_STATUS_SUCCESS;
        }
        case ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API_EXT:
        {
            auto* _payload =
                static_cast<rocprofiler_buffer_tracing_hip_api_ext_record_t*>(record.payload);
            rocprofiler::hip::iterate_args<ROCPROFILER_HIP_TABLE_ID_Runtime>(
                _payload->operation, _payload->args, callback, user_data);
            return ROCPROFILER_STATUS_SUCCESS;
        }
        case ROCPROFILER_BUFFER_TRACING_ROCDECODE_API_EXT:
        {
            auto* _payload =
                static_cast<rocprofiler_buffer_tracing_rocdecode_api_ext_record_t*>(record.payload);
            rocprofiler::rocdecode::iterate_args<ROCPROFILER_ROCDECODE_TABLE_ID_CORE>(
                _payload->operation, _payload->args, callback, user_data);
            return ROCPROFILER_STATUS_SUCCESS;
        }
    }

    return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;
}
}

#undef RETURN_STATUS_ON_FAIL
