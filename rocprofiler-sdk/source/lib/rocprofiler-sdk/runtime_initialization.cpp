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

#include "lib/rocprofiler-sdk/runtime_initialization.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/tracing/fwd.hpp"
#include "lib/rocprofiler-sdk/tracing/tracing.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace rocprofiler
{
namespace runtime_init
{
namespace
{
template <size_t OpIdx>
struct runtime_init_info;

#define SPECIALIZE_RUNTIME_INIT_INFO(NAME, PRETTY_NAME)                                            \
    template <>                                                                                    \
    struct runtime_init_info<ROCPROFILER_RUNTIME_INITIALIZATION_##NAME>                            \
    {                                                                                              \
        static constexpr auto operation_idx = ROCPROFILER_RUNTIME_INITIALIZATION_##NAME;           \
        static constexpr auto name          = "RUNTIME_INITIALIZATION_" #NAME;                     \
        static constexpr auto pretty_name   = PRETTY_NAME;                                         \
    };

SPECIALIZE_RUNTIME_INIT_INFO(NONE, "<unknown-runtime>")
SPECIALIZE_RUNTIME_INIT_INFO(HSA, "HSA runtime")
SPECIALIZE_RUNTIME_INIT_INFO(HIP, "HIP runtime")
SPECIALIZE_RUNTIME_INIT_INFO(MARKER, "Marker (ROCTx) runtime")
SPECIALIZE_RUNTIME_INIT_INFO(RCCL, "RCCL runtime")
SPECIALIZE_RUNTIME_INIT_INFO(ROCDECODE, "rocDecode runtime")
SPECIALIZE_RUNTIME_INIT_INFO(ROCJPEG, "rocJPEG runtime")

#undef SPECIALIZE_RUNTIME_INIT_INFO

template <size_t Idx, size_t... IdxTail>
std::pair<const char*, const char*>
name_by_id(const uint32_t id, std::index_sequence<Idx, IdxTail...>)
{
    if(Idx == id) return {runtime_init_info<Idx>::name, runtime_init_info<Idx>::pretty_name};
    if constexpr(sizeof...(IdxTail) > 0)
        return name_by_id(id, std::index_sequence<IdxTail...>{});
    else
        return {nullptr, nullptr};
}

template <size_t... Idx>
void
get_ids(std::vector<uint32_t>& _id_list, std::index_sequence<Idx...>)
{
    auto _emplace = [](auto& _vec, uint32_t _v) {
        if(_v < static_cast<uint32_t>(ROCPROFILER_RUNTIME_INITIALIZATION_LAST))
            _vec.emplace_back(_v);
    };

    (_emplace(_id_list, runtime_init_info<Idx>::operation_idx), ...);
}
}  // namespace

const char*
name_by_id(uint32_t id)
{
    return name_by_id(id, std::make_index_sequence<ROCPROFILER_RUNTIME_INITIALIZATION_LAST>{})
        .first;
}

const char*
pretty_name_by_id(uint32_t id)
{
    return name_by_id(id, std::make_index_sequence<ROCPROFILER_RUNTIME_INITIALIZATION_LAST>{})
        .second;
}

std::vector<uint32_t>
get_ids()
{
    auto _data = std::vector<uint32_t>{};
    _data.reserve(ROCPROFILER_RUNTIME_INITIALIZATION_LAST);
    get_ids(_data, std::make_index_sequence<ROCPROFILER_RUNTIME_INITIALIZATION_LAST>{});
    return _data;
}

void
initialize(rocprofiler_runtime_initialization_operation_t operation_idx,
           uint64_t                                       lib_version,
           uint64_t                                       lib_instance)
{
    using runtime_init_record_t = rocprofiler_buffer_tracing_runtime_initialization_record_t;
    using runtime_init_data_t   = rocprofiler_callback_tracing_runtime_initialization_data_t;

    constexpr auto callback_domain_idx = ROCPROFILER_CALLBACK_TRACING_RUNTIME_INITIALIZATION;
    constexpr auto buffered_domain_idx = ROCPROFILER_BUFFER_TRACING_RUNTIME_INITIALIZATION;
    constexpr auto corr_id =
        rocprofiler_correlation_id_t{0, rocprofiler_user_data_t{.value = 0}, 0};

    ROCP_INFO << pretty_name_by_id(operation_idx) << " has been initialized";

    auto thr_id = common::get_tid();
    auto data   = tracing::tracing_data{};

    tracing::populate_contexts(callback_domain_idx, buffered_domain_idx, operation_idx, data);

    auto tracer_data   = common::init_public_api_struct(runtime_init_data_t{});
    auto buffer_record = common::init_public_api_struct(runtime_init_record_t{});

    {
        tracer_data.version  = lib_version;
        tracer_data.instance = lib_instance;
        tracing::execute_phase_none_callbacks(data.callback_contexts,
                                              thr_id,
                                              corr_id.internal,
                                              data.external_correlation_ids,
                                              corr_id.ancestor,
                                              callback_domain_idx,
                                              operation_idx,
                                              tracer_data);
    }

    {
        buffer_record.version   = lib_version;
        buffer_record.instance  = lib_instance;
        buffer_record.timestamp = common::timestamp_ns();
        tracing::execute_buffer_record_emplace(data.buffered_contexts,
                                               thr_id,
                                               corr_id.internal,
                                               data.external_correlation_ids,
                                               corr_id.ancestor,
                                               buffered_domain_idx,
                                               operation_idx,
                                               buffer_record);
    }
}

void
finalize()
{}
}  // namespace runtime_init
}  // namespace rocprofiler
