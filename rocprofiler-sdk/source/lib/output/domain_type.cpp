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

#include "domain_type.hpp"

#include <utility>

namespace
{
template <domain_type DomainT>
struct domain_type_name;

#define DEFINE_BUFFER_TYPE_NAME(ENUM_VALUE, COLUMN_NAME, TRACE_FILENAME, STATS_FILENAME)           \
    template <>                                                                                    \
    struct domain_type_name<domain_type::ENUM_VALUE>                                               \
    {                                                                                              \
        static constexpr auto column_name    = COLUMN_NAME;                                        \
        static constexpr auto trace_filename = TRACE_FILENAME;                                     \
        static constexpr auto stats_filename = STATS_FILENAME;                                     \
    };

DEFINE_BUFFER_TYPE_NAME(HSA, "HSA_API", "hsa_api_trace", "hsa_api_stats")
DEFINE_BUFFER_TYPE_NAME(HIP, "HIP_API", "hip_api_trace", "hip_api_stats")
DEFINE_BUFFER_TYPE_NAME(MARKER, "MARKER_API", "marker_api_trace", "marker_api_stats")
DEFINE_BUFFER_TYPE_NAME(KERNEL_DISPATCH, "KERNEL_DISPATCH", "kernel_trace", "kernel_stats")
DEFINE_BUFFER_TYPE_NAME(MEMORY_COPY, "MEMORY_COPY", "memory_copy_trace", "memory_copy_stats")
DEFINE_BUFFER_TYPE_NAME(SCRATCH_MEMORY,
                        "SCRATCH_MEMORY",
                        "scratch_memory_trace",
                        "scratch_memory_stats")
DEFINE_BUFFER_TYPE_NAME(COUNTER_COLLECTION,
                        "COUNTER_COLLECTION",
                        "counter_collection",
                        "counter_collection_stats")
DEFINE_BUFFER_TYPE_NAME(RCCL, "RCCL_API", "rccl_api_trace", "rccl_api_stats")
DEFINE_BUFFER_TYPE_NAME(MEMORY_ALLOCATION,
                        "MEMORY_ALLOCATION",
                        "memory_allocation_trace",
                        "memory_allocation_stats")
DEFINE_BUFFER_TYPE_NAME(COUNTER_VALUES, "COUNTER_VALUES", "counter_values", "no_filename")
DEFINE_BUFFER_TYPE_NAME(PC_SAMPLING_HOST_TRAP,
                        "PC_SAMPLING_HOST_TRAP",
                        "pc_sampling_host_trap",
                        "pc_sampling_host_trap_stats")
DEFINE_BUFFER_TYPE_NAME(ROCDECODE, "ROCDECODE_API", "rocdecode_api_trace", "rocdecode_api_stats")
DEFINE_BUFFER_TYPE_NAME(ROCJPEG, "ROCJPEG_API", "rocjpeg_api_trace", "rocjpeg_api_stats")
DEFINE_BUFFER_TYPE_NAME(PC_SAMPLING_STOCHASTIC,
                        "PC_SAMPLING_STOCHASTIC",
                        "pc_sampling_stochastic",
                        "pc_sampling_stochastic_stats")

#undef DEFINE_BUFFER_TYPE_NAME

template <size_t Idx, size_t... TailIdx>
std::string_view
get_domain_trace_file_name(domain_type _buffer_type, std::index_sequence<Idx, TailIdx...>)
{
    if(static_cast<size_t>(_buffer_type) == Idx)
        return domain_type_name<static_cast<domain_type>(Idx)>::trace_filename;
    if constexpr(sizeof...(TailIdx) > 0)
        return get_domain_trace_file_name(_buffer_type, std::index_sequence<TailIdx...>{});
    return std::string_view{};
}

template <size_t Idx, size_t... TailIdx>
std::string_view
get_domain_stats_file_name(domain_type _buffer_type, std::index_sequence<Idx, TailIdx...>)
{
    if(static_cast<size_t>(_buffer_type) == Idx)
        return domain_type_name<static_cast<domain_type>(Idx)>::stats_filename;
    if constexpr(sizeof...(TailIdx) > 0)
        return get_domain_stats_file_name(_buffer_type, std::index_sequence<TailIdx...>{});
    return std::string_view{};
}

template <size_t Idx, size_t... IdxTail>
std::string_view
get_domain_column_name(domain_type buffer_type, std::index_sequence<Idx, IdxTail...>)
{
    if(static_cast<size_t>(buffer_type) == Idx)
        return domain_type_name<static_cast<domain_type>(Idx)>::column_name;
    if constexpr(sizeof...(IdxTail) > 0)
        return get_domain_column_name(buffer_type, std::index_sequence<IdxTail...>{});

    return std::string_view{};
}
}  // namespace

std::string_view
get_domain_trace_file_name(domain_type _buffer_type)
{
    constexpr auto buffer_type_last_v = static_cast<size_t>(domain_type::LAST);

    return get_domain_trace_file_name(_buffer_type, std::make_index_sequence<buffer_type_last_v>{});
}

std::string_view
get_domain_stats_file_name(domain_type _buffer_type)
{
    constexpr auto buffer_type_last_v = static_cast<size_t>(domain_type::LAST);

    return get_domain_stats_file_name(_buffer_type, std::make_index_sequence<buffer_type_last_v>{});
}

std::string_view
get_domain_column_name(domain_type buffer_type)
{
    constexpr auto last_v = static_cast<size_t>(domain_type::LAST);
    return get_domain_column_name(buffer_type, std::make_index_sequence<last_v>{});
}
