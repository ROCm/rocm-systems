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
//

#pragma once

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/counters.h>
#include <rocprofiler-sdk/defines.h>
#include <rocprofiler-sdk/experimental/thread-trace/trace_decoder.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hsa.h>
#include <rocprofiler-sdk/internal_threading.h>

#include "rocprofiler-sdk/cxx/details/mpl.hpp"

#include <string_view>
#include <tuple>

#define ROCPROFILER_CXX_DECLARE_OPERATORS(TYPE)                                                    \
    bool operator==(TYPE lhs, TYPE rhs) ROCPROFILER_ATTRIBUTE(pure);                               \
    bool operator!=(TYPE lhs, TYPE rhs) ROCPROFILER_ATTRIBUTE(pure);                               \
    bool operator<(TYPE lhs, TYPE rhs) ROCPROFILER_ATTRIBUTE(pure);                                \
    bool operator>(TYPE lhs, TYPE rhs) ROCPROFILER_ATTRIBUTE(pure);                                \
    bool operator<=(TYPE lhs, TYPE rhs) ROCPROFILER_ATTRIBUTE(pure);                               \
    bool operator>=(TYPE lhs, TYPE rhs) ROCPROFILER_ATTRIBUTE(pure);

#define ROCPROFILER_CXX_DEFINE_NE_OPERATOR(TYPE)                                                   \
    inline bool operator!=(TYPE lhs, TYPE rhs) { return !(lhs == rhs); }

#define ROCPROFILER_CXX_DEFINE_EQ_HANDLE_OPERATOR(TYPE)                                            \
    inline bool operator==(TYPE lhs, TYPE rhs)                                                     \
    {                                                                                              \
        return ::rocprofiler::sdk::operators::equal(lhs, rhs);                                     \
    }

#define ROCPROFILER_CXX_DEFINE_LT_HANDLE_OPERATOR(TYPE)                                            \
    inline bool operator<(TYPE lhs, TYPE rhs)                                                      \
    {                                                                                              \
        return ::rocprofiler::sdk::operators::less(lhs, rhs);                                      \
    }

#define ROCPROFILER_CXX_DEFINE_COMPARE_OPERATORS(TYPE)                                             \
    inline bool operator>(TYPE lhs, TYPE rhs) { return !(lhs == rhs || lhs < rhs); }               \
    inline bool operator<=(TYPE lhs, TYPE rhs) { return (lhs == rhs || lhs < rhs); }               \
    inline bool operator>=(TYPE lhs, TYPE rhs) { return !(lhs < rhs); }

namespace rocprofiler
{
namespace sdk
{
namespace operators
{
template <typename Tp>
bool
equal(Tp lhs, Tp rhs) ROCPROFILER_ATTRIBUTE(pure);

template <typename Tp>
bool
less(Tp lhs, Tp rhs) ROCPROFILER_ATTRIBUTE(pure);

template <typename Tp>
bool
equal(Tp lhs, Tp rhs)
{
    static_assert(sizeof(Tp) == sizeof(uint64_t), "error! only for opaque handle types");
    return lhs.handle == rhs.handle;
}

template <typename Tp>
bool
less(Tp lhs, Tp rhs)
{
    static_assert(sizeof(Tp) == sizeof(uint64_t), "error! only for opaque handle types");
    return lhs.handle < rhs.handle;
}

namespace detail
{
template <typename Tp>
struct sv_trait
{
    static constexpr auto is_string_type() noexcept
    {
        return mpl::is_string_type<mpl::unqualified_identity_t<Tp>>::value;
    }

    using type = std::conditional_t<is_string_type(), std::string_view, Tp&>;

    constexpr type operator()(Tp& v) noexcept
    {
        if constexpr(is_string_type())
            return std::string_view{v};
        else
            return v;
    }
};
}  // namespace detail

// tie_sv(): deduce each Tp from your lvalues, then build tuple<...> out of
// either Tp& or std::string_view, calling operator() on each.
template <typename... Tp>
constexpr auto
tie_sv(Tp&... vs) noexcept
{
    return std::make_tuple(detail::sv_trait<Tp>{}(vs)...);
}
}  // namespace operators
}  // namespace sdk
}  // namespace rocprofiler

// declaration of operator== and operator!=
ROCPROFILER_CXX_DECLARE_OPERATORS(rocprofiler_context_id_t)
ROCPROFILER_CXX_DECLARE_OPERATORS(rocprofiler_address_t)
ROCPROFILER_CXX_DECLARE_OPERATORS(rocprofiler_agent_id_t)
ROCPROFILER_CXX_DECLARE_OPERATORS(rocprofiler_queue_id_t)
ROCPROFILER_CXX_DECLARE_OPERATORS(rocprofiler_stream_id_t)
ROCPROFILER_CXX_DECLARE_OPERATORS(rocprofiler_buffer_id_t)
ROCPROFILER_CXX_DECLARE_OPERATORS(rocprofiler_counter_id_t)
ROCPROFILER_CXX_DECLARE_OPERATORS(rocprofiler_counter_config_id_t)
ROCPROFILER_CXX_DECLARE_OPERATORS(rocprofiler_callback_thread_t)
ROCPROFILER_CXX_DECLARE_OPERATORS(hsa_agent_t)
ROCPROFILER_CXX_DECLARE_OPERATORS(hsa_signal_t)
ROCPROFILER_CXX_DECLARE_OPERATORS(hsa_executable_t)
ROCPROFILER_CXX_DECLARE_OPERATORS(const rocprofiler_agent_v0_t&)
ROCPROFILER_CXX_DECLARE_OPERATORS(rocprofiler_dim3_t)
ROCPROFILER_CXX_DECLARE_OPERATORS(hsa_region_t)
ROCPROFILER_CXX_DECLARE_OPERATORS(hsa_amd_memory_pool_t)
ROCPROFILER_CXX_DECLARE_OPERATORS(const rocprofiler_counter_record_dimension_info_t&)
ROCPROFILER_CXX_DECLARE_OPERATORS(const rocprofiler_counter_record_dimension_instance_info_t&)
ROCPROFILER_CXX_DECLARE_OPERATORS(const rocprofiler_counter_dimension_info_t&)
ROCPROFILER_CXX_DECLARE_OPERATORS(rocprofiler_version_triplet_t)
ROCPROFILER_CXX_DECLARE_OPERATORS(rocprofiler_thread_trace_decoder_id_t)

// definitions of operator==
ROCPROFILER_CXX_DEFINE_EQ_HANDLE_OPERATOR(rocprofiler_context_id_t)
ROCPROFILER_CXX_DEFINE_EQ_HANDLE_OPERATOR(rocprofiler_address_t)
ROCPROFILER_CXX_DEFINE_EQ_HANDLE_OPERATOR(rocprofiler_agent_id_t)
ROCPROFILER_CXX_DEFINE_EQ_HANDLE_OPERATOR(rocprofiler_queue_id_t)
ROCPROFILER_CXX_DEFINE_EQ_HANDLE_OPERATOR(rocprofiler_stream_id_t)
ROCPROFILER_CXX_DEFINE_EQ_HANDLE_OPERATOR(rocprofiler_buffer_id_t)
ROCPROFILER_CXX_DEFINE_EQ_HANDLE_OPERATOR(rocprofiler_counter_id_t)
ROCPROFILER_CXX_DEFINE_EQ_HANDLE_OPERATOR(rocprofiler_counter_config_id_t)
ROCPROFILER_CXX_DEFINE_EQ_HANDLE_OPERATOR(rocprofiler_callback_thread_t)
ROCPROFILER_CXX_DEFINE_EQ_HANDLE_OPERATOR(hsa_agent_t)
ROCPROFILER_CXX_DEFINE_EQ_HANDLE_OPERATOR(hsa_signal_t)
ROCPROFILER_CXX_DEFINE_EQ_HANDLE_OPERATOR(hsa_executable_t)
ROCPROFILER_CXX_DEFINE_EQ_HANDLE_OPERATOR(hsa_region_t)
ROCPROFILER_CXX_DEFINE_EQ_HANDLE_OPERATOR(hsa_amd_memory_pool_t)
ROCPROFILER_CXX_DEFINE_EQ_HANDLE_OPERATOR(rocprofiler_thread_trace_decoder_id_t)

inline bool
operator==(const rocprofiler_agent_v0_t& lhs, const rocprofiler_agent_v0_t& rhs)
{
    return (lhs.id == rhs.id);
}

inline bool
operator==(rocprofiler_dim3_t lhs, rocprofiler_dim3_t rhs)
{
    return std::tie(lhs.x, lhs.y, lhs.z) == std::tie(rhs.x, rhs.y, rhs.z);
}

inline bool
operator==(const rocprofiler_counter_record_dimension_info_t& lhs,
           const rocprofiler_counter_record_dimension_info_t& rhs)
{
    namespace op = ::rocprofiler::sdk::operators;
    return op::tie_sv(lhs.id, lhs.instance_size, lhs.name) ==
           op::tie_sv(rhs.id, rhs.instance_size, rhs.name);
}

inline bool
operator==(const rocprofiler_counter_dimension_info_t& lhs,
           const rocprofiler_counter_dimension_info_t& rhs)
{
    namespace op = ::rocprofiler::sdk::operators;
    return op::tie_sv(lhs.dimension_name, lhs.index, lhs.size) ==
           op::tie_sv(rhs.dimension_name, rhs.index, rhs.size);
}

inline bool
operator==(const rocprofiler_counter_record_dimension_instance_info_t& lhs,
           const rocprofiler_counter_record_dimension_instance_info_t& rhs)
{
    if(std::tie(lhs.instance_id, lhs.counter_id, lhs.dimensions_count) !=
       std::tie(rhs.instance_id, rhs.counter_id, rhs.dimensions_count))
    {
        return false;
    }

    for(uint64_t i = 0; i < lhs.dimensions_count; ++i)
    {
        if(lhs.dimensions[i] == nullptr && rhs.dimensions[i] == nullptr) continue;
        if(!lhs.dimensions[i] || !rhs.dimensions[i] || *lhs.dimensions[i] != *rhs.dimensions[i])
        {
            return false;
        }
    }

    return true;
}

inline bool
operator==(rocprofiler_version_triplet_t lhs, rocprofiler_version_triplet_t rhs)
{
    return std::tie(lhs.major, lhs.minor, lhs.patch) == std::tie(rhs.major, rhs.minor, rhs.patch);
}

// definitions of operator!=
ROCPROFILER_CXX_DEFINE_NE_OPERATOR(rocprofiler_context_id_t)
ROCPROFILER_CXX_DEFINE_NE_OPERATOR(rocprofiler_address_t)
ROCPROFILER_CXX_DEFINE_NE_OPERATOR(rocprofiler_agent_id_t)
ROCPROFILER_CXX_DEFINE_NE_OPERATOR(rocprofiler_queue_id_t)
ROCPROFILER_CXX_DEFINE_NE_OPERATOR(rocprofiler_stream_id_t)
ROCPROFILER_CXX_DEFINE_NE_OPERATOR(rocprofiler_buffer_id_t)
ROCPROFILER_CXX_DEFINE_NE_OPERATOR(rocprofiler_counter_id_t)
ROCPROFILER_CXX_DEFINE_NE_OPERATOR(rocprofiler_counter_config_id_t)
ROCPROFILER_CXX_DEFINE_NE_OPERATOR(rocprofiler_callback_thread_t)
ROCPROFILER_CXX_DEFINE_NE_OPERATOR(hsa_agent_t)
ROCPROFILER_CXX_DEFINE_NE_OPERATOR(hsa_signal_t)
ROCPROFILER_CXX_DEFINE_NE_OPERATOR(hsa_executable_t)
ROCPROFILER_CXX_DEFINE_NE_OPERATOR(const rocprofiler_agent_v0_t&)
ROCPROFILER_CXX_DEFINE_NE_OPERATOR(rocprofiler_dim3_t)
ROCPROFILER_CXX_DEFINE_NE_OPERATOR(hsa_region_t)
ROCPROFILER_CXX_DEFINE_NE_OPERATOR(hsa_amd_memory_pool_t)
ROCPROFILER_CXX_DEFINE_NE_OPERATOR(rocprofiler_version_triplet_t)
ROCPROFILER_CXX_DEFINE_NE_OPERATOR(rocprofiler_thread_trace_decoder_id_t)

// definitions of operator<
ROCPROFILER_CXX_DEFINE_LT_HANDLE_OPERATOR(rocprofiler_context_id_t)
ROCPROFILER_CXX_DEFINE_LT_HANDLE_OPERATOR(rocprofiler_address_t)
ROCPROFILER_CXX_DEFINE_LT_HANDLE_OPERATOR(rocprofiler_agent_id_t)
ROCPROFILER_CXX_DEFINE_LT_HANDLE_OPERATOR(rocprofiler_queue_id_t)
ROCPROFILER_CXX_DEFINE_LT_HANDLE_OPERATOR(rocprofiler_stream_id_t)
ROCPROFILER_CXX_DEFINE_LT_HANDLE_OPERATOR(rocprofiler_buffer_id_t)
ROCPROFILER_CXX_DEFINE_LT_HANDLE_OPERATOR(rocprofiler_counter_id_t)
ROCPROFILER_CXX_DEFINE_LT_HANDLE_OPERATOR(rocprofiler_counter_config_id_t)
ROCPROFILER_CXX_DEFINE_LT_HANDLE_OPERATOR(rocprofiler_callback_thread_t)
ROCPROFILER_CXX_DEFINE_LT_HANDLE_OPERATOR(hsa_agent_t)
ROCPROFILER_CXX_DEFINE_LT_HANDLE_OPERATOR(hsa_signal_t)
ROCPROFILER_CXX_DEFINE_LT_HANDLE_OPERATOR(hsa_executable_t)
ROCPROFILER_CXX_DEFINE_LT_HANDLE_OPERATOR(hsa_region_t)
ROCPROFILER_CXX_DEFINE_LT_HANDLE_OPERATOR(hsa_amd_memory_pool_t)
ROCPROFILER_CXX_DEFINE_LT_HANDLE_OPERATOR(rocprofiler_thread_trace_decoder_id_t)

inline bool
operator<(const rocprofiler_counter_record_dimension_info_t& lhs,
          const rocprofiler_counter_record_dimension_info_t& rhs)
{
    namespace op = ::rocprofiler::sdk::operators;
    return op::tie_sv(lhs.id, lhs.instance_size, lhs.name) <
           op::tie_sv(rhs.id, rhs.instance_size, rhs.name);
}

inline bool
operator<(const rocprofiler_counter_dimension_info_t& lhs,
          const rocprofiler_counter_dimension_info_t& rhs)
{
    namespace op = ::rocprofiler::sdk::operators;
    return op::tie_sv(lhs.dimension_name, lhs.index, lhs.size) <
           op::tie_sv(rhs.dimension_name, rhs.index, rhs.size);
}

inline bool
operator<(const rocprofiler_counter_record_dimension_instance_info_t& lhs,
          const rocprofiler_counter_record_dimension_instance_info_t& rhs)
{
    if(lhs.instance_id != rhs.instance_id) return lhs.instance_id < rhs.instance_id;
    if(lhs.counter_id != rhs.counter_id) return lhs.counter_id < rhs.counter_id;
    if(lhs.dimensions_count != rhs.dimensions_count)
        return lhs.dimensions_count < rhs.dimensions_count;

    for(uint64_t i = 0; i < lhs.dimensions_count; ++i)
    {
        if(!lhs.dimensions[i] || !rhs.dimensions[i]) return *lhs.dimensions[i] < *rhs.dimensions[i];
    }

    return false;
}

inline bool
operator<(const rocprofiler_agent_v0_t& lhs, const rocprofiler_agent_v0_t& rhs)
{
    return (lhs.id < rhs.id);
}

inline bool
operator<(rocprofiler_dim3_t lhs, rocprofiler_dim3_t rhs)
{
    const auto magnitude = [](rocprofiler_dim3_t dim_v) { return dim_v.x * dim_v.y * dim_v.z; };
    auto       lhs_m     = magnitude(lhs);
    auto       rhs_m     = magnitude(rhs);

    return (lhs_m == rhs_m) ? std::tie(lhs.x, lhs.y, lhs.z) < std::tie(rhs.x, rhs.y, rhs.z)
                            : (lhs_m < rhs_m);
}

inline bool
operator<(rocprofiler_version_triplet_t lhs, rocprofiler_version_triplet_t rhs)
{
    return std::tie(lhs.major, lhs.minor, lhs.patch) < std::tie(rhs.major, rhs.minor, rhs.patch);
}

// definitions of operator>, operator<=, operator>=
ROCPROFILER_CXX_DEFINE_COMPARE_OPERATORS(rocprofiler_context_id_t)
ROCPROFILER_CXX_DEFINE_COMPARE_OPERATORS(rocprofiler_address_t)
ROCPROFILER_CXX_DEFINE_COMPARE_OPERATORS(rocprofiler_agent_id_t)
ROCPROFILER_CXX_DEFINE_COMPARE_OPERATORS(rocprofiler_queue_id_t)
ROCPROFILER_CXX_DEFINE_COMPARE_OPERATORS(rocprofiler_stream_id_t)
ROCPROFILER_CXX_DEFINE_COMPARE_OPERATORS(rocprofiler_buffer_id_t)
ROCPROFILER_CXX_DEFINE_COMPARE_OPERATORS(rocprofiler_counter_id_t)
ROCPROFILER_CXX_DEFINE_COMPARE_OPERATORS(rocprofiler_counter_config_id_t)
ROCPROFILER_CXX_DEFINE_COMPARE_OPERATORS(rocprofiler_callback_thread_t)
ROCPROFILER_CXX_DEFINE_COMPARE_OPERATORS(hsa_agent_t)
ROCPROFILER_CXX_DEFINE_COMPARE_OPERATORS(hsa_signal_t)
ROCPROFILER_CXX_DEFINE_COMPARE_OPERATORS(hsa_executable_t)
ROCPROFILER_CXX_DEFINE_COMPARE_OPERATORS(const rocprofiler_agent_v0_t&)
ROCPROFILER_CXX_DEFINE_COMPARE_OPERATORS(rocprofiler_dim3_t)
ROCPROFILER_CXX_DEFINE_COMPARE_OPERATORS(hsa_region_t)
ROCPROFILER_CXX_DEFINE_COMPARE_OPERATORS(hsa_amd_memory_pool_t)
ROCPROFILER_CXX_DEFINE_COMPARE_OPERATORS(rocprofiler_version_triplet_t)
ROCPROFILER_CXX_DEFINE_COMPARE_OPERATORS(rocprofiler_thread_trace_decoder_id_t)

// cleanup defines
#undef ROCPROFILER_CXX_DECLARE_OPERATORS
#undef ROCPROFILER_CXX_DEFINE_NE_OPERATOR
#undef ROCPROFILER_CXX_DEFINE_EQ_HANDLE_OPERATOR
#undef ROCPROFILER_CXX_DEFINE_LT_HANDLE_OPERATOR
#undef ROCPROFILER_CXX_DEFINE_COMPARE_OPERATORS
