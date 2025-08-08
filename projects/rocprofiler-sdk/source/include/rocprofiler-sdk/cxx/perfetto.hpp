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

#pragma once

#include <rocprofiler-sdk/cxx/details/mpl.hpp>

#include <cstddef>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#define ROCPROFILER_DEFINE_PERFETTO_CATEGORY(NAME, DESC, ...)                                      \
    namespace rocprofiler                                                                          \
    {                                                                                              \
    namespace sdk                                                                                  \
    {                                                                                              \
    template <>                                                                                    \
    struct perfetto_category<__VA_ARGS__>                                                          \
    {                                                                                              \
        static constexpr auto name        = NAME;                                                  \
        static constexpr auto description = DESC;                                                  \
    };                                                                                             \
    }                                                                                              \
    }

#define ROCPROFILER_DEFINE_CATEGORY(NS, VALUE, DESC)                                               \
    namespace rocprofiler                                                                          \
    {                                                                                              \
    namespace sdk                                                                                  \
    {                                                                                              \
    namespace NS                                                                                   \
    {                                                                                              \
    struct VALUE                                                                                   \
    {};                                                                                            \
    }                                                                                              \
    }                                                                                              \
    }                                                                                              \
    ROCPROFILER_DEFINE_PERFETTO_CATEGORY(#VALUE, DESC, NS::VALUE)

#define ROCPROFILER_PERFETTO_CATEGORY(TYPE)                                                        \
    ::perfetto::Category(::rocprofiler::sdk::perfetto_category<::rocprofiler::sdk::TYPE>::name)    \
        .SetDescription(                                                                           \
            ::rocprofiler::sdk::perfetto_category<::rocprofiler::sdk::TYPE>::description)

#define ROCPROFILER_PERFETTO_TRACING_CATEGORY(KIND, PREFIX, CODE, CATEGORY)                        \
    template <>                                                                                    \
    struct rocprofiler_tracing_perfetto_category<KIND, PREFIX##CODE>                               \
    {                                                                                              \
        static constexpr auto name =                                                               \
            ::rocprofiler::sdk::perfetto_category<::rocprofiler::sdk::category::CATEGORY>::name;   \
    };

#define ROCPROFILER_PERFETTO_BUFFER_TRACING_CATEGORY(...)                                          \
    ROCPROFILER_PERFETTO_TRACING_CATEGORY(                                                         \
        rocprofiler_buffer_tracing_kind_t, ROCPROFILER_BUFFER_TRACING_, __VA_ARGS__)

#define ROCPROFILER_PERFETTO_CALLBACK_TRACING_CATEGORY(...)                                        \
    ROCPROFILER_PERFETTO_TRACING_CATEGORY(                                                         \
        rocprofiler_callback_tracing_kind_t, ROCPROFILER_CALLBACK_TRACING_, __VA_ARGS__)

namespace rocprofiler
{
namespace sdk
{
template <typename Tp>
struct perfetto_category;
}  // namespace sdk
}  // namespace rocprofiler

ROCPROFILER_DEFINE_CATEGORY(category, hsa_api, "HSA API function")
ROCPROFILER_DEFINE_CATEGORY(category, hip_api, "HIP API function")
ROCPROFILER_DEFINE_CATEGORY(category, marker_api, "Marker API region")
ROCPROFILER_DEFINE_CATEGORY(category, rccl_api, "RCCL API function")
ROCPROFILER_DEFINE_CATEGORY(category, openmp, "OpenMP")
ROCPROFILER_DEFINE_CATEGORY(category, kernel_dispatch, "GPU kernel dispatch")
ROCPROFILER_DEFINE_CATEGORY(category, memory_copy, "Async memory copy")
ROCPROFILER_DEFINE_CATEGORY(category, memory_allocation, "Memory Allocation")
ROCPROFILER_DEFINE_CATEGORY(category, rocdecode_api, "rocDecode API function")
ROCPROFILER_DEFINE_CATEGORY(category, rocjpeg_api, "rocJPEG API function")
ROCPROFILER_DEFINE_CATEGORY(category, counter_collection, "Counter Collection")
ROCPROFILER_DEFINE_CATEGORY(category, kfd_events, "KFD events collection")
ROCPROFILER_DEFINE_CATEGORY(category, none, "Unknown category")

#define ROCPROFILER_PERFETTO_CATEGORIES                                                            \
    ROCPROFILER_PERFETTO_CATEGORY(category::hsa_api),                                              \
        ROCPROFILER_PERFETTO_CATEGORY(category::hip_api),                                          \
        ROCPROFILER_PERFETTO_CATEGORY(category::marker_api),                                       \
        ROCPROFILER_PERFETTO_CATEGORY(category::rccl_api),                                         \
        ROCPROFILER_PERFETTO_CATEGORY(category::openmp),                                           \
        ROCPROFILER_PERFETTO_CATEGORY(category::kernel_dispatch),                                  \
        ROCPROFILER_PERFETTO_CATEGORY(category::memory_copy),                                      \
        ROCPROFILER_PERFETTO_CATEGORY(category::counter_collection),                               \
        ROCPROFILER_PERFETTO_CATEGORY(category::memory_allocation),                                \
        ROCPROFILER_PERFETTO_CATEGORY(category::rocdecode_api),                                    \
        ROCPROFILER_PERFETTO_CATEGORY(category::rocjpeg_api),                                      \
        ROCPROFILER_PERFETTO_CATEGORY(category::none)

#include <perfetto.h>

PERFETTO_DEFINE_CATEGORIES(ROCPROFILER_PERFETTO_CATEGORIES);

namespace rocprofiler
{
namespace sdk
{
using perfetto_event_context_t = ::perfetto::EventContext;

template <typename Np, typename Tp>
auto
add_perfetto_annotation(perfetto_event_context_t& ctx, Np&& _name, Tp&& _val)
{
    namespace mpl = ::rocprofiler::sdk::mpl;

    using named_type = mpl::unqualified_identity_t<Np>;
    using value_type = mpl::unqualified_identity_t<Tp>;

    static_assert(mpl::is_string_type<named_type>::value, "Error! name is not a string type");

    auto _get_dbg = [&]() {
        auto* _dbg = ctx.event()->add_debug_annotations();
        _dbg->set_name(std::string_view{std::forward<Np>(_name)}.data());
        return _dbg;
    };

    if constexpr(std::is_same<value_type, std::string_view>::value)
    {
        _get_dbg()->set_string_value(_val.data());
    }
    else if constexpr(mpl::is_string_type<value_type>::value)
    {
        _get_dbg()->set_string_value(std::forward<Tp>(_val));
    }
    else if constexpr(std::is_same<value_type, bool>::value)
    {
        _get_dbg()->set_bool_value(_val);
    }
    else if constexpr(std::is_enum<value_type>::value)
    {
        _get_dbg()->set_int_value(static_cast<int64_t>(_val));
    }
    else if constexpr(std::is_floating_point<value_type>::value)
    {
        _get_dbg()->set_double_value(static_cast<double>(_val));
    }
    else if constexpr(std::is_integral<value_type>::value)
    {
        if constexpr(std::is_unsigned<value_type>::value)
        {
            _get_dbg()->set_uint_value(_val);
        }
        else
        {
            _get_dbg()->set_int_value(_val);
        }
    }
    else if constexpr(std::is_pointer<value_type>::value)
    {
        _get_dbg()->set_pointer_value(reinterpret_cast<uint64_t>(_val));
    }
    else if constexpr(mpl::can_stringify<value_type>::value)
    {
        auto _ss = std::stringstream{};
        _ss << std::forward<Tp>(_val);
        _get_dbg()->set_string_value(_ss.str());
    }
    else
    {
        static_assert(std::is_empty<value_type>::value, "Error! unsupported data type");
    }
}
}  // namespace sdk
}  // namespace rocprofiler

#include <rocprofiler-sdk/fwd.h>

namespace rocprofiler
{
namespace sdk
{
template <typename KindT, size_t Idx>
struct rocprofiler_tracing_perfetto_category;

ROCPROFILER_PERFETTO_BUFFER_TRACING_CATEGORY(NONE, none)
ROCPROFILER_PERFETTO_BUFFER_TRACING_CATEGORY(HSA_CORE_API, hsa_api)
ROCPROFILER_PERFETTO_BUFFER_TRACING_CATEGORY(HSA_AMD_EXT_API, hsa_api)
ROCPROFILER_PERFETTO_BUFFER_TRACING_CATEGORY(HSA_IMAGE_EXT_API, hsa_api)
ROCPROFILER_PERFETTO_BUFFER_TRACING_CATEGORY(HSA_FINALIZE_EXT_API, hsa_api)
ROCPROFILER_PERFETTO_BUFFER_TRACING_CATEGORY(HIP_RUNTIME_API, hip_api)
ROCPROFILER_PERFETTO_BUFFER_TRACING_CATEGORY(HIP_COMPILER_API, hip_api)
ROCPROFILER_PERFETTO_BUFFER_TRACING_CATEGORY(MARKER_CORE_API, marker_api)
ROCPROFILER_PERFETTO_BUFFER_TRACING_CATEGORY(MARKER_CORE_RANGE_API, marker_api)
ROCPROFILER_PERFETTO_BUFFER_TRACING_CATEGORY(MARKER_CONTROL_API, marker_api)
ROCPROFILER_PERFETTO_BUFFER_TRACING_CATEGORY(MARKER_NAME_API, marker_api)
ROCPROFILER_PERFETTO_BUFFER_TRACING_CATEGORY(MEMORY_COPY, memory_copy)
ROCPROFILER_PERFETTO_BUFFER_TRACING_CATEGORY(MEMORY_ALLOCATION, memory_allocation)
ROCPROFILER_PERFETTO_BUFFER_TRACING_CATEGORY(KERNEL_DISPATCH, kernel_dispatch)
ROCPROFILER_PERFETTO_BUFFER_TRACING_CATEGORY(SCRATCH_MEMORY, memory_allocation)
ROCPROFILER_PERFETTO_BUFFER_TRACING_CATEGORY(CORRELATION_ID_RETIREMENT, none)
ROCPROFILER_PERFETTO_BUFFER_TRACING_CATEGORY(RCCL_API, rccl_api)
ROCPROFILER_PERFETTO_BUFFER_TRACING_CATEGORY(OMPT, openmp)
ROCPROFILER_PERFETTO_BUFFER_TRACING_CATEGORY(RUNTIME_INITIALIZATION, none)
ROCPROFILER_PERFETTO_BUFFER_TRACING_CATEGORY(ROCDECODE_API, rocdecode_api)
ROCPROFILER_PERFETTO_BUFFER_TRACING_CATEGORY(ROCJPEG_API, rocjpeg_api)
ROCPROFILER_PERFETTO_BUFFER_TRACING_CATEGORY(HIP_STREAM, hip_api)
ROCPROFILER_PERFETTO_BUFFER_TRACING_CATEGORY(HIP_RUNTIME_API_EXT, hip_api)
ROCPROFILER_PERFETTO_BUFFER_TRACING_CATEGORY(HIP_COMPILER_API_EXT, hip_api)
ROCPROFILER_PERFETTO_BUFFER_TRACING_CATEGORY(ROCDECODE_API_EXT, rocdecode_api)
ROCPROFILER_PERFETTO_BUFFER_TRACING_CATEGORY(KFD_EVENT_PAGE_MIGRATE, kfd_events)
ROCPROFILER_PERFETTO_BUFFER_TRACING_CATEGORY(KFD_EVENT_PAGE_FAULT, kfd_events)
ROCPROFILER_PERFETTO_BUFFER_TRACING_CATEGORY(KFD_EVENT_QUEUE, kfd_events)
ROCPROFILER_PERFETTO_BUFFER_TRACING_CATEGORY(KFD_EVENT_UNMAP_FROM_GPU, kfd_events)
ROCPROFILER_PERFETTO_BUFFER_TRACING_CATEGORY(KFD_EVENT_DROPPED_EVENTS, kfd_events)
ROCPROFILER_PERFETTO_BUFFER_TRACING_CATEGORY(KFD_PAGE_MIGRATE, kfd_events)
ROCPROFILER_PERFETTO_BUFFER_TRACING_CATEGORY(KFD_PAGE_FAULT, kfd_events)
ROCPROFILER_PERFETTO_BUFFER_TRACING_CATEGORY(KFD_QUEUE, kfd_events)

ROCPROFILER_PERFETTO_CALLBACK_TRACING_CATEGORY(NONE, none)
ROCPROFILER_PERFETTO_CALLBACK_TRACING_CATEGORY(HSA_CORE_API, hsa_api)
ROCPROFILER_PERFETTO_CALLBACK_TRACING_CATEGORY(HSA_AMD_EXT_API, hsa_api)
ROCPROFILER_PERFETTO_CALLBACK_TRACING_CATEGORY(HSA_IMAGE_EXT_API, hsa_api)
ROCPROFILER_PERFETTO_CALLBACK_TRACING_CATEGORY(HSA_FINALIZE_EXT_API, hsa_api)
ROCPROFILER_PERFETTO_CALLBACK_TRACING_CATEGORY(HIP_RUNTIME_API, hip_api)
ROCPROFILER_PERFETTO_CALLBACK_TRACING_CATEGORY(HIP_COMPILER_API, hip_api)
ROCPROFILER_PERFETTO_CALLBACK_TRACING_CATEGORY(MARKER_CORE_API, marker_api)
ROCPROFILER_PERFETTO_CALLBACK_TRACING_CATEGORY(MARKER_CORE_RANGE_API, marker_api)
ROCPROFILER_PERFETTO_CALLBACK_TRACING_CATEGORY(MARKER_CONTROL_API, marker_api)
ROCPROFILER_PERFETTO_CALLBACK_TRACING_CATEGORY(MARKER_NAME_API, marker_api)
ROCPROFILER_PERFETTO_CALLBACK_TRACING_CATEGORY(CODE_OBJECT, none)
ROCPROFILER_PERFETTO_CALLBACK_TRACING_CATEGORY(SCRATCH_MEMORY, memory_allocation)
ROCPROFILER_PERFETTO_CALLBACK_TRACING_CATEGORY(KERNEL_DISPATCH, kernel_dispatch)
ROCPROFILER_PERFETTO_CALLBACK_TRACING_CATEGORY(MEMORY_COPY, memory_copy)
ROCPROFILER_PERFETTO_CALLBACK_TRACING_CATEGORY(MEMORY_ALLOCATION, memory_allocation)
ROCPROFILER_PERFETTO_CALLBACK_TRACING_CATEGORY(RCCL_API, rccl_api)
ROCPROFILER_PERFETTO_CALLBACK_TRACING_CATEGORY(OMPT, openmp)
ROCPROFILER_PERFETTO_CALLBACK_TRACING_CATEGORY(RUNTIME_INITIALIZATION, none)
ROCPROFILER_PERFETTO_CALLBACK_TRACING_CATEGORY(ROCDECODE_API, rocdecode_api)
ROCPROFILER_PERFETTO_CALLBACK_TRACING_CATEGORY(ROCJPEG_API, rocjpeg_api)
ROCPROFILER_PERFETTO_CALLBACK_TRACING_CATEGORY(HIP_STREAM, hip_api)

template <typename KindT, size_t Idx, size_t... Tail>
const char*
get_perfetto_category(KindT kind, std::index_sequence<Idx, Tail...>)
{
    if(kind == Idx) return rocprofiler_tracing_perfetto_category<KindT, Idx>::name;
    // recursion until tail empty
    if constexpr(sizeof...(Tail) > 0)
        return get_perfetto_category(kind, std::index_sequence<Tail...>{});
    return nullptr;
}

template <typename KindT>
const char*
get_perfetto_category(KindT kind)
{
    if constexpr(std::is_same<KindT, rocprofiler_buffer_tracing_kind_t>::value)
    {
        return get_perfetto_category<KindT>(
            kind, std::make_index_sequence<ROCPROFILER_BUFFER_TRACING_LAST>{});
    }
    else if constexpr(std::is_same<KindT, rocprofiler_callback_tracing_kind_t>::value)
    {
        return get_perfetto_category<KindT>(
            kind, std::make_index_sequence<ROCPROFILER_CALLBACK_TRACING_LAST>{});
    }
    else
    {
        static_assert(rocprofiler::sdk::mpl::assert_false<KindT>::value, "Unsupported data type");
    }

    return nullptr;
}
}  // namespace sdk
}  // namespace rocprofiler

#undef ROCPROFILER_DEFINE_PERFETTO_CATEGORY
#undef ROCPROFILER_DEFINE_CATEGORY
#undef ROCPROFILER_PERFETTO_CATEGORY
#undef ROCPROFILER_PERFETTO_CATEGORIES
#undef ROCPROFILER_PERFETTO_TRACING_CATEGORY
#undef ROCPROFILER_PERFETTO_BUFFER_TRACING_CATEGORY
#undef ROCPROFILER_PERFETTO_CALLBACK_TRACING_CATEGORY
