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
ROCPROFILER_DEFINE_CATEGORY(category, kernel_dispatch, "GPU kernel dispatch")
ROCPROFILER_DEFINE_CATEGORY(category, memory_copy, "Async memory copy")

#define ROCPROFILER_PERFETTO_CATEGORIES                                                            \
    ROCPROFILER_PERFETTO_CATEGORY(category::hsa_api),                                              \
        ROCPROFILER_PERFETTO_CATEGORY(category::hip_api),                                          \
        ROCPROFILER_PERFETTO_CATEGORY(category::marker_api),                                       \
        ROCPROFILER_PERFETTO_CATEGORY(category::rccl_api),                                         \
        ROCPROFILER_PERFETTO_CATEGORY(category::kernel_dispatch),                                  \
        ROCPROFILER_PERFETTO_CATEGORY(category::memory_copy)

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

#undef ROCPROFILER_DEFINE_PERFETTO_CATEGORY
#undef ROCPROFILER_DEFINE_CATEGORY
#undef ROCPROFILER_PERFETTO_CATEGORY
#undef ROCPROFILER_PERFETTO_CATEGORIES
