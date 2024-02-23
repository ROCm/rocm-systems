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

#include <cstddef>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#define ROCPROFILER_DEFINE_NAME_TRAIT(NAME, DESC, ...)                                             \
    namespace rocprofiler                                                                          \
    {                                                                                              \
    template <>                                                                                    \
    struct perfetto_category<__VA_ARGS__>                                                          \
    {                                                                                              \
        static constexpr auto value       = NAME;                                                  \
        static constexpr auto description = DESC;                                                  \
    };                                                                                             \
    }

namespace rocprofiler
{
template <typename Tp>
struct perfetto_category;

namespace trait
{
template <typename... Tp>
using name = perfetto_category<Tp...>;
}
}  // namespace rocprofiler

#define ROCPROFILER_DEFINE_NS_API(NS, NAME)                                                        \
    namespace rocprofiler                                                                          \
    {                                                                                              \
    namespace NS                                                                                   \
    {                                                                                              \
    struct NAME;                                                                                   \
    }                                                                                              \
    }

#define ROCPROFILER_DEFINE_CATEGORY(NS, VALUE, DESC)                                               \
    ROCPROFILER_DEFINE_NS_API(NS, VALUE)                                                           \
    ROCPROFILER_DEFINE_NAME_TRAIT(#VALUE, DESC, NS::VALUE)

ROCPROFILER_DEFINE_CATEGORY(category, hsa_api, "HSA API function")
ROCPROFILER_DEFINE_CATEGORY(category, hip_api, "HIP API function")
ROCPROFILER_DEFINE_CATEGORY(category, marker_api, "Marker API region")
ROCPROFILER_DEFINE_CATEGORY(category, kernel_dispatch, "GPU kernel dispatch")
ROCPROFILER_DEFINE_CATEGORY(category, memory_copy, "Async memory copy")

#define ROCPROFILER_PERFETTO_CATEGORY(TYPE)                                                        \
    ::perfetto::Category(::rocprofiler::perfetto_category<::rocprofiler::TYPE>::value)             \
        .SetDescription(::rocprofiler::perfetto_category<::rocprofiler::TYPE>::description)

#define ROCPROFILER_PERFETTO_CATEGORIES                                                            \
    ROCPROFILER_PERFETTO_CATEGORY(category::hsa_api),                                              \
        ROCPROFILER_PERFETTO_CATEGORY(category::hip_api),                                          \
        ROCPROFILER_PERFETTO_CATEGORY(category::marker_api),                                       \
        ROCPROFILER_PERFETTO_CATEGORY(category::kernel_dispatch),                                  \
        ROCPROFILER_PERFETTO_CATEGORY(category::memory_copy)

#include <perfetto.h>

PERFETTO_DEFINE_CATEGORIES(ROCPROFILER_PERFETTO_CATEGORIES);

namespace concepts
{
template <typename Tp>
struct is_string_type : std::false_type
{};

template <>
struct is_string_type<std::string> : std::true_type
{};

template <>
struct is_string_type<char*> : std::true_type
{};

template <>
struct is_string_type<const char*> : std::true_type
{};

template <>
struct is_string_type<std::string_view> : std::true_type
{};

template <typename Tp>
struct is_string_type<const Tp> : is_string_type<std::decay_t<Tp>>
{};

template <typename Tp>
struct is_string_type<Tp&> : is_string_type<std::decay_t<Tp>>
{};

template <typename Tp>
struct is_string_type<volatile Tp> : is_string_type<std::decay_t<Tp>>
{};

template <typename Tp, size_t N>
struct is_string_type<Tp[N]> : is_string_type<std::decay_t<Tp[N]>>
{};

template <typename Tp, size_t N>
struct is_string_type<const Tp[N]> : is_string_type<std::decay_t<Tp[N]>>
{};

template <typename Tp, size_t N>
struct is_string_type<volatile Tp[N]> : is_string_type<std::decay_t<Tp[N]>>
{};

template <typename Tp>
struct unqualified_type
{
    using type = std::remove_reference_t<std::remove_cv_t<std::decay_t<Tp>>>;
};

template <typename Tp>
using unqualified_type_t = typename unqualified_type<Tp>::type;

template <typename Tp>
struct can_stringify
{
private:
    static constexpr auto sfinae(int)
        -> decltype(std::declval<std::ostream&>() << std::declval<Tp>(), bool())
    {
        return true;
    }

    static constexpr auto sfinae(long) { return false; }

public:
    static constexpr bool value = sfinae(0);
    constexpr auto        operator()() const { return sfinae(0); }
};
}  // namespace concepts

using perfetto_event_context_t = ::perfetto::EventContext;

template <typename Np, typename Tp>
auto
add_perfetto_annotation(perfetto_event_context_t& ctx, Np&& _name, Tp&& _val)
{
    using named_type = concepts::unqualified_type_t<Np>;
    using value_type = concepts::unqualified_type_t<Tp>;

    static_assert(concepts::is_string_type<named_type>::value, "Error! name is not a string type");

    auto _get_dbg = [&]() {
        auto* _dbg = ctx.event()->add_debug_annotations();
        _dbg->set_name(std::string_view{std::forward<Np>(_name)}.data());
        return _dbg;
    };

    if constexpr(std::is_same<value_type, std::string_view>::value)
    {
        _get_dbg()->set_string_value(_val.data());
    }
    else if constexpr(concepts::is_string_type<value_type>::value)
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
    else if constexpr(concepts::can_stringify<value_type>::value)
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
