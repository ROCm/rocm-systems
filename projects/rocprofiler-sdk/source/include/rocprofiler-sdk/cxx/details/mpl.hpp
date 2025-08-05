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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#define ROCPROFILER_IMPL_HAS_CONCEPT(NAME, TRAIT)                                                  \
    template <typename Tp, typename = typename Tp::TRAIT>                                          \
    inline constexpr bool NAME(int)                                                                \
    {                                                                                              \
        return true;                                                                               \
    }                                                                                              \
                                                                                                   \
    template <typename Tp>                                                                         \
    inline constexpr bool NAME(long)                                                               \
    {                                                                                              \
        return false;                                                                              \
    }

#define ROCPROFILER_IMPL_SFINAE_CONCEPT(NAME, ...)                                                 \
    template <typename Tp>                                                                         \
    struct NAME                                                                                    \
    {                                                                                              \
    private:                                                                                       \
        static constexpr auto sfinae(int) -> decltype(__VA_ARGS__, bool()) { return true; }        \
                                                                                                   \
        static constexpr auto sfinae(long) { return false; }                                       \
                                                                                                   \
    public:                                                                                        \
        static constexpr bool value = sfinae(0);                                                   \
        constexpr auto        operator()() const { return sfinae(0); }                             \
    };

namespace rocprofiler
{
namespace sdk
{
namespace mpl
{
template <typename Tp>
struct unqualified_identity
{
    using type = std::remove_cv_t<std::remove_reference_t<std::decay_t<Tp>>>;
};

template <typename Tp>
using unqualified_identity_t = typename unqualified_identity<Tp>::type;

template <typename Tp, typename Up>
struct is_same_unqualified_identity
: std::is_same<unqualified_identity_t<Tp>, unqualified_identity_t<Up>>
{};

template <typename Tp>
struct string_support
{
    using type        = Tp;
    using return_type = void;

    static constexpr auto value = false;
    static constexpr void default_value() {}
};

template <>
struct string_support<const char*>
{
    using type        = const char*;
    using return_type = type;

    static constexpr auto value = true;
    static constexpr type default_value() { return nullptr; }

    type operator()(const char* val) const { return val; }
};

template <>
struct string_support<std::string_view>
{
    using type        = std::string_view;
    using return_type = type;

    static constexpr auto value = true;
    static constexpr type default_value() { return type{}; }

    type operator()(const char* val) const { return type{val}; }
};

template <>
struct string_support<std::string>
{
    using type        = std::string;
    using return_type = type;

    static constexpr auto value = true;
    static type           default_value() { return type{}; }

    type operator()(const char* val) const { return type{val}; }
};

namespace impl
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
}  // namespace impl

template <typename Tp>
struct is_string_type : impl::is_string_type<unqualified_identity_t<Tp>>
{};

// template <typename Tp>
// struct can_stringify
// {
// private:
//     static constexpr auto sfinae(int)
//         -> decltype(std::declval<std::ostream&>() << std::declval<Tp>(), bool())
//     {
//         return true;
//     }

//     static constexpr auto sfinae(long) { return false; }

// public:
//     static constexpr bool value = sfinae(0);
//     constexpr auto        operator()() const { return sfinae(0); }
// };

ROCPROFILER_IMPL_HAS_CONCEPT(has_traits, traits_type)
ROCPROFILER_IMPL_HAS_CONCEPT(has_value_type, value_type)
ROCPROFILER_IMPL_HAS_CONCEPT(has_key_type, key_type)
ROCPROFILER_IMPL_HAS_CONCEPT(has_mapped_type, mapped_type)

ROCPROFILER_IMPL_SFINAE_CONCEPT(has_empty_member_function, std::declval<Tp>().empty())
ROCPROFILER_IMPL_SFINAE_CONCEPT(can_stringify, std::declval<std::ostream&>() << std::declval<Tp>())
ROCPROFILER_IMPL_SFINAE_CONCEPT(is_iterable,
                                std::begin(std::declval<Tp>()),
                                std::end(std::declval<Tp>()))

// compatability
template <typename Tp>
using supports_ostream = can_stringify<Tp>;

template <typename ArgT>
inline bool
is_empty(ArgT&& _v)
{
    using arg_type = unqualified_identity_t<ArgT>;

    if constexpr(has_empty_member_function<arg_type>::value)
    {
        return std::forward<ArgT>(_v).empty();
    }
    else if constexpr(is_string_type<arg_type>::value)
    {
        static_assert(std::is_constructible<std::string_view, ArgT>::value,
                      "not string_view constructible");
        return std::string_view{std::forward<ArgT>(_v)}.empty();
    }

    return false;
}

namespace impl
{
template <typename ContainerT, typename... Args>
inline auto
emplace(ContainerT& _c, int, Args&&... _args)
    -> decltype(_c.emplace_back(std::forward<Args>(_args)...))
{
    return _c.emplace_back(std::forward<Args>(_args)...);
}

template <typename ContainerT, typename... Args>
inline auto
emplace(ContainerT& _c, long, Args&&... _args) -> decltype(_c.emplace(std::forward<Args>(_args)...))
{
    return _c.emplace(std::forward<Args>(_args)...);
}

template <typename ContainerT, typename ArgT>
inline auto
reserve(ContainerT& _c, int, ArgT _arg) -> decltype(_c.reserve(_arg), bool())
{
    _c.reserve(_arg);
    return true;
}

template <typename ContainerT, typename ArgT>
inline auto
reserve(ContainerT&, long, ArgT)
{
    return false;
}
}  // namespace impl

template <typename ContainerT, typename... Args>
inline auto
emplace(ContainerT& _c, Args&&... _args)
{
    return impl::emplace(_c, 0, std::forward<Args>(_args)...);
}

template <typename ContainerT, typename ArgT>
inline auto
reserve(ContainerT& _c, ArgT _arg)
{
    return impl::reserve(_c, 0, _arg);
}

template <typename Tp>
struct assert_false
{
    static constexpr auto value = false;
};
}  // namespace mpl
}  // namespace sdk
}  // namespace rocprofiler

#undef ROCPROFILER_IMPL_HAS_CONCEPT
#undef ROCPROFILER_IMPL_SFINAE_CONCEPT
