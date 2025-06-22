// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "defines.hpp"

#include <timemory/mpl/concepts.hpp>
#include <timemory/utility/types.hpp>

#include <memory>
#include <optional>
#include <type_traits>

namespace rocprofsys
{
namespace concepts = ::tim::concepts;  // NOLINT

static constexpr size_t max_supported_threads = ROCPROFSYS_MAX_THREADS;

template <typename Tp>
struct thread_deleter;

// unique ptr type for rocprof-sys
template <typename Tp>
using unique_ptr_t = std::unique_ptr<Tp, thread_deleter<Tp>>;

using construct_on_init = std::true_type;

using tim::identity;    // NOLINT
using tim::identity_t;  // NOLINT

template <typename Tp>
struct use_placement_new_when_generating_unique_ptr : std::false_type
{};

template <typename Tp, typename... Args>
auto
make_unique(Args&&... args)
{
    return unique_ptr_t<Tp>{ new Tp{ std::forward<Args>(args)... } };
}
}  // namespace rocprofsys

namespace tim
{
namespace concepts
{
template <typename Tp>
struct is_unique_pointer : std::false_type
{};

template <typename Tp>
struct is_unique_pointer<::rocprofsys::unique_ptr_t<Tp>> : std::true_type
{};

template <typename Tp>
struct is_unique_pointer<std::unique_ptr<Tp>> : std::true_type
{};

template <typename Tp>
struct is_optional : std::false_type
{};

template <typename Tp>
struct is_optional<std::optional<Tp>> : std::true_type
{};

template <typename Tp>
struct can_stringify
{
private:
    static constexpr auto sfinae(int) -> decltype(std::declval<std::ostream&>()
                                                      << std::declval<Tp>(),
                                                  bool())
    {
        return true;
    }

    static constexpr auto sfinae(long) { return false; }

public:
    static constexpr bool value = sfinae(0);
    constexpr auto        operator()() const { return sfinae(0); }
};

template <size_t N, typename Tp, bool>
struct tuple_element_impl;

template <size_t N, typename... Tp>
struct tuple_element_impl<N, std::tuple<Tp...>, true>
{
    using type = typename std::tuple_element<N, std::tuple<Tp...>>::type;
};

template <size_t N, typename... Tp>
struct tuple_element_impl<N, std::tuple<Tp...>, false>
{
    using type = void;
};

template <size_t N, typename Tp>
struct tuple_element;

template <size_t N, typename... Tp>
struct tuple_element<N, std::tuple<Tp...>>
{
    using type =
        typename tuple_element_impl<N, std::tuple<Tp...>, (N < sizeof...(Tp))>::type;
};

template <size_t N, typename Tp>
using tuple_element_t = typename tuple_element<N, Tp>::type;
}  // namespace concepts
}  // namespace tim
