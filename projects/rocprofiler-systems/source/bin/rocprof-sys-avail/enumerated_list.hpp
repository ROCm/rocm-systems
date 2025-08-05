// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "common.hpp"
#include "defines.hpp"

#include <timemory/components/properties.hpp>
#include <timemory/mpl/concepts.hpp>
#include <timemory/utility/type_list.hpp>

#include <cstddef>
#include <tuple>
#include <type_traits>

template <typename T, typename I>
struct enumerated_list;

template <template <typename...> class TupT, typename... T>
struct enumerated_list<TupT<T...>, std::index_sequence<>>
{
    using type = type_list<T...>;
};

template <template <typename...> class TupT, size_t I, typename... T, size_t... Idx>
struct enumerated_list<TupT<T...>, std::index_sequence<I, Idx...>>
{
    using Tp                         = tim::component::enumerator_t<I>;
    static constexpr bool is_nothing = tim::concepts::is_placeholder<Tp>::value;
    using type                       = typename enumerated_list<
                              std::conditional_t<is_nothing, type_list<T...>, type_list<T..., Tp>>,
                              std::index_sequence<Idx...>>::type;
};
