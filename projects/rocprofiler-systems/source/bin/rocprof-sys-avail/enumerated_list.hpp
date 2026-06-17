// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

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
