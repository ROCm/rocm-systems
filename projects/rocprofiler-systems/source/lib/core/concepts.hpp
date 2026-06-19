// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/defines.h"

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
concept string_like = requires(std::ostream& _os, const Tp& _v) { _os << _v; };

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
