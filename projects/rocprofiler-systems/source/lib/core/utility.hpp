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

#include "concepts.hpp"

#include <timemory/mpl/concepts.hpp>
#include <timemory/utility/delimit.hpp>
#include <timemory/utility/join.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace rocprofsys
{
namespace utility
{
/// provides an alternative thread index for when using threading::get_id() is not
/// desirable
inline auto
get_thread_index()
{
    static std::atomic<int64_t> _c{ 0 };
    static thread_local int64_t _v = _c++;
    return _v;
}

/// fills any array with the result of the functor
template <size_t N, typename FuncT>
inline auto
get_filled_array(FuncT&& _func)
{
    using Tp = std::decay_t<decltype(_func())>;
    std::array<Tp, N> _v{};
    for(auto& itr : _v)
        itr = std::move(_func());
    return _v;
}

/// returns a vector with a preallocated buffer
template <typename... Tp>
inline auto
get_reserved_vector(size_t _n)
{
    std::vector<Tp...> _v{};
    _v.reserve(_n);
    return _v;
}

/// returns a vector with a preallocated buffer
template <typename... Tp>
inline decltype(auto)
get_reserved_vector(std::vector<Tp...>&& _v, size_t _n)
{
    _v.reserve(_n);
    return std::forward<std::vector<Tp...>>(_v);
}

template <typename Tp, size_t Offset>
struct offset_index_sequence;

template <size_t Idx, size_t Offset>
struct offset_index_value
{
    static constexpr size_t value = Idx + Offset;
};

template <size_t Offset, size_t... Idx>
struct offset_index_sequence<std::index_sequence<Idx...>, Offset>
{
    using type = std::integer_sequence<size_t, offset_index_value<Idx, Offset>::value...>;
};

template <size_t N, size_t OffsetN>
using make_offset_index_sequence =
    offset_index_sequence<std::make_index_sequence<N>, OffsetN>;

template <size_t StartN, size_t EndN>
using make_index_sequence_range =
    typename offset_index_sequence<std::make_index_sequence<(EndN - StartN)>,
                                   StartN>::type;

template <typename Tp>
struct generate
{
    using type = Tp;

    template <typename... Args>
    auto operator()(Args&&... _args) const
    {
        if constexpr(concepts::is_unique_pointer<Tp>::value)
        {
            using value_type = typename type::element_type;

            if constexpr(use_placement_new_when_generating_unique_ptr<value_type>::value)
            {
                // create a thread-local buffer for placement-new
                static thread_local auto _buffer = std::array<char, sizeof(value_type)>{};
                if constexpr(std::is_constructible<value_type, Args...>::value)
                {
                    return type{ new(_buffer.data())
                                     value_type{ std::forward<Args>(_args)... } };
                }
                else
                {
                    return type{ new(_buffer.data())
                                     value_type{ invoke(std::forward<Args>(_args))... } };
                }
            }
            else
            {
                if constexpr(std::is_constructible<value_type, Args...>::value)
                {
                    return type{ new value_type{ std::forward<Args>(_args)... } };
                }
                else
                {
                    return type{ new value_type{ invoke(std::forward<Args>(_args))... } };
                }
            }
        }
        else
        {
            if constexpr(std::is_constructible<type, Args...>::value)
            {
                return type{ std::forward<Args>(_args)... };
            }
            else
            {
                return type{ invoke(std::forward<Args>(_args))... };
            }
        }
    }

private:
    template <typename Up>
    static auto invoke(Up&& _v, int,
                       std::enable_if_t<std::is_invocable<Up>::value, int> = 0)
        -> decltype(std::forward<Up>(_v)())
    {
        return std::forward<Up>(_v)();
    }

    template <typename Up>
    static auto&& invoke(Up&& _v, long)
    {
        return std::forward<Up>(_v);
    }

    template <typename Up>
    static decltype(auto) invoke(Up&& _v)
    {
        return invoke(std::forward<Up>(_v), 0);
    }
};

template <template <typename, typename...> class ContainerT, typename DataT,
          typename... TailT, typename PredicateT = bool (*)(const DataT&)>
inline ContainerT<DataT, TailT...>&
filter_sort_unique(
    ContainerT<DataT, TailT...>& _v,
    PredicateT&&                 _predicate = [](const auto& itr) { return !itr; })
{
    _v.erase(std::remove_if(_v.begin(), _v.end(), std::forward<PredicateT>(_predicate)),
             _v.end());
    std::sort(_v.begin(), _v.end());

    auto _last = std::unique(_v.begin(), _v.end());
    if(std::distance(_v.begin(), _last) > 0) _v.erase(_last, _v.end());
    return _v;
}

template <typename LhsT, typename RhsT>
inline LhsT&
combine(LhsT& _lhs, RhsT&& _rhs)
{
    for(auto&& itr : _rhs)
        _lhs.emplace_back(itr);
    return _lhs;
}

template <template <typename, typename...> class ContainerT, typename Tp,
          typename... TailT>
std::string
get_regex_or(const ContainerT<Tp, TailT...>& _container, const std::string& _fallback)
{
    static_assert(tim::concepts::is_string_type<Tp>::value,
                  "get_regex_or requires a container of string types");

    if(_container.empty()) return _fallback;

    namespace join = timemory::join;
    return join::join(join::array_config{ "|", "(", ")" }, _container);
}

template <template <typename, typename...> class ContainerT, typename Tp,
          typename... TailT, typename PredicateT>
std::string
get_regex_or(const ContainerT<Tp, TailT...>& _container, PredicateT&& _predicate,
             const std::string& _fallback)
{
    static_assert(tim::concepts::is_string_type<Tp>::value,
                  "get_regex_or requires a container of string types");

    if(_container.empty()) return _fallback;

    auto _dest = std::vector<std::string>{};
    _dest.reserve(_container.size());
    for(const auto& itr : _container)
        _dest.emplace_back(_predicate(itr));

    return get_regex_or(_dest, _fallback);
}

template <typename Tp>
Tp
convert(std::string_view _inp)
{
    auto _iss = std::stringstream{};
    auto _ret = Tp{};
    _iss << _inp;
    _iss >> _ret;
    return _ret;
}

template <typename Tp = int64_t, typename ContainerT = std::set<Tp>, typename Up = Tp>
ContainerT
parse_numeric_range(std::string _input_string, const std::string& _label, Up _incr);

extern template std::set<int64_t>
parse_numeric_range<int64_t, std::set<int64_t>>(std::string, const std::string&, long);
extern template std::vector<int64_t>
parse_numeric_range<int64_t, std::vector<int64_t>>(std::string, const std::string&, long);
extern template std::unordered_set<int64_t>
parse_numeric_range<int64_t, std::unordered_set<int64_t>>(std::string, const std::string&,
                                                          long);
}  // namespace utility
}  // namespace rocprofsys
