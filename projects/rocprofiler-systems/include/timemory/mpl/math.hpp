// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
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

/** \file math.hpp
 * \headerfile math.hpp "timemory/mpl/math.hpp"
 * Provides the template meta-programming expansions for math operations
 *
 */

#pragma once

#include "timemory/mpl/concepts.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/utility/types.hpp"

#include <cassert>
#include <cmath>
#include <limits>
#include <utility>

//======================================================================================//

namespace tim
{
namespace math
{
//--------------------------------------------------------------------------------------//

template <typename Tp>
bool
is_finite(const Tp& val)
{
#if defined(TIMEMORY_WINDOWS)
    const Tp _infv = std::numeric_limits<Tp>::infinity();
    const Tp _inf  = (val < 0.0) ? -_infv : _infv;
    return (val == val && val != _inf);
#else
    return std::isfinite(val);
#endif
}

//--------------------------------------------------------------------------------------//

template <typename Tp>
TIMEMORY_INLINE Tp abs(Tp);

template <typename Tp>
TIMEMORY_INLINE Tp sqrt(Tp);

template <typename Tp>
TIMEMORY_INLINE Tp
                pow(Tp, double);

template <typename Tp>
TIMEMORY_INLINE Tp sqr(Tp);

template <typename Tp>
TIMEMORY_INLINE Tp
                min(const Tp&, const Tp&);

template <typename Tp>
TIMEMORY_INLINE Tp
                max(const Tp&, const Tp&);

template <typename Tp, typename Up = Tp>
TIMEMORY_INLINE void
assign(Tp&, Up&&);

template <typename Tp, typename Up = Tp,
          enable_if_t<!concepts::is_null_type<Tp>::value> = 0>
TIMEMORY_INLINE Tp&
                plus(Tp&, const Up&);

template <typename Tp, typename Up = Tp,
          enable_if_t<!concepts::is_null_type<Tp>::value> = 0>
TIMEMORY_INLINE Tp&
                minus(Tp&, const Up&);

template <typename Tp, typename Up = Tp,
          enable_if_t<!concepts::is_null_type<Tp>::value> = 0>
TIMEMORY_INLINE Tp&
                multiply(Tp&, const Up&);

template <typename Tp, typename Up = Tp,
          enable_if_t<!concepts::is_null_type<Tp>::value> = 0>
TIMEMORY_INLINE Tp&
                divide(Tp&, const Up&);

template <typename Tp>
TIMEMORY_INLINE Tp
                percent_diff(const Tp&, const Tp&);

//--------------------------------------------------------------------------------------//
//              dummy overloads for std::tuple<>, type_list<>, null_type
//
#define TIMEMORY_MATH_NULL_TYPE_OVERLOAD(TYPE)                                           \
    TIMEMORY_INLINE TYPE abs(TYPE) { return TYPE{}; }                                    \
    TIMEMORY_INLINE TYPE sqrt(TYPE) { return TYPE{}; }                                   \
    TIMEMORY_INLINE TYPE pow(TYPE, double) { return TYPE{}; }                            \
    TIMEMORY_INLINE TYPE sqr(TYPE) { return TYPE{}; }                                    \
    TIMEMORY_INLINE TYPE min(const TYPE&, const TYPE&) { return TYPE{}; }                \
    TIMEMORY_INLINE TYPE max(const TYPE&, const TYPE&) { return TYPE{}; }                \
    TIMEMORY_INLINE void assign(TYPE&, TYPE&&) {}                                        \
    TIMEMORY_INLINE TYPE& plus(TYPE& lhs, const TYPE&) { return lhs; }                   \
    TIMEMORY_INLINE TYPE& minus(TYPE& lhs, const TYPE&) { return lhs; }                  \
    TIMEMORY_INLINE TYPE& multiply(TYPE& lhs, const TYPE&) { return lhs; }               \
    TIMEMORY_INLINE TYPE& divide(TYPE& lhs, const TYPE&) { return lhs; }                 \
    TIMEMORY_INLINE TYPE percent_diff(const TYPE&, const TYPE&) { return TYPE{}; }       \
    template <typename Up>                                                               \
    TIMEMORY_INLINE TYPE& plus(TYPE& lhs, Up&&)                                          \
    {                                                                                    \
        return lhs;                                                                      \
    }                                                                                    \
    template <typename Up>                                                               \
    TIMEMORY_INLINE TYPE& minus(TYPE& lhs, Up&&)                                         \
    {                                                                                    \
        return lhs;                                                                      \
    }                                                                                    \
    template <typename Up>                                                               \
    TIMEMORY_INLINE TYPE& multiply(TYPE& lhs, Up&&)                                      \
    {                                                                                    \
        return lhs;                                                                      \
    }                                                                                    \
    template <typename Up>                                                               \
    TIMEMORY_INLINE TYPE& divide(TYPE& lhs, Up&&)                                        \
    {                                                                                    \
        return lhs;                                                                      \
    }                                                                                    \
    TIMEMORY_INLINE TYPE& divide(TYPE& lhs, uint64_t) { return lhs; }                    \
    TIMEMORY_INLINE TYPE& divide(TYPE& lhs, int64_t) { return lhs; }

TIMEMORY_MATH_NULL_TYPE_OVERLOAD(std::tuple<>)
TIMEMORY_MATH_NULL_TYPE_OVERLOAD(null_type)
TIMEMORY_MATH_NULL_TYPE_OVERLOAD(type_list<>)

//--------------------------------------------------------------------------------------//

template <typename Tp, enable_if_t<std::is_arithmetic<Tp>::value> = 0,
          enable_if_t<std::is_integral<Tp>::value && std::is_unsigned<Tp>::value> = 0>
TIMEMORY_INLINE auto
abs(Tp _val, type_list<>) -> decltype(Tp{})
{
    return _val;
}

template <typename Tp, enable_if_t<std::is_arithmetic<Tp>::value> = 0,
          enable_if_t<!(std::is_integral<Tp>::value && std::is_unsigned<Tp>::value)> = 0>
auto TIMEMORY_INLINE
     abs(Tp _val, type_list<>) -> decltype(std::abs(_val), Tp{})
{
    return std::abs(_val);
}

template <typename Tp, typename Vp = typename Tp::value_type>
auto
abs(Tp _val, type_list<>, ...) -> decltype(std::begin(_val), Tp{})
{
    for(auto& itr : _val)
        itr = ::tim::math::abs(itr, get_index_sequence<decay_t<decltype(itr)>>::value);
    return _val;
}

template <typename Tp, typename Kp = typename Tp::key_type,
          typename Mp = typename Tp::mapped_type>
auto
abs(Tp _val, type_list<>) -> decltype(std::begin(_val), Tp{})
{
    for(auto& itr : _val)
    {
        itr.second = ::tim::math::abs(
            itr.second, get_index_sequence<decay_t<decltype(itr.second)>>::value);
    }
    return _val;
}

template <template <typename...> class Tuple, typename... Types, size_t... Idx>
auto
abs(Tuple<Types...> _val, index_sequence<Idx...>)
    -> decltype(std::get<0>(_val), Tuple<Types...>())
{
    TIMEMORY_FOLD_EXPRESSION(std::get<Idx>(_val) = ::tim::math::abs(std::get<Idx>(_val)));
    return _val;
}

template <typename Tp>
Tp
abs(Tp _val)
{
    return ::tim::math::abs(_val, get_index_sequence<Tp>::value);
}

//--------------------------------------------------------------------------------------//

template <typename Tp, enable_if_t<std::is_arithmetic<Tp>::value> = 0>
TIMEMORY_INLINE auto
sqrt(Tp _val, type_list<>) -> decltype(std::sqrt(_val), Tp{})
{
    return std::sqrt(_val);
}

template <typename Tp, typename Vp = typename Tp::value_type>
auto
sqrt(Tp _val, type_list<>, ...) -> decltype(std::begin(_val), Tp{})
{
    for(auto& itr : _val)
        itr = ::tim::math::sqrt(itr, get_index_sequence<decay_t<decltype(itr)>>::value);
    return _val;
}

template <typename Tp, typename Kp = typename Tp::key_type,
          typename Mp = typename Tp::mapped_type>
auto
sqrt(Tp _val, type_list<>) -> decltype(std::begin(_val), Tp{})
{
    for(auto& itr : _val)
    {
        itr.second = ::tim::math::sqrt(
            itr.second, get_index_sequence<decay_t<decltype(itr.second)>>::value);
    }
    return _val;
}

template <template <typename...> class Tuple, typename... Types, size_t... Idx>
auto
sqrt(Tuple<Types...> _val, index_sequence<Idx...>)
    -> decltype(std::get<0>(_val), Tuple<Types...>())
{
    TIMEMORY_FOLD_EXPRESSION(std::get<Idx>(_val) =
                                 ::tim::math::sqrt(std::get<Idx>(_val)));
    return _val;
}

template <typename Tp>
Tp
sqrt(Tp _val)
{
    return ::tim::math::sqrt(_val, get_index_sequence<Tp>::value);
}

//--------------------------------------------------------------------------------------//

template <typename Tp, enable_if_t<std::is_arithmetic<Tp>::value> = 0>
TIMEMORY_INLINE auto
pow(Tp _val, double _m, type_list<>) -> decltype(std::pow(_val, _m), Tp{})
{
    return std::pow(_val, _m);
}

template <typename Tp, typename Vp = typename Tp::value_type>
auto
pow(Tp _val, double _m, type_list<>, ...) -> decltype(std::begin(_val), Tp{})
{
    for(auto& itr : _val)
    {
        itr =
            ::tim::math::pow(itr, _m, get_index_sequence<decay_t<decltype(itr)>>::value);
    }
    return _val;
}

template <typename Tp, typename Kp = typename Tp::key_type,
          typename Mp = typename Tp::mapped_type>
auto
pow(Tp _val, double _m, type_list<>) -> decltype(std::begin(_val), Tp{})
{
    for(auto& itr : _val)
    {
        itr.second = ::tim::math::pow(
            itr.second, _m, get_index_sequence<decay_t<decltype(itr.second)>>::value);
    }
    return _val;
}

template <template <typename...> class Tuple, typename... Types, size_t... Idx>
auto
pow(Tuple<Types...> _val, double _m, index_sequence<Idx...>)
    -> decltype(std::get<0>(_val), Tuple<Types...>())
{
    TIMEMORY_FOLD_EXPRESSION(std::get<Idx>(_val) =
                                 ::tim::math::pow(std::get<Idx>(_val), _m));
    return _val;
}

template <typename Tp>
Tp
pow(Tp _val, double _m)
{
    return ::tim::math::pow(_val, _m, get_index_sequence<Tp>::value);
}

//--------------------------------------------------------------------------------------//

template <typename Tp>
Tp
sqr(Tp _val)
{
    static_assert(!concepts::is_null_type<Tp>::value, "Error! null type");
    return ::tim::math::pow(_val, 2.0);
}

//--------------------------------------------------------------------------------------//

template <typename Tp, enable_if_t<std::is_arithmetic<Tp>::value> = 0>
TIMEMORY_INLINE Tp
                min(Tp _lhs, Tp _rhs, type_list<>)
{
    static_assert(!concepts::is_null_type<Tp>::value, "Error! null type");
    return (_rhs > _lhs) ? _lhs : _rhs;
}

template <typename Tp, typename Vp = typename Tp::value_type>
auto
min(const Tp& _lhs, const Tp& _rhs, type_list<>, ...) -> decltype(std::begin(_lhs), Tp{})
{
    static_assert(!concepts::is_null_type<Tp>::value, "Error! null type");
    auto _nl    = mpl::get_size(_lhs);
    auto _nr    = mpl::get_size(_rhs);
    using Int_t = decltype(_nl);
    assert(_nl == _nr);

    auto _n = std::min<Int_t>(_nl, _nr);
    Tp   _ret{};
    mpl::resize(_ret, _n);

    for(Int_t i = 0; i < _n; ++i)
    {
        auto litr = std::begin(_lhs) + i;
        auto ritr = std::begin(_rhs) + i;
        auto itr  = std::begin(_ret) + i;
        *itr      = min(*litr, *ritr, get_index_sequence<decay_t<decltype(*itr)>>::value);
    }
    return _ret;
}

template <typename Tp, typename Kp = typename Tp::key_type,
          typename Mp = typename Tp::mapped_type>
auto
min(const Tp& _lhs, const Tp& _rhs, type_list<>) -> decltype(std::begin(_lhs), Tp{})
{
    static_assert(!concepts::is_null_type<Tp>::value, "Error! null type");
    auto _nl    = mpl::get_size(_lhs);
    auto _nr    = mpl::get_size(_rhs);
    using Int_t = decltype(_nl);
    assert(_nl == _nr);

    auto _n = std::min<Int_t>(_nl, _nr);
    Tp   _ret{};
    mpl::resize(_ret, _n);

    for(Int_t i = 0; i < _n; ++i)
    {
        auto litr   = std::begin(_lhs) + i;
        auto ritr   = std::begin(_rhs) + i;
        auto itr    = std::begin(_ret) + i;
        itr->second = min(litr->second, ritr->second,
                          get_index_sequence<decay_t<decltype(itr->second)>>::value);
    }
    return _ret;
}

template <typename Tp, size_t... Idx>
auto
min(const Tp& _lhs, const Tp& _rhs, index_sequence<Idx...>)
    -> decltype(std::get<0>(_lhs), Tp{})
{
    static_assert(!concepts::is_null_type<Tp>::value, "Error! null type");
    Tp _ret{};
    TIMEMORY_FOLD_EXPRESSION(
        std::get<Idx>(_ret) =
            min(std::get<Idx>(_lhs), std::get<Idx>(_rhs),
                get_index_sequence<decay_t<decltype(std::get<Idx>(_ret))>>::value));
    return _ret;
}

template <typename Tp>
Tp
min(const Tp& _lhs, const Tp& _rhs)
{
    static_assert(!concepts::is_null_type<Tp>::value, "Error! null type");
    return min(_lhs, _rhs, get_index_sequence<Tp>::value);
}

//--------------------------------------------------------------------------------------//

template <typename Tp, enable_if_t<std::is_arithmetic<Tp>::value> = 0>
TIMEMORY_INLINE Tp
                max(Tp _lhs, Tp _rhs, type_list<>)
{
    static_assert(!concepts::is_null_type<Tp>::value, "Error! null type");
    return (_rhs < _lhs) ? _lhs : _rhs;
}

template <typename Tp, typename Vp = typename Tp::value_type>
auto
max(const Tp& _lhs, const Tp& _rhs, type_list<>, ...) -> decltype(std::begin(_lhs), Tp{})
{
    static_assert(!concepts::is_null_type<Tp>::value, "Error! null type");
    auto _nl    = mpl::get_size(_lhs);
    auto _nr    = mpl::get_size(_rhs);
    using Int_t = decltype(_nl);
    assert(_nl == _nr);

    auto _n = std::min<Int_t>(_nl, _nr);
    Tp   _ret{};
    mpl::resize(_ret, _n);

    for(Int_t i = 0; i < _n; ++i)
    {
        auto litr = std::begin(_lhs) + i;
        auto ritr = std::begin(_rhs) + i;
        auto itr  = std::begin(_ret) + i;
        *itr      = max(*litr, *ritr, get_index_sequence<decay_t<decltype(*itr)>>::value);
    }
    return _ret;
}

template <typename Tp, typename Kp = typename Tp::key_type,
          typename Mp = typename Tp::mapped_type>
auto
max(const Tp& _lhs, const Tp& _rhs, type_list<>) -> decltype(std::begin(_lhs), Tp{})
{
    static_assert(!concepts::is_null_type<Tp>::value, "Error! null type");
    auto _nl    = mpl::get_size(_lhs);
    auto _nr    = mpl::get_size(_rhs);
    using Int_t = decltype(_nl);
    assert(_nl == _nr);

    auto _n = std::min<Int_t>(_nl, _nr);
    Tp   _ret{};
    mpl::resize(_ret, _n);

    for(Int_t i = 0; i < _n; ++i)
    {
        auto litr   = std::begin(_lhs) + i;
        auto ritr   = std::begin(_rhs) + i;
        auto itr    = std::begin(_ret) + i;
        itr->second = max(litr->second, ritr->second,
                          get_index_sequence<decay_t<decltype(itr->second)>>::value);
    }
    return _ret;
}

template <typename Tp, size_t... Idx>
auto
max(const Tp& _lhs, const Tp& _rhs, index_sequence<Idx...>)
    -> decltype(std::get<0>(_lhs), Tp{})
{
    static_assert(!concepts::is_null_type<Tp>::value, "Error! null type");
    Tp _ret{};
    TIMEMORY_FOLD_EXPRESSION(
        std::get<Idx>(_ret) =
            max(std::get<Idx>(_lhs), std::get<Idx>(_rhs),
                get_index_sequence<decay_t<decltype(std::get<Idx>(_ret))>>::value));
    return _ret;
}

template <typename Tp>
Tp
max(const Tp& _lhs, const Tp& _rhs)
{
    static_assert(!concepts::is_null_type<Tp>::value, "Error! null type");
    return max(_lhs, _rhs, get_index_sequence<Tp>::value);
}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename Up>
void
assign(Tp& _lhs, Up&& _rhs)
{
    _lhs = std::forward<Up>(_rhs);
}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename Up>
TIMEMORY_INLINE auto
plus(Tp& _lhs, const Up& _rhs, type_list<>, ...) -> decltype(_lhs += _rhs, void())
{
    static_assert(!concepts::is_null_type<Tp>::value, "Error! null type");
    _lhs += _rhs;
}

template <typename Tp, typename Up, typename Vp = typename Tp::value_type>
auto
plus(Tp& _lhs, const Up& _rhs, type_list<>, long) -> decltype(std::begin(_lhs), void())
{
    static_assert(!concepts::is_null_type<Tp>::value, "Error! null type");
    auto _n = mpl::get_size(_rhs);
    if(mpl::get_size(_lhs) < _n)
        mpl::resize(_lhs, _n);

    for(decltype(_n) i = 0; i < _n; ++i)
    {
        auto litr = std::begin(_lhs) + i;
        auto ritr = std::begin(_rhs) + i;
        plus(*litr, *ritr, get_index_sequence<decay_t<decltype(*litr)>>::value, 0);
    }
}

template <typename Tp, typename Up, typename Kp = typename Tp::key_type,
          typename Mp = typename Tp::mapped_type>
auto
plus(Tp& _lhs, const Up& _rhs, type_list<>, int) -> decltype(std::begin(_lhs), void())
{
    static_assert(!concepts::is_null_type<Tp>::value, "Error! null type");

    for(auto litr = std::begin(_lhs); litr != std::end(_lhs); ++litr)
    {
        auto ritr = _rhs.find(litr->first);
        if(ritr == std::end(_rhs))
            continue;
        plus(litr->second, ritr->second,
             get_index_sequence<decay_t<decltype(litr->second)>>::value, 0);
    }

    for(auto ritr = std::begin(_rhs); ritr != std::end(_rhs); ++ritr)
    {
        auto litr = _lhs.find(ritr->first);
        if(litr == std::end(_lhs))
            continue;
        _lhs[ritr->first] = ritr->second;
    }
}

template <typename Tp, typename Up>
TIMEMORY_INLINE auto
plus(Tp&, const Up&, index_sequence<>, int)
{}

template <typename Tp, typename Up, size_t... Idx>
auto
plus(Tp& _lhs, const Up& _rhs, index_sequence<Idx...>, long)
    -> decltype(std::get<0>(_lhs), void())
{
    static_assert(!concepts::is_null_type<Tp>::value, "Error! null type");
    TIMEMORY_FOLD_EXPRESSION(
        plus(std::get<Idx>(_lhs), std::get<Idx>(_rhs),
             get_index_sequence<decay_t<decltype(std::get<Idx>(_lhs))>>::value, 0));
}

template <typename Tp, typename Up, enable_if_t<!concepts::is_null_type<Tp>::value>>
Tp&
plus(Tp& _lhs, const Up& _rhs)
{
    plus(_lhs, _rhs, get_index_sequence<Tp>::value, 0);
    return _lhs;
}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename Up>
TIMEMORY_INLINE auto
minus(Tp& _lhs, const Up& _rhs, type_list<>, ...) -> decltype(_lhs -= _rhs, void())
{
    static_assert(!concepts::is_null_type<Tp>::value, "Error! null type");
    _lhs -= _rhs;
}

template <typename Tp, typename Up, typename Vp = typename Tp::value_type>
auto
minus(Tp& _lhs, const Up& _rhs, type_list<>, long) -> decltype(std::begin(_lhs), void())
{
    static_assert(!concepts::is_null_type<Tp>::value, "Error! null type");
    auto _n = mpl::get_size(_rhs);
    if(mpl::get_size(_lhs) < _n)
        mpl::resize(_lhs, _n);

    for(decltype(_n) i = 0; i < _n; ++i)
    {
        auto litr = std::begin(_lhs) + i;
        auto ritr = std::begin(_rhs) + i;
        minus(*litr, *ritr, get_index_sequence<decay_t<decltype(*litr)>>::value, 0);
    }
}

template <typename Tp, typename Up, typename Kp = typename Tp::key_type,
          typename Mp = typename Tp::mapped_type>
auto
minus(Tp& _lhs, const Up& _rhs, type_list<>, int) -> decltype(std::begin(_lhs), void())
{
    static_assert(!concepts::is_null_type<Tp>::value, "Error! null type");

    for(auto litr = std::begin(_lhs); litr != std::end(_lhs); ++litr)
    {
        auto ritr = _rhs.find(litr->first);
        if(ritr == std::end(_rhs))
            continue;
        minus(litr->second, ritr->second,
              get_index_sequence<decay_t<decltype(litr->second)>>::value, 0);
    }
}

template <typename Tp, typename Up>
TIMEMORY_INLINE auto
minus(Tp&, const Up&, index_sequence<>, int)
{}

template <typename Tp, typename Up, size_t... Idx>
auto
minus(Tp& _lhs, const Up& _rhs, index_sequence<Idx...>, long)
    -> decltype(std::get<0>(_lhs), void())
{
    static_assert(!concepts::is_null_type<Tp>::value, "Error! null type");
    TIMEMORY_FOLD_EXPRESSION(
        minus(std::get<Idx>(_lhs), std::get<Idx>(_rhs),
              get_index_sequence<decay_t<decltype(std::get<Idx>(_lhs))>>::value, 0));
}

template <typename Tp, typename Up, enable_if_t<!concepts::is_null_type<Tp>::value>>
Tp&
minus(Tp& _lhs, const Up& _rhs)
{
    minus(_lhs, _rhs, get_index_sequence<Tp>::value, 0);
    return _lhs;
}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename Up>
TIMEMORY_INLINE auto
multiply(Tp& _lhs, Up _rhs, type_list<>, ...) -> decltype(_lhs *= _rhs, void())
{
    static_assert(!concepts::is_null_type<Tp>::value, "Error! null type");
    _lhs *= _rhs;
}

template <typename Tp, typename Up, typename Vp = typename Tp::value_type>
auto
multiply(Tp& _lhs, const Up& _rhs, type_list<>, long)
    -> decltype((std::begin(_lhs), std::begin(_rhs)), void())
{
    static_assert(!concepts::is_null_type<Tp>::value, "Error! null type");
    auto _n = mpl::get_size(_rhs);
    if(mpl::get_size(_lhs) < _n)
        mpl::resize(_lhs, _n);

    for(decltype(_n) i = 0; i < _n; ++i)
    {
        auto litr = std::begin(_lhs) + i;
        auto ritr = std::begin(_rhs) + i;
        multiply(*litr, *ritr, get_index_sequence<decay_t<decltype(*litr)>>::value, 0);
    }
}

template <typename Tp, typename Up, typename Vp = typename Tp::value_type,
          enable_if_t<std::is_arithmetic<Up>::value> = 0>
auto
multiply(Tp& _lhs, const Up& _rhs, type_list<>, long)
    -> decltype(std::begin(_lhs), void())
{
    static_assert(!concepts::is_null_type<Tp>::value, "Error! null type");
    auto _n = mpl::get_size(_lhs);
    for(decltype(_n) i = 0; i < _n; ++i)
    {
        auto litr = std::begin(_lhs) + i;
        multiply(*litr, _rhs, get_index_sequence<decay_t<decltype(*litr)>>::value, 0);
    }
}

template <typename Tp, typename Up, typename Kp = typename Tp::key_type,
          typename Mp                                 = typename Tp::mapped_type,
          enable_if_t<!std::is_arithmetic<Up>::value> = 0>
auto
multiply(Tp& _lhs, const Up& _rhs, type_list<>, int) -> decltype(std::begin(_lhs), void())
{
    static_assert(!concepts::is_null_type<Tp>::value, "Error! null type");

    for(auto litr = std::begin(_lhs); litr != std::end(_lhs); ++litr)
    {
        auto ritr = _rhs.find(litr->first);
        if(ritr == std::end(_rhs))
            continue;
        multiply(litr->second, ritr->second,
                 get_index_sequence<decay_t<decltype(litr->second)>>::value, 0);
    }
}

template <typename Tp, typename Up, typename Kp = typename Tp::key_type,
          typename Mp                                = typename Tp::mapped_type,
          enable_if_t<std::is_arithmetic<Up>::value> = 0>
auto
multiply(Tp& _lhs, const Up& _rhs, type_list<>, int) -> decltype(std::begin(_lhs), void())
{
    static_assert(!concepts::is_null_type<Tp>::value, "Error! null type");

    for(auto litr = std::begin(_lhs); litr != std::end(_lhs); ++litr)
    {
        multiply(litr->second, _rhs,
                 get_index_sequence<decay_t<decltype(litr->second)>>::value, 0);
    }
}

template <typename Tp, typename Up>
TIMEMORY_INLINE auto
multiply(Tp&, const Up&, index_sequence<>, int)
{}

template <typename Tp, typename Up, size_t... Idx,
          enable_if_t<!std::is_arithmetic<Up>::value> = 0>
auto
multiply(Tp& _lhs, const Up& _rhs, index_sequence<Idx...>, long)
    -> decltype(std::get<0>(_lhs), void())
{
    static_assert(!concepts::is_null_type<Tp>::value, "Error! null type");
    TIMEMORY_FOLD_EXPRESSION(
        multiply(std::get<Idx>(_lhs), std::get<Idx>(_rhs),
                 get_index_sequence<decay_t<decltype(std::get<Idx>(_lhs))>>::value, 0));
}

template <typename Tp, typename Up, size_t... Idx,
          enable_if_t<std::is_arithmetic<Up>::value> = 0>
auto
multiply(Tp& _lhs, const Up& _rhs, index_sequence<Idx...>, long)
    -> decltype(std::get<0>(_lhs), void())
{
    static_assert(!concepts::is_null_type<Tp>::value, "Error! null type");
    TIMEMORY_FOLD_EXPRESSION(
        multiply(std::get<Idx>(_lhs), _rhs,
                 get_index_sequence<decay_t<decltype(std::get<Idx>(_lhs))>>::value, 0));
}

template <typename Tp, typename Up, enable_if_t<!concepts::is_null_type<Tp>::value>>
Tp&
multiply(Tp& _lhs, const Up& _rhs)
{
    multiply(_lhs, _rhs, get_index_sequence<Tp>::value, 0);
    return _lhs;
}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename Up>
TIMEMORY_INLINE auto
divide(Tp& _lhs, Up _rhs, type_list<>, ...) -> decltype(_lhs /= _rhs, void())
{
    static_assert(!concepts::is_null_type<Tp>::value, "Error! null type");
    _lhs /= _rhs;
}

template <typename Tp, typename Up, typename Vp = typename Tp::value_type,
          enable_if_t<!std::is_arithmetic<Up>::value> = 0>
auto
divide(Tp& _lhs, const Up& _rhs, type_list<>, long) -> decltype(std::begin(_lhs), void())
{
    static_assert(!concepts::is_null_type<Tp>::value, "Error! null type");
    auto _n = mpl::get_size(_rhs);
    if(mpl::get_size(_lhs) < _n)
        mpl::resize(_lhs, _n);

    for(decltype(_n) i = 0; i < _n; ++i)
    {
        auto litr = std::begin(_lhs) + i;
        auto ritr = std::begin(_rhs) + i;
        divide(*litr, *ritr, get_index_sequence<decay_t<decltype(*litr)>>::value, 0);
    }
}

template <typename Tp, typename Up, typename Vp = typename Tp::value_type,
          enable_if_t<std::is_arithmetic<Up>::value> = 0>
auto
divide(Tp& _lhs, const Up& _rhs, type_list<>, long) -> decltype(std::begin(_lhs), void())
{
    static_assert(!concepts::is_null_type<Tp>::value, "Error! null type");
    auto _n = mpl::get_size(_lhs);
    for(decltype(_n) i = 0; i < _n; ++i)
    {
        auto litr = std::begin(_lhs) + i;
        divide(*litr, _rhs, get_index_sequence<decay_t<decltype(*litr)>>::value, 0);
    }
}

template <typename Tp, typename Up, typename Kp = typename Tp::key_type,
          typename Mp                                 = typename Tp::mapped_type,
          enable_if_t<!std::is_arithmetic<Up>::value> = 0>
auto
divide(Tp& _lhs, const Up& _rhs, type_list<>, int) -> decltype(std::begin(_lhs), void())
{
    static_assert(!concepts::is_null_type<Tp>::value, "Error! null type");

    for(auto litr = std::begin(_lhs); litr != std::end(_lhs); ++litr)
    {
        auto ritr = _rhs.find(litr->first);
        if(ritr == std::end(_rhs))
            continue;
        divide(litr->second, ritr->second,
               get_index_sequence<decay_t<decltype(litr->second)>>::value, 0);
    }
}

template <typename Tp, typename Up, typename Kp = typename Tp::key_type,
          typename Mp                                = typename Tp::mapped_type,
          enable_if_t<std::is_arithmetic<Up>::value> = 0>
auto
divide(Tp& _lhs, const Up& _rhs, type_list<>, int) -> decltype(std::begin(_lhs), void())
{
    static_assert(!concepts::is_null_type<Tp>::value, "Error! null type");

    for(auto litr = std::begin(_lhs); litr != std::end(_lhs); ++litr)
    {
        divide(litr->second, _rhs,
               get_index_sequence<decay_t<decltype(litr->second)>>::value, 0);
    }
}

template <typename Tp, typename Up>
TIMEMORY_INLINE auto
divide(Tp&, const Up&, index_sequence<>, int)
{}

template <typename Tp, typename Up, size_t... Idx,
          enable_if_t<!std::is_arithmetic<Up>::value> = 0>
auto
divide(Tp& _lhs, const Up& _rhs, index_sequence<Idx...>, long)
    -> decltype(std::get<0>(_lhs), void())
{
    static_assert(!concepts::is_null_type<Tp>::value, "Error! null type");
    TIMEMORY_FOLD_EXPRESSION(
        divide(std::get<Idx>(_lhs), std::get<Idx>(_rhs),
               get_index_sequence<decay_t<decltype(std::get<Idx>(_lhs))>>::value, 0));
}

template <typename Tp, typename Up, size_t... Idx,
          enable_if_t<std::is_arithmetic<Up>::value> = 0>
auto
divide(Tp& _lhs, const Up& _rhs, index_sequence<Idx...>, long)
    -> decltype(std::get<0>(_lhs), void())
{
    static_assert(!concepts::is_null_type<Tp>::value, "Error! null type");
    TIMEMORY_FOLD_EXPRESSION(
        divide(std::get<Idx>(_lhs), _rhs,
               get_index_sequence<decay_t<decltype(std::get<Idx>(_lhs))>>::value, 0));
}

template <typename Tp, typename Up, enable_if_t<!concepts::is_null_type<Tp>::value>>
Tp&
divide(Tp& _lhs, const Up& _rhs)
{
    divide(_lhs, _rhs, get_index_sequence<Tp>::value, 0);
    return _lhs;
}

//--------------------------------------------------------------------------------------//

template <typename Tp, enable_if_t<std::is_arithmetic<Tp>::value> = 0>
TIMEMORY_INLINE Tp
                percent_diff(Tp _lhs, Tp _rhs, type_list<>, ...)
{
    constexpr Tp _zero    = Tp(0.0);
    constexpr Tp _one     = Tp(1.0);
    constexpr Tp _hundred = Tp(100.0);
    Tp&&         _pdiff   = (_rhs > _zero) ? ((_one - (_lhs / _rhs)) * _hundred) : _zero;
    return (_pdiff < _zero) ? _zero : _pdiff;
}

template <typename Tp,
          enable_if_t<!std::is_arithmetic<Tp>::value && std::is_class<Tp>::value> = 0>
TIMEMORY_INLINE auto
percent_diff(Tp _lhs, Tp _rhs, type_list<>, ...) -> decltype(_lhs.percent_diff(_rhs))
{
    return _lhs.percent_diff(_rhs);
}

template <typename Tp, typename Vp = typename Tp::value_type>
auto
percent_diff(const Tp& _lhs, const Tp& _rhs, type_list<>, long)
    -> decltype(std::begin(_lhs), Tp{})
{
    auto _nl    = mpl::get_size(_lhs);
    auto _nr    = mpl::get_size(_rhs);
    using Int_t = decltype(_nl);

    auto _n = std::min<Int_t>(_nl, _nr);
    Tp   _ret{};
    mpl::resize(_ret, _n);

    // initialize
    for(auto& itr : _ret)
        itr = Vp{};

    // compute
    for(Int_t i = 0; i < _n; ++i)
    {
        auto litr = std::begin(_lhs) + i;
        auto ritr = std::begin(_rhs) + i;
        auto itr  = std::begin(_ret) + i;
        *itr      = percent_diff(*litr, *ritr,
                            get_index_sequence<decay_t<decltype(*itr)>>::value);
    }
    return _ret;
}

template <typename Tp, typename Kp = typename Tp::key_type,
          typename Mp = typename Tp::mapped_type>
auto
percent_diff(const Tp& _lhs, const Tp& _rhs, type_list<>, int)
    -> decltype(std::begin(_lhs), Tp{})
{
    assert(_lhs.size() == _rhs.size());
    Tp _ret{};
    for(auto litr = std::begin(_lhs); litr != std::end(_lhs); ++litr)
    {
        auto ritr = _rhs.find(litr->first);
        if(ritr == std::end(_rhs))
        {
            _ret[litr->first] = Mp{};
        }
        else
        {
            _ret[litr->first] =
                percent_diff(litr->second, ritr->second,
                             get_index_sequence<decay_t<decltype(litr->second)>>::value);
        }
    }
    return _ret;
}

template <typename Tp, size_t... Idx>
auto
percent_diff(const Tp& _lhs, const Tp& _rhs, index_sequence<Idx...>, ...)
    -> decltype(std::get<0>(_lhs), Tp{})
{
    Tp _ret{};
    TIMEMORY_FOLD_EXPRESSION(
        std::get<Idx>(_ret) = percent_diff(
            std::get<Idx>(_lhs), std::get<Idx>(_rhs),
            get_index_sequence<decay_t<decltype(std::get<Idx>(_ret))>>::value));
    return _ret;
}

template <typename Tp>
Tp
percent_diff(const Tp& _lhs, const Tp& _rhs)
{
    return percent_diff(_lhs, _rhs, get_index_sequence<Tp>::value, 0);
}

//--------------------------------------------------------------------------------------//
/// \struct tim::math::compute
/// \brief Struct for performing math operations on complex data structures without using
/// globally overload operators (e.g. `lhs += rhs`) and generic functions (`lhs =
/// abs(rhs)`)
///
template <typename Tp, typename Up = Tp>
struct compute
{
    using this_type  = compute<Tp, Up>;
    using type       = Tp;
    using value_type = Up;

    static TIMEMORY_INLINE decltype(auto) abs(const type& _v)
    {
        return this_type::abs(_v, 0);
    }

    static TIMEMORY_INLINE decltype(auto) sqr(const type& _v)
    {
        return this_type::sqr(_v, 0);
    }

    static TIMEMORY_INLINE decltype(auto) sqrt(const type& _v)
    {
        return this_type::sqrt(_v, 0);
    }

    template <typename V>
    static TIMEMORY_INLINE decltype(auto) min(const type& _l, const V& _r)
    {
        return this_type::min(_l, _r, 0);
    }

    template <typename V>
    static TIMEMORY_INLINE decltype(auto) max(const type& _l, const V& _r)
    {
        return this_type::max(_l, _r, 0);
    }

    template <typename V>
    static TIMEMORY_INLINE decltype(auto) percent_diff(const type& _l, const V& _r)
    {
        return this_type::percent_diff(_l, _r, 0);
    }

    // reference
    template <typename V>
    static TIMEMORY_INLINE decltype(auto) plus(type& _l, const V& _r)
    {
        return this_type::plus(_l, _r, 0);
    }

    template <typename V>
    static TIMEMORY_INLINE decltype(auto) minus(type& _l, const V& _r)
    {
        return this_type::minus(_l, _r, 0);
    }

    template <typename V>
    static TIMEMORY_INLINE decltype(auto) multiply(type& _l, const V& _r)
    {
        return this_type::multiply(_l, _r, 0);
    }

    template <typename V>
    static TIMEMORY_INLINE decltype(auto) divide(type& _l, const V& _r)
    {
        return this_type::divide(_l, _r, 0);
    }

    // const ref
    template <typename V>
    static TIMEMORY_INLINE auto plus(const type& _l, const V& _r)
    {
        type _t{ _l };
        return this_type::plus(_t, _r, 0);
    }

    template <typename V>
    static TIMEMORY_INLINE auto minus(const type& _l, const V& _r)
    {
        type _t{ _l };
        return this_type::minus(_t, _r, 0);
    }

    template <typename V>
    static TIMEMORY_INLINE auto multiply(const type& _l, const V& _r)
    {
        type _t{ _l };
        return this_type::multiply(_t, _r, 0);
    }

    template <typename V>
    static TIMEMORY_INLINE auto divide(const type& _l, const V& _r)
    {
        type _t{ _l };
        return this_type::divide(_t, _r, 0);
    }

private:
    //----------------------------------------------------------------------------------//
    // tim::math overload available
    //
    template <typename V>
    static TIMEMORY_INLINE auto abs(const V& _v, int) -> decltype(::tim::math::abs(_v))
    {
        return ::tim::math::abs(_v);
    }

    template <typename V>
    static TIMEMORY_INLINE auto sqr(const V& _v, int) -> decltype(::tim::math::sqr(_v))
    {
        return ::tim::math::sqr(_v);
    }

    template <typename V>
    static TIMEMORY_INLINE auto sqrt(const V& _v, int) -> decltype(::tim::math::sqrt(_v))
    {
        return ::tim::math::sqrt(_v);
    }

    template <typename V>
    static TIMEMORY_INLINE auto min(const type& _l, const V& _r, int)
        -> decltype(::tim::math::min(_l, _r, get_index_sequence<type>::value),
                    ::tim::math::min(_l, _r))
    {
        return ::tim::math::min(_l, _r);
    }

    template <typename V>
    static TIMEMORY_INLINE auto max(const type& _l, const V& _r, int)
        -> decltype(::tim::math::max(_l, _r, get_index_sequence<type>::value),
                    ::tim::math::max(_l, _r))
    {
        return ::tim::math::max(_l, _r);
    }

    template <typename V>
    static TIMEMORY_INLINE auto percent_diff(const type& _l, const V& _r, int)
        -> decltype(::tim::math::percent_diff(_l, _r, get_index_sequence<type>::value, 0),
                    ::tim::math::percent_diff(_l, _r))
    {
        return ::tim::math::percent_diff(_l, _r);
    }

    template <typename V>
    static TIMEMORY_INLINE auto plus(type& _l, const V& _r, int)
        -> decltype(::tim::math::plus(_l, _r, get_index_sequence<type>::value, 0),
                    std::declval<type&>())
    {
        return ::tim::math::plus(_l, _r);
    }

    template <typename V>
    static TIMEMORY_INLINE auto minus(type& _l, const V& _r, int)
        -> decltype(::tim::math::minus(_l, _r, get_index_sequence<type>::value, 0),
                    std::declval<type&>())
    {
        return ::tim::math::minus(_l, _r);
    }

    template <typename V>
    static TIMEMORY_INLINE auto multiply(type& _l, const V& _r, int)
        -> decltype(::tim::math::multiply(_l, _r, get_index_sequence<type>::value, 0),
                    std::declval<type&>())
    {
        return ::tim::math::multiply(_l, _r);
    }

    template <typename V, typename U = void>
    static TIMEMORY_INLINE auto divide(type& _l, const V& _r, int)
        -> decltype(::tim::math::divide(_l, _r, get_index_sequence<type>::value, 0),
                    std::declval<type&>())
    {
        return ::tim::math::divide(_l, _r);
    }

    //----------------------------------------------------------------------------------//
    // no tim::math overload available
    //
    template <typename V>
    static TIMEMORY_INLINE auto abs(const V& _v, long)
    {
        return _v;
    }

    template <typename V>
    static TIMEMORY_INLINE auto sqr(const V& _v, long)
    {
        return _v;
    }

    template <typename V>
    static TIMEMORY_INLINE auto sqrt(const V& _v, long)
    {
        return _v;
    }

    template <typename V>
    static TIMEMORY_INLINE auto min(const type& _l, const V&, long)
    {
        return _l;
    }

    template <typename V>
    static TIMEMORY_INLINE auto max(const type& _l, const V&, long)
    {
        return _l;
    }

    template <typename V>
    static TIMEMORY_INLINE auto percent_diff(const type& _l, const V&, long)
    {
        return _l;
    }

    template <typename V>
    static TIMEMORY_INLINE auto& plus(type& _l, const V&, long)
    {
        return _l;
    }

    template <typename V>
    static TIMEMORY_INLINE auto& minus(type& _l, const V&, long)
    {
        return _l;
    }

    template <typename V>
    static TIMEMORY_INLINE auto& multiply(type& _l, const V&, long)
    {
        return _l;
    }

    template <typename V, typename U = void>
    static TIMEMORY_INLINE auto& divide(type& _l, const V&, long)
    {
        return _l;
    }
};

//--------------------------------------------------------------------------------------//

#define TIMEMORY_MATH_NULL_TYPE_COMPUTE(TYPE)                                            \
    template <>                                                                          \
    struct compute<TYPE, TYPE>                                                           \
    {                                                                                    \
        using type = TYPE;                                                               \
        static type abs(const type&) { return type{}; }                                  \
        static type sqr(const type&) { return type{}; }                                  \
        static type sqrt(const type&) { return type{}; }                                 \
        static type max(const type&, const type&) { return type{}; }                     \
        static type min(const type&, const type&) { return type{}; }                     \
        static type percent_diff(const type&, const type&) { return type{}; }            \
                                                                                         \
        template <typename Vp>                                                           \
        static decltype(auto) plus(type& lhs, const Vp&)                                 \
        {                                                                                \
            return lhs;                                                                  \
        }                                                                                \
        template <typename Vp>                                                           \
        static decltype(auto) minus(type& lhs, const Vp&)                                \
        {                                                                                \
            return lhs;                                                                  \
        }                                                                                \
        template <typename Vp>                                                           \
        static decltype(auto) multiply(type& lhs, const Vp&)                             \
        {                                                                                \
            return lhs;                                                                  \
        }                                                                                \
        template <typename Vp>                                                           \
        static decltype(auto) divide(type& lhs, const Vp&)                               \
        {                                                                                \
            return lhs;                                                                  \
        }                                                                                \
    };

TIMEMORY_MATH_NULL_TYPE_COMPUTE(std::tuple<>)
TIMEMORY_MATH_NULL_TYPE_COMPUTE(null_type)
TIMEMORY_MATH_NULL_TYPE_COMPUTE(type_list<>)

//--------------------------------------------------------------------------------------//

}  // namespace math
}  // namespace tim
