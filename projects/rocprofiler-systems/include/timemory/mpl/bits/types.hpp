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

/** \file mpl/bits/types.hpp
 * \headerfile mpl/bits/types.hpp "timemory/mpl/bits/types.hpp"
 *
 * This is a declaration of the STL overload types
 *
 */

#pragma once

#include <array>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "timemory/mpl/types.hpp"

namespace tim
{
namespace stl
{
namespace ostream
{
//--------------------------------------------------------------------------------------//
//
//      operator <<
//
//--------------------------------------------------------------------------------------//

template <typename T, typename U>
std::ostream&
operator<<(std::ostream&, const std::pair<T, U>&);

//--------------------------------------------------------------------------------------//

template <typename... Types>
std::ostream&
operator<<(std::ostream&, const std::tuple<Types...>&);

//--------------------------------------------------------------------------------------//

template <typename Tp, typename... _Extra>
std::ostream&
operator<<(std::ostream&, const std::vector<Tp, _Extra...>&);

//--------------------------------------------------------------------------------------//

template <typename Tp, size_t N>
std::ostream&
operator<<(std::ostream&, const std::array<Tp, N>&);

//--------------------------------------------------------------------------------------//

template <template <typename...> class _Tuple, typename... Types, size_t... Idx>
void
tuple_printer(const _Tuple<Types...>& obj, std::ostream& os, index_sequence<Idx...>)
{
    using namespace ::tim::stl::ostream;
    constexpr size_t N = sizeof...(Types);

    if(N > 0)
        os << "(";
    char delim[N];
    TIMEMORY_FOLD_EXPRESSION(delim[Idx] = ',');
    delim[N - 1] = '\0';
    TIMEMORY_FOLD_EXPRESSION(os << std::get<Idx>(obj) << delim[Idx]);
    if(N > 0)
        os << ")";
}

//--------------------------------------------------------------------------------------//

}  // namespace ostream

//--------------------------------------------------------------------------------------//
//
//      operator +=
//      operator +
//
//--------------------------------------------------------------------------------------//

template <typename Tp, size_t N, typename Other>
std::array<Tp, N>&
operator+=(std::array<Tp, N>&, Other&&);

//--------------------------------------------------------------------------------------//

template <typename Lhs, typename Rhs, typename Other>
std::pair<Lhs, Rhs>&
operator+=(std::pair<Lhs, Rhs>&, Other&&);

//--------------------------------------------------------------------------------------//

template <typename Tp, typename... _Extra, typename Other>
std::vector<Tp, _Extra...>&
operator+=(std::vector<Tp, _Extra...>&, Other&&);

//--------------------------------------------------------------------------------------//

template <typename... Types, typename Other>
std::tuple<Types...>&
operator+=(std::tuple<Types...>&, Other&&);

//--------------------------------------------------------------------------------------//
//
//      operator -=
//      operator -
//
//--------------------------------------------------------------------------------------//

template <typename Tp, size_t N>
std::array<Tp, N>&
operator-=(std::array<Tp, N>&, const std::array<Tp, N>&);

template <typename Lhs, size_t N, typename Rhs,
          enable_if_t<std::is_arithmetic<decay_t<Rhs>>::value, int> = 0>
std::array<Lhs, N>&
operator-=(std::array<Lhs, N>&, const Rhs&);

//--------------------------------------------------------------------------------------//

template <typename Lhs, typename Rhs>
std::pair<Lhs, Rhs>&
operator-=(std::pair<Lhs, Rhs>&, const std::pair<Lhs, Rhs>&);

template <typename Lhs, typename Rhs, typename ArithT,
          enable_if_t<std::is_arithmetic<decay_t<ArithT>>::value, int> = 0>
std::pair<Lhs, Rhs>&
operator-=(std::pair<Lhs, Rhs>&, const ArithT&);

//--------------------------------------------------------------------------------------//

template <typename Tp, typename... _Extra>
std::vector<Tp, _Extra...>&
operator-=(std::vector<Tp, _Extra...>&, const std::vector<Tp, _Extra...>&);

template <typename Lhs, typename Rhs, typename... _Extra,
          enable_if_t<std::is_arithmetic<decay_t<Rhs>>::value, int> = 0>
std::vector<Lhs, _Extra...>&
operator-=(std::vector<Lhs, _Extra...>&, const Rhs&);

//--------------------------------------------------------------------------------------//

template <typename... Types>
std::tuple<Types...>&
operator-=(std::tuple<Types...>&, const std::tuple<Types...>&);

template <typename... Lhs, typename Rhs,
          enable_if_t<std::is_arithmetic<decay_t<Rhs>>::value, int> = 0>
std::tuple<Lhs...>&
operator-=(std::tuple<Lhs...>&, const Rhs&);

//--------------------------------------------------------------------------------------//
//
//      operator *=
//      operator *
//
//--------------------------------------------------------------------------------------//

template <typename Tp, size_t N>
std::array<Tp, N>&
operator*=(std::array<Tp, N>&, const std::array<Tp, N>&);

template <typename Lhs, size_t N, typename Rhs,
          enable_if_t<std::is_arithmetic<decay_t<Rhs>>::value, int> = 0>
std::array<Lhs, N>&
operator*=(std::array<Lhs, N>&, const Rhs&);

//--------------------------------------------------------------------------------------//

template <typename Lhs, typename Rhs>
std::pair<Lhs, Rhs>&
operator*=(std::pair<Lhs, Rhs>&, const std::pair<Lhs, Rhs>&);

template <typename Lhs, typename Rhs, typename ArithT,
          enable_if_t<std::is_arithmetic<decay_t<ArithT>>::value, int> = 0>
std::pair<Lhs, Rhs>&
operator*=(std::pair<Lhs, Rhs>&, const ArithT&);

//--------------------------------------------------------------------------------------//

template <typename Tp, typename... _Extra>
std::vector<Tp, _Extra...>&
operator*=(std::vector<Tp, _Extra...>&, const std::vector<Tp, _Extra...>&);

template <typename Lhs, typename Rhs, typename... _Extra,
          enable_if_t<std::is_arithmetic<decay_t<Rhs>>::value, int> = 0>
std::vector<Lhs, _Extra...>&
operator*=(std::vector<Lhs, _Extra...>&, const Rhs&);

//--------------------------------------------------------------------------------------//

template <typename... Types>
std::tuple<Types...>&
operator*=(std::tuple<Types...>&, const std::tuple<Types...>&);

template <typename... Lhs, typename Rhs,
          enable_if_t<std::is_arithmetic<decay_t<Rhs>>::value, int> = 0>
std::tuple<Lhs...>&
operator*=(std::tuple<Lhs...>&, const Rhs&);

//--------------------------------------------------------------------------------------//
//
//      operator /=
//      operator /
//
//--------------------------------------------------------------------------------------//

template <typename Tp, size_t N>
std::array<Tp, N>&
operator/=(std::array<Tp, N>&, const std::array<Tp, N>&);

template <typename Lhs, size_t N, typename Rhs,
          enable_if_t<std::is_arithmetic<decay_t<Rhs>>::value, int> = 0>
std::array<Lhs, N>&
operator/=(std::array<Lhs, N>&, const Rhs&);

//--------------------------------------------------------------------------------------//

template <typename Lhs, typename Rhs>
std::pair<Lhs, Rhs>&
operator/=(std::pair<Lhs, Rhs>&, const std::pair<Lhs, Rhs>&);

template <typename Lhs, typename Rhs, typename ArithT,
          enable_if_t<std::is_arithmetic<decay_t<ArithT>>::value, int> = 0>
std::pair<Lhs, Rhs>&
operator/=(std::pair<Lhs, Rhs>&, const ArithT&);

//--------------------------------------------------------------------------------------//

template <typename Tp, typename... _Extra>
std::vector<Tp, _Extra...>&
operator/=(std::vector<Tp, _Extra...>&, const std::vector<Tp, _Extra...>&);

template <typename Lhs, typename Rhs, typename... _Extra,
          enable_if_t<std::is_arithmetic<decay_t<Rhs>>::value, int> = 0>
std::vector<Lhs, _Extra...>&
operator/=(std::vector<Lhs, _Extra...>&, const Rhs&);

//--------------------------------------------------------------------------------------//

template <typename... Types>
std::tuple<Types...>&
operator/=(std::tuple<Types...>&, const std::tuple<Types...>&);

template <typename... Lhs, typename Rhs,
          enable_if_t<std::is_arithmetic<decay_t<Rhs>>::value, int> = 0>
std::tuple<Lhs...>&
operator/=(std::tuple<Lhs...>&, const Rhs&);

//--------------------------------------------------------------------------------------//
//
//      operator * (fundamental)
//      operator / (fundamental)
//
//--------------------------------------------------------------------------------------//

template <typename Lhs, typename Rhs,
          enable_if_t<std::is_arithmetic<decay_t<Rhs>>::value, int> = 0>
Lhs operator*(Lhs, const Rhs&);

//--------------------------------------------------------------------------------------//

template <typename Lhs, typename Rhs,
          enable_if_t<std::is_arithmetic<decay_t<Rhs>>::value, int> = 0>
Lhs
operator/(Lhs, const Rhs&);

}  // namespace stl

}  // namespace tim
