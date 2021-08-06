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

/** \file mpl/stl.hpp
 * \headerfile mpl/stl.hpp "timemory/mpl/stl.hpp"
 * Provides operators on common STL structures such as <<, +=, -=, *=, /=, +, -, *, /
 *
 */

#pragma once

#include <array>
#include <ostream>
#include <tuple>
#include <utility>
#include <vector>

#include "timemory/mpl/math.hpp"
#include "timemory/utility/macros.hpp"

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

template <typename Tp, typename... Extra>
std::ostream&
operator<<(std::ostream&, const std::vector<Tp, Extra...>&);

//--------------------------------------------------------------------------------------//

template <typename Tp, size_t N>
std::ostream&
operator<<(std::ostream&, const std::array<Tp, N>&);

//--------------------------------------------------------------------------------------//

}  // namespace ostream

//--------------------------------------------------------------------------------------//
//
//      operator +=
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

template <typename Tp, typename... Extra, typename Other>
std::vector<Tp, Extra...>&
operator+=(std::vector<Tp, Extra...>&, Other&&);

//--------------------------------------------------------------------------------------//

template <typename... Types, typename Other>
std::tuple<Types...>&
operator+=(std::tuple<Types...>&, Other&&);

//--------------------------------------------------------------------------------------//
//
//      operator -=
//
//--------------------------------------------------------------------------------------//

template <typename Tp, size_t N>
std::array<Tp, N>&
operator-=(std::array<Tp, N>&, const std::array<Tp, N>&);

//--------------------------------------------------------------------------------------//

template <typename... Types>
std::tuple<Types...>&
operator-=(std::tuple<Types...>&, const std::tuple<Types...>&);

//--------------------------------------------------------------------------------------//

template <typename Lhs, typename Rhs>
std::pair<Lhs, Rhs>&
operator-=(std::pair<Lhs, Rhs>&, const std::pair<Lhs, Rhs>&);

//--------------------------------------------------------------------------------------//

template <typename Tp, typename... Extra>
std::vector<Tp, Extra...>&
operator-=(std::vector<Tp, Extra...>&, const std::vector<Tp, Extra...>&);

//--------------------------------------------------------------------------------------//
//
//      operator *=
//
//--------------------------------------------------------------------------------------//

template <typename Tp, size_t N>
std::array<Tp, N>&
operator*=(std::array<Tp, N>&, const std::array<Tp, N>&);

//--------------------------------------------------------------------------------------//

template <typename Lhs, typename Rhs>
std::pair<Lhs, Rhs>&
operator*=(std::pair<Lhs, Rhs>&, const std::pair<Lhs, Rhs>&);

//--------------------------------------------------------------------------------------//

template <typename... Types>
std::tuple<Types...>&
operator*=(std::tuple<Types...>&, const std::tuple<Types...>&);

//--------------------------------------------------------------------------------------//

template <typename Tp, typename... Extra>
std::vector<Tp, Extra...>&
operator*=(std::vector<Tp, Extra...>&, const std::vector<Tp, Extra...>&);

//--------------------------------------------------------------------------------------//
//
//      operator /=
//
//--------------------------------------------------------------------------------------//

template <typename Tp, size_t N>
std::array<Tp, N>&
operator/=(std::array<Tp, N>&, const std::array<Tp, N>&);

//--------------------------------------------------------------------------------------//

template <typename Lhs, typename Rhs>
std::pair<Lhs, Rhs>&
operator/=(std::pair<Lhs, Rhs>&, const std::pair<Lhs, Rhs>&);

//--------------------------------------------------------------------------------------//

template <typename... Types>
std::tuple<Types...>&
operator/=(std::tuple<Types...>&, const std::tuple<Types...>&);

//--------------------------------------------------------------------------------------//

template <typename Tp, typename... Extra>
std::vector<Tp, Extra...>&
operator/=(std::vector<Tp, Extra...>&, const std::vector<Tp, Extra...>&);

//--------------------------------------------------------------------------------------//
//
//      operator +
//
//--------------------------------------------------------------------------------------//

template <typename Tp, size_t N>
std::array<Tp, N>
operator+(std::array<Tp, N> lhs, const std::array<Tp, N>& rhs)
{
    math::plus(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename... Types>
std::tuple<Types...>
operator+(std::tuple<Types...> lhs, const std::tuple<Types...>& rhs)
{
    math::plus(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename Lhs, typename Rhs>
std::pair<Lhs, Rhs>
operator+(std::pair<Lhs, Rhs> lhs, const std::pair<Lhs, Rhs>& rhs)
{
    math::plus(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename... Extra>
std::vector<Tp, Extra...>
operator+(std::vector<Tp, Extra...> lhs, const std::vector<Tp, Extra...>& rhs)
{
    math::plus(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//
//
//      operator -
//
//--------------------------------------------------------------------------------------//

template <typename Tp, size_t N>
std::array<Tp, N>
operator-(std::array<Tp, N> lhs, const std::array<Tp, N>& rhs)
{
    math::minus(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename... Types>
std::tuple<Types...>
operator-(std::tuple<Types...> lhs, const std::tuple<Types...>& rhs)
{
    math::minus(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename Lhs, typename Rhs>
std::pair<Lhs, Rhs>
operator-(std::pair<Lhs, Rhs> lhs, const std::pair<Lhs, Rhs>& rhs)
{
    math::minus(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename... Extra>
std::vector<Tp, Extra...>
operator-(std::vector<Tp, Extra...> lhs, const std::vector<Tp, Extra...>& rhs)
{
    math::minus(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//
//
//      operator *
//
//--------------------------------------------------------------------------------------//

template <typename Tp, size_t N>
std::array<Tp, N> operator*(std::array<Tp, N> lhs, const std::array<Tp, N>& rhs)
{
    math::multiply(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename... Types>
std::tuple<Types...> operator*(std::tuple<Types...> lhs, const std::tuple<Types...>& rhs)
{
    math::multiply(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename Lhs, typename Rhs>
std::pair<Lhs, Rhs> operator*(std::pair<Lhs, Rhs> lhs, const std::pair<Lhs, Rhs>& rhs)
{
    math::multiply(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename... Extra>
std::vector<Tp, Extra...> operator*(std::vector<Tp, Extra...>        lhs,
                                    const std::vector<Tp, Extra...>& rhs)
{
    math::multiply(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//
//
//      operator /
//
//--------------------------------------------------------------------------------------//

template <typename Tp, size_t N>
std::array<Tp, N>
operator/(std::array<Tp, N> lhs, const std::array<Tp, N>& rhs)
{
    math::divide(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename... Types>
std::tuple<Types...>
operator/(std::tuple<Types...> lhs, const std::tuple<Types...>& rhs)
{
    math::divide(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename Lhs, typename Rhs>
std::pair<Lhs, Rhs>
operator/(std::pair<Lhs, Rhs> lhs, const std::pair<Lhs, Rhs>& rhs)
{
    math::divide(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename... Extra>
std::vector<Tp, Extra...>
operator/(std::vector<Tp, Extra...> lhs, const std::vector<Tp, Extra...>& rhs)
{
    math::divide(lhs, rhs);
    return lhs;
}

}  // namespace stl

using namespace stl;

}  // namespace tim

#include "timemory/mpl/bits/stl.hpp"
