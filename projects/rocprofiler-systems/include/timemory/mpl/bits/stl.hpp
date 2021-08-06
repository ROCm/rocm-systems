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

#pragma once

#include "timemory/mpl/bits/types.hpp"
#include "timemory/mpl/math.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/utility/types.hpp"

#include <array>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

//======================================================================================//

#if defined(TIMEMORY_WINDOWS)

namespace std
{
template <typename Lhs, typename Rhs>
const pair<Lhs, Rhs>
operator-(pair<Lhs, Rhs>, const pair<Lhs, Rhs>&);

template <typename... Types>
const tuple<Types...>
operator-(tuple<Types...>, const tuple<Types...>&);

}  // namespace std

#endif

//======================================================================================//

namespace tim
{
/// \namespace tim::stl
/// \brief the namespace is provided to hide stl overload from global namespace but
/// provide a method of using the namespace without a "using namespace tim;"
namespace stl
{
namespace ostream
{
/// \namespace tim::stl::ostream
/// \brief the namespace provides overloads to output complex data types w/ streams
//--------------------------------------------------------------------------------------//
//
//      operator <<
//
//--------------------------------------------------------------------------------------//

template <typename T, typename U>
std::ostream&
operator<<(std::ostream& os, const std::pair<T, U>& p)
{
    os << "(" << p.first << "," << p.second << ")";
    return os;
}

//--------------------------------------------------------------------------------------//
//
template <typename... Types>
std::ostream&
operator<<(std::ostream& os, const std::tuple<Types...>& p)
{
    constexpr size_t N = sizeof...(Types);
    tuple_printer(p, os, make_index_sequence<N>{});
    return os;
}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename... ExtraT>
std::ostream&
operator<<(std::ostream& os, const std::vector<Tp, ExtraT...>& p)
{
    os << "(";
    for(size_t i = 0; i < p.size(); ++i)
        os << p.at(i) << ((i + 1 < p.size()) ? "," : "");
    os << ")";
    return os;
}

//--------------------------------------------------------------------------------------//

template <typename Tp, size_t N>
std::ostream&
operator<<(std::ostream& os, const std::array<Tp, N>& p)
{
    os << "(";
    for(size_t i = 0; i < p.size(); ++i)
        os << p.at(i) << ((i + 1 < p.size()) ? "," : "");
    os << ")";
    return os;
}

}  // namespace ostream

//--------------------------------------------------------------------------------------//
//
//      operator += (same type)
//
//--------------------------------------------------------------------------------------//

template <typename Tp, size_t N, typename OtherT>
std::array<Tp, N>&
operator+=(std::array<Tp, N>& lhs, OtherT&& rhs)
{
    math::plus(lhs, std::forward<OtherT>(rhs));
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename Lhs, typename Rhs, typename OtherT>
std::pair<Lhs, Rhs>&
operator+=(std::pair<Lhs, Rhs>& lhs, OtherT&& rhs)
{
    math::plus(lhs, std::forward<OtherT>(rhs));
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename... ExtraT, typename OtherT>
std::vector<Tp, ExtraT...>&
operator+=(std::vector<Tp, ExtraT...>& lhs, OtherT&& rhs)
{
    math::plus(lhs, std::forward<OtherT>(rhs));
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename... Types, typename OtherT>
std::tuple<Types...>&
operator+=(std::tuple<Types...>& lhs, OtherT&& rhs)
{
    math::plus(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//
//
//      operator -= (same type)
//
//--------------------------------------------------------------------------------------//

template <typename Tp, size_t N>
std::array<Tp, N>&
operator-=(std::array<Tp, N>& lhs, const std::array<Tp, N>& rhs)
{
    math::minus(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename Lhs, typename Rhs>
std::pair<Lhs, Rhs>&
operator-=(std::pair<Lhs, Rhs>& lhs, const std::pair<Lhs, Rhs>& rhs)
{
    math::minus(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename... ExtraT>
std::vector<Tp, ExtraT...>&
operator-=(std::vector<Tp, ExtraT...>& lhs, const std::vector<Tp, ExtraT...>& rhs)
{
    math::minus(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename... Types>
std::tuple<Types...>&
operator-=(std::tuple<Types...>& lhs, const std::tuple<Types...>& rhs)
{
    math::minus(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//
//
//      operator *= (same type)
//
//--------------------------------------------------------------------------------------//

template <typename Tp, size_t N>
std::array<Tp, N>&
operator*=(std::array<Tp, N>& lhs, const std::array<Tp, N>& rhs)
{
    math::multiply(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename Lhs, typename Rhs>
std::pair<Lhs, Rhs>&
operator*=(std::pair<Lhs, Rhs>& lhs, const std::pair<Lhs, Rhs>& rhs)
{
    math::multiply(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename... ExtraT>
std::vector<Tp, ExtraT...>&
operator*=(std::vector<Tp, ExtraT...>& lhs, const std::vector<Tp, ExtraT...>& rhs)
{
    math::multiply(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename... Types>
std::tuple<Types...>&
operator*=(std::tuple<Types...>& lhs, const std::tuple<Types...>& rhs)
{
    math::multiply(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//
//
//      operator /= (same type)
//
//--------------------------------------------------------------------------------------//

template <typename Tp, size_t N>
std::array<Tp, N>&
operator/=(std::array<Tp, N>& lhs, const std::array<Tp, N>& rhs)
{
    math::divide(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename Lhs, typename Rhs>
std::pair<Lhs, Rhs>&
operator/=(std::pair<Lhs, Rhs>& lhs, const std::pair<Lhs, Rhs>& rhs)
{
    math::divide(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename Tp, typename... ExtraT>
std::vector<Tp, ExtraT...>&
operator/=(std::vector<Tp, ExtraT...>& lhs, const std::vector<Tp, ExtraT...>& rhs)
{
    math::divide(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename... Types>
std::tuple<Types...>&
operator/=(std::tuple<Types...>& lhs, const std::tuple<Types...>& rhs)
{
    math::divide(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//
//
//      operator *= (fundamental)
//
//--------------------------------------------------------------------------------------//

template <typename Lhs, size_t N, typename Rhs,
          enable_if_t<std::is_arithmetic<decay_t<Rhs>>::value, int>>
std::array<Lhs, N>&
operator*=(std::array<Lhs, N>& lhs, const Rhs& rhs)
{
    math::multiply(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename Lhs, typename Rhs, typename ArithT,
          enable_if_t<std::is_arithmetic<decay_t<ArithT>>::value, int>>
std::pair<Lhs, Rhs>&
operator*=(std::pair<Lhs, Rhs>& lhs, const ArithT& rhs)
{
    math::multiply(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename Lhs, typename Rhs, typename... ExtraT,
          enable_if_t<std::is_arithmetic<decay_t<Rhs>>::value, int>>
std::vector<Lhs, ExtraT...>&
operator*=(std::vector<Lhs, ExtraT...>& lhs, const Rhs& rhs)
{
    math::multiply(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename... Lhs, typename Rhs,
          enable_if_t<std::is_arithmetic<decay_t<Rhs>>::value, int>>
std::tuple<Lhs...>&
operator*=(std::tuple<Lhs...>& lhs, const Rhs& rhs)
{
    math::multiply(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//
//
//      operator /= (fundamental)
//
//--------------------------------------------------------------------------------------//

template <typename Lhs, size_t N, typename Rhs,
          enable_if_t<std::is_arithmetic<decay_t<Rhs>>::value, int>>
std::array<Lhs, N>&
operator/=(std::array<Lhs, N>& lhs, const Rhs& rhs)
{
    math::divide(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename Lhs, typename Rhs, typename ArithT,
          enable_if_t<std::is_arithmetic<decay_t<ArithT>>::value, int>>
std::pair<Lhs, Rhs>&
operator/=(std::pair<Lhs, Rhs>& lhs, const ArithT& rhs)
{
    math::divide(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename Lhs, typename Rhs, typename... ExtraT,
          enable_if_t<std::is_arithmetic<decay_t<Rhs>>::value, int>>
std::vector<Lhs, ExtraT...>&
operator/=(std::vector<Lhs, ExtraT...>& lhs, const Rhs& rhs)
{
    math::divide(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//

template <typename... Lhs, typename Rhs,
          enable_if_t<std::is_arithmetic<decay_t<Rhs>>::value, int>>
std::tuple<Lhs...>&
operator/=(std::tuple<Lhs...>& lhs, const Rhs& rhs)
{
    math::divide(lhs, rhs);
    return lhs;
}

//--------------------------------------------------------------------------------------//
//
//      operator * (fundamental)
//      operator / (fundamental)
//
//--------------------------------------------------------------------------------------//

template <typename Lhs, typename Rhs,
          enable_if_t<std::is_arithmetic<decay_t<Rhs>>::value, int>>
Lhs operator*(Lhs lhs, const Rhs& rhs)
{
    return (lhs *= rhs);
}

//--------------------------------------------------------------------------------------//

template <typename Lhs, typename Rhs,
          enable_if_t<std::is_arithmetic<decay_t<Rhs>>::value, int>>
Lhs
operator/(Lhs lhs, const Rhs& rhs)
{
    return (lhs /= rhs);
}

//--------------------------------------------------------------------------------------//

}  // namespace stl

using namespace stl;

//======================================================================================//

}  // namespace tim

//======================================================================================//

namespace std
{
#if defined(TIMEMORY_WINDOWS)

template <typename Lhs, typename Rhs>
const pair<Lhs, Rhs>
operator-(pair<Lhs, Rhs> lhs, const pair<Lhs, Rhs>& rhs)
{
    ::tim::math::minus(lhs, rhs);
    return lhs;
}

template <typename... Types>
const tuple<Types...>
operator-(tuple<Types...> lhs, const tuple<Types...>& rhs)
{
    ::tim::math::minus(lhs, rhs);
    return lhs;
}

#endif

template <typename Tp>
tuple<>&
operator+=(tuple<>& _lhs, const Tp&)
{
    return _lhs;
}

}  // namespace std
