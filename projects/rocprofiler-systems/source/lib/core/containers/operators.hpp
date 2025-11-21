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

#include "core/defines.hpp"

#include <iterator>
#include <type_traits>

#define ROCPROFSYS_IMPORT_TEMPLATE2(template_name)
#define ROCPROFSYS_IMPORT_TEMPLATE1(template_name)

// Import a 2-type-argument operator template into boost (if necessary) and
// provide a specialization of 'is_chained_base<>' for it.
#define ROCPROFSYS_OPERATOR_TEMPLATE2(template_name2)                                    \
    ROCPROFSYS_IMPORT_TEMPLATE2(template_name2)                                          \
    template <typename T, typename U, typename B>                                        \
    struct is_chained_base<::rocprofsys::container::template_name2<T, U, B>>             \
    {                                                                                    \
        using value = ::rocprofsys::container::true_t;                                   \
    };

// Import a 1-type-argument operator template into boost (if necessary) and
// provide a specialization of 'is_chained_base<>' for it.
#define ROCPROFSYS_OPERATOR_TEMPLATE1(template_name1)                                    \
    ROCPROFSYS_IMPORT_TEMPLATE1(template_name1)                                          \
    template <typename T, typename B>                                                    \
    struct is_chained_base<::rocprofsys::container::template_name1<T, B>>                \
    {                                                                                    \
        using value = ::rocprofsys::container::true_t;                                   \
    };

#define ROCPROFSYS_OPERATOR_TEMPLATE(template_name)                                      \
    template <typename T, typename U = T, typename B = empty_base<T>,                    \
              typename O = typename is_chained_base<U>::value>                           \
    struct template_name;                                                                \
                                                                                         \
    template <typename T, typename U, typename B>                                        \
        struct template_name<T, U, B, false_t> : template_name##2 < T                    \
    , U                                                                                  \
    , B >                                                                                \
    {};                                                                                  \
                                                                                         \
    template <typename T, typename U>                                                    \
        struct template_name<T, U, empty_base<T>, true_t> : template_name##1 < T         \
    , U >                                                                                \
    {};                                                                                  \
                                                                                         \
    template <typename T, typename B>                                                    \
        struct template_name<T, T, B, false_t> : template_name##1 < T                    \
    , B >                                                                                \
    {};                                                                                  \
                                                                                         \
    template <typename T, typename U, typename B, typename O>                            \
    struct is_chained_base<template_name<T, U, B, O>>                                    \
    {                                                                                    \
        using value = ::rocprofsys::container::true_t;                                   \
    };                                                                                   \
                                                                                         \
    ROCPROFSYS_OPERATOR_TEMPLATE2(template_name##2)                                      \
    ROCPROFSYS_OPERATOR_TEMPLATE1(template_name##1)

#define ROCPROFSYS_BINARY_OPERATOR_COMMUTATIVE(NAME, OP)                                 \
    template <typename T, typename U, typename B = empty_base<T>>                        \
    struct NAME##2                                                                       \
    : B{ friend T operator OP(T lhs, const U& rhs){ return lhs OP## = rhs;               \
    }                                                                                    \
    friend T operator OP(const U& lhs, T rhs) { return rhs OP## = lhs; }                 \
    }                                                                                    \
    ;                                                                                    \
                                                                                         \
    template <typename T, typename B = empty_base<T>>                                    \
    struct NAME##1                                                                       \
    : B{ friend T operator OP(T lhs, const T& rhs){ return lhs OP## = rhs;               \
    }                                                                                    \
    }                                                                                    \
    ;

#define ROCPROFSYS_BINARY_OPERATOR_NON_COMMUTATIVE(NAME, OP)                             \
    template <typename T, typename U, typename B = empty_base<T>>                        \
    struct NAME##2                                                                       \
    : B{ friend T operator OP(T lhs, const U& rhs){ return lhs OP## = rhs;               \
    }                                                                                    \
    }                                                                                    \
    ;

namespace rocprofsys
{
namespace container
{
struct true_t
{};

struct false_t
{};

template <typename T>
class empty_base
{};

template <typename T>
struct is_chained_base
{
    using value = true_t;
};

ROCPROFSYS_BINARY_OPERATOR_COMMUTATIVE(addable, +)
ROCPROFSYS_BINARY_OPERATOR_NON_COMMUTATIVE(subtractable, -)

ROCPROFSYS_OPERATOR_TEMPLATE(addable)

template <typename T, typename B = empty_base<T>>
struct incrementable : B
{
    friend T operator++(T& x, int)
    {
        incrementable_type nrv(x);
        ++x;
        return nrv;
    }

private:  // The use of this typedef works around a Borland bug
    typedef T incrementable_type;
};

template <typename T, typename B = empty_base<T>>
struct decrementable : B
{
    friend T operator--(T& x, int)
    {
        decrementable_type nrv(x);
        --x;
        return nrv;
    }

private:  // The use of this typedef works around a Borland bug
    typedef T decrementable_type;
};

template <typename T, typename P, typename B = empty_base<T>>
struct dereferenceable : B
{
    P operator->() const { return ::std::addressof(*static_cast<const T&>(*this)); }
};

template <typename T, typename I, typename R, typename B = empty_base<T>>
struct indexable : B
{
    R operator[](I n) const { return *(static_cast<const T&>(*this) + n); }
};

template <typename T, typename B = empty_base<T>>
struct equality_comparable1 : B
{
    friend bool operator!=(const T& x, const T& y) { return !static_cast<bool>(x == y); }
};

template <typename T, typename P, typename B = empty_base<T>>
struct input_iteratable
: equality_comparable1<T, incrementable<T, dereferenceable<T, P, B>>>
{};

template <typename T, typename B = empty_base<T>>
struct output_iteratable : incrementable<T, B>
{};

template <typename T, typename P, typename B = empty_base<T>>
struct forward_iteratable : input_iteratable<T, P, B>
{};

template <typename T, typename P, typename B = empty_base<T>>
struct bidirectional_iteratable : forward_iteratable<T, P, decrementable<T, B>>
{};

// template <typename T, typename U, typename B = empty_base<T>>
// struct subtractable2;

template <typename T, typename U, typename B = empty_base<T>>
struct additive2 : addable2<T, U, subtractable2<T, U, B>>
{};

template <typename T, typename B = empty_base<T>>
struct less_than_comparable1 : B
{
    friend bool operator>(const T& x, const T& y) { return y < x; }
    friend bool operator<=(const T& x, const T& y) { return !static_cast<bool>(y < x); }
    friend bool operator>=(const T& x, const T& y) { return !static_cast<bool>(x < y); }
};

//  To avoid repeated derivation from equality_comparable,
//  which is an indirect base typename of bidirectional_iterable,
//  random_access_iteratable must not be derived from totally_ordered1
//  but from less_than_comparable1 only. (Helmut Zeisel, 02-Dec-2001)
template <typename T, typename P, typename D, typename R, typename B = empty_base<T>>
struct random_access_iteratable
: bidirectional_iteratable<
      T, P, less_than_comparable1<T, additive2<T, D, indexable<T, D, R, B>>>>
{};

template <typename CategoryT, typename Tp, typename DistanceT = std::ptrdiff_t,
          typename PointerT = Tp*, typename ReferenceT = Tp&>
struct iterator_helper
{
    using iterator_category = CategoryT;
    using value_type        = Tp;
    using difference_type   = DistanceT;
    using pointer           = PointerT;
    using reference         = ReferenceT;
};

template <typename T, typename V, typename D = std::ptrdiff_t, typename P = V*,
          typename R = V&>
struct random_access_iterator_helper
: random_access_iteratable<T, P, D, R,
                           iterator_helper<std::random_access_iterator_tag, V, D, P, R>>
{
    friend D requires_difference_operator(const T& x, const T& y) { return x - y; }
};  // random_access_iterator_helper
}  // namespace container
}  // namespace rocprofsys
