// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <iterator>

namespace rocprofsys::container
{
// Sentinel base class terminating a chain of mixins below - each mixin derives from
// its `B` template parameter so mixins can be stacked (CRTP-style) without every
// combination needing its own hand-written base case.
template <typename T>
class empty_base
{};

// Synthesizes the heterogeneous, commutative `operator+` (T+U and U+T) from T's
// existing `operator+=(const U&)`.
template <typename T, typename U, typename B = empty_base<T>>
struct addable2 : B
{
    friend constexpr T operator+(T lhs, const U& rhs)
        requires requires(T& val, const U& other) { val += other; }
    {
        return lhs += rhs;
    }
    friend constexpr T operator+(const U& lhs, T rhs)
        requires requires(T& val, const U& other) { val += other; }
    {
        return rhs += lhs;
    }
};

// Synthesizes the heterogeneous `operator-` (T-U only; subtraction is not
// commutative) from T's existing `operator-=(const U&)`.
template <typename T, typename U, typename B = empty_base<T>>
struct subtractable2 : B
{
    friend constexpr T operator-(T lhs, const U& rhs)
        requires requires(T& val, const U& other) { val -= other; }
    {
        return lhs -= rhs;
    }
};

// Synthesizes postfix `operator++(T&, int)` from T's existing prefix `operator++`.
template <typename T, typename B = empty_base<T>>
struct incrementable : B
{
    friend constexpr T operator++(T& self, int)
        requires requires(T& val) { ++val; }
    {
        incrementable_type nrv(self);
        ++self;
        return nrv;
    }

private:  // The use of this alias works around a Borland bug
    using incrementable_type = T;
};

// Synthesizes postfix `operator--(T&, int)` from T's existing prefix `operator--`.
template <typename T, typename B = empty_base<T>>
struct decrementable : B
{
    friend constexpr T operator--(T& self, int)
        requires requires(T& val) { --val; }
    {
        decrementable_type nrv(self);
        --self;
        return nrv;
    }

private:  // The use of this alias works around a Borland bug
    using decrementable_type = T;
};

// Synthesizes `operator->` from T's existing `operator*`.
template <typename T, typename P, typename B = empty_base<T>>
struct dereferenceable : B
{
    constexpr P operator->() const
        requires requires(const T& val) { *val; }
    {
        return ::std::addressof(*static_cast<const T&>(*this));
    }
};

// Synthesizes `operator[]` from T's existing `operator+` and `operator*`.
template <typename T, typename I, typename R, typename B = empty_base<T>>
struct indexable : B
{
    constexpr R operator[](I n) const
        requires requires(const T& val, I idx) { *(val + idx); }
    {
        return *(static_cast<const T&>(*this) + n);
    }
};

// Synthesizes `operator!=` from T's existing `operator==`.
template <typename T, typename B = empty_base<T>>
struct equality_comparable1 : B
{
    friend constexpr bool operator!=(const T& lhs, const T& rhs)
        requires requires(const T& val) { val == val; }
    {
        return !static_cast<bool>(lhs == rhs);
    }
};

// Minimal operator set for an input iterator: equality/inequality, prefix/postfix
// `++`, and `->` - all synthesized from `operator==`, `operator++`, and `operator*`.
template <typename T, typename P, typename B = empty_base<T>>
struct input_iteratable
: equality_comparable1<T, incrementable<T, dereferenceable<T, P, B>>>
{};

// Forward iterators require nothing beyond the input iterator operator set.
template <typename T, typename P, typename B = empty_base<T>>
struct forward_iteratable : input_iteratable<T, P, B>
{};

// Adds prefix/postfix `operator--` to forward_iteratable's operator set.
template <typename T, typename P, typename B = empty_base<T>>
struct bidirectional_iteratable : forward_iteratable<T, P, decrementable<T, B>>
{};

// Full `operator+`/`operator-` support (both operand orders for `+`) synthesized
// from T's `operator+=` and `operator-=`.
template <typename T, typename U, typename B = empty_base<T>>
struct additive2 : addable2<T, U, subtractable2<T, U, B>>
{};

// Synthesizes `operator>`, `operator<=`, `operator>=` from T's existing `operator<`.
//
// NOTE: this deliberately does NOT synthesize `operator<=>` from `<`. Doing so
// would make the synthesized `<=>` an ADL-visible rewrite source for `<` on T (base
// class friends participate in ADL for the derived type), so any `lhs < rhs` used
// to *implement* that `<=>` becomes ambiguous with - and can resolve to - the
// rewritten form derived from itself. Confirmed as a real, compiler-observable bug,
// not just a style concern: GCC 13 happened to pick T's direct `operator<` and
// compiled; Clang picked the self-referential rewritten candidate and blew the
// constexpr recursion limit evaluating `operator<=>(origin, unit)` against itself.
// A type that wants `<=>` should define it directly - the compiler then derives
// `<`, `>`, `<=`, `>=` for free, no mixin needed.
template <typename T, typename B = empty_base<T>>
struct less_than_comparable1 : B
{
    friend constexpr bool operator>(const T& lhs, const T& rhs)
        requires requires(const T& val) { val < val; }
    {
        return rhs < lhs;
    }
    friend constexpr bool operator<=(const T& lhs, const T& rhs)
        requires requires(const T& val) { val < val; }
    {
        return !static_cast<bool>(rhs < lhs);
    }
    friend constexpr bool operator>=(const T& lhs, const T& rhs)
        requires requires(const T& val) { val < val; }
    {
        return !static_cast<bool>(lhs < rhs);
    }
};

//  To avoid repeated derivation from equality_comparable,
//  which is an indirect base typename of bidirectional_iterable,
//  random_access_iteratable must not be derived from totally_ordered1
//  but from less_than_comparable1 only. (Helmut Zeisel, 02-Dec-2001)
//
// Full random-access iterator operator set: bidirectional_iteratable's operators,
// plus ordering (`<`, `>`, `<=`, `>=`) and pointer arithmetic (`+`, `-`, `[]`).
template <typename T, typename P, typename D, typename R, typename B = empty_base<T>>
struct random_access_iteratable
: bidirectional_iteratable<
      T, P, less_than_comparable1<T, additive2<T, D, indexable<T, D, R, B>>>>
{};

// Nested typedefs required by std::iterator_traits.
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

// Convenience mixin combining random_access_iteratable's synthesized operators with
// the std::iterator_traits typedefs and the `operator-` (difference) that a random
// access iterator must provide directly (it cannot be synthesized generically).
template <typename T, typename V, typename D = std::ptrdiff_t, typename P = V*,
          typename R = V&>
struct random_access_iterator_helper
: random_access_iteratable<T, P, D, R,
                           iterator_helper<std::random_access_iterator_tag, V, D, P, R>>
{
    friend constexpr D requires_difference_operator(const T& lhs, const T& rhs)
    {
        return lhs - rhs;
    }
};  // random_access_iterator_helper
}  // namespace rocprofsys::container
